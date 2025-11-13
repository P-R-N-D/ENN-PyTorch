# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import importlib
import logging
import math
from contextlib import AbstractContextManager
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import CudaGraphModule, TensorDictModule, TensorDictSequential

from ..backend.compat import patch_torch
from ..backend.system import (
    get_device,
    is_cpu_bf16_supported,
    is_cuda_bf16_supported,
    is_float8_supported,
    is_int8_supported,
)
from ..data.stats import Metadata
if TYPE_CHECKING:
    from ..model.kernels import DotProductAttention as _DotProductAttention

patch_torch()


LossCallable = Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
_LOGGER = logging.getLogger(__name__)


def _is_ptq_unavailable(
    model: nn.Module, *args: Any, **kwargs: Any
) -> tuple[nn.Module, bool, str]:
    return (model, False, "PTQ backend unavailable")


class _QATUnavailable:
    @staticmethod
    def initialize(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("QAT backend unavailable")


def _is_compiled_for_inference(model: torch.nn.Module) -> bool:
    compile_attrs = (
        "_is_compiled_for_inference",
        "__is_compiled_for_inference__",
        "__compiled_for_serving__",
        "__serving_compiled__",
        "_is_serialized_for_serving",
    )
    if any(bool(getattr(model, attr, False)) for attr in compile_attrs):
        return True

    jit = getattr(torch, "jit", None)
    script_like_types: List[type] = []
    if jit is not None:
        for name in ("ScriptModule", "RecursiveScriptModule", "TopLevelTracedModule"):
            typ = getattr(jit, name, None)
            if isinstance(typ, type):
                script_like_types.append(typ)
        for mod_name in ("_script", "_trace"):
            submod = getattr(jit, mod_name, None)
            if submod is None:
                continue
            for name in ("RecursiveScriptModule", "TopLevelTracedModule"):
                typ = getattr(submod, name, None)
                if isinstance(typ, type):
                    script_like_types.append(typ)

    if any(isinstance(model, typ) for typ in script_like_types):
        return True

    try:
        modules = tuple(model.modules())
    except Exception:
        modules = ()

    for module in modules:
        if module is model:
            continue
        if any(bool(getattr(module, attr, False)) for attr in compile_attrs):
            return True
        if any(isinstance(module, typ) for typ in script_like_types):
            return True
    return False


def _is_aot_autograd_enabled(model: torch.nn.Module) -> bool:
    indicator_attrs = (
        "_aot_autograd_graph",
        "_aot_autograd_cache",
        "_aot_compiled_autograd",
        "_aot_autograd_traced_module",
        "__aot_autograd__",
        "__compiled_with_aot_autograd__",
    )
    if any(getattr(model, attr, None) for attr in indicator_attrs):
        return True

    try:
        modules = tuple(model.modules())
    except Exception:
        modules = ()

    for module in modules:
        if module is model:
            continue
        if any(getattr(module, attr, None) for attr in indicator_attrs):
            return True
        class_name = module.__class__.__name__
        module_name = getattr(module.__class__, "__module__", "")
        if "AOTAutograd" in class_name or "aot_autograd" in module_name:
            return True
    return False


def reshape_for_mha(
    tensor: torch.Tensor, batch_size: int, head_count: int, head_dim: int
) -> torch.Tensor:

    return tensor.view(batch_size, -1, head_count, head_dim).transpose(1, 2)


def is_nvidia_te_available(model: torch.nn.Module) -> bool:
    te_flags = (
        getattr(model, "__fp8_inference_te__", False),
        getattr(model, "__fp8_training_te__", False),
        getattr(model, "__te_fp8_default__", False),
    )
    if any(te_flags):
        return True

    for module in model.modules():
        mod_name = getattr(module.__class__, "__module__", "")
        if isinstance(mod_name, str) and mod_name.startswith("transformer_engine"):
            return True
    return False


def is_scale_safe(
    dtype: torch.dtype,
    meta: Optional[Metadata[Any]],
    *args: Any,
    safety_margin: float = 8.0,
    **kwargs: Any,
) -> bool:
    if meta is None or not getattr(meta, "has_scale", False):
        return True
    if not isinstance(dtype, torch.dtype):
        return False
    max_abs = getattr(meta, "scale_max_abs", None)
    if max_abs is None:
        return True
    max_abs = float(abs(max_abs))
    if not math.isfinite(max_abs):
        return False
    if getattr(dtype, "is_complex", False):
        base_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        return is_scale_safe(base_dtype, meta, safety_margin=safety_margin)
    if getattr(dtype, "is_floating_point", False):
        info = torch.finfo(dtype)
        if max_abs > float(info.max) / safety_margin:
            return False
        min_pos = getattr(meta, "scale_min_positive", None)
        if min_pos is not None and min_pos < float(info.tiny) * safety_margin:
            return False
        return True
    if dtype == torch.bool:
        is_integral = getattr(meta, "scale_is_integral", None)
        return (is_integral is None or is_integral) and max_abs <= 1.0
    try:
        info = torch.iinfo(dtype)
    except TypeError:
        return False
    is_integral = getattr(meta, "scale_is_integral", None)
    if is_integral is False:
        return False
    return max_abs <= float(info.max)


def _is_for_cuda(module: nn.Module) -> bool:
    try:
        for tensor in module.parameters():
            if getattr(tensor, "device", None) is None:
                continue
            if tensor.device.type == "cuda":
                return True
    except Exception:
        pass
    try:
        for tensor in module.buffers():
            if getattr(tensor, "device", None) is None:
                continue
            if tensor.device.type == "cuda":
                return True
    except Exception:
        pass
    return False


class Gradient:
    @staticmethod
    def inference(model: torch.nn.Module) -> AbstractContextManager[None]:
        if (
            is_nvidia_te_available(model)
            or _is_compiled_for_inference(model)
            or _is_aot_autograd_enabled(model)
        ):
            return torch.no_grad()
        return torch.inference_mode()

    @staticmethod
    def compile(
        module: nn.Module,
        *args: Any,
        backend: Optional[str] = None,
        mode: Optional[str] = None,
        fullgraph: Optional[bool] = None,
        dynamic: Optional[bool] = None,
        options: Optional[Dict[str, Any]] = None,
        disable: bool = False,
        **kwargs: Any
    ) -> nn.Module:
        normalized_mode = ""
        if mode is not None:
            normalized_mode = str(mode).strip().lower()
        if disable or normalized_mode in {"", "disabled", "none"}:
            return module
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            return module
        if normalized_mode == "max-autotune" and not _is_for_cuda(module):
            mode = "max-autotune-no-cudagraphs"
            normalized_mode = "max-autotune-no-cudagraphs"
        kwargs: Dict[str, Any] = {}
        if backend is not None:
            kwargs["backend"] = backend
        if mode is not None:
            kwargs["mode"] = mode
        if fullgraph is not None:
            kwargs["fullgraph"] = bool(fullgraph)
        if dynamic is not None:
            kwargs["dynamic"] = bool(dynamic)
        if options:
            kwargs["options"] = dict(options)
        return compile_fn(module, **kwargs)


class Autocast:
    _preferred_fp8_backend: Optional[str] = None
    _preferred_int_backend: Optional[str] = None
    _last_float_dtype: torch.dtype = torch.float32
    _last_int_dtype: torch.dtype = torch.int64
    _metadata: Optional[Metadata[Any]] = None

    @staticmethod
    def _device(
        device: Optional[Union[torch.device, str]] = None,
    ) -> torch.device:
        if device is None:
            return get_device()
        if isinstance(device, torch.device):
            return device
        return torch.device(device)

    @classmethod
    def _fp8_backend(
        cls: object,
        preferred: Optional[str],
        *args: Any,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        dev = device if device is not None else cls._device(None)
        order: Tuple[str, ...]
        if preferred == "te":
            order = ("te", "ao")
        elif preferred == "ao":
            order = ("ao", "te")
        else:
            order = ("te", "ao")
        for backend in order:
            if backend == "te":
                ok, reason = is_float8_supported(dev)
                if not ok:
                    _LOGGER.debug("Autocast FP8 TE unavailable: %s", reason)
                    continue
                try:
                    te = importlib.import_module("transformer_engine.pytorch")

                    if getattr(te, "fp8_autocast", None) is None:
                        raise AttributeError(
                            "transformer_engine.fp8_autocast missing"
                        )
                except Exception as exc:
                    _LOGGER.debug("Autocast FP8 TE import failed: %s", exc)
                    continue
                cls._preferred_fp8_backend = "te"
                return "te"
            if backend == "ao":
                try:
                    _float8_mod = importlib.import_module("torchao.float8")

                    if getattr(_float8_mod, "fp8_autocast", None) is None:
                        raise AttributeError("torchao.float8.fp8_autocast missing")
                except Exception as exc:
                    _LOGGER.debug("Autocast FP8 torchao import failed: %s", exc)
                    continue
                cls._preferred_fp8_backend = "ao"
                return "ao"
        cls._preferred_fp8_backend = None
        return None

    @classmethod
    def _int_backend(
        cls: object,
        preferred: Optional[str],
        *args: Any,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        dev = device if device is not None else cls._device(None)
        order: Tuple[str, ...]
        if preferred == "te":
            order = ("te", "ao")
        elif preferred == "ao":
            order = ("ao", "te")
        else:
            order = ("te", "ao")
        for backend in order:
            if backend == "te":
                ok, reason = is_int8_supported(dev)
                if not ok:
                    _LOGGER.debug("Autocast INT8 TE unavailable: %s", reason)
                    continue
                try:
                    te = importlib.import_module("transformer_engine.pytorch")

                    if getattr(te, "int8_autocast", None) is None:
                        raise AttributeError(
                            "transformer_engine.int8_autocast missing"
                        )
                except Exception as exc:
                    _LOGGER.debug("Autocast INT8 TE import failed: %s", exc)
                    continue
                cls._preferred_int_backend = "te"
                return "te"
            if backend == "ao":
                try:
                    quant_mod = importlib.import_module("torchao.quantization")
                    int8_autocast = getattr(quant_mod, "int8_autocast", None)

                    if not callable(int8_autocast):
                        raise AttributeError("torchao.quantization.int8_autocast missing")
                except Exception as exc:
                    _LOGGER.debug("Autocast INT8 torchao import failed: %s", exc)
                    continue
                cls._preferred_int_backend = "ao"
                return "ao"
        cls._preferred_int_backend = None
        return None

    @classmethod
    def _nvidia_float8(
        cls: object, device: torch.device, enabled: bool
    ) -> List[contextlib.AbstractContextManager[None]]:
        contexts: List[contextlib.AbstractContextManager[None]] = []
        if not enabled:
            return contexts
        try:
            te = importlib.import_module("transformer_engine.pytorch")

            fp8_ctx = getattr(te, "fp8_autocast", None)
            if callable(fp8_ctx):
                contexts.append(fp8_ctx(enabled=True))
            else:
                raise AttributeError("transformer_engine.fp8_autocast missing")
        except Exception as exc:
            _LOGGER.debug("Autocast FP8 TE failed: %s", exc)
            cls._preferred_fp8_backend = None
        return contexts

    @classmethod
    def _torchao_float8(
        cls: object, enabled: bool
    ) -> List[contextlib.AbstractContextManager[None]]:
        contexts: List[contextlib.AbstractContextManager[None]] = []
        if not enabled:
            return contexts
        try:
            fp8_mod = importlib.import_module("torchao.float8")
            fp8_autocast = getattr(fp8_mod, "fp8_autocast", None)

            if callable(fp8_autocast):
                contexts.append(fp8_autocast(enabled=True))
            else:
                raise AttributeError("torchao.float8.fp8_autocast missing")
        except Exception as exc:
            _LOGGER.debug("Autocast FP8 torchao failed: %s", exc)
            cls._preferred_fp8_backend = None
        return contexts

    @classmethod
    def _torchao_int8(
        cls: object,
        device: torch.device,
        enabled: bool,
    ) -> List[contextlib.AbstractContextManager[None]]:
        contexts: List[contextlib.AbstractContextManager[None]] = []
        if not enabled:
            return contexts
        backend = cls._preferred_int_backend
        if backend == "te":
            try:
                te = importlib.import_module("transformer_engine.pytorch")

                int_ctx = getattr(te, "int8_autocast", None)
                if callable(int_ctx):
                    contexts.append(int_ctx(enabled=True))
                else:
                    raise AttributeError("transformer_engine.int8_autocast missing")
            except Exception as exc:
                _LOGGER.debug("Autocast INT8 TE failed: %s", exc)
                cls._preferred_int_backend = None
        elif backend == "ao":
            try:
                quant_mod = importlib.import_module("torchao.quantization")
                int8_autocast = getattr(quant_mod, "int8_autocast", None)

                if callable(int8_autocast):
                    contexts.append(int8_autocast(enabled=True))
                else:
                    raise AttributeError("torchao.quantization.int8_autocast missing")
            except Exception as exc:
                _LOGGER.debug("Autocast INT8 torchao failed: %s", exc)
                cls._preferred_int_backend = None
        return contexts

    @classmethod
    def coerce_metadata(
        cls: object,
        device: Optional[Union[torch.device, str]] = None,
        *args: Any,
        metadata: Optional[Metadata[Any]] = None,
        **kwargs: Any,
    ) -> Metadata[Any]:
        meta = metadata or cls._metadata
        device_hint: Optional[Union[torch.device, str]] = device
        if device_hint is None and meta is not None:
            with contextlib.suppress(Exception):
                device_hint = torch.device(meta.device)
        dev = cls._device(device_hint)
        if meta is None:
            meta = Metadata.for_device(dev)
        else:
            current_device = torch.device(getattr(meta, "device", dev))
            if current_device != dev:
                meta.device = dev
                meta.refresh()
            else:
                meta.ensure_device_info()
        if not getattr(meta, "float_dtypes", ()):
            meta.refresh()
        elif (
            not getattr(meta, "int_dtypes", ())
            or not getattr(meta, "float8_dtypes", ())
        ):
            meta.refresh()
        else:
            meta.ensure_device_info()
        cls._metadata = meta
        return meta

    @classmethod
    def float_amp_priority(cls: object, device: torch.device) -> Tuple[torch.dtype, ...]:
        meta = cls.coerce_metadata(device)
        candidates = getattr(meta, "float_dtypes", ())
        if candidates:
            return tuple(candidates)
        return (torch.float32,)

    @staticmethod
    def float8_formats() -> Tuple[torch.dtype, ...]:
        meta = Autocast._metadata
        if meta is not None and getattr(meta, "float8_dtypes", None):
            return tuple(meta.float8_dtypes)
        names = (
            "float8_e4m3fn",
            "float8_e4m3fnuz",
            "float8_e5m2",
            "float8_e5m2fnuz",
        )
        values: list[torch.dtype] = []
        for name in names:
            candidate = getattr(torch, name, None)
            if isinstance(candidate, torch.dtype):
                values.append(candidate)
        values = tuple(values)
        if meta is not None:
            meta.float8_dtypes = values
        return values

    @classmethod
    def integer_amp_priority(cls: object, device: torch.device) -> Tuple[torch.dtype, ...]:
        meta = cls.coerce_metadata(device)
        candidates = getattr(meta, "int_dtypes", ())
        if candidates:
            return tuple(candidates)
        return (torch.int64,)

    @classmethod
    def negotiate(
        cls: object,
        candidates: Tuple[torch.dtype, ...],
        *args: Any,
        fallback: torch.dtype,
        logger: Optional[logging.Logger] = None,
        context: str = "autocast",
        device: Optional[torch.device] = None,
        meta: Optional[Metadata[Any]] = None,
        **kwargs: Any,
    ) -> torch.dtype:
        for dtype in candidates:
            if is_scale_safe(dtype, meta):
                return dtype
        device_str = f" on {device.type}" if device is not None else ""
        fallback_order: Tuple[torch.dtype, ...]
        if getattr(fallback, "is_floating_point", False):
            fallback_order = (fallback, torch.float32, torch.float64)
        else:
            fallback_order = (fallback, torch.int64, torch.float32, torch.float64)
        for dtype in fallback_order:
            if is_scale_safe(dtype, meta):
                if logger is not None and dtype is not fallback:
                    logger.debug(
                        "Autocast %s fallback%s: promoting to %s due to data scale",
                        context,
                        device_str,
                        str(dtype).split(".")[-1],
                    )
                return dtype
        if logger is not None:
            logger.debug(
                "Autocast %s fallback%s: using %s without scale guarantee",
                context,
                device_str,
                str(fallback).split(".")[-1],
            )
        return fallback

    @classmethod
    def _nvidia_float8(
        cls: object, device: torch.device, enabled: bool
    ) -> List[contextlib.AbstractContextManager[None]]:
        contexts: List[contextlib.AbstractContextManager[None]] = []
        if not enabled:
            return contexts
        try:
            te = importlib.import_module("transformer_engine.pytorch")

            fp8_ctx = getattr(te, "fp8_autocast", None)
            if callable(fp8_ctx):
                contexts.append(fp8_ctx(enabled=True))
            else:
                raise AttributeError("transformer_engine.fp8_autocast missing")
        except Exception as exc:
            _LOGGER.debug("Autocast FP8 TE failed: %s", exc)
            cls._preferred_fp8_backend = None
        return contexts

    @classmethod
    def _torchao_float8(
        cls: object, enabled: bool
    ) -> List[contextlib.AbstractContextManager[None]]:
        contexts: List[contextlib.AbstractContextManager[None]] = []
        if not enabled:
            return contexts
        try:
            fp8_mod = importlib.import_module("torchao.float8")
            fp8_autocast = getattr(fp8_mod, "fp8_autocast", None)

            if callable(fp8_autocast):
                contexts.append(fp8_autocast(enabled=True))
            else:
                raise AttributeError("torchao.float8.fp8_autocast missing")
        except Exception as exc:
            _LOGGER.debug("Autocast FP8 torchao failed: %s", exc)
            cls._preferred_fp8_backend = None
        return contexts

    @classmethod
    def _torchao_int8(
        cls: object,
        device: torch.device,
        enabled: bool,
    ) -> List[contextlib.AbstractContextManager[None]]:
        contexts: List[contextlib.AbstractContextManager[None]] = []
        if not enabled:
            return contexts
        backend = cls._preferred_int_backend
        if backend == "te":
            try:
                te = importlib.import_module("transformer_engine.pytorch")

                int_ctx = getattr(te, "int8_autocast", None)
                if callable(int_ctx):
                    contexts.append(int_ctx(enabled=True))
                else:
                    raise AttributeError("transformer_engine.int8_autocast missing")
            except Exception as exc:
                _LOGGER.debug("Autocast INT8 TE failed: %s", exc)
                cls._preferred_int_backend = None
        elif backend == "ao":
            try:
                quant_mod = importlib.import_module("torchao.quantization")
                int8_autocast = getattr(quant_mod, "int8_autocast", None)

                if callable(int8_autocast):
                    contexts.append(int8_autocast(enabled=True))
                else:
                    raise AttributeError("torchao.quantization.int8_autocast missing")
            except Exception as exc:
                _LOGGER.debug("Autocast INT8 torchao failed: %s", exc)
                cls._preferred_int_backend = None
        return contexts

    @classmethod
    def configure(
        cls: object,
        model: Optional[nn.Module],
        *args: Any,
        metadata: Optional[Metadata[Any]] = None,
        **kwargs: Any,
    ) -> None:
        backend: Optional[str] = None
        int_backend: Optional[str] = None
        if isinstance(model, nn.Module):
            if any(
                getattr(model, attr, False)
                for attr in ("__fp8_inference_te__", "__fp8_training_te__")
            ):
                backend = "te"
            elif any(
                getattr(model, attr, False)
                for attr in ("__fp8_inference_ao__", "__fp8_training_ao__")
            ):
                backend = "ao"
            if any(
                getattr(model, attr, False)
                for attr in (
                    "__int8_training_te__",
                    "__int8_inference_te__",
                    "__te_int8_default__",
                )
            ):
                int_backend = "te"
            elif any(
                getattr(model, attr, False)
                for attr in (
                    "__int8_training_qat__",
                    "__int8_training_ptq__",
                    "__int8_inference_ao__",
                )
                ):
                    int_backend = "ao"
        cls._preferred_fp8_backend = backend
        cls._preferred_int_backend = int_backend
        meta = metadata
        device: Optional[torch.device] = None
        if meta is not None:
            device = torch.device(meta.device)
        elif isinstance(model, nn.Module):
            tensor: Optional[torch.Tensor] = None
            with contextlib.suppress(StopIteration):
                tensor = next(model.parameters())
            if tensor is None:
                with contextlib.suppress(StopIteration):
                    tensor = next(model.buffers())
            if tensor is not None:
                device = tensor.device
        meta = cls.coerce_metadata(device, metadata=meta)
        cls._metadata = meta

    @classmethod
    @contextlib.contextmanager
    def float(
        cls: object,
        device: Optional[Union[torch.device, str]] = None,
        *args: Any,
        metadata: Optional[Metadata[Any]] = None,
        **kwargs: Any,
    ) -> contextlib.AbstractContextManager[None]:
        dev = cls._device(device)
        meta = cls.coerce_metadata(dev, metadata=metadata)
        amp_candidates = tuple(meta.float_dtypes) if meta.float_dtypes else (torch.float32,)
        amp_dtype = cls.negotiate(
            amp_candidates,
            fallback=torch.float32,
            logger=_LOGGER,
            context="float",
            device=dev,
            meta=meta,
        )
        contexts: List[contextlib.AbstractContextManager[None]] = []

        backend = cls._fp8_backend(cls._preferred_fp8_backend, device=dev)
        float8_dtypes = (
            tuple(meta.float8_dtypes)
            if meta.float8_dtypes
            else cls.float8_formats()
        )
        wants_fp8 = backend is not None
        if wants_fp8 and getattr(meta, "has_scale", False):
            fp8_supported = any(
                is_scale_safe(dtype, meta, safety_margin=2.0)
                for dtype in float8_dtypes
            )
            if not fp8_supported:
                wants_fp8 = False
                _LOGGER.debug(
                    "Autocast FP8 disabled on %s: data scale exceeds float8 range",
                    dev.type,
                )
        if wants_fp8:
            if backend == "te":
                contexts.extend(cls._nvidia_float8(dev, True))
                if not contexts:
                    backend = cls._fp8_backend("ao", device=dev)
                    if backend == "ao":
                        contexts.extend(cls._torchao_float8(True))
            elif backend == "ao":
                contexts.extend(cls._torchao_float8(True))
            else:
                _LOGGER.debug(
                    "Autocast FP8 backend '%s' unsupported; disabling", backend
                )
                cls._preferred_fp8_backend = None

        requested_dtype = amp_dtype
        if (
            isinstance(cls._last_float_dtype, torch.dtype)
            and cls._last_float_dtype in amp_candidates
        ):
            requested_dtype = cls._last_float_dtype
        if dev.type == "cuda" and requested_dtype is torch.bfloat16:
            bf16_ok = False
            if torch.cuda.is_available():
                try:
                    bf16_ok = torch.cuda.is_bf16_supported()
                except Exception:
                    try:
                        device_index = dev.index
                        if device_index is None:
                            device_index = torch.cuda.current_device()
                    except Exception:
                        device_index = 0
                    try:
                        major, _ = torch.cuda.get_device_capability(device_index)
                    except Exception:
                        major = 0
                    bf16_ok = major >= 8
            if not bf16_ok:
                _LOGGER.debug(
                    "Autocast.float falling back to fp16 on CUDA device without bf16 support"
                )
                requested_dtype = torch.float16
        if dev.type == "cpu" and requested_dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            contexts.append(contextlib.nullcontext())
            cls._last_float_dtype = requested_dtype
        else:
            try:
                ctx = torch.amp.autocast(
                    device_type=dev.type,
                    dtype=requested_dtype,
                    enabled=True,
                )
                contexts.append(ctx)
            except (RuntimeError, ValueError) as exc:
                _LOGGER.debug(
                    "Autocast.float torch.amp fallback on %s: %s", dev.type, exc
                )
                contexts.append(contextlib.nullcontext())
                cls._last_float_dtype = requested_dtype
            else:
                cls._last_float_dtype = requested_dtype
        cls._metadata = meta

        with contextlib.ExitStack() as stack:
            for ctx in contexts:
                stack.enter_context(ctx)
            yield

    @classmethod
    @contextlib.contextmanager
    def suspend(
        cls: object, device: Optional[Union[torch.device, str]] = None
    ) -> contextlib.AbstractContextManager[None]:
        dev = cls._device(device)
        with contextlib.ExitStack() as stack:
            try:
                stack.enter_context(
                    torch.amp.autocast(device_type=dev.type, enabled=False)
                )
            except (RuntimeError, ValueError):
                stack.enter_context(contextlib.nullcontext())
            yield

    @classmethod
    @contextlib.contextmanager
    def integer(
        cls: object,
        device: Optional[Union[torch.device, str]] = None,
        *args: Any,
        metadata: Optional[Metadata[Any]] = None,
        **kwargs: Any,
    ) -> contextlib.AbstractContextManager[None]:
        dev = cls._device(device)
        meta = cls.coerce_metadata(dev, metadata=metadata)
        int_candidates = tuple(meta.int_dtypes) if meta.int_dtypes else (torch.int64,)
        int_dtype = cls.negotiate(
            int_candidates,
            fallback=torch.int64,
            logger=_LOGGER,
            context="int",
            device=dev,
            meta=meta,
        )
        backend = cls._int_backend(cls._preferred_int_backend, device=dev)
        contexts = cls._torchao_int8(dev, True) if backend else []
        if not contexts and backend == "te":
            fallback_backend = cls._int_backend("ao", device=dev)
            if fallback_backend == "ao":
                contexts = cls._torchao_int8(dev, True)
        if not contexts:
            contexts.append(contextlib.nullcontext())

        with contextlib.ExitStack() as stack:
            for ctx in contexts:
                stack.enter_context(ctx)
            cls._last_int_dtype = int_dtype
            cls._metadata = meta
            yield


Int8DynamicActivationInt8WeightConfig: Any | None
Int8WeightOnlyConfig: Any | None
quantize_: Any | None
ptq: Callable[..., tuple[nn.Module, bool, str]] | None
QAT: Any | None

try:
    from torchao.quantization import (
        Int8DynamicActivationInt8WeightConfig,
        Int8WeightOnlyConfig,
        quantize_,
    )
    ptq = getattr(quantize_, "ptq", None)
except ImportError:
    quantize_ = None
    Int8DynamicActivationInt8WeightConfig = None
    Int8WeightOnlyConfig = None
    ptq = None
QATConfig = None
QATStep = None
try:
    from torchao.quantization.qat import QATConfig, QATStep
except Exception:
    try:
        from torchao.quantization.qat.api import QATConfig, QATStep
    except Exception:
        try:
            from torchao.quantization.qat import (
                FromIntXQuantizationAwareTrainingConfig,
                IntXQuantizationAwareTrainingConfig,
            )

            class _ShimQATStep:
                PREPARE = "prepare"
                CONVERT = "convert"

            class _ShimQATConfig:
                def __init__(
                    self,
                    base_config: Any = None,
                    activation_config: Any = None,
                    weight_config: Any = None,
                    *args: Any,
                    step: Any = "prepare",
                    **kwargs: Any,
                ) -> None:
                    self.base_config = base_config
                    self.activation_config = activation_config
                    self.weight_config = weight_config
                    self.step = step

                def to_legacy(self) -> Any:
                    if self.step == "prepare":
                        return IntXQuantizationAwareTrainingConfig(
                            self.activation_config, self.weight_config
                        )
                    else:
                        return FromIntXQuantizationAwareTrainingConfig()

            QATConfig, QATStep = (_ShimQATConfig, _ShimQATStep)
        except Exception:

            class _NullQATConfig:
                pass

            class _NullQATStep:
                PREPARE = "prepare"
                CONVERT = "convert"

            QATConfig, QATStep = (_NullQATConfig, _NullQATStep)
try:
    from torchao.quantization import qat as _qat_module

    QAT = _qat_module if hasattr(_qat_module, "initialize") else None
except Exception:
    QAT = None
if ptq is None:
    ptq = _is_ptq_unavailable


if QAT is None:
    QAT = _QATUnavailable()


class Quantization:

    quantize: Optional[Callable[..., Any]] = quantize_
    Int8DynamicActivationInt8WeightConfig: Optional[type] = (
        Int8DynamicActivationInt8WeightConfig
    )
    Int8WeightOnlyConfig: Optional[type] = Int8WeightOnlyConfig
    QAT: Any = QAT
    QATConfig: Any = QATConfig
    QATStep: Any = QATStep
    ptq: Callable[..., tuple[nn.Module, bool, str]] = staticmethod(ptq)

    @classmethod
    def is_available(cls: object) -> bool:
        return callable(cls.quantize)

    @classmethod
    def is_qat_available(cls: object) -> bool:
        initialize = getattr(cls.QAT, "initialize", None)
        return callable(initialize)

    @classmethod
    def is_ptq_available(cls: object) -> bool:
        return callable(cls.ptq) and cls.ptq is not _is_ptq_unavailable

    @classmethod
    def _prepare_qat(
        cls: object,
        model: nn.Module,
        *args: Any,
        dynamic_activations: bool,
        group_size: int = 128,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> Any:
        if not cls.is_qat_available():
            raise RuntimeError("QAT backend unavailable")
        return cls.QAT.initialize(
            model,
            mode="qat-int8",
            dynamic_activations=dynamic_activations,
            group_size=group_size,
            logger=logger,
        )

    @classmethod
    def _apply_ptq(
        cls: object,
        model: nn.Module,
        *args: Any,
        dynamic_activations: bool,
        group_size: int = 128,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> tuple[nn.Module, bool, str]:
        if not cls.is_ptq_available():
            return (model, False, "PTQ backend unavailable")
        return cls.ptq(
            model,
            mode="int8",
            dynamic_activations=dynamic_activations,
            group_size=group_size,
            logger=logger,
        )

    @classmethod
    def _enable_ptq(
        cls: object,
        model: nn.Module,
        *args: Any,
        dynamic_activations: bool,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> tuple[nn.Module, bool, str]:
        if not cls.is_available():
            return (model, False, "torchao.quantization not installed (INT8 disabled)")
        cfg_cls = (
            cls.Int8DynamicActivationInt8WeightConfig
            if dynamic_activations
            else cls.Int8WeightOnlyConfig
        )
        if cfg_cls is None:
            return (model, False, "Quantization config unavailable")
        try:
            cfg = cfg_cls()
        except Exception as exc:
            return (model, False, f"Failed to initialize quantization config: {exc}")
        try:
            cls.quantize(model, cfg)
        except Exception as exc:
            return (model, False, f"AO failed: {exc}")
        if logger is not None:
            logger(f"[INT8][AO] applied {cfg.__class__.__name__}")
        setattr(model, "__int8_inference_ao__", True)
        return (model, True, "torchao")

    @classmethod
    def enable_qat(
        cls: object,
        model: nn.Module,
        *args: Any,
        dynamic_activations: bool,
        group_size: int = 128,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> tuple[nn.Module, bool, str]:
        if not cls.is_available():
            msg = "torchao.quantization not installed (INT8/QAT disabled)"
            if logger:
                logger(f"[INT8] {msg}")
            return (model, False, msg)
        last_err: Optional[Exception] = None
        if cls.is_qat_available():
            try:
                base_cfg = cls._prepare_qat(
                    model,
                    dynamic_activations=dynamic_activations,
                    group_size=group_size,
                    logger=logger,
                )
                setattr(model, "__int8_training_qat__", True)
                if logger:
                    logger(
                        f"[INT8][QAT] prepared with base {base_cfg.__class__.__name__}"
                    )
                return (model, True, "QAT-prepare")
            except Exception as exc:
                last_err = exc
                if logger:
                    logger(f"[INT8][QAT] prepare failed: {exc}")
        try:
            m2, ok, why = cls._apply_ptq(
                model,
                dynamic_activations=dynamic_activations,
                group_size=group_size,
                logger=logger,
            )
        except Exception as exc:
            err = exc or last_err or RuntimeError("Unknown PTQ failure")
            return (model, False, f"INT8 training path unavailable: {err}")
        if ok:
            setattr(m2, "__int8_training_ptq__", True)
            return (m2, True, f"PTQ({why})")
        return (model, False, f"PTQ failed: {why}")

class Fusion:
    @staticmethod
    def negotiate(
        device: Optional[Union[torch.device, str]] = None,
        *args: Any,
        metadata: Optional[Metadata[Any]] = None,
        **kwargs: Any,
    ) -> torch.dtype:
        dev = torch.device(device) if device is not None else get_device()
        candidates: List[torch.dtype] = []
        if dev.type == "cuda":
            try:
                if is_cuda_bf16_supported(dev):
                    candidates.append(torch.bfloat16)
            except Exception:
                pass
            candidates.extend((torch.float16, torch.float32))
        elif dev.type == "cpu":
            if is_cpu_bf16_supported():
                candidates.append(torch.bfloat16)
            candidates.extend((torch.float32, torch.float64))
        elif dev.type == "xpu":
            candidates.extend((torch.bfloat16, torch.float32))
        elif dev.type == "mps":
            candidates.extend((torch.float16, torch.float32))
        else:
            candidates.append(torch.float32)
        for dtype in candidates:
            if is_scale_safe(dtype, metadata):
                return dtype
        return (
            torch.float64
            if is_scale_safe(torch.float64, metadata)
            else candidates[-1]
        )

    @staticmethod
    def _peek_layer(module: nn.Module) -> Optional[torch.Tensor]:
        with contextlib.suppress(StopIteration):
            return next(module.parameters())
        with contextlib.suppress(StopIteration):
            return next(module.buffers())
        return None

    @staticmethod
    def _coerce_metadata(
        model: nn.Module, metadata: Optional[Metadata[Any]] = None
    ) -> Metadata[Any]:
        Autocast.configure(model, metadata=metadata)
        meta = Autocast._metadata
        if meta is None:
            ref = Fusion._peek_layer(model)
            dev = ref.device if isinstance(ref, torch.Tensor) else get_device()
            meta = Metadata.for_device(dev)
            Autocast.configure(model, metadata=meta)
        return meta

    @staticmethod
    def _align_layers(
        src: nn.Module,
        dst: nn.Module,
        params_dtype: Optional[torch.dtype],
    ) -> None:
        ref = Fusion._peek_layer(src)
        if ref is not None:
            with contextlib.suppress(Exception):
                dst.to(device=ref.device)
        if params_dtype is not None:
            with contextlib.suppress(Exception):
                dst.to(dtype=params_dtype)

    @staticmethod
    def _clone_state(
        src: nn.Module, dst: nn.Module, params_dtype: Optional[torch.dtype]
    ) -> None:
        try:
            state = src.state_dict()
        except Exception:
            return
        ref = Fusion._peek_layer(dst)
        device = ref.device if ref is not None else None
        converted = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                tensor = value.detach()
                if params_dtype is not None and tensor.is_floating_point():
                    tensor = tensor.to(dtype=params_dtype)
                if device is not None:
                    tensor = tensor.to(device=device)
                converted[key] = tensor
            else:
                converted[key] = value
        with contextlib.suppress(Exception):
            dst.load_state_dict(converted, strict=False)

    @staticmethod
    def _nvidia_linear(
        module: nn.Linear,
        params_dtype: Optional[torch.dtype],
        te: Any,
    ) -> Optional[nn.Module]:
        te_linear = getattr(te, "Linear", None)
        if te_linear is None:
            return None
        kwargs: Dict[str, Any] = {
            "in_features": module.in_features,
            "out_features": module.out_features,
            "bias": module.bias is not None,
        }
        if params_dtype is not None:
            kwargs["params_dtype"] = params_dtype
        try:
            replacement = te_linear(**kwargs)
        except Exception:
            return None
        Fusion._align_layers(module, replacement, params_dtype)
        Fusion._clone_state(module, replacement, params_dtype)
        return replacement

    @staticmethod
    def _nvidia_layer_norm(
        module: nn.LayerNorm,
        params_dtype: Optional[torch.dtype],
        te: Any,
    ) -> Optional[nn.Module]:
        te_layer_norm = getattr(te, "LayerNorm", None)
        if te_layer_norm is None:
            return None
        kwargs: Dict[str, Any] = {
            "normalized_shape": module.normalized_shape,
            "eps": module.eps,
        }
        if params_dtype is not None:
            kwargs["params_dtype"] = params_dtype
        try:
            replacement = te_layer_norm(**kwargs)
        except Exception:
            return None
        Fusion._align_layers(module, replacement, params_dtype)
        if module.elementwise_affine:
            Fusion._clone_state(module, replacement, params_dtype)
        return replacement

    @staticmethod
    def _nvidia_rms_norm(
        module: nn.Module,
        params_dtype: Optional[torch.dtype],
        te: Any,
    ) -> Optional[nn.Module]:
        te_rms_norm = getattr(te, "RMSNorm", None)
        if te_rms_norm is None:
            return None
        kwargs: Dict[str, Any] = {
            "normalized_shape": getattr(module, "normalized_shape", None),
            "eps": getattr(module, "eps", 1e-5),
        }
        if kwargs["normalized_shape"] is None:
            return None
        if params_dtype is not None:
            kwargs["params_dtype"] = params_dtype
        try:
            replacement = te_rms_norm(**kwargs)
        except Exception:
            return None
        Fusion._align_layers(module, replacement, params_dtype)
        Fusion._clone_state(module, replacement, params_dtype)
        return replacement

    @staticmethod
    def _to_nvidia_layers(
        model: nn.Module,
        *args: Any,
        apply_te_linear: bool,
        apply_te_layer_norm: bool,
        apply_te_rms_norm: bool,
        filter_linear: Optional[Callable[[nn.Linear, str], bool]],
        params_dtype: Optional[torch.dtype],
        **kwargs: Any,
    ) -> Tuple[nn.Module, int]:
        try:
            import transformer_engine.pytorch as te
        except Exception:
            return (model, 0)

        def _convert(parent: nn.Module) -> int:
            converted = 0
            for name, child in list(parent.named_children()):
                replacement: Optional[nn.Module] = None
                if apply_te_linear and isinstance(child, nn.Linear):
                    if filter_linear is None or filter_linear(child, name):
                        replacement = Fusion._nvidia_linear(
                            child, params_dtype, te
                        )
                elif apply_te_layer_norm and isinstance(child, nn.LayerNorm):
                    replacement = Fusion._nvidia_layer_norm(
                        child, params_dtype, te
                    )
                else:
                    rms_cls = getattr(torch.nn, "RMSNorm", None)
                    if (
                        apply_te_rms_norm
                        and rms_cls is not None
                        and isinstance(child, rms_cls)
                    ):
                        replacement = Fusion._nvidia_rms_norm(
                            child, params_dtype, te
                        )
                if replacement is not None:
                    setattr(parent, name, replacement)
                    converted += 1
                    continue
                converted += _convert(child)
            return converted

        count = _convert(model)
        return (model, count)

    @staticmethod
    def _to_nvidia_attention(
        model: nn.Module, *args: Any, params_dtype: Optional[torch.dtype], **kwargs: Any
    ) -> Tuple[nn.Module, int]:
        swapped = 0
        dot_cls = _dot_product_attention_cls()
        for module in model.modules():
            if dot_cls is not None and isinstance(module, dot_cls) and getattr(
                module, "_te_ok", False
            ):
                if not getattr(module, "te_first", False):
                    module.te_first = True
                swapped += 1
        return (model, swapped)

    @staticmethod
    def use_nvidia_layers(
        model: nn.Module,
        device: Optional[Union[torch.device, str]] = None,
        *args: Any,
        metadata: Optional[Metadata[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> Tuple[nn.Module, bool, str]:
        dev = torch.device(device) if device is not None else get_device()
        if dev.type != "cuda":
            return (model, False, "Non-NVIDIA device; TE not applied")
        try:
            import transformer_engine.pytorch as te
        except Exception:
            return (model, False, "transformer_engine not installed")
        te_backend = getattr(te, "__name__", "transformer_engine.pytorch")
        fp8_ok, why = is_float8_supported(dev)
        if fp8_ok:
            setattr(model, "__te_fp8_default__", True)
        params_dtype = Fusion.negotiate(dev, metadata=metadata)
        model, n_layers = Fusion._to_nvidia_layers(
            model,
            apply_te_linear=True,
            apply_te_layer_norm=True,
            apply_te_rms_norm=True,
            filter_linear=None,
            params_dtype=params_dtype,
        )
        try:
            model, attn_swapped = Fusion._to_nvidia_attention(
                model, params_dtype=params_dtype
            )
        except Exception:
            attn_swapped = 0
        n_total = (n_layers or 0) + (attn_swapped or 0)
        if logger:
            logger(
                f"[TE] swapped {n_total} modules (layers:{n_layers}, attn:{attn_swapped}); params_dtype={str(params_dtype).split('.')[-1]}, fp8={('on' if fp8_ok else 'off')} ({(why if fp8_ok else '')}), backend={te_backend}"
            )
        return (
            model,
            n_total > 0,
            f"TE applied (swapped {n_total}, layers={n_layers}, attn={attn_swapped}, dtype={params_dtype}, fp8={('on' if fp8_ok else 'off')}, backend={te_backend})",
        )

    @staticmethod
    def _enable_nvidia_training(
        model: nn.Module,
        params_dtype: torch.dtype,
        logger: Optional[Callable[[str], None]],
    ) -> Tuple[nn.Module, bool, str]:
        try:
            swapped_model, n = Fusion._to_nvidia_layers(
                model,
                apply_te_linear=True,
                apply_te_layer_norm=True,
                apply_te_rms_norm=True,
                filter_linear=lambda lyr, _: lyr.in_features % 16 == 0
                and lyr.out_features % 16 == 0,
                params_dtype=params_dtype,
            )
            if n > 0:
                setattr(swapped_model, "__fp8_training_te__", True)
                if logger:
                    logger(f"[FP8][TE] swapped {n} modules")
                return (swapped_model, True, f"TE (swapped {n})")
            return (model, False, "TE present but no eligible modules")
        except Exception as exc:
            return (model, False, f"TE swap failed: {exc}")

    @staticmethod
    def _enable_torchao_training(
        model: nn.Module,
        logger: Optional[Callable[[str], None]],
    ) -> Tuple[nn.Module, bool, str]:
        try:
            from torchao.float8 import convert_to_float8_training

            res = convert_to_float8_training(model)
            converted = res or model
            setattr(converted, "__fp8_training_ao__", True)
            if logger:
                logger("[FP8][AO] convert_to_float8_training ok")
            return (converted, True, "torchao.float8")
        except Exception as exc:
            return (model, False, f"torchao convert failed: {exc}")

    @staticmethod
    def _enable_nvidia_inference(
        model: nn.Module,
        params_dtype: torch.dtype,
        logger: Optional[Callable[[str], None]],
    ) -> Tuple[nn.Module, bool, str]:
        try:
            swapped, n = Fusion._to_nvidia_layers(
                model,
                apply_te_linear=True,
                apply_te_layer_norm=True,
                apply_te_rms_norm=True,
                filter_linear=lambda lyr, _: lyr.in_features % 16 == 0
                and lyr.out_features % 16 == 0,
                params_dtype=params_dtype,
            )
            if n > 0:
                setattr(swapped, "__fp8_inference_te__", True)
                if logger:
                    logger(
                        f"[FP8][TE] swapped {n} modules; using te.fp8_autocast"
                    )
                return (swapped, True, f"TE swap ({n})")
            return (model, False, "no eligible Linear (dims%16)")
        except Exception as exc:
            return (model, False, f"TE swap failed: {exc}")

    @staticmethod
    def _reuse_nvidia_layers(
        model: nn.Module,
        logger: Optional[Callable[[str], None]],
    ) -> Tuple[nn.Module, bool, str]:
        te_present = any(
            (
                getattr(module.__class__, "__module__", "").startswith(
                    "transformer_engine"
                )
                for module in model.modules()
            )
        )
        if te_present:
            setattr(model, "__fp8_inference_te__", True)
            if logger:
                logger("[FP8][TE] te.* already present; using te.fp8_autocast")
            return (model, True, "TE present")
        return (model, False, "TE layers not present")

    @staticmethod
    def _enable_torchao_inference(
        model: nn.Module,
        dynamic_activations: bool,
        logger: Optional[Callable[[str], None]],
    ) -> Tuple[nn.Module, bool, str]:
        try:
            from torchao.quantization import (
                Float8DynamicActivationFloat8WeightConfig,
                Float8WeightOnlyConfig,
                quantize_,
            )

            cfg = (
                Float8DynamicActivationFloat8WeightConfig()
                if dynamic_activations
                else Float8WeightOnlyConfig()
            )
            quantize_(model, cfg)
            setattr(model, "__fp8_inference_ao__", True)
            if logger:
                logger(f"[FP8][AO] applied {cfg.__class__.__name__}")
            return (model, True, "torchao")
        except Exception as exc:
            return (model, False, f"AO failed: {exc}")

    @staticmethod
    def enable_float8_training(
        model: nn.Module,
        metadata: Optional[Metadata[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[nn.Module, bool, str]:
        meta = Fusion._coerce_metadata(model, metadata)
        device = torch.device(meta.device)
        ok, reason = is_float8_supported(device)
        if not ok:
            Autocast.configure(model, metadata=meta)
            return (model, False, reason)
        if getattr(meta, "has_scale", False):
            float8_dtypes = Autocast.float8_formats()
            if not any(
                is_scale_safe(dtype, meta, safety_margin=2.0)
                for dtype in float8_dtypes
            ):
                if logger:
                    logger(
                        "[FP8] training disabled: data scale exceeds float8 range"
                    )
                Autocast.configure(model, metadata=meta)
                return (model, False, "data scale")
        params_dtype = Fusion.negotiate(device, metadata=meta)

        for backend in ("te", "torchao"):
            if backend == "te":
                m2, ok2, why = Fusion._enable_nvidia_training(
                    model, params_dtype, logger
                )
            else:
                m2, ok2, why = Fusion._enable_torchao_training(model, logger)
            if ok2:
                if logger:
                    logger(f"[FP8] training enabled via {why} ({reason})")
                Autocast.configure(m2, metadata=meta)
                return (m2, True, why)
            elif logger:
                logger(f"[FP8] {backend} path skipped: {why}")
        Autocast.configure(model, metadata=meta)
        return (model, False, "No usable FP8 backend")

    @staticmethod
    def enable_float8_prediction(
        model: nn.Module,
        metadata: Optional[Metadata[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[nn.Module, bool, str]:
        meta = Fusion._coerce_metadata(model, metadata)
        device = torch.device(meta.device)
        ok, reason = is_float8_supported(device)
        if not ok:
            Autocast.configure(model, metadata=meta)
            return (model, False, reason)
        if getattr(meta, "has_scale", False):
            float8_dtypes = Autocast.float8_formats()
            if not any(
                is_scale_safe(dtype, meta, safety_margin=2.0)
                for dtype in float8_dtypes
            ):
                if logger:
                    logger(
                        "[FP8] inference disabled: data scale exceeds float8 range"
                    )
                Autocast.configure(model, metadata=meta)
                return (model, False, "data scale")
        params_dtype = Fusion.negotiate(device, metadata=meta)
        dynamic_activations = not (
            getattr(meta, "has_scale", False)
            and getattr(meta, "scale_is_integral", None) is True
        )
        order = ("te_swap", "te_present", "ao")
        for step in order:
            if step == "te_swap":
                m2, ok2, why = Fusion._enable_nvidia_inference(
                    model, params_dtype, logger
                )
            elif step == "te_present":
                m2, ok2, why = Fusion._reuse_nvidia_layers(model, logger)
            else:
                m2, ok2, why = Fusion._enable_torchao_inference(
                    model, dynamic_activations, logger
                )
            if ok2:
                if logger:
                    logger(f"[FP8] inference enabled via {why} ({reason})")
                Autocast.configure(m2, metadata=meta)
                return (m2, True, why)
            elif logger:
                logger(f"[FP8] {step} skipped: {why}")
        Autocast.configure(model, metadata=meta)
        return (model, False, "No usable FP8 backend")

    @staticmethod
    def enable_int8_training(
        model: nn.Module,
        metadata: Optional[Metadata[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[nn.Module, bool, str]:
        meta = Fusion._coerce_metadata(model, metadata)
        device = torch.device(meta.device)
        with contextlib.suppress(Exception):
            model.to(device)
        dynamic_activations = not (
            getattr(meta, "has_scale", False)
            and getattr(meta, "scale_is_integral", None) is True
        )
        group_size = 128
        m2, ok, why = Quantization.enable_qat(
            model,
            dynamic_activations=dynamic_activations,
            group_size=group_size,
            logger=logger,
        )
        Autocast.configure(m2 if ok else model, metadata=meta)
        return (m2, ok, why)


    @staticmethod
    def enable_int8_prediction(
        model: nn.Module,
        metadata: Optional[Metadata[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[nn.Module, bool, str]:
        meta = Fusion._coerce_metadata(model, metadata)
        device = torch.device(meta.device)
        with contextlib.suppress(Exception):
            model.to(device)
        dynamic_activations = not (
            getattr(meta, "has_scale", False)
            and getattr(meta, "scale_is_integral", None) is True
        )
        m2, ok, why = Quantization._enable_ptq(
            model, dynamic_activations=dynamic_activations, logger=logger
        )
        Autocast.configure(m2 if ok else model, metadata=meta)
        return (m2, ok, why)


    @staticmethod
    def _wrap_leaf_layer(
        module: nn.Module,
        in_key: str,
        out_key: str,
        *args: Any,
        origin_name: str,
        origin_path: str,
        **kwargs: Any,
    ) -> TensorDictModule:

        origin_type = module.__class__.__name__

        class _LeafWrapper(nn.Module):
            def __init__(self, wrapped: nn.Module) -> None:
                super().__init__()
                self.module = wrapped
                self.origin_type = origin_type
                self.origin_name = origin_name
                self.origin_path = origin_path

            def forward(self, x: torch.Tensor):
                out = self.module(x)
                if isinstance(out, dict):
                    val = out.get(
                        out_key,
                        out.get("pred", next(iter(out.values()))),
                    )
                elif isinstance(out, (tuple, list)):
                    val = out[0]
                else:
                    val = out
                return (val, self.origin_type, self.origin_name, self.origin_path)

        return TensorDictModule(
            _LeafWrapper(module),
            in_keys=[in_key],
            out_keys=[
                out_key,
                f"{out_key}__origin_type",
                f"{out_key}__origin_name",
                f"{out_key}__origin_path",
            ],
        )

    @staticmethod
    def _auto_key(parent_key: str, name: str) -> str:
        return f"{parent_key}__{name}" if parent_key else name

    @classmethod
    def _wrap_recursively(
        cls,
        module: nn.Module,
        *args: Any,
        in_key: str,
        out_key: str,
        parent_path: str = "",
        **kwargs: Any,
    ) -> TensorDictSequential:

        children = list(module.named_children())
        if not children:
            return TensorDictSequential(
                cls._wrap_leaf_layer(
                    module,
                    in_key,
                    out_key,
                    origin_name=parent_path or module.__class__.__name__,
                    origin_path=parent_path or module.__class__.__name__,
                )
            )

        seq: List[nn.Module] = []
        current_key = in_key
        current_path = parent_path
        for name, child in children:
            next_key = cls._auto_key(current_key, name)
            next_path = f"{parent_path}.{name}" if parent_path else name
            if any(True for _ in child.named_children()):
                sub_seq = cls._wrap_recursively(
                    child,
                    in_key=current_key,
                    out_key=next_key,
                    parent_path=next_path,
                )
                seq.append(sub_seq)
            else:
                seq.append(
                    cls._wrap_leaf_layer(
                        child,
                        current_key,
                        next_key,
                        origin_name=name,
                        origin_path=next_path,
                    )
                )
            current_key, current_path = next_key, next_path

        def _alias(val, typ, nm, path):
            return (val, typ, nm, path)

        alias = TensorDictModule(
            _alias,
            in_keys=[
                current_key,
                f"{current_key}__origin_type",
                f"{current_key}__origin_name",
                f"{current_key}__origin_path",
            ],
            out_keys=[
                out_key,
                f"{out_key}__origin_type",
                f"{out_key}__origin_name",
                f"{out_key}__origin_path",
            ],
        )
        seq.append(alias)
        return TensorDictSequential(*seq)

    @classmethod
    def use_tensordict_layers(
        cls,
        module: nn.Module,
        *args: Any,
        in_key: str = "features",
        out_key: str = "pred",
        add_loss: bool = True,
        global_loss: Optional[LossCallable] = None,
        local_loss: Optional[LossCallable] = None,
        loss_weights: Tuple[float, float] = (1.0, 1.0),
        cudagraph: bool = False,
        compat_call: bool = True,
        **kwargs: Any,
    ) -> nn.Module:

        pipeline: TensorDictSequential = cls._wrap_recursively(
            module,
            in_key=in_key,
            out_key=out_key,
            parent_path=module.__class__.__name__,
        )

        if add_loss:

            class TensorDictLoss(nn.Module):
                def __init__(
                    self,
                    *args: Any,
                    pred_key: str,
                    net_loss: Optional[LossCallable],
                    global_loss: Optional[LossCallable],
                    local_loss: Optional[LossCallable],
                    weights: Tuple[float, float],
                    **kwargs: Any,
                ) -> None:
                    super().__init__()
                    self.pred_key = pred_key
                    self.net_loss = net_loss
                    self.global_loss = global_loss
                    self.local_loss = local_loss
                    self.weights = tuple(float(w) for w in weights)

                def _apply_loss(
                    self,
                    loss_fn: LossCallable,
                    pred: torch.Tensor,
                    target: torch.Tensor,
                ) -> torch.Tensor:
                    return loss_fn(pred, target)

                def forward(self, td: TensorDictBase) -> TensorDictBase:
                    if self.pred_key not in td.keys(include_nested=False):
                        return td
                    pred = td.get(self.pred_key)
                    if pred is None:
                        return td
                    batch = pred.shape[0]
                    flat_pred = pred.reshape(batch, -1)
                    target = td.get("labels_flat", None)
                    total_loss: Optional[torch.Tensor] = None
                    if target is not None:
                        target = target.to(device=flat_pred.device)
                        if self.global_loss is not None or self.local_loss is not None:
                            global_loss_val = (
                                self._apply_loss(self.global_loss, flat_pred, target)
                                if self.global_loss is not None
                                else 0.0
                            )
                            if (
                                self.local_loss is not None
                                and "residual_context" in td.keys(include_nested=False)
                            ):
                                residual = td.get("residual_context")
                                if isinstance(residual, torch.Tensor):
                                    residual = residual.reshape(batch, -1).to(
                                        device=target.device
                                    )
                                    local_loss_val = self._apply_loss(
                                        self.local_loss, residual, target
                                    )
                                else:
                                    local_loss_val = 0.0
                            else:
                                local_loss_val = 0.0
                            total_loss = (
                                self.weights[0] * global_loss_val
                                + self.weights[1] * local_loss_val
                            )
                        elif self.net_loss is not None:
                            total_loss = self._apply_loss(
                                self.net_loss, flat_pred, target
                            )
                    if total_loss is not None:
                        if not isinstance(total_loss, torch.Tensor):
                            total_loss = torch.as_tensor(
                                total_loss, device=flat_pred.device
                            )
                        if total_loss.ndim == 0:
                            batch_size = tuple(td.batch_size)
                            if batch_size:
                                total_loss = total_loss.expand(batch_size)
                        td.set("loss_total", total_loss)
                    elif "loss_total" in td.keys(include_nested=False):
                        td.del_("loss_total")
                    return td

            pipeline = TensorDictSequential(
                pipeline,
                TensorDictLoss(
                    pred_key=out_key,
                    net_loss=None,
                    global_loss=global_loss,
                    local_loss=local_loss,
                    weights=loss_weights,
                ),
            )

        root_config = getattr(module, "_Root__config", None)
        derived_attrs: Dict[str, Any] = {}
        for attr in ("in_dim", "out_shape"):
            if hasattr(module, attr):
                derived_attrs[attr] = getattr(module, attr)

        if cudagraph:
            pipeline = (
                CudaGraphModule(pipeline, in_keys=[in_key, "labels_flat"])
                if add_loss
                else CudaGraphModule(pipeline, in_keys=[in_key])
            )

        if compat_call:

            class TensorDictLayer(nn.Module):
                def __init__(
                    self,
                    m: nn.Module,
                    cfg: Any,
                    in_key: str,
                    out_key: str,
                ) -> None:
                    super().__init__()
                    self.m = m
                    self._in_key = in_key
                    self._out_key = out_key
                    object.__setattr__(self, "__stnet_root_config__", cfg)

                def forward(
                    self,
                    inputs: Union[TensorDictBase, torch.Tensor],
                    **kwargs: Any,
                ):
                    if isinstance(inputs, TensorDictBase):
                        td = inputs.clone(False)
                        for key, value in kwargs.items():
                            if isinstance(value, torch.Tensor):
                                td.set(key, value)
                        return self.m(td)

                    features = inputs
                    if not isinstance(features, torch.Tensor):
                        raise TypeError(
                            "Fusion.use_tensordict_layers wrapper expects Tensor or TensorDict input"
                        )
                    batch = features.shape[0] if features.ndim > 0 else 1
                    td = TensorDict(
                        {self._in_key: features},
                        batch_size=[batch],
                        device=features.device,
                    )
                    for key, value in kwargs.items():
                        if isinstance(value, torch.Tensor):
                            td.set(key, value)
                    out_td = self.m(td)
                    pred = out_td.get(self._out_key)
                    loss_val = out_td.get("loss_total", None)
                    return (pred, loss_val)

            wrapper = TensorDictLayer(pipeline, root_config, in_key, out_key)
            for name, value in derived_attrs.items():
                object.__setattr__(wrapper, name, value)
            return wrapper

        setattr(pipeline, "__stnet_root_config__", root_config)
        for name, value in derived_attrs.items():
            setattr(pipeline, name, value)
        return pipeline
@lru_cache(maxsize=1)
def _dot_product_attention_cls() -> Any:
    try:
        from ..model.kernels import DotProductAttention as _DotProductAttention

        return _DotProductAttention
    except Exception:
        return None

