# -*- coding: utf-8 -*-
from __future__ import annotations
import contextlib
import importlib
import inspect
import logging
import warnings
import math
import os
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch._dynamo
from torch import nn, optim

from .data.stats import MetaData

from .utils.platform import (
    cuda_compute_capability,
    get_device,
    get_runtime_config,
    initialize_sdpa_backends,
    is_cpu_bf16_supported,
    is_cuda_bf16_supported,
    is_float8_supported,
    is_int4_supported,
    is_int8_supported,
    optimal_optimizer_params,
)
from .backend.compat import patch_torch
from .backend.profiler import FLOP_PROFILER, attention_flops_bshd
from .model.ops import (
    DotProductAttention,
    MultiHeadAttention,
    MultiHeadAttentionCompat,
    MultiHeadAttentionNvidia,
    MultiScaleRetention,
    MultiScaleRetentionCompat,
    attn_mask_to_additive,
)

patch_torch()

_LOGGER = logging.getLogger(__name__)


def is_transformer_engine_enabled(model: torch.nn.Module) -> bool:
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


def _is_inference_compiled(model: torch.nn.Module) -> bool:
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


def inference(model: torch.nn.Module) -> AbstractContextManager[None]:
    if (
        is_transformer_engine_enabled(model)
        or _is_inference_compiled(model)
        or _is_aot_autograd_enabled(model)
    ):
        return torch.no_grad()
    return torch.inference_mode()


def _supports_scale(
    dtype: torch.dtype,
    meta: Optional[MetaData[Any]],
    *,
    safety_margin: float = 8.0,
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
        return _supports_scale(base_dtype, meta, safety_margin=safety_margin)
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


class AutoCast:
    _fp8_backend: Optional[str] = None
    _int_backend: Optional[str] = None
    _last_float_dtype: torch.dtype = torch.float32
    _last_int_dtype: torch.dtype = torch.int64
    _metadata: Optional[MetaData[Any]] = None

    @classmethod
    def _resolve_fp8_backend(
        cls,
        preferred: Optional[str],
        *,
        device: Optional[torch.device] = None,
    ) -> Optional[str]:
        dev = device if device is not None else cls._resolve_device(None)
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
                    _LOGGER.debug("AutoCast FP8 TE unavailable: %s", reason)
                    continue
                try:
                    te = importlib.import_module("transformer_engine.pytorch")

                    if getattr(te, "fp8_autocast", None) is None:
                        raise AttributeError(
                            "transformer_engine.fp8_autocast missing"
                        )
                except Exception as exc:
                    _LOGGER.debug("AutoCast FP8 TE import failed: %s", exc)
                    continue
                cls._fp8_backend = "te"
                return "te"
            if backend == "ao":
                try:
                    _float8_mod = importlib.import_module("torchao.float8")

                    if getattr(_float8_mod, "fp8_autocast", None) is None:
                        raise AttributeError("torchao.float8.fp8_autocast missing")
                except Exception as exc:
                    _LOGGER.debug("AutoCast FP8 torchao import failed: %s", exc)
                    continue
                cls._fp8_backend = "ao"
                return "ao"
        cls._fp8_backend = None
        return None

    @classmethod
    def _resolve_int_backend(
        cls,
        preferred: Optional[str],
        *,
        device: Optional[torch.device] = None,
    ) -> Optional[str]:
        dev = device if device is not None else cls._resolve_device(None)
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
                    _LOGGER.debug("AutoCast INT8 TE unavailable: %s", reason)
                    continue
                try:
                    te = importlib.import_module("transformer_engine.pytorch")

                    if getattr(te, "int8_autocast", None) is None:
                        raise AttributeError(
                            "transformer_engine.int8_autocast missing"
                        )
                except Exception as exc:
                    _LOGGER.debug("AutoCast INT8 TE import failed: %s", exc)
                    continue
                cls._int_backend = "te"
                return "te"
            if backend == "ao":
                try:
                    quant_mod = importlib.import_module("torchao.quantization")
                    int8_autocast = getattr(quant_mod, "int8_autocast", None)

                    if not callable(int8_autocast):
                        raise AttributeError("torchao.quantization.int8_autocast missing")
                except Exception as exc:
                    _LOGGER.debug("AutoCast INT8 torchao import failed: %s", exc)
                    continue
                cls._int_backend = "ao"
                return "ao"
        cls._int_backend = None
        return None

    @staticmethod
    def _resolve_device(
        device: Optional[Union[torch.device, str]] = None,
    ) -> torch.device:
        if device is None:
            return get_device()
        if isinstance(device, torch.device):
            return device
        return torch.device(device)

    @classmethod
    def _ensure_metadata(
        cls,
        device: Optional[Union[torch.device, str]] = None,
        *,
        metadata: Optional[MetaData[Any]] = None,
    ) -> MetaData[Any]:
        meta = metadata or cls._metadata
        device_hint: Optional[Union[torch.device, str]] = device
        if device_hint is None and meta is not None:
            with contextlib.suppress(Exception):
                device_hint = torch.device(meta.device)
        dev = cls._resolve_device(device_hint)
        if meta is None:
            meta = MetaData.for_device(dev)
        else:
            current_device = torch.device(getattr(meta, "device", dev))
            if current_device != dev:
                meta.device = dev
                meta.refresh()
            else:
                meta.ensure_device_info()
        if not getattr(meta, "float_dtypes", ()):  # type: ignore[attr-defined]
            meta.refresh()
        elif (
            not getattr(meta, "int_dtypes", ())
            or not getattr(meta, "float8_dtypes", ())
        ):  # type: ignore[attr-defined]
            meta.refresh()
        else:
            meta.ensure_device_info()
        cls._metadata = meta
        return meta

    @staticmethod
    def _coerce_dtype(value: Optional[Union[str, torch.dtype]]) -> Optional[torch.dtype]:
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            candidate = getattr(torch, value, None)
            if isinstance(candidate, torch.dtype):
                return candidate
        return None

    @classmethod
    def _float_amp_candidates(cls, device: torch.device) -> Tuple[torch.dtype, ...]:
        meta = cls._ensure_metadata(device)
        candidates = getattr(meta, "float_dtypes", ())
        if candidates:
            return tuple(candidates)
        return (torch.float32,)

    @staticmethod
    def _float8_dtypes() -> Tuple[torch.dtype, ...]:
        meta = AutoCast._metadata
        if meta is not None and getattr(meta, "float8_dtypes", None):
            return tuple(meta.float8_dtypes)
        values = MetaData._float8_dtypes()
        if meta is not None:
            meta.float8_dtypes = values
        return values

    @classmethod
    def _integer_candidates(cls, device: torch.device) -> Tuple[torch.dtype, ...]:
        meta = cls._ensure_metadata(device)
        candidates = getattr(meta, "int_dtypes", ())
        if candidates:
            return tuple(candidates)
        return (torch.int64,)

    @classmethod
    def _select_dtype(
        cls,
        candidates: Tuple[torch.dtype, ...],
        *,
        fallback: torch.dtype,
        logger: Optional[logging.Logger] = None,
        context: str = "autocast",
        device: Optional[torch.device] = None,
        meta: Optional[MetaData[Any]] = None,
    ) -> torch.dtype:
        for dtype in candidates:
            if _supports_scale(dtype, meta):
                return dtype
        device_str = f" on {device.type}" if device is not None else ""
        fallback_order: Tuple[torch.dtype, ...]
        if getattr(fallback, "is_floating_point", False):
            fallback_order = (fallback, torch.float32, torch.float64)
        else:
            fallback_order = (fallback, torch.int64, torch.float32, torch.float64)
        for dtype in fallback_order:
            if _supports_scale(dtype, meta):
                if logger is not None and dtype is not fallback:
                    logger.debug(
                        "AutoCast %s fallback%s: promoting to %s due to data scale",
                        context,
                        device_str,
                        str(dtype).split(".")[-1],
                    )
                return dtype
        if logger is not None:
            logger.debug(
                "AutoCast %s fallback%s: using %s without scale guarantee",
                context,
                device_str,
                str(fallback).split(".")[-1],
            )
        return fallback

    @classmethod
    def _te_fp8_context(
        cls, device: torch.device, enabled: bool
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
            _LOGGER.debug("AutoCast FP8 TE failed: %s", exc)
            cls._fp8_backend = None
        return contexts

    @classmethod
    def _ao_fp8_context(
        cls, enabled: bool
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
            _LOGGER.debug("AutoCast FP8 torchao failed: %s", exc)
            cls._fp8_backend = None
        return contexts

    @classmethod
    def _int8_context(
        cls,
        device: torch.device,
        enabled: bool,
    ) -> List[contextlib.AbstractContextManager[None]]:
        contexts: List[contextlib.AbstractContextManager[None]] = []
        if not enabled:
            return contexts
        backend = cls._int_backend
        if backend == "te":
            try:
                te = importlib.import_module("transformer_engine.pytorch")

                int_ctx = getattr(te, "int8_autocast", None)
                if callable(int_ctx):
                    contexts.append(int_ctx(enabled=True))
                else:
                    raise AttributeError("transformer_engine.int8_autocast missing")
            except Exception as exc:
                _LOGGER.debug("AutoCast INT8 TE failed: %s", exc)
                cls._int_backend = None
        elif backend == "ao":
            try:
                quant_mod = importlib.import_module("torchao.quantization")
                int8_autocast = getattr(quant_mod, "int8_autocast", None)

                if callable(int8_autocast):
                    contexts.append(int8_autocast(enabled=True))
                else:
                    raise AttributeError("torchao.quantization.int8_autocast missing")
            except Exception as exc:
                _LOGGER.debug("AutoCast INT8 torchao failed: %s", exc)
                cls._int_backend = None
        return contexts

    @classmethod
    def configure(
        cls,
        model: Optional[nn.Module],
        *,
        metadata: Optional[MetaData[Any]] = None,
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
        cls._fp8_backend = backend
        cls._int_backend = int_backend
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
        meta = cls._ensure_metadata(device, metadata=meta)
        cls._metadata = meta

    @classmethod
    @contextlib.contextmanager
    def float(
        cls,
        device: Optional[Union[torch.device, str]] = None,
        *,
        metadata: Optional[MetaData[Any]] = None,
    ) -> contextlib.AbstractContextManager[None]:
        dev = cls._resolve_device(device)
        meta = cls._ensure_metadata(dev, metadata=metadata)
        amp_candidates = tuple(meta.float_dtypes) if meta.float_dtypes else (torch.float32,)
        amp_dtype = cls._select_dtype(
            amp_candidates,
            fallback=torch.float32,
            logger=_LOGGER,
            context="float",
            device=dev,
            meta=meta,
        )
        contexts: List[contextlib.AbstractContextManager[None]] = []

        backend = cls._resolve_fp8_backend(cls._fp8_backend, device=dev)
        float8_dtypes = (
            tuple(meta.float8_dtypes)
            if meta.float8_dtypes
            else cls._float8_dtypes()
        )
        wants_fp8 = backend is not None
        if wants_fp8 and getattr(meta, "has_scale", False):
            fp8_supported = any(
                _supports_scale(dtype, meta, safety_margin=2.0)
                for dtype in float8_dtypes
            )
            if not fp8_supported:
                wants_fp8 = False
                _LOGGER.debug(
                    "AutoCast FP8 disabled on %s: data scale exceeds float8 range",
                    dev.type,
                )
        if wants_fp8:
            if backend == "te":
                contexts.extend(cls._te_fp8_context(dev, True))
                if not contexts:
                    backend = cls._resolve_fp8_backend("ao", device=dev)
                    if backend == "ao":
                        contexts.extend(cls._ao_fp8_context(True))
            elif backend == "ao":
                contexts.extend(cls._ao_fp8_context(True))
            else:
                _LOGGER.debug(
                    "AutoCast FP8 backend '%s' unsupported; disabling", backend
                )
                cls._fp8_backend = None

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
                    "AutoCast.float falling back to fp16 on CUDA device without bf16 support"
                )
                requested_dtype = torch.float16
        try:
            ctx = torch.amp.autocast(
                device_type=dev.type,
                dtype=requested_dtype,
                enabled=True,
            )
            contexts.append(ctx)
        except (RuntimeError, ValueError) as exc:
            _LOGGER.debug(
                "AutoCast.float torch.amp fallback on %s: %s", dev.type, exc
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
        cls, device: Optional[Union[torch.device, str]] = None
    ) -> contextlib.AbstractContextManager[None]:
        dev = cls._resolve_device(device)
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
        cls,
        device: Optional[Union[torch.device, str]] = None,
        *,
        metadata: Optional[MetaData[Any]] = None,
    ) -> contextlib.AbstractContextManager[None]:
        dev = cls._resolve_device(device)
        meta = cls._ensure_metadata(dev, metadata=metadata)
        int_candidates = tuple(meta.int_dtypes) if meta.int_dtypes else (torch.int64,)
        int_dtype = cls._select_dtype(
            int_candidates,
            fallback=torch.int64,
            logger=_LOGGER,
            context="int",
            device=dev,
            meta=meta,
        )
        backend = cls._resolve_int_backend(cls._int_backend, device=dev)
        contexts = cls._int8_context(dev, True) if backend else []
        if not contexts and backend == "te":
            fallback_backend = cls._resolve_int_backend("ao", device=dev)
            if fallback_backend == "ao":
                contexts = cls._int8_context(dev, True)
        if not contexts:
            contexts.append(contextlib.nullcontext())

        with contextlib.ExitStack() as stack:
            for ctx in contexts:
                stack.enter_context(ctx)
            cls._last_int_dtype = int_dtype
            cls._metadata = meta
            yield


def compile(
    module: nn.Module,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    fullgraph: Optional[bool] = None,
    dynamic: Optional[bool] = None,
    options: Optional[Dict[str, Any]] = None,
    disable: bool = False,
) -> nn.Module:
    normalized_mode = ""
    if mode is not None:
        try:
            normalized_mode = str(mode).strip().lower()
        except Exception:
            normalized_mode = ""
    if disable or normalized_mode in {"", "disabled", "none"}:
        return module
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return module
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
    try:
        return compile_fn(module, **kwargs)
    except Exception as exc:
        _LOGGER.warning("torch.compile failed (%s); returning original module", exc)
        return module


@dataclass
class LossWeightController:
    momentum: float = 0.9
    min_weight: float = 0.05
    max_weight: float = 0.95
    eps: float = 1e-06
    top_avg: float = 1.0
    bottom_avg: float = 1.0

    def weights(self) -> Tuple[float, float]:
        top = max(self.eps, self.top_avg)
        bottom = max(self.eps, self.bottom_avg)
        total = top + bottom
        if total <= 0.0:
            return (0.5, 0.5)
        ratio_top = top / total
        ratio_bottom = bottom / total
        ratio_top = float(min(max(ratio_top, self.min_weight), self.max_weight))
        ratio_bottom = float(
            min(max(ratio_bottom, self.min_weight), self.max_weight)
        )
        norm = ratio_top + ratio_bottom
        if norm <= 0.0:
            return (0.5, 0.5)
        return (ratio_top / norm, ratio_bottom / norm)

    def update(
        self,
        top_loss: Optional[torch.Tensor],
        bottom_loss: Optional[torch.Tensor],
    ) -> None:
        if top_loss is not None:
            top_val = float(top_loss.detach().abs().mean().item())
            self.top_avg = self.momentum * self.top_avg + (
                1.0 - self.momentum
            ) * max(top_val, self.eps)
        if bottom_loss is not None:
            bottom_val = float(bottom_loss.detach().abs().mean().item())
            self.bottom_avg = self.momentum * self.bottom_avg + (
                1.0 - self.momentum
            ) * max(bottom_val, self.eps)


def _import_callable(spec: str) -> Callable:
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("Empty spec for callable import")
    raw = spec.strip()
    root_pkg = __package__.split(".", 1)[0] if __package__ else "stnet"
    default_module = f"{root_pkg}.kernels"
    if ":" in raw:
        mod_part, fn_part = raw.split(":", 1)
    else:
        mod_part, fn_part = ("", raw)
    mod_part = mod_part.strip()
    fn_part = fn_part.strip()
    if not fn_part:
        raise ValueError(f"Missing function in spec: {spec}")
    if not mod_part:
        mod_name = default_module
    elif mod_part.startswith("."):
        mod_name = f"{root_pkg}{mod_part}"
    elif not mod_part.startswith(root_pkg + ".") and mod_part.split(".")[
        0
    ] not in ("importlib", "torch", "math", "sys"):
        mod_name = f"{root_pkg}.{mod_part}"
    else:
        mod_name = mod_part
    module = importlib.import_module(mod_name)
    fn = getattr(module, fn_part, None)
    if not callable(fn):
        raise TypeError(f"{mod_name}:{fn_part} is not callable or not found")
    return fn


