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
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    Union,
)

import torch
import torch._dynamo
from torch import nn, optim

from ..data.stats import MetaData

from .platform import (
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
from .compat import patch_torch
from .profiler import FLOP_PROFILER, attention_flops_bshd

patch_torch()

_LOGGER = logging.getLogger(__name__)


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


@contextlib.contextmanager
def no_synchronization(
    model: nn.Module,
    *,
    enable: bool = True,
) -> contextlib.AbstractContextManager[None]:
    if not enable:
        yield
        return
    ctx = None
    try:
        no_sync = getattr(model, "no_sync", None)
        if callable(no_sync):
            ctx = no_sync()
    except Exception:
        ctx = None
    if ctx is None:
        yield
        return
    with ctx:
        yield

try:
    from torch.distributed.algorithms.join import Join as _TorchJoin
except ImportError:
    _TorchJoin = None

Join: type[AbstractContextManager[None]] | None = _TorchJoin

if TYPE_CHECKING:
    from torch.distributed._composable.fsdp import FSDPModule
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.nn.parallel import DistributedDataParallel as DDP
else:
    DDP = object
    FSDP = object
    FSDPModule = object

JoinableModel: TypeAlias = Union["DDP", "FSDP", "FSDPModule"]


def _duplicate_last_dim(x: torch.Tensor) -> torch.Tensor:
    return torch.stack((x, x), dim=-1).reshape(*x.shape[:-1], -1)


def _retention_manual_flops(
    batch: int,
    seq_len: int,
    *,
    num_heads: int,
    head_dim: int,
    use_gate: bool,
) -> float:
    if batch <= 0 or seq_len <= 0 or num_heads <= 0 or head_dim <= 0:
        return 0.0
    attn = float(batch) * float(seq_len) * float(num_heads) * float(head_dim)
    gate_cost = attn if use_gate else 0.0
    return 4.0 * attn + gate_cost


def _is_contiguous_bshd(tensor: torch.Tensor) -> bool:
    if tensor.dim() != 4:
        return False
    _, seq_len, num_heads, head_dim = tensor.shape
    stride = tensor.stride()
    return (
        tensor.is_contiguous()
        and stride[-1] == 1
        and stride[-2] == head_dim
        and stride[-3] == num_heads * head_dim
        and stride[-4] == seq_len * num_heads * head_dim
    )


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


############################################
# TE-friendly mask helpers (addend = -inf) #
############################################


def _canonical_lengths(
    query: torch.Tensor, key: torch.Tensor, batch_first: bool
) -> tuple[int, int, int]:
    """Return (batch, seq_len_query, seq_len_key) derived from query/key shapes."""

    if batch_first:
        if query.dim() < 2 or key.dim() < 2:
            raise ValueError("expected query/key tensors with at least 2 dims when batch_first=True")
        batch = int(query.shape[0])
        seq_q = int(query.shape[1])
        seq_k = int(key.shape[1])
    else:
        if query.dim() < 2 or key.dim() < 2:
            raise ValueError("expected query/key tensors with at least 2 dims when batch_first=False")
        batch = int(query.shape[1])
        seq_q = int(query.shape[0])
        seq_k = int(key.shape[0])
    return batch, seq_q, seq_k


def _expand_bool_mask_to_bhss(
    mask: torch.Tensor,
    *,
    batch: int,
    heads: int,
    seq_q: int,
    seq_k: int,
    device: torch.device,
) -> torch.Tensor:
    """Expand a boolean mask into [B, H, S_q, S_k] layout."""

    if mask.dtype is not torch.bool:
        raise TypeError("expected boolean mask")
    if mask.dim() == 2:
        if mask.shape != (seq_q, seq_k):
            raise ValueError(f"mask shape {tuple(mask.shape)} incompatible with ({seq_q}, {seq_k})")
        expanded = mask.view(1, 1, seq_q, seq_k).expand(batch, heads, seq_q, seq_k)
    elif mask.dim() == 3:
        if mask.shape == (batch, seq_q, seq_k):
            expanded = mask.view(batch, 1, seq_q, seq_k).expand(batch, heads, seq_q, seq_k)
        else:
            raise ValueError(f"unsupported 3D mask shape {tuple(mask.shape)}")
    elif mask.dim() == 4:
        b, h, sq, sk = mask.shape
        if b != batch or sq != seq_q or sk != seq_k:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} incompatible with (batch={batch}, seq_q={seq_q}, seq_k={seq_k})"
            )
        if h == 1:
            expanded = mask.expand(batch, heads, seq_q, seq_k)
        elif h == heads:
            expanded = mask
        else:
            raise ValueError(f"mask head dimension {h} does not match expected heads {heads}")
    else:
        raise ValueError(f"unsupported mask rank {mask.dim()}")
    return expanded.to(device=device, dtype=torch.bool, non_blocking=True)


def attn_mask_to_additive(
    attn_mask: torch.Tensor | None,
    *,
    batch: int,
    heads: int,
    seq_q: int,
    seq_k: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Normalize 2D/3D/4D attention masks into a [B, H, S_q, S_k] float addend.
    - Boolean masks -> -inf at masked positions (TE-compatible)
    - Floating masks -> broadcast to [B, H, S_q, S_k]
    - None -> all zeros
    """

    if attn_mask is None:
        return torch.zeros((batch, heads, seq_q, seq_k), dtype=dtype, device=device)
    if attn_mask.dtype is torch.bool:
        expanded = _expand_bool_mask_to_bhss(
            attn_mask, batch=batch, heads=heads, seq_q=seq_q, seq_k=seq_k, device=device
        )
        neg_inf = _te_addend(dtype, device)
        zero = torch.zeros((), dtype=neg_inf.dtype, device=device)
        return torch.where(expanded, neg_inf, zero).to(dtype)
    am = attn_mask.to(device=device, dtype=dtype, non_blocking=True)
    if am.dim() == 2:
        if am.shape != (seq_q, seq_k):
            raise ValueError(f"mask shape {tuple(am.shape)} incompatible with ({seq_q}, {seq_k})")
        return am.view(1, 1, seq_q, seq_k).expand(batch, heads, seq_q, seq_k).contiguous()
    if am.dim() == 3:
        if am.shape != (batch, seq_q, seq_k):
            raise ValueError(
                f"mask shape {tuple(am.shape)} incompatible with (batch={batch}, seq_q={seq_q}, seq_k={seq_k})"
            )
        return am.view(batch, 1, seq_q, seq_k).expand(batch, heads, seq_q, seq_k).contiguous()
    if am.dim() == 4:
        b, h, sq, sk = am.shape
        if b != batch or sq != seq_q or sk != seq_k:
            raise ValueError(
                f"mask shape {tuple(am.shape)} incompatible with (batch={batch}, seq_q={seq_q}, seq_k={seq_k})"
            )
        if h == 1:
            return am.expand(batch, heads, seq_q, seq_k).contiguous()
        if h == heads:
            return am.contiguous()
        raise ValueError(f"mask head dimension {h} does not match expected heads {heads}")
    raise ValueError(f"unsupported mask rank {am.dim()}")


def _te_addend(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return -inf scalar; non-floating dtype은 float32로 승격."""

    if not torch.is_floating_point(torch.empty((), dtype=dtype)):
        dtype = torch.float32
    return torch.tensor(float("-inf"), dtype=dtype, device=device)


def _te_supported_on_device() -> bool:
    """Conservatively enable TE fused MHA only on sufficiently new CUDA devices."""

    if os.environ.get("STF_DISABLE_TE", "") == "1":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        device = get_device()
    except Exception:
        device = torch.device("cuda", 0)
    if device.type != "cuda":
        return False
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
    except Exception:
        index = 0
    try:
        props = torch.cuda.get_device_properties(index)
        maj = int(getattr(props, "major", 0))
        minr = int(getattr(props, "minor", 0))
    except Exception:
        try:
            maj, minr = torch.cuda.get_device_capability(index)
        except Exception:
            return False
    raw_cc = os.environ.get("STF_TE_MIN_CC", "")
    min_major, min_minor = 8, 0
    if raw_cc:
        try:
            parsed = int(raw_cc)
        except Exception:
            parsed = 80
        if parsed < 10:
            min_major, min_minor = parsed, 0
        else:
            min_major, min_minor = divmod(parsed, 10)
    if maj < min_major or (maj == min_major and minr < min_minor):
        return False
    try:
        if torch._dynamo.is_compiling() and os.environ.get("STF_TE_ALLOW_INDUCTOR", "0") != "1":
            return False
    except Exception:
        pass
    return True


def _adapt_mask_for_te(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    key_padding_mask: Optional[torch.Tensor],
    *,
    num_heads: int,
    batch_first: bool,
) -> Optional[torch.Tensor]:
    """Convert attention masks to TE-friendly float addends."""

    if attn_mask is None and key_padding_mask is None:
        return None

    batch, seq_q, seq_k = _canonical_lengths(query, key, batch_first)
    device = query.device
    dtype = query.dtype
    heads = int(num_heads)
    neg_inf = _te_addend(dtype, device)
    mask_dtype = neg_inf.dtype
    zero_scalar = torch.zeros((), dtype=mask_dtype, device=device)
    float_mask: Optional[torch.Tensor] = None

    try:
        if attn_mask is not None:
            if attn_mask.dtype is torch.bool:
                expanded = _expand_bool_mask_to_bhss(
                    attn_mask,
                    batch=batch,
                    heads=heads,
                    seq_q=seq_q,
                    seq_k=seq_k,
                    device=device,
                )
                float_mask = torch.where(expanded, neg_inf, zero_scalar)
            elif torch.is_floating_point(attn_mask):
                am = attn_mask.to(device=device, dtype=mask_dtype, non_blocking=True)
                if am.dim() == 2:
                    if am.shape != (seq_q, seq_k):
                        return None
                    float_mask = am.view(1, 1, seq_q, seq_k).expand(batch, heads, seq_q, seq_k).clone()
                elif am.dim() == 3:
                    if am.shape != (batch, seq_q, seq_k):
                        return None
                    float_mask = am.view(batch, 1, seq_q, seq_k).expand(batch, heads, seq_q, seq_k).clone()
                elif am.dim() == 4:
                    if am.shape[0] != batch or am.shape[2] != seq_q or am.shape[3] != seq_k:
                        return None
                    if am.shape[1] == 1:
                        float_mask = am.expand(batch, heads, seq_q, seq_k).clone()
                    elif am.shape[1] == heads:
                        float_mask = am.clone()
                    else:
                        return None
                else:
                    return None
            else:
                return None

        if key_padding_mask is not None:
            if key_padding_mask.dtype is not torch.bool:
                key_padding_mask = key_padding_mask.to(device=device, dtype=torch.bool, non_blocking=True)
            else:
                key_padding_mask = key_padding_mask.to(device=device, non_blocking=True)
            if key_padding_mask.dim() != 2 or key_padding_mask.shape != (batch, seq_k):
                return None
            padding = key_padding_mask.view(batch, 1, 1, seq_k)
            pad_values = torch.where(
                padding.expand(batch, heads, seq_q, seq_k),
                neg_inf,
                zero_scalar,
            )
            float_mask = pad_values if float_mask is None else float_mask + pad_values

        return float_mask.contiguous() if float_mask is not None else None
    except Exception:
        return None


def inference(model: torch.nn.Module) -> AbstractContextManager[None]:
    if (
        is_transformer_engine_enabled(model)
        or _is_inference_compiled(model)
        or _is_aot_autograd_enabled(model)
    ):
        return torch.no_grad()
    return torch.inference_mode()


def _has_join_hook(obj: Any | None) -> bool:
    if obj is None:
        return False
    return getattr(obj, "join_hook", None) is not None

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


def _ptq_unavailable(
    model: nn.Module,
    *args: Any,
    **kwargs: Any,
) -> tuple[nn.Module, bool, str]:
    return (model, False, "PTQ backend unavailable")


if ptq is None:
    ptq = _ptq_unavailable


if QAT is None:

    class _QATUnavailable:
        @staticmethod
        def initialize(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("QAT backend unavailable")

    QAT = _QATUnavailable()


def joining(
    model: JoinableModel,
    optimizer: optim.Optimizer | None = None,
) -> AbstractContextManager[None]:
    if Join is None:
        return contextlib.nullcontext()
    joinables = tuple(obj for obj in (model, optimizer) if _has_join_hook(obj))
    if not joinables:
        return contextlib.nullcontext()
    return Join(joinables, throw_on_early_termination=True)


class DotProductAttention(nn.Module):
    def __init__(
        self,
        num_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        te_first: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.nh = int(num_heads) if num_heads is not None else None
        self.hd = int(head_dim) if head_dim is not None else None
        cfg = get_runtime_config()
        self.te_first = bool(cfg.te_first) if te_first is None else bool(te_first)
        ok, te = self._is_te_available()
        self._te_ok = bool(
            ok
            and torch.cuda.is_available()
            and _te_supported_on_device()
            and (self.nh is not None)
            and (self.hd is not None)
        )
        self._force_pt: bool = False
        self._te_attn: Any = None
        self._te_forward_signature: inspect.Signature | None = None
        self._te_mask_param: str | None = None
        self._te_mask_type_param: str | None = None
        self._te_core_bias_param: str | None = None
        self._te_core_bias_type_param: str | None = None
        self._te_supports_mask = False
        self._te_supports_mask_type = False
        self._te_supports_core_bias = False
        self._te_supports_core_bias_type = False
        self._te_supports_attention_dropout = False
        self._te_supports_is_causal = False
        self._te_supports_training = False
        if self._te_ok:
            self._te = te
            try:
                self._te_attn = te.DotProductAttention(
                    num_attention_heads=self.nh,
                    kv_channels=self.hd,
                    qkv_format="bshd",  # TE는 [B,S,H,D] 기대
                    attention_dropout=0.0,
                )
            except Exception:
                # TE 초기화 실패 시 즉시 PyTorch 경로로 고정
                self._te_attn = None
                self._force_pt = True
            if self._te_attn is not None:
                _forward = getattr(
                    self._te_attn,
                    "forward",
                    getattr(self._te_attn, "__call__", None),
                )
                if _forward is not None:
                    try:
                        self._te_forward_signature = inspect.signature(_forward)
                    except (TypeError, ValueError):
                        self._te_forward_signature = None
                params = (
                    self._te_forward_signature.parameters
                    if self._te_forward_signature
                    else {}
                )
                if "attention_mask" in params:
                    self._te_mask_param = "attention_mask"
                elif "attn_mask" in params:
                    self._te_mask_param = "attn_mask"
                if "attn_mask_type" in params:
                    self._te_mask_type_param = "attn_mask_type"
                elif "attention_mask_type" in params:
                    self._te_mask_type_param = "attention_mask_type"
                if "core_attention_bias" in params:
                    self._te_core_bias_param = "core_attention_bias"
                if "core_attention_bias_type" in params:
                    self._te_core_bias_type_param = "core_attention_bias_type"
                self._te_supports_mask = self._te_mask_param is not None
                self._te_supports_mask_type = (
                    self._te_mask_type_param is not None
                )
                self._te_supports_core_bias = (
                    self._te_core_bias_param is not None
                )
                self._te_supports_core_bias_type = (
                    self._te_core_bias_type_param is not None
                )
                self._te_supports_attention_dropout = (
                    "attention_dropout" in params
                )
                self._te_supports_is_causal = "is_causal" in params
                self._te_supports_training = "training" in params

    @staticmethod
    def _is_te_available() -> Any:
        if os.environ.get("STF_DISABLE_TE", "") == "1":
            return (False, None)
        try:
            with contextlib.ExitStack() as stack:
                import warnings as _warnings

                stack.enter_context(_warnings.catch_warnings())
                _warnings.filterwarnings(
                    "ignore",
                    message="Detected a Jax installation.*",
                    category=RuntimeWarning,
                )
                import transformer_engine.pytorch as te

            return (True, te)
        except Exception:
            return (False, None)

    @torch._dynamo.disable()
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float | torch.Tensor = 0.0,
        is_causal: bool = False,
        training: bool | None = None,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        training = bool(training if training is not None else self.training)
        if isinstance(dropout_p, torch.Tensor):
            dropout_p = float(dropout_p.item())
        q = self._to_optimal_dtype(q)
        k = self._to_optimal_dtype(k)
        v = self._to_optimal_dtype(v)
        q_bshd = q.contiguous()
        k_bshd = k.contiguous()
        v_bshd = v.contiguous()
        if _is_contiguous_bshd(q_bshd) and _is_contiguous_bshd(k_bshd):
            try:
                attention_flops_bshd(
                    q_bshd,
                    bwd_factor=2.0 if training else 0.0,
                    dropout_p=float(dropout_p),
                    training=training,
                )
            except Exception:
                pass
        dropout_val = float(dropout_p) if training else 0.0

        B, H, L, D = q_bshd.shape
        S = k_bshd.shape[2]

        mask_bool: torch.Tensor | None = None
        bias_float: torch.Tensor | None = None
        if attn_mask is not None:
            m = attn_mask
            if m.dtype == torch.bool:
                mask_bool = m
            elif torch.is_floating_point(m):
                bias_float = m
            elif m.dtype in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                try:
                    sampled = m if m.numel() <= 4096 else m.reshape(-1)[:4096]
                    uniq = torch.unique(sampled)
                except Exception:
                    uniq = torch.tensor([], device=m.device, dtype=m.dtype)
                if uniq.numel() <= 2 and set(uniq.tolist()).issubset({0, 1}):
                    mask_bool = m != 0
                else:
                    bias_float = m.to(dtype=q_bshd.dtype)
            else:
                mask_bool = m.to(torch.bool)

            def _flatten_mask_shape(
                mask: torch.Tensor,
            ) -> tuple[torch.Tensor, int, int]:
                if mask.dim() == 0:
                    shaped = mask.to(device=q_bshd.device).view(1, 1, 1, 1)
                    shaped = shaped.expand(1, 1, L, S)
                    return shaped.contiguous(), 1, 1
                if mask.dim() < 2:
                    raise RuntimeError(
                        f"attn_mask rank {mask.dim()} not supported; expected at least 2 dimensions"
                    )
                if mask.shape[-2:] != (L, S):
                    raise RuntimeError(
                        "attn_mask trailing dims {} do not match expected (L={}, S={})".format(
                            tuple(mask.shape[-2:]), L, S
                        )
                    )
                mask = mask.to(device=q_bshd.device).contiguous()
                while True:
                    leading = mask.shape[:-2]
                    if not leading:
                        return mask.view(1, 1, L, S).contiguous(), 1, 1
                    batch_dim = leading[0]
                    if batch_dim in (B, 1):
                        head_dims = leading[1:]
                        break
                    if batch_dim == H:
                        mask = mask.unsqueeze(0)
                        continue
                    raise RuntimeError(
                        f"attn_mask batch dimension {batch_dim} incompatible with batch {B}"
                    )
                head_dims = tuple(head_dims)
                head_count = 1 if not head_dims else math.prod(head_dims)
                mask = mask.view(batch_dim, head_count, L, S)
                if head_count not in (1, H):
                    raise RuntimeError(
                        "attn_mask head dims {} collapse to {} which is not compatible with num_heads {}".format(
                            head_dims, head_count, H
                        )
                    )
                return mask.contiguous(), int(batch_dim), int(head_count)

            def _to_bhls(
                x: torch.Tensor | None, *, dtype: torch.dtype | None = None
            ) -> torch.Tensor | None:
                if x is None:
                    return None
                mask, batch_dim, head_count = _flatten_mask_shape(x)
                if batch_dim != B:
                    mask = mask.expand(B, head_count, L, S)
                    batch_dim = B
                if head_count == 1:
                    mask = mask.expand(batch_dim, H, L, S)
                elif head_count == H and batch_dim != B:
                    mask = mask.expand(B, head_count, L, S)
                if dtype is not None and mask.dtype != dtype:
                    mask = mask.to(dtype=dtype)
                return mask.contiguous()

            mask_bool = _to_bhls(mask_bool)
            if bias_float is not None:
                bias_float = _to_bhls(bias_float, dtype=q_bshd.dtype)

            # TE가 bool mask 미지원이고 core bias는 받는 경우: bool→bias(-inf/0)로 변환
            if (
                mask_bool is not None
                and bias_float is None
                and self._te_ok
                and self._te_attn is not None
                and not self._te_supports_mask
                and self._te_supports_core_bias
            ):
                finfo = torch.finfo(q_bshd.dtype)
                neg_inf = torch.full(
                    (), finfo.min, dtype=q_bshd.dtype, device=q_bshd.device
                )
                zero = torch.zeros((), dtype=q_bshd.dtype, device=q_bshd.device)
                bias_float = torch.where(mask_bool, neg_inf, zero).expand_as(mask_bool)
                mask_bool = None

        try:
            is_compiling = bool(torch._dynamo.is_compiling())
        except Exception:
            is_compiling = False
        # 보수적 TE 게이트
        use_te = (
            self.te_first
            and self._te_ok
            and not self._force_pt
            and (self._te_attn is not None)
            and (not kwargs)
            and not is_compiling
            and q_bshd.is_cuda
            and q_bshd.dtype in (torch.float16, torch.bfloat16)
            and ((mask_bool is None) or self._te_supports_mask)
            and ((bias_float is None) or self._te_supports_core_bias)
        )
        if use_te:
            # qkv_format="bshd" → TE에는 [B,S,H,D] 형식으로 전달
            q_te = q_bshd.transpose(1, 2).contiguous()
            k_te = k_bshd.transpose(1, 2).contiguous()
            v_te = v_bshd.transpose(1, 2).contiguous()
            te_kwargs: dict[str, Any] = {}
            if self._te_supports_attention_dropout:
                te_kwargs["attention_dropout"] = dropout_val
            if self._te_supports_is_causal:
                te_kwargs["is_causal"] = bool(is_causal)
            if self._te_supports_training:
                te_kwargs["training"] = training
            if mask_bool is not None and self._te_mask_param:
                te_kwargs[self._te_mask_param] = mask_bool
                if self._te_supports_mask_type and self._te_mask_type_param:
                    te_kwargs[self._te_mask_type_param] = "arbitrary"
            if bias_float is not None and self._te_core_bias_param:
                te_kwargs[self._te_core_bias_param] = bias_float
                if self._te_supports_core_bias_type and self._te_core_bias_type_param:
                    te_kwargs[self._te_core_bias_type_param] = "post_scale_bias"
            try:
                out_te = self._te_attn(q_te, k_te, v_te, **te_kwargs)
            except Exception:
                # TE 커널 예외 발생 시 이후부터는 PyTorch 경로만 사용
                self._force_pt = True
                use_te = False
            else:
                # TE 반환도 [B,S,H,D] 가정 → 다시 [B,H,S,D]로 변환
                return out_te.transpose(1, 2).contiguous()
        sdpa_bias: torch.Tensor | None = None
        if mask_bool is not None:
            finfo = torch.finfo(q_bshd.dtype)
            zero = torch.zeros((), dtype=q_bshd.dtype, device=q_bshd.device)
            neg_inf = torch.full((), finfo.min, dtype=q_bshd.dtype, device=q_bshd.device)
            sdpa_bias = torch.where(mask_bool, neg_inf, zero).expand(B, H, L, S)
        if bias_float is not None:
            base = (
                sdpa_bias
                if sdpa_bias is not None
                else torch.zeros(B, H, L, S, device=q_bshd.device, dtype=q_bshd.dtype)
            )
            sdpa_bias = base + bias_float
        final_mask = attn_mask if (attn_mask is not None and sdpa_bias is None) else sdpa_bias
        sdpa_kwargs = {
            "attn_mask": final_mask,
            "dropout_p": dropout_val,
            "is_causal": bool(is_causal),
        }
        # SDPA는 [B,H,S,D] 기대 → 여기서 변환
        q_bhsd = q_bshd.contiguous()
        k_bhsd = k_bshd.contiguous()
        v_bhsd = v_bshd.contiguous()

        # --- 마스크 정규화: 버전 독립적으로 [B,H,L,S] (또는 브로드캐스트 가능 형태)로 통일 ---
        B, H, _, _ = q_bhsd.shape
        fm = sdpa_kwargs["attn_mask"]
        if fm is not None:
            # 디바이스/ dtype 정합성 및 형태 정규화
            if fm.dtype is torch.bool:
                fm = fm.to(device=q_bhsd.device, non_blocking=True)
            else:
                fm = fm.to(device=q_bhsd.device, dtype=q_bhsd.dtype, non_blocking=True)
            fm, batch_dim, head_count = _flatten_mask_shape(fm)
            if batch_dim not in (1, B):
                raise RuntimeError(
                    f"attn_mask batch dimension {batch_dim} incompatible with batch {B}"
                )
            if head_count not in (1, H):
                raise RuntimeError(
                    f"attn_mask head count {head_count} incompatible with num_heads {H}"
                )
            sdpa_kwargs["attn_mask"] = fm.contiguous()
            # 명시적 마스크/바이어스가 있으면 is_causal은 중복 마스킹을 유발하므로 비활성화
            sdpa_kwargs["is_causal"] = False
        backends = initialize_sdpa_backends()
        sdpa_out: Optional[torch.Tensor] = None
        if backends:
            try:
                from torch.nn.attention import sdpa_kernel
            except Exception:
                backends = []
        if backends:
            with sdpa_kernel(backends):
                sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                    q_bhsd, k_bhsd, v_bhsd, **sdpa_kwargs
                )
        if sdpa_out is None:
            sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, **sdpa_kwargs
            )
        return sdpa_out

    @staticmethod
    def _to_optimal_dtype(tensor: torch.Tensor) -> torch.Tensor:
        device_type = tensor.device.type
        if device_type == "cpu" and tensor.dtype in (torch.float16, torch.bfloat16):
            return tensor.float()
        if device_type == "mps" and tensor.dtype == torch.bfloat16:
            return tensor.to(torch.float16)
        return tensor


class MultiScaleRetentionCompat(nn.Module):
    def __init__(
        self, d_model: int, nhead: int, use_gate: bool = True
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = self.d_model // self.nhead
        self.use_gate = bool(use_gate)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.g_proj = (
            nn.Linear(self.d_model, self.d_model, bias=False)
            if self.use_gate
            else None
        )
        self._beta = nn.Parameter(torch.full((self.nhead,), -0.2))
        self.norm = nn.LayerNorm(self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        *args: Any,
        attn_mask: Optional[torch.Tensor] = None,
        state: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # Handle MPS bf16 instability by running the block in fp16.
        restore_dtype: Optional[torch.dtype] = None
        x_in = x
        if getattr(x.device, "type", "cpu") == "mps" and x.dtype == torch.bfloat16:
            restore_dtype = x.dtype
            x_in = x.to(torch.float16)

        del attn_mask, kwargs
        del state
        batch, seq_len, _ = x_in.shape
        head_dim = self.head_dim
        q = self.q_proj(x_in).view(batch, seq_len, self.nhead, head_dim)
        v = self.v_proj(x_in).view(batch, seq_len, self.nhead, head_dim)
        manual_flops = _retention_manual_flops(
            batch,
            seq_len,
            num_heads=self.nhead,
            head_dim=head_dim,
            use_gate=self.use_gate and self.g_proj is not None,
        )
        lam = (
            torch.sigmoid(self._beta)
            .view(1, self.nhead, 1)
            .to(dtype=v.dtype, device=v.device)
        )
        prev = v[:, 0].clone()
        states = [prev]
        for index in range(1, seq_len):
            prev = lam * prev + v[:, index]
            states.append(prev)
        state_tensor = torch.stack(states, dim=1).contiguous()
        y = (q * state_tensor).contiguous().view(batch, seq_len, self.d_model)
        y = self.norm(y)
        if self.use_gate and self.g_proj is not None:
            gate = torch.nn.functional.silu(self.g_proj(x_in))
            y = y * gate
        if manual_flops > 0.0:
            FLOP_PROFILER.add_manual("Retention", manual_flops)
        out = self.o_proj(y)
        if restore_dtype is not None:
            out = out.to(restore_dtype)
        return out


class MultiScaleRetention(nn.Module):
    def __init__(
        self, d_model: int, nhead: int, use_gate: bool = True
    ) -> None:
        super().__init__()
        self.d_model, self.nhead, self.use_gate = (
            int(d_model),
            int(nhead),
            bool(use_gate),
        )
        self._ts_ok = False
        try:
            from torchscale.component.multiscale_retention import (
                MultiScaleRetention as _TorchScaleMSR,
            )

            self._ts_msr = _TorchScaleMSR(self.d_model, self.nhead)
            # backend-specialization bookkeeping
            self._msr_dev_tag: Optional[str] = None
            self._msr_compiled: bool = False
            self._msr_ipex_infer: bool = False
            self._ts_key_dim = int(
                getattr(self._ts_msr, "key_dim", self.d_model // self.nhead)
            )
            self._ts_ok = True
        except Exception:
            self._ts_ok = False
            self._fallback = MultiScaleRetentionCompat(
                self.d_model, self.nhead, use_gate=self.use_gate
            )
        self._rope_theta = 10000.0
        self._decay_init = 5.0
        self._decay_range = 1.0

    def _build_rel_pos(self, seq_len: Any, device: Any, dtype: Any) -> Any:
        key_dim = int(
            self._ts_key_dim
            if getattr(self, "_ts_key_dim", None) is not None
            else self.d_model // self.nhead
        )
        half = key_dim // 2
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_freq = 1.0 / self._rope_theta ** torch.linspace(
            0, 1, half, device=device, dtype=torch.float32
        )
        freqs = torch.einsum("n,d->nd", positions, inv_freq)
        sin = _duplicate_last_dim(torch.sin(freqs)).to(dtype)[None, None, :, :]
        cos = _duplicate_last_dim(torch.cos(freqs)).to(dtype)[None, None, :, :]
        length = seq_len
        idx_i = torch.arange(length, device=device)
        idx_j = torch.arange(length, device=device)
        diff = (idx_i[:, None] - idx_j[None, :]).to(dtype)
        tril = (idx_i[:, None] >= idx_j[None, :]).to(dtype)
        heads = torch.arange(self.nhead, device=device, dtype=dtype)
        gammas = 1.0 - torch.pow(
            2.0,
            -(
                self._decay_init
                + self._decay_range * (heads / max(self.nhead, 1))
            ),
        )
        gammas = torch.clamp(
            gammas, min=torch.finfo(dtype).tiny, max=1 - 1e-09
        )
        inner_mask = torch.pow(
            gammas.view(1, self.nhead, 1, 1), diff.view(1, 1, length, length)
        ) * tril.view(1, 1, length, length)
        return ((sin, cos), inner_mask)

    def _maybe_specialize(self, x: torch.Tensor) -> None:
        """Pick an optimal backend-specific implementation for TorchScale MSR."""
        if not self._ts_ok or not isinstance(x, torch.Tensor):
            return
        devt = getattr(x.device, "type", "cpu")
        if self._msr_dev_tag == devt:
            return
        self._msr_dev_tag = devt

        # Reset specialization flags
        self._msr_compiled = False
        self._msr_ipex_infer = False

        compile_fn = getattr(torch, "compile", None)

        if devt == "cuda" and callable(compile_fn):
            try:
                self._ts_msr = compile_fn(self._ts_msr, dynamic=True)
                self._msr_compiled = True
            except Exception:
                pass
        elif devt == "xpu":
            try:
                import intel_extension_for_pytorch as ipex  # type: ignore

                if not self.training:
                    target_dtype = (
                        x.dtype
                        if x.dtype in (torch.float32, torch.float16, torch.bfloat16)
                        else torch.float32
                    )
                    self._ts_msr = ipex.optimize(
                        self._ts_msr, dtype=target_dtype
                    )
                    self._msr_ipex_infer = True
                else:
                    if callable(compile_fn):
                        try:
                            self._ts_msr = compile_fn(self._ts_msr, dynamic=True)
                            self._msr_compiled = True
                        except Exception:
                            pass
            except Exception:
                if callable(compile_fn):
                    try:
                        self._ts_msr = compile_fn(self._ts_msr, dynamic=True)
                        self._msr_compiled = True
                    except Exception:
                        pass
        else:
            # CPU / MPS: keep eager (compile on MPS remains unstable)
            pass

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        state: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del attn_mask, kwargs
        self._maybe_specialize(x)
        restore_dtype: Optional[torch.dtype] = None
        x_in = x
        if getattr(x.device, "type", "cpu") == "mps" and x.dtype == torch.bfloat16:
            restore_dtype = x.dtype
            x_in = x.to(torch.float16)
        try:
            batch, seq_len, dim = x_in.shape
        except ValueError:
            manual_flops = 0.0
        else:
            manual_flops = _retention_manual_flops(
                batch,
                seq_len,
                num_heads=self.nhead,
                head_dim=max(1, dim // max(self.nhead, 1)),
                use_gate=self.use_gate,
            )
        if self._ts_ok:
            _, seq_len, _ = x_in.shape
            rel_pos = self._build_rel_pos(seq_len, x_in.device, x_in.dtype)
            out = self._ts_msr(
                x_in, rel_pos, chunkwise_recurrent=False, incremental_state=state
            )
            if manual_flops > 0.0:
                FLOP_PROFILER.add_manual("Retention", manual_flops)
            if restore_dtype is not None:
                out = out.to(restore_dtype)
            return out
        out = self._fallback(x_in, attn_mask=None, state=state)
        if restore_dtype is not None:
            out = out.to(restore_dtype)
        return out


class AdamW:
    @staticmethod
    def float(
        model_or_params: Union[
            nn.Module, Iterable[nn.Parameter], Sequence[Dict[str, Any]]
        ],
        lr: float,
        *args: Any,
        weight_decay: float = 0.0,
        metadata: Optional[MetaData[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> optim.Optimizer:
        params = (
            model_or_params.parameters()
            if hasattr(model_or_params, "parameters")
            else model_or_params
        )
        ref_tensor: Optional[torch.Tensor] = None
        if isinstance(model_or_params, nn.Module):
            ref_tensor = Module._module_reference_tensor(model_or_params)
        dev: torch.device
        if metadata is not None:
            dev = torch.device(metadata.device)
        elif ref_tensor is not None:
            dev = ref_tensor.device
        else:
            dev = get_device()
        meta = AutoCast._ensure_metadata(dev, metadata=metadata)
        dev = torch.device(meta.device)
        if hasattr(dev, "type") and dev.type == "cuda":
            try:
                from transformer_engine.pytorch.optimizers import (
                    FusedAdam as TEFusedAdam,
                )

                opt = TEFusedAdam(params, lr=lr, weight_decay=weight_decay)
                if logger:
                    logger("[OPT] Using FusedAdam (Transformer Engine)")
                return opt
            except Exception as exc:
                if logger:
                    logger(f"[OPT] TE FusedAdam unavailable: {exc}")
        if hasattr(dev, "type") and dev.type == "cuda":
            fp8_allowed = True
            if getattr(meta, "has_scale", False):
                float8_dtypes = AutoCast._float8_dtypes()
                if not any(
                    _supports_scale(dtype, meta, safety_margin=2.0)
                    for dtype in float8_dtypes
                ):
                    fp8_allowed = False
                    if logger:
                        logger(
                            "[OPT] FP8 optimizers disabled: data scale exceeds float8 range"
                        )
            ok, reason = is_float8_supported(dev)
            if fp8_allowed and ok:
                if "TE" in str(reason):
                    try:
                        from transformer_engine.pytorch.optimizers import FusedAdam

                        opt = FusedAdam(params, lr=lr, weight_decay=weight_decay)
                        if logger:
                            logger(
                                f"[OPT] Using FusedAdam (transformer_engine) — {reason}"
                            )
                        return opt
                    except Exception as exc:
                        if logger:
                            logger(
                                f"[OPT] transformer_engine.FusedAdam unavailable: {exc}"
                            )
                if "AO" in str(reason):
                    try:
                        from torchao.optim import AdamWFp8

                        opt = AdamWFp8(params, lr=lr, weight_decay=weight_decay)
                        if logger:
                            logger(f"[OPT] Using AdamW-FP8 (torchao) — {reason}")
                        return opt
                    except Exception as exc:
                        if logger:
                            logger(f"[OPT] torchao.AdamWFp8 unavailable: {exc}")
                elif logger:
                    logger(f"[OPT] FP8 optimizers not supported ({reason}) — fallback")
        flags: Dict[str, bool] = optimal_optimizer_params(
            dev, use_foreach=None, use_fused=False
        )
        opt = optim.AdamW(params, lr=lr, weight_decay=weight_decay, **flags)
        if logger:
            logger(f"[OPT] Using torch.optim.AdamW (flags={flags})")
        return opt

    @staticmethod
    def integer(
        model_or_params: Union[
            nn.Module, Iterable[nn.Parameter], Sequence[Dict[str, Any]]
        ],
        lr: float,
        *args: Any,
        weight_decay: float = 0.0,
        metadata: Optional[MetaData[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> optim.Optimizer:
        params = (
            model_or_params.parameters()
            if hasattr(model_or_params, "parameters")
            else model_or_params
        )
        ref_tensor: Optional[torch.Tensor] = None
        if isinstance(model_or_params, nn.Module):
            ref_tensor = Module._module_reference_tensor(model_or_params)
        if metadata is not None:
            dev = torch.device(metadata.device)
        elif ref_tensor is not None:
            dev = ref_tensor.device
        else:
            dev = get_device()
        meta = AutoCast._ensure_metadata(dev, metadata=metadata)
        dev = torch.device(meta.device)
        if hasattr(dev, "type") and dev.type == "cuda":
            try:
                from transformer_engine.pytorch.optimizers import (
                    FusedAdam as TEFusedAdam,
                )

                opt = TEFusedAdam(params, lr=lr, weight_decay=weight_decay)
                if logger:
                    logger("[OPT] Using FusedAdam (Transformer Engine)")
                return opt
            except Exception as exc:
                if logger:
                    logger(f"[OPT] TE FusedAdam unavailable: {exc}")
        quant_choice: Optional[str] = None
        quant_reason: Optional[str] = None
        if getattr(meta, "has_scale", False):
            if getattr(meta, "scale_is_integral", None) is False:
                if logger:
                    logger("[OPT] Low-bit optimizers disabled: data is not integral")
            else:
                max_abs = float(abs(getattr(meta, "scale_max_abs", 0.0)))
                candidates: List[Tuple[str, Callable[[Optional[torch.device]], Tuple[bool, str]]]] = []
                if max_abs <= 7.0:
                    candidates.append(("int4", is_int4_supported))
                if max_abs <= 127.0:
                    candidates.append(("int8", is_int8_supported))
                if not candidates:
                    if logger:
                        logger(
                            "[OPT] Low-bit optimizers disabled: magnitude %.3f exceeds int8 range",
                            max_abs,
                        )
                else:
                    for name, checker in candidates:
                        ok, reason = checker(dev)
                        if ok:
                            quant_choice = name
                            quant_reason = reason
                            break
                        if logger:
                            logger(
                                f"[OPT] {name.upper()} optimizers not supported ({reason}) — fallback"
                            )
        if quant_choice in {"int8", "int4"}:
            try:
                try:
                    from torchao.optim import AdamW4bit, AdamW8bit
                except ImportError:
                    from torchao.prototype.low_bit_optim import (
                        AdamW4bit,
                        AdamW8bit,
                    )
                if quant_choice == "int8":
                    opt = AdamW8bit(params, lr=lr, weight_decay=weight_decay)
                    if logger:
                        note = f" — {quant_reason}" if quant_reason else ""
                        logger(f"[OPT] TorchAO AdamW8bit{note}")
                    return opt
                opt = AdamW4bit(params, lr=lr, weight_decay=weight_decay)
                if logger:
                    note = f" — {quant_reason}" if quant_reason else ""
                    logger(f"[OPT] TorchAO AdamW4bit{note}")
                return opt
            except Exception as exc:
                if logger:
                    logger(f"[OPT] TorchAO low-bit optimizer unavailable: {exc}")
        flags: Dict[str, bool] = optimal_optimizer_params(
            dev, use_foreach=None, use_fused=False
        )
        opt = optim.AdamW(params, lr=lr, weight_decay=weight_decay, **flags)
        if logger:
            logger(f"[OPT] Using AdamW (flags={flags})")
        return opt


class Module:
    @staticmethod
    def _infer_optimal_dtype(
        device: Optional[Union[torch.device, str]] = None,
        *,
        metadata: Optional[MetaData[Any]] = None,
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
            if _supports_scale(dtype, metadata):
                return dtype
        return (
            torch.float64
            if _supports_scale(torch.float64, metadata)
            else candidates[-1]
        )

    @staticmethod
    def _module_reference_tensor(module: nn.Module) -> Optional[torch.Tensor]:
        with contextlib.suppress(StopIteration):
            return next(module.parameters())
        with contextlib.suppress(StopIteration):
            return next(module.buffers())
        return None

    @staticmethod
    def _metadata_for(
        model: nn.Module, metadata: Optional[MetaData[Any]] = None
    ) -> MetaData[Any]:
        AutoCast.configure(model, metadata=metadata)
        meta = AutoCast._metadata
        if meta is None:
            ref = Module._module_reference_tensor(model)
            dev = ref.device if isinstance(ref, torch.Tensor) else get_device()
            meta = MetaData.for_device(dev)
            AutoCast.configure(model, metadata=meta)
        return meta

    @staticmethod
    def _align_module_like(
        src: nn.Module,
        dst: nn.Module,
        params_dtype: Optional[torch.dtype],
    ) -> None:
        ref = Module._module_reference_tensor(src)
        if ref is not None:
            with contextlib.suppress(Exception):
                dst.to(device=ref.device)
        if params_dtype is not None:
            with contextlib.suppress(Exception):
                dst.to(dtype=params_dtype)

    @staticmethod
    def _copy_state(
        src: nn.Module, dst: nn.Module, params_dtype: Optional[torch.dtype]
    ) -> None:
        try:
            state = src.state_dict()
        except Exception:
            return
        ref = Module._module_reference_tensor(dst)
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
    def _make_te_linear(
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
        Module._align_module_like(module, replacement, params_dtype)
        Module._copy_state(module, replacement, params_dtype)
        return replacement

    @staticmethod
    def _make_te_layer_norm(
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
        Module._align_module_like(module, replacement, params_dtype)
        if module.elementwise_affine:
            Module._copy_state(module, replacement, params_dtype)
        return replacement

    @staticmethod
    def _make_te_rms_norm(
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
        Module._align_module_like(module, replacement, params_dtype)
        Module._copy_state(module, replacement, params_dtype)
        return replacement

    @staticmethod
    def _fuse_sequential_to_te(
        model: nn.Module, *, params_dtype: Optional[torch.dtype]
    ) -> Tuple[nn.Module, int]:
        try:
            importlib.import_module("transformer_engine.pytorch")
        except Exception:
            return (model, 0)
        return (model, 0)

    @staticmethod
    def _apply_te_module(
        model: nn.Module,
        *,
        apply_te_linear: bool,
        apply_te_layer_norm: bool,
        apply_te_rms_norm: bool,
        filter_linear: Optional[Callable[[nn.Linear, str], bool]],
        params_dtype: Optional[torch.dtype],
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
                        replacement = Module._make_te_linear(
                            child, params_dtype, te
                        )
                elif apply_te_layer_norm and isinstance(child, nn.LayerNorm):
                    replacement = Module._make_te_layer_norm(
                        child, params_dtype, te
                    )
                else:
                    rms_cls = getattr(torch.nn, "RMSNorm", None)
                    if (
                        apply_te_rms_norm
                        and rms_cls is not None
                        and isinstance(child, rms_cls)
                    ):
                        replacement = Module._make_te_rms_norm(
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
    def _apply_te_attention(
        model: nn.Module, *, params_dtype: Optional[torch.dtype]
    ) -> Tuple[nn.Module, int]:
        swapped = 0
        for module in model.modules():
            if isinstance(module, DotProductAttention) and getattr(
                module, "_te_ok", False
            ):
                if not getattr(module, "te_first", False):
                    module.te_first = True
                swapped += 1
        return (model, swapped)

    @staticmethod
    def use_te_module(
        model: nn.Module,
        device: Optional[Union[torch.device, str]] = None,
        *,
        metadata: Optional[MetaData[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
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
        params_dtype = Module._infer_optimal_dtype(dev, metadata=metadata)
        model, n_fused = Module._fuse_sequential_to_te(
            model, params_dtype=params_dtype
        )
        model, n_basic = Module._apply_te_module(
            model,
            apply_te_linear=True,
            apply_te_layer_norm=True,
            apply_te_rms_norm=True,
            filter_linear=None,
            params_dtype=params_dtype,
        )
        try:
            model, attn_swapped = Module._apply_te_attention(
                model, params_dtype=params_dtype
            )
        except Exception:
            attn_swapped = 0
        n_total = (n_fused or 0) + (n_basic or 0) + (attn_swapped or 0)
        if logger:
            logger(
                f"[TE] swapped {n_total} modules (fused:{n_fused}, basic:{n_basic}, attn:{attn_swapped}); params_dtype={str(params_dtype).split('.')[-1]}, fp8={('on' if fp8_ok else 'off')} ({(why if fp8_ok else '')}), backend={te_backend}"
            )
        return (
            model,
            n_total > 0,
            f"TE applied (swapped {n_total}, dtype={params_dtype}, fp8={('on' if fp8_ok else 'off')}, backend={te_backend})",
        )

    @staticmethod
    def _try_enable_te_training(
        model: nn.Module,
        params_dtype: torch.dtype,
        logger: Optional[Callable[[str], None]],
    ) -> Tuple[nn.Module, bool, str]:
        try:
            swapped_model, n = Module._apply_te_module(
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
    def _try_enable_ao_training(
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
    def _try_enable_te_inference_swap(
        model: nn.Module,
        params_dtype: torch.dtype,
        logger: Optional[Callable[[str], None]],
    ) -> Tuple[nn.Module, bool, str]:
        try:
            swapped, n = Module._apply_te_module(
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
    def _try_use_existing_te(
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
    def _try_enable_ao_inference(
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
        metadata: Optional[MetaData[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[nn.Module, bool, str]:
        meta = Module._metadata_for(model, metadata)
        device = torch.device(meta.device)
        ok, reason = is_float8_supported(device)
        if not ok:
            AutoCast.configure(model, metadata=meta)
            return (model, False, reason)
        if getattr(meta, "has_scale", False):
            float8_dtypes = AutoCast._float8_dtypes()
            if not any(
                _supports_scale(dtype, meta, safety_margin=2.0)
                for dtype in float8_dtypes
            ):
                if logger:
                    logger(
                        "[FP8] training disabled: data scale exceeds float8 range"
                    )
                AutoCast.configure(model, metadata=meta)
                return (model, False, "data scale")
        params_dtype = Module._infer_optimal_dtype(device, metadata=meta)

        for backend in ("te", "torchao"):
            if backend == "te":
                m2, ok2, why = Module._try_enable_te_training(
                    model, params_dtype, logger
                )
            else:
                m2, ok2, why = Module._try_enable_ao_training(model, logger)
            if ok2:
                if logger:
                    logger(f"[FP8] training enabled via {why} ({reason})")
                AutoCast.configure(m2, metadata=meta)
                return (m2, True, why)
            elif logger:
                logger(f"[FP8] {backend} path skipped: {why}")
        AutoCast.configure(model, metadata=meta)
        return (model, False, "No usable FP8 backend")

    @staticmethod
    def enable_float8_prediction(
        model: nn.Module,
        metadata: Optional[MetaData[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[nn.Module, bool, str]:
        meta = Module._metadata_for(model, metadata)
        device = torch.device(meta.device)
        ok, reason = is_float8_supported(device)
        if not ok:
            AutoCast.configure(model, metadata=meta)
            return (model, False, reason)
        if getattr(meta, "has_scale", False):
            float8_dtypes = AutoCast._float8_dtypes()
            if not any(
                _supports_scale(dtype, meta, safety_margin=2.0)
                for dtype in float8_dtypes
            ):
                if logger:
                    logger(
                        "[FP8] inference disabled: data scale exceeds float8 range"
                    )
                AutoCast.configure(model, metadata=meta)
                return (model, False, "data scale")
        params_dtype = Module._infer_optimal_dtype(device, metadata=meta)
        dynamic_activations = not (
            getattr(meta, "has_scale", False)
            and getattr(meta, "scale_is_integral", None) is True
        )
        order = ("te_swap", "te_present", "ao")
        for step in order:
            if step == "te_swap":
                m2, ok2, why = Module._try_enable_te_inference_swap(
                    model, params_dtype, logger
                )
            elif step == "te_present":
                m2, ok2, why = Module._try_use_existing_te(model, logger)
            else:
                m2, ok2, why = Module._try_enable_ao_inference(
                    model, dynamic_activations, logger
                )
            if ok2:
                if logger:
                    logger(f"[FP8] inference enabled via {why} ({reason})")
                AutoCast.configure(m2, metadata=meta)
                return (m2, True, why)
            elif logger:
                logger(f"[FP8] {step} skipped: {why}")
        AutoCast.configure(model, metadata=meta)
        return (model, False, "No usable FP8 backend")

    @staticmethod
    def enable_int8_training(
        model: nn.Module,
        metadata: Optional[MetaData[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[nn.Module, bool, str]:
        if quantize_ is None:
            msg = "torchao.quantization not installed (INT8/QAT disabled)"
            if logger:
                logger(f"[INT8] {msg}")
            return (model, False, msg)
        meta = Module._metadata_for(model, metadata)
        device = torch.device(meta.device)
        with contextlib.suppress(Exception):
            model.to(device)
        dynamic_activations = not (
            getattr(meta, "has_scale", False)
            and getattr(meta, "scale_is_integral", None) is True
        )
        group_size = 128
        try:
            base_cfg = QAT.initialize(
                model,
                mode="qat-int8",
                dynamic_activations=dynamic_activations,
                group_size=group_size,
                logger=logger,
            )
            setattr(model, "__int8_training_qat__", True)
            if logger:
                logger(
                    f"[INT8][QAT] prepared with base {base_cfg.__class__.__name__}"
                )
            AutoCast.configure(model, metadata=meta)
            return (model, True, "QAT-prepare")
        except Exception as exc:
            if logger:
                logger(f"[INT8][QAT] prepare failed: {exc}")
            last_err: Exception = exc
        try:
            m2, ok, why = ptq(
                model,
                mode="int8",
                dynamic_activations=dynamic_activations,
                group_size=group_size,
                logger=logger,
            )
            if ok:
                setattr(m2, "__int8_training_ptq__", True)
                AutoCast.configure(m2, metadata=meta)
                return (m2, True, f"PTQ({why})")
            AutoCast.configure(model, metadata=meta)
            return (model, False, f"PTQ failed: {why}")
        except Exception as exc:
            AutoCast.configure(model, metadata=meta)
            return (model, False, f"INT8 training path unavailable: {exc or last_err}")

    @staticmethod
    def enable_int8_prediction(
        model: nn.Module,
        metadata: Optional[MetaData[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[nn.Module, bool, str]:
        if quantize_ is None:
            msg = "torchao.quantization not installed (INT8 disabled)"
            if logger:
                logger(f"[INT8] {msg}")
            return (model, False, msg)
        meta = Module._metadata_for(model, metadata)
        device = torch.device(meta.device)
        with contextlib.suppress(Exception):
            model.to(device)
        dynamic_activations = not (
            getattr(meta, "has_scale", False)
            and getattr(meta, "scale_is_integral", None) is True
        )
        try:
            if dynamic_activations:
                cfg = Int8DynamicActivationInt8WeightConfig()
            else:
                cfg = Int8WeightOnlyConfig()
            quantize_(model, cfg)
            setattr(model, "__int8_inference_ao__", True)
            if logger:
                logger(f"[INT8][AO] applied {cfg.__class__.__name__}")
            AutoCast.configure(model, metadata=meta)
            return (model, True, "torchao")
        except Exception as e:
            AutoCast.configure(model, metadata=meta)
            return (model, False, f"AO failed: {e}")

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
    default_module = f"{root_pkg}.utils.optimization"
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


# ======================================================================================
# MultiHeadAttention: TE fused MHA 선호, 아니면 torch.nn.MultiheadAttention 폴백
# ======================================================================================
_HAS_TE: bool
if os.environ.get("STF_DISABLE_TE", "") == "1":
    te = None  # type: ignore
    _HAS_TE = False
else:
    _should_import_te = True
    if not torch.cuda.is_available():
        _should_import_te = False
    else:
        try:
            device = get_device()
        except Exception:
            device = torch.device("cuda", 0)
        if getattr(device, "type", "cpu") != "cuda":
            _should_import_te = False
        else:
            try:
                index = (
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                )
            except Exception:
                index = 0
            try:
                props = torch.cuda.get_device_properties(index)
                major = int(getattr(props, "major", 0))
            except Exception:
                try:
                    major, _ = torch.cuda.get_device_capability(index)
                except Exception:
                    major = 0
            min_major_env = os.environ.get("STF_TE_MIN_CC", "")
            try:
                min_major = int(min_major_env) if min_major_env else 8
            except Exception:
                min_major = 8
            if min_major >= 10:
                min_major //= 10
            if major < min_major:
                _should_import_te = False
    if _should_import_te:
        try:
            # NVIDIA Transformer Engine (optional)
            import transformer_engine.pytorch as te  # type: ignore

            _HAS_TE = True
        except Exception:  # pragma: no cover
            te = None  # type: ignore
            _HAS_TE = False
    else:
        te = None  # type: ignore
        _HAS_TE = False


def _te_fused_mha_is_preferred(min_cc: Tuple[int, int] = (8, 0)) -> bool:
    """
    TE fused MHA를 사용할지 판단: 설치 여부 + CUDA CC 기준.
    기본 기준: sm80(A100/RTX30 이상) 권장.
    """

    if not _HAS_TE:
        return False
    if not _te_supported_on_device():
        return False
    device = get_device()
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    cc = cuda_compute_capability(device)
    return cc >= min_cc


class MultiHeadAttentionCompat(nn.Module):
    """torch.nn.MultiheadAttention thin wrapper (batch_first=True)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        bias: bool = True,
        dropout: float = 0.0,
        batch_first: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.batch_first = batch_first
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        is_causal: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Call into PyTorch MHA, tolerating the optional is_causal kwarg."""

        kwargs = dict(key_padding_mask=key_padding_mask, need_weights=need_weights)
        try:
            if is_causal is not None:
                return self.mha(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                    **kwargs,
                )
        except TypeError:
            pass
        return self.mha(
            query,
            key,
            value,
            attn_mask=attn_mask,
            **kwargs,
        )


class MultiHeadAttentionNvidia(nn.Module):
    """
    NVIDIA Transformer Engine MHA 래퍼 (버전 차이를 흡수하도록 시그니처를 다변화 시도).
    생성/forward 실패 시 안전하게 torch MHA로 폴백합니다.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        bias: bool = True,
        dropout: float = 0.0,
        batch_first: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.batch_first = batch_first
        self.num_heads = int(num_heads)
        # PT 폴백을 미리 준비 (런타임 안전)
        self._fallback = MultiHeadAttentionCompat(
            embed_dim,
            num_heads,
            bias=bias,
            dropout=dropout,
            batch_first=batch_first,
            **kwargs,
        )
        self._te_mha = self._build_te_mha(embed_dim, num_heads, dropout, kwargs)
        self._force_pt: bool = self._te_mha is None
        if self._force_pt:
            warnings.warn(
                "Transformer Engine MHA 사용 불가: torch.nn.MultiheadAttention으로 폴백합니다.",
                RuntimeWarning,
            )

    @staticmethod
    def _build_te_mha(embed_dim: int, num_heads: int, dropout: float, kwargs: dict):
        if not _HAS_TE:
            return None
        if not _te_supported_on_device():
            return None
        candidates = []
        for name in ("MultiHeadAttention", "MultiheadAttention"):
            if hasattr(te, name):
                candidates.append(getattr(te, name))
        for cls in candidates:
            ctor_variants = (
                dict(
                    hidden_size=embed_dim,
                    num_attention_heads=num_heads,
                    attention_dropout=dropout,
                ),
                dict(hidden_size=embed_dim, num_heads=num_heads, attention_dropout=dropout),
                dict(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout),
            )
            for ckw in ctor_variants:
                try:
                    return cls(**{**ckw, **kwargs})
                except TypeError:
                    continue
                except Exception:
                    continue
        return None

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        is_causal: Optional[bool] = None,
    ):
        if self._force_pt or (self._te_mha is None):
            return self._fallback(  # type: ignore[operator]
                query,
                key,
                value,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                is_causal=is_causal,
            )
        te_attn_mask = attn_mask
        if attn_mask is not None or key_padding_mask is not None:
            te_attn_mask = _adapt_mask_for_te(
                query,
                key,
                attn_mask,
                key_padding_mask,
                num_heads=self.num_heads,
                batch_first=self.batch_first,
            )
            if te_attn_mask is None:
                # 변환 실패 → TE 재시도 금지하고 PT로 영구 폴백
                self._force_pt = True
                return self._fallback(  # type: ignore[operator]
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    is_causal=is_causal,
                )
        # ===== batch_first=False 처리: TE는 [B,S,D] 경로를 선호하므로 재정렬 =====
        bf = bool(self.batch_first)
        _q, _k, _v = query, key, value
        if not bf:
            _q, _k, _v = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        # ======================================================================
        for variant in (
            dict(
                query=_q,
                key=_k,
                value=_v,
                attn_mask=te_attn_mask,
                need_weights=need_weights,
                is_causal=is_causal,
            ),
            dict(query=_q, attn_mask=te_attn_mask, need_weights=need_weights),
            # 일부 TE 구현은 'attention_mask' 이름을 받음
            dict(query=_q, key=_k, value=_v, attention_mask=te_attn_mask, need_weights=need_weights),
        ):
            try:
                out = self._te_mha(**variant)
                if isinstance(out, tuple) and len(out) >= 1:
                    y, w = out[0], (out[1] if need_weights and len(out) > 1 else None)
                else:
                    y, w = out, None  # type: ignore[assignment]
                # TE는 [B,S,D]를 반환한다고 가정, 필요시 복원
                if not bf and isinstance(y, torch.Tensor) and y.dim() >= 2:
                    y = y.transpose(0, 1)
                return y, w
            except TypeError:
                continue
            except Exception:
                continue
        # TE가 모든 시도를 거부 → PT로 고정 전환 후 폴백
        self._force_pt = True
        return self._fallback(  # type: ignore[operator]
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            is_causal=is_causal,
        )


class MultiHeadAttention(nn.Module):
    """
    프로젝트 표준 MHA: TE fused MHA를 선호하고 불가 시 torch로 폴백.

    속성:
        backend: "te" | "torch"
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        bias: bool = True,
        dropout: float = 0.0,
        batch_first: bool = True,
        prefer_te_min_cc: Tuple[int, int] = (8, 0),
        **kwargs: object,
    ) -> None:
        super().__init__()
        if _te_fused_mha_is_preferred(prefer_te_min_cc):
            try:
                impl = MultiHeadAttentionNvidia(
                    embed_dim,
                    num_heads,
                    bias=bias,
                    dropout=dropout,
                    batch_first=batch_first,
                    **kwargs,
                )
                if isinstance(impl, MultiHeadAttentionNvidia) and impl._te_mha is not None:
                    self.impl = impl
                    self._backend = "te"
                else:
                    self.impl = MultiHeadAttentionCompat(
                        embed_dim,
                        num_heads,
                        bias=bias,
                        dropout=dropout,
                        batch_first=batch_first,
                        **kwargs,
                    )
                    self._backend = "torch"
            except Exception:
                self.impl = MultiHeadAttentionCompat(
                    embed_dim,
                    num_heads,
                    bias=bias,
                    dropout=dropout,
                    batch_first=batch_first,
                    **kwargs,
                )
                self._backend = "torch"
        else:
            self.impl = MultiHeadAttentionCompat(
                embed_dim,
                num_heads,
                bias=bias,
                dropout=dropout,
                batch_first=batch_first,
                **kwargs,
            )
            self._backend = "torch"
        if isinstance(self.impl, MultiHeadAttentionNvidia):
            self.impl._fallback = MultiHeadAttentionCompat(
                embed_dim,
                num_heads,
                bias=bias,
                dropout=dropout,
                batch_first=batch_first,
                **kwargs,
            )

    @property
    def backend(self) -> str:
        return self._backend

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        is_causal: Optional[bool] = None,
    ):
        return self.impl(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            is_causal=is_causal,
        )
