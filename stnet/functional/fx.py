import contextlib
import importlib
import logging
import math
from contextlib import AbstractContextManager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from ..backend.compat import patch_torch
from ..data.stats import MetaData
from ..model.kernels import DotProductAttention
from ..backend.environment import (
    get_device,
    is_cpu_bf16_supported,
    is_cuda_bf16_supported,
    is_float8_supported,
    is_int8_supported,
)

patch_torch()
__all__ = [
    "reshape_for_heads",
    "is_transformer_engine_enabled",
    "AutoCast",
    "_import_callable",
    "LayerReplacement",
    "Quantization",
    "Gradient",
    "_supports_scale",
]

def reshape_for_heads(
    tensor: torch.Tensor, batch_size: int, head_count: int, head_dim: int
) -> torch.Tensor:
    """Reshape a projection to multi-head layout."""

    return tensor.view(batch_size, -1, head_count, head_dim).transpose(1, 2)


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


class Gradient:
    @staticmethod
    def inference(model: torch.nn.Module) -> AbstractContextManager[None]:
        if (
            is_transformer_engine_enabled(model)
            or _is_inference_compiled(model)
            or _is_aot_autograd_enabled(model)
        ):
            return torch.no_grad()
        return torch.inference_mode()

    @staticmethod
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


def _import_callable(spec: str) -> Callable:
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("Empty spec for callable import")
    raw = spec.strip()
    root_pkg = __package__.split(".", 1)[0] if __package__ else "stnet"
    default_module = f"{root_pkg}.functional.fx"
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


class Quantization:
    """Utility helpers to manage PTQ/QAT backends for INT8 workflows."""

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
    def is_available(cls) -> bool:
        return callable(cls.quantize)

    @classmethod
    def is_qat_available(cls) -> bool:
        initialize = getattr(cls.QAT, "initialize", None)
        return callable(initialize)

    @classmethod
    def is_ptq_available(cls) -> bool:
        return callable(cls.ptq) and cls.ptq is not _ptq_unavailable

    @classmethod
    def prepare_qat(
        cls,
        model: nn.Module,
        *,
        dynamic_activations: bool,
        group_size: int = 128,
        logger: Optional[Callable[[str], None]] = None,
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
    def apply_ptq(
        cls,
        model: nn.Module,
        *,
        dynamic_activations: bool,
        group_size: int = 128,
        logger: Optional[Callable[[str], None]] = None,
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
    def apply_ao(
        cls,
        model: nn.Module,
        *,
        dynamic_activations: bool,
        logger: Optional[Callable[[str], None]] = None,
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
    def enable_training(
        cls,
        model: nn.Module,
        *,
        dynamic_activations: bool,
        group_size: int = 128,
        logger: Optional[Callable[[str], None]] = None,
    ) -> tuple[nn.Module, bool, str]:
        if not cls.is_available():
            msg = "torchao.quantization not installed (INT8/QAT disabled)"
            if logger:
                logger(f"[INT8] {msg}")
            return (model, False, msg)
        last_err: Optional[Exception] = None
        if cls.is_qat_available():
            try:
                base_cfg = cls.prepare_qat(
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
            m2, ok, why = cls.apply_ptq(
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

    @classmethod
    def enable_prediction(
        cls,
        model: nn.Module,
        *,
        dynamic_activations: bool,
        logger: Optional[Callable[[str], None]] = None,
    ) -> tuple[nn.Module, bool, str]:
        if not cls.is_available():
            msg = "torchao.quantization not installed (INT8 disabled)"
            if logger:
                logger(f"[INT8] {msg}")
            return (model, False, msg)
        return cls.apply_ao(model, dynamic_activations=dynamic_activations, logger=logger)

class LayerReplacement:
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
            ref = LayerReplacement._module_reference_tensor(model)
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
        ref = LayerReplacement._module_reference_tensor(src)
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
        ref = LayerReplacement._module_reference_tensor(dst)
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
        LayerReplacement._align_module_like(module, replacement, params_dtype)
        LayerReplacement._copy_state(module, replacement, params_dtype)
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
        LayerReplacement._align_module_like(module, replacement, params_dtype)
        if module.elementwise_affine:
            LayerReplacement._copy_state(module, replacement, params_dtype)
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
        LayerReplacement._align_module_like(module, replacement, params_dtype)
        LayerReplacement._copy_state(module, replacement, params_dtype)
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
                        replacement = LayerReplacement._make_te_linear(
                            child, params_dtype, te
                        )
                elif apply_te_layer_norm and isinstance(child, nn.LayerNorm):
                    replacement = LayerReplacement._make_te_layer_norm(
                        child, params_dtype, te
                    )
                else:
                    rms_cls = getattr(torch.nn, "RMSNorm", None)
                    if (
                        apply_te_rms_norm
                        and rms_cls is not None
                        and isinstance(child, rms_cls)
                    ):
                        replacement = LayerReplacement._make_te_rms_norm(
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
        params_dtype = LayerReplacement._infer_optimal_dtype(dev, metadata=metadata)
        model, n_fused = LayerReplacement._fuse_sequential_to_te(
            model, params_dtype=params_dtype
        )
        model, n_basic = LayerReplacement._apply_te_module(
            model,
            apply_te_linear=True,
            apply_te_layer_norm=True,
            apply_te_rms_norm=True,
            filter_linear=None,
            params_dtype=params_dtype,
        )
        try:
            model, attn_swapped = LayerReplacement._apply_te_attention(
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
            swapped_model, n = LayerReplacement._apply_te_module(
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
            swapped, n = LayerReplacement._apply_te_module(
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
        meta = LayerReplacement._metadata_for(model, metadata)
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
        params_dtype = LayerReplacement._infer_optimal_dtype(device, metadata=meta)

        for backend in ("te", "torchao"):
            if backend == "te":
                m2, ok2, why = LayerReplacement._try_enable_te_training(
                    model, params_dtype, logger
                )
            else:
                m2, ok2, why = LayerReplacement._try_enable_ao_training(model, logger)
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
        meta = LayerReplacement._metadata_for(model, metadata)
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
        params_dtype = LayerReplacement._infer_optimal_dtype(device, metadata=meta)
        dynamic_activations = not (
            getattr(meta, "has_scale", False)
            and getattr(meta, "scale_is_integral", None) is True
        )
        order = ("te_swap", "te_present", "ao")
        for step in order:
            if step == "te_swap":
                m2, ok2, why = LayerReplacement._try_enable_te_inference_swap(
                    model, params_dtype, logger
                )
            elif step == "te_present":
                m2, ok2, why = LayerReplacement._try_use_existing_te(model, logger)
            else:
                m2, ok2, why = LayerReplacement._try_enable_ao_inference(
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
        meta = LayerReplacement._metadata_for(model, metadata)
        device = torch.device(meta.device)
        with contextlib.suppress(Exception):
            model.to(device)
        dynamic_activations = not (
            getattr(meta, "has_scale", False)
            and getattr(meta, "scale_is_integral", None) is True
        )
        group_size = 128
        m2, ok, why = Quantization.enable_training(
            model,
            dynamic_activations=dynamic_activations,
            group_size=group_size,
            logger=logger,
        )
        AutoCast.configure(m2 if ok else model, metadata=meta)
        return (m2, ok, why)


    @staticmethod
    def enable_int8_prediction(
        model: nn.Module,
        metadata: Optional[MetaData[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[nn.Module, bool, str]:
        meta = LayerReplacement._metadata_for(model, metadata)
        device = torch.device(meta.device)
        with contextlib.suppress(Exception):
            model.to(device)
        dynamic_activations = not (
            getattr(meta, "has_scale", False)
            and getattr(meta, "scale_is_integral", None) is True
        )
        m2, ok, why = Quantization.enable_prediction(
            model, dynamic_activations=dynamic_activations, logger=logger
        )
        AutoCast.configure(m2 if ok else model, metadata=meta)
        return (m2, ok, why)


