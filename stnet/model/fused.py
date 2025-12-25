# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import importlib
import json
import logging
import math
import threading
from collections import OrderedDict
from contextlib import AbstractContextManager
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import torch
from torch import nn

from ..backend.compat import patch_torch
from ..backend.system import _log_debug, _log_info, get_device, process_cpu_count
from ..data.pipeline import Dataset, default_underflow_action, normalize_underflow_action
from ..backend.casting import env_first, env_first_int

patch_torch()


class LossFn(Protocol):
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ...


LossCallable = LossFn
_LOGGER = logging.getLogger(__name__)


# Deduplicate AMP negotiation logs (best-effort, bounded).
_NEGO_LOGGED_KEYS: "OrderedDict[object, None]" = OrderedDict()
_NEGO_LOGGED_MAX: int = 256
_NEGO_LOGGED_LOCK = threading.Lock()

# torch._inductor config is process-global; guard concurrent mutation.
_INDUCTOR_CONFIG_LOCK = threading.Lock()


def _invalidate_model_introspection_caches(model: Optional[nn.Module]) -> None:
    """Invalidate cached runtime-introspection hints on `model`.

    We cache a few expensive checks (TorchScript/compile/AOT/TE probing).
    Any time we swap modules (TE, QAT wrappers) or quantize weights, these
    hints can become stale.
    """
    if not isinstance(model, nn.Module):
        return
    for attr in (
        "__stnet_cached_is_compiled_for_inference__",
        "__stnet_cached_is_aot_autograd_enabled__",
        "__stnet_cached_is_nvidia_te_available__",
    ):
        with contextlib.suppress(Exception):
            delattr(model, attr)


def _log_negotiate_once(
    logger: Optional[logging.Logger],
    key: object,
    payload: Dict[str, Any],
    *,
    level: str = "debug",
) -> None:
    """Log a structured negotiation decision once per key.

    This avoids spam when negotiation happens repeatedly across steps.

    Thread-safe and best-effort: in free-threaded/no-GIL Python, we must
    guard the shared dedupe cache explicitly.
    """
    lg = logger if logger is not None else _LOGGER
    lvl = logging.INFO if str(level).lower() == "info" else logging.DEBUG
    try:
        if not lg.isEnabledFor(lvl):
            return
    except Exception:
        # If logger capability probing fails, continue best-effort.
        pass

    if key is None:
        key = (payload.get("context"), payload.get("device"), payload.get("selected"))

    with _NEGO_LOGGED_LOCK:
        if key in _NEGO_LOGGED_KEYS:
            try:
                _NEGO_LOGGED_KEYS.move_to_end(key)
            except Exception:
                pass
            return
        _NEGO_LOGGED_KEYS[key] = None
        try:
            while len(_NEGO_LOGGED_KEYS) > int(_NEGO_LOGGED_MAX):
                _NEGO_LOGGED_KEYS.popitem(last=False)
        except Exception:
            pass
    try:
        msg = "[AMP][NEGOTIATE] " + json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        msg = f"[AMP][NEGOTIATE] {payload}"
    try:
        if lvl == logging.INFO:
            lg.info(msg)
        else:
            lg.debug(msg)
    except Exception:
        # Best-effort logging: never fail negotiation because logging broke.
        pass


def _is_ptq_unavailable(
    model: nn.Module, *args: Any, **kwargs: Any
) -> tuple[nn.Module, bool, str]:
    return (model, False, "PTQ backend unavailable")


def _is_compiled_for_inference(model: torch.nn.Module) -> bool:
    cached = getattr(model, "__stnet_cached_is_compiled_for_inference__", None)
    if isinstance(cached, bool):
        return cached

    compile_attrs = (
        "_is_compiled_for_inference",
        "__is_compiled_for_inference__",
        "__compiled_for_serving__",
        "__serving_compiled__",
        "_is_serialized_for_serving",
    )
    if any(bool(getattr(model, attr, False)) for attr in compile_attrs):
        try:
            setattr(model, "__stnet_cached_is_compiled_for_inference__", True)
        except Exception:
            pass
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
        try:
            setattr(model, "__stnet_cached_is_compiled_for_inference__", True)
        except Exception:
            pass
        return True

    try:
        modules = tuple(model.modules())
    except (RuntimeError, AttributeError, TypeError):
        modules = ()

    for module in modules:
        if module is model:
            continue
        if any(bool(getattr(module, attr, False)) for attr in compile_attrs):
            return True
        if any(isinstance(module, typ) for typ in script_like_types):
            return True
    try:
        setattr(model, "__stnet_cached_is_compiled_for_inference__", False)
    except Exception:
        pass
    return False


def _is_aot_autograd_enabled(model: torch.nn.Module) -> bool:
    cached = getattr(model, "__stnet_cached_is_aot_autograd_enabled__", None)
    if isinstance(cached, bool):
        return cached

    indicator_attrs = (
        "_aot_autograd_graph",
        "_aot_autograd_cache",
        "_aot_compiled_autograd",
        "_aot_autograd_traced_module",
        "__aot_autograd__",
        "__compiled_with_aot_autograd__",
    )
    if any(getattr(model, attr, None) for attr in indicator_attrs):
        try:
            setattr(model, "__stnet_cached_is_aot_autograd_enabled__", True)
        except Exception:
            pass
        return True

    try:
        modules = tuple(model.modules())
    except (RuntimeError, AttributeError, TypeError):
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
    try:
        setattr(model, "__stnet_cached_is_aot_autograd_enabled__", False)
    except Exception:
        pass
    return False


def _dtype_short(dtype: Any) -> str:
    if isinstance(dtype, torch.dtype):
        return str(dtype).split(".")[-1]
    return str(dtype)


def _meta_scale_summary(meta: Optional[Dataset[Any]]) -> Dict[str, Any]:
    if meta is None:
        return {}

    def _safe(x: Any) -> Any:
        if x is None:
            return None
        if isinstance(x, (bool, int, str)):
            return x
        if isinstance(x, float):
            return x
        try:
            return float(x)
        except Exception:
            try:
                return str(x)
            except Exception:
                return None

    keys = (
        "has_scale",
        "has_nonfinite",
        "scale_max_abs",
        "scale_min_positive",
        "scale_is_integral",
        "scale_min_value",
        "scale_max_value",
        "underflow_action",
        "int_quant_bits",
    )
    out: Dict[str, Any] = {}
    for k in keys:
        out[k] = _safe(getattr(meta, k, None))
    return out


def _scale_safety_check(
    dtype: torch.dtype,
    meta: Optional[Dataset[Any]],
    *args: Any,
    safety_margin: float = 8.0,
    underflow_action: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[bool, str]:
    """Return (ok, reason) for whether dtype can represent dataset scale.

    This mirrors :func:`is_scale_safe` semantics but adds a human-readable failure reason.
    """
    if meta is None or not getattr(meta, "has_scale", False):
        return True, "no-scale"
    if not isinstance(dtype, torch.dtype):
        return False, "not-dtype"
    if bool(getattr(meta, "has_nonfinite", False)):
        return False, "nonfinite-data"

    max_abs = getattr(meta, "scale_max_abs", None)
    if max_abs is None:
        return True, "no-max-abs"
    try:
        max_abs_f = float(abs(max_abs))
    except Exception:
        return False, "max-abs-not-float"
    if not math.isfinite(max_abs_f):
        return False, "max-abs-nonfinite"

    action = normalize_underflow_action(
        underflow_action
        if underflow_action is not None
        else getattr(meta, "underflow_action", None),
        default=default_underflow_action(),
    )

    if getattr(dtype, "is_complex", False):
        base_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        ok, why = _scale_safety_check(
            base_dtype,
            meta,
            safety_margin=safety_margin,
            underflow_action=action,
        )
        return (ok, f"complex-base:{why}" if ok else f"complex-base-unsafe:{why}")

    if getattr(dtype, "is_floating_point", False):
        info = torch.finfo(dtype)
        overflow_limit = float(info.max) / max(1.0, float(safety_margin))
        if max_abs_f > overflow_limit:
            return (
                False,
                f"overflow(max_abs={max_abs_f:.6g},limit={overflow_limit:.6g})",
            )
        min_pos = getattr(meta, "scale_min_positive", None)
        if action == "forbid" and min_pos is not None:
            try:
                min_pos_f = float(min_pos)
            except Exception:
                return False, "min-pos-not-float"
            if math.isfinite(min_pos_f) and min_pos_f > 0.0:
                underflow_limit = float(info.tiny) * max(1.0, float(safety_margin))
                if min_pos_f < underflow_limit:
                    return (
                        False,
                        f"underflow(min_pos={min_pos_f:.6g},limit={underflow_limit:.6g})",
                    )
        return True, "ok"

    if dtype == torch.bool:
        is_integral = getattr(meta, "scale_is_integral", None)
        if is_integral is False:
            return False, "bool-nonintegral-data"
        if max_abs_f <= 1.0:
            return True, "ok"
        return False, f"bool-range(max_abs={max_abs_f:.6g})"

    try:
        info = torch.iinfo(dtype)
    except TypeError:
        return False, "not-integer-dtype"

    is_integral = getattr(meta, "scale_is_integral", None)
    if is_integral is False:
        return False, "nonintegral-data"

    min_v = getattr(meta, "scale_min_value", None)
    max_v = getattr(meta, "scale_max_value", None)
    if min_v is not None and max_v is not None:
        try:
            min_f = float(min_v)
            max_f = float(max_v)
        except Exception:
            return False, "int-minmax-not-float"
        if (min_f < float(info.min)) or (max_f > float(info.max)):
            return (
                False,
                (
                    f"int-range(min={min_f:.6g},max={max_f:.6g},"
                    f"allowed=[{float(info.min):.6g},{float(info.max):.6g}])"
                ),
            )
        return True, "ok"

    if max_abs_f <= float(info.max):
        return True, "ok"
    return False, f"int-max-abs(max_abs={max_abs_f:.6g},max={float(info.max):.6g})"


def is_nvidia_te_available(model: torch.nn.Module) -> bool:
    cached = getattr(model, "__stnet_cached_is_nvidia_te_available__", None)
    if isinstance(cached, bool):
        return cached

    te_flags = (
        getattr(model, "__fp8_inference_te__", False),
        getattr(model, "__fp8_training_te__", False),
        getattr(model, "__te_fp8_default__", False),
    )
    if any(te_flags):
        try:
            setattr(model, "__stnet_cached_is_nvidia_te_available__", True)
        except Exception:
            pass
        return True

    for module in model.modules():
        mod_name = getattr(module.__class__, "__module__", "")
        if isinstance(mod_name, str) and mod_name.startswith("transformer_engine"):
            try:
                setattr(model, "__stnet_cached_is_nvidia_te_available__", True)
            except Exception:
                pass
            return True
    try:
        setattr(model, "__stnet_cached_is_nvidia_te_available__", False)
    except Exception:
        pass
    return False


def is_scale_safe(
    dtype: torch.dtype,
    meta: Optional[Dataset[Any]],
    *args: Any,
    safety_margin: float = 8.0,
    underflow_action: Optional[str] = None,
    **kwargs: Any,
) -> bool:
    """Return True if `dtype` can represent the dataset scale without overflow.

    This is used during AMP / low-precision negotiation.

    Underflow handling is policy-controlled:
    - allow: underflow is allowed
    - warn: allow (loggers may warn elsewhere)
    - forbid: treat underflow as unsafe for downcasting
    """
    ok, _ = _scale_safety_check(
        dtype,
        meta,
        safety_margin=safety_margin,
        underflow_action=underflow_action,
    )
    return ok




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
        **kwargs: Any,
    ) -> nn.Module:
        normalized_mode = ""
        if mode is not None:
            normalized_mode = str(mode).strip().lower()
        if disable or normalized_mode in {"", "disabled", "none"}:
            return module
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            return module
        normalized_alias = normalized_mode.replace("_", "-").replace(" ", "-")
        if "-" in normalized_alias:
            normalized_alias = "-".join(
                part for part in normalized_alias.split("-") if part
            )
        recognized_modes = {
            "default",
            "aot_eager",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }

        canonical_mode = normalized_alias or normalized_mode

        # TorchInductor compilation can be very CPU hungry. When running multiple local
        # ranks (DDP/FSDP), each process compiling with a large thread pool can easily
        # oversubscribe the node and slow down *both* compilation and the input pipeline.
        #
        # Keep this conservative and fully overridable:
        #   - STNET_INDUCTOR_COMPILE_THREADS / STNET_COMPILE_THREADS
        #   - TORCHINDUCTOR_COMPILE_THREADS
        try:
            from torch._inductor import config as _inductor_config  # type: ignore
        except Exception:
            _inductor_config = None

        if _inductor_config is not None:
            with _INDUCTOR_CONFIG_LOCK:
                try:
                    if getattr(_inductor_config, "compile_threads", None) is not None:
                        override = env_first(
                            ("STNET_INDUCTOR_COMPILE_THREADS", "STNET_COMPILE_THREADS"),
                            None,
                        )
                        if override is None:
                            override = env_first(("TORCHINDUCTOR_COMPILE_THREADS",), None)

                        if override is not None:
                            _inductor_config.compile_threads = max(1, int(override))
                        else:
                            local_world = env_first_int(
                                (
                                    "STNET_LOCAL_WORLD_SIZE",
                                    "LOCAL_WORLD_SIZE",
                                    "SLURM_NTASKS_PER_NODE",
                                ),
                                1,
                            )
                            if int(local_world) > 1:
                                # Pick a small per-rank compile thread count. Too many threads
                                # across multiple ranks oversubscribes the node and hurts both
                                # compilation latency and the data pipeline.
                                cpu_count = int(process_cpu_count() or 1)
                                per_rank = max(
                                    1, int(cpu_count) // max(1, int(local_world))
                                )
                                _inductor_config.compile_threads = max(
                                    1, min(4, int(per_rank) // 2)
                                )
                except Exception:
                    pass
        if canonical_mode == "max-autotune" and not _is_for_cuda(module):
            canonical_mode = "max-autotune-no-cudagraphs"
        if canonical_mode == "max-autotune-no-cudagraphs":
            try:
                _opt = dict(options or {})
                _opt.setdefault("triton.cudagraphs", False)
                options = _opt
            except Exception:
                pass
        if canonical_mode in {"max-autotune", "max-autotune-no-cudagraphs"}:
            try:
                from torch._inductor import config as _inductor_config
            except Exception:
                _inductor_config = None

            if _inductor_config is not None:
                with _INDUCTOR_CONFIG_LOCK:
                    try:
                        _inductor_config.autotune_in_subproc = True
                    except Exception:
                        pass

                    try:
                        _inductor_config.autotune_local_cache = True
                    except Exception:
                        pass
                    try:
                        _inductor_config.autotune_remote_cache = None
                    except Exception:
                        pass

                    try:
                        if (
                            getattr(
                                _inductor_config,
                                "max_autotune_gemm_search_space",
                                None,
                            )
                            is not None
                        ):
                            _inductor_config.max_autotune_gemm_search_space = "DEFAULT"
                    except Exception:
                        pass

                    try:
                        if getattr(_inductor_config, "compile_threads", None) is not None:
                            # max-autotune can be very memory hungry; default to serial compilation
                            # unless the user explicitly overrides the thread count.
                            override_raw = env_first(
                                (
                                    "STNET_INDUCTOR_COMPILE_THREADS",
                                    "STNET_COMPILE_THREADS",
                                    "TORCHINDUCTOR_COMPILE_THREADS",
                                )
                            )
                            override_valid = False
                            if override_raw is not None:
                                with contextlib.suppress(Exception):
                                    int(override_raw)
                                    override_valid = True

                            if not override_valid:
                                _inductor_config.compile_threads = 1
                    except Exception:
                        pass

                    # max-autotune is extremely memory hungry. Keep GEMM tuning but disable
                    # pointwise autotune by default to reduce peak memory (RAM/VRAM).
                    try:
                        if getattr(_inductor_config, "max_autotune_pointwise", None) is not None:
                            _inductor_config.max_autotune_pointwise = False
                    except Exception:
                        pass
                    try:
                        if getattr(_inductor_config, "max_autotune_gemm", None) is not None:
                            _inductor_config.max_autotune_gemm = True
                    except Exception:
                        pass

        backend_value = backend
        mode_value: Optional[str] = None
        if canonical_mode in {"aot_eager", "aot-eager"}:
            backend_value = "aot_eager"
        elif canonical_mode in recognized_modes:
            mode_value = canonical_mode
        elif mode is not None:
            mode_value = str(mode)

        compile_kwargs: Dict[str, Any] = dict(kwargs)
        if backend_value is not None:
            compile_kwargs["backend"] = backend_value
        if mode_value is not None:
            compile_kwargs["mode"] = mode_value
        if fullgraph is not None:
            compile_kwargs["fullgraph"] = bool(fullgraph)
        if dynamic is not None:
            compile_kwargs["dynamic"] = bool(dynamic)
        if options and mode_value is not None:
            try:
                from torch._inductor import config as _inductor_config  # type: ignore
            except Exception:
                _inductor_config = None
            if _inductor_config is not None:
                for _k, _v in dict(options).items():
                    if not isinstance(_k, str):
                        continue
                    try:
                        _obj = _inductor_config
                        _parts = _k.split(".")
                        for _p in _parts[:-1]:
                            _obj = getattr(_obj, _p)
                        setattr(_obj, _parts[-1], _v)
                    except Exception:
                        pass
            options = None
        if options:
            existing = compile_kwargs.get("options", {})
            if isinstance(existing, dict):
                merged = {**dict(options), **existing}
            else:
                merged = dict(options)
            compile_kwargs["options"] = merged

        return compile_fn(module, **compile_kwargs)


class Autocast:
    _preferred_fp8_backend: Optional[str] = None
    _preferred_int_backend: Optional[str] = None
    _last_float_dtype: torch.dtype = torch.float32
    _last_int_dtype: torch.dtype = torch.int64
    _metadata: Optional[Dataset[Any]] = None  # best-effort legacy snapshot
    _metadata_tls = threading.local()  # per-thread metadata to avoid cross-thread mutation

    @classmethod
    def _get_tls_metadata(cls) -> Optional[Dataset[Any]]:
        return getattr(cls._metadata_tls, "meta", None)

    @classmethod
    def _set_tls_metadata(cls, meta: Optional[Dataset[Any]]) -> None:
        setattr(cls._metadata_tls, "meta", meta)
        # Keep a best-effort snapshot for any legacy access paths.
        cls._metadata = meta

    @classmethod
    def metadata(cls) -> Optional[Dataset[Any]]:
        """Thread-local metadata accessor (preferred over `Autocast._metadata`)."""
        return cls._get_tls_metadata()

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
                ok, reason = Dataset.is_float8_supported(dev)
                if not ok:
                    _LOGGER.debug("Autocast FP8 TE unavailable: %s", reason)
                    continue
                try:
                    te = importlib.import_module("transformer_engine.pytorch")

                    if getattr(te, "fp8_autocast", None) is None:
                        raise AttributeError("transformer_engine.fp8_autocast missing")
                except Exception as exc:
                    _LOGGER.debug("Autocast FP8 TE import failed: %s", exc)
                    continue
                cls._preferred_fp8_backend = "te"
                return "te"
            elif backend == "ao":
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
                ok, reason = Dataset.is_int8_supported(dev)
                if not ok:
                    _LOGGER.debug("Autocast INT8 TE unavailable: %s", reason)
                    continue
                try:
                    te = importlib.import_module("transformer_engine.pytorch")

                    if getattr(te, "int8_autocast", None) is None:
                        raise AttributeError("transformer_engine.int8_autocast missing")
                except Exception as exc:
                    _LOGGER.debug("Autocast INT8 TE import failed: %s", exc)
                    continue
                cls._preferred_int_backend = "te"
                return "te"
            elif backend == "ao":
                try:
                    quant_mod = importlib.import_module("torchao.quantization")
                    int8_autocast = getattr(quant_mod, "int8_autocast", None)

                    if not callable(int8_autocast):
                        raise AttributeError(
                            "torchao.quantization.int8_autocast missing"
                        )
                except Exception as exc:
                    _LOGGER.debug("Autocast INT8 torchao import failed: %s", exc)
                    continue
                cls._preferred_int_backend = "ao"
                return "ao"
        cls._preferred_int_backend = None
        return None

    @classmethod
    def coerce_metadata(
        cls: object,
        device: Optional[Union[torch.device, str]] = None,
        *args: Any,
        metadata: Optional[Dataset[Any]] = None,
        **kwargs: Any,
    ) -> Dataset[Any]:
        meta = metadata or cls._get_tls_metadata()
        device_hint: Optional[Union[torch.device, str]] = device
        if device_hint is None and meta is not None:
            with contextlib.suppress(Exception):
                device_hint = torch.device(meta.device)
        dev = cls._device(device_hint)
        if meta is None:
            meta = Dataset.for_device(dev)
        else:
            current_device = torch.device(getattr(meta, "device", dev))
            if current_device != dev:
                meta.device = dev
                meta.refresh()
            else:
                meta.ensure_device_info()
        if not getattr(meta, "float_dtypes", ()):
            meta.refresh()
        elif not getattr(meta, "int_dtypes", ()) or not getattr(
            meta, "float8_dtypes", ()
        ):
            meta.refresh()
        else:
            meta.ensure_device_info()
        cls._set_tls_metadata(meta)
        return meta

    @classmethod
    def float_amp_priority(
        cls: object, device: torch.device
    ) -> Tuple[torch.dtype, ...]:
        meta = cls.coerce_metadata(device)
        candidates = getattr(meta, "float_dtypes", ())
        if candidates:
            return tuple(candidates)
        return (torch.float32,)

    @staticmethod
    def float8_formats() -> Tuple[torch.dtype, ...]:
        meta = Autocast._get_tls_metadata()
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
    def integer_amp_priority(
        cls: object, device: torch.device
    ) -> Tuple[torch.dtype, ...]:
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
        meta: Optional[Dataset[Any]] = None,
        decision_key: object = None,
        **kwargs: Any,
    ) -> torch.dtype:
        # Evaluate candidates in order.
        #
        # Safety margin controls how conservative dtype negotiation is.
        # It can be specified either directly (safety_margin=float) or as a
        # power-of-two exponent n (safety_margin_pow2),
        # where safety_margin = 2**n.
        #
        # Safety margin is validated: non-finite or <=0 values fall back to a
        # conservative default.
        raw_pow2 = kwargs.pop("safety_margin_pow2", None)

        safety_margin_pow2: Optional[int] = None
        if raw_pow2 is not None:
            try:
                safety_margin_pow2 = int(raw_pow2)
            except Exception:
                safety_margin_pow2 = 3
            if safety_margin_pow2 < 0:
                safety_margin_pow2 = 0
            # Hard cap to avoid pathological margins.
            if safety_margin_pow2 > 30:
                safety_margin_pow2 = 30
            safety_margin = float(2 ** safety_margin_pow2)
        else:
            raw_margin = kwargs.pop("safety_margin", 8.0)
            try:
                safety_margin = float(raw_margin)
            except Exception:
                safety_margin = 8.0
            if (not math.isfinite(safety_margin)) or (safety_margin <= 0.0):
                safety_margin = 8.0
            # Infer pow2 for logging when safety_margin is exactly a power of two.
            with contextlib.suppress(Exception):
                n = int(round(math.log2(safety_margin)))
                if n >= 0:
                    ref = float(2 ** n)
                    if abs(ref - safety_margin) / max(abs(safety_margin), 1.0) < 1e-12:
                        safety_margin_pow2 = n

        # Optional override: allow callers to specify underflow policy for this
        # negotiation only.
        raw_underflow = kwargs.pop("underflow_action", None)
        if raw_underflow is None:
            raw_underflow = kwargs.pop("underflow", None)
        underflow_override: Optional[str] = None
        if raw_underflow is not None:
            underflow_override = normalize_underflow_action(
                raw_underflow, default=default_underflow_action()
            )

        # Avoid per-call allocations when logging is disabled.
        collect_checks = False
        if logger is not None:
            try:
                collect_checks = logger.isEnabledFor(logging.DEBUG) or logger.isEnabledFor(
                    logging.INFO
                )
            except Exception:
                # Best-effort: if probing fails, err on the side of collecting.
                collect_checks = True

        checks: List[Dict[str, Any]] = [] if collect_checks else []
        selected: Optional[torch.dtype] = None
        selected_from: str = "candidate"

        for dtype in candidates:
            ok, why = _scale_safety_check(
                dtype,
                meta,
                safety_margin=safety_margin,
                underflow_action=underflow_override,
            )
            if collect_checks:
                checks.append(
                    {"dtype": _dtype_short(dtype), "ok": bool(ok), "reason": str(why)}
                )
            if ok:
                selected = dtype
                break

        dev_type = str(getattr(device, "type", "")) if device is not None else ""
        dev_index = (
            int(getattr(device, "index", -1))
            if (device is not None and getattr(device, "index", None) is not None)
            else -1
        )
        device_str = f"{dev_type}:{dev_index}" if dev_type else ""

        # Resolve fallbacks when candidates are not safe.
        fallback_order: Tuple[torch.dtype, ...] = ()
        if selected is None:
            selected_from = "fallback"
            fallback_order: Tuple[torch.dtype, ...]
            if getattr(fallback, "is_floating_point", False):
                fallback_order = (fallback, torch.float32, torch.float64)
            else:
                fallback_order = (fallback, torch.int64, torch.float32, torch.float64)
            for dtype in fallback_order:
                ok, why = _scale_safety_check(
                    dtype,
                    meta,
                    safety_margin=safety_margin,
                    underflow_action=underflow_override,
                )
                if collect_checks:
                    checks.append(
                        {"dtype": _dtype_short(dtype), "ok": bool(ok), "reason": str(why)}
                    )
                if ok:
                    selected = dtype
                    break
            if selected is None:
                selected = fallback
                selected_from = "unsafe-fallback"

        # Best-effort structured log, deduped.
        if logger is not None:
            level = "info" if selected_from != "candidate" else "debug"
            lvl = logging.INFO if level == "info" else logging.DEBUG
            should_log = True
            try:
                should_log = logger.isEnabledFor(lvl)
            except Exception:
                should_log = True
            if not should_log:
                return selected

            scale_key = (
                bool(getattr(meta, "has_scale", False)) if meta is not None else False,
                bool(getattr(meta, "has_nonfinite", False)) if meta is not None else False,
                getattr(meta, "scale_max_abs", None) if meta is not None else None,
                getattr(meta, "scale_min_positive", None) if meta is not None else None,
                getattr(meta, "scale_min_value", None) if meta is not None else None,
                getattr(meta, "scale_max_value", None) if meta is not None else None,
                str(getattr(meta, "underflow_action", "")) if meta is not None else "",
                getattr(meta, "int_quant_bits", None) if meta is not None else None,
            )
            if decision_key is None:
                decision_key = (
                    "amp",
                    str(context),
                    dev_type,
                    dev_index,
                    tuple(_dtype_short(x) for x in candidates),
                    _dtype_short(fallback),
                    scale_key,
                    float(safety_margin),
                    (int(safety_margin_pow2) if safety_margin_pow2 is not None else None),
                    (underflow_override or ""),
                )
            payload: Dict[str, Any] = {
                "context": str(context),
                "device": device_str,
                "selected": _dtype_short(selected),
                "selected_from": selected_from,
                "fallback": _dtype_short(fallback),
                "candidates": [_dtype_short(x) for x in candidates],
                "fallback_order": (
                    ([_dtype_short(x) for x in fallback_order] if fallback_order else [])
                ),
                "checks": (checks if collect_checks else []),
                "safety_margin": safety_margin,
                "safety_margin_pow2": safety_margin_pow2,
                "underflow_action_override": underflow_override,
                "scale": {
                    "has_scale": bool(getattr(meta, "has_scale", False))
                    if meta is not None
                    else False,
                    "has_nonfinite": bool(getattr(meta, "has_nonfinite", False))
                    if meta is not None
                    else False,
                    "max_abs": getattr(meta, "scale_max_abs", None) if meta is not None else None,
                    "min_positive": getattr(meta, "scale_min_positive", None)
                    if meta is not None
                    else None,
                    "min_value": getattr(meta, "scale_min_value", None) if meta is not None else None,
                    "max_value": getattr(meta, "scale_max_value", None) if meta is not None else None,
                    "underflow_action": str(getattr(meta, "underflow_action", ""))
                    if meta is not None
                    else "",
                    "int_quant_bits": getattr(meta, "int_quant_bits", None)
                    if meta is not None
                    else None,
                },
            }
            _log_negotiate_once(logger, decision_key, payload, level=level)

        return selected

    @classmethod
    def _nvidia_float8(
        cls: object, device: torch.device, enabled: bool
    ) -> List[AbstractContextManager[None]]:
        contexts: List[AbstractContextManager[None]] = []
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
    ) -> List[AbstractContextManager[None]]:
        contexts: List[AbstractContextManager[None]] = []
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
    ) -> List[AbstractContextManager[None]]:
        contexts: List[AbstractContextManager[None]] = []
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
    def _torchao_int4(
        cls: object,
        device: torch.device,
        enabled: bool,
    ) -> List[AbstractContextManager[None]]:
        """Best-effort INT4 autocast/quant context (torchao-only).

        TorchAO APIs may differ by version; we probe a small set of known names.
        """
        contexts: List[AbstractContextManager[None]] = []
        if not enabled:
            return contexts
        try:
            quant_mod = importlib.import_module("torchao.quantization")
        except Exception as exc:
            _LOGGER.debug("Autocast INT4 torchao import failed: %s", exc)
            return contexts

        for name in (
            "int4_weight_only_autocast",
            "int4_autocast",
            "int4wo_autocast",
            "int4w_autocast",
        ):
            fn = getattr(quant_mod, name, None)
            if callable(fn):
                try:
                    contexts.append(fn(enabled=True))
                    return contexts
                except Exception as exc:
                    _LOGGER.debug("Autocast INT4 torchao %s failed: %s", name, exc)
                    return []
        _LOGGER.debug("Autocast INT4 torchao context not found")
        return contexts

    @classmethod
    def configure(
        cls: object,
        model: Optional[nn.Module],
        *args: Any,
        metadata: Optional[Dataset[Any]] = None,
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
        cls._set_tls_metadata(meta)

    @classmethod
    def resolve_float_dtype(
        cls,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[Union[torch.dtype, str]] = None,
        metadata: Optional[Dataset[Any]] = None,
    ) -> Optional[torch.dtype]:
        """Return the float autocast dtype that `Autocast.float(...)` would use.

        Returns None when autocast is disabled by metadata.
        """
        dev = cls._device(device)
        meta = cls.coerce_metadata(device=dev, metadata=metadata)
        disable = bool(meta.is_disabled()) if meta is not None else False
        if disable:
            return None

        def _coerce_dt(x: Any, default: torch.dtype) -> torch.dtype:
            if x is None:
                return default
            if isinstance(x, torch.dtype):
                return x
            s = str(x).strip()
            s = s.replace("torch.", "")
            return getattr(torch, s, default)

        requested_dtype = _coerce_dt(dtype, torch.float16)
        candidates: Tuple[torch.dtype, ...] = (requested_dtype, cls._last_float_dtype)
        if meta is not None:
            extra = getattr(meta, "float_dtypes", None)
            if extra:
                try:
                    candidates = tuple(_coerce_dt(x, requested_dtype) for x in extra)
                except Exception:
                    pass

        chosen = cls.negotiate(
            candidates,
            fallback=requested_dtype,
            device=dev,
            meta=meta,
            context="autocast-float",
        )
        return chosen

    @classmethod
    @contextlib.contextmanager
    def float(
        cls: object,
        device: Optional[Union[torch.device, str]] = None,
        *args: Any,
        metadata: Optional[Dataset[Any]] = None,
        **kwargs: Any,
    ) -> contextlib.AbstractContextManager[None]:
        dev = cls._device(device)
        meta = cls.coerce_metadata(dev, metadata=metadata)
        amp_candidates = (
            tuple(meta.float_dtypes) if meta.float_dtypes else (torch.float32,)
        )
        amp_dtype = cls.negotiate(
            amp_candidates,
            fallback=torch.float64,
            logger=_LOGGER,
            context="float",
            device=dev,
            meta=meta,
        )
        contexts: List[contextlib.AbstractContextManager[None]] = []

        # Track FP8 backend negotiation decisions for structured debug logging.
        debug = _LOGGER.isEnabledFor(logging.DEBUG)
        fp8_disable_reason: Optional[str] = None
        fp8_backend_requested: Optional[str] = None
        fp8_backend_used: Optional[str] = None
        fp8_checks: Dict[str, Any] = {}

        backend = cls._fp8_backend(cls._preferred_fp8_backend, device=dev)
        fp8_backend_requested = backend
        float8_dtypes = (
            tuple(meta.float8_dtypes) if meta.float8_dtypes else cls.float8_formats()
        )
        wants_fp8 = backend is not None
        if wants_fp8 and getattr(meta, "has_scale", False):
            fp8_supported = False
            for dtype in float8_dtypes:
                ok, why = _scale_safety_check(dtype, meta, safety_margin=2.0)
                if debug:
                    fp8_checks[_dtype_short(dtype)] = {"ok": bool(ok), "reason": str(why)}
                if ok:
                    fp8_supported = True
            if not fp8_supported:
                wants_fp8 = False
                fp8_disable_reason = "scale-exceeds-fp8"
                _LOGGER.debug(
                    "Autocast FP8 disabled on %s: data scale exceeds float8 range",
                    dev.type,
                )
        if wants_fp8:
            if backend == "te":
                fp8_contexts = cls._nvidia_float8(dev, True)
                contexts.extend(fp8_contexts)
                if fp8_contexts:
                    fp8_backend_used = "te"
                else:
                    backend = cls._fp8_backend("ao", device=dev)
                    if backend == "ao":
                        fp8_contexts = cls._torchao_float8(True)
                        contexts.extend(fp8_contexts)
                        if fp8_contexts:
                            fp8_backend_used = "ao"
                        else:
                            fp8_disable_reason = "fp8-backend-unavailable"
            elif backend == "ao":
                fp8_contexts = cls._torchao_float8(True)
                contexts.extend(fp8_contexts)
                if fp8_contexts:
                    fp8_backend_used = "ao"
                else:
                    fp8_disable_reason = "fp8-backend-unavailable"
            else:
                _LOGGER.debug(
                    "Autocast FP8 backend '%s' unsupported; disabling", backend
                )
                cls._preferred_fp8_backend = None
                fp8_disable_reason = "fp8-backend-unsupported"

        requested_dtype = amp_dtype
        if (
            isinstance(cls._last_float_dtype, torch.dtype)
            and cls._last_float_dtype in amp_candidates
            and cls._last_float_dtype == amp_dtype
        ):
            requested_dtype = cls._last_float_dtype
        if requested_dtype is torch.float64:
            wants_fp8 = False
            # fp64 master disables low-precision contexts.
            fp8_disable_reason = fp8_disable_reason or "master-fp64"
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
        cls._set_tls_metadata(meta)

        if debug:
            try:
                _LOGGER.debug(
                    "Autocast.context(float): %s",
                    json.dumps(
                        {
                            "device": str(dev),
                            "amp_dtype": _dtype_short(amp_dtype),
                            "amp_candidates": [_dtype_short(d) for d in amp_candidates],
                            "fp8_backend_requested": fp8_backend_requested,
                            "fp8_backend_used": fp8_backend_used,
                            "fp8_enabled": bool(fp8_backend_used),
                            "fp8_disable_reason": fp8_disable_reason,
                            "fp8_checks": fp8_checks,
                            "scale": _meta_scale_summary(meta),
                        },
                        sort_keys=True,
                        default=str,
                    ),
                )
            except Exception:
                pass

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
        metadata: Optional[Dataset[Any]] = None,
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
        quant_bits = getattr(meta, "int_quant_bits", None)
        wants_int4 = quant_bits == 4
        wants_int8 = (int_dtype == torch.int8) or (quant_bits == 8)

        debug = _LOGGER.isEnabledFor(logging.DEBUG)
        int_backend_requested: Optional[str] = None
        int_backend_used: Optional[str] = None
        int_disable_reason: Optional[str] = None

        contexts: List[contextlib.AbstractContextManager[None]] = []

        if wants_int4:
            try:
                contexts = cls._torchao_int4(dev, True)
                if contexts:
                    cls._preferred_int_backend = "ao"
                    int_backend_used = "ao"
            except Exception as exc:
                _LOGGER.debug("Autocast INT4 enable failed: %s", exc)
                contexts = []
                int_disable_reason = "int4-backend-unavailable"
        if not contexts and wants_int8:
            backend = cls._int_backend(cls._preferred_int_backend, device=dev)
            int_backend_requested = backend
            contexts = cls._torchao_int8(dev, True) if backend else []
            if contexts:
                int_backend_used = backend
            if (not contexts) and backend == "te":
                fallback_backend = cls._int_backend("ao", device=dev)
                if fallback_backend == "ao":
                    contexts = cls._torchao_int8(dev, True)
                    if contexts:
                        int_backend_used = "ao"
            if (not contexts) and wants_int8:
                int_disable_reason = "int8-backend-unavailable"
        if not contexts:
            contexts.append(contextlib.nullcontext())

        with contextlib.ExitStack() as stack:
            for ctx in contexts:
                stack.enter_context(ctx)
            cls._last_int_dtype = int_dtype
            cls._set_tls_metadata(meta)
            if debug:
                try:
                    _LOGGER.debug(
                        "Autocast.context(int): %s",
                        json.dumps(
                            {
                                "device": str(dev),
                                "int_dtype": _dtype_short(int_dtype),
                                "int_candidates": [_dtype_short(d) for d in int_candidates],
                                "quant_bits": int(quant_bits) if quant_bits is not None else None,
                                "wants_int4": bool(wants_int4),
                                "wants_int8": bool(wants_int8),
                                "int_backend_requested": int_backend_requested,
                                "int_backend_used": int_backend_used,
                                "int_enabled": bool(int_backend_used),
                                "int_disable_reason": int_disable_reason,
                                "scale": _meta_scale_summary(meta),
                            },
                            sort_keys=True,
                            default=str,
                        ),
                    )
                except Exception:
                    pass
            yield


_Int8DynamicActivationInt8WeightConfig: Any | None
_Int8WeightOnlyConfig: Any | None
_quantize: Any | None
_ptq_impl: Callable[..., tuple[nn.Module, bool, str]] | None
QATConfig = None
QATStep = None

_IntxFakeQuantizeConfig: Any | None = None
_FakeQuantizedLinear: Any | None = None
_FakeQuantizedEmbedding: Any | None = None

try:
    from torchao.quantization import (
        Int8DynamicActivationInt8WeightConfig as _Int8DynamicActivationInt8WeightConfig,
    )
    from torchao.quantization import Int8WeightOnlyConfig as _Int8WeightOnlyConfig
    from torchao.quantization import quantize_ as _quantize
except ImportError:
    _quantize = None
    _Int8DynamicActivationInt8WeightConfig = None
    _Int8WeightOnlyConfig = None


def _torchao_int8_ptq_impl(
    model: nn.Module,
    *args: Any,
    mode: str = "int8",
    dynamic_activations: bool,
    group_size: int = 128,
    logger: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> tuple[nn.Module, bool, str]:
    """TorchAO 0.14+ PTQ impl using quantize_ + config objects.

    TorchAO does not expose quantize_.ptq; PTQ is applying a PTQ config via quantize_().
    """
    if not callable(_quantize):
        return (model, False, "torchao.quantization not installed")
    if str(mode).lower() not in {"int8", "w8", "w8a8", "int8wo"}:
        return (model, False, f"Unsupported PTQ mode: {mode}")

    cfg_cls = (
        _Int8DynamicActivationInt8WeightConfig
        if dynamic_activations
        else _Int8WeightOnlyConfig
    )
    if cfg_cls is None:
        return (model, False, "Quantization config unavailable")

    try:
        if dynamic_activations:
            cfg = cfg_cls()
        else:
            # Int8WeightOnlyConfig supports group_size (optional). Use it when sane.
            gs = int(group_size) if group_size is not None else None
            if gs is not None and gs <= 0:
                gs = None
            try:
                cfg = cfg_cls(group_size=gs)
            except TypeError:
                cfg = cfg_cls()
    except Exception as exc:
        return (model, False, f"Failed to initialize quantization config: {exc}")

    try:
        _quantize(model, cfg)
    except Exception as exc:
        return (model, False, f"AO failed: {exc}")

    if logger is not None:
        logger(f"[INT8][AO] applied {cfg.__class__.__name__}")
    return (model, True, "torchao")


_ptq_impl = _torchao_int8_ptq_impl if callable(_quantize) else _is_ptq_unavailable


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
    from torchao.quantization.qat import (
        IntxFakeQuantizeConfig as _IntxFakeQuantizeConfig,
    )
    from torchao.quantization.qat import FakeQuantizedLinear as _FakeQuantizedLinear
    from torchao.quantization.qat import FakeQuantizedEmbedding as _FakeQuantizedEmbedding
except Exception:
    _IntxFakeQuantizeConfig = None
    _FakeQuantizedLinear = None
    _FakeQuantizedEmbedding = None


class Quantization:
    quantize: Optional[Callable[..., Any]] = _quantize
    Int8DynamicActivationInt8WeightConfig: Optional[type] = (
        _Int8DynamicActivationInt8WeightConfig
    )
    Int8WeightOnlyConfig: Optional[type] = _Int8WeightOnlyConfig
    QATConfig: Any = QATConfig
    QATStep: Any = QATStep
    IntxFakeQuantizeConfig: Any = _IntxFakeQuantizeConfig
    FakeQuantizedLinear: Any = _FakeQuantizedLinear
    FakeQuantizedEmbedding: Any = _FakeQuantizedEmbedding
    ptq: Callable[..., tuple[nn.Module, bool, str]] = staticmethod(_ptq_impl)

    @classmethod
    def is_available(cls: object) -> bool:
        return callable(cls.quantize)

    @classmethod
    def is_qat_available(cls: object) -> bool:
        # We implement INT8 QAT via FakeQuantized* + IntxFakeQuantizeConfig to avoid
        # base_config limitations and embedding restrictions.
        return (
            callable(cls.quantize)
            and (cls.IntxFakeQuantizeConfig is not None)
            and (cls.FakeQuantizedLinear is not None)
            and (cls.FakeQuantizedEmbedding is not None)
        )

    @classmethod
    def is_ptq_available(cls: object) -> bool:
        return callable(cls.quantize) and (
            cls.Int8DynamicActivationInt8WeightConfig is not None
            or cls.Int8WeightOnlyConfig is not None
        )

    @classmethod
    def _qat_has_wrappers(cls: object, model: nn.Module) -> bool:
        fq_lin = cls.FakeQuantizedLinear
        fq_emb = cls.FakeQuantizedEmbedding
        for m in model.modules():
            try:
                if fq_lin is not None and isinstance(m, fq_lin):
                    return True
                if fq_emb is not None and isinstance(m, fq_emb):
                    return True
            except Exception:
                # Fallback: module name-based detection
                mod = getattr(m.__class__, "__module__", "")
                if isinstance(mod, str) and "torchao.quantization.qat" in mod:
                    return True
        return False

    @classmethod
    def _qat_convert_inplace(
        cls: object,
        model: nn.Module,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Convert FakeQuantized* wrappers back to nn modules (best-effort)."""
        if not cls._qat_has_wrappers(model):
            return
        fq_lin = cls.FakeQuantizedLinear
        fq_emb = cls.FakeQuantizedEmbedding

        def _rec(parent: nn.Module) -> None:
            for name, child in list(parent.named_children()):
                new_child = child
                try:
                    if fq_lin is not None and isinstance(child, fq_lin):
                        new_child = child.to_linear()
                    elif fq_emb is not None and isinstance(child, fq_emb):
                        new_child = child.to_embedding()
                    else:
                        _rec(child)
                except Exception:
                    # keep best-effort; don't hard-fail conversions
                    _rec(child)
                if new_child is not child:
                    setattr(parent, name, new_child)

        _rec(model)
        _invalidate_model_introspection_caches(model)
        _log_debug(logger, "[QAT] converted fake-quant wrappers back to fp modules")

    @classmethod
    def _build_int8_cfg(
        cls: object,
        *,
        dynamic_activations: bool,
        group_size: int = 128,
    ) -> Any:
        cfg_cls = (
            cls.Int8DynamicActivationInt8WeightConfig
            if dynamic_activations
            else cls.Int8WeightOnlyConfig
        )
        if cfg_cls is None:
            raise RuntimeError("Quantization config unavailable")
        if dynamic_activations:
            return cfg_cls()
        gs = int(group_size) if group_size is not None else None
        if gs is not None and gs <= 0:
            gs = None
        try:
            return cfg_cls(group_size=gs)
        except TypeError:
            return cfg_cls()

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
        # If model was already QAT-prepared, unwrap first to avoid stacking wrappers.
        cls._qat_convert_inplace(model, logger=logger)

        fq_cfg = cls.IntxFakeQuantizeConfig
        fq_lin = cls.FakeQuantizedLinear
        fq_emb = cls.FakeQuantizedEmbedding
        if fq_cfg is None or fq_lin is None or fq_emb is None:
            raise RuntimeError("TorchAO QAT primitives unavailable")

        gs: Optional[int]
        try:
            gs = int(group_size) if group_size is not None else None
        except Exception:
            gs = None
        if gs is not None and gs <= 0:
            gs = None

        # NOTE:
        # - activation fake-quant is NOT supported for embeddings (TorchAO raises)
        # - symmetric per-token activation fake-quant is not supported
        act_cfg = None
        if dynamic_activations:
            act_cfg = fq_cfg(dtype=torch.int8, granularity="per_token", is_symmetric=False)

        replaced_linear = 0
        replaced_embed = 0

        def _swap(parent: nn.Module) -> None:
            nonlocal replaced_linear, replaced_embed
            for name, child in list(parent.named_children()):
                new_child = child
                if isinstance(child, nn.Linear):
                    # Per-layer group_size fallback when divisibility doesn't hold.
                    w_gs = None
                    if gs is not None:
                        with contextlib.suppress(Exception):
                            if int(child.in_features) % int(gs) == 0:
                                w_gs = gs
                    w_cfg = fq_cfg(dtype=torch.int8, group_size=w_gs, is_symmetric=True)
                    new_child = fq_lin.from_linear(child, act_cfg, w_cfg)
                    replaced_linear += 1
                elif isinstance(child, nn.Embedding):
                    # Embedding: activation fake-quant must be None.
                    emb_dim = int(getattr(child, "embedding_dim", 0) or 0)
                    w_gs = None
                    if gs is not None and emb_dim > 0:
                        with contextlib.suppress(Exception):
                            if emb_dim % int(gs) == 0:
                                w_gs = gs
                    w_cfg = fq_cfg(dtype=torch.int8, group_size=w_gs, is_symmetric=True)
                    new_child = fq_emb.from_embedding(child, w_cfg)
                    replaced_embed += 1
                else:
                    _swap(child)

                if new_child is not child:
                    setattr(parent, name, new_child)

        _swap(model)
        _invalidate_model_introspection_caches(model)
        _log_info(
            logger,
            f"[INT8][QAT] prepared (linear={replaced_linear}, embedding={replaced_embed}, act={'on' if act_cfg is not None else 'off'}, group_size={gs})",
        )
        return {"linear": replaced_linear, "embedding": replaced_embed, "group_size": gs, "act": bool(act_cfg is not None)}

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
        cls._qat_convert_inplace(model, logger=logger)
        # PTQ is applying a PTQ config via quantize_.
        m2, ok, why = cls.ptq(
            model,
            mode="int8",
            dynamic_activations=dynamic_activations,
            group_size=group_size,
            logger=logger,
        )
        if ok:
            _invalidate_model_introspection_caches(m2)
        return (m2, ok, why)

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
        cls._qat_convert_inplace(model, logger=logger)
        group_size = int(kwargs.pop("group_size", 128) or 128)
        try:
            cfg = cls._build_int8_cfg(
                dynamic_activations=dynamic_activations, group_size=group_size
            )
        except Exception as exc:
            return (model, False, f"Failed to initialize quantization config: {exc}")
        try:
            cls.quantize(model, cfg)
        except Exception as exc:
            return (model, False, f"AO failed: {exc}")
        if logger is not None:
            logger(f"[INT8][AO] applied {cfg.__class__.__name__}")
        setattr(model, "__int8_inference_ao__", True)
        _invalidate_model_introspection_caches(model)
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
            _log_info(logger, f"[INT8] {msg}")
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
                return (model, True, "QAT-prepare")
            except Exception as exc:
                last_err = exc
                _log_info(logger, f"[INT8][QAT] prepare failed: {exc}")
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



@lru_cache(maxsize=1)
def _dot_product_attention_cls() -> Any:
    try:
        from ..model.kernels import DotProductAttention as _DotProductAttention

        return _DotProductAttention
    except Exception:
        return None
