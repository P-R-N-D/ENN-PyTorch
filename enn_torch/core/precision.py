# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import importlib
import io
import json
import logging
import math
import threading
from collections import OrderedDict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Self, Tuple, Union

import torch
from torch import nn

from .concurrency import Mutex
from ..core.datatypes import default_underflow_action, normalize_underflow_action
from ..nn.graph import clear_model_cache
from .system import (
    _log_debug,
    _log_info,
    get_device,
    get_device_stats,
    is_cuda_bf16_supported,
    is_float8_supported,
    is_int8_supported,
)

_Int8DynamicActivationInt8WeightConfig = None
_Int8WeightOnlyConfig = None
_LOGGER = logging.getLogger(__name__)
_NEGO_LOGGED_KEYS: "OrderedDict[object, None]" = OrderedDict()
_NEGO_LOGGED_LOCK = Mutex()
_NEGO_LOGGED_MAX: int = 256
_PTQ_IMPL = None
_TORCHAO_IMPORT_LOCK = Mutex()
_TORCHAO_IMPORT_TRIED = False
_qp = None


def __getattr__(name: str) -> Any:
    if name == "PrecisionPolicy":
        raise AttributeError(
            "PrecisionPolicy has moved to enn_torch.core.policies; "
            "import it from that module instead."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _import_torchao_quantization() -> None:
    global _Int8DynamicActivationInt8WeightConfig
    global _Int8WeightOnlyConfig
    global _PTQ_IMPL
    global _qp
    global _TORCHAO_IMPORT_TRIED
    if _TORCHAO_IMPORT_TRIED:
        return
    with _TORCHAO_IMPORT_LOCK:
        if _TORCHAO_IMPORT_TRIED:
            return
        _TORCHAO_IMPORT_TRIED = True
        buf = io.StringIO()
        try:
            with (
                contextlib.redirect_stdout(buf),
                contextlib.redirect_stderr(buf),
            ):
                from torchao.quantization.quant_api import (
                    Int8DynamicActivationInt8WeightConfig as _Int8DynamicActivationInt8WeightConfig,
                )
                from torchao.quantization.quant_api import (
                    Int8WeightOnlyConfig as _Int8WeightOnlyConfig,
                )
                from torchao.quantization.quant_api import quantize_ as _quantize_

                try:
                    from torchao.quantization import (
                        quant_primitives as _quant_primitives,
                    )
                except Exception:
                    _quant_primitives = None
            _PTQ_IMPL = _quantize_
            _qp = _quant_primitives
        except Exception:
            _Int8DynamicActivationInt8WeightConfig = None
            _Int8WeightOnlyConfig = None
            _PTQ_IMPL = None
            _qp = None


def _is_ptq_unavailable(
    model: nn.Module, *args: Any, **kwargs: Any
) -> tuple[nn.Module, bool, str]:
    _ = args, kwargs
    return (model, False, "PTQ backend unavailable")


def _log_negotiation(
    logger: Optional[logging.Logger],
    key: object,
    payload: Dict[str, Any],
    *args: Any,
    level: str = "debug",
) -> None:
    lg, lvl = (
        (logger or _LOGGER),
        (logging.INFO if str(level).lower() == "info" else logging.DEBUG),
    )
    if not lg.isEnabledFor(lvl):
        return

    k = key or (
        payload.get("context"),
        payload.get("device"),
        payload.get("selected"),
    )
    with _NEGO_LOGGED_LOCK:
        if k in _NEGO_LOGGED_KEYS:
            _NEGO_LOGGED_KEYS.move_to_end(k)
            return
        _NEGO_LOGGED_KEYS[k] = None
        if len(_NEGO_LOGGED_KEYS) > _NEGO_LOGGED_MAX:
            _NEGO_LOGGED_KEYS.popitem(last=False)

    try:
        msg = "[AMP][NEGOTIATE] " + json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        msg = f"[AMP][NEGOTIATE] {payload}"
    lg.log(lvl, msg)


def _parse_dtype(dtype: Any) -> str:
    return str(dtype).split(".")[-1] if isinstance(dtype, torch.dtype) else str(dtype)


def _to_serializable(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, str, float)):
        return x
    try:
        return float(x)
    except:
        return str(x)


def _coerce_torch_dtype(value: Any, default: torch.dtype) -> torch.dtype:
    return (
        value
        if isinstance(value, torch.dtype)
        else (
            getattr(torch, str(value).strip().replace("torch.", ""), default)
            if value is not None
            else default
        )
    )


def _get_meta_stats(meta: Any | None) -> Dict[str, Any]:
    return (
        {
            k: _to_serializable(getattr(meta, k, None))
            for k in (
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
        }
        if meta
        else {}
    )


def _validate_dtype_safety(
    dtype: torch.dtype,
    meta: Any | None,
    *args: Any,
    safety_margin: float = 8.0,
    underflow_action: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[bool, str]:
    if not (meta and getattr(meta, "has_scale", False)):
        return True, "no-scale"
    if not isinstance(dtype, torch.dtype):
        return False, "not-dtype"
    if getattr(meta, "has_nonfinite", False):
        return False, "nonfinite-data"

    max_abs = getattr(meta, "scale_max_abs", None)
    if max_abs is None:
        mn = getattr(meta, "scale_min_value", None)
        mx = getattr(meta, "scale_max_value", None)
        if mn is None or mx is None:
            return False, "no-scale-range"
        try:
            mn_f, mx_f = float(mn), float(mx)
        except:
            return False, "minmax-not-float"
        if not (math.isfinite(mn_f) and math.isfinite(mx_f)):
            return False, "minmax-nonfinite"
        max_abs_f = max(abs(mn_f), abs(mx_f))
    else:
        try:
            max_abs_f = float(abs(max_abs))
        except:
            return False, "max-abs-not-float"
        if not math.isfinite(max_abs_f):
            return False, "max-abs-nonfinite"

    action = normalize_underflow_action(
        underflow_action or getattr(meta, "underflow_action", None),
        default=default_underflow_action(),
    )

    if getattr(dtype, "is_complex", False):
        ok, why = _validate_dtype_safety(
            torch.float32 if dtype == torch.complex64 else torch.float64,
            meta,
            safety_margin=safety_margin,
            underflow_action=action,
        )
        return (
            ok,
            f"complex-base:{why}" if ok else f"complex-base-unsafe:{why}",
        )

    if getattr(dtype, "is_floating_point", False):
        info = torch.finfo(dtype)
        if max_abs_f > (ov_limit := float(info.max) / max(1.0, float(safety_margin))):
            return False, f"overflow({max_abs_f:.6g}>{ov_limit:.6g})"
        if (
            action == "forbid"
            and (mp := getattr(meta, "scale_min_positive", None)) is not None
        ):
            try:
                mp_f = float(mp)
            except:
                return False, "min-pos-not-float"
            if (
                math.isfinite(mp_f)
                and mp_f > 0.0
                and mp_f
                < (uf_limit := float(info.tiny) * max(1.0, float(safety_margin)))
            ):
                return False, f"underflow({mp_f:.6g}<{uf_limit:.6g})"
        return True, "ok"

    if getattr(meta, "scale_is_integral", None) is False:
        return False, "nonintegral-data"
    if dtype == torch.bool:
        return (
            max_abs_f <= 1.0,
            "ok" if max_abs_f <= 1.0 else f"bool-range({max_abs_f:.6g})",
        )

    try:
        info = torch.iinfo(dtype)
    except TypeError:
        return False, "not-integer-dtype"

    if (mn := getattr(meta, "scale_min_value", None)) is not None and (
        mx := getattr(meta, "scale_max_value", None)
    ) is not None:
        try:
            mn_f, mx_f = float(mn), float(mx)
        except:
            return False, "int-minmax-not-float"
        if mn_f < float(info.min) or mx_f > float(info.max):
            return (
                False,
                f"int-range({mn_f:.6g},{mx_f:.6g} not in [{info.min},{info.max}])",
            )
        return True, "ok"
    return (
        max_abs_f <= float(info.max),
        (
            "ok"
            if max_abs_f <= float(info.max)
            else f"int-max-abs({max_abs_f:.6g}>{info.max})"
        ),
    )


def is_scale_safe(
    dtype: torch.dtype,
    meta: Any | None,
    *args: Any,
    safety_margin: float = 8.0,
    underflow_action: Optional[str] = None,
    **kwargs: Any,
) -> bool:
    ok, _ = _validate_dtype_safety(
        dtype,
        meta,
        safety_margin=safety_margin,
        underflow_action=underflow_action,
    )
    return ok


@dataclass
class DeviceMeta:
    device: torch.device
    device_type: str = "cpu"
    cuda_cc: Optional[Tuple[int, int]] = None
    float_dtypes: Tuple[torch.dtype, ...] = tuple()
    int_dtypes: Tuple[torch.dtype, ...] = tuple()
    float8_dtypes: Tuple[torch.dtype, ...] = tuple()
    int_quant_bits: Optional[int] = None
    has_scale: bool = False
    has_nonfinite: bool = False
    scale_max_abs: Optional[float] = None
    scale_min_value: Optional[float | int] = None
    scale_max_value: Optional[float | int] = None
    scale_min_positive: Optional[float] = None
    scale_is_integral: Optional[bool] = None
    is_negotiable: Optional[bool] = None
    underflow_action: str = "warn"

    def _refresh_device_info(self: Self) -> None:
        ds = get_device_stats(self.device)
        self.device = ds.device
        self.device_type = ds.device_type
        self.cuda_cc = ds.cuda_cc

    def coerce_device_info(self: Self) -> "DeviceMeta":
        self._refresh_device_info()
        return self

    def refresh(self: Self) -> "DeviceMeta":
        self._refresh_device_info()
        ds = get_device_stats(self.device)
        if not self.float_dtypes:
            self.float_dtypes = tuple(ds.float_dtypes)
        if not self.int_dtypes:
            self.int_dtypes = tuple(ds.int_dtypes)
        if self.int_quant_bits is None:
            self.int_quant_bits = int(ds.int_quant_bits)
        return self

    def is_disabled(self: Self) -> bool:
        return False

    @classmethod
    def for_device(cls: type[Self], device: Union[torch.device, str]) -> "DeviceMeta":
        ds = get_device_stats(device)
        return cls(
            device=ds.device,
            device_type=ds.device_type,
            cuda_cc=ds.cuda_cc,
            float_dtypes=tuple(ds.float_dtypes),
            int_dtypes=tuple(ds.int_dtypes),
            float8_dtypes=tuple(),
            int_quant_bits=int(ds.int_quant_bits),
            underflow_action=default_underflow_action(),
        )


class Autocast:
    _preferred_fp8_backend: Optional[str] = None
    _preferred_int_backend: Optional[str] = None
    _last_float_dtype: torch.dtype = torch.float32
    _last_int_dtype: torch.dtype = torch.int64
    _metadata: Any | None = None
    _metadata_tls = threading.local()

    @classmethod
    def _get_tls_metadata(cls: type[Self]) -> Any | None:
        return getattr(cls._metadata_tls, "meta", None)

    @classmethod
    def _set_tls_metadata(cls: type[Self], meta: Any | None) -> None:
        setattr(cls._metadata_tls, "meta", meta)
        cls._metadata = meta

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
    def _try_load_backend(
        cls: type[Self],
        mod_name: str,
        attr_name: str,
        test_supported: bool = True,
        reason_fail: str = "",
    ) -> Any:
        if not test_supported:
            _LOGGER.debug(reason_fail)
            return None
        try:
            mod = importlib.import_module(mod_name)
            if getattr(mod, attr_name, None) is None:
                raise AttributeError(f"{mod_name}.{attr_name} missing")
            return mod
        except Exception as e:
            _LOGGER.debug(f"Autocast backend {mod_name} failed: {e}")
            return None

    @classmethod
    def _resolve_backend(
        cls: type[Self],
        preferred: Optional[str],
        device: torch.device,
        kind: str,
    ) -> Optional[str]:
        order = ("ao", "te") if preferred == "ao" else ("te", "ao")
        is_supported = is_float8_supported if kind == "fp8" else is_int8_supported

        for backend in order:
            if backend == "te":
                ok, why = is_supported(device)
                if cls._try_load_backend(
                    "transformer_engine.pytorch",
                    f"{kind}_autocast",
                    ok,
                    f"Autocast {kind.upper()} TE unavailable: {why}",
                ):
                    setattr(cls, f"_preferred_{kind}_backend", "te")
                    return "te"
            elif backend == "ao":
                ok, why = is_supported(device)
                mod, attr = (
                    ("torchao.float8", "fp8_autocast")
                    if kind == "fp8"
                    else ("torchao.quantization", "int8_autocast")
                )
                if cls._try_load_backend(
                    mod,
                    attr,
                    ok,
                    f"Autocast {kind.upper()} torchao unavailable: {why}",
                ):
                    setattr(cls, f"_preferred_{kind}_backend", "ao")
                    return "ao"

        setattr(cls, f"_preferred_{kind}_backend", None)
        return None

    @classmethod
    def _fp8_backend(
        cls: type[Self],
        pref: Optional[str],
        *args: object,
        device: Optional[torch.device] = None,
        **kwargs: object,
    ) -> Optional[str]:
        return cls._resolve_backend(pref, device or cls._device(None), "fp8")

    @classmethod
    def _int_backend(
        cls: type[Self],
        pref: Optional[str],
        *args: object,
        device: Optional[torch.device] = None,
        **kwargs: object,
    ) -> Optional[str]:
        return cls._resolve_backend(pref, device or cls._device(None), "int")

    @classmethod
    def _get_backend_context(
        cls: type[Self], mod_name: str, attr_name: str, enabled: bool
    ) -> List[AbstractContextManager[None]]:
        if not enabled:
            return []
        if mod := cls._try_load_backend(mod_name, attr_name):
            return [getattr(mod, attr_name)(enabled=True)]
        return []

    @classmethod
    def _nvidia_float8(
        cls: type[Self], device: torch.device, enabled: bool
    ) -> List[AbstractContextManager[None]]:
        return cls._get_backend_context(
            "transformer_engine.pytorch", "fp8_autocast", enabled
        )

    @classmethod
    def _torchao_float8(
        cls: type[Self], enabled: bool
    ) -> List[AbstractContextManager[None]]:
        return cls._get_backend_context("torchao.float8", "fp8_autocast", enabled)

    @classmethod
    def _torchao_int8_backend(
        cls: type[Self], device: torch.device, enabled: bool
    ) -> List[AbstractContextManager[None]]:
        backend = cls._preferred_int_backend
        if backend == "te":
            return cls._get_backend_context(
                "transformer_engine.pytorch", "int8_autocast", enabled
            )
        if backend == "ao":
            return cls._get_backend_context(
                "torchao.quantization", "int8_autocast", enabled
            )
        return []

    @classmethod
    def _torchao_int4(
        cls: type[Self], device: torch.device, enabled: bool
    ) -> List[AbstractContextManager[None]]:
        return cls._get_backend_context(
            "torchao.quantization", "int4_autocast", enabled
        )

    @classmethod
    def _torchao_int8(
        cls: type[Self], device: torch.device, enabled: bool
    ) -> List[AbstractContextManager[None]]:
        return cls._torchao_int8_autocast(device, enabled)

    @classmethod
    def _torchao_int8_autocast(
        cls: type[Self], device: torch.device, enabled: bool
    ) -> List[AbstractContextManager[None]]:
        if cls._preferred_int_backend is None:
            cls._preferred_int_backend = "ao"
        return cls._torchao_int8_backend(device, enabled)

    @classmethod
    def metadata(cls: type[Self]) -> Any | None:
        return cls._get_tls_metadata()

    @staticmethod
    def float8_formats() -> Tuple[torch.dtype, ...]:
        out: List[torch.dtype] = []
        for name in (
            "float8_e4m3fn",
            "float8_e4m3fnuz",
            "float8_e5m2",
            "float8_e5m2fnuz",
        ):
            dt = getattr(torch, name, None)
            if isinstance(dt, torch.dtype):
                out.append(dt)
        return tuple(dict.fromkeys(out))

    @classmethod
    def coerce_metadata(
        cls: type[Self],
        device: Optional[Union[torch.device, str]] = None,
        *args: Any,
        metadata: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        meta = metadata or cls._get_tls_metadata()
        device_hint: Optional[Union[torch.device, str]] = device
        if device_hint is None and meta is not None:
            with contextlib.suppress(Exception):
                device_hint = torch.device(getattr(meta, "device"))
        dev = cls._device(device_hint)
        if meta is None:
            meta = DeviceMeta.for_device(dev)
        else:
            current_device = torch.device(getattr(meta, "device", dev))
            if current_device != dev:
                with contextlib.suppress(Exception):
                    setattr(meta, "device", dev)
                with contextlib.suppress(Exception):
                    f = getattr(meta, "refresh", None)
                    if callable(f):
                        f()
            else:
                ensure = getattr(meta, "ensure_device_info", None)
                if callable(ensure):
                    with contextlib.suppress(Exception):
                        ensure()
        if not getattr(meta, "float_dtypes", ()):
            refresh = getattr(meta, "refresh", None)
            if callable(refresh):
                with contextlib.suppress(Exception):
                    refresh()
        elif not getattr(meta, "int_dtypes", ()) or not getattr(
            meta, "float8_dtypes", ()
        ):
            refresh = getattr(meta, "refresh", None)
            if callable(refresh):
                with contextlib.suppress(Exception):
                    refresh()
        else:
            ensure = getattr(meta, "ensure_device_info", None)
            if callable(ensure):
                with contextlib.suppress(Exception):
                    ensure()
        cls._set_tls_metadata(meta)
        return meta

    @classmethod
    def float_amp_priority(
        cls: type[Self], device: torch.device
    ) -> Tuple[torch.dtype, ...]:
        meta = cls.coerce_metadata(device)
        candidates = getattr(meta, "float_dtypes", ())
        if candidates:
            return tuple(candidates)
        return (torch.float32,)

    @classmethod
    def integer_amp_priority(
        cls: type[Self], device: torch.device
    ) -> Tuple[torch.dtype, ...]:
        meta = cls.coerce_metadata(device)
        candidates = getattr(meta, "int_dtypes", ())
        if candidates:
            return tuple(candidates)
        return (torch.int64,)

    @classmethod
    def configure(
        cls: type[Self],
        model: Any | None = None,
        *args: Any,
        fp8_backend: Optional[str] = None,
        int_backend: Optional[str] = None,
        metadata: Any | None = None,
    ) -> None:
        backend = fp8_backend
        int_b = int_backend or (
            "ao"
            if isinstance(model, nn.Module)
            and any(
                getattr(model, a, False)
                for a in (
                    "__int8_training_qat__",
                    "__int8_training_ptq__",
                    "__int8_inference_ao__",
                )
            )
            else None
        )
        if backend is None and isinstance(model, nn.Module):
            if any(
                getattr(model, a, False)
                for a in (
                    "__fp8_training_te__",
                    "__fp8_inference_te__",
                    "__te_fp8_default__",
                )
            ):
                backend = "te"
            elif any(
                getattr(model, a, False)
                for a in ("__fp8_training_ao__", "__fp8_inference_ao__")
            ):
                backend = "ao"

        cls._preferred_fp8_backend = backend
        cls._preferred_int_backend = int_b
        meta = metadata
        device: Optional[torch.device] = None
        if meta is not None:
            with contextlib.suppress(Exception):
                device = torch.device(getattr(meta, "device"))
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
    def negotiate(
        cls: type[Self],
        candidates: Tuple[torch.dtype, ...],
        *args: Any,
        fallback: torch.dtype,
        logger: Optional[logging.Logger] = None,
        context: str = "autocast",
        device: Optional[torch.device] = None,
        meta: Any | None = None,
        decision_key: object = None,
        **kwargs: Any,
    ) -> torch.dtype:
        def _parse_margin() -> tuple[float, int]:
            p2 = kwargs.pop("safety_margin_pow2", None)
            if p2 is not None:
                return float(2 ** max(0, min(30, int(p2 or 3)))), int(p2 or 3)
            margin = float(kwargs.pop("safety_margin", 8.0))
            return (margin, int(round(math.log2(margin)))) if margin > 0 else (8.0, 3)

        safety_margin, safety_margin_pow2 = _parse_margin()
        underflow_override = normalize_underflow_action(
            kwargs.pop("underflow_action", kwargs.pop("underflow", None)),
            default=default_underflow_action(),
        )
        collect_checks = logger and (
            logger.isEnabledFor(logging.DEBUG) or logger.isEnabledFor(logging.INFO)
        )
        checks, selected, selected_from = [], None, "candidate"

        for dt in candidates:
            ok, why = _validate_dtype_safety(
                dt,
                meta,
                safety_margin=safety_margin,
                underflow_action=underflow_override,
            )
            if collect_checks:
                checks.append(
                    {
                        "dtype": _parse_dtype(dt),
                        "ok": bool(ok),
                        "reason": str(why),
                    }
                )
            if ok:
                selected = dt
                break

        fallback_order: Tuple[torch.dtype, ...] = ()
        if selected is None:
            selected_from = "fallback"
            fallback_order = (
                (fallback, torch.float32, torch.float64)
                if getattr(fallback, "is_floating_point", False)
                else (fallback, torch.int64, torch.float32, torch.float64)
            )
            for dt in fallback_order:
                ok, why = _validate_dtype_safety(
                    dt,
                    meta,
                    safety_margin=safety_margin,
                    underflow_action=underflow_override,
                )
                if collect_checks:
                    checks.append(
                        {
                            "dtype": _parse_dtype(dt),
                            "ok": bool(ok),
                            "reason": str(why),
                        }
                    )
                if ok:
                    selected = dt
                    break
            if selected is None:
                selected, selected_from = fallback, "unsafe-fallback"

        if logger is not None:
            level = "info" if selected_from != "candidate" else "debug"
            if logger.isEnabledFor(logging.INFO if level == "info" else logging.DEBUG):
                scale_key = (
                    getattr(meta, "has_scale", False),
                    getattr(meta, "has_nonfinite", False),
                    getattr(meta, "scale_max_abs", None),
                    getattr(meta, "underflow_action", None),
                )
                if decision_key is None:
                    decision_key = (
                        "amp",
                        str(context),
                        str(device),
                        tuple(_parse_dtype(x) for x in candidates),
                        _parse_dtype(fallback),
                        scale_key,
                        float(safety_margin),
                    )

                payload = {
                    "context": str(context),
                    "device": str(device),
                    "selected": _parse_dtype(selected),
                    "selected_from": selected_from,
                    "scale": _get_meta_stats(meta),
                }
                if collect_checks:
                    payload["checks"] = checks
                _log_negotiation(logger, decision_key, payload, level=level)
        return selected

    @classmethod
    def resolve_float_dtype(
        cls: type[Self],
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[Union[torch.dtype, str]] = None,
        metadata: Any | None = None,
    ) -> Optional[torch.dtype]:
        dev = cls._device(device)
        meta = cls.coerce_metadata(device=dev, metadata=metadata)
        disable = False
        if meta is not None:
            fn = getattr(meta, "is_disabled", None)
            if callable(fn):
                with contextlib.suppress(Exception):
                    disable = bool(fn())
        if disable:
            return None
        requested_dtype = _coerce_torch_dtype(dtype, torch.float16)
        candidates: Tuple[torch.dtype, ...] = (
            requested_dtype,
            cls._last_float_dtype,
        )
        extra = getattr(meta, "float_dtypes", None) if meta is not None else None
        if extra:
            with contextlib.suppress(Exception):
                candidates = tuple(
                    _coerce_torch_dtype(x, requested_dtype) for x in extra
                )
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
        cls: type[Self],
        device: Optional[Union[torch.device, str]] = None,
        *args: Any,
        metadata: Any | None = None,
        **kwargs: Any,
    ) -> contextlib.AbstractContextManager[None]:
        dev = cls._device(device)
        meta = cls.coerce_metadata(dev, metadata=metadata)
        amp_candidates = (
            tuple(getattr(meta, "float_dtypes", ()))
            if getattr(meta, "float_dtypes", ())
            else (torch.float32,)
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
        fp8_disable_reason: Optional[str] = None
        fp8_backend_used: Optional[str] = None

        backend = cls._fp8_backend(cls._preferred_fp8_backend, device=dev)
        wants_fp8 = backend is not None
        if (
            wants_fp8
            and getattr(meta, "has_scale", False)
            and not any(
                _validate_dtype_safety(dt, meta, safety_margin=2.0)[0]
                for dt in cls.float8_formats()
            )
        ):
            wants_fp8, fp8_disable_reason = False, "scale-exceeds-fp8"

        if wants_fp8:
            for b in (backend, "ao") if backend == "te" else (backend,):
                if ctxs := (
                    cls._nvidia_float8(dev, True)
                    if b == "te"
                    else cls._torchao_float8(True)
                ):
                    contexts.extend(ctxs)
                    fp8_backend_used = b
                    break
            if not fp8_backend_used:
                fp8_disable_reason = "fp8-backend-unavailable"

        requested_dtype = amp_dtype
        if (
            cls._last_float_dtype in amp_candidates
            and cls._last_float_dtype == amp_dtype
        ):
            requested_dtype = cls._last_float_dtype
        if requested_dtype is torch.float64:
            wants_fp8, fp8_disable_reason = (
                False,
                fp8_disable_reason or "master-fp64",
            )
        if (
            dev.type == "cuda"
            and requested_dtype is torch.bfloat16
            and not is_cuda_bf16_supported(dev)
        ):
            requested_dtype = torch.float16

        if dev.type == "cpu" and requested_dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            contexts.append(contextlib.nullcontext())
            cls._last_float_dtype = requested_dtype
        else:
            try:
                contexts.append(
                    torch.amp.autocast(
                        device_type=dev.type,
                        dtype=requested_dtype,
                        enabled=True,
                    )
                )
            except (RuntimeError, ValueError) as exc:
                _LOGGER.debug("Autocast.float torch.amp fallback: %s", exc)
                contexts.append(contextlib.nullcontext())
            cls._last_float_dtype = requested_dtype

        cls._set_tls_metadata(meta)
        if _LOGGER.isEnabledFor(logging.DEBUG):
            with contextlib.suppress(Exception):
                _LOGGER.debug(
                    "Autocast.context(float): %s",
                    json.dumps(
                        {
                            "device": str(dev),
                            "amp_dtype": _parse_dtype(amp_dtype),
                            "fp8_used": fp8_backend_used,
                            "scale": _get_meta_stats(meta),
                        },
                        sort_keys=True,
                        default=str,
                    ),
                )
        with contextlib.ExitStack() as stack:
            for ctx in contexts:
                stack.enter_context(ctx)
            yield

    @classmethod
    @contextlib.contextmanager
    def suspend(
        cls: type[Self], device: Optional[Union[torch.device, str]] = None
    ) -> contextlib.AbstractContextManager[None]:
        dev = cls._device(device)
        with contextlib.ExitStack() as stack:
            with contextlib.suppress(Exception):
                stack.enter_context(
                    torch.amp.autocast(device_type=dev.type, enabled=False)
                ) or stack.enter_context(contextlib.nullcontext())
            yield

    @classmethod
    @contextlib.contextmanager
    def integer(
        cls: type[Self],
        device: Optional[Union[torch.device, str]] = None,
        *args: Any,
        metadata: Any | None = None,
        **kwargs: Any,
    ) -> contextlib.AbstractContextManager[None]:
        dev = cls._device(device)
        meta = cls.coerce_metadata(dev, metadata=metadata)
        int_candidates = tuple(getattr(meta, "int_dtypes", ())) or (torch.int64,)
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

        int_backend_used: Optional[str] = None
        contexts: List[contextlib.AbstractContextManager[None]] = []
        if wants_int4:
            try:
                if contexts := cls._torchao_int4(dev, True):
                    cls._preferred_int_backend, int_backend_used = "ao", "ao"
            except Exception as exc:
                _LOGGER.debug("Autocast INT4 enable failed: %s", exc)
        if not contexts and wants_int8:
            backend = cls._int_backend(cls._preferred_int_backend, device=dev)
            contexts = cls._torchao_int8(dev, True) if backend else []
            if contexts:
                int_backend_used = backend
            elif backend == "te" and cls._int_backend("ao", device=dev) == "ao":
                if contexts := cls._torchao_int8(dev, True):
                    int_backend_used = "ao"

        if not contexts:
            contexts.append(contextlib.nullcontext())
        with contextlib.ExitStack() as stack:
            for ctx in contexts:
                stack.enter_context(ctx)
            cls._last_int_dtype = int_dtype
            cls._set_tls_metadata(meta)
            if _LOGGER.isEnabledFor(logging.DEBUG):
                with contextlib.suppress(Exception):
                    _LOGGER.debug(
                        "Autocast.context(int): %s",
                        json.dumps(
                            {
                                "device": str(dev),
                                "int_dtype": _parse_dtype(int_dtype),
                                "int_backend_used": int_backend_used,
                                "scale": _get_meta_stats(meta),
                            },
                            sort_keys=True,
                            default=str,
                        ),
                    )
            yield


class Quantization:
    @staticmethod
    def is_qat_available() -> bool:
        _import_torchao_quantization()
        return bool(_qp is not None)

    @staticmethod
    def is_ptq_available() -> bool:
        _import_torchao_quantization()
        return bool(
            _PTQ_IMPL is not None and _Int8DynamicActivationInt8WeightConfig is not None
        )

    @staticmethod
    def _prepare_qat(
        model: nn.Module,
        *args: Any,
        dynamic_activations: bool = True,
        group_size: int = 128,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> Any:
        _ = args, kwargs
        _import_torchao_quantization()
        if _qp is None:
            raise RuntimeError("torchao.quantization.quant_primitives unavailable")

        try:
            from torchao.quantization.fake_quant import (
                FakeQuantizeConfig,
                Int8ActivationConfig,
                Int8WeightConfig,
            )
            from torchao.quantization.fake_quant import prepare_qat_ as _prepare_qat

            cfg = FakeQuantizeConfig(
                activation=Int8ActivationConfig(dynamic=bool(dynamic_activations)),
                weight=Int8WeightConfig(group_size=int(group_size)),
            )
            _prepare_qat(model, cfg)
            clear_model_cache(model)
            return cfg
        except Exception as exc:
            raise RuntimeError(f"torchao QAT prepare unavailable: {exc}") from exc

    @staticmethod
    def _apply_ptq(
        model: nn.Module,
        *args: Any,
        dynamic_activations: bool = True,
        group_size: int = 128,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> tuple[nn.Module, bool, str]:
        _ = args, kwargs
        _import_torchao_quantization()
        if _PTQ_IMPL is None:
            return _is_ptq_unavailable(model)
        if _Int8DynamicActivationInt8WeightConfig is None:
            return _is_ptq_unavailable(model)
        cfg: Any
        why: str
        if bool(dynamic_activations):
            cfg = _Int8DynamicActivationInt8WeightConfig(group_size=int(group_size))
            why = "int8_dynamic_act_int8_weight"
        else:
            if _Int8WeightOnlyConfig is None:
                return (model, False, "Int8WeightOnlyConfig unavailable")
            cfg = _Int8WeightOnlyConfig(group_size=int(group_size))
            why = "int8_weight_only"
        try:
            _log_info(logger, f"[INT8][PTQ] applying {why} (group={group_size})")
            _PTQ_IMPL(model, cfg)
            clear_model_cache(model)
            return (model, True, why)
        except Exception as exc:
            return (model, False, f"PTQ failed: {exc}")

    @classmethod
    def enable_qat(
        cls: type[Self],
        model: nn.Module,
        *args: Any,
        dynamic_activations: bool = True,
        group_size: int = 128,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> tuple[nn.Module, bool, str]:
        _ = args, kwargs
        if not cls.is_qat_available():
            return (model, False, "QAT backend unavailable")
        try:
            cls._prepare_qat(
                model,
                dynamic_activations=dynamic_activations,
                group_size=group_size,
                logger=logger,
            )
            setattr(model, "__int8_training_qat__", True)
            return (model, True, "QAT-prepare")
        except Exception as exc:
            return (model, False, f"QAT prepare failed: {exc}")

    @classmethod
    def _enable_ptq(
        cls: type[Self],
        model: nn.Module,
        *args: Any,
        dynamic_activations: bool = True,
        group_size: int = 128,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> tuple[nn.Module, bool, str]:
        _ = args, kwargs
        return cls._apply_ptq(
            model,
            dynamic_activations=dynamic_activations,
            group_size=group_size,
            logger=logger,
        )

    @classmethod
    def enable_int8_training(
        cls: type[Self],
        model: nn.Module,
        *args: Any,
        dynamic_activations: bool = True,
        group_size: int = 128,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> tuple[nn.Module, bool, str]:
        _ = args, kwargs
        if getattr(model, "__int8_training_qat__", False) or getattr(
            model, "__int8_training_ptq__", False
        ):
            return (model, True, "already-enabled")
        last_err: Optional[Exception] = None
        if cls.is_qat_available():
            try:
                cls._prepare_qat(
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


def is_precision_exempted(module: object) -> bool:
    return bool(getattr(module, "__enn_precision_exempt__", False))


def _set_requires_grad(module: nn.Module, name: str, data: torch.Tensor, *, requires_grad: bool) -> None:
    setattr(module, name, nn.Parameter(data, requires_grad=requires_grad))


def cast_float_dtype(model: object, dtype: torch.dtype) -> None:
    if not isinstance(dtype, torch.dtype):
        return
    try:
        if not torch.is_floating_point(torch.empty((), dtype=dtype)):
            return
    except Exception:
        return

    with torch.no_grad():
        for mod in getattr(model, "modules", lambda: [])():
            if is_precision_exempted(mod):
                continue
            params = getattr(mod, "_parameters", None)
            if params:
                for name, p in list(params.items()):
                    if p is None or not isinstance(p, torch.Tensor):
                        continue
                    if (not p.is_floating_point()) or p.dtype == dtype:
                        continue
                    params[name] = torch.nn.Parameter(
                        p.detach().to(dtype),
                        requires_grad=bool(getattr(p, "requires_grad", True)),
                    )
            bufs = getattr(mod, "_buffers", None)
            if bufs:
                for name, b in list(bufs.items()):
                    if b is None or not isinstance(b, torch.Tensor):
                        continue
                    if (not b.is_floating_point()) or b.dtype == dtype:
                        continue
                    if b.dtype is torch.float64 and dtype is not torch.float64:
                        continue
                    bufs[name] = b.detach().to(dtype)


def cast_batchnorm_buffers_dtype(module: object, dtype: torch.dtype | None) -> None:
    if dtype is None or not isinstance(dtype, torch.dtype):
        return
    with torch.no_grad():
        for mod in getattr(module, "modules", lambda: [])():
            if is_precision_exempted(mod):
                continue
            if isinstance(
                mod,
                (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.BatchNorm3d,
                    torch.nn.SyncBatchNorm,
                ),
            ):
                for name, buf in getattr(mod, "_buffers", {}).items():
                    if buf is None or not isinstance(buf, torch.Tensor):
                        continue
                    if (not buf.is_floating_point()) or buf.dtype == dtype:
                        continue
                    with contextlib.suppress(Exception):
                        mod._buffers[name] = buf.to(dtype=dtype)


def get_layernorm_dtype(device: torch.device | str) -> torch.dtype:
    device = torch.device(device) if not isinstance(device, torch.device) else device
    try:
        meta = Autocast.coerce_metadata(device)
        cands = tuple(getattr(meta, "float_dtypes", ())) if meta is not None else ()
        if not cands:
            cands = (torch.float32,)
        chosen = Autocast.negotiate(
            tuple(cands),
            fallback=torch.float64,
            context="cpu.layernorm",
            device=device,
            meta=meta,
        )
        return torch.float64 if chosen == torch.float64 else torch.float32
    except Exception:
        return torch.float32


def preload_layers(model: nn.Module, device: torch.device | str) -> None:
    device = torch.device(device) if not isinstance(device, torch.device) else device
    from .tensor import is_meta_or_fake_tensor

    for module in model.modules():
        if not isinstance(module, nn.LayerNorm):
            continue
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        requires_grad_w = bool(getattr(weight, "requires_grad", True))
        requires_grad_b = bool(getattr(bias, "requires_grad", True))

        if device.type == "cpu":
            target_dtype = get_layernorm_dtype(device)
        else:
            target_dtype = None
            for tensor in (weight, bias):
                if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                    if not is_meta_or_fake_tensor(tensor):
                        target_dtype = tensor.dtype
                        break
            if target_dtype is None:
                target_dtype = torch.get_default_dtype()

        if module.elementwise_affine:
            if not isinstance(weight, torch.Tensor) or is_meta_or_fake_tensor(weight):
                data = torch.ones(module.normalized_shape, device=device, dtype=target_dtype)
                _set_requires_grad(module, "weight", data, requires_grad=requires_grad_w)
                weight = module.weight
            if not isinstance(bias, torch.Tensor) or is_meta_or_fake_tensor(bias):
                data = torch.zeros(module.normalized_shape, device=device, dtype=target_dtype)
                _set_requires_grad(module, "bias", data, requires_grad=requires_grad_b)
                bias = module.bias

        if device.type == "cpu":
            if isinstance(weight, torch.Tensor) and weight.dtype != target_dtype:
                data = weight.to(device=device, dtype=target_dtype)
                _set_requires_grad(module, "weight", data, requires_grad=requires_grad_w)
                weight = module.weight
            if isinstance(bias, torch.Tensor) and bias.dtype != target_dtype:
                data = bias.to(device=device, dtype=target_dtype)
                _set_requires_grad(module, "bias", data, requires_grad=requires_grad_b)
                bias = module.bias
        elif (
            isinstance(weight, torch.Tensor)
            and isinstance(bias, torch.Tensor)
            and weight.is_floating_point()
            and bias.is_floating_point()
            and (bias.dtype != weight.dtype)
        ):
            data = bias.to(device=device, dtype=weight.dtype)
            _set_requires_grad(module, "bias", data, requires_grad=requires_grad_b)


def validate_model_dtype_unity(model: nn.Module, device: torch.device | str) -> None:
    device = torch.device(device) if not isinstance(device, torch.device) else device
    mismatches: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.LayerNorm):
            continue
        tensors = [
            ("weight", getattr(module, "weight", None)),
            ("bias", getattr(module, "bias", None)),
        ]
        expected: torch.dtype | None = (
            get_layernorm_dtype(device) if device.type == "cpu" else None
        )

        for label, tensor in tensors:
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                continue
            if expected is None:
                expected = tensor.dtype
            elif tensor.dtype != expected:
                module_name = name or module.__class__.__name__
                mismatches.append(
                    f"{module_name}.{label} has dtype {tensor.dtype} (expected {expected})"
                )

        if expected is not None and device.type != "cpu":
            dtypes = {
                tensor.dtype
                for _, tensor in tensors
                if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
            }
            if len(dtypes) > 1:
                module_name = name or module.__class__.__name__
                mismatches.append(
                    f"{module_name} parameters disagree on dtype: {sorted(str(d) for d in dtypes)}"
                )

    if mismatches:
        raise RuntimeError(
            "LayerNorm parameter dtype mismatch detected:\n" + "\n".join(mismatches)
        )


def unify_model_dtype(model: nn.Module, prefer: torch.dtype | None = None) -> torch.dtype | None:
    dtypes = {p.dtype for p in model.parameters() if p is not None}
    if len(dtypes) <= 1:
        return None

    if prefer is not None:
        tgt = prefer
    elif torch.bfloat16 in dtypes:
        tgt = torch.bfloat16
    elif torch.float16 in dtypes:
        tgt = torch.float16
    else:
        tgt = torch.float32

    for mod in model.modules():
        params = getattr(mod, "_parameters", None)
        if not params:
            continue
        for name, p in list(params.items()):
            if p is None or p.dtype == tgt:
                continue
            params[name] = torch.nn.Parameter(p.detach().to(tgt), requires_grad=p.requires_grad)

    return tgt
