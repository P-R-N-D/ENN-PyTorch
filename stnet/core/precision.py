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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from .system import (
    get_device,
    get_device_stats,
    is_cuda_bf16_supported,
    is_float8_supported,
    is_int8_supported,
)

from ..data.schemas import default_underflow_action, normalize_underflow_action

_LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Negotiation logging (bounded dedupe)
# -----------------------------------------------------------------------------

_NEGO_LOGGED_KEYS: "OrderedDict[object, None]" = OrderedDict()
_NEGO_LOGGED_MAX: int = 256
_NEGO_LOGGED_LOCK = threading.Lock()


def _log_negotiate_once(
    logger: Optional[logging.Logger],
    key: object,
    payload: Dict[str, Any],
    *,
    level: str = "debug",
) -> None:
    """Log a structured negotiation decision once per key."""
    lg = logger if logger is not None else _LOGGER
    lvl = logging.INFO if str(level).lower() == "info" else logging.DEBUG
    try:
        if not lg.isEnabledFor(lvl):
            return
    except Exception:
        pass

    if key is None:
        key = (payload.get("context"), payload.get("device"), payload.get("selected"))

    with _NEGO_LOGGED_LOCK:
        if key in _NEGO_LOGGED_KEYS:
            with contextlib.suppress(Exception):
                _NEGO_LOGGED_KEYS.move_to_end(key)
            return
        _NEGO_LOGGED_KEYS[key] = None
        with contextlib.suppress(Exception):
            while len(_NEGO_LOGGED_KEYS) > int(_NEGO_LOGGED_MAX):
                _NEGO_LOGGED_KEYS.popitem(last=False)

    try:
        msg = "[AMP][NEGOTIATE] " + json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        msg = f"[AMP][NEGOTIATE] {payload}"

    with contextlib.suppress(Exception):
        if lvl == logging.INFO:
            lg.info(msg)
        else:
            lg.debug(msg)


# -----------------------------------------------------------------------------
# Scale safety checks
# -----------------------------------------------------------------------------


def _dtype_short(dtype: Any) -> str:
    if isinstance(dtype, torch.dtype):
        return str(dtype).split(".")[-1]
    return str(dtype)


def _meta_scale_summary(meta: Any | None) -> Dict[str, Any]:
    if meta is None:
        return {}

    def _safe(x: Any) -> Any:
        if x is None:
            return None
        if isinstance(x, (bool, int, str, float)):
            return x
        with contextlib.suppress(Exception):
            return float(x)
        with contextlib.suppress(Exception):
            return str(x)
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
    meta: Any | None,
    *args: Any,
    safety_margin: float = 8.0,
    underflow_action: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[bool, str]:
    """Return (ok, reason) for whether dtype can represent dataset scale."""
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
        underflow_action if underflow_action is not None else getattr(meta, "underflow_action", None),
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
            return (False, f"overflow(max_abs={max_abs_f:.6g},limit={overflow_limit:.6g})")

        min_pos = getattr(meta, "scale_min_positive", None)
        if action == "forbid" and min_pos is not None:
            try:
                min_pos_f = float(min_pos)
            except Exception:
                return False, "min-pos-not-float"
            if math.isfinite(min_pos_f) and min_pos_f > 0.0:
                underflow_limit = float(info.tiny) * max(1.0, float(safety_margin))
                if min_pos_f < underflow_limit:
                    return (False, f"underflow(min_pos={min_pos_f:.6g},limit={underflow_limit:.6g})")
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


def is_scale_safe(
    dtype: torch.dtype,
    meta: Any | None,
    *args: Any,
    safety_margin: float = 8.0,
    underflow_action: Optional[str] = None,
    **kwargs: Any,
) -> bool:
    """Return True if `dtype` can represent the dataset scale without overflow."""
    ok, _ = _scale_safety_check(dtype, meta, safety_margin=safety_margin, underflow_action=underflow_action)
    return ok


# -----------------------------------------------------------------------------
# Lightweight metadata (backend-friendly fallback)
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class DeviceMeta:
    """Minimal metadata object used when a full Dataset is not available.

    This mirrors the subset of attributes/methods used by Autocast/PrecisionPolicy
    without importing stnet.data.pipeline.Dataset (avoids backend->data cycles).
    """

    device: torch.device
    device_type: str = "cpu"
    cuda_cc: Optional[Tuple[int, int]] = None
    float_dtypes: Tuple[torch.dtype, ...] = tuple()
    int_dtypes: Tuple[torch.dtype, ...] = tuple()
    float8_dtypes: Tuple[torch.dtype, ...] = tuple()
    int_quant_bits: Optional[int] = None

    # Scale / negotiation metadata (optional)
    has_scale: bool = False
    has_nonfinite: bool = False
    scale_max_abs: Optional[float] = None
    scale_min_value: Optional[float | int] = None
    scale_max_value: Optional[float | int] = None
    scale_min_positive: Optional[float] = None
    scale_is_integral: Optional[bool] = None
    is_negotiable: Optional[bool] = None
    underflow_action: str = "warn"

    def ensure_device_info(self) -> "DeviceMeta":
        self._refresh_device_info()
        return self

    def refresh(self) -> "DeviceMeta":
        self._refresh_device_info()
        # Fill candidates from backend device stats if not explicitly set.
        ds = get_device_stats(self.device)
        if not self.float_dtypes:
            self.float_dtypes = tuple(ds.float_dtypes)
        if not self.int_dtypes:
            self.int_dtypes = tuple(ds.int_dtypes)
        if self.int_quant_bits is None:
            self.int_quant_bits = int(ds.int_quant_bits)
        return self

    def is_disabled(self) -> bool:
        return False

    def _refresh_device_info(self) -> None:
        ds = get_device_stats(self.device)
        self.device = ds.device
        self.device_type = ds.device_type
        self.cuda_cc = ds.cuda_cc

    @classmethod
    def for_device(cls, device: Union[torch.device, str]) -> "DeviceMeta":
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


# -----------------------------------------------------------------------------
# Autocast context helpers (moved from model.fused)
# -----------------------------------------------------------------------------


class Autocast:
    _preferred_fp8_backend: Optional[str] = None
    _preferred_int_backend: Optional[str] = None
    _last_float_dtype: torch.dtype = torch.float32
    _last_int_dtype: torch.dtype = torch.int64
    _metadata: Any | None = None  # best-effort legacy snapshot
    _metadata_tls = threading.local()  # per-thread metadata to avoid cross-thread mutation

    @classmethod
    def _get_tls_metadata(cls) -> Any | None:
        return getattr(cls._metadata_tls, "meta", None)

    @classmethod
    def _set_tls_metadata(cls, meta: Any | None) -> None:
        setattr(cls._metadata_tls, "meta", meta)
        # Keep a best-effort snapshot for any legacy access paths.
        cls._metadata = meta

    @classmethod
    def metadata(cls) -> Any | None:
        """Thread-local metadata accessor (preferred over `Autocast._metadata`)."""
        return cls._get_tls_metadata()

    @staticmethod
    def _device(device: Optional[Union[torch.device, str]] = None) -> torch.device:
        if device is None:
            return get_device()
        if isinstance(device, torch.device):
            return device
        return torch.device(device)

    @classmethod
    def _fp8_backend(
        cls,
        preferred: Optional[str],
        *args: Any,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        dev = device if device is not None else cls._device(None)
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
                        raise AttributeError("transformer_engine.fp8_autocast missing")
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
        cls,
        preferred: Optional[str],
        *args: Any,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        dev = device if device is not None else cls._device(None)
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
                        raise AttributeError("transformer_engine.int8_autocast missing")
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

    @staticmethod
    def float8_formats() -> Tuple[torch.dtype, ...]:
        # Prefer native dtypes when present.
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
        # De-dupe while preserving order.
        return tuple(dict.fromkeys(out))

    @classmethod
    def coerce_metadata(
        cls,
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

        # Ensure dtype lists exist.
        if not getattr(meta, "float_dtypes", ()):
            refresh = getattr(meta, "refresh", None)
            if callable(refresh):
                with contextlib.suppress(Exception):
                    refresh()
        elif not getattr(meta, "int_dtypes", ()) or not getattr(meta, "float8_dtypes", ()):
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
    def float_amp_priority(cls, device: torch.device) -> Tuple[torch.dtype, ...]:
        meta = cls.coerce_metadata(device)
        candidates = getattr(meta, "float_dtypes", ())
        if candidates:
            return tuple(candidates)
        return (torch.float32,)

    @classmethod
    def integer_amp_priority(cls, device: torch.device) -> Tuple[torch.dtype, ...]:
        meta = cls.coerce_metadata(device)
        candidates = getattr(meta, "int_dtypes", ())
        if candidates:
            return tuple(candidates)
        return (torch.int64,)

    @classmethod
    def configure(
        cls,
        model: Any | None = None,
        *,
        fp8_backend: Optional[str] = None,
        int_backend: Optional[str] = None,
        metadata: Any | None = None,
    ) -> None:
        backend = fp8_backend
        int_b = int_backend
        if backend is None and isinstance(model, nn.Module):
            if any(getattr(model, attr, False) for attr in ("__fp8_training_te__", "__fp8_inference_te__", "__te_fp8_default__")):
                backend = "te"
            elif any(getattr(model, attr, False) for attr in ("__fp8_training_ao__", "__fp8_inference_ao__")):
                backend = "ao"

        if int_b is None and isinstance(model, nn.Module):
            if any(getattr(model, attr, False) for attr in ("__int8_training_qat__", "__int8_training_ptq__", "__int8_inference_ao__")):
                int_b = "ao"

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
        cls,
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
        # Safety margin: can be passed as safety_margin or safety_margin_pow2.
        raw_pow2 = kwargs.pop("safety_margin_pow2", None)
        safety_margin_pow2: Optional[int] = None
        if raw_pow2 is not None:
            with contextlib.suppress(Exception):
                safety_margin_pow2 = int(raw_pow2)
            if safety_margin_pow2 is None:
                safety_margin_pow2 = 3
            if safety_margin_pow2 < 0:
                safety_margin_pow2 = 0
            if safety_margin_pow2 > 30:
                safety_margin_pow2 = 30
            safety_margin = float(2 ** safety_margin_pow2)
        else:
            raw_margin = kwargs.pop("safety_margin", 8.0)
            with contextlib.suppress(Exception):
                safety_margin = float(raw_margin)
            if "safety_margin" not in locals() or (not math.isfinite(safety_margin)) or safety_margin <= 0.0:
                safety_margin = 8.0
            with contextlib.suppress(Exception):
                n = int(round(math.log2(safety_margin)))
                if n >= 0:
                    ref = float(2 ** n)
                    if abs(ref - safety_margin) / max(abs(safety_margin), 1.0) < 1e-12:
                        safety_margin_pow2 = n

        raw_underflow = kwargs.pop("underflow_action", None)
        if raw_underflow is None:
            raw_underflow = kwargs.pop("underflow", None)
        underflow_override: Optional[str] = None
        if raw_underflow is not None:
            underflow_override = normalize_underflow_action(raw_underflow, default=default_underflow_action())

        collect_checks = False
        if logger is not None:
            try:
                collect_checks = logger.isEnabledFor(logging.DEBUG) or logger.isEnabledFor(logging.INFO)
            except Exception:
                collect_checks = True

        checks: List[Dict[str, Any]] = [] if collect_checks else []
        selected: Optional[torch.dtype] = None
        selected_from: str = "candidate"

        for dt in candidates:
            ok, why = _scale_safety_check(dt, meta, safety_margin=safety_margin, underflow_action=underflow_override)
            if collect_checks:
                checks.append({"dtype": _dtype_short(dt), "ok": bool(ok), "reason": str(why)})
            if ok:
                selected = dt
                break

        dev_type = str(getattr(device, "type", "")) if device is not None else ""
        dev_index = int(getattr(device, "index", -1)) if (device is not None and getattr(device, "index", None) is not None) else -1
        device_str = f"{dev_type}:{dev_index}" if dev_type else ""

        fallback_order: Tuple[torch.dtype, ...] = ()
        if selected is None:
            selected_from = "fallback"
            if getattr(fallback, "is_floating_point", False):
                fallback_order = (fallback, torch.float32, torch.float64)
            else:
                fallback_order = (fallback, torch.int64, torch.float32, torch.float64)
            for dt in fallback_order:
                ok, why = _scale_safety_check(dt, meta, safety_margin=safety_margin, underflow_action=underflow_override)
                if collect_checks:
                    checks.append({"dtype": _dtype_short(dt), "ok": bool(ok), "reason": str(why)})
                if ok:
                    selected = dt
                    break
            if selected is None:
                selected = fallback
                selected_from = "unsafe-fallback"

        if logger is not None:
            level = "info" if selected_from != "candidate" else "debug"
            lvl = logging.INFO if level == "info" else logging.DEBUG
            should_log = True
            with contextlib.suppress(Exception):
                should_log = bool(logger.isEnabledFor(lvl))
            if should_log:
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
                    "fallback_order": ([_dtype_short(x) for x in fallback_order] if fallback_order else []),
                    "checks": (checks if collect_checks else []),
                    "safety_margin": safety_margin,
                    "safety_margin_pow2": safety_margin_pow2,
                    "underflow_action_override": underflow_override,
                    "scale": _meta_scale_summary(meta),
                }
                _log_negotiate_once(logger, decision_key, payload, level=level)

        return selected

    @classmethod
    def resolve_float_dtype(
        cls,
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

        def _coerce_dt(x: Any, default: torch.dtype) -> torch.dtype:
            if x is None:
                return default
            if isinstance(x, torch.dtype):
                return x
            s = str(x).strip().replace("torch.", "")
            return getattr(torch, s, default)

        requested_dtype = _coerce_dt(dtype, torch.float16)
        candidates: Tuple[torch.dtype, ...] = (requested_dtype, cls._last_float_dtype)
        extra = getattr(meta, "float_dtypes", None) if meta is not None else None
        if extra:
            with contextlib.suppress(Exception):
                candidates = tuple(_coerce_dt(x, requested_dtype) for x in extra)

        chosen = cls.negotiate(candidates, fallback=requested_dtype, device=dev, meta=meta, context="autocast-float")
        return chosen

    @classmethod
    @contextlib.contextmanager
    def float(
        cls,
        device: Optional[Union[torch.device, str]] = None,
        *args: Any,
        metadata: Any | None = None,
        **kwargs: Any,
    ) -> contextlib.AbstractContextManager[None]:
        dev = cls._device(device)
        meta = cls.coerce_metadata(dev, metadata=metadata)
        amp_candidates = tuple(getattr(meta, "float_dtypes", ())) if getattr(meta, "float_dtypes", ()) else (torch.float32,)
        amp_dtype = cls.negotiate(
            amp_candidates,
            fallback=torch.float64,
            logger=_LOGGER,
            context="float",
            device=dev,
            meta=meta,
        )

        contexts: List[contextlib.AbstractContextManager[None]] = []

        debug = _LOGGER.isEnabledFor(logging.DEBUG)
        fp8_disable_reason: Optional[str] = None
        fp8_backend_requested: Optional[str] = None
        fp8_backend_used: Optional[str] = None
        fp8_checks: Dict[str, Any] = {}

        backend = cls._fp8_backend(cls._preferred_fp8_backend, device=dev)
        fp8_backend_requested = backend
        float8_dtypes = tuple(getattr(meta, "float8_dtypes", ())) if getattr(meta, "float8_dtypes", ()) else cls.float8_formats()

        wants_fp8 = backend is not None
        if wants_fp8 and getattr(meta, "has_scale", False):
            fp8_supported = False
            for dt in float8_dtypes:
                ok, why = _scale_safety_check(dt, meta, safety_margin=2.0)
                if debug:
                    fp8_checks[_dtype_short(dt)] = {"ok": bool(ok), "reason": str(why)}
                if ok:
                    fp8_supported = True
            if not fp8_supported:
                wants_fp8 = False
                fp8_disable_reason = "scale-exceeds-fp8"
                _LOGGER.debug("Autocast FP8 disabled on %s: data scale exceeds float8 range", dev.type)

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
                _LOGGER.debug("Autocast FP8 backend '%s' unsupported; disabling", backend)
                cls._preferred_fp8_backend = None
                fp8_disable_reason = "fp8-backend-unsupported"

        requested_dtype = amp_dtype
        if isinstance(cls._last_float_dtype, torch.dtype) and cls._last_float_dtype in amp_candidates and cls._last_float_dtype == amp_dtype:
            requested_dtype = cls._last_float_dtype

        if requested_dtype is torch.float64:
            wants_fp8 = False
            fp8_disable_reason = fp8_disable_reason or "master-fp64"

        if dev.type == "cuda" and requested_dtype is torch.bfloat16:
            bf16_ok = is_cuda_bf16_supported(dev)
            if not bf16_ok:
                _LOGGER.debug("Autocast.float falling back to fp16 on CUDA device without bf16 support")
                requested_dtype = torch.float16

        if dev.type == "cpu" and requested_dtype not in (torch.bfloat16, torch.float16):
            contexts.append(contextlib.nullcontext())
            cls._last_float_dtype = requested_dtype
        else:
            try:
                ctx = torch.amp.autocast(device_type=dev.type, dtype=requested_dtype, enabled=True)
                contexts.append(ctx)
            except (RuntimeError, ValueError) as exc:
                _LOGGER.debug("Autocast.float torch.amp fallback on %s: %s", dev.type, exc)
                contexts.append(contextlib.nullcontext())
                cls._last_float_dtype = requested_dtype
            else:
                cls._last_float_dtype = requested_dtype

        cls._set_tls_metadata(meta)

        if debug:
            with contextlib.suppress(Exception):
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

        with contextlib.ExitStack() as stack:
            for ctx in contexts:
                stack.enter_context(ctx)
            yield

    @classmethod
    @contextlib.contextmanager
    def suspend(cls, device: Optional[Union[torch.device, str]] = None) -> contextlib.AbstractContextManager[None]:
        dev = cls._device(device)
        with contextlib.ExitStack() as stack:
            try:
                stack.enter_context(torch.amp.autocast(device_type=dev.type, enabled=False))
            except (RuntimeError, ValueError):
                stack.enter_context(contextlib.nullcontext())
            yield

    @classmethod
    @contextlib.contextmanager
    def integer(
        cls,
        device: Optional[Union[torch.device, str]] = None,
        *args: Any,
        metadata: Any | None = None,
        **kwargs: Any,
    ) -> contextlib.AbstractContextManager[None]:
        dev = cls._device(device)
        meta = cls.coerce_metadata(dev, metadata=metadata)
        int_candidates = tuple(getattr(meta, "int_dtypes", ())) if getattr(meta, "int_dtypes", ()) else (torch.int64,)
        int_dtype = cls.negotiate(int_candidates, fallback=torch.int64, logger=_LOGGER, context="int", device=dev, meta=meta)

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
                with contextlib.suppress(Exception):
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
            yield

    # ---- Backend-specific context constructors ----

    @classmethod
    def _nvidia_float8(cls, device: torch.device, enabled: bool) -> List[AbstractContextManager[None]]:
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
    def _torchao_float8(cls, enabled: bool) -> List[AbstractContextManager[None]]:
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
    def _torchao_int8(cls, device: torch.device, enabled: bool) -> List[AbstractContextManager[None]]:
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
    def _torchao_int4(cls, device: torch.device, enabled: bool) -> List[AbstractContextManager[None]]:
        contexts: List[AbstractContextManager[None]] = []
        if not enabled:
            return contexts
        try:
            quant_mod = importlib.import_module("torchao.quantization")
            int4_autocast = getattr(quant_mod, "int4_autocast", None)
            if callable(int4_autocast):
                contexts.append(int4_autocast(enabled=True))
            else:
                raise AttributeError("torchao.quantization.int4_autocast missing")
        except Exception as exc:
            _LOGGER.debug("Autocast INT4 torchao failed: %s", exc)
        return contexts

    @classmethod
    def _torchao_int8(cls, device: torch.device, enabled: bool) -> List[AbstractContextManager[None]]:
        # Alias to the internal helper; preserves historical behavior.
        return cls._torchao_int8_autocast(device, enabled)

    @classmethod
    def _torchao_int8_autocast(cls, device: torch.device, enabled: bool) -> List[AbstractContextManager[None]]:
        # Use preferred backend if set; otherwise try AO.
        if cls._preferred_int_backend is None:
            cls._preferred_int_backend = "ao"
        return cls._torchao_int8(device, enabled)


# -----------------------------------------------------------------------------
# End-to-end precision policy
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class PrecisionPolicy:
    """End-to-end precision policy used by runtime."""

    master_float: torch.dtype = torch.float32
    amp_dtype: Optional[torch.dtype] = None

    fsdp_param_dtype: torch.dtype = torch.float32
    fsdp_reduce_dtype: torch.dtype = torch.float32
    fsdp_output_dtype: torch.dtype = torch.float32

    bn_buffers_dtype: torch.dtype = torch.float32
    underflow_action: str = "warn"

    @property
    def amp_float(self) -> Optional[torch.dtype]:
        return self.amp_dtype

    @classmethod
    def from_metadata(
        cls,
        device: Union[torch.device, str],
        metadata: Any | None,
        *,
        logger: Optional[logging.Logger] = None,
        safety_margin: float = 8.0,
    ) -> "PrecisionPolicy":
        dev = torch.device(device)
        meta = metadata
        if meta is None:
            meta = DeviceMeta.for_device(dev)
        else:
            with contextlib.suppress(Exception):
                setattr(meta, "device", dev)
                f = getattr(meta, "refresh", None)
                if callable(f):
                    f()

        action = normalize_underflow_action(getattr(meta, "underflow_action", None), default=default_underflow_action())
        with contextlib.suppress(Exception):
            setattr(meta, "underflow_action", action)

        is_negotiable = bool(getattr(meta, "is_negotiable", False))
        safety = float(safety_margin)

        master_float = torch.float64
        amp_dtype: Optional[torch.dtype] = None

        if dev.type == "cuda":
            if is_negotiable and is_scale_safe(torch.float32, meta, safety_margin=safety):
                master_float = torch.float32

            if is_cuda_bf16_supported(dev) and is_scale_safe(torch.bfloat16, meta, safety_margin=safety):
                amp_dtype = torch.bfloat16
            elif is_scale_safe(torch.float16, meta, safety_margin=safety):
                amp_dtype = torch.float16
        elif dev.type == "cpu":
            master_float = torch.float32 if is_negotiable else torch.float64
        elif dev.type == "xpu":
            master_float = torch.float32 if is_negotiable else torch.float64
            amp_dtype = torch.bfloat16
        elif dev.type == "mps":
            master_float = torch.float32 if is_negotiable else torch.float64
            amp_dtype = torch.float16
        else:
            if is_scale_safe(torch.float32, meta, safety_margin=safety):
                master_float = torch.float32

        bn_dtype = master_float
        fsdp_param_dtype = master_float
        fsdp_reduce_dtype = master_float
        fsdp_output_dtype = master_float

        if master_float == torch.float32 and amp_dtype is not None:
            fsdp_param_dtype = amp_dtype
            fsdp_reduce_dtype = amp_dtype
            fsdp_output_dtype = amp_dtype

        if logger is not None:
            with contextlib.suppress(Exception):
                logger.info(
                    "[PrecisionPolicy] device=%s master=%s amp=%s fsdp=(param=%s, reduce=%s, out=%s) bn=%s underflow=%s negotiable=%s safety=%s",
                    str(dev),
                    str(master_float),
                    str(amp_dtype),
                    str(fsdp_param_dtype),
                    str(fsdp_reduce_dtype),
                    str(fsdp_output_dtype),
                    str(bn_dtype),
                    str(action),
                    str(is_negotiable),
                    str(safety_margin),
                )

        return cls(
            master_float=master_float,
            amp_dtype=amp_dtype,
            fsdp_param_dtype=fsdp_param_dtype,
            fsdp_reduce_dtype=fsdp_reduce_dtype,
            fsdp_output_dtype=fsdp_output_dtype,
            bn_buffers_dtype=bn_dtype,
            underflow_action=str(action),
        )

    def to_fsdp_policy(self):
        from torch.distributed.fsdp import MixedPrecisionPolicy

        return MixedPrecisionPolicy(
            param_dtype=self.fsdp_param_dtype,
            reduce_dtype=self.fsdp_reduce_dtype,
            output_dtype=self.fsdp_output_dtype,
            cast_forward_inputs=True,
        )
