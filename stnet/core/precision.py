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

from ..data.schemas import default_underflow_action, normalize_underflow_action
from .system import (
    get_device,
    get_device_stats,
    is_cuda_bf16_supported,
    is_float8_supported,
    is_int8_supported,
)


_LOGGER = logging.getLogger(__name__)

_NEGO_LOGGED_KEYS: "OrderedDict[object, None]" = OrderedDict()
_NEGO_LOGGED_MAX: int = 256
_NEGO_LOGGED_LOCK = threading.Lock()


def _log_negotiation(
    logger: Optional[logging.Logger],
    key: object,
    payload: Dict[str, Any],
    *args: Any,
    level: str = "debug",
) -> None:
    lg, lvl = (logger or _LOGGER), (logging.INFO if str(level).lower() == "info" else logging.DEBUG)
    if not lg.isEnabledFor(lvl): return
    
    k = key or (payload.get("context"), payload.get("device"), payload.get("selected"))
    with _NEGO_LOGGED_LOCK:
        if k in _NEGO_LOGGED_KEYS:
            _NEGO_LOGGED_KEYS.move_to_end(k); return
        _NEGO_LOGGED_KEYS[k] = None
        if len(_NEGO_LOGGED_KEYS) > _NEGO_LOGGED_MAX: _NEGO_LOGGED_KEYS.popitem(last=False)
        
    try:
        msg = "[AMP][NEGOTIATE] " + json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        msg = f"[AMP][NEGOTIATE] {payload}"
    lg.log(lvl, msg)


def _parse_dtype(dtype: Any) -> str:
    return str(dtype).split(".")[-1] if isinstance(dtype, torch.dtype) else str(dtype)


def _to_serializable(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, str, float)): return x
    try: return float(x)
    except: return str(x)


def _coerce_torch_dtype(value: Any, default: torch.dtype) -> torch.dtype:
    return value if isinstance(value, torch.dtype) else getattr(torch, str(value).strip().replace("torch.", ""), default) if value is not None else default


def _get_meta_stats(meta: Any | None) -> Dict[str, Any]:
    return {k: _to_serializable(getattr(meta, k, None)) for k in ("has_scale", "has_nonfinite", "scale_max_abs", "scale_min_positive", "scale_is_integral", "scale_min_value", "scale_max_value", "underflow_action", "int_quant_bits")} if meta else {}


def _validate_dtype_safety(
    dtype: torch.dtype,
    meta: Any | None,
    *args: Any,
    safety_margin: float = 8.0,
    underflow_action: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[bool, str]:
    if not (meta and getattr(meta, "has_scale", False)): return True, "no-scale"
    if not isinstance(dtype, torch.dtype): return False, "not-dtype"
    if getattr(meta, "has_nonfinite", False): return False, "nonfinite-data"
    
    max_abs = getattr(meta, "scale_max_abs", None)
    if max_abs is None: return True, "no-max-abs"
    try: max_abs_f = float(abs(max_abs))
    except: return False, "max-abs-not-float"
    if not math.isfinite(max_abs_f): return False, "max-abs-nonfinite"
    
    action = normalize_underflow_action(underflow_action or getattr(meta, "underflow_action", None), default=default_underflow_action())
    
    if getattr(dtype, "is_complex", False):
        ok, why = _validate_dtype_safety(torch.float32 if dtype == torch.complex64 else torch.float64, meta, safety_margin=safety_margin, underflow_action=action)
        return (ok, f"complex-base:{why}" if ok else f"complex-base-unsafe:{why}")

    if getattr(dtype, "is_floating_point", False):
        info = torch.finfo(dtype)
        if max_abs_f > (ov_limit := float(info.max) / max(1.0, float(safety_margin))): return False, f"overflow({max_abs_f:.6g}>{ov_limit:.6g})"
        if action == "forbid" and (mp := getattr(meta, "scale_min_positive", None)) is not None:
            try: mp_f = float(mp)
            except: return False, "min-pos-not-float"
            if math.isfinite(mp_f) and mp_f > 0.0 and mp_f < (uf_limit := float(info.tiny) * max(1.0, float(safety_margin))):
                return False, f"underflow({mp_f:.6g}<{uf_limit:.6g})"
        return True, "ok"
    
    if getattr(meta, "scale_is_integral", None) is False: return False, "nonintegral-data"
    if dtype == torch.bool: return (max_abs_f <= 1.0, "ok" if max_abs_f <= 1.0 else f"bool-range({max_abs_f:.6g})")
    
    try: info = torch.iinfo(dtype)
    except TypeError: return False, "not-integer-dtype"
    
    if (mn := getattr(meta, "scale_min_value", None)) is not None and (mx := getattr(meta, "scale_max_value", None)) is not None:
        try: mn_f, mx_f = float(mn), float(mx)
        except: return False, "int-minmax-not-float"
        if mn_f < float(info.min) or mx_f > float(info.max): return False, f"int-range({mn_f:.6g},{mx_f:.6g} not in [{info.min},{info.max}])"
        return True, "ok"
    return (max_abs_f <= float(info.max), "ok" if max_abs_f <= float(info.max) else f"int-max-abs({max_abs_f:.6g}>{info.max})")


def is_scale_safe(
    dtype: torch.dtype,
    meta: Any | None,
    *args: Any,
    safety_margin: float = 8.0,
    underflow_action: Optional[str] = None,
    **kwargs: Any,
) -> bool:
    ok, _ = _validate_dtype_safety(
        dtype, meta, safety_margin=safety_margin, underflow_action=underflow_action
    )
    return ok


@dataclass(slots=True)
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

    def _refresh_device_info(self) -> None:
        ds = get_device_stats(self.device)
        self.device = ds.device
        self.device_type = ds.device_type
        self.cuda_cc = ds.cuda_cc

    def coerce_device_info(self) -> "DeviceMeta":
        self._refresh_device_info()
        return self

    def refresh(self) -> "DeviceMeta":
        self._refresh_device_info()
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


class Autocast:
    _preferred_fp8_backend: Optional[str] = None
    _preferred_int_backend: Optional[str] = None
    _last_float_dtype: torch.dtype = torch.float32
    _last_int_dtype: torch.dtype = torch.int64
    _metadata: Any | None = None
    _metadata_tls = threading.local()

    @classmethod
    def _get_tls_metadata(cls) -> Any | None:
        return getattr(cls._metadata_tls, "meta", None)

    @classmethod
    def _set_tls_metadata(cls, meta: Any | None) -> None:
        setattr(cls._metadata_tls, "meta", meta)
        cls._metadata = meta

    @staticmethod
    def _device(device: Optional[Union[torch.device, str]] = None) -> torch.device:
        if device is None:
            return get_device()
        if isinstance(device, torch.device):
            return device
        return torch.device(device)

    @classmethod
    def _try_load_backend(cls, mod_name: str, attr_name: str, test_supported: bool = True, reason_fail: str = "") -> Any:
        if not test_supported:
            _LOGGER.debug(reason_fail); return None
        try:
            mod = importlib.import_module(mod_name)
            if getattr(mod, attr_name, None) is None: raise AttributeError(f"{mod_name}.{attr_name} missing")
            return mod
        except Exception as e:
            _LOGGER.debug(f"Autocast backend {mod_name} failed: {e}")
            return None

    @classmethod
    def _resolve_backend(cls, preferred: Optional[str], device: torch.device, kind: str) -> Optional[str]:
        order = ("ao", "te") if preferred == "ao" else ("te", "ao")
        is_supported = is_float8_supported if kind == "fp8" else is_int8_supported
        
        for backend in order:
            if backend == "te":
                ok, why = is_supported(device)
                if cls._try_load_backend("transformer_engine.pytorch", f"{kind}_autocast", ok, f"Autocast {kind.upper()} TE unavailable: {why}"):
                    setattr(cls, f"_preferred_{kind}_backend", "te"); return "te"
            elif backend == "ao":
                mod, attr = ("torchao.float8", "fp8_autocast") if kind == "fp8" else ("torchao.quantization", "int8_autocast")
                if cls._try_load_backend(mod, attr):
                    setattr(cls, f"_preferred_{kind}_backend", "ao"); return "ao"
        
        setattr(cls, f"_preferred_{kind}_backend", None)
        return None

    @classmethod
    def _fp8_backend(cls, pref, *args, device=None, **kwargs): return cls._resolve_backend(pref, device or cls._device(None), "fp8")
    @classmethod
    def _int_backend(cls, pref, *args, device=None, **kwargs): return cls._resolve_backend(pref, device or cls._device(None), "int")

    @classmethod
    def _get_backend_context(cls, mod_name, attr_name, enabled):
        if not enabled: return []
        if (mod := cls._try_load_backend(mod_name, attr_name)): return [getattr(mod, attr_name)(enabled=True)]
        return []

    @classmethod
    def _nvidia_float8(cls, device: torch.device, enabled: bool): return cls._get_backend_context("transformer_engine.pytorch", "fp8_autocast", enabled)

    @classmethod
    def _torchao_float8(cls, enabled: bool): return cls._get_backend_context("torchao.float8", "fp8_autocast", enabled)

    @classmethod
    def _torchao_int8_backend(cls, device: torch.device, enabled: bool):
        backend = cls._preferred_int_backend
        if backend == "te":
            return cls._get_backend_context("transformer_engine.pytorch", "int8_autocast", enabled)
        if backend == "ao":
            return cls._get_backend_context("torchao.quantization", "int8_autocast", enabled)
        return []

    @classmethod
    def _torchao_int4(cls, device: torch.device, enabled: bool):
        return cls._get_backend_context("torchao.quantization", "int4_autocast", enabled)

    @classmethod
    def _torchao_int8(
        cls, device: torch.device, enabled: bool
    ) -> List[AbstractContextManager[None]]:
        return cls._torchao_int8_autocast(device, enabled)

    @classmethod
    def _torchao_int8_autocast(
        cls, device: torch.device, enabled: bool
    ) -> List[AbstractContextManager[None]]:
        if cls._preferred_int_backend is None:
            cls._preferred_int_backend = "ao"
        return cls._torchao_int8_backend(device, enabled)

    @classmethod
    def metadata(cls) -> Any | None:
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
        *args: Any,
        fp8_backend: Optional[str] = None,
        int_backend: Optional[str] = None,
        metadata: Any | None = None,
    ) -> None:
        backend = fp8_backend
        int_b = int_backend or ("ao" if isinstance(model, nn.Module) and any(getattr(model, a, False) for a in ("__int8_training_qat__", "__int8_training_ptq__", "__int8_inference_ao__")) else None)
        if backend is None and isinstance(model, nn.Module):
            if any(getattr(model, a, False) for a in ("__fp8_training_te__", "__fp8_inference_te__", "__te_fp8_default__")): backend = "te"
            elif any(getattr(model, a, False) for a in ("__fp8_training_ao__", "__fp8_inference_ao__")): backend = "ao"

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
        
        def _parse_margin():
            p2 = kwargs.pop("safety_margin_pow2", None)
            if p2 is not None: return float(2**max(0, min(30, int(p2 or 3)))), int(p2 or 3)
            margin = float(kwargs.pop("safety_margin", 8.0))
            return (margin, int(round(math.log2(margin)))) if margin > 0 else (8.0, 3)

        safety_margin, safety_margin_pow2 = _parse_margin()
        underflow_override = normalize_underflow_action(kwargs.pop("underflow_action", kwargs.pop("underflow", None)), default=default_underflow_action())
        collect_checks = logger and (logger.isEnabledFor(logging.DEBUG) or logger.isEnabledFor(logging.INFO))
        checks, selected, selected_from = [], None, "candidate"
        
        for dt in candidates:
            ok, why = _validate_dtype_safety(dt, meta, safety_margin=safety_margin, underflow_action=underflow_override)
            if collect_checks: checks.append({"dtype": _parse_dtype(dt), "ok": bool(ok), "reason": str(why)})
            if ok: selected = dt; break
        
        fallback_order: Tuple[torch.dtype, ...] = ()
        if selected is None:
            selected_from = "fallback"
            fallback_order = (fallback, torch.float32, torch.float64) if getattr(fallback, "is_floating_point", False) else (fallback, torch.int64, torch.float32, torch.float64)
            for dt in fallback_order:
                ok, why = _validate_dtype_safety(dt, meta, safety_margin=safety_margin, underflow_action=underflow_override)
                if collect_checks: checks.append({"dtype": _parse_dtype(dt), "ok": bool(ok), "reason": str(why)})
                if ok: selected = dt; break
            if selected is None:
                selected, selected_from = fallback, "unsafe-fallback"
        
        if logger is not None:
            level = "info" if selected_from != "candidate" else "debug"
            if logger.isEnabledFor(logging.INFO if level == "info" else logging.DEBUG):
                scale_key = (getattr(meta, "has_scale", False), getattr(meta, "has_nonfinite", False), getattr(meta, "scale_max_abs", None), getattr(meta, "underflow_action", None))
                if decision_key is None:
                    decision_key = ("amp", str(context), str(device), tuple(_parse_dtype(x) for x in candidates), _parse_dtype(fallback), scale_key, float(safety_margin))
                
                payload = {"context": str(context), "device": str(device), "selected": _parse_dtype(selected), "selected_from": selected_from, "scale": _get_meta_stats(meta)}
                if collect_checks: payload["checks"] = checks
                _log_negotiation(logger, decision_key, payload, level=level)
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
        requested_dtype = _coerce_torch_dtype(dtype, torch.float16)
        candidates: Tuple[torch.dtype, ...] = (requested_dtype, cls._last_float_dtype)
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
        cls,
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
            logger=_LOGGER, context="float", device=dev, meta=meta,
        )
        contexts: List[contextlib.AbstractContextManager[None]] = []
        fp8_disable_reason: Optional[str] = None
        fp8_backend_used: Optional[str] = None
        
        backend = cls._fp8_backend(cls._preferred_fp8_backend, device=dev)
        wants_fp8 = backend is not None
        if wants_fp8 and getattr(meta, "has_scale", False) and not any(_validate_dtype_safety(dt, meta, safety_margin=2.0)[0] for dt in cls.float8_formats()):
            wants_fp8, fp8_disable_reason = False, "scale-exceeds-fp8"
            
        if wants_fp8:
            for b in (backend, "ao") if backend == "te" else (backend,):
                if (ctxs := (cls._nvidia_float8(dev, True) if b == "te" else cls._torchao_float8(True))):
                    contexts.extend(ctxs); fp8_backend_used = b; break
            if not fp8_backend_used: fp8_disable_reason = "fp8-backend-unavailable"

        requested_dtype = amp_dtype
        if cls._last_float_dtype in amp_candidates and cls._last_float_dtype == amp_dtype:
            requested_dtype = cls._last_float_dtype
        if requested_dtype is torch.float64:
            wants_fp8, fp8_disable_reason = False, fp8_disable_reason or "master-fp64"
        if dev.type == "cuda" and requested_dtype is torch.bfloat16 and not is_cuda_bf16_supported(dev):
            requested_dtype = torch.float16
            
        if dev.type == "cpu" and requested_dtype not in (torch.bfloat16, torch.float16):
            contexts.append(contextlib.nullcontext()); cls._last_float_dtype = requested_dtype
        else:
            try:
                contexts.append(torch.amp.autocast(device_type=dev.type, dtype=requested_dtype, enabled=True))
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
        cls, device: Optional[Union[torch.device, str]] = None
    ) -> contextlib.AbstractContextManager[None]:
        dev = cls._device(device)
        with contextlib.ExitStack() as stack:
            with contextlib.suppress(Exception): stack.enter_context(torch.amp.autocast(device_type=dev.type, enabled=False)) or stack.enter_context(contextlib.nullcontext())
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
        int_candidates = tuple(getattr(meta, "int_dtypes", ())) or (torch.int64,)
        int_dtype = cls.negotiate(
            int_candidates, fallback=torch.int64, logger=_LOGGER, context="int", device=dev, meta=meta,
        )
        quant_bits = getattr(meta, "int_quant_bits", None)
        wants_int4 = quant_bits == 4
        wants_int8 = (int_dtype == torch.int8) or (quant_bits == 8)
        
        int_backend_used: Optional[str] = None
        contexts: List[contextlib.AbstractContextManager[None]] = []
        if wants_int4:
            try:
                if (contexts := cls._torchao_int4(dev, True)): cls._preferred_int_backend, int_backend_used = "ao", "ao"
            except Exception as exc:
                _LOGGER.debug("Autocast INT4 enable failed: %s", exc)
        if not contexts and wants_int8:
            backend = cls._int_backend(cls._preferred_int_backend, device=dev)
            contexts = cls._torchao_int8(dev, True) if backend else []
            if contexts: int_backend_used = backend
            elif backend == "te" and cls._int_backend("ao", device=dev) == "ao":
                 if (contexts := cls._torchao_int8(dev, True)): int_backend_used = "ao"

        if not contexts: contexts.append(contextlib.nullcontext())
        with contextlib.ExitStack() as stack:
            for ctx in contexts: stack.enter_context(ctx)
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


@dataclass(slots=True)
class PrecisionPolicy:
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
        *args: Any,
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
                if callable(f := getattr(meta, "refresh", None)): f()
        action = normalize_underflow_action(
            getattr(meta, "underflow_action", None), default=default_underflow_action()
        )
        with contextlib.suppress(Exception):
            setattr(meta, "underflow_action", action)
        is_negotiable = bool(getattr(meta, "is_negotiable", False))
        safety = float(safety_margin)
        amp_dtype: Optional[torch.dtype] = None
        master_float = torch.float32 if is_negotiable or (dev.type not in ("cpu", "xpu", "mps") and is_scale_safe(torch.float32, meta, safety_margin=safety)) else torch.float64

        if dev.type == "cuda":
            if is_negotiable and is_scale_safe(torch.float32, meta, safety_margin=safety): master_float = torch.float32
            if is_cuda_bf16_supported(dev) and is_scale_safe(torch.bfloat16, meta, safety_margin=safety): amp_dtype = torch.bfloat16
            elif is_scale_safe(torch.float16, meta, safety_margin=safety): amp_dtype = torch.float16
        elif dev.type == "xpu": amp_dtype = torch.bfloat16
        elif dev.type == "mps": amp_dtype = torch.float16
        
        fsdp_dt = amp_dtype if master_float == torch.float32 and amp_dtype else master_float
        return cls(
            master_float=master_float,
            amp_dtype=amp_dtype,
            fsdp_param_dtype=fsdp_dt, fsdp_reduce_dtype=fsdp_dt, fsdp_output_dtype=fsdp_dt, bn_buffers_dtype=master_float,
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
