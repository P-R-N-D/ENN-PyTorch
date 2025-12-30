# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import copy
import inspect
import json
import logging
import threading
from collections import OrderedDict
from functools import lru_cache
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Sequence, Tuple, Union)

import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn, optim

from ..core.precision import Autocast, PrecisionPolicy, is_scale_safe
from ..core.system import get_device, optimal_optimizer_params
from ..data.pipeline import Dataset
from ..nn.architecture import ModelPolicy

try:
    from torch.optim.swa_utils import AveragedModel as _SWA
    from torch.optim.swa_utils import update_bn as _update_bn
except Exception:
    _SWA = None
    _update_bn = None


_LOGGER = logging.getLogger(__name__)

_OPT_LOGGED_KEYS: "OrderedDict[object, None]" = OrderedDict()
_OPT_LOGGED_MAX: int = 128
_OPT_LOGGED_LOCK = threading.Lock()


def _is_hashable(x: object) -> bool:
    try:
        hash(x)
    except Exception:
        return False
    return True


def _safe_key(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, torch.dtype):
        return str(x)
    if isinstance(x, torch.device):
        return str(x)
    try:
        if isinstance(x, (list, tuple)):
            return tuple(_safe_key(v) for v in x)
        if isinstance(x, dict):
            items = tuple(sorted((str(k), _safe_key(v)) for k, v in x.items()))
            return items
    except Exception:
        pass
    try:
        s = str(x)
        return s
    except Exception:
        return repr(x)


def _log_opt_decision_once(
    logger: Optional[Callable[[str], None]],
    key: object,
    payload: Dict[str, Any],
    *args: Any,
    level: str = "info",
) -> None:
    if logger is None:
        lvl = logging.DEBUG if str(level).lower() == "debug" else logging.INFO
        try:
            if not _LOGGER.isEnabledFor(lvl):
                return
        except Exception:
            pass
    if key is None:
        key = ("opt", payload.get("mode"), payload.get("device"), payload.get("selected"))
    if not _is_hashable(key):
        key = _safe_key(key)
    with _OPT_LOGGED_LOCK:
        if key in _OPT_LOGGED_KEYS:
            with contextlib.suppress(Exception):
                _OPT_LOGGED_KEYS.move_to_end(key)
            return
        _OPT_LOGGED_KEYS[key] = None
        try:
            while len(_OPT_LOGGED_KEYS) > int(_OPT_LOGGED_MAX):
                _OPT_LOGGED_KEYS.popitem(last=False)
        except Exception:
            pass
    try:
        msg = "[OPT][DECISION] " + json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        msg = f"[OPT][DECISION] {payload}"
    if logger is not None:
        try:
            logger(msg)
        except Exception:
            pass
        return
    if str(level).lower() == "debug":
        _LOGGER.debug(msg)
    else:
        _LOGGER.info(msg)


@lru_cache(maxsize=256)
def _ctor_allowed_keys(ctor: Any) -> Optional[frozenset[str]]:
    try:
        sig = inspect.signature(ctor)
    except (TypeError, ValueError):
        return None
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return None
    return frozenset(sig.parameters.keys())


def _filter_kwargs(ctor: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if not kwargs:
        return {}
    allowed: Optional[frozenset[str]] = None
    try:
        allowed = _ctor_allowed_keys(ctor)
    except TypeError:
        allowed = None
    if allowed is None:
        try:
            sig = inspect.signature(ctor)
        except (TypeError, ValueError):
            return dict(kwargs)
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return dict(kwargs)
        allowed = frozenset(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _resolve_device_and_metadata(
    model_or_params: Union[nn.Module, Iterable[nn.Parameter], Sequence[Dict[str, Any]]],
    metadata: Optional["Dataset[Any]"] = None,
) -> Tuple[torch.device, "Dataset[Any]"]:
    ref_tensor: Optional[torch.Tensor] = None
    if isinstance(model_or_params, nn.Module):
        ref_tensor = ModelPolicy._peek_layer(model_or_params)
    elif metadata is None:
        try:
            if isinstance(model_or_params, Sequence) and model_or_params:
                first = model_or_params[0]
                if isinstance(first, dict):
                    for group in model_or_params:
                        if not isinstance(group, dict):
                            continue
                        ps = group.get("params", None)
                        if isinstance(ps, Sequence) and ps:
                            p0 = ps[0]
                            if torch.is_tensor(p0):
                                ref_tensor = p0
                                break
                else:
                    if torch.is_tensor(first):
                        ref_tensor = first
        except Exception:
            ref_tensor = None
    if metadata is not None:
        dev = torch.device(metadata.device)
    elif ref_tensor is not None:
        dev = ref_tensor.device
    else:
        dev = get_device()
    meta = Autocast.coerce_metadata(dev, metadata=metadata)
    dev = torch.device(meta.device)
    return dev, meta


def _coerce_params(
    model_or_params: Union[nn.Module, Iterable[nn.Parameter], Sequence[Dict[str, Any]]]
) -> Union[List[nn.Parameter], List[Dict[str, Any]]]:
    if isinstance(model_or_params, nn.Module):
        return list(model_or_params.parameters())
    if (
        isinstance(model_or_params, (list, tuple))
        and model_or_params
        and isinstance(model_or_params[0], dict)
    ):
        groups: List[Dict[str, Any]] = []
        for g in model_or_params:
            if not isinstance(g, dict):
                continue
            gg = dict(g)
            ps = gg.get("params", [])
            gg["params"] = list(ps)
            groups.append(gg)
        return groups
    return list(model_or_params)


def _master_cpu_dtypes(
    device: torch.device,
    meta: Optional["Dataset[Any]"] = None,
) -> Tuple[torch.dtype, torch.dtype]:
    master_int = torch.int64
    if PrecisionPolicy is None:
        return (torch.float32, master_int)
    try:
        policy = PrecisionPolicy.from_metadata(device, meta)
        master_float = getattr(policy, "master_float", torch.float32)
        if not isinstance(master_float, torch.dtype):
            master_float = torch.float32
        if master_float not in (torch.float32, torch.float64):
            master_float = torch.float32
        return (master_float, master_int)
    except Exception:
        return (torch.float32, master_int)


def _cpu_master_tensor(
    t: torch.Tensor,
    *args: Any,
    master_float: torch.dtype,
    master_int: torch.dtype,
    pin_memory: bool = False,
    non_blocking: Optional[bool] = None,
) -> torch.Tensor:
    if not torch.is_tensor(t):
        raise TypeError("Expected a torch.Tensor")
    if t.is_floating_point():
        target_dtype = master_float
    elif t.is_complex():
        target_dtype = torch.complex128 if master_float == torch.float64 else torch.complex64
    else:
        target_dtype = master_int
    nb = bool(non_blocking) if non_blocking is not None else bool(pin_memory)
    src = t.detach()
    if src.dtype != target_dtype:
        with contextlib.suppress(Exception):
            src = src.to(dtype=target_dtype)
    if src.device.type == "cpu":
        if pin_memory:
            with contextlib.suppress(Exception):
                out = torch.empty(src.shape, device="cpu", dtype=target_dtype, pin_memory=True)
                out.copy_(src)
                return out
            with contextlib.suppress(Exception):
                return src.clone().pin_memory()
        return src.clone()
    if pin_memory:
        with contextlib.suppress(Exception):
            out = torch.empty(src.shape, device="cpu", dtype=target_dtype, pin_memory=True)
            out.copy_(src, non_blocking=nb)
            return out
    return src.to(device="cpu", dtype=target_dtype)


def _to_cpu_detached(
    t: torch.Tensor,
    *args: Any,
    dtype: torch.dtype,
    pin_memory: bool,
    non_blocking: bool,
) -> torch.Tensor:
    if not torch.is_tensor(t):
        raise TypeError("Expected a torch.Tensor")
    src = t.detach()
    if src.dtype != dtype:
        with contextlib.suppress(Exception):
            src = src.to(dtype=dtype)
    if src.device.type == "cpu":
        return src
    if pin_memory:
        with contextlib.suppress(Exception):
            out = torch.empty(src.shape, device="cpu", dtype=dtype, pin_memory=True)
            out.copy_(src, non_blocking=bool(non_blocking))
            return out
    return src.to(device="cpu", dtype=dtype)


def _copy_tensor_into_(dst: torch.Tensor, src: torch.Tensor) -> None:
    if not (torch.is_tensor(dst) and torch.is_tensor(src)):
        return
    if dst.data_ptr() == src.data_ptr():
        return
    try:
        dst.copy_(src)
        return
    except Exception:
        pass
    try:
        dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
    except Exception:
        pass


def _optimizer_state_to_param_device_(optimizer: optim.Optimizer) -> None:
    try:
        state = optimizer.state
    except Exception:
        return
    try:
        for group in optimizer.param_groups:
            params = group.get("params", [])
            for p in params:
                if p is None:
                    continue
                p_state = state.get(p)
                if not isinstance(p_state, dict):
                    continue
                for k, v in p_state.items():
                    if not torch.is_tensor(v):
                        continue
                    try:
                        p_is_float = bool(p.is_floating_point())
                        v_is_float = bool(v.is_floating_point())
                        target_dtype = p.dtype if (p_is_float and v_is_float and v.shape == p.shape) else v.dtype
                        p_state[k] = v.to(device=p.device, dtype=target_dtype)
                    except Exception:
                        try:
                            p_state[k] = v.to(device=p.device)
                        except Exception:
                            pass
    except Exception:
        return


def stochastic_weight_average(
    model: nn.Module,
    *args: Any,
    device: Optional[torch.device] = None,
    use_buffers: bool = True,
    avg_fn: Optional[Any] = None,
    **kwargs: Any,
) -> "StochasticWeightAverage":
    return StochasticWeightAverage(model, device=device, use_buffers=use_buffers, avg_fn=avg_fn)


class ExponentialMovingAverage(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        *args: Any,
        metadata: Optional["Dataset[Any]"] = None,
        pin_memory: bool = False,
        update_every: int = 1,
        non_blocking: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if not 0.0 < float(decay) < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.pin_memory = bool(pin_memory)
        self.update_every = max(1, int(update_every))
        self.non_blocking = (bool(non_blocking) if non_blocking is not None else bool(self.pin_memory))
        self._step: int = 0
        self.metadata = metadata
        self.master_float, self.master_int = _master_cpu_dtypes(
            torch.device(Autocast.coerce_metadata(get_device(), metadata=metadata).device),
            metadata,
        )
        self.shadow: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if torch.is_tensor(v):
                    self.shadow[k] = _cpu_master_tensor(
                        v,
                        master_float=self.master_float,
                        master_int=self.master_int,
                        pin_memory=self.pin_memory,
                        non_blocking=self.non_blocking,
                    )
        self.collected: Dict[str, torch.Tensor] = {}
        self.optim_shadow: Optional[Dict[str, Any]] = None
        self.optim_collected: Optional[Dict[str, Any]] = None

    @torch.no_grad()
    def update(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
    ) -> None:
        decay = self.decay
        one_m = 1.0 - decay
        self._step += 1
        do_update = (self.update_every <= 1) or (self._step % self.update_every == 0)
        if not do_update:
            return
        for name, tensor in model.state_dict().items():
            if not torch.is_tensor(tensor):
                continue
            s = self.shadow.get(name)
            if s is None or (not torch.is_tensor(s)) or s.shape != tensor.shape:
                self.shadow[name] = _cpu_master_tensor(
                    tensor,
                    master_float=self.master_float,
                    master_int=self.master_int,
                    pin_memory=self.pin_memory,
                    non_blocking=self.non_blocking,
                )
                continue
            if tensor.is_floating_point():
                try:
                    t_cpu = _to_cpu_detached(
                        tensor,
                        dtype=s.dtype,
                        pin_memory=self.pin_memory,
                        non_blocking=self.non_blocking,
                    )
                    s.mul_(decay).add_(t_cpu, alpha=one_m)
                except Exception:
                    self.shadow[name] = _cpu_master_tensor(
                        tensor,
                        master_float=self.master_float,
                        master_int=self.master_int,
                        pin_memory=self.pin_memory,
                        non_blocking=self.non_blocking,
                    )
            else:
                try:
                    t_cpu = _to_cpu_detached(
                        tensor,
                        dtype=s.dtype,
                        pin_memory=self.pin_memory,
                        non_blocking=self.non_blocking,
                    )
                    s.copy_(t_cpu)
                except Exception:
                    self.shadow[name] = _cpu_master_tensor(
                        tensor,
                        master_float=self.master_float,
                        master_int=self.master_int,
                        pin_memory=self.pin_memory,
                        non_blocking=self.non_blocking,
                    )
        if optimizer is None:
            return
        state = optimizer.state_dict()
        if self.optim_shadow is None:
            shadow: Dict[str, Any] = {}
            if "param_groups" in state:
                shadow["param_groups"] = copy.deepcopy(state["param_groups"])
            if "state" in state:
                shadow_state: Dict[Any, Any] = {}
                for pid, slot in state["state"].items():
                    s_slot: Dict[str, Any] = {}
                    if isinstance(slot, dict):
                        for sk, sv in slot.items():
                            if torch.is_tensor(sv):
                                s_slot[sk] = _cpu_master_tensor(
                                    sv,
                                    master_float=self.master_float,
                                    master_int=self.master_int,
                                    pin_memory=self.pin_memory,
                                    non_blocking=self.non_blocking,
                                )
                            else:
                                s_slot[sk] = copy.deepcopy(sv)
                    else:
                        s_slot = copy.deepcopy(slot)
                    shadow_state[pid] = s_slot
                shadow["state"] = shadow_state
            self.optim_shadow = shadow
            return
        shadow = self.optim_shadow
        if "param_groups" in state:
            shadow["param_groups"] = copy.deepcopy(state["param_groups"])
        if "state" in state:
            shadow_state = shadow.setdefault("state", {})
            for pid, slot in state["state"].items():
                if not isinstance(slot, dict):
                    shadow_state[pid] = copy.deepcopy(slot)
                    continue
                s_slot = shadow_state.setdefault(pid, {})
                if not isinstance(s_slot, dict):
                    s_slot = {}
                    shadow_state[pid] = s_slot
                for sk, sv in slot.items():
                    if torch.is_tensor(sv):
                        if sv.is_floating_point():
                            prev = s_slot.get(sk)
                            if torch.is_tensor(prev) and prev.is_floating_point() and prev.shape == sv.shape:
                                try:
                                    sv_cpu = _to_cpu_detached(
                                        sv,
                                        dtype=prev.dtype,
                                        pin_memory=self.pin_memory,
                                        non_blocking=self.non_blocking,
                                    )
                                    prev.mul_(decay).add_(sv_cpu, alpha=one_m)
                                    continue
                                except Exception:
                                    pass
                            s_slot[sk] = _cpu_master_tensor(
                                sv,
                                master_float=self.master_float,
                                master_int=self.master_int,
                                pin_memory=self.pin_memory,
                                non_blocking=self.non_blocking,
                            )
                        else:
                            s_slot[sk] = _cpu_master_tensor(
                                sv,
                                master_float=self.master_float,
                                master_int=self.master_int,
                                pin_memory=self.pin_memory,
                                non_blocking=self.non_blocking,
                            )
                    else:
                        s_slot[sk] = copy.deepcopy(sv)

    @torch.no_grad()
    def apply_to(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
    ) -> None:
        model_state = model.state_dict()
        for name, buf in self.shadow.items():
            dst = model_state.get(name)
            if torch.is_tensor(dst) and torch.is_tensor(buf):
                _copy_tensor_into_(dst, buf)
        if optimizer is not None and self.optim_shadow is not None:
            optimizer.load_state_dict(self.optim_shadow)
            _optimizer_state_to_param_device_(optimizer)

    @torch.no_grad()
    def store(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
    ) -> None:
        collected: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if torch.is_tensor(v):
                    collected[k] = _cpu_master_tensor(
                        v,
                        master_float=self.master_float,
                        master_int=self.master_int,
                        pin_memory=self.pin_memory,
                        non_blocking=self.non_blocking,
                    )
        self.collected = collected
        if optimizer is not None:
            state = optimizer.state_dict()
            off: Dict[str, Any] = {}
            if "param_groups" in state:
                off["param_groups"] = copy.deepcopy(state["param_groups"])
            if "state" in state:
                off_state: Dict[Any, Any] = {}
                for pid, slot in state["state"].items():
                    if isinstance(slot, dict):
                        s2: Dict[str, Any] = {}
                        for sk, sv in slot.items():
                            if torch.is_tensor(sv):
                                s2[sk] = _cpu_master_tensor(
                                    sv,
                                    master_float=self.master_float,
                                    master_int=self.master_int,
                                    pin_memory=self.pin_memory,
                                    non_blocking=self.non_blocking,
                                )
                            else:
                                s2[sk] = copy.deepcopy(sv)
                        off_state[pid] = s2
                    else:
                        off_state[pid] = copy.deepcopy(slot)
                off["state"] = off_state
            self.optim_collected = off

    @torch.no_grad()
    def restore(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
    ) -> None:
        if self.collected:
            model.load_state_dict(self.collected, strict=False)
        if optimizer is not None and self.optim_collected:
            optimizer.load_state_dict(self.optim_collected)
            _optimizer_state_to_param_device_(optimizer)


class _TensorDictCompat(nn.Module):
    def __init__(self, averaged_module: nn.Module, key: str) -> None:
        super().__init__()
        self._averaged_module = averaged_module
        self._key = str(key)

    def forward(self, x: torch.Tensor) -> Any:
        bs = int(x.shape[0]) if (hasattr(x, "ndim") and x.ndim >= 1) else 1
        td = TensorDict({self._key: x}, batch_size=[bs], device=x.device)
        return self._averaged_module(td)


class AdamW:
    @staticmethod
    def float(
        model_or_params: Union[nn.Module, Iterable[nn.Parameter], Sequence[Dict[str, Any]]],
        lr: float,
        *args: Any,
        weight_decay: float = 0.0,
        metadata: Optional[Dataset[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> optim.Optimizer:
        params = _coerce_params(model_or_params)
        dev, meta = _resolve_device_and_metadata(model_or_params if isinstance(model_or_params, nn.Module) else params, metadata)
        dev_index = int(getattr(dev, "index", -1)) if getattr(dev, "index", None) is not None else -1
        scale_key = (
            bool(getattr(meta, "has_scale", False)),
            bool(getattr(meta, "has_nonfinite", False)),
            _safe_key(getattr(meta, "scale_max_abs", None)),
            _safe_key(getattr(meta, "scale_min_positive", None)),
            _safe_key(getattr(meta, "scale_min_value", None)),
            _safe_key(getattr(meta, "scale_max_value", None)),
            str(getattr(meta, "underflow_action", "")),
            _safe_key(getattr(meta, "int_quant_bits", None)),
        )
        decision_key = ("opt", "adamw-float", str(getattr(dev, "type", "")), dev_index, scale_key)
        attempts: List[Dict[str, Any]] = []
        common_kwargs: Dict[str, Any] = {"lr": lr, "weight_decay": weight_decay, **kwargs}
        if getattr(dev, "type", None) == "cuda":
            try:
                from transformer_engine.pytorch.optimizers import FusedAdam as TEFusedAdam
                opt = TEFusedAdam(params, **_filter_kwargs(TEFusedAdam, common_kwargs))
                attempts.append({"backend": "te.FusedAdam", "ok": True})
                _log_opt_decision_once(
                    logger,
                    decision_key,
                    {
                        "mode": "adamw-float",
                        "device": f"{dev.type}:{dev_index}",
                        "selected": "te.FusedAdam",
                        "attempts": attempts,
                        "scale": {
                            "has_scale": bool(getattr(meta, "has_scale", False)),
                            "has_nonfinite": bool(getattr(meta, "has_nonfinite", False)),
                            "max_abs": _safe_key(getattr(meta, "scale_max_abs", None)),
                            "min_positive": _safe_key(getattr(meta, "scale_min_positive", None)),
                            "min_value": _safe_key(getattr(meta, "scale_min_value", None)),
                            "max_value": _safe_key(getattr(meta, "scale_max_value", None)),
                            "underflow_action": str(getattr(meta, "underflow_action", "")),
                            "int_quant_bits": _safe_key(getattr(meta, "int_quant_bits", None)),
                        },
                    },
                    level="info",
                )
                return opt
            except Exception as exc:
                attempts.append({"backend": "te.FusedAdam", "ok": False, "error": str(exc)})
        if getattr(dev, "type", None) == "cuda":
            fp8_allowed = True
            if getattr(meta, "has_scale", False):
                float8_dtypes = Autocast.float8_formats()
                if not any(is_scale_safe(dtype, meta, safety_margin=2.0) for dtype in float8_dtypes):
                    fp8_allowed = False
                    attempts.append({"backend": "fp8", "ok": False, "reason": "data scale exceeds float8 range"})
            fp8_hw_ok, fp8_hw_reason = Dataset.is_float8_supported(dev)
            if fp8_allowed and fp8_hw_ok:
                try:
                    with contextlib.suppress(Exception):
                        __import__("torchao.float8")
                    AdamWFp8 = None
                    with contextlib.suppress(Exception):
                        from torchao.optim import AdamWFp8
                    if AdamWFp8 is None:
                        from torchao.prototype.float8.optim import AdamWFp8
                    opt = AdamWFp8(params, **_filter_kwargs(AdamWFp8, common_kwargs))
                    attempts.append({"backend": "torchao.AdamWFp8", "ok": True, "reason": str(fp8_hw_reason)})
                    _log_opt_decision_once(
                        logger,
                        decision_key,
                        {
                            "mode": "adamw-float",
                            "device": f"{dev.type}:{dev_index}",
                            "selected": "torchao.AdamWFp8",
                            "attempts": attempts,
                            "fp8_reason": str(fp8_hw_reason),
                            "scale": {
                                "has_scale": bool(getattr(meta, "has_scale", False)),
                                "has_nonfinite": bool(getattr(meta, "has_nonfinite", False)),
                                "max_abs": _safe_key(getattr(meta, "scale_max_abs", None)),
                                "min_positive": _safe_key(getattr(meta, "scale_min_positive", None)),
                                "min_value": _safe_key(getattr(meta, "scale_min_value", None)),
                                "max_value": _safe_key(getattr(meta, "scale_max_value", None)),
                                "underflow_action": str(getattr(meta, "underflow_action", "")),
                                "int_quant_bits": _safe_key(getattr(meta, "int_quant_bits", None)),
                            },
                        },
                        level="info",
                    )
                    return opt
                except Exception as exc:
                    attempts.append(
                        {"backend": "torchao.AdamWFp8", "ok": False, "error": str(exc), "reason": str(fp8_hw_reason)}
                    )
            else:
                attempts.append({"backend": "fp8", "ok": False, "reason": str(fp8_hw_reason)})
        quant_bits = getattr(meta, "int_quant_bits", None)
        if quant_bits in (4, 8):
            try:
                try:
                    from torchao.optim import AdamW4bit, AdamW8bit
                except ImportError:
                    from torchao.prototype.low_bit_optim import AdamW4bit, AdamW8bit
                if int(quant_bits) == 8:
                    opt = AdamW8bit(params, **_filter_kwargs(AdamW8bit, common_kwargs))
                    selected = "torchao.AdamW8bit"
                else:
                    opt = AdamW4bit(params, **_filter_kwargs(AdamW4bit, common_kwargs))
                    selected = "torchao.AdamW4bit"
                attempts.append({"backend": selected, "ok": True, "bits": int(quant_bits)})
                _log_opt_decision_once(
                    logger,
                    decision_key,
                    {
                        "mode": "adamw-float",
                        "device": f"{dev.type}:{dev_index}",
                        "selected": selected,
                        "attempts": attempts,
                        "scale": {
                            "has_scale": bool(getattr(meta, "has_scale", False)),
                            "has_nonfinite": bool(getattr(meta, "has_nonfinite", False)),
                            "max_abs": _safe_key(getattr(meta, "scale_max_abs", None)),
                            "min_positive": _safe_key(getattr(meta, "scale_min_positive", None)),
                            "min_value": _safe_key(getattr(meta, "scale_min_value", None)),
                            "max_value": _safe_key(getattr(meta, "scale_max_value", None)),
                            "underflow_action": str(getattr(meta, "underflow_action", "")),
                            "int_quant_bits": _safe_key(getattr(meta, "int_quant_bits", None)),
                        },
                    },
                    level="info",
                )
                return opt
            except Exception as exc:
                attempts.append(
                    {"backend": "torchao.AdamW(lowbit)", "ok": False, "error": str(exc), "bits": quant_bits}
                )
        flags: Dict[str, bool] = optimal_optimizer_params(dev, use_foreach=None, use_fused=False)
        fallback_kwargs = dict(flags)
        fallback_kwargs.update(common_kwargs)
        opt = optim.AdamW(params, **_filter_kwargs(optim.AdamW, fallback_kwargs))
        attempts.append({"backend": "torch.optim.AdamW", "ok": True, "flags": flags})
        _log_opt_decision_once(
            logger,
            decision_key,
            {"mode": "adamw-float", "device": f"{dev.type}:{dev_index}", "selected": "torch.optim.AdamW", "attempts": attempts},
            level="info",
        )
        return opt

    @staticmethod
    def integer(
        model_or_params: Union[nn.Module, Iterable[nn.Parameter], Sequence[Dict[str, Any]]],
        lr: float,
        *args: Any,
        weight_decay: float = 0.0,
        metadata: Optional[Dataset[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> optim.Optimizer:
        params = _coerce_params(model_or_params)
        dev, meta = _resolve_device_and_metadata(model_or_params if isinstance(model_or_params, nn.Module) else params, metadata)
        dev_index = int(getattr(dev, "index", -1)) if getattr(dev, "index", None) is not None else -1
        scale_key = (
            bool(getattr(meta, "has_scale", False)),
            bool(getattr(meta, "has_nonfinite", False)),
            _safe_key(getattr(meta, "scale_max_abs", None)),
            _safe_key(getattr(meta, "scale_min_positive", None)),
            _safe_key(getattr(meta, "scale_min_value", None)),
            _safe_key(getattr(meta, "scale_max_value", None)),
            bool(getattr(meta, "scale_is_integral", False)),
            str(getattr(meta, "underflow_action", "")),
            _safe_key(getattr(meta, "int_quant_bits", None)),
        )
        decision_key = ("opt", "adamw-integer", str(getattr(dev, "type", "")), dev_index, scale_key)
        decision: Dict[str, Any] = {
            "mode": "adamw-integer",
            "device": f"{dev.type}:{dev_index}",
            "selected": None,
            "attempts": [],
            "scale": {
                "has_scale": bool(getattr(meta, "has_scale", False)),
                "has_nonfinite": bool(getattr(meta, "has_nonfinite", False)),
                "is_integral": _safe_key(getattr(meta, "scale_is_integral", None)),
                "max_abs": _safe_key(getattr(meta, "scale_max_abs", None)),
                "min": _safe_key(getattr(meta, "scale_min_value", None)),
                "max": _safe_key(getattr(meta, "scale_max_value", None)),
                "int_quant_bits": _safe_key(getattr(meta, "int_quant_bits", None)),
            },
        }
        common_kwargs: Dict[str, Any] = {"lr": lr, "weight_decay": weight_decay, **kwargs}
        opt: Optional[optim.Optimizer] = None
        selected: Optional[str] = None
        if getattr(dev, "type", None) == "cuda":
            try:
                from transformer_engine.pytorch.optimizers import FusedAdam as TEFusedAdam
                opt = TEFusedAdam(params, **_filter_kwargs(TEFusedAdam, common_kwargs))
                selected = "transformer_engine.FusedAdam"
                decision["attempts"].append({"name": "TE.FusedAdam", "ok": True})
            except Exception as exc:
                decision["attempts"].append({"name": "TE.FusedAdam", "ok": False, "error": str(exc)})
        quant_choice: Optional[str] = None
        quant_reason: Optional[str] = None
        explicit_bits = getattr(meta, "int_quant_bits", None)
        if opt is None and explicit_bits in (4, 8):
            quant_choice = "int8" if int(explicit_bits) == 8 else "int4"
            quant_reason = "requested-by-metadata"
            decision["attempts"].append({"name": f"{quant_choice}-requested", "ok": True})
        if opt is None and quant_choice is None and bool(getattr(meta, "has_scale", False)):
            if getattr(meta, "scale_is_integral", None) is False:
                decision["attempts"].append({"name": "lowbit", "ok": False, "reason": "non-integral"})
            else:
                min_v = getattr(meta, "scale_min_value", None)
                max_v = getattr(meta, "scale_max_value", None)
                try:
                    min_f = float(min_v) if min_v is not None else None
                except Exception:
                    min_f = None
                try:
                    max_f = float(max_v) if max_v is not None else None
                except Exception:
                    max_f = None
                try:
                    max_abs = float(abs(getattr(meta, "scale_max_abs", 0.0)))
                except Exception:
                    max_abs = 0.0
                candidates: List[Tuple[str, Callable[[Optional[torch.device]], Tuple[bool, str]]]] = []
                if (min_f is not None) and (max_f is not None):
                    if (min_f >= -8.0) and (max_f <= 7.0):
                        candidates.append(("int4", Dataset.is_int4_supported))
                    if (min_f >= -128.0) and (max_f <= 127.0):
                        candidates.append(("int8", Dataset.is_int8_supported))
                else:
                    if max_abs <= 7.0:
                        candidates.append(("int4", Dataset.is_int4_supported))
                    if max_abs <= 127.0:
                        candidates.append(("int8", Dataset.is_int8_supported))
                for name, checker in candidates:
                    ok, reason = checker(dev)
                    decision["attempts"].append({"name": f"{name}-supported", "ok": bool(ok), "reason": str(reason)})
                    if ok:
                        quant_choice = name
                        quant_reason = str(reason)
                        break
                if not candidates:
                    decision["attempts"].append({"name": "lowbit", "ok": False, "reason": f"magnitude>{max_abs}"})
        if opt is None and quant_choice in {"int8", "int4"}:
            try:
                try:
                    from torchao.optim import AdamW4bit, AdamW8bit
                except ImportError:
                    from torchao.prototype.low_bit_optim import AdamW4bit, AdamW8bit
                if quant_choice == "int8":
                    opt = AdamW8bit(params, **_filter_kwargs(AdamW8bit, common_kwargs))
                    selected = "torchao.optim.AdamW8bit"
                    decision["attempts"].append({"name": "AO.AdamW8bit", "ok": True, "reason": str(quant_reason)})
                else:
                    opt = AdamW4bit(params, **_filter_kwargs(AdamW4bit, common_kwargs))
                    selected = "torchao.optim.AdamW4bit"
                    decision["attempts"].append({"name": "AO.AdamW4bit", "ok": True, "reason": str(quant_reason)})
            except Exception as exc:
                decision["attempts"].append({"name": "AO.AdamW(lowbit)", "ok": False, "error": str(exc)})
        if opt is None:
            flags: Dict[str, bool] = optimal_optimizer_params(dev, use_foreach=None, use_fused=False)
            fallback_kwargs = dict(flags)
            fallback_kwargs.update(common_kwargs)
            opt = optim.AdamW(params, **_filter_kwargs(optim.AdamW, fallback_kwargs))
            selected = f"torch.optim.AdamW(flags={flags})"
            decision["attempts"].append({"name": "torch.optim.AdamW", "ok": True, "flags": flags})
        decision["selected"] = selected
        _log_opt_decision_once(logger, decision_key, decision)
        return opt


class StochasticWeightAverage:
    def __init__(
        self,
        model: nn.Module,
        *args: Any,
        device: Optional[torch.device] = None,
        use_buffers: bool = True,
        avg_fn: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if _SWA is None:
            raise RuntimeError("torch.optim.swa_utils is not available")
        self._source = model
        self._averaged = _SWA(model, device=device, use_buffers=use_buffers, avg_fn=avg_fn)

    @property
    def module(self) -> nn.Module:
        return self._averaged.module

    @property
    def source(self) -> nn.Module:
        return self._source

    def update_weight(self, model: Optional[nn.Module] = None) -> None:
        target = model if model is not None else self._source
        if target is None:
            raise RuntimeError("StochasticWeightAverage was initialised without a source model")
        with torch.no_grad():
            try:
                self._averaged.update_parameters(target)
            except Exception as e:
                raise RuntimeError(f"SWA update failed: {e}") from e

    @contextlib.contextmanager
    def reduction(self, model: nn.Module) -> Iterator[None]:
        avg_params = dict(self.module.named_parameters())
        backup: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            try:
                for name, param in model.named_parameters():
                    avg_param = avg_params.get(name)
                    if avg_param is None:
                        continue
                    backup[name] = param.detach().clone()
                    _copy_tensor_into_(param, avg_param)
                yield
            finally:
                for name, param in model.named_parameters():
                    buf = backup.get(name)
                    if buf is not None:
                        _copy_tensor_into_(param, buf)

    def update_batch_norm(
        self,
        feature_iter: Iterable[TensorDictBase | Dict[str, Any] | torch.Tensor | Any],
        *args: Any,
        device: Optional[torch.device] = None,
        in_key: str = "features",
        **kwargs: Any,
    ) -> None:
        if _update_bn is None:
            raise RuntimeError("torch.optim.swa_utils.update_bn is not available")
        key = str(in_key)

        def _features(it: Iterable[Any]) -> Iterator[torch.Tensor]:
            for item in it:
                tensor: Optional[torch.Tensor] = None
                match item:
                    case TensorDictBase():
                        maybe = item.get(key, None)
                        tensor = maybe if torch.is_tensor(maybe) else None
                    case torch.Tensor():
                        tensor = item
                    case dict():
                        maybe = item.get(key)
                        tensor = maybe if torch.is_tensor(maybe) else None
                    case tuple() | list():
                        if item and torch.is_tensor(item[0]):
                            tensor = item[0]
                    case _:
                        tensor = None
                if tensor is not None:
                    yield tensor
                  
        adapter = _TensorDictCompat(self._averaged, key)
        _update_bn(_features(feature_iter), adapter, device=device)

    def save_state_dict(self) -> Dict[str, Any]:
        return self._averaged.state_dict()

    def load_state_dict(self, state: Dict[str, Any]) -> Any:
        return self._averaged.load_state_dict(state)
