# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import inspect
import importlib.util
import json
import logging
import math
import os
import tempfile
from collections import OrderedDict
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Self,
    Sequence,
    Tuple,
    Union,
)

import torch
from ..core.datatypes import env_bool
from ..core.concurrency import Mutex
from ..core.policies import ModelPolicy, PrecisionPolicy
from ..core.precision import StatelessAutocast, is_scale_safe
from ..core.system import get_device, optimal_optimizer_params
from ..data.pipeline import Dataset
from torch import nn, optim
from torch.optim.swa_utils import update_bn
_LOGGER = logging.getLogger(__name__)
_OPT_LOGGED_KEYS: "OrderedDict[object, None]" = OrderedDict()
_OPT_LOGGED_LOCK = Mutex()
_OPT_LOGGED_MAX: int = 128


def _is_hashable(x: object) -> bool:
    with contextlib.suppress(Exception):
        hash(x)
        return True
    return False


def _to_immutable(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (torch.dtype, torch.device)):
        return str(x)
    if isinstance(x, (list, tuple)):
        return tuple(_to_immutable(v) for v in x)
    if isinstance(x, dict):
        return tuple(sorted((str(k), _to_immutable(v)) for k, v in x.items()))
    return str(x)


def _log_optimizer(
    logger: Optional[Callable[[str], None]],
    key: object,
    payload: Dict[str, Any],
    *args: Any,
    level: str = "info",
) -> None:
    if logger is None and not _LOGGER.isEnabledFor(
        logging.DEBUG if str(level).lower() == "debug" else logging.INFO
    ):
        return
    key = key or (
        "opt",
        payload.get("mode"),
        payload.get("device"),
        payload.get("selected"),
    )
    if not _is_hashable(key):
        key = _to_immutable(key)
    with _OPT_LOGGED_LOCK:
        if key in _OPT_LOGGED_KEYS:
            _OPT_LOGGED_KEYS.move_to_end(key)
            return
        _OPT_LOGGED_KEYS[key] = None
        if len(_OPT_LOGGED_KEYS) > _OPT_LOGGED_MAX:
            _OPT_LOGGED_KEYS.popitem(last=False)
    try:
        msg = "[OPT][DECISION] " + json.dumps(
            payload, sort_keys=True, default=str
        )
    except Exception:
        msg = f"[OPT][DECISION] {payload}"
    if logger is not None:
        logger(msg)
        return
    log_fn = getattr(_LOGGER, str(level).lower(), _LOGGER.info)
    log_fn(msg)


def _get_expected_args(ctor: Any) -> Optional[frozenset[str]]:
    try:
        sig = inspect.signature(ctor)
        return (
            None
            if any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            else frozenset(sig.parameters.keys())
        )
    except Exception:
        return None


def _coerce_kwargs(ctor: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if not kwargs:
        return {}
    allowed = _get_expected_args(ctor)
    if allowed is None:
        try:
            sig = inspect.signature(ctor)
            if any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            ):
                return dict(kwargs)
            allowed = frozenset(sig.parameters.keys())
        except Exception:
            return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in allowed}


def _dataset_for_device(
    model_or_params: Union[
        nn.Module, Iterable[nn.Parameter], Sequence[Dict[str, Any]]
    ],
    metadata: Optional["Dataset[Any]"] = None,
) -> Tuple[torch.device, "Dataset[Any]"]:
    ref_tensor: Optional[torch.Tensor] = None
    if isinstance(model_or_params, nn.Module):
        ref_tensor = ModelPolicy._peek_layer(model_or_params)
    elif (
        metadata is None
        and isinstance(model_or_params, Sequence)
        and model_or_params
    ):
        try:
            first = model_or_params[0]
            if isinstance(first, dict):
                ref_tensor = next(
                    (
                        p
                        for g in model_or_params
                        if isinstance(g, dict)
                        for p in g.get("params", [])
                        if torch.is_tensor(p)
                    ),
                    None,
                )
            elif torch.is_tensor(first):
                ref_tensor = first
        except Exception:
            ref_tensor = None
    dev = (
        torch.device(metadata.device)
        if metadata
        else ref_tensor.device if ref_tensor is not None else get_device()
    )
    meta = StatelessAutocast.coerce_metadata(dev, metadata=metadata)
    return torch.device(meta.device), meta


def _coerce_params(
    model_or_params: Union[
        nn.Module, Iterable[nn.Parameter], Sequence[Dict[str, Any]]
    ],
) -> Union[List[nn.Parameter], List[Dict[str, Any]]]:
    if isinstance(model_or_params, nn.Module):
        return list(model_or_params.parameters())
    if (
        isinstance(model_or_params, (list, tuple))
        and model_or_params
        and isinstance(model_or_params[0], dict)
    ):
        return [
            dict(g, params=list(g.get("params", [])))
            for g in model_or_params
            if isinstance(g, dict)
        ]
    return list(model_or_params)


def _iter_param_tensors(
    params: Union[List[nn.Parameter], List[Dict[str, Any]]],
) -> Iterator[torch.Tensor]:
    if (
        isinstance(params, (list, tuple))
        and params
        and isinstance(params[0], dict)
    ):
        for g in params:
            if not isinstance(g, dict):
                continue
            ps = g.get("params", None)
            if ps is None:
                continue
            for p in ps:
                if torch.is_tensor(p):
                    yield p
    else:
        for p in params:
            if torch.is_tensor(p):
                yield p


def _master_cpu_dtypes(
    device: torch.device,
    meta: Optional["Dataset[Any]"] = None,
) -> Tuple[torch.dtype, torch.dtype]:
    master_float, master_int = torch.float32, torch.int64
    if PrecisionPolicy:
        try:
            master_float = getattr(
                PrecisionPolicy.from_metadata(device, meta),
                "master_float",
                torch.float32,
            )
            if master_float not in (torch.float32, torch.float64):
                master_float = torch.float32
        except Exception:
            pass
    return master_float, master_int


def _cpu_offload(
    t: torch.Tensor,
    dtype: torch.dtype,
    pin_memory: bool = False,
    non_blocking: Optional[bool] = None,
) -> torch.Tensor:
    if not torch.is_tensor(t):
        raise TypeError(f"Expected Tensor, got {type(t)}")
    nb = bool(non_blocking) if non_blocking is not None else bool(pin_memory)
    src = t.detach()
    if src.dtype != dtype:
        with contextlib.suppress(Exception):
            src = src.to(dtype=dtype)
    if src.device.type == "cpu":
        if pin_memory:
            with contextlib.suppress(Exception):
                out = torch.empty(
                    src.shape, device="cpu", dtype=dtype, pin_memory=True
                )
                out.copy_(src, non_blocking=nb)
                return out
            with contextlib.suppress(Exception):
                return src.clone().pin_memory()
        return src.clone()
    if pin_memory:
        with contextlib.suppress(Exception):
            out = torch.empty(
                src.shape, device="cpu", dtype=dtype, pin_memory=True
            )
            out.copy_(src, non_blocking=nb)
            return out
    return src.to(device="cpu", dtype=dtype)


def _init_step_tensor(
    value: object,
    *,
    param: torch.Tensor,
    capturable: bool,
    fused: bool,
) -> torch.Tensor:
    desired_device = (
        param.device if (capturable or fused) else torch.device("cpu")
    )
    desired_dtype = (
        param.dtype if torch.is_floating_point(param) else torch.float32
    )
    if isinstance(value, torch.Tensor):
        step_tensor = value.detach()
        if step_tensor.ndim != 0:
            step_tensor = step_tensor.reshape(())
        if step_tensor.device != desired_device:
            step_tensor = step_tensor.to(desired_device)
        if step_tensor.dtype != desired_dtype:
            step_tensor = step_tensor.to(desired_dtype)
    else:
        base = float(value) if value is not None else 0.0
        step_tensor = torch.tensor(
            base, dtype=desired_dtype, device=desired_device
        )
    return step_tensor


def init_optimizer_state(optim_obj: object) -> None:
    if optim_obj is None:
        return
    try:
        param_groups = getattr(optim_obj, "param_groups", None) or []
    except Exception:
        return
    for group in param_groups:
        amsgrad = group.get("amsgrad", False)
        capturable = bool(group.get("capturable", False))
        fused = bool(group.get("fused", False))
        for param in group.get("params", []) or []:
            if not getattr(param, "requires_grad", False):
                continue
            state = optim_obj.state.get(param)
            state = {} if state is None else state
            step_value = state.get("step")
            state["step"] = _init_step_tensor(
                step_value,
                param=param,
                capturable=capturable,
                fused=fused,
            )
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(param)
            if "exp_avg_sq" not in state:
                state["exp_avg_sq"] = torch.zeros_like(param)
            if amsgrad and "max_exp_avg_sq" not in state:
                state["max_exp_avg_sq"] = torch.zeros_like(param)
            optim_obj.state[param] = state


def _has_batchnorm_modules(model: nn.Module) -> bool:
    try:
        bn_types = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            getattr(nn, "SyncBatchNorm", nn.BatchNorm1d),
        )
        for m in model.modules():
            if isinstance(m, bn_types):
                return True
    except Exception:
        return False
    return False


def exponential_weight_average(
    model: nn.Module,
    decay: float = 0.9999,
    *args: Any,
    metadata: Optional["Dataset[Any]"] = None,
    pin_memory: bool = False,
    update_every: int = 1,
    non_blocking: Optional[bool] = None,
) -> "ExponentialMovingAverage":
    return ExponentialMovingAverage(
        model,
        decay=decay,
        *args,
        metadata=metadata,
        pin_memory=pin_memory,
        update_every=update_every,
        non_blocking=non_blocking,
    )


def stochastic_weight_average(
    model: nn.Module,
    *args: Any,
    device: Optional[torch.device] = None,
    use_buffers: bool = True,
    avg_fn: Optional[Any] = None,
    **kwargs: Any,
) -> "StochasticWeightAverage":
    return StochasticWeightAverage(
        model,
        *args,
        device=device,
        use_buffers=use_buffers,
        avg_fn=avg_fn,
        **kwargs,
    )


class AdamW:
    @staticmethod
    def float(
        model_or_params: Union[
            nn.Module, Iterable[nn.Parameter], Sequence[Dict[str, Any]]
        ],
        lr: float,
        *args: Any,
        weight_decay: float = 0.0,
        metadata: Optional[Dataset[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> optim.Optimizer:
        return AdamW._try_backends(
            model_or_params,
            lr,
            weight_decay,
            metadata,
            logger,
            kwargs,
            mode="float",
        )

    @staticmethod
    def integer(
        model_or_params: Union[
            nn.Module, Iterable[nn.Parameter], Sequence[Dict[str, Any]]
        ],
        lr: float,
        *args: Any,
        weight_decay: float = 0.0,
        metadata: Optional[Dataset[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> optim.Optimizer:
        return AdamW._try_backends(
            model_or_params,
            lr,
            weight_decay,
            metadata,
            logger,
            kwargs,
            mode="integer",
        )

    @staticmethod
    def _try_backends(
        model_or_params: Union[
            nn.Module, Iterable[nn.Parameter], Sequence[Dict[str, Any]]
        ],
        lr: float,
        weight_decay: float,
        metadata: Optional[Dataset[Any]],
        logger: Optional[Callable[[str], None]],
        kwargs: Dict[str, Any],
        mode: str,
    ) -> optim.Optimizer:
        params = _coerce_params(model_or_params)
        dev, meta = _dataset_for_device(
            (
                model_or_params
                if isinstance(model_or_params, nn.Module)
                else params
            ),
            metadata,
        )

        precision = PrecisionPolicy.from_metadata(
            device=dev, metadata=meta, logger=_LOGGER
        )
        master_float = getattr(precision, "master_float", torch.float32)

        float_param_dtypes = sorted(
            {
                p.dtype
                for p in _iter_param_tensors(params)
                if p.is_floating_point() and bool(getattr(p, "requires_grad", False))
            },
            key=lambda d: str(d),
        )
        if len(float_param_dtypes) > 1:
            raise RuntimeError(
                "AdamW parameter dtype mismatch detected (optimizer requires a single master dtype). "
                f"Found dtypes: {', '.join(str(d) for d in float_param_dtypes)}"
            )
        param_dtype = float_param_dtypes[0] if float_param_dtypes else master_float
        if param_dtype != master_float:
            if bool(getattr(meta, "has_scale", False)):
                raise RuntimeError(
                    "AdamW requires parameters to use PrecisionPolicy.master_float before optimizer creation. "
                    f"param_dtype={param_dtype}, master_float={master_float}. "
                    "Cast model parameters (storage dtype) to master_float first."
                )

        allow_torchao = env_bool("ENN_OPTIMIZER_ALLOW_TORCHAO", default=False)


        def _attempt_load(
            mod_name: str,
            cls_name: str,
            extra: List[Dict[str, Any]],
            *,
            enabled: bool = True,
            reason: Optional[str] = None,
        ) -> Optional[optim.Optimizer]:
            fq = f"{mod_name}.{cls_name}"
            present = False
            with contextlib.suppress(Exception):
                present = importlib.util.find_spec(mod_name) is not None
            if not enabled:
                extra.append(
                    {
                        "name": fq,
                        "ok": None,
                        "present": bool(present),
                        "skipped": True,
                        "reason": str(reason or "disabled"),
                    }
                )
                return None
            if not present:
                extra.append(
                    {
                        "name": fq,
                        "ok": False,
                        "present": False,
                        "error": "module not found",
                    }
                )
                return None
            try:
                mod = __import__(mod_name, fromlist=[cls_name])
                cls = getattr(mod, cls_name)
                return cls(params, **_coerce_kwargs(cls, common_kwargs))
            except Exception as exc:
                extra.append(
                    {
                        "name": fq,
                        "ok": False,
                        "present": True,
                        "error": str(exc),
                    }
                )
                return None

        attempts: List[Dict[str, Any]] = []
        selected_opt: Optional[optim.Optimizer] = None
        selected_name: Optional[str] = None
        common_kwargs: Dict[str, Any] = {
            "lr": lr,
            "weight_decay": weight_decay,
            **kwargs,
        }
        te_enabled = (
            dev.type == "cuda" and master_float == torch.float32 and param_dtype == torch.float32
        )
        selected_opt = _attempt_load(
            "transformer_engine.pytorch.optimizers",
            "FusedAdam",
            attempts,
            enabled=te_enabled,
            reason=(
                None
                if te_enabled
                else f"requires master_float=float32 (got {master_float})"
            ),
        )
        if selected_opt:
            selected_name = "te.FusedAdam"

        ao_enabled = (
            allow_torchao
            and dev.type == "cuda"
            and master_float == torch.float32
            and param_dtype == torch.float32
            and bool(getattr(meta, "has_scale", False))
            and not bool(getattr(meta, "has_nonfinite", False))
        )

        if not selected_opt and mode == "float" and dev.type == "cuda":
            float8_dtypes = StatelessAutocast.float8_formats()
            hw_ok, _ = Dataset.is_float8_supported(dev)
            safe_fp8 = ao_enabled and hw_ok and any(
                is_scale_safe(dtype, meta, safety_margin=2.0) for dtype in float8_dtypes
            )
            if safe_fp8:
                for pkg in ("torchao.optim", "torchao.prototype.float8.optim"):
                    selected_opt = _attempt_load(pkg, "AdamWFp8", attempts)
                    if selected_opt:
                        selected_name = "torchao.AdamWFp8"
                        break
            else:
                reason = "disabled by ENN_OPTIMIZER_ALLOW_TORCHAO=0"
                if allow_torchao and not bool(getattr(meta, "has_scale", False)):
                    reason = "requires has_scale=true (scale stats missing)"
                elif allow_torchao and bool(getattr(meta, "has_nonfinite", False)):
                    reason = "requires finite scale stats (has_nonfinite=true)"
                elif allow_torchao and master_float != torch.float32:
                    reason = f"requires master_float=float32 (got {master_float})"
                elif allow_torchao and not hw_ok:
                    reason = "hardware does not support float8"
                elif allow_torchao and not safe_fp8:
                    reason = "float8 not scale-safe for this dataset"
                for pkg in ("torchao.optim", "torchao.prototype.float8.optim"):
                    _attempt_load(
                        pkg,
                        "AdamWFp8",
                        attempts,
                        enabled=False,
                        reason=reason,
                    )
        if not selected_opt:
            quant_bits = getattr(meta, "int_quant_bits", None)
            use_int = ao_enabled and (
                quant_bits in (4, 8)
                or (
                    mode == "integer"
                    and bool(getattr(meta, "scale_is_integral", None) is not False)
                )
            )
            if use_int:

                def _scale_safe_int(meta: Any, bits: int) -> bool:
                    if not bool(getattr(meta, "has_scale", False)):
                        return False
                    if bool(getattr(meta, "has_nonfinite", False)):
                        return False
                    if getattr(meta, "scale_is_integral", None) is False:
                        return False
                    lo, hi = (-128.0, 127.0) if bits == 8 else (-8.0, 7.0)
                    if (
                        mn := getattr(meta, "scale_min_value", None)
                    ) is not None and (
                        mx := getattr(meta, "scale_max_value", None)
                    ) is not None:
                        try:
                            mn_f, mx_f = float(mn), float(mx)
                        except Exception:
                            return False
                        if not (math.isfinite(mn_f) and math.isfinite(mx_f)):
                            return False
                        return mn_f >= lo and mx_f <= hi
                    max_abs = getattr(meta, "scale_max_abs", None)
                    if max_abs is None:
                        return False
                    try:
                        max_abs_f = float(abs(max_abs))
                    except Exception:
                        return False
                    return math.isfinite(max_abs_f) and max_abs_f <= hi

                safe_int8 = _scale_safe_int(meta, 8)
                safe_int4 = _scale_safe_int(meta, 4)
                target_bits: Optional[int] = None
                if quant_bits == 8:
                    if safe_int8:
                        target_bits = 8
                elif quant_bits == 4:
                    if safe_int4:
                        target_bits = 4
                else:
                    if Dataset.is_int8_supported(dev)[0] and safe_int8:
                        target_bits = 8
                    elif safe_int4:
                        target_bits = 4
                if target_bits is not None:
                    cls_name = f"AdamW{target_bits}bit"
                    for pkg in (
                        "torchao.optim",
                        "torchao.prototype.low_bit_optim",
                    ):
                        selected_opt = _attempt_load(pkg, cls_name, attempts)
                        if selected_opt:
                            selected_name = f"torchao.{cls_name}"
                            break
            else:
                if quant_bits in (4, 8) or mode == "integer":
                    reason = "disabled by ENN_OPTIMIZER_ALLOW_TORCHAO=0"
                    if allow_torchao and not bool(getattr(meta, "has_scale", False)):
                        reason = "requires has_scale=true (scale stats missing)"
                    elif allow_torchao and bool(getattr(meta, "has_nonfinite", False)):
                        reason = "requires finite scale stats (has_nonfinite=true)"
                    elif allow_torchao and master_float != torch.float32:
                        reason = f"requires master_float=float32 (got {master_float})"
                    for cls_name in ("AdamW8bit", "AdamW4bit"):
                        for pkg in (
                            "torchao.optim",
                            "torchao.prototype.low_bit_optim",
                        ):
                            _attempt_load(
                                pkg,
                                cls_name,
                                attempts,
                                enabled=False,
                                reason=reason,
                            )
        if not selected_opt:
            flags = optimal_optimizer_params(
                dev, use_foreach=None, use_fused=False
            )
            selected_opt = optim.AdamW(
                params,
                **_coerce_kwargs(optim.AdamW, {**common_kwargs, **flags}),
            )
            selected_name = "torch.optim.AdamW"
        scale_info = {
            k: _to_immutable(getattr(meta, k, None))
            for k in (
                "has_scale",
                "has_nonfinite",
                "scale_max_abs",
                "scale_min_positive",
                "scale_min_value",
                "scale_max_value",
                "underflow_action",
                "int_quant_bits",
                "scale_is_integral",
            )
        }
        payload = {
            "mode": f"adamw-{mode}",
            "device": f"{dev.type}:{dev.index}",
            "selected": selected_name,
            "attempts": attempts,
            "precision": {
                "master_float": str(master_float),
                "param_dtype": str(param_dtype),
                "allow_torchao": bool(allow_torchao),
            },
            "scale": scale_info,
        }
        _log_optimizer(
            logger,
            (
                "opt",
                f"adamw-{mode}",
                dev.type,
                dev.index,
                tuple(scale_info.values()),
                str(master_float),
                bool(allow_torchao),
            ),
            payload,
        )
        return selected_opt


class ExponentialMovingAverage(nn.Module):
    def __init__(
        self: Self,
        model: nn.Module,
        *args: Any,
        decay: float = 0.9999,
        metadata: Optional["Dataset[Any]"] = None,
        pin_memory: bool | None = None,
        update_every: int = 1,
        non_blocking: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.decay = float(decay)
        self.pin_memory = bool(pin_memory) if pin_memory is not None else False
        self.update_every = max(1, int(update_every))
        self.non_blocking = (
            bool(non_blocking) if non_blocking is not None else False
        )
        meta = metadata if isinstance(metadata, Dataset) else None
        dev = torch.device(
            StatelessAutocast.coerce_metadata(get_device(), metadata=meta).device
        )
        self.master_float, self.master_int = _master_cpu_dtypes(dev, meta)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.collected: Dict[str, torch.Tensor] = {}
        self._scratch: Dict[torch.dtype, torch.Tensor] = {}
        self._step: int = 0
        with torch.no_grad():
            self._init_shadow(model)

    def _iter_named_params(
        self: Self, model: nn.Module
    ) -> Iterator[tuple[str, torch.Tensor]]:
        for name, p in model.named_parameters(recurse=True):
            if not isinstance(p, torch.Tensor):
                continue
            if getattr(p, "is_meta", False):
                continue
            if p.numel() <= 0:
                continue
            if (not p.is_floating_point()) and (not p.is_complex()):
                continue
            yield name, p

    def _target_dtype(self: Self, t: torch.Tensor) -> torch.dtype:
        if t.is_floating_point():
            return self.master_float
        if t.is_complex():
            return (
                torch.complex128
                if t.dtype == torch.complex128
                else torch.complex64
            )
        return self.master_int

    def _scratch_view(
        self: Self, numel: int, shape: torch.Size, dtype: torch.dtype
    ) -> torch.Tensor:
        buf = self._scratch.get(dtype, None)
        if buf is None or buf.numel() < numel:
            self._scratch[dtype] = torch.empty(
                (int(numel),),
                device="cpu",
                dtype=dtype,
                pin_memory=bool(self.pin_memory),
            )
            buf = self._scratch[dtype]
        return buf[: int(numel)].view(shape)

    def _init_shadow(self: Self, model: nn.Module) -> None:
        shadow: Dict[str, torch.Tensor] = {}
        for name, p in self._iter_named_params(model):
            dt = self._target_dtype(p)
            shadow[name] = _cpu_offload(
                p.detach(),
                dtype=dt,
                pin_memory=bool(self.pin_memory),
                non_blocking=bool(self.non_blocking),
            )
        self.shadow = shadow

    @torch.no_grad()
    def update(
        self: Self, model: nn.Module, optimizer: object | None = None
    ) -> None:
        _ = optimizer
        self._step += 1
        if self.update_every > 1 and (self._step % self.update_every) != 0:
            return
        decay = float(self.decay)
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"EMA decay must be in [0, 1). got decay={decay}")
        weight = 1.0 - decay
        shadow = self.shadow
        pin_memory = bool(self.pin_memory)
        for name, p in self._iter_named_params(model):
            dt = self._target_dtype(p)
            cur = shadow.get(name, None)
            if (
                cur is None
                or (not isinstance(cur, torch.Tensor))
                or cur.shape != p.shape
                or cur.dtype != dt
            ):
                shadow[name] = _cpu_offload(
                    p.detach(),
                    dtype=dt,
                    pin_memory=pin_memory,
                    non_blocking=False,
                )
                continue
            try:
                tmp = self._scratch_view(p.numel(), p.shape, dt)
                tmp.copy_(p.detach(), non_blocking=False)
            except Exception:
                tmp = _cpu_offload(
                    p.detach(),
                    dtype=dt,
                    pin_memory=pin_memory,
                    non_blocking=False,
                )
            if cur.is_floating_point() or cur.is_complex():
                cur.lerp_(tmp, weight)
            else:
                cur.copy_(tmp)

    @torch.no_grad()
    def store(
        self: Self, model: nn.Module, optimizer: object | None = None
    ) -> None:
        _ = optimizer
        self.collected = {}
        collected = self.collected
        for name, p in self._iter_named_params(model):
            collected[name] = _cpu_offload(
                p.detach(),
                dtype=p.dtype,
                pin_memory=bool(self.pin_memory),
                non_blocking=False,
            )

    @torch.no_grad()
    def restore(
        self: Self, model: nn.Module, optimizer: object | None = None
    ) -> None:
        _ = optimizer
        if not self.collected:
            return
        for name, p in self._iter_named_params(model):
            prev = self.collected.get(name, None)
            if not isinstance(prev, torch.Tensor):
                continue
            try:
                p.copy_(
                    prev.to(device=p.device, dtype=p.dtype), non_blocking=False
                )
            except Exception:
                with contextlib.suppress(Exception):
                    p.data = prev.to(device=p.device, dtype=p.dtype)

    @torch.no_grad()
    def apply(
        self: Self, model: nn.Module, optimizer: object | None = None
    ) -> None:
        _ = optimizer
        shadow = self.shadow
        if not shadow:
            return
        for name, p in self._iter_named_params(model):
            ema_v = shadow.get(name, None)
            if not isinstance(ema_v, torch.Tensor):
                continue
            try:
                p.copy_(
                    ema_v.to(device=p.device, dtype=p.dtype),
                    non_blocking=False,
                )
            except Exception:
                with contextlib.suppress(Exception):
                    p.data = ema_v.to(device=p.device, dtype=p.dtype)


class StochasticWeightAverage(nn.Module):
    def __init__(
        self: Self,
        model: nn.Module,
        *args: Any,
        metadata: Optional["Dataset[Any]"] = None,
        update_every: int = 1,
        avg_dtype: Optional[torch.dtype] = None,
        pin_memory: bool | None = None,
        non_blocking: bool | None = None,
        stream_chunk_mb: int = 8,
        stream_max_inflight_mb: int = 64,
        use_mmap: bool | None = None,
        mmap_dir: str | None = None,
        mmap_prefix: str = "enn_swa_shadow",
        mmap_cleanup: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._model = model
        self._has_bn = _has_batchnorm_modules(model)
        meta = metadata if isinstance(metadata, Dataset) else None
        dev = torch.device(
            StatelessAutocast.coerce_metadata(get_device(), metadata=meta).device
        )
        self._model_device = dev
        self._master_float, self._master_int = _master_cpu_dtypes(dev, meta)
        self._master_complex = (
            torch.complex128
            if self._master_float == torch.float64
            else torch.complex64
        )
        self._avg_dtype = avg_dtype
        self._pin_memory = (
            bool(pin_memory) if pin_memory is not None else False
        )
        self._non_blocking = (
            bool(non_blocking)
            if non_blocking is not None
            else bool(self._pin_memory)
        )
        self._update_every = max(1, int(update_every))
        self._step: int = 0
        self._n_averaged: int = 0
        self._shadow: Dict[str, torch.Tensor] = {}
        self._scratch: Dict[torch.dtype, torch.Tensor] = {}
        self._stream_chunk_bytes = max(
            1 << 20, int(stream_chunk_mb) * 1024 * 1024
        )
        self._stream_max_inflight_bytes = max(
            int(self._stream_chunk_bytes),
            int(stream_max_inflight_mb) * 1024 * 1024,
        )
        self._lock = Mutex()
        self._use_mmap = bool(use_mmap) if use_mmap is not None else True
        self._mmap_dir: str | None = None
        self._mmap_prefix = str(mmap_prefix or "enn_swa_shadow")
        self._mmap_cleanup = (
            bool(mmap_cleanup) if mmap_cleanup is not None else False
        )
        self._mmap_created_temp = False
        self._mmap_files: Dict[torch.dtype, str] = {}
        self._mmap_base: Dict[torch.dtype, torch.Tensor] = {}
        self._mmap_layout: Dict[
            str, tuple[torch.dtype, int, int, tuple[int, ...]]
        ] = {}
        if self._use_mmap:
            try:
                self._init_mmap_shadow(model, mmap_dir=mmap_dir)

                if mmap_cleanup is None and self._mmap_created_temp:
                    self._mmap_cleanup = True
            except Exception:
                self._use_mmap = False
                self._mmap_dir = None
                self._mmap_files.clear()
                self._mmap_base.clear()
                self._mmap_layout.clear()

    @property
    def n_averaged(self: Self) -> int:
        return int(self._n_averaged)

    @property
    def shadow(self: Self) -> Dict[str, torch.Tensor]:
        return self._shadow

    def _iter_named_params(
        self: Self, model: nn.Module
    ) -> Iterator[tuple[str, torch.Tensor]]:
        for name, p in model.named_parameters(recurse=True):
            if not isinstance(p, torch.Tensor):
                continue
            if getattr(p, "is_meta", False) or p.device.type == "meta":
                continue
            if p.numel() <= 0:
                continue
            yield name, p

    def _target_dtype(self: Self, t: torch.Tensor) -> torch.dtype:
        if self._avg_dtype is not None:
            return self._avg_dtype
        if t.is_floating_point():
            return self._master_float
        if t.is_complex():
            return self._master_complex
        if t.dtype == torch.bool:
            return torch.bool
        return self._master_int

    @staticmethod
    def _dtype_tag(dt: torch.dtype) -> str:
        s = str(dt)
        if s.startswith("torch."):
            s = s.split("torch.", 1)[1]
        return (
            s.replace("bfloat16", "bf16")
            .replace("float16", "f16")
            .replace("float32", "f32")
            .replace("float64", "f64")
            .replace("complex64", "c64")
            .replace("complex128", "c128")
            .replace("int64", "i64")
            .replace("int32", "i32")
            .replace("int16", "i16")
            .replace("int8", "i8")
            .replace("uint8", "u8")
            .replace("bool", "b1")
        )

    def _init_mmap_shadow(
        self: Self, model: nn.Module, *args: Any, mmap_dir: str | None
    ) -> None:
        items: list[tuple[str, torch.dtype, tuple[int, ...], int]] = []
        if mmap_dir:
            d = str(mmap_dir)
            os.makedirs(d, exist_ok=True)
            self._mmap_dir = d
            self._mmap_created_temp = False
        else:
            self._mmap_dir = tempfile.mkdtemp(prefix=f"{self._mmap_prefix}_")
            self._mmap_created_temp = True
        for name, p in self._iter_named_params(model):
            dt = self._target_dtype(p.detach())
            shape = tuple(int(x) for x in p.shape)
            numel = int(p.numel())
            items.append((name, dt, shape, numel))
        if not items:
            self._use_mmap = False
            return
        items.sort(key=lambda x: x[0])
        totals: Dict[torch.dtype, int] = {}
        for _, dt, _, n in items:
            totals[dt] = int(totals.get(dt, 0)) + int(n)
        for dt, total in totals.items():
            if int(total) <= 0:
                continue
            tag = self._dtype_tag(dt)
            base_path = os.path.join(
                str(self._mmap_dir), f"{self._mmap_prefix}.{tag}.bin"
            )
            if os.path.exists(base_path):
                base_path = os.path.join(
                    str(self._mmap_dir),
                    f"{self._mmap_prefix}.{tag}.{os.urandom(4).hex()}.bin",
                )
            base = torch.from_file(
                base_path,
                shared=True,
                size=int(total),
                dtype=dt,
                device="cpu",
            )
            self._mmap_files[dt] = base_path
            self._mmap_base[dt] = base

        offsets: Dict[torch.dtype, int] = {dt: 0 for dt in totals}
        shadow: Dict[str, torch.Tensor] = {}
        layout: Dict[str, tuple[torch.dtype, int, int, tuple[int, ...]]] = {}
        for name, dt, shape, numel in items:
            base = self._mmap_base.get(dt)
            if base is None:
                continue
            off = int(offsets.get(dt, 0))
            view = base[off : off + int(numel)].view(shape)
            shadow[name] = view
            layout[name] = (dt, off, int(numel), shape)
            offsets[dt] = off + int(numel)
        self._shadow = shadow
        self._mmap_layout = layout

    def _scratch_view(
        self: Self, numel: int, dtype: torch.dtype
    ) -> torch.Tensor:
        n = int(numel)
        if n <= 0:
            return torch.empty((0,), device="cpu", dtype=dtype)
        buf = self._scratch.get(dtype, None)
        if buf is None or buf.numel() < n:
            self._scratch[dtype] = torch.empty(
                (n,),
                device="cpu",
                dtype=dtype,
                pin_memory=bool(self._pin_memory),
            )
            buf = self._scratch[dtype]
        return buf[:n]

    def _sync_device(self: Self, dev_type: str) -> None:
        try:
            if dev_type == "cuda" and hasattr(torch, "cuda"):
                torch.cuda.synchronize()
            elif dev_type == "xpu" and hasattr(torch, "xpu"):
                torch.xpu.synchronize()
            elif dev_type == "mps" and hasattr(torch, "mps"):
                torch.mps.synchronize()
        except Exception:
            pass

    def _stream_copy_(
        self: Self,
        dst: torch.Tensor,
        src: torch.Tensor,
        *args: Any,
        dtype: torch.dtype,
    ) -> None:
        v = src.detach()
        if getattr(v, "is_meta", False) or v.device.type == "meta":
            return
        try:
            if hasattr(v, "to_local"):
                v = v.to_local()
        except Exception:
            pass
        v_flat = v.reshape(-1)
        dst_flat = dst.reshape(-1)
        total = int(v_flat.numel())
        if total <= 0:
            return
        dt_size = int(
            torch.empty((), dtype=dtype, device="cpu").element_size()
        )
        chunk_elems = max(1, int(self._stream_chunk_bytes // max(1, dt_size)))
        for off in range(0, total, chunk_elems):
            n = min(chunk_elems, total - off)
            buf = self._scratch_view(n, dtype)
            buf.copy_(
                v_flat[off : off + n], non_blocking=bool(self._non_blocking)
            )
            if bool(self._non_blocking) and v.device.type != "cpu":
                self._sync_device(v.device.type)
            dst_flat[off : off + n].copy_(buf, non_blocking=False)

    def _stream_avg_(
        self: Self,
        dst: torch.Tensor,
        src: torch.Tensor,
        *args: Any,
        n_averaged: int,
        dtype: torch.dtype,
    ) -> None:
        v = src.detach()
        if getattr(v, "is_meta", False) or v.device.type == "meta":
            return
        try:
            if hasattr(v, "to_local"):
                v = v.to_local()
        except Exception:
            pass
        v_flat = v.reshape(-1)
        dst_flat = dst.reshape(-1)
        total = int(v_flat.numel())
        if total <= 0:
            return
        inv_n1 = 1.0 / float(int(n_averaged) + 1)
        one_m = 1.0 - inv_n1
        dt_size = int(
            torch.empty((), dtype=dtype, device="cpu").element_size()
        )
        chunk_elems = max(1, int(self._stream_chunk_bytes // max(1, dt_size)))
        for off in range(0, total, chunk_elems):
            n = min(chunk_elems, total - off)
            buf = self._scratch_view(n, dtype)
            buf.copy_(
                v_flat[off : off + n], non_blocking=bool(self._non_blocking)
            )
            if bool(self._non_blocking) and v.device.type != "cpu":
                self._sync_device(v.device.type)
            dst_chunk = dst_flat[off : off + n]
            dst_chunk.mul_(one_m).add_(buf, alpha=inv_n1)

    def update(self: Self, model: nn.Module | None = None) -> None:
        if model is None:
            model = self._model
        with self._lock:
            self._step += 1
            if (self._step % int(self._update_every)) != 0:
                return
            n = int(self._n_averaged)
            shadow = self._shadow
            for name, p in self._iter_named_params(model):
                v = p.detach()
                dt = self._target_dtype(v)
                cur = shadow.get(name, None)

                if (
                    cur is None
                    or (not torch.is_tensor(cur))
                    or tuple(cur.shape) != tuple(v.shape)
                    or cur.dtype != dt
                ):
                    cur = torch.empty(
                        v.shape,
                        dtype=dt,
                        device="cpu",
                        pin_memory=False,
                    )
                    shadow[name] = cur
                if n == 0:
                    self._stream_copy_(cur, v, dtype=dt)
                    continue

                if cur.is_floating_point() or cur.is_complex():
                    self._stream_avg_(cur, v, n_averaged=n, dtype=dt)
                else:
                    self._stream_copy_(cur, v, dtype=dt)
            self._n_averaged = int(n + 1)

    @contextlib.contextmanager
    def reduction(self: Self, target: nn.Module) -> Any:
        if target is None:
            yield target
            return
        backup: Dict[str, torch.Tensor] = {}
        try:
            with torch.no_grad():
                for k, p in target.named_parameters(recurse=True):
                    if k in self._shadow:
                        backup[k] = p.detach().clone()
                        p.detach().copy_(self._shadow[k], non_blocking=False)
            yield target
        finally:
            with torch.no_grad():
                for k, p in target.named_parameters(recurse=True):
                    if k in backup:
                        p.detach().copy_(backup[k], non_blocking=False)

    def apply(self: Self, target: nn.Module | None = None) -> None:
        if target is None:
            target = self._model
        if target is None:
            return
        with torch.no_grad():
            for k, p in target.named_parameters(recurse=True):
                if k not in self._shadow:
                    continue
                src = self._shadow[k]
                if not torch.is_tensor(src):
                    continue
                if getattr(src, "is_meta", False) or src.device.type == "meta":
                    continue
                p.detach().copy_(src, non_blocking=False)

    def checkpoint_state_dict(
        self: Self,
        model: nn.Module,
        *args: Any,
        include_buffers: bool = True,
        max_buffer_mb: int = 25,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        shadow = self._shadow
        with torch.no_grad():
            for name, p in model.named_parameters(recurse=True):
                sv = shadow.get(name, None)
                tv = sv if torch.is_tensor(sv) else p
                if not torch.is_tensor(tv):
                    continue
                if getattr(tv, "is_meta", False) or tv.device.type == "meta":
                    continue
                t = tv.detach()
                if t.device.type != "cpu":
                    t = t.to("cpu")
                out[name] = t
            if include_buffers:
                max_bytes = int(max(0, int(max_buffer_mb)) * 1024 * 1024)
                for name, b in model.named_buffers(recurse=True):
                    if not torch.is_tensor(b) or b.numel() <= 0:
                        continue
                    if getattr(b, "is_meta", False) or b.device.type == "meta":
                        continue
                    try:
                        if (
                            max_bytes
                            and (b.numel() * b.element_size()) > max_bytes
                        ):
                            continue
                    except Exception:
                        pass
                    t = b.detach()
                    if t.device.type != "cpu":
                        t = t.to("cpu")
                    out[name] = t
        return out

    def update_batch_norm(
        self: Self,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device | None = None,
    ) -> None:
        if not self._has_bn:
            return
        if device is None:
            device = self._model_device
        with self.reduction(self._model):
            update_bn(dataloader, self._model, device=device)

    def apply_and_update_batch_norm(
        self: Self,
        dataloader: torch.utils.data.DataLoader,
        *,
        model: nn.Module | None = None,
        device: torch.device | None = None,
    ) -> None:
        if model is None:
            model = self._model
        if model is None or not self._has_bn:
            return
        if device is None:
            device = self._model_device
        self.apply(model)
        update_bn(dataloader, model, device=device)

    def close(self: Self) -> None:
        with self._lock:
            try:
                self._shadow.clear()
            except Exception:
                pass
            try:
                self._scratch.clear()
            except Exception:
                pass
            try:
                self._mmap_base.clear()
            except Exception:
                pass
            try:
                self._mmap_files.clear()
            except Exception:
                pass
            try:
                self._mmap_layout.clear()
            except Exception:
                pass
            if self._mmap_dir and bool(self._mmap_cleanup):
                try:
                    import gc
                    import shutil

                    gc.collect()
                    shutil.rmtree(self._mmap_dir, ignore_errors=True)
                except Exception:
                    pass
            self._mmap_dir = None

    def save_state_dict(self: Self) -> Dict[str, Any]:
        return {
            "n_averaged": int(self._n_averaged),
            "update_every": int(self._update_every),
            "avg_dtype": (
                str(self._avg_dtype) if self._avg_dtype is not None else None
            ),
            "use_mmap": bool(self._use_mmap),
            "mmap_dir": str(self._mmap_dir) if self._mmap_dir else None,
            "mmap_files": dict(self._mmap_files),
            "mmap_layout": {
                k: (
                    str(v[0]),
                    int(v[1]),
                    int(v[2]),
                    tuple(int(x) for x in v[3]),
                )
                for k, v in self._mmap_layout.items()
            },
        }

    def load_state_dict(self: Self, state: Dict[str, Any]) -> None:
        try:
            self._n_averaged = int(state.get("n_averaged", 0))
        except Exception:
            self._n_averaged = 0
        try:
            self._update_every = max(
                1, int(state.get("update_every", self._update_every))
            )
        except Exception:
            pass
