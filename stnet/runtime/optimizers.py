# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import copy
import inspect
import json
import logging
import math
import threading
from collections import OrderedDict
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn, optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from ..core.precision import Autocast, PrecisionPolicy, is_scale_safe
from ..core.system import get_device, optimal_optimizer_params
from ..data.pipeline import Dataset
from ..nn.architecture import ModelPolicy


_LOGGER = logging.getLogger(__name__)

_OPT_LOGGED_KEYS: "OrderedDict[object, None]" = OrderedDict()
_OPT_LOGGED_MAX: int = 128
_OPT_LOGGED_LOCK = threading.Lock()


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


def _iter_batch(items: Iterable[Any], key: str) -> Iterator[torch.Tensor]:
    for item in items:
        tensor = None
        if isinstance(item, (TensorDictBase, dict)):
            tensor = item.get(key)
        elif isinstance(item, torch.Tensor):
            tensor = item
        elif isinstance(item, (tuple, list)) and item and torch.is_tensor(item[0]):
            tensor = item[0]
        if isinstance(tensor, torch.Tensor):
            yield tensor


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
    key = key or ("opt", payload.get("mode"), payload.get("device"), payload.get("selected"))
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
        msg = "[OPT][DECISION] " + json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        msg = f"[OPT][DECISION] {payload}"
    if logger is not None:
        logger(msg)
        return
    log_fn = getattr(_LOGGER, str(level).lower(), _LOGGER.info)
    log_fn(msg)


@lru_cache(maxsize=256)
def _get_expected_args(ctor: Any) -> Optional[frozenset[str]]:
    try:
        sig = inspect.signature(ctor)
        return (
            None
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
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
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                return dict(kwargs)
            allowed = frozenset(sig.parameters.keys())
        except Exception:
            return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in allowed}


def _dataset_for_device(
    model_or_params: Union[nn.Module, Iterable[nn.Parameter], Sequence[Dict[str, Any]]],
    metadata: Optional["Dataset[Any]"] = None,
) -> Tuple[torch.device, "Dataset[Any]"]:
    ref_tensor: Optional[torch.Tensor] = None
    if isinstance(model_or_params, nn.Module):
        ref_tensor = ModelPolicy._peek_layer(model_or_params)
    elif metadata is None and isinstance(model_or_params, Sequence) and model_or_params:
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
        else ref_tensor.device
        if ref_tensor is not None
        else get_device()
    )
    meta = Autocast.coerce_metadata(dev, metadata=metadata)
    return torch.device(meta.device), meta


def _coerce_params(
    model_or_params: Union[nn.Module, Iterable[nn.Parameter], Sequence[Dict[str, Any]]],
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
                out = torch.empty(src.shape, device="cpu", dtype=dtype, pin_memory=True)
                out.copy_(src, non_blocking=nb)
                return out
            with contextlib.suppress(Exception):
                return src.clone().pin_memory()
        return src.clone()
    if pin_memory:
        with contextlib.suppress(Exception):
            out = torch.empty(src.shape, device="cpu", dtype=dtype, pin_memory=True)
            out.copy_(src, non_blocking=nb)
            return out
    return src.to(device="cpu", dtype=dtype)


def _safe_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    if torch.is_tensor(dst) and torch.is_tensor(src) and dst.data_ptr() != src.data_ptr():
        with contextlib.suppress(Exception):
            dst.copy_(src)
            return
        with contextlib.suppress(Exception):
            dst.copy_(src.to(device=dst.device, dtype=dst.dtype))


def _sync_optimizer_state(optimizer: optim.Optimizer) -> None:
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
                        target_dtype = (
                            p.dtype
                            if (p_is_float and v_is_float and v.shape == p.shape)
                            else v.dtype
                        )
                        p_state[k] = v.to(device=p.device, dtype=target_dtype)
                    except Exception:
                        try:
                            p_state[k] = v.to(device=p.device)
                        except Exception:
                            pass
    except Exception:
        return


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
        model, *args, device=device, use_buffers=use_buffers, avg_fn=avg_fn, **kwargs
    )


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
        return AdamW._try_backends(
            model_or_params, lr, weight_decay, metadata, logger, kwargs, mode="float"
        )

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
        return AdamW._try_backends(
            model_or_params, lr, weight_decay, metadata, logger, kwargs, mode="integer"
        )

    @staticmethod
    def _try_backends(
        model_or_params: Union[nn.Module, Iterable[nn.Parameter], Sequence[Dict[str, Any]]],
        lr: float,
        weight_decay: float,
        metadata: Optional[Dataset[Any]],
        logger: Optional[Callable[[str], None]],
        kwargs: Dict[str, Any],
        mode: str,
    ) -> optim.Optimizer:
        params = _coerce_params(model_or_params)
        dev, meta = _dataset_for_device(
            model_or_params if isinstance(model_or_params, nn.Module) else params,
            metadata,
        )

        def _attempt_load(
            mod_name: str,
            cls_name: str,
            extra: List[Dict[str, Any]],
        ) -> Optional[optim.Optimizer]:
            try:
                mod = __import__(mod_name, fromlist=[cls_name])
                cls = getattr(mod, cls_name)
                return cls(params, **_coerce_kwargs(cls, common_kwargs))
            except Exception as exc:
                extra.append(
                    {
                        "name": f"{mod_name}.{cls_name}",
                        "ok": False,
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

        if dev.type == "cuda":
            selected_opt = _attempt_load(
                "transformer_engine.pytorch.optimizers", "FusedAdam", attempts
            )
            if selected_opt:
                selected_name = "te.FusedAdam"

        if not selected_opt and mode == "float" and dev.type == "cuda":
            float8_dtypes = Autocast.float8_formats()
            safe_fp8 = not getattr(meta, "has_scale", False) or any(
                is_scale_safe(dtype, meta, safety_margin=2.0) for dtype in float8_dtypes
            )
            hw_ok, _ = Dataset.is_float8_supported(dev)
            if safe_fp8 and hw_ok:
                for pkg in ("torchao.optim", "torchao.prototype.float8.optim"):
                    selected_opt = _attempt_load(pkg, "AdamWFp8", attempts)
                    if selected_opt:
                        selected_name = "torchao.AdamWFp8"
                        break

        if not selected_opt:
            quant_bits = getattr(meta, "int_quant_bits", None)
            use_int = quant_bits in (4, 8) or (
                mode == "integer"
                and getattr(meta, "has_scale", False)
                and getattr(meta, "scale_is_integral", None) is not False
            )
            if use_int:
                def _scale_safe_int(meta: Any, bits: int) -> bool:
                    if not getattr(meta, "has_scale", False):
                        return True
                    if getattr(meta, "has_nonfinite", False):
                        return False
                    if getattr(meta, "scale_is_integral", None) is False:
                        return False
                    lo, hi = (-128.0, 127.0) if bits == 8 else (-8.0, 7.0)
                    if (mn := getattr(meta, "scale_min_value", None)) is not None and (
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
                    for pkg in ("torchao.optim", "torchao.prototype.low_bit_optim"):
                        selected_opt = _attempt_load(pkg, cls_name, attempts)
                        if selected_opt:
                            selected_name = f"torchao.{cls_name}"
                            break

        if not selected_opt:
            flags = optimal_optimizer_params(dev, use_foreach=None, use_fused=False)
            selected_opt = optim.AdamW(
                params, **_coerce_kwargs(optim.AdamW, {**common_kwargs, **flags})
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
            "scale": scale_info,
        }
        _log_optimizer(
            logger,
            ("opt", f"adamw-{mode}", dev.type, dev.index, tuple(scale_info.values())),
            payload,
        )
        return selected_opt


class ExponentialMovingAverage(nn.Module):
    def __init__(
        self,
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
        self.non_blocking = bool(non_blocking) if non_blocking is not None else False
        meta = metadata if isinstance(metadata, Dataset) else None
        dev = torch.device(Autocast.coerce_metadata(get_device(), metadata=meta).device)
        self.master_float, self.master_int = _master_cpu_dtypes(dev, meta)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.collected: Dict[str, torch.Tensor] = {}
        self._scratch: Dict[torch.dtype, torch.Tensor] = {}
        self._step: int = 0
        with torch.no_grad():
            self._init_shadow(model)

    def _iter_named_params(self, model: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
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

    def _target_dtype(self, t: torch.Tensor) -> torch.dtype:
        if t.is_floating_point():
            return self.master_float
        if t.is_complex():
            return torch.complex128 if t.dtype == torch.complex128 else torch.complex64
        return self.master_int

    def _scratch_view(self, numel: int, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
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

    def _init_shadow(self, model: nn.Module) -> None:
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
    def update(self, model: nn.Module, optimizer: object | None = None) -> None:
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
    def store(self, model: nn.Module, optimizer: object | None = None) -> None:
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
    def restore(self, model: nn.Module, optimizer: object | None = None) -> None:
        _ = optimizer
        if not self.collected:
            return
        for name, p in self._iter_named_params(model):
            prev = self.collected.get(name, None)
            if not isinstance(prev, torch.Tensor):
                continue
            try:
                p.copy_(prev.to(device=p.device, dtype=p.dtype), non_blocking=False)
            except Exception:
                with contextlib.suppress(Exception):
                    p.data = prev.to(device=p.device, dtype=p.dtype)

    @torch.no_grad()
    def apply(self, model: nn.Module, optimizer: object | None = None) -> None:
        _ = optimizer
        shadow = self.shadow
        if not shadow:
            return
        for name, p in self._iter_named_params(model):
            ema_v = shadow.get(name, None)
            if not isinstance(ema_v, torch.Tensor):
                continue
            try:
                p.copy_(ema_v.to(device=p.device, dtype=p.dtype), non_blocking=False)
            except Exception:
                with contextlib.suppress(Exception):
                    p.data = ema_v.to(device=p.device, dtype=p.dtype)


class StochasticWeightAverage:
    def __init__(
        self,
        model: nn.Module,
        *args: Any,
        device: Optional[torch.device] = None,
        use_buffers: bool = True,
        avg_fn: Optional[Any] = None,
        metadata: Optional["Dataset[Any]"] = None,
        **kwargs: Any,
    ) -> None:
        self._source = model
        averaged_device = device or torch.device("cpu")
        self._averaged = AveragedModel(
            model,
            device=averaged_device,
            use_buffers=use_buffers,
            avg_fn=avg_fn,
        )
        self.n_averaged: int = 0
        dev = torch.device(Autocast.coerce_metadata(get_device(), metadata=metadata).device)
        self.master_float, _ = _master_cpu_dtypes(dev, meta=metadata)
        self.shadow: Dict[str, torch.Tensor] = {}

    @property
    def module(self) -> nn.Module:
        return self._averaged.module

    @property
    def source(self) -> nn.Module:
        return self._source

    @torch.no_grad()
    def update(self, model: Optional[nn.Module] = None) -> None:
        target = model if model is not None else self._source
        if target is None:
            raise RuntimeError("StochasticWeightAverage was initialised without a source model")
        self.update_weight(target)
        self._update_shadow(target)

    def update_weight(self, model: Optional[nn.Module] = None) -> None:
        target = model if model is not None else self._source
        if target is None:
            raise RuntimeError("StochasticWeightAverage was initialised without a source model")
        with torch.no_grad():
            try:
                self._averaged.update_parameters(target)
            except Exception as e:
                raise RuntimeError(f"SWA update failed: {e}") from e

    @torch.no_grad()
    def _update_shadow(self, model: nn.Module) -> None:
        self.n_averaged += 1
        n = int(self.n_averaged)
        inv_n = 1.0 / float(max(1, n))
        one_m = 1.0 - inv_n

        for name, p in model.named_parameters(recurse=True):
            if p is None or (not isinstance(p, torch.Tensor)):
                continue
            if not bool(getattr(p, "requires_grad", False)):
                continue
            if not bool(p.is_floating_point() or p.is_complex()):
                continue

            key = str(name)
            cur = self.shadow.get(key)
            if cur is None or (not torch.is_tensor(cur)) or cur.shape != p.shape:
                self.shadow[key] = _cpu_offload(p, self.master_float, pin_memory=False)
                continue

            src_cpu = _cpu_offload(p, dtype=cur.dtype, pin_memory=False)
            if bool(p.is_complex()):
                cur.copy_(src_cpu)
            else:
                cur.mul_(one_m).add_(src_cpu, alpha=inv_n)

    @torch.no_grad()
    def apply(self, model: nn.Module) -> None:
        if not self.shadow:
            return
        for name, p in model.named_parameters(recurse=True):
            if p is None or (not isinstance(p, torch.Tensor)):
                continue
            buf = self.shadow.get(str(name))
            if buf is None:
                continue
            _safe_copy(p, buf)

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
                    _safe_copy(param, avg_param)
                yield
            finally:
                for name, param in model.named_parameters():
                    buf = backup.get(name)
                    if buf is not None:
                        _safe_copy(param, buf)

    def update_batch_norm(
        self,
        feature_iter: Iterable[TensorDictBase | Dict[str, Any] | torch.Tensor | Any],
        *args: Any,
        device: Optional[torch.device] = None,
        in_key: str = "features",
        **kwargs: Any,
    ) -> None:
        key = str(in_key)

        adapter = _TensorDictCompat(self._averaged, key)
        update_bn(_iter_batch(feature_iter, key), adapter, device=device)

    def save_state_dict(self) -> Dict[str, Any]:
        return self._averaged.state_dict()

    def load_state_dict(self, state: Dict[str, Any]) -> Any:
        return self._averaged.load_state_dict(state)
