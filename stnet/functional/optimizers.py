# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import copy
import json
import logging
import threading
# json is used for structured optimizer decision logs.
from collections import OrderedDict
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Sequence, Tuple, Union)

import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn, optim

from ..backend.system import get_device, optimal_optimizer_params
from ..data.pipeline import Dataset
from ..model.fused import Autocast, ModelPolicy, is_scale_safe

try:
    from torch.optim.swa_utils import SWALR
    from torch.optim.swa_utils import AveragedModel as _SWA
    from torch.optim.swa_utils import update_bn as _update_bn
except Exception:
    _SWA = None
    SWALR = None
    _update_bn = None


_LOGGER = logging.getLogger(__name__)


# Deduplicate optimizer decision logs (best-effort, bounded).
_OPT_LOGGED_KEYS: "OrderedDict[object, None]" = OrderedDict()
_OPT_LOGGED_MAX: int = 128
_OPT_LOGGED_LOCK = threading.Lock()


def _log_opt_decision_once(
    logger: Optional[Callable[[str], None]],
    key: object,
    payload: Dict[str, Any],
    *,
    level: str = "info",
) -> None:
    with _OPT_LOGGED_LOCK:
        if key is None:
            key = ("opt", payload.get("mode"), payload.get("device"), payload.get("selected"))
        if key in _OPT_LOGGED_KEYS:
            try:
                _OPT_LOGGED_KEYS.move_to_end(key)
            except Exception:
                pass
            return
        _OPT_LOGGED_KEYS[key] = None
        try:
            if len(_OPT_LOGGED_KEYS) > int(_OPT_LOGGED_MAX):
                _OPT_LOGGED_KEYS.popitem(last=False)
        except Exception:
            pass
    try:
        msg = "[OPT][DECISION] " + json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        msg = f"[OPT][DECISION] {payload}"
    if logger:
        logger(msg)
        return
    if level == "debug":
        _LOGGER.debug(msg)
    else:
        _LOGGER.info(msg)
class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        self.collected: Dict[str, torch.Tensor] = {}
        self.optim_shadow: Optional[Dict[str, Any]] = None
        self.optim_collected: Optional[Dict[str, Any]] = None

    @torch.no_grad()
    def update(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
    ) -> None:
        for name, tensor in model.state_dict().items():
            if torch.is_tensor(tensor) and tensor.dtype.is_floating_point:
                self.shadow[name].mul_(self.decay).add_(tensor, alpha=(1.0 - self.decay))
            else:
                self.shadow[name] = tensor

        if optimizer is None:
            return

        state = optimizer.state_dict()
        if self.optim_shadow is None:
            self.optim_shadow = copy.deepcopy(state)
            return

        shadow = self.optim_shadow
        if "param_groups" in state:
            shadow["param_groups"] = copy.deepcopy(state["param_groups"])

        if "state" in state:
            shadow_state = shadow.setdefault("state", {})
            for pid, slot in state["state"].items():
                s_slot = shadow_state.setdefault(pid, {})
                for sk, sv in slot.items():
                    if torch.is_tensor(sv) and sv.dtype.is_floating_point:
                        if sk in s_slot and torch.is_tensor(s_slot[sk]):
                            s_slot[sk].mul_(self.decay).add_(sv, alpha=(1.0 - self.decay))
                        else:
                            s_slot[sk] = sv.detach().clone()
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
            if name in model_state:
                model_state[name].copy_(buf)

        if optimizer is not None and self.optim_shadow is not None:
            optimizer.load_state_dict(self.optim_shadow)

    @torch.no_grad()
    def store(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
    ) -> None:
        self.collected = copy.deepcopy(model.state_dict())
        if optimizer is not None:
            self.optim_collected = copy.deepcopy(optimizer.state_dict())

    @torch.no_grad()
    def restore(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
    ) -> None:
        if self.collected:
            model.load_state_dict(self.collected)
        if optimizer is not None and self.optim_collected:
            optimizer.load_state_dict(self.optim_collected)


class _TensorDictCompat(nn.Module):
    def __init__(self, averaged_module: nn.Module, key: str) -> None:
        super().__init__()
        self._averaged_module = averaged_module
        self._key = key

    def forward(self, x: torch.Tensor) -> Any:
        td = TensorDict({self._key: x}, batch_size=[x.shape[0]], device=x.device)
        return self._averaged_module(td)


def stochastic_weight_average(
    model: nn.Module,
    *args: Any,
    device: Optional[torch.device] = None,
    use_buffers: bool = True,
    avg_fn: Optional[Any] = None,
    **kwargs: Any,
) -> "StochasticWeightAverage":

    return StochasticWeightAverage(
        model, device=device, use_buffers=use_buffers, avg_fn=avg_fn
    )


class AdamW:
    @staticmethod
    def _from_metadata(
        model_or_params: Union[
            nn.Module, Iterable[nn.Parameter], Sequence[Dict[str, Any]]
        ],
        metadata: Optional["Dataset[Any]"] = None,
    ) -> Tuple[torch.device, "Dataset[Any]"]:
        ref_tensor: Optional[torch.Tensor] = None
        if isinstance(model_or_params, nn.Module):
            ref_tensor = ModelPolicy._peek_layer(model_or_params)
        if metadata is not None:
            dev = torch.device(metadata.device)
        elif ref_tensor is not None:
            dev = ref_tensor.device
        else:
            dev = get_device()
        meta = Autocast.coerce_metadata(dev, metadata=metadata)
        dev = torch.device(meta.device)
        return dev, meta

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
        params = (
            model_or_params.parameters()
            if hasattr(model_or_params, "parameters")
            else model_or_params
        )
        dev, meta = AdamW._from_metadata(model_or_params, metadata)

        dev_index = int(getattr(dev, "index", -1)) if getattr(dev, "index", None) is not None else -1
        scale_key = (
            bool(getattr(meta, "has_scale", False)),
            bool(getattr(meta, "has_nonfinite", False)),
            getattr(meta, "scale_max_abs", None),
            getattr(meta, "scale_min_positive", None),
            getattr(meta, "scale_min_value", None),
            getattr(meta, "scale_max_value", None),
            str(getattr(meta, "underflow_action", "")),
            getattr(meta, "int_quant_bits", None),
        )
        decision_key = ("opt", "adamw-float", str(getattr(dev, "type", "")), dev_index, scale_key)

        attempts: List[Dict[str, Any]] = []

        # 1) Transformer Engine fused AdamW (float)
        if hasattr(dev, "type") and dev.type == "cuda":
            try:
                from transformer_engine.pytorch.optimizers import \
                    FusedAdam as TEFusedAdam

                opt = TEFusedAdam(params, lr=lr, weight_decay=weight_decay)
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
                            "max_abs": getattr(meta, "scale_max_abs", None),
                            "min_positive": getattr(meta, "scale_min_positive", None),
                            "min_value": getattr(meta, "scale_min_value", None),
                            "max_value": getattr(meta, "scale_max_value", None),
                            "underflow_action": str(getattr(meta, "underflow_action", "")),
                            "int_quant_bits": getattr(meta, "int_quant_bits", None),
                        },
                    },
                    level="info",
                )
                return opt
            except Exception as exc:
                attempts.append({"backend": "te.FusedAdam", "ok": False, "error": str(exc)})

        # 2) FP8 optimizer path (TorchAO).
        #
        # IMPORTANT: Dataset.is_float8_supported() describes hardware support (and sometimes
        # runtime constraints). It is NOT a signal for which software backend is installed.
        # Do not branch on substrings like "TE"/"AO" in the reason string.
        if hasattr(dev, "type") and dev.type == "cuda":
            fp8_allowed = True
            if getattr(meta, "has_scale", False):
                float8_dtypes = Autocast.float8_formats()
                if not any(
                    is_scale_safe(dtype, meta, safety_margin=2.0)
                    for dtype in float8_dtypes
                ):
                    fp8_allowed = False
                    attempts.append(
                        {
                            "backend": "fp8",
                            "ok": False,
                            "reason": "data scale exceeds float8 range",
                        }
                    )

            fp8_hw_ok, fp8_hw_reason = Dataset.is_float8_supported(dev)
            if fp8_allowed and fp8_hw_ok:
                try:
                    # Some TorchAO builds require importing torchao.float8 before optimizers.
                    with contextlib.suppress(Exception):
                        __import__("torchao.float8")

                    AdamWFp8 = None
                    with contextlib.suppress(Exception):
                        from torchao.optim import AdamWFp8  # type: ignore
                    if AdamWFp8 is None:
                        # Older/newer TorchAO layouts
                        from torchao.prototype.float8.optim import AdamWFp8  # type: ignore

                    opt = AdamWFp8(params, lr=lr, weight_decay=weight_decay)
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
                                "max_abs": getattr(meta, "scale_max_abs", None),
                                "min_positive": getattr(meta, "scale_min_positive", None),
                                "min_value": getattr(meta, "scale_min_value", None),
                                "max_value": getattr(meta, "scale_max_value", None),
                                "underflow_action": str(getattr(meta, "underflow_action", "")),
                                "int_quant_bits": getattr(meta, "int_quant_bits", None),
                            },
                        },
                        level="info",
                    )
                    return opt
                except Exception as exc:
                    attempts.append(
                        {
                            "backend": "torchao.AdamWFp8",
                            "ok": False,
                            "error": str(exc),
                            "reason": str(fp8_hw_reason),
                        }
                    )
            else:
                attempts.append({"backend": "fp8", "ok": False, "reason": str(fp8_hw_reason)})

        # 3) Low-bit optimizer path (TorchAO) when explicitly requested via metadata.int_quant_bits.
        quant_bits = getattr(meta, "int_quant_bits", None)
        if quant_bits in (4, 8):
            try:
                try:
                    from torchao.optim import AdamW4bit, AdamW8bit  # type: ignore
                except ImportError:
                    from torchao.prototype.low_bit_optim import AdamW4bit, AdamW8bit  # type: ignore

                if int(quant_bits) == 8:
                    opt = AdamW8bit(params, lr=lr, weight_decay=weight_decay)
                    selected = "torchao.AdamW8bit"
                else:
                    opt = AdamW4bit(params, lr=lr, weight_decay=weight_decay)
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
                            "max_abs": getattr(meta, "scale_max_abs", None),
                            "min_positive": getattr(meta, "scale_min_positive", None),
                            "min_value": getattr(meta, "scale_min_value", None),
                            "max_value": getattr(meta, "scale_max_value", None),
                            "underflow_action": str(getattr(meta, "underflow_action", "")),
                            "int_quant_bits": getattr(meta, "int_quant_bits", None),
                        },
                    },
                    level="info",
                )
                return opt
            except Exception as exc:
                attempts.append({"backend": "torchao.AdamW(lowbit)", "ok": False, "error": str(exc), "bits": quant_bits})

        # 4) Fallback: torch.optim.AdamW
        flags: Dict[str, bool] = optimal_optimizer_params(
            dev, use_foreach=None, use_fused=False
        )
        opt = optim.AdamW(params, lr=lr, weight_decay=weight_decay, **flags)
        attempts.append({"backend": "torch.optim.AdamW", "ok": True, "flags": flags})
        _log_opt_decision_once(
            logger,
            decision_key,
            {
                "mode": "adamw-float",
                "device": f"{dev.type}:{dev_index}",
                "selected": "torch.optim.AdamW",
                "attempts": attempts,
            },
            level="info",
        )
        return opt

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
        params = (
            model_or_params.parameters()
            if hasattr(model_or_params, "parameters")
            else model_or_params
        )
        dev, meta = AdamW._from_metadata(model_or_params, metadata)

        dev_index = int(getattr(dev, "index", -1)) if getattr(dev, "index", None) is not None else -1
        scale_key = (
            bool(getattr(meta, "has_scale", False)),
            bool(getattr(meta, "has_nonfinite", False)),
            getattr(meta, "scale_max_abs", None),
            getattr(meta, "scale_min_positive", None),
            getattr(meta, "scale_min_value", None),
            getattr(meta, "scale_max_value", None),
            bool(getattr(meta, "scale_is_integral", False)),
            str(getattr(meta, "underflow_action", "")),
            getattr(meta, "int_quant_bits", None),
        )
        decision_key = ("opt", "adamw-integer", str(getattr(dev, "type", "")), dev_index, scale_key)

        decision: Dict[str, Any] = {
            "mode": "adamw-integer",
            "device": str(dev),
            "selected": None,
            "attempts": [],
            "scale": {
                "has_scale": bool(getattr(meta, "has_scale", False)),
                "has_nonfinite": bool(getattr(meta, "has_nonfinite", False)),
                "is_integral": getattr(meta, "scale_is_integral", None),
                "max_abs": getattr(meta, "scale_max_abs", None),
                "min": getattr(meta, "scale_min_value", None),
                "max": getattr(meta, "scale_max_value", None),
                "int_quant_bits": getattr(meta, "int_quant_bits", None),
            },
        }

        opt: Optional[optim.Optimizer] = None
        selected: Optional[str] = None

        # 1) Prefer Transformer Engine fused optimizer when available.
        if getattr(dev, "type", None) == "cuda":
            try:
                from transformer_engine.pytorch.optimizers import FusedAdam as TEFusedAdam

                opt = TEFusedAdam(params, lr=lr, weight_decay=weight_decay)
                selected = "transformer_engine.FusedAdam"
                decision["attempts"].append({"name": "TE.FusedAdam", "ok": True})
            except Exception as exc:
                decision["attempts"].append({"name": "TE.FusedAdam", "ok": False, "error": str(exc)})

        # 2) Low-bit integer optimizers (TorchAO) when dataset is integral and within range.
        quant_choice: Optional[str] = None
        quant_reason: Optional[str] = None
        if opt is None and bool(getattr(meta, "has_scale", False)):
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

                max_abs = float(abs(getattr(meta, "scale_max_abs", 0.0)))
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
                        quant_reason = reason
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
                    opt = AdamW8bit(params, lr=lr, weight_decay=weight_decay)
                    selected = "torchao.optim.AdamW8bit"
                    decision["attempts"].append({"name": "AO.AdamW8bit", "ok": True, "reason": str(quant_reason)})
                else:
                    opt = AdamW4bit(params, lr=lr, weight_decay=weight_decay)
                    selected = "torchao.optim.AdamW4bit"
                    decision["attempts"].append({"name": "AO.AdamW4bit", "ok": True, "reason": str(quant_reason)})
            except Exception as exc:
                decision["attempts"].append({"name": "AO.AdamW(lowbit)", "ok": False, "error": str(exc)})

        # 3) Default fallback.
        if opt is None:
            flags: Dict[str, bool] = optimal_optimizer_params(dev, use_foreach=None, use_fused=False)
            opt = optim.AdamW(params, lr=lr, weight_decay=weight_decay, **flags)
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
        self._averaged = _SWA(
            model, device=device, use_buffers=use_buffers, avg_fn=avg_fn
        )

    @property
    def module(self) -> nn.Module:
        return self._averaged.module

    @property
    def source(self) -> nn.Module:
        return self._source

    def update_weight(self, model: Optional[nn.Module] = None) -> None:
        target = model if model is not None else self._source
        if target is None:
            raise RuntimeError(
                "StochasticWeightAverage was initialised without a source model"
            )
        with torch.no_grad():
            try:
                self._averaged.update_parameters(target)
            except Exception as e:
                raise RuntimeError(f"SWA update failed: {e}")

    @contextlib.contextmanager
    def reduction(self, model: nn.Module) -> Iterator[None]:
        backup: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            try:
                for (name, param), (_, avg_param) in zip(
                    model.named_parameters(), self.module.named_parameters()
                ):
                    backup[name] = param.detach().clone()
                    param.data.copy_(avg_param.data.to(param.device, dtype=param.dtype))
                yield
            finally:
                for name, param in model.named_parameters():
                    if name in backup:
                        param.data.copy_(backup[name])

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

        def _features(it: Iterable[Any]) -> Iterator[torch.Tensor]:
            for item in it:
                tensor: Optional[torch.Tensor] = None
                if isinstance(item, TensorDictBase):
                    tensor = item.get(in_key, None)
                elif isinstance(item, torch.Tensor):
                    tensor = item
                elif isinstance(item, dict):
                    maybe = item.get(in_key)
                    tensor = maybe if torch.is_tensor(maybe) else None
                else:
                    with contextlib.suppress(Exception):
                        td = TensorDict(item, batch_size=[len(item)])
                        maybe = td.get(in_key)
                        tensor = maybe if torch.is_tensor(maybe) else None
                if tensor is not None:
                    yield tensor

        adapter = _TensorDictCompat(self._averaged, in_key)
        _update_bn(_features(feature_iter), adapter, device=device)

    def save_state_dict(self) -> Dict[str, Any]:
        return self._averaged.state_dict()

    def load_state_dict(self, state: Dict[str, Any]) -> Any:
        return self._averaged.load_state_dict(state)
