# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import copy
import logging
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Sequence, Tuple, Union)

import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn, optim

from ..backend.system import (get_device, is_float8_supported,
                              is_int4_supported, is_int8_supported,
                              optimal_optimizer_params)
from ..api.templates import DataPolicy
from .fx import Autocast, Fusion, is_scale_safe

try:
    from torch.optim.swa_utils import SWALR
    from torch.optim.swa_utils import AveragedModel as _SWA
    from torch.optim.swa_utils import update_bn as _update_bn
except Exception:
    _SWA = None
    SWALR = None
    _update_bn = None


_LOGGER = logging.getLogger(__name__)


def _log_info(logger: Optional[Callable[[str], None]], msg: str) -> None:
    if logger:
        logger(msg)
    else:
        _LOGGER.info(msg)


def _log_debug(logger: Optional[Callable[[str], None]], msg: str) -> None:
    _LOGGER.debug(msg)


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
        metadata: Optional["DataPolicy[Any]"] = None,
    ) -> Tuple[torch.device, "DataPolicy[Any]"]:
        ref_tensor: Optional[torch.Tensor] = None
        if isinstance(model_or_params, nn.Module):
            ref_tensor = Fusion._peek_layer(model_or_params)
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
        metadata: Optional[DataPolicy[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> optim.Optimizer:
        params = (
            model_or_params.parameters()
            if hasattr(model_or_params, "parameters")
            else model_or_params
        )
        dev, meta = AdamW._from_metadata(model_or_params, metadata)
        if hasattr(dev, "type") and dev.type == "cuda":
            try:
                from transformer_engine.pytorch.optimizers import \
                    FusedAdam as TEFusedAdam

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
                float8_dtypes = Autocast.float8_formats()
                if not any(
                    is_scale_safe(dtype, meta, safety_margin=2.0)
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
                        from transformer_engine.pytorch.optimizers import \
                            FusedAdam

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
        metadata: Optional[DataPolicy[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> optim.Optimizer:
        params = (
            model_or_params.parameters()
            if hasattr(model_or_params, "parameters")
            else model_or_params
        )
        dev, meta = AdamW._from_metadata(model_or_params, metadata)
        if hasattr(dev, "type") and dev.type == "cuda":
            try:
                from transformer_engine.pytorch.optimizers import \
                    FusedAdam as TEFusedAdam

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
                candidates: List[
                    Tuple[str, Callable[[Optional[torch.device]], Tuple[bool, str]]]
                ] = []
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
                    from torchao.prototype.low_bit_optim import (AdamW4bit,
                                                                 AdamW8bit)
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
