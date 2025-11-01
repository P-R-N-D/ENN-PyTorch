from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
from torch import nn, optim

from ..data.stats import MetaData
from .kernels import DotProductAttention
from ..utils.platform import (
    get_device,
    is_cpu_bf16_supported,
    is_cuda_bf16_supported,
    is_float8_supported,
    is_int4_supported,
    is_int8_supported,
    optimal_optimizer_params,
)
from ..backend.engine import AutoCast, Accelerator, _supports_scale

__all__ = ["AdamW"]

class AdamW:
    @staticmethod
    def float(
        model_or_params: Union[
            nn.Module, Iterable[nn.Parameter], Sequence[Dict[str, Any]]
        ],
        lr: float,
        *args: Any,
        weight_decay: float = 0.0,
        metadata: Optional[MetaData[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> optim.Optimizer:
        params = (
            model_or_params.parameters()
            if hasattr(model_or_params, "parameters")
            else model_or_params
        )
        ref_tensor: Optional[torch.Tensor] = None
        if isinstance(model_or_params, nn.Module):
            ref_tensor = Accelerator._module_reference_tensor(model_or_params)
        dev: torch.device
        if metadata is not None:
            dev = torch.device(metadata.device)
        elif ref_tensor is not None:
            dev = ref_tensor.device
        else:
            dev = get_device()
        meta = AutoCast._ensure_metadata(dev, metadata=metadata)
        dev = torch.device(meta.device)
        if hasattr(dev, "type") and dev.type == "cuda":
            try:
                from transformer_engine.pytorch.optimizers import (
                    FusedAdam as TEFusedAdam,
                )

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
                float8_dtypes = AutoCast._float8_dtypes()
                if not any(
                    _supports_scale(dtype, meta, safety_margin=2.0)
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
                        from transformer_engine.pytorch.optimizers import FusedAdam

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
        metadata: Optional[MetaData[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> optim.Optimizer:
        params = (
            model_or_params.parameters()
            if hasattr(model_or_params, "parameters")
            else model_or_params
        )
        ref_tensor: Optional[torch.Tensor] = None
        if isinstance(model_or_params, nn.Module):
            ref_tensor = Accelerator._module_reference_tensor(model_or_params)
        if metadata is not None:
            dev = torch.device(metadata.device)
        elif ref_tensor is not None:
            dev = ref_tensor.device
        else:
            dev = get_device()
        meta = AutoCast._ensure_metadata(dev, metadata=metadata)
        dev = torch.device(meta.device)
        if hasattr(dev, "type") and dev.type == "cuda":
            try:
                from transformer_engine.pytorch.optimizers import (
                    FusedAdam as TEFusedAdam,
                )

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
                candidates: List[Tuple[str, Callable[[Optional[torch.device]], Tuple[bool, str]]]] = []
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
                    from torchao.prototype.low_bit_optim import (
                        AdamW4bit,
                        AdamW8bit,
                    )
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
