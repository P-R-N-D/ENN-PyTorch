from __future__ import annotations

from typing import TypeAlias

import torch
from torch import nn

from ..toolkit.compat import SDPBackend, sdpa_kernel, secure_torch

secure_torch()


class StochasticDepth(nn.Module):
    def __init__(self, p: float = 0.0, mode: str = "row") -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = float(p)
        self.mode = str(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        if keep_prob <= 0:
            return torch.zeros_like(x)
        if self.mode == "row":
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        else:
            shape = (1,) * x.ndim
        noise = x.new_empty(shape).bernoulli_(keep_prob).div_(keep_prob)
        return x * noise


def _norm(norm_type: str, d_model: int) -> nn.Module:
    kind = str(norm_type).lower()
    if kind in {"layernorm", "layer_norm", "ln"}:
        return nn.LayerNorm(d_model)
    if kind in {"rmsnorm", "rms_norm", "rms"} and hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(d_model)
    if kind in {"batchnorm", "batchnorm1d", "bn", "bn1d"}:
        return nn.BatchNorm1d(d_model)
    return nn.LayerNorm(d_model)


def _stochastic_depth_scheduler(max_rate: float, depth: int) -> list[float]:
    if depth <= 0:
        return []
    if max_rate <= 0:
        return [0.0 for _ in range(depth)]
    step = float(max_rate) / max(1, depth)
    return [step * (index + 1) for index in range(depth)]


from .module import (
    CrossTransformer,
    DataFidelityLoss,
    GeGLU,
    Meta,
    MetaNet,
    MultipleQuantileLoss,
    PatchAttention,
    SpatialSubnet,
    SpatioTemporalNet,
    StandardNormalLoss,
    StudentsTLoss,
    SwiGLU,
    TemporalSubnet,
)
from .network import Config, Model, PatchParameters

__all__ = [
    "sdpa_kernel",
    "SDPBackend",
    "Model",
    "Config",
    "PatchParameters",
    "SpatialSubnet",
    "TemporalSubnet",
    "SpatioTemporalNet",
    "PatchAttention",
    "CrossTransformer",
    "Meta",
    "MetaNet",
    "GeGLU",
    "SwiGLU",
    "MultipleQuantileLoss",
    "StandardNormalLoss",
    "StudentsTLoss",
    "DataFidelityLoss",
    "StochasticDepth",
    "_norm",
    "_stochastic_depth_scheduler",
]

ZLoss: TypeAlias = StandardNormalLoss
TLoss: TypeAlias = StudentsTLoss
