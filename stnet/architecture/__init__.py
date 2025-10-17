from __future__ import annotations

from typing import TypeAlias

from ..toolkit.compat import SDPBackend, patch_torch, sdpa_kernel

patch_torch()
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
    StochasticDepth,
    StudentsTLoss,
    SwiGLU,
    TemporalSubnet,
    _norm,
    _stochastic_depth_scheduler,
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
