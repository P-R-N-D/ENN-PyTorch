# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TypeAlias

from ..toolkit.compat import SDPBackend, patch_torch, sdpa_kernel

_TORCH_COMPAT = patch_torch()
from .config import (
    ModelConfig,
    PatchConfig,
    model_config,
    patch_config,
    coerce_model_config,
    coerce_patch_config,
)
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
from .network import Model

__all__ = [
    "sdpa_kernel",
    "SDPBackend",
    "Model",
    "ModelConfig",
    "PatchConfig",
    "BuildConfig",
    "model_config",
    "patch_config",
    "coerce_model_config",
    "coerce_patch_config",
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
BuildConfig: TypeAlias = ModelConfig
