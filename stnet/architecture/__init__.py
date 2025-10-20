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
    GlobalEncoder,
    GlobalEncoderLayer,
    LocalProcessor,
    MultipleQuantileLoss,
    Payload,
    SpatialEncoder,
    SpatialEncoderLayer,
    StandardNormalLoss,
    StochasticDepth,
    StudentsTLoss,
    SwiGLU,
    TemporalEncoder,
    TemporalEncoderLayer,
    norm_layer,
    schedule_stochastic_depth,
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
    "SpatialEncoder",
    "TemporalEncoder",
    "LocalProcessor",
    "SpatialEncoderLayer",
    "TemporalEncoderLayer",
    "GlobalEncoderLayer",
    "CrossTransformer",
    "Payload",
    "GlobalEncoder",
    "GeGLU",
    "SwiGLU",
    "MultipleQuantileLoss",
    "StandardNormalLoss",
    "StudentsTLoss",
    "DataFidelityLoss",
    "StochasticDepth",
    "norm_layer",
    "schedule_stochastic_depth",
]
ZLoss: TypeAlias = StandardNormalLoss
TLoss: TypeAlias = StudentsTLoss
BuildConfig: TypeAlias = ModelConfig
