# -*- coding: utf-8 -*-
from __future__ import annotations

from ..utils.compat import SDPBackend, patch_torch, sdpa_kernel

patch_torch()

from ..config import (
    ModelConfig,
    PatchConfig,
    coerce_model_config,
    coerce_patch_config,
    model_config,
    patch_config,
)
from .functional import (
    DataFidelityLoss,
    GeGLU,
    MultipleQuantileLoss,
    PositionalEncoding,
    StandardNormalLoss,
    StudentsTLoss,
    SwiGLU,
    TiledLoss,
)
from .layers import (
    GlobalEncoderLayer,
    CrossAttention,
    PatchAttention,
    PatchEmbedding,
    StochasticDepth,
    TemporalEncoderLayer,
    norm_layer,
    schedule_stochastic_depth,
)
from .modules import (
    CrossTransformer,
    GlobalEncoder,
    GlobalEncoderBlock,
    LocalProcessor,
    Payload,
    Root,
    SpatialEncoder,
    TemporalEncoder,
    TemporalEncoderBlock,
    PointTransformer,
)

__all__ = [
    "sdpa_kernel",
    "SDPBackend",
    "Root",
    "ModelConfig",
    "PatchConfig",
    "model_config",
    "patch_config",
    "coerce_model_config",
    "coerce_patch_config",
    "SpatialEncoder",
    "TemporalEncoder",
    "LocalProcessor",
    "PatchEmbedding",
    "PatchAttention",
    "PointTransformer",
    "TemporalEncoderLayer",
    "TemporalEncoderBlock",
    "GlobalEncoderLayer",
    "GlobalEncoderBlock",
    "CrossTransformer",
    "Payload",
    "GlobalEncoder",
    "CrossAttention",
    "GeGLU",
    "PositionalEncoding",
    "SwiGLU",
    "MultipleQuantileLoss",
    "StandardNormalLoss",
    "StudentsTLoss",
    "DataFidelityLoss",
    "TiledLoss",
    "StochasticDepth",
    "norm_layer",
    "schedule_stochastic_depth",
]
ZLoss = StandardNormalLoss
TLoss = StudentsTLoss
