# -*- coding: utf-8 -*-
from __future__ import annotations

from ..backend.compat import SDPBackend, patch_torch, sdpa_kernel

patch_torch()

from ..api.config import (
    ModelConfig,
    PatchConfig,
    coerce_model_config,
    coerce_patch_config,
    model_config,
    patch_config,
)
from ..functional import (
    DataFidelityLoss,
    GeGLU,
    MultipleQuantileLoss,
    PositionalEncoding,
    StandardNormalLoss,
    StudentsTLoss,
    SwiGLU,
    TiledLoss,
)
from .nn import (
    CrossAttention,
    CrossTransformer,
    DilatedAttention,
    GlobalEncoder,
    LocalProcessor,
    LongNet,
    PatchAttention,
    PatchEmbedding,
    Payload,
    PointTransformer,
    Root,
    SpatialEncoder,
    StochasticDepth,
    TemporalEncoder,
    TemporalEncoderBlock,
    TemporalEncoderLayer,
    norm_layer,
    schedule_stochastic_depth,
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
    "DilatedAttention",
    "LongNet",
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
