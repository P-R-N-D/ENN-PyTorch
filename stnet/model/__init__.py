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
from .activations import GeGLU, SwiGLU
from ..backend.losses import (
    DataFidelityLoss,
    LinearCombinationLoss,
    MultipleQuantileLoss,
    StandardNormalLoss,
    StudentsTLoss,
    TiledLoss,
)
from .layers import (
    CrossAttention,
    CrossTransformer,
    DilatedAttention,
    GlobalEncoder,
    LocalProcessor,
    LongNet,
    PatchAttention,
    PatchEmbedding,
    Payload,
    PositionalEncoding,
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
    "LinearCombinationLoss",
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
