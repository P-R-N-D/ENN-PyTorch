# -*- coding: utf-8 -*-
from __future__ import annotations

from ..api.config import (
    ModelConfig,
    PatchConfig,
    coerce_model_config,
    coerce_patch_config,
    model_config,
    patch_config,
)
from ..backend.compat import SDPBackend, patch_torch, sdpa_kernel
from ..functional.losses import (
    DataFidelityLoss,
    LinearCombinationLoss,
    MultipleQuantileLoss,
    StandardNormalLoss,
    StudentsTLoss,
    TiledLoss,
)
from .activations import GeGLU, SwiGLU
from .kernels import (
    DotProductAttention,
    MultiHeadAttention,
    MultiScaleRetention,
    MultiScaleRetentionCompat,
    to_additive_mask,
)
from .layers import (
    CompatLayer,
    CrossAttention,
    CrossTransformer,
    DilatedAttention,
    GlobalEncoder,
    LocalProcessor,
    LongNet,
    LossWeightPolicy,
    PatchAttention,
    PatchEmbedding,
    Payload,
    PositionalEncoding,
    PointTransformer,
    Root,
    SpatialEncoder,
    StochasticDepth,
    TemporalEncoder,
    RetNet,
    Retention,
    norm_layer,
    stochastic_depth_schedule,
)

patch_torch()

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
    "CompatLayer",
    "PointTransformer",
    "Retention",
    "RetNet",
    "DilatedAttention",
    "LongNet",
    "LossWeightPolicy",
    "CrossTransformer",
    "Payload",
    "GlobalEncoder",
    "CrossAttention",
    "MultiHeadAttention",
    "DotProductAttention",
    "MultiScaleRetention",
    "MultiScaleRetentionCompat",
    "to_additive_mask",
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
    "stochastic_depth_schedule",
]

ZLoss = StandardNormalLoss
TLoss = StudentsTLoss
