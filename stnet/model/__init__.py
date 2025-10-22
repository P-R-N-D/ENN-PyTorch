# -*- coding: utf-8 -*-
from __future__ import annotations

from ..config import (  # noqa: E402  # requires torch patches before import
    BuildConfig,
    ModelConfig,
    PatchConfig,
    coerce_model_config,
    coerce_patch_config,
    model_config,
    patch_config,
)
from ..utils.compat import SDPBackend, patch_torch, sdpa_kernel

patch_torch()
from .functional import (  # noqa: E402  # requires torch patches before import
    DataFidelityLoss,
    GeGLU,
    MultipleQuantileLoss,
    PositionalEncoding,
    StandardNormalLoss,
    StudentsTLoss,
    SwiGLU,
    TiledLoss,
)  # noqa: E402  # requires torch patches before import
from .layers import (  # noqa: E402  # requires torch patches before import
    GlobalEncoderLayer,
    CrossAttention,
    PatchAttention,
    PatchEmbedding,
    PointTransformer,
    StochasticDepth,
    TemporalEncoderLayer,
    norm_layer,
    schedule_stochastic_depth,
)  # noqa: E402  # requires torch patches before import
from .modules import (  # noqa: E402  # requires torch patches before import
    CrossTransformer,
    GlobalEncoder,
    GlobalEncoderBlock,
    LocalProcessor,
    Payload,
    Root,
    SpatialEncoder,
    TemporalEncoder,
    TemporalEncoderBlock,
)  # noqa: E402  # requires torch patches before import

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
