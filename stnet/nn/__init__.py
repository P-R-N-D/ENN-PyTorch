# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TypeAlias

from ..utils.compat import SDPBackend, patch_torch, sdpa_kernel

patch_torch()
from .config import (  # noqa: E402  # requires torch patches before import
    ModelConfig,
    PatchConfig,
    model_config,
    patch_config,
    coerce_model_config,
    coerce_patch_config,
)  # noqa: E402  # requires torch patches before import
from .functional import (  # noqa: E402  # requires torch patches before import
    DataFidelityLoss,
    GeGLU,
    MultipleQuantileLoss,
    StandardNormalLoss,
    StudentsTLoss,
    SwiGLU,
    TiledLoss,
)  # noqa: E402  # requires torch patches before import
from .module import (  # noqa: E402  # requires torch patches before import
    CrossTransformer,
    GlobalEncoder,
    GlobalEncoderLayer,
    LocalProcessor,
    Payload,
    SpatialEncoder,
    SpatialEncoderLayer,
    StochasticDepth,
    TemporalEncoder,
    TemporalEncoderLayer,
    norm_layer,
    schedule_stochastic_depth,
)  # noqa: E402  # requires torch patches before import
from .container import Model  # noqa: E402  # requires torch patches before import

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
    "TiledLoss",
    "StochasticDepth",
    "norm_layer",
    "schedule_stochastic_depth",
]
ZLoss: TypeAlias = StandardNormalLoss
TLoss: TypeAlias = StudentsTLoss
BuildConfig: TypeAlias = ModelConfig
