# -*- coding: utf-8 -*-
from __future__ import annotations

from .losses import (
    DataFidelityLoss,
    LinearCombinationLoss,
    LossWeightController,
    MultipleQuantileLoss,
    StandardNormalLoss,
    StudentsTLoss,
    TiledLoss,
    TensorDictLoss,
    expand_mask_like_prediction,
)
from .optimizers import AdamW, TensorDictOptimizer
from .fx import (
    AutoCast,
    Gradient,
    Fusion,
    is_transformer_engine_enabled,
    reshape_for_heads,
)

__all__ = [
    "expand_mask_like_prediction",
    "MultipleQuantileLoss",
    "StandardNormalLoss",
    "StudentsTLoss",
    "DataFidelityLoss",
    "LinearCombinationLoss",
    "TiledLoss",
    "LossWeightController",
    "TensorDictLoss",
    "AdamW",
    "TensorDictOptimizer",
    "AutoCast",
    "Fusion",
    "Gradient",
    "is_transformer_engine_enabled",
    "reshape_for_heads",
]
