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
    expand_to_pred,
)
from .optimizers import AdamW, StochasticWeightAverage, stochastic_weight_average
from .fx import (
    Autocast,
    Gradient,
    Fusion,
    Quantization,
    is_scale_safe,
    is_nvidia_te_available,
    reshape_for_mha,
)

__all__ = [
    "expand_to_pred",
    "MultipleQuantileLoss",
    "StandardNormalLoss",
    "StudentsTLoss",
    "DataFidelityLoss",
    "LinearCombinationLoss",
    "TiledLoss",
    "LossWeightController",
    "AdamW",
    "StochasticWeightAverage",
    "stochastic_weight_average",
    "Autocast",
    "Fusion",
    "Quantization",
    "Gradient",
    "is_scale_safe",
    "is_nvidia_te_available",
    "reshape_for_mha",
]
