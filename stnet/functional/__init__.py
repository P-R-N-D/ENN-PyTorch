from __future__ import annotations

from ..backend.compat import patch_torch

patch_torch()

from .fx import GeGLU, PositionalEncoding, SwiGLU, reshape_for_heads
from .loss import (
    DataFidelityLoss,
    LinearCombinationLoss,
    MultipleQuantileLoss,
    StandardNormalLoss,
    StudentsTLoss,
    TiledLoss,
    expand_mask_like_prediction,
)
from .optim import AdamW, Module

__all__ = [
    "reshape_for_heads",
    "PositionalEncoding",
    "GeGLU",
    "SwiGLU",
    "expand_mask_like_prediction",
    "MultipleQuantileLoss",
    "StandardNormalLoss",
    "StudentsTLoss",
    "DataFidelityLoss",
    "LinearCombinationLoss",
    "TiledLoss",
    "AdamW",
    "Module",
]
