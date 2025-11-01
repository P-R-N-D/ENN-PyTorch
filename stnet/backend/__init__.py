from __future__ import annotations

from ..api.config import (
    ModelConfig,
    PatchConfig,
    RuntimeConfig,
    coerce_runtime_config,
    runtime_config,
)
from ..api.io import load_model, new_model, save_model
from ..api.run import launch, predict, train
from .distributed import joining
from .losses import (
    DataFidelityLoss,
    LinearCombinationLoss,
    MultipleQuantileLoss,
    StandardNormalLoss,
    StudentsTLoss,
    TiledLoss,
)
from .optimizers import AdamW

learn = train
infer = predict

__all__ = [
    "ModelConfig",
    "PatchConfig",
    "RuntimeConfig",
    "runtime_config",
    "coerce_runtime_config",
    "new_model",
    "load_model",
    "save_model",
    "joining",
    "train",
    "learn",
    "predict",
    "infer",
    "launch",
    "AdamW",
    "MultipleQuantileLoss",
    "StandardNormalLoss",
    "StudentsTLoss",
    "DataFidelityLoss",
    "LinearCombinationLoss",
    "TiledLoss",
]
