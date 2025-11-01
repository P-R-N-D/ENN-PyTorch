from __future__ import annotations

from ..api.config import (
    ModelConfig,
    PatchConfig,
    RuntimeConfig,
    coerce_runtime_config,
    runtime_config,
)
from ..api.io import load_model, new_model, save_model
from .distributed import joining
from . import launch as launch
from .launch import predict, train

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
]
