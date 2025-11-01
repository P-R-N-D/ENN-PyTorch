from __future__ import annotations

import sys

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

# Preserve the legacy ``stnet.backend.launch`` module path for downstream users.
sys.modules.setdefault("stnet.backend.launch", sys.modules["stnet.api.run"])

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
