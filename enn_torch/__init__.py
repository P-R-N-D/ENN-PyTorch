# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from tensordict import TensorDictBase

    from .core.config import ModelConfig
    from .core.config import RuntimeConfig
    from .nn.wrappers import Model

__all__ = [
    "core",
    "data",
    "nn",
    "runtime",
    "load_model",
    "new_model",
    "save_model",
    "train",
    "predict",
    "ModelConfig",
    "RuntimeConfig",
]


def __getattr__(name: str) -> ModuleType:
    if name in {"core", "data", "nn", "runtime"}:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module '{__name__}' has no attribute {name!r}")


def new_model(*args: Any, **kwargs: Any) -> Model:
    from .runtime import workflows

    return workflows.new_model(*args, **kwargs)


def load_model(*args: Any, **kwargs: Any) -> Model:
    from .runtime import workflows

    return workflows.load_model(*args, **kwargs)


def save_model(*args: Any, **kwargs: Any) -> str:
    from .runtime import workflows

    return workflows.save_model(*args, **kwargs)


def train(*args: Any, **kwargs: Any) -> Model:
    from .runtime import workflows

    return workflows.train(*args, **kwargs)


def predict(
    *args: Any, **kwargs: Any
) -> TensorDictBase | dict[str, TensorDictBase]:
    from .runtime import workflows

    return workflows.predict(*args, **kwargs)
