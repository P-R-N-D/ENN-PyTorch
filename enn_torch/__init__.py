# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tensordict import TensorDictBase

    from .nn.architecture import Model

__all__ = [
    "config",
    "core",
    "data",
    "nn",
    "runtime",
    "load_model",
    "new_model",
    "predict",
    "save_model",
    "train",
]


def __getattr__(name: str) -> ModuleType:
    if name == "config":
        module = importlib.import_module("enn_torch.core.config")
        sys.modules.setdefault(f"{__name__}.config", module)
        globals()["config"] = module
        return module
    if name in {"core", "data", "nn", "runtime"}:
        return importlib.import_module(f"enn_torch.{name}")
    raise AttributeError(f"module 'enn_torch' has no attribute {name!r}")


def new_model(*args: Any, **kwargs: Any) -> Model:
    from .runtime import workflow

    return workflow.new_model(*args, **kwargs)


def load_model(*args: Any, **kwargs: Any) -> Model:
    from .runtime import workflow

    return workflow.load_model(*args, **kwargs)


def save_model(*args: Any, **kwargs: Any) -> str:
    from .runtime import workflow

    return workflow.save_model(*args, **kwargs)


def train(*args: Any, **kwargs: Any) -> Model:
    from .runtime import workflow

    return workflow.train(*args, **kwargs)


def predict(*args: Any, **kwargs: Any) -> TensorDictBase | dict[str, TensorDictBase]:
    from .runtime import workflow

    return workflow.predict(*args, **kwargs)
