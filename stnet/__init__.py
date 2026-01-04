# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tensordict import TensorDictBase

    from .nn.architecture import Model


__all__ = [
    "api",
    "config",
    "core",
    "data",
    "nn",
    "runtime",
    "get_prediction",
    "load_model",
    "new_model",
    "predict",
    "save_model",
    "train",
]


def __getattr__(name: str) -> ModuleType:
    if name in {"core", "data", "nn", "runtime", "api", "config"}:
        return importlib.import_module(f"stnet.{name}")
    raise AttributeError(f"module 'stnet' has no attribute {name!r}")


def new_model(*args: Any, **kwargs: Any) -> Model:
    from . import api

    return api.new_model(*args, **kwargs)


def load_model(*args: Any, **kwargs: Any) -> Model:
    from . import api

    return api.load_model(*args, **kwargs)


def save_model(*args: Any, **kwargs: Any) -> str:
    from . import api

    return api.save_model(*args, **kwargs)


def train(*args: Any, **kwargs: Any) -> Model:
    from . import api

    return api.train(*args, **kwargs)


def predict(*args: Any, **kwargs: Any) -> TensorDictBase | dict[str, TensorDictBase]:
    from . import api

    return api.predict(*args, **kwargs)


def get_prediction(*args: Any, **kwargs: Any) -> TensorDictBase:
    from . import api

    return api.get_prediction(*args, **kwargs)
