# -*- coding: utf-8 -*-
from __future__ import annotations

import sys as _sys
from typing import Any

from ..config import (
    ModelConfig,
    PatchConfig,
    RuntimeConfig,
    coerce_runtime_config,
    runtime_config,
)


def new_model(*args: Any, **kwargs: Any) -> Any:
    from ..utils.io import new_model as _impl

    return _impl(*args, **kwargs)


def load_model(*args: Any, **kwargs: Any) -> Any:
    from ..utils.io import load_model as _impl

    return _impl(*args, **kwargs)


def save_model(*args: Any, **kwargs: Any) -> Any:
    from ..utils.io import save_model as _impl

    return _impl(*args, **kwargs)


def joining(*args: Any, **kwargs: Any) -> Any:
    from ..utils.optimization import joining as _impl

    return _impl(*args, **kwargs)

def train(*args: Any, **kwargs: Any) -> Any:
    from .launch import train as _impl

    return _impl(*args, **kwargs)


def predict(*args: Any, **kwargs: Any) -> Any:
    from .launch import predict as _impl

    return _impl(*args, **kwargs)


def learn(*args: Any, **kwargs: Any) -> Any:
    return train(*args, **kwargs)


def infer(*args: Any, **kwargs: Any) -> Any:
    return predict(*args, **kwargs)


from . import launch as launch

operation = launch
_sys.modules[f"{__name__}.operation"] = operation

api = launch

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
    "operation",
    "api",
]
