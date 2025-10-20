# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any

from ..architecture.config import ModelConfig, PatchConfig
from .config import OpsConfig, coerce_ops_config, ops_config


def new_model(*args: Any, **kwargs: Any) -> Any:
    from .management import new_model as _impl

    return _impl(*args, **kwargs)


def load_model(*args: Any, **kwargs: Any) -> Any:
    from .management import load_model as _impl

    return _impl(*args, **kwargs)


def save_model(*args: Any, **kwargs: Any) -> Any:
    from .management import save_model as _impl

    return _impl(*args, **kwargs)


def joining(*args: Any, **kwargs: Any) -> Any:
    from ..toolkit.optimization import joining as _impl

    return _impl(*args, **kwargs)

def train(*args: Any, **kwargs: Any) -> Any:
    from .operation import train as _impl

    return _impl(*args, **kwargs)


def predict(*args: Any, **kwargs: Any) -> Any:
    from .operation import predict as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "ModelConfig",
    "PatchConfig",
    "OpsConfig",
    "ops_config",
    "coerce_ops_config",
    "new_model",
    "load_model",
    "save_model",
    "joining",
    "train",
    "predict",
]
