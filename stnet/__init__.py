# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
from typing import Any, Optional

from tensordict import TensorDictBase

from .nn.architecture import Model


__all__ = [
            'core', 'data', 'nn', 'runtime', 'api', 'config',
            'new_model', 'load_model', 'save_model', 'train', 'predict', 'get_prediction',
        ]


def __getattr__(name: str) -> Optional[Any]:
    if name in {'core', 'data', 'nn', 'runtime', 'api', 'config'}:
        return importlib.import_module(f'stnet.{name}')
    raise AttributeError(f"module 'stnet' has no attribute '{name}'")


def new_model(*args: Any, **kwargs: Any) -> Model:
    from .api import new_model as _new_model
    return _new_model(*args: Any, **kwargs: Any)


def load_model(*args: Any, **kwargs: Any) -> Model:
    from .api import load_model as _load_model
    return _load_model(*args: Any, **kwargs: Any)


def save_model(*args: Any, **kwargs: Any) -> None:
    from .api import save_model as _save_model
    return _save_model(*args: Any, **kwargs: Any)


def train(*args: Any, **kwargs: Any) -> Model:
    from .api import train as _train
    return _train(*args: Any, **kwargs: Any)


def predict(*args: Any, **kwargs: Any) -> TensorDictBase:
    from .api import predict as _predict
    return _predict(*args: Any, **kwargs: Any)


def get_prediction(*args: Any, **kwargs: Any) -> TensorDictBase:
    from .api import get_prediction as _get_prediction
    return _get_prediction(*args: Any, **kwargs: Any)
