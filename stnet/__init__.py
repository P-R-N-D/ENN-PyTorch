# -*- coding: utf-8 -*-
from __future__ import annotations
import importlib
__all__ = ['core', 'data', 'nn', 'runtime', 'api', 'new_model', 'load_model', 'save_model', 'train', 'predict', 'get_prediction']

def __getattr__(name):
    if name in {'core', 'data', 'nn', 'runtime', 'api'}:
        return importlib.import_module(f'stnet.{name}')
    raise AttributeError(f"module 'stnet' has no attribute '{name}'")

def new_model(*args, **kwargs):
    from .api import new_model as _new_model
    return _new_model(*args, **kwargs)

def load_model(*args, **kwargs):
    from .api import load_model as _load_model
    return _load_model(*args, **kwargs)

def save_model(*args, **kwargs):
    from .api import save_model as _save_model
    return _save_model(*args, **kwargs)

def train(*args, **kwargs):
    from .api import train as _train
    return _train(*args, **kwargs)

def predict(*args, **kwargs):
    from .api import predict as _predict
    return _predict(*args, **kwargs)

def get_prediction(*args, **kwargs):
    from .api import get_prediction as _get_prediction
    return _get_prediction(*args, **kwargs)
