# -*- coding: utf-8 -*-
from __future__ import annotations

from tensordict import set_list_to_stack

set_list_to_stack(True).set()

from importlib import import_module

from . import api, backend, data, functional, model

__all__ = [
    "api",
    "backend",
    "data",
    "functional",
    "model",
    "PatchConfig",
    "build_config",
    "new_model",
    "load_model",
    "save_model",
    "learn",
    "infer",
]

_EXPORTS = {
    "PatchConfig": ("stnet.api.config", "PatchConfig"),
    "build_config": ("stnet.api.config", "build_config"),
    "new_model": ("stnet.api.io", "new_model"),
    "load_model": ("stnet.api.io", "load_model"),
    "save_model": ("stnet.api.io", "save_model"),
    "learn": ("stnet.api.run", "train"),
    "infer": ("stnet.api.run", "predict"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
