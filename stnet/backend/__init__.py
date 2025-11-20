# -*- coding: utf-8 -*-
from __future__ import annotations

from importlib import import_module

from . import compat, distributed, export, profiler, runtime, system

__all__ = [
    "compat",
    "distributed",
    "export",
    "profiler",
    "runtime",
    "system",
    "new_model",
    "load_model",
    "save_model",
    "train",
    "learn",
    "predict",
    "infer",
]

_EXPORTS = {
    "new_model": ("stnet.api.io", "new_model"),
    "load_model": ("stnet.api.io", "load_model"),
    "save_model": ("stnet.api.io", "save_model"),
    "train": ("stnet.api.run", "train"),
    "learn": ("stnet.api.run", "train"),
    "predict": ("stnet.api.run", "predict"),
    "infer": ("stnet.api.run", "predict"),
}


def __getattr__(name: str):  # pragma: no cover - thin import wrapper
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
