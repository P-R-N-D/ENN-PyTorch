# -*- coding: utf-8 -*-
from __future__ import annotations

from importlib import import_module

from . import compat, distributed, profiler, system

__all__ = [
    "compat",
    "distributed",
    "profiler",
    "system",
    "export",
    "runtime",
]


def __getattr__(name: str):
    if name in {"export", "runtime"}:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
