# -*- coding: utf-8 -*-
from __future__ import annotations

from types import ModuleType

from . import compat, distributed, system

__all__ = [
    "compat",
    "distributed",
    "system",
    "export",
    "runtime",
]


def __getattr__(name: str) -> "ModuleType":
    if name in {"export", "runtime"}:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
