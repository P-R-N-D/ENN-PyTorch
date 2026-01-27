# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
from types import ModuleType

__all__ = [
    "distributed",
    "io",
    "main",
    "losses",
    "optimizers",
    "workflow",
]


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
