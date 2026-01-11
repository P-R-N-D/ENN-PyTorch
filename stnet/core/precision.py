# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings

from . import amp as _amp


warnings.warn(
    "stnet.core.precision is deprecated; use stnet.core.amp instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [name for name in dir(_amp) if not name.startswith("_")]

globals().update({name: getattr(_amp, name) for name in __all__})


def __getattr__(name: str):
    return getattr(_amp, name)


def __dir__() -> list[str]:
    return sorted(__all__)
