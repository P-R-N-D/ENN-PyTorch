# -*- coding: utf-8 -*-
"""Compatibility module for legacy ``import enn_torch.config`` callers."""
from __future__ import annotations

from .core.config import *  # noqa: F403
from .core.config import __all__  # type: ignore  # noqa: F401

