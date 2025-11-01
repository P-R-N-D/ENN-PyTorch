"""High-level orchestration entrypoints for STNet."""
from __future__ import annotations

from ..backend import launch as launch
from ..backend.launch import predict, train

__all__ = ["train", "predict", "launch"]
