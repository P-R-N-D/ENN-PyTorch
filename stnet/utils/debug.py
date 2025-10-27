# -*- coding: utf-8 -*-
"""Debug utilities shared across runtime and model modules."""
from __future__ import annotations

from typing import Any

import torch

__all__ = ["is_fake_tensor"]

try:  # pragma: no cover - optional dependency
    from torchdistx.fake import is_fake as _tdx_is_fake  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - torchdistx not installed
    _tdx_is_fake = None  # type: ignore

try:  # pragma: no cover - private API best-effort
    from torch._subclasses.fake_tensor import FakeTensor  # type: ignore
except Exception:  # pragma: no cover - fallback when private API unavailable
    FakeTensor = tuple()  # type: ignore


def is_fake_tensor(value: Any) -> bool:
    """Return ``True`` when ``value`` references a FakeTensor placeholder."""
    if not isinstance(value, torch.Tensor):
        return False
    if _tdx_is_fake is not None:
        try:
            return bool(_tdx_is_fake(value))
        except Exception:
            # torchdistx is optional; fall back to local heuristics when it errors.
            pass
    return isinstance(value, FakeTensor) or getattr(value, "fake_mode", None) is not None
