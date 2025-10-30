# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any

import sys
import torch

from .platform import Distributed, System
from .platform import (
    get_device,
    get_runtime_config,
    initialize_sdpa_backends,
)
from .compat import (
    SDPBackend,
    TorchCompat,
    _to_sdpa_backends,
    patch_torch,
    sdpa_kernel,
)
from .optimization import (
    AdamW,
    AutoCast,
    DotProductAttention,
    LossWeightController,
    MultiScaleRetention,
    MultiScaleRetentionCompat,
    Module,
)
from .profiler import FlopCounter, attention_flops_bshd
from . import datatype

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


def is_meta_tensor(value: Any) -> bool:
    """Check whether a tensor is backed by the meta device placeholder."""

    return isinstance(value, torch.Tensor) and getattr(value, "is_meta", False)


def is_meta_or_fake_tensor(value: Any) -> bool:
    """Return ``True`` when ``value`` is either a meta tensor or a fake tensor."""

    return is_meta_tensor(value) or is_fake_tensor(value)

dtypes = datatype
from ..data.transforms import (
    IncrementalPCA,
    StandardScaler,
    VarianceThreshold,
    postprocess,
    preprocess,
)

__all__ = [
    "AutoCast",
    "DotProductAttention",
    "MultiScaleRetention",
    "MultiScaleRetentionCompat",
    "AdamW",
    "Module",
    "LossWeightController",
    "attention_flops_bshd",
    "FlopCounter",
    "Distributed",
    "System",
    "get_device",
    "get_runtime_config",
    "initialize_sdpa_backends",
    "patch_torch",
    "TorchCompat",
    "SDPBackend",
    "sdpa_kernel",
    "_to_sdpa_backends",
    "VarianceThreshold",
    "StandardScaler",
    "IncrementalPCA",
    "preprocess",
    "postprocess",
    "datatype",
    "dtypes",
    "is_fake_tensor",
    "is_meta_tensor",
    "is_meta_or_fake_tensor",
]

sys.modules[__name__ + ".dtypes"] = datatype

patch_torch()
