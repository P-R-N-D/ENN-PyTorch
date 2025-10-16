# -*- coding: utf-8 -*-
from __future__ import annotations

from .compat import SDPBackend, _to_sdpa_backends, patch_torch, sdpa_kernel
from .optimization import (
    AdamW,
    Autocast,
    FlopCounter,
    GatedMultiScaleRetention,
    ScaledDotProductAttention,
    attention_flops_bshd,
)
from .preprocessing import IncrementalPCA, StandardScaler, VarianceThreshold
from .capability import get_device, get_runtime_config, resolve_sdpa_backends

__all__ = [
    "Autocast",
    "ScaledDotProductAttention",
    "GatedMultiScaleRetention",
    "AdamW",
    "attention_flops_bshd",
    "FlopCounter",
    "get_device",
    "get_runtime_config",
    "resolve_sdpa_backends",
    "patch_torch",
    "SDPBackend",
    "sdpa_kernel",
    "_to_sdpa_backends",
    "VarianceThreshold",
    "StandardScaler",
    "IncrementalPCA",
]

patch_torch()
