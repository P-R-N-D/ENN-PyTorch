# -*- coding: utf-8 -*-
from __future__ import annotations

from .capability import (
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
    TunedAMP,
    FlopCounter,
    TunedMSR,
    LossWeightOptimizer,
    ModuleTuner,
    TunedDPA,
    TunedAdamW,
    attention_flops_bshd,
)
from ..architecture.module import (
    IncrementalPCA,
    StandardScaler,
    VarianceThreshold,
)

__all__ = [
    "TunedAMP",
    "TunedDPA",
    "TunedMSR",
    "TunedAdamW",
    "ModuleTuner",
    "LossWeightOptimizer",
    "attention_flops_bshd",
    "FlopCounter",
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
]
patch_torch()
