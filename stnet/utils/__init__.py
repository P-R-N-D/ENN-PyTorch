# -*- coding: utf-8 -*-
from __future__ import annotations

import sys

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
]

sys.modules[__name__ + ".dtypes"] = datatype

patch_torch()
