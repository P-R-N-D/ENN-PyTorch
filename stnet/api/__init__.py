"""Public orchestration API for STNet."""

from __future__ import annotations

import sys as _sys

from .run import launch, predict, train
from ..backend.compat import (
    SDPBackend,
    TorchCompat,
    _to_sdpa_backends,
    is_fake_tensor,
    is_meta_or_fake_tensor,
    is_meta_tensor,
    patch_torch,
    sdpa_kernel,
)
from ..backend.distributed import joining, no_synchronization
from ..backend.environment import (
    Distributed,
    Network,
    System,
    get_device,
    get_runtime_config,
    initialize_sdpa_backends,
)
from ..backend.profiler import FlopCounter, attention_flops_bshd
from ..data import datatype as datatype_module
from ..data.transforms import (
    IncrementalPCA,
    StandardScaler,
    VarianceThreshold,
    postprocess,
    preprocess,
)
from ..functional import AdamW, AutoCast, Gradient, LayerReplacement, LossWeightController
from ..model.kernels import DotProductAttention, MultiScaleRetention, MultiScaleRetentionCompat

datatype = datatype_module
dtypes = datatype_module

__all__ = [
    "train",
    "predict",
    "launch",
    "AutoCast",
    "DotProductAttention",
    "MultiScaleRetention",
    "MultiScaleRetentionCompat",
    "AdamW",
    "LayerReplacement",
    "Gradient",
    "LossWeightController",
    "attention_flops_bshd",
    "FlopCounter",
    "joining",
    "no_synchronization",
    "Distributed",
    "Network",
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

_sys.modules[__name__ + ".dtypes"] = datatype_module
