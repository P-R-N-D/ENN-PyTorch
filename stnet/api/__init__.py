"""Public orchestration API for STNet."""

from __future__ import annotations

import sys as _sys

from . import runtime as _runtime
from .run import launch, predict, train

AdamW = _runtime.AdamW
AutoCast = _runtime.AutoCast
Distributed = _runtime.Distributed
DotProductAttention = _runtime.DotProductAttention
FlopCounter = _runtime.FlopCounter
IncrementalPCA = _runtime.IncrementalPCA
LossWeightController = _runtime.LossWeightController
Module = _runtime.Module
MultiScaleRetention = _runtime.MultiScaleRetention
MultiScaleRetentionCompat = _runtime.MultiScaleRetentionCompat
Network = _runtime.Network
SDPBackend = _runtime.SDPBackend
StandardScaler = _runtime.StandardScaler
System = _runtime.System
TorchCompat = _runtime.TorchCompat
VarianceThreshold = _runtime.VarianceThreshold
_to_sdpa_backends = _runtime._to_sdpa_backends
attention_flops_bshd = _runtime.attention_flops_bshd
datatype = _runtime.datatype
dtypes = _runtime.dtypes
get_device = _runtime.get_device
get_runtime_config = _runtime.get_runtime_config
initialize_sdpa_backends = _runtime.initialize_sdpa_backends
inference = _runtime.inference
is_fake_tensor = _runtime.is_fake_tensor
is_meta_or_fake_tensor = _runtime.is_meta_or_fake_tensor
is_meta_tensor = _runtime.is_meta_tensor
joining = _runtime.joining
no_synchronization = _runtime.no_synchronization
patch_torch = _runtime.patch_torch
postprocess = _runtime.postprocess
preprocess = _runtime.preprocess
sdpa_kernel = _runtime.sdpa_kernel

__all__ = [
    "train",
    "predict",
    "launch",
    "AutoCast",
    "DotProductAttention",
    "MultiScaleRetention",
    "MultiScaleRetentionCompat",
    "AdamW",
    "Module",
    "LossWeightController",
    "attention_flops_bshd",
    "FlopCounter",
    "inference",
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

_sys.modules[__name__ + ".dtypes"] = _runtime.datatype
