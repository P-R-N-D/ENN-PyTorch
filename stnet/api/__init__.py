# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import sys as _sys
from typing import Any, Dict, Tuple

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

_LAZY_IMPORTS: Dict[str, Tuple[str, str | None]] = {
    "launch": ("stnet.api.run", "launch"),
    "predict": ("stnet.api.run", "predict"),
    "train": ("stnet.api.run", "train"),
    "SDPBackend": ("stnet.backend.compat", "SDPBackend"),
    "TorchCompat": ("stnet.backend.compat", "TorchCompat"),
    "_to_sdpa_backends": ("stnet.backend.compat", "_to_sdpa_backends"),
    "is_fake_tensor": ("stnet.backend.compat", "is_fake_tensor"),
    "is_meta_or_fake_tensor": ("stnet.backend.compat", "is_meta_or_fake_tensor"),
    "is_meta_tensor": ("stnet.backend.compat", "is_meta_tensor"),
    "patch_torch": ("stnet.backend.compat", "patch_torch"),
    "sdpa_kernel": ("stnet.backend.compat", "sdpa_kernel"),
    "Distributed": ("stnet.backend.distributed", "Distributed"),
    "Network": ("stnet.backend.distributed", "Network"),
    "joining": ("stnet.backend.distributed", "joining"),
    "no_synchronization": ("stnet.backend.distributed", "no_synchronization"),
    "System": ("stnet.backend.environment", "System"),
    "get_device": ("stnet.backend.environment", "get_device"),
    "get_runtime_config": ("stnet.backend.environment", "get_runtime_config"),
    "initialize_sdpa_backends": (
        "stnet.backend.environment",
        "initialize_sdpa_backends",
    ),
    "FlopCounter": ("stnet.backend.profiler", "FlopCounter"),
    "attention_flops_bshd": ("stnet.backend.profiler", "attention_flops_bshd"),
    "IncrementalPCA": ("stnet.data.transforms", "IncrementalPCA"),
    "StandardScaler": ("stnet.data.transforms", "StandardScaler"),
    "VarianceThreshold": ("stnet.data.transforms", "VarianceThreshold"),
    "postprocess": ("stnet.data.transforms", "postprocess"),
    "preprocess": ("stnet.data.transforms", "preprocess"),
    "AdamW": ("stnet.functional", "AdamW"),
    "AutoCast": ("stnet.functional", "AutoCast"),
    "Gradient": ("stnet.functional", "Gradient"),
    "LayerReplacement": ("stnet.functional", "LayerReplacement"),
    "LossWeightController": ("stnet.functional", "LossWeightController"),
    "DotProductAttention": ("stnet.model.kernels", "DotProductAttention"),
    "MultiScaleRetention": ("stnet.model.kernels", "MultiScaleRetention"),
    "MultiScaleRetentionCompat": ("stnet.model.kernels", "MultiScaleRetentionCompat"),
    "datatype": ("stnet.data.datatype", None),
    "dtypes": ("stnet.data.datatype", None),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'stnet.api' has no attribute '{name}'")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    if name in {"datatype", "dtypes"}:
        _sys.modules[__name__ + ".dtypes"] = module
        globals()["datatype"] = module
        globals()["dtypes"] = module
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__))
