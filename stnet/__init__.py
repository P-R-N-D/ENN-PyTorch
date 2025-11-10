# -*- coding: utf-8 -*-
from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

from tensordict import set_list_to_stack

set_list_to_stack(True).set()

try:
    import torch
    import torch._inductor.config as _inductor_cfg

    _inductor_cfg.triton.cudagraph_skip_dynamic_graphs = True
except Exception:
    pass

_MODEL_EXPORTS = (
    "Root",
    "SpatialEncoder",
    "TemporalEncoder",
    "LocalProcessor",
    "PatchAttention",
    "CrossAttention",
    "PointTransformer",
    "Retention",
    "RetNet",
    "DilatedAttention",
    "LongNet",
    "CrossTransformer",
    "Payload",
    "GlobalEncoder",
    "GeGLU",
    "SwiGLU",
    "MultipleQuantileLoss",
    "StandardNormalLoss",
    "StudentsTLoss",
    "DataFidelityLoss",
)

_BACKEND_EXPORTS: Dict[str, Tuple[str, str]] = {
    "train": ("stnet.api.run", "train"),
    "predict": ("stnet.api.run", "predict"),
    "new_model": ("stnet.api.io", "new_model"),
    "load_model": ("stnet.api.io", "load_model"),
    "save_model": ("stnet.api.io", "save_model"),
    "joining": ("stnet.backend.distributed", "joining"),
}

_CONFIG_EXPORTS: Dict[str, Tuple[str, str]] = {
    "ModelConfig": ("stnet.api.config", "ModelConfig"),
    "PatchConfig": ("stnet.api.config", "PatchConfig"),
    "BuildConfig": ("stnet.api.config", "BuildConfig"),
    "OpsMode": ("stnet.api.config", "OpsMode"),
    "RuntimeConfig": ("stnet.api.config", "RuntimeConfig"),
    "runtime_config": ("stnet.api.config", "runtime_config"),
    "coerce_runtime_config": ("stnet.api.config", "coerce_runtime_config"),
    "model_config": ("stnet.api.config", "model_config"),
    "patch_config": ("stnet.api.config", "patch_config"),
    "coerce_model_config": ("stnet.api.config", "coerce_model_config"),
    "coerce_patch_config": ("stnet.api.config", "coerce_patch_config"),
}

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    **{name: ("stnet.model", name) for name in _MODEL_EXPORTS},
    **_BACKEND_EXPORTS,
    **_CONFIG_EXPORTS,
}

__all__ = [
    "ModelConfig",
    "PatchConfig",
    "BuildConfig",
    "OpsMode",
    "RuntimeConfig",
    "runtime_config",
    "coerce_runtime_config",
    "model_config",
    "patch_config",
    "coerce_model_config",
    "coerce_patch_config",
    *_BACKEND_EXPORTS.keys(),
    *_MODEL_EXPORTS,
]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'stnet' has no attribute '{name}'")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__))