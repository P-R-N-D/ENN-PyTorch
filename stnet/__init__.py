# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any

from tensordict import set_list_to_stack

set_list_to_stack(True).set()  # 리스트를 배치 차원으로 자동 스택

# 동적 shape 구간은 자동으로 CUDA Graphs 캡처 스킵
# (PyTorch inductor 설정: dynamic graph는 그래프 캡처에서 생략)
try:
    import torch
    import torch._inductor.config as _inductor_cfg

    _inductor_cfg.triton.cudagraph_skip_dynamic_graphs = True
except Exception:
    pass

from .api.config import (
    BuildConfig,
    ModelConfig,
    OpsMode,
    PatchConfig,
    RuntimeConfig,
    coerce_model_config,
    coerce_patch_config,
    coerce_runtime_config,
    model_config,
    patch_config,
    runtime_config,
)

__all__ = [
    "Root",
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
    "train",
    "predict",
    "new_model",
    "load_model",
    "save_model",
    "joining",
    "SpatialEncoder",
    "TemporalEncoder",
    "LocalProcessor",
    "PatchAttention",
    "CrossAttention",
    "PointTransformer",
    "TemporalEncoderLayer",
    "TemporalEncoderBlock",
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
    "TensorDictLoss",
]


def __getattr__(name: str) -> Any:
    if name in {
        "Root",
        "SpatialEncoder",
        "TemporalEncoder",
        "LocalProcessor",
        "PatchAttention",
        "CrossAttention",
        "PointTransformer",
        "TemporalEncoderLayer",
        "TemporalEncoderBlock",
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
        "TensorDictLoss",
    }:
        from . import model as model_ns

        mapping = {
            "Root": model_ns.Root,
            "SpatialEncoder": model_ns.SpatialEncoder,
            "TemporalEncoder": model_ns.TemporalEncoder,
            "LocalProcessor": model_ns.LocalProcessor,
            "PatchAttention": model_ns.PatchAttention,
            "CrossAttention": model_ns.CrossAttention,
            "PointTransformer": model_ns.PointTransformer,
            "TemporalEncoderLayer": model_ns.TemporalEncoderLayer,
            "TemporalEncoderBlock": model_ns.TemporalEncoderBlock,
            "DilatedAttention": model_ns.DilatedAttention,
            "LongNet": model_ns.LongNet,
            "CrossTransformer": model_ns.CrossTransformer,
            "Payload": model_ns.Payload,
            "GlobalEncoder": model_ns.GlobalEncoder,
            "GeGLU": model_ns.GeGLU,
            "SwiGLU": model_ns.SwiGLU,
            "MultipleQuantileLoss": model_ns.MultipleQuantileLoss,
            "StandardNormalLoss": model_ns.StandardNormalLoss,
            "StudentsTLoss": model_ns.StudentsTLoss,
            "DataFidelityLoss": model_ns.DataFidelityLoss,
            "TensorDictLoss": model_ns.TensorDictLoss,
        }
        return mapping[name]
    if name in {"train", "predict", "new_model", "load_model", "save_model", "joining"}:
        from .backend import joining as _joining, load_model as _load_model
        from .backend import new_model as _new_model, predict as _predict, save_model as _save_model
        from .backend import train as _train

        mapping = {
            "train": _train,
            "predict": _predict,
            "new_model": _new_model,
            "load_model": _load_model,
            "save_model": _save_model,
            "joining": _joining,
        }
        return mapping[name]
    raise AttributeError(f"module 'stnet' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(__all__))

