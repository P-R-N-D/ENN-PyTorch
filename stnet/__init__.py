# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any

from . import model as model_ns
from .config import (
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

Root = model_ns.Root
SpatialEncoder = model_ns.SpatialEncoder
TemporalEncoder = model_ns.TemporalEncoder
LocalProcessor = model_ns.LocalProcessor
PatchAttention = model_ns.PatchAttention
CrossAttention = model_ns.CrossAttention
PointTransformer = model_ns.PointTransformer
TemporalEncoderLayer = model_ns.TemporalEncoderLayer
TemporalEncoderBlock = model_ns.TemporalEncoderBlock
GlobalEncoderLayer = model_ns.GlobalEncoderLayer
GlobalEncoderBlock = model_ns.GlobalEncoderBlock
CrossTransformer = model_ns.CrossTransformer
Payload = model_ns.Payload
GlobalEncoder = model_ns.GlobalEncoder
GeGLU = model_ns.GeGLU
SwiGLU = model_ns.SwiGLU
MultipleQuantileLoss = model_ns.MultipleQuantileLoss
StandardNormalLoss = model_ns.StandardNormalLoss
StudentsTLoss = model_ns.StudentsTLoss
DataFidelityLoss = model_ns.DataFidelityLoss
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
    "GlobalEncoderLayer",
    "GlobalEncoderBlock",
    "CrossTransformer",
    "Payload",
    "GlobalEncoder",
    "GeGLU",
    "SwiGLU",
    "MultipleQuantileLoss",
    "StandardNormalLoss",
    "StudentsTLoss",
    "DataFidelityLoss",
]


def new_model(*args: Any, **kwargs: Any) -> Any:
    from .utils.io import new_model as _impl

    return _impl(*args, **kwargs)


def load_model(*args: Any, **kwargs: Any) -> Any:
    from .utils.io import load_model as _impl

    return _impl(*args, **kwargs)


def save_model(*args: Any, **kwargs: Any) -> Any:
    from .utils.io import save_model as _impl

    return _impl(*args, **kwargs)


def joining(*args: Any, **kwargs: Any) -> Any:
    from .utils.optimization import joining as _impl

    return _impl(*args, **kwargs)


def train(*args: Any, **kwargs: Any) -> Any:
    from .runtime.launch import train as _impl

    return _impl(*args, **kwargs)


def predict(*args: Any, **kwargs: Any) -> Any:
    from .runtime.launch import predict as _impl

    return _impl(*args, **kwargs)
