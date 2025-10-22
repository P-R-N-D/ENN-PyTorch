# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any

from . import nn
from .runtime import OpsConfig, ops_config, coerce_ops_config

Model = nn.Model
ModelConfig = nn.ModelConfig
PatchConfig = nn.PatchConfig
BuildConfig = nn.BuildConfig
model_config = nn.model_config
patch_config = nn.patch_config
coerce_model_config = nn.coerce_model_config
coerce_patch_config = nn.coerce_patch_config
SpatialEncoder = nn.SpatialEncoder
TemporalEncoder = nn.TemporalEncoder
LocalProcessor = nn.LocalProcessor
SpatialEncoderLayer = nn.SpatialEncoderLayer
TemporalEncoderLayer = nn.TemporalEncoderLayer
GlobalEncoderLayer = nn.GlobalEncoderLayer
CrossTransformer = nn.CrossTransformer
Payload = nn.Payload
GlobalEncoder = nn.GlobalEncoder
GeGLU = nn.GeGLU
SwiGLU = nn.SwiGLU
MultipleQuantileLoss = nn.MultipleQuantileLoss
StandardNormalLoss = nn.StandardNormalLoss
StudentsTLoss = nn.StudentsTLoss
DataFidelityLoss = nn.DataFidelityLoss
__all__ = [
    "Model",
    "ModelConfig",
    "PatchConfig",
    "BuildConfig",
    "OpsConfig",
    "ops_config",
    "coerce_ops_config",
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
    "SpatialEncoderLayer",
    "TemporalEncoderLayer",
    "GlobalEncoderLayer",
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
    from .runtime.management import new_model as _impl

    return _impl(*args, **kwargs)


def load_model(*args: Any, **kwargs: Any) -> Any:
    from .runtime.management import load_model as _impl

    return _impl(*args, **kwargs)


def save_model(*args: Any, **kwargs: Any) -> Any:
    from .runtime.management import save_model as _impl

    return _impl(*args, **kwargs)


def joining(*args: Any, **kwargs: Any) -> Any:
    from .utils.optimization import joining as _impl

    return _impl(*args, **kwargs)


def train(*args: Any, **kwargs: Any) -> Any:
    from .runtime.operation import train as _impl

    return _impl(*args, **kwargs)


def predict(*args: Any, **kwargs: Any) -> Any:
    from .runtime.operation import predict as _impl

    return _impl(*args, **kwargs)
