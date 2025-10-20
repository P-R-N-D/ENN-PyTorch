# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any

from . import architecture
from .workflow import OpsConfig, ops_config, coerce_ops_config

Model = architecture.Model
ModelConfig = architecture.ModelConfig
PatchConfig = architecture.PatchConfig
BuildConfig = architecture.BuildConfig
model_config = architecture.model_config
patch_config = architecture.patch_config
coerce_model_config = architecture.coerce_model_config
coerce_patch_config = architecture.coerce_patch_config
SpatialEncoder = architecture.SpatialEncoder
TemporalEncoder = architecture.TemporalEncoder
LocalProcessor = architecture.LocalProcessor
SpatialEncoderLayer = architecture.SpatialEncoderLayer
TemporalEncoderLayer = architecture.TemporalEncoderLayer
GlobalEncoderLayer = architecture.GlobalEncoderLayer
CrossTransformer = architecture.CrossTransformer
Payload = architecture.Payload
GlobalEncoder = architecture.GlobalEncoder
GeGLU = architecture.GeGLU
SwiGLU = architecture.SwiGLU
MultipleQuantileLoss = architecture.MultipleQuantileLoss
StandardNormalLoss = architecture.StandardNormalLoss
StudentsTLoss = architecture.StudentsTLoss
DataFidelityLoss = architecture.DataFidelityLoss
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
    from .workflow.management import new_model as _impl

    return _impl(*args, **kwargs)


def load_model(*args: Any, **kwargs: Any) -> Any:
    from .workflow.management import load_model as _impl

    return _impl(*args, **kwargs)


def save_model(*args: Any, **kwargs: Any) -> Any:
    from .workflow.management import save_model as _impl

    return _impl(*args, **kwargs)


def joining(*args: Any, **kwargs: Any) -> Any:
    from .toolkit.optimization import joining as _impl

    return _impl(*args, **kwargs)


def train(*args: Any, **kwargs: Any) -> Any:
    from .workflow.operation import train as _impl

    return _impl(*args, **kwargs)


def predict(*args: Any, **kwargs: Any) -> Any:
    from .workflow.operation import predict as _impl

    return _impl(*args, **kwargs)
