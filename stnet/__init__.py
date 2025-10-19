# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any

from . import architecture
from .workflow import OpsConfig

Model = architecture.Model
ModelConfig = architecture.ModelConfig
PatchConfig = architecture.PatchConfig
BuildConfig = architecture.BuildConfig
coerce_config = architecture.coerce_config
SpatialSubnet = architecture.SpatialSubnet
TemporalSubnet = architecture.TemporalSubnet
SpatioTemporalNet = architecture.SpatioTemporalNet
PatchAttention = architecture.PatchAttention
CrossTransformer = architecture.CrossTransformer
Meta = architecture.Meta
MetaNet = architecture.MetaNet
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
    "coerce_config",
    "train",
    "predict",
    "new_model",
    "load_model",
    "save_model",
    "joining",
    "SpatialSubnet",
    "TemporalSubnet",
    "SpatioTemporalNet",
    "PatchAttention",
    "CrossTransformer",
    "Meta",
    "MetaNet",
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
