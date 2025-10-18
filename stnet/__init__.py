from __future__ import annotations

from typing import Any

from . import architecture

Model = architecture.Model
Config = architecture.Config
PatchParameters = architecture.PatchParameters
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
    "Config",
    "PatchParameters",
    "coerce_config",
    "train",
    "predict",
    "new_model",
    "load_model",
    "save_model",
    "to_onnx",
    "to_coreml",
    "to_litert",
    "to_executorch",
    "to_script",
    "to_tensorflow",
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


def train(*args: Any, **kwargs: Any) -> Any:
    from .workflow.operation import train as _impl

    return _impl(*args, **kwargs)


def predict(*args: Any, **kwargs: Any) -> Any:
    from .workflow.operation import predict as _impl

    return _impl(*args, **kwargs)


def to_onnx(*args: Any, **kwargs: Any) -> Any:
    from .workflow.management import to_onnx as _impl

    return _impl(*args, **kwargs)


def to_coreml(*args: Any, **kwargs: Any) -> Any:
    from .workflow.management import to_coreml as _impl

    return _impl(*args, **kwargs)


def to_litert(*args: Any, **kwargs: Any) -> Any:
    from .workflow.management import to_litert as _impl

    return _impl(*args, **kwargs)


def to_executorch(*args: Any, **kwargs: Any) -> Any:
    from .workflow.management import to_executorch as _impl

    return _impl(*args, **kwargs)


def to_script(*args: Any, **kwargs: Any) -> Any:
    from .workflow.management import to_script as _impl

    return _impl(*args, **kwargs)


def to_tensorflow(*args: Any, **kwargs: Any) -> Any:
    from .workflow.management import to_tensorflow as _impl

    return _impl(*args, **kwargs)
