# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any

from ..architecture.network import Config, PatchParameters


def new_model(*args: Any, **kwargs: Any) -> Any:
    from .management import new_model as _impl

    return _impl(*args, **kwargs)


def load_model(*args: Any, **kwargs: Any) -> Any:
    from .management import load_model as _impl

    return _impl(*args, **kwargs)


def save_model(*args: Any, **kwargs: Any) -> Any:
    from .management import save_model as _impl

    return _impl(*args, **kwargs)


def to_onnx(*args: Any, **kwargs: Any) -> Any:
    from .management import to_onnx as _impl

    return _impl(*args, **kwargs)


def to_coreml(*args: Any, **kwargs: Any) -> Any:
    from .management import to_coreml as _impl

    return _impl(*args, **kwargs)


def to_tensorrt(*args: Any, **kwargs: Any) -> Any:
    from .management import to_tensorrt as _impl

    return _impl(*args, **kwargs)


def to_litert(*args: Any, **kwargs: Any) -> Any:
    from .management import to_litert as _impl

    return _impl(*args, **kwargs)


def to_executorch(*args: Any, **kwargs: Any) -> Any:
    from .management import to_executorch as _impl

    return _impl(*args, **kwargs)


def to_script(*args: Any, **kwargs: Any) -> Any:
    from .management import to_script as _impl

    return _impl(*args, **kwargs)


def to_tensorflow(*args: Any, **kwargs: Any) -> Any:
    from .management import to_tensorflow as _impl

    return _impl(*args, **kwargs)


def train(*args: Any, **kwargs: Any) -> Any:
    from .operation import train as _impl

    return _impl(*args, **kwargs)


def predict(*args: Any, **kwargs: Any) -> Any:
    from .operation import predict as _impl

    return _impl(*args, **kwargs)


to_tflite = to_litert
to_tf = to_tensorflow
to_trt = to_tensorrt

__all__ = [
    'Config',
    'PatchParameters',
    'new_model',
    'load_model',
    'save_model',
    'to_onnx',
    'to_tensorrt',
    'to_coreml',
    'to_litert',
    'to_executorch',
    'to_script',
    'to_tensorflow',
    'train',
    'predict',
]
