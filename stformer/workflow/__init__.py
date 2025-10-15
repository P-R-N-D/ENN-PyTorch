# -*- coding: utf-8 -*-
from __future__ import annotations

from ..architecture.network import Config, PatchParameters
from .management import (
    new_model,
    load_model,
    save_model,
    to_onnx,
    to_coreml,
    to_tensorrt,
    to_litert,
    to_executorch,
    to_script,
    to_tensorflow,
)
from .operation import train, predict

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