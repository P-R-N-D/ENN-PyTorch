# -*- coding: utf-8 -*-
from __future__ import annotations

import sys as _sys

from . import datatype as datatype_module
from .pipeline import BatchLoader, collate, launch
from .nodes import BatchSampler, SampleReader
from .stats import Metadata, TensorDictMetadata
from .transforms import (
    IncrementalPCA,
    StandardScaler,
    VarianceThreshold,
    batch_to_tensordict,
    postprocess,
    preprocess,
    tensordict_to_dict,
)
from .datatype import convert, to_torch_tensor

__all__ = [
    "BatchSampler",
    "BatchLoader",
    "SampleReader",
    "collate",
    "launch",
    "Metadata",
    "TensorDictMetadata",
    "VarianceThreshold",
    "StandardScaler",
    "IncrementalPCA",
    "batch_to_tensordict",
    "preprocess",
    "postprocess",
    "tensordict_to_dict",
    "convert",
    "to_torch_tensor",
    "datatype",
    "dtypes",
]


_sys.modules[__name__ + ".dtypes"] = datatype_module


datatype = datatype_module
dtypes = datatype_module
