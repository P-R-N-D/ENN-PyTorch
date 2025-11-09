# -*- coding: utf-8 -*-
from __future__ import annotations

import sys as _sys

from . import datatype as datatype_module
from .nodes import Dataset, DevicePrefetcher, preload_memmap
from .pipeline import BatchLoader, Disposable, ThreadLoadBalancer, collate, fetch
from .stats import Metadata, TensorDictMetadata
from .transforms import (
    IncrementalPCA,
    StandardScaler,
    VarianceThreshold,
    postprocess,
    preprocess,
)
from .datatype import (
    to_dict,
    to_platform_dtype,
    to_tensordict,
    to_torch_tensor,
    to_tuple,
)

__all__ = [
    "BatchLoader",
    "DevicePrefetcher",
    "Dataset",
    "collate",
    "fetch",
    "preload_memmap",
    "Disposable",
    "ThreadLoadBalancer",
    "Metadata",
    "TensorDictMetadata",
    "VarianceThreshold",
    "StandardScaler",
    "IncrementalPCA",
    "preprocess",
    "postprocess",
    "to_dict",
    "to_tensordict",
    "to_platform_dtype",
    "to_torch_tensor",
    "to_tuple",
    "datatype",
    "dtypes",
]


_sys.modules[__name__ + ".dtypes"] = datatype_module


datatype = datatype_module
dtypes = datatype_module
