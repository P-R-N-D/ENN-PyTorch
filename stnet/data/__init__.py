# -*- coding: utf-8 -*-
from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

import sys as _sys

from . import datatype as datatype_module

__all__ = [
    "Connector",
    "Dataset",
    "Disposable",
    "Loader",
    "Multiplexer",
    "Prefetcher",
    "Sampler",
    "SourceSpec",
    "collate",
    "dataset",
    "fetch",
    "preload_memmap",
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

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "Connector": ("stnet.data.nodes", "Connector"),
    "Dataset": ("stnet.data.nodes", "Dataset"),
    "Disposable": ("stnet.data.nodes", "Disposable"),
    "Loader": ("stnet.data.nodes", "Loader"),
    "Multiplexer": ("stnet.data.nodes", "Multiplexer"),
    "Prefetcher": ("stnet.data.nodes", "Prefetcher"),
    "Sampler": ("stnet.data.nodes", "Sampler"),
    "SourceSpec": ("stnet.data.nodes", "SourceSpec"),
    "preload_memmap": ("stnet.data.nodes", "preload_memmap"),
    "collate": ("stnet.data.pipeline", "collate"),
    "dataset": ("stnet.data.pipeline", "dataset"),
    "fetch": ("stnet.data.pipeline", "fetch"),
    "Metadata": ("stnet.data.stats", "Metadata"),
    "TensorDictMetadata": ("stnet.data.stats", "TensorDictMetadata"),
    "VarianceThreshold": ("stnet.data.transforms", "VarianceThreshold"),
    "StandardScaler": ("stnet.data.transforms", "StandardScaler"),
    "IncrementalPCA": ("stnet.data.transforms", "IncrementalPCA"),
    "preprocess": ("stnet.data.transforms", "preprocess"),
    "postprocess": ("stnet.data.transforms", "postprocess"),
    "to_dict": ("stnet.data.datatype", "to_dict"),
    "to_tensordict": ("stnet.data.datatype", "to_tensordict"),
    "to_platform_dtype": ("stnet.data.datatype", "to_platform_dtype"),
    "to_torch_tensor": ("stnet.data.datatype", "to_torch_tensor"),
    "to_tuple": ("stnet.data.datatype", "to_tuple"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'stnet.data' has no attribute '{name}'")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__))


_sys.modules[__name__ + ".dtypes"] = datatype_module


datatype = datatype_module
dtypes = datatype_module
