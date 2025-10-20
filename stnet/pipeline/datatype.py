# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from ..toolkit.compat import patch_arrow

_ARROW = patch_arrow()


_CANONICAL_DTYPE_MAP: Dict[str, Dict[str, Any]] = {
    "float64": {
        "torch": torch.float64,
        "numpy": np.float64,
        "arrow": "float64",
        "python": float,
    },
    "float32": {
        "torch": torch.float32,
        "numpy": np.float32,
        "arrow": "float32",
        "python": float,
    },
    "float16": {
        "torch": torch.float16,
        "numpy": np.float16,
        "arrow": "float16",
        "python": float,
    },
    "bfloat16": {
        "torch": getattr(torch, "bfloat16", torch.float32),
        "numpy": np.float32,
        "arrow": "bfloat16",
        "python": float,
    },
    "int64": {
        "torch": torch.int64,
        "numpy": np.int64,
        "arrow": "int64",
        "python": int,
    },
    "int32": {
        "torch": torch.int32,
        "numpy": np.int32,
        "arrow": "int32",
        "python": int,
    },
    "int16": {
        "torch": torch.int16,
        "numpy": np.int16,
        "arrow": "int16",
        "python": int,
    },
    "int8": {
        "torch": torch.int8,
        "numpy": np.int8,
        "arrow": "int8",
        "python": int,
    },
    "uint8": {
        "torch": torch.uint8,
        "numpy": np.uint8,
        "arrow": "uint8",
        "python": int,
    },
    "bool": {
        "torch": torch.bool,
        "numpy": np.bool_,
        "arrow": "bool",
        "python": bool,
    },
}

_DTYPE_ALIASES = {
    "float": "float32",
    "float_": "float32",
    "double": "float64",
    "float32": "float32",
    "float64": "float64",
    "float16": "float16",
    "half": "float16",
    "halffloat": "float16",
    "boolean": "bool",
    "bool_": "bool",
    "bool": "bool",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "f16": "float16",
    "f32": "float32",
    "f64": "float64",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "u8": "uint8",
}


def _canonical_dtype_name(src: Any) -> str:
    if src is None:
        raise TypeError("dtype cannot be None")
    pa_mod = getattr(_ARROW, "module", None)
    if isinstance(src, torch.dtype):
        key = str(src)
    elif isinstance(src, str):
        key = src
    elif isinstance(src, np.dtype):
        key = src.name
    else:
        data_type_cls = getattr(pa_mod, "DataType", None) if pa_mod else None
        if data_type_cls is not None and isinstance(src, data_type_cls):
            try:
                key = np.dtype(src.to_pandas_dtype()).name
            except Exception:
                key = str(src)
        else:
            try:
                key = np.dtype(src).name
            except Exception:
                key = str(src)
    key = key.strip().lower()
    if key.startswith("torch."):
        key = key.split(".", 1)[1]
    if key.startswith("numpy."):
        key = key.split(".", 1)[1]
    key = key.lstrip("<>|=")
    canonical = _DTYPE_ALIASES.get(key, key)
    if canonical not in _CANONICAL_DTYPE_MAP:
        raise TypeError(f"unsupported dtype: {src!r}")
    return canonical


def convert(src: Any, platform: str) -> Any:
    platform_key = str(platform).strip().lower()
    platform_aliases = {
        "torch": "torch",
        "pytorch": "torch",
        "numpy": "numpy",
        "np": "numpy",
        "arrow": "arrow",
        "pyarrow": "arrow",
        "python": "python",
        "native": "python",
        "name": "name",
        "canonical": "name",
    }
    if platform_key not in platform_aliases:
        raise ValueError(f"unsupported platform: {platform!r}")
    normalized = platform_aliases[platform_key]
    canonical = _canonical_dtype_name(src)
    if normalized == "name":
        return canonical
    mapping = _CANONICAL_DTYPE_MAP.get(canonical)
    if mapping is None or normalized not in mapping:
        raise TypeError(f"unsupported dtype conversion: {src!r} -> {platform!r}")
    return mapping[normalized]


def to_tensor(obj: Any) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    if hasattr(obj, "to_tensor"):
        return obj.to_tensor()
    if hasattr(obj, "as_tensor"):
        return obj.as_tensor()
    return torch.as_tensor(obj)

