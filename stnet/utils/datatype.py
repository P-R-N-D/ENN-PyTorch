# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch

from .compat import patch_arrow

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

_PLATFORM_ALIASES = {
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

_BOOL_TEXT = {
    "true": True,
    "True": True,
    "1": True,
    "false": False,
    "False": False,
    "0": False,
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


def to(src: Any, platform: str) -> Any:
    platform_key = str(platform).strip().lower()
    normalized = _PLATFORM_ALIASES.get(platform_key)
    if normalized is None:
        raise ValueError(f"unsupported platform: {platform!r}")
    canonical = _canonical_dtype_name(src)
    if normalized == "name":
        return canonical
    mapping = _CANONICAL_DTYPE_MAP.get(canonical)
    if mapping is None or normalized not in mapping:
        raise TypeError(f"unsupported dtype conversion: {src!r} -> {platform!r}")
    return mapping[normalized]


def to_torch(obj: Any) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    for attr in ("to_torch", "to_tensor", "as_tensor"):
        method = getattr(obj, attr, None)
        if callable(method):
            return method()
    return torch.as_tensor(obj)





def ensure_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    elif isinstance(value, (int, float)):
        return bool(value)
    elif isinstance(value, str):
        normalized = value.strip()
        if normalized in _BOOL_TEXT:
            return _BOOL_TEXT[normalized]
    raise TypeError(f"{name} must be a boolean-compatible value")


def ensure_int(
    value: Any,
    *,
    name: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer-compatible value") from exc
    if minimum is not None and ivalue < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {ivalue}")
    if maximum is not None and ivalue > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {ivalue}")
    return ivalue


def ensure_float(
    value: Any,
    *,
    name: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    try:
        fvalue = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a float-compatible value") from exc
    if minimum is not None and fvalue < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {fvalue}")
    if maximum is not None and fvalue > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {fvalue}")
    return fvalue


def ensure_int_tuple(
    value: Any,
    *,
    name: str,
    dims: int,
    allow_none: bool = False,
    keep_scalar: bool = False,
) -> Optional[Union[int, Tuple[int, ...]]]:
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{name} cannot be None")
    if isinstance(value, int):
        ivalue = ensure_int(value, name=name, minimum=1)
        if keep_scalar:
            return cast(Union[int, Tuple[int, ...]], ivalue)
        return tuple([ivalue] * dims)
    if isinstance(value, (list, tuple)):
        if len(value) != dims:
            raise ValueError(f"{name} must have length {dims}, got {len(value)}")
        items = tuple(ensure_int(v, name=name, minimum=1) for v in value)
        return items
    raise TypeError(f"{name} must be an int or sequence of {dims} integers")


def ensure_int_sequence(xs: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(x) for x in xs)
