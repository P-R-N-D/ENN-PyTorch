# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import os

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase

_TRUE = {"1", "true", "yes", "y", "on", "enable", "enabled"}
_FALSE = {"0", "false", "no", "n", "off", "disable", "disabled"}


def parse_bool(value: object) -> Optional[bool]:
    """Parse common boolean env tokens.

    Returns:
        True/False for recognized tokens, otherwise None.
    """

    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    if s in _TRUE:
        return True
    elif s in _FALSE:
        return False
    return None


def env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def env_bool(name: str, default: bool = False) -> bool:
    v = parse_bool(os.environ.get(name))
    if v is None:
        return bool(default)
    return bool(v)


def env_int(name: str, default: int = 0) -> int:
    s = env_str(name)
    if s is None:
        return int(default)
    try:
        return int(s)
    except Exception:
        return int(default)


def env_float(name: str, default: float = 0.0) -> float:
    s = env_str(name)
    if s is None:
        return float(default)
    try:
        return float(s)
    except Exception:
        return float(default)


def env_first(keys: Sequence[str]) -> Optional[str]:
    """Return the first non-empty env value among keys."""

    for k in keys:
        v = os.environ.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def env_first_bool(keys: Sequence[str], default: bool = False) -> bool:
    v = parse_bool(env_first(keys))
    if v is None:
        return bool(default)
    return bool(v)


def env_first_int(keys: Sequence[str], default: int = 0) -> int:
    v = env_first(keys)
    if v is None:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def env_first_float(keys: Sequence[str], default: float = 0.0) -> float:
    v = env_first(keys)
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)

_CANONICAL_DTYPES: Dict[str, Dict[str, Any]] = {
    "float64": {
        "torch": torch.float64,
        "numpy": np.float64,
        "python": float,
    },
    "float32": {
        "torch": torch.float32,
        "numpy": np.float32,
        "python": float,
    },
    "float16": {
        "torch": torch.float16,
        "numpy": np.float16,
        "python": float,
    },
    "bfloat16": {
        "torch": getattr(torch, "bfloat16", torch.float32),
        "numpy": np.float32,
        "python": float,
    },
    "int64": {
        "torch": torch.int64,
        "numpy": np.int64,
        "python": int,
    },
    "int32": {
        "torch": torch.int32,
        "numpy": np.int32,
        "python": int,
    },
    "int16": {
        "torch": torch.int16,
        "numpy": np.int16,
        "python": int,
    },
    "int8": {
        "torch": torch.int8,
        "numpy": np.int8,
        "python": int,
    },
    "uint8": {
        "torch": torch.uint8,
        "numpy": np.uint8,
        "python": int,
    },
    "bool": {
        "torch": torch.bool,
        "numpy": np.bool_,
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
    "python": "python",
    "native": "python",
    "name": "name",
    "canonical": "name",
}

BatchLike = Union[Mapping[str, Any], TensorDictBase]


def _canonical_dtype(src: Any) -> str:
    if src is None:
        raise TypeError("dtype cannot be None")
    if isinstance(src, torch.dtype):
        key = str(src)
    elif isinstance(src, str):
        key = src
    elif isinstance(src, np.dtype):
        key = src.name
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
    if canonical not in _CANONICAL_DTYPES:
        raise TypeError(f"unsupported dtype: {src!r} (normalized key={key!r})")
    return canonical


def to_platform_dtype(src: Any, platform: str) -> Any:
    platform_key = str(platform).strip().lower()
    normalized = _PLATFORM_ALIASES.get(platform_key)
    if normalized is None:
        raise ValueError(f"unsupported platform: {platform!r}")
    canonical = _canonical_dtype(src)
    if normalized == "name":
        return canonical
    mapping = _CANONICAL_DTYPES.get(canonical)
    if mapping is None or normalized not in mapping:
        raise TypeError(f"unsupported dtype conversion: {src!r} -> {platform!r}")
    return mapping[normalized]


def to_torch_tensor(obj: Any) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    for attr in ("to_torch_tensor", "to_torch", "to_tensor", "as_tensor"):
        method = getattr(obj, attr, None)
        if callable(method):
            with torch.no_grad():
                out = method()
            if isinstance(out, torch.Tensor):
                return out
            try:
                return torch.as_tensor(out)
            except Exception:
                continue
    return torch.as_tensor(obj)


def _get_batch_size(d: Mapping[str, Any]) -> list[int] | list:
    for value in d.values():
        if torch.is_tensor(value) and value.ndim >= 1:
            return [value.shape[0]]
    return []


def to_tensordict(
    batch: BatchLike,
    *args: Any,
    device: Optional[torch.device] = None,
    batch_size: Optional[Iterable[int]] = None,
    **kwargs: Any,
) -> TensorDict:
    if isinstance(batch, TensorDictBase):
        return batch.to(device) if device is not None else batch                              
    if not isinstance(batch, Mapping):
        raise TypeError(f"Unexpected batch type: {type(batch)}")

    resolved_batch = list(batch_size) if batch_size is not None else _get_batch_size(batch)
    td = TensorDict({}, batch_size=resolved_batch, device=device)

    for key, value in batch.items():
        if torch.is_tensor(value):
            tensor = value
            if device is not None:
                tensor = tensor.to(device)
            td.set(key, tensor)
        elif isinstance(value, TensorDictBase):
            nested = value.to(device) if device is not None else value
            td.set(key, nested)
        else:
            td.set_non_tensor(key, value)
    return td


@torch.no_grad()
def to_dict(
    td_or_dict: Union[TensorDictBase, Mapping[str, Any]],
    *args: Any,
    detach: bool = True,
    cpu: bool = True,
    keys: Optional[Iterable[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    if isinstance(td_or_dict, TensorDictBase):
        td = td_or_dict
        key_iter = keys or td.keys()
        items = ((key, td.get(key)) for key in list(key_iter))
        out: Dict[str, Any] = {}
        for key, value in items:
            if torch.is_tensor(value):
                tensor = value.detach() if detach else value
                if cpu and tensor.is_cuda:
                    tensor = tensor.cpu()
                out[key] = tensor
            elif isinstance(value, TensorDictBase):
                out[key] = to_dict(value, detach=detach, cpu=cpu)
            else:
                out[key] = value
        return out
    if not isinstance(td_or_dict, Mapping):
        raise TypeError(f"to_dict expects TensorDictBase or Mapping, got: {type(td_or_dict)}")

    out: Dict[str, Any] = {}
    for key, value in td_or_dict.items():
        if torch.is_tensor(value):
            tensor = value.detach() if detach else value
            tensor = tensor.cpu() if cpu and tensor.is_cuda else tensor
            out[key] = tensor
        else:
            out[key] = value
    return out


def to_tuple(x: Any) -> Tuple:
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    elif isinstance(x, torch.Tensor):
        return tuple(x.flatten().detach().cpu().tolist())
    elif hasattr(x, "tolist"):
        values = x.tolist()
        return tuple(values if isinstance(values, (list, tuple)) else [values])
    else:
        return (x,)
