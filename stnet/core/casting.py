# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Optional

import numpy as np
import torch

_TRUE = frozenset({"1", "true", "yes", "y", "on", "enable", "enabled"})
_FALSE = frozenset({"0", "false", "no", "n", "off", "disable", "disabled"})


def parse_bool(value: object) -> bool | None:
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
    if s in _FALSE:
        return False
    return None


def _env_clean(value: object | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def env_str(name: str, default: str | None = None) -> str | None:
    s = _env_clean(os.getenv(name))
    return s if s is not None else default


def _env_cast(name: str, cast: Callable[[str], Any], default: Any) -> Any:
    s = env_str(name)
    if s is None:
        return default
    try:
        return cast(s)
    except (ValueError, TypeError):
        return default


def env_bool(name: str | Sequence[str], default: bool = False) -> bool:
    raw: object | None
    if isinstance(name, Sequence) and not isinstance(name, (str, bytes, bytearray)):
        raw = env_first(list(name), default=None)
    else:
        raw = os.environ.get(str(name))

    v = parse_bool(raw)
    if v is None:
        return bool(default)
    return bool(v)


def env_int(name: str, default: int = 0) -> int:
    return int(_env_cast(name, int, int(default)))


def env_float(name: str, default: float = 0.0) -> float:
    return float(_env_cast(name, float, float(default)))


def env_first(keys: Sequence[str], default: str | None = None) -> str | None:
    """Return the first non-empty env value among keys.

    Args:
        keys: environment variable names in priority order.
        default: value returned when none of the keys are set (or all are empty).

    Notes:
        This keeps call-sites compact and avoids sprinkling `or <default>` everywhere.
    """

    for k in keys:
        s = _env_clean(os.getenv(k))
        if s is not None:
            return s
    return default


def env_flag(*keys: str, default: bool = False) -> bool:
    """Parse a boolean-ish environment flag across multiple keys.

    - Recognized boolean tokens follow `parse_bool()`
    - Any other non-empty value is treated as True (legacy-compatible behavior)
    """
    if not keys:
        return bool(default)

    raw = env_first(keys)
    v = parse_bool(raw)
    if v is not None:
        return bool(v)

    if raw is None:
        return bool(default)

    # Legacy behavior: any non-empty string => True
    return bool(str(raw).strip())


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
    except (ValueError, TypeError):
        return int(default)


def env_first_float(keys: Sequence[str], default: float = 0.0) -> float:
    v = env_first(keys)
    if v is None:
        return float(default)
    try:
        return float(v)
    except (ValueError, TypeError):
        return float(default)

_CANONICAL_DTYPES: dict[str, dict[str, Any]] = {
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

_DTYPE_ALIASES: dict[str, str] = {
    "float": "float32",
    "float_": "float32",
    "double": "float64",
    "fp64": "float64",
    "float32": "float32",
    "fp32": "float32",
    "float64": "float64",
    "float16": "float16",
    "half": "float16",
    "halffloat": "float16",
    "fp16": "float16",
    "boolean": "bool",
    "bool_": "bool",
    "bool": "bool",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "f16": "float16",
    "f32": "float32",
    "f64": "float64",
    # ints
    "long": "int64",
    "int": "int32",
    "short": "int16",
    "char": "int8",
    "byte": "uint8",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "u8": "uint8",
}

_PLATFORM_ALIASES: dict[str, str] = {
    "torch": "torch",
    "pytorch": "torch",
    "numpy": "numpy",
    "np": "numpy",
    "python": "python",
    "native": "python",
    "name": "name",
    "canonical": "name",
}


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
    # Handle representations like dtype('float32') or dtype(float32)
    if key.startswith("dtype(") and key.endswith(")"):
        inner = key[5:].strip("()").strip().strip("'\"")
        if inner:
            key = inner
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
    if mapping is None:
        raise TypeError(f"unsupported dtype conversion: {src!r} -> {platform!r}")
    try:
        return mapping[normalized]
    except KeyError as e:
        raise TypeError(f"unsupported dtype conversion: {src!r} -> {platform!r}") from e


def parse_torch_dtype(src: Any) -> Optional[torch.dtype]:
    """Best-effort torch dtype parser.

    Accepts:
      - torch.dtype
      - strings like "float32", "torch.float32", "fp32", "long", ...
      - numpy dtypes / dtype-like objects

    Returns None when parsing fails (caller can fall back to a default).
    """

    if src is None:
        return None
    if isinstance(src, torch.dtype):
        return src
    try:
        return to_platform_dtype(src, "torch")
    except Exception:
        pass

    # Fallback: torch.<name> attribute lookup.
    try:
        key = str(src).strip()
    except Exception:
        return None

    if not key:
        return None
    if key.startswith("torch."):
        key = key.split(".", 1)[1]
    with contextlib.suppress(Exception):
        dt = getattr(torch, key)
        if isinstance(dt, torch.dtype):
            return dt
    return None


def dtype_from_name(name: Any, default: torch.dtype) -> torch.dtype:
    """Parse a torch dtype from a name-like input; fall back to `default`."""

    dt = parse_torch_dtype(name)
    return dt if isinstance(dt, torch.dtype) else default


def to_torch_tensor(obj: Any) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    for attr in ("to_torch_tensor", "to_torch", "to_tensor", "as_tensor"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                out = method()
            except TypeError:
                # exists but requires args / incompatible signature
                continue
            except Exception:
                # best-effort: ignore custom conversion failures
                continue
            if isinstance(out, torch.Tensor):
                return out
            try:
                return torch.as_tensor(out)
            except Exception:
                continue
    try:
        return torch.as_tensor(obj)
    except Exception as e:
        raise TypeError(f"cannot convert to torch.Tensor: {type(obj)}") from e


def _is_numpy_non_numeric(x: np.ndarray | np.generic) -> bool:
    dt = getattr(x, "dtype", None)
    if dt is None:
        return False
    # Skip dtype kinds that are not safely convertible to tensors
    # - object / string: "O", "U", "S"
    # - datetime / timedelta: "M", "m"
    # - void / structured: "V"
    return dt.kind in {"O", "U", "S", "M", "m", "V"}
