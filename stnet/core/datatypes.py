# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeAlias

import numpy
import torch

_CANONICAL_DTYPES: dict[str, dict[str, Any]] = {
    "float64": {
        "torch": torch.float64,
        "numpy": numpy.float64,
        "python": float,
    },
    "float32": {
        "torch": torch.float32,
        "numpy": numpy.float32,
        "python": float,
    },
    "float16": {
        "torch": torch.float16,
        "numpy": numpy.float16,
        "python": float,
    },
    "bfloat16": {
        "torch": getattr(torch, "bfloat16", torch.float32),
        "numpy": numpy.float32,
        "python": float,
    },
    "int64": {
        "torch": torch.int64,
        "numpy": numpy.int64,
        "python": int,
    },
    "int32": {
        "torch": torch.int32,
        "numpy": numpy.int32,
        "python": int,
    },
    "int16": {
        "torch": torch.int16,
        "numpy": numpy.int16,
        "python": int,
    },
    "int8": {
        "torch": torch.int8,
        "numpy": numpy.int8,
        "python": int,
    },
    "uint8": {
        "torch": torch.uint8,
        "numpy": numpy.uint8,
        "python": int,
    },
    "bool": {
        "torch": torch.bool,
        "numpy": numpy.bool_,
        "python": bool,
    },
}
_DEF_UNDERFLOW_ACTIONS = {"allow", "warn", "forbid"}
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
_FALSE = frozenset({"0", "false", "no", "n", "off", "disable", "disabled"})
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
_TRUE = frozenset({"1", "true", "yes", "y", "on", "enable", "enabled"})
JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = (
    JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
)
PathLike: TypeAlias = str | os.PathLike[str] | Path

def _env_cast(name: str, cast: Callable[[str], Any], default: Any) -> Any:
    s = env_str(name)
    if s is None:
        return default
    try:
        return cast(s)
    except (ValueError, TypeError):
        return default
def _env_clean(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode(errors="ignore")
    s = str(value).replace("\\r\\n", "\n").replace("\\n", "\n").strip()
    return s or None
def _canonical_dtype(src: Any) -> str:
    if src is None:
        raise TypeError("dtype cannot be None")
    if isinstance(src, torch.dtype):
        key = str(src)
    elif isinstance(src, str):
        key = src
    elif isinstance(src, numpy.dtype):
        key = src.name
    else:
        try:
            key = numpy.dtype(src).name
        except Exception:
            key = str(src)
    key = key.strip().lower()
    if key.startswith("torch."):
        key = key.split(".", 1)[1]
    if key.startswith("numpy."):
        key = key.split(".", 1)[1]
    if key.startswith("dtype(") and key.endswith(")"):
        inner = key[5:].strip("()").strip().strip("'\"")
        if inner:
            key = inner
    key = key.lstrip("<>|=")
    canonical = _DTYPE_ALIASES.get(key, key)
    if canonical not in _CANONICAL_DTYPES:
        raise TypeError(f"unsupported dtype: {src!r} (normalized key={key!r})")
    return canonical
def _atomic_swap(path: PathLike):
    p = os.fspath(path)
    parent = os.path.dirname(p) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=os.path.basename(p) + ".", suffix=".tmp", dir=parent
    )
    os.close(fd)
    try:
        yield tmp_name
        os.replace(tmp_name, p)
    finally:
        with contextlib.suppress(Exception):
            os.remove(tmp_name)

def parse_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode(errors="ignore")
    s = str(value).replace("\\r\\n", "\n").replace("\\n", "\n").strip().lower()
    if not s:
        return None
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    return None
def env_str(name: str, default: str | None = None) -> str | None:
    s = _env_clean(os.getenv(name))
    return s if s is not None else default
def env_bool(name: str | Sequence[str], default: bool = False) -> bool:
    raw: object | None
    if isinstance(name, Sequence) and not isinstance(
        name, (str, bytes, bytearray)
    ):
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
    for k in keys:
        s = _env_clean(os.getenv(k))
        if s is not None:
            return s
    return default
def env_flag(*keys: str, default: bool = False) -> bool:
    if not keys:
        return bool(default)
    raw = env_first(keys)
    v = parse_bool(raw)
    if v is not None:
        return bool(v)
    if raw is None:
        return bool(default)
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
        raise TypeError(
            f"unsupported dtype conversion: {src!r} -> {platform!r}"
        )
    try:
        return mapping[normalized]
    except KeyError as e:
        raise TypeError(
            f"unsupported dtype conversion: {src!r} -> {platform!r}"
        ) from e
def parse_torch_dtype(src: Any) -> torch.dtype | None:
    if src is None:
        return None
    if isinstance(src, torch.dtype):
        return src
    try:
        return to_platform_dtype(src, "torch")
    except Exception:
        pass
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
    dt = parse_torch_dtype(name)
    return dt if isinstance(dt, torch.dtype) else default
def get_meta_path(mmt_path: str) -> str:
    return str(mmt_path) + ".meta.json"
def read_json(path: PathLike) -> JsonValue:
    with open(os.fspath(path), "r", encoding="utf-8-sig") as f:
        return json.load(f)
def coerce_json(obj: object) -> JsonValue:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (torch.device, Path, torch.dtype)):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return {
            "__tensor__": True,
            "shape": list(map(int, obj.shape)),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
        }
    if isinstance(obj, dict):
        return {str(k): coerce_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [coerce_json(v) for v in obj]
    return str(obj)
def write_json(
    path: PathLike, payload: Any, *args: Any, indent: int | None = 2
) -> None:
    with _atomic_swap(path) as tmp, open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent)
def save_temp(path: PathLike, payload: Any, **opts: Any) -> None:
    with _atomic_swap(path) as tmp:
        torch.save(payload, tmp, **opts)
def default_underflow_action() -> str:
    raw = (
        str(
            env_first(
                ("STNET_DATA_UNDERFLOW_ACTION", "STNET_UNDERFLOW_ACTION"),
                default="warn",
            )
            or "warn"
        )
        .strip()
        .lower()
    )
    return raw if raw in _DEF_UNDERFLOW_ACTIONS else "warn"
def normalize_underflow_action(
    value: object, *args: Any, default: str = "warn"
) -> str:
    r = str(value if value is not None else default).strip().lower()
    if r in _DEF_UNDERFLOW_ACTIONS:
        return r
    d = str(default).strip().lower()
    return d if d in _DEF_UNDERFLOW_ACTIONS else "warn"
