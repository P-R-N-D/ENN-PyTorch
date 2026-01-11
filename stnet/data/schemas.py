# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, TypeAlias

import torch
from tensordict import TensorDictBase

from ..core.datatypes import env_first


PathLike: TypeAlias = str | os.PathLike[str] | Path
JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]

_FEATURE_KEY_ALIASES = frozenset({"x", "feature", "features", "input", "inputs", "in"})

_LABEL_KEY_ALIASES = frozenset({"y", "label", "labels", "output", "outputs", "out"})

_DEF_UNDERFLOW_ACTIONS = {"allow", "warn", "forbid"}


def _td_set(td: TensorDictBase, key: str, value: Any) -> None:
    try:
        td.set(key, value)
    except Exception:
        td[key] = value


def _td_del(td: TensorDictBase, key: str) -> None:
    with contextlib.suppress(Exception):
        td.del_(key) if hasattr(td, "del_") else td.__delitem__(key)


def _resolve_key(data: Any, aliases: frozenset, name: str, required: bool) -> Optional[str]:
    if not isinstance(data, (Mapping, TensorDictBase)):
        raise TypeError(f"get_{name}_key expects Mapping/TensorDict")
    matches = [str(k) for k in data.keys() if isinstance(k, str) and k.casefold() in aliases]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(f"Expected exactly one {name} key among {sorted(aliases)}; found {matches}")
    if required:
        raise KeyError(
            f"Expected exactly one {name} key among {sorted(aliases)}; found {matches or 'none'}"
        )
    return None


def get_feature_key(data: Any) -> str:
    return _resolve_key(data, _FEATURE_KEY_ALIASES, "feature", True)


def get_label_key(data: Any, *args, required: bool = True) -> Optional[str]:
    return _resolve_key(data, _LABEL_KEY_ALIASES, "label", required)


def get_meta_path(mmt_path: str) -> str:
    return str(mmt_path) + ".meta.json"


def read_json(path: PathLike) -> JsonValue:
    with open(path, "r", encoding="utf-8-sig") as f:
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


@contextlib.contextmanager
def _atomic_swap(path: str):
    p = os.fspath(path)
    parent = os.path.dirname(p) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=os.path.basename(p) + ".", suffix=".tmp", dir=parent)
    os.close(fd)
    try:
        yield tmp_name
        os.replace(tmp_name, p)
    finally:
        with contextlib.suppress(Exception):
            os.remove(tmp_name)


def write_json(path: str, payload: Any, *args, indent: int | None = 2) -> None:
    with _atomic_swap(path) as tmp, open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent)


def save_temp(path: str, payload: Any, **opts) -> None:
    with _atomic_swap(path) as tmp:
        torch.save(payload, tmp, **opts)


def default_underflow_action() -> str:
    raw = (
        str(
            env_first(("STNET_DATA_UNDERFLOW_ACTION", "STNET_UNDERFLOW_ACTION"), default="warn")
            or "warn"
        )
        .strip()
        .lower()
    )
    return raw if raw in _DEF_UNDERFLOW_ACTIONS else "warn"


def normalize_underflow_action(value: object, *args, default: str = "warn") -> str:
    if (
        r := str(value if value is not None else default).strip().lower()
    ) in _DEF_UNDERFLOW_ACTIONS:
        return r
    return d if (d := str(default).strip().lower()) in _DEF_UNDERFLOW_ACTIONS else "warn"


def canonicalize_keys_(
    td: TensorDictBase,
    *args: Any,
    x_key: str = "X",
    y_key: str = "Y",
    allow_missing_labels: bool = False,
) -> TensorDictBase:
    if not isinstance(td, TensorDictBase):
        raise TypeError("canonicalize_xy_keys_ expects a TensorDict")
    fkey = get_feature_key(td)
    if fkey != x_key:
        _td_set(td, x_key, td[fkey])
        _td_del(td, fkey)
    lkey = get_label_key(td, required=not bool(allow_missing_labels))
    if lkey is not None and lkey != y_key:
        _td_set(td, y_key, td[lkey])
        _td_del(td, lkey)
    return td


def get_row(
    data: Any, *args, labels_required: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    return data[get_feature_key(data)], (
        data[l] if (l := get_label_key(data, required=labels_required)) else None
    )
