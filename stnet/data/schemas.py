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

from ..core.casting import env_first


PathLike: TypeAlias = str | os.PathLike[str] | Path
JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]

_FEATURE_KEY_ALIASES = frozenset({"x", "feature", "features", "input", "inputs", "in"})

_LABEL_KEY_ALIASES = frozenset({"y", "label", "labels", "output", "outputs", "out"})

_DEF_UNDERFLOW_ACTIONS = {"allow", "warn", "forbid"}


def _casefold_str(x: Any) -> Optional[str]:
    if isinstance(x, str):
        return x.casefold()
    return None


def _td_set(td: TensorDictBase, key: str, value: Any) -> None:
    try:
        td.set(key, value)
    except Exception:
        td[key] = value


def _td_del(td: TensorDictBase, key: str) -> None:
    try:
        td.del_(key)
    except Exception:
        with contextlib.suppress(Exception):
            del td[key]


def get_feature_key(data: Any) -> str:
    if not isinstance(data, (Mapping, TensorDictBase)):
        raise TypeError("resolve_feature_key expects a Mapping or TensorDict")
    matches: list[str] = []
    for k in data.keys():
        ck = _casefold_str(k)
        if ck is not None and ck in _FEATURE_KEY_ALIASES:
            matches.append(str(k))
    if len(matches) != 1:
        raise KeyError(
            f"Expected exactly one feature key among {sorted(_FEATURE_KEY_ALIASES)}; "
            f"found {matches or 'none'}"
        )
    return matches[0]


def get_label_key(data: Any, *args: Any, required: bool = True) -> Optional[str]:
    if not isinstance(data, (Mapping, TensorDictBase)):
        raise TypeError("resolve_label_key expects a Mapping or TensorDict")
    matches: list[str] = []
    for k in data.keys():
        ck = _casefold_str(k)
        if ck is not None and ck in _LABEL_KEY_ALIASES:
            matches.append(str(k))
    if len(matches) == 0:
        if required:
            raise KeyError(f"Expected one label key among {sorted(_LABEL_KEY_ALIASES)}; found none")
        return None
    if len(matches) != 1:
        raise KeyError(
            f"Expected exactly one label key among {sorted(_LABEL_KEY_ALIASES)}; found {matches}"
        )
    return matches[0]


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
            "shape": [int(x) for x in obj.shape],
            "dtype": str(obj.dtype),
            "device": str(obj.device),
        }
    if isinstance(obj, dict):
        return {str(k): coerce_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [coerce_json(v) for v in obj]
    return str(obj)


def write_json(path: str, payload: Any, *args: Any, indent: int | None = 2) -> None:
    p = os.fspath(path)
    parent = os.path.dirname(p) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=os.path.basename(p) + ".", suffix=".tmp", dir=parent)
    os.close(fd)
    try:
        with open(tmp_name, "w", encoding="utf-8") as f:
            if indent is None:
                json.dump(payload, f)
            else:
                json.dump(payload, f, indent=int(indent))
        os.replace(tmp_name, p)
    finally:
        with contextlib.suppress(Exception):
            os.remove(tmp_name)


def save_temp(path: str, payload: Any, **opts: Any) -> None:
    p = os.fspath(path)
    parent = os.path.dirname(p) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=os.path.basename(p) + ".", suffix=".tmp", dir=parent)
    os.close(fd)
    try:
        torch.save(payload, tmp_name, **opts)
        os.replace(tmp_name, p)
    finally:
        with contextlib.suppress(Exception):
            os.remove(tmp_name)


def default_underflow_action() -> str:
    raw = str(
        env_first(("STNET_DATA_UNDERFLOW_ACTION", "STNET_UNDERFLOW_ACTION"), default="warn") or "warn"
    ).strip().lower()
    return raw if raw in _DEF_UNDERFLOW_ACTIONS else "warn"


def normalize_underflow_action(value: object, *args: Any, default: str = "warn") -> str:
    raw = str(value if value is not None else default).strip().lower()
    if raw in _DEF_UNDERFLOW_ACTIONS:
        return raw
    fallback = str(default).strip().lower()
    return fallback if fallback in _DEF_UNDERFLOW_ACTIONS else "warn"


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
    data: Any,
    *args: Any,
    labels_required: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    fkey = get_feature_key(data)
    lkey = get_label_key(data, required=labels_required)
    x = data[fkey]
    y = data[lkey] if (lkey is not None) else None
    return x, y
