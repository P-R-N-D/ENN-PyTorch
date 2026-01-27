# -*- coding: utf-8 -*-
from __future__ import annotations

from contextlib import suppress
from typing import Any, Mapping, Optional

import torch
from tensordict import TensorDictBase


_FEATURE_KEY_ALIASES = frozenset(
    {"x", "feature", "features", "input", "inputs", "in"}
)
_LABEL_KEY_ALIASES = frozenset(
    {"y", "label", "labels", "output", "outputs", "out"}
)


def _td_set(td: TensorDictBase, key: str, value: Any) -> None:
    try:
        td.set(key, value)
    except Exception:
        td[key] = value


def _td_del(td: TensorDictBase, key: str) -> None:
    with suppress(Exception):
        td.del_(key) if hasattr(td, "del_") else td.__delitem__(key)


def _resolve_key(
    data: Any, aliases: frozenset, name: str, required: bool
) -> Optional[str]:
    if not isinstance(data, (Mapping, TensorDictBase)):
        raise TypeError(f"get_{name}_key expects Mapping/TensorDict")
    matches = [
        str(k)
        for k in data.keys()
        if isinstance(k, str) and k.casefold() in aliases
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(
            f"Expected exactly one {name} key among {sorted(aliases)}; found {matches}"
        )
    if required:
        raise KeyError(
            f"Expected exactly one {name} key among {sorted(aliases)}; found {matches or 'none'}"
        )
    return None


def get_feature_key(data: Any) -> str:
    return _resolve_key(data, _FEATURE_KEY_ALIASES, "feature", True)


def get_label_key(
    data: Any, *args: Any, required: bool = True
) -> Optional[str]:
    del args
    return _resolve_key(data, _LABEL_KEY_ALIASES, "label", required)


def canonicalize_keys_(
    td: TensorDictBase,
    *args: Any,
    x_key: str = "X",
    y_key: str = "Y",
    allow_missing_labels: bool = False,
) -> TensorDictBase:
    del args
    if not isinstance(td, TensorDictBase):
        raise TypeError("canonicalize_keys_ expects a TensorDict")
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
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    del args
    return data[get_feature_key(data)], (
        data[l]
        if (l := get_label_key(data, required=labels_required))
        else None
    )


def is_feature_label_batch_mapping(obj: Any) -> bool:
    if not isinstance(obj, Mapping) or not obj:
        return False
    for k in obj.keys():
        if not isinstance(k, str):
            continue
        ck = k.casefold()
        if ck in _FEATURE_KEY_ALIASES or ck in _LABEL_KEY_ALIASES:
            return True
    return False
