# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import tempfile
import contextlib
from typing import Any, Mapping, Optional, Tuple

import torch

try:
    from tensordict import TensorDictBase
except Exception:  # pragma: no cover
    class TensorDictBase:  # type: ignore[no-redef]
        pass


# -----------------------------------------------------------------------------
# Feature/label key resolution (TensorDict-first contract)
# -----------------------------------------------------------------------------

# Keys are matched case-insensitively (via casefold).
_FEATURE_KEY_ALIASES = frozenset({"x", "feature", "features", "input", "inputs", "in"})
_LABEL_KEY_ALIASES = frozenset({"y", "label", "labels", "output", "outputs", "out"})


def _casefold_str(x: Any) -> Optional[str]:
    if isinstance(x, str):
        return x.casefold()
    return None


def resolve_feature_key(data: Any) -> str:
    """Resolve the (unique) feature key in a mapping/TensorDict.

    Accepted aliases (case-insensitive):
      x, feature, features, input, inputs, in
    """
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


def resolve_label_key(data: Any, *, required: bool = True) -> Optional[str]:
    """Resolve the (unique) label key in a mapping/TensorDict.

    Accepted aliases (case-insensitive):
      y, label, labels, output, outputs, out

    If required=False, returns None when no label key exists.
    """
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


def extract_xy(
    data: Any,
    *,
    labels_required: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Extract (X, Y) tensors from a mapping/TensorDict using alias resolution."""
    fkey = resolve_feature_key(data)
    lkey = resolve_label_key(data, required=labels_required)
    x = data[fkey]
    y = data[lkey] if (lkey is not None) else None
    return x, y


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


def canonicalize_xy_keys_(
    td: TensorDictBase,
    *,
    x_key: str = "X",
    y_key: str = "Y",
    allow_missing_labels: bool = False,
) -> TensorDictBase:
    """Rename feature/label keys in-place to a canonical (X, Y) convention."""
    if not isinstance(td, TensorDictBase):
        raise TypeError("canonicalize_xy_keys_ expects a TensorDict")

    fkey = resolve_feature_key(td)
    if fkey != x_key:
        _td_set(td, x_key, td[fkey])
        _td_del(td, fkey)

    lkey = resolve_label_key(td, required=not bool(allow_missing_labels))
    if lkey is not None and lkey != y_key:
        _td_set(td, y_key, td[lkey])
        _td_del(td, lkey)

    return td


# -----------------------------------------------------------------------------
# Memmap / metadata atomic I/O utilities
# -----------------------------------------------------------------------------

def mmt_meta_path(mmt_path: str) -> str:
    """Sidecar metadata path for a MemoryMappedTensor file."""
    return str(mmt_path) + ".meta.json"


def atomic_write_json(path: str, payload: Any, *, indent: int | None = 2) -> None:
    """Atomically write JSON (best-effort).

    Used for small sidecar metadata files that must not be left half-written
    under multi-process or abrupt-interrupt scenarios.
    """
    import json as _json

    p = os.fspath(path)
    parent = os.path.dirname(p) or "."
    os.makedirs(parent, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix=os.path.basename(p) + ".", suffix=".tmp", dir=parent)
    os.close(fd)
    try:
        with open(tmp_name, "w", encoding="utf-8") as f:
            if indent is None:
                _json.dump(payload, f)
            else:
                _json.dump(payload, f, indent=int(indent))
        os.replace(tmp_name, p)
    finally:
        with contextlib.suppress(Exception):
            os.remove(tmp_name)


def atomic_torch_save(path: str, payload: Any, **opts: Any) -> None:
    """Atomically write a torch checkpoint file.

    Writes to a temporary file in the same directory and then replaces
    it into place. This avoids corrupting checkpoints if interrupted
    mid-write.
    """
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


# -----------------------------------------------------------------------------
# Underflow policy utilities (used by dtype negotiation / memmap writing)
# -----------------------------------------------------------------------------

_DEF_UNDERFLOW_ACTIONS = {"allow", "warn", "forbid"}


def default_underflow_action() -> str:
    """Return underflow policy used by precision negotiation.

    - allow: underflow (flush-to-zero / subnormals) is allowed
    - warn: allowed but may be logged
    - forbid: treat underflow as unsafe for downcasting
    """
    from ..core.casting import env_first

    raw = str(
        env_first(("STNET_DATA_UNDERFLOW_ACTION", "STNET_UNDERFLOW_ACTION"), default="warn") or "warn"
    ).strip().lower()
    return raw if raw in _DEF_UNDERFLOW_ACTIONS else "warn"


def normalize_underflow_action(value: object, *, default: str = "warn") -> str:
    raw = str(value if value is not None else default).strip().lower()
    if raw in _DEF_UNDERFLOW_ACTIONS:
        return raw
    fallback = str(default).strip().lower()
    return fallback if fallback in _DEF_UNDERFLOW_ACTIONS else "warn"
