# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from contextlib import suppress
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
from tensordict import TensorDictBase

from .datatype import to_tuple, to_torch_tensor


def _assert_finites(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if torch.is_floating_point(tensor) or torch.is_complex(tensor):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} tensor contains non-finite values")
    return tensor


def _preprocess_x(x_tuple: Any) -> torch.Tensor:
    try:
        values = [float(v) for v in to_tuple(x_tuple)]
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "preprocess: feature tuples must contain only numeric values. "
            f"Invalid value={x_tuple!r}"
        ) from exc
    for value in values:
        try:
            if not math.isfinite(value):
                raise ValueError("preprocess: feature tuples must be finite")
        except TypeError as exc:
            raise TypeError(
                "preprocess: feature tuples must contain only numeric finite values. "
                f"Invalid value={value!r}"
            ) from exc
    tensor = torch.as_tensor(values, dtype=torch.float64)
    return _assert_finites(tensor, "feature")


def _preprocess_batch(
    x_value: Any, y_value: Any
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple], Tuple[int, ...]] | None:
    if isinstance(x_value, torch.Tensor):
        feature_tensor: Any = x_value
    else:
        feature_tensor = None
        for attr in ("to_torch_tensor", "to_torch", "to_tensor", "as_tensor"):
            if not hasattr(x_value, attr):
                continue
            with suppress(Exception):
                candidate = getattr(x_value, attr)()
            if isinstance(candidate, torch.Tensor):
                feature_tensor = candidate
                break
        if not isinstance(feature_tensor, torch.Tensor):
            with suppress(Exception):
                feature_tensor = torch.as_tensor(x_value)
    if not isinstance(feature_tensor, torch.Tensor):
        return None

    try:
        label_tensor = to_torch_tensor(y_value)
    except Exception:
        return None
    if not isinstance(label_tensor, torch.Tensor):
        with suppress(Exception):
            label_tensor = torch.as_tensor(y_value)
    if not isinstance(label_tensor, torch.Tensor):
        return None

    feature_tensor = _assert_finites(
        feature_tensor.detach().to(dtype=torch.float64), "feature"
    )
    if feature_tensor.dim() == 0:
        feature_tensor = feature_tensor.reshape(1, 1)
    elif feature_tensor.dim() == 1:
        feature_tensor = feature_tensor.reshape(-1, 1)
    else:
        batch_dim = int(feature_tensor.shape[0]) if feature_tensor.shape else 1
        feature_tensor = feature_tensor.reshape(batch_dim, -1)
    batch_size = int(feature_tensor.shape[0])
    label_tensor = _assert_finites(label_tensor.detach(), "label")
    if label_tensor.dim() == 0:
        label_tensor = label_tensor.unsqueeze(0)
    if label_tensor.dim() == 1 and label_tensor.shape[0] == batch_size:
        label_tensor = label_tensor.unsqueeze(-1)
    if label_tensor.shape[0] != batch_size:
        return None
    label_shape = tuple(label_tensor.shape[1:])
    batch_keys = [(int(index),) for index in range(batch_size)]
    return (feature_tensor, label_tensor, batch_keys, label_shape)


def _preprocess_y(value: Any) -> torch.Tensor:
    try:
        tensor = to_torch_tensor(value)
    except Exception:
        tensor = None
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(value)
    t = tensor.detach()
    if t.is_floating_point() or t.is_complex():
        t = t.to(dtype=torch.float64)
    else:
        t = t.to(dtype=torch.int64)
    return t


def preprocess(
    data: Union[Dict[Tuple, torch.Tensor], TensorDictBase],
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple], Tuple[int, ...]]:
    if isinstance(data, TensorDictBase):
        if "features" not in data.keys():
            raise ValueError("preprocess(TensorDict): missing 'features'")
        feats = torch.as_tensor(data.get("features"))
        if feats.ndim == 1:
            feats = feats.unsqueeze(0)
        if "labels" in data.keys():
            labels = torch.as_tensor(data.get("labels"))
        elif "labels_flat" in data.keys():
            labels = torch.as_tensor(data.get("labels_flat"))
        else:
            raise ValueError(
                "preprocess(TensorDict): missing 'labels' or 'labels_flat'"
            )
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        if labels.shape[0] != feats.shape[0]:
            raise ValueError("preprocess(TensorDict): features and labels batch mismatch")
        label_shape = tuple(labels.shape[1:]) if labels.dim() > 1 else (1,)
        keys = [(int(i),) for i in range(int(feats.shape[0]))]
        return (feats, labels, keys, label_shape)
    if isinstance(data, dict) and "X" in data and ("Y" in data):
        x, y = (data["X"], data["Y"])
        batch_result = _preprocess_batch(x, y)
        if batch_result is not None:
            return batch_result
        xr, yt = (
            _preprocess_x(x).unsqueeze(0),
            _assert_finites(_preprocess_y(y), "label"),
        )
        if yt.dim() == 0 or yt.dim() == 1:
            yt = yt.unsqueeze(0)
        keys = [to_tuple(x)]
        label_shape = tuple(yt.shape[1:])
        return (xr, yt, keys, label_shape)
    elif isinstance(data, (tuple, list)) and len(data) >= 2:
        x, y = (data[0], data[1])
        batch_result = _preprocess_batch(x, y)
        if batch_result is not None:
            return batch_result
        xr = _preprocess_x(x).unsqueeze(0)
        yt = _assert_finites(_preprocess_y(y), "label")
        if yt.dim() == 0:
            yt = yt.unsqueeze(0)
        elif yt.shape[0] != 1:
            yt = yt.unsqueeze(0)
        keys = [to_tuple(x)]
        label_shape = tuple(yt.shape[1:])
        return (xr, yt, keys, label_shape)
    elif isinstance(data, dict) and len(data) > 0:
        items = list(data.items())
        if any((isinstance(k, str) for k, _ in items)):
            raise TypeError(
                "preprocess: keys in a multi-sample dict must be tuples. "
                "Provide single samples as {'X': ..., 'Y': ...}."
            )
        keys: List[Tuple] = [to_tuple(k) for k, _ in items]
        feats = torch.stack([_preprocess_x(k) for k in keys], dim=0)
        lbl_list = [_assert_finites(_preprocess_y(v), "label") for _, v in items]
        if all((t.shape == lbl_list[0].shape for t in lbl_list)):
            labels = torch.stack(lbl_list, dim=0)
        else:
            labels = torch.cat([t.unsqueeze(0) for t in lbl_list], dim=0)
        labels = _assert_finites(labels, "label")
        label_shape = tuple(labels.shape[1:])
        return (feats, labels, keys, label_shape)
    else:
        raise ValueError(
            "preprocess: unsupported input format. Provide a dict or an (X, Y) pair."
        )


def postprocess(
    keys: List[Tuple], preds: torch.Tensor | Sequence[torch.Tensor]
) -> Dict[Tuple, torch.Tensor]:
    if isinstance(preds, torch.Tensor):
        if preds.dim() == 0:
            preds = preds.unsqueeze(0)
        if preds.shape[0] != len(keys):
            raise ValueError(f"preds batch={preds.shape[0]} != len(keys)={len(keys)}")
        rows = [preds[i].detach().cpu() for i in range(len(keys))]
    else:
        if len(preds) != len(keys):
            raise ValueError(f"len(preds)={len(preds)} != len(keys)={len(keys)}")
        rows = [
            p.detach().cpu() if isinstance(p, torch.Tensor) else torch.as_tensor(p)
            for p in preds
        ]
    fixed_keys: List[Tuple] = []
    seen = set()
    for i, k in enumerate(keys):
        if not isinstance(k, tuple):
            try:
                k = tuple(k)
            except TypeError:
                k = (k,)
        k_out = k
        if k in seen:
            k_out = k + (i,)
        seen.add(k_out)
        fixed_keys.append(k_out)
    return {k: v for k, v in zip(fixed_keys, rows)}
