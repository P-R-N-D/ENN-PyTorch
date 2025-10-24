# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch

from ..utils.optimization import inference


def compute_y_range(
    loader: Iterable[Any],
    q_low: float = 0.005,
    q_high: float = 0.995,
    *,
    max_batches: int | None = None,
    labels_key: str = "labels",
) -> tuple[float, float]:
    ys: list[np.ndarray] = []
    for i, batch in enumerate(loader):
        if isinstance(batch, dict):
            y = batch.get(labels_key)
        elif isinstance(batch, Sequence) and len(batch) > 0:
            y = batch[-1]
        else:
            raise TypeError("Unsupported batch structure for compute_y_range")
        if y is None:
            raise ValueError(
                "compute_y_range could not locate labels in the provided batch"
            )
        if isinstance(y, torch.Tensor):
            y_arr = y.detach().cpu().numpy()
        else:
            y_arr = np.asarray(y)
        ys.append(np.ravel(y_arr))
        if max_batches is not None and i >= max_batches:
            break
    if not ys:
        raise ValueError("compute_y_range received no batches")
    y_all = np.concatenate(ys, axis=0)
    lo, hi = np.quantile(y_all, [q_low, q_high])
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise ValueError(f"invalid y-range: lo={lo}, hi={hi}")
    return float(lo), float(hi)


def recompute_y_stats(model: torch.nn.Module, loader: Iterable[Any]) -> None:
    """Recompute label statistics for *model* using *loader*."""

    dev = next(model.parameters()).device
    model.y_min.fill_(float("inf"))
    model.y_max.fill_(float("-inf"))
    model.y_sum.zero_()
    model.y_sum2.zero_()
    model.y_count.zero_()
    model.y_stats_ready.fill_(False)
    model.eval()
    with inference(model):
        for X, Y in loader:
            if Y is None:
                continue
            model.update_y_stats(Y.to(dev))
        model.finalize_y_stats()


def inverse_y_from_stats(model: torch.nn.Module, y_flat: torch.Tensor) -> torch.Tensor:
    """Undo standardization using the model's stored label statistics."""

    has_stats = getattr(model, "has_valid_y_stats", None)
    if callable(has_stats) and not has_stats():
        return y_flat

    mean = getattr(model, "y_mean", None)
    std = getattr(model, "y_std", None)
    if mean is None or std is None:
        return y_flat

    device = y_flat.device
    dtype = y_flat.dtype
    eps = 1e-6
    if hasattr(model, "y_eps"):
        with contextlib.suppress(Exception):
            eps = float(model.y_eps.item())
    mu = mean.detach().to(device=device, dtype=dtype)
    sigma = std.detach().to(device=device, dtype=dtype).clamp_min(eps)
    return y_flat * sigma + mu

