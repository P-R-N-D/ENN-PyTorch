# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

from ..utils.optimization import inference


def compute_y_range(
    loader: Iterable[Any],
    q_low: float = 0.005,
    q_high: float = 0.995,
    *,
    labels_key: str = "Y",
    max_batches: int | None = None,
) -> tuple[float, float]:
    ys = []
    for i, batch in enumerate(loader):
        y = batch.get(labels_key) if isinstance(batch, dict) else batch[-1]
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        ys.append(np.asarray(y).ravel())
        if max_batches is not None and i >= max_batches:
            break
    y_all = np.concatenate(ys, axis=0) if ys else np.array([], dtype=float)
    if y_all.size == 0:
        raise ValueError("no labels to compute y-range")
    finite = y_all[np.isfinite(y_all)]
    if finite.size == 0:
        raise ValueError("no finite labels to compute y-range")
    lo, hi = np.quantile(finite, [q_low, q_high])
    if not (hi > lo):
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

