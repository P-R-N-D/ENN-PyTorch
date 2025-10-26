# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Dict, Generic, MutableMapping, Optional, Tuple, TypeVar, TYPE_CHECKING

import numpy as np
import torch

from ..utils.platform import is_cpu_bf16_supported, is_int8_supported

if TYPE_CHECKING:  # pragma: no cover - type-checking helper
    from ..utils.optimization import DataScale


TExtra = TypeVar("TExtra")


@dataclass
class MetaData(Generic[TExtra]):
    """Container for runtime metadata shared across training and inference.

    The class tracks device capabilities, preferred casting dtypes and cached
    normalization statistics so that components such as :class:`AutoCast`
    or post-processing utilities can reuse a single source of truth.
    """

    device: torch.device
    float_dtypes: Tuple[torch.dtype, ...] = field(default_factory=tuple)
    int_dtypes: Tuple[torch.dtype, ...] = field(default_factory=tuple)
    float8_dtypes: Tuple[torch.dtype, ...] = field(default_factory=tuple)
    scale: Optional["DataScale"] = None
    stats: MutableMapping[str, torch.Tensor] = field(default_factory=dict)
    extra: Dict[str, TExtra] = field(default_factory=dict)
    last_float: Optional[torch.dtype] = None
    last_int: Optional[torch.dtype] = None

    @staticmethod
    def _distinct_dtypes(candidates: Sequence[torch.dtype]) -> Tuple[torch.dtype, ...]:
        seen: Dict[torch.dtype, None] = {}
        for dtype in candidates:
            if isinstance(dtype, torch.dtype) and dtype not in seen:
                seen[dtype] = None
        return tuple(seen.keys())

    @staticmethod
    def _float8_dtypes() -> Tuple[torch.dtype, ...]:
        names = (
            "float8_e4m3fn",
            "float8_e4m3fnuz",
            "float8_e5m2",
            "float8_e5m2fnuz",
        )
        values: list[torch.dtype] = []
        for name in names:
            candidate = getattr(torch, name, None)
            if isinstance(candidate, torch.dtype):
                values.append(candidate)
        return tuple(values)

    @classmethod
    def _float_amp_candidates(cls, device: torch.device) -> Tuple[torch.dtype, ...]:
        dev_type = device.type
        candidates: list[torch.dtype] = []
        if dev_type == "cuda":
            if torch.cuda.is_bf16_supported():
                candidates.append(torch.bfloat16)
            candidates.append(torch.float16)
            candidates.append(torch.float32)
        elif dev_type == "xpu":
            candidates.extend((torch.bfloat16, torch.float32))
        elif dev_type == "mps":
            candidates.extend((torch.float16, torch.float32))
        elif dev_type == "cpu":
            if is_cpu_bf16_supported():
                candidates.append(torch.bfloat16)
            candidates.extend((torch.float32, torch.float64))
        else:
            candidates.append(torch.float32)
        if not candidates:
            candidates.append(torch.float32)
        return cls._distinct_dtypes(candidates)

    @classmethod
    def _integer_candidates(cls, device: torch.device) -> Tuple[torch.dtype, ...]:
        candidates: list[torch.dtype] = []
        int8_ok, _ = is_int8_supported(device)
        if int8_ok:
            candidates.append(torch.int8)
        candidates.extend((torch.int16, torch.int32, torch.int64))
        return cls._distinct_dtypes(candidates or (torch.int64,))

    @classmethod
    def for_device(
        cls,
        device: torch.device | str,
        *,
        scale: Optional["DataScale"] = None,
        extra: Optional[Mapping[str, TExtra]] = None,
    ) -> "MetaData[TExtra]":
        dev = torch.device(device)
        meta = cls(
            device=dev,
            float_dtypes=cls._float_amp_candidates(dev),
            int_dtypes=cls._integer_candidates(dev),
            float8_dtypes=cls._float8_dtypes(),
            scale=scale,
        )
        if extra:
            meta.extra.update(dict(extra))
        return meta

    def refresh(self) -> None:
        """Recompute dtype capabilities for the current device."""

        dev = torch.device(self.device)
        self.float_dtypes = self._float_amp_candidates(dev)
        self.int_dtypes = self._integer_candidates(dev)
        self.float8_dtypes = self._float8_dtypes()

    def update_scale(self, scale: Optional["DataScale"]) -> None:
        self.scale = scale

    def record_float_dtype(self, dtype: Optional[torch.dtype]) -> None:
        if isinstance(dtype, torch.dtype):
            self.last_float = dtype

    def record_int_dtype(self, dtype: Optional[torch.dtype]) -> None:
        if isinstance(dtype, torch.dtype):
            self.last_int = dtype

    def set_stat(self, name: str, value: torch.Tensor) -> None:
        if isinstance(value, torch.Tensor):
            self.stats[name] = value.detach().clone()

    def get_stat(self, name: str) -> Optional[torch.Tensor]:
        value = self.stats.get(name)
        return value.detach().clone() if isinstance(value, torch.Tensor) else None


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


def recompute_y_stats(
    model: torch.nn.Module,
    loader: Iterable[Any],
    *,
    metadata: MetaData[Any] | None = None,
) -> None:
    """Recompute label statistics for *model* using *loader*.

    If *metadata* is supplied the freshly computed statistics and device
    capabilities are cached on the provided :class:`MetaData` instance so that
    subsequent inference stages can reuse them without touching the model.
    """

    dev = next(model.parameters()).device
    if metadata is not None:
        metadata.device = dev
        metadata.refresh()

    model.y_min.fill_(float("inf"))
    model.y_max.fill_(float("-inf"))
    model.y_sum.zero_()
    model.y_sum2.zero_()
    model.y_count.zero_()
    model.y_stats_ready.fill_(False)
    model.eval()
    from ..utils.optimization import inference  # local import to avoid cycles

    with inference(model):
        for X, Y in loader:
            if Y is None:
                continue
            model.update_y_stats(Y.to(dev))
        model.finalize_y_stats()

    if metadata is not None:
        for attr in (
            "y_min",
            "y_max",
            "y_sum",
            "y_sum2",
            "y_count",
            "y_mean",
            "y_std",
        ):
            value = getattr(model, attr, None)
            if isinstance(value, torch.Tensor):
                metadata.set_stat(attr, value)


def inverse_y_from_stats(
    model: torch.nn.Module,
    y_flat: torch.Tensor,
    *,
    metadata: MetaData[Any] | None = None,
) -> torch.Tensor:
    """Undo standardization using the stored label statistics."""

    has_stats = getattr(model, "has_valid_y_stats", None)
    if callable(has_stats) and not has_stats():
        return y_flat

    if metadata is not None:
        mean = metadata.get_stat("y_mean")
        std = metadata.get_stat("y_std")
    else:
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

