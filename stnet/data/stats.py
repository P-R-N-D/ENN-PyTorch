# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Dict, Generic, MutableMapping, Optional, Tuple, TypeVar

import math

import numpy as np
import torch

from ..api.utils import (
    cuda_compute_capability,
    is_cpu_bf16_supported,
    is_int8_supported,
)


TExtra = TypeVar("TExtra")


@dataclass
class MetaData(Generic[TExtra]):
    """Container for runtime metadata shared across training and inference.

    The class tracks device capabilities, preferred casting dtypes and cached
    normalization statistics so that components such as :class:`AutoCast`
    or post-processing utilities can reuse a single source of truth. Device
    information, including the device type and CUDA compute capability where
    available, is recorded for downstream consumers.
    """

    device: torch.device
    device_type: str = field(init=False, default="cpu")
    cuda_cc: Optional[Tuple[int, int]] = field(init=False, default=None)
    float_dtypes: Tuple[torch.dtype, ...] = field(default_factory=tuple)
    int_dtypes: Tuple[torch.dtype, ...] = field(default_factory=tuple)
    float8_dtypes: Tuple[torch.dtype, ...] = field(default_factory=tuple)
    scale_max_abs: Optional[float] = None
    scale_min_abs: Optional[float] = None
    scale_is_integral: Optional[bool] = None
    stats: MutableMapping[str, torch.Tensor] = field(default_factory=dict)
    extra: Dict[str, TExtra] = field(default_factory=dict)
    def __post_init__(self) -> None:
        self._refresh_device_info()

    def _refresh_device_info(self) -> None:
        dev = torch.device(self.device)
        self.device = dev
        self.device_type = dev.type
        if dev.type == "cuda":
            major, minor = cuda_compute_capability(dev)
            if major > 0 or minor > 0:
                self.cuda_cc = (major, minor)
            else:
                self.cuda_cc = None
        else:
            self.cuda_cc = None

    @property
    def cuda_cc_str(self) -> Optional[str]:
        if self.cuda_cc is None:
            return None
        major, minor = self.cuda_cc
        if major <= 0 and minor <= 0:
            return None
        return f"sm_{major}{minor}"

    @property
    def has_scale(self) -> bool:
        return self.scale_max_abs is not None

    @property
    def scale_min_positive(self) -> Optional[float]:
        if self.scale_min_abs is None or not np.isfinite(self.scale_min_abs):
            return None
        if self.scale_min_abs <= 0.0:
            return None
        return float(self.scale_min_abs)

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
        scale_max_abs: Optional[float] = None,
        scale_min_abs: Optional[float] = None,
        scale_is_integral: Optional[bool] = None,
        extra: Optional[Mapping[str, TExtra]] = None,
    ) -> "MetaData[TExtra]":
        dev = torch.device(device)
        meta = cls(
            device=dev,
            float_dtypes=cls._float_amp_candidates(dev),
            int_dtypes=cls._integer_candidates(dev),
            float8_dtypes=cls._float8_dtypes(),
            scale_max_abs=scale_max_abs,
            scale_min_abs=scale_min_abs,
            scale_is_integral=scale_is_integral,
        )
        meta.ensure_device_info()
        if extra:
            meta.extra.update(dict(extra))
        return meta

    def refresh(self) -> None:
        """Recompute dtype capabilities for the current device."""

        dev = torch.device(self.device)
        self._refresh_device_info()
        self.float_dtypes = self._float_amp_candidates(dev)
        self.int_dtypes = self._integer_candidates(dev)
        self.float8_dtypes = self._float8_dtypes()

    def ensure_device_info(self) -> None:
        self._refresh_device_info()

    def clear_scale(self) -> None:
        self.scale_max_abs = None
        self.scale_min_abs = None
        self.scale_is_integral = None

    def update_scale(
        self,
        *,
        max_abs: Optional[float] = None,
        min_abs: Optional[float] = None,
        is_integral: Optional[bool] = None,
    ) -> None:
        if max_abs is None and min_abs is None and is_integral is None:
            self.clear_scale()
            return
        self.scale_max_abs = float(max_abs) if max_abs is not None else None
        self.scale_min_abs = float(min_abs) if min_abs is not None else None
        self.scale_is_integral = bool(is_integral) if is_integral is not None else None

    @staticmethod
    def _scale_from_tensor(
        tensor: Optional[torch.Tensor],
    ) -> Optional[Tuple[float, Optional[float], bool]]:
        if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return None
        with torch.no_grad():
            values = tensor.detach()
            finite_mask = torch.isfinite(values)
            if not bool(finite_mask.any().item()):
                return None
            finite_vals = values[finite_mask]
            abs_vals = finite_vals.abs()
            max_abs = float(abs_vals.max().item()) if abs_vals.numel() else 0.0
            pos_vals = abs_vals[abs_vals > 0]
            min_abs = float(pos_vals.min().item()) if pos_vals.numel() else None
            if finite_vals.is_floating_point():
                tol = 1e-6 if finite_vals.dtype in (torch.float32, torch.float64) else 5e-4
                frac = (finite_vals - torch.round(finite_vals)).abs()
                is_integral = bool(frac.lt(tol).all().item()) if frac.numel() else True
            else:
                is_integral = True
        return (max_abs, min_abs, is_integral)

    def accumulate_scale(self, tensor: Optional[torch.Tensor]) -> None:
        summary = self._scale_from_tensor(tensor)
        if summary is None:
            return
        max_abs, min_abs, is_integral = summary
        if not math.isfinite(max_abs):
            return
        max_abs = float(max_abs)
        self.scale_max_abs = (
            max_abs
            if self.scale_max_abs is None
            else float(max(max_abs, float(self.scale_max_abs)))
        )
        if min_abs is not None:
            min_abs = float(min_abs)
            if self.scale_min_abs is None:
                self.scale_min_abs = min_abs
            else:
                self.scale_min_abs = float(min(float(self.scale_min_abs), min_abs))
        if self.scale_is_integral is None:
            self.scale_is_integral = bool(is_integral)
        else:
            self.scale_is_integral = bool(self.scale_is_integral and is_integral)

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

    from ..backend.fx import inference  # local import to avoid cycles

    try:
        ref = next(model.parameters())
    except StopIteration:
        ref = None
    dev = ref.device if isinstance(ref, torch.Tensor) else torch.device("cpu")

    if metadata is not None:
        metadata.device = dev
        metadata.refresh()

    eps = 1e-6
    if hasattr(model, "y_eps"):
        with contextlib.suppress(Exception):
            y_eps = getattr(model, "y_eps")
            if isinstance(y_eps, torch.Tensor):
                eps = float(y_eps.item())
            else:
                eps = float(y_eps)

    y_min: torch.Tensor | None = None
    y_max: torch.Tensor | None = None
    y_sum: torch.Tensor | None = None
    y_sum2: torch.Tensor | None = None
    y_count: torch.Tensor | None = None

    model.eval()
    with inference(model):
        for X, Y in loader:
            if Y is None:
                continue
            y = torch.as_tensor(Y).detach()
            if y.ndim > 2:
                y = y.view(y.shape[0], -1)
            elif y.ndim == 1:
                y = y.view(-1, 1)
            if y_min is None:
                D = int(y.shape[1])
                device = torch.device("cpu")
                y_min = torch.full((D,), float("inf"), dtype=torch.float64, device=device)
                y_max = torch.full((D,), float("-inf"), dtype=torch.float64, device=device)
                y_sum = torch.zeros(D, dtype=torch.float64, device=device)
                y_sum2 = torch.zeros(D, dtype=torch.float64, device=device)
                y_count = torch.zeros(D, dtype=torch.float64, device=device)
            y64 = y.to(dtype=torch.float64, device="cpu")
            finite = torch.isfinite(y64)
            if not finite.any():
                continue
            assert y_min is not None and y_max is not None
            assert y_sum is not None and y_sum2 is not None and y_count is not None
            valid_values = torch.where(finite, y64, torch.zeros_like(y64))
            y_sum.add_(valid_values.sum(dim=0))
            y_sum2.add_((valid_values * valid_values).sum(dim=0))
            y_count.add_(finite.sum(dim=0, dtype=torch.float64))
            min_candidates = torch.where(
                finite, y64, torch.full_like(y64, float("inf"))
            )
            max_candidates = torch.where(
                finite, y64, torch.full_like(y64, float("-inf"))
            )
            y_min.copy_(torch.minimum(y_min, min_candidates.min(dim=0).values))
            y_max.copy_(torch.maximum(y_max, max_candidates.max(dim=0).values))

    if y_min is None or y_count is None:
        raise ValueError("no labels to compute y-stats")

    assert y_max is not None and y_sum is not None and y_sum2 is not None
    valid_mask = y_count > 0
    y_mean = torch.zeros_like(y_sum)
    var = torch.zeros_like(y_sum2)
    y_mean[valid_mask] = y_sum[valid_mask] / y_count[valid_mask]
    var[valid_mask] = (
        y_sum2[valid_mask] / y_count[valid_mask]
        - y_mean[valid_mask].pow(2)
    )
    y_std = torch.sqrt(var.clamp_min(eps**2))

    ready = bool(valid_mask.any().item())

    def _copy_to_model(name: str, value: torch.Tensor) -> None:
        attr = getattr(model, name, None)
        if isinstance(attr, torch.Tensor):
            with contextlib.suppress(Exception):
                attr.data.copy_(value.to(device=attr.device, dtype=attr.dtype))

    _copy_to_model("y_min", y_min.to(dtype=torch.float64))
    _copy_to_model("y_max", y_max.to(dtype=torch.float64))
    _copy_to_model("y_sum", y_sum)
    _copy_to_model("y_sum2", y_sum2)
    _copy_to_model("y_count", y_count)
    _copy_to_model("y_mean", y_mean.to(dtype=torch.float64))
    _copy_to_model("y_std", y_std.to(dtype=torch.float64))

    attr_ready = getattr(model, "y_stats_ready", None)
    if isinstance(attr_ready, torch.Tensor):
        with contextlib.suppress(Exception):
            attr_ready.fill_(ready)

    if metadata is not None:
        metadata.set_stat("y_min", y_min.to(dtype=torch.float64))
        metadata.set_stat("y_max", y_max.to(dtype=torch.float64))
        metadata.set_stat("y_sum", y_sum)
        metadata.set_stat("y_sum2", y_sum2)
        metadata.set_stat("y_count", y_count)
        metadata.set_stat("y_mean", y_mean.to(dtype=torch.float64))
        metadata.set_stat("y_std", y_std.to(dtype=torch.float64))


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

