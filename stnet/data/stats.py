# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Dict, Generic, MutableMapping, Optional, Tuple, TypeVar

import math

import numpy as np
import torch
from tensordict import tensorclass

from ..backend.environment import cuda_compute_capability


TExtra = TypeVar("TExtra")


@dataclass
class Metadata(Generic[TExtra]):

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
        from ..functional.fx import Autocast as _Autocast

        return _Autocast.float8_formats()

    @classmethod
    def _float_amp_candidates(cls: object, device: torch.device) -> Tuple[torch.dtype, ...]:
        from ..functional.fx import Autocast as _Autocast

        return _Autocast.float_amp_priority(device)

    @classmethod
    def _integer_candidates(cls: object, device: torch.device) -> Tuple[torch.dtype, ...]:
        from ..functional.fx import Autocast as _Autocast

        return _Autocast.integer_amp_priority(device)

    @classmethod
    def for_device(
        cls: object,
        device: torch.device | str,
        *args: Any,
        scale_max_abs: Optional[float] = None,
        scale_min_abs: Optional[float] = None,
        scale_is_integral: Optional[bool] = None,
        extra: Optional[Mapping[str, TExtra]] = None,
        **kwargs: Any,
    ) -> "Metadata[TExtra]":
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
        *args: Any,
        max_abs: Optional[float] = None,
        min_abs: Optional[float] = None,
        is_integral: Optional[bool] = None,
        **kwargs: Any,
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


@tensorclass(shadow=True)
class TensorDictMetadata:

    use_amp: bool = False
    amp_dtype: Optional[Any] = None
    device: Optional[Any] = None

    def autocast(self):
        if self.use_amp and isinstance(self.amp_dtype, torch.dtype):
            if isinstance(self.device, torch.device):
                dev_type = self.device.type
            else:
                dev_type = "cuda" if torch.cuda.is_available() else "cpu"
            return torch.autocast(device_type=dev_type, dtype=self.amp_dtype)
        return contextlib.nullcontext()


