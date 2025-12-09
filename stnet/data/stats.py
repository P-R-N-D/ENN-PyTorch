# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, MutableMapping, Optional, Tuple, TypeVar

import torch
from tensordict import tensorclass

from ..backend.system import cuda_compute_capability

TExtra = TypeVar("TExtra")

_BOOTSTRAP_DEPTH = 0


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
            try:
                major, minor = cuda_compute_capability(dev)
            except Exception:
                major, minor = (0, 0)
            if major > 0 or minor > 0:
                self.cuda_cc = (int(major), int(minor))
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
        v = self.scale_min_abs
        if v is None:
            return None
        try:
            if not math.isfinite(float(v)) or float(v) <= 0.0:
                return None
        except Exception:
            return None
        return float(v)

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
    def _float_amp_candidates(
        cls: object, device: torch.device
    ) -> Tuple[torch.dtype, ...]:
        from ..functional.fx import Autocast as _Autocast

        global _BOOTSTRAP_DEPTH
        if _BOOTSTRAP_DEPTH:
            return (torch.float32,)
        _BOOTSTRAP_DEPTH += 1
        try:
            return _Autocast.float_amp_priority(device)
        except Exception:
            return (torch.float32,)
        finally:
            _BOOTSTRAP_DEPTH = max(0, _BOOTSTRAP_DEPTH - 1)

    @classmethod
    def _integer_candidates(
        cls: object, device: torch.device
    ) -> Tuple[torch.dtype, ...]:
        from ..functional.fx import Autocast as _Autocast

        global _BOOTSTRAP_DEPTH
        if _BOOTSTRAP_DEPTH:
            return (torch.int64,)
        _BOOTSTRAP_DEPTH += 1
        try:
            return _Autocast.integer_amp_priority(device)
        except Exception:
            return (torch.int64,)
        finally:
            _BOOTSTRAP_DEPTH = max(0, _BOOTSTRAP_DEPTH - 1)

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
        float_candidates = cls._float_amp_candidates(dev)
        int_candidates = cls._integer_candidates(dev)
        meta = cls(
            device=dev,
            float_dtypes=float_candidates,
            int_dtypes=int_candidates,
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
        if max_abs is not None:
            try:
                v = float(max_abs)
                if math.isfinite(v):
                    self.scale_max_abs = v
            except Exception:
                pass
        if min_abs is not None:
            try:
                v = float(min_abs)
                if math.isfinite(v) and v > 0.0:
                    self.scale_min_abs = v
            except Exception:
                pass
        self.scale_is_integral = bool(is_integral) if is_integral is not None else None

    @staticmethod
    def _scale_from_tensor(
        tensor: Optional[torch.Tensor],
    ) -> Optional[Tuple[float, Optional[float], bool]]:
        if (
            tensor is None
            or not isinstance(tensor, torch.Tensor)
            or tensor.numel() == 0
        ):
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
                tol = (
                    1e-6
                    if finite_vals.dtype in (torch.float32, torch.float64)
                    else 5e-4
                )
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


@dataclass
class BatchPolicy:
    sample_bytes: int
    host_sample_bytes: Optional[int] = None

    prebatch: int = 1
    prefetch_factor: int = 2
    num_workers: int = 0
    num_streams: int = 1
    max_concurrency: int = 1

    min_batch: int = 1
    max_batch: Optional[int] = None

    device_margin: float = 0.8
    host_margin: float = 0.8

    def __post_init__(self) -> None:
        self.sample_bytes = max(int(self.sample_bytes or 0), 0)
        if self.host_sample_bytes is None:
            self.host_sample_bytes = self.sample_bytes

        self.prebatch = max(int(self.prebatch or 0), 0)
        self.prefetch_factor = max(int(self.prefetch_factor or 0), 0)
        self.num_workers = max(int(self.num_workers or 0), 0)
        self.num_streams = max(int(self.num_streams or 0), 0)
        self.max_concurrency = max(int(self.max_concurrency or 0), 0)

        self.min_batch = max(int(self.min_batch or 1), 1)
        if self.max_batch is not None:
            self.max_batch = max(int(self.max_batch), 1)

        self.device_margin = float(self.device_margin)
        self.host_margin = float(self.host_margin)

    @property
    def host_concurrency(self) -> int:
        w = max(self.num_workers, 1)
        pf = max(self.prefetch_factor, 1)
        pre = max(self.prebatch, 1)
        return max(w * pf * pre, 1)

    @property
    def device_concurrency(self) -> int:
        mc = max(self.max_concurrency, 1)
        ns = max(self.num_streams, 1)
        pre = max(self.prebatch, 1)
        return max(mc * ns * pre, 1)

    @staticmethod
    def _cap_from_bytes(
        free_bytes: int,
        per_sample_bytes: int,
        margin: float,
        concurrency: int,
    ) -> int:
        if free_bytes <= 0 or per_sample_bytes <= 0:
            return 0
        safe = int(max(0, float(free_bytes)) * float(margin))
        if safe <= 0:
            return 0
        per = max(per_sample_bytes, 1) * max(concurrency, 1)
        if per <= 0:
            return 0
        return max(int(safe // per), 0)

    def cap_from_device(self, free_device_bytes: Optional[int]) -> int:
        if free_device_bytes is None:
            return 0
        return self._cap_from_bytes(
            int(free_device_bytes),
            int(self.sample_bytes),
            self.device_margin,
            self.device_concurrency,
        )

    def cap_from_host(self, free_host_bytes: Optional[int]) -> int:
        if free_host_bytes is None:
            return 0
        host_bytes = int(self.host_sample_bytes or 0)
        return self._cap_from_bytes(
            int(free_host_bytes),
            host_bytes,
            self.host_margin,
            self.host_concurrency,
        )

    def suggest_batch(
        self,
        free_device_bytes: Optional[int],
        free_host_bytes: Optional[int] = None,
    ) -> int:
        candidates = []
        dev_cap = self.cap_from_device(free_device_bytes)
        if dev_cap:
            candidates.append(dev_cap)
        host_cap = self.cap_from_host(free_host_bytes)
        if host_cap:
            candidates.append(host_cap)

        if not candidates:
            b = self.max_batch if self.max_batch is not None else self.min_batch
        else:
            b = min(candidates)
            if self.max_batch is not None:
                b = min(b, int(self.max_batch))

        b = max(int(b), self.min_batch)
        return max(b, 1)


@tensorclass(shadow=True)
class TensorDictMetadata:
    use_amp: bool = False
    amp_dtype: Optional[Any] = None
    device: Optional[Any] = None

    def autocast(self) -> contextlib.AbstractContextManager[Any]:
        if not (self.use_amp and isinstance(self.amp_dtype, torch.dtype)):
            return contextlib.nullcontext()

        if isinstance(self.device, torch.device):
            dev_type = self.device.type
        else:
            dev_type = "cuda" if torch.cuda.is_available() else "cpu"

        autocast_fn = getattr(torch, "autocast", None)
        amp_mod = getattr(torch, "amp", None)
        if hasattr(amp_mod, "autocast"):
            autocast_fn = amp_mod.autocast

        if callable(autocast_fn):
            return autocast_fn(device_type=dev_type, dtype=self.amp_dtype)

        return contextlib.nullcontext()
