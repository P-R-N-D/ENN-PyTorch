# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import os
import contextlib
import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Generic, MutableMapping, Optional, Tuple, TypeVar, Union

import torch

TExtra = TypeVar("TExtra")

_BOOTSTRAP_DEPTH = 0
@dataclass(slots=True)
class WorkerPolicy:
    nproc_per_node: int = 1
    device: str = "cpu"
    local_world_size: int = 1

    intra_ops: int = 1
    inter_ops: int = 1

    num_workers: int = 1
    prebatch: int = 1
    prefetch_factor: int = 2
    max_concurrency: int = 1
    h2d_streams: int = 1

    @staticmethod
    def _cpu_count() -> int:
        try:
            return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
        except Exception:
            return os.cpu_count() or 1

    @staticmethod
    def _detect_accelerator() -> Tuple[str, int]:
        dev_type = "cpu"
        n = 0

        try:
            accel = getattr(torch, "accelerator", None)
            if accel is not None and hasattr(accel, "is_available") and accel.is_available():
                current = getattr(accel, "current_accelerator", None)
                if callable(current):
                    dev = current(False)
                    if isinstance(dev, torch.device):
                        dev_type = dev.type
                dc = getattr(accel, "device_count", None)
                if callable(dc):
                    n = int(dc())
        except Exception:
            dev_type, n = "cpu", 0

        try:
            if n <= 0:
                if torch.cuda.is_available():
                    dev_type = "cuda"
                    n = int(torch.cuda.device_count())
                else:
                    xpu = getattr(torch, "xpu", None)
                    if xpu is not None and callable(getattr(xpu, "is_available", None)) and xpu.is_available():
                        dev_type = "xpu"
                        n = int(getattr(xpu, "device_count", lambda: 1)())
                    else:
                        mps_backend = getattr(torch.backends, "mps", None)
                        if mps_backend is not None and callable(getattr(mps_backend, "is_available", None)) and mps_backend.is_available():
                            dev_type = "mps"
                            n = 1
        except Exception:
            pass

        if n <= 0:
            dev_type, n = "cpu", 0
        return dev_type, max(0, n)

    @classmethod
    def autotune(cls) -> "WorkerPolicy":
        ncpu = cls._cpu_count()
        dev_type, nacc = cls._detect_accelerator()
        is_accel = nacc > 0

        if ncpu <= 2:
            inter_ops = 1
            intra_ops = max(1, ncpu - inter_ops)
            num_workers = max(1, ncpu)
        elif 2 < ncpu <= 8:
            inter_ops = max(1, ncpu // 4)
            intra_ops = max(1, ncpu - inter_ops)
            num_workers = max(2, min(8, ncpu // 2))
        else:
            inter_ops = max(2, min(8, ncpu // 6))
            intra_ops = max(1, ncpu - inter_ops)
            num_workers = max(4, min(16, ncpu // 2))

        max_concurrency = max(1, num_workers)

        if is_accel:
            prebatch = 4
            prefetch_factor = 2
        else:
            prebatch = 1
            prefetch_factor = 1

        env_pre = os.environ.get("STNET_PREBATCH")
        if env_pre:
            try:
                prebatch = max(1, int(env_pre))
            except Exception:
                pass
        env_pf = os.environ.get("STNET_PREFETCH_FACTOR")
        if env_pf:
            try:
                prefetch_factor = max(1, int(env_pf))
            except Exception:
                pass

        max_total_ahead = 8 if is_accel else 4
        total = prebatch * prefetch_factor
        if total > max_total_ahead:
            scale = max_total_ahead / float(total)
            prefetch_factor = max(1, int(prefetch_factor * scale))
            total = prebatch * prefetch_factor
            if total > max_total_ahead:
                prebatch = max(1, int(max_total_ahead // max(1, prefetch_factor)))

        local_world = max(1, nacc or 1) if is_accel else 1

        return cls(
            nproc_per_node=local_world,
            device=dev_type,
            local_world_size=local_world,
            intra_ops=int(intra_ops),
            inter_ops=int(inter_ops),
            num_workers=int(num_workers),
            prebatch=int(prebatch),
            prefetch_factor=int(prefetch_factor),
            max_concurrency=int(max_concurrency),
            h2d_streams=2 if dev_type in ("cuda", "xpu") else 1,
        )

    def as_threads_dict(self) -> Dict[str, int]:
        return {
            "intra_ops": int(self.intra_ops),
            "inter_ops": int(self.inter_ops),
            "num_workers": int(self.num_workers),
            "max_concurrancy": int(self.max_concurrency),
            "prebatch": int(self.prebatch),
            "prefetch_factor": int(self.prefetch_factor),
        }

    def as_procs_dict(self) -> Dict[str, Union[int, str]]:
        return {
            "nproc_per_node": int(self.nproc_per_node),
            "device": str(self.device),
        }

    def apply_torch_threads(self) -> None:
        try:
            torch.set_num_threads(max(1, int(self.intra_ops)))
        except Exception:
            pass
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(max(1, int(self.inter_ops)))
            except Exception:
                pass


@dataclass
class DataPolicy(Generic[TExtra]):
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
        if dev.type == "cuda" and torch.cuda.is_available():
            major, minor = self.cuda_compute_capability(dev)
            if major > 0 or minor > 0:
                self.cuda_cc = (int(major), int(minor))
            else:
                self.cuda_cc = None
        else:
            self.cuda_cc = None

    @staticmethod
    def cuda_compute_capability(device: Union[torch.device, str]) -> Tuple[int, int]:
        dev = torch.device(device)
        if dev.type != "cuda" or not torch.cuda.is_available():
            return (0, 0)
        try:
            major, minor = torch.cuda.get_device_capability(dev)
        except Exception:
            return (0, 0)
        return (int(major), int(minor))

    @staticmethod
    def is_cpu_bf16_supported() -> bool:
        try:
            mkldnn_ops = getattr(torch.ops, "mkldnn", None)
            if mkldnn_ops is not None and hasattr(mkldnn_ops, "_is_mkldnn_bf16_supported"):
                return bool(torch.ops.mkldnn._is_mkldnn_bf16_supported())
        except Exception:
            pass
        return False

    @staticmethod
    def is_cuda_bf16_supported() -> bool:
        try:
            if not torch.cuda.is_available():
                return False
            f = getattr(torch.cuda, "is_bf16_supported", None)
            if callable(f):
                return bool(f())
            major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
            return major >= 8
        except Exception:
            return False

    @staticmethod
    def _resolve_device(device: Optional[Union[torch.device, str]]) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @classmethod
    def is_float8_supported(
        cls, device: Optional[Union[torch.device, str]] = None
    ) -> Tuple[bool, str]:
        dev = cls._resolve_device(device)
        if dev.type != "cuda" or not torch.cuda.is_available():
            return (False, f"FP8 requires CUDA (found {dev.type})")
        major, minor = cls.cuda_compute_capability(dev)
        if major <= 0:
            return (False, "Unable to query CUDA compute capability")
        if major < 9:
            return (False, f"FP8 requires sm_90+ (found sm_{major}{minor})")
        try:
            import transformer_engine.pytorch as te  # type: ignore
            backend = getattr(te, "__name__", "transformer_engine.pytorch")
            return (True, backend)
        except Exception:
            return (False, "transformer_engine.pytorch not found")

    @classmethod
    def is_int8_supported(
        cls, device: Optional[Union[torch.device, str]] = None
    ) -> Tuple[bool, str]:
        dev = cls._resolve_device(device)
        if dev.type != "cuda" or not torch.cuda.is_available():
            return (False, f"INT8 requires CUDA (found {dev.type})")
        major, minor = cls.cuda_compute_capability(dev)
        if major <= 0:
            return (False, "Unable to query CUDA compute capability")
        if major < 7:
            return (False, f"INT8 requires sm_70+ (found sm_{major}{minor})")
        try:
            importlib.import_module("torchao.quantization")
            return (True, "torchao.quantization")
        except Exception:
            return (True, f"sm_{major}{minor}")

    @classmethod
    def is_int4_supported(
        cls, device: Optional[Union[torch.device, str]] = None
    ) -> Tuple[bool, str]:
        dev = cls._resolve_device(device)
        if dev.type != "cuda" or not torch.cuda.is_available():
            return (False, f"INT4 requires CUDA (found {dev.type})")
        major, minor = cls.cuda_compute_capability(dev)
        if major <= 0:
            return (False, "Unable to query CUDA compute capability")
        if major < 8:
            return (False, f"INT4 requires sm_80+ (found sm_{major}{minor})")
        try:
            importlib.import_module("torchao.optim")
            return (True, "torchao.optim")
        except Exception:
            with contextlib.suppress(Exception):
                importlib.import_module("torchao.prototype.low_bit_optim")
                return (True, f"sm_{major}{minor}")
        return (False, "torchao low-bit optimizers unavailable")

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
    ) -> "DataPolicy[TExtra]":
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
    local_world_size: int = 1

    min_batch: int = 1
    max_batch: Optional[int] = None

    device_margin: float = 0.8
    host_margin: float = 0.8
    device_budget_ratio: float = 0.0
    device_budget_min_bytes: int = 0
    device_budget_max_bytes: Optional[int] = None

    host_budget_ratio: float = 0.0
    host_budget_min_bytes: int = 0
    host_budget_max_bytes: Optional[int] = None

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

        self.device_margin = max(0.0, min(1.0, float(self.device_margin)))
        self.host_margin = max(0.0, min(1.0, float(self.host_margin)))

        self.device_budget_ratio = max(0.0, min(1.0, float(self.device_budget_ratio or 0.0)))
        self.host_budget_ratio = max(0.0, min(1.0, float(self.host_budget_ratio or 0.0)))

        self.device_budget_min_bytes = max(int(self.device_budget_min_bytes or 0), 0)
        self.host_budget_min_bytes = max(int(self.host_budget_min_bytes or 0), 0)

        if self.device_budget_max_bytes is not None:
            self.device_budget_max_bytes = max(int(self.device_budget_max_bytes), 0)
        if self.host_budget_max_bytes is not None:
            self.host_budget_max_bytes = max(int(self.host_budget_max_bytes), 0)

    def host_inflight_batches_per_proc(self) -> int:
        return (
            max(1, self.max_concurrency) * max(1, self.prebatch)
            + max(1, self.prefetch_factor)
            + max(1, self.num_streams)
            + 1
        )

    @staticmethod
    def _budget_bytes(
        total_bytes: Optional[int],
        *,
        budget_ratio: float,
        budget_min_bytes: int,
        budget_max_bytes: Optional[int],
    ) -> int:
        total = int(total_bytes) if total_bytes is not None else 0
        ratio = float(budget_ratio or 0.0)
        base = int(float(total) * ratio) if total > 0 and ratio > 0.0 else 0
        budget = max(int(budget_min_bytes or 0), base)
        if (budget <= 0) and (total <= 0) and (budget_max_bytes is not None):
            budget = int(budget_max_bytes)
        elif budget_max_bytes is not None:
            budget = min(budget, int(budget_max_bytes))
        return max(0, int(budget))

    def suggest_batch(
        self,
        *,
        dev_free: Optional[int] = None,
        host_free: Optional[int] = None,
        dev_total: Optional[int] = None,
        host_total: Optional[int] = None,
        local_world_size: Optional[int] = None,
    ) -> int:
        lw = (
            int(local_world_size)
            if local_world_size is not None
            else int(self.local_world_size or 1)
        )
        if lw <= 0:
            lw = 1

        use_dev_budget = (
            self.device_budget_ratio > 0.0
            or self.device_budget_min_bytes > 0
            or self.device_budget_max_bytes is not None
        )
        use_host_budget = (
            self.host_budget_ratio > 0.0
            or self.host_budget_min_bytes > 0
            or self.host_budget_max_bytes is not None
        )

        dev_cap: Optional[int] = None
        if dev_free is not None and dev_free >= 0 and self.sample_bytes > 0:
            denom = max(1, int(self.sample_bytes))
            usable = int(float(dev_free) * float(self.device_margin))
            if use_dev_budget:
                budget = self._budget_bytes(
                    dev_total,
                    budget_ratio=self.device_budget_ratio,
                    budget_min_bytes=self.device_budget_min_bytes,
                    budget_max_bytes=self.device_budget_max_bytes,
                )
                if budget > 0:
                    usable = min(int(usable), int(budget))
            dev_cap = int(max(0, usable) // denom)

        host_cap: Optional[int] = None
        if host_free is not None and host_free >= 0 and (self.host_sample_bytes or 0) > 0:
            inflight = self.host_inflight_batches_per_proc()
            denom = (
                max(1, int(self.host_sample_bytes or 0))
                * max(1, inflight)
                * max(1, lw)
            )
            usable = int(float(host_free) * float(self.host_margin))
            if use_host_budget:
                budget = self._budget_bytes(
                    host_total,
                    budget_ratio=self.host_budget_ratio,
                    budget_min_bytes=self.host_budget_min_bytes,
                    budget_max_bytes=self.host_budget_max_bytes,
                )
                if budget > 0:
                    usable = min(int(usable), int(budget))
            host_cap = int(max(0, usable) // denom)

        candidates = [c for c in (dev_cap, host_cap) if isinstance(c, int) and c >= 0]
        if not candidates:
            b = self.max_batch if self.max_batch is not None else self.min_batch
        else:
            b = min(candidates)
            if self.max_batch is not None:
                b = min(b, int(self.max_batch))

        b = max(int(b), int(self.min_batch))
        return max(1, b)


