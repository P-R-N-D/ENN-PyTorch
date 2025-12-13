# -*- coding: utf-8 -*-
from __future__ import annotations

import collections.abc as _abc
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
            return len(os.sched_getaffinity(0))
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


@dataclass(slots=True)
class LoaderPolicy:
    max_batches_accel: int = 4
    max_batches_cpu: int = 2
    soft_cap_multiplier: int = 2

    def hard_inflight_batches(self, device: torch.device | str) -> int:
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if dev.type in ("cuda", "xpu", "mps"):
            return max(1, int(self.max_batches_accel))
        return max(1, int(self.max_batches_cpu))

    def apply_soft_limits(self, wp: WorkerPolicy, device: torch.device | str) -> WorkerPolicy:
        hard = int(self.hard_inflight_batches(device))
        soft_cap = max(1, int(hard * max(1, int(self.soft_cap_multiplier))))

        num_workers = max(0, int(getattr(wp, "num_workers", 0) or 0))
        if num_workers > soft_cap:
            wp.num_workers = max(1, int(soft_cap))
            num_workers = int(wp.num_workers)
        prefetch_factor = max(1, int(getattr(wp, "prefetch_factor", 1) or 1))
        prebatch = max(0, int(getattr(wp, "prebatch", 0) or 0))

        inflight = num_workers * prefetch_factor + prebatch
        if inflight > soft_cap:
            budget = max(1, soft_cap - prebatch)
            if num_workers > 0:
                prefetch_factor = max(1, min(prefetch_factor, budget // max(1, num_workers)))
            else:
                prefetch_factor = 1
            wp.prefetch_factor = int(prefetch_factor)

            inflight = num_workers * int(wp.prefetch_factor) + prebatch
            if inflight > soft_cap:
                wp.prebatch = max(0, int(soft_cap - num_workers * int(wp.prefetch_factor)))

        with contextlib.suppress(Exception):
            wp.max_concurrency = max(1, min(int(getattr(wp, "max_concurrency", 1) or 1), soft_cap))
        return wp

    def wrap_input(self, loader: Any, device: torch.device | str, *, name: str) -> Any:
        from ..data.nodes import BufferedLoader

        max_batches = self.hard_inflight_batches(device)
        return BufferedLoader(loader, max_batches=max_batches, name=name)


@dataclass
class Session:
    sources: Any
    device: torch.device | str

    val_frac: float = 0.0
    non_blocking_copy: bool = True
    labels_dtype: Optional[torch.dtype] = None
    sanitize: bool = True
    flatten_features: bool = True

    train_weights: Optional[Mapping[str, float]] = None
    val_weights: Optional[Mapping[str, float]] = None

    worker_policy: Optional[WorkerPolicy] = None
    loader_policy: LoaderPolicy = field(default_factory=LoaderPolicy)

    raw_training_loader: Any = None
    raw_validation_loader: Any = None

    training_loader: Any = None
    validation_loader: Any = None
    disposable: Any = None

    _opened: bool = False

    def open(
        self,
        *,
        train_state: Optional[Dict[str, Any]] = None,
        val_state: Optional[Dict[str, Any]] = None,
    ) -> "Session":
        from ..data.pipeline import fetch

        dev = torch.device(self.device) if not isinstance(self.device, torch.device) else self.device

        wp = self.worker_policy or WorkerPolicy.autotune()
        wp = self.loader_policy.apply_soft_limits(wp, dev)
        self.worker_policy = wp

        dl = fetch(
            sources=self.sources,
            device=dev,
            val_frac=float(self.val_frac),
            non_blocking_copy=bool(self.non_blocking_copy),
            labels_dtype=self.labels_dtype,
            sanitize=bool(self.sanitize),
            flatten_features=bool(self.flatten_features),
            train_weights=self.train_weights,
            val_weights=self.val_weights,
            worker_policy=wp,
        )

        train_loader = dl.get("training_loader")
        val_loader = dl.get("validation_loader")
        self.disposable = dl.get("disposable")

        self.raw_training_loader = train_loader
        self.raw_validation_loader = val_loader

        if train_state and train_loader is not None and hasattr(train_loader, "load_state_dict"):
            try:
                train_loader.load_state_dict(train_state)
            except Exception:
                pass
        if val_state and val_loader is not None and hasattr(val_loader, "load_state_dict"):
            try:
                val_loader.load_state_dict(val_state)
            except Exception:
                pass

        self.training_loader = (
            self.loader_policy.wrap_input(train_loader, dev, name="train-input")
            if train_loader is not None
            else None
        )
        self.validation_loader = (
            self.loader_policy.wrap_input(val_loader, dev, name="val-input")
            if val_loader is not None
            else None
        )

        self._opened = True
        return self

    def close(self) -> None:
        if not self._opened:
            return
        keep = getattr(self, "disposable", None)
        if keep is not None:
            try:
                keep.cleanup()
            except Exception:
                pass
        self._opened = False

    def __enter__(self) -> "Session":
        return self.open()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class LazyDict(_abc.Mapping):
    def __init__(self, keys: Any, getter: Any, *, name: str = "LazyDict", cache: bool = False) -> None:
        self._keys = keys
        self._getter = getter
        self._name = str(name or "LazyDict")
        self._cache_enabled = bool(cache)
        self._cache: Optional[dict[Any, Any]] = {} if self._cache_enabled else None

    def __len__(self) -> int:
        return int(len(self._keys))

    def __iter__(self):
        return iter(self._keys)

    def __getitem__(self, key: Any) -> Any:
        if self._cache is not None and key in self._cache:
            return self._cache[key]
        v = self._getter(key)
        if self._cache is not None:
            self._cache[key] = v
        return v

    def __contains__(self, key: object) -> bool:
        try:
            return key in self._keys
        except Exception:
            return False

    def keys(self):
        return self._keys

    def values(self):
        return (self[k] for k in self._keys)

    def items(self):
        return ((k, self[k]) for k in self._keys)

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def collect(self) -> dict[Any, Any]:
        return {k: self[k] for k in self._keys}

    def materialize(self) -> dict[Any, Any]:
        return self.collect()


@dataclass
class Dataset(Generic[TExtra]):
    device: torch.device
    device_type: str = field(init=False, default="cpu")
    cuda_cc: Optional[Tuple[int, int]] = field(init=False, default=None)
    float_dtypes: Tuple[torch.dtype, ...] = field(default_factory=tuple)
    int_dtypes: Tuple[torch.dtype, ...] = field(default_factory=tuple)
    float8_dtypes: Tuple[torch.dtype, ...] = field(default_factory=tuple)
    input_data: Any = None
    output_data: Any = None

    feature_dtype: torch.dtype = torch.float32
    label_float_dtype: torch.dtype = torch.float32

    scale_max_abs: Optional[float] = None
    scale_min_abs: Optional[float] = None
    scale_is_integral: Optional[bool] = None
    stats: MutableMapping[str, torch.Tensor] = field(default_factory=dict)
    extra: Dict[str, TExtra] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._refresh_device_info()
        self._refresh_dtypes_from_env()

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
            import transformer_engine.pytorch as te
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
    ) -> "Dataset[TExtra]":
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


    @staticmethod
    def _dtype_from_env(var: str, default: torch.dtype) -> torch.dtype:
        raw = os.environ.get(var)
        if raw is None:
            return default
        name = str(raw).strip()
        if not name:
            return default
        if name.startswith("torch."):
            name = name.split(".", 1)[1]
        dt = getattr(torch, name, None)
        return dt if isinstance(dt, torch.dtype) else default

    @staticmethod
    def _normalize_float_dtype(dtype: torch.dtype) -> torch.dtype:
        try:
            if torch.is_floating_point(torch.empty((), dtype=dtype)):
                return dtype
        except Exception:
            pass
        return torch.float32

    def _refresh_dtypes_from_env(self) -> None:
        self.feature_dtype = self._normalize_float_dtype(
            self._dtype_from_env("STNET_FEATURE_DTYPE", getattr(self, "feature_dtype", torch.float32))
        )
        self.label_float_dtype = self._normalize_float_dtype(
            self._dtype_from_env("STNET_LABEL_DTYPE", getattr(self, "label_float_dtype", torch.float32))
        )

    @staticmethod
    def _to_dtype(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return t.to(dtype=dtype)

    @staticmethod
    def _assert_finites(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if torch.is_floating_point(tensor) or torch.is_complex(tensor):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} tensor contains non-finite values")
        return tensor

    def _preprocess_x(self, x_tuple: Any) -> torch.Tensor:
        import math
        from ..data.datatype import to_tuple

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
        tensor = torch.as_tensor(values, dtype=self.feature_dtype)
        return self._assert_finites(tensor, "feature")

    def _preprocess_y(self, value: Any) -> torch.Tensor:
        from ..data.datatype import to_torch_tensor

        try:
            tensor = to_torch_tensor(value)
        except Exception:
            tensor = None
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(value)
        t = tensor.detach()
        if t.is_floating_point() or t.is_complex():
            t = self._to_dtype(t, self.label_float_dtype)
        else:
            t = self._to_dtype(t, torch.int64)
        return t

    def _preprocess_batch(self, x_value: Any, y_value: Any):
        from contextlib import suppress
        from ..data.datatype import to_torch_tensor

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

        feature_tensor = self._assert_finites(
            self._to_dtype(feature_tensor.detach(), self.feature_dtype), "feature"
        )
        if feature_tensor.dim() == 0:
            feature_tensor = feature_tensor.reshape(1, 1)
        elif feature_tensor.dim() == 1:
            feature_tensor = feature_tensor.reshape(-1, 1)
        else:
            batch_dim = int(feature_tensor.shape[0]) if feature_tensor.shape else 1
            feature_tensor = feature_tensor.reshape(batch_dim, -1)
        batch_size = int(feature_tensor.shape[0])

        label_tensor = label_tensor.detach()
        if label_tensor.is_floating_point() or label_tensor.is_complex():
            label_tensor = self._to_dtype(label_tensor, self.label_float_dtype)
        else:
            label_tensor = self._to_dtype(label_tensor, torch.int64)
        label_tensor = self._assert_finites(label_tensor, "label")

        if label_tensor.dim() == 0:
            label_tensor = label_tensor.unsqueeze(0)
        if label_tensor.dim() == 1 and label_tensor.shape[0] == batch_size:
            label_tensor = label_tensor.unsqueeze(-1)
        if label_tensor.shape[0] != batch_size:
            return None

        label_shape = tuple(label_tensor.shape[1:])
        batch_keys = [(int(index),) for index in range(batch_size)]
        return (feature_tensor, label_tensor, batch_keys, label_shape)

    def preprocess(self, data: Any):
        from ..data.datatype import to_tuple
        from collections.abc import Mapping as _Mapping

        track_in = False
        with contextlib.suppress(Exception):
            v = str(os.environ.get("STNET_DATASET_TRACK_INPUT", "")).strip().lower()
            track_in = v in {"1", "true", "yes", "y", "on"}
        if track_in:
            self.input_data = data
        else:
            self.input_data = None

        try:
            from tensordict import TensorDictBase
        except Exception:
            TensorDictBase = None

        if TensorDictBase is not None and isinstance(data, TensorDictBase):
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
                raise ValueError("preprocess(TensorDict): missing 'labels' or 'labels_flat'")
            if labels.ndim == 1:
                labels = labels.unsqueeze(0)
            if labels.shape[0] != feats.shape[0]:
                raise ValueError("preprocess(TensorDict): features and labels batch mismatch")

            feats = self._assert_finites(self._to_dtype(feats.detach(), self.feature_dtype), "feature")
            labels = labels.detach()
            if labels.is_floating_point() or labels.is_complex():
                labels = self._to_dtype(labels, self.label_float_dtype)
            else:
                labels = self._to_dtype(labels, torch.int64)
            labels = self._assert_finites(labels, "label")

            label_shape = tuple(labels.shape[1:])
            keys = [(int(i),) for i in range(int(feats.shape[0]))]
            self.accumulate_scale(feats)
            return (feats, labels, keys, label_shape)

        if isinstance(data, _Mapping) and "X" in data and ("Y" in data):
            x, y = (data["X"], data["Y"])
            batch_result = self._preprocess_batch(x, y)
            if batch_result is not None:
                feats, labels, keys, label_shape = batch_result
                self.accumulate_scale(feats)
                return (feats, labels, keys, label_shape)
            xr, yt = (
                self._preprocess_x(x).unsqueeze(0),
                self._assert_finites(self._preprocess_y(y), "label"),
            )
            if yt.dim() == 0 or yt.dim() == 1:
                yt = yt.unsqueeze(0)
            keys = [to_tuple(x)]
            label_shape = tuple(yt.shape[1:])
            self.accumulate_scale(xr)
            return (xr, yt, keys, label_shape)

        if isinstance(data, (tuple, list)) and len(data) >= 2:
            x, y = (data[0], data[1])
            batch_result = self._preprocess_batch(x, y)
            if batch_result is not None:
                feats, labels, keys, label_shape = batch_result
                self.accumulate_scale(feats)
                return (feats, labels, keys, label_shape)
            xr = self._preprocess_x(x).unsqueeze(0)
            yt = self._assert_finites(self._preprocess_y(y), "label")
            if yt.dim() == 0:
                yt = yt.unsqueeze(0)
            elif yt.shape[0] != 1:
                yt = yt.unsqueeze(0)
            keys = [to_tuple(x)]
            label_shape = tuple(yt.shape[1:])
            self.accumulate_scale(xr)
            return (xr, yt, keys, label_shape)

        if isinstance(data, _Mapping) and len(data) > 0:
            items = list(data.items())
            if any((isinstance(k, str) for k, _ in items)):
                raise TypeError(
                    "preprocess: keys in a multi-sample dict must be tuples. "
                    "Provide single samples as {'X': ..., 'Y': ...}."
                )
            keys = [to_tuple(k) for k, _ in items]
            feats = torch.stack([self._preprocess_x(k) for k in keys], dim=0)
            lbl_list = [self._assert_finites(self._preprocess_y(v), "label") for _, v in items]
            if lbl_list and all((t.shape == lbl_list[0].shape for t in lbl_list)):
                labels = torch.stack(lbl_list, dim=0)
            else:
                labels = torch.cat([t.unsqueeze(0) for t in lbl_list], dim=0)
            labels = self._assert_finites(labels, "label")
            label_shape = tuple(labels.shape[1:])
            self.accumulate_scale(feats)
            return (feats, labels, keys, label_shape)

        raise ValueError(
            "preprocess: unsupported input format. Provide a mapping or an (X, Y) pair."
        )

    def postprocess(self, keys: list, preds: torch.Tensor | Sequence[torch.Tensor]):
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

        fixed_keys: list = []
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

        out = {k: v for k, v in zip(fixed_keys, rows)}
        track_out = False
        with contextlib.suppress(Exception):
            v = str(os.environ.get("STNET_DATASET_TRACK_OUTPUT", "")).strip().lower()
            track_out = v in {"1", "true", "yes", "y", "on"}
        if track_out:
            self.output_data = out
        else:
            self.output_data = None
        return out

    def batch_to_device(
        self,
        batch: Any,
        device: torch.device,
        *,
        non_blocking: Optional[bool] = None,
        pin_memory: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from ..data.datatype import to_torch_tensor

        try:
            from tensordict import TensorDictBase
        except Exception:
            TensorDictBase = None

        def _extract_xy(sample: Any):
            if TensorDictBase is not None and isinstance(sample, TensorDictBase):
                x = sample.get("features", sample.get("X", None))
                y = sample.get("labels", sample.get("Y", None))
                return x, y
            if isinstance(sample, Mapping):
                x = sample.get("features", sample.get("X", None))
                y = sample.get("labels", sample.get("Y", None))
                return x, y
            return (None, None)

        dev = device if isinstance(device, torch.device) else torch.device(device)
        dev_type = dev.type

        if non_blocking is None:
            non_blocking = bool(dev_type in {"cuda", "xpu"})
        if pin_memory is None:
            pin_memory = bool(dev_type in {"cuda", "xpu"})

        X: Optional[torch.Tensor] = None
        Y: Optional[torch.Tensor] = None

        if (TensorDictBase is not None and isinstance(batch, TensorDictBase)) or isinstance(batch, Mapping):
            x_raw, y_raw = _extract_xy(batch)
            if x_raw is None or y_raw is None:
                feats, labs, *_ = self.preprocess(batch)
                x_raw, y_raw = feats, labs
            X = to_torch_tensor(x_raw)
            Y = to_torch_tensor(y_raw)
        elif isinstance(batch, tuple) and len(batch) >= 2 and not isinstance(batch[0], (Mapping,)):
            X = to_torch_tensor(batch[0])
            Y = to_torch_tensor(batch[1])
        elif isinstance(batch, (list, tuple)):
            if len(batch) == 0:
                X = torch.empty((0,), device="cpu")
                Y = torch.empty((0,), device="cpu")
            elif all((TensorDictBase is not None and isinstance(elem, TensorDictBase)) or isinstance(elem, Mapping) for elem in batch):
                xs: list[torch.Tensor] = []
                ys: list[torch.Tensor] = []
                for elem in batch:
                    x_raw, y_raw = _extract_xy(elem)
                    if x_raw is None or y_raw is None:
                        feats, labs, *_ = self.preprocess(elem)
                        x_raw, y_raw = feats, labs
                    xs.append(to_torch_tensor(x_raw))
                    ys.append(to_torch_tensor(y_raw))

                def _stack(tensors: list[torch.Tensor]) -> torch.Tensor:
                    if tensors and all(torch.is_tensor(t) and t.device.type != "cpu" for t in tensors):
                        return torch.stack(tensors, dim=0)
                    if pin_memory and dev_type in {"cuda", "xpu"} and tensors:
                        t0 = tensors[0]
                        out = torch.empty(
                            (len(tensors), *tuple(t0.shape)),
                            dtype=t0.dtype,
                            device="cpu",
                            pin_memory=True,
                        )
                        try:
                            torch.stack(tensors, dim=0, out=out)
                            return out
                        except Exception:
                            return torch.stack(tensors, dim=0).pin_memory()
                    return torch.stack(tensors, dim=0)

                X = _stack(xs)
                Y = _stack(ys)
            else:
                feats, labs, *_ = self.preprocess(batch)
                X = to_torch_tensor(feats)
                Y = to_torch_tensor(labs)
        else:
            feats, labs, *_ = self.preprocess(batch)
            X = to_torch_tensor(feats)
            Y = to_torch_tensor(labs)

        if X is None or Y is None:
            raise ValueError(
                f"batch_to_device: could not extract (X, Y) from batch type: {type(batch)!r}"
            )

        if dev_type == "cpu":
            if X.device.type != "cpu":
                X = X.to("cpu")
            if Y.device.type != "cpu":
                Y = Y.to("cpu")
            return X, Y

        if X.device == dev and Y.device == dev:
            return X, Y

        if dev_type in {"cuda", "xpu"}:
            stream = None
            with contextlib.suppress(Exception):
                stream = getattr(torch, dev_type).current_stream(dev)

            def _to_dev(t: torch.Tensor) -> torch.Tensor:
                t_cpu = t
                if t_cpu.device.type != "cpu":
                    return t_cpu.to(dev, non_blocking=False)
                if pin_memory and (not (hasattr(t_cpu, "is_pinned") and t_cpu.is_pinned())):
                    pinned = torch.empty_like(t_cpu, device="cpu", pin_memory=True)
                    pinned.copy_(t_cpu, non_blocking=False)
                    t_cpu = pinned
                out_dev = t_cpu.to(dev, non_blocking=bool(non_blocking))
                if stream is not None and hasattr(t_cpu, "is_pinned") and t_cpu.is_pinned():
                    with contextlib.suppress(Exception):
                        t_cpu.record_stream(stream)
                return out_dev

            return _to_dev(X), _to_dev(Y)

        return X.to(dev, non_blocking=False), Y.to(dev, non_blocking=False)


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


