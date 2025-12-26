# -*- coding: utf-8 -*-
from __future__ import annotations

import collections.abc as _abc
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import threading
from contextlib import suppress
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
)

import torch
import numpy as np

from ..core.compat import ensure_torchdata
from ..core.casting import dtype_from_name, env_bool, env_first_int
from ..core.staging import Buffer, Pool, ProducerError, best_effort_close
from ..core.system import Thread, Memory
from . import schemas as _schemas
from .schemas import _FEATURE_KEY_ALIASES, _LABEL_KEY_ALIASES, default_underflow_action, normalize_underflow_action

_LOGGER = logging.getLogger(__name__)

TensorLike = Any

# ---- torchdata nodes (robust defaults) ---------------------------------------

_TORCHDATA_AVAILABLE = False


class _MissingTorchDataBaseNode:
    """Placeholder type to avoid NameError/TypeError cascades when torchdata is missing."""


BaseNode: type = _MissingTorchDataBaseNode  # will be overwritten if torchdata imports
_Loader: Any = None
ParallelMapper: Any = None

_Batcher: Any = None
MultiNodeWeightedSampler: Any = None
PinMemory: Any = None
_Prefetcher: Any = None
SamplerWrapper: Any = None
_Unbatcher: Any = None

try:
    from torchdata.nodes import BaseNode as _TDBaseNode
    from torchdata.nodes import Loader as _TDLoader
    from torchdata.nodes import ParallelMapper as _TDParallelMapper

    BaseNode = _TDBaseNode
    _Loader = _TDLoader
    ParallelMapper = _TDParallelMapper
    _TORCHDATA_AVAILABLE = True
except Exception as _e:
    ensure_torchdata(err=_e, context="stnet.data.nodes")

try:
    from torchdata.nodes import Batcher as _TDBatcher
    from torchdata.nodes import MultiNodeWeightedSampler as _TDMultiNodeWeightedSampler
    from torchdata.nodes import PinMemory as _TDPinMemory
    from torchdata.nodes import Prefetcher as _TDPrefetcher
    from torchdata.nodes import SamplerWrapper as _TDSamplerWrapper
    from torchdata.nodes import Unbatcher as _TDUnbatcher

    _Batcher = _TDBatcher
    MultiNodeWeightedSampler = _TDMultiNodeWeightedSampler
    PinMemory = _TDPinMemory
    _Prefetcher = _TDPrefetcher
    SamplerWrapper = _TDSamplerWrapper
    _Unbatcher = _TDUnbatcher
except Exception as _e:
    ensure_torchdata(err=_e, context="stnet.data.nodes")

# ---- tensordict mmap ---------------------------------------------------------

try:
    from tensordict import MemoryMappedTensor
except ImportError:  # pragma: no cover
    MemoryMappedTensor = None  # type: ignore

# ---- torch.utils.data Sampler base ------------------------------------------

try:
    from torch.utils.data import Sampler as _Sampler
except Exception:  # pragma: no cover
    _Sampler = object


def _is_accelerator_available() -> bool:
    """Best-effort check for an available accelerator backend."""
    try:
        if torch.cuda.is_available():
            return True
    except Exception:
        pass

    # Intel XPU (optional)
    try:
        xpu = getattr(torch, "xpu", None)
        if xpu is not None:
            fn = getattr(xpu, "is_available", None)
            if callable(fn) and fn():
                return True
    except Exception:
        pass

    # Apple MPS (optional)
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None:
            fn = getattr(mps, "is_available", None)
            if callable(fn) and fn():
                return True
    except Exception:
        pass

    return False


def _gpu_guard_ok(use_cuda_stream: bool, device: torch.device, guard_bytes: int) -> bool:
    """Return True if CUDA free-memory guard is satisfied (or guard disabled)."""
    if (not use_cuda_stream) or int(guard_bytes) <= 0:
        return True
    try:
        free_b, _total_b = torch.cuda.mem_get_info(
            device=device if isinstance(device, torch.device) else None
        )
        return bool(int(free_b) >= int(guard_bytes))
    except Exception:
        # best-effort: do not block training
        return True


def _host_guard_ok(guard_bytes: int) -> bool:
    """Return True if host available-memory guard is satisfied (or guard disabled)."""
    if int(guard_bytes) <= 0:
        return True
    try:
        return bool(int(Memory.available()) >= int(guard_bytes))
    except Exception:
        return True


class BatchState:
    """Per-session (or per-loader) scaling factor for auto batch sizing.

    - Cross-process safe: uses multiprocessing.Value so updates propagate to loader workers.
    - Thread-safe: Value has an internal lock; we also guard bounds carefully.
    """

    __slots__ = ("_v", "_min_scale", "_max_scale")

    def __init__(self, scale: float = 1.0, *, min_scale: float = 0.5, max_scale: float = 2.0) -> None:
        self._min_scale = float(min_scale)
        self._max_scale = float(max_scale)
        self._v = mp.Value("d", 1.0, lock=True)
        self.reset(scale)

    def __getstate__(self):
        return (self._v, float(self._min_scale), float(self._max_scale))

    def __setstate__(self, state):
        try:
            v, mn, mx = state
        except Exception:
            v, mn, mx = None, 0.5, 2.0

        self._min_scale = float(mn) if isinstance(mn, (int, float, str)) else 0.5
        self._max_scale = float(mx) if isinstance(mx, (int, float, str)) else 2.0

        if v is None:
            self._v = mp.Value("d", 1.0, lock=True)
        else:
            self._v = v

        # Backward-compat: handle (scale, mn, mx)
        try:
            if not hasattr(self._v, "get_lock"):
                scale = float(v)
                self._v = mp.Value("d", 1.0, lock=True)
                self.reset(scale)
        except Exception:
            self._v = mp.Value("d", 1.0, lock=True)

    def get(self) -> float:
        """Return the current scaling factor.

        This is a hot-path call (can happen once per batch). Acquiring the
        multiprocessing.Value lock on every read adds measurable overhead.

        We therefore use a lock-free fast path and fall back to a locked read
        only if the observed value looks invalid/out-of-bounds.
        """
        mn = float(self._min_scale)
        mx = float(self._max_scale)

        try:
            v = float(self._v.value)
        except Exception:
            v = float('nan')

        # Fast path: in normal operation the writer always stores a finite,
        # clamped value, so we can avoid the lock.
        if math.isfinite(v) and (v > 0.0) and (mn <= v <= mx):
            return float(v)

        # Slow path: take the lock and re-read, then clamp defensively.
        try:
            with self._v.get_lock():
                v = float(self._v.value)
        except Exception:
            with suppress(Exception):
                v = float(self._v.value)
            if not isinstance(v, (int, float)):
                v = 1.0

        if (not math.isfinite(v)) or (not (v > 0.0)):
            v = 1.0
        if v < mn:
            v = mn
        elif v > mx:
            v = mx
        return float(v)

    def reset(self, value: float = 1.0) -> None:
        try:
            v = float(value)
        except Exception:
            v = 1.0
        if not (v > 0.0):
            v = 1.0
        v = max(float(self._min_scale), min(float(self._max_scale), float(v)))
        try:
            with self._v.get_lock():
                self._v.value = float(v)
        except Exception:
            with suppress(Exception):
                self._v.value = float(v)

    def request_scale_up(self, factor: float) -> None:
        try:
            f = float(factor)
        except Exception:
            f = 1.0
        if not (f > 0.0):
            return
        try:
            with self._v.get_lock():
                cur = float(self._v.value)
                self._v.value = float(min(float(self._max_scale), cur * float(f)))
        except Exception:
            pass

    def request_scale_down(self, factor: float) -> None:
        try:
            f = float(factor)
        except Exception:
            f = 1.0
        if not (f > 0.0):
            return
        try:
            with self._v.get_lock():
                cur = float(self._v.value)
                self._v.value = float(max(float(self._min_scale), cur * float(f)))
        except Exception:
            pass


class Sampler(_Sampler):
    _per_sample_mem_bytes: int = 0

    @classmethod
    def _load_meta(cls: type["Sampler"], memmap_dir: str) -> Mapping[str, Any]:
        meta_path = os.path.join(os.fspath(memmap_dir), "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"meta.json not found under: {memmap_dir}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, Mapping):
            raise ValueError(f"meta.json under {memmap_dir} must contain a mapping")
        return meta

    def __init__(
        self,
        memmap_dir: str,
        *args: Any,
        split: str = "train",
        val_frac: float = 0.0,
        sampler_scale: Optional["BatchState"] = None,
        **kwargs: Any,
    ) -> None:
        self.dir = os.fspath(memmap_dir)
        self.split = str(split)
        self._meta: Mapping[str, Any] = self._load_meta(self.dir)

        # Per-session/per-loader scale controller (NOT global).
        self._sampler_scale = sampler_scale if sampler_scale is not None else BatchState()

        # Runtime dynamic-batch cap (computed in pipeline._batch_interval).
        # 0 => unknown/unlimited (will still be clamped by split end).
        self._S_B_cap = 0

        self._N = int(self._meta.get("N", 0))
        if self._N <= 0:
            raise ValueError(f"meta.json under {self.dir} has non-positive N={self._N}")

        feat_rel = str(self._meta.get("features_path", "features.mmt"))
        feat_path = os.path.join(self.dir, feat_rel)

        # labels_path may be omitted (features-only memmap).
        lab_rel_raw = self._meta.get("labels_path", "labels.mmt")
        features_only = bool(self._meta.get("features_only", False))

        lab_rel = ""
        lab_path = ""
        if lab_rel_raw not in (None, "", False):
            with suppress(Exception):
                lab_rel = str(lab_rel_raw)
        if lab_rel.strip().lower() in ("none", "null"):
            lab_rel = ""
        if lab_rel:
            lab_path = os.path.join(self.dir, lab_rel)
            if not os.path.isfile(lab_path):
                if features_only:
                    lab_rel = ""
                    lab_path = ""
                else:
                    raise FileNotFoundError(f"labels.mmt not found under: {lab_path}")

        fdim = int(self._meta.get("feature_dim", 0))
        lshape_meta = list(self._meta.get("label_shape") or [])

        f_dtype = dtype_from_name(self._meta.get("features_dtype", "float64"), torch.float64)
        l_dtype = dtype_from_name(self._meta.get("labels_dtype", "int64"), torch.int64)

        # Optional row_ids emission (metadata). Disable to avoid per-batch arange allocations.
        self._include_row_ids = env_bool("STNET_INCLUDE_ROW_IDS", default=True)

        # Keep raw memmap config for optional per-thread handles (no-GIL).
        self._feat_path = feat_path
        self._lab_path = lab_path if lab_path else None
        self._feat_dtype = f_dtype
        self._label_dtype = l_dtype
        self._feat_shape = torch.Size([self._N, fdim])

        if MemoryMappedTensor is None:
            raise ImportError(
                "tensordict is required for MemoryMappedTensor-backed pipelines. "
                "Please install 'tensordict' (or install stnet-pytorch with its default dependencies)."
            )

        self._features = MemoryMappedTensor.from_filename(
            filename=feat_path, dtype=f_dtype, shape=torch.Size([self._N, fdim])
        )

        lshape = tuple(lshape_meta) if lshape_meta else tuple()
        self._label_shape_full = torch.Size([self._N] + list(lshape))

        self._labels = None
        if self._lab_path is not None:
            self._labels = MemoryMappedTensor.from_filename(
                filename=str(self._lab_path),
                dtype=l_dtype,
                shape=torch.Size([self._N] + list(lshape)),
            )

        # Thread-local memmap handles are optional; default "auto" for no-GIL builds.
        self._mmap_tls: Optional[threading.local] = None
        self._mmap_init_lock: Optional[threading.Lock] = threading.Lock()

        # Instance-local limit lock (avoid global contention on no-GIL).
        self._mmap_limit_lock: Optional[threading.Lock] = threading.Lock()

        self._mmap_thread_local = False
        self._mmap_thread_local_max = 0
        self._mmap_thread_local_created = 0
        self._mmap_thread_local_overflow_warned = False

        default_tl = False
        with suppress(Exception):
            default_tl = bool(Thread.nogil_optimizations_enabled())
        self._mmap_thread_local = env_bool(
            (
                # Primary (code):
                "STNET_MEMMAP_THREAD_LOCAL_HANDLES",
                # Alias (README/backward-compat):
                "STNET_MEMMAP_THREAD_LOCAL",
                # Legacy: treat no-GIL toggle as opt-in.
                "STNET_NOGIL",
            ),
            default=default_tl,
        )

        if self._mmap_thread_local:
            cpu = int(process_cpu_count() or 8)
            default_max = max(8, min(64, cpu))
            self._mmap_thread_local_max = env_first_int(
                ("STNET_MEMMAP_THREAD_LOCAL_MAX", "STNET_MEMMAP_TL_MAX"),
                default=default_max,
            )
            self._mmap_thread_local_max = int(self._mmap_thread_local_max)

        self._memmap_features = self._features
        self._memmap_labels = self._labels
        self._X = self._memmap_features
        self._Y = self._memmap_labels

        # Base batch size stored separately; _S_B is a dynamic property that applies sampler_scale.
        self._S_B_base = 1
        self._S_B = 1  # initialize via setter for compatibility

        self._S_shuffle = True
        self._S_seed = 0
        self._S_epoch = 0

        # Epoch-local len snapshot (keeps __len__ stable even if sampler_scale changes mid-epoch).
        self._len_epoch = -1
        self._len_B_snapshot: Optional[int] = None

        self._num_shards = 1
        self._shard_id = 0
        self._key = ""
        self._label_shape: Tuple[int, ...] = tuple(lshape) if lshape else tuple()

        train_start = int(self._meta.get("train_start", 0))
        train_end = int(self._meta.get("train_end", self._N))
        val_start = int(self._meta.get("val_start", 0))
        val_end = int(self._meta.get("val_end", 0))

        if val_frac and not (val_end > val_start):
            vf = float(val_frac)
            vc = max(0, min(self._N, int(self._N * vf)))
            val_start, val_end = max(0, self._N - vc), self._N
            train_start, train_end = 0, val_start

        if self.split == "val":
            self._start, self._end = ((val_start, val_end) if val_end > val_start else (0, 0))
        else:
            self._start, self._end = (train_start, train_end)

    @property
    def sampler_scale(self) -> "BatchState":
        return self._sampler_scale

    @property
    def base_batch_size(self) -> int:
        try:
            b = int(getattr(self, "_S_B_base", 1) or 1)
        except Exception:
            b = 1
        return max(1, int(b))

    def _effective_batch_size(self) -> int:
        base = self.base_batch_size
        scale = 1.0
        with suppress(Exception):
            scale = float(self._sampler_scale.get())
        if not (scale > 0.0):
            scale = 1.0

        eff = int(round(float(base) * float(scale)))
        eff = max(1, int(eff))

        cap = 0
        with suppress(Exception):
            cap = int(getattr(self, "_S_B_cap", 0) or 0)
        if cap > 0:
            eff = min(int(eff), int(cap))
        return max(1, int(eff))

    def _len_batch_size(self) -> int:
        """Epoch-local snapshot for __len__ stability."""
        epoch = int(getattr(self, "_S_epoch", 0) or 0)
        snap_epoch = int(getattr(self, "_len_epoch", -1) or -1)
        snap = getattr(self, "_len_B_snapshot", None)
        if snap is None or snap_epoch != epoch:
            snap = max(1, int(self._effective_batch_size()))
            self._len_B_snapshot = int(snap)
            self._len_epoch = int(epoch)
        return max(1, int(snap))

    # Backward-compatible dynamic attribute:
    @property
    def _S_B(self) -> int:  # noqa: N802 (keep legacy internal name)
        return self._effective_batch_size()

    @_S_B.setter
    def _S_B(self, value: int) -> None:  # noqa: N802
        try:
            v = int(value)
        except Exception:
            v = 1
        if v <= 0:
            v = 1
        setattr(self, "_S_B_base", int(v))

    def __getstate__(self):
        """Make the Sampler picklable (spawn-based multiprocessing)."""
        state = dict(self.__dict__)
        for key in (
            "_features",
            "_labels",
            "_memmap_features",
            "_memmap_labels",
            "_X",
            "_Y",
            "_mmap_tls",
            "_mmap_init_lock",
            "_mmap_limit_lock",
        ):
            state.pop(key, None)
        state["_mmap_tls"] = None
        state["_mmap_init_lock"] = None
        state["_mmap_limit_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._mmap_tls = None
        self._mmap_init_lock = threading.Lock()
        self._mmap_limit_lock = threading.Lock()

        if MemoryMappedTensor is None:
            raise ImportError(
                "tensordict is required for MemoryMappedTensor-backed pipelines. "
                "Please install 'tensordict' (or install stnet-pytorch with its default dependencies)."
            )

        self._features = MemoryMappedTensor.from_filename(
            filename=str(self._feat_path),
            dtype=self._feat_dtype,
            shape=self._feat_shape,
        )
        self._labels = None
        if getattr(self, "_lab_path", None):
            self._labels = MemoryMappedTensor.from_filename(
                filename=str(self._lab_path),
                dtype=self._label_dtype,
                shape=self._label_shape_full,
            )
        self._memmap_features = self._features
        self._memmap_labels = self._labels
        self._X = self._memmap_features
        self._Y = self._memmap_labels

    def _get_mmaps(self):
        """Return (features, labels) handles for the current thread.

        On free-threading / no-GIL builds, some extension types may internally rely
        on per-object state. Thread-local handles avoid sharing that state across threads.
        """
        if not getattr(self, "_mmap_thread_local", False):
            return self._features, self._labels

        has_labels = bool(getattr(self, "_labels", None) is not None and getattr(self, "_lab_path", None))

        tls = getattr(self, "_mmap_tls", None)
        if tls is None:
            tls = threading.local()
            self._mmap_tls = tls

        f = getattr(tls, "features", None)
        l = getattr(tls, "labels", None)
        if f is not None and (not has_labels or l is not None):
            return f, (l if has_labels else None)

        max_pairs = int(getattr(self, "_mmap_thread_local_max", 0) or 0)
        if max_pairs > 0:
            lock = getattr(self, "_mmap_limit_lock", None)
            if lock is None:
                lock = threading.Lock()
                self._mmap_limit_lock = lock

            with lock:
                created = int(getattr(self, "_mmap_thread_local_created", 0) or 0)
                if created >= max_pairs:
                    if not bool(getattr(self, "_mmap_thread_local_overflow_warned", False)):
                        setattr(self, "_mmap_thread_local_overflow_warned", True)
                        with suppress(Exception):
                            _LOGGER.warning(
                                "[memmap] thread-local handle limit reached (%d). "
                                "Falling back to shared memmap handles for overflow threads. "
                                "Override: STNET_MEMMAP_THREAD_LOCAL_MAX / STNET_MEMMAP_TL_MAX.",
                                int(max_pairs),
                            )
                    return self._features, self._labels
                setattr(self, "_mmap_thread_local_created", created + 1)

        init_lock = getattr(self, "_mmap_init_lock", None)
        if init_lock is None:
            init_lock = threading.Lock()
            self._mmap_init_lock = init_lock

        with init_lock:
            f = getattr(tls, "features", None)
            l = getattr(tls, "labels", None)
            if f is None or (has_labels and l is None):
                try:
                    f = MemoryMappedTensor.from_filename(
                        filename=str(self._feat_path),
                        dtype=self._feat_dtype,
                        shape=self._feat_shape,
                    )
                    if has_labels:
                        l = MemoryMappedTensor.from_filename(
                            filename=str(self._lab_path),
                            dtype=self._label_dtype,
                            shape=self._label_shape_full,
                        )
                    else:
                        l = None
                except Exception:
                    # Undo created counter on failure.
                    if max_pairs > 0:
                        lock = getattr(self, "_mmap_limit_lock", None)
                        if lock is None:
                            lock = threading.Lock()
                            self._mmap_limit_lock = lock
                        with lock:
                            created = int(getattr(self, "_mmap_thread_local_created", 0) or 0)
                            setattr(self, "_mmap_thread_local_created", max(0, created - 1))
                    return self._features, self._labels

                setattr(tls, "features", f)
                if has_labels:
                    setattr(tls, "labels", l)

        return f, (l if has_labels else None)

    @property
    def start(self) -> int:
        return int(self._start)

    @property
    def end(self) -> int:
        return int(self._end)

    @property
    def meta(self) -> Mapping[str, Any]:
        return dict(self._meta or {})

    def _slice(self, start: int, end: int) -> Mapping[str, torch.Tensor]:
        start = int(start)
        end = int(end)
        features, labels = self._get_mmaps()

        x = features[start:end]
        xt = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

        out: Dict[str, torch.Tensor] = {"X": xt}
        if bool(getattr(self, "_include_row_ids", True)):
            out["row_ids"] = torch.arange(start, end, dtype=torch.int64)

        if labels is not None:
            y = labels[start:end]
            if self._label_shape:
                y = y.reshape(end - start, *self._label_shape)
            yt = y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
            out["Y"] = yt

        return out

    def _gather(self, idx_tensor: torch.Tensor, features: Any, labels: Any) -> Mapping[str, torch.Tensor]:
        """Gather rows by index tensor (host indices)."""
        idx_tensor = idx_tensor.to(dtype=torch.long, copy=False)
        try:
            x = features.index_select(0, idx_tensor)
        except Exception:
            x = features[idx_tensor] if hasattr(features, "__getitem__") else torch.as_tensor(features)[idx_tensor]

        out: Dict[str, torch.Tensor] = {"X": x}
        if bool(getattr(self, "_include_row_ids", True)):
            out["row_ids"] = idx_tensor.to(dtype=torch.int64, copy=False)

        if labels is not None:
            try:
                y = labels.index_select(0, idx_tensor)
            except Exception:
                y = labels[idx_tensor] if hasattr(labels, "__getitem__") else torch.as_tensor(labels)[idx_tensor]
            if self._label_shape:
                y = y.reshape(y.shape[0], *self._label_shape)
            out["Y"] = y

        return out

    def __getitem__(self, idx: int | Tuple[int, int] | Sequence[int] | torch.Tensor) -> Mapping[str, torch.Tensor]:
        """Index into the memmap dataset."""
        features, labels = self._get_mmaps()
        base = int(self._start)

        match idx:
            case tuple() as t if len(t) == 2:
                s, e = int(t[0]), int(t[1])
                return self._slice(base + s, base + e)

            case torch.Tensor() as t:
                idx_t = t.detach()
                if idx_t.device.type != "cpu":
                    idx_t = idx_t.to("cpu")
                if idx_t.ndim == 0:
                    return self.__getitem__(int(idx_t.item()))
                idx_tensor = idx_t.to(dtype=torch.long, copy=False).reshape(-1)
                if idx_tensor.numel() == 0:
                    return self._slice(base, base)
                idx_tensor = idx_tensor + base
                return self._gather(idx_tensor, features, labels)

            case seq if isinstance(seq, Sequence) and not isinstance(seq, (str, bytes, bytearray)):
                if len(seq) == 0:
                    return self._slice(base, base)
                idx_tensor = torch.as_tensor(seq, dtype=torch.long).reshape(-1) + base
                return self._gather(idx_tensor, features, labels)

            case _:
                i = base + int(idx)  # scalar
                out = self._slice(i, i + 1)
                # Squeeze scalars for ergonomic downstream use.
                with suppress(Exception):
                    x = out.get("X", None)
                    if torch.is_tensor(x):
                        out["X"] = x.squeeze(0)
                    y = out.get("Y", None)
                    if torch.is_tensor(y):
                        out["Y"] = y.squeeze(0)
                    r = out.get("row_ids", None)
                    if torch.is_tensor(r):
                        out["row_ids"] = r.squeeze(0)
                return out

    def _shard(self) -> None:
        try:
            dist = getattr(torch, "distributed", None)
            if dist is not None and dist.is_available() and dist.is_initialized():
                self._num_shards = max(1, int(dist.get_world_size()))
                self._shard_id = max(0, int(dist.get_rank()))
                return
        except Exception:
            pass

        world = env_first_int(("WORLD_SIZE",), default=1)
        rank = env_first_int(("RANK",), default=0)

        if world > 1:
            self._num_shards = max(1, int(world))
            self._shard_id = max(0, min(int(world) - 1, int(rank)))
        else:
            self._num_shards = 1
            self._shard_id = 0

    def compose(
        self,
        *args: Any,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        key: str = "",
        **kwargs: Any,
    ) -> "BaseNode":
        # Base batch size stored in _S_B_base via the legacy _S_B setter.
        self._S_B = max(1, int(batch_size))
        self._S_shuffle = bool(shuffle)
        self._S_seed = int(seed)
        self._S_epoch = 0
        self._key = str(key)

        # Initialize epoch-local len snapshot.
        self._len_epoch = int(self._S_epoch)
        self._len_B_snapshot = max(1, int(self._effective_batch_size()))

        self._shard()
        if SamplerWrapper is None:
            raise RuntimeError("torchdata.nodes.SamplerWrapper is required")
        return SamplerWrapper(self)

    def __iter__(self) -> Iterator[tuple[str, tuple[int, int]]]:
        start = int(getattr(self, "start", 0))
        end = int(getattr(self, "end", 0))
        if end <= start:
            return

        # Refresh distributed sharding each epoch/iteration.
        self._shard()

        # Refresh epoch-local len snapshot at the start of iteration.
        self._len_epoch = int(getattr(self, "_S_epoch", 0) or 0)
        self._len_B_snapshot = max(1, int(self._effective_batch_size()))

        ns = int(getattr(self, "_num_shards", 1) or 1)
        si = int(getattr(self, "_shard_id", 0) or 0)

        base = int(self.base_batch_size)
        max_scale = 2.0
        with suppress(Exception):
            max_scale = float(getattr(self._sampler_scale, "_max_scale", 2.0))
        block = max(1, int(math.ceil(float(base) * float(max_scale))))

        with suppress(Exception):
            cap = int(getattr(self, "_S_B_cap", 0) or 0)
            if cap > 0:
                block = min(int(block), int(cap))

        total = int(end - start)
        n_blocks = max(1, int((total + block - 1) // block))

        if bool(getattr(self, "_S_shuffle", True)):
            g = torch.Generator(device="cpu")
            g.manual_seed(int(getattr(self, "_S_seed", 0)) + int(getattr(self, "_S_epoch", 0)))
            order = torch.randperm(n_blocks, generator=g, dtype=torch.int64)
        else:
            order = torch.arange(n_blocks, dtype=torch.int64)

        if ns > 1:
            order = order[si::ns]

        for b in order:
            bs = start + int(b) * int(block)
            be = min(end, bs + int(block))
            cur = int(bs)
            while cur < int(be):
                # Iteration remains dynamic (for OOM recovery), even if __len__ is snapshot-stable.
                B = int(self._effective_batch_size())
                nxt = min(int(be), cur + max(1, int(B)))
                if nxt <= cur:
                    break
                yield (self._key, (int(cur), int(nxt)))
                cur = int(nxt)

    def __len__(self) -> int:
        start = int(getattr(self, "start", 0))
        end = int(getattr(self, "end", 0))
        B = max(1, int(self._len_batch_size()))
        total = max(0, (end - start + B - 1) // B)
        ns = int(getattr(self, "_num_shards", 1) or 1)
        si = int(getattr(self, "_shard_id", 0) or 0)
        if ns <= 1:
            return int(total)
        return max(0, (int(total) - si + ns - 1) // ns)

    def set_epoch(self, epoch: int) -> None:
        self._S_epoch = int(epoch)
        # Update epoch-local len snapshot.
        self._len_epoch = int(self._S_epoch)
        self._len_B_snapshot = max(1, int(self._effective_batch_size()))

    def get(self, start: int, end: int) -> Mapping[str, Any]:
        from ..core.graph import inference_mode as _inference_mode

        s = int(start)
        e = int(end)
        n = max(0, e - s)
        features, labels = self._get_mmaps()

        with _inference_mode(torch.nn.Module()):
            if n <= 0:
                X0 = features.narrow(0, 0, 0)
                out0: Dict[str, Any] = {"X": X0}
                if labels is not None:
                    out0["Y"] = labels.narrow(0, 0, 0)
                return out0

            X = features.narrow(0, s, n)
            out: Dict[str, Any] = {"X": X}

            if labels is not None:
                Y = labels.narrow(0, s, n)
                if self._label_shape:
                    Y = Y.reshape(n, *self._label_shape)
                out["Y"] = Y

            # Keep dataset-level pinning opt-in only; prefetcher typically owns pinning.
            pin_in_dataset = env_bool("STNET_PIN_IN_DATASET", default=False)
            if pin_in_dataset and _is_accelerator_available():
                with suppress(Exception):
                    out["X"] = out["X"].pin_memory()
                    if out.get("Y") is not None:
                        out["Y"] = out["Y"].pin_memory()

            return out



# -----------------------------------------------------------------------------
# BatchIterator: streaming + memmap utilities (shared across compute/elastic/nodes)
# -----------------------------------------------------------------------------

class BatchIterator:
    """Centralized memmap/streaming/chunking utilities."""

    class KeyIndexMappingView(_abc.Mapping):
        """A mapping-view over `data` that iterates in the provided `keys` order."""

        __slots__ = ("_data", "_keys")

        def __init__(self, data: Mapping[Any, Any], keys: Sequence[Any]) -> None:
            self._data = data
            self._keys = keys

        def __len__(self) -> int:
            return int(len(self._keys))

        def __iter__(self):
            return iter(self._keys)

        def __getitem__(self, k: Any) -> Any:
            return self._data[k]

    @staticmethod
    def key_index_mapping_getters(
        data: Mapping[Any, Any],
        *,
        keys: Optional[Sequence[Any]] = None,
    ) -> Tuple[int, Any]:
        """Build (count, get_batch) helpers for key-index mappings.

        IMPORTANT: This intentionally does *not* provide random access or writer-side shuffling.
        Shuffling should be handled by the sampler at read time.

        The returned get_batch() is optimized for sequential access patterns:
        - expects monotonically increasing (s, e) windows within a pass
        - supports automatic reset when s==0 (used by two-pass writers)
        """

        if keys is None:
            count = int(len(data))

            def _iter_keys() -> Any:
                return iter(data.keys())

        else:
            count = int(len(keys))

            def _iter_keys() -> Any:
                return iter(keys)

        if count <= 0:
            raise ValueError("Empty mapping: no keys")

        it = _iter_keys()
        pos = 0

        def get_batch(s: int, e: int):
            nonlocal it, pos
            s_i = int(s)
            e_i = int(e)
            if s_i < 0 or e_i < s_i:
                raise ValueError(f"invalid batch slice: s={s_i}, e={e_i}")

            # Two-pass writer resets to s==0 on the second pass.
            if s_i == 0 and pos != 0:
                it = _iter_keys()
                pos = 0

            # Sequential-only contract: avoids O(N^2) islice-based skipping.
            if s_i != pos:
                raise RuntimeError(
                    "key_index_mapping_getters: non-sequential access requested "
                    f"(expected s={pos}, got s={s_i}). "
                    "Disable writer-side shuffle; let the sampler handle shuffling."
                )

            need = e_i - s_i
            if need <= 0:
                return BatchIterator.KeyIndexMappingView(data, ())

            batch_keys: list[Any] = []
            for _ in range(int(need)):
                try:
                    k = next(it)
                except StopIteration:
                    break
                batch_keys.append(k)

            pos += int(len(batch_keys))
            return BatchIterator.KeyIndexMappingView(data, batch_keys)

        return count, get_batch

    @staticmethod
    def is_feature_label_batch_mapping(obj: Any) -> bool:
        """True if the object looks like a {features, labels} style batch mapping.

        The project historically accepted a mix of key names (e.g. X/Y, features/labels).
        We now treat feature/label column names case-insensitively and recognize:

        - features: x, feature, features, input, inputs, in
        - labels:   y, label, labels, output, outputs, out

        If the mapping uses non-string keys (e.g. {tuple_feature: label}), this returns False
        so it can fall back to the compact-dataset heuristic.
        """
        if not isinstance(obj, Mapping) or not obj:
            return False

        for k in obj.keys():
            if not isinstance(k, str):
                continue
            ck = k.casefold()
            if ck in _FEATURE_KEY_ALIASES or ck in _LABEL_KEY_ALIASES:
                return True
        return False

    @staticmethod
    def _resolve_memmap_store_float(*, negotiable: bool) -> torch.dtype:
        from ..core.casting import env_str

        req = str(env_str("STNET_MEMMAP_FLOAT_DTYPE") or "").strip()
        if req.startswith("torch."):
            req = req.split(".", 1)[1]
        req_dtype = getattr(torch, req, None) if req else None
        if not isinstance(req_dtype, torch.dtype):
            req_dtype = torch.float32
        try:
            if not torch.is_floating_point(torch.empty((), dtype=req_dtype)):
                req_dtype = torch.float32
        except Exception:
            req_dtype = torch.float32
        return torch.float32 if (bool(negotiable) and req_dtype != torch.float64) else torch.float64

    @staticmethod
    def _to_cpu_contig(t: torch.Tensor) -> torch.Tensor:
        t = t.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        if not t.is_contiguous():
            t = t.contiguous()
        return t

    @staticmethod
    def _flat2d_cpu_contig(t: torch.Tensor, n: int) -> torch.Tensor:
        t_cpu = BatchIterator._to_cpu_contig(t)
        if t_cpu.ndim == 0:
            t_cpu = t_cpu.reshape(1)
        return t_cpu.reshape(int(n), -1)

    @staticmethod
    def _batch_n(x: torch.Tensor) -> int:
        xd = int(getattr(x, "ndim", 0) or 0)
        return int(x.shape[0]) if xd > 0 else 1

    @staticmethod
    def _idx_to_cpu_int64(idx: Any) -> torch.Tensor:
        if not isinstance(idx, torch.Tensor):
            idx = torch.as_tensor(idx)
        if idx.device.type != "cpu":
            idx = idx.detach().cpu()
        if idx.dtype not in (torch.int64, torch.int32):
            idx = idx.to(dtype=torch.int64, copy=False)
        idx = idx.reshape(-1)
        return idx.to(dtype=torch.int64, copy=False)

    @staticmethod
    def mmt_meta_path(mmt_path: str) -> str:
        """Sidecar metadata path for a MemoryMappedTensor file."""
        return _schemas.mmt_meta_path(mmt_path)

    @staticmethod
    def atomic_write_json(path: str, payload: Any, *, indent: int | None = 2) -> None:
        """Atomically write JSON (best-effort).

        Delegates to stnet.data.schemas.atomic_write_json().
        """
        _schemas.atomic_write_json(path, payload, indent=indent)

    @staticmethod
    def _atomic_write_json(path: str, payload: Any, *, indent: int = 2) -> None:
        BatchIterator.atomic_write_json(path, payload, indent=int(indent))

    @staticmethod
    def atomic_torch_save(path: str, payload: Any, **opts: Any) -> None:
        """Atomically write a torch checkpoint file.

        Delegates to stnet.data.schemas.atomic_torch_save().
        """
        _schemas.atomic_torch_save(path, payload, **opts)

    @staticmethod
    def _atomic_torch_save(path: str, payload: Any, **opts: Any) -> None:
        BatchIterator.atomic_torch_save(path, payload, **opts)

    @staticmethod
    def write_memmap_streaming_two_pass(
        *,
        ds: Any,
        out_dir: str,
        count: int,
        get_batch: Any,
        val_frac: float,
        seed_value: Any,
        underflow_action: str,
        shuffle: bool = False,
        get_by_indices: Any = None,
        default_label_shape: Any = None,
        allow_missing_labels: bool = False,
        features_only: bool = False,
        chunk_size: int = 32,
    ) -> Tuple[int, Tuple[int, ...]]:
        """Write a memmap dataset in two passes.

        Pass 1: infer shapes and collect scale stats (for dtype negotiation).
        Pass 2: write contiguous memmaps (optionally shuffled).
        """
        out_dir = os.fspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        count_i = int(count)
        if count_i <= 0:
            raise ValueError("count must be > 0")

        env_chunk = env_first_int(("STNET_MEMMAP_CHUNK_SIZE", "STNET_MEMMAP_CHUNK"), None)
        if env_chunk is not None and int(env_chunk) > 0:
            chunk_size = int(env_chunk)

        req_chunk = int(chunk_size or 0)
        auto_chunk = req_chunk <= 0
        chunk_first = max(1, min(count_i, req_chunk if req_chunk > 0 else min(count_i, 256)))

        allow_missing = bool(allow_missing_labels) or bool(features_only)
        default_lshape = tuple(default_label_shape) if default_label_shape is not None else (1,)

        stats: Dict[str, Any] = {
            "has_scale": False,
            "has_nonfinite": False,
            "scale_max_abs": None,
            "scale_min_value": None,
            "scale_max_value": None,
            "scale_min_positive": None,
            "scale_is_integral": None,
        }
        in_dim: Optional[int] = None
        label_shape: Optional[Tuple[int, ...]] = None

        # --- Pass 1: infer shapes + stats ---
        for s in range(0, count_i, int(chunk_first)):
            e = min(count_i, s + int(chunk_first))
            batch = get_batch(int(s), int(e))
            fx, lb, _, _ = ds.preprocess(batch)
            n = BatchIterator._batch_n(fx)
            if n <= 0:
                continue
            expected = int(e) - int(s)
            if n != expected:
                raise RuntimeError(
                    f"Pass1 batch size mismatch for out_dir={out_dir!r}: expected {expected}, got {n} (s={s}, e={e})."
                )

            fx_flat = BatchIterator._flat2d_cpu_contig(fx, n)
            cur_in_dim = int(fx_flat.shape[1])
            if in_dim is None:
                in_dim = cur_in_dim
            elif cur_in_dim != int(in_dim):
                raise RuntimeError(f"feature dim mismatch: expected {in_dim}, got {cur_in_dim}")

            if lb is None:
                if not allow_missing:
                    raise RuntimeError("memmap writer requires labels (got None)")
                cur_label_shape = tuple(default_lshape)
                lb_flat = None
            else:
                cur_label_shape = tuple(lb.shape[1:])
                lb_flat = BatchIterator._flat2d_cpu_contig(lb, n)

            if label_shape is None:
                label_shape = cur_label_shape
            elif tuple(label_shape) != tuple(cur_label_shape):
                raise RuntimeError(f"label shape mismatch: expected {label_shape}, got {cur_label_shape}")

            f_stats = Dataset.tensor_scale_stats(fx_flat)
            if bool(features_only):
                stats = Dataset.merge_scale_stats(stats, f_stats)
            else:
                if lb_flat is None:
                    l_stats = {
                        "has_scale": True,
                        "has_nonfinite": False,
                        "scale_max_abs": 0.0,
                        "scale_min_value": 0.0,
                        "scale_max_value": 0.0,
                        "scale_min_positive": None,
                        "scale_is_integral": None,
                    }
                else:
                    l_stats = Dataset.tensor_scale_stats(lb_flat)
                stats = Dataset.merge_scale_stats(stats, Dataset.merge_scale_stats(f_stats, l_stats))

        if in_dim is None or label_shape is None:
            raise RuntimeError("Failed to infer in_dim/label_shape from data")

        negotiable = Dataset.is_fp32_castable(
            stats,
            underflow_action=underflow_action,
            safety_margin=1.0,
        )
        store_float = BatchIterator._resolve_memmap_store_float(negotiable=bool(negotiable))

        # Auto-tune chunk size for pass 2 (bound memory).
        if auto_chunk:
            elem_size = int(torch.empty((), dtype=store_float).element_size())
            label_numel = 0 if bool(features_only) else int(np.prod(label_shape))
            row_bytes = max(1, (int(in_dim) + int(label_numel)) * int(elem_size))

            target_bytes = env_first_int(("STNET_MEMMAP_CHUNK_BYTES",), None)
            if target_bytes is None:
                target_mb = env_first_int(("STNET_MEMMAP_CHUNK_MB",), 64)
                target_bytes = int(target_mb) * 1024 * 1024

            try:
                from ..core.system import Memory

                avail = int(Memory.available() or 0)
                if avail > 0:
                    target_bytes = int(min(int(target_bytes), max(8 * 1024 * 1024, avail // 16)))
            except Exception:
                pass

            chunk_second = int(max(1, min(count_i, max(32, int(target_bytes) // int(row_bytes)))))
        else:
            chunk_second = int(max(1, min(count_i, req_chunk)))

        val_count = max(0, min(count_i, int(round(count_i * float(val_frac)))))
        train_count = max(0, count_i - val_count)
        train_start, train_end = 0, int(train_count)
        val_start, val_end = int(train_end), int(train_end) + int(val_count)

        features_path = os.path.join(out_dir, "features.mmt")
        labels_path = os.path.join(out_dir, "labels.mmt")

        features_mmt = MemoryMappedTensor.empty(
            (count_i, int(in_dim)),
            dtype=store_float,
            filename=features_path,
            existsok=True,
        )

        write_labels = not bool(features_only)
        labels_mmt = None
        if write_labels:
            labels_mmt = MemoryMappedTensor.empty(
                (count_i, *tuple(label_shape)),
                dtype=store_float,
                filename=labels_path,
                existsok=True,
            )

        # Shuffle indexer
        shuffle_indexer = None
        shuffle_impl = "none"
        order: Optional[torch.Tensor] = None
        shuffle_seed: Optional[int] = None

        if bool(shuffle):
            if get_by_indices is None:
                raise ValueError("shuffle=True requires get_by_indices")

            max_elems = env_first_int(
                ("STNET_MEMMAP_RANDPERM_MAX_ELEMS", "STNET_MEMMAP_SHUFFLE_MAX_ELEMS"),
                5_000_000,
            )
            use_full = (max_elems is not None) and (count_i <= int(max_elems))
            seed_i = None if seed_value is None else (int(seed_value) & 0x7FFFFFFFFFFFFFFF)

            if use_full:
                g = None
                if seed_i is not None:
                    g = torch.Generator(device="cpu")
                    g.manual_seed(seed_i)
                shuffle_seed = seed_i
                order = torch.randperm(count_i, generator=g, dtype=torch.int64)

                def _idx(s: int, e: int) -> torch.Tensor:
                    assert order is not None
                    return order[int(s) : int(e)]

                shuffle_indexer = _idx
                shuffle_impl = "randperm"
            else:
                if seed_i is None:
                    seed_i = int(torch.randint(0, 2**63 - 1, (1,), dtype=torch.int64).item())
                shuffle_seed = seed_i

                # Feistel PRP on 2^k domain + cycle-walking into [0, count).
                k = max(1, int((count_i - 1)).bit_length())
                if (k % 2) == 1:
                    k += 1
                half = k // 2
                mask = (1 << half) - 1
                domain_mask = (1 << k) - 1 if k < 64 else 0xFFFFFFFFFFFFFFFF

                seed_u = torch.tensor(seed_i & 0xFFFFFFFFFFFFFFFF, dtype=torch.uint64)
                mask_u = torch.tensor(mask, dtype=torch.uint64)
                domain_u = torch.tensor(domain_mask, dtype=torch.uint64)
                count_u = torch.tensor(count_i, dtype=torch.uint64)

                k0 = seed_u ^ torch.tensor(0x9E3779B97F4A7C15, dtype=torch.uint64)
                k1 = seed_u ^ torch.tensor(0xBF58476D1CE4E5B9, dtype=torch.uint64)
                k2 = seed_u ^ torch.tensor(0x94D049BB133111EB, dtype=torch.uint64)
                k3 = seed_u ^ torch.tensor(0xD6E8FEB86659FD93, dtype=torch.uint64)
                round_keys = (k0, k1, k2, k3)

                _c_mul1 = torch.tensor(0x9E3779B97F4A7C15, dtype=torch.uint64)
                _c_mul2 = torch.tensor(0xC2B2AE3D27D4EB4F, dtype=torch.uint64)

                def _round_fn(r: torch.Tensor, rk: torch.Tensor) -> torch.Tensor:
                    x = (r ^ rk) & mask_u
                    x = (x * _c_mul1) & domain_u
                    x = (x ^ (x >> 33)) & domain_u
                    x = (x * _c_mul2) & domain_u
                    x = (x ^ (x >> 29)) & domain_u
                    return x & mask_u

                def _feistel(x: torch.Tensor) -> torch.Tensor:
                    x = x & domain_u
                    l = (x >> half) & mask_u
                    r = x & mask_u
                    for rk in round_keys:
                        f = _round_fn(r, rk)
                        l, r = r, (l ^ f) & mask_u
                    return (((l << half) | r) & domain_u)

                # Deterministic fallback permutation (affine mod count) in case
                # cycle-walking doesn't converge quickly for some domain/seed.
                def _gcd(a: int, b: int) -> int:
                    while b:
                        a, b = b, a % b
                    return abs(int(a))

                a0 = (seed_i | 1) % count_i
                if a0 == 0:
                    a0 = 1
                while _gcd(a0, count_i) != 1:
                    a0 = (a0 + 2) % count_i
                    if a0 == 0:
                        a0 = 1
                b0 = seed_i % count_i

                def _affine(pos: torch.Tensor) -> torch.Tensor:
                    p = pos.to(dtype=torch.int64)
                    return ((p * int(a0) + int(b0)) % int(count_i)).to(dtype=torch.int64)

                def _permute(pos: torch.Tensor) -> torch.Tensor:
                    x = pos.to(dtype=torch.uint64)
                    y = _feistel(x)
                    bad = y >= count_u

                    max_iter = 64
                    it = 0
                    while bool(bad.any()):
                        it += 1
                        if it > max_iter:
                            return _affine(pos)
                        y_bad = _feistel(y[bad])
                        y[bad] = y_bad
                        bad = y >= count_u
                    return y.to(dtype=torch.int64)

                def _idx(s: int, e: int) -> torch.Tensor:
                    pos = torch.arange(int(s), int(e), device="cpu", dtype=torch.int64)
                    return _permute(pos)

                shuffle_indexer = _idx
                shuffle_impl = "prp"

        # Optional scaler stats (train split only).
        compute_scaler_stats = bool(write_labels) and (not bool(allow_missing_labels))
        x_sum: Optional[torch.Tensor] = None
        x_sum_sq: Optional[torch.Tensor] = None
        y_sum: Optional[torch.Tensor] = None
        y_sum_sq: Optional[torch.Tensor] = None

        if compute_scaler_stats and int(train_end) > 0:
            x_sum = torch.zeros((int(in_dim),), dtype=torch.float64, device=torch.device("cpu"))
            x_sum_sq = torch.zeros((int(in_dim),), dtype=torch.float64, device=torch.device("cpu"))
            out_dim = int(np.prod(label_shape))
            y_sum = torch.zeros((int(out_dim),), dtype=torch.float64, device=torch.device("cpu"))
            y_sum_sq = torch.zeros((int(out_dim),), dtype=torch.float64, device=torch.device("cpu"))

        written = 0
        for s in range(0, count_i, int(chunk_second)):
            e = min(count_i, s + int(chunk_second))
            if shuffle_indexer is None:
                batch = get_batch(int(s), int(e))
            else:
                idx = shuffle_indexer(int(s), int(e))
                batch = get_by_indices(idx)

            fx, lb, _, _ = ds.preprocess(batch)
            n = BatchIterator._batch_n(fx)
            if n <= 0:
                continue
            expected = int(e) - int(s)
            if n != expected:
                raise RuntimeError(
                    f"Pass2 batch size mismatch for out_dir={out_dir!r}: expected {expected}, got {n} (s={s}, e={e})."
                )

            fx_flat = BatchIterator._flat2d_cpu_contig(fx, n)
            if int(fx_flat.shape[1]) != int(in_dim):
                raise RuntimeError(
                    f"feature dim mismatch: expected {in_dim}, got {int(fx_flat.shape[1])}"
                )

            fx_out = fx_flat if fx_flat.dtype == store_float else fx_flat.to(dtype=store_float)

            if x_sum is not None and x_sum_sq is not None:
                end_pos = int(s) + int(n)
                overlap = max(0, min(end_pos, int(train_end)) - int(s))
                if overlap > 0:
                    fx_slice = fx_out[:overlap]
                    fx64 = fx_slice if fx_slice.dtype == torch.float64 else fx_slice.to(dtype=torch.float64)
                    x_sum += fx64.sum(dim=0)
                    x_sum_sq += fx64.square().sum(dim=0)

            features_mmt[int(s) : int(s) + int(n)].copy_(fx_out)

            if write_labels:
                assert labels_mmt is not None
                if lb is None:
                    if not allow_missing:
                        raise RuntimeError("memmap writer requires labels (got None)")
                    # Avoid large pre-allocated zero buffers: write zeros directly.
                    labels_mmt[int(s) : int(s) + int(n)].zero_()
                else:
                    if tuple(lb.shape[1:]) != tuple(label_shape):
                        raise RuntimeError(
                            f"label shape mismatch: expected {label_shape}, got {tuple(lb.shape[1:])}"
                        )
                    lb_cpu = BatchIterator._to_cpu_contig(lb)
                    lb_out = lb_cpu if lb_cpu.dtype == store_float else lb_cpu.to(dtype=store_float)

                    labels_mmt[int(s) : int(s) + int(n)].copy_(lb_out)

                    if y_sum is not None and y_sum_sq is not None:
                        end_pos = int(s) + int(n)
                        overlap = max(0, min(end_pos, int(train_end)) - int(s))
                        if overlap > 0:
                            lb_slice = lb_out[:overlap].reshape(int(overlap), -1)
                            lb64 = lb_slice if lb_slice.dtype == torch.float64 else lb_slice.to(dtype=torch.float64)
                            y_sum += lb64.sum(dim=0)
                            y_sum_sq += lb64.square().sum(dim=0)

            written += int(n)

        if int(written) != int(count_i):
            raise RuntimeError(f"memmap written={written}, expected={count_i}")

        scaler_stats_path: Optional[str] = None
        if (
            compute_scaler_stats
            and int(train_end) > 0
            and x_sum is not None
            and x_sum_sq is not None
            and y_sum is not None
            and y_sum_sq is not None
        ):
            payload = {
                "version": 1,
                "train_count": int(train_end),
                "x_sum": x_sum,
                "x_sum_sq": x_sum_sq,
                "y_sum": y_sum,
                "y_sum_sq": y_sum_sq,
            }
            scaler_stats_path = "scaler_stats.pt"
            try:
                BatchIterator._atomic_torch_save(os.path.join(out_dir, scaler_stats_path), payload)
            except Exception:
                scaler_stats_path = None

        meta_json: Dict[str, Any] = {
            "N": int(count_i),
            "feature_dim": int(in_dim),
            "features_path": "features.mmt",
            "labels_path": ("labels.mmt" if write_labels else None),
            "label_shape": list(label_shape),
            "features_dtype": str(store_float).replace("torch.", ""),
            "labels_dtype": (str(store_float).replace("torch.", "") if write_labels else None),
            "fractions": [float(1.0 - float(val_frac)), float(val_frac)],
            "shuffled": bool(shuffle),
            "shuffle_seed": int(shuffle_seed) if shuffle_seed is not None else None,
            "shuffle_mode": "physical" if bool(shuffle) else "none",
            "shuffle_impl": shuffle_impl,
            "train_start": int(train_start),
            "train_end": int(train_end),
            "val_start": int(val_start),
            "val_end": int(val_end),
            "scaler_stats_path": scaler_stats_path,
            "has_scale": bool(stats.get("has_scale")),
            "has_nonfinite": bool(stats.get("has_nonfinite")),
            "scale_max_abs": stats.get("scale_max_abs"),
            "scale_min_value": stats.get("scale_min_value"),
            "scale_max_value": stats.get("scale_max_value"),
            "scale_min_positive": stats.get("scale_min_positive"),
            "scale_is_integral": stats.get("scale_is_integral"),
            "is_negotiable": bool(negotiable),
            "underflow_action": str(underflow_action),
            "features_only": bool(features_only),
        }

        BatchIterator._atomic_write_json(os.path.join(out_dir, "meta.json"), meta_json, indent=2)
        return int(in_dim), tuple(label_shape)

    @staticmethod
    def preload_memmap(
        data: Mapping[str, Any],
        *,
        memmap_dir: str,
        val_frac: float = 0.0,
        shuffle: bool = False,
        seed: Optional[int] = None,
        underflow_action: Optional[str] = None,
        chunk_size: int = 4096,
        allow_missing_labels: bool = False,
        features_only: bool = False,
        default_label_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Materialize an in-memory dataset into a memmap directory."""
        if not isinstance(data, Mapping):
            raise TypeError("preload_memmap expects a Mapping with at least 'features'")
        if "features" not in data:
            raise ValueError("preload_memmap expects 'features'")

        raw_X = data["features"]
        raw_Y = data.get("labels")

        def _len0(obj: Any) -> int:
            if isinstance(obj, torch.Tensor):
                return int(obj.shape[0]) if getattr(obj, "ndim", 0) > 0 else 1
            try:
                return int(len(obj))
            except Exception:
                t = torch.as_tensor(obj)
                return int(t.shape[0]) if getattr(t, "ndim", 0) > 0 else 1

        count = _len0(raw_X)
        if count <= 0:
            raise ValueError("cannot create memmap with zero samples")
        if not bool(features_only):
            if raw_Y is None:
                if not bool(allow_missing_labels):
                    raise ValueError("preload_memmap expects 'labels' unless allow_missing_labels=True")
            else:
                if _len0(raw_Y) != int(count):
                    raise ValueError("features and labels must have the same length")

        ua = normalize_underflow_action(underflow_action, default=default_underflow_action())

        ds = Dataset.for_device("cpu", feature_dtype=torch.float64, label_float_dtype=torch.float64)
        ds.underflow_action = ua

        def _slice_any(obj: Any, s: int, e: int, *, name: str) -> Any:
            if obj is None:
                return None
            if torch.is_tensor(obj):
                return obj[int(s) : int(e)]
            if isinstance(obj, np.ndarray):
                return obj[int(s) : int(e)]
            try:
                return obj[int(s) : int(e)]
            except Exception:
                pass
            try:
                return [obj[i] for i in range(int(s), int(e))]
            except Exception as ex:
                raise TypeError(f"Object {name} does not support slicing [{s}:{e}]") from ex

        def _gather_any(obj: Any, idx: torch.Tensor, *, name: str) -> Any:
            if obj is None:
                return None
            idx_cpu = BatchIterator._idx_to_cpu_int64(idx)
            if torch.is_tensor(obj):
                return obj[idx_cpu]
            if isinstance(obj, np.ndarray):
                return obj[idx_cpu.numpy()]
            try:
                return obj[idx_cpu.numpy()]
            except Exception:
                pass
            try:
                return [obj[int(i)] for i in idx_cpu.tolist()]
            except Exception as ex:
                raise TypeError(f"Object {name} does not support gather by indices") from ex

        def get_batch(s: int, e: int) -> Mapping[str, Any]:
            out: Dict[str, Any] = {"features": _slice_any(raw_X, s, e, name="features")}
            if raw_Y is not None and not bool(features_only):
                out["labels"] = _slice_any(raw_Y, s, e, name="labels")
            return out

        def get_by_indices(idx: torch.Tensor) -> Mapping[str, Any]:
            out: Dict[str, Any] = {"features": _gather_any(raw_X, idx, name="features")}
            if raw_Y is not None and not bool(features_only):
                out["labels"] = _gather_any(raw_Y, idx, name="labels")
            return out

        BatchIterator.write_memmap_streaming_two_pass(
            ds=ds,
            out_dir=os.fspath(memmap_dir),
            count=int(count),
            get_batch=get_batch,
            get_by_indices=get_by_indices,
            val_frac=float(val_frac),
            seed_value=int(seed) if seed is not None else None,
            underflow_action=str(ua),
            shuffle=bool(shuffle),
            allow_missing_labels=bool(allow_missing_labels),
            features_only=bool(features_only),
            default_label_shape=tuple(default_label_shape) if default_label_shape is not None else None,
            chunk_size=int(chunk_size),
        )
        return None

    @staticmethod
    def iter_source_paths(obj: Any):
        if obj is None:
            return
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, dict):
            if obj.get("kind") == "memmap" and isinstance(obj.get("path"), str):
                yield obj["path"]
            else:
                for v in obj.values():
                    yield from BatchIterator.iter_source_paths(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from BatchIterator.iter_source_paths(v)

    @staticmethod
    def from_meta(memmap_dir: str) -> Dict[str, Any]:
        import json as _json

        meta_path = os.path.join(os.fspath(memmap_dir), "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            raw = _json.load(f)
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def merge_meta_dicts(metas: list[dict]) -> dict:
        if not metas:
            return {}
        base = dict(metas[0])

        def _strictest_underflow(a: Optional[str], b: Optional[str]) -> Optional[str]:
            order = {"allow": 0, "warn": 1, "forbid": 2}
            if a is None:
                return b
            if b is None:
                return a
            return a if order.get(a, 1) >= order.get(b, 1) else b

        feature_dim = base.get("feature_dim")
        label_shape = base.get("label_shape")

        def _has_scale(m: dict) -> bool:
            return bool(m.get("has_scale", False)) or any(
                m.get(k) is not None
                for k in (
                    "scale_max_abs",
                    "scale_min_value",
                    "scale_max_value",
                    "scale_min_positive",
                )
            )

        has_scale = _has_scale(base)
        has_nonfinite = bool(base.get("has_nonfinite", False))
        max_abs = base.get("scale_max_abs")
        min_val = base.get("scale_min_value")
        max_val = base.get("scale_max_value")
        min_pos = base.get("scale_min_positive")
        is_integral = base.get("scale_is_integral")
        is_negotiable = base.get("is_negotiable")
        underflow_action = base.get("underflow_action")

        for m in metas[1:]:
            if feature_dim is not None and m.get("feature_dim") is not None:
                if int(m.get("feature_dim")) != int(feature_dim):
                    raise ValueError(
                        f"feature_dim mismatch across sources: {feature_dim} vs {m.get('feature_dim')}"
                    )
            if label_shape is not None and m.get("label_shape") is not None:
                if tuple(m.get("label_shape")) != tuple(label_shape):
                    raise ValueError(
                        f"label_shape mismatch across sources: {label_shape} vs {m.get('label_shape')}"
                    )

            has_scale = has_scale or _has_scale(m)
            has_nonfinite = has_nonfinite or bool(m.get("has_nonfinite", False))

            a = m.get("scale_max_abs")
            if a is not None:
                max_abs = a if max_abs is None else max(float(max_abs), float(a))

            mn = m.get("scale_min_value")
            if mn is not None:
                try:
                    min_val = mn if min_val is None else (mn if mn <= min_val else min_val)
                except Exception:
                    min_val = mn if min_val is None else min(float(min_val), float(mn))

            mx = m.get("scale_max_value")
            if mx is not None:
                try:
                    max_val = mx if max_val is None else (mx if mx >= max_val else max_val)
                except Exception:
                    max_val = mx if max_val is None else max(float(max_val), float(mx))

            p = m.get("scale_min_positive")
            if p is not None:
                min_pos = p if min_pos is None else min(float(min_pos), float(p))

            i = m.get("scale_is_integral")
            if i is not None:
                is_integral = bool(i) if is_integral is None else bool(is_integral) and bool(i)

            n = m.get("is_negotiable")
            if n is not None:
                is_negotiable = bool(n) if is_negotiable is None else bool(is_negotiable) and bool(n)

            underflow_action = _strictest_underflow(
                str(underflow_action) if underflow_action is not None else None,
                str(m.get("underflow_action")) if m.get("underflow_action") is not None else None,
            )

        base["has_scale"] = bool(has_scale)
        base["has_nonfinite"] = bool(has_nonfinite)
        base["scale_max_abs"] = max_abs
        base["scale_min_value"] = min_val
        base["scale_max_value"] = max_val
        base["scale_min_positive"] = min_pos
        base["scale_is_integral"] = is_integral
        base["is_negotiable"] = is_negotiable
        base["underflow_action"] = underflow_action
        return base

    @staticmethod
    def merge_meta_infos(sources: Any) -> Dict[str, Any]:
        sources = BatchIterator.expand_sources(sources)
        metas: list[dict] = []
        for path in BatchIterator.iter_source_paths(sources):
            try:
                metas.append(BatchIterator.from_meta(path))
            except Exception:
                continue
        if not metas:
            return {}
        return dict(BatchIterator.merge_meta_dicts(metas))

    @staticmethod
    def load_scaler_stats(sources: Any) -> Optional[Dict[str, Any]]:
        expanded = BatchIterator.expand_sources(sources)
        total = 0
        x_sum: Optional[torch.Tensor] = None
        x_sum_sq: Optional[torch.Tensor] = None
        y_sum: Optional[torch.Tensor] = None
        y_sum_sq: Optional[torch.Tensor] = None

        for path in BatchIterator.iter_source_paths(expanded):
            try:
                meta = BatchIterator.from_meta(path)
            except Exception:
                return None
            rel = meta.get("scaler_stats_path")
            if not rel:
                return None
            stats_path = os.path.join(os.fspath(path), os.fspath(rel))
            if not os.path.isfile(stats_path):
                return None
            try:
                payload = torch.load(stats_path, map_location="cpu", weights_only=True)
            except TypeError:
                # Older PyTorch: no weights_only support.
                payload = torch.load(stats_path, map_location="cpu")
            except Exception:
                return None
            if not isinstance(payload, dict):
                return None
            if int(payload.get("version") or 0) != 1:
                return None
            c = int(payload.get("train_count") or 0)
            if c <= 0:
                return None

            xs = payload.get("x_sum")
            xs2 = payload.get("x_sum_sq")
            ys = payload.get("y_sum")
            ys2 = payload.get("y_sum_sq")
            if xs is None or xs2 is None or ys is None or ys2 is None:
                return None

            xs = xs.detach().to(dtype=torch.float64, device="cpu") if isinstance(xs, torch.Tensor) else torch.as_tensor(xs, dtype=torch.float64)
            xs2 = xs2.detach().to(dtype=torch.float64, device="cpu") if isinstance(xs2, torch.Tensor) else torch.as_tensor(xs2, dtype=torch.float64)
            ys = ys.detach().to(dtype=torch.float64, device="cpu") if isinstance(ys, torch.Tensor) else torch.as_tensor(ys, dtype=torch.float64)
            ys2 = ys2.detach().to(dtype=torch.float64, device="cpu") if isinstance(ys2, torch.Tensor) else torch.as_tensor(ys2, dtype=torch.float64)

            if x_sum is None:
                x_sum = xs.clone()
                x_sum_sq = xs2.clone()
                y_sum = ys.clone()
                y_sum_sq = ys2.clone()
            else:
                if xs.shape != x_sum.shape or xs2.shape != x_sum_sq.shape:
                    return None
                if ys.shape != y_sum.shape or ys2.shape != y_sum_sq.shape:
                    return None
                x_sum += xs
                x_sum_sq += xs2
                y_sum += ys
                y_sum_sq += ys2

            total += c

        if total <= 0 or x_sum is None or x_sum_sq is None or y_sum is None or y_sum_sq is None:
            return None

        return {
            "train_count": int(total),
            "x_sum": x_sum,
            "x_sum_sq": x_sum_sq,
            "y_sum": y_sum,
            "y_sum_sq": y_sum_sq,
        }

    @staticmethod
    def expand_sources(sources: Any) -> Any:
        import json as _json

        def _expand_from_root(spec: Any) -> Tuple[Any, bool]:
            if not isinstance(spec, dict) or "path" not in spec or "kind" not in spec:
                return spec, False
            root = os.fspath(spec.get("path") or "")
            mn_path = os.path.join(root, "multinode.json")
            if not os.path.isfile(mn_path):
                return spec, False
            with open(mn_path, "r", encoding="utf-8") as f:
                payload = _json.load(f)
            if isinstance(payload, dict):
                resolved = {
                    str(k): {"kind": "memmap", "path": os.path.join(root, str(v))}
                    for k, v in payload.items()
                }
                return resolved, True
            if isinstance(payload, list):
                resolved = [{"kind": "memmap", "path": os.path.join(root, str(v))} for v in payload]
                return resolved, True
            return spec, False

        expanded, ok = _expand_from_root(sources)
        if ok:
            return expanded
        if isinstance(sources, (list, tuple)) and len(sources) == 1:
            expanded, ok = _expand_from_root(sources[0])
            if ok:
                return expanded
        return sources


def preload_memmap(
    data: Mapping[str, Any],
    *,
    memmap_dir: str,
    train_frac: float = 0.0,
    val_frac: float = 0.0,
    shuffle: bool = False,
    seed: int | None = None,
    **kwargs: Any,
) -> None:
    """Create a memory-mapped dataset on disk.

    Backwards-compat shim; implementation centralized in stnet.data.pipeline.BatchIterator.

    Notes:
      - `shuffle=True` performs *physical* shuffle at write time (no perm file).
      - `train_frac` is currently metadata-only; the split is driven by `val_frac`.
    """
    chunk_size = int(kwargs.pop("chunk_size", 4096) or 4096)
    with suppress(Exception):
        env_cs = int(env_first_int(("STNET_MEMMAP_CHUNK_SIZE",), default=0) or 0)
        if env_cs > 0:
            chunk_size = max(1, int(env_cs))

    underflow_action = kwargs.pop("underflow_action", None)
    allow_missing_labels = bool(kwargs.pop("allow_missing_labels", False))
    features_only = bool(kwargs.pop("features_only", False))
    default_label_shape = kwargs.pop("default_label_shape", None)

    BatchIterator.preload_memmap(
        data,
        memmap_dir=os.fspath(memmap_dir),
        val_frac=float(val_frac),
        shuffle=bool(shuffle),
        seed=int(seed) if seed is not None else None,
        underflow_action=underflow_action,
        chunk_size=int(chunk_size),
        allow_missing_labels=bool(allow_missing_labels),
        features_only=bool(features_only),
        default_label_shape=tuple(default_label_shape) if default_label_shape is not None else None,
    )
    return None


SourceType = Literal["memmap"]


class Source(TypedDict):
    format: SourceType
    path: str


def _flatten_args(items: Sequence[Any]) -> Iterator[Any]:
    for item in items:
        if isinstance(item, Mapping):
            yield from _flatten_args(list(item.values()))
        elif isinstance(item, (list, tuple, set)):
            yield from _flatten_args(list(item))
        else:
            yield item


class Disposable:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._keep: list[Any] = list(_flatten_args(list(args)))
        if kwargs:
            self._keep.extend(list(_flatten_args(list(kwargs.values()))))

    def add(self, *args: Any, **kwargs: Any) -> None:
        self._keep.extend(list(_flatten_args(list(args))))
        if kwargs:
            self._keep.extend(list(_flatten_args(list(kwargs.values()))))

    def cleanup(self) -> None:
        for obj in self._keep:
            best_effort_close(obj)

    def close(self) -> None:
        self.cleanup()

    def __iter__(self) -> Iterator[Any]:
        return iter(self._keep)


class Wrapper:
    def __init__(
        self,
        *args: Any,
        stop_criteria: str = "ALL_DATASETS_EXHAUSTED",
        weights: Optional[Mapping[str, float]] = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        self.stop_criteria = str(stop_criteria)
        self.weights = dict(weights) if isinstance(weights, Mapping) else None
        self.seed = int(seed)
        self._epoch = 0
        self._node: Optional[Any] = None
        self._source_keys: list[str] = []

    def set_epoch(self, epoch: int) -> None:
        """Per-epoch reseed for multi-source mixing (torchdata MultiNodeWeightedSampler)."""
        self._epoch = int(epoch)
        node = getattr(self, "_node", None)
        if node is None:
            return

        reset = getattr(node, "reset", None)
        if not callable(reset):
            return

        def _key(attr: str, fallback: str) -> str:
            k = getattr(node, attr, None) or getattr(type(node), attr, None)
            return k if isinstance(k, str) else fallback

        epoch_key = _key("EPOCH_KEY", "epoch")
        ws_key = _key("WEIGHTED_SAMPLER_STATE_KEY", "weighted_sampler_state")
        ny_key = _key("NUM_YIELDED_KEY", "num_yielded")
        ex_key = _key("DATASETS_EXHAUSTED_KEY", "datasets_exhausted")
        dns_key = _key("DATASET_NODE_STATES_KEY", "dataset_node_states")

        keys = list(getattr(self, "_source_keys", []) or [])
        initial_state: Dict[str, Any] = {epoch_key: int(self._epoch), ny_key: 0, ws_key: None}
        if keys:
            initial_state[ex_key] = {k: False for k in keys}
            initial_state[dns_key] = {k: None for k in keys}

        try:
            reset(initial_state)
            return
        except Exception:
            pass

        with suppress(Exception):
            setattr(node, "seed", int(self.seed) + int(self._epoch))
        with suppress(Exception):
            reset(None)

    def compose(self, sources: Mapping[str, "BaseNode"] | Sequence["BaseNode"] | "BaseNode") -> "BaseNode":
        if isinstance(sources, BaseNode):
            return sources
        if isinstance(sources, (list, tuple)):
            if len(sources) == 1:
                return sources[0]
            sources_map = {str(i): n for i, n in enumerate(sources)}
        elif isinstance(sources, Mapping):
            sources_map = dict(sources)
            if len(sources_map) == 1:
                return next(iter(sources_map.values()))
        else:
            raise TypeError("sources must be a BaseNode, Sequence[BaseNode], or Mapping[str, BaseNode]")

        if MultiNodeWeightedSampler is None:
            raise RuntimeError("torchdata.nodes.MultiNodeWeightedSampler is required for multi-source mixing")

        w = self.weights or {k: 1.0 for k in sources_map}
        self._source_keys = list(sources_map.keys())
        node = MultiNodeWeightedSampler(sources_map, w, stop_criteria=self.stop_criteria, seed=int(self.seed))
        self._node = node
        return node


@dataclass(frozen=True)
class _MapBatch:
    """Picklable adapter for prebatch mapping (avoid nested closures)."""

    fn: Callable[[Any], Any]

    def __call__(self, x: Any) -> Any:
        return self.fn(x)


class Connector:
    def __init__(
        self,
        *args: Any,
        map_fn: Callable[[Any], Any],
        io_workers: Optional[int] = None,
        prebatch: Optional[int] = None,
        prefetch_factor: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
        non_blocking: bool = True,
        pin_memory: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        self.map_fn = map_fn
        from ..core.system import WorkerPolicy

        wp = WorkerPolicy.autotune()
        wp.apply_torch_threads()

        self.io_workers = int(io_workers) if io_workers is not None else int(getattr(wp, "num_workers", 1))
        self.io_workers = max(1, self.io_workers)

        self.prebatch = (
            int(prebatch) if prebatch is not None else int(getattr(wp, "prebatch", max(1, self.io_workers * 2)))
        )
        with suppress(Exception):
            self.prebatch = max(1, int(self.prebatch))

        pf = int(prefetch_factor) if prefetch_factor is not None else int(getattr(wp, "prefetch_factor", 1))
        with suppress(Exception):
            pf = max(1, int(pf))
        self._prefetch_factor = pf
        self.prefetch_factor = self._prefetch_factor

        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.non_blocking = bool(non_blocking)

        pin = bool(pin_memory) if pin_memory is not None else (getattr(self.device, "type", "cpu") in {"cuda", "xpu"})
        self._pin_memory = pin
        self.pin_memory = self._pin_memory

        self.max_concurrency = max(1, int(self.io_workers))
        try:
            from ..core.system import get_tlb

            get_tlb(io_workers=self.io_workers)
        except Exception:
            pass

    def compose(self, source: "BaseNode") -> "BaseNode":
        from ..core.system import wrap_with_tlb

        if ParallelMapper is None:
            raise RuntimeError("torchdata.nodes.ParallelMapper is required")

        node: BaseNode = source
        mapper = self.map_fn

        if (self.prebatch or 0) and int(self.prebatch) > 1:
            if _Batcher is None or _Unbatcher is None:
                raise RuntimeError("torchdata.nodes Batcher/Unbatcher are required for prebatch>1")

            B = max(1, int(self.prebatch))
            node = _Batcher(node, batch_size=B, drop_last=False)

            pm_map_fn = wrap_with_tlb(_MapBatch(mapper))
        else:
            pm_map_fn = wrap_with_tlb(mapper)

        node = ParallelMapper(
            node,
            map_fn=pm_map_fn,
            num_workers=self.io_workers,
            in_order=False,
            method="thread",
            max_concurrent=int(self.max_concurrency),
        )

        if (self.prebatch or 0) and int(self.prebatch) > 1:
            node = _Unbatcher(node)

        return node


def _normalize_device_spec(device: torch.device | str | Sequence[torch.device | str]) -> torch.device | list[torch.device]:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    if isinstance(device, Sequence) and not isinstance(device, (str, bytes, bytearray)):
        devs: list[torch.device] = []
        for d in device:
            devs.append(d if isinstance(d, torch.device) else torch.device(str(d)))
        return devs if devs else torch.device("cpu")
    return torch.device(device)  # type: ignore[arg-type]


def _primary_device(device_spec: torch.device | list[torch.device]) -> torch.device:
    return device_spec[0] if isinstance(device_spec, list) and device_spec else device_spec


class Loader:
    @staticmethod
    def compose(
        source: "BaseNode",
        *args: Any,
        device: torch.device | str | Sequence[torch.device | str],
        prefetch_factor: int = 2,
        non_blocking: bool = True,
        length: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        **kwargs: Any,
    ) -> "Loader":
        if not _TORCHDATA_AVAILABLE or _Loader is None:
            raise RuntimeError("torchdata is required to compose a Loader (torchdata.nodes.Loader).")
        if not isinstance(source, BaseNode):
            raise TypeError("Loader.compose expects a torchdata.nodes.BaseNode source.")

        return Loader(
            device=device,
            node=source,
            prefetch_factor=int(prefetch_factor),
            non_blocking=bool(non_blocking),
            length=length,
            pin_memory=pin_memory,
        )

    def __init__(
        self,
        device: torch.device | str | Sequence[torch.device | str],
        *args: Any,
        node: BaseNode | None = None,
        dataset: BaseNode | None = None,
        prefetch_factor: int = 2,
        non_blocking: bool = True,
        length: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        if not _TORCHDATA_AVAILABLE or _Loader is None:
            raise RuntimeError("torchdata is required to construct Loader (torchdata.nodes.Loader).")

        node_obj = node or dataset
        if not isinstance(node_obj, BaseNode):
            raise TypeError("Loader supports only torchdata.nodes.BaseNode instances.")

        self._device = _normalize_device_spec(device)
        self._non_blocking = bool(non_blocking)
        self._length = int(length) if length is not None else None

        # Interpret prefetch_factor as a bounded device-transfer prefetch depth (compat-friendly).
        depth = max(1, int(prefetch_factor))
        with suppress(Exception):
            depth_env = int(env_first_int(("STNET_PREFETCH_DEPTH",), default=0) or 0)
            if depth_env > 0:
                depth = int(depth_env)
        self._prefetch_depth = max(1, min(32, int(depth)))

        # Pin host only where it meaningfully helps async H2D.
        prim = _primary_device(self._device)
        dev_t = getattr(prim, "type", "cpu")
        default_pin = dev_t in {"cuda", "xpu"}
        self._pin_host = bool(pin_memory) if pin_memory is not None else bool(default_pin)

        # Memory guards: defaults are conservative.
        if dev_t == "cuda" and self._non_blocking:
            gpu_guard_mb = 2048
        elif dev_t in {"xpu"} and self._non_blocking:
            gpu_guard_mb = 512
        else:
            gpu_guard_mb = 0
        host_guard_mb = 1024 if self._non_blocking else 0

        with suppress(Exception):
            gpu_guard_mb = int(env_first_int(("STNET_GPU_GUARD_MB",), default=gpu_guard_mb) or gpu_guard_mb)
        with suppress(Exception):
            host_guard_mb = int(env_first_int(("STNET_HOST_GUARD_MB",), default=host_guard_mb) or host_guard_mb)

        self._gpu_guard_bytes = int(max(0, gpu_guard_mb) * (1 << 20))
        self._host_guard_bytes = int(max(0, host_guard_mb) * (1 << 20))

        # Base iterable: torchdata Loader wrapper.
        self._node = node_obj
        self._base_iterable = node_obj if isinstance(node_obj, _Loader) else _Loader(node_obj)

        # Multi-device mapping (thread-local)
        self._thread2dev: Dict[int, torch.device] = {}

        # Best-effort thread hint for sharding heuristics.
        self._threads_hint = self._infer_mapper_threads(node_obj) if node_obj is not None else 1

        # Sharding hints (used by upstream graphs occasionally)
        self._num_shards = 1
        self._shard_id = 0
        try:
            from ..core.system import num_accelerators

            acc = max(1, int(num_accelerators()))
            thr = max(1, int(self._threads_hint))
            self._num_shards = acc * thr
            dev_idx = self._local_device_index()
            self._shard_id = max(0, min(self._num_shards - 1, int(dev_idx * thr)))
        except Exception:
            pass

    def __iter__(self) -> Iterator[Any]:
        dev = self._device_for_current_thread()
        dev_t = getattr(dev, "type", "cpu")
        use_accel = dev_t in {"cuda", "xpu", "mps"}
        use_prefetch = bool(use_accel and self._non_blocking)

        iterable: Any = self._base_iterable
        if use_prefetch:
            iterable = Prefetcher(
                iterable,
                device=dev,
                depth=int(self._prefetch_depth),
                non_blocking=True,
                memory_backpressure=True,
                gpu_guard_bytes=int(self._gpu_guard_bytes),
                host_guard_bytes=int(self._host_guard_bytes),
                pin_host=bool(self._pin_host and dev_t in {"cuda", "xpu"}),
            )
        return iter(iterable)

    def __len__(self) -> int:
        if self._length is not None:
            return int(self._length)
        try:
            return int(len(self._base_iterable))
        except Exception:
            return 1

    def _device_for_current_thread(self) -> torch.device:
        if isinstance(self._device, list):
            tid = threading.get_ident()
            dev = self._thread2dev.get(tid)
            if dev is None:
                idx = len(self._thread2dev) % max(1, len(self._device))
                dev = self._device[idx]
                self._thread2dev[tid] = dev
            return dev
        return self._device

    def _infer_mapper_threads(self, node: Any) -> int:
        if node is None:
            return 1
        if hasattr(node, "num_workers"):
            with suppress(Exception):
                return int(getattr(node, "num_workers"))
        for key in ("child", "source", "node", "_node"):
            sub = getattr(node, key, None)
            if sub is not None:
                count = self._infer_mapper_threads(sub)
                if count > 0:
                    return count
        return 1

    def _local_device_index(self) -> int:
        try:
            if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                return int(torch.cuda.current_device())
            xpu = getattr(torch, "xpu", None)
            if xpu is not None:
                fn = getattr(xpu, "is_available", None)
                if callable(fn) and fn():
                    return int(xpu.current_device())
        except Exception:
            pass
        return 0


class BufferedLoader(Buffer):
    """Small in-memory backpressure wrapper for any iterable/loader (session-based)."""

    def __init__(
        self,
        iterable: Any,
        *,
        max_batches: int = 4,
        name: str = "buffer",
        daemon: bool = True,
        _session: bool = False,
    ) -> None:
        super().__init__(max_batches=max_batches)
        self._src = iterable
        self._name = str(name or "buffer")
        self._daemon = bool(daemon)
        self._session = bool(_session)

        self._join_timeout_s = 0.5
        with suppress(Exception):
            jt_ms = int(env_first_int(("STNET_THREAD_JOIN_TIMEOUT_MS",), default=500) or 500)
            self._join_timeout_s = max(0.0, float(jt_ms) / 1000.0)

    def __len__(self) -> int:
        try:
            return int(len(self._src))  # type: ignore[arg-type]
        except Exception:
            return 1

    def __iter__(self) -> Iterator[Any]:
        if not bool(self._session):
            return iter(
                BufferedLoader(
                    self._src,
                    max_batches=int(self.max_batches),
                    name=self._name,
                    daemon=self._daemon,
                    _session=True,
                )
            )
        return self._iter_session()

    def _iter_session(self) -> Iterator[Any]:
        import traceback

        src_iter = iter(self._src)
        sentinel = object()

        def _producer() -> None:
            try:
                for item in src_iter:
                    if self.is_stopped():
                        break
                    if not self.put(item):
                        break
            except BaseException as exc:
                with suppress(Exception):
                    self.put(ProducerError(exc=exc, tb=traceback.format_exc()))
            finally:
                with suppress(Exception):
                    self.put(sentinel)

        t = threading.Thread(
            target=_producer,
            name=f"{self._name}-producer",
            daemon=self._daemon,
        )
        t.start()

        try:
            while True:
                try:
                    item = self.get(timeout=None)
                except queue.Empty:
                    break

                if item is sentinel:
                    break

                if isinstance(item, ProducerError):
                    raise RuntimeError(f"BufferedLoader producer crashed: {item.exc}\n{item.tb}") from item.exc

                yield item
        finally:
            self.stop()
            best_effort_close(src_iter)
            with suppress(Exception):
                if t.is_alive():
                    t.join(timeout=float(getattr(self, "_join_timeout_s", 0.5)))
            best_effort_close(t)


class Prefetcher(Buffer):
    def __init__(
        self,
        iterable: Any,
        *args: Any,
        device: torch.device | str,
        depth: int = 2,
        non_blocking: bool = True,
        oom_safe: bool = True,
        memory_backpressure: bool | None = None,
        gpu_guard_bytes: int | None = None,
        host_guard_bytes: int | None = None,
        _session: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_batches=depth)
        self._src = iterable
        self._device = torch.device(device) if not isinstance(device, torch.device) else device
        self._depth = max(1, int(depth))
        self._non_blocking = bool(non_blocking)

        self._backpressure = bool(memory_backpressure) if memory_backpressure is not None else bool(oom_safe)
        self._gpu_guard_bytes = int(gpu_guard_bytes or 0)
        self._host_guard_bytes = int(host_guard_bytes or 0)

        use_accel = isinstance(self._device, torch.device) and self._device.type in ("cuda", "xpu", "mps")
        self._pin = bool(kwargs.get("pin_host", use_accel))

        # Optional pinned staging pool (CUDA-only): reduces repeated pinned allocations
        self._pin_pool = False
        self._host_pool: Optional[Pool] = None
        if (
            self._pin
            and self._non_blocking
            and self._device.type == "cuda"
            and torch.cuda.is_available()
        ):
            use_pool = env_bool("STNET_PREFETCH_PIN_POOL", default=True)
            cap_default = max(8, max(2, int(self._depth) * 2))
            cap = env_first_int(("STNET_PREFETCH_PIN_POOL_CAPACITY",), default=cap_default)
            if use_pool and int(cap) > 0:
                self._host_pool = Pool(capacity=int(cap), pin_memory=True)
                self._pin_pool = True

        self._gpu_stream: Optional[torch.cuda.Stream] = None
        self._gpu_event_pool: Optional[queue.SimpleQueue] = None

        self._session = bool(_session)

        # Guard-check caching (avoid per-batch driver / syscalls).
        ttl_ms = int(env_first_int(("STNET_PREFETCH_GUARD_TTL_MS",), default=10) or 10)
        self._guard_ttl_s = max(0.0, float(ttl_ms) / 1000.0)

        self._join_timeout_s = 0.5
        with suppress(Exception):
            jt_ms = int(env_first_int(("STNET_THREAD_JOIN_TIMEOUT_MS",), default=500) or 500)
            self._join_timeout_s = max(0.0, float(jt_ms) / 1000.0)

    def _spawn_session(self) -> "Prefetcher":
        return Prefetcher(
            self._src,
            device=self._device,
            depth=int(self._depth),
            non_blocking=bool(self._non_blocking),
            memory_backpressure=bool(self._backpressure),
            gpu_guard_bytes=int(self._gpu_guard_bytes),
            host_guard_bytes=int(self._host_guard_bytes),
            pin_host=bool(self._pin),
            _session=True,
        )

    def _to_device(self, x: Any, device: torch.device) -> Any:
        if torch.is_tensor(x):
            if x.device == device:
                return x
            return x.to(device, non_blocking=self._non_blocking)
        if isinstance(x, (list, tuple)):
            return type(x)(self._to_device(t, device) for t in x)
        if isinstance(x, dict):
            out: dict[Any, Any] = {}
            for k, v in x.items():
                if k == "row_ids":
                    out[k] = v
                else:
                    out[k] = self._to_device(v, device)
            return out
        return x

    def _pin_memory(self, x: Any) -> Any:
        if not self._pin:
            return x
        if torch.is_tensor(x) and x.device.type == "cpu":
            with suppress(Exception):
                if hasattr(x, "is_pinned") and bool(x.is_pinned()):
                    return x
            return x.pin_memory()
        if isinstance(x, (list, tuple)):
            return type(x)(self._pin_memory(t) for t in x)
        if isinstance(x, dict):
            out: dict[Any, Any] = {}
            for k, v in x.items():
                if k == "row_ids":
                    out[k] = v
                else:
                    out[k] = self._pin_memory(v)
            return out
        return x

    def _pin_batch(self, x: Any) -> tuple[Any, list[Optional[Pool.Token]]]:
        """Pin/stage tensors in `x` and return (pinned_x, pool_tokens)."""
        if not self._pin:
            return x, []

        pool = self._host_pool
        if pool is None:
            return self._pin_memory(x), []

        tokens: list[Optional[Pool.Token]] = []

        def stage(obj: Any) -> Any:
            if torch.is_tensor(obj) and getattr(obj, "device", None) is not None:
                if obj.device.type != "cpu":
                    return obj
                try:
                    if hasattr(obj, "is_pinned") and bool(obj.is_pinned()):
                        return obj
                except Exception:
                    pass
                buf, tok = pool.get_like(obj, return_handle=True, block=False)
                buf.copy_(obj, non_blocking=False)
                tokens.append(tok)
                return buf
            if isinstance(obj, (list, tuple)):
                return type(obj)(stage(t) for t in obj)
            if isinstance(obj, dict):
                out: dict[Any, Any] = {}
                for k, v in obj.items():
                    out[k] = v if k == "row_ids" else stage(v)
                return out
            return obj

        return stage(x), tokens

    def _producer_loop(
        self,
        it: Iterator[Any],
        sentinel: object,
        *,
        device: torch.device,
        use_device: bool,
        use_cuda_stream: bool,
        gpu_guard_bytes: int,
        host_guard_bytes: int,
    ) -> None:
        import time
        import traceback

        last_check_t = 0.0
        last_guards_ok = True
        ttl_s = float(getattr(self, "_guard_ttl_s", 0.0) or 0.0)

        def guards_ok(force: bool = False) -> bool:
            nonlocal last_check_t, last_guards_ok
            if not self._backpressure:
                return True
            if (not force) and ttl_s > 0.0:
                now = time.monotonic()
                if (now - last_check_t) < ttl_s:
                    return bool(last_guards_ok)

            last_check_t = time.monotonic()
            host_ok = _host_guard_ok(host_guard_bytes)
            gpu_ok = _gpu_guard_ok(use_cuda_stream, device, gpu_guard_bytes)
            last_guards_ok = bool(host_ok and gpu_ok)
            return bool(last_guards_ok)

        try:
            if use_cuda_stream and isinstance(device, torch.device):
                with suppress(Exception):
                    if device.index is not None:
                        torch.cuda.set_device(device.index)

            for batch in it:
                if self.is_stopped():
                    break

                batch, pool_tokens = self._pin_batch(batch)

                if self._backpressure:
                    ok = guards_ok(force=True)
                    while (not self.is_stopped()) and (not ok):
                        # Exponential backoff with a small ceiling to reduce CPU churn
                        time.sleep(0.001)
                        ok = guards_ok(force=False)

                if use_device:
                    if use_cuda_stream and self._gpu_stream is not None:
                        ev = None
                        pool = self._gpu_event_pool
                        if pool is not None:
                            with suppress(Exception):
                                ev = pool.get()

                        try:
                            with torch.cuda.stream(self._gpu_stream):
                                batch_dev = self._to_device(batch, device)
                                if ev is not None:
                                    ev.record(self._gpu_stream)

                            if self._host_pool is not None and pool_tokens and ev is not None:
                                for tok in pool_tokens:
                                    self._host_pool.release_after(tok, ev)

                            if not self.put((batch_dev, ev)):
                                if ev is not None and pool is not None:
                                    with suppress(Exception):
                                        pool.put(ev)
                                break
                        except Exception:
                            if ev is not None and pool is not None:
                                with suppress(Exception):
                                    pool.put(ev)
                            raise
                    else:
                        batch_dev = self._to_device(batch, device)
                        if not self.put((batch_dev, None)):
                            break
                else:
                    if not self.put((batch, None)):
                        break

        except BaseException as exc:
            with suppress(Exception):
                self.put(ProducerError(exc=exc, tb=traceback.format_exc()))
        finally:
            with suppress(Exception):
                self.put(sentinel)

    def __iter__(self) -> Iterator[Any]:
        # Return a fresh session per iteration to avoid reusing threads/queues.
        if not bool(self._session):
            return iter(self._spawn_session())

        device = getattr(self, "_device", torch.device("cpu"))
        use_device = device.type in {"cuda", "mps", "xpu"}
        use_cuda_stream = device.type == "cuda" and hasattr(torch, "cuda") and torch.cuda.is_available()

        iterable = getattr(self, "_iterable", self._src)
        gpu_guard_bytes = int(getattr(self, "_gpu_guard_bytes", 0) or 0)
        host_guard_bytes = int(getattr(self, "_host_guard_bytes", 0) or 0)

        # Main process: producer thread + bounded buffer.
        if use_cuda_stream and self._gpu_stream is None:
            self._gpu_stream = torch.cuda.Stream(device=device)

        # Event pool (no per-batch allocation; safe reuse via producer/consumer handshake).
        if use_cuda_stream:
            pool: queue.SimpleQueue = queue.SimpleQueue()
            for _ in range(max(1, int(getattr(self, "_depth", 2) or 2))):
                with suppress(Exception):
                    pool.put(torch.cuda.Event(enable_timing=False))
            self._gpu_event_pool = pool
        else:
            self._gpu_event_pool = None

        sentinel = object()
        it = iter(iterable)

        t = threading.Thread(
            target=self._producer_loop,
            name="PrefetcherProducer",
            daemon=True,
            args=(it, sentinel),
            kwargs={
                "device": device,
                "use_device": use_device,
                "use_cuda_stream": use_cuda_stream,
                "gpu_guard_bytes": gpu_guard_bytes,
                "host_guard_bytes": host_guard_bytes,
            },
        )
        t.start()

        try:
            while True:
                try:
                    item = self.get(timeout=None)
                except queue.Empty:
                    break

                if item is sentinel:
                    break

                if isinstance(item, ProducerError):
                    raise RuntimeError(f"Prefetcher producer crashed: {item.exc}\n{item.tb}") from item.exc

                batch, ev = item
                if use_cuda_stream and ev is not None:
                    cs = torch.cuda.current_stream(device=device if isinstance(device, torch.device) else None)
                    with suppress(Exception):
                        cs.wait_event(ev)
                    pool = self._gpu_event_pool
                    if pool is not None:
                        with suppress(Exception):
                            pool.put(ev)

                yield batch
        finally:
            self.stop()
            best_effort_close(it)
            with suppress(Exception):
                if t.is_alive():
                    t.join(timeout=float(getattr(self, "_join_timeout_s", 0.5)))
            best_effort_close(t)
