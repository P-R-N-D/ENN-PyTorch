# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import json
import os
import math
import multiprocessing as mp
import threading
import queue
from contextlib import suppress
from typing import (Any, Callable, Dict, Iterator, Literal, Mapping, Optional,
                    Sequence, Tuple, TypedDict)

import torch

from ..backend.compat import ensure_torchdata
from ..backend.system import Thread, process_cpu_count

_LOGGER = logging.getLogger(__name__)
_MMAP_TL_LIMIT_LOCK = threading.Lock()

try:
    import psutil as _psutil
except ImportError:
    _psutil = None

TensorLike = Any

try:
    from torchdata.nodes import BaseNode
    from torchdata.nodes import Loader as _Loader
    from torchdata.nodes import ParallelMapper
except Exception as _e:
    ensure_torchdata(err=_e, context="stnet.data.nodes")

try:
    from tensordict import MemoryMappedTensor
except ImportError:  # pragma: no cover
    MemoryMappedTensor = None  # type: ignore

try:
    from torchdata.nodes import Batcher as _Batcher
    from torchdata.nodes import MultiNodeWeightedSampler, PinMemory
    from torchdata.nodes import Prefetcher as _Prefetcher
    from torchdata.nodes import SamplerWrapper
    from torchdata.nodes import Unbatcher as _Unbatcher
except Exception as _e:
    ensure_torchdata(err=_e, context="stnet.data.nodes")

try:
    from torch.utils.data import Sampler as _Sampler
except Exception:
    _Sampler = object

from .datatype import env_bool, env_first_int
from .collections import LazyTensor


def _to_device(batch: TensorLike, device: torch.device, non_blocking: bool = True) -> TensorLike:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, Mapping):
        return {k: _to_device(v, device, non_blocking) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        seq = [_to_device(v, device, non_blocking) for v in batch]
        return type(batch)(seq) if isinstance(batch, tuple) else seq
    return batch


class SamplerScale:
    """Per-session (or per-loader) scaling factor for auto batch sizing.

    NOTE:
      - Cross-process safe: uses multiprocessing.Value so updates propagate to loader workers.
      - Thread-safe: Value has an internal lock; we also guard bounds carefully.

    Design note (DynamicBatcher responsibility):
      - We intentionally keep "dynamic batch sizing" out of BufferedLoader.
      - The Sampler is the correct owner because it is the component that decides
        sampling granularity (batch size) and is already shared/pickled into worker
        processes.
      - Runtime recovery (e.g. OOM) is achieved by calling sampler_scale.request_scale_down(),
        and the very next sampler batch will reflect the new effective batch size via
        Sampler._S_B (dynamic property).
    """

    __slots__ = ("_v", "_min_scale", "_max_scale")

    def __init__(self, scale: float = 1.0, *, min_scale: float = 0.5, max_scale: float = 2.0) -> None:
        self._min_scale = float(min_scale)
        self._max_scale = float(max_scale)
        # Shared double, synchronized (propagates across spawn/fork worker processes).
        self._v = mp.Value("d", 1.0, lock=True)
        self.reset(scale)

    def __getstate__(self):
        # Keep the shared value handle so children see updates.
        return (self._v, float(self._min_scale), float(self._max_scale))

    def __setstate__(self, state):
        try:
            # New format: (mp.Value, min, max)
            v, mn, mx = state
        except Exception:
            v, mn, mx = None, 0.5, 2.0
        try:
            self._min_scale = float(mn)
        except Exception:
            self._min_scale = 0.5
        try:
            self._max_scale = float(mx)
        except Exception:
            self._max_scale = 2.0
        if v is None:
            self._v = mp.Value("d", 1.0, lock=True)
        else:
            self._v = v
        # Backward-compat: if someone pickled old-style (scale,mn,mx), handle it.
        try:
            if not hasattr(self._v, "get_lock"):
                # state was actually (scale,mn,mx)
                scale = float(v)
                self._v = mp.Value("d", 1.0, lock=True)
                self.reset(scale)
            else:
                # Keep existing value; no implicit reset.
                pass
        except Exception:
            self._v = mp.Value("d", 1.0, lock=True)

    def get(self) -> float:
        try:
            with self._v.get_lock():
                return float(self._v.value)
        except Exception:
            # best-effort fallback
            try:
                return float(self._v.value)
            except Exception:
                return 1.0

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

    @staticmethod
    def _dtype_from_name(name: Any, default: torch.dtype) -> torch.dtype:
        try:
            return getattr(torch, str(name))
        except Exception:
            return default

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
        sampler_scale: Optional["SamplerScale"] = None,
        **kwargs: Any,
    ) -> None:
        self.dir = os.fspath(memmap_dir)
        self.split = str(split)
        self._meta: Mapping[str, Any] = self._load_meta(self.dir)

        # Per-session/per-loader scale controller (NOT global).
        self._sampler_scale = sampler_scale if sampler_scale is not None else SamplerScale()

        # Runtime dynamic-batch cap (computed in pipeline._batch_interval).
        # 0 => unknown/unlimited (will still be clamped by split end).
        self._S_B_cap = 0
        self._N = int(self._meta.get("N", 0))
        if self._N <= 0:
            raise ValueError(f"meta.json under {self.dir} has non-positive N={self._N}")
        feat_rel = str(self._meta.get("features_path", "features.mmt"))
        lab_rel = str(self._meta.get("labels_path", "labels.mmt"))
        feat_path = os.path.join(self.dir, feat_rel)
        lab_path = os.path.join(self.dir, lab_rel)
        fdim = int(self._meta.get("feature_dim", 0))
        lshape_meta = list(self._meta.get("label_shape") or [])
        f_dtype = self._dtype_from_name(
            self._meta.get("features_dtype", "float64"), torch.float64
        )
        l_dtype = self._dtype_from_name(self._meta.get("labels_dtype", "int64"), torch.int64)
        # Keep the raw memmap config so we can optionally open per-thread handles (useful on no-GIL).
        self._feat_path = feat_path
        self._lab_path = lab_path
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
        self._labels = MemoryMappedTensor.from_filename(
            filename=lab_path, dtype=l_dtype, shape=torch.Size([self._N] + list(lshape))
        )
        # Optional: open per-thread MemoryMappedTensor handles. This is
        # conservative by default (auto-enabled only when no-GIL optimizations
        # are active). Override with STNET_MEMMAP_THREAD_LOCAL=0/1.
        # NOTE: keep the TLS object lazy/optional so the Sampler remains picklable
        # on platforms that use the "spawn" start method.
        self._mmap_tls: Optional[threading.local] = None
        self._mmap_init_lock = threading.Lock()
        self._mmap_thread_local = False
        # Limit the number of thread-local memmap handle pairs. On no-GIL builds,
        # it's easy to spawn many more threads, and each thread-local mmap consumes
        # file descriptors / VMAs.
        #
        # Override with:
        #   - STNET_MEMMAP_THREAD_LOCAL_MAX / STNET_MEMMAP_TL_MAX
        #   - Set <= 0 for "unlimited" (not recommended on large servers).
        self._mmap_thread_local_max = 0
        self._mmap_thread_local_created = 0
        self._mmap_thread_local_overflow_warned = False

        def _read_int_env(keys, default):
            for k in keys:
                try:
                    v = os.environ.get(k)
                    if v is not None and str(v).strip():
                        return int(v)
                except Exception:
                    pass
            return int(default)

        # Thread-local mmap handles are optional; default is "auto" for no-GIL builds.
        default_tl = False
        with suppress(Exception):
            default_tl = bool(Thread.nogil_optimizations_enabled())
        self._mmap_thread_local = env_bool("STNET_MEMMAP_THREAD_LOCAL", default=default_tl)

        if self._mmap_thread_local:
            # Default max: scale with CPU count, cap at 64 to avoid FD blowups.
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
        # Base batch size is stored separately; _S_B is a dynamic property that applies sampler_scale.
        self._S_B_base = 1
        # initialize via setter for compatibility
        self._S_B = 1
        self._S_shuffle = True
        self._S_seed = 0
        self._S_epoch = 0
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
            self._start, self._end = (
                (val_start, val_end) if val_end > val_start else (0, 0)
            )
        else:
            self._start, self._end = (train_start, train_end)

    @property
    def sampler_scale(self) -> "SamplerScale":
        return self._sampler_scale

    @property
    def base_batch_size(self) -> int:
        try:
            b = int(getattr(self, "_S_B_base", 1) or 1)
        except Exception:
            b = 1
        return max(1, int(b))

    def _effective_batch_size(self) -> int:
        # Base * current scale, clamped by cap when present.
        base = self.base_batch_size
        scale = 1.0
        with suppress(Exception):
            scale = float(self._sampler_scale.get())
        if not (scale > 0.0):
            scale = 1.0

        # Use rounding so small scale changes can still take effect for mid/large base.
        eff = int(round(float(base) * float(scale)))
        eff = max(1, int(eff))

        cap = 0
        with suppress(Exception):
            cap = int(getattr(self, "_S_B_cap", 0) or 0)
        if cap > 0:
            eff = min(int(eff), int(cap))
        return max(1, int(eff))

    # Backward-compatible dynamic attribute:
    # Any existing code that reads Sampler._S_B now gets the scaled batch size immediately.
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
        """Make the Sampler picklable.

        TorchData graphs may be pickled when using spawn-based multiprocessing.
        threading.local / Lock objects are not picklable, and MemoryMappedTensor
        holds process-local resources (file descriptors / mmaps).

        We drop those objects from the state and re-open the memmaps on unpickle.
        """

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
        ):
            state.pop(key, None)
        state["_mmap_tls"] = None
        state["_mmap_init_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate thread-local holders / locks lazily in the new process.
        self._mmap_tls = None
        self._mmap_init_lock = threading.Lock()

        if MemoryMappedTensor is None:
            raise ImportError(
                "tensordict is required for MemoryMappedTensor-backed pipelines. "
                "Please install 'tensordict' (or install stnet-pytorch with its default dependencies)."
            )

        # Re-open memmap-backed tensors in the current process.
        self._features = MemoryMappedTensor.from_filename(
            filename=str(self._feat_path),
            dtype=self._feat_dtype,
            shape=self._feat_shape,
        )
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
        """Return (features, labels) MemoryMappedTensor handles for the current thread.

        On free-threading / no-GIL builds, some extension types may internally rely
        on per-object state that used to be implicitly protected by the GIL. Using
        thread-local handles avoids sharing that state across threads while still
        allowing true parallelism.
        """
        if not getattr(self, "_mmap_thread_local", False):
            return self._features, self._labels

        tls = getattr(self, "_mmap_tls", None)
        if tls is None:
            tls = threading.local()
            self._mmap_tls = tls

        f = getattr(tls, "features", None)
        l = getattr(tls, "labels", None)
        if f is not None and l is not None:
            return f, l

        max_pairs = int(getattr(self, "_mmap_thread_local_max", 0) or 0)
        if max_pairs > 0:
            # Only limit the number of *created* thread-local pairs per Sampler instance.
            with _MMAP_TL_LIMIT_LOCK:
                created = int(getattr(self, "_mmap_thread_local_created", 0) or 0)
                if created >= max_pairs:
                    # Overflow: fall back to shared handles.
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

        # Only lock around first-time initialization per thread.
        with init_lock:
            f = getattr(tls, "features", None)
            l = getattr(tls, "labels", None)
            if f is None or l is None:
                try:
                    f = MemoryMappedTensor.from_filename(
                        filename=str(self._feat_path),
                        dtype=self._feat_dtype,
                        shape=self._feat_shape,
                    )
                    l = MemoryMappedTensor.from_filename(
                        filename=str(self._lab_path),
                        dtype=self._label_dtype,
                        shape=self._label_shape_full,
                    )
                except Exception:
                    # If creation failed, return shared handles and undo our "created" counter increment.
                    if max_pairs > 0:
                        with _MMAP_TL_LIMIT_LOCK:
                            created = int(getattr(self, "_mmap_thread_local_created", 0) or 0)
                            setattr(self, "_mmap_thread_local_created", max(0, created - 1))
                    return self._features, self._labels

                setattr(tls, "features", f)
                setattr(tls, "labels", l)

        return f, l

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
        row_ids = torch.arange(start, end, dtype=torch.int64)
        x = features[start:end]
        y = labels[start:end]
        if self._label_shape:
            y = y.reshape(end - start, *self._label_shape)
        xt = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        yt = y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
        return {"X": xt, "Y": yt, "row_ids": row_ids}

    def __getitem__(
        self, idx: int | Tuple[int, int] | Sequence[int]
    ) -> Mapping[str, torch.Tensor]:
        features, labels = self._get_mmaps()
        if isinstance(idx, tuple) and len(idx) == 2:
            s, e = int(idx[0]), int(idx[1])
            return self._slice(s, e)
        if isinstance(idx, torch.Tensor) and idx.dtype in (torch.int64, torch.int32):
            idx = idx.tolist()
        if isinstance(idx, Sequence) and not isinstance(idx, (str, bytes, bytearray)):
            if len(idx) == 0:
                return self._slice(0, 0)
            idx_tensor = torch.as_tensor(list(idx), dtype=torch.long)
            try:
                x = features.index_select(0, idx_tensor)
            except Exception:
                x = (
                    features[idx_tensor]
                    if hasattr(features, "__getitem__")
                    else torch.as_tensor(features)[idx_tensor]
                )
            try:
                y = labels.index_select(0, idx_tensor)
            except Exception:
                y = (
                    labels[idx_tensor]
                    if hasattr(labels, "__getitem__")
                    else torch.as_tensor(labels)[idx_tensor]
                )
            if self._label_shape:
                y = y.reshape(y.shape[0], *self._label_shape)
            row_ids = idx_tensor.to(dtype=torch.int64, copy=False)
            return {"X": x, "Y": y, "row_ids": row_ids}
        i = self._start + int(idx)
        out = self._slice(i, i + 1)
        try:
            x = out.get("X", None)
            y = out.get("Y", None)
            if torch.is_tensor(x):
                x = x.squeeze(0)
            if torch.is_tensor(y):
                y = y.squeeze(0)
            r = out.get("row_ids", None)
            if torch.is_tensor(r):
                r = r.squeeze(0)
            return {"X": x, "Y": y, "row_ids": r}
        except Exception:
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

        env_world = os.environ.get("WORLD_SIZE")
        env_rank = os.environ.get("RANK")
        try:
            world = int(env_world) if env_world is not None else 1
        except Exception:
            world = 1
        try:
            rank = int(env_rank) if env_rank is not None else 0
        except Exception:
            rank = 0

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

        # Base batch size is stored in _S_B_base via the legacy _S_B setter.
        self._S_B = max(1, int(batch_size))
        self._S_shuffle = bool(shuffle)
        self._S_seed = int(seed)
        self._S_epoch = 0
        self._key = str(key)
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

        for b in order.tolist():
            bs = start + int(b) * int(block)
            be = min(end, bs + int(block))
            cur = int(bs)
            while cur < int(be):
                B = int(self._effective_batch_size())
                nxt = min(int(be), cur + max(1, int(B)))
                if nxt <= cur:
                    break
                yield (self._key, (int(cur), int(nxt)))
                cur = int(nxt)

    def __len__(self) -> int:
        start = int(getattr(self, "start", 0))
        end = int(getattr(self, "end", 0))
        B = max(1, int(self._effective_batch_size()))
        total = max(0, (end - start + B - 1) // B)
        ns = int(getattr(self, "_num_shards", 1) or 1)
        si = int(getattr(self, "_shard_id", 0) or 0)
        if ns <= 1:
            return int(total)
        return max(0, (int(total) - si + ns - 1) // ns)

    def set_epoch(self, epoch: int) -> None:
        self._S_epoch = int(epoch)

    def get(self, start: int, end: int) -> Mapping[str, Any]:
        from ..model.fused import Gradient as FxGradient

        s = int(start)
        e = int(end)
        n = max(0, e - s)
        features, labels = self._get_mmaps()

        def _is_accelerator_available() -> bool:
            if torch.cuda.is_available():
                return True
            try:
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    return True
            except Exception:
                pass
            try:
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    return True
            except Exception:
                pass
            return False

            with FxGradient.inference(torch.nn.Module()):
                if n <= 0:
                    X = features.narrow(0, 0, 0)
                    Y = labels.narrow(0, 0, 0)
                    return {"X": X, "Y": Y}
                X = features.narrow(0, s, n)
                Y = labels.narrow(0, s, n)
                if self._label_shape:
                    Y = Y.reshape(n, *self._label_shape)
                pin_in_dataset = env_bool("STNET_PIN_IN_DATASET", default=False)
                if pin_in_dataset and _is_accelerator_available():
                    with suppress(Exception):
                        X = X.pin_memory()
                        Y = Y.pin_memory()
                return {"X": X, "Y": Y}
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

    This function is kept for backwards compatibility but the implementation is
    centralized in :class:`stnet.data.collections.LazyTensor`.

    Notes:
      - `shuffle=True` performs *physical* shuffle at write time (no perm file).
      - `train_frac` is currently metadata-only; the split is driven by `val_frac`.
    """
    # Preserve old kwargs but route into the unified implementation.
    chunk_size = int(kwargs.pop("chunk_size", 4096) or 4096)
    with suppress(Exception):
        env_cs = int(env_first_int(("STNET_MEMMAP_CHUNK_SIZE",), default=0) or 0)
        if env_cs > 0:
            chunk_size = max(1, int(env_cs))

    underflow_action = kwargs.pop("underflow_action", None)
    allow_missing_labels = bool(kwargs.pop("allow_missing_labels", False))
    default_label_shape = kwargs.pop("default_label_shape", None)

    # Any unexpected kwargs are ignored for compatibility.
    LazyTensor.preload_memmap(
        data,
        memmap_dir=os.fspath(memmap_dir),
        val_frac=float(val_frac),
        shuffle=bool(shuffle),
        seed=int(seed) if seed is not None else None,
        underflow_action=underflow_action,
        chunk_size=int(chunk_size),
        allow_missing_labels=bool(allow_missing_labels),
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
            cleaned = False
            for name in (
                "cleanup",
                "close",
                "shutdown",
                "stop",
                "terminate",
                "join",
                "disconnect",
                "release",
            ):
                if hasattr(obj, name):
                    with suppress(Exception):
                        getattr(obj, name)()
                    cleaned = True
                    break
            if cleaned:
                continue
            if callable(obj):
                with suppress(Exception):
                    obj()

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
        # Holds a reference to the composed MultiNodeWeightedSampler (if used).
        # This allows set_epoch() to update the sampler's epoch/seed.
        self._node: Optional[Any] = None
        self._source_keys: list[str] = []

    def set_epoch(self, epoch: int) -> None:
        """Per-epoch reseed for multi-source mixing.

        torchdata.nodes.MultiNodeWeightedSampler stores an epoch counter in its state dict
        (EPOCH_KEY), and uses it (together with the base seed) to initialize its RNG.
        To reliably change the mixing order each epoch, we set EPOCH_KEY and clear the
        cached weighted-sampler RNG state, while resetting bookkeeping/exhaustion flags.

        Stability notes:
        - Avoid mutating an arbitrary existing state dict (can contain stale child-node states
          and can vary across torchdata versions).
        - Prefer reset(initial_state) with a minimal, consistent state dict built from the
          node's documented state keys.
        """

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

        # State keys (prefer node's constants; fall back to common names).
        epoch_key = _key("EPOCH_KEY", "epoch")
        ws_key = _key("WEIGHTED_SAMPLER_STATE_KEY", "weighted_sampler_state")
        ny_key = _key("NUM_YIELDED_KEY", "num_yielded")
        ex_key = _key("DATASETS_EXHAUSTED_KEY", "datasets_exhausted")
        dns_key = _key("DATASET_NODE_STATES_KEY", "dataset_node_states")

        keys = list(getattr(self, "_source_keys", []) or [])
        initial_state: Dict[str, Any] = {
            epoch_key: int(self._epoch),
            ny_key: 0,
            ws_key: None,
        }
        if keys:
            # Child nodes: pass None so each source resets cleanly, and per-dataset epoch hooks
            # can apply independently.
            initial_state[ex_key] = {k: False for k in keys}
            initial_state[dns_key] = {k: None for k in keys}

        try:
            reset(initial_state)
            return
        except Exception:
            pass

        # Fallback: perturb seed and reset to start. This keeps training running even if
        # torchdata internals differ, while still varying the RNG each epoch.
        with suppress(Exception):
            setattr(node, "seed", int(self.seed) + int(self._epoch))
        with suppress(Exception):
            reset(None)

    def compose(
        self,
        sources: Mapping[str, "BaseNode"] | Sequence["BaseNode"] | "BaseNode",
    ) -> "BaseNode":
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
            raise TypeError(
                "sources must be a BaseNode, Sequence[BaseNode], or Mapping[str, BaseNode]"
            )
        if MultiNodeWeightedSampler is None:
            raise RuntimeError(
                "torchdata.nodes.MultiNodeWeightedSampler is required for multi-source mixing"
            )
        w = self.weights or {k: 1.0 for k in sources_map}
        self._source_keys = list(sources_map.keys())
        node = MultiNodeWeightedSampler(
            sources_map,
            w,
            stop_criteria=self.stop_criteria,
            seed=int(self.seed),
        )
        self._node = node
        return node


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
        from ..backend.system import WorkerPolicy

        wp = WorkerPolicy.autotune()
        wp.apply_torch_threads()

        self.io_workers = (
            int(io_workers) if io_workers is not None else int(getattr(wp, "num_workers", 1))
        )
        self.io_workers = max(1, self.io_workers)

        self.prebatch = (
            int(prebatch)
            if prebatch is not None
            else int(getattr(wp, "prebatch", max(1, self.io_workers * 2)))
        )
        with suppress(Exception):
            self.prebatch = max(1, int(self.prebatch))

        pf = (
            int(prefetch_factor)
            if prefetch_factor is not None
            else int(getattr(wp, "prefetch_factor", 1))
        )
        with suppress(Exception):
            pf = max(1, int(pf))
        self._prefetch_factor = pf
        self.prefetch_factor = self._prefetch_factor
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.non_blocking = bool(non_blocking)
        pin = (
            bool(pin_memory)
            if pin_memory is not None
            else (getattr(self.device, "type", "cpu") in {"cuda", "xpu"})
        )
        self._pin_memory = pin
        self.pin_memory = self._pin_memory
        self.max_concurrency = max(1, int(self.io_workers))
        try:
            from ..backend.system import get_tlb
            get_tlb(io_workers=self.io_workers)
        except Exception:
            pass

    def compose(self, source: "BaseNode") -> "BaseNode":
        from ..backend.system import wrap_with_tlb

        if ParallelMapper is None:
            raise RuntimeError("torchdata.nodes.ParallelMapper is required")
        node: BaseNode = source
        mapper = self.map_fn
        if (self.prebatch or 0) and int(self.prebatch) > 1:
            if _Batcher is None or _Unbatcher is None:
                raise RuntimeError("torchdata.nodes Batcher/Unbatcher are required for prebatch>1")
            B = max(1, int(self.prebatch))
            node = _Batcher(node, batch_size=B, drop_last=False)

            def iterate(ranges: Sequence[Any]) -> Sequence[Any]:
                return mapper(ranges)

            pm_map_fn = wrap_with_tlb(iterate)
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


class Loader:
    @staticmethod
    def compose(
        source: "BaseNode",
        *args: Any,
        device: torch.device,
        prefetch_factor: int = 2,
        non_blocking: bool = True,
        length: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        **kwargs: Any,
    ) -> "Loader":

        dev = device if isinstance(device, torch.device) else torch.device(device)
        node = source
        pf = max(1, int(prefetch_factor))
        node = _Prefetcher(node, prefetch_factor=pf)
        do_pin = bool(pin_memory) if pin_memory is not None else (getattr(dev, 'type', 'cpu') in {'cuda','xpu','mps'})
        if do_pin:
            node = PinMemory(node, pin_memory_device=dev.type)
        return Loader(
            dev,
            node=node,
            prefetch_factor=pf,
            non_blocking=bool(non_blocking),
            length=length,
        )

    def __init__(
        self,
        device: torch.device,
        *args: Any,
        node: BaseNode | None = None,
        dataset: BaseNode | None = None,
        prefetch_factor: int = 2,
        non_blocking: bool = True,
        length: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        node_obj = node or dataset
        if not isinstance(node_obj, BaseNode):
            raise TypeError("Loader supports only torchdata.nodes.BaseNode instances.")
        self._device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self._prefetch_factor = max(1, int(prefetch_factor))
        self._non_blocking = bool(non_blocking)
        self._length = int(length) if length is not None else None
        self._thread2dev: Dict[int, torch.device] = {}
        self._node = node_obj
        self._threads_hint = (
            self._infer_mapper_threads(node_obj) if node_obj is not None else 1
        )
        self._num_shards = 1
        self._shard_id = 0
        try:
            from ..backend.system import num_accelerators

            acc = max(1, int(num_accelerators()))
            thr = max(1, int(self._threads_hint))
            self._num_shards = acc * thr
            dev_idx = self._local_device_index()
            self._shard_id = max(0, min(self._num_shards - 1, int(dev_idx * thr)))
        except Exception:
            pass
        base = node_obj if isinstance(node_obj, _Loader) else _Loader(node_obj)
        dev_t = getattr(self._device, "type", "cpu")
        if dev_t in {"cuda", "mps", "xpu"} and self._non_blocking:
            try:
                gpu_guard_default = "2048" if dev_t == "cuda" else "512"
                gpu_guard_mb = int(gpu_guard_default)
            except Exception:
                gpu_guard_mb = 2048 if dev_t == "cuda" else 512
            try:
                host_guard_mb = 1024
            except Exception:
                host_guard_mb = 1024
            self._iterable = Prefetcher(
                base,
                device=self._device,
                depth=4,
                non_blocking=True,
                memory_backpressure=True,
                gpu_guard_bytes=gpu_guard_mb * (1 << 20),
                host_guard_bytes=host_guard_mb * (1 << 20),
            )
        else:
            self._iterable = base

    def __iter__(self) -> Iterator[Any]:
        return iter(self._iterable)

    def __len__(self) -> int:
        if self._length is not None:
            return int(self._length)
        iterable = getattr(self, "_iterable", None)
        if iterable is None:
            return 1
        try:
            return int(len(iterable))
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
            try:
                return int(getattr(node, "num_workers"))
            except Exception:
                pass
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
            if (
                xpu is not None
                and callable(getattr(xpu, "is_available", None))
                and xpu.is_available()
            ):
                return int(xpu.current_device())
        except Exception:
            pass
        return 0




class BufferedLoader:
    """A small in-memory backpressure wrapper for any iterable/loader.

    Why this exists:
      - Many loader graphs already prefetch internally (workers/prefetch_factor/etc).
      - This wrapper provides an explicit, *small* inflight cap so upstream cannot
        outrun the consumer and blow up host RAM.

    Behavior:
      - Iterating this object starts a daemon producer thread that pulls from the
        source iterable and enqueues items.
      - The consumer yields items by dequeuing.
      - When the queue reaches max_batches, producer blocks (backpressure).
      - Exceptions in producer are forwarded to the consumer.

    Notes:
      - This is an *iterable* (not a single iterator). Each __iter__ call starts a new
        producer thread and a fresh queue, so you can iterate multiple times (epochs).
    """

    def __init__(
        self,
        iterable: Any,
        *,
        max_batches: int = 4,
        name: str = "buffer",
        daemon: bool = True,
    ) -> None:
        self._src = iterable
        self._max_batches = max(1, int(max_batches))
        self._name = str(name or "buffer")
        self._daemon = bool(daemon)

    def __len__(self) -> int:
        try:
            return int(len(self._src))  # type: ignore[arg-type]
        except Exception:
            return 1

    def __iter__(self) -> Iterator[Any]:
        import traceback

        from .collections import Buffer

        buf = Buffer(max_batches=self._max_batches)
        stop = threading.Event()
        SENTINEL = object()

        class _Err:
            __slots__ = ("exc", "tb")
            def __init__(self, exc: BaseException, tb: str) -> None:
                self.exc = exc
                self.tb = tb

        src_iter = iter(self._src)

        def _producer() -> None:
            try:
                for item in src_iter:
                    if stop.is_set() or buf.is_stopped():
                        break
                    if not buf.put(item):
                        break
            except BaseException as e:
                tb = traceback.format_exc()
                buf.put(_Err(e, tb))
            finally:
                # Always try to send a sentinel so the consumer can terminate cleanly.
                buf.put(SENTINEL)

        t = threading.Thread(
            target=_producer,
            name=f"{self._name}-producer",
            daemon=self._daemon,
        )
        t.start()

        try:
            while True:
                try:
                    item = buf.get(timeout=0.1)
                except queue.Empty:
                    if stop.is_set() and not t.is_alive():
                        break
                    continue
                if item is SENTINEL:
                    break
                if isinstance(item, _Err):
                    raise RuntimeError(
                        f"BufferedLoader producer crashed: {item.exc}\n{item.tb}"
                    ) from item.exc
                yield item
        finally:
            # Best-effort early-stop: ask producer to stop and close upstream if possible.
            stop.set()
            buf.stop()
            with suppress(Exception):
                close = getattr(src_iter, "close", None)
                if callable(close):
                    close()

class Prefetcher:
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
        **kwargs: Any,
    ) -> None:
        self._src = iterable
        self._device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )
        self._depth = max(1, int(depth))
        self._non_blocking = bool(non_blocking)
        if memory_backpressure is not None:
            oom_safe = bool(memory_backpressure)
        self._backpressure = bool(oom_safe)
        self._gpu_guard_bytes = int(gpu_guard_bytes or 0)
        self._host_guard_bytes = int(host_guard_bytes or 0)
        use_accel = isinstance(self._device, torch.device) and self._device.type in ("cuda", "xpu")
        self._pin   = bool(kwargs.get("pin_host", use_accel))                             
        self._gpu_stream = None

    def _to_device(self, x: Any, device: torch.device) -> Any:
        if torch.is_tensor(x):
            return x.to(device, non_blocking=self._non_blocking)
        if isinstance(x, (list, tuple)):
            return type(x)(self._to_device(t, device) for t in x)
        if isinstance(x, dict):
            out: dict[Any, Any] = {}
            for k, v in x.items():
                # Keep lightweight metadata on host.
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
            return x.pin_memory()
        if isinstance(x, (list, tuple)):
            return type(x)(self._pin_memory(t) for t in x)
        if isinstance(x, dict):
            out: dict[Any, Any] = {}
            for k, v in x.items():
                # row_ids are small metadata; do NOT pin to avoid unnecessary pinned allocations.
                if k == "row_ids":
                    out[k] = v
                else:
                    out[k] = self._pin_memory(v)
            return out
        return x

    def __iter__(self) -> Iterator[Any]:
        import time
        import traceback

        from torch.utils.data import get_worker_info

        info = get_worker_info()
        device = getattr(self, "_device", torch.device("cpu"))
        use_device = (device.type in {"cuda", "mps", "xpu"})
        use_cuda_stream = (device.type == "cuda" and hasattr(torch, "cuda") and torch.cuda.is_available())
        iterable = getattr(self, "_iterable", self._src)

                                                      
        def _gpu_mem_ok() -> bool:
            if not use_cuda_stream or self._gpu_guard_bytes <= 0:
                return True
            try:
                free_b, total_b = torch.cuda.mem_get_info(
                    device=device if isinstance(device, torch.device) else None
                )
                return bool(free_b >= self._gpu_guard_bytes)
            except Exception:
                return True

        def _host_mem_ok() -> bool:
            if self._host_guard_bytes <= 0:
                return True
            if _psutil is None:
                return True
            try:
                return bool(_psutil.virtual_memory().available >= self._host_guard_bytes)
            except Exception:
                return True

        if info is not None:
                                                               
            for batch in iterable:
                if self._pin:
                    batch = self._pin_memory(batch)
                yield batch
                                                                                   
                if self._backpressure:
                    time.sleep(0)
        else:

            if use_cuda_stream and self._gpu_stream is None:
                self._gpu_stream = torch.cuda.Stream(device=device)

            SENTINEL = object()

            class _Err:
                def __init__(self, exc: BaseException, tb: str) -> None:
                    self.exc = exc
                    self.tb = tb

            q: "queue.Queue[Any]" = queue.Queue(maxsize=int(self._depth))
            stop = threading.Event()
            it = iter(iterable)

            def _put(item: Any) -> None:
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.1)
                        return
                    except queue.Full:
                        continue

            def _producer() -> None:
                try:
                    if use_cuda_stream and isinstance(device, torch.device):
                        with suppress(Exception):
                            if device.index is not None:
                                torch.cuda.set_device(device.index)
                    for batch in it:
                        if stop.is_set():
                            break
                        if self._pin:
                            batch = self._pin_memory(batch)

                        tries = 0
                        if self._backpressure:
                            while (not stop.is_set()) and (
                                (not _host_mem_ok()) or (not _gpu_mem_ok())
                            ):
                                time.sleep(0.001 if tries < 1000 else 0.005)
                                tries += 1

                        if use_device:
                            if use_cuda_stream and self._gpu_stream is not None:
                                with torch.cuda.stream(self._gpu_stream):
                                    batch_dev = self._to_device(batch, device)
                                ev = torch.cuda.Event()
                                ev.record(self._gpu_stream)
                                _put((batch_dev, ev))
                            else:
                                batch_dev = self._to_device(batch, device)
                                _put((batch_dev, None))
                        else:
                            _put((batch, None))
                except BaseException as exc:
                    _put(_Err(exc, traceback.format_exc()))
                finally:
                    _put(SENTINEL)

            t = threading.Thread(target=_producer, name="PrefetcherProducer", daemon=True)
            t.start()

            def _maybe_close_upstream() -> None:
                with suppress(Exception):
                    close = getattr(it, "close", None)
                    if callable(close):
                        close()

            try:
                while True:
                    item = q.get()
                    if item is SENTINEL:
                        break
                    if isinstance(item, _Err):
                        raise RuntimeError(
                            f"Prefetcher producer crashed: {item.exc}\n{item.tb}"
                        ) from item.exc

                    batch, ev = item
                    if use_cuda_stream and ev is not None:
                        cs = torch.cuda.current_stream(
                            device=device if isinstance(device, torch.device) else None
                        )
                        with suppress(Exception):
                            cs.wait_event(ev)

                    yield batch
            finally:
                stop.set()
                _maybe_close_upstream()
                with suppress(Exception):
                    t.join(timeout=1.0)
