# -*- coding: utf-8 -*-
from __future__ import annotations

import collections.abc
import contextlib
import itertools
import logging
import math
import multiprocessing
import os
import queue
import threading
import time
import traceback
from contextlib import suppress
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Self,
)

import torch
import torch.utils.data
import torchdata.nodes
from tensordict import MemoryMappedTensor
from torchdata.nodes import (
    BaseNode,
    MultiNodeWeightedSampler,
    ParallelMapper,
    SamplerWrapper,
)

from ..core.concurrency import (
    BufferQueue,
    Mutex,
    ProducerError,
    TensorPagePool,
    close,
    new_affinity,
    new_thread,
)
from ..core.datatypes import (
    dtype_from_name,
    env_bool,
    env_first_int,
    read_json,
)
from ..core.graph import inference_mode
from ..core.policies import WorkerPolicy
from ..core.system import (
    CPU,
    Memory,
    accelerator_stream,
    accelerator_type,
    current_accelerator_stream,
    get_accelerator_index,
    get_num_accelerators,
    is_accelerator_available,
    is_stream_supported,
    new_accelerator_event,
    new_accelerator_stream,
)


_LOGGER = logging.getLogger(__name__)


def _node_state_key(node: Any, attr: str, fallback: str) -> str:
    k = getattr(node, attr, None) or getattr(type(node), attr, None)
    return k if isinstance(k, str) else fallback


def _is_accelerator_available() -> bool:
    return any(is_accelerator_available(a) for a in ("cuda", "xpu", "mps"))


def _device_guard_ok(device: torch.device, guard_bytes: int) -> bool:
    if guard_bytes <= 0:
        return True
    try:
        return (
            free_b := Memory.mem_get_info(device)[0]
        ) is None or free_b >= guard_bytes
    except Exception:
        return True


def _host_guard_ok(guard_bytes: int) -> bool:
    return (
        guard_bytes <= 0
        or getattr(Memory, "available", lambda: guard_bytes)() >= guard_bytes
    )


def _accel_event_poll_params() -> tuple[float, float, float]:
    start_us = int(
        env_first_int(
            (
                "ENN_ACCEL_EVENT_POLL_START_US",
                "ENN_CUDA_EVENT_POLL_START_US",
            ),
            default=500,
        )
        or 500
    )
    max_ms = int(
        env_first_int(
            ("ENN_ACCEL_EVENT_POLL_MAX_MS", "ENN_CUDA_EVENT_POLL_MAX_MS"),
            default=50,
        )
        or 50
    )
    stop_min_ms = int(
        env_first_int(
            (
                "ENN_ACCEL_EVENT_POLL_STOP_MIN_MS",
                "ENN_CUDA_EVENT_POLL_STOP_MIN_MS",
            ),
            default=5,
        )
        or 5
    )
    base_s = max(0.0, float(start_us) / 1_000_000.0)
    max_s = max(base_s, float(max_ms) / 1000.0)
    stop_min_s = max(0.0, float(stop_min_ms) / 1000.0)
    return base_s, max_s, stop_min_s


def _wait_accel_event_done(
    ev: Any,
    *args: Any,
    stopped: Callable[[], bool] | None = None,
    base_sleep_s: float | None = None,
    max_sleep_s: float | None = None,
    stop_min_sleep_s: float | None = None,
) -> None:
    stop_fn = stopped if stopped is not None else (lambda: False)
    d_base, d_max, d_stop_min = _accel_event_poll_params()
    base_sleep_s = base_sleep_s if base_sleep_s is not None else d_base
    max_sleep_s = max_sleep_s if max_sleep_s is not None else d_max
    stop_min_sleep_s = (
        stop_min_sleep_s if stop_min_sleep_s is not None else d_stop_min
    )
    sleep_s = max(0.0, float(base_sleep_s))
    max_s = max(sleep_s, float(max_sleep_s))
    stop_min_s = max(0.0, float(stop_min_sleep_s))
    while True:
        try:
            if ev.query():
                return
        except Exception:
            with suppress(Exception):
                ev.synchronize()
            return
        if stop_fn():
            sleep_s = max(float(sleep_s), stop_min_s)
        time.sleep(sleep_s)
        sleep_s = min(float(sleep_s) * 2.0, max_s)


def _normalize_device_spec(
    device: torch.device | str | Sequence[torch.device | str],
) -> torch.device | list[torch.device]:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    if isinstance(device, Sequence) and not isinstance(
        device, (str, bytes, bytearray)
    ):
        devs: list[torch.device] = []
        for d in device:
            devs.append(
                d if isinstance(d, torch.device) else torch.device(str(d))
            )
        return devs if devs else torch.device("cpu")
    return torch.device(device)


def _primary_device(
    device_spec: torch.device | list[torch.device],
) -> torch.device:
    return (
        device_spec[0]
        if isinstance(device_spec, list) and device_spec
        else device_spec
    )


class Governor:
    __slots__ = ("_v", "_min_scale", "_max_scale")

    def __init__(
        self: Self,
        scale: float = 1.0,
        *args: Any,
        min_scale: float = 0.5,
        max_scale: float = 2.0,
    ) -> None:
        self._min_scale = float(min_scale)
        self._max_scale = float(max_scale)
        self._v = multiprocessing.Value("d", 1.0, lock=True)
        self.reset(scale)

    def __getstate__(self: Self) -> tuple[object, float, float]:
        return (self._v, float(self._min_scale), float(self._max_scale))

    def __setstate__(self: Self, state: tuple[object, float, float]) -> None:
        try:
            v, mn, mx = state
        except Exception:
            v, mn, mx = None, 0.5, 2.0
        self._min_scale = (
            float(mn) if isinstance(mn, (int, float, str)) else 0.5
        )
        self._max_scale = (
            float(mx) if isinstance(mx, (int, float, str)) else 2.0
        )
        if v is None:
            self._v = multiprocessing.Value("d", 1.0, lock=True)
        else:
            self._v = v
        try:
            if not hasattr(self._v, "get_lock"):
                scale = float(v)
                self._v = multiprocessing.Value("d", 1.0, lock=True)
                self.reset(scale)
        except Exception:
            self._v = multiprocessing.Value("d", 1.0, lock=True)

    def get(self: Self) -> float:
        mn = float(self._min_scale)
        mx = float(self._max_scale)
        try:
            v = float(self._v.value)
        except Exception:
            v = float("nan")
        if math.isfinite(v) and (v > 0.0) and (mn <= v <= mx):
            return float(v)
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

    def reset(self: Self, value: float = 1.0) -> None:
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

    def request_scale_up(self: Self, factor: float) -> None:
        try:
            f = float(factor)
        except Exception:
            f = 1.0
        if not (f > 0.0):
            return
        try:
            with self._v.get_lock():
                cur = float(self._v.value)
                self._v.value = float(
                    min(float(self._max_scale), cur * float(f))
                )
        except Exception:
            pass

    def request_scale_down(self: Self, factor: float) -> None:
        try:
            f = float(factor)
        except Exception:
            f = 1.0
        if not (f > 0.0):
            return
        try:
            with self._v.get_lock():
                cur = float(self._v.value)
                self._v.value = float(
                    max(float(self._min_scale), cur * float(f))
                )
        except Exception:
            pass


class Sampler(torch.utils.data.Sampler):
    _per_sample_mem_bytes: int = 0

    def __init__(
        self: Self,
        memmap_dir: str,
        *args: Any,
        split: str = "train",
        val_frac: float = 0.0,
        sampler_scale: Optional["Governor"] = None,
        **kwargs: Any,
    ) -> None:
        self.dir = os.fspath(memmap_dir)
        self.split = str(split)
        self._meta: Mapping[str, Any] = self._load_meta(self.dir)
        self._sampler_scale = (
            sampler_scale if sampler_scale is not None else Governor()
        )
        self._S_B_cap = 0
        self._N = int(self._meta.get("N", 0))
        if self._N <= 0:
            raise ValueError(
                f"meta.json under {self.dir} has non-positive N={self._N}"
            )
        feat_rel = str(self._meta.get("features_path", "features.mmt"))
        feat_path = os.path.join(self.dir, feat_rel)
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
                    raise FileNotFoundError(
                        f"labels.mmt not found under: {lab_path}"
                    )
        fdim = int(self._meta.get("feature_dim", 0))
        lshape_meta = list(self._meta.get("label_shape") or [])
        f_dtype = dtype_from_name(
            self._meta.get("features_dtype", "float64"), torch.float64
        )
        l_dtype = dtype_from_name(
            self._meta.get("labels_dtype", "int64"), torch.int64
        )
        self._include_row_ids = env_bool("ENN_INCLUDE_ROW_IDS", default=True)
        self._feat_path = feat_path
        self._lab_path = lab_path if lab_path else None
        self._feat_dtype = f_dtype
        self._label_dtype = l_dtype
        self._feat_shape = torch.Size([self._N, fdim])
        if MemoryMappedTensor is None:
            raise ImportError(
                "tensordict is required for MemoryMappedTensor-backed pipelines. "
                "Please install 'tensordict' (or install enn_torch-pytorch with its default dependencies)."
            )
        self._features = MemoryMappedTensor.from_filename(
            filename=feat_path,
            dtype=f_dtype,
            shape=torch.Size([self._N, fdim]),
        )
        lshape = tuple(lshape_meta)
        self._label_shape_full = torch.Size([self._N, *lshape])
        self._labels = (
            MemoryMappedTensor.from_filename(
                filename=str(self._lab_path),
                dtype=l_dtype,
                shape=self._label_shape_full,
            )
            if self._lab_path
            else None
        )
        self._mmap_tls, self._mmap_limit_lock = None, Mutex()
        self._mmap_thread_local = False
        self._mmap_thread_local_max = 0
        self._mmap_thread_local_created = 0
        self._mmap_thread_local_overflow_warned = False
        default_tl = False
        with suppress(Exception):
            default_tl = bool(CPU.is_optimized_for_no_gil())
        self._mmap_thread_local = env_bool(
            (
                "ENN_MEMMAP_THREAD_LOCAL_HANDLES",
                "ENN_MEMMAP_THREAD_LOCAL",
                "ENN_NOGIL",
            ),
            default=default_tl,
        )
        if self._mmap_thread_local:
            cpu = int(CPU.count() or 8)
            default_max = max(8, min(64, cpu))
            self._mmap_thread_local_max = env_first_int(
                ("ENN_MEMMAP_THREAD_LOCAL_MAX", "ENN_MEMMAP_TL_MAX"),
                default=default_max,
            )
            self._mmap_thread_local_max = int(self._mmap_thread_local_max)
        self._memmap_features = self._features
        self._memmap_labels = self._labels
        self._X = self._memmap_features
        self._Y = self._memmap_labels
        self._S_B_base = 1
        self._S_B = 1
        self._S_shuffle = True
        self._S_seed = 0
        self._S_epoch = 0
        self._len_epoch = -1
        self._len_B_snapshot: Optional[int] = None
        self._num_shards = 1
        self._shard_id = 0
        self._key = ""
        self._label_shape: Tuple[int, ...] = (
            tuple(lshape) if lshape else tuple()
        )
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

    @classmethod
    def _load_meta(cls: type["Sampler"], memmap_dir: str) -> Mapping[str, Any]:
        meta_path = os.path.join(os.fspath(memmap_dir), "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"meta.json not found under: {memmap_dir}")
        meta = read_json(meta_path)
        if not isinstance(meta, Mapping):
            raise ValueError(
                f"meta.json under {memmap_dir} must contain a mapping"
            )
        return meta

    def _effective_batch_size(self: Self) -> int:
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

    def _len_batch_size(self: Self) -> int:
        epoch = int(getattr(self, "_S_epoch", 0) or 0)
        snap_epoch = int(getattr(self, "_len_epoch", -1) or -1)
        snap = getattr(self, "_len_B_snapshot", None)
        if snap is None or snap_epoch != epoch:
            snap = max(1, int(self._effective_batch_size()))
            self._len_B_snapshot = int(snap)
            self._len_epoch = int(epoch)
        return max(1, int(snap))

    @property
    def _S_B(self: Self) -> int:
        return self._effective_batch_size()

    @_S_B.setter
    def _S_B(self: Self, value: int) -> None:
        try:
            v = int(value)
        except Exception:
            v = 1
        if v <= 0:
            v = 1
        setattr(self, "_S_B_base", int(v))

    def __getstate__(self: Self) -> dict[str, object]:
        state = dict(self.__dict__)
        for key in (
            "_features",
            "_labels",
            "_memmap_features",
            "_memmap_labels",
            "_X",
            "_Y",
            "_mmap_tls",
            "_mmap_limit_lock",
        ):
            state.pop(key, None)
        state["_mmap_tls"] = None
        state["_mmap_limit_lock"] = None
        return state

    def __setstate__(self: Self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._mmap_tls = None
        self._mmap_limit_lock = Mutex()
        if MemoryMappedTensor is None:
            raise ImportError(
                "tensordict is required for MemoryMappedTensor-backed pipelines. "
                "Please install 'tensordict' (or install enn_torch-pytorch with its default dependencies)."
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

    def _get_mmaps(
        self: Self,
    ) -> tuple[MemoryMappedTensor, MemoryMappedTensor | None]:
        if not getattr(self, "_mmap_thread_local", False):
            return self._features, self._labels
        has_labels = bool(self._labels is not None and self._lab_path)
        tls = self._mmap_tls or threading.local()
        self._mmap_tls = tls
        if getattr(tls, "features", None) is not None:
            return tls.features, getattr(tls, "labels", None)
        max_pairs = int(getattr(self, "_mmap_thread_local_max", 0) or 0)
        if max_pairs > 0:
            with self._mmap_limit_lock:
                created = self._mmap_thread_local_created
                if created >= max_pairs:
                    if not self._mmap_thread_local_overflow_warned:
                        self._mmap_thread_local_overflow_warned = True
                    return self._features, self._labels
                self._mmap_thread_local_created += 1
        try:
            f_new = MemoryMappedTensor.from_filename(
                filename=str(self._feat_path),
                dtype=self._feat_dtype,
                shape=self._feat_shape,
            )
            l_new = (
                MemoryMappedTensor.from_filename(
                    filename=str(self._lab_path),
                    dtype=self._label_dtype,
                    shape=self._label_shape_full,
                )
                if has_labels
                else None
            )
            tls.features, tls.labels = f_new, l_new
            return f_new, l_new
        except Exception:
            if max_pairs > 0:
                with self._mmap_limit_lock:
                    self._mmap_thread_local_created = max(
                        0, self._mmap_thread_local_created - 1
                    )
            return self._features, self._labels

    def _slice(self: Self, start: int, end: int) -> Mapping[str, torch.Tensor]:
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

    def _gather(
        self: Self, idx_tensor: torch.Tensor, features: Any, labels: Any
    ) -> Mapping[str, torch.Tensor]:
        idx_tensor = idx_tensor.to(dtype=torch.long, copy=False)
        try:
            x = features.index_select(0, idx_tensor)
        except Exception:
            x = (
                features[idx_tensor]
                if hasattr(features, "__getitem__")
                else torch.as_tensor(features)[idx_tensor]
            )
        out: Dict[str, torch.Tensor] = {"X": x}
        if bool(getattr(self, "_include_row_ids", True)):
            out["row_ids"] = idx_tensor.to(dtype=torch.int64, copy=False)
        if labels is not None:
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
            out["Y"] = y
        return out

    def __getitem__(
        self: Self, idx: int | Tuple[int, int] | Sequence[int] | torch.Tensor
    ) -> Mapping[str, torch.Tensor]:
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
            case seq if isinstance(seq, Sequence) and not isinstance(
                seq, (str, bytes, bytearray)
            ):
                if len(seq) == 0:
                    return self._slice(base, base)
                idx_tensor = (
                    torch.as_tensor(seq, dtype=torch.long).reshape(-1) + base
                )
                return self._gather(idx_tensor, features, labels)
            case _:
                i = base + int(idx)
                out = self._slice(i, i + 1)
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

    def _shard(self: Self) -> None:
        try:
            dist = getattr(torch, "distributed", None)
            if (
                dist is not None
                and dist.is_available()
                and dist.is_initialized()
            ):
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

    def __iter__(self: Self) -> Iterator[tuple[str, tuple[int, int]]]:
        start = int(getattr(self, "start", 0))
        end = int(getattr(self, "end", 0))
        if end <= start:
            return
        self._shard()
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
            g.manual_seed(
                int(getattr(self, "_S_seed", 0))
                + int(getattr(self, "_S_epoch", 0))
            )
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
                B = int(self._effective_batch_size())
                nxt = min(int(be), cur + max(1, int(B)))
                if nxt <= cur:
                    break
                yield (self._key, (int(cur), int(nxt)))
                cur = int(nxt)

    def __len__(self: Self) -> int:
        start = int(getattr(self, "start", 0))
        end = int(getattr(self, "end", 0))
        B = max(1, int(self._len_batch_size()))
        total = max(0, (end - start + B - 1) // B)
        ns = int(getattr(self, "_num_shards", 1) or 1)
        si = int(getattr(self, "_shard_id", 0) or 0)
        if ns <= 1:
            return int(total)
        return max(0, (int(total) - si + ns - 1) // ns)

    @property
    def sampler_scale(self: Self) -> "Governor":
        return self._sampler_scale

    @property
    def base_batch_size(self: Self) -> int:
        try:
            b = int(getattr(self, "_S_B_base", 1) or 1)
        except Exception:
            b = 1
        return max(1, int(b))

    @property
    def start(self: Self) -> int:
        return int(self._start)

    @property
    def end(self: Self) -> int:
        return int(self._end)

    @property
    def meta(self: Self) -> Mapping[str, Any]:
        return dict(self._meta or {})

    def compose(
        self: Self,
        *args: Any,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        key: str = "",
        **kwargs: Any,
    ) -> "BaseNode":
        self._S_B = max(1, int(batch_size))
        self._S_shuffle = bool(shuffle)
        self._S_seed = int(seed)
        self._S_epoch = 0
        self._key = str(key)
        self._len_epoch = int(self._S_epoch)
        self._len_B_snapshot = max(1, int(self._effective_batch_size()))
        self._shard()
        if SamplerWrapper is None:
            raise RuntimeError("torchdata.nodes.SamplerWrapper is required")
        return SamplerWrapper(self)

    def set_epoch(self: Self, epoch: int) -> None:
        self._S_epoch = int(epoch)
        self._len_epoch = int(self._S_epoch)
        self._len_B_snapshot = max(1, int(self._effective_batch_size()))

    def get(self: Self, start: int, end: int) -> Mapping[str, Any]:
        s = int(start)
        e = int(end)
        n = max(0, e - s)
        features, labels = self._get_mmaps()
        with inference_mode(torch.nn.Module()):
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
            pin_in_dataset = env_bool("ENN_PIN_IN_DATASET", default=False)
            if pin_in_dataset and _is_accelerator_available():
                with suppress(Exception):
                    out["X"] = out["X"].pin_memory()
                    if out.get("Y") is not None:
                        out["Y"] = out["Y"].pin_memory()
            return out


class Multiplexer:
    def __init__(
        self: Self,
        *args: Any,
        stop_criteria: str = "ALL_DATASETS_EXHAUSTED",
        weights: Optional[
            Mapping[str, float] | Sequence[float] | float | int
        ] = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        _ = args, kwargs
        self.stop_criteria = str(stop_criteria)
        self._raw_weights = weights
        self.seed = int(seed)
        self._epoch = 0
        self._node: Optional[Any] = None
        self._source_keys: list[str] = []

    def set_epoch(self: Self, epoch: int) -> None:
        self._epoch = int(epoch)
        node = getattr(self, "_node", None)
        if node is None:
            return
        reset = getattr(node, "reset", None)
        if not callable(reset):
            return

        keys = list(getattr(self, "_source_keys", []) or [])
        has_mnws_keys = bool(
            getattr(node, "DATASETS_EXHAUSTED_KEY", None)
            or getattr(type(node), "DATASETS_EXHAUSTED_KEY", None)
        )
        if (len(keys) <= 1) and (not has_mnws_keys):
            with suppress(Exception):
                reset(None)
            with suppress(Exception):
                setattr(node, "seed", int(self.seed) + int(self._epoch))
            return

        epoch_key = _node_state_key(node, "EPOCH_KEY", "epoch")
        ws_key = _node_state_key(
            node, "WEIGHTED_SAMPLER_STATE_KEY", "weighted_sampler_state"
        )
        ny_key = _node_state_key(node, "NUM_YIELDED_KEY", "num_yielded")
        ex_key = _node_state_key(
            node, "DATASETS_EXHAUSTED_KEY", "datasets_exhausted"
        )
        dns_key = _node_state_key(
            node, "DATASET_NODE_STATES_KEY", "dataset_node_states"
        )
        keys = list(getattr(self, "_source_keys", []) or [])
        initial_state: Dict[str, Any] = {
            epoch_key: int(self._epoch),
            ny_key: 0,
            ws_key: None,
        }
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

    def compose(
        self: Self,
        sources: Mapping[str, "BaseNode"] | Sequence["BaseNode"] | "BaseNode",
    ) -> "BaseNode":
        sources_kind: str
        if isinstance(sources, BaseNode):
            sources_kind = "single"
            sources_map = {"0": sources}
        elif isinstance(sources, (list, tuple)):
            sources_kind = "sequence"
            if len(sources) < 1:
                raise ValueError("sources must be non-empty")
            sources_map = (
                {"0": sources[0]}
                if len(sources) == 1
                else {str(i): n for i, n in enumerate(sources)}
            )
        elif isinstance(sources, Mapping):
            sources_kind = "mapping"
            if len(sources) < 1:
                raise ValueError("sources must be non-empty")
            sources_map: Dict[str, BaseNode] = {}
            for k, v in dict(sources).items():
                kk = str(k)
                if kk in sources_map:
                    raise ValueError(
                        f"sources has duplicate key after str(): {kk!r}"
                    )
                sources_map[kk] = v
        else:
            raise TypeError(
                "sources must be a BaseNode, Sequence[BaseNode], or Mapping[str, BaseNode]"
            )
        if len(sources_map) <= 1:
            self._source_keys = list(sources_map.keys())
            only_key = next(iter(sources_map.keys()))
            node = sources_map[only_key]
            self._node = node
            return node
        raw = getattr(self, "_raw_weights", None)

        def _coerce_weight(v: Any, *args: Any, where: str) -> float:
            try:
                fv = float(v)
            except Exception as exc:
                raise TypeError(
                    f"weights entry must be numeric (float/int): {where}"
                ) from exc
            if not math.isfinite(fv):
                raise ValueError(f"weights entry must be finite: {where}")
            if fv < 0.0:
                raise ValueError(f"weights entry must be >= 0: {where}")
            return float(fv)

        w: Dict[str, float]
        if raw is None:
            w = {k: 1.0 for k in sources_map.keys()}
        elif isinstance(raw, Mapping):
            if sources_kind != "mapping":
                raise TypeError(
                    "weights must be a Mapping[str, float] when sources is a Mapping"
                )
            w_in: Dict[str, float] = {}
            for k, v in dict(raw).items():
                kk = str(k)
                if kk in w_in:
                    raise ValueError(
                        f"weights has duplicate key after str(): {kk!r}"
                    )
                w_in[kk] = _coerce_weight(v, where=f"weights[{kk!r}]")
            if not w_in:
                raise ValueError(
                    "weights mapping must be non-empty (use None for uniform)"
                )
            missing = set(sources_map.keys()) - set(w_in.keys())
            extra = set(w_in.keys()) - set(sources_map.keys())
            if missing or extra:
                raise ValueError(
                    "weights mapping keys must match sources keys exactly; "
                    f"missing={sorted(missing)} extra={sorted(extra)} sources={sorted(sources_map.keys())}"
                )
            if not any((float(v) > 0.0) for v in w_in.values()):
                raise ValueError(
                    "weights mapping must contain at least one positive weight"
                )
            w = {k: float(w_in[k]) for k in sources_map.keys()}
        elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
            raise TypeError(
                "scalar weights are only supported for a single source (omit weights for multi-source)"
            )
        elif isinstance(raw, collections.abc.Sequence) and not isinstance(
            raw, (str, bytes, bytearray)
        ):
            if sources_kind != "sequence":
                raise TypeError(
                    "weights must be a Sequence[float] when sources is a Sequence"
                )
            seq = list(raw)
            expected = len(sources_map)
            if len(seq) != expected:
                raise ValueError(
                    f"weights sequence length mismatch: expected {expected}, got {len(seq)}"
                )
            w_seq = [
                _coerce_weight(v, where=f"weights[{i}]")
                for i, v in enumerate(seq)
            ]
            if not any((float(v) > 0.0) for v in w_seq):
                raise ValueError(
                    "weights sequence must contain at least one positive weight"
                )
            if not all(str(k).isdigit() for k in sources_map.keys()):
                raise ValueError(
                    "sequence weights require digit-only source keys ('0','1',...)"
                )
            w = {k: float(w_seq[int(k)]) for k in sources_map.keys()}
        else:
            raise TypeError(
                "weights must be a Mapping[str, float] or Sequence[float] (or omitted for uniform)"
            )
        if not any(v > 0.0 for v in w.values()):
            raise ValueError("At least one weight must be > 0")
        self._source_keys = list(sources_map.keys())
        node = MultiNodeWeightedSampler(
            sources_map,
            w,
            stop_criteria=self.stop_criteria,
            seed=int(self.seed),
        )
        self._node = node
        return node


class Mapper:
    def __init__(
        self: Self,
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
        wp = WorkerPolicy.optimize()
        wp.set_thread_setting()
        self.io_workers = (
            int(io_workers)
            if io_workers is not None
            else int(getattr(wp, "num_workers", 1))
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
            device
            if isinstance(device, torch.device)
            else torch.device(device)
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
            new_affinity(io_workers=self.io_workers)
        except Exception:
            pass

    def compose(self: Self, source: "BaseNode") -> "BaseNode":
        node: BaseNode = source
        mapper = self.map_fn
        if (self.prebatch or 0) and int(self.prebatch) > 1:
            B = max(1, int(self.prebatch))
            node = torchdata.nodes.Batcher(node, batch_size=B, drop_last=False)
            pm_map_fn = new_thread(mapper)
        else:
            pm_map_fn = new_thread(mapper)
        node = ParallelMapper(
            node,
            map_fn=pm_map_fn,
            num_workers=self.io_workers,
            in_order=False,
            method="thread",
            max_concurrent=int(self.max_concurrency),
        )
        if (self.prebatch or 0) and int(self.prebatch) > 1:
            node = torchdata.nodes.Unbatcher(node)
        return node


class Loader:
    def __init__(
        self: Self,
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
        node_obj = node or dataset
        if not isinstance(node_obj, BaseNode):
            raise TypeError(
                "Loader supports only torchdata.nodes.BaseNode instances."
            )
        self._device = _normalize_device_spec(device)
        self._nogil = bool(CPU.is_optimized_for_no_gil())
        self._non_blocking = bool(non_blocking)
        self._length = int(length) if length is not None else None
        depth = max(1, int(prefetch_factor))
        with suppress(Exception):
            depth_env = int(
                env_first_int(("ENN_PREFETCH_DEPTH",), default=0) or 0
            )
            if depth_env > 0:
                depth = int(depth_env)
        self._prefetch_depth = max(1, min(32, int(depth)))
        prim = _primary_device(self._device)
        dev_t = getattr(prim, "type", "cpu")
        default_pin = dev_t in {"cuda", "xpu"}
        self._pin_host = (
            bool(pin_memory) if pin_memory is not None else bool(default_pin)
        )
        if dev_t == "cuda" and self._non_blocking:
            gpu_guard_mb = 2048
        elif dev_t in {"xpu"} and self._non_blocking:
            gpu_guard_mb = 512
        else:
            gpu_guard_mb = 0
        host_guard_mb = 1024 if self._non_blocking else 0
        with suppress(Exception):
            gpu_guard_mb = int(
                env_first_int(("ENN_GPU_GUARD_MB",), default=gpu_guard_mb)
                or gpu_guard_mb
            )
        with suppress(Exception):
            host_guard_mb = int(
                env_first_int(("ENN_HOST_GUARD_MB",), default=host_guard_mb)
                or host_guard_mb
            )
        self._gpu_guard_bytes = int(max(0, gpu_guard_mb) * (1 << 20))
        self._host_guard_bytes = int(max(0, host_guard_mb) * (1 << 20))
        self._node = node_obj

        self._base_iterable = (
            node_obj
            if isinstance(node_obj, torchdata.nodes.Loader)
            else torchdata.nodes.Loader(node_obj)
        )
        self._thread2dev: Dict[int, torch.device] = {}
        self._thread2dev_lock = Mutex()
        self._threads_hint = (
            self._infer_mapper_threads(node_obj) if node_obj is not None else 1
        )
        self._num_shards = 1
        self._shard_id = 0
        try:
            acc = max(1, int(get_num_accelerators("cuda") or 0))
            if acc <= 0:
                acc = max(1, int(get_num_accelerators("xpu") or 0))
            thr = max(1, int(self._threads_hint))
            self._num_shards = acc * thr
            dev_idx = self._local_device_index()
            self._shard_id = max(
                0, min(self._num_shards - 1, int(dev_idx * thr))
            )
        except Exception:
            pass

    def __iter__(self: Self) -> Iterator[Any]:
        dev = self._device_for_current_thread()
        dev_t = getattr(dev, "type", "cpu")
        use_accel = dev_t in {"cuda", "xpu", "mps"}
        use_prefetch = bool(use_accel and self._non_blocking)

        node_obj = self._node
        base: Any = (
            node_obj
            if isinstance(node_obj, torchdata.nodes.Loader)
            else torchdata.nodes.Loader(node_obj)
        )
        with suppress(Exception):
            base.reset(None)
        self._base_iterable = base

        base_it = iter(base)
        try:
            first = next(base_it)
        except StopIteration:
            return iter(())
        iterable: Any = itertools.chain((first,), base_it)
        if use_prefetch:
            iterable = Stream(
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

    def __len__(self: Self) -> int:
        if self._length is not None:
            return int(self._length)
        try:
            return int(len(self._base_iterable))
        except Exception:
            return 1

    def _device_for_current_thread(self: Self) -> torch.device:
        if isinstance(self._device, list):
            tid = threading.get_ident()
            guard = (
                self._thread2dev_lock
                if self._nogil
                else contextlib.nullcontext()
            )
            with guard:
                dev = self._thread2dev.get(tid)
                if dev is None:
                    idx = len(self._thread2dev) % max(1, len(self._device))
                    dev = self._device[idx]
                    self._thread2dev[tid] = dev
                return dev
        return self._device

    def _infer_mapper_threads(self: Self, node: Any) -> int:
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

    def _local_device_index(self: Self) -> int:
        try:
            if is_accelerator_available("cuda"):
                return int(get_accelerator_index("cuda"))
            if is_accelerator_available("xpu"):
                return int(get_accelerator_index("xpu"))
        except Exception:
            pass
        return 0

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
        if not isinstance(source, BaseNode):
            raise TypeError(
                "Loader.compose expects a torchdata.nodes.BaseNode source."
            )
        return Loader(
            device=device,
            node=source,
            prefetch_factor=int(prefetch_factor),
            non_blocking=bool(non_blocking),
            length=length,
            pin_memory=pin_memory,
        )


class Stream(BufferQueue):
    def __init__(
        self: Self,
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
        self._device = (
            torch.device(device)
            if not isinstance(device, torch.device)
            else device
        )
        self._depth = max(1, int(depth))
        self._non_blocking = bool(non_blocking)
        self._backpressure = (
            bool(memory_backpressure)
            if memory_backpressure is not None
            else bool(oom_safe)
        )
        self._gpu_guard_bytes = int(gpu_guard_bytes or 0)
        self._host_guard_bytes = int(host_guard_bytes or 0)
        use_accel = isinstance(
            self._device, torch.device
        ) and self._device.type in (
            "cuda",
            "xpu",
            "mps",
        )
        self._pin = bool(kwargs.get("pin_host", use_accel))
        self._pin_pool = False
        self._host_pool: Optional[TensorPagePool] = None
        if (
            self._pin
            and self._non_blocking
            and is_stream_supported(self._device.type)
        ):
            use_pool = env_bool("ENN_PREFETCH_PIN_POOL", default=True)
            cap_default = max(8, max(2, int(self._depth) * 2))
            cap = env_first_int(
                ("ENN_PREFETCH_PIN_POOL_CAPACITY",), default=cap_default
            )
            if use_pool and int(cap) > 0:
                self._host_pool = TensorPagePool(
                    capacity=int(cap), pin_memory=True
                )
                self._pin_pool = True
        self._accel_stream: Optional[object] = None
        self._accel_event_pool: Optional[queue.SimpleQueue] = None
        self._session = bool(_session)
        ttl_ms = int(
            env_first_int(("ENN_PREFETCH_GUARD_TTL_MS",), default=10) or 10
        )
        self._guard_ttl_s = max(0.0, float(ttl_ms) / 1000.0)
        self._join_timeout_s = 0.5
        with suppress(Exception):
            jt_ms = int(
                env_first_int(("ENN_THREAD_JOIN_TIMEOUT_MS",), default=500)
                or 500
            )
            self._join_timeout_s = max(0.0, float(jt_ms) / 1000.0)

    def _spawn_session(self: Self) -> "Stream":
        return Stream(
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

    def _apply_structure(
        self: Self, obj: object, func: Callable[[object], object]
    ) -> object:
        with suppress(Exception):
            from tensordict import TensorDictBase

            if isinstance(obj, TensorDictBase):
                return obj.apply(
                    lambda t: func(t) if torch.is_tensor(t) else t,
                    inplace=False,
                )
        if isinstance(obj, list):
            return [self._apply_structure(x, func) for x in obj]
        if isinstance(obj, tuple):
            return type(obj)(*(self._apply_structure(x, func) for x in obj))
        if isinstance(obj, dict):
            return {
                k: (
                    v
                    if k in ("row_ids", "keys")
                    else self._apply_structure(v, func)
                )
                for k, v in obj.items()
            }
        return func(obj)

    def _to_device(self: Self, x: Any, device: torch.device) -> Any:
        with suppress(Exception):
            from tensordict import TensorDictBase

            if isinstance(x, TensorDictBase):
                return x.to(
                    device,
                    non_blocking=bool(self._non_blocking),
                    non_blocking_pin=bool(self._pin and self._non_blocking),
                )

        def _f(t: object) -> object:
            if not torch.is_tensor(t) or t.device == device:
                return t
            nb = self._non_blocking and (
                t.device.type != "cpu"
                or (hasattr(t, "is_pinned") and t.is_pinned())
            )
            return t.to(device, non_blocking=nb)

        return self._apply_structure(x, _f)

    def _pin_memory(self: Self, x: Any) -> Any:
        if not self._pin:
            return x

        with suppress(Exception):
            from tensordict import TensorDictBase

            if isinstance(x, TensorDictBase):
                with suppress(Exception):
                    return x.pin_memory()

        def _f(t: object) -> object:
            if (
                torch.is_tensor(t)
                and t.device.type == "cpu"
                and not (hasattr(t, "is_pinned") and t.is_pinned())
            ):
                return t.pin_memory()
            return t

        return self._apply_structure(x, _f)

    def _stage_with_pool(
        self: Self,
        obj: Any,
        pool: TensorPagePool,
        tokens: list[Optional[TensorPagePool.Token]],
    ) -> Any:
        def _f(t: object) -> object:
            if not (torch.is_tensor(t) and t.device.type == "cpu") or (
                hasattr(t, "is_pinned") and t.is_pinned()
            ):
                return t
            buf, tok = pool.get_like(t, return_handle=True, block=False)
            buf.copy_(t, non_blocking=False)
            tokens.append(tok)
            return buf

        return self._apply_structure(obj, _f)

    def _pin_batch(
        self: Self, x: Any
    ) -> tuple[Any, list[Optional[TensorPagePool.Token]]]:
        if not self._pin:
            return x, []
        pool = self._host_pool
        if pool is None:
            return self._pin_memory(x), []
        tokens: list[Optional[TensorPagePool.Token]] = []
        return self._stage_with_pool(x, pool, tokens), tokens

    def _producer_loop(
        self: Self,
        iterable: Any,
        sentinel: object,
        *args: Any,
        device: torch.device,
        use_device: bool,
        use_accel_stream: bool,
        gpu_guard_bytes: int,
        host_guard_bytes: int,
    ) -> None:
        last_check_t = 0.0
        last_guards_ok = True
        ttl_s = float(getattr(self, "_guard_ttl_s", 0.0) or 0.0)
        it: Iterator[Any] | None = None
        try:
            it = iter(iterable)
            if use_device and isinstance(device, torch.device):
                backend = accelerator_type(device.type)
                set_dev = (
                    getattr(backend, "set_device", None)
                    if backend is not None
                    else None
                )
                with suppress(Exception):
                    if callable(set_dev) and device.index is not None:
                        set_dev(int(device.index))
            while True:
                if self.is_stopped():
                    break
                if not self.block(timeout=None):
                    break
                try:
                    batch = next(it)
                except StopIteration:
                    break
                if self.is_stopped():
                    break
                batch, pool_tokens = self._pin_batch(batch)
                if self._backpressure:
                    sleep_s = 0.001
                    while not self.is_stopped():
                        now = time.monotonic()
                        if ttl_s <= 0.0 or (now - last_check_t) >= ttl_s:
                            last_check_t = now
                            host_ok = _host_guard_ok(host_guard_bytes)
                            dev_ok = _device_guard_ok(device, gpu_guard_bytes)
                            last_guards_ok = bool(host_ok and dev_ok)
                        if bool(last_guards_ok):
                            break
                        time.sleep(sleep_s)
                        sleep_s = min(float(sleep_s) * 2.0, 0.05)
                    if self.is_stopped():
                        if self._host_pool is not None and pool_tokens:
                            for tok in pool_tokens:
                                with suppress(Exception):
                                    self._host_pool.release(tok)
                        break
                if use_device:
                    if use_accel_stream and self._accel_stream is not None:
                        ev = None
                        pool = self._accel_event_pool
                        if pool is not None:
                            while ev is None and not self.is_stopped():
                                try:
                                    ev = pool.get(timeout=0.05)
                                except queue.Empty:
                                    continue
                        if ev is not None:
                            _wait_accel_event_done(ev, stopped=self.is_stopped)
                        try:
                            with accelerator_stream(
                                self._accel_stream, device.type
                            ):
                                batch_dev = self._to_device(batch, device)
                                if ev is not None:
                                    try:
                                        ev.record(self._accel_stream)
                                    except TypeError:
                                        ev.record()
                            if self._host_pool is not None and pool_tokens:
                                if ev is not None:
                                    for tok in pool_tokens:
                                        self._host_pool.release_after(tok, ev)
                                else:
                                    for tok in pool_tokens:
                                        self._host_pool.release(tok)
                            if not self.put((batch_dev, ev), timeout=None):
                                if ev is not None and pool is not None:
                                    _wait_accel_event_done(
                                        ev, stopped=self.is_stopped
                                    )
                                    with suppress(Exception):
                                        pool.put(ev)
                                break
                        except BaseException:
                            if self._host_pool is not None and pool_tokens:
                                for tok in pool_tokens:
                                    with suppress(Exception):
                                        self._host_pool.release(tok)
                            if ev is not None and pool is not None:
                                _wait_accel_event_done(
                                    ev, stopped=self.is_stopped
                                )
                                with suppress(Exception):
                                    pool.put(ev)
                            raise
                    else:
                        batch_dev = self._to_device(batch, device)
                        if not self.put((batch_dev, None), timeout=None):
                            break
                else:
                    if not self.put((batch, None), timeout=None):
                        break
        except BaseException as exc:
            with suppress(Exception):
                self.put(ProducerError(exc=exc, tb=traceback.format_exc()))
        finally:
            with suppress(Exception):
                if it is not None:
                    close(it)
            with suppress(Exception):
                self.put(sentinel)

    def __iter__(self: Self) -> Iterator[Any]:
        if not bool(self._session):
            yield from self._spawn_session()
            return

        device = getattr(self, "_device", torch.device("cpu"))
        use_device = device.type in {"cuda", "mps", "xpu"}
        use_accel_stream = bool(
            self._non_blocking and is_stream_supported(device.type)
        )

        iterable = getattr(self, "_iterable", self._src)
        gpu_guard_bytes = int(getattr(self, "_gpu_guard_bytes", 0) or 0)
        host_guard_bytes = int(getattr(self, "_host_guard_bytes", 0) or 0)

        if use_accel_stream and self._accel_stream is None:
            self._accel_stream = new_accelerator_stream(device)
        if use_accel_stream and self._accel_stream is None:
            use_accel_stream = False

        if use_accel_stream:
            pool: queue.SimpleQueue = queue.SimpleQueue()
            created = 0
            for _ in range(max(1, int(getattr(self, "_depth", 2) or 2))):
                ev = new_accelerator_event(device, enable_timing=False)
                if ev is not None:
                    pool.put(ev)
                    created += 1
            if created <= 0:
                use_accel_stream = False
                self._accel_event_pool = None
            else:
                self._accel_event_pool = pool
        else:
            self._accel_event_pool = None

        ttl_s = float(getattr(self, "_guard_ttl_s", 0.0) or 0.0)
        last_check_t = 0.0
        last_guards_ok = True

        def _wait_guards() -> None:
            nonlocal last_check_t, last_guards_ok
            if not self._backpressure:
                return
            sleep_s = 0.001
            while True:
                if self.is_stopped():
                    raise StopIteration
                now = time.monotonic()
                if ttl_s <= 0.0 or (now - last_check_t) >= ttl_s:
                    last_check_t = now
                    host_ok = _host_guard_ok(host_guard_bytes)
                    dev_ok = _device_guard_ok(device, gpu_guard_bytes)
                    last_guards_ok = bool(host_ok and dev_ok)
                if bool(last_guards_ok):
                    return
                time.sleep(sleep_s)
                sleep_s = min(float(sleep_s) * 2.0, 0.05)

        def _stage_one(raw_batch: Any) -> tuple[Any, Optional[object]]:
            _wait_guards()
            batch, pool_tokens = self._pin_batch(raw_batch)

            if not use_device:
                return batch, None

            if use_accel_stream and self._accel_stream is not None:
                ev = None
                pool = self._accel_event_pool
                if pool is not None:
                    while ev is None and not self.is_stopped():
                        try:
                            ev = pool.get(timeout=0.05)
                        except queue.Empty:
                            continue
                if ev is not None:
                    _wait_accel_event_done(ev, stopped=self.is_stopped)
                try:
                    with accelerator_stream(self._accel_stream, device.type):
                        batch_dev = self._to_device(batch, device)
                        if ev is not None:
                            try:
                                ev.record(self._accel_stream)
                            except TypeError:
                                ev.record()
                    if self._host_pool is not None and pool_tokens:
                        if ev is not None:
                            for tok in pool_tokens:
                                self._host_pool.release_after(tok, ev)
                        else:
                            for tok in pool_tokens:
                                self._host_pool.release(tok)
                    return batch_dev, ev
                except BaseException:
                    if self._host_pool is not None and pool_tokens:
                        for tok in pool_tokens:
                            with suppress(Exception):
                                self._host_pool.release(tok)
                    if ev is not None and pool is not None:
                        _wait_accel_event_done(ev, stopped=self.is_stopped)
                        with suppress(Exception):
                            pool.put(ev)
                    raise
            else:
                batch_dev = self._to_device(batch, device)
                if self._host_pool is not None and pool_tokens:
                    for tok in pool_tokens:
                        with suppress(Exception):
                            self._host_pool.release(tok)
                return batch_dev, None

        buf = collections.deque()

        src_it = iter(iterable)

        try:
            for _ in range(max(1, int(getattr(self, "_depth", 2) or 2))):
                try:
                    raw = next(src_it)
                except StopIteration:
                    break
                buf.append(_stage_one(raw))

            if not buf:
                raise RuntimeError(
                    "Stream: source yielded 0 items during prefetch priming. "
                    "This usually means the upstream iterable is exhausted/stateful and was reused, "
                    "or the dataset filtered out all samples. Recreate/reset the Loader/BaseNode per run."
                )

            while buf:
                batch, ev = buf.popleft()

                if use_accel_stream and ev is not None:
                    cs = current_accelerator_stream(device)
                    if cs is not None:
                        with suppress(Exception):
                            cs.wait_event(ev)

                        with suppress(Exception):
                            try:
                                ev.record(cs)
                            except TypeError:
                                ev.record()
                    pool = self._accel_event_pool
                    if pool is not None:
                        with suppress(Exception):
                            pool.put(ev)

                yield batch

                try:
                    raw = next(src_it)
                except StopIteration:
                    continue
                buf.append(_stage_one(raw))
        finally:
            self.stop()
            with suppress(Exception):
                close(src_it)
            pool = self._accel_event_pool
            if use_accel_stream and pool is not None:
                while True:
                    try:
                        ev = pool.get_nowait()
                    except queue.Empty:
                        break
                    _wait_accel_event_done(ev, stopped=self.is_stopped)
                self._accel_event_pool = None
                self._accel_stream = None
