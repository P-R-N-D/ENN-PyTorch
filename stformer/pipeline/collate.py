# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Tuple, Union

import os
import collections
import time
from functools import partial
from contextlib import suppress, nullcontext

import numpy as np
import torch
from torchdata.nodes import ParallelMapper, Prefetcher as PrefetcherCompat, PinMemory, IterableWrapper

from .dataset import MemoryMappedTensorStream
from ..toolkit.capability import apply_threading_defaults
from ..connection.socket import ArrowFlight
from .distributed import is_initialized
from ..toolkit.optimization import DataLoader


def _world_info() -> Tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", os.environ.get("PMI_RANK", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("PMI_SIZE", "1")))
    return rank, local_rank, world_size, world_size > 1


class Prefetcher(Iterator[Any]):
    def __init__(
        self,
        iterable: Iterable[Any],
        device: Optional[Union[str, torch.device]],
        *args: Any,
        depth: int = 2,
        slots: int = 2,
        pin_if_needed: bool = True,
        use_record_stream: bool = True,
        amp_dtype: Optional[torch.dtype] = None,
        max_bytes: Optional[int] = None,
        autotune: bool = True,
        tune_interval: int = 50,
        depth_min: int = 1,
        depth_max: int = 8,
        enable_graphs: bool = False,
        graph_warmup: int = 2,
        **kwargs: Any
    ) -> None:
        self._it = iter(iterable)
        self._dev: torch.device = torch.device(device) if not isinstance(device, torch.device) else device
        self._backend: str = self._dev.type
        self._depth_min: int = int(max(1, depth_min))
        self._depth_max: int = int(max(self._depth_min, depth_max))
        self._queue: collections.deque = collections.deque(maxlen=max(1, int(depth)))
        self._pin_if_needed: bool = bool(pin_if_needed)
        self._use_record_stream: bool = bool(use_record_stream)
        self._amp_dtype: Optional[torch.dtype] = amp_dtype
        self._max_bytes: Optional[int] = max_bytes if isinstance(max_bytes, int) and max_bytes > 0 else None
        self._bytes_in_q: int = 0
        self._autotune: bool = bool(autotune)
        self._tune_interval: int = max(10, int(tune_interval))
        self._graph_enabled: bool = bool(enable_graphs and self._backend == "cuda")
        self._slots: int = 1 if self._graph_enabled else max(1, int(slots))
        self._streams: list[Any] = []
        try:
            if self._backend == "cuda":
                self._streams = [torch.cuda.Stream(device=self._dev) for _ in range(self._slots)]
            elif hasattr(torch, "xpu") and self._backend == "xpu":
                _XpuStream = getattr(torch.xpu, "Stream", None)
                self._streams = [_XpuStream() for _ in range(self._slots)] if _XpuStream is not None else []
            elif hasattr(torch, "mps") and self._backend == "mps":
                _MpsStream = getattr(torch.mps, "Stream", None)
                self._streams = [_MpsStream() for _ in range(self._slots)] if _MpsStream is not None else []
        except Exception:
            self._streams = []
        self._rr: int = 0
        self._copy_time_acc: float = 0.0
        self._steps: int = 0
        self._starved: int = 0
        self._last_yield_ts: Optional[float] = None
        self._ema_compute: float = 0.0
        self._static: Any = None
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._graph_warmup: int = max(1, int(graph_warmup))
        self._err: Optional[Exception] = None
        self._preload()

    def _map(self, o: Any, fn: Any) -> Any:
        if isinstance(o, (list, tuple)):
            return type(o)((self._map(x, fn) for x in o))
        if isinstance(o, dict):
            return {k: self._map(v, fn) for k, v in o.items()}
        return fn(o)

    def _bytes(self, o: Any) -> int:
        if isinstance(o, torch.Tensor):
            return int(getattr(o, "nbytes", o.numel() * max(1, o.element_size())))
        if isinstance(o, (list, tuple)):
            return sum(self._bytes(x) for x in o)
        if isinstance(o, dict):
            return sum(self._bytes(v) for v in o.values())
        return 0

    def _should_pin(self, t: Any) -> bool:
        if not isinstance(t, torch.Tensor):
            return False
        if t.device.type != "cpu":
            return False
        if self._backend not in ("cuda", "xpu", "mps"):
            return False
        if self._pin_if_needed and hasattr(t, "is_pinned") and t.is_pinned():
            return False
        return self._pin_if_needed

    def _pin_cpu(self, o: Any) -> Any:
        def _pin(t: Any) -> Any:
            if self._should_pin(t):
                with suppress(Exception):
                    return t.pin_memory()
            return t

        return self._map(o, _pin)

    def _to_device(self, o: Any) -> Any:
        def _mv(t: Any) -> Any:
            if isinstance(t, torch.Tensor):
                if self._amp_dtype is not None:
                    return t.to(self._dev, dtype=self._amp_dtype, non_blocking=self._backend in {"cuda", "xpu", "mps"})
                return t.to(self._dev, non_blocking=self._backend in {"cuda", "xpu", "mps"})
            return t

        return self._map(o, _mv)

    def _allocate_static_like(self, o: Any) -> Any:
        def _alloc(t: Any) -> Any:
            if isinstance(t, torch.Tensor):
                shape, dtype = t.shape, (self._amp_dtype or t.dtype)
                return torch.empty(shape, device=self._dev, dtype=dtype)
            return t

        return self._map(o, _alloc)

    def _copy_into(self, dst: Any, src: Any) -> Any:
        def _copy(d: Any, s: Any) -> Any:
            if isinstance(d, torch.Tensor) and isinstance(s, torch.Tensor):
                if d.dtype != (self._amp_dtype or s.dtype):
                    s = s.to(dtype=self._amp_dtype or s.dtype)
                d.copy_(s, non_blocking=self._backend in {"cuda", "xpu", "mps"})
                return d
            if isinstance(d, (list, tuple)) and isinstance(s, (list, tuple)):
                return type(d)((_copy(dd, ss) for dd, ss in zip(d, s)))
            if isinstance(d, dict) and isinstance(s, dict):
                return {k: _copy(d[k], s[k]) for k in d.keys()}
            return d
        return _copy(dst, src)

    def _current_stream(self) -> Any:
        if self._backend == "cuda":
            return torch.cuda.current_stream(self._dev)
        if self._backend == "xpu" and hasattr(torch, "xpu"):
            return torch.xpu.current_stream(self._dev)
        if self._backend == "mps" and hasattr(torch, "mps"):
            return torch.mps.current_stream()
        return None

    def _stream_ctx(self, stream: Any) -> Any:
        if stream is None:
            return nullcontext()
        if self._backend == "cuda":
            return torch.cuda.stream(stream)
        if self._backend == "xpu" and hasattr(torch, "xpu"):
            return torch.xpu.stream(stream)
        if self._backend == "mps" and hasattr(torch, "mps"):
            return torch.mps.stream(stream)
        return nullcontext()

    def _preload(self) -> None:
        while len(self._queue) < self._queue.maxlen:
            if self._max_bytes is not None and self._bytes_in_q >= self._max_bytes:
                break
            try:
                batch = next(self._it)
            except StopIteration:
                break
            except Exception as e:
                self._err = self._err or e
                break
            batch = self._pin_cpu(batch)
            slot = self._rr % self._slots
            stream = self._streams[slot] if self._streams else None
            t0 = time.perf_counter()
            with self._stream_ctx(stream):
                if self._graph_enabled:
                    if self._static is None:
                        self._static = self._allocate_static_like(self._to_device(batch))
                    moved = self._copy_into(self._static, self._to_device(batch))
                else:
                    moved = self._to_device(batch)
            t1 = time.perf_counter()
            self._copy_time_acc += (t1 - t0)
            self._queue.append((moved, slot))
            self._bytes_in_q += self._bytes(moved)
            self._rr += 1

    def __iter__(self) -> "Prefetcher":
        return self

    def __next__(self) -> Any:
        if not self._queue:
            if self._err is not None:
                e, self._err = self._err, None
                raise e
            raise StopIteration
        if len(self._queue) <= 1:
            self._starved += 1
        now = time.perf_counter()
        if self._last_yield_ts is not None:
            dt = now - self._last_yield_ts
            self._ema_compute = 0.9 * self._ema_compute + 0.1 * dt if self._ema_compute > 0 else dt
        moved, slot = self._queue.popleft()
        self._bytes_in_q -= min(self._bytes_in_q, self._bytes(moved))
        producer = self._streams[slot] if self._streams else None
        consumer = self._current_stream()
        if producer is not None and consumer is not None and hasattr(consumer, "wait_stream"):
            consumer.wait_stream(producer)
        if self._use_record_stream and consumer is not None:
            def _rec(o: Any) -> None:
                if isinstance(o, torch.Tensor) and o.device.type == self._backend:
                    with suppress(Exception):
                        o.record_stream(consumer)
                elif isinstance(o, (list, tuple)):
                    for z in o:
                        _rec(z)
                elif isinstance(o, dict):
                    for z in o.values():
                        _rec(z)
            _rec(moved)
        self._preload()
        self._steps += 1
        if self._autotune and self._steps % self._tune_interval == 0:
            self._retune_depth()
            self._copy_time_acc = 0.0
            self._starved = 0
        self._last_yield_ts = time.perf_counter()
        return moved

    def _set_depth(self, new_depth: int) -> None:
        new_depth = int(max(self._depth_min, min(self._depth_max, new_depth)))
        if new_depth == self._queue.maxlen:
            return
        self._queue = collections.deque(self._queue, maxlen=new_depth)

    def _retune_depth(self) -> None:
        steps = float(max(1, self._tune_interval))
        starve_ratio = float(self._starved) / steps
        avg_copy = self._copy_time_acc / steps
        avg_comp = max(1e-6, self._ema_compute)
        want_up = starve_ratio > 0.30 or avg_copy > 0.30 * avg_comp
        want_dn = starve_ratio < 0.05 and len(self._queue) >= self._queue.maxlen - 1
        if want_up and self._queue.maxlen < self._depth_max:
            self._set_depth(self._queue.maxlen + 1)
        elif want_dn and self._queue.maxlen > self._depth_min:
            self._set_depth(self._queue.maxlen - 1)

    def close(self) -> None:
        with suppress(Exception):
            self._queue.clear()
        self._streams = []
        self._static = None
        self._graph = None

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def graphs_capture(self, capture_fn: Any) -> bool:
        if not self._graph_enabled:
            raise RuntimeError("enable_graphs=True && backend=cuda 에서만 사용 가능")
        if not self._queue:
            self._preload()
        moved, _ = self._queue[0]
        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            capture_fn(moved)
        self._graph = g
        return True

    def graphs_replay(self) -> None:
        if self._graph is None:
            raise RuntimeError("graphs_capture()를 먼저 호출하세요")
        _ = next(self)
        self._graph.replay()


def _torch_dtype_to_arrow_dtype(dt: torch.dtype) -> str:
    return {
        torch.float32: "float32",
        torch.float64: "float64",
        torch.float16: "float16",
        getattr(torch, "bfloat16", object()): "bfloat16",
        torch.int64: "int64",
        torch.int32: "int32",
        torch.int16: "int16",
        torch.int8: "int8",
        torch.uint8: "uint8",
        torch.bool: "bool",
    }.get(dt, "float32")


def _torch_dtype_to_numpy_dtype(dt: torch.dtype) -> Any:
    return {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.float16: np.float16,
        getattr(torch, "bfloat16", object()): np.float32,
        torch.int64: np.int64,
        torch.int32: np.int32,
        torch.int16: np.int16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.bool: np.uint8,
    }.get(dt, np.float32)


def _map_dtype(x: Any, *, dt: Optional[torch.dtype]) -> Any:
    if dt is None:
        return x
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dt, copy=False)
    if isinstance(x, (list, tuple)):
        return type(x)((_map_dtype(t, dt=dt) for t in x))
    if isinstance(x, dict):
        return {k: _map_dtype(v, dt=dt) for k, v in x.items()}
    return x


def _flatten(objs: Iterable[Any]) -> Iterable[Any]:
    for o in objs:
        if o is None:
            continue
        if isinstance(o, (list, tuple, set)):
            for x in _flatten(o):
                if x is not None:
                    yield x
        else:
            yield o


class _Keep:
    __slots__ = ("_objs",)

    def __init__(self, *objs: Any) -> None:
        self._objs = list(_flatten(objs))

    def add(self, *objs: Any) -> None:
        self._objs.extend(list(_flatten(objs)))

    def cleanup(self) -> None:
        for obj in self._objs:
            for name in ("cleanup", "close", "shutdown", "stop", "terminate", "join", "disconnect", "release"):
                if hasattr(obj, name):
                    with suppress(Exception):
                        getattr(obj, name)()
                    break
            else:
                if callable(obj):
                    with suppress(Exception):
                        obj()


def forward(
    batch: Mapping[str, Any],
    *args: Any,
    labels_dtype: Optional[torch.dtype] = None,
    sanitize: bool = False,
    flatten_features: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    X, Y = batch["X"], batch["Y"]
    if flatten_features and isinstance(X, torch.Tensor):
        if X.dim() >= 2:
            X = X.flatten(start_dim=1)
    if not isinstance(Y, torch.Tensor):
        if hasattr(Y, "to_tensor"):
            Y = Y.to_tensor()
        elif hasattr(Y, "as_tensor"):
            Y = Y.as_tensor()
        else:
            Y = torch.as_tensor(Y)
    if labels_dtype is not None and getattr(Y, "dtype", None) != labels_dtype:
        Y = Y.to(dtype=labels_dtype)
    if sanitize and torch.is_floating_point(Y):
        Y = torch.nan_to_num(Y, nan=0.0, posinf=1_000_000.0, neginf=-1_000_000.0)
    return {"X": X, "Y": Y}


def dispatch(sample_or_list: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(sample_or_list, (list, tuple)):
        return [forward(*args, s, **kwargs) for s in sample_or_list]
    return forward(*args, sample_or_list, **kwargs)


def stream(
    memmap_dir: str,
    device: Union[str, torch.device],
    batch_size: int,
    val_frac: float,
    *args: Any,
    prefetch_factor: int = 2,
    non_blocking_copy: bool = True,
    labels_dtype: Optional[torch.dtype] = None,
    sanitize: bool = False,
    flatten_features: bool = False,
    **kwargs: Any,
) -> Tuple[Any, Optional[Any], _Keep]:
    rank, local_rank, world_size, is_ddp = _world_info()
    node_id = ArrowFlight.node_id(rank, local_rank)
    name_train = ArrowFlight.resource_key(memmap_dir, "train")
    name_val = ArrowFlight.resource_key(memmap_dir, "val")
    reader_tr = MemoryMappedTensorStream.from_dir(memmap_dir, split="train", batch_size=int(batch_size), val_frac=val_frac)
    reader_vl = MemoryMappedTensorStream.from_dir(memmap_dir, split="val", batch_size=int(batch_size), val_frac=val_frac) if val_frac > 0 else None
    meta = reader_tr._load_meta()
    srv = None
    if not is_ddp or local_rank == 0:
        srv, uri = ArrowFlight.start_server_standby(host="0.0.0.0", port=0)
        ArrowFlight.reg_mmt_dataset(srv, name_train, reader_tr, int(batch_size), "train")
        if reader_vl is not None:
            ArrowFlight.reg_mmt_dataset(srv, name_val, reader_vl, int(batch_size), "val")
        if is_initialized():
            ArrowFlight.MQ.publish(ArrowFlight.server_key(node_id), uri)
        uri0 = uri
    else:
        uri0 = ArrowFlight.MQ.wait(ArrowFlight.server_key(node_id), timeout_s=120.0)
    cli = ArrowFlight.Client(uri0)
    lshape = list(meta.get("label_shape", []))

    def _iter(name: str) -> Iterator[Dict[str, torch.Tensor]]:
        rdr = cli.reader(name)
        for rb in rdr:
            B = int(getattr(rb, "num_rows", 0)) or int(len(rb.column(0)))
            x_arr = rb.column(0)
            xnp = x_arr.to_numpy(zero_copy_only=False)
            if not isinstance(xnp, np.ndarray):
                xnp = np.array(xnp, copy=True)
            if getattr(xnp, "flags", None) is None or (not xnp.flags.writeable) or (not xnp.flags.c_contiguous):
                xnp = np.ascontiguousarray(xnp).copy()
            if xnp.dtype != np.float32:
                xnp = xnp.astype(np.float32, copy=False)
            Xb = torch.from_numpy(xnp).view(B, -1)
            y_arr = rb.column(1)
            ynp = y_arr.to_numpy(zero_copy_only=False)
            if not isinstance(ynp, np.ndarray):
                ynp = np.array(ynp, copy=True)
            if getattr(ynp, "flags", None) is None or (not ynp.flags.writeable) or (not ynp.flags.c_contiguous):
                ynp = np.ascontiguousarray(ynp).copy()
            Yb = torch.from_numpy(ynp)
            if lshape:
                Yb = Yb.reshape(B, -1).view(B, *lshape)
            else:
                Yb = Yb.reshape(B, -1)
            yield {"X": Xb, "Y": Yb}

    thr = apply_threading_defaults()
    map_fn = partial(dispatch, labels_dtype=labels_dtype, sanitize=sanitize, flatten_features=flatten_features)
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    node_tr = IterableWrapper(_iter(name_train))
    node_tr = ParallelMapper(
        node_tr,
        map_fn=map_fn,
        num_workers=thr["dataloader_workers"],
        method="thread",
        in_order=False,
        max_concurrent=thr["dataloader_workers"],
        prebatch=thr["prefetch_factor"],
    )
    node_tr = PrefetcherCompat(node_tr, prefetch_factor=prefetch_factor)
    node_tr = PinMemory(node_tr, pin_memory_device=dev.type)
    train_loader = DataLoader(device=device, node=node_tr, prefetch_factor=prefetch_factor, non_blocking=bool(non_blocking_copy))
    if reader_vl is not None:
        node_vl = IterableWrapper(_iter(name_val))
        node_vl = ParallelMapper(
            node_vl,
            map_fn=map_fn,
            num_workers=thr["dataloader_workers"],
            method="thread",
            in_order=False,
            max_concurrent=thr["dataloader_workers"],
            prebatch=thr["prefetch_factor"],
        )
        node_vl = PrefetcherCompat(node_vl, prefetch_factor=prefetch_factor)
        node_vl = PinMemory(node_vl, pin_memory_device=dev.type)
        val_loader = DataLoader(device=device, node=node_vl, prefetch_factor=prefetch_factor, non_blocking=bool(non_blocking_copy))
    else:
        val_loader = None
    keep = _Keep(reader_tr, reader_vl, cli, srv)
    return train_loader, val_loader, keep