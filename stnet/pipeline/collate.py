# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import os
import socket
import time
import warnings
from collections import deque
from contextlib import nullcontext, suppress
from functools import partial
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.distributed as dist
from torchdata.nodes import (
    BaseNode,
    IterableWrapper,
    Loader as TorchDataLoader,
    ParallelMapper,
    PinMemory,
    Prefetcher,
)

from torch.distributed import distributed_c10d

from ..connection.socket import Endpoint
from ..toolkit.capability import apply_threading_defaults, get_world_size
from ..toolkit.compat import has_arrow_flight, patch_arrow
from .dataset import BatchStream, MemoryMappedTensorStream


_ARROW = patch_arrow()


def get_node_length(node: Any) -> Any:
    try:
        n = len(node)
        if isinstance(n, int) and n >= 0:
            return n
    except Exception:
        pass
    for key in ("num_batches", "n_batches", "steps", "length"):
        value = getattr(node, key, None)
        if isinstance(value, int) and value >= 0:
            return value
    batch_size = getattr(node, "batch_size", None)
    drop_last = bool(getattr(node, "drop_last", False))
    for key in ("num_samples", "n_samples", "N", "size", "rows", "count"):
        num_samples = getattr(node, key, None)
        if isinstance(num_samples, int) and num_samples >= 0:
            if isinstance(batch_size, int) and batch_size > 0:
                if drop_last:
                    return num_samples // batch_size
                return int(math.ceil(num_samples / batch_size))
            return max(num_samples, 1)
    for key in ("indices", "_indices", "ids", "_ids", "index", "_index"):
        idx = getattr(node, key, None)
        try:
            num_samples = len(idx)
            if isinstance(batch_size, int) and batch_size > 0:
                if drop_last:
                    return num_samples // batch_size
                return int(math.ceil(num_samples / batch_size))
            return max(int(num_samples), 1)
        except Exception:
            pass
    for name in (
        "node",
        "_node",
        "source",
        "_source",
        "dataset",
        "_dataset",
        "parent",
        "_parent",
        "base",
        "_base",
        "reader",
        "_reader",
        "upstream",
        "_upstream",
    ):
        upstream = getattr(node, name, None)
        if upstream is not None:
            try:
                length = get_node_length(upstream)
                if isinstance(length, int) and length >= 0:
                    return length
            except Exception:
                pass
    steps = None
    if not steps:
        dataset = getattr(node, "dataset", None)
        try:
            n = len(dataset) if dataset is not None else None
        except Exception:
            n = None
        batch_size = getattr(node, "batch_size", None)
        if n is not None and batch_size:
            steps = max(1, int(math.ceil(n / batch_size)))
    if not steps:
        sampler = getattr(node, "batch_sampler", None) or getattr(
            node, "sampler", None
        )
        try:
            steps = len(sampler) if sampler is not None else None
        except Exception:
            steps = None
    if not steps:
        steps = 1
    return steps


def preprocess(
    data: Dict[Tuple, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple], Tuple[int, ...]]:
    def _to_tuple(x: Any) -> Any:
        if isinstance(x, tuple):
            return x
        if isinstance(x, list):
            return tuple(x)
        if isinstance(x, torch.Tensor):
            return tuple(x.flatten().detach().cpu().tolist())
        if hasattr(x, "tolist"):
            v = x.tolist()
            return tuple(v if isinstance(v, (list, tuple)) else [v])
        return (x,)

    def _feat_row(x_tuple: Any) -> Any:
        try:
            vals = [float(v) for v in _to_tuple(x_tuple)]
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "preprocess: feature tuples must contain only numeric values. "
                f"Invalid value={x_tuple!r}"
            ) from exc
        return torch.as_tensor(vals, dtype=torch.float32)

    def _lbl(y: Any) -> Any:
        if isinstance(y, torch.Tensor):
            return y
        if hasattr(y, "to_tensor"):
            return y.to_tensor()
        return torch.as_tensor(y)

    def _maybe_batch(
        x_value: Any, y_value: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple], Tuple[int, ...]] | None:
        def _feature_tensor(value: Any) -> torch.Tensor | None:
            if isinstance(value, torch.Tensor):
                return value
            if hasattr(value, "to_tensor"):
                try:
                    return value.to_tensor()
                except Exception:
                    return None
            if hasattr(value, "as_tensor"):
                try:
                    return value.as_tensor()
                except Exception:
                    return None
            try:
                return torch.as_tensor(value)
            except Exception:
                return None

        x_tensor = _feature_tensor(x_value)
        if x_tensor is None:
            return None
        try:
            y_tensor = _lbl(y_value)
        except Exception:
            return None
        if not isinstance(y_tensor, torch.Tensor):
            return None
        x_tensor = x_tensor.detach().to(dtype=torch.float32)
        if x_tensor.dim() == 0:
            x_tensor = x_tensor.reshape(1, 1)
        elif x_tensor.dim() == 1:
            x_tensor = x_tensor.reshape(-1, 1)
        else:
            batch_dim = int(x_tensor.shape[0]) if x_tensor.shape else 1
            x_tensor = x_tensor.reshape(batch_dim, -1)
        batch = int(x_tensor.shape[0])
        y_tensor = y_tensor.detach()
        if y_tensor.dim() == 0:
            y_tensor = y_tensor.unsqueeze(0)
        if y_tensor.dim() == 1 and y_tensor.shape[0] == batch:
            y_tensor = y_tensor.unsqueeze(-1)
        if y_tensor.shape[0] != batch:
            return None
        label_shape = tuple(y_tensor.shape[1:])
        keys = [(int(i),) for i in range(batch)]
        return (x_tensor, y_tensor, keys, label_shape)

    if isinstance(data, dict) and "X" in data and ("Y" in data):
        x, y = (data["X"], data["Y"])
        batch_result = _maybe_batch(x, y)
        if batch_result is not None:
            return batch_result
        xr, yt = (_feat_row(x).unsqueeze(0), _lbl(y))
        if yt.dim() == 0 or yt.dim() == 1:
            yt = yt.unsqueeze(0)
        keys = [_to_tuple(x)]
        label_shape = tuple(yt.shape[1:])
        return (xr, yt, keys, label_shape)
    if isinstance(data, (tuple, list)) and len(data) >= 2:
        x, y = (data[0], data[1])
        batch_result = _maybe_batch(x, y)
        if batch_result is not None:
            return batch_result
        xr = _feat_row(x).unsqueeze(0)
        yt = _lbl(y)
        if yt.dim() == 0:
            yt = yt.unsqueeze(0)
        elif yt.shape[0] != 1:
            yt = yt.unsqueeze(0)
        keys = [_to_tuple(x)]
        label_shape = tuple(yt.shape[1:])
        return (xr, yt, keys, label_shape)
    if isinstance(data, dict) and len(data) > 0:
        items = list(data.items())
        if any((isinstance(k, str) for k, _ in items)):
            raise TypeError(
                "preprocess: keys in a multi-sample dict must be tuples. "
                "Provide single samples as {'X': ..., 'Y': ...}."
            )
        keys: List[Tuple] = [_to_tuple(k) for k, _ in items]
        feats = torch.stack([_feat_row(k) for k in keys], dim=0)
        lbl_list = [_lbl(v) for _, v in items]
        if all((t.shape == lbl_list[0].shape for t in lbl_list)):
            labels = torch.stack(lbl_list, dim=0)
        else:
            labels = torch.cat([t.unsqueeze(0) for t in lbl_list], dim=0)
        label_shape = tuple(labels.shape[1:])
        return (feats, labels, keys, label_shape)
    raise ValueError(
        "preprocess: unsupported input format. Provide a dict or an (X, Y) pair."
    )


def postprocess(
    keys: List[Tuple], preds: torch.Tensor | Sequence[torch.Tensor]
) -> Dict[Tuple, torch.Tensor]:
    if isinstance(preds, torch.Tensor):
        if preds.dim() == 0:
            preds = preds.unsqueeze(0)
        if preds.shape[0] != len(keys):
            raise ValueError(
                f"preds batch={preds.shape[0]} != len(keys)={len(keys)}"
            )
        rows = [preds[i].detach().cpu() for i in range(len(keys))]
    else:
        if len(preds) != len(keys):
            raise ValueError(
                f"len(preds)={len(preds)} != len(keys)={len(keys)}"
            )
        rows = [
            p.detach().cpu()
            if isinstance(p, torch.Tensor)
            else torch.as_tensor(p)
            for p in preds
        ]
    fixed_keys: List[Tuple] = []
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
    return {k: v for k, v in zip(fixed_keys, rows)}


def _world_info() -> Tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", os.environ.get("PMI_RANK", "0")))
    local_rank = int(
        os.environ.get(
            "LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")
        )
    )
    world_env = os.environ.get("WORLD_SIZE", os.environ.get("PMI_SIZE"))
    world: Optional[int] = None
    if world_env is not None:
        with suppress(ValueError):
            world = int(world_env)
    if world is None:
        with suppress(Exception):
            world = int(get_world_size())
    world = world or 1
    return (rank, local_rank, world, world > 1)


def _negotiate() -> Tuple[int, bool]:
    if not (dist.is_available() and dist.is_initialized()):
        return (0, True)
    world = get_world_size()
    rank = dist.get_rank()
    hostname = socket.gethostname()
    hosts: list[str] = ["" for _ in range(world)]
    dist.all_gather_object(hosts, hostname)
    peers = [index for index, value in enumerate(hosts) if value == hostname]
    leader_rank = min(peers) if peers else rank
    ordered = list(dict.fromkeys(sorted(hosts)))
    node_id = ordered.index(hostname) if hostname in ordered else 0
    return (node_id, rank == leader_rank)


def _new_flight(split: str, name: str, uri: str) -> Tuple[str, str]:
    if not (dist.is_available() and dist.is_initialized()):
        return (name, uri)
    node_id, is_leader = _negotiate()
    store = distributed_c10d._get_default_store()
    key = f"flight/endpoints/{node_id}/{split}".encode("utf-8")
    if is_leader:
        store.set(key, f"{name}|{uri}".encode("utf-8"))
    dist.barrier()
    value = store.get(key).decode("utf-8")
    published_name, published_uri = value.split("|", 1)
    return (published_name, published_uri)


class H2DController(Iterator[Any]):
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
        **kwargs: Any,
    ) -> None:
        self._it = iter(iterable)
        self._dev = (
            torch.device(device)
            if not isinstance(device, torch.device)
            else device
        )
        self._backend = self._dev.type
        self._depth_min = int(max(1, depth_min))
        self._depth_max = int(max(self._depth_min, depth_max))
        self._queue: Deque[Tuple[Any, int]] = deque(maxlen=max(1, int(depth)))
        self._pin_if_needed = bool(pin_if_needed)
        self._use_record_stream = bool(use_record_stream)
        self._amp_dtype = amp_dtype
        self._max_bytes = (
            max_bytes if isinstance(max_bytes, int) and max_bytes > 0 else None
        )
        self._bytes_in_q = 0
        self._autotune = bool(autotune)
        self._tune_interval = max(10, int(tune_interval))
        self._graph_enabled = bool(enable_graphs and self._backend == "cuda")
        self._slots = 1 if self._graph_enabled else max(1, int(slots))
        self._streams: list[Any] = []
        self._streams = self._init_streams()
        self._rr = 0
        self._copy_time_acc = 0.0
        self._steps = 0
        self._starved = 0
        self._last_yield_ts: Optional[float] = None
        self._ema_compute = 0.0
        self._static: Any = None
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._graph_warmup = max(1, int(graph_warmup))
        self._err: Optional[BaseException] = None
        self._preload()

    def _init_streams(self) -> list[Any]:
        try:
            if self._backend == "cuda":
                return [
                    torch.cuda.Stream(device=self._dev)
                    for _ in range(self._slots)
                ]
            if self._backend == "xpu" and hasattr(torch, "xpu"):
                stream_type = getattr(torch.xpu, "Stream", None)
                if stream_type is not None:
                    return [stream_type() for _ in range(self._slots)]
            if self._backend == "mps" and hasattr(torch, "mps"):
                stream_type = getattr(torch.mps, "Stream", None)
                if stream_type is not None:
                    return [stream_type() for _ in range(self._slots)]
        except Exception:
            return []
        return []

    def _map(self, obj: Any, fn: Callable[[Any], Any]) -> Any:
        if isinstance(obj, (list, tuple)):
            return type(obj)((self._map(item, fn) for item in obj))
        if isinstance(obj, dict):
            return {key: self._map(value, fn) for key, value in obj.items()}
        return fn(obj)

    def _bytes(self, obj: Any) -> int:
        if isinstance(obj, torch.Tensor):
            return int(
                getattr(
                    obj, "nbytes", obj.numel() * max(1, obj.element_size())
                )
            )
        if isinstance(obj, (list, tuple)):
            return sum((self._bytes(item) for item in obj))
        if isinstance(obj, dict):
            return sum((self._bytes(value) for value in obj.values()))
        return 0

    def _should_pin(self, tensor: Any) -> bool:
        if not isinstance(tensor, torch.Tensor):
            return False
        if tensor.device.type != "cpu":
            return False
        if self._backend not in {"cuda", "xpu", "mps"}:
            return False
        if (
            self._pin_if_needed
            and hasattr(tensor, "is_pinned")
            and tensor.is_pinned()
        ):
            return False
        return self._pin_if_needed

    def _pin_cpu(self, obj: Any) -> Any:
        def _pin(tensor: Any) -> Any:
            if self._should_pin(tensor):
                with suppress(Exception):
                    return tensor.pin_memory()
            return tensor

        return self._map(obj, _pin)

    def _to_device(self, obj: Any) -> Any:
        def _move(tensor: Any) -> Any:
            if isinstance(tensor, torch.Tensor):
                kwargs: Dict[str, Any] = {
                    "non_blocking": self._backend in {"cuda", "xpu", "mps"}
                }
                if self._amp_dtype is not None:
                    kwargs["dtype"] = self._amp_dtype
                return tensor.to(self._dev, **kwargs)
            return tensor

        return self._map(obj, _move)

    def _allocate_static_like(self, obj: Any) -> Any:
        def _alloc(tensor: Any) -> Any:
            if isinstance(tensor, torch.Tensor):
                dtype = self._amp_dtype or tensor.dtype
                return torch.empty(tensor.shape, device=self._dev, dtype=dtype)
            return tensor

        return self._map(obj, _alloc)

    def _copy_into(self, dst: Any, src: Any) -> Any:
        def _copy(dst_obj: Any, src_obj: Any) -> Any:
            if isinstance(dst_obj, torch.Tensor) and isinstance(
                src_obj, torch.Tensor
            ):
                dtype = self._amp_dtype or src_obj.dtype
                if dst_obj.dtype != dtype:
                    src_obj = src_obj.to(dtype=dtype)
                dst_obj.copy_(
                    src_obj,
                    non_blocking=self._backend in {"cuda", "xpu", "mps"},
                )
                return dst_obj
            if isinstance(dst_obj, (list, tuple)) and isinstance(
                src_obj, (list, tuple)
            ):
                return type(dst_obj)(
                    (_copy(d, s) for d, s in zip(dst_obj, src_obj))
                )
            if isinstance(dst_obj, dict) and isinstance(src_obj, dict):
                return {
                    key: _copy(dst_obj[key], src_obj[key]) for key in dst_obj
                }
            return dst_obj

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

    def _enqueue(self, batch: Any, slot: int) -> None:
        stream = self._streams[slot] if self._streams else None
        start = time.perf_counter()
        with self._stream_ctx(stream):
            if self._graph_enabled:
                if self._static is None:
                    self._static = self._allocate_static_like(
                        self._to_device(batch)
                    )
                moved = self._copy_into(self._static, self._to_device(batch))
            else:
                moved = self._to_device(batch)
        end = time.perf_counter()
        self._copy_time_acc += end - start
        self._queue.append((moved, slot))
        self._bytes_in_q += self._bytes(moved)
        self._rr += 1

    def _preload(self) -> None:
        while len(self._queue) < self._queue.maxlen:
            if (
                self._max_bytes is not None
                and self._bytes_in_q >= self._max_bytes
            ):
                break
            try:
                batch = next(self._it)
            except StopIteration:
                break
            except Exception as exc:
                self._err = self._err or exc
                break
            batch = self._pin_cpu(batch)
            slot = self._rr % self._slots
            self._enqueue(batch, slot)

    def __iter__(self) -> H2DController:
        return self

    def __next__(self) -> Any:
        if not self._queue:
            if self._err is not None:
                err = self._err
                self._err = None
                raise err
            raise StopIteration
        if len(self._queue) <= 1:
            self._starved += 1
        now = time.perf_counter()
        if self._last_yield_ts is not None:
            elapsed = now - self._last_yield_ts
            self._ema_compute = (
                0.9 * self._ema_compute + 0.1 * elapsed
                if self._ema_compute > 0
                else elapsed
            )
        moved, slot = self._queue.popleft()
        self._bytes_in_q -= min(self._bytes_in_q, self._bytes(moved))
        producer = self._streams[slot] if self._streams else None
        consumer = self._current_stream()
        if (
            producer is not None
            and consumer is not None
            and hasattr(consumer, "wait_stream")
        ):
            consumer.wait_stream(producer)
        if self._use_record_stream and consumer is not None:
            self._record_stream(moved, consumer)
        self._preload()
        self._steps += 1
        if self._autotune and self._steps % self._tune_interval == 0:
            self._retune_depth()
            self._copy_time_acc = 0.0
            self._starved = 0
        self._last_yield_ts = time.perf_counter()
        return moved

    def _record_stream(self, obj: Any, stream: Any) -> None:
        def _rec(item: Any) -> None:
            if (
                isinstance(item, torch.Tensor)
                and item.device.type == self._backend
            ):
                with suppress(Exception):
                    item.record_stream(stream)
                return
            if isinstance(item, (list, tuple)):
                for sub in item:
                    _rec(sub)
                return
            if isinstance(item, dict):
                for sub in item.values():
                    _rec(sub)

        _rec(obj)

    def _set_depth(self, new_depth: int) -> None:
        depth = int(max(self._depth_min, min(self._depth_max, new_depth)))
        if depth == self._queue.maxlen:
            return
        self._queue = deque(self._queue, maxlen=depth)

    def _retune_depth(self) -> None:
        steps = float(max(1, self._tune_interval))
        starve_ratio = float(self._starved) / steps
        avg_copy = self._copy_time_acc / steps
        avg_compute = max(1e-06, self._ema_compute)
        want_up = starve_ratio > 0.3 or avg_copy > 0.3 * avg_compute
        want_down = (
            starve_ratio < 0.05 and len(self._queue) >= self._queue.maxlen - 1
        )
        if want_up and self._queue.maxlen < self._depth_max:
            self._set_depth(self._queue.maxlen + 1)
        elif want_down and self._queue.maxlen > self._depth_min:
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

    def graphs_capture(self, capture_fn: Callable[[Any], Any]) -> bool:
        if not self._graph_enabled:
            raise RuntimeError(
                "graphs_capture requires enable_graphs=True and a CUDA backend."
            )
        if not self._queue:
            self._preload()
        moved, _ = self._queue[0]
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            capture_fn(moved)
        self._graph = graph
        return True

    def graphs_replay(self) -> None:
        if self._graph is None:
            raise RuntimeError("Call graphs_capture() before graphs_replay().")
        _ = next(self)
        self._graph.replay()


class Loader:
    def __init__(
        self,
        device: torch.device,
        *args: Any,
        node: BaseNode | None = None,
        dataset: BaseNode | None = None,
        prefetch_factor: int = 2,
        non_blocking: bool = True,
        **kwargs: Any,
    ) -> None:
        node_obj = node or dataset
        if not isinstance(node_obj, BaseNode):
            raise TypeError(
                "pipeline.collate.Loader supports only torchdata.nodes.BaseNode instances."
            )
        self._node = node_obj
        self._device = device
        self._prefetch_factor = max(1, int(prefetch_factor or 2))
        self._non_blocking = bool(non_blocking)
        base = TorchDataLoader(self._node)
        dev_t = getattr(self._device, "type", "cpu")
        if dev_t in ("cuda", "mps", "xpu") and H2DController is not None:
            try:
                self._iterable = H2DController(
                    base, device=self._device, depth=self._prefetch_factor
                )
            except TypeError:
                self._iterable = base
        else:
            self._iterable = base

    def __iter__(self) -> Any:
        return iter(self._iterable)

    def __len__(self) -> Any:
        try:
            return int(get_node_length(self._node))
        except Exception:
            return 1


def _torch_dtype_to_arrow_dtype(dtype: torch.dtype) -> str:
    mapping = {
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
    }
    return mapping.get(dtype, "float32")


def _torch_dtype_to_numpy_dtype(dtype: torch.dtype) -> Any:
    mapping = {
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
    }
    return mapping.get(dtype, np.float32)


def _map_dtype(obj: Any, dtype: Optional[torch.dtype], *args: Any, **kwargs: Any) -> Any:
    if dtype is None:
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.to(dtype=dtype, copy=False)
    if isinstance(obj, (list, tuple)):
        return type(obj)((_map_dtype(item, dtype=dtype) for item in obj))
    if isinstance(obj, dict):
        return {
            key: _map_dtype(value, dtype=dtype) for key, value in obj.items()
        }
    return obj


def flatten(objs: Iterable[Any]) -> Iterable[Any]:
    for obj in objs:
        if obj is None:
            continue
        if isinstance(obj, (list, tuple, set)):
            for item in flatten(obj):
                if item is not None:
                    yield item
            continue
        yield obj


class _Keep:
    __slots__ = ("_objs",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._objs = list(flatten(args))
        if kwargs:
            self._objs.extend(list(flatten(kwargs.values())))

    def add(self, *args: Any, **kwargs: Any) -> None:
        self._objs.extend(list(flatten(args)))
        if kwargs:
            self._objs.extend(list(flatten(kwargs.values())))

    def cleanup(self) -> None:
        for obj in self._objs:
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


def to_batch(
    batch: Mapping[str, Any],
    *args: Any,
    labels_dtype: Optional[torch.dtype] = None,
    sanitize: bool = False,
    flatten_features: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    features = batch["X"]
    labels = batch["Y"]
    if (
        flatten_features
        and isinstance(features, torch.Tensor)
        and (features.dim() >= 2)
    ):
        features = features.flatten(start_dim=1)
    labels_tensor = to_tensor(labels)
    if (
        labels_dtype is not None
        and getattr(labels_tensor, "dtype", None) != labels_dtype
    ):
        labels_tensor = labels_tensor.to(dtype=labels_dtype)
    if sanitize and torch.is_floating_point(labels_tensor):
        labels_tensor = torch.nan_to_num(
            labels_tensor, nan=0.0, posinf=1000000.0, neginf=-1000000.0
        )
    return {"X": features, "Y": labels_tensor}


def to_tensor(obj: Any) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    if hasattr(obj, "to_tensor"):
        return obj.to_tensor()
    if hasattr(obj, "as_tensor"):
        return obj.as_tensor()
    return torch.as_tensor(obj)


def fetch(
    sample: Any,
    *args: Any,
    labels_dtype: Optional[torch.dtype] = None,
    sanitize: bool = False,
    flatten_features: bool = False,
    **kwargs: Any,
) -> Any:
    if isinstance(sample, (list, tuple)):
        return [
            to_batch(
                item,
                labels_dtype=labels_dtype,
                sanitize=sanitize,
                flatten_features=flatten_features,
            )
            for item in sample
        ]
    return to_batch(
        sample,
        labels_dtype=labels_dtype,
        sanitize=sanitize,
        flatten_features=flatten_features,
    )


def loader(
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
    io_backend: str = "auto",
    **kwargs: Any,
) -> Tuple[Any, Optional[Any], _Keep]:
    device_obj = (
        torch.device(device)
        if not isinstance(device, torch.device)
        else device
    )
    backend_input = io_backend or "auto"
    if not isinstance(backend_input, str):
        raise TypeError(
            "io_backend must be a string such as 'auto', 'local', or 'flight'"
        )
    backend_normalized = backend_input.lower()
    allow_fallback = backend_normalized == "auto"
    if backend_normalized == "auto":
        backend = "flight" if has_arrow_flight() else "local"
    else:
        backend = backend_normalized
    threads = apply_threading_defaults()
    map_fn = partial(
        fetch,
        labels_dtype=labels_dtype,
        sanitize=sanitize,
        flatten_features=flatten_features,
    )

    def _wrap_node(node: IterableWrapper) -> Loader:
        wrapped: Any = ParallelMapper(
            node,
            map_fn=map_fn,
            num_workers=threads["dataloader_workers"],
            method="thread",
            in_order=False,
            max_concurrent=threads["dataloader_workers"],
            prebatch=threads["prefetch_factor"],
        )
        wrapped = Prefetcher(wrapped, prefetch_factor=prefetch_factor)
        if device_obj.type in {"cuda", "xpu", "mps"}:
            wrapped = PinMemory(wrapped, pin_memory_device=device_obj.type)
        return Loader(
            device=device_obj,
            node=wrapped,
            prefetch_factor=prefetch_factor,
            non_blocking=bool(non_blocking_copy),
            **kwargs,
        )

    def _local_impl() -> Tuple[Any, Optional[Any], _Keep]:
        reader_tr = MemoryMappedTensorStream.from_dir(
            memmap_dir,
            split="train",
            batch_size=int(batch_size),
            val_frac=val_frac,
        )
        meta = reader_tr._load_meta()
        total = int(meta.get("N", 0))
        train_range = reader_tr._indices()
        train_start = int(getattr(train_range, "start", 0))
        train_end = int(getattr(train_range, "stop", total if total else 0))
        if train_end <= train_start and total:
            train_end = total
        keep = _Keep(reader_tr)
        node_tr = IterableWrapper(
            BatchStream(reader_tr, train_start, train_end, int(batch_size))
        )
        train_loader = _wrap_node(node_tr)
        val_loader: Optional[Any] = None
        if val_frac > 0 and train_end < total:
            reader_vl = MemoryMappedTensorStream.from_dir(
                memmap_dir,
                split="val",
                batch_size=int(batch_size),
                val_frac=val_frac,
            )
            keep.add(reader_vl)
            val_range = reader_vl._indices()
            val_start = int(getattr(val_range, "start", train_end))
            val_end = int(getattr(val_range, "stop", total))
            if val_end <= val_start:
                val_end = total
            node_vl = IterableWrapper(
                BatchStream(reader_vl, val_start, val_end, int(batch_size))
            )
            val_loader = _wrap_node(node_vl)
        return (train_loader, val_loader, keep)

    def _flight_impl() -> Tuple[Any, Optional[Any], _Keep]:
        rank, local_rank, _, is_ddp = _world_info()
        name_train = Endpoint.resource_key(memmap_dir, "train")
        name_val = Endpoint.resource_key(memmap_dir, "val")
        reader_tr = MemoryMappedTensorStream.from_dir(
            memmap_dir,
            split="train",
            batch_size=int(batch_size),
            val_frac=val_frac,
        )
        reader_vl = (
            MemoryMappedTensorStream.from_dir(
                memmap_dir,
                split="val",
                batch_size=int(batch_size),
                val_frac=val_frac,
            )
            if val_frac > 0
            else None
        )
        meta = reader_tr._load_meta()
        server = None
        keep = _Keep(reader_tr, reader_vl)
        try:
            host = "0.0.0.0"
            if not is_ddp:
                host = "127.0.0.1"
            uri_value = ""
            if not is_ddp or local_rank == 0:
                server, uri_value = Endpoint.start_server_standby(
                    host=host, port=0
                )
                keep.add(server)
                Endpoint.reg_mmt_dataset(
                    server, name_train, reader_tr, int(batch_size), "train"
                )
                if reader_vl is not None:
                    Endpoint.reg_mmt_dataset(
                        server, name_val, reader_vl, int(batch_size), "val"
                    )
                name_train, uri_value = _new_flight(
                    "train", name_train, uri_value
                )
                if reader_vl is not None:
                    name_val, uri_value = _new_flight(
                        "val", name_val, uri_value
                    )
            else:
                name_train, uri_value = _new_flight("train", name_train, "")
                if reader_vl is not None:
                    name_val, uri_value = _new_flight(
                        "val", name_val, uri_value
                    )
            client = Endpoint.Client(uri_value)
            keep.add(client)
            label_shape = list(meta.get("label_shape", []))

            def _iter(name: str) -> Iterator[Dict[str, torch.Tensor]]:
                reader = client.reader(name, timeout_s=120.0)
                for record_batch in reader:
                    batch_size_rb = int(
                        getattr(record_batch, "num_rows", 0)
                    ) or int(len(record_batch.column(0)))
                    features_arr = record_batch.column(0)
                    features_np = _ARROW.to_numpy(
                        features_arr, zero_copy_only=False
                    )
                    if (
                        isinstance(features_np, np.ndarray)
                        and features_np.dtype == object
                    ):
                        features_np = np.array(
                            features_arr.to_pylist(), dtype=np.float32
                        )
                    if not isinstance(features_np, np.ndarray):
                        features_np = np.array(features_np, copy=True)
                    if (
                        getattr(features_np, "flags", None) is None
                        or not features_np.flags.writeable
                        or (not features_np.flags.c_contiguous)
                    ):
                        features_np = np.ascontiguousarray(features_np).copy()
                    if features_np.dtype != np.float32:
                        features_np = features_np.astype(
                            np.float32, copy=False
                        )
                    features_tensor = torch.from_numpy(features_np).view(
                        batch_size_rb, -1
                    )
                    labels_arr = record_batch.column(1)
                    labels_np = _ARROW.to_numpy(
                        labels_arr, zero_copy_only=False
                    )
                    if (
                        isinstance(labels_np, np.ndarray)
                        and labels_np.dtype == object
                    ):
                        labels_np = np.array(
                            labels_arr.to_pylist(), dtype=np.float32
                        )
                    if not isinstance(labels_np, np.ndarray):
                        labels_np = np.array(labels_np, copy=True)
                    if (
                        getattr(labels_np, "flags", None) is None
                        or not labels_np.flags.writeable
                        or (not labels_np.flags.c_contiguous)
                    ):
                        labels_np = np.ascontiguousarray(labels_np).copy()
                    labels_tensor = torch.from_numpy(labels_np)
                    if label_shape:
                        labels_tensor = labels_tensor.reshape(
                            batch_size_rb, -1
                        ).view(batch_size_rb, *label_shape)
                    else:
                        labels_tensor = labels_tensor.reshape(
                            batch_size_rb, -1
                        )
                    yield {"X": features_tensor, "Y": labels_tensor}

            node_tr = IterableWrapper(_iter(name_train))
            train_loader = _wrap_node(node_tr)
            if reader_vl is not None:
                node_vl = IterableWrapper(_iter(name_val))
                val_loader = _wrap_node(node_vl)
            else:
                val_loader = None
            return (train_loader, val_loader, keep)
        except Exception:
            keep.cleanup()
            raise

    if backend == "local":
        return _local_impl()
    if backend != "flight":
        raise ValueError(f"Unsupported io_backend: {backend_input!r}")

    try:
        return _flight_impl()
    except Exception as exc:
        if not allow_fallback:
            raise
        warnings.warn(
            (
                "Arrow Flight backend unavailable"
                f" ({exc}). Falling back to the local memory-mapped loader."
            ),
            RuntimeWarning,
        )
        return _local_impl()