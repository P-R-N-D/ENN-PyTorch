# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import random
import time
from collections import deque
from contextlib import nullcontext, suppress
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
from tensordict import MemoryMappedTensor

try:
    from torchdata.nodes import IterableWrapper
except Exception:
    from torchdata.datapipes.iter import IterableWrapper

from .datatype import convert


_LEGACY_DTYPE_KEYS = {
    "features_dtype": ("features_arrow_dtype",),
    "labels_dtype": ("labels_arrow_dtype",),
}


_INT_PROMOTION_TARGET = torch.int64
_FLOAT_PROMOTION_TARGET = torch.float64


def _promote_storage_dtype(tensor: torch.Tensor) -> torch.Tensor:
    """Return a tensor backed by float64/int64 storage when applicable."""

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    if tensor.is_floating_point():
        return tensor.to(dtype=_FLOAT_PROMOTION_TARGET)
    if tensor.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
    }:
        return tensor.to(dtype=_INT_PROMOTION_TARGET)
    return tensor


def _read_meta(memmap_dir: str) -> Dict[str, Any]:
    path = os.path.join(memmap_dir, "meta.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_memmap_dtype(meta: Dict[str, Any], key: str) -> torch.dtype:
    keys = (key, *_LEGACY_DTYPE_KEYS.get(key, ()))
    for candidate in keys:
        value = meta.get(candidate)
        if value is not None:
            return convert(value, "torch")
    keys_display = ", ".join(keys)
    raise ValueError(f"missing dtype metadata for {key}; looked for {keys_display}")


class SampleReader:
    def __init__(
        self,
        memmap_dir: str,
        *args: Any,
        split: str = "train",
        val_frac: Optional[float] = None,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.dir = memmap_dir
        self.split = split
        self._val_frac_override = val_frac
        self._batch_size = batch_size
        self._meta: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dir(
        cls,
        memmap_dir: str,
        *args: Any,
        split: str = "train",
        batch_size: int = 1,
        val_frac: Optional[float] = None,
        **kwargs: Any,
    ) -> SampleReader:
        return cls(
            memmap_dir,
            split=split,
            val_frac=val_frac,
            batch_size=int(batch_size),
        )

    @staticmethod
    def materialize(
        data: Dict[str, Any],
        *args: Any,
        memmap_dir: str,
        train_frac: float = 1.0,
        val_frac: float = 0.0,
        shuffle: bool = False,
        **kwargs: Any,
    ) -> None:
        os.makedirs(memmap_dir, exist_ok=True)
        features = _promote_storage_dtype(torch.as_tensor(data["features"]).detach())
        labels = _promote_storage_dtype(torch.as_tensor(data["labels"]).detach())
        features = features.cpu().contiguous()
        labels = labels.cpu().contiguous()
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features/labels N mismatch")
        count = int(features.shape[0])
        feat_dim = int(features.view(count, -1).shape[1])
        label_shape: List[int] = list(labels.shape[1:])
        label_flat = int(labels.numel() // count)
        if shuffle:
            perm = torch.randperm(count)
            features = features.index_select(0, perm)
            labels = labels.index_select(0, perm)
        feat_path = os.path.join(memmap_dir, "features.mmt")
        label_path = os.path.join(memmap_dir, "labels.mmt")
        MemoryMappedTensor.from_tensor(
            features.view(count, feat_dim), filename=feat_path, existsok=True
        )
        MemoryMappedTensor.from_tensor(
            labels.view(count, label_flat), filename=label_path, existsok=True
        )
        meta = {
            "N": count,
            "feature_dim": feat_dim,
            "label_shape": label_shape,
            "features_dtype": convert(features.dtype, "name"),
            "labels_dtype": convert(labels.dtype, "name"),
            "fractions": [float(train_frac), float(val_frac)],
            "features_filename": "features.mmt",
            "labels_filename": "labels.mmt",
        }
        with open(
            os.path.join(memmap_dir, "meta.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(meta, handle)

    def _load_meta(self) -> Dict[str, Any]:
        if self._meta is None:
            with open(
                os.path.join(self.dir, "meta.json"), "r", encoding="utf-8"
            ) as handle:
                self._meta = json.load(handle)
        return self._meta

    def _indices(self) -> range:
        meta = self._load_meta()
        total = int(meta["N"])
        val_fraction = float(
            self._val_frac_override
            if self._val_frac_override is not None
            else meta.get("fractions", [1.0, 0.0])[-1]
        )
        val_count = int(round(total * val_fraction))
        train_count = total - val_count
        match self.split:
            case "train":
                return range(0, train_count)
            case "val":
                return range(train_count, total)
            case _:
                return range(0, total)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        meta = self._load_meta()
        total = int(meta["N"])
        feat_dim = int(meta["feature_dim"])
        label_shape = list(meta["label_shape"])
        label_flat = (
            int(torch.tensor(label_shape).prod().item()) if label_shape else 1
        )
        feat_path = os.path.join(
            self.dir, meta.get("features_filename", "features.mmt")
        )
        label_path = os.path.join(
            self.dir, meta.get("labels_filename", "labels.mmt")
        )
        feat_dtype = _resolve_memmap_dtype(meta, "features_dtype")
        label_dtype = _resolve_memmap_dtype(meta, "labels_dtype")
        feat_mmt = MemoryMappedTensor.from_filename(
            feat_path, dtype=feat_dtype, shape=(total, feat_dim)
        )
        label_mmt = MemoryMappedTensor.from_filename(
            label_path, dtype=label_dtype, shape=(total, label_flat)
        )
        for index in self._indices():
            feat = feat_mmt[index]
            label = label_mmt[index]
            if label_shape:
                label = label.view(*label_shape)
            feat_tensor = (
                feat
                if isinstance(feat, torch.Tensor)
                else torch.as_tensor(feat)
            )
            label_tensor = (
                label
                if isinstance(label, torch.Tensor)
                else torch.as_tensor(label)
            )
            yield (feat_tensor, label_tensor)

    def __len__(self) -> int:
        return len(self._indices())

    def batch_range(
        self, start: int, end: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        meta = self._load_meta()
        total = int(meta["N"])
        feat_dim = int(meta["feature_dim"])
        label_shape = list(meta["label_shape"])
        label_flat = (
            int(torch.tensor(label_shape).prod().item()) if label_shape else 1
        )
        feat_path = os.path.join(
            self.dir, meta.get("features_filename", "features.mmt")
        )
        label_path = os.path.join(
            self.dir, meta.get("labels_filename", "labels.mmt")
        )
        feat_dtype = _resolve_memmap_dtype(meta, "features_dtype")
        label_dtype = _resolve_memmap_dtype(meta, "labels_dtype")
        feat_mmt = MemoryMappedTensor.from_filename(
            feat_path, dtype=feat_dtype, shape=(total, feat_dim)
        )
        label_mmt = MemoryMappedTensor.from_filename(
            label_path, dtype=label_dtype, shape=(total, label_flat)
        )
        features = feat_mmt[start:end]
        labels = label_mmt[start:end].view(-1, *label_shape)
        features_tensor = (
            features
            if isinstance(features, torch.Tensor)
            else torch.as_tensor(features)
        )
        labels_tensor = (
            labels
            if isinstance(labels, torch.Tensor)
            else torch.as_tensor(labels)
        )
        return (features_tensor, labels_tensor)

class BatchReader:
    def __init__(
        self,
        mmts: SampleReader,
        start: int,
        end: int,
        batch_size: int,
    ) -> None:
        self._mmts = mmts
        self._start = int(start)
        self._end = int(end)
        self._batch = max(1, int(batch_size))

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        index = int(self._start)
        while index < self._end:
            nxt = min(index + self._batch, self._end)
            xb, yb = self._mmts.batch_range(index, nxt)
            yield {"X": xb, "Y": yb}
            index = nxt

    def __len__(self) -> int:
        if self._end <= self._start:
            return 0
        span = self._end - self._start
        return int(math.ceil(span / float(self._batch)))


class BatchSampler(IterableWrapper):
    def __init__(
        self,
        memmap_dir: str,
        part: str,
        batch_size: int,
        shuffle: bool,
        seed: int,
        *args: Any,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
        fractions: Optional[Tuple[float, float]] = None,
        **kwargs: Any,
    ) -> None:
        meta = _read_meta(memmap_dir)
        total = int(meta["N"])
        if fractions is not None:
            train_frac = float(fractions[0])
        else:
            train_frac = 1.0
            if "fractions" in meta:
                try:
                    train_frac = float(meta["fractions"][0])
                except Exception:
                    train_frac = 1.0
        if part not in {"train", "val"}:
            raise ValueError("part must be 'train' or 'val'")
        train_count = int(math.floor(total * train_frac))
        start, end = (
            (0, train_count) if part == "train" else (train_count, total)
        )
        indices = list(range(start, end))
        if shuffle:
            rng = random.Random(int(seed))
            rng.shuffle(indices)
        if world_size > 1:
            indices = indices[int(rank) :: int(world_size)]
        batch_len = int(batch_size)
        batches: List[List[int]] = []
        current: List[int] = []
        for idx in indices:
            current.append(int(idx))
            if len(current) == batch_len:
                batches.append(current)
                current = []
        if current and (not drop_last):
            batches.append(current)

        super().__init__([list(chunk) for chunk in batches])


class DevicePrefetcher(Iterator[Any]):
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

    def _pin_tensor(self, tensor: Any) -> Any:
        if self._should_pin(tensor):
            with suppress(Exception):
                return tensor.pin_memory()
        return tensor

    def _pin_cpu(self, obj: Any) -> Any:
        return self._map(obj, self._pin_tensor)

    def _move_tensor(self, tensor: Any) -> Any:
        if isinstance(tensor, torch.Tensor):
            kwargs: Dict[str, Any] = {
                "non_blocking": self._backend in {"cuda", "xpu", "mps"}
            }
            if self._amp_dtype is not None:
                kwargs["dtype"] = self._amp_dtype
            return tensor.to(self._dev, **kwargs)
        return tensor

    def _to_device(self, obj: Any) -> Any:
        return self._map(obj, self._move_tensor)

    def _allocate_tensor_like(self, tensor: Any) -> Any:
        if isinstance(tensor, torch.Tensor):
            dtype = self._amp_dtype or tensor.dtype
            return torch.empty(tensor.shape, device=self._dev, dtype=dtype)
        return tensor

    def _allocate_static_like(self, obj: Any) -> Any:
        return self._map(obj, self._allocate_tensor_like)

    def _copy_structure(self, dst_obj: Any, src_obj: Any) -> Any:
        if isinstance(dst_obj, torch.Tensor) and isinstance(src_obj, torch.Tensor):
            dtype = self._amp_dtype or src_obj.dtype
            src_tensor = src_obj.to(dtype=dtype) if dst_obj.dtype != dtype else src_obj
            dst_obj.copy_(
                src_tensor,
                non_blocking=self._backend in {"cuda", "xpu", "mps"},
            )
            return dst_obj
        if isinstance(dst_obj, (list, tuple)) and isinstance(src_obj, (list, tuple)):
            return type(dst_obj)(
                (self._copy_structure(d, s) for d, s in zip(dst_obj, src_obj))
            )
        if isinstance(dst_obj, dict) and isinstance(src_obj, dict):
            return {
                key: self._copy_structure(dst_obj[key], src_obj[key])
                for key in dst_obj.keys() & src_obj.keys()
            }
        return dst_obj

    def _copy_into(self, dst: Any, src: Any) -> Any:
        return self._copy_structure(dst, src)

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

    def __iter__(self) -> "DevicePrefetcher":
        return self

    def _record_item_stream(self, item: Any, stream: Any) -> None:
        if isinstance(item, torch.Tensor) and item.device.type == self._backend:
            with suppress(Exception):
                item.record_stream(stream)
            return
        if isinstance(item, (list, tuple)):
            for sub in item:
                self._record_item_stream(sub, stream)
            return
        if isinstance(item, dict):
            for sub in item.values():
                self._record_item_stream(sub, stream)

    def _record_stream(self, obj: Any, stream: Any) -> None:
        self._record_item_stream(obj, stream)

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
