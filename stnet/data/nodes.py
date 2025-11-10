# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import queue
import random
import threading
from contextlib import suppress
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple, TypedDict, Literal

import torch

try:
    from torchdata.nodes import BaseNode
except Exception:
    BaseNode = object

try:
    from tensordict import TensorDict, load_memmap
    from tensordict import memmap as td_memmap
except Exception as e:
    raise RuntimeError("tensordict is required for Dataset") from e

try:
    from torchdata.nodes import (
        Loader as _Loader,
        MapStyleWrapper,
        MultiNodeWeightedSampler,
        ParallelMapper,
        PinMemory,
        Prefetcher as _Prefetcher,
    )
except Exception:
    from torchdata.nodes import Loader as _Loader, ParallelMapper, PinMemory, Prefetcher as _Prefetcher
    MultiNodeWeightedSampler = None
    MapStyleWrapper = None

try:
    from torch.utils.data import Sampler as _Sampler
except Exception:
    _Sampler = object

from .datatype import to_platform_dtype


def _to_device(batch: Any, device: torch.device, non_blocking: bool = True) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, Mapping):
        return {k: _to_device(v, device, non_blocking) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        seq = [_to_device(v, device, non_blocking) for v in batch]
        return type(batch)(seq) if isinstance(batch, tuple) else seq
    return batch


class Dataset:
    def __init__(self, memmap_dir: str, *args: Any, split: str = "train", val_frac: float = 0.0, **kwargs: Any) -> None:
        self.dir = os.fspath(memmap_dir)
        self.split = str(split)
        self._meta: Optional[Mapping[str, Any]] = None

        meta_path = os.path.join(self.dir, "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"meta.json not found under: {self.dir}")
        with open(meta_path, "r", encoding="utf-8") as f:
            self._meta = json.load(f)
        self._N = int(self._meta.get("N", 0))
        td_prefix = os.path.join(self.dir, self._meta.get("tensordict_prefix", "td_memmap"))
        nb = bool(int(os.environ.get("STNET_TD_NONBLOCKING_LOAD", "0")))
        td = load_memmap(td_prefix, non_blocking=nb)
        self._features = td.get("features")
        self._labels = td.get("labels")
        lshape = list(self._meta.get("label_shape") or [])
        self._label_shape: Tuple[int, ...] = tuple(lshape) if lshape else tuple()
        train_start = int(self._meta.get("train_start", 0))
        train_end   = int(self._meta.get("train_end",   self._N))
        val_start   = int(self._meta.get("val_start",   0))
        val_end     = int(self._meta.get("val_end",     0))

        if val_frac and not (val_end > val_start):
            vf = float(val_frac)
            vc = max(0, min(self._N, int(self._N * vf)))
            val_start, val_end = max(0, self._N - vc), self._N
            train_start, train_end = 0, val_start

        if self.split == "val":
            self._start, self._end = (val_start, val_end) if val_end > val_start else (0, 0)
        else:
            self._start, self._end = (train_start, train_end)

    @property
    def start(self) -> int:
        return int(self._start)

    @property
    def end(self) -> int:
        return int(self._end)

    @property
    def meta(self) -> Mapping[str, Any]:
        return dict(self._meta or {})

    def __len__(self) -> int:
        return max(0, int(self._end) - int(self._start))

    def _slice(self, start: int, end: int) -> Mapping[str, torch.Tensor]:
        x = self._features[start:end]
        y = self._labels[start:end]
        if self._label_shape:
            y = y.view(end - start, *self._label_shape)
        xt = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        yt = y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
        return {"X": xt, "Y": yt}

    def __getitem__(self, idx: int | Tuple[int, int]) -> Mapping[str, torch.Tensor]:
        if isinstance(idx, tuple) and len(idx) == 2:
            s, e = int(idx[0]), int(idx[1])
            return self._slice(s, e)
        i = self._start + int(idx)
        return self._slice(i, i + 1)

# ---- Memmap utilities --------------------------------------------------------


def _to_high_precision(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.is_floating_point():
        return tensor.to(dtype=torch.float64)
    if tensor.dtype in {torch.uint8, torch.int8, torch.int16, torch.int32}:
        return tensor.to(dtype=torch.int64)
    return tensor


def preload_memmap(
    data: Mapping[str, Any],
    *,
    memmap_dir: str,
    train_frac: float = 1.0,
    val_frac: float = 0.0,
    shuffle: bool = False,
    seed: int | None = None,
    **kwargs: Any,
) -> None:
    if "features" not in data or "labels" not in data:
        raise ValueError("preload_memmap expects 'features' and 'labels'")

    os.makedirs(memmap_dir, exist_ok=True)

    features = _to_high_precision(data["features"]).detach().cpu().contiguous()
    labels = _to_high_precision(data["labels"]).detach().cpu().contiguous()

    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same length")

    count = int(features.shape[0])
    if count == 0:
        raise ValueError("cannot create memmap with zero samples")

    feature_flat = features.view(count, -1)
    label_flat = labels.view(count, -1)
    label_shape = tuple(labels.shape[1:])

    perm: torch.Tensor | None = None
    if shuffle:
        generator = torch.Generator(device="cpu")
        if seed is not None:
            with suppress(Exception):
                generator.manual_seed(int(seed))
        perm = torch.randperm(count, generator=generator)
        feature_flat = feature_flat.index_select(0, perm)
        label_flat = label_flat.index_select(0, perm)
        if seed is not None:
            with suppress(Exception):
                torch.save(perm, os.path.join(memmap_dir, "perm.pt"))

    td = TensorDict(
        {"features": feature_flat, "labels": label_flat},
        batch_size=[count],
        device=torch.device("cpu"),
    )
    td_memmap(td, prefix=os.path.join(memmap_dir, "td_memmap"))

    val_count = max(0, min(count, int(round(count * float(val_frac)))))
    train_count = max(0, min(count, count - val_count))
    train_start, train_end = 0, train_count
    val_start, val_end = train_end, train_end + val_count

    meta: Dict[str, Any] = {
        "N": count,
        "feature_dim": int(feature_flat.shape[1]),
        "label_shape": list(label_shape),
        "features_dtype": to_platform_dtype(feature_flat.dtype, "name"),
        "labels_dtype": to_platform_dtype(label_flat.dtype, "name"),
        "fractions": [float(train_frac), float(val_frac)],
        "shuffled": bool(shuffle),
        "shuffle_seed": int(seed) if seed is not None else None,
        "tensordict_prefix": "td_memmap",
        "train_start": train_start,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
    }

    if perm is not None and seed is not None:
        meta["perm_filename"] = "perm.pt"

    for key in ("target_scaler", "robust_q", "robust_cap", "scale_non_floating"):
        if key in kwargs and kwargs[key] is not None:
            value = kwargs[key]
            meta[key] = list(value) if isinstance(value, tuple) else value

    with open(os.path.join(memmap_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle)


# ---- Source abstraction (no legacy path strings) -----------------------------
SourceKind = Literal["memmap"]

class SourceSpec(TypedDict):
    """Structured source spec (mandatory).
    kind: currently only "memmap"
    path: directory that contains meta.json and memmap shards
    """
    kind: SourceKind
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
            for name in ("cleanup", "close", "shutdown", "stop", "terminate", "join", "disconnect", "release"):
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

    def __iter__(self):
        return iter(self._keep)


class Sampler(_Sampler):
    def __init__(
        self,
        *args: Any,
        start: int,
        end: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        self._start = int(start)
        self._end = int(end)
        self._B = max(1, int(batch_size))
        self._shuffle = bool(shuffle)
        self._rng = random.Random(int(seed))
        self._cuts = list(range(self._start, self._end, self._B))
        if self._end > self._start and (not self._cuts or self._cuts[-1] != self._end):
            self._cuts.append(self._end)

    def __iter__(self):
        n = max(0, len(self._cuts) - 1)
        idxs = list(range(n))
        if self._shuffle:
            self._rng.shuffle(idxs)
        for i in idxs:
            s = self._cuts[i]
            e = self._cuts[i + 1]
            if e > s:
                yield (s, e)

    def __len__(self) -> int:
        return max(0, (self._end - self._start + self._B - 1) // self._B)

    def compose(self, dataset: "Dataset") -> "BaseNode":
        if MapStyleWrapper is None:
            raise RuntimeError("torchdata.nodes.MapStyleWrapper is required")
        return MapStyleWrapper(dataset, self)


class Multiplexer:
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
        return MultiNodeWeightedSampler(
            sources_map, w, stop_criteria=self.stop_criteria
        )


class Connector:
    def __init__(
        self,
        *args: Any,
        map_fn: Callable[[Any], Any],
        io_workers: int,
        prebatch: int,
        prefetch_factor: int,
        device: torch.device,
        non_blocking: bool = True,
        **kwargs: Any,
    ) -> None:
        self.map_fn = map_fn
        self.io_workers = max(1, int(io_workers))
        self.prebatch = max(1, int(prebatch))
        self.prefetch_factor = max(1, int(prefetch_factor))
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.non_blocking = bool(non_blocking)

    def compose(self, source: "BaseNode") -> "BaseNode":
        node = ParallelMapper(
            source,
            map_fn=self.map_fn,
            num_workers=self.io_workers,
            in_order=False,
            method="thread",
            max_concurrent=None,
            prebatch=self.prebatch,
        )
        node = _Prefetcher(node, prefetch_factor=self.prefetch_factor)
        if self.device.type in {"cuda", "xpu", "mps"}:
            node = PinMemory(node, pin_memory_device=self.device.type)
        return node


class Loader:
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
        base = _Loader(node_obj)
        dev_t = getattr(self._device, "type", "cpu")
        if dev_t in {"cuda", "mps", "xpu"} and self._non_blocking:
            try:
                gpu_guard_default = "2048" if dev_t == "cuda" else "512"
                gpu_guard_mb = int(os.environ.get("STNET_GPU_GUARD_MB", gpu_guard_default))
            except Exception:
                gpu_guard_mb = 2048 if dev_t == "cuda" else 512
            try:
                host_guard_mb = int(os.environ.get("STNET_HOST_GUARD_MB", "1024"))
            except Exception:
                host_guard_mb = 1024
            self._iterable = Prefetcher(
                base,
                device=self._device,
                depth=self._prefetch_factor,
                non_blocking=True,
                memory_backpressure=True,
                gpu_guard_bytes=gpu_guard_mb * (1 << 20),
                host_guard_bytes=host_guard_mb * (1 << 20),
            )
        else:
            self._iterable = base

    def __iter__(self):
        return iter(self._iterable)

    def __len__(self) -> int:
        if self._length is not None:
            return int(self._length)
        try:
            return 1 if self._length is None else int(self._length)
        except Exception:
            return 1


class Prefetcher:

    def __init__(
        self,
        iterable: Any,
        *args: Any,
        device: torch.device | str,
        depth: int = 2,
        non_blocking: bool = True,
        oom_safe: bool = True,
        gpu_guard_bytes: int | None = None,
        host_guard_bytes: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._src = iterable
        self._device = torch.device(device) if not isinstance(device, torch.device) else device
        self._depth = max(1, int(depth))
        self._non_blocking = bool(non_blocking)
        self._backpressure = bool(oom_safe)
        self._gpu_guard_bytes = int(gpu_guard_bytes or 0)
        self._host_guard_bytes = int(host_guard_bytes or 0)

    def __iter__(self) -> Iterator[Any]:
        it = iter(self._src)
        q: "queue.Queue[Optional[Any]]" = queue.Queue(maxsize=self._depth)
        sentinel = object()

        def _producer():
            try:
                for item in it:
                    moved = _to_device(item, self._device, non_blocking=self._non_blocking)
                    q.put(moved, block=True)
            except StopIteration:
                pass
            finally:
                q.put(sentinel, block=True)

        th = threading.Thread(target=_producer, daemon=True)
        th.start()

        while True:
            item = q.get(block=True)
            if item is sentinel:
                break
            yield item
        th.join(timeout=0.1)
