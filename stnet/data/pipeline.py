# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import random
from functools import partial
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
from tensordict import TensorDict, TensorDictBase, stack

try:
    from torchdata.nodes import (
        BaseNode,
        Loader as _Loader,
        MapStyleWrapper,
        MultiNodeWeightedSampler,
        ParallelMapper,
        PinMemory,
        Prefetcher as _Prefetcher,
    )
except Exception:
    from torchdata.nodes import BaseNode, Loader as _Loader, ParallelMapper, PinMemory, Prefetcher as _Prefetcher
    MultiNodeWeightedSampler = None
    MapStyleWrapper = None

import torch.utils.data.Sampler as _Sampler

from .nodes import Dataset, Prefetcher

def _process(
    batch: Mapping[str, Any],
    *args: Any,
    flatten_features: bool,
    labels_dtype: Optional[torch.dtype],
    sanitize: bool,
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
    if labels_dtype is not None and isinstance(labels, torch.Tensor):
        labels = labels.to(dtype=labels_dtype, non_blocking=True)
    if sanitize and torch.is_floating_point(labels):
        labels = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
    return {"X": features, "Y": labels}


def collate(
    sample: Any,
    *args: Any,
    labels_dtype: Optional[torch.dtype] = None,
    sanitize: bool = False,
    flatten_features: bool = False,
    **kwargs: Any,
) -> Any:
    converter = partial(
        _process,
        flatten_features=flatten_features,
        labels_dtype=labels_dtype,
        sanitize=sanitize,
    )
    if isinstance(sample, TensorDictBase):
        return sample
    if isinstance(sample, (list, tuple)):
        batch_list = []
        for item in sample:
            if isinstance(item, TensorDictBase):
                batch_list.append(item); continue
            if isinstance(item, Mapping):
                conv = converter(item)
                td = TensorDict(
                    {"X": conv["X"], "Y": conv["Y"], "features": conv["X"], "labels": conv["Y"]},
                    batch_size=[],
                )
                batch_list.append(td)
            else:
                batch_list.append(item)
        if all(isinstance(elem, TensorDictBase) for elem in batch_list):
            return stack(batch_list, dim=0)
        return batch_list
    if isinstance(sample, Mapping):
        conv = converter(sample)
        return TensorDict({"X": conv["X"], "Y": conv["Y"], "features": conv["X"], "labels": conv["Y"]}, batch_size=[])
    return sample

class Disposable:
    def __init__(self) -> None:
        self._keep: list[Any] = []

    def add(self, obj: Any) -> None:
        self._keep.append(obj)

    def __iter__(self):
        return iter(self._keep)

class Sampler(_Sampler[Tuple[int, int]]):
    def __init__(self, *args: Any, start: int, end: int, batch_size: int, shuffle: bool = True, seed: int = 0, **kwargs: Any) -> None:
        self._start = int(start); self._end = int(end); self._B = max(1, int(batch_size))
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
            s = self._cuts[i]; e = self._cuts[i+1]
            if e > s:
                yield (s, e)

    def __len__(self) -> int:
        return max(0, (self._end - self._start + self._B - 1) // self._B)

    def compose(self, dataset: "Dataset") -> "BaseNode":
        if MapStyleWrapper is None:
            raise RuntimeError("torchdata.nodes.MapStyleWrapper is required")
        return MapStyleWrapper(dataset, self)


class Multiplexer:

    def __init__(self, *args: Any, stop_criteria: str = "ALL_DATASETS_EXHAUSTED", weights: Optional[Mapping[str, float]] = None, seed: int = 0, **kwargs: Any) -> None:
        self.stop_criteria = str(stop_criteria)
        self.weights = dict(weights) if isinstance(weights, Mapping) else None
        self.seed = int(seed)

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
        return MultiNodeWeightedSampler(sources_map, w, stop_criteria=self.stop_criteria)


class Fetcher:
    def __init__(self, *args: Any, map_fn: Callable[[Any], Any], io_workers: int, prebatch: int, prefetch_factor: int, device: torch.device, non_blocking: bool = True, **kwargs: Any) -> None:
        self.map_fn = map_fn
        self.io_workers = max(1, int(io_workers))
        self.prebatch = max(1, int(prebatch))
        self.prefetch_factor = max(1, int(prefetch_factor))
        self.device = device if isinstance(device, torch.device) else torch.device(device)
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
        self._device = device if isinstance(device, torch.device) else torch.device(device)
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


def _env_optimize_threads() -> Dict[str, int]:
    try:
        workers = max(1, int(os.environ.get("STNET_WORKERS", "4")))
    except Exception:
        workers = 4
    try:
        pfetch = max(1, int(os.environ.get("STNET_PREFETCH", "8")))
    except Exception:
        pfetch = 8
    return {"dataloader_workers": workers, "prefetch_factor": pfetch}


def compose(
    node_or_nodes: Union[BaseNode, Sequence[BaseNode], Mapping[str, BaseNode]],
    *args: Any,
    device: Union[str, torch.device],
    map_fn: Callable[[Any], Any],
    prefetch_factor: int,
    non_blocking_copy: bool,
    io_workers: int,
    prebatch: int,
    **kwargs: Any,
) -> Tuple[BaseNode, BaseNode, BaseNode]:
    device_obj = torch.device(device) if not isinstance(device, torch.device) else device

    mux = Multiplexer(
        stop_criteria=os.environ.get("STNET_MULTINODE_STOP", "ALL_DATASETS_EXHAUSTED"),
        seed=int(os.environ.get("STNET_BLOCK_SEED", "0") or "0"),
    )
    source = mux.compose(node_or_nodes)

    mapper = Fetcher(
        map_fn=map_fn,
        io_workers=io_workers,
        prebatch=prebatch,
        prefetch_factor=prefetch_factor,
        device=device_obj,
        non_blocking=bool(non_blocking_copy),
    )
    mapped = mapper.compose(source)
    return source, mapped, mapped


def fetch(
    memmap_dir: Union[str, Sequence[str], Mapping[str, str]],
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
) -> Tuple[Any, Optional[Any], Disposable]:
    device_obj = torch.device(device) if not isinstance(device, torch.device) else device
    threads = _env_optimize_threads()
    io_workers = max(1, int(threads.get("dataloader_workers", 2)))
    prebatch = max(1, int(threads.get("prefetch_factor", 2)))

    map_fn = partial(
        collate,
        labels_dtype=labels_dtype,
        sanitize=sanitize,
        flatten_features=flatten_features,
    )

    allocated = Disposable()

    def _node_for(directory: Union[str, os.PathLike[str]], split: str, shuffle: bool) -> Tuple[BaseNode, int]:
        path = os.fspath(directory)
        ds = Dataset(path, split=split, val_frac=float(val_frac))
        allocated.add(ds)
        samp = Sampler(start=ds.start, end=ds.end, batch_size=int(batch_size), shuffle=shuffle, seed=int(os.environ.get("STNET_SAMPLER_SEED","0") or "0"))
        node = samp.compose(ds)
        return node, len(samp)

    if isinstance(memmap_dir, Mapping):
        nodes_map: Dict[str, BaseNode] = {}
        lengths: Dict[str, int] = {}
        for key, directory in memmap_dir.items():
            node, length = _node_for(directory, split="train", shuffle=True)
            nodes_map[str(key)] = node
            lengths[str(key)] = length
        _, mapped, _ = compose(nodes_map, device=device_obj, map_fn=map_fn, prefetch_factor=int(prefetch_factor), non_blocking_copy=bool(non_blocking_copy), io_workers=io_workers, prebatch=prebatch)
        train_length = sum(lengths.values()) if lengths else None
    elif isinstance(memmap_dir, (list, tuple)):
        nodes_list: list[BaseNode] = []
        lengths: list[int] = []
        for directory in memmap_dir:
            node, length = _node_for(directory, split="train", shuffle=True)
            nodes_list.append(node); lengths.append(length)
        _, mapped, _ = compose(nodes_list, device=device_obj, map_fn=map_fn, prefetch_factor=int(prefetch_factor), non_blocking_copy=bool(non_blocking_copy), io_workers=io_workers, prebatch=prebatch)
        train_length = sum(lengths) if lengths else None
    else:
        node, length = _node_for(memmap_dir, split="train", shuffle=True)
        _, mapped, _ = compose(node, device=device_obj, map_fn=map_fn, prefetch_factor=int(prefetch_factor), non_blocking_copy=bool(non_blocking_copy), io_workers=io_workers, prebatch=prebatch)
        train_length = length

    train_loader = _Loader(device=device_obj, node=mapped, prefetch_factor=int(prefetch_factor), non_blocking=bool(non_blocking_copy), length=train_length)

    val_loader = None
    if float(val_frac) > 0.0:
        if isinstance(memmap_dir, Mapping):
            nodes_map = {}
            lengths = {}
            for key, directory in memmap_dir.items():
                node, length = _node_for(directory, split="val", shuffle=False)
                nodes_map[str(key)] = node; lengths[str(key)] = length
            _, vmapped, _ = compose(nodes_map, device=device_obj, map_fn=map_fn, prefetch_factor=int(prefetch_factor), non_blocking_copy=bool(non_blocking_copy), io_workers=io_workers, prebatch=prebatch)
            val_len = sum(lengths.values()) if lengths else None
        elif isinstance(memmap_dir, (list, tuple)):
            nodes_list = []
            lengths = []
            for directory in memmap_dir:
                node, length = _node_for(directory, split="val", shuffle=False)
                nodes_list.append(node); lengths.append(length)
            _, vmapped, _ = compose(nodes_list, device=device_obj, map_fn=map_fn, prefetch_factor=int(prefetch_factor), non_blocking_copy=bool(non_blocking_copy), io_workers=io_workers, prebatch=prebatch)
            val_len = sum(lengths) if lengths else None
        else:
            node, length = _node_for(memmap_dir, split="val", shuffle=False)
            _, vmapped, _ = compose(node, device=device_obj, map_fn=map_fn, prefetch_factor=int(prefetch_factor), non_blocking_copy=bool(non_blocking_copy), io_workers=io_workers, prebatch=prebatch)
            val_len = length
        val_loader = _Loader(device=device_obj, node=vmapped, prefetch_factor=int(prefetch_factor), non_blocking=bool(non_blocking_copy), length=val_len)

    return (train_loader, val_loader, allocated)
