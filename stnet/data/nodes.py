# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import random
import threading
from contextlib import suppress
from typing import (Any, Callable, Dict, Iterator, Literal, Mapping, Optional,
                    Sequence, Tuple, TypedDict)

import torch

TensorLike = Any

try:
    from torchdata.nodes import BaseNode
    from torchdata.nodes import Loader as _Loader
    from torchdata.nodes import ParallelMapper
except Exception:
    from torchdata.nodes import BaseNode
    from torchdata.nodes import Loader as _Loader

    ParallelMapper = None

from tensordict import MemoryMappedTensor

try:
    from torchdata.nodes import Batcher as _Batcher
    from torchdata.nodes import MultiNodeWeightedSampler, PinMemory
    from torchdata.nodes import Prefetcher as _Prefetcher
    from torchdata.nodes import SamplerWrapper
    from torchdata.nodes import Unbatcher as _Unbatcher
except Exception:
    from torchdata.nodes import PinMemory
    from torchdata.nodes import Prefetcher as _Prefetcher

    MultiNodeWeightedSampler = None
    SamplerWrapper = None
    _Batcher = None
    _Unbatcher = None

try:
    from torch.utils.data import Sampler as _Sampler
except Exception:
    _Sampler = object

from .datatype import to_platform_dtype


def _to_device(batch: TensorLike, device: torch.device, non_blocking: bool = True) -> TensorLike:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, Mapping):
        return {k: _to_device(v, device, non_blocking) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        seq = [_to_device(v, device, non_blocking) for v in batch]
        return type(batch)(seq) if isinstance(batch, tuple) else seq
    return batch


class Dataset(_Sampler):
    
    _scale: float = 1.0
    _per_sample_mem_bytes: int = 0

    @staticmethod
    def _dtype_from_name(name: Any, default: torch.dtype) -> torch.dtype:
        try:
            return getattr(torch, str(name))
        except Exception:
            return default

    @classmethod
    def _load_meta(cls: type["Dataset"], memmap_dir: str) -> Mapping[str, Any]:
        meta_path = os.path.join(os.fspath(memmap_dir), "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"meta.json not found under: {memmap_dir}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, Mapping):
            raise ValueError(f"meta.json under {memmap_dir} must contain a mapping")
        return meta

    @classmethod
    def request_scale_up(cls: type["Dataset"], factor: float) -> None:
        cls._scale = min(2.0, cls._scale * float(factor))

    @classmethod
    def request_scale_down(cls: type["Dataset"], factor: float) -> None:
        cls._scale = max(0.5, cls._scale * float(factor))

    def __init__(
        self,
        memmap_dir: str,
        *args: Any,
        split: str = "train",
        val_frac: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.dir = os.fspath(memmap_dir)
        self.split = str(split)
        self._meta: Mapping[str, Any] = self._load_meta(self.dir)
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
        self._features = MemoryMappedTensor.from_filename(
            filename=feat_path, dtype=f_dtype, shape=torch.Size([self._N, fdim])
        )
        lshape = tuple(lshape_meta) if lshape_meta else tuple()
        self._labels = MemoryMappedTensor.from_filename(
            filename=lab_path, dtype=l_dtype, shape=torch.Size([self._N] + list(lshape))
        )
        self._memmap_features = self._features
        self._memmap_labels = self._labels
        self._X = self._memmap_features
        self._Y = self._memmap_labels
        self._S_B = 1
        self._S_shuffle = True
        self._S_seed = 0
        self._S_rng = random.Random(0)
        self._S_cuts: list[int] = []
        self._num_shards = 1
        self._shard_id = 0
        self._key = ""
        self._label_shape: Tuple[int, ...] = tuple(lshape) if lshape else tuple()
        self._perm: Optional[torch.Tensor] = None
        self._perm_source: Optional[Literal["runtime", "metadata"]] = None
        perm_fn = (self._meta or {}).get("perm_filename", None)
        if perm_fn:
            perm_path = os.path.join(self.dir, str(perm_fn))
            if os.path.isfile(perm_path):
                with suppress(Exception):
                    self._perm = torch.load(perm_path, map_location="cpu")
                    meta_shuffled = bool((self._meta or {}).get("shuffled", False))
                    self._perm_source = "runtime" if not meta_shuffled else "metadata"
        if self._perm is None and False:
            gen = torch.Generator(device="cpu")
            with suppress(Exception):
                gen.manual_seed(0)
            with suppress(Exception):
                self._perm = torch.randperm(self._N, generator=gen)
                self._perm_source = "runtime"
        if self._perm is not None:
            try:
                if int(self._perm.numel()) != self._N:
                    with suppress(Exception):
                        import warnings as _warn

                        _warn.warn(
                            f"[stnet] ignoring invalid perm: length={int(self._perm.numel())}, expected N={self._N}"
                        )
                    self._perm = None
                    self._perm_source = None
                else:
                    if self._perm.dtype != torch.long:
                        with suppress(Exception):
                            self._perm = self._perm.to(dtype=torch.long)
                    if getattr(self._perm, "device", torch.device("cpu")).type != "cpu":
                        with suppress(Exception):
                            self._perm = self._perm.cpu()
            except Exception:
                self._perm = None
                self._perm_source = None
        if self._perm is None:
            self._perm_source = None
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
        if self._perm_source == "runtime" and getattr(self, "_perm", None) is not None:
            idx = self._perm[start:end]
            if idx.numel() == 0:
                x = self._features[:0]
                y = self._labels[:0]
            else:
                try:
                    x = self._features.index_select(0, idx)
                except Exception:
                    x = (
                        self._features[idx]
                        if hasattr(self._features, "__getitem__")
                        else torch.as_tensor(self._features)[idx]
                    )
                try:
                    y = self._labels.index_select(0, idx)
                except Exception:
                    y = (
                        self._labels[idx]
                        if hasattr(self._labels, "__getitem__")
                        else torch.as_tensor(self._labels)[idx]
                    )
        else:
            x = self._features[start:end]
            y = self._labels[start:end]
        if self._label_shape:
            y = y.reshape(end - start, *self._label_shape)
        xt = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        yt = y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
        return {"X": xt, "Y": yt}

    def __getitem__(
        self, idx: int | Tuple[int, int] | Sequence[int]
    ) -> Mapping[str, torch.Tensor]:
        if isinstance(idx, tuple) and len(idx) == 2:
            s, e = int(idx[0]), int(idx[1])
            return self._slice(s, e)
        if isinstance(idx, torch.Tensor) and idx.dtype in (torch.int64, torch.int32):
            idx = idx.tolist()
        if isinstance(idx, Sequence) and not isinstance(idx, (str, bytes, bytearray)):
            if len(idx) == 0:
                return self._slice(0, 0)
            idx_tensor = torch.as_tensor(list(idx), dtype=torch.long)
            if (
                self._perm_source == "runtime"
                and getattr(self, "_perm", None) is not None
            ):
                idx_tensor = self._perm.index_select(0, idx_tensor)
            try:
                x = self._features.index_select(0, idx_tensor)
            except Exception:
                x = (
                    self._features[idx_tensor]
                    if hasattr(self._features, "__getitem__")
                    else torch.as_tensor(self._features)[idx_tensor]
                )
            try:
                y = self._labels.index_select(0, idx_tensor)
            except Exception:
                y = (
                    self._labels[idx_tensor]
                    if hasattr(self._labels, "__getitem__")
                    else torch.as_tensor(self._labels)[idx_tensor]
                )
            if self._label_shape:
                y = y.reshape(y.shape[0], *self._label_shape)
            return {"X": x, "Y": y}
        i = self._start + int(idx)
        out = self._slice(i, i + 1)
        try:
            x = out.get("X", None)
            y = out.get("Y", None)
            if torch.is_tensor(x):
                x = x.squeeze(0)
            if torch.is_tensor(y):
                y = y.squeeze(0)
            return {"X": x, "Y": y}
        except Exception:
            return out

    def _shard(self) -> None:
        start = int(getattr(self, "start", 0))
        end = int(getattr(self, "end", 0))
        B = max(1, int(self._S_B))
        cuts = list(range(start, end + 1, B))
        if cuts and cuts[-1] != end:
            cuts.append(end)
        elif not cuts:
            cuts = [start, end]
        self._S_cuts = cuts
        try:
            dist = getattr(torch, "distributed", None)
            if dist is not None and dist.is_available() and dist.is_initialized():
                self._num_shards = max(1, int(dist.get_world_size()))
                self._shard_id = max(0, int(dist.get_rank()))
            else:
                env_world = os.environ.get("WORLD_SIZE")
                env_rank = os.environ.get("RANK")
                try:
                    env_world_int = int(env_world) if env_world is not None else 1
                except Exception:
                    env_world_int = 1
                try:
                    env_rank_int = int(env_rank) if env_rank is not None else 0
                except Exception:
                    env_rank_int = 0
                if env_world_int > 1:
                    self._num_shards = env_world_int
                    self._shard_id = max(0, min(env_world_int - 1, env_rank_int))
                else:
                    self._num_shards = 1
                    self._shard_id = 0
        except Exception:
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

        self._S_B = max(1, int(batch_size))
        self._S_shuffle = bool(shuffle)
        self._S_seed = int(seed)
        self._S_rng = random.Random(self._S_seed)
        self._key = str(key)
        self._shard()
        if SamplerWrapper is None:
            raise RuntimeError("torchdata.nodes.SamplerWrapper is required")
        return SamplerWrapper(self)

    def __iter__(self) -> Iterator[tuple[str, tuple[int, int]]]:
        cuts = getattr(self, "_S_cuts", None)
        if not cuts:
            self._shard()
            cuts = self._S_cuts
        n = max(0, len(cuts) - 1)
        idxs = list(range(n))
        if self._S_shuffle:
            try:
                self._S_rng.shuffle(idxs)
            except Exception:
                pass
        ns = getattr(self, "_num_shards", 1)
        si = getattr(self, "_shard_id", 0)
        if ns > 1:
            idxs = idxs[si::ns]
        for i in idxs:
            s = cuts[i]
            e = cuts[i + 1]
            if e > s:
                yield (self._key, (int(s), int(e)))

    def __len__(self) -> int:
        start = int(getattr(self, "start", 0))
        end = int(getattr(self, "end", 0))
        B = max(1, int(self._S_B))
        total = max(0, (end - start + B - 1) // B)
        ns = getattr(self, "_num_shards", 1)
        si = getattr(self, "_shard_id", 0)
        if ns <= 1:
            return total
        return max(0, (total - si + ns - 1) // ns)

    def set_epoch(self, epoch: int) -> None:
        try:
            self._S_rng.seed(self._S_seed + int(epoch))
        except Exception:
            pass

    def get(self, start: int, end: int) -> Mapping[str, Any]:
        from ..functional.fx import Gradient as FxGradient

        s = int(start)
        e = int(end)
        n = max(0, e - s)

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
                X = self._X.narrow(0, 0, 0)
                Y = self._Y.narrow(0, 0, 0)
                return {"X": X, "Y": Y}
            X = self._X.narrow(0, s, n)
            Y = self._Y.narrow(0, s, n)
            if self._label_shape:
                Y = Y.view(n, *self._label_shape)
            if _is_accelerator_available():
                try:
                    X = X.pin_memory()
                    Y = Y.pin_memory()
                except Exception:
                    pass
            return {"X": X, "Y": Y}


def _to_high_precision(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.is_floating_point():
        return tensor.to(dtype=torch.float64)
    if tensor.dtype in {torch.uint8, torch.int8, torch.int16, torch.int32}:
        return tensor.to(dtype=torch.int64)
    return tensor


def preload_memmap(
    data: Mapping[str, Any],
    *args: Any,
    memmap_dir: str,
    train_frac: float = 1.0,
    val_frac: float = 0.0,
    shuffle: bool = False,
    seed: int | None = None,
    **kwargs: Any,
) -> None:
    if not isinstance(data, Mapping):
        raise TypeError("preload_memmap expects a Mapping with 'features' and 'labels'")
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

    features_path = os.path.join(memmap_dir, "features.mmt")
    labels_path = os.path.join(memmap_dir, "labels.mmt")

    MemoryMappedTensor.from_tensor(feature_flat, filename=features_path)
    MemoryMappedTensor.from_tensor(label_flat, filename=labels_path)

    val_count = max(0, min(count, int(round(count * float(val_frac)))))
    train_count = max(0, min(count, count - val_count))
    train_start, train_end = 0, train_count
    val_start, val_end = train_end, train_end + val_count

    meta: Dict[str, Any] = {
        "N": count,
        "feature_dim": int(feature_flat.shape[1]),
        "features_path": "features.mmt",
        "labels_path": "labels.mmt",
        "label_shape": list(label_shape),
        "features_dtype": to_platform_dtype(feature_flat.dtype, "name"),
        "labels_dtype": to_platform_dtype(label_flat.dtype, "name"),
        "fractions": [float(train_frac), float(val_frac)],
        "shuffled": bool(shuffle),
        "shuffle_seed": int(seed) if seed is not None else None,
        "train_start": train_start,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
    }

    if perm is not None and seed is not None:
        meta["perm_filename"] = "perm.pt"

    with open(os.path.join(memmap_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle)


SourceKind = Literal["memmap"]


class SourceSpec(TypedDict):
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


class Sampler:
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
        io_workers: Optional[int] = None,
        prebatch: Optional[int] = None,
        prefetch_factor: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
        non_blocking: bool = True,
        pin_memory: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        self.map_fn = map_fn
        try:
            from ..backend.system import optimal_threads

            _t = optimal_threads()
        except Exception:
            _t = {
                "num_workers": 1,
                "max_concurrancy": 1,
                "prebatch": 2,
                "prefetch_factor": 1,
            }

        self.io_workers = (
            int(io_workers) if io_workers is not None else int(_t.get("num_workers", 1))
        )
        self.io_workers = max(1, self.io_workers)

        self.prebatch = (
            int(prebatch)
            if prebatch is not None
            else int(_t.get("prebatch", max(1, self.io_workers * 2)))
        )
        with suppress(Exception):
            self.prebatch = max(1, int(self.prebatch))

        pf = (
            int(prefetch_factor)
            if prefetch_factor is not None
            else int(_t.get("prefetch_factor", 1))
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
        self.max_concurrancy = max(1, int(self.io_workers))
        try:
            from ..backend.system import get_tlb
            get_tlb(io_workers=self.io_workers)
        except Exception:
            pass

    def compose(self, source: "BaseNode") -> "BaseNode":
        from ..backend.system import wrap_with_tlb
        
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
            max_concurrent=int(self.max_concurrancy),
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
        do_pin = bool(pin_memory) if pin_memory is not None else (getattr(dev, 'type', 'cpu') in {'cuda','xpu','mps'})
        if do_pin:
            node = PinMemory(node, pin_memory_device=dev.type)
        node = _Prefetcher(node, prefetch_factor=int(prefetch_factor))
        return Loader(
            dev,
            node=node,
            prefetch_factor=int(prefetch_factor),
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
        self._device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )
        self._depth = max(1, int(depth))
        self._non_blocking = bool(non_blocking)
        self._backpressure = bool(oom_safe)
        self._gpu_guard_bytes = int(gpu_guard_bytes or 0)
        self._host_guard_bytes = int(host_guard_bytes or 0)
        use_accel = isinstance(self._device, torch.device) and self._device.type in ("cuda", "xpu")
        self._pin   = bool(kwargs.get("pin_host", use_accel))                             
        self._gpu_stream = None

    def _to_device(self, x: Any, device: torch.device) -> Any:
        if torch.is_tensor(x):
            return x.to(device, non_blocking=True)
        if isinstance(x, (list, tuple)):
            return type(x)(self._to_device(t, device) for t in x)
        if isinstance(x, dict):
            return {k: self._to_device(v, device) for k, v in x.items()}
        return x

    def _pin_memory(self, x: Any) -> Any:
        if not self._pin:
            return x
        if torch.is_tensor(x) and x.device.type == "cpu":
            return x.pin_memory()
        if isinstance(x, (list, tuple)):
            return type(x)(self._pin_memory(t) for t in x)
        if isinstance(x, dict):
            return {k: self._pin_memory(v) for k, v in x.items()}
        return x

    def __iter__(self) -> Iterator[Any]:
        import time

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
            try:
                import psutil            
                return bool(psutil.virtual_memory().available >= self._host_guard_bytes)
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
            preloaded = None
            it = iter(iterable)

            def _preload() -> Any | None:
                try:
                    b = next(it)
                except StopIteration:
                    return None
                b = self._pin_memory(b)
                                                       
                tries = 0
                if self._backpressure:
                    while (not _host_mem_ok()) or (not _gpu_mem_ok()):
                        time.sleep(0.001 if tries < 1000 else 0.005)
                        tries += 1
                if use_device:
                    if use_cuda_stream and self._gpu_stream is not None:
                        with torch.cuda.stream(self._gpu_stream):
                            b = self._to_device(b, device)
                    else:
                                            
                        b = self._to_device(b, device)
                return b

                       
            preloaded = _preload()
            while preloaded is not None:
                if use_cuda_stream and self._gpu_stream is not None:
                    cs = torch.cuda.current_stream(
                        device=device if isinstance(device, torch.device) else None
                    )
                    cs.wait_stream(self._gpu_stream)
                                                         
                    def _record(x: Any) -> None:
                        if torch.is_tensor(x):
                            x.record_stream(cs)
                            return
                        if isinstance(x, (list, tuple)):
                            for t in x:
                                _record(t)
                        elif isinstance(x, dict):
                            for t in x.values():
                                _record(t)
                    try:
                        _record(preloaded)
                    except Exception:
                        pass
                yield preloaded
                preloaded = _preload()
