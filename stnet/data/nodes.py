# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import queue
import random
import threading
from contextlib import suppress, nullcontext
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple, TypedDict, Literal

import torch

from ..backend.system import wrap_with_tlb, get_tlb

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
        MultiNodeWeightedSampler,
        ParallelMapper,
        PinMemory,
        Prefetcher as _Prefetcher,
        SamplerWrapper,
    )
except Exception:
    from torchdata.nodes import Loader as _Loader, ParallelMapper, PinMemory, Prefetcher as _Prefetcher
    MultiNodeWeightedSampler = None
    SamplerWrapper = None

try:
    from torch.utils.data import Dataset as _TorchDataset, Sampler as _Sampler
except Exception:
    _TorchDataset = object
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


class Dataset(_TorchDataset):
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
        # optional permutation for random split/order
        self._perm: Optional[torch.Tensor] = None
        self._perm_runtime: bool = False
        perm_fn = (self._meta or {}).get("perm_filename", None)
        if perm_fn:
            perm_path = os.path.join(self.dir, str(perm_fn))
            if os.path.isfile(perm_path):
                with suppress(Exception):
                    self._perm = torch.load(perm_path, map_location="cpu")
                    meta_shuffled = bool((self._meta or {}).get("shuffled", False))
                    self._perm_runtime = not meta_shuffled
        if self._perm is None and bool(int(os.environ.get("STNET_RANDOM_SPLIT", "0"))):
            gen = torch.Generator(device="cpu")
            with suppress(Exception):
                gen.manual_seed(int(os.environ.get("STNET_SPLIT_SEED", "0") or "0"))
            with suppress(Exception):
                self._perm = torch.randperm(self._N, generator=gen)
                self._perm_runtime = True
        # validate permutation length / dtype / device
        if self._perm is not None:
            try:
                if int(self._perm.numel()) != self._N:
                    with suppress(Exception):
                        import warnings as _warn

                        _warn.warn(
                            f"[stnet] ignoring invalid perm: length={int(self._perm.numel())}, expected N={self._N}"
                        )
                    self._perm = None
                else:
                    # ensure proper dtype/device for fast indexing
                    if self._perm.dtype != torch.long:
                        with suppress(Exception):
                            self._perm = self._perm.to(dtype=torch.long)
                    if getattr(self._perm, "device", torch.device("cpu")).type != "cpu":
                        with suppress(Exception):
                            self._perm = self._perm.cpu()
            except Exception:
                self._perm = None
        if self._perm is None:
            self._perm_runtime = False
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
        if self._perm_runtime and getattr(self, "_perm", None) is not None:
            idx = self._perm[start:end]
            if idx.numel() == 0:
                x = self._features[:0]
                y = self._labels[:0]
            else:
                try:
                    x = self._features.index_select(0, idx)
                except Exception:
                    x = self._features[idx] if hasattr(self._features, "__getitem__") \
                        else torch.as_tensor(self._features)[idx]
                try:
                    y = self._labels.index_select(0, idx)
                except Exception:
                    y = self._labels[idx] if hasattr(self._labels, "__getitem__") \
                        else torch.as_tensor(self._labels)[idx]
        else:
            x = self._features[start:end]
            y = self._labels[start:end]
        if self._label_shape:
            y = y.reshape(end - start, *self._label_shape)
        xt = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        yt = y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
        return {"X": xt, "Y": yt}

    def __getitem__(self, idx: int | Tuple[int, int] | Sequence[int]) -> Mapping[str, torch.Tensor]:
        if isinstance(idx, tuple) and len(idx) == 2:
            s, e = int(idx[0]), int(idx[1])
            return self._slice(s, e)
        if isinstance(idx, torch.Tensor) and idx.dtype in (torch.int64, torch.int32):
            idx = idx.tolist()
        if isinstance(idx, Sequence) and not isinstance(idx, (str, bytes, bytearray)):
            # list-index mode: gather arbitrary indices
            if len(idx) == 0:
                return self._slice(0, 0)
            idx_tensor = torch.as_tensor(list(idx), dtype=torch.long)
            # optional permutation applied
            if self._perm_runtime and getattr(self, "_perm", None) is not None:
                idx_tensor = self._perm.index_select(0, idx_tensor)
            # (2) optional runtime index-range check (debug)
            #     유효 범위: [0, self._N - 1]
            #     STNET_DEBUG_IDX=1 일 때만 검사하여 오버헤드 최소화
            try:
                import os as _os

                if bool(int(_os.environ.get("STNET_DEBUG_IDX", "0"))):
                    if idx_tensor.numel():
                        _min = int(idx_tensor.min().item())
                        _max = int(idx_tensor.max().item())
                        if _min < 0 or _max >= self._N:
                            raise IndexError(
                                f"index out of range: valid [0,{self._N-1}], got [{_min},{_max}]"
                            )
            except Exception:
                pass
            try:
                x = self._features.index_select(0, idx_tensor)
            except Exception:
                # MemmapTensor 등에서 index_select 미구현 시 고급 인덱싱 폴백
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
        # standardize int-index path to single-sample (no batch dim)
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
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        # process sharding (DDP / torchrun)
        self._num_shards = 1
        self._shard_id = 0
        try:
            dist = getattr(torch, "distributed", None)
            if dist is not None and dist.is_available() and dist.is_initialized():
                self._num_shards = max(1, int(dist.get_world_size()))
                self._shard_id = max(0, int(dist.get_rank()))
            else:
                self._num_shards = max(1, int(os.environ.get("WORLD_SIZE", "1") or "1"))
                self._shard_id = max(0, int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0") or "0")))
        except Exception:
            pass
        self._cuts = list(range(self._start, self._end, self._B))
        if self._end > self._start and (not self._cuts or self._cuts[-1] != self._end):
            self._cuts.append(self._end)

    def __iter__(self):
        n = max(0, len(self._cuts) - 1)
        idxs = list(range(n))
        if self._shuffle:
            self._rng.shuffle(idxs)
        # shard by block
        ns = getattr(self, "_num_shards", 1)
        si = getattr(self, "_shard_id", 0)
        if ns > 1:
            idxs = idxs[si::ns]
        for i in idxs:
            s = self._cuts[i]
            e = self._cuts[i + 1]
            if e > s:
                # list-index mode
                yield list(range(s, e))

    def __len__(self) -> int:
        total = max(0, (self._end - self._start + self._B - 1) // self._B)
        ns = getattr(self, "_num_shards", 1)
        si = getattr(self, "_shard_id", 0)
        if ns <= 1:
            return total
        # ceil partition per shard
        return max(0, (total - si + ns - 1) // ns)

    # optional: epoch-aware shuffle determinism (DDP 전 과정 동일 시드 유지)
    def set_epoch(self, epoch: int) -> None:
        self._rng.seed(self._seed + int(epoch))

    def compose(self, dataset: "Dataset") -> "BaseNode":
        map_fn = getattr(dataset, "__getitem__", None)
        if SamplerWrapper is None or ParallelMapper is None or not callable(map_fn):
            raise RuntimeError("torchdata.nodes.SamplerWrapper and ParallelMapper are required")
        sampler_node = SamplerWrapper(self)
        return ParallelMapper(
            sampler_node,
            map_fn,
            num_workers=1,
            in_order=True,
            method="thread",
            multiprocessing_context=None,
            max_concurrent=None,
            snapshot_frequency=1,
            prebatch=None,
        )


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
        with suppress(Exception):
            self.prebatch = max(1, int(os.environ.get("STNET_PREBATCH", self.prebatch)))
        self.prefetch_factor = max(1, int(prefetch_factor))
        with suppress(Exception):
            self.prefetch_factor = max(1, int(os.environ.get("STNET_PREFETCH", self.prefetch_factor)))
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.non_blocking = bool(non_blocking)
        try:
            get_tlb(io_workers=self.io_workers)
        except Exception:
            pass

    def compose(self, source: "BaseNode") -> "BaseNode":
        node = ParallelMapper(
            source,
            map_fn=wrap_with_tlb(self.map_fn),
            num_workers=self.io_workers,
            in_order=False,
            method="thread",
            max_concurrent=None,
            prebatch=self.prebatch,
        )
        if self.device.type in {"cuda", "xpu", "mps"}:
            node = PinMemory(node, pin_memory_device=self.device.type)
        node = _Prefetcher(node, prefetch_factor=self.prefetch_factor)
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
        try:
            _env_depth = int(os.environ.get("STNET_PREFETCH", str(depth)))
        except Exception:
            _env_depth = int(depth)
        self._depth = max(1, int(_env_depth))
        self._non_blocking = bool(non_blocking)
        self._backpressure = bool(oom_safe)
        self._gpu_guard_bytes = int(gpu_guard_bytes or 0)
        self._host_guard_bytes = int(host_guard_bytes or 0)

    def __iter__(self) -> Iterator[Any]:

        def _producer():
            try:
                get_tlb().pin_thread()
            except Exception:
                pass
            try:
                dev_t = getattr(self._device, "type", "cpu")
                use_accel = dev_t in {"cuda", "xpu", "mps"}
                streams = None
                n_streams = 1
                if use_accel and self._non_blocking:
                    try:
                        n_streams = max(1, min(self._depth, int(os.environ.get("STNET_COPY_STREAMS", "3"))))
                    except Exception:
                        n_streams = max(1, min(self._depth, 3))
                    try:
                        streams = [torch.Stream(device=self._device) for _ in range(n_streams)]
                    except Exception:
                        streams = None
                        use_accel = False
                idx = 0
                for item in it:
                    if use_accel and streams is not None:
                        s = streams[idx % n_streams]
                        idx += 1
                        # 공통 스트림 컨텍스트 (CUDA/XPU/MPS)
                        with s:
                            moved = _to_device(item, self._device, non_blocking=True)
                            # 가능 시, 텐서에 스트림 사용 기록(캐싱 할당자 수명 관리)
                            try:
                                if isinstance(moved, Mapping):
                                    for _v in moved.values():
                                        if torch.is_tensor(_v):
                                            _v.record_stream(s)
                                elif torch.is_tensor(moved):
                                    moved.record_stream(s)
                            except Exception:
                                pass
                        try:
                            ev = s.record_event()
                        except Exception:
                            try:
                                s.synchronize()
                            except Exception:
                                pass
                            ev = None
                        q.put((moved, ev), block=True)
                    else:
                        moved = _to_device(item, self._device, non_blocking=self._non_blocking)
                        q.put((moved, None), block=True)
            except StopIteration:
                pass
            finally:
                q.put(sentinel, block=True)

        def _producer_wrapped():
            try:
                get_tlb().pin_thread()
            except Exception:
                pass
            return _producer()

        it = iter(self._src)

        # (1) consumer: 이벤트/스트림 경계에서 데이터 레디니스 보장
        #     producer 가 (moved, ev) 를 큐에 넣었고, 여기서 현재 실행 스트림이 ev 를 기다린다.
        #     백엔드별 current_stream 호출 후 wait_event(ev) (미지원/실패 시 ev.synchronize() 로 폴백)
        def _wait_ready(ev):
            if ev is None:
                return
            dev_t = getattr(self._device, "type", "cpu")
            try:
                if dev_t == "cuda" and hasattr(torch, "cuda"):
                    torch.cuda.current_stream(self._device).wait_event(ev)
                    return
                if dev_t == "xpu" and hasattr(torch, "xpu"):
                    torch.xpu.current_stream(self._device).wait_event(ev)
                    return
                if dev_t == "mps" and hasattr(torch, "mps"):
                    # current_stream(device) 서명 차이 폴백
                    try:
                        cs = torch.mps.current_stream(self._device)
                    except TypeError:
                        cs = torch.mps.current_stream()
                    # 우선 stream.wait_event(ev) 시도
                    try:
                        cs.wait_event(ev)
                        return
                    except Exception:
                        # 역방향: ev.wait(stream)도 시도 (버전별 지원 상이)
                        try:
                            ev.wait(cs)  # 지원 안 하면 예외
                            return
                        except Exception:
                            ev.synchronize()
                            return
            except Exception:
                pass
            # generic fallback
            try:
                ev.synchronize()
            except Exception:
                pass

        q: "queue.Queue[Optional[Any]]" = queue.Queue(maxsize=self._depth)
        sentinel = object()
        t = threading.Thread(target=_producer_wrapped, daemon=True)
        t.start()
        try:
            while True:
                item = q.get(block=True)
                if item is sentinel:
                    break
                moved, ev = item
                # (1) consumer-side event wait
                try:
                    _wait_ready(ev)
                except Exception:
                    # 최후 폴백(이벤트가 없거나 wait가 실패한 경우 백엔드별 동기화)
                    dev_t = getattr(self._device, "type", "cpu")
                    try:
                        if dev_t == "cuda" and hasattr(torch, "cuda"):
                            torch.cuda.synchronize(self._device)
                        elif dev_t == "xpu" and hasattr(torch, "xpu"):
                            torch.xpu.synchronize(self._device)
                        elif dev_t == "mps" and hasattr(torch, "mps"):
                            # mps는 장치 인자를 받지 않음
                            torch.mps.synchronize()
                    except Exception:
                        # 그래도 실패하면 그냥 패스(다음 배치에서 재시도)
                        pass
                yield moved
        finally:
            try:
                t.join(timeout=0.1)
            except Exception:
                pass
