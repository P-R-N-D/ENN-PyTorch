# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import random
from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

TensorLike = Any

import torch
from tensordict import TensorDict, TensorDictBase, stack

from ..backend.system import get_tlb, optimize_threads, Memory

try:
    from torchdata.nodes import BaseNode
except Exception:
    from torchdata.nodes import BaseNode

from .nodes import (
    Dataset,
    Disposable,
    Connector,
    Loader,
    Sampler,
    SourceSpec,
)


def _sync_device(device: torch.device) -> None:
    dev_t = getattr(device, "type", "cpu")
    try:
        if dev_t == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device=device)
        elif dev_t == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif dev_t == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception:
        pass


def _sample_size(
    _x_cpu: torch.Tensor, _y_cpu: Optional[torch.Tensor]
) -> int:
    x_one = _x_cpu[0]
    bx = int(x_one.numel()) * int(x_one.element_size())
    by = 0
    if _y_cpu is not None:
        y_one = _y_cpu[0]
        by = int(y_one.numel()) * int(y_one.element_size())
    return int(bx + by)


def _random_batches(_sample_bytes: int, _device: torch.device, _N: int) -> Sequence[int]:
    capB = 1024
    dev_t = getattr(_device, "type", "cpu")
    if dev_t == "cuda":
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            free, _ = torch.cuda.mem_get_info(device=_device)
            if _sample_bytes > 0:
                capB = max(1, int((free * 0.90) // max(1, _sample_bytes * 4)))
        except Exception:
            pass
    elif dev_t == "xpu":
        try:
            props = getattr(torch.xpu, "get_device_properties", None)
            mem_alloc = getattr(torch.xpu, "memory_allocated", None)
            if callable(props) and callable(mem_alloc):
                total = int(props(_device).total_memory)
                used = int(mem_alloc(_device))
                free = max(0, total - used)
                if _sample_bytes > 0:
                    capB = max(1, int((free * 0.90) // max(1, _sample_bytes * 4)))
        except Exception:
            pass
    elif dev_t == "mps":
        try:
            free_host = int(Memory.available())
            if _sample_bytes > 0:
                capB = max(1, int((free_host * 0.25) // max(1, _sample_bytes * 4)))
        except Exception:
            pass
    capB = max(1, min(capB, int(_N)))
    base = [
        capB // 8,
        capB // 4,
        capB // 2,
        (capB * 3) // 8,
        (capB * 3) // 4,
        capB,
    ]
    cands = sorted({max(1, c) for c in base if c > 0})
    return [c for c in cands if c <= _N]


@torch.no_grad()
def _h2d_counter(
    _x_cpu: torch.Tensor,
    _y_cpu: Optional[torch.Tensor],
    _device: torch.device,
    _bs: int,
    _steps: int = 8,
    _warmup: int = 2,
) -> float:
    N = int(_x_cpu.shape[0])
    bs = max(1, min(int(_bs), N))

    times = []
    for s in range(_steps + _warmup):
        start = 0
        if N > bs:
            start = random.randint(0, N - bs)
        xb = _x_cpu[start : start + bs]
        yb = None
        if _y_cpu is not None:
            yb = _y_cpu[start : start + bs]
        xbp = xb if (hasattr(xb, "is_pinned") and xb.is_pinned()) else (xb.pin_memory() if _device.type in {"cuda", "xpu"} else xb)
        ybp = None
        if yb is not None:
            ybp = yb if (hasattr(yb, "is_pinned") and yb.is_pinned()) else (yb.pin_memory() if _device.type in {"cuda", "xpu"} else yb)
        _sync_device(_device)
        t0 = None
        t1 = None
        if _device.type == "cuda" and torch.cuda.is_available():
            try:
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
            except Exception:
                t0 = None
        if t0 is not None:
            with torch.cuda.device(_device):
                t0.record()
                _ = xbp.to(_device, non_blocking=True)
                if ybp is not None:
                    _ = ybp.to(_device, non_blocking=True)
                t1.record()
                _sync_device(_device)
                ms = float(t0.elapsed_time(t1))
        else:
            import time as _t

            tns0 = _t.perf_counter_ns()
            _ = xbp.to(_device, non_blocking=True)
            if ybp is not None:
                _ = ybp.to(_device, non_blocking=True)
            _sync_device(_device)
            tns1 = _t.perf_counter_ns()
            ms = (tns1 - tns0) / 1e6
        if s >= _warmup:
            times.append(ms)
    if not times:
        return 0.0
    times.sort()
    return float(times[len(times) // 2])


def _batch_interval(
    _ds: "Dataset",
    _dev: torch.device,
    _tmin_ms: float = 0.8,
    _tmax_ms: float = 3.0,
) -> Tuple[int, float]:
    if len(_ds) <= 0:
        return (1, 0.0)
    probe = _ds.get(0, min(8, len(_ds)))
    x_cpu = probe["X"]
    y_cpu = probe.get("Y", None)
    if not isinstance(x_cpu, torch.Tensor):
        x_cpu = torch.as_tensor(x_cpu)
    if y_cpu is not None and not isinstance(y_cpu, torch.Tensor):
        y_cpu = torch.as_tensor(y_cpu)
    sbytes = _sample_size(x_cpu, y_cpu)
    if sbytes <= 0:
        return (max(1, min(256, len(_ds))), 0.0)
    B_cap = 1 << 16
    if _dev.type == "cuda":
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            free, _ = torch.cuda.mem_get_info(device=_dev)
            cap = int(free * 0.90)
            B_cap = max(1, int(cap // max(1, sbytes * 4)))
        except Exception:
            pass
    elif _dev.type == "xpu":
        try:
            props = getattr(torch.xpu, "get_device_properties", None)
            mem_alloc = getattr(torch.xpu, "memory_allocated", None)
            if callable(props) and callable(mem_alloc):
                total = int(props(_dev).total_memory)
                used = int(mem_alloc(_dev))
                cap = int(max(0, total - used) * 0.90)
                B_cap = max(1, int(cap // max(1, sbytes * 4)))
        except Exception:
            pass
    elif _dev.type == "mps":
        try:
            cap = int(int(Memory.available()) * 0.25)
            B_cap = max(1, int(cap // max(1, sbytes * 4)))
        except Exception:
            pass
    B_cap = max(1, min(int(B_cap * Dataset._scale), len(_ds)))

    if _dev.type == "cuda":
        try:
            per_sample = int(getattr(Dataset, "_per_sample_mem_bytes", 0) or 0)
        except Exception:
            per_sample = 0
        if per_sample > 0:
            try:
                free_bytes, _ = torch.cuda.mem_get_info(_dev)
                safe_cap = int(max(0, free_bytes) * 0.80)
                if safe_cap > 0:
                    cap_from_mem = int(safe_cap // max(1, per_sample))
                    if cap_from_mem > 0:
                        B_cap = max(1, min(B_cap, cap_from_mem))
            except Exception:
                pass

    candidates = _random_batches(sbytes, _dev, len(_ds))
    if candidates:
        B = min(candidates[-1], B_cap)
    else:
        B = min(64, B_cap)
    med = _h2d_counter(x_cpu, y_cpu, _dev, B)
    while med > 0.0 and med < _tmin_ms and B < B_cap:
        B_next = min(B * 2, B_cap)
        med_next = _h2d_counter(x_cpu, y_cpu, _dev, B_next)
        if med_next <= 0.0:
            break
        B, med = B_next, med_next
    while med > _tmax_ms and B > 1:
        B_next = max(1, B // 2)
        if B_next == B:
            break
        med_next = _h2d_counter(x_cpu, y_cpu, _dev, B_next)
        if med_next <= 0.0:
            break
        B, med = B_next, med_next
    return (max(1, int(B)), float(med))


def _is_source_spec(obj: Any) -> bool:
    if not isinstance(obj, Mapping):
        return False
    if "kind" not in obj or "path" not in obj:
        return False
    p = obj.get("path")
    try:
        os.fspath(p)
    except Exception:
        return False
    return True


def dataset(
    source: SourceSpec,
    *args: Any,
    split: str = "train",
    val_frac: float = 0.0,
    **kwargs: Any,
) -> "Dataset":
    if not isinstance(source, Mapping):
        raise TypeError(f"dataset expects a SourceSpec mapping, got {type(source)}")

    kind = str(source.get("kind"))
    if kind != "memmap":
        raise ValueError(f"Unsupported source kind: {kind!r}")
    path = os.fspath(source.get("path", ""))
    if not path:
        raise ValueError("SourceSpec['path'] must be provided")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"memmap directory not found: {path!r}")
    sp = str(split or "train")
    if sp not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got: {sp!r}")
    vf = float(val_frac)
    if not (0.0 <= vf <= 1.0):
        raise ValueError(f"val_frac must be in [0,1], got: {vf}")
    return Dataset(path, split=sp, val_frac=vf)


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
        features = features.view(features.shape[0], -1)
    if labels_dtype is not None and isinstance(labels, torch.Tensor):
        labels = labels.to(dtype=labels_dtype, non_blocking=True, copy=False)
    if sanitize and torch.is_floating_point(labels):
        torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0, out=labels)
    return {"X": features, "Y": labels}


def collate(
    batch: Any,
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
    if isinstance(batch, (list, tuple)):
        if not batch:
            return batch
        if all(isinstance(elem, TensorDictBase) for elem in batch):
            stacked = stack(list(batch), dim=0)
            try:
                conv = converter(stacked)
            except Exception:
                return stacked
            stacked["X"] = conv["X"]
            stacked["Y"] = conv["Y"]
            stacked["features"] = conv["X"]
            stacked["labels"] = conv["Y"]
            return stacked
        if all(isinstance(elem, Mapping) for elem in batch):
            Xs = [elem.get("X") for elem in batch]
            Ys = [elem.get("Y") for elem in batch]
            try:
                if all(isinstance(x, torch.Tensor) for x in Xs):
                    Xs = torch.stack(Xs, dim=0)
            except Exception:
                pass
            try:
                if all(isinstance(y, torch.Tensor) for y in Ys):
                    Ys = torch.stack(Ys, dim=0)
            except Exception:
                pass
            try:
                conv = converter({"X": Xs, "Y": Ys})
            except Exception:
                conv = {"X": Xs, "Y": Ys}
            Xs = conv.get("X", Xs)
            Ys = conv.get("Y", Ys)
            return TensorDict({"X": Xs, "Y": Ys, "features": Xs, "labels": Ys}, batch_size=[])
        return batch
    if isinstance(batch, Mapping):
        try:
            conv = converter(batch)
        except Exception:
            conv = batch
        X = conv.get("X", batch.get("X"))
        Y = conv.get("Y", batch.get("Y"))
        return TensorDict({"X": X, "Y": Y, "features": X, "labels": Y}, batch_size=[])
    return batch

def compose(
    node_or_nodes: Union[BaseNode, Sequence[BaseNode], Mapping[str, BaseNode]],
    *args: Any,
    device: Union[str, torch.device],
    map_fn: Callable[[Any], Any],
    prefetch_factor: int,
    non_blocking_copy: bool,
    io_workers: int,
    prebatch: int,
    weights: Optional[Mapping[str, float]] = None,
    **kwargs: Any,
) -> Tuple[BaseNode, BaseNode, BaseNode]:
    device_obj = (
        torch.device(device) if not isinstance(device, torch.device) else device
    )
    try:
        get_tlb(io_workers=io_workers)
    except Exception:
        pass
    mx_weights = None
    if isinstance(node_or_nodes, Mapping) and isinstance(weights, Mapping):
        mx_weights = weights
    sampler = Sampler(stop_criteria="ALL_DATASETS_EXHAUSTED", seed=0, weights=mx_weights)
    source = sampler.compose(node_or_nodes)
    mapper = Connector(
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
    sources: Union[
        SourceSpec,
        Sequence[SourceSpec],
        Mapping[str, SourceSpec],
    ],
    device: Union[str, torch.device],
    val_frac: float = 0.0,
    non_blocking_copy: bool = True,
    labels_dtype: Optional[torch.dtype] = None,
    sanitize: bool = True,
    flatten_features: bool = True,
    train_weights: Optional[Mapping[str, float]] = None,
    val_weights: Optional[Mapping[str, float]] = None,
) -> Tuple[Any, Optional[Any], Disposable]:
    device_obj = (
        torch.device(device) if not isinstance(device, torch.device) else device
    )
    hints = optimize_threads()
    io_workers = max(1, int(hints.get("num_workers", 1)))
    prebatch = max(1, int(hints.get("prebatch", max(1, io_workers * 2))))
    pf_depth = max(1, int(hints.get("prefetch_factor", 1)))

    map_fn = partial(
        collate,
        labels_dtype=labels_dtype,
        sanitize=sanitize,
        flatten_features=flatten_features,
    )

    allocated = Disposable()
    batch_size: Optional[int] = None

    def _stream_batch(_ds: Dataset, _dev: torch.device) -> Tuple[int, float]:
        try:
            return _batch_interval(_ds, _dev)
        except Exception:
            return (int(batch_size) if batch_size is not None else 0, 0.0)

    _device_obj = device_obj
    _auto_bs_candidates: list[int] = []
    _auto_ms_candidates: list[float] = []

    if isinstance(sources, Mapping) and not _is_source_spec(sources):
        datasets: Dict[str, Any] = {}
        for key, spec in sources.items():
            ds = dataset(spec, split="train", val_frac=float(val_frac))
            allocated.add(ds)
            datasets[str(key)] = ds
        if batch_size is None or int(batch_size) <= 0:
            for _k, _ds in datasets.items():
                B_i, ms_i = _stream_batch(_ds, _device_obj)
                if B_i > 0:
                    _auto_bs_candidates.append(B_i)
                    _auto_ms_candidates.append(ms_i)
            if _auto_bs_candidates:
                cand_mean = int(sum(_auto_bs_candidates) // len(_auto_bs_candidates))
                cand_max = max(_auto_bs_candidates)
                batch_size = max(1, min(cand_max, cand_mean))
                if _auto_ms_candidates:
                    _m = min(_auto_ms_candidates)
                    if _m < 0.35:
                        pf_depth = max(pf_depth, 6)
                    elif _m < 0.70:
                        pf_depth = max(pf_depth, 4)
                    elif _m < 1.00:
                        pf_depth = max(pf_depth, 3)
                pf_depth = int(max(2, min(8, pf_depth)))
            else:
                batch_size = 1
        sampler_nodes: Dict[str, BaseNode] = {}
        lengths: Dict[str, int] = {}
        for key, ds in datasets.items():
            sampler_node = ds.compose(
                batch_size=int(batch_size),
                shuffle=True,
                seed=0,
                key=str(key),
            )
            if len(ds) > 0:
                sampler_nodes[str(key)] = sampler_node
                lengths[str(key)] = len(ds)

        if isinstance(train_weights, Mapping):
            train_weights = {
                k: v for k, v in dict(train_weights).items() if k in sampler_nodes
            }

        if not sampler_nodes:
            raise RuntimeError("No non-empty training sources provided")

        def iterate(sample):
            def _one(smpl):
                if (
                    isinstance(smpl, (list, tuple))
                    and len(smpl) == 2
                    and isinstance(smpl[0], str)
                ):
                    k, rng = smpl
                    s, e = int(rng[0]), int(rng[1])
                    ds = datasets.get(k)
                    if ds is None:
                        raise KeyError(f"Unknown dataset key: {k}")
                    batch = ds.get(s, e)
                    return map_fn(batch)
                return map_fn(smpl)

            if isinstance(sample, list):
                return [_one(smpl) for smpl in sample]
            return _one(sample)

        _, mapped, _ = compose(
            sampler_nodes,
            device=device_obj,
            map_fn=iterate,
            prefetch_factor=int(pf_depth),
            non_blocking_copy=bool(non_blocking_copy),
            io_workers=io_workers,
            prebatch=prebatch,
            weights=train_weights,
        )
        train_length = sum(lengths.values()) if lengths else None

    elif isinstance(sources, (list, tuple)):
        datasets: Dict[str, Any] = {}
        for i, spec in enumerate(sources):
            key = str(i)
            ds = dataset(spec, split="train", val_frac=float(val_frac))
            allocated.add(ds)
            datasets[key] = ds
        if batch_size is None or int(batch_size) <= 0:
            for _k, _ds in datasets.items():
                B_i, ms_i = _stream_batch(_ds, _device_obj)
                if B_i > 0:
                    _auto_bs_candidates.append(B_i)
                    _auto_ms_candidates.append(ms_i)
            if _auto_bs_candidates:
                cand_mean = int(sum(_auto_bs_candidates) // len(_auto_bs_candidates))
                cand_max = max(_auto_bs_candidates)
                batch_size = max(1, min(cand_max, cand_mean))
                if _auto_ms_candidates:
                    _m = min(_auto_ms_candidates)
                    if _m < 0.35:
                        pf_depth = max(pf_depth, 6)
                    elif _m < 0.70:
                        pf_depth = max(pf_depth, 4)
                    elif _m < 1.00:
                        pf_depth = max(pf_depth, 3)
                pf_depth = int(max(2, min(8, pf_depth)))
            else:
                batch_size = 1
        sampler_list: list[BaseNode] = []
        lengths: list[int] = []
        for key, ds in datasets.items():
            sampler_node = ds.compose(
                batch_size=int(batch_size),
                shuffle=True,
                seed=0,
                key=str(key),
            )
            if len(ds) > 0:
                sampler_list.append(sampler_node)
                lengths.append(len(ds))

        if not sampler_list:
            raise RuntimeError("No non-empty training sources provided")

        def iterate(sample):
            def _one(smpl):
                if (
                    isinstance(smpl, (list, tuple))
                    and len(smpl) == 2
                    and isinstance(smpl[0], str)
                ):
                    k, rng = smpl
                    s, e = int(rng[0]), int(rng[1])
                    ds = datasets.get(k)
                    if ds is None:
                        raise KeyError(f"Unknown dataset key: {k}")
                    batch = ds.get(s, e)
                    return map_fn(batch)
                return map_fn(smpl)

            if isinstance(sample, list):
                return [_one(smpl) for smpl in sample]
            return _one(sample)

        _, mapped, _ = compose(
            sampler_list,
            device=device_obj,
            map_fn=iterate,
            prefetch_factor=int(pf_depth),
            non_blocking_copy=bool(non_blocking_copy),
            io_workers=io_workers,
            prebatch=prebatch,
            weights=train_weights,
        )
        train_length = sum(lengths) if lengths else None

    else:
        ds = dataset(sources, split="train", val_frac=float(val_frac))
        allocated.add(ds)
        if batch_size is None or int(batch_size) <= 0:
            B_i, ms_i = _stream_batch(ds, _device_obj)
            batch_size = max(1, int(B_i) if B_i > 0 else 1)
            if ms_i:
                if ms_i < 0.35:
                    pf_depth = max(pf_depth, 6)
                elif ms_i < 0.70:
                    pf_depth = max(pf_depth, 4)
                elif ms_i < 1.00:
                    pf_depth = max(pf_depth, 3)
        sampler_node = ds.compose(
            batch_size=int(batch_size),
            shuffle=True,
            seed=0,
            key="0",
        )
        datasets: Dict[str, Any] = {"0": ds}

        def iterate(sample):
            def _one(smpl):
                if (
                    isinstance(smpl, (list, tuple))
                    and len(smpl) == 2
                    and isinstance(smpl[0], str)
                ):
                    k, rng = smpl
                    s, e = int(rng[0]), int(rng[1])
                    ds_ = datasets.get(k)
                    if ds_ is None:
                        raise KeyError(f"Unknown dataset key: {k}")
                    batch = ds_.get(s, e)
                    return map_fn(batch)
                return map_fn(smpl)

            if isinstance(sample, list):
                return [_one(smpl) for smpl in sample]
            return _one(sample)

        _, mapped, _ = compose(
            sampler_node,
            device=device_obj,
            map_fn=iterate,
            prefetch_factor=int(pf_depth),
            non_blocking_copy=bool(non_blocking_copy),
            io_workers=io_workers,
            prebatch=prebatch,
            weights=train_weights,
        )
        train_length = len(ds)

    train_loader = Loader.compose(
        mapped,
        device=device_obj,
        prefetch_factor=int(pf_depth),
        non_blocking=bool(non_blocking_copy),
        length=train_length,
    )

    val_loader = None
    if float(val_frac) > 0.0:
        if isinstance(sources, Mapping) and not _is_source_spec(sources):
            datasets: Dict[str, Any] = {}
            for key, spec in sources.items():
                ds = dataset(spec, split="val", val_frac=float(val_frac))
                allocated.add(ds)
                datasets[str(key)] = ds
            if batch_size is None or int(batch_size) <= 0:
                _auto_bs_candidates.clear()
                _auto_ms_candidates.clear()
                for _k, _ds in datasets.items():
                    B_i, ms_i = _stream_batch(_ds, _device_obj)
                    if B_i > 0:
                        _auto_bs_candidates.append(B_i)
                        _auto_ms_candidates.append(ms_i)
                if _auto_bs_candidates:
                    cand_mean = int(sum(_auto_bs_candidates) // len(_auto_bs_candidates))
                    cand_max = max(_auto_bs_candidates)
                    batch_size = max(1, min(cand_max, cand_mean))
                    if _auto_ms_candidates:
                        _m = min(_auto_ms_candidates)
                        if _m < 0.35:
                            pf_depth = max(pf_depth, 6)
                        elif _m < 0.70:
                            pf_depth = max(pf_depth, 4)
                        elif _m < 1.00:
                            pf_depth = max(pf_depth, 3)
                    pf_depth = int(max(2, min(8, pf_depth)))
            sampler_nodes: Dict[str, BaseNode] = {}
            lengths: Dict[str, int] = {}
            for key, ds in datasets.items():
                sn = ds.compose(
                    batch_size=int(batch_size),
                    shuffle=False,
                    seed=0,
                    key=str(key),
                )
                if len(ds) > 0:
                    sampler_nodes[str(key)] = sn
                    lengths[str(key)] = len(ds)

            if isinstance(val_weights, Mapping):
                val_weights = {
                    k: v for k, v in dict(val_weights).items() if k in sampler_nodes
                }
            if not sampler_nodes:
                raise RuntimeError("No non-empty validation sources provided")

            def iterate(sample):
                def _one(smpl):
                    if (
                        isinstance(smpl, (list, tuple))
                        and len(smpl) == 2
                        and isinstance(smpl[0], str)
                    ):
                        k, rng = smpl
                        s, e = int(rng[0]), int(rng[1])
                        ds = datasets.get(k)
                        if ds is None:
                            raise KeyError(f"Unknown dataset key: {k}")
                        batch = ds.get(s, e)
                        return map_fn(batch)
                    return map_fn(smpl)

                if isinstance(sample, list):
                    return [_one(smpl) for smpl in sample]
                return _one(sample)

            _, mapped_val, _ = compose(
                sampler_nodes,
                device=device_obj,
                map_fn=iterate,
                prefetch_factor=int(pf_depth),
                non_blocking_copy=bool(non_blocking_copy),
                io_workers=io_workers,
                prebatch=prebatch,
                weights=val_weights,
            )
            val_loader = Loader.compose(
                mapped_val,
                device=device_obj,
                prefetch_factor=int(pf_depth),
                non_blocking=bool(non_blocking_copy),
                length=sum(lengths.values()) if lengths else None,
            )

        elif isinstance(sources, (list, tuple)):
            datasets: Dict[str, Any] = {}
            for i, spec in enumerate(sources):
                k = str(i)
                ds = dataset(spec, split="val", val_frac=float(val_frac))
                allocated.add(ds)
                datasets[k] = ds
            if batch_size is None or int(batch_size) <= 0:
                _auto_bs_candidates.clear()
                _auto_ms_candidates.clear()
                for _k, _ds in datasets.items():
                    B_i, ms_i = _stream_batch(_ds, _device_obj)
                    if B_i > 0:
                        _auto_bs_candidates.append(B_i)
                        _auto_ms_candidates.append(ms_i)
                if _auto_bs_candidates:
                    cand_mean = int(sum(_auto_bs_candidates) // len(_auto_bs_candidates))
                    cand_max = max(_auto_bs_candidates)
                    batch_size = max(1, min(cand_max, cand_mean))
                    if _auto_ms_candidates:
                        _m = min(_auto_ms_candidates)
                        if _m < 0.35:
                            pf_depth = max(pf_depth, 6)
                        elif _m < 0.70:
                            pf_depth = max(pf_depth, 4)
                        elif _m < 1.00:
                            pf_depth = max(pf_depth, 3)
                    pf_depth = int(max(2, min(8, pf_depth)))
            sampler_list: list[BaseNode] = []
            lengths: list[int] = []
            for k, ds in datasets.items():
                sn = ds.compose(
                    batch_size=int(batch_size),
                    shuffle=False,
                    seed=0,
                    key=str(k),
                )
                if len(ds) > 0:
                    sampler_list.append(sn)
                    lengths.append(len(ds))
            if not sampler_list:
                raise RuntimeError("No non-empty validation sources provided")

            def iterate(sample):
                def _one(smpl):
                    if (
                        isinstance(smpl, (list, tuple))
                        and len(smpl) == 2
                        and isinstance(smpl[0], str)
                    ):
                        k, rng = smpl
                        s, e = int(rng[0]), int(rng[1])
                        ds = datasets.get(k)
                        if ds is None:
                            raise KeyError(f"Unknown dataset key: {k}")
                        batch = ds.get(s, e)
                        return map_fn(batch)
                    return map_fn(smpl)

                if isinstance(sample, list):
                    return [_one(smpl) for smpl in sample]
                return _one(sample)

            _, mapped_val, _ = compose(
                sampler_list,
                device=device_obj,
                map_fn=iterate,
                prefetch_factor=int(pf_depth),
                non_blocking_copy=bool(non_blocking_copy),
                io_workers=io_workers,
                prebatch=prebatch,
                weights=val_weights,
            )
            val_loader = Loader.compose(
                mapped_val,
                device=device_obj,
                prefetch_factor=int(pf_depth),
                non_blocking=bool(non_blocking_copy),
                length=sum(lengths) if lengths else None,
            )

        else:
            ds = dataset(sources, split="val", val_frac=float(val_frac))
            allocated.add(ds)
            if batch_size is None or int(batch_size) <= 0:
                B_i, ms_i = _stream_batch(ds, _device_obj)
                batch_size = max(1, int(B_i) if B_i > 0 else 1)
                if ms_i:
                    if ms_i < 0.35:
                        pf_depth = max(pf_depth, 6)
                    elif ms_i < 0.70:
                        pf_depth = max(pf_depth, 4)
                    elif ms_i < 1.00:
                        pf_depth = max(pf_depth, 3)
            sampler_node = ds.compose(
                batch_size=int(batch_size),
                shuffle=False,
                seed=0,
                key="0",
            )
            datasets: Dict[str, Any] = {"0": ds}

            def iterate(sample):
                def _one(smpl):
                    if (
                        isinstance(smpl, (list, tuple))
                        and len(smpl) == 2
                        and isinstance(smpl[0], str)
                    ):
                        k, rng = smpl
                        s, e = int(rng[0]), int(rng[1])
                        ds_ = datasets.get(k)
                        if ds_ is None:
                            raise KeyError(f"Unknown dataset key: {k}")
                        batch = ds_.get(s, e)
                        return map_fn(batch)
                    return map_fn(smpl)

                if isinstance(sample, list):
                    return [_one(smpl) for smpl in sample]
                return _one(sample)

            _, mapped_val, _ = compose(
                sampler_node,
                device=device_obj,
                map_fn=iterate,
                prefetch_factor=int(pf_depth),
                non_blocking_copy=bool(non_blocking_copy),
                io_workers=io_workers,
                prebatch=prebatch,
                weights=val_weights,
            )
            val_loader = Loader.compose(
                mapped_val,
                device=device_obj,
                prefetch_factor=int(pf_depth),
                non_blocking=bool(non_blocking_copy),
                length=len(ds),
            )

    return train_loader, val_loader, allocated

