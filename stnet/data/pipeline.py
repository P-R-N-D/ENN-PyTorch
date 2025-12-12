# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import random
import contextlib
from functools import partial
from typing import (Any, Callable, Dict, Mapping, Optional, Sequence, Tuple,
                    Union)

TensorLike = Any

import torch
from tensordict import TensorDict, TensorDictBase, stack

from ..backend.system import Memory, get_tlb
from ..api.templates import BatchPolicy, WorkerPolicy

try:
    from torchdata.nodes import BaseNode
except Exception:
    from torchdata.nodes import BaseNode

from .nodes import Connector, Dataset, Disposable, Loader, Sampler, SourceSpec


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


_device_mem_get_info = Memory.device_mem_get_info


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
    if _sample_bytes <= 0 or _N <= 0:
        return [1]

    capB = 1024
    dev_t = getattr(_device, "type", "cpu")

    host_free: Optional[int] = None
    with contextlib.suppress(Exception):
        host_free = int(Memory.available())

    dev_free, _ = _device_mem_get_info(_device)

    effective_free: Optional[int] = None
    if host_free is not None:
        effective_free = host_free
    if dev_free is not None:
        effective_free = dev_free if effective_free is None else min(effective_free, dev_free)

    if effective_free is not None:
        effective_free = max(0, int(effective_free))
        capB = max(1, int((effective_free * 0.80) // max(_sample_bytes * 4, 1)))
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
    *,
    prefetch_factor: int = 2,
    num_workers: int = 0,
    prebatch: int = 1,
    worker_policy: Optional[WorkerPolicy] = None,
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
    _dev_type = _dev.type

    per_sample = int(getattr(Dataset, "_per_sample_mem_bytes", 0) or 0)
    if per_sample <= 0:
        try:
            env_v = (
                os.environ.get("STNET_PER_SAMPLE_MEM_BYTES")
                or os.environ.get("STNET_DEVICE_BYTES_PER_SAMPLE")
            )
            if env_v is not None:
                per_sample = int(env_v)
        except Exception:
            per_sample = 0
    if per_sample <= 0:
        per_sample = sbytes

    if worker_policy is not None:
        _wp = worker_policy
        _max_conc = max(1, int(getattr(_wp, "max_concurrency", 1)))
        _streams = max(1, int(getattr(_wp, "h2d_streams", 1)))
        _lws = max(1, int(getattr(_wp, "local_world_size", 1)))
    else:
        try:
            _wp = WorkerPolicy.autotune()
            _max_conc = max(1, int(getattr(_wp, "max_concurrency", 1)))
            _streams = max(1, int(getattr(_wp, "h2d_streams", 1)))
            _lws = max(1, int(getattr(_wp, "local_world_size", 1)))
        except Exception:
            _wp = None
            _max_conc, _streams, _lws = (1, 1, 1)

    dev_margin = 0.90
    host_margin = 0.10

    # When budgets are not explicitly set, we derive a conservative cap from:
    #   - per-sample bytes (data size)
    #   - estimated pipeline inflight (prefetch/streams/workers)
    # This avoids hard-coding fixed caps like 8/16GB while still guarding against
    # pathological "auto-batch uses everything" behavior on large systems.
    budget_slack = 1.25
    with contextlib.suppress(Exception):
        v = os.environ.get("STNET_BUDGET_SLACK")
        if v is not None and str(v).strip():
            budget_slack = float(v)
    budget_slack = max(1.0, min(4.0, float(budget_slack)))

    dev_budget_ratio = 1.0
    dev_budget_min_bytes = 0
    dev_budget_max_bytes: Optional[int] = None

    host_budget_ratio = 1.0
    host_budget_min_bytes = 0
    host_budget_max_bytes: Optional[int] = None

    with contextlib.suppress(Exception):
        v = os.environ.get("STNET_DEVICE_MARGIN")
        if v is not None and str(v).strip():
            dev_margin = float(v)
    with contextlib.suppress(Exception):
        v = os.environ.get("STNET_HOST_MARGIN")
        if v is not None and str(v).strip():
            host_margin = float(v)

    with contextlib.suppress(Exception):
        v = os.environ.get("STNET_DEVICE_BUDGET_RATIO")
        if v is not None and str(v).strip():
            dev_budget_ratio = float(v)
    with contextlib.suppress(Exception):
        v = os.environ.get("STNET_DEVICE_BUDGET_MIN_BYTES")
        if v is not None and str(v).strip():
            dev_budget_min_bytes = int(v)
    with contextlib.suppress(Exception):
        v = os.environ.get("STNET_DEVICE_BUDGET_MAX_BYTES")
        if v is not None and str(v).strip():
            dev_budget_max_bytes = int(v)

    with contextlib.suppress(Exception):
        v = os.environ.get("STNET_HOST_BUDGET_RATIO")
        if v is not None and str(v).strip():
            host_budget_ratio = float(v)
    with contextlib.suppress(Exception):
        v = os.environ.get("STNET_HOST_BUDGET_MIN_BYTES")
        if v is not None and str(v).strip():
            host_budget_min_bytes = int(v)
    with contextlib.suppress(Exception):
        v = os.environ.get("STNET_HOST_BUDGET_MAX_BYTES")
        if v is not None and str(v).strip():
            host_budget_max_bytes = int(v)

    dev_margin = max(0.0, min(1.0, float(dev_margin)))
    host_margin = max(0.0, min(1.0, float(host_margin)))

    dev_budget_ratio = max(0.0, min(1.0, float(dev_budget_ratio)))
    host_budget_ratio = max(0.0, min(1.0, float(host_budget_ratio)))

    dev_budget_min_bytes = max(0, int(dev_budget_min_bytes))
    host_budget_min_bytes = max(0, int(host_budget_min_bytes))
    dev_budget_max_bytes = (
        None if dev_budget_max_bytes is None else max(0, int(dev_budget_max_bytes))
    )
    host_budget_max_bytes = (
        None if host_budget_max_bytes is None else max(0, int(host_budget_max_bytes))
    )

    tpl = BatchPolicy(
        sample_bytes=per_sample,
        host_sample_bytes=sbytes,
        prefetch_factor=max(int(prefetch_factor or 1), 1),
        num_workers=max(int(num_workers or 0), 0),
        prebatch=max(int(prebatch or 1), 1),
        num_streams=_streams,
        max_concurrency=_max_conc,
        local_world_size=_lws,
        min_batch=1,
        max_batch=B_cap,
        device_margin=float(dev_margin),
        host_margin=float(host_margin),
        device_budget_ratio=float(dev_budget_ratio),
        device_budget_min_bytes=int(dev_budget_min_bytes),
        device_budget_max_bytes=(
            None if dev_budget_max_bytes is None else int(dev_budget_max_bytes)
        ),
        host_budget_ratio=float(host_budget_ratio),
        host_budget_min_bytes=int(host_budget_min_bytes),
        host_budget_max_bytes=(
            None if host_budget_max_bytes is None else int(host_budget_max_bytes)
        ),
    )

    dev_free, dev_total = _device_mem_get_info(_dev)
    host_free: Optional[int] = None
    host_total: Optional[int] = None

    try:
        host_avail = int(Memory.available())
        if host_avail > 0:
            host_free = host_avail
        with contextlib.suppress(Exception):
            _ht = Memory.total()
            if _ht is not None and _ht > 0:
                host_total = int(_ht)
    except Exception:
        host_free = None

    # If budgets were not explicitly configured, derive a conservative cap.
    # We cap the *batch size* in samples by estimating how many samples are needed
    # to keep the pipeline fed (inflight batches), then converting to bytes.
    if (
        tpl.device_budget_max_bytes is None or tpl.host_budget_max_bytes is None
    ) and int(tpl.sample_bytes or 0) > 0:
        try:
            inflight = int(tpl.host_inflight_batches_per_proc())
            lw = max(1, int(getattr(tpl, "local_world_size", 1) or 1))
            # A small sample target that scales with inflight, but still bounded by B_cap.
            # This is meant to prevent huge auto-batches on large-memory machines.
            target_batch_samples = max(1, min(int(B_cap), max(64, inflight * 32)))

            if tpl.device_budget_max_bytes is None:
                base_dev = int(tpl.sample_bytes) * int(target_batch_samples)
                cap_dev = int(float(base_dev) * float(budget_slack))
                if dev_total is not None and int(dev_total) > 0:
                    cap_dev = min(int(cap_dev), int(dev_total))
                tpl.device_budget_max_bytes = max(0, int(cap_dev))

            if tpl.host_budget_max_bytes is None and int(tpl.host_sample_bytes or 0) > 0:
                base_host = int(tpl.host_sample_bytes) * max(1, inflight) * max(
                    1, lw
                ) * int(target_batch_samples)
                cap_host = int(float(base_host) * float(budget_slack))
                if host_total is not None and int(host_total) > 0:
                    cap_host = min(int(cap_host), int(host_total))
                tpl.host_budget_max_bytes = max(0, int(cap_host))
        except Exception:
            pass

    cap_from_mem = tpl.suggest_batch(
        dev_free=dev_free,
        host_free=host_free,
        dev_total=dev_total,
        host_total=host_total,
        local_world_size=_lws,
    )
    if cap_from_mem > 0:
        B_cap = min(B_cap, cap_from_mem)

    B_cap = max(1, min(int(B_cap * Dataset._scale), len(_ds)))

    env_max = os.environ.get("STNET_MAX_BATCH_SIZE") or os.environ.get("STNET_MAX_BATCH")
    try:
        if env_max:
            B_cap = max(1, min(B_cap, int(env_max)))
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
) -> Dict[str, Any]:
    device_obj = (
        torch.device(device) if not isinstance(device, torch.device) else device
    )
    _wp = WorkerPolicy.autotune()
    _wp.apply_torch_threads()
    io_workers = int(_wp.num_workers)
    prebatch = int(_wp.prebatch)
    pf_depth = int(_wp.prefetch_factor)

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
            return _batch_interval(
                _ds,
                _dev,
                prefetch_factor=pf_depth,
                num_workers=io_workers,
                prebatch=prebatch,
                worker_policy=_wp,
            )
        except Exception:
            return (int(batch_size) if batch_size is not None else 0, 0.0)

    def _rescale_batch(_datasets: Mapping[str, Dataset], _bs: int) -> int:
        _auto_bs_candidates.clear()
        for _k, _ds in _datasets.items():
            B_i, _ = _stream_batch(_ds, _device_obj)
            if B_i > 0:
                _auto_bs_candidates.append(B_i)
        if not _auto_bs_candidates:
            return int(_bs)
        cand_mean = int(sum(_auto_bs_candidates) // len(_auto_bs_candidates))
        cand_max = max(_auto_bs_candidates)
        return int(max(1, min(cand_max, cand_mean)))

    def _cap_pf_depth(
        _device_obj: torch.device, _datasets: Mapping[str, Dataset], _pf: int, _bs: int
    ) -> int:
        try:
            host_avail = int(Memory.available())
            if host_avail <= 0:
                return int(_pf)

            dev_free, _ = _device_mem_get_info(_device_obj)

            if dev_free is not None:
                effective_avail = min(host_avail, dev_free)
            else:
                effective_avail = host_avail

            budget = int(effective_avail * 0.15)
            if budget <= 0 or _bs <= 0:
                return int(_pf)
            sbytes_max = 0
            for _k, _ds in _datasets.items():
                if len(_ds) <= 0:
                    continue
                probe = _ds.get(0, min(8, len(_ds)))
                x = probe.get("X")
                y = probe.get("Y") if isinstance(probe, Mapping) else None
                if x is None:
                    continue
                if not isinstance(x, torch.Tensor):
                    x = torch.as_tensor(x)
                if y is not None and not isinstance(y, torch.Tensor):
                    y = torch.as_tensor(y)
                sbytes_max = max(sbytes_max, _sample_size(x, y))
            if sbytes_max <= 0:
                return int(_pf)
            bytes_per_batch = int(sbytes_max) * int(_bs)
            if bytes_per_batch <= 0:
                return int(_pf)
            pf_cap = max(1, int(budget // max(1, bytes_per_batch)))
            return int(max(1, min(int(_pf), pf_cap, 8)))
        except Exception:
            return int(_pf)

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
                pf_depth_before = int(pf_depth)
                if _auto_ms_candidates:
                    _m = min(_auto_ms_candidates)
                    if _m < 0.35:
                        pf_depth = max(pf_depth, 6)
                    elif _m < 0.70:
                        pf_depth = max(pf_depth, 4)
                    elif _m < 1.00:
                        pf_depth = max(pf_depth, 3)
                pf_depth = int(max(2, min(8, pf_depth)))
                pf_depth = _cap_pf_depth(_device_obj, datasets, pf_depth, batch_size)
                if int(pf_depth) != int(pf_depth_before):
                    batch_size = _rescale_batch(datasets, int(batch_size))
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

        def iterate(sample: Any) -> Any:
            def _one(smpl: Any) -> Any:
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
                pf_depth_before = int(pf_depth)
                if _auto_ms_candidates:
                    _m = min(_auto_ms_candidates)
                    if _m < 0.35:
                        pf_depth = max(pf_depth, 6)
                    elif _m < 0.70:
                        pf_depth = max(pf_depth, 4)
                    elif _m < 1.00:
                        pf_depth = max(pf_depth, 3)
                pf_depth = int(max(2, min(8, pf_depth)))
                pf_depth = _cap_pf_depth(_device_obj, datasets, pf_depth, batch_size)
                if int(pf_depth) != int(pf_depth_before):
                    batch_size = _rescale_batch(datasets, int(batch_size))
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

        def iterate(sample: Any) -> Any:
            def _one(smpl: Any) -> Any:
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
            pf_depth_before = int(pf_depth)
            if ms_i:
                if ms_i < 0.35:
                    pf_depth = max(pf_depth, 6)
                elif ms_i < 0.70:
                    pf_depth = max(pf_depth, 4)
                elif ms_i < 1.00:
                    pf_depth = max(pf_depth, 3)
            if int(pf_depth) != int(pf_depth_before):
                batch_size = max(
                    1, int(_stream_batch(ds, _device_obj)[0]) if len(ds) > 0 else 1
                )
        sampler_node = ds.compose(
            batch_size=int(batch_size),
            shuffle=True,
            seed=0,
            key="0",
        )
        datasets: Dict[str, Any] = {"0": ds}

        def iterate(sample: Any) -> Any:
            def _one(smpl: Any) -> Any:
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
                    pf_depth_before = int(pf_depth)
                    if _auto_ms_candidates:
                        _m = min(_auto_ms_candidates)
                        if _m < 0.35:
                            pf_depth = max(pf_depth, 6)
                        elif _m < 0.70:
                            pf_depth = max(pf_depth, 4)
                        elif _m < 1.00:
                            pf_depth = max(pf_depth, 3)
                    pf_depth = int(max(2, min(8, pf_depth)))
                    pf_depth = _cap_pf_depth(_device_obj, datasets, pf_depth, batch_size)
                    if int(pf_depth) != int(pf_depth_before):
                        batch_size = _rescale_batch(datasets, int(batch_size))
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

            def iterate(sample: Any) -> Any:
                def _one(smpl: Any) -> Any:
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
                    pf_depth_before = int(pf_depth)
                    if _auto_ms_candidates:
                        _m = min(_auto_ms_candidates)
                        if _m < 0.35:
                            pf_depth = max(pf_depth, 6)
                        elif _m < 0.70:
                            pf_depth = max(pf_depth, 4)
                        elif _m < 1.00:
                            pf_depth = max(pf_depth, 3)
                    pf_depth = int(max(2, min(8, pf_depth)))
                    pf_depth = _cap_pf_depth(_device_obj, datasets, pf_depth, batch_size)
                    if int(pf_depth) != int(pf_depth_before):
                        batch_size = _rescale_batch(datasets, int(batch_size))
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

            def iterate(sample: Any) -> Any:
                def _one(smpl: Any) -> Any:
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
                pf_depth_before = int(pf_depth)
                if ms_i:
                    if ms_i < 0.35:
                        pf_depth = max(pf_depth, 6)
                    elif ms_i < 0.70:
                        pf_depth = max(pf_depth, 4)
                    elif ms_i < 1.00:
                        pf_depth = max(pf_depth, 3)
                if int(pf_depth) != int(pf_depth_before):
                    batch_size = max(
                        1,
                        int(_stream_batch(ds, _device_obj)[0]) if len(ds) > 0 else 1,
                    )
            sampler_node = ds.compose(
                batch_size=int(batch_size),
                shuffle=False,
                seed=0,
                key="0",
            )
            datasets: Dict[str, Any] = {"0": ds}

            def iterate(sample: Any) -> Any:
                def _one(smpl: Any) -> Any:
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

    return {
        "training_loader": train_loader,
        "validation_loader": val_loader,
        "disposable": allocated,
    }

