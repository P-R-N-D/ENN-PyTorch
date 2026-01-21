# -*- coding: utf-8 -*-
from __future__ import annotations

import collections.abc
import contextlib
import importlib
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field, replace
from functools import partial
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Literal,
    Mapping,
    MutableMapping,
    NotRequired,
    Optional,
    Required,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import torch
from tensordict import MemoryMappedTensor, TensorDictBase
from torchdata.nodes import BaseNode

from ..core.concurrency import Disposable, Mutex, new_affinity
from ..core.datatypes import (
    PathLike,
    default_underflow_action,
    env_first,
    env_first_float,
    env_first_int,
    normalize_underflow_action,
)
from ..core.policies import BatchPolicy, LoaderPolicy, WorkerPolicy
from ..core.system import (
    Memory,
    accelerator,
    cuda_compute_capability,
    get_device,
    get_device_stats,
    is_accelerator_available,
    is_accelerator_timer_supported,
    is_cpu_bf16_supported,
    is_cuda_bf16_supported,
    is_float8_supported,
    is_int4_supported,
    is_int8_supported,
    is_pin_supported,
    new_accelerator_event,
    sync_accelerator,
)
from . import collate

_NODES_IMPORTED = False
_NODES_LOCK = Mutex()
_device_mem_get_info = Memory.mem_get_info
SourceType = Literal["memmap"]
TExtra = TypeVar("TExtra")
logger = logging.getLogger(__name__)

def _require_nodes() -> None:
    global _NODES_IMPORTED
    if _NODES_IMPORTED:
        return
    with _NODES_LOCK:
        if _NODES_IMPORTED:
            return
        try:
            mod = importlib.import_module(
                f"{__package__ or 'stnet.data'}.nodes"
            )
        except Exception as e:
            raise RuntimeError("Requires 'torchdata'/'tensordict'") from e
        globals().update(
            {
                k: getattr(mod, k)
                for k in [
                    "Governor",
                    "Loader",
                    "Mapper",
                    "Multiplexer",
                    "Sampler",
                ]
            }
        )
        _NODES_IMPORTED = True
def _sync_device(device: torch.device) -> None:
    sync_accelerator(device)
def _is_lazy_tensor(x: Any) -> bool:
    return isinstance(x, MemoryMappedTensor)
def _feature_size_hint(obj: Any) -> Optional[int]:
    if isinstance(obj, torch.Tensor):
        return int(obj.numel()) if obj.ndim > 0 else 1
    if isinstance(obj, (tuple, list)):
        return len(obj)
    return None
def _stack_sequence(
    seq: Sequence[Any],
    *args: Any,
    dtype: Optional[torch.dtype],
    reshape_1d: bool = False,
) -> Optional[torch.Tensor]:
    if not seq:
        return None
    t0 = collate._to_safe_tensor(seq[0], dtype)
    if t0 is None:
        return None
    if reshape_1d:
        t0 = t0.reshape(-1)
    out = torch.empty((len(seq), *t0.shape), dtype=t0.dtype, device=t0.device)
    out[0].copy_(t0)
    for i, item in enumerate(seq[1:], 1):
        ti = collate._to_safe_tensor(item, dtype)
        if ti is None:
            raise ValueError("Dataset.preprocess: missing element")
        if reshape_1d:
            ti = ti.reshape(-1)
        if ti.shape != t0.shape:
            raise ValueError(
                f"Dataset.preprocess: shape mismatch {t0.shape} vs {ti.shape}"
            )
        out[i].copy_(ti)
    return out
def _get_sample_size(
    _x_cpu: torch.Tensor, _y_cpu: Optional[torch.Tensor]
) -> int:
    x_one = _x_cpu[0]
    bx = int(x_one.numel()) * int(x_one.element_size())
    by = 0
    if _y_cpu is not None:
        y_one = _y_cpu[0]
        by = int(y_one.numel()) * int(y_one.element_size())
    return int(bx + by)
def _get_random_batch(
    _sample_bytes: int, _device: torch.device, _N: int
) -> Sequence[int]:
    if _sample_bytes <= 0 or _N <= 0:
        return [1]
    capB = 1024
    host_free = getattr(Memory, "available", lambda: None)()
    dev_free, _ = _device_mem_get_info(_device)
    effective_free = (
        min(host_free, dev_free)
        if (host_free is not None and dev_free is not None)
        else (dev_free or host_free)
    )
    if effective_free is not None:
        capB = max(
            1,
            int(
                (max(0, int(effective_free)) * 0.80)
                // max(_sample_bytes * 4, 1)
            ),
        )
    capB = max(1, min(capB, int(_N)))
    cands = sorted(
        {max(1, int(capB * f)) for f in (0.125, 0.25, 0.375, 0.5, 0.75, 1.0)}
    )
    return [c for c in cands if c <= _N]
def _h2d_counter(
    _x_cpu: torch.Tensor,
    _y_cpu: Optional[torch.Tensor],
    _device: torch.device,
    _bs: int,
    _steps: int = 10,
    _warmup: int = 2,
) -> float:
    N = int(_x_cpu.shape[0])
    bs = max(1, min(int(_bs), N))
    dev_t = str(getattr(_device, "type", "cpu") or "cpu")
    max_probe_bytes = int(
        env_first_int(
            (
                "STNET_H2D_PROBE_MAX_BYTES",
                "STNET_H2D_PROBE_BYTES",
                "STNET_H2D_COUNTER_MAX_BYTES",
            ),
            default=256 * 1024 * 1024,
        )
        or 0
    )
    if max_probe_bytes > 0:
        sample_bytes = int(_get_sample_size(_x_cpu, _y_cpu) or 0)
        if sample_bytes > 0:
            bs_cap = max(1, int(max_probe_bytes) // max(1, sample_bytes))
            bs = max(1, min(bs, bs_cap))
    pin_supported = bool(is_pin_supported(dev_t))
    max_pin_bytes = int(
        env_first_int(
            (
                "STNET_H2D_PROBE_MAX_PIN_BYTES",
                "STNET_H2D_PROBE_PIN_MAX_BYTES",
            ),
            default=max_probe_bytes,
        )
        or 0
    )
    if max_pin_bytes < 0:
        max_pin_bytes = 0
    ev0 = ev1 = None
    if is_accelerator_timer_supported(dev_t):
        ev0 = new_accelerator_event(_device, enable_timing=True)
        ev1 = new_accelerator_event(_device, enable_timing=True)
    times: list[float] = []
    for s in range(int(_steps) + int(_warmup)):
        start = random.randint(0, N - bs) if N > bs else 0
        xb = _x_cpu[start : start + bs]
        yb = _y_cpu[start : start + bs] if _y_cpu is not None else None
        pin_batch = bool(pin_supported)
        if pin_batch and max_pin_bytes > 0:
            bbytes = int(xb.element_size() * xb.numel())
            if yb is not None:
                bbytes += int(yb.element_size() * yb.numel())
            if bbytes > int(max_pin_bytes):
                pin_batch = False
        if pin_batch:
            with contextlib.suppress(Exception):
                xb = xb if (hasattr(xb, "is_pinned") and xb.is_pinned()) else xb.pin_memory()
            if yb is not None:
                with contextlib.suppress(Exception):
                    yb = yb if (hasattr(yb, "is_pinned") and yb.is_pinned()) else yb.pin_memory()
        _sync_device(_device)
        if ev0 is not None and ev1 is not None:
            with accelerator(_device):
                ev0.record()
                xb.to(_device, non_blocking=bool(pin_batch))
                yb.to(_device, non_blocking=bool(pin_batch)) if yb is not None else None
                ev1.record()
            _sync_device(_device)
            ms = float(ev0.elapsed_time(ev1))
        else:
            tns0 = time.perf_counter_ns()
            xb.to(_device, non_blocking=bool(pin_batch))
            yb.to(_device, non_blocking=bool(pin_batch)) if yb is not None else None
            _sync_device(_device)
            ms = (time.perf_counter_ns() - tns0) / 1e6
        if s >= int(_warmup):
            times.append(ms)
    return float(sorted(times)[len(times) // 2]) if times else 0.0
def _set_batch_interval(
    _ds: "Sampler",
    _dev: torch.device,
    _tmin_ms: float = 0.8,
    _tmax_ms: float = 3.0,
    *args: Any,
    prefetch_factor: int = 2,
    num_workers: int = 0,
    prebatch: int = 1,
    worker_policy: Optional[WorkerPolicy] = None,
) -> Tuple[int, float]:
    if len(_ds) <= 0:
        return (1, 0.0)
    sbytes_cached = int(getattr(_ds, "_S_sample_bytes", 0) or 0)
    probe = _ds.get(0, min(8, len(_ds)))
    x_cpu, y_cpu = collate.get_row(probe, labels_required=False)
    if not isinstance(x_cpu, torch.Tensor):
        x_cpu = torch.as_tensor(x_cpu)
    if y_cpu is not None and not isinstance(y_cpu, torch.Tensor):
        y_cpu = torch.as_tensor(y_cpu)
    sbytes = (
        sbytes_cached if sbytes_cached > 0 else _get_sample_size(x_cpu, y_cpu)
    )
    if sbytes > 0 and sbytes_cached <= 0:
        with contextlib.suppress(Exception):
            setattr(_ds, "_S_sample_bytes", int(sbytes))
    if sbytes <= 0:
        return (max(1, min(256, len(_ds))), 0.0)
    B_cap = 1 << 16
    max_batch_env = int(
        env_first_int(("STNET_MAX_BATCH_SIZE", "STNET_MAX_BATCH"), default=0) or 0
    )
    per_sample = int(getattr(_ds, "_per_sample_mem_bytes", 0) or 0)
    if per_sample <= 0:
        per_sample = int(
            env_first_int(
                (
                    "STNET_PER_SAMPLE_MEM_BYTES",
                    "STNET_DEVICE_BYTES_PER_SAMPLE",
                ),
                default=0,
            )
            or 0
        )
    if per_sample <= 0:
        per_sample = sbytes
    if worker_policy is not None:
        _wp = worker_policy
        _max_conc = max(1, int(getattr(_wp, "max_concurrency", 1) or 1))
        _streams = max(1, int(getattr(_wp, "h2d_streams", 1) or 1))
        _lws = max(1, int(getattr(_wp, "local_world_size", 1) or 1))
    else:
        try:
            _wp = WorkerPolicy.optimize()
            _max_conc = max(1, int(getattr(_wp, "max_concurrency", 1) or 1))
            _streams = max(1, int(getattr(_wp, "h2d_streams", 1) or 1))
            _lws = max(1, int(getattr(_wp, "local_world_size", 1) or 1))
        except Exception:
            _wp = None
            _max_conc, _streams, _lws = (1, 1, 1)
    budget_slack = max(
        1.0,
        min(
            4.0,
            float(
                env_first_float(("STNET_BUDGET_SLACK",), default=1.25) or 1.25
            ),
        ),
    )

    def _get_bytes(name):
        return int(env_first_int((name,), default=0) or 0)

    def _get_opt_bytes(name):
        return val if (val := _get_bytes(name)) > 0 else None

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
        device_margin=float(
            env_first_float(("STNET_DEVICE_MARGIN",), default=0.90) or 0.90
        ),
        host_margin=float(
            env_first_float(("STNET_HOST_MARGIN",), default=0.10) or 0.10
        ),
        device_budget_ratio=float(
            env_first_float(("STNET_DEVICE_BUDGET_RATIO",), default=1.0) or 1.0
        ),
        device_budget_min_bytes=_get_bytes("STNET_DEVICE_BUDGET_MIN_BYTES"),
        device_budget_max_bytes=_get_opt_bytes(
            "STNET_DEVICE_BUDGET_MAX_BYTES"
        ),
        host_budget_ratio=float(
            env_first_float(("STNET_HOST_BUDGET_RATIO",), default=1.0) or 1.0
        ),
        host_budget_min_bytes=_get_bytes("STNET_HOST_BUDGET_MIN_BYTES"),
        host_budget_max_bytes=_get_opt_bytes("STNET_HOST_BUDGET_MAX_BYTES"),
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
    probe_bs_cache: Optional[int] = None
    med_probe_cache: Optional[float] = None
    b_init_hint: Optional[int] = None
    if (
        tpl.device_budget_max_bytes is None
        or tpl.host_budget_max_bytes is None
    ) and int(tpl.sample_bytes or 0) > 0:
        try:
            inflight = int(tpl.host_inflight_batches_per_proc())
            lw = max(1, int(getattr(tpl, "local_world_size", 1) or 1))

            target_ms = float(
                max(
                    float(_tmin_ms),
                    min(
                        float(_tmax_ms),
                        0.5 * (float(_tmin_ms) + float(_tmax_ms)),
                    ),
                )
            )
            probe_bs = max(1, min(int(B_cap), 64))
            if int(max_batch_env) > 0:
                probe_bs = max(1, min(int(probe_bs), int(max_batch_env)))
            probe_max_bytes = int(
                env_first_int(
                    (
                        "STNET_H2D_PROBE_MAX_BYTES",
                        "STNET_H2D_PROBE_BYTES",
                        "STNET_H2D_COUNTER_MAX_BYTES",
                    ),
                    default=256 * 1024 * 1024,
                )
                or 0
            )
            if probe_max_bytes > 0 and int(sbytes) > 0:
                probe_bs = max(
                    1,
                    min(int(probe_bs), int(probe_max_bytes) // max(1, int(sbytes))),
                )
            med_probe = 0.0
            with contextlib.suppress(Exception):
                med_probe = float(
                    _h2d_counter(
                        x_cpu, y_cpu, _dev, probe_bs, _steps=4, _warmup=1
                    )
                )
            if (
                isinstance(med_probe, (float, int))
                and math.isfinite(float(med_probe))
                and float(med_probe) > 0.0
            ):
                bs_est = int(
                    math.ceil((target_ms * float(probe_bs)) / float(med_probe))
                )
                target_batch_samples = max(1, min(int(B_cap), bs_est))
                probe_bs_cache = int(probe_bs)
                med_probe_cache = float(med_probe)
            else:
                target_batch_samples = max(
                    1,
                    min(
                        int(B_cap),
                        int(
                            (64 * 1024 * 1024) // max(1, int(tpl.sample_bytes))
                        ),
                    ),
                )
            b_init_hint = int(target_batch_samples)
            new_dev_cap: Optional[int] = tpl.device_budget_max_bytes
            new_host_cap: Optional[int] = tpl.host_budget_max_bytes
            if new_dev_cap is None:
                base_dev = int(tpl.sample_bytes) * int(target_batch_samples)
                cap_dev = int(float(base_dev) * float(budget_slack))
                if dev_total is not None and int(dev_total) > 0:
                    cap_dev = min(int(cap_dev), int(dev_total))
                cap_dev = max(0, int(cap_dev))
                new_dev_cap = None if cap_dev <= 0 else cap_dev
            if new_host_cap is None and int(tpl.host_sample_bytes or 0) > 0:
                base_host = (
                    int(tpl.host_sample_bytes)
                    * max(1, inflight)
                    * max(1, lw)
                    * int(target_batch_samples)
                )
                cap_host = int(float(base_host) * float(budget_slack))
                if host_total is not None and int(host_total) > 0:
                    cap_host = min(int(cap_host), int(host_total))
                cap_host = max(0, int(cap_host))
                new_host_cap = None if cap_host <= 0 else cap_host
            if (new_dev_cap != tpl.device_budget_max_bytes) or (
                new_host_cap != tpl.host_budget_max_bytes
            ):
                tpl = replace(
                    tpl,
                    device_budget_max_bytes=new_dev_cap,
                    host_budget_max_bytes=new_host_cap,
                )
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
    B_cap = max(1, min(int(B_cap), len(_ds)))
    if int(max_batch_env) > 0:
        B_cap = max(1, min(B_cap, int(max_batch_env)))
    with contextlib.suppress(Exception):
        setattr(_ds, "_S_B_cap", int(B_cap))
    candidates = _get_random_batch(sbytes, _dev, len(_ds))
    if candidates:
        B = min(candidates[-1], B_cap)
    else:
        B = min(64, B_cap)
    if b_init_hint is not None:
        try:
            B_hint = max(1, min(int(B_cap), int(b_init_hint)))
            if candidates:
                cands = [
                    int(c)
                    for c in candidates
                    if isinstance(c, int) and c > 0 and int(c) <= int(B_cap)
                ]
                if cands:
                    le = [c for c in cands if int(c) <= int(B_hint)]
                    if le:
                        B = int(max(le))
                    else:
                        B = int(min(cands))
                else:
                    B = int(B_hint)
            else:
                B = int(B_hint)
        except Exception:
            pass
    if (
        probe_bs_cache is not None
        and med_probe_cache is not None
        and int(B) == int(probe_bs_cache)
    ):
        med = float(med_probe_cache)
    else:
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
def _is_source(obj: Any) -> bool:
    if not isinstance(obj, Mapping):
        return False
    if "path" not in obj:
        return False
    if "format" not in obj and "kind" not in obj:
        return False
    p = obj.get("path")
    try:
        return bool(os.fspath(p))
    except Exception:
        return False
def _merge_opt(v1, v2, op):
    if v1 is None:
        return v2
    if v2 is None:
        return v1
    return op(v1, v2)
def _fetch_stream_batch(
    ds: "Sampler",
    device: torch.device,
    *args: Any,
    pf: int,
    io_workers: int,
    prebatch: int,
    worker_policy: WorkerPolicy,
) -> Tuple[int, float]:
    try:
        return _set_batch_interval(
            ds,
            device,
            prefetch_factor=int(pf),
            num_workers=int(io_workers),
            prebatch=int(prebatch),
            worker_policy=worker_policy,
        )
    except Exception:
        return (0, 0.0)
def _fetch_auto_batch_size(
    datasets: Mapping[str, "Sampler"],
    device: torch.device,
    *args: Any,
    pf: int,
    io_workers: int,
    prebatch: int,
    worker_policy: WorkerPolicy,
    fallback: int,
) -> int:
    candidates: List[int] = []
    for ds in datasets.values():
        b_i, _ = _fetch_stream_batch(
            ds,
            device,
            pf=int(pf),
            io_workers=int(io_workers),
            prebatch=int(prebatch),
            worker_policy=worker_policy,
        )
        if b_i > 0:
            candidates.append(int(b_i))
    if not candidates:
        return max(1, int(fallback))
    cand_mean = int(sum(candidates) // max(1, len(candidates)))
    cand_max = int(max(candidates))
    return int(max(1, min(cand_max, cand_mean)))
def _fetch_cap_pf_depth(
    datasets: Mapping[str, "Sampler"],
    device: torch.device,
    *args: Any,
    pf: int,
    bs: int,
    loader_policy: "LoaderPolicy",
    io_workers: int,
    prebatch: int,
    memory_budget_fraction: float = 0.15,
) -> int:
    try:
        host_avail = int(Memory.available())
        if host_avail <= 0:
            return int(pf)
        dev_free, _ = _device_mem_get_info(device)
        effective_avail = (
            min(host_avail, dev_free) if dev_free is not None else host_avail
        )
        budget = int(effective_avail * float(memory_budget_fraction))
        if budget <= 0 or bs <= 0:
            return int(pf)
        sbytes_max = 0
        for ds in datasets.values():
            if len(ds) <= 0:
                continue
            cached = int(getattr(ds, "_S_sample_bytes", 0) or 0)
            if cached > 0:
                sbytes_max = max(sbytes_max, cached)
                continue
            probe = ds.get(0, min(8, len(ds)))
            x = probe.get("X")
            y = probe.get("Y") if isinstance(probe, Mapping) else None
            if x is None:
                continue
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x)
            if y is not None and not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y)
            sb = _get_sample_size(x, y)
            if sb > 0:
                with contextlib.suppress(Exception):
                    setattr(ds, "_S_sample_bytes", int(sb))
            sbytes_max = max(sbytes_max, sb)
        if sbytes_max <= 0:
            return int(pf)
        bytes_per_batch = int(sbytes_max) * int(bs)
        if bytes_per_batch <= 0:
            return int(pf)
        pf_cap = max(1, int(budget // max(1, bytes_per_batch)))
        with contextlib.suppress(Exception):
            hard = int(loader_policy.hard_inflight_batches(device))
            soft_cap = max(
                1, int(hard * max(1, int(loader_policy.soft_cap_multiplier)))
            )
            pb = max(1, int(prebatch))
            workers = max(1, int(io_workers) if int(io_workers) > 0 else 1)
            inflight_pf_cap = max(1, int((soft_cap - pb) // max(1, workers)))
            pf_cap = min(int(pf_cap), int(inflight_pf_cap))
        return int(max(1, min(int(pf), int(pf_cap), 8)))
    except Exception:
        return int(pf)
def _fetch_iterate_sample(
    sample: Any,
    *args: Any,
    datasets: Mapping[str, "Sampler"],
    collate: Callable[[Any], Any],
) -> Any:
    if (
        isinstance(sample, (list, tuple))
        and sample
        and all(isinstance(elem, tuple) and len(elem) == 2 for elem in sample)
    ):
        batches: list[Any] = []
        for k, span in sample:
            ds = datasets.get(str(k))
            if ds is None:
                continue
            s, e = int(span[0]), int(span[1])
            batch = ds.get(s, e)
            if batch is not None:
                batches.append(batch)
        if not batches:
            return None
        if len(batches) == 1:
            return collate(batches[0])
        flattened = _fetch_merge_batches(batches)
        return collate(flattened)
    if isinstance(sample, tuple) and len(sample) == 2:
        k, span = sample
        ds = datasets.get(str(k))
        if ds is None:
            return None
        s, e = int(span[0]), int(span[1])
        batch = ds.get(s, e)
        return collate(batch) if batch is not None else None
    return collate(sample)
def _fetch_merge_batches(batches: Sequence[Any]) -> Any:
    if TensorDictBase is not None and all(
        isinstance(b, TensorDictBase) for b in batches
    ):
        with contextlib.suppress(Exception):
            return torch.cat(list(batches), dim=0)
    if all(isinstance(b, Mapping) for b in batches):
        merged: dict[str, Any] = {}
        keys = set(chain.from_iterable(b.keys() for b in batches))
        for key in keys:
            vals = [b.get(key) for b in batches if key in b]
            if not vals:
                continue
            if all(isinstance(v, torch.Tensor) for v in vals):
                merged[key] = torch.cat(cast(list[torch.Tensor], vals), dim=0)
                continue
            if key == "row_ids":
                tensors = []
                for v in vals:
                    if v is None:
                        continue
                    tensors.append(
                        v
                        if isinstance(v, torch.Tensor)
                        else torch.as_tensor(v)
                    )
                if tensors:
                    merged[key] = torch.cat(tensors, dim=0)
                continue
            if all(isinstance(v, (list, tuple)) for v in vals):
                merged[key] = list(
                    chain.from_iterable(
                        v if isinstance(v, list) else list(v) for v in vals
                    )
                )
                continue
            with contextlib.suppress(Exception):
                tensors = [
                    v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
                    for v in vals
                ]
                merged[key] = torch.cat(tensors, dim=0)
                continue
            merged[key] = list(
                chain.from_iterable(
                    v if isinstance(v, (list, tuple)) else [v] for v in vals
                )
            )
        return merged
    return list(
        chain.from_iterable(
            b if isinstance(b, (list, tuple)) else [b] for b in batches
        )
    )
def _fetch_normalize_sources(sources: Any) -> Dict[str, Source]:
    if isinstance(sources, Mapping) and (not _is_source(sources)):
        out: Dict[str, Source] = {}
        for k, v in sources.items():
            kk = str(k)
            if kk in out:
                raise ValueError(f"duplicate source key after str(): {kk!r}")
            out[kk] = v
        return out
    if isinstance(sources, (list, tuple)):
        return {str(i): v for i, v in enumerate(sources)}
    return {"0": sources}
def _fetch_build_datasets(
    specs: Mapping[str, Source],
    *args: Any,
    split: str,
    val_frac: float,
    sampler_scale: "Governor",
    allocated: "Disposable",
    collect_epochables: bool = False,
    epochables: Optional[list[Any]] = None,
) -> Dict[str, "Sampler"]:
    out: Dict[str, "Sampler"] = {}
    for k, spec in specs.items():
        ds = new_dataset(
            spec, split=split, val_frac=val_frac, sampler_scale=sampler_scale
        )
        allocated.add(ds)
        out[str(k)] = ds
        if collect_epochables and epochables is not None:
            epochables.append(ds)
    return out
def _fetch_build_sampler_nodes(
    datasets: Mapping[str, "Sampler"],
    *args: Any,
    bs: int,
    shuffle: bool,
    seed: int,
) -> Tuple[Dict[str, BaseNode], Dict[str, int]]:
    nodes: Dict[str, BaseNode] = {}
    lengths: Dict[str, int] = {}
    for k, ds in datasets.items():
        sn = ds.compose(
            batch_size=int(bs),
            shuffle=bool(shuffle),
            seed=int(seed),
            key=str(k),
        )
        if len(ds) > 0:
            nodes[str(k)] = sn
            lengths[str(k)] = int(len(ds))
    return nodes, lengths

def iter_dataset(
    data: object,
) -> tuple[list[tuple[str, object]], object | None]:
    if isinstance(data, TensorDictBase):
        return ([("0", data)], None)

    if (
        isinstance(data, collections.abc.Mapping)
        and data
        and all(
            (
                isinstance(v, (TensorDictBase, collections.abc.Mapping))
                for v in data.values()
            )
        )
    ):
        manifest: dict[str, str] = {}
        items: list[tuple[str, object]] = []
        for k, d in data.items():
            key = str(k)
            items.append((key, d))
            manifest[key] = key
        return (items, manifest)
    if (
        isinstance(data, Sequence)
        and data
        and all(
            (
                isinstance(d, (TensorDictBase, collections.abc.Mapping))
                for d in data
            )
        )
    ):
        manifest_list: list[str] = []
        items2: list[tuple[str, object]] = []
        for i, d in enumerate(data):
            key = str(i)
            items2.append((key, d))
            manifest_list.append(key)
        return (items2, manifest_list)
    return ([("0", data)], None)
def new_dataset(
    source: Source,
    *args: Any,
    split: str = "train",
    val_frac: float = 0.0,
    sampler_scale: Optional["Governor"] = None,
    **kwargs: Any,
) -> "Sampler":
    _require_nodes()
    if not isinstance(source, Mapping):
        raise TypeError(
            f"dataset expects a Source mapping, got {type(source)}"
        )
    fmt = source.get("format")
    if fmt is None:
        fmt = source.get("kind")
    if fmt is None:
        raise ValueError("Source['format'] or Source['kind'] must be provided")
    fmt = str(fmt)
    if fmt != "memmap":
        raise ValueError(f"Unsupported source format: {fmt!r}")
    path = os.fspath(source.get("path", "")).strip()
    if not path:
        raise ValueError("Source['path'] must be provided")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"memmap directory not found: {path!r}")
    sp = str(split or "train")
    if sp not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got: {sp!r}")
    vf = float(val_frac)
    if not (0.0 <= vf <= 1.0):
        raise ValueError(f"val_frac must be in [0,1], got: {vf}")
    return Sampler(path, split=sp, val_frac=vf, sampler_scale=sampler_scale)
def compose(
    node_or_nodes: Union[BaseNode, Sequence[BaseNode], Mapping[str, BaseNode]],
    *args: Any,
    device: Union[str, torch.device],
    map_fn: Callable[[Any], Any],
    prefetch_factor: int,
    non_blocking_copy: bool,
    io_workers: int,
    prebatch: int,
    weights: Optional[Mapping[str, float] | Sequence[float]] = None,
    seed: int = 0,
    epochables: Optional[list[Any]] = None,
    **kwargs: Any,
) -> Tuple[BaseNode, BaseNode, BaseNode]:
    _require_nodes()
    device_obj = (
        torch.device(device)
        if not isinstance(device, torch.device)
        else device
    )
    with contextlib.suppress(Exception):
        new_affinity(io_workers=io_workers)
    sampler = Multiplexer(
        stop_criteria="ALL_DATASETS_EXHAUSTED", seed=int(seed), weights=weights
    )
    source = sampler.compose(node_or_nodes)
    if epochables is not None and getattr(sampler, "_node", None) is not None:
        with contextlib.suppress(Exception):
            epochables.insert(0, sampler)
    mapper = Mapper(
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
    sources: Union[Source, Sequence[Source], Mapping[str, Source]],
    device: Union[str, torch.device],
    *args: Any,
    batch_size: Optional[int] = None,
    flatten_features: bool = True,
    labels_dtype: Optional[torch.dtype] = torch.long,
    sanitize: bool = False,
    non_blocking_copy: bool = True,
    train_shuffle: bool = True,
    train_weights: Optional[Mapping[str, float] | Sequence[float]] = None,
    val_weights: Optional[Mapping[str, float] | Sequence[float]] = None,
    val_frac: float = 0.0,
    loader_policy: Optional["LoaderPolicy"] = None,
    worker_policy: Optional[WorkerPolicy] = None,
    sampler_scale: Optional["Governor"] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    _require_nodes()
    device_obj = (
        torch.device(device)
        if not isinstance(device, torch.device)
        else device
    )
    lp = (
        loader_policy
        if isinstance(loader_policy, LoaderPolicy)
        else LoaderPolicy()
    )
    wp = (
        worker_policy
        if isinstance(worker_policy, WorkerPolicy)
        else WorkerPolicy.optimize()
    )
    wp.set_thread_setting()
    io_workers = int(getattr(wp, "num_workers", 0) or 0)
    prebatch = int(getattr(wp, "prebatch", 1) or 1)
    pf_depth_fixed = max(
        1, min(8, int(getattr(wp, "prefetch_factor", 1) or 1))
    )
    collate_fn = collate.Collator(
        flatten_features=bool(flatten_features),
        labels_dtype=labels_dtype,
        sanitize=bool(sanitize),
    )
    allocated = Disposable()
    scale_ctl = sampler_scale if sampler_scale is not None else Governor()
    train_epochables: List[Any] = []
    specs = _fetch_normalize_sources(sources)
    spec_keys = list(specs.keys())

    def _create_loader_stage(
        split, shuffle, weights, collect_epochables=False, epochables=None
    ):
        datasets = _fetch_build_datasets(
            specs,
            split=split,
            val_frac=float(val_frac),
            sampler_scale=scale_ctl,
            allocated=allocated,
            collect_epochables=collect_epochables,
            epochables=epochables,
        )
        if not datasets:
            return None
        bs = int(batch_size) if batch_size and int(batch_size) > 0 else 0
        if bs <= 0:
            bs = _fetch_auto_batch_size(
                datasets,
                device_obj,
                pf=pf_depth_fixed,
                io_workers=io_workers,
                prebatch=prebatch,
                worker_policy=wp,
                fallback=1,
            )
        pf = _fetch_cap_pf_depth(
            datasets,
            device_obj,
            pf=pf_depth_fixed,
            bs=bs,
            loader_policy=lp,
            io_workers=io_workers,
            prebatch=prebatch,
        )
        pf = int(max(1, min(pf, pf_depth_fixed, 8)))
        if not (batch_size and int(batch_size) > 0) and pf != pf_depth_fixed:
            bs = _fetch_auto_batch_size(
                datasets,
                device_obj,
                pf=pf,
                io_workers=io_workers,
                prebatch=prebatch,
                worker_policy=wp,
                fallback=bs,
            )

        nodes, lengths = _fetch_build_sampler_nodes(
            datasets, bs=bs, shuffle=shuffle, seed=seed
        )
        if not nodes:
            raise RuntimeError(f"No non-empty sources for split={split}")

        weights_out = None
        if len(nodes) > 1 and weights:
            w_map = (
                {
                    str(k): float(v)
                    for k, v in dict(weights).items()
                    if str(k) in nodes
                }
                if isinstance(weights, Mapping)
                else None
            )
            if w_map:
                if not any(v > 0.0 for v in w_map.values()):
                    raise ValueError("Weights must be > 0")
                weights_out = w_map

        iter_fn = partial(
            _fetch_iterate_sample, datasets=datasets, collate=collate_fn
        )
        _, mapped, _ = compose(
            nodes,
            device=device_obj,
            map_fn=iter_fn,
            prefetch_factor=pf,
            non_blocking_copy=non_blocking_copy,
            io_workers=io_workers,
            prebatch=prebatch,
            weights=weights_out,
            epochables=epochables if collect_epochables else None,
            seed=seed,
        )
        return Loader.compose(
            mapped,
            device=device_obj,
            prefetch_factor=pf,
            non_blocking=non_blocking_copy,
            length=sum(lengths.values()) if lengths else None,
        )

    train_loader = _create_loader_stage(
        "train",
        train_shuffle,
        train_weights,
        collect_epochables=True,
        epochables=train_epochables,
    )
    val_loader = (
        _create_loader_stage("val", False, val_weights)
        if float(val_frac) > 0.0
        else None
    )
    with contextlib.suppress(Exception):
        if train_loader is not None:
            setattr(train_loader, "_stnet_sampler_scale", scale_ctl)
            setattr(train_loader, "_stnet_epochables", list(train_epochables))
        if val_loader is not None:
            setattr(val_loader, "_stnet_sampler_scale", scale_ctl)
    return {
        "training_loader": train_loader,
        "validation_loader": val_loader,
        "disposable": allocated,
        "sampler_scale": scale_ctl,
    }
def preload_memmap(
    data: Mapping[str, Any],
    *args: Any,
    memmap_dir: PathLike,
    val_frac: float = 0.0,
    shuffle: bool = False,
    seed: int | None = None,
    underflow_action: str | None = None,
    chunk_size: int = 4096,
    allow_missing_labels: bool = False,
    features_only: bool = False,
    default_label_shape: Tuple[int, ...] | None = None,
) -> None:
    del args
    if not isinstance(data, Mapping):
        raise TypeError(
            "preload_memmap expects a Mapping with at least 'features'"
        )
    if "features" not in data:
        raise ValueError("preload_memmap expects 'features'")

    raw_X = data["features"]
    raw_Y = data.get("labels")

    def _len0(obj: Any) -> int:
        if isinstance(obj, torch.Tensor):
            return (
                int(obj.shape[0])
                if int(getattr(obj, "ndim", 0) or 0) > 0
                else 1
            )
        try:
            return int(len(obj))
        except Exception:
            return 0

    count = _len0(raw_X)
    if count <= 0:
        raise ValueError("cannot create memmap with zero samples")

    if not bool(features_only):
        if raw_Y is None:
            if not bool(allow_missing_labels):
                raise ValueError(
                    "preload_memmap expects 'labels' unless allow_missing_labels=True"
                )
        else:
            if _len0(raw_Y) != int(count):
                raise ValueError(
                    "features and labels must have the same length"
                )

    ua = normalize_underflow_action(
        underflow_action, default=default_underflow_action()
    )
    ds = Dataset.for_device(
        "cpu", feature_dtype=torch.float64, label_float_dtype=torch.float64
    )
    ds.underflow_action = ua

    from . import collate
    from .collate import _BatchIndexGetter, _BatchSliceGetter

    get_batch = _BatchSliceGetter(
        raw_X, raw_Y, features_only=bool(features_only)
    )
    get_by_indices = (
        _BatchIndexGetter(raw_X, raw_Y, features_only=bool(features_only))
        if bool(shuffle)
        else None
    )
    collate.stream_memmap(
        ds=ds,
        out_dir=os.fspath(memmap_dir),
        count=int(count),
        get_batch=get_batch,
        get_by_indices=get_by_indices,
        val_frac=float(val_frac),
        seed_value=int(seed) if seed is not None else None,
        underflow_action=str(ua),
        shuffle=bool(shuffle),
        allow_missing_labels=bool(allow_missing_labels),
        features_only=bool(features_only),
        default_label_shape=(
            tuple(int(x) for x in default_label_shape)
            if default_label_shape is not None
            else None
        ),
        chunk_size=int(chunk_size),
    )
    return None

class Source(TypedDict):
    path: Required[str]
    format: NotRequired[SourceType]
    kind: NotRequired[SourceType]
@dataclass
class Session:
    sources: Any
    device: torch.device | str
    val_frac: float = 0.0
    non_blocking_copy: bool = True
    labels_dtype: Optional[torch.dtype] = None
    sanitize: bool = True
    flatten_features: bool = True
    train_shuffle: bool = True
    seed: int = 0
    train_weights: Optional[Mapping[str, float] | Sequence[float]] = None
    val_weights: Optional[Mapping[str, float] | Sequence[float]] = None
    worker_policy: Optional[WorkerPolicy] = None
    loader_policy: LoaderPolicy = field(default_factory=LoaderPolicy)
    raw_training_loader: Any = None
    raw_validation_loader: Any = None
    training_loader: Any = None
    validation_loader: Any = None
    disposable: Any = None
    sampler_scale: Optional["Governor"] = None
    _opened: bool = False

    def __enter__(self) -> "Session":
        return self.open()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def open(
        self,
        *args: Any,
        train_state: Optional[Dict[str, Any]] = None,
        val_state: Optional[Dict[str, Any]] = None,
    ) -> "Session":
        _require_nodes()
        if self.sampler_scale is None:
            self.sampler_scale = Governor()
        dev = (
            torch.device(self.device)
            if not isinstance(self.device, torch.device)
            else self.device
        )
        wp = self.worker_policy or WorkerPolicy.optimize()
        wp = self.loader_policy.apply_soft_limits(wp, dev)
        self.worker_policy = wp
        dl = fetch(
            sources=self.sources,
            device=dev,
            val_frac=float(self.val_frac),
            non_blocking_copy=bool(self.non_blocking_copy),
            labels_dtype=self.labels_dtype,
            sanitize=bool(self.sanitize),
            flatten_features=bool(self.flatten_features),
            train_shuffle=bool(self.train_shuffle),
            seed=int(self.seed),
            train_weights=self.train_weights,
            val_weights=self.val_weights,
            worker_policy=wp,
            sampler_scale=self.sampler_scale,
            loader_policy=self.loader_policy,
        )
        train_loader = dl.get("training_loader")
        val_loader = dl.get("validation_loader")
        self.disposable = dl.get("disposable")
        self.raw_training_loader = train_loader
        self.raw_validation_loader = val_loader
        if (
            train_state
            and train_loader is not None
            and hasattr(train_loader, "load_state_dict")
        ):
            with contextlib.suppress(Exception):
                train_loader.load_state_dict(train_state)
        if (
            val_state
            and val_loader is not None
            and hasattr(val_loader, "load_state_dict")
        ):
            with contextlib.suppress(Exception):
                val_loader.load_state_dict(val_state)
        self.training_loader = (
            self.loader_policy.wrap_input(
                train_loader, dev, name="train-input"
            )
            if train_loader is not None
            else None
        )
        self.validation_loader = (
            self.loader_policy.wrap_input(val_loader, dev, name="val-input")
            if val_loader is not None
            else None
        )
        with contextlib.suppress(Exception):
            if self.training_loader is not None:
                setattr(
                    self.training_loader,
                    "_stnet_sampler_scale",
                    self.sampler_scale,
                )
            if self.validation_loader is not None:
                setattr(
                    self.validation_loader,
                    "_stnet_sampler_scale",
                    self.sampler_scale,
                )
        with contextlib.suppress(Exception):
            if (
                self.raw_training_loader is not None
                and self.training_loader is not None
            ):
                epochables = getattr(
                    self.raw_training_loader, "_stnet_epochables", None
                )
                if epochables is not None:
                    setattr(
                        self.training_loader, "_stnet_epochables", epochables
                    )
        self._opened = True
        return self

    def close(self) -> None:
        if not self._opened:
            return
        keep = getattr(self, "disposable", None)
        if keep is not None:
            with contextlib.suppress(Exception):
                keep.cleanup()
        self._opened = False
@dataclass
class Dataset(Generic[TExtra]):
    device: torch.device
    device_type: str = field(init=False, default="cpu")
    cuda_cc: Optional[Tuple[int, int]] = field(init=False, default=None)
    float_dtypes: Tuple[torch.dtype, ...] = field(default_factory=tuple)
    int_dtypes: Tuple[torch.dtype, ...] = field(default_factory=tuple)
    float8_dtypes: Tuple[torch.dtype, ...] = field(default_factory=tuple)
    int_quant_bits: Optional[int] = None
    input_data: Any = None
    output_data: Any = None
    feature_dtype: torch.dtype = torch.float32
    label_float_dtype: torch.dtype = torch.float32
    has_scale: bool = False
    has_nonfinite: bool = False
    scale_max_abs: Optional[float] = None
    scale_min_value: Optional[float | int] = None
    scale_max_value: Optional[float | int] = None
    scale_min_positive: Optional[float] = None
    scale_is_integral: Optional[bool] = None
    is_negotiable: Optional[bool] = None
    underflow_action: str = field(default_factory=default_underflow_action)
    stats: MutableMapping[str, torch.Tensor] = field(default_factory=dict)
    extra: Dict[str, TExtra] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._refresh_device_info()
        self._refresh_dtypes_from_env()
        self._refresh_quant_from_env()

    def _refresh_device_info(self) -> None:
        stats = get_device_stats(self.device)
        self.device = stats.device
        self.device_type = stats.device_type
        self.cuda_cc = stats.cuda_cc

    @staticmethod
    def _resolve_device(
        device: Optional[Union[torch.device, str]],
    ) -> torch.device:
        if device is not None:
            return torch.device(device)
        with contextlib.suppress(Exception):
            return get_device()
        return torch.device("cpu")

    @staticmethod
    def _parse_dtypes_env(value: str) -> Tuple[torch.dtype, ...]:
        entries: list[torch.dtype] = []
        for token in str(value).split(","):
            name = token.strip()
            if not name:
                continue
            dtype = getattr(torch, name, None)
            if isinstance(dtype, torch.dtype):
                entries.append(dtype)
        return tuple(entries)

    def _refresh_dtypes_from_env(self) -> None:
        float_env = env_first(
            ("STNET_DATA_FLOAT_DTYPES", "STNET_FLOAT_DTYPES")
        )
        if float_env:
            parsed = self._parse_dtypes_env(float_env)
            if parsed:
                self.float_dtypes = parsed
        int_env = env_first(("STNET_DATA_INT_DTYPES", "STNET_INT_DTYPES"))
        if int_env:
            parsed = self._parse_dtypes_env(int_env)
            if parsed:
                self.int_dtypes = parsed

    def _refresh_quant_from_env(self) -> None:
        bits = env_first_int(
            ("STNET_DATA_INT_QUANT_BITS", "STNET_INT_QUANT_BITS"), default=0
        )
        if bits > 0:
            self.int_quant_bits = int(bits)

    @property
    def device_stats(self):
        return get_device_stats(self.device)

    def ensure_device_info(self) -> "Dataset[TExtra]":
        self._refresh_device_info()
        return self

    def refresh(self) -> "Dataset[TExtra]":
        self._refresh_device_info()
        self._refresh_dtypes_from_env()
        self._refresh_quant_from_env()
        dev_stats = get_device_stats(self.device)
        if not self.float_dtypes:
            self.float_dtypes = tuple(getattr(dev_stats, "float_dtypes", ()))
        if not self.int_dtypes:
            self.int_dtypes = tuple(getattr(dev_stats, "int_dtypes", ()))
        if self.int_quant_bits is None:
            self.int_quant_bits = int(getattr(dev_stats, "int_quant_bits", 8))
        if not self.float_dtypes:
            floats: list[torch.dtype] = [torch.float32]
            if self.device_type == "cuda" and is_accelerator_available("cuda"):
                floats.insert(0, torch.float16)
                if self.is_cuda_bf16_supported(self.device):
                    floats.insert(0, torch.bfloat16)
            elif self.device_type == "cpu" and self.is_cpu_bf16_supported():
                floats.insert(0, torch.bfloat16)
            self.float_dtypes = tuple(dict.fromkeys(floats))
        if not self.int_dtypes:
            self.int_dtypes = (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            )
        if not self.float8_dtypes:
            self.float8_dtypes = tuple()
        if self.int_quant_bits is None:
            self.int_quant_bits = 8
        return self

    def preprocess(
        self,
        data: Any,
        *args: Any,
        return_keys: bool = True,
        cast: bool = True,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Sequence[Any], Tuple[int, ...]
    ]:
        features: Optional[torch.Tensor] = None
        labels: Optional[torch.Tensor] = None
        keys: Sequence[Any] = ()
        feat_dtype = self.feature_dtype if bool(cast) else None
        label_dtype = self.label_float_dtype if bool(cast) else None
        if isinstance(data, TensorDictBase):
            fkey = collate.get_feature_key(data)
            features = data.get(fkey, None)
            lkey = collate.get_label_key(data, required=False)
            labels = data.get(lkey, None) if lkey is not None else None
            if bool(return_keys):
                row_ids = data.get("row_ids", None)
                if row_ids is None:
                    row_ids = data.get("keys", None)
                keys = row_ids if row_ids is not None else ()
        elif isinstance(data, Mapping):
            if (
                collate._resolve_key(
                    data, collate._FEATURE_KEY_ALIASES, "feature", False
                )
                is not None
                or collate._resolve_key(
                    data, collate._LABEL_KEY_ALIASES, "label", False
                )
                is not None
            ):
                fkey = collate.get_feature_key(data)
                features = data.get(fkey, None)
                lkey = collate.get_label_key(data, required=False)
                labels = data.get(lkey, None) if lkey is not None else None
                if bool(return_keys):
                    row_ids = data.get("row_ids", None)
                    if row_ids is None:
                        row_ids = data.get("keys", None)
                    keys = row_ids if row_ids is not None else ()
            else:
                it = iter(data.items())
                try:
                    k0, v0 = next(it)
                except StopIteration:
                    keys = ()
                else:
                    if isinstance(v0, (list, tuple)) and len(v0) >= 2:
                        n = len(data)
                        keys_list: Optional[list[Any]] = (
                            [k0] if bool(return_keys) else None
                        )
                        x0 = collate._to_safe_tensor(v0[0], feat_dtype)
                        if x0 is None:
                            raise ValueError(
                                "Dataset.preprocess: missing feature in tuple mapping"
                            )
                        feat = torch.empty(
                            (n, *x0.shape), dtype=x0.dtype, device=x0.device
                        )
                        feat[0].copy_(x0)
                        labels_out: Optional[torch.Tensor] = None
                        y0: Optional[torch.Tensor] = None
                        if v0[1] is not None:
                            y0 = collate._to_safe_tensor(v0[1], label_dtype)
                            if y0 is not None:
                                labels_out = torch.empty(
                                    (n, *y0.shape),
                                    dtype=y0.dtype,
                                    device=y0.device,
                                )
                                labels_out[0].copy_(y0)
                        for i, (k, v) in enumerate(it, start=1):
                            if keys_list is not None:
                                keys_list.append(k)
                            if not isinstance(v, (list, tuple)) or len(v) < 1:
                                raise ValueError(
                                    "Dataset.preprocess: invalid tuple mapping element"
                                )
                            x_i = collate._to_safe_tensor(v[0], feat_dtype)
                            if x_i is None:
                                raise ValueError(
                                    "Dataset.preprocess: missing feature in tuple mapping"
                                )
                            if tuple(x_i.shape) != tuple(x0.shape):
                                raise ValueError(
                                    f"Dataset.preprocess: inconsistent feature shapes in tuple mapping: "
                                    f"{tuple(x0.shape)} vs {tuple(x_i.shape)}"
                                )
                            feat[i].copy_(x_i)
                            if labels_out is not None:
                                if len(v) < 2 or v[1] is None:
                                    labels_out = None
                                else:
                                    y_i = collate._to_safe_tensor(
                                        v[1], label_dtype
                                    )
                                    if (
                                        y_i is None
                                        or y0 is None
                                        or tuple(y_i.shape) != tuple(y0.shape)
                                    ):
                                        labels_out = None
                                    else:
                                        labels_out[i].copy_(y_i)
                        features = feat
                        labels = labels_out
                        if bool(return_keys):
                            keys = keys_list
                    else:
                        keys_list: list[Any] = [k0]
                        values_list: list[Any] = [v0]
                        ksize0 = _feature_size_hint(k0)
                        key_as_feature = (
                            ksize0 is not None and 1 < int(ksize0) <= 64
                        )
                        for k, v in it:
                            keys_list.append(k)
                            values_list.append(v)
                            if (
                                key_as_feature
                                and _feature_size_hint(k) != ksize0
                            ):
                                key_as_feature = False
                        if bool(return_keys):
                            keys = keys_list
                        parsed = False
                        if key_as_feature and keys_list and values_list:
                            has_missing_labels = any(
                                (v is None for v in values_list)
                            )
                            try:
                                features = _stack_sequence(
                                    keys_list,
                                    dtype=feat_dtype,
                                    reshape_1d=True,
                                )
                                if has_missing_labels:
                                    labels = None
                                    parsed = features is not None
                                else:
                                    labels = _stack_sequence(
                                        values_list, dtype=label_dtype
                                    )
                                    parsed = (
                                        features is not None
                                        and labels is not None
                                    )
                            except Exception:
                                parsed = False
                        if not parsed:
                            features = _stack_sequence(
                                values_list, dtype=feat_dtype
                            )
                            labels = None
        if features is None:
            raise ValueError(
                "Dataset.preprocess: unable to locate feature tensor(s)"
            )
        features = collate._to_safe_tensor(features, feat_dtype)
        if features.ndim == 0:
            features = features.reshape(1, 1)
        elif features.ndim == 1:
            features = features.reshape(1, -1)
        if not bool(features.is_contiguous()) and not _is_lazy_tensor(
            features
        ):
            features = features.contiguous()
        if labels is not None:
            labels = collate._to_safe_tensor(labels, label_dtype)
            if labels.ndim == 0:
                labels = labels.reshape(1, 1)
            if labels.shape[0] != features.shape[0]:
                labels = labels.reshape(features.shape[0], -1)
            if not bool(labels.is_contiguous()) and not _is_lazy_tensor(
                labels
            ):
                labels = labels.contiguous()
            label_shape = tuple(labels.shape[1:])
        else:
            label_shape = tuple()
        if bool(return_keys):
            if isinstance(keys, torch.Tensor):
                with contextlib.suppress(Exception):
                    keys = keys.detach().cpu().tolist()
            elif keys is None:
                keys = ()
            if not keys:
                keys = range(int(features.shape[0]))
        else:
            keys = ()
        return features, labels, keys, label_shape

    @property
    def scale_min_abs(self) -> Optional[float]:
        return self.scale_min_positive

    @staticmethod
    def tensor_scale_stats(t: torch.Tensor) -> Dict[str, Any]:
        if not isinstance(t, torch.Tensor):
            raise TypeError("tensor_scale_stats expects a torch.Tensor")
        if t.numel() == 0:
            return {
                "has_scale": False,
                "has_nonfinite": False,
                "scale_max_abs": None,
                "scale_min_value": None,
                "scale_max_value": None,
                "scale_min_positive": None,
                "scale_is_integral": None,
            }
        is_complex = False
        _is_complex_fn = getattr(torch, "is_complex", None)
        if callable(_is_complex_fn):
            with contextlib.suppress(Exception):
                is_complex = bool(_is_complex_fn(t))
        else:
            v = getattr(t, "is_complex", False)
            with contextlib.suppress(Exception):
                is_complex = bool(v() if callable(v) else v)
        if is_complex:
            return Dataset.tensor_scale_stats(t.detach().abs())
        if t.is_floating_point():
            x = t.detach()
            finite = torch.isfinite(x)
            all_finite = bool(finite.all().item())
            has_nonfinite = not all_finite
            if all_finite:
                try:
                    min_val = float(x.min().item())
                    max_val = float(x.max().item())
                except Exception:
                    min_val, max_val = (None, None)
                if min_val is None or max_val is None:
                    max_abs = float("nan")
                    min_pos = None
                else:
                    max_abs = float(max(abs(min_val), abs(max_val)))
                    min_pos = None
                    with contextlib.suppress(Exception):
                        pos = x > 0
                        if bool(pos.any().item()):
                            min_pos = float(x[pos].min().item())
                    with contextlib.suppress(Exception):
                        neg = x < 0
                        if bool(neg.any().item()):
                            max_neg = float(x[neg].max().item())
                            cand = float(-max_neg)
                            if cand > 0.0:
                                min_pos = (
                                    cand
                                    if (min_pos is None or cand < min_pos)
                                    else min_pos
                                )
                return {
                    "has_scale": True,
                    "has_nonfinite": bool(has_nonfinite),
                    "scale_max_abs": max_abs,
                    "scale_min_value": min_val,
                    "scale_max_value": max_val,
                    "scale_min_positive": min_pos,
                    "scale_is_integral": None,
                }
            if bool(finite.any().item()):
                xf = x[finite]
                try:
                    min_val = float(xf.min().item()) if xf.numel() else None
                    max_val = float(xf.max().item()) if xf.numel() else None
                except Exception:
                    min_val, max_val = (None, None)
                if min_val is None or max_val is None:
                    max_abs = float("nan")
                else:
                    max_abs = float(max(abs(min_val), abs(max_val)))
                min_pos = None
                with contextlib.suppress(Exception):
                    pos = xf > 0
                    if bool(pos.any().item()):
                        min_pos = float(xf[pos].min().item())
                with contextlib.suppress(Exception):
                    neg = xf < 0
                    if bool(neg.any().item()):
                        max_neg = float(xf[neg].max().item())
                        cand = float(-max_neg)
                        if cand > 0.0:
                            min_pos = (
                                cand
                                if (min_pos is None or cand < min_pos)
                                else min_pos
                            )
            else:
                max_abs = float("nan")
                min_val = None
                max_val = None
                min_pos = None
            return {
                "has_scale": True,
                "has_nonfinite": bool(has_nonfinite),
                "scale_max_abs": max_abs,
                "scale_min_value": min_val,
                "scale_max_value": max_val,
                "scale_min_positive": min_pos,
                "scale_is_integral": None,
            }
        x = t.detach()
        if x.dtype == torch.bool:
            x_i64 = x.to(dtype=torch.int64)
        else:
            x_i64 = x.to(dtype=torch.int64) if x.dtype != torch.int64 else x
        try:
            min_i = int(x_i64.min().item()) if x_i64.numel() else 0
            max_i = int(x_i64.max().item()) if x_i64.numel() else 0
        except Exception:
            vals = x_i64.detach().cpu().reshape(-1).tolist()
            min_i = int(min(vals)) if vals else 0
            max_i = int(max(vals)) if vals else 0
        max_abs = float(max(abs(min_i), abs(max_i)))
        return {
            "has_scale": True,
            "has_nonfinite": False,
            "scale_max_abs": max_abs,
            "scale_min_value": min_i,
            "scale_max_value": max_i,
            "scale_min_positive": None,
            "scale_is_integral": True,
        }

    @staticmethod
    def merge_scale_stats(
        a: Mapping[str, Any], b: Mapping[str, Any]
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out["has_scale"] = bool(a.get("has_scale")) or bool(b.get("has_scale"))
        out["has_nonfinite"] = bool(a.get("has_nonfinite")) or bool(
            b.get("has_nonfinite")
        )
        out["scale_max_abs"] = _merge_opt(
            a.get("scale_max_abs"),
            b.get("scale_max_abs"),
            lambda v1, v2: v1 if v1 >= v2 else v2,
        )
        out["scale_min_value"] = _merge_opt(
            a.get("scale_min_value"),
            b.get("scale_min_value"),
            lambda v1, v2: v1 if v1 <= v2 else v2,
        )
        out["scale_max_value"] = _merge_opt(
            a.get("scale_max_value"),
            b.get("scale_max_value"),
            lambda v1, v2: v1 if v1 >= v2 else v2,
        )
        out["scale_min_positive"] = _merge_opt(
            a.get("scale_min_positive"),
            b.get("scale_min_positive"),
            lambda v1, v2: v1 if v1 <= v2 else v2,
        )
        ia = a.get("scale_is_integral")
        ib = b.get("scale_is_integral")
        if ia is None:
            out["scale_is_integral"] = ib
        elif ib is None:
            out["scale_is_integral"] = ia
        else:
            out["scale_is_integral"] = bool(ia) and bool(ib)
        return out

    @classmethod
    def is_fp32_castable(
        cls,
        stats: Mapping[str, Any],
        *args: Any,
        underflow_action: Optional[str] = None,
        safety_margin: float = 1.0,
    ) -> bool:
        if not stats.get("has_scale"):
            return True
        if bool(stats.get("has_nonfinite")):
            return False
        max_abs = stats.get("scale_max_abs")
        if max_abs is None:
            return True
        try:
            max_abs_f = float(abs(max_abs))
        except Exception:
            return False
        if not math.isfinite(max_abs_f):
            return False
        info = torch.finfo(torch.float32)
        if max_abs_f > float(info.max) / max(1.0, float(safety_margin)):
            return False
        action = normalize_underflow_action(
            underflow_action, default=default_underflow_action()
        )
        if action == "forbid":
            min_pos = stats.get("scale_min_positive")
            if min_pos is not None:
                try:
                    min_pos_f = float(min_pos)
                except Exception:
                    return False
                if math.isfinite(min_pos_f) and min_pos_f > 0.0:
                    if min_pos_f < float(info.tiny) * max(
                        1.0, float(safety_margin)
                    ):
                        return False
        return True

    def update_scale_stats(self, stats: Mapping[str, Any]) -> None:
        self.has_scale = bool(stats.get("has_scale") or False)
        self.has_nonfinite = bool(stats.get("has_nonfinite") or False)
        self.scale_max_abs = (
            float(stats["scale_max_abs"])
            if stats.get("scale_max_abs") is not None
            else None
        )
        self.scale_min_value = stats.get("scale_min_value")
        self.scale_max_value = stats.get("scale_max_value")
        self.scale_min_positive = (
            float(stats["scale_min_positive"])
            if stats.get("scale_min_positive") is not None
            else None
        )
        self.scale_is_integral = (
            bool(stats["scale_is_integral"])
            if stats.get("scale_is_integral") is not None
            else None
        )
        self.is_negotiable = bool(
            self.has_scale
            and (not self.has_nonfinite)
            and self.is_fp32_castable(
                stats,
                underflow_action=self.underflow_action,
                safety_margin=1.0,
            )
        )

    def batch_to_device(
        self,
        batch: Any,
        device: Union[str, torch.device],
        non_blocking: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        dev = torch.device(device)
        feats, labels, _, _ = self.preprocess(batch, return_keys=False)
        feats = feats.to(device=dev, non_blocking=non_blocking)
        if labels is not None:
            labels = labels.to(device=dev, non_blocking=non_blocking)
        return feats, labels

    @staticmethod
    def cuda_compute_capability(
        device: Union[torch.device, str],
    ) -> Tuple[int, int]:
        return cuda_compute_capability(device)

    @staticmethod
    def is_cpu_bf16_supported() -> bool:
        return bool(is_cpu_bf16_supported())

    @staticmethod
    def is_cuda_bf16_supported(
        device: Optional[Union[torch.device, str]] = None,
    ) -> bool:
        return bool(is_cuda_bf16_supported(device))

    @classmethod
    def is_float8_supported(
        cls, device: Optional[Union[torch.device, str]] = None
    ) -> Tuple[bool, str]:
        return is_float8_supported(device)

    @classmethod
    def is_int8_supported(
        cls, device: Optional[Union[torch.device, str]] = None
    ) -> Tuple[bool, str]:
        return is_int8_supported(device)

    @classmethod
    def is_int4_supported(
        cls, device: Optional[Union[torch.device, str]] = None
    ) -> Tuple[bool, str]:
        return is_int4_supported(device)

    @classmethod
    def for_device(
        cls,
        device: Union[torch.device, str],
        *args: Any,
        feature_dtype: torch.dtype = torch.float32,
        label_float_dtype: torch.dtype = torch.float32,
        float_dtypes: Optional[Sequence[torch.dtype]] = None,
        int_dtypes: Optional[Sequence[torch.dtype]] = None,
        extra: Optional[Mapping[str, TExtra]] = None,
    ) -> "Dataset[TExtra]":
        dev = torch.device(device)
        float_dtypes_seq = tuple(float_dtypes) if float_dtypes else ()
        int_dtypes_seq = tuple(int_dtypes) if int_dtypes else ()
        extra_map: Dict[str, TExtra] = dict(extra) if extra else {}
        return cls(
            device=dev,
            float_dtypes=float_dtypes_seq,
            int_dtypes=int_dtypes_seq,
            feature_dtype=feature_dtype,
            label_float_dtype=label_float_dtype,
            extra=extra_map,
        )
