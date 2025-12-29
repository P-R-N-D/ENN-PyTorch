# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import math
import os
import random
from dataclasses import dataclass, field, replace
from typing import (Any, Callable, Dict, Generic, Mapping, MutableMapping,
                    Optional, Sequence, Tuple, TypeVar, Union)

import torch
from ..core.casting import env_first, env_first_float, env_first_int
from ..core.compat import MIN_TORCHDATA_VERSION, ensure_torchdata
from ..core.system import Memory, WorkerPolicy
from ..core.system import accel_device_context as _accel_device_context
from ..core.system import accel_is_available as _accel_is_available
from ..core.system import accel_make_event as _accel_make_event
from ..core.system import \
    accel_pinned_h2d_supported_for_device_type as \
    _accel_pinned_h2d_supported_for_device_type

TExtra = TypeVar("TExtra")

try:
    from tensordict import MemoryMappedTensor, TensorDict, TensorDictBase
except Exception:
    TensorDict = None  # type: ignore[assignment]
    MemoryMappedTensor = None  # type: ignore[assignment]

    class TensorDictBase:  # type: ignore[no-redef]
        pass


from .schemas import (_FEATURE_KEY_ALIASES, _LABEL_KEY_ALIASES, _casefold_str,
                      canonicalize_xy_keys_, default_underflow_action,
                      extract_xy, normalize_underflow_action,
                      resolve_feature_key, resolve_label_key)

def _is_lazy_tensor(x: Any) -> bool:
    if MemoryMappedTensor is None:
        return False
    try:
        return isinstance(x, MemoryMappedTensor)
    except Exception:
        return False
from ..core.system import accel_synchronize as _accel_synchronize
from ..core.system import \
    accel_timing_events_supported_for_device_type as \
    _accel_timing_events_supported_for_device_type
from ..core.system import \
    cuda_compute_capability as _sys_cuda_compute_capability
from ..core.system import get_device as _sys_get_device
from ..core.system import get_device_stats, get_tlb
from ..core.system import is_cpu_bf16_supported as _sys_is_cpu_bf16_supported
from ..core.system import is_cuda_bf16_supported as _sys_is_cuda_bf16_supported
from ..core.system import is_float8_supported as _sys_is_float8_supported
from ..core.system import is_int4_supported as _sys_is_int4_supported
from ..core.system import is_int8_supported as _sys_is_int8_supported


@dataclass(slots=True)
class LoaderPolicy:
    max_batches_accel: int = 4
    max_batches_cpu: int = 2
    soft_cap_multiplier: int = 2

    def hard_inflight_batches(self, device: torch.device | str) -> int:
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if dev.type in ("cuda", "xpu", "mps"):
            return max(1, int(self.max_batches_accel))
        return max(1, int(self.max_batches_cpu))

    def apply_soft_limits(self, wp: WorkerPolicy, device: torch.device | str) -> WorkerPolicy:
        hard = int(self.hard_inflight_batches(device))
        soft_cap = max(1, int(hard * max(1, int(self.soft_cap_multiplier))))

        prefetch_factor = max(1, int(getattr(wp, "prefetch_factor", 1) or 1))
        prebatch = max(1, int(getattr(wp, "prebatch", 1) or 1))

        # Preserve explicit num_workers=0.
        num_workers_req = max(0, int(getattr(wp, "num_workers", 0) or 0))

        # Bound workers by the inflight cap. When num_workers_req==0, keep it 0.
        max_workers_inflight = max(0, int((soft_cap - prebatch) // max(1, prefetch_factor)))
        num_workers = min(num_workers_req, max_workers_inflight, soft_cap)
        num_workers = max(0, int(num_workers))

        inflight = int(num_workers) * int(prefetch_factor) + int(prebatch)
        if inflight > int(soft_cap) and num_workers > 0:
            prefetch_factor = max(
                1,
                int((int(soft_cap) - int(prebatch)) // max(1, int(num_workers))),
            )
            max_workers_inflight = max(
                0, int((soft_cap - prebatch) // max(1, prefetch_factor))
            )
            num_workers = min(int(num_workers), int(max_workers_inflight), int(soft_cap))
            num_workers = max(0, int(num_workers))

        wp.num_workers = int(num_workers)
        wp.prebatch = int(prebatch)
        wp.prefetch_factor = int(prefetch_factor)

        with contextlib.suppress(Exception):
            wp.max_concurrency = max(
                1,
                min(
                    int(getattr(wp, "max_concurrency", 1) or 1),
                    int(wp.num_workers) if int(wp.num_workers) > 0 else 1,
                    int(soft_cap),
                ),
            )
        return wp

    def wrap_input(self, loader: Any, device: torch.device | str, *, name: str) -> Any:
        from ..data.nodes import BatchQueue

        max_batches = self.hard_inflight_batches(device)
        return BatchQueue(loader, max_batches=max_batches, name=name)


@dataclass
class BatchPolicy:
    sample_bytes: int
    host_sample_bytes: Optional[int] = None

    prebatch: int = 1
    prefetch_factor: int = 1
    num_workers: int = 0
    num_streams: int = 1
    max_concurrency: int = 1
    local_world_size: int = 1

    min_batch: int = 1
    max_batch: Optional[int] = None

    device_margin: float = 0.8
    host_margin: float = 0.8
    device_budget_ratio: float = 0.0
    device_budget_min_bytes: int = 0
    device_budget_max_bytes: Optional[int] = None

    host_budget_ratio: float = 0.0
    host_budget_min_bytes: int = 0
    host_budget_max_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        self.sample_bytes = max(int(self.sample_bytes or 0), 0)
        if self.host_sample_bytes is None:
            self.host_sample_bytes = self.sample_bytes

        self.prebatch = max(int(self.prebatch or 0), 0)
        self.prefetch_factor = max(int(self.prefetch_factor or 0), 0)
        self.num_workers = max(int(self.num_workers or 0), 0)
        self.num_streams = max(int(self.num_streams or 0), 0)
        self.max_concurrency = max(int(self.max_concurrency or 0), 0)

        self.min_batch = max(int(self.min_batch or 1), 1)
        if self.max_batch is not None:
            self.max_batch = max(int(self.max_batch), 1)

        self.device_margin = max(0.0, min(1.0, float(self.device_margin)))
        self.host_margin = max(0.0, min(1.0, float(self.host_margin)))

        self.device_budget_ratio = max(0.0, min(1.0, float(self.device_budget_ratio or 0.0)))
        self.host_budget_ratio = max(0.0, min(1.0, float(self.host_budget_ratio or 0.0)))

        self.device_budget_min_bytes = max(int(self.device_budget_min_bytes or 0), 0)
        self.host_budget_min_bytes = max(int(self.host_budget_min_bytes or 0), 0)

        if self.device_budget_max_bytes is not None:
            self.device_budget_max_bytes = max(int(self.device_budget_max_bytes), 0)
        if self.host_budget_max_bytes is not None:
            self.host_budget_max_bytes = max(int(self.host_budget_max_bytes), 0)

    def host_inflight_batches_per_proc(self) -> int:
        return (
            max(1, self.max_concurrency) * max(1, self.prebatch)
            + max(1, self.prefetch_factor)
            + max(1, self.num_streams)
            + 1
        )

    @staticmethod
    def _budget_bytes(
        total_bytes: Optional[int],
        *,
        budget_ratio: float,
        budget_min_bytes: int,
        budget_max_bytes: Optional[int],
    ) -> int:
        total = int(total_bytes) if total_bytes is not None else 0
        ratio = float(budget_ratio or 0.0)
        base = int(float(total) * ratio) if total > 0 and ratio > 0.0 else 0
        budget = max(int(budget_min_bytes or 0), base)
        if (budget <= 0) and (total <= 0) and (budget_max_bytes is not None):
            budget = int(budget_max_bytes)
        elif budget_max_bytes is not None:
            budget = min(budget, int(budget_max_bytes))
        return max(0, int(budget))

    def suggest_batch(
        self,
        *,
        dev_free: Optional[int] = None,
        host_free: Optional[int] = None,
        dev_total: Optional[int] = None,
        host_total: Optional[int] = None,
        local_world_size: Optional[int] = None,
    ) -> int:
        lw = (
            int(local_world_size)
            if local_world_size is not None
            else int(self.local_world_size or 1)
        )
        if lw <= 0:
            lw = 1

        use_dev_budget = (
            self.device_budget_ratio > 0.0
            or self.device_budget_min_bytes > 0
            or self.device_budget_max_bytes is not None
        )
        use_host_budget = (
            self.host_budget_ratio > 0.0
            or self.host_budget_min_bytes > 0
            or self.host_budget_max_bytes is not None
        )

        dev_cap: Optional[int] = None
        if dev_free is not None and dev_free >= 0 and self.sample_bytes > 0:
            denom = max(1, int(self.sample_bytes))
            usable = int(float(dev_free) * float(self.device_margin))
            if use_dev_budget:
                budget = self._budget_bytes(
                    dev_total,
                    budget_ratio=self.device_budget_ratio,
                    budget_min_bytes=self.device_budget_min_bytes,
                    budget_max_bytes=self.device_budget_max_bytes,
                )
                if budget > 0:
                    usable = min(int(usable), int(budget))
            dev_cap = int(max(0, usable) // denom)

        host_cap: Optional[int] = None
        if host_free is not None and host_free >= 0 and (self.host_sample_bytes or 0) > 0:
            inflight = self.host_inflight_batches_per_proc()
            denom = (
                max(1, int(self.host_sample_bytes or 0))
                * max(1, inflight)
                * max(1, lw)
            )
            usable = int(float(host_free) * float(self.host_margin))
            if use_host_budget:
                budget = self._budget_bytes(
                    host_total,
                    budget_ratio=self.host_budget_ratio,
                    budget_min_bytes=self.host_budget_min_bytes,
                    budget_max_bytes=self.host_budget_max_bytes,
                )
                if budget > 0:
                    usable = min(int(usable), int(budget))
            host_cap = int(max(0, usable) // denom)

        candidates = [c for c in (dev_cap, host_cap) if isinstance(c, int) and c >= 0]
        if not candidates:
            b = self.max_batch if self.max_batch is not None else self.min_batch
        else:
            b = min(candidates)
            if self.max_batch is not None:
                b = min(b, int(self.max_batch))

        b = max(int(b), int(self.min_batch))
        return max(1, b)


_TORCHDATA_IMPORT_ERROR: Exception | None = None
try:
    from torchdata.nodes import BaseNode
except Exception as _e:
    # torchdata is an optional dependency unless the streaming pipeline is used.
    # Defer the hard failure to runtime when nodes/pipeline composition is requested.
    BaseNode = object  # type: ignore[assignment]
    _TORCHDATA_IMPORT_ERROR = _e


# NOTE: nodes.py imports Dataset from this module. Importing nodes before Dataset is defined
# can trigger circular imports. We therefore import nodes lazily at the end of the file.
_NODES_IMPORT_ERROR: Exception | None = None
BatchIO = Mapper = Disposable = Loader = Sampler = Source = Multiplexer = BatchQueue = BatchScaler = None  # type: ignore[assignment]
FeatureEngineering = Bootstrap = GraphModel = CompiledGraphModel = Prefetch = None  # type: ignore[assignment]


def _require_nodes() -> None:
    if _TORCHDATA_IMPORT_ERROR is not None:
        ensure_torchdata(err=_TORCHDATA_IMPORT_ERROR, context="stnet.data.pipeline")
    if _NODES_IMPORT_ERROR is not None:
        raise ImportError(
            f"stnet.data.pipeline: data-pipeline components require torchdata>={MIN_TORCHDATA_VERSION} (and tensordict). "
            f"Install/upgrade: pip install -U 'torchdata>={MIN_TORCHDATA_VERSION}'."
        ) from _NODES_IMPORT_ERROR


def _sync_device(device: torch.device) -> None:
    _accel_synchronize(device)


_device_mem_get_info = Memory.device_mem_get_info


def _sample_size(_x_cpu: torch.Tensor, _y_cpu: Optional[torch.Tensor]) -> int:
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

    times: list[float] = []

    # Reuse backend events instead of creating them per-iteration.
    # CUDA/XPU typically support Event-based timing; MPS/CPU fall back to wall-clock.
    dev_t = str(getattr(_device, "type", "cpu") or "cpu")
    pin_ok = bool(_accel_pinned_h2d_supported_for_device_type(dev_t))
    non_blocking = bool(pin_ok)

    ev0 = ev1 = None
    if _accel_timing_events_supported_for_device_type(dev_t):
        ev0 = _accel_make_event(_device, enable_timing=True)
        ev1 = _accel_make_event(_device, enable_timing=True)

    for s in range(_steps + _warmup):
        start = 0
        if N > bs:
            start = random.randint(0, N - bs)
        xb = _x_cpu[start : start + bs]
        yb = None
        if _y_cpu is not None:
            yb = _y_cpu[start : start + bs]

        # Pin only if needed and supported.
        if pin_ok:
            try:
                xbp = xb if (hasattr(xb, "is_pinned") and bool(xb.is_pinned())) else xb.pin_memory()
            except Exception:
                xbp = xb
            if yb is not None:
                try:
                    ybp = yb if (hasattr(yb, "is_pinned") and bool(yb.is_pinned())) else yb.pin_memory()
                except Exception:
                    ybp = yb
            else:
                ybp = None
        else:
            xbp = xb
            ybp = yb

        _sync_device(_device)

        if ev0 is not None and ev1 is not None:
            # Measure with backend events (ms).
            with _accel_device_context(_device):
                with contextlib.suppress(Exception):
                    ev0.record()
                _ = xbp.to(_device, non_blocking=bool(non_blocking))
                if ybp is not None:
                    _ = ybp.to(_device, non_blocking=bool(non_blocking))
                with contextlib.suppress(Exception):
                    ev1.record()
            _sync_device(_device)
            try:
                ms = float(ev0.elapsed_time(ev1))
            except Exception:
                ms = 0.0
        else:
            import time as _t

            tns0 = _t.perf_counter_ns()
            _ = xbp.to(_device, non_blocking=bool(non_blocking))
            if ybp is not None:
                _ = ybp.to(_device, non_blocking=bool(non_blocking))
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
    _ds: "Sampler",
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

    # Try to reuse cached sample bytes from prior probes.
    sbytes_cached = int(getattr(_ds, "_S_sample_bytes", 0) or 0)

    probe = _ds.get(0, min(8, len(_ds)))
    # Support both dict and TensorDict batches with case-insensitive aliases.
    x_cpu, y_cpu = extract_xy(probe, labels_required=False)

    if not isinstance(x_cpu, torch.Tensor):
        x_cpu = torch.as_tensor(x_cpu)
    if y_cpu is not None and not isinstance(y_cpu, torch.Tensor):
        y_cpu = torch.as_tensor(y_cpu)

    sbytes = sbytes_cached if sbytes_cached > 0 else _sample_size(x_cpu, y_cpu)
    if sbytes > 0 and sbytes_cached <= 0:
        with contextlib.suppress(Exception):
            setattr(_ds, "_S_sample_bytes", int(sbytes))

    if sbytes <= 0:
        return (max(1, min(256, len(_ds))), 0.0)

    B_cap = 1 << 16

    # Prefer instance-scoped per-sample bytes if present.
    per_sample = int(getattr(_ds, "_per_sample_mem_bytes", 0) or 0)
    if per_sample <= 0:
        per_sample = int(
            env_first_int(
                ("STNET_PER_SAMPLE_MEM_BYTES", "STNET_DEVICE_BYTES_PER_SAMPLE"),
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
            _wp = WorkerPolicy.autotune()
            _max_conc = max(1, int(getattr(_wp, "max_concurrency", 1) or 1))
            _streams = max(1, int(getattr(_wp, "h2d_streams", 1) or 1))
            _lws = max(1, int(getattr(_wp, "local_world_size", 1) or 1))
        except Exception:
            _wp = None
            _max_conc, _streams, _lws = (1, 1, 1)

    # Environment-derived knobs (centralized via datatype.py helpers).
    dev_margin = float(env_first_float(("STNET_DEVICE_MARGIN",), default=0.90) or 0.90)
    host_margin = float(env_first_float(("STNET_HOST_MARGIN",), default=0.10) or 0.10)

    # Soft slack for auto-derived budgets (only used when budgets are unset).
    budget_slack = float(env_first_float(("STNET_BUDGET_SLACK",), default=1.25) or 1.25)
    budget_slack = max(1.0, min(4.0, float(budget_slack)))

    dev_budget_ratio = float(env_first_float(("STNET_DEVICE_BUDGET_RATIO",), default=1.0) or 1.0)
    dev_budget_min_bytes = int(env_first_int(("STNET_DEVICE_BUDGET_MIN_BYTES",), default=0) or 0)
    _dev_budget_max = int(env_first_int(("STNET_DEVICE_BUDGET_MAX_BYTES",), default=0) or 0)
    dev_budget_max_bytes: Optional[int] = None if _dev_budget_max <= 0 else int(_dev_budget_max)

    host_budget_ratio = float(env_first_float(("STNET_HOST_BUDGET_RATIO",), default=1.0) or 1.0)
    host_budget_min_bytes = int(env_first_int(("STNET_HOST_BUDGET_MIN_BYTES",), default=0) or 0)
    _host_budget_max = int(env_first_int(("STNET_HOST_BUDGET_MAX_BYTES",), default=0) or 0)
    host_budget_max_bytes: Optional[int] = None if _host_budget_max <= 0 else int(_host_budget_max)

    dev_margin = max(0.0, min(1.0, float(dev_margin)))
    host_margin = max(0.0, min(1.0, float(host_margin)))

    dev_budget_ratio = max(0.0, min(1.0, float(dev_budget_ratio)))
    host_budget_ratio = max(0.0, min(1.0, float(host_budget_ratio)))

    dev_budget_min_bytes = max(0, int(dev_budget_min_bytes))
    host_budget_min_bytes = max(0, int(host_budget_min_bytes))
    dev_budget_max_bytes = None if dev_budget_max_bytes is None else max(0, int(dev_budget_max_bytes))
    host_budget_max_bytes = None if host_budget_max_bytes is None else max(0, int(host_budget_max_bytes))

    # Treat 0 (or less) as "unset/disabled" for max-bytes.
    if dev_budget_max_bytes is not None and int(dev_budget_max_bytes) <= 0:
        dev_budget_max_bytes = None
    if host_budget_max_bytes is not None and int(host_budget_max_bytes) <= 0:
        host_budget_max_bytes = None

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
        device_budget_max_bytes=(None if dev_budget_max_bytes is None else int(dev_budget_max_bytes)),
        host_budget_ratio=float(host_budget_ratio),
        host_budget_min_bytes=int(host_budget_min_bytes),
        host_budget_max_bytes=(None if host_budget_max_bytes is None else int(host_budget_max_bytes)),
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

    # Cache probe results to optionally reuse later for initial H2D med estimate.
    probe_bs_cache: Optional[int] = None
    med_probe_cache: Optional[float] = None
    b_init_hint: Optional[int] = None

    # If budgets were not explicitly configured, derive a conservative cap.
    if (tpl.device_budget_max_bytes is None or tpl.host_budget_max_bytes is None) and int(
        tpl.sample_bytes or 0
    ) > 0:
        try:
            inflight = int(tpl.host_inflight_batches_per_proc())
            lw = max(1, int(getattr(tpl, "local_world_size", 1) or 1))

            target_ms = float(
                max(
                    float(_tmin_ms),
                    min(float(_tmax_ms), 0.5 * (float(_tmin_ms) + float(_tmax_ms))),
                )
            )

            probe_bs = max(1, min(int(B_cap), 64))
            med_probe = 0.0
            with contextlib.suppress(Exception):
                med_probe = float(_h2d_counter(x_cpu, y_cpu, _dev, probe_bs, _steps=4, _warmup=1))

            if (
                isinstance(med_probe, (float, int))
                and math.isfinite(float(med_probe))
                and float(med_probe) > 0.0
            ):
                bs_est = int(math.ceil((target_ms * float(probe_bs)) / float(med_probe)))
                target_batch_samples = max(1, min(int(B_cap), bs_est))
                probe_bs_cache = int(probe_bs)
                med_probe_cache = float(med_probe)
                b_init_hint = int(target_batch_samples)
            else:
                target_bytes = 64 * 1024 * 1024
                target_batch_samples = max(
                    1, min(int(B_cap), int(target_bytes // max(1, int(tpl.sample_bytes))))
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

            if (new_dev_cap != tpl.device_budget_max_bytes) or (new_host_cap != tpl.host_budget_max_bytes):
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
    with contextlib.suppress(Exception):
        setattr(_ds, "_S_B_cap", int(B_cap))

    env_max = int(
        env_first_int(("STNET_MAX_BATCH_SIZE", "STNET_MAX_BATCH"), default=0) or 0
    )
    if env_max > 0:
        B_cap = max(1, min(B_cap, int(env_max)))

    with contextlib.suppress(Exception):
        setattr(_ds, "_S_B_cap", int(B_cap))

    candidates = _random_batches(sbytes, _dev, len(_ds))
    if candidates:
        B = min(candidates[-1], B_cap)
    else:
        B = min(64, B_cap)

    if b_init_hint is not None:
        try:
            B_hint = max(1, min(int(B_cap), int(b_init_hint)))
            if candidates:
                cands = [
                    int(c) for c in candidates if isinstance(c, int) and c > 0 and int(c) <= int(B_cap)
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

    if probe_bs_cache is not None and med_probe_cache is not None and int(B) == int(probe_bs_cache):
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


def _is_source_spec(obj: Any) -> bool:
    if not isinstance(obj, Mapping):
        return False
    if "path" not in obj:
        return False
    if "format" not in obj and "kind" not in obj:
        return False
    p = obj.get("path")
    try:
        os.fspath(p)
    except Exception:
        return False
    return True


def dataset(
    source: Source,
    *args: Any,
    split: str = "train",
    val_frac: float = 0.0,
    sampler_scale: Optional["BatchScaler"] = None,
    **kwargs: Any,
) -> "Sampler":
    _require_nodes()
    if not isinstance(source, Mapping):
        raise TypeError(f"dataset expects a Source mapping, got {type(source)}")

    fmt = source.get("format")
    if fmt is None:
        fmt = source.get("kind")
    if fmt is None:
        raise ValueError("Source['format'] or Source['kind'] must be provided")
    fmt = str(fmt)
    if fmt != "memmap":
        raise ValueError(f"Unsupported source format: {fmt!r}")

    path = os.fspath(source.get("path", ""))
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


def _process(
    batch: Any,
    *args: Any,
    flatten_features: bool,
    labels_dtype: Optional[torch.dtype],
    sanitize: bool,
    **kwargs: Any,
) -> Dict[str, Any]:
    features: Any = None
    labels: Any = None

    # Prefer the TensorDict contract: locate the unique feature/label keys
    # case-insensitively. Fall back to legacy {"X","Y"} if needed.
    if isinstance(batch, (Mapping, TensorDictBase)):
        with contextlib.suppress(Exception):
            features, labels = extract_xy(batch, labels_required=False)

    if features is None and isinstance(batch, Mapping):
        features = batch.get("X")
        labels = batch.get("Y", None)

    if (
        flatten_features
        and isinstance(features, torch.Tensor)
        and (features.dim() >= 2)
    ):
        features = features.reshape(features.shape[0], -1)

    if labels_dtype is not None and isinstance(labels, torch.Tensor):
        labels = labels.to(dtype=labels_dtype, non_blocking=True, copy=False)

    if sanitize and isinstance(labels, torch.Tensor) and labels.is_floating_point():
        torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0, out=labels)

    out: Dict[str, Any] = {"X": features}
    if labels is not None:
        out["Y"] = labels
    if isinstance(batch, (Mapping, TensorDictBase)):
        with contextlib.suppress(Exception):
            if "row_ids" in batch:
                out["row_ids"] = batch.get("row_ids")
    return out


def _td_batch_size_from_X(x: Any) -> list[int]:
    if isinstance(x, torch.Tensor) and x.ndim >= 1:
        return [int(x.shape[0])]
    return []


@dataclass(slots=True)
class Collate:

    labels_dtype: Optional[torch.dtype] = None
    sanitize: bool = False
    flatten_features: bool = False

    def __call__(self, batch: Any) -> Any:
        labels_dtype = self.labels_dtype
        sanitize = bool(self.sanitize)
        flatten_features = bool(self.flatten_features)

        if isinstance(batch, (list, tuple)):
            if not batch:
                return batch

            if all(isinstance(elem, TensorDictBase) for elem in batch):
                stacked = torch.stack(batch, dim=0)
                with contextlib.suppress(Exception):
                    canonicalize_xy_keys_(stacked, allow_missing_labels=True)
                try:
                    conv = _process(
                        stacked,
                        flatten_features=flatten_features,
                        labels_dtype=labels_dtype,
                        sanitize=sanitize,
                    )
                except Exception:
                    return stacked

                # Update only the fields that exist (avoid exception-driven control flow).
                if isinstance(conv, Mapping):
                    if "X" in conv:
                        stacked["X"] = conv["X"]
                    if "Y" in conv and conv.get("Y", None) is not None:
                        stacked["Y"] = conv["Y"]
                    if "row_ids" in conv and conv.get("row_ids", None) is not None:
                        stacked["row_ids"] = conv["row_ids"]
                return stacked

            if all(isinstance(elem, Mapping) for elem in batch):
                if TensorDict is not None:
                    samples = []
                    for elem in batch:
                        try:
                            x_i, y_i = extract_xy(elem, labels_required=False)
                        except Exception:
                            x_i = elem.get("X")
                            if x_i is None:
                                x_i = elem.get("x")
                            y_i = elem.get("Y", None)
                            if y_i is None and "y" in elem:
                                y_i = elem.get("y")

                        x_t = _to_tensor_safe(x_i)
                        y_t = _to_tensor_safe(y_i)

                        sample_dict = {"X": x_t}
                        if y_t is not None:
                            sample_dict["Y"] = y_t

                        try:
                            rid = elem.get("row_ids", None)
                            if rid is not None:
                                rid_t = rid if isinstance(rid, torch.Tensor) else torch.as_tensor(rid)
                                sample_dict["row_ids"] = rid_t.reshape(())
                        except Exception:
                            pass

                        samples.append(TensorDict(sample_dict, batch_size=[]))

                    stacked = torch.stack(samples, dim=0)
                    canonicalize_xy_keys_(stacked)

                    try:
                        out = _process(
                            stacked,
                            flatten_features=flatten_features,
                            labels_dtype=labels_dtype,
                            sanitize=sanitize,
                        )
                    except Exception:
                        out = stacked

                    if isinstance(out, Mapping):
                        if "X" in out:
                            stacked.set("X", out["X"])
                        if "Y" in out and out.get("Y", None) is not None:
                            stacked.set("Y", out["Y"])
                        if "row_ids" in out and out.get("row_ids", None) is not None:
                            stacked.set("row_ids", out["row_ids"])

                    return stacked

                # Fallback (no TensorDict): stack into plain tensors/dicts.
                Xs: list[Any] = []
                Ys: list[Any] = []
                for elem in batch:
                    try:
                        x_i, y_i = extract_xy(elem, labels_required=False)
                    except Exception:
                        x_i = elem.get("X")
                        y_i = elem.get("Y", None)
                    Xs.append(x_i)
                    Ys.append(y_i)

                Xs = [_to_tensor_safe(x) for x in Xs]
                Ys = [_to_tensor_safe(y) for y in Ys]

                X: Any
                Y: Any
                if all(isinstance(x, torch.Tensor) for x in Xs):
                    X = torch.stack(Xs, dim=0)
                else:
                    X = Xs

                if all(isinstance(y, torch.Tensor) for y in Ys):
                    Y = torch.stack(Ys, dim=0)
                else:
                    Y = Ys

                data: dict[str, Any] = {"X": X}
                if isinstance(Y, torch.Tensor):
                    data["Y"] = Y

                try:
                    rids = [elem.get("row_ids") for elem in batch]
                    if rids and all(r is not None for r in rids):
                        parts = [
                            (r if isinstance(r, torch.Tensor) else torch.as_tensor(r)).reshape(-1)
                            for r in rids
                        ]
                        row_ids = torch.cat(parts, dim=0) if parts else None
                        if row_ids is not None:
                            data["row_ids"] = row_ids
                except Exception:
                    pass

                return data

            return batch

        if isinstance(batch, Mapping):
            try:
                conv = _process(
                    batch,
                    flatten_features=flatten_features,
                    labels_dtype=labels_dtype,
                    sanitize=sanitize,
                )
            except Exception:
                conv = batch

            X = conv.get("X", None) if isinstance(conv, Mapping) else None
            Y = conv.get("Y", None) if isinstance(conv, Mapping) else None

            if X is None:
                with contextlib.suppress(Exception):
                    X, Y = extract_xy(batch, labels_required=False)
            if X is None:
                X = batch.get("X")
            if Y is None:
                Y = batch.get("Y")

            row_ids = None
            if isinstance(conv, Mapping):
                row_ids = conv.get("row_ids", None)
            if row_ids is None:
                row_ids = batch.get("row_ids")

            data: dict[str, Any] = {"X": X}
            if isinstance(Y, torch.Tensor):
                data["Y"] = Y
            if row_ids is not None:
                data["row_ids"] = row_ids
            if TensorDict is None:
                return data
            return TensorDict(data, batch_size=_td_batch_size_from_X(X))

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
    seed: int = 0,
    epochables: Optional[list[Any]] = None,
    **kwargs: Any,
) -> Tuple[BaseNode, BaseNode, BaseNode]:
    _require_nodes()
    device_obj = torch.device(device) if not isinstance(device, torch.device) else device
    with contextlib.suppress(Exception):
        get_tlb(io_workers=io_workers)

    mx_weights = weights if isinstance(node_or_nodes, Mapping) and isinstance(weights, Mapping) else None
    sampler = Multiplexer(stop_criteria="ALL_DATASETS_EXHAUSTED", seed=int(seed), weights=mx_weights)
    source = sampler.compose(node_or_nodes)

    if epochables is not None and getattr(sampler, "_node", None) is not None:
        with contextlib.suppress(Exception):
            # Put it first so it runs before per-dataset epoch hooks.
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

    # Keep the 3-tuple return for backward compatibility: (source, mapped, output).
    # Historically "output" was the same as "mapped"; we keep that invariant.
    return source, mapped, mapped


def fetch(
    sources: Union[
        Source,
        Sequence[Source],
        Mapping[str, Source],
    ],
    device: Union[str, torch.device],
    val_frac: float = 0.0,
    non_blocking_copy: bool = True,
    labels_dtype: Optional[torch.dtype] = None,
    sanitize: bool = True,
    flatten_features: bool = True,
    train_shuffle: bool = True,
    seed: int = 0,
    train_weights: Optional[Mapping[str, float]] = None,
    val_weights: Optional[Mapping[str, float]] = None,
    worker_policy: Optional[WorkerPolicy] = None,
    sampler_scale: Optional["BatchScaler"] = None,
    *,
    loader_policy: Optional[LoaderPolicy] = None,
) -> Dict[str, Any]:
    _require_nodes()
    device_obj = torch.device(device) if not isinstance(device, torch.device) else device

    lp = loader_policy if isinstance(loader_policy, LoaderPolicy) else LoaderPolicy()

    _wp = worker_policy if isinstance(worker_policy, WorkerPolicy) else WorkerPolicy.autotune()
    _wp.apply_torch_threads()

    io_workers = int(getattr(_wp, "num_workers", 0) or 0)
    prebatch = max(1, int(getattr(_wp, "prebatch", 1) or 1))

    # Clamp prefetch_factor once, and treat it as an upper bound.
    pf_depth_fixed = max(1, int(getattr(_wp, "prefetch_factor", 1) or 1))
    pf_depth_fixed = max(1, min(8, int(pf_depth_fixed)))
    pf_depth = int(pf_depth_fixed)

    collate_fn = Collate(
        labels_dtype=labels_dtype,
        sanitize=bool(sanitize),
        flatten_features=bool(flatten_features),
    )

    allocated = Disposable()
    batch_size: Optional[int] = None
    scale_ctl = sampler_scale if sampler_scale is not None else BatchScaler()

    train_epochables: list[Any] = []

    _device_obj = device_obj
    _auto_bs_candidates: list[int] = []

    def _stream_batch(_ds: Sampler, *, pf: int) -> Tuple[int, float]:
        try:
            return _batch_interval(
                _ds,
                _device_obj,
                prefetch_factor=int(pf),
                num_workers=io_workers,
                prebatch=prebatch,
                worker_policy=_wp,
            )
        except Exception:
            return (0, 0.0)

    def _auto_batch_size(
        _datasets: Mapping[str, Sampler], *, pf: int, fallback: int
    ) -> int:
        _auto_bs_candidates.clear()
        for _ds in _datasets.values():
            B_i, _ = _stream_batch(_ds, pf=pf)
            if B_i > 0:
                _auto_bs_candidates.append(int(B_i))
        if not _auto_bs_candidates:
            return max(1, int(fallback))
        cand_mean = int(sum(_auto_bs_candidates) // max(1, len(_auto_bs_candidates)))
        cand_max = int(max(_auto_bs_candidates))
        return int(max(1, min(cand_max, cand_mean)))

    def _rescale_batch(_datasets: Mapping[str, Sampler], _bs: int, *, pf: int) -> int:
        return _auto_batch_size(_datasets, pf=pf, fallback=int(_bs))

    def _cap_pf_depth(_datasets: Mapping[str, Sampler], _pf: int, _bs: int) -> int:
        try:
            host_avail = int(Memory.available())
            if host_avail <= 0:
                return int(_pf)

            dev_free, _ = _device_mem_get_info(_device_obj)
            effective_avail = min(host_avail, dev_free) if dev_free is not None else host_avail

            budget = int(effective_avail * 0.15)
            if budget <= 0 or _bs <= 0:
                return int(_pf)

            sbytes_max = 0
            for _ds in _datasets.values():
                if len(_ds) <= 0:
                    continue
                cached = int(getattr(_ds, "_S_sample_bytes", 0) or 0)
                if cached > 0:
                    sbytes_max = max(sbytes_max, cached)
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
                sb = _sample_size(x, y)
                if sb > 0:
                    with contextlib.suppress(Exception):
                        setattr(_ds, "_S_sample_bytes", int(sb))
                sbytes_max = max(sbytes_max, sb)

            if sbytes_max <= 0:
                return int(_pf)

            bytes_per_batch = int(sbytes_max) * int(_bs)
            if bytes_per_batch <= 0:
                return int(_pf)

            pf_cap = max(1, int(budget // max(1, bytes_per_batch)))

            # Also respect inflight limits derived from LoaderPolicy.
            with contextlib.suppress(Exception):
                hard = int(lp.hard_inflight_batches(_device_obj))
                soft_cap = max(1, int(hard * max(1, int(lp.soft_cap_multiplier))))
                pb = max(1, int(prebatch))
                workers = max(1, int(io_workers) if int(io_workers) > 0 else 1)
                inflight_pf_cap = max(1, int((soft_cap - pb) // max(1, workers)))
                pf_cap = min(int(pf_cap), int(inflight_pf_cap))

            return int(max(1, min(int(_pf), int(pf_cap), 8)))
        except Exception:
            return int(_pf)

    def _make_iterate(
        _datasets: Mapping[str, Sampler], _collate: Callable[[Any], Any]
    ) -> Callable[[Any], Any]:
        def iterate(sample: Any) -> Any:
            if isinstance(sample, tuple) and len(sample) == 2:
                k, span = sample
                ds = _datasets.get(str(k))
                if ds is None:
                    return None
                s, e = int(span[0]), int(span[1])
                batch = ds.get(s, e)
                return _collate(batch) if batch is not None else None
            return _collate(sample)

        return iterate

    def _normalize_sources(_sources: Any) -> Dict[str, Source]:
        if isinstance(_sources, Mapping) and (not _is_source_spec(_sources)):
            return {str(k): v for k, v in _sources.items()}
        if isinstance(_sources, (list, tuple)):
            return {str(i): v for i, v in enumerate(_sources)}
        return {"0": _sources}

    def _build_datasets(
        _specs: Mapping[str, Source], *, split: str, collect_epochables: bool
    ) -> Dict[str, Sampler]:
        out: Dict[str, Sampler] = {}
        for k, spec in _specs.items():
            ds = dataset(spec, split=split, val_frac=val_frac, sampler_scale=scale_ctl)
            allocated.add(ds)
            out[str(k)] = ds
            if collect_epochables:
                train_epochables.append(ds)
        return out

    def _build_sampler_nodes(
        _datasets: Mapping[str, Sampler], *, bs: int, shuffle: bool
    ) -> Tuple[Dict[str, BaseNode], Dict[str, int]]:
        nodes: Dict[str, BaseNode] = {}
        lengths: Dict[str, int] = {}
        for k, ds in _datasets.items():
            sn = ds.compose(batch_size=int(bs), shuffle=bool(shuffle), seed=seed, key=str(k))
            if len(ds) > 0:
                nodes[str(k)] = sn
                lengths[str(k)] = int(len(ds))
        return nodes, lengths

    specs = _normalize_sources(sources)

    # ------------------- Train split -------------------
    datasets_train = _build_datasets(specs, split="train", collect_epochables=True)
    if batch_size is None or batch_size <= 0:
        batch_size = int(_auto_batch_size(datasets_train, pf=int(pf_depth_fixed), fallback=1))

    pf_depth = int(_cap_pf_depth(datasets_train, int(pf_depth_fixed), int(batch_size)))
    pf_depth = int(max(1, min(int(pf_depth), int(pf_depth_fixed))))
    if pf_depth != int(pf_depth_fixed):
        batch_size = int(_rescale_batch(datasets_train, int(batch_size), pf=int(pf_depth)))

    sampler_nodes, lengths = _build_sampler_nodes(
        datasets_train, bs=int(batch_size), shuffle=bool(train_shuffle)
    )
    if isinstance(train_weights, Mapping):
        train_weights = {
            str(k): float(v) for k, v in dict(train_weights).items() if str(k) in sampler_nodes
        }
    if not sampler_nodes:
        raise RuntimeError("No non-empty training sources provided.")

    train_length: Optional[int] = int(sum(lengths.values())) if lengths else None
    iterate_train = _make_iterate(datasets_train, collate_fn)
    _, mapped_train, _ = compose(
        sampler_nodes,
        device=device_obj,
        map_fn=iterate_train,
        prefetch_factor=int(pf_depth),
        non_blocking_copy=non_blocking_copy,
        io_workers=io_workers,
        prebatch=prebatch,
        weights=train_weights,
        seed=seed,
        epochables=train_epochables,
    )
    train_loader = Loader.compose(
        mapped_train,
        device=device_obj,
        prefetch_factor=int(pf_depth),
        non_blocking=bool(non_blocking_copy),
        length=train_length,
    )

    # ------------------- Val split -------------------
    val_loader = None
    if val_frac and val_frac > 0:
        datasets_val = _build_datasets(specs, split="val", collect_epochables=False)

        batch_size_val = batch_size
        if batch_size_val is None or batch_size_val <= 0:
            batch_size_val = int(_auto_batch_size(datasets_val, pf=int(pf_depth_fixed), fallback=1))

        pf_depth_val = int(_cap_pf_depth(datasets_val, int(pf_depth_fixed), int(batch_size_val)))
        pf_depth_val = int(max(1, min(int(pf_depth_val), int(pf_depth_fixed))))
        if pf_depth_val != int(pf_depth_fixed):
            batch_size_val = int(
                _rescale_batch(datasets_val, int(batch_size_val), pf=int(pf_depth_val))
            )

        sampler_nodes_val, lengths_val = _build_sampler_nodes(
            datasets_val, bs=int(batch_size_val), shuffle=False
        )
        if isinstance(val_weights, Mapping):
            val_weights = {
                str(k): float(v) for k, v in dict(val_weights).items() if str(k) in sampler_nodes_val
            }
        if not sampler_nodes_val:
            raise RuntimeError("No non-empty validation sources provided.")

        val_length: Optional[int] = int(sum(lengths_val.values())) if lengths_val else None
        iterate_val = _make_iterate(datasets_val, collate_fn)
        _, mapped_val, _ = compose(
            sampler_nodes_val,
            device=device_obj,
            map_fn=iterate_val,
            prefetch_factor=int(pf_depth_val),
            non_blocking_copy=non_blocking_copy,
            io_workers=io_workers,
            prebatch=prebatch,
            weights=val_weights,
            seed=seed,
        )
        val_loader = Loader.compose(
            mapped_val,
            device=device_obj,
            prefetch_factor=int(pf_depth_val),
            non_blocking=bool(non_blocking_copy),
            length=val_length,
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

    train_weights: Optional[Mapping[str, float]] = None
    val_weights: Optional[Mapping[str, float]] = None

    worker_policy: Optional[WorkerPolicy] = None
    loader_policy: LoaderPolicy = field(default_factory=LoaderPolicy)

    raw_training_loader: Any = None
    raw_validation_loader: Any = None

    training_loader: Any = None
    validation_loader: Any = None
    disposable: Any = None
    sampler_scale: Optional["BatchScaler"] = None

    _opened: bool = False

    def open(
        self,
        *,
        train_state: Optional[Dict[str, Any]] = None,
        val_state: Optional[Dict[str, Any]] = None,
    ) -> "Session":
        _require_nodes()
        if self.sampler_scale is None:
            self.sampler_scale = BatchScaler()

        dev = torch.device(self.device) if not isinstance(self.device, torch.device) else self.device

        # Use a per-session policy instance (avoid unexpected shared mutations).
        wp = self.worker_policy or WorkerPolicy.autotune()
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

        if train_state and train_loader is not None and hasattr(train_loader, "load_state_dict"):
            with contextlib.suppress(Exception):
                train_loader.load_state_dict(train_state)

        if val_state and val_loader is not None and hasattr(val_loader, "load_state_dict"):
            with contextlib.suppress(Exception):
                val_loader.load_state_dict(val_state)

        self.training_loader = (
            self.loader_policy.wrap_input(train_loader, dev, name="train-input")
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
                setattr(self.training_loader, "_stnet_sampler_scale", self.sampler_scale)
            if self.validation_loader is not None:
                setattr(self.validation_loader, "_stnet_sampler_scale", self.sampler_scale)

        with contextlib.suppress(Exception):
            if self.raw_training_loader is not None and self.training_loader is not None:
                epochables = getattr(self.raw_training_loader, "_stnet_epochables", None)
                if epochables is not None:
                    setattr(self.training_loader, "_stnet_epochables", epochables)

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

    def __enter__(self) -> "Session":
        return self.open()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

def _to_tensor_safe(obj: Any, dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
    if obj is None:
        return None

    t = obj if isinstance(obj, torch.Tensor) else torch.as_tensor(obj)
    if dtype is not None and isinstance(t, torch.Tensor):
        with contextlib.suppress(Exception):
            if t.dtype != dtype:
                t = t.to(dtype=dtype, copy=False)
    return t


def _feature_size_hint(obj: Any) -> Optional[int]:
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return 1
        with contextlib.suppress(Exception):
            return int(obj.numel())
        return None
    if isinstance(obj, (tuple, list)):
        with contextlib.suppress(Exception):
            return int(len(obj))
        return None
    return None


def _stack_sequence(
    seq: Sequence[Any],
    *,
    dtype: Optional[torch.dtype],
    reshape_1d: bool = False,
) -> Optional[torch.Tensor]:
    if not seq:
        return None

    t0 = _to_tensor_safe(seq[0], dtype)
    if t0 is None:
        return None
    if reshape_1d:
        t0 = t0.reshape(-1)

    n = int(len(seq))
    out = torch.empty((n, *tuple(t0.shape)), dtype=t0.dtype, device=t0.device)
    out[0].copy_(t0)

    for i in range(1, n):
        ti = _to_tensor_safe(seq[i], dtype)
        if ti is None:
            raise ValueError("Dataset.preprocess: missing tensor element")
        if reshape_1d:
            ti = ti.reshape(-1)
        if tuple(ti.shape) != tuple(t0.shape):
            raise ValueError(
                f"Dataset.preprocess: inconsistent shapes in stacked sequence: "
                f"{tuple(t0.shape)} vs {tuple(ti.shape)}"
            )
        out[i].copy_(ti)

    return out


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

    # --- Scale / precision negotiation metadata (populated from data or meta.json)
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

        # Fill in defaults from core.device_stats when not explicitly provided.
        dev_stats = get_device_stats(self.device)
        if not self.float_dtypes:
            self.float_dtypes = tuple(getattr(dev_stats, "float_dtypes", ()))
        if not self.int_dtypes:
            self.int_dtypes = tuple(getattr(dev_stats, "int_dtypes", ()))
        if self.int_quant_bits is None:
            self.int_quant_bits = int(getattr(dev_stats, "int_quant_bits", 8))

        if not self.float_dtypes:
            floats: list[torch.dtype] = [torch.float32]
            if self.device_type == "cuda" and _accel_is_available("cuda"):
                floats.insert(0, torch.float16)
                if self.is_cuda_bf16_supported(self.device):
                    floats.insert(0, torch.bfloat16)
            elif self.device_type == "cpu" and self.is_cpu_bf16_supported():
                floats.insert(0, torch.bfloat16)
            self.float_dtypes = tuple(dict.fromkeys(floats))

        if not self.int_dtypes:
            self.int_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64)

        if not self.float8_dtypes:
            self.float8_dtypes = tuple()

        if self.int_quant_bits is None:
            self.int_quant_bits = 8
        return self

    def preprocess(
        self,
        data: Any,
        *,
        return_keys: bool = True,
        cast: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Sequence[Any], Tuple[int, ...]]:
        features: Optional[torch.Tensor] = None
        labels: Optional[torch.Tensor] = None
        keys: Sequence[Any] = ()

        feat_dtype = self.feature_dtype if bool(cast) else None
        label_dtype = self.label_float_dtype if bool(cast) else None

        if isinstance(data, TensorDictBase):
            # Strict, case-insensitive alias resolution. Enforces "exactly one".
            fkey = resolve_feature_key(data)
            features = data.get(fkey, None)
            lkey = resolve_label_key(data, required=False)
            labels = data.get(lkey, None) if lkey is not None else None
            if bool(return_keys):
                keys = data.get("row_ids", None) or data.get("keys", None) or ()
        elif isinstance(data, Mapping):
            # If the mapping explicitly contains feature/label columns, use the strict
            # alias contract. Otherwise, fall back to the {key: label} / {key: (x,y)}
            # heuristic used for compact datasets.
            has_column_keys = False
            for k in data.keys():
                ck = _casefold_str(k)
                if ck is None:
                    continue
                if ck in _FEATURE_KEY_ALIASES or ck in _LABEL_KEY_ALIASES:
                    has_column_keys = True
                    break

            if has_column_keys:
                fkey = resolve_feature_key(data)
                features = data.get(fkey, None)
                lkey = resolve_label_key(data, required=False)
                labels = data.get(lkey, None) if lkey is not None else None
                if bool(return_keys):
                    keys = data.get("row_ids", None) or data.get("keys", None) or ()
            else:
                # Heuristic: mapping may be {key: (feature, label)} or {feature: label}.
                # Avoid materializing list(data.items()) which can cause large transient allocations.
                it = iter(data.items())
                try:
                    k0, v0 = next(it)
                except StopIteration:
                    keys = ()
                else:
                    # Case A: {row_id: (feature, label)}
                    if isinstance(v0, (list, tuple)) and len(v0) >= 2:
                        n = len(data)
                        keys_list: Optional[list[Any]] = [k0] if bool(return_keys) else None

                        x0 = _to_tensor_safe(v0[0], feat_dtype)
                        if x0 is None:
                            raise ValueError("Dataset.preprocess: missing feature in tuple mapping")

                        feat = torch.empty((n, *x0.shape), dtype=x0.dtype, device=x0.device)
                        feat[0].copy_(x0)

                        labels_out: Optional[torch.Tensor] = None
                        y0: Optional[torch.Tensor] = None
                        if v0[1] is not None:
                            y0 = _to_tensor_safe(v0[1], label_dtype)
                            if y0 is not None:
                                labels_out = torch.empty((n, *y0.shape), dtype=y0.dtype, device=y0.device)
                                labels_out[0].copy_(y0)

                        for i, (k, v) in enumerate(it, start=1):
                            if keys_list is not None:
                                keys_list.append(k)
                            if not isinstance(v, (list, tuple)) or len(v) < 1:
                                raise ValueError("Dataset.preprocess: invalid tuple mapping element")

                            x_i = _to_tensor_safe(v[0], feat_dtype)
                            if x_i is None:
                                raise ValueError("Dataset.preprocess: missing feature in tuple mapping")
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
                                    y_i = _to_tensor_safe(v[1], label_dtype)
                                    if y_i is None or y0 is None or tuple(y_i.shape) != tuple(y0.shape):
                                        labels_out = None
                                    else:
                                        labels_out[i].copy_(y_i)

                        features = feat
                        labels = labels_out
                        if bool(return_keys):
                            keys = keys_list

                    # Case B: mapping may be {key: feature} or {feature: label}
                    else:
                        keys_list: list[Any] = [k0]
                        values_list: list[Any] = [v0]

                        ksize0 = _feature_size_hint(k0)
                        key_as_feature = ksize0 is not None and 1 < int(ksize0) <= 64

                        for k, v in it:
                            keys_list.append(k)
                            values_list.append(v)
                            if key_as_feature and _feature_size_hint(k) != ksize0:
                                key_as_feature = False

                        if bool(return_keys):
                            keys = keys_list

                        parsed = False
                        if key_as_feature and keys_list and values_list:
                            has_missing_labels = any((v is None for v in values_list))
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
                                    labels = _stack_sequence(values_list, dtype=label_dtype)
                                    parsed = features is not None and labels is not None
                            except Exception:
                                parsed = False

                        if not parsed:
                            features = _stack_sequence(values_list, dtype=feat_dtype)
                            labels = None

        if features is None:
            raise ValueError("Dataset.preprocess: unable to locate feature tensor(s)")

        features = _to_tensor_safe(features, feat_dtype)
        if features.ndim == 0:
            features = features.reshape(1, 1)
        elif features.ndim == 1:
            features = features.reshape(1, -1)
        if not bool(features.is_contiguous()) and not _is_lazy_tensor(features):
            features = features.contiguous()

        if labels is not None:
            labels = _to_tensor_safe(labels, label_dtype)
            if labels.ndim == 0:
                labels = labels.reshape(1, 1)
            if labels.shape[0] != features.shape[0]:
                labels = labels.reshape(features.shape[0], -1)
            if not bool(labels.is_contiguous()) and not _is_lazy_tensor(labels):
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
                # Fast path (common): all values are finite, so we can avoid boolean indexing
                # that would materialize a full-size copy.
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

                    # Compute smallest strictly-positive magnitude without allocating abs(x).
                    min_pos = None
                    with contextlib.suppress(Exception):
                        pos = x > 0
                        if bool(pos.any().item()):
                            min_pos = float(x[pos].min().item())
                    with contextlib.suppress(Exception):
                        neg = x < 0
                        if bool(neg.any().item()):
                            # Among negatives, the one closest to zero is the maximum negative.
                            max_neg = float(x[neg].max().item())
                            cand = float(-max_neg)
                            if cand > 0.0:
                                min_pos = cand if (min_pos is None or cand < min_pos) else min_pos

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
                            min_pos = cand if (min_pos is None or cand < min_pos) else min_pos
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
    def merge_scale_stats(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out["has_scale"] = bool(a.get("has_scale") or b.get("has_scale"))
        out["has_nonfinite"] = bool(a.get("has_nonfinite") or b.get("has_nonfinite"))

        def _max(v1: Any, v2: Any) -> Any:
            if v1 is None:
                return v2
            if v2 is None:
                return v1
            try:
                return float(v1) if float(v1) >= float(v2) else float(v2)
            except Exception:
                return v1

        def _min_pos(v1: Any, v2: Any) -> Any:
            if v1 is None:
                return v2
            if v2 is None:
                return v1
            try:
                return float(v1) if float(v1) <= float(v2) else float(v2)
            except Exception:
                return v1

        def _min_num(v1: Any, v2: Any) -> Any:
            if v1 is None:
                return v2
            if v2 is None:
                return v1
            try:
                return v1 if v1 <= v2 else v2
            except Exception:
                try:
                    f1, f2 = float(v1), float(v2)
                    return v1 if f1 <= f2 else v2
                except Exception:
                    return v1

        out["scale_max_abs"] = _max(a.get("scale_max_abs"), b.get("scale_max_abs"))
        out["scale_min_value"] = _min_num(a.get("scale_min_value"), b.get("scale_min_value"))
        out["scale_max_value"] = _max(a.get("scale_max_value"), b.get("scale_max_value"))
        out["scale_min_positive"] = _min_pos(a.get("scale_min_positive"), b.get("scale_min_positive"))

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
        *,
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
                    if min_pos_f < float(info.tiny) * max(1.0, float(safety_margin)):
                        return False
        return True

    def update_scale_stats(self, stats: Mapping[str, Any]) -> None:
        self.has_scale = bool(stats.get("has_scale") or False)
        self.has_nonfinite = bool(stats.get("has_nonfinite") or False)
        self.scale_max_abs = (
            float(stats["scale_max_abs"]) if stats.get("scale_max_abs") is not None else None
        )
        self.scale_min_value = stats.get("scale_min_value")
        self.scale_max_value = stats.get("scale_max_value")
        self.scale_min_positive = (
            float(stats["scale_min_positive"]) if stats.get("scale_min_positive") is not None else None
        )
        self.scale_is_integral = (
            bool(stats["scale_is_integral"]) if stats.get("scale_is_integral") is not None else None
        )
        self.is_negotiable = bool(
            self.has_scale
            and (not self.has_nonfinite)
            and self.is_fp32_castable(
                stats, underflow_action=self.underflow_action, safety_margin=1.0
            )
        )

    def batch_to_device(
        self, batch: Any, device: Union[str, torch.device], non_blocking: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        dev = torch.device(device)
        feats, labels, _, _ = self.preprocess(batch, return_keys=False)
        feats = feats.to(device=dev, non_blocking=non_blocking)
        if labels is not None:
            labels = labels.to(device=dev, non_blocking=non_blocking)
        return feats, labels

    def _refresh_device_info(self) -> None:
        stats = get_device_stats(self.device)
        self.device = stats.device
        self.device_type = stats.device_type
        self.cuda_cc = stats.cuda_cc

    @staticmethod
    def cuda_compute_capability(device: Union[torch.device, str]) -> Tuple[int, int]:
        return _sys_cuda_compute_capability(device)

    @staticmethod
    def is_cpu_bf16_supported() -> bool:
        return bool(_sys_is_cpu_bf16_supported())

    @staticmethod
    def is_cuda_bf16_supported(device: Optional[Union[torch.device, str]] = None) -> bool:
        return bool(_sys_is_cuda_bf16_supported(device))

    @staticmethod
    def _resolve_device(device: Optional[Union[torch.device, str]]) -> torch.device:
        if device is not None:
            return torch.device(device)
        # Use the project's centralized device selection (CUDA/XPU/MPS/CPU).
        with contextlib.suppress(Exception):
            return _sys_get_device()
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
        float_env = env_first(("STNET_DATA_FLOAT_DTYPES", "STNET_FLOAT_DTYPES"))
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

    @classmethod
    def is_float8_supported(
        cls, device: Optional[Union[torch.device, str]] = None
    ) -> Tuple[bool, str]:
        return _sys_is_float8_supported(device)

    @classmethod
    def is_int8_supported(cls, device: Optional[Union[torch.device, str]] = None) -> Tuple[bool, str]:
        return _sys_is_int8_supported(device)

    @classmethod
    def is_int4_supported(cls, device: Optional[Union[torch.device, str]] = None) -> Tuple[bool, str]:
        return _sys_is_int4_supported(device)

    @classmethod
    def for_device(
        cls,
        device: Union[torch.device, str],
        *,
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

# Late import: avoid circular dependency (nodes.py imports Dataset from this module).
try:
    from .nodes import (
        BatchIO,
        BatchScaler,
        Disposable,
        Loader,
        Mapper,
        Multiplexer,
        Sampler,
        Source,
    )
except Exception as _e:
    _NODES_IMPORT_ERROR = _e
