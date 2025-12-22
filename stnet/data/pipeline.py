# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import math
import contextlib
import random
from dataclasses import dataclass, field, replace
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

TensorLike = Any

TExtra = TypeVar("TExtra")

import torch
try:
    from tensordict import TensorDict, TensorDictBase, stack
except Exception:
    TensorDict = None  # type: ignore[assignment]
    stack = None  # type: ignore[assignment]

    class TensorDictBase:  # type: ignore[no-redef]
        pass

from ..backend.system import Memory, WorkerPolicy, get_tlb
from ..backend.compat import ensure_torchdata, MIN_TORCHDATA_VERSION
from .datatype import env_first


_DEF_UNDERFLOW_ACTIONS = {"allow", "warn", "forbid"}


def default_underflow_action() -> str:
    """Return underflow policy used by precision negotiation.

    - allow: underflow (flush-to-zero / subnormals) is allowed
    - warn: allowed but may be logged
    - forbid: treat underflow as unsafe for downcasting
    """
    raw = str(env_first(('STNET_DATA_UNDERFLOW_ACTION','STNET_UNDERFLOW_ACTION'), default='warn') or 'warn').strip().lower()
    return raw if raw in _DEF_UNDERFLOW_ACTIONS else 'warn'


def normalize_underflow_action(value: object, *, default: str = 'warn') -> str:
    raw = str(value if value is not None else default).strip().lower()
    if raw in _DEF_UNDERFLOW_ACTIONS:
        return raw
    return str(default).strip().lower() if str(default).strip().lower() in _DEF_UNDERFLOW_ACTIONS else 'warn'

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

        num_workers = max(1, int(getattr(wp, "num_workers", 1) or 1))

        max_workers_inflight = max(1, int((soft_cap - prebatch) // max(1, prefetch_factor)))
        num_workers = max(1, min(int(num_workers), int(max_workers_inflight), int(soft_cap)))

        inflight = int(num_workers) * int(prefetch_factor) + int(prebatch)
        if inflight > int(soft_cap):
            prefetch_factor = max(1, int((int(soft_cap) - int(prebatch)) // max(1, int(num_workers))))
            max_workers_inflight = max(1, int((soft_cap - prebatch) // max(1, prefetch_factor)))
            num_workers = max(1, min(int(num_workers), int(max_workers_inflight), int(soft_cap)))

        wp.num_workers = int(num_workers)
        wp.prebatch = int(prebatch)
        wp.prefetch_factor = int(prefetch_factor)

        with contextlib.suppress(Exception):
            wp.max_concurrency = max(
                1,
                min(
                    int(getattr(wp, "max_concurrency", 1) or 1),
                    int(wp.num_workers),
                    int(soft_cap),
                ),
            )
        return wp

    def wrap_input(self, loader: Any, device: torch.device | str, *, name: str) -> Any:
        from ..data.nodes import BufferedLoader

        max_batches = self.hard_inflight_batches(device)
        return BufferedLoader(loader, max_batches=max_batches, name=name)


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


_NODES_IMPORT_ERROR: Exception | None = None
try:
    from .nodes import Connector, Disposable, Loader, Sampler, SamplerScale, Source, Wrapper
except Exception as _e:
    Connector = Disposable = Loader = Sampler = Source = Wrapper = SamplerScale = None  # type: ignore[assignment]
    _NODES_IMPORT_ERROR = _e

def _require_nodes() -> None:
    if _TORCHDATA_IMPORT_ERROR is not None:
        ensure_torchdata(err=_TORCHDATA_IMPORT_ERROR, context="stnet.data.pipeline")
    if _NODES_IMPORT_ERROR is not None:
        raise ImportError(
            f"stnet.data.pipeline: data-pipeline components require torchdata>={MIN_TORCHDATA_VERSION} (and tensordict). "
            f"Install/upgrade: pip install -U 'torchdata>={MIN_TORCHDATA_VERSION}'."
        ) from _NODES_IMPORT_ERROR


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

    per_sample = int(getattr(Sampler, "_per_sample_mem_bytes", 0) or 0)
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
            _wp = worker_policy if isinstance(worker_policy, WorkerPolicy) else WorkerPolicy.autotune()
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
    # Soft slack for auto-derived budgets (only used when budgets are unset).
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

    # Cache probe results to optionally reuse later for initial H2D med estimate.
    probe_bs_cache: Optional[int] = None
    med_probe_cache: Optional[float] = None
    # Hint for choosing the initial B in the subsequent auto-batch loop.
    b_init_hint: Optional[int] = None

    # If budgets were not explicitly configured, derive a conservative cap.
    # We cap the *batch size* in samples by estimating how many samples are needed
    # to keep the pipeline fed (inflight batches), then converting to bytes.
    if (
        tpl.device_budget_max_bytes is None or tpl.host_budget_max_bytes is None
    ) and int(tpl.sample_bytes or 0) > 0:
        try:
            inflight = int(tpl.host_inflight_batches_per_proc())
            lw = max(1, int(getattr(tpl, "local_world_size", 1) or 1))
            # Derive a per-batch sample target using a quick H2D profile.
            #
            # Goal: keep per-batch H2D transfer time within a reasonable range
            # (same objective as the later auto-batch loop that targets [_tmin_ms, _tmax_ms]),
            # without hard-coding caps like 8/16GB or magic sample counts.
            #
            # We keep the probe cheap: few steps, minimal warmup.
            target_ms = float(
                max(
                    float(_tmin_ms),
                    min(float(_tmax_ms), 0.5 * (float(_tmin_ms) + float(_tmax_ms))),
                )
            )
            probe_bs = max(1, min(int(B_cap), 64))
            med_probe = 0.0
            with contextlib.suppress(Exception):
                med_probe = float(
                    _h2d_counter(x_cpu, y_cpu, _dev, probe_bs, _steps=4, _warmup=1)
                )

            # Validate probe result (avoid NaN/inf/<=0).
            if (med_probe is not None) and (isinstance(med_probe, (float, int))) and math.isfinite(float(med_probe)) and float(med_probe) > 0.0:
                # Assume near-linear scaling: ms ~ bs. Choose bs so ms ~= target_ms.
                # bs_est = ceil(target_ms * probe_bs / med_probe)
                bs_est = int(math.ceil((target_ms * float(probe_bs)) / float(med_probe)))
                target_batch_samples = max(1, min(int(B_cap), bs_est))
                probe_bs_cache = int(probe_bs)
                med_probe_cache = float(med_probe)
                b_init_hint = int(target_batch_samples)
            else:
                # Fallback to a byte-based target (aim for ~64MiB per transfer).
                target_bytes = 64 * 1024 * 1024
                target_batch_samples = max(
                    1,
                    min(int(B_cap), int(target_bytes // max(1, int(tpl.sample_bytes)))),
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
                base_host = int(tpl.host_sample_bytes) * max(1, inflight) * max(
                    1, lw
                ) * int(target_batch_samples)
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

    # IMPORTANT:
    # - No global/per-function scaling here anymore.
    # - We compute an upper cap once and store it on the Sampler.
    #   Runtime scaling is applied during actual sampling (dynamic batch).
    B_cap = max(1, min(int(B_cap), len(_ds)))
    with contextlib.suppress(Exception):
        setattr(_ds, "_S_B_cap", int(B_cap))

    env_max = os.environ.get("STNET_MAX_BATCH_SIZE") or os.environ.get("STNET_MAX_BATCH")
    try:
        if env_max:
            B_cap = max(1, min(B_cap, int(env_max)))
    except Exception:
        pass

    # Update cap after env clamp as well.
    with contextlib.suppress(Exception):
        setattr(_ds, "_S_B_cap", int(B_cap))

    candidates = _random_batches(sbytes, _dev, len(_ds))
    if candidates:
        B = min(candidates[-1], B_cap)
    else:
        B = min(64, B_cap)

    # Pull the initial B toward the H2D-derived hint when available to reduce
    # the number of doubling/halving iterations below.
    # Compromise: instead of hard-overriding B, snap toward the hint using the
    # candidate set (which encodes coarse memory/heuristic constraints).
    if b_init_hint is not None:
        try:
            B_hint = max(1, min(int(B_cap), int(b_init_hint)))
            if candidates:
                # Keep only usable candidates under current cap.
                cands = [
                    int(c)
                    for c in candidates
                    if isinstance(c, int) and c > 0 and int(c) <= int(B_cap)
                ]
                if cands:
                    # Prefer the largest candidate <= hint (avoids starting above
                    # the H2D target and immediately halving).
                    le = [c for c in cands if int(c) <= int(B_hint)]
                    if le:
                        B = int(max(le))
                    else:
                        # All candidates are above hint; choose the smallest candidate.
                        B = int(min(cands))
                else:
                    B = int(B_hint)
            else:
                B = int(B_hint)
        except Exception:
            pass

    # Reuse probe result when possible to avoid redundant H2D measurement.
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
    sampler_scale: Optional["SamplerScale"] = None,
    **kwargs: Any,
) -> "Sampler":
    _require_nodes()
    if not isinstance(source, Mapping):
        raise TypeError(f"dataset expects a Source mapping, got {type(source)}")

    format = source.get("format")
    if format is None:
        format = source.get("kind")
    if format is None:
        raise ValueError("Source['format'] or Source['kind'] must be provided")
    format = str(format)
    if format != "memmap":
        raise ValueError(f"Unsupported source format: {format!r}")
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
        features = features.reshape(features.shape[0], -1)
    if labels_dtype is not None and isinstance(labels, torch.Tensor):
        labels = labels.to(dtype=labels_dtype, non_blocking=True, copy=False)
    if sanitize and torch.is_floating_point(labels):
        torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0, out=labels)
    out: Dict[str, Any] = {"X": features, "Y": labels}
    if isinstance(batch, Mapping) and "row_ids" in batch:
        out["row_ids"] = batch.get("row_ids")
    return out


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
            if "row_ids" in conv:
                stacked["row_ids"] = conv["row_ids"]
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
            # Optional stable row ids (used for inference result materialization).
            row_ids = None
            try:
                rids = [elem.get("row_ids") for elem in batch]
                if all(r is not None for r in rids):
                    parts = []
                    for r in rids:
                        rt = r if isinstance(r, torch.Tensor) else torch.as_tensor(r)
                        parts.append(rt.reshape(-1))
                    row_ids = torch.cat(parts, dim=0)
            except Exception:
                row_ids = None

            data = {"X": Xs, "Y": Ys, "features": Xs, "labels": Ys}
            if row_ids is not None:
                data["row_ids"] = row_ids
            if TensorDict is None:
                return data
            return TensorDict(data, batch_size=[])
        return batch
    if isinstance(batch, Mapping):
        try:
            conv = converter(batch)
        except Exception:
            conv = batch
        X = conv.get("X", batch.get("X"))
        Y = conv.get("Y", batch.get("Y"))
        row_ids = conv.get("row_ids", batch.get("row_ids"))
        data = {"X": X, "Y": Y, "features": X, "labels": Y}
        if row_ids is not None:
            data["row_ids"] = row_ids
        if TensorDict is None:
            return data
        return TensorDict(data, batch_size=[])
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
    sampler = Wrapper(stop_criteria="ALL_DATASETS_EXHAUSTED", seed=int(seed), weights=mx_weights)
    source = sampler.compose(node_or_nodes)
    # If multi-source mixing is active, register the Wrapper for per-epoch reseeding.
    if epochables is not None and getattr(sampler, "_node", None) is not None:
        with contextlib.suppress(Exception):
            # Put it first so it runs before per-dataset epoch hooks.
            epochables.insert(0, sampler)
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
    sampler_scale: Optional["SamplerScale"] = None,
) -> Dict[str, Any]:
    _require_nodes()
    device_obj = (
        torch.device(device) if not isinstance(device, torch.device) else device
    )
    _wp = worker_policy if isinstance(worker_policy, WorkerPolicy) else WorkerPolicy.autotune()
    _wp.apply_torch_threads()
    io_workers = int(_wp.num_workers)
    prebatch = int(_wp.prebatch)
    pf_depth = int(_wp.prefetch_factor)
    pf_depth_fixed = max(1, int(pf_depth))

    map_fn = partial(
        collate,
        labels_dtype=labels_dtype,
        sanitize=sanitize,
        flatten_features=flatten_features,
    )

    allocated = Disposable()
    batch_size: Optional[int] = None
    scale_ctl = sampler_scale if sampler_scale is not None else SamplerScale()
    # Objects that support per-epoch reseeding via .set_epoch(epoch).
    # We attach these to the training loader so the runtime can call set_epoch()
    # at the start of every epoch (mirrors DistributedSampler behavior).
    train_epochables: list[Any] = []

    def _stream_batch(_ds: Sampler, _dev: torch.device) -> Tuple[int, float]:
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

    def _rescale_batch(_datasets: Mapping[str, Sampler], _bs: int) -> int:
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
        _device_obj: torch.device, _datasets: Mapping[str, Sampler], _pf: int, _bs: int
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
            with contextlib.suppress(Exception):
                lp = LoaderPolicy()
                hard = int(lp.hard_inflight_batches(_device_obj))
                soft_cap = max(1, int(hard * max(1, int(lp.soft_cap_multiplier))))
                pb = max(1, int(prebatch))
                workers = max(1, int(io_workers))
                inflight_pf_cap = max(1, int((soft_cap - pb) // max(1, workers)))
                pf_cap = min(int(pf_cap), int(inflight_pf_cap))
            return int(max(1, min(int(_pf), int(pf_cap), 8)))
        except Exception:
            return int(_pf)

    _device_obj = device_obj
    _auto_bs_candidates: list[int] = []
    _auto_ms_candidates: list[float] = []

    if isinstance(sources, Mapping) and not _is_source_spec(sources):
        datasets: Dict[str, Any] = {}
        for key, spec in sources.items():
            ds = dataset(spec, split="train", val_frac=float(val_frac), sampler_scale=scale_ctl)
            allocated.add(ds)
            train_epochables.append(ds)
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
                pf_depth = int(max(1, min(8, pf_depth)))
                pf_depth = int(pf_depth_fixed)
                pf_depth = _cap_pf_depth(_device_obj, datasets, pf_depth, batch_size)
                pf_depth = int(max(1, min(int(pf_depth), int(pf_depth_fixed))))
                if int(pf_depth) != int(pf_depth_before):
                    batch_size = _rescale_batch(datasets, int(batch_size))
            else:
                batch_size = 1
            sampler_nodes: Dict[str, BaseNode] = {}
            lengths: Dict[str, int] = {}
            for key, ds in datasets.items():
                sampler_node = ds.compose(
                    batch_size=int(batch_size),
                    shuffle=bool(train_shuffle),
                    seed=int(seed),
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
            seed=int(seed),
            epochables=train_epochables,
        )
        train_length = sum(lengths.values()) if lengths else None

    elif isinstance(sources, (list, tuple)):
        datasets: Dict[str, Any] = {}
        for i, spec in enumerate(sources):
            key = str(i)
            ds = dataset(spec, split="train", val_frac=float(val_frac), sampler_scale=scale_ctl)
            allocated.add(ds)
            train_epochables.append(ds)
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
                pf_depth = int(max(1, min(8, pf_depth)))
                pf_depth = int(pf_depth_fixed)
                pf_depth = _cap_pf_depth(_device_obj, datasets, pf_depth, batch_size)
                pf_depth = int(max(1, min(int(pf_depth), int(pf_depth_fixed))))
                if int(pf_depth) != int(pf_depth_before):
                    batch_size = _rescale_batch(datasets, int(batch_size))
            else:
                batch_size = 1
        sampler_list: list[BaseNode] = []
        lengths: list[int] = []
        for key, ds in datasets.items():
            sampler_node = ds.compose(
                batch_size=int(batch_size),
                shuffle=bool(train_shuffle),
                seed=int(seed),
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
            seed=int(seed),
            epochables=train_epochables,
        )
        train_length = sum(lengths) if lengths else None
    else:
        ds = dataset(sources, split="train", val_frac=float(val_frac), sampler_scale=scale_ctl)
        allocated.add(ds)
        train_epochables.append(ds)
        if batch_size is None or int(batch_size) <= 0:
            B_i, ms_i = _stream_batch(ds, _device_obj)
            batch_size = max(1, int(B_i) if B_i > 0 else 1)
            pf_depth_before = int(pf_depth)
            if int(pf_depth) != int(pf_depth_before):
                batch_size = max(
                    1, int(_stream_batch(ds, _device_obj)[0]) if len(ds) > 0 else 1
                )
        sampler_node = ds.compose(
            batch_size=int(batch_size),
            shuffle=bool(train_shuffle),
            seed=int(seed),
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
            seed=int(seed),
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
                ds = dataset(spec, split="val", val_frac=float(val_frac), sampler_scale=scale_ctl)
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
                    pf_depth = int(max(1, min(8, pf_depth)))
                    pf_depth = int(pf_depth_fixed)
                    pf_depth = _cap_pf_depth(_device_obj, datasets, pf_depth, batch_size)
                    pf_depth = int(max(1, min(int(pf_depth), int(pf_depth_fixed))))
                    if int(pf_depth) != int(pf_depth_before):
                        batch_size = _rescale_batch(datasets, int(batch_size))
            sampler_nodes: Dict[str, BaseNode] = {}
            lengths: Dict[str, int] = {}
            for key, ds in datasets.items():
                sn = ds.compose(
                    batch_size=int(batch_size),
                    shuffle=False,
                    seed=int(seed),
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
                seed=int(seed),
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
                ds = dataset(spec, split="val", val_frac=float(val_frac), sampler_scale=scale_ctl)
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
                    pf_depth = int(max(1, min(8, pf_depth)))
                    pf_depth = int(pf_depth_fixed)
                    pf_depth = _cap_pf_depth(_device_obj, datasets, pf_depth, batch_size)
                    pf_depth = int(max(1, min(int(pf_depth), int(pf_depth_fixed))))
                    if int(pf_depth) != int(pf_depth_before):
                        batch_size = _rescale_batch(datasets, int(batch_size))
            sampler_list: list[BaseNode] = []
            lengths: list[int] = []
            for k, ds in datasets.items():
                sn = ds.compose(
                    batch_size=int(batch_size),
                    shuffle=False,
                    seed=int(seed),
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
                seed=int(seed),
            )
            val_loader = Loader.compose(
                mapped_val,
                device=device_obj,
                prefetch_factor=int(pf_depth),
                non_blocking=bool(non_blocking_copy),
                length=sum(lengths) if lengths else None,
            )

        else:
            ds = dataset(sources, split="val", val_frac=float(val_frac), sampler_scale=scale_ctl)
            allocated.add(ds)
            if batch_size is None or int(batch_size) <= 0:
                B_i, ms_i = _stream_batch(ds, _device_obj)
                batch_size = max(1, int(B_i) if B_i > 0 else 1)
                pf_depth_before = int(pf_depth)
                if int(pf_depth) != int(pf_depth_before):
                    batch_size = max(
                        1,
                        int(_stream_batch(ds, _device_obj)[0]) if len(ds) > 0 else 1,
                    )
            sampler_node = ds.compose(
                batch_size=int(batch_size),
                shuffle=False,
                seed=int(seed),
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
                seed=int(seed),
            )
            val_loader = Loader.compose(
                mapped_val,
                device=device_obj,
                prefetch_factor=int(pf_depth),
                non_blocking=bool(non_blocking_copy),
                length=len(ds),
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
    sampler_scale: Optional["SamplerScale"] = None

    _opened: bool = False

    def open(
        self,
        *,
        train_state: Optional[Dict[str, Any]] = None,
        val_state: Optional[Dict[str, Any]] = None,
    ) -> "Session":
        # Ensure torchdata/tensordict pipeline components are available BEFORE
        # constructing SamplerScale. Otherwise SamplerScale may be None due to
        # deferred import failure handling and we'd raise a confusing TypeError.
        _require_nodes()
        if self.sampler_scale is None:
            self.sampler_scale = SamplerScale()

        dev = torch.device(self.device) if not isinstance(self.device, torch.device) else self.device

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
        )

        train_loader = dl.get("training_loader")
        val_loader = dl.get("validation_loader")
        self.disposable = dl.get("disposable")

        self.raw_training_loader = train_loader
        self.raw_validation_loader = val_loader

        if train_state and train_loader is not None and hasattr(train_loader, "load_state_dict"):
            try:
                train_loader.load_state_dict(train_state)
            except Exception:
                pass
        if val_state and val_loader is not None and hasattr(val_loader, "load_state_dict"):
            try:
                val_loader.load_state_dict(val_state)
            except Exception:
                pass

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

        # Propagate scale controller to (wrapped) loaders as well.
        with contextlib.suppress(Exception):
            if self.training_loader is not None:
                setattr(self.training_loader, "_stnet_sampler_scale", self.sampler_scale)
            if self.validation_loader is not None:
                setattr(self.validation_loader, "_stnet_sampler_scale", self.sampler_scale)

        # Propagate per-epoch epochables (samplers) to the wrapped training loader.
        # This enables the runtime to call .set_epoch(epoch) at the beginning of each epoch,
        # ensuring shuffle changes across epochs (mirrors DistributedSampler best practice).
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
            try:
                keep.cleanup()
            except Exception:
                pass
        self._opened = False

    def __enter__(self) -> "Session":
        return self.open()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


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

    def ensure_device_info(self) -> "Dataset[TExtra]":
        self._refresh_device_info()
        return self

    def refresh(self) -> "Dataset[TExtra]":
        self._refresh_device_info()
        self._refresh_dtypes_from_env()
        self._refresh_quant_from_env()

        if not self.float_dtypes:
            floats: list[torch.dtype] = [torch.float32]
            if self.device_type == "cuda" and torch.cuda.is_available():
                floats.insert(0, torch.float16)
                if self.is_cuda_bf16_supported():
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
        self, data: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Sequence[Any], Tuple[int, ...]]:
        def _to_tensor(obj: Any, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
            t = obj if isinstance(obj, torch.Tensor) else torch.as_tensor(obj)
            if dtype is not None:
                with contextlib.suppress(Exception):
                    t = t.to(dtype=dtype)
            return t

        features: Optional[torch.Tensor] = None
        labels: Optional[torch.Tensor] = None
        keys: Sequence[Any] = ()

        def _pick(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
            for k in keys:
                value = mapping.get(k, None)
                if value is not None:
                    return value
            return None

        if isinstance(data, TensorDictBase):
            features = _pick(data, ("features", "X"))
            labels = _pick(data, ("labels", "Y", "targets", "target"))
            keys = _pick(data, ("row_ids", "keys")) or ()
            if features is None:
                for v in data.values():
                    if isinstance(v, torch.Tensor):
                        features = v
                        break
        elif isinstance(data, Mapping):
            if "X" in data or "features" in data:
                features = _pick(data, ("X", "features"))
                labels = _pick(data, ("Y", "labels", "targets", "target"))
                keys = _pick(data, ("row_ids", "keys")) or ()
            else:
                items = list(data.items())
                keys = [k for (k, _v) in items]
                values = [_v for (_k, _v) in items]

                if values and isinstance(values[0], (list, tuple)) and len(values[0]) >= 2:
                    feat_list = [_to_tensor(v[0], dtype=self.feature_dtype) for v in values]
                    label_list = [_to_tensor(v[1], dtype=self.label_float_dtype) for v in values]
                    features = torch.stack(feat_list) if feat_list else None
                    labels = torch.stack(label_list) if label_list else None
                else:
                    parsed = False

                    def _key_feature_size(obj: Any) -> Optional[int]:
                        if isinstance(obj, torch.Tensor):
                            if obj.ndim == 0:
                                return 1
                            try:
                                return int(obj.numel())
                            except Exception:
                                return None
                        if isinstance(obj, (tuple, list)):
                            try:
                                return int(len(obj))
                            except Exception:
                                return None
                        return None

                    if keys and values:
                        ksize0 = _key_feature_size(keys[0])
                        if ksize0 is not None and 1 < int(ksize0) <= 64:
                            if all(_key_feature_size(k) == ksize0 for k in keys):
                                # Fast-path: interpret mapping keys as feature vectors.
                                # When values contain None (common for prediction), treat labels as missing.
                                has_missing_labels = any((v is None for v in values))
                                try:
                                    feat_list = [
                                        _to_tensor(k, dtype=self.feature_dtype).reshape(-1) for k in keys
                                    ]
                                    features = torch.stack(feat_list, dim=0) if feat_list else None
                                    if has_missing_labels:
                                        labels = None
                                        parsed = features is not None
                                    else:
                                        label_list = [
                                            _to_tensor(v, dtype=self.label_float_dtype) for v in values
                                        ]
                                        labels = torch.stack(label_list, dim=0) if label_list else None
                                        parsed = features is not None and labels is not None
                                except Exception:
                                    parsed = False

                    if not parsed:
                        features = torch.stack([_to_tensor(v, dtype=self.feature_dtype) for v in values])
                        labels = None

        if features is None:
            raise ValueError("Dataset.preprocess: unable to locate feature tensor(s)")

        features = _to_tensor(features, dtype=self.feature_dtype).contiguous()
        if features.ndim == 0:
            features = features.reshape(1, 1)
        elif features.ndim == 1:
            # Interpret a 1D tensor as a single sample with (1, in_dim)
            # rather than (N, 1). This aligns with the model reference
            # convention: features are shaped (B, in_dim).
            features = features.reshape(1, -1)

        if labels is not None:
            labels = _to_tensor(labels, dtype=self.label_float_dtype).contiguous()
            if labels.ndim == 0:
                labels = labels.reshape(1, 1)
            if labels.shape[0] != features.shape[0]:
                labels = labels.reshape(features.shape[0], -1)
            label_shape = tuple(labels.shape[1:])
        else:
            label_shape = tuple()

        if isinstance(keys, torch.Tensor):
            with contextlib.suppress(Exception):
                keys = keys.detach().cpu().tolist()
        elif keys is None:
            keys = ()

        if not keys:
            keys = range(int(features.shape[0]))

        return features, labels, keys, label_shape

    # ---- Precision negotiation helpers
    @property
    def scale_min_abs(self) -> Optional[float]:
        """Backward-compatible alias for scale_min_positive.

        Older meta.json used scale_min_abs; internally we treat it as the
        minimum *positive* magnitude excluding zeros.
        """
        return self.scale_min_positive

    @staticmethod
    def tensor_scale_stats(t: torch.Tensor) -> Dict[str, Any]:
        """Compute scale statistics used for safe downcasting decisions.

        Returns a dict with keys:
        - has_scale (bool)
        - has_nonfinite (bool)
        - scale_max_abs (float|None)
        - scale_min_value (float|int|None)
        - scale_max_value (float|int|None)
        - scale_min_positive (float|None) # smallest abs(value)>0 among finite entries
        - scale_is_integral (bool|None)
        """
        if not isinstance(t, torch.Tensor):
            raise TypeError('tensor_scale_stats expects a torch.Tensor')
        if t.numel() == 0:
            return {
                'has_scale': False,
                'has_nonfinite': False,
                'scale_max_abs': None,
                'scale_min_value': None,
                'scale_max_value': None,
                'scale_min_positive': None,
                'scale_is_integral': None,
            }
        # robust complex detection across torch versions
        is_complex = False
        _is_complex_fn = getattr(torch, "is_complex", None)
        if callable(_is_complex_fn):
            try:
                is_complex = bool(_is_complex_fn(t))
            except Exception:
                is_complex = False
        else:
            # fallback for very old builds: Tensor.is_complex may be property or method
            v = getattr(t, "is_complex", False)
            try:
                is_complex = bool(v() if callable(v) else v)
            except Exception:
                is_complex = False
        if is_complex:
            return Dataset.tensor_scale_stats(t.detach().abs())
        if t.is_floating_point():
            x = t.detach()
            finite = torch.isfinite(x)
            has_nonfinite = bool((~finite).any().item())
            if finite.any().item():
                xf = x[finite]
                # Track signed min/max (useful for int-range negotiation and diagnostics).
                try:
                    min_val = float(xf.min().item()) if xf.numel() else None
                    max_val = float(xf.max().item()) if xf.numel() else None
                except Exception:
                    min_val, max_val = (None, None)

                absf = xf.abs()
                max_abs = float(absf.max().item()) if absf.numel() else 0.0
                nonzero = absf > 0
                if nonzero.any().item():
                    min_pos = float(absf[nonzero].min().item())
                else:
                    min_pos = None
            else:
                max_abs = float('nan')
                min_val = None
                max_val = None
                min_pos = None
            return {
                'has_scale': True,
                'has_nonfinite': has_nonfinite,
                'scale_max_abs': max_abs,
                'scale_min_value': min_val,
                'scale_max_value': max_val,
                'scale_min_positive': min_pos,
                'scale_is_integral': None,
            }
        # integer / bool
        x = t.detach()
        if x.dtype == torch.bool:
            x_i64 = x.to(dtype=torch.int64)
        else:
            # Keep integer values exact for boundary checks; avoid abs() overflow on signed min.
            x_i64 = x.to(dtype=torch.int64) if x.dtype != torch.int64 else x

        try:
            min_i = int(x_i64.min().item()) if x_i64.numel() else 0
            max_i = int(x_i64.max().item()) if x_i64.numel() else 0
        except Exception:
            # Extremely defensive: fall back to Python list path (slow, but only for stats).
            vals = x_i64.detach().cpu().reshape(-1).tolist()
            min_i = int(min(vals)) if vals else 0
            max_i = int(max(vals)) if vals else 0

        # abs() overflow-safe: compute max(|min|, |max|) in Python int space.
        max_abs = float(max(abs(min_i), abs(max_i)))

        # For integral stats, subnormal/underflow is not meaningful; keep min_positive=None.
        min_pos = None
        is_integral = True
        return {
            'has_scale': True,
            'has_nonfinite': False,
            'scale_max_abs': max_abs,
            'scale_min_value': min_i,
            'scale_max_value': max_i,
            'scale_min_positive': min_pos,
            'scale_is_integral': is_integral,
        }

    @staticmethod
    def merge_scale_stats(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
        """Merge two scale stats dicts conservatively (worst-case).

        - max_abs takes max
        - min_value takes min of non-None
        - max_value takes max of non-None
        - min_positive takes min of non-None
        - has_nonfinite OR
        - scale_is_integral AND when both specified
        """
        out: Dict[str, Any] = {}
        out['has_scale'] = bool(a.get('has_scale') or b.get('has_scale'))
        out['has_nonfinite'] = bool(a.get('has_nonfinite') or b.get('has_nonfinite'))

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

        out['scale_max_abs'] = _max(a.get('scale_max_abs'), b.get('scale_max_abs'))
        out['scale_min_value'] = _min_num(a.get('scale_min_value'), b.get('scale_min_value'))
        out['scale_max_value'] = _max(a.get('scale_max_value'), b.get('scale_max_value'))
        out['scale_min_positive'] = _min_pos(a.get('scale_min_positive'), b.get('scale_min_positive'))

        ia = a.get('scale_is_integral')
        ib = b.get('scale_is_integral')
        if ia is None:
            out['scale_is_integral'] = ib
        elif ib is None:
            out['scale_is_integral'] = ia
        else:
            out['scale_is_integral'] = bool(ia) and bool(ib)
        return out

    @classmethod
    def is_fp32_castable(
        cls,
        stats: Mapping[str, Any],
        *,
        underflow_action: Optional[str] = None,
        safety_margin: float = 1.0,
    ) -> bool:
        """Return True if data can be safely represented in FP32 without overflow/NaN/Inf.

        If underflow_action == 'forbid', also rejects values smaller than fp32.tiny
        (after applying safety_margin).
        """
        if not stats.get('has_scale'):
            return True
        if bool(stats.get('has_nonfinite')):
            return False
        max_abs = stats.get('scale_max_abs')
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
        action = normalize_underflow_action(underflow_action, default=default_underflow_action())
        if action == 'forbid':
            min_pos = stats.get('scale_min_positive')
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
        """Populate this Dataset's scale metadata from a stats dict.

        This is intended to be called from memmap writers (nodes/run) or
        from runtime after reading meta.json.
        """
        self.has_scale = bool(stats.get('has_scale') or False)
        self.has_nonfinite = bool(stats.get('has_nonfinite') or False)
        self.scale_max_abs = (float(stats['scale_max_abs']) if stats.get('scale_max_abs') is not None else None)
        self.scale_min_value = stats.get('scale_min_value')
        self.scale_max_value = stats.get('scale_max_value')
        self.scale_min_positive = (float(stats['scale_min_positive']) if stats.get('scale_min_positive') is not None else None)
        self.scale_is_integral = (bool(stats['scale_is_integral']) if stats.get('scale_is_integral') is not None else None)
        # If float scale present, negotiable means safe to cast to fp32
        self.is_negotiable = bool(
            self.has_scale
            and (not self.has_nonfinite)
            and self.is_fp32_castable(stats, underflow_action=self.underflow_action, safety_margin=1.0)
        )

    def batch_to_device(
        self, batch: Any, device: Union[str, torch.device], non_blocking: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        dev = torch.device(device)
        feats, labels, _, _ = self.preprocess(batch)
        feats = feats.to(device=dev, non_blocking=non_blocking)
        if labels is not None:
            labels = labels.to(device=dev, non_blocking=non_blocking)
        return feats, labels

    def _refresh_device_info(self) -> None:
        dev = torch.device(self.device)
        self.device = dev
        self.device_type = dev.type
        if dev.type == "cuda" and torch.cuda.is_available():
            major, minor = self.cuda_compute_capability(dev)
            if major > 0 or minor > 0:
                self.cuda_cc = (int(major), int(minor))
            else:
                self.cuda_cc = None
        else:
            self.cuda_cc = None

    @staticmethod
    def cuda_compute_capability(device: Union[torch.device, str]) -> Tuple[int, int]:
        dev = torch.device(device)
        if dev.type != "cuda" or not torch.cuda.is_available():
            return (0, 0)
        try:
            major, minor = torch.cuda.get_device_capability(dev)
        except Exception:
            return (0, 0)
        return (int(major), int(minor))

    @staticmethod
    def is_cpu_bf16_supported() -> bool:
        try:
            mkldnn = getattr(torch.backends, "mkldnn", None)
            if mkldnn is None:
                return False
            if not bool(mkldnn.is_available()) or not bool(getattr(mkldnn, "enabled", True)):
                return False
            mkldnn_ops = getattr(torch.ops, "mkldnn", None)
            f = getattr(mkldnn_ops, "_is_mkldnn_bf16_supported", None) if mkldnn_ops is not None else None
            if callable(f):
                return bool(f())
        except Exception:
            return False
        return False

    @staticmethod
    def is_cuda_bf16_supported() -> bool:
        try:
            if not torch.cuda.is_available():
                return False
            f = getattr(torch.cuda, "is_bf16_supported", None)
            if callable(f):
                try:
                    return bool(f(including_emulation=False))
                except TypeError:
                    return bool(f())
            major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
            return major >= 8
        except Exception:
            return False

    @staticmethod
    def _resolve_device(device: Optional[Union[torch.device, str]]) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _refresh_dtypes_from_env(self) -> None:
        def _parse_dtypes(value: str) -> Tuple[torch.dtype, ...]:
            entries = []
            for token in value.split(","):
                name = token.strip()
                if not name:
                    continue
                dtype = getattr(torch, name, None)
                if isinstance(dtype, torch.dtype):
                    entries.append(dtype)
            return tuple(entries)

        float_env = os.environ.get("STNET_DATA_FLOAT_DTYPES") or os.environ.get(
            "STNET_FLOAT_DTYPES"
        )
        if float_env:
            parsed = _parse_dtypes(float_env)
            if parsed:
                self.float_dtypes = parsed

        int_env = os.environ.get("STNET_DATA_INT_DTYPES") or os.environ.get(
            "STNET_INT_DTYPES"
        )
        if int_env:
            parsed = _parse_dtypes(int_env)
            if parsed:
                self.int_dtypes = parsed

    def _refresh_quant_from_env(self) -> None:
        env_bits = os.environ.get("STNET_DATA_INT_QUANT_BITS") or os.environ.get(
            "STNET_INT_QUANT_BITS"
        )
        if env_bits:
            env_bits = env_bits.strip()
            try:
                bits = int(env_bits)
                if bits > 0:
                    self.int_quant_bits = bits
            except Exception:
                pass

    @classmethod
    def is_float8_supported(
        cls, device: Optional[Union[torch.device, str]] = None
    ) -> Tuple[bool, str]:
        try:
            dev = cls._resolve_device(device)
            if dev.type == "cuda" and torch.cuda.is_available():
                try:
                    import torch.cuda.amp as _tca

                    with contextlib.suppress(Exception):
                        if getattr(_tca, "is_float8_available", None) is not None:
                            ok, reason = _tca.is_float8_available()
                            return (bool(ok), str(reason))
                except Exception:
                    pass
                major, minor = cls.cuda_compute_capability(dev)
                if major >= 9:
                    return (True, "Hopper+ supports FP8")
                return (False, "FP8 requires sm90+")
            return (False, "FP8 requires CUDA sm90+ and torch.cuda")
        except Exception:
            return (False, "Unknown float8 support")

    @classmethod
    def is_int8_supported(cls, device: Optional[Union[torch.device, str]] = None) -> Tuple[bool, str]:
        try:
            dev = cls._resolve_device(device)
            if dev.type == "cuda" and torch.cuda.is_available():
                return (True, "Int8 supported on CUDA")
            return (True, "Int8 supported on CPU")
        except Exception:
            return (False, "Unknown int8 support")

    @classmethod
    def is_int4_supported(cls, device: Optional[Union[torch.device, str]] = None) -> Tuple[bool, str]:
        try:
            dev = cls._resolve_device(device)
            if dev.type == "cuda" and torch.cuda.is_available():
                return (True, "Int4 supported on CUDA")
            return (True, "Int4 supported on CPU")
        except Exception:
            return (False, "Unknown int4 support")

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
        tpl = replace(
            cls(
                device=dev,
                float_dtypes=float_dtypes_seq,
                int_dtypes=int_dtypes_seq,
                feature_dtype=feature_dtype,
                label_float_dtype=label_float_dtype,
                extra=extra_map,
            )
        )
        return tpl
