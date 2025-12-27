# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import ctypes
import gc
import importlib
import itertools
import logging
import math
import multiprocessing
import os
import platform
import sys
import sysconfig
import threading
import time
from dataclasses import dataclass, replace
from datetime import timezone, tzinfo
from pathlib import Path
from threading import Lock
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
import torch.multiprocessing as mp

from .casting import env_bool, env_first, env_first_float, env_first_int, env_float, env_str, parse_bool

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None


_RUNTIME_CFG = SimpleNamespace(
    deterministic=False,
    allow_tf32=None,
    cudnn_benchmark=None,
    matmul_precision=None,
    sdpa_backends=None,
    te_first=True,
)
_RUNTIME_CFG_LOCK = threading.Lock()

# fp32_precision is (effectively) process-global; keep a small cache to avoid
# repeated writes in tight loops.
_FP32_PRECISION_CACHE: dict[str, str] = {}
_FP32_PRECISION_LOCK = threading.Lock()

_EMPTY_CACHE_LOCK = threading.Lock()
_EMPTY_CACHE_LAST_CALL_S_BY_DEVICE: dict[Tuple[str, int], float] = {}

_LOGGER = logging.getLogger(__name__)


# Torch thread configuration is process-global. In particular,
# torch.set_num_interop_threads() can only be called once per process.
_TORCH_THREAD_CFG_LOCK = threading.Lock()
_TORCH_NUM_THREADS_SET: Optional[int] = None
_TORCH_INTEROP_THREADS_SET: Optional[int] = None
_TORCH_INTEROP_LOCKED: bool = False


def _log_info(logger: Optional[Any], msg: str) -> None:
    """Best-effort info logging with an optional user-provided logger."""
    if logger is None:
        _LOGGER.info(msg)
        return
    try:
        if callable(logger):
            logger(msg)
        elif hasattr(logger, "info"):
            logger.info(msg)
        else:
            _LOGGER.info(msg)
    except Exception:
        _LOGGER.info(msg)


def _log_debug(logger: Optional[Any], msg: str) -> None:
    """Best-effort debug logging with an optional user-provided logger."""
    if logger is None:
        _LOGGER.debug(msg)
        return
    try:
        if callable(logger):
            logger(msg)
        elif hasattr(logger, "debug"):
            logger.debug(msg)
        else:
            _LOGGER.debug(msg)
    except Exception:
        _LOGGER.debug(msg)


def _empty_cache_device_key(
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[str, int]:
    """Return a stable (device_type, device_index) key for rate limiting.

    If device is None, returns a dedicated ('all', -1) key (meaning: global cache clear).
    If device is provided without an index, best-effort uses the backend's current device.
    """
    if device is None:
        return ("all", -1)

    try:
        dev = device if isinstance(device, torch.device) else torch.device(str(device))
    except (TypeError, ValueError, RuntimeError):
        return ("all", -1)

    idx = int(dev.index) if dev.index is not None else -1

    # Best-effort fill-in for "cuda"/"xpu" when index is omitted.
    if dev.type == "cuda" and idx < 0:
        try:
            if torch.cuda.is_available():
                idx = int(torch.cuda.current_device())
        except (RuntimeError, TypeError, ValueError):
            idx = -1
    elif dev.type == "xpu" and idx < 0:
        try:
            xpu = getattr(torch, "xpu", None)
            cur = getattr(xpu, "current_device", None) if xpu is not None else None
            if callable(cur):
                idx = int(cur())
        except (RuntimeError, TypeError, ValueError):
            idx = -1

    return (str(dev.type), int(idx))


def _linux_cgroup_cpu_quota() -> Optional[float]:
    """Return cgroup CPU quota as a float CPU count (Linux only).

    - cgroup v2: reads cpu.max (quota/period)
    - cgroup v1: reads cpu.cfs_quota_us / cpu.cfs_period_us

    Returns None when no quota is detected (unlimited) or on errors.
    """
    if not sys.platform.startswith("linux"):
        return None

    def _read_text(path: str) -> Optional[str]:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip()
        except Exception:
            return None

    def _read_int(path: str) -> Optional[int]:
        s = _read_text(path)
        if not s or s == "max":
            return None
        try:
            return int(str(s).strip())
        except Exception:
            return None

    # cgroup v2
    try:
        root = "/sys/fs/cgroup"
        if os.path.exists(os.path.join(root, "cgroup.controllers")):
            rel = "/"
            with open("/proc/self/cgroup", "r", encoding="utf-8", errors="ignore") as fh:
                for ln in fh:
                    parts = ln.strip().split(":")
                    if len(parts) >= 3 and parts[0] == "0":
                        rel = parts[2] or "/"
                        break
            grp = os.path.join(root, rel.lstrip("/"))
            raw = _read_text(os.path.join(grp, "cpu.max"))
            if raw:
                parts = raw.split()
                if parts and parts[0] != "max":
                    quota = int(parts[0])
                    period = int(parts[1]) if len(parts) >= 2 else 100000
                    if quota > 0 and period > 0:
                        return float(quota) / float(period)
            return None
    except Exception:
        pass

    # cgroup v1
    try:
        cpu_rel: Optional[str] = None
        with open("/proc/self/cgroup", "r", encoding="utf-8", errors="ignore") as fh:
            for ln in fh:
                parts = ln.strip().split(":")
                if len(parts) >= 3:
                    ctrls = parts[1].split(",") if parts[1] else []
                    if "cpu" in ctrls:
                        cpu_rel = parts[2] or "/"
                        break
        if cpu_rel is None:
            return None

        base = None
        for cand in ("/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct", "/sys/fs/cgroup/cpuacct"):
            if os.path.isdir(cand):
                base = cand
                break
        if base is None:
            return None

        grp = os.path.join(base, cpu_rel.lstrip("/"))
        quota = _read_int(os.path.join(grp, "cpu.cfs_quota_us"))
        period = _read_int(os.path.join(grp, "cpu.cfs_period_us"))
        if quota is None or period is None:
            return None
        if int(quota) <= 0 or int(quota) == -1 or int(period) <= 0:
            return None
        return float(quota) / float(period)
    except Exception:
        return None


def _darwin_sysctl_cpu_count() -> Optional[int]:
    """Best-effort logical CPU count on macOS via sysctl.

    This is a fallback for cases where os.cpu_count() returns None.
    """
    if platform.system() != "Darwin":
        return None

    try:
        libc = ctypes.CDLL("libc.dylib")
    except Exception:
        return None

    sysctlbyname = getattr(libc, "sysctlbyname", None)
    if sysctlbyname is None:
        return None

    try:
        sysctlbyname.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        sysctlbyname.restype = ctypes.c_int
    except Exception:
        pass

    for name in (b"hw.logicalcpu", b"hw.ncpu"):
        try:
            val = ctypes.c_int(0)
            size = ctypes.c_size_t(ctypes.sizeof(val))
            ret = int(sysctlbyname(name, ctypes.byref(val), ctypes.byref(size), None, 0))
            if ret == 0:
                n = int(val.value)
                if n > 0:
                    return int(n)
        except Exception:
            continue
    return None


def accel_backend_for_device_type(device_type: Union[str, torch.device, None]) -> Optional[Any]:
    dev = str(getattr(device_type, "type", device_type) or "").lower()
    if dev == "cuda":
        return getattr(torch, "cuda", None)
    if dev == "xpu":
        return getattr(torch, "xpu", None)
    if dev == "mps":
        return getattr(torch, "mps", None)
    acc = getattr(torch, "accelerator", None)
    if acc is not None:
        dt = getattr(acc, "device_type", None)
        if dt == dev:
            return acc
    return None


def accel_is_available(device_type: Union[str, torch.device, None]) -> bool:
    backend = accel_backend_for_device_type(device_type)
    fn = getattr(backend, "is_available", None) if backend is not None else None
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            return False
    return False


def accel_device_count(device_type: Union[str, torch.device, None]) -> int:
    backend = accel_backend_for_device_type(device_type)
    fn = getattr(backend, "device_count", None) if backend is not None else None
    if callable(fn):
        try:
            return int(fn())
        except Exception:
            return 0
    return 0


def accel_current_device_index(device_type: Union[str, torch.device, None]) -> Optional[int]:
    backend = accel_backend_for_device_type(device_type)
    fn = getattr(backend, "current_device", None) if backend is not None else None
    if callable(fn):
        try:
            return int(fn())
        except Exception:
            return None
    return None


def accel_set_device_index(device_type: Union[str, torch.device, None], index: int) -> None:
    backend = accel_backend_for_device_type(device_type)
    fn = getattr(backend, "set_device", None) if backend is not None else None
    if callable(fn):
        with contextlib.suppress(Exception):
            fn(index)


def accel_manual_seed_all(device_type: Union[str, torch.device, None], seed: int) -> None:
    backend = accel_backend_for_device_type(device_type)
    fn = getattr(backend, "manual_seed_all", None) if backend is not None else None
    if callable(fn):
        with contextlib.suppress(Exception):
            fn(seed)


def accel_make_event(device: torch.device, enable_timing: bool = True) -> Optional[Any]:
    backend = accel_backend_for_device_type(getattr(device, "type", None))
    Event = getattr(backend, "Event", None) if backend is not None else None
    if Event is None:
        return None
    try:
        return Event(enable_timing=bool(enable_timing))
    except TypeError:
        with contextlib.suppress(Exception):
            return Event()
    except Exception:
        return None


def accel_stream_context(stream: Any, device_type: Union[str, torch.device, None] = None) -> contextlib.AbstractContextManager:
    backend = accel_backend_for_device_type(device_type)
    stream_ctx = getattr(backend, "stream", None) if backend is not None else None
    if callable(stream_ctx):
        try:
            return stream_ctx(stream)
        except Exception:
            return contextlib.nullcontext()
    return contextlib.nullcontext()


def accel_synchronize(device: torch.device) -> None:
    backend = accel_backend_for_device_type(getattr(device, "type", None))
    fn = getattr(backend, "synchronize", None) if backend is not None else None
    if callable(fn):
        with contextlib.suppress(Exception):
            try:
                fn(device=device)
                return
            except TypeError:
                fn(device)


def accel_timing_events_supported_for_device_type(device_type: Union[str, torch.device, None]) -> bool:
    backend = accel_backend_for_device_type(device_type)
    Event = getattr(backend, "Event", None) if backend is not None else None
    return callable(Event)


def accel_memory_allocated(device: torch.device) -> Optional[int]:
    backend = accel_backend_for_device_type(getattr(device, "type", None))
    fn = getattr(backend, "memory_allocated", None) if backend is not None else None
    if callable(fn):
        with contextlib.suppress(Exception):
            try:
                val = fn(device=device)
            except TypeError:
                val = fn(device)
            except Exception:
                val = fn()
            if val is not None:
                return max(0, int(val))
    return None


def accel_max_memory_allocated(device: torch.device) -> Optional[int]:
    backend = accel_backend_for_device_type(getattr(device, "type", None))
    fn = getattr(backend, "max_memory_allocated", None) if backend is not None else None
    if callable(fn):
        with contextlib.suppress(Exception):
            try:
                val = fn(device=device)
            except TypeError:
                val = fn(device)
            except Exception:
                val = fn()
            if val is not None:
                return max(0, int(val))
    return None


def accel_reset_peak_memory_stats(device: torch.device) -> None:
    backend = accel_backend_for_device_type(getattr(device, "type", None))
    fn = getattr(backend, "reset_peak_memory_stats", None) if backend is not None else None
    if callable(fn):
        with contextlib.suppress(Exception):
            try:
                fn(device=device)
            except TypeError:
                fn(device)
            except Exception:
                fn()


def accel_pinned_h2d_supported_for_device_type(device_type: Union[str, torch.device, None]) -> bool:
    dev = str(getattr(device_type, "type", device_type) or "").lower()
    if dev in {"cuda", "xpu"}:
        return accel_is_available(dev)
    return False


def device_mem_util_percent(device: torch.device) -> Optional[float]:
    free, total = Memory.device_mem_get_info(device)
    if free is None or total is None or total <= 0:
        return None
    used = max(0, int(total) - int(free))
    return float(used) * 100.0 / float(total) if total > 0 else None


def _windows_allowed_cpu_indices() -> Optional[list[int]]:
    """Best-effort allowed CPU indices on Windows.

    - Prefer process affinity mask (single processor group).
    - Fall back to processor-group counts and process group affinity (multi-group).

    Returned indices are *logical* indices usable by this process. For multi-group
    systems we return a contiguous [0..N) index space, and later map it to
    (group, within-group) inside the pinning routine.
    """
    if platform.system() != "Windows":
        return None

    try:
        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    except Exception:
        return None

    # 1) Process affinity mask (covers current group).
    try:
        get_proc = getattr(k32, "GetCurrentProcess", None)
        get_mask = getattr(k32, "GetProcessAffinityMask", None)
        if callable(get_proc) and callable(get_mask):
            h = get_proc()
            proc_mask = ctypes.c_size_t(0)
            sys_mask = ctypes.c_size_t(0)
            ok = int(get_mask(h, ctypes.byref(proc_mask), ctypes.byref(sys_mask)))
            if ok:
                m = int(proc_mask.value)
                if m:
                    return [i for i in range(m.bit_length()) if (m >> i) & 1]
    except Exception:
        pass

    # 2) Processor groups: build a contiguous index space for allowed groups.
    try:
        get_group_cnt = getattr(k32, "GetActiveProcessorGroupCount", None)
        get_group_procs = getattr(k32, "GetActiveProcessorCount", None)
        if not (callable(get_group_cnt) and callable(get_group_procs)):
            return None

        group_count = int(get_group_cnt())
        if group_count <= 0:
            return None

        counts: list[int] = []
        for i in range(group_count):
            c = 0
            with contextlib.suppress(Exception):
                c = int(get_group_procs(ctypes.c_ushort(int(i))))
            if not c:
                with contextlib.suppress(Exception):
                    c = int(get_group_procs(int(i)))
            counts.append(max(0, int(c)))

        # Restrict to groups this process is allowed to run on (if supported).
        groups: list[int] | None = None
        get_pg_aff = getattr(k32, "GetProcessGroupAffinity", None)
        if callable(get_pg_aff):
            try:
                h = getattr(k32, "GetCurrentProcess", lambda: None)()
                cnt = ctypes.c_ushort(0)
                arr = (ctypes.c_ushort * int(group_count))()
                ok = int(get_pg_aff(h, ctypes.byref(cnt), arr))
                if ok:
                    ng = int(cnt.value)
                    if ng > 0:
                        groups = [int(arr[i]) for i in range(ng)]
            except Exception:
                groups = None

        if not groups:
            groups = list(range(group_count))

        total = 0
        for g in groups:
            if 0 <= int(g) < group_count:
                total += int(counts[int(g)])
        if total > 0:
            return list(range(total))
    except Exception:
        return None

    return None


def get_allowed_cpus() -> list[int]:
    """Return CPU IDs usable by the current process.

    Notes:
      - On Linux, this respects affinity and cpuset restrictions.
      - On Windows, this is best-effort and may be a synthetic contiguous index space
        on systems with multiple processor groups.
    """
    # 1) Native affinity (Linux, some Unix)
    with contextlib.suppress(Exception):
        cpus = os.sched_getaffinity(0)
        if cpus:
            out = sorted({int(c) for c in cpus})
            if out:
                return out

    # 2) psutil per-process affinity (Windows/Linux)
    try:
        import psutil  # type: ignore

        proc = psutil.Process()
        fn = getattr(proc, "cpu_affinity", None)
        if callable(fn):
            cpus = fn()
            if cpus:
                out = sorted({int(c) for c in cpus})
                if out:
                    return out
    except Exception:
        pass

    # 3) Windows-specific fallbacks
    with contextlib.suppress(Exception):
        cpus = _windows_allowed_cpu_indices()
        if cpus:
            out = sorted({int(c) for c in cpus})
            if out:
                return out

    # 4) Last resort
    n: Optional[int] = None
    with contextlib.suppress(Exception):
        v = os.cpu_count()
        if isinstance(v, int) and v > 0:
            n = int(v)
    if n is None and platform.system() == "Darwin":
        with contextlib.suppress(Exception):
            v = _darwin_sysctl_cpu_count()
            if isinstance(v, int) and v > 0:
                n = int(v)
    n = int(n) if isinstance(n, int) and n > 0 else 1
    return list(range(n))


def process_cpu_count() -> int:
    """Best-effort CPU count usable by this process/thread.

    Priority order:
      1) Explicit overrides: STNET_CPU_COUNT / PYTHON_CPU_COUNT / -X cpu_count
      2) Python 3.13+: os.process_cpu_count() (respects affinity and Python overrides)
      3) os.sched_getaffinity(0) / psutil affinity / Windows groups via get_allowed_cpus()
      4) os.cpu_count()

    Linux containers note:
      - If no explicit override is provided, we additionally clamp the result by the
        cgroup CPU quota (v1/v2). This prevents severe oversubscription when the
        host CPU count is visible but the container has a smaller quota.

    The cgroup clamp uses *floor* semantics (e.g., 1.9 -> 1) and always returns >= 1.
    """
    # Explicit per-project override (highest priority).
    for key in ("STNET_CPU_COUNT", "STNET_EFFECTIVE_CPU_COUNT"):
        v = os.environ.get(key)
        if v is not None and str(v).strip():
            with contextlib.suppress(Exception):
                n = int(str(v).strip())
                if n > 0:
                    return int(n)

    # Respect Python overrides (Python 3.13+: os.process_cpu_count() also honors these).
    v = os.environ.get("PYTHON_CPU_COUNT")
    if v is not None and str(v).strip():
        with contextlib.suppress(Exception):
            n = int(str(v).strip())
            if n > 0:
                return int(n)

    with contextlib.suppress(Exception):
        xopt = getattr(sys, "_xoptions", {})
        if isinstance(xopt, dict) and "cpu_count" in xopt:
            n = int(xopt["cpu_count"])
            if n > 0:
                return int(n)

    n: Optional[int] = None

    # Python 3.13+: respects affinity/cgroup and -X cpu_count / PYTHON_CPU_COUNT override.
    with contextlib.suppress(Exception):
        fn = getattr(os, "process_cpu_count", None)
        if callable(fn):
            v = fn()
            if isinstance(v, int) and v > 0:
                n = int(v)

    if n is None:
        # Cross-platform affinity best-effort.
        with contextlib.suppress(Exception):
            n = max(1, int(len(get_allowed_cpus())))

    if n is None:
        with contextlib.suppress(Exception):
            v = os.cpu_count()
            if isinstance(v, int) and v > 0:
                n = int(v)

    n = int(n) if isinstance(n, int) and n > 0 else 1

    # Clamp by cgroup CPU quota (Linux) unless the user explicitly overrode.
    quota = None
    with contextlib.suppress(Exception):
        quota = _linux_cgroup_cpu_quota()
    if quota is not None and quota > 0.0:
        q = max(1, int(math.floor(float(quota))))
        n = max(1, min(int(n), int(q)))

    return max(1, int(n))


def _read_thread_cap_multiplier(default: int) -> int:
    v = os.environ.get("STNET_THREAD_CAP_MULTIPLIER") or os.environ.get("STNET_THREADS_CAP_MULTIPLIER")
    if v is not None and str(v).strip():
        with contextlib.suppress(Exception):
            default = int(v)
    return max(1, min(8, int(default)))


def _default_cap_mult(ncpu_raw: int, *, is_accel: bool, nogil: bool) -> int:
    """Heuristic oversubscription multiplier.

    - Accelerator: allow modest oversubscription to keep input pipeline busy.
    - CPU-only: avoid oversubscription by default (hurts CPU-bound work).
    - no-GIL: allow slightly more oversubscription when thread parallelism is real.
    """
    cap_mult = 3 if (nogil and is_accel) else (2 if is_accel else 1)

    # Be conservative on small CPU budgets (common in containers / CI).
    if ncpu_raw <= 4:
        cap_mult = 1
    elif ncpu_raw <= 8:
        cap_mult = min(int(cap_mult), 2)

    return _read_thread_cap_multiplier(int(cap_mult))


def _effective_local_world_size(default: int) -> int:
    return max(
        1,
        env_first_int(
            ("STNET_LOCAL_WORLD_SIZE", "LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE"),
            int(default),
        ),
    )


def _effective_thread_cap(*, ncpu: int, cap_mult: int, local_world: int, distribute: bool) -> int:
    node_thread_cap = max(2, int(ncpu) * int(cap_mult))
    if distribute and int(local_world) > 1:
        return max(2, int(node_thread_cap) // max(1, int(local_world)))
    return int(node_thread_cap)


@dataclass(slots=True)
class WorkerPolicy:
    nproc_per_node: int = 1
    device: str = "cpu"
    local_world_size: int = 1

    intra_ops: int = 1
    inter_ops: int = 1

    num_workers: int = 1
    prebatch: int = 1
    prefetch_factor: int = 1
    max_concurrency: int = 1
    h2d_streams: int = 1

    @staticmethod
    def _cpu_count() -> int:
        return process_cpu_count()

    @staticmethod
    def _detect_accelerator() -> Tuple[str, int]:
        dev_type = "cpu"
        n = 0

        try:
            accel = getattr(torch, "accelerator", None)
            if accel is not None and hasattr(accel, "is_available") and accel.is_available():
                current = getattr(accel, "current_accelerator", None)
                if callable(current):
                    dev = current(False)
                    if isinstance(dev, torch.device):
                        dev_type = dev.type
                dc = getattr(accel, "device_count", None)
                if callable(dc):
                    n = int(dc())
        except Exception:
            dev_type, n = "cpu", 0

        try:
            if n <= 0:
                if torch.cuda.is_available():
                    dev_type = "cuda"
                    n = int(torch.cuda.device_count())
                else:
                    xpu = getattr(torch, "xpu", None)
                    if xpu is not None and callable(getattr(xpu, "is_available", None)) and xpu.is_available():
                        dev_type = "xpu"
                        n = int(getattr(xpu, "device_count", lambda: 1)())
                    else:
                        mps_backend = getattr(torch.backends, "mps", None)
                        if mps_backend is not None and callable(getattr(mps_backend, "is_available", None)) and mps_backend.is_available():
                            dev_type = "mps"
                            n = 1
        except Exception:
            pass

        if n <= 0:
            dev_type, n = "cpu", 0
        return dev_type, max(0, n)

    @classmethod
    def autotune(cls) -> "WorkerPolicy":
        ncpu_raw = max(1, int(cls._cpu_count() or 1))
        dev_type, nacc = cls._detect_accelerator()
        is_accel = bool(nacc and int(nacc) > 0)

        local_world_guess = max(1, int(nacc or 1)) if is_accel else 1
        local_world_guess = max(1, int(env_first_int(
            ("STNET_LOCAL_WORLD_SIZE", "LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE"),
            local_world_guess,
        )))

        # Free-threading/No-GIL: enable more aggressive defaults only when actually beneficial.
        _nogil = False
        with contextlib.suppress(Exception):
            _nogil = bool(CPUAffinity.nogil_optimizations_enabled())

        cap_mult = _default_cap_mult(ncpu_raw, is_accel=is_accel, nogil=_nogil)

        distribute_default = int(local_world_guess) > 1
        distribute = bool(env_first_int(("STNET_DISTRIBUTE_THREAD_CAP",), int(distribute_default)))

        thread_cap = _effective_thread_cap(
            ncpu=ncpu_raw,
            cap_mult=cap_mult,
            local_world=int(local_world_guess),
            distribute=bool(distribute),
        )

        eff_cores = max(1, int(thread_cap) // max(1, int(cap_mult)))

        soft_inflight = 8 if is_accel else 4
        with contextlib.suppress(Exception):
            from ..data.pipeline import LoaderPolicy

            lp = LoaderPolicy()
            hard = int(lp.hard_inflight_batches(dev_type))
            soft_inflight = max(1, int(hard * max(1, int(lp.soft_cap_multiplier))))

        soft_auto_enabled = bool(env_first_int(("STNET_SOFT_INFLIGHT_AUTO",), 1))
        soft_inflight_max_default = (32 if is_accel else 24) if _nogil else (16 if is_accel else 12)
        soft_inflight_max = max(8, env_first_int(("STNET_SOFT_INFLIGHT_MAX",), soft_inflight_max_default))
        soft_inflight_explicit = env_first_int(("STNET_SOFT_INFLIGHT_CAP",), 0)
        if soft_inflight_explicit > 0:
            soft_inflight = max(1, int(soft_inflight_explicit))
        elif soft_auto_enabled:
            soft_base = max(0, env_first_int(("STNET_SOFT_INFLIGHT_BASE",), 2))
            soft_div = max(1, env_first_int(("STNET_SOFT_INFLIGHT_DIV",), 4))
            auto_soft = int(soft_base) + max(0, int(eff_cores) // int(soft_div))
            soft_inflight = max(int(soft_inflight), min(int(auto_soft), int(soft_inflight_max)))

        soft_inflight = max(1, min(int(soft_inflight), int(thread_cap)))

        if is_accel:
            if eff_cores <= 4:
                model_ratio = 1.00
            elif eff_cores <= 8:
                model_ratio = 0.90
            elif eff_cores <= 16:
                model_ratio = 0.80
            elif eff_cores <= 32:
                model_ratio = 0.70
            else:
                model_ratio = 0.60
        else:
            if eff_cores <= 4:
                model_ratio = 1.00
            elif eff_cores <= 8:
                model_ratio = 0.95
            elif eff_cores <= 16:
                model_ratio = 0.90
            else:
                model_ratio = 0.85

        with contextlib.suppress(Exception):
            env_key = "STNET_MODEL_CORE_RATIO_ACCEL" if is_accel else "STNET_MODEL_CORE_RATIO"
            model_ratio = float(env_float(env_key, float(model_ratio)))
        model_ratio = float(max(0.25, min(1.0, model_ratio)))

        model_budget = max(2, int(round(float(eff_cores) * model_ratio)))

        if model_budget <= 2:
            inter_ops = 1
        elif model_budget <= 8:
            inter_ops = max(1, model_budget // 4)
        else:
            inter_ops = max(2, min(8, model_budget // 6))
        inter_ops = max(1, min(int(inter_ops), max(1, int(model_budget) - 1)))
        intra_ops = max(1, int(model_budget) - int(inter_ops))

        data_budget = max(1, int(thread_cap) - (int(intra_ops) + int(inter_ops)))

        prebatch = 1
        prefetch_factor = 1

        env_pre = env_str("STNET_PREBATCH")
        if env_pre:
            with contextlib.suppress(Exception):
                prebatch = max(1, int(env_pre))
        elif _nogil:
            prebatch = 2

        env_pf = env_str("STNET_PREFETCH_FACTOR")
        if env_pf:
            with contextlib.suppress(Exception):
                prefetch_factor = max(1, int(env_pf))
        elif _nogil:
            prefetch_factor = 2

        prebatch = max(1, int(prebatch))
        prefetch_factor = max(1, int(prefetch_factor))

        base_workers = max(1, int(data_budget))
        base_workers = min(int(base_workers), int(thread_cap), int(soft_inflight))

        max_workers = max(
            1,
            int((int(soft_inflight) - int(prebatch)) // max(1, int(prefetch_factor))),
        )
        num_workers = max(1, min(int(base_workers), int(max_workers), int(soft_inflight)))

        max_concurrency = max(1, int(num_workers))

        total_threads = int(intra_ops) + int(inter_ops) + int(num_workers)
        if total_threads > int(thread_cap):
            overflow = int(total_threads) - int(thread_cap)
            if dev_type == "cpu":
                num_workers = max(1, int(num_workers) - int(overflow))
            else:
                intra_ops = max(1, int(intra_ops) - int(overflow))

            intra_ops = max(1, int(thread_cap) - int(inter_ops) - int(num_workers))
            max_concurrency = max(1, min(int(max_concurrency), int(num_workers)))

        local_world = int(local_world_guess)

        return cls(
            nproc_per_node=local_world,
            device=dev_type,
            local_world_size=local_world,
            intra_ops=int(intra_ops),
            inter_ops=int(inter_ops),
            num_workers=int(num_workers),
            prebatch=int(prebatch),
            prefetch_factor=int(prefetch_factor),
            max_concurrency=int(max_concurrency),
            h2d_streams=2 if dev_type in ("cuda", "xpu") else 1,
        )

    def as_threads_dict(self) -> dict[str, int]:
        return {
            "intra_ops": int(self.intra_ops),
            "inter_ops": int(self.inter_ops),
            "num_workers": int(self.num_workers),
            "max_concurrency": int(self.max_concurrency),
            "prebatch": int(self.prebatch),
            "prefetch_factor": int(self.prefetch_factor),
        }

    def as_procs_dict(self) -> dict[str, Union[int, str]]:
        return {
            "nproc_per_node": int(self.nproc_per_node),
            "device": str(self.device),
        }

    def apply_torch_threads(self) -> None:
        """Apply torch intra/inter-op thread settings (best-effort).

        Notes:
          - torch.set_num_interop_threads() can only be called once per process.
          - Keep idempotent to avoid repeated overhead and noisy exceptions.
        """
        global _TORCH_NUM_THREADS_SET, _TORCH_INTEROP_THREADS_SET, _TORCH_INTEROP_LOCKED

        intra = max(1, int(self.intra_ops))
        inter = max(1, int(self.inter_ops))

        with _TORCH_THREAD_CFG_LOCK:
            if _TORCH_NUM_THREADS_SET != int(intra):
                with contextlib.suppress(Exception):
                    torch.set_num_threads(int(intra))
                    _TORCH_NUM_THREADS_SET = int(intra)

            if hasattr(torch, "set_num_interop_threads") and not bool(_TORCH_INTEROP_LOCKED):
                if _TORCH_INTEROP_THREADS_SET is None:
                    try:
                        torch.set_num_interop_threads(int(inter))
                        _TORCH_INTEROP_THREADS_SET = int(inter)
                    except Exception:
                        # Per PyTorch docs, inter-op threads can be set only once
                        # and before parallel work starts. Avoid retry storms.
                        _TORCH_INTEROP_LOCKED = True
                elif int(_TORCH_INTEROP_THREADS_SET) != int(inter):
                    # Keep the first value.
                    pass


def empty_device_cache(
    *,
    device: Optional[Union[torch.device, str]] = None,
    do_gc: bool = True,
    min_interval_s: Optional[float] = None,
) -> None:
    """Best-effort device cache clearing with rate limiting.

    Centralizes cache clearing across CUDA/XPU/MPS/accelerator backends and avoids
    `empty_cache()` storms (which can become very expensive on fast/no-GIL loops).

    Env:
      - STNET_EMPTY_CACHE=0/false/off to disable
      - STNET_EMPTY_CACHE_MIN_INTERVAL_S (default: 0.5)
    """
    if not env_bool("STNET_EMPTY_CACHE", True):
        return

    if min_interval_s is None:
        min_interval_s = env_first_float(("STNET_EMPTY_CACHE_MIN_INTERVAL_S",), 0.5)
    with contextlib.suppress(Exception):
        min_interval_s = float(min_interval_s)  # type: ignore[arg-type]
    if not isinstance(min_interval_s, (int, float)):
        min_interval_s = 0.5
    if float(min_interval_s) < 0:
        min_interval_s = 0.0

    now = time.monotonic()
    key = _empty_cache_device_key(device)
    with _EMPTY_CACHE_LOCK:
        last = float(_EMPTY_CACHE_LAST_CALL_S_BY_DEVICE.get(key, 0.0))
        if min_interval_s and (now - last) < float(min_interval_s):
            return
        _EMPTY_CACHE_LAST_CALL_S_BY_DEVICE[key] = float(now)

    if do_gc:
        with contextlib.suppress(Exception):
            gc.collect()

    with contextlib.suppress(Exception):
        accelerator = getattr(torch, "accelerator", None)
        memory_mod = getattr(accelerator, "memory", None) if accelerator is not None else None
        empty_cache = getattr(memory_mod, "empty_cache", None) if memory_mod is not None else None
        if callable(empty_cache):
            empty_cache()

    # Backend-specific clearing:
    # - If device is provided, clear only that backend (avoid cross-backend blast radius).
    # - If device is None, keep the previous "clear everything best-effort" behavior.
    target = None
    with contextlib.suppress(Exception):
        if device is not None:
            target = device if isinstance(device, torch.device) else torch.device(str(device))

    with contextlib.suppress(Exception):
        if target is None or target.type == "cuda":
            if torch.cuda.is_available():
                if target is not None and target.index is not None:
                    with torch.cuda.device(int(target.index)):
                        torch.cuda.empty_cache()
                else:
                    torch.cuda.empty_cache()

    with contextlib.suppress(Exception):
        if target is None or getattr(target, "type", None) == "mps":
            mps_mod = getattr(torch, "mps", None)
            empty_cache = getattr(mps_mod, "empty_cache", None) if mps_mod is not None else None
            if callable(empty_cache):
                empty_cache()

    with contextlib.suppress(Exception):
        if target is None or getattr(target, "type", None) == "xpu":
            xpu_mod = getattr(torch, "xpu", None)
            empty_cache = getattr(xpu_mod, "empty_cache", None) if xpu_mod is not None else None
            if callable(empty_cache):
                empty_cache()
            else:
                memory_mod = getattr(xpu_mod, "memory", None) if xpu_mod is not None else None
                empty_cache = getattr(memory_mod, "empty_cache", None) if memory_mod is not None else None
                if callable(empty_cache):
                    empty_cache()


def set_float32_precision(
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    autocast_dtype: Optional[torch.dtype] = None,
    enable_tf32: bool = True,
) -> None:
    """Best-effort FP32 precision tuning for CUDA matmul/cudnn.

    Note: These backend knobs are effectively process-global, so we only cache by device type.
    """
    if device.type != "cuda":
        return

    use_tf32 = False
    if enable_tf32:
        for _dt in (dtype, autocast_dtype):
            if _dt is None:
                continue
            if _dt not in (torch.float32, torch.float64):
                use_tf32 = True
                break

    precision = "tf32" if use_tf32 else "ieee"
    key = "cuda"
    with _FP32_PRECISION_LOCK:
        if _FP32_PRECISION_CACHE.get(key) == precision:
            return
        _FP32_PRECISION_CACHE[key] = precision

    with contextlib.suppress(Exception):
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
                torch.backends.cuda.matmul.fp32_precision = precision

    with contextlib.suppress(Exception):
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "conv"):
            if hasattr(torch.backends.cudnn.conv, "fp32_precision"):
                torch.backends.cudnn.conv.fp32_precision = precision
    with contextlib.suppress(Exception):
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "rnn"):
            if hasattr(torch.backends.cudnn.rnn, "fp32_precision"):
                torch.backends.cudnn.rnn.fp32_precision = precision


_TZ_ALIASES = {
    "KST": "Asia/Seoul",
    "GMT": "Etc/GMT",
    "UTC": "UTC",
}


def resolve_timezone(name: Optional[str] = None) -> Optional[tzinfo]:
    """Resolve a tz database name into a tzinfo.

    Returns None if zoneinfo is unavailable or the zone can't be resolved.
    """
    resolved = (name or "GMT").strip()
    alias = _TZ_ALIASES.get(resolved.upper(), resolved)
    if alias.upper() == "UTC":
        return timezone.utc
    if ZoneInfo is None:
        return None
    with contextlib.suppress(Exception):
        return ZoneInfo(alias)  # type: ignore[return-value]
    return None


def epoch_time_ns() -> int:
    """Nanoseconds since the Unix epoch as an int."""
    return int(time.time_ns())


def posix_time(tz_name: Optional[str] = None) -> int:
    """Backward-compatible wrapper returning epoch ns.

    `tz_name` is accepted for compatibility but does not affect epoch time.
    """
    _ = tz_name
    return epoch_time_ns()


def system_info() -> Tuple[str, str, str, str]:
    sysname = platform.system() or ""
    release = platform.release() or ""
    kernel = f"{sysname} {release}".strip()
    os_name = sysname
    with contextlib.suppress(Exception):
        if sysname == "Linux" and hasattr(platform, "freedesktop_os_release"):
            info = platform.freedesktop_os_release()
            name = info.get("NAME") or "Linux"
            version = info.get("VERSION_ID") or ""
            os_name = (name if not version else f"{name} {version}").strip()
        elif sysname == "Windows":
            win = platform.win32_ver()
            os_name = f"Windows {win[0] or ''}".strip()
        elif sysname == "Darwin":
            mac = platform.mac_ver()[0]
            os_name = f"macOS {mac or ''}".strip()
    arch = platform.machine() or ""
    accelerators: list[str] = []

    with contextlib.suppress(Exception):
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                accelerators.append(f"cuda:{idx}={torch.cuda.get_device_name(idx)}")

    with contextlib.suppress(Exception):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerators.append("mps=Apple MPS")

    with contextlib.suppress(Exception):
        if hasattr(torch, "xpu") and hasattr(torch.xpu, "device_count"):
            count = torch.xpu.device_count()
            if count and count > 0:
                get_name = getattr(torch.xpu, "get_device_name", None)
                for idx in range(count):
                    name = get_name(idx) if callable(get_name) else "XPU"
                    accelerators.append(f"xpu:{idx}={name}")

    with contextlib.suppress(Exception):
        if hasattr(torch, "is_vulkan_available") and torch.is_vulkan_available():
            accelerators.append("vulkan=available")

    return os_name, kernel, arch, ";".join(accelerators)


def cpu_info(max_bytes: Optional[int] = None) -> str:
    names: list[str] = []
    total = process_cpu_count()
    brand = ""
    with contextlib.suppress(Exception):
        import cpuinfo  # type: ignore

        info = cpuinfo.get_cpu_info() or {}
        brand = info.get("brand_raw") or info.get("brand") or ""
    if not brand and os.path.exists("/proc/cpuinfo"):
        with contextlib.suppress(Exception):
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as handle:
                lines = [ln.strip() for ln in handle.readlines() if "model name" in ln]
            if lines:
                names = [ln.split(":", 1)[1].strip() for ln in lines]
    if not names and platform.system() == "Darwin":
        with contextlib.suppress(Exception):
            import subprocess

            output = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
            brand = output.decode("utf-8", "ignore").strip()
    if not names and platform.system() == "Windows":
        brand = platform.processor()
        if not brand:
            with contextlib.suppress(Exception):
                import subprocess

                output = subprocess.check_output(
                    ["powershell", "-Command", "(Get-CimInstance Win32_Processor).Name"]
                )
                brand = output.decode("utf-8", "ignore").strip()
    if not names:
        fallback = brand or platform.processor() or "CPU"
        names = [fallback] * total
    pairs = [f"{idx}:{names[idx] if idx < len(names) else names[0]}" for idx in range(total)]
    result = ";".join(pairs)
    if max_bytes is not None and max_bytes > 0:
        encoded = result.encode("utf-8")
        if len(encoded) > max_bytes:
            result = encoded[:max_bytes].decode("utf-8", "ignore")
    return result


def get_runtime_config() -> SimpleNamespace:
    return _RUNTIME_CFG


def is_main_loadable() -> bool:
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return False
    main_path = getattr(main_mod, "__file__", None)
    if not main_path:
        return False
    try:
        main_path = os.fspath(main_path)
    except TypeError:
        return False
    if isinstance(main_path, str) and main_path.startswith("<") and main_path.endswith(">"):
        return False
    return os.path.exists(main_path)


def initialize_python_path() -> str:
    separator = os.pathsep
    current_env = os.environ.get("PYTHONPATH", "")
    env_paths = [path for path in current_env.split(separator) if path]
    paths: list[str] = list(env_paths)
    seen: set[str] = set(env_paths)

    def _prioritized_path(candidate: Path | str | None) -> None:
        if candidate is None:
            return
        try:
            path_str = os.fspath(candidate)
        except TypeError:
            return
        if not path_str:
            return
        if path_str in seen:
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return
        seen.add(path_str)
        paths.insert(0, path_str)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    def _target_path(entry: str) -> None:
        if entry in seen:
            return
        seen.add(entry)
        paths.append(entry)

    try:
        package_dir = Path(__file__).resolve().parents[1]
    except Exception:
        package_dir = None
    project_dir = package_dir.parent if package_dir is not None else None
    cwd_dir: Path | None = None
    with contextlib.suppress(Exception):
        cwd_dir = Path.cwd().resolve()
    main_dir: Path | None = None
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        main_file = getattr(main_module, "__file__", None)
        if main_file:
            with contextlib.suppress(Exception):
                main_dir = Path(main_file).resolve().parent

    for candidate in (package_dir, project_dir, main_dir, cwd_dir):
        _prioritized_path(candidate)

    for entry in list(sys.path):
        if not entry:
            continue
        try:
            entry_str = os.fspath(entry)
        except TypeError:
            continue
        if not entry_str:
            continue
        _target_path(entry_str)

    python_path = separator.join(paths)
    os.environ["PYTHONPATH"] = python_path
    return python_path


def optimal_start_method() -> str:
    current = mp.get_start_method(allow_none=True)
    if current is not None:
        return str(current)
    for method in ("forkserver", "spawn"):
        try:
            multiprocessing.get_context(method)
        except ValueError:
            continue
        return method
    raise RuntimeError("No supported multiprocessing start method (tried forkserver, spawn).")


def set_multiprocessing_env() -> None:
    with contextlib.suppress(RuntimeError):
        mp.set_sharing_strategy("file_system")

    if mp.get_start_method(allow_none=True) is not None:
        return

    last_error: Optional[BaseException] = None
    for method in ("forkserver", "spawn"):
        try:
            multiprocessing.get_context(method)
        except ValueError as exc:
            last_error = exc
            continue
        try:
            for module in (multiprocessing, mp):
                module.set_start_method(method, force=True)
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            continue
        return
    raise RuntimeError(
        "Unable to configure multiprocessing start method (tried forkserver, spawn)."
    ) from last_error


def default_temp() -> str:
    if sys.platform.startswith("win"):
        return os.environ.get("TEMP", r"C:\Windows\Temp")
    return "/tmp" if os.path.isdir("/tmp") else "/var/tmp"


def new_dir(prefix: str) -> str:
    base = default_temp()
    os.makedirs(base, exist_ok=True)
    directory = os.path.join(base, f"{prefix}_{os.getpid()}_{os.urandom(4).hex()}")
    os.makedirs(directory, exist_ok=True)
    return directory


def get_sdpa_backends() -> list[object]:
    names = _RUNTIME_CFG.sdpa_backends or []
    if not names:
        return []
    try:
        from torch.nn.attention import SDPBackend  # type: ignore
    except Exception:
        return []
    mapping = {
        "FLASH": "FLASH_ATTENTION",
        "FLASH_ATTENTION": "FLASH_ATTENTION",
        "EFFICIENT": "EFFICIENT_ATTENTION",
        "MEM_EFFICIENT": "EFFICIENT_ATTENTION",
        "CUDNN": "CUDNN_ATTENTION",
        "MATH": "MATH",
    }
    backends: list[object] = []
    for name in names:
        key = mapping.get(str(name), str(name))
        if hasattr(SDPBackend, key):
            backends.append(getattr(SDPBackend, key))
    return backends


# Backward-compatible alias (typo in historical name).
get_dpa_backends = get_sdpa_backends


def _resolve_device(device: Optional[Union[torch.device, str]]) -> torch.device:
    if device is not None:
        return device if isinstance(device, torch.device) else torch.device(str(device))
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cuda_compute_capability(device: Union[torch.device, str]) -> Tuple[int, int]:
    dev = _resolve_device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return (0, 0)
    try:
        major, minor = torch.cuda.get_device_capability(dev)
    except Exception:
        return (0, 0)
    return (int(major), int(minor))


def is_cpu_bf16_supported() -> bool:
    """Return True when BF16 ops are supported on CPU (best-effort)."""
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


def is_cuda_bf16_supported(device: Optional[Union[torch.device, str]] = None) -> bool:
    """Return True when BF16 ops are supported on CUDA device (best-effort)."""
    try:
        if not torch.cuda.is_available():
            return False
        dev = _resolve_device(device)
        if dev.type != "cuda":
            return False
        with torch.cuda.device(dev):
            f = getattr(torch.cuda, "is_bf16_supported", None)
            if callable(f):
                try:
                    return bool(f(including_emulation=False))
                except TypeError:
                    return bool(f())
            major, _ = torch.cuda.get_device_capability(dev)
            return int(major) >= 8
    except Exception:
        return False


def is_float8_supported(device: Optional[Union[torch.device, str]] = None) -> Tuple[bool, str]:
    try:
        dev = _resolve_device(device)
        if dev.type == "cuda" and torch.cuda.is_available():
            try:
                import torch.cuda.amp as _tca

                with contextlib.suppress(Exception):
                    if getattr(_tca, "is_float8_available", None) is not None:
                        ok, reason = _tca.is_float8_available()
                        return (bool(ok), str(reason))
            except Exception:
                pass
            major, _minor = cuda_compute_capability(dev)
            if major >= 9:
                return (True, "Hopper+ supports FP8")
            return (False, "FP8 requires sm90+")
        return (False, "FP8 requires CUDA sm90+ and torch.cuda")
    except Exception:
        return (False, "Unknown float8 support")


def is_int8_supported(device: Optional[Union[torch.device, str]] = None) -> Tuple[bool, str]:
    try:
        dev = _resolve_device(device)
        if dev.type == "cuda" and torch.cuda.is_available():
            return (True, "Int8 supported on CUDA")
        return (True, "Int8 supported on CPU")
    except Exception:
        return (False, "Unknown int8 support")


def is_int4_supported(device: Optional[Union[torch.device, str]] = None) -> Tuple[bool, str]:
    try:
        dev = _resolve_device(device)
        if dev.type == "cuda" and torch.cuda.is_available():
            return (True, "Int4 supported on CUDA")
        return (True, "Int4 supported on CPU")
    except Exception:
        return (False, "Unknown int4 support")


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


@dataclass(slots=True)
class Device:
    """Device capability snapshot used for precision negotiation.

    This intentionally lives in core.system to avoid core->data imports.
    """

    device: torch.device
    device_type: str
    cuda_cc: Optional[Tuple[int, int]]
    float_dtypes: Tuple[torch.dtype, ...]
    int_dtypes: Tuple[torch.dtype, ...]
    float8_dtypes: Tuple[torch.dtype, ...]
    int_quant_bits: int


_DEVICE_STATS_CACHE: dict[Tuple[str, int], Device] = {}
_DEVICE_STATS_LOCK = threading.Lock()


def get_device_stats(device: Optional[Union[torch.device, str]] = None) -> Device:
    """Return a cached Device for the given device."""
    dev = _resolve_device(device)
    key = (str(dev.type), int(dev.index) if dev.index is not None else -1)
    with _DEVICE_STATS_LOCK:
        cached = _DEVICE_STATS_CACHE.get(key)
        if cached is not None:
            return cached

    # Compute capabilities.
    device_type = str(dev.type)
    cc = None
    if device_type == "cuda" and torch.cuda.is_available():
        major, minor = cuda_compute_capability(dev)
        cc = (int(major), int(minor)) if (major > 0 or minor > 0) else None

    # Dtypes from env (if present).
    float_env = env_first(("STNET_DATA_FLOAT_DTYPES", "STNET_FLOAT_DTYPES"))
    float_dtypes = _parse_dtypes_env(float_env) if float_env else tuple()
    int_env = env_first(("STNET_DATA_INT_DTYPES", "STNET_INT_DTYPES"))
    int_dtypes = _parse_dtypes_env(int_env) if int_env else tuple()

    # Defaults.
    if not float_dtypes:
        floats: list[torch.dtype] = [torch.float32]
        if device_type == "cuda" and torch.cuda.is_available():
            floats.insert(0, torch.float16)
            if is_cuda_bf16_supported(dev):
                floats.insert(0, torch.bfloat16)
        elif device_type == "cpu" and is_cpu_bf16_supported():
            floats.insert(0, torch.bfloat16)
        float_dtypes = tuple(dict.fromkeys(floats))

    if not int_dtypes:
        int_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64)

    # Quant bits.
    bits = env_first_int(("STNET_DATA_INT_QUANT_BITS", "STNET_INT_QUANT_BITS"), default=0)
    quant_bits = int(bits) if int(bits) > 0 else 8

    stats = Device(
        device=dev,
        device_type=device_type,
        cuda_cc=cc,
        float_dtypes=tuple(float_dtypes),
        int_dtypes=tuple(int_dtypes),
        float8_dtypes=tuple(),
        int_quant_bits=quant_bits,
    )
    with _DEVICE_STATS_LOCK:
        _DEVICE_STATS_CACHE[key] = stats
    return stats


def get_device(
    *args: Any,
    deterministic: Optional[bool] = None,
    allow_tf32: Optional[bool] = None,
    cudnn_benchmark: Optional[bool] = None,
    matmul_precision: Optional[str] = None,
    sdpa_backends: Optional[Sequence[str]] = None,
    te_first: Optional[bool] = None,
    **kwargs: Any,
) -> torch.device:
    """Select a default device and configure global runtime knobs.

    The runtime configuration is global and must be protected under no-GIL.
    """
    del args, kwargs  # compatibility with older call sites

    with _RUNTIME_CFG_LOCK:
        cfg = _RUNTIME_CFG

        if deterministic is not None:
            cfg.deterministic = bool(deterministic)
        det_flag = bool(cfg.deterministic)

        allow_val = (
            bool(allow_tf32)
            if allow_tf32 is not None
            else (
                bool(cfg.allow_tf32)
                if cfg.allow_tf32 is not None and deterministic is None
                else (False if det_flag else True)
            )
        )
        cfg.allow_tf32 = bool(allow_val)

        benchmark_val = (
            bool(cudnn_benchmark)
            if cudnn_benchmark is not None
            else (
                bool(cfg.cudnn_benchmark)
                if cfg.cudnn_benchmark is not None and deterministic is None
                else (False if det_flag else True)
            )
        )
        cfg.cudnn_benchmark = bool(benchmark_val)

        precision_val = (
            str(matmul_precision)
            if matmul_precision is not None
            else (
                str(cfg.matmul_precision)
                if cfg.matmul_precision is not None and deterministic is None
                else ("highest" if det_flag else "high")
            )
        )
        cfg.matmul_precision = str(precision_val)

        if sdpa_backends is not None:
            cfg.sdpa_backends = [str(x) for x in sdpa_backends]
        if te_first is not None:
            cfg.te_first = bool(te_first)

        # Snapshot for use outside lock.
        det = bool(cfg.deterministic)
        allow_tf32_val = bool(cfg.allow_tf32)
        cudnn_bench_val = bool(cfg.cudnn_benchmark)
        matmul_prec = str(cfg.matmul_precision)

    # Apply global matmul precision if supported.
    with contextlib.suppress(Exception):
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(matmul_prec)

    if torch.cuda.is_available():
        idx = 0
        with contextlib.suppress(Exception):
            idx_env = env_first_int(("LOCAL_RANK",), default=0)
            ndev = max(1, int(torch.cuda.device_count()))
            idx = int(idx_env) % int(ndev)

        with contextlib.suppress(Exception):
            torch.cuda.set_device(int(idx))
        device = torch.device(f"cuda:{idx}")

        # Configure CuDNN determinism/benchmark.
        with contextlib.suppress(Exception):
            torch.backends.cudnn.deterministic = bool(det)
        with contextlib.suppress(Exception):
            torch.backends.cudnn.benchmark = bool(cudnn_bench_val)

        # Apply TF32 policy (best-effort).
        with contextlib.suppress(Exception):
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32_val)
        with contextlib.suppress(Exception):
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = bool(allow_tf32_val)

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif hasattr(torch, "is_vulkan_available") and torch.is_vulkan_available():
        device = torch.device("vulkan")
    else:
        device = torch.device("cpu")
    return device


def optimal_optimizer_params(device: torch.device, use_foreach: Optional[bool], use_fused: bool) -> dict[str, bool]:
    devt = device.type
    flags: dict[str, bool] = {}
    flags["foreach"] = devt in {"cuda", "xpu"} if use_foreach is None else bool(use_foreach)
    if use_fused and devt in {"cuda", "xpu"}:
        flags["fused"] = True
        flags["foreach"] = False
    return flags


def optimal_procs() -> dict[str, Union[int, str]]:
    return WorkerPolicy.autotune().as_procs_dict()


def cpu_count() -> int:
    return process_cpu_count()


def num_accelerators() -> int:
    cuda_iface = getattr(torch, "cuda", None)
    if cuda_iface is not None:
        with contextlib.suppress(Exception):
            if cuda_iface.is_available():
                return int(cuda_iface.device_count()) or 0

    with contextlib.suppress(Exception):
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and callable(getattr(xpu, "is_available", None)) and xpu.is_available():
            count = int(getattr(xpu, "device_count", lambda: 1)()) or 1
            return max(count, 1)

    with contextlib.suppress(Exception):
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and callable(getattr(mps, "is_available", None)) and mps.is_available():
            return 1

    with contextlib.suppress(Exception):
        if hasattr(torch, "is_vulkan_available") and torch.is_vulkan_available():
            return 1

    return 0


def optimal_threads() -> dict[str, Union[int, bool]]:
    return WorkerPolicy.autotune().as_threads_dict()


def optimize_threads(
    intra: Optional[int] = None,
    inter: Optional[int] = None,
) -> dict[str, Union[int, bool]]:
    """Autotune and apply torch thread settings.

    Torch thread configuration is process-global; this helper centralizes the
    policy and optionally allows overriding intra/inter-op thread counts.
    """
    wp = WorkerPolicy.autotune()
    if intra is not None:
        wp = replace(wp, intra_ops=int(intra))
    if inter is not None:
        wp = replace(wp, inter_ops=int(inter))
    wp.apply_torch_threads()
    return wp.as_threads_dict()


class Thread:
    """Lightweight CPU affinity pinning + telemetry-based retuning for IO-heavy pipelines."""

    __slots__ = (
        "_psutil",
        "_allowed_cpus",
        "_ring",
        "_tls",
        "_lock",
        "_io_workers",
        "_samples",
        "_cpu_ns",
        "_wall_ns",
        "_last_retune_ts",
        "_enabled",
        "_pin_attempts",
        "_pin_success",
        "_omp_ok",
        "_nogil",
        "_flush_every",
        "_sample_every",
    )

    def __init__(self, io_workers: int) -> None:
        self._psutil = self._import_psutil()
        self._allowed_cpus = get_allowed_cpus()
        self._ring = itertools.cycle(self._allowed_cpus)
        self._tls = threading.local()
        self._lock = Lock()
        self._io_workers = max(1, int(io_workers))
        self._samples = 0
        self._cpu_ns = 0
        self._wall_ns = 0
        self._last_retune_ts = time.perf_counter()
        self._pin_attempts = 0
        self._pin_success = 0
        self._omp_ok = self.spread_threads()
        self._enabled = (len(self._allowed_cpus) >= 2) or self._omp_ok

        # Free-threading/No-GIL: reduce contention in telemetry and allow slightly higher caps.
        with contextlib.suppress(Exception):
            self._nogil = bool(Thread.nogil_optimizations_enabled())
        if not hasattr(self, "_nogil"):
            self._nogil = False

        flush_default = 64 if self._nogil else 1
        sample_default = 8 if self._nogil else 1
        self._flush_every = max(
            1,
            min(
                1024,
                env_first_int(("STNET_TLB_FLUSH_EVERY", "STNET_TLB_FLUSH"), flush_default),
            ),
        )
        self._sample_every = max(
            1,
            min(
                1024,
                env_first_int(("STNET_TLB_SAMPLE_EVERY", "STNET_TLB_SAMPLE"), sample_default),
            ),
        )
        self.tune_threads(io_workers, initial=True)

    # --- Free-threading helpers (mirrors freethreading.py semantics) ---

    @staticmethod
    def is_free_threaded_build() -> bool:
        """True if the *interpreter build* supports free-threading."""
        try:
            v = sysconfig.get_config_var("Py_GIL_DISABLED")
            return bool(int(v)) if v is not None else False
        except Exception:
            return False

    @staticmethod
    def is_gil_enabled() -> bool:
        """True if the GIL is enabled in the current process."""
        fn = getattr(sys, "_is_gil_enabled", None)
        if callable(fn):
            with contextlib.suppress(Exception):
                return bool(fn())
            return True
        # Python < 3.13: always GIL.
        return True

    @staticmethod
    def nogil_active() -> bool:
        """True only when this is a free-threaded build *and* the GIL is disabled."""
        return Thread.is_free_threaded_build() and (not Thread.is_gil_enabled())

    @staticmethod
    def nogil_optimizations_enabled() -> bool:
        """Whether STNet should enable no-GIL specific optimizations.

        Default: follow runtime state (nogil_active()).
        Override env:
          - STNET_NOGIL_OPT / STNET_NO_GIL_OPT / STNET_FREE_THREADING_OPT
        """
        for key in ("STNET_NOGIL_OPT", "STNET_NO_GIL_OPT", "STNET_FREE_THREADING_OPT"):
            override = parse_bool(os.environ.get(key))
            if override is not None:
                return bool(override)
        return Thread.nogil_active()

    @staticmethod
    def _import_psutil() -> ModuleType | None:
        with contextlib.suppress(Exception):
            return importlib.import_module("psutil")
        return None

    def total_procs(self) -> list[int]:
        """Backward-compatible API: returns the process-usable CPU IDs."""
        return list(self._allowed_cpus)

    @staticmethod
    def spread_threads() -> bool:
        plat = sys.platform
        if plat.startswith("linux"):
            candidates = ["libgomp.so.1", "libgomp.so", "libiomp5.so", "libomp.so"]
        elif plat == "darwin":
            candidates = ["libomp.dylib", "libiomp5.dylib"]
        elif os.name == "nt":
            candidates = ["libiomp5md.dll", "vcomp140.dll"]
        else:
            candidates = []
        for name in candidates:
            try:
                lib = ctypes.CDLL(name)
            except OSError:
                continue
            try:
                fn = getattr(lib, "omp_set_proc_bind")
                fn.argtypes = [ctypes.c_int]
                fn.restype = None
                fn(4)
                return True
            except Exception:
                pass
            try:
                kmp = getattr(lib, "kmp_set_defaults")
                kmp.restype = None
                kmp(b"KMP_AFFINITY=granularity=fine,scatter")
                return True
            except Exception:
                pass
        return False

    def _next_core(self) -> int:
        with self._lock:
            return int(next(self._ring))

    @staticmethod
    def _windows_group_segments(k32: Any) -> Tuple[list[Tuple[int, int]], int]:
        """Return allowed (group, count) segments and total count."""
        get_group_cnt = getattr(k32, "GetActiveProcessorGroupCount", None)
        get_group_procs = getattr(k32, "GetActiveProcessorCount", None)
        if not (callable(get_group_cnt) and callable(get_group_procs)):
            return ([], 0)

        group_count = int(get_group_cnt())
        if group_count <= 0:
            return ([], 0)

        counts: list[int] = []
        for i in range(group_count):
            c = 0
            with contextlib.suppress(Exception):
                c = int(get_group_procs(ctypes.c_ushort(int(i))))
            if not c:
                with contextlib.suppress(Exception):
                    c = int(get_group_procs(int(i)))
            counts.append(max(0, int(c)))

        groups: list[int] | None = None
        get_pg_aff = getattr(k32, "GetProcessGroupAffinity", None)
        if callable(get_pg_aff):
            try:
                h = getattr(k32, "GetCurrentProcess", lambda: None)()
                cnt = ctypes.c_ushort(0)
                arr = (ctypes.c_ushort * int(group_count))()
                ok = int(get_pg_aff(h, ctypes.byref(cnt), arr))
                if ok:
                    ng = int(cnt.value)
                    if ng > 0:
                        groups = [int(arr[i]) for i in range(ng)]
            except Exception:
                groups = None

        if not groups:
            groups = list(range(group_count))

        segs: list[Tuple[int, int]] = []
        total = 0
        for g in groups:
            if 0 <= int(g) < group_count:
                c = int(counts[int(g)])
                if c > 0:
                    segs.append((int(g), int(c)))
                    total += int(c)
        return segs, int(total)

    @staticmethod
    def _pin_thread_windows(core: int) -> bool:
        try:
            k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            GetCurrentThread = k32.GetCurrentThread
            SetThreadAffinityMask = k32.SetThreadAffinityMask
            SetThreadGroupAffinity = k32.SetThreadGroupAffinity
            SetThreadIdealProcessorEx = getattr(k32, "SetThreadIdealProcessorEx", None)

            # If we can get a precise process affinity mask, use it (single group).
            try:
                get_proc = getattr(k32, "GetCurrentProcess", None)
                get_mask = getattr(k32, "GetProcessAffinityMask", None)
                if callable(get_proc) and callable(get_mask):
                    h = get_proc()
                    proc_mask = ctypes.c_size_t(0)
                    sys_mask = ctypes.c_size_t(0)
                    ok = int(get_mask(h, ctypes.byref(proc_mask), ctypes.byref(sys_mask)))
                    if ok:
                        m = int(proc_mask.value)
                        if m:
                            allowed = [i for i in range(m.bit_length()) if (m >> i) & 1]
                            idx = int(core) % max(1, len(allowed))
                            within = int(allowed[idx])
                            thread = GetCurrentThread()
                            mask = ctypes.c_size_t(1 << int(within))
                            prev = SetThreadAffinityMask(thread, mask.value)
                            return bool(prev)
            except Exception:
                pass

            # Multi-group / fallback mapping.
            segs, total = Thread._windows_group_segments(k32)
            if total <= 0 or not segs:
                return False

            idx = int(core) % int(total)
            within = idx
            group = segs[0][0]
            for g, cnt in segs:
                if within < cnt:
                    group = int(g)
                    break
                within -= int(cnt)

            thread = GetCurrentThread()

            class GROUP_AFFINITY(ctypes.Structure):
                _fields_ = [
                    ("Mask", ctypes.c_ulonglong),
                    ("Group", ctypes.c_ushort),
                    ("Reserved", ctypes.c_ushort * 3),
                ]

            affinity = GROUP_AFFINITY(
                ctypes.c_ulonglong(1 << int(within)),
                ctypes.c_ushort(int(group)),
                (ctypes.c_ushort * 3)(0, 0, 0),
            )
            ok = bool(SetThreadGroupAffinity(thread, ctypes.byref(affinity), None))
            if ok and SetThreadIdealProcessorEx is not None:
                with contextlib.suppress(Exception):

                    class PROCESSOR_NUMBER(ctypes.Structure):
                        _fields_ = [
                            ("Group", ctypes.c_ushort),
                            ("Number", ctypes.c_ubyte),
                            ("Reserved", ctypes.c_ubyte),
                        ]

                    proc_num = PROCESSOR_NUMBER(int(group), int(within), 0)
                    SetThreadIdealProcessorEx(thread, ctypes.byref(proc_num), None)
            return bool(ok)
        except Exception:
            return False

    @staticmethod
    def _pin_thread_linux(core: int) -> bool:
        try:
            tid = threading.get_native_id()
            os.sched_setaffinity(tid, {int(core)})
            return True
        except Exception:
            with contextlib.suppress(Exception):
                os.sched_setaffinity(0, {int(core)})
                return True
            return False

    def pin_thread(self) -> None:
        if not self._enabled:
            return
        attempts = getattr(self._tls, "attempts", 0)
        if getattr(self._tls, "pinned", False) or attempts >= 4:
            return
        self._tls.attempts = attempts + 1

        core = self._next_core()
        ok = False

        if os.name == "nt":
            ok = self._pin_thread_windows(core)
        else:
            plat = sys.platform
            if plat.startswith("linux"):
                ok = self._pin_thread_linux(core)
            elif plat == "darwin":
                with contextlib.suppress(Exception):
                    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
                    THREAD_AFFINITY_POLICY = 4

                    class thread_affinity_policy_data_t(ctypes.Structure):
                        _fields_ = [("affinity_tag", ctypes.c_int)]

                    policy = thread_affinity_policy_data_t(int(core) + 1)
                    lib.mach_thread_self.restype = ctypes.c_uint
                    lib.thread_policy_set.argtypes = [
                        ctypes.c_uint,
                        ctypes.c_int,
                        ctypes.c_void_p,
                        ctypes.c_uint,
                    ]
                    port = lib.mach_thread_self()
                    ok = (
                        lib.thread_policy_set(
                            port, THREAD_AFFINITY_POLICY, ctypes.byref(policy), 1
                        )
                        == 0
                    )
            # Other platforms: no pinning.

        self._tls.pinned = bool(ok)
        self._pin_attempts += 1
        if ok:
            self._pin_success += 1
        if self._pin_attempts >= 16 and self._pin_success == 0 and not self._omp_ok:
            self._enabled = False

    @staticmethod
    def optimize_threads(intra: Optional[int] = None, inter: Optional[int] = None) -> None:
        # Backward-compatible alias; canonical implementation lives at module scope.
        optimize_threads(intra=intra, inter=inter)

    def tune_threads(
        self,
        io_workers: Optional[int] = None,
        *_unused_args: Any,
        initial: bool = False,
        **_unused_kwargs: Any,
    ) -> None:
        if not self._enabled:
            return
        if initial:
            cpus = max(1, len(self._allowed_cpus))
            tuned_workers = max(
                1,
                min(
                    int(io_workers if io_workers is not None else self._io_workers),
                    cpus,
                ),
            )
            self._io_workers = tuned_workers

            dev_type, nacc = WorkerPolicy._detect_accelerator()
            is_accel = bool(nacc and int(nacc) > 0)

            cap_mult = _default_cap_mult(cpus, is_accel=is_accel, nogil=bool(self._nogil))
            local_world = _effective_local_world_size(1)
            distribute_default = local_world > 1
            distribute = bool(env_first_int(("STNET_DISTRIBUTE_THREAD_CAP",), int(distribute_default)))
            thread_cap = _effective_thread_cap(
                ncpu=cpus,
                cap_mult=cap_mult,
                local_world=local_world,
                distribute=bool(distribute),
            )

            try:
                intra_now = int(torch.get_num_threads())
            except Exception:
                intra_now = int(cpus)

            want_inter = max(1, min(tuned_workers // 2, 4))
            total = int(intra_now) + int(want_inter) + int(tuned_workers)
            if total > int(thread_cap):
                new_intra = max(1, int(thread_cap) - int(want_inter) - int(tuned_workers))
                if int(new_intra) != int(intra_now):
                    optimize_threads(intra=int(new_intra))
                    intra_now = int(new_intra)

                total = int(intra_now) + int(want_inter) + int(tuned_workers)
                if total > int(thread_cap):
                    want_inter = max(1, int(thread_cap) - int(tuned_workers) - int(intra_now))

            optimize_threads(inter=int(want_inter))
            return

        self._retune_threads()

    def _retune_threads(self) -> None:
        if not self._enabled or self._samples < 128:
            return
        now = time.perf_counter()
        if (now - self._last_retune_ts) < 1.0:
            return

        with self._lock:
            cpu_ns, wall_ns = self._cpu_ns, self._wall_ns
            self._cpu_ns = self._wall_ns = self._samples = 0
        self._last_retune_ts = now

        if wall_ns <= 0:
            return

        cpus = max(1, len(self._allowed_cpus))
        workers = max(1, self._io_workers)

        dev_type, nacc = WorkerPolicy._detect_accelerator()
        is_accel = bool(nacc and int(nacc) > 0)

        cap_mult = _default_cap_mult(cpus, is_accel=is_accel, nogil=bool(self._nogil))
        local_world = _effective_local_world_size(1)
        distribute_default = local_world > 1
        distribute = bool(env_first_int(("STNET_DISTRIBUTE_THREAD_CAP",), int(distribute_default)))
        thread_cap = _effective_thread_cap(
            ncpu=cpus,
            cap_mult=cap_mult,
            local_world=local_world,
            distribute=bool(distribute),
        )

        try:
            intra = max(1, int(torch.get_num_threads()))
        except Exception:
            intra = int(cpus)

        inter = 1
        if hasattr(torch, "get_num_interop_threads"):
            with contextlib.suppress(Exception):
                inter = max(1, int(torch.get_num_interop_threads()))

        total = int(intra) + int(inter) + int(workers)
        if total <= int(thread_cap):
            return

        new_intra = max(1, int(thread_cap) - int(workers) - int(inter))
        if int(new_intra) < int(intra):
            optimize_threads(intra=int(new_intra))
            intra = int(new_intra)

        total = int(intra) + int(inter) + int(workers)
        if total > int(thread_cap):
            new_inter = max(1, int(thread_cap) - int(workers) - int(intra))
            if int(new_inter) < int(inter):
                optimize_threads(inter=int(new_inter))

    def new_thread(self, fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        if not self._enabled:
            return fn

        pin_thread = self.pin_thread
        tls = self._tls
        lock = self._lock
        tune = self.tune_threads
        sample_every = int(self._sample_every) if int(self._sample_every) > 0 else 1
        flush_every = int(self._flush_every) if int(self._flush_every) > 0 else 1
        perf_counter_ns = time.perf_counter_ns
        thread_time_ns = getattr(time, "thread_time_ns", None)

        def _inner(x: Any) -> Any:
            pin_thread()

            local_samples = getattr(tls, "samples", 0) + 1
            setattr(tls, "samples", local_samples)

            do_sample = (sample_every <= 1) or (local_samples % sample_every == 0)

            if do_sample:
                t0 = perf_counter_ns()
                tc0 = thread_time_ns() if callable(thread_time_ns) else 0
                y = fn(x)
                tc1 = thread_time_ns() if callable(thread_time_ns) else 0
                t1 = perf_counter_ns()

                scale = int(sample_every) if int(sample_every) > 1 else 1
                d_cpu = max(0, int(tc1) - int(tc0)) * scale
                d_wall = max(0, int(t1) - int(t0)) * scale
            else:
                y = fn(x)
                d_cpu = 0
                d_wall = 0

            setattr(tls, "cpu_ns", getattr(tls, "cpu_ns", 0) + int(d_cpu))
            setattr(tls, "wall_ns", getattr(tls, "wall_ns", 0) + int(d_wall))

            if local_samples >= flush_every:
                with lock:
                    self._samples += int(getattr(tls, "samples", 0) or 0)
                    self._cpu_ns += int(getattr(tls, "cpu_ns", 0) or 0)
                    self._wall_ns += int(getattr(tls, "wall_ns", 0) or 0)
                setattr(tls, "samples", 0)
                setattr(tls, "cpu_ns", 0)
                setattr(tls, "wall_ns", 0)
                tune()

            return y

        return _inner

    def optimize_procs(self, io_workers: int) -> int:
        if not self._enabled:
            return int(io_workers)
        cpus = max(1, len(self._allowed_cpus))
        tuned = max(1, min(int(io_workers), cpus))
        self._io_workers = tuned
        return tuned


_TLB_SINGLETON: Optional[Thread] = None
_TLB_SINGLETON_LOCK = Lock()


def get_tlb(io_workers: Optional[int] = None) -> Thread:
    global _TLB_SINGLETON
    if _TLB_SINGLETON is None:
        with _TLB_SINGLETON_LOCK:
            if _TLB_SINGLETON is None:
                default_workers = io_workers if io_workers is not None else max(1, process_cpu_count() // 2)
                _TLB_SINGLETON = Thread(io_workers=int(default_workers))
    tlb = _TLB_SINGLETON
    if tlb is not None and io_workers is not None:
        with contextlib.suppress(Exception):
            tlb.optimize_procs(int(io_workers))
    return tlb


def worker_init_pin(_: Any) -> None:
    get_tlb().pin_thread()


def wrap_with_tlb(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    return get_tlb().new_thread(fn)


class Memory:
    @staticmethod
    def total() -> Optional[int]:
        try:
            import psutil  # type: ignore

            vm = psutil.virtual_memory()
            if getattr(vm, "total", None):
                return int(vm.total)
        except Exception:
            pass
        try:
            if sys.platform.startswith("linux"):
                with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        if line.startswith("MemTotal:"):
                            parts = line.split()
                            if len(parts) >= 2 and parts[1].isdigit():
                                return int(parts[1]) * 1024
            elif sys.platform == "darwin":
                import subprocess

                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode("utf-8", "ignore")
                if out.strip().isdigit():
                    return int(out.strip())
            elif os.name == "nt" or sys.platform.startswith("win"):
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                    return int(stat.ullTotalPhys)
        except Exception:
            pass
        return None

    @staticmethod
    def available() -> int:
        base = Memory._sys_available()
        cgroup = Memory._linux_limit()
        winjob = Memory._windows_limit()
        rlim = Memory._bsd_limit()
        candidates = [x for x in (base, cgroup, winjob, rlim) if isinstance(x, int) and x >= 0]
        return max(0, min(candidates)) if candidates else 0

    @staticmethod
    def _parse_free_total(info: Any) -> Tuple[Optional[int], Optional[int]]:
        free: Optional[int] = None
        total: Optional[int] = None

        if isinstance(info, (list, tuple)):
            if len(info) >= 1:
                with contextlib.suppress(Exception):
                    free = int(info[0])
            if len(info) >= 2:
                with contextlib.suppress(Exception):
                    total = int(info[1])

        elif isinstance(info, dict):
            free_v = info.get("free") or info.get("free_memory") or info.get("free_bytes")
            total_v = info.get("total") or info.get("total_memory") or info.get("total_bytes")

            if total_v is None and info.get("bytes_limit", None) is not None:
                total_v = info.get("bytes_limit", None)
            if free_v is None and total_v is not None:
                used_v = info.get("bytes_used") or info.get("bytes_in_use") or info.get("bytes_used_current")
                if used_v is not None:
                    with contextlib.suppress(Exception):
                        free_v = int(total_v) - int(used_v)

            with contextlib.suppress(Exception):
                if free_v is not None:
                    free = int(free_v)
            with contextlib.suppress(Exception):
                if total_v is not None:
                    total = int(total_v)

        if free is not None:
            free = max(0, int(free))
        if total is not None:
            total = max(0, int(total))
        return free, total

    @staticmethod
    def device_mem_get_info(device: torch.device) -> Tuple[Optional[int], Optional[int]]:
        free: Optional[int] = None
        total: Optional[int] = None

        with contextlib.suppress(Exception):
            acc = getattr(torch, "accelerator", None)
            if acc is not None:
                get_memory_info = getattr(acc, "get_memory_info", None)
                if callable(get_memory_info):
                    info = get_memory_info(device)
                    free, total = Memory._parse_free_total(info)

                if free is None or total is None:
                    mem_mod = getattr(acc, "memory", None)
                    mem_get_info = getattr(mem_mod, "mem_get_info", None) if mem_mod is not None else None
                    if callable(mem_get_info):
                        info = mem_get_info(device)
                        f2, t2 = Memory._parse_free_total(info)
                        if free is None:
                            free = f2
                        if total is None:
                            total = t2

        dev_t = getattr(device, "type", "cpu")

        if free is None or total is None:
            if dev_t == "cuda" and torch.cuda.is_available():
                with contextlib.suppress(Exception):
                    f2, t2 = torch.cuda.mem_get_info(device=device)
                    if free is None:
                        free = int(f2)
                    if total is None:
                        total = int(t2)

            elif dev_t == "xpu" and hasattr(torch, "xpu"):
                with contextlib.suppress(Exception):
                    mem_mod = getattr(torch.xpu, "memory", None)
                    mem_get_info = getattr(mem_mod, "mem_get_info", None) if mem_mod is not None else None
                    if not callable(mem_get_info):
                        mem_get_info = getattr(torch.xpu, "mem_get_info", None)
                    if callable(mem_get_info):
                        f2, t2 = mem_get_info(device)
                        if free is None:
                            free = int(f2)
                        if total is None:
                            total = int(t2)

            elif dev_t == "mps" and hasattr(torch, "mps"):
                with contextlib.suppress(Exception):
                    total_v = int(torch.mps.recommended_max_memory())
                    used_v = int(torch.mps.driver_allocated_memory())
                    if total_v > 0:
                        if total is None:
                            total = total_v
                        if free is None:
                            free = max(0, total_v - used_v)

        if free is not None:
            free = max(0, int(free))
        if total is not None:
            total = max(0, int(total))
        return free, total

    @staticmethod
    def _sys_available() -> Optional[int]:
        try:
            import psutil  # type: ignore

            vm = psutil.virtual_memory()
            if getattr(vm, "available", None) is not None:
                return int(vm.available)
            if getattr(vm, "total", 0) and getattr(vm, "used", None) is not None:
                return int(vm.total - vm.used)
        except Exception:
            pass
        try:
            if sys.platform.startswith("linux"):
                with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        if line.startswith("MemAvailable:"):
                            parts = line.split()
                            if len(parts) >= 2 and parts[1].isdigit():
                                return int(parts[1]) * 1024
            elif sys.platform == "darwin":
                import subprocess

                out = subprocess.check_output(["vm_stat"]).decode("utf-8", "ignore")
                page = None
                free = None
                inactive = None
                speculative = 0
                for ln in out.splitlines():
                    if "page size of" in ln:
                        page = int(ln.split()[-2])
                    elif ln.startswith("Pages free:"):
                        free = int(ln.split(":")[1].split()[0])
                    elif ln.startswith("Pages inactive:"):
                        inactive = int(ln.split(":")[1].split()[0])
                    elif ln.startswith("Pages speculative:"):
                        speculative = int(ln.split(":")[1].split()[0])
                if page and free is not None and inactive is not None:
                    return int((free + inactive + (speculative or 0)) * page)
            elif os.name == "nt" or sys.platform.startswith("win"):
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                    return int(stat.ullAvailPhys)
        except Exception:
            pass
        return None

    @staticmethod
    def _linux_limit() -> Optional[int]:
        if not sys.platform.startswith("linux"):
            return None

        def _read_int(path: str) -> Optional[int]:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    s = f.read().strip()
                if not s or s == "max":
                    return None
                return int(s)
            except Exception:
                return None

        try:
            root = "/sys/fs/cgroup"
            if os.path.exists(os.path.join(root, "cgroup.controllers")):
                rel = "/"
                with open("/proc/self/cgroup", "r", encoding="utf-8", errors="ignore") as fh:
                    for ln in fh:
                        parts = ln.strip().split(":")
                        if len(parts) >= 3 and parts[0] == "0":
                            rel = parts[2] or "/"
                            break
                grp = os.path.join(root, rel.lstrip("/"))
                lim = _read_int(os.path.join(grp, "memory.max"))
                cur = _read_int(os.path.join(grp, "memory.current"))
                if lim is not None and cur is not None:
                    return max(0, lim - cur)

            mem_rel = None
            with open("/proc/self/cgroup", "r", encoding="utf-8", errors="ignore") as fh:
                for ln in fh:
                    parts = ln.strip().split(":")
                    if len(parts) >= 3 and "memory" in parts[1].split(","):
                        mem_rel = parts[2] or "/"
                        break
            if mem_rel is not None:
                grp = os.path.join("/sys/fs/cgroup/memory", mem_rel.lstrip("/"))
                lim = _read_int(os.path.join(grp, "memory.limit_in_bytes"))
                use = _read_int(os.path.join(grp, "memory.usage_in_bytes"))
                if lim is not None and use is not None:
                    if lim >= (1 << 60):
                        return None
                    return max(0, lim - use)
        except Exception:
            return None
        return None

    @staticmethod
    def _windows_limit() -> Optional[int]:
        if not (os.name == "nt" or sys.platform.startswith("win")):
            return None
        try:
            from ctypes import wintypes as wt

            import psutil  # type: ignore

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            IsProcessInJob = kernel32.IsProcessInJob
            IsProcessInJob.argtypes = [wt.HANDLE, wt.HANDLE, ctypes.POINTER(wt.BOOL)]
            IsProcessInJob.restype = wt.BOOL
            hProc = kernel32.GetCurrentProcess()
            inJob = wt.BOOL()
            if not IsProcessInJob(hProc, None, ctypes.byref(inJob)) or not inJob.value:
                return None

            class LARGE_INTEGER(ctypes.Union):
                _fields_ = [("QuadPart", ctypes.c_longlong)]

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_ulonglong),
                    ("WriteOperationCount", ctypes.c_ulonglong),
                    ("OtherOperationCount", ctypes.c_ulonglong),
                    ("ReadTransferCount", ctypes.c_ulonglong),
                    ("WriteTransferCount", ctypes.c_ulonglong),
                    ("OtherTransferCount", ctypes.c_ulonglong),
                ]

            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", LARGE_INTEGER),
                    ("PerJobUserTimeLimit", LARGE_INTEGER),
                    ("LimitFlags", ctypes.c_uint),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", ctypes.c_uint),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", ctypes.c_uint),
                    ("SchedulingClass", ctypes.c_uint),
                ]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            JobObjectExtendedLimitInformation = 9
            QueryInformationJobObject = kernel32.QueryInformationJobObject
            QueryInformationJobObject.argtypes = [
                wt.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_ulong,
                ctypes.POINTER(ctypes.c_ulong),
            ]
            QueryInformationJobObject.restype = wt.BOOL
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            retlen = ctypes.c_ulong(0)
            ok = QueryInformationJobObject(
                None,
                JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
                ctypes.byref(retlen),
            )
            if not ok:
                return None

            JOB_OBJECT_LIMIT_WORKINGSET = 0x00000001
            JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
            JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200

            flags = int(getattr(info.BasicLimitInformation, "LimitFlags", 0))
            cand_limits: list[int] = []

            if flags & JOB_OBJECT_LIMIT_PROCESS_MEMORY:
                v = int(getattr(info, "ProcessMemoryLimit", 0))
                if 0 < v < (1 << 60):
                    cand_limits.append(v)
            if flags & JOB_OBJECT_LIMIT_JOB_MEMORY:
                v = int(getattr(info, "JobMemoryLimit", 0))
                if 0 < v < (1 << 60):
                    cand_limits.append(v)

            if flags & JOB_OBJECT_LIMIT_WORKINGSET:
                v = int(getattr(info.BasicLimitInformation, "MaximumWorkingSetSize", 0))
                if 0 < v < (1 << 60):
                    cand_limits.append(v)

            if not cand_limits:
                return None

            rss = int(psutil.Process(os.getpid()).memory_info().rss)
            avail_candidates = [max(0, lim - rss) for lim in cand_limits]
            return max(0, min(avail_candidates)) if avail_candidates else None
        except Exception:
            return None

    @staticmethod
    def _bsd_limit() -> Optional[int]:
        try:
            import resource

            import psutil  # type: ignore

            rss = psutil.Process(os.getpid()).memory_info().rss
            cand: list[int] = []
            for name in ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_RSS"):
                lim = getattr(resource, name, None)
                if lim is None:
                    continue
                soft, _ = resource.getrlimit(lim)
                if soft == getattr(resource, "RLIM_INFINITY", -1) or soft <= 0:
                    continue
                cand.append(max(0, int(soft) - int(rss)))
            return min(cand) if cand else None
        except Exception:
            return None

    @staticmethod
    def prefer_local_numa() -> bool:
        try:
            import numa  # type: ignore

            if hasattr(numa, "available") and numa.available():
                node = numa.current_node()
                numa.set_membind([node])
                return True
        except Exception:
            pass
        try:
            if sys.platform.startswith("linux"):
                lib = ctypes.CDLL("libnuma.so.1")
                if int(lib.numa_available()) < 0:
                    return False
                cpu = 0
                if hasattr(os, "sched_getaffinity"):
                    cpus = list(os.sched_getaffinity(0))
                    cpu = int(cpus[0]) if cpus else 0
                lib.numa_node_of_cpu.argtypes = [ctypes.c_int]
                lib.numa_node_of_cpu.restype = ctypes.c_int
                node = int(lib.numa_node_of_cpu(ctypes.c_int(cpu)))
                lib.numa_set_preferred.argtypes = [ctypes.c_int]
                lib.numa_set_preferred.restype = None
                lib.numa_set_preferred(ctypes.c_int(node))
                return True
        except Exception:
            return False
        return False
