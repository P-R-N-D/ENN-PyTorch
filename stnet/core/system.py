# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import ctypes
import gc
import importlib
import json
import logging
import math
import multiprocessing
import os
import platform
import sys
import sysconfig
import threading
import time
from functools import lru_cache
from dataclasses import dataclass, replace
from datetime import timezone, tzinfo
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import torch
import torch.multiprocessing

from .datatypes import (
    env_bool,
    env_first,
    env_first_float,
    env_first_int,
    env_int,
    parse_bool,
)

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


if TYPE_CHECKING:
    from .concurrency import Mutex as _Mutex
    _FP32_PRECISION_LOCK: _Mutex
    _EMPTY_CACHE_LOCK: _Mutex
    _DEVICE_STATS_LOCK: _Mutex
    _CPU_PROC_LOCK: _Mutex
    _RUNTIME_CFG_LOCK: _Mutex
    
_LOGGER = logging.getLogger(__name__)

_FP32_PRECISION_CACHE: dict[str, str] = {}
_EMPTY_CACHE_LAST_CALL_S_BY_DEVICE: dict[Tuple[str, int], float] = {}
_DEVICE_STATS_CACHE: dict[Tuple[str, int], "Device"] = {}
_CPU_PROC_CACHE: Optional[int] = None

_RUNTIME_CFG = SimpleNamespace(
    deterministic=False,
    allow_tf32=None,
    cudnn_benchmark=None,
    matmul_precision=None,
    sdpa_backends=None,
    te_first=True,
)

_LAZY_LOCK_NAMES = {
    "_FP32_PRECISION_LOCK",
    "_EMPTY_CACHE_LOCK",
    "_DEVICE_STATS_LOCK",
    "_CPU_PROC_LOCK",
    "_RUNTIME_CFG_LOCK",
}

_LAZY_LOCK_INIT_LOCK = threading.Lock()

_TZ_ALIASES = {
    k: v
    for k, v in [
        ("Z", "UTC"),
        ("UTC", "UTC"),
        ("GMT", "Etc/GMT"),
        ("KST", "Asia/Seoul"),
        ("JST", "Asia/Tokyo"),
        ("HKT", "Asia/Hong_Kong"),
        ("SGT", "Asia/Singapore"),
        ("ICT", "Asia/Bangkok"),
        ("WITA", "Asia/Makassar"),
        ("WIT", "Asia/Jayapura"),
        ("PHT", "Asia/Manila"),
        ("MYT", "Asia/Kuala_Lumpur"),
        ("AEST", "Australia/Sydney"),
        ("AEDT", "Australia/Sydney"),
        ("ACST", "Australia/Adelaide"),
        ("ACDT", "Australia/Adelaide"),
        ("AWST", "Australia/Perth"),
        ("NZST", "Pacific/Auckland"),
        ("NZDT", "Pacific/Auckland"),
        ("CET", "Europe/Berlin"),
        ("CEST", "Europe/Berlin"),
        ("EET", "Europe/Athens"),
        ("EEST", "Europe/Athens"),
        ("WET", "Europe/Lisbon"),
        ("WEST", "Europe/Lisbon"),
        ("BST", "Europe/London"),
        ("IST", "Europe/Dublin"),
        ("MSK", "Europe/Moscow"),
        ("TRT", "Europe/Istanbul"),
        ("ET", "America/New_York"),
        ("CT", "America/Chicago"),
        ("MT", "America/Denver"),
        ("PT", "America/Los_Angeles"),
        ("EST", "America/New_York"),
        ("EDT", "America/New_York"),
        ("CST", "America/Chicago"),
        ("CDT", "America/Chicago"),
        ("MST", "America/Denver"),
        ("MDT", "America/Denver"),
        ("PST", "America/Los_Angeles"),
        ("PDT", "America/Los_Angeles"),
        ("AKST", "America/Anchorage"),
        ("AKDT", "America/Anchorage"),
        ("HST", "Pacific/Honolulu"),
        ("AST", "America/Halifax"),
        ("ADT", "America/Halifax"),
        ("NST", "America/St_Johns"),
        ("NDT", "America/St_Johns"),
        ("BRT", "America/Sao_Paulo"),
        ("ART", "America/Argentina/Buenos_Aires"),
        ("SAST", "Africa/Johannesburg"),
        ("EAT", "Africa/Nairobi"),
        ("CAT", "Africa/Maputo"),
        ("WAT", "Africa/Lagos"),
    ]
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_LOCK_NAMES:
        g = globals()
        lock = g.get(name, None)
        if lock is not None:
            return lock
        with _LAZY_LOCK_INIT_LOCK:
            lock = g.get(name, None)
            if lock is None:
                from .concurrency import Mutex

                lock = Mutex()
                g[name] = lock
        return lock
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _mutex_lock(name: str):
    return getattr(sys.modules[__name__], name)


def _device_from(device: Optional[Union[torch.device, str]]) -> torch.device:
    if device is not None:
        return device if isinstance(device, torch.device) else torch.device(str(device))
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _log_msg(logger, msg, level):
    try:
        (logger(msg) if callable(logger) else getattr(logger, level)(msg)) if logger else getattr(
            _LOGGER, level
        )(msg)
    except:
        getattr(_LOGGER, level)(msg)


def _log_info(logger, msg):
    _log_msg(logger, msg, "info")


def _log_debug(logger, msg):
    _log_msg(logger, msg, "debug")


def _call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    with contextlib.suppress(Exception):
        if callable(fn):
            return fn(*args, **kwargs)
    return None


def _call_with_args(
    fn: Any,
    variants: Sequence[tuple[tuple[Any, ...], dict[str, Any]]],
) -> Any:
    if not callable(fn):
        return None
    for a, k in variants:
        try:
            return fn(*a, **k)
        except TypeError:
            continue
        except Exception:
            return None
    return None


def _clear_device_index(
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[str, int]:
    if device is None:
        return ("all", -1)
    try:
        dev = device if isinstance(device, torch.device) else torch.device(str(device))
    except (TypeError, ValueError, RuntimeError):
        return ("all", -1)
    idx = int(dev.index) if dev.index is not None else -1
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


def _read_text_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception:
        return None


def _read_int_file(path: str) -> Optional[int]:
    return int(s) if (s := _read_text_file(path)) and s != "max" and s.isdigit() else None


def _get_thread_limit(default: int) -> int:
    v = env_first_int(("STNET_THREAD_CAP_MULTIPLIER", "STNET_THREADS_CAP_MULTIPLIER"), default)
    return max(1, min(8, int(v)))


def _default_thread_limit(ncpu_raw: int, *args: Any, is_accel: bool, nogil: bool) -> int:
    cap_mult = (
        (3 if nogil and is_accel else (2 if is_accel else 1))
        if ncpu_raw > 8
        else (min(3 if nogil and is_accel else 2, 2) if ncpu_raw > 4 else 1)
    )
    return _get_thread_limit(int(cap_mult))


def _optimal_local_worlds(default: int) -> int:
    return max(
        1,
        env_first_int(
            ("STNET_LOCAL_WORLD_SIZE", "LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE"),
            int(default),
        ),
    )


def _optimal_threads(
    *args: Any, ncpu: int, cap_mult: int, local_world: int, distribute: bool
) -> int:
    node_thread_cap = max(2, int(ncpu) * int(cap_mult))
    if distribute and int(local_world) > 1:
        return max(2, int(node_thread_cap) // max(1, int(local_world)))
    return int(node_thread_cap)


def _get_allowed_cpu_linux() -> Optional[float]:
    if not sys.platform.startswith("linux"):
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
            raw = _read_text_file(os.path.join(grp, "cpu.max"))
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
        for cand in (
            "/sys/fs/cgroup/cpu",
            "/sys/fs/cgroup/cpu,cpuacct",
            "/sys/fs/cgroup/cpuacct",
        ):
            if os.path.isdir(cand):
                base = cand
                break
        if base is None:
            return None
        grp = os.path.join(base, cpu_rel.lstrip("/"))
        quota = _read_int_file(os.path.join(grp, "cpu.cfs_quota_us"))
        period = _read_int_file(os.path.join(grp, "cpu.cfs_period_us"))
        if quota is None or period is None:
            return None
        if int(quota) <= 0 or int(quota) == -1 or int(period) <= 0:
            return None
        return float(quota) / float(period)
    except Exception:
        return None


def _get_allowed_cpu_darwin() -> Optional[int]:
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


def _get_allowed_cpu_windows() -> Optional[list[int]]:
    if platform.system() != "Windows":
        return None
    try:
        k32 = ctypes.windll.kernel32
    except Exception:
        return None
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
            c = (
                _call(get_group_procs, ctypes.c_ushort(i))
                or _call(get_group_procs, i)
                or 0
            )
            counts.append(max(0, int(c)))

        groups: list[int] | None = None
        get_pg_aff = getattr(k32, "GetProcessGroupAffinity", None)
        if callable(get_pg_aff):
            h = k32.GetCurrentProcess()
            cnt = ctypes.c_ushort(0)
            arr = (ctypes.c_ushort * group_count)()
            if get_pg_aff(h, ctypes.byref(cnt), arr) and cnt.value > 0:
                groups = [int(arr[i]) for i in range(int(cnt.value))]

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


def _parse_torch_dtype(value: str) -> Tuple[torch.dtype, ...]:
    entries: list[torch.dtype] = []
    for token in str(value).split(","):
        name = token.strip()
        if not name:
            continue
        dtype = getattr(torch, name, None)
        if isinstance(dtype, torch.dtype):
            entries.append(dtype)
    return tuple(entries)


def _get_allowed_cpu_fallback() -> list[int]:
    n: Optional[int] = None
    with contextlib.suppress(Exception):
        v = os.cpu_count()
        if isinstance(v, int) and v > 0:
            n = int(v)
    if n is None and platform.system() == "Darwin":
        with contextlib.suppress(Exception):
            v = _get_allowed_cpu_darwin()
            if isinstance(v, int) and v > 0:
                n = int(v)
    n = int(n) if isinstance(n, int) and n > 0 else 1
    return list(range(n))


def _get_cgroup_quota() -> int:
    quota = _get_allowed_cpu_linux()
    if quota is None or quota <= 0:
        return 0
    with contextlib.suppress(Exception):
        return max(0, int(math.floor(float(quota))))
    return 0


def _acc_mod(dt: str):
    return getattr(torch, dt, None) if dt in ("cuda", "xpu", "mps") else accelerator_type(dt)


def _acc_op(dt, op, default=None):
    return _call(getattr(_acc_mod(dt), op, None)) or default


def empty_device_cache(
    *args: Any,
    device: Optional[Union[torch.device, str]] = None,
    do_gc: bool = True,
    min_interval_s: Optional[float] = None,
) -> None:
    if not env_bool("STNET_EMPTY_CACHE", True):
        return
    if min_interval_s is None:
        min_interval_s = env_first_float(("STNET_EMPTY_CACHE_MIN_INTERVAL_S",), 0.5)
    with contextlib.suppress(Exception):
        min_interval_s = float(min_interval_s)
    if not isinstance(min_interval_s, (int, float)):
        min_interval_s = 0.5
    if float(min_interval_s) < 0:
        min_interval_s = 0.0
    now = time.monotonic()
    key = _clear_device_index(device)
    with _mutex_lock("_EMPTY_CACHE_LOCK"):
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
    target = None
    with contextlib.suppress(Exception):
        if device is not None:
            target = device if isinstance(device, torch.device) else torch.device(str(device))
    dt = getattr(target, "type", None) if target is not None else None
    dt_s = str(dt or "all")
    if dt_s == "all":
        if torch.cuda.is_available():
            if target is not None and target.index is not None:
                with torch.cuda.device(int(target.index)):
                    _call(getattr(torch.cuda, "empty_cache", None))
            else:
                _call(getattr(torch.cuda, "empty_cache", None))
        mps_mod = getattr(torch, "mps", None)
        _call(getattr(mps_mod, "empty_cache", None))
        xpu_mod = getattr(torch, "xpu", None)
        if _call(getattr(xpu_mod, "empty_cache", None)) is None:
            memory_mod = getattr(xpu_mod, "memory", None) if xpu_mod is not None else None
            _call(getattr(memory_mod, "empty_cache", None))
        return
    match dt_s:
        case "cuda":
            if torch.cuda.is_available():
                if target is not None and target.index is not None:
                    with torch.cuda.device(int(target.index)):
                        _call(getattr(torch.cuda, "empty_cache", None))
                else:
                    _call(getattr(torch.cuda, "empty_cache", None))
        case "mps":
            mps_mod = getattr(torch, "mps", None)
            _call(getattr(mps_mod, "empty_cache", None))
        case "xpu":
            xpu_mod = getattr(torch, "xpu", None)
            if _call(getattr(xpu_mod, "empty_cache", None)) is None:
                memory_mod = getattr(xpu_mod, "memory", None) if xpu_mod is not None else None
                _call(getattr(memory_mod, "empty_cache", None))
        case _:
            pass


def is_oom_error(exc: BaseException) -> bool:
    with contextlib.suppress(Exception):
        typ = getattr(torch, "OutOfMemoryError", None)
        if isinstance(typ, type) and isinstance(exc, typ):
            return True
    for mod_name in ("cuda", "xpu", "mps"):
        with contextlib.suppress(Exception):
            mod = getattr(torch, mod_name, None)
            typ = getattr(mod, "OutOfMemoryError", None) if mod is not None else None
            if isinstance(typ, type) and isinstance(exc, typ):
                return True
    msg = str(exc).lower()
    if not msg:
        return False
    patterns = (
        "out of memory",
        "cuda out of memory",
        "cuda error: out of memory",
        "hip out of memory",
        "xpu out of memory",
        "mps backend out of memory",
        "not enough memory",
        "failed to allocate memory",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
    )
    return any(p in msg for p in patterns)


def accelerator_type(dev_type: str) -> Optional[ModuleType]:
    dt = str(dev_type or "cpu").strip().lower()
    if dt in ("cuda", "xpu", "mps"):
        return getattr(torch, dt, None)
    acc = getattr(torch, "accelerator", None)
    return acc if acc and getattr(acc, "device_type", None) == dt else None


@lru_cache(maxsize=8)
def is_accelerator_available(dev_type: str) -> bool:
    dt = str(dev_type or "cpu").strip().lower()
    if dt == "mps":
        return bool(
            _call(getattr(torch.backends.mps, "is_available", None)) or _acc_op(dt, "is_available")
        )
    return bool(_acc_op(dt, "is_available"))


@lru_cache(maxsize=8)
def get_num_accelerators(dev_type: str) -> int:
    dt = str(dev_type or "cpu").strip().lower()
    if not is_accelerator_available(dt):
        return 0
    if dt == "mps":
        return 1
    return max(0, int(_acc_op(dt, "device_count") or 0))


def get_accelerator_index(dev_type: str) -> int:
    dt = str(dev_type or "cpu").strip().lower()
    if not is_accelerator_available(dt):
        return -1
    if dt == "mps":
        return 0
    v = _acc_op(dt, "current_device")
    return int(v) if v is not None else 0


def set_accelerator_index(dev_type: str, idx: int) -> None:
    dt = str(dev_type or "cpu").strip().lower()
    if is_accelerator_available(dt):
        mod = _acc_mod(dt)
        _call(getattr(mod, "set_device", None), int(idx))


def set_accelerator_seed(seed: int) -> None:
    for dt in ("cuda", "xpu"):
        if is_accelerator_available(dt):
            _call(getattr(_acc_mod(dt), "manual_seed_all", None), int(seed))


def available_device_memory(device: Union[torch.device, str]) -> Optional[float]:
    try:
        dev = device if isinstance(device, torch.device) else torch.device(str(device))
    except Exception:
        return None
    if dev.type not in {"cuda", "xpu", "mps"}:
        return None
    try:
        free_b, total_b = Memory.mem_get_info(dev)
        if total_b is not None and int(total_b) > 0 and free_b is not None:
            used_b = max(0, int(total_b) - int(free_b))
            return 100.0 * float(used_b) / float(total_b)
    except:
        pass

    try:
        mod = _acc_mod(dev.type)
        if dev.type == "cuda":
            props = _call(mod.get_device_properties, dev.index or 0)
            total, used = getattr(props, "total_memory", 0), _call(mod.memory_allocated, dev)
        elif dev.type == "xpu":
            props = _call(mod.get_device_properties, dev.index or 0)
            total, used = getattr(props, "total_memory", 0), _call(mod.memory_allocated, dev)
        elif dev.type == "mps":
            total, used = _call(mod.recommended_max_memory), _call(mod.current_allocated_memory)
        else:
            return None
        return (100.0 * used / total) if total and used is not None else None
    except:
        return None
    return None


def available_accelerator_memory(device: Union[torch.device, str]) -> Optional[int]:
    try:
        dev = device if isinstance(device, torch.device) else torch.device(str(device))
    except Exception:
        return None
    v = _call(getattr(Memory, "device_mem_get_info", None), dev)
    if isinstance(v, tuple) and len(v) >= 2 and v[1]:
        return int(v[1])

    if not is_accelerator_available(dev.type):
        return None
    mod = _acc_mod(dev.type)
    try:
        if dev.type in ("cuda", "xpu"):
            props = _call(mod.get_device_properties, dev.index if dev.index is not None else 0)
            return int(props.total_memory) if props else None
        if dev.type == "mps":
            return int(_call(mod.recommended_max_memory))
    except:
        pass
    return None


def allocated_accelerator_memory(device: Union[torch.device, str]) -> Optional[int]:
    try:
        dev = device if isinstance(device, torch.device) else torch.device(str(device))
    except Exception:
        return None
    if not is_accelerator_available(dev.type):
        return None

    mod = _acc_mod(dev.type)
    if dev.type == "mps":
        v = _call(mod.current_allocated_memory)
    else:
        v = _call(mod.memory_allocated, dev)

    return max(0, int(v)) if v is not None else None
    return None


def flush_accelerator_memory_stats(device: Union[torch.device, str]) -> None:
    try:
        dev = device if isinstance(device, torch.device) else torch.device(str(device))
    except Exception:
        return
    if not is_accelerator_available(dev.type):
        return

    mod = _acc_mod(dev.type)
    if dev.type == "mps":
        _call(mod.reset_peak_memory_stats)
    else:
        _call(mod.reset_peak_memory_stats, dev)


def accelerator_max_allocated_memory(device: Union[torch.device, str]) -> Optional[int]:
    try:
        dev = device if isinstance(device, torch.device) else torch.device(str(device))
    except Exception:
        return None
    if not is_accelerator_available(dev.type):
        return None

    mod = _acc_mod(dev.type)
    if dev.type == "mps":
        v = _call(mod.max_memory_allocated)
    else:
        v = _call(mod.max_memory_allocated, dev)

    return max(0, int(v)) if v is not None else None
    return None


def collect_accelerator_ipc() -> None:
    if not is_accelerator_available("cuda"):
        return
    _call(getattr(torch.cuda, "ipc_collect", None))


def accelerator(device: torch.device) -> contextlib.AbstractContextManager[None]:
    try:
        dev = device if isinstance(device, torch.device) else torch.device(str(device))
    except Exception:
        return contextlib.nullcontext()
    mod = _acc_mod(dev.type)
    if mod and hasattr(mod, "device"):
        return mod.device(dev if dev.index is None else dev.index)
    return contextlib.nullcontext()


def sync_accelerator(device: Union[torch.device, str]) -> None:
    try:
        dev = device if isinstance(device, torch.device) else torch.device(str(device))
        if is_accelerator_available(dev.type):
            mod = _acc_mod(dev.type)
            if dev.type == "mps":
                _call(mod.synchronize)
            else:
                _call(mod.synchronize, dev)
    except:
        pass


@lru_cache(maxsize=8)
def is_accelerator_timer_supported(dev_type: str) -> bool:
    dt = str(dev_type or "cpu").strip().lower()
    backend = accelerator_type(dt)
    if backend is None:
        return False
    avail = getattr(backend, "is_available", None)
    if callable(avail):
        v = _call(avail)
        if v is not None and not bool(v):
            return False
    Event = getattr(backend, "Event", None)
    if Event is None:
        return False
    ev0 = _call(Event, enable_timing=True)
    ev1 = _call(Event, enable_timing=True)
    if ev0 is None or ev1 is None:
        ev0 = _call(Event)
        ev1 = _call(Event)
        if ev0 is None or ev1 is None:
            return False
    need = ("record", "synchronize", "elapsed_time")
    for ev in (ev0, ev1):
        for name in need:
            if not callable(getattr(ev, name, None)):
                return False
    return True




def new_accelerator_event(
    device: Union[torch.device, str],
    *args: Any,
    enable_timing: bool = False,
) -> object | None:
    del args
    try:
        dev = device if isinstance(device, torch.device) else torch.device(str(device))
    except Exception:
        return None
    dt = str(getattr(dev, "type", "cpu") or "cpu")
    mod = _acc_mod(dt)
    if not mod or not hasattr(mod, "Event"):
        return None
    return _call(mod.Event, enable_timing=enable_timing) or _call(mod.Event)


@lru_cache(maxsize=8)
def is_stream_supported(dev_type: str) -> bool:
    dt = str(dev_type or "cpu").strip().lower()
    if dt not in {"cuda", "xpu"}:
        return False
    backend = accelerator_type(dt)
    if backend is None:
        return False
    avail = getattr(backend, "is_available", None)
    if callable(avail):
        v = _call(avail)
        if v is not None and not bool(v):
            return False
    Stream = getattr(backend, "Stream", None)
    Event = getattr(backend, "Event", None)
    stream_cm = getattr(backend, "stream", None)
    cur = getattr(backend, "current_stream", None)
    return bool(callable(Stream) and (Event is not None) and callable(stream_cm) and callable(cur))


@lru_cache(maxsize=8)
def is_pin_supported(dev_type: str) -> bool:
    dt = str(dev_type or "cpu").strip().lower()
    match dt:
        case "cuda" | "xpu":
            backend = accelerator_type(dt)
            if backend is None:
                return False
            avail = getattr(backend, "is_available", None)
            v = _call(avail)
            if v is not None and not bool(v):
                return False
            return bool(
                callable(getattr(backend, "current_stream", None))
                and getattr(backend, "Event", None) is not None
            )
        case _:
            return False


def accelerator_stream(stream: object, dev_type: str) -> contextlib.AbstractContextManager[None]:
    mod = _acc_mod(dev_type)
    return mod.stream(stream) if mod and hasattr(mod, "stream") else contextlib.nullcontext()


def new_accelerator_stream(device: torch.device) -> object | None:
    try:
        dev = device if isinstance(device, torch.device) else torch.device(str(device))
    except Exception:
        return None
    dt = str(getattr(dev, "type", "cpu") or "cpu")
    mod = _acc_mod(dt)
    return mod.Stream(device=dev) if mod and hasattr(mod, "Stream") else None


def current_accelerator_stream(device: torch.device) -> object | None:
    try:
        dev = device if isinstance(device, torch.device) else torch.device(str(device))
        mod = _acc_mod(dev.type)
        return mod.current_stream(device=dev) if mod and hasattr(mod, "current_stream") else None
    except:
        return None


def set_float32_precision(
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    autocast_dtype: Optional[torch.dtype] = None,
    enable_tf32: bool = True,
) -> None:
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
    with _mutex_lock("_FP32_PRECISION_LOCK"):
        if _FP32_PRECISION_CACHE.get(key) == precision:
            return
        _FP32_PRECISION_CACHE[key] = precision
    cuda_b = getattr(torch.backends, "cuda", None)
    matmul = getattr(cuda_b, "matmul", None) if cuda_b is not None else None
    if matmul is not None and hasattr(matmul, "fp32_precision"):
        _call(setattr, matmul, "fp32_precision", precision)
    cudnn = getattr(torch.backends, "cudnn", None)
    conv = getattr(cudnn, "conv", None) if cudnn is not None else None
    rnn = getattr(cudnn, "rnn", None) if cudnn is not None else None
    if conv is not None and hasattr(conv, "fp32_precision"):
        _call(setattr, conv, "fp32_precision", precision)
    if rnn is not None and hasattr(rnn, "fp32_precision"):
        _call(setattr, rnn, "fp32_precision", precision)


def timezone_from(name: Optional[str] = None) -> Optional[tzinfo]:
    resolved = (name or "GMT").strip()
    alias = _TZ_ALIASES.get(resolved.upper(), resolved)
    if alias.upper() == "UTC":
        return timezone.utc
    if ZoneInfo is None:
        return None
    with contextlib.suppress(Exception):
        return ZoneInfo(alias)
    return None


def time_ns() -> int:
    return int(time.time_ns())


def posix_time(tz_name: Optional[str] = None) -> int:
    _ = tz_name
    return time_ns()


def system_info() -> Tuple[str, str, str, str]:
    sysname = platform.system() or ""
    release = platform.release() or ""
    kernel = f"{sysname} {release}".strip()
    os_name = sysname
    if sysname == "Linux" and hasattr(platform, "freedesktop_os_release"):
        info = _call(getattr(platform, "freedesktop_os_release", None))
        if isinstance(info, dict):
            name = info.get("NAME") or "Linux"
            version = info.get("VERSION_ID") or ""
            os_name = (name if not version else f"{name} {version}").strip()
    elif sysname == "Windows":
        win = _call(getattr(platform, "win32_ver", None))
        if isinstance(win, (list, tuple)) and win:
            os_name = f"Windows {win[0] or ''}".strip()
    elif sysname == "Darwin":
        mac = _call(getattr(platform, "mac_ver", None))
        if isinstance(mac, (list, tuple)) and mac:
            os_name = f"macOS {mac[0] or ''}".strip()
    arch = platform.machine() or ""
    accelerators: list[str] = []
    for dt in ("cuda", "xpu"):
        if is_accelerator_available(dt):
            for idx in range(get_num_accelerators(dt)):
                name = _call(getattr(_acc_mod(dt), "get_device_name", None), idx) or dt.upper()
                accelerators.append(f"{dt}:{idx}={name}")
    if is_accelerator_available("mps"):
        accelerators.append("mps=Apple MPS")
    return os_name, kernel, arch, ";".join(accelerators)


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


def init_python_path() -> str:
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
    current = torch.multiprocessing.get_start_method(allow_none=True)
    platform = sys.platform
    if platform.startswith("win"):
        candidates = ("spawn",)
    elif platform.startswith(("darwin", "linux")):
        candidates = ("forkserver", "spawn")
    else:
        candidates = ("spawn",)
    if current in candidates:
        return str(current)
    for method in candidates:
        try:
            multiprocessing.get_context(method)
        except ValueError:
            continue
        return method
    raise RuntimeError("No supported multiprocessing start method (tried forkserver, spawn).")


def init_start_method() -> None:
    with contextlib.suppress(RuntimeError):
        torch.multiprocessing.set_sharing_strategy("file_system")
    existing = torch.multiprocessing.get_start_method(allow_none=True)
    if existing is not None and existing != "fork":
        return
    last_error: Optional[BaseException] = None
    platform = sys.platform
    if platform.startswith("win"):
        candidates = ("spawn",)
    elif platform.startswith(("darwin", "linux")):
        candidates = ("forkserver", "spawn")
    else:
        candidates = ("spawn",)
    for method in candidates:
        try:
            multiprocessing.get_context(method)
        except ValueError as exc:
            last_error = exc
            continue
        try:
            for module in (multiprocessing, torch.multiprocessing):
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
        from torch.nn.attention import SDPBackend
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


def get_dpa_backends() -> list[object]:
    return get_sdpa_backends()


def cuda_compute_capability(device: Union[torch.device, str]) -> Tuple[int, int]:
    dev = _device_from(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return (0, 0)
    try:
        major, minor = torch.cuda.get_device_capability(dev)
        return (int(major), int(minor))
    except Exception:
        return (0, 0)


def is_cpu_bf16_supported() -> bool:
    try:
        mkldnn = getattr(torch.backends, "mkldnn", None)
        if mkldnn is None:
            return False
        if not bool(mkldnn.is_available()) or not bool(getattr(mkldnn, "enabled", True)):
            return False
        mkldnn_ops = getattr(torch.ops, "mkldnn", None)
        f = (
            getattr(mkldnn_ops, "_is_mkldnn_bf16_supported", None)
            if mkldnn_ops is not None
            else None
        )
        if callable(f):
            return bool(f())
    except Exception:
        return False
    return False


def is_cuda_bf16_supported(device: Optional[Union[torch.device, str]] = None) -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        dev = _device_from(device)
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


def is_float8_supported(
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[bool, str]:
    try:
        dev = _device_from(device)
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


def is_int8_supported(
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[bool, str]:
    try:
        dev = _device_from(device)
        if dev.type == "cuda" and torch.cuda.is_available():
            return (True, "Int8 supported on CUDA")
        return (True, "Int8 supported on CPU")
    except Exception:
        return (False, "Unknown int8 support")


def is_int4_supported(
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[bool, str]:
    try:
        dev = _device_from(device)
        if dev.type == "cuda" and torch.cuda.is_available():
            return (True, "Int4 supported on CUDA")
        return (True, "Int4 supported on CPU")
    except Exception:
        return (False, "Unknown int4 support")


def get_device_stats(device: Optional[Union[torch.device, str]] = None) -> Device:
    dev = _device_from(device)
    key = (str(dev.type), int(dev.index) if dev.index is not None else -1)
    with _mutex_lock("_DEVICE_STATS_LOCK"):
        cached = _DEVICE_STATS_CACHE.get(key)
        if cached is not None:
            return cached
    device_type = str(dev.type)
    cc = None
    if device_type == "cuda" and torch.cuda.is_available():
        major, minor = cuda_compute_capability(dev)
        cc = (int(major), int(minor)) if (major > 0 or minor > 0) else None
    float_env = env_first(("STNET_DATA_FLOAT_DTYPES", "STNET_FLOAT_DTYPES"))
    float_dtypes = _parse_torch_dtype(float_env) if float_env else tuple()
    int_env = env_first(("STNET_DATA_INT_DTYPES", "STNET_INT_DTYPES"))
    int_dtypes = _parse_torch_dtype(int_env) if int_env else tuple()
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
    with _mutex_lock("_DEVICE_STATS_LOCK"):
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
    del args, kwargs
    with _mutex_lock("_RUNTIME_CFG_LOCK"):
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
        det = bool(cfg.deterministic)
        allow_tf32_val = bool(cfg.allow_tf32)
        cudnn_bench_val = bool(cfg.cudnn_benchmark)
        matmul_prec = str(cfg.matmul_precision)
    _call(getattr(torch, "set_float32_matmul_precision", None), matmul_prec)

    if is_accelerator_available("cuda"):
        idx_env = env_first_int(
            ("LOCAL_RANK", "STNET_ACCELERATOR_INDEX", "STNET_DEVICE_INDEX"), default=0
        )
        ndev = max(1, int(get_num_accelerators("cuda") or 1))
        idx = 0
        with contextlib.suppress(Exception):
            idx = int(idx_env) % int(ndev)
        with _mutex_lock("_CPU_PROC_LOCK"):
            set_accelerator_index("cuda", int(idx))
        device = torch.device(f"cuda:{idx}")
        cudnn = getattr(torch.backends, "cudnn", None)
        if cudnn is not None:
            _call(setattr, cudnn, "deterministic", bool(det))
            _call(setattr, cudnn, "benchmark", bool(cudnn_bench_val))
        cuda_b = getattr(torch.backends, "cuda", None)
        matmul = getattr(cuda_b, "matmul", None) if cuda_b is not None else None
        if matmul is not None and hasattr(matmul, "allow_tf32"):
            _call(setattr, matmul, "allow_tf32", bool(allow_tf32_val))
        cudnn_b = getattr(torch.backends, "cudnn", None)
        if cudnn_b is not None and hasattr(cudnn_b, "allow_tf32"):
            _call(setattr, cudnn_b, "allow_tf32", bool(allow_tf32_val))
    elif is_accelerator_available("xpu"):
        idx = 0
        idx_env = env_first_int(("LOCAL_RANK",), default=0)
        ndev = max(1, int(get_num_accelerators("xpu") or 1))
        with contextlib.suppress(Exception):
            idx = int(idx_env) % int(ndev)
        set_accelerator_index("xpu", int(idx))
        device = torch.device(f"xpu:{idx}")
    elif is_accelerator_available("mps"):
        device = torch.device("mps")
    elif hasattr(torch, "is_vulkan_available") and torch.is_vulkan_available():
        device = torch.device("vulkan")
    else:
        device = torch.device("cpu")
    return device


def optimal_optimizer_params(
    device: torch.device, use_foreach: Optional[bool], use_fused: bool
) -> dict[str, bool]:
    devt = device.type
    flags: dict[str, bool] = {}
    flags["foreach"] = devt in {"cuda", "xpu"} if use_foreach is None else bool(use_foreach)
    if use_fused and devt in {"cuda", "xpu"}:
        flags["fused"] = True
        flags["foreach"] = False
    return flags


@dataclass(slots=True)
class Device:
    device: torch.device
    device_type: str
    cuda_cc: Optional[Tuple[int, int]]
    float_dtypes: Tuple[torch.dtype, ...]
    int_dtypes: Tuple[torch.dtype, ...]
    float8_dtypes: Tuple[torch.dtype, ...]
    int_quant_bits: int


class CPU:
    @staticmethod
    def allowed() -> list[int]:
        try:
            if os.name == "nt":
                cpus = _get_allowed_cpu_windows()
            elif sys.platform.startswith("linux"):
                cpus = os.sched_getaffinity(0)
            elif sys.platform == "darwin":
                cpus = _get_allowed_cpu_darwin()
            else:
                cpus = _get_allowed_cpu_fallback()
        except Exception:
            cpus = _get_allowed_cpu_fallback()

        if isinstance(cpus, set):
            cpus = list(cpus)
        if isinstance(cpus, (list, tuple)):
            cpus = [int(v) for v in cpus]
            return cpus or _get_allowed_cpu_fallback()
        if isinstance(cpus, int) and cpus > 0:
            return list(range(int(cpus)))

        spec = importlib.util.find_spec("psutil")
        if spec is not None:
            psutil = importlib.import_module("psutil")
            proc = psutil.Process()
            fn = getattr(proc, "cpu_affinity", None)
            if callable(fn):
                cpus = fn()
                if cpus:
                    return sorted({int(c) for c in cpus})
        return _get_allowed_cpu_fallback()

    @staticmethod
    def count() -> int:
        global _CPU_PROC_CACHE
        if _CPU_PROC_CACHE is not None:
            return _CPU_PROC_CACHE

        with _mutex_lock("_CPU_PROC_LOCK"):
            if _CPU_PROC_CACHE is not None:
                return _CPU_PROC_CACHE

            env_int = env_first_int(
                ("STNET_CPU_LIMIT", "STNET_CPU_COUNT", "OMP_NUM_THREADS"), default=0
            )
            xopt_int = 0
            with contextlib.suppress(Exception):
                xopt = getattr(sys, "_xoptions", {})
                if isinstance(xopt, dict):
                    xopt_int = int(xopt.get("stnet.cpu_limit", 0))

            base = max(env_int, xopt_int)
            if base <= 0:
                with contextlib.suppress(Exception):
                    fn = getattr(os, "process_cpu_count", None)
                    if callable(fn):
                        base = int(fn() or 0)
                if base <= 0:
                    base = int(os.cpu_count() or 1)

            base = max(1, int(base))

            allowed = CPU.allowed()
            if allowed:
                base = min(base, len(allowed))

            quota = _get_cgroup_quota()
            if quota > 0:
                base = min(base, quota)

            _CPU_PROC_CACHE = int(base)
            return _CPU_PROC_CACHE

    @staticmethod
    def info(max_bytes: Optional[int] = None) -> str:
        cpuinfo = None
        spec = importlib.util.find_spec("cpuinfo")
        if spec is not None:
            cpuinfo = importlib.import_module("cpuinfo")

        info: dict[str, Any] = {
            "cpu_count": int(CPU.count()),
            "os": sys.platform,
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version,
        }

        if cpuinfo is not None:
            with contextlib.suppress(Exception):
                raw = cpuinfo.get_CPU.info()
                if isinstance(raw, dict):
                    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
                    info["cpuinfo"] = raw

        out = json.dumps(info, sort_keys=True, ensure_ascii=True, default=str)
        if max_bytes is not None and max_bytes > 0:
            b = out.encode("utf-8")
            if len(b) > max_bytes:
                out = b[: max_bytes - 3].decode("utf-8", errors="ignore") + "..."
        return out

    @staticmethod
    def is_free_threaded_build() -> bool:
        val = sysconfig.get_config_var("Py_GIL_DISABLED")
        with contextlib.suppress(Exception):
            return bool(int(val or 0))
        return False

    @staticmethod
    def is_gil_enabled() -> bool:
        if hasattr(sys, "_is_gil_enabled"):
            with contextlib.suppress(Exception):
                return bool(sys._is_gil_enabled())
        return not CPU.is_free_threaded_build()

    @staticmethod
    def is_no_gil_enforced() -> bool:
        return CPU.is_free_threaded_build() and (not CPU.is_gil_enabled())

    @staticmethod
    def is_optimized_for_no_gil() -> bool:
        for key in (
            "STNET_NOGIL_OPTIMIZED",
            "STNET_NO_GIL_OPTIMIZED",
            "STNET_FREE_THREADED_OPTIMIZED",
        ):
            override = parse_bool(os.environ.get(key))
            if override is not None:
                return bool(override)
        return CPU.is_no_gil_enforced()


class Memory:
    @staticmethod
    def _parse_free(info: Any) -> Tuple[Optional[int], Optional[int]]:
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
                used_v = (
                    info.get("bytes_used")
                    or info.get("bytes_in_use")
                    or info.get("bytes_used_current")
                )
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
    def _sys_available_memory() -> Optional[int]:
        try:
            import psutil

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
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    return int(stat.ullAvailPhys)
        except Exception:
            pass
        return None

    @staticmethod
    def _linux_limit() -> Optional[int]:
        if not sys.platform.startswith("linux"):
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
                lim = _read_int_file(os.path.join(grp, "memory.max"))
                cur = _read_int_file(os.path.join(grp, "memory.current"))
                if lim is not None and cur is not None:
                    return max(0, lim - cur)
            if mem_rel := next(
                (
                    ln.strip().split(":")[2]
                    for ln in open("/proc/self/cgroup")
                    if "memory" in ln.split(":")[1].split(",")
                ),
                None,
            ):
                grp = os.path.join("/sys/fs/cgroup/memory", mem_rel.lstrip("/"))
                lim, use = (
                    _read_int_file(os.path.join(grp, "memory.limit_in_bytes")),
                    _read_int_file(os.path.join(grp, "memory.usage_in_bytes")),
                )
                if lim and use and lim < (1 << 60):
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
            import psutil

            kernel32 = ctypes.windll.kernel32
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
            import psutil

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
    def total() -> Optional[int]:
        try:
            import psutil

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

                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode(
                    "utf-8", "ignore"
                )
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
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    return int(stat.ullTotalPhys)
        except Exception:
            pass
        return None

    @staticmethod
    def available() -> int:
        base = Memory._sys_available_memory()
        cgroup = Memory._linux_limit()
        winjob = Memory._windows_limit()
        rlim = Memory._bsd_limit()
        candidates = [x for x in (base, cgroup, winjob, rlim) if isinstance(x, int) and x >= 0]
        return max(0, min(candidates)) if candidates else 0

    @staticmethod
    def mem_get_info(device: torch.device) -> Tuple[Optional[int], Optional[int]]:
        free: Optional[int] = None
        total: Optional[int] = None
        with contextlib.suppress(Exception):
            acc = getattr(torch, "accelerator", None)
            if acc is not None:
                get_memory_info = getattr(acc, "get_memory_info", None)
                if callable(get_memory_info):
                    info = get_memory_info(device)
                    free, total = Memory._parse_free(info)
                if free is None or total is None:
                    mem_mod = getattr(acc, "memory", None)
                    mem_get_info = (
                        getattr(mem_mod, "mem_get_info", None) if mem_mod is not None else None
                    )
                    if callable(mem_get_info):
                        info = mem_get_info(device)
                        f2, t2 = Memory._parse_free(info)
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
                    mem_get_info = (
                        getattr(mem_mod, "mem_get_info", None) if mem_mod is not None else None
                    )
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
    def prefer_local_numa() -> bool:
        try:
            import numa

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
