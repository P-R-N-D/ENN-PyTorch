# -*- coding: utf-8 -*-
from __future__ import annotations

# =============================================================================
# 1. Standard Library Imports
# =============================================================================
import _thread
import contextlib
import contextvars
import ctypes
import functools
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
import warnings
from dataclasses import dataclass
from datetime import timezone, tzinfo
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Optional, Sequence, TYPE_CHECKING, Tuple, Union

# =============================================================================
# 2. Third-Party Imports
# =============================================================================
import torch
import torch.multiprocessing

# =============================================================================
# 3. Local Imports
# =============================================================================
from ..core.datatypes import (
    env_bool,
    env_first,
    env_first_float,
    env_first_int,
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


# =============================================================================
# Globals & Constants
# =============================================================================
_LOGGER = logging.getLogger(__name__)

_CPU_PROC_CACHE: Optional[int] = None
_DEVICE_STATS_CACHE: dict[Tuple[str, int], "Device"] = {}
_ENN_MP_MAIN_STUB_PATH: Optional[str] = None
_EMPTY_CACHE_LAST_CALL_S_BY_DEVICE: dict[Tuple[str, int], float] = {}
_FP32_PRECISION_CACHE: dict[str, str] = {}
_FP32_API_CHOICE_CACHE_KEY = "__enn_tf32_api_choice__"
_FP8_SUPPORT_CACHE: dict[tuple[str, int, int, int], tuple[bool, str]] = {}
_FP8_SUPPORT_CACHE_LOCK = threading.Lock()

_ENN_ORIG_SET_F32_MATMUL_PREC = getattr(torch, "set_float32_matmul_precision", None)
_ENN_ORIG_GET_F32_MATMUL_PREC = getattr(torch, "get_float32_matmul_precision", None)

_LAZY_LOCK_INIT_LOCK = threading.Lock()
_LAZY_LOCK_NAMES = frozenset({
    "_FP32_PRECISION_LOCK",
    "_EMPTY_CACHE_LOCK",
    "_DEVICE_STATS_LOCK",
    "_CPU_PROC_LOCK",
    "_RUNTIME_CFG_LOCK",
})

_RUNTIME_CFG_OVERRIDE: contextvars.ContextVar[dict[str, object] | None] = contextvars.ContextVar(
    "_enn_runtime_cfg_override",
    default=None,
)

_RUNTIME_CFG = SimpleNamespace(
    deterministic=False,
    allow_tf32=None,
    cudnn_benchmark=None,
    matmul_precision=None,
    sdpa_backends=None,
    te_first=True,
)

_TZ_ALIASES: dict[str, str] = {
    "Z": "UTC", "UTC": "UTC", "GMT": "Etc/GMT", "KST": "Asia/Seoul",
    "JST": "Asia/Tokyo", "HKT": "Asia/Hong_Kong", "SGT": "Asia/Singapore",
    "ICT": "Asia/Bangkok", "WITA": "Asia/Makassar", "WIT": "Asia/Jayapura",
    "PHT": "Asia/Manila", "MYT": "Asia/Kuala_Lumpur", "AEST": "Australia/Sydney",
    "AEDT": "Australia/Sydney", "ACST": "Australia/Adelaide", "ACDT": "Australia/Adelaide",
    "AWST": "Australia/Perth", "NZST": "Pacific/Auckland", "NZDT": "Pacific/Auckland",
    "CET": "Europe/Berlin", "CEST": "Europe/Berlin", "EET": "Europe/Athens",
    "EEST": "Europe/Athens", "WET": "Europe/Lisbon", "WEST": "Europe/Lisbon",
    "BST": "Europe/London", "IST": "Europe/Dublin", "MSK": "Europe/Moscow",
    "TRT": "Europe/Istanbul", "ET": "America/New_York", "CT": "America/Chicago",
    "MT": "America/Denver", "PT": "America/Los_Angeles", "EST": "America/New_York",
    "EDT": "America/New_York", "CST": "America/Chicago", "CDT": "America/Chicago",
    "MST": "America/Denver", "MDT": "America/Denver", "PST": "America/Los_Angeles",
    "PDT": "America/Los_Angeles", "AKST": "America/Anchorage", "AKDT": "America/Anchorage",
    "HST": "Pacific/Honolulu", "AST": "America/Halifax", "ADT": "America/Halifax",
    "NST": "America/St_Johns", "NDT": "America/St_Johns", "BRT": "America/Sao_Paulo",
    "ART": "America/Argentina/Buenos_Aires", "SAST": "Africa/Johannesburg",
    "EAT": "Africa/Nairobi", "CAT": "Africa/Maputo", "WAT": "Africa/Lagos",
}


# =============================================================================
# Internal Environment & Lazy Init Helpers
# =============================================================================
def _env_flag(key: str, default: bool = False) -> bool:
    v = os.environ.get(key, None)
    if v is None:
        return bool(default)
    match str(v).strip().lower():
        case "1" | "true" | "t" | "yes" | "y" | "on":
            return True
        case _:
            return False


@functools.lru_cache(maxsize=8)
def _cuda_fp32_precision_api_choice(*, use_new_api: bool) -> str:
    if not bool(use_new_api):
        return "legacy_only"
    try:
        if hasattr(torch, "backends") and hasattr(torch.backends, "fp32_precision"):
            return "new_api"
    except Exception:
        pass
    try:
        if callable(getattr(torch, "set_float32_matmul_precision", None)):
            return "legacy_setter"
    except Exception:
        pass
    return "legacy_only"


def _cuda_fp32_precision_api_choice_from_env() -> str:
    use_new_api = bool(env_bool("ENN_TF32_USE_NEW_API", default=False))
    return _cuda_fp32_precision_api_choice(use_new_api=use_new_api)


def _clear_fp32_precision_api_cache() -> None:
    with contextlib.suppress(Exception):
        _cuda_fp32_precision_api_choice.cache_clear()  # type: ignore[attr-defined]


def __getattr__(name: str) -> Any:
    if name in _LAZY_LOCK_NAMES:
        g = globals()
        lock = g.get(name, None)
        if lock is not None:
            return lock
        with _LAZY_LOCK_INIT_LOCK:
            lock = g.get(name, None)
            if lock is None:
                lock = threading.Lock()
                g[name] = lock
        return lock
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _mutex_lock(name: str) -> _thread.LockType:
    return getattr(sys.modules[__name__], name)


def _enn_set_fp32_precision_new_api(prec: str) -> None:
    with contextlib.suppress(Exception):
        torch.backends.fp32_precision = prec
        torch.backends.cuda.matmul.fp32_precision = prec
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None and hasattr(cudnn, "fp32_precision"):
        with contextlib.suppress(Exception):
            cudnn.fp32_precision = prec


def _install_matmul_precision_legacy_shim_if_needed() -> None:
    if _FP32_PRECISION_CACHE.get("legacy_matmul_shim_installed", "") == "1":
        return
    current_setter = getattr(torch, "set_float32_matmul_precision", None)
    current_getter = getattr(torch, "get_float32_matmul_precision", None)
    
    if not callable(current_setter) or not callable(_ENN_ORIG_SET_F32_MATMUL_PREC):
        return
    if not (
        hasattr(torch, "backends")
        and hasattr(torch.backends, "fp32_precision")
        and hasattr(torch.backends, "cuda")
        and hasattr(torch.backends.cuda, "matmul")
        and hasattr(torch.backends.cuda.matmul, "fp32_precision")
    ):
        return

    def _shim(precision: str) -> None:
        p = str(precision).strip().lower()
        match p:
            case "medium":
                with contextlib.suppress(Exception):
                    _ENN_ORIG_SET_F32_MATMUL_PREC("medium")
            case "high" | "tf32":
                _enn_set_fp32_precision_new_api("tf32")
            case "highest" | "ieee":
                _enn_set_fp32_precision_new_api("ieee")
            case _:
                with contextlib.suppress(Exception):
                    warnings.warn(
                        f"Invalid matmul precision {precision!r}; leaving current precision unchanged.",
                        UserWarning,
                        stacklevel=2,
                    )

    def _shim_get() -> str:
        with contextlib.suppress(Exception):
            if callable(_ENN_ORIG_GET_F32_MATMUL_PREC):
                legacy_v = str(_ENN_ORIG_GET_F32_MATMUL_PREC() or "").strip().lower()
                if legacy_v in {"highest", "high", "medium"}:
                    return legacy_v
        try:
            v_cuda = None
            with contextlib.suppress(Exception):
                v_cuda = str(torch.backends.cuda.matmul.fp32_precision or "").strip().lower()
            if v_cuda in {"tf32", "ieee"}:
                return "high" if v_cuda == "tf32" else "highest"
            v = str(getattr(torch.backends, "fp32_precision", "ieee") or "ieee").strip().lower()
            return "high" if v == "tf32" else "highest"
        except Exception:
            return "highest"

    try:
        if current_setter is _ENN_ORIG_SET_F32_MATMUL_PREC:
            torch.set_float32_matmul_precision = _shim
        if callable(current_getter) and current_getter is _ENN_ORIG_GET_F32_MATMUL_PREC:
            torch.get_float32_matmul_precision = _shim_get
        _FP32_PRECISION_CACHE["legacy_matmul_shim_installed"] = "1"
    except Exception:
        pass


def _device_from(device: Optional[Union[torch.device, str]]) -> torch.device:
    match device:
        case torch.device():
            return device
        case str():
            return torch.device(device)
        case None:
            return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        case _:
            return torch.device(str(device))


def _log_msg(logger: logging.Logger | Callable[[str], None] | None, msg: str, level: str) -> None:
    try:
        if callable(logger):
            logger(msg)
        elif logger:
            getattr(logger, level)(msg)
        else:
            getattr(_LOGGER, level)(msg)
    except Exception:
        getattr(_LOGGER, level)(msg)


def _log_info(logger: logging.Logger | Callable[[str], None] | None, msg: str) -> None:
    _log_msg(logger, msg, "info")


def _log_debug(logger: logging.Logger | Callable[[str], None] | None, msg: str) -> None:
    _log_msg(logger, msg, "debug")


def _call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    with contextlib.suppress(Exception):
        if callable(fn):
            return fn(*args, **kwargs)
    return None


def _clear_device_index(device: Optional[Union[torch.device, str]] = None) -> Tuple[str, int]:
    if device is None:
        return ("all", -1)
    try:
        dev = device if isinstance(device, torch.device) else torch.device(str(device))
    except (TypeError, ValueError, RuntimeError):
        return ("all", -1)
        
    idx = int(dev.index) if dev.index is not None else -1
    match dev.type:
        case "cuda" if idx < 0:
            try:
                idx = int(torch.cuda.current_device()) if torch.cuda.is_available() else -1
            except (RuntimeError, TypeError, ValueError):
                idx = -1
        case "xpu" if idx < 0:
            try:
                xpu = getattr(torch, "xpu", None)
                cur = getattr(xpu, "current_device", None) if xpu is not None else None
                idx = int(cur()) if callable(cur) else -1
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
    s = _read_text_file(path)
    if s and s != "max" and s.isdigit():
        return int(s)
    return None


def _is_main_importable() -> bool:
    try:
        import __main__
        main_file = getattr(__main__, "__file__", None)
    except Exception:
        main_file = None
        
    if not main_file:
        return False
    main_file = str(main_file)
    if (main_file.startswith("<") and main_file.endswith(">")) or main_file in {"-c", "-m", "-"}:
        return False
        
    abs_path = main_file if os.path.isabs(main_file) else os.path.join(os.getcwd(), main_file)
    return os.path.isfile(abs_path)


@contextlib.contextmanager
def _start_context() -> Any:
    stub = _ENN_MP_MAIN_STUB_PATH
    if not stub or not sys.argv:
        yield
        return
    old = sys.argv[0]
    try:
        sys.argv[0] = str(stub)
        yield
    finally:
        with contextlib.suppress(Exception):
            sys.argv[0] = old


def _validate_main_importability() -> None:
    if _is_main_importable():
        return
    try:
        import __main__
        main_mod = __main__
    except Exception:
        return
        
    with contextlib.suppress(Exception):
        setattr(main_mod, "__spec__", None)
        
    stub_dir = "/tmp" if os.path.isdir("/tmp") else os.getcwd()
    stub_path = os.path.join(stub_dir, "enn_mp_main.py")
    try:
        if not os.path.isfile(stub_path):
            with open(stub_path, "w", encoding="utf-8") as f:
                f.write(
                    "# Auto-generated by enn_torch for multiprocessing spawn/forkserver.\n"
                    "if __name__ == '__main__':\n"
                    "    pass\n"
                )
    except Exception:
        return
        
    with contextlib.suppress(Exception):
        setattr(main_mod, "__file__", stub_path)
        
    global _ENN_MP_MAIN_STUB_PATH
    _ENN_MP_MAIN_STUB_PATH = stub_path


def _get_thread_limit(default: int) -> int:
    v = env_first_int(("ENN_THREAD_CAP_MULTIPLIER", "ENN_THREADS_CAP_MULTIPLIER"), default)
    return max(1, min(8, int(v)))


def _default_thread_limit(ncpu_raw: int, *args: Any, is_accel: bool, nogil: bool) -> int:
    if ncpu_raw > 8:
        cap_mult = 3 if (nogil and is_accel) else (2 if is_accel else 1)
    elif ncpu_raw > 4:
        cap_mult = min(3 if (nogil and is_accel) else 2, 2)
    else:
        cap_mult = 1
    return _get_thread_limit(int(cap_mult))


def _optimal_local_worlds(default: int) -> int:
    return max(
        1,
        env_first_int(
            ("ENN_LOCAL_WORLD_SIZE", "LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE"),
            int(default),
        ),
    )


def _optimal_threads(*args: Any, ncpu: int, cap_mult: int, local_world: int, distribute: bool) -> int:
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
        for cand in ("/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct", "/sys/fs/cgroup/cpuacct"):
            if os.path.isdir(cand):
                base = cand
                break
        if base is None:
            return None
            
        grp = os.path.join(base, cpu_rel.lstrip("/"))
        quota = _read_int_file(os.path.join(grp, "cpu.cfs_quota_us"))
        period = _read_int_file(os.path.join(grp, "cpu.cfs_period_us"))
        if quota is None or period is None or int(quota) <= 0 or int(quota) == -1 or int(period) <= 0:
            return None
        return float(quota) / float(period)
    except Exception:
        return None


def _get_allowed_cpu_darwin() -> Optional[int]:
    if platform.system() != "Darwin":
        return None
    try:
        libc = ctypes.CDLL("libc.dylib")
        sysctlbyname = getattr(libc, "sysctlbyname", None)
        if sysctlbyname is None:
            return None
            
        sysctlbyname.argtypes = [
            ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t
        ]
        sysctlbyname.restype = ctypes.c_int
        
        for name in (b"hw.logicalcpu", b"hw.ncpu"):
            val = ctypes.c_int(0)
            size = ctypes.c_size_t(ctypes.sizeof(val))
            if int(sysctlbyname(name, ctypes.byref(val), ctypes.byref(size), None, 0)) == 0:
                n = int(val.value)
                if n > 0:
                    return int(n)
    except Exception:
        pass
    return None


def _get_allowed_cpu_windows() -> Optional[list[int]]:
    if platform.system() != "Windows":
        return None
    try:
        k32 = ctypes.windll.kernel32
        get_proc = getattr(k32, "GetCurrentProcess", None)
        get_mask = getattr(k32, "GetProcessAffinityMask", None)
        if callable(get_proc) and callable(get_mask):
            h = get_proc()
            proc_mask = ctypes.c_size_t(0)
            sys_mask = ctypes.c_size_t(0)
            if int(get_mask(h, ctypes.byref(proc_mask), ctypes.byref(sys_mask))):
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

        counts: list[int] = [max(0, int(_call(get_group_procs, ctypes.c_ushort(i)) or _call(get_group_procs, i) or 0)) for i in range(group_count)]

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

        total = sum(int(counts[int(g)]) for g in groups if 0 <= int(g) < group_count)
        if total > 0:
            return list(range(total))
    except Exception:
        pass
    return None


def _parse_torch_dtype(value: str) -> Tuple[torch.dtype, ...]:
    entries: list[torch.dtype] = []
    for token in str(value).split(","):
        name = token.strip()
        if not name: continue
        dtype = getattr(torch, name, None)
        if isinstance(dtype, torch.dtype):
            entries.append(dtype)
    return tuple(entries)


def _get_allowed_cpu_fallback() -> list[int]:
    n: Optional[int] = None
    with contextlib.suppress(Exception):
        if isinstance(v := os.cpu_count(), int) and v > 0:
            n = int(v)
    if n is None and platform.system() == "Darwin":
        with contextlib.suppress(Exception):
            if isinstance(v := _get_allowed_cpu_darwin(), int) and v > 0:
                n = int(v)
    return list(range(int(n) if isinstance(n, int) and n > 0 else 1))


def _get_cgroup_quota() -> int:
    quota = _get_allowed_cpu_linux()
    if quota is None or quota <= 0:
        return 0
    with contextlib.suppress(Exception):
        return max(0, int(math.floor(float(quota))))
    return 0


def _acc_mod(dt: str) -> ModuleType | None:
    return getattr(torch, dt, None) if dt in ("cuda", "xpu", "mps") else accelerator_type(dt)


def _acc_op(dt: str, op: str, default: object | None = None) -> object | None:
    return _call(getattr(_acc_mod(dt), op, None)) or default


# =============================================================================
# Core System APIs
# =============================================================================
def empty_device_cache(
    *args: Any,
    device: Optional[Union[torch.device, str]] = None,
    do_gc: bool = True,
    min_interval_s: Optional[float] = None,
) -> None:
    if not env_bool("ENN_EMPTY_CACHE", True):
        return
        
    min_interval = float(min_interval_s if min_interval_s is not None else env_first_float(("ENN_EMPTY_CACHE_MIN_INTERVAL_S",), 0.5))
    min_interval = max(0.0, min_interval)
    
    now = time.monotonic()
    key = _clear_device_index(device)
    
    with _mutex_lock("_EMPTY_CACHE_LOCK"):
        last = float(_EMPTY_CACHE_LAST_CALL_S_BY_DEVICE.get(key, 0.0))
        if min_interval and (now - last) < min_interval:
            return
        _EMPTY_CACHE_LAST_CALL_S_BY_DEVICE[key] = float(now)
        
    if do_gc:
        with contextlib.suppress(Exception):
            gc.collect()
            
    with contextlib.suppress(Exception):
        acc = getattr(torch, "accelerator", None)
        if acc and callable(ec := getattr(getattr(acc, "memory", None), "empty_cache", None)):
            ec()
            
    target = None
    with contextlib.suppress(Exception):
        target = device if isinstance(device, torch.device) else (torch.device(str(device)) if device else None)
        
    dt_s = str(getattr(target, "type", None) or "all")
    
    match dt_s:
        case "all":
            if torch.cuda.is_available():
                if target is not None and target.index is not None:
                    with torch.cuda.device(int(target.index)):
                        _call(getattr(torch.cuda, "empty_cache", None))
                else:
                    _call(getattr(torch.cuda, "empty_cache", None))
            _call(getattr(getattr(torch, "mps", None), "empty_cache", None))
            xpu_mod = getattr(torch, "xpu", None)
            if _call(getattr(xpu_mod, "empty_cache", None)) is None:
                _call(getattr(getattr(xpu_mod, "memory", None), "empty_cache", None))
                
        case "cuda":
            if torch.cuda.is_available():
                if target is not None and target.index is not None:
                    with torch.cuda.device(int(target.index)):
                        _call(getattr(torch.cuda, "empty_cache", None))
                else:
                    _call(getattr(torch.cuda, "empty_cache", None))
                    
        case "mps":
            _call(getattr(getattr(torch, "mps", None), "empty_cache", None))
            
        case "xpu":
            xpu_mod = getattr(torch, "xpu", None)
            if _call(getattr(xpu_mod, "empty_cache", None)) is None:
                _call(getattr(getattr(xpu_mod, "memory", None), "empty_cache", None))


def is_oom_error(exc: BaseException) -> bool:
    with contextlib.suppress(Exception):
        if isinstance(typ := getattr(torch, "OutOfMemoryError", None), type) and isinstance(exc, typ):
            return True
            
    for mod_name in ("cuda", "xpu", "mps"):
        with contextlib.suppress(Exception):
            if isinstance(typ := getattr(getattr(torch, mod_name, None), "OutOfMemoryError", None), type) and isinstance(exc, typ):
                return True
                
    msg = str(exc).lower()
    if not msg:
        return False
        
    patterns = (
        "out of memory", "cuda out of memory", "cuda error: out of memory",
        "hip out of memory", "xpu out of memory", "mps backend out of memory",
        "not enough memory", "failed to allocate memory",
        "cublas_status_alloc_failed", "cudnn_status_alloc_failed",
    )
    return any(p in msg for p in patterns)


def accelerator_type(dev_type: str) -> Optional[ModuleType]:
    dt = str(dev_type or "cpu").strip().lower()
    match dt:
        case "cuda" | "xpu" | "mps":
            return getattr(torch, dt, None)
        case _:
            acc = getattr(torch, "accelerator", None)
            return acc if acc and getattr(acc, "device_type", None) == dt else None


def is_accelerator_available(dev_type: str) -> bool:
    dt = str(dev_type or "cpu").strip().lower()
    match dt:
        case "mps":
            return bool(_call(getattr(torch.backends.mps, "is_available", None)) or _acc_op(dt, "is_available"))
        case _:
            return bool(_acc_op(dt, "is_available"))


def get_num_accelerators(dev_type: str) -> int:
    dt = str(dev_type or "cpu").strip().lower()
    if not is_accelerator_available(dt): return 0
    if dt == "mps": return 1
    return max(0, int(_acc_op(dt, "device_count") or 0))


def get_accelerator_index(dev_type: str) -> int:
    dt = str(dev_type or "cpu").strip().lower()
    if not is_accelerator_available(dt): return -1
    if dt == "mps": return 0
    v = _acc_op(dt, "current_device")
    return int(v) if v is not None else 0


def set_accelerator_index(dev_type: str, idx: int) -> None:
    dt = str(dev_type or "cpu").strip().lower()
    if is_accelerator_available(dt):
        _call(getattr(_acc_mod(dt), "set_device", None), int(idx))


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
            return 100.0 * float(max(0, int(total_b) - int(free_b))) / float(total_b)
    except Exception:
        pass

    try:
        mod = _acc_mod(dev.type)
        match dev.type:
            case "cuda" | "xpu":
                props = _call(mod.get_device_properties, dev.index or 0)
                total = getattr(props, "total_memory", 0)
                used = _call(mod.memory_allocated, dev)
            case "mps":
                total = _call(mod.recommended_max_memory)
                used = _call(mod.current_allocated_memory)
            case _:
                return None
        return (100.0 * used / total) if total and used is not None else None
    except Exception:
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
        
    try:
        mod = _acc_mod(dev.type)
        match dev.type:
            case "cuda" | "xpu":
                props = _call(mod.get_device_properties, dev.index if dev.index is not None else 0)
                return int(props.total_memory) if props else None
            case "mps":
                return int(_call(mod.recommended_max_memory))
    except Exception:
        pass
    return None


def allocated_accelerator_memory(device: Union[torch.device, str]) -> Optional[int]:
    try:
        dev = device if isinstance(device, torch.device) else torch.device(str(device))
        if not is_accelerator_available(dev.type):
            return None
        mod = _acc_mod(dev.type)
        v = _call(mod.current_allocated_memory) if dev.type == "mps" else _call(mod.memory_allocated, dev)
        return max(0, int(v)) if v is not None else None
    except Exception:
        return None


def flush_accelerator_memory_stats(device: Union[torch.device, str]) -> None:
    try:
        dev = device if isinstance(device, torch.device) else torch.device(str(device))
        if not is_accelerator_available(dev.type):
            return
        mod = _acc_mod(dev.type)
        if dev.type == "mps":
            _call(mod.reset_peak_memory_stats)
        else:
            _call(mod.reset_peak_memory_stats, dev)
    except Exception:
        pass


def accelerator_max_allocated_memory(device: Union[torch.device, str]) -> Optional[int]:
    try:
        dev = device if isinstance(device, torch.device) else torch.device(str(device))
        if not is_accelerator_available(dev.type):
            return None
        mod = _acc_mod(dev.type)
        v = _call(mod.max_memory_allocated) if dev.type == "mps" else _call(mod.max_memory_allocated, dev)
        return max(0, int(v)) if v is not None else None
    except Exception:
        return None


def collect_accelerator_ipc() -> None:
    if is_accelerator_available("cuda"):
        _call(getattr(torch.cuda, "ipc_collect", None))


def accelerator(device: torch.device) -> contextlib.AbstractContextManager[None]:
    try:
        dev = device if isinstance(device, torch.device) else torch.device(str(device))
        mod = _acc_mod(dev.type)
        if mod and hasattr(mod, "device"):
            return mod.device(dev if dev.index is None else dev.index)
    except Exception:
        pass
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
    except Exception:
        pass


def is_accelerator_timer_supported(dev_type: str) -> bool:
    dt = str(dev_type or "cpu").strip().lower()
    backend = accelerator_type(dt)
    if backend is None: return False
    
    if callable(avail := getattr(backend, "is_available", None)):
        v = _call(avail)
        if v is not None and not bool(v): return False
        
    Event = getattr(backend, "Event", None)
    if Event is None: return False
    
    ev0 = _call(Event, enable_timing=True) or _call(Event)
    ev1 = _call(Event, enable_timing=True) or _call(Event)
    if ev0 is None or ev1 is None: return False
    
    need = ("record", "synchronize", "elapsed_time")
    for ev in (ev0, ev1):
        if any(not callable(getattr(ev, name, None)) for name in need):
            return False
    return True


def new_accelerator_event(device: Union[torch.device, str], *args: Any, enable_timing: bool = False) -> object | None:
    del args
    try:
        dev = device if isinstance(device, torch.device) else torch.device(str(device))
        dt = str(getattr(dev, "type", "cpu") or "cpu")
        mod = _acc_mod(dt)
        if not mod or not hasattr(mod, "Event"): return None
        return _call(mod.Event, enable_timing=enable_timing) or _call(mod.Event)
    except Exception:
        return None


def is_stream_supported(dev_type: str) -> bool:
    dt = str(dev_type or "cpu").strip().lower()
    if dt not in {"cuda", "xpu"}: return False
    
    backend = accelerator_type(dt)
    if backend is None: return False
    
    if callable(avail := getattr(backend, "is_available", None)):
        v = _call(avail)
        if v is not None and not bool(v): return False
        
    return bool(
        callable(getattr(backend, "Stream", None)) and 
        (getattr(backend, "Event", None) is not None) and 
        callable(getattr(backend, "stream", None)) and 
        callable(getattr(backend, "current_stream", None))
    )


def is_pin_supported(dev_type: str) -> bool:
    dt = str(dev_type or "cpu").strip().lower()
    match dt:
        case "cuda" | "xpu":
            backend = accelerator_type(dt)
            if backend is None: return False
            if callable(avail := getattr(backend, "is_available", None)):
                v = _call(avail)
                if v is not None and not bool(v): return False
            return bool(callable(getattr(backend, "current_stream", None)) and getattr(backend, "Event", None) is not None)
        case _:
            return False


def accelerator_stream(stream: object, dev_type: str) -> contextlib.AbstractContextManager[None]:
    mod = _acc_mod(dev_type)
    return mod.stream(stream) if mod and hasattr(mod, "stream") else contextlib.nullcontext()


def new_accelerator_stream(device: torch.device) -> object | None:
    try:
        dev = device if isinstance(device, torch.device) else torch.device(str(device))
        dt = str(getattr(dev, "type", "cpu") or "cpu")
        mod = _acc_mod(dt)
        return mod.Stream(device=dev) if mod and hasattr(mod, "Stream") else None
    except Exception:
        return None


def current_accelerator_stream(device: torch.device) -> object | None:
    try:
        dev = device if isinstance(device, torch.device) else torch.device(str(device))
        mod = _acc_mod(dev.type)
        return mod.current_stream(device=dev) if mod and hasattr(mod, "current_stream") else None
    except Exception:
        return None


def set_float32_precision(
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    autocast_dtype: Optional[torch.dtype] = None,
    enable_tf32: bool = True,
) -> None:
    if device.type != "cuda":
        return

    use_tf32 = bool(enable_tf32)
    if torch.float64 in (dtype, autocast_dtype):
        use_tf32 = False

    api_choice = _cuda_fp32_precision_api_choice_from_env()
    dev = device if isinstance(device, torch.device) else torch.device(device)
    dev_key = f"{dev.type}:{dev.index if dev.index is not None else -1}"
    cache_key = f"fp32_prec:{dev_key}:{dtype or ''}:{autocast_dtype or ''}:tf32={'1' if use_tf32 else '0'}:api={api_choice}"

    with _mutex_lock("_FP32_PRECISION_LOCK"):
        if _FP32_PRECISION_CACHE.get(_FP32_API_CHOICE_CACHE_KEY) != api_choice:
            keep = {k: v for k, v in _FP32_PRECISION_CACHE.items() if k in ("legacy_matmul_shim_installed", "legacy_matmul_shim_installed_get") and v}
            _FP32_PRECISION_CACHE.clear()
            _FP32_PRECISION_CACHE.update(keep)
            _FP32_PRECISION_CACHE[_FP32_API_CHOICE_CACHE_KEY] = str(api_choice)

        if _FP32_PRECISION_CACHE.get(cache_key) == "1":
            return

    did_apply = False
    try:
        match api_choice:
            case "legacy_only":
                with contextlib.suppress(Exception): torch.backends.cuda.matmul.allow_tf32 = bool(use_tf32)
                with contextlib.suppress(Exception): torch.backends.cudnn.allow_tf32 = bool(use_tf32)
                did_apply = True
            case "new_api":
                prec = "tf32" if use_tf32 else "ieee"
                _enn_set_fp32_precision_new_api(prec)
                _install_matmul_precision_legacy_shim_if_needed()
                did_apply = True
            case _:
                precision = "high" if use_tf32 else "highest"
                used_set_prec = False
                if api_choice == "legacy_setter" and callable(set_prec := getattr(torch, "set_float32_matmul_precision", None)):
                    with contextlib.suppress(Exception):
                        set_prec(str(precision))
                        used_set_prec = True
                else:
                    with contextlib.suppress(Exception): torch.backends.cuda.matmul.allow_tf32 = bool(use_tf32)
                
                if used_set_prec and (cudnn := getattr(torch.backends, "cudnn", None)) and hasattr(cudnn, "fp32_precision"):
                    with contextlib.suppress(Exception): cudnn.fp32_precision = ("tf32" if use_tf32 else "ieee")
                else:
                    with contextlib.suppress(Exception): torch.backends.cudnn.allow_tf32 = bool(use_tf32)
                did_apply = True
    finally:
        if did_apply:
            with _mutex_lock("_FP32_PRECISION_LOCK"):
                _FP32_PRECISION_CACHE[cache_key] = "1"


def timezone_from(name: Optional[str] = None) -> Optional[tzinfo]:
    resolved = (name or "GMT").strip()
    alias = _TZ_ALIASES.get(resolved.upper(), resolved)
    if alias.upper() == "UTC":
        return timezone.utc
    if ZoneInfo is not None:
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
    
    match sysname:
        case "Linux":
            os_name = sysname
            if hasattr(platform, "freedesktop_os_release"):
                info = _call(getattr(platform, "freedesktop_os_release", None))
                if isinstance(info, dict):
                    name = info.get("NAME") or "Linux"
                    os_name = (name if not (version := info.get("VERSION_ID")) else f"{name} {version}").strip()
        case "Windows":
            os_name = sysname
            if isinstance(win := _call(getattr(platform, "win32_ver", None)), (list, tuple)) and win:
                os_name = f"Windows {win[0] or ''}".strip()
        case "Darwin":
            os_name = sysname
            if isinstance(mac := _call(getattr(platform, "mac_ver", None)), (list, tuple)) and mac:
                os_name = f"macOS {mac[0] or ''}".strip()
        case _:
            os_name = sysname

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
    base = _RUNTIME_CFG
    if not (ov := _RUNTIME_CFG_OVERRIDE.get()):
        return base
    merged = SimpleNamespace(**getattr(base, "__dict__", {}))
    for k, v in ov.items():
        setattr(merged, str(k), v)
    return merged


def get_runtime_cfg() -> SimpleNamespace:
    return get_runtime_config()


@contextlib.contextmanager
def runtime_cfg_override(**kwargs: object) -> Any:
    if not kwargs:
        yield get_runtime_cfg()
        return
    prev = _RUNTIME_CFG_OVERRIDE.get()
    merged: dict[str, object] = dict(prev) if isinstance(prev, dict) else {}
    merged.update({str(k): v for k, v in kwargs.items()})
    token = _RUNTIME_CFG_OVERRIDE.set(merged)
    try:
        yield get_runtime_cfg()
    finally:
        _RUNTIME_CFG_OVERRIDE.reset(token)


def set_runtime_cfg(key: str | dict[str, object] | None = None, value: object | None = None, /, **kwargs: object) -> SimpleNamespace:
    cfg = _RUNTIME_CFG
    match key:
        case str():
            setattr(cfg, key, value)
        case dict():
            for k, v in key.items():
                setattr(cfg, str(k), v)
        case None:
            pass
        case _:
            raise TypeError(f"set_runtime_cfg() expects a str|dict|None for first arg, got {type(key)!r}")

    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def init_python_path() -> str:
    separator = os.pathsep
    env_paths = [path for path in os.environ.get("PYTHONPATH", "").split(separator) if path]
    paths: list[str] = list(env_paths)
    seen: set[str] = set(env_paths)

    def _prioritized_path(candidate: Path | str | None) -> None:
        if candidate is None: return
        try: path_str = os.fspath(candidate)
        except TypeError: return
        if not path_str: return
        
        if path_str in seen:
            if path_str not in sys.path: sys.path.insert(0, path_str)
            return
        seen.add(path_str); paths.insert(0, path_str)
        if path_str not in sys.path: sys.path.insert(0, path_str)

    try: package_dir = Path(__file__).resolve().parents[1]
    except Exception: package_dir = None
    project_dir = package_dir.parent if package_dir is not None else None
    
    cwd_dir: Path | None = None
    with contextlib.suppress(Exception): cwd_dir = Path.cwd().resolve()
        
    main_dir: Path | None = None
    if (main_module := sys.modules.get("__main__")) and (main_file := getattr(main_module, "__file__", None)):
        with contextlib.suppress(Exception): main_dir = Path(main_file).resolve().parent
            
    for candidate in (package_dir, project_dir, main_dir, cwd_dir):
        _prioritized_path(candidate)
        
    for entry in list(sys.path):
        if not entry: continue
        try: entry_str = os.fspath(entry)
        except TypeError: continue
        if entry_str and entry_str not in seen:
            seen.add(entry_str); paths.append(entry_str)
            
    python_path = separator.join(paths)
    os.environ["PYTHONPATH"] = python_path
    return python_path


def optimal_start_method() -> str:
    match sys.platform:
        case p if p.startswith("win"):
            candidates = ("spawn",)
        case _:
            match os.name:
                case "posix":
                    _validate_main_importability()
                    candidates = ("forkserver", "spawn")
                case _:
                    candidates = ("spawn",)
                    
    for method in candidates:
        try: multiprocessing.get_context(method)
        except ValueError: continue
        return method
    raise RuntimeError("No supported multiprocessing start method (tried forkserver, spawn).")


def init_start_method() -> None:
    with contextlib.suppress(RuntimeError):
        torch.multiprocessing.set_sharing_strategy("file_system")
        
    existing = torch.multiprocessing.get_start_method(allow_none=True)
    if os.name == "posix":
        _validate_main_importability()
        
    match sys.platform:
        case p if p.startswith("win"):
            if existing == "spawn": return
            candidates = ("spawn",)
        case _:
            match os.name:
                case "posix":
                    if existing == "forkserver": return
                    candidates = ("forkserver", "spawn")
                case _:
                    if existing == "spawn": return
                    candidates = ("spawn",)
                    
    last_error: Optional[BaseException] = None
    for method in candidates:
        try:
            multiprocessing.get_context(method)
            for module in (multiprocessing, torch.multiprocessing):
                module.set_start_method(method, force=True)
            return
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            continue
            
    raise RuntimeError("Unable to configure multiprocessing start method (tried forkserver, spawn).") from last_error


def _linux_is_tmpfs(mountpoint: str) -> bool:
    if not sys.platform.startswith("linux"): return False
    try: mp = os.path.realpath(str(mountpoint))
    except Exception: mp = str(mountpoint)
    try:
        best_mnt, best_fs = "", ""
        with open("/proc/mounts", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3 or not (mnt := parts[1]): continue
                fstype = parts[2].lower()
                if mp == mnt or mp.startswith(mnt.rstrip("/") + "/"):
                    if len(mnt) > len(best_mnt):
                        best_mnt, best_fs = mnt, fstype
        return best_fs in {"tmpfs", "ramfs"}
    except Exception:
        return False


def default_temp(*, large: bool = False) -> str:
    if (override := os.environ.get("ENN_TEMP_DIR") or os.environ.get("ENN_TMPDIR")) and os.path.isdir(override) and os.access(override, os.W_OK):
        return override
    if sys.platform.startswith("win"):
        return os.environ.get("TEMP", r"C:\Windows\Temp")
        
    tmp = "/tmp" if os.path.isdir("/tmp") and os.access("/tmp", os.W_OK) else None
    vtmp = "/var/tmp" if os.path.isdir("/var/tmp") and os.access("/var/tmp", os.W_OK) else None
    
    if large and vtmp is not None:
        if sys.platform.startswith("linux") and tmp is not None and not _linux_is_tmpfs(tmp):
            pass # Keep logic fallback
        else:
            return vtmp
            
    if tmp is not None: return tmp
    if vtmp is not None: return vtmp
    try:
        if (cwd := os.getcwd()) and os.path.isdir(cwd) and os.access(cwd, os.W_OK):
            return cwd
    except Exception: pass
    return "/tmp"


def new_dir(prefix: str, *, large: bool = False) -> str:
    base = default_temp(large=large)
    os.makedirs(base, exist_ok=True)
    directory = os.path.join(base, f"{prefix}_{os.getpid()}_{os.urandom(4).hex()}")
    os.makedirs(directory, exist_ok=True)
    return directory


def get_sdpa_backends() -> list[object]:
    if not (names := _RUNTIME_CFG.sdpa_backends): return []
    try: from torch.nn.attention import SDPBackend
    except Exception: return []
    
    mapping = {
        "FLASH": "FLASH_ATTENTION", "FLASH_ATTENTION": "FLASH_ATTENTION",
        "EFFICIENT": "EFFICIENT_ATTENTION", "MEM_EFFICIENT": "EFFICIENT_ATTENTION",
        "CUDNN": "CUDNN_ATTENTION", "MATH": "MATH",
    }
    return [getattr(SDPBackend, mapping.get(str(name), str(name))) for name in names if hasattr(SDPBackend, mapping.get(str(name), str(name)))]


def get_dpa_backends() -> list[object]:
    return get_sdpa_backends()


def cuda_compute_capability(device: Union[torch.device, str]) -> Tuple[int, int]:
    dev = _device_from(device)
    if dev.type != "cuda" or not torch.cuda.is_available(): return (0, 0)
    try:
        major, minor = torch.cuda.get_device_capability(dev)
        return (int(major), int(minor))
    except Exception:
        return (0, 0)


def is_cpu_bf16_supported() -> bool:
    try:
        mkldnn = getattr(torch.backends, "mkldnn", None)
        if not mkldnn or not bool(mkldnn.is_available()) or not bool(getattr(mkldnn, "enabled", True)):
            return False
        if callable(f := getattr(getattr(torch.ops, "mkldnn", None), "_is_mkldnn_bf16_supported", None)):
            return bool(f())
    except Exception: pass
    return False


def is_cuda_bf16_supported(device: Optional[Union[torch.device, str]] = None) -> bool:
    try:
        if not torch.cuda.is_available(): return False
        dev = _device_from(device)
        if dev.type != "cuda": return False
        with torch.cuda.device(dev):
            if callable(f := getattr(torch.cuda, "is_bf16_supported", None)):
                try: return bool(f(including_emulation=False))
                except TypeError: return bool(f())
            major, _ = torch.cuda.get_device_capability(dev)
            return int(major) >= 8
    except Exception:
        return False


def is_float8_supported(device: Optional[Union[torch.device, str]] = None) -> Tuple[bool, str]:
    try:
        dev = _device_from(device)
        if not (dev.type == "cuda" and torch.cuda.is_available()):
            return (False, "FP8 requires CUDA")
        major, minor = cuda_compute_capability(dev)
        if (int(major), int(minor)) < (8, 9):
            return (False, "FP8 requires sm89+ (Ada) or sm90+ (Hopper)")
            
        cache = globals().setdefault("_FP8_SUPPORT_CACHE", {})
        lock = globals().setdefault("_FP8_SUPPORT_CACHE_LOCK", threading.Lock())
        
        idx = getattr(dev, "index", None)
        if idx is None:
            with contextlib.suppress(Exception): idx = int(torch.cuda.current_device())
        key = ("cuda", int(idx) if idx is not None else -1, int(major), int(minor))
        
        with lock:
            if (cached := cache.get(key)) is not None: return cached
            
        do_selftest = bool(env_bool("ENN_FP8_SELFTEST", default=True))
        do_quant = bool(env_bool("ENN_FP8_SELFTEST_QUANTIZE", default=True))

        def _has_torchao() -> bool:
            with contextlib.suppress(Exception):
                import torchao.float8 # type: ignore
                return True
            return False

        def _has_te() -> bool:
            with contextlib.suppress(Exception):
                import transformer_engine.pytorch as te # type: ignore
                return callable(getattr(te, "autocast", None))
            return False

        def _selftest_ao() -> Tuple[bool, str]:
            if not _has_torchao(): return (False, "torchao-missing")
            if not do_selftest: return (True, "ao-ok(no-selftest)")
            try: import torchao.float8 as ao_f8
            except Exception as exc: return (False, f"torchao-import-failed:{str(exc)[:120]}")
            
            dts = [getattr(torch, nm) for nm in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, nm)]
            if not dts: return (False, "torch-no-float8-dtypes")
            
            last = ""
            try:
                with ao_f8.fp8_autocast(enabled=True):
                    for dt in dts:
                        try:
                            x = torch.randn((16, 16), device=dev, dtype=torch.float16).to(dt)
                            y = torch.randn((16, 16), device=dev, dtype=torch.float16).to(dt)
                            _ = x @ y
                            break
                        except Exception as exc:
                            last = str(exc)
                            continue
                    else: return (False, f"ao-matmul-failed:{last[:120]}")
            except Exception as exc: return (False, f"ao-autocast-failed:{str(exc)[:120]}")
            
            if do_quant:
                try:
                    from torchao.quantization import quantize_, Float8WeightOnlyConfig
                    try:
                        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
                        cfgs = [Float8DynamicActivationFloat8WeightConfig(), Float8WeightOnlyConfig()]
                    except Exception:
                        cfgs = [Float8WeightOnlyConfig()]
                except Exception as exc: return (False, f"ao-quant-import-failed:{str(exc)[:120]}")
                
                ok_any, last2 = False, ""
                for cfg in cfgs:
                    try:
                        with ao_f8.fp8_autocast(enabled=True):
                            lin = torch.nn.Linear(64, 64, bias=False, device=dev, dtype=torch.float16)
                            x = torch.randn((8, 64), device=dev, dtype=torch.float16)
                            quantize_(lin, cfg)
                            if not any((getattr(m.__class__, "__module__", "").startswith("torchao") or "Float8" in m.__class__.__name__) for m in lin.modules()):
                                raise RuntimeError("ao-quantize-produced-no-torchao-modules")
                            if not torch.is_tensor(lin(x)): raise RuntimeError("ao-quant-forward-non-tensor")
                        ok_any = True; break
                    except Exception as exc:
                        last2 = str(exc); continue
                if not ok_any: return (False, f"ao-quant-selftest-failed:{last2[:120]}")
            return (True, "ao-selftest-ok")

        def _selftest_te() -> Tuple[bool, str]:
            if not _has_te(): return (False, "te-missing")
            if not do_selftest: return (True, "te-ok(no-selftest)")
            try:
                import transformer_engine.pytorch as te
                def _run_te_fp8_module(recipe_obj=None) -> Tuple[bool, str]:
                    if not callable(linear_ctor := getattr(te, "Linear", None)): return (False, "te-linear-missing")
                    try: mod = linear_ctor(64, 64, bias=False, params_dtype=torch.float16, device=dev)
                    except TypeError: mod = linear_ctor(64, 64, bias=False).to(device=dev, dtype=torch.float16)
                    x = torch.randn((8, 64), device=dev, dtype=torch.float16)
                    try:
                        if recipe_obj is None:
                            with te.autocast(enabled=True): y = mod(x)
                            return ((torch.is_tensor(y) and y.shape == (8, 64)), "te-selftest-ok:te.Linear+autocast")
                        with te.autocast(enabled=True, recipe=recipe_obj): y = mod(x)
                        return ((torch.is_tensor(y) and y.shape == (8, 64)), "te-selftest-ok:te.Linear+autocast+recipe")
                    except Exception as exc: return (False, f"te-linear-forward-failed:{str(exc)[:120]}")

                recipe = None
                with contextlib.suppress(Exception):
                    from transformer_engine.common.recipe import DelayedScaling, Format
                    recipe = DelayedScaling(fp8_format=Format.HYBRID)
                if recipe is not None:
                    if (ok_recipe := _run_te_fp8_module(recipe))[0]: return ok_recipe
                if (ok_plain := _run_te_fp8_module())[0]: return ok_plain
                return (False, ok_plain[1] if recipe is None else f"{ok_recipe[1]};{ok_plain[1]}")
            except Exception as exc: return (False, f"te-selftest-failed:{str(exc)[:120]}")

        ok_ao, why_ao = _selftest_ao()
        if ok_ao:
            ok_te, why_te = _selftest_te()
            out = (True, f"fp8-ok:ao+te:{why_ao};{why_te}" if ok_te else f"fp8-ok:ao-only:{why_ao};te={why_te}")
        else:
            ok_te, why_te = _selftest_te()
            out = (True, f"fp8-ok:te-only:{why_te}") if ok_te else (False, f"fp8-unavailable:ao={why_ao},te={why_te}")
            
        with lock: cache[key] = out
        return out
    except Exception: return (False, "Unknown float8 support")


def is_int8_supported(device: Optional[Union[torch.device, str]] = None) -> Tuple[bool, str]:
    try:
        dev = _device_from(device)
        if dev.type == "cuda" and torch.cuda.is_available(): return (True, "Int8 supported on CUDA")
        return (True, "Int8 supported on CPU")
    except Exception: return (False, "Unknown int8 support")


def is_int4_supported(device: Optional[Union[torch.device, str]] = None) -> Tuple[bool, str]:
    try:
        dev = _device_from(device)
        if dev.type == "cuda" and torch.cuda.is_available(): return (True, "Int4 supported on CUDA")
        return (True, "Int4 supported on CPU")
    except Exception: return (False, "Unknown int4 support")


def get_device_stats(device: Optional[Union[torch.device, str]] = None) -> Device:
    dev = _device_from(device)
    key = (str(dev.type), int(dev.index) if dev.index is not None else -1)
    with _mutex_lock("_DEVICE_STATS_LOCK"):
        if (cached := _DEVICE_STATS_CACHE.get(key)) is not None: return cached
        
    device_type = str(dev.type)
    cc = None
    if device_type == "cuda" and torch.cuda.is_available():
        major, minor = cuda_compute_capability(dev)
        if major > 0 or minor > 0: cc = (int(major), int(minor))
        
    float_dtypes = _parse_torch_dtype(env_first(("ENN_DATA_FLOAT_DTYPES", "ENN_FLOAT_DTYPES"))) if env_first(("ENN_DATA_FLOAT_DTYPES", "ENN_FLOAT_DTYPES")) else tuple()
    int_dtypes = _parse_torch_dtype(env_first(("ENN_DATA_INT_DTYPES", "ENN_INT_DTYPES"))) if env_first(("ENN_DATA_INT_DTYPES", "ENN_INT_DTYPES")) else tuple()
    
    if not float_dtypes:
        floats: list[torch.dtype] = [torch.float32]
        match device_type:
            case "cuda" if torch.cuda.is_available():
                floats.insert(0, torch.float16)
                if is_cuda_bf16_supported(dev): floats.insert(0, torch.bfloat16)
            case "cpu" if is_cpu_bf16_supported():
                floats.insert(0, torch.bfloat16)
        float_dtypes = tuple(dict.fromkeys(floats))
        
    if not int_dtypes:
        int_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64)
        
    bits = env_first_int(("ENN_DATA_INT_QUANT_BITS", "ENN_INT_QUANT_BITS"), default=0)
    stats = Device(
        device=dev, device_type=device_type, cuda_cc=cc,
        float_dtypes=tuple(float_dtypes), int_dtypes=tuple(int_dtypes), float8_dtypes=tuple(),
        int_quant_bits=int(bits) if int(bits) > 0 else 8,
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
        if deterministic is not None: cfg.deterministic = bool(deterministic)
        det_flag = bool(cfg.deterministic)
        
        cfg.allow_tf32 = bool(allow_tf32 if allow_tf32 is not None else (cfg.allow_tf32 if cfg.allow_tf32 is not None and deterministic is None else not det_flag))
        cfg.cudnn_benchmark = bool(cudnn_benchmark if cudnn_benchmark is not None else (cfg.cudnn_benchmark if cfg.cudnn_benchmark is not None and deterministic is None else not det_flag))
        cfg.matmul_precision = str(matmul_precision if matmul_precision is not None else (cfg.matmul_precision if cfg.matmul_precision is not None and deterministic is None else ("highest" if det_flag else "high")))
        if sdpa_backends is not None: cfg.sdpa_backends = [str(x) for x in sdpa_backends]
        if te_first is not None: cfg.te_first = bool(te_first)
        
        det, allow_tf32_val, cudnn_bench_val = bool(cfg.deterministic), bool(cfg.allow_tf32), bool(cfg.cudnn_benchmark)

    if is_accelerator_available("cuda"):
        idx = 0
        with contextlib.suppress(Exception):
            idx = int(env_first_int(("LOCAL_RANK", "ENN_ACCELERATOR_INDEX", "ENN_DEVICE_INDEX"), default=0)) % max(1, int(get_num_accelerators("cuda") or 1))
        with _mutex_lock("_CPU_PROC_LOCK"): set_accelerator_index("cuda", int(idx))
        device = torch.device(f"cuda:{idx}")
        if (cudnn := getattr(torch.backends, "cudnn", None)):
            _call(setattr, cudnn, "deterministic", bool(det))
            _call(setattr, cudnn, "benchmark", bool(cudnn_bench_val))
        set_float32_precision(device=device, enable_tf32=bool(allow_tf32_val))
    elif is_accelerator_available("xpu"):
        idx = 0
        with contextlib.suppress(Exception):
            idx = int(env_first_int(("LOCAL_RANK",), default=0)) % max(1, int(get_num_accelerators("xpu") or 1))
        set_accelerator_index("xpu", int(idx))
        device = torch.device(f"xpu:{idx}")
    elif is_accelerator_available("mps"):
        device = torch.device("mps")
    elif hasattr(torch, "is_vulkan_available") and torch.is_vulkan_available():
        device = torch.device("vulkan")
    else:
        device = torch.device("cpu")
    return device


def get_module_device(module: torch.nn.Module) -> torch.device:
    with contextlib.suppress(Exception):
        for p in module.parameters(recurse=True):
            if isinstance(p, torch.Tensor): return p.device
    with contextlib.suppress(Exception):
        for b in module.buffers(recurse=True):
            if isinstance(b, torch.Tensor): return b.device
    return torch.device("cpu")


def optimal_optimizer_params(device: torch.device, use_foreach: Optional[bool], use_fused: bool) -> dict[str, bool]:
    flags: dict[str, bool] = {"foreach": (device.type in {"cuda", "xpu"} if use_foreach is None else bool(use_foreach))}
    if use_fused and device.type in {"cuda", "xpu"}:
        flags["fused"] = True
        flags["foreach"] = False
    return flags


@dataclass
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
            match os.name:
                case "nt": cpus = _get_allowed_cpu_windows()
                case _:
                    match sys.platform:
                        case p if p.startswith("linux"): cpus = os.sched_getaffinity(0)
                        case "darwin": cpus = _get_allowed_cpu_darwin()
                        case _: cpus = _get_allowed_cpu_fallback()
        except Exception:
            cpus = _get_allowed_cpu_fallback()

        if isinstance(cpus, set): cpus = list(cpus)
        if isinstance(cpus, (list, tuple)): return [int(v) for v in cpus] or _get_allowed_cpu_fallback()
        if isinstance(cpus, int) and cpus > 0: return list(range(int(cpus)))

        if (spec := importlib.util.find_spec("psutil")) is not None:
            if callable(fn := getattr(importlib.import_module("psutil").Process(), "cpu_affinity", None)):
                if cpus := fn(): return sorted({int(c) for c in cpus})
        return _get_allowed_cpu_fallback()

    @staticmethod
    def count() -> int:
        global _CPU_PROC_CACHE
        if _CPU_PROC_CACHE is not None: return _CPU_PROC_CACHE

        with _mutex_lock("_CPU_PROC_LOCK"):
            if _CPU_PROC_CACHE is not None: return _CPU_PROC_CACHE

            base = max(env_first_int(("ENN_CPU_LIMIT", "ENN_CPU_COUNT", "OMP_NUM_THREADS"), default=0), 0)
            with contextlib.suppress(Exception):
                if isinstance(xopt := getattr(sys, "_xoptions", {}), dict):
                    base = max(base, int(xopt.get("enn_torch.cpu_limit", 0)))
                    
            if base <= 0:
                with contextlib.suppress(Exception):
                    if callable(fn := getattr(os, "process_cpu_count", None)):
                        base = int(fn() or 0)
                if base <= 0: base = int(os.cpu_count() or 1)

            base = max(1, int(base))
            if allowed := CPU.allowed(): base = min(base, len(allowed))
            if (quota := _get_cgroup_quota()) > 0: base = min(base, quota)

            _CPU_PROC_CACHE = int(base)
            return _CPU_PROC_CACHE

    @staticmethod
    def info(max_bytes: Optional[int] = None) -> str:
        info: dict[str, Any] = {
            "cpu_count": int(CPU.count()), "os": sys.platform,
            "machine": platform.machine(), "processor": platform.processor(), "python": sys.version,
        }
        if importlib.util.find_spec("cpuinfo") is not None:
            with contextlib.suppress(Exception):
                if isinstance(raw := importlib.import_module("cpuinfo").get_CPU.info(), dict):
                    info["cpuinfo"] = {k: v for k, v in raw.items() if not k.startswith("_")}
                    
        out = json.dumps(info, sort_keys=True, ensure_ascii=True, default=str)
        if max_bytes is not None and max_bytes > 0:
            b = out.encode("utf-8")
            if len(b) > max_bytes: out = b[: max_bytes - 3].decode("utf-8", errors="ignore") + "..."
        return out

    @staticmethod
    def is_free_threaded_build() -> bool:
        if isinstance(tag := getattr(getattr(sys, "implementation", None), "cache_tag", "") or "", str) and tag.endswith("t"): return True
        with contextlib.suppress(Exception): return bool(int(sysconfig.get_config_var("Py_GIL_DISABLED") or 0))
        return False

    @staticmethod
    def is_gil_enabled() -> bool:
        if hasattr(sys, "_is_gil_enabled"):
            with contextlib.suppress(Exception): return bool(sys._is_gil_enabled())
        return not CPU.is_free_threaded_build()

    @staticmethod
    def is_no_gil_enforced() -> bool:
        return CPU.is_free_threaded_build() and (not CPU.is_gil_enabled())

    @staticmethod
    def is_optimized_for_no_gil() -> bool:
        for key in ("ENN_NOGIL_OPTIMIZED", "ENN_NO_GIL_OPTIMIZED", "ENN_FREE_THREADED_OPTIMIZED"):
            if (override := parse_bool(os.environ.get(key))) is not None: return bool(override)
        return CPU.is_no_gil_enforced()


class Memory:
    @staticmethod
    def _parse_free(info: Any) -> Tuple[Optional[int], Optional[int]]:
        free, total = None, None
        match info:
            case (list() | tuple()) if len(info) >= 1:
                with contextlib.suppress(Exception): free = int(info[0])
                if len(info) >= 2:
                    with contextlib.suppress(Exception): total = int(info[1])
            case dict():
                free_v = info.get("free") or info.get("free_memory") or info.get("free_bytes")
                total_v = info.get("total") or info.get("total_memory") or info.get("total_bytes") or info.get("bytes_limit", None)
                if free_v is None and total_v is not None:
                    if (used_v := info.get("bytes_used") or info.get("bytes_in_use") or info.get("bytes_used_current")) is not None:
                        with contextlib.suppress(Exception): free_v = int(total_v) - int(used_v)
                with contextlib.suppress(Exception):
                    if free_v is not None: free = int(free_v)
                    if total_v is not None: total = int(total_v)
                    
        return max(0, int(free)) if free is not None else None, max(0, int(total)) if total is not None else None

    @staticmethod
    def _sys_available_memory() -> Optional[int]:
        try:
            import psutil
            vm = psutil.virtual_memory()
            if getattr(vm, "available", None) is not None: return int(vm.available)
            if getattr(vm, "total", 0) and getattr(vm, "used", None) is not None: return int(vm.total - vm.used)
        except Exception: pass
        
        try:
            match sys.platform:
                case p if p.startswith("linux"):
                    with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as fh:
                        for line in fh:
                            if line.startswith("MemAvailable:") and len(parts := line.split()) >= 2 and parts[1].isdigit():
                                return int(parts[1]) * 1024
                case "darwin":
                    import subprocess
                    out = subprocess.check_output(["vm_stat"]).decode("utf-8", "ignore")
                    page, free, inactive, speculative = None, None, None, 0
                    for ln in out.splitlines():
                        if "page size of" in ln: page = int(ln.split()[-2])
                        elif ln.startswith("Pages free:"): free = int(ln.split(":")[1].split()[0])
                        elif ln.startswith("Pages inactive:"): inactive = int(ln.split(":")[1].split()[0])
                        elif ln.startswith("Pages speculative:"): speculative = int(ln.split(":")[1].split()[0])
                    if page and free is not None and inactive is not None: return int((free + inactive + speculative) * page)
                case p if p.startswith("win") or os.name == "nt":
                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
                    stat = MEMORYSTATUSEX()
                    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)): return int(stat.ullAvailPhys)
        except Exception: pass
        return None

    @staticmethod
    def _linux_limit() -> Optional[int]:
        if not sys.platform.startswith("linux"): return None
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
                if (lim := _read_int_file(os.path.join(grp, "memory.max"))) is not None and (cur := _read_int_file(os.path.join(grp, "memory.current"))) is not None:
                    return max(0, lim - cur)
                    
            if mem_rel := next((ln.strip().split(":")[2] for ln in open("/proc/self/cgroup") if "memory" in ln.split(":")[1].split(",")), None):
                grp = os.path.join("/sys/fs/cgroup/memory", mem_rel.lstrip("/"))
                if (lim := _read_int_file(os.path.join(grp, "memory.limit_in_bytes"))) and (use := _read_int_file(os.path.join(grp, "memory.usage_in_bytes"))) and lim < (1 << 60):
                    return max(0, lim - use)
        except Exception: pass
        return None

    @staticmethod
    def _windows_limit() -> Optional[int]:
        if not (os.name == "nt" or sys.platform.startswith("win")): return None
        try:
            from ctypes import wintypes as wt
            import psutil
            kernel32 = ctypes.windll.kernel32
            IsProcessInJob = kernel32.IsProcessInJob
            IsProcessInJob.argtypes = [wt.HANDLE, wt.HANDLE, ctypes.POINTER(wt.BOOL)]
            IsProcessInJob.restype = wt.BOOL
            hProc, inJob = kernel32.GetCurrentProcess(), wt.BOOL()
            if not IsProcessInJob(hProc, None, ctypes.byref(inJob)) or not inJob.value: return None

            class LARGE_INTEGER(ctypes.Union): _fields_ = [("QuadPart", ctypes.c_longlong)]
            class IO_COUNTERS(ctypes.Structure): _fields_ = [("ReadOperationCount", ctypes.c_ulonglong), ("WriteOperationCount", ctypes.c_ulonglong), ("OtherOperationCount", ctypes.c_ulonglong), ("ReadTransferCount", ctypes.c_ulonglong), ("WriteTransferCount", ctypes.c_ulonglong), ("OtherTransferCount", ctypes.c_ulonglong)]
            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure): _fields_ = [("PerProcessUserTimeLimit", LARGE_INTEGER), ("PerJobUserTimeLimit", LARGE_INTEGER), ("LimitFlags", ctypes.c_uint), ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint), ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint), ("SchedulingClass", ctypes.c_uint)]
            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure): _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION), ("IoInfo", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

            QueryInformationJobObject = kernel32.QueryInformationJobObject
            QueryInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong)]
            QueryInformationJobObject.restype = wt.BOOL
            info, retlen = JOBOBJECT_EXTENDED_LIMIT_INFORMATION(), ctypes.c_ulong(0)
            if not QueryInformationJobObject(None, 9, ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(retlen)): return None
            
            flags = int(getattr(info.BasicLimitInformation, "LimitFlags", 0))
            cand_limits: list[int] = []
            if flags & 0x00000100 and 0 < (v := int(getattr(info, "ProcessMemoryLimit", 0))) < (1 << 60): cand_limits.append(v)
            if flags & 0x00000200 and 0 < (v := int(getattr(info, "JobMemoryLimit", 0))) < (1 << 60): cand_limits.append(v)
            if flags & 0x00000001 and 0 < (v := int(getattr(info.BasicLimitInformation, "MaximumWorkingSetSize", 0))) < (1 << 60): cand_limits.append(v)
            if not cand_limits: return None
            rss = int(psutil.Process(os.getpid()).memory_info().rss)
            return max(0, min(max(0, lim - rss) for lim in cand_limits)) if cand_limits else None
        except Exception: return None

    @staticmethod
    def _bsd_limit() -> Optional[int]:
        try:
            import resource, psutil
            rss = psutil.Process(os.getpid()).memory_info().rss
            cand: list[int] = []
            for name in ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_RSS"):
                if (lim := getattr(resource, name, None)) is None: continue
                soft, _ = resource.getrlimit(lim)
                if soft == getattr(resource, "RLIM_INFINITY", -1) or soft <= 0: continue
                cand.append(max(0, int(soft) - int(rss)))
            return min(cand) if cand else None
        except Exception: return None

    @staticmethod
    def total() -> Optional[int]:
        try:
            import psutil
            if getattr(vm := psutil.virtual_memory(), "total", None): return int(vm.total)
        except Exception: pass
        try:
            match sys.platform:
                case p if p.startswith("linux"):
                    with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as fh:
                        for line in fh:
                            if line.startswith("MemTotal:") and len(parts := line.split()) >= 2 and parts[1].isdigit():
                                return int(parts[1]) * 1024
                case "darwin":
                    import subprocess
                    if (out := subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode("utf-8", "ignore").strip()).isdigit():
                        return int(out)
                case p if p.startswith("win") or os.name == "nt":
                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong)]
                    stat = MEMORYSTATUSEX()
                    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)): return int(stat.ullTotalPhys)
        except Exception: pass
        return None

    @staticmethod
    def available() -> int:
        candidates = [x for x in (Memory._sys_available_memory(), Memory._linux_limit(), Memory._windows_limit(), Memory._bsd_limit()) if isinstance(x, int) and x >= 0]
        return max(0, min(candidates)) if candidates else 0

    @staticmethod
    def mem_get_info(device: torch.device) -> Tuple[Optional[int], Optional[int]]:
        free, total = None, None
        with contextlib.suppress(Exception):
            if (acc := getattr(torch, "accelerator", None)):
                if callable(get_memory_info := getattr(acc, "get_memory_info", None)):
                    free, total = Memory._parse_free(get_memory_info(device))
                if free is None or total is None:
                    if callable(mem_get_info := getattr(getattr(acc, "memory", None), "mem_get_info", None)):
                        f2, t2 = Memory._parse_free(mem_get_info(device))
                        if free is None: free = f2
                        if total is None: total = t2
                        
        if free is None or total is None:
            match getattr(device, "type", "cpu"):
                case "cuda" if torch.cuda.is_available():
                    with contextlib.suppress(Exception):
                        f2, t2 = torch.cuda.mem_get_info(device=device)
                        if free is None: free = int(f2)
                        if total is None: total = int(t2)
                case "xpu" if hasattr(torch, "xpu"):
                    with contextlib.suppress(Exception):
                        if callable(mem_get_info := getattr(getattr(torch.xpu, "memory", None), "mem_get_info", None) or getattr(torch.xpu, "mem_get_info", None)):
                            f2, t2 = mem_get_info(device)
                            if free is None: free = int(f2)
                            if total is None: total = int(t2)
                case "mps" if hasattr(torch, "mps"):
                    with contextlib.suppress(Exception):
                        if (total_v := int(torch.mps.recommended_max_memory())) > 0:
                            if total is None: total = total_v
                            if free is None: free = max(0, total_v - int(torch.mps.driver_allocated_memory()))
                            
        return max(0, int(free)) if free is not None else None, max(0, int(total)) if total is not None else None

    @staticmethod
    def prefer_local_numa() -> bool:
        try:
            import numa
            if hasattr(numa, "available") and numa.available():
                numa.set_membind([numa.current_node()])
                return True
        except Exception: pass
        try:
            if sys.platform.startswith("linux"):
                lib = ctypes.CDLL("libnuma.so.1")
                if int(lib.numa_available()) < 0: return False
                cpu = int(list(os.sched_getaffinity(0))[0]) if hasattr(os, "sched_getaffinity") and list(os.sched_getaffinity(0)) else 0
                lib.numa_node_of_cpu.argtypes, lib.numa_node_of_cpu.restype = [ctypes.c_int], ctypes.c_int
                lib.numa_set_preferred.argtypes, lib.numa_set_preferred.restype = [ctypes.c_int], None
                lib.numa_set_preferred(ctypes.c_int(int(lib.numa_node_of_cpu(ctypes.c_int(cpu)))))
                return True
        except Exception: pass
        return False


class Monitor:
    _LOGGER = logging.getLogger(__name__)

    _nvml: object | None = None
    _NVML_BACKOFF_UNTIL: float = 0.0
    _NVML_FAIL_COUNT: int = 0
    _NVML_HANDLE_CACHE: dict[int, object] = {}
    _NVML_UTIL_CACHE: dict[int, tuple[float, float | None, float | None]] = {}
    _NVML_READY: bool = False
    _NVML_TRIED: bool = False
    _NVML_LOCK = threading.Lock()
    _NVML_QUERY_LOCK = threading.Lock()

    _PSUTIL: object | None = None
    _PSUTIL_TRIED: bool = False

    _TIMING_EVENTS_UNSUPPORTED = object()
    _TIMING_EVENT_TLS = threading.local()

    @classmethod
    def is_nvml_disabled(cls: type[Self]) -> bool:
        return env_bool("ENN_NVML_DISABLE", False) or not env_bool("ENN_NVML", True)

    @classmethod
    def nvml_cfg(cls: type[Self], key: str, default: object, cast_fn: type = int) -> object:
        return cast_fn(env_first((f"ENN_NVML_{key}", f"ENN_NVML_{key}_S"), default=default))

    @classmethod
    def is_nvml_blocked(cls: type[Self], now: object | None = None) -> bool:
        now_f = float(now or time.perf_counter())
        with cls._NVML_LOCK:
            until = float(cls._NVML_BACKOFF_UNTIL or 0.0)
        return bool(until > 0.0 and now_f < until)

    @classmethod
    def is_nvml_available(cls: type[Self]) -> bool:
        nogil = bool(CPU.is_optimized_for_no_gil())
        if cls.is_nvml_blocked():
            if nogil:
                with cls._NVML_LOCK: return bool(cls._NVML_READY)
            return bool(cls._NVML_READY)

        if nogil:
            with cls._NVML_LOCK:
                if cls._NVML_TRIED: return bool(cls._NVML_READY)
        elif cls._NVML_TRIED:
            return bool(cls._NVML_READY)

        if cls.is_nvml_disabled():
            with cls._NVML_LOCK:
                cls._NVML_TRIED, cls._NVML_READY, cls._nvml = True, False, None
            return False

        with cls._NVML_LOCK:
            if ((until := float(cls._NVML_BACKOFF_UNTIL or 0.0)) > 0.0 and float(time.perf_counter()) < until) or cls._NVML_TRIED:
                return bool(cls._NVML_READY)
            cls._NVML_TRIED = True
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    import pynvml
                cls._nvml = pynvml
                getattr(pynvml, "nvmlInit")()
                cls._NVML_READY = True
            except Exception as exc:
                cls._nvml, cls._NVML_READY = None, False
                if env_bool("ENN_DEBUG", False):
                    cls._LOGGER.debug("NVML init failed: %s", exc, exc_info=True)
        return bool(cls._NVML_READY)

    @classmethod
    def gpu_nvml_utils(cls: type[Self], device: Union[torch.device, str, int]) -> tuple[float | None, float | None]:
        try: dev = device if isinstance(device, torch.device) else torch.device(device)
        except Exception: return (None, None)
        
        if getattr(dev, "type", "") != "cuda": return (None, None)
        idx_i = int(dev.index if dev.index is not None else get_accelerator_index("cuda"))

        gpu_util, mem_util = None, None
        if cls.is_nvml_available() and (nvml := cls._nvml) is not None:
            if cls.is_nvml_blocked(now := time.perf_counter()): return (None, None)
            
            try: min_interval = float(cls.nvml_cfg("MIN_INTERVAL", 0.0, float))
            except Exception: min_interval = 0.0

            if min_interval > 0.0 and not bool(CPU.is_optimized_for_no_gil()):
                if (cached := cls._NVML_UTIL_CACHE.get(idx_i)) is not None:
                    if now - float(cached[0]) < min_interval: return (cached[1], cached[2])

            with cls._NVML_QUERY_LOCK:
                if cls.is_nvml_blocked(now): return (None, None)
                if min_interval > 0.0:
                    if (cached := cls._NVML_UTIL_CACHE.get(idx_i)) is not None:
                        if now - float(cached[0]) < min_interval: return (cached[1], cached[2])
                try:
                    h = cls._NVML_HANDLE_CACHE.setdefault(idx_i, getattr(nvml, "nvmlDeviceGetHandleByIndex")(idx_i))
                    u = getattr(nvml, "nvmlDeviceGetUtilizationRates")(h)
                    if getattr(mi := getattr(nvml, "nvmlDeviceGetMemoryInfo")(h), "total", 0):
                        mem_util = 100.0 * float(mi.used) / float(mi.total)
                    gpu_util = float(getattr(u, "gpu", 0.0))
                    with cls._NVML_LOCK: cls._NVML_FAIL_COUNT, cls._NVML_BACKOFF_UNTIL = 0, 0.0
                except Exception:
                    with contextlib.suppress(Exception): cls._NVML_HANDLE_CACHE.pop(idx_i, None)
                    with contextlib.suppress(Exception): cls._NVML_UTIL_CACHE.pop(idx_i, None)
                    
                    try: fail_max = int(cls.nvml_cfg("FAIL_MAX", 3))
                    except Exception: fail_max = 3
                    try: backoff_s = float(cls.nvml_cfg("BACKOFF", 30.0 if bool(CPU.is_optimized_for_no_gil()) else 10.0, float))
                    except Exception: backoff_s = 0.0

                    trigger_backoff = False
                    with cls._NVML_LOCK:
                        cls._NVML_FAIL_COUNT = int(cls._NVML_FAIL_COUNT) + 1
                        if backoff_s > 0.0 and int(cls._NVML_FAIL_COUNT) >= int(fail_max):
                            cls._NVML_BACKOFF_UNTIL, cls._NVML_FAIL_COUNT, trigger_backoff = float(time.perf_counter()) + float(backoff_s), 0, True
                            
                    if trigger_backoff:
                        with contextlib.suppress(Exception): cls._NVML_HANDLE_CACHE.clear()
                        with contextlib.suppress(Exception): cls._NVML_UTIL_CACHE.clear()
                        with contextlib.suppress(Exception): cls._LOGGER.warning("[NVML] backing off %.1fs", float(backoff_s))
                    gpu_util, mem_util = None, None
                    
                if gpu_util is not None or mem_util is not None:
                    cls._NVML_UTIL_CACHE[idx_i] = (float(now), gpu_util, mem_util)

        if mem_util is None:
            with contextlib.suppress(Exception): mem_util = available_device_memory(torch.device("cuda", idx_i))
        return (gpu_util, mem_util)

    @staticmethod
    def xpu_mem_util(device: Union[torch.device, str, int]) -> float | None:
        try:
            if getattr(dev := device if isinstance(device, torch.device) else torch.device(device), "type", "") != "xpu": return None
            with contextlib.suppress(Exception): return available_device_memory(dev)
        except Exception: pass
        return None

    @staticmethod
    def mps_mem_util(device: Union[torch.device, str, int]) -> float | None:
        try:
            if getattr(dev := device if isinstance(device, torch.device) else torch.device(device), "type", "") != "mps": return None
            with contextlib.suppress(Exception): return available_device_memory(dev)
        except Exception: pass
        return None

    @classmethod
    def cpu_load(cls: type[Self]) -> float | None:
        if not cls._PSUTIL_TRIED:
            cls._PSUTIL_TRIED = True
            with contextlib.suppress(Exception):
                import psutil
                cls._PSUTIL = psutil
        try: return float(getattr(cls._PSUTIL, "cpu_percent")(interval=0.0)) if cls._PSUTIL else None
        except Exception: return None

    @staticmethod
    def is_clock_synchronized(dev_type: str) -> bool:
        if env_bool("ENN_TIMER_SYNC", False) or env_bool("ENN_WALLCLOCK_TIMER_SYNC", False): return True
        return env_bool(f"ENN_{str(dev_type or 'cpu').upper()}_TIMER_SYNC", False)

    @staticmethod
    def is_event_timer_available(device: torch.device) -> bool:
        try: return bool(is_accelerator_timer_supported(str(getattr(device, "type", "cpu"))))
        except Exception: return False

    @staticmethod
    def new_event_timer(device: torch.device) -> object | None:
        try: dev_type = str(getattr(device, "type", "cpu"))
        except Exception: dev_type = "cpu"
        if not is_accelerator_timer_supported(dev_type): return None
        return new_accelerator_event(device, enable_timing=True)

    @classmethod
    def get_thread_events(cls: type[Self], device: torch.device, slot: str) -> object | None:
        try: dev_type = str(getattr(device, "type", "cpu"))
        except Exception: dev_type = "cpu"
        if not is_accelerator_timer_supported(dev_type): return None
        
        d = getattr(tls := cls._TIMING_EVENT_TLS, "events", None)
        if d is None:
            d = {}
            setattr(tls, "events", d)
            
        try: dev_idx = int(device.index) if device.index is not None else -1
        except Exception: dev_idx = -1
        
        key = (str(slot), str(dev_type), int(dev_idx))
        if (cached := d.get(key, cls._TIMING_EVENTS_UNSUPPORTED)) is cls._TIMING_EVENTS_UNSUPPORTED:
            if (ev_s := cls.new_event_timer(device)) is None or (ev_e := cls.new_event_timer(device)) is None:
                d[key] = None
                return None
            cached = (ev_s, ev_e)
            d[key] = cached
        return cached


with contextlib.suppress(Exception):
    _install_matmul_precision_legacy_shim_if_needed()
