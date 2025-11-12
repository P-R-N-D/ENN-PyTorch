# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import ctypes
import importlib
import itertools
import multiprocessing
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.multiprocessing as mp

try:
    from zoneinfo import ZoneInfo  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - zoneinfo may be unavailable
    ZoneInfo = None  # type: ignore[assignment]


@dataclass
class _RuntimeConfig:
    deterministic: bool = False
    allow_tf32: Optional[bool] = None
    cudnn_benchmark: Optional[bool] = None
    matmul_precision: Optional[str] = None
    sdpa_backends: Optional[List[str]] = None
    te_first: bool = True


_RUNTIME_CONFIG = _RuntimeConfig()


_TZ_ALIASES = {
    "KST": "Asia/Seoul",
    "GMT": "Etc/GMT",
    "UTC": "UTC",
}


def resolve_timezone(name: Optional[str] = None) -> Optional[timezone]:
    """Return a tzinfo instance for the given alias or IANA name."""
    resolved = (name or "GMT").strip()
    alias = _TZ_ALIASES.get(resolved.upper(), resolved)
    if alias.upper() == "UTC":
        return timezone.utc
    if ZoneInfo is None:
        return None
    with contextlib.suppress(Exception):
        return ZoneInfo(alias)
    return None


def posix_time(tz_name: Optional[str] = None) -> int:
    """Current timestamp in nanoseconds for the specified timezone alias."""
    tz = resolve_timezone(tz_name)
    now = datetime.now(tz=tz) if tz is not None else datetime.now()
    return int(now.timestamp() * 1_000_000_000)


def system_info() -> Tuple[str, str, str, str]:
    """Gather basic OS, kernel, architecture, and accelerator information."""
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
    accelerators: List[str] = []
    try:
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                accelerators.append(f"cuda:{idx}={torch.cuda.get_device_name(idx)}")
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerators.append("mps=Apple MPS")
    except Exception:
        pass
    try:
        if hasattr(torch, "xpu") and hasattr(torch.xpu, "device_count"):
            count = torch.xpu.device_count()  # type: ignore[attr-defined]
            if count and count > 0:
                get_name = getattr(torch.xpu, "get_device_name", None)
                for idx in range(count):
                    name = get_name(idx) if callable(get_name) else "XPU"
                    accelerators.append(f"xpu:{idx}={name}")
    except Exception:
        pass
    try:
        if hasattr(torch, "is_vulkan_available") and torch.is_vulkan_available():
            accelerators.append("vulkan=available")
    except Exception:
        pass
    return os_name, kernel, arch, ";".join(accelerators)


def cpu_info(max_bytes: Optional[int] = None) -> str:
    """Return a semicolon-delimited list of per-core CPU names."""
    names: List[str] = []
    total = os.cpu_count() or 1
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


def _num_cuda_devices() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        return max(0, int(torch.cuda.device_count()))
    except Exception:
        return 0


def get_runtime_config() -> _RuntimeConfig:
    return _RUNTIME_CONFIG


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
    paths: List[str] = list(env_paths)
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
    raise RuntimeError(
        "No supported multiprocessing start method "
        "(tried forkserver, spawn)."
    )


def set_multiprocessing_env() -> None:
    try:
        mp.set_sharing_strategy("file_system")
    except RuntimeError:
        pass
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
        "Unable to configure multiprocessing start method "
        "(tried forkserver, spawn)."
    ) from last_error


def default_temp() -> str:
    return (
        os.environ.get("TEMP", r"C:\Windows\Temp")
        if sys.platform.startswith("win")
        else "/tmp"
        if os.path.isdir("/tmp")
        else "/var/tmp"
    )


def new_dir(prefix: str) -> str:
    base = default_temp()
    os.makedirs(base, exist_ok=True)
    directory = os.path.join(base, f"{prefix}_{os.getpid()}_{os.urandom(4).hex()}")
    os.makedirs(directory, exist_ok=True)
    return directory


def get_dpa_backends() -> List[object]:
    names = _RUNTIME_CONFIG.sdpa_backends or []
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
    backends: List[object] = []
    for name in names:
        key = mapping.get(name, name)
        if hasattr(SDPBackend, key):
            backends.append(getattr(SDPBackend, key))
    return backends


def is_cpu_bf16_supported() -> bool:
    try:
        mkldnn_ops = getattr(torch.ops, "mkldnn", None)
        if mkldnn_ops is not None and hasattr(
            mkldnn_ops, "_is_mkldnn_bf16_supported"
        ):
            return bool(torch.ops.mkldnn._is_mkldnn_bf16_supported())
    except Exception:
        pass
    return False


def is_cuda_bf16_supported() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        f = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(f):
            return bool(f())
        major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
        return major >= 8
    except Exception:
        return False


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
    cfg = _RUNTIME_CONFIG
    if deterministic is not None:
        cfg.deterministic = bool(deterministic)
    det_flag = cfg.deterministic
    allow_val = (
        bool(allow_tf32)
        if allow_tf32 is not None
        else bool(cfg.allow_tf32)
        if cfg.allow_tf32 is not None and deterministic is None
        else False
        if det_flag
        else True
    )
    cfg.allow_tf32 = allow_val
    benchmark_val = (
        bool(cudnn_benchmark)
        if cudnn_benchmark is not None
        else bool(cfg.cudnn_benchmark)
        if cfg.cudnn_benchmark is not None and deterministic is None
        else False
        if det_flag
        else True
    )
    cfg.cudnn_benchmark = benchmark_val
    precision_val = (
        str(matmul_precision)
        if matmul_precision is not None
        else str(cfg.matmul_precision)
        if cfg.matmul_precision is not None and deterministic is None
        else "highest"
        if det_flag
        else "high"
    )
    cfg.matmul_precision = precision_val
    if sdpa_backends is not None:
        cfg.sdpa_backends = [str(x) for x in sdpa_backends]
    if te_first is not None:
        cfg.te_first = bool(te_first)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = cfg.deterministic
        torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(cfg.allow_tf32)
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision(str(cfg.matmul_precision))
        except Exception:
            pass
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif hasattr(torch, "is_vulkan_available") and torch.is_vulkan_available():
        device = torch.device("vulkan")
    else:
        device = torch.device("cpu")
    return device


def optimal_optimizer_params(
    device: torch.device, use_foreach: Optional[bool], use_fused: bool
) -> Dict[str, bool]:
    devt = device.type
    flags: Dict[str, bool] = {}
    flags["foreach"] = (
        devt in {"cuda", "xpu"} if use_foreach is None else bool(use_foreach)
    )
    if use_fused and devt in {"cuda", "xpu"}:
        flags["fused"] = True
        flags["foreach"] = False
    return flags


def cuda_compute_capability(device: torch.device) -> Tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return (0, 0)
    try:
        major, minor = torch.cuda.get_device_capability(device)
    except Exception:
        return (0, 0)
    return (int(major), int(minor))


def is_float8_supported(
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[bool, str]:
    dev = torch.device(device) if device is not None else get_device()
    if dev.type != "cuda" or not torch.cuda.is_available():
        return (False, f"FP8 requires CUDA (found {dev.type})")
    major, minor = cuda_compute_capability(dev)
    if major <= 0:
        return (False, "Unable to query CUDA compute capability")
    if major < 9:
        return (False, f"FP8 requires sm_90+ (found sm_{major}{minor})")
    try:
        import transformer_engine.pytorch as te

        backend = getattr(te, "__name__", "transformer_engine.pytorch")
        return (True, backend)
    except Exception:
        return (False, "transformer_engine not found")


def is_int8_supported(
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[bool, str]:
    dev = torch.device(device) if device is not None else get_device()
    if dev.type != "cuda" or not torch.cuda.is_available():
        return (False, f"INT8 requires CUDA (found {dev.type})")
    major, minor = cuda_compute_capability(dev)
    if major <= 0:
        return (False, "Unable to query CUDA compute capability")
    if major < 7:
        return (False, f"INT8 requires sm_70+ (found sm_{major}{minor})")
    try:
        importlib.import_module("torchao.quantization")
        return (True, "torchao.quantization")
    except Exception:
        return (True, f"sm_{major}{minor}")


def is_int4_supported(
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[bool, str]:
    dev = torch.device(device) if device is not None else get_device()
    if dev.type != "cuda" or not torch.cuda.is_available():
        return (False, f"INT4 requires CUDA (found {dev.type})")
    major, minor = cuda_compute_capability(dev)
    if major <= 0:
        return (False, "Unable to query CUDA compute capability")
    if major < 8:
        return (False, f"INT4 requires sm_80+ (found sm_{major}{minor})")
    try:
        importlib.import_module("torchao.optim")
        return (True, "torchao.optim")
    except Exception:
        with contextlib.suppress(Exception):
            importlib.import_module("torchao.prototype.low_bit_optim")
            return (True, f"sm_{major}{minor}")
    return (False, "torchao low-bit optimizers unavailable")


def optimal_procs() -> Dict[str, Union[int, str]]:
    n_gpu = _num_cuda_devices()
    return {"nproc_per_node": n_gpu or 1, "device": "cuda" if n_gpu else "cpu"}


def cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


def num_accelerators() -> int:
    """Return the number of accelerator devices available on the host."""
    try:
        import torch
    except Exception:
        return 0
    # CUDA
    try:
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            return int(torch.cuda.device_count()) or 0
    except Exception:
        pass
    # Intel XPU (native XPU / IPEX)
    try:
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and callable(getattr(xpu, "is_available", None)) and xpu.is_available():
            count = int(getattr(xpu, "device_count", lambda: 1)()) or 1
            return max(count, 1)
    except Exception:
        pass
    # Apple MPS
    try:
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and callable(getattr(mps, "is_available", None)) and mps.is_available():
            return 1
    except Exception:
        pass
    # Vulkan (prototype/unstable)
    try:
        if hasattr(torch, "is_vulkan_available") and torch.is_vulkan_available():
            return 1
    except Exception:
        pass
    return 0


def optimal_threads() -> Dict[str, Union[int, bool]]:
    """Heuristics for CPU/GPU threading & mapper parallelism (no env-vars).
    Returns keys for ParallelMapper(method='threads'):
      - "intra_ops", "inter_ops", "num_workers", "max_concurrancy"
    """
    ncpu = cpu_count()
    try:
        import torch
        has_cuda = getattr(torch, "cuda", None) is not None and torch.cuda.is_available()
    except Exception:
        has_cuda = False
    nacc = num_accelerators()

    if ncpu <= 2:
        inter_ops = 1
        intra_ops = max(1, ncpu - inter_ops)
        num_workers = max(1, ncpu)
    elif ncpu <= 8:
        inter_ops = max(1, ncpu // 4)
        intra_ops = max(1, ncpu - inter_ops)
        num_workers = max(2, min(8, ncpu // 2))
    else:
        inter_ops = max(2, min(8, ncpu // 6))
        intra_ops = max(1, ncpu - inter_ops)
        num_workers = max(4, min(16, ncpu // 2))

    # no environment-variable overrides; compute directly
    max_concurrancy = (nacc * 2) if (nacc > 0 and has_cuda) else max(2, num_workers)

    return {
        "intra_ops": int(max(1, intra_ops)),
        "inter_ops": int(max(1, inter_ops)),
        "num_workers": int(max(1, num_workers)),
        "max_concurrancy": int(max(1, max_concurrancy)),
    }


def optimize_threads() -> Dict[str, Union[int, bool]]:
    """Apply thread hints via PyTorch APIs only (no env-vars)."""
    threads = optimal_threads()
    try:
        import torch
        torch.set_num_threads(int(threads["intra_ops"]))
    except Exception:
        pass
    try:
        import torch
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(int(threads["inter_ops"]))
    except Exception:
        pass
    return threads


class Thread:
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
    )

    def __init__(self, io_workers: int) -> None:
        self._psutil = self._import_psutil()
        self._allowed_cpus = self.total_procs() or list(range(max(1, os.cpu_count() or 1)))
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
        self.tune_threads(io_workers, initial=True)

    @staticmethod
    def _import_psutil():
        try:
            return importlib.import_module("psutil")
        except Exception:
            return None

    def total_procs(self) -> List[int]:
        if self._psutil is not None:
            try:
                proc = self._psutil.Process()
                if hasattr(proc, "cpu_affinity"):
                    cpus = proc.cpu_affinity()
                    if cpus:
                        return sorted({int(c) for c in cpus})
            except Exception:
                pass
        if os.name == "nt":
            try:
                k32 = ctypes.windll.kernel32
                k32.GetActiveProcessorGroupCount.restype = ctypes.c_ushort
                k32.GetActiveProcessorCount.argtypes = [ctypes.c_ushort]
                k32.GetActiveProcessorCount.restype = ctypes.c_ushort
                group_count = int(k32.GetActiveProcessorGroupCount())
                counts = [int(k32.GetActiveProcessorCount(i)) for i in range(max(1, group_count))]
                groups = list(range(group_count))
                try:
                    GetCurrentProcess = k32.GetCurrentProcess
                    GetProcessGroupAffinity = getattr(k32, "GetProcessGroupAffinity", None)
                    if GetProcessGroupAffinity:
                        handle = GetCurrentProcess()
                        arr_type = ctypes.c_ushort * max(1, group_count)
                        arr = arr_type()
                        needed = ctypes.c_ushort(group_count)
                        if GetProcessGroupAffinity(handle, ctypes.byref(needed), arr):
                            groups = list(arr)[: int(needed.value)]
                except Exception:
                    pass
                flattened: List[int] = []
                base = 0
                for gid, cnt in enumerate(counts):
                    if gid in groups:
                        flattened.extend(range(base, base + cnt))
                    base += cnt
                if flattened:
                    return flattened
            except Exception:
                pass
        try:
            return sorted(int(c) for c in os.sched_getaffinity(0))
        except Exception:
            pass
        return list(range(max(1, os.cpu_count() or 1)))

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
    def _pin_thread_windows(core: int) -> bool:
        try:
            k32 = ctypes.windll.kernel32
            GetActiveProcessorGroupCount = k32.GetActiveProcessorGroupCount
            GetActiveProcessorCount = k32.GetActiveProcessorCount
            GetCurrentThread = k32.GetCurrentThread
            SetThreadAffinityMask = k32.SetThreadAffinityMask
            SetThreadGroupAffinity = k32.SetThreadGroupAffinity
            SetThreadIdealProcessorEx = getattr(k32, "SetThreadIdealProcessorEx", None)
            GetActiveProcessorGroupCount.restype = ctypes.c_ushort
            GetActiveProcessorCount.argtypes = [ctypes.c_ushort]
            GetActiveProcessorCount.restype = ctypes.c_ushort
            group_count = int(GetActiveProcessorGroupCount())
            counts = [int(GetActiveProcessorCount(i)) for i in range(max(1, group_count))]
            total = sum(counts) or (os.cpu_count() or 1)
            idx = int(core) % max(1, total)
            group = 0
            within = idx
            for gid, cnt in enumerate(counts):
                if within < cnt:
                    group = gid
                    break
                within -= cnt
            thread = GetCurrentThread()
            if group_count <= 1:
                mask = ctypes.c_size_t(1 << within)
                prev = SetThreadAffinityMask(thread, mask.value)
                return bool(prev)

            class GROUP_AFFINITY(ctypes.Structure):
                _fields_ = [
                    ("Mask", ctypes.c_ulonglong),
                    ("Group", ctypes.c_ushort),
                    ("Reserved", ctypes.c_ushort * 3),
                ]

            affinity = GROUP_AFFINITY(
                ctypes.c_ulonglong(1 << within),
                ctypes.c_ushort(group),
                (ctypes.c_ushort * 3)(0, 0, 0),
            )
            ok = SetThreadGroupAffinity(thread, ctypes.byref(affinity), None)
            if ok and SetThreadIdealProcessorEx is not None:
                try:

                    class PROCESSOR_NUMBER(ctypes.Structure):
                        _fields_ = [
                            ("Group", ctypes.c_ushort),
                            ("Number", ctypes.c_ubyte),
                            ("Reserved", ctypes.c_ubyte),
                        ]

                    proc_num = PROCESSOR_NUMBER(group, within, 0)
                    SetThreadIdealProcessorEx(thread, ctypes.byref(proc_num), None)
                except Exception:
                    pass
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
            try:
                os.sched_setaffinity(0, {int(core)})
                return True
            except Exception:
                return False

    @staticmethod
    def _pin_thread_bsd(core: int) -> bool:
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
            elif "bsd" in plat:
                ok = self._pin_thread_bsd(core)
            elif plat == "darwin":
                try:
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
                except Exception:
                    ok = False
        self._tls.pinned = bool(ok)
        self._pin_attempts += 1
        if ok:
            self._pin_success += 1
        if self._pin_attempts >= 16 and self._pin_success == 0 and not self._omp_ok:
            self._enabled = False

    @staticmethod
    def optimize_threads(intra: Optional[int] = None, inter: Optional[int] = None) -> None:
        if intra is not None:
            try:
                torch.set_num_threads(max(1, int(intra)))
            except Exception:
                pass
        if inter is not None and hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(max(1, int(inter)))
            except Exception:
                pass

    def tune_threads(
        self, io_workers: Optional[int] = None, *_, initial: bool = False, **__
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
            try:
                intra = int(torch.get_num_threads())
            except Exception:
                intra = cpus
            if intra * tuned_workers > cpus:
                new_intra = max(1, cpus // tuned_workers)
                self.optimize_threads(intra=new_intra)
            want_inter = max(1, min(tuned_workers // 2, 4))
            self.optimize_threads(inter=want_inter)
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
        cpu_ratio = cpu_ns / float(wall_ns)
        cpus = max(1, len(self._allowed_cpus))
        workers = max(1, self._io_workers)
        if cpu_ratio >= 0.5:
            target_intra = max(1, cpus // workers)
            if cpus >= 8:
                target_intra = min(target_intra, 2)
            self.optimize_threads(intra=target_intra)
            self.optimize_threads(inter=max(1, min(2, workers)))
        else:
            relaxed = min(4, max(1, cpus // max(1, workers // 2)))
            current = max(1, torch.get_num_threads())
            if current < relaxed:
                self.optimize_threads(intra=relaxed)

    def new_thread(self, fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        if not self._enabled:
            return fn

        def _inner(x: Any) -> Any:
            self.pin_thread()
            t0 = time.perf_counter_ns()
            thread_time = getattr(time, "thread_time_ns", None)
            tc0 = thread_time() if callable(thread_time) else 0
            y = fn(x)
            tc1 = thread_time() if callable(thread_time) else 0
            t1 = time.perf_counter_ns()
            with self._lock:
                self._samples += 1
                self._cpu_ns += max(0, int(tc1) - int(tc0))
                self._wall_ns += max(0, int(t1) - int(t0))
            self.tune_threads()
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


def get_tlb(io_workers: Optional[int] = None) -> Thread:
    global _TLB_SINGLETON
    if _TLB_SINGLETON is None:
        default_workers = io_workers if io_workers is not None else max(1, (os.cpu_count() or 4) // 2)
        _TLB_SINGLETON = Thread(io_workers=default_workers)
    return _TLB_SINGLETON


def worker_init_pin(_: Any) -> None:
    get_tlb().pin_thread()


def wrap_with_tlb(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    return get_tlb().new_thread(fn)
