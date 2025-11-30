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
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.multiprocessing as mp

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


_RUNTIME_CFG = SimpleNamespace(
    deterministic=False,
    allow_tf32=None,
    cudnn_benchmark=None,
    matmul_precision=None,
    sdpa_backends=None,
    te_first=True,
)


_TZ_ALIASES = {
    "KST": "Asia/Seoul",
    "GMT": "Etc/GMT",
    "UTC": "UTC",
}


def resolve_timezone(name: Optional[str] = None) -> Optional[timezone]:
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
    tz = resolve_timezone(tz_name)
    now = datetime.now(tz=tz) if tz is not None else datetime.now()
    return int(now.timestamp() * 1_000_000_000)


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
            count = torch.xpu.device_count()
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
    names: List[str] = []
    total = os.cpu_count() or 1
    brand = ""
    with contextlib.suppress(Exception):
        import cpuinfo

        info = cpuinfo.get_cpu_info() or {}
        brand = info.get("brand_raw") or info.get("brand") or ""
    if not brand and os.path.exists("/proc/cpuinfo"):
        with contextlib.suppress(Exception):
            with open(
                "/proc/cpuinfo", "r", encoding="utf-8", errors="ignore"
            ) as handle:
                lines = [ln.strip() for ln in handle.readlines() if "model name" in ln]
            if lines:
                names = [ln.split(":", 1)[1].strip() for ln in lines]
    if not names and platform.system() == "Darwin":
        with contextlib.suppress(Exception):
            import subprocess

            output = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"]
            )
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
    pairs = [
        f"{idx}:{names[idx] if idx < len(names) else names[0]}" for idx in range(total)
    ]
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
    if (
        isinstance(main_path, str)
        and main_path.startswith("<")
        and main_path.endswith(">")
    ):
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
        "No supported multiprocessing start method " "(tried forkserver, spawn)."
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
        "Unable to configure multiprocessing start method " "(tried forkserver, spawn)."
    ) from last_error


def default_temp() -> str:
    return (
        os.environ.get("TEMP", r"C:\Windows\Temp")
        if sys.platform.startswith("win")
        else "/tmp" if os.path.isdir("/tmp") else "/var/tmp"
    )


def new_dir(prefix: str) -> str:
    base = default_temp()
    os.makedirs(base, exist_ok=True)
    directory = os.path.join(base, f"{prefix}_{os.getpid()}_{os.urandom(4).hex()}")
    os.makedirs(directory, exist_ok=True)
    return directory


def get_dpa_backends() -> List[object]:

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
    backends: List[object] = []
    for name in names:
        key = mapping.get(name, name)
        if hasattr(SDPBackend, key):
            backends.append(getattr(SDPBackend, key))
    return backends


def is_cpu_bf16_supported() -> bool:
    try:
        mkldnn_ops = getattr(torch.ops, "mkldnn", None)
        if mkldnn_ops is not None and hasattr(mkldnn_ops, "_is_mkldnn_bf16_supported"):
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
    cfg = _RUNTIME_CFG
    if deterministic is not None:
        cfg.deterministic = bool(deterministic)
    det_flag = cfg.deterministic
    allow_val = (
        bool(allow_tf32)
        if allow_tf32 is not None
        else (
            bool(cfg.allow_tf32)
            if cfg.allow_tf32 is not None and deterministic is None
            else False if det_flag else True
        )
    )
    cfg.allow_tf32 = allow_val
    benchmark_val = (
        bool(cudnn_benchmark)
        if cudnn_benchmark is not None
        else (
            bool(cfg.cudnn_benchmark)
            if cfg.cudnn_benchmark is not None and deterministic is None
            else False if det_flag else True
        )
    )
    cfg.cudnn_benchmark = benchmark_val
    precision_val = (
        str(matmul_precision)
        if matmul_precision is not None
        else (
            str(cfg.matmul_precision)
            if cfg.matmul_precision is not None and deterministic is None
            else "highest" if det_flag else "high"
        )
    )
    cfg.matmul_precision = precision_val
    if sdpa_backends is not None:
        cfg.sdpa_backends = [str(x) for x in sdpa_backends]
    if te_first is not None:
        cfg.te_first = bool(te_first)
    if torch.cuda.is_available():
        try:
            idx_env = int(os.environ.get("LOCAL_RANK", 0))
        except Exception:
            idx_env = 0
        try:
            ndev = max(1, int(torch.cuda.device_count()))
        except Exception:
            ndev = 1
        idx = idx_env % ndev
        with contextlib.suppress(Exception):
            torch.cuda.set_device(idx)
        device = torch.device(f"cuda:{idx}")
        torch.backends.cudnn.deterministic = cfg.deterministic
        torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)
        try:
            torch.set_float32_matmul_precision(str(cfg.matmul_precision))
            fp32_precision = "tf32" if allow_val else "ieee"
            with contextlib.suppress(Exception):
                torch.backends.fp32_precision = fp32_precision
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.fp32_precision = fp32_precision
            if hasattr(torch.backends, "cudnn"):
                with contextlib.suppress(Exception):
                    torch.backends.cudnn.fp32_precision = fp32_precision
                if hasattr(torch.backends.cudnn, "conv"):
                    torch.backends.cudnn.conv.fp32_precision = fp32_precision
                if hasattr(torch.backends.cudnn, "rnn"):
                    torch.backends.cudnn.rnn.fp32_precision = fp32_precision
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

    try:
        import torch
    except Exception:
        return 0

    cuda_iface = getattr(torch, "cuda", None)
    if cuda_iface is not None:
        try:
            if cuda_iface.is_available():
                return int(cuda_iface.device_count()) or 0
        except Exception:
            pass

    try:
        xpu = getattr(torch, "xpu", None)
        if (
            xpu is not None
            and callable(getattr(xpu, "is_available", None))
            and xpu.is_available()
        ):
            count = int(getattr(xpu, "device_count", lambda: 1)()) or 1
            return max(count, 1)
    except Exception:
        pass

    try:
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if (
            mps is not None
            and callable(getattr(mps, "is_available", None))
            and mps.is_available()
        ):
            return 1
    except Exception:
        pass

    try:
        if hasattr(torch, "is_vulkan_available") and torch.is_vulkan_available():
            return 1
    except Exception:
        pass

    return 0


def optimal_threads() -> Dict[str, Union[int, bool]]:
    ncpu = cpu_count()
    try:
        import torch
        is_accelerated = torch.accelerator.is_available()
    except Exception:
        is_accelerated = False
    if ncpu <= 2:
        inter_ops = 1
        intra_ops = max(1, ncpu - inter_ops)
        num_workers = max(1, ncpu)
    elif 2 < ncpu <= 8:
        inter_ops = max(1, ncpu // 4)
        intra_ops = max(1, ncpu - inter_ops)
        num_workers = max(2, min(8, ncpu // 2))
    else:
        inter_ops = max(2, min(8, ncpu // 6))
        intra_ops = max(1, ncpu - inter_ops)
        num_workers = max(4, min(16, ncpu // 2))

    max_concurrancy = int(max(1, num_workers))
    prebatch = 64 if is_accelerated else 1
    prefetch_factor = 4 if is_accelerated else 1

    return {
        "intra_ops": int(max(1, intra_ops)),
        "inter_ops": int(max(1, inter_ops)),
        "num_workers": int(max(1, num_workers)),
        "max_concurrancy": int(max(1, max_concurrancy)),
        "prebatch": int(max(1, prebatch)),
        "prefetch_factor": int(max(1, prefetch_factor)),
    }


def optimize_threads() -> Dict[str, Union[int, bool]]:

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
        self._allowed_cpus = self.total_procs() or list(
            range(max(1, os.cpu_count() or 1))
        )
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
    def _import_psutil() -> ModuleType | None:
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
                counts = [
                    int(k32.GetActiveProcessorCount(i))
                    for i in range(max(1, group_count))
                ]
                groups = list(range(group_count))
                try:
                    GetCurrentProcess = k32.GetCurrentProcess
                    GetProcessGroupAffinity = getattr(
                        k32, "GetProcessGroupAffinity", None
                    )
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
            counts = [
                int(GetActiveProcessorCount(i)) for i in range(max(1, group_count))
            ]
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
    def optimize_threads(
        intra: Optional[int] = None, inter: Optional[int] = None
    ) -> None:
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
        default_workers = (
            io_workers if io_workers is not None else max(1, (os.cpu_count() or 4) // 2)
        )
        _TLB_SINGLETON = Thread(io_workers=default_workers)
    return _TLB_SINGLETON


def worker_init_pin(_: Any) -> None:
    get_tlb().pin_thread()


def wrap_with_tlb(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    return get_tlb().new_thread(fn)
class Memory:

    @staticmethod
    def total() -> Optional[int]:
        import sys
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
                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode("utf-8", "ignore")
                if out.strip().isdigit():
                    return int(out.strip())
            elif os.name == "nt" or sys.platform.startswith("win"):
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong)]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
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
    def _sys_available() -> Optional[int]:
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
                import ctypes

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
            import ctypes
            import psutil
            from ctypes import wintypes as wt

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
                import ctypes

                lib = ctypes.CDLL("libnuma.so.1")
                if lib.numa_available() < 0:
                    return False
                cpu = 0
                if hasattr(os, "sched_getaffinity"):
                    cpus = list(os.sched_getaffinity(0))
                    cpu = int(cpus[0]) if cpus else 0
                lib.numa_node_of_cpu.argtypes = [ctypes.c_int]
                lib.numa_node_of_cpu.restype = ctypes.c_int
                node = lib.numa_node_of_cpu(ctypes.c_int(cpu))
                lib.numa_set_preferred.argtypes = [ctypes.c_int]
                lib.numa_set_preferred.restype = None
                lib.numa_set_preferred(ctypes.c_int(node))
                return True
        except Exception:
            return False
        return False

    class Page:

        __slots__ = ("_buf", "_numel", "_dtype")

        def __init__(self, numel: int, dtype: "torch.dtype"):
            import torch

            self._numel = int(max(1, numel))
            self._dtype = dtype
            self._buf = torch.empty(
                self._numel, dtype=self._dtype, device="cpu", pin_memory=True
            )

        @property
        def numel(self) -> int:
            return self._numel

        @property
        def dtype(self):
            return self._dtype

        def view(self, *shape: int):
            import torch

            needed = 1
            for s in shape:
                needed *= int(s)
            if needed > self._numel:
                self._numel = int(needed)
                self._buf = torch.empty(
                    self._numel, dtype=self._dtype, device="cpu", pin_memory=True
                )
            return self._buf[:needed].view(*shape)

    class Pool:

        class Token:
            __slots__ = ("i", "g")

            def __init__(self, i: int, g: int):
                self.i = i
                self.g = g

        class _Entry:
            __slots__ = ("page", "busy", "fence", "gen")

            def __init__(self, page: "Memory.Page"):
                self.page = page
                self.busy = False
                self.fence = None
                self.gen = 0

        def __init__(self, capacity: int = 4):
            import threading

            self._cap = max(1, int(capacity))
            self._pages: list[Memory.Pool._Entry] = []
            self._rr = 0
            self._lock = threading.Lock()

        def _evt_done(self, evt: object) -> bool:
            if evt is None:
                return True
            try:
                q = getattr(evt, "query", None)
                if callable(q):
                    return bool(q())
            except Exception:
                return False
            return False

        def _scavenge(self) -> None:
            for e in self._pages:
                if e.busy and e.fence is not None and self._evt_done(e.fence):
                    e.busy = False
                    e.fence = None

        def _ensure_view(
            self, e: "Memory.Pool._Entry", shape: "Tuple[int, ...]", dtype: "torch.dtype"
        ):
            need = 1
            for s in shape:
                need *= int(s)
            if (e.page.dtype != dtype) or (e.page.numel < need):
                e.page = Memory.Page(numel=need, dtype=dtype)
                e.gen += 1
            return e.page.view(*shape)

        def get(
            self,
            shape: "Tuple[int, ...]",
            dtype: "torch.dtype",
            *,
            return_handle: bool = False,
        ):
            with self._lock:
                self._scavenge()
                n = len(self._pages)
                if n:
                    start = self._rr
                    for k in range(n):
                        idx = (start + k) % n
                        e = self._pages[idx]
                        if not e.busy:
                            e.busy = True
                            e.fence = None
                            self._rr = (idx + 1) % max(1, n)
                            view = self._ensure_view(e, shape, dtype)
                            if return_handle:
                                return view, Memory.Pool.Token(idx, e.gen)
                            return view
                need = 1
                for s in shape:
                    need *= int(s)
                new = Memory.Pool._Entry(Memory.Page(numel=need, dtype=dtype))
                new.busy = True
                if len(self._pages) < self._cap:
                    self._pages.append(new)
                    idx = len(self._pages) - 1
                    self._rr = (idx + 1) % self._cap
                    view = new.page.view(*shape)
                    if return_handle:
                        return view, Memory.Pool.Token(idx, new.gen)
                    return view
                start = self._rr
                for k in range(self._cap):
                    idx = (start + k) % self._cap
                    if not self._pages[idx].busy:
                        self._pages[idx] = new
                        self._rr = (idx + 1) % self._cap
                        view = new.page.view(*shape)
                        if return_handle:
                            return view, Memory.Pool.Token(idx, new.gen)
                        return view
                view = new.page.view(*shape)
                if return_handle:
                    return view, None
                return view

        def get_like(self, t: "torch.Tensor", *args: Any, return_handle: bool = False):
            return self.get(tuple(t.shape), t.dtype, return_handle=return_handle)

        def release_after(self, token: "Memory.Pool.Token", wait_event: object | None) -> None:
            if token is None:
                return
            with self._lock:
                i = int(getattr(token, "i", -1))
                g = int(getattr(token, "g", -1))
                if 0 <= i < len(self._pages):
                    e = self._pages[i]
                    if e.gen == g:
                        e.busy = True
                        e.fence = wait_event

        def release(self, token: "Memory.Pool.Token") -> None:
            if token is None:
                return
            with self._lock:
                i = int(getattr(token, "i", -1))
                g = int(getattr(token, "g", -1))
                if 0 <= i < len(self._pages):
                    e = self._pages[i]
                    if e.gen == g:
                        e.busy = False
                        e.fence = None

        def collect(self) -> None:
            with self._lock:
                self._scavenge()

    class Cache:

        def __init__(self, root: str, max_queue: int = 8):
            import os
            import queue
            import threading

            self._q = queue.Queue(maxsize=max_queue)
            self._root = root
            os.makedirs(root, exist_ok=True)
            self._t = threading.Thread(target=self._run, daemon=True)
            self._err = None
            self._err_event = threading.Event()
            self._t.start()

        def submit(
            self,
            tensor: "torch.Tensor",
            path: Optional[str] = None,
            idx: Optional[int] = None,
            wait_event: Optional[object] = None,
            release_cb: Optional[object] = None,
        ) -> None:
            import contextlib
            import queue

            if self._err_event.is_set():
                raise RuntimeError(f"Async writer error: {self._err!r}")
            if path is None:
                if idx is None:
                    raise ValueError("either path or idx required")
                path = os.path.join(self._root, f"chunk_{int(idx):06d}.pt")
            try:
                self._q.put((tensor, path, wait_event, release_cb), timeout=0.05)
            except queue.Full:
                if wait_event is not None:
                    with contextlib.suppress(Exception):
                        wait_event.synchronize()
                self._save_tensor(tensor, path)
                if callable(release_cb):
                    with contextlib.suppress(Exception):
                        release_cb()

        def _save_tensor(self, tensor: "torch.Tensor", path: str) -> None:
            import torch
            import json, os

            try:
                if path.endswith(".mmt"):
                    from tensordict import MemoryMappedTensor
                    buf = tensor
                    if hasattr(tensor, "is_pinned") and tensor.is_pinned():
                        buf = torch.empty_like(tensor, device="cpu", pin_memory=False); buf.copy_(tensor, non_blocking=False)
                    MemoryMappedTensor.from_tensor(buf.contiguous(), filename=path)
                    meta = {"shape": list(buf.shape), "dtype": str(buf.dtype).replace("torch.", "")}
                    with open(path + ".json", "w", encoding="utf-8") as f:
                        json.dump(meta, f)
                    return
            except Exception:
                pass

            if hasattr(tensor, "is_pinned") and tensor.is_pinned():
                buf = torch.empty_like(tensor, device="cpu", pin_memory=False)
                buf.copy_(tensor, non_blocking=False)
            else:
                buf = tensor.contiguous()
            torch.save(buf, path)

        def close(self) -> None:
            self._q.put((None, None, None, None))
            self._t.join()

        def _run(self):
            import contextlib

            while True:
                item = self._q.get()
                if isinstance(item, tuple) and len(item) == 4:
                    tensor, path, evt, rel = item
                else:
                    tensor, path = item
                    evt = None
                    rel = None
                if tensor is None:
                    break
                try:
                    if evt is not None:
                        with contextlib.suppress(Exception):
                            evt.synchronize()
                    self._save_tensor(tensor, path)
                    if callable(rel):
                        with contextlib.suppress(Exception):
                            rel()
                except Exception as e:
                    self._err = e
                    self._err_event.set()
                    break

        def had_error(self) -> bool:
            return bool(self._err_event.is_set())

    class Buffer:
        def __init__(self, max_batches: int):
            import queue
            import threading
            self.max_batches = max(1, int(max_batches))
            self._q: "queue.Queue[torch.Tensor]" = queue.Queue(maxsize=self.max_batches)
            self._stop = threading.Event()

        def put(self, tensor: "torch.Tensor") -> None:
            import time, logging

            start = time.monotonic()
            try:
                self._q.put(tensor, block=True, timeout=None)
            except Exception as e:
                logging.error(f"Buffer.put encountered unexpected exception: {e!r}")
                raise
            elapsed = time.monotonic() - start
            if elapsed > 0.1:
                logging.warning(
                    f"Buffer.put blocked for {elapsed:.3f} s (max_batches={self.max_batches})"
                )

        def get(self, block: bool = True, timeout: float | None = None) -> "torch.Tensor":
            return self._q.get(block=block, timeout=timeout)

        def empty(self) -> bool:
            return self._q.empty()

        def size(self) -> int:
            return self._q.qsize()

        def stop(self) -> None:
            self._stop.set()

        def is_stopped(self) -> bool:
            return bool(self._stop.is_set())
