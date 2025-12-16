# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import ctypes
import importlib
from dataclasses import dataclass, replace
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

_FP32_PRECISION_CACHE: Dict[Tuple[str, int], str] = {}


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
        try:
            return len(os.sched_getaffinity(0))
        except Exception:
            return os.cpu_count() or 1

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

        def _read_int_env(keys: Sequence[str], default: int) -> int:
            for k in keys:
                with contextlib.suppress(Exception):
                    v = os.environ.get(k)
                    if v is not None and str(v).strip():
                        return int(v)
            return int(default)

        def _read_float_env(keys: Sequence[str], default: float) -> float:
            for k in keys:
                with contextlib.suppress(Exception):
                    v = os.environ.get(k)
                    if v is not None and str(v).strip():
                        return float(v)
            return float(default)

        local_world_guess = max(1, int(nacc or 1)) if is_accel else 1
        local_world_guess = max(
            1,
            _read_int_env(
                ("STNET_LOCAL_WORLD_SIZE", "LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE"),
                local_world_guess,
            ),
        )

        cap_mult = 2
        with contextlib.suppress(Exception):
            v = os.environ.get("STNET_THREAD_CAP_MULTIPLIER")
            if v is None:
                v = os.environ.get("STNET_THREADS_CAP_MULTIPLIER")
            if v is not None and str(v).strip():
                cap_mult = int(v)
        cap_mult = max(1, min(8, int(cap_mult)))
        node_thread_cap = max(2, int(ncpu_raw) * int(cap_mult))

        distribute = (is_accel and int(local_world_guess) > 1)
        distribute = bool(_read_int_env(("STNET_DISTRIBUTE_THREAD_CAP",), int(distribute)))
        thread_cap = int(node_thread_cap)
        if distribute:
            thread_cap = max(2, int(node_thread_cap) // max(1, int(local_world_guess)))

        eff_cores = max(1, int(thread_cap) // max(1, int(cap_mult)))

        soft_inflight = 8 if is_accel else 4
        with contextlib.suppress(Exception):
            from ..data.pipeline import LoaderPolicy

            lp = LoaderPolicy()
            hard = int(lp.hard_inflight_batches(dev_type))
            soft_inflight = max(1, int(hard * max(1, int(lp.soft_cap_multiplier))))

        soft_auto_enabled = bool(_read_int_env(("STNET_SOFT_INFLIGHT_AUTO",), 1))
        soft_inflight_max_default = 16 if is_accel else 12
        soft_inflight_max = max(8, _read_int_env(("STNET_SOFT_INFLIGHT_MAX",), soft_inflight_max_default))
        soft_inflight_explicit = _read_int_env(("STNET_SOFT_INFLIGHT_CAP",), 0)
        if soft_inflight_explicit > 0:
            soft_inflight = max(1, int(soft_inflight_explicit))
        elif soft_auto_enabled:
            soft_base = max(0, _read_int_env(("STNET_SOFT_INFLIGHT_BASE",), 2))
            soft_div = max(1, _read_int_env(("STNET_SOFT_INFLIGHT_DIV",), 4))
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
            v = os.environ.get(env_key)
            if v is not None and str(v).strip():
                model_ratio = float(v)
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

        env_pre = os.environ.get("STNET_PREBATCH")
        if env_pre:
            with contextlib.suppress(Exception):
                prebatch = max(1, int(env_pre))
        env_pf = os.environ.get("STNET_PREFETCH_FACTOR")
        if env_pf:
            with contextlib.suppress(Exception):
                prefetch_factor = max(1, int(env_pf))

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

    def as_threads_dict(self) -> Dict[str, int]:
        out = {
            "intra_ops": int(self.intra_ops),
            "inter_ops": int(self.inter_ops),
            "num_workers": int(self.num_workers),
            "max_concurrency": int(self.max_concurrency),
            "prebatch": int(self.prebatch),
            "prefetch_factor": int(self.prefetch_factor),
        }
        out["max_concurrancy"] = out["max_concurrency"]
        return out

    def as_procs_dict(self) -> Dict[str, Union[int, str]]:
        return {
            "nproc_per_node": int(self.nproc_per_node),
            "device": str(self.device),
        }

    def apply_torch_threads(self) -> None:
        try:
            torch.set_num_threads(max(1, int(self.intra_ops)))
        except Exception:
            pass
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(max(1, int(self.inter_ops)))
            except Exception:
                pass

def set_float32_precision(
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    autocast_dtype: Optional[torch.dtype] = None,
) -> None:
    if device.type != "cuda":
        return

    use_tf32 = False
    for _dt in (dtype, autocast_dtype):
        if _dt is None:
            continue
        if _dt not in (torch.float32, torch.float64):
            use_tf32 = True
            break

    precision = "tf32" if use_tf32 else "ieee"
    key = (device.type, int(device.index) if device.index is not None else -1)
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
    from ..data.pipeline import Dataset

    return Dataset.is_cpu_bf16_supported()


def is_cuda_bf16_supported() -> bool:
    from ..data.pipeline import Dataset

    return Dataset.is_cuda_bf16_supported()


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
    from ..data.pipeline import Dataset

    return Dataset.cuda_compute_capability(device)


def is_float8_supported(
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[bool, str]:
    from ..data.pipeline import Dataset

    return Dataset.is_float8_supported(device)


def is_int8_supported(
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[bool, str]:
    from ..data.pipeline import Dataset

    return Dataset.is_int8_supported(device)


def is_int4_supported(
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[bool, str]:
    from ..data.pipeline import Dataset

    return Dataset.is_int4_supported(device)


def optimal_procs() -> Dict[str, Union[int, str]]:
    return WorkerPolicy.autotune().as_procs_dict()


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
    return WorkerPolicy.autotune().as_threads_dict()


def optimize_threads() -> Dict[str, Union[int, bool]]:
    wp = WorkerPolicy.autotune()
    wp.apply_torch_threads()
    return wp.as_threads_dict()


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
        wp = WorkerPolicy.autotune()
        if intra is not None:
            wp = replace(wp, intra_ops=int(intra))
        if inter is not None:
            wp = replace(wp, inter_ops=int(inter))
        wp.apply_torch_threads()

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
            def _read_int_env(keys, default):
                for k in keys:
                    with contextlib.suppress(Exception):
                        v = os.environ.get(k)
                        if v is not None and str(v).strip():
                            return int(v)
                return int(default)

            cap_mult = 2
            with contextlib.suppress(Exception):
                v = os.environ.get("STNET_THREAD_CAP_MULTIPLIER")
                if v is None:
                    v = os.environ.get("STNET_THREADS_CAP_MULTIPLIER")
                if v is not None and str(v).strip():
                    cap_mult = int(v)
            cap_mult = max(1, min(8, int(cap_mult)))
            node_thread_cap = max(2, int(cpus) * int(cap_mult))

            local_world = max(
                1,
                _read_int_env(
                    ("STNET_LOCAL_WORLD_SIZE", "LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE"),
                    1,
                ),
            )
            distribute = bool(_read_int_env(("STNET_DISTRIBUTE_THREAD_CAP",), int(local_world > 1)))
            thread_cap = int(node_thread_cap)
            if distribute and local_world > 1:
                thread_cap = max(2, int(node_thread_cap) // int(local_world))

            try:
                intra = int(torch.get_num_threads())
            except Exception:
                intra = cpus

            want_inter = max(1, min(tuned_workers // 2, 4))
            total = int(intra) + int(want_inter) + int(tuned_workers)
            if total > int(thread_cap):
                new_intra = max(
                    1,
                    int(thread_cap) - int(want_inter) - int(tuned_workers),
                )
                if int(new_intra) != int(intra):
                    self.optimize_threads(intra=int(new_intra))
                    intra = int(new_intra)
                total = int(intra) + int(want_inter) + int(tuned_workers)
                if total > int(thread_cap):
                    want_inter = max(
                        1,
                        int(thread_cap) - int(tuned_workers) - int(intra),
                    )
            self.optimize_threads(inter=int(want_inter))
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

        def _read_int_env(keys, default):
            for k in keys:
                with contextlib.suppress(Exception):
                    v = os.environ.get(k)
                    if v is not None and str(v).strip():
                        return int(v)
            return int(default)

        cap_mult = 2
        with contextlib.suppress(Exception):
            v = os.environ.get("STNET_THREAD_CAP_MULTIPLIER")
            if v is None:
                v = os.environ.get("STNET_THREADS_CAP_MULTIPLIER")
            if v is not None and str(v).strip():
                cap_mult = int(v)
        cap_mult = max(1, min(8, int(cap_mult)))
        node_thread_cap = max(2, int(cpus) * int(cap_mult))

        local_world = max(
            1,
            _read_int_env(
                ("STNET_LOCAL_WORLD_SIZE", "LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE"),
                1,
            ),
        )
        distribute = bool(_read_int_env(("STNET_DISTRIBUTE_THREAD_CAP",), int(local_world > 1)))
        thread_cap = int(node_thread_cap)
        if distribute and local_world > 1:
            thread_cap = max(2, int(node_thread_cap) // int(local_world))

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
            self.optimize_threads(intra=int(new_intra))
            intra = int(new_intra)
        total = int(intra) + int(inter) + int(workers)
        if total > int(thread_cap):
            new_inter = max(1, int(thread_cap) - int(workers) - int(intra))
            if int(new_inter) < int(inter):
                self.optimize_threads(inter=int(new_inter))

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
            free_v = (
                info.get("free", None)
                or info.get("free_memory", None)
                or info.get("free_bytes", None)
            )
            total_v = (
                info.get("total", None)
                or info.get("total_memory", None)
                or info.get("total_bytes", None)
            )

            if total_v is None and info.get("bytes_limit", None) is not None:
                total_v = info.get("bytes_limit", None)
            if free_v is None and total_v is not None:
                used_v = (
                    info.get("bytes_used", None)
                    or info.get("bytes_in_use", None)
                    or info.get("bytes_used_current", None)
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
                    mem_get_info = (
                        getattr(mem_mod, "mem_get_info", None)
                        if mem_mod is not None
                        else None
                    )
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
                    mem_get_info = (
                        getattr(mem_mod, "mem_get_info", None)
                        if mem_mod is not None
                        else None
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
