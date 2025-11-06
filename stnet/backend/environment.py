
# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import importlib
import multiprocessing
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.multiprocessing as mp


@dataclass
class _RuntimeConfig:
    deterministic: bool = False
    allow_tf32: Optional[bool] = None
    cudnn_benchmark: Optional[bool] = None
    matmul_precision: Optional[str] = None
    sdpa_backends: Optional[List[str]] = None
    te_first: bool = True


_RUNTIME_CONFIG = _RuntimeConfig()


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

    def _ensure_front(candidate: Path | str | None) -> None:
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

    def _ensure_env(entry: str) -> None:
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
        _ensure_front(candidate)

    for entry in list(sys.path):
        if not entry:
            continue
        try:
            entry_str = os.fspath(entry)
        except TypeError:
            continue
        if not entry_str:
            continue
        _ensure_env(entry_str)

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
        os.environ.get("TEMP", "C:\Windows\Temp")
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


def initialize_sdpa_backends() -> List[object]:
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


def _safe_cuda_device_count() -> int:
    try:
        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        return 0
    return 0


def optimal_procs() -> Dict[str, Union[int, str]]:
    n_gpu = _safe_cuda_device_count()
    return {"nproc_per_node": n_gpu or 1, "device": "cuda" if n_gpu else "cpu"}


def cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


def optimal_threads() -> Dict[str, Union[int, bool]]:
    n_cpu = cpu_count()
    n_gpu = _safe_cuda_device_count()
    intra = max(1, n_cpu // max(1, n_gpu))
    inter = max(1, n_cpu // 2)
    workers = max(2, n_cpu // 2)
    if n_gpu:
        workers = max(n_gpu * 2, workers)
    return {
        "intraop": intra,
        "interop": inter,
        "dataloader_workers": workers,
        "prefetch_factor": 2,
        "pin_memory": bool(n_gpu > 0),
    }


def optimize_threads() -> Dict[str, Union[int, bool]]:
    threads = optimal_threads()
    os.environ.setdefault("OMP_NUM_THREADS", str(threads["intraop"]))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads["intraop"]))
    try:
        torch.set_num_threads(int(threads["intraop"]))
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(int(threads["interop"]))
    except Exception:
        pass
    return threads
