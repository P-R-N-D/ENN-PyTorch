# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import importlib
import math
import multiprocessing
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
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


def initialize_python_path() -> None:
    try:
        package_dir = Path(__file__).resolve().parents[1]
    except Exception:
        return
    project_dir = package_dir.parent
    candidates: List[str] = []
    seen: set[str] = set()
    for entry in (package_dir, project_dir):
        try:
            resolved = os.fspath(entry)
        except TypeError:
            continue
        if not resolved:
            continue
        if not Path(resolved).exists():
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates.append(resolved)
    if not candidates:
        return
    separator = os.pathsep
    current = os.environ.get("PYTHONPATH", "")
    paths = [path for path in current.split(separator) if path]
    for candidate in candidates:
        if candidate not in paths:
            paths.insert(0, candidate)
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    os.environ["PYTHONPATH"] = separator.join(paths)


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
        "Unable to configure multiprocessing start method (tried forkserver, spawn)."
    ) from last_error


def default_temp() -> str:
    return (
        os.environ.get("TEMP", "C:\\Windows\\Temp")
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


def is_port_available(host: str, port: int) -> bool:
    if port <= 0:
        return False
    try:
        infos = socket.getaddrinfo(
            host,
            port,
            socket.AF_UNSPEC,
            socket.SOCK_STREAM,
        )
    except socket.gaierror:
        return False
    for family, socktype, proto, _, sockaddr in infos:
        with contextlib.closing(socket.socket(family, socktype, proto)) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(sockaddr)
                return True
            except OSError:
                continue
    return False


def get_available_addr(endpoint: Optional[str]) -> str:
    import ipaddress

    default_host = "127.0.0.1"

    def _normalize(endpoint_str: str) -> Tuple[str, int]:
        value = endpoint_str.strip()
        if not value:
            return default_host, 0
        if value.startswith("["):
            if "]" not in value:
                raise ValueError(f"invalid endpoint: {endpoint_str}")
            closing = value.index("]")
            host_part = value[1:closing]
            remainder = value[closing + 1 :]
            if remainder.startswith(":") and remainder[1:]:
                return host_part, int(remainder[1:])
            if remainder in ("", ":"):
                return host_part, 0
            raise ValueError(f"invalid endpoint: {endpoint_str}")
        host_part, sep, port_part = value.rpartition(":")
        if sep and port_part.isdigit():
            return host_part or default_host, int(port_part)
        return value, 0

    host: str
    port: int
    if endpoint is None:
        host, port = default_host, 0
    else:
        host, port = _normalize(str(endpoint))

    if port <= 0 or not is_port_available(host, port):
        last_error: Optional[BaseException] = None
        try:
            infos = socket.getaddrinfo(
                host,
                0,
                socket.AF_UNSPEC,
                socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise RuntimeError(f"unable to resolve host for endpoint: {host}") from exc
        for family, socktype, proto, _, sockaddr in infos:
            with contextlib.closing(socket.socket(family, socktype, proto)) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(sockaddr)
                    port = int(sock.getsockname()[1])
                    break
                except OSError as exc:
                    last_error = exc
                    continue
        else:
            if last_error is not None:
                raise RuntimeError(
                    f"unable to identify an available port for host: {host}"
                ) from last_error
            raise RuntimeError(
                f"unable to identify an available port for host: {host}"
            )

    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        addr = None
    if addr is not None and addr.version == 6:
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def get_world_size(device: Optional[torch.device] = None) -> int:
    try:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
    except Exception:
        pass

    dev = device
    if dev is None:
        with contextlib.suppress(Exception):
            dev = get_device()
    if dev is None:
        dev = torch.device("cpu")
    if dev.type == "cuda":
        try:
            return int(torch.cuda.device_count())
        except Exception:
            return 1
    elif dev.type == "xpu":
        xpu = getattr(torch, "xpu", None)
        if xpu is not None:
            with contextlib.suppress(Exception):
                count = int(xpu.device_count())
                if count > 0:
                    return count
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, 4))


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
    for token in names:
        name = str(token).strip().upper()
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
        major, _ = torch.cuda.get_device_capability(
            torch.cuda.current_device()
        )
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
    device: Optional[Union[torch.device, str]] = None
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
    device: Optional[Union[torch.device, str]] = None
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


def optimal_procs() -> dict:
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return {"nproc_per_node": n_gpu or 1, "device": "cuda" if n_gpu else "cpu"}


def cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


def optimal_threads() -> dict:
    n_cpu = cpu_count()
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    intra = max(1, min(n_cpu, int(round(0.8 * n_cpu))))
    inter = max(1, min(4, int(math.sqrt(intra))))
    workers = (
        max(2, min(8 * n_gpu, n_cpu // max(1, n_gpu)))
        if n_gpu > 0
        else max(2, min(8, n_cpu // 2))
    )
    return {
        "intraop": intra,
        "interop": inter,
        "dataloader_workers": workers,
        "prefetch_factor": 2,
        "pin_memory": bool(n_gpu > 0),
    }


def optimize_threads() -> dict:
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
