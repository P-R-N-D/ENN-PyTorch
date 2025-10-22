# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import importlib
import ipaddress
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
    main_dir: Path | None = None
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        main_file = getattr(main_module, "__file__", None)
        if main_file:
            with contextlib.suppress(Exception):
                main_dir = Path(main_file).resolve().parent

    for candidate in (package_dir, project_dir, main_dir):
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


class Network:
    """Unified networking helper for host sanitation, resolution, and endpoints."""

    def __init__(
        self,
        *,
        allow_loopback: bool = False,
        prefer_ipv6: bool | None = None,
        fallback: Any | None = None,
        default: str = "127.0.0.1",
        allow_hostname: bool = True,
    ) -> None:
        self.allow_loopback = allow_loopback
        self.prefer_ipv6 = prefer_ipv6
        self.fallback = fallback
        self.default = self.coerce_host(default) or "127.0.0.1"
        self.allow_hostname = allow_hostname

    @staticmethod
    def coerce_host(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def strip_host_tokens(value: str) -> tuple[str, bool, bool]:
        stripped = value.strip()
        bracketed = False
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1].strip()
            bracketed = True
        zone_removed = False
        if "%" in stripped:
            stripped = stripped.split("%", 1)[0].strip()
            zone_removed = True
        return stripped, bracketed, zone_removed

    @classmethod
    def normalize_ip_literal(
        cls,
        value: Any,
        *,
        allow_loopback: bool = False,
    ) -> str | None:
        candidate_text = cls.coerce_host(value)
        if not candidate_text:
            return None
        stripped_text, _, _ = cls.strip_host_tokens(candidate_text)
        if not stripped_text:
            return None
        try:
            parsed_address = ipaddress.ip_address(stripped_text)
        except ValueError:
            return None
        if not allow_loopback and (
            parsed_address.is_unspecified or parsed_address.is_loopback
        ):
            return None
        return parsed_address.compressed

    def normalize(self, value: Any) -> str | None:
        return self.normalize_ip_literal(
            value,
            allow_loopback=self.allow_loopback,
        )

    @classmethod
    def resolve_host_ip(
        cls,
        host: Any,
        *,
        allow_loopback: bool = False,
        prefer_ipv6: bool | None = None,
    ) -> str | None:
        host_text = cls.coerce_host(host)
        if not host_text:
            return None
        literal = cls.normalize_ip_literal(host_text, allow_loopback=allow_loopback)
        if literal:
            return literal
        stripped_text, _, _ = cls.strip_host_tokens(host_text)
        if not stripped_text:
            return None
        addrinfo: list[tuple[Any, ...]] | None = None
        try:
            addrinfo = socket.getaddrinfo(
                stripped_text,
                None,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror:
            return None
        except Exception:
            with contextlib.suppress(Exception):
                addrinfo = socket.getaddrinfo(stripped_text, None)
        if addrinfo is None or not addrinfo:
            return None
        preferred_versions: tuple[int, ...]
        if prefer_ipv6 is None or prefer_ipv6:
            preferred_versions = (6, 4)
        else:
            preferred_versions = (4, 6)
        results: dict[int, list[str]] = {4: [], 6: []}
        for resolved in addrinfo:
            try:
                sockaddr = resolved[4]
            except Exception:
                continue
            if not sockaddr:
                continue
            address_text = sockaddr[0]
            literal_addr = cls.normalize_ip_literal(
                address_text,
                allow_loopback=allow_loopback,
            )
            if literal_addr:
                ip_version = ipaddress.ip_address(literal_addr).version
                results.setdefault(ip_version, []).append(literal_addr)
        for version in preferred_versions:
            if results.get(version):
                return results[version][0]
        for literal_addr_list in results.values():
            if literal_addr_list:
                return literal_addr_list[0]
        return None

    def resolve(self, host: Any) -> str | None:
        return self.resolve_host_ip(
            host,
            allow_loopback=self.allow_loopback,
            prefer_ipv6=self.prefer_ipv6,
        )

    @classmethod
    def format_endpoint_host(
        cls,
        host: Any,
        *,
        fallback: Any | None = None,
        default: str = "127.0.0.1",
        allow_loopback: bool = False,
        allow_hostname: bool = True,
    ) -> str:
        default_value = cls.coerce_host(default) or "127.0.0.1"
        candidates: tuple[tuple[Any | None, bool, bool], ...] = (
            (host, allow_loopback, allow_hostname),
            (fallback, allow_loopback, allow_hostname),
            (default_value, True, allow_hostname),
        )
        for candidate, loopback_ok, hostnames_ok in candidates:
            text = cls.coerce_host(candidate)
            if not text:
                continue
            stripped, bracketed, zone_removed = cls.strip_host_tokens(text)
            if not stripped:
                continue
            literal = cls.normalize_ip_literal(stripped, allow_loopback=loopback_ok)
            if literal:
                return f"[{literal}]" if ":" in literal else literal
            if (
                hostnames_ok
                and not bracketed
                and not zone_removed
                and ":" not in stripped
            ):
                return stripped

        literal_default = cls.normalize_ip_literal(default_value, allow_loopback=True)
        if literal_default:
            return f"[{literal_default}]" if ":" in literal_default else literal_default
        return default_value

    def format(self, host: Any) -> str:
        return self.format_endpoint_host(
            host,
            fallback=self.fallback,
            default=self.default,
            allow_loopback=self.allow_loopback,
            allow_hostname=self.allow_hostname,
        )

    @staticmethod
    def normalize_endpoint(endpoint_str: str, default_host: str) -> Tuple[str, int]:
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

    @classmethod
    def is_port_available(cls, host: str, port: int) -> bool:
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

    @classmethod
    def get_available_addr(
        cls,
        endpoint: Optional[str],
        *,
        default_host: str = "127.0.0.1",
    ) -> str:
        normalized_default = cls.coerce_host(default_host) or "127.0.0.1"
        if endpoint is None:
            host, port = normalized_default, 0
        else:
            host, port = cls.normalize_endpoint(str(endpoint), normalized_default)

        if port <= 0 or not cls.is_port_available(host, port):
            last_error: Optional[BaseException] = None
            try:
                infos = socket.getaddrinfo(
                    host,
                    0,
                    socket.AF_UNSPEC,
                    socket.SOCK_STREAM,
                )
            except socket.gaierror as exc:
                raise RuntimeError(
                    f"unable to resolve host for endpoint: {host}"
                ) from exc
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

    def allocate(self, endpoint: Optional[str]) -> str:
        return self.get_available_addr(endpoint, default_host=self.default)


def coerce_host(value: Any) -> str:
    return Network.coerce_host(value)


def normalize_ip_literal(value: Any, *, allow_loopback: bool = False) -> str | None:
    return Network.normalize_ip_literal(value, allow_loopback=allow_loopback)


def resolve_host_ip(
    host: Any,
    *,
    allow_loopback: bool = False,
    prefer_ipv6: bool | None = None,
) -> str | None:
    return Network.resolve_host_ip(
        host,
        allow_loopback=allow_loopback,
        prefer_ipv6=prefer_ipv6,
    )


def format_endpoint_host(
    host: Any,
    *,
    fallback: Any | None = None,
    default: str = "127.0.0.1",
    allow_loopback: bool = False,
    allow_hostname: bool = True,
) -> str:
    return Network.format_endpoint_host(
        host,
        fallback=fallback,
        default=default,
        allow_loopback=allow_loopback,
        allow_hostname=allow_hostname,
    )


def normalize_endpoint(endpoint_str: str, default_host: str) -> Tuple[str, int]:
    return Network.normalize_endpoint(endpoint_str, default_host)


def is_port_available(host: str, port: int) -> bool:
    return Network.is_port_available(host, port)


def get_available_addr(endpoint: Optional[str]) -> str:
    return Network.get_available_addr(endpoint)


def get_preferred_ip(
    hostname: Optional[str] = None,
    *,
    prefer_ipv6: bool = True,
    allow_loopback: bool = True,
) -> str:
    names: List[str] = []
    if hostname:
        candidate = str(hostname).strip()
        if candidate:
            names.append(candidate)
    try:
        system_host = socket.gethostname()
    except Exception:
        system_host = ""
    if system_host:
        names.append(system_host)
    if allow_loopback:
        names.append("localhost")

    buckets: Dict[
        Tuple[str, int],
        List[Union[ipaddress.IPv4Address, ipaddress.IPv6Address]],
    ] = {
        ("global", 4): [],
        ("global", 6): [],
        ("loopback", 4): [],
        ("loopback", 6): [],
    }
    seen: set[Tuple[int, str]] = set()

    for name in names:
        if not name:
            continue
        infos: List[Tuple[Any, ...]] = []
        try:
            infos = socket.getaddrinfo(
                name,
                None,
                socket.AF_UNSPEC,
                socket.SOCK_STREAM,
            )
        except socket.gaierror:
            continue
        except Exception:
            with contextlib.suppress(Exception):
                infos = socket.getaddrinfo(name, None)
        for info in infos:
            try:
                sockaddr = info[4]
            except Exception:
                sockaddr = None
            if not sockaddr:
                continue
            addr_text = sockaddr[0]
            base, _, _ = addr_text.partition("%")
            try:
                parsed = ipaddress.ip_address(base)
            except ValueError:
                continue
            if parsed.is_unspecified:
                continue
            key = (parsed.version, parsed.compressed)
            if key in seen:
                continue
            seen.add(key)
            bucket_key = ("loopback" if parsed.is_loopback else "global", parsed.version)
            buckets.setdefault(bucket_key, []).append(parsed)

    if prefer_ipv6:
        order: Tuple[Tuple[str, int], ...] = (
            ("global", 6),
            ("global", 4),
            ("loopback", 6),
            ("loopback", 4),
        )
    else:
        order = (
            ("global", 4),
            ("global", 6),
            ("loopback", 4),
            ("loopback", 6),
        )

    for bucket_key in order:
        if not allow_loopback and bucket_key[0] == "loopback":
            continue
        bucket = buckets.get(bucket_key) or []
        if bucket:
            return bucket[0].compressed

    fallback_ipv6_loop = "::1"
    fallback_ipv4_loop = "127.0.0.1"
    if allow_loopback:
        return fallback_ipv6_loop if prefer_ipv6 else fallback_ipv4_loop
    return "::" if prefer_ipv6 else "0.0.0.0"


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
