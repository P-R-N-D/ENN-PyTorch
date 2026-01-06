# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import inspect
import ipaddress
import itertools
import os
import socket
import warnings
from contextlib import AbstractContextManager
from functools import lru_cache
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Iterable, TypeAlias

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

from .casting import env_bool, env_int
from .system import get_device, get_num_accelerators, process_cpu_count

fully_shard = None

try:
    from torch.distributed._composable.fsdp import fully_shard
except ImportError:
    with contextlib.suppress(ImportError):
        from torch.distributed.fsdp import fully_shard

try:
    from torch.distributed.algorithms.join import Join as _TorchJoin
except ImportError:
    _TorchJoin = None

if TYPE_CHECKING:
    from torch.distributed._composable.fsdp import FSDPModule
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
else:
    FSDP = object
    FSDPModule = object

try:
    from torch.distributed.tensor import DTensor as _DTensor
except Exception:
    try:
        from torch.distributed._tensor import DTensor as _DTensor
    except Exception:
        _DTensor = None


JoinType: TypeAlias = type[AbstractContextManager[None]] | None
Join: JoinType = _TorchJoin
JoinableModel: TypeAlias = DDP | FSDP | FSDPModule


def _unshard_fsdp_module(module: torch.nn.Module) -> None:
    with contextlib.suppress(Exception):
        if not (dist.is_available() and dist.is_initialized()):
            return
    unshard = getattr(module, "unshard", None)
    if not callable(unshard):
        return
    with contextlib.suppress(Exception):
        p0 = next(module.parameters(recurse=False), None)
        if p0 is not None:
            if _DTensor is not None:
                if not (isinstance(p0, _DTensor) or isinstance(getattr(p0, "data", None), _DTensor)):
                    return
            else:
                tname = type(p0).__name__
                dname = type(getattr(p0, "data", SimpleNamespace())).__name__
                if tname != "DTensor" and dname != "DTensor":
                    return
    with contextlib.suppress(Exception):
        handle = unshard(async_op=True)
        if handle is None:
            return
        wait = getattr(handle, "wait", None)
        if callable(wait):
            wait()


def _strip_ip_expr(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _strip_ipv6_expr(value: str) -> tuple[str, bool, str | None]:
    stripped = value.strip()
    bracketed = False
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1].strip()
        bracketed = True
    zone: str | None = None
    if "%" in stripped:
        base, _, rest = stripped.partition("%")
        stripped = base.strip()
        rest = rest.strip()
        zone = rest or None
    return stripped, bracketed, zone


def _looks_like_ip_literal(value: str) -> bool:
    if not value:
        return False
    base, _, _zone = _strip_ipv6_expr(value)
    if not base:
        return False
    try:
        ipaddress.ip_address(base)
        return True
    except ValueError:
        return False


def _canonize_ip_expr(
    value: Any,
    *args: Any,
    allow_loopback: bool = False,
    allow_link_local: bool = False,
    **kwargs: Any,
) -> str | None:
    candidate_text = _strip_ip_expr(value)
    if not candidate_text:
        return None
    base, _bracketed, zone = _strip_ipv6_expr(candidate_text)
    if not base:
        return None
    try:
        parsed_address = ipaddress.ip_address(base)
    except ValueError:
        return None
    if not allow_loopback and (parsed_address.is_unspecified or parsed_address.is_loopback):
        return None
    if parsed_address.is_link_local and not allow_link_local:
        return None
    if parsed_address.version == 6 and parsed_address.is_link_local and zone:
        return f"{parsed_address.compressed}%{zone}"
    return parsed_address.compressed


def _format_endpoint(host: str, port: int) -> str:
    host = _strip_ip_expr(host) or "127.0.0.1"
    port = int(port)
    base, bracketed, _zone = _strip_ipv6_expr(host)
    is_ipv6 = False
    try:
        is_ipv6 = ipaddress.ip_address(base).version == 6
    except ValueError:
        is_ipv6 = False
    if is_ipv6 and not bracketed:
        host = f"[{host}]"
    return f"{host}:{port}"


def _parse_endpoint(endpoint: str) -> tuple[str, int]:
    text = _strip_ip_expr(endpoint)
    if not text:
        return "", 0
    if text.startswith("["):
        inner, sep, rest = text[1:].partition("]")
        host_part = inner.strip()
        port_part = ""
        if sep and rest.startswith(":"):
            port_part = rest[1:].strip()
        port = 0
        if port_part.isdigit():
            with contextlib.suppress(ValueError):
                port = int(port_part)
        if port <= 0 or port > 65535:
            port = 0
        host = host_part
        return host, port
    if _looks_like_ip_literal(text):
        return text, 0
    left, sep, right = text.rpartition(":")
    if sep and right.strip().isdigit():
        port = 0
        with contextlib.suppress(ValueError):
            port = int(right.strip())
        if 1 <= port <= 65535:
            host = left.strip()
            if host:
                return host, port
    return text, 0


def _canonize_host_expr(endpoint_str: str, default_host: str, *args: Any, allow_link_local: bool) -> tuple[str, int]:
    endpoint = _strip_ip_expr(endpoint_str)
    if not endpoint:
        return default_host, 0
    host_raw, port = _parse_endpoint(endpoint)
    host_raw = _strip_ip_expr(host_raw) or default_host
    literal_host = _canonize_ip_expr(host_raw, allow_loopback=True, allow_link_local=allow_link_local)
    if literal_host:
        host = literal_host
    else:
        host = default_host if _looks_like_ip_literal(host_raw) else host_raw
    if port <= 0 or port > 65535:
        port = 0
    return host, port


def _has_join_hook(obj: Any | None) -> bool:
    if obj is None:
        return False
    return getattr(obj, "join_hook", None) is not None


def _get_device_id(device: Optional[torch.device]) -> Optional[Iterable[int]]:
    if device is None:
        return None
    dev_type = getattr(device, "type", None)
    if dev_type not in {"cuda", "xpu"}:
        return None
    index = getattr(device, "index", None)
    if index is None:
        return None
    return [int(index)]


def _safe_getaddrinfo(host: str) -> list[tuple[Any, ...]]:
    try:
        return socket.getaddrinfo(host, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return []
    except Exception:
        with contextlib.suppress(Exception):
            return socket.getaddrinfo(host, None)
    return []


def _get_preferred_ip_cached(
    hostname: str | None,
    prefer_ipv6: bool,
    allow_loopback: bool,
    allow_link_local: bool,
) -> str:
    names: list[str] = []
    if hostname:
        candidate = hostname.strip()
        if candidate:
            names.append(candidate)
    with contextlib.suppress(Exception):
        system_host = socket.gethostname()
        if system_host:
            names.append(system_host)
    if allow_loopback:
        names.append("localhost")
    buckets: dict[tuple[str, int], list[str]] = {
        ("global", 4): [],
        ("global", 6): [],
        ("link_local", 4): [],
        ("link_local", 6): [],
        ("loopback", 4): [],
        ("loopback", 6): [],
    }
    seen: set[tuple[int, str]] = set()
    for name in names:
        if not name:
            continue
        for info in _safe_getaddrinfo(name):
            try:
                sockaddr = info[4]
            except Exception:
                sockaddr = None
            if not sockaddr:
                continue
            addr_text = sockaddr[0]
            canon = _canonize_ip_expr(addr_text, allow_loopback=True, allow_link_local=allow_link_local)
            if not canon:
                continue
            base, _, _zone = _strip_ipv6_expr(canon)
            try:
                parsed = ipaddress.ip_address(base)
            except ValueError:
                continue
            if parsed.is_unspecified:
                continue
            key = (parsed.version, canon)
            if key in seen:
                continue
            seen.add(key)
            if parsed.is_loopback:
                bucket = "loopback"
            elif parsed.is_link_local:
                bucket = "link_local"
            else:
                bucket = "global"
            buckets[(bucket, parsed.version)].append(canon)
    if prefer_ipv6:
        order: tuple[tuple[str, int], ...] = (
            ("global", 6),
            ("global", 4),
            ("link_local", 6),
            ("link_local", 4),
            ("loopback", 6),
            ("loopback", 4),
        )
    else:
        order = (
            ("global", 4),
            ("global", 6),
            ("link_local", 4),
            ("link_local", 6),
            ("loopback", 4),
            ("loopback", 6),
        )
    for bucket_key in order:
        bucket_name, _ver = bucket_key
        if bucket_name == "loopback" and not allow_loopback:
            continue
        if bucket_name == "link_local" and not allow_link_local:
            continue
        bucket = buckets.get(bucket_key) or []
        if bucket:
            return bucket[0]
    if allow_loopback:
        return "::1" if prefer_ipv6 else "127.0.0.1"
    return "::" if prefer_ipv6 else "0.0.0.0"


def _get_default_process_group() -> Any:
    try:
        return dist.group.WORLD
    except Exception:
        pass
    try:
        return dist.distributed_c10d._get_default_group()
    except Exception:
        return None


def _ddp_supported_params() -> set[str]:
    try:
        sig = inspect.signature(DDP.__init__)
        return set(sig.parameters.keys())
    except Exception:
        return set()


def _fully_shard_supported_params() -> set[str]:
    if fully_shard is None:
        return set()
    try:
        sig = inspect.signature(fully_shard)
        return set(sig.parameters.keys())
    except Exception:
        return set()


def resolve_ip_expr(
    host: Any,
    *args: Any,
    allow_loopback: bool = False,
    prefer_ipv6: bool | None = None,
    allow_link_local: bool | None = None,
    **kwargs: Any,
) -> str | None:
    host_text = _strip_ip_expr(host)
    if not host_text:
        return None
    if allow_link_local is None:
        allow_link_local = env_bool("STNET_ALLOW_LINK_LOCAL", False)
    literal = _canonize_ip_expr(host_text, allow_loopback=allow_loopback, allow_link_local=allow_link_local)
    if literal:
        return literal
    stripped_text, _bracketed, zone = _strip_ipv6_expr(host_text)
    if not stripped_text:
        return None
    addrinfo = _safe_getaddrinfo(stripped_text)
    if not addrinfo:
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
        literal_addr = _canonize_ip_expr(address_text, allow_loopback=allow_loopback, allow_link_local=allow_link_local)
        if literal_addr:
            base, _, _ = _strip_ipv6_expr(literal_addr)
            try:
                ip_version = ipaddress.ip_address(base).version
            except ValueError:
                continue
            results.setdefault(ip_version, []).append(literal_addr)
    for version in preferred_versions:
        options = results.get(version) or []
        if options:
            return options[0]
    if zone and allow_link_local:
        for version in preferred_versions:
            for addr in results.get(version) or []:
                base, _, _ = _strip_ipv6_expr(addr)
                try:
                    ip = ipaddress.ip_address(base)
                except ValueError:
                    continue
                if ip.version == 6 and ip.is_link_local:
                    return f"{ip.compressed}%{zone}"
    return None


def validate_ip_expr(
    host: Any,
    *args: Any,
    fallback: Any | None = None,
    default: str = "127.0.0.1",
    allow_loopback: bool = False,
    allow_hostname: bool = True,
    allow_link_local: bool | None = None,
    **kwargs: Any,
) -> str:
    if allow_link_local is None:
        allow_link_local = env_bool("STNET_ALLOW_LINK_LOCAL", False)
    host_text = _strip_ip_expr(host)
    if host_text:
        if _looks_like_ip_literal(host_text):
            literal = _canonize_ip_expr(host_text, allow_loopback=allow_loopback, allow_link_local=allow_link_local)
            if literal:
                return literal
        elif allow_hostname:
            return host_text
    fb_text = _strip_ip_expr(fallback)
    if fb_text:
        if _looks_like_ip_literal(fb_text):
            literal_fb = _canonize_ip_expr(fb_text, allow_loopback=allow_loopback, allow_link_local=allow_link_local)
            if literal_fb:
                return literal_fb
        elif allow_hostname:
            return fb_text
    return _canonize_ip_expr(default, allow_loopback=True, allow_link_local=True) or default


def is_port_available(host: str, port: int, *args: Any, allow_link_local: bool | None = None) -> bool:
    if port <= 0 or port > 65535:
        return False
    if allow_link_local is None:
        allow_link_local = env_bool("STNET_ALLOW_LINK_LOCAL", False)
    literal_host = _canonize_ip_expr(host, allow_loopback=True, allow_link_local=allow_link_local)
    if not literal_host:
        literal_host = resolve_ip_expr(host, allow_loopback=True, prefer_ipv6=True, allow_link_local=allow_link_local)
    if not literal_host:
        return False
    base, _, _zone = _strip_ipv6_expr(literal_host)
    try:
        parsed = ipaddress.ip_address(base)
    except ValueError:
        return False
    bind_host = literal_host
    if parsed.version == 6:
        family = socket.AF_INET6
        bind_addr: tuple[Any, ...] = (bind_host, port, 0, 0)
    else:
        family = socket.AF_INET
        bind_addr = (bind_host, port)
    try:
        with contextlib.closing(socket.socket(family, socket.SOCK_STREAM)) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6 and hasattr(socket, "IPPROTO_IPV6") and hasattr(socket, "IPV6_V6ONLY"):
                with contextlib.suppress(OSError):
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind(bind_addr)
            return True
    except OSError:
        return False


def get_available_host(
    endpoint: Optional[str],
    *args: Any,
    default_host: str = "127.0.0.1",
    allow_link_local: bool | None = None,
    **kwargs: Any,
) -> str:
    if allow_link_local is None:
        allow_link_local = env_bool("STNET_ALLOW_LINK_LOCAL", False)
    if endpoint:
        normalized = _strip_ip_expr(endpoint)
        if normalized:
            host, port = _canonize_host_expr(normalized, default_host, allow_link_local=allow_link_local)
            if port > 0:
                return _format_endpoint(host, port)
    literal_default = _canonize_ip_expr(default_host, allow_loopback=True, allow_link_local=True)
    host_candidate = literal_default or _strip_ip_expr(default_host) or "127.0.0.1"
    host = _canonize_ip_expr(host_candidate, allow_loopback=True, allow_link_local=allow_link_local)
    if not host:
        host = resolve_ip_expr(host_candidate, allow_loopback=True, prefer_ipv6=True, allow_link_local=allow_link_local)
    if not host:
        host = "127.0.0.1"
    base, _, _zone = _strip_ipv6_expr(host)
    try:
        parsed_host = ipaddress.ip_address(base)
    except ValueError:
        parsed_host = None
    if parsed_host and parsed_host.version == 6:
        family = socket.AF_INET6
        bind_addr: tuple[Any, ...] = (host, 0, 0, 0)
    else:
        family = socket.AF_INET
        bind_addr = (host, 0)
    selected_port = 0
    try:
        with contextlib.closing(socket.socket(family, socket.SOCK_STREAM)) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6 and hasattr(socket, "IPPROTO_IPV6") and hasattr(socket, "IPV6_V6ONLY"):
                with contextlib.suppress(OSError):
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind(bind_addr)
            selected_port = int(sock.getsockname()[1])
    except OSError:
        selected_port = 0
    return _format_endpoint(host, selected_port)


def supported_ip_ver(*args: Any, allow_loopback: bool = True, **kwargs: Any) -> tuple[bool, bool]:
    ipv4_ok = False
    ipv6_ok = False
    ipv4_host = "127.0.0.1" if allow_loopback else "0.0.0.0"
    try:
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock4:
            sock4.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock4.bind((ipv4_host, 0))
            ipv4_ok = True
    except OSError:
        pass
    if getattr(socket, "has_ipv6", False):
        ipv6_host = "::1" if allow_loopback else "::"
        try:
            with contextlib.closing(socket.socket(socket.AF_INET6, socket.SOCK_STREAM)) as sock6:
                if hasattr(socket, "IPPROTO_IPV6") and hasattr(socket, "IPV6_V6ONLY"):
                    with contextlib.suppress(OSError):
                        sock6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                sock6.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock6.bind((ipv6_host, 0, 0, 0))
                ipv6_ok = True
        except OSError:
            pass
    return ipv4_ok, ipv6_ok


def get_preferred_ip(
    hostname: Optional[str] = None,
    *args: Any,
    prefer_ipv6: bool = True,
    allow_loopback: bool = True,
    allow_link_local: bool | None = None,
    **kwargs: Any,
) -> str:
    if allow_link_local is None:
        allow_link_local = env_bool("STNET_ALLOW_LINK_LOCAL", False)
    hn = None if hostname is None else str(hostname)
    return _get_preferred_ip_cached(hn, bool(prefer_ipv6), bool(allow_loopback), bool(allow_link_local))


def initialize_master_addr(
    endpoint: Optional[str],
    *args: Any,
    prefer_ipv6: bool = True,
    allow_loopback: bool = True,
    allow_link_local: bool | None = None,
    **kwargs: Any,
) -> tuple[str, int]:
    if allow_link_local is None:
        allow_link_local = env_bool("STNET_ALLOW_LINK_LOCAL", False)
    default_host = get_preferred_ip(
        allow_loopback=allow_loopback,
        prefer_ipv6=prefer_ipv6,
        allow_link_local=allow_link_local,
    )
    default_host = default_host or ("::1" if prefer_ipv6 else "127.0.0.1")
    normalized = _strip_ip_expr(endpoint) if endpoint is not None else ""
    if normalized:
        host, port = _canonize_host_expr(normalized, default_host, allow_link_local=allow_link_local)
    else:
        host, port = (default_host, 0)
    host = _strip_ip_expr(host) or default_host
    if host in {"", "0.0.0.0", "::"}:
        host = default_host
    if _looks_like_ip_literal(host):
        literal = _canonize_ip_expr(host, allow_loopback=allow_loopback, allow_link_local=allow_link_local)
        master_addr = literal or default_host
    else:
        master_addr = host
    os.environ.setdefault("MASTER_ADDR", master_addr)
    if port > 0:
        os.environ.setdefault("MASTER_PORT", str(int(port)))
    return master_addr, int(port)


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
    match getattr(dev, "type", "cpu"):
        case "cuda" | "xpu" | "mps":
            with contextlib.suppress(Exception):
                count = int(get_num_accelerators(str(getattr(dev, "type", "cpu") or "cpu")))
                if count > 0:
                    return count
            return 1
        case _:
            ncpu = process_cpu_count()
            return max(1, min(int(ncpu), 4))


@contextlib.contextmanager
def no_sync(
    model: torch.nn.Module,
    *args: Any,
    enable: bool = True,
) -> AbstractContextManager[None]:
    if not enable:
        yield
        return
    ctx: AbstractContextManager[None] | None = None
    try:
        method = getattr(model, "no_sync", None)
        if callable(method):
            ctx = method()
    except Exception:
        ctx = None
    if ctx is None:
        yield
        return
    with ctx:
        yield


def joining(
    model: JoinableModel,
    optimizer: Optimizer | None = None,
) -> AbstractContextManager[None]:
    if Join is None:
        return contextlib.nullcontext()
    joinables = tuple(obj for obj in (model, optimizer) if _has_join_hook(obj))
    if not joinables:
        return contextlib.nullcontext()
    return Join(joinables, throw_on_early_termination=True)


def is_distributed() -> bool:
    try:
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def distributed_barrier(device: Optional[torch.device] = None) -> None:
    if not is_distributed():
        return
    try:
        dist.barrier(device_ids=_get_device_id(device))
    except TypeError:
        dist.barrier()


def distributed_broadcast(
    module: torch.nn.Module,
    *args: Any,
    src: int = 0,
    coalesce_mb: int | None = None,
    strict: bool | None = None,
    **kwargs: Any,
) -> None:
    if not is_distributed():
        return
    if coalesce_mb is None:
        coalesce_mb = env_int("STNET_BROADCAST_COALESCE_MB", 25)
    if strict is None:
        strict = env_bool("STNET_DISTRIBUTED_STRICT", False)
    tensors: list[torch.Tensor] = []
    seen: set[int] = set()
    for t in itertools.chain(module.buffers(recurse=True), module.parameters(recurse=True)):
        if not isinstance(t, torch.Tensor):
            continue
        if getattr(t, "is_meta", False):
            continue
        if t.numel() == 0:
            continue
        tid = id(t)
        if tid in seen:
            continue
        seen.add(tid)
        tensors.append(t)
    with contextlib.suppress(Exception):
        from torch.distributed.tensor import DTensor

        tensors = [t for t in tensors if not isinstance(t, DTensor)]
    if not tensors:
        return
    buffer_size_bytes = int(max(1, coalesce_mb)) * 1024 * 1024
    pg = _get_default_process_group()
    with torch.no_grad():
        coalesced = getattr(dist, "_broadcast_coalesced", None)
        if callable(coalesced) and pg is not None:
            try:
                coalesced(pg, tensors, buffer_size_bytes, src)
                return
            except Exception as e:
                if strict:
                    raise
                warnings.warn(
                    f"distributed_broadcast: coalesced broadcast failed ({type(e).__name__}: {e}); "
                    "falling back to per-tensor broadcast."
                )
        try:
            for t in tensors:
                dist.broadcast(t, src=src)
        except Exception as e:
            if strict:
                raise
            warnings.warn(f"distributed_broadcast: per-tensor broadcast failed ({type(e).__name__}: {e}).")


def distributed_sync(
    module: torch.nn.Module,
    device: Optional[torch.device] = None,
    src: int = 0,
) -> None:
    if not is_distributed():
        return
    _m = module.module if hasattr(module, "module") else module
    distributed_broadcast(_m, src=src)
    distributed_barrier(device)


def to_ddp(
    module: torch.nn.Module,
    *args: Any,
    device: torch.device,
    **kwargs: Any,
) -> torch.nn.Module:
    module = module.to(device)
    if isinstance(module, DDP):
        return module
    device_ids = _get_device_id(device)
    ddp_kwargs: dict[str, Any] = {
        "broadcast_buffers": True,
        "find_unused_parameters": False,
    }
    if device_ids is not None:
        ddp_kwargs["device_ids"] = list(device_ids)
    params = _ddp_supported_params()
    bucket_mb = env_int("STNET_DDP_BUCKET_MB", 25)
    if "bucket_cap_mb" in params:
        ddp_kwargs.setdefault("bucket_cap_mb", bucket_mb)
    if "gradient_as_bucket_view" in params:
        ddp_kwargs.setdefault("gradient_as_bucket_view", True)
    if "static_graph" in params:
        ddp_kwargs.setdefault("static_graph", False)
    ddp_kwargs.update(kwargs)
    return DDP(module, **ddp_kwargs)


def to_fsdp(
    module: torch.nn.Module,
    *args: Any,
    mesh: Any | None,
    mp_policy: Any | None = None,
    reshard_after_forward: bool = False,
    sync_module_states: bool = True,
    **user_kwargs: Any,
) -> torch.nn.Module:
    if fully_shard is None:
        raise RuntimeError(
            "Composable FSDP is not available in this PyTorch build (missing fully_shard). "
            "Install a PyTorch version that provides torch.distributed._composable.fsdp." 
        )
    params = _fully_shard_supported_params()
    fsdp_kwargs: dict[str, Any] = dict(user_kwargs)
    if "forward_prefetch" in params and "forward_prefetch" not in fsdp_kwargs:
        fsdp_kwargs["forward_prefetch"] = env_bool("STNET_FSDP_FWD_PREFETCH", True)
    if "limit_all_gathers" in params and "limit_all_gathers" not in fsdp_kwargs:
        fsdp_kwargs["limit_all_gathers"] = env_bool("STNET_FSDP_LIMIT_AG", True)
    if "use_orig_params" in params and "use_orig_params" not in fsdp_kwargs:
        fsdp_kwargs["use_orig_params"] = env_bool("STNET_FSDP_USE_ORIG_PARAMS", True)
    if mesh is not None:
        if "mesh" in params and "mesh" not in fsdp_kwargs:
            fsdp_kwargs["mesh"] = mesh
        elif "process_group" in params and "process_group" not in fsdp_kwargs:
            fsdp_kwargs["process_group"] = mesh
    if mp_policy is not None and "mp_policy" in params and "mp_policy" not in fsdp_kwargs:
        fsdp_kwargs["mp_policy"] = mp_policy
    if "reshard_after_forward" in params and "reshard_after_forward" not in fsdp_kwargs:
        fsdp_kwargs["reshard_after_forward"] = reshard_after_forward
    if "sync_module_states" in params and "sync_module_states" not in fsdp_kwargs:
        fsdp_kwargs["sync_module_states"] = sync_module_states
    sharded = fully_shard(module, *args, **fsdp_kwargs)
    with contextlib.suppress(AttributeError):
        sharded.set_requires_gradient_sync(True)
    try:
        from torch.distributed.fsdp import register_fsdp_forward_method as _reg_fsdp_forward_method
    except Exception:
        _reg_fsdp_forward_method = None
    if callable(_reg_fsdp_forward_method):
        for _name in ("forward", "decode", "predict"):
            if hasattr(sharded, _name):
                with contextlib.suppress(Exception):
                    _reg_fsdp_forward_method(sharded, _name)
    return sharded
