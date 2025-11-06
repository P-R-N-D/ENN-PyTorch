# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import inspect
import ipaddress
import os
import socket
from contextlib import AbstractContextManager
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    Union,
)

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

from .environment import get_device


def _env_flag(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return bool(default)
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


try:
    from torch.distributed.algorithms.join import Join as _TorchJoin
except ImportError:
    _TorchJoin = None

Join: type[AbstractContextManager[None]] | None = _TorchJoin

if TYPE_CHECKING:
    from torch.distributed._composable.fsdp import FSDPModule
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
else:
    FSDP = object
    FSDPModule = object

JoinableModel: TypeAlias = Union["DDP", "FSDP", "FSDPModule"]


def _strip_ip_expr(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _strip_ipv6_expr(value: str) -> tuple[str, bool, bool]:
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


def _canonize_ip_expr(
    value: Any,
    *args: Any,
    allow_loopback: bool = False,
    **kwargs: Any,
) -> str | None:
    candidate_text = _strip_ip_expr(value)
    if not candidate_text:
        return None
    stripped_text, _, _ = _strip_ipv6_expr(candidate_text)
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


def _canonize_host_expr(endpoint_str: str, default_host: str) -> Tuple[str, int]:
    endpoint = _strip_ip_expr(endpoint_str)
    if not endpoint:
        return default_host, 0
    host_port = endpoint
    if host_port.startswith("["):
        host_port = host_port.lstrip("[")
        bracketed = True
    else:
        bracketed = False
    host, sep, port = host_port.partition("]" if bracketed else ":")
    if bracketed and sep:
        _, _, port = port.partition(":")
    host = host.strip()
    literal_host = _canonize_ip_expr(host, allow_loopback=True)
    if not literal_host:
        literal_host = default_host
    try:
        parsed_port = int(port.strip()) if port else 0
    except (TypeError, ValueError):
        parsed_port = 0
    if parsed_port <= 0 or parsed_port > 65535:
        parsed_port = 0
    return literal_host, parsed_port


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


def resolve_ip_expr(
    host: Any,
    *args: Any,
    allow_loopback: bool = False,
    prefer_ipv6: bool | None = None,
    **kwargs: Any,
) -> str | None:
    host_text = _strip_ip_expr(host)
    if not host_text:
        return None
    literal = _canonize_ip_expr(host_text, allow_loopback=allow_loopback)
    if literal:
        return literal
    stripped_text, _, _ = _strip_ipv6_expr(host_text)
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
        literal_addr = _canonize_ip_expr(
            address_text,
            allow_loopback=allow_loopback,
        )
        if literal_addr:
            ip_version = ipaddress.ip_address(literal_addr).version
            results.setdefault(ip_version, []).append(literal_addr)
    for version in preferred_versions:
        options = results.get(version) or []
        if options:
            return options[0]
    return None


def validate_ip_expr(
    host: Any,
    *args: Any,
    fallback: Any | None = None,
    default: str = "127.0.0.1",
    allow_loopback: bool = False,
    allow_hostname: bool = True,
    **kwargs: Any,
) -> str:
    if allow_hostname:
        literal = _strip_ip_expr(host)
    else:
        literal = _canonize_ip_expr(host, allow_loopback=allow_loopback)
    if literal:
        return literal
    literal_fallback = _canonize_ip_expr(
        fallback,
        allow_loopback=allow_loopback,
    )
    if literal_fallback:
        return literal_fallback
    return _canonize_ip_expr(default, allow_loopback=True) or default


def is_port_available(host: str, port: int) -> bool:
    literal_host = _canonize_ip_expr(host, allow_loopback=True)
    if not literal_host:
        return False
    try:
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((literal_host, port))
            return True
    except OSError:
        return False


def get_available_host(
    endpoint: Optional[str],
    *args: Any,
    default_host: str = "127.0.0.1",
    **kwargs: Any,
) -> str:
    if endpoint:
        normalized = _strip_ip_expr(endpoint)
        if normalized:
            host, port = _canonize_host_expr(normalized, default_host)
            if port > 0:
                return f"{host}:{port}"
    literal_default = _canonize_ip_expr(default_host, allow_loopback=True)
    host = literal_default or _strip_ip_expr(default_host) or "127.0.0.1"
    selected_port = 0
    try:
        parsed_host = ipaddress.ip_address(host)
    except ValueError:
        parsed_host = None

    family = socket.AF_INET
    bind_addr: tuple[Any, ...] = (host, 0)
    if parsed_host and parsed_host.version == 6:
        family = socket.AF_INET6
        bind_addr = (host, 0, 0, 0)

    try:
        with contextlib.closing(socket.socket(family, socket.SOCK_STREAM)) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if (
                family == socket.AF_INET6
                and hasattr(socket, "IPPROTO_IPV6")
                and hasattr(socket, "IPV6_V6ONLY")
            ):
                with contextlib.suppress(OSError):
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind(bind_addr)
            selected_port = sock.getsockname()[1]
    except OSError:
        selected_port = 0

    return f"{host}:{selected_port}"


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
                        sock6.setsockopt(
                            socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1
                        )
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
    **kwargs: Any,
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


def initialize_master_addr(
    endpoint: Optional[str],
    *args: Any,
    prefer_ipv6: bool = True,
    allow_loopback: bool = True,
    **kwargs: Any,
) -> tuple[str, int]:
    default_host = get_preferred_ip(
        allow_loopback=allow_loopback, prefer_ipv6=prefer_ipv6
    )
    default_host = default_host or ("::1" if prefer_ipv6 else "127.0.0.1")
    normalized = _strip_ip_expr(endpoint) if endpoint is not None else None
    if normalized:
        host, port = _canonize_host_expr(normalized, default_host)
    else:
        host, port = (default_host, 0)
    host = host.strip()
    if host in {"", "0.0.0.0", "::"}:
        host = default_host
    literal = _canonize_ip_expr(host, allow_loopback=allow_loopback)
    master_addr = literal or _strip_ip_expr(host) or default_host
    os.environ.setdefault("MASTER_ADDR", master_addr)
    if port > 0:
        os.environ.setdefault("MASTER_PORT", str(port))
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
    if dev.type == "cuda":
        try:
            return int(torch.cuda.device_count())
        except Exception:
            return 1
    if dev.type == "xpu":
        xpu = getattr(torch, "xpu", None)
        if xpu is not None:
            with contextlib.suppress(Exception):
                count = int(xpu.device_count())
                if count > 0:
                    return count
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, 4))


@contextlib.contextmanager
def no_synchronization(
    model: torch.nn.Module,
    *args: Any,
    enable: bool = True,
) -> AbstractContextManager[None]:
    if not enable:
        yield
        return

    ctx: AbstractContextManager[None] | None = None
    try:
        no_sync = getattr(model, "no_sync", None)
        if callable(no_sync):
            ctx = no_sync()
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
    **kwargs: Any,
) -> None:
    if not is_distributed():
        return

    try:
        from torch.distributed._tensor import DTensor
    except Exception:
        DTensor = tuple()

    for buffer in module.buffers(recurse=True):
        data = getattr(buffer, "data", None)
        if not isinstance(data, torch.Tensor):
            continue
        try:
            dist.broadcast(data, src=src)
        except Exception:
            continue

    for param in module.parameters(recurse=True):
        data = getattr(param, "data", None)
        if not isinstance(data, torch.Tensor):
            continue
        try:
            if isinstance(data, DTensor):
                local = data.to_local()
                dist.broadcast(local, src=src)
                param.data = type(data).from_local(
                    local,
                    device_mesh=data.device_mesh,
                    placements=tuple(data.placements),
                    run_check=False,
                )
            else:
                dist.broadcast(data, src=src)
        except Exception:
            continue


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
    if not is_distributed():
        return module
    if isinstance(module, DDP):
        return module

    device_ids = _get_device_id(device)
    ddp_kwargs = {
        "broadcast_buffers": True,
        "find_unused_parameters": False,
    }
    if device_ids is not None:
        ddp_kwargs["device_ids"] = list(device_ids)
    try:
        sig = inspect.signature(DDP.__init__)
        params = set(sig.parameters.keys())
    except Exception:
        params = set()
    bucket_mb = _env_int("STNET_DDP_BUCKET_MB", 25)
    if "bucket_cap_mb" in params:
        ddp_kwargs["bucket_cap_mb"] = bucket_mb
    if "gradient_as_bucket_view" in params:
        ddp_kwargs["gradient_as_bucket_view"] = _env_flag("STNET_DDP_BUCKET_VIEW", True)
    if "static_graph" in params:
        ddp_kwargs["static_graph"] = _env_flag("STNET_DDP_STATIC_GRAPH", False)
    return DDP(module, **ddp_kwargs)


def to_fsdp(
    module: torch.nn.Module,
    *args: Any,
    mesh: Any | None,
    mp_policy: Any | None = None,
    reshard_after_forward: bool = False,
    sync_module_states: bool = True,
    ignored_params: Sequence[torch.nn.Parameter] | None = None,
    **kwargs: Any,
) -> torch.nn.Module:
    sig = inspect.signature(fully_shard)
    params = sig.parameters

    args = [module]
    kwargs = {}
    if "forward_prefetch" in params:
        kwargs["forward_prefetch"] = _env_flag("STNET_FSDP_FWD_PREFETCH", True)
    if "limit_all_gathers" in params:
        kwargs["limit_all_gathers"] = _env_flag("STNET_FSDP_LIMIT_AG", True)
    if "use_orig_params" in params:
        kwargs["use_orig_params"] = _env_flag("STNET_FSDP_USE_ORIG_PARAMS", True)

    if "mesh" in params:
        kwargs["mesh"] = mesh
    elif "process_group" in params and mesh is not None:
        kwargs["process_group"] = mesh

    if "mp_policy" in params and mp_policy is not None:
        kwargs["mp_policy"] = mp_policy
    if "reshard_after_forward" in params:
        kwargs["reshard_after_forward"] = reshard_after_forward
    if "sync_module_states" in params:
        kwargs["sync_module_states"] = sync_module_states
    if "ignored_params" in params and ignored_params is not None:
        kwargs["ignored_params"] = ignored_params

    sharded = fully_shard(*args, **kwargs)
    try:
        sharded.set_requires_gradient_sync(True)
    except AttributeError:
        pass
    return sharded
