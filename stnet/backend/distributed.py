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

from .environment import System

try:  # pragma: no cover - optional dependency
    from torch.distributed.algorithms.join import Join as _TorchJoin
except ImportError:  # pragma: no cover - optional dependency
    _TorchJoin = None

Join: type[AbstractContextManager[None]] | None = _TorchJoin

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from torch.distributed._composable.fsdp import FSDPModule
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
else:  # pragma: no cover - fallback when optional deps missing
    FSDP = object
    FSDPModule = object

JoinableModel: TypeAlias = Union["DDP", "FSDP", "FSDPModule"]


class Network:
    @staticmethod
    def coerce_host(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _strip_host_tokens(value: str) -> tuple[str, bool, bool]:
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

    @staticmethod
    def normalize_ip_literal(
        value: Any,
        *,
        allow_loopback: bool = False,
    ) -> str | None:
        candidate_text = Network.coerce_host(value)
        if not candidate_text:
            return None
        stripped_text, _, _ = Network._strip_host_tokens(candidate_text)
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

    @staticmethod
    def resolve_host_ip(
        host: Any,
        *,
        allow_loopback: bool = False,
        prefer_ipv6: bool | None = None,
    ) -> str | None:
        host_text = Network.coerce_host(host)
        if not host_text:
            return None
        literal = Network.normalize_ip_literal(host_text, allow_loopback=allow_loopback)
        if literal:
            return literal
        stripped_text, _, _ = Network._strip_host_tokens(host_text)
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
            literal_addr = Network.normalize_ip_literal(
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

    @staticmethod
    def format_endpoint_host(
        host: Any,
        *,
        fallback: Any | None = None,
        default: str = "127.0.0.1",
        allow_loopback: bool = False,
        allow_hostname: bool = True,
    ) -> str:
        if allow_hostname:
            literal = Network.coerce_host(host)
        else:
            literal = Network.normalize_ip_literal(host, allow_loopback=allow_loopback)
        if literal:
            return literal
        literal_fallback = Network.normalize_ip_literal(
            fallback,
            allow_loopback=allow_loopback,
        )
        if literal_fallback:
            return literal_fallback
        return Network.normalize_ip_literal(default, allow_loopback=True) or default

    @staticmethod
    def normalize_endpoint(endpoint_str: str, default_host: str) -> Tuple[str, int]:
        endpoint = Network.coerce_host(endpoint_str)
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
        literal_host = Network.normalize_ip_literal(host, allow_loopback=True)
        if not literal_host:
            literal_host = default_host
        try:
            parsed_port = int(port.strip()) if port else 0
        except (TypeError, ValueError):
            parsed_port = 0
        if parsed_port <= 0 or parsed_port > 65535:
            parsed_port = 0
        return literal_host, parsed_port

    @staticmethod
    def is_port_available(host: str, port: int) -> bool:
        literal_host = Network.normalize_ip_literal(host, allow_loopback=True)
        if not literal_host:
            return False
        try:
            with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((literal_host, port))
                return True
        except OSError:
            return False

    @staticmethod
    def get_available_addr(
        endpoint: Optional[str],
        *,
        default_host: str = "127.0.0.1",
    ) -> str:
        if endpoint:
            normalized = Network.coerce_host(endpoint)
            if normalized:
                host, port = Network.normalize_endpoint(normalized, default_host)
                if port > 0:
                    return f"{host}:{port}"
        literal_default = Network.normalize_ip_literal(default_host, allow_loopback=True)
        host = literal_default or Network.coerce_host(default_host) or "127.0.0.1"
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

    @staticmethod
    def probe_stack_support(*, allow_loopback: bool = True) -> tuple[bool, bool]:
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

    @staticmethod
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


class Distributed:
    @staticmethod
    def initialize_master_addr(
        endpoint: Optional[str],
        *,
        prefer_ipv6: bool = True,
        allow_loopback: bool = True,
    ) -> tuple[str, int]:
        default_host = Network.get_preferred_ip(
            allow_loopback=allow_loopback, prefer_ipv6=prefer_ipv6
        )
        default_host = default_host or ("::1" if prefer_ipv6 else "127.0.0.1")
        normalized = Network.coerce_host(endpoint) if endpoint is not None else None
        if normalized:
            host, port = Network.normalize_endpoint(normalized, default_host)
        else:
            host, port = (default_host, 0)
        host = host.strip()
        if host in {"", "0.0.0.0", "::"}:
            host = default_host
        literal = Network.normalize_ip_literal(host, allow_loopback=allow_loopback)
        master_addr = literal or Network.coerce_host(host) or default_host
        os.environ.setdefault("MASTER_ADDR", master_addr)
        if port > 0:
            os.environ.setdefault("MASTER_PORT", str(port))
        return master_addr, int(port)

    @staticmethod
    def get_world_size(device: Optional[torch.device] = None) -> int:
        try:
            if dist.is_available() and dist.is_initialized():
                return int(dist.get_world_size())
        except Exception:
            pass

        dev = device
        if dev is None:
            with contextlib.suppress(Exception):
                dev = System.get_device()
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


def get_available_addr(
    endpoint: Optional[str],
    *,
    default_host: str = "127.0.0.1",
) -> str:
    return Network.get_available_addr(endpoint, default_host=default_host)


def probe_stack_support(*, allow_loopback: bool = True) -> tuple[bool, bool]:
    return Network.probe_stack_support(allow_loopback=allow_loopback)


def get_preferred_ip(
    hostname: Optional[str] = None,
    *,
    prefer_ipv6: bool = True,
    allow_loopback: bool = True,
) -> str:
    return Network.get_preferred_ip(
        hostname,
        prefer_ipv6=prefer_ipv6,
        allow_loopback=allow_loopback,
    )


def initialize_master_addr(
    endpoint: Optional[str],
    *,
    prefer_ipv6: bool = True,
    allow_loopback: bool = True,
) -> tuple[str, int]:
    return Distributed.initialize_master_addr(
        endpoint,
        prefer_ipv6=prefer_ipv6,
        allow_loopback=allow_loopback,
    )


def get_world_size(device: Optional[torch.device] = None) -> int:
    return Distributed.get_world_size(device=device)


@contextlib.contextmanager
def no_synchronization(
    model: torch.nn.Module,
    *,
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


def _has_join_hook(obj: Any | None) -> bool:
    if obj is None:
        return False
    return getattr(obj, "join_hook", None) is not None


def joining(
    model: JoinableModel,
    optimizer: Optimizer | None = None,
) -> AbstractContextManager[None]:
    if Join is None:
        return contextlib.nullcontext()

    joinables = tuple(obj for obj in (model, optimizer) if _has_join_hook(obj))
    if not joinables:
        return contextlib.nullcontext()

    return Join(joinables, throw_on_early_termination=True)  # type: ignore[arg-type]


def is_dist_avail_and_initialized() -> bool:
    """Return ``True`` when torch.distributed is available and initialized."""

    try:
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def _device_ids_from(device: Optional[torch.device]) -> Optional[Iterable[int]]:
    if device is None:
        return None
    dev_type = getattr(device, "type", None)
    if dev_type not in {"cuda", "xpu"}:
        return None
    index = getattr(device, "index", None)
    if index is None:
        return None
    return [int(index)]


def distributed_barrier(device: Optional[torch.device] = None) -> None:
    """Synchronize all ranks if the process group is initialized."""

    if not is_dist_avail_and_initialized():
        return
    try:
        dist.barrier(device_ids=_device_ids_from(device))
    except TypeError:
        # ``device_ids`` keyword is not supported by all backends.
        dist.barrier()


def broadcast_model_states(
    module: torch.nn.Module,
    *,
    src: int = 0,
) -> None:
    """Broadcast parameters and buffers from ``src`` to all other ranks.

    The helper is aware of DTensor instances and re-wraps the broadcasted
    local shard to preserve placements.
    """

    if not is_dist_avail_and_initialized():
        return

    try:
        from torch.distributed._tensor import DTensor  # type: ignore
    except Exception:  # pragma: no cover - DTensor optional
        DTensor = tuple()  # type: ignore[assignment]

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
            if isinstance(data, DTensor):  # type: ignore[arg-type]
                local = data.to_local()
                dist.broadcast(local, src=src)
                param.data = type(data).from_local(  # type: ignore[assignment]
                    local,
                    device_mesh=data.device_mesh,
                    placements=tuple(data.placements),
                    run_check=False,
                )
            else:
                dist.broadcast(data, src=src)
        except Exception:
            continue


def wrap_ddp_if_needed(
    module: torch.nn.Module,
    *,
    device: torch.device,
) -> torch.nn.Module:
    """Wrap ``module`` with :class:`~torch.nn.parallel.DistributedDataParallel`.

    The wrapper is only applied when the default process group is initialized.
    The module is always moved to ``device``.
    """

    module = module.to(device)
    if not is_dist_avail_and_initialized():
        return module
    if isinstance(module, DDP):
        return module

    device_ids = _device_ids_from(device)
    ddp_kwargs = {
        "broadcast_buffers": True,
        "find_unused_parameters": False,
    }
    if device_ids is not None:
        ddp_kwargs["device_ids"] = list(device_ids)
    return DDP(module, **ddp_kwargs)


def wrap_fsdp_module(
    module: torch.nn.Module,
    *,
    mesh: Any | None,
    mp_policy: Any | None = None,
    reshard_after_forward: bool = False,
    sync_module_states: bool = True,
    ignored_params: Sequence[torch.nn.Parameter] | None = None,
) -> torch.nn.Module:
    """Wrap ``module`` using ``torch.distributed.fsdp.fully_shard``.

    This helper ensures ``requires_gradient_sync`` is enabled on the returned
    FSDP wrapper so gradient synchronization happens by default.
    Additional keyword arguments mirror those accepted by ``fully_shard``.
    """

    sig = inspect.signature(fully_shard)
    params = sig.parameters

    args = [module]
    kwargs = {}

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


__all__ = [
    "Join",
    "JoinableModel",
    "Network",
    "Distributed",
    "broadcast_model_states",
    "coerce_host",
    "distributed_barrier",
    "format_endpoint_host",
    "get_available_addr",
    "get_preferred_ip",
    "get_world_size",
    "initialize_master_addr",
    "is_dist_avail_and_initialized",
    "is_port_available",
    "joining",
    "no_synchronization",
    "normalize_endpoint",
    "normalize_ip_literal",
    "probe_stack_support",
    "resolve_host_ip",
    "wrap_ddp_if_needed",
    "wrap_fsdp_module",
]

