# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import inspect
import ipaddress
import os
import socket
from contextlib import AbstractContextManager
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch.optim import Optimizer

from .datatypes import env_bool
from .system import CPU, get_device, get_num_accelerators

try:
    from torch.distributed._composable.fsdp import fully_shard
except ImportError:
    with contextlib.suppress(ImportError):
        from torch.distributed.fsdp import fully_shard
try:
    from torch.distributed.algorithms.join import Join as _TorchJoin
except ImportError:
    _TorchJoin = None
Join = _TorchJoin
try:
    from torch.distributed.tensor import DTensor as _DTensor
except ImportError:
    try:
        from torch.distributed._tensor import DTensor as _DTensor
    except ImportError:
        _DTensor = None


_DTENSOR_ACTIVE: bool = False
_GLOOX_GLOO_PG_CACHE: dict[tuple[int, ...], ProcessGroup] = {}


def _set_dtensor_active() -> None:
    global _DTENSOR_ACTIVE
    _DTENSOR_ACTIVE = True


def _from_hsdp_module(module: torch.nn.Module) -> None:
    if not (is_distributed() and callable(unshard := getattr(module, "unshard", None))):
        return
    try:
        if (p0 := next(module.parameters(recurse=False), None)) is not None:
            is_dt = (
                isinstance(p0, _DTensor)
                or isinstance(getattr(p0, "data", None), _DTensor)
                if _DTensor
                else (
                    type(p0).__name__ == "DTensor"
                    or type(getattr(p0, "data", None)).__name__ == "DTensor"
                )
            )
            if is_dt:
                _set_dtensor_active()
            if not is_dt:
                return
        if (handle := unshard(async_op=True)) and callable(
            wait := getattr(handle, "wait", None)
        ):
            wait()
    except:
        pass


def _coerce_ip_addr(v: Any, strip_zone: bool = True) -> str:
    s = str(v).strip() if v is not None else ""
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    if strip_zone and "%" in s:
        s = s.partition("%")[0].strip()
    return s


def _is_ip_addr(value: str) -> bool:
    try:
        ipaddress.ip_address(_coerce_ip_addr(value))
        return True
    except:
        return False


def _canonize_ip(
    value: Any, loopback: bool = False, link_local: bool = False
) -> str | None:
    s = _coerce_ip_addr(value)
    if not s:
        return None
    try:
        ip = ipaddress.ip_address(s)
        if (not loopback and (ip.is_unspecified or ip.is_loopback)) or (
            ip.is_link_local and not link_local
        ):
            return None
        return f"{ip.compressed}{'%' + value.partition('%')[2] if '%' in str(value) and ip.version == 6 else ''}"
    except:
        return None


def _format_endpoint(host: str, port: int) -> str:
    host_text = str(host).strip() if host is not None else ""
    if host_text.startswith("[") and host_text.endswith("]"):
        host_text = host_text[1:-1].strip()
    h, p = host_text or "127.0.0.1", int(port)
    try:
        h = (
            f"[{h}]"
            if ipaddress.ip_address(_coerce_ip_addr(h)).version == 6 and ":" in h
            else h
        )
    except:
        pass
    return f"{h}:{p}"


def _parse_endpoint(text: str) -> tuple[str, int]:
    text = text.strip()
    if not text:
        return "", 0
    if text.startswith("["):
        host, _, rest = text[1:].partition("]")
        return host.strip(), (
            int(rest[1:]) if rest.startswith(":") and rest[1:].isdigit() else 0
        )
    if _is_ip_addr(text):
        return text, 0
    host, sep, port = text.rpartition(":")
    if sep and port.isdigit():
        return host.strip(), int(port)
    return text, 0


def _canonize_host(ep: str, default: str, link_local: bool) -> tuple[str, int]:
    if not ep:
        return default, 0
    host, port = _parse_endpoint(ep)
    host = _canonize_ip(host or default, loopback=True, link_local=link_local) or (
        default if _is_ip_addr(host) else host
    )
    return host, port if 0 < port <= 65535 else 0


def _has_join_hook(obj: Any | None) -> bool:
    return obj is not None and getattr(obj, "join_hook", None) is not None


def _get_device_id(device: Optional[torch.device]) -> Optional[Iterable[int]]:
    return (
        [int(device.index)]
        if device and device.type in {"cuda", "xpu"} and device.index is not None
        else None
    )


def _safe_getaddrinfo(host: str) -> list[tuple[Any, ...]]:
    with contextlib.suppress(Exception):
        return socket.getaddrinfo(
            host, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
        )
    return []


def _get_preferred_ip_cached(
    hostname: str | None,
    prefer_ipv6: bool,
    allow_loopback: bool,
    allow_link_local: bool,
) -> str:
    names = [h for h in [hostname, socket.gethostname()] if h] + (
        ["localhost"] if allow_loopback else []
    )
    found, seen = [], set()
    for name in names:
        for info in _safe_getaddrinfo(name):
            if not info[4] or not (
                canon := _canonize_ip(
                    info[4][0], loopback=True, link_local=allow_link_local
                )
            ):
                continue
            try:
                ip = ipaddress.ip_address(_coerce_ip_addr(canon))
            except:
                continue
            if ip.is_unspecified:
                continue
            if (ip.version, canon) in seen:
                continue
            seen.add((ip.version, canon))

            score = 1 if ip.is_loopback else (2 if ip.is_link_local else 3)
            if (score == 1 and not allow_loopback) or (
                score == 2 and not allow_link_local
            ):
                continue
            found.append((score, ip.version == (6 if prefer_ipv6 else 4), canon))
    found.sort(key=lambda x: x[:2], reverse=True)
    if found:
        return found[0][2]
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


def _hsdp_supported_params() -> set[str]:
    if fully_shard is None:
        return set()
    try:
        sig = inspect.signature(fully_shard)
        return set(sig.parameters.keys())
    except Exception:
        return set()


def _get_gloox_gloo_process_group(pg: ProcessGroup | None) -> ProcessGroup:
    if not is_distributed():
        return pg or dist.group.WORLD

    base_pg = pg or dist.group.WORLD

    try:
        if dist.get_backend(base_pg) == "gloo":
            return base_pg
    except Exception:
        pass

    try:
        ranks = tuple(dist.get_process_group_ranks(base_pg))
    except Exception:
        ranks = tuple(range(dist.get_world_size(base_pg)))

    cached = _GLOOX_GLOO_PG_CACHE.get(ranks)
    if cached is not None:
        return cached

    try:
        gloo_pg = dist.new_group(ranks=list(ranks), backend="gloo")
    except Exception:
        return base_pg

    _GLOOX_GLOO_PG_CACHE[ranks] = gloo_pg
    return gloo_pg


def _iter_buckets_by_bytes(
    tensors: list[Tensor], max_bucket_bytes: int
) -> Iterable[list[Tensor]]:
    bucket: list[Tensor] = []
    bucket_bytes = 0

    for t in tensors:
        t_bytes = t.numel() * t.element_size()
        if bucket and bucket_bytes + t_bytes > max_bucket_bytes:
            yield bucket
            bucket = []
            bucket_bytes = 0

        bucket.append(t)
        bucket_bytes += t_bytes

    if bucket:
        yield bucket


def _broadcast_bucket_gloox(
    tensors: list[Tensor], *args: Any, src_rank: int, group: ProcessGroup
) -> None:
    if not tensors:
        return

    rank = dist.get_rank(group)

    cpu_parts: list[Tensor] = []
    for t in tensors:
        if rank == src_rank:
            cpu_t = t.detach().to(device="cpu")
        else:
            cpu_t = torch.empty_like(t, device="cpu")
        cpu_parts.append(cpu_t.contiguous())

    flat = torch._utils._flatten_dense_tensors(cpu_parts)
    dist.broadcast(flat, src=src_rank, group=group)

    out_parts = torch._utils._unflatten_dense_tensors(flat, cpu_parts)
    for orig, cpu in zip(tensors, out_parts):
        if orig.device.type == "cpu":
            orig.copy_(cpu)
        else:
            orig.copy_(cpu, non_blocking=True)


def _broadcast_large_tensor_gloox(
    tensor: Tensor,
    *args: Any,
    src_rank: int,
    group: ProcessGroup,
    chunk_mb: int,
    max_inflight_mb: int,
) -> None:
    flat = tensor.reshape(-1)
    elem_size = flat.element_size()

    max_inflight_bytes = max(0, int(max_inflight_mb) * 1024 * 1024)

    chunk_bytes = max(1, int(chunk_mb) * 1024 * 1024)
    if max_inflight_bytes > 0 and chunk_bytes > max_inflight_bytes:
        chunk_bytes = max_inflight_bytes
    chunk_elems = max(1, chunk_bytes // elem_size)

    use_async = max_inflight_bytes > 0

    rank = dist.get_rank(group)

    pending: list[tuple[dist.Work, Tensor, int, int]] = []
    inflight_bytes = 0

    offset = 0
    total = flat.numel()

    while offset < total:
        n = min(chunk_elems, total - offset)

        if rank == src_rank:
            cpu_chunk = flat[offset : offset + n].detach().to(device="cpu")
        else:
            cpu_chunk = torch.empty((n,), dtype=flat.dtype, device="cpu")

        work = dist.broadcast(cpu_chunk, src=src_rank, group=group, async_op=use_async)

        if use_async:
            pending.append((work, cpu_chunk, offset, n))
            inflight_bytes += n * elem_size

            if inflight_bytes >= max_inflight_bytes:
                for w, c, o, nn in pending:
                    w.wait()
                    if rank != src_rank:
                        flat[o : o + nn].copy_(c, non_blocking=True)

                pending.clear()
                inflight_bytes = 0
        else:
            if rank != src_rank:
                flat[offset : offset + n].copy_(cpu_chunk, non_blocking=True)

        offset += n

    for w, c, o, nn in pending:
        w.wait()
        if rank != src_rank:
            flat[o : o + nn].copy_(c, non_blocking=True)


def _broadcast_large_tensor(
    tensor: Tensor,
    *args: Any,
    group: ProcessGroup,
    src_rank: int,
    chunk_mb: int,
    max_inflight_mb: int,
) -> None:
    if not is_distributed():
        return

    flat = tensor.reshape(-1)
    elem_size = flat.element_size()

    max_inflight_bytes = max(0, int(max_inflight_mb) * 1024 * 1024)
    chunk_bytes = max(1, int(chunk_mb) * 1024 * 1024)
    if max_inflight_bytes > 0 and chunk_bytes > max_inflight_bytes:
        chunk_bytes = max_inflight_bytes
    chunk_elems = max(1, chunk_bytes // max(1, elem_size))

    use_async = max_inflight_bytes > 0

    pending: list[dist.Work] = []
    inflight_bytes = 0

    offset = 0
    total = flat.numel()

    while offset < total:
        n = min(chunk_elems, total - offset)
        view = flat[offset : offset + n]

        work = dist.broadcast(view, src=src_rank, group=group, async_op=use_async)

        if use_async:
            pending.append(work)
            inflight_bytes += n * elem_size
            if inflight_bytes >= max_inflight_bytes:
                for w in pending:
                    w.wait()
                pending.clear()
                inflight_bytes = 0
        else:
            pass

        offset += n

    for w in pending:
        w.wait()


def _all_reduce_tensor_gloox(
    tensor: Tensor,
    *args: Any,
    group: ProcessGroup,
    chunk_mb: int,
    max_inflight_mb: int,
    average: bool,
    world_size: int,
) -> None:
    if tensor.numel() == 0:
        return

    if tensor.device.type == "cpu":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        if average:
            tensor.div_(world_size)
        return

    if not tensor.is_contiguous():
        tmp = tensor.contiguous()
        _all_reduce_tensor_gloox(
            tmp,
            group=group,
            chunk_mb=chunk_mb,
            max_inflight_mb=max_inflight_mb,
            average=average,
            world_size=world_size,
        )
        tensor.copy_(tmp)
        return

    flat = tensor.view(-1)
    elem_size = flat.element_size()
    chunk_bytes = max(1, int(chunk_mb) * 1024 * 1024)
    chunk_elems = max(1, chunk_bytes // elem_size)

    max_inflight_bytes = max(0, int(max_inflight_mb) * 1024 * 1024)
    use_async = max_inflight_bytes > 0

    pending: list[tuple[Any, Tensor, int, int]] = []
    inflight = 0

    offset = 0
    total = flat.numel()
    while offset < total:
        n = min(chunk_elems, total - offset)
        cpu_chunk = flat[offset : offset + n].detach().to("cpu")

        work = dist.all_reduce(
            cpu_chunk,
            op=dist.ReduceOp.SUM,
            group=group,
            async_op=use_async,
        )

        if use_async:
            pending.append((work, cpu_chunk, offset, n))
            inflight += n * elem_size

            if inflight >= max_inflight_bytes:
                for w, c, o, nn in pending:
                    w.wait()
                    if average:
                        c.div_(world_size)
                    flat[o : o + nn].copy_(c, non_blocking=True)
                pending.clear()
                inflight = 0
        else:
            if average:
                cpu_chunk.div_(world_size)
            flat[offset : offset + n].copy_(cpu_chunk, non_blocking=True)

        offset += n

    for w, c, o, nn in pending:
        w.wait()
        if average:
            c.div_(world_size)
        flat[o : o + nn].copy_(c, non_blocking=True)


def resolve_ip_expr(
    host: Any,
    *args: Any,
    allow_loopback: bool = False,
    prefer_ipv6: bool | None = None,
    allow_link_local: bool | None = None,
    **kwargs: Any,
) -> str | None:
    host_text = str(host).strip() if host else ""
    if not host_text:
        return None
    link_local = (
        allow_link_local
        if allow_link_local is not None
        else env_bool("ENN_ALLOW_LINK_LOCAL", False)
    )
    if lit := _canonize_ip(host_text, loopback=allow_loopback, link_local=link_local):
        return lit
    addrs = _safe_getaddrinfo(_coerce_ip_addr(host_text))
    if not addrs:
        return None
    res = {4: [], 6: []}
    for a in addrs:
        if a[4] and (
            canon := _canonize_ip(
                a[4][0], loopback=allow_loopback, link_local=link_local
            )
        ):
            with contextlib.suppress(Exception):
                res[ipaddress.ip_address(_coerce_ip_addr(canon)).version].append(canon)
    vers = (6, 4) if (prefer_ipv6 is None or prefer_ipv6) else (4, 6)
    for v in vers:
        if res[v]:
            return res[v][0]
    if "%" in host_text and link_local:
        for v in vers:
            for ip in res[v]:
                if v == 6 and ipaddress.ip_address(_coerce_ip_addr(ip)).is_link_local:
                    return f"{ip}%{host_text.partition('%')[2]}"
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
    link_local = (
        allow_link_local
        if allow_link_local is not None
        else env_bool("ENN_ALLOW_LINK_LOCAL", False)
    )
    for h in [host, fallback]:
        txt = str(h).strip() if h else ""
        if txt:
            if _is_ip_addr(txt):
                if lit := _canonize_ip(
                    txt, loopback=allow_loopback, link_local=link_local
                ):
                    return lit
            elif allow_hostname:
                return txt
    return _canonize_ip(default, loopback=True, link_local=True) or default


def is_port_available(
    host: str, port: int, *args: Any, allow_link_local: bool | None = None
) -> bool:
    if port <= 0 or port > 65535:
        return False
    link_local = (
        allow_link_local
        if allow_link_local is not None
        else env_bool("ENN_ALLOW_LINK_LOCAL", False)
    )
    host_ip = _canonize_ip(
        host, loopback=True, link_local=link_local
    ) or resolve_ip_expr(
        host,
        allow_loopback=True,
        prefer_ipv6=True,
        allow_link_local=link_local,
    )
    if not host_ip:
        return False
    try:
        ver = ipaddress.ip_address(_coerce_ip_addr(host_ip)).version
        family = socket.AF_INET6 if ver == 6 else socket.AF_INET
        addr = (host_ip, port, 0, 0) if ver == 6 else (host_ip, port)
        with contextlib.closing(socket.socket(family, socket.SOCK_STREAM)) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if ver == 6 and hasattr(socket, "IPV6_V6ONLY"):
                with contextlib.suppress(OSError):
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind(addr)
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
    link_local = (
        allow_link_local
        if allow_link_local is not None
        else env_bool("ENN_ALLOW_LINK_LOCAL", False)
    )
    if endpoint:
        h, p = _canonize_host(endpoint, default_host, link_local)
        if p > 0:
            return _format_endpoint(h, p)

    cand = (
        _canonize_ip(default_host, loopback=True, link_local=True)
        or str(default_host).strip()
        or "127.0.0.1"
    )
    host = (
        _canonize_ip(cand, loopback=True, link_local=link_local)
        or resolve_ip_expr(
            cand,
            allow_loopback=True,
            prefer_ipv6=True,
            allow_link_local=link_local,
        )
        or "127.0.0.1"
    )
    try:
        ver = ipaddress.ip_address(_coerce_ip_addr(host)).version
        family, addr = (
            (socket.AF_INET6, (host, 0, 0, 0))
            if ver == 6
            else (socket.AF_INET, (host, 0))
        )
        with contextlib.closing(socket.socket(family, socket.SOCK_STREAM)) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if ver == 6 and hasattr(socket, "IPV6_V6ONLY"):
                with contextlib.suppress(OSError):
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind(addr)
            return _format_endpoint(host, int(sock.getsockname()[1]))
    except OSError:
        return _format_endpoint(host, 0)


def supported_ip_ver(
    *args: Any, allow_loopback: bool = True, **kwargs: Any
) -> tuple[bool, bool]:
    ipv4_ok = False
    ipv6_ok = False
    ipv4_host = "127.0.0.1" if allow_loopback else "0.0.0.0"
    try:
        with contextlib.closing(
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ) as sock4:
            sock4.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock4.bind((ipv4_host, 0))
            ipv4_ok = True
    except OSError:
        pass
    if getattr(socket, "has_ipv6", False):
        ipv6_host = "::1" if allow_loopback else "::"
        try:
            with contextlib.closing(
                socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            ) as sock6:
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
        allow_link_local = env_bool("ENN_ALLOW_LINK_LOCAL", False)
    hn = None if hostname is None else str(hostname)
    return _get_preferred_ip_cached(
        hn, bool(prefer_ipv6), bool(allow_loopback), bool(allow_link_local)
    )


def init_master_addr(
    endpoint: Optional[str],
    *args: Any,
    prefer_ipv6: bool = True,
    allow_loopback: bool = True,
    allow_link_local: bool | None = None,
    **kwargs: Any,
) -> tuple[str, int]:
    link_local = (
        allow_link_local
        if allow_link_local is not None
        else env_bool("ENN_ALLOW_LINK_LOCAL", False)
    )
    default_host = get_preferred_ip(
        allow_loopback=allow_loopback,
        prefer_ipv6=prefer_ipv6,
        allow_link_local=link_local,
    )
    default_host = default_host or ("::1" if prefer_ipv6 else "127.0.0.1")

    host, port = _canonize_host(endpoint or "", default_host, link_local)
    if host in {"", "0.0.0.0", "::"}:
        host = default_host

    master_addr = (
        _canonize_ip(host, loopback=allow_loopback, link_local=link_local)
        or host
        or default_host
    )
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
                count = int(
                    get_num_accelerators(str(getattr(dev, "type", "cpu") or "cpu"))
                )
                if count > 0:
                    return count
            return 1
        case _:
            ncpu = CPU.count()
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

    ctx = getattr(model, "no_sync", lambda: contextlib.nullcontext())()
    with ctx:
        yield


def joining(
    model: Any,
    optimizer: Optimizer | None = None,
) -> AbstractContextManager[None]:
    if Join is None:
        return contextlib.nullcontext()
    joinables = tuple(obj for obj in (model, optimizer) if _has_join_hook(obj))
    if not joinables:
        return contextlib.nullcontext()
    return Join(joinables, throw_on_early_termination=True)


def broadcast_scalar(value: int | float, device: torch.device, src: int = 0) -> int:
    if not is_distributed():
        return int(value)
    try:
        tensor = torch.tensor([int(value)], device=device, dtype=torch.int32)
        dist.broadcast(tensor, src=src)
        return int(tensor.item())
    except Exception:
        return int(value)


def is_distributed() -> bool:
    try:
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def get_rank(default: int | None = None) -> int | None:
    if not is_distributed():
        return default
    try:
        return int(dist.get_rank())
    except Exception:
        return default


def is_rank0() -> bool:
    r = get_rank(default=0)
    return int(r or 0) == 0


def distributed_barrier(
    device: Optional[torch.device] = None, group: ProcessGroup | None = None
) -> None:
    if not is_distributed():
        return

    if group is None and device is not None and not isinstance(device, torch.device):
        with contextlib.suppress(Exception):
            from torch.distributed.distributed_c10d import ProcessGroup as _PG

            if isinstance(device, _PG):
                group = device
                device = None

    pg = group or dist.group.WORLD
    try:
        dist.barrier(group=pg, device_ids=_get_device_id(device))
    except TypeError:
        dist.barrier(group=pg)


def distributed_broadcast(
    target_module: torch.nn.Module,
    *args: Any,
    src_rank: int = 0,
    group: ProcessGroup | None = None,
    include_buffers: bool = True,
    max_buffer_size_mb: int = 25,
    policy: "CollectivePolicy | None" = None,
) -> None:
    from .policies import CollectivePolicy

    if not is_distributed():
        return

    group = group or dist.group.WORLD

    if dist.get_world_size(group) <= 1:
        return

    policy = policy or CollectivePolicy.from_env()

    if include_buffers:
        include_buffers = bool(policy.include_buffers)
    if max_buffer_size_mb == 25:
        max_buffer_size_mb = int(policy.max_buffer_size_mb)

    tensors: list[Tensor] = []

    if policy.include_parameters:
        tensors.extend([p.data for p in target_module.parameters()])

    if include_buffers:
        max_bytes = int(max_buffer_size_mb) * 1024 * 1024
        for b in target_module.buffers():
            if b is None:
                continue

            if b.numel() * b.element_size() > max_bytes:
                continue
            tensors.append(b.data)

    if not tensors:
        return

    coalesce_bytes = max(1, int(policy.coalesce_mb)) * 1024 * 1024
    max_tensor_bytes = max(1, int(policy.max_tensor_mb_for_coalesce)) * 1024 * 1024

    def _is_small(t: Tensor) -> bool:
        return (t.numel() * t.element_size()) <= max_tensor_bytes

    small_tensors = [t for t in tensors if _is_small(t)]
    large_tensors = [t for t in tensors if not _is_small(t)]

    world_size = dist.get_world_size(group)
    local_world_size = env_first_int(
        ["LOCAL_WORLD_SIZE", "SLURM_STEP_NUM_TASKS"], default=world_size
    )
    multi_node = bool(world_size > local_world_size)

    chunk_mb = int(policy.inter_stream_mb if multi_node else policy.intra_stream_mb)
    max_inflight_mb = int(policy.max_inflight_mb)

    backend = (policy.backend or "c10d").strip().lower()

    if policy.debug_collectives and is_rank0():
        total_bytes = sum(t.numel() * t.element_size() for t in tensors)
        small_bytes = sum(t.numel() * t.element_size() for t in small_tensors)
        large_bytes = total_bytes - small_bytes
        print(
            f"[collectives/broadcast] backend={backend} tensors={len(tensors)} "
            f"small={len(small_tensors)} ({small_bytes / 1024 / 1024:.1f} MiB) "
            f"large={len(large_tensors)} ({large_bytes / 1024 / 1024:.1f} MiB) "
            f"chunk={chunk_mb}MiB inflight={max_inflight_mb}MiB multi_node={multi_node}"
        )

    if backend == "gloox":
        gloo_group = _get_gloox_gloo_process_group(group)

        for bucket in _iter_buckets_by_bytes(
            small_tensors, max_bucket_bytes=coalesce_bytes
        ):
            _broadcast_bucket_gloox(bucket, src_rank=src_rank, group=gloo_group)

        for t in large_tensors:
            _broadcast_large_tensor_gloox(
                t,
                src_rank=src_rank,
                group=gloo_group,
                chunk_mb=chunk_mb,
                max_inflight_mb=max_inflight_mb,
            )

        return

    if small_tensors:
        element_size = small_tensors[0].element_size()
        coalesce_numel = max(1, coalesce_bytes // element_size)
        dist._broadcast_coalesced(
            group, small_tensors, buffer_size=coalesce_numel, src=src_rank
        )

    for t in large_tensors:
        _broadcast_large_tensor(
            t,
            group=group,
            src_rank=src_rank,
            chunk_mb=chunk_mb,
            max_inflight_mb=max_inflight_mb,
        )


def distributed_sync(
    target_module: nn.Module,
    device: torch.device | None = None,
    group: ProcessGroup | None = None,
    include_buffers: bool = True,
    max_buffer_size_mb: int = 25,
    *args: Any,
    policy: "CollectivePolicy | None" = None,
) -> None:
    if not is_distributed():
        return

    pg = group or _get_default_process_group()
    if pg is None:
        return

    if dist.get_world_size(pg) <= 1:
        return

    device = device or get_distributed_device()

    distributed_broadcast(
        target_module=target_module,
        group=pg,
        include_buffers=include_buffers,
        max_buffer_size_mb=max_buffer_size_mb,
        src_rank=0,
        policy=policy,
    )

    distributed_barrier(group=pg, device=device)


def distributed_all_reduce_grads(
    module: nn.Module,
    *args: Any,
    group: ProcessGroup | None = None,
    average: bool = True,
    policy: "CollectivePolicy | None" = None,
) -> None:
    if not is_distributed():
        return

    pg = group or _get_default_process_group()
    if pg is None:
        return

    world_size = dist.get_world_size(pg)
    if world_size <= 1:
        return

    from .policies import CollectivePolicy

    policy = policy or CollectivePolicy.from_env()

    grads: list[Tensor] = []
    for p in module.parameters():
        g = p.grad
        if g is None:
            continue
        if getattr(g, "is_sparse", False):
            raise NotImplementedError(
                "distributed_all_reduce_grads does not support sparse gradients"
            )
        grads.append(g)

    if not grads:
        return

    local_world_size = env_first_int(
        [
            "LOCAL_WORLD_SIZE",
            "MPI_LOCALNRANKS",
            "SLURM_NTASKS_PER_NODE",
            "OMPI_COMM_WORLD_LOCAL_SIZE",
        ],
        default=world_size,
    )
    multi_node = world_size > local_world_size

    chunk_mb = policy.inter_stream_mb if multi_node else policy.intra_stream_mb
    max_inflight_mb = policy.max_inflight_mb

    backend = str(getattr(policy, "backend", "c10d")).strip().lower()
    if backend == "gloox":
        gloo_pg = _get_gloox_gloo_process_group(pg)
        for g in grads:
            _all_reduce_tensor_gloox(
                g,
                group=gloo_pg,
                chunk_mb=chunk_mb,
                max_inflight_mb=max_inflight_mb,
                average=average,
                world_size=world_size,
            )
        return

    for g in grads:
        if g.numel() == 0:
            continue
        if not g.is_contiguous():
            tmp = g.contiguous()
            dist.all_reduce(tmp, op=dist.ReduceOp.SUM, group=pg)
            if average:
                tmp.div_(world_size)
            g.copy_(tmp)
        else:
            dist.all_reduce(g, op=dist.ReduceOp.SUM, group=pg)
            if average:
                g.div_(world_size)


def is_dtensor_active() -> bool:
    return bool(_DTENSOR_ACTIVE)


def to_hsdp_module(
    module: torch.nn.Module,
    *args: Any,
    mesh: Any | None,
    mp_policy: Any | None = None,
    reshard_after_forward: bool = False,
    sync_module_states: bool = True,
    **user_kwargs: Any,
) -> torch.nn.Module:
    if fully_shard is None:
        raise RuntimeError("Missing fully_shard")
    params = _hsdp_supported_params()
    fsdp_kwargs: dict[str, Any] = dict(user_kwargs)
    pg_obj: Any | None = None
    mesh_obj: Any | None = None
    if mesh is not None:
        _set_dtensor_active()
        with contextlib.suppress(Exception):
            from torch.distributed.distributed_c10d import ProcessGroup

            if isinstance(mesh, ProcessGroup):
                pg_obj = mesh
            else:
                mesh_obj = mesh
        if pg_obj is None and mesh_obj is None:
            mesh_obj = mesh
    defaults: dict[str, Any] = {
        "forward_prefetch": env_bool("ENN_FSDP_FWD_PREFETCH", True),
        "limit_all_gathers": env_bool("ENN_FSDP_LIMIT_AG", True),
        "use_orig_params": env_bool("ENN_FSDP_USE_ORIG_PARAMS", True),
        "mp_policy": mp_policy,
        "reshard_after_forward": reshard_after_forward,
        "sync_module_states": sync_module_states,
    }
    if mesh_obj is not None:
        if "mesh" in params:
            defaults["mesh"] = mesh_obj
        elif "device_mesh" in params:
            defaults["device_mesh"] = mesh_obj

    if pg_obj is not None and "process_group" in params:
        defaults["process_group"] = pg_obj
    fsdp_kwargs.update(
        {
            k: v
            for k, v in defaults.items()
            if k in params and k not in fsdp_kwargs and v is not None
        }
    )
    sharded = fully_shard(module, *args, **fsdp_kwargs)
    with contextlib.suppress(AttributeError):
        sharded.set_requires_gradient_sync(True)
    with contextlib.suppress(ImportError):
        from torch.distributed.fsdp import register_fsdp_forward_method

        for _name in (
            "forward",
            "decode",
            "predict",
            "forward_export",
            "forward_state",
            "forward_stream",
        ):
            if hasattr(sharded, _name):
                register_fsdp_forward_method(sharded, _name)
    return sharded


def get_distributed_mesh(
    device: torch.device | None = None,
) -> tuple[Any | None, str]:
    if not is_distributed():
        return (None, "none")
    dev = device
    if dev is None:
        with contextlib.suppress(Exception):
            dev = get_device()
    dev_type = str(getattr(dev, "type", "cpu"))
    if dev_type not in {"cuda", "xpu"}:
        return (None, "none")
    try:
        world = int(dist.get_world_size())
    except Exception:
        return (None, "none")
    if world <= 1:
        return (None, "none")
    local_world_size = None
    for k in (
        "LOCAL_WORLD_SIZE",
        "MPI_LOCALNRANKS",
        "SLURM_NTASKS_PER_NODE",
        "OMPI_COMM_WORLD_LOCAL_SIZE",
    ):
        if v := os.environ.get(k):
            with contextlib.suppress(ValueError):
                local_world_size = int(v)
                break
    if local_world_size is None:
        if dev_type == "cuda":
            with contextlib.suppress(Exception):
                local_world_size = torch.cuda.device_count()
        elif dev_type == "xpu":
            with contextlib.suppress(Exception):
                local_world_size = int(get_num_accelerators("xpu"))
    local_world_size = int(local_world_size or 1)
    is_consistent = True
    if dist.is_initialized():
        try:
            my_size = torch.tensor([local_world_size], device=dev, dtype=torch.long)
            gathered = [torch.zeros_like(my_size) for _ in range(world)]
            dist.all_gather(gathered, my_size)
            all_sizes = [t.item() for t in gathered]
            is_consistent = all(s == local_world_size for s in all_sizes)
        except Exception:
            is_consistent = False
    if importlib.util.find_spec("torch.distributed.device_mesh") is None:
        return (None, "none")
    device_mesh = importlib.import_module("torch.distributed.device_mesh")
    init_device_mesh = getattr(device_mesh, "init_device_mesh", None)
    if init_device_mesh is None:
        return (None, "none")
    if is_consistent and world > local_world_size and world % local_world_size == 0:
        dp_replicate = world // local_world_size
        dp_shard = local_world_size
        try:
            mesh = init_device_mesh(
                dev_type,
                (dp_replicate, dp_shard),
                mesh_dim_names=("dp_replicate", "dp_shard"),
            )
            return (mesh, "hsdp2")
        except Exception:
            pass
    try:
        mesh = init_device_mesh(
            dev_type,
            (world,),
            mesh_dim_names=("dp",),
        )
        return (mesh, "fsdp2")
    except Exception:
        return (None, "none")
