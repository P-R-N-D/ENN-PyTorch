# -*- coding: utf-8 -*-
from __future__ import annotations

import concurrent.futures
import contextlib
import ctypes
import dataclasses
import gc
import importlib
import importlib.util
import inspect
import ipaddress
import json
import logging
import os
import random
import re
import shutil
import socket
import threading
import time
import warnings
from collections import deque
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
import torch.distributed as dist
from ..core.concurrency import Mutex
from ..core.datatypes import PathLike, env_bool, env_int, save_temp, write_json
from ..core.system import (
    CPU,
    Memory,
    get_device,
    get_num_accelerators,
    init_python_path,
    init_start_method,
    set_accelerator_index,
    set_accelerator_seed,
)
from torch import nn
from torch.optim import Optimizer
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
_LOGGER = logging.getLogger(__name__)
_INFLIGHT_LOCK_NAME = ".inflight.lock"
_INFLIGHT_LOCK_TTL_SEC = int(os.environ.get("ENN_DCP_INFLIGHT_LOCK_TTL_SEC", "21600"))


def _atomic_create_json(path, payload: dict) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        with open(path, "x", encoding="utf-8") as f:
            json.dump(payload, f)
        return True
    except FileExistsError:
        return False
    except Exception:
        with contextlib.suppress(Exception):
            if path.exists():
                path.unlink()
        return False


def _safe_rmtree(path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    try:
        if not path.exists():
            return
    except Exception:
        return
    ts = int(time.time())
    dp = path.with_name(f"{path.name}.delete_pending.{ts}")
    with contextlib.suppress(Exception):
        path.rename(dp)
    shutil.rmtree(dp, ignore_errors=True)


def _cleanup_delete_pending(root) -> None:
    with contextlib.suppress(Exception):
        for p in root.glob("*.delete_pending.*"):
            _safe_rmtree(p)


def _coerce_dcp_keys(state: object) -> object:
    if isinstance(state, dict):
        keys_to_drop: list[object] = []
        for key, value in state.items():
            key_str = str(key)
            if (
                key_str.endswith("._extra_state")
                or key_str.endswith("_extra_state")
                or key_str.endswith("output_baked_flag")
            ):
                keys_to_drop.append(key)
                continue
            state[key] = _coerce_dcp_keys(value)
        for key in keys_to_drop:
            with contextlib.suppress(Exception):
                state.pop(key, None)
    return state


def _overlay_avg_state_dict(dst: object, avg: Mapping[str, Any]) -> object:
    if not isinstance(dst, dict) or not isinstance(avg, Mapping):
        return dst
    for k, v in avg.items():
        if not isinstance(k, str) or not torch.is_tensor(v):
            continue
        cur = dst.get(k, None)
        if not torch.is_tensor(cur):
            continue
        try:
            vv = v.detach()
            if tuple(vv.shape) != tuple(cur.shape):
                continue
            if vv.dtype != cur.dtype:
                vv = vv.to(dtype=cur.dtype)
            cur.copy_(vv, non_blocking=True)
        except Exception:
            continue
    return dst


def _clone_state_dict(state: object, *, to_cpu: bool = False) -> object:
    if torch.is_tensor(state):
        t = state.detach()
        if to_cpu and getattr(t, "device", None) is not None and t.device.type != "cpu":
            return t.to(device="cpu")
        return t.clone()
    if isinstance(state, dict):
        return {k: _clone_state_dict(v, to_cpu=to_cpu) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        cloned = (_clone_state_dict(v, to_cpu=to_cpu) for v in state)
        return type(state)(cloned)
    return state


def _future_result(fut: object) -> object:
    if fut is None:
        return None
    wait_fn = getattr(fut, "wait", None)
    if callable(wait_fn):
        r = wait_fn()
        result_fn = getattr(fut, "result", None)
        if callable(result_fn):
            return result_fn()
        return r
    result_fn = getattr(fut, "result", None)
    if callable(result_fn):
        return result_fn()
    return fut


def _future_done(fut: object) -> bool:
    if fut is None:
        return True
    done_fn = getattr(fut, "done", None)
    if callable(done_fn):
        try:
            return bool(done_fn())
        except Exception:
            return False
    return False


def _add_future_callback(fut: object, fn: Callable[[], None]) -> None:
    if fut is None:
        return
    then = getattr(fut, "then", None)
    if callable(then):
        with contextlib.suppress(Exception):
            then(lambda _: fn())
            return
    add_done = getattr(fut, "add_done_callback", None)
    if callable(add_done):
        with contextlib.suppress(Exception):
            add_done(lambda _: fn())


def _set_dtensor_active() -> None:
    global _DTENSOR_ACTIVE
    _DTENSOR_ACTIVE = True


def _from_hsdp_module(module: torch.nn.Module) -> None:
    if not (
        is_distributed()
        and callable(unshard := getattr(module, "unshard", None))
    ):
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
            if ipaddress.ip_address(_coerce_ip_addr(h)).version == 6
            and ":" in h
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
    host = _canonize_ip(
        host or default, loopback=True, link_local=link_local
    ) or (default if _is_ip_addr(host) else host)
    return host, port if 0 < port <= 65535 else 0


def _has_join_hook(obj: Any | None) -> bool:
    return obj is not None and getattr(obj, "join_hook", None) is not None


def _get_device_id(device: Optional[torch.device]) -> Optional[Iterable[int]]:
    return (
        [int(device.index)]
        if device
        and device.type in {"cuda", "xpu"}
        and device.index is not None
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
            found.append(
                (score, ip.version == (6 if prefer_ipv6 else 4), canon)
            )
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

        work = dist.broadcast(
            cpu_chunk, src=src_rank, group=group, async_op=use_async
        )

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

        work = dist.broadcast(
            view, src=src_rank, group=group, async_op=use_async
        )

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
    if lit := _canonize_ip(
        host_text, loopback=allow_loopback, link_local=link_local
    ):
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
                res[
                    ipaddress.ip_address(_coerce_ip_addr(canon)).version
                ].append(canon)
    vers = (6, 4) if (prefer_ipv6 is None or prefer_ipv6) else (4, 6)
    for v in vers:
        if res[v]:
            return res[v][0]
    if "%" in host_text and link_local:
        for v in vers:
            for ip in res[v]:
                if (
                    v == 6
                    and ipaddress.ip_address(_coerce_ip_addr(ip)).is_link_local
                ):
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
        with contextlib.closing(
            socket.socket(family, socket.SOCK_STREAM)
        ) as sock:
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
        with contextlib.closing(
            socket.socket(family, socket.SOCK_STREAM)
        ) as sock:
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
                if hasattr(socket, "IPPROTO_IPV6") and hasattr(
                    socket, "IPV6_V6ONLY"
                ):
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
                    get_num_accelerators(
                        str(getattr(dev, "type", "cpu") or "cpu")
                    )
                )
                if count > 0:
                    return count
            return 1
        case _:
            ncpu = CPU.count()
            return max(1, min(int(ncpu), 4))


def is_process_group(obj: object) -> bool:
    if obj is None:
        return False
    with contextlib.suppress(Exception):
        from torch.distributed.distributed_c10d import ProcessGroup

        return isinstance(obj, ProcessGroup)
    return False


def resolve_process_group(meta: object, model: object) -> object | None:
    candidates: list[tuple[object, str]] = [
        (meta, "process_group"),
        (meta, "distributed_process_group"),
    ]
    tm = model.module if hasattr(model, "module") else model
    candidates.extend(
        [
            (tm, "process_group"),
            (tm, "distributed_process_group"),
        ]
    )
    for obj, attr in candidates:
        try:
            pg = getattr(obj, attr, None)
        except Exception:
            pg = None
        if is_process_group(pg):
            return pg
    return None


def get_group_world_size(group: object | None) -> int:
    try:
        if group is None or not is_process_group(group):
            if dist.is_available() and dist.is_initialized():
                return int(dist.get_world_size())
            return int(get_world_size())
        return int(dist.get_world_size(group=group))
    except Exception:
        return int(get_world_size())


def distributed_all_reduce_sum(
    t: torch.Tensor, group: object | None = None
) -> None:
    if not isinstance(t, torch.Tensor):
        return
    try:
        if not (dist.is_available() and dist.is_initialized()):
            return
    except Exception:
        return

    try:
        if group is None or not is_process_group(group):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        else:
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
    except Exception:
        with contextlib.suppress(Exception):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)


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


def broadcast_scalar(
    value: int | float, device: torch.device, src: int = 0
) -> int:
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

    if (
        group is None
        and device is not None
        and not isinstance(device, torch.device)
    ):
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
    max_tensor_bytes = (
        max(1, int(policy.max_tensor_mb_for_coalesce)) * 1024 * 1024
    )

    def _is_small(t: Tensor) -> bool:
        return (t.numel() * t.element_size()) <= max_tensor_bytes

    small_tensors = [t for t in tensors if _is_small(t)]
    large_tensors = [t for t in tensors if not _is_small(t)]

    world_size = dist.get_world_size(group)
    local_world_size = env_first_int(
        ["LOCAL_WORLD_SIZE", "SLURM_STEP_NUM_TASKS"], default=world_size
    )
    multi_node = bool(world_size > local_world_size)

    chunk_mb = int(
        policy.inter_stream_mb if multi_node else policy.intra_stream_mb
    )
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
            _broadcast_bucket_gloox(
                bucket, src_rank=src_rank, group=gloo_group
            )

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
            my_size = torch.tensor(
                [local_world_size], device=dev, dtype=torch.long
            )
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
    if (
        is_consistent
        and world > local_world_size
        and world % local_world_size == 0
    ):
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


class ProcessBroker:
    DL_STATE_FILE: str = "dataloader.json"

    _IGNORED_WARNING_PATTERNS: tuple[str, ...] = (
        "torch.distributed is disabled, unavailable or uninitialized",
        "torch.distributed is disabled",
        "TypedStorage is deprecated",
        "Found a non-scalar tensor with numel=1 and ndim!=0",
        "distributed_broadcast: coalesced broadcast failed",
        "distributed_broadcast: per-tensor broadcast failed",
        "found no DeviceMesh from dtensor args",
        "mixed precision.*may be unavailable",
        "Either mode or options can be specified, but both can't be specified at the same time\\.",
    )
    _IGNORED_WARNING_MESSAGE_RE = re.compile(
        r".*(?:"
        + "|".join((f"(?:{p})" for p in _IGNORED_WARNING_PATTERNS))
        + r").*"
    )

    @classmethod
    def apply_warning_filters(cls) -> None:
        with contextlib.suppress(Exception):
            warnings.filterwarnings(
                "ignore",
                message=cls._IGNORED_WARNING_MESSAGE_RE.pattern,
                category=UserWarning,
            )

    @classmethod
    def clear_process_group(cls) -> None:
        try:
            if dist.is_available() and dist.is_initialized():
                with contextlib.suppress(Exception):
                    dist.barrier()
                with contextlib.suppress(Exception):
                    dist.destroy_process_group()
        except Exception:
            pass

    @classmethod
    def set_seed(cls, seed_value: int | None) -> None:
        if seed_value is None:
            return
        try:
            seed_i = int(seed_value)
        except Exception:
            return
        with contextlib.suppress(Exception):
            torch.manual_seed(seed_i)
        with contextlib.suppress(Exception):
            set_accelerator_seed(seed_i)
        with contextlib.suppress(Exception):
            random.seed(seed_i)
        with contextlib.suppress(Exception):
            import numpy

            numpy.random.seed(seed_i)

    @classmethod
    def bootstrap(
        cls,
        *args: Any,
        seed: int | None = None,
        clear_pg: bool = True,
        apply_warning_filters: bool = True,
        **kwargs: Any,
    ) -> None:
        if apply_warning_filters:
            cls.apply_warning_filters()
        if clear_pg:
            cls.clear_process_group()

        init_python_path()
        with contextlib.suppress(Exception):
            torch.multiprocessing.allow_connection_pickling()
        with contextlib.suppress(Exception):
            init_start_method()
        if seed is not None:
            cls.set_seed(seed)

    @classmethod
    def get_backend_type(cls, device: torch.device) -> str:
        dev_type = str(getattr(device, "type", "cpu")).lower()
        if dev_type == "cuda":
            return "nccl"
        if dev_type == "xpu":
            return "xccl"
        if dev_type in ("cpu", "mps", "dml", "privateuseone"):
            return "gloo"
        if dev_type in ("hpu", "npu"):
            return "hccl"
        if dev_type == "xla":
            return "xla"
        get_default = getattr(
            torch.distributed, "get_default_backend_for_device", None
        )
        if callable(get_default):
            with contextlib.suppress(Exception):
                return str(get_default(device)).lower()
            with contextlib.suppress(Exception):
                return str(get_default(dev_type)).lower()
        return "gloo"

    @classmethod
    def ensure_default_socket_ifname(cls) -> None:
        iface = None
        gloo_if = os.environ.get("GLOO_SOCKET_IFNAME")
        tp_if = os.environ.get("TP_SOCKET_IFNAME")
        if gloo_if or tp_if:
            if gloo_if and (not tp_if):
                os.environ.setdefault("TP_SOCKET_IFNAME", str(gloo_if))
            elif tp_if and (not gloo_if):
                os.environ.setdefault("GLOO_SOCKET_IFNAME", str(tp_if))
            return

        try:
            with open("/proc/net/route", "r", encoding="utf-8") as f:
                for line in f.readlines()[1:]:
                    fields = line.strip().split()
                    if len(fields) >= 2 and fields[1] == "00000000":
                        iface = fields[0]
                        if iface:
                            break
        except Exception:
            iface = None

        if iface is None:
            try:
                import psutil

                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    s.connect(("8.8.8.8", 80))
                    ip = s.getsockname()[0]
                finally:
                    s.close()
                if ip:
                    for name, addrs in psutil.net_if_addrs().items():
                        for a in addrs:
                            if (
                                getattr(a, "family", None) == socket.AF_INET
                                and getattr(a, "address", None) == ip
                            ):
                                iface = str(name)
                                break
                        if iface:
                            break
            except Exception:
                iface = None

        if iface:
            os.environ.setdefault("GLOO_SOCKET_IFNAME", iface)
            os.environ.setdefault("TP_SOCKET_IFNAME", iface)

    @classmethod
    def _configure_torch_nccl_env(cls, device: torch.device) -> None:
        if str(getattr(device, "type", "cpu")) != "cuda":
            return
        world = 1
        with contextlib.suppress(Exception):
            world = int(env_int("WORLD_SIZE", 1) or 1)
        if "TORCH_NCCL_ENABLE_MONITORING" not in os.environ:
            default_mon = 0 if int(world) <= 1 else 1
            mon = int(env_int("ENN_TORCH_NCCL_ENABLE_MONITORING", default_mon))
            os.environ["TORCH_NCCL_ENABLE_MONITORING"] = str(int(mon))
        if "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC" not in os.environ:
            default_hb = 3600 if int(world) <= 1 else 600
            hb = int(
                env_int("ENN_TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", default_hb)
            )
            os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = str(int(hb))
        if "TORCH_NCCL_DUMP_ON_TIMEOUT" not in os.environ:
            default_dump = 0 if int(world) <= 1 else 1
            dump = int(env_int("ENN_TORCH_NCCL_DUMP_ON_TIMEOUT", default_dump))
            os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = str(int(dump))
        if "TORCH_NCCL_ASYNC_ERROR_HANDLING" not in os.environ:
            default_ae = 0 if int(world) <= 1 else 3
            ae = int(
                env_int("ENN_TORCH_NCCL_ASYNC_ERROR_HANDLING", default_ae)
            )
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = str(int(ae))
        if "TORCH_NCCL_BLOCKING_WAIT" not in os.environ:
            default_bw = 1 if int(world) <= 1 else 0
            bw = int(env_int("ENN_TORCH_NCCL_BLOCKING_WAIT", default_bw))
            os.environ["TORCH_NCCL_BLOCKING_WAIT"] = str(int(bw))

    @classmethod
    def configure_torch_nccl_env(cls, device: torch.device) -> None:
        cls._configure_torch_nccl_env(device)

    @classmethod
    def _configure_torch_gloo_env(cls, device: torch.device) -> None:
        cls.ensure_default_socket_ifname()

    @classmethod
    def configure_torch_gloo_env(cls, device: torch.device) -> None:
        cls._configure_torch_gloo_env(device)

    @classmethod
    def _configure_torch_xccl_env(cls, device: torch.device) -> None:
        return

    @classmethod
    def configure_torch_xccl_env(cls, device: torch.device) -> None:
        cls._configure_torch_xccl_env(device)

    @classmethod
    def _coerce_process_group_backend(
        cls, backend: object, device: torch.device
    ) -> object:
        backend_clean: object = backend
        b = str(backend) if backend is not None else ""
        if isinstance(backend, str):
            s = backend.replace("\\n", "").replace("\n", "").replace("\r", "")
            s = s.strip()
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if len(parts) > 1 and all(":" in p for p in parts):
                norm_parts: list[str] = []
                for p in parts:
                    dev_part, _, be_part = p.partition(":")
                    norm_parts.append(
                        f"{dev_part.strip().lower()}:{be_part.strip().lower()}"
                    )
                s = ",".join(norm_parts)
            else:
                s = s.lower()
            backend_clean = s
            b = s
        else:
            b = b.replace("\\n", "").replace("\n", "").replace("\r", "")
            b = b.strip().lower()

        if env_bool("ENN_DISABLE_PG_CPU_BACKEND", False):
            return backend_clean

        dev = str(getattr(device, "type", "cpu")).strip().lower()
        if isinstance(backend_clean, str) and ("," in b and ":" in b):
            return backend_clean
        if b == "nccl" and dev == "cuda":
            return "cpu:gloo,cuda:nccl"
        if b == "xccl" and dev == "xpu":
            return "cpu:gloo,xpu:xccl"
        return backend_clean

    @classmethod
    def configure_backend_env(
        cls, backend: object, device: torch.device
    ) -> None:
        backend_pg = cls._coerce_process_group_backend(backend, device)
        b = str(backend_pg) if backend_pg is not None else ""
        b = b.replace("\\n", "").replace("\n", "").replace("\r", "")
        b = b.lower()
        if "," in b and ":" in b:
            for part in (p.strip() for p in b.split(",") if p.strip()):
                _dev, _, be = part.partition(":")
                if be == "nccl":
                    cls._configure_torch_nccl_env(device)
                elif be == "xccl":
                    cls._configure_torch_xccl_env(device)
                elif be == "gloo":
                    cls._configure_torch_gloo_env(device)
            return
        if b == "nccl":
            cls._configure_torch_nccl_env(device)
        elif b == "xccl":
            cls._configure_torch_xccl_env(device)
        elif b == "gloo":
            cls._configure_torch_gloo_env(device)

    @classmethod
    def init_backend(
        cls, device: torch.device, local_rank: int | None = None
    ) -> None:
        with contextlib.suppress(Exception):
            if device.type == "cuda" and hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = True
        rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
        if local_rank is not None:
            with contextlib.suppress(Exception):
                rank = int(local_rank)
        if device.type in {"cuda", "xpu"}:
            n = max(1, int(get_num_accelerators(device.type) or 1))
            set_accelerator_index(device.type, int(rank) % int(n))
        else:
            cls.ensure_default_socket_ifname()

    @classmethod
    def init_process_group(
        cls, backend: object, device: torch.device, local_rank: int
    ) -> None:
        if torch.distributed.is_initialized():
            return
        backend_pg = cls._coerce_process_group_backend(backend, device)
        dev_id = None
        dev_type = getattr(device, "type", "cpu")
        backend_name = str(backend) if backend is not None else ""
        backend_name = (
            backend_name.replace("\\n", "").replace("\n", "").replace("\r", "")
        ).lower()
        if backend_name in ("nccl", "xccl") and dev_type in ("cuda", "xpu"):
            index = (
                device.index
                if getattr(device, "index", None) is not None
                else env_int("LOCAL_RANK", int(local_rank))
            )
            try:
                dev_id = torch.device(dev_type, index)
            except Exception:
                dev_id = index

        timeout = None
        try:
            import datetime

            to_s = int(env_int("ENN_PROCESS_GROUP_TIMEOUT_SEC", 0) or 0)
            if to_s <= 0 and backend_name in ("nccl", "xccl"):
                ws = int(env_int("WORLD_SIZE", 1) or 1)
                if ws <= 1:
                    to_s = 3600
            if int(to_s) > 0:
                timeout = datetime.timedelta(seconds=int(to_s))
        except Exception:
            timeout = None

        def _init_with(bkend: object) -> None:
            kwargs: dict[str, Any] = {"backend": bkend}
            if dev_id is not None:
                kwargs["device_id"] = dev_id
            if timeout is not None:
                kwargs["timeout"] = timeout
            try:
                torch.distributed.init_process_group(**kwargs)
                return
            except TypeError:
                pass
            kwargs.pop("device_id", None)
            try:
                torch.distributed.init_process_group(**kwargs)
                return
            except TypeError:
                pass
            kwargs.pop("timeout", None)
            torch.distributed.init_process_group(**kwargs)

        try:
            _init_with(backend_pg)
        except Exception:
            if str(backend_pg) == str(backend):
                raise
            _init_with(backend)

    @classmethod
    def loader_state_path(cls, directory: PathLike) -> str:
        return os.path.join(os.fspath(directory), cls.DL_STATE_FILE)

    @classmethod
    def get_loader_state(cls, directory: PathLike) -> str:
        return cls.loader_state_path(directory)

    @classmethod
    def _rank0_only(cls) -> bool:
        return is_rank0()

    @classmethod
    def log_rank0(
        cls,
        logger: logging.Logger,
        msg: str,
        *args: Any,
        only_rank0: bool = True,
        level: str = "info",
        **kwargs: Any,
    ) -> None:
        if only_rank0 and not is_rank0():
            return
        try:
            log_fn = getattr(logger, str(level).lower(), logger.info)
        except Exception:
            log_fn = logger.info
        with contextlib.suppress(Exception):
            log_fn(msg, *args)

    @classmethod
    def rank0_logger(
        cls,
        logger: logging.Logger,
        *,
        only_rank0: bool = True,
        level: str = "info",
    ) -> Callable[..., None]:
        def _fn(
            msg: str,
            *args: Any,
            only_main_rank: bool = True,
            **kwargs: Any,
        ) -> None:
            cls.log_rank0(
                logger,
                msg,
                *args,
                only_rank0=bool(only_main_rank) and bool(only_rank0),
                level=level,
            )

        return _fn

    @classmethod
    def make_progress_bar(
        cls,
        *args: Any,
        title: str,
        total: int,
        device: torch.device,
        **kwargs: Any,
    ) -> object:
        if not cls._rank0_only():
            return None
        if int(total) <= 0:
            return None
        try:
            import sys

            from tqdm.auto import tqdm

            return tqdm(
                total=int(total),
                desc=f"{title} ({device.type.upper()}) ",
                unit="I/O < 0.01 MB/s, COM < 0.01 TFLOPS",
                bar_format="{desc}"
                + "{bar} {percentage:3.2f} % "
                + "({unit}) Elapsed: {elapsed}, Remaining: {remaining}",
                colour="green",
                ascii=True,
                position=int(kwargs.get("position", 0) or 0),
                leave=bool(kwargs.get("leave", False)),
                file=sys.stdout,
            )
        except Exception:
            return None

    @classmethod
    def get_progress_bar(
        cls, title: str, total: int, device: torch.device, **kwargs: Any
    ) -> object:
        return cls.make_progress_bar(
            title=title, total=total, device=device, **kwargs
        )

    @classmethod
    def update_progress_bar(
        cls,
        bar: object,
        finish: bool,
        *args: Any,
        mbps: float | None = None,
        tflops: float | None = None,
        **kwargs: Any,
    ) -> None:
        if bar is None:
            return
        try:
            mbps_val = float(mbps) if mbps is not None else 0.0
        except Exception:
            mbps_val = 0.0
        try:
            tflops_val = float(tflops) if tflops is not None else 0.0
        except Exception:
            tflops_val = 0.0
        io_expr = (
            f"I/O = {mbps_val:.2f} MB/s"
            if mbps_val >= 0.01
            else "I/O < 0.01 MB/s"
        )
        com_expr = (
            f"COM = {tflops_val:.2f} TFLOPS"
            if tflops_val >= 0.01
            else "COM < 0.01 TFLOPS"
        )
        with contextlib.suppress(Exception):
            bar.unit = io_expr + ", " + com_expr
        try:
            inc = int(finish)
        except Exception:
            inc = 1
        if inc > 0:
            with contextlib.suppress(Exception):
                bar.update(inc)

@dataclasses.dataclass
class _PendingOp:
    kind: str
    epoch: int
    future: object | None
    started_monotonic: float
    epoch_dir: str | None = None
    has_optimizer: bool = False
    ok: bool = False


class Checkpointer:
    def __init__(
        self,
        ckpt_dir: PathLike,
        *args: Any,
        keep_last: int = 1,
        use_async: bool = True,
        dcp_subdir: str = "dcp_epochs",
        avg_subdir: str = "avg",
        avg_ext: str = ".pt",
        mmap_load: bool | None = None,
        cpu_offload: bool | None = None,
        device: torch.device | None = None,
    ) -> None:
        self._device = device
        self.root = Path(ckpt_dir)
        self.dcp_root = self.root / dcp_subdir
        self.avg_root = self.root / avg_subdir
        self.avg_ext = (
            avg_ext if str(avg_ext).startswith(".") else f".{avg_ext}"
        )
        self.keep_last = max(1, int(keep_last))
        self.max_in_flight = 1
        self.use_async = bool(use_async)
        self.mmap_load = mmap_load
        self._cpu_offload = cpu_offload

        self._pending_dcp: deque[_PendingOp] = deque()
        self._pending_avg: deque[_PendingOp] = deque()
        self._pending_lock = Mutex(reentrant=True)

        self._avg_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="enn-avg-ckpt"
        )
        self._dcp_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="enn-dcp-ckpt"
        )

        self._stager: object | None = None
        self._stager_lock = Mutex(reentrant=True)
        self._stager_owner_thread: int | None = None
        self._stager_closed = False

        try:
            import torch.distributed as dist

            self._dist = dist
            self._rank = dist.get_rank() if dist.is_initialized() else 0
            self._world = dist.get_world_size() if dist.is_initialized() else 1
        except Exception:
            self._dist = None
            self._rank = 0
            self._world = 1

        self._local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
        self._local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1") or 1)
        if self._local_world < 1:
            self._local_world = 1
        self._node_rank = (
            self._rank // self._local_world if self._local_world else 0
        )

        self._dcp_process_group: object | None = None
        self._dcp_should_participate: bool = True
        if device is not None and self._is_distributed():
            try:
                mesh, kind = get_distributed_mesh(device)
                if kind == "hsdp2" and mesh is not None:
                    coord = None
                    with contextlib.suppress(Exception):
                        coord = mesh.get_coordinate()
                    if isinstance(coord, tuple) and len(coord) >= 1:
                        self._dcp_should_participate = int(coord[0]) == 0

                    pg = None
                    with contextlib.suppress(Exception):
                        pg = mesh.get_group("dp_shard")
                    if pg is not None:
                        self._dcp_process_group = pg
            except Exception:
                self._dcp_process_group = None
                self._dcp_should_participate = True

        self.dcp_root.mkdir(parents=True, exist_ok=True)
        self.avg_root.mkdir(parents=True, exist_ok=True)
        try:
            self._dcp_inprogress_ttl_sec = max(
                0, int(env_int("ENN_DCP_INPROGRESS_TTL_SEC", 1800) or 1800)
            )
        except Exception:
            self._dcp_inprogress_ttl_sec = 1800
        try:
            self._dcp_inprogress_grace_sec = max(
                0, int(env_int("ENN_DCP_INPROGRESS_GRACE_SEC", 60) or 60)
            )
        except Exception:
            self._dcp_inprogress_grace_sec = 60

        self._inflight_lock = self.dcp_root / _INFLIGHT_LOCK_NAME
        self._inflight_mu = threading.Lock()

        self._sync_pg = None
        if self._is_distributed():
            try:
                self._sync_pg = dist.new_group(backend="gloo")
            except Exception:
                self._sync_pg = None

        self._cleanup_stale_inflight_lock()
        if self._is_dcp_leader():
            _cleanup_delete_pending(self.dcp_root)

    def _sync_group(self):
        if not self._is_distributed():
            return None
        return self._sync_pg if self._sync_pg is not None else dist.group.WORLD

    def _sync_device(self) -> str:
        if not self._is_distributed():
            return "cpu"
        pg = self._sync_group()
        try:
            backend = dist.get_backend(pg)
        except Exception:
            backend = None
        if backend == "nccl":
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"

    def _bcast_bool_from_leader(self, flag: bool) -> bool:
        if not self._is_distributed():
            return bool(flag)
        pg = self._sync_group()
        t = torch.tensor(
            [1 if flag else 0], device=self._sync_device(), dtype=torch.int32
        )
        dist.broadcast(t, src=0, group=pg)
        return bool(int(t.item()))

    def _is_dcp_leader(self) -> bool:
        if not self._is_distributed():
            return True
        pg = self._sync_group() or dist.group.WORLD
        try:
            return int(dist.get_rank(pg)) == 0
        except Exception:
            return self._is_global_rank0()

    def _cleanup_stale_inflight_lock(self) -> None:
        if not self._is_dcp_leader():
            return
        try:
            if not self._inflight_lock.exists():
                return
            if _INFLIGHT_LOCK_TTL_SEC <= 0:
                return
            age = time.time() - float(self._inflight_lock.stat().st_mtime)
            if age < float(_INFLIGHT_LOCK_TTL_SEC):
                return
            with contextlib.suppress(Exception):
                self._inflight_lock.unlink()
        except Exception:
            return

    def _try_acquire_inflight_lock(self, epoch: int) -> bool:
        if not self._is_dcp_leader():
            return False
        self._cleanup_stale_inflight_lock()
        payload = {"epoch": int(epoch), "pid": int(os.getpid()), "ts": time.time()}
        return _atomic_create_json(self._inflight_lock, payload)

    def _release_inflight_lock(self) -> None:
        if not self._is_dcp_leader():
            return
        with contextlib.suppress(Exception):
            if self._inflight_lock.exists():
                self._inflight_lock.unlink()

    def poll(self) -> None:
        with contextlib.suppress(Exception):
            self._cleanup_stale_inflight_lock()
        self._maybe_wait_for_budget(block=False)
        busy = False
        with contextlib.suppress(Exception):
            with self._pending_lock:
                busy = bool(self._pending_dcp)
        if self._is_dcp_leader():
            with contextlib.suppress(Exception):
                if self._inflight_lock.exists():
                    busy = True
        if not busy:
            with contextlib.suppress(Exception):
                self._cleanup_stale_dcp_inprogress()
        if self._is_dcp_leader():
            _cleanup_delete_pending(self.dcp_root)

    def is_idle(self) -> bool:
        self.poll()
        local_busy = False
        with self._pending_lock:
            local_busy = bool(self._pending_dcp) or bool(self._pending_avg)
        if self._is_dcp_leader():
            with contextlib.suppress(Exception):
                if self._inflight_lock.exists():
                    local_busy = True
        return not local_busy

    def _list_dcp_inprogress(self) -> list[Path]:
        out: list[Path] = []
        for p in self.dcp_root.glob("epoch_*"):
            if not p.is_dir():
                continue
            try:
                if self._epoch_done_file(p).is_file():
                    continue
                if self._epoch_failed_file(p).is_file():
                    continue
            except Exception:
                pass
            out.append(p)
        out.sort(key=lambda x: x.name)
        return out

    def _dcp_last_activity_time(self, epoch_dir: Path) -> float | None:
        last: float | None
        try:
            last = float(epoch_dir.stat().st_mtime)
        except Exception:
            last = None
        with contextlib.suppress(Exception):
            ip = self._epoch_inprogress_file(epoch_dir)
            if ip.is_file():
                mt = float(ip.stat().st_mtime)
                last = mt if last is None else max(last, mt)
        if not env_bool("ENN_DCP_INPROGRESS_SCAN_FILES", False):
            return last
        max_files = int(env_int("ENN_DCP_INPROGRESS_SCAN_FILES_MAX", 256) or 256)
        if max_files <= 0:
            return last
        try:
            seen = 0
            with os.scandir(epoch_dir) as it:
                for ent in it:
                    try:
                        st = ent.stat(follow_symlinks=False)
                        mt = float(st.st_mtime)
                        last = mt if last is None else max(last, mt)
                    except Exception:
                        pass
                    seen += 1
                    if seen >= max_files:
                        break
        except Exception:
            pass
        return last

    def _cleanup_stale_dcp_inprogress(self) -> None:
        if not self._is_global_rank0():
            return
        ttl = int(self._dcp_inprogress_ttl_sec)
        grace = int(self._dcp_inprogress_grace_sec)
        if ttl <= 0:
            return
        now = time.time()
        for p in self._list_dcp_inprogress():
            last = self._dcp_last_activity_time(p)
            if last is None:
                continue
            age = now - float(last)
            if age < max(1, grace):
                continue
            if env_bool("ENN_DCP_ORPHAN_FINALIZE", True):
                with contextlib.suppress(Exception):
                    if self._try_mark_orphan_complete(p):
                        continue
            if age < ttl:
                continue
            if env_bool("ENN_DCP_KEEP_FAILED", False):
                m = re.match(r"epoch_(\d+)", p.name)
                if m:
                    with contextlib.suppress(Exception):
                        write_json(
                            p / "failed.json",
                            {
                                "format": "enn-dcp-failed-v1",
                                "epoch": int(m.group(1)),
                                "time": now,
                                "reason": "stale_inprogress",
                            },
                            indent=2,
                        )
                continue
            with contextlib.suppress(Exception):
                shutil.rmtree(p, ignore_errors=True)

    def _wait_for_dcp_inprogress_slot(self) -> None:
        last_cleanup = 0.0
        try:
            self._cleanup_stale_dcp_inprogress()
        finally:
            last_cleanup = time.monotonic()
        while True:
            inprog = self._list_dcp_inprogress()
            if len(inprog) < int(self.max_in_flight):
                return
            if time.monotonic() - last_cleanup >= 5.0:
                self._cleanup_stale_dcp_inprogress()
                last_cleanup = time.monotonic()
            self._maybe_wait_for_budget()
            time.sleep(0.2)

    def _is_distributed(self) -> bool:
        return bool(self._dist is not None and self._dist.is_initialized())

    def _is_global_rank0(self) -> bool:
        return (not self._is_distributed()) or self._rank == 0

    def _is_local_rank0(self) -> bool:
        return (not self._is_distributed()) or self._local_rank == 0

    def _epoch_dir(self, epoch: int) -> Path:
        return self.dcp_root / f"epoch_{epoch:06d}"

    def _epoch_done_file(self, epoch_dir: Path) -> Path:
        return epoch_dir / "done.json"

    def _epoch_failed_file(self, epoch_dir: Path) -> Path:
        return epoch_dir / "failed.json"

    def _epoch_inprogress_file(self, epoch_dir: Path) -> Path:
        return epoch_dir / "inprogress.json"

    def _dcp_has_distcp_file(self, epoch_dir: Path) -> bool:
        try:
            with os.scandir(epoch_dir) as it:
                for ent in it:
                    try:
                        if ent.is_file(follow_symlinks=False) and ent.name.endswith(".distcp"):
                            return True
                    except Exception:
                        continue
        except Exception:
            return False
        return False

    def _pending_has_epoch_dir(self, epoch_dir: Path) -> bool:
        key = str(epoch_dir)
        with self._pending_lock:
            for op in self._pending_dcp:
                if op.epoch_dir and str(op.epoch_dir) == key:
                    return True
        return False

    def _default_dcp_model_kind(self) -> str:
        return os.environ.get("ENN_DCP_MODEL_KIND", "avg").strip().lower()

    def _cpu_offload_enabled(self) -> bool:
        v = getattr(self, "_cpu_offload", None)
        if isinstance(v, bool):
            return bool(v)
        if ("ENN_DCP_CPU_OFFLOAD" in os.environ) or (
            "ENN_CKPT_CPU_OFFLOAD" in os.environ
        ):
            return bool(
                env_bool(
                    ("ENN_DCP_CPU_OFFLOAD", "ENN_CKPT_CPU_OFFLOAD"),
                    default=True,
                )
            )
        try:
            if self._is_distributed() and int(getattr(self, "_world", 1) or 1) > 1:
                return True
        except Exception:
            pass
        try:
            dev = self._device or get_device()
            if getattr(dev, "type", None) == "cuda":
                max_gb = int(env_int("ENN_DCP_CPU_OFFLOAD_AUTO_MAX_GB", 24) or 24)
                max_gb = max(4, max_gb)
                total = None
                with contextlib.suppress(Exception):
                    total = Memory.total()
                if isinstance(total, int) and total > 0:
                    if (float(total) / (1024.0**3)) <= float(max_gb):
                        return False
                    return True
                return False
        except Exception:
            return False
        return True

    def _try_mark_orphan_complete(self, epoch_dir: Path) -> bool:
        if not self._is_global_rank0():
            return False
        try:
            if self._epoch_done_file(epoch_dir).is_file() or self._epoch_failed_file(epoch_dir).is_file():
                return False
        except Exception:
            pass
        if not self._dcp_has_distcp_file(epoch_dir):
            return False
        if self._pending_has_epoch_dir(epoch_dir):
            return False
        m = re.match(r"epoch_(\d+)", epoch_dir.name)
        if not m:
            return False
        epoch_i = int(m.group(1))
        has_opt = True
        model_kind = self._default_dcp_model_kind()
        try:
            ip = self._epoch_inprogress_file(epoch_dir)
            if ip.is_file():
                with open(ip, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                if isinstance(meta, dict):
                    has_opt = bool(meta.get("has_optimizer", True))
                    mk = meta.get("model_kind", None)
                    if isinstance(mk, str) and mk.strip():
                        model_kind = mk.strip().lower()
        except Exception:
            pass
        self._finalize_dcp_epoch(
            epoch_i,
            epoch_dir,
            extra_meta={"has_optimizer": bool(has_opt), "model_kind": str(model_kind)},
        )
        return True

    def _avg_node_dir(self, node_rank: int | None = None) -> Path:
        nr = self._node_rank if node_rank is None else int(node_rank)
        return self.avg_root / f"node_{nr:04d}"

    def _avg_epoch_path(
        self, epoch: int, node_rank: int | None = None
    ) -> Path:
        d = self._avg_node_dir(node_rank)
        return d / f"epoch_{epoch:06d}{self.avg_ext}"

    def _avg_latest_file(self, node_rank: int | None = None) -> Path:
        return self._avg_node_dir(node_rank) / "latest.json"

    def _ensure_stager(self) -> object | None:
        if self._stager_closed:
            return None
        tid = threading.get_ident()
        with self._stager_lock:
            if self._stager is not None and self._stager_owner_thread == tid:
                return self._stager
            if self._stager is not None:
                with contextlib.suppress(Exception):
                    close = getattr(self._stager, "close", None)
                    if callable(close):
                        close()
                self._stager = None
                self._stager_owner_thread = None

            try:
                from torch.distributed.checkpoint.staging import DefaultStager
                from torch.distributed.checkpoint.staging import StagingOptions

                use_pinned_memory = bool(
                    self._device is not None and self._device.type == "cuda"
                )
                opts = StagingOptions(
                    use_pinned_memory=use_pinned_memory,
                    use_shared_memory=False,
                    use_async_staging=True,
                    use_non_blocking_copy=True,
                )
                self._stager = DefaultStager(config=opts)
                self._stager_owner_thread = tid
            except Exception:
                self._stager = None
                self._stager_owner_thread = None
            return self._stager

    def _close_stager(self) -> None:
        with self._stager_lock:
            if self._stager is None:
                return
            with contextlib.suppress(Exception):
                close = getattr(self._stager, "close", None)
                if callable(close):
                    close()
            self._stager = None
            self._stager_owner_thread = None

    def _post_dcp_cleanup(self) -> None:
        with contextlib.suppress(Exception):
            import torch.distributed.checkpoint.state_dict_saver as sds

            close_fn = getattr(sds, "close", None)
            if callable(close_fn):
                close_fn()
        self._close_stager()
        do_gc = bool(env_bool("ENN_DCP_GC_COLLECT", default=False))
        do_trim = bool(env_bool("ENN_DCP_MALLOC_TRIM", default=False))
        if not (do_gc or do_trim):
            return
        if do_gc:
            with contextlib.suppress(Exception):
                gc.collect()
        if do_trim:
            with contextlib.suppress(Exception):
                libc = ctypes.CDLL("libc.so.6")
                trim = getattr(libc, "malloc_trim", None)
                if callable(trim):
                    trim(0)

    def _maybe_finalize_dcp_op(self, op: _PendingOp) -> None:
        if op.kind != "dcp" or not op.epoch_dir:
            return
        try:
            epoch_dir = Path(op.epoch_dir)
        except Exception:
            return
        try:
            if self._epoch_done_file(epoch_dir).is_file():
                return
        except Exception:
            pass
        self._finalize_dcp_epoch(
            int(op.epoch),
            epoch_dir,
            extra_meta={"has_optimizer": bool(op.has_optimizer)},
        )

    def _housekeep_completed_dcp(self, op: _PendingOp) -> None:
        if op.kind != "dcp" or not op.epoch_dir:
            return
        try:
            epoch_dir = Path(op.epoch_dir)
        except Exception:
            return
        try:
            if not self._epoch_done_file(epoch_dir).is_file():
                self._finalize_dcp_epoch(
                    int(op.epoch),
                    epoch_dir,
                    extra_meta={"has_optimizer": bool(op.has_optimizer)},
                )
            else:
                self._prune_dcp()
        except Exception as exc:
            _LOGGER.warning(
                "DCP housekeeping failed for epoch=%d: %s", int(op.epoch), exc
            )

    def _maybe_wait_for_budget(self, *, block: bool = True) -> None:
        done_ops: list[tuple[_PendingOp, bool, str | None]] = []
        with self._pending_lock:
            while self._pending_dcp and _future_done(self._pending_dcp[0].future):
                op = self._pending_dcp[0]
                fut = op.future
                self._pending_lock.release()
                ok = True
                err: str | None = None
                try:
                    _future_result(fut)
                except Exception as exc:
                    ok = False
                    err = f"{type(exc).__name__}: {exc}"
                    if self._is_global_rank0():
                        _LOGGER.exception(
                            "DCP async_save failed (epoch=%d): %s",
                            int(op.epoch),
                            exc,
                        )
                finally:
                    self._pending_lock.acquire()
                if self._pending_dcp and self._pending_dcp[0] is op:
                    self._pending_dcp.popleft()
                    with contextlib.suppress(Exception):
                        op.future = None
                    done_ops.append((op, ok, err))
                    self._post_dcp_cleanup()
                else:
                    break
            while self._pending_avg and _future_done(
                self._pending_avg[0].future
            ):
                self._pending_avg.popleft()

            if block:
                while len(self._pending_dcp) >= self.max_in_flight:
                    op = self._pending_dcp[0]
                    fut = op.future
                    self._pending_lock.release()
                    ok = True
                    err: str | None = None
                    try:
                        _future_result(fut)
                    except Exception as exc:
                        ok = False
                        err = f"{type(exc).__name__}: {exc}"
                        if self._is_global_rank0():
                            _LOGGER.exception(
                                "DCP async_save failed while waiting budget (epoch=%d): %s",
                                int(op.epoch),
                                exc,
                            )
                    finally:
                        self._pending_lock.acquire()
                    removed = False
                    if self._pending_dcp and self._pending_dcp[0] is op:
                        self._pending_dcp.popleft()
                        removed = True
                    else:
                        with contextlib.suppress(ValueError):
                            self._pending_dcp.remove(op)
                            removed = True
                    if removed:
                        with contextlib.suppress(Exception):
                            op.future = None
                        done_ops.append((op, ok, err))
                        self._post_dcp_cleanup()

            while len(self._pending_avg) >= 1:
                op = self._pending_avg[0]
                fut = op.future
                self._pending_lock.release()
                try:
                    _future_result(fut)
                finally:
                    self._pending_lock.acquire()
                with contextlib.suppress(Exception):
                    op.future = None
                if self._pending_avg and self._pending_avg[0] is op:
                    self._pending_avg.popleft()
                while self._pending_avg and _future_done(
                    self._pending_avg[0].future
                ):
                    self._pending_avg.popleft()

        for op, ok, err in done_ops:
            if op.kind != "dcp":
                continue
            if ok:
                self._maybe_finalize_success_epoch(
                    int(op.epoch), bool(op.has_optimizer)
                )
            else:
                self._cleanup_failed_epoch_dir(int(op.epoch), reason=err)

    def _register_pending(
        self,
        kind: str,
        epoch: int,
        fut: object | None,
        *,
        epoch_dir: str | None = None,
        has_optimizer: bool = False,
    ) -> None:
        with self._pending_lock:
            q = self._pending_dcp if kind == "dcp" else self._pending_avg
            q.append(
                _PendingOp(
                    kind=kind,
                    epoch=int(epoch),
                    future=fut,
                    started_monotonic=time.monotonic(),
                    epoch_dir=epoch_dir,
                    has_optimizer=bool(has_optimizer),
                )
            )

    def _cleanup_failed_epoch_dir(
        self, epoch: int, reason: str | None = None
    ) -> None:
        keep_failed = env_bool("ENN_DCP_KEEP_FAILED", False)
        epoch_dir = self._epoch_dir(int(epoch))
        if keep_failed:
            if self._is_global_rank0():
                try:
                    payload = {
                        "format": "enn-dcp-failed-v1",
                        "epoch": int(epoch),
                        "time": time.time(),
                        "reason": str(reason or ""),
                    }
                    write_json(epoch_dir / "failed.json", payload, indent=2)
                except Exception:
                    pass
            return
        if self._is_global_rank0():
            with contextlib.suppress(Exception):
                shutil.rmtree(epoch_dir, ignore_errors=True)

    def _maybe_finalize_success_epoch(
        self, epoch: int, has_optimizer: bool
    ) -> None:
        epoch_dir = self._epoch_dir(int(epoch))
        try:
            if self._epoch_done_file(epoch_dir).is_file():
                self._prune_dcp()
                return
            if self._epoch_failed_file(epoch_dir).is_file():
                return
            if (not epoch_dir.is_dir()) or (not self._dcp_has_distcp_file(epoch_dir)):
                return
            self._finalize_dcp_epoch(
                int(epoch),
                epoch_dir,
                extra_meta={"has_optimizer": bool(has_optimizer)},
            )
        except Exception as exc:
            if env_bool("ENN_DCP_DEBUG", False) and self._is_global_rank0():
                _LOGGER.warning(
                    "DCP finalize/prune failed (epoch=%d): %s",
                    int(epoch),
                    exc,
                )

    def _finalize_dcp_epoch(
        self, epoch: int, epoch_dir: Path, extra_meta: dict[str, Any]
    ) -> None:
        if not self._is_global_rank0():
            return
        try:
            with contextlib.suppress(Exception):
                ip = self._epoch_inprogress_file(epoch_dir)
                if ip.exists():
                    ip.unlink()
            done_file = self._epoch_done_file(epoch_dir)
            payload = {
                "format": "enn-dcp-epoch-v1",
                "epoch": int(epoch),
                "created_time": time.time(),
                "rank": int(self._rank),
                "world_size": int(self._world),
                "node_rank": int(self._node_rank),
                **(extra_meta or {}),
            }
            write_json(done_file, payload, indent=2)
        finally:
            with contextlib.suppress(Exception):
                self._prune_dcp()

    def _prune_dcp(self) -> None:
        if not self._is_global_rank0():
            return
        try:
            epoch_dirs: list[Path] = []
            for p in self.dcp_root.glob("epoch_*"):
                if not p.is_dir():
                    continue
                if self._epoch_done_file(p).is_file():
                    epoch_dirs.append(p)
            epoch_dirs.sort(key=lambda x: x.name)
            if len(epoch_dirs) <= self.keep_last:
                return
            for p in epoch_dirs[: max(0, len(epoch_dirs) - self.keep_last)]:
                try:
                    _safe_rmtree(p)
                except Exception as exc:
                    if env_bool("ENN_DCP_LOG_PRUNE", False):
                        _LOGGER.warning("DCP prune failed for %s: %s", str(p), exc)
        except Exception:
            return

    def _prune_avg(self, node_rank: int | None = None) -> None:
        d = self._avg_node_dir(node_rank)
        if not d.is_dir():
            return
        try:
            files = sorted(
                d.glob(f"epoch_*{self.avg_ext}"), key=lambda x: x.name
            )
            if len(files) <= self.keep_last:
                return
            for p in files[: max(0, len(files) - self.keep_last)]:
                with contextlib.suppress(Exception):
                    p.unlink()
        except Exception:
            return

    def _save_avg_epoch(
        self, epoch: int, avg_state_dict: Mapping[str, Any]
    ) -> Path:
        node_rank = int(self._node_rank)
        d = self._avg_node_dir(node_rank)
        d.mkdir(parents=True, exist_ok=True)
        payload = _coerce_dcp_keys(avg_state_dict)
        path = self._avg_epoch_path(epoch, node_rank)
        save_temp(path, payload)
        meta = {
            "format": "enn-avg-state-v1",
            "epoch": int(epoch),
            "created_time": time.time(),
            "rank": int(self._rank),
            "local_rank": int(self._local_rank),
            "node_rank": int(node_rank),
            "world_size": int(self._world),
            "local_world_size": int(self._local_world),
        }
        write_json(self._avg_latest_file(node_rank), meta, indent=2)
        with contextlib.suppress(Exception):
            self._prune_avg(node_rank)
        return path

    def _schedule_avg_save(
        self, epoch: int, avg_state_dict: Mapping[str, Any]
    ) -> None:
        try:
            future = self._avg_executor.submit(
                self._save_avg_epoch, int(epoch), avg_state_dict
            )
        except Exception as exc:
            _LOGGER.exception("Average checkpoint scheduling failed: %s", exc)
            return
        self._register_pending("avg", int(epoch), future)

    def _background_dcp_enabled(self) -> bool:
        try:
            world = int(getattr(self, "_world", 1) or 1)
        except Exception:
            world = 1
        default_bg = world <= 1
        enabled = bool(env_bool("ENN_DCP_BACKGROUND", default=default_bg))
        return bool(enabled and world <= 1)

    def _monitor_dcp_future(
        self,
        *,
        epoch: int,
        epoch_dir: str,
        dcp_future: object,
        has_optimizer: bool,
        model_kind: str,
    ) -> None:
        ok = True
        err: str | None = None
        pending_exc: Exception | None = None
        try:
            _future_result(dcp_future)
        except Exception as exc:
            ok = False
            err = f"{type(exc).__name__}: {exc}"
            pending_exc = exc
            if self._is_global_rank0():
                _LOGGER.exception(
                    "DCP async_save failed (epoch=%d): %s", int(epoch), exc
                )
        try:
            if self._is_dcp_leader():
                if ok:
                    self._finalize_dcp_epoch(
                        int(epoch),
                        Path(epoch_dir),
                        extra_meta={
                            "has_optimizer": bool(has_optimizer),
                            "model_kind": str(model_kind),
                        },
                    )
                    _cleanup_delete_pending(self.dcp_root)
                else:
                    self._cleanup_failed_epoch_dir(int(epoch), reason=err)
        finally:
            with contextlib.suppress(Exception):
                self._release_inflight_lock()
            self._post_dcp_cleanup()
        if pending_exc is not None:
            raise pending_exc

    def _save_dcp_epoch_background(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        avg_state_dict: Mapping[str, Any] | None = None,
        avg_state_dict_factory: Callable[[], Mapping[str, Any] | None] | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        epoch_i = int(epoch)
        epoch_dir = self._epoch_dir(epoch_i)
        pending_exc: Exception | None = None
        try:
            if (
                avg_state_dict is None
                and avg_state_dict_factory is not None
                and self._is_local_rank0()
            ):
                try:
                    avg_state_dict = avg_state_dict_factory()
                except Exception as exc:
                    if self._is_global_rank0():
                        _LOGGER.exception(
                            "Average snapshot build failed (epoch=%d): %s",
                            int(epoch_i),
                            exc,
                        )
                    avg_state_dict = None

            if avg_state_dict is not None and self._is_local_rank0():
                self._schedule_avg_save(epoch_i, avg_state_dict)

            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import StateDictOptions
            from torch.distributed.checkpoint.state_dict import get_state_dict
            from torch.distributed.checkpoint import FileSystemWriter

            epoch_dir.mkdir(parents=True, exist_ok=True)

            want_kind = self._default_dcp_model_kind()
            want_avg = want_kind in ("avg", "average", "ema", "swa")

            save_opt = bool(optimizer is not None) and env_bool(
                ("ENN_DCP_SAVE_OPTIMIZER", "ENN_CKPT_SAVE_OPTIMIZER"), default=True
            )
            try:
                optim_every = int(
                    env_int(
                        "ENN_DCP_OPTIM_EVERY_EPOCHS",
                        env_int("ENN_DCP_OPTIM_EVERY", 1) or 1,
                    )
                    or 1
                )
            except Exception:
                optim_every = 1
            optim_every = max(1, int(optim_every))
            if save_opt and optim_every > 1 and (int(epoch_i) % int(optim_every)) != 0:
                save_opt = False

            cpu_offload = bool(self._cpu_offload_enabled())
            supports_cpu_offload = True
            try:
                opts = StateDictOptions(
                    full_state_dict=False, cpu_offload=bool(cpu_offload)
                )
            except TypeError:
                supports_cpu_offload = False
                opts = StateDictOptions(full_state_dict=False)

            model_sd, optim_sd = get_state_dict(
                model, (optimizer if save_opt else []), options=opts
            )

            save_model_sd: object = model_sd
            if want_avg and avg_state_dict is not None:
                if supports_cpu_offload and cpu_offload:
                    _overlay_avg_state_dict(model_sd, avg_state_dict)
                    save_model_sd = model_sd
                else:
                    default_direct = bool(
                        int(self._world) <= 1 and self._dcp_process_group is None
                    )
                    direct_avg = env_bool(
                        "ENN_DCP_AVG_DIRECT", default=default_direct
                    )
                    if direct_avg:
                        save_model_sd = dict(avg_state_dict)
                    else:
                        dev_t = str(
                            getattr(getattr(self, "_device", None), "type", "cpu")
                            or "cpu"
                        )
                        default_to_cpu = dev_t in ("cuda", "xpu", "mps")
                        to_cpu = env_bool(
                            "ENN_DCP_CLONE_TO_CPU", default=default_to_cpu
                        )
                        save_model_sd = _clone_state_dict(
                            model_sd, to_cpu=bool(to_cpu)
                        )
                        _overlay_avg_state_dict(save_model_sd, avg_state_dict)

            dcp_state: dict[str, Any] = {"model": _coerce_dcp_keys(save_model_sd)}
            if save_opt:
                dcp_state["optimizer"] = _coerce_dcp_keys(optim_sd)
            if extra_state is not None:
                dcp_state["extra"] = _coerce_dcp_keys(extra_state)

            if self._is_global_rank0():
                with contextlib.suppress(Exception):
                    write_json(
                        self._epoch_inprogress_file(epoch_dir),
                        {
                            "format": "enn-dcp-inprogress-v1",
                            "epoch": int(epoch_i),
                            "created_time": time.time(),
                            "has_optimizer": bool(save_opt),
                            "model_kind": ("avg" if want_avg else "active"),
                        },
                        indent=2,
                    )

            try:
                sync_files = bool(env_bool("ENN_DCP_SYNC_FILES", default=False))
                writer = FileSystemWriter(
                    str(epoch_dir), sync_files=sync_files, overwrite=True
                )
            except TypeError:
                writer = FileSystemWriter(str(epoch_dir))

            planner: object | None = None
            with contextlib.suppress(Exception):
                from torch.distributed.checkpoint import DefaultSavePlanner

                planner = DefaultSavePlanner(dedup_save_to_lowest_rank=True)

            dcp.save(
                state_dict=dcp_state,
                checkpoint_id=str(epoch_dir),
                storage_writer=writer,
                planner=planner,
                process_group=self._dcp_process_group,
            )

            if self._is_dcp_leader():
                self._finalize_dcp_epoch(
                    epoch_i,
                    epoch_dir,
                    extra_meta={
                        "has_optimizer": bool(save_opt),
                        "model_kind": ("avg" if want_avg else "active"),
                    },
                )
                _cleanup_delete_pending(self.dcp_root)
        except Exception as exc:
            pending_exc = exc
            _LOGGER.exception("DCP epoch checkpoint failed: %s", exc)
            with contextlib.suppress(Exception):
                if self._is_dcp_leader():
                    self._cleanup_failed_epoch_dir(
                        int(epoch_i), reason=f"{type(exc).__name__}: {exc}"
                    )
        finally:
            with contextlib.suppress(Exception):
                self._release_inflight_lock()
            self._post_dcp_cleanup()
        if pending_exc is not None:
            raise pending_exc

    def request_save_epoch(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        avg_state_dict: Mapping[str, Any] | None = None,
        avg_state_dict_factory: Callable[[], Mapping[str, Any] | None] | None = None,
        extra_state: dict[str, Any] | None = None,
        save_optimizer: bool | None = None,
        save_avg: bool | None = None,
        force_sync: bool = False,
        block_if_busy: bool = False,
    ) -> None:
        epoch_i = int(epoch)

        self._maybe_wait_for_budget(block=False)
        with contextlib.suppress(Exception):
            self._cleanup_stale_dcp_inprogress()

        accept = True
        if self._is_dcp_leader():
            if not self._dcp_should_participate:
                accept = False
            with self._pending_lock:
                if self._pending_dcp:
                    accept = False
            with contextlib.suppress(Exception):
                if self._inflight_lock.exists():
                    accept = False
            if accept:
                accept = self._try_acquire_inflight_lock(epoch_i)

        accept = self._bcast_bool_from_leader(bool(accept))

        if not self._dcp_should_participate:
            return
        if not accept:
            if block_if_busy:
                while True:
                    time.sleep(0.2)
                    self.poll()
                    accept_local = False
                    if self._is_dcp_leader():
                        with self._pending_lock:
                            accept_local = not bool(self._pending_dcp)
                        with contextlib.suppress(Exception):
                            accept_local = accept_local and (
                                not self._inflight_lock.exists()
                            )
                        if accept_local:
                            accept_local = self._try_acquire_inflight_lock(
                                epoch_i
                            )
                    accept = self._bcast_bool_from_leader(bool(accept_local))
                    if accept:
                        break
            else:
                return

        if self._background_dcp_enabled():
            try:
                fut = self._dcp_executor.submit(
                    self._save_dcp_epoch_background,
                    epoch=epoch_i,
                    model=model,
                    optimizer=optimizer,
                    avg_state_dict=avg_state_dict,
                    avg_state_dict_factory=avg_state_dict_factory,
                    extra_state=extra_state,
                )
                self._register_pending(
                    "dcp",
                    epoch_i,
                    fut,
                    epoch_dir=str(self._epoch_dir(epoch_i)),
                    has_optimizer=bool(optimizer is not None),
                )
            except Exception as exc:
                _LOGGER.exception(
                    "DCP background scheduling failed (epoch=%d): %s",
                    int(epoch_i),
                    exc,
                )
                with contextlib.suppress(Exception):
                    self._release_inflight_lock()
                self._post_dcp_cleanup()
                if force_sync:
                    raise
            return

        if (
            avg_state_dict is None
            and avg_state_dict_factory is not None
            and self._is_local_rank0()
        ):
            try:
                avg_state_dict = avg_state_dict_factory()
            except Exception as exc:
                if self._is_global_rank0():
                    _LOGGER.exception(
                        "Average snapshot build failed (epoch=%d): %s",
                        int(epoch_i),
                        exc,
                    )
                avg_state_dict = None

        if avg_state_dict is not None and self._is_local_rank0():
            self._schedule_avg_save(epoch_i, avg_state_dict)

        dcp_future: object | None = None
        try:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint import FileSystemWriter
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
            )
            from torch.distributed.checkpoint.state_dict import get_state_dict

            epoch_dir = self._epoch_dir(epoch_i)
            epoch_dir.mkdir(parents=True, exist_ok=True)

            want_kind = self._default_dcp_model_kind()
            want_avg = want_kind in ("avg", "average", "ema", "swa")

            save_opt = bool(optimizer is not None) and env_bool(
                ("ENN_DCP_SAVE_OPTIMIZER", "ENN_CKPT_SAVE_OPTIMIZER"), default=True
            )
            try:
                optim_every = int(
                    env_int(
                        "ENN_DCP_OPTIM_EVERY_EPOCHS",
                        env_int("ENN_DCP_OPTIM_EVERY", 1) or 1,
                    )
                    or 1
                )
            except Exception:
                optim_every = 1
            optim_every = max(1, int(optim_every))
            if save_opt and optim_every > 1 and (int(epoch_i) % int(optim_every)) != 0:
                save_opt = False

            cpu_offload = bool(self._cpu_offload_enabled())
            supports_cpu_offload = True
            try:
                opts = StateDictOptions(
                    full_state_dict=False, cpu_offload=bool(cpu_offload)
                )
            except TypeError:
                supports_cpu_offload = False
                opts = StateDictOptions(full_state_dict=False)
            model_sd, optim_sd = get_state_dict(
                model, (optimizer if save_opt else []), options=opts
            )

            save_model_sd: object = model_sd
            if want_avg and avg_state_dict is not None:
                if supports_cpu_offload and cpu_offload:
                    _overlay_avg_state_dict(model_sd, avg_state_dict)
                    save_model_sd = model_sd
                else:
                    default_direct = bool(int(self._world) <= 1 and self._dcp_process_group is None)
                    direct_avg = env_bool("ENN_DCP_AVG_DIRECT", default=default_direct)
                    if direct_avg:
                        save_model_sd = dict(avg_state_dict)
                    else:
                        dev_t = str(getattr(getattr(self, "_device", None), "type", "cpu") or "cpu")
                        default_to_cpu = dev_t in ("cuda", "xpu", "mps")
                        to_cpu = env_bool("ENN_DCP_CLONE_TO_CPU", default=default_to_cpu)
                        save_model_sd = _clone_state_dict(model_sd, to_cpu=bool(to_cpu))
                        _overlay_avg_state_dict(save_model_sd, avg_state_dict)

            dcp_state: dict[str, Any] = {
                "model": _coerce_dcp_keys(save_model_sd)
            }
            if save_opt:
                dcp_state["optimizer"] = _coerce_dcp_keys(optim_sd)
            if extra_state is not None:
                dcp_state["extra"] = _coerce_dcp_keys(extra_state)

            if self._is_global_rank0():
                with contextlib.suppress(Exception):
                    write_json(
                        self._epoch_inprogress_file(epoch_dir),
                        {
                            "format": "enn-dcp-inprogress-v1",
                            "epoch": int(epoch_i),
                            "created_time": time.time(),
                            "has_optimizer": bool(save_opt),
                            "model_kind": ("avg" if want_avg else "active"),
                        },
                        indent=2,
                    )

            sync_files = bool(env_bool("ENN_DCP_SYNC_FILES", default=False))
            try:
                writer = FileSystemWriter(str(epoch_dir), sync_files=sync_files)
            except TypeError:
                writer = FileSystemWriter(str(epoch_dir))
            planner: object | None = None
            with contextlib.suppress(Exception):
                from torch.distributed.checkpoint import DefaultSavePlanner

                planner = DefaultSavePlanner(dedup_save_to_lowest_rank=True)

            if self.use_async and hasattr(dcp, "async_save"):
                stager = self._ensure_stager()
                kwargs: dict[str, Any] = {
                    "state_dict": dcp_state,
                    "checkpoint_id": str(epoch_dir),
                    "storage_writer": writer,
                    "planner": planner,
                    "process_group": self._dcp_process_group,
                }
                if stager is not None:
                    kwargs["async_stager"] = stager

                with contextlib.suppress(Exception):
                    from torch.distributed.checkpoint.state_dict_saver import (
                        AsyncCheckpointerType,
                    )

                    async_type = (
                        os.environ.get("ENN_DCP_ASYNC_TYPE", "thread")
                        .strip()
                        .lower()
                    )
                    if async_type == "process":
                        kwargs["async_checkpointer_type"] = (
                            AsyncCheckpointerType.PROCESS
                        )
                    elif async_type == "thread":
                        kwargs["async_checkpointer_type"] = (
                            AsyncCheckpointerType.THREAD
                        )

                try:
                    sig = inspect.signature(dcp.async_save)
                    supported = set(sig.parameters.keys())
                    kwargs = {
                        k: v
                        for k, v in kwargs.items()
                        if k in supported and v is not None
                    }
                except Exception:
                    kwargs = {k: v for k, v in kwargs.items() if v is not None}

                dcp_future = dcp.async_save(**kwargs)
            else:
                dcp.save(
                    state_dict=dcp_state,
                    checkpoint_id=str(epoch_dir),
                    storage_writer=writer,
                    planner=planner,
                    process_group=self._dcp_process_group,
                )
                dcp_future = None

            if dcp_future is not None:
                monitor = None
                try:
                    monitor = self._dcp_executor.submit(
                        self._monitor_dcp_future,
                        epoch=epoch_i,
                        epoch_dir=str(epoch_dir),
                        dcp_future=dcp_future,
                        has_optimizer=bool(save_opt),
                        model_kind=("avg" if want_avg else "active"),
                    )
                except Exception as exc:
                    if self._is_global_rank0():
                        _LOGGER.exception(
                            "DCP monitor scheduling failed (epoch=%d): %s",
                            int(epoch_i),
                            exc,
                        )
                    monitor = None

                if monitor is not None:
                    self._register_pending(
                        "dcp",
                        epoch_i,
                        monitor,
                        epoch_dir=str(epoch_dir),
                        has_optimizer=bool(save_opt),
                    )
                else:
                    try:
                        _future_result(dcp_future)
                        if self._is_dcp_leader():
                            self._finalize_dcp_epoch(
                                epoch_i,
                                epoch_dir,
                                extra_meta={
                                    "has_optimizer": bool(save_opt),
                                    "model_kind": ("avg" if want_avg else "active"),
                                },
                            )
                            _cleanup_delete_pending(self.dcp_root)
                    finally:
                        self._release_inflight_lock()
                        self._post_dcp_cleanup()
            else:
                try:
                    self._finalize_dcp_epoch(
                        epoch_i,
                        epoch_dir,
                        extra_meta={
                            "has_optimizer": bool(save_opt),
                            "model_kind": ("avg" if want_avg else "active"),
                        },
                    )
                    if self._is_dcp_leader():
                        _cleanup_delete_pending(self.dcp_root)
                finally:
                    self._release_inflight_lock()
                    self._post_dcp_cleanup()
        except Exception as exc:
            _LOGGER.exception("DCP epoch checkpoint failed: %s", exc)
            with contextlib.suppress(Exception):
                self._cleanup_failed_epoch_dir(int(epoch_i), reason=f"{type(exc).__name__}: {exc}")
            with contextlib.suppress(Exception):
                self._release_inflight_lock()
            self._post_dcp_cleanup()
            if force_sync:
                raise

    def wait(self) -> None:
        pending: list[_PendingOp]
        with self._pending_lock:
            pending = list(self._pending_dcp) + list(self._pending_avg)
        for op in pending:
            ok = True
            err: str | None = None
            try:
                _future_result(op.future)
            except Exception as exc:
                ok = False
                err = f"{type(exc).__name__}: {exc}"
                if self._is_global_rank0() and op.kind == "dcp":
                    _LOGGER.exception(
                        "Pending DCP save failed during wait (epoch=%d): %s",
                        int(op.epoch),
                        exc,
                    )
            finally:
                with contextlib.suppress(Exception):
                    op.future = None

            if op.kind == "dcp":
                if ok:
                    self._maybe_finalize_success_epoch(
                        int(op.epoch), bool(op.has_optimizer)
                    )
                else:
                    self._cleanup_failed_epoch_dir(int(op.epoch), reason=err)
                self._post_dcp_cleanup()
        with self._pending_lock:
            self._pending_dcp.clear()
            self._pending_avg.clear()

    def abort_inflight(self) -> None:
        with self._pending_lock:
            pending = list(self._pending_dcp) + list(self._pending_avg)
            self._pending_dcp.clear()
            self._pending_avg.clear()

        for op in pending:
            fut = getattr(op, "future", None)
            if fut is None:
                continue
            cancel_fn = getattr(fut, "cancel", None)
            if callable(cancel_fn):
                with contextlib.suppress(Exception):
                    cancel_fn()
            with contextlib.suppress(Exception):
                op.future = None

        with contextlib.suppress(Exception):
            import torch.distributed.checkpoint.state_dict_saver as sds
            close_fn = getattr(sds, "close", None)
            if callable(close_fn):
                close_fn()

        if self._is_dcp_leader():
            self._release_inflight_lock()
            with contextlib.suppress(Exception):
                for p in self._list_dcp_inprogress():
                    _safe_rmtree(p)
            _cleanup_delete_pending(self.dcp_root)

    def close(self, *, abort_inflight: bool = True) -> None:
        if abort_inflight and (not self.is_idle()):
            self.abort_inflight()
        else:
            self.wait()
        with contextlib.suppress(Exception):
            import torch.distributed.checkpoint.state_dict_saver as sds

            close_fn = getattr(sds, "close", None)
            if callable(close_fn):
                close_fn()
        with self._stager_lock:
            self._stager_closed = True
            if self._stager is not None:
                with contextlib.suppress(Exception):
                    close = getattr(self._stager, "close", None)
                    if callable(close):
                        close()
                self._stager = None
                self._stager_owner_thread = None
        with contextlib.suppress(Exception):
            self._avg_executor.shutdown(wait=True)
        with contextlib.suppress(Exception):
            self._dcp_executor.shutdown(wait=True)

    def find_latest_dcp_epoch(self) -> int | None:
        try:
            completed: list[int] = []
            for p in self.dcp_root.glob("epoch_*"):
                if not p.is_dir():
                    continue
                if self._epoch_done_file(p).is_file():
                    m = re.match(r"epoch_(\\d+)", p.name)
                    if m:
                        completed.append(int(m.group(1)))
            return max(completed) if completed else None
        except Exception:
            return None

    def load_latest_dcp(
        self,
        *,
        model: nn.Module,
        optimizer: object | None = None,
        strict: bool = False,
    ) -> int | None:
        latest = self.find_latest_dcp_epoch()
        if latest is None:
            return None
        epoch_dir = self._epoch_dir(latest)
        if not self._epoch_done_file(epoch_dir).is_file():
            return None
        try:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint import FileSystemReader
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
            )
            from torch.distributed.checkpoint.state_dict import (
                get_model_state_dict,
            )
            from torch.distributed.checkpoint.state_dict import (
                get_optimizer_state_dict,
            )
            from torch.distributed.checkpoint.state_dict import (
                set_model_state_dict,
            )
            from torch.distributed.checkpoint.state_dict import (
                set_optimizer_state_dict,
            )

            opts = StateDictOptions(full_state_dict=False, strict=bool(strict))
            model_sd = get_model_state_dict(model, options=opts)
            state: dict[str, Any] = {"model": model_sd}
            optim_sd: Any | None = None
            if optimizer is not None:
                optim_sd = get_optimizer_state_dict(
                    model, optimizer, options=opts
                )
                state["optimizer"] = optim_sd
            dcp.load(
                state_dict=state,
                storage_reader=FileSystemReader(str(epoch_dir)),
            )
            set_model_state_dict(model, model_sd, options=opts)
            if optimizer is not None and optim_sd is not None:
                set_optimizer_state_dict(
                    model, optimizer, optim_sd, options=opts
                )
            return latest
        except Exception as exc:
            _LOGGER.exception("DCP load failed: %s", exc)
            if strict:
                raise
            return None

    def load_model_from_torchsave_broadcast(
        self,
        *,
        model: nn.Module,
        torch_save_path: PathLike,
        strict: bool = False,
    ) -> None:
        try:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.format_utils import (
                BroadcastingTorchSaveReader,
            )
            from torch.distributed.checkpoint.format_utils import (
                DynamicMetaLoadPlanner,
            )
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
            )
            from torch.distributed.checkpoint.state_dict import (
                get_model_state_dict,
            )
            from torch.distributed.checkpoint.state_dict import (
                set_model_state_dict,
            )

            opts = StateDictOptions(full_state_dict=False, strict=bool(strict))
            model_sd = get_model_state_dict(model, options=opts)
            dcp.load(
                state_dict={"model": model_sd},
                storage_reader=BroadcastingTorchSaveReader(),
                planner=DynamicMetaLoadPlanner(),
                checkpoint_id=str(torch_save_path),
            )
            set_model_state_dict(model, model_sd, options=opts)
        except Exception as exc:
            _LOGGER.exception("Broadcasting TorchSave load failed: %s", exc)
            if strict:
                raise
