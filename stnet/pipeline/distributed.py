# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import ipaddress
import os
import socket
import time
from urllib.parse import urlparse
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch.distributed as dist

try:
    from ..connection.socket import ArrowFlight, FlightModule
except ImportError:

    class _ArrowFlightUnavailable:
        @staticmethod
        def start_server_standby(*args: Any, **kwargs: Any) -> Any:
            details = " with args/kwargs" if args or kwargs else ""
            raise RuntimeError(
                "Arrow Flight server is unavailable"
                f"{details}. Install pyarrow[flight] to enable it."
            )

    class FlightModule:
        @staticmethod
        def connect(*args: Any, **kwargs: Any) -> None:
            details = " with args/kwargs" if args or kwargs else ""
            raise RuntimeError(f"Arrow Flight module is unavailable{details}.")

    ArrowFlight = _ArrowFlightUnavailable()

from ..toolkit.capability import get_available_addr, get_world_size


GRPC_DEFAULT_PORT = 5005


def is_initialized() -> bool:
    try:
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def _require_dist() -> None:
    if not is_initialized():
        raise RuntimeError("torch.distributed not initialized")


def _world_size() -> int:
    _require_dist()
    return get_world_size()


def _rank() -> int:
    _require_dist()
    return dist.get_rank()


def wait_key(key: str, timeout_s: Optional[float] = 30.0) -> str:
    if not is_initialized():
        raise RuntimeError("distributed not initialized")
    store = dist.distributed_c10d._get_default_store()
    if timeout_s is None:
        value: bytes = store.get(key)
        return value.decode("utf-8")
    if timeout_s <= 0:
        result = store.wait([key], timeout=0.0)
        if not result or not result[0]:
            raise TimeoutError(f"timeout waiting for key: {key}")
        value = store.get(key)
        return value.decode("utf-8")
    deadline = time.time() + timeout_s
    remaining = timeout_s
    while True:
        try:
            result = store.wait([key], timeout=max(remaining, 0.0))
        except Exception as exc:  # pragma: no cover - backend specific errors
            raise TimeoutError(f"timeout waiting for key: {key}") from exc
        if result and result[0]:
            value = store.get(key)
            return value.decode("utf-8")
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError(f"timeout waiting for key: {key}")


def publish_key(key: str, value: str) -> None:
    if not is_initialized():
        return
    store = dist.distributed_c10d._get_default_store()
    store.set(key, value.encode("utf-8"))


def get_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _local_rank() -> int:
    env_keys = [
        "LOCAL_RANK",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
    ]
    for key in env_keys:
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return 0


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "localhost"


def _infer_local_ip() -> str:
    try:
        return socket.gethostbyname(_hostname())
    except Exception:
        return "127.0.0.1"


def _gather_all(obj: Any) -> List[Any]:
    _require_dist()
    buffer: List[Any] = [None for _ in range(_world_size())]
    dist.all_gather_object(buffer, obj)
    return buffer


def _format_flight_host(host: Any, fallback: Any | None = None, *, default: str = "127.0.0.1") -> str:
    candidate: str = ""

    def _coerce(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if value is None:
            return ""
        coerced = str(value).strip()
        return coerced

    candidate = _coerce(host)
    if not candidate:
        candidate = _coerce(fallback)
    if not candidate:
        candidate = default

    try:
        parsed = ipaddress.ip_address(candidate.strip("[]"))
    except ValueError:
        return candidate

    compressed = parsed.compressed
    if parsed.version == 6:
        return f"[{compressed}]"
    return compressed


def _mq_addr(
    kind: str, *args: Any, base_path: str = "/dev/shm/stf_mq", **kwargs: Any
) -> str:
    os.makedirs(base_path, exist_ok=True)
    return f"ipc://{base_path}/stf_{_hostname()}_{kind}.ipc"


@dataclass
class MessageQueueConfig:
    local_rank: int
    _zmq: Any = field(init=False, repr=False)
    _pull: Any | None = field(init=False, default=None, repr=False)
    _pub: Any | None = field(init=False, default=None, repr=False)
    _push: Any | None = field(init=False, default=None, repr=False)
    _sub: Any | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            from ..connection.queue import zmq
        except Exception:
            try:
                import zmq
            except Exception as exc:
                raise RuntimeError(
                    "pyzmq is required. Install it with `pip install pyzmq`."
                ) from exc
        self._zmq = zmq.Context.instance()
        up_addr = _mq_addr("up")
        down_addr = _mq_addr("down")
        if self.local_rank == 0:
            self._pull = self._zmq.socket(zmq.PULL)
            self._pull.bind(up_addr)
            self._pub = self._zmq.socket(zmq.PUB)
            self._pub.bind(down_addr)
            return
        self._push = self._zmq.socket(zmq.PUSH)
        self._push.connect(up_addr)
        self._sub = self._zmq.socket(zmq.SUB)
        self._sub.connect(down_addr)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")

    def push_up(self, payload: bytes | memoryview) -> None:
        if self.local_rank == 0 or self._push is None:
            raise RuntimeError("rank0 cannot push_up")
        self._push.send(payload, copy=False)

    def recv_up(self, flags: int = 0) -> memoryview:
        if self.local_rank != 0 or self._pull is None:
            raise RuntimeError("non-leader cannot recv_up")
        frame = self._pull.recv(flags=flags, copy=False)
        return frame.buffer

    def pub_down(
        self,
        payload: bytes | memoryview,
        *args: Any,
        topic: bytes = b"",
        **kwargs: Any,
    ) -> None:
        if self.local_rank != 0 or self._pub is None:
            raise RuntimeError("only leader can pub_down")
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            payload = bytes(payload)
        self._pub.send_multipart([topic, payload], copy=False)

    def sub_down(self, flags: int = 0) -> tuple[bytes, memoryview]:
        if self.local_rank == 0 or self._sub is None:
            raise RuntimeError("leader cannot sub_down")
        frames = self._sub.recv_multipart(flags=flags, copy=False)
        topic_bytes = bytes(frames[0])
        payload_mv = frames[1].buffer
        return (topic_bytes, payload_mv)


class IOController:
    def __init__(self) -> None:
        _require_dist()
        self.rank: int = _rank()
        self.world_size: int = _world_size()
        self.local_rank: int = _local_rank()
        self.is_node_leader: bool = self.local_rank == 0
        self._mq = MessageQueueConfig(local_rank=self.local_rank)
        self._flight_port: int = GRPC_DEFAULT_PORT
        self._bind_host: str = "0.0.0.0"
        self._my_ip: str = _infer_local_ip()
        self._server: Any | None = None
        self._clients: Dict[str, Any] = {}
        self._leaders: Dict[str, Dict[str, Any]] = {}

    def start(self) -> IOController:
        host_name = _hostname()
        infos = _gather_all(
            {
                "rank": self.rank,
                "host": host_name,
                "local_rank": self.local_rank,
                "ip": self._my_ip,
            }
        )
        per_host: Dict[str, List[Dict[str, Any]]] = {}
        for info in infos:
            host = info["host"]
            per_host.setdefault(host, []).append(info)
        leaders: Dict[str, Dict[str, Any]] = {}
        for host, entries in per_host.items():
            leader = next(
                (item for item in entries if item["local_rank"] == 0),
                sorted(entries, key=lambda val: val["rank"])[0],
            )
            leaders[host] = dict(leader)
        self._leaders = leaders
        def _wait_for_flight_port(
            target_host: str,
            *,
            total_timeout_s: float = 60.0,
            poll_timeout_s: float = 2.0,
        ) -> int | None:
            if total_timeout_s is not None and total_timeout_s <= 0:
                total_timeout_s = 0.0
            deadline = None if total_timeout_s is None else time.time() + total_timeout_s
            while True:
                try:
                    port_value = wait_key(
                        f"flight_port:{target_host}",
                        timeout_s=poll_timeout_s,
                    )
                except TimeoutError:
                    if deadline is not None and time.time() >= deadline:
                        return None
                    continue
                try:
                    return int(port_value)
                except (TypeError, ValueError):
                    return None

        def _cache_leader(host: str, info: Mapping[str, Any] | None) -> Dict[str, Any]:
            def _sanitize_ip(value: Any) -> Optional[str]:
                if not isinstance(value, str):
                    return None
                candidate = value.strip()
                if not candidate:
                    return None
                stripped = candidate
                if stripped.startswith("[") and stripped.endswith("]"):
                    stripped = stripped[1:-1].strip()
                    if not stripped:
                        return None
                if "%" in stripped:
                    base, _, _ = stripped.partition("%")
                    stripped = base.strip()
                    if not stripped:
                        return None
                try:
                    parsed = ipaddress.ip_address(stripped)
                except ValueError:
                    return None
                if parsed.is_unspecified or parsed.is_loopback:
                    return None
                return str(parsed)

            def _resolve_ip_from_host(hostname: Any) -> Optional[str]:
                if not isinstance(hostname, str):
                    return None
                candidate = hostname.strip()
                if not candidate:
                    return None
                addrinfo: List[tuple[Any, ...]] | None = None
                try:
                    addrinfo = socket.getaddrinfo(
                        candidate,
                        None,
                        family=socket.AF_UNSPEC,
                        type=socket.SOCK_STREAM,
                    )
                except socket.gaierror:
                    return None
                except Exception:
                    with contextlib.suppress(Exception):
                        addrinfo = socket.getaddrinfo(candidate, None)
                if not addrinfo:
                    return None
                for resolved in addrinfo:
                    try:
                        sockaddr = resolved[4]
                    except Exception:
                        sockaddr = None
                    if not sockaddr:
                        continue
                    addr = sockaddr[0]
                    sanitized = _sanitize_ip(addr)
                    if sanitized:
                        return sanitized
                return None

            if isinstance(info, dict):
                info_dict: Dict[str, Any] = info
            elif info is None:
                info_dict = {}
            elif isinstance(info, Mapping):
                info_dict = dict(info)
            else:
                try:
                    info_dict = dict(info)  # type: ignore[arg-type]
                except Exception:
                    info_dict = {}

            existing = self._leaders.get(host)
            if isinstance(existing, dict):
                leader_info = existing
            elif isinstance(existing, Mapping):
                leader_info = dict(existing)
            elif existing is None:
                leader_info = {}
            else:
                try:
                    leader_info = dict(existing)  # type: ignore[arg-type]
                except Exception:
                    leader_info = {}

            for key, value in info_dict.items():
                if key in {"ip", "host"}:
                    continue
                existing_value = leader_info.get(key, ...)
                if existing_value is ... or existing_value != value:
                    leader_info[key] = value

            ip_value = info_dict.get("ip")
            sanitized_ip = _sanitize_ip(ip_value)
            if sanitized_ip:
                leader_info["ip"] = sanitized_ip
            elif ip_value:
                leader_info["ip"] = ip_value
            host_value = info_dict.get("host")
            resolved_host: Optional[str] = None
            if isinstance(host_value, str) and host_value.strip():
                resolved_host = host_value.strip()
                leader_info["host"] = resolved_host
            else:
                resolved_host = leader_info.get("host") or host
                leader_info.setdefault("host", resolved_host)
            cached_ip = leader_info.get("ip")
            cached_sanitized_ip = _sanitize_ip(cached_ip)
            if isinstance(cached_ip, str) and not cached_sanitized_ip:
                leader_info.pop("ip", None)
            elif cached_sanitized_ip:
                leader_info["ip"] = cached_sanitized_ip
            has_valid_ip = bool(_sanitize_ip(leader_info.get("ip")))
            if (not has_valid_ip) and resolved_host:
                resolved_ip = _resolve_ip_from_host(resolved_host)
                if resolved_ip:
                    leader_info["ip"] = resolved_ip
            self._leaders[host] = leader_info
            return leader_info

        if self.is_node_leader:
            resolved = get_available_addr(f"{self._bind_host}:{self._flight_port}")
            if ":" in resolved:
                bind_host, port_str = resolved.rsplit(":", 1)
                bind_port = int(port_str)
            else:
                bind_host = resolved
                bind_port = int(self._flight_port)
            server, uri = ArrowFlight.start_server_standby(
                host=bind_host,
                port=bind_port,
                wait_ready_s=15.0,
            )
            self._server = server
            parsed = urlparse(uri if isinstance(uri, str) else str(uri))
            if parsed.hostname:
                self._bind_host = parsed.hostname
            if parsed.port:
                self._flight_port = int(parsed.port)
            publish_key(f"flight_port:{host_name}", str(self._flight_port))
            leader_info = _cache_leader(
                host_name, {"ip": self._my_ip, "host": host_name}
            )
            leader_info["flight_port"] = self._flight_port
            for host, info in leaders.items():
                if host == host_name:
                    continue
                remote_port = _wait_for_flight_port(host)
                if remote_port is None:
                    leader_info = _cache_leader(host, info)
                    leader_info["flight_port"] = None
                    continue
                leader_info = _cache_leader(host, info)
                leader_info["flight_port"] = remote_port
                endpoint_host = leader_info.get("ip")
                if not endpoint_host or endpoint_host in {"0.0.0.0", ""}:
                    endpoint_host = leader_info.get("host") or info.get("host")
                formatted_host = _format_flight_host(endpoint_host, fallback=host)
                endpoint = f"grpc+tcp://{formatted_host}:{remote_port}"
                for attempt in range(10):
                    try:
                        self._clients[host] = FlightModule.connect(endpoint)
                        break
                    except Exception:
                        if attempt == 9:
                            self._clients.pop(host, None)
                            break
                        time.sleep(min(0.5 * (attempt + 1), 5.0))
        else:
            local_port = _wait_for_flight_port(host_name)
            if local_port is not None:
                self._flight_port = local_port
            leader_info = _cache_leader(
                host_name, {"ip": self._my_ip, "host": host_name}
            )
            leader_info["flight_port"] = self._flight_port
        return self

    def push_local_up(self, payload: bytes | memoryview) -> None:
        if self.is_node_leader:
            raise RuntimeError("leader cannot push_local_up")
        self._mq.push_up(payload)

    def pull_local_up(self, flags: int = 0) -> memoryview:
        if not self.is_node_leader:
            raise RuntimeError("non-leader cannot pull_local_up")
        return self._mq.recv_up(flags=flags)

    def broadcast_down(
        self,
        payload: bytes | memoryview,
        *args: Any,
        topic: bytes = b"",
        **kwargs: Any,
    ) -> None:
        if not self.is_node_leader:
            raise RuntimeError("non-leader cannot broadcast_down")
        self._mq.pub_down(payload, topic=topic)

    def subscribe_down(self, flags: int = 0) -> tuple[bytes, memoryview]:
        if self.is_node_leader:
            raise RuntimeError("leader cannot subscribe_down")
        return self._mq.sub_down(flags=flags)

    def flight_endpoint(self) -> str | None:
        if not self.is_node_leader:
            return None
        formatted_host = _format_flight_host(self._my_ip, fallback=self._bind_host)
        return f"grpc+tcp://{formatted_host}:{self._flight_port}"
