# -*- coding: utf-8 -*-
from __future__ import annotations

from urllib.parse import urlparse

import os
import socket
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, cast

import torch.distributed as dist

try:
    from .socket import Endpoint
except ImportError:

    class _EndpointUnavailable:
        @staticmethod
        def start_server_standby(
            *args: Any,
            host: str = "0.0.0.0",
            port: int = 0,
            wait_ready_s: float = 10.0,
            **kwargs: Any,
        ) -> Any:
            raise RuntimeError(
                "Arrow Flight server is unavailable. "
                "Install pyarrow[flight] to enable it."
            )

        @staticmethod
        def connect(
            endpoint: Any,
            *args: Any,
            wait_ready_s: float = 5.0,
            poll_interval_s: float = 0.05,
            **kwargs: Any,
        ) -> None:
            raise RuntimeError("Arrow Flight module is unavailable.")

    Endpoint = _EndpointUnavailable()

from ..utils.platform import Distributed


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
    return Distributed.get_world_size()


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
    elif timeout_s <= 0:
        result = store.wait([key], timeout=0.0)
        if not result or not result[0]:
            raise TimeoutError(f"timeout waiting for key: {key}")
        value = store.get(key)
        return value.decode("utf-8")
    else:
        deadline = time.time() + timeout_s
        remaining = timeout_s
        while True:
            try:
                result = store.wait([key], timeout=max(remaining, 0.0))
            except Exception as exc:
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


def get_available_port(host: Optional[str] = None) -> int:
    host_text = Distributed.Network.coerce_host(host)
    default_host = (
        host_text or Distributed.get_preferred_ip(allow_loopback=True) or "127.0.0.1"
    )
    locator = Distributed.Network(
        allow_loopback=True,
        fallback=default_host,
        default=default_host,
    )
    try:
        allocated = locator.allocate(f"{host_text}:0" if host_text else None)
        _, port = Distributed.normalize_endpoint(allocated, default_host)
        return int(port)
    except Exception:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return int(sock.getsockname()[1])


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


def _local_hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "localhost"


def _as_dict(
    candidate: Mapping[str, Any] | Iterable[Tuple[Any, Any]] | None
) -> Dict[str, Any]:
    if isinstance(candidate, dict):
        return dict(candidate)
    elif isinstance(candidate, Mapping):
        return dict(candidate)
    result: Dict[str, Any] = {}
    if isinstance(candidate, Iterable):
        for item in cast(Iterable[Tuple[Any, Any]], candidate):
            try:
                key, value = item
            except Exception:
                continue
            result[key] = value
    return result


def _merge_leader_info(
    existing: Mapping[str, Any] | Iterable[Tuple[Any, Any]] | None,
    info: Mapping[str, Any] | Iterable[Tuple[Any, Any]] | None,
    host: str,
) -> Dict[str, Any]:
    resolver = Distributed.Network(allow_loopback=True)
    incoming_info = _as_dict(info)
    merged_info = _as_dict(existing)

    for key, value in incoming_info.items():
        if key in {"ip", "host"}:
            continue
        cached_value = merged_info.get(key, ...)
        if cached_value is ... or cached_value != value:
            merged_info[key] = value

    ip_value = incoming_info.get("ip")
    sanitized_ip = resolver.normalize(ip_value)
    if sanitized_ip:
        merged_info["ip"] = sanitized_ip
    elif ip_value:
        coerced_ip = resolver.coerce_host(ip_value)
        if coerced_ip:
            merged_info["ip"] = coerced_ip

    host_value = incoming_info.get("host")
    normalized_host = resolver.coerce_host(host_value)
    if normalized_host:
        merged_info["host"] = normalized_host
    else:
        merged_info.setdefault("host", host)

    cached_ip = merged_info.get("ip")
    cached_sanitized_ip = resolver.normalize(cached_ip)
    if isinstance(cached_ip, str) and not cached_sanitized_ip:
        merged_info.pop("ip", None)
    elif cached_sanitized_ip:
        merged_info["ip"] = cached_sanitized_ip

    has_valid_ip = bool(resolver.normalize(merged_info.get("ip")))
    target_host = merged_info.get("host") or host
    if (not has_valid_ip) and target_host:
        resolved_ip = resolver.resolve(target_host)
        if resolved_ip:
            merged_info["ip"] = resolved_ip
    return merged_info


def _wait_for_flight_port_value(
    target_host: str,
    *,
    total_timeout_s: float | None = 60.0,
    poll_timeout_s: float = 2.0,
) -> Optional[int]:
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


def _gather_all(obj: Any) -> List[Any]:
    _require_dist()
    buffer: List[Any] = [None for _ in range(_world_size())]
    dist.all_gather_object(buffer, obj)
    return buffer


def _mq_addr(kind: str, *args: Any, base_path: str = "/dev/shm/stf_mq", **kwargs: Any) -> str:
    os.makedirs(base_path, exist_ok=True)
    return f"ipc://{base_path}/stf_{_local_hostname()}_{kind}.ipc"


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
            from .queue import zmq
        except Exception:
            try:
                import zmq
            except Exception as exc:
                raise RuntimeError(
                    "pyzmq is required. Install it with `pip install stnet-pytorch[queue]` or `pip install pyzmq`."
                ) from exc
        self._zmq = zmq.Context.instance()
        up_addr = _mq_addr("up")
        down_addr = _mq_addr("down")
        if self.local_rank == 0:
            self._pull = self._zmq.socket(zmq.PULL)
            self._pull.bind(up_addr)
            self._pub = self._zmq.socket(zmq.PUB)
            self._pub.bind(down_addr)
        else:
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
        self._local_ip: str = Distributed.get_preferred_ip(allow_loopback=True)
        self._server: Any | None = None
        self._clients: Dict[str, Any] = {}
        self._leaders: Dict[str, Dict[str, Any]] = {}

    def start(self) -> IOController:
        local_hostname = _local_hostname()
        gathered_members = _gather_all(
            {
                "rank": self.rank,
                "host": local_hostname,
                "local_rank": self.local_rank,
                "ip": self._local_ip,
            }
        )
        members_by_host: Dict[str, List[Dict[str, Any]]] = {}
        for member in gathered_members:
            host_identifier = member["host"]
            members_by_host.setdefault(host_identifier, []).append(member)
        leader_by_host: Dict[str, Dict[str, Any]] = {}
        for host_identifier, participants in members_by_host.items():
            host_leader = next(
                (item for item in participants if item["local_rank"] == 0),
                sorted(participants, key=lambda participant: participant["rank"])[0],
            )
            leader_by_host[host_identifier] = dict(host_leader)
        self._leaders = leader_by_host
        if self.is_node_leader:
            locator = Distributed.Network()
            resolved = locator.allocate(f"{self._bind_host}:{self._flight_port}")
            parsed_bind = urlparse(f"tcp://{resolved}")
            bind_host = parsed_bind.hostname or self._bind_host
            bind_port = parsed_bind.port or int(self._flight_port)
            server, uri = Endpoint.start_server_standby(
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
            publish_key(f"flight_port:{local_hostname}", str(self._flight_port))
            leader_info = _merge_leader_info(
                self._leaders.get(local_hostname),
                {"ip": self._local_ip, "host": local_hostname},
                local_hostname,
            )
            leader_info["flight_port"] = self._flight_port
            self._leaders[local_hostname] = leader_info
            for host_identifier, info in leader_by_host.items():
                if host_identifier == local_hostname:
                    continue
                remote_port = _wait_for_flight_port_value(host_identifier)
                merged = _merge_leader_info(
                    self._leaders.get(host_identifier),
                    info,
                    host_identifier,
                )
                if remote_port is None:
                    merged["flight_port"] = None
                    self._leaders[host_identifier] = merged
                    continue
                merged["flight_port"] = remote_port
                self._leaders[host_identifier] = merged
                endpoint_host = merged.get("ip")
                if not endpoint_host or endpoint_host in {"0.0.0.0", ""}:
                    endpoint_host = merged.get("host") or info.get("host")
                formatter = Distributed.Network(
                    fallback=host_identifier,
                    allow_loopback=True,
                )
                formatted_host = formatter.format(endpoint_host)
                endpoint = f"grpc+tcp://{formatted_host}:{remote_port}"
                for attempt in range(10):
                    try:
                        self._clients[host_identifier] = Endpoint.connect(endpoint)
                        break
                    except Exception:
                        if attempt == 9:
                            self._clients.pop(host_identifier, None)
                            break
                        time.sleep(min(0.5 * (attempt + 1), 5.0))
        else:
            local_port = _wait_for_flight_port_value(local_hostname)
            if local_port is not None:
                self._flight_port = local_port
            leader_info = _merge_leader_info(
                self._leaders.get(local_hostname),
                {"ip": self._local_ip, "host": local_hostname},
                local_hostname,
            )
            leader_info["flight_port"] = self._flight_port
            self._leaders[local_hostname] = leader_info
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
        formatter = Distributed.Network(
            fallback=self._bind_host,
            allow_loopback=True,
        )
        formatted_host = formatter.format(self._local_ip)
        return f"grpc+tcp://{formatted_host}:{self._flight_port}"
