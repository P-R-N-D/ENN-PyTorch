from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch.distributed as dist

from ..connection.socket import ArrowFlight

try:
    from ..connection.socket import FlightModule
except ImportError:

    class FlightModule:
        @staticmethod
        def connect(*args: Any, **kwargs: Any) -> None:
            details = " with args/kwargs" if args or kwargs else ""
            raise RuntimeError(f"Arrow Flight module is unavailable{details}.")


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
    return dist.get_world_size()


def _rank() -> int:
    _require_dist()
    return dist.get_rank()


def wait_key(key: str, timeout_s: float = 30.0) -> str:
    if not is_initialized():
        raise RuntimeError("distributed not initialized")
    store = dist.distributed_c10d._get_default_store()
    start = time.time()
    while True:
        try:
            value: bytes = store.get(key)
            return value.decode("utf-8")
        except Exception:
            if time.time() - start > timeout_s:
                raise TimeoutError(f"timeout waiting for key: {key}")
            time.sleep(0.05)


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
        self._my_ip: str = _infer_local_ip()
        self._server: Any | None = None
        self._clients: Dict[str, Any] = {}
        self._leaders: Dict[str, Dict[str, Any]] = {}

    def start(self) -> IOController:
        infos = _gather_all(
            {
                "rank": self.rank,
                "host": _hostname(),
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
            leaders[host] = leader
        self._leaders = leaders
        if self.is_node_leader:
            self._server = ArrowFlight(
                location=f"grpc+tcp://0.0.0.0:{self._flight_port}", datasets={}
            )
            for host, info in leaders.items():
                if host == _hostname():
                    continue
                endpoint = f"grpc+tcp://{info['ip']}:{self._flight_port}"
                try:
                    self._clients[host] = FlightModule.connect(endpoint)
                except Exception:
                    continue
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
        return f"grpc+tcp://{self._my_ip}:{self._flight_port}"
