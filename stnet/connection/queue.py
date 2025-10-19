# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import json
import queue
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional, Protocol

import torch.distributed as dist


class _ZMQProxy:
    _module: Any | None = None

    def __getattr__(self, name: str) -> Any:
        if self._module is None:
            try:
                import zmq
            except Exception as exc:
                raise RuntimeError(
                    "pyzmq is required. Install it with `pip install stnet-pytorch[queue]` or `pip install pyzmq` to use MessageQueue."
                ) from exc
            type(self)._module = zmq
        return getattr(self._module, name)


zmq = _ZMQProxy()


@dataclass(frozen=True, slots=True)
class Message:
    topic: str
    offset: int
    ts_ms: int
    payload: bytes
    headers: Dict[str, str] = field(default_factory=dict)
    key: Optional[str] = None

    def to_bytes(self) -> bytes:
        obj = {
            "topic": self.topic,
            "offset": self.offset,
            "ts_ms": self.ts_ms,
            "payload_b64": base64.b64encode(self.payload).decode("ascii"),
            "headers": self.headers,
            "key": self.key,
        }
        return json.dumps(obj, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def from_bytes(data: bytes) -> Message:
        obj = json.loads(data.decode("utf-8"))
        return Message(
            topic=str(obj["topic"]),
            offset=int(obj["offset"]),
            ts_ms=int(obj["ts_ms"]),
            payload=base64.b64decode(obj["payload_b64"]),
            headers=dict(obj.get("headers", {})),
            key=obj.get("key"),
        )


class Publisher(Protocol):
    def publish(
        self,
        payload: bytes,
        *args: Any,
        headers: Optional[Dict[str, str]] = None,
        key: Optional[str] = None,
        **kwargs: Any,
    ) -> int:
        if False:
            raise RuntimeError(payload, headers, key)
        raise NotImplementedError


class Subscriber(Protocol):
    def recv(
        self,
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Optional[Message]:
        if False:
            raise RuntimeError(timeout)
        raise NotImplementedError

    def __iter__(self) -> Iterator[Message]:
        raise NotImplementedError


class CompatQueue(Publisher, Subscriber):
    def __init__(self, topic: str) -> None:
        self._topic = topic
        self._queue: queue.Queue[Message] = queue.Queue()
        self._offset = 0

    def publish(
        self,
        payload: bytes,
        *args: Any,
        headers: Optional[Dict[str, str]] = None,
        key: Optional[str] = None,
        **kwargs: Any,
    ) -> int:
        self._offset += 1
        message = Message(
            topic=self._topic,
            offset=self._offset,
            ts_ms=int(time.time() * 1000),
            payload=payload,
            headers=dict(headers) if headers else {},
            key=key,
        )
        self._queue.put(message)
        return self._offset

    def recv(
        self,
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Optional[Message]:
        try:
            if timeout is None:
                return self._queue.get()
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def __iter__(self) -> Iterator[Message]:
        while True:
            yield self._queue.get()


class MessageQueue(Publisher, Subscriber):
    def __init__(
        self,
        topic: str,
        *args: Any,
        pub_bind: Optional[str] = None,
        sub_connect: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._context = zmq.Context.instance()
        self._topic = topic.encode("utf-8")
        self._pub = self._context.socket(zmq.PUB)
        if pub_bind:
            self._pub.bind(pub_bind)
        self._sub = self._context.socket(zmq.SUB)
        if sub_connect:
            self._sub.connect(sub_connect)
        self._sub.setsockopt(zmq.SUBSCRIBE, self._topic)
        self._offset = 0

    def publish(
        self,
        payload: bytes,
        *args: Any,
        headers: Optional[Dict[str, str]] = None,
        key: Optional[str] = None,
        **kwargs: Any,
    ) -> int:
        self._offset += 1
        message = Message(
            topic=self._topic.decode("utf-8"),
            offset=self._offset,
            ts_ms=int(time.time() * 1000),
            payload=payload,
            headers=dict(headers) if headers else {},
            key=key or uuid.uuid4().hex,
        )
        self._pub.send_multipart([self._topic, message.to_bytes()])
        return self._offset

    def recv(
        self,
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Optional[Message]:
        poller = zmq.Poller()
        poller.register(self._sub, zmq.POLLIN)
        timeout_ms = int(timeout * 1000) if timeout is not None else None
        events = dict(poller.poll(timeout=timeout_ms))
        if self._sub in events:
            topic, data = self._sub.recv_multipart()
            if topic != self._topic:
                return None
            return Message.from_bytes(data)
        return None

    def __iter__(self) -> Iterator[Message]:
        while True:
            _, data = self._sub.recv_multipart()
            yield Message.from_bytes(data)

    def close(self) -> None:
        self._pub.close(0)
        self._sub.close(0)


class DistributedQueue:
    def __init__(self) -> None:
        self._enabled = False
        self._store: Any | None = None
        try:
            self._enabled = dist.is_available() and dist.is_initialized()
            if self._enabled:
                self._store = dist.distributed_c10d._get_default_store()
        except Exception:
            self._enabled = False
            self._store = None

    def publish(self, key: str, value: str) -> None:
        if not self._enabled or self._store is None:
            return
        self._store.set(key, value.encode("utf-8"))

    def wait(self, key: str, timeout_s: float = 120.0) -> Optional[str]:
        if not self._enabled or self._store is None:
            return None
        start = time.time()
        while True:
            try:
                return self._store.get(key).decode("utf-8")
            except Exception:
                if time.time() - start > float(timeout_s):
                    raise TimeoutError(f"DistributedQueue wait timeout: {key}")
                time.sleep(0.05)
