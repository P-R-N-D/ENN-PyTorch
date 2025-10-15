# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Protocol

import base64
import json
import queue
import time
import uuid


class _ZMQProxy:
    _mod = None

    def __getattr__(self, name: str):
        if self._mod is None:
            try:
                import zmq
            except Exception as e:  
                raise RuntimeError("pyzmq가 필요합니다: MessageQueue를 사용하려면 pyzmq를 설치하세요.") from e
            type(self)._mod = zmq
        return getattr(self._mod, name)


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
    def from_bytes(data: bytes) -> "Message":
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
        **kwargs: Any
    ) -> int: ...


class Subscriber(Protocol):
    def recv(self, *args: Any, timeout: Optional[float] = None, **kwargs: Any) -> Optional[Message]: ...
    def __iter__(self) -> Iterator[Message]: ...


class CompatQueue(Publisher, Subscriber):
    def __init__(self, topic: str) -> None:
        self._topic = topic
        self._q: queue.Queue[Message] = queue.Queue()
        self._offset = 0

    def publish(
        self,
        payload: bytes,
        *args: Any,
        headers: Optional[Dict[str, str]] = None,
        key: Optional[str] = None,
        **kwargs: Any
    ) -> int:
        self._offset += 1
        msg = Message(
            topic=self._topic,
            offset=self._offset,
            ts_ms=int(time.time() * 1000),
            payload=payload,
            headers=dict(headers) if headers else {},
            key=key,
        )
        self._q.put(msg)
        return self._offset

    def recv(self, *args: Any, timeout: Optional[float] = None, **kwargs: Any) -> Optional[Message]:
        try:
            if timeout is None:
                return self._q.get()
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def __iter__(self) -> Iterator[Message]:
        while True:
            yield self._q.get()


class MessageQueue(Publisher, Subscriber):
    def __init__(
        self,
        topic: str,
        *args: Any,
        pub_bind: Optional[str] = None,
        sub_connect: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        self._ctx = zmq.Context.instance()
        self._topic: bytes = topic.encode("utf-8")
        self._pub = self._ctx.socket(zmq.PUB)
        if pub_bind:
            self._pub.bind(pub_bind)
        self._sub = self._ctx.socket(zmq.SUB)
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
        **kwargs: Any
    ) -> int:
        self._offset += 1
        msg = Message(
            topic=self._topic.decode("utf-8"),
            offset=self._offset,
            ts_ms=int(time.time() * 1000),
            payload=payload,
            headers=dict(headers) if headers else {},
            key=key or uuid.uuid4().hex,
        )
        self._pub.send_multipart([self._topic, msg.to_bytes()])
        return self._offset

    def recv(self, *args: Any, timeout: Optional[float] = None, **kwargs: Any) -> Optional[Message]:
        poller = zmq.Poller()
        poller.register(self._sub, zmq.POLLIN)
        ms = int(timeout * 1000) if timeout is not None else None
        events = dict(poller.poll(timeout=ms))
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