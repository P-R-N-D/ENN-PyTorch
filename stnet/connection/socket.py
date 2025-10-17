from __future__ import annotations

import os
import threading
import time
from typing import Any, Iterator, Optional, Tuple

import torch.distributed as dist

from ..pipeline.dataset import MemoryMappedTensorStream
from ..toolkit.compat import patch_arrow


_ARROW = patch_arrow()
pa = _ARROW.module
flight = _ARROW.flight
if flight is None:
    raise RuntimeError("pyarrow.flight is required for ArrowFlight support")


class ZeroMQ:
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
                    raise TimeoutError(f"ZeroMQ wait timeout: {key}")
                time.sleep(0.05)


class ArrowFlight:
    MQ = ZeroMQ()

    @staticmethod
    def node_id(rank: int, local_rank: int) -> int:
        return int(rank) - int(local_rank)

    @staticmethod
    def resource_key(memmap_dir: str, split: str) -> str:
        return f"{os.path.basename(memmap_dir.rstrip(os.sep))}/{split}"

    @staticmethod
    def server_key(node_id: int) -> str:
        return f"flight://node/{node_id}"

    @staticmethod
    def _canon_name(value: Any) -> str:
        try:
            if isinstance(value, (bytes, bytearray, memoryview)):
                value = bytes(value).decode("utf-8")
        except Exception:
            value = str(value)
        return str(value).strip("/")

    @staticmethod
    def _bytes_of(value: Any) -> bytes:
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        if isinstance(value, memoryview):
            return value.tobytes()
        if isinstance(value, str):
            return value.encode("utf-8")
        try:
            return bytes(value)
        except Exception:
            return str(value).encode("utf-8")

    @staticmethod
    def _name_from_descriptor(descriptor: Any) -> str:
        path = getattr(descriptor, "path", None)
        if isinstance(path, (list, tuple)) and path:
            parts: list[str] = []
            for part in path:
                try:
                    parts.append(ArrowFlight._bytes_of(part).decode("utf-8"))
                except Exception:
                    parts.append(str(part))
            return ArrowFlight._canon_name("/".join(parts))
        command = getattr(descriptor, "command", None)
        if command is not None:
            try:
                return ArrowFlight._canon_name(ArrowFlight._bytes_of(command))
            except Exception:
                return ArrowFlight._canon_name(command)
        return ArrowFlight._canon_name(descriptor)

    @staticmethod
    def _build_descriptor(name: str) -> flight.FlightDescriptor:
        canonical = ArrowFlight._canon_name(name)
        parts = [
            segment.encode("utf-8")
            for segment in canonical.split("/")
            if segment
        ]
        if parts:
            return flight.FlightDescriptor.for_path(*parts)
        return flight.FlightDescriptor.for_command(canonical.encode("utf-8"))

    class Server(flight.FlightServerBase):
        def __init__(
            self, location: str | Tuple[str, int] | flight.Location
        ) -> None:
            super().__init__(location)
            self._lock = threading.RLock()
            self._datasets: dict[str, pa.RecordBatchReader] = {}

        def list_flights(
            self, context: Any, criteria: bytes | None
        ) -> Iterator[flight.FlightInfo]:
            _ = (context, criteria)
            del _
            with self._lock:
                for name, reader in self._datasets.items():
                    canonical = ArrowFlight._canon_name(name)
                    parts = [
                        segment.encode("utf-8")
                        for segment in canonical.split("/")
                        if segment
                    ]
                    if parts:
                        descriptor = flight.FlightDescriptor.for_path(*parts)
                    else:
                        descriptor = flight.FlightDescriptor.for_command(
                            canonical.encode("utf-8")
                        )
                    yield flight.FlightInfo(
                        reader.schema,
                        descriptor,
                        endpoints=[],
                        total_records=-1,
                        total_bytes=-1,
                    )

        def get_flight_info(
            self, context: Any, descriptor: flight.FlightDescriptor
        ) -> flight.FlightInfo:
            _ = context
            del _
            name = ArrowFlight._name_from_descriptor(descriptor)
            key = ArrowFlight._canon_name(name)
            with self._lock:
                reader = self._datasets.get(key)
                if reader is None:
                    for candidate_key in list(self._datasets.keys()):
                        if candidate_key.endswith(key):
                            reader = self._datasets[candidate_key]
                            key = candidate_key
                            break
                if reader is None:
                    raise KeyError(
                        f"dataset not found: {key} ; candidates={list(self._datasets.keys())}"
                    )
            ticket = flight.Ticket(key.encode("utf-8"))
            endpoints = [flight.FlightEndpoint(ticket, [])]
            return flight.FlightInfo(
                reader.schema,
                descriptor,
                endpoints=endpoints,
                total_records=-1,
                total_bytes=-1,
            )

        def do_get(
            self, context: Any, ticket: flight.Ticket
        ) -> flight.RecordBatchStream:
            _ = context
            del _
            name = ArrowFlight._canon_name(ticket.ticket)
            with self._lock:
                reader = self._datasets[name]
            return flight.RecordBatchStream(reader)

        def add_reader(self, name: str, reader: pa.RecordBatchReader) -> None:
            with self._lock:
                key = ArrowFlight._canon_name(name)
                self._datasets[key] = reader

    class Client:
        def __init__(
            self, uri: str | Tuple[str, int] | flight.Location
        ) -> None:
            self._client = flight.FlightClient(uri)

        def reader(
            self,
            name: str,
            *,
            timeout_s: float = 30.0,
            poll_interval_s: float = 0.05,
        ) -> pa.RecordBatchReader:
            descriptor = ArrowFlight._build_descriptor(name)
            deadline = time.time() + float(timeout_s)
            last_error: BaseException | None = None
            while time.time() < deadline:
                try:
                    info = self._client.get_flight_info(descriptor)
                    endpoints = list(getattr(info, "endpoints", []) or [])
                    if endpoints:
                        ticket = getattr(endpoints[0], "ticket", None)
                        if ticket is not None:
                            return self._client.do_get(ticket).to_reader()
                        last_error = RuntimeError(
                            "FlightInfo endpoints present but no ticket"
                        )
                except Exception as exc:
                    last_error = exc
                time.sleep(poll_interval_s)
            raise RuntimeError(
                f"Flight dataset '{name}' not ready: no endpoints within {timeout_s}s"
            ) from last_error

    @staticmethod
    def start_server_standby(
        *, host: str = "0.0.0.0", port: int = 0, wait_ready_s: float = 2.0
    ) -> Tuple[ArrowFlight.Server, str]:
        server = ArrowFlight.Server(location=f"grpc://{host}:{port}")
        thread = threading.Thread(target=server.serve, daemon=True)
        thread.start()
        start = time.time()
        actual_port = getattr(server, "port", 0) or port
        while actual_port in (None, 0) and time.time() - start < wait_ready_s:
            time.sleep(0.01)
            actual_port = getattr(server, "port", 0) or port
        uri = f"grpc://{host}:{actual_port}"
        return (server, uri)

    @staticmethod
    def reg_mmt_dataset(
        server: ArrowFlight.Server,
        name: str,
        mmts: MemoryMappedTensorStream,
        batch_size: int,
        split: str,
    ) -> None:
        meta = mmts._load_meta()
        total = int(meta["N"])
        fractions = meta.get("fractions", [1.0, 0.0])
        train_end = int(total * float(fractions[0])) if fractions else total
        start, end = (0, train_end) if split == "train" else (train_end, total)

        def _batches() -> Iterator[pa.RecordBatch]:
            index = start
            while index < end:
                nxt = min(index + int(batch_size), end)
                xb, yb = mmts.batch_range(index, nxt)
                yield MemoryMappedTensorStream.to_record_batch(xb, yb)
                index = nxt

        generator = _batches()
        first = next(generator)
        rest = list(generator)
        schema = first.schema
        reader = pa.RecordBatchReader.from_batches(schema, [first, *rest])
        server.add_reader(ArrowFlight._canon_name(name), reader)


def open_flight_client(
    uri: str | Tuple[str, int] | flight.Location
) -> flight.FlightClient:
    return flight.FlightClient(uri)


def start_flight_server(
    host: str = "0.0.0.0", port: int = 0
) -> Tuple[ArrowFlight.Server, str]:
    return ArrowFlight.start_server_standby(host=host, port=port)
