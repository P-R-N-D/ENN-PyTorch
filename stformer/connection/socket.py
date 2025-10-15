# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Iterable, Iterator, Optional

import os
import threading
import time

import pyarrow as pa
import pyarrow.flight as flight
import torch.distributed as dist

from ..pipeline.dataset import MemoryMappedTensorStream as MMTS


class ZeroMQ:
    def __init__(self) -> None:
        self._enabled: bool = False
        self._store: Optional[Any] = None
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
        t0 = time.time()
        while True:
            try:
                return self._store.get(key).decode("utf-8")
            except Exception:
                if (time.time() - t0) > float(timeout_s):
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
    def _canon_name(s: Any) -> str:
        try:
            if isinstance(s, (bytes, bytearray, memoryview)):
                s = bytes(s).decode("utf-8")
        except Exception:
            s = str(s)
        return str(s).strip("/")

    @staticmethod
    def _bytes_of(x: Any) -> bytes:
        if isinstance(x, (bytes, bytearray)):
            return bytes(x)
        if isinstance(x, memoryview):
            return x.tobytes()
        if isinstance(x, str):
            return x.encode("utf-8")
        try:
            return bytes(x)
        except Exception:
            return str(x).encode("utf-8")

    @staticmethod
    def _name_from_descriptor(descriptor: Any) -> str:
        path = getattr(descriptor, "path", None)
        if isinstance(path, (list, tuple)) and path:
            parts: list[str] = []
            for p in path:
                try:
                    parts.append(ArrowFlight._bytes_of(p).decode("utf-8"))
                except Exception:
                    parts.append(str(p))
            return ArrowFlight._canon_name("/".join(parts))
        cmd = getattr(descriptor, "command", None)
        if cmd is not None:
            try:
                return ArrowFlight._canon_name(ArrowFlight._bytes_of(cmd))
            except Exception:
                return ArrowFlight._canon_name(cmd)
        return ArrowFlight._canon_name(descriptor)

    @staticmethod
    def _build_descriptor(name: str) -> flight.FlightDescriptor:
        s = ArrowFlight._canon_name(name)
        parts = [p.encode("utf-8") for p in s.split("/") if p]
        if parts:
            return flight.FlightDescriptor.for_path(*parts)
        return flight.FlightDescriptor.for_command(s.encode("utf-8"))


    class Server(flight.FlightServerBase):
        def __init__(self, location: str | tuple[str, int] | flight.Location) -> None:
            super().__init__(location)
            self._lock = threading.RLock()
            self._datasets: dict[str, pa.RecordBatchReader] = {}

        def list_flights(self, context: Any, criteria: bytes | None) -> Iterable[flight.FlightInfo]:
            with self._lock:
                for name, rdr in self._datasets.items():
                    parts = [seg.encode("utf-8") for seg in ArrowFlight._canon_name(name).split("/") if seg]
                    desc = (
                        flight.FlightDescriptor.for_path(*parts)
                        if parts
                        else flight.FlightDescriptor.for_command(ArrowFlight._canon_name(name).encode("utf-8"))
                    )
                    yield flight.FlightInfo(
                        rdr.schema, desc, endpoints=[], total_records=-1, total_bytes=-1
                    )

        def get_flight_info(self, context: Any, descriptor: flight.FlightDescriptor) -> flight.FlightInfo:
            name = ArrowFlight._name_from_descriptor(descriptor)
            key = ArrowFlight._canon_name(name)
            with self._lock:
                rdr = self._datasets.get(key)
                if rdr is None:
                    for k in list(self._datasets.keys()):
                        if k.endswith(key):
                            rdr = self._datasets[k]
                            key = k
                            break
                if rdr is None:
                    raise KeyError(f"dataset not found: {key} ; candidates={list(self._datasets.keys())}")
            ticket = flight.Ticket(key.encode("utf-8"))
            endpoints = [flight.FlightEndpoint(ticket, [])]
            return flight.FlightInfo(rdr.schema, descriptor, endpoints=endpoints, total_records=-1, total_bytes=-1)

        def do_get(self, context: Any, ticket: flight.Ticket) -> flight.RecordBatchStream:
            name = ArrowFlight._canon_name(ticket.ticket)
            with self._lock:
                rdr = self._datasets[name]
            return flight.RecordBatchStream(rdr)

        def add_reader(self, name: str, reader: pa.RecordBatchReader) -> None:
            with self._lock:
                key = ArrowFlight._canon_name(name)
                self._datasets[key] = reader


    class Client:
        def __init__(self, uri: str | tuple[str, int] | flight.Location) -> None:
            self._cli = flight.FlightClient(uri)

        def reader(
            self,
            name: str,
            *args: Any,
            timeout_s: float = 30.0,
            poll_interval_s: float = 0.05,
            **kwargs: Any
        ) -> pa.RecordBatchReader:
            desc = ArrowFlight._build_descriptor(name)
            deadline = time.time() + float(timeout_s)
            last_err: Optional[BaseException] = None
            while time.time() < deadline:
                try:
                    info = self._cli.get_flight_info(desc)
                    endpoints = list(getattr(info, "endpoints", []) or [])
                    if endpoints:
                        tk = getattr(endpoints[0], "ticket", None)
                        if tk is not None:
                            return self._cli.do_get(tk).to_reader()
                        last_err = RuntimeError("FlightInfo endpoints present but no ticket")
                except Exception as e:
                    last_err = e
                time.sleep(poll_interval_s)
            raise RuntimeError(
                f"Flight dataset '{name}' not ready: no endpoints within {timeout_s}s"
            ) from last_err


    @staticmethod
    def start_server_standby(
        *args: Any,
        host: str = "0.0.0.0",
        port: int = 0,
        wait_ready_s: float = 2.0,
        **kwargs: Any
    ) -> tuple["ArrowFlight.Server", str]:
        srv = ArrowFlight.Server(location=f"grpc://{host}:{port}")
        t = threading.Thread(target=srv.serve, daemon=True)
        t.start()
        t0 = time.time()
        actual_port = getattr(srv, "port", 0) or port
        while actual_port in (None, 0) and (time.time() - t0) < wait_ready_s:
            time.sleep(0.01)
            actual_port = getattr(srv, "port", 0) or port
        uri = f"grpc://{host}:{actual_port}"
        return srv, uri

    @staticmethod
    def reg_mmt_dataset(
        srv: "ArrowFlight.Server",
        name: str,
        mmts: MMTS,
        batch_size: int,
        split: str,
    ) -> None:
        meta = mmts._load_meta()
        N = int(meta["N"])
        frac = meta.get("fractions", [1.0, 0.0])
        train_end = int(N * float(frac[0])) if frac else N
        start, end = ((0, train_end) if split == "train" else (train_end, N))

        def _batches() -> Iterator[pa.RecordBatch]:
            s = start
            while s < end:
                e = min(s + int(batch_size), end)
                Xb, Yb = mmts.batch_range(s, e)
                yield MMTS.to_record_batch(Xb, Yb)
                s = e

        gen = _batches()
        first = next(gen)
        rest = list(gen)
        schema = first.schema
        rdr = pa.RecordBatchReader.from_batches(schema, [first, *rest])
        srv.add_reader(ArrowFlight._canon_name(name), rdr)


def open_flight_client(uri: str | tuple[str, int] | flight.Location) -> flight.FlightClient:
    return flight.FlightClient(uri)


def start_flight_server(host: str = "0.0.0.0", port: int = 0) -> tuple[ArrowFlight.Server, str]:
    return ArrowFlight.start_server_standby(host=host, port=port)