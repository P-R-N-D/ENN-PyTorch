# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import threading
import time
from urllib.parse import urlparse
from typing import TYPE_CHECKING, Any, Iterator, Tuple

from ..utils.capability import Network, get_preferred_ip
from ..utils.compat import patch_arrow


if TYPE_CHECKING:
    from ..data.dataset import SampleReader


def _memory_mapped_tensor_stream() -> type["SampleReader"]:
    from ..data.dataset import SampleReader

    return SampleReader


_ARROW = patch_arrow()
pa = _ARROW.module
flight = _ARROW.flight
if flight is None:
    raise ImportError("pyarrow.flight is required for Endpoint support")


class Endpoint:
    @staticmethod
    def resource_key(memmap_dir: str, split: str) -> str:
        return f"{os.path.basename(memmap_dir.rstrip(os.sep))}/{split}"

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
        elif isinstance(value, memoryview):
            return value.tobytes()
        elif isinstance(value, str):
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
                    parts.append(Endpoint._bytes_of(part).decode("utf-8"))
                except Exception:
                    parts.append(str(part))
            return Endpoint._canon_name("/".join(parts))
        command = getattr(descriptor, "command", None)
        if command is not None:
            try:
                return Endpoint._canon_name(Endpoint._bytes_of(command))
            except Exception:
                return Endpoint._canon_name(command)
        return Endpoint._canon_name(descriptor)

    @staticmethod
    def _build_descriptor(name: str) -> flight.FlightDescriptor:
        canonical = Endpoint._canon_name(name)
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
            del context, criteria
            with self._lock:
                for name, reader in self._datasets.items():
                    canonical = Endpoint._canon_name(name)
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
            del context
            name = Endpoint._name_from_descriptor(descriptor)
            key = Endpoint._canon_name(name)
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
            del context
            name = Endpoint._canon_name(ticket.ticket)
            with self._lock:
                reader = self._datasets[name]
            return flight.RecordBatchStream(reader)

        def add_reader(self, name: str, reader: pa.RecordBatchReader) -> None:
            with self._lock:
                key = Endpoint._canon_name(name)
                self._datasets[key] = reader

    class Client:
        def __init__(
            self, uri: str | Tuple[str, int] | flight.Location
        ) -> None:
            self._client = flight.FlightClient(uri)

        def reader(
            self,
            name: str,
            *args: Any,
            timeout_s: float = 30.0,
            poll_interval_s: float = 0.05,
            **kwargs: Any,
        ) -> pa.RecordBatchReader:
            descriptor = Endpoint._build_descriptor(name)
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
    def _serve_in_thread(
        server: "Endpoint.Server", thread_error: list[BaseException]
    ) -> None:
        try:
            server.serve()
        except BaseException as exc:
            thread_error.append(exc)
            raise

    @staticmethod
    def _await_server_bind(
        server: "Endpoint.Server",
        resolved_port: int,
        fallback_port: int,
        deadline: float,
        bound: threading.Event,
        thread: threading.Thread,
        thread_error: list[BaseException],
    ) -> int:
        while time.time() < deadline:
            actual = getattr(server, "port", 0) or resolved_port
            if actual not in (None, 0):
                bound.set()
                return actual
            if thread_error:
                raise RuntimeError(
                    "Arrow Flight server thread failed during startup"
                ) from thread_error[0]
            if not thread.is_alive():
                raise RuntimeError(
                    "Arrow Flight server thread exited before binding to a port"
                )
            time.sleep(0.01)
        return getattr(server, "port", 0) or fallback_port

    @staticmethod
    def _iter_batches(
        mmts: "SampleReader",
        batch_size: int,
        start: int,
        end: int,
    ) -> Iterator[pa.RecordBatch]:
        mmts_cls = _memory_mapped_tensor_stream()
        index = start
        while index < end:
            nxt = min(index + int(batch_size), end)
            xb, yb = mmts.batch_range(index, nxt)
            yield mmts_cls.to_record_batch(xb, yb)
            index = nxt

    @staticmethod
    def start_server_standby(
        *args: Any,
        host: str = "0.0.0.0",
        port: int = 0,
        wait_ready_s: float = 10.0,
        **kwargs: Any,
    ) -> Tuple[Endpoint.Server, str]:
        locator = Network()
        resolved = locator.allocate(f"{host}:{port}" if host else None)
        parsed_endpoint = urlparse(f"grpc://{resolved}")
        resolved_host = parsed_endpoint.hostname or (host or "0.0.0.0")
        resolved_port = parsed_endpoint.port or port or 0
        location_obj = None
        try:
            location_obj = flight.Location.for_grpc_tcp(resolved_host, resolved_port)
        except Exception:
            location_obj = None
        location = (
            location_obj if location_obj is not None else parsed_endpoint.geturl()
        )
        server = Endpoint.Server(location=location)
        thread_error: list[BaseException] = []
        bound = threading.Event()

        advertise_host = resolved_host
        if advertise_host in ("", "0.0.0.0", "::"):
            advertise_host = get_preferred_ip(allow_loopback=True)
            if not advertise_host:
                advertise_host = "127.0.0.1"

        thread = threading.Thread(
            target=Endpoint._serve_in_thread,
            args=(server, thread_error),
            daemon=True,
        )
        thread.start()
        deadline = time.time() + float(wait_ready_s)

        actual_port = Endpoint._await_server_bind(
            server,
            resolved_port,
            port,
            deadline,
            bound,
            thread,
            thread_error,
        )
        if not bound.is_set():
            if thread_error:
                raise RuntimeError(
                    f"Arrow Flight server failed to bind on {host}"
                ) from thread_error[0]
            raise RuntimeError(
                f"Arrow Flight server did not bind to a port on {host}"
            )
        advertise_location = None
        try:
            advertise_location = flight.Location.for_grpc_tcp(
                advertise_host, actual_port
            )
        except Exception:
            advertise_location = None
        formatter = Network(
            fallback=advertise_host,
            default=advertise_host,
            allow_loopback=True,
        )
        advertise_uri_host = formatter.format(advertise_host)
        if advertise_location is not None:
            uri_bytes = getattr(advertise_location, "uri", None)
            if isinstance(uri_bytes, (bytes, bytearray)):
                uri = uri_bytes.decode("utf-8", "ignore")
            else:
                uri = f"grpc+tcp://{advertise_uri_host}:{actual_port}"
        else:
            uri = f"grpc+tcp://{advertise_uri_host}:{actual_port}"
        for key in ("NO_PROXY", "no_proxy"):
            current = os.environ.get(key, "")
            entries = [entry.strip() for entry in current.split(",") if entry.strip()]
            if advertise_host and advertise_host not in entries:
                entries.append(advertise_host)
                os.environ[key] = ",".join(entries)
        deadline = time.time() + float(wait_ready_s)
        ready = False
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                with flight.FlightClient(advertise_location or uri) as client:
                    list(client.list_flights())
                ready = True
                break
            except Exception as exc:
                last_error = exc
                time.sleep(0.05)
        if not ready:
            if thread_error:
                raise RuntimeError(
                    f"Arrow Flight server not ready: {uri}"
                ) from thread_error[0]
            if last_error is not None:
                raise RuntimeError(
                    f"Arrow Flight server not ready: {uri}"
                ) from last_error
            raise RuntimeError(f"Arrow Flight server not ready: {uri}")
        return (server, uri)

    @staticmethod
    def reg_mmt_dataset(
        server: Endpoint.Server,
        name: str,
        mmts: "SampleReader",
        batch_size: int,
        split: str,
    ) -> None:
        meta = mmts._load_meta()
        total = int(meta["N"])
        fractions = meta.get("fractions", [1.0, 0.0])
        train_end = int(total * float(fractions[0])) if fractions else total
        start, end = (0, train_end) if split == "train" else (train_end, total)

        generator = Endpoint._iter_batches(mmts, batch_size, start, end)
        first = next(generator)
        rest = list(generator)
        schema = first.schema
        reader = pa.RecordBatchReader.from_batches(schema, [first, *rest])
        server.add_reader(Endpoint._canon_name(name), reader)

    @staticmethod
    def connect(
        endpoint: str | Tuple[str, int] | flight.Location,
        *args: Any,
        wait_ready_s: float = 5.0,
        poll_interval_s: float = 0.05,
        **kwargs: Any,
    ) -> Endpoint.Client:
        deadline = time.time() + float(wait_ready_s)
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                client = Endpoint.Client(endpoint)
                list(client._client.list_flights())
                return client
            except Exception as exc:
                last_error = exc
                time.sleep(float(poll_interval_s))
        raise RuntimeError(
            f"Unable to connect to Arrow Flight endpoint: {endpoint}"
        ) from last_error


def client(uri: str | Tuple[str, int] | flight.Location) -> flight.FlightClient:
    return flight.FlightClient(uri)


def server(host: str = "0.0.0.0", port: int = 0) -> Tuple[Endpoint.Server, str]:
    return Endpoint.start_server_standby(host=host, port=port)
