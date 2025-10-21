# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import mmap
import os
from types import TracebackType
from typing import Any, Iterator, Optional

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.cuda as pa_cuda
except Exception:
    pa = None
    pa_cuda = None
try:
    from multiprocessing import shared_memory
except Exception:
    shared_memory = None
try:
    import kvikio
except Exception:
    kvikio = None


class MemoryMap:
    def __init__(
        self,
        path: str,
        size: int,
        *args: Any,
        access: int = mmap.ACCESS_WRITE,
        **kwargs: Any,
    ) -> None:
        size = int(size)
        if size <= 0:
            raise ValueError("size must be > 0")
        self.path = path
        self.size = size
        self.access = access
        self._fd: Optional[int] = None
        self._mmap: Optional[mmap.mmap] = None

    def open(self) -> mmap.mmap:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT)
        try:
            os.ftruncate(fd, self.size)
            mm = mmap.mmap(fd, self.size, access=self.access)
        except Exception:
            os.close(fd)
            raise
        self._fd = fd
        self._mmap = mm
        return mm

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.flush()
            self._mmap.close()
            self._mmap = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    @property
    def is_open(self) -> bool:
        return self._mmap is not None

    @contextlib.contextmanager
    def mapped(self) -> Iterator[mmap.mmap]:
        mm = self.open()
        try:
            yield mm
        finally:
            self.close()

    def __enter__(self) -> mmap.mmap:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def to_arrow_buffer(self) -> "pa.Buffer":
        if pa is None:
            raise RuntimeError(
                "pyarrow is required for converting to Arrow buffers"
            )
        if self._mmap is None:
            raise RuntimeError("call open() first")
        return pa.py_buffer(memoryview(self._mmap))


class SharedMemory:
    def __init__(
        self,
        name: str,
        size: int,
        *args: Any,
        create: bool = False,
        **kwargs: Any,
    ) -> None:
        if shared_memory is None:
            raise RuntimeError(
                "multiprocessing.shared_memory is not available on this system"
            )
        self._shm = shared_memory.SharedMemory(
            name=name, create=create, size=size
        )
        self.name: str = self._shm.name
        self.size: int = self._shm.size

    @property
    def buf(self) -> memoryview:
        return self._shm.buf

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        self._shm.unlink()


class CudaIpc:
    def __init__(self) -> None:
        if pa_cuda is None or pa is None:
            raise RuntimeError(
                "pyarrow with CUDA support is required (see Arrow CUDA docs)"
            )

    def export_handle(self, buf: "pa_cuda.CudaBuffer") -> bytes:
        handle = buf.export_for_ipc()
        serialized = handle.serialize()
        return serialized.to_pybytes()

    def open_handle(self, handle_bytes: bytes) -> "pa_cuda.CudaBuffer":
        assert pa is not None and pa_cuda is not None
        handle_buf = pa.py_buffer(handle_bytes)
        ipc_handle = pa_cuda.IpcMemHandle.from_buffer(handle_buf)
        ctx = pa_cuda.Context()
        return ctx.open_ipc_buffer(ipc_handle)


class Ucxx:
    def __init__(self) -> None:
        self._backend: str
        self._module: Any
        try:
            import ucxx as backend

            self._backend = "ucxx"
            self._module = backend
            return
        except Exception:
            pass
        try:
            import ucp as backend

            self._backend = "ucx-py"
            self._module = backend
        except Exception as exc:
            raise RuntimeError(
                "Neither 'ucxx' nor 'ucx-py' is installed"
            ) from exc

    async def local_echo(self, payload: bytes) -> bytes:
        if self._backend == "ucxx":
            try:
                ucxx = self._module
                w = ucxx.create_worker()
                addr = w.getAddress()
                ep1 = w.createEndpointFromWorkerAddress(addr)
                ep2 = w.createEndpointFromWorkerAddress(addr)
                tx = np.frombuffer(payload, dtype=np.uint8)
                rx = np.empty_like(tx)
                await ep1.send(tx)
                await ep2.recv(rx)
                return bytes(rx)
            except Exception:
                pass
        if self._backend == "ucx-py":
            try:
                ucp = self._module
                addr = ucp.get_worker_address()
                ep1 = await ucp.create_endpoint_from_worker_address(addr)
                ep2 = await ucp.create_endpoint_from_worker_address(addr)
                tx = np.frombuffer(payload, dtype=np.uint8)
                rx = np.empty_like(tx)
                await ep1.send(tx)
                await ep2.recv(rx)
                return bytes(rx)
            except Exception:
                return payload
        return payload


class GpuDirectStorage:
    def __init__(self, path: str, mode: str = "r") -> None:
        if kvikio is None:
            raise RuntimeError("kvikio is required for GPUDirect Storage")
        self.path = path
        self.mode = mode

    def open(self) -> Any:
        CuFile = getattr(kvikio, "CuFile", None)
        if CuFile is None:
            CuFile = getattr(getattr(kvikio, "cufile", None), "CuFile", None)
        if CuFile is None:
            raise RuntimeError(
                "kvikio.CuFile is not available in this KvikIO version"
            )
        return CuFile(self.path, self.mode)
