# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Protocol, runtime_checkable

import torch

from .datatype import env_flag as _env_flag


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _prod_int(shape: Sequence[int]) -> int:
    try:
        n = int(math.prod(int(s) for s in shape))
    except Exception:
        n = 1
        for s in shape:
            n *= int(s)
    return int(max(1, n))


@runtime_checkable
class _QueryEvent(Protocol):
    def query(self) -> bool: ...


@runtime_checkable
class _SyncEvent(Protocol):
    def synchronize(self) -> Any: ...


@runtime_checkable
class _WaitEvent(Protocol):
    def wait(self, timeout: float | None = None) -> Any: ...



# -----------------------------------------------------------------------------
# Pinned CPU page + pool
# -----------------------------------------------------------------------------

class Page:
    """Resizable pinned (or regular) CPU buffer used for staging."""

    __slots__ = ("_buf", "_numel", "_dtype", "_pinned")

    def __init__(self, numel: int, dtype: torch.dtype, *, pin_memory: bool = True) -> None:
        self._numel = int(max(1, int(numel)))
        self._dtype = dtype
        self._pinned = bool(pin_memory)
        self._buf = torch.empty(
            self._numel,
            dtype=self._dtype,
            device="cpu",
            pin_memory=bool(self._pinned),
        )

    @property
    def numel(self) -> int:
        return int(self._numel)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def pinned(self) -> bool:
        return bool(self._pinned)

    def ensure(self, numel: int) -> None:
        need = int(max(1, int(numel)))
        if need <= int(self._numel):
            return
        self._numel = need
        self._buf = torch.empty(
            self._numel,
            dtype=self._dtype,
            device="cpu",
            pin_memory=bool(self._pinned),
        )

    def view(self, *shape: int) -> torch.Tensor:
        need = _prod_int(shape)
        self.ensure(need)
        return self._buf[:need].view(*shape)


class Pool:
    """Small reusable CPU staging pool.

    - Reuses pinned CPU buffers for fast H2D.
    - Avoids unbounded pinned allocations under contention by falling back to
      one-off **unpinned** buffers when capacity is exhausted (unless block=True).
    """

    @dataclass(slots=True)
    class Token:
        i: int
        g: int

    @dataclass(slots=True)
    class _Entry:
        page: Page
        busy: bool = False
        fence: object | None = None
        gen: int = 0

    def __init__(self, capacity: int = 4, *, pin_memory: bool = True) -> None:
        import threading

        self._cap = max(1, int(capacity))
        self._pin = bool(pin_memory)
        self._pages: list[Pool._Entry] = []
        self._rr = 0
        self._cv = threading.Condition()

    @property
    def capacity(self) -> int:
        return int(self._cap)

    def _evt_done(self, evt: object | None) -> bool:
        if evt is None:
            return True
        if isinstance(evt, _QueryEvent):
            try:
                return bool(evt.query())
            except Exception:
                return False
        is_set = getattr(evt, "is_set", None)
        if callable(is_set):
            try:
                return bool(is_set())
            except Exception:
                return False
        return False

    def _scavenge_locked(self) -> int:
        freed = 0
        for e in self._pages:
            if e.busy and e.fence is not None and self._evt_done(e.fence):
                e.busy = False
                e.fence = None
                freed += 1
        return freed

    def get(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        *,
        return_handle: bool = False,
        block: bool = False,
        timeout: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Pool.Token | None]:
        """Get a CPU staging tensor.

        If the pool is exhausted:
          - block=False (default): return an untracked **unpinned** tensor.
          - block=True: wait for a page to become available (with periodic fence checks).

        Returns:
          - tensor (default)
          - (tensor, Token|None) when return_handle=True
        """
        import time

        shape_t = tuple(int(s) for s in shape)
        dtype_t = dtype
        need = _prod_int(shape_t)

        deadline: float | None = None
        if timeout is not None:
            deadline = time.monotonic() + float(timeout)

        # Periodic fence checks when blocking (CUDA events cannot notify us).
        check_interval = 0.01

        while True:
            # Phase 1: select an entry or decide to grow/wait/overflow.
            idx: int | None = None
            need_new_page = False
            want_grow = False

            with self._cv:
                self._scavenge_locked()
                n = len(self._pages)

                if n:
                    start = self._rr % n
                    for k in range(n):
                        j = (start + k) % n
                        e = self._pages[j]
                        if not e.busy:
                            e.busy = True
                            e.fence = None
                            idx = j
                            self._rr = (j + 1) % max(1, n)
                            # Decide if we must replace the page (dtype/size/pin mismatch).
                            if (e.page.dtype != dtype_t) or (e.page.numel < need) or (e.page.pinned != self._pin):
                                need_new_page = True
                            break

                if idx is None:
                    if n < self._cap:
                        want_grow = True
                    else:
                        if block:
                            if deadline is not None:
                                remaining = deadline - time.monotonic()
                                if remaining <= 0:
                                    want_grow = False
                                    break
                                self._cv.wait(timeout=min(check_interval, remaining))
                            else:
                                self._cv.wait(timeout=check_interval)
                            continue
                        break

            # Phase 2: materialize (possibly allocate) outside the lock.
            if idx is not None:
                new_page: Page | None = None
                if need_new_page:
                    new_page = Page(numel=need, dtype=dtype_t, pin_memory=self._pin)

                with self._cv:
                    # Entry is still busy and reserved for us.
                    e = self._pages[idx]
                    if need_new_page and new_page is not None:
                        e.page = new_page
                        e.gen += 1
                    gen = int(e.gen)

                view = e.page.view(*shape_t)
                if return_handle:
                    return view, Pool.Token(int(idx), gen)
                return view

            if want_grow:
                # Allocate new page outside lock.
                page = Page(numel=need, dtype=dtype_t, pin_memory=self._pin)
                entry = Pool._Entry(page=page, busy=True, fence=None, gen=0)

                with self._cv:
                    if len(self._pages) < self._cap:
                        self._pages.append(entry)
                        idx2 = len(self._pages) - 1
                        self._rr = (idx2 + 1) % self._cap
                        view = entry.page.view(*shape_t)
                        if return_handle:
                            return view, Pool.Token(int(idx2), int(entry.gen))
                        return view
                # Lost the race to grow; retry.
                continue

            # Overflow or timeout.
            break

        # Overflow: allocate one-off **unpinned** tensor (untracked).
        view = torch.empty(need, dtype=dtype_t, device="cpu", pin_memory=False).view(*shape_t)
        if return_handle:
            return view, None
        return view

    def get_like(
        self,
        t: torch.Tensor,
        *args: Any,
        return_handle: bool = False,
        block: bool = False,
        timeout: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Pool.Token | None]:
        return self.get(
            tuple(int(s) for s in t.shape),
            t.dtype,
            return_handle=return_handle,
            block=block,
            timeout=timeout,
        )

    def release_after(self, token: Pool.Token | None, wait_event: object | None) -> None:
        if token is None:
            return
        with self._cv:
            i = int(getattr(token, "i", -1))
            g = int(getattr(token, "g", -1))
            if 0 <= i < len(self._pages):
                e = self._pages[i]
                if e.gen == g:
                    e.busy = True
                    e.fence = wait_event
                    self._cv.notify()

    def release(self, token: Pool.Token | None) -> None:
        if token is None:
            return
        with self._cv:
            i = int(getattr(token, "i", -1))
            g = int(getattr(token, "g", -1))
            if 0 <= i < len(self._pages):
                e = self._pages[i]
                if e.gen == g:
                    e.busy = False
                    e.fence = None
                    self._cv.notify()

    def collect(self) -> None:
        with self._cv:
            freed = self._scavenge_locked()
            if freed:
                self._cv.notify_all()


# -----------------------------------------------------------------------------
# Async on-disk cache writer
# -----------------------------------------------------------------------------

class Cache:
    """Asynchronous tensor writer with bounded backpressure."""

    def __init__(self, root: str, max_queue: int = 8) -> None:
        import queue
        import threading

        self._root = os.fspath(root)
        os.makedirs(self._root, exist_ok=True)

        max_q = int(max_queue)
        self._sem = threading.Semaphore(max_q) if max_q > 0 else None
        self._q: "queue.SimpleQueue[tuple[Any, Any, Any, Any]]" = queue.SimpleQueue()

        self._t = threading.Thread(target=self._run, daemon=True)
        self._err: BaseException | None = None
        self._err_event = threading.Event()
        self._closed = threading.Event()
        self._t.start()

    def submit(
        self,
        tensor: torch.Tensor,
        path: Optional[str] = None,
        idx: Optional[int] = None,
        wait_event: Optional[object] = None,
        release_cb: Optional[object] = None,
    ) -> None:
        """Submit a tensor for async saving.

        If a bounded queue is configured and backpressure acquisition times out,
        falls back to synchronous saving in the caller thread.
        """
        import os as _os

        if self._err_event.is_set():
            raise RuntimeError(f"Async writer error: {self._err!r}")
        if self._closed.is_set():
            raise RuntimeError("Cache is closed")

        if path is None:
            if idx is None:
                raise ValueError("either path or idx required")
            path = _os.path.join(self._root, f"chunk_{int(idx):06d}.pt")
        path = _os.fspath(path)

        acquired = False
        if self._sem is not None:
            acquired = bool(self._sem.acquire(timeout=0.05))
            if not acquired:
                # Synchronous fallback: preserve correctness over throughput.
                self._wait(wait_event)
                self._save_tensor(tensor, path)
                if callable(release_cb):
                    with contextlib.suppress(Exception):
                        release_cb()
                return

        try:
            self._q.put((tensor, path, wait_event, release_cb))
        except Exception:
            if acquired and self._sem is not None:
                with contextlib.suppress(Exception):
                    self._sem.release()
            raise

    def _wait(self, evt: object | None) -> None:
        if evt is None:
            return
        try:
            if isinstance(evt, _SyncEvent):
                evt.synchronize()
                return
        except Exception:
            pass
        try:
            if isinstance(evt, _WaitEvent):
                evt.wait()
                return
        except Exception:
            pass

    def _save_tensor(self, tensor: torch.Tensor, path: str) -> None:
        import os as _os

        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)

        buf = tensor.detach()
        if buf.device.type != "cpu":
            buf = buf.to(device="cpu", non_blocking=False)

        if hasattr(buf, "is_pinned") and callable(getattr(buf, "is_pinned")) and bool(buf.is_pinned()):
            tmp = torch.empty_like(buf, device="cpu", pin_memory=False)
            tmp.copy_(buf, non_blocking=False)
            buf = tmp

        buf = buf.contiguous()

        if str(path).endswith(".mmt"):
            from tensordict import MemoryMappedTensor

            # Write to a temp file and atomically replace, so readers never see a half-written .mmt.
            parent = _os.path.dirname(path) or "."
            _os.makedirs(parent, exist_ok=True)

            import tempfile as _tempfile

            fd, tmp_name = _tempfile.mkstemp(prefix=_os.path.basename(path) + ".", suffix=".tmp", dir=parent)
            _os.close(fd)
            try:
                MemoryMappedTensor.from_tensor(buf, filename=tmp_name, existsok=True)
                _os.replace(tmp_name, path)
            finally:
                with contextlib.suppress(Exception):
                    _os.remove(tmp_name)

            # Sidecar meta: keep the naming consistent across the codebase.
            from .pipeline import BatchIterator

            meta = {
                "shape": [int(x) for x in buf.shape],
                "dtype": str(buf.dtype).replace("torch.", ""),
            }
            BatchIterator.atomic_write_json(BatchIterator.mmt_meta_path(path), meta, indent=None)
            return

        # torch.save pickles (rows.pt / pred.pt)
        if str(path).endswith((".pt", ".pth")):
            # Use the shared atomic writer for consistency.
            from .pipeline import BatchIterator

            BatchIterator.atomic_torch_save(buf, path)
        else:
            torch.save(buf, path)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._q.put((None, None, None, None))
        self._t.join()

    def __enter__(self):  # pragma: no cover
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover
        self.close()
        return False

    def _run(self) -> None:
        while True:
            item = self._q.get()
            match item:
                case (tensor, path, evt, rel):
                    pass
                case _:
                    self._err = RuntimeError(f"Invalid cache queue item: {type(item)!r}")
                    self._err_event.set()
                    break

            if tensor is None:
                break

            try:
                self._wait(evt)
                self._save_tensor(tensor, path)
                if callable(rel):
                    with contextlib.suppress(Exception):
                        rel()
            except Exception as e:
                self._err = e
                self._err_event.set()
                break
            finally:
                if self._sem is not None:
                    with contextlib.suppress(Exception):
                        self._sem.release()

    def had_error(self) -> bool:
        return bool(self._err_event.is_set())


@dataclass(slots=True)
class ProducerError:
    """Payload wrapper used to forward producer exceptions to the consumer."""

    exc: BaseException
    tb: str


def best_effort_close(obj: Any, *, join_timeout: float | None = 1.0) -> None:
    """Best-effort resource cleanup for common close/stop/join APIs."""
    for name in (
        "cleanup",
        "close",
        "shutdown",
        "stop",
        "terminate",
        "disconnect",
        "release",
        "join",
    ):
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                if name == "join" and join_timeout is not None:
                    fn(timeout=float(join_timeout))
                else:
                    fn()
            except Exception:
                pass
            return

    if callable(obj):
        try:
            obj()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Buffer: bounded in-memory buffer with stop notification
# -----------------------------------------------------------------------------

class Buffer:
    """Bounded in-memory buffer with backpressure and stop notification."""

    def __init__(self, max_batches: int) -> None:
        import collections
        import threading

        self.max_batches = max(1, int(max_batches))
        self._buf: "collections.deque[Any]" = collections.deque()
        self._stop = threading.Event()
        self._cv = threading.Condition()
        self._warn_blocking = _env_flag("STNET_BUFFER_WARN_BLOCKING", "STNET_DEBUG", default=False)

    def put(self, item: Any, *, timeout: float | None = None) -> bool:
        """Put an item into the buffer."""
        import logging
        import time

        if self._stop.is_set():
            return False

        t0 = time.monotonic()
        deadline = None if timeout is None else (t0 + float(timeout))
        with self._cv:
            if self._stop.is_set():
                return False

            while len(self._buf) >= self.max_batches and not self._stop.is_set():
                if deadline is None:
                    self._cv.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._cv.wait(timeout=remaining)
            if self._stop.is_set():
                return False

            self._buf.append(item)
            self._cv.notify()

        elapsed = time.monotonic() - t0
        if self._warn_blocking and elapsed > 0.1:
            logging.warning(
                "Buffer.put blocked for %.3f s (max_batches=%d)",
                float(elapsed),
                int(self.max_batches),
            )
        return True

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        """Get an item from the buffer."""
        import queue
        import time

        if not bool(block):
            with self._cv:
                if not self._buf:
                    raise queue.Empty
                item = self._buf.popleft()
                self._cv.notify()
                return item

        t0 = time.monotonic()
        deadline = None if timeout is None else (t0 + float(timeout))

        with self._cv:
            while not self._buf and not self._stop.is_set():
                if deadline is None:
                    self._cv.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise queue.Empty
                    self._cv.wait(timeout=remaining)

            if not self._buf:
                raise queue.Empty
            item = self._buf.popleft()
            self._cv.notify()
            return item

    def empty(self) -> bool:
        with self._cv:
            return not bool(self._buf)

    def size(self) -> int:
        with self._cv:
            return int(len(self._buf))

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def is_stopped(self) -> bool:
        return bool(self._stop.is_set())

    def __len__(self) -> int:  # pragma: no cover
        return self.size()
