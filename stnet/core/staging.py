# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Optional, Protocol, Tuple, runtime_checkable

import torch

from .casting import env_first, env_first_float, env_flag

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

    @dataclass(slots=True)
    class Token:
        i: int
        g: int

    @dataclass(slots=True)
    class _Entry:
        page: Page
        busy: bool = False
        fence: object | None = None
        fence_evt: object | None = None
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

    def fence_event(
        self,
        token: Pool.Token | None,
        factory: Callable[[], object] | None,
    ) -> object | None:
        if token is None or factory is None:
            return None

        i = int(getattr(token, "i", -1))
        g = int(getattr(token, "g", -1))
        if i < 0:
            return None

        # Fast path: already created and still matches this generation.
        with self._cv:
            if 0 <= i < len(self._pages):
                e = self._pages[i]
                if e.gen == g and e.fence_evt is not None:
                    return e.fence_evt

        # Create outside the lock (backend event creation can be slow).
        try:
            ev_new = factory()
        except Exception:
            return None
        if ev_new is None:
            return None

        with self._cv:
            if 0 <= i < len(self._pages):
                e = self._pages[i]
                if e.gen == g:
                    if e.fence_evt is None:
                        e.fence_evt = ev_new
                        return ev_new
                    return e.fence_evt

        # Token no longer matches a live entry; return the new event anyway.
        return ev_new


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


@lru_cache(maxsize=1)
def _cache_backpressure_mode() -> str:
    raw = env_first(
        ("STNET_CACHE_BACKPRESSURE_MODE", "STNET_CACHE_BACKPRESSURE", "STNET_CACHE_MODE"),
        default="block",
    )
    s = str(raw or "block").strip().lower()
    if s in {"sync", "synchronous"}:
        return "sync"
    if s in {"raise", "error"}:
        return "raise"
    return "block"


@lru_cache(maxsize=1)
def _cache_backpressure_timeout_s() -> float:
    t = env_first_float(
        ("STNET_CACHE_BACKPRESSURE_TIMEOUT_S", "STNET_CACHE_SUBMIT_TIMEOUT_S"),
        default=0.05,
    )
    with contextlib.suppress(Exception):
        return max(0.0, float(t))
    return 0.05


@lru_cache(maxsize=1)
def _cache_early_release_enabled() -> bool:
    return bool(env_flag("STNET_CACHE_EARLY_RELEASE", "STNET_CACHE_RELEASE_EARLY", default=True))


@lru_cache(maxsize=1)
def _cache_force_unpin_enabled() -> bool:
    return bool(env_flag("STNET_CACHE_FORCE_UNPIN", "STNET_CACHE_UNPIN", default=False))


class Cache:

    def __init__(self, root: str, max_queue: int = 8) -> None:
        import queue
        import threading

        self._root = os.fspath(root)
        os.makedirs(self._root, exist_ok=True)

        # Always keep the queue bounded (max_queue<=0 is coerced to 1 to avoid OOM).
        max_q = max(1, int(max_queue))
        self._sem = threading.Semaphore(max_q)
        self._q: "queue.SimpleQueue[tuple[Any, Any, Any, Any]]" = queue.SimpleQueue()

        # Cache env knobs at construction time (hot path should not repeatedly parse env vars).
        self._bp_mode = str(_cache_backpressure_mode() or "block")
        self._bp_timeout_s = float(_cache_backpressure_timeout_s() or 0.0)
        self._early_release = bool(_cache_early_release_enabled())
        self._force_unpin = bool(_cache_force_unpin_enabled())

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
        sem = self._sem
        if sem is not None:
            mode = str(getattr(self, "_bp_mode", "block") or "block").lower()
            timeout_s = float(getattr(self, "_bp_timeout_s", 0.05) or 0.0)

            if mode == "sync":
                acquired = bool(sem.acquire(timeout=max(0.0, float(timeout_s))))
                if not acquired:
                    # Synchronous fallback: preserve correctness over throughput.
                    self._wait(wait_event)
                    buf, released = self._prepare_tensor_for_save(
                        tensor,
                        release_cb=(release_cb if callable(release_cb) else None),
                        early_release=False,
                        force_unpin=bool(getattr(self, "_force_unpin", False)),
                    )
                    self._save_tensor(buf, path)
                    if callable(release_cb) and not released:
                        with contextlib.suppress(Exception):
                            release_cb()
                    return

            elif mode == "raise":
                acquired = bool(sem.acquire(timeout=max(0.0, float(timeout_s))))
                if not acquired:
                    raise RuntimeError("Cache queue is full")

            else:
                # Block (default): wait for the writer thread to catch up.
                if timeout_s > 0:
                    acquired = bool(sem.acquire(timeout=float(timeout_s)))
                while not acquired:
                    if self._err_event.is_set():
                        raise RuntimeError(f"Async writer error: {self._err!r}")
                    if self._closed.is_set():
                        raise RuntimeError("Cache is closed")
                    acquired = bool(sem.acquire(timeout=0.1))

        try:
            self._q.put((tensor, path, wait_event, release_cb))
        except Exception:
            if acquired and sem is not None:
                with contextlib.suppress(Exception):
                    sem.release()
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

    @staticmethod
    def _is_pinned_cpu(t: torch.Tensor) -> bool:
        if not torch.is_tensor(t):
            return False
        if getattr(t, "device", None) is None or t.device.type != "cpu":
            return False
        is_pinned = getattr(t, "is_pinned", None)
        if callable(is_pinned):
            with contextlib.suppress(Exception):
                return bool(is_pinned())
        return False

    def _prepare_tensor_for_save(
        self,
        tensor: torch.Tensor,
        *,
        release_cb: Callable[[], Any] | None = None,
        early_release: bool | None = None,
        force_unpin: bool | None = None,
    ) -> tuple[torch.Tensor, bool]:
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)

        buf = tensor.detach()
        if buf.device.type != "cpu":
            buf = buf.to(device="cpu", non_blocking=False)

        if early_release is None:
            early_release = bool(getattr(self, "_early_release", True))
        if force_unpin is None:
            force_unpin = bool(getattr(self, "_force_unpin", False))

        released = False
        pinned = Cache._is_pinned_cpu(buf)

        # If a pool-backed pinned buffer is handed in, we can free it for reuse by copying into a
        # pageable buffer and releasing the pool token before disk I/O.
        if pinned and (bool(force_unpin) or (bool(early_release) and callable(release_cb))):
            try:
                tmp = torch.empty_like(buf, device="cpu", pin_memory=False)
                tmp.copy_(buf, non_blocking=False)
                buf = tmp
                if bool(early_release) and callable(release_cb):
                    with contextlib.suppress(Exception):
                        release_cb()
                    released = True
            except Exception:
                # Best-effort fallback: keep the pinned buffer and release after saving.
                released = False

        if not bool(buf.is_contiguous()):
            buf = buf.contiguous()

        return buf, released

    def _save_tensor(self, tensor: torch.Tensor, path: str) -> None:
        import os as _os

        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)

        buf = tensor.detach()
        if buf.device.type != "cpu":
            buf = buf.to(device="cpu", non_blocking=False)
        if not bool(buf.is_contiguous()):
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
            from ..data.pipeline import BatchIO

            meta = {
                "shape": [int(x) for x in buf.shape],
                "dtype": str(buf.dtype).replace("torch.", ""),
            }
            BatchIO.atomic_write_json(BatchIO.mmt_meta_path(path), meta, indent=None)
            return

        # torch.save pickles (rows.pt / pred.pt)
        if str(path).endswith((".pt", ".pth")):
            from ..data.pipeline import BatchIO

            BatchIO.atomic_torch_save(buf, path)
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

            rel_cb = rel if callable(rel) else None
            released_early = False

            try:
                self._wait(evt)
                buf, released_early = self._prepare_tensor_for_save(
                    tensor,
                    release_cb=rel_cb,
                    early_release=bool(getattr(self, "_early_release", True)),
                    force_unpin=bool(getattr(self, "_force_unpin", False)),
                )
                self._save_tensor(buf, path)

                if rel_cb is not None and not released_early:
                    with contextlib.suppress(Exception):
                        rel_cb()

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

    exc: BaseException
    tb: str


def best_effort_close(obj: Any, *, join_timeout: float | None = 1.0) -> None:
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

    def __init__(self, max_batches: int) -> None:
        import collections
        import threading

        self.max_batches = max(1, int(max_batches))
        self._buf: "collections.deque[Any]" = collections.deque()
        self._stop = threading.Event()
        self._cv = threading.Condition()
        self._warn_blocking = env_flag("STNET_BUFFER_WARN_BLOCKING", "STNET_DEBUG", default=False)

    def put(self, item: Any, *, timeout: float | None = None) -> bool:
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

    def wait_for_space(self, *, timeout: float | None = None) -> bool:
        import time

        if self._stop.is_set():
            return False

        t0 = time.monotonic()
        deadline = None if timeout is None else (t0 + float(timeout))
        with self._cv:
            while len(self._buf) >= self.max_batches and not self._stop.is_set():
                if deadline is None:
                    self._cv.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._cv.wait(timeout=remaining)
            return not bool(self._stop.is_set())

    def clear(self) -> None:
        with self._cv:
            self._buf.clear()
            self._cv.notify_all()

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def is_stopped(self) -> bool:
        return bool(self._stop.is_set())

    def __len__(self) -> int:  # pragma: no cover
        return self.size()
