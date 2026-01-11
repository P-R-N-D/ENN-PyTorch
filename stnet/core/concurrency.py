# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import ctypes
import importlib
import itertools
import platform
import sys
import math
import os
import collections
import logging
import queue
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from typing import Any, Callable, Optional, Protocol, Tuple, runtime_checkable

import torch

from .datatypes import env_first, env_first_float, env_first_int, env_flag
from .policies import WorkerPolicy, optimize_threads

from .system import (
    CPU,
    _default_thread_limit,
    _optimal_local_worlds,
    _optimal_threads,
)


def _get_throttle_state() -> str:
    s = (
        str(
            env_first(
                ("STNET_CACHE_BACKPRESSURE_MODE", "STNET_CACHE_BACKPRESSURE", "STNET_CACHE_MODE"),
                default="block",
            )
            or "block"
        )
        .strip()
        .lower()
    )
    return (
        "sync"
        if s in {"sync", "synchronous"}
        else ("raise" if s in {"raise", "error"} else "block")
    )


@lru_cache(maxsize=1)
def _get_throttle_timeout() -> float:
    return max(
        0.0,
        float(
            env_first_float(
                ("STNET_CACHE_BACKPRESSURE_TIMEOUT_S", "STNET_CACHE_SUBMIT_TIMEOUT_S"), default=0.05
            )
            or 0.05
        ),
    )


@lru_cache(maxsize=1)
def _is_early_release_enabled() -> bool:
    return bool(env_flag("STNET_CACHE_EARLY_RELEASE", "STNET_CACHE_RELEASE_EARLY", default=True))


@lru_cache(maxsize=1)
def _is_force_unpin_enabled() -> bool:
    return bool(env_flag("STNET_CACHE_FORCE_UNPIN", "STNET_CACHE_UNPIN", default=False))


def _prod_int(shape: Sequence[int]) -> int:
    return int(max(1, math.prod(int(s) for s in shape)))


def close(obj: Any, *args: Any, join_timeout: float | None = 1.0) -> None:
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
        if callable(fn := getattr(obj, name, None)):
            with contextlib.suppress(Exception):
                fn(
                    timeout=float(join_timeout)
                ) if name == "join" and join_timeout is not None else fn()
            return
    with contextlib.suppress(Exception):
        obj() if callable(obj) else None


class _QueryEvent(Protocol):
    def query(self) -> bool: ...


@runtime_checkable
class _SyncEvent(Protocol):
    def synchronize(self) -> Any: ...


@runtime_checkable
class _WaitEvent(Protocol):
    def wait(self, timeout: float | None = None) -> Any: ...


@dataclass(slots=True)
class _PoolToken:
    i: int
    g: int


@dataclass(slots=True)
class _PoolEntry:
    page: Page
    busy: bool = False
    fence: object | None = None
    fence_evt: object | None = None
    gen: int = 0


class ProducerError:
    exc: BaseException
    tb: str


class Page:
    __slots__ = ("_buf", "_numel", "_dtype", "_pinned")

    def __init__(self, numel: int, dtype: torch.dtype, *args: Any, pin_memory: bool = True) -> None:
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
    Token = _PoolToken
    _Entry = _PoolEntry

    def __init__(self, capacity: int = 4, *args: Any, pin_memory: bool = True) -> None:
        self._cap = max(1, int(capacity))
        self._pin = bool(pin_memory)
        self._pages: list[Pool._Entry] = []
        self._rr = 0
        self._cv = threading.Condition()

    def _event_finished(self, evt: object | None) -> bool:
        if evt is None:
            return True
        with contextlib.suppress(Exception):
            if isinstance(evt, _QueryEvent):
                return bool(evt.query())
            if callable(is_set := getattr(evt, "is_set", None)):
                return bool(is_set())
        return False

    def _scavenge_lock(self) -> int:
        freed = 0
        for e in self._pages:
            if e.busy and e.fence is not None and self._event_finished(e.fence):
                e.busy = False
                e.fence = None
                freed += 1
        return freed

    @property
    def capacity(self) -> int:
        return int(self._cap)

    def get(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        *args: Any,
        return_handle: bool = False,
        block: bool = False,
        timeout: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Pool.Token | None]:
        shape_t = tuple(int(s) for s in shape)
        need = _prod_int(shape_t)
        deadline = (time.monotonic() + float(timeout)) if timeout is not None else None
        check_interval = 0.01

        while True:
            idx, need_new, grow = None, False, False
            with self._cv:
                self._scavenge_lock()
                n = len(self._pages)
                if n:
                    for k in range(n):
                        j = (self._rr + k) % n
                        e = self._pages[j]
                        if not e.busy:
                            e.busy, e.fence, idx, self._rr = True, None, j, (j + 1) % max(1, n)
                            need_new = (
                                (e.page.dtype != dtype)
                                or (e.page.numel < need)
                                or (e.page.pinned != self._pin)
                            )
                            break
                if idx is None:
                    if n < self._cap:
                        grow = True
                    elif block:
                        if deadline and (wait := deadline - time.monotonic()) <= 0:
                            break
                        self._cv.wait(
                            timeout=min(check_interval, wait) if deadline else check_interval
                        )
                        continue
                    else:
                        break

            if idx is not None:
                new_page = Page(numel=need, dtype=dtype, pin_memory=self._pin) if need_new else None
                with self._cv:
                    e = self._pages[idx]
                    if new_page:
                        e.page, e.gen = new_page, e.gen + 1
                    view, token = e.page.view(*shape_t), Pool.Token(int(idx), int(e.gen))
                return (view, token) if return_handle else view

            if grow:
                entry = Pool._Entry(
                    page=Page(numel=need, dtype=dtype, pin_memory=self._pin),
                    busy=True,
                    fence=None,
                    gen=0,
                )
                with self._cv:
                    if len(self._pages) < self._cap:
                        self._pages.append(entry)
                        self._rr = len(self._pages) % self._cap
                        view, token = entry.page.view(*shape_t), Pool.Token(len(self._pages) - 1, 0)
                        return (view, token) if return_handle else view
                continue
            break

        view = torch.empty(need, dtype=dtype, device="cpu", pin_memory=False).view(*shape_t)
        return (view, None) if return_handle else view

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
        with self._cv:
            if 0 <= i < len(self._pages) and self._pages[i].gen == g and self._pages[i].fence_evt:
                return self._pages[i].fence_evt

        try:
            ev_new = factory()
        except:
            return None
        if not ev_new:
            return None

        with self._cv:
            if 0 <= i < len(self._pages) and self._pages[i].gen == g:
                if not self._pages[i].fence_evt:
                    self._pages[i].fence_evt = ev_new
                return self._pages[i].fence_evt
        return ev_new

    def release_after(self, token: Pool.Token | None, wait_event: object | None) -> None:
        if token is None:
            return
        with self._cv:
            i = int(getattr(token, "i", -1))
            g = int(getattr(token, "g", -1))
            if 0 <= i < len(self._pages) and self._pages[i].gen == g:
                self._pages[i].busy, self._pages[i].fence = True, wait_event
                self._cv.notify()

    def release(self, token: Pool.Token | None) -> None:
        if token is None:
            return
        with self._cv:
            i = int(getattr(token, "i", -1))
            g = int(getattr(token, "g", -1))
            if 0 <= i < len(self._pages) and self._pages[i].gen == g:
                self._pages[i].busy, self._pages[i].fence = False, None
                self._cv.notify()

    def collect(self) -> None:
        with self._cv:
            if self._scavenge_lock():
                self._cv.notify_all()


class Cache:
    def __init__(self, root: str, max_queue: int = 8) -> None:
        self._root = os.fspath(root)
        os.makedirs(self._root, exist_ok=True)
        max_q = max(1, int(max_queue))
        self._sem = threading.Semaphore(max_q)
        self._q: "queue.SimpleQueue[tuple[Any, Any, Any, Any]]" = queue.SimpleQueue()
        self._bp_mode = str(_get_throttle_state() or "block")
        self._bp_timeout_s = float(_get_throttle_timeout() or 0.0)
        self._early_release = bool(_is_early_release_enabled())
        self._force_unpin = bool(_is_force_unpin_enabled())
        self._t = threading.Thread(target=self._run, daemon=True)
        self._err: BaseException | None = None
        self._err_event = threading.Event()
        self._closed = threading.Event()
        self._t.start()

    def _wait(self, evt: object | None) -> None:
        if evt is None:
            return
        with contextlib.suppress(Exception):
            if isinstance(evt, _SyncEvent):
                evt.synchronize()
                return
            if isinstance(evt, _WaitEvent):
                evt.wait()
                return

    @staticmethod
    def _is_cpu_pinned(t: torch.Tensor) -> bool:
        if not torch.is_tensor(t):
            return False
        if getattr(t, "device", None) is None or t.device.type != "cpu":
            return False
        with contextlib.suppress(Exception):
            return bool(t.is_pinned()) if hasattr(t, "is_pinned") else False
        return False

    def _init_tensor(
        self,
        tensor: torch.Tensor,
        *args: Any,
        release_cb: Callable[[], Any] | None = None,
        early_release: bool | None = None,
        force_unpin: bool | None = None,
    ) -> tuple[torch.Tensor, bool]:
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        buf = tensor.detach()

        if hasattr(buf, "to_local"):
            buf = buf.to_local()
        if buf.device.type != "cpu":
            buf = buf.to(device="cpu", non_blocking=False)

        early = early_release if early_release is not None else self._early_release
        unpin = force_unpin if force_unpin is not None else self._force_unpin
        released = False

        if Cache._is_cpu_pinned(buf) and (unpin or (early and callable(release_cb))):
            try:
                tmp = torch.empty_like(buf, device="cpu", pin_memory=False)
                tmp.copy_(buf, non_blocking=False)
                buf, released = tmp, bool(early and callable(release_cb))
                if released:
                    with contextlib.suppress(Exception):
                        release_cb()
            except Exception:
                released = False

        if not buf.is_contiguous():
            buf = buf.contiguous()
        return buf, released

    def _save_tensor(self, tensor: torch.Tensor, path: str) -> None:
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        buf = tensor.detach()

        if hasattr(buf, "to_local"):
            buf = buf.to_local()
        if buf.device.type != "cpu":
            buf = buf.to(device="cpu", non_blocking=False)
        if not buf.is_contiguous():
            buf = buf.contiguous()

        if str(path).endswith(".mmt"):
            from tensordict import MemoryMappedTensor

            parent = os.path.dirname(path) or "."
            os.makedirs(parent, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=os.path.basename(path) + ".", suffix=".tmp", dir=parent
            )
            os.close(fd)
            try:
                MemoryMappedTensor.from_tensor(buf, filename=tmp_name, existsok=True)
                os.replace(tmp_name, path)
            finally:
                with contextlib.suppress(Exception):
                    os.remove(tmp_name)

            from ..data import schemas

            schemas.write_json(
                schemas.get_meta_path(path),
                {"shape": list(map(int, buf.shape)), "dtype": str(buf.dtype).replace("torch.", "")},
                indent=None,
            )
            return

        if str(path).endswith((".pt", ".pth")):
            from ..data import schemas

            schemas.save_temp(path, buf)
        else:
            torch.save(buf, path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item[0] is None:
                break

            tensor, path, evt, rel = item
            rel_cb = rel if callable(rel) else None
            try:
                self._wait(evt)
                buf, released_early = self._init_tensor(tensor, release_cb=rel_cb)
                self._save_tensor(buf, path)
                if rel_cb is not None and not released_early:
                    with contextlib.suppress(Exception):
                        rel_cb()
            except Exception as e:
                self._err, _ = e, self._err_event.set()
                break
            finally:
                with contextlib.suppress(Exception):
                    self._sem.release() if self._sem else None

    def submit(
        self,
        tensor: torch.Tensor,
        path: Optional[str] = None,
        idx: Optional[int] = None,
        wait_event: Optional[object] = None,
        release_cb: Optional[object] = None,
    ) -> None:
        if self._err_event.is_set():
            raise RuntimeError(f"Async writer error: {self._err!r}")
        if self._closed.is_set():
            raise RuntimeError("Cache is closed")

        path = (
            os.path.join(self._root, f"chunk_{int(idx):06d}.pt")
            if path is None and idx is not None
            else os.fspath(path)
        )
        acquired = False
        if self._sem:
            mode, timeout = self._bp_mode, self._bp_timeout_s
            if mode in ("sync", "raise"):
                acquired = self._sem.acquire(timeout=timeout)
                if not acquired:
                    if mode == "raise":
                        raise RuntimeError("Cache queue is full")
                    self._wait(wait_event)
                    buf, released = self._init_tensor(
                        tensor, release_cb=release_cb, early_release=False
                    )
                    self._save_tensor(buf, path)
                    if callable(release_cb) and not released:
                        with contextlib.suppress(Exception):
                            release_cb()
                    return
            else:
                acquired = self._sem.acquire(timeout=timeout) if timeout > 0 else False
                while not acquired:
                    if self._err_event.is_set() or self._closed.is_set():
                        raise RuntimeError("Cache unavailable")
                    acquired = self._sem.acquire(timeout=0.1)
        try:
            self._q.put((tensor, path, wait_event, release_cb))
        except Exception:
            if acquired:
                with contextlib.suppress(Exception):
                    self._sem.release()
            raise

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._q.put((None, None, None, None))
        self._t.join()

    def had_error(self) -> bool:
        return bool(self._err_event.is_set())


class Buffer:
    def __init__(self, max_batches: int) -> None:
        self.max_batches = max(1, int(max_batches))
        self._buf: "collections.deque[Any]" = collections.deque()
        self._stop = threading.Event()
        self._cv = threading.Condition()
        self._warn_blocking = env_flag("STNET_BUFFER_WARN_BLOCKING", "STNET_DEBUG", default=False)

    def put(self, item: Any, *args: Any, timeout: float | None = None) -> bool:
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

    def __len__(self) -> int:
        return self.size()

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
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

    def block(self, *args: Any, timeout: float | None = None) -> bool:
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


_TLB_SINGLETON: Optional["Thread"] = None
_TLB_SINGLETON_LOCK = threading.Lock()


def get_affinity(io_workers: Optional[int] = None) -> "Thread":
    global _TLB_SINGLETON
    if _TLB_SINGLETON is None:
        with _TLB_SINGLETON_LOCK:
            if _TLB_SINGLETON is None:
                default_workers = (
                    int(io_workers)
                    if io_workers is not None
                    else max(1, int(CPU.count()) // 2)
                )
                _TLB_SINGLETON = Thread(io_workers=int(default_workers))
    elif io_workers is not None:
        _TLB_SINGLETON.tune(io_workers=int(io_workers))
    return _TLB_SINGLETON


class _AffinityCallable:
    __slots__ = (
        "_parent",
        "_fn",
        "_pin_thread",
        "_tls",
        "_lock",
        "_tune",
        "_sample_every",
        "_flush_every",
        "_perf_counter_ns",
        "_thread_time_ns",
    )

    def __init__(
        self,
        parent: "Thread",
        fn: Callable[[Any], Any],
        pin_thread: Callable[[], None],
        tls: Any,
        lock: Any,
        tune: Callable[..., None],
        sample_every: int,
        flush_every: int,
        perf_counter_ns: Callable[[], int],
        thread_time_ns: Optional[Callable[[], int]],
    ) -> None:
        self._parent = parent
        self._fn = fn
        self._pin_thread = pin_thread
        self._tls = tls
        self._lock = lock
        self._tune = tune
        self._sample_every = sample_every
        self._flush_every = flush_every
        self._perf_counter_ns = perf_counter_ns
        self._thread_time_ns = thread_time_ns

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not getattr(self._tls, "pinned", False):
            self._pin_thread()
        count = getattr(self._tls, "count", 0) + 1
        self._tls.count = count
        do_sample = count % self._sample_every == 0
        t0 = self._perf_counter_ns() if do_sample else 0
        c0 = self._thread_time_ns() if do_sample and callable(self._thread_time_ns) else 0
        out = self._fn(*args, **kwargs)
        if do_sample:
            t1 = self._perf_counter_ns()
            c1 = self._thread_time_ns() if callable(self._thread_time_ns) else None
            try:
                with self._lock:
                    self._parent._total_time += max(0, int(t1) - int(t0))
                    if c1 is not None:
                        self._parent._total_cpu += max(0, int(c1) - int(c0))
            except Exception:
                pass
        if count % self._flush_every == 0:
            try:
                self._tune(initial=False)
            except Exception:
                pass
        return out


class Thread:
    def __init__(
        self,
        io_workers: int,
        enabled: bool = True,
        allow_omp_bind: bool = True,
    ) -> None:
        self._allowed_cpus = sorted({int(x) for x in CPU.allowed()})
        self._proc_cycle = itertools.cycle(list(self._allowed_cpus))
        self._enabled = bool(enabled) and bool(self._allowed_cpus)
        self._nogil = bool(CPU.is_optimized_for_no_gil())
        self._io_workers = max(1, min(int(io_workers), max(1, len(self._allowed_cpus))))
        self._pin_attempts = 0
        self._pin_success = 0
        self._tls = threading.local()
        self._lock = Lock()
        self._total_time = 0
        self._total_cpu = 0
        self._omp_ok = bool(allow_omp_bind) and bool(self.spread_threads())

        self._flush_every = max(1, int(env_first_int(("STNET_TLB_FLUSH_EVERY",), 256)))
        self._sample_every = max(1, int(env_first_int(("STNET_TLB_SAMPLE_EVERY",), 8)))

    @staticmethod
    def _import_psutil() -> Optional[ModuleType]:
        spec = importlib.util.find_spec("psutil")
        if spec is None:
            return None
        return importlib.import_module("psutil")

    def _next_core(self) -> int:
        return int(next(self._proc_cycle))

    @staticmethod
    def _pin_thread_windows(core: int) -> bool:
        try:
            k32 = ctypes.WinDLL("kernel32")
            k32.GetCurrentThread.restype = ctypes.c_void_p
            handle = k32.GetCurrentThread()
            mask = ctypes.c_size_t(1 << int(core))
            k32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            k32.SetThreadAffinityMask.restype = ctypes.c_size_t
            prev = k32.SetThreadAffinityMask(handle, mask)
            return bool(prev)
        except Exception:
            return False

    @staticmethod
    def _pin_thread_linux(core: int) -> bool:
        try:
            tid = threading.get_native_id()
            if tid <= 0:
                return False
            os.sched_setaffinity(int(tid), {int(core)})
            return True
        except Exception:
            return False

    def tune(self, io_workers: Optional[int] = None) -> None:
        if io_workers is not None:
            self._io_workers = max(1, min(int(io_workers), max(1, len(self._allowed_cpus))))
        self.tune_threads(io_workers=self._io_workers, initial=True)

    def _retune_threads(self) -> None:
        if not self._enabled:
            return
        if getattr(self._tls, "in_retune", False):
            return
        self._tls.in_retune = True
        try:
            dev_type, nacc = WorkerPolicy._available_accelerator()
            is_accel = bool(nacc and int(nacc) > 0)
            cpus = max(1, len(self._allowed_cpus))
            cap_mult = _default_thread_limit(cpus, is_accel=is_accel, nogil=bool(self._nogil))
            local_world = _optimal_local_worlds(1)
            distribute_default = local_world > 1
            distribute = bool(
                env_first_int(("STNET_DISTRIBUTE_THREAD_CAP",), int(distribute_default))
            )
            thread_cap = _optimal_threads(
                ncpu=cpus,
                cap_mult=cap_mult,
                local_world=local_world,
                distribute=bool(distribute),
            )
            try:
                intra = int(torch.get_num_threads())
            except Exception:
                intra = int(cpus)
            try:
                inter = int(torch.get_num_interop_threads())
            except Exception:
                inter = 1
            workers = int(self._io_workers)
            total = int(intra) + int(inter) + int(workers)
            if total > int(thread_cap):
                new_intra = max(1, int(thread_cap) - int(inter) - int(workers))
                if int(new_intra) < int(intra):
                    optimize_threads(intra=int(new_intra))
                    intra = int(new_intra)
            total = int(intra) + int(inter) + int(workers)
            if total > int(thread_cap):
                new_inter = max(1, int(thread_cap) - int(workers) - int(intra))
                if int(new_inter) < int(inter):
                    optimize_threads(inter=int(new_inter))
        finally:
            self._tls.in_retune = False

    def total_procs(self) -> list[int]:
        return list(self._allowed_cpus)

    @staticmethod
    def spread_threads() -> bool:
        plat = sys.platform
        if plat.startswith("linux"):
            candidates = ["libgomp.so.1", "libgomp.so", "libiomp5.so", "libomp.so"]
        elif plat == "darwin":
            candidates = ["libomp.dylib", "libiomp5.dylib"]
        elif os.name == "nt":
            candidates = ["libiomp5md.dll", "vcomp140.dll"]
        else:
            candidates = []
        for name in candidates:
            try:
                lib = ctypes.CDLL(name)
            except OSError:
                continue
            try:
                fn = getattr(lib, "omp_set_proc_bind")
                fn.argtypes = [ctypes.c_int]
                fn.restype = None
                fn(4)
                return True
            except Exception:
                pass
            try:
                kmp = getattr(lib, "kmp_set_defaults")
                kmp.restype = None
                kmp(b"KMP_AFFINITY=granularity=fine,scatter")
                return True
            except Exception:
                pass
        return False

    def pin_thread(self) -> None:
        if not self._enabled:
            return
        attempts = getattr(self._tls, "attempts", 0)
        if getattr(self._tls, "pinned", False) or attempts >= 4:
            return
        self._tls.attempts = attempts + 1
        core = self._next_core()
        ok = False
        if os.name == "nt":
            ok = self._pin_thread_windows(core)
        else:
            plat = sys.platform
            if plat.startswith("linux"):
                ok = self._pin_thread_linux(core)
            elif plat == "darwin":
                with contextlib.suppress(Exception):
                    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
                    THREAD_AFFINITY_POLICY = 4

                    class thread_affinity_policy_data_t(ctypes.Structure):
                        _fields_ = [("affinity_tag", ctypes.c_int)]

                    policy = thread_affinity_policy_data_t(int(core) + 1)
                    lib.mach_thread_self.restype = ctypes.c_uint
                    lib.thread_policy_set.argtypes = [
                        ctypes.c_uint,
                        ctypes.c_int,
                        ctypes.c_void_p,
                        ctypes.c_uint,
                    ]
                    port = lib.mach_thread_self()
                    ok = (
                        lib.thread_policy_set(
                            port,
                            THREAD_AFFINITY_POLICY,
                            ctypes.byref(policy),
                            1,
                        )
                        == 0
                    )
        self._tls.pinned = bool(ok)
        self._pin_attempts += 1
        if ok:
            self._pin_success += 1
        if self._pin_attempts >= 16 and self._pin_success == 0 and not self._omp_ok:
            self._enabled = False

    def tune_threads(
        self,
        io_workers: Optional[int] = None,
        *_unused_args: Any,
        initial: bool = False,
        **_unused_kwargs: Any,
    ) -> None:
        if not self._enabled:
            return
        if initial:
            cpus = max(1, len(self._allowed_cpus))
            tuned_workers = max(
                1,
                min(
                    int(io_workers if io_workers is not None else self._io_workers),
                    cpus,
                ),
            )
            self._io_workers = tuned_workers
            dev_type, nacc = WorkerPolicy._available_accelerator()
            is_accel = bool(nacc and int(nacc) > 0)
            cap_mult = _default_thread_limit(cpus, is_accel=is_accel, nogil=bool(self._nogil))
            local_world = _optimal_local_worlds(1)
            distribute_default = local_world > 1
            distribute = bool(
                env_first_int(("STNET_DISTRIBUTE_THREAD_CAP",), int(distribute_default))
            )
            thread_cap = _optimal_threads(
                ncpu=cpus,
                cap_mult=cap_mult,
                local_world=local_world,
                distribute=bool(distribute),
            )
            try:
                intra_now = int(torch.get_num_threads())
            except Exception:
                intra_now = int(cpus)
            want_inter = max(1, min(tuned_workers // 2, 4))
            total = int(intra_now) + int(want_inter) + int(tuned_workers)
            if total > int(thread_cap):
                new_intra = max(1, int(thread_cap) - int(want_inter) - int(tuned_workers))
                if int(new_intra) != int(intra_now):
                    optimize_threads(intra=int(new_intra))
                    intra_now = int(new_intra)
                total = int(intra_now) + int(want_inter) + int(tuned_workers)
                if total > int(thread_cap):
                    want_inter = max(1, int(thread_cap) - int(tuned_workers) - int(intra_now))
            optimize_threads(inter=int(want_inter))
            return
        self._retune_threads()

    def new_thread(self, fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        if not self._enabled:
            return fn
        sample_every = int(self._sample_every) if int(self._sample_every) > 0 else 1
        flush_every = int(self._flush_every) if int(self._flush_every) > 0 else 1
        return _AffinityCallable(
            parent=self,
            fn=fn,
            pin_thread=self.pin_thread,
            tls=self._tls,
            lock=self._lock,
            tune=self.tune_threads,
            sample_every=sample_every,
            flush_every=flush_every,
            perf_counter_ns=time.perf_counter_ns,
            thread_time_ns=getattr(time, "thread_time_ns", None),
        )

    def optimize_procs(self, io_workers: int) -> int:
        if not self._enabled:
            return int(io_workers)
        cpus = max(1, len(self._allowed_cpus))
        tuned = max(1, min(int(io_workers), cpus))
        self._io_workers = tuned
        return tuned
