# -*- coding: utf-8 -*-
from __future__ import annotations

import collections.abc as _abc
import contextlib
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Protocol, runtime_checkable

import numpy as np
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
        import json
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

            MemoryMappedTensor.from_tensor(buf, filename=path)
            meta = {
                "shape": list(buf.shape),
                "dtype": str(buf.dtype).replace("torch.", ""),
            }
            tmp_json = path + ".json.tmp"
            final_json = path + ".json"
            with open(tmp_json, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            _os.replace(tmp_json, final_json)
            return

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


# -----------------------------------------------------------------------------
# LazyTensor: streaming + memmap utilities (shared across run/runtime/nodes)
# -----------------------------------------------------------------------------

class LazyTensor:
    """Centralized memmap/streaming/chunking utilities."""

    class KeyIndexMappingView(_abc.Mapping):
        """A mapping-view over `data` that iterates in the provided `keys` order."""

        __slots__ = ("_data", "_keys")

        def __init__(self, data: Mapping[Any, Any], keys: Sequence[Any]) -> None:
            self._data = data
            self._keys = keys

        def __len__(self) -> int:
            return int(len(self._keys))

        def __iter__(self):
            return iter(self._keys)

        def __getitem__(self, k: Any) -> Any:
            return self._data[k]

    @staticmethod
    def key_index_mapping_getters(
        data: Mapping[Any, Any],
        *,
        keys: Optional[Sequence[Any]] = None,
    ) -> Tuple[int, Any]:
        """Build (count, get_batch) helpers for key-index mappings.

        IMPORTANT: This intentionally does *not* provide random access or writer-side shuffling.
        Shuffling should be handled by the sampler at read time.

        The returned get_batch() is optimized for sequential access patterns:
        - expects monotonically increasing (s, e) windows within a pass
        - supports automatic reset when s==0 (used by two-pass writers)
        """

        if keys is None:
            count = int(len(data))

            def _iter_keys() -> Any:
                return iter(data.keys())

        else:
            count = int(len(keys))

            def _iter_keys() -> Any:
                return iter(keys)

        if count <= 0:
            raise ValueError("Empty mapping: no keys")

        it = _iter_keys()
        pos = 0

        def get_batch(s: int, e: int):
            nonlocal it, pos
            s_i = int(s)
            e_i = int(e)
            if s_i < 0 or e_i < s_i:
                raise ValueError(f"invalid batch slice: s={s_i}, e={e_i}")

            # Two-pass writer resets to s==0 on the second pass.
            if s_i == 0 and pos != 0:
                it = _iter_keys()
                pos = 0

            # Sequential-only contract: avoids O(N^2) islice-based skipping.
            if s_i != pos:
                raise RuntimeError(
                    "key_index_mapping_getters: non-sequential access requested "
                    f"(expected s={pos}, got s={s_i}). "
                    "Disable writer-side shuffle; let the sampler handle shuffling."
                )

            need = e_i - s_i
            if need <= 0:
                return LazyTensor.KeyIndexMappingView(data, ())

            batch_keys: list[Any] = []
            for _ in range(int(need)):
                try:
                    k = next(it)
                except StopIteration:
                    break
                batch_keys.append(k)

            pos += int(len(batch_keys))
            return LazyTensor.KeyIndexMappingView(data, batch_keys)

        return count, get_batch

    @staticmethod
    def is_feature_label_batch_mapping(obj: Any) -> bool:
        """True if the object looks like a {features, labels} style batch mapping.

        The project historically accepted a mix of key names (e.g. X/Y, features/labels).
        We now treat feature/label column names case-insensitively and recognize:

        - features: x, feature, features, input, inputs, in
        - labels:   y, label, labels, output, outputs, out

        If the mapping uses non-string keys (e.g. {tuple_feature: label}), this returns False
        so it can fall back to the compact-dataset heuristic.
        """
        if not isinstance(obj, Mapping) or not obj:
            return False

        # Centralized alias sets live in pipeline.py.
        from .pipeline import _FEATURE_KEY_ALIASES, _LABEL_KEY_ALIASES

        for k in obj.keys():
            if not isinstance(k, str):
                continue
            ck = k.casefold()
            if ck in _FEATURE_KEY_ALIASES or ck in _LABEL_KEY_ALIASES:
                return True
        return False

    @staticmethod
    def _resolve_memmap_store_float(*, negotiable: bool) -> torch.dtype:
        from .datatype import env_str

        req = str(env_str("STNET_MEMMAP_FLOAT_DTYPE") or "").strip()
        if req.startswith("torch."):
            req = req.split(".", 1)[1]
        req_dtype = getattr(torch, req, None) if req else None
        if not isinstance(req_dtype, torch.dtype):
            req_dtype = torch.float32
        try:
            if not torch.is_floating_point(torch.empty((), dtype=req_dtype)):
                req_dtype = torch.float32
        except Exception:
            req_dtype = torch.float32
        return torch.float32 if (bool(negotiable) and req_dtype != torch.float64) else torch.float64

    @staticmethod
    def _to_cpu_contig(t: torch.Tensor) -> torch.Tensor:
        t = t.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        if not t.is_contiguous():
            t = t.contiguous()
        return t

    @staticmethod
    def _flat2d_cpu_contig(t: torch.Tensor, n: int) -> torch.Tensor:
        t_cpu = LazyTensor._to_cpu_contig(t)
        if t_cpu.ndim == 0:
            t_cpu = t_cpu.reshape(1)
        return t_cpu.reshape(int(n), -1)

    @staticmethod
    def _batch_n(x: torch.Tensor) -> int:
        xd = int(getattr(x, "ndim", 0) or 0)
        return int(x.shape[0]) if xd > 0 else 1

    @staticmethod
    def _idx_to_cpu_int64(idx: Any) -> torch.Tensor:
        if not isinstance(idx, torch.Tensor):
            idx = torch.as_tensor(idx)
        if idx.device.type != "cpu":
            idx = idx.detach().cpu()
        if idx.dtype not in (torch.int64, torch.int32):
            idx = idx.to(dtype=torch.int64, copy=False)
        idx = idx.reshape(-1)
        return idx.to(dtype=torch.int64, copy=False)

    @staticmethod
    def mmt_meta_path(mmt_path: str) -> str:
        """Sidecar metadata path for a MemoryMappedTensor file."""
        return str(mmt_path) + ".meta.json"

    @staticmethod
    def atomic_write_json(path: str, payload: Any, *, indent: int | None = 2) -> None:
        """Atomically write JSON (best-effort).

        Used for small sidecar metadata files that must not be left half-written
        under multi-process or abrupt-interrupt scenarios.
        """

        import json as _json

        p = os.fspath(path)
        parent = os.path.dirname(p) or "."
        os.makedirs(parent, exist_ok=True)

        fd, tmp_name = tempfile.mkstemp(prefix=os.path.basename(p) + ".", suffix=".tmp", dir=parent)
        os.close(fd)
        try:
            with open(tmp_name, "w", encoding="utf-8") as f:
                if indent is None:
                    _json.dump(payload, f)
                else:
                    _json.dump(payload, f, indent=int(indent))
            os.replace(tmp_name, p)
        finally:
            with contextlib.suppress(Exception):
                os.remove(tmp_name)

    # Backward-compatible alias (internal call sites)
    @staticmethod
    def _atomic_write_json(path: str, payload: Any, *, indent: int = 2) -> None:
        LazyTensor.atomic_write_json(path, payload, indent=int(indent))

    @staticmethod
    def atomic_torch_save(path: str, payload: Any, **opts: Any) -> None:
        """Atomically write a torch checkpoint file.

        Writes to a temporary file in the same directory and then replaces
        it into place. This avoids corrupting checkpoints if interrupted
        mid-write.
        """

        p = os.fspath(path)
        parent = os.path.dirname(p) or "."
        os.makedirs(parent, exist_ok=True)

        fd, tmp_name = tempfile.mkstemp(prefix=os.path.basename(p) + ".", suffix=".tmp", dir=parent)
        os.close(fd)
        try:
            torch.save(payload, tmp_name, **opts)
            os.replace(tmp_name, p)
        finally:
            with contextlib.suppress(Exception):
                os.remove(tmp_name)

    # Backward-compatible alias (internal call sites)
    @staticmethod
    def _atomic_torch_save(path: str, payload: Any, **opts: Any) -> None:
        LazyTensor.atomic_torch_save(path, payload, **opts)

    @staticmethod
    def write_memmap_streaming_two_pass(
        *,
        ds: Any,
        out_dir: str,
        count: int,
        get_batch: Any,
        val_frac: float,
        seed_value: Any,
        underflow_action: str,
        shuffle: bool = False,
        get_by_indices: Any = None,
        default_label_shape: Any = None,
        allow_missing_labels: bool = False,
        features_only: bool = False,
        chunk_size: int = 32,
    ) -> Tuple[int, Tuple[int, ...]]:
        """Write a memmap dataset in two passes.

        Pass 1: infer shapes and collect scale stats (for dtype negotiation).
        Pass 2: write contiguous memmaps (optionally shuffled).
        """
        from tensordict import MemoryMappedTensor
        from .pipeline import Dataset
        from .datatype import env_first_int

        out_dir = os.fspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        count_i = int(count)
        if count_i <= 0:
            raise ValueError("count must be > 0")

        env_chunk = env_first_int(("STNET_MEMMAP_CHUNK_SIZE", "STNET_MEMMAP_CHUNK"), None)
        if env_chunk is not None and int(env_chunk) > 0:
            chunk_size = int(env_chunk)

        req_chunk = int(chunk_size or 0)
        auto_chunk = req_chunk <= 0
        chunk_first = max(1, min(count_i, req_chunk if req_chunk > 0 else min(count_i, 256)))

        allow_missing = bool(allow_missing_labels) or bool(features_only)
        default_lshape = tuple(default_label_shape) if default_label_shape is not None else (1,)

        stats: Dict[str, Any] = {
            "has_scale": False,
            "has_nonfinite": False,
            "scale_max_abs": None,
            "scale_min_value": None,
            "scale_max_value": None,
            "scale_min_positive": None,
            "scale_is_integral": None,
        }
        in_dim: Optional[int] = None
        label_shape: Optional[Tuple[int, ...]] = None

        # --- Pass 1: infer shapes + stats ---
        for s in range(0, count_i, int(chunk_first)):
            e = min(count_i, s + int(chunk_first))
            batch = get_batch(int(s), int(e))
            fx, lb, _, _ = ds.preprocess(batch)
            n = LazyTensor._batch_n(fx)
            if n <= 0:
                continue
            expected = int(e) - int(s)
            if n != expected:
                raise RuntimeError(
                    f"Pass1 batch size mismatch for out_dir={out_dir!r}: expected {expected}, got {n} (s={s}, e={e})."
                )

            fx_flat = LazyTensor._flat2d_cpu_contig(fx, n)
            cur_in_dim = int(fx_flat.shape[1])
            if in_dim is None:
                in_dim = cur_in_dim
            elif cur_in_dim != int(in_dim):
                raise RuntimeError(f"feature dim mismatch: expected {in_dim}, got {cur_in_dim}")

            if lb is None:
                if not allow_missing:
                    raise RuntimeError("memmap writer requires labels (got None)")
                cur_label_shape = tuple(default_lshape)
                lb_flat = None
            else:
                cur_label_shape = tuple(lb.shape[1:])
                lb_flat = LazyTensor._flat2d_cpu_contig(lb, n)

            if label_shape is None:
                label_shape = cur_label_shape
            elif tuple(label_shape) != tuple(cur_label_shape):
                raise RuntimeError(f"label shape mismatch: expected {label_shape}, got {cur_label_shape}")

            f_stats = Dataset.tensor_scale_stats(fx_flat)
            if bool(features_only):
                stats = Dataset.merge_scale_stats(stats, f_stats)
            else:
                if lb_flat is None:
                    l_stats = {
                        "has_scale": True,
                        "has_nonfinite": False,
                        "scale_max_abs": 0.0,
                        "scale_min_value": 0.0,
                        "scale_max_value": 0.0,
                        "scale_min_positive": None,
                        "scale_is_integral": None,
                    }
                else:
                    l_stats = Dataset.tensor_scale_stats(lb_flat)
                stats = Dataset.merge_scale_stats(stats, Dataset.merge_scale_stats(f_stats, l_stats))

        if in_dim is None or label_shape is None:
            raise RuntimeError("Failed to infer in_dim/label_shape from data")

        negotiable = Dataset.is_fp32_castable(
            stats,
            underflow_action=underflow_action,
            safety_margin=1.0,
        )
        store_float = LazyTensor._resolve_memmap_store_float(negotiable=bool(negotiable))

        # Auto-tune chunk size for pass 2 (bound memory).
        if auto_chunk:
            elem_size = int(torch.empty((), dtype=store_float).element_size())
            label_numel = 0 if bool(features_only) else int(np.prod(label_shape))
            row_bytes = max(1, (int(in_dim) + int(label_numel)) * int(elem_size))

            target_bytes = env_first_int(("STNET_MEMMAP_CHUNK_BYTES",), None)
            if target_bytes is None:
                target_mb = env_first_int(("STNET_MEMMAP_CHUNK_MB",), 64)
                target_bytes = int(target_mb) * 1024 * 1024

            try:
                from ..backend.system import Memory

                avail = int(Memory.available() or 0)
                if avail > 0:
                    target_bytes = int(min(int(target_bytes), max(8 * 1024 * 1024, avail // 16)))
            except Exception:
                pass

            chunk_second = int(max(1, min(count_i, max(32, int(target_bytes) // int(row_bytes)))))
        else:
            chunk_second = int(max(1, min(count_i, req_chunk)))

        val_count = max(0, min(count_i, int(round(count_i * float(val_frac)))))
        train_count = max(0, count_i - val_count)
        train_start, train_end = 0, int(train_count)
        val_start, val_end = int(train_end), int(train_end) + int(val_count)

        features_path = os.path.join(out_dir, "features.mmt")
        labels_path = os.path.join(out_dir, "labels.mmt")

        features_mmt = MemoryMappedTensor.empty(
            (count_i, int(in_dim)),
            dtype=store_float,
            filename=features_path,
            existsok=True,
        )

        write_labels = not bool(features_only)
        labels_mmt = None
        if write_labels:
            labels_mmt = MemoryMappedTensor.empty(
                (count_i, *tuple(label_shape)),
                dtype=store_float,
                filename=labels_path,
                existsok=True,
            )

        # Shuffle indexer
        shuffle_indexer = None
        shuffle_impl = "none"
        order: Optional[torch.Tensor] = None
        shuffle_seed: Optional[int] = None

        if bool(shuffle):
            if get_by_indices is None:
                raise ValueError("shuffle=True requires get_by_indices")

            max_elems = env_first_int(
                ("STNET_MEMMAP_RANDPERM_MAX_ELEMS", "STNET_MEMMAP_SHUFFLE_MAX_ELEMS"),
                5_000_000,
            )
            use_full = (max_elems is not None) and (count_i <= int(max_elems))
            seed_i = None if seed_value is None else (int(seed_value) & 0x7FFFFFFFFFFFFFFF)

            if use_full:
                g = None
                if seed_i is not None:
                    g = torch.Generator(device="cpu")
                    g.manual_seed(seed_i)
                shuffle_seed = seed_i
                order = torch.randperm(count_i, generator=g, dtype=torch.int64)

                def _idx(s: int, e: int) -> torch.Tensor:
                    assert order is not None
                    return order[int(s) : int(e)]

                shuffle_indexer = _idx
                shuffle_impl = "randperm"
            else:
                if seed_i is None:
                    seed_i = int(torch.randint(0, 2**63 - 1, (1,), dtype=torch.int64).item())
                shuffle_seed = seed_i

                # Feistel PRP on 2^k domain + cycle-walking into [0, count).
                k = max(1, int((count_i - 1)).bit_length())
                if (k % 2) == 1:
                    k += 1
                half = k // 2
                mask = (1 << half) - 1
                domain_mask = (1 << k) - 1 if k < 64 else 0xFFFFFFFFFFFFFFFF

                seed_u = torch.tensor(seed_i & 0xFFFFFFFFFFFFFFFF, dtype=torch.uint64)
                mask_u = torch.tensor(mask, dtype=torch.uint64)
                domain_u = torch.tensor(domain_mask, dtype=torch.uint64)
                count_u = torch.tensor(count_i, dtype=torch.uint64)

                k0 = seed_u ^ torch.tensor(0x9E3779B97F4A7C15, dtype=torch.uint64)
                k1 = seed_u ^ torch.tensor(0xBF58476D1CE4E5B9, dtype=torch.uint64)
                k2 = seed_u ^ torch.tensor(0x94D049BB133111EB, dtype=torch.uint64)
                k3 = seed_u ^ torch.tensor(0xD6E8FEB86659FD93, dtype=torch.uint64)
                round_keys = (k0, k1, k2, k3)

                _c_mul1 = torch.tensor(0x9E3779B97F4A7C15, dtype=torch.uint64)
                _c_mul2 = torch.tensor(0xC2B2AE3D27D4EB4F, dtype=torch.uint64)

                def _round_fn(r: torch.Tensor, rk: torch.Tensor) -> torch.Tensor:
                    x = (r ^ rk) & mask_u
                    x = (x * _c_mul1) & domain_u
                    x = (x ^ (x >> 33)) & domain_u
                    x = (x * _c_mul2) & domain_u
                    x = (x ^ (x >> 29)) & domain_u
                    return x & mask_u

                def _feistel(x: torch.Tensor) -> torch.Tensor:
                    x = x & domain_u
                    l = (x >> half) & mask_u
                    r = x & mask_u
                    for rk in round_keys:
                        f = _round_fn(r, rk)
                        l, r = r, (l ^ f) & mask_u
                    return (((l << half) | r) & domain_u)

                # Deterministic fallback permutation (affine mod count) in case
                # cycle-walking doesn't converge quickly for some domain/seed.
                def _gcd(a: int, b: int) -> int:
                    while b:
                        a, b = b, a % b
                    return abs(int(a))

                a0 = (seed_i | 1) % count_i
                if a0 == 0:
                    a0 = 1
                while _gcd(a0, count_i) != 1:
                    a0 = (a0 + 2) % count_i
                    if a0 == 0:
                        a0 = 1
                b0 = seed_i % count_i

                def _affine(pos: torch.Tensor) -> torch.Tensor:
                    p = pos.to(dtype=torch.int64)
                    return ((p * int(a0) + int(b0)) % int(count_i)).to(dtype=torch.int64)

                def _permute(pos: torch.Tensor) -> torch.Tensor:
                    x = pos.to(dtype=torch.uint64)
                    y = _feistel(x)
                    bad = y >= count_u

                    max_iter = 64
                    it = 0
                    while bool(bad.any()):
                        it += 1
                        if it > max_iter:
                            return _affine(pos)
                        y_bad = _feistel(y[bad])
                        y[bad] = y_bad
                        bad = y >= count_u
                    return y.to(dtype=torch.int64)

                def _idx(s: int, e: int) -> torch.Tensor:
                    pos = torch.arange(int(s), int(e), device="cpu", dtype=torch.int64)
                    return _permute(pos)

                shuffle_indexer = _idx
                shuffle_impl = "prp"

        # Optional scaler stats (train split only).
        compute_scaler_stats = bool(write_labels) and (not bool(allow_missing_labels))
        x_sum: Optional[torch.Tensor] = None
        x_sum_sq: Optional[torch.Tensor] = None
        y_sum: Optional[torch.Tensor] = None
        y_sum_sq: Optional[torch.Tensor] = None

        if compute_scaler_stats and int(train_end) > 0:
            x_sum = torch.zeros((int(in_dim),), dtype=torch.float64, device=torch.device("cpu"))
            x_sum_sq = torch.zeros((int(in_dim),), dtype=torch.float64, device=torch.device("cpu"))
            out_dim = int(np.prod(label_shape))
            y_sum = torch.zeros((int(out_dim),), dtype=torch.float64, device=torch.device("cpu"))
            y_sum_sq = torch.zeros((int(out_dim),), dtype=torch.float64, device=torch.device("cpu"))

        written = 0
        for s in range(0, count_i, int(chunk_second)):
            e = min(count_i, s + int(chunk_second))
            if shuffle_indexer is None:
                batch = get_batch(int(s), int(e))
            else:
                idx = shuffle_indexer(int(s), int(e))
                batch = get_by_indices(idx)

            fx, lb, _, _ = ds.preprocess(batch)
            n = LazyTensor._batch_n(fx)
            if n <= 0:
                continue
            expected = int(e) - int(s)
            if n != expected:
                raise RuntimeError(
                    f"Pass2 batch size mismatch for out_dir={out_dir!r}: expected {expected}, got {n} (s={s}, e={e})."
                )

            fx_flat = LazyTensor._flat2d_cpu_contig(fx, n)
            if int(fx_flat.shape[1]) != int(in_dim):
                raise RuntimeError(
                    f"feature dim mismatch: expected {in_dim}, got {int(fx_flat.shape[1])}"
                )

            fx_out = fx_flat if fx_flat.dtype == store_float else fx_flat.to(dtype=store_float)

            if x_sum is not None and x_sum_sq is not None:
                end_pos = int(s) + int(n)
                overlap = max(0, min(end_pos, int(train_end)) - int(s))
                if overlap > 0:
                    fx_slice = fx_out[:overlap]
                    fx64 = fx_slice if fx_slice.dtype == torch.float64 else fx_slice.to(dtype=torch.float64)
                    x_sum += fx64.sum(dim=0)
                    x_sum_sq += fx64.square().sum(dim=0)

            features_mmt[int(s) : int(s) + int(n)].copy_(fx_out)

            if write_labels:
                assert labels_mmt is not None
                if lb is None:
                    if not allow_missing:
                        raise RuntimeError("memmap writer requires labels (got None)")
                    # Avoid large pre-allocated zero buffers: write zeros directly.
                    labels_mmt[int(s) : int(s) + int(n)].zero_()
                else:
                    if tuple(lb.shape[1:]) != tuple(label_shape):
                        raise RuntimeError(
                            f"label shape mismatch: expected {label_shape}, got {tuple(lb.shape[1:])}"
                        )
                    lb_cpu = LazyTensor._to_cpu_contig(lb)
                    lb_out = lb_cpu if lb_cpu.dtype == store_float else lb_cpu.to(dtype=store_float)

                    labels_mmt[int(s) : int(s) + int(n)].copy_(lb_out)

                    if y_sum is not None and y_sum_sq is not None:
                        end_pos = int(s) + int(n)
                        overlap = max(0, min(end_pos, int(train_end)) - int(s))
                        if overlap > 0:
                            lb_slice = lb_out[:overlap].reshape(int(overlap), -1)
                            lb64 = lb_slice if lb_slice.dtype == torch.float64 else lb_slice.to(dtype=torch.float64)
                            y_sum += lb64.sum(dim=0)
                            y_sum_sq += lb64.square().sum(dim=0)

            written += int(n)

        if int(written) != int(count_i):
            raise RuntimeError(f"memmap written={written}, expected={count_i}")

        scaler_stats_path: Optional[str] = None
        if (
            compute_scaler_stats
            and int(train_end) > 0
            and x_sum is not None
            and x_sum_sq is not None
            and y_sum is not None
            and y_sum_sq is not None
        ):
            payload = {
                "version": 1,
                "train_count": int(train_end),
                "x_sum": x_sum,
                "x_sum_sq": x_sum_sq,
                "y_sum": y_sum,
                "y_sum_sq": y_sum_sq,
            }
            scaler_stats_path = "scaler_stats.pt"
            try:
                LazyTensor._atomic_torch_save(os.path.join(out_dir, scaler_stats_path), payload)
            except Exception:
                scaler_stats_path = None

        meta_json: Dict[str, Any] = {
            "N": int(count_i),
            "feature_dim": int(in_dim),
            "features_path": "features.mmt",
            "labels_path": ("labels.mmt" if write_labels else None),
            "label_shape": list(label_shape),
            "features_dtype": str(store_float).replace("torch.", ""),
            "labels_dtype": (str(store_float).replace("torch.", "") if write_labels else None),
            "fractions": [float(1.0 - float(val_frac)), float(val_frac)],
            "shuffled": bool(shuffle),
            "shuffle_seed": int(shuffle_seed) if shuffle_seed is not None else None,
            "shuffle_mode": "physical" if bool(shuffle) else "none",
            "shuffle_impl": shuffle_impl,
            "train_start": int(train_start),
            "train_end": int(train_end),
            "val_start": int(val_start),
            "val_end": int(val_end),
            "scaler_stats_path": scaler_stats_path,
            "has_scale": bool(stats.get("has_scale")),
            "has_nonfinite": bool(stats.get("has_nonfinite")),
            "scale_max_abs": stats.get("scale_max_abs"),
            "scale_min_value": stats.get("scale_min_value"),
            "scale_max_value": stats.get("scale_max_value"),
            "scale_min_positive": stats.get("scale_min_positive"),
            "scale_is_integral": stats.get("scale_is_integral"),
            "is_negotiable": bool(negotiable),
            "underflow_action": str(underflow_action),
            "features_only": bool(features_only),
        }

        LazyTensor._atomic_write_json(os.path.join(out_dir, "meta.json"), meta_json, indent=2)
        return int(in_dim), tuple(label_shape)

    @staticmethod
    def preload_memmap(
        data: Mapping[str, Any],
        *,
        memmap_dir: str,
        val_frac: float = 0.0,
        shuffle: bool = False,
        seed: Optional[int] = None,
        underflow_action: Optional[str] = None,
        chunk_size: int = 4096,
        allow_missing_labels: bool = False,
        features_only: bool = False,
        default_label_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Materialize an in-memory dataset into a memmap directory."""
        from .pipeline import Dataset, default_underflow_action, normalize_underflow_action

        if not isinstance(data, Mapping):
            raise TypeError("preload_memmap expects a Mapping with at least 'features'")
        if "features" not in data:
            raise ValueError("preload_memmap expects 'features'")

        raw_X = data["features"]
        raw_Y = data.get("labels")

        def _len0(obj: Any) -> int:
            if isinstance(obj, torch.Tensor):
                return int(obj.shape[0]) if getattr(obj, "ndim", 0) > 0 else 1
            try:
                return int(len(obj))
            except Exception:
                t = torch.as_tensor(obj)
                return int(t.shape[0]) if getattr(t, "ndim", 0) > 0 else 1

        count = _len0(raw_X)
        if count <= 0:
            raise ValueError("cannot create memmap with zero samples")
        if not bool(features_only):
            if raw_Y is None:
                if not bool(allow_missing_labels):
                    raise ValueError("preload_memmap expects 'labels' unless allow_missing_labels=True")
            else:
                if _len0(raw_Y) != int(count):
                    raise ValueError("features and labels must have the same length")

        ua = normalize_underflow_action(underflow_action, default=default_underflow_action())

        ds = Dataset.for_device("cpu", feature_dtype=torch.float64, label_float_dtype=torch.float64)
        ds.underflow_action = ua

        def _slice_any(obj: Any, s: int, e: int, *, name: str) -> Any:
            if obj is None:
                return None
            if torch.is_tensor(obj):
                return obj[int(s) : int(e)]
            if isinstance(obj, np.ndarray):
                return obj[int(s) : int(e)]
            try:
                return obj[int(s) : int(e)]
            except Exception:
                pass
            try:
                return [obj[i] for i in range(int(s), int(e))]
            except Exception as ex:
                raise TypeError(f"Object {name} does not support slicing [{s}:{e}]") from ex

        def _gather_any(obj: Any, idx: torch.Tensor, *, name: str) -> Any:
            if obj is None:
                return None
            idx_cpu = LazyTensor._idx_to_cpu_int64(idx)
            if torch.is_tensor(obj):
                return obj[idx_cpu]
            if isinstance(obj, np.ndarray):
                return obj[idx_cpu.numpy()]
            try:
                return obj[idx_cpu.numpy()]
            except Exception:
                pass
            try:
                return [obj[int(i)] for i in idx_cpu.tolist()]
            except Exception as ex:
                raise TypeError(f"Object {name} does not support gather by indices") from ex

        def get_batch(s: int, e: int) -> Mapping[str, Any]:
            out: Dict[str, Any] = {"features": _slice_any(raw_X, s, e, name="features")}
            if raw_Y is not None and not bool(features_only):
                out["labels"] = _slice_any(raw_Y, s, e, name="labels")
            return out

        def get_by_indices(idx: torch.Tensor) -> Mapping[str, Any]:
            out: Dict[str, Any] = {"features": _gather_any(raw_X, idx, name="features")}
            if raw_Y is not None and not bool(features_only):
                out["labels"] = _gather_any(raw_Y, idx, name="labels")
            return out

        LazyTensor.write_memmap_streaming_two_pass(
            ds=ds,
            out_dir=os.fspath(memmap_dir),
            count=int(count),
            get_batch=get_batch,
            get_by_indices=get_by_indices,
            val_frac=float(val_frac),
            seed_value=int(seed) if seed is not None else None,
            underflow_action=str(ua),
            shuffle=bool(shuffle),
            allow_missing_labels=bool(allow_missing_labels),
            features_only=bool(features_only),
            default_label_shape=tuple(default_label_shape) if default_label_shape is not None else None,
            chunk_size=int(chunk_size),
        )
        return None

    @staticmethod
    def iter_source_paths(obj: Any):
        if obj is None:
            return
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, dict):
            if obj.get("kind") == "memmap" and isinstance(obj.get("path"), str):
                yield obj["path"]
            else:
                for v in obj.values():
                    yield from LazyTensor.iter_source_paths(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from LazyTensor.iter_source_paths(v)

    @staticmethod
    def from_meta(memmap_dir: str) -> Dict[str, Any]:
        import json as _json

        meta_path = os.path.join(os.fspath(memmap_dir), "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            raw = _json.load(f)
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def merge_meta_dicts(metas: list[dict]) -> dict:
        if not metas:
            return {}
        base = dict(metas[0])

        def _strictest_underflow(a: Optional[str], b: Optional[str]) -> Optional[str]:
            order = {"allow": 0, "warn": 1, "forbid": 2}
            if a is None:
                return b
            if b is None:
                return a
            return a if order.get(a, 1) >= order.get(b, 1) else b

        feature_dim = base.get("feature_dim")
        label_shape = base.get("label_shape")

        def _has_scale(m: dict) -> bool:
            return bool(m.get("has_scale", False)) or any(
                m.get(k) is not None
                for k in (
                    "scale_max_abs",
                    "scale_min_value",
                    "scale_max_value",
                    "scale_min_positive",
                )
            )

        has_scale = _has_scale(base)
        has_nonfinite = bool(base.get("has_nonfinite", False))
        max_abs = base.get("scale_max_abs")
        min_val = base.get("scale_min_value")
        max_val = base.get("scale_max_value")
        min_pos = base.get("scale_min_positive")
        is_integral = base.get("scale_is_integral")
        is_negotiable = base.get("is_negotiable")
        underflow_action = base.get("underflow_action")

        for m in metas[1:]:
            if feature_dim is not None and m.get("feature_dim") is not None:
                if int(m.get("feature_dim")) != int(feature_dim):
                    raise ValueError(
                        f"feature_dim mismatch across sources: {feature_dim} vs {m.get('feature_dim')}"
                    )
            if label_shape is not None and m.get("label_shape") is not None:
                if tuple(m.get("label_shape")) != tuple(label_shape):
                    raise ValueError(
                        f"label_shape mismatch across sources: {label_shape} vs {m.get('label_shape')}"
                    )

            has_scale = has_scale or _has_scale(m)
            has_nonfinite = has_nonfinite or bool(m.get("has_nonfinite", False))

            a = m.get("scale_max_abs")
            if a is not None:
                max_abs = a if max_abs is None else max(float(max_abs), float(a))

            mn = m.get("scale_min_value")
            if mn is not None:
                try:
                    min_val = mn if min_val is None else (mn if mn <= min_val else min_val)
                except Exception:
                    min_val = mn if min_val is None else min(float(min_val), float(mn))

            mx = m.get("scale_max_value")
            if mx is not None:
                try:
                    max_val = mx if max_val is None else (mx if mx >= max_val else max_val)
                except Exception:
                    max_val = mx if max_val is None else max(float(max_val), float(mx))

            p = m.get("scale_min_positive")
            if p is not None:
                min_pos = p if min_pos is None else min(float(min_pos), float(p))

            i = m.get("scale_is_integral")
            if i is not None:
                is_integral = bool(i) if is_integral is None else bool(is_integral) and bool(i)

            n = m.get("is_negotiable")
            if n is not None:
                is_negotiable = bool(n) if is_negotiable is None else bool(is_negotiable) and bool(n)

            underflow_action = _strictest_underflow(
                str(underflow_action) if underflow_action is not None else None,
                str(m.get("underflow_action")) if m.get("underflow_action") is not None else None,
            )

        base["has_scale"] = bool(has_scale)
        base["has_nonfinite"] = bool(has_nonfinite)
        base["scale_max_abs"] = max_abs
        base["scale_min_value"] = min_val
        base["scale_max_value"] = max_val
        base["scale_min_positive"] = min_pos
        base["scale_is_integral"] = is_integral
        base["is_negotiable"] = is_negotiable
        base["underflow_action"] = underflow_action
        return base

    @staticmethod
    def merge_meta_infos(sources: Any) -> Dict[str, Any]:
        sources = LazyTensor.expand_sources(sources)
        metas: list[dict] = []
        for path in LazyTensor.iter_source_paths(sources):
            try:
                metas.append(LazyTensor.from_meta(path))
            except Exception:
                continue
        if not metas:
            return {}
        return dict(LazyTensor.merge_meta_dicts(metas))

    @staticmethod
    def load_scaler_stats(sources: Any) -> Optional[Dict[str, Any]]:
        expanded = LazyTensor.expand_sources(sources)
        total = 0
        x_sum: Optional[torch.Tensor] = None
        x_sum_sq: Optional[torch.Tensor] = None
        y_sum: Optional[torch.Tensor] = None
        y_sum_sq: Optional[torch.Tensor] = None

        for path in LazyTensor.iter_source_paths(expanded):
            try:
                meta = LazyTensor.from_meta(path)
            except Exception:
                return None
            rel = meta.get("scaler_stats_path")
            if not rel:
                return None
            stats_path = os.path.join(os.fspath(path), os.fspath(rel))
            if not os.path.isfile(stats_path):
                return None
            try:
                payload = torch.load(stats_path, map_location="cpu")
            except Exception:
                return None
            if not isinstance(payload, dict):
                return None
            if int(payload.get("version") or 0) != 1:
                return None
            c = int(payload.get("train_count") or 0)
            if c <= 0:
                return None

            xs = payload.get("x_sum")
            xs2 = payload.get("x_sum_sq")
            ys = payload.get("y_sum")
            ys2 = payload.get("y_sum_sq")
            if xs is None or xs2 is None or ys is None or ys2 is None:
                return None

            xs = xs.detach().to(dtype=torch.float64, device="cpu") if isinstance(xs, torch.Tensor) else torch.as_tensor(xs, dtype=torch.float64)
            xs2 = xs2.detach().to(dtype=torch.float64, device="cpu") if isinstance(xs2, torch.Tensor) else torch.as_tensor(xs2, dtype=torch.float64)
            ys = ys.detach().to(dtype=torch.float64, device="cpu") if isinstance(ys, torch.Tensor) else torch.as_tensor(ys, dtype=torch.float64)
            ys2 = ys2.detach().to(dtype=torch.float64, device="cpu") if isinstance(ys2, torch.Tensor) else torch.as_tensor(ys2, dtype=torch.float64)

            if x_sum is None:
                x_sum = xs.clone()
                x_sum_sq = xs2.clone()
                y_sum = ys.clone()
                y_sum_sq = ys2.clone()
            else:
                if xs.shape != x_sum.shape or xs2.shape != x_sum_sq.shape:
                    return None
                if ys.shape != y_sum.shape or ys2.shape != y_sum_sq.shape:
                    return None
                x_sum += xs
                x_sum_sq += xs2
                y_sum += ys
                y_sum_sq += ys2

            total += c

        if total <= 0 or x_sum is None or x_sum_sq is None or y_sum is None or y_sum_sq is None:
            return None

        return {
            "train_count": int(total),
            "x_sum": x_sum,
            "x_sum_sq": x_sum_sq,
            "y_sum": y_sum,
            "y_sum_sq": y_sum_sq,
        }

    @staticmethod
    def expand_sources(sources: Any) -> Any:
        import json as _json

        def _expand_from_root(spec: Any) -> Tuple[Any, bool]:
            if not isinstance(spec, dict) or "path" not in spec or "kind" not in spec:
                return spec, False
            root = os.fspath(spec.get("path") or "")
            mn_path = os.path.join(root, "multinode.json")
            if not os.path.isfile(mn_path):
                return spec, False
            with open(mn_path, "r", encoding="utf-8") as f:
                payload = _json.load(f)
            if isinstance(payload, dict):
                resolved = {
                    str(k): {"kind": "memmap", "path": os.path.join(root, str(v))}
                    for k, v in payload.items()
                }
                return resolved, True
            if isinstance(payload, list):
                resolved = [{"kind": "memmap", "path": os.path.join(root, str(v))} for v in payload]
                return resolved, True
            return spec, False

        expanded, ok = _expand_from_root(sources)
        if ok:
            return expanded
        if isinstance(sources, (list, tuple)) and len(sources) == 1:
            expanded, ok = _expand_from_root(sources[0])
            if ok:
                return expanded
        return sources
