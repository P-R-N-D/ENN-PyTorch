# -*- coding: utf-8 -*-
from __future__ import annotations

import collections.abc as _abc
import math
import os
import contextlib
import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import torch

_BOOTSTRAP_DEPTH = 0

class LazyDict(_abc.Mapping):
    def __init__(self, keys: Any, getter: Any, *, name: str = "LazyDict", cache: bool = False) -> None:
        self._keys = keys
        self._getter = getter
        self._name = str(name or "LazyDict")
        self._cache_enabled = bool(cache)
        self._cache: Optional[dict[Any, Any]] = {} if self._cache_enabled else None

    def __len__(self) -> int:
        return int(len(self._keys))

    def __iter__(self):
        return iter(self._keys)

    def __getitem__(self, key: Any) -> Any:
        if self._cache is not None and key in self._cache:
            return self._cache[key]
        v = self._getter(key)
        if self._cache is not None:
            self._cache[key] = v
        return v

    def __contains__(self, key: object) -> bool:
        try:
            return key in self._keys
        except Exception:
            return False

    def keys(self):
        return self._keys

    def values(self):
        return (self[k] for k in self._keys)

    def items(self):
        return ((k, self[k]) for k in self._keys)

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def collect(self) -> dict[Any, Any]:
        return {k: self[k] for k in self._keys}

    def materialize(self) -> dict[Any, Any]:
        return self.collect()


class Page:

    __slots__ = ("_buf", "_numel", "_dtype")

    def __init__(self, numel: int, dtype: "torch.dtype") -> None:
        import torch

        self._numel = int(max(1, numel))
        self._dtype = dtype
        self._buf = torch.empty(
            self._numel, dtype=self._dtype, device="cpu", pin_memory=True
        )

    @property
    def numel(self) -> int:
        return self._numel

    @property
    def dtype(self) -> "torch.dtype":
        return self._dtype

    def view(self, *shape: int) -> "torch.Tensor":
        import torch

        needed = 1
        for s in shape:
            needed *= int(s)
        if needed > self._numel:
            self._numel = int(needed)
            self._buf = torch.empty(
                self._numel, dtype=self._dtype, device="cpu", pin_memory=True
            )
        return self._buf[:needed].view(*shape)


class Pool:

    class Token:
        __slots__ = ("i", "g")

        def __init__(self, i: int, g: int) -> None:
            self.i = i
            self.g = g

    class _Entry:
        __slots__ = ("page", "busy", "fence", "gen")

        def __init__(self, page: "Page") -> None:
            self.page = page
            self.busy = False
            self.fence = None
            self.gen = 0

    def __init__(self, capacity: int = 4) -> None:
        import threading

        self._cap = max(1, int(capacity))
        self._pages: list[Pool._Entry] = []
        self._rr = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._cap

    def _evt_done(self, evt: object) -> bool:
        if evt is None:
            return True
        try:
            q = getattr(evt, "query", None)
            if callable(q):
                return bool(q())
        except Exception:
            return False
        return False

    def _scavenge(self) -> None:
        for e in self._pages:
            if e.busy and e.fence is not None and self._evt_done(e.fence):
                e.busy = False
                e.fence = None

    def _ensure_view(
        self, e: "Pool._Entry", shape: "Tuple[int, ...]", dtype: "torch.dtype"
    ) -> "torch.Tensor":
        need = 1
        for s in shape:
            need *= int(s)
        if (e.page.dtype != dtype) or (e.page.numel < need):
            e.page = Page(numel=need, dtype=dtype)
            e.gen += 1
        return e.page.view(*shape)

    def get(
        self,
        shape: "Tuple[int, ...]",
        dtype: "torch.dtype",
        *,
        return_handle: bool = False,
    ) -> "torch.Tensor" | "tuple[torch.Tensor, Pool.Token | None]":
        with self._lock:
            self._scavenge()
            n = len(self._pages)
            if n:
                start = self._rr
                for k in range(n):
                    idx = (start + k) % n
                    e = self._pages[idx]
                    if not e.busy:
                        e.busy = True
                        e.fence = None
                        self._rr = (idx + 1) % max(1, n)
                        view = self._ensure_view(e, shape, dtype)
                        if return_handle:
                            return view, Pool.Token(idx, e.gen)
                        return view
            need = 1
            for s in shape:
                need *= int(s)
            new = Pool._Entry(Page(numel=need, dtype=dtype))
            new.busy = True
            if len(self._pages) < self._cap:
                self._pages.append(new)
                idx = len(self._pages) - 1
                self._rr = (idx + 1) % self._cap
                view = new.page.view(*shape)
                if return_handle:
                    return view, Pool.Token(idx, new.gen)
                return view
            start = self._rr
            for k in range(self._cap):
                idx = (start + k) % self._cap
                if not self._pages[idx].busy:
                    self._pages[idx] = new
                    self._rr = (idx + 1) % self._cap
                    view = new.page.view(*shape)
                    if return_handle:
                        return view, Pool.Token(idx, new.gen)
                    return view
            view = new.page.view(*shape)
            if return_handle:
                return view, None
            return view

    def get_like(
        self, t: "torch.Tensor", *args: Any, return_handle: bool = False
    ) -> "torch.Tensor" | "tuple[torch.Tensor, Pool.Token | None]":
        return self.get(tuple(t.shape), t.dtype, return_handle=return_handle)

    def release_after(self, token: "Pool.Token", wait_event: object | None) -> None:
        if token is None:
            return
        with self._lock:
            i = int(getattr(token, "i", -1))
            g = int(getattr(token, "g", -1))
            if 0 <= i < len(self._pages):
                e = self._pages[i]
                if e.gen == g:
                    e.busy = True
                    e.fence = wait_event

    def release(self, token: "Pool.Token") -> None:
        if token is None:
            return
        with self._lock:
            i = int(getattr(token, "i", -1))
            g = int(getattr(token, "g", -1))
            if 0 <= i < len(self._pages):
                e = self._pages[i]
                if e.gen == g:
                    e.busy = False
                    e.fence = None

    def collect(self) -> None:
        with self._lock:
            self._scavenge()


class Cache:

    def __init__(self, root: str, max_queue: int = 8) -> None:
        import os
        import queue
        import threading

        self._q = queue.Queue(maxsize=max_queue)
        self._root = root
        os.makedirs(root, exist_ok=True)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._err = None
        self._err_event = threading.Event()
        self._t.start()

    def submit(
        self,
        tensor: "torch.Tensor",
        path: Optional[str] = None,
        idx: Optional[int] = None,
        wait_event: Optional[object] = None,
        release_cb: Optional[object] = None,
    ) -> None:
        import contextlib
        import queue

        if self._err_event.is_set():
            raise RuntimeError(f"Async writer error: {self._err!r}")
        if path is None:
            if idx is None:
                raise ValueError("either path or idx required")
            path = os.path.join(self._root, f"chunk_{int(idx):06d}.pt")
        try:
            self._q.put((tensor, path, wait_event, release_cb), timeout=0.05)
        except queue.Full:
            if wait_event is not None:
                with contextlib.suppress(Exception):
                    wait_event.synchronize()
            self._save_tensor(tensor, path)
            if callable(release_cb):
                with contextlib.suppress(Exception):
                    release_cb()

    def _save_tensor(self, tensor: "torch.Tensor", path: str) -> None:
        import json
        import os

        import torch

        try:
            if path.endswith(".mmt"):
                from tensordict import MemoryMappedTensor

                buf = tensor
                if hasattr(tensor, "is_pinned") and tensor.is_pinned():
                    buf = torch.empty_like(tensor, device="cpu", pin_memory=False)
                    buf.copy_(tensor, non_blocking=False)
                MemoryMappedTensor.from_tensor(buf.contiguous(), filename=path)
                meta = {"shape": list(buf.shape), "dtype": str(buf.dtype).replace("torch.", "")}
                with open(path + ".json", "w", encoding="utf-8") as f:
                    json.dump(meta, f)
                return
        except Exception:
            pass

        if hasattr(tensor, "is_pinned") and tensor.is_pinned():
            buf = torch.empty_like(tensor, device="cpu", pin_memory=False)
            buf.copy_(tensor, non_blocking=False)
        else:
            buf = tensor.contiguous()
        torch.save(buf, path)

    def close(self) -> None:
        self._q.put((None, None, None, None))
        self._t.join()

    def _run(self) -> None:
        import contextlib

        while True:
            item = self._q.get()
            if isinstance(item, tuple) and len(item) == 4:
                tensor, path, evt, rel = item
            else:
                tensor, path = item
                evt = None
                rel = None
            if tensor is None:
                break
            try:
                if evt is not None:
                    with contextlib.suppress(Exception):
                        evt.synchronize()
                self._save_tensor(tensor, path)
                if callable(rel):
                    with contextlib.suppress(Exception):
                        rel()
            except Exception as e:
                self._err = e
                self._err_event.set()
                break

    def had_error(self) -> bool:
        return bool(self._err_event.is_set())


class Buffer:
    """Bounded in-memory buffer with backpressure.

    This is a small wrapper around :class:`queue.Queue` that adds:
      - A stop flag so blocked producers/consumers can be interrupted.
      - Type-agnostic payloads (Any).

    It is intentionally minimal: higher-level coordination (sentinels,
    producer/consumer threads) lives in BufferedLoader.
    """

    def __init__(self, max_batches: int) -> None:
        import queue
        import threading

        self.max_batches = max(1, int(max_batches))
        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=self.max_batches)
        self._stop = threading.Event()

    def put(self, item: Any, *, timeout: float | None = None) -> bool:
        """Put an item into the buffer.

        Returns True if the item was enqueued, False if the buffer was stopped
        before the item could be enqueued.

        This method is interruptible even when `timeout` is None (infinite):
        it uses a small internal timeout loop to periodically check the stop flag.
        """
        import logging
        import queue
        import time

        if self._stop.is_set():
            return False

        start = time.monotonic()
        if timeout is None:
            while not self._stop.is_set():
                try:
                    self._q.put(item, block=True, timeout=0.1)
                    break
                except queue.Full:
                    continue
            else:
                return False
        else:
            deadline = time.monotonic() + float(timeout)
            while not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                try:
                    self._q.put(item, block=True, timeout=min(0.1, remaining))
                    break
                except queue.Full:
                    continue
            else:
                return False

        elapsed = time.monotonic() - start
        if elapsed > 0.1:
            logging.warning(
                f"Buffer.put blocked for {elapsed:.3f} s (max_batches={self.max_batches})"
            )
        return True

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        """Get an item from the buffer.

        If the buffer is stopped and empty, this raises queue.Empty.

        Like put(), this method is interruptible even when `timeout` is None.
        """
        import queue
        import time

        if not bool(block):
            if self._stop.is_set() and self._q.empty():
                raise queue.Empty
            return self._q.get(block=False)

        if timeout is None:
            while True:
                if self._stop.is_set() and self._q.empty():
                    raise queue.Empty
                try:
                    return self._q.get(block=True, timeout=0.1)
                except queue.Empty:
                    continue

        deadline = time.monotonic() + float(timeout)
        while True:
            if self._stop.is_set() and self._q.empty():
                raise queue.Empty
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise queue.Empty
            try:
                return self._q.get(block=True, timeout=min(0.1, remaining))
            except queue.Empty:
                continue

    def empty(self) -> bool:
        return self._q.empty()

    def size(self) -> int:
        return self._q.qsize()

    def stop(self) -> None:
        self._stop.set()

    def is_stopped(self) -> bool:
        return bool(self._stop.is_set())


