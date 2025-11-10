# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import queue
import threading
from typing import Any, Iterator, Mapping, Optional, Tuple, TypedDict, Literal

import torch

try:
    from torchdata.nodes import BaseNode
except Exception:
    BaseNode = object

try:
    from tensordict import TensorDictBase
    from tensordict import load_memmap
except Exception as e:
    raise RuntimeError("tensordict is required for Dataset") from e


def _to_device(batch: Any, device: torch.device, non_blocking: bool = True) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, Mapping):
        return {k: _to_device(v, device, non_blocking) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        seq = [_to_device(v, device, non_blocking) for v in batch]
        return type(batch)(seq) if isinstance(batch, tuple) else seq
    return batch


class Dataset:
    def __init__(self, memmap_dir: str, *args: Any, split: str = "train", val_frac: float = 0.0, **kwargs: Any) -> None:
        self.dir = os.fspath(memmap_dir)
        self.split = str(split)
        self._meta: Optional[Mapping[str, Any]] = None

        meta_path = os.path.join(self.dir, "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"meta.json not found under: {self.dir}")
        with open(meta_path, "r", encoding="utf-8") as f:
            self._meta = json.load(f)
        self._N = int(self._meta.get("N", 0))
        td_prefix = os.path.join(self.dir, self._meta.get("tensordict_prefix", "td_memmap"))
        nb = bool(int(os.environ.get("STNET_TD_NONBLOCKING_LOAD", "0")))
        td = load_memmap(td_prefix, non_blocking=nb)
        self._features = td.get("features")
        self._labels = td.get("labels")
        lshape = list(self._meta.get("label_shape") or [])
        self._label_shape: Tuple[int, ...] = tuple(lshape) if lshape else tuple()
        train_start = int(self._meta.get("train_start", 0))
        train_end   = int(self._meta.get("train_end",   self._N))
        val_start   = int(self._meta.get("val_start",   0))
        val_end     = int(self._meta.get("val_end",     0))

        if val_frac and not (val_end > val_start):
            vf = float(val_frac)
            vc = max(0, min(self._N, int(self._N * vf)))
            val_start, val_end = max(0, self._N - vc), self._N
            train_start, train_end = 0, val_start

        if self.split == "val":
            self._start, self._end = (val_start, val_end) if val_end > val_start else (0, 0)
        else:
            self._start, self._end = (train_start, train_end)

    @property
    def start(self) -> int:
        return int(self._start)

    @property
    def end(self) -> int:
        return int(self._end)

    @property
    def meta(self) -> Mapping[str, Any]:
        return dict(self._meta or {})

    def __len__(self) -> int:
        return max(0, int(self._end) - int(self._start))

    def _slice(self, start: int, end: int) -> Mapping[str, torch.Tensor]:
        x = self._features[start:end]
        y = self._labels[start:end]
        if self._label_shape:
            y = y.view(end - start, *self._label_shape)
        xt = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        yt = y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
        return {"X": xt, "Y": yt}

    def __getitem__(self, idx: int | Tuple[int, int]) -> Mapping[str, torch.Tensor]:
        if isinstance(idx, tuple) and len(idx) == 2:
            s, e = int(idx[0]), int(idx[1])
            return self._slice(s, e)
        i = self._start + int(idx)
        return self._slice(i, i + 1)

# ---- Source abstraction (no legacy path strings) -----------------------------
SourceKind = Literal["memmap"]

class SourceSpec(TypedDict):
    """Structured source spec (mandatory).
    kind: currently only "memmap"
    path: directory that contains meta.json and memmap shards
    """
    kind: SourceKind
    path: str

def dataset(
    source: SourceSpec,
    *,
    split: str = "train",
    val_frac: float = 0.0,
) -> "Dataset":
    """Create Dataset strictly from SourceSpec."""
    kind = str(source.get("kind"))
    if kind != "memmap":
        raise ValueError(f"Unsupported source kind: {kind!r}")
    path = os.fspath(source.get("path", ""))
    if not path:
        raise ValueError("SourceSpec['path'] must be provided")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"memmap directory not found: {path!r}")
    sp = str(split or "train")
    if sp not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got: {sp!r}")
    vf = float(val_frac)
    if not (0.0 <= vf <= 1.0):
        raise ValueError(f"val_frac must be in [0,1], got: {vf}")
    return Dataset(path, split=sp, val_frac=vf)


class Prefetcher:

    def __init__(
        self,
        iterable: Any,
        *args: Any,
        device: torch.device | str,
        depth: int = 2,
        non_blocking: bool = True,
        oom_safe: bool = True,
        gpu_guard_bytes: int | None = None,
        host_guard_bytes: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._src = iterable
        self._device = torch.device(device) if not isinstance(device, torch.device) else device
        self._depth = max(1, int(depth))
        self._non_blocking = bool(non_blocking)
        self._backpressure = bool(oom_safe)
        self._gpu_guard_bytes = int(gpu_guard_bytes or 0)
        self._host_guard_bytes = int(host_guard_bytes or 0)

    def __iter__(self) -> Iterator[Any]:
        it = iter(self._src)
        q: "queue.Queue[Optional[Any]]" = queue.Queue(maxsize=self._depth)
        sentinel = object()

        def _producer():
            try:
                for item in it:
                    moved = _to_device(item, self._device, non_blocking=self._non_blocking)
                    q.put(moved, block=True)
            except StopIteration:
                pass
            finally:
                q.put(sentinel, block=True)

        th = threading.Thread(target=_producer, daemon=True)
        th.start()

        while True:
            item = q.get(block=True)
            if item is sentinel:
                break
            yield item
        th.join(timeout=0.1)
