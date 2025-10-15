# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, Iterator, List, Optional, Tuple

import json
import math
import os
import random

import numpy as np
import pyarrow as pa
import torch

try:
    from torchdata.nodes import IterableWrapper  
except Exception:
    from torchdata.datapipes.iter import IterableWrapper 

from tensordict import MemoryMappedTensor
from . import _meta

_TORCH2NAME: Dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    getattr(torch, "bfloat16", torch.float32): "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}

_NAME2TORCH: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": getattr(torch, "bfloat16", torch.float32),
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


class MemoryMappedTensorStream:
    def __init__(
        self,
        memmap_dir: str,
        *args: Any,
        split: str = "train",
        val_frac: Optional[float] = None,
        batch_size: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        self.dir = memmap_dir
        self.split = split
        self._val_frac_override = val_frac
        self._batch_size = batch_size
        self._meta: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dir(
        cls,
        memmap_dir: str,
        *args: Any,
        split: str = "train",
        batch_size: int = 1,
        val_frac: Optional[float] = None,
        **kwargs: Any
    ) -> "MemoryMappedTensorStream":
        return cls(memmap_dir, split=split, val_frac=val_frac, batch_size=int(batch_size))

    @staticmethod
    def materialize(
        data: Dict[str, Any],
        *args: Any,
        memmap_dir: str,
        train_frac: float = 1.0,
        val_frac: float = 0.0,
        shuffle: bool = False,
        **kwargs: Any
    ) -> None:
        os.makedirs(memmap_dir, exist_ok=True)
        X = torch.as_tensor(data["features"]).detach().cpu().contiguous()
        Y = torch.as_tensor(data["labels"]).detach().cpu().contiguous()
        if X.shape[0] != Y.shape[0]:
            raise ValueError("features/labels N mismatch")
        N = int(X.shape[0])
        F = int(X.view(N, -1).shape[1])
        lshape: List[int] = list(Y.shape[1:])
        Lflat = int(Y.numel() // N)
        if shuffle:
            perm = torch.randperm(N)
            X = X.index_select(0, perm)
            Y = Y.index_select(0, perm)
        fx = os.path.join(memmap_dir, "features.mmt")
        lb = os.path.join(memmap_dir, "labels.mmt")
        MemoryMappedTensor.from_tensor(X.view(N, F), filename=fx, existsok=True)
        MemoryMappedTensor.from_tensor(Y.view(N, Lflat), filename=lb, existsok=True)
        meta = {
            "N": N,
            "feature_dim": F,
            "label_shape": lshape,
            "features_arrow_dtype": _TORCH2NAME[X.dtype],
            "labels_arrow_dtype": _TORCH2NAME[Y.dtype],
            "fractions": [float(train_frac), float(val_frac)],
            "features_filename": "features.mmt",
            "labels_filename": "labels.mmt",
        }
        with open(os.path.join(memmap_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def _load_meta(self) -> Dict[str, Any]:
        if self._meta is None:
            with open(os.path.join(self.dir, "meta.json"), "r", encoding="utf-8") as f:
                self._meta = json.load(f)
        return self._meta

    def _indices(self) -> range:
        m = self._load_meta()
        N = int(m["N"])
        vf = float(
            self._val_frac_override
            if self._val_frac_override is not None
            else m.get("fractions", [1.0, 0.0])[-1]
        )
        n_val = int(round(N * vf))
        n_tr = N - n_val
        if self.split == "train":
            return range(0, n_tr)
        if self.split == "val":
            return range(n_tr, N)
        return range(0, N)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        m = self._load_meta()
        N = int(m["N"])
        F = int(m["feature_dim"])
        lshape = list(m["label_shape"])
        Lflat = int(torch.tensor(lshape).prod().item()) if lshape else 1
        fx = os.path.join(self.dir, m.get("features_filename", "features.mmt"))
        lb = os.path.join(self.dir, m.get("labels_filename", "labels.mmt"))
        fdt = _NAME2TORCH[m.get("features_arrow_dtype", "float32")]
        ldt = _NAME2TORCH[m.get("labels_arrow_dtype", "float32")]
        f_mmt = MemoryMappedTensor.from_filename(fx, dtype=fdt, shape=(N, F))
        l_mmt = MemoryMappedTensor.from_filename(lb, dtype=ldt, shape=(N, Lflat))
        for i in self._indices():
            x = f_mmt[i]
            y = l_mmt[i].view(*lshape)
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x)
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y)
            yield x, y

    def __len__(self) -> int:
        return len(self._indices())

    def batch_range(self, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self._load_meta()
        N = int(m["N"])
        F = int(m["feature_dim"])
        lshape = list(m["label_shape"])
        Lflat = int(torch.tensor(lshape).prod().item()) if lshape else 1
        fx = os.path.join(self.dir, m.get("features_filename", "features.mmt"))
        lb = os.path.join(self.dir, m.get("labels_filename", "labels.mmt"))
        fdt = _NAME2TORCH[m.get("features_arrow_dtype", "float32")]
        ldt = _NAME2TORCH[m.get("labels_arrow_dtype", "float32")]
        f_mmt = MemoryMappedTensor.from_filename(fx, dtype=fdt, shape=(N, F))
        l_mmt = MemoryMappedTensor.from_filename(lb, dtype=ldt, shape=(N, Lflat))
        Xb = f_mmt[start:end]
        Yb = l_mmt[start:end].view(-1, *lshape)
        Xb_t = Xb if isinstance(Xb, torch.Tensor) else torch.as_tensor(Xb)
        Yb_t = Yb if isinstance(Yb, torch.Tensor) else torch.as_tensor(Yb)
        return Xb_t, Yb_t

    @staticmethod
    def to_record_batch(Xb: torch.Tensor, Yb: torch.Tensor) -> pa.RecordBatch:
        B = int(Xb.shape[0])
        F = int(Xb.view(B, -1).shape[1])
        Lflat = int(Yb.view(B, -1).shape[1])
        fa = pa.FixedSizeListArray.from_arrays(
            pa.array(np.asarray(Xb.contiguous().view(-1).cpu().numpy())), F
        )
        la = pa.FixedSizeListArray.from_arrays(
            pa.array(np.asarray(Yb.contiguous().view(-1).cpu().numpy())), Lflat
        )
        return pa.record_batch([fa, la], names=["features", "labels"])


class Batch(IterableWrapper):
    def __init__(
        self,
        *args: Any,
        memmap_dir: str,
        part: str,
        batch_size: int,
        shuffle: bool,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
        fractions: Optional[Tuple[float, float]] = None,
        **kwargs: Any
    ) -> None:
        meta = _meta(memmap_dir)
        N = int(meta["N"])
        if fractions is not None:
            train_frac = float(fractions[0])
        else:
            train_frac = 1.0
            if "fractions" in meta:
                try:
                    train_frac = float(meta["fractions"][0])
                except Exception:
                    pass
        if part not in ("train", "val"):
            raise ValueError("part must be 'train' or 'val'")
        train_cnt = int(math.floor(N * train_frac))
        start, end = (0, train_cnt) if part == "train" else (train_cnt, N)
        idx = list(range(start, end))
        if shuffle:
            rng = random.Random(int(seed))
            rng.shuffle(idx)
        if world_size > 1:
            idx = idx[int(rank) :: int(world_size)]
        B = int(batch_size)
        out: List[List[int]] = []
        cur: List[int] = []
        for i in idx:
            cur.append(int(i))
            if len(cur) == B:
                out.append(cur)
                cur = []
        if cur and (not drop_last):
            out.append(cur)

        def _iter() -> Iterator[List[int]]:
            for it in out:
                yield list(it)

        super().__init__(_iter())