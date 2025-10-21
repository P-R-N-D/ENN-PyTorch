# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from tensordict import MemoryMappedTensor

try:
    from torchdata.nodes import IterableWrapper
except Exception:
    from torchdata.datapipes.iter import IterableWrapper

from ..toolkit.compat import patch_arrow
from .datatype import to


_ARROW = patch_arrow()
pa = _ARROW.module


def _read_meta(memmap_dir: str) -> Dict[str, Any]:
    path = os.path.join(memmap_dir, "meta.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class SampleReader:
    def __init__(
        self,
        memmap_dir: str,
        *args: Any,
        split: str = "train",
        val_frac: Optional[float] = None,
        batch_size: Optional[int] = None,
        **kwargs: Any,
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
        **kwargs: Any,
    ) -> SampleReader:
        return cls(
            memmap_dir,
            split=split,
            val_frac=val_frac,
            batch_size=int(batch_size),
        )

    @staticmethod
    def materialize(
        data: Dict[str, Any],
        *args: Any,
        memmap_dir: str,
        train_frac: float = 1.0,
        val_frac: float = 0.0,
        shuffle: bool = False,
        **kwargs: Any,
    ) -> None:
        os.makedirs(memmap_dir, exist_ok=True)
        features = (
            torch.as_tensor(data["features"]).detach().cpu().contiguous()
        )
        labels = torch.as_tensor(data["labels"]).detach().cpu().contiguous()
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features/labels N mismatch")
        count = int(features.shape[0])
        feat_dim = int(features.view(count, -1).shape[1])
        label_shape: List[int] = list(labels.shape[1:])
        label_flat = int(labels.numel() // count)
        if shuffle:
            perm = torch.randperm(count)
            features = features.index_select(0, perm)
            labels = labels.index_select(0, perm)
        feat_path = os.path.join(memmap_dir, "features.mmt")
        label_path = os.path.join(memmap_dir, "labels.mmt")
        MemoryMappedTensor.from_tensor(
            features.view(count, feat_dim), filename=feat_path, existsok=True
        )
        MemoryMappedTensor.from_tensor(
            labels.view(count, label_flat), filename=label_path, existsok=True
        )
        meta = {
            "N": count,
            "feature_dim": feat_dim,
            "label_shape": label_shape,
            "features_arrow_dtype": to(features.dtype, "arrow"),
            "labels_arrow_dtype": to(labels.dtype, "arrow"),
            "fractions": [float(train_frac), float(val_frac)],
            "features_filename": "features.mmt",
            "labels_filename": "labels.mmt",
        }
        with open(
            os.path.join(memmap_dir, "meta.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(meta, handle)

    def _load_meta(self) -> Dict[str, Any]:
        if self._meta is None:
            with open(
                os.path.join(self.dir, "meta.json"), "r", encoding="utf-8"
            ) as handle:
                self._meta = json.load(handle)
        return self._meta

    def _indices(self) -> range:
        meta = self._load_meta()
        total = int(meta["N"])
        val_fraction = float(
            self._val_frac_override
            if self._val_frac_override is not None
            else meta.get("fractions", [1.0, 0.0])[-1]
        )
        val_count = int(round(total * val_fraction))
        train_count = total - val_count
        match self.split:
            case "train":
                return range(0, train_count)
            case "val":
                return range(train_count, total)
            case _:
                return range(0, total)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        meta = self._load_meta()
        total = int(meta["N"])
        feat_dim = int(meta["feature_dim"])
        label_shape = list(meta["label_shape"])
        label_flat = (
            int(torch.tensor(label_shape).prod().item()) if label_shape else 1
        )
        feat_path = os.path.join(
            self.dir, meta.get("features_filename", "features.mmt")
        )
        label_path = os.path.join(
            self.dir, meta.get("labels_filename", "labels.mmt")
        )
        feat_dtype = to(meta.get("features_arrow_dtype", "float32"), "torch")
        label_dtype = to(meta.get("labels_arrow_dtype", "float32"), "torch")
        feat_mmt = MemoryMappedTensor.from_filename(
            feat_path, dtype=feat_dtype, shape=(total, feat_dim)
        )
        label_mmt = MemoryMappedTensor.from_filename(
            label_path, dtype=label_dtype, shape=(total, label_flat)
        )
        for index in self._indices():
            feat = feat_mmt[index]
            label = label_mmt[index].view(*label_shape)
            feat_tensor = (
                feat
                if isinstance(feat, torch.Tensor)
                else torch.as_tensor(feat)
            )
            label_tensor = (
                label
                if isinstance(label, torch.Tensor)
                else torch.as_tensor(label)
            )
            yield (feat_tensor, label_tensor)

    def __len__(self) -> int:
        return len(self._indices())

    def batch_range(
        self, start: int, end: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        meta = self._load_meta()
        total = int(meta["N"])
        feat_dim = int(meta["feature_dim"])
        label_shape = list(meta["label_shape"])
        label_flat = (
            int(torch.tensor(label_shape).prod().item()) if label_shape else 1
        )
        feat_path = os.path.join(
            self.dir, meta.get("features_filename", "features.mmt")
        )
        label_path = os.path.join(
            self.dir, meta.get("labels_filename", "labels.mmt")
        )
        feat_dtype = to(meta.get("features_arrow_dtype", "float32"), "torch")
        label_dtype = to(meta.get("labels_arrow_dtype", "float32"), "torch")
        feat_mmt = MemoryMappedTensor.from_filename(
            feat_path, dtype=feat_dtype, shape=(total, feat_dim)
        )
        label_mmt = MemoryMappedTensor.from_filename(
            label_path, dtype=label_dtype, shape=(total, label_flat)
        )
        features = feat_mmt[start:end]
        labels = label_mmt[start:end].view(-1, *label_shape)
        features_tensor = (
            features
            if isinstance(features, torch.Tensor)
            else torch.as_tensor(features)
        )
        labels_tensor = (
            labels
            if isinstance(labels, torch.Tensor)
            else torch.as_tensor(labels)
        )
        return (features_tensor, labels_tensor)

    @staticmethod
    def to_record_batch(
        features: torch.Tensor, labels: torch.Tensor
    ) -> pa.RecordBatch:
        batch = int(features.shape[0])
        feat_dim = int(features.view(batch, -1).shape[1])
        label_flat = int(labels.view(batch, -1).shape[1])
        feat_values = pa.array(
            np.asarray(features.contiguous().view(-1).cpu().numpy())
        )
        feat_array = _ARROW.fixed_shape_list_from_arrays(feat_values, feat_dim)
        label_values = pa.array(
            np.asarray(labels.contiguous().view(-1).cpu().numpy())
        )
        label_array = _ARROW.fixed_shape_list_from_arrays(label_values, label_flat)
        return pa.record_batch(
            [feat_array, label_array], names=["features", "labels"]
        )


class BatchReader:
    def __init__(
        self,
        mmts: SampleReader,
        start: int,
        end: int,
        batch_size: int,
    ) -> None:
        self._mmts = mmts
        self._start = int(start)
        self._end = int(end)
        self._batch = max(1, int(batch_size))

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        index = int(self._start)
        while index < self._end:
            nxt = min(index + self._batch, self._end)
            xb, yb = self._mmts.batch_range(index, nxt)
            yield {"X": xb, "Y": yb}
            index = nxt

    def __len__(self) -> int:
        if self._end <= self._start:
            return 0
        span = self._end - self._start
        return int(math.ceil(span / float(self._batch)))


class BatchSampler(IterableWrapper):
    def __init__(
        self,
        memmap_dir: str,
        part: str,
        batch_size: int,
        shuffle: bool,
        seed: int,
        *args: Any,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
        fractions: Optional[Tuple[float, float]] = None,
        **kwargs: Any,
    ) -> None:
        meta = _read_meta(memmap_dir)
        total = int(meta["N"])
        if fractions is not None:
            train_frac = float(fractions[0])
        else:
            train_frac = 1.0
            if "fractions" in meta:
                try:
                    train_frac = float(meta["fractions"][0])
                except Exception:
                    train_frac = 1.0
        if part not in {"train", "val"}:
            raise ValueError("part must be 'train' or 'val'")
        train_count = int(math.floor(total * train_frac))
        start, end = (
            (0, train_count) if part == "train" else (train_count, total)
        )
        indices = list(range(start, end))
        if shuffle:
            rng = random.Random(int(seed))
            rng.shuffle(indices)
        if world_size > 1:
            indices = indices[int(rank) :: int(world_size)]
        batch_len = int(batch_size)
        batches: List[List[int]] = []
        current: List[int] = []
        for idx in indices:
            current.append(int(idx))
            if len(current) == batch_len:
                batches.append(current)
                current = []
        if current and (not drop_last):
            batches.append(current)

        super().__init__([list(chunk) for chunk in batches])
