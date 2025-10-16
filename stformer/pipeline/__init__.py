from __future__ import annotations

import json
import os

from .collate import stream, to_batch
from .dataset import Batch, MemoryMappedTensorStream
from .distributed import IOController


def _meta(memmap_dir: str) -> dict:
    path = os.path.join(memmap_dir, "meta.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = [
    "Batch",
    "to_batch",
    "stream",
    "IOController",
    "MemoryMappedTensorStream",
]
