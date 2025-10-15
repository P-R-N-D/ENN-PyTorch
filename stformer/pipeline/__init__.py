# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os


def _meta(memmap_dir: str) -> dict:
    with open(os.path.join(memmap_dir, 'meta.json'), 'r', encoding='utf-8') as f:
        return json.load(f)


from .dataset import Batch, MemoryMappedTensorStream  # noqa: E402
from .collate import forward, stream  # noqa: E402
from .distributed import DistributedIOCoordinator  # noqa: E402


__all__ = [
    'Batch',
    'forward',
    'stream',
    'DistributedIOCoordinator',
    'MemoryMappedTensorStream',
]
