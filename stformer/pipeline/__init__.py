# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os

from .dataset import Batch, MemoryMappedTensorStream
from .collate import forward, stream
from .distributed import DistributedIOCoordinator

__all__ = [
    'Batch',
    'forward',
    'stream',
    'DistributedIOCoordinator',
    'MemoryMappedTensorStream',
]

def _meta(memmap_dir: str) -> dict:
    with open(os.path.join(memmap_dir, 'meta.json'), 'r', encoding='utf-8') as f:
        return json.load(f)
