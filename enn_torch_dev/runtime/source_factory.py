from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from enn_torch_dev.data import KVBatch


@runtime_checkable
class RuntimePassSourceFactory(Protocol):
    """Create one fresh finite KVBatch source for a runtime pass index."""

    def create_pass_source(self, pass_index: int) -> Iterable[KVBatch]:
        ...
