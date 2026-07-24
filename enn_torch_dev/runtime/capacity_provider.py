from __future__ import annotations

from typing import Protocol, runtime_checkable

from .pressure import ResourceCapacity


@runtime_checkable
class ResourceCapacityProvider(Protocol):
    """Resolve the resource capacity to use for one runtime pass."""

    def capacity(self) -> ResourceCapacity:
        ...
