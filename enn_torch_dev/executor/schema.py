from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GraphValue:
    """
    One value stored in ``KVStore``.

    ``data`` is the actual runtime payload. Metadata fields are intentionally
    lightweight so this schema can wrap tensors, TensorDict-like containers,
    lazy values, or later executor-specific payloads without committing the
    store to a heavy backend.
    """

    data: Any
    layout: str | None = None
    mask_key: str | None = None
    origin: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class KeyRef:
    """
    Reference to a key in ``KVStore``.

    Node executors use ``KeyRef`` to bind module inputs without hard-coding
    Python variables into the execution path.
    """

    key: str
    optional: bool = False
    default: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str):
            raise TypeError("KeyRef.key must be a string.")
        if not self.key:
            raise ValueError("KeyRef.key must be a non-empty string.")
        if self.key != self.key.strip():
            raise ValueError(
                "KeyRef.key must not have leading or trailing whitespace."
            )
