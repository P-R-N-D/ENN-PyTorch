from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from .schema import GraphValue, KeyRef


class KVStore:
    """
    Runtime key-value store for executor-driven graph execution.

    This is a data plane, not an execution backend:
      - no graph topology;
      - no topological ordering;
      - no parent/child ownership;
      - no tile/stream policy.

    Values are stored as ``GraphValue`` wrappers so later executor stages can
    attach lightweight metadata while keeping the initial implementation small.
    """

    def __init__(self, initial: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, GraphValue] = {}

        if initial is not None:
            for key, value in initial.items():
                self.set(key, value)

    def __contains__(self, key: object) -> bool:
        return (
            isinstance(key, str)
            and self._is_valid_key(key)
            and key in self._data
        )

    def __len__(self) -> int:
        return len(self._data)

    @staticmethod
    def _validate_key(key: str) -> str:
        if not isinstance(key, str):
            raise TypeError("KVStore key must be a string.")
        if not key:
            raise ValueError("KVStore key must be a non-empty string.")
        if key != key.strip():
            raise ValueError(
                "KVStore key must not have leading or trailing whitespace."
            )
        return key

    @staticmethod
    def _is_valid_key(key: object) -> bool:
        return (
            isinstance(key, str)
            and bool(key)
            and key == key.strip()
        )

    def has(self, key: object) -> bool:
        return self._is_valid_key(key) and key in self._data

    def keys(self) -> tuple[str, ...]:
        return tuple(self._data.keys())

    def get(
        self,
        key: str,
        *,
        optional: bool = False,
        default: Any = None,
    ) -> Any:
        key = self._validate_key(key)
        if key not in self._data:
            if optional:
                return default
            raise KeyError(f"KVStore missing key: {key!r}")

        return self._data[key].data

    def get_value(self, key: str) -> GraphValue:
        key = self._validate_key(key)
        if key not in self._data:
            raise KeyError(f"KVStore missing key: {key!r}")

        return self._data[key]

    def set(
        self,
        key: str,
        value: Any,
        *,
        layout: str | None = None,
        mask_key: str | None = None,
        origin: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        self.set_value(
            key,
            GraphValue(
                data=value,
                layout=layout,
                mask_key=mask_key,
                origin=origin,
                meta=dict(meta or {}),
            ),
        )

    def set_value(self, key: str, value: GraphValue) -> None:
        key = self._validate_key(key)
        if not isinstance(value, GraphValue):
            raise TypeError(f"set_value expects GraphValue, got {type(value)!r}")

        self._data[key] = value

    def resolve(self, ref: KeyRef) -> Any:
        if not isinstance(ref, KeyRef):
            raise TypeError(f"KVStore.resolve expects KeyRef, got {type(ref)!r}")

        return self.get(
            ref.key,
            optional=ref.optional,
            default=ref.default,
        )

    def delete(self, key: str, *, missing_ok: bool = False) -> None:
        key = self._validate_key(key)
        if key not in self._data:
            if missing_ok:
                return
            raise KeyError(f"KVStore missing key: {key!r}")

        del self._data[key]

    def clear(self) -> None:
        self._data.clear()

    def to(
        self,
        device: torch.device | str,
        *,
        keys: list[str] | tuple[str, ...] | None = None,
        non_blocking: bool = False,
    ) -> "KVStore":
        target_keys = self.keys() if keys is None else tuple(keys)

        for key in target_keys:
            gv = self.get_value(key)
            value = gv.data

            if isinstance(value, Tensor):
                self.set(
                    key,
                    value.to(device=device, non_blocking=non_blocking),
                    layout=gv.layout,
                    mask_key=gv.mask_key,
                    origin=gv.origin,
                    meta=gv.meta,
                )

        return self

    def shallow_copy(self) -> "KVStore":
        copied = KVStore()
        copied._data = dict(self._data)
        return copied
