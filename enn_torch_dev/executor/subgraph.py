from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from torch import nn

from .schema import KeyRef
from .store import KVStore


def _validate_subgraph_key(value: object, field_name: str) -> str:
    label = f"SubgraphSpec.{field_name}"
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    if not value:
        raise ValueError(f"{label} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(
            f"{label} must not have leading or trailing whitespace."
        )
    return value


@dataclass(slots=True)
class SubgraphSpec:
    """
    Parent-node schema that aggregates already-registered child nodes.

    ``children`` contains graph node names. A child may be a leaf
    ``NodeExecutor`` or another ``SubgraphExecutor``.
    """

    name: str
    children: list[str]
    module_key: str | None = None
    input_kwargs: dict[str, KeyRef] = field(default_factory=dict)
    output_key: str = ""

    def __post_init__(self) -> None:
        self.name = _validate_subgraph_key(self.name, "name")
        if self.module_key is None:
            self.module_key = self.name
        else:
            self.module_key = _validate_subgraph_key(
                self.module_key, "module_key"
            )
        self.output_key = _validate_subgraph_key(self.output_key, "output_key")
        self.children = self._validate_children(self.children)
        self.input_kwargs = self._validate_input_kwargs(self.input_kwargs)

    @staticmethod
    def _validate_children(children: object) -> list[str]:
        if isinstance(children, (str, bytes, bytearray)) or children is None:
            raise TypeError("SubgraphSpec.children must be a sequence of strings.")

        try:
            values = list(children)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(
                "SubgraphSpec.children must be a sequence of strings."
            ) from exc

        if not values:
            raise ValueError("SubgraphSpec.children must not be empty.")

        out: list[str] = []
        seen: set[str] = set()
        for child in values:
            normalized = _validate_subgraph_key(child, "children")
            if normalized in seen:
                raise ValueError(
                    f"SubgraphSpec.children contains duplicate child: {normalized!r}"
                )
            seen.add(normalized)
            out.append(normalized)
        return out

    @staticmethod
    def _validate_input_kwargs(value: object) -> dict[str, KeyRef]:
        if not isinstance(value, Mapping):
            raise TypeError("SubgraphSpec.input_kwargs must be a mapping.")

        out: dict[str, KeyRef] = {}
        for key, ref in value.items():
            norm_key = _validate_subgraph_key(key, "input_kwargs key")
            if not isinstance(ref, KeyRef):
                raise TypeError(
                    "SubgraphSpec.input_kwargs values must be KeyRef instances."
                )
            out[norm_key] = ref
        return out


class SubgraphExecutor:
    """Parent executor that aggregates child node outputs."""

    def __init__(
        self,
        spec: SubgraphSpec,
        child_output_refs: Sequence[KeyRef],
    ) -> None:
        if not isinstance(spec, SubgraphSpec):
            raise TypeError(
                f"SubgraphExecutor expects SubgraphSpec, got {type(spec)!r}"
            )

        refs = list(child_output_refs)
        if not refs:
            raise ValueError("SubgraphExecutor requires child output references.")
        for ref in refs:
            if not isinstance(ref, KeyRef):
                raise TypeError(
                    "SubgraphExecutor child_output_refs must contain KeyRef instances."
                )

        self.spec = spec
        self._child_output_refs = tuple(refs)

    @property
    def module_key(self) -> str:
        module_key = self.spec.module_key
        if module_key is None:
            raise RuntimeError("SubgraphSpec.module_key was not initialized.")
        return module_key

    @property
    def output_key(self) -> str:
        return self.spec.output_key

    def output_ref(self) -> KeyRef:
        return KeyRef(self.output_key)

    @property
    def children(self) -> tuple[str, ...]:
        return tuple(self.spec.children)

    @property
    def child_output_refs(self) -> tuple[KeyRef, ...]:
        return self._child_output_refs

    def set_children(
        self,
        children: Sequence[str],
        child_output_refs: Sequence[KeyRef],
    ) -> None:
        normalized_children = SubgraphSpec._validate_children(children)
        refs = list(child_output_refs)
        if len(refs) != len(normalized_children):
            raise ValueError(
                "SubgraphExecutor children and child_output_refs must have "
                "the same length."
            )
        for ref in refs:
            if not isinstance(ref, KeyRef):
                raise TypeError(
                    "SubgraphExecutor child_output_refs must contain KeyRef instances."
                )

        self.spec.children = normalized_children
        self._child_output_refs = tuple(refs)

    def run(self, store: KVStore, module: nn.Module) -> Any:
        if not isinstance(store, KVStore):
            raise TypeError(
                f"SubgraphExecutor.run expects KVStore, got {type(store)!r}"
            )
        if not isinstance(module, nn.Module):
            raise TypeError("SubgraphExecutor.run module must be an nn.Module.")

        values = [store.resolve(ref) for ref in self._child_output_refs]
        kwargs = {
            name: store.resolve(ref)
            for name, ref in self.spec.input_kwargs.items()
        }
        out = module(values, **kwargs)
        store.set(self.spec.output_key, out, origin=self.spec.name)
        return out
