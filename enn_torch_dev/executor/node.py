from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import nn

from .schema import KeyRef
from .store import KVStore


@dataclass(slots=True)
class NodeSpec:
    """Minimal executable leaf-node schema."""

    name: str
    module: nn.Module
    input_args: list[KeyRef] = field(default_factory=list)
    input_kwargs: dict[str, KeyRef] = field(default_factory=dict)
    output_key: str = ""


class NodeExecutor:
    """Leaf executor that calls one ``nn.Module``."""

    def __init__(self, spec: NodeSpec) -> None:
        if not isinstance(spec, NodeSpec):
            raise TypeError(f"NodeExecutor expects NodeSpec, got {type(spec)!r}")
        if not isinstance(spec.module, nn.Module):
            raise TypeError("NodeSpec.module must be an nn.Module.")
        if not isinstance(spec.name, str) or not spec.name.strip():
            raise ValueError("NodeSpec.name must be a non-empty string.")
        if not isinstance(spec.output_key, str) or not spec.output_key.strip():
            raise ValueError("NodeSpec.output_key must be a non-empty string.")

        self.spec = spec

    @property
    def module(self) -> nn.Module:
        return self.spec.module

    def run(self, store: KVStore) -> Any:
        if not isinstance(store, KVStore):
            raise TypeError(f"NodeExecutor.run expects KVStore, got {type(store)!r}")

        args = [store.resolve(ref) for ref in self.spec.input_args]
        kwargs = {name: store.resolve(ref) for name, ref in self.spec.input_kwargs.items()}
        out = self.spec.module(*args, **kwargs)
        store.set(self.spec.output_key, out, origin=self.spec.name)
        return out
