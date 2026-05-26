from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import nn

from .schema import KeyRef
from .store import KVStore


def _validate_spec_key(value: object, field_name: str) -> str:
    label = f"NodeSpec.{field_name}"
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
class NodeSpec:
    """
    Minimal executable leaf-node schema.

    ``module_key`` points to a module registered by the owning graph executor.
    The node spec itself does not own an ``nn.Module`` so parameter registration
    can stay centralized in a future ``GraphExecutor(nn.Module)``.
    """

    name: str
    module_key: str | None = None
    input_args: list[KeyRef] = field(default_factory=list)
    input_kwargs: dict[str, KeyRef] = field(default_factory=dict)
    output_key: str = ""

    def __post_init__(self) -> None:
        self.name = _validate_spec_key(self.name, "name")
        if self.module_key is None:
            self.module_key = self.name
        else:
            self.module_key = _validate_spec_key(
                self.module_key, "module_key"
            )
        self.output_key = _validate_spec_key(self.output_key, "output_key")


class NodeExecutor:
    """Leaf executor that binds store values and calls one ``nn.Module``."""

    def __init__(self, spec: NodeSpec) -> None:
        if not isinstance(spec, NodeSpec):
            raise TypeError(f"NodeExecutor expects NodeSpec, got {type(spec)!r}")

        self.spec = spec

    @property
    def module_key(self) -> str:
        module_key = self.spec.module_key
        if module_key is None:
            raise RuntimeError("NodeSpec.module_key was not initialized.")
        return module_key

    @property
    def output_key(self) -> str:
        return self.spec.output_key

    def output_ref(self) -> KeyRef:
        return KeyRef(self.output_key)

    def run(self, store: KVStore, module: nn.Module) -> Any:
        if not isinstance(store, KVStore):
            raise TypeError(f"NodeExecutor.run expects KVStore, got {type(store)!r}")
        if not isinstance(module, nn.Module):
            raise TypeError("NodeExecutor.run module must be an nn.Module.")

        args = [store.resolve(ref) for ref in self.spec.input_args]
        kwargs = {
            name: store.resolve(ref)
            for name, ref in self.spec.input_kwargs.items()
        }
        out = module(*args, **kwargs)
        store.set(self.spec.output_key, out, origin=self.spec.name)
        return out
