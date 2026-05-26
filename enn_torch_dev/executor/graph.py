from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from .node import NodeExecutor, NodeSpec
from .store import KVStore


def _is_valid_graph_key(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _validate_graph_key(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    if not value:
        raise ValueError(f"{label} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{label} must not have leading or trailing whitespace.")
    return value


def _validate_module_key(value: object) -> str:
    key = _validate_graph_key(value, "GraphExecutor module_key")
    if "." in key:
        raise ValueError("GraphExecutor module_key must not contain '.'.")
    return key


class GraphExecutor(nn.Module):
    def __init__(
        self,
        nodes: Sequence[tuple[NodeSpec, nn.Module]] | None = None,
    ) -> None:
        super().__init__()
        self.modules_by_key = nn.ModuleDict()
        self._nodes: dict[str, NodeExecutor] = {}
        self._order: list[str] = []

        if nodes is not None:
            for spec, module in nodes:
                self.add_node(spec, module)

    @staticmethod
    def _validate_node_name(name: object) -> str:
        return _validate_graph_key(name, "GraphExecutor node name")

    @staticmethod
    def _is_valid_node_name(name: object) -> bool:
        return _is_valid_graph_key(name)

    def add_node(self, spec: NodeSpec, module: nn.Module) -> str:
        if not isinstance(spec, NodeSpec):
            raise TypeError(f"add_node expects NodeSpec, got {type(spec)!r}")
        if not isinstance(module, nn.Module):
            raise TypeError("add_node module must be an nn.Module.")

        name = self._validate_node_name(spec.name)
        module_key = _validate_module_key(spec.module_key)

        if name in self._nodes:
            raise ValueError(f"Duplicate node name: {name!r}")
        if module_key in self.modules_by_key:
            raise ValueError(f"Duplicate module_key: {module_key!r}")

        self.modules_by_key[module_key] = module
        self._nodes[name] = NodeExecutor(spec)
        self._order.append(name)
        return name

    def remove_node(self, name: str, *, missing_ok: bool = False) -> None:
        name = self._validate_node_name(name)
        if name not in self._nodes:
            if missing_ok:
                return
            raise KeyError(f"Unknown node: {name!r}")

        node = self._nodes.pop(name)
        module_key = node.module_key

        if module_key in self.modules_by_key:
            del self.modules_by_key[module_key]

        try:
            self._order.remove(name)
        except ValueError:
            pass

    def get_node(self, name: str) -> NodeExecutor:
        name = self._validate_node_name(name)
        if name not in self._nodes:
            raise KeyError(f"Unknown node: {name!r}")
        return self._nodes[name]

    def has_node(self, name: object) -> bool:
        return self._is_valid_node_name(name) and name in self._nodes

    def node_names(self) -> tuple[str, ...]:
        return tuple(self._order)

    def run(self, store: KVStore) -> KVStore:
        if not isinstance(store, KVStore):
            raise TypeError(f"GraphExecutor.run expects KVStore, got {type(store)!r}")

        for name in tuple(self._order):
            node = self._nodes.get(name)
            if node is None:
                raise RuntimeError(
                    f"GraphExecutor execution order references missing node: {name!r}"
                )

            module_key = node.module_key
            if module_key not in self.modules_by_key:
                raise RuntimeError(
                    f"GraphExecutor node {name!r} references missing module_key: {module_key!r}"
                )

            module = self.modules_by_key[module_key]
            node.run(store, module)

        return store

    def forward(self, store: KVStore) -> KVStore:
        return self.run(store)
