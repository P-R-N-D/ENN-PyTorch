from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from .node import NodeExecutor, NodeSpec
from .store import KVStore
from .subgraph import SubgraphExecutor, SubgraphSpec

GraphNodeExecutor = NodeExecutor | SubgraphExecutor


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
        self._nodes: dict[str, GraphNodeExecutor] = {}
        self._order: list[str] = []
        self._children_by_parent: dict[str, tuple[str, ...]] = {}
        self._parents_by_child: dict[str, set[str]] = {}

        if nodes is not None:
            for spec, module in nodes:
                self.add_node(spec, module)

    @staticmethod
    def _validate_node_name(name: object) -> str:
        return _validate_graph_key(name, "GraphExecutor node name")

    @staticmethod
    def _is_valid_node_name(name: object) -> bool:
        return _is_valid_graph_key(name)

    def _add_executor(
        self,
        *,
        name: str,
        module_key: str,
        executor: GraphNodeExecutor,
        module: nn.Module,
    ) -> str:
        if not isinstance(module, nn.Module):
            raise TypeError("executor module must be an nn.Module.")

        name = self._validate_node_name(name)
        module_key = _validate_module_key(module_key)

        if name in self._nodes:
            raise ValueError(f"Duplicate node name: {name!r}")
        if module_key in self.modules_by_key:
            raise ValueError(f"Duplicate module_key: {module_key!r}")

        self.modules_by_key[module_key] = module
        self._nodes[name] = executor
        self._order.append(name)
        return name

    def add_node(self, spec: NodeSpec, module: nn.Module) -> str:
        if not isinstance(spec, NodeSpec):
            raise TypeError(f"add_node expects NodeSpec, got {type(spec)!r}")

        executor = NodeExecutor(spec)
        return self._add_executor(
            name=spec.name,
            module_key=executor.module_key,
            executor=executor,
            module=module,
        )

    def add_subgraph(self, spec: SubgraphSpec, module: nn.Module) -> str:
        if not isinstance(spec, SubgraphSpec):
            raise TypeError(
                f"add_subgraph expects SubgraphSpec, got {type(spec)!r}"
            )

        children = tuple(self._validate_node_name(child) for child in spec.children)
        for child in children:
            if child not in self._nodes:
                raise KeyError(f"Unknown child node: {child!r}")

        child_output_refs = [
            self.get_node(child).output_ref()
            for child in children
        ]
        executor = SubgraphExecutor(spec, child_output_refs)
        name = self._add_executor(
            name=spec.name,
            module_key=executor.module_key,
            executor=executor,
            module=module,
        )

        self._children_by_parent[name] = children
        for child in children:
            self._parents_by_child.setdefault(child, set()).add(name)
        return name

    def _remove_node_unchecked(self, name: str) -> None:
        node = self._nodes.pop(name)
        module_key = node.module_key

        if module_key in self.modules_by_key:
            del self.modules_by_key[module_key]

        try:
            self._order.remove(name)
        except ValueError:
            pass

        children = self._children_by_parent.pop(name, ())
        for child in children:
            parents = self._parents_by_child.get(child)
            if parents is None:
                continue
            parents.discard(name)
            if not parents:
                self._parents_by_child.pop(child, None)

        self._parents_by_child.pop(name, None)

    def remove_node(self, name: str, *, missing_ok: bool = False) -> None:
        name = self._validate_node_name(name)
        if name not in self._nodes:
            if missing_ok:
                return
            raise KeyError(f"Unknown node: {name!r}")

        parents = self._parents_by_child.get(name)
        if parents:
            raise ValueError(
                f"Cannot remove node {name!r}; it is referenced by parents: "
                f"{sorted(parents)!r}"
            )

        self._remove_node_unchecked(name)

    def _collect_subtree_postorder(self, name: str) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []

        def visit(node_name: str) -> None:
            if node_name in seen:
                return
            seen.add(node_name)
            for child_name in self._children_by_parent.get(node_name, ()):
                visit(child_name)
            ordered.append(node_name)

        visit(name)
        return tuple(ordered)

    def remove_subtree(self, name: str, *, missing_ok: bool = False) -> None:
        name = self._validate_node_name(name)
        if name not in self._nodes:
            if missing_ok:
                return
            raise KeyError(f"Unknown node: {name!r}")

        removal_order = self._collect_subtree_postorder(name)
        removal_set = set(removal_order)

        external_refs: dict[str, tuple[str, ...]] = {}
        for node_name in removal_order:
            external = tuple(
                sorted(self._parents_by_child.get(node_name, set()) - removal_set)
            )
            if external:
                external_refs[node_name] = external

        if external_refs:
            raise ValueError(
                "Cannot remove subtree with external parent references: "
                f"{external_refs!r}"
            )

        for node_name in removal_order:
            if node_name in self._nodes:
                self._remove_node_unchecked(node_name)

    def get_node(self, name: str) -> GraphNodeExecutor:
        name = self._validate_node_name(name)
        if name not in self._nodes:
            raise KeyError(f"Unknown node: {name!r}")
        return self._nodes[name]

    def has_node(self, name: object) -> bool:
        return self._is_valid_node_name(name) and name in self._nodes

    def node_names(self) -> tuple[str, ...]:
        return tuple(self._order)

    def child_names(self, name: str) -> tuple[str, ...]:
        name = self._validate_node_name(name)
        if name not in self._nodes:
            raise KeyError(f"Unknown node: {name!r}")
        return self._children_by_parent.get(name, ())

    def parent_names(self, name: str) -> tuple[str, ...]:
        name = self._validate_node_name(name)
        if name not in self._nodes:
            raise KeyError(f"Unknown node: {name!r}")
        return tuple(sorted(self._parents_by_child.get(name, set())))

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
