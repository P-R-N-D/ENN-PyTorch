from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from .node import NodeExecutor, NodeSpec
from .schema import KeyRef
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
        self._producer_by_output_key: dict[str, str] = {}

        if nodes is not None:
            for spec, module in nodes:
                self.add_node(spec, module)

    @staticmethod
    def _validate_node_name(name: object) -> str:
        return _validate_graph_key(name, "GraphExecutor node name")

    @staticmethod
    def _is_valid_node_name(name: object) -> bool:
        return _is_valid_graph_key(name)

    def _has_structural_path(self, start: str, target: str) -> bool:
        if start == target:
            return True

        seen: set[str] = set()
        stack = [start]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            for child in self._children_by_parent.get(current, ()):
                if child == target:
                    return True
                if child not in seen:
                    stack.append(child)
        return False

    def _normalize_subgraph_children(
        self,
        parent: str,
        children: Sequence[str],
    ) -> tuple[str, ...]:
        parent = self._validate_node_name(parent)
        if isinstance(children, (str, bytes, bytearray)) or children is None:
            raise TypeError("subgraph children must be a sequence of node names.")

        values = list(children)
        if not values:
            raise ValueError("subgraph children must not be empty.")

        normalized: list[str] = []
        seen: set[str] = set()
        for child in values:
            child_name = self._validate_node_name(child)
            if child_name == parent:
                raise ValueError("Subgraph cannot contain itself as a child.")
            if child_name in seen:
                raise ValueError(f"duplicate child node: {child_name!r}")
            if child_name not in self._nodes:
                raise KeyError(f"Unknown child node: {child_name!r}")
            if self._has_structural_path(child_name, parent):
                raise ValueError(
                    f"Adding child {child_name!r} to subgraph {parent!r} "
                    "would create a structural cycle."
                )
            seen.add(child_name)
            normalized.append(child_name)
        return tuple(normalized)

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
        output_key = _validate_graph_key(executor.output_key, "GraphExecutor output_key")

        if name in self._nodes:
            raise ValueError(f"Duplicate node name: {name!r}")
        if module_key in self.modules_by_key:
            raise ValueError(f"Duplicate module_key: {module_key!r}")

        producer = self._producer_by_output_key.get(output_key)
        if producer is not None:
            raise ValueError(
                f"Duplicate output_key: {output_key!r} already produced by node {producer!r}"
            )

        self.modules_by_key[module_key] = module
        self._nodes[name] = executor
        self._order.append(name)
        self._producer_by_output_key[output_key] = name
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

        children = self._normalize_subgraph_children(spec.name, spec.children)

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

    def _get_subgraph_executor(self, name: str) -> SubgraphExecutor:
        node = self.get_node(name)
        if not isinstance(node, SubgraphExecutor):
            raise TypeError(f"Node {name!r} is not a SubgraphExecutor.")
        return node

    def set_subgraph_children(
        self,
        name: str,
        children: Sequence[str],
    ) -> tuple[str, ...]:
        name = self._validate_node_name(name)
        node = self._get_subgraph_executor(name)
        normalized = self._normalize_subgraph_children(name, children)
        child_output_refs = [
            self.get_node(child).output_ref()
            for child in normalized
        ]

        old_children = self._children_by_parent.get(name, ())
        for child in old_children:
            parents = self._parents_by_child.get(child)
            if parents is None:
                continue
            parents.discard(name)
            if not parents:
                self._parents_by_child.pop(child, None)

        self._children_by_parent[name] = normalized
        for child in normalized:
            self._parents_by_child.setdefault(child, set()).add(name)

        node.set_children(normalized, child_output_refs)
        return normalized

    def attach_child(self, parent: str, child: str) -> tuple[str, ...]:
        parent = self._validate_node_name(parent)
        self._get_subgraph_executor(parent)
        current = list(self.child_names(parent))
        current.append(self._validate_node_name(child))
        return self.set_subgraph_children(parent, current)

    def detach_child(
        self,
        parent: str,
        child: str,
        *,
        missing_ok: bool = False,
    ) -> tuple[str, ...]:
        parent = self._validate_node_name(parent)
        self._get_subgraph_executor(parent)
        child = self._validate_node_name(child)
        current = list(self.child_names(parent))

        if child not in current:
            if missing_ok:
                return tuple(current)
            raise KeyError(f"Subgraph {parent!r} does not contain child {child!r}.")

        updated = [name for name in current if name != child]
        if not updated:
            raise ValueError("Subgraph children must not be empty.")

        return self.set_subgraph_children(parent, updated)

    def _input_refs_for_node(self, name: str) -> tuple[KeyRef, ...]:
        node = self._nodes[name]
        refs: list[KeyRef] = []

        if isinstance(node, NodeExecutor):
            refs.extend(node.spec.input_args)
            refs.extend(node.spec.input_kwargs.values())
        elif isinstance(node, SubgraphExecutor):
            refs.extend(node.spec.input_kwargs.values())
        else:
            raise TypeError(f"Unsupported graph node executor: {type(node)!r}")

        return tuple(refs)

    def _dependencies_for_node(self, name: str) -> set[str]:
        if name not in self._nodes:
            raise KeyError(f"Unknown node: {name!r}")

        deps: set[str] = set()
        for child in self._children_by_parent.get(name, ()):  # structural dep
            if child not in self._nodes:
                raise RuntimeError(
                    f"Subgraph {name!r} references missing child node: {child!r}"
                )
            deps.add(child)

        for ref in self._input_refs_for_node(name):
            if ref.optional:
                continue
            producer = self._producer_by_output_key.get(ref.key)
            if producer is not None:
                deps.add(producer)

        return deps

    def _dependencies_by_node(self) -> dict[str, set[str]]:
        return {
            name: self._dependencies_for_node(name)
            for name in self._order
            if name in self._nodes
        }

    def _cycle_node_names(
        self,
        deps: dict[str, set[str]] | None = None,
    ) -> set[str]:
        work = {
            name: set(node_deps)
            for name, node_deps in (deps or self._dependencies_by_node()).items()
        }
        remaining = set(work)
        ready = [
            name
            for name in self._order
            if name in remaining and not work[name]
        ]

        while ready:
            name = ready.pop(0)
            if name not in remaining:
                continue

            remaining.remove(name)

            for candidate in self._order:
                if candidate not in remaining:
                    continue
                candidate_deps = work[candidate]
                if name not in candidate_deps:
                    continue
                candidate_deps.remove(name)
                if not candidate_deps and candidate not in ready:
                    ready.append(candidate)

        return remaining

    def execution_order(self) -> tuple[str, ...]:
        deps = self._dependencies_by_node()
        remaining = set(deps)
        ready = [name for name in self._order if name in remaining and not deps[name]]
        ordered: list[str] = []

        while ready:
            name = ready.pop(0)
            if name not in remaining:
                continue

            ordered.append(name)
            remaining.remove(name)

            for candidate in self._order:
                if candidate not in remaining:
                    continue
                candidate_deps = deps[candidate]
                if name not in candidate_deps:
                    continue
                candidate_deps.remove(name)
                if not candidate_deps and candidate not in ready:
                    ready.append(candidate)

        if remaining:
            unresolved = {name: tuple(sorted(deps[name])) for name in sorted(remaining)}
            raise RuntimeError(
                "Cycle detected in graph execution dependencies: "
                f"{unresolved!r}"
            )

        return tuple(ordered)

    def _dependent_names(self, name: str) -> tuple[str, ...]:
        deps = self._dependencies_by_node()
        return tuple(
            sorted(
                candidate
                for candidate, candidate_deps in deps.items()
                if candidate != name and name in candidate_deps
            )
        )

    def _remove_node_unchecked(self, name: str) -> None:
        node = self._nodes.pop(name)
        module_key = node.module_key
        output_key = node.output_key

        if module_key in self.modules_by_key:
            del self.modules_by_key[module_key]

        if self._producer_by_output_key.get(output_key) == name:
            del self._producer_by_output_key[output_key]

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

        structural_parents = tuple(sorted(self._parents_by_child.get(name, set())))
        if structural_parents:
            raise ValueError(
                f"Cannot remove node {name!r}; it is referenced by "
                f"parents: {list(structural_parents)!r}"
            )

        dependents = self._dependent_names(name)
        if dependents:
            deps = self._dependencies_by_node()
            cycle_nodes = self._cycle_node_names(deps)

            def _reachable(src: str, dst: str) -> bool:
                if src == dst:
                    return True
                seen: set[str] = set()
                stack = [src]
                while stack:
                    current = stack.pop()
                    if current in seen:
                        continue
                    seen.add(current)
                    for nxt in deps.get(current, set()):
                        if nxt == dst:
                            return True
                        if nxt not in seen:
                            stack.append(nxt)
                return False

            blocking_dependents = tuple(
                dependent
                for dependent in dependents
                if (
                    dependent not in cycle_nodes
                    or not (_reachable(name, dependent) and _reachable(dependent, name))
                )
            )
            if name not in cycle_nodes or blocking_dependents:
                raise ValueError(
                    f"Cannot remove node {name!r}; it is referenced by "
                    f"dependent nodes: {list(dependents)!r}"
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
        deps = self._dependencies_by_node()
        cycle_nodes = self._cycle_node_names(deps)
        allow_single_node_cycle_repair = len(removal_order) == 1 and name in cycle_nodes

        external_refs: dict[str, tuple[str, ...]] = {}
        for node_name in removal_order:
            external_candidates = {
                candidate
                for candidate, candidate_deps in deps.items()
                if candidate not in removal_set and node_name in candidate_deps
            }
            structural_external = (
                self._parents_by_child.get(node_name, set()) - removal_set
            )

            if allow_single_node_cycle_repair:
                def _reachable(src: str, dst: str) -> bool:
                    if src == dst:
                        return True
                    seen: set[str] = set()
                    stack = [src]
                    while stack:
                        current = stack.pop()
                        if current in seen:
                            continue
                        seen.add(current)
                        for nxt in deps.get(current, set()):
                            if nxt == dst:
                                return True
                            if nxt not in seen:
                                stack.append(nxt)
                    return False

                dataflow_external = external_candidates - structural_external
                external_candidates = set(structural_external)
                external_candidates.update(
                    candidate
                    for candidate in dataflow_external
                    if not (
                        _reachable(node_name, candidate)
                        and _reachable(candidate, node_name)
                    )
                )

            external = tuple(sorted(external_candidates))
            if external:
                external_refs[node_name] = external

        if external_refs:
            raise ValueError(
                "Cannot remove subtree with external parent/dependent references: "
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

        for name in self.execution_order():
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
