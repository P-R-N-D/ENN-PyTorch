from __future__ import annotations

import pytest
import torch
from torch import nn

from enn_torch_dev.executor import GraphExecutor, KVStore, KeyRef, NodeSpec, SubgraphSpec
from enn_torch_dev.nn import Reducer


class _Scale(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def _make_graph() -> GraphExecutor:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(name="a", input_args=[KeyRef("x")], output_key="a.out"),
        _Scale(1.0),
    )
    graph.add_node(
        NodeSpec(name="b", input_args=[KeyRef("a.out")], output_key="b.out"),
        _Scale(3.0),
    )
    graph.add_subgraph(
        SubgraphSpec(name="merge", children=["a", "b"], output_key="merge.out"),
        Reducer(op="mean"),
    )
    graph.add_node(
        NodeSpec(name="c", input_args=[KeyRef("x")], output_key="c.out"),
        _Scale(5.0),
    )
    graph.add_subgraph(
        SubgraphSpec(name="root", children=["merge", "c"], output_key="root.out"),
        Reducer(op="mean"),
    )
    return graph


def test_dependency_and_dependent_names() -> None:
    graph = _make_graph()

    assert graph.dependency_names("b") == ("a",)
    assert graph.dependency_names("merge") == ("a", "b")
    assert graph.dependency_names("root") == ("merge", "c")
    assert graph.dependent_names("a") == ("b", "merge")
    assert graph.dependent_names("merge") == ("root",)


def test_root_and_leaf_names() -> None:
    graph = _make_graph()

    assert graph.root_names() == ("root",)
    assert graph.leaf_names() == ("a", "b", "c")


def test_output_key_and_output_keys() -> None:
    graph = _make_graph()

    assert graph.output_key("root") == "root.out"
    assert graph.output_keys() == ("root.out",)
    assert graph.output_keys(["a", "root"]) == ("a.out", "root.out")


def test_collect_outputs_by_node_and_key() -> None:
    graph = _make_graph()
    store = KVStore({"x": torch.ones(2, 3)})

    graph.run(store)

    by_node = graph.collect_outputs(store)
    by_key = graph.collect_outputs(store, by="key")

    assert set(by_node) == {"root"}
    assert set(by_key) == {"root.out"}
    assert torch.allclose(by_node["root"], torch.full((2, 3), 3.5))
    assert torch.allclose(by_key["root.out"], torch.full((2, 3), 3.5))


def test_collect_outputs_explicit_names() -> None:
    graph = _make_graph()
    store = KVStore({"x": torch.ones(2, 3)})

    graph.run(store)

    outputs = graph.collect_outputs(store, ["a", "merge"])

    assert set(outputs) == {"a", "merge"}
    assert torch.allclose(outputs["a"], torch.ones(2, 3))
    assert torch.allclose(outputs["merge"], torch.full((2, 3), 2.0))


def test_collect_outputs_missing_store_value_raises() -> None:
    graph = _make_graph()
    store = KVStore()

    with pytest.raises(KeyError, match="root.out"):
        graph.collect_outputs(store)


def test_collect_outputs_rejects_invalid_by() -> None:
    graph = _make_graph()

    with pytest.raises(ValueError, match="node"):
        graph.collect_outputs(KVStore(), by="invalid")


def test_output_keys_rejects_string_names() -> None:
    graph = _make_graph()

    with pytest.raises(TypeError, match="sequence"):
        graph.output_keys("root")


def test_validate_passes_for_valid_graph() -> None:
    graph = _make_graph()

    graph.validate()


def test_validate_detects_output_producer_index_mismatch() -> None:
    graph = _make_graph()
    graph._producer_by_output_key["a.out"] = "wrong"

    with pytest.raises(RuntimeError, match="producer index"):
        graph.validate()


def test_validate_detects_parent_index_mismatch() -> None:
    graph = _make_graph()
    graph._parents_by_child["a"] = set()

    with pytest.raises(RuntimeError, match="parent index"):
        graph.validate()


def test_validate_detects_children_index_mismatch() -> None:
    graph = _make_graph()
    graph._children_by_parent["merge"] = ("a",)

    with pytest.raises(RuntimeError, match="children index"):
        graph.validate()


def test_validate_detects_child_output_ref_mismatch() -> None:
    graph = _make_graph()
    merge = graph.get_node("merge")
    merge._child_output_refs = (KeyRef("wrong.out"), KeyRef("b.out"))

    with pytest.raises(RuntimeError, match="child output references"):
        graph.validate()


def test_validate_detects_missing_children_index_for_subgraph() -> None:
    graph = _make_graph()
    graph._children_by_parent.pop("merge")
    for child in ("a", "b"):
        parents = graph._parents_by_child.get(child)
        if parents is not None:
            parents.discard("merge")
            if not parents:
                graph._parents_by_child.pop(child, None)

    with pytest.raises(RuntimeError, match="children index is missing"):
        graph.validate()


def test_validate_detects_duplicate_module_key() -> None:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(name="a", module_key="shared", output_key="a.out"),
        nn.Identity(),
    )
    graph.add_node(
        NodeSpec(name="b", module_key="other", output_key="b.out"),
        nn.Identity(),
    )
    graph.get_node("b").spec.module_key = "shared"

    with pytest.raises(RuntimeError, match="duplicate module_key"):
        graph.validate()


def test_validate_detects_duplicate_children_in_subgraph() -> None:
    graph = _make_graph()
    merge = graph.get_node("merge")
    merge.spec.children = ["a", "a"]
    merge._child_output_refs = (KeyRef("a.out"), KeyRef("a.out"))
    graph._children_by_parent["merge"] = ("a", "a")
    parents = graph._parents_by_child.get("b")
    if parents is not None:
        parents.discard("merge")
        if not parents:
            graph._parents_by_child.pop("b", None)

    with pytest.raises(RuntimeError, match="duplicate children"):
        graph.validate()
