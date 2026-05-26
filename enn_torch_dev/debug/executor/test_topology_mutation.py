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


def _make_three_leaf_graph() -> GraphExecutor:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(name="a", input_args=[KeyRef("x")], output_key="a.out"),
        _Scale(1.0),
    )
    graph.add_node(
        NodeSpec(name="b", input_args=[KeyRef("x")], output_key="b.out"),
        _Scale(3.0),
    )
    graph.add_node(
        NodeSpec(name="c", input_args=[KeyRef("x")], output_key="c.out"),
        _Scale(5.0),
    )
    return graph


def _run_merge(graph: GraphExecutor) -> torch.Tensor:
    store = KVStore({"x": torch.ones(2, 3)})
    graph.run(store)
    return store.get("merge.out")


def test_set_subgraph_children_updates_topology_and_result() -> None:
    graph = _make_three_leaf_graph()
    graph.add_subgraph(
        SubgraphSpec(name="merge", children=["a", "b"], output_key="merge.out"),
        Reducer(op="mean"),
    )

    assert torch.allclose(_run_merge(graph), torch.full((2, 3), 2.0))

    updated = graph.set_subgraph_children("merge", ["a", "c"])

    assert updated == ("a", "c")
    assert graph.child_names("merge") == ("a", "c")
    assert graph.parent_names("b") == ()
    assert graph.parent_names("c") == ("merge",)
    assert torch.allclose(_run_merge(graph), torch.full((2, 3), 3.0))


def test_attach_child_updates_topology_and_result() -> None:
    graph = _make_three_leaf_graph()
    graph.add_subgraph(
        SubgraphSpec(name="merge", children=["a", "b"], output_key="merge.out"),
        Reducer(op="mean"),
    )

    updated = graph.attach_child("merge", "c")

    assert updated == ("a", "b", "c")
    assert graph.child_names("merge") == ("a", "b", "c")
    assert graph.parent_names("c") == ("merge",)
    assert torch.allclose(_run_merge(graph), torch.full((2, 3), 3.0))


def test_detach_child_updates_topology_and_result() -> None:
    graph = _make_three_leaf_graph()
    graph.add_subgraph(
        SubgraphSpec(name="merge", children=["a", "b", "c"], output_key="merge.out"),
        Reducer(op="mean"),
    )

    updated = graph.detach_child("merge", "b")

    assert updated == ("a", "c")
    assert graph.child_names("merge") == ("a", "c")
    assert graph.parent_names("b") == ()
    assert torch.allclose(_run_merge(graph), torch.full((2, 3), 3.0))


def test_detach_child_rejects_empty_subgraph() -> None:
    graph = _make_three_leaf_graph()
    graph.add_subgraph(
        SubgraphSpec(name="merge", children=["a"], output_key="merge.out"),
        Reducer(op="mean"),
    )

    with pytest.raises(ValueError, match="children"):
        graph.detach_child("merge", "a")

    assert graph.child_names("merge") == ("a",)
    assert graph.parent_names("a") == ("merge",)


def test_detach_child_missing_ok() -> None:
    graph = _make_three_leaf_graph()
    graph.add_subgraph(
        SubgraphSpec(name="merge", children=["a"], output_key="merge.out"),
        Reducer(op="mean"),
    )

    assert graph.detach_child("merge", "b", missing_ok=True) == ("a",)

    with pytest.raises(KeyError, match="does not contain"):
        graph.detach_child("merge", "b")


def test_set_subgraph_children_rejects_leaf_parent() -> None:
    graph = _make_three_leaf_graph()

    with pytest.raises(TypeError, match="SubgraphExecutor"):
        graph.set_subgraph_children("a", ["b"])


def test_set_subgraph_children_rejects_unknown_or_duplicate_child() -> None:
    graph = _make_three_leaf_graph()
    graph.add_subgraph(
        SubgraphSpec(name="merge", children=["a"], output_key="merge.out"),
        Reducer(op="mean"),
    )

    with pytest.raises(KeyError, match="missing"):
        graph.set_subgraph_children("merge", ["missing"])

    with pytest.raises(ValueError, match="duplicate child"):
        graph.set_subgraph_children("merge", ["a", "a"])


def test_attach_child_rejects_duplicate_child() -> None:
    graph = _make_three_leaf_graph()
    graph.add_subgraph(
        SubgraphSpec(name="merge", children=["a"], output_key="merge.out"),
        Reducer(op="mean"),
    )

    with pytest.raises(ValueError, match="duplicate child"):
        graph.attach_child("merge", "a")


def test_set_subgraph_children_rejects_structural_cycle() -> None:
    graph = _make_three_leaf_graph()
    graph.add_subgraph(
        SubgraphSpec(name="ab", children=["a", "b"], output_key="ab.out"),
        Reducer(op="mean"),
    )
    graph.add_subgraph(
        SubgraphSpec(name="root", children=["ab", "c"], output_key="root.out"),
        Reducer(op="mean"),
    )

    with pytest.raises(ValueError, match="structural cycle"):
        graph.attach_child("ab", "root")

    with pytest.raises(ValueError, match="structural cycle"):
        graph.set_subgraph_children("ab", ["a", "root"])


def test_topology_mutation_updates_execution_order() -> None:
    graph = _make_three_leaf_graph()
    graph.add_subgraph(
        SubgraphSpec(name="merge", children=["a"], output_key="merge.out"),
        Reducer(op="mean"),
    )

    assert graph.execution_order()[-1] == "merge"

    graph.set_subgraph_children("merge", ["c"])

    order = graph.execution_order()
    assert order.index("c") < order.index("merge")


def test_attach_child_rejects_leaf_parent() -> None:
    graph = _make_three_leaf_graph()

    with pytest.raises(TypeError, match="SubgraphExecutor"):
        graph.attach_child("a", "b")


def test_detach_child_rejects_leaf_parent() -> None:
    graph = _make_three_leaf_graph()

    with pytest.raises(TypeError, match="SubgraphExecutor"):
        graph.detach_child("a", "b")
