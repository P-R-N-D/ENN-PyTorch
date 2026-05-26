from __future__ import annotations

import pytest
import torch
from torch import nn

from enn_torch_dev.executor import (
    GraphExecutor,
    KVStore,
    KeyRef,
    NodeSpec,
    SubgraphExecutor,
    SubgraphSpec,
)
from enn_torch_dev.nn import Reducer


class _Scale(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class _WeightedReducer(nn.Module):
    def forward(
        self,
        values: list[torch.Tensor],
        weight: torch.Tensor,
    ) -> torch.Tensor:
        return values[0] + values[1] * weight


def test_subgraph_executor_reduces_child_outputs() -> None:
    store = KVStore(
        {
            "a": torch.ones(2, 3),
            "b": torch.full((2, 3), 3.0),
        }
    )
    executor = SubgraphExecutor(
        SubgraphSpec(
            name="merge",
            children=["a", "b"],
            output_key="out",
        ),
        [KeyRef("a"), KeyRef("b")],
    )

    out = executor.run(store, Reducer(op="mean"))

    assert torch.allclose(out, torch.full((2, 3), 2.0))
    assert torch.allclose(store.get("out"), torch.full((2, 3), 2.0))


def test_subgraph_executor_resolves_input_kwargs() -> None:
    store = KVStore(
        {
            "a": torch.ones(2, 3),
            "b": torch.full((2, 3), 3.0),
            "w": torch.tensor(2.0),
        }
    )
    executor = SubgraphExecutor(
        SubgraphSpec(
            name="weighted",
            children=["a", "b"],
            input_kwargs={"weight": KeyRef("w")},
            output_key="out",
        ),
        [KeyRef("a"), KeyRef("b")],
    )

    out = executor.run(store, _WeightedReducer())

    assert torch.allclose(out, torch.full((2, 3), 7.0))


def test_graph_executor_runs_leaf_nodes_then_subgraph() -> None:
    x = torch.ones(2, 3)
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(name="a", input_args=[KeyRef("x")], output_key="a.out"),
        _Scale(1.0),
    )
    graph.add_node(
        NodeSpec(name="b", input_args=[KeyRef("x")], output_key="b.out"),
        _Scale(3.0),
    )
    graph.add_subgraph(
        SubgraphSpec(
            name="merge",
            children=["a", "b"],
            output_key="merge.out",
        ),
        Reducer(op="mean"),
    )
    store = KVStore({"x": x})

    graph.run(store)

    assert torch.allclose(store.get("a.out"), x)
    assert torch.allclose(store.get("b.out"), x * 3.0)
    assert torch.allclose(store.get("merge.out"), x * 2.0)


def test_graph_executor_runs_two_level_subgraph_tree() -> None:
    x = torch.ones(2, 3)
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(name="a", input_args=[KeyRef("x")], output_key="a.out"),
        _Scale(1.0),
    )
    graph.add_node(
        NodeSpec(name="b", input_args=[KeyRef("x")], output_key="b.out"),
        _Scale(3.0),
    )
    graph.add_subgraph(
        SubgraphSpec(
            name="ab",
            children=["a", "b"],
            output_key="ab.out",
        ),
        Reducer(op="mean"),
    )
    graph.add_node(
        NodeSpec(name="c", input_args=[KeyRef("x")], output_key="c.out"),
        _Scale(5.0),
    )
    graph.add_subgraph(
        SubgraphSpec(
            name="root",
            children=["ab", "c"],
            output_key="root.out",
        ),
        Reducer(op="mean"),
    )

    store = KVStore({"x": x})
    graph.run(store)

    assert torch.allclose(store.get("ab.out"), x * 2.0)
    assert torch.allclose(store.get("root.out"), x * 3.5)


def test_graph_executor_get_node_can_return_subgraph_executor() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="a", output_key="a.out"), nn.Identity())
    graph.add_node(NodeSpec(name="b", output_key="b.out"), nn.Identity())
    graph.add_subgraph(
        SubgraphSpec(name="merge", children=["a", "b"], output_key="out"),
        Reducer(op="mean"),
    )

    node = graph.get_node("merge")

    assert isinstance(node, SubgraphExecutor)
    assert node.children == ("a", "b")
    assert node.output_key == "out"


def test_graph_executor_child_and_parent_names() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="a", output_key="a.out"), nn.Identity())
    graph.add_node(NodeSpec(name="b", output_key="b.out"), nn.Identity())
    graph.add_subgraph(
        SubgraphSpec(name="merge", children=["a", "b"], output_key="out"),
        Reducer(op="mean"),
    )

    assert graph.child_names("merge") == ("a", "b")
    assert graph.parent_names("a") == ("merge",)
    assert graph.parent_names("b") == ("merge",)
    assert graph.child_names("a") == ()


def test_graph_executor_remove_node_removes_root_only_and_keeps_children() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="a", output_key="a.out"), nn.Identity())
    graph.add_node(NodeSpec(name="b", output_key="b.out"), nn.Identity())
    graph.add_subgraph(
        SubgraphSpec(name="merge", children=["a", "b"], output_key="out"),
        Reducer(op="mean"),
    )

    graph.remove_node("merge")

    assert graph.node_names() == ("a", "b")
    assert graph.has_node("a")
    assert graph.has_node("b")
    assert not graph.has_node("merge")
    assert graph.parent_names("a") == ()
    assert graph.parent_names("b") == ()


def test_graph_executor_remove_node_rejects_node_referenced_by_parent() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="a", output_key="a.out"), nn.Identity())
    graph.add_node(NodeSpec(name="b", output_key="b.out"), nn.Identity())
    graph.add_subgraph(
        SubgraphSpec(name="merge", children=["a", "b"], output_key="merge.out"),
        Reducer(op="mean"),
    )
    graph.add_subgraph(
        SubgraphSpec(name="root", children=["merge"], output_key="root.out"),
        Reducer(op="mean"),
    )

    with pytest.raises(ValueError, match="referenced by parents"):
        graph.remove_node("merge")


def test_graph_executor_remove_subtree_removes_descendants() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="a", output_key="a.out"), nn.Identity())
    graph.add_node(NodeSpec(name="b", output_key="b.out"), nn.Identity())
    graph.add_subgraph(
        SubgraphSpec(name="merge", children=["a", "b"], output_key="merge.out"),
        Reducer(op="mean"),
    )
    graph.add_node(NodeSpec(name="c", output_key="c.out"), nn.Identity())
    graph.add_subgraph(
        SubgraphSpec(name="root", children=["merge", "c"], output_key="root.out"),
        Reducer(op="mean"),
    )

    graph.remove_subtree("root")

    assert graph.node_names() == ()
    for name in ("a", "b", "merge", "c", "root"):
        assert not graph.has_node(name)


def test_graph_executor_remove_subtree_rejects_external_parent_reference() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="a", output_key="a.out"), nn.Identity())
    graph.add_node(NodeSpec(name="b", output_key="b.out"), nn.Identity())
    graph.add_node(NodeSpec(name="c", output_key="c.out"), nn.Identity())
    graph.add_subgraph(
        SubgraphSpec(name="merge", children=["a", "b"], output_key="merge.out"),
        Reducer(op="mean"),
    )
    graph.add_subgraph(
        SubgraphSpec(name="another", children=["a", "c"], output_key="another.out"),
        Reducer(op="mean"),
    )

    with pytest.raises(ValueError, match="external parent"):
        graph.remove_subtree("merge")


def test_graph_executor_add_subgraph_rejects_unknown_child() -> None:
    graph = GraphExecutor()

    with pytest.raises(KeyError, match="missing"):
        graph.add_subgraph(
            SubgraphSpec(name="merge", children=["missing"], output_key="out"),
            Reducer(op="mean"),
        )


def test_graph_executor_add_subgraph_rejects_duplicate_name_and_module_key() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="a", output_key="a.out"), nn.Identity())
    graph.add_node(NodeSpec(name="b", output_key="b.out"), nn.Identity())

    graph.add_subgraph(
        SubgraphSpec(name="merge", children=["a", "b"], output_key="out"),
        Reducer(op="mean"),
    )

    with pytest.raises(ValueError, match="Duplicate node name"):
        graph.add_subgraph(
            SubgraphSpec(
                name="merge",
                module_key="merge2",
                children=["a", "b"],
                output_key="out2",
            ),
            Reducer(op="mean"),
        )

    with pytest.raises(ValueError, match="Duplicate module_key"):
        graph.add_subgraph(
            SubgraphSpec(
                name="merge2",
                module_key="merge",
                children=["a", "b"],
                output_key="out2",
            ),
            Reducer(op="mean"),
        )


def test_subgraph_spec_validates_children_and_kwargs() -> None:
    with pytest.raises(ValueError, match="children"):
        SubgraphSpec(name="empty", children=[], output_key="out")

    with pytest.raises(TypeError, match="children"):
        SubgraphSpec(name="bad", children="abc", output_key="out")

    with pytest.raises(ValueError, match="duplicate child"):
        SubgraphSpec(name="dup", children=["a", "a"], output_key="out")

    with pytest.raises(TypeError, match="KeyRef"):
        SubgraphSpec(
            name="bad_kwargs",
            children=["a"],
            input_kwargs={"weight": object()},
            output_key="out",
        )


def test_graph_executor_state_dict_exposes_subgraph_module() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="a", output_key="a.out"), nn.Identity())
    graph.add_node(NodeSpec(name="b", output_key="b.out"), nn.Identity())
    graph.add_subgraph(
        SubgraphSpec(
            name="merge",
            module_key="merge_module",
            children=["a", "b"],
            output_key="out",
        ),
        nn.Linear(3, 3),
    )

    keys = set(graph.state_dict())

    assert "modules_by_key.merge_module.weight" in keys
    assert "modules_by_key.merge_module.bias" in keys
