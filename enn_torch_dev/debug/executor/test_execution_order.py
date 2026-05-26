from __future__ import annotations

import pytest
import torch
from torch import nn

from enn_torch_dev.executor import GraphExecutor, KVStore, KeyRef, NodeSpec, SubgraphSpec
from enn_torch_dev.nn import Reducer


class _AddOne(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


def test_execution_order_separate_from_registration_order() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="b", input_args=[KeyRef("a_out")], output_key="b_out"), _AddOne())
    graph.add_node(NodeSpec(name="a", input_args=[KeyRef("x")], output_key="a_out"), _AddOne())

    assert graph.node_names() == ("b", "a")
    assert graph.execution_order() == ("a", "b")


def test_external_input_key_is_not_treated_as_dependency() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="n1", input_args=[KeyRef("external")], output_key="k1"), _AddOne())
    graph.add_node(NodeSpec(name="n2", input_args=[KeyRef("external")], output_key="k2"), _AddOne())

    assert graph.execution_order() == ("n1", "n2")


def test_subgraph_and_nested_subgraph_execution_order() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="n1", input_args=[KeyRef("x")], output_key="k1"), nn.Identity())
    graph.add_node(NodeSpec(name="n2", input_args=[KeyRef("k1")], output_key="k2"), nn.Identity())
    graph.add_subgraph(
        SubgraphSpec(
            name="sg1",
            children=["n1", "n2"],
            input_kwargs={"x": KeyRef("x")},
            output_key="sg1_out",
        ),
        Reducer(),
    )
    graph.add_subgraph(
        SubgraphSpec(
            name="sg2",
            children=["sg1"],
            input_kwargs={"x": KeyRef("x")},
            output_key="sg2_out",
        ),
        Reducer(),
    )

    assert graph.execution_order() == ("n1", "n2", "sg1", "sg2")


def test_reject_duplicate_output_key() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="n1", input_args=[KeyRef("x")], output_key="dup"), nn.Identity())

    with pytest.raises(ValueError, match="Duplicate output_key"):
        graph.add_node(NodeSpec(name="n2", input_args=[KeyRef("x")], output_key="dup"), nn.Identity())


def test_cycle_detection() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="a", input_args=[KeyRef("b_out")], output_key="a_out"), nn.Identity())
    graph.add_node(NodeSpec(name="b", input_args=[KeyRef("a_out")], output_key="b_out"), nn.Identity())

    with pytest.raises(RuntimeError, match="Cycle detected"):
        graph.execution_order()


def test_remove_rejects_dataflow_dependents() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="a", input_args=[KeyRef("x")], output_key="a_out"), nn.Identity())
    graph.add_node(NodeSpec(name="b", input_args=[KeyRef("a_out")], output_key="b_out"), nn.Identity())

    with pytest.raises(ValueError, match="dependent"):
        graph.remove_node("a")


def test_remove_subtree_rejects_external_dataflow_dependents() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="a", input_args=[KeyRef("x")], output_key="a_out"), nn.Identity())
    graph.add_node(NodeSpec(name="b", input_args=[KeyRef("a_out")], output_key="b_out"), nn.Identity())
    graph.add_subgraph(
        SubgraphSpec(
            name="sg",
            children=["a", "b"],
            input_kwargs={"x": KeyRef("x")},
            output_key="sg_out",
        ),
        Reducer(),
    )
    graph.add_node(NodeSpec(name="outside", input_args=[KeyRef("a_out")], output_key="outside_out"), nn.Identity())

    with pytest.raises(ValueError, match="external parent/dependent"):
        graph.remove_subtree("sg")


def test_output_key_reusable_after_remove() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="a", input_args=[KeyRef("x")], output_key="shared"), nn.Identity())
    graph.remove_node("a")

    graph.add_node(NodeSpec(name="b", input_args=[KeyRef("x")], output_key="shared"), nn.Identity())

    store = KVStore({"x": torch.tensor([1.0])})
    graph.run(store)
    assert store.has("shared")


class _EchoOptional(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_optional_keyrefs_do_not_create_hard_dependencies_or_cycles() -> None:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="a",
            input_kwargs={
                "x": KeyRef("b_out", optional=True, default=torch.tensor([1.0]))
            },
            output_key="a_out",
        ),
        _EchoOptional(),
    )
    graph.add_node(
        NodeSpec(
            name="b",
            input_kwargs={
                "x": KeyRef("a_out", optional=True, default=torch.tensor([2.0]))
            },
            output_key="b_out",
        ),
        _EchoOptional(),
    )

    assert graph.execution_order() == ("a", "b")

    store = KVStore()
    graph.run(store)

    assert torch.equal(store.get("a_out"), torch.tensor([1.0]))
    assert torch.equal(store.get("b_out"), torch.tensor([1.0]))


def test_optional_keyrefs_do_not_block_node_removal() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="a", output_key="a_out"), nn.Identity())
    graph.add_node(
        NodeSpec(
            name="b",
            input_kwargs={
                "x": KeyRef("a_out", optional=True, default=torch.tensor([0.0]))
            },
            output_key="b_out",
        ),
        _EchoOptional(),
    )

    graph.remove_node("a")

    assert not graph.has_node("a")
    assert graph.has_node("b")


def test_remove_node_can_break_dataflow_cycle() -> None:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(name="a", input_args=[KeyRef("b_out")], output_key="a_out"),
        nn.Identity(),
    )
    graph.add_node(
        NodeSpec(name="b", input_args=[KeyRef("a_out")], output_key="b_out"),
        nn.Identity(),
    )

    with pytest.raises(RuntimeError, match="Cycle detected"):
        graph.execution_order()

    graph.remove_node("a")

    assert not graph.has_node("a")
    assert graph.execution_order() == ("b",)


def test_remove_subtree_can_break_single_node_dataflow_cycle() -> None:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(name="a", input_args=[KeyRef("b_out")], output_key="a_out"),
        nn.Identity(),
    )
    graph.add_node(
        NodeSpec(name="b", input_args=[KeyRef("a_out")], output_key="b_out"),
        nn.Identity(),
    )

    with pytest.raises(RuntimeError, match="Cycle detected"):
        graph.execution_order()

    graph.remove_subtree("a")

    assert not graph.has_node("a")
    assert graph.execution_order() == ("b",)


def test_remove_node_in_cycle_still_rejects_external_dependent() -> None:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(name="a", input_args=[KeyRef("b_out")], output_key="a_out"),
        nn.Identity(),
    )
    graph.add_node(
        NodeSpec(name="b", input_args=[KeyRef("a_out")], output_key="b_out"),
        nn.Identity(),
    )
    graph.add_node(
        NodeSpec(name="outside", input_args=[KeyRef("a_out")], output_key="outside_out"),
        nn.Identity(),
    )

    with pytest.raises(ValueError, match="dependent nodes"):
        graph.remove_node("a")


def test_remove_subtree_in_cycle_still_rejects_external_dependent() -> None:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(name="a", input_args=[KeyRef("b_out")], output_key="a_out"),
        nn.Identity(),
    )
    graph.add_node(
        NodeSpec(name="b", input_args=[KeyRef("a_out")], output_key="b_out"),
        nn.Identity(),
    )
    graph.add_node(
        NodeSpec(
            name="outside",
            input_args=[KeyRef("a_out")],
            output_key="outside_out",
        ),
        nn.Identity(),
    )

    with pytest.raises(ValueError, match="external parent/dependent"):
        graph.remove_subtree("a")
