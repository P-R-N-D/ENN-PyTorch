from __future__ import annotations

import pytest
import torch
from torch import nn

from enn_torch_dev.executor import GraphExecutor, KVStore, KeyRef, NodeExecutor, NodeSpec


def test_graph_executor_runs_nodes_in_insertion_order() -> None:
    x = torch.randn(2, 4)
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(name="linear1", input_args=[KeyRef("x")], output_key="h"),
        nn.Linear(4, 8),
    )
    graph.add_node(
        NodeSpec(name="linear2", input_args=[KeyRef("h")], output_key="y"),
        nn.Linear(8, 3),
    )
    store = KVStore({"x": x})

    out_store = graph.run(store)

    assert out_store is store
    assert store.has("h")
    assert store.has("y")
    assert store.get("h").shape == (2, 8)
    assert store.get("y").shape == (2, 3)


def test_graph_executor_forward_delegates_to_run() -> None:
    x = torch.randn(2, 4)
    graph = GraphExecutor(
        [
            (
                NodeSpec(name="identity", input_args=[KeyRef("x")], output_key="y"),
                nn.Identity(),
            )
        ]
    )
    store = KVStore({"x": x})

    out_store = graph(store)

    assert out_store is store
    assert torch.equal(store.get("y"), x)


def test_graph_executor_get_node_returns_node_executor() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="identity", input_args=[KeyRef("x")], output_key="y"), nn.Identity())

    node = graph.get_node("identity")

    assert isinstance(node, NodeExecutor)
    assert node.spec.name == "identity"


def test_graph_executor_remove_node_removes_node_module_and_order() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="first", input_args=[KeyRef("x")], output_key="h"), nn.Identity())
    graph.add_node(NodeSpec(name="second", input_args=[KeyRef("h")], output_key="y"), nn.Identity())

    graph.remove_node("second")

    assert graph.node_names() == ("first",)
    assert not graph.has_node("second")
    with pytest.raises(KeyError, match="second"):
        graph.get_node("second")
    assert "second" not in graph.modules_by_key

    store = KVStore({"x": torch.randn(2, 3)})
    graph.run(store)

    assert store.has("h")
    assert not store.has("y")


def test_graph_executor_rejects_duplicate_node_name() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="node", output_key="x"), nn.Identity())

    with pytest.raises(ValueError, match="Duplicate node name"):
        graph.add_node(NodeSpec(name="node", module_key="node2", output_key="y"), nn.Identity())


def test_graph_executor_rejects_duplicate_module_key() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="node1", module_key="shared", output_key="x"), nn.Identity())

    with pytest.raises(ValueError, match="Duplicate module_key"):
        graph.add_node(NodeSpec(name="node2", module_key="shared", output_key="y"), nn.Identity())


def test_graph_executor_rejects_dotted_module_key() -> None:
    graph = GraphExecutor()

    with pytest.raises(ValueError, match="module_key"):
        graph.add_node(NodeSpec(name="node.with.dot", output_key="y"), nn.Identity())


def test_graph_executor_state_dict_exposes_registered_modules() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="identity", module_key="id1", output_key="y"), nn.Linear(4, 4))

    state = graph.state_dict()

    assert "modules_by_key.id1.weight" in state
    assert "modules_by_key.id1.bias" in state


def test_graph_executor_has_node_invalid_name_policy() -> None:
    graph = GraphExecutor()
    graph.add_node(NodeSpec(name="valid", output_key="y"), nn.Identity())

    assert graph.has_node("valid") is True
    assert graph.has_node("missing") is False
    assert graph.has_node(1) is False
    assert graph.has_node("") is False


def test_graph_executor_get_node_invalid_name_policy() -> None:
    graph = GraphExecutor()

    with pytest.raises(TypeError):
        graph.get_node(1)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        graph.get_node("   ")
