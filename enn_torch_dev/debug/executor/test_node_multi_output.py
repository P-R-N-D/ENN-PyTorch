from __future__ import annotations

import pytest
import torch
from torch import nn

from enn_torch_dev.executor import GraphExecutor, KVStore, KeyRef, NodeSpec
from enn_torch_dev.nn import RecurrentContextHead


class _Pair(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x + 1.0, x + 2.0


class _Scalar(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


class _Triple(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return x, x + 1.0, x + 2.0


def test_node_multi_output_stores_each_key() -> None:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="pair",
            input_args=[KeyRef("x")],
            output_key="pair.out",
            output_keys=("pair.out", "pair.state"),
        ),
        _Pair(),
    )
    store = KVStore({"x": torch.tensor([1.0])})

    graph.run(store)

    assert torch.equal(store.get("pair.out"), torch.tensor([2.0]))
    assert torch.equal(store.get("pair.state"), torch.tensor([3.0]))
    assert graph.output_key("pair") == "pair.out"
    assert graph.output_keys(["pair"]) == ("pair.out",)


def test_node_single_output_still_stores_whole_output() -> None:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(name="scalar", input_args=[KeyRef("x")], output_key="scalar.out"),
        _Scalar(),
    )
    store = KVStore({"x": torch.tensor([1.0])})

    graph.run(store)

    assert torch.equal(store.get("scalar.out"), torch.tensor([2.0]))


def test_node_spec_validates_output_keys() -> None:
    with pytest.raises(TypeError, match="output_keys"):
        NodeSpec(name="bad", output_key="bad.out", output_keys="bad.out")

    with pytest.raises(ValueError, match="empty"):
        NodeSpec(name="bad", output_key="bad.out", output_keys=())

    with pytest.raises(ValueError, match="duplicate"):
        NodeSpec(
            name="bad",
            output_key="bad.out",
            output_keys=("bad.out", "bad.out"),
        )

    with pytest.raises(ValueError, match="first entry"):
        NodeSpec(
            name="bad",
            output_key="bad.out",
            output_keys=("other.out", "bad.state"),
        )


def test_node_multi_output_rejects_non_sequence_output() -> None:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="bad",
            input_args=[KeyRef("x")],
            output_key="bad.out",
            output_keys=("bad.out", "bad.state"),
        ),
        _Scalar(),
    )

    with pytest.raises(TypeError, match="tuple or list"):
        graph.run(KVStore({"x": torch.tensor([1.0])}))


def test_node_multi_output_rejects_output_length_mismatch() -> None:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="bad",
            input_args=[KeyRef("x")],
            output_key="bad.out",
            output_keys=("bad.out", "bad.state"),
        ),
        _Triple(),
    )

    with pytest.raises(ValueError, match="expected 2 outputs"):
        graph.run(KVStore({"x": torch.tensor([1.0])}))


def test_graph_rejects_duplicate_secondary_output_key() -> None:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="pair",
            output_key="pair.out",
            output_keys=("pair.out", "shared.out"),
        ),
        _Pair(),
    )

    with pytest.raises(ValueError, match="Duplicate output_key"):
        graph.add_node(NodeSpec(name="other", output_key="shared.out"), nn.Identity())


def test_graph_dependency_tracks_secondary_output_key() -> None:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="consumer",
            input_args=[KeyRef("pair.state")],
            output_key="consumer.out",
        ),
        nn.Identity(),
    )
    graph.add_node(
        NodeSpec(
            name="pair",
            input_args=[KeyRef("x")],
            output_key="pair.out",
            output_keys=("pair.out", "pair.state"),
        ),
        _Pair(),
    )

    assert graph.execution_order() == ("pair", "consumer")
    assert graph.dependency_names("consumer") == ("pair",)

    store = KVStore({"x": torch.tensor([1.0])})
    graph.run(store)
    assert torch.equal(store.get("consumer.out"), torch.tensor([3.0]))


def test_remove_node_cleans_secondary_output_key_index() -> None:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(name="pair", output_key="pair.out", output_keys=("pair.out", "pair.state")),
        _Pair(),
    )

    graph.remove_node("pair")
    graph.add_node(NodeSpec(name="other", output_key="pair.state"), nn.Identity())

    assert graph.output_key("other") == "pair.state"


def test_recurrent_context_head_can_route_output_and_state() -> None:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="ctx",
            input_args=[KeyRef("x")],
            input_kwargs={
                "state": KeyRef("state.in", optional=True, default=None),
                "return_state": KeyRef("return_state"),
            },
            output_key="ctx.out",
            output_keys=("ctx.out", "ctx.state"),
        ),
        RecurrentContextHead(input_dim=4, hidden_dim=3),
    )
    store = KVStore(
        {
            "x": torch.randn(2, 5, 4),
            "return_state": True,
        }
    )

    graph.run(store)

    assert store.get("ctx.out").shape == (2, 5, 4)
    assert store.get("ctx.state").shape == (1, 2, 3)
