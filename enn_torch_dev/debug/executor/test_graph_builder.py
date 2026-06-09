from __future__ import annotations

import pytest
import torch
from torch import nn

from enn_torch_dev.executor import GraphBuilder, KVStore, KeyRef


class _AddOne(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


class _AddBias(nn.Module):
    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return x + bias


class _Split(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, x + 1.0


def test_graph_builder_builds_runnable_graph_from_string_refs() -> None:
    builder = GraphBuilder()

    returned = builder.add(
        name="encode",
        module=_AddOne(),
        input_args=["x"],
        output_key="encoded",
    )
    assert returned is builder

    builder.add(
        name="head",
        module=_AddBias(),
        input_args=["encoded"],
        input_kwargs={"bias": "bias"},
        output_key="logits",
    )

    graph = builder.build()
    store = KVStore(
        {
            "x": torch.tensor([1.0]),
            "bias": torch.tensor([3.0]),
        }
    )

    result = graph.run(store)

    assert result is store
    assert graph.node_names() == ("encode", "head")
    assert graph.execution_order() == ("encode", "head")
    assert torch.equal(store.get("encoded"), torch.tensor([2.0]))
    assert torch.equal(store.get("logits"), torch.tensor([5.0]))


def test_graph_builder_preserves_key_refs_for_optional_defaults() -> None:
    builder = GraphBuilder()
    builder.add(
        name="node",
        module=_AddBias(),
        input_args=[KeyRef("x")],
        input_kwargs={
            "bias": KeyRef("missing.bias", optional=True, default=torch.tensor([4.0])),
        },
        output_key="y",
    )
    graph = builder.build()
    store = KVStore({"x": torch.tensor([2.0])})

    graph.run(store)

    node = graph.get_node("node")
    assert node.spec.input_args == [KeyRef("x")]
    assert node.spec.input_kwargs["bias"].optional is True
    assert torch.equal(store.get("y"), torch.tensor([6.0]))


def test_graph_builder_passes_multi_output_keys_to_node_spec() -> None:
    graph = (
        GraphBuilder()
        .add(
            name="split",
            module=_Split(),
            input_args=["x"],
            output_key="left",
            output_keys=("left", "right"),
        )
        .build()
    )
    store = KVStore({"x": torch.tensor([2.0])})

    graph.run(store)

    assert torch.equal(store.get("left"), torch.tensor([2.0]))
    assert torch.equal(store.get("right"), torch.tensor([3.0]))


def test_graph_builder_rejects_non_module() -> None:
    with pytest.raises(TypeError, match="module must be an nn.Module"):
        GraphBuilder().add(
            name="bad",
            module=object(),  # type: ignore[arg-type]
            output_key="y",
        )


def test_graph_builder_rejects_string_input_args() -> None:
    with pytest.raises(TypeError, match="input_args"):
        GraphBuilder().add(
            name="bad",
            module=nn.Identity(),
            input_args="x",  # type: ignore[arg-type]
            output_key="y",
        )


def test_graph_builder_rejects_mapping_input_args() -> None:
    with pytest.raises(TypeError, match="input_args"):
        GraphBuilder().add(
            name="bad",
            module=nn.Identity(),
            input_args={"x": "store_key"},  # type: ignore[arg-type]
            output_key="y",
        )


def test_graph_builder_rejects_non_sequence_iterable_input_args() -> None:
    with pytest.raises(TypeError, match="input_args"):
        GraphBuilder().add(
            name="bad",
            module=nn.Identity(),
            input_args=iter(["x"]),  # type: ignore[arg-type]
            output_key="y",
        )


def test_graph_builder_rejects_non_mapping_input_kwargs() -> None:
    with pytest.raises(TypeError, match="input_kwargs"):
        GraphBuilder().add(
            name="bad",
            module=nn.Identity(),
            input_kwargs=[("x", "x")],  # type: ignore[arg-type]
            output_key="y",
        )


def test_graph_builder_build_delegates_graph_validation() -> None:
    builder = GraphBuilder()
    builder.add(name="node", module=nn.Identity(), output_key="x")
    builder.add(name="node", module=nn.Identity(), output_key="y")

    with pytest.raises(ValueError, match="Duplicate node name"):
        builder.build()
