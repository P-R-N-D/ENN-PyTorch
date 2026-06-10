from __future__ import annotations

import pytest
import torch
from torch import nn

from enn_torch_dev.executor import (
    BranchBuilder,
    GraphBuilder,
    GraphExecutor,
    KVStore,
    KeyRef,
)


class _AddOne(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


class _AddBias(nn.Module):
    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return x + bias


class _ParamScale(nn.Module):
    def __init__(self, value: float = 2.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


def test_branch_builder_local_uses_default_input_key() -> None:
    builder = BranchBuilder.local(input_key="tile.x")

    returned = builder.add(
        name="local",
        module=_AddOne(),
        output_key="local.out",
    )
    graph = builder.build()
    store = KVStore({"tile.x": torch.tensor([1.0])})

    result = graph.run(store)

    assert returned is builder
    assert isinstance(graph, GraphExecutor)
    assert builder.role == "local"
    assert graph.get_node("local").spec.input_args == [KeyRef("tile.x")]
    assert result is store
    assert torch.equal(store.get("local.out"), torch.tensor([2.0]))


def test_branch_builder_global_helper_builds_graph() -> None:
    graph = (
        BranchBuilder.global_(input_key="x")
        .add(
            name="global",
            module=_AddOne(),
            output_key="global.out",
        )
        .build()
    )
    store = KVStore({"x": torch.tensor([2.0])})

    graph.run(store)

    assert graph.get_node("global").spec.input_args == [KeyRef("x")]
    assert torch.equal(store.get("global.out"), torch.tensor([3.0]))


def test_branch_builder_stream_helper_builds_graph() -> None:
    builder = BranchBuilder.stream(input_key="chunk.x")
    graph = (
        builder.add(
            name="step",
            module=_AddOne(),
            output_key="chunk.out",
        )
        .build()
    )
    store = KVStore({"chunk.x": torch.tensor([3.0])})

    graph.run(store)

    assert builder.role == "stream"
    assert graph.get_node("step").spec.input_args == [KeyRef("chunk.x")]
    assert torch.equal(store.get("chunk.out"), torch.tensor([4.0]))


def test_branch_builder_respects_explicit_inputs() -> None:
    graph = (
        BranchBuilder.local(input_key="tile.x")
        .add(
            name="local",
            module=_AddBias(),
            input_args=["manual.x"],
            input_kwargs={"bias": "bias"},
            output_key="local.out",
        )
        .build()
    )
    store = KVStore(
        {
            "manual.x": torch.tensor([2.0]),
            "bias": torch.tensor([5.0]),
        }
    )

    graph.run(store)

    node = graph.get_node("local")
    assert node.spec.input_args == [KeyRef("manual.x")]
    assert node.spec.input_kwargs["bias"] == KeyRef("bias")
    assert torch.equal(store.get("local.out"), torch.tensor([7.0]))


def test_branch_builder_registers_module_parameters() -> None:
    module = _ParamScale()
    graph = (
        BranchBuilder.local(input_key="tile.x")
        .add(
            name="scale",
            module=module,
            output_key="local.out",
        )
        .build()
    )

    assert list(graph.parameters()) == [module.weight]
    assert any(
        key.endswith("modules_by_key.scale.weight")
        for key in graph.state_dict()
    )


def test_branch_builder_accepts_existing_graph_builder() -> None:
    graph_builder = GraphBuilder()
    builder = BranchBuilder.local(
        input_key="x",
        graph_builder=graph_builder,
    )

    graph = (
        builder.add(
            name="node",
            module=_AddOne(),
            output_key="out",
        )
        .build()
    )
    store = KVStore({"x": torch.tensor([1.0])})

    graph.run(store)

    assert builder.graph_builder is graph_builder
    assert torch.equal(store.get("out"), torch.tensor([2.0]))


def test_branch_builder_rejects_invalid_constructor_inputs() -> None:
    with pytest.raises(ValueError, match="input_key"):
        BranchBuilder.local(input_key="")

    with pytest.raises(ValueError, match="role"):
        BranchBuilder(input_key="x", role="plain")

    with pytest.raises(TypeError, match="GraphBuilder"):
        BranchBuilder.local(
            input_key="x",
            graph_builder=object(),  # type: ignore[arg-type]
        )


def test_branch_builder_delegates_add_validation() -> None:
    with pytest.raises(TypeError, match="module must be an nn.Module"):
        BranchBuilder.local(input_key="x").add(
            name="bad",
            module=object(),  # type: ignore[arg-type]
            output_key="out",
        )


def test_branch_builder_delegates_graph_validation() -> None:
    builder = BranchBuilder.local(input_key="x")
    builder.add(name="node", module=nn.Identity(), output_key="x")
    builder.add(name="node", module=nn.Identity(), output_key="y")

    with pytest.raises(ValueError, match="Duplicate node name"):
        builder.build()
