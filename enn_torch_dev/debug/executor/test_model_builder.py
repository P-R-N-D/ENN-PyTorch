from __future__ import annotations

import pytest
import torch
from torch import nn

from enn_torch_dev.executor import (
    ExecutorModeSpec,
    GraphBuilder,
    KVStore,
    Model,
    ModelBuilder,
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


def test_model_builder_builds_plain_model_from_modules() -> None:
    builder = ModelBuilder()

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

    model = builder.build()
    store = KVStore(
        {
            "x": torch.tensor([1.0]),
            "bias": torch.tensor([3.0]),
        }
    )

    result = model(store)

    assert isinstance(model, Model)
    assert model.plan.mode == ExecutorModeSpec()
    assert result is store
    assert torch.equal(store.get("encoded"), torch.tensor([2.0]))
    assert torch.equal(store.get("logits"), torch.tensor([5.0]))


def test_model_builder_registers_graph_parameters() -> None:
    module = _ParamScale()
    model = (
        ModelBuilder()
        .add(
            name="scale",
            module=module,
            input_args=["x"],
            output_key="out",
        )
        .build()
    )

    assert list(model.parameters()) == [module.weight]
    assert any(
        key.endswith("modules_by_key.scale.weight")
        for key in model.state_dict()
    )

    model.to(dtype=torch.float64)
    assert module.weight.dtype == torch.float64

    store = KVStore({"x": torch.tensor([3.0], dtype=torch.float64)})
    result = model(store)

    assert result is store
    assert torch.equal(store.get("out"), torch.tensor([6.0], dtype=torch.float64))


def test_model_builder_accepts_existing_graph_builder() -> None:
    graph_builder = GraphBuilder()
    graph_builder.add(
        name="node",
        module=_AddOne(),
        input_args=["x"],
        output_key="out",
    )

    builder = ModelBuilder(graph_builder=graph_builder)
    model = builder.build()
    store = KVStore({"x": torch.tensor([1.0])})

    model(store)

    assert builder.graph_builder is graph_builder
    assert torch.equal(store.get("out"), torch.tensor([2.0]))


def test_model_builder_rejects_invalid_graph_builder() -> None:
    with pytest.raises(TypeError, match="GraphBuilder"):
        ModelBuilder(graph_builder=object())  # type: ignore[arg-type]


def test_model_builder_delegates_add_validation() -> None:
    with pytest.raises(TypeError, match="module must be an nn.Module"):
        ModelBuilder().add(
            name="bad",
            module=object(),  # type: ignore[arg-type]
            output_key="y",
        )


def test_model_builder_delegates_graph_validation_on_build() -> None:
    builder = ModelBuilder()
    builder.add(name="node", module=nn.Identity(), output_key="x")
    builder.add(name="node", module=nn.Identity(), output_key="y")

    with pytest.raises(ValueError, match="Duplicate node name"):
        builder.build()
