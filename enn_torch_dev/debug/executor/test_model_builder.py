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


class _Double(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0


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


def test_model_builder_build_tile_runs_tiled_model() -> None:
    model = (
        ModelBuilder()
        .add(
            name="local",
            module=_Double(),
            input_args=["tile.x"],
            output_key="local.out",
        )
        .build_tile(
            tile_shape=(2,),
            input_key="x",
            tile_input_key="tile.x",
            output_name="local",
            output_key="tile.out",
        )
    )
    x = torch.arange(4, dtype=torch.float32)
    store = KVStore({"x": x})

    result = model(store)

    assert model.plan.mode == ExecutorModeSpec(tile=True)
    assert model.spec.tile_shape == (2,)
    assert torch.equal(result, x * 2.0)
    assert torch.equal(store.get("tile.out"), result)


def test_model_builder_build_tile_preserves_tile_policy_settings() -> None:
    model = (
        ModelBuilder()
        .add(
            name="local",
            module=_Double(),
            input_args=["tile.x"],
            output_key="local.out",
        )
        .build_tile(
            tile_shape=(2,),
            tile_stride=(1,),
            tile_dims=(0,),
            input_key="x",
            tile_input_key="tile.x",
            output_name="local",
        )
    )

    assert model.spec.tile_shape == (2,)
    assert model.spec.tile_stride == (1,)
    assert model.spec.tile_dims == (0,)
    assert model.plan.tile_pipeline is not None
    assert model.plan.tile_pipeline.tile_policy.tile_shape == (2,)
    assert model.plan.tile_pipeline.tile_policy.stride == (1,)
    assert model.plan.tile_pipeline.tile_policy.dims == (0,)


def test_model_builder_build_tile_registers_graph_parameters() -> None:
    module = _ParamScale()
    model = (
        ModelBuilder()
        .add(
            name="scale",
            module=module,
            input_args=["tile.x"],
            output_key="tile.out",
        )
        .build_tile(
            tile_shape=(2,),
            input_key="x",
            tile_input_key="tile.x",
            output_name="scale",
            output_key="model.out",
        )
    )

    assert list(model.parameters()) == [module.weight]
    assert any(
        key.endswith("modules_by_key.scale.weight")
        for key in model.state_dict()
    )


def test_model_builder_build_stream_runs_stateful_model() -> None:
    model = (
        ModelBuilder()
        .add(
            name="step",
            module=_Double(),
            input_args=["chunk.x"],
            output_key="chunk.out",
        )
        .build_stream(
            chunk_input_key="chunk.x",
            output_name="step",
            outputs_key="stream.outputs",
        )
    )
    store = KVStore()

    outputs = model(
        store,
        chunks=[torch.tensor([1.0]), torch.tensor([2.0])],
    )

    assert model.plan.mode == ExecutorModeSpec(stream=True)
    assert model.spec.stateful is True
    assert [out.item() for out in outputs] == [2.0, 4.0]
    assert store.get("stream.outputs") is outputs


def test_model_builder_build_stream_preserves_stream_spec_settings() -> None:
    model = (
        ModelBuilder()
        .add(
            name="step",
            module=_Double(),
            input_args=["chunk.x"],
            output_key="chunk.out",
        )
        .build_stream(
            chunk_input_key="chunk.x",
            output_name="step",
            output_by="key",
            chunk_index_key="chunk.index",
            outputs_key="stream.outputs",
            state_detach=True,
            state_clone=True,
            reset_state=True,
        )
    )

    assert model.plan.stream_pipeline is not None
    stream_spec = model.plan.stream_pipeline.spec
    assert stream_spec.chunk_input_key == "chunk.x"
    assert stream_spec.output_name == "step"
    assert stream_spec.output_by == "key"
    assert stream_spec.chunk_index_key == "chunk.index"
    assert stream_spec.outputs_key == "stream.outputs"
    assert stream_spec.state_detach is True
    assert stream_spec.state_clone is True
    assert stream_spec.reset_state is True


def test_model_builder_build_stream_registers_graph_parameters() -> None:
    module = _ParamScale()
    model = (
        ModelBuilder()
        .add(
            name="scale",
            module=module,
            input_args=["chunk.x"],
            output_key="chunk.out",
        )
        .build_stream(chunk_input_key="chunk.x", output_name="scale")
    )

    assert list(model.parameters()) == [module.weight]
    assert any(
        key.endswith("modules_by_key.scale.weight")
        for key in model.state_dict()
    )


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


def test_model_builder_build_tile_delegates_graph_validation() -> None:
    builder = ModelBuilder()
    builder.add(name="node", module=nn.Identity(), output_key="x")
    builder.add(name="node", module=nn.Identity(), output_key="y")

    with pytest.raises(ValueError, match="Duplicate node name"):
        builder.build_tile(
            tile_shape=(2,),
            input_key="x",
            tile_input_key="tile.x",
            output_name="node",
        )


def test_model_builder_build_stream_delegates_graph_validation() -> None:
    builder = ModelBuilder()
    builder.add(name="node", module=nn.Identity(), output_key="x")
    builder.add(name="node", module=nn.Identity(), output_key="y")

    with pytest.raises(ValueError, match="Duplicate node name"):
        builder.build_stream(
            chunk_input_key="chunk.x",
            output_name="node",
        )


def test_model_builder_build_stream_delegates_graph_executor_output_validation() -> None:
    builder = ModelBuilder()
    builder.add(
        name="node",
        module=nn.Identity(),
        input_args=["chunk.x"],
        output_key="x",
    )

    with pytest.raises(KeyError, match="missing"):
        builder.build_stream(
            chunk_input_key="chunk.x",
            output_name="missing",
        )
