from __future__ import annotations

import torch
from torch import nn

from enn_torch_dev.executor import GraphBuilder, KVStore, Model, ModelExecutionSpec


class _AddOne(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


class _AddBias(nn.Module):
    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return x + bias


class _Double(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0


def test_graph_builder_plain_graph_runs_through_public_model() -> None:
    graph = (
        GraphBuilder()
        .add(
            name="encode",
            module=_AddOne(),
            input_args=["x"],
            output_key="encoded",
        )
        .add(
            name="head",
            module=_AddBias(),
            input_args=["encoded"],
            input_kwargs={"bias": "bias"},
            output_key="logits",
        )
        .build()
    )
    model = Model.from_components(ModelExecutionSpec(), graph=graph)
    store = KVStore(
        {
            "x": torch.tensor([1.0]),
            "bias": torch.tensor([3.0]),
        }
    )

    result = model(store)

    assert result is store
    assert torch.equal(store.get("encoded"), torch.tensor([2.0]))
    assert torch.equal(store.get("logits"), torch.tensor([5.0]))


def test_graph_builder_tile_graph_runs_through_public_model() -> None:
    spec = ModelExecutionSpec(tile=True, tile_shape=(2,))
    tile_graph = (
        GraphBuilder()
        .add(
            name="tile_node",
            module=_Double(),
            input_args=["tile.x"],
            output_key="tile.out",
        )
        .build()
    )
    tile_pipeline = spec.create_tile_pipeline(
        tile_graph,
        input_key="x",
        tile_input_key="tile.x",
        output_name="tile_node",
        output_key="model.out",
    )
    model = Model.from_components(spec, tile_pipeline=tile_pipeline)
    x = torch.arange(4, dtype=torch.float32)
    store = KVStore({"x": x})

    result = model(store)

    assert torch.equal(result, x * 2.0)
    assert torch.equal(store.get("model.out"), result)


def test_graph_builder_stream_graph_runs_through_public_model() -> None:
    spec = ModelExecutionSpec(stateful=True)
    stream_graph = (
        GraphBuilder()
        .add(
            name="step",
            module=_Double(),
            input_args=["chunk.x"],
            output_key="chunk.out",
        )
        .build()
    )
    stream_pipeline = spec.create_stream_pipeline(
        stream_graph,
        chunk_input_key="chunk.x",
        output_name="step",
        outputs_key="stream.outputs",
    )
    model = Model.from_components(spec, stream_pipeline=stream_pipeline)
    store = KVStore()

    outputs = model(
        store,
        chunks=[torch.tensor([1.0]), torch.tensor([2.0])],
    )

    assert [out.item() for out in outputs] == [2.0, 4.0]
    assert store.get("stream.outputs") is outputs
