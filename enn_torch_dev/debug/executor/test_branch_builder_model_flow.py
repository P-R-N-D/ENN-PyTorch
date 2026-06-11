from __future__ import annotations

import torch
from torch import nn

from enn_torch_dev.executor import (
    BranchBuilder,
    ExecutorModeSpec,
    KVStore,
    Model,
    ModelExecutionSpec,
)
from enn_torch_dev.nn import LocalGlobalFusion


class _AddBias(nn.Module):
    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return x + bias


class _Double(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0


def test_branch_builder_local_graph_runs_through_tile_model() -> None:
    local_graph = (
        BranchBuilder.local(input_key="tile.x")
        .add(
            name="local",
            module=_Double(),
            output_key="local.out",
        )
        .build()
    )
    spec = ModelExecutionSpec(tile=True, tile_shape=(2,))
    tile_pipeline = spec.create_tile_pipeline(
        local_graph,
        input_key="x",
        tile_input_key="tile.x",
        output_name="local",
        output_key="tile.out",
    )
    model = Model.from_components(spec, tile_pipeline=tile_pipeline)
    x = torch.arange(4, dtype=torch.float32)
    store = KVStore({"x": x})

    result = model(store)

    assert isinstance(model, Model)
    assert model.plan.mode == ExecutorModeSpec(tile=True)
    assert torch.equal(result, x * 2.0)
    assert torch.equal(store.get("tile.out"), result)


def test_branch_builder_local_and_global_graphs_run_through_global_local_model() -> None:
    local_graph = (
        BranchBuilder.local(input_key="tile.x")
        .add(
            name="local",
            module=_AddBias(),
            input_kwargs={"bias": "local.bias"},
            output_key="local.out",
        )
        .build()
    )
    global_graph = (
        BranchBuilder.global_(input_key="x")
        .add(
            name="global",
            module=_AddBias(),
            input_kwargs={"bias": "global.bias"},
            output_key="global.out",
        )
        .build()
    )
    spec = ModelExecutionSpec(
        context="global_local",
        tile=True,
        tile_shape=(2,),
    )
    tile_pipeline = spec.create_tile_pipeline(
        local_graph,
        input_key="x",
        tile_input_key="tile.x",
        output_name="local",
    )
    global_local_pipeline = spec.create_global_local_pipeline(
        global_graph=global_graph,
        tile_pipeline=tile_pipeline,
        fusion=LocalGlobalFusion(init_logit=0.0, learnable=False),
        global_output_name="global",
        fused_output_key="fused.out",
    )
    model = Model.from_components(
        spec,
        global_local_pipeline=global_local_pipeline,
    )
    x = torch.arange(4, dtype=torch.float32)
    store = KVStore(
        {
            "x": x,
            "global.bias": torch.tensor(10.0),
            "local.bias": torch.tensor(2.0),
        }
    )

    result = model(store)

    assert isinstance(model, Model)
    assert model.plan.mode == ExecutorModeSpec(tile=True, global_local=True)
    assert torch.equal(result, x + 6.0)
    assert torch.equal(store.get("fused.out"), result)


def test_branch_builder_stream_graph_runs_through_stream_model() -> None:
    stream_graph = (
        BranchBuilder.stream(input_key="chunk.x")
        .add(
            name="step",
            module=_Double(),
            output_key="chunk.out",
        )
        .build()
    )
    spec = ModelExecutionSpec(stateful=True)
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

    assert isinstance(model, Model)
    assert model.plan.mode == ExecutorModeSpec(stream=True)
    assert [out.item() for out in outputs] == [2.0, 4.0]
    assert store.get("stream.outputs") is outputs
