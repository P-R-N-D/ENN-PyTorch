from __future__ import annotations

import pytest
import torch
from torch import nn

from enn_torch_dev.executor import (
    ExecutorModeSpec,
    ExecutorPlan,
    ExecutorRunner,
    GlobalLocalPipeline,
    GlobalLocalPipelineSpec,
    GraphExecutor,
    KVStore,
    KeyRef,
    NodeSpec,
    StreamPipeline,
    StreamPipelineSpec,
    TilePipeline,
    TilePipelineSpec,
    TilePolicy,
)
from enn_torch_dev.nn import LocalGlobalFusion


class _AddOne(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


class _Double(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0


class _AddBias(nn.Module):
    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return x + bias


def _make_graph(
    *,
    input_key: str = "x",
    output_key: str = "out",
    module: nn.Module | None = None,
) -> GraphExecutor:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="node",
            input_args=[KeyRef(input_key)],
            output_key=output_key,
        ),
        _AddOne() if module is None else module,
    )
    return graph


def _make_tile_pipeline(*, output_key: str | None = "tile.out") -> TilePipeline:
    graph = _make_graph(input_key="tile.x", output_key="local.out")
    return TilePipeline(
        graph,
        TilePolicy(tile_shape=(2,)),
        TilePipelineSpec(
            input_key="x",
            tile_input_key="tile.x",
            output_name="node",
            output_key=output_key,
        ),
    )


def _make_stream_pipeline() -> StreamPipeline:
    graph = _make_graph(input_key="chunk.x", output_key="chunk.out", module=_Double())
    return StreamPipeline(
        graph,
        StreamPipelineSpec(
            chunk_input_key="chunk.x",
            output_name="node",
            outputs_key="stream.outputs",
        ),
    )


def _make_global_local_pipeline() -> GlobalLocalPipeline:
    global_graph = GraphExecutor()
    global_graph.add_node(
        NodeSpec(
            name="global",
            input_args=[KeyRef("x")],
            input_kwargs={"bias": KeyRef("global.bias")},
            output_key="global.out",
        ),
        _AddBias(),
    )

    local_graph = GraphExecutor()
    local_graph.add_node(
        NodeSpec(
            name="local",
            input_args=[KeyRef("tile.x")],
            input_kwargs={"bias": KeyRef("local.bias")},
            output_key="local.out",
        ),
        _AddBias(),
    )

    tile_pipeline = TilePipeline(
        local_graph,
        TilePolicy(tile_shape=(2,)),
        TilePipelineSpec(
            input_key="x",
            tile_input_key="tile.x",
            output_name="local",
        ),
    )
    return GlobalLocalPipeline(
        global_graph=global_graph,
        tile_pipeline=tile_pipeline,
        fusion=LocalGlobalFusion(init_logit=0.0, learnable=False),
        spec=GlobalLocalPipelineSpec(
            global_output_name="global",
            fused_output_key="fused.out",
        ),
    )


def test_executor_runner_runs_plain_graph() -> None:
    plan = ExecutorPlan(mode=ExecutorModeSpec(), graph=_make_graph())
    runner = ExecutorRunner(plan)
    store = KVStore({"x": torch.tensor([1.0])})

    result = runner.run(store)

    assert result is store
    assert torch.equal(store.get("out"), torch.tensor([2.0]))


def test_executor_runner_runs_tile_pipeline() -> None:
    plan = ExecutorPlan(
        mode=ExecutorModeSpec(tile=True),
        tile_pipeline=_make_tile_pipeline(),
    )
    runner = ExecutorRunner(plan)
    store = KVStore({"x": torch.arange(4, dtype=torch.float32)})

    result = runner.run(store)

    assert torch.equal(result, torch.arange(4, dtype=torch.float32) + 1.0)
    assert torch.equal(store.get("tile.out"), result)


def test_executor_runner_runs_stream_pipeline() -> None:
    plan = ExecutorPlan(
        mode=ExecutorModeSpec(stream=True),
        stream_pipeline=_make_stream_pipeline(),
    )
    runner = ExecutorRunner(plan)
    store = KVStore()

    outputs = runner.run(
        store,
        chunks=[torch.tensor([1.0]), torch.tensor([2.0])],
    )

    assert [out.item() for out in outputs] == [2.0, 4.0]
    assert store.get("stream.outputs") is outputs


def test_executor_runner_runs_global_local_pipeline() -> None:
    plan = ExecutorPlan(
        mode=ExecutorModeSpec(tile=True, global_local=True),
        global_local_pipeline=_make_global_local_pipeline(),
    )
    runner = ExecutorRunner(plan)
    x = torch.arange(4, dtype=torch.float32)
    store = KVStore(
        {
            "x": x,
            "global.bias": torch.tensor(10.0),
            "local.bias": torch.tensor(2.0),
        }
    )

    result = runner.run(store)

    assert torch.equal(result, x + 6.0)
    assert torch.equal(store.get("fused.out"), result)


def test_executor_runner_uses_stream_as_outer_layer_for_stream_tile_mode() -> None:
    plan = ExecutorPlan(
        mode=ExecutorModeSpec(tile=True, stream=True),
        tile_pipeline=_make_tile_pipeline(),
        stream_pipeline=_make_stream_pipeline(),
    )
    runner = ExecutorRunner(plan)
    store = KVStore({"x": torch.arange(4, dtype=torch.float32)})

    outputs = runner.run(store, chunks=[torch.tensor([3.0])])

    assert [out.item() for out in outputs] == [6.0]
    assert store.get("stream.outputs") is outputs
    assert not store.has("tile.out")


def test_executor_runner_uses_stream_as_outer_layer_for_stream_global_local_mode() -> None:
    plan = ExecutorPlan(
        mode=ExecutorModeSpec(tile=True, stream=True, global_local=True),
        stream_pipeline=_make_stream_pipeline(),
        global_local_pipeline=_make_global_local_pipeline(),
    )
    runner = ExecutorRunner(plan)
    store = KVStore(
        {
            "x": torch.arange(4, dtype=torch.float32),
            "global.bias": torch.tensor(10.0),
            "local.bias": torch.tensor(2.0),
        }
    )

    outputs = runner.run(store, chunks=[torch.tensor([4.0])])

    assert [out.item() for out in outputs] == [8.0]
    assert store.get("stream.outputs") is outputs
    assert not store.has("fused.out")


def test_executor_runner_rejects_invalid_plan_and_store() -> None:
    with pytest.raises(TypeError, match="ExecutorPlan"):
        ExecutorRunner(object())

    runner = ExecutorRunner(ExecutorPlan(mode=ExecutorModeSpec(), graph=_make_graph()))
    with pytest.raises(TypeError, match="KVStore"):
        runner.run(object())


def test_executor_runner_requires_chunks_for_stream_mode() -> None:
    runner = ExecutorRunner(
        ExecutorPlan(
            mode=ExecutorModeSpec(stream=True),
            stream_pipeline=_make_stream_pipeline(),
        )
    )

    with pytest.raises(ValueError, match="requires chunks"):
        runner.run(KVStore())


def test_executor_runner_rejects_chunks_for_non_stream_mode() -> None:
    runner = ExecutorRunner(ExecutorPlan(mode=ExecutorModeSpec(), graph=_make_graph()))

    with pytest.raises(ValueError, match="only valid for stream"):
        runner.run(KVStore({"x": torch.tensor([1.0])}), chunks=[torch.tensor([1.0])])
