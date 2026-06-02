from __future__ import annotations

import pytest
import torch
from torch import nn

from enn_torch_dev.executor import (
    ExecutorModeSpec,
    ExecutorModel,
    ExecutorPlan,
    GlobalLocalPipeline,
    GlobalLocalPipelineSpec,
    GraphExecutor,
    KVStore,
    KeyRef,
    ModelExecutionSpec,
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


def test_executor_model_runs_plain_graph() -> None:
    spec = ModelExecutionSpec()
    plan = spec.create_plan(graph=_make_graph())
    model = ExecutorModel(spec=spec, plan=plan)
    store = KVStore({"x": torch.tensor([1.0])})

    result = model.run(store)

    assert result is store
    assert torch.equal(store.get("out"), torch.tensor([2.0]))


def test_executor_model_from_components_runs_tile_pipeline() -> None:
    model = ExecutorModel.from_components(
        ModelExecutionSpec(tile=True, tile_shape=(2,)),
        tile_pipeline=_make_tile_pipeline(),
    )
    store = KVStore({"x": torch.arange(4, dtype=torch.float32)})

    result = model.run(store)

    assert torch.equal(result, torch.arange(4, dtype=torch.float32) + 1.0)
    assert torch.equal(store.get("tile.out"), result)


def test_executor_model_from_components_runs_stateful_stream_pipeline() -> None:
    model = ExecutorModel.from_components(
        ModelExecutionSpec(stateful=True),
        stream_pipeline=_make_stream_pipeline(),
    )
    store = KVStore()

    outputs = model.run(
        store,
        chunks=[torch.tensor([1.0]), torch.tensor([2.0])],
    )

    assert [out.item() for out in outputs] == [2.0, 4.0]
    assert store.get("stream.outputs") is outputs


def test_executor_model_from_components_runs_global_local_pipeline() -> None:
    model = ExecutorModel.from_components(
        ModelExecutionSpec(context="global_local", tile=True, tile_shape=(2,)),
        global_local_pipeline=_make_global_local_pipeline(),
    )
    x = torch.arange(4, dtype=torch.float32)
    store = KVStore(
        {
            "x": x,
            "global.bias": torch.tensor(10.0),
            "local.bias": torch.tensor(2.0),
        }
    )

    result = model.run(store)

    assert torch.equal(result, x + 6.0)
    assert torch.equal(store.get("fused.out"), result)


def test_executor_model_from_components_runs_stateful_global_local_as_stream_outer() -> None:
    model = ExecutorModel.from_components(
        ModelExecutionSpec(
            context="global_local",
            tile=True,
            stateful=True,
            tile_shape=(2,),
        ),
        stream_pipeline=_make_stream_pipeline(),
        global_local_pipeline=_make_global_local_pipeline(),
    )
    store = KVStore(
        {
            "x": torch.arange(4, dtype=torch.float32),
            "global.bias": torch.tensor(10.0),
            "local.bias": torch.tensor(2.0),
        }
    )

    outputs = model.run(store, chunks=[torch.tensor([3.0])])

    assert [out.item() for out in outputs] == [6.0]
    assert store.get("stream.outputs") is outputs
    assert not store.has("fused.out")


def test_executor_model_rejects_invalid_spec_and_plan() -> None:
    with pytest.raises(TypeError, match="ModelExecutionSpec"):
        ExecutorModel(spec=object(), plan=object())

    spec = ModelExecutionSpec()
    with pytest.raises(TypeError, match="ExecutorPlan"):
        ExecutorModel(spec=spec, plan=object())


def test_executor_model_rejects_spec_plan_mode_mismatch() -> None:
    spec = ModelExecutionSpec()
    plan = ExecutorPlan(
        mode=ExecutorModeSpec(tile=True),
        tile_pipeline=_make_tile_pipeline(),
    )

    with pytest.raises(ValueError, match="plan mode must match"):
        ExecutorModel(spec=spec, plan=plan)


def test_executor_model_from_components_validates_spec_type() -> None:
    with pytest.raises(TypeError, match="ModelExecutionSpec"):
        ExecutorModel.from_components(object())


def test_executor_model_delegates_runner_validation() -> None:
    model = ExecutorModel.from_components(
        ModelExecutionSpec(stateful=True),
        stream_pipeline=_make_stream_pipeline(),
    )

    with pytest.raises(ValueError, match="requires chunks"):
        model.run(KVStore())

    plain = ExecutorModel.from_components(
        ModelExecutionSpec(),
        graph=_make_graph(),
    )
    with pytest.raises(ValueError, match="only valid for stream"):
        plain.run(KVStore({"x": torch.tensor([1.0])}), chunks=[torch.tensor([1.0])])


def test_executor_model_public_plain_flow_end_to_end() -> None:
    spec = ModelExecutionSpec(context="local", tile=False, stateful=False)
    model = ExecutorModel.from_components(spec, graph=_make_graph())
    store = KVStore({"x": torch.tensor([1.0])})

    result = model.run(store)

    assert model.spec is spec
    assert model.plan.mode == ExecutorModeSpec()
    assert model.plan.execution_layers == ("graph",)
    assert model.plan.component_names == ("graph",)
    assert model.runner.plan is model.plan
    assert result is store
    assert torch.equal(store.get("out"), torch.tensor([2.0]))


def test_executor_model_public_tiled_flow_end_to_end() -> None:
    tile_pipeline = _make_tile_pipeline()
    spec = ModelExecutionSpec(
        context="local",
        tile=True,
        stateful=False,
        tile_shape=tile_pipeline.tile_policy.tile_shape,
    )
    model = ExecutorModel.from_components(spec, tile_pipeline=tile_pipeline)
    store = KVStore({"x": torch.arange(4, dtype=torch.float32)})

    result = model.run(store)

    assert model.plan.mode == ExecutorModeSpec(tile=True)
    assert model.plan.execution_layers == ("tile",)
    assert model.plan.component_names == ("tile_pipeline",)
    assert torch.equal(result, torch.arange(4, dtype=torch.float32) + 1.0)
    assert torch.equal(store.get("tile.out"), result)


def test_executor_model_public_stateful_flow_end_to_end() -> None:
    spec = ModelExecutionSpec(context="local", tile=False, stateful=True)
    model = ExecutorModel.from_components(
        spec,
        stream_pipeline=_make_stream_pipeline(),
    )
    store = KVStore()

    outputs = model.run(
        store,
        chunks=[torch.tensor([1.0]), torch.tensor([2.0])],
    )

    assert model.plan.mode == ExecutorModeSpec(stream=True)
    assert model.plan.execution_layers == ("stream",)
    assert model.plan.component_names == ("stream_pipeline",)
    assert [out.item() for out in outputs] == [2.0, 4.0]
    assert store.get("stream.outputs") is outputs


def test_executor_model_public_global_local_flow_end_to_end() -> None:
    global_local_pipeline = _make_global_local_pipeline()
    spec = ModelExecutionSpec(
        context="global_local",
        tile=True,
        stateful=False,
        tile_shape=global_local_pipeline.tile_pipeline.tile_policy.tile_shape,
    )
    model = ExecutorModel.from_components(
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

    result = model.run(store)

    assert model.plan.mode == ExecutorModeSpec(tile=True, global_local=True)
    assert model.plan.execution_layers == ("global_local",)
    assert model.plan.component_names == ("global_local_pipeline",)
    assert torch.equal(result, x + 6.0)
    assert torch.equal(store.get("fused.out"), result)


def test_executor_model_public_stateful_global_local_flow_uses_stream_outer() -> None:
    global_local_pipeline = _make_global_local_pipeline()
    spec = ModelExecutionSpec(
        context="global_local",
        tile=True,
        stateful=True,
        tile_shape=global_local_pipeline.tile_pipeline.tile_policy.tile_shape,
    )
    model = ExecutorModel.from_components(
        spec,
        stream_pipeline=_make_stream_pipeline(),
        global_local_pipeline=global_local_pipeline,
    )
    store = KVStore(
        {
            "x": torch.arange(4, dtype=torch.float32),
            "global.bias": torch.tensor(10.0),
            "local.bias": torch.tensor(2.0),
        }
    )

    outputs = model.run(store, chunks=[torch.tensor([3.0])])

    assert model.plan.mode == ExecutorModeSpec(
        tile=True,
        stream=True,
        global_local=True,
    )
    assert model.plan.execution_layers == ("stream", "global_local")
    assert model.plan.component_names == ("stream_pipeline", "global_local_pipeline")
    assert [out.item() for out in outputs] == [6.0]
    assert store.get("stream.outputs") is outputs
    assert not store.has("fused.out")


def test_executor_model_tile_factory_flow_end_to_end() -> None:
    spec = ModelExecutionSpec(tile=True, tile_shape=(2,))
    tile_graph = _make_graph(input_key="tile.x", output_key="tile.out")

    tile_pipeline = spec.create_tile_pipeline(
        tile_graph,
        input_key="x",
        tile_input_key="tile.x",
        output_name="node",
        output_key="model.out",
    )
    plan = spec.create_plan(tile_pipeline=tile_pipeline)
    model = ExecutorModel(spec=spec, plan=plan)
    store = KVStore({"x": torch.arange(4, dtype=torch.float32)})

    result = model.run(store)

    assert model.plan.mode == ExecutorModeSpec(tile=True)
    assert model.plan.execution_layers == ("tile",)
    assert model.plan.component_names == ("tile_pipeline",)
    assert torch.equal(result, torch.arange(4, dtype=torch.float32) + 1.0)
    assert torch.equal(store.get("model.out"), result)


def test_executor_model_global_local_factory_flow_end_to_end() -> None:
    spec = ModelExecutionSpec(context="global_local", tile=True, tile_shape=(2,))

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
    plan = spec.create_plan(global_local_pipeline=global_local_pipeline)
    model = ExecutorModel(spec=spec, plan=plan)

    x = torch.arange(4, dtype=torch.float32)
    store = KVStore(
        {
            "x": x,
            "global.bias": torch.tensor(10.0),
            "local.bias": torch.tensor(2.0),
        }
    )

    result = model.run(store)

    assert model.plan.mode == ExecutorModeSpec(tile=True, global_local=True)
    assert model.plan.execution_layers == ("global_local",)
    assert model.plan.component_names == ("global_local_pipeline",)
    assert torch.equal(result, x + 6.0)
    assert torch.equal(store.get("fused.out"), result)


def test_executor_model_stream_factory_flow_end_to_end() -> None:
    spec = ModelExecutionSpec(stateful=True)
    stream_graph = _make_graph(
        input_key="chunk.x",
        output_key="chunk.out",
        module=_Double(),
    )

    stream_pipeline = spec.create_stream_pipeline(
        stream_graph,
        chunk_input_key="chunk.x",
        output_name="node",
        outputs_key="stream.outputs",
    )
    plan = spec.create_plan(stream_pipeline=stream_pipeline)
    model = ExecutorModel(spec=spec, plan=plan)
    store = KVStore()

    outputs = model.run(
        store,
        chunks=[torch.tensor([1.0]), torch.tensor([2.0])],
    )

    assert model.plan.mode == ExecutorModeSpec(stream=True)
    assert model.plan.execution_layers == ("stream",)
    assert model.plan.component_names == ("stream_pipeline",)
    assert [out.item() for out in outputs] == [2.0, 4.0]
    assert store.get("stream.outputs") is outputs
