from __future__ import annotations

import pytest
import torch
from torch import nn

from enn_torch_dev.executor import (
    ExecutorModeSpec,
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


def _make_graph(*, input_key: str = "x", output_key: str = "out") -> GraphExecutor:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="node",
            input_args=[KeyRef(input_key)],
            output_key=output_key,
        ),
        nn.Identity(),
    )
    return graph


def _make_tile_pipeline() -> TilePipeline:
    graph = _make_graph(input_key="tile.x", output_key="tile.out")
    return TilePipeline(
        graph,
        TilePolicy(tile_shape=(1,)),
        TilePipelineSpec(
            input_key="x",
            tile_input_key="tile.x",
            output_name="node",
        ),
    )


def _make_stream_pipeline() -> StreamPipeline:
    graph = _make_graph(input_key="chunk.x", output_key="chunk.out")
    return StreamPipeline(
        graph,
        StreamPipelineSpec(chunk_input_key="chunk.x", output_name="node"),
    )


def _make_global_local_pipeline() -> GlobalLocalPipeline:
    return GlobalLocalPipeline(
        global_graph=_make_graph(input_key="x", output_key="global.out"),
        tile_pipeline=_make_tile_pipeline(),
        fusion=LocalGlobalFusion(init_logit=0.0, learnable=False),
        spec=GlobalLocalPipelineSpec(global_output_name="node"),
    )


def test_model_execution_spec_defaults_to_plain_local_mode() -> None:
    spec = ModelExecutionSpec()

    assert spec.context == "local"
    assert not spec.tile
    assert not spec.stateful
    assert spec.tile_shape is None
    assert spec.executor_mode == ExecutorModeSpec()


def test_model_execution_spec_maps_local_tiled_mode() -> None:
    spec = ModelExecutionSpec(
        context="local",
        tile=True,
        tile_shape=[2, 3],
        tile_stride=[1, 2],
        tile_dims=[-2, -1],
    )

    assert spec.tile_shape == (2, 3)
    assert spec.tile_stride == (1, 2)
    assert spec.tile_dims == (-2, -1)
    assert spec.to_executor_mode_spec() == ExecutorModeSpec(tile=True)


def test_model_execution_spec_maps_global_local_mode() -> None:
    spec = ModelExecutionSpec(
        context="global_local",
        tile=True,
        tile_shape=(4, 4),
    )

    assert spec.uses_global_local
    assert spec.executor_mode == ExecutorModeSpec(tile=True, global_local=True)


def test_model_execution_spec_maps_stateful_local_mode() -> None:
    spec = ModelExecutionSpec(stateful=True)

    assert spec.executor_mode == ExecutorModeSpec(stream=True)


def test_model_execution_spec_maps_stateful_tiled_mode() -> None:
    spec = ModelExecutionSpec(
        tile=True,
        stateful=True,
        tile_shape=(8,),
    )

    assert spec.executor_mode == ExecutorModeSpec(tile=True, stream=True)


def test_model_execution_spec_maps_stateful_global_local_mode() -> None:
    spec = ModelExecutionSpec(
        context="global_local",
        tile=True,
        stateful=True,
        tile_shape=(8, 8),
    )

    assert spec.executor_mode == ExecutorModeSpec(
        tile=True,
        stream=True,
        global_local=True,
    )


def test_model_execution_spec_requires_tile_shape_when_tiling() -> None:
    with pytest.raises(ValueError, match="tile=True requires tile_shape"):
        ModelExecutionSpec(tile=True)


def test_model_execution_spec_rejects_tile_options_when_tile_is_disabled() -> None:
    with pytest.raises(ValueError, match="require tile=True"):
        ModelExecutionSpec(tile_shape=(4,))

    with pytest.raises(ValueError, match="require tile=True"):
        ModelExecutionSpec(tile_stride=(4,))

    with pytest.raises(ValueError, match="require tile=True"):
        ModelExecutionSpec(tile_dims=(-1,))


def test_model_execution_spec_requires_tile_for_global_local_context() -> None:
    with pytest.raises(ValueError, match="global_local.*requires tile=True"):
        ModelExecutionSpec(context="global_local")


def test_model_execution_spec_validates_context() -> None:
    with pytest.raises(TypeError, match="context"):
        ModelExecutionSpec(context=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="local.*global_local"):
        ModelExecutionSpec(context="global")


def test_model_execution_spec_validates_bool_flags() -> None:
    with pytest.raises(TypeError, match="tile"):
        ModelExecutionSpec(tile=1)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="stateful"):
        ModelExecutionSpec(stateful="yes")  # type: ignore[arg-type]


def test_model_execution_spec_validates_tile_shape() -> None:
    with pytest.raises(TypeError, match="tile_shape"):
        ModelExecutionSpec(tile=True, tile_shape=4)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="tile_shape"):
        ModelExecutionSpec(tile=True, tile_shape=())

    with pytest.raises(TypeError, match="tile_shape"):
        ModelExecutionSpec(tile=True, tile_shape=(True,))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="positive"):
        ModelExecutionSpec(tile=True, tile_shape=(0,))


def test_model_execution_spec_validates_tile_stride() -> None:
    with pytest.raises(TypeError, match="tile_stride"):
        ModelExecutionSpec(tile=True, tile_shape=(4,), tile_stride=2)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="tile_stride length"):
        ModelExecutionSpec(tile=True, tile_shape=(4, 4), tile_stride=(2,))

    with pytest.raises(TypeError, match="tile_stride"):
        ModelExecutionSpec(tile=True, tile_shape=(4,), tile_stride=(False,))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="positive"):
        ModelExecutionSpec(tile=True, tile_shape=(4,), tile_stride=(-1,))


def test_model_execution_spec_validates_tile_dims() -> None:
    with pytest.raises(TypeError, match="tile_dims"):
        ModelExecutionSpec(tile=True, tile_shape=(4,), tile_dims=-1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="tile_dims length"):
        ModelExecutionSpec(tile=True, tile_shape=(4, 4), tile_dims=(-1,))

    with pytest.raises(TypeError, match="tile_dims"):
        ModelExecutionSpec(tile=True, tile_shape=(4,), tile_dims=(True,))  # type: ignore[arg-type]


def test_model_execution_spec_create_tile_policy_from_tile_config() -> None:
    spec = ModelExecutionSpec(
        tile=True,
        tile_shape=[2, 3],
        tile_stride=[1, 2],
        tile_dims=[-2, -1],
    )

    policy = spec.create_tile_policy()

    assert isinstance(policy, TilePolicy)
    assert policy.tile_shape == (2, 3)
    assert policy.stride == (1, 2)
    assert policy.dims == (-2, -1)
    assert policy.drop_last is False


def test_model_execution_spec_create_tile_policy_defaults_stride_to_tile_shape() -> None:
    spec = ModelExecutionSpec(
        tile=True,
        tile_shape=(4, 5),
    )

    policy = spec.create_tile_policy()

    assert policy.tile_shape == (4, 5)
    assert policy.stride == (4, 5)
    assert policy.dims is None


def test_model_execution_spec_create_tile_policy_requires_tile_enabled() -> None:
    with pytest.raises(ValueError, match="requires tile=True"):
        ModelExecutionSpec().create_tile_policy()


def test_model_execution_spec_create_tile_policy_matches_direct_tile_policy() -> None:
    spec = ModelExecutionSpec(tile=True, tile_shape=(3,), tile_stride=(2,), tile_dims=(-1,))

    policy = spec.create_tile_policy()
    expected = TilePolicy(tile_shape=(3,), stride=(2,), dims=(-1,))

    assert policy == expected


def test_model_execution_spec_create_tile_pipeline_spec_from_keys() -> None:
    spec = ModelExecutionSpec(tile=True, tile_shape=(2, 3))

    pipeline_spec = spec.create_tile_pipeline_spec(
        input_key="x",
        tile_input_key="tile.x",
        output_name="local",
    )

    assert isinstance(pipeline_spec, TilePipelineSpec)
    assert pipeline_spec.input_key == "x"
    assert pipeline_spec.tile_input_key == "tile.x"
    assert pipeline_spec.output_name == "local"
    assert pipeline_spec.output_key is None
    assert pipeline_spec.output_by == "node"
    assert pipeline_spec.tile_index_key is None
    assert pipeline_spec.tile_meta_key is None


def test_model_execution_spec_create_tile_pipeline_spec_supports_optional_keys() -> None:
    spec = ModelExecutionSpec(tile=True, tile_shape=(2,))

    pipeline_spec = spec.create_tile_pipeline_spec(
        input_key="x",
        tile_input_key="tile.x",
        output_name="local",
        output_key="model.out",
        output_by="key",
        tile_index_key="tile.index",
        tile_meta_key="tile.meta",
    )

    assert pipeline_spec.output_key == "model.out"
    assert pipeline_spec.output_by == "key"
    assert pipeline_spec.tile_index_key == "tile.index"
    assert pipeline_spec.tile_meta_key == "tile.meta"


def test_model_execution_spec_create_tile_pipeline_spec_requires_tile_enabled() -> None:
    with pytest.raises(ValueError, match="requires tile=True"):
        ModelExecutionSpec().create_tile_pipeline_spec(
            input_key="x",
            tile_input_key="tile.x",
            output_name="local",
        )


def test_model_execution_spec_create_tile_pipeline_spec_delegates_validation() -> None:
    spec = ModelExecutionSpec(tile=True, tile_shape=(2,))

    with pytest.raises(ValueError, match="output_by"):
        spec.create_tile_pipeline_spec(
            input_key="x",
            tile_input_key="tile.x",
            output_name="local",
            output_by="bad",
        )


def test_model_execution_spec_create_tile_pipeline_from_graph_and_keys() -> None:
    spec = ModelExecutionSpec(tile=True, tile_shape=(1,))
    graph = _make_graph(input_key="tile.x", output_key="tile.out")

    pipeline = spec.create_tile_pipeline(
        graph,
        input_key="x",
        tile_input_key="tile.x",
        output_name="node",
        output_key="model.out",
    )

    assert isinstance(pipeline, TilePipeline)
    assert pipeline.graph is graph
    assert pipeline.tile_policy == spec.create_tile_policy()
    assert pipeline.spec.input_key == "x"
    assert pipeline.spec.tile_input_key == "tile.x"
    assert pipeline.spec.output_name == "node"
    assert pipeline.spec.output_key == "model.out"

    store = KVStore({"x": torch.tensor([1.0, 2.0])})
    out = pipeline.run(store)

    assert torch.equal(out, torch.tensor([1.0, 2.0]))
    assert torch.equal(store.get("model.out"), out)


def test_model_execution_spec_create_tile_pipeline_supports_optional_spec_keys() -> None:
    spec = ModelExecutionSpec(
        tile=True,
        tile_shape=(1,),
        tile_stride=(1,),
        tile_dims=(-1,),
    )

    pipeline = spec.create_tile_pipeline(
        _make_graph(input_key="tile.x", output_key="tile.out"),
        input_key="x",
        tile_input_key="tile.x",
        output_name="node",
        output_by="key",
        tile_index_key="tile.index",
        tile_meta_key="tile.meta",
    )

    assert pipeline.tile_policy.tile_shape == (1,)
    assert pipeline.tile_policy.stride == (1,)
    assert pipeline.tile_policy.dims == (-1,)
    assert pipeline.spec.output_by == "key"
    assert pipeline.spec.tile_index_key == "tile.index"
    assert pipeline.spec.tile_meta_key == "tile.meta"


def test_model_execution_spec_create_tile_pipeline_requires_tile_enabled() -> None:
    with pytest.raises(ValueError, match="requires tile=True"):
        ModelExecutionSpec().create_tile_pipeline(
            _make_graph(input_key="tile.x", output_key="tile.out"),
            input_key="x",
            tile_input_key="tile.x",
            output_name="node",
        )


def test_model_execution_spec_create_tile_pipeline_delegates_validation() -> None:
    spec = ModelExecutionSpec(tile=True, tile_shape=(1,))

    with pytest.raises(ValueError, match="output_by"):
        spec.create_tile_pipeline(
            _make_graph(input_key="tile.x", output_key="tile.out"),
            input_key="x",
            tile_input_key="tile.x",
            output_name="node",
            output_by="bad",
        )

    with pytest.raises(KeyError, match="missing"):
        spec.create_tile_pipeline(
            _make_graph(input_key="tile.x", output_key="tile.out"),
            input_key="x",
            tile_input_key="tile.x",
            output_name="missing",
        )


def test_model_execution_spec_create_plan_for_plain_mode() -> None:
    spec = ModelExecutionSpec()

    plan = spec.create_plan(graph=_make_graph())

    assert isinstance(plan, ExecutorPlan)
    assert plan.mode == ExecutorModeSpec()
    assert plan.component_names == ("graph",)


def test_model_execution_spec_create_plan_for_tiled_mode() -> None:
    spec = ModelExecutionSpec(tile=True, tile_shape=(1,))

    plan = spec.create_plan(tile_pipeline=_make_tile_pipeline())

    assert plan.mode == ExecutorModeSpec(tile=True)
    assert plan.component_names == ("tile_pipeline",)


def test_model_execution_spec_create_plan_for_stateful_mode() -> None:
    spec = ModelExecutionSpec(stateful=True)

    plan = spec.create_plan(stream_pipeline=_make_stream_pipeline())

    assert plan.mode == ExecutorModeSpec(stream=True)
    assert plan.component_names == ("stream_pipeline",)


def test_model_execution_spec_create_plan_for_stateful_tiled_mode() -> None:
    spec = ModelExecutionSpec(tile=True, stateful=True, tile_shape=(1,))

    plan = spec.create_plan(
        tile_pipeline=_make_tile_pipeline(),
        stream_pipeline=_make_stream_pipeline(),
    )

    assert plan.mode == ExecutorModeSpec(tile=True, stream=True)
    assert plan.execution_layers == ("stream", "tile")
    assert plan.component_names == ("stream_pipeline", "tile_pipeline")


def test_model_execution_spec_create_plan_for_global_local_mode() -> None:
    spec = ModelExecutionSpec(context="global_local", tile=True, tile_shape=(1,))

    plan = spec.create_plan(global_local_pipeline=_make_global_local_pipeline())

    assert plan.mode == ExecutorModeSpec(tile=True, global_local=True)
    assert plan.component_names == ("global_local_pipeline",)


def test_model_execution_spec_create_plan_for_stateful_global_local_mode() -> None:
    spec = ModelExecutionSpec(
        context="global_local",
        tile=True,
        stateful=True,
        tile_shape=(1,),
    )

    plan = spec.create_plan(
        stream_pipeline=_make_stream_pipeline(),
        global_local_pipeline=_make_global_local_pipeline(),
    )

    assert plan.mode == ExecutorModeSpec(
        tile=True,
        stream=True,
        global_local=True,
    )
    assert plan.execution_layers == ("stream", "global_local")
    assert plan.component_names == ("stream_pipeline", "global_local_pipeline")


def test_model_execution_spec_create_plan_uses_executor_plan_validation() -> None:
    with pytest.raises(ValueError, match="plain mode requires graph"):
        ModelExecutionSpec().create_plan()

    with pytest.raises(ValueError, match="tile mode requires tile_pipeline"):
        ModelExecutionSpec(tile=True, tile_shape=(1,)).create_plan()

    with pytest.raises(ValueError, match="stream mode requires stream_pipeline"):
        ModelExecutionSpec(stateful=True).create_plan()

    with pytest.raises(ValueError, match="global_local mode requires"):
        ModelExecutionSpec(context="global_local", tile=True, tile_shape=(1,)).create_plan()
