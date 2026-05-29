from __future__ import annotations

import pytest
from torch import nn

from enn_torch_dev.executor import (
    ExecutorModeSpec,
    ExecutorPlan,
    GlobalLocalPipeline,
    GlobalLocalPipelineSpec,
    GraphExecutor,
    KeyRef,
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
        StreamPipelineSpec(
            chunk_input_key="chunk.x",
            output_name="node",
        ),
    )


def _make_global_local_pipeline() -> GlobalLocalPipeline:
    global_graph = _make_graph(input_key="x", output_key="global.out")
    return GlobalLocalPipeline(
        global_graph=global_graph,
        tile_pipeline=_make_tile_pipeline(),
        fusion=LocalGlobalFusion(init_logit=0.0, learnable=False),
        spec=GlobalLocalPipelineSpec(global_output_name="node"),
    )


def test_executor_plan_plain_mode_requires_graph() -> None:
    plan = ExecutorPlan(
        mode=ExecutorModeSpec(),
        graph=_make_graph(),
    )

    assert plan.is_plain
    assert plan.execution_layers == ("graph",)
    assert plan.component_names == ("graph",)


def test_executor_plan_plain_mode_rejects_missing_graph() -> None:
    with pytest.raises(ValueError, match="plain mode requires graph"):
        ExecutorPlan(mode=ExecutorModeSpec())


def test_executor_plan_tile_mode_requires_tile_pipeline() -> None:
    plan = ExecutorPlan(
        mode=ExecutorModeSpec(tile=True),
        tile_pipeline=_make_tile_pipeline(),
    )

    assert not plan.is_plain
    assert plan.execution_layers == ("tile",)
    assert plan.component_names == ("tile_pipeline",)


def test_executor_plan_stream_mode_requires_stream_pipeline() -> None:
    plan = ExecutorPlan(
        mode=ExecutorModeSpec(stream=True),
        stream_pipeline=_make_stream_pipeline(),
    )

    assert plan.execution_layers == ("stream",)
    assert plan.component_names == ("stream_pipeline",)


def test_executor_plan_stream_tile_composition_uses_stream_as_outer_layer() -> None:
    plan = ExecutorPlan(
        mode=ExecutorModeSpec(tile=True, stream=True),
        tile_pipeline=_make_tile_pipeline(),
        stream_pipeline=_make_stream_pipeline(),
    )

    assert plan.execution_layers == ("stream", "tile")
    assert plan.component_names == ("stream_pipeline", "tile_pipeline")


def test_executor_plan_global_local_mode_uses_global_local_pipeline() -> None:
    plan = ExecutorPlan(
        mode=ExecutorModeSpec(tile=True, global_local=True),
        global_local_pipeline=_make_global_local_pipeline(),
    )

    assert plan.execution_layers == ("global_local",)
    assert plan.component_names == ("global_local_pipeline",)


def test_executor_plan_stream_global_local_composition_uses_stream_outer_layer() -> None:
    plan = ExecutorPlan(
        mode=ExecutorModeSpec(tile=True, stream=True, global_local=True),
        stream_pipeline=_make_stream_pipeline(),
        global_local_pipeline=_make_global_local_pipeline(),
    )

    assert plan.execution_layers == ("stream", "global_local")
    assert plan.component_names == ("stream_pipeline", "global_local_pipeline")


def test_executor_plan_rejects_redundant_tile_pipeline_for_global_local_mode() -> None:
    with pytest.raises(ValueError, match="embedded TilePipeline"):
        ExecutorPlan(
            mode=ExecutorModeSpec(tile=True, global_local=True),
            tile_pipeline=_make_tile_pipeline(),
            global_local_pipeline=_make_global_local_pipeline(),
        )


def test_executor_plan_rejects_components_not_enabled_by_mode() -> None:
    with pytest.raises(ValueError, match="graph is only valid"):
        ExecutorPlan(
            mode=ExecutorModeSpec(tile=True),
            graph=_make_graph(),
            tile_pipeline=_make_tile_pipeline(),
        )

    with pytest.raises(ValueError, match="tile_pipeline requires tile=True"):
        ExecutorPlan(
            mode=ExecutorModeSpec(stream=True),
            tile_pipeline=_make_tile_pipeline(),
            stream_pipeline=_make_stream_pipeline(),
        )

    with pytest.raises(ValueError, match="stream_pipeline requires stream=True"):
        ExecutorPlan(
            mode=ExecutorModeSpec(tile=True),
            tile_pipeline=_make_tile_pipeline(),
            stream_pipeline=_make_stream_pipeline(),
        )

    with pytest.raises(ValueError, match="global_local_pipeline requires"):
        ExecutorPlan(
            mode=ExecutorModeSpec(tile=True),
            tile_pipeline=_make_tile_pipeline(),
            global_local_pipeline=_make_global_local_pipeline(),
        )


def test_executor_plan_rejects_missing_components_for_enabled_modes() -> None:
    with pytest.raises(ValueError, match="tile mode requires tile_pipeline"):
        ExecutorPlan(mode=ExecutorModeSpec(tile=True))

    with pytest.raises(ValueError, match="stream mode requires stream_pipeline"):
        ExecutorPlan(mode=ExecutorModeSpec(stream=True))

    with pytest.raises(ValueError, match="global_local mode requires"):
        ExecutorPlan(mode=ExecutorModeSpec(tile=True, global_local=True))


def test_executor_plan_rejects_invalid_component_types() -> None:
    with pytest.raises(TypeError, match="mode"):
        ExecutorPlan(mode=object())

    with pytest.raises(TypeError, match="graph"):
        ExecutorPlan(mode=ExecutorModeSpec(), graph=object())

    with pytest.raises(TypeError, match="tile_pipeline"):
        ExecutorPlan(mode=ExecutorModeSpec(tile=True), tile_pipeline=object())

    with pytest.raises(TypeError, match="stream_pipeline"):
        ExecutorPlan(mode=ExecutorModeSpec(stream=True), stream_pipeline=object())

    with pytest.raises(TypeError, match="global_local_pipeline"):
        ExecutorPlan(
            mode=ExecutorModeSpec(tile=True, global_local=True),
            global_local_pipeline=object(),
        )
