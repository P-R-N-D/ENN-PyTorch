from __future__ import annotations

import pytest
import torch
from torch import nn

from enn_torch_dev.executor import (
    GlobalLocalPipeline,
    GlobalLocalPipelineSpec,
    GraphExecutor,
    KVStore,
    KeyRef,
    NodeSpec,
    TilePipeline,
    TilePipelineSpec,
    TilePolicy,
)
from enn_torch_dev.nn import LocalGlobalFusion


class _AddBias(nn.Module):
    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return x + bias


class _TakeFirst(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:1]


def _make_global_graph() -> GraphExecutor:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="global",
            input_args=[KeyRef("x")],
            input_kwargs={"bias": KeyRef("global.bias")},
            output_key="global.out",
        ),
        _AddBias(),
    )
    return graph


def _make_tile_pipeline() -> TilePipeline:
    tile_graph = GraphExecutor()
    tile_graph.add_node(
        NodeSpec(
            name="local",
            input_args=[KeyRef("tile.x")],
            input_kwargs={"bias": KeyRef("local.bias")},
            output_key="local.out",
        ),
        _AddBias(),
    )
    return TilePipeline(
        tile_graph,
        TilePolicy(tile_shape=(2,)),
        TilePipelineSpec(
            input_key="x",
            tile_input_key="tile.x",
            output_name="local",
        ),
    )


def test_global_local_pipeline_runs_global_local_and_fuses() -> None:
    pipeline = GlobalLocalPipeline(
        global_graph=_make_global_graph(),
        tile_pipeline=_make_tile_pipeline(),
        fusion=LocalGlobalFusion(init_logit=0.0, learnable=False),
        spec=GlobalLocalPipelineSpec(global_output_name="global"),
    )
    x = torch.arange(5, dtype=torch.float32)
    store = KVStore(
        {
            "x": x,
            "global.bias": torch.tensor(10.0),
            "local.bias": torch.tensor(2.0),
        }
    )

    out = pipeline.run(store)

    assert torch.equal(out, x + 6.0)
    assert torch.equal(store.get("global.out"), x + 10.0)
    assert not store.has("tile.x")
    assert not store.has("local.out")


def test_global_local_pipeline_writes_fused_output_key() -> None:
    pipeline = GlobalLocalPipeline(
        global_graph=_make_global_graph(),
        tile_pipeline=_make_tile_pipeline(),
        fusion=LocalGlobalFusion(init_logit=0.0, learnable=False),
        spec=GlobalLocalPipelineSpec(
            global_output_name="global",
            fused_output_key="fused.out",
        ),
    )
    x = torch.arange(4, dtype=torch.float32)
    store = KVStore(
        {
            "x": x,
            "global.bias": torch.tensor(4.0),
            "local.bias": torch.tensor(2.0),
        }
    )

    out = pipeline.run(store)

    assert store.has("fused.out")
    assert torch.equal(store.get("fused.out"), out)


def test_global_local_pipeline_supports_global_output_by_key() -> None:
    pipeline = GlobalLocalPipeline(
        global_graph=_make_global_graph(),
        tile_pipeline=_make_tile_pipeline(),
        fusion=LocalGlobalFusion(init_logit=0.0, learnable=False),
        spec=GlobalLocalPipelineSpec(
            global_output_name="global",
            global_output_by="key",
        ),
    )
    x = torch.arange(3, dtype=torch.float32)
    store = KVStore(
        {
            "x": x,
            "global.bias": torch.tensor(10.0),
            "local.bias": torch.tensor(2.0),
        }
    )

    out = pipeline.run(store)

    assert torch.equal(out, x + 6.0)


def test_global_local_pipeline_forwards_fusion_shape_errors() -> None:
    global_graph = GraphExecutor()
    global_graph.add_node(
        NodeSpec(name="global", input_args=[KeyRef("x")], output_key="global.out"),
        _TakeFirst(),
    )
    pipeline = GlobalLocalPipeline(
        global_graph=global_graph,
        tile_pipeline=_make_tile_pipeline(),
        fusion=LocalGlobalFusion(),
        spec=GlobalLocalPipelineSpec(global_output_name="global"),
    )

    with pytest.raises(ValueError, match="same shape"):
        pipeline.run(
            KVStore(
                {
                    "x": torch.arange(5, dtype=torch.float32),
                    "local.bias": torch.tensor(0.0),
                }
            )
        )


def test_global_local_pipeline_rejects_unknown_global_output() -> None:
    with pytest.raises(KeyError, match="missing"):
        GlobalLocalPipeline(
            global_graph=_make_global_graph(),
            tile_pipeline=_make_tile_pipeline(),
            fusion=LocalGlobalFusion(),
            spec=GlobalLocalPipelineSpec(global_output_name="missing"),
        )


def test_global_local_pipeline_validates_inputs() -> None:
    with pytest.raises(ValueError, match="global_output_by"):
        GlobalLocalPipelineSpec(global_output_name="global", global_output_by="bad")

    with pytest.raises(TypeError, match="GraphExecutor"):
        GlobalLocalPipeline(
            global_graph=object(),
            tile_pipeline=_make_tile_pipeline(),
            fusion=LocalGlobalFusion(),
            spec=GlobalLocalPipelineSpec(global_output_name="global"),
        )
