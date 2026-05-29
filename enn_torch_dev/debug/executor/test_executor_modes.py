from __future__ import annotations

import torch
from torch import nn

from enn_torch_dev.executor import (
    GlobalLocalPipeline,
    GlobalLocalPipelineSpec,
    GraphExecutor,
    KVStore,
    KeyRef,
    NodeSpec,
    StateRoute,
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


class _AddBias(nn.Module):
    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return x + bias


class _RunningSum(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        return_state: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        prev = torch.zeros_like(x) if state is None else state
        out = prev + x
        if return_state:
            return out, out
        return out


def _make_tile_pipeline(*, output_key: str | None = None) -> TilePipeline:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="local",
            input_args=[KeyRef("tile.x")],
            output_key="local.out",
        ),
        _AddOne(),
    )
    return TilePipeline(
        graph,
        TilePolicy(tile_shape=(2,)),
        TilePipelineSpec(
            input_key="x",
            tile_input_key="tile.x",
            output_name="local",
            output_key=output_key,
        ),
    )


def test_tile_mode_is_position_split_execute_reconstruct() -> None:
    pipeline = _make_tile_pipeline(output_key="tile.out")
    x = torch.arange(6, dtype=torch.float32)
    store = KVStore({"x": x})

    out = pipeline.run(store)

    assert torch.equal(out, x + 1.0)
    assert torch.equal(store.get("tile.out"), x + 1.0)

    # Tile mode reconstructs into a full tensor and does not create stream state.
    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape
    assert not store.has("state.in")
    assert not store.has("state.out")


def test_stream_mode_is_ordered_chunk_execution_with_state_carry() -> None:
    route = StateRoute("sum.state.in", "sum.state.out")
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="sum",
            input_args=[KeyRef("chunk.x")],
            input_kwargs=route.input_kwargs(),
            output_key="sum.out",
            output_keys=route.output_keys("sum.out"),
        ),
        _RunningSum(),
    )
    pipeline = StreamPipeline(
        graph,
        StreamPipelineSpec(
            chunk_input_key="chunk.x",
            output_name="sum",
            outputs_key="stream.outputs",
        ),
        state_routes=[route],
    )
    store = KVStore()

    outputs = pipeline.run(
        store,
        [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])],
    )

    assert [out.item() for out in outputs] == [1.0, 3.0, 6.0]
    assert store.get("stream.outputs") is outputs
    assert torch.equal(store.get("sum.state.in"), torch.tensor([6.0]))

    # Stream mode returns an output sequence and keeps per-chunk graph values
    # isolated from the base store.
    assert isinstance(outputs, list)
    assert not store.has("chunk.x")
    assert not store.has("sum.out")
    assert not store.has("sum.state.out")


def test_stream_reset_state_starts_a_new_sequence() -> None:
    route = StateRoute("sum.state.in", "sum.state.out")
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="sum",
            input_args=[KeyRef("chunk.x")],
            input_kwargs=route.input_kwargs(),
            output_key="sum.out",
            output_keys=route.output_keys("sum.out"),
        ),
        _RunningSum(),
    )
    pipeline = StreamPipeline(
        graph,
        StreamPipelineSpec(
            chunk_input_key="chunk.x",
            output_name="sum",
            reset_state=True,
        ),
        state_routes=[route],
    )
    store = KVStore({"sum.state.in": torch.tensor([100.0])})

    outputs = pipeline.run(store, [torch.tensor([1.0]), torch.tensor([2.0])])

    assert [out.item() for out in outputs] == [1.0, 3.0]
    assert torch.equal(store.get("sum.state.in"), torch.tensor([3.0]))


def test_global_local_pipeline_is_tiled_fusion_not_streaming() -> None:
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
    pipeline = GlobalLocalPipeline(
        global_graph=global_graph,
        tile_pipeline=tile_pipeline,
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
            "global.bias": torch.tensor(10.0),
            "local.bias": torch.tensor(2.0),
        }
    )

    out = pipeline.run(store)

    assert torch.equal(out, x + 6.0)
    assert torch.equal(store.get("fused.out"), x + 6.0)
    assert torch.equal(store.get("global.out"), x + 10.0)

    # GlobalLocalPipeline fuses global and tiled local branches; it does not
    # perform stream state carry.
    assert not store.has("sum.state.in")
    assert not store.has("sum.state.out")
