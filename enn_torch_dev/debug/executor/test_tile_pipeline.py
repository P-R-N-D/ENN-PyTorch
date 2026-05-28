from __future__ import annotations

import pytest
import torch
from torch import nn

from enn_torch_dev.executor import (
    GraphExecutor,
    KVStore,
    KeyRef,
    NodeSpec,
    TilePipeline,
    TilePipelineSpec,
    TilePolicy,
    TileReconstructor,
)


class _AddBias(nn.Module):
    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return x + bias


class _AddIndexAndMetaStart(nn.Module):
    def forward(self, x: torch.Tensor, index: int, meta: object) -> torch.Tensor:
        return x + float(index) + float(meta.start[0])


def _make_bias_graph() -> GraphExecutor:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="root",
            input_args=[KeyRef("tile.x")],
            input_kwargs={"bias": KeyRef("global.bias")},
            output_key="root.out",
        ),
        _AddBias(),
    )
    return graph


def test_tile_pipeline_split_execute_reconstructs_tensor() -> None:
    graph = _make_bias_graph()
    pipeline = TilePipeline(
        graph,
        TilePolicy(tile_shape=(2,)),
        TilePipelineSpec(
            input_key="x",
            tile_input_key="tile.x",
            output_name="root",
        ),
    )
    store = KVStore(
        {
            "x": torch.arange(5, dtype=torch.float32),
            "global.bias": torch.tensor(10.0),
        }
    )

    out = pipeline.run(store)

    assert torch.equal(out, torch.arange(5, dtype=torch.float32) + 10.0)


def test_tile_pipeline_writes_reconstructed_output_key() -> None:
    graph = _make_bias_graph()
    pipeline = TilePipeline(
        graph,
        TilePolicy(tile_shape=(2,)),
        TilePipelineSpec(
            input_key="x",
            tile_input_key="tile.x",
            output_name="root",
            output_key="tiled.out",
        ),
    )
    store = KVStore(
        {
            "x": torch.arange(5, dtype=torch.float32),
            "global.bias": torch.tensor(1.0),
        }
    )

    out = pipeline.run(store)

    assert store.has("tiled.out")
    assert torch.equal(store.get("tiled.out"), out)
    assert not store.has("tile.x")
    assert not store.has("root.out")


def test_tile_pipeline_output_by_key() -> None:
    graph = _make_bias_graph()
    pipeline = TilePipeline(
        graph,
        TilePolicy(tile_shape=(2,)),
        TilePipelineSpec(
            input_key="x",
            tile_input_key="tile.x",
            output_name="root",
            output_by="key",
        ),
    )
    store = KVStore(
        {
            "x": torch.arange(4, dtype=torch.float32),
            "global.bias": torch.tensor(2.0),
        }
    )

    out = pipeline.run(store)

    assert torch.equal(out, torch.arange(4, dtype=torch.float32) + 2.0)


def test_tile_pipeline_forwards_index_and_meta() -> None:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="root",
            input_args=[KeyRef("tile.x")],
            input_kwargs={
                "index": KeyRef("tile.index"),
                "meta": KeyRef("tile.meta"),
            },
            output_key="root.out",
        ),
        _AddIndexAndMetaStart(),
    )
    pipeline = TilePipeline(
        graph,
        TilePolicy(tile_shape=(2,), stride=(2,)),
        TilePipelineSpec(
            input_key="x",
            tile_input_key="tile.x",
            output_name="root",
            tile_index_key="tile.index",
            tile_meta_key="tile.meta",
        ),
    )
    store = KVStore({"x": torch.zeros(4, dtype=torch.float32)})

    out = pipeline.run(store)

    assert torch.equal(out, torch.tensor([0.0, 0.0, 3.0, 3.0]))


def test_tile_pipeline_allows_custom_reconstructor() -> None:
    graph = _make_bias_graph()
    pipeline = TilePipeline(
        graph,
        TilePolicy(tile_shape=(3,), stride=(2,)),
        TilePipelineSpec(
            input_key="x",
            tile_input_key="tile.x",
            output_name="root",
        ),
        tile_reconstructor=TileReconstructor(),
    )
    store = KVStore(
        {
            "x": torch.arange(5, dtype=torch.float32),
            "global.bias": torch.tensor(0.0),
        }
    )

    out = pipeline.run(store)

    assert out.shape == (5,)


def test_tile_pipeline_rejects_empty_tile_split() -> None:
    graph = _make_bias_graph()
    pipeline = TilePipeline(
        graph,
        TilePolicy(tile_shape=(2,)),
        TilePipelineSpec(
            input_key="x",
            tile_input_key="tile.x",
            output_name="root",
        ),
    )

    with pytest.raises(ValueError, match="no tiles"):
        pipeline.run(KVStore({"x": torch.empty(0), "global.bias": torch.tensor(0.0)}))


def test_tile_pipeline_rejects_non_tensor_input_via_policy() -> None:
    graph = _make_bias_graph()
    pipeline = TilePipeline(
        graph,
        TilePolicy(tile_shape=(2,)),
        TilePipelineSpec(
            input_key="x",
            tile_input_key="tile.x",
            output_name="root",
        ),
    )

    with pytest.raises(TypeError, match="Tensor"):
        pipeline.run(KVStore({"x": [1, 2, 3], "global.bias": torch.tensor(0.0)}))


def test_tile_pipeline_rejects_unknown_output_name() -> None:
    graph = _make_bias_graph()

    with pytest.raises(KeyError, match="missing"):
        TilePipeline(
            graph,
            TilePolicy(tile_shape=(2,)),
            TilePipelineSpec(
                input_key="x",
                tile_input_key="tile.x",
                output_name="missing",
            ),
        )


def test_tile_pipeline_validates_inputs() -> None:
    graph = _make_bias_graph()

    with pytest.raises(ValueError, match="output_by"):
        TilePipelineSpec(
            input_key="x",
            tile_input_key="tile.x",
            output_name="root",
            output_by="bad",
        )

    with pytest.raises(TypeError, match="GraphExecutor"):
        TilePipeline(
            object(),
            TilePolicy(tile_shape=(2,)),
            TilePipelineSpec(
                input_key="x",
                tile_input_key="tile.x",
                output_name="root",
            ),
        )
