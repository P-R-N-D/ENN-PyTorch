from __future__ import annotations

import pytest
import torch
from torch import nn

from enn_torch_dev.executor import (
    GraphExecutor,
    KVStore,
    KeyRef,
    NodeSpec,
    TileExecutor,
    TileSpec,
)


class _AddBias(nn.Module):
    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return x + bias


class _AddIndex(nn.Module):
    def forward(self, x: torch.Tensor, index: int) -> torch.Tensor:
        return x + float(index)


class _AddMeta(nn.Module):
    def forward(self, x: torch.Tensor, meta: float | int | None) -> torch.Tensor:
        return x + float(meta or 0)


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


def test_tile_executor_runs_graph_for_each_tile() -> None:
    graph = _make_bias_graph()
    executor = TileExecutor(
        graph,
        TileSpec(
            tile_input_key="tile.x",
            output_names=["root"],
        ),
    )
    base_store = KVStore({"global.bias": torch.tensor([10.0])})

    results = executor.run(
        base_store,
        [torch.tensor([1.0]), torch.tensor([2.0])],
    )

    assert len(results) == 2
    assert torch.equal(results[0]["root"], torch.tensor([11.0]))
    assert torch.equal(results[1]["root"], torch.tensor([12.0]))


def test_tile_executor_does_not_pollute_base_store() -> None:
    graph = _make_bias_graph()
    executor = TileExecutor(
        graph,
        TileSpec(
            tile_input_key="tile.x",
            output_names=["root"],
        ),
    )
    base_store = KVStore({"global.bias": torch.tensor([10.0])})

    executor.run(base_store, [torch.tensor([1.0])])

    assert not base_store.has("tile.x")
    assert not base_store.has("root.out")
    assert base_store.has("global.bias")


def test_tile_executor_can_collect_by_output_key() -> None:
    graph = _make_bias_graph()
    executor = TileExecutor(
        graph,
        TileSpec(
            tile_input_key="tile.x",
            output_names=["root"],
            output_by="key",
        ),
    )
    base_store = KVStore({"global.bias": torch.tensor([10.0])})

    results = executor.run(base_store, [torch.tensor([1.0])])

    assert set(results[0]) == {"root.out"}
    assert torch.equal(results[0]["root.out"], torch.tensor([11.0]))


def test_tile_executor_uses_root_outputs_when_output_names_omitted() -> None:
    graph = _make_bias_graph()
    executor = TileExecutor(
        graph,
        TileSpec(tile_input_key="tile.x"),
    )
    base_store = KVStore({"global.bias": torch.tensor([10.0])})

    results = executor.run(base_store, [torch.tensor([3.0])])

    assert set(results[0]) == {"root"}
    assert torch.equal(results[0]["root"], torch.tensor([13.0]))


def test_tile_index_key_is_written_to_tile_store() -> None:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="root",
            input_args=[KeyRef("tile.x")],
            input_kwargs={"index": KeyRef("tile.index")},
            output_key="root.out",
        ),
        _AddIndex(),
    )
    executor = TileExecutor(
        graph,
        TileSpec(
            tile_input_key="tile.x",
            output_names=["root"],
            tile_index_key="tile.index",
        ),
    )

    results = executor.run(
        KVStore(),
        [torch.tensor([10.0]), torch.tensor([10.0])],
    )

    assert torch.equal(results[0]["root"], torch.tensor([10.0]))
    assert torch.equal(results[1]["root"], torch.tensor([11.0]))


def test_tile_meta_key_is_written_to_tile_store() -> None:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="root",
            input_args=[KeyRef("tile.x")],
            input_kwargs={"meta": KeyRef("tile.meta")},
            output_key="root.out",
        ),
        _AddMeta(),
    )
    executor = TileExecutor(
        graph,
        TileSpec(
            tile_input_key="tile.x",
            output_names=["root"],
            tile_meta_key="tile.meta",
        ),
    )

    results = executor.run(
        KVStore(),
        [torch.tensor([10.0]), torch.tensor([10.0])],
        metas=[0, 2],
    )

    assert torch.equal(results[0]["root"], torch.tensor([10.0]))
    assert torch.equal(results[1]["root"], torch.tensor([12.0]))


def test_run_tile_supports_single_tile_execution() -> None:
    graph = _make_bias_graph()
    executor = TileExecutor(
        graph,
        TileSpec(tile_input_key="tile.x", output_names=["root"]),
    )

    result = executor.run_tile(
        KVStore({"global.bias": torch.tensor([10.0])}),
        torch.tensor([5.0]),
        index=7,
    )

    assert torch.equal(result["root"], torch.tensor([15.0]))


def test_tile_executor_empty_tiles_returns_empty_list() -> None:
    graph = _make_bias_graph()
    executor = TileExecutor(
        graph,
        TileSpec(tile_input_key="tile.x", output_names=["root"]),
    )

    assert executor.run(KVStore({"global.bias": torch.tensor([10.0])}), []) == []


def test_tile_executor_validates_inputs() -> None:
    graph = _make_bias_graph()

    with pytest.raises(ValueError, match="output_by"):
        TileSpec(tile_input_key="tile.x", output_by="bad")

    with pytest.raises(TypeError, match="output_names"):
        TileSpec(tile_input_key="tile.x", output_names="root")

    with pytest.raises(ValueError, match="duplicate"):
        TileSpec(tile_input_key="tile.x", output_names=["root", "root"])

    with pytest.raises(TypeError, match="tiles"):
        TileExecutor(
            graph,
            TileSpec(tile_input_key="tile.x", output_names=["root"]),
        ).run(KVStore({"global.bias": torch.tensor([10.0])}), None)

    with pytest.raises(TypeError, match="metas"):
        TileExecutor(
            graph,
            TileSpec(tile_input_key="tile.x", output_names=["root"]),
        ).run(KVStore({"global.bias": torch.tensor([10.0])}), [torch.tensor([1.0])], metas="m")

    with pytest.raises(ValueError, match="metas length"):
        TileExecutor(
            graph,
            TileSpec(tile_input_key="tile.x", output_names=["root"]),
        ).run(KVStore({"global.bias": torch.tensor([10.0])}), [torch.tensor([1.0])], metas=[])

    with pytest.raises(TypeError, match="GraphExecutor"):
        TileExecutor(object(), TileSpec(tile_input_key="tile.x"))
