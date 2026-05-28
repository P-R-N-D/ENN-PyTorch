from __future__ import annotations

import pytest
import torch
from torch import nn

from enn_torch_dev.executor import (
    GraphExecutor,
    KVStore,
    KeyRef,
    NodeSpec,
    StateRoute,
    StreamPipeline,
    StreamPipelineSpec,
)


class _Double(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0


class _AddIndex(nn.Module):
    def forward(self, x: torch.Tensor, index: int) -> torch.Tensor:
        return x + float(index)


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


class _AddOptionalValue(nn.Module):
    def forward(self, x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return x + value


class _MakeOptionalValue(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 100.0


def _make_double_graph() -> GraphExecutor:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="double_node",
            input_args=[KeyRef("chunk.x")],
            output_key="double.out",
        ),
        _Double(),
    )
    return graph


def test_stream_pipeline_runs_chunks_in_order() -> None:
    pipeline = StreamPipeline(
        _make_double_graph(),
        StreamPipelineSpec(
            chunk_input_key="chunk.x",
            output_name="double_node",
        ),
    )
    store = KVStore()

    outputs = pipeline.run(
        store,
        [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])],
    )

    assert [out.item() for out in outputs] == [2.0, 4.0, 6.0]
    assert not store.has("chunk.x")
    assert not store.has("double.out")


def test_stream_pipeline_writes_outputs_key() -> None:
    pipeline = StreamPipeline(
        _make_double_graph(),
        StreamPipelineSpec(
            chunk_input_key="chunk.x",
            output_name="double_node",
            outputs_key="stream.outputs",
        ),
    )
    store = KVStore()

    outputs = pipeline.run(store, [torch.tensor([1.0]), torch.tensor([2.0])])

    assert store.has("stream.outputs")
    assert store.get("stream.outputs") is outputs
    assert [out.item() for out in store.get("stream.outputs")] == [2.0, 4.0]


def test_stream_pipeline_supports_output_by_key() -> None:
    pipeline = StreamPipeline(
        _make_double_graph(),
        StreamPipelineSpec(
            chunk_input_key="chunk.x",
            output_name="double_node",
            output_by="key",
        ),
    )
    store = KVStore()

    outputs = pipeline.run(store, [torch.tensor([2.0])])

    assert torch.equal(outputs[0], torch.tensor([4.0]))


def test_stream_pipeline_forwards_chunk_index() -> None:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="add_index",
            input_args=[KeyRef("chunk.x")],
            input_kwargs={"index": KeyRef("chunk.index")},
            output_key="add_index.out",
        ),
        _AddIndex(),
    )
    pipeline = StreamPipeline(
        graph,
        StreamPipelineSpec(
            chunk_input_key="chunk.x",
            output_name="add_index",
            chunk_index_key="chunk.index",
        ),
    )
    store = KVStore()

    outputs = pipeline.run(
        store,
        [torch.tensor([10.0]), torch.tensor([10.0]), torch.tensor([10.0])],
    )

    assert [out.item() for out in outputs] == [10.0, 11.0, 12.0]
    assert not store.has("chunk.index")


def test_stream_pipeline_carries_state_between_chunks() -> None:
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
        ),
        state_routes=[route],
    )
    store = KVStore()

    outputs = pipeline.run(
        store,
        [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])],
    )

    assert [out.item() for out in outputs] == [1.0, 3.0, 6.0]
    assert torch.equal(store.get("sum.state.in"), torch.tensor([6.0]))
    assert not store.has("sum.state.out")


def test_stream_pipeline_isolates_stale_optional_outputs_between_chunks() -> None:
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="consumer",
            input_args=[KeyRef("chunk.x")],
            input_kwargs={
                "value": KeyRef(
                    "producer.out",
                    optional=True,
                    default=torch.tensor([0.0]),
                )
            },
            output_key="consumer.out",
        ),
        _AddOptionalValue(),
    )
    graph.add_node(
        NodeSpec(
            name="producer",
            input_args=[KeyRef("chunk.x")],
            output_key="producer.out",
        ),
        _MakeOptionalValue(),
    )
    pipeline = StreamPipeline(
        graph,
        StreamPipelineSpec(
            chunk_input_key="chunk.x",
            output_name="consumer",
        ),
    )
    store = KVStore()

    outputs = pipeline.run(store, [torch.tensor([1.0]), torch.tensor([2.0])])

    assert [out.item() for out in outputs] == [1.0, 2.0]
    assert not store.has("producer.out")
    assert not store.has("consumer.out")


def test_stream_pipeline_forces_return_state_for_state_routes() -> None:
    route = StateRoute("sum.state.in", "sum.state.out", return_state_key="return_state")
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
        StreamPipelineSpec(chunk_input_key="chunk.x", output_name="sum"),
        state_routes=[route],
    )
    store = KVStore({"return_state": False})

    outputs = pipeline.run(store, [torch.tensor([1.0]), torch.tensor([2.0])])

    assert [out.item() for out in outputs] == [1.0, 3.0]
    assert torch.equal(store.get("sum.state.in"), torch.tensor([3.0]))
    assert store.get("return_state") is False


def test_stream_pipeline_can_detach_carried_state() -> None:
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
            state_detach=True,
        ),
        state_routes=[route],
    )
    store = KVStore()

    pipeline.run(store, [torch.ones(1, requires_grad=True)])

    assert not store.get("sum.state.in").requires_grad


def test_stream_pipeline_can_clone_carried_state() -> None:
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
            state_clone=True,
        ),
        state_routes=[route],
    )
    store = KVStore()

    outputs = pipeline.run(store, [torch.ones(1)])

    assert torch.equal(store.get("sum.state.in"), outputs[0])
    assert store.get("sum.state.in").data_ptr() != outputs[0].data_ptr()


def test_stream_pipeline_allows_empty_chunks() -> None:
    pipeline = StreamPipeline(
        _make_double_graph(),
        StreamPipelineSpec(
            chunk_input_key="chunk.x",
            output_name="double_node",
            outputs_key="stream.outputs",
        ),
    )
    store = KVStore()

    outputs = pipeline.run(store, [])

    assert outputs == []
    assert store.get("stream.outputs") == []
    assert not store.has("chunk.x")


def test_stream_pipeline_rejects_unknown_output_name() -> None:
    with pytest.raises(KeyError, match="missing"):
        StreamPipeline(
            _make_double_graph(),
            StreamPipelineSpec(
                chunk_input_key="chunk.x",
                output_name="missing",
            ),
        )


def test_stream_pipeline_validates_inputs() -> None:
    graph = _make_double_graph()

    with pytest.raises(ValueError, match="output_by"):
        StreamPipelineSpec(
            chunk_input_key="chunk.x",
            output_name="double_node",
            output_by="bad",
        )

    with pytest.raises(TypeError, match="state_detach"):
        StreamPipelineSpec(
            chunk_input_key="chunk.x",
            output_name="double_node",
            state_detach=1,
        )

    with pytest.raises(TypeError, match="state_clone"):
        StreamPipelineSpec(chunk_input_key="chunk.x", output_name="double_node", state_clone=1)

    with pytest.raises(TypeError, match="GraphExecutor"):
        StreamPipeline(
            object(),
            StreamPipelineSpec(chunk_input_key="chunk.x", output_name="double_node"),
        )

    with pytest.raises(TypeError, match="state_routes"):
        StreamPipeline(
            graph,
            StreamPipelineSpec(chunk_input_key="chunk.x", output_name="double_node"),
            state_routes=[object()],
        )

    pipeline = StreamPipeline(
        graph,
        StreamPipelineSpec(chunk_input_key="chunk.x", output_name="double_node"),
    )
    with pytest.raises(TypeError, match="KVStore"):
        pipeline.run(object(), [])
    with pytest.raises(TypeError, match="chunks"):
        pipeline.run(KVStore(), None)
    with pytest.raises(TypeError, match="chunks"):
        pipeline.run(KVStore(), "abc")
