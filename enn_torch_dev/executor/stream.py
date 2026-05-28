from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .graph import GraphExecutor
from .state import StateRoute
from .store import KVStore


def _validate_stream_key(value: object, field_name: str) -> str:
    label = f"StreamPipelineSpec.{field_name}"
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    if not value:
        raise ValueError(f"{label} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{label} must not have leading or trailing whitespace.")
    return value


def _validate_stream_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"StreamPipelineSpec.{field_name} must be a bool.")
    return value


def _normalize_chunks(chunks: object) -> tuple[Any, ...]:
    if chunks is None or isinstance(chunks, (str, bytes, bytearray)):
        raise TypeError("chunks must be a sequence of chunk values.")
    try:
        return tuple(chunks)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("chunks must be a sequence of chunk values.") from exc


def _normalize_state_routes(routes: object) -> tuple[StateRoute, ...]:
    if routes is None:
        return ()
    if isinstance(routes, (str, bytes, bytearray)):
        raise TypeError("state_routes must be a sequence of StateRoute values.")
    try:
        values = tuple(routes)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("state_routes must be a sequence of StateRoute values.") from exc

    for route in values:
        if not isinstance(route, StateRoute):
            raise TypeError(
                f"state_routes must contain StateRoute values, got {type(route)!r}"
            )
    return values


@dataclass(slots=True)
class StreamPipelineSpec:
    """
    Sequential chunk execution schema.

    A stream pipeline differs from tiled execution in that chunks are processed
    in order and state routes are carried after each step. It does not split the
    input into chunks; callers provide an already ordered chunk sequence.
    """

    chunk_input_key: str
    output_name: str
    output_by: str = "node"
    chunk_index_key: str | None = None
    outputs_key: str | None = None
    state_detach: bool = False
    state_clone: bool = False

    def __post_init__(self) -> None:
        self.chunk_input_key = _validate_stream_key(
            self.chunk_input_key,
            "chunk_input_key",
        )
        self.output_name = _validate_stream_key(self.output_name, "output_name")
        if self.output_by not in {"node", "key"}:
            raise ValueError(
                "StreamPipelineSpec.output_by must be either 'node' or 'key'."
            )
        if self.chunk_index_key is not None:
            self.chunk_index_key = _validate_stream_key(
                self.chunk_index_key,
                "chunk_index_key",
            )
        if self.outputs_key is not None:
            self.outputs_key = _validate_stream_key(
                self.outputs_key,
                "outputs_key",
            )
        self.state_detach = _validate_stream_bool(self.state_detach, "state_detach")
        self.state_clone = _validate_stream_bool(self.state_clone, "state_clone")


class StreamPipeline:
    """
    Run a graph over an ordered sequence of chunks.

    This executor-layer pipeline only provides sequential execution and explicit
    state carry through ``StateRoute``. It does not create chunks, reset state,
    detach tensors, cache per-stream state, or implement truncated BPTT.
    """

    def __init__(
        self,
        graph: GraphExecutor,
        spec: StreamPipelineSpec,
        *,
        state_routes: Sequence[StateRoute] = (),
    ) -> None:
        if not isinstance(graph, GraphExecutor):
            raise TypeError(f"StreamPipeline graph must be GraphExecutor, got {type(graph)!r}")
        if not isinstance(spec, StreamPipelineSpec):
            raise TypeError(f"StreamPipeline spec must be StreamPipelineSpec, got {type(spec)!r}")

        # Validate early that the requested output node exists.
        graph.output_key(spec.output_name)

        self.graph = graph
        self.spec = spec
        self.state_routes = _normalize_state_routes(state_routes)

    def _result_key(self) -> str:
        if self.spec.output_by == "node":
            return self.spec.output_name
        return self.graph.output_key(self.spec.output_name)

    def run(
        self,
        store: KVStore,
        chunks: Sequence[Any],
    ) -> list[Any]:
        if not isinstance(store, KVStore):
            raise TypeError(f"StreamPipeline.run expects KVStore, got {type(store)!r}")

        normalized_chunks = _normalize_chunks(chunks)
        result_key = self._result_key()
        outputs: list[Any] = []

        for index, chunk in enumerate(normalized_chunks):
            chunk_store = store.fork()
            chunk_store.set(self.spec.chunk_input_key, chunk, origin="StreamPipeline")
            if self.spec.chunk_index_key is not None:
                chunk_store.set(self.spec.chunk_index_key, index, origin="StreamPipeline")

            for route in self.state_routes:
                route.enable_return_state(chunk_store)

            self.graph.run(chunk_store)
            result = self.graph.collect_outputs(
                chunk_store,
                names=[self.spec.output_name],
                by=self.spec.output_by,
            )[result_key]
            outputs.append(result)

            for route in self.state_routes:
                route.carry(
                    chunk_store,
                    detach=self.spec.state_detach,
                    clone=self.spec.state_clone,
                )
                store.set_value(
                    route.state_input_key,
                    chunk_store.get_value(route.state_input_key),
                )

        if self.spec.outputs_key is not None:
            store.set(self.spec.outputs_key, outputs, origin="StreamPipeline")
        return outputs
