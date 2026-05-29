from __future__ import annotations

from dataclasses import dataclass

from .global_local import GlobalLocalPipeline
from .graph import GraphExecutor
from .modes import ExecutorModeSpec
from .stream import StreamPipeline
from .tile_pipeline import TilePipeline


def _validate_component(
    value: object,
    expected_type: type[object],
    field_name: str,
) -> None:
    if value is not None and not isinstance(value, expected_type):
        raise TypeError(
            f"ExecutorPlan.{field_name} must be {expected_type.__name__} or None."
        )


@dataclass(slots=True)
class ExecutorPlan:
    """
    Validated mapping from an ``ExecutorModeSpec`` to executor components.

    This class does not run anything and does not construct graphs or pipelines.
    It only validates that the caller supplied the executor objects required by
    the declared mode. When stream is enabled, stream is the outer execution
    layer. Tile or global/local execution is interpreted as the per-chunk inner
    layer.
    """

    mode: ExecutorModeSpec
    graph: GraphExecutor | None = None
    tile_pipeline: TilePipeline | None = None
    stream_pipeline: StreamPipeline | None = None
    global_local_pipeline: GlobalLocalPipeline | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ExecutorModeSpec):
            raise TypeError(
                f"ExecutorPlan.mode must be ExecutorModeSpec, got {type(self.mode)!r}"
            )

        _validate_component(self.graph, GraphExecutor, "graph")
        _validate_component(self.tile_pipeline, TilePipeline, "tile_pipeline")
        _validate_component(self.stream_pipeline, StreamPipeline, "stream_pipeline")
        _validate_component(
            self.global_local_pipeline,
            GlobalLocalPipeline,
            "global_local_pipeline",
        )

        if self.mode.is_plain:
            if self.graph is None:
                raise ValueError("ExecutorPlan plain mode requires graph.")
        elif self.graph is not None:
            raise ValueError("ExecutorPlan.graph is only valid for plain mode.")

        if self.mode.stream:
            if self.stream_pipeline is None:
                raise ValueError("ExecutorPlan stream mode requires stream_pipeline.")
        elif self.stream_pipeline is not None:
            raise ValueError("ExecutorPlan.stream_pipeline requires stream=True.")

        if self.mode.global_local:
            if self.global_local_pipeline is None:
                raise ValueError(
                    "ExecutorPlan global_local mode requires global_local_pipeline."
                )
            if self.tile_pipeline is not None:
                raise ValueError(
                    "ExecutorPlan global_local mode uses its embedded TilePipeline; "
                    "tile_pipeline must be None."
                )
        else:
            if self.global_local_pipeline is not None:
                raise ValueError(
                    "ExecutorPlan.global_local_pipeline requires global_local=True."
                )
            if self.mode.tile:
                if self.tile_pipeline is None:
                    raise ValueError("ExecutorPlan tile mode requires tile_pipeline.")
            elif self.tile_pipeline is not None:
                raise ValueError("ExecutorPlan.tile_pipeline requires tile=True.")

    @property
    def is_plain(self) -> bool:
        return self.mode.is_plain

    @property
    def execution_layers(self) -> tuple[str, ...]:
        if self.mode.is_plain:
            return ("graph",)

        layers: list[str] = []
        if self.mode.stream:
            layers.append("stream")
        if self.mode.global_local:
            layers.append("global_local")
        elif self.mode.tile:
            layers.append("tile")
        return tuple(layers)

    @property
    def component_names(self) -> tuple[str, ...]:
        if self.mode.is_plain:
            return ("graph",)

        names: list[str] = []
        if self.mode.stream:
            names.append("stream_pipeline")
        if self.mode.global_local:
            names.append("global_local_pipeline")
        elif self.mode.tile:
            names.append("tile_pipeline")
        return tuple(names)
