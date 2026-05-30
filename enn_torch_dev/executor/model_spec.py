from __future__ import annotations

from dataclasses import dataclass

from .global_local import GlobalLocalPipeline
from .graph import GraphExecutor
from .modes import ExecutorModeSpec
from .plan import ExecutorPlan
from .stream import StreamPipeline
from .tile_pipeline import TilePipeline


_CONTEXTS = {"local", "global_local"}


def _validate_context(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("ModelExecutionSpec.context must be a string.")
    if value not in _CONTEXTS:
        raise ValueError(
            "ModelExecutionSpec.context must be either 'local' or 'global_local'."
        )
    return value


def _validate_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"ModelExecutionSpec.{field_name} must be a bool.")
    return value


def _is_sequence(value: object) -> bool:
    return value is not None and not isinstance(value, (str, bytes, bytearray))


def _normalize_positive_int_tuple(
    value: object,
    *,
    field_name: str,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not _is_sequence(value):
        raise TypeError(
            f"ModelExecutionSpec.{field_name} must be a sequence of positive integers."
        )

    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"ModelExecutionSpec.{field_name} must be a sequence of positive integers."
        ) from exc

    if not items:
        raise ValueError(f"ModelExecutionSpec.{field_name} must not be empty.")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in items):
        raise TypeError(f"ModelExecutionSpec.{field_name} must contain integers only.")
    if not all(item > 0 for item in items):
        raise ValueError(f"ModelExecutionSpec.{field_name} values must be positive.")
    return items


def _normalize_int_tuple(
    value: object,
    *,
    field_name: str,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not _is_sequence(value):
        raise TypeError(
            f"ModelExecutionSpec.{field_name} must be a sequence of integers."
        )

    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"ModelExecutionSpec.{field_name} must be a sequence of integers."
        ) from exc

    if not items:
        raise ValueError(f"ModelExecutionSpec.{field_name} must not be empty.")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in items):
        raise TypeError(f"ModelExecutionSpec.{field_name} must contain integers only.")
    return items


@dataclass(slots=True)
class ModelExecutionSpec:
    """
    Public ``Model(...)`` execution parameter schema.

    This is a naming layer over ``ExecutorModeSpec``. Public model APIs should
    use ``context`` / ``tile`` / ``stateful`` instead of exposing ``tile``,
    ``stream``, and ``global_local`` as unrelated peer flags.
    """

    context: str = "local"
    tile: bool = False
    stateful: bool = False
    tile_shape: tuple[int, ...] | None = None
    tile_stride: tuple[int, ...] | None = None
    tile_dims: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        self.context = _validate_context(self.context)
        self.tile = _validate_bool(self.tile, "tile")
        self.stateful = _validate_bool(self.stateful, "stateful")
        self.tile_shape = _normalize_positive_int_tuple(
            self.tile_shape,
            field_name="tile_shape",
        )
        self.tile_stride = _normalize_positive_int_tuple(
            self.tile_stride,
            field_name="tile_stride",
        )
        self.tile_dims = _normalize_int_tuple(
            self.tile_dims,
            field_name="tile_dims",
        )

        if self.tile:
            if self.tile_shape is None:
                raise ValueError("ModelExecutionSpec.tile=True requires tile_shape.")
            if self.tile_stride is not None and len(self.tile_stride) != len(
                self.tile_shape
            ):
                raise ValueError(
                    "ModelExecutionSpec.tile_stride length must match tile_shape length."
                )
            if self.tile_dims is not None and len(self.tile_dims) != len(self.tile_shape):
                raise ValueError(
                    "ModelExecutionSpec.tile_dims length must match tile_shape length."
                )
        elif (
            self.tile_shape is not None
            or self.tile_stride is not None
            or self.tile_dims is not None
        ):
            raise ValueError(
                "ModelExecutionSpec tile_shape, tile_stride, and tile_dims require tile=True."
            )

        if self.context == "global_local" and not self.tile:
            raise ValueError(
                "ModelExecutionSpec context='global_local' requires tile=True in v0."
            )

    @property
    def uses_global_local(self) -> bool:
        return self.context == "global_local"

    @property
    def executor_mode(self) -> ExecutorModeSpec:
        return ExecutorModeSpec(
            tile=self.tile,
            stream=self.stateful,
            global_local=self.uses_global_local,
        )

    def to_executor_mode_spec(self) -> ExecutorModeSpec:
        return self.executor_mode

    def create_plan(
        self,
        *,
        graph: GraphExecutor | None = None,
        tile_pipeline: TilePipeline | None = None,
        stream_pipeline: StreamPipeline | None = None,
        global_local_pipeline: GlobalLocalPipeline | None = None,
    ) -> ExecutorPlan:
        """Create an ``ExecutorPlan`` from this public model execution spec."""
        return ExecutorPlan(
            mode=self.executor_mode,
            graph=graph,
            tile_pipeline=tile_pipeline,
            stream_pipeline=stream_pipeline,
            global_local_pipeline=global_local_pipeline,
        )
