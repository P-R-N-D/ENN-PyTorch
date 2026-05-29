from __future__ import annotations

from dataclasses import dataclass


def _validate_mode_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"ExecutorModeSpec.{field_name} must be a bool.")
    return value


@dataclass(slots=True)
class ExecutorModeSpec:
    """
    Declarative executor mode flags for higher-level model wrappers.

    This spec does not build graphs or pipelines. It only records which
    executor-side modes a caller intends to compose.

    ``global_local`` requires ``tile`` because ``GlobalLocalPipeline`` combines
    a global graph with a tiled/local branch. ``tile`` and ``stream`` may both be
    enabled for future compositions where ordered stream chunks are processed
    with spatial/local tiling inside each chunk.
    """

    tile: bool = False
    stream: bool = False
    global_local: bool = False

    def __post_init__(self) -> None:
        self.tile = _validate_mode_bool(self.tile, "tile")
        self.stream = _validate_mode_bool(self.stream, "stream")
        self.global_local = _validate_mode_bool(self.global_local, "global_local")

        if self.global_local and not self.tile:
            raise ValueError("ExecutorModeSpec.global_local requires tile=True.")

    @property
    def is_plain(self) -> bool:
        return not self.tile and not self.stream and not self.global_local

    @property
    def mode_names(self) -> tuple[str, ...]:
        if self.is_plain:
            return ("plain",)

        names: list[str] = []
        if self.tile:
            names.append("tile")
        if self.stream:
            names.append("stream")
        if self.global_local:
            names.append("global_local")
        return tuple(names)
