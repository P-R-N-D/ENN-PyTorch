from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .graph import GraphExecutor
from .store import KVStore


def _validate_tile_key(value: object, field_name: str) -> str:
    label = f"TileSpec.{field_name}"
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    if not value:
        raise ValueError(f"{label} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(
            f"{label} must not have leading or trailing whitespace."
        )
    return value


def _normalize_output_names(value: object) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("TileSpec.output_names must be a sequence of strings.")

    try:
        names = list(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            "TileSpec.output_names must be a sequence of strings."
        ) from exc

    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        normalized = _validate_tile_key(name, "output_names")
        if normalized in seen:
            raise ValueError(f"TileSpec.output_names contains duplicate name: {normalized!r}")
        seen.add(normalized)
        out.append(normalized)
    return out


@dataclass(slots=True)
class TileSpec:
    """
    Tile-wise graph execution schema.

    ``tile_input_key`` is written into each forked tile store before the graph
    runs. ``output_names`` is forwarded to ``GraphExecutor.collect_outputs``;
    when omitted, graph root nodes are collected.
    """

    tile_input_key: str
    output_names: list[str] | None = None
    output_by: str = "node"
    tile_index_key: str | None = None
    tile_meta_key: str | None = None

    def __post_init__(self) -> None:
        self.tile_input_key = _validate_tile_key(
            self.tile_input_key,
            "tile_input_key",
        )
        self.output_names = _normalize_output_names(self.output_names)
        if self.output_by not in {"node", "key"}:
            raise ValueError("TileSpec.output_by must be either 'node' or 'key'.")
        if self.tile_index_key is not None:
            self.tile_index_key = _validate_tile_key(
                self.tile_index_key,
                "tile_index_key",
            )
        if self.tile_meta_key is not None:
            self.tile_meta_key = _validate_tile_key(
                self.tile_meta_key,
                "tile_meta_key",
            )


class TileExecutor:
    """
    Run one graph over a sequence of already-split tiles.

    This executor does not split tensors, reconstruct tiled outputs, or perform
    local/global gating. It only provides the common tile-wise execution shell:
    fork a ``KVStore``, inject tile values, run ``GraphExecutor``, and collect
    requested outputs.
    """

    def __init__(self, graph: GraphExecutor, spec: TileSpec) -> None:
        if not isinstance(graph, GraphExecutor):
            raise TypeError(f"TileExecutor graph must be GraphExecutor, got {type(graph)!r}")
        if not isinstance(spec, TileSpec):
            raise TypeError(f"TileExecutor spec must be TileSpec, got {type(spec)!r}")

        self.graph = graph
        self.spec = spec

    @staticmethod
    def _normalize_tiles(tiles: object) -> tuple[Any, ...]:
        if isinstance(tiles, (str, bytes, bytearray)) or tiles is None:
            raise TypeError("tiles must be a sequence of tile values.")

        try:
            return tuple(tiles)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("tiles must be a sequence of tile values.") from exc

    @staticmethod
    def _normalize_metas(
        metas: Sequence[Any] | None,
        *,
        expected_len: int,
    ) -> tuple[Any, ...]:
        if metas is None:
            return tuple(None for _ in range(expected_len))
        if isinstance(metas, (str, bytes, bytearray)):
            raise TypeError("metas must be a sequence of metadata values.")

        values = tuple(metas)
        if len(values) != expected_len:
            raise ValueError(
                f"metas length must match tiles length: {len(values)} != {expected_len}"
            )
        return values

    def run_tile(
        self,
        base_store: KVStore,
        tile: Any,
        *,
        index: int | None = None,
        meta: Any = None,
    ) -> dict[str, Any]:
        if not isinstance(base_store, KVStore):
            raise TypeError(f"base_store must be KVStore, got {type(base_store)!r}")

        tile_store = base_store.fork()
        tile_store.set(self.spec.tile_input_key, tile)
        if self.spec.tile_index_key is not None:
            tile_store.set(self.spec.tile_index_key, index)
        if self.spec.tile_meta_key is not None:
            tile_store.set(self.spec.tile_meta_key, meta)

        self.graph.run(tile_store)
        return self.graph.collect_outputs(
            tile_store,
            names=self.spec.output_names,
            by=self.spec.output_by,
        )

    def run(
        self,
        base_store: KVStore,
        tiles: Sequence[Any],
        *,
        metas: Sequence[Any] | None = None,
    ) -> list[dict[str, Any]]:
        normalized_tiles = self._normalize_tiles(tiles)
        normalized_metas = self._normalize_metas(
            metas,
            expected_len=len(normalized_tiles),
        )
        return [
            self.run_tile(base_store, tile, index=index, meta=meta)
            for index, (tile, meta) in enumerate(zip(normalized_tiles, normalized_metas))
        ]
