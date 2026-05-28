from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .graph import GraphExecutor
from .store import KVStore
from .tile import TileExecutor, TileSpec
from .tile_policy import TilePolicy
from .tile_reconstruct import TileReconstructor


def _validate_pipeline_key(value: object, field_name: str) -> str:
    label = f"TilePipelineSpec.{field_name}"
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    if not value:
        raise ValueError(f"{label} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(
            f"{label} must not have leading or trailing whitespace."
        )
    return value


@dataclass(slots=True)
class TilePipelineSpec:
    """
    Split-execute-reconstruct schema for tiled tensor execution.

    ``output_name`` is always a graph node name. When ``output_by="key"``, the
    pipeline still asks ``TileExecutor`` to collect that node, but extracts each
    per-tile value from the returned dict using the node's output key.
    """

    input_key: str
    tile_input_key: str
    output_name: str
    output_key: str | None = None
    output_by: str = "node"
    tile_index_key: str | None = None
    tile_meta_key: str | None = None

    def __post_init__(self) -> None:
        self.input_key = _validate_pipeline_key(self.input_key, "input_key")
        self.tile_input_key = _validate_pipeline_key(
            self.tile_input_key,
            "tile_input_key",
        )
        self.output_name = _validate_pipeline_key(self.output_name, "output_name")
        if self.output_key is not None:
            self.output_key = _validate_pipeline_key(self.output_key, "output_key")
        if self.output_by not in {"node", "key"}:
            raise ValueError(
                "TilePipelineSpec.output_by must be either 'node' or 'key'."
            )
        if self.tile_index_key is not None:
            self.tile_index_key = _validate_pipeline_key(
                self.tile_index_key,
                "tile_index_key",
            )
        if self.tile_meta_key is not None:
            self.tile_meta_key = _validate_pipeline_key(
                self.tile_meta_key,
                "tile_meta_key",
            )


class TilePipeline:
    """
    Orchestrate the deterministic tiled execution path.

    This executor-layer pipeline connects ``TilePolicy``, ``TileExecutor``, and
    ``TileReconstructor``:

    ``Tensor -> tiles/metas -> tile graph outputs -> reconstructed Tensor``.

    It does not run a global branch, perform local/global gating, or learn
    reconstruction weights.
    """

    def __init__(
        self,
        graph: GraphExecutor,
        tile_policy: TilePolicy,
        spec: TilePipelineSpec,
        *,
        tile_reconstructor: TileReconstructor | None = None,
    ) -> None:
        if not isinstance(graph, GraphExecutor):
            raise TypeError(f"TilePipeline graph must be GraphExecutor, got {type(graph)!r}")
        if not isinstance(tile_policy, TilePolicy):
            raise TypeError(
                f"TilePipeline tile_policy must be TilePolicy, got {type(tile_policy)!r}"
            )
        if not isinstance(spec, TilePipelineSpec):
            raise TypeError(f"TilePipeline spec must be TilePipelineSpec, got {type(spec)!r}")
        if tile_reconstructor is None:
            tile_reconstructor = TileReconstructor()
        if not isinstance(tile_reconstructor, TileReconstructor):
            raise TypeError(
                "TilePipeline tile_reconstructor must be TileReconstructor, "
                f"got {type(tile_reconstructor)!r}"
            )

        graph.output_key(spec.output_name)

        self.graph = graph
        self.tile_policy = tile_policy
        self.spec = spec
        self.tile_reconstructor = tile_reconstructor
        self.tile_executor = TileExecutor(
            graph,
            TileSpec(
                tile_input_key=spec.tile_input_key,
                output_names=[spec.output_name],
                output_by=spec.output_by,
                tile_index_key=spec.tile_index_key,
                tile_meta_key=spec.tile_meta_key,
            ),
        )

    def _result_key(self) -> str:
        if self.spec.output_by == "node":
            return self.spec.output_name
        return self.graph.output_key(self.spec.output_name)

    def run(self, store: KVStore) -> Tensor:
        if not isinstance(store, KVStore):
            raise TypeError(f"TilePipeline.run expects KVStore, got {type(store)!r}")

        x = store.get(self.spec.input_key)
        tiles, metas = self.tile_policy.split(x)
        if not tiles:
            raise ValueError("TilePipeline produced no tiles to execute.")

        tile_results = self.tile_executor.run(store, tiles, metas=metas)
        result_key = self._result_key()
        tile_outputs = [result[result_key] for result in tile_results]
        out = self.tile_reconstructor.reconstruct(tile_outputs, metas)

        if self.spec.output_key is not None:
            store.set(self.spec.output_key, out, origin="TilePipeline")
        return out
