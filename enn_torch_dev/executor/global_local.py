from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from enn_torch_dev.nn import LocalGlobalFusion

from .graph import GraphExecutor
from .store import KVStore
from .tile_pipeline import TilePipeline


def _validate_global_local_key(value: object, field_name: str) -> str:
    label = f"GlobalLocalPipelineSpec.{field_name}"
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
class GlobalLocalPipelineSpec:
    """
    Global/local fusion orchestration schema.

    ``global_output_name`` is always a node name from ``global_graph``.
    ``global_output_by`` controls whether the global graph output is collected
    by node name or by output key. ``fused_output_key`` optionally writes the
    fused tensor back into the base ``KVStore``.
    """

    global_output_name: str
    fused_output_key: str | None = None
    global_output_by: str = "node"

    def __post_init__(self) -> None:
        self.global_output_name = _validate_global_local_key(
            self.global_output_name,
            "global_output_name",
        )
        if self.fused_output_key is not None:
            self.fused_output_key = _validate_global_local_key(
                self.fused_output_key,
                "fused_output_key",
            )
        if self.global_output_by not in {"node", "key"}:
            raise ValueError(
                "GlobalLocalPipelineSpec.global_output_by must be either "
                "'node' or 'key'."
            )


class GlobalLocalPipeline:
    """
    Run a global graph, run a tiled/local pipeline, then fuse both outputs.

    The global branch is intentionally supplied as a ``GraphExecutor`` so users
    can attach attention, transformer, recurrent, or other context modules as
    graph nodes. This class only orchestrates execution and scalar gate fusion.
    """

    def __init__(
        self,
        *,
        global_graph: GraphExecutor,
        tile_pipeline: TilePipeline,
        fusion: LocalGlobalFusion,
        spec: GlobalLocalPipelineSpec,
    ) -> None:
        if not isinstance(global_graph, GraphExecutor):
            raise TypeError(
                "GlobalLocalPipeline global_graph must be GraphExecutor, "
                f"got {type(global_graph)!r}"
            )
        if not isinstance(tile_pipeline, TilePipeline):
            raise TypeError(
                "GlobalLocalPipeline tile_pipeline must be TilePipeline, "
                f"got {type(tile_pipeline)!r}"
            )
        if not isinstance(fusion, LocalGlobalFusion):
            raise TypeError(
                f"GlobalLocalPipeline fusion must be LocalGlobalFusion, got {type(fusion)!r}"
            )
        if not isinstance(spec, GlobalLocalPipelineSpec):
            raise TypeError(
                "GlobalLocalPipeline spec must be GlobalLocalPipelineSpec, "
                f"got {type(spec)!r}"
            )

        global_graph.output_key(spec.global_output_name)

        self.global_graph = global_graph
        self.tile_pipeline = tile_pipeline
        self.fusion = fusion
        self.spec = spec

    def _global_result_key(self) -> str:
        if self.spec.global_output_by == "node":
            return self.spec.global_output_name
        return self.global_graph.output_key(self.spec.global_output_name)

    def run(self, store: KVStore) -> Tensor:
        if not isinstance(store, KVStore):
            raise TypeError(
                f"GlobalLocalPipeline.run expects KVStore, got {type(store)!r}"
            )

        self.global_graph.run(store)
        global_outputs = self.global_graph.collect_outputs(
            store,
            names=[self.spec.global_output_name],
            by=self.spec.global_output_by,
        )
        global_out = global_outputs[self._global_result_key()]
        local_out = self.tile_pipeline.run(store)
        fused = self.fusion(global_out, local_out)

        if self.spec.fused_output_key is not None:
            store.set(self.spec.fused_output_key, fused, origin="GlobalLocalPipeline")
        return fused
