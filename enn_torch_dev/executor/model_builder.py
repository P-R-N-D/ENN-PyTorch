from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import nn

from .graph_builder import GraphBuilder, KeyRefLike
from .model import Model
from .model_spec import ModelExecutionSpec


class ModelBuilder:
    """
    Plain graph convenience builder for public ``Model`` objects.

    ``ModelBuilder`` starts from a ``GraphBuilder`` and can assemble the
    local/plain path or an explicit tiled path. It does not build stream,
    global/local, fusion, or state-route components.
    """

    def __init__(self, *, graph_builder: GraphBuilder | None = None) -> None:
        if graph_builder is None:
            graph_builder = GraphBuilder()
        if not isinstance(graph_builder, GraphBuilder):
            raise TypeError("ModelBuilder graph_builder must be GraphBuilder.")
        self.graph_builder = graph_builder

    def add(
        self,
        *,
        name: str,
        module: nn.Module,
        output_key: str,
        input_args: Sequence[KeyRefLike] | None = None,
        input_kwargs: Mapping[str, KeyRefLike] | None = None,
        module_key: str | None = None,
        output_keys: Sequence[str] | None = None,
    ) -> "ModelBuilder":
        self.graph_builder.add(
            name=name,
            module=module,
            input_args=input_args,
            input_kwargs=input_kwargs,
            output_key=output_key,
            module_key=module_key,
            output_keys=output_keys,
        )
        return self

    def build(self, *, validate: bool = True) -> Model:
        graph = self.graph_builder.build(validate=validate)
        return Model.from_components(ModelExecutionSpec(), graph=graph)

    def build_tile(
        self,
        *,
        tile_shape: Sequence[int],
        input_key: str,
        tile_input_key: str,
        output_name: str,
        output_key: str | None = None,
        output_by: str = "node",
        tile_stride: Sequence[int] | None = None,
        tile_dims: Sequence[int] | None = None,
        tile_index_key: str | None = None,
        tile_meta_key: str | None = None,
        validate: bool = True,
    ) -> Model:
        graph = self.graph_builder.build(validate=validate)
        spec = ModelExecutionSpec(
            tile=True,
            tile_shape=tile_shape,
            tile_stride=tile_stride,
            tile_dims=tile_dims,
        )
        tile_pipeline = spec.create_tile_pipeline(
            graph,
            input_key=input_key,
            tile_input_key=tile_input_key,
            output_name=output_name,
            output_key=output_key,
            output_by=output_by,
            tile_index_key=tile_index_key,
            tile_meta_key=tile_meta_key,
        )
        return Model.from_components(spec, tile_pipeline=tile_pipeline)
