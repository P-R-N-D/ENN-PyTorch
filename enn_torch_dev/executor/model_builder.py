from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import nn

from enn_torch_dev.nn import LocalGlobalFusion

from .graph import GraphExecutor
from .graph_builder import GraphBuilder, KeyRefLike
from .model import Model
from .model_spec import ModelExecutionSpec
from .state import StateRoute


class ModelBuilder:
    """
    Plain graph convenience builder for public ``Model`` objects.

    ``ModelBuilder`` starts from a ``GraphBuilder`` and can assemble the
    local/plain path, an explicit tiled path, an explicit stream path, or an
    explicit global-local path. It does not build global graphs, fusion, or infer
    state-route components.
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

    def build_stream(
        self,
        *,
        chunk_input_key: str,
        output_name: str,
        output_by: str = "node",
        chunk_index_key: str | None = None,
        outputs_key: str | None = None,
        state_detach: bool = False,
        state_clone: bool = False,
        reset_state: bool = False,
        state_routes: Sequence[StateRoute] = (),
        validate: bool = True,
    ) -> Model:
        graph = self.graph_builder.build(validate=validate)
        spec = ModelExecutionSpec(stateful=True)
        stream_pipeline = spec.create_stream_pipeline(
            graph,
            chunk_input_key=chunk_input_key,
            output_name=output_name,
            output_by=output_by,
            chunk_index_key=chunk_index_key,
            outputs_key=outputs_key,
            state_detach=state_detach,
            state_clone=state_clone,
            reset_state=reset_state,
            state_routes=state_routes,
        )
        return Model.from_components(spec, stream_pipeline=stream_pipeline)

    def build_global_local(
        self,
        *,
        global_graph: GraphExecutor,
        fusion: LocalGlobalFusion,
        tile_shape: Sequence[int],
        input_key: str,
        tile_input_key: str,
        local_output_name: str,
        global_output_name: str,
        fused_output_key: str | None = None,
        local_output_key: str | None = None,
        local_output_by: str = "node",
        global_output_by: str = "node",
        tile_stride: Sequence[int] | None = None,
        tile_dims: Sequence[int] | None = None,
        tile_index_key: str | None = None,
        tile_meta_key: str | None = None,
        validate: bool = True,
    ) -> Model:
        local_graph = self.graph_builder.build(validate=validate)
        spec = ModelExecutionSpec(
            context="global_local",
            tile=True,
            tile_shape=tile_shape,
            tile_stride=tile_stride,
            tile_dims=tile_dims,
        )
        tile_pipeline = spec.create_tile_pipeline(
            local_graph,
            input_key=input_key,
            tile_input_key=tile_input_key,
            output_name=local_output_name,
            output_key=local_output_key,
            output_by=local_output_by,
            tile_index_key=tile_index_key,
            tile_meta_key=tile_meta_key,
        )
        global_local_pipeline = spec.create_global_local_pipeline(
            global_graph=global_graph,
            tile_pipeline=tile_pipeline,
            fusion=fusion,
            global_output_name=global_output_name,
            fused_output_key=fused_output_key,
            global_output_by=global_output_by,
        )
        return Model.from_components(
            spec,
            global_local_pipeline=global_local_pipeline,
        )
