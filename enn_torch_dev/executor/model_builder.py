from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import nn

from .graph_builder import GraphBuilder, KeyRefLike
from .model import Model
from .model_spec import ModelExecutionSpec


class ModelBuilder:
    """
    Plain graph convenience builder for public ``Model`` objects.

    ``ModelBuilder`` v0 is intentionally limited to the local/plain graph path:
    ``GraphBuilder -> GraphExecutor -> ModelExecutionSpec() -> Model``.
    It does not build tile, stream, global/local, fusion, or state-route
    components.
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
