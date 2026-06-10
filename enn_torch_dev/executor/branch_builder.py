from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import nn

from .graph import GraphExecutor
from .graph_builder import GraphBuilder, KeyRefLike


_BRANCH_ROLES = {"local", "global", "stream"}


def _validate_branch_role(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("BranchBuilder role must be a string.")
    if value not in _BRANCH_ROLES:
        raise ValueError(
            "BranchBuilder role must be one of 'local', 'global', or 'stream'."
        )
    return value


def _validate_branch_input_key(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("BranchBuilder input_key must be a string.")
    if not value:
        raise ValueError("BranchBuilder input_key must be a non-empty string.")
    if value != value.strip():
        raise ValueError("BranchBuilder input_key must not have whitespace padding.")
    return value


class BranchBuilder:
    """
    Graph-only convenience builder for repeated branch wiring.

    ``BranchBuilder`` wraps ``GraphBuilder`` with a branch role and default input
    key. It returns ``GraphExecutor`` components only; pipeline and model assembly
    stay with ``ModelExecutionSpec`` and ``ModelBuilder``.
    """

    def __init__(
        self,
        *,
        input_key: str,
        role: str = "local",
        graph_builder: GraphBuilder | None = None,
    ) -> None:
        if graph_builder is None:
            graph_builder = GraphBuilder()
        if not isinstance(graph_builder, GraphBuilder):
            raise TypeError("BranchBuilder graph_builder must be GraphBuilder.")

        self.input_key = _validate_branch_input_key(input_key)
        self.role = _validate_branch_role(role)
        self.graph_builder = graph_builder

    @classmethod
    def local(
        cls,
        *,
        input_key: str,
        graph_builder: GraphBuilder | None = None,
    ) -> "BranchBuilder":
        return cls(input_key=input_key, role="local", graph_builder=graph_builder)

    @classmethod
    def global_(
        cls,
        *,
        input_key: str,
        graph_builder: GraphBuilder | None = None,
    ) -> "BranchBuilder":
        return cls(input_key=input_key, role="global", graph_builder=graph_builder)

    @classmethod
    def stream(
        cls,
        *,
        input_key: str,
        graph_builder: GraphBuilder | None = None,
    ) -> "BranchBuilder":
        return cls(input_key=input_key, role="stream", graph_builder=graph_builder)

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
    ) -> "BranchBuilder":
        if input_args is None:
            input_args = (self.input_key,)

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

    def build(self, *, validate: bool = True) -> GraphExecutor:
        return self.graph_builder.build(validate=validate)
