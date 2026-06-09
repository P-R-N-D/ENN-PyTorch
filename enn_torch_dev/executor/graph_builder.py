from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from torch import nn

from .graph import GraphExecutor
from .node import NodeSpec
from .schema import KeyRef

KeyRefLike: TypeAlias = str | KeyRef


def _normalize_ref(value: KeyRefLike, label: str) -> KeyRef:
    if isinstance(value, KeyRef):
        return value
    if isinstance(value, str):
        return KeyRef(value)
    raise TypeError(f"{label} must be a string or KeyRef.")


def _normalize_ref_sequence(
    value: Sequence[KeyRefLike] | None,
    *,
    label: str,
) -> list[KeyRef]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a sequence of strings/KeyRefs, not a string.")
    if isinstance(value, Mapping):
        raise TypeError(
            f"{label} must be a sequence of strings/KeyRefs, not a mapping."
        )
    if not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence of strings/KeyRefs.")

    return [
        _normalize_ref(ref, f"{label}[{index}]")
        for index, ref in enumerate(value)
    ]


def _normalize_ref_mapping(
    value: Mapping[str, KeyRefLike] | None,
    *,
    label: str,
) -> dict[str, KeyRef]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{label} must be a mapping of argument names to strings/KeyRefs."
        )

    out: dict[str, KeyRef] = {}
    for key, ref in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{label} keys must be strings.")
        if not key:
            raise ValueError(f"{label} keys must be non-empty strings.")
        out[key] = _normalize_ref(ref, f"{label}[{key!r}]")
    return out


class GraphBuilder:
    """Small convenience builder for leaf-node ``GraphExecutor`` objects."""

    def __init__(self) -> None:
        self._nodes: list[tuple[NodeSpec, nn.Module]] = []

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
    ) -> "GraphBuilder":
        """
        Add one leaf ``nn.Module`` node.

        String input references are converted to ``KeyRef``. Existing ``KeyRef``
        instances are preserved so callers can use optional/default references.
        """

        if not isinstance(module, nn.Module):
            raise TypeError("GraphBuilder.add module must be an nn.Module.")

        spec = NodeSpec(
            name=name,
            module_key=module_key,
            input_args=_normalize_ref_sequence(input_args, label="input_args"),
            input_kwargs=_normalize_ref_mapping(input_kwargs, label="input_kwargs"),
            output_key=output_key,
            output_keys=output_keys,
        )
        self._nodes.append((spec, module))
        return self

    def build(self, *, validate: bool = True) -> GraphExecutor:
        """Build a new ``GraphExecutor`` from the collected nodes."""

        graph = GraphExecutor()
        for spec, module in self._nodes:
            graph.add_node(spec, module)
        if validate:
            graph.validate()
        return graph
