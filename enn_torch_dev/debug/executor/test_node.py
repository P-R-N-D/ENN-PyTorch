from __future__ import annotations

import pytest
import torch
from torch import nn

from enn_torch_dev.executor import KVStore, KeyRef, NodeExecutor, NodeSpec


def test_node_spec_defaults_module_key_to_name() -> None:
    spec = NodeSpec(name="linear", output_key="y")

    assert spec.module_key == "linear"


def test_node_executor_runs_module_and_writes_output() -> None:
    x = torch.randn(2, 4)
    store = KVStore({"x": x})
    module = nn.Linear(4, 3)

    node = NodeExecutor(
        NodeSpec(
            name="linear",
            input_args=[KeyRef("x")],
            output_key="y",
        )
    )

    out = node.run(store, module)

    assert store.has("y")
    assert torch.equal(store.get("y"), out)
    assert out.shape == (2, 3)


class _AddBias(nn.Module):
    def forward(self, x: torch.Tensor, bias: torch.Tensor | float) -> torch.Tensor:
        return x + bias


def test_node_executor_resolves_kwargs_and_optional_default() -> None:
    x = torch.randn(2, 3)
    store = KVStore({"x": x})
    module = _AddBias()
    node = NodeExecutor(
        NodeSpec(
            name="add_bias",
            input_args=[KeyRef("x")],
            input_kwargs={"bias": KeyRef("missing.bias", optional=True, default=1.5)},
            output_key="y",
        )
    )

    out = node.run(store, module)

    assert torch.allclose(out, x + 1.5)
    assert torch.allclose(store.get("y"), x + 1.5)


def test_node_executor_rejects_empty_output_key() -> None:
    with pytest.raises(ValueError, match="output_key"):
        NodeExecutor(
            NodeSpec(
                name="bad",
                output_key="",
            )
        )


def test_node_executor_rejects_non_module_argument() -> None:
    node = NodeExecutor(NodeSpec(name="identity", output_key="out"))

    with pytest.raises(TypeError, match="nn.Module"):
        node.run(KVStore(), object())


def test_node_executor_missing_input_key_raises() -> None:
    node = NodeExecutor(
        NodeSpec(
            name="identity",
            input_args=[KeyRef("missing")],
            output_key="out",
        )
    )

    with pytest.raises(KeyError, match="missing"):
        node.run(KVStore(), nn.Identity())
