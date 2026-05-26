from __future__ import annotations

import pytest
import torch
from torch import nn

from enn_torch_dev.executor import KVStore, KeyRef, NodeExecutor, NodeSpec


def test_node_executor_runs_module_and_writes_output() -> None:
    x = torch.randn(2, 4)
    store = KVStore({"x": x})
    module = nn.Linear(4, 3)

    node = NodeExecutor(
        NodeSpec(
            name="linear",
            module=module,
            input_args=[KeyRef("x")],
            output_key="y",
        )
    )

    out = node.run(store)

    assert store.has("y")
    assert torch.equal(store.get("y"), out)
    assert out.shape == (2, 3)


class _AddBias(nn.Module):
    def forward(self, x: torch.Tensor, bias: torch.Tensor | float) -> torch.Tensor:
        return x + bias


def test_node_executor_resolves_kwargs_and_optional_default() -> None:
    x = torch.randn(2, 3)
    store = KVStore({"x": x})
    node = NodeExecutor(
        NodeSpec(
            name="add_bias",
            module=_AddBias(),
            input_args=[KeyRef("x")],
            input_kwargs={"bias": KeyRef("missing.bias", optional=True, default=1.5)},
            output_key="y",
        )
    )

    out = node.run(store)

    assert torch.allclose(out, x + 1.5)
    assert torch.allclose(store.get("y"), x + 1.5)


def test_node_executor_rejects_empty_output_key() -> None:
    with pytest.raises(ValueError, match="output_key"):
        NodeExecutor(
            NodeSpec(
                name="bad",
                module=nn.Identity(),
                output_key="",
            )
        )


def test_node_executor_missing_input_key_raises() -> None:
    node = NodeExecutor(
        NodeSpec(
            name="identity",
            module=nn.Identity(),
            input_args=[KeyRef("missing")],
            output_key="out",
        )
    )

    with pytest.raises(KeyError, match="missing"):
        node.run(KVStore())
