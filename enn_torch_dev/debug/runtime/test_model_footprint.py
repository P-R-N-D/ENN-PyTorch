from __future__ import annotations

import pytest
import torch
from torch import nn

from enn_torch_dev.executor import GraphExecutor, KeyRef, NodeSpec
from enn_torch_dev.runtime import ModelFootprint, OptimizerFootprint


class _WithBuffer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.register_buffer("scale", torch.ones(4, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * self.scale[0]


def test_model_footprint_counts_parameters_and_bytes() -> None:
    model = nn.Linear(3, 2, bias=True)

    footprint = ModelFootprint.from_module(model)

    assert footprint.parameter_count == 8
    assert footprint.trainable_parameter_count == 8
    assert footprint.buffer_count == 0
    assert footprint.parameter_bytes == 8 * 4
    assert footprint.trainable_parameter_bytes == 8 * 4
    assert footprint.buffer_bytes == 0
    assert footprint.total_model_bytes == 8 * 4
    assert footprint.parameters_by_dtype == {"float32": 8}
    assert footprint.bytes_by_dtype == {"float32": 8 * 4}
    assert footprint.bytes_by_device == {"cpu": 8 * 4}


def test_model_footprint_includes_buffers() -> None:
    model = _WithBuffer()

    footprint = ModelFootprint.from_module(model)

    assert footprint.parameter_count == 8
    assert footprint.buffer_count == 4
    assert footprint.buffer_bytes == 4 * 4
    assert footprint.total_model_bytes == (8 + 4) * 4
    assert footprint.buffers_by_dtype == {"float32": 4}
    assert footprint.bytes_by_device == {"cpu": (8 + 4) * 4}


def test_model_footprint_separates_trainable_and_frozen_parameters() -> None:
    model = nn.Linear(3, 2, bias=True)
    model.bias.requires_grad_(False)

    footprint = ModelFootprint.from_module(model)

    assert footprint.parameter_count == 8
    assert footprint.trainable_parameter_count == 6
    assert footprint.parameter_bytes == 8 * 4
    assert footprint.trainable_parameter_bytes == 6 * 4


def test_model_footprint_groups_by_dtype() -> None:
    class _Mixed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Parameter(torch.ones(2, dtype=torch.float32))
            self.b = nn.Parameter(torch.ones(3, dtype=torch.float64))
            self.register_buffer("mask", torch.ones(5, dtype=torch.bool))

    footprint = ModelFootprint.from_module(_Mixed())

    assert footprint.parameters_by_dtype == {"float32": 2, "float64": 3}
    assert footprint.buffers_by_dtype == {"bool": 5}
    assert footprint.bytes_by_dtype["float32"] == 2 * 4
    assert footprint.bytes_by_dtype["float64"] == 3 * 8
    assert footprint.bytes_by_dtype["bool"] == 5


def test_model_footprint_accepts_graph_executor() -> None:
    graph = GraphExecutor(
        [
            (
                NodeSpec(
                    name="linear",
                    input_args=[KeyRef("x")],
                    output_key="pred",
                ),
                nn.Linear(3, 2, bias=False),
            )
        ]
    )

    footprint = ModelFootprint.from_module(graph)

    assert footprint.parameter_count == 6
    assert footprint.parameter_bytes == 6 * 4


def test_optimizer_footprint_handles_empty_state() -> None:
    model = nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    footprint = OptimizerFootprint.from_optimizer(optimizer)

    assert footprint.param_group_count == 1
    assert footprint.state_tensor_count == 0
    assert footprint.state_bytes == 0
    assert footprint.bytes_by_device == {}


def test_optimizer_footprint_counts_state_after_step() -> None:
    model = nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.ones(2, 3)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    footprint = OptimizerFootprint.from_optimizer(optimizer)

    assert footprint.param_group_count == 1
    assert footprint.state_tensor_count > 0
    assert footprint.state_bytes > 0
    assert footprint.bytes_by_dtype
    assert footprint.bytes_by_device == {"cpu": footprint.state_bytes}


def test_footprint_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="nn.Module"):
        ModelFootprint.from_module(object())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Optimizer"):
        OptimizerFootprint.from_optimizer(object())  # type: ignore[arg-type]


def test_model_footprint_counts_aliased_parameter_storage_once() -> None:
    class _AliasedParameters(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            base = torch.arange(4, dtype=torch.float32)
            self.left = nn.Parameter(base)
            self.right = nn.Parameter(base.view(2, 2))

    footprint = ModelFootprint.from_module(_AliasedParameters())

    assert footprint.parameter_count == 4
    assert footprint.trainable_parameter_count == 4
    assert footprint.parameter_bytes == 4 * 4
    assert footprint.trainable_parameter_bytes == 4 * 4
    assert footprint.parameters_by_dtype == {"float32": 4}
    assert footprint.bytes_by_dtype == {"float32": 4 * 4}


def test_model_footprint_counts_aliased_buffer_storage_once() -> None:
    class _AliasedBuffers(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            base = torch.arange(4, dtype=torch.float32)
            self.register_buffer("left", base)
            self.register_buffer("right", base.view(2, 2))

    footprint = ModelFootprint.from_module(_AliasedBuffers())

    assert footprint.buffer_count == 4
    assert footprint.buffer_bytes == 4 * 4
    assert footprint.total_model_bytes == 4 * 4
    assert footprint.buffers_by_dtype == {"float32": 4}
    assert footprint.bytes_by_dtype == {"float32": 4 * 4}


def test_optimizer_footprint_counts_nested_state_tensors() -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    param = next(model.parameters())
    nested: dict[str, object] = {}
    nested["self"] = nested
    nested["values"] = [
        torch.ones(2, dtype=torch.float32),
        (torch.ones(3, dtype=torch.float64),),
    ]
    optimizer.state[param] = {"nested": nested}

    footprint = OptimizerFootprint.from_optimizer(optimizer)

    assert footprint.param_group_count == 1
    assert footprint.state_tensor_count == 2
    assert footprint.state_bytes == (2 * 4) + (3 * 8)
    assert footprint.bytes_by_dtype == {"float32": 2 * 4, "float64": 3 * 8}
    assert footprint.tensors_by_dtype == {"float32": 1, "float64": 1}
    assert footprint.bytes_by_device == {"cpu": footprint.state_bytes}


def test_optimizer_footprint_counts_aliased_state_storage_once() -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    param = next(model.parameters())
    base = torch.arange(4, dtype=torch.float32)
    optimizer.state[param] = {
        "first": base,
        "nested": {"second": [base.view(2, 2)]},
    }

    footprint = OptimizerFootprint.from_optimizer(optimizer)

    assert footprint.state_tensor_count == 1
    assert footprint.state_bytes == 4 * 4
    assert footprint.bytes_by_dtype == {"float32": 4 * 4}
    assert footprint.tensors_by_dtype == {"float32": 1}
    assert footprint.bytes_by_device == {"cpu": footprint.state_bytes}
