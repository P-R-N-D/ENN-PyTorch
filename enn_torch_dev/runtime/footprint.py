from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.untyped_storage().nbytes())
    except Exception:
        return int(tensor.numel()) * int(tensor.element_size())


def _storage_marker(tensor: torch.Tensor) -> tuple[str, int] | tuple[str, int, int]:
    try:
        storage = tensor.untyped_storage()
        return (str(tensor.device), int(storage.data_ptr()))
    except Exception:
        return (
            str(tensor.device),
            int(tensor.data_ptr()),
            int(tensor.storage_offset()),
        )


def _add_count(target: dict[str, int], key: str, value: int) -> None:
    target[key] = int(target.get(key, 0)) + int(value)


def _unique_named_tensors(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    seen: set[tuple[str, int] | tuple[str, int, int]] = set()
    for name, tensor in named_tensors:
        marker = _storage_marker(tensor)
        if marker in seen:
            continue
        seen.add(marker)
        yield name, tensor


def _iter_state_tensors(value: Any, seen_containers: set[int]) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, dict):
        marker = id(value)
        if marker in seen_containers:
            return
        seen_containers.add(marker)
        for nested in value.values():
            yield from _iter_state_tensors(nested, seen_containers)
        return
    if isinstance(value, (list, tuple, set)):
        marker = id(value)
        if marker in seen_containers:
            return
        seen_containers.add(marker)
        for nested in value:
            yield from _iter_state_tensors(nested, seen_containers)


@dataclass(frozen=True, slots=True)
class ModelFootprint:
    parameter_count: int
    trainable_parameter_count: int
    buffer_count: int
    parameter_bytes: int
    trainable_parameter_bytes: int
    buffer_bytes: int
    total_model_bytes: int
    bytes_by_dtype: dict[str, int] = field(default_factory=dict)
    parameters_by_dtype: dict[str, int] = field(default_factory=dict)
    buffers_by_dtype: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_module(cls, module: nn.Module) -> "ModelFootprint":
        if not isinstance(module, nn.Module):
            raise TypeError("ModelFootprint.from_module expects an nn.Module.")

        parameter_count = 0
        trainable_parameter_count = 0
        buffer_count = 0
        parameter_bytes = 0
        trainable_parameter_bytes = 0
        buffer_bytes = 0
        bytes_by_dtype: dict[str, int] = {}
        parameters_by_dtype: dict[str, int] = {}
        buffers_by_dtype: dict[str, int] = {}

        for _name, parameter in _unique_named_tensors(
            module.named_parameters(recurse=True)
        ):
            count = int(parameter.numel())
            nbytes = _tensor_nbytes(parameter)
            dtype_key = _dtype_name(parameter.dtype)
            parameter_count += count
            parameter_bytes += nbytes
            _add_count(parameters_by_dtype, dtype_key, count)
            _add_count(bytes_by_dtype, dtype_key, nbytes)
            if parameter.requires_grad:
                trainable_parameter_count += count
                trainable_parameter_bytes += nbytes

        for _name, buffer in _unique_named_tensors(module.named_buffers(recurse=True)):
            count = int(buffer.numel())
            nbytes = _tensor_nbytes(buffer)
            dtype_key = _dtype_name(buffer.dtype)
            buffer_count += count
            buffer_bytes += nbytes
            _add_count(buffers_by_dtype, dtype_key, count)
            _add_count(bytes_by_dtype, dtype_key, nbytes)

        return cls(
            parameter_count=parameter_count,
            trainable_parameter_count=trainable_parameter_count,
            buffer_count=buffer_count,
            parameter_bytes=parameter_bytes,
            trainable_parameter_bytes=trainable_parameter_bytes,
            buffer_bytes=buffer_bytes,
            total_model_bytes=parameter_bytes + buffer_bytes,
            bytes_by_dtype=dict(sorted(bytes_by_dtype.items())),
            parameters_by_dtype=dict(sorted(parameters_by_dtype.items())),
            buffers_by_dtype=dict(sorted(buffers_by_dtype.items())),
        )


@dataclass(frozen=True, slots=True)
class OptimizerFootprint:
    state_tensor_count: int
    state_bytes: int
    param_group_count: int
    bytes_by_dtype: dict[str, int] = field(default_factory=dict)
    tensors_by_dtype: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_optimizer(cls, optimizer: torch.optim.Optimizer) -> "OptimizerFootprint":
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                "OptimizerFootprint.from_optimizer expects a torch.optim.Optimizer."
            )

        state_tensor_count = 0
        state_bytes = 0
        bytes_by_dtype: dict[str, int] = {}
        tensors_by_dtype: dict[str, int] = {}
        seen_tensors: set[tuple[str, int] | tuple[str, int, int]] = set()
        seen_containers: set[int] = set()

        for state in optimizer.state.values():
            for value in _iter_state_tensors(state, seen_containers):
                marker = _storage_marker(value)
                if marker in seen_tensors:
                    continue
                seen_tensors.add(marker)
                state_tensor_count += 1
                nbytes = _tensor_nbytes(value)
                dtype_key = _dtype_name(value.dtype)
                state_bytes += nbytes
                _add_count(bytes_by_dtype, dtype_key, nbytes)
                _add_count(tensors_by_dtype, dtype_key, 1)

        return cls(
            state_tensor_count=state_tensor_count,
            state_bytes=state_bytes,
            param_group_count=len(optimizer.param_groups),
            bytes_by_dtype=dict(sorted(bytes_by_dtype.items())),
            tensors_by_dtype=dict(sorted(tensors_by_dtype.items())),
        )
