from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from tensordict import TensorDictBase

from enn_torch_dev.data import KVBatch

from .faults import ResourceSample, StepResult, StepStatus


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _add_count(target: dict[str, int], key: str, value: int) -> None:
    target[key] = int(target.get(key, 0)) + int(value)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.untyped_storage().nbytes())
    except Exception:
        return int(tensor.numel()) * int(tensor.element_size())


def _storage_marker(tensor: torch.Tensor) -> tuple[object, ...]:
    try:
        storage = tensor.untyped_storage()
        data_ptr = int(storage.data_ptr())
        nbytes = int(storage.nbytes())
        if data_ptr == 0 and nbytes == 0:
            return ("empty", str(tensor.device), id(tensor))
        return ("storage", str(tensor.device), data_ptr)
    except Exception:
        try:
            return (
                "tensor",
                str(tensor.device),
                int(tensor.data_ptr()),
                int(tensor.storage_offset()),
            )
        except Exception:
            return ("object", id(tensor))


def _normalize_key(prefix: str, key: object) -> str:
    key_text = ".".join(str(part) for part in key) if isinstance(key, tuple) else str(key)
    if not prefix:
        return key_text
    return f"{prefix}.{key_text}"


def _iter_tensors(value: object, *, prefix: str = ""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
        return
    if isinstance(value, TensorDictBase):
        for key, nested in value.items():
            yield from _iter_tensors(nested, prefix=_normalize_key(prefix, key))
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield from _iter_tensors(nested, prefix=_normalize_key(prefix, key))


def _optional_delta(end: int | None, start: int | None) -> int | None:
    if end is None or start is None:
        return None
    return int(end) - int(start)


@dataclass(frozen=True, slots=True)
class TensorCost:
    key: str
    dtype: str
    shape: tuple[int, ...]
    numel: int
    element_size: int
    nbytes: int
    device: str


@dataclass(frozen=True, slots=True)
class DataCost:
    batch_size: int
    tensor_count: int
    total_tensor_bytes: int
    bytes_per_row: float | None
    tensors: tuple[TensorCost, ...]
    bytes_by_dtype: dict[str, int] = field(default_factory=dict)
    bytes_by_device: dict[str, int] = field(default_factory=dict)


class DataCostProbe:
    """Estimate tensor memory cost for KVBatch and TensorDict hot paths."""

    def estimate_kvbatch(self, batch: KVBatch) -> DataCost:
        if not isinstance(batch, KVBatch):
            raise TypeError("DataCostProbe.estimate_kvbatch expects a KVBatch.")
        return self.estimate_tensordict(batch.td)

    def estimate_tensordict(self, td: TensorDictBase) -> DataCost:
        if not isinstance(td, TensorDictBase):
            raise TypeError("DataCostProbe.estimate_tensordict expects TensorDictBase.")
        if not td.batch_size:
            raise ValueError("TensorDict must have a non-empty batch_size.")
        return self.estimate_mapping(td, batch_size=int(td.batch_size[0]))

    def estimate_mapping(
        self,
        mapping: Mapping[str, object] | TensorDictBase,
        *,
        batch_size: int | None = None,
    ) -> DataCost:
        if not isinstance(mapping, (Mapping, TensorDictBase)):
            raise TypeError("DataCostProbe.estimate_mapping expects a Mapping.")
        if batch_size is not None and (
            not isinstance(batch_size, int) or isinstance(batch_size, bool)
        ):
            raise TypeError("batch_size must be an integer or None.")
        if batch_size is not None and batch_size < 0:
            raise ValueError("batch_size must be non-negative.")

        inferred_batch_size = batch_size
        seen: set[tuple[object, ...]] = set()
        tensor_costs: list[TensorCost] = []
        bytes_by_dtype: dict[str, int] = {}
        bytes_by_device: dict[str, int] = {}
        total_tensor_bytes = 0

        for key, value in mapping.items():
            for tensor_key, tensor in _iter_tensors(value, prefix=str(key)):
                if inferred_batch_size is None and tensor.ndim > 0:
                    inferred_batch_size = int(tensor.shape[0])
                marker = _storage_marker(tensor)
                if marker in seen:
                    continue
                seen.add(marker)

                dtype = _dtype_name(tensor.dtype)
                device = str(tensor.device)
                nbytes = _tensor_nbytes(tensor)
                cost = TensorCost(
                    key=tensor_key,
                    dtype=dtype,
                    shape=tuple(int(dim) for dim in tensor.shape),
                    numel=int(tensor.numel()),
                    element_size=int(tensor.element_size()),
                    nbytes=nbytes,
                    device=device,
                )
                tensor_costs.append(cost)
                total_tensor_bytes += nbytes
                _add_count(bytes_by_dtype, dtype, nbytes)
                _add_count(bytes_by_device, device, nbytes)

        final_batch_size = int(inferred_batch_size or 0)
        bytes_per_row = (
            float(total_tensor_bytes) / float(final_batch_size)
            if final_batch_size > 0
            else None
        )
        return DataCost(
            batch_size=final_batch_size,
            tensor_count=len(tensor_costs),
            total_tensor_bytes=total_tensor_bytes,
            bytes_per_row=bytes_per_row,
            tensors=tuple(tensor_costs),
            bytes_by_dtype=dict(sorted(bytes_by_dtype.items())),
            bytes_by_device=dict(sorted(bytes_by_device.items())),
        )


@dataclass(frozen=True, slots=True)
class ResourceDelta:
    start_phase: str
    end_phase: str
    cpu_rss_delta_bytes: int | None = None
    cuda_allocated_delta_bytes: int | None = None
    cuda_reserved_delta_bytes: int | None = None
    cuda_max_allocated_delta_bytes: int | None = None
    cuda_max_reserved_delta_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class ModelCost:
    status: StepStatus
    batch_size: int
    row_count: int
    total_cpu_rss_delta_bytes: int | None
    total_cuda_allocated_delta_bytes: int | None
    total_cuda_reserved_delta_bytes: int | None
    total_cuda_max_allocated_delta_bytes: int | None
    total_cuda_max_reserved_delta_bytes: int | None
    phase_deltas: tuple[ResourceDelta, ...]
    cuda_device_index: int | None = None


class ModelCostProbe:
    """Estimate per-step resource deltas from RuntimeStep samples."""

    def estimate_step(self, result: StepResult) -> ModelCost:
        if not isinstance(result, StepResult):
            raise TypeError("ModelCostProbe.estimate_step expects a StepResult.")
        samples = tuple(result.resource_samples)
        cuda_device_index = self._cuda_device_index(samples)
        phase_deltas = tuple(
            self._delta_pair(start, end)
            for start, end in zip(samples, samples[1:])
        )
        total = self._delta_pair(samples[0], samples[-1]) if len(samples) >= 2 else None
        return ModelCost(
            status=result.status,
            batch_size=result.batch_size,
            row_count=result.batch_size,
            total_cpu_rss_delta_bytes=None if total is None else total.cpu_rss_delta_bytes,
            total_cuda_allocated_delta_bytes=(
                None if total is None else total.cuda_allocated_delta_bytes
            ),
            total_cuda_reserved_delta_bytes=(
                None if total is None else total.cuda_reserved_delta_bytes
            ),
            total_cuda_max_allocated_delta_bytes=(
                None if total is None else total.cuda_max_allocated_delta_bytes
            ),
            total_cuda_max_reserved_delta_bytes=(
                None if total is None else total.cuda_max_reserved_delta_bytes
            ),
            phase_deltas=phase_deltas,
            cuda_device_index=cuda_device_index,
        )

    @staticmethod
    def _cuda_device_index(samples: tuple[ResourceSample, ...]) -> int | None:
        indices = {
            sample.cuda_device_index
            for sample in samples
            if sample.cuda_device_index is not None
            and any(
                value is not None
                for value in (
                    sample.cuda_allocated_bytes,
                    sample.cuda_reserved_bytes,
                    sample.cuda_max_allocated_bytes,
                    sample.cuda_max_reserved_bytes,
                )
            )
        }
        if len(indices) != 1:
            return None
        return next(iter(indices))

    @staticmethod
    def _delta_pair(start: ResourceSample, end: ResourceSample) -> ResourceDelta:
        same_cuda_device = start.cuda_device_index == end.cuda_device_index
        return ResourceDelta(
            start_phase=start.phase,
            end_phase=end.phase,
            cpu_rss_delta_bytes=_optional_delta(end.cpu_rss_bytes, start.cpu_rss_bytes),
            cuda_allocated_delta_bytes=(
                _optional_delta(end.cuda_allocated_bytes, start.cuda_allocated_bytes)
                if same_cuda_device
                else None
            ),
            cuda_reserved_delta_bytes=(
                _optional_delta(end.cuda_reserved_bytes, start.cuda_reserved_bytes)
                if same_cuda_device
                else None
            ),
            cuda_max_allocated_delta_bytes=(
                _optional_delta(
                    end.cuda_max_allocated_bytes,
                    start.cuda_max_allocated_bytes,
                )
                if same_cuda_device
                else None
            ),
            cuda_max_reserved_delta_bytes=(
                _optional_delta(
                    end.cuda_max_reserved_bytes,
                    start.cuda_max_reserved_bytes,
                )
                if same_cuda_device
                else None
            ),
        )
