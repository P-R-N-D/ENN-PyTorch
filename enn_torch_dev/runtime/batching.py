from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace

import torch

from enn_torch_dev.data import BatchCost, KVBatch

from .cost import DataCost, DataCostProbe


def _validate_optional_limit(value: object, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer or None.")
    if value < 0:
        raise ValueError(f"{label} must be non-negative.")
    return value


def _tensor_payload_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _identity_host_bytes(batch: KVBatch) -> int:
    total = _tensor_payload_bytes(batch.row_ids)
    if batch.source_ids is not None:
        total += _tensor_payload_bytes(batch.source_ids)
    if batch.sample_ids is not None:
        total += _tensor_payload_bytes(batch.sample_ids)
    return total


def _batch_cost_from_data_cost(data_cost: DataCost, batch: KVBatch) -> BatchCost:
    host_bytes = sum(
        nbytes for device, nbytes in data_cost.bytes_by_device.items() if device == "cpu"
    )
    device_bytes = sum(
        nbytes for device, nbytes in data_cost.bytes_by_device.items() if device != "cpu"
    )
    return BatchCost(
        host_bytes=host_bytes + _identity_host_bytes(batch),
        device_bytes=device_bytes,
        num_items=data_cost.batch_size,
    )


def _ceil_scaled(value: int | None, numerator: int, denominator: int) -> int | None:
    if value is None:
        return None
    return (value * numerator + denominator - 1) // denominator


def _scale_batch_cost(cost: BatchCost, *, batch_size: int, parent_size: int) -> BatchCost:
    if parent_size <= 0:
        return BatchCost(num_items=batch_size)
    return BatchCost(
        host_bytes=_ceil_scaled(cost.host_bytes, batch_size, parent_size),
        device_bytes=_ceil_scaled(cost.device_bytes, batch_size, parent_size),
        num_items=batch_size,
        num_tokens=_ceil_scaled(cost.num_tokens, batch_size, parent_size),
        num_tiles=_ceil_scaled(cost.num_tiles, batch_size, parent_size),
    )


@dataclass(frozen=True, slots=True)
class BatchBudget:
    """Static budget limits for one KVBatch."""

    max_host_bytes: int | None = None
    max_device_bytes: int | None = None
    max_items: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_host_bytes",
            _validate_optional_limit(self.max_host_bytes, "BatchBudget.max_host_bytes"),
        )
        object.__setattr__(
            self,
            "max_device_bytes",
            _validate_optional_limit(self.max_device_bytes, "BatchBudget.max_device_bytes"),
        )
        object.__setattr__(
            self,
            "max_items",
            _validate_optional_limit(self.max_items, "BatchBudget.max_items"),
        )
        if (
            self.max_host_bytes is None
            and self.max_device_bytes is None
            and self.max_items is None
        ):
            raise ValueError("BatchBudget must define at least one limit.")


class BatchBudgetExceeded(RuntimeError):
    """Raised when a KVBatch cannot fit inside a BatchBudget."""

    def __init__(
        self,
        *,
        batch_size: int,
        cost: BatchCost,
        budget: BatchBudget,
        reason: str,
    ) -> None:
        self.batch_size = batch_size
        self.cost = cost
        self.budget = budget
        self.reason = reason
        super().__init__(
            "KVBatch exceeds budget: "
            f"batch_size={batch_size}, cost={cost!r}, budget={budget!r}, reason={reason}."
        )


class BudgetedBatcher:
    """Apply static budget limits to a stream of KVBatch objects."""

    def __init__(
        self,
        source: Iterable[KVBatch],
        budget: BatchBudget,
        *,
        cost_probe: DataCostProbe | None = None,
        split_oversized: bool = True,
        min_items: int = 1,
    ) -> None:
        if isinstance(source, KVBatch):
            raise TypeError("BudgetedBatcher.source must be an iterable of KVBatch objects.")
        if not isinstance(source, Iterable):
            raise TypeError("BudgetedBatcher.source must be an iterable of KVBatch objects.")
        if not isinstance(budget, BatchBudget):
            raise TypeError("BudgetedBatcher.budget must be a BatchBudget.")
        if cost_probe is not None and not isinstance(cost_probe, DataCostProbe):
            raise TypeError("BudgetedBatcher.cost_probe must be a DataCostProbe or None.")
        if not isinstance(split_oversized, bool):
            raise TypeError("BudgetedBatcher.split_oversized must be a bool.")
        if not isinstance(min_items, int) or isinstance(min_items, bool):
            raise TypeError("BudgetedBatcher.min_items must be an integer.")
        if min_items <= 0:
            raise ValueError("BudgetedBatcher.min_items must be positive.")

        self.source = source
        self.budget = budget
        self.cost_probe = cost_probe
        self.split_oversized = split_oversized
        self.min_items = min_items

    def __iter__(self) -> Iterator[KVBatch]:
        for batch in self.source:
            if not isinstance(batch, KVBatch):
                raise TypeError("BudgetedBatcher.source must yield KVBatch objects.")
            yield from self._yield_budgeted(batch)

    def _yield_budgeted(self, batch: KVBatch) -> Iterator[KVBatch]:
        batch, cost = self._attach_cost_if_needed(batch)
        reason = self._exceeded_reason(cost)
        if reason is None:
            yield batch
            return

        if not self.split_oversized:
            self._raise_exceeded(batch=batch, cost=cost, reason=reason)
        if batch.batch_size <= self.min_items:
            self._raise_exceeded(batch=batch, cost=cost, reason=reason)

        for subbatch in self._split_batch(batch, cost):
            yield from self._yield_budgeted(subbatch)

    def _attach_cost_if_needed(self, batch: KVBatch) -> tuple[KVBatch, BatchCost]:
        if batch.cost_hint is not None and not self._missing_required_fields(batch.cost_hint):
            cost = batch.cost_hint
        elif self.cost_probe is not None:
            cost = _batch_cost_from_data_cost(self.cost_probe.estimate_kvbatch(batch), batch)
        elif batch.cost_hint is not None:
            cost = batch.cost_hint
        else:
            cost = BatchCost(num_items=batch.batch_size)

        self._validate_cost_fields(cost)
        if batch.cost_hint is cost:
            return batch, cost
        return replace(batch, cost_hint=cost), cost

    def _missing_required_fields(self, cost: BatchCost) -> list[str]:
        missing: list[str] = []
        if self.budget.max_host_bytes is not None and cost.host_bytes is None:
            missing.append("host_bytes")
        if self.budget.max_device_bytes is not None and cost.device_bytes is None:
            missing.append("device_bytes")
        if self.budget.max_items is not None and cost.num_items is None:
            missing.append("num_items")
        return missing

    def _validate_cost_fields(self, cost: BatchCost) -> None:
        missing = self._missing_required_fields(cost)
        if missing:
            raise ValueError(
                "BatchBudget requires cost fields that are not available: "
                f"{missing!r}. Provide KVBatch.cost_hint or a DataCostProbe."
            )

    def _exceeded_reason(self, cost: BatchCost) -> str | None:
        reasons: list[str] = []
        if (
            self.budget.max_host_bytes is not None
            and cost.host_bytes is not None
            and cost.host_bytes > self.budget.max_host_bytes
        ):
            reasons.append(f"host_bytes {cost.host_bytes} > {self.budget.max_host_bytes}")
        if (
            self.budget.max_device_bytes is not None
            and cost.device_bytes is not None
            and cost.device_bytes > self.budget.max_device_bytes
        ):
            reasons.append(f"device_bytes {cost.device_bytes} > {self.budget.max_device_bytes}")
        if (
            self.budget.max_items is not None
            and cost.num_items is not None
            and cost.num_items > self.budget.max_items
        ):
            reasons.append(f"num_items {cost.num_items} > {self.budget.max_items}")
        if not reasons:
            return None
        return ", ".join(reasons)

    def _split_batch(self, batch: KVBatch, cost: BatchCost) -> Iterator[KVBatch]:
        target_size = self._target_split_size(batch, cost)
        for start in range(0, batch.batch_size, target_size):
            end = min(start + target_size, batch.batch_size)
            cost_hint = None
            if self.cost_probe is None and batch.cost_hint is not None:
                cost_hint = _scale_batch_cost(
                    cost,
                    batch_size=end - start,
                    parent_size=batch.batch_size,
                )
            yield _slice_kvbatch(batch, start, end, cost_hint=cost_hint)

    def _target_split_size(self, batch: KVBatch, cost: BatchCost) -> int:
        target_size = batch.batch_size
        if self.budget.max_items is not None and cost.num_items is not None:
            if cost.num_items > self.budget.max_items:
                target_size = min(target_size, max(self.min_items, self.budget.max_items))

        target_size = self._target_size_for_byte_limit(
            target_size,
            batch_size=batch.batch_size,
            actual=cost.host_bytes,
            limit=self.budget.max_host_bytes,
        )
        target_size = self._target_size_for_byte_limit(
            target_size,
            batch_size=batch.batch_size,
            actual=cost.device_bytes,
            limit=self.budget.max_device_bytes,
        )

        if target_size >= batch.batch_size:
            target_size = max(self.min_items, batch.batch_size // 2)
        target_size = max(self.min_items, min(target_size, batch.batch_size - 1))
        return target_size

    def _target_size_for_byte_limit(
        self,
        current: int,
        *,
        batch_size: int,
        actual: int | None,
        limit: int | None,
    ) -> int:
        if limit is None or actual is None or actual <= limit or batch_size <= 0:
            return current
        if actual == 0:
            return current
        estimated = int(limit * batch_size // actual)
        return min(current, max(self.min_items, estimated))

    def _raise_exceeded(self, *, batch: KVBatch, cost: BatchCost, reason: str) -> None:
        raise BatchBudgetExceeded(
            batch_size=batch.batch_size,
            cost=cost,
            budget=self.budget,
            reason=reason,
        )


def _slice_kvbatch(
    batch: KVBatch,
    start: int,
    end: int,
    *,
    cost_hint: BatchCost | None = None,
) -> KVBatch:
    td = batch.td[start:end].clone(recurse=True)
    return KVBatch(
        td=td,
        row_ids=batch.row_ids[start:end],
        source_ids=None if batch.source_ids is None else batch.source_ids[start:end],
        sample_ids=None if batch.sample_ids is None else batch.sample_ids[start:end],
        schema_id=batch.schema_id,
        shard_id=batch.shard_id,
        cost_hint=cost_hint,
    )
