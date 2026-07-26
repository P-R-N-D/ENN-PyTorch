from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

from enn_torch_dev.data import BatchCost

from .batching import BatchBudget
from .footprint import ModelFootprint, OptimizerFootprint
from .pressure import ResourceCapacity


def _validate_positive_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer.")
    if value <= 0:
        raise ValueError(f"{label} must be positive.")
    return value


def _validate_optional_positive_int(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    return _validate_positive_int(value, label=label)


def _validate_non_negative_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer.")
    if value < 0:
        raise ValueError(f"{label} must be non-negative.")
    return value


def _validate_ratio(value: object, *, label: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{label} must be a finite number.")
    ratio = float(value)
    if not math.isfinite(ratio):
        raise ValueError(f"{label} must be finite.")
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"{label} must be greater than 0 and at most 1.")
    return ratio


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True, slots=True)
class InitialBatchBudgetPolicy:
    """Conservative limits used for one pure initial-budget recommendation."""

    min_items: int = 1
    max_items: int | None = None
    host_utilization_ratio: float = 0.8
    device_utilization_ratio: float = 0.8
    host_reserve_bytes: int = 0
    device_reserve_bytes: int = 0
    fallback_max_items: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "min_items",
            _validate_positive_int(
                self.min_items,
                label="InitialBatchBudgetPolicy.min_items",
            ),
        )
        object.__setattr__(
            self,
            "max_items",
            _validate_optional_positive_int(
                self.max_items,
                label="InitialBatchBudgetPolicy.max_items",
            ),
        )
        object.__setattr__(
            self,
            "host_utilization_ratio",
            _validate_ratio(
                self.host_utilization_ratio,
                label="InitialBatchBudgetPolicy.host_utilization_ratio",
            ),
        )
        object.__setattr__(
            self,
            "device_utilization_ratio",
            _validate_ratio(
                self.device_utilization_ratio,
                label="InitialBatchBudgetPolicy.device_utilization_ratio",
            ),
        )
        object.__setattr__(
            self,
            "host_reserve_bytes",
            _validate_non_negative_int(
                self.host_reserve_bytes,
                label="InitialBatchBudgetPolicy.host_reserve_bytes",
            ),
        )
        object.__setattr__(
            self,
            "device_reserve_bytes",
            _validate_non_negative_int(
                self.device_reserve_bytes,
                label="InitialBatchBudgetPolicy.device_reserve_bytes",
            ),
        )
        object.__setattr__(
            self,
            "fallback_max_items",
            _validate_optional_positive_int(
                self.fallback_max_items,
                label="InitialBatchBudgetPolicy.fallback_max_items",
            ),
        )
        if self.max_items is not None and self.min_items > self.max_items:
            raise ValueError(
                "InitialBatchBudgetPolicy.min_items must not exceed max_items."
            )
        if (
            self.fallback_max_items is not None
            and self.fallback_max_items < self.min_items
        ):
            raise ValueError(
                "InitialBatchBudgetPolicy.fallback_max_items must be at least min_items."
            )


class BatchBudgetRecommendationError(RuntimeError):
    """Raised when a safe finite initial budget cannot be recommended."""

    def __init__(
        self,
        reason: str,
        *,
        dimensions: tuple[str, ...] = (),
    ) -> None:
        self.reason = reason
        self.dimensions = dimensions
        suffix = f" dimensions={dimensions!r}" if dimensions else ""
        super().__init__(f"Initial batch budget recommendation failed: {reason}.{suffix}")


@dataclass(frozen=True, slots=True)
class BatchBudgetRecommendation:
    """A recommended budget plus the exact static inputs used to derive it."""

    recommended_budget: BatchBudget
    limiting_dimensions: tuple[str, ...]
    reference_num_items: int | None
    effective_host_capacity_bytes: int | None
    device_capacity_bytes: int | None
    host_fixed_bytes: int
    device_fixed_bytes: int
    host_usable_bytes: int | None
    device_usable_bytes: int | None
    host_bytes_per_item: int | None
    device_bytes_per_item: int | None
    host_items_limit: int | None
    device_items_limit: int | None
    fallback_used: bool = False
    warnings: tuple[str, ...] = ()


def _validated_device_bytes(
    values: object,
    *,
    expected_total: int,
    label: str,
) -> dict[str, int]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must be a mapping of device names to byte counts.")
    normalized: dict[str, int] = {}
    for device, nbytes in values.items():
        if not isinstance(device, str):
            raise TypeError(f"{label} keys must be strings.")
        if not device or device != device.strip():
            raise ValueError(f"{label} keys must be non-empty normalized strings.")
        normalized[device] = _validate_non_negative_int(
            nbytes,
            label=f"{label}[{device!r}]",
        )
    if expected_total > 0 and not normalized:
        raise BatchBudgetRecommendationError(
            f"{label} lacks device provenance; build the footprint with its probe"
        )
    if sum(normalized.values()) != expected_total:
        raise BatchBudgetRecommendationError(
            f"{label} byte total does not match the footprint total"
        )
    return normalized


def _fixed_bytes_by_capacity(
    capacity: ResourceCapacity,
    model_footprint: ModelFootprint | None,
    optimizer_footprint: OptimizerFootprint | None,
) -> tuple[int, int]:
    combined: dict[str, int] = {}

    if model_footprint is not None:
        values = _validated_device_bytes(
            model_footprint.bytes_by_device,
            expected_total=_validate_non_negative_int(
                model_footprint.total_model_bytes,
                label="ModelFootprint.total_model_bytes",
            ),
            label="ModelFootprint.bytes_by_device",
        )
        for device, nbytes in values.items():
            combined[device] = combined.get(device, 0) + nbytes

    if optimizer_footprint is not None:
        values = _validated_device_bytes(
            optimizer_footprint.bytes_by_device,
            expected_total=_validate_non_negative_int(
                optimizer_footprint.state_bytes,
                label="OptimizerFootprint.state_bytes",
            ),
            label="OptimizerFootprint.bytes_by_device",
        )
        for device, nbytes in values.items():
            combined[device] = combined.get(device, 0) + nbytes

    host_bytes = combined.pop("cpu", 0)
    device_bytes = 0
    if capacity.cuda_device_index is not None:
        target = f"cuda:{capacity.cuda_device_index}"
        device_bytes += combined.pop(target, 0)
        device_bytes += combined.pop("cuda", 0)
    else:
        cuda_devices = tuple(
            sorted(
                device
                for device, nbytes in combined.items()
                if device.startswith("cuda") and nbytes > 0
            )
        )
        if cuda_devices:
            raise BatchBudgetRecommendationError(
                "CUDA footprint is present but CUDA capacity is unavailable",
                dimensions=("device",),
            )

    unsupported = tuple(
        sorted(device for device, nbytes in combined.items() if nbytes > 0)
    )
    if unsupported:
        raise BatchBudgetRecommendationError(
            "footprint devices are not represented by ResourceCapacity",
            dimensions=unsupported,
        )
    if capacity.cuda_total_bytes is None and device_bytes > 0:
        raise BatchBudgetRecommendationError(
            "CUDA footprint is present but CUDA capacity is unavailable",
            dimensions=("device",),
        )
    return host_bytes, device_bytes


def _bytes_per_item(total_bytes: int | None, num_items: int | None) -> int | None:
    if total_bytes is None or num_items is None or num_items <= 0:
        return None
    if total_bytes == 0:
        return 0
    return _ceil_div(total_bytes, num_items)


def _usable_bytes(
    capacity_bytes: int | None,
    *,
    utilization_ratio: float,
    reserve_bytes: int,
    fixed_bytes: int,
) -> int | None:
    if capacity_bytes is None:
        return None
    return math.floor(capacity_bytes * utilization_ratio) - reserve_bytes - fixed_bytes


def recommend_initial_batch_budget(
    capacity: ResourceCapacity,
    batch_cost: BatchCost,
    *,
    model_footprint: ModelFootprint | None = None,
    optimizer_footprint: OptimizerFootprint | None = None,
    policy: InitialBatchBudgetPolicy | None = None,
) -> BatchBudgetRecommendation:
    """Return a deterministic, side-effect-free initial ``BatchBudget`` recommendation."""

    if not isinstance(capacity, ResourceCapacity):
        raise TypeError("capacity must be a ResourceCapacity.")
    if not isinstance(batch_cost, BatchCost):
        raise TypeError("batch_cost must be a BatchCost.")
    if model_footprint is not None and not isinstance(model_footprint, ModelFootprint):
        raise TypeError("model_footprint must be a ModelFootprint or None.")
    if optimizer_footprint is not None and not isinstance(
        optimizer_footprint,
        OptimizerFootprint,
    ):
        raise TypeError("optimizer_footprint must be an OptimizerFootprint or None.")
    if policy is None:
        policy = InitialBatchBudgetPolicy()
    elif not isinstance(policy, InitialBatchBudgetPolicy):
        raise TypeError("policy must be an InitialBatchBudgetPolicy or None.")

    reference_num_items = batch_cost.num_items
    if reference_num_items is not None:
        _validate_non_negative_int(
            reference_num_items,
            label="BatchCost.num_items",
        )

    host_fixed_bytes, device_fixed_bytes = _fixed_bytes_by_capacity(
        capacity,
        model_footprint,
        optimizer_footprint,
    )
    host_bytes_per_item = _bytes_per_item(
        batch_cost.host_bytes,
        reference_num_items,
    )
    device_bytes_per_item = _bytes_per_item(
        batch_cost.device_bytes,
        reference_num_items,
    )

    host_usable_bytes = _usable_bytes(
        capacity.effective_cpu_bytes,
        utilization_ratio=policy.host_utilization_ratio,
        reserve_bytes=policy.host_reserve_bytes,
        fixed_bytes=host_fixed_bytes,
    )
    device_usable_bytes = _usable_bytes(
        capacity.cuda_total_bytes,
        utilization_ratio=policy.device_utilization_ratio,
        reserve_bytes=policy.device_reserve_bytes,
        fixed_bytes=device_fixed_bytes,
    )

    if host_usable_bytes is not None and host_usable_bytes < 0:
        raise BatchBudgetRecommendationError(
            "host reserve and fixed footprint exceed the usable host capacity",
            dimensions=("host",),
        )
    if device_usable_bytes is not None and device_usable_bytes < 0:
        raise BatchBudgetRecommendationError(
            "device reserve and fixed footprint exceed the usable device capacity",
            dimensions=("device",),
        )
    if capacity.cuda_total_bytes is None and (
        device_fixed_bytes > 0
        or (batch_cost.device_bytes is not None and batch_cost.device_bytes > 0)
        or (device_bytes_per_item is not None and device_bytes_per_item > 0)
    ):
        raise BatchBudgetRecommendationError(
            "device memory demand is known but CUDA capacity is unavailable",
            dimensions=("device",),
        )

    unresolved: list[str] = []
    host_items_limit: int | None = None
    device_items_limit: int | None = None

    if capacity.effective_cpu_bytes is None and host_fixed_bytes > 0:
        unresolved.append("host")
    elif host_bytes_per_item is None:
        unresolved.append("host")
    elif host_bytes_per_item > 0:
        if host_usable_bytes is None:
            unresolved.append("host")
        else:
            host_items_limit = host_usable_bytes // host_bytes_per_item

    if device_bytes_per_item is None:
        unresolved.append("device")
    elif device_bytes_per_item > 0:
        if device_usable_bytes is None:
            unresolved.append("device")
        else:
            device_items_limit = device_usable_bytes // device_bytes_per_item

    candidate_limits: list[tuple[str, int]] = []
    if host_items_limit is not None:
        candidate_limits.append(("host", host_items_limit))
    if device_items_limit is not None:
        candidate_limits.append(("device", device_items_limit))
    if policy.max_items is not None:
        candidate_limits.append(("policy_max_items", policy.max_items))

    warnings: list[str] = []
    fallback_used = False
    if unresolved:
        if policy.fallback_max_items is None:
            raise BatchBudgetRecommendationError(
                "cost or capacity is unknown and no fallback_max_items is configured",
                dimensions=tuple(unresolved),
            )
        fallback_used = True
        candidate_limits.append(("fallback", policy.fallback_max_items))
        warnings.append(
            "fallback_max_items was used because these dimensions were unresolved: "
            + ", ".join(unresolved)
        )

    if not candidate_limits:
        if policy.fallback_max_items is None:
            raise BatchBudgetRecommendationError(
                "no finite item limit can be derived from zero-cost dimensions"
            )
        fallback_used = True
        candidate_limits.append(("fallback", policy.fallback_max_items))
        warnings.append(
            "fallback_max_items was used because no resource dimension produced a finite limit"
        )

    recommended_items = min(limit for _dimension, limit in candidate_limits)
    limiting_dimensions = tuple(
        dimension
        for dimension, limit in candidate_limits
        if limit == recommended_items
    )
    if recommended_items < policy.min_items:
        raise BatchBudgetRecommendationError(
            "the computed limit is below min_items",
            dimensions=limiting_dimensions,
        )

    recommended_budget = BatchBudget(
        max_host_bytes=(
            host_usable_bytes
            if host_bytes_per_item is not None and host_usable_bytes is not None
            else None
        ),
        max_device_bytes=(
            device_usable_bytes
            if device_bytes_per_item is not None and device_usable_bytes is not None
            else None
        ),
        max_items=recommended_items,
    )
    return BatchBudgetRecommendation(
        recommended_budget=recommended_budget,
        limiting_dimensions=limiting_dimensions,
        reference_num_items=reference_num_items,
        effective_host_capacity_bytes=capacity.effective_cpu_bytes,
        device_capacity_bytes=capacity.cuda_total_bytes,
        host_fixed_bytes=host_fixed_bytes,
        device_fixed_bytes=device_fixed_bytes,
        host_usable_bytes=host_usable_bytes,
        device_usable_bytes=device_usable_bytes,
        host_bytes_per_item=host_bytes_per_item,
        device_bytes_per_item=device_bytes_per_item,
        host_items_limit=host_items_limit,
        device_items_limit=device_items_limit,
        fallback_used=fallback_used,
        warnings=tuple(warnings),
    )
