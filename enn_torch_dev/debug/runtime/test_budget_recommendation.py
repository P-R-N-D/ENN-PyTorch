from __future__ import annotations

import pytest

from enn_torch_dev.data import BatchCost
from enn_torch_dev.runtime import (
    BatchBudget,
    BatchBudgetRecommendationError,
    InitialBatchBudgetPolicy,
    ModelFootprint,
    OptimizerFootprint,
    ResourceCapacity,
    recommend_initial_batch_budget,
)


def _model_footprint(
    *,
    cpu_bytes: int = 0,
    cuda_bytes: int = 0,
    cuda_device_index: int = 0,
    include_device_provenance: bool = True,
) -> ModelFootprint:
    total = cpu_bytes + cuda_bytes
    bytes_by_device: dict[str, int] = {}
    if include_device_provenance:
        if cpu_bytes:
            bytes_by_device["cpu"] = cpu_bytes
        if cuda_bytes:
            bytes_by_device[f"cuda:{cuda_device_index}"] = cuda_bytes
    return ModelFootprint(
        parameter_count=0,
        trainable_parameter_count=0,
        buffer_count=0,
        parameter_bytes=total,
        trainable_parameter_bytes=0,
        buffer_bytes=0,
        total_model_bytes=total,
        bytes_by_device=bytes_by_device,
    )


def _optimizer_footprint(
    *,
    cpu_bytes: int = 0,
    cuda_bytes: int = 0,
    cuda_device_index: int = 0,
) -> OptimizerFootprint:
    total = cpu_bytes + cuda_bytes
    bytes_by_device: dict[str, int] = {}
    if cpu_bytes:
        bytes_by_device["cpu"] = cpu_bytes
    if cuda_bytes:
        bytes_by_device[f"cuda:{cuda_device_index}"] = cuda_bytes
    return OptimizerFootprint(
        state_tensor_count=0,
        state_bytes=total,
        param_group_count=0,
        bytes_by_device=bytes_by_device,
    )


def test_recommendation_uses_effective_cpu_capacity_and_static_footprints() -> None:
    recommendation = recommend_initial_batch_budget(
        ResourceCapacity(cpu_total_bytes=1_000, cpu_limit_bytes=900),
        BatchCost(host_bytes=200, device_bytes=0, num_items=2),
        model_footprint=_model_footprint(cpu_bytes=100),
        optimizer_footprint=_optimizer_footprint(cpu_bytes=100),
        policy=InitialBatchBudgetPolicy(
            host_utilization_ratio=1.0,
            device_utilization_ratio=1.0,
            max_items=20,
        ),
    )

    assert recommendation.effective_host_capacity_bytes == 900
    assert recommendation.host_fixed_bytes == 200
    assert recommendation.host_usable_bytes == 700
    assert recommendation.host_bytes_per_item == 100
    assert recommendation.host_items_limit == 7
    assert recommendation.limiting_dimensions == ("host",)
    assert recommendation.recommended_budget == BatchBudget(
        max_host_bytes=700,
        max_items=7,
    )


def test_recommendation_keeps_cpu_and_cuda_limits_independent() -> None:
    recommendation = recommend_initial_batch_budget(
        ResourceCapacity(
            cpu_total_bytes=10_000,
            cuda_total_bytes=1_000,
            cuda_device_index=0,
        ),
        BatchCost(host_bytes=100, device_bytes=200, num_items=2),
        model_footprint=_model_footprint(cuda_bytes=200),
        optimizer_footprint=_optimizer_footprint(cuda_bytes=100),
        policy=InitialBatchBudgetPolicy(
            host_utilization_ratio=1.0,
            device_utilization_ratio=1.0,
            max_items=100,
        ),
    )

    assert recommendation.host_items_limit == 200
    assert recommendation.device_fixed_bytes == 300
    assert recommendation.device_usable_bytes == 700
    assert recommendation.device_bytes_per_item == 100
    assert recommendation.device_items_limit == 7
    assert recommendation.limiting_dimensions == ("device",)
    assert recommendation.recommended_budget == BatchBudget(
        max_host_bytes=10_000,
        max_device_bytes=700,
        max_items=7,
    )


def test_recommendation_applies_reserve_and_utilization_before_item_limit() -> None:
    recommendation = recommend_initial_batch_budget(
        ResourceCapacity(cpu_total_bytes=1_000),
        BatchCost(host_bytes=100, device_bytes=0, num_items=1),
        model_footprint=_model_footprint(cpu_bytes=100),
        policy=InitialBatchBudgetPolicy(
            host_utilization_ratio=0.75,
            device_utilization_ratio=1.0,
            host_reserve_bytes=50,
            max_items=100,
        ),
    )

    assert recommendation.host_usable_bytes == 600
    assert recommendation.host_items_limit == 6
    assert recommendation.recommended_budget.max_items == 6


def test_recommendation_uses_conservative_ceiling_per_item_cost() -> None:
    recommendation = recommend_initial_batch_budget(
        ResourceCapacity(cpu_total_bytes=102),
        BatchCost(host_bytes=101, device_bytes=0, num_items=2),
        policy=InitialBatchBudgetPolicy(
            host_utilization_ratio=1.0,
            device_utilization_ratio=1.0,
            max_items=100,
        ),
    )

    assert recommendation.host_bytes_per_item == 51
    assert recommendation.host_items_limit == 2
    assert recommendation.recommended_budget.max_items == 2


def test_recommendation_treats_zero_cost_as_non_limiting() -> None:
    recommendation = recommend_initial_batch_budget(
        ResourceCapacity(cpu_total_bytes=1_000),
        BatchCost(host_bytes=0, device_bytes=0, num_items=4),
        policy=InitialBatchBudgetPolicy(max_items=3),
    )

    assert recommendation.host_items_limit is None
    assert recommendation.device_items_limit is None
    assert recommendation.limiting_dimensions == ("policy_max_items",)
    assert recommendation.recommended_budget.max_items == 3


def test_recommendation_uses_explicit_fallback_for_unknown_dimensions() -> None:
    recommendation = recommend_initial_batch_budget(
        ResourceCapacity(),
        BatchCost(),
        policy=InitialBatchBudgetPolicy(fallback_max_items=2),
    )

    assert recommendation.fallback_used is True
    assert recommendation.limiting_dimensions == ("fallback",)
    assert recommendation.recommended_budget == BatchBudget(max_items=2)
    assert recommendation.warnings == (
        "fallback_max_items was used because these dimensions were unresolved: host, device",
    )


def test_recommendation_rejects_unknown_dimensions_without_fallback() -> None:
    with pytest.raises(BatchBudgetRecommendationError) as exc_info:
        recommend_initial_batch_budget(ResourceCapacity(), BatchCost())

    assert exc_info.value.dimensions == ("host", "device")


def test_recommendation_does_not_clamp_insufficient_capacity_up_to_min_items() -> None:
    with pytest.raises(BatchBudgetRecommendationError, match="below min_items") as exc_info:
        recommend_initial_batch_budget(
            ResourceCapacity(cpu_total_bytes=99),
            BatchCost(host_bytes=100, device_bytes=0, num_items=1),
            policy=InitialBatchBudgetPolicy(
                host_utilization_ratio=1.0,
                device_utilization_ratio=1.0,
                min_items=1,
                max_items=10,
            ),
        )

    assert exc_info.value.dimensions == ("host",)


def test_recommendation_rejects_fixed_footprint_above_usable_capacity() -> None:
    with pytest.raises(BatchBudgetRecommendationError, match="fixed footprint"):
        recommend_initial_batch_budget(
            ResourceCapacity(cpu_total_bytes=100),
            BatchCost(host_bytes=0, device_bytes=0, num_items=1),
            model_footprint=_model_footprint(cpu_bytes=101),
            policy=InitialBatchBudgetPolicy(
                host_utilization_ratio=1.0,
                device_utilization_ratio=1.0,
                max_items=1,
            ),
        )


def test_recommendation_rejects_cuda_demand_without_cuda_capacity() -> None:
    with pytest.raises(BatchBudgetRecommendationError, match="CUDA capacity"):
        recommend_initial_batch_budget(
            ResourceCapacity(cpu_total_bytes=1_000),
            BatchCost(host_bytes=0, device_bytes=10, num_items=1),
            policy=InitialBatchBudgetPolicy(max_items=1),
        )


def test_recommendation_rejects_footprint_without_device_provenance() -> None:
    with pytest.raises(BatchBudgetRecommendationError, match="device provenance"):
        recommend_initial_batch_budget(
            ResourceCapacity(cpu_total_bytes=1_000),
            BatchCost(host_bytes=10, device_bytes=0, num_items=1),
            model_footprint=_model_footprint(
                cpu_bytes=10,
                include_device_provenance=False,
            ),
            policy=InitialBatchBudgetPolicy(max_items=1),
        )


def test_recommendation_rejects_footprint_on_different_cuda_device() -> None:
    with pytest.raises(BatchBudgetRecommendationError, match="not represented"):
        recommend_initial_batch_budget(
            ResourceCapacity(
                cpu_total_bytes=1_000,
                cuda_total_bytes=1_000,
                cuda_device_index=0,
            ),
            BatchCost(host_bytes=0, device_bytes=10, num_items=1),
            model_footprint=_model_footprint(
                cuda_bytes=10,
                cuda_device_index=1,
            ),
            policy=InitialBatchBudgetPolicy(max_items=1),
        )


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"min_items": True}, TypeError, "min_items"),
        ({"min_items": 0}, ValueError, "min_items"),
        ({"min_items": 2, "max_items": 1}, ValueError, "must not exceed"),
        ({"host_utilization_ratio": 0}, ValueError, "host_utilization_ratio"),
        ({"device_utilization_ratio": 1.1}, ValueError, "device_utilization_ratio"),
        ({"host_reserve_bytes": -1}, ValueError, "host_reserve_bytes"),
        ({"fallback_max_items": 0}, ValueError, "fallback_max_items"),
    ],
)
def test_initial_budget_policy_rejects_invalid_values(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        InitialBatchBudgetPolicy(**kwargs)  # type: ignore[arg-type]


def test_recommendation_rejects_invalid_argument_types() -> None:
    capacity = ResourceCapacity(cpu_total_bytes=1_000)
    cost = BatchCost(host_bytes=10, device_bytes=0, num_items=1)

    with pytest.raises(TypeError, match="capacity"):
        recommend_initial_batch_budget(object(), cost)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="batch_cost"):
        recommend_initial_batch_budget(capacity, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="model_footprint"):
        recommend_initial_batch_budget(
            capacity,
            cost,
            model_footprint=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="optimizer_footprint"):
        recommend_initial_batch_budget(
            capacity,
            cost,
            optimizer_footprint=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="policy"):
        recommend_initial_batch_budget(
            capacity,
            cost,
            policy=object(),  # type: ignore[arg-type]
        )
