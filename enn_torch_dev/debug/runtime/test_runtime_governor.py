from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from enn_torch_dev.runtime import (
    BatchBudget,
    ConservativeRuntimeGovernor,
    GovernorPolicy,
    ResourceSample,
    RuntimeGovernorState,
    StepResult,
    StepStatus,
)


def _result(
    status: StepStatus = StepStatus.SUCCESS,
    *,
    samples: tuple[ResourceSample, ...] = (),
) -> StepResult:
    return StepResult(
        status=status,
        phase=None,
        batch_size=1,
        row_ids=torch.arange(1),
        resource_samples=samples,
    )


def test_empty_result_stream_keeps_budget() -> None:
    budget = BatchBudget(max_items=8)
    governor = ConservativeRuntimeGovernor(budget)

    decision = governor.observe_results([])

    assert decision.previous_budget == budget
    assert decision.next_budget == budget
    assert decision.statuses == ()
    assert governor.current_budget == budget


def test_single_success_keeps_budget_before_threshold() -> None:
    budget = BatchBudget(max_items=8)
    governor = ConservativeRuntimeGovernor(
        budget,
        policy=GovernorPolicy(grow_after_successes=2),
    )

    decision = governor.observe_results([_result()])

    assert decision.next_budget == budget
    assert decision.consecutive_successes == 1


def test_success_streak_counts_observe_calls_not_result_count() -> None:
    budget = BatchBudget(max_items=8)
    governor = ConservativeRuntimeGovernor(
        budget,
        policy=GovernorPolicy(grow_after_successes=2),
    )

    decision = governor.observe_results([_result(), _result(), _result()])

    assert decision.next_budget == budget
    assert decision.consecutive_successes == 1


def test_consecutive_successes_grow_budget_and_reset_streak() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8, max_host_bytes=100),
        policy=GovernorPolicy(grow_factor=1.5, grow_after_successes=2),
    )

    governor.observe_results([_result()])
    decision = governor.observe_results([_result()])

    assert decision.next_budget == BatchBudget(max_items=12, max_host_bytes=150)
    assert decision.consecutive_successes == 0
    assert governor.current_budget == decision.next_budget


def test_oom_shrinks_all_configured_fields() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=9, max_host_bytes=101, max_device_bytes=201),
        policy=GovernorPolicy(shrink_factor=0.5),
    )

    decision = governor.observe_results([_result(StepStatus.OOM_FAULT)])

    assert decision.next_budget == BatchBudget(
        max_items=4,
        max_host_bytes=50,
        max_device_bytes=100,
    )
    assert decision.consecutive_successes == 0
    assert decision.consecutive_ooms == 1


def test_none_budget_fields_are_not_enabled_by_bounds() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8),
        policy=GovernorPolicy(
            shrink_factor=0.5,
            min_host_bytes=10,
            max_host_bytes=100,
            min_device_bytes=10,
            max_device_bytes=100,
        ),
    )

    decision = governor.observe_results([_result(StepStatus.OOM_FAULT)])

    assert decision.next_budget == BatchBudget(max_items=4)
    assert decision.next_budget.max_host_bytes is None
    assert decision.next_budget.max_device_bytes is None


def test_min_and_max_bounds_are_applied_to_configured_fields() -> None:
    shrink_governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8, max_host_bytes=100),
        policy=GovernorPolicy(shrink_factor=0.1, min_items=3, min_host_bytes=20),
    )
    grow_governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8, max_host_bytes=100),
        policy=GovernorPolicy(
            grow_factor=10,
            grow_after_successes=1,
            max_items=12,
            max_host_bytes=150,
        ),
    )

    shrink_decision = shrink_governor.observe_results([_result(StepStatus.OOM_FAULT)])
    grow_decision = grow_governor.observe_results([_result()])

    assert shrink_decision.next_budget == BatchBudget(max_items=3, max_host_bytes=20)
    assert grow_decision.next_budget == BatchBudget(max_items=12, max_host_bytes=150)


def test_mixed_results_prioritize_oom_shrink() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=10),
        policy=GovernorPolicy(shrink_factor=0.5),
    )

    decision = governor.observe_results(
        [
            _result(StepStatus.SUCCESS),
            _result(StepStatus.DATA_FAULT),
            _result(StepStatus.OOM_FAULT),
        ]
    )

    assert decision.next_budget == BatchBudget(max_items=5)
    assert decision.statuses == (
        StepStatus.SUCCESS,
        StepStatus.DATA_FAULT,
        StepStatus.OOM_FAULT,
    )
    assert decision.consecutive_ooms == 1


def test_non_oom_fault_keeps_budget_and_resets_streaks() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=10),
        policy=GovernorPolicy(grow_after_successes=2),
    )
    governor.observe_results([_result()])

    decision = governor.observe_results([_result(StepStatus.RUNTIME_FAULT)])

    assert decision.next_budget == BatchBudget(max_items=10)
    assert decision.consecutive_successes == 0
    assert decision.consecutive_ooms == 0


def test_resource_sample_peaks_are_recorded_in_decision_and_reason() -> None:
    samples = (
        ResourceSample(
            timestamp_ns=1,
            phase="before",
            cpu_rss_bytes=10,
            cuda_allocated_bytes=20,
            cuda_reserved_bytes=30,
            cuda_max_allocated_bytes=40,
            cuda_max_reserved_bytes=50,
        ),
        ResourceSample(
            timestamp_ns=2,
            phase="after",
            cpu_rss_bytes=11,
            cuda_allocated_bytes=19,
            cuda_reserved_bytes=31,
            cuda_max_allocated_bytes=39,
            cuda_max_reserved_bytes=51,
        ),
    )
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=8))

    decision = governor.observe_results([_result(samples=samples)])

    assert decision.peak_cpu_rss_bytes == 11
    assert decision.peak_cuda_allocated_bytes == 20
    assert decision.peak_cuda_reserved_bytes == 31
    assert decision.peak_cuda_max_allocated_bytes == 40
    assert decision.peak_cuda_max_reserved_bytes == 51
    assert "resource peaks" in decision.reason
    assert "peak_cpu_rss_bytes=11" in decision.reason


@pytest.mark.parametrize("shrink_factor", [0, 1, -0.1, float("inf"), float("nan"), True])
def test_invalid_policy_shrink_factor_is_rejected(shrink_factor: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        GovernorPolicy(shrink_factor=shrink_factor)  # type: ignore[arg-type]


@pytest.mark.parametrize("grow_factor", [1, 0.5, -1, float("inf"), float("nan"), True])
def test_invalid_policy_grow_factor_is_rejected(grow_factor: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        GovernorPolicy(grow_factor=grow_factor)  # type: ignore[arg-type]


@pytest.mark.parametrize("grow_after_successes", [0, -1, 1.5, True])
def test_invalid_policy_grow_after_successes_is_rejected(
    grow_after_successes: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        GovernorPolicy(grow_after_successes=grow_after_successes)  # type: ignore[arg-type]


@pytest.mark.parametrize("bound", [0, -1, 1.5, True])
def test_invalid_policy_bounds_are_rejected(bound: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        GovernorPolicy(min_items=bound)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_items": 5, "max_items": 4},
        {"min_host_bytes": 5, "max_host_bytes": 4},
        {"min_device_bytes": 5, "max_device_bytes": 4},
    ],
)
def test_invalid_policy_bound_order_is_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        GovernorPolicy(**kwargs)


def test_invalid_governor_arguments_are_rejected() -> None:
    budget = BatchBudget(max_items=1)
    state = RuntimeGovernorState(budget)

    with pytest.raises(ValueError):
        ConservativeRuntimeGovernor()
    with pytest.raises(ValueError):
        ConservativeRuntimeGovernor(budget, state=state)
    with pytest.raises(TypeError):
        ConservativeRuntimeGovernor(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ConservativeRuntimeGovernor(budget, policy=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ConservativeRuntimeGovernor(state=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ConservativeRuntimeGovernor(BatchBudget(max_items=0))


def test_explicit_state_is_immutable_and_replaced_after_observe() -> None:
    original_state = RuntimeGovernorState(
        current_budget=BatchBudget(max_items=8),
        consecutive_successes=1,
    )
    governor = ConservativeRuntimeGovernor(
        state=original_state,
        policy=GovernorPolicy(grow_after_successes=2),
    )

    with pytest.raises(FrozenInstanceError):
        original_state.consecutive_successes = 3  # type: ignore[misc]

    decision = governor.observe_results([_result()])

    assert original_state.current_budget == BatchBudget(max_items=8)
    assert original_state.consecutive_successes == 1
    assert original_state.last_decision is None
    assert governor.state is not original_state
    assert governor.state.current_budget == BatchBudget(max_items=16)
    assert governor.state.consecutive_successes == 0
    assert governor.state.last_decision == decision
