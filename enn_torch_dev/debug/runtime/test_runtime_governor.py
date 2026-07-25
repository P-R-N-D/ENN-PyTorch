from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from enn_torch_dev.runtime import (
    BatchBudget,
    ConservativeRuntimeGovernor,
    GovernorDecision,
    GovernorPolicy,
    ResourcePressureSummary,
    ResourceSample,
    RuntimeGovernorState,
    StepResult,
    StepStatus,
)


def _result(
    status: StepStatus = StepStatus.SUCCESS,
    *,
    samples: tuple[ResourceSample, ...] = (),
    loss: torch.Tensor | None = None,
    store: object | None = None,
) -> StepResult:
    return StepResult(
        status=status,
        phase=None,
        batch_size=1,
        row_ids=torch.arange(1),
        loss=loss,
        store=store,  # type: ignore[arg-type]
        resource_samples=samples,
    )


def _field_values(instance: object) -> list[object]:
    return [getattr(instance, field.name) for field in fields(instance)]


def _reason_high_dimensions(reason: str) -> str:
    return (
        reason
        .split("configured shrink limit for dimensions: ", 1)[1]
        .split(";", 1)[0]
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


def test_governor_policy_preserves_existing_positional_field_order() -> None:
    policy = GovernorPolicy(0.5, 2.0, 3, 1, 8, 10, 100, 20, 200)
    field_names = [field.name for field in fields(GovernorPolicy)]

    assert policy.max_device_bytes == 200
    assert policy.max_pressure_ratio_for_growth is None
    assert field_names[-6:] == [
        "min_cpu_pressure_ratio_for_shrink",
        "min_cuda_pressure_ratio_for_shrink",
        "cpu_shrink_after_pressure_passes",
        "cuda_shrink_after_pressure_passes",
        "cpu_pressure_shrink_factor",
        "cuda_pressure_shrink_factor",
    ]
    assert policy.min_cpu_pressure_ratio_for_shrink is None
    assert policy.min_cuda_pressure_ratio_for_shrink is None
    assert policy.cpu_shrink_after_pressure_passes is None
    assert policy.cuda_shrink_after_pressure_passes is None
    assert policy.cpu_pressure_shrink_factor is None
    assert policy.cuda_pressure_shrink_factor is None


def test_governor_decision_appends_pressure_field_selection_for_compatibility() -> None:
    decision_field_names = [field.name for field in fields(GovernorDecision)]
    state_field_names = [field.name for field in fields(RuntimeGovernorState)]

    assert decision_field_names[-9:] == [
        "consecutive_high_pressure_passes",
        "budget_shrunk_by_pressure",
        "pressure_shrunk_budget_fields",
        "consecutive_cpu_pressure_passes",
        "consecutive_cuda_pressure_passes",
        "pressure_high_dimensions",
        "pressure_triggered_dimensions",
        "pressure_selected_budget_fields",
        "pressure_applied_shrink_factors",
    ]
    assert state_field_names[-3:] == [
        "consecutive_high_pressure_passes",
        "consecutive_cpu_pressure_passes",
        "consecutive_cuda_pressure_passes",
    ]


def test_pressure_guard_is_opt_in_and_default_growth_is_unchanged() -> None:
    pressure = ResourcePressureSummary(peak_cpu_rss_ratio=1.0)
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8),
        policy=GovernorPolicy(grow_after_successes=1),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=pressure,
    )

    assert decision.next_budget == BatchBudget(max_items=16)
    assert decision.pressure_summary == pressure
    assert decision.growth_suppressed_by_pressure is False


def test_sustained_pressure_shrink_is_opt_in() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8), policy=GovernorPolicy(grow_after_successes=1)
    )

    decision = governor.observe_results(
        [_result()], pressure_summary=ResourcePressureSummary(peak_cpu_rss_ratio=0.95)
    )

    assert decision.next_budget == BatchBudget(max_items=16)
    assert decision.consecutive_high_pressure_passes == 0
    assert decision.budget_shrunk_by_pressure is False


def test_sustained_high_pressure_shrinks_after_configured_pass_count() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8, max_host_bytes=100, max_device_bytes=200),
        policy=GovernorPolicy(
            shrink_factor=0.5,
            grow_after_successes=1,
            max_pressure_ratio_for_growth=0.8,
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=2,
        ),
    )
    pressure = ResourcePressureSummary(peak_cuda_reserved_ratio=0.95)

    first = governor.observe_results([_result()], pressure_summary=pressure)
    second = governor.observe_results([_result()], pressure_summary=pressure)

    assert first.next_budget == BatchBudget(max_items=8, max_host_bytes=100, max_device_bytes=200)
    assert first.consecutive_high_pressure_passes == 1
    assert first.growth_suppressed_by_pressure is True
    assert first.budget_shrunk_by_pressure is False
    assert second.next_budget == BatchBudget(
        max_items=8,
        max_host_bytes=100,
        max_device_bytes=100,
    )
    assert second.consecutive_high_pressure_passes == 0
    assert second.budget_shrunk_by_pressure is True
    assert second.pressure_shrunk_budget_fields == ("max_device_bytes",)


def test_sustained_cpu_pressure_shrinks_host_budget_only() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8, max_host_bytes=100, max_device_bytes=200),
        policy=GovernorPolicy(
            shrink_factor=0.5,
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=1,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=ResourcePressureSummary(peak_cpu_rss_ratio=0.95),
    )

    assert decision.next_budget == BatchBudget(
        max_items=8,
        max_host_bytes=50,
        max_device_bytes=200,
    )
    assert decision.pressure_shrunk_budget_fields == ("max_host_bytes",)


@pytest.mark.parametrize(
    "ratio_field",
    [
        "peak_cuda_allocated_ratio",
        "peak_cuda_reserved_ratio",
        "peak_cuda_max_allocated_ratio",
        "peak_cuda_max_reserved_ratio",
    ],
)
def test_sustained_cuda_pressure_shrinks_device_budget_only(
    ratio_field: str,
) -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8, max_host_bytes=100, max_device_bytes=200),
        policy=GovernorPolicy(
            shrink_factor=0.5,
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=1,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=ResourcePressureSummary(**{ratio_field: 0.95}),
    )

    assert decision.next_budget == BatchBudget(
        max_items=8,
        max_host_bytes=100,
        max_device_bytes=100,
    )
    assert decision.pressure_shrunk_budget_fields == ("max_device_bytes",)


def test_sustained_cpu_and_cuda_pressure_shrink_both_byte_budgets() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8, max_host_bytes=100, max_device_bytes=200),
        policy=GovernorPolicy(
            shrink_factor=0.5,
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=1,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=ResourcePressureSummary(
            peak_cpu_rss_ratio=0.95,
            peak_cuda_reserved_ratio=0.96,
        ),
    )

    assert decision.next_budget == BatchBudget(
        max_items=8,
        max_host_bytes=50,
        max_device_bytes=100,
    )
    assert decision.pressure_shrunk_budget_fields == (
        "max_host_bytes",
        "max_device_bytes",
    )


def test_pressure_shrink_factors_fall_back_to_common_factor() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8, max_host_bytes=100, max_device_bytes=200),
        policy=GovernorPolicy(
            shrink_factor=0.6,
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=1,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=ResourcePressureSummary(
            peak_cpu_rss_ratio=0.95,
            peak_cuda_reserved_ratio=0.96,
        ),
    )

    assert decision.next_budget == BatchBudget(
        max_items=8, max_host_bytes=60, max_device_bytes=120
    )
    assert "triggered shrink factors: cpu=0.6, cuda=0.6" in decision.reason


@pytest.mark.parametrize(
    ("pressure", "policy_kwargs", "expected_budget", "factor_text"),
    [
        (
            ResourcePressureSummary(peak_cpu_rss_ratio=0.95),
            {"cpu_pressure_shrink_factor": 0.75},
            BatchBudget(max_items=8, max_host_bytes=75, max_device_bytes=200),
            "cpu=0.75",
        ),
        (
            ResourcePressureSummary(peak_cuda_reserved_ratio=0.95),
            {"cuda_pressure_shrink_factor": 0.4},
            BatchBudget(max_items=8, max_host_bytes=100, max_device_bytes=80),
            "cuda=0.4",
        ),
    ],
)
def test_dimension_pressure_factor_applies_to_matching_byte_budget(
    pressure: ResourcePressureSummary,
    policy_kwargs: dict[str, float],
    expected_budget: BatchBudget,
    factor_text: str,
) -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8, max_host_bytes=100, max_device_bytes=200),
        policy=GovernorPolicy(
            shrink_factor=0.5,
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=1,
            **policy_kwargs,
        ),
    )

    decision = governor.observe_results([_result()], pressure_summary=pressure)

    assert decision.next_budget == expected_budget
    assert f"triggered shrink factors: {factor_text}" in decision.reason


def test_cpu_and_cuda_pressure_factors_apply_independently() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8, max_host_bytes=100, max_device_bytes=200),
        policy=GovernorPolicy(
            cpu_pressure_shrink_factor=0.75,
            cuda_pressure_shrink_factor=0.4,
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=1,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=ResourcePressureSummary(
            peak_cpu_rss_ratio=0.95, peak_cuda_reserved_ratio=0.96
        ),
    )

    assert decision.next_budget == BatchBudget(
        max_items=8, max_host_bytes=75, max_device_bytes=80
    )
    assert decision.pressure_shrunk_budget_fields == (
        "max_host_bytes", "max_device_bytes"
    )
    assert decision.pressure_high_dimensions == ("cpu", "cuda")
    assert decision.pressure_triggered_dimensions == ("cpu", "cuda")
    assert decision.pressure_selected_budget_fields == (
        "max_host_bytes", "max_device_bytes"
    )
    assert decision.pressure_applied_shrink_factors == (
        ("max_host_bytes", 0.75), ("max_device_bytes", 0.4)
    )
    assert "triggered shrink factors: cpu=0.75, cuda=0.4" in decision.reason


def test_dimension_threshold_override_can_activate_cpu_only() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8, max_host_bytes=100, max_device_bytes=200),
        policy=GovernorPolicy(
            shrink_factor=0.5,
            min_cpu_pressure_ratio_for_shrink=0.8,
            cpu_shrink_after_pressure_passes=1,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=ResourcePressureSummary(
            peak_cpu_rss_ratio=0.85,
            peak_cuda_reserved_ratio=1.20,
        ),
    )

    assert decision.next_budget == BatchBudget(
        max_items=8,
        max_host_bytes=50,
        max_device_bytes=200,
    )
    assert decision.consecutive_cpu_pressure_passes == 0
    assert decision.consecutive_cuda_pressure_passes == 0
    assert decision.pressure_shrunk_budget_fields == ("max_host_bytes",)
    assert "triggered policies: cpu(limit=0.8, required=1)" in decision.reason
    assert "cuda=" not in decision.reason.split("current triggered ratios: ", 1)[1].split(";", 1)[0]


def test_dimension_thresholds_and_pass_counts_trigger_independently() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8, max_host_bytes=100, max_device_bytes=200),
        policy=GovernorPolicy(
            shrink_factor=0.5,
            min_cpu_pressure_ratio_for_shrink=0.8,
            min_cuda_pressure_ratio_for_shrink=0.95,
            cpu_shrink_after_pressure_passes=2,
            cuda_shrink_after_pressure_passes=3,
        ),
    )
    pressure = ResourcePressureSummary(
        peak_cpu_rss_ratio=0.85,
        peak_cuda_reserved_ratio=0.96,
    )

    first = governor.observe_results([_result()], pressure_summary=pressure)
    second = governor.observe_results([_result()], pressure_summary=pressure)
    third = governor.observe_results([_result()], pressure_summary=pressure)

    assert first.next_budget == BatchBudget(
        max_items=8,
        max_host_bytes=100,
        max_device_bytes=200,
    )
    assert first.consecutive_cpu_pressure_passes == 1
    assert first.consecutive_cuda_pressure_passes == 1
    assert _reason_high_dimensions(first.reason) == "cpu, cuda"
    assert first.pressure_high_dimensions == ("cpu", "cuda")
    assert first.pressure_triggered_dimensions == ()
    assert first.pressure_selected_budget_fields == ()
    assert first.pressure_applied_shrink_factors == ()

    assert second.next_budget == BatchBudget(
        max_items=8,
        max_host_bytes=50,
        max_device_bytes=200,
    )
    assert second.pressure_shrunk_budget_fields == ("max_host_bytes",)
    assert second.consecutive_cpu_pressure_passes == 0
    assert second.consecutive_cuda_pressure_passes == 2
    assert "triggered policies: cpu(limit=0.8, required=2)" in second.reason
    assert second.pressure_high_dimensions == ("cpu", "cuda")
    assert second.pressure_triggered_dimensions == ("cpu",)
    assert second.pressure_selected_budget_fields == ("max_host_bytes",)
    assert second.pressure_applied_shrink_factors == (("max_host_bytes", 0.5),)

    assert third.next_budget == BatchBudget(
        max_items=8,
        max_host_bytes=50,
        max_device_bytes=100,
    )
    assert third.pressure_shrunk_budget_fields == ("max_device_bytes",)
    assert third.consecutive_cpu_pressure_passes == 1
    assert third.consecutive_cuda_pressure_passes == 0
    assert "triggered policies: cuda(limit=0.95, required=3)" in third.reason
    assert third.pressure_high_dimensions == ("cpu", "cuda")
    assert third.pressure_triggered_dimensions == ("cuda",)
    assert third.pressure_selected_budget_fields == ("max_device_bytes",)
    assert third.pressure_applied_shrink_factors == (("max_device_bytes", 0.5),)


def test_dimension_threshold_override_resets_only_non_high_dimension() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8, max_host_bytes=100, max_device_bytes=200),
        policy=GovernorPolicy(
            min_cpu_pressure_ratio_for_shrink=0.8,
            min_cuda_pressure_ratio_for_shrink=0.95,
            cpu_shrink_after_pressure_passes=2,
            cuda_shrink_after_pressure_passes=3,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=ResourcePressureSummary(
            peak_cpu_rss_ratio=0.85,
            peak_cuda_reserved_ratio=0.90,
        ),
    )

    assert decision.next_budget == BatchBudget(
        max_items=8,
        max_host_bytes=100,
        max_device_bytes=200,
    )
    assert decision.consecutive_cpu_pressure_passes == 1
    assert decision.consecutive_cuda_pressure_passes == 0
    assert decision.growth_suppressed_by_pressure is True
    assert "cpu=1/2 (limit=0.8, ratio=0.85)" in decision.reason
    assert "cuda=0/3 (limit=0.95, ratio=0.9)" in decision.reason
    assert _reason_high_dimensions(decision.reason) == "cpu"


def test_threshold_override_uses_common_pass_count_fallback() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8, max_host_bytes=100, max_device_bytes=200),
        policy=GovernorPolicy(
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=3,
            min_cpu_pressure_ratio_for_shrink=0.8,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=ResourcePressureSummary(
            peak_cpu_rss_ratio=0.85,
            peak_cuda_reserved_ratio=0.85,
        ),
    )

    assert decision.consecutive_cpu_pressure_passes == 1
    assert decision.consecutive_cuda_pressure_passes == 0
    assert decision.budget_shrunk_by_pressure is False
    assert "cpu=1/3 (limit=0.8, ratio=0.85)" in decision.reason
    assert _reason_high_dimensions(decision.reason) == "cpu"


def test_pass_count_override_uses_common_threshold_fallback() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8, max_host_bytes=100, max_device_bytes=200),
        policy=GovernorPolicy(
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=3,
            cuda_shrink_after_pressure_passes=1,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=ResourcePressureSummary(
            peak_cpu_rss_ratio=0.95,
            peak_cuda_reserved_ratio=0.95,
        ),
    )

    assert decision.next_budget == BatchBudget(
        max_items=8,
        max_host_bytes=100,
        max_device_bytes=100,
    )
    assert decision.consecutive_cpu_pressure_passes == 1
    assert decision.consecutive_cuda_pressure_passes == 0
    assert decision.pressure_shrunk_budget_fields == ("max_device_bytes",)
    assert "cuda(limit=0.9, required=1)" in decision.reason


@pytest.mark.parametrize(
    ("first_pressure", "second_pressure", "expected_cpu", "expected_cuda"),
    [
        (
            ResourcePressureSummary(peak_cpu_rss_ratio=0.95),
            ResourcePressureSummary(peak_cuda_reserved_ratio=0.95),
            0,
            1,
        ),
        (
            ResourcePressureSummary(peak_cuda_reserved_ratio=0.95),
            ResourcePressureSummary(peak_cpu_rss_ratio=0.95),
            1,
            0,
        ),
    ],
)
def test_alternating_pressure_dimensions_do_not_share_sustained_streak(
    first_pressure: ResourcePressureSummary,
    second_pressure: ResourcePressureSummary,
    expected_cpu: int,
    expected_cuda: int,
) -> None:
    budget = BatchBudget(max_items=8, max_host_bytes=100, max_device_bytes=200)
    governor = ConservativeRuntimeGovernor(
        budget,
        policy=GovernorPolicy(
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=2,
        ),
    )

    governor.observe_results([_result()], pressure_summary=first_pressure)
    decision = governor.observe_results([_result()], pressure_summary=second_pressure)

    assert decision.next_budget == budget
    assert decision.budget_shrunk_by_pressure is False
    assert decision.consecutive_cpu_pressure_passes == expected_cpu
    assert decision.consecutive_cuda_pressure_passes == expected_cuda
    assert decision.consecutive_high_pressure_passes == max(expected_cpu, expected_cuda)


def test_cpu_trigger_preserves_incomplete_cuda_pressure_streak() -> None:
    governor = ConservativeRuntimeGovernor(
        state=RuntimeGovernorState(
            current_budget=BatchBudget(max_items=8, max_host_bytes=100, max_device_bytes=200),
            consecutive_high_pressure_passes=1,
            consecutive_cpu_pressure_passes=1,
        ),
        policy=GovernorPolicy(
            shrink_factor=0.5,
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=2,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=ResourcePressureSummary(
            peak_cpu_rss_ratio=0.95,
            peak_cuda_reserved_ratio=0.95,
        ),
    )

    assert decision.next_budget == BatchBudget(
        max_items=8, max_host_bytes=50, max_device_bytes=200
    )
    assert decision.consecutive_cpu_pressure_passes == 0
    assert decision.consecutive_cuda_pressure_passes == 1
    assert decision.consecutive_high_pressure_passes == 1


@pytest.mark.parametrize("pressure", [None, ResourcePressureSummary()])
def test_fully_unknown_pressure_resets_both_dimension_streaks(
    pressure: ResourcePressureSummary | None,
) -> None:
    governor = ConservativeRuntimeGovernor(
        state=RuntimeGovernorState(
            current_budget=BatchBudget(max_items=8),
            consecutive_high_pressure_passes=1,
            consecutive_cpu_pressure_passes=1,
            consecutive_cuda_pressure_passes=1,
        ),
        policy=GovernorPolicy(min_pressure_ratio_for_shrink=0.9),
    )

    decision = governor.observe_results([_result()], pressure_summary=pressure)

    assert decision.consecutive_cpu_pressure_passes == 0
    assert decision.consecutive_cuda_pressure_passes == 0
    assert decision.consecutive_high_pressure_passes == 0


@pytest.mark.parametrize(
    ("budget", "pressure", "expected_budget", "expected_field"),
    [
        (
            BatchBudget(max_items=8, max_host_bytes=100),
            ResourcePressureSummary(peak_cpu_rss_ratio=0.95),
            BatchBudget(max_items=8, max_host_bytes=50),
            "max_host_bytes",
        ),
        (
            BatchBudget(max_items=8, max_device_bytes=200),
            ResourcePressureSummary(peak_cuda_reserved_ratio=0.95),
            BatchBudget(max_items=8, max_device_bytes=100),
            "max_device_bytes",
        ),
    ],
)
def test_legacy_global_pressure_streak_is_inherited_by_one_high_dimension(
    budget: BatchBudget,
    pressure: ResourcePressureSummary,
    expected_budget: BatchBudget,
    expected_field: str,
) -> None:
    governor = ConservativeRuntimeGovernor(
        state=RuntimeGovernorState(
            current_budget=budget,
            consecutive_high_pressure_passes=1,
        ),
        policy=GovernorPolicy(
            shrink_factor=0.5,
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=2,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=pressure,
    )

    assert decision.next_budget == expected_budget
    assert decision.pressure_shrunk_budget_fields == (expected_field,)
    assert decision.consecutive_high_pressure_passes == 0


def test_ambiguous_legacy_pressure_streak_is_not_inherited_by_both_dimensions() -> None:
    governor = ConservativeRuntimeGovernor(
        state=RuntimeGovernorState(
            current_budget=BatchBudget(
                max_items=8,
                max_host_bytes=100,
                max_device_bytes=200,
            ),
            consecutive_high_pressure_passes=1,
        ),
        policy=GovernorPolicy(
            shrink_factor=0.5,
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=2,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=ResourcePressureSummary(
            peak_cpu_rss_ratio=0.95,
            peak_cuda_reserved_ratio=0.96,
        ),
    )

    assert decision.next_budget == BatchBudget(
        max_items=8,
        max_host_bytes=100,
        max_device_bytes=200,
    )
    assert decision.consecutive_cpu_pressure_passes == 1
    assert decision.consecutive_cuda_pressure_passes == 1
    assert decision.consecutive_high_pressure_passes == 1
    assert decision.budget_shrunk_by_pressure is False
    assert decision.pressure_shrunk_budget_fields == ()


def test_trigger_reason_reports_only_triggered_dimension_ratios() -> None:
    governor = ConservativeRuntimeGovernor(
        state=RuntimeGovernorState(
            current_budget=BatchBudget(
                max_items=8,
                max_host_bytes=100,
                max_device_bytes=200,
            ),
            consecutive_high_pressure_passes=1,
            consecutive_cpu_pressure_passes=1,
        ),
        policy=GovernorPolicy(
            shrink_factor=0.5,
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=2,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=ResourcePressureSummary(
            peak_cpu_rss_ratio=0.95,
            peak_cuda_reserved_ratio=1.20,
        ),
    )

    assert decision.next_budget == BatchBudget(
        max_items=8,
        max_host_bytes=50,
        max_device_bytes=200,
    )
    assert decision.pressure_shrunk_budget_fields == ("max_host_bytes",)
    assert "triggered dimensions: cpu" in decision.reason
    assert "cpu=0.95" in decision.reason
    triggered_ratio_text = decision.reason.split("current triggered ratios: ", 1)[1].split(";", 1)[0]
    assert "cuda=" not in triggered_ratio_text
    assert "resource pressure 1.2 remained" not in decision.reason


def test_explicit_dimension_streak_does_not_reuse_legacy_aggregate() -> None:
    budget = BatchBudget(max_items=8, max_host_bytes=100)
    governor = ConservativeRuntimeGovernor(
        state=RuntimeGovernorState(
            current_budget=budget,
            consecutive_high_pressure_passes=5,
            consecutive_cpu_pressure_passes=1,
        ),
        policy=GovernorPolicy(
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=3,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=ResourcePressureSummary(peak_cpu_rss_ratio=0.95),
    )

    assert decision.next_budget == budget
    assert decision.consecutive_cpu_pressure_passes == 2
    assert decision.consecutive_high_pressure_passes == 2


@pytest.mark.parametrize(
    "pressure",
    [
        ResourcePressureSummary(peak_cpu_rss_ratio=0.95),
        ResourcePressureSummary(peak_cuda_reserved_ratio=0.95),
    ],
)
def test_sustained_pressure_uses_items_only_without_matching_byte_budget(
    pressure: ResourcePressureSummary,
) -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8),
        policy=GovernorPolicy(
            shrink_factor=0.5,
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=1,
        ),
    )

    decision = governor.observe_results([_result()], pressure_summary=pressure)

    assert decision.next_budget == BatchBudget(max_items=4)
    assert decision.pressure_shrunk_budget_fields == ("max_items",)


@pytest.mark.parametrize(
    ("pressure", "policy_kwargs", "expected_items"),
    [
        (ResourcePressureSummary(peak_cpu_rss_ratio=0.95), {"cpu_pressure_shrink_factor": 0.75}, 6),
        (ResourcePressureSummary(peak_cuda_reserved_ratio=0.95), {"cuda_pressure_shrink_factor": 0.4}, 3),
    ],
)
def test_dimension_pressure_factor_applies_to_items_fallback(
    pressure: ResourcePressureSummary,
    policy_kwargs: dict[str, float],
    expected_items: int,
) -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8),
        policy=GovernorPolicy(
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=1,
            **policy_kwargs,
        ),
    )

    decision = governor.observe_results([_result()], pressure_summary=pressure)

    assert decision.next_budget == BatchBudget(max_items=expected_items)
    assert decision.pressure_shrunk_budget_fields == ("max_items",)


def test_dual_pressure_uses_stronger_factor_for_shared_items_fallback() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=10),
        policy=GovernorPolicy(
            cpu_pressure_shrink_factor=0.75,
            cuda_pressure_shrink_factor=0.4,
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=1,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=ResourcePressureSummary(
            peak_cpu_rss_ratio=0.95, peak_cuda_reserved_ratio=0.96
        ),
    )

    assert decision.next_budget == BatchBudget(max_items=4)
    assert decision.pressure_shrunk_budget_fields == ("max_items",)
    assert decision.pressure_high_dimensions == ("cpu", "cuda")
    assert decision.pressure_triggered_dimensions == ("cpu", "cuda")
    assert decision.pressure_selected_budget_fields == ("max_items",)
    assert decision.pressure_applied_shrink_factors == (("max_items", 0.4),)


def test_trigger_without_matching_budget_records_dimensions_but_no_field_metadata() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_device_bytes=200),
        policy=GovernorPolicy(
            min_cpu_pressure_ratio_for_shrink=0.9,
            cpu_shrink_after_pressure_passes=1,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=ResourcePressureSummary(peak_cpu_rss_ratio=0.95),
    )

    assert decision.next_budget == BatchBudget(max_device_bytes=200)
    assert decision.pressure_high_dimensions == ("cpu",)
    assert decision.pressure_triggered_dimensions == ("cpu",)
    assert decision.pressure_selected_budget_fields == ()
    assert decision.pressure_applied_shrink_factors == ()
    assert decision.pressure_shrunk_budget_fields == ()
    assert decision.budget_shrunk_by_pressure is False
    assert "no matching byte budget" in decision.reason


def test_cpu_pressure_uses_items_fallback_with_only_device_budget() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8, max_device_bytes=200),
        policy=GovernorPolicy(
            shrink_factor=0.5,
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=1,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=ResourcePressureSummary(peak_cpu_rss_ratio=0.95),
    )

    assert decision.next_budget == BatchBudget(max_items=4, max_device_bytes=200)
    assert decision.budget_shrunk_by_pressure is True
    assert decision.pressure_shrunk_budget_fields == ("max_items",)


def test_matching_host_budget_prevents_items_fallback_for_cuda_pressure() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8, max_host_bytes=100),
        policy=GovernorPolicy(
            shrink_factor=0.5,
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=1,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=ResourcePressureSummary(
            peak_cpu_rss_ratio=0.95,
            peak_cuda_reserved_ratio=0.95,
        ),
    )

    assert decision.next_budget == BatchBudget(max_items=8, max_host_bytes=50)
    assert decision.budget_shrunk_by_pressure is True
    assert decision.pressure_shrunk_budget_fields == ("max_host_bytes",)


@pytest.mark.parametrize(
    ("results", "pressure"),
    [
        ([_result()], ResourcePressureSummary(peak_cpu_rss_ratio=0.5)),
        ([_result()], None),
        ([_result()], ResourcePressureSummary()),
        ([_result(StepStatus.RUNTIME_FAULT)], ResourcePressureSummary(peak_cpu_rss_ratio=0.95)),
        ([], ResourcePressureSummary(peak_cpu_rss_ratio=0.95)),
    ],
)
def test_non_high_pressure_observations_reset_pressure_streak(
    results: list[StepResult], pressure: ResourcePressureSummary | None
) -> None:
    governor = ConservativeRuntimeGovernor(
        state=RuntimeGovernorState(
            current_budget=BatchBudget(max_items=8), consecutive_high_pressure_passes=1
        ),
        policy=GovernorPolicy(min_pressure_ratio_for_shrink=0.9),
    )

    decision = governor.observe_results(results, pressure_summary=pressure)

    assert decision.consecutive_high_pressure_passes == 0


def test_pressure_shrink_respects_minimum_bounds() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=4),
        policy=GovernorPolicy(
            shrink_factor=0.5,
            min_items=4,
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=1,
        ),
    )

    decision = governor.observe_results(
        [_result()], pressure_summary=ResourcePressureSummary(peak_cpu_rss_ratio=0.95)
    )

    assert decision.next_budget == BatchBudget(max_items=4)
    assert decision.budget_shrunk_by_pressure is False
    assert decision.pressure_shrunk_budget_fields == ()
    assert "minimum bounds" in decision.reason


def test_pressure_shrink_records_only_fields_that_actually_changed() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8, max_host_bytes=100, max_device_bytes=200),
        policy=GovernorPolicy(
            shrink_factor=0.5,
            cpu_pressure_shrink_factor=0.75,
            min_host_bytes=100,
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=1,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=ResourcePressureSummary(
            peak_cpu_rss_ratio=0.95,
            peak_cuda_reserved_ratio=0.95,
        ),
    )

    assert decision.next_budget == BatchBudget(
        max_items=8,
        max_host_bytes=100,
        max_device_bytes=100,
    )
    assert decision.budget_shrunk_by_pressure is True
    assert decision.pressure_shrunk_budget_fields == ("max_device_bytes",)
    assert decision.pressure_high_dimensions == ("cpu", "cuda")
    assert decision.pressure_triggered_dimensions == ("cpu", "cuda")
    assert decision.pressure_selected_budget_fields == (
        "max_host_bytes", "max_device_bytes"
    )
    assert decision.pressure_applied_shrink_factors == (
        ("max_host_bytes", 0.75), ("max_device_bytes", 0.5)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_pressure_ratio_for_shrink": 0},
        {"min_pressure_ratio_for_shrink": 1.1},
        {"min_pressure_ratio_for_shrink": True},
        {"shrink_after_pressure_passes": 0},
        {"shrink_after_pressure_passes": True},
        {"min_cpu_pressure_ratio_for_shrink": 0},
        {"min_cpu_pressure_ratio_for_shrink": 1.1},
        {"min_cpu_pressure_ratio_for_shrink": True},
        {"min_cuda_pressure_ratio_for_shrink": 0},
        {"min_cuda_pressure_ratio_for_shrink": 1.1},
        {"min_cuda_pressure_ratio_for_shrink": True},
        {"cpu_shrink_after_pressure_passes": 0},
        {"cpu_shrink_after_pressure_passes": True},
        {"cuda_shrink_after_pressure_passes": 0},
        {"cuda_shrink_after_pressure_passes": True},
        {"cpu_pressure_shrink_factor": 0},
        {"cpu_pressure_shrink_factor": 1},
        {"cpu_pressure_shrink_factor": 1.1},
        {"cpu_pressure_shrink_factor": True},
        {"cpu_pressure_shrink_factor": float("nan")},
        {"cpu_pressure_shrink_factor": float("inf")},
        {"cuda_pressure_shrink_factor": 0},
        {"cuda_pressure_shrink_factor": 1},
        {"cuda_pressure_shrink_factor": 1.1},
        {"cuda_pressure_shrink_factor": True},
        {"cuda_pressure_shrink_factor": float("nan")},
        {"cuda_pressure_shrink_factor": float("inf")},
        {"max_pressure_ratio_for_growth": 0.9, "min_pressure_ratio_for_shrink": 0.8},
        {
            "max_pressure_ratio_for_growth": 0.9,
            "min_cpu_pressure_ratio_for_shrink": 0.8,
        },
        {
            "max_pressure_ratio_for_growth": 0.9,
            "min_cuda_pressure_ratio_for_shrink": 0.8,
        },
    ],
)
def test_invalid_sustained_pressure_policy_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        GovernorPolicy(**kwargs)


def test_dimension_overrides_fall_back_to_common_policy_values() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8, max_host_bytes=100, max_device_bytes=200),
        policy=GovernorPolicy(
            shrink_factor=0.5,
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=2,
        ),
    )
    pressure = ResourcePressureSummary(
        peak_cpu_rss_ratio=0.95,
        peak_cuda_reserved_ratio=0.96,
    )

    first = governor.observe_results([_result()], pressure_summary=pressure)
    second = governor.observe_results([_result()], pressure_summary=pressure)

    assert first.consecutive_cpu_pressure_passes == 1
    assert first.consecutive_cuda_pressure_passes == 1
    assert second.next_budget == BatchBudget(
        max_items=8,
        max_host_bytes=50,
        max_device_bytes=100,
    )
    assert second.pressure_shrunk_budget_fields == (
        "max_host_bytes",
        "max_device_bytes",
    )
    assert "cpu(limit=0.9, required=2)" in second.reason
    assert "cuda(limit=0.9, required=2)" in second.reason


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


def test_pressure_below_limit_allows_success_growth() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8),
        policy=GovernorPolicy(
            grow_after_successes=1,
            max_pressure_ratio_for_growth=0.8,
        ),
    )
    pressure = ResourcePressureSummary(peak_cpu_rss_ratio=0.79)

    decision = governor.observe_results(
        [_result()],
        pressure_summary=pressure,
    )

    assert decision.next_budget == BatchBudget(max_items=16)
    assert decision.consecutive_successes == 0
    assert decision.pressure_summary == pressure
    assert decision.growth_suppressed_by_pressure is False


@pytest.mark.parametrize("ratio", [0.8, 0.9, 1.2])
def test_pressure_at_or_above_limit_suppresses_growth(ratio: float) -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8),
        policy=GovernorPolicy(
            grow_after_successes=1,
            max_pressure_ratio_for_growth=0.8,
        ),
    )
    pressure = ResourcePressureSummary(peak_cuda_reserved_ratio=ratio)

    decision = governor.observe_results(
        [_result()],
        pressure_summary=pressure,
    )

    assert decision.next_budget == BatchBudget(max_items=8)
    assert decision.consecutive_successes == 0
    assert decision.pressure_summary == pressure
    assert decision.growth_suppressed_by_pressure is True
    assert "reached growth limit" in decision.reason


@pytest.mark.parametrize(
    "pressure",
    [None, ResourcePressureSummary()],
)
def test_missing_pressure_suppresses_growth_when_guard_is_enabled(
    pressure: ResourcePressureSummary | None,
) -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8),
        policy=GovernorPolicy(
            grow_after_successes=1,
            max_pressure_ratio_for_growth=0.8,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=pressure,
    )

    assert decision.next_budget == BatchBudget(max_items=8)
    assert decision.consecutive_successes == 0
    assert decision.growth_suppressed_by_pressure is True
    assert "pressure is unavailable" in decision.reason


def test_pressure_guard_uses_highest_known_ratio() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8),
        policy=GovernorPolicy(
            grow_after_successes=1,
            max_pressure_ratio_for_growth=0.8,
        ),
    )
    pressure = ResourcePressureSummary(
        peak_cpu_rss_ratio=0.2,
        peak_cuda_allocated_ratio=0.81,
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=pressure,
    )

    assert decision.next_budget == BatchBudget(max_items=8)
    assert decision.growth_suppressed_by_pressure is True


def test_pressure_suppression_resets_existing_success_streak() -> None:
    governor = ConservativeRuntimeGovernor(
        state=RuntimeGovernorState(
            current_budget=BatchBudget(max_items=8),
            consecutive_successes=2,
        ),
        policy=GovernorPolicy(
            grow_after_successes=3,
            max_pressure_ratio_for_growth=0.8,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=ResourcePressureSummary(peak_cpu_rss_ratio=0.8),
    )

    assert decision.next_budget == BatchBudget(max_items=8)
    assert decision.consecutive_successes == 0
    assert decision.growth_suppressed_by_pressure is True


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
    assert decision.pressure_shrunk_budget_fields == ()


def test_oom_uses_common_shrink_factor_not_pressure_overrides() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=10, max_host_bytes=100, max_device_bytes=200),
        policy=GovernorPolicy(
            shrink_factor=0.5,
            cpu_pressure_shrink_factor=0.75,
            cuda_pressure_shrink_factor=0.4,
        ),
    )

    decision = governor.observe_results([_result(StepStatus.OOM_FAULT)])

    assert decision.next_budget == BatchBudget(
        max_items=5, max_host_bytes=50, max_device_bytes=100
    )
    assert decision.pressure_shrunk_budget_fields == ()
    assert decision.pressure_high_dimensions == ()
    assert decision.pressure_triggered_dimensions == ()
    assert decision.pressure_selected_budget_fields == ()
    assert decision.pressure_applied_shrink_factors == ()
    assert "triggered shrink factors" not in decision.reason

    recovered_governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=10, max_host_bytes=100, max_device_bytes=200),
        policy=governor.policy,
    )
    recovered_decision = recovered_governor.observe_results(
        [_result()], recovered_oom=True
    )

    assert recovered_decision.next_budget == BatchBudget(
        max_items=5, max_host_bytes=50, max_device_bytes=100
    )
    assert recovered_decision.pressure_shrunk_budget_fields == ()
    assert recovered_decision.pressure_high_dimensions == ()
    assert recovered_decision.pressure_triggered_dimensions == ()
    assert recovered_decision.pressure_selected_budget_fields == ()
    assert recovered_decision.pressure_applied_shrink_factors == ()
    assert "triggered shrink factors" not in recovered_decision.reason


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


def test_oom_has_priority_over_pressure_guard() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8),
        policy=GovernorPolicy(
            shrink_factor=0.5,
            max_pressure_ratio_for_growth=0.8,
        ),
    )
    pressure = ResourcePressureSummary(peak_cpu_rss_ratio=1.0)

    decision = governor.observe_results(
        [_result(StepStatus.OOM_FAULT)],
        pressure_summary=pressure,
    )

    assert decision.next_budget == BatchBudget(max_items=4)
    assert decision.pressure_summary == pressure
    assert decision.growth_suppressed_by_pressure is False


@pytest.mark.parametrize(
    ("results", "recovered_oom", "reason_text"),
    [
        ([_result(StepStatus.OOM_FAULT)], False, "OOM fault observed"),
        ([_result()], True, "retry-recovered OOM observed"),
    ],
)
def test_oom_signals_reset_sustained_pressure_streak_before_pressure_shrink(
    results: list[StepResult],
    recovered_oom: bool,
    reason_text: str,
) -> None:
    governor = ConservativeRuntimeGovernor(
        state=RuntimeGovernorState(
            current_budget=BatchBudget(max_items=8),
            consecutive_successes=1,
            consecutive_high_pressure_passes=1,
            consecutive_cpu_pressure_passes=1,
            consecutive_cuda_pressure_passes=1,
        ),
        policy=GovernorPolicy(
            shrink_factor=0.5,
            grow_after_successes=2,
            max_pressure_ratio_for_growth=0.8,
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=2,
        ),
    )

    decision = governor.observe_results(
        results,
        recovered_oom=recovered_oom,
        pressure_summary=ResourcePressureSummary(peak_cpu_rss_ratio=0.95),
    )

    assert decision.next_budget == BatchBudget(max_items=4)
    assert decision.consecutive_successes == 0
    assert decision.consecutive_ooms == 1
    assert decision.consecutive_high_pressure_passes == 0
    assert decision.budget_shrunk_by_pressure is False
    assert decision.growth_suppressed_by_pressure is False
    assert decision.pressure_shrunk_budget_fields == ()
    assert decision.consecutive_cpu_pressure_passes == 0
    assert decision.consecutive_cuda_pressure_passes == 0
    assert reason_text in decision.reason


def test_non_oom_fault_keeps_budget_and_resets_streaks() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=10),
        policy=GovernorPolicy(grow_after_successes=2),
    )
    governor.observe_results([_result()])

    decision = governor.observe_results(
        [_result(StepStatus.RUNTIME_FAULT)],
        pressure_summary=ResourcePressureSummary(peak_cpu_rss_ratio=1.0),
    )

    assert decision.next_budget == BatchBudget(max_items=10)
    assert decision.consecutive_successes == 0
    assert decision.consecutive_ooms == 0
    assert decision.growth_suppressed_by_pressure is False


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


def test_observe_results_consumes_lazy_stream_without_storing_results() -> None:
    events: list[str] = []
    yielded: list[StepResult] = []

    def stream():
        for index, status in enumerate((StepStatus.SUCCESS, StepStatus.DATA_FAULT)):
            events.append(f"yield-{index}")
            result = _result(status)
            yielded.append(result)
            yield result
            events.append(f"after-{index}")

    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=8))

    decision = governor.observe_results(stream())

    assert events == ["yield-0", "after-0", "yield-1", "after-1"]
    assert decision.statuses == (StepStatus.SUCCESS, StepStatus.DATA_FAULT)
    assert all(value not in yielded for value in _field_values(decision))
    assert all(value not in yielded for value in _field_values(governor.state))
    assert governor.state.last_decision == decision


def test_decision_and_state_do_not_store_step_result_store_or_loss_references() -> None:
    loss = torch.tensor(1.0)
    store = object()
    result = _result(loss=loss, store=store)
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=8))

    decision = governor.observe_results([result])

    decision_values = _field_values(decision)
    state_values = _field_values(governor.state)
    assert all(value is not result for value in decision_values)
    assert all(value is not result for value in state_values)
    assert all(value is not loss for value in decision_values)
    assert all(value is not loss for value in state_values)
    assert all(value is not store for value in decision_values)
    assert all(value is not store for value in state_values)


def test_governor_rejects_initial_budget_below_policy_min_bound() -> None:
    with pytest.raises(ValueError):
        ConservativeRuntimeGovernor(
            BatchBudget(max_items=2),
            policy=GovernorPolicy(min_items=4),
        )


def test_governor_rejects_initial_budget_above_policy_max_bound() -> None:
    with pytest.raises(ValueError):
        ConservativeRuntimeGovernor(
            BatchBudget(max_items=20),
            policy=GovernorPolicy(max_items=16),
        )


def test_governor_rejects_state_budget_outside_policy_bounds() -> None:
    state = RuntimeGovernorState(BatchBudget(max_items=2))

    with pytest.raises(ValueError):
        ConservativeRuntimeGovernor(
            state=state,
            policy=GovernorPolicy(min_items=4),
        )


def test_none_budget_fields_skip_policy_bounds_validation() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8),
        policy=GovernorPolicy(min_host_bytes=10, max_device_bytes=20),
    )

    assert governor.current_budget == BatchBudget(max_items=8)


def test_recovered_oom_success_results_shrink_without_success_streak() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8),
        policy=GovernorPolicy(shrink_factor=0.5, grow_after_successes=1),
    )

    pressure = ResourcePressureSummary(peak_cpu_rss_ratio=1.0)
    decision = governor.observe_results(
        [_result()],
        recovered_oom=True,
        pressure_summary=pressure,
    )

    assert decision.next_budget == BatchBudget(max_items=4)
    assert decision.statuses == (StepStatus.SUCCESS,)
    assert decision.consecutive_successes == 0
    assert decision.consecutive_ooms == 1
    assert decision.pressure_summary == pressure
    assert decision.growth_suppressed_by_pressure is False
    assert "retry-recovered OOM observed" in decision.reason


def test_recovered_oom_empty_results_shrink() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8),
        policy=GovernorPolicy(shrink_factor=0.5),
    )

    decision = governor.observe_results([], recovered_oom=True)

    assert decision.next_budget == BatchBudget(max_items=4)
    assert decision.statuses == ()
    assert decision.consecutive_successes == 0
    assert decision.consecutive_ooms == 1
    assert "retry-recovered OOM observed" in decision.reason


def test_recovered_oom_must_be_bool() -> None:
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=8))

    with pytest.raises(TypeError):
        governor.observe_results([], recovered_oom=1)  # type: ignore[arg-type]


def test_actual_oom_with_recovered_oom_signal_shrinks_and_counts_oom() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8),
        policy=GovernorPolicy(shrink_factor=0.5),
    )

    decision = governor.observe_results(
        [_result(StepStatus.OOM_FAULT)],
        recovered_oom=True,
    )

    assert decision.next_budget == BatchBudget(max_items=4)
    assert decision.statuses == (StepStatus.OOM_FAULT,)
    assert decision.consecutive_successes == 0
    assert decision.consecutive_ooms == 1
    assert "OOM fault observed" in decision.reason
    assert "retry-recovered OOM signal also observed" in decision.reason


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


@pytest.mark.parametrize(
    "ratio",
    [0, -0.1, 1.1, float("inf"), float("nan"), True, "0.8"],
)
def test_invalid_growth_pressure_ratio_is_rejected(ratio: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        GovernorPolicy(max_pressure_ratio_for_growth=ratio)  # type: ignore[arg-type]


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


def test_governor_decision_pressure_fields_default_for_existing_construction() -> None:
    budget = BatchBudget(max_items=8)
    decision = GovernorDecision(
        previous_budget=budget,
        next_budget=budget,
        reason="test",
        statuses=(),
        consecutive_successes=0,
        consecutive_ooms=0,
    )
    state = RuntimeGovernorState(current_budget=budget, last_decision=decision)

    assert decision.pressure_summary is None
    assert decision.growth_suppressed_by_pressure is False
    assert decision.pressure_shrunk_budget_fields == ()
    assert decision.consecutive_cpu_pressure_passes == 0
    assert decision.consecutive_cuda_pressure_passes == 0
    assert state.last_decision == decision


def test_invalid_pressure_summary_is_rejected() -> None:
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=8))

    with pytest.raises(TypeError, match="pressure_summary"):
        governor.observe_results(
            [_result()],
            pressure_summary=object(),  # type: ignore[arg-type]
        )


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


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("consecutive_cpu_pressure_passes", -1),
        ("consecutive_cpu_pressure_passes", True),
        ("consecutive_cuda_pressure_passes", -1),
        ("consecutive_cuda_pressure_passes", True),
    ],
)
def test_invalid_dimension_pressure_streaks_are_rejected(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        RuntimeGovernorState(
            current_budget=BatchBudget(max_items=8),
            **{field_name: value},
        )


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
