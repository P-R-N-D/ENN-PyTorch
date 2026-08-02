from __future__ import annotations

from dataclasses import fields

import pytest
import torch
from tensordict import TensorDict

import enn_torch
from enn_torch_dev.data import KVBatch
from enn_torch_dev.runtime import (
    AdmissionSplitPolicy,
    AdmissionUnknownAction,
    BatchBudget,
    ConservativeRuntimeGovernor,
    ConservativeRuntimeOrchestrator,
    GovernorDecision,
    GovernorPolicy,
    ObservedCostCalibrationPolicy,
    ObservedCostMetricProfile,
    ObservedCostProfile,
    PrePassAdmissionAssessment,
    PrePassAdmissionBlocked,
    PrePassAdmissionPolicy,
    PrePassAdmissionStatus,
    ResourceCapacity,
    ResourcePressureSummary,
    ResourceSample,
    RuntimePassHistory,
    RuntimePassResult,
    StepResult,
    StepStatus,
    format_runtime_history_summary,
    format_runtime_pass_summary,
    summarize_runtime_pass,
)


def _result(
    status: StepStatus = StepStatus.SUCCESS,
    *,
    batch_size: int = 1,
) -> StepResult:
    return StepResult(
        status=status,
        phase=None,
        batch_size=batch_size,
        row_ids=torch.arange(batch_size),
        error_type=None if status is StepStatus.SUCCESS else status.name,
        error_message=None if status is StepStatus.SUCCESS else status.value,
    )


def _metric(value: int | None) -> ObservedCostMetricProfile:
    return ObservedCostMetricProfile(
        max_bytes_per_item=value,
        known_samples=0 if value is None else 1,
        unknown_samples=1 if value is None else 0,
        zero_samples=1 if value == 0 else 0,
        negative_deltas_clamped=0,
    )


def _profile(*, cpu: int | None = 100) -> ObservedCostProfile:
    return ObservedCostProfile(
        policy=ObservedCostCalibrationPolicy(),
        total_observations=3,
        successful_samples=3,
        ignored_samples=0,
        rejected_samples=0,
        ignored_zero_batch_samples=0,
        ignored_by_status=(),
        min_batch_size=1,
        max_batch_size=8,
        cuda_device_index=None,
        cpu_rss=_metric(cpu),
        cuda_allocated=_metric(None),
        cuda_reserved=_metric(None),
        cuda_max_allocated=_metric(None),
        cuda_max_reserved=_metric(None),
        phase_costs=(),
    )


def _sample(cpu_rss_bytes: int | None) -> ResourceSample:
    return ResourceSample(
        timestamp_ns=1,
        phase="before_admission",
        cpu_rss_bytes=cpu_rss_bytes,
    )


def _batch(num_rows: int) -> KVBatch:
    return KVBatch(
        td=TensorDict(
            {"features": torch.arange(num_rows * 2).reshape(num_rows, 2)},
            batch_size=(num_rows,),
        ),
        row_ids=torch.arange(num_rows),
        source_ids=torch.arange(100, 100 + num_rows),
        sample_ids=torch.arange(200, 200 + num_rows),
        schema_id="runtime.admission_growth_guard",
        shard_id=23,
    )


class SequenceSampleProvider:
    def __init__(self, samples: tuple[ResourceSample, ...]) -> None:
        self.samples = samples
        self.calls: list[str] = []

    def sample(self, phase: str) -> ResourceSample:
        self.calls.append(phase)
        return self.samples[len(self.calls) - 1]


class SuccessStep:
    optimizer = None

    def __init__(self) -> None:
        self.calls: list[int] = []

    def run(self, batch: KVBatch) -> StepResult:
        self.calls.append(batch.batch_size)
        return StepResult(
            status=StepStatus.SUCCESS,
            phase=None,
            batch_size=batch.batch_size,
            row_ids=batch.row_ids.detach().cpu().clone(),
        )


def _assessment(
    status: PrePassAdmissionStatus,
    *,
    batch_size: int,
    max_admissible_items: int | None = None,
) -> PrePassAdmissionAssessment:
    return PrePassAdmissionAssessment(
        status=status,
        batch_size=batch_size,
        policy=PrePassAdmissionPolicy(),
        profile_successful_samples=3,
        cuda_device_index=None,
        dimensions=(),
        rejected_dimensions=("cpu_rss",)
        if status is PrePassAdmissionStatus.REJECT
        else (),
        unknown_dimensions=("cpu_rss",)
        if status is PrePassAdmissionStatus.UNKNOWN
        else (),
        max_admissible_items=max_admissible_items,
        warnings=(),
    )


def _decision(
    *,
    budget: BatchBudget | None = None,
    admission_recovery_max_items: int | None = None,
    suppressed: bool = False,
) -> GovernorDecision:
    resolved = budget or BatchBudget(max_items=4)
    return GovernorDecision(
        previous_budget=resolved,
        next_budget=resolved,
        reason="test decision",
        statuses=(StepStatus.SUCCESS,),
        consecutive_successes=0,
        consecutive_ooms=0,
        admission_recovery_max_items=admission_recovery_max_items,
        growth_suppressed_by_admission_recovery=suppressed,
    )


def test_policy_and_decision_fields_are_append_only() -> None:
    assert [item.name for item in fields(GovernorPolicy)][-1] == (
        "suppress_growth_after_admission_recovery"
    )
    assert [item.name for item in fields(GovernorDecision)][-2:] == [
        "admission_recovery_max_items",
        "growth_suppressed_by_admission_recovery",
    ]


def test_policy_requires_bool_growth_guard() -> None:
    with pytest.raises(TypeError, match="must be a bool"):
        GovernorPolicy(suppress_growth_after_admission_recovery=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_governor_rejects_invalid_admission_recovery_limit(value: object) -> None:
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=8))

    with pytest.raises((TypeError, ValueError), match="admission_recovery_max_items"):
        governor.observe_results(
            [_result()],
            admission_recovery_max_items=value,  # type: ignore[arg-type]
        )


def test_growth_guard_is_opt_in_and_provenance_is_still_recorded() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8),
        policy=GovernorPolicy(grow_after_successes=1),
    )

    decision = governor.observe_results(
        [_result()],
        admission_recovery_max_items=2,
    )

    assert decision.next_budget == BatchBudget(max_items=16)
    assert decision.admission_recovery_max_items == 2
    assert decision.growth_suppressed_by_admission_recovery is False


def test_growth_guard_resets_success_streak_and_prevents_threshold_growth() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8),
        policy=GovernorPolicy(
            grow_after_successes=2,
            suppress_growth_after_admission_recovery=True,
        ),
    )

    first = governor.observe_results([_result()])
    guarded = governor.observe_results(
        [_result()], admission_recovery_max_items=2
    )
    clean_one = governor.observe_results([_result()])
    clean_two = governor.observe_results([_result()])

    assert first.consecutive_successes == 1
    assert guarded.next_budget == BatchBudget(max_items=8)
    assert guarded.consecutive_successes == 0
    assert guarded.growth_suppressed_by_admission_recovery is True
    assert "recovered max-items limit=2" in guarded.reason
    assert clean_one.consecutive_successes == 1
    assert clean_two.next_budget == BatchBudget(max_items=16)


def test_growth_guard_replaces_growth_reason_at_policy_maximum() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8),
        policy=GovernorPolicy(
            max_items=8,
            grow_after_successes=1,
            suppress_growth_after_admission_recovery=True,
        ),
    )

    decision = governor.observe_results(
        [_result()], admission_recovery_max_items=2
    )

    assert decision.next_budget == BatchBudget(max_items=8)
    assert decision.consecutive_successes == 0
    assert decision.growth_suppressed_by_admission_recovery is True
    assert "growing configured budget fields" not in decision.reason
    assert "suppressing success-streak growth" in decision.reason


def test_yielded_oom_keeps_priority_over_admission_guard() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8),
        policy=GovernorPolicy(suppress_growth_after_admission_recovery=True),
    )

    decision = governor.observe_results(
        [_result(StepStatus.OOM_FAULT)],
        admission_recovery_max_items=2,
    )

    assert decision.next_budget == BatchBudget(max_items=4)
    assert decision.admission_recovery_max_items == 2
    assert decision.growth_suppressed_by_admission_recovery is False


def test_retry_recovered_oom_keeps_priority_over_admission_guard() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8),
        policy=GovernorPolicy(suppress_growth_after_admission_recovery=True),
    )

    decision = governor.observe_results(
        [_result()],
        recovered_oom=True,
        admission_recovery_max_items=2,
    )

    assert decision.next_budget == BatchBudget(max_items=4)
    assert decision.growth_suppressed_by_admission_recovery is False


def test_pressure_and_admission_growth_guards_can_both_be_visible() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8),
        policy=GovernorPolicy(
            grow_after_successes=1,
            max_pressure_ratio_for_growth=0.8,
            suppress_growth_after_admission_recovery=True,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=ResourcePressureSummary(peak_cpu_rss_ratio=0.9),
        admission_recovery_max_items=2,
    )

    assert decision.next_budget == BatchBudget(max_items=8)
    assert decision.growth_suppressed_by_pressure is True
    assert decision.growth_suppressed_by_admission_recovery is True


def test_pressure_shrink_is_not_cancelled_by_admission_guard() -> None:
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8, max_host_bytes=100),
        policy=GovernorPolicy(
            min_pressure_ratio_for_shrink=0.9,
            shrink_after_pressure_passes=1,
            suppress_growth_after_admission_recovery=True,
        ),
    )

    decision = governor.observe_results(
        [_result()],
        pressure_summary=ResourcePressureSummary(peak_cpu_rss_ratio=0.95),
        admission_recovery_max_items=2,
    )

    assert decision.next_budget == BatchBudget(max_items=8, max_host_bytes=50)
    assert decision.budget_shrunk_by_pressure is True
    assert decision.growth_suppressed_by_admission_recovery is True


def test_fault_and_empty_paths_do_not_report_admission_growth_suppression() -> None:
    policy = GovernorPolicy(suppress_growth_after_admission_recovery=True)
    fault_governor = ConservativeRuntimeGovernor(BatchBudget(max_items=8), policy=policy)
    empty_governor = ConservativeRuntimeGovernor(BatchBudget(max_items=8), policy=policy)

    fault = fault_governor.observe_results(
        [_result(StepStatus.DATA_FAULT)],
        admission_recovery_max_items=2,
    )
    empty = empty_governor.observe_results(
        [], admission_recovery_max_items=2
    )

    assert fault.growth_suppressed_by_admission_recovery is False
    assert empty.growth_suppressed_by_admission_recovery is False


def test_nested_orchestrator_recovery_passes_minimum_limit_to_governor() -> None:
    provider = SequenceSampleProvider(
        (
            _sample(500),
            _sample(700),
            _sample(100),
            _sample(100),
            _sample(100),
        )
    )
    step = SuccessStep()
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=8),
        policy=GovernorPolicy(
            grow_after_successes=1,
            suppress_growth_after_admission_recovery=True,
        ),
    )
    orchestrator = ConservativeRuntimeOrchestrator(
        step,
        governor,
        resource_capacity=ResourceCapacity(cpu_total_bytes=1_000),
        admission_profile=_profile(cpu=100),
        admission_sample_provider=provider,
        admission_split_policy=AdmissionSplitPolicy(),
    )

    result = orchestrator.run_pass((_batch(8),))
    summary = summarize_runtime_pass(result)

    assert step.calls == [2, 2, 4]
    assert result.decision.admission_recovery_max_items == 2
    assert result.decision.growth_suppressed_by_admission_recovery is True
    assert result.decision.next_budget == BatchBudget(max_items=8)
    assert summary.minimum_recovered_admissible_items == 2
    assert summary.governor_admission_recovery_max_items == 2
    assert summary.growth_suppressed_by_admission_recovery is True


def test_allowed_unknown_without_reject_does_not_activate_growth_guard() -> None:
    provider = SequenceSampleProvider((_sample(100),))
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=4),
        policy=GovernorPolicy(
            grow_after_successes=1,
            suppress_growth_after_admission_recovery=True,
        ),
    )
    orchestrator = ConservativeRuntimeOrchestrator(
        SuccessStep(),
        governor,
        resource_capacity=ResourceCapacity(cpu_total_bytes=1_000),
        admission_profile=_profile(cpu=None),
        admission_sample_provider=provider,
        admission_unknown_action=AdmissionUnknownAction.ALLOW,
        admission_split_policy=AdmissionSplitPolicy(),
    )

    result = orchestrator.run_pass((_batch(4),))

    assert result.decision.admission_recovery_max_items is None
    assert result.decision.growth_suppressed_by_admission_recovery is False
    assert result.decision.next_budget == BatchBudget(max_items=8)


def test_terminal_child_block_does_not_update_governor() -> None:
    provider = SequenceSampleProvider((_sample(700), _sample(850)))
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=4),
        policy=GovernorPolicy(suppress_growth_after_admission_recovery=True),
    )
    orchestrator = ConservativeRuntimeOrchestrator(
        SuccessStep(),
        governor,
        resource_capacity=ResourceCapacity(cpu_total_bytes=1_000),
        admission_profile=_profile(cpu=100),
        admission_sample_provider=provider,
        admission_split_policy=AdmissionSplitPolicy(max_split_depth=1),
    )

    with pytest.raises(PrePassAdmissionBlocked):
        orchestrator.run_pass((_batch(4),))

    assert governor.state.last_decision is None
    assert governor.current_budget == BatchBudget(max_items=4)


def test_history_counts_only_retained_admission_growth_suppression() -> None:
    history = RuntimePassHistory(max_records=1)
    assessment = _assessment(
        PrePassAdmissionStatus.REJECT,
        batch_size=4,
        max_admissible_items=2,
    )
    admitted = _assessment(PrePassAdmissionStatus.ADMIT, batch_size=2)
    first_result = RuntimePassResult(
        results=(_result(batch_size=2),),
        decision=_decision(admission_recovery_max_items=2, suppressed=True),
        admission_assessments=(assessment, admitted),
    )
    second_result = RuntimePassResult(
        results=(_result(),),
        decision=_decision(),
    )

    first = history.append_pass_result(first_result)
    second = history.append_pass_result(second_result)

    assert first.admission_growth_suppressed_passes == 1
    assert second.admission_growth_suppressed_passes == 0


def test_admission_growth_guard_formatter_provenance() -> None:
    assessment = _assessment(
        PrePassAdmissionStatus.REJECT,
        batch_size=4,
        max_admissible_items=2,
    )
    pass_result = RuntimePassResult(
        results=(_result(batch_size=2),),
        decision=_decision(admission_recovery_max_items=2, suppressed=True),
        admission_assessments=(
            assessment,
            _assessment(PrePassAdmissionStatus.ADMIT, batch_size=2),
        ),
    )
    pass_summary = summarize_runtime_pass(pass_result)
    pass_text = format_runtime_pass_summary(pass_summary)

    history = RuntimePassHistory(max_records=1)
    history_text = format_runtime_history_summary(
        history.append_summary(pass_summary)
    )

    assert "growth_suppressed_by_admission_recovery=True" in pass_text
    assert "governor_admission_recovery_max_items=2" in pass_text
    assert "admission_growth_suppressed_passes=1" in history_text
    assert "latest_growth_suppressed_by_admission_recovery=True" in history_text
    assert "latest_governor_admission_recovery_max_items=2" in history_text


def test_stable_namespace_is_unchanged() -> None:
    assert "GovernorPolicy" not in set(enn_torch.__all__)
    with pytest.raises(AttributeError):
        getattr(enn_torch, "GovernorPolicy")
