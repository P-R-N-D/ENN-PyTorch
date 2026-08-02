from __future__ import annotations

from dataclasses import fields

import pytest
import torch
from tensordict import TensorDict

from enn_torch_dev.data import KVBatch
from enn_torch_dev.runtime import (
    AdmissionSplitPolicy,
    BatchBudget,
    ConservativeRuntimeGovernor,
    ConservativeRuntimeOrchestrator,
    ConservativeRuntimeSession,
    GovernorDecision,
    ObservedCostCalibrationPolicy,
    ObservedCostMetricProfile,
    ObservedCostProfile,
    PrePassAdmissionAssessment,
    PrePassAdmissionPolicy,
    PrePassAdmissionStatus,
    ResourceCapacity,
    ResourceSample,
    RuntimeHistorySummary,
    RuntimePassHistory,
    RuntimePassResult,
    RuntimePassSummary,
    StepResult,
    StepStatus,
    format_runtime_history_summary,
    format_runtime_pass_summary,
    summarize_runtime_pass,
)


def _decision(
    *,
    statuses: tuple[StepStatus, ...] = (),
) -> GovernorDecision:
    budget = BatchBudget(max_items=4)
    return GovernorDecision(
        previous_budget=budget,
        next_budget=budget,
        reason="test decision",
        statuses=statuses,
        consecutive_successes=0,
        consecutive_ooms=0,
    )


def _result(batch_size: int = 1) -> StepResult:
    return StepResult(
        status=StepStatus.SUCCESS,
        phase=None,
        batch_size=batch_size,
        row_ids=torch.arange(batch_size),
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


def _pass_result(
    assessments: tuple[PrePassAdmissionAssessment, ...] = (),
    *,
    results: tuple[StepResult, ...] | None = None,
    recovered_oom: bool = False,
) -> RuntimePassResult:
    resolved_results = results if results is not None else (_result(),)
    return RuntimePassResult(
        results=resolved_results,
        decision=_decision(statuses=tuple(item.status for item in resolved_results)),
        recovered_oom=recovered_oom,
        admission_assessments=assessments,
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


def _batch(num_rows: int, *, offset: int = 0) -> KVBatch:
    td = TensorDict(
        {
            "features": torch.arange(
                offset,
                offset + num_rows * 2,
                dtype=torch.float32,
            ).reshape(num_rows, 2),
        },
        batch_size=(num_rows,),
    )
    return KVBatch(
        td=td,
        row_ids=torch.arange(offset, offset + num_rows),
        source_ids=torch.arange(100 + offset, 100 + offset + num_rows),
        sample_ids=torch.arange(200 + offset, 200 + offset + num_rows),
        schema_id="runtime.admission_observability",
        shard_id=19,
    )


class SequenceSampleProvider:
    def __init__(self, samples: tuple[ResourceSample, ...]) -> None:
        self.samples = samples
        self.calls: list[str] = []

    def sample(self, phase: str) -> ResourceSample:
        self.calls.append(phase)
        return self.samples[len(self.calls) - 1]


class SuccessRuntimeStep:
    optimizer = None

    def __init__(self) -> None:
        self.calls: list[KVBatch] = []

    def run(self, batch: KVBatch) -> StepResult:
        self.calls.append(batch)
        return StepResult(
            status=StepStatus.SUCCESS,
            phase=None,
            batch_size=batch.batch_size,
            row_ids=batch.row_ids.detach().cpu().clone(),
        )


def _recovered_orchestrator(
    sample_provider: SequenceSampleProvider,
) -> tuple[ConservativeRuntimeOrchestrator, SuccessRuntimeStep]:
    step = SuccessRuntimeStep()
    orchestrator = ConservativeRuntimeOrchestrator(
        step,
        ConservativeRuntimeGovernor(BatchBudget(max_items=4)),
        resource_capacity=ResourceCapacity(cpu_total_bytes=1_000),
        admission_profile=_profile(cpu=100),
        admission_sample_provider=sample_provider,
        admission_split_policy=AdmissionSplitPolicy(),
    )
    return orchestrator, step


def test_runtime_pass_summary_appends_admission_fields_for_compatibility() -> None:
    field_names = [field.name for field in fields(RuntimePassSummary)]

    assert field_names[-6:] == [
        "admission_assessment_count",
        "admission_admit_assessment_count",
        "admission_recovered_reject_count",
        "admission_allowed_unknown_count",
        "admission_recovery_occurred",
        "minimum_recovered_admissible_items",
    ]


def test_runtime_history_summary_appends_admission_fields_for_compatibility() -> None:
    field_names = [field.name for field in fields(RuntimeHistorySummary)]

    assert field_names[-7:] == [
        "admission_assessed_passes",
        "admission_recovery_passes",
        "admission_total_assessments",
        "admission_admit_assessments",
        "admission_recovered_rejects",
        "admission_allowed_unknowns",
        "minimum_recovered_admissible_items",
    ]


def test_summary_defaults_when_admission_is_disabled_or_pass_is_empty() -> None:
    disabled = summarize_runtime_pass(_pass_result())
    empty = summarize_runtime_pass(_pass_result(results=()))

    for summary in (disabled, empty):
        assert summary.admission_assessment_count == 0
        assert summary.admission_admit_assessment_count == 0
        assert summary.admission_recovered_reject_count == 0
        assert summary.admission_allowed_unknown_count == 0
        assert summary.admission_recovery_occurred is False
        assert summary.minimum_recovered_admissible_items is None


def test_summary_reduces_admission_assessments_to_scalar_provenance() -> None:
    assessments = (
        _assessment(PrePassAdmissionStatus.REJECT, batch_size=8, max_admissible_items=4),
        _assessment(PrePassAdmissionStatus.ADMIT, batch_size=4),
        _assessment(PrePassAdmissionStatus.REJECT, batch_size=4, max_admissible_items=2),
        _assessment(PrePassAdmissionStatus.ADMIT, batch_size=2),
        _assessment(PrePassAdmissionStatus.UNKNOWN, batch_size=2),
    )

    summary = summarize_runtime_pass(
        _pass_result(assessments, recovered_oom=True)
    )

    assert summary.admission_assessment_count == 5
    assert summary.admission_admit_assessment_count == 2
    assert summary.admission_recovered_reject_count == 2
    assert summary.admission_allowed_unknown_count == 1
    assert summary.admission_recovery_occurred is True
    assert summary.minimum_recovered_admissible_items == 2
    assert summary.recovered_oom is True
    assert all(value is not assessment for value in _field_values(summary) for assessment in assessments)


@pytest.mark.parametrize("target", [None, 0, -1, 4, 5, True])
def test_summary_rejects_inconsistent_completed_reject(target: object) -> None:
    assessment = _assessment(
        PrePassAdmissionStatus.REJECT,
        batch_size=4,
        max_admissible_items=target,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="positive reducing"):
        summarize_runtime_pass(_pass_result((assessment,)))


def test_summary_rejects_non_assessment_entries() -> None:
    pass_result = RuntimePassResult(
        results=(_result(),),
        decision=_decision(statuses=(StepStatus.SUCCESS,)),
        admission_assessments=(object(),),  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="PrePassAdmissionAssessment"):
        summarize_runtime_pass(pass_result)


def test_pass_formatter_includes_admission_provenance() -> None:
    summary = summarize_runtime_pass(
        _pass_result(
            (
                _assessment(
                    PrePassAdmissionStatus.REJECT,
                    batch_size=4,
                    max_admissible_items=2,
                ),
                _assessment(PrePassAdmissionStatus.ADMIT, batch_size=2),
                _assessment(PrePassAdmissionStatus.UNKNOWN, batch_size=2),
            )
        )
    )

    text = format_runtime_pass_summary(summary)

    for expected in (
        "admission_assessment_count=3",
        "admission_admit_assessment_count=1",
        "admission_recovered_reject_count=1",
        "admission_allowed_unknown_count=1",
        "admission_recovery_occurred=True",
        "minimum_recovered_admissible_items=2",
    ):
        assert expected in text


def test_history_aggregates_admission_provenance_with_pass_counts() -> None:
    history = RuntimePassHistory(max_records=5)
    history.append_pass_result(
        _pass_result((_assessment(PrePassAdmissionStatus.ADMIT, batch_size=2),))
    )
    aggregate = history.append_pass_result(
        _pass_result(
            (
                _assessment(
                    PrePassAdmissionStatus.REJECT,
                    batch_size=8,
                    max_admissible_items=4,
                ),
                _assessment(
                    PrePassAdmissionStatus.REJECT,
                    batch_size=4,
                    max_admissible_items=2,
                ),
                _assessment(PrePassAdmissionStatus.ADMIT, batch_size=2),
                _assessment(PrePassAdmissionStatus.UNKNOWN, batch_size=2),
            )
        )
    )

    assert aggregate.admission_assessed_passes == 2
    assert aggregate.admission_recovery_passes == 1
    assert aggregate.admission_total_assessments == 5
    assert aggregate.admission_admit_assessments == 2
    assert aggregate.admission_recovered_rejects == 2
    assert aggregate.admission_allowed_unknowns == 1
    assert aggregate.minimum_recovered_admissible_items == 2


def test_history_admission_aggregation_respects_retained_window() -> None:
    history = RuntimePassHistory(max_records=2)
    history.append_pass_result(
        _pass_result(
            (
                _assessment(
                    PrePassAdmissionStatus.REJECT,
                    batch_size=8,
                    max_admissible_items=1,
                ),
                _assessment(PrePassAdmissionStatus.ADMIT, batch_size=1),
            )
        )
    )
    history.append_pass_result(
        _pass_result((_assessment(PrePassAdmissionStatus.UNKNOWN, batch_size=2),))
    )
    aggregate = history.append_pass_result(
        _pass_result(
            (
                _assessment(
                    PrePassAdmissionStatus.REJECT,
                    batch_size=4,
                    max_admissible_items=2,
                ),
                _assessment(PrePassAdmissionStatus.ADMIT, batch_size=2),
            )
        )
    )

    assert len(history.records) == 2
    assert aggregate.admission_assessed_passes == 2
    assert aggregate.admission_recovery_passes == 1
    assert aggregate.admission_total_assessments == 3
    assert aggregate.admission_admit_assessments == 1
    assert aggregate.admission_recovered_rejects == 1
    assert aggregate.admission_allowed_unknowns == 1
    assert aggregate.minimum_recovered_admissible_items == 2


def test_history_formatter_includes_latest_admission_state() -> None:
    history = RuntimePassHistory(max_records=2)
    aggregate = history.append_pass_result(
        _pass_result(
            (
                _assessment(
                    PrePassAdmissionStatus.REJECT,
                    batch_size=4,
                    max_admissible_items=2,
                ),
                _assessment(PrePassAdmissionStatus.ADMIT, batch_size=2),
            )
        )
    )

    text = format_runtime_history_summary(aggregate)

    for expected in (
        "admission_assessed_passes=1",
        "admission_recovery_passes=1",
        "admission_total_assessments=2",
        "admission_admit_assessments=1",
        "admission_recovered_rejects=1",
        "admission_allowed_unknowns=0",
        "minimum_recovered_admissible_items=2",
        "latest_admission_recovery_occurred=True",
        "latest_admission_recovered_reject_count=1",
    ):
        assert expected in text


def test_orchestrator_recovery_flows_into_summary_and_history() -> None:
    provider = SequenceSampleProvider((_sample(700), _sample(100), _sample(100)))
    orchestrator, step = _recovered_orchestrator(provider)
    pass_result = orchestrator.run_pass((_batch(4),))

    summary = summarize_runtime_pass(pass_result)
    aggregate = RuntimePassHistory(max_records=2).append_pass_result(pass_result)

    assert [batch.batch_size for batch in step.calls] == [2, 2]
    assert summary.admission_assessment_count == 3
    assert summary.admission_recovered_reject_count == 1
    assert summary.admission_admit_assessment_count == 2
    assert summary.minimum_recovered_admissible_items == 2
    assert aggregate.admission_recovery_passes == 1
    assert aggregate.admission_recovered_rejects == 1
    assert provider.calls == ["before_admission"] * 3


def test_session_propagates_admission_provenance_without_raw_assessments() -> None:
    provider = SequenceSampleProvider((_sample(700), _sample(100), _sample(100)))
    orchestrator, _ = _recovered_orchestrator(provider)
    history = RuntimePassHistory(max_records=2)
    session = ConservativeRuntimeSession(orchestrator, history, max_passes=1)

    record = next(session.run_passes(((_batch(4),),)))

    assert record.pass_summary.admission_recovery_occurred is True
    assert record.pass_summary.admission_recovered_reject_count == 1
    assert record.history_summary.admission_recovery_passes == 1
    assert record.history_summary.admission_recovered_rejects == 1
    assert all(
        not isinstance(value, PrePassAdmissionAssessment)
        for value in _field_values(record.pass_summary)
    )


def test_terminal_block_is_not_added_to_session_history() -> None:
    provider = SequenceSampleProvider((_sample(700), _sample(850)))
    orchestrator, _ = _recovered_orchestrator(provider)
    history = RuntimePassHistory(max_records=2)
    session = ConservativeRuntimeSession(orchestrator, history, max_passes=1)

    with pytest.raises(RuntimeError):
        next(session.run_passes(((_batch(4),),)))

    assert history.records == ()
    aggregate = history.summarize()
    assert aggregate.admission_assessed_passes == 0
    assert aggregate.admission_recovery_passes == 0


def _field_values(instance: object) -> tuple[object, ...]:
    return tuple(getattr(instance, field.name) for field in fields(instance))
