from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from enn_torch_dev.runtime import (
    BatchBudget,
    GovernorDecision,
    ResourceCapacity,
    ResourcePressureSummary,
    ResourceSample,
    RuntimePassResult,
    RuntimePassSummary,
    StepResult,
    StepStatus,
    assess_resource_pressure,
    format_runtime_pass_summary,
    summarize_runtime_pass,
)


def _result(
    status: StepStatus = StepStatus.SUCCESS,
    *,
    batch_size: int = 1,
    row_count: int | None = None,
    loss: torch.Tensor | None = None,
    store: object | None = None,
) -> StepResult:
    rows = batch_size if row_count is None else row_count
    return StepResult(
        status=status,
        phase=None,
        batch_size=batch_size,
        row_ids=torch.arange(rows),
        loss=loss,
        store=store,  # type: ignore[arg-type]
    )


def _decision(
    *,
    previous_budget: BatchBudget | None = None,
    next_budget: BatchBudget | None = None,
    reason: str = "test decision",
    statuses: tuple[StepStatus, ...] = (),
    consecutive_successes: int = 0,
    consecutive_ooms: int = 0,
    pressure_summary: ResourcePressureSummary | None = None,
    growth_suppressed_by_pressure: bool = False,
    consecutive_high_pressure_passes: int = 0,
    budget_shrunk_by_pressure: bool = False,
) -> GovernorDecision:
    previous = previous_budget or BatchBudget(max_items=4)
    return GovernorDecision(
        previous_budget=previous,
        next_budget=next_budget or previous,
        reason=reason,
        statuses=statuses,
        consecutive_successes=consecutive_successes,
        consecutive_ooms=consecutive_ooms,
        peak_cpu_rss_bytes=11,
        peak_cuda_allocated_bytes=22,
        peak_cuda_reserved_bytes=33,
        peak_cuda_max_allocated_bytes=44,
        peak_cuda_max_reserved_bytes=55,
        pressure_summary=pressure_summary,
        growth_suppressed_by_pressure=growth_suppressed_by_pressure,
        consecutive_high_pressure_passes=consecutive_high_pressure_passes,
        budget_shrunk_by_pressure=budget_shrunk_by_pressure,
    )


def _pass_result(
    results: tuple[StepResult, ...],
    *,
    decision: GovernorDecision | None = None,
    recovered_oom: bool = False,
) -> RuntimePassResult:
    return RuntimePassResult(
        results=results,
        decision=decision
        or _decision(
            statuses=tuple(
                result.status for result in results if isinstance(result, StepResult)
            )
        ),
        recovered_oom=recovered_oom,
    )


def _field_values(instance: object) -> list[object]:
    return [getattr(instance, field.name) for field in fields(instance)]


def test_summarize_runtime_pass_handles_empty_result() -> None:
    decision = _decision(reason="no results", statuses=())
    summary = summarize_runtime_pass(_pass_result((), decision=decision))

    assert isinstance(summary, RuntimePassSummary)
    assert summary.total_results == 0
    assert summary.statuses == ()
    assert dict(summary.status_counts) == {}
    assert summary.total_batch_size == 0
    assert summary.total_rows == 0
    assert summary.recovered_oom is False
    assert summary.saw_oom is False
    assert summary.previous_budget == decision.previous_budget
    assert summary.next_budget == decision.next_budget
    assert summary.budget_changed is False
    assert summary.decision_reason == "no results"


def test_summarize_and_format_runtime_pass_pressure_shrink_feedback() -> None:
    summary = summarize_runtime_pass(
        _pass_result(
            (_result(),),
            decision=_decision(
                pressure_summary=ResourcePressureSummary(peak_cpu_rss_ratio=0.95),
                growth_suppressed_by_pressure=True,
                consecutive_high_pressure_passes=1,
                budget_shrunk_by_pressure=True,
            ),
        )
    )

    text = format_runtime_pass_summary(summary)

    assert summary.consecutive_high_pressure_passes == 1
    assert summary.budget_shrunk_by_pressure is True
    assert "consecutive_high_pressure_passes=1" in text
    assert "budget_shrunk_by_pressure=True" in text


def test_summarize_runtime_pass_counts_successes_and_rows() -> None:
    results = (_result(batch_size=2), _result(batch_size=3, row_count=5))
    summary = summarize_runtime_pass(_pass_result(results))

    assert summary.total_results == 2
    assert summary.statuses == (StepStatus.SUCCESS, StepStatus.SUCCESS)
    assert dict(summary.status_counts) == {StepStatus.SUCCESS: 2}
    assert summary.total_batch_size == 5
    assert summary.total_rows == 5
    assert summary.saw_oom is False


def test_summarize_runtime_pass_counts_rows_by_batch_size_for_multidimensional_row_ids() -> None:
    result = StepResult(
        status=StepStatus.SUCCESS,
        phase=None,
        batch_size=2,
        row_ids=torch.arange(4).reshape(2, 2),
    )

    summary = summarize_runtime_pass(_pass_result((result,)))

    assert summary.total_batch_size == 2
    assert summary.total_rows == 2


def test_summarize_runtime_pass_counts_mixed_statuses_and_oom() -> None:
    results = (
        _result(StepStatus.SUCCESS, batch_size=2),
        _result(StepStatus.OOM_FAULT, batch_size=4),
        _result(StepStatus.DATA_FAULT, batch_size=1),
    )
    summary = summarize_runtime_pass(_pass_result(results))

    assert dict(summary.status_counts) == {
        StepStatus.SUCCESS: 1,
        StepStatus.OOM_FAULT: 1,
        StepStatus.DATA_FAULT: 1,
    }
    assert summary.saw_oom is True
    assert summary.total_batch_size == 7


def test_summarize_runtime_pass_preserves_recovered_oom_signal() -> None:
    summary = summarize_runtime_pass(_pass_result((_result(), _result()), recovered_oom=True))

    assert summary.recovered_oom is True
    assert summary.saw_oom is False
    assert dict(summary.status_counts) == {StepStatus.SUCCESS: 2}


def test_summarize_runtime_pass_detects_budget_change_and_decision_metadata() -> None:
    decision = _decision(
        previous_budget=BatchBudget(max_items=4),
        next_budget=BatchBudget(max_items=2),
        reason="OOM fault observed; shrinking configured budget fields",
        statuses=(StepStatus.OOM_FAULT,),
        consecutive_ooms=1,
    )

    summary = summarize_runtime_pass(_pass_result((_result(StepStatus.OOM_FAULT),), decision=decision))

    assert summary.previous_budget == BatchBudget(max_items=4)
    assert summary.next_budget == BatchBudget(max_items=2)
    assert summary.budget_changed is True
    assert summary.decision_reason == decision.reason
    assert summary.consecutive_successes == 0
    assert summary.consecutive_ooms == 1
    assert summary.peak_cpu_rss_bytes == 11
    assert summary.peak_cuda_allocated_bytes == 22
    assert summary.peak_cuda_reserved_bytes == 33
    assert summary.peak_cuda_max_allocated_bytes == 44
    assert summary.peak_cuda_max_reserved_bytes == 55


def test_runtime_pass_summary_appends_feedback_and_capacity_fields_for_compatibility() -> None:
    field_names = [field.name for field in fields(RuntimePassSummary)]

    assert field_names[-5:] == [
        "pressure_summary",
        "growth_suppressed_by_pressure",
        "resource_capacity",
        "consecutive_high_pressure_passes",
        "budget_shrunk_by_pressure",
    ]


def test_summarize_runtime_pass_copies_pressure_feedback() -> None:
    pressure = ResourcePressureSummary(
        peak_cpu_rss_ratio=0.5,
        peak_cuda_reserved_ratio=0.9,
    )
    decision = _decision(
        pressure_summary=pressure,
        growth_suppressed_by_pressure=True,
    )

    summary = summarize_runtime_pass(_pass_result((_result(),), decision=decision))

    assert summary.pressure_summary == pressure
    assert summary.pressure_summary.max_observed_ratio == pytest.approx(0.9)
    assert summary.growth_suppressed_by_pressure is True


def test_summarize_runtime_pass_preserves_assessed_unknown_pressure() -> None:
    decision = _decision(pressure_summary=ResourcePressureSummary())

    summary = summarize_runtime_pass(_pass_result((_result(),), decision=decision))

    assert summary.pressure_summary == ResourcePressureSummary()
    assert summary.pressure_summary.max_observed_ratio is None
    assert summary.growth_suppressed_by_pressure is False


def test_summarize_runtime_pass_does_not_store_raw_runtime_references() -> None:
    loss = torch.tensor(1.0)
    store = object()
    result = _result(loss=loss, store=store)
    sample = ResourceSample(
        timestamp_ns=1,
        phase="summary-test",
        cpu_rss_bytes=50,
    )
    pressure = assess_resource_pressure(
        (sample,),
        ResourceCapacity(cpu_total_bytes=100),
    )
    decision = _decision(pressure_summary=pressure)

    summary = summarize_runtime_pass(_pass_result((result,), decision=decision))
    values = _field_values(summary)

    assert all(value is not result for value in values)
    assert all(value is not loss for value in values)
    assert all(value is not store for value in values)
    assert all(value is not sample for value in values)


def test_format_runtime_pass_summary_is_stable_text() -> None:
    decision = _decision(
        previous_budget=BatchBudget(max_items=4),
        next_budget=BatchBudget(max_items=2),
        reason="retry-recovered OOM observed; shrinking configured budget fields",
        consecutive_ooms=1,
    )
    summary = summarize_runtime_pass(
        _pass_result(
            (_result(StepStatus.SUCCESS, batch_size=2),),
            decision=decision,
            recovered_oom=True,
        )
    )

    text = format_runtime_pass_summary(summary)

    assert text.splitlines()[0] == "Runtime pass summary"
    assert "total_results=1" in text
    assert "total_batch_size=2" in text
    assert "statuses=success=1" in text
    assert "recovered_oom=True" in text
    assert "saw_oom=False" in text
    assert "budget_changed=True" in text
    assert "consecutive_ooms=1" in text
    assert "decision_reason=retry-recovered OOM observed" in text


def test_format_runtime_pass_summary_reports_pressure_feedback() -> None:
    pressure = ResourcePressureSummary(peak_cpu_rss_ratio=0.9)
    decision = _decision(
        pressure_summary=pressure,
        growth_suppressed_by_pressure=True,
    )

    text = format_runtime_pass_summary(
        summarize_runtime_pass(_pass_result((_result(),), decision=decision))
    )

    assert "pressure_assessed=True" in text
    assert "max_pressure_ratio=0.9" in text
    assert "growth_suppressed_by_pressure=True" in text


def test_format_runtime_pass_summary_distinguishes_unknown_and_unassessed_pressure() -> None:
    assessed_text = format_runtime_pass_summary(
        summarize_runtime_pass(
            _pass_result(
                (_result(),),
                decision=_decision(pressure_summary=ResourcePressureSummary()),
            )
        )
    )
    unassessed_text = format_runtime_pass_summary(
        summarize_runtime_pass(_pass_result((_result(),)))
    )

    assert "pressure_assessed=True" in assessed_text
    assert "max_pressure_ratio=unknown" in assessed_text
    assert "pressure_assessed=False" in unassessed_text
    assert "max_pressure_ratio=unknown" in unassessed_text


def test_format_runtime_pass_summary_handles_no_statuses() -> None:
    text = format_runtime_pass_summary(summarize_runtime_pass(_pass_result(())))

    assert "statuses=none" in text


def test_summarize_runtime_pass_rejects_invalid_arguments() -> None:
    with pytest.raises(TypeError, match="RuntimePassResult"):
        summarize_runtime_pass(object())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="StepResult"):
        summarize_runtime_pass(_pass_result((object(),)))  # type: ignore[arg-type]

    bad = RuntimePassResult(results=(), decision=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="GovernorDecision"):
        summarize_runtime_pass(bad)


def test_format_runtime_pass_summary_rejects_invalid_argument() -> None:
    with pytest.raises(TypeError, match="RuntimePassSummary"):
        format_runtime_pass_summary(object())  # type: ignore[arg-type]
