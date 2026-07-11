from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from enn_torch_dev.runtime import (
    BatchBudget,
    GovernorDecision,
    RuntimeHistorySummary,
    RuntimePassHistory,
    RuntimePassResult,
    RuntimePassSummary,
    StepResult,
    StepStatus,
    format_runtime_history_summary,
    summarize_runtime_pass,
)


def _decision(
    *,
    previous_budget: BatchBudget | None = None,
    next_budget: BatchBudget | None = None,
    reason: str = "test decision",
    statuses: tuple[StepStatus, ...] = (),
    consecutive_successes: int = 0,
    consecutive_ooms: int = 0,
) -> GovernorDecision:
    previous = previous_budget or BatchBudget(max_items=4)
    return GovernorDecision(
        previous_budget=previous,
        next_budget=next_budget or previous,
        reason=reason,
        statuses=statuses,
        consecutive_successes=consecutive_successes,
        consecutive_ooms=consecutive_ooms,
    )


def _result(
    status: StepStatus = StepStatus.SUCCESS,
    *,
    batch_size: int = 1,
    loss: torch.Tensor | None = None,
    store: object | None = None,
) -> StepResult:
    return StepResult(
        status=status,
        phase=None,
        batch_size=batch_size,
        row_ids=torch.arange(batch_size),
        loss=loss,
        store=store,  # type: ignore[arg-type]
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
        or _decision(statuses=tuple(result.status for result in results)),
        recovered_oom=recovered_oom,
    )


def _summary(
    *statuses: StepStatus,
    batch_size: int = 1,
    recovered_oom: bool = False,
    budget_changed: bool = False,
) -> RuntimePassSummary:
    results = tuple(_result(status, batch_size=batch_size) for status in statuses)
    previous = BatchBudget(max_items=4)
    next_budget = BatchBudget(max_items=2) if budget_changed else previous
    decision = _decision(
        previous_budget=previous,
        next_budget=next_budget,
        statuses=statuses,
        consecutive_ooms=1 if StepStatus.OOM_FAULT in statuses else 0,
    )
    return summarize_runtime_pass(
        _pass_result(results, decision=decision, recovered_oom=recovered_oom)
    )


def _field_values(instance: object) -> list[object]:
    return [getattr(instance, field.name) for field in fields(instance)]


def test_empty_history_summary() -> None:
    history = RuntimePassHistory()

    summary = history.summarize()

    assert isinstance(summary, RuntimeHistorySummary)
    assert summary.total_passes == 0
    assert summary.total_results == 0
    assert summary.total_batch_size == 0
    assert summary.total_rows == 0
    assert dict(summary.status_counts) == {}
    assert summary.recovered_oom_passes == 0
    assert summary.oom_passes == 0
    assert summary.budget_changed_passes == 0
    assert summary.latest_summary is None
    assert history.records == ()


def test_append_summary_updates_history_totals() -> None:
    history = RuntimePassHistory()
    first = _summary(StepStatus.SUCCESS, batch_size=2)

    aggregate = history.append_summary(first)

    assert history.records == (first,)
    assert aggregate.total_passes == 1
    assert aggregate.total_results == 1
    assert aggregate.total_batch_size == 2
    assert aggregate.total_rows == 2
    assert dict(aggregate.status_counts) == {StepStatus.SUCCESS: 1}
    assert aggregate.latest_summary == first


def test_append_pass_result_summarizes_without_storing_step_results() -> None:
    history = RuntimePassHistory()
    pass_result = _pass_result((_result(batch_size=3),))

    aggregate = history.append_pass_result(pass_result)

    assert aggregate.total_passes == 1
    assert aggregate.total_batch_size == 3
    assert history.records == (summarize_runtime_pass(pass_result),)


def test_history_aggregates_multiple_status_counts() -> None:
    history = RuntimePassHistory()
    history.append_summary(_summary(StepStatus.SUCCESS, StepStatus.DATA_FAULT))
    aggregate = history.append_summary(
        _summary(StepStatus.SUCCESS, StepStatus.OOM_FAULT, budget_changed=True)
    )

    assert aggregate.total_passes == 2
    assert aggregate.total_results == 4
    assert dict(aggregate.status_counts) == {
        StepStatus.SUCCESS: 2,
        StepStatus.DATA_FAULT: 1,
        StepStatus.OOM_FAULT: 1,
    }
    assert aggregate.oom_passes == 1
    assert aggregate.budget_changed_passes == 1


def test_history_counts_recovered_oom_passes_separately_from_yielded_ooms() -> None:
    history = RuntimePassHistory()
    history.append_summary(_summary(StepStatus.SUCCESS, recovered_oom=True))
    aggregate = history.append_summary(_summary(StepStatus.OOM_FAULT))

    assert aggregate.recovered_oom_passes == 1
    assert aggregate.oom_passes == 1
    assert dict(aggregate.status_counts) == {
        StepStatus.SUCCESS: 1,
        StepStatus.OOM_FAULT: 1,
    }


def test_history_tracks_latest_summary() -> None:
    history = RuntimePassHistory()
    first = _summary(StepStatus.SUCCESS)
    second = _summary(StepStatus.OOM_FAULT, budget_changed=True)

    history.append_summary(first)
    aggregate = history.append_summary(second)

    assert aggregate.latest_summary == second
    assert aggregate.latest_summary != first


def test_history_respects_max_records() -> None:
    history = RuntimePassHistory(max_records=2)
    first = _summary(StepStatus.SUCCESS, batch_size=1)
    second = _summary(StepStatus.DATA_FAULT, batch_size=2)
    third = _summary(StepStatus.OOM_FAULT, batch_size=3)

    history.append_summary(first)
    history.append_summary(second)
    aggregate = history.append_summary(third)

    assert history.records == (second, third)
    assert aggregate.total_passes == 2
    assert aggregate.total_batch_size == 5
    assert dict(aggregate.status_counts) == {
        StepStatus.DATA_FAULT: 1,
        StepStatus.OOM_FAULT: 1,
    }


def test_history_rejects_invalid_arguments() -> None:
    with pytest.raises(TypeError, match="max_records"):
        RuntimePassHistory(max_records=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="max_records"):
        RuntimePassHistory(max_records="2")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_records"):
        RuntimePassHistory(max_records=0)

    history = RuntimePassHistory()
    with pytest.raises(TypeError, match="append_summary"):
        history.append_summary(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="append_pass_result"):
        history.append_pass_result(object())  # type: ignore[arg-type]


def test_history_does_not_store_stepresult_loss_or_store_references() -> None:
    loss = torch.tensor(1.0)
    store = object()
    result = _result(loss=loss, store=store)
    pass_result = _pass_result((result,))
    history = RuntimePassHistory()

    aggregate = history.append_pass_result(pass_result)
    stored_summary = history.records[0]
    values = _field_values(stored_summary) + _field_values(aggregate)

    assert all(value is not result for value in values)
    assert all(value is not loss for value in values)
    assert all(value is not store for value in values)


def test_records_property_returns_snapshot() -> None:
    history = RuntimePassHistory()
    first = _summary(StepStatus.SUCCESS)
    history.append_summary(first)

    records = history.records
    history.append_summary(_summary(StepStatus.DATA_FAULT))

    assert records == (first,)
    assert len(history.records) == 2


def test_format_runtime_history_summary_is_stable_text() -> None:
    history = RuntimePassHistory()
    history.append_summary(_summary(StepStatus.SUCCESS, recovered_oom=True))
    aggregate = history.append_summary(_summary(StepStatus.OOM_FAULT, budget_changed=True))

    text = format_runtime_history_summary(aggregate)

    assert text.splitlines()[0] == "Runtime history summary"
    assert "total_passes=2" in text
    assert "total_results=2" in text
    assert "success=1" in text
    assert "oom_fault=1" in text
    assert "recovered_oom_passes=1" in text
    assert "oom_passes=1" in text
    assert "budget_changed_passes=1" in text
    assert "latest_budget_changed=True" in text


def test_format_runtime_history_summary_handles_empty_history() -> None:
    text = format_runtime_history_summary(RuntimePassHistory().summarize())

    assert "statuses=none" in text
    assert "latest_budget_changed=False" in text
    assert "latest_recovered_oom=False" in text


def test_format_runtime_history_summary_rejects_invalid_argument() -> None:
    with pytest.raises(TypeError, match="RuntimeHistorySummary"):
        format_runtime_history_summary(object())  # type: ignore[arg-type]
