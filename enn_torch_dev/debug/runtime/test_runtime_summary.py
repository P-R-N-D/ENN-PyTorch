from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from enn_torch_dev.runtime import (
    BatchBudget,
    GovernorDecision,
    RuntimePassResult,
    RuntimePassSummary,
    StepResult,
    StepStatus,
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


def test_summarize_runtime_pass_does_not_store_stepresult_loss_or_store() -> None:
    loss = torch.tensor(1.0)
    store = object()
    result = _result(loss=loss, store=store)

    summary = summarize_runtime_pass(_pass_result((result,)))
    values = _field_values(summary)

    assert all(value is not result for value in values)
    assert all(value is not loss for value in values)
    assert all(value is not store for value in values)


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
