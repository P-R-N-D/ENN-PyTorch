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
    RuntimeHistorySummary,
    RuntimePassHistory,
    RuntimePassResult,
    RuntimePassSummary,
    StepResult,
    StepStatus,
    assess_resource_pressure,
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
    pressure_summary: ResourcePressureSummary | None = None,
    growth_suppressed_by_pressure: bool = False,
    budget_shrunk_by_pressure: bool = False,
    pressure_shrunk_budget_fields: tuple[str, ...] = (),
    pressure_high_dimensions: tuple[str, ...] = (),
    pressure_triggered_dimensions: tuple[str, ...] = (),
    pressure_selected_budget_fields: tuple[str, ...] = (),
    pressure_applied_shrink_factors: tuple[tuple[str, float], ...] = (),
) -> GovernorDecision:
    previous = previous_budget or BatchBudget(max_items=4)
    return GovernorDecision(
        previous_budget=previous,
        next_budget=next_budget or previous,
        reason=reason,
        statuses=statuses,
        consecutive_successes=consecutive_successes,
        consecutive_ooms=consecutive_ooms,
        pressure_summary=pressure_summary,
        growth_suppressed_by_pressure=growth_suppressed_by_pressure,
        budget_shrunk_by_pressure=budget_shrunk_by_pressure,
        pressure_shrunk_budget_fields=pressure_shrunk_budget_fields,
        pressure_high_dimensions=pressure_high_dimensions,
        pressure_triggered_dimensions=pressure_triggered_dimensions,
        pressure_selected_budget_fields=pressure_selected_budget_fields,
        pressure_applied_shrink_factors=pressure_applied_shrink_factors,
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
    pressure_summary: ResourcePressureSummary | None = None,
    growth_suppressed_by_pressure: bool = False,
    budget_shrunk_by_pressure: bool = False,
    pressure_shrunk_budget_fields: tuple[str, ...] = (),
    pressure_high_dimensions: tuple[str, ...] = (),
    pressure_triggered_dimensions: tuple[str, ...] = (),
    pressure_selected_budget_fields: tuple[str, ...] = (),
    pressure_applied_shrink_factors: tuple[tuple[str, float], ...] = (),
) -> RuntimePassSummary:
    results = tuple(_result(status, batch_size=batch_size) for status in statuses)
    previous = BatchBudget(max_items=4)
    next_budget = BatchBudget(max_items=2) if budget_changed else previous
    decision = _decision(
        previous_budget=previous,
        next_budget=next_budget,
        statuses=statuses,
        consecutive_ooms=1 if StepStatus.OOM_FAULT in statuses else 0,
        pressure_summary=pressure_summary,
        growth_suppressed_by_pressure=growth_suppressed_by_pressure,
        budget_shrunk_by_pressure=budget_shrunk_by_pressure,
        pressure_shrunk_budget_fields=pressure_shrunk_budget_fields,
        pressure_high_dimensions=pressure_high_dimensions,
        pressure_triggered_dimensions=pressure_triggered_dimensions,
        pressure_selected_budget_fields=pressure_selected_budget_fields,
        pressure_applied_shrink_factors=pressure_applied_shrink_factors,
    )
    return summarize_runtime_pass(
        _pass_result(results, decision=decision, recovered_oom=recovered_oom)
    )


def _field_values(instance: object) -> list[object]:
    return [getattr(instance, field.name) for field in fields(instance)]


def test_runtime_history_summary_appends_pressure_fields_for_compatibility() -> None:
    field_names = [field.name for field in fields(RuntimeHistorySummary)]

    assert field_names[-21:] == [
        "pressure_assessed_passes",
        "pressure_growth_suppressed_passes",
        "peak_observed_pressure_ratio",
        "pressure_shrink_passes",
        "cpu_pressure_high_passes",
        "cuda_pressure_high_passes",
        "cpu_pressure_trigger_passes",
        "cuda_pressure_trigger_passes",
        "pressure_adjustment_attempt_passes",
        "pressure_adjustment_noop_passes",
        "pressure_trigger_without_budget_passes",
        "host_budget_pressure_shrink_passes",
        "device_budget_pressure_shrink_passes",
        "items_pressure_fallback_shrink_passes",
        "admission_assessed_passes",
        "admission_recovery_passes",
        "admission_total_assessments",
        "admission_admit_assessments",
        "admission_recovered_rejects",
        "admission_allowed_unknowns",
        "minimum_recovered_admissible_items",
    ]


def test_empty_history_summary() -> None:
    history = RuntimePassHistory(max_records=10)

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
    assert summary.pressure_assessed_passes == 0
    assert summary.pressure_growth_suppressed_passes == 0
    assert summary.peak_observed_pressure_ratio is None
    assert summary.pressure_shrink_passes == 0
    assert summary.cpu_pressure_high_passes == 0
    assert summary.cuda_pressure_high_passes == 0
    assert summary.cpu_pressure_trigger_passes == 0
    assert summary.cuda_pressure_trigger_passes == 0
    assert summary.pressure_adjustment_attempt_passes == 0
    assert summary.pressure_adjustment_noop_passes == 0
    assert summary.pressure_trigger_without_budget_passes == 0
    assert summary.host_budget_pressure_shrink_passes == 0
    assert summary.device_budget_pressure_shrink_passes == 0
    assert summary.items_pressure_fallback_shrink_passes == 0
    assert summary.latest_summary is None
    assert history.records == ()


def test_history_counts_actual_pressure_shrink_passes() -> None:
    history = RuntimePassHistory(max_records=3)
    history.append_summary(
        _summary(
            StepStatus.SUCCESS,
            pressure_summary=ResourcePressureSummary(peak_cpu_rss_ratio=0.95),
            growth_suppressed_by_pressure=True,
        )
    )
    aggregate = history.append_summary(
        _summary(
            StepStatus.SUCCESS,
            budget_changed=True,
            pressure_summary=ResourcePressureSummary(peak_cpu_rss_ratio=0.96),
            growth_suppressed_by_pressure=True,
            budget_shrunk_by_pressure=True,
        )
    )

    assert aggregate.pressure_shrink_passes == 1
    assert "pressure_shrink_passes=1" in format_runtime_history_summary(aggregate)


def test_history_aggregates_structured_pressure_provenance() -> None:
    history = RuntimePassHistory(max_records=10)
    history.append_summary(_summary(StepStatus.SUCCESS, pressure_high_dimensions=("cpu",)))
    history.append_summary(
        _summary(
            StepStatus.SUCCESS,
            pressure_high_dimensions=("cuda",),
            pressure_triggered_dimensions=("cuda",),
        )
    )
    history.append_summary(
        _summary(
            StepStatus.SUCCESS,
            pressure_high_dimensions=("cpu",),
            pressure_triggered_dimensions=("cpu",),
            pressure_selected_budget_fields=("max_host_bytes",),
            pressure_applied_shrink_factors=(("max_host_bytes", 0.75),),
        )
    )
    history.append_summary(
        _summary(
            StepStatus.SUCCESS,
            budget_changed=True,
            budget_shrunk_by_pressure=True,
            pressure_shrunk_budget_fields=("max_host_bytes", "max_device_bytes"),
            pressure_high_dimensions=("cpu", "cuda"),
            pressure_triggered_dimensions=("cpu", "cuda"),
            pressure_selected_budget_fields=("max_host_bytes", "max_device_bytes"),
            pressure_applied_shrink_factors=(
                ("max_host_bytes", 0.75),
                ("max_device_bytes", 0.4),
            ),
        )
    )
    aggregate = history.append_summary(
        _summary(
            StepStatus.SUCCESS,
            budget_changed=True,
            budget_shrunk_by_pressure=True,
            pressure_shrunk_budget_fields=("max_items",),
            pressure_high_dimensions=("cpu", "cuda"),
            pressure_triggered_dimensions=("cpu", "cuda"),
            pressure_selected_budget_fields=("max_items",),
            pressure_applied_shrink_factors=(("max_items", 0.4),),
        )
    )

    assert aggregate.cpu_pressure_high_passes == 4
    assert aggregate.cuda_pressure_high_passes == 3
    assert aggregate.cpu_pressure_trigger_passes == 3
    assert aggregate.cuda_pressure_trigger_passes == 3
    assert aggregate.pressure_adjustment_attempt_passes == 3
    assert aggregate.pressure_adjustment_noop_passes == 1
    assert aggregate.pressure_trigger_without_budget_passes == 1
    assert aggregate.host_budget_pressure_shrink_passes == 1
    assert aggregate.device_budget_pressure_shrink_passes == 1
    assert aggregate.items_pressure_fallback_shrink_passes == 1
    assert aggregate.pressure_shrink_passes == 2

    text = format_runtime_history_summary(aggregate)
    for expected in (
        "cpu_pressure_high_passes=4",
        "cuda_pressure_high_passes=3",
        "cpu_pressure_trigger_passes=3",
        "cuda_pressure_trigger_passes=3",
        "pressure_adjustment_attempt_passes=3",
        "pressure_adjustment_noop_passes=1",
        "pressure_trigger_without_budget_passes=1",
        "host_budget_pressure_shrink_passes=1",
        "device_budget_pressure_shrink_passes=1",
        "items_pressure_fallback_shrink_passes=1",
    ):
        assert expected in text


def test_history_does_not_infer_pressure_provenance_from_oom_passes() -> None:
    history = RuntimePassHistory(max_records=2)
    history.append_summary(
        _summary(
            StepStatus.OOM_FAULT,
            budget_changed=True,
            pressure_summary=ResourcePressureSummary(peak_cpu_rss_ratio=1.1),
        )
    )
    aggregate = history.append_summary(
        _summary(
            StepStatus.SUCCESS,
            recovered_oom=True,
            budget_changed=True,
            pressure_summary=ResourcePressureSummary(peak_cuda_reserved_ratio=1.2),
        )
    )

    assert aggregate.cpu_pressure_high_passes == 0
    assert aggregate.cuda_pressure_high_passes == 0
    assert aggregate.cpu_pressure_trigger_passes == 0
    assert aggregate.cuda_pressure_trigger_passes == 0
    assert aggregate.pressure_adjustment_attempt_passes == 0
    assert aggregate.pressure_adjustment_noop_passes == 0
    assert aggregate.pressure_trigger_without_budget_passes == 0
    assert aggregate.host_budget_pressure_shrink_passes == 0
    assert aggregate.device_budget_pressure_shrink_passes == 0
    assert aggregate.items_pressure_fallback_shrink_passes == 0


def test_history_does_not_count_partial_adjustment_as_full_noop() -> None:
    history = RuntimePassHistory(max_records=1)

    aggregate = history.append_summary(
        _summary(
            StepStatus.SUCCESS,
            budget_changed=True,
            budget_shrunk_by_pressure=True,
            pressure_high_dimensions=("cpu", "cuda"),
            pressure_triggered_dimensions=("cpu", "cuda"),
            pressure_selected_budget_fields=(
                "max_host_bytes",
                "max_device_bytes",
            ),
            pressure_shrunk_budget_fields=("max_device_bytes",),
            pressure_applied_shrink_factors=(
                ("max_host_bytes", 0.75),
                ("max_device_bytes", 0.4),
            ),
        )
    )

    assert aggregate.pressure_adjustment_attempt_passes == 1
    assert aggregate.pressure_adjustment_noop_passes == 0
    assert aggregate.host_budget_pressure_shrink_passes == 0
    assert aggregate.device_budget_pressure_shrink_passes == 1
    assert aggregate.cpu_pressure_trigger_passes == 1
    assert aggregate.cuda_pressure_trigger_passes == 1
    assert aggregate.pressure_shrink_passes == 1


def test_append_summary_updates_history_totals() -> None:
    history = RuntimePassHistory(max_records=10)
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
    history = RuntimePassHistory(max_records=10)
    pass_result = _pass_result((_result(batch_size=3),))

    aggregate = history.append_pass_result(pass_result)

    assert aggregate.total_passes == 1
    assert aggregate.total_batch_size == 3
    assert history.records == (summarize_runtime_pass(pass_result),)

def test_history_aggregates_multiple_status_counts() -> None:
    history = RuntimePassHistory(max_records=10)
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
    history = RuntimePassHistory(max_records=10)
    history.append_summary(_summary(StepStatus.SUCCESS, recovered_oom=True))
    aggregate = history.append_summary(_summary(StepStatus.OOM_FAULT))

    assert aggregate.recovered_oom_passes == 1
    assert aggregate.oom_passes == 1
    assert dict(aggregate.status_counts) == {
        StepStatus.SUCCESS: 1,
        StepStatus.OOM_FAULT: 1,
    }

def test_history_aggregates_pressure_feedback() -> None:
    history = RuntimePassHistory(max_records=10)
    history.append_summary(_summary(StepStatus.SUCCESS))
    history.append_summary(
        _summary(
            StepStatus.SUCCESS,
            pressure_summary=ResourcePressureSummary(),
        )
    )
    history.append_summary(
        _summary(
            StepStatus.SUCCESS,
            pressure_summary=ResourcePressureSummary(peak_cpu_rss_ratio=0.5),
        )
    )
    aggregate = history.append_summary(
        _summary(
            StepStatus.SUCCESS,
            pressure_summary=ResourcePressureSummary(
                peak_cuda_reserved_ratio=0.9
            ),
            growth_suppressed_by_pressure=True,
        )
    )

    assert aggregate.pressure_assessed_passes == 3
    assert aggregate.pressure_growth_suppressed_passes == 1
    assert aggregate.peak_observed_pressure_ratio == pytest.approx(0.9)


def test_history_pressure_aggregation_respects_retained_window() -> None:
    history = RuntimePassHistory(max_records=2)
    first = _summary(
        StepStatus.SUCCESS,
        pressure_summary=ResourcePressureSummary(peak_cpu_rss_ratio=0.95),
        growth_suppressed_by_pressure=True,
    )
    second = _summary(
        StepStatus.SUCCESS,
        pressure_summary=ResourcePressureSummary(peak_cpu_rss_ratio=0.4),
    )
    third = _summary(
        StepStatus.SUCCESS,
        pressure_summary=ResourcePressureSummary(),
    )

    history.append_summary(first)
    history.append_summary(second)
    aggregate = history.append_summary(third)

    assert history.records == (second, third)
    assert aggregate.pressure_assessed_passes == 2
    assert aggregate.pressure_growth_suppressed_passes == 0
    assert aggregate.peak_observed_pressure_ratio == pytest.approx(0.4)


def test_history_structured_provenance_respects_retained_window() -> None:
    history = RuntimePassHistory(max_records=2)
    first = _summary(
        StepStatus.SUCCESS,
        budget_changed=True,
        budget_shrunk_by_pressure=True,
        pressure_shrunk_budget_fields=("max_host_bytes",),
        pressure_high_dimensions=("cpu",),
        pressure_triggered_dimensions=("cpu",),
        pressure_selected_budget_fields=("max_host_bytes",),
        pressure_applied_shrink_factors=(("max_host_bytes", 0.75),),
    )
    second = _summary(
        StepStatus.SUCCESS,
        pressure_high_dimensions=("cuda",),
        pressure_triggered_dimensions=("cuda",),
        pressure_selected_budget_fields=("max_device_bytes",),
        pressure_applied_shrink_factors=(("max_device_bytes", 0.4),),
    )
    third = _summary(
        StepStatus.SUCCESS,
        budget_changed=True,
        budget_shrunk_by_pressure=True,
        pressure_shrunk_budget_fields=("max_items",),
        pressure_high_dimensions=("cpu", "cuda"),
        pressure_triggered_dimensions=("cpu", "cuda"),
        pressure_selected_budget_fields=("max_items",),
        pressure_applied_shrink_factors=(("max_items", 0.4),),
    )

    history.append_summary(first)
    history.append_summary(second)
    aggregate = history.append_summary(third)

    assert history.records == (second, third)
    assert aggregate.cpu_pressure_high_passes == 1
    assert aggregate.cuda_pressure_high_passes == 2
    assert aggregate.cpu_pressure_trigger_passes == 1
    assert aggregate.cuda_pressure_trigger_passes == 2
    assert aggregate.pressure_adjustment_attempt_passes == 2
    assert aggregate.pressure_adjustment_noop_passes == 1
    assert aggregate.pressure_trigger_without_budget_passes == 0
    assert aggregate.host_budget_pressure_shrink_passes == 0
    assert aggregate.device_budget_pressure_shrink_passes == 0
    assert aggregate.items_pressure_fallback_shrink_passes == 1
    assert aggregate.pressure_shrink_passes == 1

def test_history_tracks_latest_summary() -> None:
    history = RuntimePassHistory(max_records=10)
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


def test_history_has_no_unbounded_retention_path() -> None:
    history = RuntimePassHistory(max_records=1)
    first = _summary(StepStatus.SUCCESS)
    second = _summary(StepStatus.DATA_FAULT)

    history.append_summary(first)
    history.append_summary(second)

    assert history.max_records == 1
    assert history.records == (second,)


def test_history_rejects_invalid_arguments() -> None:
    with pytest.raises(TypeError):
        RuntimePassHistory()
    with pytest.raises(TypeError, match="max_records"):
        RuntimePassHistory(max_records=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="max_records"):
        RuntimePassHistory(max_records=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="max_records"):
        RuntimePassHistory(max_records="2")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_records"):
        RuntimePassHistory(max_records=0)
    with pytest.raises(ValueError, match="max_records"):
        RuntimePassHistory(max_records=-1)

    history = RuntimePassHistory(max_records=10)
    with pytest.raises(TypeError, match="append_summary"):
        history.append_summary(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="append_pass_result"):
        history.append_pass_result(object())  # type: ignore[arg-type]


def test_history_does_not_store_raw_runtime_references() -> None:
    loss = torch.tensor(1.0)
    store = object()
    result = _result(loss=loss, store=store)
    sample = ResourceSample(
        timestamp_ns=1,
        phase="history-test",
        cpu_rss_bytes=50,
    )
    pressure = assess_resource_pressure(
        (sample,),
        ResourceCapacity(cpu_total_bytes=100),
    )
    pass_result = _pass_result(
        (result,),
        decision=_decision(pressure_summary=pressure),
    )
    history = RuntimePassHistory(max_records=10)

    aggregate = history.append_pass_result(pass_result)
    stored_summary = history.records[0]
    values = _field_values(stored_summary) + _field_values(aggregate)

    assert all(value is not result for value in values)
    assert all(value is not loss for value in values)
    assert all(value is not store for value in values)
    assert all(value is not sample for value in values)


def test_records_property_returns_snapshot() -> None:
    history = RuntimePassHistory(max_records=10)
    first = _summary(StepStatus.SUCCESS)
    history.append_summary(first)

    records = history.records
    history.append_summary(_summary(StepStatus.DATA_FAULT))

    assert records == (first,)
    assert len(history.records) == 2


def test_format_runtime_history_summary_is_stable_text() -> None:
    history = RuntimePassHistory(max_records=10)
    history.append_summary(
        _summary(
            StepStatus.SUCCESS,
            recovered_oom=True,
            pressure_summary=ResourcePressureSummary(peak_cpu_rss_ratio=0.5),
        )
    )
    aggregate = history.append_summary(
        _summary(
            StepStatus.OOM_FAULT,
            budget_changed=True,
            pressure_summary=ResourcePressureSummary(
                peak_cuda_reserved_ratio=0.9
            ),
            growth_suppressed_by_pressure=True,
        )
    )

    text = format_runtime_history_summary(aggregate)

    assert text.splitlines()[0] == "Runtime history summary"
    assert "total_passes=2" in text
    assert "total_results=2" in text
    assert "success=1" in text
    assert "oom_fault=1" in text
    assert "recovered_oom_passes=1" in text
    assert "oom_passes=1" in text
    assert "budget_changed_passes=1" in text
    assert "pressure_assessed_passes=2" in text
    assert "pressure_growth_suppressed_passes=1" in text
    assert "peak_observed_pressure_ratio=0.9" in text
    assert "latest_budget_changed=True" in text
    assert "latest_pressure_assessed=True" in text
    assert "latest_max_pressure_ratio=0.9" in text
    assert "latest_growth_suppressed_by_pressure=True" in text


def test_format_runtime_history_summary_handles_empty_history() -> None:
    text = format_runtime_history_summary(RuntimePassHistory(max_records=10).summarize())

    assert "statuses=none" in text
    assert "latest_budget_changed=False" in text
    assert "latest_recovered_oom=False" in text
    assert "pressure_assessed_passes=0" in text
    assert "peak_observed_pressure_ratio=unknown" in text
    assert "latest_pressure_assessed=False" in text
    assert "latest_max_pressure_ratio=unknown" in text
    assert "latest_growth_suppressed_by_pressure=False" in text


def test_format_runtime_history_summary_rejects_invalid_argument() -> None:
    with pytest.raises(TypeError, match="RuntimeHistorySummary"):
        format_runtime_history_summary(object())  # type: ignore[arg-type]
