from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .faults import StepStatus
from .orchestration import RuntimePassResult
from .summary import RuntimePassSummary, summarize_runtime_pass


@dataclass(frozen=True, slots=True)
class RuntimeHistorySummary:
    """Aggregated inspection record for an in-memory runtime pass history."""

    total_passes: int
    total_results: int
    total_batch_size: int
    total_rows: int
    status_counts: Mapping[StepStatus, int]
    recovered_oom_passes: int
    oom_passes: int
    budget_changed_passes: int
    latest_summary: RuntimePassSummary | None = None
    pressure_assessed_passes: int = 0
    pressure_growth_suppressed_passes: int = 0
    peak_observed_pressure_ratio: float | None = None
    pressure_shrink_passes: int = 0
    cpu_pressure_high_passes: int = 0
    cuda_pressure_high_passes: int = 0
    cpu_pressure_trigger_passes: int = 0
    cuda_pressure_trigger_passes: int = 0
    pressure_adjustment_attempt_passes: int = 0
    pressure_adjustment_noop_passes: int = 0
    pressure_trigger_without_budget_passes: int = 0
    host_budget_pressure_shrink_passes: int = 0
    device_budget_pressure_shrink_passes: int = 0
    items_pressure_fallback_shrink_passes: int = 0
    admission_assessed_passes: int = 0
    admission_recovery_passes: int = 0
    admission_total_assessments: int = 0
    admission_admit_assessments: int = 0
    admission_recovered_rejects: int = 0
    admission_allowed_unknowns: int = 0
    minimum_recovered_admissible_items: int | None = None


class RuntimePassHistory:
    """Keep finite RuntimePassSummary records in memory for inspection."""

    def __init__(
        self, *, max_records: int
    ) -> None:
        if not isinstance(max_records, int) or isinstance(max_records, bool):
            raise TypeError("RuntimePassHistory.max_records must be an integer.")
        if max_records <= 0:
            raise ValueError("RuntimePassHistory.max_records must be positive.")
        self.max_records = max_records
        self._records: list[RuntimePassSummary] = []

    @property
    def records(self) -> tuple[RuntimePassSummary, ...]:
        return tuple(self._records)

    def append_summary(self, summary: RuntimePassSummary) -> RuntimeHistorySummary:
        if not isinstance(summary, RuntimePassSummary):
            raise TypeError("RuntimePassHistory.append_summary expects a RuntimePassSummary.")
        self._records.append(summary)
        self._trim_records()
        return self.summarize()

    def append_pass_result(self, pass_result: RuntimePassResult) -> RuntimeHistorySummary:
        if not isinstance(pass_result, RuntimePassResult):
            raise TypeError("RuntimePassHistory.append_pass_result expects a RuntimePassResult.")
        return self.append_summary(summarize_runtime_pass(pass_result))

    def summarize(self) -> RuntimeHistorySummary:
        status_counts: Counter[StepStatus] = Counter()
        total_results = 0
        total_batch_size = 0
        total_rows = 0
        recovered_oom_passes = 0
        oom_passes = 0
        budget_changed_passes = 0
        pressure_assessed_passes = 0
        pressure_growth_suppressed_passes = 0
        peak_observed_pressure_ratio: float | None = None
        pressure_shrink_passes = 0
        cpu_pressure_high_passes = 0
        cuda_pressure_high_passes = 0
        cpu_pressure_trigger_passes = 0
        cuda_pressure_trigger_passes = 0
        pressure_adjustment_attempt_passes = 0
        pressure_adjustment_noop_passes = 0
        pressure_trigger_without_budget_passes = 0
        host_budget_pressure_shrink_passes = 0
        device_budget_pressure_shrink_passes = 0
        items_pressure_fallback_shrink_passes = 0
        admission_assessed_passes = 0
        admission_recovery_passes = 0
        admission_total_assessments = 0
        admission_admit_assessments = 0
        admission_recovered_rejects = 0
        admission_allowed_unknowns = 0
        minimum_recovered_admissible_items: int | None = None

        for summary in self._records:
            total_results += summary.total_results
            total_batch_size += summary.total_batch_size
            total_rows += summary.total_rows
            for status, count in summary.status_counts.items():
                status_counts[status] += int(count)
            if summary.recovered_oom:
                recovered_oom_passes += 1
            if summary.saw_oom:
                oom_passes += 1
            if summary.budget_changed:
                budget_changed_passes += 1
            pressure_summary = summary.pressure_summary
            if pressure_summary is not None:
                pressure_assessed_passes += 1
                candidate_ratio = pressure_summary.max_observed_ratio
                if candidate_ratio is not None:
                    peak_observed_pressure_ratio = (
                        candidate_ratio
                        if peak_observed_pressure_ratio is None
                        else max(peak_observed_pressure_ratio, candidate_ratio)
                    )
            if summary.growth_suppressed_by_pressure:
                pressure_growth_suppressed_passes += 1
            if summary.budget_shrunk_by_pressure:
                pressure_shrink_passes += 1

            high_dimensions = summary.pressure_high_dimensions
            if "cpu" in high_dimensions:
                cpu_pressure_high_passes += 1
            if "cuda" in high_dimensions:
                cuda_pressure_high_passes += 1

            triggered_dimensions = summary.pressure_triggered_dimensions
            if "cpu" in triggered_dimensions:
                cpu_pressure_trigger_passes += 1
            if "cuda" in triggered_dimensions:
                cuda_pressure_trigger_passes += 1

            selected_fields = summary.pressure_selected_budget_fields
            shrunk_fields = summary.pressure_shrunk_budget_fields
            if selected_fields:
                pressure_adjustment_attempt_passes += 1
                if not shrunk_fields:
                    pressure_adjustment_noop_passes += 1
            elif triggered_dimensions:
                pressure_trigger_without_budget_passes += 1

            if "max_host_bytes" in shrunk_fields:
                host_budget_pressure_shrink_passes += 1
            if "max_device_bytes" in shrunk_fields:
                device_budget_pressure_shrink_passes += 1
            if "max_items" in shrunk_fields:
                items_pressure_fallback_shrink_passes += 1

            if summary.admission_assessment_count > 0:
                admission_assessed_passes += 1
            if summary.admission_recovery_occurred:
                admission_recovery_passes += 1
            admission_total_assessments += summary.admission_assessment_count
            admission_admit_assessments += summary.admission_admit_assessment_count
            admission_recovered_rejects += summary.admission_recovered_reject_count
            admission_allowed_unknowns += summary.admission_allowed_unknown_count
            candidate_limit = summary.minimum_recovered_admissible_items
            if candidate_limit is not None:
                minimum_recovered_admissible_items = (
                    candidate_limit
                    if minimum_recovered_admissible_items is None
                    else min(minimum_recovered_admissible_items, candidate_limit)
                )

        latest_summary = self._records[-1] if self._records else None
        status_counts_view: Mapping[StepStatus, int] = MappingProxyType(dict(status_counts))
        return RuntimeHistorySummary(
            total_passes=len(self._records),
            total_results=total_results,
            total_batch_size=total_batch_size,
            total_rows=total_rows,
            status_counts=status_counts_view,
            recovered_oom_passes=recovered_oom_passes,
            oom_passes=oom_passes,
            budget_changed_passes=budget_changed_passes,
            latest_summary=latest_summary,
            pressure_assessed_passes=pressure_assessed_passes,
            pressure_growth_suppressed_passes=pressure_growth_suppressed_passes,
            peak_observed_pressure_ratio=peak_observed_pressure_ratio,
            pressure_shrink_passes=pressure_shrink_passes,
            cpu_pressure_high_passes=cpu_pressure_high_passes,
            cuda_pressure_high_passes=cuda_pressure_high_passes,
            cpu_pressure_trigger_passes=cpu_pressure_trigger_passes,
            cuda_pressure_trigger_passes=cuda_pressure_trigger_passes,
            pressure_adjustment_attempt_passes=pressure_adjustment_attempt_passes,
            pressure_adjustment_noop_passes=pressure_adjustment_noop_passes,
            pressure_trigger_without_budget_passes=pressure_trigger_without_budget_passes,
            host_budget_pressure_shrink_passes=host_budget_pressure_shrink_passes,
            device_budget_pressure_shrink_passes=device_budget_pressure_shrink_passes,
            items_pressure_fallback_shrink_passes=items_pressure_fallback_shrink_passes,
            admission_assessed_passes=admission_assessed_passes,
            admission_recovery_passes=admission_recovery_passes,
            admission_total_assessments=admission_total_assessments,
            admission_admit_assessments=admission_admit_assessments,
            admission_recovered_rejects=admission_recovered_rejects,
            admission_allowed_unknowns=admission_allowed_unknowns,
            minimum_recovered_admissible_items=minimum_recovered_admissible_items,
        )

    def _trim_records(self) -> None:
        overflow = len(self._records) - self.max_records
        if overflow > 0:
            del self._records[:overflow]


def format_runtime_history_summary(summary: RuntimeHistorySummary) -> str:
    """Return stable human-readable text for runtime history inspection."""

    if not isinstance(summary, RuntimeHistorySummary):
        raise TypeError("format_runtime_history_summary expects a RuntimeHistorySummary.")

    latest = summary.latest_summary
    latest_budget_changed = latest.budget_changed if latest is not None else False
    latest_recovered_oom = latest.recovered_oom if latest is not None else False
    latest_pressure_summary = latest.pressure_summary if latest is not None else None
    latest_max_pressure_ratio = (
        latest_pressure_summary.max_observed_ratio
        if latest_pressure_summary is not None
        else None
    )
    latest_growth_suppressed = (
        latest.growth_suppressed_by_pressure if latest is not None else False
    )
    latest_admission_recovery_occurred = (
        latest.admission_recovery_occurred if latest is not None else False
    )
    latest_admission_recovered_reject_count = (
        latest.admission_recovered_reject_count if latest is not None else 0
    )
    return "\n".join(
        (
            "Runtime history summary",
            f"total_passes={summary.total_passes}",
            f"total_results={summary.total_results}",
            f"total_batch_size={summary.total_batch_size}",
            f"total_rows={summary.total_rows}",
            f"statuses={_format_status_counts(summary.status_counts)}",
            f"recovered_oom_passes={summary.recovered_oom_passes}",
            f"oom_passes={summary.oom_passes}",
            f"budget_changed_passes={summary.budget_changed_passes}",
            f"admission_assessed_passes={summary.admission_assessed_passes}",
            f"admission_recovery_passes={summary.admission_recovery_passes}",
            f"admission_total_assessments={summary.admission_total_assessments}",
            f"admission_admit_assessments={summary.admission_admit_assessments}",
            f"admission_recovered_rejects={summary.admission_recovered_rejects}",
            f"admission_allowed_unknowns={summary.admission_allowed_unknowns}",
            "minimum_recovered_admissible_items="
            f"{_format_optional_int(summary.minimum_recovered_admissible_items)}",
            f"pressure_assessed_passes={summary.pressure_assessed_passes}",
            f"pressure_growth_suppressed_passes={summary.pressure_growth_suppressed_passes}",
            f"peak_observed_pressure_ratio={_format_optional_ratio(summary.peak_observed_pressure_ratio)}",
            f"pressure_shrink_passes={summary.pressure_shrink_passes}",
            f"cpu_pressure_high_passes={summary.cpu_pressure_high_passes}",
            f"cuda_pressure_high_passes={summary.cuda_pressure_high_passes}",
            f"cpu_pressure_trigger_passes={summary.cpu_pressure_trigger_passes}",
            f"cuda_pressure_trigger_passes={summary.cuda_pressure_trigger_passes}",
            "pressure_adjustment_attempt_passes="
            f"{summary.pressure_adjustment_attempt_passes}",
            "pressure_adjustment_noop_passes="
            f"{summary.pressure_adjustment_noop_passes}",
            "pressure_trigger_without_budget_passes="
            f"{summary.pressure_trigger_without_budget_passes}",
            "host_budget_pressure_shrink_passes="
            f"{summary.host_budget_pressure_shrink_passes}",
            "device_budget_pressure_shrink_passes="
            f"{summary.device_budget_pressure_shrink_passes}",
            "items_pressure_fallback_shrink_passes="
            f"{summary.items_pressure_fallback_shrink_passes}",
            f"latest_budget_changed={latest_budget_changed}",
            f"latest_recovered_oom={latest_recovered_oom}",
            "latest_admission_recovery_occurred="
            f"{latest_admission_recovery_occurred}",
            "latest_admission_recovered_reject_count="
            f"{latest_admission_recovered_reject_count}",
            f"latest_pressure_assessed={latest_pressure_summary is not None}",
            f"latest_max_pressure_ratio={_format_optional_ratio(latest_max_pressure_ratio)}",
            f"latest_growth_suppressed_by_pressure={latest_growth_suppressed}",
        )
    )


def _format_optional_ratio(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.6g}"


def _format_optional_int(value: int | None) -> str:
    return "unknown" if value is None else str(value)


def _format_status_counts(status_counts: Mapping[StepStatus, int]) -> str:
    parts = [
        f"{status.value}={status_counts[status]}"
        for status in StepStatus
        if status_counts.get(status, 0)
    ]
    if not parts:
        return "none"
    return ", ".join(parts)
