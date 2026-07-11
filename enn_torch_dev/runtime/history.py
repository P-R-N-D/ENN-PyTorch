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


class RuntimePassHistory:
    """Keep finite RuntimePassSummary records in memory for inspection."""

    def __init__(self, *, max_records: int | None = None) -> None:
        if max_records is not None:
            if not isinstance(max_records, int) or isinstance(max_records, bool):
                raise TypeError("RuntimePassHistory.max_records must be an integer or None.")
            if max_records <= 0:
                raise ValueError("RuntimePassHistory.max_records must be positive when set.")
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
        )

    def _trim_records(self) -> None:
        if self.max_records is None:
            return
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
            f"latest_budget_changed={latest_budget_changed}",
            f"latest_recovered_oom={latest_recovered_oom}",
        )
    )


def _format_status_counts(status_counts: Mapping[StepStatus, int]) -> str:
    parts = [
        f"{status.value}={status_counts[status]}"
        for status in StepStatus
        if status_counts.get(status, 0)
    ]
    if not parts:
        return "none"
    return ", ".join(parts)
