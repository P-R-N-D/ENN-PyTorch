from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .batching import BatchBudget
from .faults import StepResult, StepStatus
from .governor import GovernorDecision
from .orchestration import RuntimePassResult
from .pressure import ResourceCapacity, ResourcePressureSummary


@dataclass(frozen=True, slots=True)
class RuntimePassSummary:
    """Compact inspection record for one finite runtime orchestration pass."""

    total_results: int
    statuses: tuple[StepStatus, ...]
    status_counts: Mapping[StepStatus, int]
    total_batch_size: int
    total_rows: int
    recovered_oom: bool
    saw_oom: bool
    previous_budget: BatchBudget
    next_budget: BatchBudget
    budget_changed: bool
    decision_reason: str
    consecutive_successes: int
    consecutive_ooms: int
    peak_cpu_rss_bytes: int | None = None
    peak_cuda_allocated_bytes: int | None = None
    peak_cuda_reserved_bytes: int | None = None
    peak_cuda_max_allocated_bytes: int | None = None
    peak_cuda_max_reserved_bytes: int | None = None
    pressure_summary: ResourcePressureSummary | None = None
    growth_suppressed_by_pressure: bool = False
    resource_capacity: ResourceCapacity | None = None
    consecutive_high_pressure_passes: int = 0
    budget_shrunk_by_pressure: bool = False
    pressure_shrunk_budget_fields: tuple[str, ...] = ()


def summarize_runtime_pass(pass_result: RuntimePassResult) -> RuntimePassSummary:
    """Build a lightweight summary from a finite RuntimePassResult."""

    if not isinstance(pass_result, RuntimePassResult):
        raise TypeError("summarize_runtime_pass expects a RuntimePassResult.")
    decision = pass_result.decision
    if not isinstance(decision, GovernorDecision):
        raise TypeError("RuntimePassResult.decision must be a GovernorDecision.")

    statuses: list[StepStatus] = []
    status_counts: Counter[StepStatus] = Counter()
    total_batch_size = 0
    total_rows = 0

    for result in pass_result.results:
        if not isinstance(result, StepResult):
            raise TypeError("RuntimePassResult.results must contain StepResult objects.")
        status = result.status
        statuses.append(status)
        status_counts[status] += 1
        total_batch_size += int(result.batch_size)
        total_rows += int(result.batch_size)

    status_counts_view: Mapping[StepStatus, int] = MappingProxyType(dict(status_counts))
    return RuntimePassSummary(
        total_results=len(statuses),
        statuses=tuple(statuses),
        status_counts=status_counts_view,
        total_batch_size=total_batch_size,
        total_rows=total_rows,
        recovered_oom=pass_result.recovered_oom,
        saw_oom=any(status is StepStatus.OOM_FAULT for status in statuses),
        previous_budget=decision.previous_budget,
        next_budget=decision.next_budget,
        budget_changed=decision.previous_budget != decision.next_budget,
        decision_reason=decision.reason,
        consecutive_successes=decision.consecutive_successes,
        consecutive_ooms=decision.consecutive_ooms,
        peak_cpu_rss_bytes=decision.peak_cpu_rss_bytes,
        peak_cuda_allocated_bytes=decision.peak_cuda_allocated_bytes,
        peak_cuda_reserved_bytes=decision.peak_cuda_reserved_bytes,
        peak_cuda_max_allocated_bytes=decision.peak_cuda_max_allocated_bytes,
        peak_cuda_max_reserved_bytes=decision.peak_cuda_max_reserved_bytes,
        pressure_summary=decision.pressure_summary,
        growth_suppressed_by_pressure=decision.growth_suppressed_by_pressure,
        resource_capacity=pass_result.resource_capacity,
        consecutive_high_pressure_passes=decision.consecutive_high_pressure_passes,
        budget_shrunk_by_pressure=decision.budget_shrunk_by_pressure,
        pressure_shrunk_budget_fields=decision.pressure_shrunk_budget_fields,
    )


def format_runtime_pass_summary(summary: RuntimePassSummary) -> str:
    """Return a stable human-readable text summary for runtime pass inspection."""

    if not isinstance(summary, RuntimePassSummary):
        raise TypeError("format_runtime_pass_summary expects a RuntimePassSummary.")

    status_text = _format_status_counts(summary.status_counts)
    pressure_summary = summary.pressure_summary
    max_pressure_ratio = (
        pressure_summary.max_observed_ratio
        if pressure_summary is not None
        else None
    )
    return "\n".join(
        (
            "Runtime pass summary",
            f"total_results={summary.total_results}",
            f"total_batch_size={summary.total_batch_size}",
            f"total_rows={summary.total_rows}",
            f"statuses={status_text}",
            f"recovered_oom={summary.recovered_oom}",
            f"saw_oom={summary.saw_oom}",
            f"budget_changed={summary.budget_changed}",
            f"previous_budget={summary.previous_budget!r}",
            f"next_budget={summary.next_budget!r}",
            f"consecutive_successes={summary.consecutive_successes}",
            f"consecutive_ooms={summary.consecutive_ooms}",
            f"resource_capacity={summary.resource_capacity!r}",
            f"pressure_assessed={pressure_summary is not None}",
            f"max_pressure_ratio={_format_optional_ratio(max_pressure_ratio)}",
            f"growth_suppressed_by_pressure={summary.growth_suppressed_by_pressure}",
            f"consecutive_high_pressure_passes={summary.consecutive_high_pressure_passes}",
            f"budget_shrunk_by_pressure={summary.budget_shrunk_by_pressure}",
            "pressure_shrunk_budget_fields="
            f"{summary.pressure_shrunk_budget_fields!r}",
            f"decision_reason={summary.decision_reason}",
        )
    )


def _format_optional_ratio(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.6g}"


def _format_status_counts(status_counts: Mapping[StepStatus, int]) -> str:
    parts = [
        f"{status.value}={status_counts[status]}"
        for status in StepStatus
        if status_counts.get(status, 0)
    ]
    if not parts:
        return "none"
    return ", ".join(parts)
