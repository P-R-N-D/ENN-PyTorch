from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .batching import BatchBudget
from .faults import StepResult, StepStatus
from .governor import GovernorDecision
from .orchestration import RuntimePassResult


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
        total_rows += int(result.row_ids.numel())

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
    )


def format_runtime_pass_summary(summary: RuntimePassSummary) -> str:
    """Return a stable human-readable text summary for runtime pass inspection."""

    if not isinstance(summary, RuntimePassSummary):
        raise TypeError("format_runtime_pass_summary expects a RuntimePassSummary.")

    status_text = _format_status_counts(summary.status_counts)
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
            f"decision_reason={summary.decision_reason}",
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
