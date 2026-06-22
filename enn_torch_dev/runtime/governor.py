from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

from .batching import BatchBudget
from .faults import ResourceSample, StepResult, StepStatus


_BUDGET_FIELDS = ("max_items", "max_host_bytes", "max_device_bytes")
_BOUND_FIELDS = {
    "max_items": ("min_items", "max_items"),
    "max_host_bytes": ("min_host_bytes", "max_host_bytes"),
    "max_device_bytes": ("min_device_bytes", "max_device_bytes"),
}


def _validate_factor(value: object, *, label: str, lower: float, upper: float | None = None) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"GovernorPolicy.{label} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"GovernorPolicy.{label} must be finite.")
    if number <= lower:
        raise ValueError(f"GovernorPolicy.{label} must be greater than {lower}.")
    if upper is not None and number >= upper:
        raise ValueError(f"GovernorPolicy.{label} must be less than {upper}.")
    return number


def _validate_positive_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer.")
    if value <= 0:
        raise ValueError(f"{label} must be positive.")
    return value


def _validate_optional_positive_int(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    return _validate_positive_int(value, label=label)


def _validate_budget(budget: object, *, label: str) -> BatchBudget:
    if not isinstance(budget, BatchBudget):
        raise TypeError(f"{label} must be a BatchBudget.")
    for field_name in _BUDGET_FIELDS:
        value = getattr(budget, field_name)
        if value is not None and value <= 0:
            raise ValueError(f"{label}.{field_name} must be positive when configured.")
    return budget


@dataclass(frozen=True, slots=True)
class GovernorPolicy:
    """Static conservative budget adjustment policy."""

    shrink_factor: float = 0.5
    grow_factor: float = 2.0
    grow_after_successes: int = 3
    min_items: int | None = None
    max_items: int | None = None
    min_host_bytes: int | None = None
    max_host_bytes: int | None = None
    min_device_bytes: int | None = None
    max_device_bytes: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shrink_factor",
            _validate_factor(
                self.shrink_factor,
                label="shrink_factor",
                lower=0.0,
                upper=1.0,
            ),
        )
        object.__setattr__(
            self,
            "grow_factor",
            _validate_factor(self.grow_factor, label="grow_factor", lower=1.0),
        )
        object.__setattr__(
            self,
            "grow_after_successes",
            _validate_positive_int(
                self.grow_after_successes,
                label="GovernorPolicy.grow_after_successes",
            ),
        )
        for field_name in (
            "min_items",
            "max_items",
            "min_host_bytes",
            "max_host_bytes",
            "min_device_bytes",
            "max_device_bytes",
        ):
            object.__setattr__(
                self,
                field_name,
                _validate_optional_positive_int(
                    getattr(self, field_name),
                    label=f"GovernorPolicy.{field_name}",
                ),
            )
        self._validate_bounds("items", self.min_items, self.max_items)
        self._validate_bounds("host_bytes", self.min_host_bytes, self.max_host_bytes)
        self._validate_bounds("device_bytes", self.min_device_bytes, self.max_device_bytes)

    @staticmethod
    def _validate_bounds(label: str, minimum: int | None, maximum: int | None) -> None:
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"GovernorPolicy min_{label} must be <= max_{label}.")


@dataclass(frozen=True, slots=True)
class GovernorDecision:
    """Public record describing one conservative governor decision."""

    previous_budget: BatchBudget
    next_budget: BatchBudget
    reason: str
    statuses: tuple[StepStatus, ...]
    consecutive_successes: int
    consecutive_ooms: int
    peak_cpu_rss_bytes: int | None = None
    peak_cuda_allocated_bytes: int | None = None
    peak_cuda_reserved_bytes: int | None = None
    peak_cuda_max_allocated_bytes: int | None = None
    peak_cuda_max_reserved_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class RuntimeGovernorState:
    """Reusable immutable state for ConservativeRuntimeGovernor."""

    current_budget: BatchBudget
    consecutive_successes: int = 0
    consecutive_ooms: int = 0
    last_decision: GovernorDecision | None = None

    def __post_init__(self) -> None:
        _validate_budget(self.current_budget, label="RuntimeGovernorState.current_budget")
        object.__setattr__(
            self,
            "consecutive_successes",
            _validate_streak(
                self.consecutive_successes,
                label="RuntimeGovernorState.consecutive_successes",
            ),
        )
        object.__setattr__(
            self,
            "consecutive_ooms",
            _validate_streak(
                self.consecutive_ooms,
                label="RuntimeGovernorState.consecutive_ooms",
            ),
        )
        if self.last_decision is not None and not isinstance(self.last_decision, GovernorDecision):
            raise TypeError("RuntimeGovernorState.last_decision must be a GovernorDecision or None.")


def _validate_streak(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer.")
    if value < 0:
        raise ValueError(f"{label} must be non-negative.")
    return value


def _peak(current: int | None, candidate: int | None) -> int | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return max(current, candidate)


class ConservativeRuntimeGovernor:
    """Observe StepResult streams and conservatively choose the next BatchBudget."""

    def __init__(
        self,
        budget: BatchBudget | None = None,
        *,
        policy: GovernorPolicy | None = None,
        state: RuntimeGovernorState | None = None,
    ) -> None:
        if budget is None and state is None:
            raise ValueError("ConservativeRuntimeGovernor requires either budget or state.")
        if budget is not None and state is not None:
            raise ValueError("ConservativeRuntimeGovernor accepts either budget or state, not both.")
        if budget is not None:
            budget = _validate_budget(budget, label="ConservativeRuntimeGovernor.budget")
        if policy is not None and not isinstance(policy, GovernorPolicy):
            raise TypeError("ConservativeRuntimeGovernor.policy must be a GovernorPolicy or None.")
        if state is not None and not isinstance(state, RuntimeGovernorState):
            raise TypeError("ConservativeRuntimeGovernor.state must be a RuntimeGovernorState or None.")

        self.policy = GovernorPolicy() if policy is None else policy
        self.state = state if state is not None else RuntimeGovernorState(budget)  # type: ignore[arg-type]

    @property
    def current_budget(self) -> BatchBudget:
        return self.state.current_budget

    def observe_results(self, results: Iterable[StepResult]) -> GovernorDecision:
        if isinstance(results, StepResult):
            raise TypeError("ConservativeRuntimeGovernor.observe_results expects an iterable of StepResult objects.")
        observed = tuple(results)
        for result in observed:
            if not isinstance(result, StepResult):
                raise TypeError("ConservativeRuntimeGovernor.observe_results must receive StepResult objects.")

        statuses = tuple(result.status for result in observed)
        peaks = self._resource_peaks(observed)
        previous_budget = self.state.current_budget
        next_budget = previous_budget
        consecutive_successes = self.state.consecutive_successes
        consecutive_ooms = self.state.consecutive_ooms

        if not observed:
            reason = "no results observed; keeping current budget"
        elif any(status is StepStatus.OOM_FAULT for status in statuses):
            next_budget = self._adjust_budget(previous_budget, mode="shrink")
            consecutive_successes = 0
            consecutive_ooms += 1
            reason = "OOM fault observed; shrinking configured budget fields"
        elif all(status is StepStatus.SUCCESS for status in statuses):
            consecutive_ooms = 0
            consecutive_successes += 1
            if consecutive_successes >= self.policy.grow_after_successes:
                next_budget = self._adjust_budget(previous_budget, mode="grow")
                consecutive_successes = 0
                reason = "success threshold reached; growing configured budget fields"
            else:
                reason = "success observed below growth threshold; keeping current budget"
        else:
            consecutive_successes = 0
            consecutive_ooms = 0
            reason = "non-OOM fault observed; keeping current budget"

        reason = self._append_peak_reason(reason, peaks)
        decision = GovernorDecision(
            previous_budget=previous_budget,
            next_budget=next_budget,
            reason=reason,
            statuses=statuses,
            consecutive_successes=consecutive_successes,
            consecutive_ooms=consecutive_ooms,
            peak_cpu_rss_bytes=peaks["peak_cpu_rss_bytes"],
            peak_cuda_allocated_bytes=peaks["peak_cuda_allocated_bytes"],
            peak_cuda_reserved_bytes=peaks["peak_cuda_reserved_bytes"],
            peak_cuda_max_allocated_bytes=peaks["peak_cuda_max_allocated_bytes"],
            peak_cuda_max_reserved_bytes=peaks["peak_cuda_max_reserved_bytes"],
        )
        self.state = RuntimeGovernorState(
            current_budget=next_budget,
            consecutive_successes=consecutive_successes,
            consecutive_ooms=consecutive_ooms,
            last_decision=decision,
        )
        return decision

    def _adjust_budget(self, budget: BatchBudget, *, mode: str) -> BatchBudget:
        values: dict[str, int | None] = {}
        for field_name in _BUDGET_FIELDS:
            current = getattr(budget, field_name)
            if current is None:
                values[field_name] = None
                continue
            if mode == "shrink":
                adjusted = math.floor(current * self.policy.shrink_factor)
            elif mode == "grow":
                adjusted = math.ceil(current * self.policy.grow_factor)
            else:
                raise ValueError(f"unknown adjustment mode: {mode!r}")
            adjusted = max(1, adjusted)
            values[field_name] = self._clamp(field_name, adjusted)
        return BatchBudget(**values)

    def _clamp(self, budget_field: str, value: int) -> int:
        min_field, max_field = _BOUND_FIELDS[budget_field]
        minimum = getattr(self.policy, min_field)
        maximum = getattr(self.policy, max_field)
        if minimum is not None:
            value = max(minimum, value)
        if maximum is not None:
            value = min(maximum, value)
        return value

    @staticmethod
    def _resource_peaks(results: tuple[StepResult, ...]) -> dict[str, int | None]:
        peaks: dict[str, int | None] = {
            "peak_cpu_rss_bytes": None,
            "peak_cuda_allocated_bytes": None,
            "peak_cuda_reserved_bytes": None,
            "peak_cuda_max_allocated_bytes": None,
            "peak_cuda_max_reserved_bytes": None,
        }
        for result in results:
            for sample in result.resource_samples:
                if not isinstance(sample, ResourceSample):
                    raise TypeError("StepResult.resource_samples must contain ResourceSample objects.")
                peaks["peak_cpu_rss_bytes"] = _peak(
                    peaks["peak_cpu_rss_bytes"],
                    sample.cpu_rss_bytes,
                )
                peaks["peak_cuda_allocated_bytes"] = _peak(
                    peaks["peak_cuda_allocated_bytes"],
                    sample.cuda_allocated_bytes,
                )
                peaks["peak_cuda_reserved_bytes"] = _peak(
                    peaks["peak_cuda_reserved_bytes"],
                    sample.cuda_reserved_bytes,
                )
                peaks["peak_cuda_max_allocated_bytes"] = _peak(
                    peaks["peak_cuda_max_allocated_bytes"],
                    sample.cuda_max_allocated_bytes,
                )
                peaks["peak_cuda_max_reserved_bytes"] = _peak(
                    peaks["peak_cuda_max_reserved_bytes"],
                    sample.cuda_max_reserved_bytes,
                )
        return peaks

    @staticmethod
    def _append_peak_reason(reason: str, peaks: dict[str, int | None]) -> str:
        observed = {key: value for key, value in peaks.items() if value is not None}
        if not observed:
            return reason
        peak_text = ", ".join(f"{key}={value}" for key, value in observed.items())
        return f"{reason}; resource peaks: {peak_text}"
