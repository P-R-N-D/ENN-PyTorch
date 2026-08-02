from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

from .batching import BatchBudget
from .faults import ResourceSample, StepResult, StepStatus
from .pressure import ResourcePressureSummary


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


def _validate_optional_shrink_factor(
    value: object,
    *,
    label: str,
) -> float | None:
    if value is None:
        return None
    return _validate_factor(value, label=label, lower=0.0, upper=1.0)


def _validate_optional_pressure_ratio(
    value: object,
    *,
    label: str,
) -> float | None:
    if value is None:
        return None
    full_label = f"GovernorPolicy.{label}"
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{full_label} must be a finite number or None.")
    ratio = float(value)
    if not math.isfinite(ratio):
        raise ValueError(f"{full_label} must be finite.")
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"{full_label} must satisfy 0 < value <= 1.")
    return ratio


def _validate_budget(budget: object, *, label: str) -> BatchBudget:
    if not isinstance(budget, BatchBudget):
        raise TypeError(f"{label} must be a BatchBudget.")
    for field_name in _BUDGET_FIELDS:
        value = getattr(budget, field_name)
        if value is not None and value <= 0:
            raise ValueError(f"{label}.{field_name} must be positive when configured.")
    return budget


def _validate_budget_within_policy_bounds(
    budget: BatchBudget,
    policy: GovernorPolicy,
    *,
    label: str,
) -> None:
    for field_name, (min_field, max_field) in _BOUND_FIELDS.items():
        value = getattr(budget, field_name)
        if value is None:
            continue
        minimum = getattr(policy, min_field)
        maximum = getattr(policy, max_field)
        if minimum is not None and value < minimum:
            raise ValueError(
                f"{label}.{field_name} must be >= GovernorPolicy.{min_field}."
            )
        if maximum is not None and value > maximum:
            raise ValueError(
                f"{label}.{field_name} must be <= GovernorPolicy.{max_field}."
            )


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
    max_pressure_ratio_for_growth: float | None = None
    min_pressure_ratio_for_shrink: float | None = None
    shrink_after_pressure_passes: int = 2
    min_cpu_pressure_ratio_for_shrink: float | None = None
    min_cuda_pressure_ratio_for_shrink: float | None = None
    cpu_shrink_after_pressure_passes: int | None = None
    cuda_shrink_after_pressure_passes: int | None = None
    cpu_pressure_shrink_factor: float | None = None
    cuda_pressure_shrink_factor: float | None = None
    suppress_growth_after_admission_recovery: bool = False

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
        object.__setattr__(
            self,
            "max_pressure_ratio_for_growth",
            _validate_optional_pressure_ratio(
                self.max_pressure_ratio_for_growth,
                label="max_pressure_ratio_for_growth",
            ),
        )
        object.__setattr__(
            self,
            "min_pressure_ratio_for_shrink",
            _validate_optional_pressure_ratio(
                self.min_pressure_ratio_for_shrink,
                label="min_pressure_ratio_for_shrink",
            ),
        )
        object.__setattr__(
            self,
            "shrink_after_pressure_passes",
            _validate_positive_int(
                self.shrink_after_pressure_passes,
                label="GovernorPolicy.shrink_after_pressure_passes",
            ),
        )
        for field_name in (
            "min_cpu_pressure_ratio_for_shrink",
            "min_cuda_pressure_ratio_for_shrink",
        ):
            object.__setattr__(
                self,
                field_name,
                _validate_optional_pressure_ratio(
                    getattr(self, field_name),
                    label=field_name,
                ),
            )
        for field_name in (
            "cpu_shrink_after_pressure_passes",
            "cuda_shrink_after_pressure_passes",
        ):
            object.__setattr__(
                self,
                field_name,
                _validate_optional_positive_int(
                    getattr(self, field_name),
                    label=f"GovernorPolicy.{field_name}",
                ),
            )
        for field_name in (
            "cpu_pressure_shrink_factor",
            "cuda_pressure_shrink_factor",
        ):
            object.__setattr__(
                self,
                field_name,
                _validate_optional_shrink_factor(
                    getattr(self, field_name),
                    label=field_name,
                ),
            )
        growth_limit = self.max_pressure_ratio_for_growth
        effective_shrink_limits = (
            (
                "CPU",
                self.min_cpu_pressure_ratio_for_shrink
                if self.min_cpu_pressure_ratio_for_shrink is not None
                else self.min_pressure_ratio_for_shrink,
            ),
            (
                "CUDA",
                self.min_cuda_pressure_ratio_for_shrink
                if self.min_cuda_pressure_ratio_for_shrink is not None
                else self.min_pressure_ratio_for_shrink,
            ),
        )
        for dimension, shrink_limit in effective_shrink_limits:
            if (
                growth_limit is not None
                and shrink_limit is not None
                and growth_limit > shrink_limit
            ):
                raise ValueError(
                    "GovernorPolicy.max_pressure_ratio_for_growth must be <= "
                    f"the effective {dimension} shrink threshold."
                )
        self._validate_bounds("items", self.min_items, self.max_items)
        self._validate_bounds("host_bytes", self.min_host_bytes, self.max_host_bytes)
        self._validate_bounds("device_bytes", self.min_device_bytes, self.max_device_bytes)
        if not isinstance(self.suppress_growth_after_admission_recovery, bool):
            raise TypeError(
                "GovernorPolicy.suppress_growth_after_admission_recovery must be a bool."
            )

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
    pressure_summary: ResourcePressureSummary | None = None
    growth_suppressed_by_pressure: bool = False
    consecutive_high_pressure_passes: int = 0
    budget_shrunk_by_pressure: bool = False
    pressure_shrunk_budget_fields: tuple[str, ...] = ()
    consecutive_cpu_pressure_passes: int = 0
    consecutive_cuda_pressure_passes: int = 0
    pressure_high_dimensions: tuple[str, ...] = ()
    pressure_triggered_dimensions: tuple[str, ...] = ()
    pressure_selected_budget_fields: tuple[str, ...] = ()
    pressure_applied_shrink_factors: tuple[tuple[str, float], ...] = ()
    admission_recovery_max_items: int | None = None
    growth_suppressed_by_admission_recovery: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeGovernorState:
    """Reusable immutable state for ConservativeRuntimeGovernor."""

    current_budget: BatchBudget
    consecutive_successes: int = 0
    consecutive_ooms: int = 0
    last_decision: GovernorDecision | None = None
    consecutive_high_pressure_passes: int = 0
    consecutive_cpu_pressure_passes: int = 0
    consecutive_cuda_pressure_passes: int = 0

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
        object.__setattr__(
            self,
            "consecutive_high_pressure_passes",
            _validate_streak(
                self.consecutive_high_pressure_passes,
                label="RuntimeGovernorState.consecutive_high_pressure_passes",
            ),
        )
        object.__setattr__(
            self,
            "consecutive_cpu_pressure_passes",
            _validate_streak(
                self.consecutive_cpu_pressure_passes,
                label="RuntimeGovernorState.consecutive_cpu_pressure_passes",
            ),
        )
        object.__setattr__(
            self,
            "consecutive_cuda_pressure_passes",
            _validate_streak(
                self.consecutive_cuda_pressure_passes,
                label="RuntimeGovernorState.consecutive_cuda_pressure_passes",
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

        resolved_policy = GovernorPolicy() if policy is None else policy
        resolved_state = (
            state if state is not None else RuntimeGovernorState(budget)  # type: ignore[arg-type]
        )
        _validate_budget_within_policy_bounds(
            resolved_state.current_budget,
            resolved_policy,
            label="ConservativeRuntimeGovernor.current_budget",
        )

        self.policy = resolved_policy
        self.state = resolved_state

    @property
    def current_budget(self) -> BatchBudget:
        return self.state.current_budget

    def observe_results(
        self,
        results: Iterable[StepResult],
        *,
        recovered_oom: bool = False,
        pressure_summary: ResourcePressureSummary | None = None,
        admission_recovery_max_items: int | None = None,
    ) -> GovernorDecision:
        if isinstance(results, StepResult):
            raise TypeError("ConservativeRuntimeGovernor.observe_results expects an iterable of StepResult objects.")
        if not isinstance(recovered_oom, bool):
            raise TypeError("ConservativeRuntimeGovernor.recovered_oom must be a bool.")
        if pressure_summary is not None and not isinstance(
            pressure_summary, ResourcePressureSummary
        ):
            raise TypeError(
                "ConservativeRuntimeGovernor.pressure_summary must be a "
                "ResourcePressureSummary or None."
            )
        admission_recovery_max_items = _validate_optional_positive_int(
            admission_recovery_max_items,
            label="ConservativeRuntimeGovernor.admission_recovery_max_items",
        )

        statuses: list[StepStatus] = []
        peaks = self._empty_resource_peaks()
        saw_any_result = False
        saw_oom = False
        all_success = True

        for result in results:
            if not isinstance(result, StepResult):
                raise TypeError("ConservativeRuntimeGovernor.observe_results must receive StepResult objects.")
            saw_any_result = True
            status = result.status
            statuses.append(status)
            if status is StepStatus.OOM_FAULT:
                saw_oom = True
            if status is not StepStatus.SUCCESS:
                all_success = False
            self._update_resource_peaks(peaks, result.resource_samples)
        if saw_any_result:
            del result

        previous_budget = self.state.current_budget
        next_budget = previous_budget
        consecutive_successes = self.state.consecutive_successes
        consecutive_ooms = self.state.consecutive_ooms
        consecutive_cpu_pressure_passes = self.state.consecutive_cpu_pressure_passes
        consecutive_cuda_pressure_passes = self.state.consecutive_cuda_pressure_passes
        legacy_is_available = (
            self.state.consecutive_high_pressure_passes > 0
            and consecutive_cpu_pressure_passes == 0
            and consecutive_cuda_pressure_passes == 0
        )
        growth_suppressed_by_pressure = False
        budget_shrunk_by_pressure = False
        pressure_shrunk_budget_fields: tuple[str, ...] = ()
        pressure_high_dimensions: tuple[str, ...] = ()
        pressure_triggered_dimensions: tuple[str, ...] = ()
        pressure_selected_budget_fields: tuple[str, ...] = ()
        pressure_applied_shrink_factors: tuple[tuple[str, float], ...] = ()
        growth_suppressed_by_admission_recovery = False

        if saw_oom or recovered_oom:
            next_budget = self._adjust_budget(previous_budget, mode="shrink")
            consecutive_successes = 0
            consecutive_ooms += 1
            consecutive_cpu_pressure_passes = 0
            consecutive_cuda_pressure_passes = 0
            if saw_oom:
                reason = "OOM fault observed; shrinking configured budget fields"
                if recovered_oom:
                    reason += "; retry-recovered OOM signal also observed"
            else:
                reason = "retry-recovered OOM observed; shrinking configured budget fields"
        elif not saw_any_result:
            consecutive_cpu_pressure_passes = 0
            consecutive_cuda_pressure_passes = 0
            reason = "no results observed; keeping current budget"
        elif all_success:
            consecutive_ooms = 0
            cpu_shrink_limit = (
                self.policy.min_cpu_pressure_ratio_for_shrink
                if self.policy.min_cpu_pressure_ratio_for_shrink is not None
                else self.policy.min_pressure_ratio_for_shrink
            )
            cuda_shrink_limit = (
                self.policy.min_cuda_pressure_ratio_for_shrink
                if self.policy.min_cuda_pressure_ratio_for_shrink is not None
                else self.policy.min_pressure_ratio_for_shrink
            )
            cpu_required = (
                self.policy.cpu_shrink_after_pressure_passes
                if self.policy.cpu_shrink_after_pressure_passes is not None
                else self.policy.shrink_after_pressure_passes
            )
            cuda_required = (
                self.policy.cuda_shrink_after_pressure_passes
                if self.policy.cuda_shrink_after_pressure_passes is not None
                else self.policy.shrink_after_pressure_passes
            )
            cpu_pressure_shrink_factor = (
                self.policy.cpu_pressure_shrink_factor
                if self.policy.cpu_pressure_shrink_factor is not None
                else self.policy.shrink_factor
            )
            cuda_pressure_shrink_factor = (
                self.policy.cuda_pressure_shrink_factor
                if self.policy.cuda_pressure_shrink_factor is not None
                else self.policy.shrink_factor
            )
            cpu_pressure_high = False
            cuda_pressure_high = False
            if pressure_summary is not None:
                (
                    cpu_pressure_high,
                    cuda_pressure_high,
                ) = self._pressure_dimensions_at_or_above(
                    pressure_summary,
                    cpu_threshold=cpu_shrink_limit,
                    cuda_threshold=cuda_shrink_limit,
                )
            pressure_high_dimensions = tuple(
                dimension
                for dimension, pressure_high in (
                    ("cpu", cpu_pressure_high),
                    ("cuda", cuda_pressure_high),
                )
                if pressure_high
            )
            legacy_high_pressure_streak = (
                self.state.consecutive_high_pressure_passes
                if legacy_is_available and cpu_pressure_high != cuda_pressure_high
                else 0
            )
            if cpu_pressure_high or cuda_pressure_high:
                consecutive_successes = 0
                growth_suppressed_by_pressure = True
                if cpu_pressure_high:
                    cpu_base = (
                        legacy_high_pressure_streak
                        if legacy_high_pressure_streak
                        else consecutive_cpu_pressure_passes
                    )
                    consecutive_cpu_pressure_passes = cpu_base + 1
                else:
                    consecutive_cpu_pressure_passes = 0
                if cuda_pressure_high:
                    cuda_base = (
                        legacy_high_pressure_streak
                        if legacy_high_pressure_streak
                        else consecutive_cuda_pressure_passes
                    )
                    consecutive_cuda_pressure_passes = cuda_base + 1
                else:
                    consecutive_cuda_pressure_passes = 0

                cpu_pressure_triggered = (
                    cpu_pressure_high
                    and consecutive_cpu_pressure_passes >= cpu_required
                )
                cuda_pressure_triggered = (
                    cuda_pressure_high
                    and consecutive_cuda_pressure_passes >= cuda_required
                )
                if cpu_pressure_triggered or cuda_pressure_triggered:
                    (
                        next_budget,
                        pressure_selected_budget_fields,
                        pressure_applied_shrink_factors,
                        pressure_shrunk_budget_fields,
                    ) = self._adjust_budget_for_pressure(
                        previous_budget,
                        cpu_pressure=cpu_pressure_triggered,
                        cuda_pressure=cuda_pressure_triggered,
                        cpu_shrink_factor=cpu_pressure_shrink_factor,
                        cuda_shrink_factor=cuda_pressure_shrink_factor,
                    )
                    budget_shrunk_by_pressure = bool(pressure_shrunk_budget_fields)
                    pressure_triggered_dimensions = tuple(
                        dimension
                        for dimension, triggered in (
                            ("cpu", cpu_pressure_triggered),
                            ("cuda", cuda_pressure_triggered),
                        )
                        if triggered
                    )
                    if cpu_pressure_triggered:
                        consecutive_cpu_pressure_passes = 0
                    if cuda_pressure_triggered:
                        consecutive_cuda_pressure_passes = 0
                    dimension_text = ", ".join(pressure_triggered_dimensions)
                    triggered_ratios: list[str] = []
                    triggered_policies: list[str] = []
                    triggered_factors: list[str] = []
                    if cpu_pressure_triggered:
                        cpu_ratio = pressure_summary.peak_cpu_rss_ratio
                        assert cpu_ratio is not None
                        assert cpu_shrink_limit is not None
                        triggered_ratios.append(f"cpu={cpu_ratio:.6g}")
                        triggered_policies.append(
                            f"cpu(limit={cpu_shrink_limit:.6g}, required={cpu_required})"
                        )
                        triggered_factors.append(f"cpu={cpu_pressure_shrink_factor:.6g}")
                    if cuda_pressure_triggered:
                        cuda_ratio = self._peak_cuda_pressure_ratio(pressure_summary)
                        assert cuda_ratio is not None
                        assert cuda_shrink_limit is not None
                        triggered_ratios.append(f"cuda={cuda_ratio:.6g}")
                        triggered_policies.append(
                            f"cuda(limit={cuda_shrink_limit:.6g}, required={cuda_required})"
                        )
                        triggered_factors.append(f"cuda={cuda_pressure_shrink_factor:.6g}")
                    ratio_text = ", ".join(triggered_ratios)
                    policy_text = ", ".join(triggered_policies)
                    factor_text = ", ".join(triggered_factors)
                    reason_prefix = (
                        "pressure streak threshold reached; "
                        f"triggered dimensions: {dimension_text}; "
                        f"triggered policies: {policy_text}; "
                        f"triggered shrink factors: {factor_text}; "
                        f"current triggered ratios: {ratio_text}; "
                    )
                    if budget_shrunk_by_pressure:
                        field_text = ", ".join(pressure_shrunk_budget_fields)
                        reason = (
                            reason_prefix
                            + f"shrinking pressure-matched budget fields: {field_text}"
                        )
                    elif pressure_selected_budget_fields:
                        field_text = ", ".join(pressure_selected_budget_fields)
                        reason = (
                            reason_prefix
                            + "configured minimum bounds kept pressure-matched budget "
                            f"fields unchanged: {field_text}"
                        )
                    else:
                        reason = (
                            reason_prefix
                            + "no matching byte budget or max_items fallback is configured"
                        )
                else:
                    progress_parts: list[str] = []
                    if cpu_shrink_limit is not None:
                        cpu_ratio = pressure_summary.peak_cpu_rss_ratio
                        cpu_ratio_text = (
                            "unknown" if cpu_ratio is None else f"{cpu_ratio:.6g}"
                        )
                        progress_parts.append(
                            "cpu="
                            f"{consecutive_cpu_pressure_passes}/{cpu_required} "
                            f"(limit={cpu_shrink_limit:.6g}, ratio={cpu_ratio_text})"
                        )
                    if cuda_shrink_limit is not None:
                        cuda_ratio = self._peak_cuda_pressure_ratio(pressure_summary)
                        cuda_ratio_text = (
                            "unknown" if cuda_ratio is None else f"{cuda_ratio:.6g}"
                        )
                        progress_parts.append(
                            "cuda="
                            f"{consecutive_cuda_pressure_passes}/{cuda_required} "
                            f"(limit={cuda_shrink_limit:.6g}, ratio={cuda_ratio_text})"
                        )
                    reason = (
                        "resource pressure reached configured shrink limit for dimensions: "
                        f"{', '.join(pressure_high_dimensions)}; "
                        f"pressure streaks {', '.join(progress_parts)}; "
                        "suppressing budget growth"
                    )
            else:
                consecutive_cpu_pressure_passes = 0
                consecutive_cuda_pressure_passes = 0
                pressure_reason = self._growth_pressure_guard_reason(pressure_summary)
                if pressure_reason is not None:
                    consecutive_successes = 0
                    growth_suppressed_by_pressure = True
                    reason = pressure_reason
                else:
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
            consecutive_cpu_pressure_passes = 0
            consecutive_cuda_pressure_passes = 0
            reason = "non-OOM fault observed; keeping current budget"

        if (
            saw_any_result
            and all_success
            and not saw_oom
            and not recovered_oom
            and self.policy.suppress_growth_after_admission_recovery
            and admission_recovery_max_items is not None
        ):
            growth_suppressed_by_admission_recovery = True
            consecutive_successes = 0
            admission_reason = (
                "success observed after admission recovery; "
                "suppressing success-streak growth; "
                f"recovered max-items limit={admission_recovery_max_items}"
            )
            if next_budget != previous_budget and not budget_shrunk_by_pressure:
                next_budget = previous_budget
                reason = admission_reason
            else:
                reason = f"{reason}; {admission_reason}"

        consecutive_high_pressure_passes = max(
            consecutive_cpu_pressure_passes,
            consecutive_cuda_pressure_passes,
        )
        reason = self._append_peak_reason(reason, peaks)
        decision = GovernorDecision(
            previous_budget=previous_budget,
            next_budget=next_budget,
            reason=reason,
            statuses=tuple(statuses),
            consecutive_successes=consecutive_successes,
            consecutive_ooms=consecutive_ooms,
            peak_cpu_rss_bytes=peaks["peak_cpu_rss_bytes"],
            peak_cuda_allocated_bytes=peaks["peak_cuda_allocated_bytes"],
            peak_cuda_reserved_bytes=peaks["peak_cuda_reserved_bytes"],
            peak_cuda_max_allocated_bytes=peaks["peak_cuda_max_allocated_bytes"],
            peak_cuda_max_reserved_bytes=peaks["peak_cuda_max_reserved_bytes"],
            pressure_summary=pressure_summary,
            growth_suppressed_by_pressure=growth_suppressed_by_pressure,
            consecutive_high_pressure_passes=consecutive_high_pressure_passes,
            budget_shrunk_by_pressure=budget_shrunk_by_pressure,
            pressure_shrunk_budget_fields=pressure_shrunk_budget_fields,
            consecutive_cpu_pressure_passes=consecutive_cpu_pressure_passes,
            consecutive_cuda_pressure_passes=consecutive_cuda_pressure_passes,
            pressure_high_dimensions=pressure_high_dimensions,
            pressure_triggered_dimensions=pressure_triggered_dimensions,
            pressure_selected_budget_fields=pressure_selected_budget_fields,
            pressure_applied_shrink_factors=pressure_applied_shrink_factors,
            admission_recovery_max_items=admission_recovery_max_items,
            growth_suppressed_by_admission_recovery=(
                growth_suppressed_by_admission_recovery
            ),
        )
        self.state = RuntimeGovernorState(
            current_budget=next_budget,
            consecutive_successes=consecutive_successes,
            consecutive_ooms=consecutive_ooms,
            last_decision=decision,
            consecutive_high_pressure_passes=consecutive_high_pressure_passes,
            consecutive_cpu_pressure_passes=consecutive_cpu_pressure_passes,
            consecutive_cuda_pressure_passes=consecutive_cuda_pressure_passes,
        )
        return decision

    def _growth_pressure_guard_reason(
        self,
        pressure_summary: ResourcePressureSummary | None,
    ) -> str | None:
        threshold = self.policy.max_pressure_ratio_for_growth
        if threshold is None:
            return None
        if pressure_summary is None or pressure_summary.max_observed_ratio is None:
            return (
                "success observed but resource pressure is unavailable; "
                "suppressing budget growth"
            )
        max_ratio = pressure_summary.max_observed_ratio
        assert max_ratio is not None
        if max_ratio >= threshold:
            return (
                f"resource pressure {max_ratio:.6g} reached growth limit "
                f"{threshold:.6g}; suppressing budget growth"
            )
        return None

    @staticmethod
    def _pressure_dimensions_at_or_above(
        pressure_summary: ResourcePressureSummary,
        *,
        cpu_threshold: float | None,
        cuda_threshold: float | None,
    ) -> tuple[bool, bool]:
        cpu_pressure = (
            cpu_threshold is not None
            and pressure_summary.peak_cpu_rss_ratio is not None
            and pressure_summary.peak_cpu_rss_ratio >= cpu_threshold
        )
        cuda_pressure = (
            cuda_threshold is not None
            and any(
                value is not None and value >= cuda_threshold
                for value in (
                    pressure_summary.peak_cuda_allocated_ratio,
                    pressure_summary.peak_cuda_reserved_ratio,
                    pressure_summary.peak_cuda_max_allocated_ratio,
                    pressure_summary.peak_cuda_max_reserved_ratio,
                )
            )
        )
        return cpu_pressure, cuda_pressure

    @staticmethod
    def _peak_cuda_pressure_ratio(
        pressure_summary: ResourcePressureSummary,
    ) -> float | None:
        known_ratios = tuple(
            value
            for value in (
                pressure_summary.peak_cuda_allocated_ratio,
                pressure_summary.peak_cuda_reserved_ratio,
                pressure_summary.peak_cuda_max_allocated_ratio,
                pressure_summary.peak_cuda_max_reserved_ratio,
            )
            if value is not None
        )
        return max(known_ratios) if known_ratios else None

    def _adjust_budget_for_pressure(
        self,
        budget: BatchBudget,
        *,
        cpu_pressure: bool,
        cuda_pressure: bool,
        cpu_shrink_factor: float,
        cuda_shrink_factor: float,
    ) -> tuple[
        BatchBudget,
        tuple[str, ...],
        tuple[tuple[str, float], ...],
        tuple[str, ...],
    ]:
        selected_fields = self._pressure_budget_fields(
            budget,
            cpu_pressure=cpu_pressure,
            cuda_pressure=cuda_pressure,
        )
        values: dict[str, int | None] = {
            field_name: getattr(budget, field_name)
            for field_name in _BUDGET_FIELDS
        }
        changed_fields: list[str] = []
        applied_factors: list[tuple[str, float]] = []
        for field_name in selected_fields:
            current = values[field_name]
            assert current is not None
            if field_name == "max_host_bytes":
                factor = cpu_shrink_factor
            elif field_name == "max_device_bytes":
                factor = cuda_shrink_factor
            else:
                factor = min(
                    factor
                    for factor, triggered in (
                        (cpu_shrink_factor, cpu_pressure),
                        (cuda_shrink_factor, cuda_pressure),
                    )
                    if triggered
                )
            applied_factors.append((field_name, factor))
            adjusted = math.floor(current * factor)
            adjusted = max(1, adjusted)
            adjusted = self._clamp(field_name, adjusted)
            values[field_name] = adjusted
            if adjusted != current:
                changed_fields.append(field_name)
        return (
            BatchBudget(**values),
            selected_fields,
            tuple(applied_factors),
            tuple(changed_fields),
        )

    @staticmethod
    def _pressure_budget_fields(
        budget: BatchBudget,
        *,
        cpu_pressure: bool,
        cuda_pressure: bool,
    ) -> tuple[str, ...]:
        selected_fields: list[str] = []
        if cpu_pressure and budget.max_host_bytes is not None:
            selected_fields.append("max_host_bytes")
        if cuda_pressure and budget.max_device_bytes is not None:
            selected_fields.append("max_device_bytes")
        if (
            not selected_fields
            and (cpu_pressure or cuda_pressure)
            and budget.max_items is not None
        ):
            selected_fields.append("max_items")
        return tuple(selected_fields)

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
    def _empty_resource_peaks() -> dict[str, int | None]:
        return {
            "peak_cpu_rss_bytes": None,
            "peak_cuda_allocated_bytes": None,
            "peak_cuda_reserved_bytes": None,
            "peak_cuda_max_allocated_bytes": None,
            "peak_cuda_max_reserved_bytes": None,
        }

    @staticmethod
    def _update_resource_peaks(
        peaks: dict[str, int | None],
        samples: Iterable[ResourceSample],
    ) -> None:
        for sample in samples:
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

    @staticmethod
    def _append_peak_reason(reason: str, peaks: dict[str, int | None]) -> str:
        observed = {key: value for key, value in peaks.items() if value is not None}
        if not observed:
            return reason
        peak_text = ", ".join(f"{key}={value}" for key, value in observed.items())
        return f"{reason}; resource peaks: {peak_text}"
