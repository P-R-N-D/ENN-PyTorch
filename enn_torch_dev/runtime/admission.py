from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real

from .calibration import (
    ObservedCostMetricProfile,
    ObservedCostProfile,
    ObservedPhaseCostProfile,
)
from .faults import ResourceSample
from .pressure import ResourceCapacity


class PrePassAdmissionStatus(Enum):
    """Outcome for a pure pre-pass resource admission assessment."""

    ADMIT = "admit"
    REJECT = "reject"
    UNKNOWN = "unknown"


def _validate_ratio(value: object, *, label: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{label} must be a finite number.")
    ratio = float(value)
    if not math.isfinite(ratio):
        raise ValueError(f"{label} must be finite.")
    if ratio <= 0 or ratio > 1:
        raise ValueError(f"{label} must be greater than zero and at most one.")
    return ratio


def _validate_non_negative_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer.")
    if value < 0:
        raise ValueError(f"{label} must be non-negative.")
    return value


def _validate_positive_int(value: object, *, label: str) -> int:
    value = _validate_non_negative_int(value, label=label)
    if value == 0:
        raise ValueError(f"{label} must be positive.")
    return value


def _validate_optional_non_negative_int(
    value: object,
    *,
    label: str,
) -> int | None:
    if value is None:
        return None
    return _validate_non_negative_int(value, label=label)


def _is_concrete_cuda_device_index(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


@dataclass(frozen=True, slots=True)
class PrePassAdmissionPolicy:
    """Conservative capacity utilization and reserve policy for admission."""

    host_utilization_ratio: float = 0.9
    device_utilization_ratio: float = 0.9
    host_reserve_bytes: int = 0
    device_reserve_bytes: int = 0
    min_profile_samples: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "host_utilization_ratio",
            _validate_ratio(
                self.host_utilization_ratio,
                label="PrePassAdmissionPolicy.host_utilization_ratio",
            ),
        )
        object.__setattr__(
            self,
            "device_utilization_ratio",
            _validate_ratio(
                self.device_utilization_ratio,
                label="PrePassAdmissionPolicy.device_utilization_ratio",
            ),
        )
        object.__setattr__(
            self,
            "host_reserve_bytes",
            _validate_non_negative_int(
                self.host_reserve_bytes,
                label="PrePassAdmissionPolicy.host_reserve_bytes",
            ),
        )
        object.__setattr__(
            self,
            "device_reserve_bytes",
            _validate_non_negative_int(
                self.device_reserve_bytes,
                label="PrePassAdmissionPolicy.device_reserve_bytes",
            ),
        )
        object.__setattr__(
            self,
            "min_profile_samples",
            _validate_positive_int(
                self.min_profile_samples,
                label="PrePassAdmissionPolicy.min_profile_samples",
            ),
        )


class PrePassAdmissionError(RuntimeError):
    """Raised when admission inputs have incompatible resource provenance."""

    def __init__(
        self,
        reason: str,
        *,
        dimensions: tuple[str, ...] = (),
    ) -> None:
        self.reason = reason
        self.dimensions = dimensions
        suffix = f" dimensions={dimensions!r}" if dimensions else ""
        super().__init__(f"Pre-pass admission failed: {reason}.{suffix}")


@dataclass(frozen=True, slots=True)
class PrePassAdmissionDimension:
    """One deterministic resource-dimension admission calculation."""

    name: str
    status: PrePassAdmissionStatus
    applicable: bool
    capacity_bytes: int | None
    usable_bytes: int | None
    current_bytes: int | None
    incremental_bytes_per_item: int | None
    projected_bytes: int | None
    headroom_bytes: int | None
    item_limit: int | None
    item_limit_is_unbounded: bool
    reason: str


@dataclass(frozen=True, slots=True)
class PrePassAdmissionAssessment:
    """Pure structured assessment for one candidate batch size."""

    status: PrePassAdmissionStatus
    batch_size: int
    policy: PrePassAdmissionPolicy
    profile_successful_samples: int
    cuda_device_index: int | None
    dimensions: tuple[PrePassAdmissionDimension, ...]
    rejected_dimensions: tuple[str, ...]
    unknown_dimensions: tuple[str, ...]
    max_admissible_items: int | None
    warnings: tuple[str, ...]

    @property
    def admitted(self) -> bool:
        return self.status is PrePassAdmissionStatus.ADMIT


def _metric_max_bytes_per_item(
    metric: ObservedCostMetricProfile,
    *,
    label: str,
) -> int | None:
    if not isinstance(metric, ObservedCostMetricProfile):
        raise TypeError(f"{label} must be an ObservedCostMetricProfile.")
    return _validate_optional_non_negative_int(
        metric.max_bytes_per_item,
        label=f"{label}.max_bytes_per_item",
    )


def _max_known(*values: int | None) -> int | None:
    known = tuple(value for value in values if value is not None)
    return max(known) if known else None


def _phase_profiles_have_cuda_evidence(
    phase_costs: tuple[ObservedPhaseCostProfile, ...],
) -> bool:
    has_cuda_evidence = False
    for index, phase_profile in enumerate(phase_costs):
        label = f"ObservedCostProfile.phase_costs[{index}]"
        if not isinstance(phase_profile, ObservedPhaseCostProfile):
            raise TypeError(f"{label} must be an ObservedPhaseCostProfile.")
        for metric_name in (
            "cuda_allocated",
            "cuda_reserved",
            "cuda_max_allocated",
            "cuda_max_reserved",
        ):
            value = _metric_max_bytes_per_item(
                getattr(phase_profile, metric_name),
                label=f"{label}.{metric_name}",
            )
            has_cuda_evidence = has_cuda_evidence or value is not None
    return has_cuda_evidence


def _usable_bytes(capacity: int, ratio: float, reserve: int) -> int:
    return max(0, math.floor(capacity * ratio) - reserve)


def _assess_dimension(
    *,
    name: str,
    applicable: bool,
    capacity_bytes: int | None,
    utilization_ratio: float,
    reserve_bytes: int,
    current_bytes: int | None,
    incremental_bytes_per_item: int | None,
    batch_size: int,
    profile_ready: bool,
) -> PrePassAdmissionDimension:
    if not applicable:
        return PrePassAdmissionDimension(
            name=name,
            status=PrePassAdmissionStatus.ADMIT,
            applicable=False,
            capacity_bytes=None,
            usable_bytes=None,
            current_bytes=None,
            incremental_bytes_per_item=None,
            projected_bytes=None,
            headroom_bytes=None,
            item_limit=None,
            item_limit_is_unbounded=False,
            reason="not applicable",
        )

    usable = (
        None
        if capacity_bytes is None
        else _usable_bytes(capacity_bytes, utilization_ratio, reserve_bytes)
    )
    headroom = (
        None
        if usable is None or current_bytes is None
        else max(0, usable - current_bytes)
    )

    effective_increment = incremental_bytes_per_item if profile_ready else None
    projected = (
        None
        if current_bytes is None or effective_increment is None
        else current_bytes + effective_increment * batch_size
    )

    item_limit: int | None = None
    item_limit_is_unbounded = False
    if usable is not None and current_bytes is not None:
        if current_bytes > usable:
            item_limit = 0
        elif effective_increment == 0:
            item_limit_is_unbounded = True
        elif effective_increment is not None:
            item_limit = (usable - current_bytes) // effective_increment

    if usable is not None and current_bytes is not None and current_bytes > usable:
        status = PrePassAdmissionStatus.REJECT
        reason = "current usage exceeds usable capacity"
    elif projected is not None and usable is not None and projected > usable:
        status = PrePassAdmissionStatus.REJECT
        reason = "projected usage exceeds usable capacity"
    elif capacity_bytes is None:
        status = PrePassAdmissionStatus.UNKNOWN
        reason = "capacity is unknown"
    elif current_bytes is None:
        status = PrePassAdmissionStatus.UNKNOWN
        reason = "current usage is unknown"
    elif not profile_ready:
        status = PrePassAdmissionStatus.UNKNOWN
        reason = "observed profile sample floor is not met"
    elif incremental_bytes_per_item is None:
        status = PrePassAdmissionStatus.UNKNOWN
        reason = "incremental cost is unknown"
    else:
        status = PrePassAdmissionStatus.ADMIT
        reason = "projected usage is within usable capacity"

    return PrePassAdmissionDimension(
        name=name,
        status=status,
        applicable=True,
        capacity_bytes=capacity_bytes,
        usable_bytes=usable,
        current_bytes=current_bytes,
        incremental_bytes_per_item=effective_increment,
        projected_bytes=projected,
        headroom_bytes=headroom,
        item_limit=item_limit,
        item_limit_is_unbounded=item_limit_is_unbounded,
        reason=reason,
    )


def assess_prepass_admission(
    capacity: ResourceCapacity,
    baseline_sample: ResourceSample,
    observed_profile: ObservedCostProfile,
    *,
    batch_size: int,
    policy: PrePassAdmissionPolicy | None = None,
) -> PrePassAdmissionAssessment:
    """Assess one candidate without execution, source consumption, or mutation."""

    if not isinstance(capacity, ResourceCapacity):
        raise TypeError("capacity must be a ResourceCapacity.")
    if not isinstance(baseline_sample, ResourceSample):
        raise TypeError("baseline_sample must be a ResourceSample.")
    if not isinstance(observed_profile, ObservedCostProfile):
        raise TypeError("observed_profile must be an ObservedCostProfile.")
    batch_size = _validate_positive_int(batch_size, label="batch_size")
    if policy is None:
        policy = PrePassAdmissionPolicy()
    elif not isinstance(policy, PrePassAdmissionPolicy):
        raise TypeError("policy must be a PrePassAdmissionPolicy or None.")

    successful_samples = _validate_non_negative_int(
        observed_profile.successful_samples,
        label="ObservedCostProfile.successful_samples",
    )
    profile_ready = successful_samples >= policy.min_profile_samples

    cpu_current = _validate_optional_non_negative_int(
        baseline_sample.cpu_rss_bytes,
        label="ResourceSample.cpu_rss_bytes",
    )
    cuda_allocated_current = _validate_optional_non_negative_int(
        baseline_sample.cuda_allocated_bytes,
        label="ResourceSample.cuda_allocated_bytes",
    )
    cuda_reserved_current = _validate_optional_non_negative_int(
        baseline_sample.cuda_reserved_bytes,
        label="ResourceSample.cuda_reserved_bytes",
    )
    cuda_max_allocated_current = _validate_optional_non_negative_int(
        baseline_sample.cuda_max_allocated_bytes,
        label="ResourceSample.cuda_max_allocated_bytes",
    )
    cuda_max_reserved_current = _validate_optional_non_negative_int(
        baseline_sample.cuda_max_reserved_bytes,
        label="ResourceSample.cuda_max_reserved_bytes",
    )

    cpu_increment = _metric_max_bytes_per_item(
        observed_profile.cpu_rss,
        label="ObservedCostProfile.cpu_rss",
    )
    cuda_allocated_increment = _max_known(
        _metric_max_bytes_per_item(
            observed_profile.cuda_allocated,
            label="ObservedCostProfile.cuda_allocated",
        ),
        _metric_max_bytes_per_item(
            observed_profile.cuda_max_allocated,
            label="ObservedCostProfile.cuda_max_allocated",
        ),
    )
    cuda_reserved_increment = _max_known(
        _metric_max_bytes_per_item(
            observed_profile.cuda_reserved,
            label="ObservedCostProfile.cuda_reserved",
        ),
        _metric_max_bytes_per_item(
            observed_profile.cuda_max_reserved,
            label="ObservedCostProfile.cuda_max_reserved",
        ),
    )

    sample_has_cuda_values = any(
        value is not None
        for value in (
            cuda_allocated_current,
            cuda_reserved_current,
            cuda_max_allocated_current,
            cuda_max_reserved_current,
        )
    )
    profile_has_total_cuda_metrics = any(
        value is not None
        for value in (
            cuda_allocated_increment,
            cuda_reserved_increment,
        )
    )
    profile_has_phase_cuda_metrics = _phase_profiles_have_cuda_evidence(
        observed_profile.phase_costs
    )
    profile_has_cuda_evidence = (
        profile_has_total_cuda_metrics or profile_has_phase_cuda_metrics
    )
    cuda_applicable = sample_has_cuda_values or profile_has_cuda_evidence

    cuda_device_index: int | None = None
    if cuda_applicable:
        if capacity.cuda_total_bytes is None or capacity.cuda_device_index is None:
            raise PrePassAdmissionError(
                "CUDA admission inputs require CUDA capacity",
                dimensions=("cuda",),
            )
        cuda_device_index = capacity.cuda_device_index

        if profile_has_cuda_evidence:
            profile_index = observed_profile.cuda_device_index
            if not _is_concrete_cuda_device_index(profile_index):
                raise PrePassAdmissionError(
                    "known CUDA profile metrics require concrete device provenance",
                    dimensions=("cuda",),
                )
            if profile_index != cuda_device_index:
                raise PrePassAdmissionError(
                    "observed profile CUDA device does not match capacity",
                    dimensions=(f"cuda:{profile_index}", f"cuda:{cuda_device_index}"),
                )

        if sample_has_cuda_values:
            sample_index = baseline_sample.cuda_device_index
            if not _is_concrete_cuda_device_index(sample_index):
                raise PrePassAdmissionError(
                    "baseline CUDA values require concrete device provenance",
                    dimensions=("cuda",),
                )
            if sample_index != cuda_device_index:
                raise PrePassAdmissionError(
                    "baseline sample CUDA device does not match capacity",
                    dimensions=(f"cuda:{sample_index}", f"cuda:{cuda_device_index}"),
                )

    dimensions = (
        _assess_dimension(
            name="cpu_rss",
            applicable=True,
            capacity_bytes=capacity.effective_cpu_bytes,
            utilization_ratio=policy.host_utilization_ratio,
            reserve_bytes=policy.host_reserve_bytes,
            current_bytes=cpu_current,
            incremental_bytes_per_item=cpu_increment,
            batch_size=batch_size,
            profile_ready=profile_ready,
        ),
        _assess_dimension(
            name="cuda_allocated",
            applicable=cuda_applicable,
            capacity_bytes=capacity.cuda_total_bytes if cuda_applicable else None,
            utilization_ratio=policy.device_utilization_ratio,
            reserve_bytes=policy.device_reserve_bytes,
            current_bytes=cuda_allocated_current,
            incremental_bytes_per_item=cuda_allocated_increment,
            batch_size=batch_size,
            profile_ready=profile_ready,
        ),
        _assess_dimension(
            name="cuda_reserved",
            applicable=cuda_applicable,
            capacity_bytes=capacity.cuda_total_bytes if cuda_applicable else None,
            utilization_ratio=policy.device_utilization_ratio,
            reserve_bytes=policy.device_reserve_bytes,
            current_bytes=cuda_reserved_current,
            incremental_bytes_per_item=cuda_reserved_increment,
            batch_size=batch_size,
            profile_ready=profile_ready,
        ),
    )

    rejected_dimensions = tuple(
        dimension.name
        for dimension in dimensions
        if dimension.status is PrePassAdmissionStatus.REJECT
    )
    unknown_dimensions = tuple(
        dimension.name
        for dimension in dimensions
        if dimension.status is PrePassAdmissionStatus.UNKNOWN
    )
    if rejected_dimensions:
        status = PrePassAdmissionStatus.REJECT
    elif unknown_dimensions:
        status = PrePassAdmissionStatus.UNKNOWN
    else:
        status = PrePassAdmissionStatus.ADMIT

    finite_item_limits = tuple(
        dimension.item_limit
        for dimension in dimensions
        if dimension.applicable and dimension.item_limit is not None
    )
    max_admissible_items = min(finite_item_limits) if finite_item_limits else None

    warnings: list[str] = []
    if not profile_ready:
        warnings.append(
            "observed profile successful sample count "
            f"{successful_samples} is below required {policy.min_profile_samples}"
        )
    warnings.extend(
        f"{dimension.name}: {dimension.reason}"
        for dimension in dimensions
        if dimension.status is PrePassAdmissionStatus.UNKNOWN
    )

    return PrePassAdmissionAssessment(
        status=status,
        batch_size=batch_size,
        policy=policy,
        profile_successful_samples=successful_samples,
        cuda_device_index=cuda_device_index,
        dimensions=dimensions,
        rejected_dimensions=rejected_dimensions,
        unknown_dimensions=unknown_dimensions,
        max_admissible_items=max_admissible_items,
        warnings=tuple(warnings),
    )
