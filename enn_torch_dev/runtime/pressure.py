from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

from .faults import ResourceSample


def _validate_optional_positive_int(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer or None.")
    if value <= 0:
        raise ValueError(f"{label} must be positive when configured.")
    return value


def _validate_optional_non_negative_int(
    value: object,
    *,
    label: str,
) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer or None.")
    if value < 0:
        raise ValueError(f"{label} must be non-negative when configured.")
    return value


def _validate_optional_ratio(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{label} must be a finite number or None.")
    ratio = float(value)
    if not math.isfinite(ratio):
        raise ValueError(f"{label} must be finite.")
    if ratio < 0:
        raise ValueError(f"{label} must be non-negative.")
    return ratio


@dataclass(frozen=True, slots=True)
class ResourceCapacity:
    """Total CPU/CUDA memory capacity used to normalize resource samples."""

    cpu_total_bytes: int | None = None
    cuda_total_bytes: int | None = None
    cuda_device_index: int | None = None
    cpu_limit_bytes: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cpu_total_bytes",
            _validate_optional_positive_int(
                self.cpu_total_bytes,
                label="ResourceCapacity.cpu_total_bytes",
            ),
        )
        object.__setattr__(
            self,
            "cuda_total_bytes",
            _validate_optional_positive_int(
                self.cuda_total_bytes,
                label="ResourceCapacity.cuda_total_bytes",
            ),
        )
        object.__setattr__(
            self,
            "cuda_device_index",
            _validate_optional_non_negative_int(
                self.cuda_device_index,
                label="ResourceCapacity.cuda_device_index",
            ),
        )
        object.__setattr__(
            self,
            "cpu_limit_bytes",
            _validate_optional_positive_int(
                self.cpu_limit_bytes,
                label="ResourceCapacity.cpu_limit_bytes",
            ),
        )
        if (self.cuda_total_bytes is None) != (self.cuda_device_index is None):
            raise ValueError(
                "ResourceCapacity.cuda_total_bytes and cuda_device_index must "
                "either both be configured or both be None."
            )

    @property
    def effective_cpu_bytes(self) -> int | None:
        """Return the lowest known physical or cgroup CPU memory capacity."""

        known = tuple(
            value
            for value in (self.cpu_total_bytes, self.cpu_limit_bytes)
            if value is not None
        )
        return min(known) if known else None


@dataclass(frozen=True, slots=True)
class ResourcePressureSummary:
    """Peak observed memory ratios against a fixed resource capacity."""

    peak_cpu_rss_ratio: float | None = None
    peak_cuda_allocated_ratio: float | None = None
    peak_cuda_reserved_ratio: float | None = None
    peak_cuda_max_allocated_ratio: float | None = None
    peak_cuda_max_reserved_ratio: float | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "peak_cpu_rss_ratio",
            "peak_cuda_allocated_ratio",
            "peak_cuda_reserved_ratio",
            "peak_cuda_max_allocated_ratio",
            "peak_cuda_max_reserved_ratio",
        ):
            object.__setattr__(
                self,
                field_name,
                _validate_optional_ratio(
                    getattr(self, field_name),
                    label=f"ResourcePressureSummary.{field_name}",
                ),
            )

    @property
    def max_observed_ratio(self) -> float | None:
        """Return the highest known pressure ratio, or None if all are unknown."""

        known = tuple(
            value
            for value in (
                self.peak_cpu_rss_ratio,
                self.peak_cuda_allocated_ratio,
                self.peak_cuda_reserved_ratio,
                self.peak_cuda_max_allocated_ratio,
                self.peak_cuda_max_reserved_ratio,
            )
            if value is not None
        )
        return max(known) if known else None


def _peak_ratio(
    current: float | None,
    observed: object,
    capacity: int | None,
    *,
    label: str,
) -> float | None:
    if observed is None:
        return current
    observed_bytes = _validate_optional_non_negative_int(observed, label=label)
    assert observed_bytes is not None
    if capacity is None:
        return current
    ratio = observed_bytes / capacity
    if current is None:
        return ratio
    return max(current, ratio)


def assess_resource_pressure(
    samples: Iterable[ResourceSample],
    capacity: ResourceCapacity,
) -> ResourcePressureSummary:
    """Stream resource samples once and return peak capacity-normalized ratios."""

    if isinstance(samples, ResourceSample) or not isinstance(samples, Iterable):
        raise TypeError(
            "assess_resource_pressure expects an iterable of ResourceSample objects."
        )
    if not isinstance(capacity, ResourceCapacity):
        raise TypeError("assess_resource_pressure capacity must be a ResourceCapacity.")

    peaks: dict[str, float | None] = {
        "peak_cpu_rss_ratio": None,
        "peak_cuda_allocated_ratio": None,
        "peak_cuda_reserved_ratio": None,
        "peak_cuda_max_allocated_ratio": None,
        "peak_cuda_max_reserved_ratio": None,
    }
    saw_sample = False

    for sample in samples:
        if not isinstance(sample, ResourceSample):
            raise TypeError(
                "assess_resource_pressure samples must contain ResourceSample objects."
            )
        saw_sample = True
        peaks["peak_cpu_rss_ratio"] = _peak_ratio(
            peaks["peak_cpu_rss_ratio"],
            sample.cpu_rss_bytes,
            capacity.effective_cpu_bytes,
            label="ResourceSample.cpu_rss_bytes",
        )

        cuda_values = (
            sample.cuda_allocated_bytes,
            sample.cuda_reserved_bytes,
            sample.cuda_max_allocated_bytes,
            sample.cuda_max_reserved_bytes,
        )
        if capacity.cuda_total_bytes is not None and any(
            value is not None for value in cuda_values
        ):
            if sample.cuda_device_index != capacity.cuda_device_index:
                raise ValueError(
                    "ResourceSample.cuda_device_index does not match "
                    "ResourceCapacity.cuda_device_index."
                )

        peaks["peak_cuda_allocated_ratio"] = _peak_ratio(
            peaks["peak_cuda_allocated_ratio"],
            sample.cuda_allocated_bytes,
            capacity.cuda_total_bytes,
            label="ResourceSample.cuda_allocated_bytes",
        )
        peaks["peak_cuda_reserved_ratio"] = _peak_ratio(
            peaks["peak_cuda_reserved_ratio"],
            sample.cuda_reserved_bytes,
            capacity.cuda_total_bytes,
            label="ResourceSample.cuda_reserved_bytes",
        )
        peaks["peak_cuda_max_allocated_ratio"] = _peak_ratio(
            peaks["peak_cuda_max_allocated_ratio"],
            sample.cuda_max_allocated_bytes,
            capacity.cuda_total_bytes,
            label="ResourceSample.cuda_max_allocated_bytes",
        )
        peaks["peak_cuda_max_reserved_ratio"] = _peak_ratio(
            peaks["peak_cuda_max_reserved_ratio"],
            sample.cuda_max_reserved_bytes,
            capacity.cuda_total_bytes,
            label="ResourceSample.cuda_max_reserved_bytes",
        )

    if saw_sample:
        del sample

    return ResourcePressureSummary(**peaks)
