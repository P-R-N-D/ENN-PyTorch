from __future__ import annotations

from dataclasses import dataclass

from .cost import ModelCost, ResourceDelta
from .faults import StepStatus


def _validate_positive_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer.")
    if value <= 0:
        raise ValueError(f"{label} must be positive.")
    return value


def _validate_optional_device_index(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer or None.")
    if value < 0:
        raise ValueError(f"{label} must be non-negative.")
    return value


def _validate_delta(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer or None.")
    return value


def _validate_phase_name(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty normalized string.")
    return value


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True, slots=True)
class ObservedCostCalibrationPolicy:
    """Bounds and device expectations for deterministic observed-cost calibration."""

    min_successful_samples: int = 1
    max_phase_pairs: int = 32
    expected_cuda_device_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "min_successful_samples",
            _validate_positive_int(
                self.min_successful_samples,
                label="ObservedCostCalibrationPolicy.min_successful_samples",
            ),
        )
        object.__setattr__(
            self,
            "max_phase_pairs",
            _validate_positive_int(
                self.max_phase_pairs,
                label="ObservedCostCalibrationPolicy.max_phase_pairs",
            ),
        )
        object.__setattr__(
            self,
            "expected_cuda_device_index",
            _validate_optional_device_index(
                self.expected_cuda_device_index,
                label="ObservedCostCalibrationPolicy.expected_cuda_device_index",
            ),
        )


class ObservedCostCalibrationError(RuntimeError):
    """Raised when an observation cannot safely join one calibration profile."""

    def __init__(
        self,
        reason: str,
        *,
        dimensions: tuple[str, ...] = (),
    ) -> None:
        self.reason = reason
        self.dimensions = dimensions
        suffix = f" dimensions={dimensions!r}" if dimensions else ""
        super().__init__(f"Observed cost calibration failed: {reason}.{suffix}")


@dataclass(frozen=True, slots=True)
class ObservedCostMetricProfile:
    """One scalar metric's conservative per-item envelope and observation counts."""

    max_bytes_per_item: int | None
    known_samples: int
    unknown_samples: int
    zero_samples: int
    negative_deltas_clamped: int


@dataclass(frozen=True, slots=True)
class ObservedPhaseCostProfile:
    """Conservative metric envelopes for one adjacent runtime phase pair."""

    start_phase: str
    end_phase: str
    cpu_rss: ObservedCostMetricProfile
    cuda_allocated: ObservedCostMetricProfile
    cuda_reserved: ObservedCostMetricProfile
    cuda_max_allocated: ObservedCostMetricProfile
    cuda_max_reserved: ObservedCostMetricProfile


@dataclass(frozen=True, slots=True)
class ObservedCostProfile:
    """Bounded aggregate profile derived from accepted successful observations."""

    policy: ObservedCostCalibrationPolicy
    total_observations: int
    successful_samples: int
    ignored_samples: int
    rejected_samples: int
    ignored_zero_batch_samples: int
    ignored_by_status: tuple[tuple[str, int], ...]
    min_batch_size: int
    max_batch_size: int
    cuda_device_index: int | None
    cpu_rss: ObservedCostMetricProfile
    cuda_allocated: ObservedCostMetricProfile
    cuda_reserved: ObservedCostMetricProfile
    cuda_max_allocated: ObservedCostMetricProfile
    cuda_max_reserved: ObservedCostMetricProfile
    phase_costs: tuple[ObservedPhaseCostProfile, ...]


class _MetricAccumulator:
    __slots__ = (
        "max_bytes_per_item",
        "known_samples",
        "unknown_samples",
        "zero_samples",
        "negative_deltas_clamped",
    )

    def __init__(self) -> None:
        self.max_bytes_per_item: int | None = None
        self.known_samples = 0
        self.unknown_samples = 0
        self.zero_samples = 0
        self.negative_deltas_clamped = 0

    def observe(self, value: int | None, *, batch_size: int) -> None:
        if value is None:
            self.unknown_samples += 1
            return

        self.known_samples += 1
        normalized = value
        if value < 0:
            normalized = 0
            self.negative_deltas_clamped += 1
        elif value == 0:
            self.zero_samples += 1

        per_item = _ceil_div(normalized, batch_size)
        if self.max_bytes_per_item is None or per_item > self.max_bytes_per_item:
            self.max_bytes_per_item = per_item

    def profile(self) -> ObservedCostMetricProfile:
        return ObservedCostMetricProfile(
            max_bytes_per_item=self.max_bytes_per_item,
            known_samples=self.known_samples,
            unknown_samples=self.unknown_samples,
            zero_samples=self.zero_samples,
            negative_deltas_clamped=self.negative_deltas_clamped,
        )


class _PhaseAccumulator:
    __slots__ = (
        "start_phase",
        "end_phase",
        "cpu_rss",
        "cuda_allocated",
        "cuda_reserved",
        "cuda_max_allocated",
        "cuda_max_reserved",
    )

    def __init__(self, start_phase: str, end_phase: str) -> None:
        self.start_phase = start_phase
        self.end_phase = end_phase
        self.cpu_rss = _MetricAccumulator()
        self.cuda_allocated = _MetricAccumulator()
        self.cuda_reserved = _MetricAccumulator()
        self.cuda_max_allocated = _MetricAccumulator()
        self.cuda_max_reserved = _MetricAccumulator()

    def observe(self, delta: ResourceDelta, *, batch_size: int) -> None:
        self.cpu_rss.observe(delta.cpu_rss_delta_bytes, batch_size=batch_size)
        self.cuda_allocated.observe(
            delta.cuda_allocated_delta_bytes,
            batch_size=batch_size,
        )
        self.cuda_reserved.observe(
            delta.cuda_reserved_delta_bytes,
            batch_size=batch_size,
        )
        self.cuda_max_allocated.observe(
            delta.cuda_max_allocated_delta_bytes,
            batch_size=batch_size,
        )
        self.cuda_max_reserved.observe(
            delta.cuda_max_reserved_delta_bytes,
            batch_size=batch_size,
        )

    def profile(self) -> ObservedPhaseCostProfile:
        return ObservedPhaseCostProfile(
            start_phase=self.start_phase,
            end_phase=self.end_phase,
            cpu_rss=self.cpu_rss.profile(),
            cuda_allocated=self.cuda_allocated.profile(),
            cuda_reserved=self.cuda_reserved.profile(),
            cuda_max_allocated=self.cuda_max_allocated.profile(),
            cuda_max_reserved=self.cuda_max_reserved.profile(),
        )


@dataclass(frozen=True, slots=True)
class _NormalizedObservation:
    batch_size: int
    cuda_device_index: int | None
    total_metrics: tuple[int | None, int | None, int | None, int | None, int | None]
    phase_deltas: tuple[ResourceDelta, ...]


class ObservedCostCalibrator:
    """Aggregate successful ``ModelCost`` values without retaining raw observations."""

    __slots__ = (
        "policy",
        "_total_observations",
        "_successful_samples",
        "_ignored_zero_batch_samples",
        "_ignored_by_status",
        "_min_batch_size",
        "_max_batch_size",
        "_cuda_device_index",
        "_cpu_rss",
        "_cuda_allocated",
        "_cuda_reserved",
        "_cuda_max_allocated",
        "_cuda_max_reserved",
        "_phase_costs",
    )

    def __init__(self, policy: ObservedCostCalibrationPolicy | None = None) -> None:
        if policy is None:
            policy = ObservedCostCalibrationPolicy()
        elif not isinstance(policy, ObservedCostCalibrationPolicy):
            raise TypeError(
                "ObservedCostCalibrator.policy must be an "
                "ObservedCostCalibrationPolicy or None."
            )

        self.policy = policy
        self._total_observations = 0
        self._successful_samples = 0
        self._ignored_zero_batch_samples = 0
        self._ignored_by_status: dict[str, int] = {}
        self._min_batch_size: int | None = None
        self._max_batch_size: int | None = None
        self._cuda_device_index: int | None = None
        self._cpu_rss = _MetricAccumulator()
        self._cuda_allocated = _MetricAccumulator()
        self._cuda_reserved = _MetricAccumulator()
        self._cuda_max_allocated = _MetricAccumulator()
        self._cuda_max_reserved = _MetricAccumulator()
        self._phase_costs: dict[tuple[str, str], _PhaseAccumulator] = {}

    def observe(self, cost: ModelCost) -> bool:
        """Observe one cost; return ``True`` only when it contributes to the profile."""

        if not isinstance(cost, ModelCost):
            raise TypeError("ObservedCostCalibrator.observe expects a ModelCost.")
        if not isinstance(cost.status, StepStatus):
            raise TypeError("ModelCost.status must be a StepStatus.")

        self._total_observations += 1
        if cost.status is not StepStatus.SUCCESS:
            key = cost.status.value
            self._ignored_by_status[key] = self._ignored_by_status.get(key, 0) + 1
            return False

        observation = self._normalize_success(cost)
        if observation is None:
            self._ignored_zero_batch_samples += 1
            return False

        self._apply(observation)
        return True

    def profile(self) -> ObservedCostProfile:
        """Return an immutable snapshot once the configured sample floor is met."""

        if self._successful_samples < self.policy.min_successful_samples:
            raise ObservedCostCalibrationError(
                "not enough successful samples to build a profile",
                dimensions=("samples",),
            )
        assert self._min_batch_size is not None
        assert self._max_batch_size is not None

        ignored_by_status = tuple(sorted(self._ignored_by_status.items()))
        ignored_samples = self._ignored_zero_batch_samples + sum(
            count for _status, count in ignored_by_status
        )
        phase_costs = tuple(
            self._phase_costs[key].profile()
            for key in sorted(self._phase_costs)
        )
        return ObservedCostProfile(
            policy=self.policy,
            total_observations=self._total_observations,
            successful_samples=self._successful_samples,
            ignored_samples=ignored_samples,
            rejected_samples=(
                self._total_observations
                - self._successful_samples
                - ignored_samples
            ),
            ignored_zero_batch_samples=self._ignored_zero_batch_samples,
            ignored_by_status=ignored_by_status,
            min_batch_size=self._min_batch_size,
            max_batch_size=self._max_batch_size,
            cuda_device_index=self._cuda_device_index,
            cpu_rss=self._cpu_rss.profile(),
            cuda_allocated=self._cuda_allocated.profile(),
            cuda_reserved=self._cuda_reserved.profile(),
            cuda_max_allocated=self._cuda_max_allocated.profile(),
            cuda_max_reserved=self._cuda_max_reserved.profile(),
            phase_costs=phase_costs,
        )

    def _normalize_success(self, cost: ModelCost) -> _NormalizedObservation | None:
        batch_size = self._validate_non_negative_model_int(
            cost.batch_size,
            label="ModelCost.batch_size",
        )
        row_count = self._validate_non_negative_model_int(
            cost.row_count,
            label="ModelCost.row_count",
        )
        if batch_size != row_count:
            raise ObservedCostCalibrationError(
                "ModelCost.batch_size and row_count must match",
                dimensions=("items",),
            )
        if batch_size == 0:
            return None

        total_metrics = (
            _validate_delta(
                cost.total_cpu_rss_delta_bytes,
                label="ModelCost.total_cpu_rss_delta_bytes",
            ),
            _validate_delta(
                cost.total_cuda_allocated_delta_bytes,
                label="ModelCost.total_cuda_allocated_delta_bytes",
            ),
            _validate_delta(
                cost.total_cuda_reserved_delta_bytes,
                label="ModelCost.total_cuda_reserved_delta_bytes",
            ),
            _validate_delta(
                cost.total_cuda_max_allocated_delta_bytes,
                label="ModelCost.total_cuda_max_allocated_delta_bytes",
            ),
            _validate_delta(
                cost.total_cuda_max_reserved_delta_bytes,
                label="ModelCost.total_cuda_max_reserved_delta_bytes",
            ),
        )

        phase_deltas: list[ResourceDelta] = []
        seen_phase_pairs: set[tuple[str, str]] = set()
        for index, delta in enumerate(cost.phase_deltas):
            if not isinstance(delta, ResourceDelta):
                raise TypeError(f"ModelCost.phase_deltas[{index}] must be a ResourceDelta.")
            start_phase = _validate_phase_name(
                delta.start_phase,
                label=f"ModelCost.phase_deltas[{index}].start_phase",
            )
            end_phase = _validate_phase_name(
                delta.end_phase,
                label=f"ModelCost.phase_deltas[{index}].end_phase",
            )
            phase_pair = (start_phase, end_phase)
            if phase_pair in seen_phase_pairs:
                raise ObservedCostCalibrationError(
                    "one ModelCost cannot repeat an adjacent phase pair",
                    dimensions=("phases",),
                )
            seen_phase_pairs.add(phase_pair)
            phase_deltas.append(
                ResourceDelta(
                    start_phase=start_phase,
                    end_phase=end_phase,
                    cpu_rss_delta_bytes=_validate_delta(
                        delta.cpu_rss_delta_bytes,
                        label=f"ModelCost.phase_deltas[{index}].cpu_rss_delta_bytes",
                    ),
                    cuda_allocated_delta_bytes=_validate_delta(
                        delta.cuda_allocated_delta_bytes,
                        label=(
                            f"ModelCost.phase_deltas[{index}]."
                            "cuda_allocated_delta_bytes"
                        ),
                    ),
                    cuda_reserved_delta_bytes=_validate_delta(
                        delta.cuda_reserved_delta_bytes,
                        label=(
                            f"ModelCost.phase_deltas[{index}]."
                            "cuda_reserved_delta_bytes"
                        ),
                    ),
                    cuda_max_allocated_delta_bytes=_validate_delta(
                        delta.cuda_max_allocated_delta_bytes,
                        label=(
                            f"ModelCost.phase_deltas[{index}]."
                            "cuda_max_allocated_delta_bytes"
                        ),
                    ),
                    cuda_max_reserved_delta_bytes=_validate_delta(
                        delta.cuda_max_reserved_delta_bytes,
                        label=(
                            f"ModelCost.phase_deltas[{index}]."
                            "cuda_max_reserved_delta_bytes"
                        ),
                    ),
                )
            )

        new_phase_pairs = {
            (delta.start_phase, delta.end_phase)
            for delta in phase_deltas
            if (delta.start_phase, delta.end_phase) not in self._phase_costs
        }
        if len(self._phase_costs) + len(new_phase_pairs) > self.policy.max_phase_pairs:
            raise ObservedCostCalibrationError(
                "observation would exceed max_phase_pairs",
                dimensions=("phases",),
            )

        cuda_device_index = _validate_optional_device_index(
            cost.cuda_device_index,
            label="ModelCost.cuda_device_index",
        )
        has_cuda_metrics = any(value is not None for value in total_metrics[1:]) or any(
            self._phase_has_cuda_metrics(delta) for delta in phase_deltas
        )
        if has_cuda_metrics and cuda_device_index is None:
            raise ObservedCostCalibrationError(
                "known CUDA metrics require a concrete CUDA device index",
                dimensions=("cuda",),
            )
        if cuda_device_index is not None:
            expected = self.policy.expected_cuda_device_index
            if expected is not None and cuda_device_index != expected:
                raise ObservedCostCalibrationError(
                    "CUDA observation does not match the policy device",
                    dimensions=(f"cuda:{cuda_device_index}",),
                )
            if (
                self._cuda_device_index is not None
                and cuda_device_index != self._cuda_device_index
            ):
                raise ObservedCostCalibrationError(
                    "CUDA observations from different devices cannot share one profile",
                    dimensions=(
                        f"cuda:{self._cuda_device_index}",
                        f"cuda:{cuda_device_index}",
                    ),
                )

        return _NormalizedObservation(
            batch_size=batch_size,
            cuda_device_index=cuda_device_index if has_cuda_metrics else None,
            total_metrics=total_metrics,
            phase_deltas=tuple(phase_deltas),
        )

    def _apply(self, observation: _NormalizedObservation) -> None:
        batch_size = observation.batch_size
        (
            cpu_rss,
            cuda_allocated,
            cuda_reserved,
            cuda_max_allocated,
            cuda_max_reserved,
        ) = observation.total_metrics

        self._cpu_rss.observe(cpu_rss, batch_size=batch_size)
        self._cuda_allocated.observe(cuda_allocated, batch_size=batch_size)
        self._cuda_reserved.observe(cuda_reserved, batch_size=batch_size)
        self._cuda_max_allocated.observe(
            cuda_max_allocated,
            batch_size=batch_size,
        )
        self._cuda_max_reserved.observe(
            cuda_max_reserved,
            batch_size=batch_size,
        )

        for delta in observation.phase_deltas:
            key = (delta.start_phase, delta.end_phase)
            accumulator = self._phase_costs.get(key)
            if accumulator is None:
                accumulator = _PhaseAccumulator(*key)
                self._phase_costs[key] = accumulator
            accumulator.observe(delta, batch_size=batch_size)

        self._successful_samples += 1
        self._min_batch_size = (
            batch_size
            if self._min_batch_size is None
            else min(self._min_batch_size, batch_size)
        )
        self._max_batch_size = (
            batch_size
            if self._max_batch_size is None
            else max(self._max_batch_size, batch_size)
        )
        if observation.cuda_device_index is not None:
            self._cuda_device_index = observation.cuda_device_index

    @staticmethod
    def _validate_non_negative_model_int(value: object, *, label: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{label} must be an integer.")
        if value < 0:
            raise ValueError(f"{label} must be non-negative.")
        return value

    @staticmethod
    def _phase_has_cuda_metrics(delta: ResourceDelta) -> bool:
        return any(
            value is not None
            for value in (
                delta.cuda_allocated_delta_bytes,
                delta.cuda_reserved_delta_bytes,
                delta.cuda_max_allocated_delta_bytes,
                delta.cuda_max_reserved_delta_bytes,
            )
        )
