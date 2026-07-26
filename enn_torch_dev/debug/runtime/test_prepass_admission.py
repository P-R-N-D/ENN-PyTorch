from __future__ import annotations

from dataclasses import replace

import pytest

from enn_torch_dev.runtime import (
    ObservedCostCalibrationPolicy,
    ObservedCostMetricProfile,
    ObservedCostProfile,
    PrePassAdmissionError,
    PrePassAdmissionPolicy,
    PrePassAdmissionStatus,
    ResourceCapacity,
    ResourceSample,
    assess_prepass_admission,
)


def _metric(value: int | None) -> ObservedCostMetricProfile:
    return ObservedCostMetricProfile(
        max_bytes_per_item=value,
        known_samples=0 if value is None else 1,
        unknown_samples=1 if value is None else 0,
        zero_samples=1 if value == 0 else 0,
        negative_deltas_clamped=0,
    )


def _profile(
    *,
    cpu: int | None = 10,
    allocated: int | None = None,
    reserved: int | None = None,
    max_allocated: int | None = None,
    max_reserved: int | None = None,
    samples: int = 3,
    device_index: int | None = None,
) -> ObservedCostProfile:
    return ObservedCostProfile(
        policy=ObservedCostCalibrationPolicy(),
        total_observations=samples,
        successful_samples=samples,
        ignored_samples=0,
        rejected_samples=0,
        ignored_zero_batch_samples=0,
        ignored_by_status=(),
        min_batch_size=1,
        max_batch_size=8,
        cuda_device_index=device_index,
        cpu_rss=_metric(cpu),
        cuda_allocated=_metric(allocated),
        cuda_reserved=_metric(reserved),
        cuda_max_allocated=_metric(max_allocated),
        cuda_max_reserved=_metric(max_reserved),
        phase_costs=(),
    )


def _sample(
    *,
    cpu: int | None = 100,
    allocated: int | None = None,
    reserved: int | None = None,
    max_allocated: int | None = None,
    max_reserved: int | None = None,
    device_index: int | None = None,
) -> ResourceSample:
    return ResourceSample(
        timestamp_ns=1,
        phase="before_step",
        cpu_rss_bytes=cpu,
        cuda_available=device_index is not None,
        cuda_device_index=device_index,
        cuda_allocated_bytes=allocated,
        cuda_reserved_bytes=reserved,
        cuda_max_allocated_bytes=max_allocated,
        cuda_max_reserved_bytes=max_reserved,
    )


def _dimension(assessment, name):
    return next(d for d in assessment.dimensions if d.name == name)


def test_cpu_only_admit() -> None:
    result = assess_prepass_admission(
        ResourceCapacity(cpu_total_bytes=1_000),
        _sample(cpu=100),
        _profile(cpu=50),
        batch_size=5,
    )
    assert result.status is PrePassAdmissionStatus.ADMIT
    assert result.admitted
    cpu = _dimension(result, "cpu_rss")
    assert cpu.usable_bytes == 900
    assert cpu.projected_bytes == 350
    assert cpu.item_limit == 16
    assert [d.name for d in result.dimensions] == [
        "cpu_rss",
        "cuda_allocated",
        "cuda_reserved",
    ]
    assert not _dimension(result, "cuda_allocated").applicable


def test_cpu_projected_reject() -> None:
    result = assess_prepass_admission(
        ResourceCapacity(cpu_total_bytes=1_000),
        _sample(cpu=700),
        _profile(cpu=50),
        batch_size=5,
    )
    assert result.status is PrePassAdmissionStatus.REJECT
    assert result.rejected_dimensions == ("cpu_rss",)
    assert result.max_admissible_items == 4


def test_baseline_over_limit_rejects_even_with_unknown_increment() -> None:
    result = assess_prepass_admission(
        ResourceCapacity(cpu_total_bytes=1_000),
        _sample(cpu=901),
        _profile(cpu=None),
        batch_size=1,
    )
    cpu = _dimension(result, "cpu_rss")
    assert result.status is PrePassAdmissionStatus.REJECT
    assert cpu.item_limit == 0
    assert cpu.reason == "current usage exceeds usable capacity"


@pytest.mark.parametrize(
    ("capacity", "sample", "profile", "reason"),
    [
        (ResourceCapacity(), _sample(cpu=100), _profile(cpu=10), "capacity is unknown"),
        (
            ResourceCapacity(cpu_total_bytes=1_000),
            _sample(cpu=None),
            _profile(cpu=10),
            "current usage is unknown",
        ),
        (
            ResourceCapacity(cpu_total_bytes=1_000),
            _sample(cpu=100),
            _profile(cpu=None),
            "incremental cost is unknown",
        ),
    ],
)
def test_cpu_unknown_inputs(capacity, sample, profile, reason) -> None:
    result = assess_prepass_admission(
        capacity, sample, profile, batch_size=2
    )
    assert result.status is PrePassAdmissionStatus.UNKNOWN
    assert result.unknown_dimensions == ("cpu_rss",)
    assert _dimension(result, "cpu_rss").reason == reason



def test_effective_cpu_capacity_uses_smallest_known_limit() -> None:
    result = assess_prepass_admission(
        ResourceCapacity(cpu_total_bytes=1_000, cpu_limit_bytes=600),
        _sample(cpu=100),
        _profile(cpu=100),
        batch_size=4,
    )
    cpu = _dimension(result, "cpu_rss")
    assert cpu.capacity_bytes == 600
    assert cpu.usable_bytes == 540
    assert cpu.item_limit == 4
    assert result.status is PrePassAdmissionStatus.ADMIT

def test_known_zero_is_non_limiting() -> None:
    result = assess_prepass_admission(
        ResourceCapacity(cpu_total_bytes=1_000),
        _sample(cpu=100),
        _profile(cpu=0),
        batch_size=100_000,
    )
    cpu = _dimension(result, "cpu_rss")
    assert result.status is PrePassAdmissionStatus.ADMIT
    assert cpu.projected_bytes == 100
    assert cpu.item_limit is None
    assert cpu.item_limit_is_unbounded
    assert result.max_admissible_items is None


def test_utilization_then_reserve_order() -> None:
    policy = PrePassAdmissionPolicy(host_utilization_ratio=0.5, host_reserve_bytes=100)
    result = assess_prepass_admission(
        ResourceCapacity(cpu_total_bytes=1_000),
        _sample(cpu=100),
        _profile(cpu=100),
        batch_size=3,
        policy=policy,
    )
    cpu = _dimension(result, "cpu_rss")
    assert cpu.usable_bytes == 400
    assert cpu.projected_bytes == 400
    assert result.status is PrePassAdmissionStatus.ADMIT


def test_profile_sample_floor_makes_increment_unknown() -> None:
    policy = PrePassAdmissionPolicy(min_profile_samples=3)
    result = assess_prepass_admission(
        ResourceCapacity(cpu_total_bytes=1_000),
        _sample(cpu=100),
        _profile(cpu=10, samples=2),
        batch_size=2,
        policy=policy,
    )
    cpu = _dimension(result, "cpu_rss")
    assert result.status is PrePassAdmissionStatus.UNKNOWN
    assert cpu.incremental_bytes_per_item is None
    assert result.warnings == (
        "observed profile successful sample count 2 is below required 3",
        "cpu_rss: observed profile sample floor is not met",
    )


@pytest.mark.parametrize(
    ("field", "current", "increment", "batch_size", "expected"),
    [
        ("allocated", 100, 100, 5, PrePassAdmissionStatus.ADMIT),
        ("allocated", 500, 100, 5, PrePassAdmissionStatus.REJECT),
        ("reserved", 100, 100, 5, PrePassAdmissionStatus.ADMIT),
        ("reserved", 500, 100, 5, PrePassAdmissionStatus.REJECT),
    ],
)
def test_cuda_dimensions_admit_and_reject(field, current, increment, batch_size, expected):
    other_field = "reserved" if field == "allocated" else "allocated"
    profile_kwargs = {field: increment, other_field: 0, "device_index": 0}
    sample_kwargs = {field: current, other_field: 0, "device_index": 0}
    result = assess_prepass_admission(
        ResourceCapacity(cpu_total_bytes=2_000, cuda_total_bytes=1_000, cuda_device_index=0),
        _sample(cpu=100, **sample_kwargs),
        _profile(cpu=0, **profile_kwargs),
        batch_size=batch_size,
    )
    assert result.status is expected
    assert _dimension(result, f"cuda_{field}").status is expected


def test_cuda_uses_larger_direct_or_peak_increment() -> None:
    result = assess_prepass_admission(
        ResourceCapacity(cpu_total_bytes=2_000, cuda_total_bytes=2_000, cuda_device_index=0),
        _sample(cpu=100, allocated=100, reserved=100, device_index=0),
        _profile(
            cpu=0,
            allocated=20,
            max_allocated=50,
            reserved=60,
            max_reserved=30,
            device_index=0,
        ),
        batch_size=2,
    )
    assert _dimension(result, "cuda_allocated").incremental_bytes_per_item == 50
    assert _dimension(result, "cuda_reserved").incremental_bytes_per_item == 60


@pytest.mark.parametrize("kind", ["profile", "sample"])
def test_cuda_device_mismatch_rejected(kind: str) -> None:
    profile_index = 1 if kind == "profile" else 0
    sample_index = 1 if kind == "sample" else 0
    with pytest.raises(PrePassAdmissionError, match="does not match capacity"):
        assess_prepass_admission(
            ResourceCapacity(cpu_total_bytes=2_000, cuda_total_bytes=1_000, cuda_device_index=0),
            _sample(cpu=100, allocated=100, device_index=sample_index),
            _profile(cpu=0, allocated=10, device_index=profile_index),
            batch_size=1,
        )


def test_cuda_inputs_without_capacity_rejected() -> None:
    with pytest.raises(PrePassAdmissionError, match="require CUDA capacity"):
        assess_prepass_admission(
            ResourceCapacity(cpu_total_bytes=2_000),
            _sample(cpu=100, allocated=100, device_index=0),
            _profile(cpu=0, allocated=10, device_index=0),
            batch_size=1,
        )



def test_profile_cuda_metrics_require_device_index() -> None:
    with pytest.raises(PrePassAdmissionError, match="profile metrics"):
        assess_prepass_admission(
            ResourceCapacity(
                cpu_total_bytes=2_000,
                cuda_total_bytes=1_000,
                cuda_device_index=0,
            ),
            _sample(cpu=100, allocated=100, device_index=0),
            _profile(cpu=0, allocated=10, device_index=None),
            batch_size=1,
        )


def test_baseline_cuda_max_counters_are_not_added_as_current_usage() -> None:
    result = assess_prepass_admission(
        ResourceCapacity(
            cpu_total_bytes=2_000,
            cuda_total_bytes=1_000,
            cuda_device_index=0,
        ),
        _sample(
            cpu=100,
            allocated=100,
            reserved=100,
            max_allocated=900,
            max_reserved=900,
            device_index=0,
        ),
        _profile(
            cpu=0,
            allocated=10,
            reserved=20,
            max_allocated=30,
            max_reserved=40,
            device_index=0,
        ),
        batch_size=2,
    )
    assert _dimension(result, "cuda_allocated").projected_bytes == 160
    assert _dimension(result, "cuda_reserved").projected_bytes == 180
    assert result.status is PrePassAdmissionStatus.ADMIT

def test_baseline_cuda_values_require_index() -> None:
    with pytest.raises(PrePassAdmissionError, match="baseline CUDA values"):
        assess_prepass_admission(
            ResourceCapacity(cpu_total_bytes=2_000, cuda_total_bytes=1_000, cuda_device_index=0),
            _sample(cpu=100, allocated=100, device_index=None),
            _profile(cpu=0, allocated=10, device_index=0),
            batch_size=1,
        )


def test_cpu_admit_cuda_unknown_makes_overall_unknown() -> None:
    result = assess_prepass_admission(
        ResourceCapacity(cpu_total_bytes=2_000, cuda_total_bytes=1_000, cuda_device_index=0),
        _sample(cpu=100, allocated=100, reserved=100, device_index=0),
        _profile(cpu=10, allocated=None, reserved=None),
        batch_size=2,
    )
    assert _dimension(result, "cpu_rss").status is PrePassAdmissionStatus.ADMIT
    assert result.status is PrePassAdmissionStatus.UNKNOWN
    assert result.unknown_dimensions == ("cuda_allocated", "cuda_reserved")


def test_reject_takes_priority_over_unknown() -> None:
    result = assess_prepass_admission(
        ResourceCapacity(cpu_total_bytes=None, cuda_total_bytes=1_000, cuda_device_index=0),
        _sample(cpu=None, allocated=850, reserved=100, device_index=0),
        _profile(cpu=None, allocated=100, reserved=None, device_index=0),
        batch_size=1,
    )
    assert _dimension(result, "cpu_rss").status is PrePassAdmissionStatus.UNKNOWN
    assert _dimension(result, "cuda_allocated").status is PrePassAdmissionStatus.REJECT
    assert result.status is PrePassAdmissionStatus.REJECT


def test_item_limits_and_overall_known_minimum() -> None:
    result = assess_prepass_admission(
        ResourceCapacity(cpu_total_bytes=1_000, cuda_total_bytes=1_000, cuda_device_index=0),
        _sample(cpu=500, allocated=600, reserved=500, device_index=0),
        _profile(cpu=100, allocated=100, reserved=200, device_index=0),
        batch_size=1,
    )
    assert _dimension(result, "cpu_rss").item_limit == 4
    assert _dimension(result, "cuda_allocated").item_limit == 3
    assert _dimension(result, "cuda_reserved").item_limit == 2
    assert result.max_admissible_items == 2


@pytest.mark.parametrize("batch_size", [True, False, 0, -1, 1.5])
def test_invalid_batch_size_rejected(batch_size: object) -> None:
    with pytest.raises((TypeError, ValueError), match="batch_size"):
        assess_prepass_admission(
            ResourceCapacity(cpu_total_bytes=1_000),
            _sample(),
            _profile(),
            batch_size=batch_size,  # type: ignore[arg-type]
        )


def test_dimension_and_warning_order_is_deterministic() -> None:
    result = assess_prepass_admission(
        ResourceCapacity(cpu_total_bytes=None, cuda_total_bytes=1_000, cuda_device_index=0),
        _sample(cpu=None, allocated=None, reserved=None),
        _profile(cpu=None, allocated=10, reserved=None, device_index=0),
        batch_size=1,
    )
    assert [d.name for d in result.dimensions] == [
        "cpu_rss",
        "cuda_allocated",
        "cuda_reserved",
    ]
    assert result.unknown_dimensions == (
        "cpu_rss",
        "cuda_allocated",
        "cuda_reserved",
    )
    assert result.warnings == (
        "cpu_rss: capacity is unknown",
        "cuda_allocated: current usage is unknown",
        "cuda_reserved: current usage is unknown",
    )


def test_inputs_are_not_mutated() -> None:
    capacity = ResourceCapacity(cpu_total_bytes=1_000)
    sample = _sample(cpu=100)
    profile = _profile(cpu=10)
    before = (capacity, sample, profile)
    assess_prepass_admission(capacity, sample, profile, batch_size=1)
    assert before == (capacity, sample, profile)


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"host_utilization_ratio": 0}, ValueError),
        ({"device_utilization_ratio": 1.1}, ValueError),
        ({"host_utilization_ratio": True}, TypeError),
        ({"host_reserve_bytes": -1}, ValueError),
        ({"device_reserve_bytes": True}, TypeError),
        ({"min_profile_samples": 0}, ValueError),
    ],
)
def test_policy_validation(kwargs, error_type) -> None:
    with pytest.raises(error_type):
        PrePassAdmissionPolicy(**kwargs)


def test_invalid_public_inputs_rejected() -> None:
    capacity = ResourceCapacity(cpu_total_bytes=1_000)
    sample = _sample()
    profile = _profile()
    with pytest.raises(TypeError, match="capacity"):
        assess_prepass_admission(object(), sample, profile, batch_size=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="baseline_sample"):
        assess_prepass_admission(capacity, object(), profile, batch_size=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="observed_profile"):
        assess_prepass_admission(capacity, sample, object(), batch_size=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="policy"):
        assess_prepass_admission(capacity, sample, profile, batch_size=1, policy=object())  # type: ignore[arg-type]


def test_invalid_metric_value_rejected() -> None:
    profile = replace(
        _profile(),
        cpu_rss=ObservedCostMetricProfile(
            max_bytes_per_item=-1,
            known_samples=1,
            unknown_samples=0,
            zero_samples=0,
            negative_deltas_clamped=0,
        ),
    )
    with pytest.raises(ValueError, match="max_bytes_per_item"):
        assess_prepass_admission(
            ResourceCapacity(cpu_total_bytes=1_000),
            _sample(),
            profile,
            batch_size=1,
        )
