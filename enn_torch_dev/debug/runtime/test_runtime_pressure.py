from __future__ import annotations

from collections.abc import Iterator

import pytest

from enn_torch_dev.runtime import (
    ResourceCapacity,
    ResourcePressureSummary,
    ResourceSample,
    assess_resource_pressure,
)


def _sample(
    *,
    cpu_rss_bytes: int | None = None,
    cuda_device_index: int | None = None,
    cuda_allocated_bytes: int | None = None,
    cuda_reserved_bytes: int | None = None,
    cuda_max_allocated_bytes: int | None = None,
    cuda_max_reserved_bytes: int | None = None,
) -> ResourceSample:
    return ResourceSample(
        timestamp_ns=1,
        phase="test",
        cpu_rss_bytes=cpu_rss_bytes,
        cuda_available=cuda_device_index is not None,
        cuda_device_index=cuda_device_index,
        cuda_allocated_bytes=cuda_allocated_bytes,
        cuda_reserved_bytes=cuda_reserved_bytes,
        cuda_max_allocated_bytes=cuda_max_allocated_bytes,
        cuda_max_reserved_bytes=cuda_max_reserved_bytes,
    )


def test_resource_capacity_preserves_existing_positional_field_order() -> None:
    capacity = ResourceCapacity(100, 200, 0)

    assert capacity.cpu_total_bytes == 100
    assert capacity.cuda_total_bytes == 200
    assert capacity.cuda_device_index == 0
    assert capacity.cpu_limit_bytes is None


def test_resource_capacity_uses_lowest_known_cpu_capacity() -> None:
    assert (
        ResourceCapacity(
            cpu_total_bytes=16_000,
            cpu_limit_bytes=4_000,
        ).effective_cpu_bytes
        == 4_000
    )
    assert (
        ResourceCapacity(
            cpu_total_bytes=16_000,
            cpu_limit_bytes=32_000,
        ).effective_cpu_bytes
        == 16_000
    )
    assert (
        ResourceCapacity(
            cpu_limit_bytes=4_000,
        ).effective_cpu_bytes
        == 4_000
    )
    assert ResourceCapacity().effective_cpu_bytes is None


def test_assess_resource_pressure_uses_effective_cpu_capacity() -> None:
    capacity = ResourceCapacity(
        cpu_total_bytes=16_000,
        cpu_limit_bytes=8_000,
        cuda_total_bytes=2_000,
        cuda_device_index=0,
    )

    summary = assess_resource_pressure(
        [
            _sample(
                cpu_rss_bytes=4_000,
                cuda_device_index=0,
                cuda_allocated_bytes=500,
            )
        ],
        capacity,
    )

    assert summary.peak_cpu_rss_ratio == pytest.approx(0.5)
    assert summary.peak_cuda_allocated_ratio == pytest.approx(0.25)


def test_assess_resource_pressure_calculates_cpu_and_cuda_ratios() -> None:
    capacity = ResourceCapacity(
        cpu_total_bytes=1_000,
        cuda_total_bytes=2_000,
        cuda_device_index=0,
    )

    summary = assess_resource_pressure(
        [
            _sample(
                cpu_rss_bytes=250,
                cuda_device_index=0,
                cuda_allocated_bytes=500,
                cuda_reserved_bytes=750,
                cuda_max_allocated_bytes=1_000,
                cuda_max_reserved_bytes=1_250,
            )
        ],
        capacity,
    )

    assert summary.peak_cpu_rss_ratio == pytest.approx(0.25)
    assert summary.peak_cuda_allocated_ratio == pytest.approx(0.25)
    assert summary.peak_cuda_reserved_ratio == pytest.approx(0.375)
    assert summary.peak_cuda_max_allocated_ratio == pytest.approx(0.5)
    assert summary.peak_cuda_max_reserved_ratio == pytest.approx(0.625)


def test_assess_resource_pressure_uses_peaks_and_does_not_clamp() -> None:
    capacity = ResourceCapacity(
        cpu_total_bytes=100,
        cuda_total_bytes=100,
        cuda_device_index=3,
    )

    summary = assess_resource_pressure(
        [
            _sample(
                cpu_rss_bytes=80,
                cuda_device_index=3,
                cuda_allocated_bytes=50,
                cuda_reserved_bytes=60,
            ),
            _sample(
                cpu_rss_bytes=120,
                cuda_device_index=3,
                cuda_allocated_bytes=110,
                cuda_reserved_bytes=90,
                cuda_max_allocated_bytes=130,
                cuda_max_reserved_bytes=140,
            ),
        ],
        capacity,
    )

    assert summary.peak_cpu_rss_ratio == pytest.approx(1.2)
    assert summary.peak_cuda_allocated_ratio == pytest.approx(1.1)
    assert summary.peak_cuda_reserved_ratio == pytest.approx(0.9)
    assert summary.peak_cuda_max_allocated_ratio == pytest.approx(1.3)
    assert summary.peak_cuda_max_reserved_ratio == pytest.approx(1.4)


def test_assess_resource_pressure_preserves_unknown_values() -> None:
    summary = assess_resource_pressure(
        [_sample(cpu_rss_bytes=0, cuda_device_index=0)],
        ResourceCapacity(),
    )

    assert summary == ResourcePressureSummary()


def test_assess_resource_pressure_preserves_missing_observations() -> None:
    summary = assess_resource_pressure(
        [_sample(cuda_device_index=0)],
        ResourceCapacity(
            cpu_total_bytes=100,
            cuda_total_bytes=100,
            cuda_device_index=0,
        ),
    )

    assert summary == ResourcePressureSummary()


def test_assess_resource_pressure_accepts_empty_iterable() -> None:
    assert assess_resource_pressure([], ResourceCapacity()) == (
        ResourcePressureSummary()
    )


def test_assess_resource_pressure_rejects_cuda_device_mismatch() -> None:
    capacity = ResourceCapacity(
        cuda_total_bytes=1_000,
        cuda_device_index=1,
    )

    with pytest.raises(ValueError, match="cuda_device_index"):
        assess_resource_pressure(
            [_sample(cuda_device_index=0, cuda_allocated_bytes=100)],
            capacity,
        )


class SinglePassSamples:
    def __init__(self) -> None:
        self.iterations = 0

    def __iter__(self) -> Iterator[ResourceSample]:
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("samples iterated more than once")
        yield _sample(cpu_rss_bytes=25)


def test_assess_resource_pressure_consumes_samples_once() -> None:
    samples = SinglePassSamples()

    summary = assess_resource_pressure(
        samples,
        ResourceCapacity(cpu_total_bytes=100),
    )

    assert samples.iterations == 1
    assert summary.peak_cpu_rss_ratio == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"cpu_total_bytes": True}, TypeError),
        ({"cpu_total_bytes": 0}, ValueError),
        ({"cpu_limit_bytes": True}, TypeError),
        ({"cpu_limit_bytes": 0}, ValueError),
        ({"cuda_total_bytes": 100}, ValueError),
        ({"cuda_device_index": 0}, ValueError),
        (
            {"cuda_total_bytes": 100, "cuda_device_index": -1},
            ValueError,
        ),
    ],
)
def test_resource_capacity_rejects_invalid_values(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        ResourceCapacity(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True])
def test_resource_pressure_summary_rejects_invalid_ratios(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ResourcePressureSummary(peak_cpu_rss_ratio=value)  # type: ignore[arg-type]


def test_assess_resource_pressure_rejects_invalid_inputs() -> None:
    sample = _sample(cpu_rss_bytes=1)

    with pytest.raises(TypeError, match="iterable"):
        assess_resource_pressure(sample, ResourceCapacity())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ResourceCapacity"):
        assess_resource_pressure([sample], object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ResourceSample"):
        assess_resource_pressure(  # type: ignore[list-item]
            [object()],
            ResourceCapacity(),
        )


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_assess_resource_pressure_rejects_invalid_observed_bytes(
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        assess_resource_pressure(
            [_sample(cpu_rss_bytes=value)],  # type: ignore[arg-type]
            ResourceCapacity(cpu_total_bytes=100),
        )
