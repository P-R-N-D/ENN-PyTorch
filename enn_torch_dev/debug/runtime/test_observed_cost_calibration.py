from __future__ import annotations

import pytest
import torch

from enn_torch_dev.runtime import (
    ModelCost,
    ModelCostProbe,
    ObservedCostCalibrationError,
    ObservedCostCalibrationPolicy,
    ObservedCostCalibrator,
    ResourceDelta,
    ResourceSample,
    StepResult,
    StepStatus,
)


def _cost(
    *,
    status: StepStatus = StepStatus.SUCCESS,
    batch_size: int = 2,
    cpu: int | None = None,
    allocated: int | None = None,
    reserved: int | None = None,
    max_allocated: int | None = None,
    max_reserved: int | None = None,
    device_index: int | None = None,
    phases: tuple[ResourceDelta, ...] = (),
    row_count: int | None = None,
) -> ModelCost:
    return ModelCost(
        status=status,
        batch_size=batch_size,
        row_count=batch_size if row_count is None else row_count,
        total_cpu_rss_delta_bytes=cpu,
        total_cuda_allocated_delta_bytes=allocated,
        total_cuda_reserved_delta_bytes=reserved,
        total_cuda_max_allocated_delta_bytes=max_allocated,
        total_cuda_max_reserved_delta_bytes=max_reserved,
        phase_deltas=phases,
        cuda_device_index=device_index,
    )


def test_calibrator_builds_conservative_ceiling_divided_envelope() -> None:
    calibrator = ObservedCostCalibrator()

    assert calibrator.observe(_cost(batch_size=2, cpu=5, allocated=9, device_index=0))
    assert calibrator.observe(_cost(batch_size=3, cpu=10, allocated=12, device_index=0))

    profile = calibrator.profile()
    assert profile.successful_samples == 2
    assert profile.min_batch_size == 2
    assert profile.max_batch_size == 3
    assert profile.cuda_device_index == 0
    assert profile.cpu_rss.max_bytes_per_item == 4
    assert profile.cuda_allocated.max_bytes_per_item == 5


def test_calibrator_distinguishes_unknown_from_observed_zero() -> None:
    calibrator = ObservedCostCalibrator()
    calibrator.observe(_cost(cpu=None))
    calibrator.observe(_cost(cpu=0))

    metric = calibrator.profile().cpu_rss
    assert metric.max_bytes_per_item == 0
    assert metric.known_samples == 1
    assert metric.unknown_samples == 1
    assert metric.zero_samples == 1
    assert metric.negative_deltas_clamped == 0


def test_calibrator_clamps_negative_delta_to_zero_without_counting_observed_zero() -> None:
    calibrator = ObservedCostCalibrator()
    calibrator.observe(_cost(cpu=-7))

    metric = calibrator.profile().cpu_rss
    assert metric.max_bytes_per_item == 0
    assert metric.known_samples == 1
    assert metric.zero_samples == 0
    assert metric.negative_deltas_clamped == 1


@pytest.mark.parametrize(
    "status",
    [
        StepStatus.OOM_FAULT,
        StepStatus.NONFINITE_FAULT,
        StepStatus.DATA_FAULT,
        StepStatus.RUNTIME_FAULT,
    ],
)
def test_calibrator_ignores_fault_costs_and_records_status(status: StepStatus) -> None:
    calibrator = ObservedCostCalibrator()
    assert calibrator.observe(_cost(status=status, cpu=100)) is False
    calibrator.observe(_cost(cpu=10))

    profile = calibrator.profile()
    assert profile.total_observations == 2
    assert profile.successful_samples == 1
    assert profile.ignored_samples == 1
    assert profile.ignored_by_status == ((status.value, 1),)
    assert profile.cpu_rss.max_bytes_per_item == 5


def test_calibrator_ignores_zero_batch_success() -> None:
    calibrator = ObservedCostCalibrator()
    assert calibrator.observe(_cost(batch_size=0, cpu=10)) is False
    calibrator.observe(_cost(batch_size=1, cpu=3))

    profile = calibrator.profile()
    assert profile.ignored_zero_batch_samples == 1
    assert profile.ignored_samples == 1
    assert profile.min_batch_size == 1


def test_calibrator_requires_configured_success_floor() -> None:
    calibrator = ObservedCostCalibrator(
        ObservedCostCalibrationPolicy(min_successful_samples=2)
    )
    calibrator.observe(_cost(cpu=1))

    with pytest.raises(ObservedCostCalibrationError, match="not enough"):
        calibrator.profile()


def test_calibrator_rejects_cuda_metrics_without_device_provenance() -> None:
    calibrator = ObservedCostCalibrator()
    with pytest.raises(ObservedCostCalibrationError, match="device index"):
        calibrator.observe(_cost(allocated=10))


def test_probe_missing_cuda_provenance_is_rejected_without_partial_calibration() -> None:
    calibrator = ObservedCostCalibrator()
    calibrator.observe(
        _cost(
            cpu=4,
            phases=(ResourceDelta("accepted", "phase", cpu_rss_delta_bytes=2),),
        )
    )
    samples = tuple(
        ResourceSample(
            timestamp_ns=index,
            phase=phase,
            cpu_rss_bytes=100 + index,
            cuda_available=True,
            cuda_device_index=device_index,
            cuda_allocated_bytes=10 + index,
            cuda_reserved_bytes=20 + index,
            cuda_max_allocated_bytes=30 + index,
            cuda_max_reserved_bytes=40 + index,
        )
        for index, (phase, device_index) in enumerate(
            (("before_step", 0), ("missing_device", None), ("after_forward", 0))
        )
    )
    cost = ModelCostProbe().estimate_step(
        StepResult(
            status=StepStatus.SUCCESS,
            phase=None,
            batch_size=2,
            row_ids=torch.tensor([0, 1]),
            resource_samples=samples,
        )
    )

    assert cost.cuda_device_index is None
    assert cost.total_cuda_allocated_delta_bytes == 2
    with pytest.raises(ObservedCostCalibrationError, match="device index"):
        calibrator.observe(cost)

    profile = calibrator.profile()
    assert profile.total_observations == 2
    assert profile.successful_samples == 1
    assert profile.rejected_samples == 1
    assert profile.cpu_rss.max_bytes_per_item == 2
    assert [(phase.start_phase, phase.end_phase) for phase in profile.phase_costs] == [
        ("accepted", "phase")
    ]


def test_calibrator_rejects_cuda_device_mismatch_without_partial_acceptance() -> None:
    calibrator = ObservedCostCalibrator()
    calibrator.observe(_cost(allocated=8, device_index=0))

    with pytest.raises(ObservedCostCalibrationError, match="different devices"):
        calibrator.observe(_cost(allocated=20, device_index=1))

    profile = calibrator.profile()
    assert profile.total_observations == 2
    assert profile.successful_samples == 1
    assert profile.rejected_samples == 1
    assert profile.cuda_device_index == 0
    assert profile.cuda_allocated.max_bytes_per_item == 4


def test_calibrator_enforces_expected_cuda_device() -> None:
    calibrator = ObservedCostCalibrator(
        ObservedCostCalibrationPolicy(expected_cuda_device_index=1)
    )
    with pytest.raises(ObservedCostCalibrationError, match="policy device"):
        calibrator.observe(_cost(allocated=4, device_index=0))


def test_calibrator_aggregates_phase_envelopes_deterministically() -> None:
    phases_a = (
        ResourceDelta(
            "before_step",
            "after_forward",
            cpu_rss_delta_bytes=5,
            cuda_allocated_delta_bytes=7,
        ),
    )
    phases_b = (
        ResourceDelta(
            "before_step",
            "after_forward",
            cpu_rss_delta_bytes=10,
            cuda_allocated_delta_bytes=8,
        ),
    )
    calibrator = ObservedCostCalibrator()
    calibrator.observe(_cost(batch_size=2, device_index=0, phases=phases_a))
    calibrator.observe(_cost(batch_size=4, device_index=0, phases=phases_b))

    profile = calibrator.profile()
    assert len(profile.phase_costs) == 1
    phase = profile.phase_costs[0]
    assert (phase.start_phase, phase.end_phase) == (
        "before_step",
        "after_forward",
    )
    assert phase.cpu_rss.max_bytes_per_item == 3
    assert phase.cuda_allocated.max_bytes_per_item == 4


def test_calibrator_rejects_phase_pair_growth_above_bound() -> None:
    calibrator = ObservedCostCalibrator(
        ObservedCostCalibrationPolicy(max_phase_pairs=1)
    )
    calibrator.observe(
        _cost(
            phases=(ResourceDelta("before", "middle", cpu_rss_delta_bytes=1),),
        )
    )

    with pytest.raises(ObservedCostCalibrationError, match="max_phase_pairs"):
        calibrator.observe(
            _cost(
                phases=(ResourceDelta("middle", "after", cpu_rss_delta_bytes=2),),
            )
        )

    assert len(calibrator.profile().phase_costs) == 1


def test_calibrator_rejects_duplicate_phase_pairs_in_one_observation() -> None:
    calibrator = ObservedCostCalibrator()
    repeated = ResourceDelta("before", "after", cpu_rss_delta_bytes=1)
    with pytest.raises(ObservedCostCalibrationError, match="repeat"):
        calibrator.observe(_cost(phases=(repeated, repeated)))


def test_calibrator_rejects_invalid_phase_names() -> None:
    calibrator = ObservedCostCalibrator()
    with pytest.raises(ValueError, match="normalized"):
        calibrator.observe(
            _cost(phases=(ResourceDelta(" before", "after", cpu_rss_delta_bytes=1),))
        )


def test_calibrator_rejects_row_count_mismatch() -> None:
    calibrator = ObservedCostCalibrator()
    with pytest.raises(ObservedCostCalibrationError, match="must match"):
        calibrator.observe(_cost(batch_size=2, row_count=3, cpu=1))


def test_calibrator_does_not_retain_raw_model_cost_objects() -> None:
    calibrator = ObservedCostCalibrator()
    cost = _cost(cpu=4)
    calibrator.observe(cost)

    assert not hasattr(calibrator, "__dict__")
    assert not hasattr(calibrator, "_costs")
    assert calibrator.profile().cpu_rss.max_bytes_per_item == 2


def test_calibration_is_deterministic_for_the_same_observation_order() -> None:
    costs = (
        _cost(batch_size=3, cpu=7),
        _cost(batch_size=2, cpu=8),
        _cost(status=StepStatus.DATA_FAULT, cpu=100),
    )
    first = ObservedCostCalibrator()
    second = ObservedCostCalibrator()
    for cost in costs:
        first.observe(cost)
        second.observe(cost)

    assert first.profile() == second.profile()


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"min_successful_samples": True}, TypeError, "min_successful_samples"),
        ({"min_successful_samples": 0}, ValueError, "min_successful_samples"),
        ({"max_phase_pairs": 0}, ValueError, "max_phase_pairs"),
        ({"expected_cuda_device_index": True}, TypeError, "expected_cuda_device_index"),
        ({"expected_cuda_device_index": -1}, ValueError, "expected_cuda_device_index"),
    ],
)
def test_calibration_policy_rejects_invalid_values(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        ObservedCostCalibrationPolicy(**kwargs)  # type: ignore[arg-type]


def test_calibrator_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="policy"):
        ObservedCostCalibrator(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ModelCost"):
        ObservedCostCalibrator().observe(object())  # type: ignore[arg-type]
