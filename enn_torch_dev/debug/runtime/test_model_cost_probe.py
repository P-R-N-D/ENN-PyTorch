from __future__ import annotations

import pytest
import torch

from enn_torch_dev.runtime import (
    ModelCostProbe,
    ResourceSample,
    RuntimePhase,
    StepResult,
    StepStatus,
)


def _sample(
    phase: str,
    *,
    cpu: int | None = None,
    allocated: int | None = None,
    reserved: int | None = None,
    max_allocated: int | None = None,
    max_reserved: int | None = None,
    device_index: int | None = None,
) -> ResourceSample:
    return ResourceSample(
        timestamp_ns=1,
        phase=phase,
        cpu_rss_bytes=cpu,
        cuda_available=allocated is not None,
        cuda_device_index=(
            device_index
            if device_index is not None
            else 0 if allocated is not None else None
        ),
        cuda_allocated_bytes=allocated,
        cuda_reserved_bytes=reserved,
        cuda_max_allocated_bytes=max_allocated,
        cuda_max_reserved_bytes=max_reserved,
    )


def _result(samples: tuple[ResourceSample, ...]) -> StepResult:
    return StepResult(
        status=StepStatus.SUCCESS,
        phase=None,
        batch_size=2,
        row_ids=torch.tensor([10, 11]),
        resource_samples=samples,
    )


def _cuda_sample(phase: str, *, device_index: object) -> ResourceSample:
    return ResourceSample(
        timestamp_ns=1,
        phase=phase,
        cpu_rss_bytes=100,
        cuda_available=True,
        cuda_device_index=device_index,  # type: ignore[arg-type]
        cuda_allocated_bytes=10,
        cuda_reserved_bytes=20,
        cuda_max_allocated_bytes=30,
        cuda_max_reserved_bytes=40,
    )


def test_model_cost_probe_computes_phase_deltas() -> None:
    result = _result(
        (
            _sample("before_step", cpu=100, allocated=10, reserved=20),
            _sample("after_to_store", cpu=160, allocated=30, reserved=45),
            _sample("after_forward", cpu=220, allocated=70, reserved=90),
        )
    )

    cost = ModelCostProbe().estimate_step(result)

    assert cost.status is StepStatus.SUCCESS
    assert cost.batch_size == 2
    assert cost.row_count == 2
    assert cost.total_cpu_rss_delta_bytes == 120
    assert cost.total_cuda_allocated_delta_bytes == 60
    assert cost.total_cuda_reserved_delta_bytes == 70
    assert len(cost.phase_deltas) == 2
    assert cost.phase_deltas[0].start_phase == "before_step"
    assert cost.phase_deltas[0].end_phase == "after_to_store"
    assert cost.phase_deltas[0].cpu_rss_delta_bytes == 60
    assert cost.phase_deltas[1].cuda_allocated_delta_bytes == 40
    assert cost.cuda_device_index == 0


def test_model_cost_probe_skips_cuda_delta_across_devices() -> None:
    result = _result(
        (
            _sample(
                "before_step",
                cpu=100,
                allocated=10,
                reserved=20,
                max_allocated=30,
                max_reserved=40,
                device_index=0,
            ),
            _sample(
                "after_forward",
                cpu=150,
                allocated=70,
                reserved=90,
                max_allocated=110,
                max_reserved=130,
                device_index=1,
            ),
        )
    )

    cost = ModelCostProbe().estimate_step(result)

    assert cost.total_cpu_rss_delta_bytes == 50
    assert cost.total_cuda_allocated_delta_bytes is None
    assert cost.total_cuda_reserved_delta_bytes is None
    assert cost.total_cuda_max_allocated_delta_bytes is None
    assert cost.total_cuda_max_reserved_delta_bytes is None
    assert cost.phase_deltas[0].cpu_rss_delta_bytes == 50
    assert cost.phase_deltas[0].cuda_allocated_delta_bytes is None
    assert cost.cuda_device_index is None


def test_model_cost_probe_row_count_uses_batch_size_for_multidimensional_row_ids() -> None:
    result = StepResult(
        status=StepStatus.SUCCESS,
        phase=None,
        batch_size=2,
        row_ids=torch.tensor([[10, 100], [11, 101]]),
        resource_samples=(
            _sample("before_step", cpu=10),
            _sample("after_forward", cpu=15),
        ),
    )

    cost = ModelCostProbe().estimate_step(result)

    assert result.row_ids.numel() == 4
    assert cost.row_count == 2
    assert cost.row_count == result.batch_size


def test_model_cost_probe_handles_cuda_none_fields() -> None:
    result = _result(
        (
            _sample("before_step", cpu=100),
            _sample("after_forward", cpu=130),
        )
    )

    cost = ModelCostProbe().estimate_step(result)
    assert cost.total_cpu_rss_delta_bytes == 30
    assert cost.total_cuda_allocated_delta_bytes is None
    assert cost.phase_deltas[0].cuda_allocated_delta_bytes is None
    assert cost.cuda_device_index is None


def test_model_cost_probe_uses_cuda_device_from_cuda_bearing_samples_only() -> None:
    result = _result(
        (
            _sample("before_step", cpu=100, device_index=1),
            _sample(
                "after_forward",
                cpu=130,
                allocated=20,
                reserved=30,
                device_index=0,
            ),
        )
    )

    cost = ModelCostProbe().estimate_step(result)
    assert cost.cuda_device_index == 0


def test_model_cost_probe_rejects_missing_cuda_provenance_between_matching_devices() -> None:
    result = _result(
        (
            _cuda_sample("before_step", device_index=0),
            _cuda_sample("after_to_store", device_index=None),
            _cuda_sample("after_forward", device_index=0),
        )
    )

    assert ModelCostProbe().estimate_step(result).cuda_device_index is None


@pytest.mark.parametrize(
    ("start_index", "end_index"),
    [(None, None), (0, None), (None, 0)],
)
def test_model_cost_probe_requires_concrete_device_for_cuda_deltas(
    start_index: int | None,
    end_index: int | None,
) -> None:
    cost = ModelCostProbe().estimate_step(
        _result(
            (
                _cuda_sample("before_step", device_index=start_index),
                _cuda_sample("after_forward", device_index=end_index),
            )
        )
    )

    assert cost.cuda_device_index is None
    assert cost.total_cuda_allocated_delta_bytes is None
    assert cost.total_cuda_reserved_delta_bytes is None
    assert cost.total_cuda_max_allocated_delta_bytes is None
    assert cost.total_cuda_max_reserved_delta_bytes is None
    phase = cost.phase_deltas[0]
    assert phase.cuda_allocated_delta_bytes is None
    assert phase.cuda_reserved_delta_bytes is None
    assert phase.cuda_max_allocated_delta_bytes is None
    assert phase.cuda_max_reserved_delta_bytes is None


@pytest.mark.parametrize("device_index", [True, False, -1, 1.5, "0"])
def test_model_cost_probe_rejects_invalid_cuda_device_indices(
    device_index: object,
) -> None:
    cost = ModelCostProbe().estimate_step(
        _result(
            (
                _cuda_sample("before_step", device_index=device_index),
                _cuda_sample("after_forward", device_index=device_index),
            )
        )
    )

    assert cost.cuda_device_index is None
    assert cost.total_cuda_allocated_delta_bytes is None


def test_model_cost_probe_preserves_concrete_device_cuda_deltas() -> None:
    start = _cuda_sample("before_step", device_index=0)
    end = ResourceSample(
        timestamp_ns=2,
        phase="after_forward",
        cpu_rss_bytes=150,
        cuda_available=True,
        cuda_device_index=0,
        cuda_allocated_bytes=70,
        cuda_reserved_bytes=90,
        cuda_max_allocated_bytes=110,
        cuda_max_reserved_bytes=130,
    )

    cost = ModelCostProbe().estimate_step(_result((start, end)))
    assert cost.cuda_device_index == 0
    assert cost.total_cuda_allocated_delta_bytes == 60
    assert cost.total_cuda_reserved_delta_bytes == 70
    assert cost.total_cuda_max_allocated_delta_bytes == 80
    assert cost.total_cuda_max_reserved_delta_bytes == 90


def test_model_cost_probe_handles_forward_only_samples() -> None:
    result = _result(
        (
            _sample("before_step", cpu=1),
            _sample("after_to_store", cpu=3),
            _sample("after_forward", cpu=8),
        )
    )

    cost = ModelCostProbe().estimate_step(result)
    assert [(d.start_phase, d.end_phase) for d in cost.phase_deltas] == [
        ("before_step", "after_to_store"),
        ("after_to_store", "after_forward"),
    ]


def test_model_cost_probe_handles_training_samples() -> None:
    result = _result(
        tuple(
            _sample(phase, cpu=index * 10)
            for index, phase in enumerate(
                [
                    "before_step",
                    "after_to_store",
                    "after_zero_grad",
                    "after_forward",
                    "after_loss",
                    "after_backward",
                    "after_optimizer",
                ]
            )
        )
    )

    cost = ModelCostProbe().estimate_step(result)
    assert len(cost.phase_deltas) == 6
    assert cost.total_cpu_rss_delta_bytes == 60
    assert cost.phase_deltas[-1].start_phase == "after_backward"
    assert cost.phase_deltas[-1].end_phase == "after_optimizer"


def test_model_cost_probe_handles_fault_truncated_samples() -> None:
    result = StepResult(
        status=StepStatus.RUNTIME_FAULT,
        phase=RuntimePhase.FORWARD,
        batch_size=2,
        row_ids=torch.tensor([0, 1]),
        resource_samples=(
            _sample("before_step", cpu=10),
            _sample("after_to_store", cpu=20),
        ),
    )

    cost = ModelCostProbe().estimate_step(result)
    assert cost.status is StepStatus.RUNTIME_FAULT
    assert len(cost.phase_deltas) == 1
    assert cost.total_cpu_rss_delta_bytes == 10


def test_model_cost_probe_handles_empty_or_single_sample() -> None:
    empty = ModelCostProbe().estimate_step(_result(()))
    one = ModelCostProbe().estimate_step(_result((_sample("before_step", cpu=1),)))

    assert empty.phase_deltas == ()
    assert empty.total_cpu_rss_delta_bytes is None
    assert one.phase_deltas == ()
    assert one.total_cpu_rss_delta_bytes is None


def test_model_cost_probe_rejects_invalid_input() -> None:
    with pytest.raises(TypeError, match="StepResult"):
        ModelCostProbe().estimate_step(object())  # type: ignore[arg-type]
