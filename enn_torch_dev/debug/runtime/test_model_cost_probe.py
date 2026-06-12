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
) -> ResourceSample:
    return ResourceSample(
        timestamp_ns=1,
        phase=phase,
        cpu_rss_bytes=cpu,
        cuda_available=allocated is not None,
        cuda_device_index=0 if allocated is not None else None,
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
