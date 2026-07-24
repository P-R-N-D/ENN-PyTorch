from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import fields

import pytest
import torch
from tensordict import TensorDict

import enn_torch
import enn_torch_dev.runtime as runtime_api
from enn_torch_dev.data import KVBatch
from enn_torch_dev.runtime import (
    BatchBudget,
    ConservativeRuntimeGovernor,
    ConservativeRuntimeOrchestrator,
    ConservativeRuntimeSession,
    GovernorPolicy,
    ResourceCapacity,
    ResourceCapacityProvider,
    ResourceMonitor,
    ResourceSample,
    RetryPolicy,
    RuntimePassHistory,
    RuntimePassResult,
    RuntimePassSummary,
    RuntimePhase,
    StepResult,
    StepStatus,
    format_runtime_pass_summary,
    summarize_runtime_pass,
)


def _batch(num_rows: int = 1, *, offset: int = 0) -> KVBatch:
    td = TensorDict(
        {
            "features": torch.arange(
                offset,
                offset + num_rows * 2,
                dtype=torch.float32,
            ).reshape(num_rows, 2),
        },
        batch_size=(num_rows,),
    )
    return KVBatch(
        td=td,
        row_ids=torch.arange(offset, offset + num_rows),
        source_ids=torch.arange(100 + offset, 100 + offset + num_rows),
        sample_ids=torch.arange(200 + offset, 200 + offset + num_rows),
        schema_id="runtime.capacity-provider",
        shard_id=13,
    )


def _sampled_result(
    batch: KVBatch,
    *,
    cpu_rss_bytes: int,
    status: StepStatus = StepStatus.SUCCESS,
) -> StepResult:
    return StepResult(
        status=status,
        phase=RuntimePhase.FORWARD if status is StepStatus.OOM_FAULT else None,
        batch_size=batch.batch_size,
        row_ids=batch.row_ids.detach().cpu().clone(),
        error_type=None if status is StepStatus.SUCCESS else "SyntheticOOM",
        error_message=None if status is StepStatus.SUCCESS else "batch too large",
        resource_samples=(
            ResourceSample(
                timestamp_ns=1,
                phase="capacity-provider-test",
                cpu_rss_bytes=cpu_rss_bytes,
            ),
        ),
    )


class FakeRuntimeStep:
    def __init__(self, fn: Callable[[KVBatch], StepResult]) -> None:
        self.fn = fn
        self.calls: list[KVBatch] = []

    def run(self, batch: KVBatch) -> StepResult:
        self.calls.append(batch)
        return self.fn(batch)


class SequenceCapacityProvider:
    def __init__(self, capacities: tuple[ResourceCapacity, ...]) -> None:
        self.capacities = capacities
        self.calls = 0

    def capacity(self) -> ResourceCapacity:
        capacity = self.capacities[self.calls]
        self.calls += 1
        return capacity


class RaisingCapacityProvider:
    def __init__(self) -> None:
        self.calls = 0

    def capacity(self) -> ResourceCapacity:
        self.calls += 1
        raise RuntimeError("capacity lookup failed")


class InvalidCapacityProvider:
    def __init__(self) -> None:
        self.calls = 0

    def capacity(self) -> ResourceCapacity:
        self.calls += 1
        return object()  # type: ignore[return-value]


class TrackingSource:
    def __init__(self, batch: KVBatch) -> None:
        self.batch = batch
        self.iterated = False

    def __iter__(self) -> Iterator[KVBatch]:
        self.iterated = True
        yield self.batch


def test_resource_monitor_satisfies_capacity_provider_protocol() -> None:
    assert isinstance(ResourceMonitor(), ResourceCapacityProvider)
    assert isinstance(
        SequenceCapacityProvider((ResourceCapacity(cpu_total_bytes=100),)),
        ResourceCapacityProvider,
    )


def test_capacity_provider_and_provenance_fields_are_appended_for_compatibility() -> None:
    assert [field.name for field in fields(RuntimePassResult)][-1] == "resource_capacity"
    assert [field.name for field in fields(RuntimePassSummary)][-6:] == [
        "resource_capacity",
        "consecutive_high_pressure_passes",
        "budget_shrunk_by_pressure",
        "pressure_shrunk_budget_fields",
        "consecutive_cpu_pressure_passes",
        "consecutive_cuda_pressure_passes",
    ]


def test_no_capacity_path_preserves_none_provenance() -> None:
    step = FakeRuntimeStep(
        lambda batch: _sampled_result(batch, cpu_rss_bytes=50)
    )
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=2))
    orchestrator = ConservativeRuntimeOrchestrator(step, governor)

    pass_result = orchestrator.run_pass([_batch()])

    assert pass_result.resource_capacity is None
    assert pass_result.decision.pressure_summary is None
    assert summarize_runtime_pass(pass_result).resource_capacity is None


def test_fixed_capacity_path_records_pass_and_summary_provenance() -> None:
    capacity = ResourceCapacity(cpu_total_bytes=100, cpu_limit_bytes=80)
    step = FakeRuntimeStep(
        lambda batch: _sampled_result(batch, cpu_rss_bytes=40)
    )
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=2),
        policy=GovernorPolicy(
            grow_after_successes=1,
            max_pressure_ratio_for_growth=0.8,
        ),
    )
    orchestrator = ConservativeRuntimeOrchestrator(
        step,
        governor,
        resource_capacity=capacity,
    )

    pass_result = orchestrator.run_pass([_batch()])
    summary = summarize_runtime_pass(pass_result)

    assert pass_result.resource_capacity == capacity
    assert pass_result.decision.pressure_summary is not None
    assert pass_result.decision.pressure_summary.peak_cpu_rss_ratio == 0.5
    assert summary.resource_capacity == capacity
    assert f"resource_capacity={capacity!r}" in format_runtime_pass_summary(summary)


def test_provider_resolves_once_per_pass_and_refreshes_between_passes() -> None:
    first_capacity = ResourceCapacity(cpu_total_bytes=100)
    second_capacity = ResourceCapacity(cpu_total_bytes=200)
    provider = SequenceCapacityProvider((first_capacity, second_capacity))
    step = FakeRuntimeStep(
        lambda batch: _sampled_result(batch, cpu_rss_bytes=50)
    )
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=2),
        policy=GovernorPolicy(
            grow_after_successes=10,
            max_pressure_ratio_for_growth=0.8,
        ),
    )
    orchestrator = ConservativeRuntimeOrchestrator(
        step,
        governor,
        resource_capacity_provider=provider,
    )

    first = orchestrator.run_pass([_batch(offset=0)])
    second = orchestrator.run_pass([_batch(offset=10)])

    assert provider.calls == 2
    assert first.resource_capacity == first_capacity
    assert second.resource_capacity == second_capacity
    assert first.decision.pressure_summary is not None
    assert second.decision.pressure_summary is not None
    assert first.decision.pressure_summary.peak_cpu_rss_ratio == 0.5
    assert second.decision.pressure_summary.peak_cpu_rss_ratio == 0.25


def test_provider_is_called_once_for_retry_and_split_attempts() -> None:
    provider = SequenceCapacityProvider((ResourceCapacity(cpu_total_bytes=100),))

    def run(batch: KVBatch) -> StepResult:
        if batch.batch_size > 2:
            return _sampled_result(
                batch,
                cpu_rss_bytes=90,
                status=StepStatus.OOM_FAULT,
            )
        return _sampled_result(batch, cpu_rss_bytes=20)

    step = FakeRuntimeStep(run)
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=4),
        policy=GovernorPolicy(shrink_factor=0.5),
    )
    orchestrator = ConservativeRuntimeOrchestrator(
        step,
        governor,
        retry_policy=RetryPolicy(max_retry_depth=2, split_factor=2),
        resource_capacity_provider=provider,
    )

    pass_result = orchestrator.run_pass([_batch(4)])

    assert provider.calls == 1
    assert [call.batch_size for call in step.calls] == [4, 2, 2]
    assert pass_result.recovered_oom is True
    assert pass_result.resource_capacity == ResourceCapacity(cpu_total_bytes=100)
    assert pass_result.decision.pressure_summary is not None
    assert pass_result.decision.pressure_summary.peak_cpu_rss_ratio == 0.9


@pytest.mark.parametrize(
    ("provider", "error_type", "message"),
    [
        (RaisingCapacityProvider(), RuntimeError, "capacity lookup failed"),
        (InvalidCapacityProvider(), TypeError, "must return a ResourceCapacity"),
    ],
)
def test_provider_failures_precede_source_consumption_and_governor_update(
    provider: ResourceCapacityProvider,
    error_type: type[Exception],
    message: str,
) -> None:
    source = TrackingSource(_batch())
    step = FakeRuntimeStep(
        lambda batch: _sampled_result(batch, cpu_rss_bytes=50)
    )
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=2))
    orchestrator = ConservativeRuntimeOrchestrator(
        step,
        governor,
        resource_capacity_provider=provider,
    )

    with pytest.raises(error_type, match=message):
        orchestrator.run_pass(source)

    assert source.iterated is False
    assert step.calls == []
    assert governor.current_budget == BatchBudget(max_items=2)
    assert governor.state.last_decision is None


def test_orchestrator_rejects_invalid_or_conflicting_provider_configuration() -> None:
    step = FakeRuntimeStep(
        lambda batch: _sampled_result(batch, cpu_rss_bytes=50)
    )
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=2))

    with pytest.raises(TypeError, match="resource_capacity_provider"):
        ConservativeRuntimeOrchestrator(
            step,
            governor,
            resource_capacity_provider=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        ConservativeRuntimeOrchestrator(
            step,
            governor,
            resource_capacity=ResourceCapacity(cpu_total_bytes=100),
            resource_capacity_provider=SequenceCapacityProvider(
                (ResourceCapacity(cpu_total_bytes=100),)
            ),
        )


def test_session_records_provider_capacity_per_pass() -> None:
    first_capacity = ResourceCapacity(cpu_total_bytes=100)
    second_capacity = ResourceCapacity(cpu_total_bytes=50)
    provider = SequenceCapacityProvider((first_capacity, second_capacity))
    step = FakeRuntimeStep(
        lambda batch: _sampled_result(batch, cpu_rss_bytes=50)
    )
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=2),
        policy=GovernorPolicy(
            grow_factor=2.0,
            grow_after_successes=1,
            max_items=8,
            max_pressure_ratio_for_growth=0.8,
        ),
    )
    orchestrator = ConservativeRuntimeOrchestrator(
        step,
        governor,
        resource_capacity_provider=provider,
    )
    history = RuntimePassHistory(max_records=2)
    session = ConservativeRuntimeSession(
        orchestrator,
        history,
        max_passes=2,
    )

    first, second = list(
        session.run_passes(
            [
                [_batch(offset=0)],
                [_batch(offset=10)],
            ]
        )
    )

    assert provider.calls == 2
    assert first.pass_result.resource_capacity == first_capacity
    assert first.pass_summary.resource_capacity == first_capacity
    assert first.pass_result.decision.next_budget == BatchBudget(max_items=4)
    assert second.pass_result.resource_capacity == second_capacity
    assert second.pass_summary.resource_capacity == second_capacity
    assert second.pass_result.decision.next_budget == BatchBudget(max_items=4)
    assert second.pass_result.decision.growth_suppressed_by_pressure is True
    assert history.records == (first.pass_summary, second.pass_summary)


def test_capacity_provider_is_dev_only_public_api() -> None:
    assert "ResourceCapacityProvider" in runtime_api.__all__
    assert runtime_api.ResourceCapacityProvider is ResourceCapacityProvider
    assert "ResourceCapacityProvider" not in enn_torch.__all__
    with pytest.raises(AttributeError):
        getattr(enn_torch, "ResourceCapacityProvider")
