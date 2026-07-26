from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest
import torch
from tensordict import TensorDict

import enn_torch
import enn_torch_dev.runtime as runtime_api
from enn_torch_dev.data import KVBatch
from enn_torch_dev.runtime import (
    AdmissionUnknownAction,
    BatchBudget,
    ConservativeRuntimeGovernor,
    ConservativeRuntimeOrchestrator,
    ObservedCostCalibrationPolicy,
    ObservedCostMetricProfile,
    ObservedCostProfile,
    PrePassAdmissionBlocked,
    PrePassAdmissionError,
    PrePassAdmissionGate,
    PrePassAdmissionStatus,
    ResourceCapacity,
    ResourceSample,
    ResourceSampleProvider,
    RetryPolicy,
    RuntimePhase,
    StepResult,
    StepStatus,
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
    cpu: int | None = 0,
    allocated: int | None = None,
    reserved: int | None = None,
    device_index: int | None = None,
) -> ObservedCostProfile:
    return ObservedCostProfile(
        policy=ObservedCostCalibrationPolicy(),
        total_observations=3,
        successful_samples=3,
        ignored_samples=0,
        rejected_samples=0,
        ignored_zero_batch_samples=0,
        ignored_by_status=(),
        min_batch_size=1,
        max_batch_size=4,
        cuda_device_index=device_index,
        cpu_rss=_metric(cpu),
        cuda_allocated=_metric(allocated),
        cuda_reserved=_metric(reserved),
        cuda_max_allocated=_metric(None),
        cuda_max_reserved=_metric(None),
        phase_costs=(),
    )


def _sample(
    *,
    cpu: int | None = 0,
    allocated: int | None = None,
    reserved: int | None = None,
    device_index: int | None = None,
) -> ResourceSample:
    return ResourceSample(
        timestamp_ns=1,
        phase="before_admission",
        cpu_rss_bytes=cpu,
        cuda_available=device_index is not None,
        cuda_device_index=device_index,
        cuda_allocated_bytes=allocated,
        cuda_reserved_bytes=reserved,
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
        schema_id="runtime.admission_gate",
        shard_id=13,
    )


class SequenceSampleProvider:
    def __init__(self, samples: tuple[object, ...]) -> None:
        self.samples = samples
        self.calls: list[str] = []

    def sample(self, phase: str) -> object:
        self.calls.append(phase)
        return self.samples[len(self.calls) - 1]


class FakeRuntimeStep:
    def __init__(self, *, oom_above: int | None = None, optimizer: object = None) -> None:
        self.oom_above = oom_above
        self.optimizer = optimizer
        self.calls: list[KVBatch] = []

    def run(self, batch: KVBatch) -> StepResult:
        self.calls.append(batch)
        is_oom = self.oom_above is not None and batch.batch_size > self.oom_above
        return StepResult(
            status=StepStatus.OOM_FAULT if is_oom else StepStatus.SUCCESS,
            phase=RuntimePhase.FORWARD if is_oom else None,
            batch_size=batch.batch_size,
            row_ids=batch.row_ids.detach().cpu().clone(),
            error_type="SyntheticOOM" if is_oom else None,
            error_message="batch too large" if is_oom else None,
        )


class FixedCapacityProvider:
    def __init__(self, value: object) -> None:
        self.value = value
        self.calls = 0

    def capacity(self) -> object:
        self.calls += 1
        return self.value


def test_resource_sample_provider_protocol_accepts_structural_provider() -> None:
    provider = SequenceSampleProvider((_sample(),))
    assert isinstance(provider, ResourceSampleProvider)


def test_gate_admits_and_samples_exactly_once() -> None:
    provider = SequenceSampleProvider((_sample(cpu=100),))
    gate = PrePassAdmissionGate(
        ResourceCapacity(cpu_total_bytes=1_000),
        _profile(cpu=10),
        provider,
    )

    assessment = gate.check(2)

    assert assessment.status is PrePassAdmissionStatus.ADMIT
    assert provider.calls == ["before_admission"]


def test_gate_rejects_without_retaining_raw_sample() -> None:
    sample = _sample(cpu=90)
    provider = SequenceSampleProvider((sample,))
    gate = PrePassAdmissionGate(
        ResourceCapacity(cpu_total_bytes=100),
        _profile(cpu=1),
        provider,
    )

    with pytest.raises(PrePassAdmissionBlocked) as exc_info:
        gate.check(1)

    blocked = exc_info.value
    assert blocked.assessment.status is PrePassAdmissionStatus.REJECT
    assert set(vars(blocked)) == {"assessment"}
    assert all(value is not sample for value in vars(blocked).values())
    assert not hasattr(gate, "baseline_sample")


def test_gate_blocks_unknown_by_default() -> None:
    gate = PrePassAdmissionGate(
        ResourceCapacity(cpu_total_bytes=1_000),
        _profile(cpu=None),
        SequenceSampleProvider((_sample(cpu=100),)),
    )

    with pytest.raises(PrePassAdmissionBlocked) as exc_info:
        gate.check(1)

    assert exc_info.value.assessment.status is PrePassAdmissionStatus.UNKNOWN


def test_gate_allows_unknown_only_when_explicitly_configured() -> None:
    gate = PrePassAdmissionGate(
        ResourceCapacity(cpu_total_bytes=1_000),
        _profile(cpu=None),
        SequenceSampleProvider((_sample(cpu=100),)),
        unknown_action=AdmissionUnknownAction.ALLOW,
    )

    assessment = gate.check(1)

    assert assessment.status is PrePassAdmissionStatus.UNKNOWN


def test_unknown_allow_never_overrides_reject() -> None:
    gate = PrePassAdmissionGate(
        ResourceCapacity(cpu_total_bytes=100),
        _profile(cpu=None),
        SequenceSampleProvider((_sample(cpu=91),)),
        unknown_action=AdmissionUnknownAction.ALLOW,
    )

    with pytest.raises(PrePassAdmissionBlocked) as exc_info:
        gate.check(1)

    assert exc_info.value.assessment.status is PrePassAdmissionStatus.REJECT


@pytest.mark.parametrize("batch_size", [True, False, 0, -1, 1.5])
def test_gate_rejects_invalid_batch_size_before_sampling(batch_size: object) -> None:
    provider = SequenceSampleProvider((_sample(),))
    gate = PrePassAdmissionGate(
        ResourceCapacity(cpu_total_bytes=1_000),
        _profile(),
        provider,
    )

    with pytest.raises((TypeError, ValueError), match="batch_size"):
        gate.check(batch_size)  # type: ignore[arg-type]

    assert provider.calls == []


def test_gate_rejects_invalid_sample_return_before_assessment() -> None:
    gate = PrePassAdmissionGate(
        ResourceCapacity(cpu_total_bytes=1_000),
        _profile(),
        SequenceSampleProvider((object(),)),
    )

    with pytest.raises(TypeError, match="return a ResourceSample"):
        gate.check(1)


def test_orchestrator_without_gate_preserves_existing_behavior() -> None:
    step = FakeRuntimeStep()
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=2))
    orchestrator = ConservativeRuntimeOrchestrator(step, governor)

    result = orchestrator.run_pass([_batch(3)])

    assert [batch.batch_size for batch in step.calls] == [2, 1]
    assert result.admission_assessments == ()


def test_orchestrator_assesses_each_budgeted_candidate_before_execution() -> None:
    step = FakeRuntimeStep()
    provider = SequenceSampleProvider((_sample(), _sample()))
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=2))
    orchestrator = ConservativeRuntimeOrchestrator(
        step,
        governor,
        resource_capacity=ResourceCapacity(cpu_total_bytes=1_000),
        admission_profile=_profile(cpu=0),
        admission_sample_provider=provider,
    )

    result = orchestrator.run_pass([_batch(4)])

    assert [batch.batch_size for batch in step.calls] == [2, 2]
    assert [item.batch_size for item in result.admission_assessments] == [2, 2]
    assert provider.calls == ["before_admission", "before_admission"]


def test_orchestrator_block_prevents_runtime_and_governor_update() -> None:
    step = FakeRuntimeStep()
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=1))
    orchestrator = ConservativeRuntimeOrchestrator(
        step,
        governor,
        resource_capacity=ResourceCapacity(cpu_total_bytes=100),
        admission_profile=_profile(cpu=1),
        admission_sample_provider=SequenceSampleProvider((_sample(cpu=90),)),
    )

    with pytest.raises(PrePassAdmissionBlocked):
        orchestrator.run_pass([_batch(1)])

    assert step.calls == []
    assert governor.current_budget == BatchBudget(max_items=1)
    assert governor.state.last_decision is None


def test_orchestrator_unknown_allow_executes_and_records_assessment() -> None:
    step = FakeRuntimeStep()
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=1))
    orchestrator = ConservativeRuntimeOrchestrator(
        step,
        governor,
        resource_capacity=ResourceCapacity(cpu_total_bytes=1_000),
        admission_profile=_profile(cpu=None),
        admission_sample_provider=SequenceSampleProvider((_sample(cpu=100),)),
        admission_unknown_action=AdmissionUnknownAction.ALLOW,
    )

    result = orchestrator.run_pass([_batch(1)])

    assert len(step.calls) == 1
    assert [item.status for item in result.admission_assessments] == [
        PrePassAdmissionStatus.UNKNOWN
    ]
    assert governor.state.last_decision is result.decision


def test_oom_retry_original_and_subbatches_are_each_assessed() -> None:
    step = FakeRuntimeStep(oom_above=2)
    provider = SequenceSampleProvider((_sample(), _sample(), _sample()))
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=4))
    orchestrator = ConservativeRuntimeOrchestrator(
        step,
        governor,
        retry_policy=RetryPolicy(max_retry_depth=2, split_factor=2),
        resource_capacity=ResourceCapacity(cpu_total_bytes=1_000),
        admission_profile=_profile(cpu=0),
        admission_sample_provider=provider,
    )

    result = orchestrator.run_pass([_batch(4)])

    assert [batch.batch_size for batch in step.calls] == [4, 2, 2]
    assert [item.batch_size for item in result.admission_assessments] == [4, 2, 2]
    assert [item.status for item in result.admission_assessments] == [
        PrePassAdmissionStatus.ADMIT,
        PrePassAdmissionStatus.ADMIT,
        PrePassAdmissionStatus.ADMIT,
    ]
    assert len(result.results) == 2
    assert result.recovered_oom


def test_optimizer_passthrough_preserves_retry_restriction() -> None:
    step = FakeRuntimeStep(oom_above=2, optimizer=object())
    provider = SequenceSampleProvider((_sample(),))
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=4))
    orchestrator = ConservativeRuntimeOrchestrator(
        step,
        governor,
        retry_policy=RetryPolicy(max_retry_depth=2, split_factor=2),
        resource_capacity=ResourceCapacity(cpu_total_bytes=1_000),
        admission_profile=_profile(cpu=0),
        admission_sample_provider=provider,
    )

    result = orchestrator.run_pass([_batch(4)])

    assert [batch.batch_size for batch in step.calls] == [4]
    assert len(result.admission_assessments) == 1
    assert [item.status for item in result.results] == [StepStatus.OOM_FAULT]


def test_capacity_provider_is_called_once_and_sampler_per_attempt() -> None:
    capacity_provider = FixedCapacityProvider(ResourceCapacity(cpu_total_bytes=1_000))
    sample_provider = SequenceSampleProvider((_sample(), _sample()))
    step = FakeRuntimeStep()
    orchestrator = ConservativeRuntimeOrchestrator(
        step,
        ConservativeRuntimeGovernor(BatchBudget(max_items=1)),
        resource_capacity_provider=capacity_provider,
        admission_profile=_profile(cpu=0),
        admission_sample_provider=sample_provider,
    )

    result = orchestrator.run_pass([_batch(1), _batch(1, offset=10)])

    assert capacity_provider.calls == 1
    assert len(sample_provider.calls) == 2
    assert len(result.admission_assessments) == 2


def test_later_block_stops_future_source_without_governor_update() -> None:
    yielded = 0

    def source() -> Iterator[KVBatch]:
        nonlocal yielded
        for index in range(3):
            yielded += 1
            yield _batch(1, offset=index * 10)

    step = FakeRuntimeStep()
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=1))
    orchestrator = ConservativeRuntimeOrchestrator(
        step,
        governor,
        resource_capacity=ResourceCapacity(cpu_total_bytes=100),
        admission_profile=_profile(cpu=0),
        admission_sample_provider=SequenceSampleProvider(
            (_sample(cpu=0), _sample(cpu=91))
        ),
    )

    with pytest.raises(PrePassAdmissionBlocked):
        orchestrator.run_pass(source())

    assert yielded == 2
    assert len(step.calls) == 1
    assert governor.state.last_decision is None


def test_invalid_capacity_provider_fails_before_source_consumption() -> None:
    yielded = False

    def source() -> Iterator[KVBatch]:
        nonlocal yielded
        yielded = True
        yield _batch(1)

    orchestrator = ConservativeRuntimeOrchestrator(
        FakeRuntimeStep(),
        ConservativeRuntimeGovernor(BatchBudget(max_items=1)),
        resource_capacity_provider=FixedCapacityProvider(object()),
        admission_profile=_profile(cpu=0),
        admission_sample_provider=SequenceSampleProvider((_sample(),)),
    )

    with pytest.raises(TypeError, match="must return a ResourceCapacity"):
        orchestrator.run_pass(source())

    assert not yielded


def test_cuda_provenance_error_prevents_runtime_and_governor_update() -> None:
    step = FakeRuntimeStep()
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=1))
    orchestrator = ConservativeRuntimeOrchestrator(
        step,
        governor,
        resource_capacity=ResourceCapacity(
            cpu_total_bytes=1_000,
            cuda_total_bytes=1_000,
            cuda_device_index=0,
        ),
        admission_profile=_profile(allocated=10, device_index=0),
        admission_sample_provider=SequenceSampleProvider(
            (_sample(allocated=10, device_index=1),)
        ),
    )

    with pytest.raises(PrePassAdmissionError, match="does not match capacity"):
        orchestrator.run_pass([_batch(1)])

    assert step.calls == []
    assert governor.state.last_decision is None


def test_orchestrator_rejects_incomplete_admission_configuration() -> None:
    step = FakeRuntimeStep()
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=1))
    provider = SequenceSampleProvider((_sample(),))

    with pytest.raises(ValueError, match="requires admission_sample_provider"):
        ConservativeRuntimeOrchestrator(
            step,
            governor,
            resource_capacity=ResourceCapacity(cpu_total_bytes=1_000),
            admission_profile=_profile(),
        )
    with pytest.raises(ValueError, match="requires resource_capacity"):
        ConservativeRuntimeOrchestrator(
            step,
            governor,
            admission_profile=_profile(),
            admission_sample_provider=provider,
        )
    with pytest.raises(ValueError, match="require admission_profile"):
        ConservativeRuntimeOrchestrator(
            step,
            governor,
            admission_sample_provider=provider,
        )
    with pytest.raises(ValueError, match="requires admission_profile"):
        ConservativeRuntimeOrchestrator(
            step,
            governor,
            admission_unknown_action=AdmissionUnknownAction.ALLOW,
        )


def test_admission_gate_api_is_development_only() -> None:
    names = {
        "AdmissionUnknownAction",
        "PrePassAdmissionBlocked",
        "PrePassAdmissionGate",
        "ResourceSampleProvider",
    }

    assert names <= set(runtime_api.__all__)
    for name in names:
        assert getattr(runtime_api, name) is not None
        assert name not in set(enn_torch.__all__)
        with pytest.raises(AttributeError):
            getattr(enn_torch, name)
