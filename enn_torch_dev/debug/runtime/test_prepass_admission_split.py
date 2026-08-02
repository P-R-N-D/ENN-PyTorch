from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from tensordict import TensorDict

import enn_torch
import enn_torch_dev.runtime as runtime_api
from enn_torch_dev.data import KVBatch
from enn_torch_dev.runtime import (
    AdmissionSplitPolicy,
    AdmissionUnknownAction,
    BatchBudget,
    ConservativeRuntimeGovernor,
    ConservativeRuntimeOrchestrator,
    GovernorPolicy,
    ObservedCostCalibrationPolicy,
    ObservedCostMetricProfile,
    ObservedCostProfile,
    PrePassAdmissionAssessment,
    PrePassAdmissionBlocked,
    PrePassAdmissionPolicy,
    PrePassAdmissionStatus,
    ResourceCapacity,
    ResourceSample,
    RetryPolicy,
    RuntimePhase,
    RuntimeRetryRunner,
    StepResult,
    StepStatus,
)


def _batch(num_rows: int, *, offset: int = 0) -> KVBatch:
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
        schema_id="runtime.admission_split",
        shard_id=17,
    )


def _result(
    batch: KVBatch,
    status: StepStatus = StepStatus.SUCCESS,
    *,
    phase: RuntimePhase | None = None,
) -> StepResult:
    return StepResult(
        status=status,
        phase=phase,
        batch_size=batch.batch_size,
        row_ids=batch.row_ids.detach().cpu().clone(),
        error_type=None if status is StepStatus.SUCCESS else status.name,
        error_message=None if status is StepStatus.SUCCESS else status.value,
    )


def _assessment(
    batch_size: int,
    *,
    status: PrePassAdmissionStatus = PrePassAdmissionStatus.REJECT,
    max_admissible_items: int | None = None,
) -> PrePassAdmissionAssessment:
    return PrePassAdmissionAssessment(
        status=status,
        batch_size=batch_size,
        policy=PrePassAdmissionPolicy(),
        profile_successful_samples=3,
        cuda_device_index=None,
        dimensions=(),
        rejected_dimensions=("cpu_rss",)
        if status is PrePassAdmissionStatus.REJECT
        else (),
        unknown_dimensions=("cpu_rss",)
        if status is PrePassAdmissionStatus.UNKNOWN
        else (),
        max_admissible_items=max_admissible_items,
        warnings=(),
    )


class SyntheticAdmissionStep:
    def __init__(
        self,
        fn: Callable[[KVBatch], StepResult | PrePassAdmissionAssessment],
        *,
        optimizer: object | None = None,
    ) -> None:
        self.fn = fn
        self.optimizer = optimizer
        self.calls: list[KVBatch] = []
        self.blocked: list[PrePassAdmissionBlocked] = []

    def run(self, batch: KVBatch) -> StepResult:
        self.calls.append(batch)
        outcome = self.fn(batch)
        if isinstance(outcome, PrePassAdmissionAssessment):
            blocked = PrePassAdmissionBlocked(outcome)
            self.blocked.append(blocked)
            raise blocked
        return outcome


def _metric(value: int | None) -> ObservedCostMetricProfile:
    return ObservedCostMetricProfile(
        max_bytes_per_item=value,
        known_samples=0 if value is None else 1,
        unknown_samples=1 if value is None else 0,
        zero_samples=1 if value == 0 else 0,
        negative_deltas_clamped=0,
    )


def _profile(*, cpu: int | None = 100) -> ObservedCostProfile:
    return ObservedCostProfile(
        policy=ObservedCostCalibrationPolicy(),
        total_observations=3,
        successful_samples=3,
        ignored_samples=0,
        rejected_samples=0,
        ignored_zero_batch_samples=0,
        ignored_by_status=(),
        min_batch_size=1,
        max_batch_size=8,
        cuda_device_index=None,
        cpu_rss=_metric(cpu),
        cuda_allocated=_metric(None),
        cuda_reserved=_metric(None),
        cuda_max_allocated=_metric(None),
        cuda_max_reserved=_metric(None),
        phase_costs=(),
    )


def _sample(cpu: int | None) -> ResourceSample:
    return ResourceSample(
        timestamp_ns=1,
        phase="before_admission",
        cpu_rss_bytes=cpu,
    )


class SequenceSampleProvider:
    def __init__(self, samples: tuple[ResourceSample, ...]) -> None:
        self.samples = samples
        self.calls: list[str] = []

    def sample(self, phase: str) -> ResourceSample:
        self.calls.append(phase)
        return self.samples[len(self.calls) - 1]


class FixedCapacityProvider:
    def __init__(self, capacity: ResourceCapacity) -> None:
        self.value = capacity
        self.calls = 0

    def capacity(self) -> ResourceCapacity:
        self.calls += 1
        return self.value


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"max_split_depth": True}, TypeError),
        ({"max_split_depth": -1}, ValueError),
        ({"min_items": True}, TypeError),
        ({"min_items": 0}, ValueError),
        ({"max_split_parts": True}, TypeError),
        ({"max_split_parts": 1}, ValueError),
    ],
)
def test_admission_split_policy_rejects_invalid_values(
    kwargs: dict[str, object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        AdmissionSplitPolicy(**kwargs)  # type: ignore[arg-type]


def test_reject_remains_terminal_without_split_policy() -> None:
    step = SyntheticAdmissionStep(
        lambda batch: _assessment(batch.batch_size, max_admissible_items=2)
    )

    with pytest.raises(PrePassAdmissionBlocked):
        list(RuntimeRetryRunner(step).run_batch(_batch(4)))

    assert [batch.batch_size for batch in step.calls] == [4]


def test_reject_splits_to_assessment_limit_with_balanced_parts() -> None:
    def run(batch: KVBatch) -> StepResult | PrePassAdmissionAssessment:
        if batch.batch_size > 3:
            return _assessment(batch.batch_size, max_admissible_items=3)
        return _result(batch)

    step = SyntheticAdmissionStep(run)
    original = _batch(10)
    results = list(
        RuntimeRetryRunner(
            step,
            admission_split_policy=AdmissionSplitPolicy(
                min_items=2,
                max_split_parts=4,
            ),
        ).run_batch(original)
    )

    assert [batch.batch_size for batch in step.calls] == [10, 3, 3, 2, 2]
    assert [result.batch_size for result in results] == [3, 3, 2, 2]
    assert torch.equal(torch.cat([result.row_ids for result in results]), original.row_ids)
    children = step.calls[1:]
    assert torch.equal(torch.cat([child.row_ids for child in children]), original.row_ids)
    assert torch.equal(
        torch.cat([child.source_ids for child in children if child.source_ids is not None]),
        original.source_ids,
    )
    assert torch.equal(
        torch.cat([child.sample_ids for child in children if child.sample_ids is not None]),
        original.sample_ids,
    )


def test_unknown_is_never_split_even_with_a_numeric_limit() -> None:
    step = SyntheticAdmissionStep(
        lambda batch: _assessment(
            batch.batch_size,
            status=PrePassAdmissionStatus.UNKNOWN,
            max_admissible_items=2,
        )
    )

    with pytest.raises(PrePassAdmissionBlocked) as exc_info:
        list(
            RuntimeRetryRunner(
                step,
                admission_split_policy=AdmissionSplitPolicy(),
            ).run_batch(_batch(4))
        )

    assert exc_info.value.assessment.status is PrePassAdmissionStatus.UNKNOWN
    assert [batch.batch_size for batch in step.calls] == [4]


@pytest.mark.parametrize("target", [None, 0, -1, 4, 5, True])
def test_invalid_or_non_reducing_targets_remain_terminal(target: object) -> None:
    step = SyntheticAdmissionStep(
        lambda batch: _assessment(
            batch.batch_size,
            max_admissible_items=target,  # type: ignore[arg-type]
        )
    )

    with pytest.raises(PrePassAdmissionBlocked):
        list(
            RuntimeRetryRunner(
                step,
                admission_split_policy=AdmissionSplitPolicy(),
            ).run_batch(_batch(4))
        )

    assert [batch.batch_size for batch in step.calls] == [4]


def test_assessment_batch_size_mismatch_remains_terminal() -> None:
    step = SyntheticAdmissionStep(
        lambda batch: _assessment(99, max_admissible_items=2)
    )

    with pytest.raises(PrePassAdmissionBlocked):
        list(
            RuntimeRetryRunner(
                step,
                admission_split_policy=AdmissionSplitPolicy(),
            ).run_batch(_batch(4))
        )


def test_split_is_terminal_when_min_items_cannot_cover_every_row() -> None:
    step = SyntheticAdmissionStep(
        lambda batch: _assessment(batch.batch_size, max_admissible_items=2)
    )

    with pytest.raises(PrePassAdmissionBlocked):
        list(
            RuntimeRetryRunner(
                step,
                admission_split_policy=AdmissionSplitPolicy(min_items=2),
            ).run_batch(_batch(5))
        )

    assert [batch.batch_size for batch in step.calls] == [5]


def test_split_is_terminal_when_required_parts_exceed_limit() -> None:
    step = SyntheticAdmissionStep(
        lambda batch: _assessment(batch.batch_size, max_admissible_items=1)
    )

    with pytest.raises(PrePassAdmissionBlocked):
        list(
            RuntimeRetryRunner(
                step,
                admission_split_policy=AdmissionSplitPolicy(max_split_parts=4),
            ).run_batch(_batch(10))
        )


def test_split_stops_at_admission_depth_without_consuming_later_siblings() -> None:
    step = SyntheticAdmissionStep(
        lambda batch: _assessment(
            batch.batch_size,
            max_admissible_items=batch.batch_size // 2,
        )
    )

    with pytest.raises(PrePassAdmissionBlocked):
        list(
            RuntimeRetryRunner(
                step,
                admission_split_policy=AdmissionSplitPolicy(max_split_depth=1),
            ).run_batch(_batch(8))
        )

    assert [batch.batch_size for batch in step.calls] == [8, 4]
    assert step.blocked[0].__traceback__ is None
    assert step.blocked[1].__traceback__ is not None


def test_recovered_internal_block_clears_its_traceback() -> None:
    def run(batch: KVBatch) -> StepResult | PrePassAdmissionAssessment:
        if batch.batch_size > 2:
            return _assessment(batch.batch_size, max_admissible_items=2)
        return _result(batch)

    step = SyntheticAdmissionStep(run)
    list(
        RuntimeRetryRunner(
            step,
            admission_split_policy=AdmissionSplitPolicy(),
        ).run_batch(_batch(4))
    )

    assert len(step.blocked) == 1
    assert step.blocked[0].__traceback__ is None


def test_admission_split_then_oom_retry_use_independent_depths() -> None:
    def run(batch: KVBatch) -> StepResult | PrePassAdmissionAssessment:
        if batch.batch_size > 4:
            return _assessment(batch.batch_size, max_admissible_items=4)
        if batch.batch_size > 2:
            return _result(batch, StepStatus.OOM_FAULT, phase=RuntimePhase.FORWARD)
        return _result(batch)

    step = SyntheticAdmissionStep(run)
    results = list(
        RuntimeRetryRunner(
            step,
            policy=RetryPolicy(max_retry_depth=1, split_factor=2),
            admission_split_policy=AdmissionSplitPolicy(max_split_depth=1),
        ).run_batch(_batch(8))
    )

    assert [batch.batch_size for batch in step.calls] == [8, 4, 2, 2, 4, 2, 2]
    assert [result.batch_size for result in results] == [2, 2, 2, 2]


def test_oom_retry_subbatches_can_use_admission_split() -> None:
    def run(batch: KVBatch) -> StepResult | PrePassAdmissionAssessment:
        if batch.batch_size == 8:
            return _result(batch, StepStatus.OOM_FAULT, phase=RuntimePhase.FORWARD)
        if batch.batch_size == 4:
            return _assessment(batch.batch_size, max_admissible_items=2)
        return _result(batch)

    step = SyntheticAdmissionStep(run)
    results = list(
        RuntimeRetryRunner(
            step,
            policy=RetryPolicy(max_retry_depth=1, split_factor=2),
            admission_split_policy=AdmissionSplitPolicy(max_split_depth=1),
        ).run_batch(_batch(8))
    )

    assert [batch.batch_size for batch in step.calls] == [8, 4, 2, 2, 4, 2, 2]
    assert [result.batch_size for result in results] == [2, 2, 2, 2]


def test_optimizer_does_not_disable_pre_execution_admission_split() -> None:
    def run(batch: KVBatch) -> StepResult | PrePassAdmissionAssessment:
        if batch.batch_size > 2:
            return _assessment(batch.batch_size, max_admissible_items=2)
        return _result(batch)

    step = SyntheticAdmissionStep(run, optimizer=object())
    results = list(
        RuntimeRetryRunner(
            step,
            admission_split_policy=AdmissionSplitPolicy(),
        ).run_batch(_batch(4))
    )

    assert [batch.batch_size for batch in step.calls] == [4, 2, 2]
    assert [result.batch_size for result in results] == [2, 2]


def test_optimizer_still_disables_post_execution_oom_retry() -> None:
    step = SyntheticAdmissionStep(
        lambda batch: _result(
            batch,
            StepStatus.OOM_FAULT,
            phase=RuntimePhase.FORWARD,
        ),
        optimizer=object(),
    )

    results = list(
        RuntimeRetryRunner(
            step,
            admission_split_policy=AdmissionSplitPolicy(),
        ).run_batch(_batch(4))
    )

    assert [batch.batch_size for batch in step.calls] == [4]
    assert [result.status for result in results] == [StepStatus.OOM_FAULT]


def test_orchestrator_records_parent_reject_before_admitted_children() -> None:
    sample_provider = SequenceSampleProvider(
        (_sample(700), _sample(100), _sample(100))
    )
    step = SyntheticAdmissionStep(lambda batch: _result(batch))
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=4))
    orchestrator = ConservativeRuntimeOrchestrator(
        step,
        governor,
        resource_capacity=ResourceCapacity(cpu_total_bytes=1_000),
        admission_profile=_profile(cpu=100),
        admission_sample_provider=sample_provider,
        admission_split_policy=AdmissionSplitPolicy(),
    )

    result = orchestrator.run_pass([_batch(4)])

    assert [batch.batch_size for batch in step.calls] == [2, 2]
    assert [item.batch_size for item in result.admission_assessments] == [4, 2, 2]
    assert [item.status for item in result.admission_assessments] == [
        PrePassAdmissionStatus.REJECT,
        PrePassAdmissionStatus.ADMIT,
        PrePassAdmissionStatus.ADMIT,
    ]
    assert sample_provider.calls == ["before_admission"] * 3


def test_budget_split_then_admission_split_preserves_global_row_order() -> None:
    samples = (
        _sample(700),
        _sample(100),
        _sample(100),
        _sample(700),
        _sample(100),
        _sample(100),
    )
    sample_provider = SequenceSampleProvider(samples)
    step = SyntheticAdmissionStep(lambda batch: _result(batch))
    capacity_provider = FixedCapacityProvider(ResourceCapacity(cpu_total_bytes=1_000))
    orchestrator = ConservativeRuntimeOrchestrator(
        step,
        ConservativeRuntimeGovernor(BatchBudget(max_items=4)),
        resource_capacity_provider=capacity_provider,
        admission_profile=_profile(cpu=100),
        admission_sample_provider=sample_provider,
        admission_split_policy=AdmissionSplitPolicy(),
    )
    original = _batch(8)

    result = orchestrator.run_pass([original])

    assert capacity_provider.calls == 1
    assert [batch.batch_size for batch in step.calls] == [2, 2, 2, 2]
    assert [item.batch_size for item in result.admission_assessments] == [
        4,
        2,
        2,
        4,
        2,
        2,
    ]
    assert torch.equal(torch.cat([item.row_ids for item in result.results]), original.row_ids)


def test_terminal_child_block_keeps_governor_unchanged() -> None:
    sample_provider = SequenceSampleProvider((_sample(700), _sample(850)))
    step = SyntheticAdmissionStep(lambda batch: _result(batch))
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=4))
    orchestrator = ConservativeRuntimeOrchestrator(
        step,
        governor,
        resource_capacity=ResourceCapacity(cpu_total_bytes=1_000),
        admission_profile=_profile(cpu=100),
        admission_sample_provider=sample_provider,
        admission_split_policy=AdmissionSplitPolicy(max_split_depth=1),
    )

    with pytest.raises(PrePassAdmissionBlocked):
        orchestrator.run_pass([_batch(4)])

    assert step.calls == []
    assert governor.current_budget == BatchBudget(max_items=4)
    assert governor.state.last_decision is None


def test_recovered_admission_split_does_not_create_governor_fault_feedback() -> None:
    sample_provider = SequenceSampleProvider(
        (_sample(700), _sample(100), _sample(100))
    )
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=4),
        policy=GovernorPolicy(grow_after_successes=1, grow_factor=2.0),
    )
    orchestrator = ConservativeRuntimeOrchestrator(
        SyntheticAdmissionStep(lambda batch: _result(batch)),
        governor,
        resource_capacity=ResourceCapacity(cpu_total_bytes=1_000),
        admission_profile=_profile(cpu=100),
        admission_sample_provider=sample_provider,
        admission_split_policy=AdmissionSplitPolicy(),
    )

    result = orchestrator.run_pass([_batch(4)])

    assert [item.status for item in result.results] == [
        StepStatus.SUCCESS,
        StepStatus.SUCCESS,
    ]
    assert result.recovered_oom is False
    assert result.decision.next_budget == BatchBudget(max_items=8)


def test_orchestrator_rejects_split_policy_without_admission_profile() -> None:
    with pytest.raises(ValueError, match="require admission_profile"):
        ConservativeRuntimeOrchestrator(
            SyntheticAdmissionStep(lambda batch: _result(batch)),
            ConservativeRuntimeGovernor(BatchBudget(max_items=1)),
            admission_split_policy=AdmissionSplitPolicy(),
        )


def test_runner_rejects_invalid_admission_split_policy_type() -> None:
    with pytest.raises(TypeError, match="admission_split_policy"):
        RuntimeRetryRunner(
            SyntheticAdmissionStep(lambda batch: _result(batch)),
            admission_split_policy=object(),  # type: ignore[arg-type]
        )


def test_admission_split_policy_is_development_only() -> None:
    assert "AdmissionSplitPolicy" in runtime_api.__all__
    assert runtime_api.AdmissionSplitPolicy is AdmissionSplitPolicy
    assert "AdmissionSplitPolicy" not in set(enn_torch.__all__)
    with pytest.raises(AttributeError):
        getattr(enn_torch, "AdmissionSplitPolicy")


def test_unknown_allow_path_does_not_invoke_split_recovery() -> None:
    sample_provider = SequenceSampleProvider((_sample(100),))
    step = SyntheticAdmissionStep(lambda batch: _result(batch))
    orchestrator = ConservativeRuntimeOrchestrator(
        step,
        ConservativeRuntimeGovernor(BatchBudget(max_items=4)),
        resource_capacity=ResourceCapacity(cpu_total_bytes=1_000),
        admission_profile=_profile(cpu=None),
        admission_sample_provider=sample_provider,
        admission_unknown_action=AdmissionUnknownAction.ALLOW,
        admission_split_policy=AdmissionSplitPolicy(),
    )

    result = orchestrator.run_pass([_batch(4)])

    assert [batch.batch_size for batch in step.calls] == [4]
    assert [item.status for item in result.admission_assessments] == [
        PrePassAdmissionStatus.UNKNOWN
    ]
