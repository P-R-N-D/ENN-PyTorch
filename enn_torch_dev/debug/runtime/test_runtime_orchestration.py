from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from tensordict import TensorDict

from enn_torch_dev.data import KVBatch
from enn_torch_dev.runtime import (
    BatchBudget,
    ConservativeRuntimeGovernor,
    ConservativeRuntimeOrchestrator,
    GovernorPolicy,
    RetryPolicy,
    RuntimePassResult,
    RuntimePhase,
    StepResult,
    StepStatus,
)


def _batch(num_rows: int = 4) -> KVBatch:
    td = TensorDict(
        {
            "features": torch.arange(num_rows * 2, dtype=torch.float32).reshape(
                num_rows,
                2,
            ),
        },
        batch_size=(num_rows,),
    )
    return KVBatch(
        td=td,
        row_ids=torch.arange(100, 100 + num_rows),
        source_ids=torch.arange(200, 200 + num_rows),
        sample_ids=torch.arange(300, 300 + num_rows),
        schema_id="runtime.orchestration",
        shard_id=7,
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


class FakeRuntimeStep:
    def __init__(self, fn: Callable[[KVBatch], StepResult]) -> None:
        self.fn = fn
        self.calls: list[KVBatch] = []

    def run(self, batch: KVBatch) -> StepResult:
        self.calls.append(batch)
        return self.fn(batch)


def _success(batch: KVBatch) -> StepResult:
    return _result(batch, StepStatus.SUCCESS)


def _oom(batch: KVBatch) -> StepResult:
    return _result(batch, StepStatus.OOM_FAULT, phase=RuntimePhase.FORWARD)


def test_orchestrator_runs_budget_retry_and_governor_for_one_pass() -> None:
    step = FakeRuntimeStep(_success)
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=2),
        policy=GovernorPolicy(grow_after_successes=1, grow_factor=2.0),
    )
    orchestrator = ConservativeRuntimeOrchestrator(step, governor)

    pass_result = orchestrator.run_pass([_batch(4)])

    assert isinstance(pass_result, RuntimePassResult)
    assert [call.batch_size for call in step.calls] == [2, 2]
    assert [result.batch_size for result in pass_result.results] == [2, 2]
    assert pass_result.recovered_oom is False
    assert pass_result.decision.previous_budget == BatchBudget(max_items=2)
    assert pass_result.decision.next_budget == BatchBudget(max_items=4)
    assert orchestrator.current_budget == BatchBudget(max_items=4)
    assert orchestrator.last_decision == pass_result.decision


def test_orchestrator_uses_governor_next_budget_on_later_pass() -> None:
    step = FakeRuntimeStep(_success)
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=2),
        policy=GovernorPolicy(grow_after_successes=1, grow_factor=2.0),
    )
    orchestrator = ConservativeRuntimeOrchestrator(step, governor)

    first = orchestrator.run_pass([_batch(4)])
    second = orchestrator.run_pass([_batch(3)])

    assert first.decision.next_budget == BatchBudget(max_items=4)
    assert second.decision.previous_budget == BatchBudget(max_items=4)
    assert [call.batch_size for call in step.calls] == [2, 2, 3]


def test_orchestrator_reports_retry_recovered_oom_to_governor() -> None:
    def run(batch: KVBatch) -> StepResult:
        if batch.batch_size > 2:
            return _oom(batch)
        return _success(batch)

    step = FakeRuntimeStep(run)
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=4),
        policy=GovernorPolicy(shrink_factor=0.5, grow_after_successes=1),
    )
    orchestrator = ConservativeRuntimeOrchestrator(
        step,
        governor,
        retry_policy=RetryPolicy(max_retry_depth=2, split_factor=2),
    )

    pass_result = orchestrator.run_pass([_batch(4)])

    assert [call.batch_size for call in step.calls] == [4, 2, 2]
    assert [result.status for result in pass_result.results] == [
        StepStatus.SUCCESS,
        StepStatus.SUCCESS,
    ]
    assert pass_result.recovered_oom is True
    assert pass_result.decision.next_budget == BatchBudget(max_items=2)
    assert pass_result.decision.consecutive_successes == 0
    assert pass_result.decision.consecutive_ooms == 1
    assert "retry-recovered OOM observed" in pass_result.decision.reason


def test_orchestrator_does_not_mark_unrecovered_oom_as_recovered() -> None:
    step = FakeRuntimeStep(_oom)
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=4),
        policy=GovernorPolicy(shrink_factor=0.5),
    )
    orchestrator = ConservativeRuntimeOrchestrator(
        step,
        governor,
        retry_policy=RetryPolicy(max_retry_depth=0),
    )

    pass_result = orchestrator.run_pass([_batch(4)])

    assert [result.status for result in pass_result.results] == [StepStatus.OOM_FAULT]
    assert pass_result.recovered_oom is False
    assert pass_result.decision.next_budget == BatchBudget(max_items=2)
    assert "OOM fault observed" in pass_result.decision.reason


def test_orchestrator_preserves_row_order_across_budget_splits() -> None:
    step = FakeRuntimeStep(_success)
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=2))
    orchestrator = ConservativeRuntimeOrchestrator(step, governor)
    batch = _batch(5)

    pass_result = orchestrator.run_pass([batch])

    assert [result.batch_size for result in pass_result.results] == [2, 2, 1]
    assert torch.equal(
        torch.cat([result.row_ids for result in pass_result.results]),
        batch.row_ids,
    )


def test_orchestrator_accepts_empty_source() -> None:
    step = FakeRuntimeStep(_success)
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=2))
    orchestrator = ConservativeRuntimeOrchestrator(step, governor)

    pass_result = orchestrator.run_pass([])

    assert pass_result.results == ()
    assert pass_result.recovered_oom is False
    assert pass_result.decision.statuses == ()
    assert pass_result.decision.next_budget == BatchBudget(max_items=2)
    assert step.calls == []


def test_orchestrator_rejects_invalid_constructor_arguments() -> None:
    step = FakeRuntimeStep(_success)
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=2))

    with pytest.raises(TypeError, match="runtime_step"):
        ConservativeRuntimeOrchestrator(object(), governor)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="governor"):
        ConservativeRuntimeOrchestrator(step, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="retry_policy"):
        ConservativeRuntimeOrchestrator(step, governor, retry_policy=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="cost_probe"):
        ConservativeRuntimeOrchestrator(step, governor, cost_probe=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="split_oversized"):
        ConservativeRuntimeOrchestrator(step, governor, split_oversized="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_items"):
        ConservativeRuntimeOrchestrator(step, governor, min_items=0)


def test_orchestrator_rejects_invalid_source() -> None:
    step = FakeRuntimeStep(_success)
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=2))
    orchestrator = ConservativeRuntimeOrchestrator(step, governor)

    with pytest.raises(TypeError, match="run_pass"):
        orchestrator.run_pass(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="run_pass"):
        orchestrator.run_pass(_batch(1))  # type: ignore[arg-type]
