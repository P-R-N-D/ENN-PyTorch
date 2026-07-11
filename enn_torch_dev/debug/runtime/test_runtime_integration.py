from __future__ import annotations

from dataclasses import dataclass

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
    RetryPolicy,
    RuntimePassHistory,
    RuntimePhase,
    StepResult,
    StepStatus,
)


def _batch(num_rows: int = 4, *, offset: int = 0) -> KVBatch:
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
        schema_id="runtime.integration",
        shard_id=11,
    )


@dataclass(frozen=True, slots=True)
class CallRecord:
    status: StepStatus
    row_ids: torch.Tensor
    source_ids: torch.Tensor
    sample_ids: torch.Tensor


class ThresholdOomRuntimeStep:
    def __init__(self, *, max_items: int = 2) -> None:
        self.max_items = max_items
        self.calls: list[CallRecord] = []

    def run(self, batch: KVBatch) -> StepResult:
        status = (
            StepStatus.OOM_FAULT
            if batch.batch_size > self.max_items
            else StepStatus.SUCCESS
        )
        assert batch.source_ids is not None
        assert batch.sample_ids is not None
        self.calls.append(
            CallRecord(
                status=status,
                row_ids=batch.row_ids.detach().cpu().clone(),
                source_ids=batch.source_ids.detach().cpu().clone(),
                sample_ids=batch.sample_ids.detach().cpu().clone(),
            )
        )
        return StepResult(
            status=status,
            phase=RuntimePhase.FORWARD if status is StepStatus.OOM_FAULT else None,
            batch_size=batch.batch_size,
            row_ids=batch.row_ids.detach().cpu().clone(),
            error_type=None if status is StepStatus.SUCCESS else "SyntheticOOM",
            error_message=None if status is StepStatus.SUCCESS else "batch too large",
        )


class FaultThenRaiseRuntimeStep:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, batch: KVBatch) -> StepResult:
        self.calls += 1
        if self.calls == 1:
            return StepResult(
                status=StepStatus.DATA_FAULT,
                phase=RuntimePhase.TO_STORE,
                batch_size=batch.batch_size,
                row_ids=batch.row_ids.detach().cpu().clone(),
                error_type="SyntheticDataFault",
                error_message="first pass fault",
            )
        raise RuntimeError("second pass execution failed")


def test_runtime_session_end_to_end_oom_recovery_budget_history_and_identity() -> None:
    step = ThresholdOomRuntimeStep(max_items=2)
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=4),
        policy=GovernorPolicy(
            shrink_factor=0.5,
            grow_factor=2.0,
            grow_after_successes=2,
            min_items=1,
            max_items=4,
        ),
    )
    orchestrator = ConservativeRuntimeOrchestrator(
        step,
        governor,
        retry_policy=RetryPolicy(
            max_retry_depth=2,
            min_items=1,
            split_factor=2,
        ),
    )
    history = RuntimePassHistory(max_records=2)
    session = ConservativeRuntimeSession(
        orchestrator,
        history,
        max_passes=3,
    )
    batches = (
        _batch(4, offset=0),
        _batch(4, offset=10),
        _batch(4, offset=20),
    )

    records = list(session.run_passes(([batch] for batch in batches)))

    assert [record.pass_index for record in records] == [0, 1, 2]
    first, second, third = records

    assert [result.status for result in first.pass_result.results] == [
        StepStatus.SUCCESS,
        StepStatus.SUCCESS,
    ]
    assert first.pass_result.recovered_oom is True
    assert first.pass_summary.recovered_oom is True
    assert first.history_summary.recovered_oom_passes == 1
    assert first.pass_result.decision.previous_budget == BatchBudget(max_items=4)
    assert first.pass_result.decision.next_budget == BatchBudget(max_items=2)

    assert second.pass_result.decision.previous_budget == BatchBudget(max_items=2)
    assert second.pass_result.decision.next_budget == BatchBudget(max_items=2)
    assert second.pass_result.decision.consecutive_successes == 1

    assert third.pass_result.decision.previous_budget == BatchBudget(max_items=2)
    assert third.pass_result.decision.next_budget == BatchBudget(max_items=4)
    assert third.pass_result.decision.consecutive_successes == 0
    assert orchestrator.current_budget == BatchBudget(max_items=4)

    assert history.records == (
        second.pass_summary,
        third.pass_summary,
    )
    assert third.history_summary.total_passes == 2
    assert third.history_summary.total_results == 4
    assert third.history_summary.total_rows == 8
    assert third.history_summary.recovered_oom_passes == 0
    assert third.history_summary.budget_changed_passes == 1
    assert dict(third.history_summary.status_counts) == {StepStatus.SUCCESS: 4}

    for record, batch in zip(records, batches, strict=True):
        assert torch.equal(
            torch.cat([result.row_ids for result in record.pass_result.results]),
            batch.row_ids,
        )

    successful_calls = [
        call for call in step.calls if call.status is StepStatus.SUCCESS
    ]
    assert torch.equal(
        torch.cat([call.row_ids for call in successful_calls]),
        torch.cat([batch.row_ids for batch in batches]),
    )
    assert torch.equal(
        torch.cat([call.source_ids for call in successful_calls]),
        torch.cat([batch.source_ids for batch in batches if batch.source_ids is not None]),
    )
    assert torch.equal(
        torch.cat([call.sample_ids for call in successful_calls]),
        torch.cat([batch.sample_ids for batch in batches if batch.sample_ids is not None]),
    )
    assert [call.status for call in step.calls] == [
        StepStatus.OOM_FAULT,
        StepStatus.SUCCESS,
        StepStatus.SUCCESS,
        StepStatus.SUCCESS,
        StepStatus.SUCCESS,
        StepStatus.SUCCESS,
        StepStatus.SUCCESS,
    ]


def test_runtime_session_keeps_completed_history_when_later_pass_raises() -> None:
    step = FaultThenRaiseRuntimeStep()
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=2))
    orchestrator = ConservativeRuntimeOrchestrator(step, governor)
    history = RuntimePassHistory(max_records=3)
    session = ConservativeRuntimeSession(
        orchestrator,
        history,
        max_passes=3,
    )
    records = session.run_passes(
        [
            [_batch(1, offset=0)],
            [_batch(1, offset=10)],
            [_batch(1, offset=20)],
        ]
    )

    first = next(records)

    assert first.pass_summary.statuses == (StepStatus.DATA_FAULT,)
    assert history.records == (first.pass_summary,)

    with pytest.raises(RuntimeError, match="second pass execution failed"):
        next(records)

    assert history.records == (first.pass_summary,)
    assert history.summarize().total_passes == 1
    assert history.summarize().status_counts[StepStatus.DATA_FAULT] == 1
    assert step.calls == 2


def test_runtime_development_api_is_exported_without_stable_namespace_leak() -> None:
    development_names = {
        "ConservativeRuntimeGovernor",
        "ConservativeRuntimeOrchestrator",
        "ConservativeRuntimeSession",
        "RuntimePassHistory",
        "RuntimePassResult",
        "RuntimePassSummary",
        "RuntimeSessionRecord",
        "summarize_runtime_pass",
    }

    assert len(runtime_api.__all__) == len(set(runtime_api.__all__))
    assert development_names <= set(runtime_api.__all__)
    for name in development_names:
        assert getattr(runtime_api, name) is not None

    assert development_names.isdisjoint(set(enn_torch.__all__))
    for name in development_names:
        with pytest.raises(AttributeError):
            getattr(enn_torch, name)
