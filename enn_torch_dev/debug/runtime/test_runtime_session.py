from __future__ import annotations

import gc
import weakref
from collections.abc import Callable

import pytest
import torch
from tensordict import TensorDict

from enn_torch_dev.data import KVBatch
from enn_torch_dev.executor import KVStore
from enn_torch_dev.runtime import (
    BatchBudget,
    ConservativeRuntimeGovernor,
    ConservativeRuntimeOrchestrator,
    ConservativeRuntimeSession,
    GovernorPolicy,
    RuntimePassHistory,
    RuntimeSessionRecord,
    StepResult,
    StepStatus,
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
        schema_id="runtime.session",
        shard_id=3,
    )


class Payload:
    pass


class FakeRuntimeStep:
    def __init__(
        self,
        fn: Callable[[KVBatch, int], StepResult] | None = None,
    ) -> None:
        self.fn = fn
        self.calls: list[KVBatch] = []

    def run(self, batch: KVBatch) -> StepResult:
        call_index = len(self.calls)
        self.calls.append(batch)
        if self.fn is not None:
            return self.fn(batch, call_index)
        return StepResult(
            status=StepStatus.SUCCESS,
            phase=None,
            batch_size=batch.batch_size,
            row_ids=batch.row_ids.detach().cpu().clone(),
        )


def _build_session(
    *,
    step: FakeRuntimeStep | None = None,
    max_passes: int = 3,
    max_records: int = 3,
    grow_after_successes: int = 3,
) -> tuple[FakeRuntimeStep, ConservativeRuntimeSession]:
    resolved_step = step or FakeRuntimeStep()
    governor = ConservativeRuntimeGovernor(
        BatchBudget(max_items=2),
        policy=GovernorPolicy(
            grow_after_successes=grow_after_successes,
            grow_factor=2.0,
        ),
    )
    orchestrator = ConservativeRuntimeOrchestrator(resolved_step, governor)
    history = RuntimePassHistory(max_records=max_records)
    session = ConservativeRuntimeSession(
        orchestrator,
        history,
        max_passes=max_passes,
    )
    return resolved_step, session


def test_run_passes_is_lazy_and_executes_one_pass_per_next() -> None:
    step, session = _build_session(max_passes=2)
    events: list[str] = []

    def sources():
        events.append("yield-0")
        yield [_batch(1, offset=0)]
        events.append("after-0")
        events.append("yield-1")
        yield [_batch(1, offset=10)]
        events.append("after-1")
        events.append("yield-2")
        yield [_batch(1, offset=20)]

    records = session.run_passes(sources())

    assert events == []
    assert step.calls == []

    first = next(records)
    assert first.pass_index == 0
    assert events == ["yield-0"]
    assert len(step.calls) == 1

    second = next(records)
    assert second.pass_index == 1
    assert events == ["yield-0", "after-0", "yield-1"]
    assert len(step.calls) == 2

    with pytest.raises(StopIteration):
        next(records)

    assert events == ["yield-0", "after-0", "yield-1"]


def test_session_record_connects_pass_summary_and_history() -> None:
    _, session = _build_session(max_passes=1, max_records=2)

    record = next(session.run_passes([[_batch(2)]]))

    assert isinstance(record, RuntimeSessionRecord)
    assert record.pass_index == 0
    assert record.pass_summary.total_results == 1
    assert record.pass_summary.total_rows == 2
    assert record.history_summary.total_passes == 1
    assert record.history_summary.latest_summary == record.pass_summary
    assert session.history.records == (record.pass_summary,)


def test_later_pass_uses_governor_updated_budget() -> None:
    step, session = _build_session(
        max_passes=2,
        max_records=2,
        grow_after_successes=1,
    )

    records = session.run_passes(
        [
            [_batch(4, offset=0)],
            [_batch(3, offset=10)],
        ]
    )
    first = next(records)
    second = next(records)

    assert [call.batch_size for call in step.calls] == [2, 2, 3]
    assert first.pass_result.decision.next_budget == BatchBudget(max_items=4)
    assert second.pass_result.decision.previous_budget == BatchBudget(max_items=4)


def test_session_preserves_bounded_history_retention() -> None:
    _, session = _build_session(max_passes=3, max_records=2)

    records = list(
        session.run_passes(
            [
                [_batch(1, offset=0)],
                [_batch(1, offset=10)],
                [_batch(1, offset=20)],
            ]
        )
    )

    assert len(records) == 3
    assert session.history.records == (
        records[1].pass_summary,
        records[2].pass_summary,
    )
    assert records[-1].history_summary.total_passes == 2


def test_step_status_fault_does_not_stop_later_passes() -> None:
    def run(batch: KVBatch, call_index: int) -> StepResult:
        status = StepStatus.DATA_FAULT if call_index == 0 else StepStatus.SUCCESS
        return StepResult(
            status=status,
            phase=None,
            batch_size=batch.batch_size,
            row_ids=batch.row_ids.detach().cpu().clone(),
        )

    step = FakeRuntimeStep(run)
    _, session = _build_session(step=step, max_passes=2, max_records=2)

    records = list(
        session.run_passes(
            [
                [_batch(1, offset=0)],
                [_batch(1, offset=10)],
            ]
        )
    )

    assert [record.pass_summary.statuses for record in records] == [
        (StepStatus.DATA_FAULT,),
        (StepStatus.SUCCESS,),
    ]
    assert len(step.calls) == 2


def test_execution_exception_propagates_without_history_update() -> None:
    def raise_error(batch: KVBatch, call_index: int) -> StepResult:
        del batch, call_index
        raise RuntimeError("session pass failed")

    step = FakeRuntimeStep(raise_error)
    _, session = _build_session(step=step, max_passes=1, max_records=2)
    records = session.run_passes([[_batch(1)]])

    with pytest.raises(RuntimeError, match="session pass failed"):
        next(records)

    assert session.history.records == ()


def test_empty_outer_source_yields_no_records() -> None:
    _, session = _build_session(max_passes=2)

    assert list(session.run_passes([])) == []
    assert session.history.records == ()


def test_session_does_not_store_yielded_pass_result() -> None:
    _, session = _build_session(max_passes=1)
    record = next(session.run_passes([[_batch(1)]]))

    assert all(value is not record.pass_result for value in vars(session).values())
    assert all(
        summary is not record.pass_result for summary in session.history.records
    )


def test_generator_frame_releases_previous_pass_payload_before_next_source() -> None:
    payload_ref: weakref.ReferenceType[Payload] | None = None
    release_checked = False

    def run(batch: KVBatch, call_index: int) -> StepResult:
        nonlocal payload_ref
        if call_index == 0:
            payload = Payload()
            payload_ref = weakref.ref(payload)
            store = KVStore({"payload": payload})
            del payload
            return StepResult(
                status=StepStatus.SUCCESS,
                phase=None,
                batch_size=batch.batch_size,
                row_ids=batch.row_ids.detach().cpu().clone(),
                store=store,
            )
        return StepResult(
            status=StepStatus.SUCCESS,
            phase=None,
            batch_size=batch.batch_size,
            row_ids=batch.row_ids.detach().cpu().clone(),
        )

    def sources():
        nonlocal release_checked
        yield [_batch(1, offset=0)]
        gc.collect()
        assert payload_ref is not None
        assert payload_ref() is None
        release_checked = True
        yield [_batch(1, offset=10)]

    step = FakeRuntimeStep(run)
    _, session = _build_session(step=step, max_passes=2, max_records=2)
    records = session.run_passes(sources())

    first = next(records)
    assert payload_ref is not None
    assert payload_ref() is not None

    del first
    gc.collect()

    second = next(records)

    assert release_checked
    assert second.pass_index == 1
    assert len(step.calls) == 2
    assert payload_ref() is None
    assert all(not hasattr(summary, "store") for summary in session.history.records)


def test_session_rejects_invalid_constructor_arguments() -> None:
    _, valid_session = _build_session()
    orchestrator = valid_session.orchestrator
    history = valid_session.history

    with pytest.raises(TypeError, match="orchestrator"):
        ConservativeRuntimeSession(object(), history, max_passes=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="history"):
        ConservativeRuntimeSession(orchestrator, object(), max_passes=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="max_passes"):
        ConservativeRuntimeSession(orchestrator, history, max_passes=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="max_passes"):
        ConservativeRuntimeSession(orchestrator, history, max_passes="2")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_passes"):
        ConservativeRuntimeSession(orchestrator, history, max_passes=0)
    with pytest.raises(ValueError, match="max_passes"):
        ConservativeRuntimeSession(orchestrator, history, max_passes=-1)


def test_session_requires_max_passes() -> None:
    _, valid_session = _build_session()
    with pytest.raises(TypeError):
        ConservativeRuntimeSession(  # type: ignore[call-arg]
            valid_session.orchestrator,
            valid_session.history,
        )


def test_run_passes_rejects_invalid_outer_source() -> None:
    _, session = _build_session(max_passes=1)

    with pytest.raises(TypeError, match="run_passes"):
        session.run_passes(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="run_passes"):
        session.run_passes(_batch(1))  # type: ignore[arg-type]
