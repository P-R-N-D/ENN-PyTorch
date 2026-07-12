from __future__ import annotations

import gc
import weakref
from collections.abc import Iterable, Iterator

import pytest
import torch
from tensordict import TensorDict

from enn_torch_dev.data import KVBatch
from enn_torch_dev.runtime import (
    BatchBudget,
    ConservativeRuntimeGovernor,
    ConservativeRuntimeOrchestrator,
    ConservativeRuntimeSession,
    RuntimePassHistory,
    RuntimePassSourceFactory,
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
        schema_id="runtime.source_factory",
        shard_id=5,
    )


class FakeRuntimeStep:
    def __init__(self) -> None:
        self.calls: list[KVBatch] = []

    def run(self, batch: KVBatch) -> StepResult:
        self.calls.append(batch)
        return StepResult(
            status=StepStatus.SUCCESS,
            phase=None,
            batch_size=batch.batch_size,
            row_ids=batch.row_ids.detach().cpu().clone(),
        )


class RaisingRuntimeStep:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, batch: KVBatch) -> StepResult:
        self.calls += 1
        del batch
        raise RuntimeError("pass execution failed")


def _build_session(
    *,
    max_passes: int = 3,
    max_records: int = 3,
) -> tuple[FakeRuntimeStep, ConservativeRuntimeSession]:
    step = FakeRuntimeStep()
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=2))
    orchestrator = ConservativeRuntimeOrchestrator(step, governor)
    history = RuntimePassHistory(max_records=max_records)
    session = ConservativeRuntimeSession(
        orchestrator,
        history,
        max_passes=max_passes,
    )
    return step, session


def _record_signature(record: RuntimeSessionRecord) -> tuple[object, ...]:
    return (
        record.pass_index,
        record.pass_summary.total_results,
        record.pass_summary.statuses,
        dict(record.pass_summary.status_counts),
        record.pass_summary.total_batch_size,
        record.pass_summary.total_rows,
        record.pass_summary.recovered_oom,
        record.pass_summary.previous_budget,
        record.pass_summary.next_budget,
        record.pass_summary.budget_changed,
        record.pass_summary.decision_reason,
        record.pass_summary.consecutive_successes,
        record.pass_summary.consecutive_ooms,
        record.history_summary.total_passes,
        record.history_summary.total_results,
        record.history_summary.total_batch_size,
        record.history_summary.total_rows,
        dict(record.history_summary.status_counts),
        record.history_summary.recovered_oom_passes,
        record.history_summary.oom_passes,
        record.history_summary.budget_changed_passes,
        record.history_summary.latest_summary == record.pass_summary,
    )


class RecordingFactory:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.sources: list[Iterator[KVBatch]] = []

    def create_pass_source(self, pass_index: int) -> Iterable[KVBatch]:
        self.calls.append(pass_index)
        source = iter((_batch(1, offset=pass_index * 10),))
        self.sources.append(source)
        return source


class RaisingFactory:
    def create_pass_source(self, pass_index: int) -> Iterable[KVBatch]:
        if pass_index == 1:
            raise RuntimeError("factory failed")
        return [_batch(1, offset=pass_index * 10)]


class EmptyFactory:
    def create_pass_source(self, pass_index: int) -> Iterable[KVBatch]:
        del pass_index
        return []


class FreshSource:
    def __init__(self, batch: KVBatch) -> None:
        self.batch = batch

    def __iter__(self) -> Iterator[KVBatch]:
        yield self.batch


class ReleasingFactory:
    def __init__(self) -> None:
        self.previous_source_ref: weakref.ReferenceType[FreshSource] | None = None
        self.release_checked = False

    def create_pass_source(self, pass_index: int) -> Iterable[KVBatch]:
        if pass_index:
            gc.collect()
            assert self.previous_source_ref is not None
            assert self.previous_source_ref() is None
            self.release_checked = True
        source = FreshSource(_batch(1, offset=pass_index * 10))
        self.previous_source_ref = weakref.ref(source)
        return source


def test_run_factory_is_lazy_and_calls_once_per_next() -> None:
    step, session = _build_session(max_passes=2)
    factory = RecordingFactory()

    records = session.run_factory(factory)

    assert isinstance(factory, RuntimePassSourceFactory)
    assert factory.calls == []
    assert step.calls == []

    first = next(records)
    assert first.pass_index == 0
    assert factory.calls == [0]
    assert len(step.calls) == 1

    second = next(records)
    assert second.pass_index == 1
    assert factory.calls == [0, 1]
    assert len(step.calls) == 2

    with pytest.raises(StopIteration):
        next(records)

    assert factory.calls == [0, 1]


def test_run_factory_creates_fresh_one_shot_sources_and_updates_history() -> None:
    _, session = _build_session(max_passes=3, max_records=2)
    factory = RecordingFactory()

    records = list(session.run_factory(factory))

    assert factory.calls == [0, 1, 2]
    assert len(factory.sources) == 3
    assert len({id(source) for source in factory.sources}) == 3
    assert [record.pass_index for record in records] == [0, 1, 2]
    assert [record.pass_summary.total_rows for record in records] == [1, 1, 1]
    assert session.history.records == (
        records[1].pass_summary,
        records[2].pass_summary,
    )
    assert records[-1].history_summary.total_passes == 2


def test_factory_exception_propagates_without_failed_history_record() -> None:
    _, session = _build_session(max_passes=3, max_records=3)
    records = session.run_factory(RaisingFactory())

    first = next(records)
    assert session.history.records == (first.pass_summary,)

    with pytest.raises(RuntimeError, match="factory failed"):
        next(records)

    assert session.history.records == (first.pass_summary,)


def test_empty_factory_source_is_a_completed_finite_pass() -> None:
    step, session = _build_session(max_passes=1)

    record = next(session.run_factory(EmptyFactory()))

    assert record.pass_index == 0
    assert record.pass_result.results == ()
    assert record.pass_summary.total_results == 0
    assert record.history_summary.total_passes == 1
    assert step.calls == []


@pytest.mark.parametrize("source", [_batch(1), object()])
def test_run_factory_rejects_invalid_created_source(source: object) -> None:
    class InvalidSourceFactory:
        def create_pass_source(self, pass_index: int) -> object:
            del pass_index
            return source

    _, session = _build_session(max_passes=1)
    records = session.run_factory(InvalidSourceFactory())

    with pytest.raises(TypeError, match="create_pass_source"):
        next(records)

    assert session.history.records == ()


def test_run_passes_and_run_factory_share_pass_execution_contract() -> None:
    _, passes_session = _build_session(max_passes=2, max_records=2)
    _, factory_session = _build_session(max_passes=2, max_records=2)
    factory = RecordingFactory()

    pass_records = list(
        passes_session.run_passes(
            [
                [_batch(1, offset=0)],
                [_batch(1, offset=10)],
            ]
        )
    )
    factory_records = list(factory_session.run_factory(factory))

    assert [_record_signature(record) for record in pass_records] == [
        _record_signature(record) for record in factory_records
    ]
    assert passes_session.history.records == tuple(
        record.pass_summary for record in pass_records
    )
    assert factory_session.history.records == tuple(
        record.pass_summary for record in factory_records
    )


def test_run_factory_propagates_pass_execution_exception_without_history_update() -> None:
    step = RaisingRuntimeStep()
    governor = ConservativeRuntimeGovernor(BatchBudget(max_items=2))
    orchestrator = ConservativeRuntimeOrchestrator(step, governor)
    history = RuntimePassHistory(max_records=2)
    session = ConservativeRuntimeSession(
        orchestrator,
        history,
        max_passes=1,
    )
    factory = RecordingFactory()
    records = session.run_factory(factory)

    with pytest.raises(RuntimeError, match="pass execution failed"):
        next(records)

    assert factory.calls == [0]
    assert step.calls == 1
    assert history.records == ()


def test_run_factory_rejects_invalid_factory() -> None:
    _, session = _build_session(max_passes=1)

    with pytest.raises(TypeError, match="source_factory"):
        session.run_factory(object())  # type: ignore[arg-type]


def test_factory_source_is_released_before_next_factory_call() -> None:
    _, session = _build_session(max_passes=2)
    factory = ReleasingFactory()
    records = session.run_factory(factory)

    first = next(records)
    assert factory.previous_source_ref is not None
    assert factory.previous_source_ref() is not None

    second = next(records)

    assert first.pass_index == 0
    assert second.pass_index == 1
    assert factory.release_checked
