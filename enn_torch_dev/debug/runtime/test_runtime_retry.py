from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from tensordict import TensorDict

import enn_torch_dev.runtime.retry as retry_module
from enn_torch_dev.data import KVBatch
from enn_torch_dev.executor import KVStore
from enn_torch_dev.runtime import (
    RetryPolicy,
    RuntimePhase,
    RuntimeRetryRunner,
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
        schema_id="runtime.retry",
        shard_id=11,
    )


def _storage_ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _result(
    batch: KVBatch,
    status: StepStatus,
    *,
    phase: RuntimePhase | None = None,
    message: str | None = None,
) -> StepResult:
    return StepResult(
        status=status,
        phase=phase,
        batch_size=batch.batch_size,
        row_ids=batch.row_ids.detach().cpu().clone(),
        error_type=None if status is StepStatus.SUCCESS else status.name,
        error_message=message,
    )


class FakeRuntimeStep:
    def __init__(self, fn: Callable[[KVBatch], StepResult]) -> None:
        self.fn = fn
        self.calls: list[KVBatch] = []

    def run(self, batch: KVBatch) -> StepResult:
        self.calls.append(batch)
        return self.fn(batch)


def _oom(batch: KVBatch) -> StepResult:
    return _result(
        batch,
        StepStatus.OOM_FAULT,
        phase=RuntimePhase.FORWARD,
        message="CUDA out of memory",
    )


def _success(batch: KVBatch) -> StepResult:
    return _result(batch, StepStatus.SUCCESS)


def test_runtime_retry_runner_returns_success_without_retry() -> None:
    step = FakeRuntimeStep(_success)
    batch = _batch(3)

    results = list(RuntimeRetryRunner(step).run_batch(batch))

    assert len(results) == 1
    assert results[0].ok
    assert [call.batch_size for call in step.calls] == [3]
    assert torch.equal(results[0].row_ids, batch.row_ids)


@pytest.mark.parametrize(
    "status",
    [
        StepStatus.DATA_FAULT,
        StepStatus.RUNTIME_FAULT,
        StepStatus.NONFINITE_FAULT,
    ],
)
def test_runtime_retry_runner_does_not_retry_non_oom_faults(status: StepStatus) -> None:
    step = FakeRuntimeStep(lambda batch: _result(batch, status, phase=RuntimePhase.FORWARD))
    batch = _batch(4)

    results = list(RuntimeRetryRunner(step).run_batch(batch))

    assert [result.status for result in results] == [status]
    assert [call.batch_size for call in step.calls] == [4]


def test_runtime_retry_runner_splits_oom_batches_and_preserves_order() -> None:
    def run(batch: KVBatch) -> StepResult:
        if batch.batch_size > 2:
            return _oom(batch)
        return _success(batch)

    step = FakeRuntimeStep(run)
    batch = _batch(4)

    results = list(
        RuntimeRetryRunner(
            step,
            policy=RetryPolicy(max_retry_depth=2, split_factor=2),
        ).run_batch(batch)
    )

    assert [result.status for result in results] == [
        StepStatus.SUCCESS,
        StepStatus.SUCCESS,
    ]
    assert [result.batch_size for result in results] == [2, 2]
    assert torch.equal(torch.cat([result.row_ids for result in results]), batch.row_ids)
    assert [call.batch_size for call in step.calls] == [4, 2, 2]


def test_runtime_retry_runner_preserves_split_batch_identity_and_metadata() -> None:
    def run(batch: KVBatch) -> StepResult:
        if batch.batch_size > 3:
            return _oom(batch)
        return _success(batch)

    step = FakeRuntimeStep(run)
    batch = _batch(5)

    results = list(
        RuntimeRetryRunner(
            step,
            policy=RetryPolicy(max_retry_depth=2, split_factor=2),
        ).run_batch(batch)
    )

    retried_batches = step.calls[1:]
    assert [subbatch.batch_size for subbatch in retried_batches] == [3, 2]
    assert [subbatch.schema_id for subbatch in retried_batches] == [
        "runtime.retry",
        "runtime.retry",
    ]
    assert [subbatch.shard_id for subbatch in retried_batches] == [11, 11]
    assert torch.equal(torch.cat([subbatch.row_ids for subbatch in retried_batches]), batch.row_ids)
    assert torch.equal(
        torch.cat(
            [
                subbatch.source_ids
                for subbatch in retried_batches
                if subbatch.source_ids is not None
            ]
        ),
        batch.source_ids,
    )
    assert torch.equal(
        torch.cat(
            [
                subbatch.sample_ids
                for subbatch in retried_batches
                if subbatch.sample_ids is not None
            ]
        ),
        batch.sample_ids,
    )
    assert torch.equal(torch.cat([result.row_ids for result in results]), batch.row_ids)


def test_runtime_retry_runner_materializes_split_identity_tensors() -> None:
    def run(batch: KVBatch) -> StepResult:
        if batch.batch_size > 3:
            return _oom(batch)
        return _success(batch)

    step = FakeRuntimeStep(run)
    batch = _batch(5)

    list(
        RuntimeRetryRunner(
            step,
            policy=RetryPolicy(max_retry_depth=2, split_factor=2),
        ).run_batch(batch)
    )

    assert [subbatch.batch_size for subbatch in step.calls[1:]] == [3, 2]
    for subbatch in step.calls[1:]:
        assert _storage_ptr(subbatch.row_ids) != _storage_ptr(batch.row_ids)
        assert subbatch.source_ids is not None
        assert batch.source_ids is not None
        assert _storage_ptr(subbatch.source_ids) != _storage_ptr(batch.source_ids)
        assert subbatch.sample_ids is not None
        assert batch.sample_ids is not None
        assert _storage_ptr(subbatch.sample_ids) != _storage_ptr(batch.sample_ids)


def test_runtime_retry_runner_does_not_run_remainder_below_min_items() -> None:
    step = FakeRuntimeStep(_oom)
    batch = _batch(5)

    results = list(
        RuntimeRetryRunner(
            step,
            policy=RetryPolicy(min_items=4, split_factor=2),
        ).run_batch(batch)
    )

    assert [result.status for result in results] == [StepStatus.OOM_FAULT]
    assert [result.batch_size for result in results] == [5]
    assert [call.batch_size for call in step.calls] == [5]


def test_runtime_retry_runner_keeps_valid_split_at_or_above_min_items() -> None:
    def run(batch: KVBatch) -> StepResult:
        if batch.batch_size > 6:
            return _oom(batch)
        return _success(batch)

    step = FakeRuntimeStep(run)
    batch = _batch(10)

    results = list(
        RuntimeRetryRunner(
            step,
            policy=RetryPolicy(min_items=4, split_factor=3),
        ).run_batch(batch)
    )

    retried_sizes = [call.batch_size for call in step.calls[1:]]
    assert retried_sizes == [4, 6]
    assert all(size >= 4 for size in retried_sizes)
    assert [result.status for result in results] == [StepStatus.SUCCESS, StepStatus.SUCCESS]
    assert [result.batch_size for result in results] == [4, 6]
    assert torch.equal(torch.cat([result.row_ids for result in results]), batch.row_ids)


def test_runtime_retry_runner_drops_failed_result_references_before_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_slice_kvbatch = retry_module.slice_kvbatch

    def spy_slice_kvbatch(
        batch: KVBatch,
        start: int,
        end: int,
        *,
        cost_hint: object = None,
    ) -> KVBatch:
        events.append(f"slice:{start}:{end}")
        return original_slice_kvbatch(batch, start, end, cost_hint=cost_hint)

    monkeypatch.setattr(retry_module, "slice_kvbatch", spy_slice_kvbatch)

    class SpyRetryRunner(RuntimeRetryRunner):
        def __init__(self, step: FakeRuntimeStep) -> None:
            super().__init__(step)
            self.cleanup_call_counts: list[int] = []
            self.cleanup_saw_heavy_refs: list[tuple[bool, bool]] = []

        def _drop_retry_result_references(self, result: StepResult) -> None:
            events.append("cleanup")
            self.cleanup_call_counts.append(len(self.runtime_step.calls))
            self.cleanup_saw_heavy_refs.append((result.store is not None, result.loss is not None))

    def run(batch: KVBatch) -> StepResult:
        events.append(f"run:{batch.batch_size}")
        if batch.batch_size > 2:
            store = KVStore()
            store.set("heavy", torch.ones(batch.batch_size), origin="test")
            return StepResult(
                status=StepStatus.OOM_FAULT,
                phase=RuntimePhase.FORWARD,
                batch_size=batch.batch_size,
                row_ids=batch.row_ids.detach().cpu().clone(),
                loss=torch.ones((), requires_grad=True),
                store=store,
                error_type="OOM_FAULT",
                error_message="CUDA out of memory",
            )
        return _success(batch)

    step = FakeRuntimeStep(run)
    runner = SpyRetryRunner(step)

    results = list(runner.run_batch(_batch(4)))

    assert [result.status for result in results] == [StepStatus.SUCCESS, StepStatus.SUCCESS]
    assert runner.cleanup_call_counts == [1]
    assert runner.cleanup_saw_heavy_refs == [(True, True)]
    assert [call.batch_size for call in step.calls] == [4, 2, 2]
    assert events == ["run:4", "cleanup", "slice:0:2", "run:2", "slice:2:4", "run:2"]


@pytest.mark.parametrize(
    "phase",
    [RuntimePhase.TO_STORE, RuntimePhase.FORWARD, RuntimePhase.LOSS],
)
def test_runtime_retry_runner_retries_side_effect_safe_oom_phases(
    phase: RuntimePhase,
) -> None:
    def run(batch: KVBatch) -> StepResult:
        if batch.batch_size > 2:
            return _result(batch, StepStatus.OOM_FAULT, phase=phase)
        return _success(batch)

    step = FakeRuntimeStep(run)
    batch = _batch(4)

    results = list(RuntimeRetryRunner(step).run_batch(batch))

    assert [result.status for result in results] == [StepStatus.SUCCESS, StepStatus.SUCCESS]
    assert [call.batch_size for call in step.calls] == [4, 2, 2]


@pytest.mark.parametrize(
    "phase",
    [RuntimePhase.BACKWARD, RuntimePhase.OPTIMIZER, None],
)
def test_runtime_retry_runner_does_not_retry_side_effect_or_unknown_oom_phases(
    phase: RuntimePhase | None,
) -> None:
    step = FakeRuntimeStep(lambda batch: _result(batch, StepStatus.OOM_FAULT, phase=phase))
    batch = _batch(4)

    results = list(RuntimeRetryRunner(step).run_batch(batch))

    assert [result.status for result in results] == [StepStatus.OOM_FAULT]
    assert [result.phase for result in results] == [phase]
    assert [call.batch_size for call in step.calls] == [4]


def test_runtime_retry_runner_does_not_retry_runtime_step_with_optimizer() -> None:
    step = FakeRuntimeStep(_oom)
    step.optimizer = object()  # type: ignore[attr-defined]
    batch = _batch(4)

    results = list(RuntimeRetryRunner(step).run_batch(batch))

    assert [result.status for result in results] == [StepStatus.OOM_FAULT]
    assert [call.batch_size for call in step.calls] == [4]


def test_runtime_retry_runner_returns_oom_when_min_items_still_fails() -> None:
    step = FakeRuntimeStep(_oom)
    batch = _batch(1)

    results = list(RuntimeRetryRunner(step).run_batch(batch))

    assert [result.status for result in results] == [StepStatus.OOM_FAULT]
    assert [result.batch_size for result in results] == [1]
    assert [call.batch_size for call in step.calls] == [1]


def test_runtime_retry_runner_stops_at_max_retry_depth() -> None:
    step = FakeRuntimeStep(_oom)
    batch = _batch(4)

    results = list(
        RuntimeRetryRunner(
            step,
            policy=RetryPolicy(max_retry_depth=1, split_factor=2),
        ).run_batch(batch)
    )

    assert [result.status for result in results] == [
        StepStatus.OOM_FAULT,
        StepStatus.OOM_FAULT,
    ]
    assert [result.batch_size for result in results] == [2, 2]
    assert [call.batch_size for call in step.calls] == [4, 2, 2]


def test_runtime_retry_runner_can_disable_oom_retry() -> None:
    step = FakeRuntimeStep(_oom)
    batch = _batch(4)

    results = list(
        RuntimeRetryRunner(
            step,
            policy=RetryPolicy(retry_oom=False),
        ).run_batch(batch)
    )

    assert [result.status for result in results] == [StepStatus.OOM_FAULT]
    assert [call.batch_size for call in step.calls] == [4]


def test_runtime_retry_runner_accepts_empty_stream() -> None:
    step = FakeRuntimeStep(_success)

    assert list(RuntimeRetryRunner(step).run_stream([])) == []
    assert step.calls == []


def test_runtime_retry_runner_runs_stream_in_source_order() -> None:
    step = FakeRuntimeStep(_success)
    batches = [_batch(1), _batch(2)]

    results = list(RuntimeRetryRunner(step).run_stream(batches))

    assert [result.batch_size for result in results] == [1, 2]
    assert torch.equal(results[0].row_ids, batches[0].row_ids)
    assert torch.equal(results[1].row_ids, batches[1].row_ids)


def test_runtime_retry_runner_rejects_invalid_policy_values() -> None:
    with pytest.raises(TypeError, match="max_retry_depth"):
        RetryPolicy(max_retry_depth=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_retry_depth"):
        RetryPolicy(max_retry_depth=-1)
    with pytest.raises(ValueError, match="min_items"):
        RetryPolicy(min_items=0)
    with pytest.raises(ValueError, match="split_factor"):
        RetryPolicy(split_factor=1)
    with pytest.raises(TypeError, match="retry_oom"):
        RetryPolicy(retry_oom="yes")  # type: ignore[arg-type]


def test_runtime_retry_runner_rejects_invalid_arguments() -> None:
    with pytest.raises(TypeError, match="runtime_step"):
        RuntimeRetryRunner(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="policy"):
        RuntimeRetryRunner(FakeRuntimeStep(_success), policy=object())  # type: ignore[arg-type]


def test_runtime_retry_runner_rejects_invalid_batches() -> None:
    runner = RuntimeRetryRunner(FakeRuntimeStep(_success))

    with pytest.raises(TypeError, match="run_batch"):
        list(runner.run_batch(object()))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="run_stream"):
        list(runner.run_stream(object()))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="KVBatch"):
        list(runner.run_stream([object()]))  # type: ignore[list-item]


def test_runtime_retry_runner_rejects_non_stepresult_returns() -> None:
    class BadRuntimeStep:
        def run(self, batch: KVBatch) -> object:
            return object()

    with pytest.raises(TypeError, match="StepResult"):
        list(RuntimeRetryRunner(BadRuntimeStep()).run_batch(_batch(1)))
