from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from enn_torch_dev.data import KVBatch

from .batching import slice_kvbatch
from .faults import StepResult, StepStatus


@runtime_checkable
class RuntimeStepProtocol(Protocol):
    """RuntimeStep-compatible object used by RuntimeRetryRunner."""

    def run(self, batch: KVBatch) -> StepResult:
        ...


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Static retry policy for OOM-class RuntimeStep faults."""

    max_retry_depth: int = 3
    min_items: int = 1
    split_factor: int = 2
    retry_oom: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.max_retry_depth, int) or isinstance(self.max_retry_depth, bool):
            raise TypeError("RetryPolicy.max_retry_depth must be an integer.")
        if self.max_retry_depth < 0:
            raise ValueError("RetryPolicy.max_retry_depth must be non-negative.")
        if not isinstance(self.min_items, int) or isinstance(self.min_items, bool):
            raise TypeError("RetryPolicy.min_items must be an integer.")
        if self.min_items <= 0:
            raise ValueError("RetryPolicy.min_items must be positive.")
        if not isinstance(self.split_factor, int) or isinstance(self.split_factor, bool):
            raise TypeError("RetryPolicy.split_factor must be an integer.")
        if self.split_factor < 2:
            raise ValueError("RetryPolicy.split_factor must be at least 2.")
        if not isinstance(self.retry_oom, bool):
            raise TypeError("RetryPolicy.retry_oom must be a bool.")


class RuntimeRetryRunner:
    """Run RuntimeStep over KVBatch streams with minimal OOM-class retry."""

    def __init__(
        self,
        runtime_step: RuntimeStepProtocol,
        *,
        policy: RetryPolicy | None = None,
    ) -> None:
        if not isinstance(runtime_step, RuntimeStepProtocol):
            raise TypeError("RuntimeRetryRunner.runtime_step must provide run(KVBatch).")
        if policy is not None and not isinstance(policy, RetryPolicy):
            raise TypeError("RuntimeRetryRunner.policy must be a RetryPolicy or None.")
        self.runtime_step = runtime_step
        self.policy = policy or RetryPolicy()

    def run_stream(self, source: Iterable[KVBatch]) -> Iterator[StepResult]:
        if isinstance(source, KVBatch):
            raise TypeError("RuntimeRetryRunner.run_stream expects an iterable of KVBatch.")
        if not isinstance(source, Iterable):
            raise TypeError("RuntimeRetryRunner.run_stream expects an iterable of KVBatch.")
        for batch in source:
            yield from self.run_batch(batch)

    def run_batch(self, batch: KVBatch) -> Iterator[StepResult]:
        if not isinstance(batch, KVBatch):
            raise TypeError("RuntimeRetryRunner.run_batch expects a KVBatch.")
        yield from self._run_with_retry(batch, retry_count=0)

    def _run_with_retry(self, batch: KVBatch, *, retry_count: int) -> Iterator[StepResult]:
        result = self.runtime_step.run(batch)
        if not isinstance(result, StepResult):
            raise TypeError("RuntimeStep.run must return a StepResult.")

        if not self._should_retry(result):
            yield result
            return
        if retry_count >= self.policy.max_retry_depth:
            yield result
            return
        if batch.batch_size <= self.policy.min_items:
            yield result
            return

        for subbatch in self._split_for_retry(batch):
            yield from self._run_with_retry(subbatch, retry_count=retry_count + 1)

    def _should_retry(self, result: StepResult) -> bool:
        return self.policy.retry_oom and result.status is StepStatus.OOM_FAULT

    def _split_for_retry(self, batch: KVBatch) -> Iterator[KVBatch]:
        target_size = (batch.batch_size + self.policy.split_factor - 1) // self.policy.split_factor
        target_size = max(self.policy.min_items, target_size)
        if target_size >= batch.batch_size:
            target_size = batch.batch_size - 1
        target_size = max(1, target_size)

        for start in range(0, batch.batch_size, target_size):
            end = min(start + target_size, batch.batch_size)
            yield slice_kvbatch(batch, start, end, cost_hint=None)
