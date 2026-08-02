from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from enn_torch_dev.data import KVBatch

from .admission import PrePassAdmissionAssessment, PrePassAdmissionStatus
from .admission_gate import AdmissionSplitPolicy, PrePassAdmissionBlocked
from .batching import slice_kvbatch
from .faults import RuntimePhase, StepResult, StepStatus

_RETRYABLE_OOM_PHASES = frozenset(
    (
        RuntimePhase.TO_STORE,
        RuntimePhase.FORWARD,
        RuntimePhase.LOSS,
    )
)


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
    """Run batches with bounded admission recovery and side-effect-safe OOM retry."""

    def __init__(
        self,
        runtime_step: RuntimeStepProtocol,
        *,
        policy: RetryPolicy | None = None,
        admission_split_policy: AdmissionSplitPolicy | None = None,
    ) -> None:
        if not isinstance(runtime_step, RuntimeStepProtocol):
            raise TypeError("RuntimeRetryRunner.runtime_step must provide run(KVBatch).")
        if policy is not None and not isinstance(policy, RetryPolicy):
            raise TypeError("RuntimeRetryRunner.policy must be a RetryPolicy or None.")
        if admission_split_policy is not None and not isinstance(
            admission_split_policy, AdmissionSplitPolicy
        ):
            raise TypeError(
                "RuntimeRetryRunner.admission_split_policy must be an "
                "AdmissionSplitPolicy or None."
            )
        self.runtime_step = runtime_step
        self.policy = policy or RetryPolicy()
        self.admission_split_policy = admission_split_policy

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
        yield from self._run_with_retry(
            batch,
            retry_count=0,
            admission_split_depth=0,
        )

    def _run_with_retry(
        self,
        batch: KVBatch,
        *,
        retry_count: int,
        admission_split_depth: int,
    ) -> Iterator[StepResult]:
        admission_ranges: tuple[tuple[int, int], ...] = ()
        result: object | None = None
        try:
            result = self.runtime_step.run(batch)
        except PrePassAdmissionBlocked as blocked:
            admission_ranges = self._admission_split_ranges(
                batch.batch_size,
                blocked.assessment,
                admission_split_depth=admission_split_depth,
            )
            if len(admission_ranges) < 2:
                raise
            blocked.__traceback__ = None

        if admission_ranges:
            for start, end in admission_ranges:
                subbatch = slice_kvbatch(batch, start, end, cost_hint=None)
                yield from self._run_with_retry(
                    subbatch,
                    retry_count=retry_count,
                    admission_split_depth=admission_split_depth + 1,
                )
            return

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

        ranges = self._split_ranges(batch.batch_size)
        if len(ranges) < 2:
            yield result
            return

        self._drop_retry_result_references(result)
        del result

        for start, end in ranges:
            subbatch = slice_kvbatch(batch, start, end, cost_hint=None)
            yield from self._run_with_retry(
                subbatch,
                retry_count=retry_count + 1,
                admission_split_depth=admission_split_depth,
            )

    def _should_retry(self, result: StepResult) -> bool:
        return (
            self.policy.retry_oom
            and result.status is StepStatus.OOM_FAULT
            and result.phase in _RETRYABLE_OOM_PHASES
            and getattr(self.runtime_step, "optimizer", None) is None
        )

    def _drop_retry_result_references(self, result: StepResult) -> None:
        """Hook called before retrying so full-batch OOM results can be released."""

    def _admission_split_ranges(
        self,
        batch_size: int,
        assessment: PrePassAdmissionAssessment,
        *,
        admission_split_depth: int,
    ) -> tuple[tuple[int, int], ...]:
        policy = self.admission_split_policy
        if policy is None:
            return ()
        if assessment.status is not PrePassAdmissionStatus.REJECT:
            return ()
        if assessment.batch_size != batch_size:
            return ()
        if admission_split_depth >= policy.max_split_depth:
            return ()

        target_size = assessment.max_admissible_items
        if (
            not isinstance(target_size, int)
            or isinstance(target_size, bool)
            or target_size <= 0
            or target_size >= batch_size
            or target_size < policy.min_items
        ):
            return ()

        part_count = (batch_size + target_size - 1) // target_size
        if part_count < 2 or part_count > policy.max_split_parts:
            return ()
        if batch_size < part_count * policy.min_items:
            return ()

        base_size, larger_parts = divmod(batch_size, part_count)
        sizes = tuple(
            base_size + (1 if index < larger_parts else 0)
            for index in range(part_count)
        )
        if any(
            size < policy.min_items or size > target_size
            for size in sizes
        ):
            return ()

        ranges: list[tuple[int, int]] = []
        start = 0
        for size in sizes:
            end = start + size
            ranges.append((start, end))
            start = end
        return tuple(ranges)

    def _split_ranges(self, batch_size: int) -> tuple[tuple[int, int], ...]:
        target_size = (batch_size + self.policy.split_factor - 1) // self.policy.split_factor
        target_size = max(self.policy.min_items, target_size)
        if target_size >= batch_size:
            return ()

        ranges: list[tuple[int, int]] = []
        start = 0
        while start < batch_size:
            end = min(start + target_size, batch_size)
            if batch_size - end and batch_size - end < self.policy.min_items:
                end = batch_size
            ranges.append((start, end))
            start = end

        if len(ranges) < 2:
            return ()
        if any(end - start < self.policy.min_items for start, end in ranges):
            return ()
        return tuple(ranges)
