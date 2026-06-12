from __future__ import annotations

from collections.abc import Callable

import torch

from enn_torch_dev.data import DataSchema, KVBatch
from enn_torch_dev.executor import GraphExecutor, KVStore

from .faults import RuntimePhase, StepResult, StepStatus


LossFn = Callable[[KVStore], torch.Tensor]


def _is_oom_error(exc: BaseException) -> bool:
    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom_type is not None and isinstance(exc, cuda_oom_type):
        return True
    if isinstance(exc, RuntimeError):
        return "out of memory" in str(exc).lower()
    return False


def _make_result(
    *,
    status: StepStatus,
    phase: RuntimePhase | None,
    batch: KVBatch,
    loss: torch.Tensor | None = None,
    store: KVStore | None = None,
    error: BaseException | None = None,
) -> StepResult:
    return StepResult(
        status=status,
        phase=phase,
        batch_size=batch.batch_size,
        row_ids=batch.row_ids.detach().cpu().clone(),
        loss=loss,
        store=store,
        error_type=None if error is None else type(error).__name__,
        error_message=None if error is None else str(error),
    )


class RuntimeStep:
    """Minimal executor runtime step with fault classification.

    RuntimeStep owns the step boundary only. It does not implement SPDL,
    prefetch, device transfer, dynamic batching, OOM recovery, AMP, precision
    fallback, or sharding.
    """

    def __init__(
        self,
        executor: GraphExecutor,
        *,
        schema: DataSchema,
        loss_fn: LossFn | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        zero_grad: bool = True,
        raise_unknown: bool = True,
    ) -> None:
        if not isinstance(executor, GraphExecutor):
            raise TypeError("RuntimeStep.executor must be a GraphExecutor.")
        if not isinstance(schema, DataSchema):
            raise TypeError("RuntimeStep.schema must be a DataSchema.")
        if loss_fn is not None and not callable(loss_fn):
            raise TypeError("RuntimeStep.loss_fn must be callable or None.")
        if optimizer is not None and not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                "RuntimeStep.optimizer must be a torch.optim.Optimizer or None."
            )
        if not isinstance(zero_grad, bool):
            raise TypeError("RuntimeStep.zero_grad must be a bool.")
        if not isinstance(raise_unknown, bool):
            raise TypeError("RuntimeStep.raise_unknown must be a bool.")

        self.executor = executor
        self.schema = schema
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.zero_grad = zero_grad
        self.raise_unknown = raise_unknown

    def _exception_result(
        self,
        *,
        batch: KVBatch,
        phase: RuntimePhase,
        error: BaseException,
        store: KVStore | None = None,
        loss: torch.Tensor | None = None,
    ) -> StepResult:
        if _is_oom_error(error):
            return _make_result(
                status=StepStatus.OOM_FAULT,
                phase=phase,
                batch=batch,
                loss=loss,
                store=store,
                error=error,
            )
        if phase is RuntimePhase.TO_STORE and isinstance(
            error,
            (KeyError, TypeError, ValueError),
        ):
            return _make_result(
                status=StepStatus.DATA_FAULT,
                phase=phase,
                batch=batch,
                loss=loss,
                store=store,
                error=error,
            )
        if self.raise_unknown:
            raise error
        return _make_result(
            status=StepStatus.RUNTIME_FAULT,
            phase=phase,
            batch=batch,
            loss=loss,
            store=store,
            error=error,
        )

    def run(self, batch: KVBatch) -> StepResult:
        if not isinstance(batch, KVBatch):
            raise TypeError("RuntimeStep.run expects a KVBatch.")

        phase = RuntimePhase.TO_STORE
        try:
            store = batch.to_store(self.schema)
        except BaseException as exc:
            return self._exception_result(batch=batch, phase=phase, error=exc)

        phase = RuntimePhase.FORWARD
        try:
            self.executor.run(store)
        except BaseException as exc:
            return self._exception_result(
                batch=batch,
                phase=phase,
                store=store,
                error=exc,
            )

        loss: torch.Tensor | None = None
        if self.loss_fn is not None:
            phase = RuntimePhase.LOSS
            try:
                loss = self.loss_fn(store)
                if not isinstance(loss, torch.Tensor):
                    raise TypeError(
                        "RuntimeStep.loss_fn must return a torch.Tensor, "
                        f"got {type(loss)!r}."
                    )
                if not bool(torch.isfinite(loss.detach()).all()):
                    return _make_result(
                        status=StepStatus.NONFINITE_FAULT,
                        phase=phase,
                        batch=batch,
                        loss=loss,
                        store=store,
                    )
            except BaseException as exc:
                return self._exception_result(
                    batch=batch,
                    phase=phase,
                    loss=loss,
                    store=store,
                    error=exc,
                )

        if loss is not None and self.optimizer is not None:
            if self.zero_grad:
                phase = RuntimePhase.OPTIMIZER
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                except BaseException as exc:
                    return self._exception_result(
                        batch=batch,
                        phase=phase,
                        loss=loss,
                        store=store,
                        error=exc,
                    )

            phase = RuntimePhase.BACKWARD
            try:
                loss.backward()
            except BaseException as exc:
                return self._exception_result(
                    batch=batch,
                    phase=phase,
                    loss=loss,
                    store=store,
                    error=exc,
                )

            phase = RuntimePhase.OPTIMIZER
            try:
                self.optimizer.step()
            except BaseException as exc:
                return self._exception_result(
                    batch=batch,
                    phase=phase,
                    loss=loss,
                    store=store,
                    error=exc,
                )

        return _make_result(
            status=StepStatus.SUCCESS,
            phase=None,
            batch=batch,
            loss=loss,
            store=store,
        )
