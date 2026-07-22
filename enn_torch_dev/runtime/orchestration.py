from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from enn_torch_dev.data import KVBatch

from .batching import BatchBudget, BudgetedBatcher
from .cost import DataCostProbe
from .faults import ResourceSample, StepResult, StepStatus
from .governor import ConservativeRuntimeGovernor, GovernorDecision
from .pressure import ResourceCapacity, assess_resource_pressure
from .retry import RetryPolicy, RuntimeRetryRunner, RuntimeStepProtocol


@dataclass(frozen=True, slots=True)
class RuntimePassResult:
    """Result record for one conservative runtime orchestration pass."""

    results: tuple[StepResult, ...]
    decision: GovernorDecision
    recovered_oom: bool = False


class _OomTrackingRuntimeStep:
    def __init__(
        self,
        runtime_step: RuntimeStepProtocol,
        *,
        collect_resource_samples: bool = False,
    ) -> None:
        if not isinstance(runtime_step, RuntimeStepProtocol):
            raise TypeError("runtime_step must provide run(KVBatch).")
        self.runtime_step = runtime_step
        self.optimizer = getattr(runtime_step, "optimizer", None)
        self.saw_oom = False
        self.collect_resource_samples = collect_resource_samples
        self.resource_samples: list[ResourceSample] = []

    def run(self, batch: KVBatch) -> StepResult:
        result = self.runtime_step.run(batch)
        if isinstance(result, StepResult):
            if self.collect_resource_samples:
                self.resource_samples.extend(result.resource_samples)
            if result.status is StepStatus.OOM_FAULT:
                self.saw_oom = True
        return result


class ConservativeRuntimeOrchestrator:
    """Wire budget, retry, and governor components for one finite runtime pass."""

    def __init__(
        self,
        runtime_step: RuntimeStepProtocol,
        governor: ConservativeRuntimeGovernor,
        *,
        retry_policy: RetryPolicy | None = None,
        cost_probe: DataCostProbe | None = None,
        resource_capacity: ResourceCapacity | None = None,
        split_oversized: bool = True,
        min_items: int = 1,
    ) -> None:
        if not isinstance(runtime_step, RuntimeStepProtocol):
            raise TypeError(
                "ConservativeRuntimeOrchestrator.runtime_step must provide run(KVBatch)."
            )
        if not isinstance(governor, ConservativeRuntimeGovernor):
            raise TypeError(
                "ConservativeRuntimeOrchestrator.governor must be a ConservativeRuntimeGovernor."
            )
        if retry_policy is not None and not isinstance(retry_policy, RetryPolicy):
            raise TypeError(
                "ConservativeRuntimeOrchestrator.retry_policy must be a RetryPolicy or None."
            )
        if cost_probe is not None and not isinstance(cost_probe, DataCostProbe):
            raise TypeError(
                "ConservativeRuntimeOrchestrator.cost_probe must be a DataCostProbe or None."
            )
        if resource_capacity is not None and not isinstance(
            resource_capacity, ResourceCapacity
        ):
            raise TypeError(
                "ConservativeRuntimeOrchestrator.resource_capacity must be a "
                "ResourceCapacity or None."
            )
        if not isinstance(split_oversized, bool):
            raise TypeError("ConservativeRuntimeOrchestrator.split_oversized must be a bool.")
        if not isinstance(min_items, int) or isinstance(min_items, bool):
            raise TypeError("ConservativeRuntimeOrchestrator.min_items must be an integer.")
        if min_items <= 0:
            raise ValueError("ConservativeRuntimeOrchestrator.min_items must be positive.")

        self.runtime_step = runtime_step
        self.governor = governor
        self.retry_policy = retry_policy
        self.cost_probe = cost_probe
        self.resource_capacity = resource_capacity
        self.split_oversized = split_oversized
        self.min_items = min_items

    @property
    def current_budget(self) -> BatchBudget:
        return self.governor.current_budget

    @property
    def last_decision(self) -> GovernorDecision | None:
        return self.governor.state.last_decision

    def run_pass(self, source: Iterable[KVBatch]) -> RuntimePassResult:
        if isinstance(source, KVBatch):
            raise TypeError(
                "ConservativeRuntimeOrchestrator.run_pass expects an iterable of KVBatch."
            )
        if not isinstance(source, Iterable):
            raise TypeError(
                "ConservativeRuntimeOrchestrator.run_pass expects an iterable of KVBatch."
            )

        tracking_step = _OomTrackingRuntimeStep(
            self.runtime_step,
            collect_resource_samples=self.resource_capacity is not None,
        )
        budgeted = BudgetedBatcher(
            source,
            self.governor.current_budget,
            cost_probe=self.cost_probe,
            split_oversized=self.split_oversized,
            min_items=self.min_items,
        )
        retry_runner = RuntimeRetryRunner(tracking_step, policy=self.retry_policy)
        results = tuple(retry_runner.run_stream(budgeted))
        yielded_oom = any(result.status is StepStatus.OOM_FAULT for result in results)
        recovered_oom = tracking_step.saw_oom and not yielded_oom
        pressure_summary = (
            assess_resource_pressure(
                tracking_step.resource_samples,
                self.resource_capacity,
            )
            if self.resource_capacity is not None
            else None
        )
        decision = self.governor.observe_results(
            results,
            recovered_oom=recovered_oom,
            pressure_summary=pressure_summary,
        )
        return RuntimePassResult(
            results=results,
            decision=decision,
            recovered_oom=recovered_oom,
        )
