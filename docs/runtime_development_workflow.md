# Development Runtime Workflow

This document describes the current supported composition for the bounded
single-node development runtime under `enn_torch_dev.runtime`.

The APIs in this document are active-development APIs. They are not exported by
the stable `enn_torch` namespace.

## Supported flow

```text
finite KVBatch pass source
  -> BudgetedBatcher
  -> RuntimeRetryRunner
  -> optional fixed-capacity pressure assessment
  -> ConservativeRuntimeGovernor
  -> ConservativeRuntimeOrchestrator
  -> RuntimePassSummary
  -> RuntimePassHistory
  -> ConservativeRuntimeSession
  -> optional RuntimePassSourceFactory for fresh per-pass sources
```

`ConservativeRuntimeSession` connects existing components across multiple finite
passes. It does not replace the retry, governor, summary, or history contracts.

## Minimal composition

```python
from enn_torch_dev.runtime import (
    BatchBudget,
    ConservativeRuntimeGovernor,
    ConservativeRuntimeOrchestrator,
    ConservativeRuntimeSession,
    GovernorPolicy,
    RetryPolicy,
    RuntimePassHistory,
)

runtime_step = ...  # object providing run(KVBatch) -> StepResult

governor = ConservativeRuntimeGovernor(
    BatchBudget(max_items=8),
    policy=GovernorPolicy(
        shrink_factor=0.5,
        grow_factor=2.0,
        grow_after_successes=3,
        min_items=1,
        max_items=32,
    ),
)

orchestrator = ConservativeRuntimeOrchestrator(
    runtime_step,
    governor,
    retry_policy=RetryPolicy(
        max_retry_depth=3,
        min_items=1,
        split_factor=2,
    ),
)

history = RuntimePassHistory(max_records=10)

session = ConservativeRuntimeSession(
    orchestrator,
    history,
    max_passes=20,
)

for record in session.run_passes(pass_sources):
    print(record.pass_index, record.pass_summary)
```

## Optional pressure-aware composition

The governor pressure guard becomes operational in orchestration only when the
caller supplies a fixed `ResourceCapacity`:

```python
from enn_torch_dev.runtime import ResourceCapacity

governor = ConservativeRuntimeGovernor(
    BatchBudget(max_items=8),
    policy=GovernorPolicy(
        grow_after_successes=3,
        max_pressure_ratio_for_growth=0.8,
    ),
)

orchestrator = ConservativeRuntimeOrchestrator(
    runtime_step,
    governor,
    retry_policy=RetryPolicy(max_retry_depth=3, min_items=1, split_factor=2),
    resource_capacity=ResourceCapacity(
        cpu_total_bytes=host_physical_bytes,
        cpu_limit_bytes=cgroup_limit_bytes,
        cuda_total_bytes=cuda_total_bytes,
        cuda_device_index=cuda_device_index,
    ),
)
```

The orchestrator includes resource samples from every raw runtime attempt, not
only the final yielded retry results. The supplied capacity remains fixed for the
orchestrator instance. Callers that need refreshed capacity must construct or
replace the orchestrator explicitly; this workflow does not poll capacity between
passes.

If no capacity is supplied, the orchestrator passes no pressure summary and the
existing governor contract applies. Pressure may suppress success-driven growth
when the opt-in guard is enabled, but it does not directly shrink a budget.

## Factory composition

Use a `RuntimePassSourceFactory` when each pass needs a newly constructed
one-shot loader or generator:

```python
from enn_torch_dev.runtime import RuntimePassSourceFactory


class PassFactory:
    def create_pass_source(self, pass_index: int):
        return build_finite_pass_source(pass_index)


for record in session.run_factory(PassFactory()):
    print(record.pass_index, record.pass_summary)
```

`run_factory(...)` is lazy and uses the same `max_passes` bound as
`run_passes(...)`. It calls the factory at most once per yielded record and does
not call it for an extra pass after the limit. Factory exceptions propagate, and
a pass whose source was not created is not appended to history.

## Source contract

`pass_sources` is an outer iterable of inner pass sources.

- The outer session is bounded by `max_passes`.
- Every inner source must be a finite iterable of `KVBatch`.
- The session does not fetch an extra outer source after reaching `max_passes`.
- Each `next()` call on the session iterator executes at most one finite pass.

`run_passes(...)` does not recreate or replay a source. `run_factory(...)` asks
caller-defined code to construct a fresh finite source for each pass, but it does
not cache sources or replay a consumed iterator.

## OOM and budget behavior

`RuntimeRetryRunner` may split a retryable OOM batch into smaller subbatches.
`ConservativeRuntimeOrchestrator` reports a retry-recovered OOM to the governor
even when the final yielded results are successful.

The conservative governor then:

- shrinks configured budget fields after a yielded or retry-recovered OOM;
- keeps the current budget after non-OOM faults;
- grows configured fields only after the configured clean-success threshold;
- optionally suppresses success growth when an explicit pressure summary is
  missing or reaches the configured growth limit;
- applies configured minimum and maximum bounds.

The next pass uses the governor's current budget.

## Inspection and retention

A yielded `RuntimeSessionRecord` contains the finite pass result for immediate
caller inspection. The caller controls how long that record remains alive.

The session itself does not retain prior pass results after the generator resumes.
Longer-lived in-memory retention is limited to lightweight `RuntimePassSummary`
objects in `RuntimePassHistory`, which requires a positive `max_records` bound.
Pass summaries expose scalar pressure ratios and growth-suppression decisions;
history aggregates pressure-assessed and pressure-suppressed pass counts plus the
highest known ratio only within the currently retained summary window. Raw
`ResourceSample` records are not retained by summary or history.

## Fault and exception semantics

A `StepStatus` fault is a completed runtime result. It does not automatically stop
the session. The governor observes it according to its documented policy.

Python exceptions from source iteration or pass execution are not suppressed.
The failing pass is not added to history when execution or summary construction
fails before history append. Previously completed history records remain intact.

## Safety boundary

This workflow is intended for bounded, single-node development execution.

It does not provide:

- persistent logging, JSONL, or CSV export;
- dashboards or telemetry backends;
- checkpoint/resume;
- automatic source replay or source caching;
- distributed execution or aggregation;
- AutoGovernor or learned tuning;
- automatic `ResourceMonitor` creation or capacity refresh;
- pressure-triggered budget shrink or field-specific tuning;
- stable `enn_torch` API exposure.

Use small synthetic inputs for baseline validation. Do not use this workflow as an
unbounded production streaming runner.

## Validation

```bash
python -m pytest enn_torch_dev/debug/runtime/test_runtime_source_factory.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```
