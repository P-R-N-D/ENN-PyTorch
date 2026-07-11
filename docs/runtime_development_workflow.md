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
  -> ConservativeRuntimeGovernor
  -> ConservativeRuntimeOrchestrator
  -> RuntimePassSummary
  -> RuntimePassHistory
  -> ConservativeRuntimeSession
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

## Source contract

`pass_sources` is an outer iterable of inner pass sources.

- The outer session is bounded by `max_passes`.
- Every inner source must be a finite iterable of `KVBatch`.
- The session does not fetch an extra outer source after reaching `max_passes`.
- Each `next()` call on the session iterator executes at most one finite pass.

The session does not recreate or replay a source. Callers that require replay
must construct a new finite source explicitly.

## OOM and budget behavior

`RuntimeRetryRunner` may split a retryable OOM batch into smaller subbatches.
`ConservativeRuntimeOrchestrator` reports a retry-recovered OOM to the governor
even when the final yielded results are successful.

The conservative governor then:

- shrinks configured budget fields after a yielded or retry-recovered OOM;
- keeps the current budget after non-OOM faults;
- grows configured fields only after the configured clean-success threshold;
- applies configured minimum and maximum bounds.

The next pass uses the governor's current budget.

## Inspection and retention

A yielded `RuntimeSessionRecord` contains the finite pass result for immediate
caller inspection. The caller controls how long that record remains alive.

The session itself does not retain prior pass results after the generator resumes.
Longer-lived in-memory retention is limited to lightweight `RuntimePassSummary`
objects in `RuntimePassHistory`, which requires a positive `max_records` bound.
History aggregates describe only the currently retained summary window.

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
- automatic source replay;
- distributed execution or aggregation;
- AutoGovernor or learned tuning;
- ResourceMonitor feedback-loop tuning;
- stable `enn_torch` API exposure.

Use small synthetic inputs for baseline validation. Do not use this workflow as an
unbounded production streaming runner.

## Validation

```bash
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```
