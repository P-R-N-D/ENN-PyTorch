# Development Runtime Workflow

This document describes the current supported composition for the bounded
single-node development runtime under `enn_torch_dev.runtime`.

The APIs in this document are active-development APIs. They are not exported by
the stable `enn_torch` namespace.

## Supported flow

```text
static footprints + reference BatchCost + ResourceCapacity
  -> optional recommend_initial_batch_budget
finite KVBatch pass source
  -> BudgetedBatcher
  -> RuntimeRetryRunner
  -> optional fixed or pass-scoped capacity pressure assessment
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

## Optional initial budget recommendation

Use `recommend_initial_batch_budget(...)` when the first governor budget should be
derived from known static facts instead of supplied manually:

```python
from enn_torch_dev.runtime import (
    InitialBatchBudgetPolicy,
    ModelFootprint,
    OptimizerFootprint,
    ResourceCapacity,
    recommend_initial_batch_budget,
)

recommendation = recommend_initial_batch_budget(
    ResourceCapacity(
        cpu_total_bytes=host_bytes,
        cuda_total_bytes=device_bytes,
        cuda_device_index=0,
    ),
    reference_batch_cost,
    reference_device_bytes_by_device={
        "cuda:0": reference_batch_cost.device_bytes or 0,
    },
    model_footprint=ModelFootprint.from_module(model),
    optimizer_footprint=OptimizerFootprint.from_optimizer(optimizer),
    policy=InitialBatchBudgetPolicy(
        max_items=32,
        fallback_max_items=1,
    ),
)

governor = ConservativeRuntimeGovernor(recommendation.recommended_budget)
```

The helper is deterministic and side-effect free. It does not execute the model,
consume a source, mutate governor or history state, or decide whether a pass is
admissible. Model and optimizer footprints retain device-resolved byte maps so
CPU and the configured CUDA device are accounted independently. Positive
reference device bytes likewise require an explicit mapping to the matching
`cuda:<index>`; aggregate non-CPU cost is not assigned to an arbitrary CUDA
capacity. Zero byte totals remain non-limiting without an item count. The result
preserves the original capacity, reference cost, resolved policy, and normalized
reference device mapping for audit and reproduction. Unknown capacity or
positive per-item cost remains unknown; an explicit `fallback_max_items` is
required when a finite limit cannot otherwise be derived. See
[`dev_initial_batch_budget.md`](dev_initial_batch_budget.md) for formulas and
boundaries.

## Optional pressure-aware composition

The governor pressure guard becomes operational when the caller supplies either
a fixed `ResourceCapacity` or a pass-scoped `ResourceCapacityProvider`.
`ResourceMonitor` already implements the provider contract, so one monitor can
produce both runtime samples and pass-start capacity snapshots:

```python
from enn_torch_dev.runtime import ResourceMonitor, RuntimeStep

governor = ConservativeRuntimeGovernor(
    BatchBudget(max_items=8),
    policy=GovernorPolicy(
        grow_after_successes=3,
        max_pressure_ratio_for_growth=0.8,
    ),
)

monitor = ResourceMonitor(cuda_device=0)
runtime_step = RuntimeStep(..., resource_monitor=monitor)

orchestrator = ConservativeRuntimeOrchestrator(
    runtime_step,
    governor,
    retry_policy=RetryPolicy(max_retry_depth=3, min_items=1, split_factor=2),
    resource_capacity_provider=monitor,
)
```

The provider is called exactly once at each pass start, before the pass source is
consumed. Its result remains fixed for that pass, including retry and split
attempts. A caller may still use `resource_capacity=...` for a fixed snapshot, but
fixed capacity and a provider cannot be configured together.

If neither fixed capacity nor a provider is supplied, the orchestrator passes no
pressure summary and the existing governor contract applies. Pressure may suppress
success-driven growth when the opt-in guard is enabled. Separately configured
sustained high pressure may shrink only the next-pass budget after its configured
pass count is reached; a single non-OOM pressure sample cannot shrink a budget.

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
- optionally shrinks the next-pass budget only after a configured sustained
  high-pressure streak, tracking CPU and CUDA persistence independently and
  selecting host bytes for CPU pressure and device bytes for CUDA pressure;
- allows CPU and CUDA shrink thresholds, required pass counts, and sustained-
  pressure shrink factors to override the common policy independently;
- applies each dimension's effective factor to its matching byte budget;
- falls back to `max_items` only when no matching triggered byte budget is
  configured, using the triggered dimension's factor or the smaller factor when
  both dimensions share the fallback;
- keeps yielded and retry-recovered OOM shrink on the common `shrink_factor`;
- preserves an incomplete streak for one dimension when only the other dimension
  reaches its effective shrink threshold and required pass count;
- applies configured minimum and maximum bounds.

The next pass uses the governor's current budget.

## Inspection and retention

A yielded `RuntimeSessionRecord` contains the finite pass result for immediate
caller inspection. The caller controls how long that record remains alive.

The session itself does not retain prior pass results after the generator resumes.
Longer-lived in-memory retention is limited to lightweight `RuntimePassSummary`
objects in `RuntimePassHistory`, which requires a positive `max_records` bound.
Pass summaries expose scalar pressure ratios, growth-suppression decisions, and
sustained-pressure shrink feedback. They also expose structured high/triggered
dimensions, selected adjustment fields, field-level applied factors, and actual
changed fields without requiring `decision_reason` parsing. History aggregation
uses that structured provenance to count retained CPU/CUDA high and trigger
passes, adjustment attempts, full no-ops, triggers without matching budgets, and
actual host/device/items shrink passes. CPU and CUDA can each contribute for one
pass, while attempt/no-op/trigger-without-budget counts increment at most once per
pass; a partial change is not a full no-op. Existing pressure-assessed,
pressure-suppressed, actual pressure-shrink, and peak-ratio aggregates remain. All
counts are recomputed only within the currently retained summary window, and OOM
status or ratios never substitute for empty structured provenance. Each pass
summary also records the scalar capacity used for normalization. Raw
`ResourceSample` records are not retained by summary or history.

## Fault and exception semantics

A `StepStatus` fault is a completed runtime result. It does not automatically stop
the session. The governor observes it according to its documented policy.

Provider, source-iteration, and pass-execution exceptions are not suppressed. A
provider failure occurs before source consumption and before governor updates.
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
- automatic `ResourceMonitor` creation;
- mid-pass capacity refresh or free-memory admission control;
- proof that an initial recommendation is safe for unobserved activation or allocator costs;
- learned field weights;
- stable `enn_torch` API exposure.

Use small synthetic inputs for baseline validation. Do not use this workflow as an
unbounded production streaming runner.

## Validation

```bash
python -m pytest enn_torch_dev/debug/runtime/test_runtime_capacity_provider.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_source_factory.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```
