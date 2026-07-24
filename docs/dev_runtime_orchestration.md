# Conservative Runtime Orchestration

This document describes the eleventh data/runtime rewrite slice for
`enn_torch_dev`.

## Goal

`BudgetedBatcher`, `RuntimeRetryRunner`, and `ConservativeRuntimeGovernor` are
separate runtime components. This slice adds a thin finite pass-level helper that
wires them together without folding their responsibilities into one class:

```text
KVBatch source
  -> BudgetedBatcher(current_budget)
  -> RuntimeRetryRunner
  -> StepResult stream
  -> optional fixed or pass-scoped ResourceCapacity
  -> all raw-attempt ResourceSample records
  -> assess_resource_pressure(...)
  -> ConservativeRuntimeGovernor.observe_results(...)
  -> GovernorDecision.next_budget
```

The orchestrator is not AutoGovernor. It does not learn, persist calibration
state, manage devices, checkpoint, or recover training semantics. It only runs a
finite pass and returns the observed results plus the governor decision for the
next pass.

## Public Objects

`enn_torch_dev.runtime` exports:

- `ResourceCapacityProvider`
- `RuntimePassResult`
- `ConservativeRuntimeOrchestrator`

The stable `enn_torch` namespace does not expose this development orchestrator.

## Contract

`ConservativeRuntimeOrchestrator` accepts:

- a `RuntimeStep`-compatible object that provides `run(KVBatch) -> StepResult`;
- a `ConservativeRuntimeGovernor` holding the active budget;
- optional `RetryPolicy` for `RuntimeRetryRunner`;
- optional `DataCostProbe` passed through to `BudgetedBatcher`;
- optional fixed `ResourceCapacity` used to assess resource samples;
- optional `ResourceCapacityProvider` resolved exactly once at each pass start;
- `split_oversized` and `min_items` values passed through to `BudgetedBatcher`.

`run_pass(source)` accepts a finite iterable of `KVBatch` objects and returns a
`RuntimePassResult` with:

- `results`: the finite tuple of yielded `StepResult` records;
- `decision`: the `GovernorDecision` produced after observing the pass results;
- `recovered_oom`: whether an internal OOM was observed by the retry runner but
  the final yielded results did not include an OOM result;
- `resource_capacity`: the fixed or provider-resolved capacity used for that pass.

`current_budget` and `last_decision` proxy the governor's current state.

## Recovered OOM Signaling

`RuntimeRetryRunner` may recover a full-batch OOM by yielding successful split
subbatch results. If the governor only sees those final successes, it could grow
the budget that caused retry churn.

The orchestrator wraps the runtime step with a small tracker that observes raw
`RuntimeStep.run(...)` results before `RuntimeRetryRunner` decides whether to
retry. When the tracker sees an OOM but the final pass results contain no OOM,
`run_pass(...)` calls:

```python
ConservativeRuntimeGovernor.observe_results(results, recovered_oom=True)
```

This keeps retry-recovered OOM pressure visible to the governor while preserving
`RuntimeRetryRunner`'s side-effect-safe retry boundary.

## Capacity Resolution and Pressure Summary Wiring

`resource_capacity` and `resource_capacity_provider` are mutually exclusive.
A provider is called exactly once after source type validation and before the
source is consumed. The returned `ResourceCapacity` remains fixed for that
entire pass, including all retry and split attempts. Provider exceptions and
invalid return types propagate without source consumption or governor updates.

When a fixed or provider-resolved capacity is available, the same wrapper records
`ResourceSample` objects from every raw runtime-step result, including OOM results
that are consumed internally by retry and do not appear in the final yielded
`results` tuple. After the finite retry stream completes, the orchestrator calls:

```python
resolved_capacity = fixed_capacity_or_provider_capacity
pressure_summary = assess_resource_pressure(raw_attempt_samples, resolved_capacity)

decision = governor.observe_results(
    results,
    recovered_oom=recovered_oom,
    pressure_summary=pressure_summary,
)
```

The orchestrator does not create a `ResourceMonitor`. A caller may explicitly
supply an existing `ResourceMonitor` as the provider because it already exposes
`capacity()`. Without fixed capacity or a provider, no pressure summary is
constructed and the previous orchestration behavior is preserved.

A CUDA sample/capacity device mismatch remains an error from
`assess_resource_pressure(...)`; the orchestrator does not hide it or update the
governor state after that failed assessment.

## Materialization Boundary

`RuntimePassResult.results` is a tuple. This is intentional for this finite
pass-level helper so callers can inspect the pass outcome together with the
budget decision. The raw-attempt sample list is pass-local and is reduced to the
scalar-only `ResourcePressureSummary` stored in `GovernorDecision`. The resolved
`ResourceCapacity` is also a scalar-only frozen record stored as pass provenance.

Do not use this class as an unbounded production streaming runner. A future
streaming orchestration slice can avoid pass-level materialization while still
surfacing recovered-OOM signals.

## Relationship to Summary

`docs/dev_runtime_summary.md` describes a lightweight inspection layer for
finished `RuntimePassResult` objects. The summary layer turns finite pass results
and governor decisions into compact records and stable debug text without running
models, retrying batches, changing budgets, or persisting logs.

## Relationship to Session

`docs/dev_runtime_session.md` describes a bounded lazy coordinator for multiple
finite pass sources. The session calls `run_pass(...)` once per yielded record
and leaves per-pass materialization, retry, and governor behavior inside the
orchestrator.

## Out of Scope

- AutoGovernor full implementation.
- Learned or model-specific tuning.
- Persistent calibration caches or history databases.
- Unbounded production streaming orchestration.
- Automatic `ResourceMonitor` creation.
- Mid-pass capacity refresh or free-memory admission control.
- Pressure-triggered budget shrink or field-specific tuning.
- SPDL queue-depth tuning.
- Device transfer.
- AMP or precision fallback.
- Checkpoint/resume.
- Optimizer rollback or training semantic recovery.
- Distributed coordination.

## Test Commands

```bash
python -m pytest enn_torch_dev/debug/runtime/test_runtime_capacity_provider.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_orchestration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_governor.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_retry.py -q
python -m pytest enn_torch_dev/debug/runtime/test_budgeted_batcher.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug/data -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```
