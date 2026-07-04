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
  -> ConservativeRuntimeGovernor.observe_results(...)
  -> GovernorDecision.next_budget
```

The orchestrator is not AutoGovernor. It does not learn, persist calibration
state, manage devices, checkpoint, or recover training semantics. It only runs a
finite pass and returns the observed results plus the governor decision for the
next pass.

## Public Objects

`enn_torch_dev.runtime` exports:

- `RuntimePassResult`
- `ConservativeRuntimeOrchestrator`

The stable `enn_torch` namespace does not expose this development orchestrator.

## Contract

`ConservativeRuntimeOrchestrator` accepts:

- a `RuntimeStep`-compatible object that provides `run(KVBatch) -> StepResult`;
- a `ConservativeRuntimeGovernor` holding the active budget;
- optional `RetryPolicy` for `RuntimeRetryRunner`;
- optional `DataCostProbe` passed through to `BudgetedBatcher`;
- `split_oversized` and `min_items` values passed through to `BudgetedBatcher`.

`run_pass(source)` accepts a finite iterable of `KVBatch` objects and returns a
`RuntimePassResult` with:

- `results`: the finite tuple of yielded `StepResult` records;
- `decision`: the `GovernorDecision` produced after observing the pass results;
- `recovered_oom`: whether an internal OOM was observed by the retry runner but
  the final yielded results did not include an OOM result.

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

## Materialization Boundary

`RuntimePassResult.results` is a tuple. This is intentional for this finite
pass-level helper so callers can inspect the pass outcome together with the
budget decision.

Do not use this class as an unbounded production streaming runner. A future
streaming orchestration slice can avoid pass-level materialization while still
surfacing recovered-OOM signals.

## Out of Scope

- AutoGovernor full implementation.
- Learned or model-specific tuning.
- Persistent calibration caches or history databases.
- Unbounded production streaming orchestration.
- ResourceMonitor feedback-loop tuning.
- SPDL queue-depth tuning.
- Device transfer.
- AMP or precision fallback.
- Checkpoint/resume.
- Optimizer rollback or training semantic recovery.
- Distributed coordination.

## Test Commands

```bash
python -m pytest enn_torch_dev/debug/runtime/test_runtime_orchestration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_governor.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_retry.py -q
python -m pytest enn_torch_dev/debug/runtime/test_budgeted_batcher.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug/data -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```
