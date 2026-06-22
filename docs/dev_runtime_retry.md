# Development Runtime Retry

This document describes the ninth data/runtime rewrite slice for `enn_torch_dev`.

## Goal

`RuntimeStep` classifies OOM errors as `StepStatus.OOM_FAULT`, but it deliberately
does not own recovery policy. `BudgetedBatcher` provides a deterministic
pre-execution budget gate. This slice adds a thin post-execution retry runner:

```text
KVBatch stream
  -> BudgetedBatcher
  -> RuntimeRetryRunner
  -> RuntimeStep.run(batch)
  -> StepResult stream
```

The goal is minimal OOM-class retry only. The runner observes `StepResult`
objects from a `RuntimeStep`-compatible object and splits failed `KVBatch` objects
when the result status is `StepStatus.OOM_FAULT`.

## Contract

`RetryPolicy` owns static retry limits:

- `max_retry_depth`: maximum split/retry depth after the first failed execution;
- `min_items`: do not split batches at or below this batch size;
- `split_factor`: approximate number of chunks to split an oversized batch into;
- `retry_oom`: enable or disable OOM retry.

`RuntimeRetryRunner` exposes two entry points:

- `run_batch(batch)` for one `KVBatch`;
- `run_stream(source)` for an iterable `KVBatch` stream.

The runner preserves source order and row order. When a batch is split, subbatches
preserve `TensorDict` payload, `row_ids`, `source_ids`, `sample_ids`, `schema_id`,
and `shard_id` through the shared runtime slicing helper. The split `KVBatch`
uses materialized slices for both the `TensorDict` payload and identity tensors.

## Retry Behavior

For each batch:

1. call `runtime_step.run(batch)`;
2. yield successful results unchanged;
3. yield non-OOM faults unchanged;
4. when the result is `StepStatus.OOM_FAULT`, `retry_oom=True`, retry depth is
   still available, and the batch is larger than `min_items`, split the batch;
5. execute split batches in row order;
6. yield the final OOM `StepResult` when the batch can no longer be split or the
   retry budget is exhausted.

The runner does not catch arbitrary Python exceptions from the runtime step. It
operates on the existing `RuntimeStep` fault-classification contract.

## Relationship to BudgetedBatcher

`BudgetedBatcher` is a pre-execution static budget gate. It can reduce incoming
batch sizes before model execution starts.

`RuntimeRetryRunner` is a post-execution fault recovery boundary. It reacts only
after `RuntimeStep.run(...)` returns an OOM-class `StepResult`.

These two components should remain separate so that budget estimation does not
own model execution and retry policy does not own resource/cost probing.

## Out of Scope

- AutoGovernor.
- Adaptive batch-size learning.
- ResourceMonitor feedback loops.
- ModelCostProbe-driven policy updates.
- GPU profile presets.
- Calibration cache.
- Device transfer.
- AMP or precision fallback.
- Checkpoint/resume.
- Distributed retry.
- SPDL queue-depth tuning.

## Test Commands

```bash
python -m pytest enn_torch_dev/debug/runtime/test_runtime_retry.py -q
python -m pytest enn_torch_dev/debug/runtime/test_budgeted_batcher.py -q
python -m pytest enn_torch_dev/debug/runtime/test_spdl_loader.py -q
python -m pytest enn_torch_dev/debug/runtime/test_plain_loader.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug/data -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```

## Next Step

The next runtime-facing slice can add a conservative governor layer that consumes
`BatchCost`, `ModelCost`, retry outcomes, and `ResourceMonitor` samples to adjust
future budgets. That governor should not be folded into `RuntimeRetryRunner`.
