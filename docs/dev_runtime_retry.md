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
- `min_items`: hard floor for every retried subbatch; batches at or below this size are not split;
- `split_factor`: approximate number of chunks to split an oversized batch into;
- `retry_oom`: enable or disable OOM retry.

`RuntimeRetryRunner` exposes two entry points:

- `run_batch(batch)` for one `KVBatch`;
- `run_stream(source)` for an iterable `KVBatch` stream.

The runner preserves source order and row order. When a batch is split, subbatches
preserve `TensorDict` payload, `row_ids`, `source_ids`, `sample_ids`, `schema_id`,
and `shard_id` through the shared runtime slicing helper. The split `KVBatch`
uses materialized slices for both the `TensorDict` payload and identity tensors.
`min_items` applies to every retried subbatch; if a split plan would create a
smaller remainder and cannot merge it into a valid multi-chunk split, the runner
yields the original OOM result instead of retrying a microbatch.

## Retry Behavior

For each batch:

1. call `runtime_step.run(batch)`;
2. yield successful results unchanged;
3. yield non-OOM faults unchanged;
4. when the result is `StepStatus.OOM_FAULT`, `retry_oom=True`, retry depth is
   still available, the phase is side-effect-safe, the runtime step has no
   optimizer, and the batch is larger than `min_items`, compute a valid split
   plan;
5. retry only `RuntimePhase.TO_STORE`, `RuntimePhase.FORWARD`, and
   `RuntimePhase.LOSS` OOM results;
6. do not retry `RuntimePhase.BACKWARD`, `RuntimePhase.OPTIMIZER`, or
   `phase=None` OOM results;
7. before executing subbatches, drop failed full-batch OOM result references to
   `store` and `loss` so heavyweight intermediates are not kept alive by the
   retry loop;
8. execute split batches in row order;
9. yield the final OOM `StepResult` when the batch can no longer be split or the
   retry budget is exhausted.

The runner does not catch arbitrary Python exceptions from the runtime step. It
operates on the existing `RuntimeStep` fault-classification contract. This is a
side-effect-safe retry boundary only: runtime steps with a non-`None`
`optimizer` attribute are not retried, even for `FORWARD` OOM results.

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
- Optimizer updates or training-step retry semantics.
- Gradient accumulation.
- Optimizer state rollback or recovery.
- Preserving one-logical-batch-one-update semantics across split retries.
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
