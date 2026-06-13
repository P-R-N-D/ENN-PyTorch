# Development Budgeted Batcher

This document describes the eighth data/runtime rewrite slice for
`enn_torch_dev`.

## Goal

`PlainLoader` and `SPDLLoader` now expose the same runtime-facing stream type:

```text
Iterator[KVBatch]
```

This slice adds a minimal budget gate over that stream:

```text
PlainLoader / SPDLLoader
  -> Iterator[KVBatch]
  -> BudgetedBatcher
  -> Iterator[KVBatch]
  -> RuntimeStep / GraphExecutor
```

The goal is not automatic OOM recovery or learned batch-size tuning yet. The
first `BudgetedBatcher` only passes, splits, or rejects `KVBatch` objects using a
static `BatchBudget` and optional `DataCostProbe` estimates.

## Contract

`BatchBudget` supports these limits:

- `max_host_bytes`;
- `max_device_bytes`;
- `max_items`.

At least one limit must be set. Each limit must be `None` or a non-negative
integer.

`BudgetedBatcher` accepts an iterable of `KVBatch` objects and yields a new
`KVBatch` stream. For each batch it follows this order:

1. use `KVBatch.cost_hint` when it contains every field required by the active
   budget;
2. when the hint is partial and a `DataCostProbe` is available, recalculate the
   required payload cost with the probe;
3. otherwise estimate cost with `DataCostProbe` when one is provided;
4. otherwise fall back to `BatchCost(num_items=batch.batch_size)`;
5. pass batches that fit the budget;
6. split oversized batches along the batch axis when `split_oversized=True`;
7. raise `BatchBudgetExceeded` when a batch cannot fit safely.

The batcher never drops data silently. Split batches preserve row order,
`row_ids`, `source_ids`, `sample_ids`, `schema_id`, and `shard_id`.

## Cost Information

`max_items` can work with the fallback item count. Byte budgets require byte cost
information from either a complete `KVBatch.cost_hint` or `DataCostProbe`. If a
hint is partial and a probe is available, the probe result is used so required
fields are not hidden by the partial hint. If a byte budget is configured but
byte cost fields are still unavailable, `BudgetedBatcher` raises an error
instead of guessing.

For `BudgetedBatcher`, byte budgets include both the `TensorDict` payload and
the `KVBatch` identity tensors (`row_ids`, `source_ids`, and `sample_ids`) that
become runtime store identity payload. Optional identity tensors are skipped when
they are `None`.

Split batches do not inherit the parent `cost_hint` unchanged. When a
`DataCostProbe` is available, split subbatches clear the hint and recompute cost
from their materialized slice. Without a probe, subbatches receive a conservative
ceil-scaled hint derived from the parent cost, including `host_bytes`,
`device_bytes`, `num_tokens`, and `num_tiles` when those fields are present;
`num_items` is set to the subbatch size.

## Example

```python
from enn_torch_dev.runtime import BatchBudget, BudgetedBatcher, DataCostProbe

budgeted = BudgetedBatcher(
    SPDLLoader(spdl_batches, adapter, cost_probe=DataCostProbe()),
    BatchBudget(max_items=32, max_host_bytes=512 * 1024 * 1024),
    cost_probe=DataCostProbe(),
)

for batch in budgeted:
    result = runtime_step.run(batch)
```

## Out of Scope

- Running `RuntimeStep`.
- Catching OOM errors.
- Retrying failed batches.
- Learning or auto-tuning batch sizes.
- AutoGovernor feedback loops.
- ResourceMonitor feedback integration.
- GPU profile presets.
- Calibration cache.
- SPDL queue-depth tuning.
- Device transfer.
- AMP or precision fallback.

## Test Commands

```bash
python -m pytest enn_torch_dev/debug/runtime/test_budgeted_batcher.py -q
python -m pytest enn_torch_dev/debug/runtime/test_spdl_loader.py -q
python -m pytest enn_torch_dev/debug/runtime/test_plain_loader.py -q
python -m pytest enn_torch_dev/debug/runtime/test_data_cost_probe.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug/data -q
python -m pytest enn_torch_dev/debug -q
```

## Next Step

The next runtime-facing slice should add an OOM retry runner that uses this
budget boundary as its deterministic split mechanism. That runner should catch
OOM-class faults from `RuntimeStep`, split or shrink the failed batch, and retry
without making `BudgetedBatcher` responsible for model execution.
