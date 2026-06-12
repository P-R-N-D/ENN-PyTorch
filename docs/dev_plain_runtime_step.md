# Development Plain Runtime Step

This document describes the third data/runtime rewrite slice for
`enn_torch_dev`.

## Goal

PR #795 introduced the tensor batch contract. PR #796 introduced
MemoryMappedTensor-backed staging and reading. This slice adds the smallest
runtime execution path:

```text
TensorDictReader
  -> PlainLoader
  -> KVBatch
  -> RuntimeStep
  -> KVStore
  -> GraphExecutor
  -> StepResult
```

The goal is not performance tuning. The goal is to create a reproducible step
boundary that preserves row identity and classifies basic failures.

## PlainLoader

`PlainLoader` is a thin sequential wrapper over `TensorDictReader`.

It intentionally does not implement:

- worker pools;
- shuffling;
- async prefetch;
- pinned memory;
- device transfer;
- SPDL integration.

Those features need a stable step boundary before they can be tuned safely.

## RuntimeStep

`RuntimeStep` runs one `KVBatch` through:

1. `KVBatch.to_store(schema)`;
2. optional `optimizer.zero_grad(set_to_none=True)` before forward;
3. `GraphExecutor.run(store)`;
4. optional `loss_fn(store)`;
5. optional `loss.backward()`;
6. optional `optimizer.step()`.

`loss_fn(store)` is intentionally store-based so graph outputs, labels, masks,
and future metadata can be read without coupling the runtime to one model
signature.

## StepResult

`StepResult` records:

- `status`;
- `phase`;
- `batch_size`;
- `row_ids`;
- optional `loss`;
- optional `store`;
- optional error type/message.

`row_ids` are preserved for future retry, batch split, resume, and sharding
logic.

## Fault Classification

This slice classifies:

- `SUCCESS`;
- `OOM_FAULT`;
- `NONFINITE_FAULT`;
- `DATA_FAULT`;
- `RUNTIME_FAULT`.

Current phase values are:

- `TO_STORE`;
- `FORWARD`;
- `LOSS`;
- `BACKWARD`;
- `OPTIMIZER`.

## Current Fault Boundaries

`DATA_FAULT` is restricted to `KVBatch.to_store(schema)`, where schema/key/dtype
and shape contract failures occur.

`OOM_FAULT` covers `torch.cuda.OutOfMemoryError` and runtime errors whose
message contains `out of memory`.

`NONFINITE_FAULT` covers NaN/Inf loss values before backward.

Unknown runtime errors are re-raised by default. Tests can set
`raise_unknown=False` to receive `RUNTIME_FAULT` instead.

## Out of Scope

- SPDL pipeline.
- Async prefetch.
- Worker pool.
- Pinned memory.
- Device transfer.
- ResourceMonitor.
- ModelFootprint.
- ModelCostProbe.
- AutoGovernor.
- OOM recovery or batch split retry.
- AMP, autocast, GradScaler, or precision fallback.
- ShardController or distributed resume.
- Polars, PyArrow, WebDataset, or Hugging Face ingestion plugins.

## Test Commands

```bash
python -m pytest enn_torch_dev/debug/runtime/test_plain_loader.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_step.py -q
python -m pytest enn_torch_dev/debug/runtime/test_reader_runtime_integration.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug -q
```

## Next Step

The next slice should add `ModelFootprint` and `ResourceMonitor`.

Those components need the step boundary from this PR so they can connect memory
and resource samples to concrete runtime phases.
