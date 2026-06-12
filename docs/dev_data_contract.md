# Development Data Contract

This document defines the first data/runtime rewrite slice.

## Goal

Training and inference runtime input must already be tensor-backed:

```text
TensorDict / MemoryMappedTensor
  -> KVBatch
  -> KVStore
  -> GraphExecutor / RuntimeStep
```

Source-native objects such as dataframe rows, Arrow tables, tar-shard sample dictionaries, or dataset rows belong to ingestion and staging plugins. They must be tensorized before entering the runtime path.

## Current Slice

This slice adds the minimum contract needed before runtime work begins:

- `FieldSpec`: one tensor field contract.
- `DataSchema`: required and optional tensor fields plus executor key mapping.
- `KeyMapping`: TensorDict key to `KVStore` key mapping.
- `TensorFieldManifest`: serializable field contract.
- `DatasetManifest`: dataset-level schema and storage metadata.
- `BatchCost`: estimated or measured tensor batch cost.
- `KVBatch`: TensorDict-backed batch that can be exposed to `KVStore`.

The slice does not add storage, SPDL, dynamic batching, recovery policy, precision policy, or sharding.

## Runtime Install

From a clean environment:

```bash
python -m pip install -e . -r requirements-dev.txt
```

The base project already depends on `torch` and `tensordict`; `requirements-dev.txt` adds test tooling.

## Test Commands

Run this slice:

```bash
python -m pytest enn_torch_dev/debug/data
```

Run the development debug suite:

```bash
python -m pytest enn_torch_dev/debug
```

## Contract Rules

1. `KVBatch` owns the TensorDict-backed batch and row identity.
2. `KVBatch.to_store(...)` is the bridge into executor-facing `KVStore`.
3. `row_id` is always written to the store for retry, split, resume, and shard checks.
4. Optional schema fields may be absent when converting through `DataSchema`.
5. Missing mapped keys still raise when converting through a raw `KeyMapping` because optional field metadata is unavailable.
6. `KVStore` remains a data plane. It does not own loading, sharding, recovery, queues, or scheduling policy.

## Out of Scope

- TensorDict or MemoryMappedTensor persistence.
- SPDL engine integration.
- RuntimeStep, optimizer, and loss handling.
- Hardware resource monitoring.
- AutoGovernor and calibration cache.
- ShardController and distributed resume.
- Optional ingestion plugins.

## Next Slices

1. TensorDict and MemoryMappedTensor staging.
2. Plain loader plus `RuntimeStep` fault classification.
3. Model footprint and resource monitoring.
4. SPDL static tensor pipeline.
5. Telemetry and `run_profile.json`.
6. AutoGovernor, BudgetedBatcher, recovery, precision runtime, and sharding.
