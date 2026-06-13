# Development SPDL Tensor Adapter

This document describes the sixth data/runtime rewrite slice for
`enn_torch_dev`.

## Goal

SPDL is the intended input-processing engine for the rewrite, but the runtime
hot path must continue to receive tensor-backed batches:

```text
SPDL pipeline output
  -> Mapping[str, Tensor] / TensorDict
  -> SpdlTensorAdapter
  -> KVBatch
  -> RuntimeStep / GraphExecutor
```

This slice adds the stable adapter boundary only. It does not construct SPDL
pipelines or tune SPDL workers.

## Contract

`SpdlTensorAdapter` accepts tensorized batches in one of these forms:

- `Mapping[str, Tensor]`;
- `TensorDictBase`.

The adapter validates the payload against `DataSchema`, strips runtime identity
keys from the TensorDict payload, and returns either:

- a schema-validated `TensorDict` through `to_tensordict(...)`; or
- a `KVBatch` through `to_kvbatch(...)`.

Runtime identity keys are reserved and must not be declared as `DataSchema`
fields:

- `row_id`, generated with `torch.arange(batch_size)` when absent;
- `source_id`, optional;
- `sample_id`, optional.

All identity tensors must be 1-D integer tensors with the same length as the
batch size. They are moved to CPU `torch.long` tensors before entering
`KVBatch`.

## Why no direct SPDL import?

The adapter intentionally avoids importing SPDL. This keeps the ENN runtime
contract stable while SPDL source plugins, queue settings, and concurrency tuning
are developed independently.

SPDL-owned work belongs before this adapter:

```text
read / decode / preprocess / collate
  -> Mapping[str, Tensor] / TensorDict
```

ENN-owned runtime work belongs after this adapter:

```text
KVBatch
  -> BudgetedBatcher / DeviceTransferPolicy / RuntimeStep
  -> OOM recovery / AutoGovernor / ShardController
```

## Example

```python
import torch

from enn_torch_dev.data import (
    DataSchema,
    FieldSpec,
    KeyMapping,
    SpdlTensorAdapter,
)

schema = DataSchema(
    schema_id="demo.spdl",
    fields=(
        FieldSpec("features", torch.float32, shape=(None, 3)),
        FieldSpec("labels", torch.float32, shape=(None, 1), required=False),
    ),
    key_mapping=KeyMapping(inputs={"features": "x"}, labels={"labels": "y"}),
)

adapter = SpdlTensorAdapter(schema)
kvbatch = adapter.to_kvbatch(
    {
        "features": torch.zeros(32, 3),
        "labels": torch.zeros(32, 1),
        "row_id": torch.arange(1000, 1032),
    }
)
store = kvbatch.to_store(schema)
```

## Out of Scope

- SPDL pipeline construction.
- SPDL async prefetch or queue-depth tuning.
- Pinned memory.
- Device transfer.
- `BudgetedBatcher`.
- OOM recovery and batch split retry.
- AutoGovernor.
- ShardController and distributed resume.
- TorchData wrapper integration.
- DataFrame, Arrow, WebDataset, or Hugging Face source plugins.

## Test Commands

```bash
python -m pytest enn_torch_dev/debug/data/test_spdl_adapter.py -q
python -m pytest enn_torch_dev/debug/data -q
python -m pytest enn_torch_dev/debug -q
```

## Next Step

The next slice should add a minimal SPDL loader that wraps an iterable SPDL
pipeline and yields `KVBatch` objects through `SpdlTensorAdapter`. After that,
`BudgetedBatcher` can consume the same `KVBatch` stream used by `PlainLoader`.
