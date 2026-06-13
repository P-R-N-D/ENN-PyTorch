# Development SPDL Loader

This document describes the seventh data/runtime rewrite slice for
`enn_torch_dev`.

## Goal

`SpdlTensorAdapter` defines the batch contract for tensorized SPDL output. This
slice adds the minimal iterable loader that turns an already tensorized SPDL-like
batch stream into the same `KVBatch` stream produced by `PlainLoader`:

```text
SPDL pipeline / iterable tensor source
  -> SPDLLoader
  -> SpdlTensorAdapter
  -> KVBatch
  -> RuntimeStep / GraphExecutor
```

`SPDLLoader` deliberately does not import SPDL. It accepts an iterable whose
items are already `Mapping[str, Tensor]` or `TensorDictBase` batches.

## Contract

`SPDLLoader` owns only the stream boundary:

- validate that the source is an iterable of tensor batches, not a single batch;
- validate that the adapter is a `SpdlTensorAdapter`;
- convert each source item through `adapter.to_kvbatch(...)`;
- pass an optional `shard_id` to every produced `KVBatch`;
- optionally attach a coarse `BatchCost` hint from `DataCostProbe`.

The loader is sequential and deterministic. It does not prefetch, shuffle,
spawn workers, transfer tensors to devices, or choose batch sizes.

`SPDLLoader` stores and iterates over the provided source object directly. If
the source is a one-shot iterator or generator, it is consumed after one full
pass and the same loader instance will not replay batches for another epoch.
Callers that need epoch-level replay should provide a re-iterable source, build
a new loader/source per epoch, or introduce a future source-factory layer.

## Cost Hint

When a `DataCostProbe` is provided, `SPDLLoader` estimates the produced
`KVBatch` and stores a coarse `BatchCost` in `KVBatch.cost_hint`:

- CPU tensor bytes are recorded as `host_bytes`;
- non-CPU tensor bytes are recorded as `device_bytes`;
- `DataCost.batch_size` is recorded as `num_items`.

This hint is intentionally smaller than `DataCost`. Rich per-tensor cost records
remain available through `DataCostProbe` for future calibration and governor
logic.

## Example

```python
import torch

from enn_torch_dev.data import DataSchema, FieldSpec, KeyMapping, SpdlTensorAdapter
from enn_torch_dev.runtime import DataCostProbe, SPDLLoader, RuntimeStep

schema = DataSchema(
    schema_id="demo.spdl.loader",
    fields=(
        FieldSpec("features", torch.float32, shape=(None, 3)),
        FieldSpec("labels", torch.float32, shape=(None, 1), required=False),
    ),
    key_mapping=KeyMapping(inputs={"features": "x"}, labels={"labels": "y"}),
)

adapter = SpdlTensorAdapter(schema)
loader = SPDLLoader(
    spdl_tensor_batches,
    adapter,
    shard_id=0,
    cost_probe=DataCostProbe(),
)

for batch in loader:
    result = runtime_step.run(batch)
```

## Relationship to PlainLoader

`PlainLoader` reads from `TensorDictReader` and `SPDLLoader` reads from an
iterable tensor source, but both expose the same runtime type:

```text
Iterator[KVBatch]
```

This keeps downstream runtime components independent from the physical data
source. `BudgetedBatcher`, OOM recovery, device transfer, and AutoGovernor should
operate on `KVBatch` streams rather than on SPDL-specific objects.

## Out of Scope

- SPDL pipeline construction.
- SPDL worker, queue-depth, or async prefetch tuning.
- Pinned memory.
- Device transfer.
- Dynamic batch-size selection.
- `BudgetedBatcher`.
- OOM recovery and batch split retry.
- AutoGovernor.
- ShardController and distributed resume.
- TorchData wrapper integration.
- DataFrame, Arrow, WebDataset, or Hugging Face source plugins.

## Test Commands

```bash
python -m pytest enn_torch_dev/debug/runtime/test_spdl_loader.py -q
python -m pytest enn_torch_dev/debug/runtime/test_plain_loader.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug/data/test_spdl_adapter.py -q
python -m pytest enn_torch_dev/debug/data -q
python -m pytest enn_torch_dev/debug -q
```

## Next Step

The next runtime-facing slice should add a minimal `BudgetedBatcher`. It should
consume `KVBatch` streams from either `PlainLoader` or `SPDLLoader`, combine
`BatchCost` / `DataCost` / `ModelCost` observations with resource budgets, and
choose conservative batch sizes without hardcoding GPU profiles.
