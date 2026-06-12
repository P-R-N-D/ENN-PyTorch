# Development Tensor Staging

This document describes the second data/runtime rewrite slice for
`enn_torch_dev`.

## Goal

The runtime hot path must remain tensor-backed. This slice adds disk-backed
staging for already tensorized data:

```text
Mapping[str, Tensor] / TensorDict
  -> TensorDictStagingWriter
  -> field-level MemoryMappedTensor files
  -> DatasetManifest JSON
  -> TensorDictReader
  -> TensorDict / KVBatch
```

Source-native objects such as dataframe rows, Arrow tables, tar-shard sample
dictionaries, and Hugging Face rows remain out of scope. They should be handled
by ingestion plugins that produce tensors before this layer.

## Storage Layout

The writer creates this directory layout:

```text
root/
  manifest.json
  tensors/
    features.mmt
    labels.mmt
  index/
    row_id.mmt
```

Each schema field is stored as a separate MemoryMappedTensor file. Runtime row
identity is stored separately as `index/row_id.mmt`.

## Schema Shape vs Storage Shape

`FieldSpec.shape` is the schema contract. It may contain `None`, for example
`(None, 3)`.

`TensorFieldManifest.storage_shape` is the concrete shape used to reopen the
MemoryMappedTensor, for example `(1000, 3)`.

The reader uses `storage_shape` to open files and `DatasetManifest.to_schema()`
to restore the schema contract.

## Row Identity

`row_id` is not a model input field. It is runtime identity used for retry,
batch split, resume, and later sharding checks.

If the source contains `row_id`, the writer stores it. Otherwise it generates:

```python
torch.arange(num_rows, dtype=torch.long)
```

## Example

```python
from pathlib import Path

import torch

from enn_torch_dev.data import (
    DataSchema,
    FieldSpec,
    KeyMapping,
    StagingSpec,
    TensorDictReader,
    TensorDictStagingWriter,
)

schema = DataSchema(
    schema_id="demo.schema",
    fields=(
        FieldSpec("features", torch.float32, shape=(None, 3)),
        FieldSpec("labels", torch.float32, shape=(None, 1), required=False),
    ),
    key_mapping=KeyMapping(inputs={"features": "x"}, labels={"labels": "y"}),
)

writer = TensorDictStagingWriter(
    StagingSpec(root=Path("stage/demo"), schema=schema, overwrite=True)
)
writer.write({"features": features, "labels": labels})

reader = TensorDictReader("stage/demo")
batch = reader.get_kvbatch(torch.arange(32))
store = batch.to_store(reader.schema)
```

## Test Commands

```bash
python -m pytest enn_torch_dev/debug/data/test_staging.py -q
python -m pytest enn_torch_dev/debug/data/test_readers.py -q
python -m pytest enn_torch_dev/debug/data -q
python -m pytest enn_torch_dev/debug -q
```

## Out of Scope

- SPDL engine integration.
- Async prefetch.
- Pinned memory.
- Device transfer.
- RuntimeStep, optimizer, and loss handling.
- ResourceMonitor.
- AutoGovernor and calibration cache.
- OOM recovery.
- AMP or precision policy.
- ShardController and distributed resume.
- Polars, PyArrow, WebDataset, or Hugging Face ingestion plugins.

## Next Step

The next slice should add a plain loader and `RuntimeStep` fault classification
on top of `TensorDictReader`, before introducing SPDL.
