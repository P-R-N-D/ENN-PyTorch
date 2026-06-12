from __future__ import annotations

import pytest
import torch

from enn_torch_dev.data import (
    DataSchema,
    FieldSpec,
    KeyMapping,
    StagingSpec,
    TensorDictReader,
    TensorDictStagingWriter,
)
from enn_torch_dev.runtime import PlainLoader


def _schema() -> DataSchema:
    return DataSchema(
        schema_id="runtime.loader",
        fields=(
            FieldSpec("features", torch.float32, shape=(None, 3)),
            FieldSpec("labels", torch.float32, shape=(None, 1)),
        ),
        key_mapping=KeyMapping(
            inputs={"features": "x"},
            labels={"labels": "y"},
        ),
    )


def _reader(tmp_path, *, num_rows: int = 7) -> TensorDictReader:
    root = tmp_path / "stage"
    TensorDictStagingWriter(
        StagingSpec(root=root, schema=_schema()),
    ).write(
        {
            "features": torch.arange(num_rows * 3, dtype=torch.float32).reshape(num_rows, 3),
            "labels": torch.arange(num_rows, dtype=torch.float32).reshape(num_rows, 1),
            "row_id": torch.arange(100, 100 + num_rows),
        }
    )
    return TensorDictReader(root)


def test_plain_loader_returns_kvbatches_in_order(tmp_path) -> None:
    reader = _reader(tmp_path, num_rows=7)

    batches = list(PlainLoader(reader, batch_size=3))

    assert [batch.batch_size for batch in batches] == [3, 3, 1]
    assert torch.equal(batches[0].row_ids, torch.tensor([100, 101, 102]))
    assert torch.equal(batches[1].row_ids, torch.tensor([103, 104, 105]))
    assert torch.equal(batches[2].row_ids, torch.tensor([106]))


def test_plain_loader_drop_last(tmp_path) -> None:
    reader = _reader(tmp_path, num_rows=7)

    batches = list(PlainLoader(reader, batch_size=3, drop_last=True))

    assert [batch.batch_size for batch in batches] == [3, 3]
    assert torch.equal(batches[-1].row_ids, torch.tensor([103, 104, 105]))


def test_plain_loader_sets_shard_id(tmp_path) -> None:
    reader = _reader(tmp_path, num_rows=4)

    batches = list(PlainLoader(reader, batch_size=2, shard_id=7))

    assert [batch.shard_id for batch in batches] == [7, 7]


def test_plain_loader_rejects_invalid_batch_size(tmp_path) -> None:
    reader = _reader(tmp_path)

    with pytest.raises(ValueError, match="positive"):
        PlainLoader(reader, batch_size=0)


def test_plain_loader_rejects_non_reader() -> None:
    with pytest.raises(TypeError, match="TensorDictReader"):
        PlainLoader(object(), batch_size=1)  # type: ignore[arg-type]
