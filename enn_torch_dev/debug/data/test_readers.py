from __future__ import annotations

import pytest
import torch
from torch import nn

from enn_torch_dev.data import (
    DataSchema,
    FieldSpec,
    KeyMapping,
    StagingSpec,
    TensorDictReader,
    TensorDictStagingWriter,
)
from enn_torch_dev.executor import GraphExecutor, KeyRef, NodeSpec


class _Double(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0


def _schema() -> DataSchema:
    return DataSchema(
        schema_id="demo.reader",
        fields=(
            FieldSpec("features", torch.float32, shape=(None, 3)),
            FieldSpec("labels", torch.float32, shape=(None, 1), required=False),
        ),
        key_mapping=KeyMapping(
            inputs={"features": "x"},
            labels={"labels": "y"},
        ),
    )


def _payload(num_rows: int = 7) -> dict[str, torch.Tensor]:
    return {
        "features": torch.arange(num_rows * 3, dtype=torch.float32).reshape(num_rows, 3),
        "labels": torch.arange(num_rows, dtype=torch.float32).reshape(num_rows, 1),
        "row_id": torch.arange(100, 100 + num_rows),
    }


def _stage(tmp_path, num_rows: int = 7) -> tuple[TensorDictReader, dict[str, torch.Tensor]]:
    payload = _payload(num_rows)
    root = tmp_path / "stage"
    TensorDictStagingWriter(
        StagingSpec(root=root, schema=_schema()),
    ).write(payload)
    return TensorDictReader(root), payload


def test_reader_loads_manifest_schema_and_num_rows(tmp_path) -> None:
    reader, _payload = _stage(tmp_path)

    assert reader.manifest.schema_id == "demo.reader"
    assert reader.schema.schema_id == "demo.reader"
    assert reader.num_rows == 7


def test_reader_get_rows_with_tensor_indices(tmp_path) -> None:
    reader, payload = _stage(tmp_path)
    indices = torch.tensor([0, 2, 4])

    td = reader.get_rows(indices)

    assert td.batch_size == torch.Size([3])
    assert torch.equal(td["features"], payload["features"][indices])
    assert torch.equal(td["labels"], payload["labels"][indices])


def test_reader_get_rows_with_slice(tmp_path) -> None:
    reader, payload = _stage(tmp_path)

    td = reader.get_rows(slice(1, 4))

    assert td.batch_size == torch.Size([3])
    assert torch.equal(td["features"], payload["features"][1:4])
    assert torch.equal(td["labels"], payload["labels"][1:4])


def test_reader_get_row_ids(tmp_path) -> None:
    reader, payload = _stage(tmp_path)
    indices = torch.tensor([0, 3, 6])

    row_ids = reader.get_row_ids(indices)

    assert torch.equal(row_ids, payload["row_id"][indices])


def test_reader_get_kvbatch_and_to_store(tmp_path) -> None:
    reader, payload = _stage(tmp_path)

    batch = reader.get_kvbatch(torch.tensor([0, 1]))
    store = batch.to_store(reader.schema)

    assert batch.schema_id == "demo.reader"
    assert torch.equal(batch.row_ids, payload["row_id"][:2])
    assert torch.equal(store.get("x"), payload["features"][:2])
    assert torch.equal(store.get("y"), payload["labels"][:2])


def test_reader_graph_executor_smoke_path(tmp_path) -> None:
    reader, _payload = _stage(tmp_path)
    graph = GraphExecutor(
        [
            (
                NodeSpec(
                    name="double_node",
                    input_args=[KeyRef("x")],
                    output_key="pred",
                ),
                _Double(),
            )
        ]
    )

    batch = reader.get_kvbatch(torch.tensor([0, 1]))
    store = batch.to_store(reader.schema)
    graph.run(store)

    assert torch.equal(store.get("pred"), store.get("x") * 2.0)


def test_reader_iter_batches_returns_rows_in_order(tmp_path) -> None:
    reader, payload = _stage(tmp_path, num_rows=7)

    batches = list(reader.iter_batches(batch_size=3))

    assert [batch.batch_size for batch in batches] == [3, 3, 1]
    assert torch.equal(batches[0].row_ids, payload["row_id"][0:3])
    assert torch.equal(batches[1].row_ids, payload["row_id"][3:6])
    assert torch.equal(batches[2].row_ids, payload["row_id"][6:7])


def test_reader_iter_batches_drop_last(tmp_path) -> None:
    reader, payload = _stage(tmp_path, num_rows=7)

    batches = list(reader.iter_batches(batch_size=3, drop_last=True))

    assert [batch.batch_size for batch in batches] == [3, 3]
    assert torch.equal(batches[0].row_ids, payload["row_id"][0:3])
    assert torch.equal(batches[1].row_ids, payload["row_id"][3:6])


def test_reader_rejects_out_of_range_indices(tmp_path) -> None:
    reader, _payload = _stage(tmp_path)

    with pytest.raises(IndexError):
        reader.get_rows(torch.tensor([0, 7]))


def test_reader_rejects_invalid_batch_size(tmp_path) -> None:
    reader, _payload = _stage(tmp_path)

    with pytest.raises(ValueError, match="positive"):
        list(reader.iter_batches(batch_size=0))
