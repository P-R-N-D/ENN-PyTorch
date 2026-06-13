from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from enn_torch_dev.data import (
    BatchCost,
    DataSchema,
    FieldSpec,
    KeyMapping,
    SpdlTensorAdapter,
)


def _schema() -> DataSchema:
    return DataSchema(
        schema_id="demo.spdl",
        fields=(
            FieldSpec("features", torch.float32, shape=(None, 3)),
            FieldSpec("labels", torch.float32, shape=(None, 1), required=False),
            FieldSpec("weights", torch.float32, shape=(None,), required=False),
        ),
        key_mapping=KeyMapping(
            inputs={"features": "x"},
            labels={"labels": "y"},
            metadata={"weights": "w"},
        ),
    )


def _payload(num_rows: int = 4) -> dict[str, torch.Tensor]:
    return {
        "features": torch.arange(num_rows * 3, dtype=torch.float32).reshape(num_rows, 3),
        "labels": torch.arange(num_rows, dtype=torch.float32).reshape(num_rows, 1),
        "weights": torch.ones(num_rows, dtype=torch.float32),
    }


def test_spdl_adapter_converts_mapping_to_kvbatch_and_store() -> None:
    payload = _payload()
    adapter = SpdlTensorAdapter(_schema())

    batch = adapter.to_kvbatch(payload)
    store = batch.to_store(_schema())

    assert batch.schema_id == "demo.spdl"
    assert batch.batch_size == 4
    assert torch.equal(batch.row_ids, torch.arange(4))
    assert torch.equal(store.get("x"), payload["features"])
    assert torch.equal(store.get("y"), payload["labels"])
    assert torch.equal(store.get("w"), payload["weights"])


def test_spdl_adapter_converts_tensordict_and_strips_identity_keys() -> None:
    payload = _payload()
    payload["row_id"] = torch.tensor([10, 11, 12, 13])
    td = TensorDict(payload, batch_size=(4,))
    adapter = SpdlTensorAdapter(_schema())

    batch = adapter.to_kvbatch(td, shard_id=2)

    assert batch.shard_id == 2
    assert torch.equal(batch.row_ids, torch.tensor([10, 11, 12, 13]))
    assert "row_id" not in batch.td.keys()
    assert batch.td.batch_size == torch.Size([4])


def test_spdl_adapter_preserves_source_and_sample_ids() -> None:
    payload = _payload(3)
    payload["source_id"] = torch.tensor([1, 1, 2])
    payload["sample_id"] = torch.tensor([100, 101, 102])
    adapter = SpdlTensorAdapter(_schema())

    batch = adapter.to_kvbatch(payload)
    store = batch.to_store(_schema())

    assert torch.equal(batch.source_ids, torch.tensor([1, 1, 2]))
    assert torch.equal(batch.sample_ids, torch.tensor([100, 101, 102]))
    assert torch.equal(store.get("source_id"), torch.tensor([1, 1, 2]))
    assert torch.equal(store.get("sample_id"), torch.tensor([100, 101, 102]))


def test_spdl_adapter_returns_tensordict_without_identity_keys() -> None:
    payload = _payload(2)
    payload["row_id"] = torch.tensor([5, 6])
    adapter = SpdlTensorAdapter(_schema())

    td = adapter.to_tensordict(payload)

    assert td.batch_size == torch.Size([2])
    assert set(td.keys()) == {"features", "labels", "weights"}


def test_spdl_adapter_accepts_empty_batches() -> None:
    payload = _payload(0)
    adapter = SpdlTensorAdapter(_schema())

    batch = adapter.to_kvbatch(payload)

    assert batch.batch_size == 0
    assert batch.row_ids.shape == torch.Size([0])
    assert batch.td.batch_size == torch.Size([0])


def test_spdl_adapter_preserves_cost_hint() -> None:
    payload = _payload(2)
    cost = BatchCost(host_bytes=24, num_items=2)
    adapter = SpdlTensorAdapter(_schema())

    batch = adapter.to_kvbatch(payload, cost_hint=cost)

    assert batch.cost_hint == cost


def test_spdl_adapter_rejects_missing_required_field() -> None:
    adapter = SpdlTensorAdapter(_schema())

    with pytest.raises(KeyError, match="features"):
        adapter.to_kvbatch({"labels": torch.zeros(4, 1)})


def test_spdl_adapter_rejects_non_tensor_field() -> None:
    payload = _payload()
    payload["features"] = "not-a-tensor"  # type: ignore[assignment]
    adapter = SpdlTensorAdapter(_schema())

    with pytest.raises(TypeError, match="features"):
        adapter.to_kvbatch(payload)


def test_spdl_adapter_rejects_dtype_mismatch() -> None:
    payload = _payload()
    payload["features"] = payload["features"].to(torch.float64)
    adapter = SpdlTensorAdapter(_schema())

    with pytest.raises(TypeError, match="dtype"):
        adapter.to_kvbatch(payload)


def test_spdl_adapter_rejects_batch_size_mismatch() -> None:
    payload = _payload()
    payload["labels"] = torch.zeros(3, 1)
    adapter = SpdlTensorAdapter(_schema())

    with pytest.raises(ValueError, match="same batch size"):
        adapter.to_kvbatch(payload)


def test_spdl_adapter_rejects_row_id_length_mismatch() -> None:
    payload = _payload(4)
    payload["row_id"] = torch.tensor([1, 2, 3])
    adapter = SpdlTensorAdapter(_schema())

    with pytest.raises(ValueError, match="row_id"):
        adapter.to_kvbatch(payload)


def test_spdl_adapter_rejects_non_integer_row_id() -> None:
    payload = _payload(2)
    payload["row_id"] = torch.tensor([0.0, 1.0])
    adapter = SpdlTensorAdapter(_schema())

    with pytest.raises(TypeError, match="integer"):
        adapter.to_kvbatch(payload)


def test_spdl_adapter_rejects_reserved_schema_field() -> None:
    schema = DataSchema(
        schema_id="bad.spdl",
        fields=(
            FieldSpec("row_id", torch.long, shape=(None,)),
            FieldSpec("features", torch.float32, shape=(None, 3)),
        ),
        key_mapping=KeyMapping(inputs={"features": "x"}),
    )

    with pytest.raises(ValueError, match="reserved"):
        SpdlTensorAdapter(schema)


def test_spdl_adapter_rejects_duplicate_identity_keys() -> None:
    with pytest.raises(ValueError, match="distinct"):
        SpdlTensorAdapter(_schema(), row_id_key="id", source_id_key="id")


def test_spdl_adapter_rejects_non_zero_batch_axis() -> None:
    schema = DataSchema(
        schema_id="bad.axis",
        fields=(FieldSpec("features", torch.float32, shape=(3, None), batch_axis=1),),
        key_mapping=KeyMapping(inputs={"features": "x"}),
    )

    with pytest.raises(NotImplementedError, match="batch_axis=0"):
        SpdlTensorAdapter(schema)
