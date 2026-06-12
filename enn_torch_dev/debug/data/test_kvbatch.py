from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from enn_torch_dev.data import BatchCost, DataSchema, FieldSpec, KVBatch, KeyMapping
from enn_torch_dev.executor import GraphBuilder


class _Double(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0


def _schema() -> DataSchema:
    return DataSchema(
        schema_id="demo.batch",
        fields=(
            FieldSpec("features", torch.float32, shape=(2, 3), role="feature"),
            FieldSpec("labels", torch.float32, shape=(2, 1), role="label", required=False),
            FieldSpec("mask", torch.bool, shape=(2,), role="mask", required=False),
        ),
        key_mapping=KeyMapping(
            inputs={"features": "x"},
            labels={"labels": "y"},
            metadata={"mask": "mask"},
        ),
    )


def _tensordict(*, include_optional: bool = True) -> TensorDict:
    payload: dict[str, torch.Tensor] = {
        "features": torch.arange(6, dtype=torch.float32).reshape(2, 3),
    }
    if include_optional:
        payload["labels"] = torch.ones(2, 1, dtype=torch.float32)
        payload["mask"] = torch.tensor([True, False])
    return TensorDict(payload, batch_size=(2,))


def test_kvbatch_to_store_exposes_tensor_keys_for_graph_executor() -> None:
    schema = _schema()
    td = _tensordict()
    batch = KVBatch(
        td=td,
        row_ids=torch.tensor([10, 11]),
        source_ids=torch.tensor([1, 1]),
        sample_ids=torch.tensor([100, 101]),
        schema_id=schema.schema_id,
    )

    store = batch.to_store(schema)
    graph = (
        GraphBuilder()
        .add(
            name="double",
            module=_Double(),
            input_args=["x"],
            output_key="pred",
        )
        .build()
    )

    graph.run(store)

    assert torch.equal(store.get("x"), td["features"])
    assert torch.equal(store.get("y"), td["labels"])
    assert torch.equal(store.get("mask"), td["mask"])
    assert torch.equal(store.get("row_id"), torch.tensor([10, 11]))
    assert torch.equal(store.get("source_id"), torch.tensor([1, 1]))
    assert torch.equal(store.get("sample_id"), torch.tensor([100, 101]))
    assert torch.equal(store.get("pred"), td["features"] * 2.0)


def test_kvbatch_allows_missing_optional_schema_fields() -> None:
    schema = _schema()
    batch = KVBatch(
        td=_tensordict(include_optional=False),
        row_ids=torch.tensor([0, 1]),
        schema_id=schema.schema_id,
    )

    store = batch.to_store(schema)

    assert store.has("x")
    assert not store.has("y")
    assert not store.has("mask")


def test_kvbatch_rejects_missing_required_field() -> None:
    schema = _schema()
    td = TensorDict({"labels": torch.ones(2, 1)}, batch_size=(2,))
    batch = KVBatch(td=td, row_ids=torch.tensor([0, 1]), schema_id=schema.schema_id)

    with pytest.raises(KeyError, match="Missing required"):
        batch.to_store(schema)


def test_kvbatch_rejects_row_id_batch_mismatch() -> None:
    with pytest.raises(ValueError, match="row_ids"):
        KVBatch(td=_tensordict(), row_ids=torch.tensor([0]))


def test_kvbatch_batch_cost_counts_flat_tensor_bytes() -> None:
    td = _tensordict()

    cost = BatchCost.from_tensordict(td)

    expected = sum(value.numel() * value.element_size() for value in td.values())
    assert cost.host_bytes == expected
    assert cost.num_items == 2


def test_kvbatch_with_key_mapping_requires_mapped_keys() -> None:
    mapping = KeyMapping(inputs={"missing": "x"})
    batch = KVBatch(td=_tensordict(), row_ids=torch.tensor([0, 1]))

    with pytest.raises(KeyError, match="missing"):
        batch.to_store(mapping)
