from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from enn_torch_dev.data import DataSchema, FieldSpec, KeyMapping, SpdlTensorAdapter
from enn_torch_dev.runtime import DataCostProbe, SPDLLoader


def _schema() -> DataSchema:
    return DataSchema(
        schema_id="runtime.spdl.loader",
        fields=(
            FieldSpec("features", torch.float32, shape=(None, 3)),
            FieldSpec("labels", torch.float32, shape=(None, 1), required=False),
        ),
        key_mapping=KeyMapping(inputs={"features": "x"}, labels={"labels": "y"}),
    )


def _payload(start: int, num_rows: int) -> dict[str, torch.Tensor]:
    return {
        "features": torch.arange(
            start * 3,
            (start + num_rows) * 3,
            dtype=torch.float32,
        ).reshape(num_rows, 3),
        "labels": torch.arange(start, start + num_rows, dtype=torch.float32).reshape(
            num_rows,
            1,
        ),
        "row_id": torch.arange(100 + start, 100 + start + num_rows),
    }


def _adapter() -> SpdlTensorAdapter:
    return SpdlTensorAdapter(_schema())


def test_spdl_loader_converts_mapping_batches_to_kvbatch_stream() -> None:
    source = [_payload(0, 2), _payload(2, 3)]

    batches = list(SPDLLoader(source, _adapter()))

    assert [batch.batch_size for batch in batches] == [2, 3]
    assert torch.equal(batches[0].row_ids, torch.tensor([100, 101]))
    assert torch.equal(batches[1].row_ids, torch.tensor([102, 103, 104]))
    assert torch.equal(batches[0].to_store(_schema()).get("x"), source[0]["features"])


def test_spdl_loader_converts_tensordict_batches_to_kvbatch_stream() -> None:
    payload = _payload(4, 2)
    td = TensorDict(payload, batch_size=(2,))

    batches = list(SPDLLoader([td], _adapter()))

    assert len(batches) == 1
    assert batches[0].td.batch_size == torch.Size([2])
    assert torch.equal(batches[0].row_ids, torch.tensor([104, 105]))


def test_spdl_loader_preserves_optional_identity_tensors() -> None:
    payload = _payload(0, 3)
    payload["source_id"] = torch.tensor([1, 1, 2])
    payload["sample_id"] = torch.tensor([10, 11, 12])

    batch = next(iter(SPDLLoader([payload], _adapter())))

    assert torch.equal(batch.source_ids, torch.tensor([1, 1, 2]))
    assert torch.equal(batch.sample_ids, torch.tensor([10, 11, 12]))


def test_spdl_loader_sets_shard_id() -> None:
    batches = list(SPDLLoader([_payload(0, 2), _payload(2, 2)], _adapter(), shard_id=5))

    assert [batch.shard_id for batch in batches] == [5, 5]


def test_spdl_loader_propagates_adapter_validation_errors() -> None:
    source = [{"labels": torch.zeros(2, 1)}]

    with pytest.raises(KeyError, match="features"):
        list(SPDLLoader(source, _adapter()))


def test_spdl_loader_accepts_empty_iterables() -> None:
    assert list(SPDLLoader([], _adapter())) == []


def test_spdl_loader_adds_cost_hint_when_cost_probe_is_provided() -> None:
    payload = _payload(0, 2)
    batch = next(iter(SPDLLoader([payload], _adapter(), cost_probe=DataCostProbe())))

    assert batch.cost_hint is not None
    assert batch.cost_hint.host_bytes == 32
    assert batch.cost_hint.device_bytes == 0
    assert batch.cost_hint.num_items == 2


def test_spdl_loader_rejects_non_adapter() -> None:
    with pytest.raises(TypeError, match="SpdlTensorAdapter"):
        SPDLLoader([], object())  # type: ignore[arg-type]


def test_spdl_loader_rejects_single_batch_mapping_as_source() -> None:
    with pytest.raises(TypeError, match="iterable of tensor batches"):
        SPDLLoader(_payload(0, 2), _adapter())  # type: ignore[arg-type]


def test_spdl_loader_rejects_invalid_cost_probe() -> None:
    with pytest.raises(TypeError, match="DataCostProbe"):
        SPDLLoader([], _adapter(), cost_probe=object())  # type: ignore[arg-type]
