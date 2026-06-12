from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from enn_torch_dev.data import (
    DataSchema,
    FieldSpec,
    KeyMapping,
    StagingSpec,
    TensorDictStagingWriter,
)


def _schema() -> DataSchema:
    return DataSchema(
        schema_id="demo.staging",
        fields=(
            FieldSpec("features", torch.float32, shape=(None, 3)),
            FieldSpec("labels", torch.float32, shape=(None, 1), required=False),
            FieldSpec("mask", torch.bool, shape=(None,), required=False),
        ),
        key_mapping=KeyMapping(
            inputs={"features": "x"},
            labels={"labels": "y"},
            metadata={"mask": "mask"},
        ),
    )


def _payload(num_rows: int = 5) -> dict[str, torch.Tensor]:
    return {
        "features": torch.arange(num_rows * 3, dtype=torch.float32).reshape(num_rows, 3),
        "labels": torch.arange(num_rows, dtype=torch.float32).reshape(num_rows, 1),
        "mask": torch.arange(num_rows) % 2 == 0,
    }


def test_stages_mapping_to_field_memmaps_and_manifest(tmp_path) -> None:
    root = tmp_path / "stage"

    result = TensorDictStagingWriter(
        StagingSpec(root=root, schema=_schema()),
    ).write(_payload())

    assert (root / "manifest.json").exists()
    assert (root / "tensors" / "features.mmt").exists()
    assert (root / "tensors" / "labels.mmt").exists()
    assert (root / "tensors" / "mask.mmt").exists()
    assert (root / "index" / "row_id.mmt").exists()
    assert result.manifest.schema_id == "demo.staging"
    assert result.manifest.num_rows == 5
    field = result.manifest.to_schema().field("features")
    assert field.dtype is torch.float32
    features_manifest = result.manifest.fields[0]
    assert features_manifest.storage_key == "tensors/features.mmt"
    assert features_manifest.storage_shape == (5, 3)
    assert torch.equal(result.row_ids, torch.arange(5))


def test_stages_tensordict_source(tmp_path) -> None:
    root = tmp_path / "stage"
    payload = _payload()
    td = TensorDict(payload, batch_size=(5,))

    result = TensorDictStagingWriter(
        StagingSpec(root=root, schema=_schema()),
    ).write(td)

    assert result.manifest.num_rows == 5
    assert (root / "manifest.json").exists()


def test_staging_rejects_missing_required_field(tmp_path) -> None:
    payload = {"labels": torch.zeros(5, 1)}

    with pytest.raises(KeyError, match="features"):
        TensorDictStagingWriter(
            StagingSpec(root=tmp_path / "stage", schema=_schema()),
        ).write(payload)


def test_staging_preserves_absent_optional_field_in_manifest(tmp_path) -> None:
    root = tmp_path / "stage"
    payload = {"features": _payload()["features"]}

    result = TensorDictStagingWriter(
        StagingSpec(root=root, schema=_schema()),
    ).write(payload)

    assert [field.name for field in result.manifest.fields] == ["features", "labels", "mask"]
    labels_manifest = result.manifest.fields[1]
    assert labels_manifest.required is False
    assert labels_manifest.storage_key is None
    assert labels_manifest.storage_shape is None
    assert result.manifest.key_mapping == _schema().key_mapping


def test_staging_rejects_dtype_mismatch(tmp_path) -> None:
    payload = _payload()
    payload["features"] = payload["features"].to(torch.float64)

    with pytest.raises(TypeError, match="dtype"):
        TensorDictStagingWriter(
            StagingSpec(root=tmp_path / "stage", schema=_schema()),
        ).write(payload)


def test_staging_rejects_shape_mismatch(tmp_path) -> None:
    payload = _payload()
    payload["features"] = torch.zeros(5, 4)

    with pytest.raises(ValueError, match="shape"):
        TensorDictStagingWriter(
            StagingSpec(root=tmp_path / "stage", schema=_schema()),
        ).write(payload)


def test_staging_rejects_mismatched_row_counts(tmp_path) -> None:
    payload = _payload()
    payload["labels"] = torch.zeros(4, 1)

    with pytest.raises(ValueError, match="same row count"):
        TensorDictStagingWriter(
            StagingSpec(root=tmp_path / "stage", schema=_schema()),
        ).write(payload)


def test_staging_rejects_existing_root_without_overwrite(tmp_path) -> None:
    root = tmp_path / "stage"
    root.mkdir()

    with pytest.raises(FileExistsError):
        TensorDictStagingWriter(StagingSpec(root=root, schema=_schema())).write(_payload())


def test_staging_overwrite_rewrites_existing_root(tmp_path) -> None:
    root = tmp_path / "stage"
    root.mkdir()
    (root / "stale.txt").write_text("stale", encoding="utf-8")

    TensorDictStagingWriter(
        StagingSpec(root=root, schema=_schema(), overwrite=True),
    ).write(_payload())

    assert not (root / "stale.txt").exists()
    assert (root / "manifest.json").exists()


def test_staging_uses_provided_row_ids(tmp_path) -> None:
    payload = _payload()
    payload["row_id"] = torch.tensor([10, 11, 12, 13, 14])

    result = TensorDictStagingWriter(
        StagingSpec(root=tmp_path / "stage", schema=_schema()),
    ).write(payload)

    assert torch.equal(result.row_ids, torch.tensor([10, 11, 12, 13, 14]))


def test_staging_generates_row_ids_when_absent(tmp_path) -> None:
    result = TensorDictStagingWriter(
        StagingSpec(root=tmp_path / "stage", schema=_schema()),
    ).write(_payload(3))

    assert torch.equal(result.row_ids, torch.arange(3))


def test_staging_rejects_row_id_declared_as_schema_field(tmp_path) -> None:
    schema = DataSchema(
        schema_id="bad.rowid",
        fields=(
            FieldSpec("row_id", torch.long, shape=(None,)),
            FieldSpec("features", torch.float32, shape=(None, 3)),
        ),
        key_mapping=KeyMapping(inputs={"features": "x"}),
    )

    with pytest.raises(ValueError, match="reserved"):
        TensorDictStagingWriter(
            StagingSpec(root=tmp_path / "stage", schema=schema),
        ).write({"features": torch.zeros(2, 3), "row_id": torch.arange(2)})
