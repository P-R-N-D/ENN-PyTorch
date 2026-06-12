from __future__ import annotations

import pytest
import torch

from enn_torch_dev.data import (
    DataSchema,
    DatasetManifest,
    FieldSpec,
    KeyMapping,
    TensorFieldManifest,
)


def _schema() -> DataSchema:
    return DataSchema(
        schema_id="demo.schema",
        fields=(
            FieldSpec(
                name="features",
                dtype=torch.float32,
                shape=(2, 3),
                role="feature",
            ),
            FieldSpec(
                name="labels",
                dtype=torch.float32,
                shape=(2, 1),
                role="label",
                required=False,
            ),
        ),
        key_mapping=KeyMapping(
            inputs={"features": "x"},
            labels={"labels": "y"},
        ),
    )


def test_field_spec_validates_tensor_contract() -> None:
    field = FieldSpec("features", torch.float32, shape=(2, 3))

    field.validate_tensor(torch.zeros(2, 3, dtype=torch.float32))

    with pytest.raises(TypeError, match="dtype"):
        field.validate_tensor(torch.zeros(2, 3, dtype=torch.float64))

    with pytest.raises(ValueError, match="shape"):
        field.validate_tensor(torch.zeros(2, 4, dtype=torch.float32))


def test_field_spec_normalizes_dtype_name_strings() -> None:
    field = FieldSpec("features", "torch.float32", shape=(2, 3))

    assert field.dtype is torch.float32
    assert field.dtype_name() == "float32"
    field.validate_tensor(torch.zeros(2, 3, dtype=torch.float32))

    with pytest.raises(TypeError, match="dtype"):
        field.validate_tensor(torch.zeros(2, 3, dtype=torch.float64))


def test_data_schema_rejects_duplicate_fields() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        DataSchema(
            schema_id="bad.schema",
            fields=(
                FieldSpec("features", torch.float32),
                FieldSpec("features", torch.float32),
            ),
            key_mapping=KeyMapping(inputs={"features": "x"}),
        )


def test_key_mapping_rejects_duplicate_kvstore_targets() -> None:
    mapping = KeyMapping(
        inputs={"features": "x"},
        labels={"labels": "x"},
    )

    with pytest.raises(ValueError, match="duplicate KVStore key"):
        mapping.all_items()


def test_data_schema_rejects_mapping_sources_not_declared_as_fields() -> None:
    with pytest.raises(ValueError, match="declared fields"):
        DataSchema(
            schema_id="bad.mapping",
            fields=(FieldSpec("features", torch.float32),),
            key_mapping=KeyMapping(inputs={"typo": "x"}),
        )


def test_dataset_manifest_roundtrips_schema_contract() -> None:
    schema = _schema()

    manifest = DatasetManifest.from_schema(schema, num_rows=128)
    restored = manifest.to_schema()

    assert manifest.schema_id == "demo.schema"
    assert manifest.num_rows == 128
    assert manifest.storage_format == "tensordict"
    assert [field.name for field in manifest.fields] == ["features", "labels"]
    assert isinstance(manifest.fields[0], TensorFieldManifest)
    assert restored.schema_id == schema.schema_id
    assert restored.key_mapping.inputs == schema.key_mapping.inputs
    assert restored.field("features").shape == (2, 3)
    assert restored.field("features").dtype is torch.float32
    assert restored.field("labels").required is False

    with pytest.raises(TypeError, match="dtype"):
        restored.field("features").validate_tensor(
            torch.zeros(2, 3, dtype=torch.float64)
        )


def test_manifest_to_dict_is_json_ready() -> None:
    manifest = DatasetManifest.from_schema(_schema(), num_rows=2)

    payload = manifest.to_dict()

    assert payload["schema_version"] == 1
    assert payload["schema_id"] == "demo.schema"
    assert payload["fields"][0]["name"] == "features"
    assert payload["key_mapping"]["inputs"] == {"features": "x"}
