from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import DataSchema, FieldSpec, KeyMapping


def _validate_key(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    if not value:
        raise ValueError(f"{label} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{label} must not have leading or trailing whitespace.")
    return value


def _normalize_shape(value: object) -> tuple[int | None, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("TensorFieldManifest.shape must be a sequence.")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("TensorFieldManifest.shape must be a sequence.") from exc
    for item in items:
        if item is not None and (not isinstance(item, int) or isinstance(item, bool)):
            raise TypeError("TensorFieldManifest.shape entries must be integers or None.")
        if isinstance(item, int) and item < 0:
            raise ValueError("TensorFieldManifest.shape integer entries must be non-negative.")
    return items


def _normalize_storage_shape(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("TensorFieldManifest.storage_shape must be a sequence.")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("TensorFieldManifest.storage_shape must be a sequence.") from exc
    for item in items:
        if not isinstance(item, int) or isinstance(item, bool):
            raise TypeError("TensorFieldManifest.storage_shape entries must be integers.")
        if item < 0:
            raise ValueError("TensorFieldManifest.storage_shape entries must be non-negative.")
    return items


def _normalize_field_manifests(
    fields: Sequence["TensorFieldManifest"],
) -> tuple["TensorFieldManifest", ...]:
    if isinstance(fields, (str, bytes, bytearray)):
        raise TypeError("DatasetManifest.fields must be a sequence of TensorFieldManifest.")
    try:
        values = tuple(fields)
    except TypeError as exc:
        raise TypeError("DatasetManifest.fields must be a sequence of TensorFieldManifest.") from exc
    if not values:
        raise ValueError("DatasetManifest.fields must not be empty.")
    names: set[str] = set()
    for field_manifest in values:
        if not isinstance(field_manifest, TensorFieldManifest):
            raise TypeError(
                "DatasetManifest.fields must contain TensorFieldManifest values, "
                f"got {type(field_manifest)!r}."
            )
        if field_manifest.name in names:
            raise ValueError(f"DatasetManifest has duplicate field: {field_manifest.name!r}")
        names.add(field_manifest.name)
    return values


@dataclass(frozen=True, slots=True)
class TensorFieldManifest:
    name: str
    dtype: str
    shape: Sequence[int | None] | None
    batch_axis: int = 0
    storage_key: str | None = None
    storage_shape: Sequence[int] | None = None
    required: bool = True
    role: str = "feature"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_key(self.name, "TensorFieldManifest.name"))
        object.__setattr__(self, "dtype", _validate_key(self.dtype, "TensorFieldManifest.dtype"))
        object.__setattr__(self, "shape", _normalize_shape(self.shape))
        if not isinstance(self.batch_axis, int) or isinstance(self.batch_axis, bool):
            raise TypeError("TensorFieldManifest.batch_axis must be an integer.")
        if self.storage_key is not None:
            object.__setattr__(
                self,
                "storage_key",
                _validate_key(self.storage_key, "TensorFieldManifest.storage_key"),
            )
        object.__setattr__(
            self,
            "storage_shape",
            _normalize_storage_shape(self.storage_shape),
        )
        if not isinstance(self.required, bool):
            raise TypeError("TensorFieldManifest.required must be a bool.")
        object.__setattr__(self, "role", _validate_key(self.role, "TensorFieldManifest.role"))

    @classmethod
    def from_field_spec(
        cls,
        spec: FieldSpec,
        *,
        storage_key: str | None = None,
        storage_shape: Sequence[int] | None = None,
    ) -> "TensorFieldManifest":
        if not isinstance(spec, FieldSpec):
            raise TypeError("from_field_spec expects a FieldSpec.")
        return cls(
            name=spec.name,
            dtype=spec.dtype_name(),
            shape=spec.shape,
            batch_axis=spec.batch_axis,
            storage_key=storage_key or spec.name,
            storage_shape=storage_shape,
            required=spec.required,
            role=spec.role,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "batch_axis": self.batch_axis,
            "storage_key": self.storage_key,
            "storage_shape": None if self.storage_shape is None else list(self.storage_shape),
            "required": self.required,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TensorFieldManifest":
        if not isinstance(payload, Mapping):
            raise TypeError("TensorFieldManifest.from_dict expects a mapping.")
        return cls(
            name=payload["name"],
            dtype=payload["dtype"],
            shape=payload.get("shape"),
            batch_axis=payload.get("batch_axis", 0),
            storage_key=payload.get("storage_key"),
            storage_shape=payload.get("storage_shape"),
            required=payload.get("required", True),
            role=payload.get("role", "feature"),
        )


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    schema_id: str
    num_rows: int
    fields: Sequence[TensorFieldManifest]
    key_mapping: KeyMapping
    schema_version: int = 1
    storage_format: str = "tensordict"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_id", _validate_key(self.schema_id, "DatasetManifest.schema_id"))
        if not isinstance(self.num_rows, int) or isinstance(self.num_rows, bool):
            raise TypeError("DatasetManifest.num_rows must be an integer.")
        if self.num_rows < 0:
            raise ValueError("DatasetManifest.num_rows must be non-negative.")
        object.__setattr__(self, "fields", _normalize_field_manifests(self.fields))
        if not isinstance(self.key_mapping, KeyMapping):
            raise TypeError("DatasetManifest.key_mapping must be a KeyMapping.")
        if not isinstance(self.schema_version, int) or isinstance(self.schema_version, bool):
            raise TypeError("DatasetManifest.schema_version must be an integer.")
        if self.schema_version <= 0:
            raise ValueError("DatasetManifest.schema_version must be positive.")
        object.__setattr__(
            self,
            "storage_format",
            _validate_key(self.storage_format, "DatasetManifest.storage_format"),
        )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("DatasetManifest.metadata must be a mapping.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatasetManifest":
        if not isinstance(payload, Mapping):
            raise TypeError("DatasetManifest.from_dict expects a mapping.")
        key_mapping_payload = payload["key_mapping"]
        if not isinstance(key_mapping_payload, Mapping):
            raise TypeError("DatasetManifest key_mapping payload must be a mapping.")
        return cls(
            schema_id=payload["schema_id"],
            num_rows=payload["num_rows"],
            fields=[
                TensorFieldManifest.from_dict(field_payload)
                for field_payload in payload["fields"]
            ],
            key_mapping=KeyMapping(
                inputs=key_mapping_payload.get("inputs", {}),
                labels=key_mapping_payload.get("labels", {}),
                metadata=key_mapping_payload.get("metadata", {}),
            ),
            schema_version=payload.get("schema_version", 1),
            storage_format=payload.get("storage_format", "tensordict"),
            metadata=payload.get("metadata", {}),
        )

    @classmethod
    def from_schema(
        cls,
        schema: DataSchema,
        *,
        num_rows: int,
        storage_format: str = "tensordict",
        metadata: Mapping[str, Any] | None = None,
    ) -> "DatasetManifest":
        if not isinstance(schema, DataSchema):
            raise TypeError("from_schema expects a DataSchema.")
        return cls(
            schema_id=schema.schema_id,
            num_rows=num_rows,
            fields=[
                TensorFieldManifest.from_field_spec(
                    field,
                )
                for field in schema.fields
            ],
            key_mapping=schema.key_mapping,
            schema_version=schema.version,
            storage_format=storage_format,
            metadata=dict(metadata or schema.metadata),
        )

    def to_schema(self) -> DataSchema:
        fields = [
            FieldSpec(
                name=field.name,
                dtype=field.dtype,
                shape=field.shape,
                batch_axis=field.batch_axis,
                required=field.required,
                role=field.role,
            )
            for field in self.fields
        ]
        return DataSchema(
            schema_id=self.schema_id,
            fields=fields,
            key_mapping=self.key_mapping,
            version=self.schema_version,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_id": self.schema_id,
            "num_rows": self.num_rows,
            "storage_format": self.storage_format,
            "fields": [field.to_dict() for field in self.fields],
            "key_mapping": {
                "inputs": dict(self.key_mapping.inputs),
                "labels": dict(self.key_mapping.labels),
                "metadata": dict(self.key_mapping.metadata),
            },
            "metadata": dict(self.metadata),
        }

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "DatasetManifest":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)
