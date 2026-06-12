from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch


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
        raise TypeError("FieldSpec.shape must be a sequence of integers or None values.")
    try:
        shape = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            "FieldSpec.shape must be a sequence of integers or None values."
        ) from exc
    for dim in shape:
        if dim is not None and (not isinstance(dim, int) or isinstance(dim, bool)):
            raise TypeError("FieldSpec.shape entries must be integers or None.")
        if isinstance(dim, int) and dim < 0:
            raise ValueError("FieldSpec.shape integer entries must be non-negative.")
    return shape


def _normalize_batch_axis(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("FieldSpec.batch_axis must be an integer.")
    return value


def _normalize_mapping(value: Mapping[str, str] | None, label: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping of strings to strings.")
    out: dict[str, str] = {}
    for source_key, target_key in value.items():
        source_key = _validate_key(source_key, f"{label} source key")
        target_key = _validate_key(target_key, f"{label}[{source_key!r}]")
        if source_key in out:
            raise ValueError(f"{label} contains duplicate source key: {source_key!r}")
        out[source_key] = target_key
    return out


def _normalize_fields(value: Sequence["FieldSpec"]) -> tuple["FieldSpec", ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("DataSchema.fields must be a sequence of FieldSpec values.")
    try:
        fields = tuple(value)
    except TypeError as exc:
        raise TypeError("DataSchema.fields must be a sequence of FieldSpec values.") from exc
    if not fields:
        raise ValueError("DataSchema.fields must not be empty.")
    names: set[str] = set()
    for field_spec in fields:
        if not isinstance(field_spec, FieldSpec):
            raise TypeError(
                f"DataSchema.fields must contain FieldSpec values, got {type(field_spec)!r}."
            )
        if field_spec.name in names:
            raise ValueError(f"DataSchema has duplicate field name: {field_spec.name!r}")
        names.add(field_spec.name)
    return fields


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Tensor field contract for a TensorDict-backed training batch."""

    name: str
    dtype: torch.dtype | str
    shape: Sequence[int | None] | None = None
    batch_axis: int = 0
    required: bool = True
    role: str = "feature"
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_key(self.name, "FieldSpec.name"))
        if not isinstance(self.dtype, (torch.dtype, str)):
            raise TypeError("FieldSpec.dtype must be a torch.dtype or dtype name string.")
        object.__setattr__(self, "shape", _normalize_shape(self.shape))
        object.__setattr__(self, "batch_axis", _normalize_batch_axis(self.batch_axis))
        if not isinstance(self.required, bool):
            raise TypeError("FieldSpec.required must be a bool.")
        object.__setattr__(self, "role", _validate_key(self.role, "FieldSpec.role"))
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("FieldSpec.description must be a string or None.")

    def dtype_name(self) -> str:
        if isinstance(self.dtype, torch.dtype):
            return str(self.dtype).removeprefix("torch.")
        return self.dtype

    def validate_tensor(self, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Field {self.name!r} expects a torch.Tensor, got {type(value)!r}.")
        if isinstance(self.dtype, torch.dtype) and value.dtype != self.dtype:
            raise TypeError(
                f"Field {self.name!r} expects dtype {self.dtype}, got {value.dtype}."
            )
        shape = tuple(self.shape)
        if shape and len(shape) != value.ndim:
            raise ValueError(
                f"Field {self.name!r} expects rank {len(shape)}, got {value.ndim}."
            )
        for dim_index, expected in enumerate(shape):
            if expected is not None and int(value.shape[dim_index]) != expected:
                raise ValueError(
                    f"Field {self.name!r} expects shape {shape}, got {tuple(value.shape)}."
                )


@dataclass(frozen=True, slots=True)
class KeyMapping:
    """Mapping from TensorDict field keys to executor KVStore keys."""

    inputs: Mapping[str, str]
    labels: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", _normalize_mapping(self.inputs, "inputs"))
        object.__setattr__(self, "labels", _normalize_mapping(self.labels, "labels"))
        object.__setattr__(
            self,
            "metadata",
            _normalize_mapping(self.metadata, "metadata"),
        )
        if not self.inputs:
            raise ValueError("KeyMapping.inputs must not be empty.")

    def all_items(self) -> tuple[tuple[str, str], ...]:
        items: list[tuple[str, str]] = []
        items.extend(self.inputs.items())
        items.extend(self.labels.items())
        items.extend(self.metadata.items())
        seen_targets: set[str] = set()
        for _source, target in items:
            if target in seen_targets:
                raise ValueError(f"KeyMapping contains duplicate KVStore key: {target!r}")
            seen_targets.add(target)
        return tuple(items)


@dataclass(frozen=True, slots=True)
class DataSchema:
    schema_id: str
    fields: Sequence[FieldSpec]
    key_mapping: KeyMapping
    version: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_id", _validate_key(self.schema_id, "DataSchema.schema_id"))
        object.__setattr__(self, "fields", _normalize_fields(self.fields))
        if not isinstance(self.key_mapping, KeyMapping):
            raise TypeError("DataSchema.key_mapping must be a KeyMapping.")
        if not isinstance(self.version, int) or isinstance(self.version, bool):
            raise TypeError("DataSchema.version must be an integer.")
        if self.version <= 0:
            raise ValueError("DataSchema.version must be positive.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("DataSchema.metadata must be a mapping.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field_spec.name for field_spec in self.fields)

    def field(self, name: str) -> FieldSpec:
        name = _validate_key(name, "field name")
        for field_spec in self.fields:
            if field_spec.name == name:
                return field_spec
        raise KeyError(f"Unknown schema field: {name!r}")

    def validate_keys(self, keys: Sequence[str]) -> None:
        available = set(keys)
        missing = [field_spec.name for field_spec in self.fields if field_spec.required and field_spec.name not in available]
        if missing:
            raise KeyError(f"Missing required TensorDict fields: {missing!r}")


@dataclass(frozen=True, slots=True)
class BatchSpec:
    schema: DataSchema
    batch_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.schema, DataSchema):
            raise TypeError("BatchSpec.schema must be a DataSchema.")
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool):
            raise TypeError("BatchSpec.batch_size must be an integer.")
        if self.batch_size <= 0:
            raise ValueError("BatchSpec.batch_size must be positive.")
