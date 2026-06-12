from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from tensordict import TensorDictBase

from .manifest import DatasetManifest, TensorFieldManifest
from .schema import DataSchema, FieldSpec
from .td_store import (
    row_id_filename,
    row_id_storage_key,
    tensor_filename,
    tensor_storage_key,
    write_memmap_tensor,
)


@dataclass(frozen=True, slots=True)
class StagingSpec:
    root: Path
    schema: DataSchema
    overwrite: bool = False
    storage_format: str = "field_memmap"
    row_id_key: str = "row_id"

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if not isinstance(self.schema, DataSchema):
            raise TypeError("StagingSpec.schema must be a DataSchema.")
        if not isinstance(self.overwrite, bool):
            raise TypeError("StagingSpec.overwrite must be a bool.")
        if not isinstance(self.storage_format, str) or not self.storage_format:
            raise ValueError("StagingSpec.storage_format must be a non-empty string.")
        if not isinstance(self.row_id_key, str) or not self.row_id_key:
            raise ValueError("StagingSpec.row_id_key must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class StagingResult:
    root: Path
    manifest: DatasetManifest
    row_ids: torch.Tensor


def _source_to_mapping(
    source: TensorDictBase | Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if isinstance(source, TensorDictBase):
        out: dict[str, torch.Tensor] = {}
        for key in source.keys():
            if not isinstance(key, str):
                raise TypeError("Nested TensorDict keys are not supported in staging yet.")
            value = source[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"TensorDict key {key!r} must hold a torch.Tensor.")
            out[key] = value
        return out
    if isinstance(source, Mapping):
        out = {}
        for key, value in source.items():
            if not isinstance(key, str):
                raise TypeError("source mapping keys must be strings.")
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"source key {key!r} must hold a torch.Tensor.")
            out[key] = value
        return out
    raise TypeError("staging source must be a TensorDictBase or Mapping[str, Tensor].")


def _prepare_root(root: Path, *, overwrite: bool) -> None:
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"staging root already exists: {root}")
        shutil.rmtree(root)
    (root / "tensors").mkdir(parents=True, exist_ok=True)
    (root / "index").mkdir(parents=True, exist_ok=True)


def _validate_batch_axis(field_spec: FieldSpec) -> None:
    if field_spec.batch_axis != 0:
        raise NotImplementedError(
            "TensorDictStagingWriter currently supports batch_axis=0 only; "
            f"{field_spec.name!r} has batch_axis={field_spec.batch_axis}."
        )


def _field_row_count(field_spec: FieldSpec, tensor: torch.Tensor) -> int:
    _validate_batch_axis(field_spec)
    if tensor.ndim == 0:
        raise ValueError(f"Field {field_spec.name!r} must include a batch dimension.")
    return int(tensor.shape[field_spec.batch_axis])


def _validate_row_ids(row_ids: torch.Tensor, *, num_rows: int, row_id_key: str) -> torch.Tensor:
    if not isinstance(row_ids, torch.Tensor):
        raise TypeError(f"{row_id_key!r} must be a torch.Tensor.")
    if row_ids.ndim != 1:
        raise ValueError(f"{row_id_key!r} must be a 1-D tensor.")
    if int(row_ids.shape[0]) != num_rows:
        raise ValueError(
            f"{row_id_key!r} length must match row count: "
            f"{int(row_ids.shape[0])} != {num_rows}."
        )
    if row_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
        raise TypeError(f"{row_id_key!r} must use an integer dtype.")
    return row_ids.detach().cpu().to(torch.long).contiguous()


class TensorDictStagingWriter:
    def __init__(self, spec: StagingSpec) -> None:
        if not isinstance(spec, StagingSpec):
            raise TypeError("TensorDictStagingWriter expects a StagingSpec.")
        self._spec = spec

    @property
    def spec(self) -> StagingSpec:
        return self._spec

    def write(
        self,
        source: TensorDictBase | Mapping[str, torch.Tensor],
    ) -> StagingResult:
        schema = self._spec.schema
        root = self._spec.root
        values = _source_to_mapping(source)

        if self._spec.row_id_key in schema.field_names:
            raise ValueError(
                f"{self._spec.row_id_key!r} is reserved for runtime row identity "
                "and must not be declared as a DataSchema field."
            )

        stored_fields: list[tuple[FieldSpec, torch.Tensor]] = []
        num_rows: int | None = None
        for field_spec in schema.fields:
            _validate_batch_axis(field_spec)
            if field_spec.name not in values:
                if field_spec.required:
                    raise KeyError(f"Missing required staging field: {field_spec.name!r}")
                continue
            tensor = values[field_spec.name]
            field_spec.validate_tensor(tensor)
            field_rows = _field_row_count(field_spec, tensor)
            if num_rows is None:
                num_rows = field_rows
            elif field_rows != num_rows:
                raise ValueError(
                    "All staged fields must have the same row count; "
                    f"{field_spec.name!r} has {field_rows}, expected {num_rows}."
                )
            stored_fields.append((field_spec, tensor))

        if num_rows is None:
            raise ValueError("No schema-declared tensor fields were provided for staging.")

        if self._spec.row_id_key in values:
            row_ids = _validate_row_ids(
                values[self._spec.row_id_key],
                num_rows=num_rows,
                row_id_key=self._spec.row_id_key,
            )
        else:
            row_ids = torch.arange(num_rows, dtype=torch.long)

        _prepare_root(root, overwrite=self._spec.overwrite)

        field_manifests: list[TensorFieldManifest] = []
        for field_spec, tensor in stored_fields:
            storage_key = tensor_storage_key(field_spec.name)
            write_memmap_tensor(
                tensor,
                tensor_filename(root, field_spec.name),
                overwrite=True,
            )
            field_manifests.append(
                TensorFieldManifest.from_field_spec(
                    field_spec,
                    storage_key=storage_key,
                    storage_shape=tuple(int(dim) for dim in tensor.shape),
                )
            )

        write_memmap_tensor(
            row_ids,
            row_id_filename(root),
            overwrite=True,
        )

        metadata = dict(schema.metadata)
        metadata.update(
            {
                "row_id_storage_key": row_id_storage_key(),
                "row_id_dtype": row_ids.dtype_name if hasattr(row_ids, "dtype_name") else "int64",
                "row_id_shape": [int(dim) for dim in row_ids.shape],
                "storage_layout": "field_memmap",
            }
        )
        manifest = DatasetManifest(
            schema_id=schema.schema_id,
            num_rows=num_rows,
            fields=field_manifests,
            key_mapping=schema.key_mapping,
            schema_version=schema.version,
            storage_format=self._spec.storage_format,
            metadata=metadata,
        )
        manifest.to_json(root / "manifest.json")
        return StagingResult(root=root, manifest=manifest, row_ids=row_ids)
