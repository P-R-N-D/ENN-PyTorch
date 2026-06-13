from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from tensordict import TensorDict, TensorDictBase

from .batch import BatchCost, KVBatch
from .schema import DataSchema, RUNTIME_IDENTITY_KVSTORE_KEYS


TensorBatchSource = Mapping[str, object] | TensorDictBase


def _validate_key(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    if not value:
        raise ValueError(f"{label} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{label} must not have leading or trailing whitespace.")
    return value


def _is_integer_tensor(value: torch.Tensor) -> bool:
    return (
        value.dtype != torch.bool
        and not value.dtype.is_floating_point
        and not value.dtype.is_complex
    )


@dataclass(frozen=True, slots=True)
class SpdlAdapterKeys:
    """Runtime identity keys accepted from a tensorized SPDL batch."""

    row_id: str = "row_id"
    source_id: str = "source_id"
    sample_id: str = "sample_id"

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_id", _validate_key(self.row_id, "row_id key"))
        object.__setattr__(
            self,
            "source_id",
            _validate_key(self.source_id, "source_id key"),
        )
        object.__setattr__(
            self,
            "sample_id",
            _validate_key(self.sample_id, "sample_id key"),
        )
        values = (self.row_id, self.source_id, self.sample_id)
        if len(set(values)) != len(values):
            raise ValueError("SPDL adapter identity keys must be distinct.")

    def as_tuple(self) -> tuple[str, str, str]:
        return self.row_id, self.source_id, self.sample_id


@dataclass(frozen=True, slots=True)
class _NormalizedSpdlBatch:
    td: TensorDictBase
    row_ids: torch.Tensor
    source_ids: torch.Tensor | None
    sample_ids: torch.Tensor | None


class SpdlTensorAdapter:
    """Convert tensorized SPDL output into TensorDict and KVBatch objects.

    This adapter deliberately does not import or construct SPDL pipelines. It is
    the stable boundary for SPDL-style batches that are already tensorized as a
    Mapping[str, Tensor] or TensorDictBase.
    """

    def __init__(
        self,
        schema: DataSchema,
        *,
        row_id_key: str = "row_id",
        source_id_key: str = "source_id",
        sample_id_key: str = "sample_id",
    ) -> None:
        if not isinstance(schema, DataSchema):
            raise TypeError("SpdlTensorAdapter.schema must be a DataSchema.")
        self._schema = schema
        self._keys = SpdlAdapterKeys(
            row_id=row_id_key,
            source_id=source_id_key,
            sample_id=sample_id_key,
        )
        self._validate_identity_keys()
        self._validate_batch_axes()

    @property
    def schema(self) -> DataSchema:
        return self._schema

    @property
    def keys(self) -> SpdlAdapterKeys:
        return self._keys

    def to_tensordict(self, batch: TensorBatchSource) -> TensorDictBase:
        """Return a schema-validated TensorDict without runtime identity keys."""

        return self._normalize(batch).td

    def to_kvbatch(
        self,
        batch: TensorBatchSource,
        *,
        shard_id: int | None = None,
        cost_hint: BatchCost | None = None,
    ) -> KVBatch:
        """Return a KVBatch that can be consumed by RuntimeStep."""

        normalized = self._normalize(batch)
        return KVBatch(
            td=normalized.td,
            row_ids=normalized.row_ids,
            source_ids=normalized.source_ids,
            sample_ids=normalized.sample_ids,
            schema_id=self._schema.schema_id,
            shard_id=shard_id,
            cost_hint=cost_hint,
        )

    def _validate_identity_keys(self) -> None:
        field_names = set(self._schema.field_names)
        reserved_field_names = set(self._keys.as_tuple()) | RUNTIME_IDENTITY_KVSTORE_KEYS
        for key in reserved_field_names:
            if key in field_names:
                raise ValueError(
                    f"{key!r} is reserved for runtime identity and must not be "
                    "declared as a DataSchema field."
                )

    def _validate_batch_axes(self) -> None:
        for field in self._schema.fields:
            if field.batch_axis != 0:
                raise NotImplementedError(
                    "SpdlTensorAdapter currently supports batch_axis=0 only; "
                    f"{field.name!r} has batch_axis={field.batch_axis}."
                )

    def _normalize(self, source: TensorBatchSource) -> _NormalizedSpdlBatch:
        data, identities, source_batch_size = self._split_source(source)
        self._validate_declared_keys(data)
        self._schema.validate_keys(tuple(data.keys()))
        self._validate_schema_fields(data)
        batch_size = self._infer_batch_size(data, source_batch_size=source_batch_size)
        self._validate_batch_sizes(data, batch_size=batch_size)

        td = TensorDict(data, batch_size=(batch_size,))
        row_ids = self._normalize_identity_tensor(
            identities.get(self._keys.row_id),
            batch_size=batch_size,
            key=self._keys.row_id,
            required=True,
        )
        source_ids = self._normalize_identity_tensor(
            identities.get(self._keys.source_id),
            batch_size=batch_size,
            key=self._keys.source_id,
            required=False,
        )
        sample_ids = self._normalize_identity_tensor(
            identities.get(self._keys.sample_id),
            batch_size=batch_size,
            key=self._keys.sample_id,
            required=False,
        )
        assert row_ids is not None
        return _NormalizedSpdlBatch(
            td=td,
            row_ids=row_ids,
            source_ids=source_ids,
            sample_ids=sample_ids,
        )

    def _split_source(
        self,
        source: TensorBatchSource,
    ) -> tuple[dict[str, torch.Tensor], dict[str, object], int | None]:
        if isinstance(source, TensorDictBase):
            source_batch_size = int(source.batch_size[0]) if source.batch_size else None
            items = ((key, source[key]) for key in source.keys())
        elif isinstance(source, Mapping):
            source_batch_size = None
            items = source.items()
        else:
            raise TypeError(
                "SpdlTensorAdapter expects a Mapping[str, Tensor] or TensorDictBase."
            )

        identity_keys = set(self._keys.as_tuple())
        data: dict[str, torch.Tensor] = {}
        identities: dict[str, object] = {}
        for raw_key, value in items:
            key = _validate_key(raw_key, "SPDL batch key")
            if key in identity_keys:
                identities[key] = value
                continue
            if key not in self._schema.field_names:
                raise KeyError(
                    "SPDL tensor batch contains a key not declared in DataSchema: "
                    f"{key!r}."
                )
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"SPDL batch key {key!r} must hold a torch.Tensor, got {type(value)!r}."
                )
            if value.ndim == 0:
                raise ValueError(f"SPDL batch key {key!r} must include a batch dimension.")
            data[key] = value
        return data, identities, source_batch_size

    def _validate_declared_keys(self, data: Mapping[str, torch.Tensor]) -> None:
        field_names = set(self._schema.field_names)
        unknown_keys = [key for key in data if key not in field_names]
        if unknown_keys:
            raise KeyError(
                "SPDL tensor batch contains keys not declared in DataSchema: "
                f"{unknown_keys!r}."
            )

    def _validate_schema_fields(self, data: Mapping[str, torch.Tensor]) -> None:
        for field in self._schema.fields:
            if field.name not in data:
                continue
            field.validate_tensor(data[field.name])

    @staticmethod
    def _infer_batch_size(
        data: Mapping[str, torch.Tensor],
        *,
        source_batch_size: int | None,
    ) -> int:
        if source_batch_size is not None:
            if source_batch_size < 0:
                raise ValueError("SPDL TensorDict batch_size must be non-negative.")
            return source_batch_size
        for value in data.values():
            return int(value.shape[0])
        raise ValueError("SPDL tensor batch must contain at least one tensor field.")

    @staticmethod
    def _validate_batch_sizes(
        data: Mapping[str, torch.Tensor],
        *,
        batch_size: int,
    ) -> None:
        for key, value in data.items():
            if int(value.shape[0]) != batch_size:
                raise ValueError(
                    "All SPDL tensor batch fields must have the same batch size; "
                    f"{key!r} has {int(value.shape[0])}, expected {batch_size}."
                )

    @staticmethod
    def _normalize_identity_tensor(
        value: object | None,
        *,
        batch_size: int,
        key: str,
        required: bool,
    ) -> torch.Tensor | None:
        if value is None:
            if not required:
                return None
            return torch.arange(batch_size, dtype=torch.long)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key!r} must hold a torch.Tensor.")
        if value.ndim != 1:
            raise ValueError(f"{key!r} must be a 1-D tensor.")
        if int(value.shape[0]) != batch_size:
            raise ValueError(
                f"{key!r} length must match batch size: "
                f"{int(value.shape[0])} != {batch_size}."
            )
        if not _is_integer_tensor(value):
            raise TypeError(f"{key!r} must use an integer dtype.")
        return value.detach().cpu().to(torch.long).contiguous()
