from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDictBase

from enn_torch_dev.executor import KVStore

from .schema import DataSchema, FieldSpec, KeyMapping


def _validate_tensor(value: object, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor.")
    if value.ndim == 0:
        raise ValueError(f"{label} must include a batch dimension.")
    return value


def _tensor_bytes(value: torch.Tensor) -> int:
    return int(value.numel() * value.element_size())


@dataclass(frozen=True, slots=True)
class BatchCost:
    """Measured or estimated cost for one TensorDict-backed batch."""

    host_bytes: int | None = None
    device_bytes: int | None = None
    num_items: int | None = None
    num_tokens: int | None = None
    num_tiles: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "host_bytes",
            "device_bytes",
            "num_items",
            "num_tokens",
            "num_tiles",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"BatchCost.{name} must be an integer or None.")
            if value < 0:
                raise ValueError(f"BatchCost.{name} must be non-negative.")

    @classmethod
    def from_tensordict(cls, td: TensorDictBase) -> "BatchCost":
        if not isinstance(td, TensorDictBase):
            raise TypeError(f"from_tensordict expects TensorDictBase, got {type(td)!r}")
        host_bytes = 0
        for value in td.values():
            if isinstance(value, torch.Tensor):
                host_bytes += _tensor_bytes(value)
        batch_size = None
        if td.batch_size:
            batch_size = int(td.batch_size[0])
        return cls(host_bytes=host_bytes, num_items=batch_size)


@dataclass(frozen=True, slots=True)
class KVBatch:
    """TensorDict-backed batch that can be exposed to executor KVStore."""

    td: TensorDictBase
    row_ids: torch.Tensor
    source_ids: torch.Tensor | None = None
    sample_ids: torch.Tensor | None = None
    schema_id: str = ""
    shard_id: int | None = None
    cost_hint: BatchCost | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.td, TensorDictBase):
            raise TypeError(f"KVBatch.td must be TensorDictBase, got {type(self.td)!r}.")
        row_ids = _validate_tensor(self.row_ids, "KVBatch.row_ids")
        batch_size = self._infer_batch_size(self.td)
        if int(row_ids.shape[0]) != batch_size:
            raise ValueError(
                "KVBatch.row_ids first dimension must match batch size: "
                f"{int(row_ids.shape[0])} != {batch_size}."
            )
        if self.source_ids is not None:
            source_ids = _validate_tensor(self.source_ids, "KVBatch.source_ids")
            if int(source_ids.shape[0]) != batch_size:
                raise ValueError("KVBatch.source_ids first dimension must match batch size.")
        if self.sample_ids is not None:
            sample_ids = _validate_tensor(self.sample_ids, "KVBatch.sample_ids")
            if int(sample_ids.shape[0]) != batch_size:
                raise ValueError("KVBatch.sample_ids first dimension must match batch size.")
        if not isinstance(self.schema_id, str):
            raise TypeError("KVBatch.schema_id must be a string.")
        if self.shard_id is not None and (
            not isinstance(self.shard_id, int) or isinstance(self.shard_id, bool)
        ):
            raise TypeError("KVBatch.shard_id must be an integer or None.")
        if self.cost_hint is not None and not isinstance(self.cost_hint, BatchCost):
            raise TypeError("KVBatch.cost_hint must be a BatchCost or None.")

    @staticmethod
    def _infer_batch_size(td: TensorDictBase) -> int:
        if not td.batch_size:
            raise ValueError("KVBatch TensorDict must have a non-empty batch_size.")
        batch_size = int(td.batch_size[0])
        if batch_size <= 0:
            raise ValueError("KVBatch batch_size must be positive.")
        return batch_size

    @property
    def batch_size(self) -> int:
        return self._infer_batch_size(self.td)

    def validate_schema(self, schema: DataSchema) -> None:
        if not isinstance(schema, DataSchema):
            raise TypeError("validate_schema expects a DataSchema.")
        schema.validate_keys(tuple(str(key) for key in self.td.keys()))
        for field_spec in schema.fields:
            if field_spec.name not in self.td.keys():
                continue
            field_spec.validate_tensor(self.td[field_spec.name])

    @staticmethod
    def _optional_fields(schema: DataSchema | None) -> dict[str, FieldSpec]:
        if schema is None:
            return {}
        return {field.name: field for field in schema.fields if not field.required}

    def to_store(self, mapping: KeyMapping | DataSchema) -> KVStore:
        schema: DataSchema | None = None
        effective_schema_id = self.schema_id
        if isinstance(mapping, DataSchema):
            schema = mapping
            if self.schema_id and self.schema_id != schema.schema_id:
                raise ValueError(
                    "KVBatch.schema_id must match DataSchema.schema_id when "
                    "converting through a schema: "
                    f"{self.schema_id!r} != {schema.schema_id!r}."
                )
            effective_schema_id = self.schema_id or schema.schema_id
            self.validate_schema(schema)
            key_mapping = schema.key_mapping
        elif isinstance(mapping, KeyMapping):
            key_mapping = mapping
        else:
            raise TypeError("to_store expects a KeyMapping or DataSchema.")

        optional_fields = self._optional_fields(schema)
        store = KVStore()
        for source_key, target_key in key_mapping.all_items():
            if source_key not in self.td.keys():
                if source_key in optional_fields:
                    continue
                raise KeyError(f"TensorDict missing key required by mapping: {source_key!r}")
            value = self.td[source_key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"TensorDict key {source_key!r} must hold a torch.Tensor, got {type(value)!r}."
                )
            store.set(
                target_key,
                value,
                origin="KVBatch",
                meta={"source_key": source_key, "schema_id": effective_schema_id},
            )

        store.set(
            "row_id",
            self.row_ids,
            origin="KVBatch",
            meta={"schema_id": effective_schema_id},
        )
        if self.source_ids is not None:
            store.set(
                "source_id",
                self.source_ids,
                origin="KVBatch",
                meta={"schema_id": effective_schema_id},
            )
        if self.sample_ids is not None:
            store.set(
                "sample_id",
                self.sample_ids,
                origin="KVBatch",
                meta={"schema_id": effective_schema_id},
            )
        return store
