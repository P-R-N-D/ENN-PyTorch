from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
from tensordict import TensorDict

from .batch import KVBatch
from .manifest import DatasetManifest, TensorFieldManifest
from .schema import DataSchema, FieldSpec
from .td_store import load_memmap_tensor, resolve_storage_path


def _dtype_from_name(name: str) -> torch.dtype:
    return FieldSpec("__dtype__", name).dtype


def _normalize_indices(
    indices: torch.Tensor | Sequence[int] | slice,
    *,
    num_rows: int,
) -> torch.Tensor | slice:
    if isinstance(indices, slice):
        start, stop, step = indices.indices(num_rows)
        if step <= 0:
            raise ValueError("negative or zero-step slices are not supported.")
        if start < 0 or stop < 0 or start > num_rows or stop > num_rows:
            raise IndexError("slice is out of range.")
        return slice(start, stop, step)
    if isinstance(indices, torch.Tensor):
        if indices.ndim == 0:
            indices = indices.reshape(1)
        if indices.ndim != 1:
            raise ValueError("indices tensor must be 1-D.")
        if indices.dtype == torch.bool:
            raise TypeError("indices tensor must not use torch.bool dtype.")
        if not indices.dtype.is_floating_point and not indices.dtype.is_complex:
            index_tensor = indices.detach().cpu().to(torch.long)
        else:
            raise TypeError("indices tensor must use an integer dtype.")
    else:
        index_tensor = torch.as_tensor(list(indices), dtype=torch.long)
    if index_tensor.numel() == 0:
        return index_tensor
    if bool((index_tensor < 0).any()) or bool((index_tensor >= num_rows).any()):
        raise IndexError("indices are out of range.")
    return index_tensor


def _batch_size_from_indices(indices: torch.Tensor | slice, *, num_rows: int) -> int:
    if isinstance(indices, slice):
        return len(range(indices.start, indices.stop, indices.step))
    return int(indices.numel())


class TensorDictReader:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._manifest = DatasetManifest.from_json(self._root / "manifest.json")
        self._schema = self._manifest.to_schema()
        self._fields = self._load_fields()
        self._row_ids = self._load_row_ids()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def manifest(self) -> DatasetManifest:
        return self._manifest

    @property
    def schema(self) -> DataSchema:
        return self._schema

    @property
    def num_rows(self) -> int:
        return self._manifest.num_rows

    def _load_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {}
        for field in self._manifest.fields:
            if field.storage_key is None or field.storage_shape is None:
                if field.required:
                    raise ValueError(
                        f"Required manifest field {field.name!r} has no storage."
                    )
                continue
            fields[field.name] = load_memmap_tensor(
                resolve_storage_path(self._root, field.storage_key),
                dtype=_dtype_from_name(field.dtype),
                shape=field.storage_shape,
            )
        return fields

    def _load_row_ids(self):
        row_id_storage_key = self._manifest.metadata.get("row_id_storage_key")
        if not isinstance(row_id_storage_key, str) or not row_id_storage_key:
            raise ValueError("DatasetManifest.metadata must include row_id_storage_key.")
        row_id_shape = self._manifest.metadata.get("row_id_shape")
        if row_id_shape is None:
            row_id_shape = [self._manifest.num_rows]
        return load_memmap_tensor(
            resolve_storage_path(self._root, row_id_storage_key),
            dtype=torch.long,
            shape=row_id_shape,
        )

    def get_rows(self, indices: torch.Tensor | Sequence[int] | slice) -> TensorDict:
        normalized = _normalize_indices(indices, num_rows=self.num_rows)
        batch_size = _batch_size_from_indices(normalized, num_rows=self.num_rows)
        data = {
            name: torch.as_tensor(tensor[normalized])
            for name, tensor in self._fields.items()
        }
        return TensorDict(data, batch_size=(batch_size,))

    def get_row_ids(self, indices: torch.Tensor | Sequence[int] | slice) -> torch.Tensor:
        normalized = _normalize_indices(indices, num_rows=self.num_rows)
        return torch.as_tensor(self._row_ids[normalized]).to(torch.long)

    def get_kvbatch(
        self,
        indices: torch.Tensor | Sequence[int] | slice,
        *,
        shard_id: int | None = None,
    ) -> KVBatch:
        td = self.get_rows(indices)
        row_ids = self.get_row_ids(indices)
        return KVBatch(
            td=td,
            row_ids=row_ids,
            schema_id=self._manifest.schema_id,
            shard_id=shard_id,
        )

    def iter_batches(
        self,
        batch_size: int,
        *,
        drop_last: bool = False,
    ) -> Iterator[KVBatch]:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        start = 0
        while start < self.num_rows:
            end = min(start + batch_size, self.num_rows)
            if drop_last and end - start < batch_size:
                break
            yield self.get_kvbatch(slice(start, end))
            start = end
