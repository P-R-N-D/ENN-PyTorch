from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from tensordict import MemoryMappedTensor


def tensor_storage_key(field_name: str) -> str:
    if not isinstance(field_name, str) or not field_name:
        raise ValueError("field_name must be a non-empty string.")
    if "/" in field_name or "\\" in field_name:
        raise ValueError("field_name must not contain path separators.")
    return f"tensors/{field_name}.mmt"


def row_id_storage_key() -> str:
    return "index/row_id.mmt"


def resolve_storage_path(root: str | Path, storage_key: str) -> Path:
    root = Path(root)
    if not isinstance(storage_key, str) or not storage_key:
        raise ValueError("storage_key must be a non-empty string.")
    path = (root / storage_key).resolve()
    root_resolved = root.resolve()
    if root_resolved not in path.parents and path != root_resolved:
        raise ValueError(f"storage_key escapes staging root: {storage_key!r}")
    return path


def tensor_filename(root: str | Path, field_name: str) -> Path:
    return resolve_storage_path(root, tensor_storage_key(field_name))


def row_id_filename(root: str | Path) -> Path:
    return resolve_storage_path(root, row_id_storage_key())


def write_memmap_tensor(
    tensor: torch.Tensor,
    filename: str | Path,
    *,
    overwrite: bool = False,
) -> MemoryMappedTensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("write_memmap_tensor expects a torch.Tensor.")
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    if filename.exists() and not overwrite:
        raise FileExistsError(f"MemoryMappedTensor file already exists: {filename}")
    return MemoryMappedTensor.from_tensor(
        tensor.detach().cpu().contiguous(),
        filename=filename,
        existsok=overwrite,
    )


def load_memmap_tensor(
    filename: str | Path,
    *,
    dtype: torch.dtype,
    shape: Sequence[int],
) -> MemoryMappedTensor:
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"MemoryMappedTensor file does not exist: {filename}")
    return MemoryMappedTensor.from_filename(
        filename=filename,
        dtype=dtype,
        shape=torch.Size(shape),
    )
