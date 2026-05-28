from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .tile_policy import TileMeta


@dataclass(slots=True)
class TileReconstructSpec:
    reduction: str = "overwrite"

    def __post_init__(self) -> None:
        if self.reduction not in {"overwrite", "sum", "mean"}:
            raise ValueError(
                "TileReconstructSpec.reduction must be one of: "
                "'overwrite', 'sum', 'mean'."
            )


class TileReconstructor:
    def __init__(self, spec: TileReconstructSpec) -> None:
        if not isinstance(spec, TileReconstructSpec):
            raise TypeError(
                f"TileReconstructor spec must be TileReconstructSpec, got {type(spec)!r}"
            )
        self.spec = spec

    def reconstruct(self, tiles: Sequence[Tensor], metas: Sequence[TileMeta]) -> Tensor:
        if isinstance(tiles, (str, bytes, bytearray)) or tiles is None:
            raise TypeError("tiles must be a sequence of Tensor.")
        if isinstance(metas, (str, bytes, bytearray)) or metas is None:
            raise TypeError("metas must be a sequence of TileMeta.")

        tiles = tuple(tiles)
        metas = tuple(metas)

        if len(tiles) == 0:
            raise ValueError("tiles must not be empty.")
        if len(tiles) != len(metas):
            raise ValueError("tiles length must match metas length.")

        first_tile = tiles[0]
        first_meta = metas[0]
        if not isinstance(first_tile, Tensor):
            raise TypeError(f"tile at index 0 must be Tensor, got {type(first_tile)!r}")
        if not isinstance(first_meta, TileMeta):
            raise TypeError(
                f"meta at index 0 must be TileMeta, got {type(first_meta)!r}"
            )

        dims = first_meta.dims
        out_shape = list(first_tile.shape)
        for dim in dims:
            out_shape[dim] = int(first_meta.full_shape[dim])

        out = torch.zeros(tuple(out_shape), dtype=first_tile.dtype, device=first_tile.device)
        counts = None
        if self.spec.reduction == "mean":
            counts = torch.zeros(tuple(out_shape), dtype=torch.float32, device=first_tile.device)

        for idx, (tile, meta) in enumerate(zip(tiles, metas)):
            if not isinstance(tile, Tensor):
                raise TypeError(f"tile at index {idx} must be Tensor, got {type(tile)!r}")
            if not isinstance(meta, TileMeta):
                raise TypeError(f"meta at index {idx} must be TileMeta, got {type(meta)!r}")
            if tile.dtype != first_tile.dtype:
                raise TypeError("all tiles must have same dtype.")
            if tile.device != first_tile.device:
                raise ValueError("all tiles must be on same device.")
            if meta.full_shape != first_meta.full_shape:
                raise ValueError("all metas must share the same full_shape.")
            if meta.dims != dims:
                raise ValueError("all metas must share the same dims.")

            expected_shape = list(out_shape)
            for d, start, end in zip(dims, meta.start, meta.end):
                expected_shape[d] = int(end - start)
            if tuple(tile.shape) != tuple(expected_shape):
                raise ValueError("tile shape mismatch against meta span/non-tiled dims.")

            sl = meta.slices
            if self.spec.reduction == "overwrite":
                out[sl] = tile
            elif self.spec.reduction == "sum":
                out[sl] = out[sl] + tile
            elif self.spec.reduction == "mean":
                out[sl] = out[sl] + tile
                assert counts is not None
                counts[sl] = counts[sl] + 1.0
            else:
                raise ValueError(f"Unsupported reduction: {self.spec.reduction!r}")

        if self.spec.reduction == "mean":
            assert counts is not None
            safe = torch.where(counts > 0, counts, torch.ones_like(counts))
            out = out / safe.to(dtype=out.dtype)

        return out
