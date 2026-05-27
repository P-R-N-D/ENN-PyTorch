from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from torch import Tensor


def _is_sequence(value: object) -> bool:
    return not isinstance(value, (str, bytes, bytearray)) and value is not None


def _normalize_int_tuple(
    value: object,
    *,
    field_name: str,
    allow_none: bool = False,
) -> tuple[int, ...] | None:
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{field_name} must be a sequence of positive integers.")
    if not _is_sequence(value):
        raise TypeError(f"{field_name} must be a sequence of positive integers.")

    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"{field_name} must be a sequence of positive integers."
        ) from exc

    if not items:
        raise ValueError(f"{field_name} must not be empty.")
    if not all(
        isinstance(item, int) and not isinstance(item, bool)
        for item in items
    ):
        raise TypeError(f"{field_name} must contain integers only.")
    if not all(item > 0 for item in items):
        raise ValueError(f"{field_name} values must be positive.")
    return items


def _normalize_dims(value: object, *, rank: int) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not _is_sequence(value):
        raise TypeError("dims must be a sequence of integers.")

    try:
        dims = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("dims must be a sequence of integers.") from exc

    if len(dims) != rank:
        raise ValueError("dims length must match tile_shape length.")
    if not all(
        isinstance(dim, int) and not isinstance(dim, bool)
        for dim in dims
    ):
        raise TypeError("dims must contain integers only.")
    return dims


def _tile_starts(
    *,
    length: int,
    tile_size: int,
    stride: int,
    drop_last: bool,
) -> tuple[int, ...]:
    if length <= 0:
        return ()
    if drop_last:
        if length < tile_size:
            return ()
        return tuple(range(0, length - tile_size + 1, stride))

    return tuple(range(0, length, stride))


@dataclass(frozen=True, slots=True)
class TileMeta:
    index: int
    start: tuple[int, ...]
    end: tuple[int, ...]
    slices: tuple[slice, ...]
    full_shape: tuple[int, ...]
    dims: tuple[int, ...]


@dataclass(slots=True)
class TilePolicy:
    tile_shape: tuple[int, ...]
    stride: tuple[int, ...] | None = None
    dims: tuple[int, ...] | None = None
    drop_last: bool = False

    def __post_init__(self) -> None:
        tile_shape = _normalize_int_tuple(
            self.tile_shape,
            field_name="tile_shape",
        )
        assert tile_shape is not None
        self.tile_shape = tile_shape

        stride = _normalize_int_tuple(
            self.stride,
            field_name="stride",
            allow_none=True,
        )
        self.stride = tile_shape if stride is None else stride
        if len(self.stride) != len(self.tile_shape):
            raise ValueError("stride length must match tile_shape length.")

        self.dims = _normalize_dims(self.dims, rank=len(self.tile_shape))
        if not isinstance(self.drop_last, bool):
            raise TypeError("drop_last must be a bool.")

    def _normalized_dims_for(self, x: Tensor) -> tuple[int, ...]:
        ndim = x.ndim
        rank = len(self.tile_shape)
        if rank > ndim:
            raise ValueError(
                f"tile rank {rank} cannot exceed tensor ndim {ndim}."
            )

        if self.dims is None:
            dims = tuple(range(ndim - rank, ndim))
        else:
            dims = tuple(dim + ndim if dim < 0 else dim for dim in self.dims)

        for dim in dims:
            if dim < 0 or dim >= ndim:
                raise ValueError(f"tile dim out of range for ndim {ndim}: {dim!r}")
        if len(set(dims)) != len(dims):
            raise ValueError("dims must not contain duplicate dimensions.")
        return dims

    def split(self, x: Tensor) -> tuple[list[Tensor], list[TileMeta]]:
        if not isinstance(x, Tensor):
            raise TypeError(f"TilePolicy.split expects Tensor, got {type(x)!r}")

        dims = self._normalized_dims_for(x)
        starts_by_dim = [
            _tile_starts(
                length=int(x.shape[dim]),
                tile_size=tile_size,
                stride=stride,
                drop_last=self.drop_last,
            )
            for dim, tile_size, stride in zip(dims, self.tile_shape, self.stride)
        ]

        if any(not starts for starts in starts_by_dim):
            return [], []

        tiles: list[Tensor] = []
        metas: list[TileMeta] = []
        full_shape = tuple(int(size) for size in x.shape)

        for index, starts in enumerate(product(*starts_by_dim)):
            ends = tuple(
                min(start + tile_size, int(x.shape[dim]))
                for start, tile_size, dim in zip(starts, self.tile_shape, dims)
            )
            slices: list[Any] = [slice(None)] * x.ndim
            for dim, start, end in zip(dims, starts, ends):
                slices[dim] = slice(start, end)
            slice_tuple = tuple(slices)

            tiles.append(x[slice_tuple])
            metas.append(
                TileMeta(
                    index=index,
                    start=tuple(int(start) for start in starts),
                    end=tuple(int(end) for end in ends),
                    slices=slice_tuple,
                    full_shape=full_shape,
                    dims=dims,
                )
            )

        return tiles, metas
