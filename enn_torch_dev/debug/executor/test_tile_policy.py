from __future__ import annotations

import pytest
import torch

from enn_torch_dev.executor import TileMeta, TilePolicy


def test_tile_policy_splits_1d_tensor() -> None:
    x = torch.arange(5)
    policy = TilePolicy(tile_shape=(2,))

    tiles, metas = policy.split(x)

    assert [tile.tolist() for tile in tiles] == [[0, 1], [2, 3], [4]]
    assert [meta.start for meta in metas] == [(0,), (2,), (4,)]
    assert [meta.end for meta in metas] == [(2,), (4,), (5,)]
    assert metas[0].full_shape == (5,)
    assert metas[0].dims == (0,)
    assert isinstance(metas[0], TileMeta)


def test_tile_policy_drop_last_discards_incomplete_1d_tile() -> None:
    x = torch.arange(5)
    policy = TilePolicy(tile_shape=(2,), drop_last=True)

    tiles, metas = policy.split(x)

    assert [tile.tolist() for tile in tiles] == [[0, 1], [2, 3]]
    assert [meta.start for meta in metas] == [(0,), (2,)]
    assert [meta.end for meta in metas] == [(2,), (4,)]


def test_tile_policy_splits_2d_tensor_with_edge_tiles() -> None:
    x = torch.arange(12).reshape(3, 4)
    policy = TilePolicy(tile_shape=(2, 3))

    tiles, metas = policy.split(x)

    assert [tuple(tile.shape) for tile in tiles] == [
        (2, 3),
        (2, 1),
        (1, 3),
        (1, 1),
    ]
    assert [meta.start for meta in metas] == [
        (0, 0),
        (0, 3),
        (2, 0),
        (2, 3),
    ]
    assert [meta.end for meta in metas] == [
        (2, 3),
        (2, 4),
        (3, 3),
        (3, 4),
    ]
    assert metas[0].slices == (slice(0, 2), slice(0, 3))


def test_tile_policy_splits_only_selected_bchw_dims() -> None:
    x = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    policy = TilePolicy(tile_shape=(2, 3), stride=(2, 3), dims=(-2, -1))

    tiles, metas = policy.split(x)

    assert len(tiles) == 4
    assert [tuple(tile.shape) for tile in tiles] == [
        (2, 3, 2, 3),
        (2, 3, 2, 2),
        (2, 3, 2, 3),
        (2, 3, 2, 2),
    ]
    assert metas[0].dims == (2, 3)
    assert metas[0].full_shape == (2, 3, 4, 5)
    assert metas[0].slices == (
        slice(None),
        slice(None),
        slice(0, 2),
        slice(0, 3),
    )


def test_tile_policy_drop_last_discards_incomplete_spatial_tiles() -> None:
    x = torch.arange(1 * 1 * 4 * 5).reshape(1, 1, 4, 5)
    policy = TilePolicy(
        tile_shape=(3, 3),
        stride=(3, 3),
        dims=(-2, -1),
        drop_last=True,
    )

    tiles, metas = policy.split(x)

    assert len(tiles) == 1
    assert tuple(tiles[0].shape) == (1, 1, 3, 3)
    assert metas[0].start == (0, 0)
    assert metas[0].end == (3, 3)


def test_tile_policy_supports_overlap_stride() -> None:
    x = torch.arange(5)
    policy = TilePolicy(tile_shape=(3,), stride=(2,))

    tiles, metas = policy.split(x)

    assert [tile.tolist() for tile in tiles] == [[0, 1, 2], [2, 3, 4], [4]]
    assert [meta.start for meta in metas] == [(0,), (2,), (4,)]


def test_tile_policy_empty_dim_returns_no_tiles() -> None:
    x = torch.empty(0)
    policy = TilePolicy(tile_shape=(2,))

    tiles, metas = policy.split(x)

    assert tiles == []
    assert metas == []


def test_tile_policy_validates_init_arguments() -> None:
    with pytest.raises(ValueError, match="tile_shape"):
        TilePolicy(tile_shape=())

    with pytest.raises(ValueError, match="positive"):
        TilePolicy(tile_shape=(0,))

    with pytest.raises(TypeError, match="integers"):
        TilePolicy(tile_shape=(True,))

    with pytest.raises(ValueError, match="stride length"):
        TilePolicy(tile_shape=(2, 2), stride=(1,))

    with pytest.raises(TypeError, match="integers"):
        TilePolicy(tile_shape=(2,), stride=(False,))

    with pytest.raises(ValueError, match="dims length"):
        TilePolicy(tile_shape=(2, 2), dims=(0,))

    with pytest.raises(TypeError, match="integers"):
        TilePolicy(tile_shape=(2,), dims=(True,))

    with pytest.raises(TypeError, match="drop_last"):
        TilePolicy(tile_shape=(2,), drop_last=1)


def test_tile_policy_validates_split_arguments() -> None:
    with pytest.raises(TypeError, match="Tensor"):
        TilePolicy(tile_shape=(2,)).split([1, 2, 3])

    with pytest.raises(ValueError, match="cannot exceed"):
        TilePolicy(tile_shape=(2, 2, 2)).split(torch.zeros(2, 2))

    with pytest.raises(ValueError, match="duplicate"):
        TilePolicy(tile_shape=(2, 2), dims=(-1, 1)).split(torch.zeros(2, 2))

    with pytest.raises(ValueError, match="out of range"):
        TilePolicy(tile_shape=(2,), dims=(2,)).split(torch.zeros(2, 2))
