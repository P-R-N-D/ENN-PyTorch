from __future__ import annotations

import pytest
import torch

from enn_torch_dev.executor import TileMeta, TileReconstructSpec, TileReconstructor


def test_reconstruct_1d() -> None:
    tiles = [torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([5])]
    metas = [
        TileMeta(0, (0,), (2,), (slice(0, 2),), (5,), (0,)),
        TileMeta(1, (2,), (4,), (slice(2, 4),), (5,), (0,)),
        TileMeta(2, (4,), (5,), (slice(4, 5),), (5,), (0,)),
    ]
    out = TileReconstructor().reconstruct(tiles, metas)
    assert out.tolist() == [1, 2, 3, 4, 5]


def test_tile_reconstructor_default_spec_is_overwrite() -> None:
    reconstructor = TileReconstructor()

    assert reconstructor.spec.reduction == "overwrite"

    metas = [TileMeta(0, (0,), (1,), (slice(0, 1),), (1,), (0,))]
    out = reconstructor.reconstruct([torch.tensor([3.0])], metas)
    assert torch.equal(out, torch.tensor([3.0]))


def test_reconstruct_2d() -> None:
    tiles = [torch.ones(2, 2), torch.full((2, 1), 2.0), torch.full((1, 2), 3.0), torch.full((1, 1), 4.0)]
    metas = [
        TileMeta(0, (0, 0), (2, 2), (slice(0, 2), slice(0, 2)), (3, 3), (0, 1)),
        TileMeta(1, (0, 2), (2, 3), (slice(0, 2), slice(2, 3)), (3, 3), (0, 1)),
        TileMeta(2, (2, 0), (3, 2), (slice(2, 3), slice(0, 2)), (3, 3), (0, 1)),
        TileMeta(3, (2, 2), (3, 3), (slice(2, 3), slice(2, 3)), (3, 3), (0, 1)),
    ]
    out = TileReconstructor().reconstruct(tiles, metas)
    assert tuple(out.shape) == (3, 3)


def test_reconstruct_bchw_spatial_and_output_channel_change() -> None:
    t0 = torch.ones(1, 2, 2, 2)
    t1 = torch.full((1, 2, 2, 1), 2.0)
    metas = [
        TileMeta(0, (0, 0), (2, 2), (slice(None), slice(None), slice(0, 2), slice(0, 2)), (1, 3, 2, 3), (2, 3)),
        TileMeta(1, (0, 2), (2, 3), (slice(None), slice(None), slice(0, 2), slice(2, 3)), (1, 3, 2, 3), (2, 3)),
    ]
    out = TileReconstructor(TileReconstructSpec()).reconstruct([t0, t1], metas)
    assert tuple(out.shape) == (1, 2, 2, 3)


def test_overlap_reduction_overwrite_sum_mean() -> None:
    tiles = [torch.tensor([1.0, 1.0, 1.0]), torch.tensor([2.0, 2.0, 2.0])]
    metas = [
        TileMeta(0, (0,), (3,), (slice(0, 3),), (5,), (0,)),
        TileMeta(1, (2,), (5,), (slice(2, 5),), (5,), (0,)),
    ]
    ow = TileReconstructor(TileReconstructSpec("overwrite")).reconstruct(tiles, metas)
    sm = TileReconstructor(TileReconstructSpec("sum")).reconstruct(tiles, metas)
    mn = TileReconstructor(TileReconstructSpec("mean")).reconstruct(tiles, metas)
    assert ow.tolist() == [1.0, 1.0, 2.0, 2.0, 2.0]
    assert sm.tolist() == [1.0, 1.0, 3.0, 2.0, 2.0]
    assert mn.tolist() == [1.0, 1.0, 1.5, 2.0, 2.0]


def test_drop_last_hole_mean_becomes_zero() -> None:
    tiles = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    metas = [
        TileMeta(0, (0,), (2,), (slice(0, 2),), (5,), (0,)),
        TileMeta(1, (2,), (4,), (slice(2, 4),), (5,), (0,)),
    ]
    out = TileReconstructor(TileReconstructSpec("mean")).reconstruct(tiles, metas)
    assert out.tolist() == [1.0, 2.0, 3.0, 4.0, 0.0]


def test_invalid_reduction() -> None:
    with pytest.raises(ValueError, match="reduction"):
        TileReconstructSpec("bad")


def test_empty_tiles_and_length_mismatch() -> None:
    recon = TileReconstructor(TileReconstructSpec())
    with pytest.raises(ValueError, match="empty"):
        recon.reconstruct([], [])
    with pytest.raises(ValueError, match="length"):
        recon.reconstruct([torch.ones(1)], [])


def test_tile_meta_type_validation() -> None:
    recon = TileReconstructor(TileReconstructSpec())
    meta = TileMeta(0, (0,), (1,), (slice(0, 1),), (1,), (0,))
    with pytest.raises(TypeError, match="Tensor"):
        recon.reconstruct([1], [meta])
    with pytest.raises(TypeError, match="TileMeta"):
        recon.reconstruct([torch.ones(1)], [1])


def test_tile_shape_span_and_non_tiled_dim_mismatch() -> None:
    recon = TileReconstructor(TileReconstructSpec())
    meta = TileMeta(0, (0,), (2,), (slice(0, 2),), (2,), (0,))
    with pytest.raises(ValueError, match="mismatch"):
        recon.reconstruct([torch.ones(3)], [meta])

    metas = [
        TileMeta(0, (0, 0), (2, 1), (slice(None), slice(None), slice(0, 2), slice(0, 1)), (1, 3, 2, 2), (2, 3)),
        TileMeta(1, (0, 1), (2, 2), (slice(None), slice(None), slice(0, 2), slice(1, 2)), (1, 3, 2, 2), (2, 3)),
    ]
    with pytest.raises(ValueError, match="mismatch"):
        recon.reconstruct([torch.ones(1, 1, 2, 1), torch.ones(2, 1, 2, 1)], metas)


def test_dtype_mismatch() -> None:
    metas = [
        TileMeta(0, (0,), (1,), (slice(0, 1),), (2,), (0,)),
        TileMeta(1, (1,), (2,), (slice(1, 2),), (2,), (0,)),
    ]
    recon = TileReconstructor(TileReconstructSpec())
    with pytest.raises(TypeError, match="dtype"):
        recon.reconstruct([torch.ones(1, dtype=torch.float32), torch.ones(1, dtype=torch.int64)], metas)
