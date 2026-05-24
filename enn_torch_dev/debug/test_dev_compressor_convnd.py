from __future__ import annotations

import pytest
import torch

from enn_torch_dev.nn.blocks import Compressor
from enn_torch_dev.nn.layers import ConvND


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3, 5, 4),
        (2, 3, 5, 6, 4),
        (2, 3, 2, 5, 6, 4),
    ],
)
def test_convnd_preserves_supported_local_shapes(shape):
    x = torch.randn(*shape)
    layer = ConvND(4, residual_scale_init=0.0)

    y = layer(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_convnd_fixed_local_ndim_rejects_rank_mismatch():
    x = torch.randn(2, 3, 5, 6, 4)
    layer = ConvND(4, local_ndim=1)

    with pytest.raises(ValueError, match="fixed local_ndim"):
        layer(x)


def test_compressor_masks_all_invalid_regions_without_nan():
    x = torch.randn(2, 3, 7, 4)
    mask = torch.ones(2, 3, 7, dtype=torch.bool)
    mask[0, 1] = False

    module = Compressor(4, num_slots=2, use_conv=False)
    out, weights = module(x, local_mask=mask, return_weights=True)

    assert out.shape == (2, 3, 2, 4)
    assert weights.shape == (2, 3, 7, 2)
    assert torch.isfinite(out).all()
    assert torch.isfinite(weights).all()
    assert torch.allclose(out[0, 1], torch.zeros_like(out[0, 1]))
    assert torch.allclose(weights[0, 1], torch.zeros_like(weights[0, 1]))


def test_compressor_chunked_pooling_matches_dense_pooling():
    x = torch.randn(2, 3, 17, 4)
    mask = torch.rand(2, 3, 17) > 0.25
    mask[0, 2] = False

    dense = Compressor(4, num_slots=3, use_conv=False, pool_chunk_size=None)
    chunked = Compressor(4, num_slots=3, use_conv=False, pool_chunk_size=5)
    chunked.load_state_dict(dense.state_dict())

    dense_out = dense(x, local_mask=mask)
    chunked_out = chunked(x, local_mask=mask)

    assert torch.allclose(chunked_out, dense_out, atol=1e-5, rtol=1e-5)


def test_compressor_chunked_pooling_handles_empty_local_dim():
    x = torch.randn(2, 3, 0, 4)

    dense = Compressor(4, num_slots=2, use_conv=False, pool_chunk_size=None)
    chunked = Compressor(4, num_slots=2, use_conv=False, pool_chunk_size=4)
    chunked.load_state_dict(dense.state_dict())

    dense_out = dense(x)
    chunked_out = chunked(x)

    assert dense_out.shape == (2, 3, 2, 4)
    assert torch.allclose(chunked_out, dense_out, atol=1e-6, rtol=1e-6)


def test_compressor_chunked_pooling_scores_are_not_recomputed_between_passes():
    x = torch.randn(2, 3, 17, 4)
    module = Compressor(4, num_slots=3, use_conv=False, pool_chunk_size=5, dropout=0.5)
    module.train()

    calls = 0
    orig_forward = module.score.forward

    def counted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return orig_forward(*args, **kwargs)

    module.score.forward = counted_forward
    try:
        _ = module(x)
    finally:
        module.score.forward = orig_forward

    assert calls == 4


def test_compressor_forward_export_has_stable_tensor_return():
    x = torch.randn(2, 3, 5, 4)
    mask = torch.ones(2, 3, 5, dtype=torch.bool)

    module = Compressor(4, num_slots=2, use_conv=False, pool_chunk_size=2)

    masked = module.forward_export(x, mask)
    unmasked = module.forward_export_nomask(x)

    assert masked.shape == (2, 3, 2, 4)
    assert unmasked.shape == (2, 3, 2, 4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compressor_amp_friendly_dtypes_are_finite(dtype):
    x = torch.randn(2, 3, 11, 4, dtype=dtype)
    mask = torch.rand(2, 3, 11) > 0.2
    mask[1, 0] = False

    module = Compressor(4, num_slots=2, use_conv=False).to(dtype=dtype)
    out = module(x, local_mask=mask)

    assert out.dtype == dtype
    assert torch.isfinite(out.float()).all()


def test_compressor_dense_preserves_float64_weights():
    x = torch.randn(2, 3, 17, 4, dtype=torch.float64)
    module = Compressor(4, num_slots=3, use_conv=False).to(dtype=torch.float64)

    out, weights = module(x, return_weights=True)

    assert out.dtype == torch.float64
    assert weights.dtype == torch.float64
    assert torch.isfinite(out).all()
    assert torch.isfinite(weights).all()


def test_compressor_chunked_pooling_matches_dense_float64():
    x = torch.randn(2, 3, 17, 4, dtype=torch.float64)
    mask = torch.rand(2, 3, 17) > 0.25
    mask[0, 2] = False

    dense = Compressor(
        4, num_slots=3, use_conv=False, pool_chunk_size=None
    ).to(dtype=torch.float64)
    chunked = Compressor(
        4, num_slots=3, use_conv=False, pool_chunk_size=5
    ).to(dtype=torch.float64)
    chunked.load_state_dict(dense.state_dict())

    dense_out = dense(x, local_mask=mask)
    chunked_out = chunked(x, local_mask=mask)

    assert torch.allclose(chunked_out, dense_out, atol=1e-10, rtol=1e-10)


def test_compressor_accepts_integral_numeric_input():
    x = torch.randint(0, 10, (2, 3, 5, 4), dtype=torch.int64)
    module = Compressor(4, num_slots=2, use_conv=False)

    out = module(x)

    assert out.is_floating_point()
    assert out.dtype == torch.float32
    assert out.shape == (2, 3, 2, 4)


def test_compressor_rejects_integral_input_when_configured():
    x = torch.randint(0, 10, (2, 3, 5, 4), dtype=torch.int64)
    module = Compressor(4, num_slots=2, use_conv=False, integral_mode="reject")

    with pytest.raises(TypeError, match="integral input"):
        module(x)


def test_compressor_default_rejects_complex_but_loads_legacy_default_state_dict():
    module = Compressor(4, num_slots=2, use_conv=False)
    legacy = Compressor(
        4, num_slots=2, use_conv=False, complex_mode="real_imag"
    )
    x = torch.randn(2, 3, 5, 4, dtype=torch.complex64)

    assert isinstance(module.complex_input_proj, torch.nn.Linear)
    assert "complex_input_proj.weight" in module.state_dict()

    module.load_state_dict(legacy.state_dict(), strict=True)

    with pytest.raises(TypeError, match="complex input"):
        module(x)


def test_compressor_explicit_reject_omits_complex_projection_when_dims_match():
    module = Compressor(
        4, num_slots=2, use_conv=False, complex_mode="reject"
    )
    x = torch.randn(2, 3, 5, 4, dtype=torch.complex64)

    assert isinstance(module.complex_input_proj, torch.nn.Identity)
    assert "complex_input_proj.weight" not in module.state_dict()

    with pytest.raises(TypeError, match="complex input"):
        module(x)


def test_compressor_explicit_reject_preserves_projection_when_input_dim_differs():
    legacy = Compressor(
        8, input_dim=4, num_slots=2, use_conv=False, complex_mode="reject"
    )
    module = Compressor(
        8, input_dim=4, num_slots=2, use_conv=False, complex_mode="reject"
    )

    assert isinstance(module.real_input_proj, torch.nn.Linear)
    assert isinstance(module.complex_input_proj, torch.nn.Linear)
    assert "real_input_proj.weight" in module.state_dict()
    assert "complex_input_proj.weight" in module.state_dict()
    assert module.complex_input_proj.weight.shape == (8, 4)

    module.load_state_dict(legacy.state_dict(), strict=True)

    with pytest.raises(TypeError, match="complex input"):
        module(torch.randn(2, 3, 5, 4, dtype=torch.complex64))


def test_compressor_accepts_complex_real_imag_input():
    x = torch.randn(2, 3, 5, 4, dtype=torch.complex64)
    module = Compressor(
        dim=8,
        input_dim=4,
        num_slots=2,
        use_conv=False,
        complex_mode="real_imag",
    )

    out = module(x)

    assert out.is_floating_point()
    assert out.dtype == torch.float32
    assert out.shape == (2, 3, 2, 8)


def test_compressor_accepts_complex_abs_input():
    x = torch.randn(2, 3, 5, 4, dtype=torch.complex64)
    module = Compressor(
        dim=4,
        input_dim=4,
        num_slots=2,
        use_conv=False,
        complex_mode="abs",
    )

    out = module(x)

    assert out.is_floating_point()
    assert out.shape == (2, 3, 2, 4)
