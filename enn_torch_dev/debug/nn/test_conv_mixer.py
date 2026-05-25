from __future__ import annotations

import pytest
import torch

from enn_torch_dev.nn.layers import ConvMixer


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3, 5, 4),
        (2, 3, 5, 6, 4),
        (2, 3, 2, 5, 6, 4),
    ],
)
def test_conv_mixer_preserves_supported_local_shapes(shape):
    x = torch.randn(*shape)
    layer = ConvMixer(4, residual_scale_init=0.0)

    y = layer(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_conv_mixer_accepts_real_floating_dtypes(dtype):
    x = torch.randn(2, 3, 5, 4, dtype=dtype)
    layer = ConvMixer(4, residual_scale_init=0.0).to(dtype=dtype)

    y = layer(x)

    assert y.shape == x.shape
    assert y.dtype == dtype
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_conv_mixer_accepts_low_precision_floats_on_cuda(dtype):
    x = torch.randn(2, 3, 5, 4, device="cuda", dtype=dtype)
    layer = ConvMixer(4, residual_scale_init=0.0).cuda().to(dtype=dtype)

    y = layer(x)

    assert y.shape == x.shape
    assert y.dtype == dtype
    assert torch.isfinite(y.float()).all()


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.complex64,
        torch.complex128,
    ],
)
def test_conv_mixer_rejects_unsupported_dtype_on_conv_route(dtype):
    layer = ConvMixer(4)
    x = torch.ones(2, 3, 5, 4, dtype=dtype)

    with pytest.raises(TypeError, match="real floating point"):
        layer(x)


def test_conv_mixer_rejects_quantized_tensor_on_conv_route():
    layer = ConvMixer(4)
    base = torch.randn(2, 3, 5, 4)
    x = torch.quantize_per_tensor(base, scale=0.1, zero_point=0, dtype=torch.qint8)

    with pytest.raises(TypeError, match="real floating point"):
        layer(x)


@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.complex64])
def test_conv_mixer_identity_fallback_does_not_validate_dtype(dtype):
    layer = ConvMixer(4)
    rank0 = torch.ones(2, 3, 4, dtype=dtype)
    rank4 = torch.ones(2, 3, 2, 3, 4, 5, 4, dtype=dtype)

    assert layer(rank0) is rank0
    assert layer(rank4) is rank4


def test_conv_mixer_fixed_local_ndim_rejects_rank_mismatch():
    x = torch.randn(2, 3, 5, 6, 4)
    layer = ConvMixer(4, local_ndim=1)

    with pytest.raises(ValueError, match="fixed local_ndim"):
        layer(x)


def test_conv_mixer_rank0_and_rank_gt3_identity_fallback():
    layer = ConvMixer(4)
    rank0 = torch.randn(2, 3, 4)
    rank4 = torch.randn(2, 3, 2, 3, 4, 5, 4)

    assert layer(rank0) is rank0
    assert layer(rank4) is rank4


def test_conv_mixer_disabled_omits_conv_parameters():
    layer = ConvMixer(4, enabled=False)
    x = torch.randn(2, 3, 5, 4)

    assert layer(x) is x
    assert not list(layer.parameters())
    assert not any(
        key.startswith(("conv1.", "conv2.", "conv3."))
        for key in layer.state_dict()
    )


def test_conv_mixer_disabled_loads_legacy_state_dict_strict():
    legacy = ConvMixer(4, enabled=True)
    disabled = ConvMixer(4, enabled=False)

    disabled.load_state_dict(legacy.state_dict(), strict=True)


def test_conv_mixer_disabled_rejects_non_legacy_unexpected_key_strict():
    disabled = ConvMixer(4, enabled=False)
    state = disabled.state_dict()
    state["unexpected.weight"] = torch.randn(1)

    with pytest.raises(RuntimeError, match="Unexpected key"):
        disabled.load_state_dict(state, strict=True)
