from __future__ import annotations

import pytest
import torch

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


def test_convnd_rank0_and_rank_gt3_identity_fallback():
    layer = ConvND(4)
    rank0 = torch.randn(2, 3, 4)
    rank4 = torch.randn(2, 3, 2, 3, 4, 5, 4)

    assert layer(rank0) is rank0
    assert layer(rank4) is rank4


def test_convnd_disabled_omits_conv_parameters():
    layer = ConvND(4, enabled=False)
    x = torch.randn(2, 3, 5, 4)

    assert layer(x) is x
    assert not list(layer.parameters())
    assert not any(
        key.startswith(("conv1.", "conv2.", "conv3."))
        for key in layer.state_dict()
    )


def test_convnd_disabled_loads_legacy_state_dict_strict():
    legacy = ConvND(4, enabled=True)
    disabled = ConvND(4, enabled=False)

    disabled.load_state_dict(legacy.state_dict(), strict=True)


def test_convnd_disabled_rejects_non_legacy_unexpected_key_strict():
    disabled = ConvND(4, enabled=False)
    state = disabled.state_dict()
    state["unexpected.weight"] = torch.randn(1)

    with pytest.raises(RuntimeError, match="Unexpected key"):
        disabled.load_state_dict(state, strict=True)
