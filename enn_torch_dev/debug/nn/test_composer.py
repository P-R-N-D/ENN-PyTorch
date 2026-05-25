from __future__ import annotations

import pytest
import torch

from enn_torch_dev.nn.blocks import Composer


def test_composer_forward_compat_returns_fixed_tuple_shapes():
    x = torch.randn(2, 3, 4, 5)
    composer = Composer(5, salience_mode="none")

    tokens, token_mask, attn_bias = composer.forward_compat(x)

    assert tokens.shape == (2, 12, 5)
    assert token_mask.shape == (2, 12)
    assert token_mask.dtype == torch.bool
    assert attn_bias.shape == (2, 1, 1, 12)


def test_composer_forward_compat_respects_masking():
    x = torch.randn(1, 2, 2, 3)
    mask = torch.tensor([[[True, False], [False, False]]])
    composer = Composer(3, salience_mode="none")

    tokens, token_mask, _ = composer.forward_compat(x, context_mask=mask)

    assert token_mask.tolist() == [[True, False, False, False]]
    assert torch.allclose(tokens[0, 1:], torch.zeros_like(tokens[0, 1:]))


@pytest.mark.parametrize("mode", ["score", "soft_topk"])
def test_composer_salience_modes_are_finite(mode):
    x = torch.randn(2, 3, 4, 5)
    mask = torch.rand(2, 3, 4) > 0.2
    composer = Composer(5, salience_mode=mode, salience_topk=3)

    summary = composer(x, context_mask=mask)

    assert summary.score is not None
    assert summary.salience is not None
    assert summary.attn_bias is not None
    assert torch.isfinite(summary.salience).all()
    assert torch.isfinite(summary.attn_bias).all()


def test_composer_rejects_negative_salience_bias_scale():
    with pytest.raises(ValueError, match="non-negative"):
        Composer(5, salience_bias_scale=-1.0)
