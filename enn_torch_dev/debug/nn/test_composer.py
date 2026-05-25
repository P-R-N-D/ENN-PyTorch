from __future__ import annotations

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
