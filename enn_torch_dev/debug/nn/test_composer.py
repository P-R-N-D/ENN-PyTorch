from __future__ import annotations

import pytest
import torch

from enn_torch_dev.nn.blocks import Composer


class _FirstFeatureScore(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :1]


def _soft_topk_first_feature_composer(
    *,
    salience_topk: int | float | None,
    temperature: float = 1.0,
) -> Composer:
    composer = Composer(
        2,
        salience_mode="soft_topk",
        salience_topk=salience_topk,
        salience_temperature=temperature,
    )
    composer.input_norm = torch.nn.Identity()
    composer.salience_score = _FirstFeatureScore()
    return composer


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
    kwargs = {"salience_topk": 3} if mode == "soft_topk" else {}
    composer = Composer(5, salience_mode=mode, **kwargs)

    summary = composer(x, context_mask=mask)

    assert summary.score is not None
    assert summary.salience is not None
    assert summary.attn_bias is not None
    assert torch.isfinite(summary.salience).all()
    assert torch.isfinite(summary.attn_bias).all()


def test_composer_rejects_negative_salience_bias_scale():
    with pytest.raises(ValueError, match="non-negative"):
        Composer(5, salience_mode="score", salience_bias_scale=-1.0)


def test_composer_none_mode_has_no_salience_parameters():
    composer = Composer(5, salience_mode="none")

    keys = set(composer.state_dict())

    assert not any(
        key.startswith(("input_norm.", "salience_score.")) for key in keys
    )


def test_composer_rejects_unused_salience_args_in_none_mode():
    with pytest.raises(ValueError, match="salience_topk"):
        Composer(5, salience_mode="none", salience_topk=3)


def test_composer_rejects_unused_topk_in_score_mode():
    with pytest.raises(ValueError, match="salience_topk"):
        Composer(5, salience_mode="score", salience_topk=3)


def test_composer_rejects_unused_temperature_in_score_mode():
    with pytest.raises(ValueError, match="salience_temperature"):
        Composer(5, salience_mode="score", salience_temperature=0.5)


def test_composer_soft_topk_uses_valid_token_count_for_integer_topk():
    x = torch.tensor(
        [[[[4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0]]]]
    )
    mask = torch.tensor([[[True, True, False, False]]])
    composer = _soft_topk_first_feature_composer(salience_topk=3)

    summary = composer(x, context_mask=mask)

    expected = torch.tensor(
        [[torch.sigmoid(torch.tensor(1.0)).item(), 0.5, 0.0, 0.0]],
        dtype=summary.salience.dtype,
        device=summary.salience.device,
    )
    assert torch.allclose(summary.salience, expected)


def test_composer_soft_topk_uses_valid_token_count_for_ratio_topk():
    x = torch.tensor(
        [[[[4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0]]]]
    )
    mask = torch.tensor([[[True, True, False, False]]])
    composer = _soft_topk_first_feature_composer(salience_topk=0.5)

    summary = composer(x, context_mask=mask)

    expected = torch.tensor(
        [[0.5, torch.sigmoid(torch.tensor(-1.0)).item(), 0.0, 0.0]],
        dtype=summary.salience.dtype,
        device=summary.salience.device,
    )
    assert torch.allclose(summary.salience, expected)


def test_composer_soft_topk_all_invalid_mask_returns_zero_salience():
    x = torch.tensor([[[[4.0, 0.0], [3.0, 0.0]]]])
    mask = torch.tensor([[[False, False]]])
    composer = _soft_topk_first_feature_composer(salience_topk=1)

    summary = composer(x, context_mask=mask)

    assert summary.salience is not None
    assert summary.attn_bias is not None
    assert torch.equal(summary.salience, torch.zeros_like(summary.salience))

    min_value = torch.finfo(summary.attn_bias.dtype).min
    assert torch.equal(
        summary.attn_bias,
        torch.full_like(summary.attn_bias, min_value),
    )


def test_composer_region_mask_matches_equivalent_context_mask():
    x = torch.randn(2, 3, 2, 5)
    region_mask = torch.tensor(
        [[True, False, True], [False, True, True]]
    )
    context_mask = region_mask.unsqueeze(-1).expand(2, 3, 2)
    composer = Composer(5, salience_mode="soft_topk", salience_topk=1)

    region_summary = composer(x, region_mask=region_mask)
    context_summary = composer(x, context_mask=context_mask)

    assert torch.equal(region_summary.token_mask, context_summary.token_mask)
    assert torch.allclose(region_summary.tokens, context_summary.tokens)
    assert region_summary.salience is not None
    assert context_summary.salience is not None
    assert torch.allclose(region_summary.salience, context_summary.salience)
    assert region_summary.attn_bias is not None
    assert context_summary.attn_bias is not None
    assert torch.allclose(region_summary.attn_bias, context_summary.attn_bias)


def test_composer_restore_masks_invalid_tokens():
    x = torch.randn(1, 2, 2, 3)
    mask = torch.tensor([[[True, False], [False, True]]])
    composer = Composer(3, salience_mode="none")

    summary = composer(x, context_mask=mask)
    tokens = torch.ones_like(summary.tokens)
    restored = composer.restore(tokens, summary)

    expected_mask = mask.unsqueeze(-1).expand_as(restored)
    assert torch.equal(
        restored,
        torch.where(expected_mask, torch.ones_like(restored), torch.zeros_like(restored)),
    )
