from __future__ import annotations

import pytest
import torch

from enn_torch_dev.nn import GlobalSelfAttentionBlock


def test_global_self_attention_block_preserves_shape() -> None:
    block = GlobalSelfAttentionBlock(embed_dim=8, num_heads=2)
    x = torch.randn(2, 5, 8)

    out = block(x)

    assert out.shape == x.shape


def test_global_self_attention_block_supports_no_residual() -> None:
    block = GlobalSelfAttentionBlock(embed_dim=8, num_heads=2, residual=False)
    x = torch.randn(2, 5, 8)

    out = block(x)

    assert out.shape == x.shape


def test_global_self_attention_block_supports_no_norm() -> None:
    block = GlobalSelfAttentionBlock(embed_dim=8, num_heads=2, norm=False)
    x = torch.randn(2, 5, 8)

    out = block(x)

    assert out.shape == x.shape


def test_global_self_attention_block_accepts_key_padding_mask() -> None:
    block = GlobalSelfAttentionBlock(embed_dim=8, num_heads=2)
    x = torch.randn(2, 5, 8)
    mask = torch.tensor(
        [
            [False, False, False, False, True],
            [False, False, True, True, True],
        ]
    )

    out = block(x, key_padding_mask=mask)

    assert out.shape == x.shape


def test_global_self_attention_block_accepts_attn_mask() -> None:
    block = GlobalSelfAttentionBlock(embed_dim=8, num_heads=2)
    x = torch.randn(2, 5, 8)
    mask = torch.zeros(5, 5, dtype=torch.bool)
    mask[0, 4] = True

    out = block(x, attn_mask=mask)

    assert out.shape == x.shape


def test_global_self_attention_block_accepts_3d_attn_mask() -> None:
    block = GlobalSelfAttentionBlock(embed_dim=8, num_heads=2)
    x = torch.randn(2, 5, 8)
    mask = torch.zeros(2 * 2, 5, 5, dtype=torch.bool)

    out = block(x, attn_mask=mask)

    assert out.shape == x.shape


def test_global_self_attention_block_accepts_composer_attention_bias() -> None:
    block = GlobalSelfAttentionBlock(embed_dim=8, num_heads=2)
    x = torch.randn(2, 5, 8)
    attn_bias = torch.zeros(2, 1, 1, 5)

    out = block(x, attn_mask=attn_bias)

    assert out.shape == x.shape


def test_global_self_attention_block_rejects_all_masked_key_padding_row() -> None:
    block = GlobalSelfAttentionBlock(embed_dim=8, num_heads=2)
    x = torch.randn(2, 5, 8)
    mask = torch.tensor(
        [
            [False, False, False, False, True],
            [True, True, True, True, True],
        ]
    )

    with pytest.raises(ValueError, match="fully mask"):
        block(x, key_padding_mask=mask)


def test_global_self_attention_block_rejects_all_masked_attn_row() -> None:
    block = GlobalSelfAttentionBlock(embed_dim=8, num_heads=2)
    x = torch.randn(2, 5, 8)
    mask = torch.zeros(5, 5, dtype=torch.bool)
    mask[2, :] = True

    with pytest.raises(ValueError, match="fully mask"):
        block(x, attn_mask=mask)


def test_global_self_attention_block_rejects_combined_all_masked_row() -> None:
    block = GlobalSelfAttentionBlock(embed_dim=4, num_heads=1)
    x = torch.randn(1, 2, 4)
    key_padding_mask = torch.tensor([[False, True]])
    attn_mask = torch.tensor([[True, False], [False, False]])

    with pytest.raises(ValueError, match="combined"):
        block(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)


def test_global_self_attention_block_rejects_embed_dim_mismatch() -> None:
    block = GlobalSelfAttentionBlock(embed_dim=8, num_heads=2)

    with pytest.raises(ValueError, match="embed_dim"):
        block(torch.randn(2, 5, 7))


def test_global_self_attention_block_rejects_bad_head_factor() -> None:
    with pytest.raises(ValueError, match="divisible"):
        GlobalSelfAttentionBlock(embed_dim=10, num_heads=3)


def test_global_self_attention_block_rejects_non_floating_input() -> None:
    block = GlobalSelfAttentionBlock(embed_dim=8, num_heads=2)

    with pytest.raises(TypeError, match="floating"):
        block(torch.ones(2, 5, 8, dtype=torch.int64))


def test_global_self_attention_block_rejects_non_3d_input() -> None:
    block = GlobalSelfAttentionBlock(embed_dim=8, num_heads=2)

    with pytest.raises(ValueError, match="ndim"):
        block(torch.randn(5, 8))


def test_global_self_attention_block_rejects_bad_masks() -> None:
    block = GlobalSelfAttentionBlock(embed_dim=8, num_heads=2)
    x = torch.randn(2, 5, 8)

    with pytest.raises(ValueError, match="shape"):
        block(x, key_padding_mask=torch.zeros(2, 4, dtype=torch.bool))

    with pytest.raises(TypeError, match="bool or floating"):
        block(x, key_padding_mask=torch.zeros(2, 5, dtype=torch.int64))

    with pytest.raises(ValueError, match="2D"):
        block(x, attn_mask=torch.zeros(4, 5, dtype=torch.bool))

    with pytest.raises(ValueError, match="3D"):
        block(x, attn_mask=torch.zeros(2, 5, 5, dtype=torch.bool))

    with pytest.raises(TypeError, match="bool or floating"):
        block(x, attn_mask=torch.zeros(5, 5, dtype=torch.int64))


def test_global_self_attention_block_rejects_nan_dropout() -> None:
    with pytest.raises(ValueError, match="dropout"):
        GlobalSelfAttentionBlock(embed_dim=8, num_heads=2, dropout=float("nan"))


def test_global_self_attention_block_allows_fp32_additive_mask_for_low_precision_input() -> None:
    if not torch.cuda.is_available():
        pytest.skip("low-precision MultiheadAttention test requires CUDA")

    block = GlobalSelfAttentionBlock(embed_dim=8, num_heads=2).cuda().to(torch.float16)
    x = torch.randn(2, 5, 8, device="cuda", dtype=torch.float16)
    attn_mask = torch.zeros(5, 5, device="cuda", dtype=torch.float32)
    out = block(x, attn_mask=attn_mask)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_global_self_attention_block_rejects_float_mask_dtype_mismatch() -> None:
    block = GlobalSelfAttentionBlock(embed_dim=8, num_heads=2)
    x = torch.randn(2, 5, 8, dtype=torch.float32)

    with pytest.raises(ValueError, match="same dtype"):
        block(x, attn_mask=torch.zeros(5, 5, dtype=torch.float64))

    with pytest.raises(ValueError, match="same dtype"):
        block(x, key_padding_mask=torch.zeros(2, 5, dtype=torch.float64))


def test_global_self_attention_block_backpropagates() -> None:
    block = GlobalSelfAttentionBlock(embed_dim=8, num_heads=2)
    x = torch.randn(2, 5, 8, requires_grad=True)

    loss = block(x).sum()
    loss.backward()

    assert x.grad is not None
    assert any(param.grad is not None for param in block.parameters())


def test_global_self_attention_block_validates_constructor_arguments() -> None:
    with pytest.raises(TypeError, match="embed_dim"):
        GlobalSelfAttentionBlock(embed_dim=True, num_heads=2)

    with pytest.raises(TypeError, match="num_heads"):
        GlobalSelfAttentionBlock(embed_dim=8, num_heads=False)

    with pytest.raises(ValueError, match="positive"):
        GlobalSelfAttentionBlock(embed_dim=0, num_heads=2)
