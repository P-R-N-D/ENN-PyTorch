from __future__ import annotations

import pytest
import torch

from enn_torch_dev.nn.blocks import Composer, ContextSummary


def test_composer_packs_and_restores_global_context_tokens():
    x = torch.randn(2, 3, 4, 5)
    module = Composer(5)

    summary = module(x)
    restored = module.restore(summary.tokens, summary)

    assert isinstance(summary, ContextSummary)
    assert summary.tokens.shape == (2, 12, 5)
    assert summary.token_mask is None
    assert summary.attn_bias is None
    assert summary.original_shape == (2, 3, 4, 5)
    assert torch.allclose(restored, x.float())


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (torch.bool, torch.float32),
        (torch.int32, torch.float32),
        (torch.int64, torch.float32),
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
        (torch.complex64, torch.float32),
        (torch.complex128, torch.float64),
    ],
)
def test_composer_accepts_all_common_input_dtypes(dtype, expected):
    if dtype == torch.bool:
        x = torch.randint(0, 2, (2, 3, 4, 5), dtype=torch.bool)
    elif dtype.is_complex:
        base_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        real = torch.randn(2, 3, 4, 5, dtype=base_dtype)
        imag = torch.randn(2, 3, 4, 5, dtype=base_dtype)
        x = torch.complex(real, imag).to(dtype=dtype)
    elif dtype.is_floating_point:
        x = torch.randn(2, 3, 4, 5, dtype=dtype)
    else:
        x = torch.randint(0, 10, (2, 3, 4, 5), dtype=dtype)

    summary = Composer(5)(x)

    assert summary.input_dtype == dtype
    assert summary.tokens.dtype == expected
    assert summary.token_dtype == expected
    assert summary.tokens.shape == (2, 12, 5)
    assert torch.isfinite(summary.tokens.float()).all()


def test_composer_int64_policy_can_preserve_integer_precision_budget():
    x = torch.randint(0, 10, (2, 3, 4, 5), dtype=torch.int64)
    module = Composer(5, int64_policy="float64")

    summary = module(x)

    assert summary.tokens.dtype == torch.float64


def test_composer_context_mask_and_region_mask_are_exclusive():
    x = torch.randn(2, 3, 4, 5)
    context_mask = torch.ones(2, 3, 4, dtype=torch.bool)
    region_mask = torch.ones(2, 3, dtype=torch.bool)
    module = Composer(5)

    with pytest.raises(ValueError, match="either context_mask or region_mask"):
        module(x, context_mask=context_mask, region_mask=region_mask)


def test_composer_region_mask_expands_to_token_mask():
    x = torch.randn(2, 3, 4, 5)
    region_mask = torch.tensor(
        [[True, False, True], [False, True, True]], dtype=torch.bool
    )

    summary = Composer(5)(x, region_mask=region_mask)

    assert summary.token_mask is not None
    assert summary.token_mask.shape == (2, 12)
    assert summary.valid_token_count.tolist() == [8, 8]
    assert summary.has_valid_tokens.tolist() == [True, True]


def test_composer_all_invalid_mask_gets_dummy_token_without_nan():
    x = torch.randn(2, 3, 4, 5)
    mask = torch.ones(2, 3, 4, dtype=torch.bool)
    mask[0] = False
    module = Composer(5, salience_mode="score", emit_mask_bias=True)

    summary = module(x, context_mask=mask)

    assert summary.has_valid_tokens.tolist() == [False, True]
    assert summary.has_dummy_token.tolist() == [True, False]
    assert summary.valid_token_count.tolist() == [0, 12]
    assert bool(summary.token_mask[0, 0])
    assert torch.allclose(summary.tokens[0, 0], torch.zeros_like(summary.tokens[0, 0]))
    assert summary.attn_bias.shape == (2, 1, 1, 12)
    assert torch.isfinite(summary.attn_bias).all()


def test_composer_mask_bias_can_be_emitted_without_salience():
    x = torch.randn(2, 3, 4, 5)
    mask = torch.ones(2, 3, 4, dtype=torch.bool)
    mask[:, 1] = False
    module = Composer(5, emit_mask_bias=True)

    summary = module(x, context_mask=mask)

    assert summary.bias_kind == "mask"
    assert summary.attn_bias.shape == (2, 1, 1, 12)
    assert torch.isfinite(summary.attn_bias).all()
    assert torch.all(summary.attn_bias[:, :, :, 4:8] < 0)


@pytest.mark.parametrize("mode", ["score", "soft_topk"])
def test_composer_salience_bias_is_finite_and_metadata_opt_in(mode):
    x = torch.randn(2, 3, 4, 5)
    module = Composer(
        5,
        salience_mode=mode,
        salience_topk=0.5 if mode == "soft_topk" else None,
    )

    summary = module(x)

    assert summary.attn_bias.shape == (2, 1, 1, 12)
    assert summary.bias_kind == "salience"
    assert torch.isfinite(summary.attn_bias).all()
    assert summary.score is None
    assert summary.salience is None


def test_composer_score_and_salience_return_are_opt_in():
    x = torch.randn(2, 3, 4, 5)
    module = Composer(
        5,
        salience_mode="score",
        return_score=True,
        return_salience=True,
    )

    summary = module(x)

    assert summary.score.shape == (2, 12)
    assert summary.salience.shape == (2, 12)
    assert torch.isfinite(summary.score).all()
    assert torch.isfinite(summary.salience).all()


def test_composer_chunked_score_matches_dense_score_eval():
    x = torch.randn(2, 3, 17, 5)
    dense = Composer(5, salience_mode="score")
    chunked = Composer(5, salience_mode="score", salience_chunk_size=5)
    chunked.load_state_dict(dense.state_dict())
    dense.eval()
    chunked.eval()

    dense_out = dense(x).attn_bias
    chunked_out = chunked(x).attn_bias

    assert torch.allclose(chunked_out, dense_out, atol=1e-6, rtol=1e-6)


def test_composer_chunked_soft_topk_matches_dense_eval():
    x = torch.randn(2, 3, 17, 5)
    dense = Composer(5, salience_mode="soft_topk", salience_topk=0.25)
    chunked = Composer(
        5,
        salience_mode="soft_topk",
        salience_topk=0.25,
        salience_chunk_size=5,
    )
    chunked.load_state_dict(dense.state_dict())
    dense.eval()
    chunked.eval()

    dense_out = dense(x).attn_bias
    chunked_out = chunked(x).attn_bias

    assert torch.allclose(chunked_out, dense_out, atol=1e-6, rtol=1e-6)


def test_composer_restore_zeroes_masked_and_dummy_slots():
    x = torch.randn(2, 3, 4, 5)
    mask = torch.ones(2, 3, 4, dtype=torch.bool)
    mask[0] = False
    mask[1, 1] = False
    module = Composer(5)

    summary = module(x, context_mask=mask)
    restored = module.restore(summary.tokens, summary)

    assert restored.shape == (2, 3, 4, 5)
    assert torch.allclose(restored[0, 0, 0], torch.zeros_like(restored[0, 0, 0]))
    assert torch.allclose(restored[0], torch.zeros_like(restored[0]))
    assert torch.allclose(restored[1, 1], torch.zeros_like(restored[1, 1]))


def test_composer_preserves_float64_stable_bias_when_module_is_float64():
    x = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    module = Composer(5, salience_mode="score").to(dtype=torch.float64)

    summary = module(x)

    assert summary.tokens.dtype == torch.float64
    assert summary.attn_bias.dtype == torch.float64
    assert torch.isfinite(summary.attn_bias).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP not available")
def test_composer_cuda_amp_bias_remains_finite():
    x = torch.randn(2, 3, 7, 5, device="cuda", dtype=torch.float16)
    module = Composer(5, salience_mode="score").cuda()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        summary = module(x)

    assert summary.attn_bias.dtype == torch.float32
    assert torch.isfinite(summary.attn_bias).all()
