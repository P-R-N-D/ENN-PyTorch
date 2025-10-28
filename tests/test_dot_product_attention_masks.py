from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stnet.utils.optimization import DotProductAttention


def _manual_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    b, h, l, _ = q.shape
    s = k.shape[2]
    if mask.dim() < 2:
        raise AssertionError("mask must have at least 2 dims")
    if mask.shape[-2:] != (l, s):
        raise AssertionError("mask trailing dims do not match sequence lengths")
    if mask.dim() == 2:
        mask = mask.view(1, 1, l, s)
    else:
        mask = mask.reshape(b, -1, l, s)
    if mask.shape[1] == 1:
        mask = mask.expand(b, h, l, s)
    elif mask.shape[1] != h:
        raise AssertionError("mask head dimension incompatible with query heads")
    if mask.dtype is torch.bool:
        finfo = torch.finfo(q.dtype)
        neg_inf = torch.full((), finfo.min, dtype=q.dtype, device=mask.device)
        zero = torch.zeros((), dtype=q.dtype, device=mask.device)
        mask = torch.where(mask, neg_inf, zero)
    else:
        mask = mask.to(dtype=q.dtype)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)


@pytest.mark.parametrize(
    "mask_builder",
    [
        lambda b, h, l, s, device: torch.full(
            (b, 1, 1, l, s), fill_value=0.0, device=device
        ).index_fill(-1, torch.tensor([s - 1], device=device), float("-inf")),
        lambda b, h, l, s, device: torch.zeros(
            (b, 1, h, 1, l, s), dtype=torch.bool, device=device
        ).index_fill(-2, torch.tensor([l - 1], device=device), True),
    ],
)
@torch.no_grad()
def test_sdpa_mask_supports_high_rank(mask_builder):
    device = torch.device("cpu")
    b, h, l, s, d = 2, 3, 4, 5, 7
    q = torch.randn(b, h, l, d, device=device)
    k = torch.randn(b, h, s, d, device=device)
    v = torch.randn(b, h, s, d, device=device)
    mask = mask_builder(b, h, l, s, device)

    attn = DotProductAttention(num_heads=h, head_dim=d)
    attn.eval()
    out = attn(q, k, v, attn_mask=mask)
    ref = _manual_sdpa(q, k, v, mask.to(dtype=q.dtype) if mask.dtype is not torch.bool else mask)

    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)
