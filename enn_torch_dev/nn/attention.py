from __future__ import annotations

from numbers import Real

import torch
from torch import Tensor, nn


def _validate_positive_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return value


def _validate_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool.")
    return value


def _validate_dropout(value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError("dropout must be a real number.")
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError("dropout must be between 0.0 and 1.0.")
    return value


def _mask_has_all_blocked_rows(mask: Tensor) -> bool:
    if mask.dtype == torch.bool:
        return bool(mask.all(dim=-1).any().item())
    if torch.is_floating_point(mask):
        return bool(torch.isneginf(mask).all(dim=-1).any().item())
    return False


class GlobalSelfAttentionBlock(nn.Module):
    """
    Encoder-style full self-attention block for global context modeling.

    Inputs and outputs use ``(B, N, C)`` layout. The block is intentionally
    non-causal and self-attention only. ``attn_mask`` may still be supplied by
    callers for custom masking, but this module does not auto-generate causal
    masks and does not implement cross-attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        bias: bool = True,
        residual: bool = True,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = _validate_positive_int(embed_dim, "embed_dim")
        self.num_heads = _validate_positive_int(num_heads, "num_heads")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.dropout = _validate_dropout(dropout)
        self.bias = _validate_bool(bias, "bias")
        self.residual = _validate_bool(residual, "residual")
        self.use_norm = _validate_bool(norm, "norm")

        self.norm = nn.LayerNorm(self.embed_dim) if self.use_norm else nn.Identity()
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            bias=self.bias,
            batch_first=True,
        )

    def _validate_input(self, x: Tensor) -> None:
        if not isinstance(x, Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)!r}")
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, N, C), got ndim={x.ndim}.")
        if int(x.shape[-1]) != self.embed_dim:
            raise ValueError(
                f"x last dimension must match embed_dim: {int(x.shape[-1])} != {self.embed_dim}"
            )
        if not torch.is_floating_point(x):
            raise TypeError("x must use a floating dtype.")

    def _validate_key_padding_mask(
        self,
        key_padding_mask: Tensor | None,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if key_padding_mask is None:
            return
        if not isinstance(key_padding_mask, Tensor):
            raise TypeError(
                f"key_padding_mask must be a torch.Tensor, got {type(key_padding_mask)!r}"
            )
        if tuple(key_padding_mask.shape) != (batch_size, seq_len):
            raise ValueError(
                "key_padding_mask must have shape (B, N): "
                f"{tuple(key_padding_mask.shape)} != {(batch_size, seq_len)}"
            )
        if key_padding_mask.device != device:
            raise ValueError("key_padding_mask must be on the same device as x.")
        if key_padding_mask.dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
            raise TypeError("key_padding_mask must use bool or floating dtype.")
        if torch.is_floating_point(key_padding_mask) and key_padding_mask.dtype != dtype:
            raise ValueError("floating key_padding_mask must have the same dtype as x.")
        if _mask_has_all_blocked_rows(key_padding_mask):
            raise ValueError("key_padding_mask must not fully mask any batch row.")

    def _validate_attn_mask(
        self,
        attn_mask: Tensor | None,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if attn_mask is None:
            return
        if not isinstance(attn_mask, Tensor):
            raise TypeError(f"attn_mask must be a torch.Tensor, got {type(attn_mask)!r}")
        if attn_mask.device != device:
            raise ValueError("attn_mask must be on the same device as x.")
        if attn_mask.dtype != torch.bool and not torch.is_floating_point(attn_mask):
            raise TypeError("attn_mask must use bool or floating dtype.")
        if torch.is_floating_point(attn_mask) and attn_mask.dtype != dtype:
            raise ValueError("floating attn_mask must have the same dtype as x.")

        if attn_mask.ndim == 2:
            expected = (seq_len, seq_len)
            if tuple(attn_mask.shape) != expected:
                raise ValueError(
                    f"2D attn_mask must have shape (N, N): {tuple(attn_mask.shape)} != {expected}"
                )
        elif attn_mask.ndim == 3:
            expected = (batch_size * self.num_heads, seq_len, seq_len)
            if tuple(attn_mask.shape) != expected:
                raise ValueError(
                    "3D attn_mask must have shape (B * num_heads, N, N): "
                    f"{tuple(attn_mask.shape)} != {expected}"
                )
        else:
            raise ValueError("attn_mask must be 2D or 3D.")

        if _mask_has_all_blocked_rows(attn_mask):
            raise ValueError("attn_mask must not fully mask any query row.")

    def forward(
        self,
        x: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        self._validate_input(x)
        batch_size = int(x.shape[0])
        seq_len = int(x.shape[1])
        self._validate_key_padding_mask(
            key_padding_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            device=x.device,

            dtype=x.dtype,
        )
        self._validate_attn_mask(
            attn_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            device=x.device,
            dtype=x.dtype,
        )

        y, _ = self.attention(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )

        if self.residual:
            y = y + x

        y = self.norm(y)
        return y
