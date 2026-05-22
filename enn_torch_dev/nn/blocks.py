from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .layers import AutoConvND


class RegionCompressor(nn.Module):
    """
    Compress region-local channel-last features into a small number of
    context slots.

    Expected input shape:
        (B, R, *local_shape, D)

    Output shape:
        (B, R, K, D)

    The compressor is intentionally split into two stages:

      1. Optional structured local mixing through ``AutoConvND``.
         Conv1d/Conv2d/Conv3d is selected from the rank of
         ``local_shape``. Unsupported local ranks fall back to identity.

      2. Shape-agnostic gated pooling.
         Local axes are flattened and each learned slot softly pools over
         the local elements.

    ``local_mask`` is optional and should have shape
    ``(B, R, *local_shape)`` with True for valid local elements.
    """

    def __init__(
        self,
        dim: int,
        *args: Any,
        num_slots: int = 4,
        use_conv: bool = True,
        conv_kernel_size: int = 3,
        conv_bias: bool = True,
        conv_activation: str = "gelu",
        conv_residual: bool = True,
        conv_residual_scale_init: float = 1.0,
        hidden_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        _ = args

        self.dim = int(dim)
        self.num_slots = int(num_slots)
        self.eps = float(eps)

        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if self.num_slots <= 0:
            raise ValueError(
                f"num_slots must be positive, got {num_slots}"
            )
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        score_hidden = self.dim if hidden_dim is None else int(hidden_dim)
        if score_hidden <= 0:
            raise ValueError(
                f"hidden_dim must be positive, got {hidden_dim}"
            )

        self.input_norm = nn.LayerNorm(self.dim)
        self.conv = AutoConvND(
            self.dim,
            kernel_size=conv_kernel_size,
            enabled=use_conv,
            bias=conv_bias,
            activation=conv_activation,
            residual=conv_residual,
            residual_scale_init=conv_residual_scale_init,
        )

        self.score_norm = nn.LayerNorm(self.dim)
        self.score = nn.Sequential(
            nn.Linear(self.dim, score_hidden),
            self._make_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(score_hidden, self.num_slots),
        )

        self.out = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
        )

    def forward(
        self,
        x: Tensor,
        *args: Any,
        local_mask: Tensor | None = None,
        return_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        _ = args

        if not isinstance(x, Tensor):
            raise TypeError(
                f"RegionCompressor expects Tensor, got {type(x)!r}"
            )
        if x.ndim < 3:
            raise ValueError(
                "RegionCompressor expects shape (B, R, *local_shape, D). "
                f"Got shape={tuple(x.shape)}."
            )
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected last dim={self.dim}, got shape={tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise TypeError(
                "RegionCompressor requires floating point input. "
                f"Got dtype={x.dtype}."
            )

        mask = self._validate_mask(local_mask, x)
        h = self.input_norm(x)

        if mask is not None:
            h = h.masked_fill(~mask.unsqueeze(-1), 0)

        h = self.conv(h)

        if mask is not None:
            h = h.masked_fill(~mask.unsqueeze(-1), 0)

        B, R = h.shape[:2]
        D = h.shape[-1]
        h_flat = h.reshape(B, R, -1, D)

        mask_flat = (
            mask.reshape(B, R, -1) if mask is not None else None
        )

        score_in = self.score_norm(h_flat)
        score = self.score(score_in)
        weights = self._slot_weights(score, mask_flat)

        z = torch.einsum("brlk,brld->brkd", weights, h_flat)
        z = self.out(z)

        if mask_flat is not None:
            valid_region = mask_flat.any(dim=2).view(B, R, 1, 1)
            z = torch.where(valid_region, z, torch.zeros_like(z))

        if return_weights:
            return z, weights
        return z

    def _validate_mask(
        self, local_mask: Tensor | None, x: Tensor
    ) -> Tensor | None:
        if local_mask is None:
            return None
        if not isinstance(local_mask, Tensor):
            raise TypeError(
                f"local_mask must be Tensor | None, got {type(local_mask)!r}"
            )
        expected = x.shape[:-1]
        if tuple(local_mask.shape) != tuple(expected):
            raise ValueError(
                "local_mask shape must match x.shape[:-1]. "
                f"local_mask.shape={tuple(local_mask.shape)}, "
                f"x.shape[:-1]={tuple(expected)}"
            )
        return local_mask.to(device=x.device, dtype=torch.bool)

    def _slot_weights(
        self, score: Tensor, mask: Tensor | None
    ) -> Tensor:
        if mask is None:
            return torch.softmax(score, dim=2)

        mask_expanded = mask.unsqueeze(-1)
        min_value = torch.finfo(score.dtype).min
        masked_score = score.masked_fill(~mask_expanded, min_value)

        weights = torch.softmax(masked_score, dim=2)
        weights = weights.masked_fill(~mask_expanded, 0)

        denom = weights.sum(dim=2, keepdim=True).clamp_min(self.eps)
        return weights / denom

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        normalized = str(name).lower().strip()
        match normalized:
            case "gelu":
                return nn.GELU()
            case "silu" | "swish":
                return nn.SiLU()
            case "relu":
                return nn.ReLU()
            case "identity" | "none" | "linear":
                return nn.Identity()
            case _:
                raise ValueError(f"Unsupported activation: {name!r}")
