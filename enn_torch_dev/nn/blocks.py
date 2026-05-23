from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class GlobalContextComposition:
    """
    Packed coarse context produced by ``GlobalContextComposer``.

    ``tokens`` is the dense sequence passed to global attention.
    ``attn_bias`` is an optional key-side salience bias with shape
    ``(B, 1, 1, T)``. Attention implementations may add it to attention
    logits before softmax.
    """

    tokens: Tensor
    token_mask: Tensor | None
    attn_bias: Tensor | None
    salience: Tensor | None
    score: Tensor | None
    original_shape: tuple[int, int, int, int]


class GlobalContextComposer(nn.Module):
    """
    Recompose compressed regional context into a coarse global context.

    Expected input shape:
        (B, R, K, D)

    Output:
        ``GlobalContextComposition`` with ``tokens`` shaped ``(B, T, D)``,
        where ``T = R * K``.

    This block is intentionally placed between ``RegionCompressor`` and the
    global attention layer. It owns coarse-context composition metadata and
    optional TriAttention-like salience biasing:

      - no hard routing by default;
      - compressed tokens stay dense;
      - soft top-k salience suppresses low-importance keys and highlights
        high-importance keys through an attention-logit bias.

    ``context_mask`` is optional and should have shape ``(B, R, K)`` with
    True for valid compressed context slots. ``region_mask`` can be provided
    instead when all slots in a region share the same validity.
    """

    SUPPORTED_SALIENCE_MODES = {"none", "score", "soft_topk"}

    def __init__(
        self,
        dim: int,
        *args: Any,
        salience_mode: str = "none",
        salience_topk: int | float | None = None,
        salience_hidden_dim: int | None = None,
        salience_temperature: float = 1.0,
        salience_bias_scale: float = 1.0,
        detach_topk_threshold: bool = True,
        activation: str = "gelu",
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        _ = args

        self.dim = int(dim)
        self.salience_mode = self._normalize_salience_mode(salience_mode)
        self.salience_topk = salience_topk
        self.salience_temperature = float(salience_temperature)
        self.salience_bias_scale = float(salience_bias_scale)
        self.detach_topk_threshold = bool(detach_topk_threshold)
        self.eps = float(eps)

        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if self.salience_temperature <= 0:
            raise ValueError(
                "salience_temperature must be positive, got "
                f"{salience_temperature}"
            )
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        hidden = (
            self.dim
            if salience_hidden_dim is None
            else int(salience_hidden_dim)
        )
        if hidden <= 0:
            raise ValueError(
                "salience_hidden_dim must be positive, got "
                f"{salience_hidden_dim}"
            )

        self.input_norm = nn.LayerNorm(self.dim)
        self.salience_score = nn.Sequential(
            nn.Linear(self.dim, hidden),
            self._make_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        context: Tensor,
        *args: Any,
        context_mask: Tensor | None = None,
        region_mask: Tensor | None = None,
    ) -> GlobalContextComposition:
        _ = args

        if not isinstance(context, Tensor):
            raise TypeError(
                "GlobalContextComposer expects Tensor, got "
                f"{type(context)!r}"
            )
        if context.ndim != 4:
            raise ValueError(
                "GlobalContextComposer expects shape (B, R, K, D). "
                f"Got shape={tuple(context.shape)}."
            )
        if context.shape[-1] != self.dim:
            raise ValueError(
                f"Expected last dim={self.dim}, got shape={tuple(context.shape)}"
            )
        if not context.is_floating_point():
            raise TypeError(
                "GlobalContextComposer requires floating point input. "
                f"Got dtype={context.dtype}."
            )

        B, R, K, D = context.shape
        mask = self._validate_context_mask(
            context,
            context_mask=context_mask,
            region_mask=region_mask,
        )

        tokens = context.reshape(B, R * K, D)
        token_mask = mask.reshape(B, R * K) if mask is not None else None

        score: Tensor | None = None
        salience: Tensor | None = None
        attn_bias: Tensor | None = None

        if self.salience_mode != "none":
            score = self.salience_score(self.input_norm(tokens)).squeeze(-1)
            if token_mask is not None:
                min_value = torch.finfo(score.dtype).min
                score = score.masked_fill(~token_mask, min_value)

            salience, attn_bias = self._salience_and_bias(
                score,
                token_mask=token_mask,
            )

        return GlobalContextComposition(
            tokens=tokens,
            token_mask=token_mask,
            attn_bias=attn_bias,
            salience=salience,
            score=score,
            original_shape=(B, R, K, D),
        )

    def restore(
        self,
        tokens: Tensor,
        composition: GlobalContextComposition,
        *args: Any,
    ) -> Tensor:
        _ = args

        if not isinstance(tokens, Tensor):
            raise TypeError(
                f"restore expects Tensor, got {type(tokens)!r}"
            )

        B, R, K, D = composition.original_shape
        expected = (B, R * K, D)
        if tuple(tokens.shape) != expected:
            raise ValueError(
                f"Expected tokens.shape={expected}, got {tuple(tokens.shape)}"
            )

        restored = tokens.reshape(B, R, K, D)
        if composition.token_mask is not None:
            mask = composition.token_mask.reshape(B, R, K, 1)
            restored = restored.masked_fill(~mask, 0)
        return restored

    def _validate_context_mask(
        self,
        context: Tensor,
        *args: Any,
        context_mask: Tensor | None,
        region_mask: Tensor | None,
    ) -> Tensor | None:
        _ = args

        if context_mask is not None and region_mask is not None:
            raise ValueError(
                "Provide either context_mask or region_mask, not both."
            )

        B, R, K, _ = context.shape

        if context_mask is not None:
            if not isinstance(context_mask, Tensor):
                raise TypeError(
                    "context_mask must be Tensor | None, got "
                    f"{type(context_mask)!r}"
                )
            if tuple(context_mask.shape) != (B, R, K):
                raise ValueError(
                    "context_mask shape must be (B, R, K). "
                    f"context_mask.shape={tuple(context_mask.shape)}, "
                    f"expected={(B, R, K)}"
                )
            return context_mask.to(device=context.device, dtype=torch.bool)

        if region_mask is not None:
            if not isinstance(region_mask, Tensor):
                raise TypeError(
                    "region_mask must be Tensor | None, got "
                    f"{type(region_mask)!r}"
                )
            if tuple(region_mask.shape) != (B, R):
                raise ValueError(
                    "region_mask shape must be (B, R). "
                    f"region_mask.shape={tuple(region_mask.shape)}, "
                    f"expected={(B, R)}"
                )
            return (
                region_mask.to(device=context.device, dtype=torch.bool)
                .unsqueeze(-1)
                .expand(B, R, K)
            )

        return None

    def _salience_and_bias(
        self,
        score: Tensor,
        *args: Any,
        token_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        _ = args

        if self.salience_mode == "score":
            salience = torch.sigmoid(score)
            if token_mask is not None:
                salience = salience.masked_fill(~token_mask, 0)
            return salience, self._bias_from_salience(
                salience,
                token_mask=token_mask,
            )

        if self.salience_mode == "soft_topk":
            k = self._resolve_topk(score.shape[-1])
            if k is None:
                salience = torch.sigmoid(score)
            else:
                topk_value = torch.topk(score, k=k, dim=-1).values[..., -1:]
                if self.detach_topk_threshold:
                    topk_value = topk_value.detach()
                centered = (score - topk_value) / self.salience_temperature
                salience = torch.sigmoid(centered)

            if token_mask is not None:
                salience = salience.masked_fill(~token_mask, 0)

            return salience, self._bias_from_salience(
                salience,
                token_mask=token_mask,
            )

        raise AssertionError(
            f"Unreachable salience mode: {self.salience_mode}"
        )

    def _bias_from_salience(
        self,
        salience: Tensor,
        *args: Any,
        token_mask: Tensor | None,
    ) -> Tensor:
        _ = args

        safe = salience.clamp_min(self.eps)
        bias = self.salience_bias_scale * torch.log(safe)

        if token_mask is not None:
            min_value = torch.finfo(bias.dtype).min
            bias = bias.masked_fill(~token_mask, min_value)

        return bias[:, None, None, :]

    def _resolve_topk(self, total_tokens: int) -> int | None:
        if self.salience_topk is None:
            return None

        if isinstance(self.salience_topk, bool):
            raise TypeError(
                "salience_topk must be int | float | None, not bool"
            )

        if isinstance(self.salience_topk, int):
            k = int(self.salience_topk)
        elif isinstance(self.salience_topk, float):
            ratio = float(self.salience_topk)
            if not (0.0 < ratio <= 1.0):
                raise ValueError(
                    "float salience_topk must be in (0, 1]. "
                    f"Got {self.salience_topk}"
                )
            k = int(torch.ceil(torch.tensor(total_tokens * ratio)).item())
        else:
            raise TypeError(
                "salience_topk must be int | float | None, got "
                f"{type(self.salience_topk)!r}"
            )

        if k <= 0:
            raise ValueError(f"salience_topk must be positive, got {k}")

        return min(k, int(total_tokens))

    @classmethod
    def _normalize_salience_mode(cls, mode: str) -> str:
        normalized = str(mode).lower().strip()
        match normalized:
            case "none" | "off" | "disabled" | "identity":
                return "none"
            case "score" | "sigmoid" | "gate":
                return "score"
            case "soft_topk" | "soft-topk" | "topk" | "triattention":
                return "soft_topk"
            case _:
                supported = ", ".join(sorted(cls.SUPPORTED_SALIENCE_MODES))
                raise ValueError(
                    f"Unsupported salience_mode: {mode!r}. "
                    f"Supported modes: {supported}"
                )

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
