from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .layers import LocalConvMixer


class Compressor(nn.Module):
    """
    Compress region-local channel-last features into a small number of
    context slots.

    Expected input shape:
        (B, R, *local_shape, D)

    Output shape:
        (B, R, K, D)

    The compressor is intentionally split into two stages:

      1. Optional structured local mixing through ``LocalConvMixer``.
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
        input_dim: int | None = None,
        integral_mode: str = "cast",
        complex_mode: str | None = None,
        output_dtype: torch.dtype | None = None,
        use_local_mixer: bool = True,
        local_kernel_size: int = 3,
        local_mixer_bias: bool = True,
        local_mixer_activation: str = "identity",
        local_ndim: int | None = None,
        local_mixer_residual: bool = True,
        local_mixer_residual_scale_init: float = 0.0,
        hidden_dim: int | None = None,
        score_hidden_dim: int | None = 128,
        activation: str = "gelu",
        dropout: float = 0.0,
        pool_chunk_size: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        _ = args

        self.dim = int(dim)
        self.num_slots = int(num_slots)
        self.eps = float(eps)
        self.pool_chunk_size = self._norm_chunk_size(pool_chunk_size)
        self.input_dim = self.dim if input_dim is None else int(input_dim)
        self.integral_mode = self._norm_integral_mode(integral_mode)
        (
            self.complex_mode,
            complex_projection_mode,
        ) = self._norm_complex_mode(complex_mode)
        self.output_dtype = self._norm_optional_real_dtype(output_dtype)

        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if self.num_slots <= 0:
            raise ValueError(
                f"num_slots must be positive, got {num_slots}"
            )
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")

        if hidden_dim is not None:
            score_hidden = int(hidden_dim)
        elif score_hidden_dim is None:
            score_hidden = self.dim
        else:
            score_hidden = int(score_hidden_dim)

        if score_hidden <= 0:
            raise ValueError(
                "score hidden dimension must be positive. "
                f"hidden_dim={hidden_dim}, score_hidden_dim={score_hidden_dim}"
            )

        self.real_input_proj = (
            nn.Identity()
            if self.input_dim == self.dim
            else nn.Linear(self.input_dim, self.dim)
        )
        match complex_projection_mode:
            case "real_imag":
                complex_adapted_dim = self.input_dim * 2
            case "abs" | "reject":
                complex_adapted_dim = self.input_dim
            case _:
                raise AssertionError(
                    "Unreachable complex projection mode: "
                    f"{complex_projection_mode}"
                )
        self.complex_input_proj = (
            nn.Identity()
            if complex_adapted_dim == self.dim
            else nn.Linear(complex_adapted_dim, self.dim)
        )

        self.input_norm = nn.LayerNorm(self.dim)
        self.local_mixer = (
            LocalConvMixer(
                self.dim,
                kernel_size=local_kernel_size,
                enabled=True,
                bias=local_mixer_bias,
                activation=local_mixer_activation,
                local_ndim=local_ndim,
                residual=local_mixer_residual,
                residual_scale_init=local_mixer_residual_scale_init,
            )
            if use_local_mixer
            else nn.Identity()
        )

        self.value_norm = nn.LayerNorm(self.dim)
        self.score_norm = nn.LayerNorm(self.dim)
        self.score = nn.Sequential(
            nn.Linear(self.dim, score_hidden),
            self._get_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(score_hidden, self.num_slots),
        )

        self.out = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
        )

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if isinstance(self.local_mixer, nn.Identity):
            legacy_conv_prefixes = (
                f"{prefix}conv.conv1.",
                f"{prefix}conv.conv2.",
                f"{prefix}conv.conv3.",
                f"{prefix}local_mixer.conv1.",
                f"{prefix}local_mixer.conv2.",
                f"{prefix}local_mixer.conv3.",
            )
            legacy_conv_keys = [
                key
                for key in tuple(state_dict.keys())
                if key == f"{prefix}conv.residual_scale"
                or key == f"{prefix}local_mixer.residual_scale"
                or key.startswith(legacy_conv_prefixes)
            ]
            for key in legacy_conv_keys:
                state_dict.pop(key, None)

        super()._load_from_state_dict(
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=strict,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )

    def forward(
        self,
        x: Tensor,
        *args: Any,
        local_mask: Tensor | None = None,
        return_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        return self._forward_impl(
            x, local_mask=local_mask, return_weights=return_weights
        )

    def _forward_impl(
        self,
        x: Tensor,
        *args: Any,
        local_mask: Tensor | None,
        return_weights: bool,
        force_dense: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        _ = args

        if not isinstance(x, Tensor):
            raise TypeError(
                f"Compressor expects Tensor, got {type(x)!r}"
            )
        if x.ndim < 3:
            raise ValueError(
                "Compressor expects shape (B, R, *local_shape, D). "
                f"Got shape={tuple(x.shape)}."
            )
        x = self._adapt_input(x)
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected last dim={self.dim}, got shape={tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise TypeError(
                "Compressor core requires real floating point input. "
                f"Got dtype={x.dtype}."
            )

        mask = self._norm_mask(local_mask, x)
        h = self.input_norm(x)

        if mask is not None:
            h = h.masked_fill(~mask.unsqueeze(-1), 0)

        h = self.local_mixer(h)
        h = self.value_norm(h)

        if mask is not None:
            h = h.masked_fill(~mask.unsqueeze(-1), 0)

        B, R = h.shape[:2]
        D = h.shape[-1]
        h_flat = h.reshape(B, R, -1, D)

        mask_flat = (
            mask.reshape(B, R, -1) if mask is not None else None
        )

        if h_flat.shape[2] == 0:
            z = h_flat.new_zeros((B, R, self.num_slots, D))
            z = self._cast_output(z)
            if return_weights:
                weights = h_flat.new_zeros((B, R, 0, self.num_slots))
                return z, weights
            return z

        use_chunked = (
            self.pool_chunk_size is not None
            and not force_dense
            and not return_weights
        )
        z, weights = self._pool(h_flat, mask_flat, chunked=use_chunked)
        z = self.out(z)

        if mask_flat is not None:
            valid_region = mask_flat.any(dim=2).view(B, R, 1, 1)
            z = torch.where(valid_region, z, torch.zeros_like(z))
        z = self._cast_output(z)

        if return_weights:
            return z, weights
        return z

    def forward_compat(self, x: Tensor, local_mask: Tensor) -> Tensor:
        out = self._forward_impl(
            x, local_mask=local_mask, return_weights=False, force_dense=True
        )
        return out if isinstance(out, Tensor) else out[0]

    def forward_compat_nomask(self, x: Tensor) -> Tensor:
        out = self._forward_impl(
            x, local_mask=None, return_weights=False, force_dense=True
        )
        return out if isinstance(out, Tensor) else out[0]

    def _norm_mask(
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

    def _pool(
        self,
        h_flat: Tensor,
        mask: Tensor | None,
        *,
        chunked: bool,
    ) -> tuple[Tensor, Tensor | None]:
        if chunked:
            return self._pool_chunked(h_flat, mask), None
        return self._pool_dense(h_flat, mask)

    def _pool_dense(
        self, h_flat: Tensor, mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        score = self.score(self.score_norm(h_flat))
        work_dtype = self._work_dtype(h_flat.dtype)
        weights = self._get_weight(score, mask)
        z = torch.matmul(
            weights.to(dtype=work_dtype).transpose(-1, -2),
            h_flat.to(dtype=work_dtype),
        )
        return z.to(dtype=h_flat.dtype), weights

    def _pool_chunked(self, h_flat: Tensor, mask: Tensor | None) -> Tensor:
        chunk_size = self.pool_chunk_size
        if chunk_size is None:
            z, _ = self._pool_dense(h_flat, mask)
            return z

        B, R, L, D = h_flat.shape
        K = self.num_slots
        if L == 0:
            return h_flat.new_zeros((B, R, K, D))

        score_max: Tensor | None = None
        chunks: list[tuple[Tensor, Tensor, Tensor | None]] = []

        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            h_chunk = h_flat[:, :, start:end, :]
            work_dtype = self._work_dtype(h_flat.dtype)
            score = self.score(self.score_norm(h_chunk)).to(dtype=work_dtype)
            if mask is not None:
                mask_chunk_base = mask[:, :, start:end]
                score = score.masked_fill(
                    ~mask_chunk_base.unsqueeze(-1), torch.finfo(score.dtype).min
                )
            else:
                mask_chunk_base = None
            chunks.append((h_chunk, score, mask_chunk_base))
            chunk_max = score.amax(dim=2)
            score_max = chunk_max if score_max is None else torch.maximum(score_max, chunk_max)

        if score_max is None:
            raise AssertionError("Compressor requires at least one local element.")

        if mask is not None:
            valid_region = mask.any(dim=2)
            score_max = torch.where(
                valid_region.unsqueeze(-1),
                score_max,
                torch.zeros_like(score_max),
            )

        work_dtype = self._work_dtype(h_flat.dtype)
        numer = h_flat.new_zeros((B, R, K, D), dtype=work_dtype)
        denom = h_flat.new_zeros((B, R, K), dtype=work_dtype)

        for h_chunk, score, mask_chunk_base in chunks:
            exp_score = torch.exp(score - score_max.unsqueeze(2))
            if mask_chunk_base is not None:
                exp_score = exp_score.masked_fill(~mask_chunk_base.unsqueeze(-1), 0.0)

            denom = denom + exp_score.sum(dim=2)
            numer = numer + torch.matmul(
                exp_score.transpose(-1, -2),
                h_chunk.to(dtype=work_dtype),
            )

        safe_eps = self._safe_eps(denom.dtype)
        z = numer / denom.clamp_min(safe_eps).unsqueeze(-1)

        if mask is not None:
            valid_region = mask.any(dim=2).view(B, R, 1, 1)
            z = torch.where(valid_region, z, torch.zeros_like(z))

        return z.to(dtype=h_flat.dtype)

    def _get_weight(
        self, score: Tensor, mask: Tensor | None
    ) -> Tensor:
        out_dtype = score.dtype
        score = score.to(dtype=self._work_dtype(score.dtype))

        if mask is None:
            return torch.softmax(score, dim=2).to(dtype=out_dtype)

        mask_expanded = mask.unsqueeze(-1)
        min_value = torch.finfo(score.dtype).min
        masked_score = score.masked_fill(~mask_expanded, min_value)

        weights = torch.softmax(masked_score, dim=2)
        weights = weights.masked_fill(~mask_expanded, 0.0)

        denom = weights.sum(dim=2, keepdim=True).clamp_min(
            self._safe_eps(weights.dtype)
        )
        weights = weights / denom

        valid = mask.any(dim=2, keepdim=True).unsqueeze(-1)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        return weights.to(dtype=out_dtype)

    def _safe_eps(self, dtype: torch.dtype) -> float:
        return max(float(self.eps), float(torch.finfo(dtype).tiny))

    def _adapt_input(self, x: Tensor) -> Tensor:
        if x.is_quantized:
            raise TypeError("Compressor does not accept quantized tensors directly.")
        target_dtype = self._module_real_dtype()
        if x.is_complex():
            if self.complex_mode == "reject":
                raise TypeError(
                    "Compressor received complex input, but complex_mode='reject'."
                )
            self._check_input_dim(x)
            match self.complex_mode:
                case "real_imag":
                    x = torch.view_as_real(x).flatten(-2)
                case "abs":
                    x = x.abs()
                case _:
                    raise AssertionError(
                        f"Unreachable complex_mode: {self.complex_mode}"
                    )
            return self.complex_input_proj(x.to(dtype=target_dtype))
        elif x.is_floating_point():
            self._check_input_dim(x)
            return self.real_input_proj(x.to(dtype=target_dtype))
        elif x.dtype == torch.bool:
            raise TypeError("Compressor does not accept bool tensors as numeric input.")
        elif self.integral_mode == "reject":
            raise TypeError(
                "Compressor received integral input, but integral_mode='reject'."
            )
        else:
            self._check_input_dim(x)
            return self.real_input_proj(x.to(dtype=target_dtype))

    def _check_input_dim(self, x: Tensor) -> None:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input last dim={self.input_dim}, "
                f"got shape={tuple(x.shape)}"
            )

    def _module_real_dtype(self) -> torch.dtype:
        dtype = self.input_norm.weight.dtype
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError(f"Compressor module dtype must be real floating, got {dtype}")
        return dtype

    def _cast_output(self, z: Tensor) -> Tensor:
        if self.output_dtype is None:
            return z
        return z.to(dtype=self.output_dtype)

    @staticmethod
    def _work_dtype(dtype: torch.dtype) -> torch.dtype:
        if dtype == torch.float64:
            return torch.float64
        if dtype in {torch.float16, torch.bfloat16, torch.float32}:
            return torch.float32
        raise TypeError(f"Unsupported Compressor core dtype: {dtype}")

    @staticmethod
    def _norm_optional_real_dtype(dtype: torch.dtype | None) -> torch.dtype | None:
        if dtype is None:
            return None
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"output_dtype must be torch.dtype | None, got {type(dtype)!r}")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError(f"output_dtype must be real floating, got {dtype}")
        return dtype

    @staticmethod
    def _norm_integral_mode(mode: str) -> str:
        normalized = str(mode).lower().strip()
        match normalized:
            case "cast" | "float" | "to_float":
                return "cast"
            case "reject" | "error" | "none":
                return "reject"
            case _:
                raise ValueError("integral_mode must be one of 'cast' or 'reject'.")

    @staticmethod
    def _norm_complex_mode(mode: str | None) -> tuple[str, str]:
        if mode is None:
            return "reject", "real_imag"

        normalized = str(mode).lower().strip().replace("-", "_")
        match normalized:
            case "real_imag" | "cartesian" | "ri":
                return "real_imag", "real_imag"
            case "abs" | "magnitude" | "mag":
                return "abs", "abs"
            case "reject" | "error" | "none":
                return "reject", "reject"
            case _:
                raise ValueError(
                    "complex_mode must be one of 'real_imag', 'abs', or 'reject'."
                )

    @staticmethod
    def _norm_chunk_size(chunk_size: int | None) -> int | None:
        if chunk_size is None:
            return None
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError(
                f"pool_chunk_size must be int | None, got {type(chunk_size)!r}"
            )
        if chunk_size <= 0:
            raise ValueError(f"pool_chunk_size must be positive, got {chunk_size}")
        return int(chunk_size)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
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
class ContextSummary:
    """
    Packed coarse context produced by ``Composer``.

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


class Composer(nn.Module):
    """
    Recompose compressed regional context into a coarse global context.

    Expected input shape:
        (B, R, K, D)

    Output:
        ``ContextSummary`` with ``tokens`` shaped ``(B, T, D)``,
        where ``T = R * K``.

    This block is intentionally placed between ``Compressor`` and the
    global attention layer. It owns coarse-context composition metadata and
    optional salience biasing:

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
        self.salience_mode = self._norm_salience_str(salience_mode)
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
            self._get_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        context: Tensor,
        *args: Any,
        context_mask: Tensor | None = None,
        region_mask: Tensor | None = None,
    ) -> ContextSummary:
        _ = args

        if not isinstance(context, Tensor):
            raise TypeError(
                "Composer expects Tensor, got "
                f"{type(context)!r}"
            )
        if context.ndim != 4:
            raise ValueError(
                "Composer expects shape (B, R, K, D). "
                f"Got shape={tuple(context.shape)}."
            )
        if context.shape[-1] != self.dim:
            raise ValueError(
                f"Expected last dim={self.dim}, got shape={tuple(context.shape)}"
            )
        if not context.is_floating_point():
            raise TypeError(
                "Composer requires floating point input. "
                f"Got dtype={context.dtype}."
            )

        B, R, K, D = context.shape
        mask = self._norm_mask(
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

            salience, attn_bias = self._get_salience_and_bias(
                score,
                token_mask=token_mask,
            )

        return ContextSummary(
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
        composition: ContextSummary,
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

    def _norm_mask(
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

    def _get_salience_and_bias(
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
            return salience, self._get_bias_from_salience(
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

            return salience, self._get_bias_from_salience(
                salience,
                token_mask=token_mask,
            )

        raise AssertionError(
            f"Unreachable salience mode: {self.salience_mode}"
        )

    def _get_bias_from_salience(
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
    def _norm_salience_str(cls, mode: str) -> str:
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
    def _get_activation(name: str) -> nn.Module:
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
