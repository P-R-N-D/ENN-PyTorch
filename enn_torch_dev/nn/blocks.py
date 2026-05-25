from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .layers import ConvND


class Compressor(nn.Module):
    """
    Compress region-local channel-last features into a small number of
    context slots.

    Expected input shape:
        (B, R, *local_shape, D)

    Output shape:
        (B, R, K, D)

    The compressor is intentionally split into two stages:

      1. Optional structured local mixing through ``ConvND``.
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
        use_conv: bool = True,
        conv_kernel_size: int = 3,
        conv_bias: bool = True,
        conv_activation: str = "gelu",
        conv_local_ndim: int | None = None,
        conv_residual: bool = True,
        conv_residual_scale_init: float = 0.0,
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
        self.conv = (
            ConvND(
                self.dim,
                kernel_size=conv_kernel_size,
                enabled=True,
                bias=conv_bias,
                activation=conv_activation,
                local_ndim=conv_local_ndim,
                residual=conv_residual,
                residual_scale_init=conv_residual_scale_init,
            )
            if use_conv
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
        if isinstance(self.conv, nn.Identity):
            legacy_conv_prefixes = (
                f"{prefix}conv.conv1.",
                f"{prefix}conv.conv2.",
                f"{prefix}conv.conv3.",
            )
            legacy_conv_keys = [
                key
                for key in tuple(state_dict.keys())
                if key == f"{prefix}conv.residual_scale"
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

        h = self.conv(h)
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

    def forward_export(self, x: Tensor, local_mask: Tensor) -> Tensor:
        out = self._forward_impl(
            x, local_mask=local_mask, return_weights=False, force_dense=True
        )
        return out if isinstance(out, Tensor) else out[0]

    def forward_export_nomask(self, x: Tensor) -> Tensor:
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
    Packed global-context representation produced by ``Composer``.

    ``tokens`` is the dense compressed-context sequence passed to a generic
    global attention stage. ``token_mask`` carries token validity metadata.
    ``attn_bias`` is an optional additive key-side bias with shape
    ``(B, 1, 1, T)`` for attention implementations that accept logit bias.

    This container is attention-implementation agnostic. It does not
    implement or assume cross-attention, self-attention internals, or a
    specific attention backend.
    """

    tokens: Tensor
    token_mask: Tensor | None
    attn_bias: Tensor | None
    salience: Tensor | None = None
    score: Tensor | None = None
    original_shape: tuple[int, int, int, int] = (0, 0, 0, 0)
    input_dtype: torch.dtype | None = None
    token_dtype: torch.dtype | None = None
    bias_dtype: torch.dtype | None = None
    valid_token_count: Tensor | None = None
    has_valid_tokens: Tensor | None = None
    has_dummy_token: Tensor | None = None
    bias_kind: str = "none"


class Composer(nn.Module):
    """
    Recompose compressed regional context slots into a global context
    token sequence.

    Expected input shape:
        (B, R, K, D)

    Output:
        ``ContextSummary`` with ``tokens`` shaped ``(B, T, D)``,
        where ``T = R * K``.

    ``Composer`` sits between a local/regional compression stage and a
    generic global attention stage. It does not implement or assume
    cross-attention or any specific attention backend. The caller decides
    how to feed ``tokens``, ``token_mask``, and optional ``attn_bias`` into
    its attention implementation.

    Optional salience bias provides Tri-Attention-inspired soft importance
    control over compressed context keys. It suppresses low-salience tokens
    through an additive attention-logit bias; it does not perform hard
    routing or token pruning.

    ``context_mask`` is optional and should have shape ``(B, R, K)`` with
    True for valid compressed context slots. ``region_mask`` can be provided
    instead when all slots in a region share the same validity.
    """

    SUPPORTED_SALIENCE_MODES = {"none", "score", "soft_topk"}
    SUPPORTED_DTYPE_POLICIES = {"auto", "float32", "float64"}
    SUPPORTED_INT64_POLICIES = {"float32", "float64"}
    SUPPORTED_COMPLEX_MODES = {"abs", "real_imag"}
    SUPPORTED_OUTPUT_DTYPES = {"stable", "input", "amp"}
    SUPPORTED_BIAS_OUTPUT_DTYPES = {"stable", "token", "input", "amp"}
    SUPPORTED_NONFINITE_POLICIES = {"error", "warn", "sanitize", "ignore"}

    def __init__(
        self,
        dim: int,
        *args: Any,
        dtype_policy: str = "auto",
        int64_policy: str = "float32",
        complex_mode: str = "abs",
        token_output_dtype: str = "stable",
        bias_output_dtype: str = "stable",
        salience_mode: str = "none",
        salience_topk: int | float | None = None,
        salience_hidden_dim: int | None = None,
        salience_temperature: float = 1.0,
        salience_bias_scale: float = 1.0,
        detach_topk_threshold: bool = True,
        ensure_nonempty: bool = True,
        emit_mask_bias: bool = False,
        mask_bias_value: float | None = None,
        score_clip: float | None = 30.0,
        centered_clip: float | None = 30.0,
        bias_min: float | None = -80.0,
        nonfinite_policy: str = "error",
        salience_chunk_size: int | None = None,
        salience_chunk_threshold: int = 65536,
        return_score: bool = False,
        return_salience: bool = False,
        activation: str = "gelu",
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        _ = args

        self.dim = int(dim)
        self.dtype_policy = self._norm_choice(
            dtype_policy, self.SUPPORTED_DTYPE_POLICIES, "dtype_policy"
        )
        self.int64_policy = self._norm_choice(
            int64_policy, self.SUPPORTED_INT64_POLICIES, "int64_policy"
        )
        self.complex_mode = self._norm_choice(
            complex_mode, self.SUPPORTED_COMPLEX_MODES, "complex_mode"
        )
        self.token_output_dtype = self._norm_choice(
            token_output_dtype,
            self.SUPPORTED_OUTPUT_DTYPES,
            "token_output_dtype",
        )
        self.bias_output_dtype = self._norm_choice(
            bias_output_dtype,
            self.SUPPORTED_BIAS_OUTPUT_DTYPES,
            "bias_output_dtype",
        )
        self.salience_mode = self._norm_salience_str(salience_mode)
        self.salience_topk = salience_topk
        self.salience_temperature = float(salience_temperature)
        self.salience_bias_scale = float(salience_bias_scale)
        self.detach_topk_threshold = bool(detach_topk_threshold)
        self.ensure_nonempty = bool(ensure_nonempty)
        self.emit_mask_bias = bool(emit_mask_bias)
        self.mask_bias_value = mask_bias_value
        self.score_clip = None if score_clip is None else float(score_clip)
        self.centered_clip = None if centered_clip is None else float(centered_clip)
        self.bias_min = None if bias_min is None else float(bias_min)
        self.nonfinite_policy = self._norm_choice(
            nonfinite_policy,
            self.SUPPORTED_NONFINITE_POLICIES,
            "nonfinite_policy",
        )
        self.salience_chunk_size = self._norm_chunk_size(salience_chunk_size)
        self.salience_chunk_threshold = int(salience_chunk_threshold)
        self.return_score = bool(return_score)
        self.return_salience = bool(return_salience)
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
        if self.score_clip is not None and self.score_clip <= 0:
            raise ValueError(f"score_clip must be positive, got {score_clip}")
        if self.centered_clip is not None and self.centered_clip <= 0:
            raise ValueError(
                f"centered_clip must be positive, got {centered_clip}"
            )
        if self.salience_chunk_threshold <= 0:
            raise ValueError(
                "salience_chunk_threshold must be positive, got "
                f"{salience_chunk_threshold}"
            )

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

        self.complex_proj = (
            nn.Linear(self.dim * 2, self.dim)
            if self.complex_mode == "real_imag"
            else nn.Identity()
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
        if context.is_quantized:
            raise TypeError("Composer does not accept quantized tensors directly.")
        if context.ndim != 4:
            raise ValueError(
                "Composer expects shape (B, R, K, D). "
                f"Got shape={tuple(context.shape)}."
            )
        B, R, K, D = context.shape
        if self.complex_mode == "real_imag" and context.is_complex():
            expected_last = self.dim
        else:
            expected_last = self.dim
        if D != expected_last:
            raise ValueError(
                f"Expected last dim={expected_last}, got shape={tuple(context.shape)}"
            )
        if B <= 0 or R <= 0 or K <= 0 or D <= 0:
            raise ValueError(
                "Composer requires positive B, R, K, and D. "
                f"Got shape={tuple(context.shape)}."
            )

        input_dtype = context.dtype
        stable_dtype = self._stable_dtype(context)
        tokens = self._adapt_context(context, stable_dtype).reshape(B, R * K, self.dim)
        tokens = self._cast_token_output(tokens, input_dtype=input_dtype)

        mask = self._norm_mask(
            context,
            context_mask=context_mask,
            region_mask=region_mask,
        )
        token_mask = mask.reshape(B, R * K) if mask is not None else None

        token_mask, tokens, valid_token_count, has_valid_tokens, has_dummy_token = (
            self._ensure_nonempty_tokens(tokens, token_mask)
        )

        score: Tensor | None = None
        salience: Tensor | None = None
        attn_bias: Tensor | None = None
        bias_kind = "none"

        if self.salience_mode != "none":
            salience_result = self._compute_salience(
                tokens,
                token_mask=token_mask,
                stable_dtype=stable_dtype,
            )
            score = salience_result["score"]
            salience = salience_result["salience"]
            attn_bias = salience_result["bias"]
            bias_kind = "salience"

        if attn_bias is not None:
            if token_mask is not None:
                attn_bias = self._apply_mask_to_bias(
                    attn_bias,
                    token_mask,
                    stable_dtype=stable_dtype,
                    token_dtype=tokens.dtype,
                    input_dtype=input_dtype,
                )
                bias_kind = "salience+mask"
            else:
                attn_bias = self._finalize_bias(
                    attn_bias,
                    stable_dtype=stable_dtype,
                    token_dtype=tokens.dtype,
                    input_dtype=input_dtype,
                )
        elif token_mask is not None and self.emit_mask_bias:
            attn_bias = self._get_mask_bias(
                token_mask,
                stable_dtype=stable_dtype,
                token_dtype=tokens.dtype,
                input_dtype=input_dtype,
            )
            bias_kind = "mask"

        return ContextSummary(
            tokens=tokens,
            token_mask=token_mask,
            attn_bias=attn_bias,
            salience=salience,
            score=score,
            original_shape=(B, R, K, self.dim),
            input_dtype=input_dtype,
            token_dtype=tokens.dtype,
            bias_dtype=attn_bias.dtype if attn_bias is not None else None,
            valid_token_count=valid_token_count,
            has_valid_tokens=has_valid_tokens,
            has_dummy_token=has_dummy_token,
            bias_kind=bias_kind,
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
        if not isinstance(composition, ContextSummary):
            raise TypeError(
                "composition must be ContextSummary, got "
                f"{type(composition)!r}"
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
        if composition.has_dummy_token is not None and composition.has_dummy_token.any():
            restored = restored.clone()
            restored[composition.has_dummy_token, 0, 0, :] = 0
        return restored

    def _adapt_context(self, context: Tensor, stable_dtype: torch.dtype) -> Tensor:
        if context.is_complex():
            match self.complex_mode:
                case "abs":
                    return context.abs().to(dtype=stable_dtype)
                case "real_imag":
                    x = torch.view_as_real(context).flatten(-2)
                    x = x.to(dtype=self._module_real_dtype())
                    return self.complex_proj(x).to(dtype=stable_dtype)
                case _:
                    raise AssertionError(
                        f"Unreachable complex_mode: {self.complex_mode}"
                    )
        return context.to(dtype=stable_dtype)

    def _cast_token_output(
        self, tokens: Tensor, *, input_dtype: torch.dtype
    ) -> Tensor:
        mode = self.token_output_dtype
        if mode == "stable":
            return tokens
        if mode in {"input", "amp"} and self._is_real_floating_dtype(input_dtype):
            return tokens.to(dtype=input_dtype)
        return tokens

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

    def _ensure_nonempty_tokens(
        self, tokens: Tensor, token_mask: Tensor | None
    ) -> tuple[Tensor | None, Tensor, Tensor, Tensor, Tensor]:
        B, T = tokens.shape[:2]
        device = tokens.device
        if token_mask is None:
            valid_token_count = torch.full(
                (B,), int(T), device=device, dtype=torch.long
            )
            has_valid_tokens = torch.ones(B, device=device, dtype=torch.bool)
            has_dummy_token = torch.zeros(B, device=device, dtype=torch.bool)
            return (
                None,
                tokens,
                valid_token_count,
                has_valid_tokens,
                has_dummy_token,
            )

        valid_token_count = token_mask.sum(dim=1, dtype=torch.long)
        has_valid_tokens = valid_token_count > 0
        has_dummy_token = torch.zeros(B, device=device, dtype=torch.bool)
        if self.ensure_nonempty and not bool(has_valid_tokens.all().item()):
            bad = ~has_valid_tokens
            token_mask = token_mask.clone()
            tokens = tokens.clone()
            token_mask[bad, 0] = True
            tokens[bad, 0, :] = 0
            has_dummy_token[bad] = True
        return (
            token_mask,
            tokens,
            valid_token_count,
            has_valid_tokens,
            has_dummy_token,
        )

    def _compute_salience(
        self,
        tokens: Tensor,
        *,
        token_mask: Tensor | None,
        stable_dtype: torch.dtype,
    ) -> dict[str, Tensor | None]:
        T = int(tokens.shape[1])
        chunk_size = self._resolve_salience_chunk_size(T)
        use_chunked = chunk_size is not None and chunk_size < T
        if (
            use_chunked
            and self.salience_mode == "soft_topk"
            and not self.detach_topk_threshold
        ):
            warnings.warn(
                "Composer soft_topk with detach_topk_threshold=False uses dense "
                "salience computation to preserve threshold gradient semantics.",
                UserWarning,
                stacklevel=2,
            )
            use_chunked = False
        if use_chunked and self.salience_mode == "soft_topk" and self.training:
            warnings.warn(
                "Composer soft_topk uses dense salience computation during training "
                "to avoid recomputing dropout-dependent scores across chunk passes.",
                UserWarning,
                stacklevel=2,
            )
            use_chunked = False

        if not use_chunked:
            return self._compute_salience_dense(
                tokens, token_mask=token_mask, stable_dtype=stable_dtype
            )
        if self.salience_mode == "score":
            return self._compute_score_salience_chunked(
                tokens,
                token_mask=token_mask,
                stable_dtype=stable_dtype,
                chunk_size=chunk_size,
            )
        if self.salience_mode == "soft_topk":
            return self._compute_soft_topk_salience_chunked(
                tokens,
                token_mask=token_mask,
                stable_dtype=stable_dtype,
                chunk_size=chunk_size,
            )
        raise AssertionError(f"Unreachable salience mode: {self.salience_mode}")

    def _compute_salience_dense(
        self,
        tokens: Tensor,
        *,
        token_mask: Tensor | None,
        stable_dtype: torch.dtype,
    ) -> dict[str, Tensor | None]:
        score = self._score_tokens(tokens, stable_dtype=stable_dtype)
        if self.salience_mode == "score":
            bias, salience = self._score_to_bias_and_salience(score)
        elif self.salience_mode == "soft_topk":
            k = self._resolve_topk(score.shape[-1])
            if k is None:
                bias, salience = self._score_to_bias_and_salience(score)
            else:
                score_for_topk = self._mask_score_for_topk(
                    score, token_mask=token_mask
                )
                topk_value = torch.topk(score_for_topk, k=k, dim=-1).values[..., -1:]
                if self.detach_topk_threshold:
                    topk_value = topk_value.detach()
                centered = (score - topk_value) / self.salience_temperature
                bias, salience = self._centered_to_bias_and_salience(centered)
        else:
            raise AssertionError(f"Unreachable salience mode: {self.salience_mode}")

        if token_mask is not None and salience is not None:
            salience = salience.masked_fill(~token_mask, 0)
        return {
            "score": score if self.return_score else None,
            "salience": salience if self.return_salience else None,
            "bias": bias,
        }

    def _compute_score_salience_chunked(
        self,
        tokens: Tensor,
        *,
        token_mask: Tensor | None,
        stable_dtype: torch.dtype,
        chunk_size: int,
    ) -> dict[str, Tensor | None]:
        bias_chunks: list[Tensor] = []
        score_chunks: list[Tensor] = []
        salience_chunks: list[Tensor] = []
        T = int(tokens.shape[1])
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            score = self._score_tokens(tokens[:, start:end], stable_dtype=stable_dtype)
            bias, salience = self._score_to_bias_and_salience(score)
            bias_chunks.append(bias)
            if self.return_score:
                score_chunks.append(score)
            if self.return_salience:
                if token_mask is not None:
                    salience = salience.masked_fill(~token_mask[:, start:end], 0)
                salience_chunks.append(salience)
        return {
            "score": torch.cat(score_chunks, dim=1) if score_chunks else None,
            "salience": torch.cat(salience_chunks, dim=1) if salience_chunks else None,
            "bias": torch.cat(bias_chunks, dim=1),
        }

    def _compute_soft_topk_salience_chunked(
        self,
        tokens: Tensor,
        *,
        token_mask: Tensor | None,
        stable_dtype: torch.dtype,
        chunk_size: int,
    ) -> dict[str, Tensor | None]:
        T = int(tokens.shape[1])
        k = self._resolve_topk(T)
        if k is None:
            return self._compute_score_salience_chunked(
                tokens,
                token_mask=token_mask,
                stable_dtype=stable_dtype,
                chunk_size=chunk_size,
            )

        top_candidates: Tensor | None = None
        with torch.no_grad():
            for start in range(0, T, chunk_size):
                end = min(start + chunk_size, T)
                score = self._score_tokens(
                    tokens[:, start:end], stable_dtype=stable_dtype
                )
                chunk_mask = token_mask[:, start:end] if token_mask is not None else None
                score = self._mask_score_for_topk(score, token_mask=chunk_mask)
                take = min(k, int(score.shape[-1]))
                chunk_top = torch.topk(score, k=take, dim=-1).values
                top_candidates = (
                    chunk_top
                    if top_candidates is None
                    else torch.cat((top_candidates, chunk_top), dim=-1)
                )
                if int(top_candidates.shape[-1]) > k:
                    top_candidates = torch.topk(top_candidates, k=k, dim=-1).values
        if top_candidates is None:
            raise AssertionError("Composer requires at least one token.")
        threshold = torch.topk(top_candidates, k=min(k, top_candidates.shape[-1]), dim=-1).values[..., -1:]
        threshold = threshold.detach()

        bias_chunks: list[Tensor] = []
        score_chunks: list[Tensor] = []
        salience_chunks: list[Tensor] = []
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            score = self._score_tokens(tokens[:, start:end], stable_dtype=stable_dtype)
            centered = (score - threshold) / self.salience_temperature
            bias, salience = self._centered_to_bias_and_salience(centered)
            bias_chunks.append(bias)
            if self.return_score:
                score_chunks.append(score)
            if self.return_salience:
                if token_mask is not None:
                    salience = salience.masked_fill(~token_mask[:, start:end], 0)
                salience_chunks.append(salience)
        return {
            "score": torch.cat(score_chunks, dim=1) if score_chunks else None,
            "salience": torch.cat(salience_chunks, dim=1) if salience_chunks else None,
            "bias": torch.cat(bias_chunks, dim=1),
        }

    def _score_tokens(self, tokens: Tensor, *, stable_dtype: torch.dtype) -> Tensor:
        device_type = tokens.device.type
        module_dtype = self._module_real_dtype()
        with torch.autocast(device_type=device_type, enabled=False):
            x = tokens.to(dtype=module_dtype)
            score = self.salience_score(self.input_norm(x)).squeeze(-1)
            score = score.to(dtype=stable_dtype)
            if self.score_clip is not None:
                score = score.clamp(-self.score_clip, self.score_clip)
            return self._handle_nonfinite("score", score)

    def _score_to_bias_and_salience(self, score: Tensor) -> tuple[Tensor, Tensor]:
        with torch.autocast(device_type=score.device.type, enabled=False):
            bias = self.salience_bias_scale * F.logsigmoid(score)
            if self.bias_min is not None:
                bias = bias.clamp_min(self.bias_min)
            bias = self._handle_nonfinite("attn_bias", bias)
            salience = torch.sigmoid(score)
            salience = self._handle_nonfinite("salience", salience)
            return bias, salience

    def _centered_to_bias_and_salience(self, centered: Tensor) -> tuple[Tensor, Tensor]:
        with torch.autocast(device_type=centered.device.type, enabled=False):
            if self.centered_clip is not None:
                centered = centered.clamp(-self.centered_clip, self.centered_clip)
            bias = self.salience_bias_scale * F.logsigmoid(centered)
            if self.bias_min is not None:
                bias = bias.clamp_min(self.bias_min)
            bias = self._handle_nonfinite("attn_bias", bias)
            salience = torch.sigmoid(centered)
            salience = self._handle_nonfinite("salience", salience)
            return bias, salience

    def _mask_score_for_topk(
        self, score: Tensor, *, token_mask: Tensor | None
    ) -> Tensor:
        if token_mask is None:
            return score
        return score.masked_fill(
            ~token_mask, self._default_mask_bias_value(score.dtype)
        )

    def _apply_mask_to_bias(
        self,
        bias: Tensor,
        token_mask: Tensor,
        *,
        stable_dtype: torch.dtype,
        token_dtype: torch.dtype,
        input_dtype: torch.dtype,
    ) -> Tensor:
        with torch.autocast(device_type=bias.device.type, enabled=False):
            value = self._default_mask_bias_value(stable_dtype)
            out = bias.to(dtype=stable_dtype).masked_fill(~token_mask, value)
            return self._finalize_bias(
                out,
                stable_dtype=stable_dtype,
                token_dtype=token_dtype,
                input_dtype=input_dtype,
            )

    def _finalize_bias(
        self,
        bias: Tensor,
        *,
        stable_dtype: torch.dtype,
        token_dtype: torch.dtype,
        input_dtype: torch.dtype,
    ) -> Tensor:
        out = bias.to(dtype=stable_dtype)[:, None, None, :]
        return self._cast_bias_output(
            out, token_dtype=token_dtype, input_dtype=input_dtype
        )

    def _get_mask_bias(
        self,
        token_mask: Tensor,
        *,
        stable_dtype: torch.dtype,
        token_dtype: torch.dtype,
        input_dtype: torch.dtype,
    ) -> Tensor:
        with torch.autocast(device_type=token_mask.device.type, enabled=False):
            bias = torch.zeros(
                token_mask.shape, device=token_mask.device, dtype=stable_dtype
            )
            bias = bias.masked_fill(
                ~token_mask, self._default_mask_bias_value(stable_dtype)
            )
            bias = bias[:, None, None, :]
            return self._cast_bias_output(
                bias, token_dtype=token_dtype, input_dtype=input_dtype
            )

    def _cast_bias_output(
        self,
        bias: Tensor,
        *,
        token_dtype: torch.dtype,
        input_dtype: torch.dtype,
    ) -> Tensor:
        mode = self.bias_output_dtype
        if mode == "stable":
            return bias
        if mode in {"token", "amp"} and self._is_real_floating_dtype(token_dtype):
            return bias.to(dtype=token_dtype)
        if mode == "input" and self._is_real_floating_dtype(input_dtype):
            return bias.to(dtype=input_dtype)
        return bias

    def _default_mask_bias_value(self, dtype: torch.dtype) -> float:
        if self.mask_bias_value is not None:
            return float(self.mask_bias_value)
        if dtype in {torch.float16, torch.bfloat16}:
            return -1.0e4
        if dtype == torch.float64:
            return -1.0e12
        return -1.0e9

    def _handle_nonfinite(self, name: str, value: Tensor) -> Tensor:
        if self.nonfinite_policy == "ignore" or not value.is_floating_point():
            return value
        if value.numel() == 0:
            return value
        with torch.no_grad():
            is_finite = bool(torch.isfinite(value.detach()).all().item())
        if is_finite:
            return value
        message = f"Composer produced non-finite values in {name}."
        if self.nonfinite_policy == "error":
            raise FloatingPointError(message)
        if self.nonfinite_policy == "warn":
            warnings.warn(message, RuntimeWarning, stacklevel=3)
            return value
        if self.nonfinite_policy == "sanitize":
            fill = self._default_mask_bias_value(value.dtype) if name == "attn_bias" else 0.0
            return torch.nan_to_num(value, nan=fill, posinf=0.0, neginf=fill)
        raise AssertionError(f"Unreachable nonfinite_policy: {self.nonfinite_policy}")

    def _resolve_salience_chunk_size(self, total_tokens: int) -> int | None:
        if self.salience_chunk_size is not None:
            return min(self.salience_chunk_size, int(total_tokens))
        if int(total_tokens) >= self.salience_chunk_threshold:
            return min(self.salience_chunk_threshold, int(total_tokens))
        return None

    def _stable_dtype(self, x: Tensor) -> torch.dtype:
        if self.dtype_policy == "float64":
            return torch.float64
        if self.dtype_policy == "float32":
            return torch.float32
        if x.dtype in {torch.float64, torch.complex128}:
            return torch.float64
        if x.dtype == torch.int64 and self.int64_policy == "float64":
            return torch.float64
        return torch.float32

    def _module_real_dtype(self) -> torch.dtype:
        dtype = self.input_norm.weight.dtype
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError(f"Composer module dtype must be real floating, got {dtype}")
        return dtype

    @staticmethod
    def _is_real_floating_dtype(dtype: torch.dtype) -> bool:
        try:
            t = torch.empty((), dtype=dtype)
        except Exception:
            return False
        return bool(t.is_floating_point())

    @staticmethod
    def _norm_choice(value: str, supported: set[str], name: str) -> str:
        normalized = str(value).lower().strip().replace("-", "_")
        if normalized not in supported:
            allowed = ", ".join(sorted(supported))
            raise ValueError(f"{name} must be one of {allowed}. Got {value!r}")
        return normalized

    @staticmethod
    def _norm_chunk_size(chunk_size: int | None) -> int | None:
        if chunk_size is None:
            return None
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError(
                f"salience_chunk_size must be int | None, got {type(chunk_size)!r}"
            )
        if chunk_size <= 0:
            raise ValueError(
                f"salience_chunk_size must be positive, got {chunk_size}"
            )
        return int(chunk_size)

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
