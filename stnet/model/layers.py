# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
from torch import nn

from ..utils.optimization import DotProductAttention, MultiScaleRetention


try:
    # Prefer torch.compiler.disable (PyTorch ≥2.5)
    _disable_torch_compile = torch.compiler.disable  # type: ignore[attr-defined]
except Exception:
    try:
        # Fallback for PyTorch 2.0–2.4
        import torch._dynamo as _dynamo  # type: ignore

        _disable_torch_compile = _dynamo.disable  # type: ignore[attr-defined]
    except Exception:

        def _disable_torch_compile(fn):  # type: ignore
            return fn


def _disable_module_torch_compile(module: nn.Module) -> nn.Module:
    forward = getattr(module, "forward", None)
    if callable(forward):
        module.forward = _disable_torch_compile(forward)  # type: ignore[assignment]
    return module


class _Float32NormMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._restore_float32()

    def _restore_float32(self) -> None:
        with torch.no_grad():
            for param in self.parameters(recurse=False):
                if torch.is_floating_point(param.data) and param.data.dtype is not torch.float32:
                    param.data = param.data.to(param.data.device, dtype=torch.float32)
            for buffer in self.buffers(recurse=False):
                if torch.is_floating_point(buffer) and buffer.dtype is not torch.float32:
                    buffer.copy_(buffer.to(buffer.device, dtype=torch.float32))

    def _cast_input(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if not torch.is_floating_point(x):
            return x, False
        if x.dtype == torch.float32:
            return x, False
        return x.to(torch.float32), True

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        forward_fn = super().forward  # type: ignore[attr-defined]
        cast_x, should_cast_back = self._cast_input(x)
        y = forward_fn(cast_x)
        if should_cast_back and torch.is_floating_point(y) and y.dtype != x.dtype:
            y = y.to(x.dtype)
        return y

    def _apply(self, fn: Any) -> "_Float32NormMixin":  # type: ignore[override]
        result = super()._apply(fn)
        self._restore_float32()
        return result


class _LayerNormFloat32(_Float32NormMixin, nn.LayerNorm):
    pass


if hasattr(nn, "RMSNorm"):

    class _RMSNormFloat32(_Float32NormMixin, nn.RMSNorm):  # type: ignore[misc]
        pass

else:
    _RMSNormFloat32 = None  # type: ignore[assignment]


class _BatchNorm1dFloat32(_Float32NormMixin, nn.BatchNorm1d):
    pass


class StochasticDepth(nn.Module):
    def __init__(self, p: float = 0.0, mode: str = "row") -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = float(p)
        self.mode = str(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        if keep_prob <= 0.0:
            return torch.zeros_like(x)
        if self.mode == "row":
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        else:
            shape = (1,) * x.ndim
        noise = x.new_empty(shape).bernoulli_(keep_prob).div_(keep_prob)
        return x * noise


def norm_layer(norm_type: str, d_model: int) -> nn.Module:
    kind = str(norm_type).lower()
    if kind in {"layernorm", "layer_norm", "ln"}:
        return _disable_module_torch_compile(_LayerNormFloat32(d_model))
    if kind in {"rmsnorm", "rms_norm", "rms"} and _RMSNormFloat32 is not None:
        return _disable_module_torch_compile(_RMSNormFloat32(d_model))
    if kind in {"batchnorm", "batchnorm1d", "bn", "bn1d"}:
        return _disable_module_torch_compile(_BatchNorm1dFloat32(d_model))
    return _disable_module_torch_compile(_LayerNormFloat32(d_model))


def schedule_stochastic_depth(max_rate: float, depth: int) -> list[float]:
    if depth <= 0:
        return []
    if max_rate <= 0.0:
        return [0.0 for _ in range(depth)]
    step = float(max_rate) / max(1, depth)
    return [step * float(index + 1) for index in range(depth)]


def reshape_for_heads(
    tensor: torch.Tensor, batch_size: int, head_count: int, head_dim: int
) -> torch.Tensor:
    return tensor.view(batch_size, -1, head_count, head_dim).transpose(1, 2)


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        *args: Any,
        in_channels: int = 1,
        d_model: int = 128,
        ndim: int = 2,
        patch: Tuple[int, ...] = (4, 4, 4),
        stride: Optional[Tuple[int, ...]] = None,
        grid: Optional[Tuple[int, ...]] = None,
        grid_3d: Optional[Tuple[int, int, int]] = None,
        dropout: float = 0.0,
        pad_to_multiple: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if ndim not in (1, 2, 3):
            raise ValueError(f"ndim must be 1, 2, or 3, got {ndim!r}")
        self.ndim = int(ndim)
        self.grid = grid
        self.grid_3d = grid_3d
        if any((p <= 0 for p in patch)):
            raise ValueError(f"patch sizes must be positive, got {patch}")
        if stride is not None and (
            len(stride) < self.ndim or any((s <= 0 for s in stride[: self.ndim]))
        ):
            raise ValueError(
                (
                    "stride must have length >= "
                    f"{self.ndim} with positive values, got {stride}"
                )
            )
        self.dropout = nn.Dropout(dropout)
        self.d_model = int(d_model)
        self.patch = patch
        self.pad_to_multiple = bool(pad_to_multiple)
        stride = patch if stride is None else stride
        match self.ndim:
            case 1:
                self.proj = nn.Conv1d(
                    in_channels,
                    d_model,
                    kernel_size=(patch[0],),
                    stride=(stride[0],),
                )
            case 2:
                self.proj = nn.Conv2d(
                    in_channels,
                    d_model,
                    kernel_size=(patch[0], patch[1]),
                    stride=(stride[0], stride[1]),
                )
            case 3:
                self.proj = nn.Conv3d(
                    in_channels,
                    d_model,
                    kernel_size=(patch[0], patch[1], patch[2]),
                    stride=(stride[0], stride[1], stride[2]),
                )

    def _normalize_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            b, fdim = x.shape
            match self.ndim:
                case 1:
                    if self.grid is None:
                        length = fdim
                        kernel = self.patch[0]
                        need = (length + kernel - 1) // kernel * kernel
                        if fdim < need:
                            x = torch.nn.functional.pad(x, (0, need - fdim))
                        return x.view(b, 1, -1)
                    (grid_length,) = self.grid
                    if fdim < grid_length:
                        x = torch.nn.functional.pad(x, (0, grid_length - fdim))
                    elif fdim > grid_length:
                        raise ValueError(
                            f"[B,F] grid(L={grid_length}) but F={fdim} > L."
                        )
                    return x.view(b, 1, grid_length)
                case 2:
                    if self.grid is None:
                        side = int(math.ceil(math.sqrt(fdim)))
                        h = w = side
                    else:
                        h, w = self.grid
                    need = h * w
                    if fdim < need:
                        x = torch.nn.functional.pad(x, (0, need - fdim))
                    elif fdim > need:
                        raise ValueError(
                            f"[B,F] grid(H={h},W={w}) but F={fdim} > H*W."
                        )
                    return x.view(b, h, w)
                case 3:
                    if self.grid_3d is None:
                        side = int(round(fdim ** (1.0 / 3.0)))
                        t = h = w = max(1, side)
                    else:
                        t, h, w = self.grid_3d
                    need = t * h * w
                    if fdim < need:
                        x = torch.nn.functional.pad(x, (0, need - fdim))
                    elif fdim > need:
                        raise ValueError(
                            f"[B,F] grid(T={t},H={h},W={w}) but F={fdim} > T*H*W."
                        )
                    return x.view(b, t, h, w)
        return x

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        match self.ndim:
            case 1:
                length = x.shape[-1]
                kernel = self.patch[0]
                need = (length + kernel - 1) // kernel * kernel
                if length < need:
                    x = torch.nn.functional.pad(x, (0, need - length))
            case 2:
                h, w = x.shape[-2:]
                kh, kw = self.patch[:2]
                need_h = (h + kh - 1) // kh * kh
                need_w = (w + kw - 1) // kw * kw
                if h < need_h or w < need_w:
                    x = torch.nn.functional.pad(
                        x, (0, need_w - w, 0, need_h - h)
                    )
            case 3:
                t, h, w = x.shape[-3:]
                kt, kh, kw = self.patch
                need_t = (t + kt - 1) // kt * kt
                need_h = (h + kh - 1) // kh * kh
                need_w = (w + kw - 1) // kw * kw
                if t < need_t or h < need_h or w < need_w:
                    x = torch.nn.functional.pad(
                        x,
                        (0, need_w - w, 0, need_h - h, 0, need_t - t),
                    )
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        x = self._normalize_shape(x)
        x = torch.atleast_3d(x)
        if x.shape[1] != self.patch[0] and self.pad_to_multiple:
            x = self._pad(x)
        y = self.proj(x)
        match self.ndim:
            case 1:
                b, d, length = y.shape
                tokens = y.transpose(1, 2).contiguous().view(b, length, d)
                meta = (length, 1, 1)
            case 2:
                b, d, h, w = y.shape
                tokens = y.permute(0, 2, 3, 1).contiguous().view(b, h * w, d)
                meta = (1, h, w)
            case 3:
                b, d, t, h, w = y.shape
                tokens = (
                    y.permute(0, 2, 3, 4, 1).contiguous().view(b, t * h * w, d)
                )
                meta = (t, h, w)
            case _:
                raise RuntimeError("Unsupported ndim for PatchEmbedding")
        return (self.dropout(tokens), meta)


class CrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        bias: bool = True,
        *,
        use_gate: bool = True,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead for attention")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.norm_q = norm_layer(norm_type, d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.use_gate = bool(use_gate)
        if self.use_gate:
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("gate", None)
        self.sdpa = DotProductAttention()

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Nq, D = q.shape
        qn = self.q_proj(self.norm_q(q))
        kv = self.kv_proj(kv)
        k, v = kv.chunk(2, dim=-1)
        qh, kh, vh = (
            reshape_for_heads(qn, B, self.nhead, self.head_dim),
            reshape_for_heads(k, B, self.nhead, self.head_dim),
            reshape_for_heads(v, B, self.nhead, self.head_dim),
        )
        yh = self.sdpa(qh, kh, vh, attn_mask=attn_mask)
        y = yh.transpose(1, 2).contiguous().view(B, Nq, D)
        y = self.out_proj(self.dropout(y))
        if self.use_gate:
            y = torch.sigmoid(self.gate) * y
        return q + y


class PatchAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        *,
        coord_dim: int = 3,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead for PatchAttention")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = self.d_model // self.nhead
        self.coord_dim = int(coord_dim)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=True)
        self.rel_bias = nn.Sequential(
            nn.Linear(self.coord_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.nhead),
        )
        self.rel_value = nn.Sequential(
            nn.Linear(self.coord_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.attn = DotProductAttention(
            num_heads=self.nhead, head_dim=self.head_dim
        )

    def _expand_mask(
        self, mask: torch.Tensor, *, target: torch.Size
    ) -> torch.Tensor:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[-2:] != target[-2:]:
            raise ValueError("Attention mask shape mismatch in PatchAttention")
        return mask

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        if coords.shape[:2] != (B, N):
            raise ValueError("coords must have shape (B, N, C)")
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        qh, kh, vh = (
            reshape_for_heads(q, B, self.nhead, self.head_dim),
            reshape_for_heads(k, B, self.nhead, self.head_dim),
            reshape_for_heads(v, B, self.nhead, self.head_dim),
        )
        rel = coords.unsqueeze(2) - coords.unsqueeze(1)
        rel_bias = self.rel_bias(rel).permute(0, 3, 1, 2)
        rel_value = (
            self.rel_value(rel)
            .view(B, N, N, self.nhead, self.head_dim)
            .permute(0, 3, 1, 2, 4)
        )
        additive: Optional[torch.Tensor] = None
        if attn_mask is not None:
            mask = self._expand_mask(attn_mask, target=rel_bias.shape)
            if mask.dtype == torch.bool:
                additive = rel_bias.new_zeros(rel_bias.shape)
                additive = additive.masked_fill(~mask, float("-inf"))
            else:
                additive = mask.to(dtype=rel_bias.dtype)
        scores = torch.einsum("bhid,bhjd->bhij", qh, kh) / math.sqrt(
            float(self.head_dim)
        )
        scores = scores + rel_bias
        if additive is not None:
            scores = scores + additive
            attn_bias = rel_bias + additive
        else:
            attn_bias = rel_bias
        weights = torch.softmax(scores, dim=-1)
        base = self.attn(
            qh,
            kh,
            vh,
            attn_mask=attn_bias.to(dtype=qh.dtype),
            training=self.training,
        )
        rel_context = torch.einsum("bhij,bhijd->bhid", weights, rel_value)
        context = base + rel_context
        return context.transpose(1, 2).contiguous().view(B, N, self.d_model)


class TemporalEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()
        self.msr = MultiScaleRetention(d_model, nhead)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        h = self.msr(x, attn_mask=attn_mask, state=state)
        if isinstance(h, tuple):
            out, new_state = h
            if new_state is None:
                new_state = state
        else:
            out, new_state = h, state
        return out, new_state


class GlobalEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()
        self.msr = MultiScaleRetention(d_model, nhead)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        h = self.msr(x, attn_mask=attn_mask, state=state)
        if isinstance(h, tuple):
            out, new_state = h
            if new_state is None:
                new_state = state
        else:
            out, new_state = h, state
        return out, new_state
