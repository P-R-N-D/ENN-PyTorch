# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

import torch
from torch import nn

from ..utils.compat import patch_torch
from ..utils.optimization import (
    DotProductAttention,
    MultiScaleRetention,
    MultiScaleRetentionCompat,
)

from .functional import SwiGLU

patch_torch()
if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import ModelConfig


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
        return nn.LayerNorm(d_model)
    elif kind in {"rmsnorm", "rms_norm", "rms"} and hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(d_model)
    elif kind in {"batchnorm", "batchnorm1d", "bn", "bn1d"}:
        return nn.BatchNorm1d(d_model)
    else:
        return nn.LayerNorm(d_model)

def schedule_stochastic_depth(max_rate: float, depth: int) -> list[float]:
    if depth <= 0:
        return []
    elif max_rate <= 0.0:
        return [0.0 for _ in range(depth)]
    else:
        step = float(max_rate) / max(1, depth)
        return [step * float(index + 1) for index in range(depth)]

def reshape_for_heads(
    tensor: torch.Tensor, batch_size: int, head_count: int, head_dim: int
) -> torch.Tensor:
    return tensor.view(batch_size, -1, head_count, head_dim).transpose(1, 2)


_MODELING_TYPE_ALIASES: dict[str, str] = {
    "ss": "sxs",
    "spatial": "sxs",
    "sxs": "sxs",
    "tt": "txt",
    "temporal": "txt",
    "txt": "txt",
    "txs": "txs",
    "st": "sxt",
    "ts": "sxt",
    "sxt": "sxt",
    "spatiotemporal": "sxt",
    "spatio-temporal": "sxt",
    "temporospatial": "sxt",
    "temporo-spatial": "sxt",
}

def _normalize_modeling_type(value: Any) -> str:
    mode = str(value).strip().lower()
    normalized = _MODELING_TYPE_ALIASES.get(mode)
    if normalized is None:
        raise ValueError(f"Unsupported modeling type '{value}'")
    return normalized

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
            len(stride) < self.ndim
            or any((s <= 0 for s in stride[: self.ndim]))
        ):
            raise ValueError(
                (
                    "stride must have length >= "
                    f"{self.ndim} with positive values, got {stride}"
                )
            )
        match self.ndim:
            case 1:
                if grid is not None and len(grid) not in (0, 1):
                    raise ValueError(
                        f"1D grid must be None or (S,), got {grid}"
                    )
            case 2:
                if grid is not None and len(grid) != 2:
                    raise ValueError(f"2D grid must be (H,W), got {grid}")
            case 3:
                if grid_3d is not None and len(grid_3d) != 3:
                    raise ValueError(f"3D grid must be (T,H,W), got {grid_3d}")
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
                        need_hw = h * w
                        if fdim > need_hw:
                            raise ValueError(
                                f"[B,F] grid({h}x{w}) but F={fdim} > H*W={need_hw}."
                            )
                    need = h * w
                    if fdim < need:
                        x = torch.nn.functional.pad(x, (0, need - fdim))
                    x = x.view(b, 1, h, w)
                    if self.pad_to_multiple:
                        ht = int(math.ceil(h / self.patch[0]) * self.patch[0])
                        wt = int(math.ceil(w / self.patch[1]) * self.patch[1])
                        if ht != h or wt != w:
                            x = torch.nn.functional.pad(
                                x, (0, wt - w, 0, ht - h)
                            )
                    return x.contiguous(memory_format=torch.channels_last)
                case 3:
                    if self.grid_3d is None:
                        raise ValueError(
                            "Provide grid_3d=(T,H,W) for 3D with [B,F]."
                        )
                    t, h, w = self.grid_3d
                    need = t * h * w
                    if fdim < need:
                        x = torch.nn.functional.pad(x, (0, need - fdim))
                    elif fdim > need:
                        raise ValueError(
                            f"[B,F] grid_3d product {need}, but F={fdim} > product."
                        )
                    x = x.view(b, 1, t, h, w)
                    if self.pad_to_multiple:
                        tt = int(math.ceil(t / self.patch[0]) * self.patch[0])
                        ht = int(math.ceil(h / self.patch[1]) * self.patch[1])
                        wt = int(math.ceil(w / self.patch[2]) * self.patch[2])
                        if tt != t or ht != h or wt != w:
                            x = torch.nn.functional.pad(
                                x, (0, wt - w, 0, ht - h, 0, tt - t)
                            )
                    return x.contiguous(memory_format=torch.channels_last_3d)
        match self.ndim:
            case 1:
                return x if x.ndim == 3 else x.unsqueeze(1)
            case 2:
                x = x if x.ndim == 4 else x.unsqueeze(1)
                return x.contiguous(memory_format=torch.channels_last)
            case 3:
                x = x if x.ndim == 5 else x.unsqueeze(1)
                return x.contiguous(memory_format=torch.channels_last_3d)
        return x

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        x = self._normalize_shape(x)
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
        return (self.dropout(tokens), meta)

class SinusoidalEncoding(nn.Module):
    def __init__(self, d_model: int, axes: Tuple[str, ...]) -> None:
        super().__init__()
        self.axes = axes
        assert len(axes) >= 1 and all(
            (a in ("l", "t", "t_score", "h", "w") for a in axes)
        )
        assert d_model % len(axes) == 0
        self.d_axis = d_model // len(axes)
        assert self.d_axis % 2 == 0
        self.register_buffer(
            "_cache_meta",
            torch.tensor([-1, -1, -1], dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer("_cache_pe", torch.empty(0, 0), persistent=False)
        self._cache_device = None
        self._cache_dtype = None

    @staticmethod
    def _to_1d(n: int, dim: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(n, dim, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(
        self,
        meta: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        t, h, w = meta
        if (
            self._cache_device == device
            and self._cache_dtype == dtype
            and (self._cache_meta.numel() == 3)
            and (
                tuple((int(x) for x in self._cache_meta.tolist())) == (t, h, w)
            )
        ):
            return self._cache_pe
        chunks: List[torch.Tensor] = []
        if "t" in self.axes:
            pt = self._to_1d(t, self.d_axis, device).view(t, 1, 1, self.d_axis)
            chunks.append(pt.expand(t, h, w, self.d_axis))
        if "h" in self.axes:
            ph = self._to_1d(h, self.d_axis, device).view(1, h, 1, self.d_axis)
            chunks.append(ph.expand(t, h, w, self.d_axis))
        if "w" in self.axes:
            pw = self._to_1d(w, self.d_axis, device).view(1, 1, w, self.d_axis)
            chunks.append(pw.expand(t, h, w, self.d_axis))
        if "l" in self.axes:
            length_axis = t
            pl = self._to_1d(length_axis, self.d_axis, device).view(
                length_axis, 1, 1, self.d_axis
            )
            chunks.append(pl.expand(length_axis, 1, 1, self.d_axis))
        pe = (
            torch.cat(chunks, dim=-1)
            .view(-1, self.d_axis * len(self.axes))
            .to(dtype=dtype)
        )
        self._cache_meta = torch.tensor([t, h, w], dtype=torch.int64)
        self._cache_pe = pe
        self._cache_device = device
        self._cache_dtype = dtype
        return pe

class GatedCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.norm_q = norm_layer(norm_type, d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(1))
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
        return q + torch.sigmoid(self.gate) * y

class SpatialEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        *args: Any,
        coord_dim: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_type: str = "layernorm",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(
                "d_model must be divisible by nhead for SpatialEncoderLayer"
            )
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = self.d_model // self.nhead
        self.coord_dim = int(coord_dim)
        self.norm1 = norm_layer(norm_type, self.d_model)
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
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(p=drop_path, mode="row")
        self.norm2 = norm_layer(norm_type, self.d_model)
        hid = int(self.d_model * mlp_ratio * (2.0 / 3.0))
        self.ffn = SwiGLU(
            self.d_model, hid, out_dim=self.d_model, dropout=dropout
        )

    def _apply_attn_mask(
        self, scores: torch.Tensor, attn_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if attn_mask is None:
            return scores
        mask = attn_mask
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[-2:] != scores.shape[-2:]:
            raise ValueError("Attention mask shape mismatch in SpatialEncoderLayer")
        return scores.masked_fill(~mask.to(dtype=torch.bool), float("-inf"))

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        if coords.shape[:2] != (B, N):
            raise ValueError("coords must have shape (B, N, C)")
        qkv = self.qkv(self.norm1(x))
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
        scores = torch.einsum("bhid,bhjd->bhij", qh, kh) / math.sqrt(
            float(self.head_dim)
        )
        scores = scores + rel_bias
        scores = self._apply_attn_mask(scores, attn_mask)
        weights = torch.softmax(scores, dim=-1)
        value = vh.unsqueeze(2).expand(-1, -1, N, -1, -1) + rel_value
        y = torch.einsum("bhij,bhijd->bhid", weights, value)
        y = y.transpose(1, 2).contiguous().view(B, N, D)
        x = x + self.drop_path(self.dropout(y))
        x = x + self.drop_path(self.dropout(self.ffn(self.norm2(x))))
        return x

class SpatialEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        depth: int,
        *args: Any,
        coord_dim: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_type: str = "layernorm",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        drops = schedule_stochastic_depth(drop_path, depth)
        self.blocks = nn.ModuleList(
            [
                SpatialEncoderLayer(
                    d_model,
                    nhead,
                    coord_dim=coord_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drops[i],
                    norm_type=norm_type,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(norm_type, d_model)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, coords, attn_mask=attn_mask)
        return self.norm(x)

class TemporalEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(norm_type, d_model)
        self.msr = MultiScaleRetention(d_model, nhead)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(p=drop_path, mode="row")
        self.norm2 = norm_layer(norm_type, d_model)
        hid = int(d_model * mlp_ratio * (2.0 / 3.0))
        self.ffn = SwiGLU(d_model, hid, out_dim=d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        h = self.msr(self.norm1(x), attn_mask=causal_mask, state=state)
        x = x + self.drop_path(self.dropout(h))
        x = x + self.drop_path(self.dropout(self.ffn(self.norm2(x))))
        return (x, state)

class TemporalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        depth: int,
        *args: Any,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_type: str = "layernorm",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        drops = schedule_stochastic_depth(drop_path, depth)
        self.blocks = nn.ModuleList(
            [
                TemporalEncoderLayer(
                    d_model,
                    nhead,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drops[i],
                    norm_type=norm_type,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(norm_type, d_model)

    def forward(
        self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        state = None
        for blk in self.blocks:
            x, state = blk(x, causal_mask=causal_mask, state=state)
        return self.norm(x)

class CrossTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        *args: Any,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.cross_s = GatedCrossAttention(
            d_model, nhead, dropout=dropout, norm_type=norm_type
        )
        self.cross_t = GatedCrossAttention(
            d_model, nhead, dropout=dropout, norm_type=norm_type
        )
        self.mix_norm = norm_layer(norm_type, 2 * d_model)
        hid = int(2 * d_model * mlp_ratio * (2.0 / 3.0))
        self.mix = SwiGLU(2 * d_model, hid, out_dim=d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(p=drop_path, mode="row")

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        temporal_tokens: torch.Tensor,
        mode: str = "spatiotemporal",
    ) -> torch.Tensor:
        mode_l = _normalize_modeling_type(mode)
        s_context = self.cross_s(spatial_tokens, temporal_tokens)
        t_context = self.cross_t(temporal_tokens, spatial_tokens)
        if mode_l == "sxs":
            return s_context
        if mode_l == "txt":
            return t_context
        if mode_l == "txs":
            base = torch.cat(
                [
                    t_context,
                    s_context.mean(dim=1, keepdim=True).expand(
                        -1, t_context.size(1), -1
                    ),
                ],
                dim=-1,
            )
            fused = self.mix(self.mix_norm(base))
            return t_context + self.drop_path(self.dropout(fused))
        base = torch.cat(
            [
                s_context,
                t_context.mean(dim=1, keepdim=True).expand(
                    -1, s_context.size(1), -1
                ),
            ],
            dim=-1,
        )
        fused = self.mix(self.mix_norm(base))
        return s_context + self.drop_path(self.dropout(fused))

class Payload:
    tokens: torch.Tensor
    context: torch.Tensor
    flat: torch.Tensor
    offset: torch.Tensor
    context_shape: Tuple[int, ...]

class GlobalEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        *args: Any,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_type: str = "layernorm",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(norm_type, d_model)
        self.msr = MultiScaleRetention(d_model, nhead)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(p=drop_path, mode="row")
        self.norm2 = norm_layer(norm_type, d_model)
        hid = int(d_model * mlp_ratio * (2.0 / 3.0))
        self.ffn = SwiGLU(d_model, hid, out_dim=d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        h = self.msr(self.norm1(x), attn_mask=causal_mask, state=state)
        x = x + self.drop_path(self.dropout(h))
        x = x + self.drop_path(self.dropout(self.ffn(self.norm2(x))))
        return x, state

class GlobalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        depth: int,
        *args: Any,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_type: str = "layernorm",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        drops = schedule_stochastic_depth(drop_path, depth)
        self.blocks = nn.ModuleList(
            [
                GlobalEncoderLayer(
                    d_model,
                    nhead,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drops[i],
                    norm_type=norm_type,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(norm_type, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        state: Optional[dict] = None
        for blk in self.blocks:
            tokens, state = blk(tokens, causal_mask=None, state=state)
        return self.norm(tokens)

class LocalProcessor(nn.Module):
    def __init__(
        self, in_dim: int, out_shape: Sequence[int], config: ModelConfig
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_shape = tuple((int(v) for v in out_shape))
        self.out_dim = int(prod(self.out_shape))
        self.d_model = int(config.depth)
        self.nhead = int(config.heads)
        self.modeling_type = _normalize_modeling_type(
            config.modeling_type
        )
        self.spatial_tokens = max(1, int(config.spatial_latents))
        self.temporal_tokens = max(1, int(config.temporal_latents))
        self.mlp_ratio = float(config.mlp_ratio)
        self.dropout = float(config.dropout)
        self.drop_path = float(config.drop_path)
        self.norm_type = str(config.normalization_method)
        self.spatial_tokenizer = nn.Linear(
            self.in_dim, self.spatial_tokens * self.d_model
        )
        self.temporal_tokenizer = nn.Linear(
            self.in_dim, self.temporal_tokens * self.d_model
        )
        self.register_buffer(
            "spatial_coords_template",
            self._build_spatial_coords(
                self.spatial_tokens, device=torch.device("cpu")
            ),
            persistent=False,
        )
        self.spatial_encoder = SpatialEncoder(
            self.d_model,
            self.nhead,
            depth=max(1, int(config.spatial_depth)),
            coord_dim=self.spatial_coords_template.shape[-1],
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            drop_path=self.drop_path,
            norm_type=self.norm_type,
        )
        self.temporal_encoder = TemporalEncoder(
            self.d_model,
            self.nhead,
            depth=max(1, int(config.temporal_depth)),
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            drop_path=self.drop_path,
            norm_type=self.norm_type,
        )
        self.perception = CrossTransformer(
            self.d_model,
            self.nhead,
            dropout=self.dropout,
            norm_type=self.norm_type,
            mlp_ratio=self.mlp_ratio,
            drop_path=self.drop_path,
        )
        self.norm = norm_layer(self.norm_type, self.d_model)
        hid = int(self.d_model * max(1.0, self.mlp_ratio))
        self.head = nn.Sequential(
            norm_layer(self.norm_type, self.d_model),
            nn.Linear(self.d_model, hid),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hid, self.out_dim),
        )

    @staticmethod
    def _build_spatial_coords(
        n_tokens: int, device: torch.device
    ) -> torch.Tensor:
        side = max(1, int(round(n_tokens ** (1.0 / 3.0))))
        coords: List[Tuple[float, float, float]] = []
        for idx in range(n_tokens):
            z = idx // (side * side)
            rem = idx % (side * side)
            y = rem // side
            x = rem % side
            if side == 1:
                coords.append((0.0, 0.0, 0.0))
            else:
                coords.append(
                    (
                        x / (side - 1 if side > 1 else 1),
                        y / (side - 1 if side > 1 else 1),
                        z / (side - 1 if side > 1 else 1),
                    )
                )
        return torch.tensor(coords, dtype=torch.float32, device=device)

    def _spatial_coords(
        self, batch: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        coords = self.spatial_coords_template.to(device=device, dtype=dtype)
        return coords.unsqueeze(0).expand(batch, -1, -1)

    def forward(self, x: torch.Tensor) -> Payload:
        B = x.shape[0]
        spatial_tokens = self.spatial_tokenizer(x).view(
            B, self.spatial_tokens, self.d_model
        )
        temporal_tokens = self.temporal_tokenizer(x).view(
            B, self.temporal_tokens, self.d_model
        )
        coords = self._spatial_coords(B, x.device, spatial_tokens.dtype)
        spatial_out = self.spatial_encoder(spatial_tokens, coords)
        temporal_out = self.temporal_encoder(temporal_tokens)
        mode = self.modeling_type
        if mode == "sxs":
            tokens = spatial_out
        elif mode == "txt":
            tokens = temporal_out
        elif mode == "txs":
            tokens = self.perception(temporal_out, spatial_out, mode="txs")
        elif mode == "sxt":
            tokens = self.perception(spatial_out, temporal_out, mode="sxt")
        else:
            raise RuntimeError(f"Unhandled modeling type '{mode}'")
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        flat = self.head(pooled)
        context = flat.view(B, *self.out_shape)
        dims = tuple(range(1, context.ndim))
        offset = context.mean(dim=dims, keepdim=True)
        return Payload(
            tokens=tokens,
            context=context,
            flat=flat,
            offset=offset,
            context_shape=self.out_shape,
        )

    def decode(
        self, tokens: torch.Tensor, *args: Any, apply_norm: bool = False, **kwargs: Any
    ) -> torch.Tensor:
        if apply_norm:
            tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        flat = self.head(pooled)
        return flat.view(tokens.shape[0], *self.out_shape)
