# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import math
import warnings
from dataclasses import dataclass
from importlib import import_module
from math import prod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    cast,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDictBase

try:
    from torch.utils.checkpoint import checkpoint as activation_checkpoint
except Exception:
    activation_checkpoint = None

from ..api import is_meta_or_fake_tensor
from ..backend.compat import patch_torch
from ..functional.fx import AutoCast, Gradient, reshape_for_heads
from .activations import SwiGLU
from .kernels import (
    DotProductAttention,
    MultiHeadAttention,
    MultiScaleRetention,
    attn_mask_to_additive,
)

try:
    from torch._inductor import config as _inductor_config
except Exception:
    _inductor_config = None
else:
    try:
        triton_cfg = getattr(_inductor_config, "triton")
        if hasattr(triton_cfg, "cudagraph_skip_dynamic_graphs"):
            triton_cfg.cudagraph_skip_dynamic_graphs = True
    except Exception:
        pass

try:
    from ..backend.compat import RMSNorm as _Norm
except Exception:
    _Norm = nn.LayerNorm


try:
    from ..backend.profiler import FLOP_PROFILER
except Exception:
    FLOP_PROFILER = None


patch_torch()


if TYPE_CHECKING:
    from ..api.config import ModelConfig


LayerNorm = nn.LayerNorm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, axes: Tuple[str, ...]) -> None:
        super().__init__()
        self.axes = axes
        if len(axes) < 1 or any(a not in ("l", "t", "t_score", "h", "w") for a in axes):
            raise ValueError("axes must be drawn from {'l','t','t_score','h','w'}")
        if d_model % len(axes) != 0:
            raise ValueError("d_model must be divisible by number of axes")
        self.d_axis = d_model // len(axes)
        if self.d_axis % 2 != 0:
            raise ValueError("per-axis dimension must be even")
        self.register_buffer(
            "_cache_meta",
            torch.tensor([-1, -1, -1], dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer("_cache_pe", torch.empty(0, 0), persistent=False)
        self._cache_device: torch.device | None = None
        self._cache_dtype: torch.dtype | None = None

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
            and self._cache_meta.numel() == 3
            and tuple(int(x) for x in self._cache_meta.tolist()) == (t, h, w)
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
            pl = self._to_1d(t, self.d_axis, device).view(t, 1, 1, self.d_axis)
            chunks.append(pl.expand(t, 1, 1, self.d_axis))
        if "t_score" in self.axes:
            pt_score = self._to_1d(t, self.d_axis, device).view(t, 1, 1, self.d_axis)
            chunks.append(pt_score.expand(t, h, w, self.d_axis))
        pe = torch.cat(chunks, dim=-1).view(-1, self.d_axis * len(self.axes))
        pe = pe.to(dtype=dtype)
        self._cache_meta = torch.tensor([t, h, w], dtype=torch.int64)
        self._cache_pe = pe
        self._cache_device = device
        self._cache_dtype = dtype
        return pe


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
        return LayerNorm(d_model)
    if kind in {"rmsnorm", "rms_norm", "rms"} and hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(d_model)
    if kind in {"batchnorm", "batchnorm1d", "bn", "bn1d"}:
        return nn.BatchNorm1d(d_model)
    return LayerNorm(d_model)


def schedule_stochastic_depth(max_rate: float, depth: int) -> list[float]:
    if depth <= 0:
        return []
    if max_rate <= 0.0:
        return [0.0 for _ in range(depth)]
    step = float(max_rate) / max(1, depth)
    return [step * float(index + 1) for index in range(depth)]


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
        self.static_spatial: Optional[Tuple[int, ...]] = getattr(self, "static_spatial", None)
        if self.static_spatial is None:
            hw = getattr(self, "static_hw", None)
            if hw is not None and self.ndim == 2:
                self.static_spatial = (int(hw[0]), int(hw[1]))
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
                    x = F.pad(x, (0, need - length))
            case 2:
                h, w = x.shape[-2:]
                kh, kw = self.patch[:2]
                need_h = (h + kh - 1) // kh * kh
                need_w = (w + kw - 1) // kw * kw
                if h < need_h or w < need_w:
                    x = F.pad(x, (0, need_w - w, 0, need_h - h))
            case 3:
                t, h, w = x.shape[-3:]
                kt, kh, kw = self.patch
                need_t = (t + kt - 1) // kt * kt
                need_h = (h + kh - 1) // kh * kh
                need_w = (w + kw - 1) // kw * kw
                if t < need_t or h < need_h or w < need_w:
                    x = F.pad(
                        x,
                        (0, need_w - w, 0, need_h - h, 0, need_t - t),
                    )
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        x = self._normalize_shape(x)
        x = torch.atleast_3d(x)
        if self.static_spatial is not None:
            x = self._pad_or_crop_to_nd(x, self.static_spatial)
        elif x.shape[1] != self.patch[0] and self.pad_to_multiple:
            x = self._pad_dynamic(x)
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

    def _pad_or_crop_to_nd(self, x: torch.Tensor, target: Tuple[int, ...]) -> torch.Tensor:
        if len(target) != self.ndim:
            raise ValueError(
                f"static_spatial must have length {self.ndim}, got {len(target)}"
            )
        tgt = [int(v) for v in target]
        if any(v <= 0 for v in tgt):
            raise ValueError("static_spatial values must be positive")
        spatial = list(x.shape[-self.ndim :])
        pads: list[int] = []
        for cur, want in reversed(list(zip(spatial, tgt))):
            pad_right = max(want - cur, 0)
            pads.extend([0, pad_right])
        if any(pads):
            x = F.pad(x, tuple(pads))
        slices = [slice(None)] * x.ndim
        base = x.ndim - self.ndim
        for offset, want in enumerate(tgt):
            slices[base + offset] = slice(0, want)
        return x[tuple(slices)]

    def _pad_dynamic(self, x: torch.Tensor) -> torch.Tensor:
        return self._pad(x)


class CrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        bias: bool = True,
        *args: Any,
        use_gate: bool = True,
        **kwargs: Any,
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
        *args: Any,
        coord_dim: int = 3,
        **kwargs: Any,
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
        if attn_mask is None:
            additive = torch.zeros_like(rel_bias)
        else:
            mask = attn_mask
            if mask.dtype is torch.bool:
                mask = torch.logical_not(mask)
            additive = attn_mask_to_additive(
                mask,
                batch=B,
                heads=self.nhead,
                seq_q=N,
                seq_k=N,
                dtype=rel_bias.dtype,
                device=rel_bias.device,
            )
        scores = torch.einsum("bhid,bhjd->bhij", qh, kh) / math.sqrt(
            float(self.head_dim)
        )
        scores = scores + rel_bias + additive
        attn_bias = rel_bias + additive
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


class Retention(nn.Module):
    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()
        self.msr = MultiScaleRetention(d_model, nhead)

    def forward(
        self,
        x: torch.Tensor,
        *args: Any,
        attn_mask: Optional[torch.Tensor] = None,
        state: Optional[dict] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        h = self.msr(x, attn_mask=attn_mask, state=state)
        if isinstance(h, tuple):
            out, new_state = h
            if new_state is None:
                new_state = state
        else:
            out, new_state = h, state
        return out, new_state


def _build_dilated_mask(
    seq_len: int,
    *args: Any,
    dilation: int = 1,
    window_size: Optional[int] = None,
    causal: bool = False,
    **kwargs: Any,
) -> torch.Tensor:

    if dilation < 1:
        raise ValueError(f"dilation must be >= 1, got {dilation}")
    L = int(seq_len)
    device = torch.device("cpu")
    i = torch.arange(L, device=device).unsqueeze(1).expand(L, L)
    j = torch.arange(L, device=device).unsqueeze(0).expand(L, L)
    dist = (i - j).abs()
    congruent = ((i - j) % dilation) == 0
    within = (
        torch.ones_like(congruent, dtype=torch.bool)
        if window_size is None
        else (dist <= window_size)
    )
    not_future = (j <= i) if causal else torch.ones_like(congruent, dtype=torch.bool)
    allowed = congruent & within & not_future
    return (~allowed).contiguous()


class DilatedAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *args: Any,
        dilation: int = 1,
        window_size: Optional[int] = None,
        causal: bool = False,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        batch_first: bool = True,
        bias: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dilation = int(dilation)
        self.window_size = window_size
        self.causal = causal
        self.batch_first = batch_first
        self.nhead = int(num_heads)
        self.head_dim = int(embed_dim // max(self.nhead, 1))
        self.dropout_p = float(dropout)
        self.__stf_attention_profile__ = {
            "format": "xs",
            "num_heads": self.nhead,
            "head_dim": self.head_dim,
            "dropout_attr": "dropout_p",
            "effective_window_attr": ["window_size"],
            "include_softmax_scale_dropout": True,
        }

        self.norm1 = _Norm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = _Norm(embed_dim)
        hidden = int(mlp_ratio * embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim, bias=True),
        )

        self._mask_cache_cpu: Dict[int, torch.Tensor] = {}
        self._mask_cache_gpu: Dict[Tuple[int, int], torch.Tensor] = {}

    def _get_mask(self, L: int, device: torch.device) -> torch.Tensor:
        key = (L, device.index if device.type == "cuda" else -1)
        mask_gpu = self._mask_cache_gpu.get(key)
        if mask_gpu is not None and mask_gpu.device == device:
            return mask_gpu
        mask_cpu = self._mask_cache_cpu.get(L)
        if mask_cpu is None:
            mask_cpu = (
                _build_dilated_mask(
                    L,
                    dilation=self.dilation,
                    window_size=self.window_size,
                    causal=self.causal,
                )
                .detach()
                .contiguous()
                .cpu()
            )
            self._mask_cache_cpu[L] = mask_cpu
        mask_gpu = mask_cpu.to(device=device, non_blocking=True)
        self._mask_cache_gpu[key] = mask_gpu
        return mask_gpu

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.batch_first:
            x = x.transpose(0, 1)
        B, L, _ = x.shape
        residual = x
        x = self.norm1(x)
        mask = self._get_mask(L, x.device)
        attn_out, attn_w = self.attn(
            x,
            x,
            x,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            is_causal=self.causal,
        )
        x = residual + self.dropout(attn_out)
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        if not self.batch_first:
            x = x.transpose(0, 1)
        return x, (attn_w if need_weights else None)


class PointTransformer(nn.Module):
    
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
        self.coord_dim = int(coord_dim)
        self.d_model = int(d_model)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(p=drop_path, mode="row")
        self.norm1 = norm_layer(norm_type, self.d_model)
        self.attn = PatchAttention(
            self.d_model, nhead, coord_dim=self.coord_dim
        )
        self.norm2 = norm_layer(norm_type, self.d_model)
        hid = int(self.d_model * mlp_ratio * (2.0 / 3.0))
        self.ffn = SwiGLU(
            self.d_model, hid, out_dim=self.d_model, dropout=dropout
        )
        self._ln_materialized = False

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if is_meta_or_fake_tensor(x):
            raise RuntimeError("meta/fake tensor reached PointTransformer.forward (x)")
        if is_meta_or_fake_tensor(coords):
            raise RuntimeError("meta/fake tensor reached PointTransformer.forward")
        if coords.shape[:2] != x.shape[:2] or coords.size(-1) != self.coord_dim:
            raise ValueError(
                f"coords must be (B, N, {self.coord_dim}), got {tuple(coords.shape)} vs x {tuple(x.shape)}"
            )
        x = x.contiguous()
        coords = coords.contiguous()
        if attn_mask is not None:
            attn_mask = attn_mask.contiguous()
            if is_meta_or_fake_tensor(attn_mask):
                raise RuntimeError("attn_mask is meta before attention")
        def _materialize_ln_(ln: nn.LayerNorm, ref: torch.Tensor) -> None:
            if not isinstance(ln, nn.LayerNorm):
                return
            dev = ref.device
            target_dtype = torch.float32 if dev.type == "cpu" else ref.dtype
            w = getattr(ln, "weight", None)
            b = getattr(ln, "bias", None)
            try:
                from torch.distributed._tensor import DTensor as _DTensor
            except Exception:
                dtensor_types: Tuple[type, ...] = tuple()
            else:
                dtensor_types = (_DTensor,)
            w_data = getattr(w, "data", None)
            if (
                isinstance(w, torch.Tensor)
                and (
                    is_meta_or_fake_tensor(w)
                    or isinstance(w, dtensor_types)
                    or isinstance(w_data, dtensor_types)
                )
            ):
                ln.weight = nn.Parameter(
                    torch.ones(ln.normalized_shape, device=dev, dtype=target_dtype)
                )
            b_data = getattr(b, "data", None)
            if (
                isinstance(b, torch.Tensor)
                and (
                    is_meta_or_fake_tensor(b)
                    or isinstance(b, dtensor_types)
                    or isinstance(b_data, dtensor_types)
                )
            ):
                ln.bias = nn.Parameter(
                    torch.zeros(ln.normalized_shape, device=dev, dtype=target_dtype)
                )

        _materialize_ln_(self.norm1, x)
        _materialize_ln_(self.norm2, x)
        self._ln_materialized = True
        _x = x
        if isinstance(_x, torch.Tensor) and is_meta_or_fake_tensor(_x):
            raise RuntimeError("x is meta before LayerNorm")
        if _x.device.type == "cpu" and _x.is_floating_point() and _x.dtype != torch.float32:
            _x = _x.float()
        _x = self.norm1(_x)
        if isinstance(_x, torch.Tensor) and is_meta_or_fake_tensor(_x):
            raise RuntimeError("x is meta after LayerNorm")
        if _x.device.type == "cpu" and x.is_floating_point() and x.dtype != torch.float32:
            _x = _x.to(x.dtype)
        y = self.attn(_x, coords, attn_mask=attn_mask)
        x = x + self.drop_path(self.dropout(y))
        _x2 = x
        if is_meta_or_fake_tensor(_x2):
            raise RuntimeError("x is meta before LayerNorm(norm2)")
        if (
            _x2.device.type == "cpu"
            and _x2.is_floating_point()
            and _x2.dtype != torch.float32
        ):
            _x2 = _x2.float()
        _x2 = self.norm2(_x2)
        if is_meta_or_fake_tensor(_x2):
            raise RuntimeError("x is meta after LayerNorm(norm2)")
        if (
            _x2.device.type == "cpu"
            and x.is_floating_point()
            and x.dtype != torch.float32
        ):
            _x2 = _x2.to(x.dtype)
        x = x + self.drop_path(self.dropout(self.ffn(_x2)))
        return x


_MODELING_TYPE_ALIASES: dict[str, str] = {
    "ss": "ss",
    "spatial": "ss",
    "sxs": "ss",
    "tt": "tt",
    "temporal": "tt",
    "txt": "tt",
    "ts": "ts",
    "txs": "ts",
    "temporal-spatial": "ts",
    "temporo-spatial": "ts",
    "temporospatial": "ts",
    "st": "st",
    "sxt": "st",
    "spatiotemporal": "st",
    "spatio-temporal": "st",
}

def _normalize_modeling_type(value: Any) -> str:
    mode = str(value).strip().lower()
    normalized = _MODELING_TYPE_ALIASES.get(mode)
    if normalized is None:
        raise ValueError(f"Unsupported modeling type '{value}'")
    return normalized

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
                PointTransformer(
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
        if is_meta_or_fake_tensor(x):
            raise RuntimeError("x is meta/fake before SpatialEncoder.forward")
        if is_meta_or_fake_tensor(coords):
            raise RuntimeError("coords is meta/fake before SpatialEncoder.forward")
        if x.dim() != 3:
            raise ValueError(
                f"SpatialEncoder expects (B, N, C) tokens, got shape {tuple(x.shape)}"
            )
        if coords.dim() != 3:
            raise ValueError(
                f"SpatialEncoder expects (B, N, D) coords, got shape {tuple(coords.shape)}"
            )
        if x.shape[:2] != coords.shape[:2]:
            raise ValueError(
                "tokens/coords batch or length mismatch: "
                f"tokens={tuple(x.shape)} vs coords={tuple(coords.shape)}"
            )
        x = x.contiguous()
        coords = coords.contiguous()
        if attn_mask is not None:
            if is_meta_or_fake_tensor(attn_mask):
                raise RuntimeError("attn_mask is meta/fake before SpatialEncoder.forward")
            attn_mask = attn_mask.contiguous()
        for blk in self.blocks:
            x = blk(x, coords, attn_mask=attn_mask)
        out = self.norm(x)
        if is_meta_or_fake_tensor(out):
            raise RuntimeError("SpatialEncoder produced meta/fake tensor")
        return out.contiguous()

class RetNet(nn.Module):
    
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
        self.retention = Retention(d_model, nhead)
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
        if is_meta_or_fake_tensor(x):
            raise RuntimeError("meta/fake tensor reached RetNet.forward")
        x = x.contiguous()
        if causal_mask is not None:
            causal_mask = causal_mask.contiguous()
        h, state = self.retention(
            self.norm1(x), attn_mask=causal_mask, state=state
        )
        x = x + self.drop_path(self.dropout(h))
        x = x + self.drop_path(self.dropout(self.ffn(self.norm2(x))))
        return x, state


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
                RetNet(
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
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        state: Optional[dict] = None,
        *args: Any,
        return_state: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[dict]]]:
        next_state = state
        for blk in self.blocks:
            x, next_state = blk(x, causal_mask=causal_mask, state=next_state)
        x = self.norm(x)
        if return_state:
            return x, next_state
        return x

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
        self.cross_s = CrossAttention(
            d_model, nhead, dropout=dropout, norm_type=norm_type
        )
        self.cross_t = CrossAttention(
            d_model, nhead, dropout=dropout, norm_type=norm_type
        )
        self.mix_norm = norm_layer(norm_type, 2 * d_model)
        hid = int(2 * d_model * mlp_ratio * (2.0 / 3.0))
        self.mix = SwiGLU(2 * d_model, hid, out_dim=d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(p=drop_path, mode="row")
        self._fixed_mode: Optional[str] = getattr(self, "modeling_type", None)

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        temporal_tokens: torch.Tensor,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        if self._fixed_mode is not None:
            mode_l = _normalize_modeling_type(self._fixed_mode)
            if mode_l == "ss":
                return self.cross_s(spatial_tokens, temporal_tokens)
            if mode_l == "tt":
                return self.cross_t(temporal_tokens, spatial_tokens)
            if mode_l == "ts":
                s_context = self.cross_s(spatial_tokens, temporal_tokens)
                t_context = self.cross_t(temporal_tokens, spatial_tokens)
                base = torch.cat(
                    [
                        t_context,
                        s_context.mean(dim=1, keepdim=True).expand_as(t_context),
                    ],
                    dim=-1,
                )
                fused = self.mix(self.mix_norm(base))
                return t_context + self.drop_path(self.dropout(fused))
            if mode_l == "st":
                s_context = self.cross_s(spatial_tokens, temporal_tokens)
                t_context = self.cross_t(temporal_tokens, spatial_tokens)
                base = torch.cat(
                    [
                        s_context,
                        t_context.mean(dim=1, keepdim=True).expand_as(s_context),
                    ],
                    dim=-1,
                )
                fused = self.mix(self.mix_norm(base))
                return s_context + self.drop_path(self.dropout(fused))
        requested = mode if mode is not None else "spatiotemporal"
        return self._forward_dynamic(spatial_tokens, temporal_tokens, requested)

    def _forward_dynamic(
        self,
        spatial_tokens: torch.Tensor,
        temporal_tokens: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        mode_l = _normalize_modeling_type(mode)
        s_context = self.cross_s(spatial_tokens, temporal_tokens)
        t_context = self.cross_t(temporal_tokens, spatial_tokens)
        if mode_l == "ss":
            return s_context
        if mode_l == "tt":
            return t_context
        if mode_l == "ts":
            base = torch.cat(
                [
                    t_context,
                    s_context.mean(dim=1, keepdim=True).expand_as(t_context),
                ],
                dim=-1,
            )
            fused = self.mix(self.mix_norm(base))
            return t_context + self.drop_path(self.dropout(fused))
        if mode_l == "st":
            base = torch.cat(
                [
                    s_context,
                    t_context.mean(dim=1, keepdim=True).expand_as(s_context),
                ],
                dim=-1,
            )
            fused = self.mix(self.mix_norm(base))
            return s_context + self.drop_path(self.dropout(fused))
        raise RuntimeError(f"Unhandled mode: {mode_l}")

@dataclass
class Payload:
    tokens: torch.Tensor
    context: torch.Tensor
    flat: torch.Tensor
    offset: torch.Tensor
    context_shape: Tuple[int, ...]

class LongNet(nn.Module):
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int,
        *args: Any,
        dilation_growth: int = 2,
        base_dilation: int = 1,
        window_size: Optional[int] = None,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        causal: bool = False,
        batch_first: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.nhead = int(num_heads)
        self.head_dim = int(embed_dim // max(self.nhead, 1))
        self.dropout_p = float(dropout)
        self.__stf_attention_profile__ = {
            "format": "xs",
            "num_heads": self.nhead,
            "head_dim": self.head_dim,
            "dropout_attr": "dropout_p",
            "effective_window_attr": ["window_size", "block_size"],
            "include_softmax_scale_dropout": True,
        }
        self.batch_first = batch_first
        self._impl: Optional[nn.Module] = None
        try:
            ts_longnet = import_module("torchscale.net.longnet")
        except Exception:
            self._impl = None
        else:
            ctor_variants = (
                {
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "depth": depth,
                    "dropout": dropout,
                },
                {
                    "d_model": embed_dim,
                    "nhead": num_heads,
                    "num_layers": depth,
                    "dropout": dropout,
                },
            )
            longnet_ctor = getattr(ts_longnet, "LongNet", None)
            impl = None
            if callable(longnet_ctor):
                for kw in ctor_variants:
                    try:
                        impl = longnet_ctor(**kw)
                        break
                    except Exception:
                        continue
            self._impl = impl

        if self._impl is not None:
            self._using = "torchscale"
            self._impl_batch_first = getattr(self._impl, "batch_first", True)
        else:
            self._using = "fallback"
            self._impl_batch_first = True
            layers: List[nn.Module] = []
            dilation = base_dilation
            for _ in range(depth):
                layers.append(
                    DilatedAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        dilation=dilation,
                        window_size=window_size,
                        dropout=dropout,
                        mlp_ratio=mlp_ratio,
                        causal=causal,
                        batch_first=True,
                    )
                )
                dilation = max(1, dilation * max(1, int(dilation_growth)))
            self.layers = nn.ModuleList(layers)
            self.norm = nn.LayerNorm(embed_dim)

    @property
    def using(self) -> str:
        return self._using

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        **_: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._impl is not None:
            out = x
            if (
                self._impl_batch_first
                and out.dim() == 3
                and out.shape[0] != out.shape[1]
                and not self._impl_batch_first
            ):
                out = out.transpose(0, 1)
            try:
                out = self._impl(out)
            except Exception as exc:
                warnings.warn(
                    f"torchscale LongNet 호출 실패: {exc}. 입력을 그대로 반환.",
                    RuntimeWarning,
                )
            if self._impl_batch_first is not True:
                out = out.transpose(0, 1)
            return out, None
        attn_w: Optional[torch.Tensor] = None
        out = x
        for layer in self.layers:
            out, attn_w = layer(
                out,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
            )
        out = self.norm(out)
        return out, attn_w


class GlobalEncoder(nn.Module):
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int,
        *args: Any,
        dilation_growth: int = 2,
        base_dilation: int = 1,
        window_size: Optional[int] = None,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        causal: bool = False,
        batch_first: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.backbone = LongNet(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            dilation_growth=dilation_growth,
            base_dilation=base_dilation,
            window_size=window_size,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            causal=causal,
            batch_first=batch_first,
        )

    @property
    def using(self) -> str:
        return self.backbone.using

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        **_: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.backbone(
            x,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )

class LocalProcessor(nn.Module):
    
    def __init__(
        self, in_dim: int, out_shape: Sequence[int], config: ModelConfig
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_shape = tuple((int(v) for v in out_shape))
        self.out_dim = int(math.prod(self.out_shape) if self.out_shape else 1)
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
        self._fixed_mode: Optional[str] = getattr(self, "modeling_type", None)
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
        spatial_raw = self.spatial_tokenizer(x)
        expected_spatial = B * self.spatial_tokens * self.d_model
        if spatial_raw.numel() != expected_spatial:
            raise RuntimeError(
                "spatial tokenizer output has unexpected numel: "
                f"got {spatial_raw.numel()} vs expected {expected_spatial}"
            )
        spatial_tokens = (
            spatial_raw.reshape(B, self.spatial_tokens, self.d_model).contiguous()
        )
        temporal_raw = self.temporal_tokenizer(x)
        expected_temporal = B * self.temporal_tokens * self.d_model
        if temporal_raw.numel() != expected_temporal:
            raise RuntimeError(
                "temporal tokenizer output has unexpected numel: "
                f"got {temporal_raw.numel()} vs expected {expected_temporal}"
            )
        temporal_tokens = (
            temporal_raw.reshape(B, self.temporal_tokens, self.d_model).contiguous()
        )
        coords = self._spatial_coords(B, x.device, spatial_tokens.dtype)
        spatial_out = self.spatial_encoder(spatial_tokens, coords)
        temporal_out = self.temporal_encoder(temporal_tokens)
        mode = self._fixed_mode
        if mode is not None:
            mode_l = _normalize_modeling_type(mode)
            if mode_l == "ss":
                tokens = spatial_out
            elif mode_l == "tt":
                tokens = temporal_out
            elif mode_l == "ts":
                if hasattr(self.perception, "_forward_dynamic"):
                    tokens = self.perception._forward_dynamic(
                        temporal_out, spatial_out, "ts"
                    )
                else:
                    tokens = self.perception(temporal_out, spatial_out, mode="ts")
            elif mode_l == "st":
                if hasattr(self.perception, "_forward_dynamic"):
                    tokens = self.perception._forward_dynamic(
                        spatial_out, temporal_out, "st"
                    )
                else:
                    tokens = self.perception(spatial_out, temporal_out, mode="st")
            else:
                raise RuntimeError(f"Unhandled modeling type '{mode}'")
        else:
            tokens = self._forward_dynamic(spatial_out, temporal_out)
        tokens = self.norm(tokens)
        tokens = tokens.contiguous()
        pooled = tokens.mean(dim=1)
        flat = self.head(pooled)
        flat = flat.contiguous()
        context = flat.reshape(B, *self.out_shape).contiguous()
        dims = tuple(range(1, context.ndim))
        offset = context.mean(dim=dims, keepdim=True).contiguous()
        return Payload(
            tokens=tokens,
            context=context,
            flat=flat,
            offset=offset,
            context_shape=self.out_shape,
        )

    def _forward_dynamic(self, spatial_out, temporal_out):
        mode = getattr(self, "modeling_type", None)
        if mode is None:
            raise RuntimeError("modeling_type is not set")
        mode_l = _normalize_modeling_type(mode)
        if mode_l == "ss":
            return spatial_out
        if mode_l == "tt":
            return temporal_out
        if mode_l == "ts":
            return self.perception(temporal_out, spatial_out, mode="ts")
        if mode_l == "st":
            return self.perception(spatial_out, temporal_out, mode="st")
        raise RuntimeError(f"Unhandled modeling type '{mode_l}'")

    def decode(
        self, tokens: torch.Tensor, *args: Any, apply_norm: bool = False, **kwargs: Any
    ) -> torch.Tensor:
        if apply_norm:
            tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        flat = self.head(pooled)
        return flat.reshape(tokens.shape[0], *self.out_shape)


class LossWeightPolicy(Protocol):
    
    def weights(self) -> Tuple[float, float]:
        ...

    def update(
        self,
        top_loss: Optional[torch.Tensor],
        bottom_loss: Optional[torch.Tensor],
    ) -> None:
        raise NotImplementedError


class Root(nn.Module):
    
    def __init__(
        self,
        in_dim: int,
        out_shape: Sequence[int],
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_shape = tuple((int(x) for x in out_shape))
        self.out_dim = int(prod(self.out_shape))
        if config.device is not None:
            self._device = torch.device(config.device)
        else:
            if torch.cuda.is_available():
                device_name = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device_name = "mps"
            elif getattr(torch, "is_vulkan_available", None) and torch.is_vulkan_available():
                device_name = "vulkan"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                device_name = "xpu"
            else:
                device_name = "cpu"
            self._device = torch.device(device_name)
        self.is_norm_linear = bool(getattr(config, "use_linear_branch", False))
        self.linear_branch = (
            nn.Linear(self.in_dim, self.out_dim).to(self._device)
            if self.is_norm_linear
            else None
        )
        self.local_net = LocalProcessor(
            self.in_dim, self.out_shape, config=config
        ).to(self._device)
        global_net = GlobalEncoder(
            int(config.depth),
            int(config.heads),
            depth=max(1, int(getattr(config, "temporal_depth", 1))),
            mlp_ratio=float(getattr(config, "mlp_ratio", 4.0)),
            dropout=float(getattr(config, "dropout", 0.0)),
            batch_first=True,
        ).to(self._device)
        self.global_net = global_net
        self.microbatch = int(config.microbatch)
        if self.microbatch <= 0:
            raise ValueError(
                f"config.microbatch must be >= 1, got {config.microbatch}"
            )
        self._activation_checkpoint = bool(
            getattr(config, "activation_checkpoint", False)
        )
        try:
            self.register_buffer(
                "output_baked_flag",
                torch.tensor(0, dtype=torch.uint8),
                persistent=True,
            )
        except Exception:
            pass
        raw_mode = getattr(config, "compile_mode", "disabled")
        mode = str(raw_mode or "").strip()
        normalized_mode = mode.lower()
        disable_compile = normalized_mode in {"", "disabled", "none"}
        compile_mode_arg = normalized_mode if not disable_compile else None
        self.local_net = Gradient.compile(
            self.local_net,
            mode=compile_mode_arg,
            fullgraph=False,
            dynamic=False,
            backend="inductor",
            disable=disable_compile,
        )
        self.global_net = Gradient.compile(
            self.global_net,
            mode=compile_mode_arg,
            fullgraph=False,
            dynamic=False,
            backend="inductor",
            disable=disable_compile,
        )
        self.__config = config
        self._base_dtype: Optional[torch.dtype] = getattr(self, "base_dtype", None)

    @staticmethod
    def _graph_safe_cast(
        x: torch.Tensor, device: torch.device, dtype: Optional[torch.dtype]
    ) -> torch.Tensor:
        target_dtype = dtype or x.dtype
        if x.device != device:
            return x.to(device=device, dtype=target_dtype, non_blocking=True)
        if x.dtype != target_dtype:
            return x.to(dtype=target_dtype)
        return x

    def forward(
        self,
        features: torch.Tensor | TensorDictBase,
        *args: Any,
        labels_flat: Optional[torch.Tensor] = None,
        net_loss: Optional[nn.Module] = None,
        global_loss: Optional[nn.Module] = None,
        local_loss: Optional[nn.Module] = None,
        loss_weights: Optional[
            Union[Tuple[float, float], LossWeightPolicy]
        ] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]] | TensorDictBase:
        td_input: TensorDictBase | None = None
        if isinstance(features, TensorDictBase):
            td_input = features
            td_labels = td_input.get("labels_flat", None)
            td_features = td_input.get("features")
            if td_features is None:
                raise KeyError("TensorDict input requires a 'features' key")
            features = td_features
            if labels_flat is None and td_labels is not None:
                labels_flat = td_labels
            td_net_loss = td_input.get("net_loss", None)
            td_global_loss = td_input.get("global_loss", None)
            td_local_loss = td_input.get("local_loss", None)
            if net_loss is None and td_net_loss is not None:
                net_loss = td_net_loss
            if global_loss is None and td_global_loss is not None:
                global_loss = td_global_loss
            if local_loss is None and td_local_loss is not None:
                local_loss = td_local_loss
            td_loss_weights = td_input.get("loss_weights", None)
            if loss_weights is None and td_loss_weights is not None:
                loss_weights = td_loss_weights
        if features.ndim == 3 and features.shape[1] == 1:
            features = features.reshape(features.shape[0], -1)
        assert features.ndim == 2 and features.shape[1] == self.in_dim
        b = features.shape[0]
        device = self._device
        amp_enabled = device.type != "cpu"
        base_param = next(self.local_net.parameters())
        base_dtype = self._base_dtype or base_param.dtype
        infer_mode = labels_flat is None or (
            net_loss is None and global_loss is None and (local_loss is None)
        )
        num_slices = (b + self.microbatch - 1) // self.microbatch
        token_chunks: List[torch.Tensor] = []
        context_chunks: List[torch.Tensor] = []
        use_activation_checkpoint = bool(self._activation_checkpoint and not infer_mode)
        if not infer_mode:
            self.local_net.train()
            self.global_net.train()
            for idx in range(num_slices):
                s = idx * self.microbatch
                e = min(b, (idx + 1) * self.microbatch)
                x_slice = self._graph_safe_cast(features[s:e], device, base_dtype)
                ctx = AutoCast.float(device) if amp_enabled else AutoCast.suspend(device)
                with ctx:
                    out: Payload = self.local_net(x_slice)
                out_tokens = torch.nan_to_num(
                    out.tokens, nan=0.0, posinf=0.0, neginf=0.0
                ).to(dtype=base_dtype)
                out_context = torch.nan_to_num(
                    out.context, nan=0.0, posinf=0.0, neginf=0.0
                ).to(dtype=base_dtype)
                token_chunks.append(out_tokens)
                context_chunks.append(out_context)
        else:
            self.local_net.eval()
            self.global_net.eval()
            for idx in range(num_slices):
                s = idx * self.microbatch
                e = min(b, (idx + 1) * self.microbatch)
                x_slice = self._graph_safe_cast(features[s:e], device, base_dtype)
                with contextlib.ExitStack() as stack:
                    stack.enter_context(Gradient.inference(self.local_net))
                    stack.enter_context(
                        AutoCast.float(device)
                        if amp_enabled
                        else AutoCast.suspend(device)
                    )
                    out = self.local_net(x_slice)
                out_tokens = torch.nan_to_num(
                    out.tokens, nan=0.0, posinf=0.0, neginf=0.0
                ).to(dtype=base_dtype)
                out_context = torch.nan_to_num(
                    out.context, nan=0.0, posinf=0.0, neginf=0.0
                ).to(dtype=base_dtype)
                token_chunks.append(out_tokens)
                context_chunks.append(out_context)
        tokens = torch.cat(token_chunks, dim=0).to(device=device, dtype=base_dtype)
        context = torch.cat(context_chunks, dim=0).to(device=device, dtype=base_dtype)
        tokens = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
        context = torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        assembled = torch.nan_to_num(context.reshape(b, -1), nan=0.0, posinf=0.0, neginf=0.0)
        if self.is_norm_linear and self.linear_branch is not None:
            bl = self.linear_branch(
                self._graph_safe_cast(features, self._device, assembled.dtype)
            )
            assembled = assembled + bl
        tokens = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
        t32 = tokens.to(torch.float32)
        tokens_centered = (t32 - t32.mean(dim=1, keepdim=True)).to(dtype=tokens.dtype)
        if infer_mode:
            with Gradient.inference(self.global_net):
                with (
                    AutoCast.float(device)
                    if amp_enabled
                    else AutoCast.suspend(device)
                ):
                    refined_tokens, _ = self.global_net(tokens_centered)
            refined_tokens = torch.nan_to_num(
                refined_tokens, nan=0.0, posinf=0.0, neginf=0.0
            )
            decode_tokens = refined_tokens.detach().clone()
            with Gradient.inference(self.local_net):
                with (
                    AutoCast.float(device)
                    if amp_enabled
                    else AutoCast.suspend(device)
                ):
                    residual_context = self.local_net.decode(
                        decode_tokens, apply_norm=True
                    )
            residual_context = torch.nan_to_num(
                residual_context, nan=0.0, posinf=0.0, neginf=0.0
            )
        else:
            with torch.enable_grad():

                def _global_tokens(inp: torch.Tensor) -> torch.Tensor:
                    with (
                        AutoCast.float(device)
                        if amp_enabled
                        else AutoCast.suspend(device)
                    ):
                        out, _ = self.global_net(inp)
                    return out

                if use_activation_checkpoint and activation_checkpoint is not None:
                    refined_tokens = activation_checkpoint(_global_tokens, tokens_centered)
                else:
                    refined_tokens = _global_tokens(tokens_centered)
                refined_tokens = torch.nan_to_num(
                    refined_tokens, nan=0.0, posinf=0.0, neginf=0.0
                )

                def _decode_tokens(inp: torch.Tensor) -> torch.Tensor:
                    with (
                        AutoCast.float(device)
                        if amp_enabled
                        else AutoCast.suspend(device)
                    ):
                        return self.local_net.decode(inp, apply_norm=True)

                if use_activation_checkpoint and activation_checkpoint is not None:
                    residual_context = activation_checkpoint(_decode_tokens, refined_tokens)
                else:
                    residual_context = _decode_tokens(refined_tokens)
                residual_context = torch.nan_to_num(
                    residual_context, nan=0.0, posinf=0.0, neginf=0.0
                )
        residual = torch.nan_to_num(
            residual_context.reshape(b, -1), nan=0.0, posinf=0.0, neginf=0.0
        )
        if residual.dtype != assembled.dtype:
            residual = residual.to(dtype=assembled.dtype)
        y_hat = torch.nan_to_num(
            assembled + residual, nan=0.0, posinf=0.0, neginf=0.0
        )
        is_cls_loss = (
            isinstance(net_loss, (nn.CrossEntropyLoss, nn.NLLLoss))
            if net_loss is not None
            else False
        )
        y_hat_out = y_hat
        loss_val: Optional[torch.Tensor] = None
        if labels_flat is not None and (
            global_loss is not None or local_loss is not None
        ):
            controller: Optional[LossWeightPolicy] = None
            weights: Tuple[float, float]
            if loss_weights is None:
                weights = (1.0, 0.0)
            elif isinstance(loss_weights, (tuple, list)):
                seq = list(loss_weights)
                if len(seq) != 2:
                    raise ValueError("loss_weights requires two values")
                weights = (float(seq[0]), float(seq[1]))
            else:
                controller = cast(LossWeightPolicy, loss_weights)
                weights = controller.weights()
            tgt = labels_flat.to(device=y_hat_out.device, dtype=y_hat_out.dtype)
            total = y_hat_out.new_tensor(0.0, dtype=y_hat_out.dtype)
            top_component: Optional[torch.Tensor] = None
            bottom_component: Optional[torch.Tensor] = None
            y_top = y_hat_out
            y_bot = assembled.to(device=y_hat_out.device, dtype=y_hat_out.dtype)
            if global_loss is not None:
                top_component = global_loss(y_top, tgt)
                total = total + weights[0] * top_component
            if local_loss is not None:
                bottom_component = local_loss(y_bot, tgt)
                total = total + weights[1] * bottom_component
            if controller is not None:
                controller.update(top_component, bottom_component)
            loss_val = total
        elif net_loss is not None and labels_flat is not None:
            if is_cls_loss:
                tgt = labels_flat.to(device=y_hat_out.device).long()
                loss_val = net_loss(y_hat_out, tgt)
            else:
                tgt = labels_flat.to(
                    device=y_hat_out.device, dtype=y_hat_out.dtype
                )
                loss_val = net_loss(y_hat_out, tgt)
        pred = y_hat_out.reshape(b, *self.out_shape)
        if td_input is not None:
            out_td = td_input.clone()
            out_td.set("pred", pred)
            out_td.set("refined_tokens", refined_tokens)
            out_td.set("residual_context", residual_context)
            if loss_val is not None:
                loss_td = loss_val
                if isinstance(loss_td, torch.Tensor) and loss_td.ndim == 0:
                    batch_size = tuple(out_td.batch_size)
                    if len(batch_size):
                        loss_td = loss_td.expand(batch_size)
                out_td.set("loss_total", loss_td)
            else:
                with contextlib.suppress(KeyError):
                    out_td.del_("loss_total")
            return out_td
        return (pred, loss_val)

    @staticmethod
    def flatten_labels(
        labels: Sequence[torch.Tensor],
        *args: Any,
        dtype: Optional[torch.dtype] = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        out = torch.stack([label.reshape(-1) for label in labels], dim=0)
        if dtype is not None:
            out = out.to(dtype=dtype)
        if pin_memory and out.device.type == "cpu":
            out = out.pin_memory()
        return (out.contiguous(), tuple(labels[0].shape))

    @staticmethod
    def unflatten_labels(
        flat: torch.Tensor, shape: Sequence[int]
    ) -> torch.Tensor:
        return flat.reshape(flat.shape[0], *shape).contiguous()
