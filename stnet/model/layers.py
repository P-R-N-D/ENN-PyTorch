# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.optimization import (
    DotProductAttention,
    MultiHeadAttention,
    MultiScaleRetention,
)

try:
    from stnet.utils.compat import RMSNorm as _Norm  # type: ignore
except Exception:
    _Norm = nn.LayerNorm


try:
    # 프로파일러가 없어도 동작하게 안전 import
    from stnet.utils.profiler import FLOP_PROFILER  # noqa: F401
except Exception:
    FLOP_PROFILER = None  # noqa: N816


try:
    # Prefer torch.compiler.disable (PyTorch ≥2.5)
    _torch_compile_disable = torch.compiler.disable  # type: ignore[attr-defined]
except Exception:
    try:
        # Fallback for PyTorch 2.0–2.4
        import torch._dynamo as _dynamo  # type: ignore

        _torch_compile_disable = _dynamo.disable  # type: ignore[attr-defined]
    except Exception:

        def _torch_compile_disable(fn=None, *, recursive=False):  # type: ignore
            if fn is None:
                return lambda real_fn: real_fn
            return fn


if not hasattr(torch, "compiler"):
    class _TorchCompilerNamespace:
        @staticmethod
        def disable(fn=None, *, recursive=False):  # type: ignore
            return _torch_compile_disable(fn, recursive=recursive)


    torch.compiler = _TorchCompilerNamespace()  # type: ignore[attr-defined]
elif not hasattr(torch.compiler, "disable"):

    def _compiler_disable_passthrough(fn=None, *, recursive=False):  # type: ignore
        return _torch_compile_disable(fn, recursive=recursive)


    torch.compiler.disable = _compiler_disable_passthrough  # type: ignore[attr-defined]


# rollback: use vanilla LayerNorm in eager/CPU runs
LayerNorm = nn.LayerNorm


def _disable_torch_compile(fn=None, *, recursive: bool = False):
    if fn is None:
        return lambda real_fn: _disable_torch_compile(real_fn, recursive=recursive)
    try:
        return _torch_compile_disable(fn, recursive=recursive)  # type: ignore[misc]
    except TypeError:
        return _torch_compile_disable(fn)


def _disable_module_torch_compile(module: nn.Module) -> nn.Module:
    forward = getattr(module, "forward", None)
    if callable(forward):
        module.forward = _disable_torch_compile(forward)  # type: ignore[assignment]
    return module


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
        return _disable_module_torch_compile(LayerNorm(d_model))
    if kind in {"rmsnorm", "rms_norm", "rms"} and hasattr(nn, "RMSNorm"):
        return _disable_module_torch_compile(nn.RMSNorm(d_model))  # type: ignore[misc]
    if kind in {"batchnorm", "batchnorm1d", "bn", "bn1d"}:
        return _disable_module_torch_compile(nn.BatchNorm1d(d_model))
    return _disable_module_torch_compile(LayerNorm(d_model))


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

    @_disable_torch_compile
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
        self.register_buffer("_zero_mask", torch.empty(0, dtype=torch.bool), persistent=False)

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
        if attn_mask is None:
            zero_mask = self._zero_mask
            needs_init = (
                zero_mask.numel() != N * N
                or zero_mask.shape[-1] != N
                or zero_mask.device != x.device
            )
            if needs_init:
                zero_mask = torch.ones((N, N), dtype=torch.bool, device=x.device)
            else:
                zero_mask = zero_mask.to(device=x.device)
            self._zero_mask = zero_mask
            attn_mask = zero_mask
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
        mask = self._expand_mask(attn_mask, target=rel_bias.shape)
        if mask.dtype == torch.bool:
            min_value = torch.finfo(rel_bias.dtype).min
            additive = (~mask).to(dtype=rel_bias.dtype) * min_value
        else:
            additive = mask.to(dtype=rel_bias.dtype)
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


@_disable_torch_compile
def _build_dilated_mask(
    seq_len: int,
    *,
    dilation: int = 1,
    window_size: Optional[int] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Dilated attention용 boolean mask 생성: shape (L, L), True는 mask-out.
      - (i - j) % dilation == 0
      - window_size가 주어지면 |i - j| <= window_size
      - causal이면 j <= i
    """

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
    return (~allowed).contiguous()  # True = masked


class DilatedAttention(nn.Module):
    """
    Dilated attention 블록. 내부 attention은 프로젝트 표준 MultiHeadAttention 사용.

    Args:
        embed_dim, num_heads, dilation, window_size, causal, dropout, mlp_ratio,
        batch_first, bias

    IO:
        x: (B, L, C) if batch_first
        key_padding_mask: (B, L) bool, True = pad
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dilation: int = 1,
        window_size: Optional[int] = None,
        causal: bool = False,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        batch_first: bool = True,
        bias: bool = True,
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

    @_disable_torch_compile
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

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # CG 친화성: 스텝 경계 호출은 상위 루프/엔진에서 담당 (모듈 내부 호출 금지)
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
