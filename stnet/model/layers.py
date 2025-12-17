# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import math
import os
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

try:
    from torch.nn import StochasticDepth as _TorchStochasticDepth
except Exception:

    class StochasticDepth(nn.Module):
        def __init__(self, p: float = 0.0, mode: str = "row") -> None:
            super().__init__()
            self.p = float(p)
            self.mode = str(mode)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if (not self.training) or self.p <= 0.0:
                return x
            keep = 1.0 - self.p
            if keep <= 0.0:
                return torch.zeros_like(x)
            if self.mode == "row" and x.dim() >= 2:
                noise_shape = (x.shape[0], *([1] * (x.dim() - 1)))
                noise = x.new_empty(noise_shape).bernoulli_(keep).div_(keep)
            else:
                noise = x.new_empty(x.shape).bernoulli_(keep).div_(keep)
            return x * noise

else:
    StochasticDepth = _TorchStochasticDepth

_Norm = nn.LayerNorm

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    _HAS_FLEX_ATTENTION = True
except Exception:
    create_block_mask = None
    flex_attention = None
    _HAS_FLEX_ATTENTION = False

if os.environ.get("STNET_DISABLE_FLEX_ATTENTION") in {"1", "true", "True"}:
    _HAS_FLEX_ATTENTION = False

_FLEX_ATTENTION_KWARGS: set[str] = set()
if _HAS_FLEX_ATTENTION and flex_attention is not None:
    with contextlib.suppress(Exception):
        import inspect

        _FLEX_ATTENTION_KWARGS = set(inspect.signature(flex_attention).parameters.keys())

_DILATED_MASK_CACHE_MAX = 32
_FLEX_BLOCK_MASK_CACHE_MAX = 16

_DILATED_MASK_CACHE_MAX_L = int(os.environ.get("STNET_DILATED_MASK_CACHE_MAX_L", "4096"))
_DILATED_MASK_CACHE_ENTRY_MAX_BYTES = int(
    os.environ.get("STNET_DILATED_MASK_CACHE_ENTRY_MAX_BYTES", str(64 * 1024 * 1024))
)
_FLEX_BLOCK_MASK_CACHE_EST_MAX_BYTES = int(
    os.environ.get("STNET_FLEX_BLOCK_MASK_CACHE_EST_MAX_BYTES", str(128 * 1024 * 1024))
)

from ..functional.profiler import FLOP_PROFILER
from .kernels import DotProductAttention, MultiHeadAttention, MultiScaleRetention


def norm_layer(norm_type: str, dim: int) -> nn.Module:
    norm = str(norm_type).strip().lower()
    if norm in {"ln", "layernorm", "layer_norm", "layer-norm"}:
        return nn.LayerNorm(dim)
    if norm in {"bn", "batchnorm", "batch_norm", "batch-norm"}:
        return nn.BatchNorm1d(dim)
    if norm in {"rms", "rmsnorm", "rms_norm", "rms-norm"}:
        try:
            from torch.nn import RMSNorm

            return RMSNorm(dim)
        except Exception:
            return nn.LayerNorm(dim)
    return nn.LayerNorm(dim)


def reshape_for_mha(x: torch.Tensor, batch: int, heads: int, head_dim: int) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected (B, N, D) tensor for MHA reshape, got shape {tuple(x.shape)}")
    return x.view(batch, -1, heads, head_dim).transpose(1, 2).contiguous()


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
        self.rel_weight = nn.Parameter(torch.zeros(self.nhead, self.coord_dim))
        self._fallback_attn: DotProductAttention | None = None
        if not _HAS_FLEX_ATTENTION:
            self._fallback_attn = DotProductAttention(
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
            reshape_for_mha(q, B, self.nhead, self.head_dim),
            reshape_for_mha(k, B, self.nhead, self.head_dim),
            reshape_for_mha(v, B, self.nhead, self.head_dim),
        )
        if (not _HAS_FLEX_ATTENTION) or (not x.is_cuda):
            attn = self._fallback_attn
            if attn is None:
                attn = DotProductAttention(num_heads=self.nhead, head_dim=self.head_dim)
                self._fallback_attn = attn
            yh = attn(qh, kh, vh, attn_mask=attn_mask, training=self.training)
            return yh.transpose(1, 2).contiguous().view(B, N, self.d_model)

        coords_f32 = coords.to(dtype=torch.float32, device=x.device).contiguous()
        W = self.rel_weight.to(dtype=coords_f32.dtype, device=coords_f32.device)

        block = None
        mask_bias: torch.Tensor | None = None
        mask_bias_kind: str | None = None
        if isinstance(attn_mask, torch.Tensor):
            m = attn_mask
            if m.dtype == torch.bool:
                m = m.to(device=qh.device)
                if m.dim() == 2:
                    if m.shape != (N, N):
                        raise ValueError(f"bool attn_mask shape {tuple(m.shape)} incompatible with (N,N)=({N},{N})")

                    def mask_mod(b: int, h: int, qi: int, kj: int) -> torch.Tensor:
                        return ~m[qi, kj]
                elif m.dim() == 3:
                    if m.shape != (B, N, N):
                        raise ValueError(f"bool attn_mask shape {tuple(m.shape)} incompatible with (B={B},N={N})")

                    def mask_mod(b: int, h: int, qi: int, kj: int) -> torch.Tensor:
                        return ~m[b, qi, kj]
                elif m.dim() == 4:
                    b0, hm, s1, s2 = m.shape
                    if (b0 != B) or (s1 != N) or (s2 != N):
                        raise ValueError(f"bool attn_mask shape {tuple(m.shape)} incompatible with (B={B},N={N})")
                    if hm == 1:

                        def mask_mod(b: int, h: int, qi: int, kj: int) -> torch.Tensor:
                            return ~m[b, 0, qi, kj]
                    elif hm != self.nhead:
                        raise ValueError(
                            f"bool attn_mask head dim {hm} incompatible with nhead={self.nhead}"
                        )
                    else:

                        def mask_mod(b: int, h: int, qi: int, kj: int) -> torch.Tensor:
                            return ~m[b, h, qi, kj]
                else:
                    raise ValueError(f"bool attn_mask rank {m.dim()} not supported")

                _block_size: int
                if N <= 2048:
                    _block_size = 128
                elif N <= 16384:
                    _block_size = 256
                else:
                    _block_size = 512

                block = create_block_mask(
                    mask_mod,
                    B,
                    self.nhead,
                    N,
                    N,
                    device=qh.device,
                    BLOCK_SIZE=_block_size,
                )
            else:
                bias = m.to(device=qh.device, dtype=qh.dtype)
                if bias.dim() == 2:
                    if bias.shape != (N, N):
                        raise ValueError(f"attn_mask shape {tuple(bias.shape)} incompatible with (N,N)=(~,{N})")
                    mask_bias_kind = "2d"
                elif bias.dim() == 3:
                    if bias.shape != (B, N, N):
                        raise ValueError(f"attn_mask shape {tuple(bias.shape)} incompatible with (B,N,N)=({B},{N},{N})")
                    mask_bias_kind = "3d"
                elif bias.dim() == 4:
                    b0, hm, s1, s2 = bias.shape
                    if (b0 != B) or (s1 != N) or (s2 != N):
                        raise ValueError(f"attn_mask shape {tuple(bias.shape)} incompatible with (B={B},N={N})")
                    if hm == 1:
                        mask_bias_kind = "4d1"
                    elif hm != self.nhead:
                        raise ValueError(
                            f"attn_mask head dim {hm} incompatible with nhead={self.nhead}"
                        )
                    else:
                        mask_bias_kind = "4d"
                else:
                    raise ValueError(f"attn_mask rank {bias.dim()} not supported")
                mask_bias = bias.contiguous()

        Bc, Nc, Cc = coords_f32.shape
        if (Bc != B) or (Nc != N) or (Cc != self.coord_dim):
            raise ValueError(
                f"coords_f32 shape {coords_f32.shape} incompatible with (B,N,C)=({B},{N},{self.coord_dim})"
            )
        coord_proj = torch.matmul(coords_f32, W.t()).transpose(1, 2).contiguous()

        def score_mod(
            score: torch.Tensor, b: int, h: int, qi: int, kj: int
        ) -> torch.Tensor:
            total = score
            coord_term = coord_proj[b, h, qi] - coord_proj[b, h, kj]
            total = total + coord_term.to(dtype=score.dtype)
            if mask_bias is not None:
                if mask_bias_kind == "2d":
                    total = total + mask_bias[qi, kj].to(dtype=score.dtype)
                elif mask_bias_kind == "3d":
                    total = total + mask_bias[b, qi, kj].to(dtype=score.dtype)
                elif mask_bias_kind == "4d1":
                    total = total + mask_bias[b, 0, qi, kj].to(dtype=score.dtype)
                else:
                    total = total + mask_bias[b, h, qi, kj].to(dtype=score.dtype)
            return total

        scale = 1.0 / math.sqrt(float(self.head_dim))
        out = flex_attention(
            qh,
            kh,
            vh,
            score_mod=score_mod,
            block_mask=block,
            scale=scale,
            enable_gqa=False,
            return_lse=False,
            kernel_options=None,
            return_aux=None,
        )
        H, Dh = self.nhead, self.head_dim
        flops = 2.0 * B * H * N * Dh * N + 2.0 * B * H * N * N * Dh + (B * H * N * N * self.coord_dim)
        try:
            FLOP_PROFILER.add("PatchAttention", float(flops))
        except Exception:
            pass
        return out.transpose(1, 2).contiguous().view(B, N, self.d_model)


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
        h = self.msr(x, attn_mask=attn_mask, state=state, **kwargs)
        if isinstance(h, tuple):
            out, new_state = h
            if new_state is None:
                new_state = state
        else:
            out, new_state = h, state
        return out, new_state


def _get_dilated_mask(
    seq_len: int,
    *args: Any,
    device: Optional[torch.device] = None,
    dilation: int = 1,
    window_size: Optional[int] = None,
    causal: bool = False,
    **kwargs: Any,
) -> torch.Tensor:

    if dilation < 1:
        raise ValueError(f"dilation must be >= 1, got {dilation}")
    L = int(seq_len)
    device = torch.device("cpu") if device is None else torch.device(device)
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
        if embed_dim % max(1, num_heads) != 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
            )

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.dilation = int(dilation)
        self.window_size = window_size
        self.causal = bool(causal)
        self.batch_first = bool(batch_first)
        self.nhead = self.num_heads
        self.head_dim = int(self.embed_dim // max(self.nhead, 1))
        self.dropout_p = float(dropout)
        self.__stf_attention_profile__ = {
            "format": "xs",
            "num_heads": self.nhead,
            "head_dim": self.head_dim,
            "dropout_attr": "dropout_p",
            "effective_window_attr": ["window_size"],
            "include_softmax_scale_dropout": True,
        }
        self.norm1 = _Norm(self.embed_dim)
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = _Norm(self.embed_dim)
        hidden = int(mlp_ratio * self.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, hidden, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.embed_dim, bias=True),
        )
        self.length_bucket_multiple: int = 64

    @staticmethod
    def _device_key(device: torch.device) -> Tuple[str, int]:
        idx = -1
        with contextlib.suppress(Exception):
            if device.index is not None:
                idx = int(device.index)
        return (str(device.type), idx)

    def _get_mask(self, L: int, device: torch.device) -> torch.Tensor:
        if int(L) > _DILATED_MASK_CACHE_MAX_L:
            return _get_dilated_mask(
                int(L),
                dilation=self.dilation,
                window_size=self.window_size,
                causal=self.causal,
                device=device,
            )

        win_key = int(self.window_size) if self.window_size is not None else -1
        key = (
            int(L),
            int(self.dilation),
            win_key,
            int(self.causal),
            self._device_key(device),
        )
        cache = getattr(self, "_mask_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_mask_cache", cache)
        cached = cache.get(key)
        if cached is not None:
            return cached
        mask = _get_dilated_mask(
            int(L),
            dilation=self.dilation,
            window_size=self.window_size,
            causal=self.causal,
            device=device,
        )

        with contextlib.suppress(Exception):
            mask_bytes = int(mask.numel()) * int(mask.element_size())
            if mask_bytes > _DILATED_MASK_CACHE_ENTRY_MAX_BYTES:
                return mask

        if key not in cache and len(cache) >= _DILATED_MASK_CACHE_MAX:
            cache.pop(next(iter(cache)))
        cache[key] = mask
        return mask

    def _get_flex_block_mask(
        self,
        B: int,
        H: int,
        L_q: int,
        L_k: int,
        *,
        device: torch.device,
        block_size: int,
        win: Optional[int],
    ) -> Any:
        win_key = int(win) if win is not None else -1
        key = (
            self._device_key(device),
            int(B),
            int(H),
            int(L_q),
            int(L_k),
            int(block_size),
            int(self.dilation),
            win_key,
            int(self.causal),
        )
        cache = getattr(self, "_flex_block_mask_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_flex_block_mask_cache", cache)
        cached = cache.get(key)
        if cached is not None:
            return cached

        try:
            est_bool_bytes = int(B) * int(H) * int(L_q) * int(L_k)
        except Exception:
            est_bool_bytes = _FLEX_BLOCK_MASK_CACHE_EST_MAX_BYTES + 1
        skip_cache = est_bool_bytes > _FLEX_BLOCK_MASK_CACHE_EST_MAX_BYTES

        def _mask_mod(b, h, q_idx, kv_idx):
            dq = q_idx - kv_idx
            keep = torch.ones_like(dq, dtype=torch.bool)
            if self.causal:
                keep &= (kv_idx <= q_idx)
            if win is not None:
                keep &= (dq.abs() <= win)
            if self.dilation > 1:
                keep &= ((dq % self.dilation) == 0)
            return keep

        block_mask = create_block_mask(
            _mask_mod,
            B,
            H,
            L_q,
            L_k,
            device=device,
            BLOCK_SIZE=int(block_size),
        )

        if skip_cache:
            return block_mask

        if key not in cache and len(cache) >= _FLEX_BLOCK_MASK_CACHE_MAX:
            cache.pop(next(iter(cache)))
        cache[key] = block_mask
        return block_mask

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        transposed = False
        if not self.batch_first:
            x = x.transpose(0, 1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.transpose(0, 1)
            transposed = True

        B, L, D = x.shape
        if D != self.embed_dim:
            raise ValueError(f"x.shape[-1]={D} must match embed_dim={self.embed_dim}")

        kpm: Optional[torch.Tensor] = None
        if key_padding_mask is not None:
            if key_padding_mask.shape[0] != B or key_padding_mask.shape[1] != L:
                raise ValueError(
                    f"key_padding_mask must be (B, L)=({B},{L}), got {tuple(key_padding_mask.shape)}"
                )
            kpm = key_padding_mask
            if kpm.dtype is not torch.bool:
                kpm = kpm.to(torch.bool)

        base_mult = max(int(getattr(self, "length_bucket_multiple", 64)), 1)
        if L <= 512:
            mult = base_mult
        elif L <= 2048:
            mult = base_mult * 2
        else:
            mult = base_mult * 4
        mult = max(mult, 1)

        L_k = int(((L + mult - 1) // mult) * mult)
        pad_len = L_k - L

        x_k = x
        kpm_k: Optional[torch.Tensor] = kpm
        if pad_len > 0:
            pad = x.new_zeros((B, pad_len, D))
            x_k = torch.cat((x, pad), dim=1)
            if kpm is not None:
                pad_mask = torch.ones((B, pad_len), device=kpm.device, dtype=torch.bool)
                kpm_k = torch.cat((kpm, pad_mask), dim=1)
            else:
                kpm_k = torch.zeros((B, L_k), device=x.device, dtype=torch.bool)
                kpm_k[:, -pad_len:] = True

        residual = x_k
        x_k = self.norm1(x_k)

        attn_w: Optional[torch.Tensor] = None

        L_q = L

        if _HAS_FLEX_ATTENTION and x_k.is_cuda:
            qkv = self.qkv(x_k)
            q, k, v = qkv.chunk(3, dim=-1)

            H = self.num_heads
            Dh = self.head_dim

            training = bool(self.training)
            scale = 1.0 / math.sqrt(float(Dh))
            dropout_p = float(self.dropout_p) if training else 0.0

            qh = q[:, :L_q, :].view(B, L_q, H, Dh).transpose(1, 2)
            kh = k.view(B, L_k, H, Dh).transpose(1, 2)
            vh = v.view(B, L_k, H, Dh).transpose(1, 2)

            win = int(self.window_size) if self.window_size is not None else None

            if L_k <= 2048:
                _block_size = 128
            elif L_k <= 16384:
                _block_size = 256
            else:
                _block_size = 512

            max_group = int(getattr(self, "flex_batch_microbatch", 0) or B)
            max_group = max(1, min(B, max_group))

            def _run_flex_for_group(group_size: int) -> torch.Tensor:
                # Pre-allocate output to avoid repeated small allocations + `torch.cat`
                # (helps especially when `group_size` is small due to OOM back-off).
                out_full: Optional[torch.Tensor] = None
                for b0 in range(0, B, group_size):
                    b1 = min(B, b0 + group_size)
                    B_g = b1 - b0

                    qh_g = qh[b0:b1]
                    kh_g = kh[b0:b1]
                    vh_g = vh[b0:b1]

                    kpm_g: Optional[torch.Tensor] = None
                    if kpm_k is not None:
                        kpm_g = kpm_k[b0:b1]

                    if kpm_g is None:
                        block_mask_g = self._get_flex_block_mask(
                            B_g,
                            H,
                            L_q,
                            L_k,
                            device=x_k.device,
                            block_size=_block_size,
                            win=win,
                        )
                    else:

                        def mask_mod_g(b, h, q_idx, kv_idx):
                            dq = q_idx - kv_idx
                            keep = torch.ones_like(dq, dtype=torch.bool)

                            if self.causal:
                                keep &= (kv_idx <= q_idx)

                            if win is not None:
                                keep &= (dq.abs() <= win)

                            if self.dilation > 1:
                                keep &= ((dq % self.dilation) == 0)

                            if kpm_g is not None:
                                is_pad_q = kpm_g[b, q_idx]
                                is_pad_k = kpm_g[b, kv_idx]
                                bad = (is_pad_q | is_pad_k)
                                keep = keep & (~bad)

                            return keep

                        block_mask_g = create_block_mask(
                            mask_mod_g,
                            B_g,
                            H,
                            L_q,
                            L_k,
                            device=x_k.device,
                            BLOCK_SIZE=_block_size,
                        )

                    if flex_attention is None:
                        raise RuntimeError("flex_attention was not imported")

                    if _FLEX_ATTENTION_KWARGS:
                        flex_kwargs: dict[str, Any] = {"block_mask": block_mask_g}
                        if "scale" in _FLEX_ATTENTION_KWARGS:
                            flex_kwargs["scale"] = scale
                        if dropout_p > 0.0:
                            if "dropout_p" in _FLEX_ATTENTION_KWARGS:
                                flex_kwargs["dropout_p"] = dropout_p
                            elif "dropout" in _FLEX_ATTENTION_KWARGS:
                                flex_kwargs["dropout"] = dropout_p
                        y_g = flex_attention(qh_g, kh_g, vh_g, **flex_kwargs)
                    else:
                        try:
                            y_g = flex_attention(
                                qh_g,
                                kh_g,
                                vh_g,
                                block_mask=block_mask_g,
                                scale=scale,
                                dropout_p=dropout_p,
                            )
                        except TypeError:
                            try:
                                y_g = flex_attention(
                                    qh_g,
                                    kh_g,
                                    vh_g,
                                    block_mask=block_mask_g,
                                    scale=scale,
                                )
                            except TypeError:
                                y_g = flex_attention(
                                    qh_g, kh_g, vh_g, block_mask=block_mask_g
                                )
                    out_g = self.out_proj(
                        y_g.transpose(1, 2).contiguous().view(B_g, L_q, self.embed_dim)
                    )
                    if out_full is None:
                        # Fast-path: one group == whole batch.
                        if B_g == B:
                            return out_g
                        out_full = out_g.new_empty((B, *out_g.shape[1:]))

                    out_full[b0:b1] = out_g

                if out_full is None:
                    raise RuntimeError("Internal error: flex_attention produced no outputs")
                return out_full

            group = max_group
            last_oom: Optional[RuntimeError] = None
            while group >= 1:
                try:
                    attn_out = _run_flex_for_group(group)
                    last_oom = None
                    break
                except RuntimeError as e:
                    msg = str(e)
                    if "CUDA out of memory" not in msg and "out of memory" not in msg:
                        raise
                    last_oom = e
                    if x_k.device.type == "cuda":
                        with contextlib.suppress(Exception):
                            torch.cuda.empty_cache()
                    group //= 2

            if last_oom is not None:
                raise last_oom

        else:
            qkv = self.qkv(x_k)
            q, k, v = qkv.chunk(3, dim=-1)

            H = self.num_heads
            Dh = self.head_dim

            qh = q[:, :L_q, :].view(B, L_q, H, Dh).transpose(1, 2)
            kh = k.view(B, L_k, H, Dh).transpose(1, 2)
            vh = v.view(B, L_k, H, Dh).transpose(1, 2)

            base_mask_full = self._get_mask(L_k, x_k.device)
            base_mask = base_mask_full[:L_q, :]

            attn_mask = base_mask
            if kpm_k is not None:
                kpm_b = kpm_k.to(torch.bool)
                attn_mask = (
                    base_mask[None, None, :, :]
                    | kpm_b[:, None, None, :]
                    | kpm_b[:, None, :L_q, None]
                )

            y = F.scaled_dot_product_attention(
                qh,
                kh,
                vh,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
            )
            attn_out = self.out_proj(
                y.transpose(1, 2).contiguous().view(B, L_q, self.embed_dim)
            )

        x_out = x + self.dropout(attn_out)
        res2 = x_out
        x_out = self.norm2(x_out)
        if self.training and torch.is_grad_enabled():
            try:
                x_out = _checkpoint(
                    self.ffn,
                    x_out,
                    use_reentrant=True,
                    preserve_rng_state=True,
                    determinism_check="none",
                )
            except TypeError:
                x_out = _checkpoint(
                    self.ffn,
                    x_out,
                    use_reentrant=True,
                    preserve_rng_state=True,
                )
        else:
            x_out = self.ffn(x_out)
        x_out = res2 + self.dropout(x_out)

        if transposed:
            x_out = x_out.transpose(0, 1)

        if need_weights:
            return x_out, attn_w
        return x_out, None

class CrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        *args: Any,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        drop_path: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.norm_q = norm_layer(norm_type, self.d_model)
        self.norm_kv = norm_layer(norm_type, self.d_model)
        self.attn = MultiHeadAttention(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            dropout=dropout,
            batch_first=True,
            bias=True,
        )
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(p=drop_path, mode="row")

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        qn = self.norm_q(q_tokens)
        kvn = self.norm_kv(kv_tokens)
        ctx, _ = self.attn(qn, kvn, kvn, need_weights=False)
        ctx = self.out_proj(ctx)
        return q_tokens + self.drop_path(self.dropout(ctx))


__all__ = [
    "CrossAttention",
    "DilatedAttention",
    "PatchAttention",
    "Retention",
    "StochasticDepth",
    "norm_layer",
]
