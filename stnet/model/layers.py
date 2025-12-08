# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import math
from collections import deque
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Deque,
    Dict,
    Iterator,
    List,
    Mapping,
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
from torch.utils.checkpoint import checkpoint as activation_checkpoint

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
    from torch.nn.attention.flex_attention import (create_block_mask,
                                                   flex_attention)
    _HAS_FLEX_ATTENTION = True
except Exception:
    _HAS_FLEX_ATTENTION = False

_acc_memory = None
for _name in ("cuda", "xpu", "mps"):
    _module = getattr(torch, _name, None)
    if _module is not None and hasattr(_module, "memory_stats"):
        _acc_memory = _module
        break

from ..backend.compat import torch_no_compile
from ..backend.profiler import FLOP_PROFILER
from ..functional.fx import Autocast, Gradient
from .kernels import (DotProductAttention, MultiHeadAttention,
                      MultiScaleRetention)

if TYPE_CHECKING:
    from .activations import SwiGLU as _SwiGLU


@torch.no_grad()
def _norm_vector(coords: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    B, N, C = coords.shape
    if C == 2:
        z = torch.zeros(B, N, 1, dtype=coords.dtype, device=coords.device)
        coords = torch.cat([coords, z], dim=-1)
    mins = coords.amin(dim=1, keepdim=True)
    maxs = coords.amax(dim=1, keepdim=True)
    rng = (maxs - mins).clamp_min(eps)
    x = (coords - mins) / rng
    return x.clamp_(0.0, 1.0 - 1e-7)

@torch.no_grad()
def _to_z_index(coords01: torch.Tensor, bits: int = 10) -> torch.Tensor:
    def _expand_bits(v: torch.Tensor) -> torch.Tensor:
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8))  & 0x0300F00F
        v = (v | (v << 4))  & 0x030C30C3
        v = (v | (v << 2))  & 0x09249249
        return v

    B, N, _ = coords01.shape
    maxv = (1 << bits) - 1
    xyz = (coords01 * maxv).to(torch.int32)
    x, y, z = xyz.unbind(dim=-1)
    xx = _expand_bits(x)
    yy = _expand_bits(y) << 1
    zz = _expand_bits(z) << 2
    return (xx | yy | zz)

@torch.no_grad()
def _serialize_z_index(
    coords: torch.Tensor,
    *args: Any,
    bits: int,
    patch: int,
    shift_order: bool,
    block_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    coords01 = _norm_vector(coords)
    keys = _to_z_index(coords01, bits=bits)
    perm = keys.argsort(dim=-1, stable=True)                               
    if shift_order and (block_index % 2 == 1):
        B, N = perm.shape
        shift = (patch // 2) % max(N, 1)
        if shift:
            roll = torch.arange(N, device=perm.device).roll(shift)
            perm = perm.gather(1, roll.unsqueeze(0).expand(B, N))
    invperm = torch.empty_like(perm)
    scatter_src = torch.arange(perm.size(1), device=perm.device)
    if scatter_src.dim() < perm.dim():
        scatter_src = scatter_src.view(1, -1).expand_as(perm)
    invperm.scatter_(1, perm, scatter_src)
    return perm, invperm

def _block_ranges(N: int, patch: int) -> "Iterator[tuple[int, int]]":
    for s in range(0, N, patch):
        e = min(s + patch, N)
        yield s, e

                                
                           
                                
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
        if not _HAS_FLEX_ATTENTION:
            attn = DotProductAttention(num_heads=self.nhead, head_dim=self.head_dim)
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

    def _get_mask(self, L: int, device: torch.device) -> torch.Tensor:
        mask_cpu = _get_dilated_mask(
            L,
            dilation=self.dilation,
            window_size=self.window_size,
            causal=self.causal,
        )
        if device.type == "cpu":
            return mask_cpu
        return mask_cpu.to(device=device, non_blocking=True)

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

        if _HAS_FLEX_ATTENTION:
            qkv = self.qkv(x_k)
            q, k, v = qkv.chunk(3, dim=-1)

            H = self.num_heads
            Dh = self.head_dim

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
                outputs: List[torch.Tensor] = []
                for b0 in range(0, B, group_size):
                    b1 = min(B, b0 + group_size)
                    B_g = b1 - b0

                    qh_g = qh[b0:b1]  # (B_g, H, L_q, Dh)
                    kh_g = kh[b0:b1]  # (B_g, H, L_k, Dh)
                    vh_g = vh[b0:b1]  # (B_g, H, L_k, Dh)

                    kpm_g: Optional[torch.Tensor] = None
                    if kpm_k is not None:
                        kpm_g = kpm_k[b0:b1]

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
                            keep &= ~(is_pad_q | is_pad_k)

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

                    y_g = flex_attention(qh_g, kh_g, vh_g, block_mask=block_mask_g)
                    out_g = self.out_proj(
                        y_g.transpose(1, 2).contiguous().view(B_g, L_q, self.embed_dim)
                    )
                    outputs.append(out_g)

                if len(outputs) == 1:
                    return outputs[0]
                return torch.cat(outputs, dim=0)  # type: ignore[return-value]

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
        x_out = self.ffn(x_out)
        x_out = res2 + self.dropout(x_out)

        if transposed:
            x_out = x_out.transpose(0, 1)

        if need_weights:
            return x_out, attn_w
        return x_out, None

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
        self.nhead = int(nhead)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(p=drop_path, mode="row")
        self.norm1 = norm_layer(norm_type, self.d_model)
        self.attn = PatchAttention(self.d_model, self.nhead, coord_dim=self.coord_dim)
        self.norm2 = norm_layer(norm_type, self.d_model)
        hid = int(self.d_model * mlp_ratio * (2.0 / 3.0))
        from .activations import SwiGLU

        self.ffn = SwiGLU(self.d_model, hid, out_dim=self.d_model, dropout=dropout)
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
            if attn_mask.dim() == 4:
                Bm, Hm, Nm1, Nm2 = attn_mask.shape
                if Bm != B or Nm1 != N or Nm2 != N:
                    raise ValueError(
                        f"attn_mask shape {tuple(attn_mask.shape)} incompatible with (B={B},H=?,N={N})"
                    )
                if Hm not in (1, self.nhead):
                    raise ValueError(
                        f"per-head attn_mask H dimension {Hm} must be 1 or match nhead={self.nhead}"
                    )

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
            if isinstance(w, torch.Tensor) and (
                is_meta_or_fake_tensor(w)
                or isinstance(w, dtensor_types)
                or isinstance(w_data, dtensor_types)
            ):
                ln.weight = nn.Parameter(
                    torch.ones(ln.normalized_shape, device=dev, dtype=target_dtype)
                )
            b_data = getattr(b, "data", None)
            if isinstance(b, torch.Tensor) and (
                is_meta_or_fake_tensor(b)
                or isinstance(b, dtensor_types)
                or isinstance(b_data, dtensor_types)
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
        if (
            _x.device.type == "cpu"
            and _x.is_floating_point()
            and _x.dtype != torch.float32
        ):
            _x = _x.float()
        _x = self.norm1(_x)
        if isinstance(_x, torch.Tensor) and is_meta_or_fake_tensor(_x):
            raise RuntimeError("x is meta after LayerNorm")
        if (
            _x.device.type == "cpu"
            and x.is_floating_point()
            and x.dtype != torch.float32
        ):
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


def _coerce_modeling_types(value: Any) -> str:
    mode = str(value).strip().lower()
    normalized = _MODELING_TYPE_ALIASES.get(mode)
    if normalized is None:
        raise ValueError(f"Unsupported modeling type '{value}'")
    return normalized


def stochastic_depth_schedule(drop_path: float, depth: int) -> List[float]:
    if depth <= 0:
        return []
    if drop_path <= 0.0:
        return [0.0 for _ in range(depth)]
    if depth == 1:
        return [float(drop_path)]
    step = float(drop_path) / float(depth - 1)
    return [float(i * step) for i in range(depth)]


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


def is_meta_or_fake_tensor(x: Any) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    if x.is_meta:
        return True
    try:
        from torch._subclasses.fake_tensor import FakeTensor                

        if isinstance(x, FakeTensor):
            return True
    except Exception:
        pass
    try:
        return bool(getattr(x, "fake_mode", None))
    except Exception:
        return False


def reshape_for_mha(x: torch.Tensor, batch: int, heads: int, head_dim: int) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected (B, N, D) tensor for MHA reshape, got shape {tuple(x.shape)}")
    return x.view(batch, -1, heads, head_dim).transpose(1, 2).contiguous()


class SpatialNet(nn.Module):
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
        drops = stochastic_depth_schedule(drop_path, depth)
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
            raise RuntimeError("x is meta/fake before SpatialNet.forward")
        if is_meta_or_fake_tensor(coords):
            raise RuntimeError("coords is meta/fake before SpatialNet.forward")
        if x.dim() != 3:
            raise ValueError(
                f"SpatialNet expects (B, N, C) tokens, got shape {tuple(x.shape)}"
            )
        if coords.dim() != 3:
            raise ValueError(
                f"SpatialNet expects (B, N, D) coords, got shape {tuple(coords.shape)}"
            )
        if x.shape[:2] != coords.shape[:2]:
            raise ValueError(
                "tokens/coords batch or length mismatch: "
                f"tokens={tuple(x.shape)} vs coords={tuple(coords.shape)}"
            )
        x = x.contiguous()
        coords = coords.contiguous()
        if coords.dtype != torch.float32:
            coords = coords.float()
        if attn_mask is not None:
            if is_meta_or_fake_tensor(attn_mask):
                raise RuntimeError(
                    "attn_mask is meta/fake before SpatialNet.forward"
                )
            attn_mask = attn_mask.contiguous()
        for i, blk in enumerate(self.blocks):                                                       
            B, N, D = x.shape
            _patch = getattr(self, "patch_size", 512)
            _shift = getattr(self, "shift_order", True)
            _bits  = getattr(self, "morton_bits", 10)
            perm, invperm = _serialize_z_index(coords, bits=_bits, patch=_patch,
                                                   shift_order=_shift, block_index=i)
            x_s = x.gather(1, perm.unsqueeze(-1).expand(B, N, D))
            c_s = coords.gather(1, perm.unsqueeze(-1).expand(B, N, coords.size(-1)))
            out_s = torch.empty_like(x_s)
                                                
            m_s = None
            if attn_mask is not None:
                                                          
                if attn_mask.dim() == 3:
                    m_s = attn_mask.gather(
                        1, perm.unsqueeze(-1).expand(B, N, N)
                    ).gather(
                        2, perm.unsqueeze(1).expand(B, N, N)
                    )
                elif attn_mask.dim() == 4:
                    if attn_mask.size(1) != 1:
                        raise ValueError("attn_mask with per-head shape not supported here")
                    m_s = attn_mask.squeeze(1).gather(
                        1, perm.unsqueeze(-1).expand(B, N, N)
                    ).gather(
                        2, perm.unsqueeze(1).expand(B, N, N)
                    )
                else:
                    raise ValueError("attn_mask must be (B,N,N)")
            for s, e in _block_ranges(N, _patch):
                xb = x_s[:, s:e, :]
                cb = c_s[:, s:e, :]
                mb = None if m_s is None else m_s[:, s:e, s:e]
                out_s[:, s:e, :] = blk(xb, cb, attn_mask=mb)
            x = out_s.gather(1, invperm.unsqueeze(-1).expand(B, N, D))
        out = self.norm(x)
        if is_meta_or_fake_tensor(out):
            raise RuntimeError("SpatialNet produced meta/fake tensor")
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
        self.d_model = int(d_model)
        self.nhead = int(nhead)

        self.norm1 = norm_layer(norm_type, self.d_model)

        self.retention = Retention(self.d_model, self.nhead)

        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(p=drop_path, mode="row")
        self.norm2 = norm_layer(norm_type, self.d_model)
        hid = int(self.d_model * mlp_ratio * (2.0 / 3.0))
        from .activations import SwiGLU

        self.ffn = SwiGLU(self.d_model, hid, out_dim=self.d_model, dropout=dropout)

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

        h, new_state = self.retention(
            self.norm1(x),
            attn_mask=causal_mask,
            state=state,
        )

        x = x + self.drop_path(self.dropout(h))
        x = x + self.drop_path(self.dropout(self.ffn(self.norm2(x))))
        return x, new_state


class TemporalNet(nn.Module):
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
        drops = stochastic_depth_schedule(drop_path, depth)
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
        from .activations import SwiGLU

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
        if spatial_tokens.dim() != 3 or temporal_tokens.dim() != 3:
            raise ValueError(
                f"CrossTransformer expects 3D tensors, got "
                f"spatial={tuple(spatial_tokens.shape)}, temporal={tuple(temporal_tokens.shape)}"
            )
        Bs, Ns, Ds = spatial_tokens.shape
        Bt, Nt, Dt = temporal_tokens.shape
        if Bs != Bt:
            raise ValueError(
                f"CrossTransformer batch mismatch: spatial B={Bs}, temporal B={Bt}"
            )
        if Ds != Dt:
            raise ValueError(
                f"CrossTransformer hidden dim mismatch: spatial D={Ds}, temporal D={Dt}"
            )
        if self._fixed_mode is not None:
            mode_l = _coerce_modeling_types(self._fixed_mode)
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
        return self._forward_dynamically(spatial_tokens, temporal_tokens, requested)

    def _forward_dynamically(
        self,
        spatial_tokens: torch.Tensor,
        temporal_tokens: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        mode_l = _coerce_modeling_types(mode)
        if mode_l == "ss":
            s_context = self.cross_s(spatial_tokens, temporal_tokens)
            return s_context
        if mode_l == "tt":
            t_context = self.cross_t(temporal_tokens, spatial_tokens)
            return t_context
        s_context = self.cross_s(spatial_tokens, temporal_tokens)
        t_context = self.cross_t(temporal_tokens, spatial_tokens)

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
class Request:
    features: torch.Tensor
    labels_flat: Optional[torch.Tensor] = None
    net_loss: Optional[nn.Module] = None
    global_loss: Optional[nn.Module] = None
    local_loss: Optional[nn.Module] = None
    loss_weights: Optional[Union[Tuple[float, float], "LossWeightPolicy"]] = None
    slice_range: Optional[Tuple[int, int]] = None

@dataclass
class Response:
    pred: torch.Tensor
    loss: Optional[torch.Tensor] = None
    refined_tokens: Optional[torch.Tensor] = None
    residual_context: Optional[torch.Tensor] = None
    slice_range: Optional[Tuple[int, int]] = None


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
        length_bucket_multiple: Optional[int] = None,
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
        self.batch_first = bool(batch_first)
        self._impl = None
        self._using = "fallback"
        self._impl_batch_first = self.batch_first
        layers: List[nn.Module] = []
        dilation = base_dilation
        for _ in range(depth):
            attn = DilatedAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dilation=dilation,
                window_size=window_size,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                causal=causal,
                batch_first=self.batch_first,
            )
            if length_bucket_multiple is not None:
                try:
                    attn.length_bucket_multiple = int(length_bucket_multiple)
                except Exception:
                    pass
            layers.append(attn)
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
        attn_w: Optional[torch.Tensor] = None
        out = x
        need_transpose_fallback = (
            out.dim() == 3
            and (self.batch_first is False)
            and out.shape[0] != out.shape[1]
        )
        if need_transpose_fallback:
            out = out.transpose(0, 1)
        for layer in self.layers:
            out, attn_w = layer(
                out,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
            )
        out = self.norm(out)
        if need_transpose_fallback and out.dim() == 3 and out.shape[0] != out.shape[1]:
            out = out.transpose(0, 1)
        return out, attn_w


class Controller(nn.Module):
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
        length_bucket_multiple: Optional[int] = None,
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
            length_bucket_multiple=length_bucket_multiple,
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


class Processor(nn.Module):
    def __init__(
        self, in_dim: int, out_shape: Sequence[int], config: ModelConfig
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_shape = tuple((int(v) for v in out_shape))
        self.out_dim = int(math.prod(self.out_shape) if self.out_shape else 1)
        self.d_model = int(config.depth)
        self.nhead = int(config.heads)
        self.modeling_type = _coerce_modeling_types(config.modeling_type)
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
            self._get_spatial_coords(self.spatial_tokens, device=torch.device("cpu")),
            persistent=False,
        )
        self.spatial_net = SpatialNet(
            self.d_model,
            self.nhead,
            depth=max(1, int(config.spatial_depth)),
            coord_dim=self.spatial_coords_template.shape[-1],
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            drop_path=self.drop_path,
            norm_type=self.norm_type,
        )
        self.temporal_net = TemporalNet(
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
        self.head_hidden_dim = hid
        self.head = nn.Sequential(
            norm_layer(self.norm_type, self.d_model),
            nn.Linear(self.d_model, hid),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hid, self.out_dim),
        )

    @staticmethod
    def _get_spatial_coords(n_tokens: int, device: torch.device) -> torch.Tensor:
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

    @torch_no_compile(reason='Safe from CUDAGraph error', recursive=True)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        spatial_raw = self.spatial_tokenizer(x)
        expected_spatial = B * self.spatial_tokens * self.d_model
        if spatial_raw.numel() != expected_spatial:
            raise RuntimeError(
                "spatial tokenizer output has unexpected numel: "
                f"got {spatial_raw.numel()} vs expected {expected_spatial}"
            )
        spatial_tokens = spatial_raw.reshape(
            B, self.spatial_tokens, self.d_model
        ).contiguous()
        temporal_raw = self.temporal_tokenizer(x)
        expected_temporal = B * self.temporal_tokens * self.d_model
        if temporal_raw.numel() != expected_temporal:
            raise RuntimeError(
                "temporal tokenizer output has unexpected numel: "
                f"got {temporal_raw.numel()} vs expected {expected_temporal}"
            )
        temporal_tokens = temporal_raw.reshape(
            B, self.temporal_tokens, self.d_model
        ).contiguous()
        try:
            total_tokens = self.spatial_tokens + self.temporal_tokens
            fl_tok = (
                2.0 * float(B) * float(self.in_dim) * float(total_tokens * self.d_model)
            )
            FLOP_PROFILER.add("Tokenizer", float(fl_tok))
        except Exception:
            pass
        coords = self._spatial_coords(B, x.device, spatial_tokens.dtype)
        spatial_out = self.spatial_net(spatial_tokens, coords)
        temporal_out = self.temporal_net(temporal_tokens)
        mode = self._fixed_mode
        if mode is not None:
            mode_l = _coerce_modeling_types(mode)
            match mode_l:
                case 'ss':
                    tokens = spatial_out
                case 'tt':
                    tokens = temporal_out
                case 'ts':
                    if hasattr(self.perception, "_forward_dynamically"):
                        tokens = self.perception._forward_dynamically(
                            temporal_out, spatial_out, "ts"
                        )
                    else:
                        tokens = self.perception(temporal_out, spatial_out, mode="ts")
                case 'st':
                    if hasattr(self.perception, "_forward_dynamically"):
                        tokens = self.perception._forward_dynamically(
                            spatial_out, temporal_out, "st"
                        )
                    else:
                        tokens = self.perception(spatial_out, temporal_out, mode="st")
                case _:
                    raise RuntimeError(f"Unhandled modeling type '{mode}'")
        else:
            tokens = self._forward_dynamically(spatial_out, temporal_out)
        tokens = self.norm(tokens)
        tokens = tokens.contiguous()
        pooled = tokens.mean(dim=1)
        flat = self.head(pooled)
        try:
            hid = int(self.head_hidden_dim)
            fl_head = (
                2.0 * float(B) * float(self.d_model) * float(hid)
                + 2.0 * float(B) * float(hid) * float(self.out_dim)
            )
            FLOP_PROFILER.add("Head", float(fl_head))
        except Exception:
            pass

        flat = flat.contiguous()
        context = flat.reshape(B, *self.out_shape).contiguous()
        dims = tuple(range(1, context.ndim))
        offset = context.mean(dim=dims, keepdim=True).contiguous()
        return (tokens, context)

    def _forward_dynamically(
        self, spatial_out: torch.Tensor, temporal_out: torch.Tensor
    ) -> torch.Tensor:
        mode = getattr(self, "modeling_type", None)
        if mode is None:
            raise RuntimeError("modeling_type is not set")
        mode_l = _coerce_modeling_types(mode)
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
    def weights(self) -> Tuple[float, float]: ...

    def update(
        self,
        top_loss: Optional[torch.Tensor],
        bottom_loss: Optional[torch.Tensor],
    ) -> None:
        raise NotImplementedError


class Scaler(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.calib_mode: str = "none"
        self.register_buffer("x_mean", torch.zeros(1))
        self.register_buffer("x_std", torch.ones(1))
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))
        self.register_buffer("affine_a", torch.ones(1))
        self.register_buffer("affine_b", torch.zeros(1))
        self.register_buffer("pw_x", torch.empty(0))
        self.register_buffer("pw_y", torch.empty(0))

    @torch.no_grad()
    def update_x(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        x_work = x.detach()
        if x_work.dim() == 1:
            x_flat = x_work.view(-1, 1)
        else:
            x_flat = x_work.reshape(-1, x_work.shape[-1])
        mean = x_flat.mean(dim=0)
        std = x_flat.std(dim=0, unbiased=False).clamp_min(self.eps)
        if self.x_mean.shape != mean.shape:
            self.x_mean.resize_(mean.shape)
        if self.x_std.shape != std.shape:
            self.x_std.resize_(std.shape)
        self.x_mean.copy_(mean)
        self.x_std.copy_(std)

    @torch.no_grad()
    def update_y(self, y: torch.Tensor) -> None:
        if y.numel() == 0:
            return
        y_work = y.detach()
        if y_work.dim() == 1:
            y_flat = y_work.view(-1, 1)
        else:
            y_flat = y_work.reshape(-1, y_work.shape[-1])
        mean = y_flat.mean(dim=0)
        std = y_flat.std(dim=0, unbiased=False).clamp_min(self.eps)
        if self.y_mean.shape != mean.shape:
            self.y_mean.resize_(mean.shape)
        if self.y_std.shape != std.shape:
            self.y_std.resize_(std.shape)
        self.y_mean.copy_(mean)
        self.y_std.copy_(std)

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        if x.dim() == 1:
            feat_dim = x.shape[0]
        else:
            feat_dim = x.shape[-1]
        with torch.no_grad():
            if self.x_mean.numel() == 1 and feat_dim != 1:
                self.x_mean.resize_(feat_dim)
                self.x_std.resize_(feat_dim)
                self.x_mean.zero_()
                self.x_std.fill_(1.0)
            elif self.x_mean.numel() != feat_dim:
                raise RuntimeError(
                    "Scaler.normalize_x: feature dimension mismatch: "
                    f"got {feat_dim} features, expected {int(self.x_mean.numel())}"
                )
        if x.dim() == 1:
            return (x - self.x_mean) / (self.x_std + self.eps)
        view_shape = [1] * (x.dim() - 1) + [-1]
        mean = self.x_mean.view(*view_shape)
        std = self.x_std.view(*view_shape)
        return (x - mean) / (std + self.eps)

    def denormalize_x(self, x_scaled: torch.Tensor) -> torch.Tensor:
        if x_scaled.numel() == 0:
            return x_scaled
        if x_scaled.dim() == 1:
            feat_dim = x_scaled.shape[0]
        else:
            feat_dim = x_scaled.shape[-1]
        if self.x_mean.numel() not in (feat_dim, 1):
            raise RuntimeError(
                "Scaler.denormalize_x: feature dimension mismatch: "
                f"got {feat_dim} features, expected {int(self.x_mean.numel())}"
            )
        if x_scaled.dim() == 1:
            return x_scaled * (self.x_std + self.eps) + self.x_mean
        view_shape = [1] * (x_scaled.dim() - 1) + [-1]
        std = self.x_std.view(*view_shape)
        mean = self.x_mean.view(*view_shape)
        return x_scaled * (std + self.eps) + mean

    def _y_stats_vector(self) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.y_mean
        std = self.y_std
        if mean.ndim > 1:
            mean_flat = mean.reshape(-1)
            std_flat = std.reshape(-1)
            with torch.no_grad():
                if self.y_mean.shape != mean_flat.shape:
                    self.y_mean.resize_(mean_flat.shape)
                if self.y_std.shape != std_flat.shape:
                    self.y_std.resize_(std_flat.shape)
                self.y_mean.copy_(mean_flat)
                self.y_std.copy_(std_flat)
            mean = self.y_mean
            std = self.y_std
        return mean, std

    def normalize_y(self, y: torch.Tensor) -> torch.Tensor:
        if y.numel() == 0:
            return y

        orig_shape = y.shape
        if y.dim() == 1:
            y_flat = y.view(1, -1)
            batch_first = False
        else:
            y_flat = y.view(y.shape[0], -1)
            batch_first = True

        mean, std = self._y_stats_vector()

        if mean.numel() == 1 and std.numel() == 1:
            z_flat = (y_flat - mean) / (std + self.eps)
        else:
            if y_flat.shape[1] != mean.numel():
                raise RuntimeError(
                    "Scaler.normalize_y: feature dimension mismatch: "
                    f"got {y_flat.shape[1]} features, expected {int(mean.numel())}"
                )
            z_flat = (y_flat - mean.view(1, -1)) / (std.view(1, -1) + self.eps)

        if batch_first:
            return z_flat.view(orig_shape)
        else:
            return z_flat.view(-1)

    def denormalize_y(self, z: torch.Tensor) -> torch.Tensor:
        if z.numel() == 0:
            return z

        orig_shape = z.shape
        if z.dim() == 1:
            z_flat = z.view(1, -1)
            batch_first = False
        else:
            z_flat = z.view(z.shape[0], -1)
            batch_first = True

        mean, std = self._y_stats_vector()

        if mean.numel() == 1 and std.numel() == 1:
            y_flat = z_flat * std + mean
        else:
            if z_flat.shape[1] != mean.numel():
                raise RuntimeError(
                    "Scaler.denormalize_y: feature dimension mismatch: "
                    f"got {z_flat.shape[1]} features, expected {int(mean.numel())}"
                )
            y_flat = z_flat * std.view(1, -1) + mean.view(1, -1)

        if batch_first:
            return y_flat.view(orig_shape)
        else:
            return y_flat.view(-1)

    def calibrate(self, z_raw: torch.Tensor) -> torch.Tensor:
        if self.calib_mode == "piecewise" and self.pw_x.numel() > 0 and self.pw_y.numel() > 0:
            return self._piecewise(z_raw)
        if self.calib_mode in ("affine", "none") and self.affine_a.numel() > 0:
            return self.affine(z_raw)
        return z_raw

    def affine(self, z_raw: torch.Tensor) -> torch.Tensor:
        if self.affine_a.numel() == 0:
            return z_raw
        return z_raw * self.affine_a + self.affine_b

    @torch.no_grad()
    def set_affine(self, a: torch.Tensor, b: torch.Tensor) -> None:
        if self.affine_a.shape != a.shape:
            self.affine_a.resize_(a.shape)
        if self.affine_b.shape != b.shape:
            self.affine_b.resize_(b.shape)
        self.affine_a.copy_(a.to(self.affine_a.device, dtype=self.affine_a.dtype))
        self.affine_b.copy_(b.to(self.affine_b.device, dtype=self.affine_b.dtype))
        self.pw_x.resize_(0)
        self.pw_y.resize_(0)
        self.calib_mode = "affine"

    @torch.no_grad()
    def fit(
        self,
        z_raw: torch.Tensor,
        z_true: torch.Tensor,
        mode: str = "affine",
        num_bins: int = 8,
    ) -> None:
        if mode == "affine":
            self._fit_affine(z_raw, z_true)
            self.calib_mode = "affine"
        elif mode == "piecewise":
            self._fit_piecewise(z_raw, z_true, num_bins=num_bins)
            self.calib_mode = "piecewise"
        else:
            raise ValueError(f"Unsupported calibration mode: {mode}")

    @torch.no_grad()
    def _fit_affine(self, z_raw: torch.Tensor, z_true: torch.Tensor) -> None:
        if z_raw.numel() == 0 or z_true.numel() == 0:
            return

        x = z_raw.detach()
        y = z_true.detach()
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        x = x.reshape(-1, x.shape[-1]).to(dtype=torch.float64)
        y = y.reshape(-1, y.shape[-1]).to(dtype=torch.float64)

        x_mean = x.mean(dim=0)
        y_mean = y.mean(dim=0)
        x_centered = x - x_mean
        y_centered = y - y_mean

        denom = (x_centered * x_centered).sum(dim=0)
        num = (x_centered * y_centered).sum(dim=0)

        denom_safe = denom.clone()
        tiny_mask = denom_safe.abs() < self.eps
        denom_safe[tiny_mask] = 1.0

        a64 = num / denom_safe
        b64 = y_mean - a64 * x_mean

        a64[tiny_mask] = 1.0
        b64[tiny_mask] = 0.0

        a = a64.to(dtype=torch.float32)
        b = b64.to(dtype=torch.float32)

        if self.affine_a.shape != a.shape:
            self.affine_a.resize_(a.shape)
        if self.affine_b.shape != b.shape:
            self.affine_b.resize_(b.shape)
        self.affine_a.copy_(a)
        self.affine_b.copy_(b)

        self.pw_x.resize_(0)
        self.pw_y.resize_(0)

    @torch.no_grad()
    def _fit_piecewise(
        self,
        z_raw: torch.Tensor,
        z_true: torch.Tensor,
        num_bins: int = 8,
    ) -> None:
        if z_raw.numel() == 0 or z_true.numel() == 0:
            return
        if num_bins < 2:
            self._fit_affine(z_raw, z_true)
            self.calib_mode = "affine"
            return

        x = z_raw.detach()
        y = z_true.detach()
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        x = x.reshape(-1, x.shape[-1]).to(dtype=torch.float64)
        y = y.reshape(-1, y.shape[-1]).to(dtype=torch.float64)

        _, C = x.shape
        device = self.affine_a.device
        knots_x = torch.empty(C, num_bins, dtype=torch.float32, device=device)
        knots_y = torch.empty(C, num_bins, dtype=torch.float32, device=device)

        for j in range(C):
            xj = x[:, j]
            yj = y[:, j]
            if xj.numel() == 0:
                knots_x[j] = torch.linspace(-1.0, 1.0, num_bins, device=device)
                knots_y[j] = knots_x[j]
                continue
            xj_sorted, idx = torch.sort(xj)
            yj_sorted = yj[idx]
            idx_q = torch.linspace(
                0,
                max(0, xj_sorted.numel() - 1),
                num_bins,
                dtype=torch.long,
                device=xj_sorted.device,
            )
            qx = xj_sorted[idx_q]
            qy = yj_sorted[idx_q]
            knots_x[j] = qx.to(dtype=torch.float32, device=device)
            knots_y[j] = qy.to(dtype=torch.float32, device=device)

        if self.pw_x.shape != knots_x.shape:
            self.pw_x.resize_(knots_x.shape)
        if self.pw_y.shape != knots_y.shape:
            self.pw_y.resize_(knots_y.shape)
        self.pw_x.copy_(knots_x)
        self.pw_y.copy_(knots_y)

        if self.affine_a.numel() != C:
            self.affine_a.resize_(C)
            self.affine_b.resize_(C)
        self.affine_a.fill_(1.0)
        self.affine_b.zero_()

    def _piecewise(self, z_raw: torch.Tensor) -> torch.Tensor:
        if self.pw_x.numel() == 0 or self.pw_y.numel() == 0:
            return z_raw
        with torch.no_grad():
            if self.pw_x.dim() == 2 and self.pw_y.dim() == 2:
                C_saved, Kx = self.pw_x.shape
                _, Ky = self.pw_y.shape
                K = min(Kx, Ky)
                if Kx != K:
                    self.pw_x = self.pw_x[:, :K]
                if Ky != K:
                    self.pw_y = self.pw_y[:, :K]
                C_target = int(z_raw.shape[-1]) if z_raw.ndim > 0 else C_saved
                if C_saved < C_target:
                    extra_x = self.pw_x[-1:].expand(C_target - C_saved, -1)
                    extra_y = self.pw_y[-1:].expand(C_target - C_saved, -1)
                    self.pw_x = torch.cat([self.pw_x, extra_x], dim=0)
                    self.pw_y = torch.cat([self.pw_y, extra_y], dim=0)
                elif C_saved > C_target:
                    self.pw_x = self.pw_x[:C_target]
                    self.pw_y = self.pw_y[:C_target]

        orig_shape = z_raw.shape
        if z_raw.ndim == 1:
            z = z_raw.unsqueeze(-1)
        else:
            z = z_raw
        z = z.reshape(-1, z.shape[-1])

        _, C = z.shape
        device = z.device
        dtype = z.dtype

        out = torch.empty_like(z)
        for j in range(C):
            xj = z[:, j]
            knots_x = self.pw_x[j].to(device=device, dtype=dtype)
            knots_y = self.pw_y[j].to(device=device, dtype=dtype)
            if knots_x.numel() < 2:
                out[:, j] = xj
                continue
            idx = torch.bucketize(xj, knots_x)
            idx = idx.clamp(1, knots_x.numel() - 1)
            x0 = knots_x[idx - 1]
            x1 = knots_x[idx]
            y0 = knots_y[idx - 1]
            y1 = knots_y[idx]
            t = (xj - x0) / (x1 - x0 + self.eps)
            out[:, j] = y0 + t * (y1 - y0)

        out = out.reshape(orig_shape)
        return out


class History(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("start", torch.zeros(1, dtype=torch.float64), persistent=True)
        self.register_buffer("end", torch.zeros(1, dtype=torch.float64), persistent=True)
        self.timezone: str = "UTC"
        self.register_buffer("peers", torch.zeros(1, dtype=torch.int64), persistent=True)
        self.register_buffer("epochs", torch.zeros(1, dtype=torch.int64), persistent=True)
        self.os: str = ""
        self.kernel: str = ""
        self.cpu: List[str] = []
        self.arch: List[str] = []
        self.ram_gb: int = 0
        self.python: str = ""
        self.backends: List[str] = []

        self.register_buffer("sampled_n", torch.zeros(1, dtype=torch.int64), persistent=True)
        self.register_buffer("sampled_x_mean", torch.zeros(1, dtype=torch.float64), persistent=True)
        self.register_buffer("sampled_x_var", torch.zeros(1, dtype=torch.float64), persistent=True)
        self.register_buffer(
            "sampled_x_min",
            torch.full((1,), float("inf"), dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "sampled_x_max",
            torch.full((1,), float("-inf"), dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer("sampled_y_mean", torch.zeros(1, dtype=torch.float64), persistent=True)
        self.register_buffer("sampled_y_var", torch.zeros(1, dtype=torch.float64), persistent=True)
        self.register_buffer(
            "sampled_y_min",
            torch.full((1,), float("inf"), dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "sampled_y_max",
            torch.full((1,), float("-inf"), dtype=torch.float64),
            persistent=True,
        )

        self.register_buffer("reduced_n", torch.zeros(1, dtype=torch.int64), persistent=True)
        self.register_buffer("reduced_x_mean", torch.zeros(1, dtype=torch.float64), persistent=True)
        self.register_buffer("reduced_x_var", torch.zeros(1, dtype=torch.float64), persistent=True)
        self.register_buffer(
            "reduced_x_min",
            torch.full((1,), float("inf"), dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "reduced_x_max",
            torch.full((1,), float("-inf"), dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer("reduced_y_mean", torch.zeros(1, dtype=torch.float64), persistent=True)
        self.register_buffer("reduced_y_var", torch.zeros(1, dtype=torch.float64), persistent=True)
        self.register_buffer(
            "reduced_y_min",
            torch.full((1,), float("inf"), dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "reduced_y_max",
            torch.full((1,), float("-inf"), dtype=torch.float64),
            persistent=True,
        )
        self._global_step: int = 0
        self._records: List[Dict[str, Any]] = []
        self.max_history_steps: int = 0
        
    @torch.no_grad()
    def start_session(self, start_posix: float, timezone: Optional[str] = None) -> None:
        self.start.fill_(round(float(start_posix), 6))

        if timezone is None or not str(timezone).strip():
            try:
                import datetime
                import time as _time

                now = datetime.datetime.now().astimezone()
                tzinfo = now.tzinfo
                tz_key = getattr(tzinfo, "key", None) if tzinfo is not None else None
                tz_name = tzinfo.tzname(now) if tzinfo is not None else None
                tz_env = None
                with contextlib.suppress(Exception):
                    tz_env = _time.tzname[0]
                tz = tz_key or tz_name or tz_env or "UTC"
                self.timezone = str(tz)
            except Exception:
                self.timezone = "UTC"
        else:
            self.timezone = str(timezone)

    @torch.no_grad()
    def end_session(self, end_posix: float, peers: int) -> None:
        self.end.fill_(round(float(end_posix), 6))
        self.peers.fill_(int(peers))

    @torch.no_grad()
    def set_epochs(self, epochs: int) -> None:
        self.epochs.fill_(max(0, int(epochs)))

    @torch.no_grad()
    def set_system_info(
        self,
        os_name: str,
        kernel: str,
        cpu_list: List[str],
        arch_list: List[str],
        ram_gb: int,
        python_version: str,
        backends: List[str],
    ) -> None:
        import platform

        self.os = str(os_name)
        self.kernel = str(kernel)
        self.python = str(python_version)

        cpu_models: List[str] = []
        arch_norm: List[str] = []
        try:
            try:
                import psutil                
            except Exception:
                psutil = None                            

            if psutil is not None:
                n_cores = psutil.cpu_count(logical=True) or 1
            else:
                n_cores = 1

            model_name: Optional[str] = None
            try:
                with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if "model name" in line or "Hardware" in line:
                            parts = line.split(":", 1)
                            if len(parts) == 2:
                                model_name = parts[1].strip()
                            break
            except Exception:
                pass

            if not model_name:
                model_name = platform.processor() or (cpu_list[0] if cpu_list else "Unknown CPU")

            arch_name = platform.machine() or (arch_list[0] if arch_list else "unknown")

            cpu_models = [str(model_name) for _ in range(int(n_cores))]
            arch_norm = [str(arch_name) for _ in range(int(n_cores))]
        except Exception:
            cpu_models = list(cpu_list)
            arch_norm = list(arch_list)

        self.cpu = cpu_models
        self.arch = arch_norm

        try:
            try:
                import psutil                
            except Exception:
                psutil = None                            

            if psutil is not None:
                total_bytes = psutil.virtual_memory().total
                self.ram_gb = float(round(total_bytes / (1024.0 ** 3), 2))
            else:
                self.ram_gb = float(ram_gb)
        except Exception:
            self.ram_gb = float(ram_gb)

        backend_devices: List[str] = []
        try:
            if torch.cuda.is_available():
                num_cuda = torch.cuda.device_count()
                for idx in range(num_cuda):
                    try:
                        name = torch.cuda.get_device_name(idx)
                    except Exception:
                        name = "CUDA Device"
                    backend_devices.append(f"cuda:{idx}, {name}")

            mps = getattr(torch.backends, "mps", None)
            if mps is not None and getattr(mps, "is_available", None) and torch.backends.mps.is_available():                              
                chip_name = platform.processor() or "Apple Silicon"
                backend_devices.append(f"mps:0, {chip_name}")

            for idx, model_name in enumerate(cpu_models):
                backend_devices.append(f"cpu:{idx}, {model_name}")

        except Exception:
            backend_devices = list(backends)

        self.backends = backend_devices

    @torch.no_grad()
    def record_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *args: Any,
        use_for_sample: bool = True,
        use_for_reduced: bool = True,
        step: Optional[int] = None,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if x.numel() == 0 or y.numel() == 0:
            return
        x_det = x.detach()
        y_det = y.detach()

        def _stats(t: torch.Tensor):
            if t.is_floating_point() or t.is_complex():
                v, m = torch.var_mean(t, correction=0)
                mn, mx = torch.aminmax(t)
            else:
                mn, mx = torch.aminmax(t)
                m = t.sum(dtype=torch.float64) / float(t.numel())
                v = torch.zeros((), dtype=torch.float64, device=m.device)
            return v, m, mn, mx

        xvar_dev, xm_dev, xmin_dev, xmax_dev = _stats(x_det)
        yvar_dev, ym_dev, ymin_dev, ymax_dev = _stats(y_det)

        stats_device = self.sampled_x_mean.device
        xm   = xm_dev.to(device=stats_device, dtype=torch.float64)
        xvar = xvar_dev.to(device=stats_device, dtype=torch.float64)
        xmin = xmin_dev.to(device=stats_device, dtype=torch.float64)
        xmax = xmax_dev.to(device=stats_device, dtype=torch.float64)
        ym   = ym_dev.to(device=stats_device, dtype=torch.float64)
        yvar = yvar_dev.to(device=stats_device, dtype=torch.float64)
        ymin = ymin_dev.to(device=stats_device, dtype=torch.float64)
        ymax = ymax_dev.to(device=stats_device, dtype=torch.float64)
        if use_for_sample:
            n = int(self.sampled_n.item())
            n_new = n + 1
            w_old = n / n_new if n_new > 0 else 0.0
            w_new = 1.0 / n_new
            self.sampled_n.fill_(n_new)
            self.sampled_x_mean.mul_(w_old).add_(xm * w_new)
            self.sampled_x_var.mul_(w_old).add_(xvar * w_new)
            self.sampled_x_min.copy_(torch.minimum(self.sampled_x_min, xmin.view(1)))
            self.sampled_x_max.copy_(torch.maximum(self.sampled_x_max, xmax.view(1)))
            self.sampled_y_mean.mul_(w_old).add_(ym * w_new)
            self.sampled_y_var.mul_(w_old).add_(yvar * w_new)
            self.sampled_y_min.copy_(torch.minimum(self.sampled_y_min, ymin.view(1)))
            self.sampled_y_max.copy_(torch.maximum(self.sampled_y_max, ymax.view(1)))
        if use_for_reduced:
            n = int(self.reduced_n.item())
            n_new = n + 1
            w_old = n / n_new if n_new > 0 else 0.0
            w_new = 1.0 / n_new
            self.reduced_n.fill_(n_new)
            self.reduced_x_mean.mul_(w_old).add_(xm * w_new)
            self.reduced_x_var.mul_(w_old).add_(xvar * w_new)
            self.reduced_x_min.copy_(torch.minimum(self.reduced_x_min, xmin.view(1)))
            self.reduced_x_max.copy_(torch.maximum(self.reduced_x_max, xmax.view(1)))
            self.reduced_y_mean.mul_(w_old).add_(ym * w_new)
            self.reduced_y_var.mul_(w_old).add_(yvar * w_new)
            self.reduced_y_min.copy_(torch.minimum(self.reduced_y_min, ymin.view(1)))
            self.reduced_y_max.copy_(torch.maximum(self.reduced_y_max, ymax.view(1)))
        self._append(
            xm=xm,
            xvar=xvar,
            xmin=xmin,
            xmax=xmax,
            ym=ym,
            yvar=yvar,
            ymin=ymin,
            ymax=ymax,
            batch_size=int(x.shape[0]),
            step=step,
            extra=extra,
        )

    def _append(
        self,
        *args: Any,
        xm: torch.Tensor,
        xvar: torch.Tensor,
        xmin: torch.Tensor,
        xmax: torch.Tensor,
        ym: torch.Tensor,
        yvar: torch.Tensor,
        ymin: torch.Tensor,
        ymax: torch.Tensor,
        batch_size: int,
        step: Optional[int],
        extra: Optional[Mapping[str, Any]],
        **kwargs: Any,
    ) -> None:
        t = int(step) if step is not None else int(self._global_step)
        self._global_step = t + 1

        def _f(val: torch.Tensor) -> float:
            return float(val.item())

        rec: Dict[str, Any] = {
            "timestep": t,
            "batch_size": int(batch_size),
            "batch_x_mean": _f(xm),
            "batch_x_var": _f(xvar),
            "batch_x_min": _f(xmin),
            "batch_x_max": _f(xmax),
            "batch_y_mean": _f(ym),
            "batch_y_var": _f(yvar),
            "batch_y_min": _f(ymin),
            "batch_y_max": _f(ymax),
            "sampled_n": int(self.sampled_n.item()),
            "sampled_x_mean": _f(self.sampled_x_mean),
            "sampled_x_var": _f(self.sampled_x_var),
            "sampled_x_min": _f(self.sampled_x_min),
            "sampled_x_max": _f(self.sampled_x_max),
            "sampled_y_mean": _f(self.sampled_y_mean),
            "sampled_y_var": _f(self.sampled_y_var),
            "sampled_y_min": _f(self.sampled_y_min),
            "sampled_y_max": _f(self.sampled_y_max),
            "reduced_n": int(self.reduced_n.item()),
            "reduced_x_mean": _f(self.reduced_x_mean),
            "reduced_x_var": _f(self.reduced_x_var),
            "reduced_x_min": _f(self.reduced_x_min),
            "reduced_x_max": _f(self.reduced_x_max),
            "reduced_y_mean": _f(self.reduced_y_mean),
            "reduced_y_var": _f(self.reduced_y_var),
            "reduced_y_min": _f(self.reduced_y_min),
            "reduced_y_max": _f(self.reduced_y_max),
        }
        if extra is not None:
            rec["extra"] = dict(extra)
        self._records.append(rec)
        max_steps = int(self.max_history_steps or 0)
        if max_steps > 0 and len(self._records) > max_steps:
            overflow = len(self._records) - max_steps
            if overflow > 0:
                del self._records[:overflow]

    def save(self) -> Sequence[Mapping[str, Any]]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()
        self._global_step = 0

def resize_scaler_buffer(
    model: nn.Module,
    state: Mapping[str, Any],
) -> None:
    from .layers import Scaler

    scaler: Optional[Scaler] = None
    for module in model.modules():
        if isinstance(module, Scaler):
            scaler = module
            break
    if scaler is None:
        return

    view: Mapping[str, Any]
    if "scaler.x_mean" in state or "module.scaler.x_mean" in state:
        view = state
    elif "model" in state and isinstance(state["model"], Mapping):
        view = state["model"]
    else:
        view = state

    buf_names = ("x_mean", "x_std", "y_mean", "y_std", "affine_a", "affine_b", "pw_x", "pw_y")
    prefixes = ("scaler.", "module.scaler.")

    for prefix in prefixes:
        for name in buf_names:
            key = prefix + name
            if key not in view:
                continue
            src = view[key]
            if not isinstance(src, torch.Tensor):
                continue
            buf = getattr(scaler, name, None)
            if not isinstance(buf, torch.Tensor):
                continue
            if tuple(buf.shape) == tuple(src.shape):
                continue
            try:
                buf.resize_(src.shape)
            except Exception:
                new_buf = buf.new_zeros(src.shape)
                buf.resize_(new_buf.shape)
                buf.copy_(new_buf)


class Instance(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_shape: Sequence[int],
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_shape = tuple((int(x) for x in out_shape))
        self.out_dim = int(math.prod(self.out_shape))
        if config.device is not None:
            self._device = torch.device(config.device)
        else:
            if torch.cuda.is_available():
                device_name = "cuda"
            elif (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                device_name = "mps"
            elif (
                getattr(torch, "is_vulkan_available", None)
                and torch.is_vulkan_available()
            ):
                device_name = "vulkan"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                device_name = "xpu"
            else:
                device_name = "cpu"
            self._device = torch.device(device_name)
        self.scaler = Scaler().to(self._device)
        self.logger = History()
        self.is_norm_linear = bool(getattr(config, "use_linear_branch", False))
        self.linear_branch = (
            nn.Linear(self.in_dim, self.out_dim).to(self._device)
            if self.is_norm_linear
            else None
        )
        self.processor = Processor(self.in_dim, self.out_shape, config=config).to(self._device)
        try:
            bucket = int(getattr(config, "length_bucket_multiple", 64))
        except Exception:
            bucket = 64
        controller = Controller(
            int(config.depth),
            int(config.heads),
            depth=max(1, int(getattr(config, "temporal_depth", 1))),
            mlp_ratio=float(getattr(config, "mlp_ratio", 4.0)),
            dropout=float(getattr(config, "dropout", 0.0)),
            batch_first=True,
            length_bucket_multiple=bucket,
        ).to(self._device)
        self.controller = controller
        try:
            self.max_queue = int(getattr(config, "queue_size", 2))
        except Exception:
            self.max_queue = 2
        self.input: Deque[Request] = deque()
        self.output: Deque[Response] = deque()
        self.microbatch = 0
        self._auto_microbatch_pending = True
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

        compile_dynamic = bool(
            getattr(config, "compile_dynamic", normalized_mode == "reduce-overhead")
        )
        compile_cudagraphs = bool(
            getattr(config, "compile_cudagraphs", normalized_mode != "reduce-overhead")
        )
        compile_kwargs: Dict[str, Any] = {}
        if not compile_cudagraphs:
            compile_kwargs["options"] = {"triton.cudagraphs": False}
        self._pad_compiled_microbatch = bool(
            getattr(config, "pad_compiled_microbatch", True)
        ) and (not disable_compile)
        self.processor = Gradient.compile(
            self.processor,
            mode=compile_mode_arg,
            fullgraph=False,
            dynamic=compile_dynamic,
            backend="inductor",
            disable=disable_compile,
            **compile_kwargs,
        )
        self.controller = Gradient.compile(
            self.controller,
            mode=None,
            fullgraph=False,
            dynamic=False,
            backend="inductor",
            disable=True,
        )
        self.__config = config
        self._base_dtype: Optional[torch.dtype] = getattr(self, "base_dtype", None)

    @staticmethod
    def _cast_graph_safe(
        x: torch.Tensor, device: torch.device, dtype: Optional[torch.dtype]
    ) -> torch.Tensor:
        target_dtype = dtype or x.dtype
        if x.device != device:
            return x.to(device=device, dtype=target_dtype, non_blocking=True)
        if x.dtype != target_dtype:
            return x.to(dtype=target_dtype)
        return x

    def _auto_microbatch(self, features: torch.Tensor | TensorDictBase, device: torch.device) -> int:
        dev_t = getattr(device, "type", "cpu")
        if isinstance(features, TensorDictBase):
            X = features.get("features")
        else:
            X = features
        if not isinstance(X, torch.Tensor):
            return 64
        one = X[:1]
        bytes_per_sample = int(one.nelement()) * int(one.element_size())
        fudge = 8
        max_batch = 1 << 14
        free_bytes = None
        match dev_t:
            case 'cuda':
                with contextlib.suppress(Exception):
                    free, _ = torch.cuda.mem_get_info(device)
                    free_bytes = int(free)
            case 'xpu':
                with contextlib.suppress(Exception):
                    props = getattr(torch.xpu, "get_device_properties", None)
                    mem_alloc = getattr(torch.xpu, "memory_allocated", None)
                    if callable(props) and callable(mem_alloc):
                        total = int(props(device).total_memory); used = int(mem_alloc(device))
                        free_bytes = max(0, total - used)
            case 'mps':
                with contextlib.suppress(Exception):
                    from ..backend.system import Memory
                    free_bytes = int(Memory.available() * 0.25)
        if not free_bytes or free_bytes <= 0:
            return 64
        budget = int(free_bytes * 0.90)
        denom = max(1, bytes_per_sample * fudge)
        mb = max(1, min(max_batch, budget // denom))
        return int(mb)

    def forward(
        self,
        features: torch.Tensor | TensorDictBase,
        *args: Any,
        labels_flat: Optional[torch.Tensor] = None,
        net_loss: Optional[nn.Module] = None,
        global_loss: Optional[nn.Module] = None,
        local_loss: Optional[nn.Module] = None,
        loss_weights: Optional[Union[Tuple[float, float], LossWeightPolicy]] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]] | TensorDictBase:
        calibrate_output: bool = bool(kwargs.pop("calibrate_output", True))
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

        x_raw = features
        if x_raw.ndim == 3 and x_raw.shape[1] == 1:
            x_raw = x_raw.reshape(x_raw.shape[0], -1)
        x_scaled = self.scaler.normalize_x(x_raw)

        device = self._device

        is_cls_loss = (
            isinstance(net_loss, (nn.CrossEntropyLoss, nn.NLLLoss))
            if net_loss is not None
            else False
        )
        has_any_loss = (
            (net_loss is not None)
            or (global_loss is not None)
            or (local_loss is not None)
        )
        has_supervision = labels_flat is not None and has_any_loss
        is_train_path = bool(self.training and torch.is_grad_enabled() and has_supervision)
        infer_mode = not is_train_path
        base_param = next(self.processor.parameters())
        base_dtype = self._base_dtype or base_param.dtype
        features = x_scaled.to(device=device, dtype=base_dtype)
        assert features.ndim == 2 and features.shape[1] == self.in_dim
        b = features.shape[0]
        amp_enabled = device.type != "cpu"
        if self._auto_microbatch_pending:
            try:
                mb = self._auto_microbatch(features, device)
                self.microbatch = max(1, int(mb))
            except Exception:
                self.microbatch = max(1, int(getattr(self, "microbatch", 64) or 64))
            self._auto_microbatch_pending = False
        num_slices = (b + self.microbatch - 1) // self.microbatch
        token_chunks: List[torch.Tensor] = []
        context_chunks: List[torch.Tensor] = []
        processed_ranges: List[Tuple[int, int]] = []
        use_activation_checkpoint = bool(self._activation_checkpoint and not infer_mode)
        self.input.clear()
        self.output.clear()
        if is_train_path:
            self.processor.train()
            self.controller.train()
        else:
            self.processor.eval()
            self.controller.eval()

        def _process_one(req: Request) -> None:
            x_slice = req.features
            if req.slice_range is not None:
                processed_ranges.append(req.slice_range)
            if is_train_path:
                ctx = Autocast.float(device) if amp_enabled else Autocast.suspend(device)
                with ctx:
                    tok, ctx_out = self.processor(x_slice)
            else:
                with contextlib.ExitStack() as stack:
                    stack.enter_context(Gradient.inference(self.processor))
                    stack.enter_context(Autocast.float(device) if amp_enabled else Autocast.suspend(device))
                    tok, ctx_out = self.processor(x_slice)
            out_tokens = torch.nan_to_num(tok, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=base_dtype)
            out_context = torch.nan_to_num(ctx_out, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=base_dtype)
            token_chunks.append(out_tokens)
            context_chunks.append(out_context)

        for idx in range(num_slices):
            s = idx * self.microbatch
            e = min(b, (idx + 1) * self.microbatch)
            shard = self._cast_graph_safe(features[s:e], device, base_dtype)
            self.input.append(Request(features=shard, slice_range=(s, e)))
            if len(self.input) >= max(1, int(self.max_queue)):
                _process_one(self.input.popleft())
        while self.input:
            _process_one(self.input.popleft())
        tokens = torch.cat(token_chunks, dim=0).to(device=device, dtype=base_dtype)
        context = torch.cat(context_chunks, dim=0).to(device=device, dtype=base_dtype)
        tokens = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
        context = torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        assembled = torch.nan_to_num(
            context.reshape(b, -1), nan=0.0, posinf=0.0, neginf=0.0
        )
        if self.is_norm_linear and self.linear_branch is not None:
            bl = self.linear_branch(
                self._cast_graph_safe(features, self._device, assembled.dtype)
            )
            assembled = assembled + bl
        tokens = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)

        mean32 = tokens.mean(dim=1, keepdim=True, dtype=torch.float32)
        tokens_centered = (tokens - mean32.to(dtype=tokens.dtype)).contiguous()
        ctrl_mb = 1

        if infer_mode:
            with contextlib.suppress(Exception):
                import torch._dynamo as _dynamo

                _dynamo.graph_break()

            with Gradient.inference(self.controller):

                def _infer_controller(chunk: torch.Tensor) -> torch.Tensor:
                    with (
                        Autocast.float(device)
                        if amp_enabled
                        else Autocast.suspend(device)
                    ):
                        out, _ = self.controller(chunk)
                    return out

                if ctrl_mb < int(b):
                    refined_tokens = torch.cat(
                        [
                            _infer_controller(tokens_centered[s : s + ctrl_mb])
                            for s in range(0, int(b), ctrl_mb)
                        ],
                        dim=0,
                    )
                else:
                    refined_tokens = _infer_controller(tokens_centered)

            refined_tokens = torch.nan_to_num(
                refined_tokens, nan=0.0, posinf=0.0, neginf=0.0
            )
            decode_tokens = refined_tokens.detach().clone()

            with Gradient.inference(self.processor):

                def _infer_decode(chunk: torch.Tensor) -> torch.Tensor:
                    with (
                        Autocast.float(device)
                        if amp_enabled
                        else Autocast.suspend(device)
                    ):
                        return self.processor.decode(chunk, apply_norm=True)

                if ctrl_mb < int(b):
                    residual_context = torch.cat(
                        [
                            _infer_decode(decode_tokens[s : s + ctrl_mb])
                            for s in range(0, int(b), ctrl_mb)
                        ],
                        dim=0,
                    )
                else:
                    residual_context = _infer_decode(decode_tokens)

            residual_context = torch.nan_to_num(
                residual_context, nan=0.0, posinf=0.0, neginf=0.0
            )
        else:
            with torch.enable_grad():

                with contextlib.suppress(Exception):
                    import torch._dynamo as _dynamo

                    _dynamo.graph_break()

                def _global_tokens(inp: torch.Tensor) -> torch.Tensor:
                    with (
                        Autocast.float(device)
                        if amp_enabled
                        else Autocast.suspend(device)
                    ):
                        out, _ = self.controller(inp)
                    return out

                def _run_controller(chunk: torch.Tensor) -> torch.Tensor:
                    return activation_checkpoint(_global_tokens, chunk)

                if ctrl_mb < int(b):
                    refined_tokens = torch.cat(
                        [_run_controller(tokens_centered[s:s + ctrl_mb]) for s in range(0, int(b), ctrl_mb)],
                        dim=0,
                    )
                else:
                    refined_tokens = _run_controller(tokens_centered)
                refined_tokens = torch.nan_to_num(
                    refined_tokens, nan=0.0, posinf=0.0, neginf=0.0
                )

                def _decode_tokens(inp: torch.Tensor) -> torch.Tensor:
                    with (
                        Autocast.float(device)
                        if amp_enabled
                        else Autocast.suspend(device)
                    ):
                        return self.processor.decode(inp, apply_norm=True)

                def _run_decode(chunk: torch.Tensor) -> torch.Tensor:
                    if use_activation_checkpoint:
                        return activation_checkpoint(_decode_tokens, chunk)
                    return _decode_tokens(chunk)

                if ctrl_mb < int(b):
                    residual_context = torch.cat(
                        [_run_decode(refined_tokens[s:s + ctrl_mb]) for s in range(0, int(b), ctrl_mb)],
                        dim=0,
                    )
                else:
                    residual_context = _run_decode(refined_tokens)
                residual_context = torch.nan_to_num(
                    residual_context, nan=0.0, posinf=0.0, neginf=0.0
                )
        residual = torch.nan_to_num(
            residual_context.reshape(b, -1), nan=0.0, posinf=0.0, neginf=0.0
        )
        if residual.dtype != assembled.dtype:
            residual = residual.to(dtype=assembled.dtype)
        y_hat = torch.nan_to_num(assembled + residual, nan=0.0, posinf=0.0, neginf=0.0)

        z_pred_raw = y_hat
        pred = y_hat.reshape(b, *self.out_shape)
        y_hat_for_loss = y_hat
        loss_val: Optional[torch.Tensor] = None
        z_true: Optional[torch.Tensor] = None
        if labels_flat is not None:
            y_true_raw = labels_flat.to(device=y_hat_for_loss.device)
            z_true = self.scaler.normalize_y(y_true_raw)

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
            tgt = z_true.to(
                device=y_hat_for_loss.device, dtype=y_hat_for_loss.dtype
            )
            total = y_hat_for_loss.new_tensor(0.0, dtype=y_hat_for_loss.dtype)
            top_component: Optional[torch.Tensor] = None
            bottom_component: Optional[torch.Tensor] = None
            y_bot = assembled.to(
                device=y_hat_for_loss.device, dtype=y_hat_for_loss.dtype
            )

            if global_loss is not None:
                top_component = global_loss(z_pred_raw, z_true)
                total = total + weights[0] * top_component
            if local_loss is not None:
                bottom_component = local_loss(y_bot, tgt)
                total = total + weights[1] * bottom_component
            if controller is not None:
                controller.update(top_component, bottom_component)
            loss_val = total
        elif net_loss is not None and labels_flat is not None:
            if is_cls_loss:
                tgt = labels_flat.to(device=y_hat_for_loss.device).long()
                loss_val = net_loss(y_hat_for_loss, tgt)
            else:
                if z_true is None:
                    tgt = self.scaler.normalize_y(
                        labels_flat.to(device=y_hat_for_loss.device)
                    )
                else:
                    tgt = z_true.to(
                        device=y_hat_for_loss.device, dtype=y_hat_for_loss.dtype
                    )
                tgt = tgt.to(dtype=y_hat_for_loss.dtype)
                loss_val = net_loss(y_hat_for_loss, tgt)

        if loss_val is not None and not isinstance(loss_val, torch.Tensor):
            loss_val = torch.as_tensor(
                loss_val, device=y_hat_for_loss.device, dtype=y_hat_for_loss.dtype
            )

        if infer_mode and calibrate_output:
            z_cal = self.scaler.calibrate(z_pred_raw)
            pred = self.scaler.denormalize_y(z_cal).reshape(b, *self.out_shape)

        self.output.clear()
        if processed_ranges:
            offset = 0
            for (s, e) in processed_ranges:
                length = int(e - s)
                if length <= 0:
                    continue
                start = offset
                end = offset + length
                self.output.append(
                    Response(
                        pred=pred[start:end],
                        loss=loss_val,
                        refined_tokens=refined_tokens[start:end],
                        residual_context=residual_context[start:end],
                        slice_range=(s, e),
                    )
                )
                offset = end
        else:
            self.output.append(
                Response(
                    pred=pred,
                    loss=loss_val,
                    refined_tokens=refined_tokens,
                    residual_context=residual_context,
                    slice_range=(0, b),
                )
            )

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

    def history(self) -> Sequence[Mapping[str, Any]]:
        run_hist = getattr(self, "_train_history", None)
        if isinstance(run_hist, list):
            return run_hist

        hist = getattr(self, "logger", None)
        if isinstance(hist, History):
            return hist.save()
        return []

    @staticmethod
    def flatten_y(
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
    def unflatten_y(flat: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
        return flat.reshape(flat.shape[0], *shape).contiguous()
