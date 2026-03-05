# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import logging
import math
import os
import uuid
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Self,
    Sequence,
    Tuple,
    TypeVar,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.compat import StochasticDepth
from ..core.concurrency import Mutex
from ..core.datatypes import env_bool, env_float, env_int
from ..core.policies import ATTENTION_POLICY, AttentionBackend
from .graph import (
    is_checkpoint,
    is_compiling,
    is_export_or_trace,
    is_meta_or_fake_tensor,
    is_symbolic,
    torch_compiler_disable,
)
from .kernels import (
    DotProductAttention,
    MultiHeadAttention,
    MultiScaleRetention,
    _attention_math_bshd,
    _FLEX_KWARGS,
    _ensure_flex_kwargs_initialized,
    get_flex_kernel,
    get_kernel_manager,
)
try:
    from torch.nn.attention.flex_attention import create_block_mask

    _HAS_FLEX_ATTENTION_LIB = True
except ImportError:
    create_block_mask = None
    _HAS_FLEX_ATTENTION_LIB = False

_LOGGER = logging.getLogger(__name__)

_DILATED_MASK_CACHE_MAX = 32
_DILATED_MASK_CACHE_MAX_L = env_int("ENN_DILATED_MASK_CACHE_MAX_L", 4096)
_DILATED_MASK_CACHE_ENTRY_MAX_BYTES = env_int(
    "ENN_DILATED_MASK_CACHE_ENTRY_MAX_BYTES", 64 * 1024 * 1024
)
_FLEX_BLOCK_MASK_CACHE_MAX = 16
_FLEX_BLOCK_MASK_CACHE_EST_MAX_BYTES = env_int(
    "ENN_FLEX_BLOCK_MASK_CACHE_EST_MAX_BYTES", 128 * 1024 * 1024
)
_GATE_STATS_CKPT_FWD = env_bool("ENN_GATE_STATS_CKPT_FWD", False)
_Norm = nn.LayerNorm
TCache = TypeVar("TCache")

if env_bool("ENN_DISABLE_FLEX_ATTENTION", False):
    _HAS_FLEX_ATTENTION = False
else:
    _HAS_FLEX_ATTENTION = (
        _HAS_FLEX_ATTENTION_LIB and get_flex_kernel().has_torch_backend
    )


def _get_dilated_mask(
    seq_len: int | torch.SymInt,
    *args: Any,
    device: Optional[torch.device] = None,
    dilation: int = 1,
    window_size: Optional[int] = None,
    causal: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    if dilation < 1:
        raise ValueError(f"dilation must be >= 1, got {dilation}")
    tracing = bool(is_symbolic())
    if not tracing:
        jit = getattr(torch, "jit", None)
        is_tracing_fn = getattr(jit, "is_tracing", None) if jit is not None else None
        is_scripting_fn = getattr(jit, "is_scripting", None) if jit is not None else None
        tracing = bool(
            (is_tracing_fn() if callable(is_tracing_fn) else False)
            or (is_scripting_fn() if callable(is_scripting_fn) else False)
        )
    SymInt = getattr(torch, "SymInt", None)
    is_symint = bool(SymInt is not None and isinstance(seq_len, SymInt))
    L = seq_len if (tracing or is_symint) else int(seq_len)
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
    not_future = (
        (j <= i) if causal else torch.ones_like(congruent, dtype=torch.bool)
    )
    allowed = congruent & within & not_future
    return allowed.contiguous()


def _tensor_stats(
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if t.is_floating_point() or t.is_complex():
        v, m = torch.var_mean(t, correction=0)
        mn, mx = torch.aminmax(t)
        return v, m, mn, mx
    mn, mx = torch.aminmax(t)
    m = t.sum(dtype=torch.float64) / float(t.numel())
    v = torch.zeros((), dtype=torch.float64, device=m.device)
    return v, m, mn, mx


def resize_scaler_buffer(model: nn.Module, state: Mapping[str, Any]) -> None:
    scaler = next(
        (module for module in model.modules() if isinstance(module, Scaler)), None
    )
    if scaler is None:
        return
    view: Mapping[str, Any]
    if "scaler.x_mean" in state or "module.scaler.x_mean" in state:
        view = state
    elif "model" in state and isinstance(state["model"], Mapping):
        view = state["model"]
    else:
        view = state
    buf_names = (
        "x_mean",
        "x_std",
        "y_mean",
        "y_std",
        "y_min",
        "y_max",
        "y_q_low",
        "y_q_high",
        "affine_a",
        "affine_b",
        "pw_x",
        "pw_y",
        "y_out_scale",
        "y_out_bias",
        "y_out_clip_low",
        "y_out_clip_high",
    )
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
                new_buf = buf.detach().new_zeros(src.shape)
                try:
                    scaler._buffers[name] = new_buf
                except Exception:
                    setattr(scaler, name, new_buf)


def norm_layer(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    norm = str(norm_type).strip().lower()
    match norm:
        case "ln" | "layernorm" | "layer_norm" | "layer-norm":
            return nn.LayerNorm(dim, eps=eps)
        case "bn" | "batchnorm" | "batch_norm" | "batch-norm":
            return nn.BatchNorm1d(dim, eps=eps)
        case "rms" | "rmsnorm" | "rms_norm" | "rms-norm":
            try:
                from torch.nn import RMSNorm

                return RMSNorm(dim, eps=eps)
            except Exception:
                return nn.LayerNorm(dim, eps=eps)
        case _:
            return nn.LayerNorm(dim, eps=eps)


class _FlexDilatedMaskMod:
    __slots__ = ("L_q", "L_k", "dilation", "win", "causal")

    def __init__(
        self: Self,
        *args: Any,
        L_q: int,
        L_k: int,
        dilation: int,
        win: Optional[int],
        causal: bool,
    ) -> None:
        self.L_q = L_q if torch.is_tensor(L_q) else int(L_q)
        self.L_k = L_k if torch.is_tensor(L_k) else int(L_k)
        self.dilation = max(1, int(dilation))
        self.win = None if win is None else int(win)
        self.causal = bool(causal)

    def __call__(
        self: Self, b: int, h: int, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        _ = (b, h)
        dq = q_idx - kv_idx
        keep = kv_idx == kv_idx
        L_q = self.L_q
        L_k = self.L_k
        if torch.is_tensor(L_q) or torch.is_tensor(L_k):
            Lq = (
                L_q.to(device=kv_idx.device)
                if torch.is_tensor(L_q)
                else kv_idx.new_tensor(int(L_q))
            )
            Lk = (
                L_k.to(device=kv_idx.device)
                if torch.is_tensor(L_k)
                else kv_idx.new_tensor(int(L_k))
            )
            keep = (
                keep
                & (q_idx >= 0)
                & (q_idx < Lq)
                & (kv_idx >= 0)
                & (kv_idx < Lk)
            )
        else:
            keep = (
                keep
                & (q_idx >= 0)
                & (q_idx < int(L_q))
                & (kv_idx >= 0)
                & (kv_idx < int(L_k))
            )
        if self.causal:
            keep = keep & (kv_idx <= q_idx)
        if self.win is not None:
            keep = keep & (dq.abs() <= int(self.win))
        if self.dilation > 1:
            keep = keep & ((dq % int(self.dilation)) == 0)
        return keep


class _FlexDilatedScoreMod:
    __slots__ = ("L_q", "L_k", "dilation", "win", "causal")

    def __init__(
        self,
        *,
        L_q: int,
        L_k: int,
        dilation: int,
        win: Optional[int],
        causal: bool,
    ) -> None:
        self.L_q = L_q if torch.is_tensor(L_q) else int(L_q)
        self.L_k = L_k if torch.is_tensor(L_k) else int(L_k)
        self.dilation = max(1, int(dilation))
        self.win = None if win is None else int(win)
        self.causal = bool(causal)

    def __call__(
        self,
        score: torch.Tensor,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        _ = (b, h)
        dq = q_idx - kv_idx
        keep = kv_idx == kv_idx

        L_q = self.L_q
        L_k = self.L_k
        if torch.is_tensor(L_q) or torch.is_tensor(L_k):
            Lq = (
                L_q.to(device=kv_idx.device)
                if torch.is_tensor(L_q)
                else kv_idx.new_tensor(int(L_q))
            )
            Lk = (
                L_k.to(device=kv_idx.device)
                if torch.is_tensor(L_k)
                else kv_idx.new_tensor(int(L_k))
            )
            keep = (
                keep
                & (q_idx >= 0)
                & (q_idx < Lq)
                & (kv_idx >= 0)
                & (kv_idx < Lk)
            )
        else:
            keep = (
                keep
                & (q_idx >= 0)
                & (q_idx < int(L_q))
                & (kv_idx >= 0)
                & (kv_idx < int(L_k))
            )

        if self.causal:
            keep = keep & (kv_idx <= q_idx)
        if self.win is not None:
            keep = keep & (dq.abs() <= int(self.win))
        if self.dilation > 1:
            keep = keep & ((dq % int(self.dilation)) == 0)

        neg = score.new_full((), torch.finfo(score.dtype).min)
        return torch.where(keep, score, neg)




class _FlexKeyBiasScoreMod:
    __slots__ = ("bias_bk", "per_batch")

    def __init__(self: Self, bias_bk: torch.Tensor) -> None:
        if (not torch.is_tensor(bias_bk)) or bias_bk.dim() != 2:
            raise ValueError(f"bias_bk must be 2D (B_or_1,K); got {type(bias_bk)} {getattr(bias_bk,'shape',None)}")
        self.bias_bk = bias_bk
        self.per_batch = bool(int(bias_bk.shape[0]) > 1)

    def set_bias(self: Self, bias_bk: torch.Tensor) -> None:
        if (not torch.is_tensor(bias_bk)) or bias_bk.dim() != 2:
            raise ValueError(f"bias_bk must be 2D (B_or_1,K); got {type(bias_bk)} {getattr(bias_bk,'shape',None)}")
        self.bias_bk = bias_bk
        self.per_batch = bool(int(bias_bk.shape[0]) > 1)

    def __call__(
        self: Self,
        score: torch.Tensor,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        del h, q_idx
        bias_bk = self.bias_bk
        K = bias_bk.shape[1]
        kv = kv_idx.to(torch.int64)
        valid = (kv >= 0) & (kv < K)
        kv_safe = kv.clamp(0, K - 1)
        row = b.to(torch.int64)
        if not bool(self.per_batch):
            row = row * 0
        else:
            B0 = bias_bk.shape[0]
            row = row.clamp(0, B0 - 1)
        flat = (row * K + kv_safe).to(torch.int64)
        bias_flat = bias_bk.reshape(-1).contiguous()
        flat_1d = flat.reshape(-1)
        bias = bias_flat.gather(0, flat_1d).reshape(flat.shape)
        bias = torch.where(valid, bias, torch.zeros_like(bias))
        return score + bias


def _coerce_attn_bias_to_bk(
    attn_bias: torch.Tensor, *, B: int, K: int, like: torch.Tensor
) -> torch.Tensor:
    t = attn_bias
    match t.dim():
        case 4:
            if int(t.shape[-1]) != int(K):
                raise RuntimeError(
                    f"attn_bias K mismatch: {tuple(t.shape)} vs K={int(K)}"
                )
            b0 = int(t.shape[0])
            if b0 not in (1, int(B)):
                raise RuntimeError(
                    f"attn_bias B mismatch: {tuple(t.shape)} vs B={int(B)}"
                )
            t = t.reshape(b0, int(K))
        case 2:
            if int(t.shape[1]) != int(K) or int(t.shape[0]) not in (1, int(B)):
                raise RuntimeError(
                    f"attn_bias shape mismatch: {tuple(t.shape)} vs (1|B,K)=({int(B)},{int(K)})"
                )
        case 1:
            if int(t.numel()) != int(K):
                raise RuntimeError(
                    f"attn_bias len mismatch: {int(t.numel())} vs K={int(K)}"
                )
            t = t.view(1, int(K))
        case _:
            raise RuntimeError(
                f"Unsupported attn_bias rank {int(t.dim())}: {tuple(t.shape)}"
            )

    return t.to(device=like.device, dtype=like.dtype, non_blocking=True).contiguous()


def _coerce_attn_bias_to_b11k(
    attn_bias: torch.Tensor, *, B: object, K: object, like: torch.Tensor
) -> torch.Tensor:
    t = attn_bias
    exporting = bool(is_export_or_trace() or is_compiling() or is_symbolic())
    if t.dim() == 4:
        if not exporting:
            if int(t.shape[-1]) != int(K):
                raise RuntimeError(
                    f"attn_bias K mismatch: {tuple(t.shape)} vs K={int(K)}"
                )
            b0 = int(t.shape[0])
            if b0 not in (1, int(B)):
                raise RuntimeError(
                    f"attn_bias B mismatch: {tuple(t.shape)} vs B={int(B)}"
                )
        return (
            t.to(device=like.device, dtype=like.dtype, non_blocking=True)
            .contiguous()
        )

    match t.dim():
        case 2:
            t2 = t
        case 1:
            t2 = t.reshape(1, t.shape[0])
        case _:
            raise RuntimeError(
                f"Unsupported attn_bias rank {int(t.dim())}: {tuple(t.shape)}"
            )

    if not exporting:
        if int(t2.shape[1]) != int(K) or int(t2.shape[0]) not in (1, int(B)):
            raise RuntimeError(
                f"attn_bias shape mismatch: {tuple(t2.shape)} vs (1|B,K)=({int(B)},{int(K)})"
            )

    t4 = t2.reshape(t2.shape[0], 1, 1, t2.shape[1])
    return (
        t4.to(device=like.device, dtype=like.dtype, non_blocking=True)
        .contiguous()
    )

class Retention(nn.Module):
    def __init__(
        self: Self,
        d_model: int,
        nhead: int,
        *args: Any,
        mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _ = (args, kwargs)
        self.msr = MultiScaleRetention(d_model, nhead)
        self.mode = str(mode or "temporal").strip().lower()

    @staticmethod
    def _coerce_mode(mode: Optional[str]) -> str:
        if mode is None:
            return "temporal"
        m = str(mode).strip().lower()
        match m:
            case "t" | "temporal" | "time" | "causal":
                return "temporal"
            case (
                "s"
                | "spatial"
                | "space"
                | "bi"
                | "bidir"
                | "bidirectional"
                | "noncausal"
                | "non-causal"
            ):
                return "spatial"
            case _:
                return m

    def _forward_bidirectional(
        self: Self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        restore_dtype: Optional[torch.dtype] = None
        x_in = x
        if (
            getattr(x_in.device, "type", "cpu") == "mps"
            and x_in.dtype == torch.bfloat16
        ):
            restore_dtype = x_in.dtype
            x_in = x_in.to(torch.float16)
        if x_in.dim() != 3:
            raise ValueError(
                f"Retention(spatial) expects (B,L,D), got {tuple(x_in.shape)}"
            )
        B, L, D = x_in.shape
        tracing = bool(is_symbolic())
        if (not tracing) and L <= 0:
            out0 = x_in.new_zeros(x_in.shape)
            return (
                out0.to(restore_dtype) if restore_dtype is not None else out0
            )
        msr = self.msr
        H = int(msr.nhead)
        Dh = int(msr.head_dim)
        q = msr.q_proj(x_in).view(B, L, H, Dh)
        v = msr.v_proj(x_in).view(B, L, H, Dh)
        v = msr._apply_kpm_to_v(v, attn_mask)
        lam_h = msr._decay_lambda(v.device, v.dtype).to(
            dtype=v.dtype, device=v.device
        )
        state_fwd = msr._scan_causal(v, lam_h)
        state_bwd = msr._scan_causal(v.flip(1), lam_h).flip(1)
        calc_dtype = (
            torch.float32
            if v.dtype in (torch.float16, torch.bfloat16)
            else v.dtype
        )
        bi_state = (
            state_fwd.to(calc_dtype)
            + state_bwd.to(calc_dtype)
            - v.to(calc_dtype)
        ).to(dtype=v.dtype)
        y = (q * bi_state).contiguous().view(B, L, int(msr.d_model))
        y = msr.norm(y)
        if (
            bool(getattr(msr, "use_gate", False))
            and getattr(msr, "g_proj", None) is not None
        ):
            gate = F.silu(msr.g_proj(x_in))
            y = y * gate
        out = msr.o_proj(y)
        if restore_dtype is not None:
            out = out.to(restore_dtype)
        return out

    def forward(
        self: Self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        state: Any = None,
        mode: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Any]:
        _ = kwargs
        eff_mode = self._coerce_mode(
            mode if mode is not None else getattr(self, "mode", None)
        )
        if eff_mode != "spatial":
            h = self.msr(
                x, attn_mask=attn_mask, state=state, return_state=True
            )
            if isinstance(h, tuple):
                out, new_state = h
            else:
                out, new_state = h, state
            if isinstance(new_state, torch.Tensor) and (
                not torch.is_grad_enabled()
            ):
                new_state = new_state.detach()
            return out, new_state
        out = self._forward_bidirectional(x, attn_mask=attn_mask)
        return out, None


class DilatedAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        causal: bool = False,
        window_size: Optional[int] = None,
        dilation: int = 1,
        ffn_ratio: float = 4.0,
        mlp_ratio: Optional[float] = None,
        activation: str = "gelu",
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.nhead = int(num_heads)
        if self.embed_dim % self.nhead != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.nhead})"
            )
        self.head_dim = self.embed_dim // self.nhead
        self.batch_first = bool(batch_first)
        self.causal = bool(causal)
        self.window_size = (
            int(window_size) if window_size is not None else None
        )
        self.dilation = int(dilation) if dilation is not None else 1
        if self.dilation < 1:
            raise ValueError("dilation must be >= 1")
        self.dropout_p = float(dropout)
        self.norm1 = _Norm(self.embed_dim)
        self.norm2 = _Norm(self.embed_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.drop_path = (
            StochasticDepth(drop_path) if drop_path > 0 else nn.Identity()
        )
        self.mha = MultiHeadAttention(
            self.embed_dim,
            self.nhead,
            dropout=self.dropout_p,
            bias=bias,
            batch_first=self.batch_first,
        )
        self.dpa = DotProductAttention(num_heads=self.nhead, head_dim=self.head_dim)
        if mlp_ratio is not None:
            ffn_ratio = mlp_ratio
        hidden = int(self.embed_dim * float(ffn_ratio))
        match activation.lower():
            case "gelu":
                act = nn.GELU()
            case "silu":
                act = nn.SiLU()
            case "relu":
                act = nn.ReLU()
            case _:
                raise ValueError(f"Unsupported activation: {activation}")
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, hidden, bias=bias),
            act,
            nn.Dropout(self.dropout_p),
            nn.Linear(hidden, self.embed_dim, bias=bias),
        )
        self._mask_cache_len: int = 0
        self._mask_cache: Optional[torch.Tensor] = None
        self._mask_cache_device: Optional[torch.device] = None
        self._flex_cache_len: int = 0
        self._flex_cache_B: int = 0
        self._flex_cache_device: Optional[torch.device] = None
        self._flex_block_mask = None
        self._flex_score_cache_len: int = 0
        self._flex_score_mod: Optional[_FlexDilatedScoreMod] = None

    def _get_torch_mha(self) -> Optional[nn.MultiheadAttention]:
        impl = getattr(self.mha, "impl", None)
        torch_mha = getattr(impl, "mha", None)
        return torch_mha

    def _project_qkv_for_flex(
        self, x_bld: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, nn.Linear]:
        torch_mha = self._get_torch_mha()
        if torch_mha is None:
            raise RuntimeError(
                "Flex path requires torch MultiheadAttention backend"
            )
        qkv = F.linear(x_bld, torch_mha.in_proj_weight, torch_mha.in_proj_bias)
        B, L, _ = qkv.shape
        qkv = qkv.view(B, L, 3, self.nhead, self.head_dim).transpose(1, 3)
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]
        out_proj = torch_mha.out_proj
        return q, k, v, out_proj

    def _get_mask(self, L: int | torch.SymInt, device: torch.device) -> torch.Tensor:
        if (
            (not isinstance(L, int))
            or bool(is_export_or_trace())
            or bool(is_compiling())
            or bool(is_symbolic())
        ):
            return _get_dilated_mask(
                L,
                dilation=self.dilation,
                window_size=self.window_size,
                causal=self.causal,
                device=device,
            )

        if (
            self._mask_cache is None
            or self._mask_cache_len != L
            or self._mask_cache_device != device
        ):
            m = _get_dilated_mask(
                L,
                dilation=self.dilation,
                window_size=self.window_size,
                causal=self.causal,
                device=device,
            )
            self._mask_cache = m
            self._mask_cache_len = L
            self._mask_cache_device = device
        return self._mask_cache

    def _get_flex_block_mask(self, L: int, B: int, device: torch.device):
        if is_export_or_trace() or is_compiling():
            return None
        if not _HAS_FLEX_ATTENTION or create_block_mask is None:
            return None
        B = int(B) if not torch.is_tensor(B) else int(B.item())
        if (
            self._flex_block_mask is None
            or self._flex_cache_len != L
            or self._flex_cache_B != B
            or self._flex_cache_device != device
        ):
            mask_mod = _FlexDilatedMaskMod(
                L_q=L,
                L_k=L,
                dilation=self.dilation,
                win=self.window_size,
                causal=self.causal,
            )
            self._flex_block_mask = create_block_mask(
                mask_mod=mask_mod,
                B=B,
                H=self.nhead,
                Q_LEN=L,
                KV_LEN=L,
                device=device,
            )
            if env_bool("ENN_FLEX_VALIDATE_BLOCK_MASK", False):
                with contextlib.suppress(Exception):
                    kv = getattr(self._flex_block_mask, "kv_indices", None)
                    if torch.is_tensor(kv):
                        mn = int(kv.min().item())
                        mx = int(kv.max().item())
                        if mn < 0 or mx >= int(L):
                            raise RuntimeError(
                                f"Flex BlockMask kv_indices out of range: min={mn}, max={mx}, L={int(L)}"
                            )
            self._flex_cache_len = L
            self._flex_cache_B = B
            self._flex_cache_device = device
        return self._flex_block_mask

    def _get_flex_score_mod(self, L: int) -> _FlexDilatedScoreMod:
        if self._flex_score_mod is None or self._flex_score_cache_len != L:
            self._flex_score_mod = _FlexDilatedScoreMod(
                L_q=L,
                L_k=L,
                dilation=self.dilation,
                win=self.window_size,
                causal=self.causal,
            )
            self._flex_score_cache_len = L
        return self._flex_score_mod

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = True,
        cache_position: Optional[torch.Tensor] = None,
        strict_cache: bool = False,
        skip_ffn_checkpoint: bool = False,
        return_attn_mask: bool = False,
    ):
        del cache_position, strict_cache, skip_ffn_checkpoint

        transposed = False
        if not self.batch_first:
            x = x.transpose(0, 1)
            if context is not None:
                context = context.transpose(0, 1)
            transposed = True

        B, L, _ = x.shape
        device = x.device
        compiling = bool(is_compiling())
        exporting = bool(is_export_or_trace()) and (not compiling)

        trace_like = bool(exporting or compiling or bool(is_symbolic()))

        km = None
        k_flex = ""
        k_mha = ""
        k_dpa = ""
        if not trace_like:
            km = get_kernel_manager()
            site = getattr(self, "_enn_kernel_site", None)
            if not isinstance(site, str) or not site:
                site = f"{self.__class__.__name__}@{id(self):x}"
                setattr(self, "_enn_kernel_site", site)
            dev_i = int(device.index) if device.index is not None else 0
            kind = "self" if context is None else "ctx"
            with contextlib.suppress(Exception):
                child_site = getattr(self.dpa, "_enn_kernel_site", None)
                if (not isinstance(child_site, str)) or (not child_site):
                    setattr(self.dpa, "_enn_kernel_site", f"{site}:dilated-{kind}.dpa")
            kbase = f"attn:{site}:dilated-{kind}@{device.type}:{dev_i}"
            k_flex = f"{kbase}:flex"
            k_mha = f"{kbase}:mha"
            k_dpa = f"{kbase}:dpa"

        def _is_finite_tensor(t: Any) -> bool:
            if not isinstance(t, torch.Tensor):
                return True
            if (not t.is_floating_point()) or t.numel() <= 0:
                return True
            with torch.no_grad():
                return bool(torch.isfinite(t).all().item())

        def _validate_out(out: Any) -> bool:
            if isinstance(out, tuple) and out:
                return _is_finite_tensor(out[0])
            return _is_finite_tensor(out)

        x_norm = self.norm1(x)
        kv = x_norm if context is None else self.norm1(context)

        q_pad = key_padding_mask
        if context is not None:
            q_pad = None

        torch_mha = self._get_torch_mha()
        flex_kernel_ok = bool(
            _HAS_FLEX_ATTENTION
            and getattr(get_flex_kernel(), "has_torch_backend", False)
            and (create_block_mask is not None)
        )

        allow_mha = (not (exporting or compiling)) and (km is not None) and (not km.is_dead(k_mha))
        allow_dpa = bool(
            (context is None)
            and (not need_weights)
            and (key_padding_mask is None)
            and (torch_mha is not None)
        ) and ((km is None) or (not km.is_dead(k_dpa)))

        allow_flex = False
        if (
            (context is None)
            and (not need_weights)
            and (key_padding_mask is None)
            and (torch_mha is not None)
            and flex_kernel_ok
            and (not exporting)
            and (not compiling)
        ):
            _ensure_flex_kwargs_initialized()
            flex_supports_score_mod = "score_mod" in _FLEX_KWARGS
            flex_supports_block_mask = "block_mask" in _FLEX_KWARGS
            needs_mask = bool(self.causal) or (self.window_size is not None) or (int(self.dilation) != 1)
            use_score_mod = bool(x_norm.is_cuda and flex_supports_score_mod)
            block_mask_ok = bool(flex_supports_block_mask)
            allow_flex = ((not needs_mask) or use_score_mod or block_mask_ok) and (km is not None) and (not km.is_dead(k_flex))

        plan = ATTENTION_POLICY.plan(
            q=x_norm,
            need_weights=bool(need_weights),
            has_bias=False,
            exporting=exporting,
            compiling=compiling,
            allow_flex=allow_flex,
            allow_mha=allow_mha,
            allow_dpa=allow_dpa,
        )

        attn_mask_keep: Optional[torch.Tensor] = None
        attn_weights = None
        attn_out: torch.Tensor | None = None

        for _ in range(3):
            try:
                if plan.backend == AttentionBackend.FLEX and allow_flex:
                    q, k, v, out_proj = self._project_qkv_for_flex(x_norm)
                    _ensure_flex_kwargs_initialized()
                    flex_supports_score_mod = "score_mod" in _FLEX_KWARGS
                    flex_supports_block_mask = "block_mask" in _FLEX_KWARGS
                    needs_mask = bool(self.causal) or (self.window_size is not None) or (int(self.dilation) != 1)
                    use_score_mod = bool(q.is_cuda and flex_supports_score_mod)
                    score_mod = self._get_flex_score_mod(int(L)) if (needs_mask and use_score_mod) else None
                    block_mask = None
                    if needs_mask and (score_mod is None):
                        if flex_supports_block_mask:
                            block_mask = self._get_flex_block_mask(int(L), int(B), device)
                        if block_mask is None:
                            raise RuntimeError("Flex needs mask (causal/window/dilation) but no score_mod/block_mask available")
                    def _call_flex() -> torch.Tensor:
                        a = get_flex_kernel()(
                            q,
                            k,
                            v,
                            score_mod=score_mod,
                            block_mask=block_mask,
                            scale=None,
                            dropout_p=self.dropout_p if self.training else 0.0,
                            training=bool(self.training),
                        )
                        a = a.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
                        return out_proj(a)

                    if km is not None:
                        attn_out = km.run(
                            k_flex,
                            _call_flex,
                            validate=_validate_out,
                            sticky=True,
                            safe_on_exception=False,
                        )
                    else:
                        attn_out = _call_flex()
                    attn_weights = None
                    break

                if plan.backend == AttentionBackend.DPA and allow_dpa:
                    q, k, v, out_proj = self._project_qkv_for_flex(x_norm)
                    use_is_causal = False
                    mask_keep = None
                    if self.dilation == 1 and self.window_size is None:
                        use_is_causal = bool(self.causal)
                    else:
                        mask_keep = self._get_mask(L, device)
                    dpa_mask = mask_keep
                    if (
                        dpa_mask is not None
                        and dpa_mask.dim() == 2
                        and (exporting or compiling or bool(is_symbolic()))
                    ):
                        dpa_mask = dpa_mask.view(1, 1, L, L)
                    def _call_dpa() -> torch.Tensor:
                        a = self.dpa(
                            q,
                            k,
                            v,
                            attn_mask=dpa_mask,
                            is_causal=use_is_causal,
                            dropout_p=self.dropout_p if self.training else 0.0,
                            training=bool(self.training),
                        )
                        a = a.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
                        return out_proj(a)

                    if km is not None:
                        attn_out = km.run(
                            k_dpa,
                            _call_dpa,
                            validate=_validate_out,
                            sticky=True,
                            safe_on_exception=False,
                        )
                    else:
                        attn_out = _call_dpa()
                    attn_weights = None
                    if mask_keep is not None:
                        attn_mask_keep = mask_keep
                    break

                use_is_causal = False
                attn_mask = None
                if self.dilation == 1 and self.window_size is None:
                    use_is_causal = bool(self.causal)
                else:
                    attn_mask_keep = self._get_mask(L, device)
                    attn_mask = ~attn_mask_keep

                def _call_mha() -> tuple[torch.Tensor, Any]:
                    return self.mha(
                        x_norm,
                        kv,
                        kv,
                        key_padding_mask=key_padding_mask,
                        attn_mask=attn_mask,
                        need_weights=need_weights,
                        average_attn_weights=average_attn_weights,
                        is_causal=use_is_causal,
                    )

                if km is not None:
                    attn_out, attn_weights = km.run(
                        k_mha,
                        _call_mha,
                        validate=_validate_out,
                        sticky=True,
                        safe_on_exception=False,
                    )
                else:
                    attn_out, attn_weights = _call_mha()
                break

            except Exception:
                if plan.backend == AttentionBackend.FLEX:
                    allow_flex = False
                elif plan.backend == AttentionBackend.DPA:
                    allow_dpa = False
                else:
                    allow_mha = False
                plan = ATTENTION_POLICY.plan(
                    q=x_norm,
                    need_weights=bool(need_weights),
                    has_bias=False,
                    exporting=exporting,
                    compiling=compiling,
                    allow_flex=allow_flex,
                    allow_mha=allow_mha,
                    allow_dpa=allow_dpa,
                )
                attn_out = None

        if attn_out is None:
            use_is_causal = False
            attn_mask = None
            if self.dilation == 1 and self.window_size is None:
                use_is_causal = bool(self.causal)
            else:
                attn_mask_keep = self._get_mask(L, device)
                attn_mask = ~attn_mask_keep
            attn_out, attn_weights = self.mha(
                x_norm,
                kv,
                kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                need_weights=need_weights,
                average_attn_weights=average_attn_weights,
                is_causal=use_is_causal,
            )

        if q_pad is not None:
            attn_out = attn_out.masked_fill(q_pad.unsqueeze(-1), 0.0)

        x = x + self.drop_path(self.dropout(attn_out))
        x = x + self.drop_path(self.ffn(self.norm2(x)))

        if transposed:
            x = x.transpose(0, 1)

        if return_attn_mask:
            if attn_mask_keep is None:
                attn_mask_keep = self._get_mask(L, device)
            return x, attn_weights, attn_mask_keep
        return x, attn_weights




class CrossAttention(nn.Module):
    def __init__(
        self: Self,
        d_model: int,
        nhead: int,
        *args: Any,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        bias: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _ = (args, kwargs)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        if self.d_model % max(1, self.nhead) != 0:
            raise ValueError(
                f"d_model {self.d_model} must be divisible by nhead {self.nhead}"
            )
        self.head_dim = int(self.d_model // max(1, self.nhead))
        self.dropout_p = float(dropout)

        def _norm() -> nn.Module:
            nt = str(norm_type or "layernorm").strip().lower()
            if nt in {"rms", "rmsnorm"} and hasattr(nn, "RMSNorm"):
                return nn.RMSNorm(self.d_model)
            return nn.LayerNorm(self.d_model)

        self.norm_q = _norm()
        self.norm_kv = _norm()
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.attn = DotProductAttention(
            num_heads=self.nhead, head_dim=self.head_dim
        )
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self._flex_bias_fail_count: int = 0
        self._flex_bias_fail_max: int = max(
            0, int(env_int("ENN_RESAMPLER_FLEX_RETRY", 3))
        )
        self._disable_flex_bias_runtime: bool = False
        self._flex_keybias_score_mod: Optional[_FlexKeyBiasScoreMod] = None

    def forward(
        self: Self,
        latents: torch.Tensor,
        tokens: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents.dim() != 3 or tokens.dim() != 3:
            raise ValueError(
                "CrossAttention expects latents (B,Lq,D) and tokens (B,Lk,D), "
                f"got {tuple(latents.shape)} and {tuple(tokens.shape)}"
            )
        B = latents.size(0)
        Lq = latents.size(1)
        D = latents.size(2)
        Lk = tokens.size(1)
        Dk = tokens.size(2)
        if (latents.size(-1) != self.d_model) or (tokens.size(-1) != self.d_model):
            raise ValueError(
                f"CrossAttention expects last dim D={self.d_model}, got latents D={D} tokens D={Dk}"
            )

        diag_cross = env_bool("ENN_DIAG_NONFINITE_CROSSATTN", default=False) or env_bool(
            "ENN_DIAG_NONFINITE_PRE_SANITIZE", default=False
        )
        dump_dir = str(os.environ.get("ENN_NONFINITE_DUMP_DIR", "") or "").strip()
        strict = env_bool("ENN_SANITIZE_NAN_STRICT", default=False)
        limit = int(env_int("ENN_CROSSATTN_DUMP_LIMIT", 8) or 8)
        dumped = int(getattr(self, "_enn_cross_nonfinite_dumped", 0) or 0)

        def _stats(x: object) -> object:
            if not torch.is_tensor(x):
                return None
            tt = x.detach()
            out: dict[str, object] = {
                "shape": [int(v) for v in tuple(tt.shape)],
                "dtype": str(tt.dtype),
                "device": str(tt.device),
                "numel": int(tt.numel()),
            }
            if tt.is_floating_point() and tt.numel() > 0:
                with torch.no_grad():
                    flat = tt.reshape(-1)
                    n = int(min(4096, int(flat.numel())))
                    samp = flat[:n]
                    out["absmax"] = float(samp.abs().max().item())
                    out["nan"] = int(torch.isnan(samp).sum().item())
                    out["inf"] = int(torch.isinf(samp).sum().item())
                    out["nonfinite"] = int((~torch.isfinite(samp)).sum().item())
            return out

        def _sample(x: object) -> object:
            if not torch.is_tensor(x):
                return None
            tt = x.detach()
            if tt.numel() <= 0:
                return tt.to("cpu")
            with contextlib.suppress(Exception):
                if tt.dim() >= 1:
                    tt = tt[: min(2, int(tt.shape[0]))]
                if tt.dim() >= 2:
                    tt = tt[:, : min(128, int(tt.shape[1]))]
                if tt.dim() >= 3:
                    tt = tt[:, :, : min(128, int(tt.shape[2]))]
                tt = tt.contiguous()
            return tt.to("cpu")

        def _check(tag: str, x: torch.Tensor, *, extra: dict[str, object] | None = None) -> None:
            nonlocal dumped
            if (not diag_cross) or (not isinstance(x, torch.Tensor)):
                return
            if (not x.is_floating_point()) or x.numel() <= 0:
                return
            with torch.no_grad():
                try:
                    a = x.detach().abs().amax()
                    bad = not bool(torch.isfinite(a).item())
                except Exception:
                    bad = not bool(torch.isfinite(x).all().item())
            if not bad:
                return

            payload = {
                "where": f"CrossAttention.{tag}",
                "x_stats": _stats(x),
                "x_sample": _sample(x),
                "attn_bias_stats": _stats(attn_bias),
                "extra": extra or {},
            }

            path = None
            if dump_dir and ((limit < 0) or (dumped < limit)):
                with contextlib.suppress(Exception):
                    os.makedirs(dump_dir, exist_ok=True)
                    rank = str(os.environ.get("RANK", "0") or "0")
                    rid = uuid.uuid4().hex[:8]
                    safe = "".join((c if (c.isalnum() or c in "._-") else "_") for c in tag)
                    path = os.path.join(dump_dir, f"crossattn_nonfinite.{safe}.rank{rank}.{rid}.pt")
                    torch.save(payload, path)
                    dumped += 1
                    setattr(self, "_enn_cross_nonfinite_dumped", int(dumped))
                    _LOGGER.error("[ENN] CrossAttention non-finite at %s; dumped: %s", tag, str(path))

            if strict:
                raise RuntimeError(f"[ENN] CrossAttention produced non-finite at {tag}. dump={path}")

        q_in = self.norm_q(latents)
        kv_in = self.norm_kv(tokens)

        if diag_cross:
            _check(
                "norm_q",
                q_in,
                extra={
                    "norm_q": {
                        "weight": _stats(getattr(self.norm_q, "weight", None)),
                        "bias": _stats(getattr(self.norm_q, "bias", None)),
                    }
                },
            )
            _check(
                "norm_kv",
                kv_in,
                extra={
                    "norm_kv": {
                        "weight": _stats(getattr(self.norm_kv, "weight", None)),
                        "bias": _stats(getattr(self.norm_kv, "bias", None)),
                    }
                },
            )

        H = int(self.nhead)
        Dh = int(self.head_dim)

        q = self.q_proj(q_in).reshape(B, Lq, H, Dh).transpose(1, 2)
        k = self.k_proj(kv_in).reshape(B, Lk, H, Dh).transpose(1, 2)
        v = self.v_proj(kv_in).reshape(B, Lk, H, Dh).transpose(1, 2)

        if diag_cross:
            _check(
                "q_proj",
                q,
                extra={"q_proj": {"weight": _stats(self.q_proj.weight), "bias": _stats(self.q_proj.bias)}},
            )
            _check(
                "k_proj",
                k,
                extra={"k_proj": {"weight": _stats(self.k_proj.weight), "bias": _stats(self.k_proj.bias)}},
            )
            _check(
                "v_proj",
                v,
                extra={"v_proj": {"weight": _stats(self.v_proj.weight), "bias": _stats(self.v_proj.bias)}},
            )

        compiling = bool(is_compiling())
        exporting = bool(is_export_or_trace()) and (not compiling)
        km = None
        k_mha = ""
        if not (exporting or compiling or bool(is_symbolic())):
            km = get_kernel_manager()
            site = getattr(self, "_enn_kernel_site", None)
            if not isinstance(site, str) or not site:
                site = f"{self.__class__.__name__}@{id(self):x}"
                setattr(self, "_enn_kernel_site", site)
            with contextlib.suppress(Exception):
                child_site = getattr(self.attn, "_enn_kernel_site", None)
                if (not isinstance(child_site, str)) or (not child_site):
                    setattr(self.attn, "_enn_kernel_site", f"{site}:cross.dpa")
            dev_i = int(q.device.index) if q.device.index is not None else 0
            kbase = f"attn:{site}:cross@{q.device.type}:{dev_i}"
            k_mha = f"{kbase}:mha"
        has_bias = attn_bias is not None

        def _is_fp8_tensor(x: torch.Tensor) -> bool:
            try:
                if "float8" in str(x.dtype):
                    return True
                t = type(x)
                mod = getattr(t, "__module__", "") or ""
                name = getattr(t, "__name__", "") or ""
                if "torchao" in mod and ("float8" in mod or "float8" in name.lower()):
                    return True
                if "Float8Tensor" in name:
                    return True
            except Exception:
                pass
            return False

        force_no_flex = bool(q.dtype == torch.float64) or bool(_is_fp8_tensor(q))

        def _classify_flex_failure(exc: BaseException) -> str:
            msg = str(exc)
            tname = type(exc).__name__
            m = msg.lower()
            if "flexattention not available" in m:
                return "struct"
            if "not available" in m and "flex" in m:
                return "struct"
            if "score_mod" in m and ("not supported" in m or "unsupported" in m):
                return "struct"
            if tname in ("TypeError", "ValueError"):
                return "struct"
            if ("shape" in m and "mismatch" in m) or (
                "attn_bias" in m and "mismatch" in m
            ):
                return "struct"
            if exporting or compiling:
                return "struct"
            if "no valid triton configs" in m or "outofresources" in m or "out of resources" in m:
                return "transient"
            if "torch._dynamo" in m or "torch._inductor" in m or "compileerror" in m:
                return "transient"
            return "transient"

        allow_flex = (
            (not force_no_flex)
            and (not exporting)
            and (not compiling)
            and _HAS_FLEX_ATTENTION
        )
        if allow_flex and has_bias:
            _ensure_flex_kwargs_initialized()
            allow_flex = (not self._disable_flex_bias_runtime) and ("score_mod" in _FLEX_KWARGS)

        allow_mha = (not (exporting or compiling)) and (km is not None) and (not km.is_dead(k_mha))
        allow_dpa = True

        plan = ATTENTION_POLICY.plan(
            q=q,
            need_weights=False,
            has_bias=has_bias,
            exporting=exporting,
            compiling=compiling,
            allow_flex=allow_flex,
            allow_mha=allow_mha,
            allow_dpa=allow_dpa,
        )

        attn_proj: torch.Tensor | None = None
        while attn_proj is None:
            if plan.backend == AttentionBackend.FLEX and allow_flex:
                try:
                    _ensure_flex_kwargs_initialized()
                    score_mod = None
                    if has_bias:
                        bias_bk = _coerce_attn_bias_to_bk(
                            attn_bias, B=B, K=Lk, like=q
                        )
                        sm = self._flex_keybias_score_mod
                        if sm is None:
                            sm = _FlexKeyBiasScoreMod(bias_bk)
                            self._flex_keybias_score_mod = sm
                        else:
                            sm.set_bias(bias_bk)
                        score_mod = sm

                    attn_out = get_flex_kernel()(
                        q,
                        k,
                        v,
                        score_mod=score_mod,
                        block_mask=None,
                        scale=None,
                        dropout_p=self.dropout_p if self.training else 0.0,
                        training=bool(self.training),
                        is_causal=False,
                    )
                    attn_out = attn_out.transpose(1, 2).contiguous().view(B, Lq, D)
                    attn_proj = self.out_proj(attn_out)
                except Exception as exc:
                    kind = _classify_flex_failure(exc)
                    self._flex_bias_fail_count += 1
                    if kind == "struct":
                        self._disable_flex_bias_runtime = True
                    elif (
                        self._flex_bias_fail_max > 0
                        and self._flex_bias_fail_count >= self._flex_bias_fail_max
                    ):
                        self._disable_flex_bias_runtime = True
                    allow_flex = False

            if attn_proj is None and plan.backend == AttentionBackend.MHA and allow_mha:
                try:
                    q_seq = q_in.transpose(0, 1)
                    kv_seq = kv_in.transpose(0, 1)

                    in_proj_bias = None
                    if (
                        (self.q_proj.bias is not None)
                        and (self.k_proj.bias is not None)
                        and (self.v_proj.bias is not None)
                    ):
                        in_proj_bias = torch.cat(
                            [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias],
                            dim=0,
                        )

                    attn_mask = None
                    if has_bias:
                        bias_bk = _coerce_attn_bias_to_bk(
                            attn_bias, B=int(B), K=int(Lk), like=q_in
                        )
                        if int(bias_bk.shape[0]) == 1:
                            bias_bk = bias_bk.expand(int(B), int(Lk))
                        bias4 = (
                            bias_bk[:, None, None, :]
                            .expand(int(B), int(H), int(Lq), int(Lk))
                            .contiguous()
                        )
                        attn_mask = bias4.view(int(B) * int(H), int(Lq), int(Lk)).to(
                            dtype=q_in.dtype
                        )

                    attn_out, _ = F.multi_head_attention_forward(
                        q_seq,
                        kv_seq,
                        kv_seq,
                        embed_dim_to_check=int(D),
                        num_heads=int(H),
                        in_proj_weight=None,
                        in_proj_bias=in_proj_bias,
                        bias_k=None,
                        bias_v=None,
                        add_zero_attn=False,
                        dropout_p=self.dropout_p if self.training else 0.0,
                        out_proj_weight=self.out_proj.weight,
                        out_proj_bias=self.out_proj.bias,
                        training=bool(self.training),
                        key_padding_mask=None,
                        need_weights=False,
                        attn_mask=attn_mask,
                        use_separate_proj_weight=True,
                        q_proj_weight=self.q_proj.weight,
                        k_proj_weight=self.k_proj.weight,
                        v_proj_weight=self.v_proj.weight,
                        static_k=None,
                        static_v=None,
                        average_attn_weights=False,
                        is_causal=False,
                    )
                    attn_proj = attn_out.transpose(0, 1)
                except Exception:
                    allow_mha = False

            if attn_proj is None and allow_dpa:
                attn_out = self.attn(
                    q,
                    k,
                    v,
                    attn_mask=_coerce_attn_bias_to_b11k(
                        attn_bias, B=B, K=Lk, like=q
                    ) if has_bias else None,
                    is_causal=False,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    training=bool(self.training),
                )
                attn_out = attn_out.transpose(1, 2).contiguous().view(B, Lq, D)
                attn_proj = self.out_proj(attn_out)

            if attn_proj is None:
                plan = ATTENTION_POLICY.plan(
                    q=q,
                    need_weights=False,
                    has_bias=has_bias,
                    exporting=exporting,
                    compiling=compiling,
                    allow_flex=allow_flex,
                    allow_mha=allow_mha,
                    allow_dpa=allow_dpa,
                )
                if plan.backend == AttentionBackend.FLEX and not allow_flex:
                    break
                if plan.backend == AttentionBackend.MHA and not allow_mha:
                    break
                if plan.backend == AttentionBackend.DPA and not allow_dpa:
                    break

        if attn_proj is None:
            attn_out = self.attn(
                q,
                k,
                v,
                attn_mask=_coerce_attn_bias_to_b11k(attn_bias, B=B, K=Lk, like=q)
                if has_bias
                else None,
                is_causal=False,
                dropout_p=self.dropout_p if self.training else 0.0,
                training=bool(self.training),
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, Lq, D)
            attn_proj = self.out_proj(attn_out)

        if diag_cross and isinstance(attn_proj, torch.Tensor):
            _check(
                "attn_proj",
                attn_proj,
                extra={
                    "out_proj": {"weight": _stats(self.out_proj.weight), "bias": _stats(self.out_proj.bias)},
                },
            )

        if (
            (not exporting)
            and (not compiling)
            and env_bool("ENN_RESAMPLER_FALLBACK_ON_NONFINITE", default=True)
            and isinstance(attn_proj, torch.Tensor)
            and attn_proj.is_floating_point()
            and attn_proj.numel() > 0
        ):
            with torch.no_grad():
                samp = attn_proj.reshape(-1)[:1024]
                bad = not bool(torch.isfinite(samp).all().item())
            if bad:
                if not bool(getattr(self, "_nonfinite_warned", False)):
                    warnings.warn(
                        "[ENN] CrossAttention: attention produced NaN/Inf; rerunning with AMP disabled attention math.",
                        UserWarning,
                        stacklevel=2,
                    )
                    setattr(self, "_nonfinite_warned", True)
                self._disable_flex_bias_runtime = True

                from ..core.precision import StatelessAutocast
                from ..core.policies import PrecisionPolicy

                meta = StatelessAutocast.metadata()
                master = PrecisionPolicy.from_metadata(
                    device=q.device, metadata=meta
                ).master_float
                math_dtype = (
                    torch.promote_types(q.dtype, master)
                    if q.is_floating_point()
                    else master
                )

                with StatelessAutocast.suspend(q.device):
                    qh = q.to(dtype=math_dtype)
                    kh = k.to(dtype=math_dtype)
                    vh = v.to(dtype=math_dtype)
                    fm = None
                    if has_bias:
                        bias_bk = _coerce_attn_bias_to_bk(
                            attn_bias, B=int(B), K=int(Lk), like=qh
                        )
                        if int(bias_bk.shape[0]) == 1:
                            bias_bk = bias_bk.expand(int(B), int(Lk))
                        fm = (
                            bias_bk[:, None, None, :]
                            .expand(int(B), 1, 1, int(Lk))
                            .contiguous()
                            .to(dtype=qh.dtype)
                        )
                    attn_math = _attention_math_bshd(
                        qh,
                        kh,
                        vh,
                        attn_mask=fm,
                        is_causal=False,
                        dropout_p=self.dropout_p if self.training else 0.0,
                        training=bool(self.training),
                    )
                    attn_math = attn_math.transpose(1, 2).contiguous().view(B, Lq, D)
                    wdt = getattr(self.out_proj.weight, "dtype", attn_math.dtype)
                    if attn_math.dtype != wdt:
                        attn_math = attn_math.to(dtype=wdt)
                    attn_proj = self.out_proj(attn_math)
                    sanitize = env_bool(
                        "ENN_RESAMPLER_FALLBACK_SANITIZE",
                        default=env_bool(
                            "ENN_RESAMPLER_FALLBACK_SANITIZE_FP32", default=True
                        ),
                    )
                    if sanitize:
                        with contextlib.suppress(Exception):
                            attn_proj = torch.nan_to_num(
                                attn_proj, nan=0.0, posinf=0.0, neginf=0.0
                            )

        return attn_proj


class LatentAttention(nn.Module):
    def __init__(
        self: Self,
        d_model: int,
        nhead: int,
        *args: Any,
        norm_type: str = "layernorm",
        eps: float = 1e-6,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _ = (args, kwargs)
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = self.d_model // self.nhead
        self.dropout_p = float(dropout)
        self.norm1 = norm_layer(norm_type=norm_type, dim=self.d_model, eps=eps)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=True)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.attn = DotProductAttention(
            num_heads=self.nhead, head_dim=self.head_dim
        )
        self._disable_flex_runtime: bool = False
        self._flex_fail_count: int = 0
        self._flex_fail_max: int = max(
            0, int(env_int("ENN_LATENT_FLEX_RETRY", 3))
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"LatentAttention expects x (B,K,D), got shape {tuple(x.shape)}"
            )
        B, K, D = x.shape
        if D != self.d_model:
            raise ValueError(
                f"LatentAttention expects last dim D={self.d_model}, got {D}"
            )

        y = self.norm1(x)

        qkv = self.qkv(y)
        q, k, v = (
            qkv.view(B, K, 3, self.nhead, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )

        compiling = bool(is_compiling())
        exporting = bool(is_export_or_trace()) and (not compiling)
        km = None
        k_mha = ""
        if not (exporting or compiling or bool(is_symbolic())):
            km = get_kernel_manager()
            site = getattr(self, "_enn_kernel_site", None)
            if not isinstance(site, str) or not site:
                site = f"{self.__class__.__name__}@{id(self):x}"
                setattr(self, "_enn_kernel_site", site)
            with contextlib.suppress(Exception):
                child_site = getattr(self.attn, "_enn_kernel_site", None)
                if (not isinstance(child_site, str)) or (not child_site):
                    setattr(self.attn, "_enn_kernel_site", f"{site}:latent.dpa")
            dev_i = int(q.device.index) if q.device.index is not None else 0
            kbase = f"attn:{site}:latent@{q.device.type}:{dev_i}"
            k_mha = f"{kbase}:mha"
        allow_flex = not self._disable_flex_runtime
        allow_mha = (not (exporting or compiling)) and (km is not None) and (not km.is_dead(k_mha))
        allow_dpa = True

        plan = ATTENTION_POLICY.plan(
            q=q,
            need_weights=False,
            has_bias=False,
            exporting=exporting,
            compiling=compiling,
            allow_flex=allow_flex,
            allow_mha=allow_mha,
            allow_dpa=allow_dpa,
        )

        def _classify(exc: BaseException) -> str:
            msg = str(exc)
            tname = type(exc).__name__
            m = msg.lower()
            if "flexattention not available" in m:
                return "struct"
            if "not available" in m and "flex" in m:
                return "struct"
            if tname in ("TypeError", "ValueError"):
                return "struct"
            if exporting or compiling:
                return "struct"
            if "float8" in m or "fp8" in m:
                return "struct"
            if (
                "no valid triton configs" in m
                or "outofresources" in m
                or "out of resources" in m
            ):
                return "transient"
            if (
                "torch._dynamo" in m
                or "torch._inductor" in m
                or "compileerror" in m
            ):
                return "transient"
            return "transient"

        attn_proj: torch.Tensor | None = None
        dropout_p = self.dropout_p if self.training else 0.0

        if plan.backend == AttentionBackend.FLEX and allow_flex:
            try:
                if getattr(q, "dtype", None) == torch.float64 or (
                    "float8" in str(getattr(q, "dtype", ""))
                ):
                    raise RuntimeError("flex disabled for fp64/fp8")
                _ensure_flex_kwargs_initialized()
                attn_out = get_flex_kernel()(
                    q,
                    k,
                    v,
                    score_mod=None,
                    block_mask=None,
                    scale=None,
                    dropout_p=dropout_p,
                    training=bool(self.training),
                    is_causal=False,
                )
                attn_out = attn_out.transpose(1, 2).contiguous().view(B, K, D)
                attn_proj = self.out_proj(attn_out)
            except Exception as exc:
                kind = _classify(exc)
                self._flex_fail_count += 1
                if kind == "struct":
                    self._disable_flex_runtime = True
                elif (
                    self._flex_fail_max > 0
                    and self._flex_fail_count >= self._flex_fail_max
                ):
                    self._disable_flex_runtime = True
                plan = ATTENTION_POLICY.plan(
                    q=q,
                    need_weights=False,
                    has_bias=False,
                    exporting=exporting,
                    compiling=compiling,
                    allow_flex=False,
                    allow_mha=allow_mha,
                    allow_dpa=allow_dpa,
                )
                attn_proj = None

        if attn_proj is None and plan.backend == AttentionBackend.MHA and allow_mha:
            try:
                y_seq = y.transpose(0, 1)
                attn_out, _ = F.multi_head_attention_forward(
                    y_seq,
                    y_seq,
                    y_seq,
                    embed_dim_to_check=int(D),
                    num_heads=int(self.nhead),
                    in_proj_weight=self.qkv.weight,
                    in_proj_bias=self.qkv.bias,
                    bias_k=None,
                    bias_v=None,
                    add_zero_attn=False,
                    dropout_p=dropout_p,
                    out_proj_weight=self.out_proj.weight,
                    out_proj_bias=self.out_proj.bias,
                    training=bool(self.training),
                    key_padding_mask=None,
                    need_weights=False,
                    attn_mask=None,
                    use_separate_proj_weight=False,
                    q_proj_weight=None,
                    k_proj_weight=None,
                    v_proj_weight=None,
                    static_k=None,
                    static_v=None,
                    average_attn_weights=False,
                    is_causal=False,
                )
                attn_proj = attn_out.transpose(0, 1)
            except Exception:
                plan = ATTENTION_POLICY.plan(
                    q=q,
                    need_weights=False,
                    has_bias=False,
                    exporting=exporting,
                    compiling=compiling,
                    allow_flex=False,
                    allow_mha=False,
                    allow_dpa=allow_dpa,
                )
                attn_proj = None

        if attn_proj is None:
            attn_out = self.attn(
                q,
                k,
                v,
                attn_mask=None,
                is_causal=False,
                dropout_p=dropout_p,
                training=bool(self.training),
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, K, D)
            attn_proj = self.out_proj(attn_out)

        if (
            (not exporting)
            and (not compiling)
            and env_bool("ENN_LATENT_FALLBACK_ON_NONFINITE", default=True)
            and isinstance(attn_proj, torch.Tensor)
            and attn_proj.is_floating_point()
            and attn_proj.numel() > 0
        ):
            with torch.no_grad():
                samp = attn_proj.reshape(-1)[:1024]
                bad = not bool(torch.isfinite(samp).all().item())
            if bad:
                self._disable_flex_runtime = True
                if not bool(getattr(self, "_enn_nonfinite_warned", False)):
                    warnings.warn(
                        "[ENN] LatentAttention: attention produced NaN/Inf; rerunning attention math with AMP disabled (policy fallback).",
                        UserWarning,
                        stacklevel=2,
                    )
                    setattr(self, "_enn_nonfinite_warned", True)

                from ..core.precision import StatelessAutocast
                from ..core.policies import PrecisionPolicy

                meta = StatelessAutocast.metadata()
                master = PrecisionPolicy.from_metadata(
                    device=q.device, metadata=meta
                ).master_float
                math_dtype = (
                    torch.promote_types(q.dtype, master)
                    if q.is_floating_point()
                    else master
                )
                with StatelessAutocast.suspend(q.device):
                    qh = q.to(dtype=math_dtype)
                    kh = k.to(dtype=math_dtype)
                    vh = v.to(dtype=math_dtype)
                    attn_math = _attention_math_bshd(
                        qh,
                        kh,
                        vh,
                        attn_mask=None,
                        is_causal=False,
                        dropout_p=dropout_p,
                        training=bool(self.training),
                    )
                    attn_math = attn_math.transpose(1, 2).contiguous().view(B, K, D)
                    wdt = getattr(self.out_proj.weight, "dtype", attn_math.dtype)
                    if attn_math.dtype != wdt:
                        attn_math = attn_math.to(dtype=wdt)
                    attn_proj = self.out_proj(attn_math)
                    if env_bool("ENN_LATENT_FALLBACK_SANITIZE", default=True):
                        with contextlib.suppress(Exception):
                            attn_proj = torch.nan_to_num(
                                attn_proj,
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            )

        return attn_proj


class SigmoidGate(nn.Module):
    def __init__(
        self: Self,
        d_model: int,
        hidden_dim: int = 64,
        *args: Any,
        eps: float = 1e-6,
        clip_eps: float = 1e-6,
        p_floor: float = 0.0,
        p_ceil: float = 1.0,
        tile_size: Optional[int] = None,
        tile_shape: Optional[Sequence[int]] = None,
        event_shape: Optional[Sequence[int]] = None,
        stat_width_frac: float = 0.05,
        stat_edge_frac: float = 0.02,
        detach_inputs: bool = True,
        use_tokens: bool = True,
        use_refined: bool = True,
        use_stats: bool = True,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.clip_eps = float(clip_eps)
        self.p_floor = float(p_floor)
        self.p_ceil = float(p_ceil)
        self.stat_width_frac = float(stat_width_frac)
        self.stat_edge_frac = float(stat_edge_frac)
        self.detach_inputs = bool(detach_inputs)
        self.use_tokens = bool(use_tokens)
        self.use_refined = bool(use_refined)
        self.use_stats = bool(use_stats)
        ts = 0 if tile_size is None else int(tile_size)
        self.tile_size = int(ts) if ts > 0 else 0
        if event_shape is None:
            self.event_shape: Tuple[int, ...] = ()
        else:
            self.event_shape = tuple(int(v) for v in event_shape)
        if tile_shape is None:
            self.tile_shape: Tuple[int, ...] = ()
        elif isinstance(tile_shape, int) and not isinstance(tile_shape, bool):
            self.tile_shape = (int(tile_shape),)
        else:
            self.tile_shape = tuple(int(v) for v in tile_shape)
        self.event_shape = tuple(max(1, int(v)) for v in self.event_shape)
        self.tile_shape = tuple(max(1, int(v)) for v in self.tile_shape)
        in_dim = 0
        if self.use_tokens:
            in_dim += int(d_model)
        if self.use_refined:
            in_dim += int(d_model)
        if self.use_stats:
            in_dim += 2
        in_dim = max(1, int(in_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 1),
        )
        self.tile_net: Optional[nn.Module] = None
        if self.tile_size > 0 or len(self.tile_shape) > 0:
            tile_in = 3
            self.tile_net = nn.Sequential(
                nn.LayerNorm(tile_in),
                nn.Linear(tile_in, int(hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(hidden_dim), 1),
            )
        with contextlib.suppress(Exception):
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
        with contextlib.suppress(Exception):
            if self.tile_net is not None:
                last = self.tile_net[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)
        self.register_buffer(
            "_fb_count", torch.zeros((), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "_fb_active_low_sum",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_fb_active_high_sum",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_fb_width_sum",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_fb_edge_low_sum",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_fb_edge_high_sum",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self._fb_lock = Mutex()

    def __getstate__(self: Self) -> dict[str, object]:
        state = super().__getstate__()
        state.pop("_fb_lock", None)
        return state

    def __setstate__(self: Self, state: dict[str, object]) -> None:
        super().__setstate__(state)
        self._fb_lock = Mutex()

    @torch.no_grad()
    def consume_fallback_tensor_stats(self: Self) -> torch.Tensor:
        lock = getattr(self, "_fb_lock", None)
        if lock is None:
            lock = Mutex()
            setattr(self, "_fb_lock", lock)
        with lock:
            stats = torch.stack(
                [
                    self._fb_count.detach(),
                    self._fb_active_low_sum.detach(),
                    self._fb_active_high_sum.detach(),
                    self._fb_width_sum.detach(),
                    self._fb_edge_low_sum.detach(),
                    self._fb_edge_high_sum.detach(),
                ]
            )
            self._fb_count.zero_()
            self._fb_active_low_sum.zero_()
            self._fb_active_high_sum.zero_()
            self._fb_width_sum.zero_()
            self._fb_edge_low_sum.zero_()
            self._fb_edge_high_sum.zero_()
            return stats

    @torch.no_grad()
    def consume_fallback_stats(self: Self) -> torch.Tensor:
        return self.consume_fallback_tensor_stats()

    @torch_compiler_disable(
        reason="SigmoidGate fallback stats update", recursive=False
    )
    def _fb_add_stats(
        self: Self,
        count: float,
        width_sum: torch.Tensor,
        active_low_sum: torch.Tensor,
        active_high_sum: torch.Tensor,
        edge_low_sum: torch.Tensor,
        edge_high_sum: torch.Tensor,
    ) -> None:
        if is_symbolic():
            return
        with torch.no_grad():
            lock = getattr(self, "_fb_lock", None)
            if lock is None:
                lock = Mutex()
                setattr(self, "_fb_lock", lock)
            with lock:
                self._fb_count.add_(float(count))
                self._fb_width_sum.add_(width_sum.to(dtype=torch.float32))
                self._fb_active_low_sum.add_(
                    active_low_sum.to(dtype=torch.float32)
                )
                self._fb_active_high_sum.add_(
                    active_high_sum.to(dtype=torch.float32)
                )
                self._fb_edge_low_sum.add_(
                    edge_low_sum.to(dtype=torch.float32)
                )
                self._fb_edge_high_sum.add_(
                    edge_high_sum.to(dtype=torch.float32)
                )

    def _expand_tiles(
        self: Self, p_tile: torch.Tensor, dim: int
    ) -> torch.Tensor:
        if self.tile_size <= 0:
            raise RuntimeError(
                "SigmoidGate._expand_tiles called with tile_size<=0"
            )
        b = p_tile.size(0)
        tile = int(self.tile_size)
        n_tiles = p_tile.size(1)
        d_pad = int(n_tiles * tile)
        p_full = (
            p_tile.unsqueeze(-1).expand(b, n_tiles, tile).reshape(b, d_pad)
        )
        return p_full[:, : int(dim)]

    @staticmethod
    def _prod_int(shape: Sequence[int]) -> int:
        out = 1
        for v in shape:
            out *= int(v)
        return int(out)

    def _normalize_tile_shape(
        self: Self, event_shape: Sequence[int]
    ) -> Tuple[int, ...]:
        ts = tuple(int(v) for v in (getattr(self, "tile_shape", ()) or ()))
        if not ts or not event_shape:
            return ()
        ndim = int(len(event_shape))
        if ndim <= 0:
            return ()
        if len(ts) == 1:
            ts = ts * ndim
        elif len(ts) < ndim:
            ts = (1,) * (ndim - len(ts)) + ts
        elif len(ts) > ndim:
            ts = ts[-ndim:]
        return tuple(max(1, int(v)) for v in ts)

    def _calc_edge_reg(
        self: Self,
        p: torch.Tensor,
        p_low: torch.Tensor,
        p_high: torch.Tensor,
        p_ceil: float,
        p_floor: float,
        eps: float,
        width: torch.Tensor,
        out_full: torch.Tensor,
        return_edge_reg: bool,
        return_edge_reg_lr: bool,
        edge_reg_min_width_frac: float,
        edge_reg_frac: float,
        edge_reg_power: float,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        try:
            full = max(float(p_ceil - p_floor), float(eps))
            min_w = float(max(float(edge_reg_min_width_frac) * full, eps))
            mask = (width >= min_w).to(dtype=torch.float32)
            w_safe = torch.maximum(width, width.new_full((), float(eps)))
            q = torch.clamp(
                (p.to(dtype=torch.float32) - p_low.to(dtype=torch.float32))
                / w_safe,
                0.0,
                1.0,
            )
            m = float(min(max(float(edge_reg_frac), eps), 0.49))
            inv_m = 1.0 / m
            pen_low = (F.relu(m - q) * inv_m).pow(float(edge_reg_power)) * mask
            pen_high = (F.relu(q - (1.0 - m)) * inv_m).pow(
                float(edge_reg_power)
            ) * mask
            denom = mask.sum() + float(eps)
            er_l = pen_low.sum() / denom
            er_h = pen_high.sum() / denom
        except Exception:
            er_l, er_h = None, None
        out_dtype = out_full.dtype
        if return_edge_reg_lr:
            return (
                out_full,
                (
                    er_l.to(dtype=out_dtype)
                    if er_l is not None
                    else out_full.new_zeros((), dtype=out_dtype)
                ),
                (
                    er_h.to(dtype=out_dtype)
                    if er_h is not None
                    else out_full.new_zeros((), dtype=out_dtype)
                ),
            )
        if return_edge_reg:
            er = (
                (er_l + er_h)
                if (er_l is not None and er_h is not None)
                else out_full.new_zeros((), dtype=out_dtype)
            )
            return out_full, er.to(dtype=out_dtype)
        return out_full

    def _compute_tile_score(
        self: Self,
        global_logit: torch.Tensor,
        base: torch.Tensor,
        residue: torch.Tensor,
        tokens_dtype: torch.dtype,
        eps: float,
        p_floor: float,
        p_ceil: float,
        z_min: torch.Tensor | None,
        z_max: torch.Tensor | None,
        fallback_bounds: bool,
        stat_width_frac: float,
        stat_edge_frac: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = base.detach() if self.detach_inputs else base
        r = residue.detach() if self.detach_inputs else residue
        b32, r32 = b.to(dtype=torch.float32), r.to(dtype=torch.float32)
        B = b32.shape[0]
        D = b32.shape[1]
        event_shape_t = tuple(int(v) for v in self.event_shape)
        tile_shape_t = tuple(
            int(v) for v in self._normalize_tile_shape(self.event_shape)
        )
        grid_shape = tuple(
            (int(d) + int(t) - 1) // int(t)
            for d, t in zip(event_shape_t, tile_shape_t)
        )
        pad_shape = tuple(
            int(g) * int(t) for g, t in zip(grid_shape, tile_shape_t)
        )
        pads = []
        for orig, padded in reversed(list(zip(event_shape_t, pad_shape))):
            pads.extend([0, int(padded - orig)])
        pad_needed = any(
            int(p) > int(o) for p, o in zip(pad_shape, event_shape_t)
        )
        b_nd, r_nd = (
            b32.reshape(B, *event_shape_t),
            r32.reshape(B, *event_shape_t),
        )
        if pad_needed:
            b_nd, r_nd = F.pad(b_nd, tuple(pads)), F.pad(r_nd, tuple(pads))
        mask_bool = None
        if pad_needed:
            for i, (orig, padded) in enumerate(zip(event_shape_t, pad_shape)):
                if int(padded) == int(orig):
                    continue
                v = (
                    torch.arange(int(padded), device=b_nd.device) < int(orig)
                ).view(
                    *([1] * i),
                    int(padded),
                    *([1] * (len(event_shape_t) - i - 1)),
                )
                mask_bool = v if mask_bool is None else (mask_bool & v)
            if mask_bool is None:
                mask_bool = torch.ones(
                    pad_shape, device=b_nd.device, dtype=torch.bool
                )
        view_shape, interleaved = [B], []
        for g, t in zip(grid_shape, tile_shape_t):
            view_shape.extend([int(g), int(t)])
            interleaved.extend([int(g), int(t)])
        b_tile, r_tile = b_nd.reshape(*view_shape), r_nd.reshape(*view_shape)
        tile_dims = tuple(range(2, 1 + 2 * len(event_shape_t), 2))
        mask = (
            mask_bool.reshape(*interleaved)
            .unsqueeze(0)
            .to(dtype=torch.float32)
            if mask_bool is not None
            else None
        )
        if mask is None:
            b_rms_t = torch.sqrt((b_tile * b_tile).mean(dim=tile_dims) + eps)
            r_rms_t = torch.sqrt((r_tile * r_tile).mean(dim=tile_dims) + eps)
        else:
            denom = torch.clamp(mask.sum(dim=tile_dims), min=1.0)
            b_rms_t = torch.sqrt(
                ((b_tile * b_tile) * mask).sum(dim=tile_dims) / denom + eps
            )
            r_rms_t = torch.sqrt(
                ((r_tile * r_tile) * mask).sum(dim=tile_dims) / denom + eps
            )
        tile_feats = torch.stack(
            [b_rms_t, r_rms_t, r_rms_t / (b_rms_t + eps)], dim=-1
        ).to(dtype=tokens_dtype)
        sig = torch.sigmoid(
            global_logit.view(B, *([1] * len(event_shape_t)))
            + self.tile_net(tile_feats).squeeze(-1)
        )
        p_low, p_high = (
            sig.new_full(sig.shape, float(p_floor)),
            sig.new_full(sig.shape, float(p_ceil)),
        )
        if z_min is not None and z_max is not None:
            try:
                zmin_t = (
                    z_min.reshape(1, *interleaved)
                    if z_min.numel() > 1
                    else z_min
                )
                zmax_t = (
                    z_max.reshape(1, *interleaved)
                    if z_max.numel() > 1
                    else z_max
                )
                if pad_needed and z_min.numel() > 1:
                    zmin_t = F.pad(
                        z_min.reshape(1, *event_shape_t), tuple(pads)
                    ).reshape(1, *interleaved)
                    zmax_t = F.pad(
                        z_max.reshape(1, *event_shape_t), tuple(pads)
                    ).reshape(1, *interleaved)
                r_safe = torch.where(
                    r_tile.abs() < eps,
                    torch.where(r_tile >= 0, 1.0, -1.0) * eps,
                    r_tile,
                )
                p_a, p_b = (
                    (zmin_t - b_tile) / r_safe,
                    (zmax_t - b_tile) / r_safe,
                )
                p_min, p_max = torch.minimum(p_a, p_b), torch.maximum(p_a, p_b)
                if mask_bool is not None:
                    mask_t = mask_bool.reshape(*interleaved).unsqueeze(0)
                    p_min = torch.where(
                        mask_t,
                        p_min,
                        p_min.new_full((), torch.finfo(p_min.dtype).min),
                    )
                    p_max = torch.where(
                        mask_t,
                        p_max,
                        p_max.new_full((), torch.finfo(p_max.dtype).max),
                    )
                p_low = torch.clamp(
                    p_min.amax(dim=tile_dims),
                    min=float(p_floor),
                    max=float(p_ceil),
                )
                p_high = torch.maximum(
                    torch.clamp(
                        p_max.amin(dim=tile_dims),
                        min=float(p_floor),
                        max=float(p_ceil),
                    ),
                    p_low + float(eps),
                )
            except Exception:
                pass
        p_tile = p_low + (p_high - p_low) * sig.to(dtype=p_low.dtype)
        clip = float(max(self.clip_eps, eps))
        p_tile = torch.clamp(
            p_tile,
            min=(
                float(p_floor) + clip
                if float(p_ceil) - clip > float(p_floor) + clip
                else float(p_floor)
            ),
            max=float(p_ceil) - clip,
        )
        if (
            bool(fallback_bounds)
            and self.training
            and (
                (not is_checkpoint() and torch.is_grad_enabled())
                or (
                    _GATE_STATS_CKPT_FWD
                    and is_checkpoint()
                    and (not torch.is_grad_enabled())
                )
            )
        ):
            self._update_stats(
                p_tile,
                p_low,
                p_high,
                p_ceil,
                p_floor,
                eps,
                stat_width_frac,
                stat_edge_frac,
            )
        expand_view, exp_shape = [B], [B]
        for g, t in zip(grid_shape, tile_shape_t):
            expand_view.extend([int(g), 1])
            exp_shape.extend([int(g), int(t)])
        p_crop = (
            p_tile.reshape(*expand_view)
            .expand(*exp_shape)
            .contiguous()
            .view(B, *pad_shape)[
                (slice(None),) + tuple(slice(0, int(d)) for d in event_shape_t)
            ]
        )
        return p_crop.reshape(B, D), p_low, p_high

    def _update_stats(
        self: Self,
        p: torch.Tensor,
        p_low: torch.Tensor,
        p_high: torch.Tensor,
        p_ceil: float,
        p_floor: float,
        eps: float,
        stat_width_frac: float,
        stat_edge_frac: float,
    ) -> None:
        try:
            width = (p_high - p_low).to(dtype=torch.float32)
            denom = max(float(p_ceil - p_floor), float(eps))
            tthr = float(max(float(stat_width_frac) * denom, eps))
            active_low = (p_low - float(p_floor)).to(
                dtype=torch.float32
            ) >= tthr
            active_high = (float(p_ceil) - p_high).to(
                dtype=torch.float32
            ) >= tthr
            edge_thr = torch.maximum(
                width, width.new_full((), float(eps))
            ) * float(max(float(stat_edge_frac), 0.0))
            self._fb_add_stats(
                float(p.numel()),
                width.sum(),
                active_low.float().sum(),
                active_high.float().sum(),
                ((p - p_low) <= edge_thr).float().sum(),
                ((p_high - p) <= edge_thr).float().sum(),
            )
        except Exception:
            pass

    def forward(
        self: Self,
        *args: Any,
        tokens: torch.Tensor,
        refined_tokens: Optional[torch.Tensor] = None,
        base: Optional[torch.Tensor] = None,
        residue: Optional[torch.Tensor] = None,
        z_min: Optional[torch.Tensor] = None,
        z_max: Optional[torch.Tensor] = None,
        fallback_bounds: bool = False,
        return_edge_reg: bool = False,
        return_edge_reg_lr: bool = False,
        edge_reg_frac: float = 0.02,
        edge_reg_min_width_frac: float = 0.05,
        edge_reg_power: float = 2.0,
    ) -> (
        torch.Tensor
        | Tuple[torch.Tensor, torch.Tensor]
        | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        feats: list[torch.Tensor] = []
        if self.use_tokens:
            t = tokens.mean(dim=1)
            if self.detach_inputs:
                t = t.detach()
            feats.append(t)
        if self.use_refined and refined_tokens is not None:
            r = refined_tokens.mean(dim=1)
            if self.detach_inputs:
                r = r.detach()
            feats.append(r)
        if self.use_stats and base is not None and residue is not None:
            b = base.detach() if self.detach_inputs else base
            res = residue.detach() if self.detach_inputs else residue
            b32 = b.to(dtype=torch.float32)
            r32 = res.to(dtype=torch.float32)
            b_rms = torch.sqrt(
                torch.mean(b32 * b32, dim=1, keepdim=True) + self.eps
            )
            r_rms = torch.sqrt(
                torch.mean(r32 * r32, dim=1, keepdim=True) + self.eps
            )
            feats.append(b_rms.to(dtype=tokens.dtype))
            feats.append(r_rms.to(dtype=tokens.dtype))
        x = feats[0] if len(feats) == 1 else torch.cat(feats, dim=1)
        global_logit = self.net(x).squeeze(-1)
        use_tile_nd = False
        symbolic = False
        try:
            symbolic = bool(is_symbolic())
        except Exception:
            symbolic = False
        if (
            self.tile_net is not None
            and base is not None
            and residue is not None
            and base.dim() == 2
            and residue.dim() == 2
            and len(getattr(self, "tile_shape", ()) or ()) > 0
            and len(getattr(self, "event_shape", ()) or ()) > 0
        ):
            try:
                tile_ok = (
                    bool(self._normalize_tile_shape(self.event_shape))
                    and int(self._prod_int(self.event_shape)) > 0
                )
                if not tile_ok:
                    use_tile_nd = False
                elif symbolic:
                    use_tile_nd = True
                else:
                    use_tile_nd = int(base.shape[0]) == int(
                        residue.shape[0]
                    ) and int(self._prod_int(self.event_shape)) == int(
                        base.shape[1]
                    )
            except Exception:
                use_tile_nd = False
        if use_tile_nd:
            out_full, p_low, p_high = self._compute_tile_score(
                global_logit,
                base,
                residue,
                tokens.dtype,
                self.eps,
                self.p_floor,
                self.p_ceil,
                z_min,
                z_max,
                fallback_bounds,
                self.stat_width_frac,
                self.stat_edge_frac,
            )
        else:
            sig = torch.sigmoid(global_logit)
            p_low = sig.new_full(sig.shape, float(self.p_floor))
            p_high = sig.new_full(sig.shape, float(self.p_ceil))
            if (
                z_min is not None
                and z_max is not None
                and base is not None
                and residue is not None
            ):
                b = base.detach() if self.detach_inputs else base
                r = residue.detach() if self.detach_inputs else residue
                b32 = b.to(dtype=torch.float32)
                r32 = r.to(dtype=torch.float32)
                zmin = z_min.to(device=b32.device, dtype=torch.float32)
                zmax = z_max.to(device=b32.device, dtype=torch.float32)
                if zmin.numel() != 1:
                    zmin = zmin.reshape(1, -1)
                if zmax.numel() != 1:
                    zmax = zmax.reshape(1, -1)
                sign = torch.where(r32 >= 0, 1.0, -1.0)
                r_safe = torch.where(
                    r32.abs() < self.eps, sign * self.eps, r32
                )
                p_a = (zmin - b32) / r_safe
                p_b = (zmax - b32) / r_safe
                p_low_bound = torch.minimum(p_a, p_b).max(dim=1).values
                p_high_bound = torch.maximum(p_a, p_b).min(dim=1).values
                p_low = torch.clamp(
                    p_low_bound,
                    min=float(self.p_floor),
                    max=float(self.p_ceil),
                )
                p_high = torch.clamp(
                    p_high_bound,
                    min=float(self.p_floor),
                    max=float(self.p_ceil),
                )
                p_high = torch.maximum(p_high, p_low + float(self.eps))
            p = p_low + (p_high - p_low) * sig.to(dtype=p_low.dtype)
            clip = float(max(self.clip_eps, self.eps))
            p = torch.clamp(
                p,
                min=(
                    float(self.p_floor) + clip
                    if float(self.p_ceil) - clip > float(self.p_floor) + clip
                    else float(self.p_floor)
                ),
                max=float(self.p_ceil) - clip,
            )
            if (
                bool(fallback_bounds)
                and self.training
                and (
                    (not is_checkpoint() and torch.is_grad_enabled())
                    or (
                        _GATE_STATS_CKPT_FWD
                        and is_checkpoint()
                        and (not torch.is_grad_enabled())
                    )
                )
            ):
                self._update_stats(
                    p,
                    p_low,
                    p_high,
                    self.p_ceil,
                    self.p_floor,
                    self.eps,
                    self.stat_width_frac,
                    self.stat_edge_frac,
                )
            out_full = p.unsqueeze(-1)
        return self._calc_edge_reg(
            out_full if use_tile_nd else p,
            p_low,
            p_high,
            self.p_ceil,
            self.p_floor,
            self.eps,
            (p_high - p_low).to(dtype=torch.float32),
            out_full,
            return_edge_reg,
            return_edge_reg_lr,
            edge_reg_min_width_frac,
            edge_reg_frac,
            edge_reg_power,
        )


class Embedding(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        continuous_idx: Sequence[int] = (),
        categorical: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.continuous_idx = tuple(int(i) for i in continuous_idx)
        if self.continuous_idx:
            self.register_buffer(
                "_cont_idx",
                torch.tensor(self.continuous_idx, dtype=torch.long),
                persistent=False,
            )
        else:
            self.register_buffer(
                "_cont_idx",
                torch.empty((0,), dtype=torch.long),
                persistent=False,
            )

        cats: list[dict[str, Any]] = []
        for c in categorical:
            cats.append(dict(c))
        self._cats = cats

        used: set[int] = set()
        for i in self.continuous_idx:
            if i < 0 or i >= self.in_dim:
                raise ValueError(
                    f"continuous_idx contains out-of-range idx={i} for in_dim={self.in_dim}"
                )
            used.add(i)

        self._embeds = nn.ModuleList()
        self._cat_out_dims: list[int] = []
        for c in self._cats:
            if "idx" not in c and "span" not in c:
                raise ValueError("categorical spec must include either 'idx' or 'span'")

            mode = str(c.get("mode") or ("onehot" if "span" in c else "index")).strip().lower()
            c["mode"] = mode
            name = str(c.get("name") or f"cat{len(self._embeds)}")
            c["name"] = name

            num = int(c.get("num_embeddings") or c.get("num") or 0)
            dim = int(c.get("embedding_dim") or c.get("dim") or 0)
            if num <= 0 or dim <= 0:
                raise ValueError(
                    f"categorical spec {name} must set num_embeddings>0 and embedding_dim>0"
                )

            if mode == "index":
                idx = int(c["idx"])
                if idx < 0 or idx >= self.in_dim:
                    raise ValueError(
                        f"categorical idx out of range: {name}.idx={idx} for in_dim={self.in_dim}"
                    )
                if idx in used:
                    raise ValueError(
                        f"feature index collision at idx={idx} for categorical '{name}'"
                    )
                used.add(idx)
            else:
                span = tuple(int(v) for v in c.get("span") or ())
                if len(span) != 2:
                    raise ValueError(
                        f"categorical spec {name} mode={mode} requires span=(start,end)"
                    )
                start, end = span
                if start < 0 or end <= start or end > self.in_dim:
                    raise ValueError(
                        f"invalid span for {name}: {span} for in_dim={self.in_dim}"
                    )
                for idx in range(start, end):
                    if idx in used:
                        raise ValueError(
                            f"feature index collision at idx={idx} within span for categorical '{name}'"
                        )
                    used.add(idx)
                if (end - start) != num:
                    raise ValueError(
                        f"categorical {name} span width ({end-start}) must equal num_embeddings ({num})"
                    )

            emb = nn.Embedding(num_embeddings=num, embedding_dim=dim)
            self._embeds.append(emb)
            self._cat_out_dims.append(dim)

        self.out_dim = int(len(self.continuous_idx) + sum(self._cat_out_dims))

    @property
    def uses_x_norm(self) -> bool:
        if self.continuous_idx:
            return True
        for c in self._cats:
            if str(c.get("mode") or "").lower() in {"onehot", "multi_hot", "multihot"}:
                return True
        return False

    def forward(self, x_raw: torch.Tensor, *, x_norm: torch.Tensor | None = None) -> torch.Tensor:
        if x_raw.dim() == 1:
            x_raw = x_raw.view(1, -1)
        if x_raw.dim() != 2:
            raise ValueError(
                f"Embedding expects a 2D tensor [B, in_dim], got {tuple(x_raw.shape)}"
            )
        if int(x_raw.shape[1]) != int(self.in_dim):
            raise ValueError(
                f"Embedding expects in_dim={self.in_dim}, got x_raw.shape[1]={int(x_raw.shape[1])}"
            )

        x_cont_src = x_norm if x_norm is not None else x_raw

        parts: list[torch.Tensor] = []
        if self.continuous_idx:
            cont = x_cont_src.index_select(dim=1, index=self._cont_idx)
            if cont.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                cont = cont.to(dtype=torch.float32)
            parts.append(cont)

        for emb, c in zip(self._embeds, self._cats):
            mode = str(c.get("mode") or "index").lower()
            if mode == "index":
                idx = int(c["idx"])
                offset = int(c.get("offset") or 0)
                clamp = bool(c.get("clamp") or False)
                ids = x_raw[:, idx].to(dtype=torch.long)
                if offset:
                    ids = ids + int(offset)
                if clamp:
                    ids = ids.clamp(min=0, max=int(emb.num_embeddings) - 1)
                parts.append(emb(ids))
            elif mode in {"onehot", "multi_hot", "multihot"}:
                span = tuple(int(v) for v in c.get("span") or ())
                start, end = span
                one = x_cont_src[:, start:end]
                if one.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                    one = one.to(dtype=torch.float32)
                parts.append(one.matmul(emb.weight))
            else:
                raise ValueError(
                    f"unsupported categorical mode={mode} for {c.get('name')}"
                )

        if not parts:
            return torch.empty(
                (int(x_raw.shape[0]), 0), device=x_raw.device, dtype=torch.float32
            )

        out = torch.cat(parts, dim=1)
        return out

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any], *, in_dim: int) -> "Embedding":
        cats = spec.get("categorical") or spec.get("cats") or ()
        cont = None
        if "continuous_idx" in spec:
            cont = spec.get("continuous_idx")
        elif "continuous" in spec:
            cont = spec.get("continuous")
        if cont is None:
            used: set[int] = set()
            for c in cats or ():
                try:
                    if "idx" in c:
                        used.add(int(c["idx"]))
                    elif "span" in c:
                        s = c.get("span")
                        if s is not None and len(s) == 2:
                            start, end = int(s[0]), int(s[1])
                            for i in range(start, end):
                                used.add(int(i))
                except Exception:
                    continue
            cont = [i for i in range(int(in_dim)) if int(i) not in used]
        return cls(
            in_dim=int(in_dim),
            continuous_idx=tuple(int(i) for i in (cont or ())),
            categorical=tuple(cats or ()),
        )


class Scaler(nn.Module):
    def __init__(self: Self, eps: float = 1e-6) -> None:
        super().__init__()
        self.__enn_precision_exempt__ = True
        self.eps = float(eps)
        self.calib_mode: str = "none"
        self.register_buffer("x_mean", torch.zeros(1, dtype=torch.float64))
        self.register_buffer("x_std", torch.ones(1, dtype=torch.float64))
        self.register_buffer("y_mean", torch.zeros(1, dtype=torch.float64))
        self.register_buffer("y_std", torch.ones(1, dtype=torch.float64))
        self.register_buffer(
            "y_min", torch.full((1,), float("-inf"), dtype=torch.float64)
        )
        self.register_buffer(
            "y_max", torch.full((1,), float("inf"), dtype=torch.float64)
        )
        self.register_buffer(
            "y_q_low", torch.full((1,), float("-inf"), dtype=torch.float64)
        )
        self.register_buffer(
            "y_q_high", torch.full((1,), float("inf"), dtype=torch.float64)
        )
        self.register_buffer("affine_a", torch.ones(1, dtype=torch.float64))
        self.register_buffer("affine_b", torch.zeros(1, dtype=torch.float64))
        self.register_buffer("pw_x", torch.zeros((1, 1), dtype=torch.float64))
        self.register_buffer("pw_y", torch.zeros((1, 1), dtype=torch.float64))
        self.register_buffer("y_out_scale", torch.ones(1, dtype=torch.float64))
        self.register_buffer("y_out_bias", torch.zeros(1, dtype=torch.float64))
        self.register_buffer(
            "y_out_clip_low",
            torch.full((1,), float(torch.finfo(torch.float32).min), dtype=torch.float64),
        )
        self.register_buffer(
            "y_out_clip_high",
            torch.full((1,), float(torch.finfo(torch.float32).max), dtype=torch.float64),
        )
        self.output_ab_enabled: bool = False
        self._stats_cache_lock = Mutex()
        self._stats_cache_max = 8
        self._x_stats_cache: Dict[
            Tuple[str, int, torch.dtype], Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._y_stats_cache: Dict[
            Tuple[str, int, torch.dtype], Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._output_ab_log_once: bool = False
        self._output_ab_clip_only: bool = False
        self._output_ab_clip_only_reason: str = ""
        self._output_ab_clip_only_warned: bool = False

    def __getstate__(self: Self) -> dict[str, object]:
        state = super().__getstate__()
        state.pop("_stats_cache_lock", None)
        state.pop("_x_stats_cache", None)
        state.pop("_y_stats_cache", None)
        return state

    def __setstate__(self: Self, state: dict[str, object]) -> None:
        super().__setstate__(state)
        self._stats_cache_lock = Mutex()
        if not hasattr(self, "_stats_cache_max"):
            self._stats_cache_max = 8
        self._x_stats_cache = {}
        self._y_stats_cache = {}
        if not hasattr(self, "_output_ab_log_once"):
            self._output_ab_log_once = False
        if not hasattr(self, "_output_ab_clip_only"):
            self._output_ab_clip_only = False
        if not hasattr(self, "_output_ab_clip_only_reason"):
            self._output_ab_clip_only_reason = ""
        if not hasattr(self, "_output_ab_clip_only_warned"):
            self._output_ab_clip_only_warned = False

    def _resolve_master_dtype_for_io(self: Self, t: torch.Tensor) -> torch.dtype:
        pref = str(os.environ.get("ENN_SCALER_MASTER_DTYPE", "") or "").strip().lower()
        if pref in {"fp64", "float64", "f64", "64"}:
            return torch.float64
        if pref in {"fp32", "float32", "f32", "32"}:
            return torch.float32
        if pref in {"int64", "i64", "long"}:
            return torch.int64
        for attr in ("master_dtype", "_master_dtype", "scaler_master_dtype", "_scaler_master_dtype"):
            v = getattr(self, attr, None)
            if isinstance(v, torch.dtype):
                return v
        if not (t.is_floating_point() or t.is_complex()):
            return torch.int64
        if getattr(t.device, "type", None) == "cuda" and t.dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return torch.float64

    def _should_restore_input_dtype(self: Self, input_dtype: torch.dtype) -> bool:
        if env_bool("ENN_SCALER_RESTORE_INPUT_DTYPE", False):
            return True
        if input_dtype in (torch.float32, torch.float64):
            return True
        if input_dtype in (torch.float16, torch.bfloat16):
            return False
        return True

    def _scaler_guard_enabled(self: Self) -> bool:
        return bool(env_bool("ENN_SCALER_GUARD", False))

    def _scaler_guard_params(self: Self, t: torch.Tensor) -> tuple[float, float, bool]:
        std_min = float(os.environ.get("ENN_SCALER_COLLAPSE_STD_MIN", "") or 0.0)
        eps_min = float(os.environ.get("ENN_SCALER_EPS_MIN", "") or float(self.eps))
        force_fp32_on_cuda = bool(env_bool("ENN_SCALER_GUARD_FORCE_FP32_ON_CUDA", True))
        if std_min <= 0.0:
            if t.dtype == torch.bfloat16:
                std_min = float(os.environ.get("ENN_SCALER_COLLAPSE_STD_MIN_BF16", "") or 1e-3)
            elif t.dtype == torch.float16:
                std_min = float(os.environ.get("ENN_SCALER_COLLAPSE_STD_MIN_FP16", "") or 5e-4)
            else:
                std_min = float(os.environ.get("ENN_SCALER_COLLAPSE_STD_MIN_FP32", "") or 1e-6)
        if eps_min <= 0.0:
            eps_min = float(self.eps)
        if t.dtype == torch.bfloat16:
            eps_min = max(eps_min, float(os.environ.get("ENN_SCALER_EPS_MIN_BF16", "") or 1e-4))
        elif t.dtype == torch.float16:
            eps_min = max(eps_min, float(os.environ.get("ENN_SCALER_EPS_MIN_FP16", "") or 1e-5))
        return float(std_min), float(eps_min), bool(force_fp32_on_cuda)

    def _guard_is_collapse(self: Self, out2: torch.Tensor, *, std_min: float) -> bool:
        if out2.numel() == 0:
            return False
        if out2.is_floating_point() and not torch.isfinite(out2).all():
            return True
        if out2.is_floating_point():
            v = out2.to(dtype=torch.float32)
            std = v.std(unbiased=False) if v.dim() == 1 else v.std(dim=-1, unbiased=False).mean()
            return bool(float(std.item()) <= float(std_min))
        return False

    def _cached_mean_std(
        self: Self,
        mean_buf: torch.Tensor,
        std_buf: torch.Tensor,
        t: torch.Tensor,
        cache_key_prefix: str,
        *,
        master_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(t.device), int(t.shape[-1]), master_dtype)
        lock = getattr(self, "_stats_cache_lock", None)
        if lock is None:
            lock = Mutex()
            setattr(self, "_stats_cache_lock", lock)
        with lock:
            cache = getattr(self, f"_{cache_key_prefix}_stats_cache", None)
            if cache is None:
                cache = {}
                setattr(self, f"_{cache_key_prefix}_stats_cache", cache)
            cached = cache.get(key)
            if cached is None:
                mean_b = mean_buf.to(device=t.device, dtype=master_dtype)
                std_b = std_buf.to(device=t.device, dtype=master_dtype)
                if len(cache) > int(getattr(self, "_stats_cache_max", 8) or 8):
                    cache.clear()
                cache[key] = (mean_b, std_b)
                return mean_b, std_b
            return cached

    def _invalidate_stats_cache(self: Self) -> None:
        lock = getattr(self, "_stats_cache_lock", None)
        if lock is None:
            lock = Mutex()
            setattr(self, "_stats_cache_lock", lock)
        with lock:
            self._x_stats_cache.clear()
            self._y_stats_cache.clear()

    def _apply(
        self: Self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> "Scaler":
        recurse = kwargs.pop("recurse", True)
        if args:
            with contextlib.suppress(Exception):
                recurse = bool(args[0])
        try:
            super()._apply(fn, recurse=recurse)
        except TypeError:
            super()._apply(fn)
        with contextlib.suppress(Exception):
            for name in (
                "scale",
                "min_value",
                "max_value",
                "max_abs",
                "min_positive",
            ):
                t = getattr(self, name, None)
                if (
                    isinstance(t, torch.Tensor)
                    and t.is_floating_point()
                    and t.dtype != torch.float64
                ):
                    setattr(self, name, t.to(dtype=torch.float64))
        self._invalidate_stats_cache()
        return self

    def _load_from_state_dict(
        self: Self,
        state_dict: Mapping[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._invalidate_stats_cache()
        with contextlib.suppress(Exception):
            self._sanitize_nonfinite_bounds_inplace()
        with contextlib.suppress(Exception):
            self._restore_calib_state_after_load()

    @torch.no_grad()
    def _sanitize_nonfinite_bounds_inplace(self: Self) -> None:
        BIG32_MIN = float(torch.finfo(torch.float32).min)
        BIG32_MAX = float(torch.finfo(torch.float32).max)
        EXTREME_ABS = float(os.environ.get("ENN_SCALER_BOUNDS_EXTREME_ABS", "") or 1e307)

        def _fix_bound(name: str, fill: float) -> None:
            t = getattr(self, name, None)
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                if t.numel() == 0:
                    return
                tt = t.detach().clone()
                bad = (~torch.isfinite(tt)) | (tt.abs() >= EXTREME_ABS)
                if bad.any():
                    tt[bad] = fill
                    t.resize_(tt.shape).copy_(tt)

        def _fix_finite(name: str, fill: float, clamp32: bool = False) -> None:
            t = getattr(self, name, None)
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                if t.numel() == 0:
                    return
                tt = t.detach().clone()
                bad = ~torch.isfinite(tt)
                if bad.any():
                    tt[bad] = fill
                if clamp32:
                    tt = tt.clamp(min=BIG32_MIN, max=BIG32_MAX)
                t.resize_(tt.shape).copy_(tt)

        _fix_bound("y_min", float("-inf"))
        _fix_bound("y_max", float("inf"))
        _fix_bound("y_q_low", float("-inf"))
        _fix_bound("y_q_high", float("inf"))

        _fix_finite("y_out_scale", 1.0, clamp32=True)
        _fix_finite("y_out_bias", 0.0, clamp32=True)
        _fix_finite("y_out_clip_low", BIG32_MIN, clamp32=True)
        _fix_finite("y_out_clip_high", BIG32_MAX, clamp32=True)

        def _fix_mean(name: str, fill: float = 0.0) -> None:
            t = getattr(self, name, None)
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                if t.numel() == 0:
                    return
                tt = t.detach().clone()
                bad = (~torch.isfinite(tt)) | (tt.abs() >= EXTREME_ABS)
                if bad.any():
                    tt[bad] = float(fill)
                    t.resize_(tt.shape).copy_(tt)

        def _fix_std(name: str, fill: float = 1.0) -> None:
            t = getattr(self, name, None)
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                if t.numel() == 0:
                    return
                tt = t.detach().clone()
                bad = (~torch.isfinite(tt)) | (tt.abs() >= EXTREME_ABS) | (tt <= 0.0)
                if bad.any():
                    tt[bad] = float(fill)
                tt = tt.clamp(min=1e-12, max=BIG32_MAX)
                t.resize_(tt.shape).copy_(tt)

        _fix_mean("x_mean", 0.0)
        _fix_std("x_std", 1.0)
        _fix_mean("y_mean", 0.0)
        _fix_std("y_std", 1.0)

        with contextlib.suppress(Exception):
            if isinstance(self.y_min, torch.Tensor) and isinstance(self.y_max, torch.Tensor):
                lo = self.y_min
                hi = self.y_max
                if lo.numel() == hi.numel() and lo.is_floating_point():
                    swap = lo > hi
                    if swap.any():
                        lo2 = torch.minimum(lo, hi)
                        hi2 = torch.maximum(lo, hi)
                        self.y_min.resize_(lo2.shape).copy_(lo2)
                        self.y_max.resize_(hi2.shape).copy_(hi2)
            if isinstance(self.y_q_low, torch.Tensor) and isinstance(self.y_q_high, torch.Tensor):
                lo = self.y_q_low
                hi = self.y_q_high
                if lo.numel() == hi.numel() and lo.is_floating_point():
                    swap = lo > hi
                    if swap.any():
                        lo2 = torch.minimum(lo, hi)
                        hi2 = torch.maximum(lo, hi)
                        self.y_q_low.resize_(lo2.shape).copy_(lo2)
                        self.y_q_high.resize_(hi2.shape).copy_(hi2)
            if isinstance(self.y_out_clip_low, torch.Tensor) and isinstance(self.y_out_clip_high, torch.Tensor):
                lo = self.y_out_clip_low
                hi = self.y_out_clip_high
                if lo.numel() == hi.numel() and lo.is_floating_point():
                    swap = lo > hi
                    if swap.any():
                        lo2 = torch.minimum(lo, hi)
                        hi2 = torch.maximum(lo, hi)
                        self.y_out_clip_low.resize_(lo2.shape).copy_(lo2)
                        self.y_out_clip_high.resize_(hi2.shape).copy_(hi2)

    def _reset_output_ab_clip_only_state(self: Self) -> None:
        with contextlib.suppress(Exception):
            self._output_ab_clip_only = False
            self._output_ab_clip_only_reason = ""
            self._output_ab_clip_only_warned = False

    @torch.no_grad()
    def _restore_calib_state_after_load(self: Self) -> None:
        self._reset_output_ab_clip_only_state()
        if env_bool("ENN_DISABLE_OUTPUT_AB", default=False):
            self.disable_output_ab()

        mode = str(getattr(self, "calib_mode", "none") or "none").strip().lower()
        if mode not in {"none", "affine", "piecewise", "ab"}:
            mode = "none"

        inferred = "none"
        with contextlib.suppress(Exception):
            pwx = getattr(self, "pw_x", None)
            pwy = getattr(self, "pw_y", None)
            if (
                isinstance(pwx, torch.Tensor)
                and isinstance(pwy, torch.Tensor)
                and pwx.dim() == 2
                and pwy.dim() == 2
                and pwx.numel() >= 2
                and pwy.numel() >= 2
            ):
                inferred = "piecewise"

        if inferred == "none":
            with contextlib.suppress(Exception):
                a = getattr(self, "affine_a", None)
                b = getattr(self, "affine_b", None)
                if (
                    isinstance(a, torch.Tensor)
                    and isinstance(b, torch.Tensor)
                    and a.is_floating_point()
                    and b.is_floating_point()
                    and a.numel() > 0
                    and b.numel() > 0
                ):
                    a0 = a.detach().to(dtype=torch.float64).reshape(-1)
                    b0 = b.detach().to(dtype=torch.float64).reshape(-1)
                    if (a0 - 1.0).abs().max().item() > 1e-12 or b0.abs().max().item() > 1e-12:
                        inferred = "affine"

        if mode in {"none", "ab"}:
            self.calib_mode = inferred

        if env_bool("ENN_DISABLE_OUTPUT_AB", default=False):
            self.disable_output_ab()
            return

        enable_ab = False
        default_lo = float(torch.finfo(torch.float32).min)
        default_hi = float(torch.finfo(torch.float32).max)
        with contextlib.suppress(Exception):
            s = getattr(self, "y_out_scale", None)
            bb = getattr(self, "y_out_bias", None)
            lo = getattr(self, "y_out_clip_low", None)
            hi = getattr(self, "y_out_clip_high", None)

            if isinstance(s, torch.Tensor) and isinstance(bb, torch.Tensor) and s.numel() and bb.numel():
                s0 = s.detach().to(dtype=torch.float64).reshape(-1)
                b0 = bb.detach().to(dtype=torch.float64).reshape(-1)
                if (s0 - 1.0).abs().max().item() > 1e-12 or b0.abs().max().item() > 1e-12:
                    enable_ab = True

            if (
                (not enable_ab)
                and isinstance(lo, torch.Tensor)
                and isinstance(hi, torch.Tensor)
                and lo.numel()
                and hi.numel()
            ):
                lo0 = lo.detach().to(dtype=torch.float64).reshape(-1)
                hi0 = hi.detach().to(dtype=torch.float64).reshape(-1)
                if (
                    (lo0 - default_lo).abs().max().item() > 1e-6
                    or (hi0 - default_hi).abs().max().item() > 1e-6
                ):
                    enable_ab = True

            if enable_ab and env_bool("ENN_OUTPUT_AB_SANITY_DISABLE", default=True):
                try:
                    b_abs_th = float(env_float("ENN_OUTPUT_AB_SANITY_BIAS_ABS_MAX", 1000.0))
                    ratio_th = float(env_float("ENN_OUTPUT_AB_SANITY_BIAS_SPAN_RATIO_MAX", 1e3))
                    span_eps = float(env_float("ENN_OUTPUT_AB_SANITY_SPAN_EPS", 1e-12))
                    sample_max = max(1, int(env_int("ENN_OUTPUT_AB_SANITY_SAMPLE_MAX", 4096) or 4096))
                    disable_on_mismatch = bool(env_bool("ENN_OUTPUT_AB_SANITY_DISABLE_ON_SHAPE_MISMATCH", default=True))
                    disable_on_degen_span = bool(env_bool("ENN_OUTPUT_AB_SANITY_DISABLE_ON_DEGENERATE_SPAN", default=True))
                    degen_span_frac_max = float(env_float("ENN_OUTPUT_AB_SANITY_DEGENERATE_SPAN_FRAC_MAX", 0.01))

                    if isinstance(bb, torch.Tensor) and bb.numel():
                        b0 = bb.detach().reshape(-1)
                        b_abs_max = float(b0.abs().max().item())

                        ratio_max = float("nan")
                        span_min = float("nan")
                        span_max = float("nan")
                        bad_ratio_frac = 0.0
                        ratio_error = ""
                        neg_span_frac = 0.0
                        degen_span_frac = 0.0

                        len_s = int(s.numel()) if isinstance(s, torch.Tensor) else 0
                        len_b = int(b0.numel())
                        len_lo = int(lo.numel()) if isinstance(lo, torch.Tensor) else 0
                        len_hi = int(hi.numel()) if isinstance(hi, torch.Tensor) else 0
                        shape_mismatch = (
                            (len_s > 1 and len_b > 1 and len_s != len_b)
                            or (len_b > 1 and len_lo > 1 and len_b != len_lo)
                            or (len_b > 1 and len_hi > 1 and len_b != len_hi)
                            or (len_lo > 1 and len_hi > 1 and len_lo != len_hi)
                        )

                        if isinstance(lo, torch.Tensor) and isinstance(hi, torch.Tensor) and lo.numel() and hi.numel():
                            try:
                                lo0 = lo.detach().reshape(-1)
                                hi0 = hi.detach().reshape(-1)

                                lb = int(b0.numel())
                                llo = int(lo0.numel())
                                lhi = int(hi0.numel())
                                base_n = int(max(lb, llo, lhi))
                                if base_n > 0:
                                    sn = int(min(sample_max, base_n))
                                    if sn <= 1:
                                        idx_base = torch.tensor([base_n - 1], device=b0.device, dtype=torch.long)
                                    else:
                                        idx_base = torch.arange(sn, device=b0.device, dtype=torch.long)
                                        idx_base = (idx_base * (base_n - 1)) // max(1, (sn - 1))

                                    def _gather(v: torch.Tensor, lv: int) -> torch.Tensor:
                                        if lv <= 0:
                                            return v[:0]
                                        if lv == 1:
                                            return v[:1].expand((int(idx_base.numel()),))
                                        if lv == base_n:
                                            return v.index_select(0, idx_base)
                                        return v.index_select(0, (idx_base % lv))

                                    b_s = _gather(b0, lb).to(dtype=torch.float64)
                                    lo_s = _gather(lo0, llo).to(dtype=torch.float64)
                                    hi_s = _gather(hi0, lhi).to(dtype=torch.float64)
                                    span = (hi_s - lo_s).abs()
                                    span_min = float(span.min().item()) if span.numel() else float("nan")
                                    span_max = float(span.max().item()) if span.numel() else float("nan")
                                    ratio = b_s.abs() / torch.clamp(span, min=max(float(span_eps), 1e-12))
                                    ratio_max = float(ratio.max().item()) if ratio.numel() else float("nan")
                                    bad_ratio_frac = float((ratio > ratio_th).to(dtype=torch.float32).mean().item()) if ratio.numel() else 0.0
                                    neg_span_frac = float((hi_s < lo_s).to(dtype=torch.float32).mean().item()) if hi_s.numel() and lo_s.numel() else 0.0
                                    degen_span_frac = float((span <= max(float(span_eps), 1e-12)).to(dtype=torch.float32).mean().item()) if span.numel() else 0.0
                            except Exception as e:
                                ratio_error = f"{type(e).__name__}: {e}"

                        disable_reason = ""
                        if (math.isfinite(b_abs_max) and b_abs_max > b_abs_th):
                            disable_reason = "bias_abs_max"
                        elif (math.isfinite(ratio_max) and ratio_max > ratio_th):
                            disable_reason = "ratio_max"
                        elif disable_on_degen_span and (neg_span_frac > 0.0):
                            disable_reason = "negative_span"
                        elif disable_on_degen_span and (degen_span_frac > degen_span_frac_max):
                            disable_reason = "degenerate_span"
                        elif disable_on_mismatch and bool(shape_mismatch):
                            disable_reason = "shape_mismatch"

                        if disable_reason:
                            enable_ab = False
                            if (
                                str(disable_reason) != "shape_mismatch"
                                and env_bool("ENN_OUTPUT_AB_SANITY_CLIP_ONLY", default=True)
                            ):
                                self._output_ab_clip_only = True
                                self._output_ab_clip_only_reason = str(disable_reason)
                                if env_bool("ENN_OUTPUT_AB_SANITY_LOG", default=True):
                                    extra = f" ratio_error={ratio_error}" if ratio_error else ""
                                    _LOGGER.warning(
                                        "[ENN][scaler] output_ab params suspicious; enabling CLIP-ONLY fallback (scale/bias ignored): "
                                        "bias_abs_max=%.6g(th=%.6g) ratio_max=%.6g(th=%.6g) bad_ratio_frac=%.4f span[min/max]=%.6g/%.6g "
                                        "reason=%s%s",
                                        float(b_abs_max), float(b_abs_th), float(ratio_max), float(ratio_th),
                                        float(bad_ratio_frac), float(span_min), float(span_max),
                                        str(disable_reason), str(extra),
                                    )
                            else:
                                if env_bool("ENN_OUTPUT_AB_SANITY_LOG", default=True):
                                    extra = f" ratio_error={ratio_error}" if ratio_error else ""
                                    _LOGGER.warning(
                                        "[ENN][scaler] disabling output_ab after load (suspicious params): "
                                        "bias_abs_max=%.6g(th=%.6g) ratio_max=%.6g(th=%.6g) bad_ratio_frac=%.4f span[min/max]=%.6g/%.6g. "
                                        "reason=%s%s",
                                        float(b_abs_max), float(b_abs_th), float(ratio_max), float(ratio_th),
                                        float(bad_ratio_frac), float(span_min), float(span_max),
                                        str(disable_reason), str(extra),
                                    )
                            with contextlib.suppress(Exception):
                                setattr(self, "_output_ab_sanity_reason", str(disable_reason))
                                setattr(self, "_output_ab_sanity_bias_abs_max", float(b_abs_max))
                                setattr(self, "_output_ab_sanity_ratio_max", float(ratio_max))
                except Exception:
                    pass

        self.output_ab_enabled = bool(enable_ab)

        if (
            bool(self.output_ab_enabled)
            and env_bool("ENN_SCALER_OUTPUT_AB_LOG", default=False)
            and (not bool(getattr(self, "_output_ab_log_once", False)))
        ):
            self._output_ab_log_once = True
            try:
                max_elems = max(1, int(env_int("ENN_SCALER_OUTPUT_AB_LOG_MAX_ELEMS", 4096) or 4096))

                def _sample(v: torch.Tensor) -> tuple[torch.Tensor, bool, int]:
                    vv = v.detach().reshape(-1)
                    n = int(vv.numel())
                    if n <= 0:
                        return vv, False, 0
                    if n <= max_elems:
                        return vv, False, n
                    if max_elems == 1:
                        idx = torch.tensor([n - 1], device=vv.device, dtype=torch.long)
                    else:
                        idx = torch.arange(max_elems, device=vv.device, dtype=torch.long)
                        idx = (idx * (n - 1)) // max(1, (max_elems - 1))
                    return vv.index_select(0, idx), True, n

                s0 = getattr(self, "y_out_scale", None)
                b0 = getattr(self, "y_out_bias", None)
                lo0 = getattr(self, "y_out_clip_low", None)
                hi0 = getattr(self, "y_out_clip_high", None)
                if isinstance(s0, torch.Tensor) and isinstance(b0, torch.Tensor) and isinstance(lo0, torch.Tensor) and isinstance(hi0, torch.Tensor):
                    s_s, s_sampled, s_n = _sample(s0)
                    b_s, b_sampled, b_n = _sample(b0)
                    lo_s, lo_sampled, lo_n = _sample(lo0)
                    hi_s, hi_sampled, hi_n = _sample(hi0)

                    s_cpu = s_s.to(device="cpu", dtype=torch.float64)
                    b_cpu = b_s.to(device="cpu", dtype=torch.float64)
                    lo_cpu = lo_s.to(device="cpu", dtype=torch.float64)
                    hi_cpu = hi_s.to(device="cpu", dtype=torch.float64)
                    span_cpu = hi_cpu - lo_cpu

                    denom = torch.clamp(span_cpu.abs(), min=1e-12)
                    ratio = b_cpu.abs() / denom
                    frac_ratio_1e2 = float((ratio > 1e2).to(dtype=torch.float32).mean().item())
                    frac_ratio_1e3 = float((ratio > 1e3).to(dtype=torch.float32).mean().item())

                    _LOGGER.warning(
                        "[ENN][scaler] output_ab enabled after load (mode=%s). "
                        "scale[min/mean/max]=%.6g/%.6g/%.6g bias[abs_max]=%.6g "
                        "clip_low[min/max]=%.6g/%.6g clip_high[min/max]=%.6g/%.6g span[min/max]=%.6g/%.6g "
                        "|bias|/|span|>1e2 frac=%.3f >1e3 frac=%.3f (numel=%d, max_elems=%d, sampled=%s).",
                        str(getattr(self, "calib_mode", "")),
                        float(s_cpu.min().item()) if s_cpu.numel() else float("nan"),
                        float(s_cpu.mean().item()) if s_cpu.numel() else float("nan"),
                        float(s_cpu.max().item()) if s_cpu.numel() else float("nan"),
                        float(b_cpu.abs().max().item()) if b_cpu.numel() else float("nan"),
                        float(lo_cpu.min().item()) if lo_cpu.numel() else float("nan"),
                        float(lo_cpu.max().item()) if lo_cpu.numel() else float("nan"),
                        float(hi_cpu.min().item()) if hi_cpu.numel() else float("nan"),
                        float(hi_cpu.max().item()) if hi_cpu.numel() else float("nan"),
                        float(span_cpu.min().item()) if span_cpu.numel() else float("nan"),
                        float(span_cpu.max().item()) if span_cpu.numel() else float("nan"),
                        float(frac_ratio_1e2),
                        float(frac_ratio_1e3),
                        int(max(int(s_n), int(b_n), int(lo_n), int(hi_n))),
                        int(max_elems),
                        str(bool(s_sampled or b_sampled or lo_sampled or hi_sampled)),
                    )
            except Exception:
                pass

    def _update_stats_impl(
        self: Self,
        tensor: torch.Tensor,
        mean_buf: torch.Tensor,
        std_buf: torch.Tensor,
    ) -> None:
        if tensor.numel() == 0:
            return
        t_flat = (
            tensor.detach().view(-1, 1)
            if tensor.dim() == 1
            else tensor.detach().reshape(-1, tensor.shape[-1])
        ).to(dtype=torch.float64)
        mean, std = (
            t_flat.mean(dim=0),
            t_flat.std(dim=0, unbiased=False).clamp_min(self.eps),
        )
        if mean_buf.shape != mean.shape:
            mean_buf.resize_(mean.shape)
        if std_buf.shape != std.shape:
            std_buf.resize_(std.shape)
        mean_buf.copy_(mean)
        std_buf.copy_(std)
        self._invalidate_stats_cache()

    @torch.no_grad()
    def update_x(self: Self, x: torch.Tensor) -> None:
        self._update_stats_impl(x, self.x_mean, self.x_std)

    @torch.no_grad()
    def update_y(self: Self, y: torch.Tensor) -> None:
        self._update_stats_impl(y, self.y_mean, self.y_std)

    def _apply_affine_no_broadcast(
        self: Self, t: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        import torch.nn.functional as F

        if t.dim() == 0:
            return t
        orig_shape = t.shape
        if t.dim() == 1:
            t2 = t.unsqueeze(0)
        else:
            t2 = t.reshape(-1, orig_shape[-1])
        c = t2.shape[-1]
        w = weight.reshape(-1)
        b = bias.reshape(-1)
        if w.numel() == 1 and c != 1:
            w = w.expand((c,))
        if b.numel() == 1 and c != 1:
            b = b.expand((c,))
        running_mean = t2.new_zeros((c,))
        running_var = t2.new_ones((c,))
        out2 = F.batch_norm(
            t2,
            running_mean,
            running_var,
            weight=w.to(dtype=t2.dtype, device=t2.device),
            bias=b.to(dtype=t2.dtype, device=t2.device),
            training=False,
            momentum=0.0,
            eps=max(float(self.eps), 1e-6),
        )
        if t.dim() == 1:
            return out2.squeeze(0)
        return out2.reshape(orig_shape)

    def _normalize_impl(
        self: Self,
        t: torch.Tensor,
        mean_buf: torch.Tensor,
        std_buf: torch.Tensor,
        cache_key_prefix: str,
    ) -> torch.Tensor:
        if t.numel() == 0:
            return t
        input_dtype = t.dtype
        feature_dim = int(t.shape[-1])
        master_dtype = self._resolve_master_dtype_for_io(t)
        if master_dtype is torch.int64:
            master_dtype = torch.float64
        if is_symbolic() or is_export_or_trace() or is_compiling():
            mean_b = mean_buf.to(device=t.device, dtype=master_dtype)
            std_b = std_buf.to(device=t.device, dtype=master_dtype)
        else:
            mean_b, std_b = self._cached_mean_std(
                mean_buf, std_buf, t, cache_key_prefix, master_dtype=master_dtype
            )
        if (not is_symbolic()) and mean_b.numel() not in (1, int(feature_dim)):
            raise RuntimeError(
                f"Scaler feature dimension mismatch: t.shape={tuple(t.shape)} "
                f"mean.shape={tuple(mean_b.shape)} std.shape={tuple(std_b.shape)}"
            )
        orig_shape = tuple(t.shape)
        t2 = (t.reshape(1, -1) if t.dim() == 1 else t.reshape(-1, feature_dim)).to(dtype=master_dtype)
        std_min, eps_min, force_fp32_on_cuda = self._scaler_guard_params(t)
        eps_use = float(max(float(self.eps), float(eps_min)))
        denom = (std_b + eps_use).clamp_min(eps_use)
        out2 = (t2 - mean_b) / denom
        guard_enabled = self._scaler_guard_enabled()
        auto_guard = (
            (not guard_enabled)
            and (str(cache_key_prefix) == "x")
            and (not torch.is_grad_enabled())
            and (getattr(t.device, "type", None) == "cuda")
            and bool(env_bool("ENN_SCALER_GUARD_AUTO_INFER", True))
        )
        if (guard_enabled or auto_guard) and getattr(t.device, "type", None) == "cuda":
            if self._guard_is_collapse(out2, std_min=float(std_min)):
                md2 = torch.float32 if force_fp32_on_cuda else master_dtype
                mean2 = mean_buf.to(device=t.device, dtype=md2)
                std2 = std_buf.to(device=t.device, dtype=md2)
                t32 = (t.reshape(1, -1) if t.dim() == 1 else t.reshape(-1, feature_dim)).to(dtype=md2)
                eps_use2 = float(max(eps_use, 1e-4 if t.dtype == torch.bfloat16 else 1e-5))
                denom2 = (std2 + eps_use2).clamp_min(eps_use2)
                out2 = (t32 - mean2) / denom2
                if env_bool("ENN_SCALER_GUARD_LOG", bool(guard_enabled)):
                    warnings.warn(
                        "Scaler.normalize_x: detected collapse/nonfinite; retried this batch with safer fp32 math.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                if self._guard_is_collapse(out2, std_min=float(std_min)):
                    with contextlib.suppress(Exception):
                        in_std = t32.std(unbiased=False) if t32.dim() == 1 else t32.std(dim=-1, unbiased=False).mean()
                        batch_rows = int(t32.shape[0]) if t32.dim() >= 2 else 1
                        if float(in_std.item()) > float(std_min) * 10.0:
                            if batch_rows > 1:
                                mean3 = t32.mean(dim=0)
                                std3 = t32.std(dim=0, unbiased=False).clamp_min(eps_use2)
                                out2 = (t32 - mean3) / std3
                                if env_bool("ENN_SCALER_GUARD_LOG", bool(guard_enabled)):
                                    warnings.warn(
                                        "Scaler.normalize_x: collapse persisted; fell back to per-batch stats (stored stats look corrupted).",
                                        RuntimeWarning,
                                        stacklevel=2,
                                    )
                            elif t32.dim() >= 2 and int(t32.shape[1]) > 1:
                                if str(cache_key_prefix) == "x":
                                    with contextlib.suppress(Exception):
                                        in_scale = float(t32.abs().amax().item()) if t32.numel() else 0.0
                                    if not ("in_scale" in locals()) or not (in_scale > 0.0):
                                        in_scale = 1.0
                                    max_std = float(max(1.0, in_scale * 1000.0 + 1.0))
                                    feature_row = t32[0]
                                    feature_row = torch.where(
                                        torch.isfinite(feature_row),
                                        feature_row,
                                        torch.zeros_like(feature_row),
                                    )
                                    mean2v = mean2
                                    std2v = std2
                                    with contextlib.suppress(Exception):
                                        if int(mean2v.numel()) == int(feature_row.numel()):
                                            mean2v = mean2v.reshape(feature_row.shape)
                                    with contextlib.suppress(Exception):
                                        if int(std2v.numel()) == int(feature_row.numel()):
                                            std2v = std2v.reshape(feature_row.shape)
                                    if mean2v.shape != feature_row.shape:
                                        mean2v = feature_row
                                    if std2v.shape != feature_row.shape:
                                        std2v = feature_row.abs().clamp_min(1.0)
                                    finite_mean = torch.isfinite(mean2v)
                                    finite_std = torch.isfinite(std2v) & (std2v > eps_use2)
                                    mean3 = torch.where(finite_mean, mean2v, feature_row)
                                    std3 = torch.where(
                                        finite_std,
                                        std2v,
                                        feature_row.abs().clamp_min(1.0),
                                    ).clamp(min=eps_use2, max=max_std)
                                    with contextlib.suppress(Exception):
                                        mean_lim = float(max(1.0, in_scale * 1000.0 + 1.0))
                                        if mean3.numel() and float(mean3.abs().amax().item()) > mean_lim:
                                            mean3 = torch.zeros_like(mean3)
                                    out2 = (t32 - mean3) / std3
                                    if env_bool("ENN_SCALER_GUARD_LOG", bool(guard_enabled)):
                                        warnings.warn(
                                            "Scaler.normalize_x: collapse persisted; sanitized stored stats for single-row input (invalid entries fell back to input-derived values).",
                                            RuntimeWarning,
                                            stacklevel=2,
                                        )
                                else:
                                    mean3 = t32.mean(dim=1, keepdim=True)
                                    std3 = t32.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps_use2)
                                    out2 = (t32 - mean3) / std3
                                    if env_bool("ENN_SCALER_GUARD_LOG", bool(guard_enabled)):
                                        warnings.warn(
                                            "Scaler.normalize_x: collapse persisted; fell back to per-sample feature stats (single-row input; stored stats look corrupted).",
                                            RuntimeWarning,
                                            stacklevel=2,
                                        )
        out = out2.reshape(orig_shape) if t.dim() != 1 else out2.reshape(-1)
        if t.is_floating_point() and out.dtype != input_dtype and self._should_restore_input_dtype(input_dtype):
            out = out.to(dtype=input_dtype)
        return out

    def normalize_x(self: Self, x: torch.Tensor) -> torch.Tensor:
        return self._normalize_impl(x, self.x_mean, self.x_std, "x")

    def _denormalize_impl(
        self: Self,
        t: torch.Tensor,
        mean_buf: torch.Tensor,
        std_buf: torch.Tensor,
        cache_key_prefix: str,
    ) -> torch.Tensor:
        if t.numel() == 0:
            return t
        input_dtype = t.dtype
        feature_dim = int(t.shape[-1])
        master_dtype = self._resolve_master_dtype_for_io(t)
        if master_dtype is torch.int64:
            master_dtype = torch.float64
        if is_symbolic() or is_export_or_trace() or is_compiling():
            mean_b = mean_buf.to(device=t.device, dtype=master_dtype)
            std_b = std_buf.to(device=t.device, dtype=master_dtype)
        else:
            mean_b, std_b = self._cached_mean_std(
                mean_buf, std_buf, t, cache_key_prefix, master_dtype=master_dtype
            )
        if (not is_symbolic()) and mean_b.numel() not in (1, int(feature_dim)):
            raise RuntimeError(
                f"Scaler feature dimension mismatch: t.shape={tuple(t.shape)} "
                f"mean.shape={tuple(mean_b.shape)} std.shape={tuple(std_b.shape)}"
            )
        orig_shape = tuple(t.shape)
        t2 = (t.reshape(1, -1) if t.dim() == 1 else t.reshape(-1, feature_dim)).to(dtype=master_dtype)
        _, eps_min, _ = self._scaler_guard_params(t)
        eps_use = float(max(float(self.eps), float(eps_min)))
        scale = (std_b + eps_use).clamp_min(eps_use)
        out2 = t2 * scale + mean_b
        out = out2.reshape(orig_shape) if t.dim() != 1 else out2.reshape(-1)
        if t.is_floating_point() and out.dtype != input_dtype and self._should_restore_input_dtype(input_dtype):
            out = out.to(dtype=input_dtype)
        return out

    def denormalize_x(self: Self, x_scaled: torch.Tensor) -> torch.Tensor:
        return self._denormalize_impl(x_scaled, self.x_mean, self.x_std, "x")

    def _y_stats_vector(self: Self) -> Tuple[torch.Tensor, torch.Tensor]:
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
            self._invalidate_stats_cache()
            mean = self.y_mean
            std = self.y_std
        return mean, std

    def normalize_y(self: Self, y: torch.Tensor) -> torch.Tensor:
        orig_shape = y.shape
        mean_vec, std_vec = self._y_stats_vector()
        z = self._normalize_impl(
            y.view(1, -1) if y.dim() == 1 else y.view(y.shape[0], -1),
            mean_vec,
            std_vec,
            "y",
        )
        return z.view(-1) if y.dim() == 1 else z.view(orig_shape)

    def denormalize_y(self: Self, z: torch.Tensor) -> torch.Tensor:
        orig_shape = z.shape
        mean_vec, std_vec = self._y_stats_vector()
        y = self._denormalize_impl(
            z.view(1, -1) if z.dim() == 1 else z.view(z.shape[0], -1),
            mean_vec,
            std_vec,
            "y",
        )
        return y.view(-1) if z.dim() == 1 else y.view(orig_shape)

    def calibrate(self: Self, z_raw: torch.Tensor) -> torch.Tensor:
        disable_pw = env_bool(
            (
                "ENN_DISABLE_PIECEWISE_CALIB",
                "ENN_EXPORT_DISABLE_PIECEWISE_CALIB",
            ),
            default=False,
        )
        if disable_pw or is_symbolic() or is_meta_or_fake_tensor(z_raw):
            if self.calib_mode in ("piecewise", "affine"):
                return self.affine(z_raw)
            return z_raw
        match self.calib_mode:
            case "piecewise":
                if self.pw_x.numel() >= 2 and self.pw_y.numel() >= 2:
                    z = self._piecewise(z_raw)
                else:
                    z = z_raw
                return self._apply_output_ab(z)
            case "affine":
                return self._apply_output_ab(self.affine(z_raw))
            case "ab":
                return self._apply_output_ab(self._piecewise(z_raw))
            case "none":
                return self._apply_output_ab(z_raw)
            case _:
                return self._apply_output_ab(z_raw)

    def _apply_output_ab(self: Self, z: torch.Tensor) -> torch.Tensor:
        if z.numel() == 0:
            return z

        clip_only = bool(getattr(self, "_output_ab_clip_only", False)) or bool(
            env_bool("ENN_OUTPUT_AB_CLIP_ONLY", default=False)
        )
        enabled = bool(getattr(self, "output_ab_enabled", False))
        if (not enabled) and (not clip_only):
            return z

        if enabled and (not clip_only) and env_bool("ENN_OUTPUT_AB_SANITY_RUNTIME_DISABLE", default=True) and (not bool(getattr(self, "_output_ab_runtime_sanity_checked", False))):
            setattr(self, "_output_ab_runtime_sanity_checked", True)
            try:
                b_abs_th = float(env_float("ENN_OUTPUT_AB_SANITY_BIAS_ABS_MAX", 1000.0))
                disable_on_mismatch = bool(env_bool("ENN_OUTPUT_AB_SANITY_DISABLE_ON_SHAPE_MISMATCH", default=True))
                bb = getattr(self, "y_out_bias", None)
                s = getattr(self, "y_out_scale", None)
                lo = getattr(self, "y_out_clip_low", None)
                hi = getattr(self, "y_out_clip_high", None)
                len_s = int(s.numel()) if isinstance(s, torch.Tensor) else 0
                len_b = int(bb.numel()) if isinstance(bb, torch.Tensor) else 0
                len_lo = int(lo.numel()) if isinstance(lo, torch.Tensor) else 0
                len_hi = int(hi.numel()) if isinstance(hi, torch.Tensor) else 0
                shape_mismatch = (
                    (len_s > 1 and len_b > 1 and len_s != len_b)
                    or (len_b > 1 and len_lo > 1 and len_b != len_lo)
                    or (len_b > 1 and len_hi > 1 and len_b != len_hi)
                    or (len_lo > 1 and len_hi > 1 and len_lo != len_hi)
                )
                b_abs_max = float("nan")
                if isinstance(bb, torch.Tensor) and bb.numel():
                    b_abs_max = float(bb.detach().reshape(-1).abs().max().item())
                if (math.isfinite(b_abs_max) and b_abs_max > b_abs_th) or (disable_on_mismatch and bool(shape_mismatch)):
                    self.output_ab_enabled = False
                    if env_bool("ENN_OUTPUT_AB_SANITY_LOG", default=True) and (not bool(getattr(self, "_output_ab_runtime_sanity_warned", False))):
                        setattr(self, "_output_ab_runtime_sanity_warned", True)
                        _LOGGER.warning(
                            "[ENN][scaler] disabling output_ab at runtime (sanity): bias_abs_max=%.6g(th=%.6g) shape_mismatch=%s lens(s=%d,b=%d,lo=%d,hi=%d).",
                            float(b_abs_max), float(b_abs_th), str(bool(shape_mismatch)),
                            int(len_s), int(len_b), int(len_lo), int(len_hi),
                        )
                    return z
            except Exception:
                pass

        input_dtype = z.dtype
        master_dtype = self._resolve_master_dtype_for_io(z)
        if master_dtype is torch.int64:
            master_dtype = torch.float64
        zz = z.to(dtype=master_dtype)

        out = zz
        if enabled and (not clip_only):
            scale = self.y_out_scale.to(device=zz.device, dtype=zz.dtype)
            bias = self.y_out_bias.to(device=zz.device, dtype=zz.dtype)
            out = self._apply_affine_no_broadcast(zz, weight=scale, bias=bias)
        lo = self.y_out_clip_low.to(device=out.device, dtype=out.dtype)
        hi = self.y_out_clip_high.to(device=out.device, dtype=out.dtype)
        if out.is_floating_point():
            with contextlib.suppress(Exception):
                finfo = torch.finfo(out.dtype)
                lo = torch.where(torch.isfinite(lo), lo, out.new_tensor(float(finfo.min)))
                hi = torch.where(torch.isfinite(hi), hi, out.new_tensor(float(finfo.max)))
                clip_eps = float(os.environ.get("ENN_SCALER_OUTPUT_AB_CLIP_EPS", "") or 0.0)
                if clip_eps < 0.0:
                    clip_eps = 0.0
                mask = (hi == lo) if clip_eps == 0.0 else (hi - lo).abs() <= float(clip_eps)
                lo = torch.where(mask, out.new_tensor(float(finfo.min)), lo)
                hi = torch.where(mask, out.new_tensor(float(finfo.max)), hi)
        out = torch.minimum(torch.maximum(out, lo), hi)
        if z.is_floating_point() and out.dtype != input_dtype and self._should_restore_input_dtype(input_dtype):
            out = out.to(dtype=input_dtype)
        return out

    @torch.no_grad()
    def disable_output_ab(self: Self) -> None:
        self.output_ab_enabled = False
        self._reset_output_ab_clip_only_state()

    @torch.no_grad()
    def fit_output_ab(
        self: Self,
        pred_mean: torch.Tensor,
        pred_std: torch.Tensor,
        ref_mean: torch.Tensor,
        ref_std: torch.Tensor,
        clip_low=None,
        clip_high=None,
        eps=None,
        mix_alpha: float | None = None,
        scale_clamp: float | None = None,
        enable: bool = True,
    ) -> None:
        self._reset_output_ab_clip_only_state()
        pm = pred_mean.detach().reshape(-1).to(dtype=torch.float64)
        ps = pred_std.detach().reshape(-1).to(dtype=torch.float64)
        rm = ref_mean.detach().reshape(-1).to(dtype=torch.float64)
        rs = ref_std.detach().reshape(-1).to(dtype=torch.float64)
        if pm.numel() == 0 or ps.numel() == 0 or rm.numel() == 0 or rs.numel() == 0:
            self.disable_output_ab()
            return
        eps_f = float(self.eps if eps is None else eps)
        eps_f = max(eps_f, 1e-9)
        ps = torch.clamp(ps, min=eps_f)
        rs = torch.clamp(rs, min=eps_f)
        scale = rs / ps
        bias = rm - scale * pm
        if mix_alpha is not None:
            try:
                alpha = float(mix_alpha)
            except Exception:
                alpha = 1.0
            alpha = max(0.0, min(1.0, alpha))
            if alpha < 1.0:
                pm_g = pm.mean()
                ps_g = torch.clamp(ps.mean(), min=eps_f)
                rm_g = rm.mean()
                rs_g = torch.clamp(rs.mean(), min=eps_f)
                s_g = rs_g / ps_g
                b_g = rm_g - s_g * pm_g
                scale = alpha * scale + (1.0 - alpha) * s_g
                bias = alpha * bias + (1.0 - alpha) * b_g
        if scale_clamp is not None:
            try:
                sc = float(scale_clamp)
            except Exception:
                sc = 0.0
            if sc > 0.0:
                scale = torch.clamp(scale, min=1.0 / sc, max=sc)
        if not torch.isfinite(scale).all():
            scale = torch.where(torch.isfinite(scale), scale, torch.ones_like(scale))
        if not torch.isfinite(bias).all():
            bias = torch.where(torch.isfinite(bias), bias, torch.zeros_like(bias))

        self.y_out_scale.resize_(scale.shape).copy_(scale)
        self.y_out_bias.resize_(bias.shape).copy_(bias)

        BIG32_MIN = float(torch.finfo(torch.float32).min)
        BIG32_MAX = float(torch.finfo(torch.float32).max)
        if clip_low is None:
            lo = torch.full_like(scale, BIG32_MIN)
        else:
            lo = torch.as_tensor(clip_low, dtype=torch.float64).reshape(-1)
        if clip_high is None:
            hi = torch.full_like(scale, BIG32_MAX)
        else:
            hi = torch.as_tensor(clip_high, dtype=torch.float64).reshape(-1)
        if lo.numel() == 1 and scale.numel() != 1:
            lo = lo.expand_as(scale)
        if hi.numel() == 1 and scale.numel() != 1:
            hi = hi.expand_as(scale)
        if not torch.isfinite(lo).all():
            lo = torch.where(torch.isfinite(lo), lo, torch.full_like(lo, BIG32_MIN))
        if not torch.isfinite(hi).all():
            hi = torch.where(torch.isfinite(hi), hi, torch.full_like(hi, BIG32_MAX))
        lo = lo.clamp(min=BIG32_MIN, max=BIG32_MAX)
        hi = hi.clamp(min=BIG32_MIN, max=BIG32_MAX)
        lo2 = torch.minimum(lo, hi)
        hi2 = torch.maximum(lo, hi)
        lo, hi = lo2, hi2

        self.y_out_clip_low.resize_(lo.shape).copy_(lo)
        self.y_out_clip_high.resize_(hi.shape).copy_(hi)
        self.output_ab_enabled = bool(enable)
        self._reset_output_ab_clip_only_state()
        if self.output_ab_enabled and self.calib_mode == "none":
            self.calib_mode = "ab"

    def inverse_calibrate(
        self: Self,
        z_target: torch.Tensor,
        *,
        apply_output_ab: bool = True,
    ) -> torch.Tensor:
        if z_target.numel() == 0:
            return z_target
        mode = str(getattr(self, "calib_mode", "none") or "none").strip().lower()
        if is_symbolic() or is_meta_or_fake_tensor(z_target):
            return z_target

        input_dtype = z_target.dtype
        master_dtype = self._resolve_master_dtype_for_io(z_target)
        if master_dtype is torch.int64:
            master_dtype = torch.float64
        z = z_target.to(dtype=master_dtype)

        def _inv_output_ab(t: torch.Tensor) -> torch.Tensor:
            if (
                t.numel() == 0
                or (not apply_output_ab)
                or not bool(getattr(self, "output_ab_enabled", False))
            ):
                return t
            scale = self.y_out_scale.to(device=t.device, dtype=t.dtype).reshape(-1)
            bias = self.y_out_bias.to(device=t.device, dtype=t.dtype).reshape(-1)
            if t.dim() == 0:
                return t
            orig = t.shape
            t2 = t.unsqueeze(0) if t.dim() == 1 else t.reshape(-1, orig[-1])
            c = int(t2.shape[-1])
            if scale.numel() == 1 and c != 1:
                s2 = scale.expand((c,))
            else:
                s2 = scale[:c] if scale.numel() >= c else scale.expand((c,))
            if bias.numel() == 1 and c != 1:
                b2 = bias.expand((c,))
            else:
                b2 = bias[:c] if bias.numel() >= c else bias.expand((c,))
            eps_v = max(float(self.eps), 1e-6)
            mask = s2.abs() < eps_v
            s_safe = torch.where(mask, torch.ones_like(s2), s2)
            out2 = (t2 - b2.view(1, -1)) / s_safe.view(1, -1)
            if mask.any():
                out2[:, mask] = t2[:, mask]
            return out2.squeeze(0) if t.dim() == 1 else out2.reshape(orig)

        def _inv_affine(t: torch.Tensor) -> torch.Tensor:
            a = self.affine_a.to(device=t.device, dtype=t.dtype).reshape(-1)
            b = self.affine_b.to(device=t.device, dtype=t.dtype).reshape(-1)
            if t.dim() == 0:
                return t
            orig = t.shape
            t2 = t.unsqueeze(0) if t.dim() == 1 else t.reshape(-1, orig[-1])
            c = int(t2.shape[-1])
            if a.numel() == 1 and c != 1:
                a2 = a.expand((c,))
            else:
                a2 = a[:c] if a.numel() >= c else a.expand((c,))
            if b.numel() == 1 and c != 1:
                b2 = b.expand((c,))
            else:
                b2 = b[:c] if b.numel() >= c else b.expand((c,))
            eps_v = max(float(self.eps), 1e-6)
            mask = a2.abs() < eps_v
            a_safe = torch.where(mask, torch.ones_like(a2), a2)
            out2 = (t2 - b2.view(1, -1)) / a_safe.view(1, -1)
            if mask.any():
                out2[:, mask] = t2[:, mask]
            return out2.squeeze(0) if t.dim() == 1 else out2.reshape(orig)

        def _inv_piecewise(t: torch.Tensor) -> torch.Tensor:
            if self.pw_x.numel() < 2 or self.pw_y.numel() < 2:
                return t
            if self.pw_x.dim() != 2 or self.pw_y.dim() != 2:
                return t
            pw_x = self.pw_x
            pw_y = self.pw_y
            c_saved, kx = pw_x.shape
            _, ky = pw_y.shape
            k = int(min(kx, ky))
            if k < 2:
                return t
            pw_x = pw_x[:, :k]
            pw_y = pw_y[:, :k]

            orig = t.shape
            tt = t.unsqueeze(-1) if t.ndim == 1 else t
            tt = tt.reshape(-1, int(tt.shape[-1]))
            _, c_target = tt.shape
            out = torch.empty_like(tt)
            last = max(0, int(c_saved) - 1)

            for j in range(int(c_target)):
                src = j if j < int(c_saved) else last
                v = tt[:, j]
                kyv = pw_y[src].to(device=v.device, dtype=v.dtype)
                kxv = pw_x[src].to(device=v.device, dtype=v.dtype)
                if not (is_symbolic() or is_meta_or_fake_tensor(kyv)):
                    with contextlib.suppress(Exception):
                        if not bool((kyv[1:] >= kyv[:-1]).all().item()):
                            ky_sorted, order = torch.sort(kyv)
                            kxv = kxv[order]
                            kyv = ky_sorted
                idx = torch.bucketize(v, kyv)
                idx = idx.clamp(1, kyv.numel() - 1)
                y0 = kyv[idx - 1]
                y1 = kyv[idx]
                x0 = kxv[idx - 1]
                x1 = kxv[idx]
                tlin = (v - y0) / (y1 - y0 + self.eps)
                out[:, j] = x0 + tlin * (x1 - x0)
            return out.reshape(orig)

        z = _inv_output_ab(z)
        if mode in ("none", ""):
            out = z
        elif mode == "affine":
            out = _inv_affine(z)
        elif mode == "piecewise":
            out = _inv_piecewise(z)
        elif mode == "ab":
            out = _inv_affine(_inv_piecewise(z))
        else:
            out = z

        if z_target.is_floating_point() and out.dtype != input_dtype and self._should_restore_input_dtype(input_dtype):
            out = out.to(dtype=input_dtype)
        return out

    def affine(self: Self, z_raw: torch.Tensor) -> torch.Tensor:
        if z_raw.numel() == 0:
            return z_raw
        input_dtype = z_raw.dtype
        master_dtype = self._resolve_master_dtype_for_io(z_raw)
        if master_dtype is torch.int64:
            master_dtype = torch.float64
        z = z_raw.to(dtype=master_dtype)
        a = self.affine_a.to(device=z.device, dtype=master_dtype)
        b = self.affine_b.to(device=z.device, dtype=master_dtype)
        out = self._apply_affine_no_broadcast(z, weight=a, bias=b)
        if z_raw.is_floating_point() and out.dtype != input_dtype and self._should_restore_input_dtype(input_dtype):
            out = out.to(dtype=input_dtype)
        return out

    @torch.no_grad()
    def set_affine(self: Self, a: torch.Tensor, b: torch.Tensor) -> None:
        if self.affine_a.shape != a.shape:
            self.affine_a.resize_(a.shape)
        if self.affine_b.shape != b.shape:
            self.affine_b.resize_(b.shape)
        self.affine_a.copy_(
            a.to(self.affine_a.device, dtype=self.affine_a.dtype)
        )
        self.affine_b.copy_(
            b.to(self.affine_b.device, dtype=self.affine_b.dtype)
        )
        self.pw_x.resize_((1, 1))
        self.pw_x.zero_()
        self.pw_y.resize_((1, 1))
        self.pw_y.zero_()
        self.calib_mode = "affine"

    @torch.no_grad()
    def fit(
        self: Self,
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
    def _fit_affine(
        self: Self, z_raw: torch.Tensor, z_true: torch.Tensor
    ) -> None:
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
        tiny_mask = denom.abs() < self.eps
        denom_safe = torch.where(tiny_mask, torch.ones_like(denom), denom)
        a64 = num / denom_safe
        b64 = y_mean - a64 * x_mean
        a64[tiny_mask] = 1.0
        b64[tiny_mask] = 0.0
        a = a64.to(dtype=torch.float64, device=self.affine_a.device)
        b = b64.to(dtype=torch.float64, device=self.affine_b.device)
        if self.affine_a.shape != a.shape:
            self.affine_a.resize_(a.shape)
        if self.affine_b.shape != b.shape:
            self.affine_b.resize_(b.shape)
        self.affine_a.copy_(a)
        self.affine_b.copy_(b)
        self.pw_x.resize_((1, 1))
        self.pw_x.zero_()
        self.pw_y.resize_((1, 1))
        self.pw_y.zero_()

    @torch.no_grad()
    def _fit_piecewise(
        self: Self,
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
        knots_x = torch.empty(C, num_bins, dtype=torch.float64, device=device)
        knots_y = torch.empty(C, num_bins, dtype=torch.float64, device=device)
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
            knots_x[j] = qx.to(dtype=torch.float64, device=device)
            knots_y[j] = qy.to(dtype=torch.float64, device=device)
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

    def _piecewise(self: Self, z_raw: torch.Tensor) -> torch.Tensor:
        if (
            env_bool(
                (
                    "ENN_DISABLE_PIECEWISE_CALIB",
                    "ENN_EXPORT_DISABLE_PIECEWISE_CALIB",
                ),
                default=False,
            )
            or is_symbolic()
            or is_meta_or_fake_tensor(z_raw)
        ):
            return (
                self.affine(z_raw)
                if self.calib_mode in ("piecewise", "affine")
                else z_raw
            )
        if self.pw_x.numel() < 2 or self.pw_y.numel() < 2:
            return z_raw
        if self.pw_x.dim() != 2 or self.pw_y.dim() != 2:
            return z_raw
        pw_x = self.pw_x
        pw_y = self.pw_y
        C_saved, Kx = pw_x.shape
        _, Ky = pw_y.shape
        K = int(min(Kx, Ky))
        if K < 2:
            return z_raw
        pw_x = pw_x[:, :K]
        pw_y = pw_y[:, :K]
        orig_shape = z_raw.shape
        z = z_raw.unsqueeze(-1) if z_raw.ndim == 1 else z_raw
        z = z.reshape(-1, int(z.shape[-1]))
        _, C_target = z.shape
        device = z.device
        dtype = z.dtype
        out = torch.empty_like(z)
        last_idx = max(0, int(C_saved) - 1)
        for j in range(int(C_target)):
            src_j = j if j < int(C_saved) else last_idx
            xj = z[:, j]
            knots_x = pw_x[src_j].to(device=device, dtype=dtype)
            knots_y = pw_y[src_j].to(device=device, dtype=dtype)
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


class Recorder(nn.Module):
    __enn_precision_exempt__: bool = True

    def __init__(self: Self) -> None:
        super().__init__()
        self.__enn_precision_exempt__ = True
        self.register_buffer(
            "start", torch.zeros(1, dtype=torch.float64), persistent=True
        )
        self.register_buffer(
            "end", torch.zeros(1, dtype=torch.float64), persistent=True
        )
        self.timezone: str = "UTC"
        self.register_buffer(
            "peers", torch.zeros(1, dtype=torch.int64), persistent=True
        )
        self.register_buffer(
            "epochs", torch.zeros(1, dtype=torch.int64), persistent=True
        )
        self.os: str = ""
        self.kernel: str = ""
        self.cpu: List[str] = []
        self.arch: List[str] = []
        self.ram_gb: float = 0.0
        self.python: str = ""
        self.backends: List[str] = []
        self.register_buffer(
            "sampled_n", torch.zeros(1, dtype=torch.int64), persistent=True
        )
        self.register_buffer(
            "sampled_x_mean",
            torch.zeros(1, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "sampled_x_var",
            torch.zeros(1, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "sampled_x_min",
            torch.full((1,), torch.finfo(torch.float64).max, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "sampled_x_max",
            torch.full((1,), torch.finfo(torch.float64).min, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "sampled_y_mean",
            torch.zeros(1, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "sampled_y_var",
            torch.zeros(1, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "sampled_y_min",
            torch.full((1,), torch.finfo(torch.float64).max, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "sampled_y_max",
            torch.full((1,), torch.finfo(torch.float64).min, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "reduced_n", torch.zeros(1, dtype=torch.int64), persistent=True
        )
        self.register_buffer(
            "reduced_x_mean",
            torch.zeros(1, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "reduced_x_var",
            torch.zeros(1, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "reduced_x_min",
            torch.full((1,), torch.finfo(torch.float64).max, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "reduced_x_max",
            torch.full((1,), torch.finfo(torch.float64).min, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "reduced_y_mean",
            torch.zeros(1, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "reduced_y_var",
            torch.zeros(1, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "reduced_y_min",
            torch.full((1,), torch.finfo(torch.float64).max, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "reduced_y_max",
            torch.full((1,), torch.finfo(torch.float64).min, dtype=torch.float64),
            persistent=True,
        )
        self._global_step: int = 0
        self._records: List[Dict[str, Any]] = []
        self.max_history_steps: int = 0

    def _apply(
        self: Self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        recurse = kwargs.pop("recurse", True)
        if args:
            with contextlib.suppress(Exception):
                recurse = bool(args[0])
        try:
            out = super()._apply(fn, recurse=recurse)
        except TypeError:
            out = super()._apply(fn)
        with torch.no_grad():
            for name, buf in list(getattr(self, "_buffers", {}).items()):
                if not isinstance(buf, torch.Tensor):
                    continue
                if buf.is_floating_point() and buf.dtype is not torch.float64:
                    with contextlib.suppress(Exception):
                        self._buffers[name] = buf.to(dtype=torch.float64)
                elif (
                    buf.dtype
                    in (
                        torch.int8,
                        torch.int16,
                        torch.int32,
                        torch.int64,
                        torch.uint8,
                    )
                    and buf.dtype is not torch.int64
                ):
                    with contextlib.suppress(Exception):
                        self._buffers[name] = buf.to(dtype=torch.int64)
        return out

    @torch.no_grad()
    def start_session(
        self: Self, start_posix: float, timezone: Optional[str] = None
    ) -> None:
        self.start.fill_(round(float(start_posix), 6))
        if timezone is None or not str(timezone).strip():
            try:
                import datetime
                import time

                now = datetime.datetime.now().astimezone()
                tzinfo = now.tzinfo
                tz_key = (
                    getattr(tzinfo, "key", None)
                    if tzinfo is not None
                    else None
                )
                tz_name = tzinfo.tzname(now) if tzinfo is not None else None
                tz_env = None
                try:
                    tz_env = time.tzname[0]
                except (AttributeError, IndexError, TypeError):
                    tz_env = None
                tz = tz_key or tz_name or tz_env or "UTC"
                self.timezone = str(tz)
            except Exception:
                self.timezone = "UTC"
        else:
            self.timezone = str(timezone)

    @torch.no_grad()
    def end_session(self: Self, end_posix: float, peers: int) -> None:
        self.end.fill_(round(float(end_posix), 6))
        self.peers.fill_(int(peers))

    @torch.no_grad()
    def set_epochs(self: Self, epochs: int) -> None:
        self.epochs.fill_(max(0, int(epochs)))

    @torch.no_grad()
    def set_system_info(
        self: Self,
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
            from ..core.system import CPU

            n_cores = max(1, int(CPU.count() or 1))
            model_name: Optional[str] = None
            with contextlib.suppress(Exception):
                info = CPU.info()
                first = info.split(";", 1)[0]
                cand = first.split(":", 1)[1] if ":" in first else first
                cand = str(cand).strip()
                if cand:
                    model_name = cand
            if not model_name:
                model_name = platform.processor() or (
                    cpu_list[0] if cpu_list else "Unknown CPU"
                )
            arch_name = platform.machine() or (
                arch_list[0] if arch_list else "unknown"
            )
            cpu_models = [str(model_name) for _ in range(int(n_cores))]
            arch_norm = [str(arch_name) for _ in range(int(n_cores))]
        except Exception:
            cpu_models = list(cpu_list)
            arch_norm = list(arch_list)
        self.cpu = cpu_models
        self.arch = arch_norm
        try:
            from ..core.system import Memory

            total_bytes = Memory.total()
            if total_bytes is not None and int(total_bytes) > 0:
                self.ram_gb = float(round(float(total_bytes) / (1024.0**3), 2))
            else:
                self.ram_gb = float(ram_gb)
        except Exception:
            self.ram_gb = float(ram_gb)
        backend_devices: List[str] = []
        try:
            from ..core.system import get_num_accelerators
            from ..core.system import is_accelerator_available

            if is_accelerator_available("cuda"):
                num_cuda = int(get_num_accelerators("cuda") or 0)
                for idx in range(num_cuda):
                    try:
                        name = torch.cuda.get_device_name(idx)
                    except Exception:
                        name = "CUDA Device"
                    backend_devices.append(f"cuda:{idx}, {name}")
            if is_accelerator_available("xpu"):
                num_xpu = int(get_num_accelerators("xpu") or 0)
                xpu_mod = getattr(torch, "xpu", None)
                get_name = (
                    getattr(xpu_mod, "get_device_name", None)
                    if xpu_mod is not None
                    else None
                )
                for idx in range(num_xpu):
                    name = "XPU Device"
                    if callable(get_name):
                        with contextlib.suppress(Exception):
                            name = str(get_name(idx) or name)
                    backend_devices.append(f"xpu:{idx}, {name}")
            if is_accelerator_available("mps"):
                chip_name = platform.processor() or "Apple Silicon"
                backend_devices.append(f"mps:0, {chip_name}")
            for idx, model_name in enumerate(cpu_models):
                backend_devices.append(f"cpu:{idx}, {model_name}")
        except Exception:
            backend_devices = list(backends)
        self.backends = backend_devices

    def _accumulate(
        self: Self,
        count_buf: torch.Tensor,
        mean_buf: torch.Tensor,
        var_buf: torch.Tensor,
        min_buf: torch.Tensor,
        max_buf: torch.Tensor,
        new_n: int,
        new_mean: torch.Tensor,
        new_var: torch.Tensor,
        new_min: torch.Tensor,
        new_max: torch.Tensor,
    ) -> None:
        n = int(count_buf.item())
        total = n + new_n
        w_old, w_new = (n / total, new_n / total) if total > 0 else (0.0, 0.0)
        count_buf.fill_(total)
        mean_buf.mul_(w_old).add_(new_mean * w_new)
        var_buf.mul_(w_old).add_(new_var * w_new)
        min_buf.copy_(torch.minimum(min_buf, new_min.view(1)))
        max_buf.copy_(torch.maximum(max_buf, new_max.view(1)))

    @torch.no_grad()
    def record_batch(
        self: Self,
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
        dev = self.sampled_x_mean.device
        xv, xm, xmn, xmx = [
            t.to(device=dev, dtype=torch.float64) for t in _tensor_stats(x_det)
        ]
        yv, ym, ymn, ymx = [
            t.to(device=dev, dtype=torch.float64) for t in _tensor_stats(y_det)
        ]
        if use_for_sample:
            self._accumulate(
                self.sampled_n,
                self.sampled_x_mean,
                self.sampled_x_var,
                self.sampled_x_min,
                self.sampled_x_max,
                1,
                xm,
                xv,
                xmn,
                xmx,
            )
            self._accumulate(
                self.sampled_n,
                self.sampled_y_mean,
                self.sampled_y_var,
                self.sampled_y_min,
                self.sampled_y_max,
                1,
                ym,
                yv,
                ymn,
                ymx,
            )
        if use_for_reduced:
            self._accumulate(
                self.reduced_n,
                self.reduced_x_mean,
                self.reduced_x_var,
                self.reduced_x_min,
                self.reduced_x_max,
                1,
                xm,
                xv,
                xmn,
                xmx,
            )
            self._accumulate(
                self.reduced_n,
                self.reduced_y_mean,
                self.reduced_y_var,
                self.reduced_y_min,
                self.reduced_y_max,
                1,
                ym,
                yv,
                ymn,
                ymx,
            )
        self._append(
            xm=xm,
            xvar=xv,
            xmin=xmn,
            xmax=xmx,
            ym=ym,
            yvar=yv,
            ymin=ymn,
            ymax=ymx,
            batch_size=int(x.shape[0]),
            step=step,
            extra=extra,
        )

    def _append(
        self: Self,
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
        rec: Dict[str, Any] = {
            "timestep": t,
            "batch_size": int(batch_size),
            "batch_x_mean": float(xm.item()),
            "batch_x_var": float(xvar.item()),
            "batch_x_min": float(xmin.item()),
            "batch_x_max": float(xmax.item()),
            "batch_y_mean": float(ym.item()),
            "batch_y_var": float(yvar.item()),
            "batch_y_min": float(ymin.item()),
            "batch_y_max": float(ymax.item()),
            "sampled_n": int(self.sampled_n.item()),
            "sampled_x_mean": float(self.sampled_x_mean.item()),
            "sampled_x_var": float(self.sampled_x_var.item()),
            "sampled_x_min": float(self.sampled_x_min.item()),
            "sampled_x_max": float(self.sampled_x_max.item()),
            "sampled_y_mean": float(self.sampled_y_mean.item()),
            "sampled_y_var": float(self.sampled_y_var.item()),
            "sampled_y_min": float(self.sampled_y_min.item()),
            "sampled_y_max": float(self.sampled_y_max.item()),
            "reduced_n": int(self.reduced_n.item()),
            "reduced_x_mean": float(self.reduced_x_mean.item()),
            "reduced_x_var": float(self.reduced_x_var.item()),
            "reduced_x_min": float(self.reduced_x_min.item()),
            "reduced_x_max": float(self.reduced_x_max.item()),
            "reduced_y_mean": float(self.reduced_y_mean.item()),
            "reduced_y_var": float(self.reduced_y_var.item()),
            "reduced_y_min": float(self.reduced_y_min.item()),
            "reduced_y_max": float(self.reduced_y_max.item()),
        }
        if extra is not None:
            rec["extra"] = dict(extra)
        self._records.append(rec)
        max_steps = int(self.max_history_steps or 0)
        if max_steps > 0 and len(self._records) > max_steps:
            overflow = len(self._records) - max_steps
            if overflow > 0:
                del self._records[:overflow]

    def save(self: Self) -> Sequence[Mapping[str, Any]]:
        return list(self._records)

    def clear(self: Self) -> None:
        self._records.clear()
        self._global_step = 0

    def _apply(
        self: Self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> "Recorder":
        recurse = kwargs.pop("recurse", True)
        if args:
            with contextlib.suppress(Exception):
                recurse = bool(args[0])
        try:
            super()._apply(fn, recurse=recurse)
        except TypeError:
            super()._apply(fn)
        with contextlib.suppress(Exception):
            for name, buf in self._buffers.items():
                if buf is None or (not isinstance(buf, torch.Tensor)):
                    continue
                if buf.is_floating_point():
                    if buf.dtype != torch.float64:
                        setattr(self, name, buf.to(dtype=torch.float64))
                else:
                    if buf.dtype in (
                        torch.int8,
                        torch.int16,
                        torch.int32,
                        torch.int64,
                    ):
                        if buf.dtype != torch.int64:
                            setattr(self, name, buf.to(dtype=torch.int64))
        return self
