# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import os
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
from ..core.datatypes import env_bool, env_int
from ..core.policies import ATTENTION_POLICY, AttentionBackend
from .activations import GeGLU
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
)
try:
    from torch.nn.attention.flex_attention import create_block_mask

    _HAS_FLEX_ATTENTION_LIB = True
except ImportError:
    create_block_mask = None
    _HAS_FLEX_ATTENTION_LIB = False

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
    tracing = False
    try:
        tracing = bool(is_symbolic())
    except Exception:
        tracing = False
    if not tracing:
        try:
            tracing = bool(torch.jit.is_tracing() or torch.jit.is_scripting())
        except Exception:
            tracing = False
    L = seq_len if tracing else int(seq_len)
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
        if self.per_batch:
            return score + self.bias_bk[b, kv_idx]
        return score + self.bias_bk[0, kv_idx]


def _coerce_attn_bias_to_bk(
    attn_bias: torch.Tensor, *, B: int, K: int, like: torch.Tensor
) -> torch.Tensor:
    t = attn_bias
    if t.dim() == 4:
        if int(t.shape[-1]) != int(K):
            raise RuntimeError(f"attn_bias K mismatch: {tuple(t.shape)} vs K={int(K)}")
        b0 = int(t.shape[0])
        if b0 not in (1, int(B)):
            raise RuntimeError(f"attn_bias B mismatch: {tuple(t.shape)} vs B={int(B)}")
        t = t.reshape(b0, int(K))
    elif t.dim() == 2:
        if int(t.shape[1]) != int(K) or int(t.shape[0]) not in (1, int(B)):
            raise RuntimeError(f"attn_bias shape mismatch: {tuple(t.shape)} vs (1|B,K)=({int(B)},{int(K)})")
    elif t.dim() == 1:
        if int(t.numel()) != int(K):
            raise RuntimeError(f"attn_bias len mismatch: {int(t.numel())} vs K={int(K)}")
        t = t.view(1, int(K))
    else:
        raise RuntimeError(f"Unsupported attn_bias rank {int(t.dim())}: {tuple(t.shape)}")

    return t.to(device=like.device, dtype=like.dtype, non_blocking=True).contiguous()

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
        del args, kwargs
        self.msr = MultiScaleRetention(d_model, nhead)
        self.mode = str(mode or "temporal").strip().lower()

    @staticmethod
    def _coerce_mode(mode: Optional[str]) -> str:
        if mode is None:
            return "temporal"
        m = str(mode).strip().lower()
        if m in ("t", "temporal", "time", "causal"):
            return "temporal"
        if m in (
            "s",
            "spatial",
            "space",
            "bi",
            "bidir",
            "bidirectional",
            "noncausal",
            "non-causal",
        ):
            return "spatial"
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
        if mlp_ratio is not None:
            ffn_ratio = mlp_ratio
        hidden = int(self.embed_dim * float(ffn_ratio))
        if activation.lower() == "gelu":
            act = nn.GELU()
        elif activation.lower() == "silu":
            act = nn.SiLU()
        elif activation.lower() == "relu":
            act = nn.ReLU()
        else:
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

    def _get_mask(self, L: int, device: torch.device) -> torch.Tensor:
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
        x_norm = self.norm1(x)
        kv = x_norm if context is None else self.norm1(context)
        q_pad = key_padding_mask
        if context is not None:
            q_pad = None
        use_flex = (
            context is None
            and (not need_weights)
            and (not return_attn_mask)
            and key_padding_mask is None
            and _HAS_FLEX_ATTENTION
            and getattr(get_flex_kernel(), "has_torch_backend", False)
            and (self._get_torch_mha() is not None)
            and (not is_export_or_trace())
            and (not is_compiling())
        )
        attn_mask_keep: Optional[torch.Tensor] = None
        attn_weights = None
        if use_flex:
            q, k, v, out_proj = self._project_qkv_for_flex(x_norm)
            _ensure_flex_kwargs_initialized()
            flex_supports_score_mod = "score_mod" in _FLEX_KWARGS
            flex_supports_block_mask = "block_mask" in _FLEX_KWARGS
            use_score_mod = bool(q.is_cuda and flex_supports_score_mod)
            score_mod = self._get_flex_score_mod(int(L)) if use_score_mod else None
            block_mask = None
            if (not use_score_mod) and flex_supports_block_mask:
                block_mask = self._get_flex_block_mask(int(L), int(B), device)
            needs_mask = bool(self.causal) or (self.window_size is not None) or (int(self.dilation) != 1)
            if needs_mask and (not use_score_mod) and (block_mask is None):
                use_flex = False
            if use_flex:
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
                attn_out = out_proj(a)
        if not use_flex:
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

    def forward(
        self: Self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor
    ) -> torch.Tensor:
        qn = self.norm_q(q_tokens)
        kvn = self.norm_kv(kv_tokens)
        ctx, _ = self.attn(qn, kvn, kvn, need_weights=False)
        ctx = self.out_proj(ctx)
        return q_tokens + self.drop_path(self.dropout(ctx))


class Resampler(nn.Module):
    def __init__(
        self: Self,
        d_model: int,
        nhead: int,
        *args: Any,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_type: str = "layernorm",
        bias: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del args, kwargs
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
        self.dropout = nn.Dropout(self.dropout_p)
        self.drop_path = StochasticDepth(p=float(drop_path), mode="row")
        self.norm_ffn = _norm()
        hid = int(self.d_model * float(mlp_ratio) * (2.0 / 3.0))
        self.ffn = GeGLU(
            self.d_model, hid, out_dim=self.d_model, dropout=dropout
        )
        self._flex_bias_fail_count: int = 0
        self._flex_bias_fail_max: int = max(0, int(env_int("ENN_RESAMPLER_FLEX_RETRY", 3)))
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
                "Resampler expects latents (B,Lq,D) and tokens (B,Lk,D), "
                f"got {tuple(latents.shape)} and {tuple(tokens.shape)}"
            )
        B, Lq, D = latents.shape
        _, Lk, Dk = tokens.shape
        if (latents.size(-1) != self.d_model) or (
            tokens.size(-1) != self.d_model
        ):
            raise ValueError(
                f"Resampler expects last dim D={self.d_model}, got latents D={D} tokens D={Dk}"
            )
        q_in = self.norm_q(latents)
        kv_in = self.norm_kv(tokens)
        H = int(self.nhead)
        Dh = int(self.head_dim)
        q = self.q_proj(q_in).view(B, Lq, H, Dh).transpose(1, 2)
        k = self.k_proj(kv_in).view(B, Lk, H, Dh).transpose(1, 2)
        v = self.v_proj(kv_in).view(B, Lk, H, Dh).transpose(1, 2)
        attn_out = None
        exporting = bool(is_export_or_trace())
        compiling = bool(is_compiling())
        plan = ATTENTION_POLICY.plan(
            q=q,
            need_weights=False,
            has_bias=attn_bias is not None,
            exporting=exporting,
            compiling=compiling,
        )
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
            if ("shape" in m and "mismatch" in m) or ("attn_bias" in m and "mismatch" in m):
                return "struct"
            if exporting or compiling:
                return "struct"
            if "no valid triton configs" in m or "outofresources" in m or "out of resources" in m:
                return "transient"
            if "torch._dynamo" in m or "torch._inductor" in m or "compileerror" in m:
                return "transient"
            return "transient"

        if (
            (not self._disable_flex_bias_runtime)
            and (not force_no_flex)
            and attn_bias is not None
            and plan.backend == AttentionBackend.FLEX
            and plan.use_score_mod_for_bias
        ):
            try:
                if not _HAS_FLEX_ATTENTION:
                    raise RuntimeError("FlexAttention not available")
                _ensure_flex_kwargs_initialized()
                if "score_mod" not in _FLEX_KWARGS:
                    raise RuntimeError("FlexAttention score_mod not supported")
                bias_bk = _coerce_attn_bias_to_bk(attn_bias, B=int(B), K=int(Lk), like=q)
                sm = self._flex_keybias_score_mod
                if sm is None:
                    sm = _FlexKeyBiasScoreMod(bias_bk)
                    self._flex_keybias_score_mod = sm
                else:
                    sm.set_bias(bias_bk)
                attn_out = get_flex_kernel()(
                    q,
                    k,
                    v,
                    score_mod=sm,
                    block_mask=None,
                    scale=None,
                    dropout_p=(self.dropout_p if self.training else 0.0),
                    training=bool(self.training),
                    is_causal=False,
                )
            except Exception as exc:
                kind = _classify_flex_failure(exc)
                self._flex_bias_fail_count += 1
                if kind == "struct":
                    self._disable_flex_bias_runtime = True
                elif self._flex_bias_fail_max > 0 and self._flex_bias_fail_count >= self._flex_bias_fail_max:
                    self._disable_flex_bias_runtime = True

        if attn_out is None and attn_bias is not None:
            fp32_math = bool(env_bool("ENN_RESAMPLER_FP32_BIAS_MATH_FALLBACK", True))
            if q.dtype == torch.float64 or (q.dtype == torch.float32 and fp32_math and self._disable_flex_bias_runtime):
                with contextlib.suppress(Exception):
                    bias_bk = _coerce_attn_bias_to_bk(attn_bias, B=int(B), K=int(Lk), like=q)
                    if int(bias_bk.shape[0]) == 1:
                        bias_bk = bias_bk.expand(int(B), int(Lk))
                    bias4 = bias_bk.view(int(B), 1, 1, int(Lk)).expand(int(B), int(H), int(Lq), int(Lk)).contiguous()
                    attn_out = _attention_math_bshd(
                        q,
                        k,
                        v,
                        attn_mask=bias4,
                        is_causal=False,
                        dropout_p=(self.dropout_p if self.training else 0.0),
                        training=bool(self.training),
                    )

        if attn_out is None:
            attn_out = self.attn(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=(self.dropout_p if self.training else 0.0),
                is_causal=False,
            )
        if attn_out.dim() == 4:
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, Lq, D)
        else:
            raise RuntimeError(
                f"Resampler attention returned unexpected shape {tuple(attn_out.shape)}"
            )
        latents = latents + self.drop_path(
            self.dropout(self.out_proj(attn_out))
        )
        latents = latents + self.drop_path(
            self.dropout(self.ffn(self.norm_ffn(latents)))
        )
        return latents


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
        self._stats_cache_lock = Mutex()
        self._stats_cache_max = 8
        self._x_stats_cache: Dict[
            Tuple[str, int, torch.dtype], Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._y_stats_cache: Dict[
            Tuple[str, int, torch.dtype], Tuple[torch.Tensor, torch.Tensor]
        ] = {}

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
        self: Self, fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> "Scaler":
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
        if self._scaler_guard_enabled() and getattr(t.device, "type", None) == "cuda":
            if self._guard_is_collapse(out2, std_min=float(std_min)):
                md2 = torch.float32 if force_fp32_on_cuda else master_dtype
                mean2 = mean_buf.to(device=t.device, dtype=md2)
                std2 = std_buf.to(device=t.device, dtype=md2)
                t32 = (t.reshape(1, -1) if t.dim() == 1 else t.reshape(-1, feature_dim)).to(dtype=md2)
                eps_use2 = float(max(eps_use, 1e-4 if t.dtype == torch.bfloat16 else 1e-5))
                denom2 = (std2 + eps_use2).clamp_min(eps_use2)
                out2 = (t32 - mean2) / denom2
                if env_bool("ENN_SCALER_GUARD_LOG", True):
                    warnings.warn(
                        "Scaler.normalize_x: detected collapse/nonfinite; retried this batch with safer fp32 math.",
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
                    return self._piecewise(z_raw)
                return z_raw
            case "affine":
                return self.affine(z_raw)
            case "none":
                return z_raw
            case _:
                return z_raw

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
            torch.full((1,), float("inf"), dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "sampled_x_max",
            torch.full((1,), float("-inf"), dtype=torch.float64),
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
            torch.full((1,), float("inf"), dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "sampled_y_max",
            torch.full((1,), float("-inf"), dtype=torch.float64),
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
            torch.full((1,), float("inf"), dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "reduced_x_max",
            torch.full((1,), float("-inf"), dtype=torch.float64),
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

    def _apply(self: Self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Self:
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
        self: Self, fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> "Recorder":
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
