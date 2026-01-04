# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import math
import threading
from collections import OrderedDict
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..core.casting import env_bool, env_int, env_str
from ..core.compat import StochasticDepth
from ..core.profiler import FLOP_PROFILER
from ..core.system import empty_device_cache
from ..core.graph import torch_compiler_disable
from .kernels import DotProductAttention, MultiHeadAttention, MultiScaleRetention

_Norm = nn.LayerNorm

try:
    from torch.nn.attention.flex_attention import (create_block_mask,
                                                   flex_attention)
    _HAS_FLEX_ATTENTION = True
except Exception:
    create_block_mask = None
    flex_attention = None
    _HAS_FLEX_ATTENTION = False


if env_bool("STNET_DISABLE_FLEX_ATTENTION", False):
    _HAS_FLEX_ATTENTION = False

_FLEX_ATTENTION_KWARGS: set[str] = set()
if _HAS_FLEX_ATTENTION and flex_attention is not None:
    with contextlib.suppress(Exception):
        import inspect
        _FLEX_ATTENTION_KWARGS = set(inspect.signature(flex_attention).parameters.keys())

_FLEX_BLOCK_MASK_CACHE_MAX = 16
_FLEX_BLOCK_MASK_CACHE_EST_MAX_BYTES = env_int(
    "STNET_FLEX_BLOCK_MASK_CACHE_EST_MAX_BYTES", 128 * 1024 * 1024
)

_DILATED_MASK_CACHE_MAX = 32
_DILATED_MASK_CACHE_MAX_L = env_int("STNET_DILATED_MASK_CACHE_MAX_L", 4096)
_DILATED_MASK_CACHE_ENTRY_MAX_BYTES = env_int(
    "STNET_DILATED_MASK_CACHE_ENTRY_MAX_BYTES", 64 * 1024 * 1024
)


def _device_key(device: torch.device) -> Tuple[str, int]:
    idx = -1
    with contextlib.suppress(Exception):
        if device.index is not None:
            idx = int(device.index)
    return (str(device.type), idx)


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
    return allowed.contiguous()


def _stable_softmax(scores: torch.Tensor) -> torch.Tensor:
    maxv = scores.max(dim=-1, keepdim=True).values
    maxv = torch.where(torch.isfinite(maxv), maxv, torch.zeros_like(maxv))
    exp = torch.exp(scores - maxv)
    denom = exp.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return exp / denom


def _tensor_stats(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if t.is_floating_point() or t.is_complex():
        v, m = torch.var_mean(t, correction=0)
        mn, mx = torch.aminmax(t)
        return v, m, mn, mx
    mn, mx = torch.aminmax(t)
    m = t.sum(dtype=torch.float64) / float(t.numel())
    v = torch.zeros((), dtype=torch.float64, device=m.device)
    return v, m, mn, mx


def _stream_flex_attention(
    self: Any,
    qh: torch.Tensor,
    kh: torch.Tensor,
    vh: torch.Tensor,
    *args: Any,
    group_size: int,
    block_size: int,
    win: int | None,
    scale: float,
    dropout_p: float,
    kpm_k: Optional[torch.Tensor],
    q_pad: Optional[torch.Tensor],
) -> torch.Tensor:
    if flex_attention is None:
        raise RuntimeError("flex_attention was not imported")
    B, H, L_q, _ = qh.shape
    L_k = int(kh.shape[2])
    embed_dim = int(self.embed_dim)
    out_full: Optional[torch.Tensor] = None
    for b0 in range(0, B, group_size):
        b1 = min(B, b0 + group_size)
        B_g = b1 - b0
        qh_g, kh_g, vh_g = qh[b0:b1], kh[b0:b1], vh[b0:b1]
        kpm_g = kpm_k[b0:b1] if (kpm_k is not None and kpm_k.dim() == 2) else None
        if kpm_g is None:
            block_mask_g = self._get_flex_block_mask(
                B_g, H, L_q, L_k, device=qh.device, block_size=block_size, win=win
            )
        else:
            mask_mod = _FlexMaskMod(L_q, L_k, self.causal, win, self.dilation, kpm_g)
            block_mask_g = create_block_mask(
                mask_mod, B_g, H, L_q, L_k, device=qh.device, BLOCK_SIZE=block_size
            )

        if _FLEX_ATTENTION_KWARGS:
            flex_kwargs: dict[str, Any] = {"block_mask": block_mask_g}
            if "scale" in _FLEX_ATTENTION_KWARGS:
                flex_kwargs["scale"] = scale
            if dropout_p > 0.0:
                for name in ("dropout_p", "dropout"):
                    if name in _FLEX_ATTENTION_KWARGS:
                        flex_kwargs[name] = dropout_p
                        break
            y_g = flex_attention(qh_g, kh_g, vh_g, **flex_kwargs)
        else:
            try:
                y_g = flex_attention(
                    qh_g, kh_g, vh_g, block_mask=block_mask_g, scale=scale, dropout_p=dropout_p
                )
            except TypeError:
                try:
                    y_g = flex_attention(qh_g, kh_g, vh_g, block_mask=block_mask_g, scale=scale)
                except TypeError:
                    y_g = flex_attention(qh_g, kh_g, vh_g, block_mask=block_mask_g)

        out_g = self.out_proj(y_g.transpose(1, 2).contiguous().view(B_g, L_q, embed_dim))
        if q_pad is not None:
            out_g = out_g.masked_fill(q_pad[b0:b1].unsqueeze(-1), 0.0)
        if out_full is None:
            if B_g == B:
                return out_g
            out_full = out_g.new_empty((B, *out_g.shape[1:]))
        out_full[b0:b1] = out_g
    if out_full is None:
        raise RuntimeError("Internal error: flex_attention produced no outputs")
    return out_full


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
    buf_names = ("x_mean", "x_std", "y_mean", "y_std", "y_min", "y_max", "y_q_low", "y_q_high", "affine_a", "affine_b", "pw_x", "pw_y")
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


def checkpoint_mode() -> str:
    raw = str(env_str("STNET_CHECKPOINT_MODE") or env_str("STNET_CHECKPOINT") or "ffn").strip().lower()
    match raw:
        case "0" | "false" | "none" | "off" | "disable" | "disabled":
            return "none"
        case "attn" | "attention":
            return "attn"
        case "all" | "full":
            return "all"
        case _:
            return "ffn"


def norm_layer(norm_type: str, dim: int) -> nn.Module:
    norm = str(norm_type).strip().lower()
    match norm:
        case "ln" | "layernorm" | "layer_norm" | "layer-norm":
            return nn.LayerNorm(dim)
        case "bn" | "batchnorm" | "batch_norm" | "batch-norm":
            return nn.BatchNorm1d(dim)
        case "rms" | "rmsnorm" | "rms_norm" | "rms-norm":
            try:
                from torch.nn import RMSNorm

                return RMSNorm(dim)
            except Exception:
                return nn.LayerNorm(dim)
        case _:
            return nn.LayerNorm(dim)


class _FlexMaskMod:
    def __init__(
        self, L_q: int, L_k: int, causal: bool, win: int | None, dilation: int, kpm: torch.Tensor
    ) -> None:
        self._kv_limit = L_q if L_k > L_q else None
        self.causal = causal
        self.win = win
        self.dilation = dilation
        self.kpm = kpm.to(torch.bool)

    def __call__(self, b, h, q_idx, kv_idx):
        dq = q_idx - kv_idx
        keep = torch.ones_like(dq, dtype=torch.bool)
        if self._kv_limit is not None:
            keep &= kv_idx < self._kv_limit
        if self.causal:
            keep &= kv_idx <= q_idx
        if self.win is not None:
            keep &= dq.abs() <= self.win
        if self.dilation > 1:
            keep &= (dq % self.dilation) == 0
        return keep & ~self.kpm[b, kv_idx]
    

class _FlexDilatedMaskMod:
    __slots__ = ("L_q", "L_k", "dilation", "win", "causal")

    def __init__(
        self,
        *args: Any,
        L_q: int,
        L_k: int,
        dilation: int,
        win: Optional[int],
        causal: bool,
    ) -> None:
        self.L_q = int(L_q)
        self.L_k = int(L_k)
        self.dilation = max(1, int(dilation))
        self.win = None if win is None else int(win)
        self.causal = bool(causal)

    def __call__(self, b: int, h: int, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        _ = (b, h)
        dq = q_idx - kv_idx
        keep = torch.ones_like(dq, dtype=torch.bool)
        try:
            if int(self.L_k) > int(self.L_q):
                keep &= (kv_idx < int(self.L_q))
        except Exception:
            pass
        if self.causal:
            keep &= (kv_idx <= q_idx)
        if self.win is not None:
            keep &= (dq.abs() <= int(self.win))
        if self.dilation > 1:
            keep &= ((dq % int(self.dilation)) == 0)
        return keep


class Retention(nn.Module):
    def __init__(self, d_model: int, nhead: int, *args: Any, mode: str | None = None, **kwargs: Any) -> None:
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

    def _forward_bidirectional(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        restore_dtype: Optional[torch.dtype] = None
        x_in = x
        if getattr(x_in.device, "type", "cpu") == "mps" and x_in.dtype == torch.bfloat16:
            restore_dtype = x_in.dtype
            x_in = x_in.to(torch.float16)
        if x_in.dim() != 3:
            raise ValueError(f"Retention(spatial) expects (B,L,D), got {tuple(x_in.shape)}")
        B, L, D = map(int, x_in.shape)
        if L <= 0:
            out0 = x_in.new_zeros(x_in.shape)
            return out0.to(restore_dtype) if restore_dtype is not None else out0
        msr = self.msr
        H = int(msr.nhead)
        Dh = int(msr.head_dim)
        q = msr.q_proj(x_in).view(B, L, H, Dh)
        v = msr.v_proj(x_in).view(B, L, H, Dh)
        v = msr._apply_kpm_to_v(v, attn_mask)  # type: ignore[attr-defined]
        lam_h = msr._decay_lambda(v.device, v.dtype).to(dtype=v.dtype, device=v.device)
        state_fwd = msr._scan_causal(v, lam_h)  # type: ignore[attr-defined]
        state_bwd = msr._scan_causal(v.flip(1), lam_h).flip(1)  # type: ignore[attr-defined]
        calc_dtype = torch.float32 if v.dtype in (torch.float16, torch.bfloat16) else v.dtype
        bi_state = (
            state_fwd.to(calc_dtype)
            + state_bwd.to(calc_dtype)
            - v.to(calc_dtype)
        ).to(dtype=v.dtype)
        y = (q * bi_state).contiguous().view(B, L, int(msr.d_model))
        y = msr.norm(y)
        if bool(getattr(msr, "use_gate", False)) and getattr(msr, "g_proj", None) is not None:
            gate = F.silu(msr.g_proj(x_in))
            y = y * gate
        out = msr.o_proj(y)
        if restore_dtype is not None:
            out = out.to(restore_dtype)
        return out

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        state: Any = None,
        mode: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Any]:
        _ = kwargs
        eff_mode = self._coerce_mode(mode if mode is not None else getattr(self, "mode", None))
        if eff_mode != "spatial":
            h = self.msr(x, attn_mask=attn_mask, state=state, return_state=True)
            if isinstance(h, tuple):
                out, new_state = h
            else:
                out, new_state = h, state
            if isinstance(new_state, torch.Tensor) and (not torch.is_grad_enabled()):
                new_state = new_state.detach()
            return out, new_state
        out = self._forward_bidirectional(x, attn_mask=attn_mask)
        return out, None

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
        self._dot_attn = DotProductAttention(num_heads=self.nhead, head_dim=self.head_dim)
        self._dot_attn_mask_kw: str | None = "attn_mask"
        self._dot_attn_dropout_kw: str | None = None
        self._dot_attn_training_kw: str | None = "training"
        self._dot_attn_causal_kw: str | None = None
        with contextlib.suppress(Exception):
            import inspect
            params = inspect.signature(self._dot_attn.forward).parameters
            if "attn_mask" in params:
                self._dot_attn_mask_kw = "attn_mask"
            elif "mask" in params:
                self._dot_attn_mask_kw = "mask"
            elif "attention_mask" in params:
                self._dot_attn_mask_kw = "attention_mask"
            else:
                self._dot_attn_mask_kw = None
            if "training" in params:
                self._dot_attn_training_kw = "training"
            else:
                self._dot_attn_training_kw = None
            if "dropout_p" in params:
                self._dot_attn_dropout_kw = "dropout_p"
            elif "dropout" in params:
                self._dot_attn_dropout_kw = "dropout"
            else:
                self._dot_attn_dropout_kw = None
            if "is_causal" in params:
                self._dot_attn_causal_kw = "is_causal"
            elif "causal" in params:
                self._dot_attn_causal_kw = "causal"
            else:
                self._dot_attn_causal_kw = None
        self._mask_cache_lock = threading.Lock()
        self._flex_block_mask_cache_lock = threading.Lock()
        self._mask_cache = OrderedDict()
        self._flex_block_mask_cache = OrderedDict()
        self._mask_cache_last: tuple[Any, Any] | None = None
        self._flex_block_mask_cache_last: tuple[Any, Any] | None = None

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_mask_cache_lock", None)
        state.pop("_flex_block_mask_cache_lock", None)
        state.pop("_mask_cache", None)
        state.pop("_flex_block_mask_cache", None)
        state.pop("_mask_cache_last", None)
        state.pop("_flex_block_mask_cache_last", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._mask_cache_lock = threading.Lock()
        self._flex_block_mask_cache_lock = threading.Lock()
        self._mask_cache = OrderedDict()
        self._flex_block_mask_cache = OrderedDict()
        self._mask_cache_last = None
        self._flex_block_mask_cache_last = None

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
            _device_key(device),
        )
        last = getattr(self, "_mask_cache_last", None)
        if last is not None and last[0] == key:
            return last[1]
        cache = getattr(self, "_mask_cache", None)
        if cache is None:
            cache = OrderedDict()
            setattr(self, "_mask_cache", cache)
        lock = getattr(self, "_mask_cache_lock", None)
        if lock is None:
            lock = threading.Lock()
            setattr(self, "_mask_cache_lock", lock)
        with lock:
            cached = cache.get(key)
            if cached is not None:
                with contextlib.suppress(Exception):
                    cache.move_to_end(key)
                self._mask_cache_last = (key, cached)
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
        with lock:
            cached = cache.get(key)
            if cached is not None:
                with contextlib.suppress(Exception):
                    cache.move_to_end(key)
                self._mask_cache_last = (key, cached)
                return cached
            cache[key] = mask
            self._mask_cache_last = (key, mask)
            with contextlib.suppress(Exception):
                cache.move_to_end(key)
            try:
                while len(cache) > int(_DILATED_MASK_CACHE_MAX):
                    cache.popitem(last=False)
            except Exception:
                pass
        return mask

    def _get_flex_block_mask(
        self,
        B: int,
        H: int,
        L_q: int,
        L_k: int,
        *args: Any,
        device: torch.device,
        block_size: int,
        win: Optional[int],
    ) -> Any:
        win_key = int(win) if win is not None else -1
        key = (
            _device_key(device),
            int(B),
            int(H),
            int(L_q),
            int(L_k),
            int(block_size),
            int(self.dilation),
            win_key,
            int(self.causal),
        )
        last = getattr(self, "_flex_block_mask_cache_last", None)
        if last is not None and last[0] == key:
            return last[1]
        cache = getattr(self, "_flex_block_mask_cache", None)
        if cache is None:
            cache = OrderedDict()
            setattr(self, "_flex_block_mask_cache", cache)
        lock = getattr(self, "_flex_block_mask_cache_lock", None)
        if lock is None:
            lock = threading.Lock()
            setattr(self, "_flex_block_mask_cache_lock", lock)
        with lock:
            cached = cache.get(key)
            if cached is not None:
                with contextlib.suppress(Exception):
                    cache.move_to_end(key)
                self._flex_block_mask_cache_last = (key, cached)
                return cached
        try:
            est_bool_bytes = int(B) * int(H) * int(L_q) * int(L_k)
        except Exception:
            est_bool_bytes = _FLEX_BLOCK_MASK_CACHE_EST_MAX_BYTES + 1
        skip_cache = est_bool_bytes > _FLEX_BLOCK_MASK_CACHE_EST_MAX_BYTES
        if create_block_mask is None:
            raise RuntimeError("create_block_mask was not imported")
        mask_mod = _FlexDilatedMaskMod(
            L_q=int(L_q),
            L_k=int(L_k),
            dilation=int(self.dilation),
            win=win,
            causal=bool(self.causal),
        )
        block_mask = create_block_mask(
            mask_mod,
            B,
            H,
            L_q,
            L_k,
            device=device,
            BLOCK_SIZE=int(block_size),
        )
        if skip_cache:
            return block_mask
        with lock:
            cached = cache.get(key)
            if cached is not None:
                with contextlib.suppress(Exception):
                    cache.move_to_end(key)
                self._flex_block_mask_cache_last = (key, cached)
                return cached
            cache[key] = block_mask
            self._flex_block_mask_cache_last = (key, block_mask)
            with contextlib.suppress(Exception):
                cache.move_to_end(key)
            try:
                while len(cache) > int(_FLEX_BLOCK_MASK_CACHE_MAX):
                    cache.popitem(last=False)
            except Exception:
                pass
        return block_mask

    def _call_dot_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args: Any,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float,
        is_causal: bool,
    ) -> torch.Tensor:
        attn = getattr(self, "_dot_attn", None)
        if attn is None:
            attn = DotProductAttention(num_heads=self.nhead, head_dim=self.head_dim)
            self._dot_attn = attn
        kwargs: dict[str, Any] = {}
        mask_kw = getattr(self, "_dot_attn_mask_kw", "attn_mask")
        if attn_mask is not None:
            if mask_kw is not None:
                kwargs[str(mask_kw)] = attn_mask
            else:
                kwargs["attn_mask"] = attn_mask
        train_kw = getattr(self, "_dot_attn_training_kw", None)
        if train_kw is not None:
            kwargs[str(train_kw)] = bool(self.training)
        drop_kw = getattr(self, "_dot_attn_dropout_kw", None)
        if drop_kw is not None:
            kwargs[str(drop_kw)] = float(dropout_p)
        causal_kw = getattr(self, "_dot_attn_causal_kw", None)
        if causal_kw is not None:
            kwargs[str(causal_kw)] = bool(is_causal)
        return attn(q, k, v, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        transposed = False
        if not self.batch_first:
            if x.dim() != 3:
                raise ValueError(
                    f"DilatedAttention expects a 3D tensor, got shape {tuple(x.shape)}"
                )
            L0, B0, _ = x.shape
            x = x.transpose(0, 1)
            if key_padding_mask is not None:
                if key_padding_mask.dim() != 2:
                    raise ValueError(
                        f"key_padding_mask must be 2D, got rank {key_padding_mask.dim()}"
                    )
                if key_padding_mask.shape == (B0, L0):
                    pass
                elif key_padding_mask.shape == (L0, B0):
                    key_padding_mask = key_padding_mask.transpose(0, 1)
                else:
                    raise ValueError(
                        "key_padding_mask shape mismatch for batch_first=False: "
                        f"expected (B,L)=({B0},{L0}) or (L,B)=({L0},{B0}), got {tuple(key_padding_mask.shape)}"
                    )
            transposed = True
        if x.dim() != 3:
            raise ValueError(
                f"DilatedAttention expects a 3D tensor (B,L,D), got shape {tuple(x.shape)}"
            )
        B, L, D = x.shape
        if D != self.embed_dim:
            raise ValueError(f"x.shape[-1]={D} must match embed_dim={self.embed_dim}")
        want_weights = bool(need_weights)
        avg_weights = bool(average_attn_weights) if want_weights else False
        kpm: Optional[torch.Tensor] = None
        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, L):
                raise ValueError(
                    f"key_padding_mask must be (B, L)=({B},{L}), got {tuple(key_padding_mask.shape)}"
                )
            kpm = key_padding_mask
            if kpm.dtype is not torch.bool:
                kpm = kpm.to(torch.bool)
            if kpm.device != x.device:
                with contextlib.suppress(Exception):
                    kpm = kpm.to(device=x.device, non_blocking=True)
            with contextlib.suppress(Exception):
                kpm = kpm.contiguous()
        q_pad: Optional[torch.Tensor] = None
        if kpm is not None:
            q_pad = kpm
            with contextlib.suppress(Exception):
                q_pad = q_pad.contiguous()
        use_flex = bool(_HAS_FLEX_ATTENTION and x.is_cuda and (not want_weights))
        if want_weights:
            L_k = int(L)
            pad_len = 0
        else:
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
            x_pad = x.new_zeros((B, L_k, D))
            x_pad[:, :L, :].copy_(x)
            x_k = x_pad
            if kpm is not None:
                kpm_b = kpm.to(torch.bool)
                kpm_pad = torch.ones((B, L_k), device=kpm_b.device, dtype=torch.bool)
                kpm_pad[:, :L].copy_(kpm_b)
                kpm_k = kpm_pad
            elif (not bool(self.causal)) and (not use_flex):
                kpm_pad_1d = torch.zeros((L_k,), device=x.device, dtype=torch.bool)
                kpm_pad_1d[L:] = True
                kpm_k = kpm_pad_1d
            else:
                kpm_k = None
        x_k = self.norm1(x_k)
        qkv = self.qkv(x_k)
        q, k, v = qkv.chunk(3, dim=-1)
        H = self.num_heads
        Dh = self.head_dim
        L_q = int(L)
        qh = q[:, :L_q, :].reshape(B, L_q, H, Dh).transpose(1, 2)
        kh = k.reshape(B, L_k, H, Dh).transpose(1, 2)
        vh = v.reshape(B, L_k, H, Dh).transpose(1, 2)
        training = bool(self.training)
        dropout_p = float(self.dropout_p) if training else 0.0
        attn_w: Optional[torch.Tensor] = None

        if want_weights:
            is_simple = (int(self.dilation) == 1) and (self.window_size is None)
            base_mask_keep: Optional[torch.Tensor] = None
            is_causal = False
            if is_simple and bool(self.causal) and (kpm_k is None):
                is_causal = True
            elif not (is_simple and (not bool(self.causal))):
                base_mask_full = self._get_mask(L_k, x_k.device)
                base_mask_keep = base_mask_full[:L_q, :]
            key_mask: Optional[torch.Tensor] = None
            if kpm_k is not None:
                kpm_b = kpm_k.to(torch.bool)
                if kpm_b.dim() == 1:
                    key_mask = kpm_b[None, None, None, :]
                else:
                    key_mask = kpm_b[:, None, None, :]
            base_mask_out: Optional[torch.Tensor] = None
            if base_mask_keep is not None:
                base_mask_out = (~base_mask_keep).to(torch.bool)[None, None, :, :]
            causal_mask: Optional[torch.Tensor] = None
            if is_causal:
                causal_mask = torch.ones((L_q, L_k), device=qh.device, dtype=torch.bool).triu(diagonal=1)
                causal_mask = causal_mask[None, None, :, :]
            env_mb = int(env_int("STNET_ATTN_WEIGHTS_BATCH_MICROBATCH", 0))
            est = int(B) * int(H) * int(L_q) * int(L_k)
            if env_mb > 0:
                group = max(1, min(int(B), int(env_mb)))
            else:
                if est >= 64 * 1024 * 1024:
                    group = 1
                elif est >= 32 * 1024 * 1024:
                    group = 2
                elif est >= 16 * 1024 * 1024:
                    group = 4
                elif est >= 8 * 1024 * 1024:
                    group = 8
                else:
                    group = int(B)
                group = max(1, min(int(B), int(group)))
            out_full = qh.new_empty((B, L_q, self.embed_dim))
            if avg_weights:
                attn_w_full = qh.new_empty((B, L_q, L_k))
            else:
                attn_w_full = qh.new_empty((B, H, L_q, L_k))
            last_oom: Optional[RuntimeError] = None
            while group >= 1:
                try:
                    for b0 in range(0, B, group):
                        b1 = min(B, b0 + group)
                        qg = qh[b0:b1]
                        kg = kh[b0:b1]
                        vg = vh[b0:b1]
                        scores = torch.matmul(qg, kg.transpose(-2, -1))
                        scores = scores * (1.0 / math.sqrt(float(Dh)))
                        scores = scores.to(torch.float32)
                        if causal_mask is not None:
                            scores.masked_fill_(causal_mask, float("-inf"))
                        if base_mask_out is not None:
                            scores.masked_fill_(base_mask_out, float("-inf"))
                        if key_mask is not None:
                            km = key_mask if key_mask.shape[0] == 1 else key_mask[b0:b1]
                            scores.masked_fill_(km, float("-inf"))
                        probs = _stable_softmax(scores)
                        if dropout_p > 0.0:
                            probs = F.dropout(probs, p=dropout_p, training=True)
                        if avg_weights:
                            attn_w_full[b0:b1] = probs.mean(dim=1).to(dtype=qh.dtype)
                        else:
                            attn_w_full[b0:b1] = probs.to(dtype=qh.dtype)
                        probs_out = probs.to(dtype=vg.dtype)
                        yg = torch.matmul(probs_out, vg)
                        attn_out_g = self.out_proj(
                            yg.transpose(1, 2).contiguous().view((b1 - b0), L_q, self.embed_dim)
                        )
                        if q_pad is not None:
                            attn_out_g = attn_out_g.masked_fill(q_pad[b0:b1].unsqueeze(-1), 0.0)
                        out_full[b0:b1] = attn_out_g
                    last_oom = None
                    break
                except RuntimeError as e:
                    msg = str(e)
                    if "CUDA out of memory" not in msg and "out of memory" not in msg:
                        raise
                    last_oom = e
                    if x_k.device.type == "cuda":
                        with contextlib.suppress(Exception):
                            empty_device_cache(device=x_k.device, do_gc=False, min_interval_s=0.0)
                    group //= 2
            if last_oom is not None:
                raise last_oom
            attn_out = out_full
            attn_w = attn_w_full
        elif use_flex:
            win = int(self.window_size) if self.window_size is not None else None
            if L_k <= 2048:
                _block_size = 128
            elif L_k <= 16384:
                _block_size = 256
            else:
                _block_size = 512
            scale = 1.0 / math.sqrt(float(Dh))
            max_group = int(getattr(self, "flex_batch_microbatch", 0) or B)
            max_group = max(1, min(B, max_group))

            group = max_group
            last_oom: Optional[RuntimeError] = None
            while group >= 1:
                try:
                    attn_out = _stream_flex_attention(self, qh, kh, vh, group_size=group, block_size=_block_size, win=win, scale=scale, dropout_p=dropout_p, kpm_k=kpm_k, q_pad=q_pad)
                    last_oom = None
                    break
                except RuntimeError as e:
                    msg = str(e)
                    if "CUDA out of memory" not in msg and "out of memory" not in msg:
                        raise
                    last_oom = e
                    if x_k.device.type == "cuda":
                        with contextlib.suppress(Exception):
                            empty_device_cache(device=x_k.device, do_gc=False, min_interval_s=0.0)
                    group //= 2
            if last_oom is not None:
                raise last_oom
        else:
            qkv = self.qkv(x_k)
            q, k, v = qkv.chunk(3, dim=-1)
            H = self.num_heads
            Dh = self.head_dim
            qh = q[:, :L_q, :].reshape(B, L_q, H, Dh).transpose(1, 2)
            kh = k.reshape(B, L_k, H, Dh).transpose(1, 2)
            vh = v.reshape(B, L_k, H, Dh).transpose(1, 2)
            training = bool(self.training)
            dropout_p = float(self.dropout_p) if training else 0.0
            is_simple = (int(self.dilation) == 1) and (self.window_size is None)
            attn_out: Optional[torch.Tensor] = None
            if (
                is_simple
                and bool(self.causal)
                and (kpm_k is not None)
                and isinstance(kpm_k, torch.Tensor)
            ):
                kpm_check: Optional[torch.Tensor] = None
                if isinstance(key_padding_mask, torch.Tensor) and key_padding_mask.device.type == "cpu":
                    kpm_check = key_padding_mask
                elif isinstance(kpm_k, torch.Tensor) and kpm_k.device.type == "cpu":
                    kpm_check = kpm_k
                if kpm_check is not None:
                    kpm_b = kpm_check.to(torch.bool)
                    right_padded = True
                    if int(kpm_b.shape[1]) >= 2:
                        right_padded = not (kpm_b[:, :-1] & (~kpm_b[:, 1:])).any().item()
                    if right_padded:
                        y_full = self._call_dot_attn(
                            qh,
                            kh,
                            vh,
                            attn_mask=None,
                            dropout_p=dropout_p,
                            is_causal=True,
                        )
                        attn_out = self.out_proj(
                            y_full.transpose(1, 2).contiguous().view(B, L_q, self.embed_dim)
                        )
                        if q_pad is not None:
                            attn_out = attn_out.masked_fill(q_pad.unsqueeze(-1), 0.0)
            if attn_out is None:
                base_mask_keep: Optional[torch.Tensor] = None
                is_causal = False
                if is_simple and bool(self.causal) and (kpm_k is None):
                    is_causal = True
                elif not (is_simple and (not bool(self.causal))):
                    base_mask_full = self._get_mask(L_k, x_k.device)
                    base_mask_keep = base_mask_full[:L_q, :]
                    is_causal = False
                key_mask: Optional[torch.Tensor] = None
                if kpm_k is not None:
                    kpm_b = kpm_k.to(torch.bool)
                    if kpm_b.dim() == 1:
                        key_mask = kpm_b[None, None, None, :]
                    else:
                        key_mask = kpm_b[:, None, None, :]
                base_mask: Optional[torch.Tensor] = None
                if base_mask_keep is not None:
                    base_mask = (~base_mask_keep).to(torch.bool)
                if base_mask is None:
                    y = self._call_dot_attn(
                        qh,
                        kh,
                        vh,
                        attn_mask=key_mask,
                        dropout_p=dropout_p,
                        is_causal=bool(is_causal),
                    )
                else:
                    if key_mask is None:
                        y = self._call_dot_attn(
                            qh,
                            kh,
                            vh,
                            attn_mask=base_mask,
                            dropout_p=dropout_p,
                            is_causal=False,
                        )
                    elif int(key_mask.shape[0]) == 1:
                        attn_mask = base_mask[None, None, :, :] | key_mask
                        y = self._call_dot_attn(
                            qh,
                            kh,
                            vh,
                            attn_mask=attn_mask,
                            dropout_p=dropout_p,
                            is_causal=False,
                        )
                    else:
                        env_mb = int(env_int("STNET_SDPA_BATCH_MICROBATCH", 0))
                        group = int(env_mb)
                        if group <= 0:
                            est = int(B) * int(L_q) * int(L_k)
                            if est >= 64 * 1024 * 1024:
                                group = 1
                            elif est >= 16 * 1024 * 1024:
                                group = 2
                            elif est >= 4 * 1024 * 1024:
                                group = 4
                            else:
                                group = int(B)
                        group = max(1, min(int(B), int(group)))
                        base4 = base_mask[None, None, :, :]
                        out_full = qh.new_empty((B, H, L_q, Dh))
                        last_oom: Optional[RuntimeError] = None
                        while group >= 1:
                            try:
                                for b0 in range(0, B, group):
                                    b1 = min(B, b0 + group)
                                    attn_mask_g = base4 | key_mask[b0:b1]
                                    y_g = self._call_dot_attn(
                                        qh[b0:b1],
                                        kh[b0:b1],
                                        vh[b0:b1],
                                        attn_mask=attn_mask_g,
                                        dropout_p=dropout_p,
                                        is_causal=False,
                                    )
                                    out_full[b0:b1] = y_g
                                last_oom = None
                                break
                            except RuntimeError as e:
                                msg = str(e)
                                if "CUDA out of memory" not in msg and "out of memory" not in msg:
                                    raise
                                last_oom = e
                                if x_k.device.type == "cuda":
                                    with contextlib.suppress(Exception):
                                        empty_device_cache(device=x_k.device, do_gc=False, min_interval_s=0.0)
                                group //= 2
                        if last_oom is not None:
                            raise last_oom
                        y = out_full
                attn_out = self.out_proj(
                    y.transpose(1, 2).contiguous().view(B, L_q, self.embed_dim)
                )
                if q_pad is not None:
                    attn_out = attn_out.masked_fill(q_pad.unsqueeze(-1), 0.0)
        x_out = x + self.dropout(attn_out)
        res2 = x_out
        x_out = self.norm2(x_out)
        do_ckpt_ffn = (
            self.training
            and torch.is_grad_enabled()
            and _STNET_CHECKPOINT_MODE in {"ffn", "all"}
        )
        if do_ckpt_ffn:
            try:
                x_out = checkpoint(
                    self.ffn,
                    x_out,
                    use_reentrant=True,
                    preserve_rng_state=True,
                    determinism_check="none",
                )
            except TypeError:
                x_out = checkpoint(
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
        if want_weights:
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


class SigmoidGate(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 64,
        *args: Any,
        eps: float = 1e-6,
        clip_eps: float = 1e-6,
        p_floor: float = 0.0,
        p_ceil: float = 1.0,
        tile_size: Optional[int] = None,
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
        if self.tile_size > 0:
            tile_in = 3  # [base_rms, res_rms, res/base]
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
        self.register_buffer("_fb_count", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_fb_active_low_sum", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_fb_active_high_sum", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_fb_width_sum", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_fb_edge_low_sum", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_fb_edge_high_sum", torch.zeros((), dtype=torch.float32), persistent=False)
        self._fb_lock = threading.Lock()

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_fb_lock", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._fb_lock = threading.Lock()

    @torch.no_grad()
    def consume_fallback_tensor_stats(self) -> torch.Tensor:
        lock = getattr(self, "_fb_lock", None)
        if lock is None:
            lock = threading.Lock()
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

    @torch_compiler_disable(reason="SigmoidGate fallback stats update", recursive=False)
    def _fb_add_stats(
        self,
        count: float,
        width_sum: torch.Tensor,
        active_low_sum: torch.Tensor,
        active_high_sum: torch.Tensor,
        edge_low_sum: torch.Tensor,
        edge_high_sum: torch.Tensor,
    ) -> None:
        try:
            dyn = getattr(torch, "_dynamo", None)
            if dyn is not None:
                is_comp = getattr(dyn, "is_compiling", None)
                if callable(is_comp) and bool(is_comp()):
                    return
        except Exception:
            pass
        try:
            comp = getattr(torch, "compiler", None)
            is_exp = getattr(comp, "is_exporting", None)
            if callable(is_exp) and bool(is_exp()):
                return
        except Exception:
            pass
        try:
            if getattr(torch, "jit", None) is not None:
                if torch.jit.is_tracing() or torch.jit.is_scripting():
                    return
        except Exception:
            pass
        try:
            onnx = getattr(torch, "onnx", None)
            is_onnx = getattr(onnx, "is_in_onnx_export", None)
            if callable(is_onnx) and bool(is_onnx()):
                return
        except Exception:
            pass
        with torch.no_grad():
            lock = getattr(self, "_fb_lock", None)
            if lock is None:
                lock = threading.Lock()
                setattr(self, "_fb_lock", lock)
            with lock:
                self._fb_count.add_(float(count))
                self._fb_width_sum.add_(width_sum.to(dtype=torch.float32))
                self._fb_active_low_sum.add_(active_low_sum.to(dtype=torch.float32))
                self._fb_active_high_sum.add_(active_high_sum.to(dtype=torch.float32))
                self._fb_edge_low_sum.add_(edge_low_sum.to(dtype=torch.float32))
                self._fb_edge_high_sum.add_(edge_high_sum.to(dtype=torch.float32))

    def _expand_tiles(self, p_tile: torch.Tensor, dim: int) -> torch.Tensor:
        if self.tile_size <= 0:
            raise RuntimeError("SigmoidGate._expand_tiles called with tile_size<=0")
        b = int(p_tile.shape[0])
        tile = int(self.tile_size)
        n_tiles = int(p_tile.shape[1])
        d_pad = int(n_tiles * tile)
        p_full = p_tile.unsqueeze(-1).expand(b, n_tiles, tile).reshape(b, d_pad)
        return p_full[:, : int(dim)]

    def forward(
        self,
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
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            b_rms = torch.sqrt(torch.mean(b32 * b32, dim=1, keepdim=True) + self.eps)
            r_rms = torch.sqrt(torch.mean(r32 * r32, dim=1, keepdim=True) + self.eps)
            feats.append(b_rms.to(dtype=tokens.dtype))
            feats.append(r_rms.to(dtype=tokens.dtype))
        x = feats[0] if len(feats) == 1 else torch.cat(feats, dim=1)
        global_logit = self.net(x).squeeze(-1)  # (B,)
        use_tile = (
            self.tile_size > 0
            and self.tile_net is not None
            and base is not None
            and residue is not None
            and base.dim() == 2
            and residue.dim() == 2
            and int(base.shape[0]) == int(residue.shape[0])
        )
        if not use_tile:
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
                r_safe = torch.where(r32.abs() < self.eps, sign * self.eps, r32)
                p_a = (zmin - b32) / r_safe
                p_b = (zmax - b32) / r_safe
                p_min_dim = torch.minimum(p_a, p_b)
                p_max_dim = torch.maximum(p_a, p_b)
                p_low_bound = p_min_dim.max(dim=1).values
                p_high_bound = p_max_dim.min(dim=1).values
                p_low = torch.clamp(p_low_bound, min=float(self.p_floor), max=float(self.p_ceil))
                p_high = torch.clamp(p_high_bound, min=float(self.p_floor), max=float(self.p_ceil))
                p_high = torch.maximum(p_high, p_low + float(self.eps))
            p = p_low + (p_high - p_low) * sig.to(dtype=p_low.dtype)
            clip = float(max(self.clip_eps, self.eps))
            lo = float(self.p_floor) + clip
            hi = float(self.p_ceil) - clip
            if hi > lo:
                p = torch.clamp(p, min=lo, max=hi)
            else:
                p = torch.clamp(p, min=float(self.p_floor), max=float(self.p_ceil))
            edge_reg: Optional[torch.Tensor] = None
            edge_reg_low: Optional[torch.Tensor] = None
            edge_reg_high: Optional[torch.Tensor] = None
            if bool(fallback_bounds):
                try:
                    width = (p_high - p_low).to(dtype=torch.float32)
                    denom = max(float(self.p_ceil - self.p_floor), float(self.eps))
                    tthr = float(self.stat_width_frac) * denom
                    tthr = float(max(tthr, self.eps))
                    trim_low = (p_low - float(self.p_floor)).to(dtype=torch.float32)
                    trim_high = (float(self.p_ceil) - p_high).to(dtype=torch.float32)
                    active_low = trim_low >= tthr
                    active_high = trim_high >= tthr
                    w_safe = torch.maximum(width, width.new_full((), float(self.eps)))
                    ethr = float(max(float(self.stat_edge_frac), 0.0))
                    edge_thr = w_safe * float(ethr)
                    edge_low = (p - p_low) <= edge_thr
                    edge_high = (p_high - p) <= edge_thr
                    self._fb_add_stats(
                        float(p.shape[0]),
                        width.sum(),
                        active_low.to(dtype=torch.float32).sum(),
                        active_high.to(dtype=torch.float32).sum(),
                        edge_low.to(dtype=torch.float32).sum(),
                        edge_high.to(dtype=torch.float32).sum(),
                    )
                except Exception:
                    pass
            if bool(return_edge_reg) or bool(return_edge_reg_lr):
                try:
                    width = (p_high - p_low).to(dtype=torch.float32)
                    full = max(float(self.p_ceil - self.p_floor), float(self.eps))
                    min_w = float(edge_reg_min_width_frac) * full
                    min_w = float(max(min_w, self.eps))
                    mask = (width >= min_w).to(dtype=torch.float32)
                    w_safe = torch.maximum(width, width.new_full((), float(self.eps)))
                    q = (p.to(dtype=torch.float32) - p_low.to(dtype=torch.float32)) / w_safe
                    q = torch.clamp(q, 0.0, 1.0)
                    m = float(edge_reg_frac)
                    m = float(min(max(m, self.eps), 0.49))
                    inv_m = 1.0 / m
                    d_low = F.relu(m - q) * inv_m
                    d_high = F.relu(q - (1.0 - m)) * inv_m
                    pen_low = d_low.pow(float(edge_reg_power)) * mask
                    pen_high = d_high.pow(float(edge_reg_power)) * mask
                    denom = mask.sum() + float(self.eps)
                    edge_reg_low = pen_low.sum() / denom
                    edge_reg_high = pen_high.sum() / denom
                    edge_reg = edge_reg_low + edge_reg_high
                except Exception:
                    edge_reg = None
                    edge_reg_low = None
                    edge_reg_high = None
            out_dtype = residue.dtype if isinstance(residue, torch.Tensor) else tokens.dtype
            out = p.to(dtype=out_dtype).unsqueeze(-1)  # (B, 1)
            if bool(return_edge_reg_lr):
                if edge_reg_low is None:
                    edge_reg_low = out.new_tensor(0.0, dtype=torch.float32)
                if edge_reg_high is None:
                    edge_reg_high = out.new_tensor(0.0, dtype=torch.float32)
                return out, edge_reg_low.to(dtype=out_dtype), edge_reg_high.to(dtype=out_dtype)
            if bool(return_edge_reg):
                if edge_reg is None:
                    edge_reg = out.new_tensor(0.0, dtype=torch.float32)
                return out, edge_reg.to(dtype=out_dtype)
            return out
        assert base is not None and residue is not None
        b = base.detach() if self.detach_inputs else base
        r = residue.detach() if self.detach_inputs else residue
        b32 = b.to(dtype=torch.float32)
        r32 = r.to(dtype=torch.float32)
        B = int(b32.shape[0])
        D = int(b32.shape[1])
        tile = int(self.tile_size)
        n_tiles = int((D + tile - 1) // tile)
        d_pad = int(n_tiles * tile)
        pad = int(d_pad - D)
        if pad > 0:
            b32p = F.pad(b32, (0, pad))
            r32p = F.pad(r32, (0, pad))
        else:
            b32p = b32
            r32p = r32
        b_tile = b32p.reshape(B, n_tiles, tile)
        r_tile = r32p.reshape(B, n_tiles, tile)
        if pad > 0:
            ar = torch.arange(d_pad, device=b_tile.device)
            mask_bool = (ar < D).reshape(1, n_tiles, tile)
            mask = mask_bool.to(dtype=torch.float32)
        else:
            mask_bool = None
            mask = None
        if mask is None:
            denom = float(tile)
            b_rms_t = torch.sqrt((b_tile * b_tile).mean(dim=2) + self.eps)
            r_rms_t = torch.sqrt((r_tile * r_tile).mean(dim=2) + self.eps)
        else:
            denom = torch.clamp(mask.sum(dim=2), min=1.0)
            b_rms_t = torch.sqrt(((b_tile * b_tile) * mask).sum(dim=2) / denom + self.eps)
            r_rms_t = torch.sqrt(((r_tile * r_tile) * mask).sum(dim=2) / denom + self.eps)
        ratio = r_rms_t / (b_rms_t + float(self.eps))
        tile_feats = torch.stack([b_rms_t, r_rms_t, ratio], dim=-1).to(dtype=tokens.dtype)
        tile_logit = self.tile_net(tile_feats).squeeze(-1)
        logit = global_logit.unsqueeze(1) + tile_logit
        sig = torch.sigmoid(logit)
        p_low = sig.new_full(sig.shape, float(self.p_floor))
        p_high = sig.new_full(sig.shape, float(self.p_ceil))
        if z_min is not None and z_max is not None:
            zmin = z_min.to(device=b_tile.device, dtype=torch.float32)
            zmax = z_max.to(device=b_tile.device, dtype=torch.float32)
            if zmin.numel() != 1:
                zmin = zmin.reshape(1, -1)
            if zmax.numel() != 1:
                zmax = zmax.reshape(1, -1)
            if zmin.numel() != 1 and int(zmin.shape[-1]) != int(D):
                zmin = None
                zmax = None
            elif zmin is not None and zmax is not None:
                if zmin.numel() == 1:
                    zminp = zmin.expand(1, d_pad)
                    zmaxp = zmax.expand(1, d_pad)
                else:
                    if pad > 0:
                        zminp = F.pad(zmin, (0, pad))
                        zmaxp = F.pad(zmax, (0, pad))
                    else:
                        zminp = zmin
                        zmaxp = zmax
                zmin_t = zminp.reshape(1, n_tiles, tile)
                zmax_t = zmaxp.reshape(1, n_tiles, tile)
                sign = torch.where(r_tile >= 0, 1.0, -1.0)
                r_safe = torch.where(r_tile.abs() < self.eps, sign * self.eps, r_tile)
                p_a = (zmin_t - b_tile) / r_safe
                p_b = (zmax_t - b_tile) / r_safe
                p_min_dim = torch.minimum(p_a, p_b)
                p_max_dim = torch.maximum(p_a, p_b)
                if mask_bool is not None:
                    neg_inf = torch.finfo(p_min_dim.dtype).min
                    pos_inf = torch.finfo(p_max_dim.dtype).max
                    p_min_dim = torch.where(mask_bool, p_min_dim, p_min_dim.new_full((), neg_inf))
                    p_max_dim = torch.where(mask_bool, p_max_dim, p_max_dim.new_full((), pos_inf))
                p_low_bound = p_min_dim.max(dim=2).values
                p_high_bound = p_max_dim.min(dim=2).values
                p_low = torch.clamp(p_low_bound, min=float(self.p_floor), max=float(self.p_ceil))
                p_high = torch.clamp(p_high_bound, min=float(self.p_floor), max=float(self.p_ceil))
                p_high = torch.maximum(p_high, p_low + float(self.eps))
        p_tile = p_low + (p_high - p_low) * sig.to(dtype=p_low.dtype)
        clip = float(max(self.clip_eps, self.eps))
        lo = float(self.p_floor) + clip
        hi = float(self.p_ceil) - clip
        if hi > lo:
            p_tile = torch.clamp(p_tile, min=lo, max=hi)
        else:
            p_tile = torch.clamp(p_tile, min=float(self.p_floor), max=float(self.p_ceil))
        if bool(fallback_bounds):
            try:
                width = (p_high - p_low).to(dtype=torch.float32)
                denom = max(float(self.p_ceil - self.p_floor), float(self.eps))
                tthr = float(self.stat_width_frac) * denom
                tthr = float(max(tthr, self.eps))
                trim_low = (p_low - float(self.p_floor)).to(dtype=torch.float32)
                trim_high = (float(self.p_ceil) - p_high).to(dtype=torch.float32)
                active_low = trim_low >= tthr
                active_high = trim_high >= tthr
                w_safe = torch.maximum(width, width.new_full((), float(self.eps)))
                ethr = float(max(float(self.stat_edge_frac), 0.0))
                edge_thr = w_safe * float(ethr)
                edge_low = (p_tile - p_low) <= edge_thr
                edge_high = (p_high - p_tile) <= edge_thr
                self._fb_add_stats(
                    float(p_tile.numel()),
                    width.sum(),
                    active_low.to(dtype=torch.float32).sum(),
                    active_high.to(dtype=torch.float32).sum(),
                    edge_low.to(dtype=torch.float32).sum(),
                    edge_high.to(dtype=torch.float32).sum(),
                )
            except Exception:
                pass
        edge_reg: Optional[torch.Tensor] = None
        edge_reg_low: Optional[torch.Tensor] = None
        edge_reg_high: Optional[torch.Tensor] = None
        if bool(return_edge_reg) or bool(return_edge_reg_lr):
            try:
                width = (p_high - p_low).to(dtype=torch.float32)
                full = max(float(self.p_ceil - self.p_floor), float(self.eps))
                min_w = float(edge_reg_min_width_frac) * full
                min_w = float(max(min_w, self.eps))
                mask_w = (width >= min_w).to(dtype=torch.float32)
                w_safe = torch.maximum(width, width.new_full((), float(self.eps)))
                q = (p_tile.to(dtype=torch.float32) - p_low.to(dtype=torch.float32)) / w_safe
                q = torch.clamp(q, 0.0, 1.0)
                m = float(edge_reg_frac)
                m = float(min(max(m, self.eps), 0.49))
                inv_m = 1.0 / m
                d_low = F.relu(m - q) * inv_m
                d_high = F.relu(q - (1.0 - m)) * inv_m
                pen_low = d_low.pow(float(edge_reg_power)) * mask_w
                pen_high = d_high.pow(float(edge_reg_power)) * mask_w
                denom = mask_w.sum() + float(self.eps)
                edge_reg_low = pen_low.sum() / denom
                edge_reg_high = pen_high.sum() / denom
                edge_reg = edge_reg_low + edge_reg_high
            except Exception:
                edge_reg = None
                edge_reg_low = None
                edge_reg_high = None
        out_dtype = residue.dtype if isinstance(residue, torch.Tensor) else tokens.dtype
        out_full = self._expand_tiles(p_tile.to(dtype=out_dtype), dim=D)  # (B, D)
        if bool(return_edge_reg_lr):
            if edge_reg_low is None:
                edge_reg_low = out_full.new_tensor(0.0, dtype=torch.float32)
            if edge_reg_high is None:
                edge_reg_high = out_full.new_tensor(0.0, dtype=torch.float32)
            return out_full, edge_reg_low.to(dtype=out_dtype), edge_reg_high.to(dtype=out_dtype)
        if bool(return_edge_reg):
            if edge_reg is None:
                edge_reg = out_full.new_tensor(0.0, dtype=torch.float32)
            return out_full, edge_reg.to(dtype=out_dtype)
        return out_full


class Scaler(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.__stnet_precision_exempt__ = True
        self.eps = float(eps)
        self.calib_mode: str = "none"
        self.register_buffer("x_mean", torch.zeros(1, dtype=torch.float64))
        self.register_buffer("x_std", torch.ones(1, dtype=torch.float64))
        self.register_buffer("y_mean", torch.zeros(1, dtype=torch.float64))
        self.register_buffer("y_std", torch.ones(1, dtype=torch.float64))
        self.register_buffer("y_min", torch.full((1,), float("-inf"), dtype=torch.float64))
        self.register_buffer("y_max", torch.full((1,), float("inf"), dtype=torch.float64))
        self.register_buffer("y_q_low", torch.full((1,), float("-inf"), dtype=torch.float64))
        self.register_buffer("y_q_high", torch.full((1,), float("inf"), dtype=torch.float64))
        self.register_buffer("affine_a", torch.ones(1, dtype=torch.float64))
        self.register_buffer("affine_b", torch.zeros(1, dtype=torch.float64))
        self.register_buffer("pw_x", torch.empty(0, dtype=torch.float64))
        self.register_buffer("pw_y", torch.empty(0, dtype=torch.float64))
        self._stats_cache_lock = threading.Lock()
        self._stats_cache_max = 8
        self._x_stats_cache: Dict[Tuple[str, int, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._y_stats_cache: Dict[Tuple[str, int, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_stats_cache_lock", None)
        state.pop("_x_stats_cache", None)
        state.pop("_y_stats_cache", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._stats_cache_lock = threading.Lock()
        if not hasattr(self, "_stats_cache_max"):
            self._stats_cache_max = 8
        self._x_stats_cache = {}
        self._y_stats_cache = {}

    def _invalidate_stats_cache(self) -> None:
        lock = getattr(self, "_stats_cache_lock", None)
        if lock is None:
            lock = threading.Lock()
            setattr(self, "_stats_cache_lock", lock)
        with lock:
            self._x_stats_cache.clear()
            self._y_stats_cache.clear()

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "Scaler":
        super()._apply(fn)
        with contextlib.suppress(Exception):
            for name in ("scale", "min_value", "max_value", "max_abs", "min_positive"):
                t = getattr(self, name, None)
                if isinstance(t, torch.Tensor) and t.is_floating_point() and t.dtype != torch.float64:
                    setattr(self, name, t.to(dtype=torch.float64))
        self._invalidate_stats_cache()
        return self

    def _load_from_state_dict(
        self,
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

    @torch.no_grad()
    def update_x(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        x_work = x.detach()
        if x_work.dim() == 1:
            x_flat = x_work.view(-1, 1)
        else:
            x_flat = x_work.reshape(-1, x_work.shape[-1])
        x_flat = x_flat.to(dtype=torch.float64)
        mean = x_flat.mean(dim=0)
        std = x_flat.std(dim=0, unbiased=False).clamp_min(self.eps)
        if self.x_mean.shape != mean.shape:
            self.x_mean.resize_(mean.shape)
        if self.x_std.shape != std.shape:
            self.x_std.resize_(std.shape)
        self.x_mean.copy_(mean)
        self.x_std.copy_(std)
        self._invalidate_stats_cache()

    @torch.no_grad()
    def update_y(self, y: torch.Tensor) -> None:
        if y.numel() == 0:
            return
        y_work = y.detach()
        if y_work.dim() == 1:
            y_flat = y_work.view(-1, 1)
        else:
            y_flat = y_work.reshape(-1, y_work.shape[-1])
        y_flat = y_flat.to(dtype=torch.float64)
        mean = y_flat.mean(dim=0)
        std = y_flat.std(dim=0, unbiased=False).clamp_min(self.eps)
        if self.y_mean.shape != mean.shape:
            self.y_mean.resize_(mean.shape)
        if self.y_std.shape != std.shape:
            self.y_std.resize_(std.shape)
        self.y_mean.copy_(mean)
        self.y_std.copy_(std)
        self._invalidate_stats_cache()

    @staticmethod
    def _is_compiling_or_tracing() -> bool:
        with contextlib.suppress(Exception):
            if torch.jit.is_tracing() or torch.jit.is_scripting():
                return True
        with contextlib.suppress(Exception):
            if getattr(torch, "_dynamo", None) is not None and torch._dynamo.is_compiling():
                return True
        with contextlib.suppress(Exception):
            if getattr(torch, "compiler", None) is not None and torch.compiler.is_compiling():
                return True
        with contextlib.suppress(Exception):
            if getattr(torch, "onnx", None) is not None and hasattr(torch.onnx, "is_in_onnx_export"):
                if torch.onnx.is_in_onnx_export():
                    return True
        return False

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        compiling = self._is_compiling_or_tracing()
        if not compiling:
            if x.numel() == 0:
                return x
            feat_dim = int(x.shape[0] if x.dim() == 1 else x.shape[-1])
            if self.x_mean.numel() not in (1, feat_dim) or self.x_std.numel() not in (1, feat_dim):
                raise RuntimeError(
                    "Scaler.normalize_x: feature dimension mismatch: "
                    f"got {feat_dim} features, expected {int(self.x_mean.numel())}"
                )
        key = (x.device.type, int(x.device.index) if x.device.index is not None else -1, x.dtype)
        with self._stats_cache_lock:
            cached = self._x_stats_cache.get(key)
        if cached is None:
            mean_b = self.x_mean.to(device=x.device, dtype=x.dtype)
            std_b = self.x_std.to(device=x.device, dtype=x.dtype)
            with self._stats_cache_lock:
                if len(self._x_stats_cache) >= int(self._stats_cache_max):
                    self._x_stats_cache.clear()
                self._x_stats_cache[key] = (mean_b, std_b)
        else:
            mean_b, std_b = cached
        if x.dim() == 1:
            return (x - mean_b) / (std_b + self.eps)
        view_shape = [1] * (x.dim() - 1) + [-1]
        if compiling:
            mean = mean_b.view(*view_shape)
            std = std_b.view(*view_shape)
        else:
            mean = mean_b if mean_b.numel() == 1 else mean_b.view(*view_shape)
            std = std_b if std_b.numel() == 1 else std_b.view(*view_shape)
        return (x - mean) / (std + self.eps)

    def denormalize_x(self, x_scaled: torch.Tensor) -> torch.Tensor:
        compiling = self._is_compiling_or_tracing()
        if not compiling:
            if x_scaled.numel() == 0:
                return x_scaled
            feat_dim = int(x_scaled.shape[0] if x_scaled.dim() == 1 else x_scaled.shape[-1])
            if self.x_mean.numel() not in (1, feat_dim) or self.x_std.numel() not in (1, feat_dim):
                raise RuntimeError(
                    "Scaler.denormalize_x: feature dimension mismatch: "
                    f"got {feat_dim} features, expected {int(self.x_mean.numel())}"
                )
        key = (
            x_scaled.device.type,
            int(x_scaled.device.index) if x_scaled.device.index is not None else -1,
            x_scaled.dtype,
        )
        with self._stats_cache_lock:
            cached = self._x_stats_cache.get(key)
        if cached is None:
            mean_b = self.x_mean.to(device=x_scaled.device, dtype=x_scaled.dtype)
            std_b = self.x_std.to(device=x_scaled.device, dtype=x_scaled.dtype)
            with self._stats_cache_lock:
                if len(self._x_stats_cache) >= int(self._stats_cache_max):
                    self._x_stats_cache.clear()
                self._x_stats_cache[key] = (mean_b, std_b)
        else:
            mean_b, std_b = cached
        if x_scaled.dim() == 1:
            return x_scaled * (std_b + self.eps) + mean_b
        view_shape = [1] * (x_scaled.dim() - 1) + [-1]
        if compiling:
            std = std_b.view(*view_shape)
            mean = mean_b.view(*view_shape)
        else:
            std = std_b if std_b.numel() == 1 else std_b.view(*view_shape)
            mean = mean_b if mean_b.numel() == 1 else mean_b.view(*view_shape)
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
            self._invalidate_stats_cache()
            mean = self.y_mean
            std = self.y_std
        return mean, std

    def normalize_y(self, y: torch.Tensor) -> torch.Tensor:
        compiling = self._is_compiling_or_tracing()
        if not compiling and y.numel() == 0:
            return y
        orig_shape = y.shape
        if y.dim() == 1:
            y_flat = y.view(1, -1)
            batch_first = False
        else:
            y_flat = y.view(y.shape[0], -1)
            batch_first = True
        mean_vec, std_vec = self._y_stats_vector()
        key = (
            y_flat.device.type,
            int(y_flat.device.index) if y_flat.device.index is not None else -1,
            y_flat.dtype,
        )
        with self._stats_cache_lock:
            cached = self._y_stats_cache.get(key)
        if cached is None:
            mean = mean_vec.to(device=y_flat.device, dtype=y_flat.dtype)
            std = std_vec.to(device=y_flat.device, dtype=y_flat.dtype)
            with self._stats_cache_lock:
                if len(self._y_stats_cache) >= int(self._stats_cache_max):
                    self._y_stats_cache.clear()
                self._y_stats_cache[key] = (mean, std)
        else:
            mean, std = cached
        if compiling:
            z_flat = (y_flat - mean.view(1, -1)) / (std.view(1, -1) + self.eps)
        else:
            if mean.numel() == 1 and std.numel() == 1:
                z_flat = (y_flat - mean) / (std + self.eps)
            else:
                if y_flat.shape[1] != mean.numel():
                    raise RuntimeError(
                        "Scaler.normalize_y: feature dimension mismatch: "
                        f"got {y_flat.shape[1]} features, expected {int(mean.numel())}"
                    )
                z_flat = (y_flat - mean.view(1, -1)) / (std.view(1, -1) + self.eps)
        return z_flat.view(orig_shape) if batch_first else z_flat.view(-1)

    def denormalize_y(self, z: torch.Tensor) -> torch.Tensor:
        compiling = self._is_compiling_or_tracing()
        if not compiling and z.numel() == 0:
            return z
        orig_shape = z.shape
        if z.dim() == 1:
            z_flat = z.view(1, -1)
            batch_first = False
        else:
            z_flat = z.view(z.shape[0], -1)
            batch_first = True
        mean_vec, std_vec = self._y_stats_vector()
        key = (
            z_flat.device.type,
            int(z_flat.device.index) if z_flat.device.index is not None else -1,
            z_flat.dtype,
        )
        with self._stats_cache_lock:
            cached = self._y_stats_cache.get(key)
        if cached is None:
            mean = mean_vec.to(device=z_flat.device, dtype=z_flat.dtype)
            std = std_vec.to(device=z_flat.device, dtype=z_flat.dtype)
            with self._stats_cache_lock:
                if len(self._y_stats_cache) >= int(self._stats_cache_max):
                    self._y_stats_cache.clear()
                self._y_stats_cache[key] = (mean, std)
        else:
            mean, std = cached
        if compiling:
            y_flat = z_flat * std.view(1, -1) + mean.view(1, -1)
        else:
            if mean.numel() == 1 and std.numel() == 1:
                y_flat = z_flat * std + mean
            else:
                if z_flat.shape[1] != mean.numel():
                    raise RuntimeError(
                        "Scaler.denormalize_y: feature dimension mismatch: "
                        f"got {z_flat.shape[1]} features, expected {int(mean.numel())}"
                    )
                y_flat = z_flat * std.view(1, -1) + mean.view(1, -1)
        return y_flat.view(orig_shape) if batch_first else y_flat.view(-1)

    def calibrate(self, z_raw: torch.Tensor) -> torch.Tensor:
        match self.calib_mode:
            case "piecewise":
                if self.pw_x.numel() > 0 and self.pw_y.numel() > 0:
                    return self._piecewise(z_raw)
                return z_raw
            case "affine" | "none":
                if self.affine_a.numel() > 0:
                    return self.affine(z_raw)
                return z_raw
            case _:
                return z_raw

    def affine(self, z_raw: torch.Tensor) -> torch.Tensor:
        if self.affine_a.numel() == 0:
            return z_raw
        a = self.affine_a.to(device=z_raw.device, dtype=z_raw.dtype)
        b = self.affine_b.to(device=z_raw.device, dtype=z_raw.dtype)
        return z_raw * a + b

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
        tiny_mask = denom.abs() < self.eps
        if bool(tiny_mask.any().item()):
            denom_safe = denom.clone()
            denom_safe[tiny_mask] = 1.0
        else:
            denom_safe = denom
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

    def _piecewise(self, z_raw: torch.Tensor) -> torch.Tensor:
        if self.pw_x.numel() == 0 or self.pw_y.numel() == 0:
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
    __stnet_precision_exempt__: bool = True

    def __init__(self) -> None:
        super().__init__()
        self.__stnet_precision_exempt__ = True
        self.register_buffer("start", torch.zeros(1, dtype=torch.float64), persistent=True)
        self.register_buffer("end", torch.zeros(1, dtype=torch.float64), persistent=True)
        self.timezone: str = "UTC"
        self.register_buffer("peers", torch.zeros(1, dtype=torch.int64), persistent=True)
        self.register_buffer("epochs", torch.zeros(1, dtype=torch.int64), persistent=True)
        self.os: str = ""
        self.kernel: str = ""
        self.cpu: List[str] = []
        self.arch: List[str] = []
        self.ram_gb: float = 0.0
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
                import time
                now = datetime.datetime.now().astimezone()
                tzinfo = now.tzinfo
                tz_key = getattr(tzinfo, "key", None) if tzinfo is not None else None
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
            from ..core.system import cpu_info, process_cpu_count
            n_cores = max(1, int(process_cpu_count() or 1))
            model_name: Optional[str] = None
            with contextlib.suppress(Exception):
                info = cpu_info()
                first = info.split(";", 1)[0]
                cand = first.split(":", 1)[1] if ":" in first else first
                cand = str(cand).strip()
                if cand:
                    model_name = cand
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
            from ..core.system import Memory
            total_bytes = Memory.total()
            if total_bytes is not None and int(total_bytes) > 0:
                self.ram_gb = float(round(float(total_bytes) / (1024.0 ** 3), 2))
            else:
                self.ram_gb = float(ram_gb)
        except Exception:
            self.ram_gb = float(ram_gb)
        backend_devices: List[str] = []
        try:
            from ..core.system import get_num_accelerators, is_accelerator_available
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
                get_name = getattr(xpu_mod, "get_device_name", None) if xpu_mod is not None else None
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

        xvar_dev, xm_dev, xmin_dev, xmax_dev = _tensor_stats(x_det)
        yvar_dev, ym_dev, ymin_dev, ymax_dev = _tensor_stats(y_det)
        stats_device = self.sampled_x_mean.device
        xm = xm_dev.to(device=stats_device, dtype=torch.float64)
        xvar = xvar_dev.to(device=stats_device, dtype=torch.float64)
        xmin = xmin_dev.to(device=stats_device, dtype=torch.float64)
        xmax = xmax_dev.to(device=stats_device, dtype=torch.float64)
        ym = ym_dev.to(device=stats_device, dtype=torch.float64)
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

    def save(self) -> Sequence[Mapping[str, Any]]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()
        self._global_step = 0

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "Recorder":
        super()._apply(fn)
        with contextlib.suppress(Exception):
            for name, buf in self._buffers.items():
                if buf is None or (not isinstance(buf, torch.Tensor)):
                    continue
                if buf.is_floating_point():
                    if buf.dtype != torch.float64:
                        setattr(self, name, buf.to(dtype=torch.float64))
                else:
                    if buf.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                        if buf.dtype != torch.int64:
                            setattr(self, name, buf.to(dtype=torch.int64))
        return self


_STNET_CHECKPOINT_MODE = checkpoint_mode()
