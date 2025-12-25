# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import math
import threading
from collections import OrderedDict
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

from ..backend.compat import StochasticDepth
from ..backend.casting import env_bool, env_int, env_str

_Norm = nn.LayerNorm

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

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

_DILATED_MASK_CACHE_MAX = 32
_FLEX_BLOCK_MASK_CACHE_MAX = 16

_DILATED_MASK_CACHE_MAX_L = env_int("STNET_DILATED_MASK_CACHE_MAX_L", 4096)
_DILATED_MASK_CACHE_ENTRY_MAX_BYTES = env_int(
    "STNET_DILATED_MASK_CACHE_ENTRY_MAX_BYTES", 64 * 1024 * 1024
)
_FLEX_BLOCK_MASK_CACHE_EST_MAX_BYTES = env_int(
    "STNET_FLEX_BLOCK_MASK_CACHE_EST_MAX_BYTES", 128 * 1024 * 1024
)


def _stnet_checkpoint_mode() -> str:
    raw = str(env_str("STNET_CHECKPOINT_MODE") or env_str("STNET_CHECKPOINT") or "ffn").strip().lower()
    match raw:
        case "0" | "false" | "none" | "off" | "disable" | "disabled":
            return "none"
        case "attn" | "attention":
            return "attn"
        case "all" | "full":
            return "all"
        case _:
            # Default keeps the previous behavior: checkpoint only the FFN/MLP path.
            return "ffn"


_STNET_CHECKPOINT_MODE = _stnet_checkpoint_mode()

from ..functional.profiler import FLOP_PROFILER
from ..backend.system import empty_device_cache
from .kernels import (
    DotProductAttention,
    MultiHeadAttention,
    MultiScaleRetention,
    reshape_for_mha,
)


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


class _FlexBoolMaskMod:
    """Callable mask adapter for torch flex_attention block masks.

    The project uses boolean attention masks with semantics:
        True = masked/disallowed

    flex_attention / create_block_mask expects:
        True = allowed

    So we invert (~) the supplied mask.
    """

    __slots__ = ("m", "mode")

    def __init__(self, m: torch.Tensor, mode: str) -> None:
        self.m = m
        self.mode = mode

    def __call__(self, b: int, h: int, qi: int, kj: int) -> torch.Tensor:
        m = self.m
        match self.mode:
            case "2d":
                return ~m[qi, kj]
            case "3d":
                return ~m[b, qi, kj]
            case "4d1":
                return ~m[b, 0, qi, kj]
            case "4d":
                return ~m[b, h, qi, kj]
            case _:
                # Should never happen; keep behavior conservative.
                return ~m[qi, kj]


class _FlexScoreMod:
    """Callable score modifier for torch flex_attention.

    Adds a coordinate-dependent relative bias + optional additive attn_mask bias.
    """

    __slots__ = ("coord_proj", "mask_bias", "mask_bias_kind")

    def __init__(
        self,
        coord_proj: torch.Tensor,
        *,
        mask_bias: torch.Tensor | None,
        mask_bias_kind: str | None,
    ) -> None:
        self.coord_proj = coord_proj
        self.mask_bias = mask_bias
        self.mask_bias_kind = mask_bias_kind

    def __call__(
        self,
        score: torch.Tensor,
        b: int,
        h: int,
        qi: int,
        kj: int,
    ) -> torch.Tensor:
        total = score
        coord_term = self.coord_proj[b, h, qi] - self.coord_proj[b, h, kj]
        total = total + coord_term.to(dtype=score.dtype)

        bias = self.mask_bias
        if bias is not None:
            match self.mask_bias_kind:
                case "2d":
                    total = total + bias[qi, kj].to(dtype=score.dtype)
                case "3d":
                    total = total + bias[b, qi, kj].to(dtype=score.dtype)
                case "4d1":
                    total = total + bias[b, 0, qi, kj].to(dtype=score.dtype)
                case _:
                    total = total + bias[b, h, qi, kj].to(dtype=score.dtype)

        return total


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

        # Thread-safe bounded caches (important for free-threaded / no-GIL Python).
        # NOTE: runtime-only cache (not part of state_dict).
        self._block_mask_cache_lock = threading.Lock()
        self._block_mask_cache = OrderedDict()

    @staticmethod
    def _device_key(device: torch.device) -> Tuple[str, int]:
        idx = -1
        with contextlib.suppress(Exception):
            if device.index is not None:
                idx = int(device.index)
        return (str(device.type), idx)

    def __getstate__(self):
        state = super().__getstate__()
        # Locks are not picklable; caches are runtime-only.
        state.pop("_block_mask_cache_lock", None)
        state.pop("_block_mask_cache", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._block_mask_cache_lock = threading.Lock()
        self._block_mask_cache = OrderedDict()

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        *,
        block_mask: Any = None,
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

        block = block_mask
        mask_bias: torch.Tensor | None = None
        mask_bias_kind: str | None = None
        if isinstance(attn_mask, torch.Tensor):
            m = attn_mask
            if m.dtype == torch.bool:
                if block is not None:
                    # Caller provided a precomputed block mask.
                    pass
                else:
                    # Build (and cache) a flex block mask derived from the boolean attention mask.
                    # Cache key includes the source tensor identity + version so in-place edits
                    # cannot reuse stale masks.
                    src_ptr = int(m.data_ptr()) if m.numel() else 0
                    src_ver = int(getattr(m, "_version", 0))
                    src_shape = tuple(int(x) for x in m.shape)
                    src_rank = int(m.dim())

                    _block_size: int
                    if N <= 2048:
                        _block_size = 128
                    elif N <= 16384:
                        _block_size = 256
                    else:
                        _block_size = 512

                    cache_key = (
                        src_rank,
                        src_shape,
                        self._device_key(m.device),
                        src_ptr,
                        src_ver,
                        int(B),
                        int(self.nhead),
                        int(N),
                        int(_block_size),
                        self._device_key(qh.device),
                    )

                    cache = getattr(self, "_block_mask_cache", None)
                    if cache is None:
                        cache = OrderedDict()
                        setattr(self, "_block_mask_cache", cache)
                    lock = getattr(self, "_block_mask_cache_lock", None)
                    if lock is None:
                        lock = threading.Lock()
                        setattr(self, "_block_mask_cache_lock", lock)

                    with lock:
                        cached = cache.get(cache_key)
                        if cached is not None:
                            with contextlib.suppress(Exception):
                                cache.move_to_end(cache_key)
                            block = cached

                    if block is None:
                        # Move mask to the attention device only when we need to (cache miss).
                        if m.device != qh.device:
                            m = m.to(device=qh.device)

                        match m.dim():
                            case 2:
                                if m.shape != (N, N):
                                    raise ValueError(
                                        f"bool attn_mask shape {tuple(m.shape)} incompatible with (N,N)=({N},{N})"
                                    )
                                mask_mod = _FlexBoolMaskMod(m, "2d")

                            case 3:
                                if m.shape != (B, N, N):
                                    raise ValueError(
                                        f"bool attn_mask shape {tuple(m.shape)} incompatible with (B={B},N={N})"
                                    )
                                mask_mod = _FlexBoolMaskMod(m, "3d")

                            case 4:
                                b0, hm, s1, s2 = m.shape
                                if (b0 != B) or (s1 != N) or (s2 != N):
                                    raise ValueError(
                                        f"bool attn_mask shape {tuple(m.shape)} incompatible with (B={B},N={N})"
                                    )
                                if hm == 1:
                                    mask_mod = _FlexBoolMaskMod(m, "4d1")
                                elif hm != self.nhead:
                                    raise ValueError(
                                        f"bool attn_mask head dim {hm} incompatible with nhead={self.nhead}"
                                    )
                                else:
                                    mask_mod = _FlexBoolMaskMod(m, "4d")

                            case _:
                                raise ValueError(f"bool attn_mask rank {m.dim()} not supported")

                        if create_block_mask is None:
                            raise RuntimeError("create_block_mask was not imported")

                        block_new = create_block_mask(
                            mask_mod,
                            B,
                            self.nhead,
                            N,
                            N,
                            device=qh.device,
                            BLOCK_SIZE=_block_size,
                        )

                        # Best-effort caching: bound entry count and estimated memory.
                        est_bytes = int(B) * int(self.nhead) * int(N) * int(N)
                        if est_bytes <= int(_FLEX_BLOCK_MASK_CACHE_EST_MAX_BYTES):
                            with lock:
                                if cache.get(cache_key) is None:
                                    cache[cache_key] = block_new
                                    with contextlib.suppress(Exception):
                                        cache.move_to_end(cache_key)
                                    while len(cache) > int(_FLEX_BLOCK_MASK_CACHE_MAX):
                                        with contextlib.suppress(Exception):
                                            cache.popitem(last=False)
                        block = block_new
            else:
                bias = m.to(device=qh.device, dtype=qh.dtype)
                match bias.dim():
                    case 2:
                        if bias.shape != (N, N):
                            raise ValueError(
                                f"attn_mask shape {tuple(bias.shape)} incompatible with (N,N)=(~,{N})"
                            )
                        mask_bias_kind = "2d"
                    case 3:
                        if bias.shape != (B, N, N):
                            raise ValueError(
                                f"attn_mask shape {tuple(bias.shape)} incompatible with (B,N,N)=({B},{N},{N})"
                            )
                        mask_bias_kind = "3d"
                    case 4:
                        b0, hm, s1, s2 = bias.shape
                        if (b0 != B) or (s1 != N) or (s2 != N):
                            raise ValueError(
                                f"attn_mask shape {tuple(bias.shape)} incompatible with (B={B},N={N})"
                            )
                        if hm == 1:
                            mask_bias_kind = "4d1"
                        elif hm != self.nhead:
                            raise ValueError(
                                f"attn_mask head dim {hm} incompatible with nhead={self.nhead}"
                            )
                        else:
                            mask_bias_kind = "4d"
                    case _:
                        raise ValueError(f"attn_mask rank {bias.dim()} not supported")
                mask_bias = bias.contiguous()

        Bc, Nc, Cc = coords_f32.shape
        if (Bc != B) or (Nc != N) or (Cc != self.coord_dim):
            raise ValueError(
                f"coords_f32 shape {coords_f32.shape} incompatible with (B,N,C)=({B},{N},{self.coord_dim})"
            )
        coord_proj = torch.matmul(coords_f32, W.t()).transpose(1, 2).contiguous()

        score_mod = _FlexScoreMod(
            coord_proj,
            mask_bias=mask_bias,
            mask_bias_kind=mask_bias_kind,
        )

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
    # PyTorch SDPA semantics: bool mask True = allowed (keep)
    return allowed.contiguous()


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

        # Use project-local attention wrapper instead of calling F.scaled_dot_product_attention directly.
        # (DotProductAttention lives in stnet/model/kernels.py)
        self._dot_attn = DotProductAttention(num_heads=self.nhead, head_dim=self.head_dim)
        # Cache supported kwarg names to avoid per-forward inspect overhead (signature varies by version).
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

        # Thread-safe bounded caches (important for free-threaded / no-GIL Python).
        # NOTE: these are best-effort runtime caches (not part of state_dict).
        self._mask_cache_lock = threading.Lock()
        self._flex_block_mask_cache_lock = threading.Lock()
        self._mask_cache = OrderedDict()
        self._flex_block_mask_cache = OrderedDict()

    @staticmethod
    def _device_key(device: torch.device) -> Tuple[str, int]:
        idx = -1
        with contextlib.suppress(Exception):
            if device.index is not None:
                idx = int(device.index)
        return (str(device.type), idx)

    def __getstate__(self):
        state = super().__getstate__()
        # Locks are not picklable; caches are runtime-only (drop both).
        state.pop("_mask_cache_lock", None)
        state.pop("_flex_block_mask_cache_lock", None)
        state.pop("_mask_cache", None)
        state.pop("_flex_block_mask_cache", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._mask_cache_lock = threading.Lock()
        self._flex_block_mask_cache_lock = threading.Lock()
        self._mask_cache = OrderedDict()
        self._flex_block_mask_cache = OrderedDict()

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
                return cached
            cache[key] = mask
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
                return cached

        try:
            est_bool_bytes = int(B) * int(H) * int(L_q) * int(L_k)
        except Exception:
            est_bool_bytes = _FLEX_BLOCK_MASK_CACHE_EST_MAX_BYTES + 1
        skip_cache = est_bool_bytes > _FLEX_BLOCK_MASK_CACHE_EST_MAX_BYTES

        def _mask_mod(b, h, q_idx, kv_idx):
            dq = q_idx - kv_idx
            keep = torch.ones_like(dq, dtype=torch.bool)
            # Bucket padding handling:
            # When we pad keys to L_k > L_q (length bucketing), prevent attention to padded keys.
            # This keeps flex_attention outputs consistent with the SDPA path without building
            # an explicit padding mask tensor.
            try:
                if int(L_k) > int(L_q):
                    keep &= (kv_idx < int(L_q))
            except Exception:
                pass
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

        with lock:
            cached = cache.get(key)
            if cached is not None:
                with contextlib.suppress(Exception):
                    cache.move_to_end(key)
                return cached
            cache[key] = block_mask
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
        *,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float,
        is_causal: bool,
    ) -> torch.Tensor:
        """
        Call DotProductAttention with best-effort kwarg compatibility.
        """
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
                # Best-effort: some wrappers accept **kwargs even if signature doesn't expose it.
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
        # Canonical internal layout: (B, L, D)
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
                # Accept either (B, L) or (L, B) when batch_first=False.
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
            # Avoid inflating attention masks with query padding; mask output instead.
            q_pad = kpm
            with contextlib.suppress(Exception):
                q_pad = q_pad.contiguous()

        # Flex attention can support per-batch padding via a block mask, but it cannot return
        # attention weights. When need_weights=True, we prefer a direct "math" path that
        # computes both output and weights in one pass.
        use_flex = bool(_HAS_FLEX_ATTENTION and x.is_cuda and (not want_weights))

        # Length bucketing reduces fragmentation/overhead for the fast paths, but when we must
        # return weights it's pure overhead (it increases L_k and therefore the weights tensor).
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
            # Avoid `torch.cat` (extra allocations). Pre-allocate once and copy.
            x_pad = x.new_zeros((B, L_k, D))
            x_pad[:, :L, :].copy_(x)
            x_k = x_pad
            if kpm is not None:
                kpm_b = kpm.to(torch.bool)
                kpm_pad = torch.ones((B, L_k), device=kpm_b.device, dtype=torch.bool)
                kpm_pad[:, :L].copy_(kpm_b)
                kpm_k = kpm_pad
            elif (not bool(self.causal)) and (not use_flex):
                # Non-causal attention can "see" padded keys → must mask bucket padding.
                # Use a 1D mask (L_k,) so SDPA can broadcast without B replication.
                kpm_pad_1d = torch.zeros((L_k,), device=x.device, dtype=torch.bool)
                kpm_pad_1d[L:] = True
                kpm_k = kpm_pad_1d
            else:
                # Causal attention never attends to kv positions >= L (kv_idx > q_idx for all q_idx < L).
                kpm_k = None

        x_k = self.norm1(x_k)

        # Projections + reshape once, shared by all paths.
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

        def _masked_softmax(scores: torch.Tensor) -> torch.Tensor:
            # scores: float32, may contain -inf. This implementation avoids NaNs for fully masked rows
            # by defining softmax(-inf, ..., -inf) := 0.
            maxv = scores.max(dim=-1, keepdim=True).values
            maxv = torch.where(torch.isfinite(maxv), maxv, torch.zeros_like(maxv))
            exp = torch.exp(scores - maxv)
            denom = exp.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            return exp / denom

        if want_weights:
            # Compute output + attention weights efficiently without materializing extra large masks.
            is_simple = (int(self.dilation) == 1) and (self.window_size is None)

            base_mask_keep: Optional[torch.Tensor] = None
            is_causal = False

            if is_simple and bool(self.causal) and (kpm_k is None):
                # Causal-only, no padding: use a simple triangular mask.
                is_causal = True
            elif not (is_simple and (not bool(self.causal))):
                # Need explicit keep-mask for:
                #  - causal + any padding mask
                #  - dilation/windowed attention (with or without causal)
                base_mask_full = self._get_mask(L_k, x_k.device)
                base_mask_keep = base_mask_full[:L_q, :]

            key_mask: Optional[torch.Tensor] = None
            if kpm_k is not None:
                kpm_b = kpm_k.to(torch.bool)
                if kpm_b.dim() == 1:
                    key_mask = kpm_b[None, None, None, :]  # (1,1,1,L_k)
                else:
                    key_mask = kpm_b[:, None, None, :]  # (B,1,1,L_k)

            base_mask_out4: Optional[torch.Tensor] = None
            if base_mask_keep is not None:
                base_mask_out4 = (~base_mask_keep).to(torch.bool)[None, None, :, :]  # (1,1,Lq,Lk)

            causal_mask: Optional[torch.Tensor] = None
            if is_causal:
                causal_mask = torch.ones((L_q, L_k), device=qh.device, dtype=torch.bool).triu(diagonal=1)
                causal_mask = causal_mask[None, None, :, :]

            # Heuristic batch microbatching to reduce peak memory when computing scores/probs.
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

            # Pre-allocate outputs to avoid repeated cat/alloc.
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

                        # (B_g,H,Lq,Dh) @ (B_g,H,Dh,Lk) -> (B_g,H,Lq,Lk)
                        scores = torch.matmul(qg, kg.transpose(-2, -1))
                        scores = scores * (1.0 / math.sqrt(float(Dh)))
                        scores = scores.to(torch.float32)

                        if causal_mask is not None:
                            scores.masked_fill_(causal_mask, float("-inf"))
                        if base_mask_out4 is not None:
                            scores.masked_fill_(base_mask_out4, float("-inf"))

                        if key_mask is not None:
                            km = key_mask if key_mask.shape[0] == 1 else key_mask[b0:b1]
                            scores.masked_fill_(km, float("-inf"))

                        probs = _masked_softmax(scores)
                        if dropout_p > 0.0:
                            probs = F.dropout(probs, p=dropout_p, training=True)

                        # Write weights first (optionally averaged) in the module dtype to save memory.
                        if avg_weights:
                            attn_w_full[b0:b1] = probs.mean(dim=1).to(dtype=qh.dtype)
                        else:
                            attn_w_full[b0:b1] = probs.to(dtype=qh.dtype)

                        # Compute attention output using the (possibly dropped) probabilities.
                        probs_out = probs.to(dtype=vg.dtype)
                        yg = torch.matmul(probs_out, vg)  # (B_g,H,Lq,Dh)
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
                        kpm_g = kpm_k[b0:b1] if kpm_k.dim() == 2 else None

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

                            try:
                                if int(L_k) > int(L_q):
                                    keep &= (kv_idx < int(L_q))
                            except Exception:
                                pass

                            if self.causal:
                                keep &= (kv_idx <= q_idx)

                            if win is not None:
                                keep &= (dq.abs() <= win)

                            if self.dilation > 1:
                                keep &= ((dq % self.dilation) == 0)

                            # Mask only keys; query padding is handled by zeroing the output.
                            keep = keep & (~kpm_g[b, kv_idx])
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

            # Right-padding causal fast-path:
            # For simple causal attention, right padding is safe with is_causal=True because
            # padded keys live strictly in the future for all valid query positions.
            #
            # IMPORTANT: to avoid GPU->CPU implicit sync, run the right-padding check only
            # when we have a host-side (CPU) padding mask.
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
                # IMPORTANT:
                # PyTorch SDPA raises if both attn_mask and is_causal are set.
                # Only use is_causal when there are no additional masking needs.
                base_mask_keep: Optional[torch.Tensor] = None
                is_causal = False

                if is_simple and bool(self.causal) and (kpm_k is None):
                    # Safe fast-path: no padding mask, no dilation/window mask.
                    is_causal = True
                elif not (is_simple and (not bool(self.causal))):
                    # Need explicit mask for:
                    #  - causal + any padding mask
                    #  - dilation/windowed attention (with or without causal)
                    base_mask_full = self._get_mask(L_k, x_k.device)
                    base_mask_keep = base_mask_full[:L_q, :]
                    is_causal = False

                key_mask: Optional[torch.Tensor] = None
                if kpm_k is not None:
                    kpm_b = kpm_k.to(torch.bool)
                    # Support either (B, L_k) or (L_k,) masks:
                    if kpm_b.dim() == 1:
                        # DotProductAttention bool mask is mask-out: True = disallowed (padding).
                        key_mask = kpm_b[None, None, None, :]  # (1,1,1,L_k)
                    else:
                        key_mask = kpm_b[:, None, None, :]     # (B,1,1,L_k)

                base_mask: Optional[torch.Tensor] = None
                if base_mask_keep is not None:
                    # Convert keep-mask to mask-out semantics expected by DotProductAttention.
                    base_mask = (~base_mask_keep).to(torch.bool)

                # Combine masks with minimal materialization.
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
                        # Mask-out combination uses OR semantics.
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
                        # Batch-specific padding + base_mask would otherwise force a full (B, Lq, Lk)
                        # materialization (expand+clone). Micro-batch to reduce peak memory.
                        env_mb = int(env_int("STNET_SDPA_BATCH_MICROBATCH", 0))

                        # Heuristic default: microbatch only when mask would be large.
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

                        base4 = base_mask[None, None, :, :]  # (1,1,Lq,Lk)
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
