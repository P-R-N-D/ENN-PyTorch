# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import math
import os
import threading
from collections import OrderedDict
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

from ..data.datatype import env_str

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

_CHECKPOINT_KWARGS: set[str] = set()
with contextlib.suppress(Exception):
    import inspect

    _CHECKPOINT_KWARGS = set(inspect.signature(_checkpoint).parameters.keys())

# Module-level lock to guard runtime-only cache initialization in free-threaded Python.
_RUNTIME_INIT_LOCK = threading.Lock()

_DILATED_MASK_CACHE_MAX = 32
_FLEX_BLOCK_MASK_CACHE_MAX = 16

_DILATED_MASK_CACHE_MAX_L = int(os.environ.get("STNET_DILATED_MASK_CACHE_MAX_L", "4096"))
_DILATED_MASK_CACHE_ENTRY_MAX_BYTES = int(
    os.environ.get("STNET_DILATED_MASK_CACHE_ENTRY_MAX_BYTES", str(64 * 1024 * 1024))
)
_FLEX_BLOCK_MASK_CACHE_EST_MAX_BYTES = int(
    os.environ.get("STNET_FLEX_BLOCK_MASK_CACHE_EST_MAX_BYTES", str(128 * 1024 * 1024))
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
                match m.dim():
                    case 2:
                        if m.shape != (N, N):
                            raise ValueError(
                                f"bool attn_mask shape {tuple(m.shape)} incompatible with (N,N)=({N},{N})"
                            )

                        def mask_mod(b: int, h: int, qi: int, kj: int) -> torch.Tensor:
                            return ~m[qi, kj]

                    case 3:
                        if m.shape != (B, N, N):
                            raise ValueError(
                                f"bool attn_mask shape {tuple(m.shape)} incompatible with (B={B},N={N})"
                            )

                        def mask_mod(b: int, h: int, qi: int, kj: int) -> torch.Tensor:
                            return ~m[b, qi, kj]

                    case 4:
                        b0, hm, s1, s2 = m.shape
                        if (b0 != B) or (s1 != N) or (s2 != N):
                            raise ValueError(
                                f"bool attn_mask shape {tuple(m.shape)} incompatible with (B={B},N={N})"
                            )
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

                    case _:
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

        def score_mod(
            score: torch.Tensor, b: int, h: int, qi: int, kj: int
        ) -> torch.Tensor:
            total = score
            coord_term = coord_proj[b, h, qi] - coord_proj[b, h, kj]
            total = total + coord_term.to(dtype=score.dtype)
            if mask_bias is not None:
                match mask_bias_kind:
                    case "2d":
                        total = total + mask_bias[qi, kj].to(dtype=score.dtype)
                    case "3d":
                        total = total + mask_bias[b, qi, kj].to(dtype=score.dtype)
                    case "4d1":
                        total = total + mask_bias[b, 0, qi, kj].to(dtype=score.dtype)
                    case _:
                        total = total + mask_bias[b, h, qi, kj].to(dtype=score.dtype)
            return total

        scale = 1.0 / math.sqrt(float(self.head_dim))
        if flex_attention is None:
            raise RuntimeError("flex_attention was not imported")

        # Call flex_attention with only supported kwargs (signature varies by PyTorch version).
        flex_kwargs: dict[str, Any] = {}
        if "score_mod" in _FLEX_ATTENTION_KWARGS:
            flex_kwargs["score_mod"] = score_mod
        if "block_mask" in _FLEX_ATTENTION_KWARGS:
            flex_kwargs["block_mask"] = block
        if "scale" in _FLEX_ATTENTION_KWARGS:
            flex_kwargs["scale"] = scale
        if "enable_gqa" in _FLEX_ATTENTION_KWARGS:
            flex_kwargs["enable_gqa"] = False
        if "return_lse" in _FLEX_ATTENTION_KWARGS:
            flex_kwargs["return_lse"] = False
        if "kernel_options" in _FLEX_ATTENTION_KWARGS:
            flex_kwargs["kernel_options"] = None

        out = flex_attention(qh, kh, vh, **flex_kwargs)
        if isinstance(out, tuple):
            out = out[0]
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

    def _ensure_runtime_caches(self) -> None:
        # Defensive init for old pickles / unexpected construction paths.
        # Use a module-level lock to avoid races in free-threaded Python.
        if (
            getattr(self, "_mask_cache", None) is not None
            and getattr(self, "_mask_cache_lock", None) is not None
            and getattr(self, "_flex_block_mask_cache", None) is not None
            and getattr(self, "_flex_block_mask_cache_lock", None) is not None
        ):
            return
        with _RUNTIME_INIT_LOCK:
            if getattr(self, "_mask_cache", None) is None:
                self._mask_cache = OrderedDict()
            if getattr(self, "_mask_cache_lock", None) is None:
                self._mask_cache_lock = threading.Lock()
            if getattr(self, "_flex_block_mask_cache", None) is None:
                self._flex_block_mask_cache = OrderedDict()
            if getattr(self, "_flex_block_mask_cache_lock", None) is None:
                self._flex_block_mask_cache_lock = threading.Lock()

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

    @staticmethod
    def _build_dilated_keep_mask(
        seq_len: int,
        *,
        device: torch.device,
        dilation: int = 1,
        window_size: Optional[int] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Return a boolean *keep* mask (True = allowed / attend)."""
        if int(dilation) < 1:
            raise ValueError(f"dilation must be >= 1, got {dilation}")
        L = int(seq_len)
        device = torch.device("cpu") if device is None else torch.device(device)

        # delta[q, k] = q - k
        idx = torch.arange(L, device=device)
        delta = idx[:, None] - idx[None, :]

        if int(dilation) == 1:
            keep = torch.ones((L, L), device=device, dtype=torch.bool)
        else:
            keep = (delta % int(dilation)) == 0

        if window_size is not None:
            win = int(window_size)
            keep &= (delta.abs() <= win)

        if causal:
            keep &= (delta >= 0)

        return keep.contiguous()

    def _get_mask_keep(self, L: int, device: torch.device) -> torch.Tensor:
        self._ensure_runtime_caches()

        if int(L) > _DILATED_MASK_CACHE_MAX_L:
            return self._build_dilated_keep_mask(
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

        cache = self._mask_cache
        lock = self._mask_cache_lock

        with lock:
            cached = cache.get(key)
            if cached is not None:
                with contextlib.suppress(Exception):
                    cache.move_to_end(key)
                return cached

        mask = self._build_dilated_keep_mask(
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
        self._ensure_runtime_caches()

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

        cache = self._flex_block_mask_cache
        lock = self._flex_block_mask_cache_lock

        with lock:
            cached = cache.get(key)
            if cached is not None:
                with contextlib.suppress(Exception):
                    cache.move_to_end(key)
                return cached

        # Rough estimate: bool mask is 1 byte per element.
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

        if create_block_mask is None:
            raise RuntimeError("create_block_mask was not imported")

        block_mask = create_block_mask(
            _mask_mod,
            int(B),
            int(H),
            int(L_q),
            int(L_k),
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
            transposed = True

            # key_padding_mask is commonly provided as (B, L) regardless of batch_first.
            # Accept both (B, L) and (L, B) for compatibility.
            if key_padding_mask is not None:
                if key_padding_mask.dim() != 2:
                    raise ValueError(
                        f"key_padding_mask must be rank-2, got {key_padding_mask.dim()}"
                    )
                if key_padding_mask.shape == (x.shape[1], x.shape[0]):  # (L, B)
                    key_padding_mask = key_padding_mask.transpose(0, 1)
                elif key_padding_mask.shape != (x.shape[0], x.shape[1]):  # (B, L)
                    raise ValueError(
                        f"key_padding_mask must be (B,L)=({x.shape[0]},{x.shape[1]}) or (L,B)=({x.shape[1]},{x.shape[0]}), got {tuple(key_padding_mask.shape)}"
                    )

        B, L, D = x.shape
        if D != self.embed_dim:
            raise ValueError(f"x.shape[-1]={D} must match embed_dim={self.embed_dim}")

        # Convention in this codebase: bool masks use True = masked / padded.
        kpm: Optional[torch.Tensor] = None
        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, L):
                raise ValueError(
                    f"key_padding_mask must be (B, L)=({B},{L}), got {tuple(key_padding_mask.shape)}"
                )
            kpm = (
                key_padding_mask
                if key_padding_mask.dtype is torch.bool
                else key_padding_mask.to(torch.bool)
            )
            with contextlib.suppress(Exception):
                kpm = kpm.contiguous()

        # Query padding is handled by zeroing the output (avoids inflating attn masks).
        q_pad: Optional[torch.Tensor] = None
        if kpm is not None:
            q_pad = kpm  # (B, L)

        is_simple = (int(self.dilation) == 1) and (self.window_size is None)

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

        # Decide flex vs SDPA after we know bucketing and whether weights are requested.
        use_flex = bool(_HAS_FLEX_ATTENTION and x.is_cuda and (not need_weights))
        if use_flex:
            if is_simple:
                # SDPA is typically faster for the simple (no window/dilation) case.
                use_flex = False
            elif kpm is not None:
                # With per-batch padding, SDPA is often cheaper than regenerating
                # per-batch flex block masks. Keep flex only for very large masks.
                est = int(B) * int(L) * int(L_k)
                if est < 4 * 1024 * 1024:
                    use_flex = False

        x_k = x
        kpm_k: Optional[torch.Tensor] = kpm
        if pad_len > 0:
            # Avoid `torch.cat` (extra allocations). Pre-allocate once and copy.
            x_pad = x.new_zeros((B, L_k, D))
            x_pad[:, :L, :].copy_(x)
            x_k = x_pad

            if kpm is not None:
                kpm_b = kpm if kpm.dtype is torch.bool else kpm.to(torch.bool)
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

        # Move query padding mask to device only if/when needed.
        q_pad_dev: Optional[torch.Tensor] = None
        if q_pad is not None and q_pad.device != x_k.device:
            with contextlib.suppress(Exception):
                q_pad_dev = q_pad.to(device=x_k.device, non_blocking=True)
        else:
            q_pad_dev = q_pad
        with contextlib.suppress(Exception):
            if q_pad_dev is not None:
                q_pad_dev = q_pad_dev.contiguous()

        qkv = self.qkv(x_k)
        q, k, v = qkv.chunk(3, dim=-1)

        H = self.num_heads
        Dh = self.head_dim
        L_q = L

        qh = q[:, :L_q, :].reshape(B, L_q, H, Dh).transpose(1, 2)
        kh = k.reshape(B, L_k, H, Dh).transpose(1, 2)
        vh = v.reshape(B, L_k, H, Dh).transpose(1, 2)

        training = bool(self.training)
        dropout_p = float(self.dropout_p) if training else 0.0

        attn_w: Optional[torch.Tensor] = None
        attn_out: Optional[torch.Tensor] = None

        if use_flex:
            win = int(self.window_size) if self.window_size is not None else None

            if L_k <= 2048:
                _block_size = 128
            elif L_k <= 16384:
                _block_size = 256
            else:
                _block_size = 512

            max_group = int(getattr(self, "flex_batch_microbatch", 0) or B)
            max_group = max(1, min(B, max_group))

            # Ensure key padding mask is on the same device when needed.
            kpm_k_dev: Optional[torch.Tensor] = None
            if kpm_k is not None:
                if kpm_k.device != x_k.device:
                    with contextlib.suppress(Exception):
                        kpm_k_dev = kpm_k.to(device=x_k.device, non_blocking=True)
                else:
                    kpm_k_dev = kpm_k
                with contextlib.suppress(Exception):
                    if kpm_k_dev is not None:
                        kpm_k_dev = kpm_k_dev.to(torch.bool).contiguous()

            scale = 1.0 / math.sqrt(float(Dh))

            def _run_flex_for_group(group_size: int) -> torch.Tensor:
                out_full: Optional[torch.Tensor] = None
                for b0 in range(0, B, group_size):
                    b1 = min(B, b0 + group_size)
                    B_g = b1 - b0

                    qh_g = qh[b0:b1]
                    kh_g = kh[b0:b1]
                    vh_g = vh[b0:b1]

                    kpm_g: Optional[torch.Tensor] = None
                    if kpm_k_dev is not None:
                        kpm_g = kpm_k_dev[b0:b1]

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
                            keep &= (~kpm_g[b, kv_idx])

                            return keep

                        if create_block_mask is None:
                            raise RuntimeError("create_block_mask was not imported")

                        block_mask_g = create_block_mask(
                            mask_mod_g,
                            int(B_g),
                            int(H),
                            int(L_q),
                            int(L_k),
                            device=x_k.device,
                            BLOCK_SIZE=int(_block_size),
                        )

                    if flex_attention is None:
                        raise RuntimeError("flex_attention was not imported")

                    flex_kwargs: dict[str, Any] = {"block_mask": block_mask_g}
                    if _FLEX_ATTENTION_KWARGS:
                        if "scale" in _FLEX_ATTENTION_KWARGS:
                            flex_kwargs["scale"] = scale
                        if dropout_p > 0.0:
                            if "dropout_p" in _FLEX_ATTENTION_KWARGS:
                                flex_kwargs["dropout_p"] = dropout_p
                            elif "dropout" in _FLEX_ATTENTION_KWARGS:
                                flex_kwargs["dropout"] = dropout_p
                    y_g = flex_attention(qh_g, kh_g, vh_g, **flex_kwargs)

                    out_g = self.out_proj(
                        y_g.transpose(1, 2).contiguous().view(B_g, L_q, self.embed_dim)
                    )
                    if q_pad_dev is not None:
                        out_g = out_g.masked_fill(q_pad_dev[b0:b1].unsqueeze(-1), 0.0)

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
                            empty_device_cache(
                                device=x_k.device, do_gc=False, min_interval_s=0.0
                            )
                    group //= 2

            if last_oom is not None:
                raise last_oom

        else:
            # Varlen causal fast-path:
            # For right-padded batches under simple causal attention, we can avoid
            # constructing a (B, Lq, Lk) boolean mask by slicing each group to its
            # true length and using is_causal=True.
            if (
                (not need_weights)
                and is_simple
                and bool(self.causal)
                and (kpm_k is not None)
                and isinstance(kpm_k, torch.Tensor)
                and (kpm_k.dim() == 2)
                and (kpm_k.device.type == "cpu")
            ):
                kpm_b = kpm_k.to(torch.bool)
                right_padded = True
                if int(kpm_b.shape[1]) >= 2:
                    right_padded = not (kpm_b[:, :-1] & (~kpm_b[:, 1:])).any().item()
                if right_padded:
                    lengths = (
                        (~kpm_b)
                        .sum(dim=-1)
                        .clamp(min=0, max=int(L_q))
                        .to(torch.int64)
                    )
                    y_full = qh.new_zeros((B, H, L_q, Dh))
                    unique_lengths = torch.unique(lengths).tolist()
                    for li in unique_lengths:
                        li = int(li)
                        if li <= 0:
                            continue
                        idx_cpu = (lengths == li).nonzero(as_tuple=False).squeeze(-1)
                        if idx_cpu.numel() == 0:
                            continue
                        idx = idx_cpu.to(device=qh.device, non_blocking=True)
                        q_g = qh.index_select(0, idx)[:, :, :li, :]
                        k_g = kh.index_select(0, idx)[:, :, :li, :]
                        v_g = vh.index_select(0, idx)[:, :, :li, :]
                        y_g = F.scaled_dot_product_attention(
                            q_g,
                            k_g,
                            v_g,
                            attn_mask=None,
                            dropout_p=dropout_p,
                            is_causal=True,
                        )
                        y_full[idx, :, :li, :] = y_g

                    attn_out = self.out_proj(
                        y_full.transpose(1, 2).contiguous().view(B, L_q, self.embed_dim)
                    )
                    if q_pad_dev is not None:
                        attn_out = attn_out.masked_fill(q_pad_dev.unsqueeze(-1), 0.0)

            if attn_out is None:
                # Ensure masks are on the same device as attention tensors.
                kpm_k_dev: Optional[torch.Tensor] = None
                if kpm_k is not None:
                    if kpm_k.device != x_k.device:
                        with contextlib.suppress(Exception):
                            kpm_k_dev = kpm_k.to(device=x_k.device, non_blocking=True)
                    else:
                        kpm_k_dev = kpm_k
                    with contextlib.suppress(Exception):
                        if kpm_k_dev is not None:
                            kpm_k_dev = kpm_k_dev.to(torch.bool).contiguous()

                # SDPA boolean attn_mask semantics:
                # True indicates the element *should* take part in attention (keep mask).
                base_keep: Optional[torch.Tensor] = None
                is_causal_sdpa = False

                # Only use SDPA's is_causal fast-path when there are no additional mask needs.
                if is_simple and bool(self.causal) and (kpm_k_dev is None) and (not need_weights):
                    is_causal_sdpa = True
                else:
                    # Need explicit mask for:
                    #  - causal + any padding mask
                    #  - dilation/windowed attention (with or without causal)
                    if (not is_simple) or bool(self.causal):
                        base_keep_full = self._get_mask_keep(L_k, x_k.device)
                        base_keep = base_keep_full[:L_q, :]

                key_keep: Optional[torch.Tensor] = None
                if kpm_k_dev is not None:
                    if kpm_k_dev.dim() == 1:
                        key_keep = (~kpm_k_dev)[None, None, None, :]  # (1,1,1,L_k)
                    else:
                        key_keep = (~kpm_k_dev)[:, None, None, :]      # (B,1,1,L_k)

                if need_weights:
                    # Explicit attention to return weights (slow/alloc-heavy by design).
                    base4: Optional[torch.Tensor] = None
                    if base_keep is not None:
                        base4 = base_keep[None, None, :, :]  # (1,1,Lq,Lk)

                    # Decide microbatch size to reduce peak memory in the explicit path.
                    env_mb = 0
                    with contextlib.suppress(Exception):
                        env_mb = int(
                            os.environ.get(
                                "STNET_ATTENTION_WEIGHTS_BATCH_MICROBATCH", "0"
                            )
                            or 0
                        )

                    group = int(env_mb)
                    if group <= 0:
                        est = int(B) * int(H) * int(L_q) * int(L_k)
                        if est >= 64 * 1024 * 1024:
                            group = 1
                        elif est >= 16 * 1024 * 1024:
                            group = 2
                        elif est >= 4 * 1024 * 1024:
                            group = 4
                        else:
                            group = int(B)
                    group = max(1, min(int(B), int(group)))

                    # Pre-allocate outputs.
                    out_full = qh.new_empty((B, L_q, self.embed_dim))
                    if average_attn_weights:
                        w_full = qh.new_zeros((B, L_q, L))
                    else:
                        w_full = qh.new_zeros((B, H, L_q, L))

                    scale = 1.0 / math.sqrt(float(Dh))
                    mask_min = torch.finfo(qh.dtype).min

                    last_oom: Optional[RuntimeError] = None
                    while group >= 1:
                        try:
                            for b0 in range(0, B, group):
                                b1 = min(B, b0 + group)
                                qh_g = qh[b0:b1]
                                kh_g = kh[b0:b1]
                                vh_g = vh[b0:b1]

                                keep_g: Optional[torch.Tensor] = None
                                if base4 is not None and key_keep is not None:
                                    keep_g = base4 & key_keep[b0:b1]
                                elif base4 is not None:
                                    keep_g = base4
                                elif key_keep is not None:
                                    keep_g = key_keep[b0:b1]
                                # keep_g is broadcastable to (B_g,1,Lq,Lk)

                                scores = torch.matmul(qh_g, kh_g.transpose(-2, -1)) * scale
                                if keep_g is not None:
                                    scores = scores.masked_fill(~keep_g, mask_min)

                                probs = torch.softmax(scores, dim=-1)

                                if keep_g is not None:
                                    probs = probs * keep_g.to(dtype=probs.dtype)
                                    denom = probs.sum(dim=-1, keepdim=True)
                                    probs = probs / denom.clamp(min=1e-9)
                                    probs = probs.masked_fill(denom <= 0, 0.0)

                                # Return weights before dropout (more useful for inspection).
                                w_g = probs[..., :L]
                                if average_attn_weights:
                                    w_g = w_g.mean(dim=1)
                                if q_pad_dev is not None:
                                    if average_attn_weights:
                                        w_g = w_g.masked_fill(
                                            q_pad_dev[b0:b1].unsqueeze(-1), 0.0
                                        )
                                    else:
                                        w_g = w_g.masked_fill(
                                            q_pad_dev[b0:b1].unsqueeze(1).unsqueeze(-1), 0.0
                                        )
                                w_full[b0:b1] = w_g

                                probs_drop = (
                                    F.dropout(probs, p=dropout_p, training=True)
                                    if dropout_p > 0.0
                                    else probs
                                )
                                y_g = torch.matmul(probs_drop, vh_g)

                                out_g = self.out_proj(
                                    y_g.transpose(1, 2).contiguous().view(
                                        b1 - b0, L_q, self.embed_dim
                                    )
                                )
                                if q_pad_dev is not None:
                                    out_g = out_g.masked_fill(
                                        q_pad_dev[b0:b1].unsqueeze(-1), 0.0
                                    )
                                out_full[b0:b1] = out_g

                            last_oom = None
                            break
                        except RuntimeError as e:
                            msg = str(e)
                            if "CUDA out of memory" not in msg and "out of memory" not in msg:
                                raise
                            last_oom = e
                            if x_k.device.type == "cuda":
                                with contextlib.suppress(Exception):
                                    empty_device_cache(
                                        device=x_k.device, do_gc=False, min_interval_s=0.0
                                    )
                            group //= 2

                    if last_oom is not None:
                        raise last_oom

                    attn_out = out_full
                    attn_w = w_full

                else:
                    # Standard SDPA path (no attention weights returned).
                    if base_keep is None:
                        y = F.scaled_dot_product_attention(
                            qh,
                            kh,
                            vh,
                            attn_mask=key_keep,
                            dropout_p=dropout_p,
                            is_causal=bool(is_causal_sdpa),
                        )
                    else:
                        if key_keep is None:
                            y = F.scaled_dot_product_attention(
                                qh,
                                kh,
                                vh,
                                attn_mask=base_keep,
                                dropout_p=dropout_p,
                                is_causal=False,
                            )
                        elif int(key_keep.shape[0]) == 1:
                            attn_keep = base_keep[None, None, :, :] & key_keep
                            y = F.scaled_dot_product_attention(
                                qh,
                                kh,
                                vh,
                                attn_mask=attn_keep,
                                dropout_p=dropout_p,
                                is_causal=False,
                            )
                        else:
                            # Batch-specific padding + base_keep would otherwise force a full (B, Lq, Lk)
                            # materialization. Micro-batch to reduce peak memory.
                            env_mb = 0
                            with contextlib.suppress(Exception):
                                env_mb = int(
                                    os.environ.get("STNET_SDPA_BATCH_MICROBATCH", "0")
                                    or 0
                                )

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

                            base4 = base_keep[None, None, :, :]  # (1,1,Lq,Lk)
                            out_full = qh.new_empty((B, H, L_q, Dh))

                            last_oom: Optional[RuntimeError] = None
                            while group >= 1:
                                try:
                                    for b0 in range(0, B, group):
                                        b1 = min(B, b0 + group)
                                        attn_keep_g = base4 & key_keep[b0:b1]
                                        y_g = F.scaled_dot_product_attention(
                                            qh[b0:b1],
                                            kh[b0:b1],
                                            vh[b0:b1],
                                            attn_mask=attn_keep_g,
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
                                            empty_device_cache(
                                                device=x_k.device,
                                                do_gc=False,
                                                min_interval_s=0.0,
                                            )
                                    group //= 2

                            if last_oom is not None:
                                raise last_oom
                            y = out_full

                    attn_out = self.out_proj(
                        y.transpose(1, 2).contiguous().view(B, L_q, self.embed_dim)
                    )
                    if q_pad_dev is not None:
                        attn_out = attn_out.masked_fill(q_pad_dev.unsqueeze(-1), 0.0)

        if attn_out is None:
            raise RuntimeError("Internal error: attention produced no output")

        x_out = x + self.dropout(attn_out)
        res2 = x_out
        x_out = self.norm2(x_out)

        do_ckpt_ffn = (
            self.training
            and torch.is_grad_enabled()
            and _STNET_CHECKPOINT_MODE in {"ffn", "all"}
        )
        if do_ckpt_ffn:
            ckpt_kwargs: dict[str, Any] = {}
            if "use_reentrant" in _CHECKPOINT_KWARGS:
                ckpt_kwargs["use_reentrant"] = True
            if "preserve_rng_state" in _CHECKPOINT_KWARGS:
                ckpt_kwargs["preserve_rng_state"] = True
            if "determinism_check" in _CHECKPOINT_KWARGS:
                ckpt_kwargs["determinism_check"] = "none"
            x_out = _checkpoint(self.ffn, x_out, **ckpt_kwargs)
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
