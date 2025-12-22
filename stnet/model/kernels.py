# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import importlib
import math
import os
import sys
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


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean-ish environment variable consistently."""
    raw = os.environ.get(name)
    if raw is None:
        with contextlib.suppress(Exception):
            raw = env_str(name)
    if raw is None:
        return bool(default)
    s = str(raw).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on", "enable", "enabled"}


# Allow runtime opt-out.
if _env_flag("STNET_DISABLE_FLEX_ATTENTION", default=False):
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


def _pick_block_size(L: int) -> int:
    # Keep this as a tiny helper; it's used by multiple attention modules here.
    if int(L) <= 2048:
        return 128
    if int(L) <= 16384:
        return 256
    return 512


def _call_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_mask: Any | None = None,
    score_mod: Any | None = None,
    scale: float | None = None,
    dropout_p: float = 0.0,
    enable_gqa: bool = False,
    return_lse: bool = False,
    kernel_options: Any | None = None,
) -> torch.Tensor:
    """Call flex_attention with only supported kwargs (signature varies by PyTorch version)."""
    if flex_attention is None:
        raise RuntimeError("flex_attention was not imported")

    kwargs: dict[str, Any] = {}
    if score_mod is not None and "score_mod" in _FLEX_ATTENTION_KWARGS:
        kwargs["score_mod"] = score_mod
    if block_mask is not None and "block_mask" in _FLEX_ATTENTION_KWARGS:
        kwargs["block_mask"] = block_mask
    if scale is not None and "scale" in _FLEX_ATTENTION_KWARGS:
        kwargs["scale"] = float(scale)

    if "enable_gqa" in _FLEX_ATTENTION_KWARGS:
        kwargs["enable_gqa"] = bool(enable_gqa)
    if "return_lse" in _FLEX_ATTENTION_KWARGS:
        kwargs["return_lse"] = bool(return_lse)
    if "kernel_options" in _FLEX_ATTENTION_KWARGS:
        kwargs["kernel_options"] = kernel_options

    dp = float(dropout_p)
    if dp > 0.0:
        if "dropout_p" in _FLEX_ATTENTION_KWARGS:
            kwargs["dropout_p"] = dp
        elif "dropout" in _FLEX_ATTENTION_KWARGS:
            kwargs["dropout"] = dp

    out = flex_attention(q, k, v, **kwargs)
    if isinstance(out, tuple):
        out = out[0]
    return out


def _resolve_attention_primitives() -> None:
    """
    Resolve attention primitives used by modules in this file.

    This avoids an error-prone `from .kernels import ...` pattern that can become a self-import
    when this file itself is `stnet.model.kernels`.
    """
    needed = ("DotProductAttention", "MultiHeadAttention", "MultiScaleRetention", "reshape_for_mha")
    g = globals()

    # If already present (e.g., defined later in this module), nothing to do.
    if all(callable(g.get(n)) for n in needed):
        return

    # Otherwise, try a small set of sibling module candidates.
    candidates: list[str] = []
    if __package__:
        candidates = [
            f"{__package__}.attention",
            f"{__package__}.attention_kernels",
            f"{__package__}.kernels_attention",
            f"{__package__}.ops",
            f"{__package__}.kernels",
        ]

    for modname in candidates:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        # Skip self-import.
        if mod is sys.modules.get(__name__):
            continue

        ok = True
        for n in needed:
            obj = getattr(mod, n, None)
            if not callable(obj):
                ok = False
                break
            g[n] = obj
        if ok:
            return

    missing = [n for n in needed if not callable(g.get(n))]
    raise ImportError(
        f"Missing attention primitives: {missing}. "
        "Define them in this module or make them importable from a sibling module."
    )


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
        _resolve_attention_primitives()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead for PatchAttention")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = self.d_model // self.nhead
        self.coord_dim = int(coord_dim)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=True)
        self.rel_weight = nn.Parameter(torch.zeros(self.nhead, self.coord_dim))

        # Always instantiate the fallback as a submodule so `.to()`/`.cuda()` moves it correctly.
        self._fallback_attn = DotProductAttention(num_heads=self.nhead, head_dim=self.head_dim)

    @staticmethod
    def _parse_flex_attn_mask(
        attn_mask: torch.Tensor,
        *,
        B: int,
        N: int,
        H: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Any | None, torch.Tensor | None, str | None]:
        """
        Normalize an attention mask for flex_attention usage.

        Returns (block_mask, bias, bias_kind). Exactly one of block_mask or bias is returned.
        """
        if attn_mask.dtype == torch.bool:
            m = attn_mask.to(device=device)

            def mask_mod(b: int, h: int, qi: int, kj: int) -> torch.Tensor:  # pragma: no cover
                raise RuntimeError("mask_mod should be specialized per-rank")

            match m.dim():
                case 2:
                    if m.shape != (N, N):
                        raise ValueError(
                            f"bool attn_mask shape {tuple(m.shape)} incompatible with (N,N)=({N},{N})"
                        )

                    def mask_mod(b: int, h: int, qi: int, kj: int) -> torch.Tensor:
                        # bool mask convention here: True = masked. flex expects keep-mask.
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

                    elif hm != H:
                        raise ValueError(f"bool attn_mask head dim {hm} incompatible with nhead={H}")
                    else:

                        def mask_mod(b: int, h: int, qi: int, kj: int) -> torch.Tensor:
                            return ~m[b, h, qi, kj]

                case _:
                    raise ValueError(f"bool attn_mask rank {m.dim()} not supported")

            if create_block_mask is None:
                raise RuntimeError("create_block_mask was not imported")

            block = create_block_mask(
                mask_mod,
                int(B),
                int(H),
                int(N),
                int(N),
                device=device,
                BLOCK_SIZE=int(_pick_block_size(N)),
            )
            return block, None, None

        # Non-bool masks are treated as additive bias.
        bias = attn_mask.to(device=device, dtype=dtype)
        kind: str
        match bias.dim():
            case 2:
                if bias.shape != (N, N):
                    raise ValueError(
                        f"attn_mask shape {tuple(bias.shape)} incompatible with (N,N)=(~,{N})"
                    )
                kind = "2d"
            case 3:
                if bias.shape != (B, N, N):
                    raise ValueError(
                        f"attn_mask shape {tuple(bias.shape)} incompatible with (B,N,N)=({B},{N},{N})"
                    )
                kind = "3d"
            case 4:
                b0, hm, s1, s2 = bias.shape
                if (b0 != B) or (s1 != N) or (s2 != N):
                    raise ValueError(
                        f"attn_mask shape {tuple(bias.shape)} incompatible with (B={B},N={N})"
                    )
                if hm == 1:
                    kind = "4d1"
                elif hm != H:
                    raise ValueError(f"attn_mask head dim {hm} incompatible with nhead={H}")
                else:
                    kind = "4d"
            case _:
                raise ValueError(f"attn_mask rank {bias.dim()} not supported")

        return None, bias.contiguous(), kind

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _resolve_attention_primitives()
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

        # Use the fast, non-flex path if flex is unavailable or not on CUDA.
        if (not _HAS_FLEX_ATTENTION) or (not x.is_cuda):
            yh = self._fallback_attn(qh, kh, vh, attn_mask=attn_mask, training=self.training)
            return yh.transpose(1, 2).contiguous().view(B, N, self.d_model)

        coords_f32 = coords.to(dtype=torch.float32, device=x.device).contiguous()
        if coords_f32.shape != (B, N, self.coord_dim):
            raise ValueError(
                f"coords_f32 shape {coords_f32.shape} incompatible with (B,N,C)=({B},{N},{self.coord_dim})"
            )

        # Project coordinates to per-head scalars.
        W = self.rel_weight
        if W.device != coords_f32.device:
            W = W.to(device=coords_f32.device)
        if W.dtype != coords_f32.dtype:
            W = W.to(dtype=coords_f32.dtype)

        coord_proj = torch.matmul(coords_f32, W.t()).transpose(1, 2).contiguous()  # (B,H,N)

        # Normalize/prepare the optional attention mask for flex.
        block = None
        mask_bias: torch.Tensor | None = None
        mask_bias_kind: str | None = None
        if isinstance(attn_mask, torch.Tensor):
            block, mask_bias, mask_bias_kind = self._parse_flex_attn_mask(
                attn_mask, B=B, N=N, H=self.nhead, device=qh.device, dtype=qh.dtype
            )

        # Cast once so score_mod doesn't cast per-element.
        if coord_proj.dtype != qh.dtype:
            coord_proj = coord_proj.to(dtype=qh.dtype)
        if mask_bias is not None and mask_bias.dtype != qh.dtype:
            mask_bias = mask_bias.to(dtype=qh.dtype)

        def score_mod(score: torch.Tensor, b: int, h: int, qi: int, kj: int) -> torch.Tensor:
            total = score
            total = total + (coord_proj[b, h, qi] - coord_proj[b, h, kj]).to(dtype=score.dtype)
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
        out = _call_flex_attention(qh, kh, vh, score_mod=score_mod, block_mask=block, scale=scale)

        H, Dh = self.nhead, self.head_dim
        flops = (
            2.0 * B * H * N * Dh * N
            + 2.0 * B * H * N * N * Dh
            + (B * H * N * N * self.coord_dim)
        )
        with contextlib.suppress(Exception):
            FLOP_PROFILER.add("PatchAttention", float(flops))
        return out.transpose(1, 2).contiguous().view(B, N, self.d_model)


class Retention(nn.Module):
    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()
        _resolve_attention_primitives()
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
    """
    Dilated/windowed attention with optional flex_attention backend.

    Mask semantics used internally:
      - key_padding_mask (kpm): bool with True = padded/masked tokens (common convention).
      - SDPA mask(s): bool keep-mask with True = allowed (this file's convention).
      - keep-masks returned by _build_dilated_keep_mask/_get_mask_keep: True = allowed.
    """

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
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")

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

    @staticmethod
    def _is_oom_error(e: RuntimeError) -> bool:
        msg = str(e)
        return ("CUDA out of memory" in msg) or ("out of memory" in msg)

    @staticmethod
    def _oom_safe_run(
        *,
        init_group: int,
        device: torch.device,
        run: Any,
    ) -> Any:
        """Run `run(group)` shrinking group on OOM. Returns run(group) output."""
        group = max(1, int(init_group))
        last_oom: Optional[RuntimeError] = None
        while group >= 1:
            try:
                return run(group)
            except RuntimeError as e:
                if not DilatedAttention._is_oom_error(e):
                    raise
                last_oom = e
                if device.type == "cuda":
                    with contextlib.suppress(Exception):
                        empty_device_cache(device=device, do_gc=False, min_interval_s=0.0)
                if group == 1:
                    break
                group //= 2
        if last_oom is not None:
            raise last_oom
        raise RuntimeError("Internal error: OOM retry loop exited unexpectedly")

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

        # Use int32 to reduce peak memory when constructing delta.
        idx = torch.arange(L, device=device, dtype=torch.int32)
        delta = idx[:, None] - idx[None, :]

        if int(dilation) == 1:
            keep = torch.ones((L, L), device=device, dtype=torch.bool)
        else:
            keep = (delta.remainder(int(dilation))) == 0

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
                keep &= ((dq.remainder(self.dilation)) == 0)

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

    @staticmethod
    def _to_bool_contig_on_device(
        t: torch.Tensor | None, *, device: torch.device
    ) -> torch.Tensor | None:
        if t is None:
            return None
        out = t
        if out.device != device:
            with contextlib.suppress(Exception):
                out = out.to(device=device, non_blocking=True)
        with contextlib.suppress(Exception):
            if out.dtype is not torch.bool:
                out = out.to(torch.bool)
            out = out.contiguous()
        return out

    def _normalize_x_and_kpm(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], bool]:
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

        kpm: Optional[torch.Tensor] = None
        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, L):
                raise ValueError(
                    f"key_padding_mask must be (B, L)=({B},{L}), got {tuple(key_padding_mask.shape)}"
                )
            kpm = key_padding_mask if key_padding_mask.dtype is torch.bool else key_padding_mask.to(torch.bool)
            with contextlib.suppress(Exception):
                kpm = kpm.contiguous()

        return x, kpm, transposed

    def _bucket_multiple(self, L: int) -> int:
        base_mult = max(int(getattr(self, "length_bucket_multiple", 64)), 1)
        if int(L) <= 512:
            mult = base_mult
        elif int(L) <= 2048:
            mult = base_mult * 2
        else:
            mult = base_mult * 4
        return max(int(mult), 1)

    def _choose_use_flex(self, *, x: torch.Tensor, need_weights: bool, is_simple: bool, kpm: Optional[torch.Tensor], L: int, L_k: int) -> bool:
        use_flex = bool(_HAS_FLEX_ATTENTION and x.is_cuda and (not need_weights))
        if not use_flex:
            return False
        if is_simple:
            # SDPA is typically faster for the simple (no window/dilation) case.
            return False
        if kpm is not None:
            # With per-batch padding, SDPA is often cheaper than regenerating per-batch block masks.
            est = int(x.shape[0]) * int(L) * int(L_k)
            if est < 4 * 1024 * 1024:
                return False
        return True

    def _pad_to_bucket(
        self,
        x: torch.Tensor,
        *,
        kpm: Optional[torch.Tensor],
        L_k: int,
        use_flex: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], int, int, int]:
        B, L, D = x.shape
        pad_len = int(L_k) - int(L)

        x_k = x
        kpm_k: Optional[torch.Tensor] = kpm
        if pad_len <= 0:
            return x_k, kpm_k, int(L), int(L_k), 0

        # Avoid `torch.cat` (extra allocations). Pre-allocate once and copy.
        x_pad = x.new_zeros((B, int(L_k), D))
        x_pad[:, :L, :].copy_(x)
        x_k = x_pad

        if kpm is not None:
            kpm_b = kpm if kpm.dtype is torch.bool else kpm.to(torch.bool)
            kpm_pad = torch.ones((B, int(L_k)), device=kpm_b.device, dtype=torch.bool)
            kpm_pad[:, :L].copy_(kpm_b)
            kpm_k = kpm_pad
        elif (not bool(self.causal)) and (not use_flex):
            # Non-causal attention can "see" padded keys → must mask bucket padding.
            # Use a 1D mask (L_k,) so SDPA can broadcast without B replication.
            kpm_pad_1d = torch.zeros((int(L_k),), device=x.device, dtype=torch.bool)
            kpm_pad_1d[L:] = True
            kpm_k = kpm_pad_1d
        else:
            # Causal attention never attends to kv positions >= L (kv_idx > q_idx for all q_idx < L).
            kpm_k = None

        return x_k, kpm_k, int(L), int(L_k), pad_len

    def _run_flex_attention(
        self,
        *,
        qh: torch.Tensor,
        kh: torch.Tensor,
        vh: torch.Tensor,
        x_device: torch.device,
        q_pad_dev: Optional[torch.Tensor],
        kpm_k: Optional[torch.Tensor],
        L_q: int,
        L_k: int,
        dropout_p: float,
    ) -> torch.Tensor:
        B, H, _, Dh = qh.shape
        win = int(self.window_size) if self.window_size is not None else None
        block_size = _pick_block_size(L_k)
        max_group = int(getattr(self, "flex_batch_microbatch", 0) or B)
        max_group = max(1, min(B, max_group))

        kpm_k_dev = self._to_bool_contig_on_device(kpm_k, device=x_device)
        scale = 1.0 / math.sqrt(float(Dh))

        def _run_for_group(group_size: int) -> torch.Tensor:
            out_full: Optional[torch.Tensor] = None
            for b0 in range(0, B, group_size):
                b1 = min(B, b0 + group_size)
                B_g = b1 - b0

                qh_g = qh[b0:b1]
                kh_g = kh[b0:b1]
                vh_g = vh[b0:b1]

                kpm_g: Optional[torch.Tensor] = None
                if kpm_k_dev is not None:
                    # kpm_k_dev may be (B,Lk) or (Lk,).
                    kpm_g = kpm_k_dev[b0:b1] if kpm_k_dev.dim() == 2 else kpm_k_dev

                if kpm_g is None:
                    block_mask_g = self._get_flex_block_mask(
                        B_g,
                        H,
                        L_q,
                        L_k,
                        device=x_device,
                        block_size=block_size,
                        win=win,
                    )
                else:
                    # Per-batch padding pattern: build a block mask for this group.
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
                            keep &= ((dq.remainder(self.dilation)) == 0)

                        # Mask only keys; query padding is handled by zeroing the output.
                        if kpm_g.dim() == 2:
                            keep &= (~kpm_g[b, kv_idx])
                        else:
                            keep &= (~kpm_g[kv_idx])

                        return keep

                    if create_block_mask is None:
                        raise RuntimeError("create_block_mask was not imported")

                    block_mask_g = create_block_mask(
                        mask_mod_g,
                        int(B_g),
                        int(H),
                        int(L_q),
                        int(L_k),
                        device=x_device,
                        BLOCK_SIZE=int(block_size),
                    )

                y_g = _call_flex_attention(
                    qh_g,
                    kh_g,
                    vh_g,
                    block_mask=block_mask_g,
                    scale=scale,
                    dropout_p=dropout_p,
                )

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

        return self._oom_safe_run(init_group=max_group, device=x_device, run=_run_for_group)

    def _run_varlen_causal_fastpath(
        self,
        *,
        qh: torch.Tensor,
        kh: torch.Tensor,
        vh: torch.Tensor,
        q_pad_dev: Optional[torch.Tensor],
        kpm_k: torch.Tensor,
        dropout_p: float,
        L_q: int,
        L: int,
    ) -> torch.Tensor | None:
        # Varlen causal fast-path:
        # For right-padded batches under simple causal attention, avoid constructing a (B, Lq, Lk) mask.
        B, H, _, Dh = qh.shape
        kpm_b = kpm_k.to(torch.bool)
        right_padded = True
        if int(kpm_b.shape[1]) >= 2:
            right_padded = not (kpm_b[:, :-1] & (~kpm_b[:, 1:])).any().item()
        if not right_padded:
            return None

        lengths = (~kpm_b).sum(dim=-1).clamp(min=0, max=int(L_q)).to(torch.int64)
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
                q_g, k_g, v_g, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
            y_full[idx, :, :li, :] = y_g

        out = self.out_proj(y_full.transpose(1, 2).contiguous().view(B, L_q, self.embed_dim))
        if q_pad_dev is not None:
            out = out.masked_fill(q_pad_dev.unsqueeze(-1), 0.0)
        return out

    def _pick_microbatch(self, *, B: int, est: int, env_name: str) -> int:
        env_mb = 0
        with contextlib.suppress(Exception):
            env_mb = int(os.environ.get(env_name, "0") or 0)

        group = int(env_mb)
        if group <= 0:
            if est >= 64 * 1024 * 1024:
                group = 1
            elif est >= 16 * 1024 * 1024:
                group = 2
            elif est >= 4 * 1024 * 1024:
                group = 4
            else:
                group = int(B)
        return max(1, min(int(B), int(group)))

    def _run_attention_sdpa(
        self,
        *,
        qh: torch.Tensor,
        kh: torch.Tensor,
        vh: torch.Tensor,
        x_device: torch.device,
        q_pad_dev: Optional[torch.Tensor],
        kpm_k: Optional[torch.Tensor],
        L_q: int,
        L_k: int,
        L: int,
        dropout_p: float,
        need_weights: bool,
        average_attn_weights: bool,
        is_simple: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, H, _, Dh = qh.shape

        # Ensure masks are on the same device as attention tensors.
        kpm_k_dev = self._to_bool_contig_on_device(kpm_k, device=x_device)

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
                base_keep_full = self._get_mask_keep(L_k, x_device)
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

            group = self._pick_microbatch(
                B=B,
                est=int(B) * int(H) * int(L_q) * int(L_k),
                env_name="STNET_ATTENTION_WEIGHTS_BATCH_MICROBATCH",
            )

            out_full = qh.new_empty((B, L_q, self.embed_dim))
            if average_attn_weights:
                w_full = qh.new_zeros((B, L_q, L))
            else:
                w_full = qh.new_zeros((B, H, L_q, L))

            scale = 1.0 / math.sqrt(float(Dh))
            mask_min = torch.finfo(qh.dtype).min

            def _run_for_group(group_size: int):
                for b0 in range(0, B, group_size):
                    b1 = min(B, b0 + group_size)
                    qh_g = qh[b0:b1]
                    kh_g = kh[b0:b1]
                    vh_g = vh[b0:b1]

                    keep_g: Optional[torch.Tensor] = None
                    if base4 is not None and key_keep is not None:
                        if int(key_keep.shape[0]) == 1:
                            keep_g = base4 & key_keep
                        else:
                            keep_g = base4 & key_keep[b0:b1]
                    elif base4 is not None:
                        keep_g = base4
                    elif key_keep is not None:
                        if int(key_keep.shape[0]) == 1:
                            keep_g = key_keep
                        else:
                            keep_g = key_keep[b0:b1]

                    scores = torch.matmul(qh_g, kh_g.transpose(-2, -1)) * scale
                    if keep_g is not None:
                        scores = scores.masked_fill(~keep_g, mask_min)

                    probs = torch.softmax(scores, dim=-1)

                    if keep_g is not None:
                        probs = probs * keep_g.to(dtype=probs.dtype)
                        denom = probs.sum(dim=-1, keepdim=True)
                        probs = probs / denom.clamp(min=1e-9)
                        probs = probs.masked_fill(denom <= 0, 0.0)

                    w_g = probs[..., :L]
                    if average_attn_weights:
                        w_g = w_g.mean(dim=1)
                    if q_pad_dev is not None:
                        if average_attn_weights:
                            w_g = w_g.masked_fill(q_pad_dev[b0:b1].unsqueeze(-1), 0.0)
                        else:
                            w_g = w_g.masked_fill(q_pad_dev[b0:b1].unsqueeze(1).unsqueeze(-1), 0.0)
                    w_full[b0:b1] = w_g

                    probs_drop = F.dropout(probs, p=dropout_p, training=True) if dropout_p > 0.0 else probs
                    y_g = torch.matmul(probs_drop, vh_g)

                    out_g = self.out_proj(
                        y_g.transpose(1, 2).contiguous().view(b1 - b0, L_q, self.embed_dim)
                    )
                    if q_pad_dev is not None:
                        out_g = out_g.masked_fill(q_pad_dev[b0:b1].unsqueeze(-1), 0.0)
                    out_full[b0:b1] = out_g

                return True

            self._oom_safe_run(init_group=group, device=x_device, run=_run_for_group)
            return out_full, w_full

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
                    qh, kh, vh, attn_mask=base_keep, dropout_p=dropout_p, is_causal=False
                )
            elif int(key_keep.shape[0]) == 1:
                attn_keep = base_keep[None, None, :, :] & key_keep
                y = F.scaled_dot_product_attention(
                    qh, kh, vh, attn_mask=attn_keep, dropout_p=dropout_p, is_causal=False
                )
            else:
                group = self._pick_microbatch(
                    B=B, est=int(B) * int(L_q) * int(L_k), env_name="STNET_SDPA_BATCH_MICROBATCH"
                )

                base4 = base_keep[None, None, :, :]  # (1,1,Lq,Lk)
                out_full = qh.new_empty((B, H, L_q, Dh))

                def _run_for_group(group_size: int):
                    for b0 in range(0, B, group_size):
                        b1 = min(B, b0 + group_size)
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
                    return out_full

                y = self._oom_safe_run(init_group=group, device=x_device, run=_run_for_group)

        attn_out = self.out_proj(y.transpose(1, 2).contiguous().view(B, L_q, self.embed_dim))
        if q_pad_dev is not None:
            attn_out = attn_out.masked_fill(q_pad_dev.unsqueeze(-1), 0.0)
        return attn_out, None

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x, kpm, transposed = self._normalize_x_and_kpm(x, key_padding_mask)

        B, L, D = x.shape
        is_simple = (int(self.dilation) == 1) and (self.window_size is None)

        mult = self._bucket_multiple(L)
        L_k = int(((int(L) + mult - 1) // mult) * mult)

        use_flex = self._choose_use_flex(x=x, need_weights=need_weights, is_simple=is_simple, kpm=kpm, L=L, L_k=L_k)

        x_k, kpm_k, L_q, L_k, pad_len = self._pad_to_bucket(x, kpm=kpm, L_k=L_k, use_flex=use_flex)
        x_k = self.norm1(x_k)

        q_pad_dev = self._to_bool_contig_on_device(kpm, device=x_k.device)

        qkv = self.qkv(x_k)
        q, k, v = qkv.chunk(3, dim=-1)

        H = self.num_heads
        Dh = self.head_dim

        qh = q[:, :L_q, :].reshape(B, L_q, H, Dh).transpose(1, 2)
        kh = k.reshape(B, L_k, H, Dh).transpose(1, 2)
        vh = v.reshape(B, L_k, H, Dh).transpose(1, 2)

        dropout_p = float(self.dropout_p) if bool(self.training) else 0.0

        attn_w: Optional[torch.Tensor] = None
        attn_out: Optional[torch.Tensor] = None

        if use_flex:
            attn_out = self._run_flex_attention(
                qh=qh,
                kh=kh,
                vh=vh,
                x_device=x_k.device,
                q_pad_dev=q_pad_dev,
                kpm_k=kpm_k,
                L_q=L_q,
                L_k=L_k,
                dropout_p=dropout_p,
            )
        else:
            if (
                (not need_weights)
                and is_simple
                and bool(self.causal)
                and (kpm_k is not None)
                and isinstance(kpm_k, torch.Tensor)
                and (kpm_k.dim() == 2)
                and (kpm_k.device.type == "cpu")
            ):
                # Avoid extra syncs in the hot path: only call when conditions match.
                kpm_b = kpm_k.to(torch.bool)
                right_padded = True
                if int(kpm_b.shape[1]) >= 2:
                    right_padded = not (kpm_b[:, :-1] & (~kpm_b[:, 1:])).any().item()
                if right_padded:
                    attn_out = self._run_varlen_causal_fastpath(
                        qh=qh,
                        kh=kh,
                        vh=vh,
                        q_pad_dev=q_pad_dev,
                        kpm_k=kpm_k,
                        dropout_p=dropout_p,
                        L_q=L_q,
                        L=L,
                    )

            if attn_out is None:
                attn_out, attn_w = self._run_attention_sdpa(
                    qh=qh,
                    kh=kh,
                    vh=vh,
                    x_device=x_k.device,
                    q_pad_dev=q_pad_dev,
                    kpm_k=kpm_k,
                    L_q=L_q,
                    L_k=L_k,
                    L=L,
                    dropout_p=dropout_p,
                    need_weights=need_weights,
                    average_attn_weights=average_attn_weights,
                    is_simple=is_simple,
                )

        if attn_out is None:
            raise RuntimeError("Internal error: attention produced no output")

        x_out = x + self.dropout(attn_out)
        res2 = x_out
        x_out = self.norm2(x_out)

        do_ckpt_ffn = (
            self.training and torch.is_grad_enabled() and _STNET_CHECKPOINT_MODE in {"ffn", "all"}
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
        _resolve_attention_primitives()
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
