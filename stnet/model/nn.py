# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import logging
import math
import threading
import weakref
from collections.abc import Iterator
from typing import (
    Any,
    Callable,
    Dict,
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

try:
    from tensordict import TensorDictBase
except ImportError:
    class TensorDictBase:  # type: ignore[no-redef]
        pass


_LOGGER = logging.getLogger(__name__)

from ..api.config import ModelConfig
from ..backend.compat import (
    StochasticDepth,
    graph_break,
    is_meta_or_fake_tensor,
    torch_no_compile,
)
from ..backend.system import empty_device_cache
from ..data.datatype import env_first_int, env_int
from ..data.pipeline import resolve_feature_key, resolve_label_key
from ..functional.profiler import FLOP_PROFILER
from ..model.fused import Autocast, Gradient
from .layers import (
    CrossAttention,
    DilatedAttention,
    PatchAttention,
    Retention,
    norm_layer,
)


# -------------------------
# Small internal utilities
# -------------------------

def _infer_module_device(module: nn.Module, fallback: torch.device) -> torch.device:
    """Best-effort device inference that follows .to() / DDP moves."""
    try:
        p0 = next(module.parameters(), None)
        if p0 is not None:
            return p0.device
    except Exception:
        pass
    try:
        b0 = next(module.buffers(), None)
        if b0 is not None:
            return b0.device
    except Exception:
        pass
    return fallback


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
def _expand_morton_bits_3d(v: torch.Tensor) -> torch.Tensor:
    """Interleave bits for 3D Morton (Z-order) encoding."""
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8)) & 0x0300F00F
    v = (v | (v << 4)) & 0x030C30C3
    v = (v | (v << 2)) & 0x09249249
    return v


@torch.no_grad()
def _to_z_index(coords01: torch.Tensor, bits: int = 10) -> torch.Tensor:
    maxv = (1 << int(bits)) - 1
    xyz = (coords01 * maxv).to(torch.int32)
    x, y, z = xyz.unbind(dim=-1)
    xx = _expand_morton_bits_3d(x)
    yy = _expand_morton_bits_3d(y) << 1
    zz = _expand_morton_bits_3d(z) << 2
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
        N = int(perm.size(1))
        shift = (patch // 2) % max(N, 1)
        if shift:
            # Equivalent to gather(perm, rolled_arange), but avoids building an index tensor.
            perm = torch.roll(perm, shifts=int(shift), dims=1)
    invperm = torch.empty_like(perm)
    scatter_src = torch.arange(perm.size(1), device=perm.device)
    if scatter_src.dim() < perm.dim():
        scatter_src = scatter_src.view(1, -1).expand_as(perm)
    invperm.scatter_(1, perm, scatter_src)
    return perm, invperm


def _block_ranges(N: int, patch: int) -> Iterator[tuple[int, int]]:
    patch_i = max(1, int(patch))
    for s in range(0, int(N), patch_i):
        e = min(s + patch_i, int(N))
        yield s, e


def _materialize_layernorm_(ln: nn.Module, ref: torch.Tensor) -> None:
    if not isinstance(ln, nn.LayerNorm):
        return

    dev = ref.device
    target_dtype = (
        torch.float32
        if (
            dev.type == "cpu"
            and ref.is_floating_point()
            and ref.dtype in (torch.float16, torch.bfloat16)
        )
        else ref.dtype
    )

    w = getattr(ln, "weight", None)
    if isinstance(w, torch.Tensor) and is_meta_or_fake_tensor(w):
        ln.weight = nn.Parameter(
            torch.ones(ln.normalized_shape, device=dev, dtype=target_dtype)
        )

    b = getattr(ln, "bias", None)
    if isinstance(b, torch.Tensor) and is_meta_or_fake_tensor(b):
        ln.bias = nn.Parameter(
            torch.zeros(ln.normalized_shape, device=dev, dtype=target_dtype)
        )


def _apply_norm_fp16_safe(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if is_meta_or_fake_tensor(x):
        raise RuntimeError("meta/fake tensor reached normalization")

    x_in = x
    cast_back = False
    if (
        x_in.device.type == "cpu"
        and x_in.is_floating_point()
        and x_in.dtype in (torch.float16, torch.bfloat16)
    ):
        x_in = x_in.float()
        cast_back = True

    y = norm(x_in)
    if is_meta_or_fake_tensor(y):
        raise RuntimeError("meta/fake tensor produced by normalization")

    if cast_back:
        y = y.to(dtype=x.dtype)
    return y


def _sanitize_tensor(
    t: torch.Tensor,
    *,
    enabled: bool,
    inplace: bool,
) -> torch.Tensor:
    """Best-effort NaN/Inf sanitization.

    Kept as a module-level helper to avoid per-forward nested function creation
    in hot paths.
    """
    if not bool(enabled):
        return t
    if not (t.is_floating_point() or t.is_complex()):
        return t
    if bool(inplace):
        # In-place path avoids extra allocations, but relies on the caller to
        # ensure it's safe for autograd.
        return torch.nan_to_num_(t, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def _microbatch_prealloc(
    inp: torch.Tensor,
    microbatch: int,
    run_fn: Callable[[torch.Tensor], Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    *,
    pad_to: Optional[int] = None,
    out_dtype: Optional[torch.dtype] = None,
    cast_slice: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    stage: str = "microbatch",
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Generic microbatch runner that preallocates output buffers.

    Notes:
        - When autograd is on, padding buffers are not reused to avoid corrupting saved tensors.
        - Shapes must remain consistent across slices.
    """
    if inp.ndim < 1:
        raise ValueError(
            f"{stage}: expected batched input with ndim>=1, got shape={tuple(inp.shape)}"
        )

    total_b = int(inp.shape[0])
    mb_i = max(1, min(total_b, int(microbatch) if microbatch else total_b))
    pad_i = int(pad_to) if pad_to is not None else None
    if pad_i is not None:
        pad_i = max(1, pad_i)
        if pad_i < mb_i:
            raise ValueError(
                f"{stage}: pad_to ({pad_i}) must be >= microbatch ({mb_i})"
            )

    out_bufs: Optional[List[torch.Tensor]] = None
    reuse_pad_buffer = not torch.is_grad_enabled()
    pad_buf: Optional[torch.Tensor] = None

    for s in range(0, total_b, mb_i):
        x_slice = inp[s : s + mb_i]
        if cast_slice is not None:
            x_slice = cast_slice(x_slice)

        slice_n = int(x_slice.shape[0])
        x_in = x_slice
        did_pad = False
        if pad_i is not None and slice_n < pad_i:
            want_shape = (pad_i, *x_slice.shape[1:])
            if reuse_pad_buffer:
                if (
                    pad_buf is None
                    or pad_buf.shape != want_shape
                    or pad_buf.dtype != x_slice.dtype
                    or pad_buf.device != x_slice.device
                ):
                    pad_buf = x_slice.new_empty(want_shape)
                pad_buf.zero_()
                pad_buf[:slice_n].copy_(x_slice)
                x_in = pad_buf
            else:
                x_in = x_slice.new_zeros(want_shape)
                x_in[:slice_n].copy_(x_slice)
            did_pad = True

        out = run_fn(x_in)

        if torch.is_tensor(out):
            outs = (out,)
        else:
            outs = cast(Tuple[torch.Tensor, ...], out)

        if len(outs) == 0:
            raise RuntimeError(f"{stage}: run_fn returned an empty tuple at slice s={s}")

        processed: List[torch.Tensor] = []
        for j, t in enumerate(outs):
            if not torch.is_tensor(t):
                raise TypeError(
                    f"{stage}: run_fn output #{j} is not a Tensor (type={type(t)})"
                )
            y = t
            if did_pad:
                if y.shape[0] < slice_n:
                    raise RuntimeError(
                        f"{stage}: output batch too small after pad-slice: got={int(y.shape[0])}, expected>={slice_n} (s={s})"
                    )
                y = y[:slice_n]
            if int(y.shape[0]) != slice_n:
                raise RuntimeError(
                    f"{stage}: output batch mismatch at s={s}: got={int(y.shape[0])}, expected={slice_n}"
                )
            if out_dtype is not None and y.dtype != out_dtype:
                y = y.to(dtype=out_dtype)
            processed.append(y)

        if out_bufs is None:
            out_bufs = [y.new_empty((total_b, *y.shape[1:])) for y in processed]
        else:
            if len(out_bufs) != len(processed):
                raise RuntimeError(
                    f"{stage}: output arity changed across microbatches: first={len(out_bufs)}, now={len(processed)} (s={s})"
                )
            for k, (buf, y) in enumerate(zip(out_bufs, processed)):
                if buf.shape[1:] != y.shape[1:]:
                    raise RuntimeError(
                        f"{stage}: output shape changed for output#{k}: first={tuple(buf.shape)}, now={(total_b, *y.shape[1:])} (s={s})"
                    )

        for buf, y in zip(out_bufs, processed):
            buf[s : s + slice_n].copy_(y)

    if out_bufs is None:
        raise RuntimeError(
            f"{stage}: produced no outputs (b={total_b}, microbatch={mb_i})"
        )

    if len(out_bufs) == 1:
        return out_bufs[0]
    return tuple(out_bufs)


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

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if is_meta_or_fake_tensor(x):
            raise RuntimeError("meta/fake tensor reached PointTransformer.forward (x)")
        if is_meta_or_fake_tensor(coords):
            raise RuntimeError(
                "meta/fake tensor reached PointTransformer.forward (coords)"
            )
        if coords.shape[:2] != x.shape[:2] or coords.size(-1) != self.coord_dim:
            raise ValueError(
                f"coords must be (B, N, {self.coord_dim}), got {tuple(coords.shape)} vs x {tuple(x.shape)}"
            )
        N = int(x.shape[1])
        x = x.contiguous()
        coords = coords.contiguous()
        if attn_mask is not None:
            attn_mask = attn_mask.contiguous()
            if is_meta_or_fake_tensor(attn_mask):
                raise RuntimeError("attn_mask is meta/fake before attention")
            if attn_mask.dim() == 0:
                pass
            elif attn_mask.dim() < 2:
                raise ValueError(
                    f"attn_mask must have rank 0 or >=2; got rank {attn_mask.dim()}"
                )
            elif attn_mask.shape[-2:] != (N, N):
                raise ValueError(
                    f"attn_mask trailing dims {tuple(attn_mask.shape[-2:])} must match (N={N}, N={N})"
                )

        _materialize_layernorm_(self.norm1, x)
        _materialize_layernorm_(self.norm2, x)

        x1 = _apply_norm_fp16_safe(self.norm1, x)
        y = self.attn(x1, coords, attn_mask=attn_mask)
        x = x + self.drop_path(self.dropout(y))

        x2 = _apply_norm_fp16_safe(self.norm2, x)
        x = x + self.drop_path(self.dropout(self.ffn(x2)))
        return x


# Canonical modeling types:
#   ss: spatial-only
#   tt: temporal-only
#   st: mixed (spatio-temporal / tempo-spatial treated identically)
_MODELING_TYPE_ALIASES: dict[str, str] = {
    # spatial-only
    "ss": "ss",
    "spatial": "ss",
    "sxs": "ss",
    # temporal-only
    "tt": "tt",
    "temporal": "tt",
    "txt": "tt",
    # mixed (treat all synonyms identically)
    "st": "st",
    "ts": "st",
    "sxt": "st",
    "txs": "st",
    "temporal-spatial": "st",
    "temporo-spatial": "st",
    "temporospatial": "st",
    "tempospatial": "st",
    "tempo-spatial": "st",
    "temporalspatial": "st",
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


class SpatialAxis(nn.Module):
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

        # Runtime-configurable knobs (avoid getattr(...) all over forward).
        self.patch_size: int = int(getattr(self, "patch_size", 512) or 512)
        self.shift_order: bool = bool(getattr(self, "shift_order", True))
        self.morton_bits: int = int(getattr(self, "morton_bits", 10) or 10)

        self.register_buffer(
            "_perm_cache",
            torch.empty(0, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "_invperm_cache",
            torch.empty(0, dtype=torch.int64),
            persistent=False,
        )
        # Cache is only valid when coords are shared across batch (expanded/stride0 or B==1).
        # Meta also captures coords identity/version to avoid stale permutations when coords change in-place.
        self._perm_cache_meta: Optional[Tuple[int, int, int, int, int, int, int]] = None
        self._perm_cache_lock = threading.Lock()

    def __getstate__(self):
        state = super().__getstate__()
        # Locks are not picklable; recreate on load.
        state.pop("_perm_cache_lock", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._perm_cache_lock = threading.Lock()

    def _ensure_permutation_cache(
        self, coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        depth = len(self.blocks)
        if coords.dim() != 3:
            raise ValueError(
                f"SpatialAxis coords must be (B,N,C), got shape {tuple(coords.shape)}"
            )
        B, N, _ = coords.shape
        patch = int(self.patch_size)
        shift = bool(self.shift_order)
        bits = int(self.morton_bits)

        # Heuristic: expanded coords (stride(0)==0) are identical across batch.
        shared_template = (B <= 1) or (coords.stride(0) == 0)

        if shared_template:
            coords_ptr = int(coords.data_ptr()) if coords.numel() else 0
            coords_ver = int(getattr(coords, "_version", 0))
            meta = (
                int(N),
                int(patch),
                int(bits),
                int(shift),
                int(depth),
                coords_ptr,
                coords_ver,
            )

            lock = getattr(self, "_perm_cache_lock", None)
            if lock is None:
                lock = threading.Lock()
                setattr(self, "_perm_cache_lock", lock)

            with lock:
                perm_cache = getattr(self, "_perm_cache", None)
                inv_cache = getattr(self, "_invperm_cache", None)
                if (
                    isinstance(perm_cache, torch.Tensor)
                    and isinstance(inv_cache, torch.Tensor)
                    and self._perm_cache_meta == meta
                    and perm_cache.shape == (depth, N)
                    and inv_cache.shape == (depth, N)
                    and perm_cache.device == coords.device
                    and inv_cache.device == coords.device
                ):
                    return perm_cache, inv_cache

            coords_one = coords[:1].contiguous()
            perms: List[torch.Tensor] = []
            invperms: List[torch.Tensor] = []
            for i in range(depth):
                perm_i, inv_i = _serialize_z_index(
                    coords_one,
                    bits=bits,
                    patch=patch,
                    shift_order=shift,
                    block_index=i,
                )
                perms.append(perm_i[0].to(dtype=torch.int64))
                invperms.append(inv_i[0].to(dtype=torch.int64))

            perm_new = torch.stack(perms, dim=0).to(device=coords.device)
            inv_new = torch.stack(invperms, dim=0).to(device=coords.device)

            with lock:
                perm_cache = getattr(self, "_perm_cache", None)
                inv_cache = getattr(self, "_invperm_cache", None)
                if (
                    isinstance(perm_cache, torch.Tensor)
                    and isinstance(inv_cache, torch.Tensor)
                    and self._perm_cache_meta == meta
                    and perm_cache.shape == (depth, N)
                    and inv_cache.shape == (depth, N)
                    and perm_cache.device == coords.device
                    and inv_cache.device == coords.device
                ):
                    return perm_cache, inv_cache

                self._perm_cache = perm_new
                self._invperm_cache = inv_new
                self._perm_cache_meta = meta
            return perm_new, inv_new

        # Per-sample coords: compute permutations for each batch entry (no caching).
        perms_b: List[torch.Tensor] = []
        invperms_b: List[torch.Tensor] = []
        coords_batch = coords.contiguous()
        for i in range(depth):
            perm_i, inv_i = _serialize_z_index(
                coords_batch,
                bits=bits,
                patch=patch,
                shift_order=shift,
                block_index=i,
            )
            perms_b.append(perm_i.to(dtype=torch.int64))
            invperms_b.append(inv_i.to(dtype=torch.int64))
        perm_b = torch.stack(perms_b, dim=0)
        inv_b = torch.stack(invperms_b, dim=0)
        return perm_b, inv_b

    @torch_no_compile(reason="SpatialAxis uses dynamic gather/scatter", recursive=True)
    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if is_meta_or_fake_tensor(x):
            raise RuntimeError("x is meta/fake before SpatialAxis.forward")
        if is_meta_or_fake_tensor(coords):
            raise RuntimeError("coords is meta/fake before SpatialAxis.forward")
        if x.dim() != 3:
            raise ValueError(
                f"SpatialAxis expects (B, N, C) tokens, got shape {tuple(x.shape)}"
            )
        if coords.dim() != 3:
            raise ValueError(
                f"SpatialAxis expects (B, N, D) coords, got shape {tuple(coords.shape)}"
            )
        if x.shape[:2] != coords.shape[:2]:
            raise ValueError(
                "tokens/coords batch or length mismatch: "
                f"tokens={tuple(x.shape)} vs coords={tuple(coords.shape)}"
            )
        x = x.contiguous()

        # Preserve expanded/shared coordinate templates (stride(0)==0) to keep caching effective.
        Bc = int(coords.size(0))
        if Bc > 1 and coords.stride(0) == 0:
            coords0 = coords[0].contiguous()
            coords = coords0.unsqueeze(0).expand(Bc, -1, -1)
        else:
            coords = coords.contiguous()

        if coords.dtype != torch.float32:
            coords = coords.float()

        if attn_mask is not None:
            if is_meta_or_fake_tensor(attn_mask):
                raise RuntimeError("attn_mask is meta/fake before SpatialAxis.forward")
            attn_mask = attn_mask.contiguous()

        perm_cache, inv_cache = self._ensure_permutation_cache(coords)
        patch = int(self.patch_size)

        for i, blk in enumerate(self.blocks):
            B, N, D = x.shape
            perm_i = perm_cache[i]
            inv_i = inv_cache[i]
            if perm_i.dim() == 2:
                perm = perm_i
                invperm = inv_i
            else:
                perm = perm_i.unsqueeze(0).expand(B, N)
                invperm = inv_i.unsqueeze(0).expand(B, N)

            x_s = x.gather(1, perm.unsqueeze(-1).expand(B, N, D))
            c_s = coords.gather(1, perm.unsqueeze(-1).expand(B, N, coords.size(-1)))

            if torch.is_grad_enabled():
                out_s = torch.empty_like(x_s)
            else:
                # Inference / no-grad: reuse buffer to avoid an extra allocation.
                out_s = x_s

            # Vectorize per-block processing by folding blocks into the batch dimension.
            # This reduces Python overhead while keeping peak memory bounded.
            full = (N // patch) * patch
            if full > 0:
                nblk = full // patch

                x_full = x_s[:, :full, :].contiguous().view(B, nblk, patch, D)
                c_full = c_s[:, :full, :].contiguous().view(B, nblk, patch, coords.size(-1))

                x_flat = x_full.reshape(B * nblk, patch, D)
                c_flat = c_full.reshape(B * nblk, patch, coords.size(-1))

                mb_flat = None
                if attn_mask is not None:
                    # Build attention mask per-block only (avoid allocating (B,N,N)).
                    if attn_mask.dim() == 0:
                        mb_flat = attn_mask
                    else:
                        idx = perm[:, :full].contiguous().view(B, nblk, patch)  # (B,nblk,patch)
                        if attn_mask.dim() == 3:
                            base = attn_mask
                        elif attn_mask.dim() == 4:
                            if attn_mask.size(1) != 1:
                                raise ValueError(
                                    "attn_mask with per-head shape not supported here"
                                )
                            base = attn_mask.squeeze(1)
                        else:
                            raise ValueError("attn_mask must be rank 0, 3, or 4 here")

                        rows = base.gather(
                            1,
                            idx.reshape(B, nblk * patch)
                            .unsqueeze(-1)
                            .expand(B, nblk * patch, N),
                        ).view(B, nblk, patch, N)
                        mb = rows.gather(3, idx.unsqueeze(-2).expand(B, nblk, patch, patch))
                        mb_flat = mb.reshape(B * nblk, patch, patch).contiguous()

                y_flat = blk(x_flat, c_flat, attn_mask=mb_flat)
                out_s[:, :full, :].copy_(y_flat.reshape(B, nblk, patch, D).reshape(B, full, D))

            # Tail (non-full block)
            if full < N:
                s, e = full, N
                xb = x_s[:, s:e, :]
                cb = c_s[:, s:e, :]

                mb = None
                if attn_mask is not None:
                    if attn_mask.dim() == 0:
                        mb = attn_mask
                    else:
                        idx = perm[:, s:e]  # (B, M)
                        M = int(idx.shape[1])
                        if attn_mask.dim() == 3:
                            rows = attn_mask.gather(1, idx.unsqueeze(-1).expand(B, M, N))
                            mb = rows.gather(2, idx.unsqueeze(1).expand(B, M, M))
                        elif attn_mask.dim() == 4:
                            if attn_mask.size(1) != 1:
                                raise ValueError(
                                    "attn_mask with per-head shape not supported here"
                                )
                            base = attn_mask.squeeze(1)
                            rows = base.gather(1, idx.unsqueeze(-1).expand(B, M, N))
                            mb = rows.gather(2, idx.unsqueeze(1).expand(B, M, M))
                        else:
                            raise ValueError("attn_mask must be rank 0, 3, or 4 here")

                out_s[:, s:e, :] = blk(xb, cb, attn_mask=mb)

            x = out_s.gather(1, invperm.unsqueeze(-1).expand(B, N, D))

        out = self.norm(x)
        if is_meta_or_fake_tensor(out):
            raise RuntimeError("SpatialAxis produced meta/fake tensor")
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


class TemporalAxis(nn.Module):
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


class CrossTransformer(nn.Module):
    """Symmetric cross-axis fusion.

    Important:
        Mixed modes (ts/st/spatiotemporal/temporospatial/...) are treated identically:
        - Inputs are always (spatial_tokens, temporal_tokens)
        - Output token order is always [spatial_part, temporal_part]
    """

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
        self.cross_s = CrossAttention(d_model, nhead, dropout=dropout, norm_type=norm_type)
        self.cross_t = CrossAttention(d_model, nhead, dropout=dropout, norm_type=norm_type)
        self.mix_norm = norm_layer(norm_type, 2 * d_model)
        hid = int(2 * d_model * mlp_ratio * (2.0 / 3.0))
        from .activations import SwiGLU

        self.mix = SwiGLU(2 * d_model, hid, out_dim=d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(p=drop_path, mode="row")
        # Optional fixed mode (may be set externally).
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

        requested = self._fixed_mode if self._fixed_mode is not None else (mode or "st")
        mode_l = _coerce_modeling_types(requested)

        if mode_l == "ss":
            return self.cross_s(spatial_tokens, temporal_tokens)
        if mode_l == "tt":
            return self.cross_t(temporal_tokens, spatial_tokens)

        # Mixed mode: always compute both and return a stable concatenation.
        s_context = self.cross_s(spatial_tokens, temporal_tokens)  # (B, Ns, D)
        t_context = self.cross_t(temporal_tokens, spatial_tokens)  # (B, Nt, D)

        # Spatial enhancement conditioned on temporal summary.
        t_summary = t_context.mean(dim=1, keepdim=True).expand_as(s_context)
        base_s = torch.cat([s_context, t_summary], dim=-1)
        fused_s = self.mix(self.mix_norm(base_s))
        out_s = s_context + self.drop_path(self.dropout(fused_s))

        # Temporal enhancement conditioned on spatial summary.
        s_summary = s_context.mean(dim=1, keepdim=True).expand_as(t_context)
        base_t = torch.cat([t_context, s_summary], dim=-1)
        fused_t = self.mix(self.mix_norm(base_t))
        out_t = t_context + self.drop_path(self.dropout(fused_t))

        # Stable order: spatial part first, then temporal part.
        return torch.cat([out_s, out_t], dim=1)


def _auto_microbatch(
    device: torch.device,
    hard_max: int,
    per_sample_bytes: int,
) -> int:
    if hard_max <= 0 or per_sample_bytes <= 0:
        return 1

    dev_t = device.type
    dev_free: Optional[int] = None
    host_free: Optional[int] = None

    from ..backend.system import Memory as _Mem

    try:
        host_free = int(_Mem.available())
    except Exception:
        host_free = None

    if dev_t == "cuda" and torch.cuda.is_available():
        try:
            free, _ = torch.cuda.mem_get_info(device)
            dev_free = int(free)
        except Exception:
            dev_free = None
    elif dev_t == "xpu" and hasattr(torch, "xpu"):
        try:
            mem_get_info = getattr(torch.xpu, "mem_get_info", None)
            if callable(mem_get_info):
                free, _ = mem_get_info(device)
                dev_free = int(free)
        except Exception:
            dev_free = None
    elif dev_t == "mps":
        dev_free = None

    effective_free: Optional[int]
    if dev_t in {"cuda", "xpu", "mps"}:
        if host_free is not None and dev_free is not None:
            effective_free = min(host_free, dev_free)
        else:
            effective_free = host_free if dev_free is None else dev_free
    else:
        effective_free = host_free

    if effective_free is None or effective_free <= 0:
        return hard_max

    budget = int(effective_free * 0.35)
    max_mb = max(1, int(budget // max(per_sample_bytes, 1)))
    return max(1, min(hard_max, max_mb))


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
        dilation = int(base_dilation)
        for _ in range(int(depth)):
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
        average_attn_weights: bool = False,
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
                average_attn_weights=average_attn_weights,
            )
        out = self.norm(out)
        if need_transpose_fallback and out.dim() == 3 and out.shape[0] != out.shape[1]:
            out = out.transpose(0, 1)
        return out, attn_w


class Context(nn.Module):
    """Controller backbone wrapper.

    Originally a thin wrapper around LongNet. This module now also owns
    controller-stage microbatch sizing/execution, so Root.forward doesn't
    have to manage controller microbatching details.
    """

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

        # Controller-stage microbatch state.
        self.microbatch: int = 0
        self._auto_microbatch_pending: bool = True

    @property
    def using(self) -> str:
        return self.backbone.using

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = False,
        **_: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.backbone(
            x,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )

    def run(
        self,
        tokens: torch.Tensor,
        *,
        device: torch.device,
        meta: Any,
        amp_enabled: bool,
        auto_microbatch_fn: Callable[[torch.Tensor], int],
        graph_break_fn: Optional[Callable[[], None]] = None,
    ) -> torch.Tensor:
        """Run controller stage on tokens with internal microbatching.

        Returns:
            refined_tokens: Tensor with same shape as tokens
        """
        if tokens.ndim != 3:
            raise ValueError(
                f"Context.run expects tokens (B,N,D), got shape {tuple(tokens.shape)}"
            )
        B = int(tokens.shape[0])
        if graph_break_fn is not None:
            graph_break_fn()

        if self._auto_microbatch_pending:
            try:
                mb = int(auto_microbatch_fn(tokens))
                self.microbatch = max(1, min(B, mb))
            except Exception:
                self.microbatch = max(1, B)
            self._auto_microbatch_pending = False

        mb = max(1, min(B, int(self.microbatch) if self.microbatch else B))

        infer_mode = not torch.is_grad_enabled()
        controller_ctx = (
            Gradient.inference(self.backbone) if infer_mode else contextlib.nullcontext()
        )

        def _run_controller_chunk(chunk: torch.Tensor) -> torch.Tensor:
            with (Autocast.float(device, metadata=meta) if amp_enabled else Autocast.suspend(device)):
                out, _ = self.backbone(chunk)
            return out

        with controller_ctx:
            refined = cast(
                torch.Tensor,
                _microbatch_prealloc(tokens, mb, _run_controller_chunk, stage="controller"),
            )
        return refined


class Subcontext(nn.Module):
    def __init__(self, in_dim: int, out_shape: Sequence[int], config: ModelConfig) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_shape = tuple((int(v) for v in out_shape))
        self.out_dim = int(math.prod(self.out_shape) if self.out_shape else 1)
        self.d_model = int(config.d_model)
        self.nhead = int(config.heads)
        self.modeling_type = _coerce_modeling_types(config.modeling_type)
        self.spatial_tokens = max(1, int(config.spatial_latents))
        self.temporal_tokens = max(1, int(config.temporal_latents))
        self.mlp_ratio = float(config.mlp_ratio)
        self.dropout = float(config.dropout)
        self.drop_path = float(config.drop_path)
        self.norm_type = str(config.normalization_method)

        self.spatial_tokenizer = nn.Linear(self.in_dim, self.spatial_tokens * self.d_model)
        self.temporal_tokenizer = nn.Linear(self.in_dim, self.temporal_tokens * self.d_model)

        self.register_buffer(
            "spatial_coords_template",
            self._get_spatial_coords(self.spatial_tokens, device=torch.device("cpu")),
            persistent=False,
        )

        self.spatial_net = SpatialAxis(
            self.d_model,
            self.nhead,
            depth=max(1, int(config.spatial_depth)),
            coord_dim=self.spatial_coords_template.shape[-1],
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            drop_path=self.drop_path,
            norm_type=self.norm_type,
        )
        self.temporal_net = TemporalAxis(
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
        # Use ceil(...) so that side**3 >= n_tokens, preserving spatial diversity
        # even for small token counts (e.g. 2-7 tokens).
        side = max(1, int(math.ceil(n_tokens ** (1.0 / 3.0))))
        coords: List[Tuple[float, float, float]] = []
        for idx in range(n_tokens):
            z = idx // (side * side)
            rem = idx % (side * side)
            y = rem // side
            x = rem % side
            if side == 1:
                coords.append((0.0, 0.0, 0.0))
            else:
                denom = float(side - 1)
                coords.append((x / denom, y / denom, z / denom))
        return torch.tensor(coords, dtype=torch.float32, device=device)

    def _spatial_coords(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        _ = dtype
        # Spatial coords are constants; keep float32 to avoid autocast-induced dtype churn.
        coords = self.spatial_coords_template
        if coords.device != device or coords.dtype is not torch.float32:
            coords = coords.to(device=device, dtype=torch.float32)
        return coords.unsqueeze(0).expand(int(batch), -1, -1)

    @torch_no_compile(reason="Subcontext orchestrates eager + compiled submodules", recursive=False)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = int(x.shape[0])

        spatial_raw = self.spatial_tokenizer(x)
        expected_spatial = B * self.spatial_tokens * self.d_model
        if spatial_raw.numel() != expected_spatial:
            raise RuntimeError(
                "spatial tokenizer output has unexpected numel: "
                f"got {spatial_raw.numel()} vs expected {expected_spatial}"
            )
        spatial_tokens = spatial_raw.reshape(B, self.spatial_tokens, self.d_model).contiguous()

        temporal_raw = self.temporal_tokenizer(x)
        expected_temporal = B * self.temporal_tokens * self.d_model
        if temporal_raw.numel() != expected_temporal:
            raise RuntimeError(
                "temporal tokenizer output has unexpected numel: "
                f"got {temporal_raw.numel()} vs expected {expected_temporal}"
            )
        temporal_tokens = temporal_raw.reshape(B, self.temporal_tokens, self.d_model).contiguous()

        # Avoid side-effectful bookkeeping inside torch.compile graphs.
        try:
            import torch._dynamo as _dynamo  # type: ignore

            if not _dynamo.is_compiling():
                total_tokens = self.spatial_tokens + self.temporal_tokens
                fl_tok = 2.0 * float(B) * float(self.in_dim) * float(total_tokens * self.d_model)
                FLOP_PROFILER.add("Tokenizer", float(fl_tok))
        except Exception:
            pass

        coords = self._spatial_coords(B, x.device, spatial_tokens.dtype)
        spatial_out = self.spatial_net(spatial_tokens, coords)
        temporal_out = self.temporal_net(temporal_tokens)

        mode_l = _coerce_modeling_types(self.modeling_type)
        if mode_l == "ss":
            tokens = spatial_out
        elif mode_l == "tt":
            tokens = temporal_out
        else:
            # Mixed: always pass (spatial, temporal); CrossTransformer is symmetric and stable.
            tokens = self.perception(spatial_out, temporal_out, mode="st")

        tokens = self.norm(tokens).contiguous()
        pooled = tokens.mean(dim=1)
        flat = self.head(pooled)

        try:
            import torch._dynamo as _dynamo  # type: ignore

            if not _dynamo.is_compiling():
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
        return (tokens, context)

    def decode(self, tokens: torch.Tensor, *args: Any, apply_norm: bool = False, **kwargs: Any) -> torch.Tensor:
        if apply_norm:
            tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        flat = self.head(pooled)
        return flat.reshape(tokens.shape[0], *self.out_shape)


class _CompiledDecode(nn.Module):
    def __init__(self, norm: nn.Module, head: nn.Module, out_shape: Sequence[int]) -> None:
        super().__init__()
        self._norm = weakref.proxy(norm)
        self._head = weakref.proxy(head)
        self._out_shape = tuple(int(x) for x in out_shape)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self._norm(tokens)
        pooled = tokens.mean(dim=1)
        flat = self._head(pooled)
        return flat.reshape(tokens.shape[0], *self._out_shape)


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
        # Precision-exempt: scaler stats must remain FP64 for numerical stability.
        self.__stnet_precision_exempt__ = True
        self.eps = float(eps)
        self.calib_mode: str = "none"
        self.register_buffer("x_mean", torch.zeros(1, dtype=torch.float64))
        self.register_buffer("x_std", torch.ones(1, dtype=torch.float64))
        self.register_buffer("y_mean", torch.zeros(1, dtype=torch.float64))
        self.register_buffer("y_std", torch.ones(1, dtype=torch.float64))
        self.register_buffer("affine_a", torch.ones(1, dtype=torch.float64))
        self.register_buffer("affine_b", torch.zeros(1, dtype=torch.float64))
        self.register_buffer("pw_x", torch.empty(0, dtype=torch.float64))
        self.register_buffer("pw_y", torch.empty(0, dtype=torch.float64))

        # Cache device/dtype-converted stats to avoid per-forward allocations.
        self._stats_cache_lock = threading.Lock()
        self._stats_cache_max = 8
        self._x_stats_cache: Dict[Tuple[str, int, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._y_stats_cache: Dict[Tuple[str, int, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _invalidate_stats_cache(self) -> None:
        with self._stats_cache_lock:
            self._x_stats_cache.clear()
            self._y_stats_cache.clear()

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "Scaler":
        super()._apply(fn)
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

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
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
        mean = mean_b if mean_b.numel() == 1 else mean_b.view(*view_shape)
        std = std_b if std_b.numel() == 1 else std_b.view(*view_shape)
        return (x - mean) / (std + self.eps)

    def denormalize_x(self, x_scaled: torch.Tensor) -> torch.Tensor:
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
        if y.numel() == 0:
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
        if z.numel() == 0:
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
        """Piecewise-linear calibration without mutating module buffers.

        Previous code resized/sliced/reassigned self.pw_x/self.pw_y inside forward,
        which is unsafe under multithreading and can break torch.compile assumptions.
        This implementation treats saved knots as read-only and uses local views.
        """
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

        # Local views only (do not mutate buffers).
        pw_x = pw_x[:, :K]
        pw_y = pw_y[:, :K]

        orig_shape = z_raw.shape
        z = z_raw.unsqueeze(-1) if z_raw.ndim == 1 else z_raw
        z = z.reshape(-1, int(z.shape[-1]))

        _, C_target = z.shape
        device = z.device
        dtype = z.dtype

        out = torch.empty_like(z)

        # If saved calibration has fewer channels, reuse the last channel for extras.
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


class History(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Precision-exempt: history/logging buffers must remain FP64/INT64.
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
                import time as _time

                now = datetime.datetime.now().astimezone()
                tzinfo = now.tzinfo
                tz_key = getattr(tzinfo, "key", None) if tzinfo is not None else None
                tz_name = tzinfo.tzname(now) if tzinfo is not None else None
                tz_env = None
                try:
                    tz_env = _time.tzname[0]
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
            from ..backend.system import cpu_info, process_cpu_count

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
            from ..backend.system import Memory

            total_bytes = Memory.total()
            if total_bytes is not None and int(total_bytes) > 0:
                self.ram_gb = float(round(float(total_bytes) / (1024.0 ** 3), 2))
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
            if (
                mps is not None
                and getattr(mps, "is_available", None)
                and torch.backends.mps.is_available()
            ):
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
                new_buf = buf.detach().new_zeros(src.shape)
                try:
                    scaler._buffers[name] = new_buf
                except Exception:
                    setattr(scaler, name, new_buf)


class Root(nn.Module):
    def __init__(self, in_dim: int, out_shape: Sequence[int], config: ModelConfig) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_shape = tuple((int(x) for x in out_shape))
        self.out_dim = int(math.prod(self.out_shape))

        if config.device is not None:
            self._device = torch.device(config.device)
        else:
            if torch.cuda.is_available():
                device_name = "cuda"
            elif (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
                device_name = "mps"
            elif (getattr(torch, "is_vulkan_available", None) and torch.is_vulkan_available()):
                device_name = "vulkan"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                device_name = "xpu"
            else:
                device_name = "cpu"
            self._device = torch.device(device_name)

        self.scaler = Scaler().to(self._device)

        # Avoid shape mutation inside forward(): ensure x stats match feature dimension up front.
        with torch.no_grad():
            if self.scaler.x_mean.numel() != self.in_dim:
                self.scaler.x_mean.resize_(self.in_dim)
                self.scaler.x_std.resize_(self.in_dim)
                self.scaler.x_mean.zero_()
                self.scaler.x_std.fill_(1.0)

        # Keep History on the same device so Root.forward device inference can't be hijacked by CPU buffers.
        self.logger = History().to(self._device)

        self.is_norm_linear = bool(getattr(config, "use_linear_branch", False))
        self.linear_branch = nn.Linear(self.in_dim, self.out_dim).to(self._device) if self.is_norm_linear else None

        self.processor = Subcontext(self.in_dim, self.out_shape, config=config).to(self._device)

        try:
            bucket = int(getattr(config, "length_bucket_multiple", 64))
        except Exception:
            bucket = 64

        self.controller = Context(
            int(config.d_model),
            int(config.heads),
            depth=max(1, int(getattr(config, "temporal_depth", 1))),
            mlp_ratio=float(getattr(config, "mlp_ratio", 4.0)),
            dropout=float(getattr(config, "dropout", 0.0)),
            batch_first=True,
            length_bucket_multiple=bucket,
        ).to(self._device)

        # Encoder-stage microbatch.
        self.microbatch: int = 0
        self._auto_microbatch_pending: bool = True

        self._decode_compiled: Optional[nn.Module] = None
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

        compile_requested = compile_mode_arg is not None
        compile_available = callable(getattr(torch, "compile", None))
        compile_enabled = bool(compile_requested and compile_available)
        if compile_requested and not compile_available:
            _LOGGER.warning(
                "torch.compile requested (compile_mode=%r) but torch.compile is unavailable; running eagerly",
                raw_mode,
            )
        compile_dynamic = bool(getattr(config, "compile_dynamic", normalized_mode == "reduce-overhead"))
        compile_cudagraphs = bool(
            getattr(
                config,
                "compile_cudagraphs",
                normalized_mode not in {"reduce-overhead", "max-autotune-no-cudagraphs"},
            )
        )
        compile_kwargs: Dict[str, Any] = {}
        if not compile_cudagraphs:
            compile_kwargs["options"] = {"triton.cudagraphs": False}

        # max-autotune modes are significantly more memory hungry; default to compiling only the small decode head.
        compile_heavy_submodules = bool(
            getattr(
                config,
                "compile_heavy_submodules",
                normalized_mode not in {"max-autotune", "max-autotune-no-cudagraphs"},
            )
        )
        if compile_enabled and (not compile_heavy_submodules) and normalized_mode in {"max-autotune", "max-autotune-no-cudagraphs"}:
            _LOGGER.warning(
                "max-autotune hotfix: compiling only decode head (skip temporal/perception). "
                "Set config.compile_heavy_submodules=True to override."
            )

        compiled_decode = False
        compiled_temporal = False
        compiled_perception = False
        if compile_enabled:
            try:
                _raw_head = self.processor.head
                _decode_mod = _CompiledDecode(self.processor.norm, _raw_head, self.out_shape).to(self._device)
                _compiled = Gradient.compile(
                    _decode_mod,
                    mode=compile_mode_arg,
                    fullgraph=False,
                    dynamic=compile_dynamic,
                    backend="inductor",
                    disable=False,
                    **compile_kwargs,
                )
                self._decode_compiled = _compiled
                compiled_decode = _compiled is not _decode_mod
            except Exception:
                self._decode_compiled = None
                _LOGGER.warning(
                    "torch.compile failed for decode head; continuing without compilation",
                    exc_info=True,
                )
            if getattr(self._device, "type", None) == "cuda":
                empty_device_cache(device=self._device, do_gc=True, min_interval_s=0.0)

            if compile_heavy_submodules:
                try:
                    _orig = self.processor.temporal_net
                    _compiled = Gradient.compile(
                        _orig,
                        mode=compile_mode_arg,
                        fullgraph=False,
                        dynamic=compile_dynamic,
                        backend="inductor",
                        disable=False,
                        **compile_kwargs,
                    )
                    self.processor.temporal_net = _compiled
                    compiled_temporal = _compiled is not _orig
                except Exception:
                    _LOGGER.warning(
                        "torch.compile failed for processor.temporal_net; continuing eagerly",
                        exc_info=True,
                    )
            if getattr(self._device, "type", None) == "cuda":
                empty_device_cache(device=self._device, do_gc=True, min_interval_s=0.0)

            if compile_heavy_submodules:
                try:
                    _orig = self.processor.perception
                    _compiled = Gradient.compile(
                        _orig,
                        mode=compile_mode_arg,
                        fullgraph=False,
                        dynamic=compile_dynamic,
                        backend="inductor",
                        disable=False,
                        **compile_kwargs,
                    )
                    self.processor.perception = _compiled
                    compiled_perception = _compiled is not _orig
                except Exception:
                    _LOGGER.warning(
                        "torch.compile failed for processor.perception; continuing eagerly",
                        exc_info=True,
                    )
            if getattr(self._device, "type", None) == "cuda":
                empty_device_cache(device=self._device, do_gc=True, min_interval_s=0.0)

        self._compiled_submodules = {
            "decode": bool(compiled_decode),
            "temporal_net": bool(compiled_temporal),
            "perception": bool(compiled_perception),
        }
        self._pad_compiled_microbatch = bool(compiled_decode or compiled_temporal or compiled_perception)

        # AMP negotiation caching (keyed by device + dataset scale stats).
        self._amp_dtype_cache: Dict[Tuple[Any, ...], torch.dtype] = {}
        self._amp_dtype_cache_last_key: Tuple[Any, ...] | None = None
        self._amp_dtype_cache_last_dtype: torch.dtype | None = None
        self._amp_dtype_cache_max = 64
        self._amp_dtype_cache_lock = threading.Lock()
        self.__config = config

    @staticmethod
    def _cast_graph_safe(x: torch.Tensor, device: torch.device, dtype: Optional[torch.dtype]) -> torch.Tensor:
        target_dtype = dtype or x.dtype
        if x.device != device:
            return x.to(device=device, dtype=target_dtype, non_blocking=True)
        if x.dtype != target_dtype:
            return x.to(dtype=target_dtype)
        return x

    def _auto_microbatch(self, features: torch.Tensor | TensorDictBase, device: torch.device) -> int:
        if isinstance(features, TensorDictBase):
            X = None
            with contextlib.suppress(Exception):
                fkey = resolve_feature_key(features)
                X = features.get(fkey, None)
            if X is None:
                # Backward-compatible fallback (should be rare once callers follow the alias contract)
                X = features.get("features", None) or features.get("X", None)
        else:
            X = features
        if not isinstance(X, torch.Tensor):
            return 64
        b = int(X.shape[0] if X.ndim > 0 else 1)
        hard_max = int(env_int("STNET_MICROBATCH_MAX", 64))
        hard_max = max(1, min(hard_max, b))

        per_sample = int(
            env_first_int(
                ("STNET_PER_SAMPLE_MEM_BYTES", "STNET_DEVICE_BYTES_PER_SAMPLE"),
                default=0,
            )
            or 0
        )
        if per_sample <= 0:
            one = X[:1]
            bytes_per_sample = int(one.nelement()) * int(one.element_size())
            per_sample = int(bytes_per_sample * 8)

        stage_div = max(1, int(env_int("STNET_MICROBATCH_STAGE_DIV", 4)))
        per_sample = max(1, int(per_sample // stage_div))

        mb_size = _auto_microbatch(device=device, hard_max=hard_max, per_sample_bytes=per_sample)
        return int(mb_size)

    def forward(
        self,
        features: torch.Tensor | TensorDictBase,
        *args: Any,
        labels_flat: Optional[torch.Tensor] = None,
        net_loss: Optional[nn.Module] = None,
        global_loss: Optional[nn.Module] = None,
        local_loss: Optional[nn.Module] = None,
        loss_weights: Optional[Union[Tuple[float, float], LossWeightPolicy]] = None,
        calibrate_output: bool = True,
        sanitize_nan: bool = True,
        return_loss: Optional[bool] = None,
        return_loss_components: bool = False,
        return_aux: bool = True,
        **kwargs: Any,
    ) -> (
        torch.Tensor
        | Tuple[torch.Tensor, Optional[torch.Tensor]]
        | Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
        | TensorDictBase
    ):
        if not isinstance(calibrate_output, bool):
            raise TypeError("calibrate_output must be a bool")
        if not isinstance(sanitize_nan, bool):
            raise TypeError("sanitize_nan must be a bool")
        if return_loss is not None and not isinstance(return_loss, bool):
            raise TypeError("return_loss must be a bool or None")
        if not isinstance(return_loss_components, bool):
            raise TypeError("return_loss_components must be a bool")
        if not isinstance(return_aux, bool):
            raise TypeError("return_aux must be a bool")

        grad_enabled = torch.is_grad_enabled()
        infer_mode = not grad_enabled
        sanitize_enabled = bool(sanitize_nan)
        sanitize_inplace = bool(sanitize_enabled and infer_mode)

        td_input: TensorDictBase | None = None
        if isinstance(features, TensorDictBase):
            td_input = features
            td_labels_flat = td_input.get("labels_flat", None)

            # Feature / label extraction is case-insensitive and enforces "exactly one"
            # column per role.
            fkey = resolve_feature_key(td_input)
            td_features = td_input.get(fkey, None)
            if td_features is None:
                raise KeyError(f"TensorDict input requires a feature column (got key={fkey!r} but value is None)")
            features = td_features

            if labels_flat is None:
                # Prefer pre-flattened labels if provided.
                if isinstance(td_labels_flat, torch.Tensor):
                    labels_flat = td_labels_flat
                else:
                    # Otherwise, try the alias-based label column and flatten it.
                    lkey = resolve_label_key(td_input, required=False)
                    if lkey is not None:
                        with contextlib.suppress(Exception):
                            raw = td_input.get(lkey, None)
                            if isinstance(raw, torch.Tensor):
                                if raw.ndim == 0:
                                    raw = raw.reshape(1, 1)
                                elif raw.ndim == 1:
                                    raw = raw.reshape(-1, 1)
                                labels_flat = raw.reshape(raw.shape[0], -1)

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

        device = _infer_module_device(self.processor, self._device)

        x_raw = features
        if isinstance(x_raw, torch.Tensor) and x_raw.ndim == 3 and x_raw.shape[1] == 1:
            x_raw = x_raw.reshape(x_raw.shape[0], -1)

        if isinstance(x_raw, torch.Tensor) and x_raw.device != device:
            x_raw = x_raw.to(device=device, non_blocking=True)

        x_scaled = self.scaler.normalize_x(x_raw)
        graph_break()

        meta = None
        try:
            meta = Autocast.coerce_metadata(device)
        except Exception:
            meta = None
            _LOGGER.debug("Autocast.coerce_metadata failed; falling back to fp32", exc_info=True)

        amp_candidates: Tuple[torch.dtype, ...] = ()
        if meta is not None:
            try:
                amp_candidates = tuple(getattr(meta, "float_dtypes", ()))
            except Exception:
                amp_candidates = ()
        if not amp_candidates:
            amp_candidates = (torch.float32,)

        # Cache AMP negotiation keyed by device + scale statistics.
        safety_margin_pow2 = 3
        try:
            safety_margin_pow2 = int(getattr(self.__config, "safety_margin_pow2", 3))
        except Exception:
            safety_margin_pow2 = 3
        safety_margin_pow2 = max(0, min(30, safety_margin_pow2))

        dev_index = int(device.index) if getattr(device, "index", None) is not None else -1
        if meta is None:
            cache_key = (device.type, dev_index, amp_candidates, None, int(safety_margin_pow2))
        else:
            has_scale = bool(getattr(meta, "has_scale", False))
            has_nonfinite = bool(getattr(meta, "has_nonfinite", False))
            max_abs_v = getattr(meta, "scale_max_abs", None)
            min_pos_v = getattr(meta, "scale_min_positive", None)
            try:
                max_abs_f = float(max_abs_v) if max_abs_v is not None else None
            except Exception:
                max_abs_f = None
            try:
                min_pos_f = float(min_pos_v) if min_pos_v is not None else None
            except Exception:
                min_pos_f = None
            underflow_action = getattr(meta, "underflow_action", "")
            cache_key = (
                device.type,
                dev_index,
                amp_candidates,
                bool(has_scale),
                bool(has_nonfinite),
                max_abs_f,
                min_pos_f,
                str(underflow_action),
                int(safety_margin_pow2),
            )

        with self._amp_dtype_cache_lock:
            if self._amp_dtype_cache_last_key == cache_key:
                amp_dtype = self._amp_dtype_cache_last_dtype
            else:
                amp_dtype = self._amp_dtype_cache.get(cache_key)

        if amp_dtype is None:
            negotiated = Autocast.negotiate(
                tuple(amp_candidates),
                fallback=torch.float64,
                logger=_LOGGER,
                context="instance.forward",
                device=device,
                meta=meta,
                decision_key=cache_key,
                safety_margin_pow2=int(safety_margin_pow2),
            )
            with self._amp_dtype_cache_lock:
                amp_dtype = self._amp_dtype_cache.get(cache_key)
                if amp_dtype is None:
                    amp_dtype = negotiated
                    if len(self._amp_dtype_cache) >= int(self._amp_dtype_cache_max):
                        self._amp_dtype_cache.clear()
                    self._amp_dtype_cache[cache_key] = amp_dtype
                self._amp_dtype_cache_last_key = cache_key
                self._amp_dtype_cache_last_dtype = amp_dtype
        else:
            with self._amp_dtype_cache_lock:
                self._amp_dtype_cache_last_key = cache_key
                self._amp_dtype_cache_last_dtype = amp_dtype

        amp_enabled = amp_dtype is not torch.float64

        is_cls_loss = isinstance(net_loss, (nn.CrossEntropyLoss, nn.NLLLoss)) if net_loss is not None else False
        has_any_loss = (net_loss is not None) or (global_loss is not None) or (local_loss is not None)
        has_supervision = labels_flat is not None and has_any_loss
        is_train_path = bool(self.training and grad_enabled and has_supervision)

        _did_unshard_processor = False
        _unshard = getattr(self.processor, "unshard", None)
        _reshard = getattr(self.processor, "reshard", None)
        if callable(_unshard):
            try:
                _unshard(async_op=False)
                _did_unshard_processor = True
            except TypeError:
                try:
                    _unshard()
                    _did_unshard_processor = True
                except Exception:
                    _did_unshard_processor = False
            except Exception:
                _did_unshard_processor = False

        try:
            requested_base = getattr(self, "base_dtype", None) or getattr(self, "_base_dtype", None)
            if requested_base is not None:
                base_dtype = requested_base
            else:
                base_dtype = torch.float32 if amp_enabled else amp_dtype

            if isinstance(x_scaled, torch.Tensor) and x_scaled.device != device:
                x_scaled = x_scaled.to(device=device, non_blocking=True)

            features_t = x_scaled.to(dtype=base_dtype) if x_scaled.dtype != base_dtype else x_scaled
            if features_t.ndim != 2 or features_t.shape[1] != self.in_dim:
                raise ValueError(
                    f"Expected features shaped (B, {self.in_dim}), got {tuple(features_t.shape)}"
                )
            b = int(features_t.shape[0])

            # --- Encoder stage ---
            if self._auto_microbatch_pending:
                try:
                    mb_enc = self._auto_microbatch(features_t, device)
                    self.microbatch = max(1, int(mb_enc))
                except Exception:
                    self.microbatch = max(1, int(getattr(self, "microbatch", 64) or 64))
                self._auto_microbatch_pending = False
            mb = max(1, min(int(b), int(self.microbatch) or int(b)))

            def _encode(inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                with (Autocast.float(device, metadata=meta) if amp_enabled else Autocast.suspend(device)):
                    return self.processor(inp)

            enc_ctx = Gradient.inference(self.processor) if infer_mode else contextlib.nullcontext()
            with enc_ctx:
                tokens, context = cast(
                    Tuple[torch.Tensor, torch.Tensor],
                    _microbatch_prealloc(
                        features_t,
                        mb,
                        _encode,
                        pad_to=int(mb) if self._pad_compiled_microbatch else None,
                        out_dtype=base_dtype,
                        cast_slice=lambda t: self._cast_graph_safe(t, device, base_dtype),
                        stage="encoder",
                    ),
                )

            tokens = _sanitize_tensor(tokens, enabled=sanitize_enabled, inplace=sanitize_inplace)
            context = _sanitize_tensor(context, enabled=sanitize_enabled, inplace=sanitize_inplace)

            if int(tokens.shape[0]) != int(b):
                raise RuntimeError(
                    "Internal error: token batch mismatch after microbatch concat. "
                    f"got={int(tokens.shape[0])}, expected={int(b)}"
                )

            assembled = context.reshape(b, -1)
            if self.is_norm_linear and self.linear_branch is not None:
                bl = self.linear_branch(self._cast_graph_safe(features_t, device, assembled.dtype))
                assembled = assembled + bl

            # Center tokens (optionally detach in training path).
            mean_dtype = torch.float32 if amp_enabled else tokens.dtype
            mean = tokens.mean(dim=1, keepdim=True, dtype=mean_dtype)
            tokens_centered = tokens - mean.to(dtype=tokens.dtype)
            if not tokens_centered.is_contiguous():
                tokens_centered = tokens_centered.contiguous()
            if is_train_path:
                tokens_centered = tokens_centered.detach()

            # --- Controller stage (moved into Context.run) ---
            refined_tokens = self.controller.run(
                tokens_centered,
                device=device,
                meta=meta,
                amp_enabled=amp_enabled,
                auto_microbatch_fn=lambda t: self._auto_microbatch(t, device),
                graph_break_fn=graph_break,
            )
            refined_tokens = _sanitize_tensor(
                refined_tokens,
                enabled=sanitize_enabled,
                inplace=sanitize_inplace,
            )

            ctrl_mb = max(1, min(int(b), int(self.controller.microbatch) or int(b)))

            # --- Decode stage ---
            graph_break()
            processor_ctx = Gradient.inference(self.processor) if infer_mode else contextlib.nullcontext()
            with processor_ctx:
                dc = getattr(self, "_decode_compiled", None)

                def _run_decode_chunk(chunk: torch.Tensor) -> torch.Tensor:
                    with (Autocast.float(device, metadata=meta) if amp_enabled else Autocast.suspend(device)):
                        if dc is not None:
                            return cast(torch.Tensor, dc(chunk))
                        return self.processor.decode(chunk, apply_norm=True)

                residual_context = cast(
                    torch.Tensor,
                    _microbatch_prealloc(
                        refined_tokens,
                        ctrl_mb,
                        _run_decode_chunk,
                        pad_to=(int(ctrl_mb) if (dc is not None and self._pad_compiled_microbatch) else None),
                        stage="decoder",
                    ),
                )

            residual_context = _sanitize_tensor(
                residual_context,
                enabled=sanitize_enabled,
                inplace=sanitize_inplace,
            )
            residual = residual_context.reshape(b, -1)
            if residual.dtype != assembled.dtype:
                residual = residual.to(dtype=assembled.dtype)
            y_hat = assembled + residual
            y_hat = _sanitize_tensor(y_hat, enabled=sanitize_enabled, inplace=sanitize_inplace)

            pred = y_hat.reshape(b, *self.out_shape)

            # --- Losses ---
            loss_val: Optional[torch.Tensor] = None
            top_component: Optional[torch.Tensor] = None
            bottom_component: Optional[torch.Tensor] = None

            # Regression-only z_true (avoid touching scaler for classification).
            z_true: Optional[torch.Tensor] = None
            if labels_flat is not None and not is_cls_loss:
                y_true_raw = labels_flat.to(device=y_hat.device)
                if not y_true_raw.is_floating_point():
                    y_true_raw = y_true_raw.to(dtype=torch.float32)
                z_true = self.scaler.normalize_y(y_true_raw)

            use_global_local = labels_flat is not None and (global_loss is not None or local_loss is not None)
            use_net = labels_flat is not None and (net_loss is not None) and (not use_global_local)

            if use_global_local:
                # Weight policy
                if loss_weights is None:
                    weights = (1.0, 0.0)
                elif isinstance(loss_weights, (tuple, list)):
                    seq = list(loss_weights)
                    if len(seq) != 2:
                        raise ValueError("loss_weights requires two values")
                    weights = (float(seq[0]), float(seq[1]))
                else:
                    weights = cast(LossWeightPolicy, loss_weights).weights()

                if z_true is None:
                    raise RuntimeError("Internal error: z_true missing for regression loss path")

                tgt = z_true.to(device=y_hat.device, dtype=y_hat.dtype)
                total = y_hat.new_tensor(0.0, dtype=y_hat.dtype)
                y_bot = assembled.to(device=y_hat.device, dtype=y_hat.dtype)

                if global_loss is not None:
                    use_base_detach = bool(
                        is_train_path and (local_loss is not None) and (float(weights[1]) > 1e-12)
                    )
                    z_top = (
                        _sanitize_tensor(
                            assembled.detach() + residual,
                            enabled=sanitize_enabled,
                            inplace=sanitize_inplace,
                        )
                        if use_base_detach
                        else _sanitize_tensor(
                            y_hat,
                            enabled=sanitize_enabled,
                            inplace=sanitize_inplace,
                        )
                    )
                    top_component = cast(torch.Tensor, global_loss(z_top, z_true))
                    total = total + weights[0] * top_component

                if local_loss is not None:
                    bottom_component = cast(torch.Tensor, local_loss(y_bot, tgt))
                    total = total + weights[1] * bottom_component

                loss_val = total

            elif use_net:
                if is_cls_loss:
                    tgt = labels_flat.to(device=y_hat.device).long()
                    loss_val = cast(torch.Tensor, net_loss(y_hat, tgt))  # type: ignore[misc]
                else:
                    if z_true is None:
                        raise RuntimeError("Internal error: z_true missing for regression net_loss path")
                    tgt = z_true.to(device=y_hat.device, dtype=y_hat.dtype)
                    loss_val = cast(torch.Tensor, net_loss(y_hat, tgt))  # type: ignore[misc]

            if loss_val is not None and not isinstance(loss_val, torch.Tensor):
                loss_val = torch.as_tensor(loss_val, device=y_hat.device, dtype=y_hat.dtype)

            # --- Inference-time calibration (regression only) ---
            if infer_mode and calibrate_output and (not is_cls_loss):
                z_cal = self.scaler.calibrate(y_hat)
                pred = self.scaler.denormalize_y(z_cal).reshape(b, *self.out_shape)

            if td_input is not None:
                if hasattr(td_input, "copy"):
                    out_td = td_input.copy()
                else:
                    try:
                        out_td = td_input.clone(recurse=False)
                    except TypeError:
                        out_td = td_input.clone(False)

                out_td.set("pred", pred)
                if return_aux:
                    out_td.set("refined_tokens", refined_tokens)
                    out_td.set("residual_context", residual_context)
                else:
                    with contextlib.suppress(KeyError):
                        out_td.del_("refined_tokens")
                    with contextlib.suppress(KeyError):
                        out_td.del_("residual_context")

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

                if top_component is not None:
                    top_td = top_component
                    if isinstance(top_td, torch.Tensor) and top_td.ndim == 0:
                        batch_size = tuple(out_td.batch_size)
                        if len(batch_size):
                            top_td = top_td.expand(batch_size)
                    out_td.set("loss_top", top_td)
                else:
                    with contextlib.suppress(KeyError):
                        out_td.del_("loss_top")

                if bottom_component is not None:
                    bottom_td = bottom_component
                    if isinstance(bottom_td, torch.Tensor) and bottom_td.ndim == 0:
                        batch_size = tuple(out_td.batch_size)
                        if len(batch_size):
                            bottom_td = bottom_td.expand(batch_size)
                    out_td.set("loss_bottom", bottom_td)
                else:
                    with contextlib.suppress(KeyError):
                        out_td.del_("loss_bottom")

                return out_td

            if return_loss is False:
                return pred
            if return_loss_components:
                return (pred, loss_val, top_component, bottom_component)
            return (pred, loss_val)

        finally:
            if _did_unshard_processor and callable(_reshard):
                with contextlib.suppress(Exception):
                    _reshard()

    def predict(self, features: torch.Tensor | TensorDictBase, *args: Any, **kwargs: Any) -> torch.Tensor | TensorDictBase:
        kwargs.setdefault("return_loss", False)
        return self.forward(features, *args, **kwargs)

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
