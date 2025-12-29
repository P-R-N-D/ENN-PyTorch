# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import logging
import math
import threading
import weakref
from functools import lru_cache
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn

from ..core.config import ModelConfig
from ..core.compat import is_meta_or_fake_tensor
from ..core.system import _log_debug, _log_info, empty_device_cache, get_device
from ..core.casting import env_first_int, env_int
from ..data.pipeline import (
    Dataset,
    resolve_feature_key,
    resolve_label_key,
)
from ..core.profiler import FLOP_PROFILER
from ..core.graph import (
    compile,
    graph_break,
    inference_mode,
    invalidate_model_introspection_caches,
    torch_compile_disable,
)
from ..core.precision import Autocast, is_scale_safe
from .primitives import Recorder, Scaler, ResidualGate
from .blocks import (
    LongNet,
    PointTransformer,
    RetNet,
    TensorDictBase,
    _ControllerChunkRunner,
    _auto_microbatch,
    _coerce_modeling_types,
    _infer_module_device,
    _microbatch_prealloc,
    _sanitize_tensor,
    _serialize_z_index,
    stochastic_depth_schedule,
    CrossTransformer,
    LossWeightPolicy,
    norm_layer,
)

_LOGGER = logging.getLogger(__name__)


class SpatialExtractor(nn.Module):
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
                f"SpatialExtractor coords must be (B,N,C), got shape {tuple(coords.shape)}"
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

            # Compute the expensive argsort only once; odd blocks are a cheap roll.
            perm0, inv0 = _serialize_z_index(
                coords_one,
                bits=bits,
                patch=patch,
                shift_order=shift,
                block_index=0,
            )
            perm0 = perm0[0].to(dtype=torch.int64)
            inv0 = inv0[0].to(dtype=torch.int64)

            shift_amt = 0
            if shift:
                shift_amt = (patch // 2) % max(int(N), 1)

            perm1 = perm0
            inv1 = inv0
            if shift_amt:
                perm1 = torch.roll(perm0, shifts=int(shift_amt), dims=0)
                inv1 = (inv0 + int(shift_amt)) % max(int(N), 1)

            perms: List[torch.Tensor] = []
            invperms: List[torch.Tensor] = []
            for i in range(depth):
                if shift and shift_amt and (i % 2 == 1):
                    perms.append(perm1)
                    invperms.append(inv1)
                else:
                    perms.append(perm0)
                    invperms.append(inv0)

            perm_new = torch.stack(perms, dim=0)
            inv_new = torch.stack(invperms, dim=0)

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

        # Compute the expensive argsort only once; odd blocks are a cheap roll.
        perm0, inv0 = _serialize_z_index(
            coords_batch,
            bits=bits,
            patch=patch,
            shift_order=shift,
            block_index=0,
        )
        perm0 = perm0.to(dtype=torch.int64)
        inv0 = inv0.to(dtype=torch.int64)

        shift_amt = 0
        if shift:
            shift_amt = (patch // 2) % max(int(N), 1)

        perm1 = perm0
        inv1 = inv0
        if shift_amt:
            perm1 = torch.roll(perm0, shifts=int(shift_amt), dims=1)
            inv1 = (inv0 + int(shift_amt)) % max(int(N), 1)

        for i in range(depth):
            if shift and shift_amt and (i % 2 == 1):
                perms_b.append(perm1)
                invperms_b.append(inv1)
            else:
                perms_b.append(perm0)
                invperms_b.append(inv0)
        perm_b = torch.stack(perms_b, dim=0)
        inv_b = torch.stack(invperms_b, dim=0)
        return perm_b, inv_b

    @torch_compile_disable(reason="SpatialExtractor uses dynamic gather/scatter", recursive=True)
    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if is_meta_or_fake_tensor(x):
            raise RuntimeError("x is meta/fake before SpatialExtractor.forward")
        if is_meta_or_fake_tensor(coords):
            raise RuntimeError("coords is meta/fake before SpatialExtractor.forward")
        if x.dim() != 3:
            raise ValueError(
                f"SpatialExtractor expects (B, N, C) tokens, got shape {tuple(x.shape)}"
            )
        if coords.dim() != 3:
            raise ValueError(
                f"SpatialExtractor expects (B, N, D) coords, got shape {tuple(coords.shape)}"
            )
        if x.shape[:2] != coords.shape[:2]:
            raise ValueError(
                "tokens/coords batch or length mismatch: "
                f"tokens={tuple(x.shape)} vs coords={tuple(coords.shape)}"
            )
        x = x.contiguous()

        # Preserve expanded/shared coordinate templates (stride(0)==0) to keep caching effective.
        B, N, D = x.shape
        Bc = int(coords.size(0))
        if Bc != B:
            raise ValueError(
                f"coords batch mismatch: tokens batch={B} vs coords batch={Bc}"
            )

        if B > 1 and coords.stride(0) == 0:
            coords0 = coords[0].contiguous()
            coords = coords0.unsqueeze(0).expand(B, -1, -1)
        else:
            coords = coords.contiguous()

        if coords.dtype != torch.float32:
            coords = coords.float()

        # Pre-validate / normalize attn_mask once (avoid repeating dim checks per block).
        # We support two forms:
        #   - scalar (0D) mask: forwarded as-is
        #   - (B,N,N) or (B,1,N,N): block-sliced internally according to permutation
        mask_scalar: torch.Tensor | None = None
        mask_base: torch.Tensor | None = None
        if attn_mask is not None:
            if is_meta_or_fake_tensor(attn_mask):
                raise RuntimeError("attn_mask is meta/fake before SpatialExtractor.forward")

            # A 0D mask is cheap to move; full masks should already be on-device.
            if attn_mask.device != x.device:
                if int(attn_mask.dim()) == 0:
                    attn_mask = attn_mask.to(device=x.device)
                else:
                    raise ValueError(
                        "attn_mask must be on the same device as tokens: "
                        f"attn_mask.device={attn_mask.device} vs x.device={x.device}"
                    )
            attn_mask = attn_mask.contiguous()
            match int(attn_mask.dim()):
                case 0:
                    mask_scalar = attn_mask
                case 3:
                    if int(attn_mask.size(0)) != int(B):
                        raise ValueError(
                            f"attn_mask batch mismatch: expected B={B}, got {int(attn_mask.size(0))}"
                        )
                    if attn_mask.shape[-2:] != (N, N):
                        raise ValueError(
                            f"attn_mask last 2 dims must be (N,N)=({N},{N}), got {tuple(attn_mask.shape)}"
                        )
                    mask_base = attn_mask
                case 4:
                    if int(attn_mask.size(0)) != int(B):
                        raise ValueError(
                            f"attn_mask batch mismatch: expected B={B}, got {int(attn_mask.size(0))}"
                        )
                    if attn_mask.shape[-2:] != (N, N):
                        raise ValueError(
                            f"attn_mask last 2 dims must be (N,N)=({N},{N}), got {tuple(attn_mask.shape)}"
                        )
                    if attn_mask.size(1) != 1:
                        raise ValueError("attn_mask with per-head shape not supported here")
                    mask_base = attn_mask.squeeze(1)
                case _:
                    raise ValueError("attn_mask must be rank 0, 3, or 4 here")

        b_index: torch.Tensor | None = None
        if mask_base is not None:
            b_index = torch.arange(B, device=mask_base.device)

        perm_cache, inv_cache = self._ensure_permutation_cache(coords)
        patch = int(self.patch_size)
        full = (N // patch) * patch
        nblk = (full // patch) if full > 0 else 0
        coord_dim = int(coords.size(-1))

        # If shift_order is enabled, only two permutation templates are used: even/odd blocks.
        shift = bool(self.shift_order)
        shift_amt = 0
        if shift:
            shift_amt = (patch // 2) % max(int(N), 1)

        # Permutations are either shared (depth, N) or per-sample (depth, B, N).
        shared_perm = int(perm_cache.dim()) == 2

        # Parity templates (even/odd) to avoid repeatedly slicing perm_cache.
        perm_even = perm_cache[0]
        inv_even = inv_cache[0]
        perm_odd = perm_even
        inv_odd = inv_even
        if shift and shift_amt and len(self.blocks) > 1:
            perm_odd = perm_cache[1]
            inv_odd = inv_cache[1]

        # Pre-sliced indices for mask building (per parity). These are views (no new allocations)
        # and help avoid repeatedly materializing expanded index tensors (especially when shared_perm=True).
        idx_full_even: torch.Tensor | None = None
        idx_full_odd: torch.Tensor | None = None
        idx_tail_even: torch.Tensor | None = None
        idx_tail_odd: torch.Tensor | None = None
        tail_len = int(N - full)

        if mask_base is not None:
            if full > 0:
                if shared_perm:
                    idx_full_even = perm_even[:full].reshape(nblk, patch)
                    idx_full_odd = perm_odd[:full].reshape(nblk, patch) if (shift and shift_amt) else idx_full_even
                else:
                    idx_full_even = perm_even[:, :full].reshape(B, nblk, patch)
                    idx_full_odd = perm_odd[:, :full].reshape(B, nblk, patch) if (shift and shift_amt) else idx_full_even
            if tail_len > 0:
                if shared_perm:
                    idx_tail_even = perm_even[full:N]
                    idx_tail_odd = perm_odd[full:N] if (shift and shift_amt) else idx_tail_even
                else:
                    idx_tail_even = perm_even[:, full:N]
                    idx_tail_odd = perm_odd[:, full:N] if (shift and shift_amt) else idx_tail_even

        # Optional per-forward mask cache (per permutation template) to avoid rebuilding mb for every block.
        # Guarded by a conservative size cap to avoid holding 2x huge tensors when shift_order alternates.
        mb_full_even: torch.Tensor | None = None
        mb_full_odd: torch.Tensor | None = None
        mb_tail_even: torch.Tensor | None = None
        mb_tail_odd: torch.Tensor | None = None

        cache_full_masks = False
        cache_tail_masks = False
        if mask_base is not None:
            try:
                elem = int(mask_base.element_size())
            except Exception:
                elem = 1
            if full > 0 and nblk > 0:
                try:
                    one_bytes = int(B) * int(nblk) * int(patch) * int(patch) * elem
                    cache_full_masks = one_bytes <= (32 << 20)  # cache at most ~32MB per parity
                except Exception:
                    cache_full_masks = False
            if tail_len > 0:
                try:
                    tail_bytes = int(B) * int(tail_len) * int(tail_len) * elem
                    cache_tail_masks = tail_bytes <= (16 << 20)  # tail is usually small; still cap it
                except Exception:
                    cache_tail_masks = False

        b_full: torch.Tensor | None = None
        b_tail: torch.Tensor | None = None
        if mask_base is not None:
            base = mask_base
            if b_index is None:
                b_index = torch.arange(B, device=base.device)
            b_full = b_index.view(B, 1, 1, 1)
            b_tail = b_index.view(B, 1, 1)

        for i, blk in enumerate(self.blocks):
            use_odd = bool(shift_amt and (i % 2 == 1))

            if shared_perm:
                perm_1d = perm_odd if use_odd else perm_even
                inv_1d = inv_odd if use_odd else inv_even
                x_s = x.index_select(1, perm_1d)
                c_s = coords.index_select(1, perm_1d)
            else:
                perm_2d = perm_odd if use_odd else perm_even
                inv_2d = inv_odd if use_odd else inv_even
                x_s = x.gather(1, perm_2d.unsqueeze(-1).expand(B, N, D))
                c_s = coords.gather(1, perm_2d.unsqueeze(-1).expand(B, N, coord_dim))

            if torch.is_grad_enabled():
                out_s = torch.empty_like(x_s)
            else:
                # Inference / no-grad: reuse buffer to avoid an extra allocation.
                out_s = x_s

            # Vectorize per-block processing by folding blocks into the batch dimension.
            # This reduces Python overhead while keeping peak memory bounded.
            if full > 0:
                x_full = x_s[:, :full, :].contiguous().view(B, nblk, patch, D)
                c_full = c_s[:, :full, :].contiguous().view(B, nblk, patch, coord_dim)
                x_flat = x_full.reshape(B * nblk, patch, D)
                c_flat = c_full.reshape(B * nblk, patch, coord_dim)

                mb_flat: torch.Tensor | None = None
                if mask_scalar is not None:
                    mb_flat = mask_scalar
                elif mask_base is not None and b_full is not None:
                    cached = None
                    if cache_full_masks:
                        cached = mb_full_odd if use_odd else mb_full_even

                    if cached is None:
                        idx = idx_full_odd if use_odd else idx_full_even
                        if idx is not None:
                            if shared_perm:
                                # idx: (nblk, patch) -> broadcast across B without explicit expand
                                ii = idx.view(1, nblk, patch, 1)
                                jj = idx.view(1, nblk, 1, patch)
                            else:
                                # idx: (B, nblk, patch)
                                ii = idx.unsqueeze(-1)
                                jj = idx.unsqueeze(-2)
                            mb = mask_base[b_full, ii, jj]
                            cached = mb.reshape(B * nblk, patch, patch).contiguous()
                            if cache_full_masks:
                                if use_odd:
                                    mb_full_odd = cached
                                else:
                                    mb_full_even = cached
                    mb_flat = cached

                y_flat = blk(x_flat, c_flat, attn_mask=mb_flat)
                out_s[:, :full, :].copy_(y_flat.reshape(B, nblk, patch, D).reshape(B, full, D))

            # Tail (non-full block)
            if full < N:
                s, e = full, N
                xb = x_s[:, s:e, :]
                cb = c_s[:, s:e, :]

                mb: torch.Tensor | None = None
                if mask_scalar is not None:
                    mb = mask_scalar
                elif mask_base is not None and b_tail is not None:
                    cached_tail = None
                    if cache_tail_masks:
                        cached_tail = mb_tail_odd if use_odd else mb_tail_even

                    if cached_tail is None:
                        idx2 = idx_tail_odd if use_odd else idx_tail_even
                        if idx2 is not None:
                            if shared_perm:
                                M = int(idx2.shape[0])
                                ii = idx2.view(1, M, 1)
                                jj = idx2.view(1, 1, M)
                            else:
                                M = int(idx2.shape[1])
                                ii = idx2.unsqueeze(-1)
                                jj = idx2.unsqueeze(-2)
                            cached_tail = mask_base[b_tail, ii, jj].reshape(B, M, M).contiguous()
                            if cache_tail_masks:
                                if use_odd:
                                    mb_tail_odd = cached_tail
                                else:
                                    mb_tail_even = cached_tail
                    mb = cached_tail

                out_s[:, s:e, :] = blk(xb, cb, attn_mask=mb)

            if shared_perm:
                x = out_s.index_select(1, inv_1d)
            else:
                x = out_s.gather(1, inv_2d.unsqueeze(-1).expand(B, N, D))

        out = self.norm(x)
        if is_meta_or_fake_tensor(out):
            raise RuntimeError("SpatialExtractor produced meta/fake tensor")
        return out.contiguous()


class TemporalExtractor(nn.Module):
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


class Enhancer(nn.Module):
    """Controller backbone wrapper.

    Originally a thin wrapper around LongNet. This module now also owns
    controller-stage microbatch sizing/execution, so Model.forward doesn't
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
                f"Enhancer.run expects tokens (B,N,D), got shape {tuple(tokens.shape)}"
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
            inference_mode(self.backbone) if infer_mode else contextlib.nullcontext()
        )

        runner = _ControllerChunkRunner(
            self.backbone,
            device=device,
            meta=meta,
            amp_enabled=amp_enabled,
        )
        with controller_ctx:
            refined = cast(
                torch.Tensor,
                _microbatch_prealloc(tokens, mb, runner, stage="controller"),
            )
        return refined


class Fuser(nn.Module):
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

        self.spatial_net = SpatialExtractor(
            self.d_model,
            self.nhead,
            depth=max(1, int(config.spatial_depth)),
            coord_dim=self.spatial_coords_template.shape[-1],
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            drop_path=self.drop_path,
            norm_type=self.norm_type,
        )
        self.temporal_net = TemporalExtractor(
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
        n = int(n_tokens)
        if side <= 1 or n <= 0:
            return torch.zeros((max(0, n), 3), dtype=torch.float32, device=device)

        idx = torch.arange(n, device=device, dtype=torch.long)
        side2 = int(side) * int(side)
        z = idx // side2
        rem = idx % side2
        y = rem // int(side)
        x = rem % int(side)
        coords = torch.stack((x, y, z), dim=1).to(dtype=torch.float32)
        coords = coords / float(side - 1)
        return coords

    def _spatial_coords(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        _ = dtype
        # Spatial coords are constants; keep float32 to avoid autocast-induced dtype churn.
        coords = self.spatial_coords_template
        if coords.device != device or coords.dtype is not torch.float32:
            coords = coords.to(device=device, dtype=torch.float32)
        return coords.unsqueeze(0).expand(int(batch), -1, -1)

    @torch_compile_disable(reason="Fuser orchestrates eager + compiled submodules", recursive=False)
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


class Model(nn.Module):
    def __init__(self, in_dim: int, out_shape: Sequence[int], config: ModelConfig) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_shape = tuple((int(x) for x in out_shape))
        self.out_dim = int(math.prod(self.out_shape))

        if config.device is not None:
            self._device = torch.device(config.device)
            # Keep "current device" consistent for backends that use implicit device context.
            if self._device.type in {"cuda", "xpu"} and self._device.index is not None:
                with contextlib.suppress(Exception):
                    from ..core.system import accel_set_device_index

                    accel_set_device_index(self._device.type, int(self._device.index))
        else:
            # Centralize backend preference logic (CUDA > XPU > MPS > Vulkan > CPU).
            self._device = get_device()

        self.scaler = Scaler().to(self._device)

        # Avoid shape mutation inside forward(): ensure x stats match feature dimension up front.
        with torch.no_grad():
            if self.scaler.x_mean.numel() != self.in_dim:
                self.scaler.x_mean.resize_(self.in_dim)
                self.scaler.x_std.resize_(self.in_dim)
                self.scaler.x_mean.zero_()
                self.scaler.x_std.fill_(1.0)

        # Recorder/logging is runtime metadata; keep it on CPU to reduce device
        # traffic and GPU memory pressure.
        self.logger = Recorder()

        self.is_norm_linear = bool(getattr(config, "use_linear_branch", False))
        self.linear_branch = nn.Linear(self.in_dim, self.out_dim).to(self._device) if self.is_norm_linear else None

        self.processor = Fuser(self.in_dim, self.out_shape, config=config).to(self._device)

        try:
            bucket = int(getattr(config, "length_bucket_multiple", 64))
        except Exception:
            bucket = 64

        self.controller = Enhancer(
            int(config.d_model),
            int(config.heads),
            depth=max(1, int(getattr(config, "temporal_depth", 1))),
            mlp_ratio=float(getattr(config, "mlp_ratio", 4.0)),
            dropout=float(getattr(config, "dropout", 0.0)),
            batch_first=True,
            length_bucket_multiple=bucket,
        ).to(self._device)

        # Residual gating: z_hat = base + p * residue
        self.p_gate: Optional[ResidualGate]
        # Fallback bounds: when scaler.y_min/y_max are unavailable, use mean ± k*std.
        # `fallback_k` is stored as a buffer so it can be checkpointed and (optionally)
        # tuned during training.
        k_default = float(getattr(config, 'p_gate_fallback_k', 6.0))
        k_low_cfg = getattr(config, 'p_gate_fallback_k_low', None)
        k_high_cfg = getattr(config, 'p_gate_fallback_k_high', None)
        # Keep the legacy symmetric k for compatibility; allow asymmetric overrides.
        self.p_gate_fallback_k: float = float(k_default)
        self.p_gate_fallback_k_low: float = float(k_default if k_low_cfg is None else float(k_low_cfg))
        self.p_gate_fallback_k_high: float = float(k_default if k_high_cfg is None else float(k_high_cfg))
        self.p_gate_auto_k_enabled: bool = bool(getattr(config, 'p_gate_auto_k_enabled', False))
        self.p_gate_auto_k_interval: int = int(getattr(config, 'p_gate_auto_k_interval', 100) or 0)
        self.p_gate_auto_k_warmup: int = int(getattr(config, 'p_gate_auto_k_warmup', 0) or 0)
        self.p_gate_auto_k_ema_alpha: float = float(getattr(config, 'p_gate_auto_k_ema_alpha', 0.1))
        self.p_gate_auto_k_target_tight: float = float(getattr(config, 'p_gate_auto_k_target_tight', 0.02))
        self.p_gate_auto_k_tolerance: float = float(getattr(config, 'p_gate_auto_k_tolerance', 0.5))
        self.p_gate_auto_k_step_up: float = float(getattr(config, 'p_gate_auto_k_step_up', 0.1))
        self.p_gate_auto_k_step_down: float = float(getattr(config, 'p_gate_auto_k_step_down', 0.02))
        # Optional per-side step sizes (default to symmetric step_up/step_down).
        self.p_gate_auto_k_step_up_low: float = float(getattr(config, 'p_gate_auto_k_step_up_low', self.p_gate_auto_k_step_up))
        self.p_gate_auto_k_step_down_low: float = float(getattr(config, 'p_gate_auto_k_step_down_low', self.p_gate_auto_k_step_down))
        self.p_gate_auto_k_step_up_high: float = float(getattr(config, 'p_gate_auto_k_step_up_high', self.p_gate_auto_k_step_up))
        self.p_gate_auto_k_step_down_high: float = float(getattr(config, 'p_gate_auto_k_step_down_high', self.p_gate_auto_k_step_down))

        # Optional edge-based tuning: when constraint activation is in-range but p frequently
        # hugs dynamic endpoints, shrink fallback bounds by reducing k.
        self.p_gate_auto_k_edge_enabled: bool = bool(getattr(config, 'p_gate_auto_k_edge_enabled', False))
        self.p_gate_auto_k_target_edge: float = float(getattr(config, 'p_gate_auto_k_target_edge', 0.05))
        self.p_gate_auto_k_edge_tolerance: float = float(getattr(config, 'p_gate_auto_k_edge_tolerance', 0.5))
        self.p_gate_auto_k_edge_ema_alpha: float = float(getattr(config, 'p_gate_auto_k_edge_ema_alpha', self.p_gate_auto_k_ema_alpha))
        self.p_gate_auto_k_edge_step_down_low: float = float(getattr(config, 'p_gate_auto_k_edge_step_down_low', 0.01))
        self.p_gate_auto_k_edge_step_down_high: float = float(getattr(config, 'p_gate_auto_k_edge_step_down_high', 0.01))
        self.p_gate_auto_k_min: float = float(getattr(config, 'p_gate_auto_k_min', 1.0))
        self.p_gate_auto_k_max: float = float(getattr(config, 'p_gate_auto_k_max', 16.0))
        self.p_gate_auto_k_width_frac: float = float(getattr(config, 'p_gate_auto_k_width_frac', 0.05))
        self.p_gate_auto_k_edge_frac: float = float(getattr(config, 'p_gate_auto_k_edge_frac', 0.02))
        self.p_gate_auto_k_log_interval: int = int(getattr(config, 'p_gate_auto_k_log_interval', 200) or 0)
        # If auto-k is enabled, ensure fallback ks are positive (otherwise we'd disable fallback entirely).
        if self.p_gate_auto_k_enabled:
            if self.p_gate_fallback_k_low <= 0.0:
                self.p_gate_fallback_k_low = max(float(self.p_gate_auto_k_min), 1e-6)
            if self.p_gate_fallback_k_high <= 0.0:
                self.p_gate_fallback_k_high = max(float(self.p_gate_auto_k_min), 1e-6)
        self.p_gate_fallback_enabled: bool = bool(
            self.p_gate_fallback_k_low > 0.0 and self.p_gate_fallback_k_high > 0.0
        )

        # Tile-wise p gating (optional). When set, p is predicted per output tile
        # along the flattened y dimension; set this to the same value as ops.loss_tile_size
        # if you want tight coupling with TiledLoss slicing.
        self.p_gate_tile_size: Optional[int] = getattr(config, 'p_gate_tile_size', None)

        # Optional quantile bounds for p tightening (requires scaler.y_q_low/y_q_high).
        self.p_gate_bounds_use_quantile: bool = bool(getattr(config, 'p_gate_bounds_use_quantile', False))
        self.p_gate_bounds_q_low: float = float(getattr(config, 'p_gate_bounds_q_low', 0.005))
        self.p_gate_bounds_q_high: float = float(getattr(config, 'p_gate_bounds_q_high', 0.995))
        self.p_gate_bounds_q_max_samples: int = int(getattr(config, 'p_gate_bounds_q_max_samples', 8192) or 0)
        self.p_gate_bounds_clip_to_minmax: bool = bool(getattr(config, 'p_gate_bounds_clip_to_minmax', True))

        try:
            # Asymmetric fallback ks (lower/upper) for mean±kσ bounds.
            self.register_buffer(
                "p_gate_fallback_k_low_buf",
                torch.tensor(float(self.p_gate_fallback_k_low), dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "p_gate_fallback_k_high_buf",
                torch.tensor(float(self.p_gate_fallback_k_high), dtype=torch.float32),
                persistent=True,
            )
            # Legacy: keep an average as a convenience for older tooling.
            self.register_buffer(
                "p_gate_fallback_k_buf",
                torch.tensor(
                    float(0.5 * (self.p_gate_fallback_k_low + self.p_gate_fallback_k_high)),
                    dtype=torch.float32,
                ),
                persistent=True,
            )
            self.register_buffer(
                "p_gate_auto_k_step_buf",
                torch.tensor(0, dtype=torch.int64),
                persistent=True,
            )
            # Per-side EMAs for constraint activation.
            self.register_buffer(
                "p_gate_auto_k_ema_low_buf",
                torch.tensor(0.0, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "p_gate_auto_k_ema_high_buf",
                torch.tensor(0.0, dtype=torch.float32),
                persistent=True,
            )
            # Per-side EMAs for boundary hugging (edge-based tuning/diagnostics).
            self.register_buffer(
                "p_gate_auto_k_edge_ema_low_buf",
                torch.tensor(0.0, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "p_gate_auto_k_edge_ema_high_buf",
                torch.tensor(0.0, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "p_gate_auto_k_edge_ema_buf",
                torch.tensor(0.0, dtype=torch.float32),
                persistent=True,
            )
            # Overall EMA retained for backwards compatibility/logging.
            self.register_buffer(
                "p_gate_auto_k_ema_buf",
                torch.tensor(0.0, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "p_gate_auto_k_updates_buf",
                torch.tensor(0, dtype=torch.int64),
                persistent=True,
            )
        except Exception:
            pass
        if bool(getattr(config, 'p_gate_enabled', False)):
            self.p_gate = ResidualGate(
                d_model=int(config.d_model),
                hidden_dim=int(getattr(config, 'p_gate_hidden_dim', 64)),
                detach_inputs=bool(getattr(config, 'p_gate_detach_inputs', True)),
                p_floor=float(getattr(config, 'p_gate_p_floor', 0.0)),
                p_ceil=float(getattr(config, 'p_gate_p_ceil', 1.0)),
                tile_size=getattr(config, 'p_gate_tile_size', None),
                clip_eps=float(getattr(config, 'p_gate_clip_eps', 1e-6)),
                eps=float(getattr(config, 'p_gate_eps', 1e-6)),
                stat_width_frac=float(getattr(config, 'p_gate_auto_k_width_frac', 0.05)),
                stat_edge_frac=float(getattr(config, 'p_gate_auto_k_edge_frac', 0.02)),
            ).to(self._device)
        else:
            self.p_gate = None

        # Auxiliary losses (added to total loss inside Model.forward)
        self.unsup_xx_weight = float(getattr(config, 'unsup_xx_weight', 0.0))
        self.unsup_yy_weight = float(getattr(config, 'unsup_yy_weight', 0.0))
        self.p_prior_weight = float(getattr(config, 'p_prior_weight', 0.0))
        self.p_prior_alpha = float(getattr(config, 'p_prior_alpha', 2.0))
        self.p_prior_beta = float(getattr(config, 'p_prior_beta', 2.0))
        # Optional p edge-hugging regularizer (added to total loss in forward).
        self.p_gate_edge_reg_weight = float(getattr(config, 'p_gate_edge_reg_weight', 0.0))
        self.p_gate_edge_reg_frac = float(
            getattr(config, 'p_gate_edge_reg_frac', getattr(config, 'p_gate_auto_k_edge_frac', 0.02))
        )
        self.p_gate_edge_reg_min_width_frac = float(
            getattr(config, 'p_gate_edge_reg_min_width_frac', getattr(config, 'p_gate_auto_k_width_frac', 0.05))
        )
        self.p_gate_edge_reg_power = float(getattr(config, 'p_gate_edge_reg_power', 2.0))
        # Optional per-side weights and fallback-only mode for edge regularizer.
        try:
            w_low_cfg = getattr(config, 'p_gate_edge_reg_weight_low', None)
            w_high_cfg = getattr(config, 'p_gate_edge_reg_weight_high', None)
            self.p_gate_edge_reg_weight_low = (
                float(self.p_gate_edge_reg_weight) if w_low_cfg is None else float(w_low_cfg)
            )
            self.p_gate_edge_reg_weight_high = (
                float(self.p_gate_edge_reg_weight) if w_high_cfg is None else float(w_high_cfg)
            )
        except Exception:
            self.p_gate_edge_reg_weight_low = float(self.p_gate_edge_reg_weight)
            self.p_gate_edge_reg_weight_high = float(self.p_gate_edge_reg_weight)
        self.p_gate_edge_reg_fallback_only = bool(getattr(config, 'p_gate_edge_reg_fallback_only', False))



        self.x_recon_head: Optional[nn.Module]
        if self.unsup_xx_weight > 0.0:
            hid = max(8, int(int(config.d_model) // 2))
            self.x_recon_head = nn.Sequential(
                nn.LayerNorm(int(config.d_model)),
                nn.Linear(int(config.d_model), hid),
                nn.SiLU(),
                nn.Linear(hid, int(in_dim)),
            ).to(self._device)
        else:
            self.x_recon_head = None

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
        mode_raw = str(raw_mode or "").strip().lower()

        compile_mode_canonical = mode_raw.replace("_", "-").replace(" ", "-")
        if "-" in compile_mode_canonical:
            compile_mode_canonical = "-".join(
                part for part in compile_mode_canonical.split("-") if part
            )

        # Match stnet.core.graph.compile mode normalization.
        mode_compact = compile_mode_canonical.replace("-", "")
        match compile_mode_canonical:
            case "" | "none" | "disabled" | "disable" | "off" | "false" | "0":
                compile_mode_canonical = "disabled"
            case (
                "default"
                | "reduce-overhead"
                | "max-autotune"
                | "max-autotune-no-cudagraphs"
                | "aot-eager"
            ):
                pass
            case _:
                match mode_compact:
                    case "reduceoverhead":
                        compile_mode_canonical = "reduce-overhead"
                    case "maxautotune":
                        compile_mode_canonical = "max-autotune"
                    case "maxautotunenocudagraphs" | "maxautotunenocudagraph":
                        compile_mode_canonical = "max-autotune-no-cudagraphs"
                    case "aoteager":
                        compile_mode_canonical = "aot-eager"
                    case _:
                        pass

        compile_mode_arg = None if compile_mode_canonical == "disabled" else compile_mode_canonical

        compile_requested = compile_mode_arg is not None
        compile_available = callable(getattr(torch, "compile", None))
        compile_enabled = bool(compile_requested and compile_available)
        if compile_requested and not compile_available:
            _LOGGER.warning(
                "torch.compile requested (compile_mode=%r) but torch.compile is unavailable; running eagerly",
                raw_mode,
            )
        compile_dynamic = bool(
            getattr(config, "compile_dynamic", compile_mode_canonical == "reduce-overhead")
        )
        compile_cudagraphs_default = compile_mode_canonical not in {
            "reduce-overhead",
            "max-autotune-no-cudagraphs",
        }
        compile_cudagraphs = bool(
            getattr(config, "compile_cudagraphs", compile_cudagraphs_default)
        )
        compile_kwargs: Dict[str, Any] = {}
        if not compile_cudagraphs:
            compile_kwargs["options"] = {"triton.cudagraphs": False}

        # max-autotune modes are significantly more memory hungry; default to compiling only the small decode head.
        compile_heavy_submodules_default = compile_mode_canonical not in {
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }
        compile_heavy_submodules = bool(
            getattr(config, "compile_heavy_submodules", compile_heavy_submodules_default)
        )
        if (
            compile_enabled
            and (not compile_heavy_submodules)
            and compile_mode_canonical in {"max-autotune", "max-autotune-no-cudagraphs"}
        ):
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
                _compiled = compile(
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
                    _compiled = compile(
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
                    _compiled = compile(
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

        # Keep a stable, non-mangled reference for runtime tools (save/load, launchers).
        # Do not rely on name-mangled attributes outside of this class.
        self.__config = config
        self.__stnet_instance_config__ = config

    @property
    def config(self) -> ModelConfig:
        """The :class:`~stnet.core.config.ModelConfig` used to construct this instance."""
        return self.__config

    def to(self, *args: Any, **kwargs: Any) -> "Model":
        """Move the model to a device/dtype while keeping Recorder on CPU.

        Model owns device placement for the core model, but Recorder is runtime
        metadata/logging and should remain on CPU (especially important for
        CUDA/XPU where moving it would allocate device memory and can trigger
        unnecessary device syncs).
        """
        out = super().to(*args, **kwargs)
        with contextlib.suppress(Exception):
            if isinstance(getattr(self, "logger", None), Recorder):
                self.logger.cpu()
        return cast(Model, out)

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

            enc_ctx = inference_mode(self.processor) if infer_mode else contextlib.nullcontext()
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

            # --- Controller stage (moved into Enhancer.run) ---
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
            processor_ctx = inference_mode(self.processor) if infer_mode else contextlib.nullcontext()
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

            p: Optional[torch.Tensor] = None
            edge_reg_low: Optional[torch.Tensor] = None
            edge_reg_high: Optional[torch.Tensor] = None
            if self.p_gate is not None:
                z_min: Optional[torch.Tensor] = None
                z_max: Optional[torch.Tensor] = None
                fallback_bounds = False
                try:
                    y_min = getattr(self.scaler, 'y_min', None)
                    y_max = getattr(self.scaler, 'y_max', None)
                    y_q_low = getattr(self.scaler, 'y_q_low', None)
                    y_q_high = getattr(self.scaler, 'y_q_high', None)

                    mean = self.scaler.y_mean.to(device=assembled.device, dtype=assembled.dtype)
                    std = self.scaler.y_std.to(device=assembled.device, dtype=assembled.dtype)
                    denom = std + float(self.scaler.eps)

                    def _finite_pair(lo: object, hi: object) -> bool:
                        if not (isinstance(lo, torch.Tensor) and isinstance(hi, torch.Tensor)):
                            return False
                        try:
                            return bool(torch.isfinite(lo).all().item()) and bool(torch.isfinite(hi).all().item())
                        except Exception:
                            return False

                    have_minmax = _finite_pair(y_min, y_max)
                    have_quant = _finite_pair(y_q_low, y_q_high)

                    use_quant = bool(getattr(self, 'p_gate_bounds_use_quantile', False))
                    clip_quant = bool(getattr(self, 'p_gate_bounds_clip_to_minmax', True))

                    ylo_t: Optional[torch.Tensor] = None
                    yhi_t: Optional[torch.Tensor] = None

                    if use_quant and have_quant:
                        ylo_t = cast(torch.Tensor, y_q_low).to(device=assembled.device, dtype=assembled.dtype)
                        yhi_t = cast(torch.Tensor, y_q_high).to(device=assembled.device, dtype=assembled.dtype)
                        # Optionally clip quantile bounds within min/max bounds when available.
                        if clip_quant and have_minmax:
                            ymin_t = cast(torch.Tensor, y_min).to(device=assembled.device, dtype=assembled.dtype)
                            ymax_t = cast(torch.Tensor, y_max).to(device=assembled.device, dtype=assembled.dtype)
                            ylo_t = torch.maximum(ylo_t, ymin_t)
                            yhi_t = torch.minimum(yhi_t, ymax_t)
                    elif have_minmax:
                        ylo_t = cast(torch.Tensor, y_min).to(device=assembled.device, dtype=assembled.dtype)
                        yhi_t = cast(torch.Tensor, y_max).to(device=assembled.device, dtype=assembled.dtype)

                    if ylo_t is not None and yhi_t is not None:
                        z_min = (ylo_t - mean) / denom
                        z_max = (yhi_t - mean) / denom
                    else:
                        # Legacy/unknown bounds: fall back to mean ± k * std.
                        # This corresponds to z in [-k*std/(std+eps), +k*std/(std+eps)].
                        if bool(getattr(self, 'p_gate_fallback_enabled', False)):
                            k_low_buf = getattr(self, 'p_gate_fallback_k_low_buf', None)
                            k_high_buf = getattr(self, 'p_gate_fallback_k_high_buf', None)
                            if isinstance(k_low_buf, torch.Tensor):
                                k_low = k_low_buf.to(device=assembled.device, dtype=assembled.dtype)
                            else:
                                k_low = mean.new_tensor(float(getattr(self, 'p_gate_fallback_k_low', getattr(self, 'p_gate_fallback_k', 0.0))))
                            if isinstance(k_high_buf, torch.Tensor):
                                k_high = k_high_buf.to(device=assembled.device, dtype=assembled.dtype)
                            else:
                                k_high = mean.new_tensor(float(getattr(self, 'p_gate_fallback_k_high', getattr(self, 'p_gate_fallback_k', 0.0))))
                            z_scale = std / denom
                            z_min = (-k_low) * z_scale
                            z_max = (k_high) * z_scale
                            fallback_bounds = True
                except Exception:
                    z_min = None
                    z_max = None
                do_edge_reg = (
                    self.training
                    and grad_enabled
                    and ((self.p_gate_edge_reg_weight_low > 0.0) or (self.p_gate_edge_reg_weight_high > 0.0))
                    and (z_min is not None)
                    and (z_max is not None)
                    and ((not self.p_gate_edge_reg_fallback_only) or bool(fallback_bounds))
                )
                if do_edge_reg:
                    p, edge_reg_low, edge_reg_high = self.p_gate(
                        tokens=tokens,
                        refined_tokens=refined_tokens,
                        base=assembled,
                        residue=residual,
                        z_min=z_min,
                        z_max=z_max,
                        fallback_bounds=bool(fallback_bounds),
                        return_edge_reg_lr=True,
                        edge_reg_frac=float(self.p_gate_edge_reg_frac),
                        edge_reg_min_width_frac=float(self.p_gate_edge_reg_min_width_frac),
                        edge_reg_power=float(self.p_gate_edge_reg_power),
                    )
                else:
                    p = self.p_gate(
                        tokens=tokens,
                        refined_tokens=refined_tokens,
                        base=assembled,
                        residue=residual,
                        z_min=z_min,
                        z_max=z_max,
                        fallback_bounds=bool(fallback_bounds),
                    )

            y_hat = assembled + (residual if p is None else p * residual)
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
                            assembled.detach() + (residual if p is None else p * residual),
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

            # --- Auxiliary losses (unsupervised / regularization) ---
            if self.training and grad_enabled:
                aux_total = y_hat.new_tensor(0.0, dtype=y_hat.dtype)
                aux_used = False

                if self.unsup_xx_weight > 0.0 and self.x_recon_head is not None:
                    x_recon = self.x_recon_head(tokens.mean(dim=1))
                    loss_xx = F.smooth_l1_loss(
                        x_recon.to(dtype=torch.float32),
                        features_t.to(dtype=torch.float32),
                        reduction="mean",
                    )
                    aux_total = aux_total + self.unsup_xx_weight * loss_xx.to(dtype=aux_total.dtype)
                    aux_used = True

                if self.unsup_yy_weight > 0.0:
                    teacher = y_hat.detach()
                    student = assembled
                    if p is not None:
                        w = p.detach()
                        # p can be (B, 1) (scalar gate) or (B, D) (tile-wise-expanded).
                        if w.dim() == 2 and int(w.shape[1]) != 1:
                            w = w.mean(dim=1)
                        else:
                            w = w.squeeze(-1)
                        w = w.clamp(min=0.0)
                        per = F.smooth_l1_loss(student, teacher, reduction="none").mean(dim=1)
                        loss_yy = (per * w).mean()
                    else:
                        loss_yy = F.smooth_l1_loss(student, teacher, reduction="mean")
                    aux_total = aux_total + self.unsup_yy_weight * loss_yy.to(dtype=aux_total.dtype)
                    aux_used = True

                if self.p_prior_weight > 0.0 and p is not None:
                    # Beta prior on p (encourage non-extreme gating); supports skew when alpha!=beta.
                    clip_eps = float(getattr(self.p_gate, "clip_eps", 1e-6)) if self.p_gate is not None else 1e-6
                    p01 = p.squeeze(-1).clamp(min=clip_eps, max=1.0 - clip_eps)
                    a = float(self.p_prior_alpha)
                    b = float(self.p_prior_beta)
                    loss_p = -(((a - 1.0) * torch.log(p01)) + ((b - 1.0) * torch.log1p(-p01))).mean()
                    aux_total = aux_total + self.p_prior_weight * loss_p.to(dtype=aux_total.dtype)
                    aux_used = True

                if edge_reg_low is not None and self.p_gate_edge_reg_weight_low > 0.0:
                    aux_total = aux_total + self.p_gate_edge_reg_weight_low * edge_reg_low.to(dtype=aux_total.dtype)
                    aux_used = True

                if edge_reg_high is not None and self.p_gate_edge_reg_weight_high > 0.0:
                    aux_total = aux_total + self.p_gate_edge_reg_weight_high * edge_reg_high.to(dtype=aux_total.dtype)
                    aux_used = True

                if aux_used:
                    loss_val = aux_total if loss_val is None else (loss_val + aux_total)

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
                    if p is not None:
                        out_td.set("p_gate", p)
                else:
                    with contextlib.suppress(KeyError):
                        out_td.del_("refined_tokens")
                    with contextlib.suppress(KeyError):
                        out_td.del_("residual_context")
                    with contextlib.suppress(KeyError):
                        out_td.del_("p_gate")

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
        if isinstance(hist, Recorder):
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


# -----------------------------------------------------------------------------
# Precision / model policies (merged here; previously in fused.py)
# -----------------------------------------------------------------------------




class ModelPolicy:
    @staticmethod
    def negotiate(
        device: Optional[Union[torch.device, str]] = None,
        *args: Any,
        metadata: Optional[Dataset[Any]] = None,
        **kwargs: Any,
    ) -> torch.dtype:
        dev = torch.device(device) if device is not None else get_device()
        candidates: List[torch.dtype] = []
        match dev.type:
            case "cuda":
                try:
                    if Dataset.is_cuda_bf16_supported(dev):
                        candidates.append(torch.bfloat16)
                except Exception:
                    pass
                candidates.extend((torch.float16, torch.float32))
            case "cpu":
                if Dataset.is_cpu_bf16_supported():
                    candidates.append(torch.bfloat16)
                candidates.extend((torch.float32, torch.float64))
            case "xpu":
                candidates.extend((torch.bfloat16, torch.float32))
            case "mps":
                candidates.extend((torch.float16, torch.float32))
            case _:
                candidates.append(torch.float32)
        for dtype in candidates:
            if is_scale_safe(dtype, metadata):
                return dtype
        return (
            torch.float64 if is_scale_safe(torch.float64, metadata) else candidates[-1]
        )

    @staticmethod
    def _peek_layer(module: nn.Module) -> Optional[torch.Tensor]:
        with contextlib.suppress(StopIteration):
            return next(module.parameters())
        with contextlib.suppress(StopIteration):
            return next(module.buffers())
        return None

    @staticmethod
    def _coerce_metadata(
        model: nn.Module, metadata: Optional[Dataset[Any]] = None
    ) -> Dataset[Any]:
        Autocast.configure(model, metadata=metadata)
        meta = Autocast.metadata()
        if meta is None:
            ref = ModelPolicy._peek_layer(model)
            dev = ref.device if isinstance(ref, torch.Tensor) else get_device()
            meta = Dataset.for_device(dev)
            Autocast.configure(model, metadata=meta)
        return meta

    @staticmethod
    def _align_layers(
        src: nn.Module,
        dst: nn.Module,
        params_dtype: Optional[torch.dtype],
    ) -> None:
        ref = ModelPolicy._peek_layer(src)
        if ref is not None:
            with contextlib.suppress(Exception):
                dst.to(device=ref.device)
        if params_dtype is not None:
            with contextlib.suppress(Exception):
                dst.to(dtype=params_dtype)

    @staticmethod
    def _clone_state(
        src: nn.Module, dst: nn.Module, params_dtype: Optional[torch.dtype]
    ) -> None:
        try:
            state = src.state_dict()
        except (RuntimeError, AttributeError):
            return
        # Prefer direct load_state_dict to avoid building a full converted copy.
        try:
            dst.load_state_dict(state, strict=False)
            return
        except Exception:
            pass

        # Fallback: selective conversion only when needed.
        ref = ModelPolicy._peek_layer(dst)
        device = ref.device if ref is not None else None
        converted: Dict[str, Any] = {}
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                converted[key] = value
                continue
            tensor = value.detach()
            if (
                params_dtype is not None
                and tensor.is_floating_point()
                and tensor.dtype != params_dtype
            ):
                with contextlib.suppress(Exception):
                    tensor = tensor.to(dtype=params_dtype)
            if (
                device is not None
                and getattr(tensor, "device", None) is not None
                and tensor.device != device
            ):
                with contextlib.suppress(Exception):
                    tensor = tensor.to(device=device)
            converted[key] = tensor
        with contextlib.suppress(Exception):
            dst.load_state_dict(converted, strict=False)

    @staticmethod
    def _nvidia_linear(
        module: nn.Linear,
        params_dtype: Optional[torch.dtype],
        te: Any,
    ) -> Optional[nn.Module]:
        te_linear = getattr(te, "Linear", None)
        if te_linear is None:
            return None
        kwargs: Dict[str, Any] = {
            "in_features": module.in_features,
            "out_features": module.out_features,
            "bias": module.bias is not None,
        }
        if params_dtype is not None:
            kwargs["params_dtype"] = params_dtype
        try:
            replacement = te_linear(**kwargs)
        except Exception:
            return None
        ModelPolicy._align_layers(module, replacement, params_dtype)
        ModelPolicy._clone_state(module, replacement, params_dtype)
        return replacement

    @staticmethod
    def _nvidia_layer_norm(
        module: nn.LayerNorm,
        params_dtype: Optional[torch.dtype],
        te: Any,
    ) -> Optional[nn.Module]:
        te_layer_norm = getattr(te, "LayerNorm", None)
        if te_layer_norm is None:
            return None
        kwargs: Dict[str, Any] = {
            "normalized_shape": module.normalized_shape,
            "eps": module.eps,
        }
        if params_dtype is not None:
            kwargs["params_dtype"] = params_dtype
        try:
            replacement = te_layer_norm(**kwargs)
        except Exception:
            return None
        ModelPolicy._align_layers(module, replacement, params_dtype)
        if module.elementwise_affine:
            ModelPolicy._clone_state(module, replacement, params_dtype)
        return replacement

    @staticmethod
    def _nvidia_rms_norm(
        module: nn.Module,
        params_dtype: Optional[torch.dtype],
        te: Any,
    ) -> Optional[nn.Module]:
        te_rms_norm = getattr(te, "RMSNorm", None)
        if te_rms_norm is None:
            return None
        kwargs: Dict[str, Any] = {
            "normalized_shape": getattr(module, "normalized_shape", None),
            "eps": getattr(module, "eps", 1e-5),
        }
        if kwargs["normalized_shape"] is None:
            return None
        if params_dtype is not None:
            kwargs["params_dtype"] = params_dtype
        try:
            replacement = te_rms_norm(**kwargs)
        except Exception:
            return None
        ModelPolicy._align_layers(module, replacement, params_dtype)
        ModelPolicy._clone_state(module, replacement, params_dtype)
        return replacement

    @staticmethod
    def _to_nvidia_layers(
        model: nn.Module,
        *args: Any,
        apply_te_linear: bool,
        apply_te_layer_norm: bool,
        apply_te_rms_norm: bool,
        filter_linear: Optional[Callable[[nn.Linear, str], bool]],
        params_dtype: Optional[torch.dtype],
        **kwargs: Any,
    ) -> Tuple[nn.Module, int]:
        try:
            import transformer_engine.pytorch as te
        except Exception:
            return (model, 0)

        def _convert(parent: nn.Module) -> int:
            converted = 0
            for name, child in list(parent.named_children()):
                replacement: Optional[nn.Module] = None
                if apply_te_linear and isinstance(child, nn.Linear):
                    if filter_linear is None or filter_linear(child, name):
                        replacement = ModelPolicy._nvidia_linear(child, params_dtype, te)
                elif apply_te_layer_norm and isinstance(child, nn.LayerNorm):
                    replacement = ModelPolicy._nvidia_layer_norm(child, params_dtype, te)
                else:
                    rms_cls = getattr(torch.nn, "RMSNorm", None)
                    if (
                        apply_te_rms_norm
                        and rms_cls is not None
                        and isinstance(child, rms_cls)
                    ):
                        replacement = ModelPolicy._nvidia_rms_norm(child, params_dtype, te)
                if replacement is not None:
                    setattr(parent, name, replacement)
                    converted += 1
                    continue
                converted += _convert(child)
            return converted

        count = _convert(model)
        if count:
            invalidate_model_introspection_caches(model)
        return (model, count)

    @staticmethod
    def _to_nvidia_attention(
        model: nn.Module, *args: Any, params_dtype: Optional[torch.dtype], **kwargs: Any
    ) -> Tuple[nn.Module, int]:
        swapped = 0
        dot_cls = _dot_product_attention_cls()
        for module in model.modules():
            if (
                dot_cls is not None
                and isinstance(module, dot_cls)
                and getattr(module, "_te_ok", False)
            ):
                if not getattr(module, "te_first", False):
                    module.te_first = True
                swapped += 1
        if swapped:
            invalidate_model_introspection_caches(model)
        return (model, swapped)

    @staticmethod
    def use_nvidia_layers(
        model: nn.Module,
        device: Optional[Union[torch.device, str]] = None,
        *args: Any,
        metadata: Optional[Dataset[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> Tuple[nn.Module, bool, str]:
        dev = torch.device(device) if device is not None else get_device()
        if dev.type != "cuda":
            return (model, False, "Non-NVIDIA device; TE not applied")
        try:
            import transformer_engine.pytorch as te
        except Exception:
            return (model, False, "transformer_engine not installed")
        te_backend = getattr(te, "__name__", "transformer_engine.pytorch")
        fp8_ok, why = Dataset.is_float8_supported(dev)
        if fp8_ok:
            setattr(model, "__te_fp8_default__", True)
        params_dtype = kwargs.pop("params_dtype", None)
        if not isinstance(params_dtype, torch.dtype):
            params_dtype = ModelPolicy.negotiate(dev, metadata=metadata)
        # TE kernels are not intended for fp64 params; keep torch layers.
        if params_dtype is torch.float64:
            return (model, False, "TE disabled for fp64 params")
        model, n_layers = ModelPolicy._to_nvidia_layers(
            model,
            apply_te_linear=True,
            apply_te_layer_norm=True,
            apply_te_rms_norm=True,
            filter_linear=None,
            params_dtype=params_dtype,
        )
        try:
            model, attn_swapped = ModelPolicy._to_nvidia_attention(
                model, params_dtype=params_dtype
            )
        except Exception:
            attn_swapped = 0
        n_total = (n_layers or 0) + (attn_swapped or 0)
        _log_info(
            logger,
            f"[TE] swapped {n_total} modules (layers:{n_layers}, attn:{attn_swapped}); params_dtype={str(params_dtype).split('.')[-1]}, fp8={('on' if fp8_ok else 'off')} ({(why if fp8_ok else '')}), backend={te_backend}",
        )
        return (
            model,
            n_total > 0,
            f"TE applied (swapped {n_total}, layers={n_layers}, attn={attn_swapped}, dtype={params_dtype}, fp8={('on' if fp8_ok else 'off')}, backend={te_backend})",
        )

    @staticmethod
    def _enable_nvidia_training(
        model: nn.Module,
        params_dtype: torch.dtype,
        logger: Optional[Callable[[str], None]],
    ) -> Tuple[nn.Module, bool, str]:
        try:
            swapped_model, n = ModelPolicy._to_nvidia_layers(
                model,
                apply_te_linear=True,
                apply_te_layer_norm=True,
                apply_te_rms_norm=True,
                filter_linear=lambda lyr, _: lyr.in_features % 16 == 0
                and lyr.out_features % 16 == 0,
                params_dtype=params_dtype,
            )
            if n > 0:
                setattr(swapped_model, "__fp8_training_te__", True)
                if logger:
                    logger(f"[FP8][TE] swapped {n} modules")
                return (swapped_model, True, f"TE (swapped {n})")
            return (model, False, "TE present but no eligible modules")
        except Exception as exc:
            return (model, False, f"TE swap failed: {exc}")

    @staticmethod
    def _enable_torchao_training(
        model: nn.Module,
        logger: Optional[Callable[[str], None]],
    ) -> Tuple[nn.Module, bool, str]:
        try:
            from torchao.float8 import convert_to_float8_training

            res = convert_to_float8_training(model)
            converted = res or model
            setattr(converted, "__fp8_training_ao__", True)
            if logger:
                logger("[FP8][AO] convert_to_float8_training ok")
            return (converted, True, "torchao.float8")
        except Exception as exc:
            return (model, False, f"torchao convert failed: {exc}")

    @staticmethod
    def _enable_nvidia_inference(
        model: nn.Module,
        params_dtype: torch.dtype,
        logger: Optional[Callable[[str], None]],
    ) -> Tuple[nn.Module, bool, str]:
        try:
            swapped, n = ModelPolicy._to_nvidia_layers(
                model,
                apply_te_linear=True,
                apply_te_layer_norm=True,
                apply_te_rms_norm=True,
                filter_linear=lambda lyr, _: lyr.in_features % 16 == 0
                and lyr.out_features % 16 == 0,
                params_dtype=params_dtype,
            )
            if n > 0:
                setattr(swapped, "__fp8_inference_te__", True)
                if logger:
                    logger(f"[FP8][TE] swapped {n} modules; using te.fp8_autocast")
                return (swapped, True, f"TE swap ({n})")
            return (model, False, "no eligible Linear (dims%16)")
        except Exception as exc:
            return (model, False, f"TE swap failed: {exc}")

    @staticmethod
    def _reuse_nvidia_layers(
        model: nn.Module,
        logger: Optional[Callable[[str], None]],
    ) -> Tuple[nn.Module, bool, str]:
        te_present = any(
            (
                getattr(module.__class__, "__module__", "").startswith(
                    "transformer_engine"
                )
                for module in model.modules()
            )
        )
        if te_present:
            setattr(model, "__fp8_inference_te__", True)
            invalidate_model_introspection_caches(model)
            if logger:
                logger("[FP8][TE] te.* already present; using te.fp8_autocast")
            return (model, True, "TE present")
        return (model, False, "TE layers not present")

    @staticmethod
    def _enable_torchao_inference(
        model: nn.Module,
        dynamic_activations: bool,
        logger: Optional[Callable[[str], None]],
    ) -> Tuple[nn.Module, bool, str]:
        try:
            from torchao.quantization import (
                Float8DynamicActivationFloat8WeightConfig,
                Float8WeightOnlyConfig,
                quantize_,
            )

            cfg = (
                Float8DynamicActivationFloat8WeightConfig()
                if dynamic_activations
                else Float8WeightOnlyConfig()
            )
            quantize_(model, cfg)
            setattr(model, "__fp8_inference_ao__", True)
            _log_info(logger, f"[FP8][AO] applied {cfg.__class__.__name__}")
            return (model, True, "torchao")
        except Exception as exc:
            return (model, False, f"AO failed: {exc}")

    @staticmethod
    def enable_float8_training(
        model: nn.Module,
        metadata: Optional[Dataset[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[nn.Module, bool, str]:
        meta = ModelPolicy._coerce_metadata(model, metadata)
        device = torch.device(meta.device)
        ok, reason = Dataset.is_float8_supported(device)
        if not ok:
            Autocast.configure(model, metadata=meta)
            return (model, False, reason)
        if getattr(meta, "has_scale", False):
            float8_dtypes = Autocast.float8_formats()
            if not any(
                is_scale_safe(dtype, meta, safety_margin=2.0) for dtype in float8_dtypes
            ):
                _log_info(
                    logger, "[FP8] training disabled: data scale exceeds float8 range"
                )
                Autocast.configure(model, metadata=meta)
                return (model, False, "data scale")
        params_dtype = ModelPolicy.negotiate(device, metadata=meta)

        for backend in ("te", "torchao"):
            if backend == "te":
                m2, ok2, why = ModelPolicy._enable_nvidia_training(
                    model, params_dtype, logger
                )
            else:
                m2, ok2, why = ModelPolicy._enable_torchao_training(model, logger)
            if ok2:
                _log_info(logger, f"[FP8] training enabled via {why} ({reason})")
                Autocast.configure(m2, metadata=meta)
                return (m2, True, why)
            else:
                _log_debug(logger, f"[FP8] {backend} path skipped: {why}")
        Autocast.configure(model, metadata=meta)
        return (model, False, "No usable FP8 backend")

    @staticmethod
    def enable_float8_prediction(
        model: nn.Module,
        metadata: Optional[Dataset[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[nn.Module, bool, str]:
        meta = ModelPolicy._coerce_metadata(model, metadata)
        device = torch.device(meta.device)
        ok, reason = Dataset.is_float8_supported(device)
        if not ok:
            Autocast.configure(model, metadata=meta)
            return (model, False, reason)
        if getattr(meta, "has_scale", False):
            float8_dtypes = Autocast.float8_formats()
            if not any(
                is_scale_safe(dtype, meta, safety_margin=2.0) for dtype in float8_dtypes
            ):
                _log_info(
                    logger, "[FP8] inference disabled: data scale exceeds float8 range"
                )
                Autocast.configure(model, metadata=meta)
                return (model, False, "data scale")
        params_dtype = ModelPolicy.negotiate(device, metadata=meta)
        dynamic_activations = not (
            getattr(meta, "has_scale", False)
            and getattr(meta, "scale_is_integral", None) is True
        )
        order = ("te_swap", "te_present", "ao")
        for step in order:
            if step == "te_swap":
                m2, ok2, why = ModelPolicy._enable_nvidia_inference(
                    model, params_dtype, logger
                )
            elif step == "te_present":
                m2, ok2, why = ModelPolicy._reuse_nvidia_layers(model, logger)
            else:
                m2, ok2, why = ModelPolicy._enable_torchao_inference(
                    model, dynamic_activations, logger
                )
            if ok2:
                _log_info(logger, f"[FP8] inference enabled via {why} ({reason})")
                Autocast.configure(m2, metadata=meta)
                return (m2, True, why)
            else:
                _log_debug(logger, f"[FP8] {step} skipped: {why}")
        Autocast.configure(model, metadata=meta)
        return (model, False, "No usable FP8 backend")

    @staticmethod
    def enable_int8_training(
        model: nn.Module,
        metadata: Optional[Dataset[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[nn.Module, bool, str]:
        meta = ModelPolicy._coerce_metadata(model, metadata)
        device = torch.device(meta.device)
        with contextlib.suppress(Exception):
            model.to(device)
        dynamic_activations = not (
            getattr(meta, "has_scale", False)
            and getattr(meta, "scale_is_integral", None) is True
        )
        group_size = 128
        m2, ok, why = Quantization.enable_qat(
            model,
            dynamic_activations=dynamic_activations,
            group_size=group_size,
            logger=logger,
        )
        Autocast.configure(m2 if ok else model, metadata=meta)
        return (m2, ok, why)

    @staticmethod
    def enable_int8_prediction(
        model: nn.Module,
        metadata: Optional[Dataset[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[nn.Module, bool, str]:
        meta = ModelPolicy._coerce_metadata(model, metadata)
        device = torch.device(meta.device)
        with contextlib.suppress(Exception):
            model.to(device)
        dynamic_activations = not (
            getattr(meta, "has_scale", False)
            and getattr(meta, "scale_is_integral", None) is True
        )
        m2, ok, why = Quantization._enable_ptq(
            model, dynamic_activations=dynamic_activations, logger=logger
        )
        Autocast.configure(m2 if ok else model, metadata=meta)
        return (m2, ok, why)


# -----------------------------------------------------------------------------
# Quantization helpers (merged here; previously in fused.py)
# -----------------------------------------------------------------------------


def _is_ptq_unavailable(
    model: nn.Module, *args: Any, **kwargs: Any
) -> tuple[nn.Module, bool, str]:
    return (model, False, "PTQ backend unavailable")


_Int8DynamicActivationInt8WeightConfig: Any | None
_Int8WeightOnlyConfig: Any | None
_PTQ_IMPL: Callable[..., tuple[nn.Module, bool, str]] | None

try:
    from torchao.quantization.quant_api import (
        Int8DynamicActivationInt8WeightConfig as _Int8DynamicActivationInt8WeightConfig,
        Int8WeightOnlyConfig as _Int8WeightOnlyConfig,
        quantize_ as _quantize,
    )

    try:
        from torchao.quantization import quant_primitives as _qp
    except Exception:
        _qp = None

    _PTQ_IMPL = _quantize
except Exception:  # pragma: no cover
    _Int8DynamicActivationInt8WeightConfig = None
    _Int8WeightOnlyConfig = None
    _PTQ_IMPL = None
    _qp = None


class Quantization:
    """Best-effort quantization utilities.

    - QAT: prepare model for fake-quant aware training
    - PTQ: apply post-training quantization where available

    This lives in model space because it fundamentally changes the module graph.
    """

    @staticmethod
    def is_qat_available() -> bool:
        return bool(_qp is not None)

    @staticmethod
    def is_ptq_available() -> bool:
        return bool(_PTQ_IMPL is not None and _Int8DynamicActivationInt8WeightConfig is not None)

    @staticmethod
    def _prepare_qat(
        model: nn.Module,
        *args: Any,
        dynamic_activations: bool = True,
        group_size: int = 128,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> Any:
        if _qp is None:
            raise RuntimeError("torchao.quantization.quant_primitives unavailable")
        # QAT uses fake-quantize modules; keep it conservative.
        _log_debug(logger, f"[INT8][QAT] prepare(dynamic_activations={dynamic_activations}, group={group_size})")

        # torchao does not provide a single stable public QAT API across versions.
        # Best-effort: attach fake quant stubs where possible.
        # NOTE: This intentionally does not attempt to calibrate.
        try:
            # Some versions expose `FakeQuantize` in torchao.
            from torchao.quantization.fake_quant import (
                FakeQuantizeConfig,
                Int8ActivationConfig,
                Int8WeightConfig,
                prepare_qat_ as _prepare_qat,
            )

            cfg = FakeQuantizeConfig(
                activation=Int8ActivationConfig(dynamic=bool(dynamic_activations)),
                weight=Int8WeightConfig(group_size=int(group_size)),
            )
            _prepare_qat(model, cfg)
            invalidate_model_introspection_caches(model)
            return cfg
        except Exception as exc:
            raise RuntimeError(f"torchao QAT prepare unavailable: {exc}") from exc

    @staticmethod
    def _apply_ptq(
        model: nn.Module,
        *args: Any,
        dynamic_activations: bool = True,
        group_size: int = 128,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> tuple[nn.Module, bool, str]:
        if _PTQ_IMPL is None:
            return _is_ptq_unavailable(model)

        if _Int8DynamicActivationInt8WeightConfig is None:
            return _is_ptq_unavailable(model)

        cfg: Any
        why: str
        if bool(dynamic_activations):
            cfg = _Int8DynamicActivationInt8WeightConfig(group_size=int(group_size))
            why = "int8_dynamic_act_int8_weight"
        else:
            # Weight-only PTQ (activations remain fp16/fp32).
            if _Int8WeightOnlyConfig is None:
                return (model, False, "Int8WeightOnlyConfig unavailable")
            cfg = _Int8WeightOnlyConfig(group_size=int(group_size))
            why = "int8_weight_only"

        try:
            _log_info(logger, f"[INT8][PTQ] applying {why} (group={group_size})")
            _PTQ_IMPL(model, cfg)
            invalidate_model_introspection_caches(model)
            return (model, True, why)
        except Exception as exc:
            return (model, False, f"PTQ failed: {exc}")

    @classmethod
    def enable_qat(
        cls,
        model: nn.Module,
        *args: Any,
        dynamic_activations: bool = True,
        group_size: int = 128,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> tuple[nn.Module, bool, str]:
        if not cls.is_qat_available():
            return (model, False, "QAT backend unavailable")
        try:
            cls._prepare_qat(
                model,
                dynamic_activations=dynamic_activations,
                group_size=group_size,
                logger=logger,
            )
            setattr(model, "__int8_training_qat__", True)
            return (model, True, "QAT-prepare")
        except Exception as exc:
            return (model, False, f"QAT prepare failed: {exc}")

    @classmethod
    def _enable_ptq(
        cls,
        model: nn.Module,
        *args: Any,
        dynamic_activations: bool = True,
        group_size: int = 128,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> tuple[nn.Module, bool, str]:
        return cls._apply_ptq(
            model,
            dynamic_activations=dynamic_activations,
            group_size=group_size,
            logger=logger,
        )

    @classmethod
    def enable_int8_training(
        cls,
        model: nn.Module,
        *args: Any,
        dynamic_activations: bool = True,
        group_size: int = 128,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> tuple[nn.Module, bool, str]:
        """Enable INT8 training.

        Prefer QAT when available; fall back to PTQ if possible.
        """
        if getattr(model, "__int8_training_qat__", False) or getattr(
            model, "__int8_training_ptq__", False
        ):
            return (model, True, "already-enabled")

        last_err: Optional[Exception] = None
        if cls.is_qat_available():
            try:
                cls._prepare_qat(
                    model,
                    dynamic_activations=dynamic_activations,
                    group_size=group_size,
                    logger=logger,
                )
                setattr(model, "__int8_training_qat__", True)
                return (model, True, "QAT-prepare")
            except Exception as exc:
                last_err = exc
                _log_info(logger, f"[INT8][QAT] prepare failed: {exc}")

        try:
            m2, ok, why = cls._apply_ptq(
                model,
                dynamic_activations=dynamic_activations,
                group_size=group_size,
                logger=logger,
            )
        except Exception as exc:
            err = exc or last_err or RuntimeError("Unknown PTQ failure")
            return (model, False, f"INT8 training path unavailable: {err}")

        if ok:
            setattr(m2, "__int8_training_ptq__", True)
            return (m2, True, f"PTQ({why})")
        return (model, False, f"PTQ failed: {why}")


@lru_cache(maxsize=1)
def _dot_product_attention_cls() -> Any:
    try:
        from .kernels import DotProductAttention as _DotProductAttention

        return _DotProductAttention
    except Exception:
        return None
