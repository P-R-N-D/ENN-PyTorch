# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import logging
import math
import threading
import weakref
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn

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
from .primitives import CrossAttention, DilatedAttention, PatchAttention, Retention, norm_layer
from .blocks import (
    History,
    LongNet,
    PointTransformer,
    RetNet,
    Scaler,
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
)

_LOGGER = logging.getLogger(__name__)


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

        # History/logging is runtime metadata; keep it on CPU to reduce device
        # traffic and GPU memory pressure.
        self.logger = History()

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

    def to(self, *args: Any, **kwargs: Any) -> "Root":
        """Move the model to a device/dtype while keeping History on CPU.

        Root owns device placement for the core model, but History is runtime
        metadata/logging and should remain on CPU (especially important for
        CUDA/XPU where moving it would allocate device memory and can trigger
        unnecessary device syncs).
        """
        out = super().to(*args, **kwargs)
        with contextlib.suppress(Exception):
            if isinstance(getattr(self, "logger", None), History):
                self.logger.cpu()
        return cast(Root, out)

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
