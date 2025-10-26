# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any, List, Optional, Protocol, Sequence, Tuple, Union, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ..utils.compat import patch_torch
from ..utils.optimization import (
    AutoCast,
    compile,
    inference,
)

from .functional import SwiGLU
from .layers import (
    GlobalEncoderLayer,
    CrossAttention,
    PatchAttention,
    StochasticDepth,
    TemporalEncoderLayer,
    norm_layer,
    schedule_stochastic_depth,
)

patch_torch()
if TYPE_CHECKING:
    from .config import ModelConfig


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
        self.d_model = int(d_model)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(p=drop_path, mode="row")
        self.norm1 = norm_layer(norm_type, self.d_model)
        self.attn = PatchAttention(
            self.d_model, nhead, coord_dim=coord_dim
        )
        self.norm2 = norm_layer(norm_type, self.d_model)
        hid = int(self.d_model * mlp_ratio * (2.0 / 3.0))
        self.ffn = SwiGLU(
            self.d_model, hid, out_dim=self.d_model, dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if coords.shape[:2] != x.shape[:2]:
            raise ValueError("coords must have shape (B, N, C)")
        y = self.attn(self.norm1.forward(x), coords, attn_mask=attn_mask)
        x = x + self.drop_path(self.dropout(y))
        x = x + self.drop_path(self.dropout(self.ffn(self.norm2.forward(x))))
        return x


_MODELING_TYPE_ALIASES: dict[str, str] = {
    "ss": "sxs",
    "spatial": "sxs",
    "sxs": "sxs",
    "tt": "txt",
    "temporal": "txt",
    "txt": "txt",
    "txs": "txs",
    "st": "sxt",
    "ts": "sxt",
    "sxt": "sxt",
    "spatiotemporal": "sxt",
    "spatio-temporal": "sxt",
    "temporospatial": "sxt",
    "temporo-spatial": "sxt",
}

def _normalize_modeling_type(value: Any) -> str:
    mode = str(value).strip().lower()
    normalized = _MODELING_TYPE_ALIASES.get(mode)
    if normalized is None:
        raise ValueError(f"Unsupported modeling type '{value}'")
    return normalized

class SpatialEncoder(nn.Module):
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
        drops = schedule_stochastic_depth(drop_path, depth)
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
        for blk in self.blocks:
            x = blk(x, coords, attn_mask=attn_mask)
        return self.norm(x)

class TemporalEncoderBlock(nn.Module):
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
        self.norm1 = norm_layer(norm_type, d_model)
        self.retention = TemporalEncoderLayer(d_model, nhead)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(p=drop_path, mode="row")
        self.norm2 = norm_layer(norm_type, d_model)
        hid = int(d_model * mlp_ratio * (2.0 / 3.0))
        self.ffn = SwiGLU(d_model, hid, out_dim=d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        h, state = self.retention(
            self.norm1.forward(x), attn_mask=causal_mask, state=state
        )
        x = x + self.drop_path(self.dropout(h))
        x = x + self.drop_path(self.dropout(self.ffn(self.norm2.forward(x))))
        return x, state


class TemporalEncoder(nn.Module):
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
        drops = schedule_stochastic_depth(drop_path, depth)
        self.blocks = nn.ModuleList(
            [
                TemporalEncoderBlock(
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
        *,
        return_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[dict]]]:
        next_state = state
        for blk in self.blocks:
            x, next_state = blk(x, causal_mask=causal_mask, state=next_state)
        x = self.norm.forward(x)
        if return_state:
            return x, next_state
        return x

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
        self.mix = SwiGLU(2 * d_model, hid, out_dim=d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(p=drop_path, mode="row")

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        temporal_tokens: torch.Tensor,
        mode: str = "spatiotemporal",
    ) -> torch.Tensor:
        mode_l = _normalize_modeling_type(mode)
        s_context = self.cross_s(spatial_tokens, temporal_tokens)
        t_context = self.cross_t(temporal_tokens, spatial_tokens)
        if mode_l == "sxs":
            return s_context
        if mode_l == "txt":
            return t_context
        if mode_l == "txs":
            base = torch.cat(
                [
                    t_context,
                    s_context.mean(dim=1, keepdim=True).expand(
                        -1, t_context.size(1), -1
                    ),
                ],
                dim=-1,
            )
            fused = self.mix(self.mix_norm.forward(base))
            return t_context + self.drop_path(self.dropout(fused))
        base = torch.cat(
            [
                s_context,
                t_context.mean(dim=1, keepdim=True).expand(
                    -1, s_context.size(1), -1
                ),
            ],
            dim=-1,
        )
        fused = self.mix(self.mix_norm.forward(base))
        return s_context + self.drop_path(self.dropout(fused))

@dataclass
class Payload:
    tokens: torch.Tensor
    context: torch.Tensor
    flat: torch.Tensor
    offset: torch.Tensor
    context_shape: Tuple[int, ...]

class GlobalEncoderBlock(nn.Module):
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
        self.norm1 = norm_layer(norm_type, d_model)
        self.retention = GlobalEncoderLayer(d_model, nhead)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(p=drop_path, mode="row")
        self.norm2 = norm_layer(norm_type, d_model)
        hid = int(d_model * mlp_ratio * (2.0 / 3.0))
        self.ffn = SwiGLU(d_model, hid, out_dim=d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        h, state = self.retention(
            self.norm1.forward(x), attn_mask=causal_mask, state=state
        )
        x = x + self.drop_path(self.dropout(h))
        x = x + self.drop_path(self.dropout(self.ffn(self.norm2.forward(x))))
        return x, state

class GlobalEncoder(nn.Module):
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
        drops = schedule_stochastic_depth(drop_path, depth)
        self.blocks = nn.ModuleList(
            [
                GlobalEncoderBlock(
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
        tokens: torch.Tensor,
        *,
        state: Optional[dict] = None,
        return_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[dict]]]:
        next_state = state
        for blk in self.blocks:
            tokens, next_state = blk(tokens, causal_mask=None, state=next_state)
        tokens = self.norm.forward(tokens)
        if return_state:
            return tokens, next_state
        return tokens

class LocalProcessor(nn.Module):
    def __init__(
        self, in_dim: int, out_shape: Sequence[int], config: ModelConfig
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_shape = tuple((int(v) for v in out_shape))
        self.out_dim = int(math.prod(self.out_shape) if self.out_shape else 1)
        self.d_model = int(config.depth)
        self.nhead = int(config.heads)
        self.modeling_type = _normalize_modeling_type(
            config.modeling_type
        )
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
            self._build_spatial_coords(
                self.spatial_tokens, device=torch.device("cpu")
            ),
            persistent=False,
        )
        self.spatial_encoder = SpatialEncoder(
            self.d_model,
            self.nhead,
            depth=max(1, int(config.spatial_depth)),
            coord_dim=self.spatial_coords_template.shape[-1],
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            drop_path=self.drop_path,
            norm_type=self.norm_type,
        )
        self.temporal_encoder = TemporalEncoder(
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
        self.head = nn.Sequential(
            norm_layer(self.norm_type, self.d_model),
            nn.Linear(self.d_model, hid),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hid, self.out_dim),
        )

    @staticmethod
    def _build_spatial_coords(
        n_tokens: int, device: torch.device
    ) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> Payload:
        B = x.shape[0]
        spatial_tokens = self.spatial_tokenizer(x).view(
            B, self.spatial_tokens, self.d_model
        )
        temporal_tokens = self.temporal_tokenizer(x).view(
            B, self.temporal_tokens, self.d_model
        )
        coords = self._spatial_coords(B, x.device, spatial_tokens.dtype)
        spatial_out = self.spatial_encoder.forward(spatial_tokens, coords)
        temporal_out = self.temporal_encoder.forward(temporal_tokens)
        mode = self.modeling_type
        if mode == "sxs":
            tokens = spatial_out
        elif mode == "txt":
            tokens = temporal_out
        elif mode == "txs":
            tokens = self.perception(temporal_out, spatial_out, mode="txs")
        elif mode == "sxt":
            tokens = self.perception(spatial_out, temporal_out, mode="sxt")
        else:
            raise RuntimeError(f"Unhandled modeling type '{mode}'")
        tokens = self.norm.forward(tokens)
        pooled = tokens.mean(dim=1)
        flat = self.head(pooled)
        context = flat.view(B, *self.out_shape)
        dims = tuple(range(1, context.ndim))
        offset = context.mean(dim=dims, keepdim=True)
        return Payload(
            tokens=tokens,
            context=context,
            flat=flat,
            offset=offset,
            context_shape=self.out_shape,
        )

    def decode(
        self, tokens: torch.Tensor, *args: Any, apply_norm: bool = False, **kwargs: Any
    ) -> torch.Tensor:
        if apply_norm:
            tokens = self.norm.forward(tokens)
        pooled = tokens.mean(dim=1)
        flat = self.head(pooled)
        return flat.view(tokens.shape[0], *self.out_shape)


class LossWeightPolicy(Protocol):
    def weights(self) -> Tuple[float, float]:
        ...

    def update(
        self,
        top_loss: Optional[torch.Tensor],
        bottom_loss: Optional[torch.Tensor],
    ) -> None:
        raise NotImplementedError


class Root(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_shape: Sequence[int],
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_shape = tuple((int(x) for x in out_shape))
        self.out_dim = int(prod(self.out_shape))
        self._loss_space = str(getattr(config, "loss_space", "z")).lower()
        self._y_low = float(getattr(config, "y_low", 0.0))
        self._y_high = float(getattr(config, "y_high", 100.0))
        self._y_eps_range = float(getattr(config, "y_eps_range", 1e-3))
        self._y_eps_rel = float(getattr(config, "y_eps_rel", 0.02))
        self._z_reg_lambda = float(getattr(config, "z_reg_lambda", 1e-2))
        self._range_penalty_lambda = float(
            getattr(config, "range_penalty_lambda", 0.0)
        )
        self._calib_enable = bool(getattr(config, "calibrate_output", True))
        
        c_scale = float(getattr(config, "calibrate_init_scale", 1.0))
        c_bias  = float(getattr(config, "calibrate_init_bias", 0.0))
        self.calib_scale = nn.Parameter(torch.ones(1, dtype=torch.float64) * c_scale)
        self.calib_bias  = nn.Parameter(torch.ones(1, dtype=torch.float64) * c_bias)
        
        def _calib_load_pre_hook(module, state_dict, prefix, local_md, strict, missing_keys, unexpected_keys, error_msgs):
            for name in ("calib_scale", "calib_bias"):
                key = prefix + name
                if key not in state_dict:
                    continue
                v = state_dict[key]
                if not isinstance(v, torch.Tensor):
                    v = torch.as_tensor(v)
                if v.ndim == 0:
                    v = v.view(1)
                elif v.ndim != 1 or v.numel() != 1:
                    v = v.reshape(-1)[:1]
                state_dict[key] = v.to(dtype=module.calib_scale.dtype)
                
        self._calib_pre_hook_handle = self._register_load_state_dict_pre_hook(
            _calib_load_pre_hook, with_module=True
        )
        if config.device is not None:
            self._device = torch.device(config.device)
        else:
            if torch.cuda.is_available():
                device_name = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device_name = "mps"
            elif getattr(torch, "is_vulkan_available", None) and torch.is_vulkan_available():
                device_name = "vulkan"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                device_name = "xpu"
            else:
                device_name = "cpu"
            self._device = torch.device(device_name)
        self.register_buffer(
            "y_low_buf",
            torch.tensor(self._y_low, device=self._device, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "y_high_buf",
            torch.tensor(self._y_high, device=self._device, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "y_eps_range_buf",
            torch.tensor(
                self._y_eps_range, device=self._device, dtype=torch.float32
            ),
            persistent=True,
        )
        self.set_y_range(self._y_low, self._y_high, self._y_eps_range)
        self.is_norm_linear = bool(getattr(config, "use_linear_branch", False))
        self.linear_branch = (
            nn.Linear(self.in_dim, self.out_dim).to(self._device)
            if self.is_norm_linear
            else None
        )
        self.local_net = LocalProcessor(
            self.in_dim, self.out_shape, config=config
        ).to(self._device)
        global_net = GlobalEncoder(
            int(config.depth),
            int(config.heads),
            depth=max(1, int(getattr(config, "temporal_depth", 1))),
            mlp_ratio=float(getattr(config, "mlp_ratio", 4.0)),
            dropout=float(getattr(config, "dropout", 0.0)),
            drop_path=float(getattr(config, "drop_path", 0.0)),
            norm_type=str(getattr(config, "normalization_method", "layernorm")),
        ).to(self._device)
        self.global_net = global_net
        self.microbatch = int(config.microbatch)
        if self.microbatch <= 0:
            raise ValueError(
                f"config.microbatch must be >= 1, got {config.microbatch}"
            )
        try:
            self.register_buffer(
                "output_baked_flag",
                torch.tensor(0, dtype=torch.uint8),
                persistent=True,
            )
        except Exception:
            pass
        mode = str(getattr(config, "compile_mode", "default"))
        enable_compilation = bool(getattr(config, "enable_compilation", False))
        if enable_compilation:
            try:
                self.local_net = compile(
                    self.local_net,
                    mode=mode,
                    fullgraph=False,
                    dynamic=False,
                    backend="inductor",
                )
                self.global_net = compile(
                    self.global_net,
                    mode=mode,
                    fullgraph=False,
                    dynamic=False,
                    backend="inductor",
                )
            except Exception:
                pass
        self.__config = config
        self._label_dim = int(prod(out_shape))
        self.register_buffer(
            "y_stats_ready", torch.tensor(False, dtype=torch.bool)
        )
        self.register_buffer("y_eps", torch.tensor(1e-06, dtype=torch.float32))
        d = self._label_dim
        dev = getattr(self, "_device", torch.device("cpu"))
        self.register_buffer(
            "y_min",
            torch.full((d,), float("inf"), device=dev, dtype=torch.float32),
        )
        self.register_buffer(
            "y_max",
            torch.full((d,), float("-inf"), device=dev, dtype=torch.float32),
        )
        self.register_buffer(
            "y_sum", torch.zeros(d, device=dev, dtype=torch.float64)
        )
        self.register_buffer(
            "y_sum2", torch.zeros(d, device=dev, dtype=torch.float64)
        )
        self.register_buffer(
            "y_count", torch.zeros(d, device=dev, dtype=torch.float64)
        )
        self.register_buffer(
            "y_mean", torch.zeros(d, device=dev, dtype=torch.float32)
        )
        self.register_buffer(
            "y_std", torch.ones(d, device=dev, dtype=torch.float32)
        )
        self.register_buffer(
            "x_seen_elems", torch.zeros((), device=dev, dtype=torch.float64)
        )
        self.register_buffer(
            "x_mean",
            torch.zeros(self.in_dim, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "x_std",
            torch.ones(self.in_dim, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "x_stats_ready",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
        self._x_eps: float = 1e-06
        self._input_scale_method: str = "standard"
        self._x_sum: torch.Tensor | None = None
        self._x_sum2: torch.Tensor | None = None
        self._x_count: torch.Tensor | None = None

    def set_input_scale_method(self, method: str = "standard") -> None:
        self._input_scale_method = (
            "standard" if str(method).lower() == "standard" else "none"
        )

    def update_x_stats(self, X: torch.Tensor) -> None:
        if X is None:
            return
        with inference(self):
            x = torch.as_tensor(X).detach()
            x = torch.atleast_2d(x)
            if x.dim() != 2:
                x = x.view(x.shape[0], -1)
            x64 = x.to(dtype=torch.float64, device="cpu")
            if self._x_sum is None or self._x_sum.shape[0] != x64.shape[1]:
                D = int(x64.shape[1])
                self._x_sum = torch.zeros(D, dtype=torch.float64)
                self._x_sum2 = torch.zeros(D, dtype=torch.float64)
                self._x_count = torch.zeros((), dtype=torch.int64)
            xnz = torch.nan_to_num(x64, nan=0.0, posinf=0.0, neginf=0.0)
            self._x_sum += xnz.sum(dim=0)
            self._x_sum2 += (xnz * xnz).sum(dim=0)
            self._x_count += int(x64.shape[0])

    def finalize_x_stats(self) -> None:
        with inference(self):
            if (
                self._x_sum is None
                or self._x_sum2 is None
                or self._x_count is None
                or (int(self._x_count.item()) == 0)
            ):
                self.x_stats_ready.fill_(False)
                return
            dev = next(self.parameters()).device
            s = self._x_sum.to(device=dev)
            s2 = self._x_sum2.to(device=dev)
            c = torch.tensor(
                float(int(self._x_count.item())), dtype=torch.float64, device=dev
            )
            if dist.is_available() and dist.is_initialized():
                backend = ""
                try:
                    backend = str(dist.get_backend()).lower()
                except Exception:
                    backend = ""
                work_tensors = [s, s2, c]
                original_devices = [t.device for t in work_tensors]
                target_device = None
                if backend == "nccl" and torch.cuda.is_available():
                    try:
                        target_device = torch.device("cuda", torch.cuda.current_device())
                    except Exception:
                        target_device = torch.device("cuda", 0)
                if target_device is not None:
                    work_tensors = [
                        t.to(device=target_device) if t.device != target_device else t
                        for t in work_tensors
                    ]
                dist.all_reduce(work_tensors[0], op=dist.ReduceOp.SUM)
                dist.all_reduce(work_tensors[1], op=dist.ReduceOp.SUM)
                dist.all_reduce(work_tensors[2], op=dist.ReduceOp.SUM)
                work_tensors = [
                    t.to(device=original_devices[i])
                    if t.device != original_devices[i]
                    else t
                    for i, t in enumerate(work_tensors)
                ]
                s, s2, c = work_tensors
            s = s.cpu()
            s2 = s2.cpu()
            c = float(c.cpu().item())
            c = max(1.0, c)
            mean = (s / c).to(torch.float32)
            var = s2 / c - mean.to(torch.float64).pow(2)
            std = torch.sqrt(var.clamp_min(self._x_eps**2)).to(torch.float32)
            self.x_mean.data.copy_(mean)
            self.x_std.data.copy_(std)
            self.x_stats_ready.data.fill_(True)
            self._x_sum = None
            self._x_sum2 = None
            self._x_count = None

    def _normalize_inputs(self, X: torch.Tensor) -> torch.Tensor:
        if self._input_scale_method != "standard" or not bool(
            self.x_stats_ready.item()
        ):
            return X
        mu = self.x_mean.to(device=X.device, dtype=X.dtype)
        sd = self.x_std.to(device=X.device, dtype=X.dtype).clamp_min(
            self._x_eps
        )
        return (X - mu) / sd

    def update_y_stats(self, y_raw: torch.Tensor) -> None:
        with inference(self):
            y = (
                y_raw.detach()
                .view(y_raw.shape[0], -1)
                .to(device=self.y_min.device, dtype=torch.float32)
            )
            _, d = y.shape
            if d != self._label_dim:
                raise ValueError(
                    f"Target flattened dim {d} != model label_dim {self._label_dim}"
                )
            _min_res = torch.nanmin(y, dim=0)
            _max_res = torch.nanmax(y, dim=0)
            batch_min = getattr(
                _min_res,
                "values",
                _min_res[0] if isinstance(_min_res, (tuple, list)) else _min_res,
            )
            batch_max = getattr(
                _max_res,
                "values",
                _max_res[0] if isinstance(_max_res, (tuple, list)) else _max_res,
            )
            batch_sum = torch.nansum(y, dim=0, dtype=torch.float64)
            batch_sum2 = torch.nansum(y.to(torch.float64) ** 2, dim=0)
            batch_cnt = torch.sum(torch.isfinite(y), dim=0, dtype=torch.float64)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(batch_min, op=dist.ReduceOp.MIN)
                dist.all_reduce(batch_max, op=dist.ReduceOp.MAX)
                dist.all_reduce(batch_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(batch_sum2, op=dist.ReduceOp.SUM)
                dist.all_reduce(batch_cnt, op=dist.ReduceOp.SUM)
            self.y_min.copy_(torch.fmin(self.y_min, batch_min))
            self.y_max.copy_(torch.fmax(self.y_max, batch_max))
            self.y_sum.add_(batch_sum)
            self.y_sum2.add_(batch_sum2)
            self.y_count.add_(batch_cnt)

    def finalize_y_stats(self) -> None:
        with inference(self):
            valid = self.y_count > 0
            y_mean = torch.zeros_like(
                self.y_sum, dtype=torch.float64, device=self.y_sum.device
            )
            var = torch.zeros_like(self.y_sum2, device=self.y_sum2.device)
            y_mean[valid] = self.y_sum[valid] / self.y_count[valid]
            var[valid] = (
                self.y_sum2[valid] / self.y_count[valid] - y_mean[valid] ** 2
            )
            y_std = torch.sqrt(
                torch.clamp(
                    var,
                    min=float(self.y_eps.item()) ** 2
                    if hasattr(self, "y_eps")
                    else 1e-12,
                )
            )
            self.y_mean.copy_(y_mean.to(torch.float32))
            self.y_std.copy_(y_std.to(torch.float32))
            self.y_stats_ready.fill_(bool((self.y_count > 0).any().item()))

    def has_valid_y_stats(self) -> bool:
        try:
            ready = bool(self.y_stats_ready.item())
        except Exception:
            ready = False
        if not ready:
            return False
        if hasattr(self, "y_std"):
            try:
                return bool((self.y_std > 0).any().item())
            except Exception:
                return False
        try:
            return bool((self.y_max > self.y_min).any().item())
        except Exception:
            return False

    def set_y_range(self, y_low: float, y_high: float, eps: float | None = None) -> None:
        if not (float(y_high) > float(y_low)):
            raise ValueError(f"y_high({y_high}) must be > y_low({y_low})")
        if eps is not None and float(eps) <= 0.0:
            raise ValueError("eps must be positive when provided")
        self.y_low_buf.copy_(
            torch.tensor(float(y_low), dtype=self.y_low_buf.dtype, device=self.y_low_buf.device)
        )
        self.y_high_buf.copy_(
            torch.tensor(float(y_high), dtype=self.y_high_buf.dtype, device=self.y_high_buf.device)
        )
        if eps is not None:
            self.y_eps_range_buf.copy_(
                torch.tensor(float(eps), dtype=self.y_eps_range_buf.dtype, device=self.y_eps_range_buf.device)
            )

    def _to_logit_range(self, y: torch.Tensor) -> torch.Tensor:
        A = self.y_low_buf.to(device=y.device, dtype=y.dtype)
        B = self.y_high_buf.to(device=y.device, dtype=y.dtype)
        if (
            not torch.isfinite(A).all().item()
            or not torch.isfinite(B).all().item()
            or not bool((B > A).all().item())
        ):
            base_low = float(self._y_low)
            base_high = float(self._y_high)
            if (not math.isfinite(base_low)) or (not math.isfinite(base_high)) or (
                base_high <= base_low
            ):
                base_low, base_high = 0.0, 100.0
            A = y.new_tensor(base_low)
            B = y.new_tensor(base_high)
        span = float((B - A).abs().item())
        eps_abs = float(self.y_eps_range_buf.item())
        eps_rel = float(max(0.0, self._y_eps_rel)) * span
        eps = max(eps_abs, eps_rel, 1e-9)
        max_eps = max(1e-9, 0.5 * span - 1e-9)
        eps = min(eps, max_eps)
        eps_val = torch.tensor(eps, device=y.device, dtype=y.dtype)
        denom = B - A + 2.0 * eps_val
        y01 = (y - A + eps_val) / denom
        min_norm = eps_val / denom
        y01 = torch.clamp(y01, min=min_norm, max=1.0 - min_norm)
        return torch.log(y01 / (1.0 - y01))

    def _from_logit_range(self, z: torch.Tensor) -> torch.Tensor:
        A = self.y_low_buf.to(device=z.device, dtype=z.dtype)
        B = self.y_high_buf.to(device=z.device, dtype=z.dtype)
        if (
            not torch.isfinite(A).all().item()
            or not torch.isfinite(B).all().item()
            or not bool((B > A).all().item())
        ):
            base_low = float(self._y_low)
            base_high = float(self._y_high)
            if (not math.isfinite(base_low)) or (not math.isfinite(base_high)) or (
                base_high <= base_low
            ):
                base_low, base_high = 0.0, 100.0
            A = z.new_tensor(base_low)
            B = z.new_tensor(base_high)
        span = float((B - A).abs().item())
        eps_abs = float(self.y_eps_range_buf.item())
        eps_rel = float(max(0.0, self._y_eps_rel)) * span
        eps = max(eps_abs, eps_rel, 1e-9)
        max_eps = max(1e-9, 0.5 * span - 1e-9)
        eps = min(eps, max_eps)
        eps_val = torch.tensor(eps, device=z.device, dtype=z.dtype)
        y = torch.sigmoid(z) * (B - A + 2.0 * eps_val) + (A - eps_val)
        return torch.clamp(y, min=A, max=B)

    def _to_zscore(self, y: torch.Tensor) -> torch.Tensor:
        if not self.has_valid_y_stats():
            return y
        mu = self.y_mean.to(device=y.device, dtype=y.dtype)
        sd = self.y_std.to(device=y.device, dtype=y.dtype)
        eps = float(self.y_eps.item()) if hasattr(self, "y_eps") else 1e-06
        sd = torch.clamp(sd, min=eps)
        return (y - mu) / sd

    def _from_zscore(self, z: torch.Tensor) -> torch.Tensor:
        if not self.has_valid_y_stats():
            return z
        mu = self.y_mean.to(device=z.device, dtype=z.dtype)
        sd = self.y_std.to(device=z.device, dtype=z.dtype)
        eps = float(self.y_eps.item()) if hasattr(self, "y_eps") else 1e-06
        sd = torch.clamp(sd, min=eps)
        return z * sd + mu

    def forward(
        self,
        features: torch.Tensor,
        labels_flat: Optional[torch.Tensor] = None,
        net_loss: Optional[nn.Module] = None,
        global_loss: Optional[nn.Module] = None,
        local_loss: Optional[nn.Module] = None,
        loss_weights: Optional[Union[Tuple[float, float], LossWeightPolicy]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        features = self._normalize_inputs(features)
        if features.ndim == 3 and features.shape[1] == 1:
            features = features.view(features.shape[0], -1)
        assert features.ndim == 2 and features.shape[1] == self.in_dim
        b = features.shape[0]
        device = self._device
        amp_enabled = device.type != "cpu"
        base_dtype = next(self.local_net.parameters()).dtype
        infer_mode = labels_flat is None or (
            net_loss is None and global_loss is None and (local_loss is None)
        )
        try:
            self.x_seen_elems += torch.tensor(
                features.numel(),
                device=self.x_seen_elems.device,
                dtype=self.x_seen_elems.dtype,
            )
        except Exception:
            pass
        num_slices = (b + self.microbatch - 1) // self.microbatch
        token_chunks: List[torch.Tensor] = []
        context_chunks: List[torch.Tensor] = []
        if not infer_mode:
            self.local_net.train()
            self.global_net.train()
            for idx in range(num_slices):
                s = idx * self.microbatch
                e = min(b, (idx + 1) * self.microbatch)
                x_slice = features[s:e].to(
                    device, dtype=base_dtype, non_blocking=True
                )
                with AutoCast.float(device, enabled=amp_enabled):
                    out: Payload = self.local_net.forward(x_slice)
                if (not torch.isfinite(out.tokens).all()) or (
                    not torch.isfinite(out.context).all()
                ):
                    x_f32 = x_slice.to(dtype=torch.float32)
                    with AutoCast.float(device, enabled=False):
                        out = self.local_net(x_f32)
                if (not torch.isfinite(out.tokens).all()) or (
                    not torch.isfinite(out.context).all()
                ):
                    raise RuntimeError(
                        "[local_net.forward] produced non-finite tokens/context in training"
                    )
                out_tokens = (
                    out.tokens
                    if out.tokens.dtype == base_dtype
                    else out.tokens.to(base_dtype)
                )
                out_context = (
                    out.context
                    if out.context.dtype == base_dtype
                    else out.context.to(base_dtype)
                )
                token_chunks.append(out_tokens)
                context_chunks.append(out_context)
        else:
            self.local_net.eval()
            self.global_net.eval()
            for idx in range(num_slices):
                s = idx * self.microbatch
                e = min(b, (idx + 1) * self.microbatch)
                x_slice = features[s:e].to(
                    device, dtype=base_dtype, non_blocking=True
                )
                with contextlib.ExitStack() as stack:
                    stack.enter_context(inference(self.local_net))
                    stack.enter_context(
                        AutoCast.float(device, enabled=amp_enabled)
                    )
                    out = self.local_net.forward(x_slice)
                if (not torch.isfinite(out.tokens).all()) or (
                    not torch.isfinite(out.context).all()
                ):
                    x_f32 = x_slice.to(dtype=torch.float32)
                    with inference(self.local_net):
                        with AutoCast.float(device, enabled=False):
                            out = self.local_net(x_f32)
                if not torch.isfinite(out.tokens).all() or not torch.isfinite(out.context).all():
                    out_tokens = torch.nan_to_num(
                        out.tokens, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    out_context = torch.nan_to_num(
                        out.context, nan=0.0, posinf=0.0, neginf=0.0
                    )
                else:
                    out_tokens = out.tokens
                    out_context = out.context
                token_chunks.append(
                    out_tokens
                    if out_tokens.dtype == base_dtype
                    else out_tokens.to(base_dtype)
                )
                context_chunks.append(
                    out_context
                    if out_context.dtype == base_dtype
                    else out_context.to(base_dtype)
                )
        tokens = torch.cat(token_chunks, dim=0).to(device=device, dtype=base_dtype)
        context = torch.cat(context_chunks, dim=0).to(device=device, dtype=base_dtype)
        if (not torch.isfinite(tokens).all()) or (not torch.isfinite(context).all()):
            if self.training:
                raise RuntimeError(
                    "[concat] non-finite tokens/context after local_net forward"
                )
            tokens = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
            context = torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        assembled = context.view(b, -1)
        if not torch.isfinite(assembled).all():
            if self.training:
                raise RuntimeError("[assembled] non-finite in training")
            assembled = torch.nan_to_num(assembled, nan=0.0, posinf=0.0, neginf=0.0)
        if self.is_norm_linear and self.linear_branch is not None:
            bl = self.linear_branch(features.to(device, dtype=assembled.dtype))
            assembled = assembled + bl
        t = tokens
        if not torch.isfinite(t).all():
            if self.training:
                raise RuntimeError("[tokens] non-finite before centering in training")
            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        tokens = t
        t32 = t.to(torch.float32)
        tokens_centered = (t32 - t32.mean(dim=1, keepdim=True)).to(dtype=t.dtype)
        if infer_mode:
            with inference(self.global_net):
                with AutoCast.float(device, enabled=amp_enabled):
                    refined_tokens = self.global_net.forward(tokens_centered)
            if not torch.isfinite(refined_tokens).all():
                tokens_centered = tokens_centered.to(dtype=torch.float32)
                with inference(self.global_net):
                    with AutoCast.float(device, enabled=False):
                        refined_tokens = self.global_net.forward(tokens_centered)
            decode_tokens = refined_tokens.detach().clone()
            with inference(self.local_net):
                with AutoCast.float(device, enabled=amp_enabled):
                    residual_context = self.local_net.decode(
                        decode_tokens, apply_norm=True
                    )
            if not torch.isfinite(residual_context).all():
                decode_tokens = decode_tokens.to(dtype=torch.float32)
                with inference(self.local_net):
                    with AutoCast.float(device, enabled=False):
                        residual_context = self.local_net.decode(
                            decode_tokens, apply_norm=True
                        )
        else:
            # Train path: autograd ON
            with torch.enable_grad():
                with AutoCast.float(device, enabled=amp_enabled):
                    refined_tokens = self.global_net.forward(tokens_centered)
                if not torch.isfinite(refined_tokens).all():
                    tokens_centered = tokens_centered.to(dtype=torch.float32)
                    with AutoCast.float(device, enabled=False):
                        refined_tokens = self.global_net.forward(tokens_centered)
                with AutoCast.float(device, enabled=amp_enabled):
                    residual_context = self.local_net.decode(
                        refined_tokens, apply_norm=True
                    )
                if not torch.isfinite(residual_context).all():
                    refined_tokens = refined_tokens.to(dtype=torch.float32)
                    with AutoCast.float(device, enabled=False):
                        residual_context = self.local_net.decode(
                            refined_tokens, apply_norm=True
                        )
        residual = residual_context.view(b, -1)
        if not torch.isfinite(residual).all():
            if self.training:
                raise RuntimeError("[residual] non-finite in training")
            residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
        y_hat_z = assembled + residual
        if residual.dtype != assembled.dtype:
            residual = residual.to(dtype=assembled.dtype)
            y_hat_z = assembled + residual
        eps = (
            float(self.y_eps_range_buf.item())
            if hasattr(self, "y_eps_range_buf")
            else 1e-3
        )
        eps = min(max(eps, 1e-9), 1.0 - 1e-9)
        # Guard against configurations that set epsilon above 0.5. The
        # theoretical logit bounds are symmetric only while `eps <= 0.5`, and
        # larger values invert the clamp range which would raise at runtime.
        eps = min(eps, 0.5 - 1e-9)
        z_max = math.log((1.0 - eps) / eps)
        y_hat_z = y_hat_z.clamp(min=-z_max, max=z_max)
        if not torch.isfinite(y_hat_z).all():
            if self.training:
                raise RuntimeError("[y_hat_z] non-finite in training")
            y_hat_z = torch.nan_to_num(y_hat_z, nan=0.0, posinf=0.0, neginf=0.0)
        is_cls_loss = (
            isinstance(net_loss, (nn.CrossEntropyLoss, nn.NLLLoss))
            if net_loss is not None
            else False
        )
        y_hat_out = y_hat_z
        if not is_cls_loss:
            if self._loss_space == "logit":
                y_hat_out = self._from_logit_range(y_hat_z)
            elif self._loss_space == "z":
                y_hat_out = self._from_zscore(y_hat_z)
            else:
                y_hat_out = y_hat_z

        if self._calib_enable and (not is_cls_loss):
            cs = self.calib_scale.to(dtype=y_hat_out.dtype, device=y_hat_out.device).view(1)
            cb = self.calib_bias.to(dtype=y_hat_out.dtype, device=y_hat_out.device).view(1)
            y_hat_out = y_hat_out * cs + cb
        loss_val: Optional[torch.Tensor] = None
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
            base_tensor = (
                y_hat_z if self._loss_space in {"logit", "z"} else y_hat_out
            )
            total = base_tensor.new_tensor(0.0, dtype=base_tensor.dtype)
            top_component: Optional[torch.Tensor] = None
            bottom_component: Optional[torch.Tensor] = None
            if self._loss_space == "logit":
                tgt = self._to_logit_range(
                    labels_flat.to(device=y_hat_z.device, dtype=y_hat_z.dtype)
                )
                y_top = y_hat_z
                y_bot = assembled
                if global_loss is not None:
                    top_component = global_loss(y_top, tgt)
                    total = total + weights[0] * top_component
                if local_loss is not None:
                    bottom_component = local_loss(y_bot, tgt)
                    total = total + weights[1] * bottom_component
            elif self._loss_space == "z" and self.has_valid_y_stats():
                tgt_z = self._to_zscore(
                    labels_flat.to(device=y_hat_z.device, dtype=y_hat_z.dtype)
                )
                y_top = y_hat_z
                y_bot = assembled
                if global_loss is not None:
                    top_component = global_loss(y_top, tgt_z)
                    total = total + weights[0] * top_component
                if local_loss is not None:
                    bottom_component = local_loss(y_bot, tgt_z)
                    total = total + weights[1] * bottom_component
            else:
                tgt_y = labels_flat.to(
                    device=y_hat_out.device, dtype=y_hat_out.dtype
                )
                y_top = y_hat_out
                if self.has_valid_y_stats():
                    y_bot = self._from_zscore(assembled)
                else:
                    y_bot = assembled
                if global_loss is not None:
                    top_component = global_loss(y_top, tgt_y)
                    total = total + weights[0] * top_component
                if local_loss is not None:
                    bottom_component = local_loss(y_bot, tgt_y)
                    total = total + weights[1] * bottom_component
            if controller is not None:
                controller.update(top_component, bottom_component)
            loss_val = total
        elif net_loss is not None and labels_flat is not None:
            if is_cls_loss:
                tgt = labels_flat.to(device=y_hat_out.device).long()
                loss_val = net_loss(y_hat_out, tgt)
            elif self._loss_space == "logit":
                tgt = self._to_logit_range(
                    labels_flat.to(device=y_hat_z.device, dtype=y_hat_z.dtype)
                )
                loss_val = net_loss(y_hat_z, tgt)
            elif self._loss_space == "z" and self.has_valid_y_stats():
                tgt = self._to_zscore(
                    labels_flat.to(device=y_hat_z.device, dtype=y_hat_z.dtype)
                )
                loss_val = net_loss(y_hat_z, tgt)
            else:
                loss_val = net_loss(
                    y_hat_out,
                    labels_flat.to(
                        device=y_hat_out.device, dtype=y_hat_out.dtype
                    ),
                )
        if (
            not is_cls_loss
            and isinstance(loss_val, torch.Tensor)
            and self._range_penalty_lambda > 0.0
        ):
            y_low = self.y_low_buf.to(device=y_hat_out.device, dtype=y_hat_out.dtype)
            y_high = self.y_high_buf.to(device=y_hat_out.device, dtype=y_hat_out.dtype)
            penalty_hi = F.relu(y_hat_out - y_high)
            penalty_lo = F.relu(y_low - y_hat_out)
            penalty = (penalty_hi.pow(2) + penalty_lo.pow(2)).mean()
            if torch.isfinite(penalty).all():
                loss_val = loss_val + self._range_penalty_lambda * penalty
        if (
            not is_cls_loss
            and self._z_reg_lambda > 0.0
            and isinstance(loss_val, torch.Tensor)
        ):
            loss_val = loss_val + self._z_reg_lambda * y_hat_z.pow(2).mean()
        return (y_hat_out.view(b, *self.out_shape), loss_val)

    def stats(self) -> dict:
        try:
            x_seen = float(self.x_seen_elems.item())
        except Exception:
            x_seen = 0.0
        y_cnt = self.y_count.to(torch.float64)
        y_seen = float(y_cnt.sum().item())
        denom = y_cnt.sum().clamp_min(1.0)
        y_avg = float((self.y_sum.sum() / denom).item())
        return {
            "accumulated_x": x_seen,
            "accumulated_y": y_seen,
            "y_min": self.y_min.clone(),
            "y_max": self.y_max.clone(),
            "y_mean": self.y_mean.clone(),
            "y_std": self.y_std.clone(),
            "y_avg": y_avg,
            "y_count": self.y_count.clone(),
        }

    @staticmethod
    def flatten_labels(
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
    def unflatten_labels(
        flat: torch.Tensor, shape: Sequence[int]
    ) -> torch.Tensor:
        return flat.view(flat.shape[0], *shape)
