# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any, List, Optional, Protocol, Sequence, Tuple, Union, cast

import torch
from torch import nn

from ..utils.compat import patch_torch
from ..utils.optimization import (
    AutoCast,
    compile,
    inference,
)


try:
    # Prefer torch.compiler.disable (PyTorch ≥2.5)
    _disable_torch_compile = torch.compiler.disable  # type: ignore[attr-defined]
except Exception:
    try:
        # Fallback for PyTorch 2.0–2.4
        import torch._dynamo as _dynamo  # type: ignore

        _disable_torch_compile = _dynamo.disable  # type: ignore[attr-defined]
    except Exception:

        def _disable_torch_compile(fn=None, *, recursive=False):  # type: ignore[no-untyped-def]
            if fn is None:
                return lambda real_fn: real_fn
            return fn


if not hasattr(torch, "compiler"):
    class _TorchCompilerNamespace:
        @staticmethod
        def disable(fn=None, *, recursive=False):  # type: ignore[no-untyped-def]
            return _disable_torch_compile(fn, recursive=recursive)


    torch.compiler = _TorchCompilerNamespace()  # type: ignore[attr-defined]
elif not hasattr(torch.compiler, "disable"):

    def _compiler_disable_passthrough(fn=None, *, recursive=False):  # type: ignore[no-untyped-def]
        return _disable_torch_compile(fn, recursive=recursive)


    torch.compiler.disable = _compiler_disable_passthrough  # type: ignore[attr-defined]

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
        self.coord_dim = int(coord_dim)
        self.d_model = int(d_model)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(p=drop_path, mode="row")
        self.norm1 = norm_layer(norm_type, self.d_model)
        self.attn = PatchAttention(
            self.d_model, nhead, coord_dim=self.coord_dim
        )
        self.norm2 = norm_layer(norm_type, self.d_model)
        hid = int(self.d_model * mlp_ratio * (2.0 / 3.0))
        self.ffn = SwiGLU(
            self.d_model, hid, out_dim=self.d_model, dropout=dropout
        )
        self._ln_materialized = False

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # (0) 입력/마스크 메타 유입 즉시 차단
        if x.is_meta or coords.is_meta:
            raise RuntimeError("meta/fake tensor reached PointTransformer.forward")
        if coords.shape[:2] != x.shape[:2] or coords.size(-1) != self.coord_dim:
            raise ValueError(
                f"coords must be (B, N, {self.coord_dim}), got {tuple(coords.shape)} vs x {tuple(x.shape)}"
            )
        x = x.contiguous()
        coords = coords.contiguous()
        if attn_mask is not None:
            attn_mask = attn_mask.contiguous()
            if getattr(attn_mask, "is_meta", False):
                raise RuntimeError("attn_mask is meta before attention")

        # (1) 첫 호출 시 LayerNorm 파라미터가 meta면 즉시 실체화(materialize)
        if not getattr(self, "_ln_materialized", False):

            def _materialize_ln_(ln: nn.LayerNorm, ref: torch.Tensor) -> None:
                if not isinstance(ln, nn.LayerNorm):
                    return
                dev = ref.device
                target_dtype = torch.float32 if dev.type == "cpu" else ref.dtype
                if getattr(getattr(ln, "weight", None), "is_meta", False):
                    ln.weight = nn.Parameter(
                        torch.ones(ln.normalized_shape, device=dev, dtype=target_dtype)
                    )
                if getattr(getattr(ln, "bias", None), "is_meta", False):
                    ln.bias = nn.Parameter(
                        torch.zeros(ln.normalized_shape, device=dev, dtype=target_dtype)
                    )

            _materialize_ln_(self.norm1, x)
            _materialize_ln_(self.norm2, x)
            self._ln_materialized = True

        # (2) CPU/eager safety: LN 전/후 meta 가드 + CPU는 fp32 강제
        _x = x
        if isinstance(_x, torch.Tensor) and getattr(_x, "is_meta", False):
            raise RuntimeError("x is meta before LayerNorm")
        if _x.device.type == "cpu" and _x.is_floating_point() and _x.dtype != torch.float32:
            _x = _x.float()
        _x = self.norm1(_x)
        if isinstance(_x, torch.Tensor) and getattr(_x, "is_meta", False):
            raise RuntimeError("x is meta after LayerNorm")
        if _x.device.type == "cpu" and x.is_floating_point() and x.dtype != torch.float32:
            _x = _x.to(x.dtype)
        y = self.attn(_x, coords, attn_mask=attn_mask)
        x = x + self.drop_path(self.dropout(y))

        # (3) norm2 경로도 동일 보호
        _x2 = x
        if getattr(_x2, "is_meta", False):
            raise RuntimeError("x is meta before LayerNorm(norm2)")
        if (
            _x2.device.type == "cpu"
            and _x2.is_floating_point()
            and _x2.dtype != torch.float32
        ):
            _x2 = _x2.float()
        _x2 = self.norm2(_x2)
        if getattr(_x2, "is_meta", False):
            raise RuntimeError("x is meta after LayerNorm(norm2)")
        if (
            _x2.device.type == "cpu"
            and x.is_floating_point()
            and x.dtype != torch.float32
        ):
            _x2 = _x2.to(x.dtype)
        x = x + self.drop_path(self.dropout(self.ffn(_x2)))
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

@torch.compiler.disable(recursive=True)  # type: ignore[attr-defined]
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

    @_disable_torch_compile
    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        if x.is_meta:
            raise RuntimeError("meta/fake tensor reached TemporalEncoderBlock.forward")
        x = x.contiguous()
        if causal_mask is not None:
            causal_mask = causal_mask.contiguous()
        h, state = self.retention(
            self.norm1(x), attn_mask=causal_mask, state=state
        )
        x = x + self.drop_path(self.dropout(h))
        x = x + self.drop_path(self.dropout(self.ffn(self.norm2(x))))
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
        x = self.norm(x)
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
            fused = self.mix(self.mix_norm(base))
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
        fused = self.mix(self.mix_norm(base))
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
            self.norm1(x), attn_mask=causal_mask, state=state
        )
        x = x + self.drop_path(self.dropout(h))
        x = x + self.drop_path(self.dropout(self.ffn(self.norm2(x))))
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
        tokens = self.norm(tokens)
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
        spatial_out = self.spatial_encoder(spatial_tokens, coords)
        temporal_out = self.temporal_encoder(temporal_tokens)
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
        tokens = self.norm(tokens)
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
            tokens = self.norm(tokens)
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

    def set_input_scale_method(self, method: str = "standard") -> None:
        self._input_scale_method = (
            "standard" if str(method).lower() == "standard" else "none"
        )

    def forward(
        self,
        features: torch.Tensor,
        *args: Any,
        labels_flat: Optional[torch.Tensor] = None,
        net_loss: Optional[nn.Module] = None,
        global_loss: Optional[nn.Module] = None,
        local_loss: Optional[nn.Module] = None,
        loss_weights: Optional[
            Union[Tuple[float, float], LossWeightPolicy]
        ] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
                ctx = AutoCast.float(device) if amp_enabled else AutoCast.suspend(device)
                with ctx:
                    out: Payload = self.local_net(x_slice)
                if (not torch.isfinite(out.tokens).all()) or (
                    not torch.isfinite(out.context).all()
                ):
                    x_f32 = x_slice.to(dtype=torch.float32)
                    with AutoCast.suspend(device):
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
                        AutoCast.float(device)
                        if amp_enabled
                        else AutoCast.suspend(device)
                    )
                    out = self.local_net(x_slice)
                if (not torch.isfinite(out.tokens).all()) or (
                    not torch.isfinite(out.context).all()
                ):
                    x_f32 = x_slice.to(dtype=torch.float32)
                    with inference(self.local_net):
                        with AutoCast.suspend(device):
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
                with (
                    AutoCast.float(device)
                    if amp_enabled
                    else AutoCast.suspend(device)
                ):
                    refined_tokens = self.global_net(tokens_centered)
            if not torch.isfinite(refined_tokens).all():
                tokens_centered = tokens_centered.to(dtype=torch.float32)
                with inference(self.global_net):
                    with AutoCast.suspend(device):
                        refined_tokens = self.global_net(tokens_centered)
            decode_tokens = refined_tokens.detach().clone()
            with inference(self.local_net):
                with (
                    AutoCast.float(device)
                    if amp_enabled
                    else AutoCast.suspend(device)
                ):
                    residual_context = self.local_net.decode(
                        decode_tokens, apply_norm=True
                    )
            if not torch.isfinite(residual_context).all():
                decode_tokens = decode_tokens.to(dtype=torch.float32)
                with inference(self.local_net):
                    with AutoCast.suspend(device):
                        residual_context = self.local_net.decode(
                            decode_tokens, apply_norm=True
                        )
        else:
            # Train path: autograd ON
            with torch.enable_grad():
                with (
                    AutoCast.float(device)
                    if amp_enabled
                    else AutoCast.suspend(device)
                ):
                    refined_tokens = self.global_net(tokens_centered)
                if not torch.isfinite(refined_tokens).all():
                    tokens_centered = tokens_centered.to(dtype=torch.float32)
                    with AutoCast.suspend(device):
                        refined_tokens = self.global_net(tokens_centered)
                with (
                    AutoCast.float(device)
                    if amp_enabled
                    else AutoCast.suspend(device)
                ):
                    residual_context = self.local_net.decode(
                        refined_tokens, apply_norm=True
                    )
                if not torch.isfinite(residual_context).all():
                    refined_tokens = refined_tokens.to(dtype=torch.float32)
                    with AutoCast.suspend(device):
                        residual_context = self.local_net.decode(
                            refined_tokens, apply_norm=True
                        )
        residual = residual_context.view(b, -1)
        if not torch.isfinite(residual).all():
            if self.training:
                raise RuntimeError("[residual] non-finite in training")
            residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
        y_hat = assembled + residual
        if residual.dtype != assembled.dtype:
            residual = residual.to(dtype=assembled.dtype)
            y_hat = assembled + residual
        if not torch.isfinite(y_hat).all():
            if self.training:
                raise RuntimeError("[y_hat] non-finite in training")
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=0.0, neginf=0.0)
        is_cls_loss = (
            isinstance(net_loss, (nn.CrossEntropyLoss, nn.NLLLoss))
            if net_loss is not None
            else False
        )
        y_hat_out = y_hat
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
            tgt = labels_flat.to(device=y_hat_out.device, dtype=y_hat_out.dtype)
            total = y_hat_out.new_tensor(0.0, dtype=y_hat_out.dtype)
            top_component: Optional[torch.Tensor] = None
            bottom_component: Optional[torch.Tensor] = None
            y_top = y_hat_out
            y_bot = assembled.to(device=y_hat_out.device, dtype=y_hat_out.dtype)
            if global_loss is not None:
                top_component = global_loss(y_top, tgt)
                total = total + weights[0] * top_component
            if local_loss is not None:
                bottom_component = local_loss(y_bot, tgt)
                total = total + weights[1] * bottom_component
            if controller is not None:
                controller.update(top_component, bottom_component)
            loss_val = total
        elif net_loss is not None and labels_flat is not None:
            if is_cls_loss:
                tgt = labels_flat.to(device=y_hat_out.device).long()
                loss_val = net_loss(y_hat_out, tgt)
            else:
                tgt = labels_flat.to(
                    device=y_hat_out.device, dtype=y_hat_out.dtype
                )
                loss_val = net_loss(y_hat_out, tgt)
        return (y_hat_out.view(b, *self.out_shape), loss_val)

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
