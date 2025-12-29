# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Protocol, Tuple, Union, cast

import torch
import torch.nn as nn

try:
    from tensordict import TensorDictBase
except ImportError:
    class TensorDictBase:  # type: ignore[no-redef]
        pass


_LOGGER = logging.getLogger(__name__)

from ..core.compat import StochasticDepth, is_meta_or_fake_tensor
from ..core.precision import Autocast
from .primitives import (CrossAttention, DilatedAttention, PatchAttention,
                         Retention, norm_layer)

# -------------------------
# Small internal utilities
# -------------------------

def _infer_module_device(module: nn.Module, fallback: torch.device) -> torch.device:
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


class _ControllerChunkRunner:

    __slots__ = ("backbone", "device", "meta", "amp_enabled")

    def __init__(
        self,
        backbone: nn.Module,
        *,
        device: torch.device,
        meta: Any,
        amp_enabled: bool,
    ) -> None:
        self.backbone = backbone
        self.device = device
        self.meta = meta
        self.amp_enabled = bool(amp_enabled)

    def __call__(self, chunk: torch.Tensor) -> torch.Tensor:
        with (
            Autocast.float(self.device, metadata=self.meta)
            if self.amp_enabled
            else Autocast.suspend(self.device)
        ):
            out, _ = self.backbone(chunk)
        return cast(torch.Tensor, out)


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
    if not bool(enabled):
        return t
    if not (t.is_floating_point() or t.is_complex()):
        return t
    # NOTE: Avoid in-place sanitization even in inference. In-place writes can
    # introduce subtle aliasing/anomaly issues when tensors are views or shared
    # across pipeline stages (e.g., Fuser -> Enhancer). Always return a new tensor.
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

        match mode_l:
            case "ss":
                return self.cross_s(spatial_tokens, temporal_tokens)
            case "tt":
                return self.cross_t(temporal_tokens, spatial_tokens)
            case _:
                pass

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

    from ..core.system import Memory as _Mem

    try:
        host_free = int(_Mem.available())
    except Exception:
        host_free = None

    if dev_t in {"cuda", "xpu", "mps"}:
        with contextlib.suppress(Exception):
            free, _ = _Mem.device_mem_get_info(device)
            if free is not None:
                dev_free = int(free)

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


class LossWeightPolicy(Protocol):
    def weights(self) -> Tuple[float, float]: ...

    def update(
        self,
        top_loss: Optional[torch.Tensor],
        bottom_loss: Optional[torch.Tensor],
    ) -> None:
        raise NotImplementedError
