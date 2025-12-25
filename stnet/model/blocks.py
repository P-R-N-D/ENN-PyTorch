# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import logging
import math
import threading
import weakref
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

from ..backend.compat import (
    StochasticDepth,
    graph_break,
    is_meta_or_fake_tensor,
    torch_no_compile,
)
from ..backend.system import empty_device_cache
from ..backend.casting import env_first_int, env_int
from ..data.pipeline import resolve_feature_key, resolve_label_key
from ..functional.profiler import FLOP_PROFILER
from ..model.fused import Autocast, Gradient
from .primitives import (
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


class _ControllerChunkRunner:
    """Callable used by Context.run microbatch executor.

    Kept at module scope to avoid nested function definitions (cleaner pickling
    and friendlier to free-threaded/no-GIL builds).
    """

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
