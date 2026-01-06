# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import contextlib
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn

try:
    from torch.utils.checkpoint import checkpoint as _checkpoint
except Exception:
    _checkpoint = None
from ..core.distributed import _unshard_fsdp_module
try:
    from .layers import _HAS_FLEX_ATTENTION as _STNET_HAS_FLEX_ATTENTION
except Exception:
    _STNET_HAS_FLEX_ATTENTION = False

from ..core.compat import StochasticDepth, is_meta_or_fake_tensor
from .layers import CrossAttention, DilatedAttention, Retention, norm_layer


_LOGGER = logging.getLogger(__name__)

_MODELING_TYPE_ALIASES: dict[str, str] = {
    "ss": "ss",
    "spatial": "ss",
    "sxs": "ss",
    "tt": "tt",
    "temporal": "tt",
    "txt": "tt",
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


def _is_export_or_trace() -> bool:
    with contextlib.suppress(Exception):
        if torch.jit.is_tracing() or torch.jit.is_scripting():
            return True
    with contextlib.suppress(Exception):
        if getattr(torch, "_dynamo", None) is not None and torch._dynamo.is_compiling():
            return True
    with contextlib.suppress(Exception):
        if getattr(torch, "compiler", None) is not None and torch.compiler.is_compiling():
            return True
    with contextlib.suppress(Exception):
        if getattr(torch, "onnx", None) is not None and hasattr(torch.onnx, "is_in_onnx_export"):
            if torch.onnx.is_in_onnx_export():
                return True
    return False

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


def _autofit_microbatch(
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
            free, _ = _Mem.mem_get_info(device)
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


def _coerce_tensor(
    t: torch.Tensor,
    *args: Any,
    enabled: bool,
    inplace: bool,
) -> torch.Tensor:
    if not bool(enabled):
        return t
    if not (t.is_floating_point() or t.is_complex()):
        return t
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def _prealloc_microbatch(
    inp: torch.Tensor,
    microbatch: int,
    run_fn: Callable[[torch.Tensor], Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    *args: Any,
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
        mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.mode = str(mode or "temporal").strip().lower()
        self.norm1 = norm_layer(norm_type, self.d_model)
        self.retention = Retention(self.d_model, self.nhead, mode=self.mode)
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
        mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        if is_meta_or_fake_tensor(x):
            raise RuntimeError("meta/fake tensor reached RetNet.forward")
        x = x.contiguous()
        if causal_mask is not None:
            causal_mask = causal_mask.contiguous()
        eff_mode = mode if mode is not None else getattr(self, "mode", None)
        h, new_state = self.retention(
            self.norm1(x),
            attn_mask=causal_mask,
            state=state,
            mode=eff_mode,
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
        cross: Optional[Sequence[nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if cross is None:
            cross_mods: List[nn.Module] = [
                CrossAttention(d_model, nhead, dropout=dropout, norm_type=norm_type),
                CrossAttention(d_model, nhead, dropout=dropout, norm_type=norm_type),
            ]
        else:
            cross_mods = list(cross)
            if len(cross_mods) != 2:
                raise ValueError(f"CrossTransformer expects exactly 2 cross modules, got {len(cross_mods)}")
        self.cross = nn.ModuleList(cross_mods)
        self.cross_s = self.cross[0]
        self.cross_t = self.cross[1]
        self.mix_norm = norm_layer(norm_type, 2 * d_model)
        hid = int(2 * d_model * mlp_ratio * (2.0 / 3.0))
        from .activations import SwiGLU
        self.mix = SwiGLU(2 * d_model, hid, out_dim=d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(p=drop_path, mode="row")
        self._fixed_mode: Optional[str] = getattr(self, "modeling_type", None)

    def forward(
        self,
        tokens_a: torch.Tensor,
        tokens_b: torch.Tensor,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        spatial_tokens, temporal_tokens = tokens_a, tokens_b
        if spatial_tokens.dim() != 3 or temporal_tokens.dim() != 3:
            raise ValueError(
                f"CrossTransformer expects 3D tensors, got "
                f"a={tuple(spatial_tokens.shape)}, b={tuple(temporal_tokens.shape)}"
            )
        Bs, Ns, Ds = spatial_tokens.shape
        Bt, Nt, Dt = temporal_tokens.shape
        if Bs != Bt:
            raise ValueError(f"CrossTransformer batch mismatch: a B={Bs}, b B={Bt}")
        if Ds != Dt:
            raise ValueError(f"CrossTransformer hidden dim mismatch: a D={Ds}, b D={Dt}")

        spatial_tokens = spatial_tokens.contiguous()
        temporal_tokens = temporal_tokens.contiguous()

        requested = self._fixed_mode if self._fixed_mode is not None else (mode or "st")
        mode_l = _coerce_modeling_types(requested)

        def _impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            match mode_l:
                case "ss":
                    return self.cross[0](a, b)
                case "tt":
                    return self.cross[1](b, a)
                case _:
                    pass
            s_context = self.cross[0](a, b)
            t_context = self.cross[1](b, a)
            t_summary = t_context.mean(dim=1, keepdim=True).expand_as(s_context)
            base_s = torch.cat([s_context, t_summary], dim=-1)
            fused_s = self.mix(self.mix_norm(base_s))
            out_s = s_context + self.drop_path(self.dropout(fused_s))
            s_summary = s_context.mean(dim=1, keepdim=True).expand_as(t_context)
            base_t = torch.cat([t_context, s_summary], dim=-1)
            fused_t = self.mix(self.mix_norm(base_t))
            out_t = t_context + self.drop_path(self.dropout(fused_t))
            return torch.cat([out_s, out_t], dim=1)

        def _ckpt_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            if torch.is_grad_enabled():
                _unshard_fsdp_module(self)
            return _impl(a, b)

        do_ckpt = (
            self.training
            and torch.is_grad_enabled()
            and _checkpoint is not None
            and not _is_export_or_trace()
        )
        if do_ckpt:
            return cast(
                torch.Tensor,
                _checkpoint(_ckpt_impl, spatial_tokens, temporal_tokens, use_reentrant=True, preserve_rng_state=True),
            )
        return _impl(spatial_tokens, temporal_tokens)


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
        self._ckpt_enabled = True
        self._ckpt_reentrant = True
        self._ckpt_min_bytes = int(64 * 1024 * 1024)

    @property
    def using(self) -> str:
        return self._using

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_w: Optional[torch.Tensor] = None
        out = x
        need_transpose_fallback = (
            out.dim() == 3
            and (self.batch_first is False)
            and out.shape[0] != out.shape[1]
        )
        layout_batch_first = self.batch_first
        if need_transpose_fallback:
            out = out.transpose(0, 1)
            layout_batch_first = True

        do_ckpt = (
            self.training
            and torch.is_grad_enabled()
            and _checkpoint is not None
            and (not bool(need_weights))
            and bool(getattr(self, "_ckpt_enabled", True))
            and not _is_export_or_trace()
        )
        if do_ckpt:
            est_bytes = 0
            with contextlib.suppress(Exception):
                if layout_batch_first:
                    B = int(out.shape[0]); L = int(out.shape[1]); D = int(out.shape[2])
                else:
                    L = int(out.shape[0]); B = int(out.shape[1]); D = int(out.shape[2])
                H = int(getattr(self, "nhead", 1) or 1)
                bytes_e = int(out.element_size())
                base_bytes = int(B) * int(L) * int(D) * int(bytes_e)
                score_bytes = 4 if out.dtype in (torch.float16, torch.bfloat16) else int(bytes_e)
                peak_per_layer = 0
                flex_ok = bool(_STNET_HAS_FLEX_ATTENTION and out.is_cuda and (not bool(need_weights)))
                has_kpm = isinstance(key_padding_mask, torch.Tensor)
                for lyr in self.layers:
                    dilation = int(getattr(lyr, "dilation", 1) or 1)
                    win = getattr(lyr, "window_size", None)
                    is_simple = (dilation == 1) and (win is None)
                    dense_scores = True
                    if out.is_cuda:
                        if flex_ok:
                            dense_scores = False
                        else:
                            if is_simple and (not has_kpm):
                                dense_scores = False
                    scores_bytes = int(B) * int(H) * int(L) * int(L) * int(score_bytes) if dense_scores else 0
                    attn_linear = int(base_bytes) * 5
                    attn_bytes = int(attn_linear) + int(scores_bytes)
                    hidden = 0
                    with contextlib.suppress(Exception):
                        f0 = getattr(lyr, "ffn", None)
                        if isinstance(f0, nn.Sequential) and len(f0) > 0 and hasattr(f0[0], "out_features"):
                            hidden = int(getattr(f0[0], "out_features", 0) or 0)
                    if hidden > 0:
                        ffn_bytes = int(B) * int(L) * (int(2) * int(hidden) + int(D)) * int(bytes_e)
                    else:
                        ffn_bytes = int(base_bytes) * 9
                    layer_saved = int(attn_bytes + ffn_bytes)
                    peak_per_layer = max(int(peak_per_layer), int(layer_saved))
                est_bytes = int(peak_per_layer) * max(1, int(len(self.layers)))
            do_ckpt = bool(est_bytes >= int(getattr(self, "_ckpt_min_bytes", 0) or 0))

        for layer in self.layers:
            if do_ckpt:
                def _f(t: torch.Tensor, _layer: nn.Module = layer) -> torch.Tensor:
                    if torch.is_grad_enabled():
                        _unshard_fsdp_module(self)
                        _unshard_fsdp_module(_layer)
                    y, _ = _layer(
                        t,
                        key_padding_mask=key_padding_mask,
                        need_weights=False,
                        average_attn_weights=False,
                        skip_ffn_checkpoint=True,
                    )
                    return y

                out = cast(
                    torch.Tensor,
                    _checkpoint(_f, out, use_reentrant=True, preserve_rng_state=True),
                )
                attn_w = None
            else:
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
