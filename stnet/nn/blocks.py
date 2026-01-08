# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import contextlib
from importlib import import_module
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn

from ..core.distributed import _unshard_fsdp_module
from ..core.compat import StochasticDepth, is_meta_or_fake_tensor
from ..core.graph import coerce_checkpoint, is_export_or_trace
from .layers import CrossAttention, DilatedAttention, Retention, norm_layer

_STNET_HAS_FLEX_ATTENTION = getattr(
    import_module(".layers", __package__), "_HAS_FLEX_ATTENTION", False
)


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


def _size_of_retnet(x: torch.Tensor, blk0: nn.Module, *args: Any, mode: str) -> int:
    if not isinstance(x, torch.Tensor) or x.dim() != 3:
        return 0
    B, L, D = map(int, x.shape)
    if B <= 0 or L <= 0 or D <= 0:
        return 0
    bytes_e = int(x.element_size())
    base_bytes = int(B) * int(L) * int(D) * int(bytes_e)
    ret_factor = 8 if str(mode or "temporal").strip().lower() == "spatial" else 6
    ffn = getattr(blk0, "ffn", None)
    hid = (
        getattr(ffn, "hidden_dim", 0) or getattr(ffn, "hid", 0) or int(float(D) * 4.0 * (2.0 / 3.0))
    )
    ffn_bytes = int(B) * int(L) * (int(3) * int(hid) + int(D)) * bytes_e
    return int(base_bytes * ret_factor + ffn_bytes + base_bytes * 2)


def _infer_module_device(module: nn.Module, fallback: torch.device) -> torch.device:
    p = next(module.parameters(), None)
    if p is not None:
        return p.device
    b = next(module.buffers(), None)
    return b.device if b is not None else fallback


def _autofit_microbatch(device: torch.device, hard_max: int, per_sample_bytes: int) -> int:
    if hard_max <= 0 or per_sample_bytes <= 0:
        return 1
    from ..core.system import Memory as _Mem

    try:
        host_free = int(_Mem.available())
    except:
        host_free = None
    dev_free = None
    if device.type in {"cuda", "xpu", "mps"}:
        with contextlib.suppress(Exception):
            dev_free = int(_Mem.mem_get_info(device)[0])
    eff = min(host_free, dev_free) if (host_free and dev_free) else (dev_free or host_free)
    if not eff or eff <= 0:
        return hard_max
    return max(1, min(hard_max, int((eff * 0.35) // max(per_sample_bytes, 1))))


def _coerce_tensor(t: torch.Tensor, *args, enabled: bool, inplace: bool) -> torch.Tensor:
    return (
        torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        if enabled and (t.is_floating_point() or t.is_complex())
        else t
    )


def _prealloc_microbatch(
    inp: torch.Tensor,
    microbatch: int,
    run_fn: Callable,
    *args,
    pad_to: Optional[int] = None,
    out_dtype: Optional[torch.dtype] = None,
    cast_slice: Optional[Callable] = None,
    stage: str = "microbatch",
):
    if inp.ndim < 1:
        raise ValueError(f"{stage}: expected batched input, got {inp.shape}")
    total_b = int(inp.shape[0])
    mb_i = max(1, min(total_b, int(microbatch) if microbatch else total_b))
    pad_i = int(pad_to) if pad_to is not None else None
    if pad_i is not None and (pad_i := max(1, pad_i)) < mb_i:
        raise ValueError(f"{stage}: pad_to {pad_i} < microbatch {mb_i}")
    out_bufs: Optional[List[torch.Tensor]] = None
    pad_buf = None
    for s in range(0, total_b, mb_i):
        x_slice = inp[s : s + mb_i]
        if cast_slice:
            x_slice = cast_slice(x_slice)
        slice_n = int(x_slice.shape[0])
        x_in = x_slice
        did_pad = False
        if pad_i is not None and slice_n < pad_i:
            want_shape = (pad_i, *x_slice.shape[1:])
            if not torch.is_grad_enabled():
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
        outs = (out,) if torch.is_tensor(out) else out
        if not outs:
            raise RuntimeError(f"{stage}: empty output at s={s}")
        processed: List[torch.Tensor] = []
        for j, t in enumerate(outs):
            if not torch.is_tensor(t):
                raise TypeError(f"{stage}: output #{j} invalid")
            y = t[:slice_n] if did_pad else t
            if y.shape[0] != slice_n:
                raise RuntimeError(f"{stage}: output size mismatch")
            if out_dtype and y.dtype != out_dtype:
                y = y.to(dtype=out_dtype)
            processed.append(y)
        if out_bufs is None:
            out_bufs = [y.new_empty((total_b, *y.shape[1:])) for y in processed]
        else:
            if len(out_bufs) != len(processed):
                raise RuntimeError(f"{stage}: arity mismatch")
            for k, (buf, y) in enumerate(zip(out_bufs, processed)):
                if buf.shape[1:] != y.shape[1:]:
                    raise RuntimeError(f"{stage}: shape mismatch output#{k}")
        for buf, y in zip(out_bufs, processed):
            buf[s : s + slice_n].copy_(y)
    if not out_bufs:
        raise RuntimeError(f"{stage}: no outputs")
    return out_bufs[0] if len(out_bufs) == 1 else tuple(out_bufs)


def _coerce_modeling_types(value: Any) -> str:
    mode = str(value).strip().lower()
    normalized = _MODELING_TYPE_ALIASES.get(mode)
    if normalized is None:
        raise ValueError(f"Unsupported modeling type '{value}'")
    return normalized


def stochastic_depth_schedule(drop_path: float, depth: int) -> List[float]:
    if depth <= 0:
        return []
    if drop_path <= 0.0 or depth == 1:
        return [float(drop_path) if depth == 1 else 0.0] * depth
    return [float(i * float(drop_path) / float(depth - 1)) for i in range(depth)]


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
        if is_meta_or_fake_tensor(x) and (not is_export_or_trace()):
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
                raise ValueError(f"CrossTransformer expects 2 modules, got {len(cross_mods)}")
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
            raise ValueError("Expects 3D tensors")
        if not torch.jit.is_tracing():
            if spatial_tokens.size(0) != temporal_tokens.size(0):
                raise ValueError("Batch mismatch")
            if spatial_tokens.size(2) != temporal_tokens.size(2):
                raise ValueError("Hidden dim mismatch")
        mode_l = _coerce_modeling_types(self._fixed_mode or mode or "st")

        def _impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            if mode_l == "ss":
                return self.cross[0](a, b)
            if mode_l == "tt":
                return self.cross[1](b, a)
            s_context = self.cross[0](a, b)
            t_context = self.cross[1](b, a)
            base_s = torch.cat(
                [s_context, t_context.mean(dim=1, keepdim=True).expand_as(s_context)], dim=-1
            )
            fused_s = self.mix(self.mix_norm(base_s))
            out_s = s_context + self.drop_path(self.dropout(fused_s))
            base_t = torch.cat(
                [t_context, s_context.mean(dim=1, keepdim=True).expand_as(t_context)], dim=-1
            )
            fused_t = self.mix(self.mix_norm(base_t))
            out_t = t_context + self.drop_path(self.dropout(fused_t))
            return torch.cat([out_s, out_t], dim=1)

        def _ckpt_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            if torch.is_grad_enabled():
                _unshard_fsdp_module(self)
            return _impl(a, b)

        if self.training and torch.is_grad_enabled() and not is_export_or_trace():
            return coerce_checkpoint(
                _ckpt_impl,
                spatial_tokens,
                temporal_tokens,
                use_reentrant=True,
                preserve_rng_state=True,
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
        self._ckpt_min_bytes = int(64 * 1024 * 1024)

    @property
    def using(self) -> str:
        return self._using

    def _should_enable_checkpoint(self, out, layout_batch_first, need_weights, key_padding_mask):
        if not (
            self.training
            and torch.is_grad_enabled()
            and not need_weights
            and self._ckpt_enabled
            and not is_export_or_trace()
        ):
            return False
        try:
            B, L, D = (
                (out.shape[0], out.shape[1], out.shape[2])
                if layout_batch_first
                else (out.shape[1], out.shape[0], out.shape[2])
            )
            H, bytes_e = int(self.nhead), int(out.element_size())
            base = int(B) * int(L) * int(D) * bytes_e
            score_b = 4 if out.dtype in (torch.float16, torch.bfloat16) else bytes_e
            peak = 0
            flex = _STNET_HAS_FLEX_ATTENTION and out.is_cuda and not need_weights
            for lyr in self.layers:
                dense = not (
                    out.is_cuda
                    and (
                        flex
                        or (
                            (getattr(lyr, "dilation", 1) == 1)
                            and getattr(lyr, "window_size", None) is None
                            and key_padding_mask is None
                        )
                    )
                )
                scores = (int(B) * int(H) * int(L) * int(L) * score_b) if dense else 0
                hid = 0
                if (
                    (ffn := getattr(lyr, "ffn", None))
                    and isinstance(ffn, nn.Sequential)
                    and len(ffn) > 0
                ):
                    hid = getattr(ffn[0], "out_features", 0)
                ffn_b = (int(B) * int(L) * (2 * hid + int(D)) * bytes_e) if hid > 0 else base * 9
                peak = max(peak, int(base * 5 + scores + ffn_b))
            return (peak * len(self.layers)) >= self._ckpt_min_bytes
        except:
            return False

    def _ckpt_fn(self, t, layer, key_padding_mask):
        if torch.is_grad_enabled():
            _unshard_fsdp_module(self)
            _unshard_fsdp_module(layer)
        return layer(
            t,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            average_attn_weights=False,
            skip_ffn_checkpoint=True,
        )[0]

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_w: Optional[torch.Tensor] = None
        out, need_transpose_fallback = x, False
        layout_batch_first = self.batch_first
        if out.dim() == 3 and not self.batch_first and out.shape[0] != out.shape[1]:
            out, layout_batch_first, need_transpose_fallback = out.transpose(0, 1), True, True
        do_ckpt = self._should_enable_checkpoint(
            out, layout_batch_first, need_weights, key_padding_mask
        )
        for layer in self.layers:
            if do_ckpt:
                out = coerce_checkpoint(
                    self._ckpt_fn,
                    out,
                    layer,
                    key_padding_mask,
                    use_reentrant=True,
                    preserve_rng_state=True,
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
