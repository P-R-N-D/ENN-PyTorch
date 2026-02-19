# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import logging
import os
from importlib import import_module
from typing import Any, Callable, List, Optional, Self, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.compat import StochasticDepth
from ..core.datatypes import env_bool, env_int
from ..runtime.distributed import _from_hsdp_module
from .activations import GeGLU
from .graph import coerce_checkpoint, is_checkpoint, is_export_or_trace, is_symbolic, canonicalize_compile_mode
from .layers import CrossAttention, DilatedAttention, LatentAttention, Retention, norm_layer
_LOGGER = logging.getLogger(__name__)
_PRESET_ALIASES: dict[str, str] = {
    "ss": "spatial",
    "spatial": "spatial",
    "sxs": "spatial",
    "tt": "temporal",
    "temporal": "temporal",
    "txt": "temporal",
    "st": "spatiotemporal",
    "ts": "spatiotemporal",
    "sxt": "spatiotemporal",
    "txs": "spatiotemporal",
    "temporal-spatial": "spatiotemporal",
    "temporo-spatial": "spatiotemporal",
    "temporospatial": "spatiotemporal",
    "tempospatial": "spatiotemporal",
    "tempo-spatial": "spatiotemporal",
    "temporalspatial": "spatiotemporal",
    "spatiotemporal": "spatiotemporal",
    "spatio-temporal": "spatiotemporal",
}
_ENN_HAS_FLEX_ATTENTION = getattr(
    import_module(".layers", __package__), "_HAS_FLEX_ATTENTION", False
)


def _safe_norm(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if not torch.is_grad_enabled():
        return norm(x)
    if isinstance(norm, nn.LayerNorm):
        weight = norm.weight
        bias = norm.bias
        return F.layer_norm(
            x,
            norm.normalized_shape,
            weight.clone() if weight is not None else None,
            bias.clone() if bias is not None else None,
            norm.eps,
        )
    return norm(x)


def _enn_longnet_ckpt_clone_if_needed(t: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        return t
    if not bool(getattr(t.device, "type", None) == "cuda"):
        return t
    if not bool(is_checkpoint()):
        return t
    if bool(torch.is_grad_enabled()) and (not env_bool("ENN_CKPT_CUDAGRAPH_CLONE_RECOMPUTE", default=False)):
        return t
    try:
        from ..core.system import get_runtime_cfg

        cfg = get_runtime_cfg()
        mode = canonicalize_compile_mode(getattr(cfg, "compile_mode", "disabled"))
        if mode not in {"max-autotune", "reduce-overhead"}:
            return t
        if not bool(getattr(cfg, "compile_cudagraphs", True)):
            return t
    except Exception:
        return t
    return t.clone()


def _size_of_retnet(
    x: torch.Tensor, blk0: nn.Module, *args: Any, mode: str
) -> int:
    if not isinstance(x, torch.Tensor) or x.dim() != 3:
        return 0
    B, L, D = map(int, x.shape)
    if B <= 0 or L <= 0 or D <= 0:
        return 0
    bytes_e = int(x.element_size())
    base_bytes = int(B) * int(L) * int(D) * int(bytes_e)
    ret_factor = (
        8 if str(mode or "temporal").strip().lower() == "spatial" else 6
    )
    ffn = getattr(blk0, "ffn", None)
    hid = (
        getattr(ffn, "hidden_dim", 0)
        or getattr(ffn, "hid", 0)
        or int(float(D) * 4.0 * (2.0 / 3.0))
    )
    ffn_bytes = int(B) * int(L) * (int(3) * int(hid) + int(D)) * bytes_e
    return int(base_bytes * ret_factor + ffn_bytes + base_bytes * 2)


def _infer_module_device(
    module: nn.Module, fallback: torch.device
) -> torch.device:
    p = next(module.parameters(), None)
    if p is not None:
        return p.device
    b = next(module.buffers(), None)
    return b.device if b is not None else fallback


def _autofit_microbatch(
    device: torch.device, hard_max: int, per_sample_bytes: int
) -> int:
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
    eff = (
        min(host_free, dev_free)
        if (host_free and dev_free)
        else (dev_free or host_free)
    )
    if not eff or eff <= 0:
        return hard_max
    return max(1, min(hard_max, int((eff * 0.35) // max(per_sample_bytes, 1))))


def _coerce_tensor(
    t: torch.Tensor, *args: object, enabled: bool, inplace: bool
) -> torch.Tensor:
    return (
        torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        if enabled and (t.is_floating_point() or t.is_complex())
        else t
    )


def _prealloc_microbatch(
    inp: torch.Tensor,
    microbatch: int,
    run_fn: Callable[[torch.Tensor], torch.Tensor | Sequence[torch.Tensor]],
    *args: object,
    pad_to: Optional[int] = None,
    out_dtype: Optional[torch.dtype] = None,
    cast_slice: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    stage: str = "microbatch",
) -> torch.Tensor | tuple[torch.Tensor, ...]:
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
            out_bufs = [
                y.new_empty((total_b, *y.shape[1:])) for y in processed
            ]
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


def _coerce_preset(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("none", "null", "off", "false", "0", ""):
        return None
    key = s.replace("_", "-").replace(" ", "-")
    normalized = _PRESET_ALIASES.get(key)
    if normalized is None:
        raise ValueError(f"Unsupported preset '{value}'")
    return normalized


def stochastic_depth_schedule(drop_path: float, depth: int) -> List[float]:
    if depth <= 0:
        return []
    if drop_path <= 0.0 or depth == 1:
        return [float(drop_path) if depth == 1 else 0.0] * depth
    return [
        float(i * float(drop_path) / float(depth - 1)) for i in range(depth)
    ]


class LatentTransformer(nn.Module):
    def __init__(
        self,
        *args: Any,
        d_model: int,
        nhead: int,
        mlp_ratio: float,
        norm_type: str,
        eps: float = 1e-6,
        dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        del args
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.attn = LatentAttention(
            d_model=self.d_model,
            nhead=self.nhead,
            norm_type=str(norm_type),
            eps=float(eps),
            dropout=float(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.drop_path = (
            StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(norm_type=norm_type, dim=self.d_model, eps=eps)
        inner_dim = int(self.d_model * mlp_ratio)
        self.ff = nn.Sequential(
            GeGLU(self.d_model, inner_dim, out_dim=inner_dim, bias=True),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, self.d_model, bias=True),
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        attn_proj = self.attn(x)
        x = x + self.drop_path(self.dropout(attn_proj))
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x


class Resampler(nn.Module):
    def __init__(
        self: Self,
        d_model: int,
        nhead: int,
        *args: Any,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_type: str = "layernorm",
        bias: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del args, kwargs
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        if self.d_model % max(1, self.nhead) != 0:
            raise ValueError(
                f"d_model {self.d_model} must be divisible by nhead {self.nhead}"
            )
        self.dropout_p = float(dropout)

        self.cross_attn = CrossAttention(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=float(dropout),
            norm_type=str(norm_type),
            bias=bool(bias),
        )

        self.dropout = nn.Dropout(self.dropout_p)
        self.drop_path = StochasticDepth(p=float(drop_path), mode="row")

        def _norm() -> nn.Module:
            nt = str(norm_type or "layernorm").strip().lower()
            if nt in {"rms", "rmsnorm"} and hasattr(nn, "RMSNorm"):
                return nn.RMSNorm(self.d_model)
            return nn.LayerNorm(self.d_model)

        self.norm_ffn = _norm()
        hid = int(self.d_model * float(mlp_ratio) * (2.0 / 3.0))
        self.ffn = GeGLU(
            self.d_model, hid, out_dim=self.d_model, dropout=dropout
        )

    def forward(
        self: Self,
        latents: torch.Tensor,
        tokens: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_proj = self.cross_attn(latents, tokens, attn_bias=attn_bias)
        latents = latents + self.drop_path(self.dropout(attn_proj))
        latents = latents + self.drop_path(
            self.dropout(self.ffn(self.norm_ffn(latents)))
        )
        return latents

class Perceiver(nn.Module):
    def __init__(
        self: Self,
        d_model: int,
        nhead: int,
        num_latents: int,
        depth: int,
        *args: Any,
        self_attn_layers: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_type: str = "layernorm",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del args, kwargs
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_latents = max(1, int(num_latents))
        self.depth = max(1, int(depth))
        self.self_attn_layers = max(0, int(self_attn_layers))
        self.norm_type = str(norm_type)
        self.latents = nn.Parameter(
            torch.randn(self.num_latents, self.d_model) * 0.02
        )
        total_layers = int(self.depth) * (1 + int(self.self_attn_layers))
        drops = stochastic_depth_schedule(float(drop_path), total_layers)
        dp_it = iter(drops)
        self.cross = nn.ModuleList(
            [
                Resampler(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dropout=float(dropout),
                    mlp_ratio=float(mlp_ratio),
                    drop_path=float(next(dp_it, 0.0)),
                    norm_type=str(norm_type),
                )
                for _ in range(int(self.depth))
            ]
        )
        self.self_blocks = nn.ModuleList(
            [
                LatentTransformer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                    drop_path=float(next(dp_it, 0.0)),
                    norm_type=str(norm_type),
                )
                for _ in range(int(self.depth) * int(self.self_attn_layers))
            ]
        )
        self.norm = norm_layer(str(norm_type), self.d_model)

    def forward(
        self: Self,
        tokens: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(
                f"Perceiver expects tokens (B,N,D), got shape {tuple(tokens.shape)}"
            )
        B = tokens.size(0)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        j = 0
        for i in range(int(self.depth)):
            latents = self.cross[i](latents, tokens, attn_bias=attn_bias)
            for _ in range(int(self.self_attn_layers)):
                if j < len(self.self_blocks):
                    latents = self.self_blocks[j](latents)
                j += 1
        out = self.norm(latents)

        if (not torch.is_grad_enabled()) and (getattr(out.device, "type", None) == "cuda"):
            if env_bool("ENN_PRED_COLLAPSE_STAGE_DIAG", default=False):
                with contextlib.suppress(Exception):
                    def _pair_nonfinite(t: torch.Tensor) -> int:
                        if not t.is_floating_point():
                            return 0
                        return int((~torch.isfinite(t[:2])).sum().item())
                    self._enn_last_perceiver_diag = {
                        "pre_norm_absmax": float(latents[:2].abs().max().item()),
                        "pre_norm_nonfinite": _pair_nonfinite(latents),
                        "post_norm_absmax": float(out[:2].abs().max().item()),
                        "post_norm_nonfinite": _pair_nonfinite(out),
                        "norm_w_absmax": float(self.norm.weight.detach().abs().max().item())
                        if hasattr(self.norm, "weight") else None,
                    }

        return out


class RetNet(nn.Module):
    def __init__(
        self: Self,
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

        self.ffn = SwiGLU(
            self.d_model, hid, out_dim=self.d_model, dropout=dropout
        )

    def forward(
        self: Self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        state: Optional[dict] = None,
        mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        if getattr(x, "is_meta", False) and (not is_symbolic()):
            strict = str(
                os.environ.get("ENN_STRICT_META_FAKE", "0")
            ).strip().lower() in {"1", "true", "yes", "y"}
            if strict:
                raise RuntimeError("meta tensor reached RetNet.forward")
            try:
                shape = tuple(int(s) for s in x.shape)
            except Exception:
                shape = tuple(getattr(x, "shape", ()))
            x = torch.zeros(shape, dtype=x.dtype, device=torch.device("cpu"))
        x = x.contiguous()
        if causal_mask is not None:
            causal_mask = causal_mask.contiguous()
        eff_mode = mode if mode is not None else getattr(self, "mode", None)
        x_norm1 = _safe_norm(self.norm1, x)
        h, new_state = self.retention(
            x_norm1,
            attn_mask=causal_mask,
            state=state,
            mode=eff_mode,
        )
        x = x + self.drop_path(self.dropout(h))
        x_norm2 = _safe_norm(self.norm2, x)
        x = x + self.drop_path(self.dropout(self.ffn(x_norm2)))
        return x, new_state


class LongNet(nn.Module):
    def __init__(
        self: Self,
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
    def using(self: Self) -> str:
        return self._using

    def _should_enable_checkpoint(
        self: Self,
        out: torch.Tensor,
        layout_batch_first: bool,
        need_weights: bool,
        key_padding_mask: Optional[torch.Tensor],
    ) -> bool:
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
            score_b = (
                4 if out.dtype in (torch.float16, torch.bfloat16) else bytes_e
            )
            peak = 0
            flex = _ENN_HAS_FLEX_ATTENTION and out.is_cuda and not need_weights
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
                scores = (
                    (int(B) * int(H) * int(L) * int(L) * score_b)
                    if dense
                    else 0
                )
                hid = 0
                if (
                    (ffn := getattr(lyr, "ffn", None))
                    and isinstance(ffn, nn.Sequential)
                    and len(ffn) > 0
                ):
                    hid = getattr(ffn[0], "out_features", 0)
                ffn_b = (
                    (int(B) * int(L) * (2 * hid + int(D)) * bytes_e)
                    if hid > 0
                    else base * 9
                )
                peak = max(peak, int(base * 5 + scores + ffn_b))
            return (peak * len(self.layers)) >= self._ckpt_min_bytes
        except:
            return False

    def _ckpt_fn(
        self: Self,
        t: torch.Tensor,
        layer: nn.Module,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if torch.is_grad_enabled() or is_checkpoint():
            _from_hsdp_module(self)
            _from_hsdp_module(layer)
        if bool(torch.is_grad_enabled() or is_checkpoint()) and bool(getattr(t.device, "type", None) == "cuda"):
            from .graph import cudagraph_mark_step_begin

            cudagraph_mark_step_begin()
        out = layer(
            t,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            average_attn_weights=False,
            skip_ffn_checkpoint=True,
        )[0]
        return _enn_longnet_ckpt_clone_if_needed(out)

    def forward(
        self: Self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_w: Optional[torch.Tensor] = None
        out, need_transpose_fallback = x, False
        layout_batch_first = self.batch_first
        if (
            out.dim() == 3
            and not self.batch_first
            and out.shape[0] != out.shape[1]
        ):
            out, layout_batch_first, need_transpose_fallback = (
                out.transpose(0, 1),
                True,
                True,
            )
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
        if (
            need_transpose_fallback
            and out.dim() == 3
            and out.shape[0] != out.shape[1]
        ):
            out = out.transpose(0, 1)
        return out, attn_w
