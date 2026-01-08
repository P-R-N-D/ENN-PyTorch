# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import io
import logging
import math
import threading
import weakref
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    Protocol,
    cast,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDictBase

from ..config import ModelConfig
from ..core.casting import env_bool, env_first_int, env_int
from ..core.compat import is_meta_or_fake_tensor
from ..core.distributed import _unshard_fsdp_module
from ..core.graph import (
    graph_break,
    inference_mode,
    clear_model_cache,
    torch_compiler_disable,
    torch_compiler_supported,
    compile as compile_module,
    canonicalize_compile_mode,
    is_export_or_trace,
    coerce_checkpoint,
)
from ..core.precision import Autocast, is_scale_safe
from ..core.profiler import FLOP_PROFILER
from ..core.system import (
    _log_debug,
    _log_info,
    empty_device_cache,
    get_device,
    Thread,
)
from ..data.pipeline import Dataset, get_feature_key, get_label_key
from .blocks import (
    CrossTransformer,
    LongNet,
    RetNet,
    _autofit_microbatch,
    _coerce_modeling_types,
    _coerce_tensor,
    _infer_module_device,
    _prealloc_microbatch,
    _size_of_retnet,
    norm_layer,
    stochastic_depth_schedule,
)
from .layers import (
    Recorder,
    Scaler,
    SigmoidGate,
)


_LOGGER = logging.getLogger(__name__)

_Int8DynamicActivationInt8WeightConfig = None
_Int8WeightOnlyConfig = None

_PTQ_IMPL = None

_qp = None

_TORCHAO_IMPORT_TRIED = False
_TORCHAO_IMPORT_LOCK = threading.Lock()


def _import_torchao_quantization() -> None:
    global _Int8DynamicActivationInt8WeightConfig
    global _Int8WeightOnlyConfig
    global _PTQ_IMPL
    global _qp
    global _TORCHAO_IMPORT_TRIED
    if _TORCHAO_IMPORT_TRIED:
        return
    with _TORCHAO_IMPORT_LOCK:
        if _TORCHAO_IMPORT_TRIED:
            return
        _TORCHAO_IMPORT_TRIED = True
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                from torchao.quantization.quant_api import (
                    Int8DynamicActivationInt8WeightConfig as _Int8DynamicActivationInt8WeightConfig,
                    Int8WeightOnlyConfig as _Int8WeightOnlyConfig,
                    quantize_ as _quantize_,
                )

                try:
                    from torchao.quantization import quant_primitives as _quant_primitives
                except Exception:
                    _quant_primitives = None
            _PTQ_IMPL = _quantize_
            _qp = _quant_primitives
        except Exception:
            _Int8DynamicActivationInt8WeightConfig = None
            _Int8WeightOnlyConfig = None
            _PTQ_IMPL = None
            _qp = None


def _prod_int(shape: Sequence[int]) -> int:
    return int(math.prod(int(v) for v in shape))


def _normalize_tile_shape(
    tile_shape: Sequence[int] | int | None,
    event_shape: Sequence[int],
) -> Optional[Tuple[int, ...]]:
    if tile_shape is None or not event_shape:
        return None
    if isinstance(tile_shape, int) and not isinstance(tile_shape, bool):
        ts = (int(tile_shape),)
    else:
        ts = tuple(int(v) for v in tile_shape)
    if not ts:
        return None
    ndim = len(event_shape)
    if len(ts) == 1:
        ts = ts * ndim
    elif len(ts) < ndim:
        ts = (1,) * (ndim - len(ts)) + ts
    else:
        ts = ts[-ndim:]
    return tuple(max(1, int(v)) for v in ts)


def _tile_counts_grid(
    event_shape: Sequence[int],
    tile_shape: Sequence[int],
    *args: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ev = tuple(int(v) for v in event_shape)
    ts = tuple(int(v) for v in tile_shape)
    grid = tuple((d + t - 1) // t for d, t in zip(ev, ts))
    counts_1d = []
    for d, t, g in zip(ev, ts, grid):
        if g <= 0:
            counts_1d.append(torch.zeros((0,), device=device, dtype=dtype))
        elif g == 1:
            counts_1d.append(torch.tensor([float(d)], device=device, dtype=dtype))
        else:
            last = max(1, int(d - (g - 1) * t))
            head = torch.full((g - 1,), float(t), device=device, dtype=dtype)
            tail = torch.tensor([float(last)], device=device, dtype=dtype)
            counts_1d.append(torch.cat((head, tail), dim=0))
    out = torch.ones(grid, device=device, dtype=dtype)
    for i, c in enumerate(counts_1d):
        shape = [1] * len(grid)
        shape[i] = grid[i]
        out = out * c.view(*shape)
    return out


def _reduce_flat_to_grid(
    x: torch.Tensor,
    event_shape: Sequence[int],
    tile_shape: Sequence[int],
    *args: Any,
    reduce: str = "mean",
    eps: float = 1e-6,
    work_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    ev = tuple(int(v) for v in event_shape)
    ts = tuple(int(v) for v in tile_shape)
    grid = tuple((d + t - 1) // t for d, t in zip(ev, ts))
    B = x.size(0)
    x_ev = x.reshape(B, *ev)
    pads = []
    for orig, g, t in reversed(list(zip(ev, grid, ts))):
        pads.extend([0, int(g * t - orig)])
    if any(p != 0 for p in pads):
        x_ev = F.pad(x_ev, tuple(pads))

    view_shape = [B]
    for g, t in zip(grid, ts):
        view_shape.extend([int(g), int(t)])
    blk = x_ev.reshape(*view_shape).to(dtype=work_dtype)
    tile_dims = tuple(range(2, 1 + 2 * len(grid), 2))
    sum_v = blk.sum(dim=tile_dims)
    if reduce == "sum":
        return sum_v
    counts = _tile_counts_grid(ev, ts, device=blk.device, dtype=blk.dtype).unsqueeze(0)
    return sum_v / torch.clamp(counts, min=float(eps))


def _tv_loss_grid(
    p_grid: torch.Tensor, *args: Any, power: float = 1.0, eps: float = 1e-6
) -> torch.Tensor:
    total = None
    for axis in range(1, p_grid.dim()):
        diff = torch.diff(p_grid, dim=axis)
        pen = (diff.abs() + float(eps)).pow(float(power))
        total = pen if total is None else (total + pen)
    if total is None:
        return p_grid.new_tensor(0.0, dtype=torch.float32)
    return total.mean()


@lru_cache(maxsize=1)
def _dot_product_attention_cls() -> Any:
    try:
        from .kernels import DotProductAttention

        return DotProductAttention
    except Exception:
        return None


def _is_ptq_unavailable(model: nn.Module, *args: Any, **kwargs: Any) -> tuple[nn.Module, bool, str]:
    return (model, False, "PTQ backend unavailable")


class _CallableFuser:
    __slots__ = ("backbone", "device", "meta", "amp_enabled")

    def __init__(
        self,
        backbone: "TokenFuser",
        *args: Any,
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


class _ProxyDecoder(nn.Module):
    def __init__(self, norm: nn.Module, head: nn.Module, out_shape: Sequence[int]) -> None:
        super().__init__()
        self._norm_ref = weakref.ref(norm)
        self._head_ref = weakref.ref(head)
        self._out_shape = tuple(int(x) for x in out_shape)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        norm = self._norm_ref()
        head = self._head_ref()
        if norm is None or head is None:
            raise RuntimeError("Decoder references were cleared before use.")
        tokens = norm(tokens)
        pooled = tokens.mean(dim=1)
        flat = head(pooled)
        return flat.reshape(tokens.shape[0], *self._out_shape)


class SpatialExtractor(nn.Module):
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
        del args, kwargs
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
                    mode="spatial",
                )
                for i in range(int(depth))
            ]
        )
        self.norm = norm_layer(norm_type, int(d_model))
        self._ckpt_enabled = True
        self._ckpt_min_bytes = int(64 * 1024 * 1024)

    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del coords, args, kwargs
        out = x
        do_ckpt = (
            self.training
            and torch.is_grad_enabled()
            and bool(getattr(self, "_ckpt_enabled", True))
            and not is_export_or_trace()
        )
        if do_ckpt:
            est = 0
            with contextlib.suppress(Exception):
                blk0 = self.blocks[0] if len(self.blocks) > 0 else None
                if blk0 is not None:
                    per_blk = _size_of_retnet(out, blk0, mode="spatial")
                    est = int(per_blk) * int(len(self.blocks))
                else:
                    est = int(out.numel()) * int(out.element_size()) * int(len(self.blocks))
            do_ckpt = bool(est >= int(getattr(self, "_ckpt_min_bytes", 0) or 0))
        for blk in self.blocks:
            if do_ckpt:

                def _f(t: torch.Tensor, _blk: RetNet = blk) -> torch.Tensor:
                    if torch.is_grad_enabled():
                        _unshard_fsdp_module(self)
                        _unshard_fsdp_module(_blk)
                    y, _ = _blk(t, causal_mask=attn_mask, state=None, mode="spatial")
                    return y

                out = cast(
                    torch.Tensor,
                    coerce_checkpoint(_f, out, use_reentrant=True, preserve_rng_state=True),
                )
            else:
                out, _ = blk(out, causal_mask=attn_mask, state=None, mode="spatial")
        return self.norm(out)


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
        del args, kwargs
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.depth = int(depth)
        self.head_dim = int(self.d_model // max(1, self.nhead))
        drops = stochastic_depth_schedule(drop_path, depth)
        self.blocks = nn.ModuleList(
            [
                RetNet(
                    self.d_model,
                    self.nhead,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drops[i],
                    norm_type=norm_type,
                    mode="temporal",
                )
                for i in range(int(depth))
            ]
        )
        self.norm = norm_layer(norm_type, int(d_model))
        self._ckpt_enabled = True
        self._ckpt_min_bytes = int(64 * 1024 * 1024)

    @staticmethod
    def _coerce_state_tensor(
        state: Any,
        *args: Any,
        depth: int,
        batch_size: int,
        nhead: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if state is None:
            return None
        strict = env_bool(("STNET_STRICT_TEMPORAL_STATE", "STNET_STRICT_STATE"), default=False)

        def _bad(reason: str) -> Optional[torch.Tensor]:
            if not strict:
                return None
            cand_type = type(state).__name__
            cand_shape = None
            try:
                if isinstance(state, torch.Tensor):
                    cand_shape = tuple(state.shape)
            except Exception:
                cand_shape = None
            raise ValueError(
                "Invalid temporal_state: "
                + reason
                + ". Expected Tensor (depth,B,H,Dh) or Tensor (B,depth,H,Dh) or list/tuple length=depth of per-layer (B,H,Dh). "
                + f"Got type={cand_type} shape={cand_shape}."
            )

        cand: Any = state
        if isinstance(state, Mapping):
            layers = state.get("layers", None)
            if layers is not None:
                cand = layers
            else:
                for key in ("state", "temporal_state", "retention_state", "msr_state"):
                    v = state.get(key, None)
                    if isinstance(v, torch.Tensor):
                        cand = v
                        break
        if isinstance(cand, torch.Tensor):
            t = cand
            if t.dim() == 4 and int(t.shape[2]) == 1:
                t = t[:, :, 0, :]
            if t.dim() == 4:
                if int(t.shape[0]) == int(depth):
                    st = t
                elif int(t.shape[1]) == int(depth):
                    st = t.permute(1, 0, 2, 3)
                else:
                    return _bad(f"4D tensor missing depth axis={int(depth)}")
            elif t.dim() == 3:
                if tuple(map(int, t.shape)) != (
                    int(batch_size),
                    int(nhead),
                    int(head_dim),
                ):
                    return _bad(
                        f"3D tensor shape={tuple(map(int, t.shape))} expected=({int(batch_size)},{int(nhead)},{int(head_dim)})"
                    )
                st = torch.zeros(
                    (int(depth), int(batch_size), int(nhead), int(head_dim)),
                    device=device,
                    dtype=dtype,
                )
                st[0] = t.to(device=device, dtype=dtype)
            else:
                return _bad(f"tensor dim must be 3 or 4, got {int(t.dim())}")
            if tuple(map(int, st.shape)) != (
                int(depth),
                int(batch_size),
                int(nhead),
                int(head_dim),
            ):
                return _bad(
                    f"coerced tensor shape={tuple(map(int, st.shape))} expected=({int(depth)},{int(batch_size)},{int(nhead)},{int(head_dim)})"
                )
            if st.device != device or st.dtype != dtype:
                st = st.to(device=device, dtype=dtype)
            return st.contiguous()
        if isinstance(cand, (list, tuple)):
            out = torch.zeros(
                (int(depth), int(batch_size), int(nhead), int(head_dim)),
                device=device,
                dtype=dtype,
            )
            n = min(int(depth), len(cand))
            for i in range(n):
                v = cand[i]
                if isinstance(v, Mapping):
                    for key in (
                        "state",
                        "temporal_state",
                        "retention_state",
                        "msr_state",
                    ):
                        vv = v.get(key, None)
                        if isinstance(vv, torch.Tensor):
                            v = vv
                            break
                if not isinstance(v, torch.Tensor):
                    continue
                t = v
                if t.dim() == 4 and int(t.shape[2]) == 1:
                    t = t[:, :, 0, :]
                if t.dim() != 3:
                    continue
                if tuple(map(int, t.shape)) != (
                    int(batch_size),
                    int(nhead),
                    int(head_dim),
                ):
                    continue
                out[i] = t.to(device=device, dtype=dtype)
            return out.contiguous()

        return _bad(f"unrecognized state type {type(cand).__name__}")

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        state: Any = None,
        *args: Any,
        return_state: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        del args, kwargs
        B = x.size(0)
        st_tensor: Optional[torch.Tensor] = None
        if state is not None:
            depth = int(self.depth)
            H = int(self.nhead)
            Dh = int(self.head_dim)
            if isinstance(state, torch.Tensor):
                t = state
                if t.dim() == 4 and int(t.shape[2]) == 1:
                    t = t[:, :, 0, :]
                st: Optional[torch.Tensor] = None
                if t.dim() == 4:
                    if tuple(map(int, t.shape)) == (depth, B, H, Dh):
                        st = t
                    elif tuple(map(int, t.shape)) == (B, depth, H, Dh):
                        st = t.permute(1, 0, 2, 3)
                elif t.dim() == 3:
                    if tuple(map(int, t.shape)) == (B, H, Dh):
                        st = t.new_zeros((depth, B, H, Dh))
                        st[0] = t
                if st is not None:
                    if st.device != x.device or st.dtype != x.dtype:
                        st = st.to(device=x.device, dtype=x.dtype)
                    st_tensor = st.contiguous()
                else:
                    st_tensor = self._coerce_state_tensor(
                        state,
                        depth=depth,
                        batch_size=B,
                        nhead=H,
                        head_dim=Dh,
                        device=x.device,
                        dtype=x.dtype,
                    )
            else:
                st_tensor = self._coerce_state_tensor(
                    state,
                    depth=depth,
                    batch_size=B,
                    nhead=H,
                    head_dim=Dh,
                    device=x.device,
                    dtype=x.dtype,
                )
        do_ckpt = (
            self.training
            and torch.is_grad_enabled()
            and bool(getattr(self, "_ckpt_enabled", True))
            and not return_state
            and st_tensor is None
            and not is_export_or_trace()
        )
        if do_ckpt:
            est = 0
            with contextlib.suppress(Exception):
                blk0 = self.blocks[0] if len(self.blocks) > 0 else None
                if blk0 is not None:
                    per_blk = _size_of_retnet(x, blk0, mode="temporal")
                    est = int(per_blk) * int(len(self.blocks))
                else:
                    est = int(x.numel()) * int(x.element_size()) * int(len(self.blocks))
            do_ckpt = bool(est >= int(getattr(self, "_ckpt_min_bytes", 0) or 0))
        next_state: Optional[torch.Tensor] = None
        if return_state:
            next_state = x.new_empty((int(self.depth), B, int(self.nhead), int(self.head_dim)))
        for i, blk in enumerate(self.blocks):
            blk_state = None
            if st_tensor is not None and i < int(self.depth):
                blk_state = st_tensor[i]
            if do_ckpt:

                def _f(t: torch.Tensor, _blk: RetNet = blk) -> torch.Tensor:
                    if torch.is_grad_enabled():
                        _unshard_fsdp_module(self)
                        _unshard_fsdp_module(_blk)
                    y, _ = _blk(t, causal_mask=causal_mask, state=None, mode="temporal")
                    return y

                x = cast(
                    torch.Tensor,
                    coerce_checkpoint(_f, x, use_reentrant=True, preserve_rng_state=True),
                )
            else:
                x, blk_next_state = blk(
                    x, causal_mask=causal_mask, state=blk_state, mode="temporal"
                )
                if next_state is not None:
                    if blk_next_state is None:
                        blk_next_state = x.new_zeros((B, int(self.nhead), int(self.head_dim)))
                    next_state[i] = blk_next_state
        x = self.norm(x)
        if next_state is not None:
            if not torch.is_grad_enabled():
                next_state = next_state.detach()
            return x, next_state.contiguous()
        return x


class TokenCollector(nn.Module):
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
        self.microbatch: int = 0
        self._auto_microbatch_pending: bool = True
        self._runtime_lock = threading.Lock()

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_runtime_lock", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._runtime_lock = threading.Lock()

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

    def forward_export(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out, _ = self.backbone(
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            average_attn_weights=False,
        )
        return out

    def run(
        self,
        tokens: torch.Tensor,
        *args: Any,
        device: torch.device,
        meta: Any,
        amp_enabled: bool,
        auto_microbatch_fn: Callable[[torch.Tensor], int],
        graph_break_fn: Optional[Callable[[], None]] = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                f"TokenCollector.run expects tokens (B,N,D), got shape {tuple(tokens.shape)}"
            )
        B = tokens.size(0)
        if graph_break_fn is not None:
            graph_break_fn()
        with self._runtime_lock:
            pending = bool(self._auto_microbatch_pending)
        if pending:
            try:
                mb_guess = int(auto_microbatch_fn(tokens))
                mb_new = max(1, min(B, mb_guess))
            except Exception:
                mb_new = max(1, B)
            with self._runtime_lock:
                self.microbatch = int(mb_new)
                self._auto_microbatch_pending = False
        with self._runtime_lock:
            mb_cur = int(self.microbatch) if self.microbatch else B
        mb = max(1, min(B, mb_cur))
        infer_mode = (not torch.is_grad_enabled()) and (not is_export_or_trace())
        controller_ctx = inference_mode(self.backbone) if infer_mode else contextlib.nullcontext()
        runner = _CallableFuser(
            self.backbone,
            device=device,
            meta=meta,
            amp_enabled=amp_enabled,
        )
        with controller_ctx:
            refined = cast(
                torch.Tensor,
                _prealloc_microbatch(tokens, mb, runner, stage="controller"),
            )
        return refined


class TokenizedView(nn.Module):
    def __init__(self, in_dim: int, tokens: int, d_model: int, extractor: nn.Module) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.tokens = int(tokens)
        self.d_model = int(d_model)
        self.tokenizer = nn.Linear(self.in_dim, self.tokens * self.d_model)
        self.extractor = extractor

    @property
    def depth(self) -> int:
        return int(getattr(self.extractor, "depth", 0) or 0)

    @property
    def nhead(self) -> int:
        return int(getattr(self.extractor, "nhead", 0) or 0)

    @property
    def head_dim(self) -> int:
        return int(getattr(self.extractor, "head_dim", 0) or 0)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        B = x.size(0)
        tokens = self.tokenizer(x).reshape(B, self.tokens, self.d_model).contiguous()
        return self.extractor(tokens, *args, **kwargs)


class TokenFuser(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_shape: Sequence[int],
        config: ModelConfig,
        *args: Any,
        views: Optional[Mapping[str, nn.Module] | Sequence[Tuple[str, nn.Module]]] = None,
        fusions: Optional[
            Mapping[str | Tuple[str, str], nn.Module]
            | Sequence[Tuple[str | Tuple[str, str], nn.Module]]
        ] = None,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_shape = tuple((int(v) for v in out_shape))
        self.out_dim = int(math.prod(self.out_shape) if self.out_shape else 1)
        self.d_model = int(config.d_model)
        self.nhead = int(config.heads)
        self.modeling_type = _coerce_modeling_types(config.modeling_type)
        self.spatial_tokens = max(1, int(getattr(config, "spatial_latents", 1)))
        self.temporal_tokens = max(1, int(getattr(config, "temporal_latents", 1)))
        self.fused_tokens = max(1, int(self.spatial_tokens + self.temporal_tokens))
        self.mlp_ratio = float(getattr(config, "mlp_ratio", 4.0))
        self.dropout = float(getattr(config, "dropout", 0.0))
        self.drop_path = float(getattr(config, "drop_path", 0.0))
        self.norm_type = str(getattr(config, "normalization_method", "layernorm"))
        self.gate_blend_alpha = float(
            getattr(config, "fuser_blend_alpha", getattr(config, "fuser_gate_blend", 0.0))
        )
        self.gate_blend_alpha = float(min(max(self.gate_blend_alpha, 0.0), 1.0))
        self.view_encoders = nn.ModuleDict()
        if views is None:
            spatial_extractor = SpatialExtractor(
                self.d_model,
                self.nhead,
                depth=max(1, int(getattr(config, "spatial_depth", 1))),
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                drop_path=self.drop_path,
                norm_type=self.norm_type,
            )
            temporal_extractor = TemporalExtractor(
                self.d_model,
                self.nhead,
                depth=max(1, int(getattr(config, "temporal_depth", 1))),
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                drop_path=self.drop_path,
                norm_type=self.norm_type,
            )
            self.view_encoders["spatial"] = TokenizedView(
                self.in_dim, self.spatial_tokens, self.d_model, spatial_extractor
            )
            self.view_encoders["temporal"] = TokenizedView(
                self.in_dim, self.temporal_tokens, self.d_model, temporal_extractor
            )
        else:
            items = views.items() if isinstance(views, Mapping) else list(views)
            for name, mod in items:
                key = str(name)
                if not isinstance(mod, nn.Module):
                    raise TypeError("views must contain modules")
                self.view_encoders[key] = mod
        if "spatial" in self.view_encoders:
            self.spatial_tokenized_view = self.view_encoders["spatial"]
            self.spatial_net = self.spatial_tokenized_view
        if "temporal" in self.view_encoders:
            self.temporal_tokenized_view = self.view_encoders["temporal"]
            self.temporal_net = self.temporal_tokenized_view
        self.pair_fusers = nn.ModuleDict()
        self._pair_endpoints: dict[str, tuple[str, str]] = {}
        if fusions is None:
            key, (a, b) = self._canon_pair_key_static(("spatial", "temporal"))
            self.pair_fusers[key] = CrossTransformer(
                self.d_model,
                self.nhead,
                dropout=self.dropout,
                norm_type=self.norm_type,
                mlp_ratio=self.mlp_ratio,
                drop_path=self.drop_path,
            )
            self._pair_endpoints[key] = (a, b)
        else:
            items = fusions.items() if isinstance(fusions, Mapping) else list(fusions)
            for raw_key, mod in items:
                if not isinstance(mod, nn.Module):
                    raise TypeError("fusions must contain modules")
                key, (a, b) = self._canon_pair_key_static(raw_key)
                if a == b:
                    raise ValueError("fusion endpoints must differ")
                self.pair_fusers[key] = mod
                self._pair_endpoints[key] = (a, b)
        if "spatial|temporal" in self.pair_fusers:
            self.perception = self.pair_fusers["spatial|temporal"]
        elif len(self.pair_fusers) == 1:
            only_key = next(iter(self.pair_fusers.keys()))
            self.perception = self.pair_fusers[only_key]
        hid = int(self.d_model * max(1.0, self.mlp_ratio))
        self._agg_norm = norm_layer(self.norm_type, self.d_model)
        self._agg_phi = nn.Sequential(
            nn.Linear(self.d_model, hid),
            nn.GELU(),
            nn.Linear(hid, self.d_model),
        )
        self._agg_gate = nn.Sequential(
            nn.Linear(self.d_model, hid),
            nn.GELU(),
            nn.Linear(hid, 1),
        )
        self._token_generator = nn.Linear(self.d_model, self.fused_tokens * self.d_model)
        self.norm = norm_layer(self.norm_type, self.d_model)
        self.head_hidden_dim = hid
        self.head = nn.Sequential(
            norm_layer(self.norm_type, self.d_model),
            nn.Linear(self.d_model, hid),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(hid, self.out_dim),
        )
        self.views = self.view_encoders
        self.fusions = self.pair_fusers

    @staticmethod
    def _canon_pair_key_static(
        raw: str | Tuple[str, str],
    ) -> tuple[str, tuple[str, str]]:
        if isinstance(raw, str):
            parts = [p for p in raw.split("|") if p]
            if len(parts) != 2:
                raise ValueError("fusion key string must be 'a|b'")
            a, b = str(parts[0]), str(parts[1])
        elif isinstance(raw, (tuple, list)) and len(raw) == 2:
            a, b = str(raw[0]), str(raw[1])
        else:
            raise TypeError("fusion key must be 'a|b' or (a, b)")
        x, y = (a, b) if a <= b else (b, a)
        return f"{x}|{y}", (x, y)

    def _canon_pair_key(self, raw: str | Tuple[str, str]) -> tuple[str, tuple[str, str]]:
        return self._canon_pair_key_static(raw)

    @staticmethod
    def _as_3d_tokens(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            return t.unsqueeze(1)
        if t.dim() != 3:
            raise ValueError(f"expected tokens shaped (B,N,D) or (B,D), got {tuple(t.shape)}")
        return t

    def _aggregate_tokens(self, token_sets: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(token_sets) < 1:
            raise ValueError("no token sets to aggregate")
        summaries = torch.stack([self._agg_norm(ts.mean(dim=1)) for ts in token_sets], dim=1)
        feats = self._agg_phi(summaries)
        logits = self._agg_gate(feats).squeeze(-1)
        w_soft = torch.softmax(logits, dim=1)
        K = feats.size(1)
        w_uni = feats.new_ones((feats.size(0), K)) / K
        a = float(self.gate_blend_alpha)
        w = (1.0 - a) * w_uni + a * w_soft
        fused_vec = torch.bmm(w.unsqueeze(1), feats).squeeze(1)
        B = fused_vec.size(0)
        tokens = (
            self._token_generator(fused_vec)
            .reshape(B, self.fused_tokens, self.d_model)
            .contiguous()
        )
        return tokens

    @staticmethod
    def _pick_view(
        views: Mapping[str, torch.Tensor],
        preferred: Sequence[str],
        fallback_first: bool = True,
    ) -> torch.Tensor:
        for k in preferred:
            if k in views:
                return views[k]
        if fallback_first:
            return next(iter(views.values()))
        raise KeyError(f"missing views: {preferred}")

    def _run_views(
        self,
        x: torch.Tensor,
        *args: Any,
        temporal_state: Any = None,
        want_state: bool = False,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> tuple[dict[str, torch.Tensor], Optional[Any]]:
        out: dict[str, torch.Tensor] = {}
        next_state: Optional[Any] = None
        for name, mod in self.view_encoders.items():
            if name == "temporal":
                if want_state:
                    y = mod(
                        x,
                        state=temporal_state,
                        return_state=True,
                        causal_mask=causal_mask,
                    )
                    if isinstance(y, (tuple, list)) and len(y) == 2:
                        tokens, next_state = y[0], y[1]
                    else:
                        tokens = y
                else:
                    tokens = mod(
                        x,
                        state=temporal_state,
                        return_state=False,
                        causal_mask=causal_mask,
                    )
            else:
                tokens = mod(x)
            tokens = self._as_3d_tokens(cast(torch.Tensor, tokens))
            out[name] = tokens
        return out, next_state

    def _run_fusions(self, views: Mapping[str, torch.Tensor]) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for key, fuser in self.pair_fusers.items():
            a, b = self._pair_endpoints.get(key, (None, None))
            if a is None or b is None:
                continue
            if a not in views or b not in views:
                continue
            out.append(fuser(views[a], views[b]))
        return out

    @torch_compiler_disable(
        reason="TokenFuser orchestrates eager + compiled submodules", recursive=False
    )
    def forward(
        self,
        x: torch.Tensor,
        *args: Any,
        temporal_state: Any = None,
        return_temporal_state: bool = False,
        causal_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        del args, kwargs
        views, next_state = self._run_views(
            x,
            temporal_state=temporal_state,
            want_state=bool(return_temporal_state),
            causal_mask=causal_mask,
        )
        mode_l = _coerce_modeling_types(self.modeling_type)
        if mode_l == "ss":
            chosen = [self._pick_view(views, ("spatial", "s"))]
        elif mode_l == "tt":
            chosen = [self._pick_view(views, ("temporal", "t"))]
        else:
            fused = self._run_fusions(views)
            chosen = fused if len(fused) > 0 else list(views.values())
        chosen = [self._as_3d_tokens(t) for t in chosen]
        tokens = self._aggregate_tokens(chosen)
        context = self.decode(tokens, apply_norm=False)
        if bool(return_temporal_state):
            return tokens, context, next_state
        return tokens, context

    def forward_state(
        self,
        x: torch.Tensor,
        *args: Any,
        temporal_state: Any = None,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, context, next_state = self.forward(
            x,
            temporal_state=temporal_state,
            return_temporal_state=True,
            causal_mask=causal_mask,
        )
        if isinstance(next_state, torch.Tensor):
            return tokens, context, next_state
        B = x.size(0)
        tn = getattr(self, "temporal_net", None)
        if tn is not None:
            depth = int(getattr(tn, "depth", 0))
            nhead = int(getattr(tn, "nhead", self.nhead))
            head_dim = int(getattr(tn, "head_dim", max(1, self.d_model // max(1, nhead))))
        else:
            depth = 0
            nhead = int(self.nhead)
            head_dim = int(max(1, self.d_model // max(1, nhead)))
        filler = tokens.new_zeros((max(1, depth), B, max(1, nhead), max(1, head_dim)))
        return tokens, context, filler

    def forward_stream(
        self,
        x: torch.Tensor,
        *args: Any,
        temporal_state: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, context, next_state = self.forward_state(
            x,
            temporal_state=temporal_state,
            causal_mask=causal_mask,
        )
        return tokens, context, next_state

    def decode(self, tokens: torch.Tensor, *args: Any, apply_norm: bool = False) -> torch.Tensor:
        if apply_norm:
            tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        flat = self.head(pooled)
        return flat.reshape(tokens.shape[0], *self.out_shape)

    def forward_export(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(x, torch.Tensor):
            raise TypeError("forward_export expects a Tensor")
        views, _ = self._run_views(x, temporal_state=None, want_state=False, causal_mask=None)
        mode_l = _coerce_modeling_types(self.modeling_type)
        if mode_l == "ss":
            chosen = [self._pick_view(views, ("spatial", "s"))]
        elif mode_l == "tt":
            chosen = [self._pick_view(views, ("temporal", "t"))]
        else:
            fused = self._run_fusions(views)
            chosen = fused if len(fused) > 0 else list(views.values())
        chosen = [self._as_3d_tokens(t) for t in chosen]
        tokens = self._aggregate_tokens(chosen)
        context = self.decode(tokens, apply_norm=False)
        return tokens, context


class Model(nn.Module):
    @staticmethod
    def _get_cfg(cfg, name, default, type_=float):
        return type_(getattr(cfg, name, default))

    def __init__(self, in_dim: int, out_shape: Sequence[int], config: ModelConfig) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_shape = tuple((int(x) for x in out_shape))
        self.out_dim = int(math.prod(self.out_shape))
        if config.device is not None:
            self._device = torch.device(config.device)
            if self._device.type in {"cuda", "xpu"} and self._device.index is not None:
                with contextlib.suppress(Exception):
                    from ..core.system import set_accelerator_index

                    set_accelerator_index(self._device.type, int(self._device.index))
        else:
            self._device = get_device()
        self.scaler = Scaler().to(self._device)
        with torch.no_grad():
            if self.scaler.x_mean.numel() != self.in_dim:
                self.scaler.x_mean.resize_(self.in_dim)
                self.scaler.x_std.resize_(self.in_dim)
                self.scaler.x_mean.zero_()
                self.scaler.x_std.fill_(1.0)
        self.logger = Recorder()
        self.is_norm_linear = bool(getattr(config, "use_linear_branch", False))
        self.linear_branch = (
            nn.Linear(self.in_dim, self.out_dim).to(self._device) if self.is_norm_linear else None
        )
        self.fuser = TokenFuser(self.in_dim, self.out_shape, config=config).to(self._device)
        self.processor = self.fuser
        bucket = self._get_cfg(config, "length_bucket_multiple", 64, int)
        self.temporal_token_collector = TokenCollector(
            int(config.d_model),
            int(config.heads),
            depth=max(1, int(getattr(config, "temporal_depth", 1))),
            mlp_ratio=float(getattr(config, "mlp_ratio", 4.0)),
            dropout=float(getattr(config, "dropout", 0.0)),
            batch_first=True,
            length_bucket_multiple=bucket,
        ).to(self._device)
        self.controller = self.temporal_token_collector
        self.p_gate: Optional[SigmoidGate]
        k_def = self._get_cfg(config, "p_gate_fallback_k", 6.0)
        self.p_gate_fallback_k: float = float(k_def)
        self.p_gate_fallback_k_low = float(getattr(config, "p_gate_fallback_k_low", k_def) or k_def)
        self.p_gate_fallback_k_high = float(
            getattr(config, "p_gate_fallback_k_high", k_def) or k_def
        )
        self.p_gate_auto_k_enabled: bool = True
        self.p_gate_auto_k_interval = max(
            1, self._get_cfg(config, "p_gate_auto_k_interval", 100, int)
        )
        self.p_gate_auto_k_warmup = max(0, self._get_cfg(config, "p_gate_auto_k_warmup", 0, int))
        self.p_gate_auto_k_ema_alpha = self._get_cfg(config, "p_gate_auto_k_ema_alpha", 0.1)
        self.p_gate_auto_k_target_tight = self._get_cfg(config, "p_gate_auto_k_target_tight", 0.02)
        self.p_gate_auto_k_tolerance = self._get_cfg(config, "p_gate_auto_k_tolerance", 0.5)
        self.p_gate_auto_k_step_up = self._get_cfg(config, "p_gate_auto_k_step_up", 0.1)
        self.p_gate_auto_k_step_down = self._get_cfg(config, "p_gate_auto_k_step_down", 0.02)
        self.p_gate_auto_k_step_up_low = self._get_cfg(
            config, "p_gate_auto_k_step_up_low", self.p_gate_auto_k_step_up
        )
        self.p_gate_auto_k_step_down_low = self._get_cfg(
            config, "p_gate_auto_k_step_down_low", self.p_gate_auto_k_step_down
        )
        self.p_gate_auto_k_step_up_high = self._get_cfg(
            config, "p_gate_auto_k_step_up_high", self.p_gate_auto_k_step_up
        )
        self.p_gate_auto_k_step_down_high = self._get_cfg(
            config, "p_gate_auto_k_step_down_high", self.p_gate_auto_k_step_down
        )
        self.p_gate_auto_k_edge_enabled: bool = True
        self.p_gate_auto_k_target_edge = self._get_cfg(config, "p_gate_auto_k_target_edge", 0.05)
        self.p_gate_auto_k_edge_tolerance = self._get_cfg(
            config, "p_gate_auto_k_edge_tolerance", 0.5
        )
        self.p_gate_auto_k_edge_ema_alpha: float = float(
            getattr(config, "p_gate_auto_k_edge_ema_alpha", self.p_gate_auto_k_ema_alpha)
        )
        self.p_gate_auto_k_edge_step_down_low: float = float(
            getattr(config, "p_gate_auto_k_edge_step_down_low", 0.01)
        )
        self.p_gate_auto_k_edge_step_down_high: float = float(
            getattr(config, "p_gate_auto_k_edge_step_down_high", 0.01)
        )
        self.p_gate_auto_k_min: float = float(getattr(config, "p_gate_auto_k_min", 1.0))
        self.p_gate_auto_k_max: float = float(getattr(config, "p_gate_auto_k_max", 16.0))
        self.p_gate_auto_k_width_frac: float = float(
            getattr(config, "p_gate_auto_k_width_frac", 0.05)
        )
        self.p_gate_auto_k_edge_frac: float = float(
            getattr(config, "p_gate_auto_k_edge_frac", 0.02)
        )
        self.p_gate_auto_k_log_interval: int = int(
            getattr(config, "p_gate_auto_k_log_interval", 200) or 0
        )
        if self.p_gate_auto_k_enabled and (
            self.p_gate_fallback_k_low <= 0.0 or self.p_gate_fallback_k_high <= 0.0
        ):
            m = max(float(self.p_gate_auto_k_min), 1e-6)
            self.p_gate_fallback_k_low = max(self.p_gate_fallback_k_low, m)
            self.p_gate_fallback_k_high = max(self.p_gate_fallback_k_high, m)
        self.p_gate_fallback_enabled: bool = bool(
            self.p_gate_fallback_k_low > 0.0 and self.p_gate_fallback_k_high > 0.0
        )
        self.p_gate_tile_size: Optional[int] = getattr(config, "p_gate_tile_size", None)
        raw_tile_shape = getattr(config, "p_gate_tile_shape", None)
        tile_shape: Optional[Tuple[int, ...]]
        try:
            if raw_tile_shape is None:
                tile_shape = None
            else:
                if isinstance(raw_tile_shape, int) and not isinstance(raw_tile_shape, bool):
                    tile_shape = (int(raw_tile_shape),)
                else:
                    tile_shape = tuple(int(v) for v in raw_tile_shape)
                tile_shape = tuple(max(1, int(v)) for v in tile_shape)

                out_ndim = int(len(self.out_shape))
                if out_ndim > 0:
                    if len(tile_shape) == 1:
                        tile_shape = tile_shape * out_ndim
                    elif len(tile_shape) < out_ndim:
                        tile_shape = (1,) * (out_ndim - len(tile_shape)) + tile_shape
                    elif len(tile_shape) > out_ndim:
                        tile_shape = tile_shape[-out_ndim:]
        except Exception:
            tile_shape = None
        self.p_gate_tile_shape: Optional[Tuple[int, ...]] = tile_shape
        self.p_gate_bounds_use_quantile: bool = bool(
            getattr(config, "p_gate_bounds_use_quantile", False)
        )
        self.p_gate_bounds_q_low: float = float(getattr(config, "p_gate_bounds_q_low", 0.005))
        self.p_gate_bounds_q_high: float = float(getattr(config, "p_gate_bounds_q_high", 0.995))
        self.p_gate_bounds_q_max_samples: int = int(
            getattr(config, "p_gate_bounds_q_max_samples", 8192) or 0
        )
        self.p_gate_bounds_clip_to_minmax: bool = bool(
            getattr(config, "p_gate_bounds_clip_to_minmax", True)
        )
        try:
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
        self.p_gate = SigmoidGate(
            d_model=int(config.d_model),
            hidden_dim=int(getattr(config, "p_gate_hidden_dim", 64)),
            detach_inputs=bool(getattr(config, "p_gate_detach_inputs", True)),
            p_floor=float(getattr(config, "p_gate_p_floor", 0.0)),
            p_ceil=float(getattr(config, "p_gate_p_ceil", 1.0)),
            tile_size=getattr(config, "p_gate_tile_size", None),
            tile_shape=self.p_gate_tile_shape,
            event_shape=self.out_shape,
            clip_eps=float(getattr(config, "p_gate_clip_eps", 1e-6)),
            eps=float(getattr(config, "p_gate_eps", 1e-6)),
            stat_width_frac=float(getattr(config, "p_gate_auto_k_width_frac", 0.05)),
            stat_edge_frac=float(getattr(config, "p_gate_auto_k_edge_frac", 0.02)),
        ).to(self._device)
        self.unsup_xx_weight = float(getattr(config, "unsup_xx_weight", 0.0))
        self.unsup_yy_weight = float(getattr(config, "unsup_yy_weight", 0.0))
        self.p_prior_weight = float(getattr(config, "p_prior_weight", 0.0))
        self.p_prior_alpha = float(getattr(config, "p_prior_alpha", 2.0))
        self.p_prior_beta = float(getattr(config, "p_prior_beta", 2.0))
        self.p_gate_edge_reg_weight = float(getattr(config, "p_gate_edge_reg_weight", 0.0))
        self.p_gate_edge_reg_frac = float(
            getattr(
                config,
                "p_gate_edge_reg_frac",
                getattr(config, "p_gate_auto_k_edge_frac", 0.02),
            )
        )
        self.p_gate_edge_reg_min_width_frac = float(
            getattr(
                config,
                "p_gate_edge_reg_min_width_frac",
                getattr(config, "p_gate_auto_k_width_frac", 0.05),
            )
        )
        self.p_gate_edge_reg_power = float(getattr(config, "p_gate_edge_reg_power", 2.0))
        self.p_gate_budget_weight = float(getattr(config, "p_gate_budget_weight", 0.0))
        self.p_gate_budget_target = float(getattr(config, "p_gate_budget_target", 0.5))
        self.p_gate_tv_weight = float(getattr(config, "p_gate_tv_weight", 0.0))
        self.p_gate_tv_power = float(getattr(config, "p_gate_tv_power", 1.0))
        self.p_gate_teacher_weight = float(getattr(config, "p_gate_teacher_weight", 0.0))
        self.p_gate_teacher_temp = float(getattr(config, "p_gate_teacher_temp", 0.25))
        self.p_gate_teacher_tau = float(getattr(config, "p_gate_teacher_tau", 0.0))
        self.p_gate_teacher_relu = bool(getattr(config, "p_gate_teacher_relu", False))
        try:
            w_low_cfg = getattr(config, "p_gate_edge_reg_weight_low", None)
            w_high_cfg = getattr(config, "p_gate_edge_reg_weight_high", None)
            self.p_gate_edge_reg_weight_low = (
                float(self.p_gate_edge_reg_weight) if w_low_cfg is None else float(w_low_cfg)
            )
            self.p_gate_edge_reg_weight_high = (
                float(self.p_gate_edge_reg_weight) if w_high_cfg is None else float(w_high_cfg)
            )
        except Exception:
            self.p_gate_edge_reg_weight_low = float(self.p_gate_edge_reg_weight)
            self.p_gate_edge_reg_weight_high = float(self.p_gate_edge_reg_weight)
        self.p_gate_edge_reg_fallback_only = bool(
            getattr(config, "p_gate_edge_reg_fallback_only", False)
        )
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
        self.microbatch: int = 0
        self._auto_microbatch_pending: bool = True
        self._runtime_lock = threading.Lock()
        self._eager_processor_temporal_net = getattr(self.processor, "temporal_net", None)
        self._eager_processor_perception = getattr(self.processor, "perception", None)
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
        compile_mode_canonical = canonicalize_compile_mode(raw_mode)
        compile_mode_arg = None if compile_mode_canonical == "disabled" else compile_mode_canonical
        compile_requested = compile_mode_arg is not None
        compile_available = bool(torch_compiler_supported())
        compile_enabled = bool(compile_requested and compile_available)
        nogil_opt = False
        with contextlib.suppress(Exception):
            nogil_opt = bool(Thread.is_optimized_for_no_gil())
        if nogil_opt and compile_enabled:
            _LOGGER.info(
                "No-GIL optimized mode detected; using conservative torch.compile defaults "
                "(disable cudagraphs and skip heavy submodules by default)"
            )
        if compile_requested and not compile_available:
            _LOGGER.warning(
                "torch.compile requested (compile_mode=%r) but torch.compile is unavailable; running eagerly",
                raw_mode,
            )
        compile_dynamic = bool(
            getattr(config, "compile_dynamic", compile_mode_canonical == "reduce-overhead")
        )
        compile_cudagraphs_default = (not bool(nogil_opt)) and compile_mode_canonical not in {
            "reduce-overhead",
            "max-autotune-no-cudagraphs",
        }
        compile_cudagraphs = bool(getattr(config, "compile_cudagraphs", compile_cudagraphs_default))
        self._compile_cudagraphs = bool(
            compile_enabled and compile_cudagraphs and getattr(self._device, "type", None) == "cuda"
        )
        compile_patch_ctx = contextlib.nullcontext()
        compile_options = (
            {"triton.cudagraphs": False}
            if (
                compile_enabled
                and getattr(self._device, "type", None) == "cuda"
                and (not bool(compile_cudagraphs))
            )
            else None
        )
        compile_heavy_submodules_default = (not bool(nogil_opt)) and compile_mode_canonical not in {
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
            with compile_patch_ctx:
                try:
                    _raw_head = self.processor.head
                    _decode_mod = _ProxyDecoder(self.processor.norm, _raw_head, self.out_shape).to(
                        self._device
                    )
                    _compiled = compile_module(
                        _decode_mod,
                        mode=compile_mode_arg,
                        fullgraph=False,
                        dynamic=compile_dynamic,
                        backend="inductor",
                        options=compile_options,
                        disable=False,
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
                        _orig = self._eager_processor_temporal_net
                        if _orig is not None:
                            _compiled = compile_module(
                                _orig,
                                mode=compile_mode_arg,
                                fullgraph=False,
                                dynamic=compile_dynamic,
                                backend="inductor",
                                options=compile_options,
                                disable=False,
                            )
                            self.processor.temporal_net = _compiled
                            with contextlib.suppress(Exception):
                                if hasattr(self.processor, "view_encoders") and (
                                    "temporal" in self.processor.view_encoders
                                ):
                                    self.processor.view_encoders["temporal"] = _compiled
                            compiled_temporal = _compiled is not _orig
                    except Exception:
                        _LOGGER.warning(
                            "torch.compile failed for fuser temporal view; continuing eagerly",
                            exc_info=True,
                        )
                if getattr(self._device, "type", None) == "cuda":
                    empty_device_cache(device=self._device, do_gc=True, min_interval_s=0.0)
                if compile_heavy_submodules:
                    try:
                        _orig = self._eager_processor_perception
                        if _orig is not None:
                            _compiled = compile_module(
                                _orig,
                                mode=compile_mode_arg,
                                fullgraph=False,
                                dynamic=compile_dynamic,
                                backend="inductor",
                                options=compile_options,
                                disable=False,
                            )
                            self.processor.perception = _compiled
                            with contextlib.suppress(Exception):
                                if hasattr(self.processor, "pair_fusers"):
                                    if "spatial|temporal" in self.processor.pair_fusers:
                                        self.processor.pair_fusers["spatial|temporal"] = _compiled
                                    elif len(self.processor.pair_fusers) == 1:
                                        _pair = next(iter(self.processor.pair_fusers.keys()))
                                        self.processor.pair_fusers[_pair] = _compiled
                            compiled_perception = _compiled is not _orig
                    except Exception:
                        _LOGGER.warning(
                            "torch.compile failed for fuser primary pair fuser; continuing eagerly",
                            exc_info=True,
                        )
                if getattr(self._device, "type", None) == "cuda":
                    empty_device_cache(device=self._device, do_gc=True, min_interval_s=0.0)
        self._compiled_submodules = {
            "decode": bool(compiled_decode),
            "temporal_net": bool(compiled_temporal),
            "perception": bool(compiled_perception),
        }
        self._pad_compiled_microbatch = bool(
            compiled_decode or compiled_temporal or compiled_perception
        )
        self._amp_dtype_cache: Dict[Tuple[Any, ...], torch.dtype] = {}
        self._amp_dtype_cache_last_key: Tuple[Any, ...] | None = None
        self._amp_dtype_cache_last_dtype: torch.dtype | None = None
        self._amp_dtype_cache_max = 64
        self._amp_dtype_cache_lock = threading.Lock()
        self.__config = config
        self.__stnet_instance_config__ = config

    @property
    def config(self) -> ModelConfig:
        return self.__config

    def to(self, *args: Any, **kwargs: Any) -> "Model":
        out = super().to(*args, **kwargs)
        with contextlib.suppress(Exception):
            if isinstance(getattr(self, "logger", None), Recorder):
                self.logger.cpu()
        return cast(Model, out)

    def __getstate__(self):
        ctx = contextlib.nullcontext()
        with contextlib.suppress(Exception):
            ctx = self.eager_for_export()
        with ctx:
            state = super().__getstate__()
        state.pop("_runtime_lock", None)
        state.pop("_amp_dtype_cache_lock", None)
        state.pop("_amp_dtype_cache", None)
        state.pop("_amp_dtype_cache_last_key", None)
        state.pop("_amp_dtype_cache_last_dtype", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._runtime_lock = threading.Lock()
        self._amp_dtype_cache_lock = threading.Lock()
        self._amp_dtype_cache = {}
        self._amp_dtype_cache_last_key = None
        self._amp_dtype_cache_last_dtype = None
        if not hasattr(self, "_amp_dtype_cache_max"):
            self._amp_dtype_cache_max = 64

    @staticmethod
    def _cast_graph_safe(
        x: torch.Tensor, device: torch.device, dtype: Optional[torch.dtype]
    ) -> torch.Tensor:
        target_dtype = dtype or x.dtype
        if x.device != device:
            return x.to(device=device, dtype=target_dtype, non_blocking=True)
        if x.dtype != target_dtype:
            return x.to(dtype=target_dtype)
        return x

    @contextlib.contextmanager
    def eager_for_export(self):
        proc = getattr(self, "processor", None)
        if proc is None:
            proc = getattr(self, "fuser", None)
        if proc is None:
            yield self
            return

        def _sync_after_swap(swapped: str) -> None:
            if swapped == "temporal_net":
                with contextlib.suppress(Exception):
                    if hasattr(proc, "view_encoders") and ("temporal" in proc.view_encoders):
                        proc.view_encoders["temporal"] = proc.temporal_net
            elif swapped == "perception":
                with contextlib.suppress(Exception):
                    if hasattr(proc, "pair_fusers"):
                        if "spatial|temporal" in proc.pair_fusers:
                            proc.pair_fusers["spatial|temporal"] = proc.perception
                        elif len(proc.pair_fusers) == 1:
                            k = next(iter(proc.pair_fusers.keys()))
                            proc.pair_fusers[k] = proc.perception

        swaps: list[tuple[str, Any]] = []
        with contextlib.suppress(Exception):
            eager_temporal = getattr(self, "_eager_processor_temporal_net", None)
            if isinstance(eager_temporal, nn.Module):
                cur = getattr(proc, "temporal_net", None)
                if cur is not eager_temporal and cur is not None:
                    swaps.append(("temporal_net", cur))
                    proc.temporal_net = eager_temporal
                    _sync_after_swap("temporal_net")
        with contextlib.suppress(Exception):
            eager_perception = getattr(self, "_eager_processor_perception", None)
            if isinstance(eager_perception, nn.Module):
                cur = getattr(proc, "perception", None)
                if cur is not eager_perception and cur is not None:
                    swaps.append(("perception", cur))
                    proc.perception = eager_perception
                    _sync_after_swap("perception")
        try:
            yield self
        finally:
            for name, old in swaps:
                with contextlib.suppress(Exception):
                    setattr(proc, name, old)
                _sync_after_swap(name)

    def _run_forward_core(
        self,
        features: torch.Tensor,
        *,
        export: bool = False,
        temporal_state: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        sanitize_nan: bool = True,
        calibrate_output: bool = True,
        device: Optional[torch.device] = None,
        base_dtype: Optional[torch.dtype] = None,
    ):
        x = self._cast_graph_safe(features, device or self._device, base_dtype or features.dtype)
        x = self.scaler.normalize_x(x)
        b = x.size(0)
        if export:
            tokens, context = self.fuser.forward_export(x)
            next_state = None
        else:
            tokens, context, next_state = self.fuser(
                x,
                temporal_state=temporal_state,
                return_temporal_state=True,
                causal_mask=causal_mask,
            )
        if sanitize_nan:
            tokens = _coerce_tensor(tokens, enabled=True, inplace=not export)
            context = _coerce_tensor(context, enabled=True, inplace=not export)
        assembled = context.reshape(b, -1)
        if self.is_norm_linear and self.linear_branch is not None:
            assembled = assembled + self.linear_branch(
                self._cast_graph_safe(x, self._device, assembled.dtype)
            )
        tokens_centered = (
            tokens - tokens.mean(dim=1, keepdim=True, dtype=tokens.dtype).to(tokens.dtype)
        ).contiguous()
        if export:
            refined = self.temporal_token_collector.forward_export(tokens_centered)
        else:
            refined = self.temporal_token_collector.forward(tokens_centered)[0]
        if sanitize_nan:
            refined = _coerce_tensor(refined, enabled=True, inplace=not export)
        residual = self.fuser.decode(refined, apply_norm=True)
        if sanitize_nan:
            residual = _coerce_tensor(residual, enabled=True, inplace=not export)
        enhanced = residual.reshape(b, -1).to(dtype=assembled.dtype)
        delta = enhanced - assembled
        p = None
        if self.p_gate is not None:
            p = self.p_gate(
                tokens=tokens,
                refined_tokens=refined,
                base=assembled,
                residue=delta,
                fallback_bounds=False,
            ).to(dtype=assembled.dtype)
            clip = float(
                min(
                    max(
                        max(
                            float(getattr(self.p_gate, "clip_eps", 1e-6)),
                            float(getattr(self.p_gate, "eps", 0.0)),
                        ),
                        0.0,
                    ),
                    0.49,
                )
            )
            p = p.clamp(clip, 1.0 - clip)
            y_hat = assembled + p * delta
        else:
            y_hat = assembled + assembled.new_tensor(0.5) * delta
        if sanitize_nan:
            y_hat = _coerce_tensor(y_hat, enabled=True, inplace=not export)
        if calibrate_output:
            y_hat = self.scaler.calibrate(y_hat)
        pred = self.scaler.denormalize_y(y_hat).reshape(b, *self.out_shape)
        if sanitize_nan:
            pred = _coerce_tensor(pred, enabled=True, inplace=not export)
        return pred, next_state, p, assembled, enhanced, delta, tokens, refined

    def forward_export(self, features: torch.Tensor) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            raise TypeError("forward_export expects Tensor")
        base_dtype = None
        param = next(self.parameters(), None)
        if param is not None:
            base_dtype = param.dtype
        return self._run_forward_core(
            features,
            export=True,
            sanitize_nan=False,
            calibrate_output=True,
            base_dtype=base_dtype,
        )[0]

    def forward_stream(
        self,
        features: torch.Tensor,
        *args: Any,
        temporal_state: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        calibrate_output: bool = True,
        sanitize_nan: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(features, torch.Tensor):
            raise TypeError("forward_stream expects a Tensor input")
        pred, next_state, _, _, _, _, _, _ = self._run_forward_core(
            features,
            export=False,
            temporal_state=temporal_state,
            causal_mask=causal_mask,
            sanitize_nan=sanitize_nan,
            calibrate_output=calibrate_output,
        )
        return pred, next_state

    def _compute_aux_losses(
        self,
        loss_val,
        y_hat,
        assembled,
        enhanced,
        tokens,
        p,
        z_true,
        features_t,
        is_cls_loss,
        edge_reg_low,
        edge_reg_high,
    ):
        if not (self.training and torch.is_grad_enabled()):
            return loss_val
        aux_total = y_hat.new_tensor(0.0, dtype=y_hat.dtype)
        aux_used = False
        if self.unsup_xx_weight > 0.0 and self.x_recon_head is not None:
            loss_xx = F.smooth_l1_loss(
                self.x_recon_head(tokens.mean(dim=1)).float(),
                features_t.float(),
                reduction="mean",
            )
            aux_total += self.unsup_xx_weight * loss_xx.to(aux_total.dtype)
            aux_used = True
        if self.unsup_yy_weight > 0.0:
            if p is not None:
                if p.dim() != 2 or p.shape[1] == 1:
                    w = p.detach().squeeze(-1).clamp(min=0.0)
                else:
                    w = p.detach().mean(dim=1).clamp(min=0.0)
                loss_yy = (
                    F.smooth_l1_loss(assembled, y_hat.detach(), reduction="none").mean(dim=1) * w
                ).mean()
            else:
                loss_yy = F.smooth_l1_loss(assembled, y_hat.detach(), reduction="mean")
            aux_total += self.unsup_yy_weight * loss_yy.to(aux_total.dtype)
            aux_used = True
        if self.p_prior_weight > 0.0 and p is not None:
            clip_eps = max(
                float(getattr(self.p_gate, "clip_eps", 1e-6)),
                float(getattr(self.p_gate, "eps", 0.0)),
            )
            p01 = p.squeeze(-1).clamp(min=clip_eps, max=1.0 - clip_eps)
            loss_p = -(
                ((self.p_prior_alpha - 1.0) * torch.log(p01))
                + ((self.p_prior_beta - 1.0) * torch.log1p(-p01))
            ).mean()
            aux_total += self.p_prior_weight * loss_p.to(aux_total.dtype)
            aux_used = True
        if p is not None and (
            self.p_gate_budget_weight > 0.0
            or self.p_gate_tv_weight > 0.0
            or self.p_gate_teacher_weight > 0.0
        ):
            gate_eps = 1e-6
            p_floor = 0.0
            p_ceil = 1.0
            if self.p_gate is not None:
                with contextlib.suppress(Exception):
                    gate_eps = float(getattr(self.p_gate, "clip_eps", gate_eps))
                with contextlib.suppress(Exception):
                    gate_eps = max(gate_eps, float(getattr(self.p_gate, "eps", 0.0)))
                with contextlib.suppress(Exception):
                    p_floor = float(getattr(self.p_gate, "p_floor", p_floor))
                with contextlib.suppress(Exception):
                    p_ceil = float(getattr(self.p_gate, "p_ceil", p_ceil))
            gate_eps = max(0.0, float(gate_eps))
            if self.p_gate_budget_weight > 0.0:
                tgt = float(self.p_gate_budget_target)
                if p_ceil >= p_floor:
                    tgt = max(p_floor, min(p_ceil, tgt))
                loss_budget = (p.to(dtype=torch.float32).mean() - p.new_tensor(tgt)).square()
                aux_total += self.p_gate_budget_weight * loss_budget.to(aux_total.dtype)
                aux_used = True
            p_grid = None
            w_grid = None
            tile_shape = None
            if (self.p_gate_tv_weight > 0.0 or self.p_gate_teacher_weight > 0.0) and self.out_shape:
                tile_shape = _normalize_tile_shape(
                    getattr(self, "p_gate_tile_shape", None), self.out_shape
                )
            if (self.p_gate_tv_weight > 0.0 or self.p_gate_teacher_weight > 0.0) and self.out_shape:
                try:
                    if tile_shape is not None:
                        p_grid = _reduce_flat_to_grid(
                            p.to(dtype=torch.float32),
                            self.out_shape,
                            tile_shape,
                            reduce="mean",
                            eps=gate_eps,
                            work_dtype=torch.float32,
                        )
                        w_grid = _tile_counts_grid(
                            self.out_shape,
                            tile_shape,
                            device=p_grid.device,
                            dtype=torch.float32,
                        ).unsqueeze(0)
                    else:
                        p_grid = p.to(dtype=torch.float32).mean(dim=1, keepdim=True)
                except Exception:
                    p_grid = None
                    w_grid = None
            if self.p_gate_tv_weight > 0.0 and p_grid is not None:
                tv = _tv_loss_grid(p_grid, power=float(self.p_gate_tv_power), eps=gate_eps)
                aux_total += self.p_gate_tv_weight * tv.to(dtype=aux_total.dtype)
                aux_used = True
            if (
                self.p_gate_teacher_weight > 0.0
                and p_grid is not None
                and z_true is not None
                and (not is_cls_loss)
            ):
                base_det = assembled.detach().to(dtype=torch.float32)
                ref_det = enhanced.detach().to(dtype=torch.float32)
                tgt_det = z_true.detach().to(device=base_det.device, dtype=torch.float32)
                err_base = (base_det - tgt_det).square()
                err_ref = (ref_det - tgt_det).square()
                improve = err_base - err_ref
                if bool(self.p_gate_teacher_relu):
                    improve = F.relu(improve)
                scale = 0.5 * (err_base + err_ref)
                if tile_shape is not None:
                    imp_grid = _reduce_flat_to_grid(
                        improve,
                        self.out_shape,
                        tile_shape,
                        reduce="mean",
                        eps=gate_eps,
                        work_dtype=torch.float32,
                    )
                    scale_grid = _reduce_flat_to_grid(
                        scale,
                        self.out_shape,
                        tile_shape,
                        reduce="mean",
                        eps=gate_eps,
                        work_dtype=torch.float32,
                    )
                else:
                    imp_grid = improve.mean(dim=1, keepdim=True)
                    scale_grid = scale.mean(dim=1, keepdim=True)
                scale_grid = torch.clamp(scale_grid, min=float(gate_eps))
                temp = max(float(self.p_gate_teacher_temp), float(gate_eps))
                score = ((imp_grid / scale_grid - float(self.p_gate_teacher_tau)) / temp).clamp(
                    min=-20.0, max=20.0
                )
                p01 = torch.sigmoid(score)
                p_target = p01 * float(p_ceil - p_floor) + float(p_floor)
                lo = float(p_floor) + float(gate_eps)
                hi = float(p_ceil) - float(gate_eps)
                if hi > lo:
                    p_target = p_target.clamp(min=lo, max=hi)
                if w_grid is not None:
                    w = w_grid.to(device=p_grid.device, dtype=torch.float32)
                    loss_teacher = (
                        (p_grid.to(dtype=torch.float32) - p_target).square() * w
                    ).sum() / w.sum().clamp_min(float(gate_eps))
                else:
                    loss_teacher = F.mse_loss(
                        p_grid.to(dtype=torch.float32), p_target, reduction="mean"
                    )
                aux_total += self.p_gate_teacher_weight * loss_teacher.to(dtype=aux_total.dtype)
                aux_used = True
        if self.p_gate_edge_reg_weight_low > 0.0 and edge_reg_low is not None:
            aux_total += self.p_gate_edge_reg_weight_low * edge_reg_low.to(dtype=aux_total.dtype)
            aux_used = True
        if self.p_gate_edge_reg_weight_high > 0.0 and edge_reg_high is not None:
            aux_total += self.p_gate_edge_reg_weight_high * edge_reg_high.to(dtype=aux_total.dtype)
            aux_used = True
        if aux_used:
            return (loss_val + aux_total) if (loss_val is not None) else aux_total
        return loss_val

    def _auto_microbatch(
        self, features: torch.Tensor | TensorDictBase, device: torch.device
    ) -> int:
        if isinstance(features, TensorDictBase):
            X = None
            with contextlib.suppress(Exception):
                fkey = get_feature_key(features)
                X = features.get(fkey, None)
            if X is None:
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
        mb_size = _autofit_microbatch(device=device, hard_max=hard_max, per_sample_bytes=per_sample)
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
        | Tuple[
            torch.Tensor,
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            Optional[torch.Tensor],
        ]
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
        infer_mode = (not grad_enabled) and (not is_export_or_trace())
        sanitize_enabled = bool(sanitize_nan)
        sanitize_inplace = bool(sanitize_enabled and infer_mode)
        td_input: TensorDictBase | None = None
        if isinstance(features, TensorDictBase):
            td_input = features
            td_labels_flat = td_input.get("labels_flat", None)
            fkey = get_feature_key(td_input)
            td_features = td_input.get(fkey, None)
            if td_features is None:
                raise KeyError(
                    f"TensorDict input requires a feature column (got key={fkey!r} but value is None)"
                )
            features = td_features
            if labels_flat is None:
                if isinstance(td_labels_flat, torch.Tensor):
                    labels_flat = td_labels_flat
                else:
                    lkey = get_label_key(td_input, required=False)
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
        device = _infer_module_device(self.fuser, self._device)
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
        safety_margin_pow2 = 3
        try:
            safety_margin_pow2 = int(getattr(self.__config, "safety_margin_pow2", 3))
        except Exception:
            safety_margin_pow2 = 3
        safety_margin_pow2 = max(0, min(30, safety_margin_pow2))
        dev_index = int(device.index) if getattr(device, "index", None) is not None else -1
        if meta is None:
            cache_key = (
                device.type,
                dev_index,
                amp_candidates,
                None,
                int(safety_margin_pow2),
            )
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
        is_cls_loss = (
            isinstance(net_loss, (nn.CrossEntropyLoss, nn.NLLLoss))
            if net_loss is not None
            else False
        )
        has_any_loss = (
            (net_loss is not None) or (global_loss is not None) or (local_loss is not None)
        )
        has_supervision = labels_flat is not None and has_any_loss
        is_train_path = bool(self.training and grad_enabled and has_supervision)
        _did_unshard_fuser = False
        _unshard = getattr(self.fuser, "unshard", None)
        _reshard = getattr(self.fuser, "reshard", None)
        if callable(_unshard):
            try:
                _unshard(async_op=False)
                _did_unshard_fuser = True
            except TypeError:
                try:
                    _unshard()
                    _did_unshard_fuser = True
                except Exception:
                    _did_unshard_fuser = False
            except Exception:
                _did_unshard_fuser = False
        try:
            requested_base = getattr(self, "base_dtype", None) or getattr(self, "_base_dtype", None)
            if requested_base is not None:
                base_dtype = requested_base
            else:
                try:
                    base_dtype = next(self.parameters()).dtype
                except Exception:
                    base_dtype = torch.float32 if amp_enabled else amp_dtype
            if isinstance(x_scaled, torch.Tensor) and x_scaled.device != device:
                x_scaled = x_scaled.to(device=device, non_blocking=True)
            features_t = x_scaled.to(dtype=base_dtype) if x_scaled.dtype != base_dtype else x_scaled
            if features_t.ndim != 2 or features_t.shape[1] != self.in_dim:
                raise ValueError(
                    f"Expected features shaped (B, {self.in_dim}), got {tuple(features_t.shape)}"
                )
            b = int(features_t.shape[0])
            do_auto_mb = False
            with self._runtime_lock:
                if self._auto_microbatch_pending:
                    self._auto_microbatch_pending = False
                    do_auto_mb = True
            if do_auto_mb:
                try:
                    mb_enc = self._auto_microbatch(features_t, device)
                    with self._runtime_lock:
                        self.microbatch = max(1, int(mb_enc))
                except Exception:
                    with self._runtime_lock:
                        self.microbatch = max(1, int(getattr(self, "microbatch", 64) or 64))
            with self._runtime_lock:
                mb_cfg = int(self.microbatch) if self.microbatch else int(b)
            mb = max(1, min(int(b), mb_cfg))

            def _encode(inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                with (
                    Autocast.float(device, metadata=meta)
                    if amp_enabled
                    else Autocast.suspend(device)
                ):
                    return self.fuser(inp)

            enc_ctx = inference_mode(self.fuser) if infer_mode else contextlib.nullcontext()
            with enc_ctx:
                tokens, context = cast(
                    Tuple[torch.Tensor, torch.Tensor],
                    _prealloc_microbatch(
                        features_t,
                        mb,
                        _encode,
                        pad_to=int(mb) if self._pad_compiled_microbatch else None,
                        out_dtype=base_dtype,
                        cast_slice=lambda t: self._cast_graph_safe(t, device, base_dtype),
                        stage="encoder",
                    ),
                )
            tokens = _coerce_tensor(tokens, enabled=sanitize_enabled, inplace=sanitize_inplace)
            context = _coerce_tensor(context, enabled=sanitize_enabled, inplace=sanitize_inplace)
            if int(tokens.shape[0]) != int(b):
                raise RuntimeError(
                    "Internal error: token batch mismatch after microbatch concat. "
                    f"got={int(tokens.shape[0])}, expected={int(b)}"
                )
            assembled = context.reshape(b, -1)
            if self.is_norm_linear and self.linear_branch is not None:
                bl = self.linear_branch(self._cast_graph_safe(features_t, device, assembled.dtype))
                assembled = assembled + bl
            mean_dtype = torch.float32 if amp_enabled else tokens.dtype
            mean = tokens.mean(dim=1, keepdim=True, dtype=mean_dtype)
            tokens_centered = tokens - mean.to(dtype=tokens.dtype)
            if not tokens_centered.is_contiguous():
                tokens_centered = tokens_centered.contiguous()
            if is_train_path:
                tokens_centered = tokens_centered.detach()
            refined_tokens = self.temporal_token_collector.run(
                tokens_centered,
                device=device,
                meta=meta,
                amp_enabled=amp_enabled,
                auto_microbatch_fn=lambda t: self._auto_microbatch(t, device),
                graph_break_fn=graph_break,
            )
            refined_tokens = _coerce_tensor(
                refined_tokens,
                enabled=sanitize_enabled,
                inplace=sanitize_inplace,
            )
            ctrl_mb = max(1, min(int(b), int(self.temporal_token_collector.microbatch) or int(b)))
            graph_break()
            processor_ctx = inference_mode(self.fuser) if infer_mode else contextlib.nullcontext()
            with processor_ctx:
                dc = getattr(self, "_decode_compiled", None)

                def _run_decode_chunk(chunk: torch.Tensor) -> torch.Tensor:
                    with (
                        Autocast.float(device, metadata=meta)
                        if amp_enabled
                        else Autocast.suspend(device)
                    ):
                        if dc is not None:
                            return cast(torch.Tensor, dc(chunk))
                        return self.fuser.decode(chunk, apply_norm=True)

                residual_context = cast(
                    torch.Tensor,
                    _prealloc_microbatch(
                        refined_tokens,
                        ctrl_mb,
                        _run_decode_chunk,
                        pad_to=(
                            int(ctrl_mb)
                            if (dc is not None and self._pad_compiled_microbatch)
                            else None
                        ),
                        stage="decoder",
                    ),
                )
            residual_context = _coerce_tensor(
                residual_context,
                enabled=sanitize_enabled,
                inplace=sanitize_inplace,
            )
            enhanced = residual_context.reshape(b, -1)
            if enhanced.dtype != assembled.dtype:
                enhanced = enhanced.to(dtype=assembled.dtype)
            delta = enhanced - assembled
            z_true: Optional[torch.Tensor] = None
            if labels_flat is not None and not is_cls_loss:
                y_true_raw = labels_flat.to(device=assembled.device)
                if not y_true_raw.is_floating_point():
                    y_true_raw = y_true_raw.to(dtype=torch.float32)
                z_true = self.scaler.normalize_y(y_true_raw).to(
                    device=assembled.device, dtype=assembled.dtype
                )
            p: Optional[torch.Tensor] = None
            edge_reg_low: Optional[torch.Tensor] = None
            edge_reg_high: Optional[torch.Tensor] = None
            if self.p_gate is not None:
                z_min: Optional[torch.Tensor] = None
                z_max: Optional[torch.Tensor] = None
                fallback_bounds = False
                try:
                    y_min = getattr(self.scaler, "y_min", None)
                    y_max = getattr(self.scaler, "y_max", None)
                    y_q_low = getattr(self.scaler, "y_q_low", None)
                    y_q_high = getattr(self.scaler, "y_q_high", None)
                    mean = self.scaler.y_mean.to(device=assembled.device, dtype=assembled.dtype)
                    std = self.scaler.y_std.to(device=assembled.device, dtype=assembled.dtype)
                    denom = std + float(self.scaler.eps)

                    def _finite_pair(lo: object, hi: object) -> bool:
                        if not (isinstance(lo, torch.Tensor) and isinstance(hi, torch.Tensor)):
                            return False
                        try:
                            return bool(torch.isfinite(lo).all().item()) and bool(
                                torch.isfinite(hi).all().item()
                            )
                        except Exception:
                            return False

                    have_minmax = _finite_pair(y_min, y_max)
                    have_quant = _finite_pair(y_q_low, y_q_high)
                    use_quant = bool(getattr(self, "p_gate_bounds_use_quantile", False))
                    clip_quant = bool(getattr(self, "p_gate_bounds_clip_to_minmax", True))
                    ylo_t: Optional[torch.Tensor] = None
                    yhi_t: Optional[torch.Tensor] = None
                    if use_quant and have_quant:
                        ylo_t = cast(torch.Tensor, y_q_low).to(
                            device=assembled.device, dtype=assembled.dtype
                        )
                        yhi_t = cast(torch.Tensor, y_q_high).to(
                            device=assembled.device, dtype=assembled.dtype
                        )
                        if clip_quant and have_minmax:
                            ymin_t = cast(torch.Tensor, y_min).to(
                                device=assembled.device, dtype=assembled.dtype
                            )
                            ymax_t = cast(torch.Tensor, y_max).to(
                                device=assembled.device, dtype=assembled.dtype
                            )
                            ylo_t = torch.maximum(ylo_t, ymin_t)
                            yhi_t = torch.minimum(yhi_t, ymax_t)
                    elif have_minmax:
                        ylo_t = cast(torch.Tensor, y_min).to(
                            device=assembled.device, dtype=assembled.dtype
                        )
                        yhi_t = cast(torch.Tensor, y_max).to(
                            device=assembled.device, dtype=assembled.dtype
                        )
                    if ylo_t is not None and yhi_t is not None:
                        z_min = (ylo_t - mean) / denom
                        z_max = (yhi_t - mean) / denom
                    else:
                        if bool(getattr(self, "p_gate_fallback_enabled", False)):
                            k_low_buf = getattr(self, "p_gate_fallback_k_low_buf", None)
                            k_high_buf = getattr(self, "p_gate_fallback_k_high_buf", None)
                            if isinstance(k_low_buf, torch.Tensor):
                                k_low = k_low_buf.to(device=assembled.device, dtype=assembled.dtype)
                            else:
                                k_low = mean.new_tensor(
                                    float(
                                        getattr(
                                            self,
                                            "p_gate_fallback_k_low",
                                            getattr(self, "p_gate_fallback_k", 0.0),
                                        )
                                    )
                                )
                            if isinstance(k_high_buf, torch.Tensor):
                                k_high = k_high_buf.to(
                                    device=assembled.device, dtype=assembled.dtype
                                )
                            else:
                                k_high = mean.new_tensor(
                                    float(
                                        getattr(
                                            self,
                                            "p_gate_fallback_k_high",
                                            getattr(self, "p_gate_fallback_k", 0.0),
                                        )
                                    )
                                )
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
                    and (
                        (self.p_gate_edge_reg_weight_low > 0.0)
                        or (self.p_gate_edge_reg_weight_high > 0.0)
                    )
                    and (z_min is not None)
                    and (z_max is not None)
                    and ((not self.p_gate_edge_reg_fallback_only) or bool(fallback_bounds))
                )
                if do_edge_reg:
                    p, edge_reg_low, edge_reg_high = self.p_gate(
                        tokens=tokens,
                        refined_tokens=refined_tokens,
                        base=assembled,
                        residue=delta,
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
                        residue=delta,
                        z_min=z_min,
                        z_max=z_max,
                        fallback_bounds=bool(fallback_bounds),
                    )
            if p is not None:
                p_eps = (
                    float(getattr(self.p_gate, "clip_eps", 1e-6))
                    if self.p_gate is not None
                    else 1e-6
                )
                with contextlib.suppress(Exception):
                    p_eps = max(p_eps, float(getattr(self.p_gate, "eps", 0.0)))
                hi = 1.0 - p_eps
                if hi > p_eps:
                    p = p.clamp(min=p_eps, max=hi)
            if p is None:
                y_hat = assembled + assembled.new_tensor(0.5) * delta
            else:
                y_hat = assembled + p * delta
            y_hat = _coerce_tensor(y_hat, enabled=sanitize_enabled, inplace=sanitize_inplace)
            pred = y_hat.reshape(b, *self.out_shape)
            loss_val: Optional[torch.Tensor] = None
            top_component: Optional[torch.Tensor] = None
            bottom_component: Optional[torch.Tensor] = None
            use_global_local = labels_flat is not None and (
                global_loss is not None or local_loss is not None
            )
            use_net = labels_flat is not None and (net_loss is not None) and (not use_global_local)
            if use_global_local:
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
                    if use_base_detach:
                        base_det = assembled.detach()
                        delta_det = enhanced - base_det
                        z_top = _coerce_tensor(
                            base_det
                            + (
                                base_det.new_tensor(0.5) * delta_det if p is None else p * delta_det
                            ),
                            enabled=sanitize_enabled,
                            inplace=sanitize_inplace,
                        )
                    else:
                        z_top = _coerce_tensor(
                            y_hat,
                            enabled=sanitize_enabled,
                            inplace=sanitize_inplace,
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
                    loss_val = cast(torch.Tensor, net_loss(y_hat, tgt))
                else:
                    if z_true is None:
                        raise RuntimeError(
                            "Internal error: z_true missing for regression net_loss path"
                        )
                    tgt = z_true.to(device=y_hat.device, dtype=y_hat.dtype)
                    loss_val = cast(torch.Tensor, net_loss(y_hat, tgt))
            if loss_val is not None and not isinstance(loss_val, torch.Tensor):
                loss_val = torch.as_tensor(loss_val, device=y_hat.device, dtype=y_hat.dtype)
            loss_val = self._compute_aux_losses(
                loss_val,
                y_hat,
                assembled,
                enhanced,
                tokens,
                p,
                z_true,
                features_t,
                is_cls_loss,
                edge_reg_low,
                edge_reg_high,
            )
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
            if _did_unshard_fuser and callable(_reshard):
                with contextlib.suppress(Exception):
                    _reshard()

    def predict(
        self, features: torch.Tensor | TensorDictBase, *args: Any, **kwargs: Any
    ) -> torch.Tensor | TensorDictBase:
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
        return torch.float64 if is_scale_safe(torch.float64, metadata) else candidates[-1]

    @staticmethod
    def _peek_layer(module: nn.Module) -> Optional[torch.Tensor]:
        with contextlib.suppress(StopIteration):
            return next(module.parameters())
        with contextlib.suppress(StopIteration):
            return next(module.buffers())
        return None

    @staticmethod
    def _coerce_metadata(model: nn.Module, metadata: Optional[Dataset[Any]] = None) -> Dataset[Any]:
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
    def _clone_state(src: nn.Module, dst: nn.Module, params_dtype: Optional[torch.dtype]) -> None:
        try:
            state = src.state_dict()
        except (RuntimeError, AttributeError):
            return
        try:
            dst.load_state_dict(state, strict=False)
            return
        except Exception:
            pass
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
                    if apply_te_rms_norm and rms_cls is not None and isinstance(child, rms_cls):
                        replacement = ModelPolicy._nvidia_rms_norm(child, params_dtype, te)
                if replacement is not None:
                    setattr(parent, name, replacement)
                    converted += 1
                    continue
                converted += _convert(child)
            return converted

        count = _convert(model)
        if count:
            clear_model_cache(model)
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
            clear_model_cache(model)
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
            model, attn_swapped = ModelPolicy._to_nvidia_attention(model, params_dtype=params_dtype)
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
                getattr(module.__class__, "__module__", "").startswith("transformer_engine")
                for module in model.modules()
            )
        )
        if te_present:
            setattr(model, "__fp8_inference_te__", True)
            clear_model_cache(model)
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
            if not any(is_scale_safe(dtype, meta, safety_margin=2.0) for dtype in float8_dtypes):
                _log_info(logger, "[FP8] training disabled: data scale exceeds float8 range")
                Autocast.configure(model, metadata=meta)
                return (model, False, "data scale")
        params_dtype = ModelPolicy.negotiate(device, metadata=meta)
        for backend in ("te", "torchao"):
            if backend == "te":
                m2, ok2, why = ModelPolicy._enable_nvidia_training(model, params_dtype, logger)
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
            if not any(is_scale_safe(dtype, meta, safety_margin=2.0) for dtype in float8_dtypes):
                _log_info(logger, "[FP8] inference disabled: data scale exceeds float8 range")
                Autocast.configure(model, metadata=meta)
                return (model, False, "data scale")
        params_dtype = ModelPolicy.negotiate(device, metadata=meta)
        dynamic_activations = not (
            getattr(meta, "has_scale", False) and getattr(meta, "scale_is_integral", None) is True
        )
        order = ("te_swap", "te_present", "ao")
        for step in order:
            if step == "te_swap":
                m2, ok2, why = ModelPolicy._enable_nvidia_inference(model, params_dtype, logger)
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
            getattr(meta, "has_scale", False) and getattr(meta, "scale_is_integral", None) is True
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
            getattr(meta, "has_scale", False) and getattr(meta, "scale_is_integral", None) is True
        )
        m2, ok, why = Quantization._enable_ptq(
            model, dynamic_activations=dynamic_activations, logger=logger
        )
        Autocast.configure(m2 if ok else model, metadata=meta)
        return (m2, ok, why)


class LossWeightPolicy(Protocol):
    def weights(self) -> Tuple[float, float]: ...

    def update(
        self,
        top_loss: Optional[torch.Tensor],
        bottom_loss: Optional[torch.Tensor],
    ) -> None:
        raise NotImplementedError


class Quantization:
    @staticmethod
    def is_qat_available() -> bool:
        _import_torchao_quantization()
        return bool(_qp is not None)

    @staticmethod
    def is_ptq_available() -> bool:
        _import_torchao_quantization()
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
        _import_torchao_quantization()
        if _qp is None:
            raise RuntimeError("torchao.quantization.quant_primitives unavailable")
        _log_debug(
            logger,
            f"[INT8][QAT] prepare(dynamic_activations={dynamic_activations}, group={group_size})",
        )
        try:
            from torchao.quantization.fake_quant import (
                FakeQuantizeConfig,
                Int8ActivationConfig,
                Int8WeightConfig,
            )
            from torchao.quantization.fake_quant import prepare_qat_ as _prepare_qat

            cfg = FakeQuantizeConfig(
                activation=Int8ActivationConfig(dynamic=bool(dynamic_activations)),
                weight=Int8WeightConfig(group_size=int(group_size)),
            )
            _prepare_qat(model, cfg)
            clear_model_cache(model)
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
        _import_torchao_quantization()
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
            if _Int8WeightOnlyConfig is None:
                return (model, False, "Int8WeightOnlyConfig unavailable")
            cfg = _Int8WeightOnlyConfig(group_size=int(group_size))
            why = "int8_weight_only"
        try:
            _log_info(logger, f"[INT8][PTQ] applying {why} (group={group_size})")
            _PTQ_IMPL(model, cfg)
            clear_model_cache(model)
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
