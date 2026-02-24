# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import logging
import math
import uuid
import weakref
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Self,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.concurrency import Mutex, is_gil_enabled
from ..core.config import ModelConfig
from ..core.datatypes import env_bool, env_first_int, env_int, env_str
from ..core.policies import LossWeightPolicy, PrecisionPolicy
from ..core.precision import AutocastState, StatefulAutocast, StatelessAutocast
from ..core.system import (
    CPU,
    empty_device_cache,
    get_device,
    get_runtime_cfg,
    is_oom_error,
    set_runtime_cfg,
)
from ..core.tensor import is_meta_or_fake_tensor, symint_safe_expand_as
from ..data.collate import get_feature_key, get_label_key
from ..runtime.distributed import _from_hsdp_module
from .blocks import (
    LongNet,
    Perceiver,
    RetNet,
    _autofit_microbatch,
    _coerce_preset,
    _coerce_tensor,
    _infer_module_device,
    _prealloc_microbatch,
    _size_of_retnet,
    norm_layer,
    stochastic_depth_schedule,
)
from .graph import (
    canonicalize_compile_mode,
    coerce_checkpoint,
    compile as compile_module,
    cudagraph_mark_step_begin,
    cudagraph_mark_step_end,
    graph_break,
    inference_mode,
    is_export_or_trace,
    is_symbolic,
    torch_compiler_disable,
    torch_compiler_supported,
)
from .layers import Recorder, Scaler, SigmoidGate
from tensordict import TensorDictBase
_LOGGER = logging.getLogger(__name__)


def _is_process_group(obj: object) -> bool:
    if obj is None:
        return False
    with contextlib.suppress(Exception):
        from torch.distributed.distributed_c10d import ProcessGroup

        return isinstance(obj, ProcessGroup)
    return False


def _all_reduce_sum(t: torch.Tensor, pg: object | None) -> None:
    try:
        dist = torch.distributed
        if not (dist.is_available() and dist.is_initialized()):
            return
    except Exception:
        return

    if pg is None or (not _is_process_group(pg)):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    else:
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=pg)


def update_delta_gate_auto_k(
    target_module: object,
    *args: Any,
    step: int,
    pg: object | None = None,
    local_rank: int = 0,
) -> None:
    if target_module is None:
        return
    if not bool(getattr(target_module, "delta_gate_auto_k_enabled", False)):
        return

    gate = getattr(target_module, "delta_gate", None)
    if gate is None or not hasattr(gate, "consume_fallback_stats"):
        return

    interval = int(
        getattr(target_module, "delta_gate_auto_k_interval", 0) or 0
    )
    if interval <= 0:
        return

    warmup = int(getattr(target_module, "delta_gate_auto_k_warmup", 0) or 0)
    if int(step) < int(warmup):
        if int(step) % max(1, int(interval)) == 0:
            with contextlib.suppress(Exception):
                gate.consume_fallback_stats()
        return

    if int(step) % int(interval) != 0:
        return

    step_buf = getattr(target_module, "delta_gate_auto_k_step_buf", None)
    if isinstance(step_buf, torch.Tensor):
        with contextlib.suppress(Exception):
            step_buf.fill_(int(step))

    stats = gate.consume_fallback_stats()
    if not isinstance(stats, torch.Tensor) or stats.numel() < 6:
        return

    if pg is not None or (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    ):
        with contextlib.suppress(Exception):
            _all_reduce_sum(stats, pg)

    count = float(stats[0].item())
    if not math.isfinite(count) or count <= 0.0:
        return

    active_low_rate = float((stats[1] / stats[0]).item())
    active_high_rate = float((stats[2] / stats[0]).item())
    width_mean = float((stats[3] / stats[0]).item())
    edge_low_rate = float((stats[4] / stats[0]).item())
    edge_high_rate = float((stats[5] / stats[0]).item())

    alpha = float(getattr(target_module, "delta_gate_auto_k_ema_alpha", 0.1))
    alpha = max(0.0, min(1.0, alpha))

    ema_low_buf = getattr(target_module, "delta_gate_auto_k_ema_low_buf", None)
    ema_high_buf = getattr(
        target_module, "delta_gate_auto_k_ema_high_buf", None
    )
    ema_low_prev = (
        float(ema_low_buf.item())
        if isinstance(ema_low_buf, torch.Tensor)
        else 0.0
    )
    ema_high_prev = (
        float(ema_high_buf.item())
        if isinstance(ema_high_buf, torch.Tensor)
        else 0.0
    )
    ema_low_new = (1.0 - alpha) * ema_low_prev + alpha * active_low_rate
    ema_high_new = (1.0 - alpha) * ema_high_prev + alpha * active_high_rate

    if isinstance(ema_low_buf, torch.Tensor):
        with contextlib.suppress(Exception):
            ema_low_buf.fill_(float(ema_low_new))
    if isinstance(ema_high_buf, torch.Tensor):
        with contextlib.suppress(Exception):
            ema_high_buf.fill_(float(ema_high_new))

    ema_overall = 0.5 * (float(ema_low_new) + float(ema_high_new))
    ema_buf = getattr(target_module, "delta_gate_auto_k_ema_buf", None)
    if isinstance(ema_buf, torch.Tensor):
        with contextlib.suppress(Exception):
            ema_buf.fill_(float(ema_overall))

    edge_enabled = bool(
        getattr(target_module, "delta_gate_auto_k_edge_enabled", False)
    )
    edge_alpha = float(
        getattr(target_module, "delta_gate_auto_k_edge_ema_alpha", alpha)
    )
    edge_alpha = max(0.0, min(1.0, edge_alpha))

    edge_ema_low_buf = getattr(
        target_module, "delta_gate_auto_k_edge_ema_low_buf", None
    )
    edge_ema_high_buf = getattr(
        target_module, "delta_gate_auto_k_edge_ema_high_buf", None
    )
    edge_ema_buf = getattr(
        target_module, "delta_gate_auto_k_edge_ema_buf", None
    )

    edge_ema_low_prev = (
        float(edge_ema_low_buf.item())
        if isinstance(edge_ema_low_buf, torch.Tensor)
        else 0.0
    )
    edge_ema_high_prev = (
        float(edge_ema_high_buf.item())
        if isinstance(edge_ema_high_buf, torch.Tensor)
        else 0.0
    )
    edge_ema_low_new = (
        1.0 - edge_alpha
    ) * edge_ema_low_prev + edge_alpha * edge_low_rate
    edge_ema_high_new = (
        1.0 - edge_alpha
    ) * edge_ema_high_prev + edge_alpha * edge_high_rate

    if isinstance(edge_ema_low_buf, torch.Tensor):
        with contextlib.suppress(Exception):
            edge_ema_low_buf.fill_(float(edge_ema_low_new))
    if isinstance(edge_ema_high_buf, torch.Tensor):
        with contextlib.suppress(Exception):
            edge_ema_high_buf.fill_(float(edge_ema_high_new))
    if isinstance(edge_ema_buf, torch.Tensor):
        with contextlib.suppress(Exception):
            edge_ema_buf.fill_(
                float(0.5 * (edge_ema_low_new + edge_ema_high_new))
            )

    k_low_buf = getattr(target_module, "delta_gate_fallback_k_low_buf", None)
    k_high_buf = getattr(target_module, "delta_gate_fallback_k_high_buf", None)
    k_legacy_buf = getattr(target_module, "delta_gate_fallback_k_buf", None)

    use_legacy = not (
        isinstance(k_low_buf, torch.Tensor)
        and isinstance(k_high_buf, torch.Tensor)
    )
    if use_legacy:
        if not isinstance(k_legacy_buf, torch.Tensor):
            return
        k_low_buf = k_legacy_buf
        k_high_buf = k_legacy_buf

    k_low_prev = (
        float(k_low_buf.item()) if isinstance(k_low_buf, torch.Tensor) else 0.0
    )
    k_high_prev = (
        float(k_high_buf.item())
        if isinstance(k_high_buf, torch.Tensor)
        else 0.0
    )
    k_low_new = k_low_prev
    k_high_new = k_high_prev

    target = float(
        getattr(target_module, "delta_gate_auto_k_target_tight", 0.02)
    )
    tol = float(getattr(target_module, "delta_gate_auto_k_tolerance", 0.5))
    hi = target * (1.0 + tol)
    lo = max(0.0, target * (1.0 - tol))

    step_up_low = float(
        getattr(target_module, "delta_gate_auto_k_step_up_low", 0.1)
    )
    step_down_low = float(
        getattr(target_module, "delta_gate_auto_k_step_down_low", 0.02)
    )
    step_up_high = float(
        getattr(target_module, "delta_gate_auto_k_step_up_high", 0.1)
    )
    step_down_high = float(
        getattr(target_module, "delta_gate_auto_k_step_down_high", 0.02)
    )

    edge_target = float(
        getattr(target_module, "delta_gate_auto_k_target_edge", 0.05)
    )
    edge_tol = float(
        getattr(target_module, "delta_gate_auto_k_edge_tolerance", 0.5)
    )
    edge_hi = edge_target * (1.0 + edge_tol)

    edge_step_down_low = float(
        getattr(target_module, "delta_gate_auto_k_edge_step_down_low", 0.01)
    )
    edge_step_down_high = float(
        getattr(target_module, "delta_gate_auto_k_edge_step_down_high", 0.01)
    )

    k_min = float(getattr(target_module, "delta_gate_auto_k_min", 1.0))
    k_max = float(getattr(target_module, "delta_gate_auto_k_max", 16.0))
    if k_max < k_min:
        k_max = k_min

    edge_low_eff = (
        float(edge_ema_low_new)
        if math.isfinite(edge_ema_low_new)
        else float(edge_low_rate)
    )
    edge_high_eff = (
        float(edge_ema_high_new)
        if math.isfinite(edge_ema_high_new)
        else float(edge_high_rate)
    )

    if math.isfinite(ema_low_new):
        if ema_low_new > hi and step_up_low > 0.0:
            k_low_new = k_low_prev * (1.0 + step_up_low)
        elif ema_low_new < lo and step_down_low > 0.0:
            k_low_new = k_low_prev * max(0.0, (1.0 - step_down_low))
        elif (
            edge_enabled
            and math.isfinite(edge_low_eff)
            and edge_low_eff > edge_hi
            and edge_step_down_low > 0.0
        ):
            k_low_new = k_low_prev * max(0.0, (1.0 - edge_step_down_low))

    if math.isfinite(ema_high_new):
        if ema_high_new > hi and step_up_high > 0.0:
            k_high_new = k_high_prev * (1.0 + step_up_high)
        elif ema_high_new < lo and step_down_high > 0.0:
            k_high_new = k_high_prev * max(0.0, (1.0 - step_down_high))
        elif (
            edge_enabled
            and math.isfinite(edge_high_eff)
            and edge_high_eff > edge_hi
            and edge_step_down_high > 0.0
        ):
            k_high_new = k_high_prev * max(0.0, (1.0 - edge_step_down_high))

    if math.isfinite(k_low_new):
        k_low_new = max(k_min, min(k_max, k_low_new))
    if math.isfinite(k_high_new):
        k_high_new = max(k_min, min(k_max, k_high_new))

    k_low_changed = bool(
        math.isfinite(k_low_new) and abs(k_low_new - k_low_prev) > 1e-12
    )
    k_high_changed = bool(
        math.isfinite(k_high_new) and abs(k_high_new - k_high_prev) > 1e-12
    )
    k_changed = bool(k_low_changed or k_high_changed)

    if k_changed:
        with contextlib.suppress(Exception):
            if isinstance(k_low_buf, torch.Tensor):
                k_low_buf.fill_(float(k_low_new))
            if isinstance(k_high_buf, torch.Tensor):
                k_high_buf.fill_(float(k_high_new))

        if isinstance(k_legacy_buf, torch.Tensor) and not use_legacy:
            with contextlib.suppress(Exception):
                k_legacy_buf.fill_(float(0.5 * (k_low_new + k_high_new)))

        with contextlib.suppress(Exception):
            setattr(
                target_module,
                "delta_gate_fallback_enabled",
                bool(k_low_new > 0.0 and k_high_new > 0.0),
            )

        upd_buf = getattr(target_module, "delta_gate_auto_k_updates_buf", None)
        if isinstance(upd_buf, torch.Tensor):
            with contextlib.suppress(Exception):
                upd_buf.add_(1)

    log_interval = int(
        getattr(target_module, "delta_gate_auto_k_log_interval", 0) or 0
    )
    log_due = (
        bool(k_changed)
        if log_interval <= 0
        else bool(int(step) % int(log_interval) == 0)
    )

    if int(local_rank) == 0 and log_due:
        _LOGGER.info(
            "[delta_gate] auto_k step=%d seen=%d activeL_sma=%.4f activeH_sma=%.4f width_mean=%.4f edgeL_sma=%.4f edgeH_sma=%.4f activeL_ema=%.4f activeH_ema=%.4f edgeL_ema=%.4f edgeH_ema=%.4f kL=%.4f -> %.4f kH=%.4f -> %.4f",
            int(step),
            int(count),
            float(active_low_rate),
            float(active_high_rate),
            float(width_mean),
            float(edge_low_rate),
            float(edge_high_rate),
            float(ema_low_new),
            float(ema_high_new),
            float(edge_ema_low_new),
            float(edge_ema_high_new),
            float(k_low_prev),
            float(k_low_new),
            float(k_high_prev),
            float(k_high_new),
        )


_SUBMODEL_UNSET: object = object()
_META_UNSET: object = object()
TConfig = TypeVar("TConfig")


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
            counts_1d.append(
                torch.tensor([float(d)], device=device, dtype=dtype)
            )
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
    counts = _tile_counts_grid(
        ev, ts, device=blk.device, dtype=blk.dtype
    ).unsqueeze(0)
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


def _dot_product_attention_cls() -> Any:
    try:
        from .kernels import DotProductAttention

        return DotProductAttention
    except Exception:
        return None


class Collector(nn.Module):
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
        self._runtime_lock = Mutex()

    def __getstate__(self: Self) -> dict[str, object]:
        state = super().__getstate__()
        state.pop("_runtime_lock", None)
        return state

    def __setstate__(self: Self, state: dict[str, object]) -> None:
        super().__setstate__(state)
        self._runtime_lock = Mutex()

    @property
    def using(self: Self) -> str:
        return self.backbone.using

    def forward(
        self: Self,
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
        self: Self,
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
        self: Self,
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
                f"Collector.run expects tokens (B,N,D), got shape {tuple(tokens.shape)}"
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
        infer_mode = (not torch.is_grad_enabled()) and (
            not is_export_or_trace()
        )
        controller_ctx = (
            inference_mode(self.backbone)
            if infer_mode
            else contextlib.nullcontext()
        )
        backbone = self.backbone

        def runner(chunk: torch.Tensor) -> torch.Tensor:
            with (
                StatelessAutocast.float(device, metadata=meta)
                if amp_enabled
                else StatelessAutocast.suspend(device)
            ):
                out, _ = backbone(chunk)
            return cast(torch.Tensor, out)

        with controller_ctx:
            refined = cast(
                torch.Tensor,
                _prealloc_microbatch(tokens, mb, runner, stage="controller"),
            )
        return refined


class Template(nn.Module):
    def __init__(
        self: Self,
        in_dim: int,
        tokens: int,
        d_model: int,
        nhead: int,
        depth: int,
        *args: Any,
        mode: str = "spatial",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_type: str = "layernorm",
        weight: float = 1.0,
        eps: float = 1e-6,
        ckpt_enabled: bool = True,
        ckpt_min_bytes: int = 64 * 1024 * 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del args, kwargs
        self.in_dim = int(in_dim)
        self.tokens = max(1, int(tokens))
        self.d_model = int(d_model)
        self.nhead = max(1, int(nhead))
        self.depth = max(1, int(depth))
        self.head_dim = int(self.d_model // max(1, self.nhead))
        self.mode = self._coerce_mode(mode)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)
        self.drop_path = float(drop_path)
        self.norm_type = str(norm_type)
        self.tokenizer = nn.Linear(self.in_dim, self.tokens * self.d_model)
        setattr(self.tokenizer, "_enn_no_ao_quant", True)
        drops = stochastic_depth_schedule(
            float(self.drop_path), int(self.depth)
        )
        self.blocks = nn.ModuleList(
            [
                RetNet(
                    self.d_model,
                    self.nhead,
                    mlp_ratio=float(self.mlp_ratio),
                    dropout=float(self.dropout),
                    drop_path=float(drops[i] if i < len(drops) else 0.0),
                    norm_type=str(self.norm_type),
                    mode=str(self.mode),
                )
                for i in range(int(self.depth))
            ]
        )
        self.norm = norm_layer(self.norm_type, self.d_model)
        self.register_buffer(
            "weight",
            torch.as_tensor(float(weight), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "eps",
            torch.as_tensor(float(eps), dtype=torch.float32),
            persistent=True,
        )
        self._ckpt_enabled = bool(ckpt_enabled)
        self._ckpt_min_bytes = int(ckpt_min_bytes)

    @staticmethod
    def _coerce_mode(mode: str) -> str:
        m = str(mode or "spatial").strip().lower()
        if m in {"s", "spatial", "ss", "sxs"}:
            return "spatial"
        if m in {"t", "temporal", "tt", "txt"}:
            return "temporal"
        raise ValueError(
            f"Unknown mode '{mode}' (expected 'spatial' or 'temporal')"
        )

    def set_mode(self: Self, mode: str) -> None:
        self.mode = self._coerce_mode(mode)

    def set_weight(self: Self, weight: float) -> None:
        self.weight.data = torch.as_tensor(
            float(weight), dtype=self.weight.dtype, device=self.weight.device
        )

    def set_eps(self: Self, eps: float) -> None:
        self.eps.data = torch.as_tensor(
            float(eps), dtype=self.eps.dtype, device=self.eps.device
        )

    @staticmethod
    def _coerce_state_tensor(
        state: Any,
        B: int,
        depth: int,
        nhead: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if state is None:
            return None
        if not isinstance(state, torch.Tensor):
            raise TypeError("state must be a torch.Tensor or None")
        if state.dim() != 4:
            raise ValueError(
                f"state must be shaped (depth,B,H,Dh), got {tuple(state.shape)}"
            )
        if int(state.shape[0]) != int(depth):
            raise ValueError(
                f"state depth mismatch: expected {int(depth)} got {int(state.shape[0])}"
            )
        if int(state.shape[1]) != int(B):
            raise ValueError(
                f"state batch mismatch: expected {int(B)} got {int(state.shape[1])}"
            )
        if int(state.shape[2]) != int(nhead):
            raise ValueError(
                f"state head mismatch: expected {int(nhead)} got {int(state.shape[2])}"
            )
        if int(state.shape[3]) != int(head_dim):
            raise ValueError(
                f"state head_dim mismatch: expected {int(head_dim)} got {int(state.shape[3])}"
            )
        if state.dtype != dtype:
            state = state.to(dtype=dtype)
        if state.device != device:
            state = state.to(device=device)
        return state.contiguous()

    @torch_compiler_disable(
        reason="Template.tokenizer can collapse under cudagraph/compile; keep eager",
        recursive=False,
    )
    def _tokenize(self: Self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        return (
            self.tokenizer(x.contiguous())
            .reshape(B, self.tokens, self.d_model)
            .contiguous()
        )

    def _tokenize_export(self: Self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        return (
            self.tokenizer(x.contiguous())
            .reshape(B, self.tokens, self.d_model)
            .contiguous()
        )

    def forward(
        self: Self,
        x: torch.Tensor,
        *args: Any,
        state: Optional[torch.Tensor] = None,
        return_state: bool = False,
        causal_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        del args, kwargs
        tokens = self._tokenize_export(x) if is_export_or_trace() else self._tokenize(x)
        m = str(self.mode)
        if m == "spatial":
            if state is not None:
                raise ValueError(
                    "state is only supported when mode=='temporal'"
                )
            if return_state:
                raise ValueError(
                    "return_state is only supported when mode=='temporal'"
                )
            return self._forward_spatial(tokens, causal_mask=causal_mask)
        if m == "temporal":
            return self._forward_temporal(
                tokens,
                state=state,
                return_state=return_state,
                causal_mask=causal_mask,
            )
        raise RuntimeError(f"Invalid Template.mode {m!r}")

    def _forward_spatial(
        self: Self,
        tokens: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = tokens
        do_ckpt = bool(
            self.training
            and torch.is_grad_enabled()
            and bool(getattr(self, "_ckpt_enabled", True))
        )
        if do_ckpt and int(getattr(self, "_ckpt_min_bytes", 0) or 0) > 0:
            try:
                est = int(_size_of_retnet(x, self.blocks[0], mode="spatial"))
            except Exception:
                est = 0
            do_ckpt = bool(
                est >= int(getattr(self, "_ckpt_min_bytes", 0) or 0)
            )
        for blk in self.blocks:
            if do_ckpt:

                def _f(t: torch.Tensor, _blk: RetNet = blk) -> torch.Tensor:
                    if torch.is_grad_enabled():
                        _from_hsdp_module(self)
                        _from_hsdp_module(_blk)
                    y, _ = _blk(
                        t, causal_mask=causal_mask, state=None, mode="spatial"
                    )
                    return y

                x = cast(
                    torch.Tensor,
                    coerce_checkpoint(
                        _f, x, use_reentrant=True, preserve_rng_state=True
                    ),
                )
            else:
                x, _ = blk(
                    x, causal_mask=causal_mask, state=None, mode="spatial"
                )
        return self.norm(x)

    def _forward_temporal(
        self: Self,
        tokens: torch.Tensor,
        *args: Any,
        state: Optional[torch.Tensor],
        return_state: bool,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> Any:
        x = tokens
        B = x.size(0)
        st_tensor = self._coerce_state_tensor(
            state,
            B=B,
            depth=int(self.depth),
            nhead=int(self.nhead),
            head_dim=int(self.head_dim),
            dtype=x.dtype,
            device=x.device,
        )
        do_ckpt = bool(
            self.training
            and torch.is_grad_enabled()
            and bool(getattr(self, "_ckpt_enabled", True))
            and (st_tensor is None)
            and (not return_state)
        )
        if do_ckpt and int(getattr(self, "_ckpt_min_bytes", 0) or 0) > 0:
            try:
                est = int(_size_of_retnet(x, self.blocks[0], mode="temporal"))
            except Exception:
                est = 0
            do_ckpt = bool(
                est >= int(getattr(self, "_ckpt_min_bytes", 0) or 0)
            )
        next_state: Optional[torch.Tensor] = None
        if return_state:
            next_state = x.new_empty(
                (int(self.depth), B, int(self.nhead), int(self.head_dim))
            )
        for i, blk in enumerate(self.blocks):
            blk_state = None
            if st_tensor is not None and i < int(self.depth):
                blk_state = st_tensor[i]
            if do_ckpt:

                def _f(t: torch.Tensor, _blk: RetNet = blk) -> torch.Tensor:
                    if torch.is_grad_enabled():
                        _from_hsdp_module(self)
                        _from_hsdp_module(_blk)
                    y, _ = _blk(
                        t, causal_mask=causal_mask, state=None, mode="temporal"
                    )
                    return y

                x = cast(
                    torch.Tensor,
                    coerce_checkpoint(
                        _f, x, use_reentrant=True, preserve_rng_state=True
                    ),
                )
            else:
                x, blk_next_state = blk(
                    x,
                    causal_mask=causal_mask,
                    state=blk_state,
                    mode="temporal",
                )
                if next_state is not None:
                    if blk_next_state is None:
                        blk_next_state = x.new_zeros(
                            (B, int(self.nhead), int(self.head_dim))
                        )
                    next_state[i] = blk_next_state
        x = self.norm(x)
        if next_state is not None:
            if not torch.is_grad_enabled():
                next_state = next_state.detach()
            return x, next_state.contiguous()
        return x




_SUBFUSER_BINDINGS = weakref.WeakKeyDictionary()


class SubFuser(nn.Module):

    def __init__(
        self: Self,
        tokens: int,
        *args: Any,
        children: Optional[Sequence[str]] = None,
        reduction: str = "mean",
        weight: float = 1.0,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del args, kwargs
        self.tokens = max(1, int(tokens))
        self.reduction = self._coerce_reduction(reduction)
        self._children: list[str] = (
            [str(c) for c in children] if children is not None else []
        )
        self.register_buffer(
            "weight",
            torch.as_tensor(float(weight), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "eps",
            torch.as_tensor(float(eps), dtype=torch.float32),
            persistent=True,
        )

    def __getstate__(self: Self) -> dict[str, object]:
        return self.__dict__.copy()

    def __setstate__(self: Self, state: dict[str, object]) -> None:
        if "children" in state and "_children" not in state:
            state["_children"] = state.pop("children")
        self.__dict__.update(state)
        try:
            self.reduction = self._coerce_reduction(getattr(self, "reduction", "mean"))
        except Exception:
            self.reduction = "mean"
        if not hasattr(self, "_children"):
            self._children = []

    def _bind(self: Self, owner: object, name: str) -> None:
        try:
            _SUBFUSER_BINDINGS[self] = (weakref.ref(owner), str(name))
        except Exception:
            return

    def _owner_and_name(self: Self) -> tuple[object, str]:
        pair = _SUBFUSER_BINDINGS.get(self)
        if not pair:
            raise RuntimeError(
                "SubFuser is not bound to an owning Fuser; use it via Model/Fuser APIs"
            )
        owner_ref, name = pair
        owner = owner_ref() if callable(owner_ref) else None
        if owner is None:
            raise RuntimeError("SubFuser owning Fuser reference is not available")
        return owner, str(name)

    @staticmethod
    def _coerce_reduction(reduction: str) -> str:
        r = str(reduction or "mean").strip().lower()
        if r in {"mean", "avg", "average", "weighted_mean", "weighted"}:
            return "mean"
        if r in {"linear", "lin", "cooperative", "axbycz"}:
            return "linear"
        raise ValueError(
            f"Unknown reduction {reduction!r} (expected 'mean' or 'linear')"
        )

    def set_reduction(self: Self, reduction: str) -> None:
        self.reduction = self._coerce_reduction(reduction)

    def set_weight(self: Self, weight: float) -> None:
        self.weight.data = torch.as_tensor(
            float(weight), dtype=self.weight.dtype, device=self.weight.device
        )

    def set_eps(self: Self, eps: float) -> None:
        self.eps.data = torch.as_tensor(
            float(eps), dtype=self.eps.dtype, device=self.eps.device
        )

    def get(
        self: Self, node_spec: Optional[str] = None, default: Optional[nn.Module] = None
    ) -> Optional[nn.Module]:
        if node_spec is None or (not str(node_spec).strip()):
            return self
        owner, _ = self._owner_and_name()
        getter = getattr(owner, "get", None)
        if callable(getter):
            return getter(node_spec, default=default)
        return default

    def node_name(self: Self) -> str:
        _, my_name = self._owner_and_name()
        return str(my_name)

    def get_children(
        self: Self, node_spec: Optional[str] = None
    ) -> list[nn.Module]:
        owner, my_name = self._owner_and_name()
        target = (
            my_name if node_spec is None or (not str(node_spec).strip()) else str(node_spec)
        )
        fn = getattr(owner, "get_children", None)
        if callable(fn):
            return list(fn(target))
        return []

    def get_parent(
        self: Self, node_spec: Optional[str] = None
    ) -> Optional[nn.Module]:
        owner, my_name = self._owner_and_name()
        target = (
            my_name if node_spec is None or (not str(node_spec).strip()) else str(node_spec)
        )
        fn = getattr(owner, "get_parent", None)
        if callable(fn):
            return fn(target)
        return None

    def get_subtree(
        self: Self, by: str = "name", include_self: bool = False
    ) -> list[str]:
        owner, my_name = self._owner_and_name()
        fn = getattr(owner, "get_subtree", None)
        if callable(fn):
            return list(fn(my_name, by=by, include_self=include_self))
        return []

    def new(self: Self, name: Optional[str] = None, *args: Any, parent: Optional[str] = None, **kwargs: Any) -> str:
        owner, my_name = self._owner_and_name()
        if parent is None or (not str(parent).strip()):
            parent = my_name
        fn = getattr(owner, "new", None)
        if not callable(fn):
            raise RuntimeError("Owning Fuser does not support new()")
        return fn(name, *args, parent=parent, **kwargs)

    def attach(self: Self, node_spec: str, parent: Optional[str] = None) -> str:
        owner, my_name = self._owner_and_name()
        if parent is None or (not str(parent).strip()):
            parent = my_name
        fn = getattr(owner, "attach", None)
        if not callable(fn):
            raise RuntimeError("Owning Fuser does not support attach()")
        return fn(node_spec, parent=parent)

    def detach(self: Self, node_spec: Optional[str] = None) -> str:
        owner, my_name = self._owner_and_name()
        target = my_name if node_spec is None or (not str(node_spec).strip()) else str(node_spec)
        fn = getattr(owner, "detach", None)
        if not callable(fn):
            raise RuntimeError("Owning Fuser does not support detach()")
        return fn(target)

    def update(self: Self, node_spec: Optional[str] = None, *args: Any, **kwargs: Any) -> str:
        owner, my_name = self._owner_and_name()
        target = my_name if node_spec is None or (not str(node_spec).strip()) else str(node_spec)
        fn = getattr(owner, "update", None)
        if not callable(fn):
            raise RuntimeError("Owning Fuser does not support update()")
        return fn(target, *args, **kwargs)

    def remove(self: Self, node_spec: Optional[str] = None, *args: Any, **kwargs: Any) -> None:
        owner, my_name = self._owner_and_name()
        target = my_name if node_spec is None or (not str(node_spec).strip()) else str(node_spec)
        fn = getattr(owner, "remove", None)
        if not callable(fn):
            raise RuntimeError("Owning Fuser does not support remove()")
        fn(target, *args, **kwargs)

    def forward(self: Self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError(
            "SubFuser is a topology node and must be executed via Fuser.forward()"
        )
class Fuser(nn.Module):
    def __init__(
        self: Self,
        in_dim: int,
        out_shape: Sequence[int],
        config: ModelConfig,
        *args: Any,
        tasks: Optional[Sequence[Mapping[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del args, kwargs
        self.in_dim = int(in_dim)
        self.out_shape = tuple((int(v) for v in out_shape))
        self.out_dim = int(math.prod(self.out_shape) if self.out_shape else 1)
        self.cfg: ModelConfig = config
        self.__enn_instance_config__ = config
        self.d_model = int(config.d_model)
        self.nhead = int(config.heads)
        raw_preset = getattr(config, "preset", None)
        if raw_preset is None and hasattr(config, "modeling_type"):
            raw_preset = getattr(config, "modeling_type", None)
        self.preset = _coerce_preset(raw_preset)
        self.spatial_tokens = max(
            1, int(getattr(config, "spatial_latents", 1))
        )
        self.temporal_tokens = max(
            1, int(getattr(config, "temporal_latents", 1))
        )
        self.fused_tokens = max(
            1, int(self.spatial_tokens + self.temporal_tokens)
        )
        self.mlp_ratio = float(getattr(config, "mlp_ratio", 4.0))
        self.dropout = float(getattr(config, "dropout", 0.0))
        self.drop_path = float(getattr(config, "drop_path", 0.0))
        self.norm_type = str(
            getattr(config, "normalization_method", "layernorm")
        )
        raw_fd = getattr(config, "fuser_depth", None)
        if raw_fd is None:
            raw_fd = max(
                1,
                int(getattr(config, "spatial_depth", 1)),
                int(getattr(config, "temporal_depth", 1)),
            )
        else:
            try:
                raw_fd = int(raw_fd)
            except Exception:
                raw_fd = max(
                    1,
                    int(getattr(config, "spatial_depth", 1)),
                    int(getattr(config, "temporal_depth", 1)),
                )
            if int(raw_fd) <= 0:
                raw_fd = max(
                    1,
                    int(getattr(config, "spatial_depth", 1)),
                    int(getattr(config, "temporal_depth", 1)),
                )
        self.perceiver_depth = max(1, int(raw_fd))
        raw_sa = getattr(config, "fuser_self_attn_layers", 1)
        try:
            raw_sa = int(raw_sa)
        except Exception:
            raw_sa = 1
        self.self_attn_layers = max(0, int(raw_sa))
        self.perceiver = Perceiver(
            d_model=self.d_model,
            nhead=self.nhead,
            num_latents=self.fused_tokens,
            depth=self.perceiver_depth,
            self_attn_layers=self.self_attn_layers,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            drop_path=self.drop_path,
            norm_type=self.norm_type,
        )
        self._perceiver_stateful_autocast_enabled = bool(
            env_bool("ENN_FUSER_PERCEIVER_STATEFUL_AUTOCAST", True)
        )
        pol = str(env_str("ENN_FUSER_PERCEIVER_AUTOCAST_POLICY") or "sticky").strip()
        cd = int(env_int("ENN_FUSER_PERCEIVER_AUTOCAST_COOLDOWN_STEPS", 64) or 64)
        mr = int(env_int("ENN_FUSER_PERCEIVER_AUTOCAST_MAX_RETRIES", 1) or 1)
        self._perceiver_autocast_state = AutocastState(
            policy=pol,
            cooldown_steps=cd,
            max_retries=mr,
        )
        self._perceiver_autocast = StatefulAutocast(
            self._perceiver_autocast_state, name="Fuser.perceiver", logger=_LOGGER
        )
        hid = int(self.d_model * max(1.0, self.mlp_ratio))
        self.norm = norm_layer(self.norm_type, self.d_model)
        self.head_hidden_dim = hid
        self.head = nn.Sequential(
            norm_layer(self.norm_type, self.d_model),
            nn.Linear(self.d_model, hid),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(hid, self.out_dim),
        )
        self.tasks: nn.ModuleDict = nn.ModuleDict()
        self._user_submodels: dict[str, nn.Module] = {}
        self._task_meta: dict[str, dict[str, Any]] = {}
        self._legacy_task_id_to_name: dict[str, str] = {}
        self._topology_lock = Mutex(reentrant=True)
        self._topology_version: int = 0
        self._exec_plan: Optional[dict[str, Any]] = None
        self._root_children: list[str] = []
        self._parent: dict[str, Optional[str]] = {}
        self._root_reduction: str = "mean"
        raw_stream = getattr(config, "stream_task_name", None)
        if raw_stream is None:
            raw_stream = getattr(config, "stream_task_id", None)
        self.stream_task_name: str = (
            str(raw_stream).strip() if raw_stream else ""
        )
        self.stream_task_id: str = self.stream_task_name
        if tasks is None:
            self._init_default_tasks(config)
        else:
            self.rebuild_tasks_from_specs(tasks)
        self._resolve_stream_task_id()
        self._compile_cudagraphs = bool(
            getattr(config, "compile_cudagraphs", False)
        )
        self._decode_graph: nn.Module | None = None
        self._backbone_graph: nn.Module | None = None
        try:
            from .graph import CallArguments
            from .graph import GraphSequential

            class _PackPerceiverArgs(nn.Module):
                def forward(
                    self,
                    all_tokens: torch.Tensor,
                    attn_bias: Optional[torch.Tensor] = None,
                ) -> CallArguments:
                    return CallArguments(
                        args=(all_tokens,), kwargs={"attn_bias": attn_bias}
                    )

            class _PackFusedAndDecode(nn.Module):
                def __init__(self, decode_mod: nn.Module) -> None:
                    super().__init__()
                    self.decode = decode_mod

                def forward(
                    self, fused_tokens: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    ctx = cast(torch.Tensor, self.decode(fused_tokens))
                    return fused_tokens, ctx

            device = next(self.parameters()).device
            self._decode_graph = (
                GraphSequential(
                    steps=[
                        GraphSequential.path("norm"),
                        GraphSequential.mean(dim=1),
                        GraphSequential.path("head"),
                    ],
                    out_shape=self.out_shape,
                    name="decode",
                    root=self,
                )
                .to(device)
                .bind()
            )
            self._backbone_graph = (
                GraphSequential(
                    steps=[
                        GraphSequential.cudagraph_begin(),
                        GraphSequential.own(
                            _PackPerceiverArgs(), name="pack_args"
                        ),
                        GraphSequential.path("perceiver"),
                        GraphSequential.break_graph(),
                        GraphSequential.own(
                            _PackFusedAndDecode(self._decode_graph),
                            name="pack_decode",
                        ),
                        GraphSequential.cudagraph_end(),
                    ],
                    name="backbone",
                    root=self,
                )
                .to(device)
                .bind()
            )
        except Exception:
            self._decode_graph = None
            self._backbone_graph = None

    def __getstate__(self: Self) -> dict[str, object]:
        d = self.__dict__.copy()
        d.pop("_user_submodels", None)
        d.pop("_topology_lock", None)
        d.pop("_exec_plan", None)
        return d

    def __setstate__(self: Self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._user_submodels = {}
        self._topology_lock = Mutex(reentrant=True)
        self._exec_plan = None
        self._topology_version = int(getattr(self, "_topology_version", 0) or 0)
        if not isinstance(getattr(self, "_root_children", None), list):
            self._root_children = []
        if not isinstance(getattr(self, "_parent", None), dict):
            self._parent = {}
        with contextlib.suppress(Exception):
            self._root_reduction = SubFuser._coerce_reduction(
                str(getattr(self, "_root_reduction", "mean") or "mean")
            )
        if not isinstance(getattr(self, "_root_reduction", None), str):
            self._root_reduction = "mean"

        task_names = [str(n) for n in getattr(self, "tasks", {}).keys()]
        if task_names:
            normalized_parent: dict[str, Optional[str]] = {}
            for nm in task_names:
                raw_parent = self._parent.get(nm)
                if raw_parent is None or (not str(raw_parent).strip()):
                    normalized_parent[nm] = None
                    continue
                pk = str(raw_parent).strip()
                if (
                    pk == nm
                    or pk not in self.tasks
                    or (not isinstance(self.tasks[pk], SubFuser))
                ):
                    normalized_parent[nm] = None
                else:
                    normalized_parent[nm] = pk
            self._parent = normalized_parent

            sanitized_roots: list[str] = []
            seen_roots: set[str] = set()
            for name in list(getattr(self, "_root_children", []) or []):
                nm = str(name)
                if (
                    nm in self.tasks
                    and nm not in seen_roots
                    and (self._parent.get(nm) is None)
                ):
                    sanitized_roots.append(nm)
                    seen_roots.add(nm)
            for nm in task_names:
                if (self._parent.get(nm) is None) and (nm not in seen_roots):
                    sanitized_roots.append(nm)
                    seen_roots.add(nm)
            self._root_children = sanitized_roots

            with contextlib.suppress(Exception):
                parent_to_children: dict[str, list[str]] = {}
                for child, parent in self._parent.items():
                    if parent is None:
                        continue
                    p = str(parent)
                    parent_to_children.setdefault(p, []).append(str(child))

                for p_name, mod in getattr(self, "tasks", {}).items():
                    if not isinstance(mod, SubFuser):
                        continue
                    wanted = parent_to_children.get(str(p_name), [])
                    existing = [
                        str(c) for c in list(getattr(mod, "_children", []) or [])
                    ]
                    merged: list[str] = []
                    seen: set[str] = set()
                    for c in existing:
                        if (
                            c in wanted
                            and c in self.tasks
                            and c not in seen
                            and c != str(p_name)
                        ):
                            merged.append(c)
                            seen.add(c)
                    for c in wanted:
                        if c in self.tasks and c not in seen and c != str(p_name):
                            merged.append(c)
                            seen.add(c)
                    mod._children = merged

        with contextlib.suppress(Exception):
            for n, m in getattr(self, "tasks", {}).items():
                if isinstance(m, SubFuser):
                    m._bind(self, str(n))



    def _init_default_tasks(
        self: Self, config: Optional[ModelConfig] = None
    ) -> None:
        cfg = (
            config
            or getattr(self, "cfg", None)
            or getattr(self, "__enn_instance_config__", None)
        )
        if cfg is None:
            raise RuntimeError(
                "Fuser requires a ModelConfig to initialize default tasks"
            )
        pr = getattr(self, "preset", None)
        if pr is None:
            return
        if pr == "spatial":
            self.add_task(
                "spatial",
                mode="spatial",
                tokens=int(self.spatial_tokens),
                depth=int(getattr(cfg, "spatial_depth", 1)),
                weight=1.0,
                eps=1e-6,
                submodel=None,
            )
        elif pr == "temporal":
            self.add_task(
                "temporal",
                mode="temporal",
                tokens=int(self.temporal_tokens),
                depth=int(getattr(cfg, "temporal_depth", 1)),
                weight=1.0,
                eps=1e-6,
                submodel=None,
            )
        else:
            self.add_task(
                "spatial",
                mode="spatial",
                tokens=int(self.spatial_tokens),
                depth=int(getattr(cfg, "spatial_depth", 1)),
                weight=1.0,
                eps=1e-6,
                submodel=None,
            )
            self.add_task(
                "temporal",
                mode="temporal",
                tokens=int(self.temporal_tokens),
                depth=int(getattr(cfg, "temporal_depth", 1)),
                weight=1.0,
                eps=1e-6,
                submodel=None,
            )

    def _resolve_stream_task_id(self: Self) -> None:
        preferred = self._normalize_task_name(
            getattr(self, "stream_task_name", "")
        )
        if not preferred:
            preferred = self._normalize_task_name(
                getattr(self, "stream_task_id", "")
            )
        chosen: Optional[str] = None
        if preferred:
            try:
                candidate = self.resolve_task_name(preferred)
            except KeyError:
                candidate = None
            if candidate is not None and candidate in self.tasks:
                if getattr(self.tasks[candidate], "mode", "") == "temporal":
                    chosen = candidate
        if chosen is None:
            for k, t in self.tasks.items():
                if getattr(t, "mode", "") == "temporal":
                    chosen = k
                    break
        if chosen is None:
            for fallback in ("temporal", "stream"):
                if fallback in self.tasks:
                    chosen = fallback
                    break
        if chosen is None:
            chosen = next(iter(self.tasks.keys()), "")
        self.stream_task_name = str(chosen or "")
        self.stream_task_id = self.stream_task_name
        cfg = getattr(self, "cfg", None) or getattr(
            self, "__enn_instance_config__", None
        )
        if cfg is not None:
            with contextlib.suppress(Exception):
                setattr(cfg, "stream_task_name", self.stream_task_name or "")
            with contextlib.suppress(Exception):
                setattr(cfg, "stream_task_id", self.stream_task_name or None)

    def _normalize_task_name(self: Self, value: object) -> str:
        s = "" if value is None else str(value)
        s = s.replace("\r", "").replace("\n", "").strip()
        if "." in s:
            s = s.replace(".", "_")
        return s

    def _generate_unique_uuid_name(
        self: Self, *args: Any, exclude: Optional[str] = None
    ) -> str:
        exclude = str(exclude) if exclude is not None else None
        while True:
            candidate = uuid.uuid4().hex
            if exclude is not None and candidate == exclude:
                continue
            if candidate not in self.tasks:
                return candidate

    def _ensure_unique_task_name(
        self: Self,
        preferred: object,
        *args: Any,
        exclude: Optional[str] = None,
    ) -> str:
        candidate = self._normalize_task_name(preferred)
        if not candidate:
            return self._generate_unique_uuid_name(exclude=exclude)
        if exclude is not None and candidate == exclude:
            return candidate
        if candidate in self.tasks:
            return self._generate_unique_uuid_name(exclude=exclude)
        return candidate

    def _topology_guard(self: Self) -> contextlib.AbstractContextManager[object]:
        if bool(is_gil_enabled()):
            return contextlib.nullcontext()
        return self._topology_lock

    def _invalidate_exec_plan(self: Self) -> None:
        self._topology_version = int(getattr(self, "_topology_version", 0) or 0) + 1
        self._exec_plan = None

    def _detach_unlocked(self: Self, name: str) -> None:
        key = str(name)
        parent = self._parent.get(key)
        if parent is None:
            with contextlib.suppress(ValueError):
                self._root_children.remove(key)
        else:
            if parent in self.tasks:
                p = self.tasks[parent]
                if isinstance(p, SubFuser):
                    with contextlib.suppress(ValueError):
                        p._children.remove(key)
        self._parent[key] = None

    def _is_ancestor_unlocked(self: Self, maybe_ancestor: str, node: str) -> bool:
        anc = str(maybe_ancestor)
        cur = str(node)
        if anc == cur:
            return True
        seen: set[str] = set()
        while True:
            p = self._parent.get(cur)
            if p is None:
                return False
            if p == anc:
                return True
            if p in seen:
                return True
            seen.add(p)
            cur = p

    def _attach_unlocked(self: Self, name: str, parent: Optional[str]) -> None:
        key = str(name)
        if key not in self.tasks:
            raise KeyError(f"Unknown node '{key}'")
        if parent is not None:
            pkey = str(parent)
            if pkey not in self.tasks:
                raise KeyError(f"Unknown parent node '{pkey}'")
            pmod = self.tasks[pkey]
            if not isinstance(pmod, SubFuser):
                raise TypeError(f"Parent '{pkey}' must be a SubFuser(taskset)")
            if self._is_ancestor_unlocked(key, pkey):
                raise ValueError("Cycle detected: cannot attach an ancestor under its descendant")
        self._detach_unlocked(key)
        if parent is None:
            if key not in self._root_children:
                self._root_children.append(key)
            self._parent[key] = None
            return
        pkey = str(parent)
        pmod = self.tasks[pkey]
        if not isinstance(pmod, SubFuser):
            raise TypeError(f"Parent '{pkey}' must be a SubFuser(taskset)")
        if key not in pmod._children:
            pmod._children.append(key)
        self._parent[key] = pkey

    def _build_exec_plan_unlocked(self: Self) -> dict[str, Any]:
        roots = list(self._root_children) if self._root_children else list(self.tasks.keys())
        roots = [r for r in roots if r in self.tasks]
        order: list[str] = []
        visited: set[str] = set()
        stack: set[str] = set()
        has_tasksets = False
        has_linear = bool(str(getattr(self, "_root_reduction", "mean")) == "linear")

        def _dfs(n: str) -> None:
            nonlocal has_tasksets, has_linear
            if n in visited:
                return
            if n in stack:
                raise RuntimeError("Cycle detected in task topology")
            stack.add(n)
            mod = self.tasks[n] if n in self.tasks else None
            if isinstance(mod, SubFuser):
                has_tasksets = True
                if str(getattr(mod, "reduction", "mean")) == "linear":
                    has_linear = True
                for c in list(getattr(mod, "_children", []) or []):
                    if c in self.tasks:
                        _dfs(str(c))
            visited.add(n)
            stack.remove(n)
            order.append(n)

        for r in roots:
            _dfs(str(r))

        return {
            "roots": roots,
            "order": order,
            "has_tasksets": bool(has_tasksets),
            "has_linear": bool(has_linear),
        }

    def _get_exec_plan(self: Self) -> dict[str, Any]:
        plan = self._exec_plan
        if isinstance(plan, dict):
            return plan
        with self._topology_guard():
            plan = self._exec_plan
            if isinstance(plan, dict):
                return plan
            built = self._build_exec_plan_unlocked()
            self._exec_plan = built
            return built

    @property
    def task_names(self: Self) -> list[str]:
        return list(self.tasks.keys())

    def get_task_name(self: Self, task_spec: str) -> str:
        return self.resolve_task_name(task_spec)

    def get_submodel(self: Self, task_spec: str) -> Optional[nn.Module]:
        key = self.resolve_task_name(task_spec)
        sm = self._user_submodels.get(key)
        return sm if isinstance(sm, nn.Module) else None

    def get(
        self: Self, task_spec: str, default: Optional[nn.Module] = None
    ) -> Optional[nn.Module]:
        if task_spec is None:
            return default
        raw = self._normalize_task_name(task_spec)
        if not raw:
            return default
        lowered = raw.lower()
        if lowered.startswith("name:"):
            raw = self._normalize_task_name(raw.split(":", 1)[1])
        elif lowered.startswith("id:"):
            raw = self._normalize_task_name(raw.split(":", 1)[1])
        if raw in self.tasks:
            return self.tasks[raw]
        mapped = self._legacy_task_id_to_name.get(raw)
        if mapped and mapped in self.tasks:
            return self.tasks[mapped]
        return default

    def resolve_task_name(self: Self, task_spec: str) -> str:
        raw = self._normalize_task_name(task_spec)
        lowered = raw.lower()
        if lowered.startswith("name:"):
            raw = self._normalize_task_name(raw.split(":", 1)[1])
        elif lowered.startswith("id:"):
            raw = self._normalize_task_name(raw.split(":", 1)[1])
        if raw in self.tasks:
            return raw
        mapped = self._legacy_task_id_to_name.get(raw)
        if mapped and mapped in self.tasks:
            return mapped
        raise KeyError(
            f"Unknown task '{task_spec}'. Known tasks: {sorted(self.tasks.keys())}"
        )

    def resolve_task_id(self: Self, task_spec: str) -> str:
        return self.resolve_task_name(task_spec)

    def node_name(self: Self, node_spec: Optional[object] = None) -> str:
        if node_spec is None:
            return ""
        if isinstance(node_spec, str):
            if not node_spec.strip():
                return ""
            return self.resolve_task_name(node_spec)
        if node_spec is self:
            return ""
        if isinstance(node_spec, SubFuser):
            pair = _SUBFUSER_BINDINGS.get(node_spec)
            if pair:
                owner_ref, name = pair
                owner = owner_ref() if callable(owner_ref) else None
                if owner is self:
                    return str(name)
        for k, m in getattr(self, "tasks", {}).items():
            if m is node_spec:
                return str(k)
        raise KeyError(
            f"Unknown node {type(node_spec)}. Known nodes: {sorted(self.tasks.keys())}"
        )

    def get_children(self: Self, node_spec: Optional[object] = None) -> list[nn.Module]:
        try:
            key = self.node_name(node_spec)
        except KeyError:
            return []

        if not key:
            with self._topology_guard():
                names = list(self._root_children)
        else:
            mod = self.tasks[key] if key in self.tasks else None
            if isinstance(mod, SubFuser):
                names = list(getattr(mod, "_children", []) or [])
            else:
                names = []

        out: list[nn.Module] = []
        for nm in names:
            if nm in self.tasks:
                out.append(self.tasks[nm])
        return out

    def get_parent(
        self: Self, node_spec: Optional[object] = None
    ) -> Optional[nn.Module]:
        try:
            key = self.node_name(node_spec)
        except KeyError:
            return None
        if not key:
            return None
        with self._topology_guard():
            p = self._parent.get(key)
        if p is None or (not str(p).strip()) or (p not in self.tasks):
            return None
        return self.tasks[str(p)]

    def get_subtree(
        self: Self,
        node_spec: Optional[object] = None,
        *args: Any,
        by: str = "name",
        include_self: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        del args, kwargs
        if node_spec is None or node_spec is self or (
            isinstance(node_spec, str) and not node_spec.strip()
        ):
            plan = self._get_exec_plan()
            order = [str(x) for x in (plan.get("order") or [])]
            if str(by or "name").strip().lower() in {"topo", "topological", "tree"}:
                return order
            return sorted(set(order))

        try:
            root = self.node_name(node_spec)
        except KeyError:
            return []
        if not root or root not in self.tasks:
            return []

        topo: list[str] = []
        seen: set[str] = set()

        def _post(n: str) -> None:
            if n in seen:
                return
            mod = self.tasks[n] if n in self.tasks else None
            if isinstance(mod, SubFuser):
                for c in list(getattr(mod, "_children", []) or []):
                    cs = str(c)
                    if cs in seen:
                        continue
                    _post(cs)
            seen.add(n)
            topo.append(n)

        _post(str(root))
        if not include_self:
            topo = [x for x in topo if x != str(root)]
        mode = str(by or "name").strip().lower()
        if mode in {"topo", "topological", "tree"}:
            return topo
        return sorted(set(topo))

    def attach(self: Self, node_spec: str, parent: Optional[str] = None) -> str:
        key = self.resolve_task_name(node_spec)
        parent_key: Optional[str] = None
        if parent is not None and str(parent).strip():
            parent_key = self.resolve_task_name(str(parent))
        with self._topology_guard():
            self._attach_unlocked(key, parent_key)
        self._invalidate_exec_plan()
        self._resolve_stream_task_id()
        return key

    def detach(self: Self, node_spec: str) -> str:
        return self.attach(node_spec, parent=None)


    def new(
        self: Self,
        name: Optional[str] = None,
        *args: Any,
        kind: str = "task",
        parent: Optional[str] = None,
        children: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> str:
        del args
        kind_raw = str(kwargs.pop("type", kind) or kind).strip().lower()
        if kind_raw in {"task", "template", "leaf"}:
            return self.add_task(name, parent=parent, **kwargs)
        if kind_raw in {"taskset", "subfuser", "set"}:
            if (
                "reduction" not in kwargs
                or kwargs.get("reduction") is None
                or (not str(kwargs.get("reduction")).strip())
            ):
                inherited = str(getattr(self, "_root_reduction", "mean"))
                if parent is not None and str(parent).strip():
                    with contextlib.suppress(Exception):
                        pkey = self.resolve_task_name(str(parent))
                        pmod = self.tasks[pkey] if pkey in self.tasks else None
                        if isinstance(pmod, SubFuser):
                            inherited = str(
                                getattr(pmod, "reduction", inherited) or inherited
                            )
                kwargs["reduction"] = inherited
            return self.add_taskset(
                name,
                parent=parent,
                children=children,
                **kwargs,
            )
        raise ValueError(
            f"Unknown kind {kind!r}. Expected 'task' or 'taskset'."
        )

    def update(self: Self, node_spec: str, *args: Any, **kwargs: Any) -> str:
        del args
        node = self.get(node_spec)
        if node is None:
            raise KeyError(
                f"Unknown node '{node_spec}'. Known nodes: {sorted(self.tasks.keys())}"
            )
        parent = kwargs.pop("parent", _META_UNSET)
        if isinstance(node, SubFuser):
            if parent is not _META_UNSET:
                kwargs["parent"] = parent
            return self.update_taskset(node_spec, **kwargs)
        if "children" in kwargs or "reduction" in kwargs:
            raise TypeError(
                "update(children=...) or update(reduction=...) is only valid for taskset nodes"
            )
        key = self.update_task(node_spec, **kwargs)
        if parent is not _META_UNSET:
            self.attach(key, parent=None if parent is None else str(parent))
        return key

    def remove(self: Self, node_spec: str, *args: Any, strict: bool = False, **kwargs: Any) -> None:
        del args, kwargs
        node = self.get(node_spec)
        if node is None:
            if strict:
                raise KeyError(
                    f"Unknown node '{node_spec}'. Known nodes: {sorted(self.tasks.keys())}"
                )
            return
        if isinstance(node, SubFuser):
            return self.remove_taskset(node_spec, promote_children=False, strict=strict)
        return self.remove_task(node_spec, strict=strict)

    def list_tasks(self: Self, by: str = "name") -> list[str]:
        return self.get_subtree(by=by)

    def list(self: Self, by: str = "name") -> list[str]:
        return self.get_subtree(by=by)

    def spec(self: Self) -> list[Dict[str, Any]]:
        plan = self._get_exec_plan()
        order = [str(x) for x in (plan.get("order") or [])]
        specs: list[Dict[str, Any]] = []
        for name in order:
            if name not in self.tasks:
                continue
            t = self.tasks[name]
            meta = self._task_meta.get(name, {})
            base: Dict[str, Any] = {
                "name": name,
                "description": str(meta.get("description") or ""),
                "tags": list(meta.get("tags") or []),
                "weight": float(
                    getattr(t, "weight", torch.as_tensor(1.0)).detach().cpu().item()
                    if isinstance(getattr(t, "weight", None), torch.Tensor)
                    else float(getattr(t, "weight", 1.0))
                ),
                "eps": float(
                    getattr(t, "eps", torch.as_tensor(1e-6)).detach().cpu().item()
                    if isinstance(getattr(t, "eps", None), torch.Tensor)
                    else float(getattr(t, "eps", 1e-6))
                ),
                "parent": str(self._parent.get(name) or ""),
            }
            if isinstance(t, SubFuser):
                base.update(
                    {
                        "type": "taskset",
                        "tokens": int(getattr(t, "tokens", self.fused_tokens)),
                        "reduction": str(getattr(t, "reduction", "mean")),
                        "children": list(getattr(t, "_children", []) or []),
                    }
                )
            else:
                base.update(
                    {
                        "type": "task",
                        "mode": str(getattr(t, "mode", "spatial")),
                        "tokens": int(getattr(t, "tokens", 1)),
                        "depth": int(getattr(t, "depth", 1)),
                        "has_submodel": bool(name in self._user_submodels),
                    }
                )
            specs.append(base)
        return specs

    def graph(self: Self) -> Dict[str, Any]:
        plan = self._get_exec_plan()
        roots = [str(x) for x in (plan.get("roots") or [])]

        def _node(n: str) -> Dict[str, Any]:
            if n not in self.tasks:
                return {"type": "missing", "name": n}
            m = self.tasks[n]
            if isinstance(m, SubFuser):
                return {
                    "type": "taskset",
                    "name": n,
                    "reduction": str(getattr(m, "reduction", "mean")),
                    "children": [_node(str(c)) for c in list(getattr(m, "_children", []) or [])],
                }
            return {
                "type": "task",
                "name": n,
                "mode": str(getattr(m, "mode", "")),
                "tokens": int(getattr(m, "tokens", 1)),
                "depth": int(getattr(m, "depth", 1)),
            }

        return {
            "type": "fuser",
            "reduction": str(getattr(self, "_root_reduction", "mean")),
            "roots": [_node(r) for r in roots],
        }

    def task_specs(self: Self) -> list[Dict[str, Any]]:
        specs = self.spec()
        specs.append(
            {
                "type": "__fuser__",
                "reduction": str(getattr(self, "_root_reduction", "mean")),
                "roots": list(self._root_children),
            }
        )
        return specs

    def rebuild_tasks_from_specs(
        self: Self, specs: Sequence[Dict[str, Any]]
    ) -> None:
        self.tasks = nn.ModuleDict()
        self._task_meta = {}
        self._user_submodels = {}
        self._legacy_task_id_to_name = {}
        self._root_children = []
        self._parent = {}
        self._root_reduction = "mean"
        self._exec_plan = None
        if not specs:
            self._init_default_tasks()
            self._resolve_stream_task_id()
            return
        roots_override: Optional[list[str]] = None
        name_map: dict[str, str] = {}
        pending_children: dict[str, list[str]] = {}
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            stype = str(spec.get("type") or "task").strip().lower()
            if stype in {"__fuser__", "fuser", "root"}:
                roots_raw = spec.get("roots")
                if isinstance(roots_raw, (list, tuple)):
                    roots_override = [self._normalize_task_name(x) for x in roots_raw]
                red = str(spec.get("reduction") or "mean")
                with contextlib.suppress(Exception):
                    self._root_reduction = SubFuser._coerce_reduction(red)
                continue

            legacy_ids: list[str] = []
            for k in ("task_id", "legacy_task_id", "id"):
                v = spec.get(k)
                if v is not None and str(v).strip():
                    legacy_ids.append(str(v).strip())
            preferred_name = spec.get("name")
            if preferred_name is None or not str(preferred_name).strip():
                preferred_name = legacy_ids[0] if legacy_ids else None

            orig_norm = self._normalize_task_name(preferred_name)

            if stype in {"taskset", "subfuser"}:
                nm = self._ensure_unique_task_name(preferred_name)
                node = SubFuser(
                    tokens=int(self.fused_tokens),
                    children=None,
                    reduction=str(spec.get("reduction") or "mean"),
                    weight=float(spec.get("weight", 1.0)),
                    eps=float(spec.get("eps", 1e-6)),
                )
                self.tasks[nm] = node
                with contextlib.suppress(Exception):
                    node._bind(self, nm)
                tags_list: list[str] = []
                raw_tags = spec.get("tags")
                if raw_tags is not None:
                    tags_iter = (raw_tags,) if isinstance(raw_tags, str) else raw_tags
                    for t in tags_iter:
                        s = str(t).strip()
                        if s and s not in tags_list:
                            tags_list.append(s)
                self._task_meta[nm] = {
                    "description": str(spec.get("description") or ""),
                    "tags": tags_list,
                }
                self._root_children.append(nm)
                self._parent[nm] = None
                name_map[orig_norm or nm] = nm
                pending_children[nm] = [
                    self._normalize_task_name(x)
                    for x in (spec.get("children") or [])
                    if x is not None and str(x).strip()
                ]
                for lid in legacy_ids:
                    lid_norm = self._normalize_task_name(lid)
                    if lid_norm and lid_norm != nm:
                        self._legacy_task_id_to_name[lid_norm] = nm
                continue

            final_name = self.add_task(
                preferred_name,
                mode=spec.get("mode", "spatial"),
                description=spec.get("description"),
                tags=spec.get("tags"),
                weight=spec.get("weight", 1.0),
                eps=spec.get("eps", 1e-6),
                submodel=None,
                tokens=spec.get("tokens"),
                depth=spec.get("depth"),
                parent=None,
            )
            name_map[orig_norm or final_name] = final_name
            for lid in legacy_ids:
                lid_norm = self._normalize_task_name(lid)
                if lid_norm and lid_norm != final_name:
                    self._legacy_task_id_to_name[lid_norm] = final_name

        with self._topology_guard():
            for ts_name, kids in pending_children.items():
                if ts_name not in self.tasks:
                    continue
                ts = self.tasks[ts_name]
                if not isinstance(ts, SubFuser):
                    continue
                ts._children = []
                for child_raw in kids:
                    c_norm = self._normalize_task_name(child_raw)
                    c_key = name_map.get(c_norm, c_norm)
                    if not c_key or c_key not in self.tasks:
                        continue
                    if c_key == ts_name:
                        continue
                    self._attach_unlocked(c_key, ts_name)
            if roots_override:
                desired: list[str] = []
                for r in roots_override:
                    rk = name_map.get(self._normalize_task_name(r), self._normalize_task_name(r))
                    if rk in self.tasks and (self._parent.get(rk) is None) and (rk not in desired):
                        desired.append(rk)
                for r in list(self._root_children):
                    if r in self.tasks and (self._parent.get(r) is None) and (r not in desired):
                        desired.append(r)
                self._root_children = desired

        self._resolve_stream_task_id()
        self._invalidate_exec_plan()

    def remap_legacy_task_ids_in_state_dict(
        self: Self,
        state_dict: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        if not self._legacy_task_id_to_name:
            return state_dict
        changed = False
        remapped: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            parts = k.split(".")
            for i in range(len(parts) - 1):
                if parts[i] == "tasks":
                    legacy = parts[i + 1]
                    mapped = self._legacy_task_id_to_name.get(legacy)
                    if mapped:
                        parts[i + 1] = mapped
            new_k = ".".join(parts)
            if new_k != k:
                changed = True
            remapped[new_k] = v
        return remapped if changed else state_dict

    def add_task(
        self: Self,
        name: Optional[str] = None,
        *args: Any,
        mode: str = "spatial",
        description: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        weight: float = 1.0,
        eps: float = 1e-6,
        submodel: Optional[nn.Module] = None,
        tokens: Optional[int] = None,
        depth: Optional[int] = None,
        parent: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> str:
        mode = Template._coerce_mode(mode)
        preferred = name
        if preferred is None or not str(preferred).strip():
            preferred = task_id
        nm = self._ensure_unique_task_name(preferred)
        if tokens is None:
            tokens = int(
                self.spatial_tokens
                if mode == "spatial"
                else self.temporal_tokens
            )
        if depth is None:
            cfg = self.cfg
            depth = int(
                getattr(cfg, "spatial_depth", 1)
                if mode == "spatial"
                else getattr(cfg, "temporal_depth", 1)
            )
        tmpl = Template(
            self.in_dim,
            int(tokens),
            self.d_model,
            self.nhead,
            int(depth),
            mode=mode,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            drop_path=self.drop_path,
            norm_type=self.norm_type,
            weight=weight,
            eps=eps,
            ckpt_enabled=getattr(self.cfg, "ckpt_enabled", True),
            ckpt_min_bytes=getattr(
                self.cfg, "ckpt_min_bytes", 64 * 1024 * 1024
            ),
        )
        tags_list: list[str] = []
        if tags is not None:
            tags_iter = (tags,) if isinstance(tags, str) else tags
            for t in tags_iter:
                s = str(t).strip()
                if s and s not in tags_list:
                    tags_list.append(s)
        self.tasks[nm] = tmpl
        self._task_meta[nm] = {
            "description": str(description) if description is not None else "",
            "tags": tags_list,
        }
        if submodel is not None:
            self._user_submodels[nm] = submodel
        parent_key: Optional[str] = None
        if parent is not None and str(parent).strip():
            parent_key = self.resolve_task_name(str(parent))
        with self._topology_guard():
            self._attach_unlocked(nm, parent_key)
        self._invalidate_exec_plan()
        self._resolve_stream_task_id()
        return nm

    def update_task(
        self: Self,
        task_name: str,
        *args: Any,
        mode: object = _META_UNSET,
        name: object = _META_UNSET,
        description: object = _META_UNSET,
        tags: object = _META_UNSET,
        submodel: object = _SUBMODEL_UNSET,
        weight: object = _META_UNSET,
        eps: object = _META_UNSET,
    ) -> str:
        key = self.resolve_task_name(task_name)
        tmpl = self.tasks[key]
        if isinstance(tmpl, SubFuser):
            raise TypeError(
                "update_task() is only valid for Template tasks; use update_taskset() for tasksets"
            )
        meta = self._task_meta.get(key)
        if meta is None:
            meta = {"description": "", "tags": []}
            self._task_meta[key] = meta
        if mode is not _META_UNSET:
            tmpl.set_mode(str(mode))
        if weight is not _META_UNSET:
            tmpl.set_weight(float(weight))
        if eps is not _META_UNSET:
            tmpl.set_eps(float(eps))
        if description is not _META_UNSET:
            meta["description"] = (
                "" if description is None else str(description)
            )
        if tags is not _META_UNSET:
            tags_list: list[str] = []
            if tags is not None:
                tags_iter = (tags,) if isinstance(tags, str) else tags
                for t in tags_iter:
                    s = str(t).strip()
                    if s and s not in tags_list:
                        tags_list.append(s)
            meta["tags"] = tags_list
        if submodel is not _SUBMODEL_UNSET:
            if submodel is None:
                self._user_submodels.pop(key, None)
            else:
                self._user_submodels[key] = submodel
        if name is not _META_UNSET:
            new_key = self._ensure_unique_task_name(name, exclude=key)
            if new_key != key:
                mod = self.tasks[key]
                del self.tasks[key]
                self.tasks[new_key] = mod
                self._task_meta[new_key] = self._task_meta.pop(key, {})
                if key in self._user_submodels:
                    self._user_submodels[new_key] = self._user_submodels.pop(
                        key
                    )
                with self._topology_guard():
                    p = self._parent.pop(key, None)
                    self._parent[new_key] = p
                    self._root_children = [
                        (new_key if x == key else x) for x in self._root_children
                    ]
                    for _n, _m in self.tasks.items():
                        if isinstance(_m, SubFuser):
                            _m._children = [
                                (new_key if c == key else c)
                                for c in list(getattr(_m, "_children", []) or [])
                            ]
                for lid, mapped in list(self._legacy_task_id_to_name.items()):
                    if mapped == key:
                        self._legacy_task_id_to_name[lid] = new_key
                if getattr(self, "stream_task_name", "") == key:
                    self.stream_task_name = new_key
                if getattr(self, "stream_task_id", "") == key:
                    self.stream_task_id = new_key
                key = new_key
                self._invalidate_exec_plan()
        self._resolve_stream_task_id()
        return key

    def remove_task(
        self: Self, task_name: str, *args: Any, strict: bool = False
    ) -> None:
        del args
        if strict and not self.tasks:
            raise KeyError("No tasks are configured")
        try:
            key = self.resolve_task_name(task_name)
        except KeyError:
            if strict:
                raise
            return
        if key in self.tasks and isinstance(self.tasks[key], SubFuser):
            raise TypeError(
                "remove_task() cannot remove a taskset; use remove_taskset()"
            )
        with self._topology_guard():
            with contextlib.suppress(Exception):
                self._detach_unlocked(key)
            self._parent.pop(key, None)
        if key in self.tasks:
            del self.tasks[key]
        self._task_meta.pop(key, None)
        self._user_submodels.pop(key, None)
        for lid, mapped in list(self._legacy_task_id_to_name.items()):
            if mapped == key:
                del self._legacy_task_id_to_name[lid]
        if getattr(self, "stream_task_name", "") == key:
            self.stream_task_name = ""
        if getattr(self, "stream_task_id", "") == key:
            self.stream_task_id = ""
        self._invalidate_exec_plan()
        self._resolve_stream_task_id()

    def add_taskset(
        self: Self,
        name: Optional[str] = None,
        *args: Any,
        children: Optional[Sequence[str]] = None,
        parent: Optional[str] = None,
        reduction: str = "mean",
        description: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        weight: float = 1.0,
        eps: float = 1e-6,
        tokens: Optional[int] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        del args, kwargs
        preferred = name
        if preferred is None or not str(preferred).strip():
            preferred = task_id
        nm = self._ensure_unique_task_name(preferred)
        tok = int(self.fused_tokens if tokens is None else int(tokens))
        if int(tok) != int(self.fused_tokens):
            raise ValueError(
                "Taskset tokens must match Fuser.fused_tokens (fixed latent count)"
            )
        node = SubFuser(
            tokens=int(tok),
            children=None,
            reduction=str(reduction),
            weight=float(weight),
            eps=float(eps),
        )
        tags_list: list[str] = []
        if tags is not None:
            tags_iter = (tags,) if isinstance(tags, str) else tags
            for t in tags_iter:
                s = str(t).strip()
                if s and s not in tags_list:
                    tags_list.append(s)
        self.tasks[nm] = node
        with contextlib.suppress(Exception):
            node._bind(self, nm)
        self._task_meta[nm] = {
            "description": str(description) if description is not None else "",
            "tags": tags_list,
        }
        parent_key: Optional[str] = None
        if parent is not None and str(parent).strip():
            parent_key = self.resolve_task_name(str(parent))
        with self._topology_guard():
            self._attach_unlocked(nm, parent_key)
        if children is not None:
            self.update_taskset(nm, children=children)
        self._invalidate_exec_plan()
        self._resolve_stream_task_id()
        return nm

    def update_taskset(
        self: Self,
        taskset_name: str,
        *args: Any,
        reduction: object = _META_UNSET,
        children: object = _META_UNSET,
        parent: object = _META_UNSET,
        name: object = _META_UNSET,
        description: object = _META_UNSET,
        tags: object = _META_UNSET,
        weight: object = _META_UNSET,
        eps: object = _META_UNSET,
        **kwargs: Any,
    ) -> str:
        del args, kwargs
        key = self.resolve_task_name(taskset_name)
        node = self.tasks[key]
        if not isinstance(node, SubFuser):
            raise TypeError("update_taskset() requires a taskset(SubFuser) node")
        meta = self._task_meta.get(key)
        if meta is None:
            meta = {"description": "", "tags": []}
            self._task_meta[key] = meta

        if name is not _META_UNSET:
            new_key = self._ensure_unique_task_name(name, exclude=key)
            if new_key != key:
                mod = self.tasks[key]
                del self.tasks[key]
                self.tasks[new_key] = mod
                self._task_meta[new_key] = self._task_meta.pop(key, {})
                with self._topology_guard():
                    p = self._parent.pop(key, None)
                    self._parent[new_key] = p
                    self._root_children = [
                        (new_key if x == key else x) for x in self._root_children
                    ]
                    for _n, _m in self.tasks.items():
                        if isinstance(_m, SubFuser):
                            _m._children = [
                                (new_key if c == key else c)
                                for c in list(getattr(_m, "_children", []) or [])
                            ]
                    for ch, par in list(self._parent.items()):
                        if par == key:
                            self._parent[ch] = new_key
                for lid, mapped in list(self._legacy_task_id_to_name.items()):
                    if mapped == key:
                        self._legacy_task_id_to_name[lid] = new_key
                if getattr(self, "stream_task_name", "") == key:
                    self.stream_task_name = new_key
                if getattr(self, "stream_task_id", "") == key:
                    self.stream_task_id = new_key
                key = new_key
                node = self.tasks[key]
                with contextlib.suppress(Exception):
                    if isinstance(node, SubFuser):
                        node._bind(self, key)
                self._invalidate_exec_plan()

        if reduction is not _META_UNSET:
            node.set_reduction(str(reduction))
        if weight is not _META_UNSET:
            node.set_weight(float(weight))
        if eps is not _META_UNSET:
            node.set_eps(float(eps))
        if description is not _META_UNSET:
            meta["description"] = "" if description is None else str(description)
        if tags is not _META_UNSET:
            tags_list: list[str] = []
            if tags is not None:
                tags_iter = (tags,) if isinstance(tags, str) else tags
                for t in tags_iter:
                    s = str(t).strip()
                    if s and s not in tags_list:
                        tags_list.append(s)
            meta["tags"] = tags_list

        if parent is not _META_UNSET:
            parent_key: Optional[str] = None
            if parent is not None and str(parent).strip():
                parent_key = self.resolve_task_name(str(parent))
            with self._topology_guard():
                self._attach_unlocked(key, parent_key)
            self._invalidate_exec_plan()

        if children is not _META_UNSET:
            new_children: list[str] = []
            if children is not None:
                for c in (children if isinstance(children, (list, tuple)) else [children]):
                    c_name = self.resolve_task_name(str(c))
                    if c_name == key:
                        continue
                    if c_name not in new_children:
                        new_children.append(c_name)
            with self._topology_guard():
                old_children = list(getattr(node, "_children", []) or [])
                for c in old_children:
                    if c not in new_children and c in self.tasks:
                        self._attach_unlocked(c, None)
                node._children = []
                for c in new_children:
                    if c in self.tasks:
                        self._attach_unlocked(c, key)
            self._invalidate_exec_plan()

        self._resolve_stream_task_id()
        return key

    def remove_taskset(
        self: Self,
        taskset_name: str,
        *args: Any,
        promote_children: bool = False,
        strict: bool = False,
        **kwargs: Any,
    ) -> None:
        del args, kwargs
        if strict and not self.tasks:
            raise KeyError("No tasks are configured")
        try:
            key = self.resolve_task_name(taskset_name)
        except KeyError:
            if strict:
                raise
            return
        if key not in self.tasks:
            return
        node = self.tasks[key]
        if not isinstance(node, SubFuser):
            raise TypeError("remove_taskset() requires a taskset(SubFuser) node")

        def _collect_subtree(nm: str) -> list[str]:
            out: list[str] = []
            def _dfs(n: str) -> None:
                mod = self.tasks[n] if n in self.tasks else None
                if isinstance(mod, SubFuser):
                    for c in list(getattr(mod, "_children", []) or []):
                        if c in self.tasks:
                            _dfs(str(c))
                out.append(n)
            _dfs(str(nm))
            return out

        with self._topology_guard():
            parent = self._parent.get(key)
            kids = list(getattr(node, "_children", []) or [])
            if promote_children:
                if parent is None:
                    try:
                        idx = self._root_children.index(key)
                    except Exception:
                        idx = len(self._root_children)
                    with contextlib.suppress(Exception):
                        self._root_children.remove(key)
                    insert_at = idx
                    for c in kids:
                        if c in self.tasks:
                            self._attach_unlocked(c, None)
                            with contextlib.suppress(Exception):
                                self._root_children.remove(c)
                            self._root_children.insert(insert_at, c)
                            insert_at += 1
                else:
                    pmod = self.tasks[parent] if parent in self.tasks else None
                    if isinstance(pmod, SubFuser):
                        try:
                            idx = pmod._children.index(key)
                        except Exception:
                            idx = len(pmod._children)
                        with contextlib.suppress(Exception):
                            pmod._children.remove(key)
                        insert_at = idx
                        for c in kids:
                            if c in self.tasks:
                                self._attach_unlocked(c, parent)
                                with contextlib.suppress(Exception):
                                    pmod._children.remove(c)
                                pmod._children.insert(insert_at, c)
                                insert_at += 1
                with contextlib.suppress(Exception):
                    self._detach_unlocked(key)
                self._parent.pop(key, None)
                del self.tasks[key]
                self._task_meta.pop(key, None)
            else:
                subtree = _collect_subtree(key)
                for n in subtree:
                    with contextlib.suppress(Exception):
                        self._detach_unlocked(n)
                    self._parent.pop(n, None)
                    if n in self.tasks:
                        self._user_submodels.pop(n, None)
                        del self.tasks[n]
                    self._task_meta.pop(n, None)
                    for lid, mapped in list(self._legacy_task_id_to_name.items()):
                        if mapped == n:
                            del self._legacy_task_id_to_name[lid]

        self._invalidate_exec_plan()
        self._resolve_stream_task_id()

    def _select_tasks(self: Self) -> list[str]:
        plan = self._get_exec_plan()
        roots = plan.get("roots")
        if isinstance(roots, (list, tuple)) and roots:
            return [str(x) for x in roots if str(x) in self.tasks]
        return list(self.tasks.keys())

    def _build_attn_bias(
        self: Self,
        names: Sequence[str],
        token_sets: Sequence[torch.Tensor],
        *args: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if len(token_sets) <= 1:
            return None
        exporting = bool(is_export_or_trace())
        w_list: list[torch.Tensor] = []
        e_list: list[torch.Tensor] = []
        counts: list[int] = []
        for n, t in zip(names, token_sets):
            tmpl = self.tasks[n]
            if not isinstance(tmpl, nn.Module):
                raise TypeError(f"Task '{n}' is not an nn.Module")
            w = getattr(tmpl, "weight", None)
            e = getattr(tmpl, "eps", None)
            if not isinstance(w, torch.Tensor):
                w = torch.as_tensor(1.0, dtype=torch.float32, device=device)
            if not isinstance(e, torch.Tensor):
                e = torch.as_tensor(1e-6, dtype=torch.float32, device=device)
            w_list.append(w.to(device=device, dtype=torch.float32).reshape(()))
            e_list.append(e.to(device=device, dtype=torch.float32).reshape(()))
            if exporting:
                tok = getattr(tmpl, "tokens", None)
                if tok is not None:
                    cnt = int(tok)
                else:
                    cnt = int(t.size(1))
            else:
                cnt = int(t.size(1))
            counts.append(cnt)
        if any(c <= 0 for c in counts):
            return None
        w_t = torch.stack(w_list, dim=0)
        e_t = torch.stack(e_list, dim=0)
        n_t = torch.tensor(counts, dtype=torch.float32, device=device)
        per_tok = torch.maximum(w_t, e_t) / torch.clamp(n_t, min=1.0)
        logw = torch.log(per_tok.clamp_min(1e-12))
        if exporting:
            parts: list[torch.Tensor] = []
            for i, cnt in enumerate(counts):
                c = int(cnt)
                parts.append(logw[i].expand((c,)))
            bias_vec = torch.cat(parts, dim=0).to(dtype=dtype)
        else:
            rep = torch.tensor(counts, dtype=torch.long, device=device)
            bias_vec = torch.repeat_interleave(logw, rep, dim=0).to(
                dtype=dtype
            )
        return bias_vec.view(1, 1, 1, -1)

    @staticmethod
    def _tensor_has_nonfinite(t: torch.Tensor) -> bool:
        if (not isinstance(t, torch.Tensor)) or (not (t.is_floating_point() or t.is_complex())):
            return False
        if t.numel() == 0:
            return False
        with torch.no_grad():
            try:
                a = t.detach().abs().amax()
                return not bool(torch.isfinite(a).item())
            except Exception:
                return not bool(torch.isfinite(t).all().item())

    def _perceiver_master_dtype(self) -> torch.dtype:
        p0 = next(
            (p for p in self.perceiver.parameters() if torch.is_tensor(p)), None
        )
        if torch.is_tensor(p0):
            dev = p0.device
        else:
            p0 = next((p for p in self.parameters() if torch.is_tensor(p)), None)
            dev = p0.device if torch.is_tensor(p0) else get_device()

        key = (
            str(getattr(dev, "type", "cpu")),
            int(getattr(dev, "index", -1) or -1),
        )
        cache = getattr(self, "_perceiver_master_dtype_cache", None)
        if isinstance(cache, dict):
            cached = cache.get(key, None)
            if isinstance(cached, torch.dtype):
                return cached

        meta = StatelessAutocast.coerce_metadata(dev)
        try:
            md = PrecisionPolicy.from_metadata(
                device=dev, metadata=meta, logger=_LOGGER
            ).master_float
        except Exception:
            md = torch.float32

        if not isinstance(cache, dict):
            cache = {}
            setattr(self, "_perceiver_master_dtype_cache", cache)
        cache[key] = md
        return md

    def _call_perceiver(
        self,
        all_tokens: torch.Tensor,
        *,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if bool(is_export_or_trace()):
            return cast(torch.Tensor, self.perceiver(all_tokens, attn_bias=attn_bias))
        if (
            hasattr(torch, "compiler")
            and hasattr(torch.compiler, "is_compiling")
            and bool(torch.compiler.is_compiling())
        ):
            return cast(torch.Tensor, self.perceiver(all_tokens, attn_bias=attn_bias))
        if not bool(getattr(self, "_perceiver_stateful_autocast_enabled", False)):
            return cast(
                torch.Tensor, self.perceiver(all_tokens, attn_bias=attn_bias)
            )
        master = self._perceiver_master_dtype()
        return cast(
            torch.Tensor,
            self._perceiver_autocast.call(
                self.perceiver,
                all_tokens,
                attn_bias=attn_bias,
                device=all_tokens.device,
                master_dtype=master,
                retry_master_dtype=master,
                validate=self._tensor_has_nonfinite,
            ),
        )

    def _call_linear_fuse(
        self,
        fuse_fn: Callable[[Sequence[str], Sequence[torch.Tensor]], torch.Tensor],
        child_names: Sequence[str],
        child_tokens: Sequence[torch.Tensor],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if bool(is_export_or_trace()):
            return fuse_fn(child_names, child_tokens)
        if (
            hasattr(torch, "compiler")
            and hasattr(torch.compiler, "is_compiling")
            and bool(torch.compiler.is_compiling())
        ):
            return fuse_fn(child_names, child_tokens)
        if not bool(getattr(self, "_perceiver_stateful_autocast_enabled", False)):
            return fuse_fn(child_names, child_tokens)
        master = self._perceiver_master_dtype()
        return cast(
            torch.Tensor,
            self._perceiver_autocast.call(
                fuse_fn,
                list(child_names),
                list(child_tokens),
                device=device,
                master_dtype=master,
                retry_master_dtype=master,
                validate=self._tensor_has_nonfinite,
            ),
        )

    def _forward_topology(
        self: Self,
        x: torch.Tensor,
        *args: Any,
        temporal_state: object = None,
        return_temporal_state: bool = False,
        causal_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        del args, kwargs
        plan = self._get_exec_plan()
        roots = [str(r) for r in (plan.get("roots") or []) if str(r) in self.tasks]
        order = [str(n) for n in (plan.get("order") or []) if str(n) in self.tasks]

        state_is_map = isinstance(temporal_state, Mapping)
        next_state_map: dict[str, torch.Tensor] = {}
        next_state: Optional[torch.Tensor] = None
        if (
            (not state_is_map)
            and (temporal_state is not None)
            and (not isinstance(temporal_state, torch.Tensor))
        ):
            raise TypeError(
                "temporal_state must be None, a torch.Tensor, or a Mapping[str, torch.Tensor]"
            )
        if (not state_is_map) and isinstance(temporal_state, torch.Tensor):
            sid = str(getattr(self, "stream_task_id", "") or "").strip()
            if not sid:
                raise ValueError(
                    "temporal_state was provided but stream_task_id is not set (no temporal task selected)"
                )

        token_cache: dict[str, torch.Tensor] = {}

        def _pair_has_nonfinite(t: torch.Tensor) -> bool:
            if (not isinstance(t, torch.Tensor)) or (not t.is_floating_point()):
                return False
            if t.numel() == 0:
                return False
            with torch.no_grad():
                try:
                    a = t.detach().abs().amax()
                    return not bool(torch.isfinite(a).item())
                except Exception:
                    return not bool(torch.isfinite(t).all().item())

        def _fallback_fused_from_all(all_tokens: torch.Tensor) -> torch.Tensor:
            tgt = int(self.fused_tokens)
            n = int(all_tokens.shape[1])
            if n == tgt:
                return all_tokens
            if n > tgt:
                return all_tokens[:, :tgt, :]
            pad = tgt - n
            if pad <= 0:
                return all_tokens
            z = all_tokens.new_zeros((all_tokens.shape[0], pad, all_tokens.shape[2]))
            return torch.cat((all_tokens, z), dim=1)

        def _linear_fuse(
            child_names: Sequence[str],
            child_tokens: Sequence[torch.Tensor],
        ) -> torch.Tensor:
            if not child_tokens:
                raise RuntimeError("linear reduction requires at least one child")
            tok0 = child_tokens[0]
            B = int(tok0.shape[0])
            latents = self.perceiver.latents.unsqueeze(0).expand(B, -1, -1)
            if latents.dtype != tok0.dtype:
                latents = latents.to(dtype=tok0.dtype)
            if latents.device != tok0.device:
                latents = latents.to(device=tok0.device)
            j = 0
            for i in range(int(getattr(self.perceiver, "depth", 1))):
                base = latents
                delta_sum = torch.zeros_like(base)
                for nm, toks in zip(child_names, child_tokens):
                    mod = self.tasks[nm] if nm in self.tasks else None
                    w = getattr(mod, "weight", None)
                    e = getattr(mod, "eps", None)
                    if not isinstance(w, torch.Tensor):
                        w = torch.as_tensor(1.0, dtype=tok0.dtype, device=tok0.device)
                    if not isinstance(e, torch.Tensor):
                        e = torch.as_tensor(1e-6, dtype=tok0.dtype, device=tok0.device)
                    coef = w.to(device=tok0.device, dtype=base.dtype).reshape(())
                    eps_t = e.to(device=tok0.device, dtype=base.dtype).reshape(())
                    coef = torch.where(coef == 0, eps_t, coef)
                    out = self.perceiver.cross[i](base, toks, attn_bias=None)
                    delta_sum = delta_sum + coef * (out - base)
                latents = base + delta_sum
                for _ in range(int(getattr(self.perceiver, "self_attn_layers", 0))):
                    if j < len(self.perceiver.self_blocks):
                        latents = self.perceiver.self_blocks[j](latents)
                    j += 1
            return self.perceiver.norm(latents)

        for name in order:
            mod = self.tasks[name]
            if isinstance(mod, SubFuser):
                child_names = [str(c) for c in list(getattr(mod, "_children", []) or [])]
                child_names = [c for c in child_names if c in token_cache]
                if not child_names:
                    raise RuntimeError(f"Taskset '{name}' has no executed children")
                child_tokens = [token_cache[c] for c in child_names]
                red = str(getattr(mod, "reduction", "mean"))
                if red == "mean":
                    if len(child_tokens) == 1:
                        all_tokens = child_tokens[0]
                    else:
                        all_tokens = torch.cat(child_tokens, dim=1)
                    attn_bias = self._build_attn_bias(
                        child_names,
                        child_tokens,
                        device=all_tokens.device,
                        dtype=all_tokens.dtype,
                    )
                    fused = self._call_perceiver(all_tokens, attn_bias=attn_bias)
                    if env_bool("ENN_FUSER_NONFINITE_FALLBACK", default=True) and _pair_has_nonfinite(fused):
                        fused = _fallback_fused_from_all(all_tokens)
                    token_cache[name] = fused
                elif red == "linear":
                    if env_bool("ENN_FUSER_NONFINITE_FALLBACK", default=True):
                        all_tokens = (
                            child_tokens[0]
                            if len(child_tokens) == 1
                            else torch.cat(child_tokens, dim=1)
                        )
                    else:
                        all_tokens = None
                    fused = self._call_linear_fuse(_linear_fuse, child_names, child_tokens, device=child_tokens[0].device)
                    if all_tokens is not None and _pair_has_nonfinite(fused):
                        fused = _fallback_fused_from_all(all_tokens)
                    token_cache[name] = fused
                else:
                    raise ValueError(f"Unknown taskset reduction '{red}'")
                continue

            out_tokens: Any
            if (
                isinstance(mod, Template)
                and str(getattr(mod, "mode", "")) == "temporal"
            ):
                st_in: Optional[torch.Tensor] = None
                if state_is_map:
                    st_raw = cast(Mapping[str, object], temporal_state).get(
                        str(name), None
                    )
                    if st_raw is not None and not isinstance(st_raw, torch.Tensor):
                        raise TypeError(
                            f"temporal_state[{name!r}] must be a torch.Tensor or None; got {type(st_raw)}"
                        )
                    st_in = cast(Optional[torch.Tensor], st_raw)
                else:
                    if str(name) == str(getattr(self, "stream_task_id", "")):
                        st_in = cast(Optional[torch.Tensor], temporal_state)
                if return_temporal_state:
                    out_tokens, st_out = mod(
                        x,
                        state=st_in,
                        return_state=True,
                        causal_mask=causal_mask,
                    )
                    if isinstance(st_out, torch.Tensor):
                        if state_is_map:
                            next_state_map[str(name)] = st_out
                        elif str(name) == str(getattr(self, "stream_task_id", "")):
                            next_state = st_out
                else:
                    out_tokens = mod(
                        x,
                        state=st_in,
                        return_state=False,
                        causal_mask=causal_mask,
                    )
            else:
                if isinstance(mod, Template):
                    out_tokens = mod(x, causal_mask=causal_mask)
                else:
                    out_tokens = mod(x)

            if not isinstance(out_tokens, torch.Tensor):
                raise TypeError(
                    f"Task '{name}' must return a torch.Tensor tokens (B,N,D); got {type(out_tokens)}"
                )
            sm = self._user_submodels.get(str(name))
            if sm is not None:
                out_tokens = sm(out_tokens)
                if not isinstance(out_tokens, torch.Tensor):
                    raise TypeError(
                        f"BYOM for task '{name}' must return torch.Tensor; got {type(out_tokens)}"
                    )
            if out_tokens.dim() != 3 or out_tokens.size(-1) != self.d_model:
                raise ValueError(
                    f"Task '{name}' must return tokens shaped (B,N,{self.d_model}); got {tuple(out_tokens.shape)}"
                )
            token_cache[name] = out_tokens

        graph_break()
        if not roots:
            raise RuntimeError("No root nodes are configured")
        token_sets = [token_cache[r] for r in roots if r in token_cache]
        if not token_sets:
            raise RuntimeError("No root nodes produced tokens")

        root_reduction = str(getattr(self, "_root_reduction", "mean"))
        if root_reduction == "mean":
            all_tokens = token_sets[0] if len(token_sets) == 1 else torch.cat(token_sets, dim=1)
            attn_bias = self._build_attn_bias(
                roots,
                token_sets,
                device=all_tokens.device,
                dtype=all_tokens.dtype,
            )
            fused_tokens = self._call_perceiver(all_tokens, attn_bias=attn_bias)
            if env_bool("ENN_FUSER_NONFINITE_FALLBACK", default=True) and _pair_has_nonfinite(fused_tokens):
                fused_tokens = _fallback_fused_from_all(all_tokens)
            graph_break()
            context = self.decode(fused_tokens, apply_norm=True)
        elif root_reduction == "linear":
            all_tokens = token_sets[0] if len(token_sets) == 1 else torch.cat(token_sets, dim=1)
            fused_tokens = self._call_linear_fuse(_linear_fuse, roots, token_sets, device=token_sets[0].device)
            if env_bool("ENN_FUSER_NONFINITE_FALLBACK", default=True) and _pair_has_nonfinite(fused_tokens):
                fused_tokens = _fallback_fused_from_all(all_tokens)
            graph_break()
            context = self.decode(fused_tokens, apply_norm=True)
        else:
            raise ValueError(f"Unknown root reduction '{root_reduction}'")

        if return_temporal_state:
            if state_is_map:
                return fused_tokens, context, next_state_map
            return fused_tokens, context, next_state
        return fused_tokens, context

    def forward(
        self: Self,
        x: torch.Tensor,
        *args: Any,
        temporal_state: object = None,
        return_temporal_state: bool = False,
        causal_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        del args, kwargs
        plan = self._get_exec_plan()
        if bool(plan.get("has_tasksets")) or bool(plan.get("has_linear")):
            return self._forward_topology(
                x,
                temporal_state=temporal_state,
                return_temporal_state=return_temporal_state,
                causal_mask=causal_mask,
            )
        names = self._select_tasks()
        token_sets: list[torch.Tensor] = []
        state_is_map = isinstance(temporal_state, Mapping)
        next_state_map: dict[str, torch.Tensor] = {}
        next_state: Optional[torch.Tensor] = None
        if (
            (not state_is_map)
            and (temporal_state is not None)
            and (not isinstance(temporal_state, torch.Tensor))
        ):
            raise TypeError(
                "temporal_state must be None, a torch.Tensor, or a Mapping[str, torch.Tensor]"
            )
        if (not state_is_map) and isinstance(temporal_state, torch.Tensor):
            sid = str(getattr(self, "stream_task_id", "") or "").strip()
            if not sid:
                raise ValueError(
                    "temporal_state was provided but stream_task_id is not set (no temporal task selected)"
                )
        for name in names:
            tmpl = self.tasks[name]
            out_tokens: Any
            if (
                isinstance(tmpl, Template)
                and str(getattr(tmpl, "mode", "")) == "temporal"
            ):
                st_in: Optional[torch.Tensor] = None
                if state_is_map:
                    st_raw = cast(Mapping[str, object], temporal_state).get(
                        str(name), None
                    )
                    if st_raw is not None and not isinstance(
                        st_raw, torch.Tensor
                    ):
                        raise TypeError(
                            f"temporal_state[{name!r}] must be a torch.Tensor or None; got {type(st_raw)}"
                        )
                    st_in = cast(Optional[torch.Tensor], st_raw)
                else:
                    if str(name) == str(getattr(self, "stream_task_id", "")):
                        st_in = cast(Optional[torch.Tensor], temporal_state)
                if return_temporal_state:
                    out_tokens, st_out = tmpl(
                        x,
                        state=st_in,
                        return_state=True,
                        causal_mask=causal_mask,
                    )
                    if isinstance(st_out, torch.Tensor):
                        if state_is_map:
                            next_state_map[str(name)] = st_out
                        elif str(name) == str(
                            getattr(self, "stream_task_id", "")
                        ):
                            next_state = st_out
                else:
                    out_tokens = tmpl(
                        x,
                        state=st_in,
                        return_state=False,
                        causal_mask=causal_mask,
                    )
            else:
                if isinstance(tmpl, Template):
                    out_tokens = tmpl(x, causal_mask=causal_mask)
                else:
                    out_tokens = tmpl(x)
            if not isinstance(out_tokens, torch.Tensor):
                raise TypeError(
                    f"Task '{name}' must return a torch.Tensor tokens (B,N,D); got {type(out_tokens)}"
                )
            sm = self._user_submodels.get(str(name))
            if sm is not None:
                out_tokens = sm(out_tokens)
                if not isinstance(out_tokens, torch.Tensor):
                    raise TypeError(
                        f"BYOM for task '{name}' must return torch.Tensor; got {type(out_tokens)}"
                    )
            if out_tokens.dim() != 3 or out_tokens.size(-1) != self.d_model:
                raise ValueError(
                    f"Task '{name}' must return tokens shaped (B,N,{self.d_model}); got {tuple(out_tokens.shape)}"
                )
            token_sets.append(out_tokens)
        graph_break()
        if len(token_sets) < 1:
            raise RuntimeError("No tasks produced tokens")
        if len(token_sets) == 1:
            all_tokens = token_sets[0]
        else:
            all_tokens = torch.cat(token_sets, dim=1)
        graph_break()
        attn_bias = self._build_attn_bias(
            names,
            token_sets,
            device=all_tokens.device,
            dtype=all_tokens.dtype,
        )
        if (not torch.is_grad_enabled()) and (getattr(all_tokens.device, "type", None) == "cuda"):
            if env_bool("ENN_PRED_COLLAPSE_STAGE_DIAG", default=False):
                with contextlib.suppress(Exception):
                    def _pair_stats(t: torch.Tensor) -> dict[str, object]:
                        tt = t.detach()
                        b = int(tt.shape[0]) if tt.ndim > 0 else 1
                        if b >= 2:
                            d = (tt[0] - tt[1]).abs()
                            diff = float(d.max().item())
                        else:
                            diff = 0.0
                        sl = tt[: min(2, b)]
                        finite = bool(torch.isfinite(sl).all().item()) if sl.is_floating_point() else True
                        nonfinite = int((~torch.isfinite(sl)).sum().item()) if sl.is_floating_point() else 0
                        absmax = float(sl.abs().max().item()) if sl.numel() else 0.0
                        return {
                            "shape": [int(x) for x in tt.shape],
                            "dtype": str(tt.dtype),
                            "device": str(tt.device),
                            "pair_diff_max": diff,
                            "pair_absmax": absmax,
                            "pair_finite": finite,
                            "pair_nonfinite": nonfinite,
                        }

                    self._enn_last_fuser_diag = {
                        "all_tokens": _pair_stats(all_tokens),
                        "attn_bias": (_pair_stats(attn_bias) if isinstance(attn_bias, torch.Tensor) else None),
                        "task_count": int(len(token_sets)),
                    }
        infer_cuda = bool(
            (not torch.is_grad_enabled())
            and (not self.training)
            and (not is_export_or_trace())
            and (getattr(all_tokens.device, "type", None) == "cuda")
        )
        disable_pred_cg = bool(
            infer_cuda
            and env_bool("ENN_PRED_DISABLE_CUDAGRAPHS", default=True)
        )
        compile_cg_enabled = bool(getattr(self, "_compile_cudagraphs", False))
        if (not compile_cg_enabled) and (not is_export_or_trace()):
            with contextlib.suppress(Exception):
                compile_cg_enabled = bool(
                    getattr(get_runtime_cfg(), "compile_cudagraphs", False)
                )
        cg_ok = bool(compile_cg_enabled and (not disable_pred_cg))
        if bool(getattr(self, "_perceiver_stateful_autocast_enabled", False)):
            cg_ok = False
        fallback_enabled = bool(
            infer_cuda and env_bool("ENN_FUSER_NONFINITE_FALLBACK", default=True)
        )
        runtime_skip_perceiver = False

        def _fallback_fused_from_all() -> torch.Tensor:
            try:
                tgt = int(self.fused_tokens)
            except Exception:
                tgt = int(getattr(self, "fused_tokens", all_tokens.shape[1]))
            if tgt <= 0:
                tgt = int(all_tokens.shape[1])
            n = int(all_tokens.shape[1])
            if n == tgt:
                return all_tokens
            if n > tgt:
                return all_tokens[:, :tgt, :]
            pad = tgt - n
            if pad <= 0:
                return all_tokens
            z = all_tokens.new_zeros((all_tokens.shape[0], pad, all_tokens.shape[2]))
            return torch.cat((all_tokens, z), dim=1)

        def _pair_has_nonfinite(t: torch.Tensor) -> bool:
            if (not isinstance(t, torch.Tensor)) or (not t.is_floating_point()):
                return False
            if t.numel() == 0:
                return False
            with torch.no_grad():
                try:
                    a = t.detach().abs().amax()
                    return not bool(torch.isfinite(a).item())
                except Exception:
                    return not bool(torch.isfinite(t).all().item())

        def _maybe_dump_fuser_nonfinite(
            reason: str,
            *,
            all_tokens: torch.Tensor,
            fused_tokens: object = None,
            context: object = None,
            attn_bias: object = None,
        ) -> None:
            dump_dir = str(env_str("ENN_NONFINITE_DUMP_DIR") or "").strip()
            if not dump_dir:
                return
            dumped = int(getattr(self, "_enn_fuser_nonfinite_dumped", 0) or 0)
            limit = int(env_int("ENN_FUSER_NONFINITE_DUMP_LIMIT", default=2) or 2)
            if limit >= 0 and dumped >= limit:
                return
            with contextlib.suppress(Exception):
                import os as _os

                all_ranks = env_bool("ENN_NONFINITE_DUMP_ALL_RANKS", default=False)
                if (not all_ranks) and int(_os.environ.get("LOCAL_RANK", "0") or 0) != 0:
                    return

                _os.makedirs(dump_dir, exist_ok=True)
                rank = int(_os.environ.get("RANK", "0") or 0)
                safe_reason = "".join(
                    (c if (c.isalnum() or c in "._-") else "_") for c in str(reason)
                )
                fname = f"fuser_nonfinite.{safe_reason}.rank{rank}.{uuid.uuid4().hex[:8]}.pt"
                out_path = _os.path.join(dump_dir, fname)

                def _stats(t: object) -> object:
                    if not torch.is_tensor(t):
                        return None
                    x = t.detach()
                    out: dict[str, object] = {
                        "shape": [int(v) for v in tuple(x.shape)],
                        "dtype": str(x.dtype),
                        "device": str(x.device),
                        "numel": int(x.numel()),
                    }
                    if x.is_floating_point() and x.numel() > 0:
                        with torch.no_grad():
                            flat = x.reshape(-1)
                            n = int(min(4096, int(flat.numel())))
                            if n > 0:
                                samp = flat[:n]
                                out["absmax"] = float(samp.abs().max().item())
                                out["nonfinite"] = int((~torch.isfinite(samp)).sum().item())
                    return out

                def _sample(t: object) -> object:
                    if not torch.is_tensor(t):
                        return None
                    x = t.detach()
                    if x.numel() > 0:
                        with contextlib.suppress(Exception):
                            if x.dim() >= 1:
                                x = x[: min(2, int(x.shape[0]))]
                            if x.dim() >= 2:
                                x = x[:, : min(128, int(x.shape[1]))]
                            if x.dim() >= 3:
                                x = x[:, :, : min(128, int(x.shape[2]))]
                        with contextlib.suppress(Exception):
                            x = x.contiguous()
                    return x.to("cpu")

                bad_params: list[str] = []
                with torch.no_grad():
                    for pn, pv in self.perceiver.named_parameters(recurse=True):
                        if pv is None or (not isinstance(pv, torch.Tensor)):
                            continue
                        if (not pv.is_floating_point()) or pv.numel() <= 0:
                            continue
                        with contextlib.suppress(Exception):
                            if not bool(torch.isfinite(pv).all().item()):
                                bad_params.append(str(pn))
                                if len(bad_params) >= 256:
                                    break

                call_idx = int(getattr(self, "_enn_fuser_nonfinite_call_idx", 0) or 0) + 1
                setattr(self, "_enn_fuser_nonfinite_call_idx", int(call_idx))

                payload: dict[str, object] = {
                    "reason": str(reason),
                    "call_idx": int(call_idx),
                    "infer_cuda": bool(infer_cuda),
                    "cg_ok": bool(cg_ok),
                    "fallback_enabled": bool(fallback_enabled),
                    "all_tokens_stats": _stats(all_tokens),
                    "fused_tokens_stats": _stats(fused_tokens),
                    "context_stats": _stats(context),
                    "attn_bias_stats": _stats(attn_bias),
                    "all_tokens_sample": _sample(all_tokens),
                    "fused_tokens_sample": _sample(fused_tokens),
                    "context_sample": _sample(context),
                    "attn_bias_sample": _sample(attn_bias),
                    "perceiver_bad_params": bad_params,
                }
                torch.save(payload, out_path)
                setattr(self, "_enn_fuser_nonfinite_dumped", dumped + 1)
                _LOGGER.error("[ENN] Fuser: dumped non-finite snapshot to: %s", str(out_path))

        if runtime_skip_perceiver:
            fused_tokens = _fallback_fused_from_all()
            graph_break()
            context = self.decode(fused_tokens, apply_norm=True)
        else:
            dump_dir = str(env_str("ENN_NONFINITE_DUMP_DIR") or "").strip()
            strict_params = env_bool(
                "ENN_FUSER_STRICT_NONFINITE_PARAMS",
                default=env_bool("ENN_SANITIZE_NAN_STRICT", default=False),
            )
            check_every = int(env_int("ENN_FUSER_PARAM_CHECK_EVERY", 1) or 1)
            if (strict_params or dump_dir) and check_every > 0:
                call_i = int(getattr(self, "_enn_fuser_call_idx", 0) or 0) + 1
                setattr(self, "_enn_fuser_call_idx", int(call_i))
                if (
                    (call_i % int(check_every)) == 0
                    and (not getattr(self, "_enn_fuser_param_bad", False))
                ):
                    bad_name = None
                    with torch.no_grad():
                        for pn, pv in self.perceiver.named_parameters(recurse=True):
                            if (
                                not isinstance(pv, torch.Tensor)
                                or (not pv.is_floating_point())
                                or pv.numel() <= 0
                            ):
                                continue
                            try:
                                a = pv.detach().abs().amax()
                                ok = bool(torch.isfinite(a).item())
                            except Exception:
                                ok = bool(torch.isfinite(pv).all().item())
                            if not ok:
                                bad_name = str(pn)
                                break
                    if bad_name is not None:
                        setattr(self, "_enn_fuser_param_bad", True)
                        _maybe_dump_fuser_nonfinite(
                            f"perceiver_params_pre.{call_i}",
                            all_tokens=all_tokens,
                            fused_tokens=None,
                            context=None,
                            attn_bias=attn_bias,
                        )
                        _LOGGER.error(
                            "[ENN] Fuser: perceiver parameters already non-finite at call=%d: %s",
                            int(call_i),
                            str(bad_name),
                        )
                        if strict_params:
                            raise RuntimeError(
                                f"[ENN] Fuser: perceiver params non-finite at call={int(call_i)}: {bad_name}"
                            )
            bg = getattr(self, "_backbone_graph", None)
            if isinstance(bg, nn.Module) and cg_ok:
                fused_tokens, context = cast(
                    tuple[torch.Tensor, torch.Tensor], bg(all_tokens, attn_bias)
                )
            else:
                if cg_ok:
                    cudagraph_mark_step_begin()
                    fused_tokens = self._call_perceiver(all_tokens, attn_bias=attn_bias)
                    cudagraph_mark_step_end()
                else:
                    fused_tokens = self._call_perceiver(all_tokens, attn_bias=attn_bias)
                graph_break()
                if fallback_enabled and _pair_has_nonfinite(fused_tokens):
                    if not bool(getattr(self, "_nonfinite_fallback_warned", False)):
                        _LOGGER.warning(
                            "[ENN] Fuser: perceiver produced non-finite fused_tokens; "
                            "falling back to raw template tokens (all_tokens). "
                            "Set ENN_FUSER_NONFINITE_FALLBACK=0 to disable this."
                        )
                        setattr(self, "_nonfinite_fallback_warned", True)

                    _maybe_dump_fuser_nonfinite(
                        "fused_tokens",
                        all_tokens=all_tokens,
                        fused_tokens=fused_tokens,
                        context=None,
                        attn_bias=attn_bias,
                    )

                    if env_bool("ENN_FUSER_DIAG_NONFINITE_PARAMS", default=True) and (not bool(getattr(self, "_nonfinite_params_warned", False))):
                        with contextlib.suppress(Exception):
                            bad: list[str] = []
                            for pn, pv in self.perceiver.named_parameters():
                                if pv is None or (not isinstance(pv, torch.Tensor)):
                                    continue
                                if pv.is_floating_point() and (not torch.isfinite(pv).all()):
                                    bad.append(str(pn))
                                    if len(bad) >= 8:
                                        break
                            if bad:
                                _LOGGER.error(
                                    "[ENN] Fuser: non-finite parameters detected in perceiver: %s",
                                    ", ".join(bad),
                                )
                                setattr(self, "_nonfinite_params_warned", True)

                    fused_tokens = _fallback_fused_from_all()
                context = self.decode(fused_tokens, apply_norm=True)

            if fallback_enabled and (_pair_has_nonfinite(fused_tokens) or _pair_has_nonfinite(context)):
                if not bool(getattr(self, "_nonfinite_fallback_warned", False)):
                    _LOGGER.warning(
                        "[ENN] Fuser: non-finite outputs detected (tokens/context); "
                        "falling back to raw template tokens (all_tokens)."
                    )
                    setattr(self, "_nonfinite_fallback_warned", True)
                _maybe_dump_fuser_nonfinite(
                    "tokens_or_context",
                    all_tokens=all_tokens,
                    fused_tokens=fused_tokens,
                    context=context,
                    attn_bias=attn_bias,
                )
                fused_tokens = _fallback_fused_from_all()
                graph_break()
                context = self.decode(fused_tokens, apply_norm=True)

        if (not torch.is_grad_enabled()) and (getattr(fused_tokens.device, "type", None) == "cuda"):
            if env_bool("ENN_PRED_COLLAPSE_STAGE_DIAG", default=False):
                with contextlib.suppress(Exception):
                    prev = getattr(self, "_enn_last_fuser_diag", None)
                    if isinstance(prev, dict):
                        prev["fused_tokens"] = {
                            "pair_absmax": float(fused_tokens[:2].abs().max().item()),
                            "pair_nonfinite": int((~torch.isfinite(fused_tokens[:2])).sum().item())
                            if fused_tokens.is_floating_point() else 0,
                        }
                        prev["context"] = {
                            "pair_absmax": float(context[:2].abs().max().item()),
                            "pair_nonfinite": int((~torch.isfinite(context[:2])).sum().item())
                            if context.is_floating_point() else 0,
                        }

        if (not is_export_or_trace()) and env_bool("ENN_FUSER_NONFINITE_FALLBACK", default=True):
            allow_train = env_bool("ENN_FUSER_NONFINITE_FALLBACK_TRAINING", default=False)
            if (not torch.is_grad_enabled()) or allow_train:
                has_nonfinite = False
                if isinstance(fused_tokens, torch.Tensor) and fused_tokens.is_floating_point():
                    with torch.no_grad():
                        try:
                            ft_absmax = fused_tokens.detach().abs().amax()
                            has_nonfinite = not bool(torch.isfinite(ft_absmax).item())
                        except Exception:
                            has_nonfinite = not bool(torch.isfinite(fused_tokens).all().item())
                if (not has_nonfinite) and isinstance(context, torch.Tensor) and context.is_floating_point():
                    with torch.no_grad():
                        try:
                            ctx_absmax = context.detach().abs().amax()
                            has_nonfinite = not bool(torch.isfinite(ctx_absmax).item())
                        except Exception:
                            has_nonfinite = not bool(torch.isfinite(context).all().item())
                if has_nonfinite:
                    if not getattr(self, "_enn_nonfinite_fallback_logged", False):
                        _LOGGER.warning(
                            "[ENN] Fuser perceiver produced non-finite output; "
                            "falling back to pre-perceiver tokens (set ENN_FUSER_NONFINITE_FALLBACK=0 to disable)."
                        )
                        setattr(self, "_enn_nonfinite_fallback_logged", True)
                    _maybe_dump_fuser_nonfinite(
                        "infer_outputs",
                        all_tokens=all_tokens,
                        fused_tokens=fused_tokens,
                        context=context,
                        attn_bias=attn_bias,
                    )
                    fb = all_tokens
                    try:
                        want_n = int(fused_tokens.shape[1])
                        have_n = int(fb.shape[1])
                        if have_n > want_n:
                            fb = fb[:, :want_n]
                        elif have_n < want_n:
                            if have_n <= 0:
                                fb = fb.new_zeros((int(fb.shape[0]), want_n, int(fb.shape[-1])))
                            else:
                                reps = int(math.ceil(float(want_n) / float(have_n)))
                                fb = fb.repeat(1, reps, 1)[:, :want_n]
                    except Exception:
                        pass
                    fused_tokens = fb.to(dtype=fused_tokens.dtype, device=fused_tokens.device)
                    graph_break()
                    context = self.decode(fused_tokens, apply_norm=True)
                    if env_bool("ENN_FUSER_DIAG_NONFINITE_PARAMS", default=False) and not getattr(
                        self, "_enn_nonfinite_params_logged", False
                    ):
                        bad_params: list[str] = []
                        with torch.no_grad():
                            for n, p in self.perceiver.named_parameters(recurse=True):
                                if not isinstance(p, torch.Tensor) or (not p.is_floating_point()):
                                    continue
                                try:
                                    a = p.detach().abs().amax()
                                    if not bool(torch.isfinite(a).item()):
                                        bad_params.append(str(n))
                                except Exception:
                                    with contextlib.suppress(Exception):
                                        if not bool(torch.isfinite(p).all().item()):
                                            bad_params.append(str(n))
                        if bad_params:
                            head = ", ".join(bad_params[:20])
                            more = "" if len(bad_params) <= 20 else f" (+{len(bad_params)-20} more)"
                            _LOGGER.warning(
                                "[ENN] Perceiver parameters include non-finite values: %s%s",
                                head,
                                more,
                            )
                        setattr(self, "_enn_nonfinite_params_logged", True)
        if return_temporal_state:
            if state_is_map:
                return fused_tokens, context, next_state_map
            return fused_tokens, context, next_state
        return fused_tokens, context

    def forward_export(
        self: Self,
        x: torch.Tensor,
        *args: Any,
        causal_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs
        plan = getattr(self, "_exec_plan", None)
        if not isinstance(plan, dict):
            plan = self._build_exec_plan_unlocked()

        roots = [str(r) for r in (plan.get("roots") or []) if str(r) in self.tasks]
        order = [str(n) for n in (plan.get("order") or []) if str(n) in self.tasks]
        if not roots:
            roots = [str(k) for k in self.tasks.keys()]
        if not order:
            order = [str(k) for k in self.tasks.keys()]

        token_cache: dict[str, torch.Tensor] = {}

        def _linear_fuse(
            child_names: Sequence[str],
            child_tokens: Sequence[torch.Tensor],
        ) -> torch.Tensor:
            if not child_tokens:
                raise RuntimeError("linear reduction requires at least one child")
            tok0 = child_tokens[0]
            B = tok0.size(0)
            latents = self.perceiver.latents.unsqueeze(0).expand(B, -1, -1)
            if latents.dtype != tok0.dtype:
                latents = latents.to(dtype=tok0.dtype)
            if latents.device != tok0.device:
                latents = latents.to(device=tok0.device)
            j = 0
            for i in range(int(getattr(self.perceiver, "depth", 1))):
                base = latents
                delta_sum = torch.zeros_like(base)
                for nm, toks in zip(child_names, child_tokens):
                    mod = self.tasks[nm] if nm in self.tasks else None
                    w = getattr(mod, "weight", None)
                    e = getattr(mod, "eps", None)
                    if not isinstance(w, torch.Tensor):
                        w = torch.as_tensor(1.0, dtype=tok0.dtype, device=tok0.device)
                    if not isinstance(e, torch.Tensor):
                        e = torch.as_tensor(1e-6, dtype=tok0.dtype, device=tok0.device)
                    coef = w.to(device=tok0.device, dtype=base.dtype).reshape(())
                    eps_t = e.to(device=tok0.device, dtype=base.dtype).reshape(())
                    coef = torch.where(coef == 0, eps_t, coef)
                    out = self.perceiver.cross[i](base, toks, attn_bias=None)
                    delta_sum = delta_sum + coef * (out - base)
                latents = base + delta_sum
                for _ in range(int(getattr(self.perceiver, "self_attn_layers", 0))):
                    if j < len(self.perceiver.self_blocks):
                        latents = self.perceiver.self_blocks[j](latents)
                    j += 1
            return self.perceiver.norm(latents)

        for name in order:
            mod = self.tasks[name]
            if isinstance(mod, SubFuser):
                child_names = [str(c) for c in list(getattr(mod, "_children", []) or [])]
                child_names = [c for c in child_names if c in token_cache]
                if not child_names:
                    raise RuntimeError(f"Taskset '{name}' has no executed children")
                child_tokens = [token_cache[c] for c in child_names]
                red = str(getattr(mod, "reduction", "mean"))
                if red == "mean":
                    all_tokens = child_tokens[0] if len(child_tokens) == 1 else torch.cat(child_tokens, dim=1)
                    attn_bias = self._build_attn_bias(
                        child_names,
                        child_tokens,
                        device=all_tokens.device,
                        dtype=all_tokens.dtype,
                    )
                    token_cache[name] = self._call_perceiver(all_tokens, attn_bias=attn_bias)
                elif red == "linear":
                    token_cache[name] = self._call_linear_fuse(_linear_fuse, child_names, child_tokens, device=child_tokens[0].device)
                else:
                    raise ValueError(f"Unknown taskset reduction '{red}'")
                continue

            out_tokens: Any
            if isinstance(mod, Template) and str(getattr(mod, "mode", "")) == "temporal":
                out_tokens = mod(
                    x,
                    state=None,
                    return_state=False,
                    causal_mask=causal_mask,
                )
            else:
                out_tokens = mod(x, causal_mask=causal_mask) if isinstance(mod, Template) else mod(x)

            if not isinstance(out_tokens, torch.Tensor):
                raise TypeError(f"Task '{name}' must return a torch.Tensor")
            sm = self._user_submodels.get(str(name))
            if sm is not None:
                out_tokens = sm(out_tokens)
                if not isinstance(out_tokens, torch.Tensor):
                    raise TypeError(
                        f"BYOM for task '{name}' must return torch.Tensor; got {type(out_tokens)}"
                    )
            token_cache[name] = out_tokens

        token_sets = [token_cache[r] for r in roots if r in token_cache]
        if not token_sets:
            raise RuntimeError("No root nodes produced tokens")

        root_reduction = str(getattr(self, "_root_reduction", "mean"))
        if root_reduction == "linear":
            fused_tokens = self._call_linear_fuse(_linear_fuse, roots, token_sets, device=token_sets[0].device)
        elif root_reduction == "mean":
            all_tokens = token_sets[0] if len(token_sets) == 1 else torch.cat(token_sets, dim=1)
            attn_bias = self._build_attn_bias(
                roots,
                token_sets,
                device=all_tokens.device,
                dtype=all_tokens.dtype,
            )
            fused_tokens = self._call_perceiver(all_tokens, attn_bias=attn_bias)
        else:
            raise ValueError(f"Unknown root reduction '{root_reduction}'")

        context = self.decode(fused_tokens, apply_norm=True)
        return cast(torch.Tensor, fused_tokens), cast(torch.Tensor, context)

    def decode(
        self: Self, tokens: torch.Tensor, *args: Any, apply_norm: bool = True
    ) -> torch.Tensor:
        if apply_norm:
            dg = getattr(self, "_decode_graph", None)
            if isinstance(dg, nn.Module):
                return cast(torch.Tensor, dg(tokens))
        x = tokens
        if apply_norm:
            x = self.norm(x)
        pooled = x.mean(dim=1)
        out = self.head(pooled)
        return out.reshape(out.shape[0], *self.out_shape)


class Model(nn.Module):
    @staticmethod
    def _get_cfg(
        cfg: object,
        name: str,
        default: TConfig,
        type_: type[TConfig] = float,
    ) -> TConfig:
        return type_(getattr(cfg, name, default))

    def __init__(
        self: Self, in_dim: int, out_shape: Sequence[int], config: ModelConfig
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_shape = tuple((int(x) for x in out_shape))
        self.out_dim = int(math.prod(self.out_shape))
        if config.device is not None:
            self._device = torch.device(config.device)
            if (
                self._device.type in {"cuda", "xpu"}
                and self._device.index is not None
            ):
                with contextlib.suppress(Exception):
                    from ..core.system import set_accelerator_index

                    set_accelerator_index(
                        self._device.type, int(self._device.index)
                    )
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
            nn.Linear(self.in_dim, self.out_dim).to(self._device)
            if self.is_norm_linear
            else None
        )
        self.fuser = Fuser(self.in_dim, self.out_shape, config=config).to(
            self._device
        )
        self.processor = self.fuser
        bucket = self._get_cfg(config, "length_bucket_multiple", 64, int)
        self.temporal_token_collector = Collector(
            int(config.d_model),
            int(config.heads),
            depth=max(1, int(getattr(config, "temporal_depth", 1))),
            mlp_ratio=float(getattr(config, "mlp_ratio", 4.0)),
            dropout=float(getattr(config, "dropout", 0.0)),
            batch_first=True,
            length_bucket_multiple=bucket,
        ).to(self._device)
        self.controller = self.temporal_token_collector
        self.delta_gate: Optional[SigmoidGate]
        k_def = self._get_cfg(config, "delta_gate_fallback_k", 6.0)
        self.delta_gate_fallback_k: float = float(k_def)
        self.delta_gate_fallback_k_low = float(
            getattr(config, "delta_gate_fallback_k_low", k_def) or k_def
        )
        self.delta_gate_fallback_k_high = float(
            getattr(config, "delta_gate_fallback_k_high", k_def) or k_def
        )
        self.delta_gate_auto_k_enabled: bool = True
        self.delta_gate_auto_k_interval = max(
            1, self._get_cfg(config, "delta_gate_auto_k_interval", 100, int)
        )
        self.delta_gate_auto_k_warmup = max(
            0, self._get_cfg(config, "delta_gate_auto_k_warmup", 0, int)
        )
        self.delta_gate_auto_k_ema_alpha = self._get_cfg(
            config, "delta_gate_auto_k_ema_alpha", 0.1
        )
        self.delta_gate_auto_k_target_tight = self._get_cfg(
            config, "delta_gate_auto_k_target_tight", 0.02
        )
        self.delta_gate_auto_k_tolerance = self._get_cfg(
            config, "delta_gate_auto_k_tolerance", 0.5
        )
        self.delta_gate_auto_k_step_up = self._get_cfg(
            config, "delta_gate_auto_k_step_up", 0.1
        )
        self.delta_gate_auto_k_step_down = self._get_cfg(
            config, "delta_gate_auto_k_step_down", 0.02
        )
        self.delta_gate_auto_k_step_up_low = self._get_cfg(
            config,
            "delta_gate_auto_k_step_up_low",
            self.delta_gate_auto_k_step_up,
        )
        self.delta_gate_auto_k_step_down_low = self._get_cfg(
            config,
            "delta_gate_auto_k_step_down_low",
            self.delta_gate_auto_k_step_down,
        )
        self.delta_gate_auto_k_step_up_high = self._get_cfg(
            config,
            "delta_gate_auto_k_step_up_high",
            self.delta_gate_auto_k_step_up,
        )
        self.delta_gate_auto_k_step_down_high = self._get_cfg(
            config,
            "delta_gate_auto_k_step_down_high",
            self.delta_gate_auto_k_step_down,
        )
        self.delta_gate_auto_k_edge_enabled: bool = True
        self.delta_gate_auto_k_target_edge = self._get_cfg(
            config, "delta_gate_auto_k_target_edge", 0.05
        )
        self.delta_gate_auto_k_edge_tolerance = self._get_cfg(
            config, "delta_gate_auto_k_edge_tolerance", 0.5
        )
        self.delta_gate_auto_k_edge_ema_alpha: float = float(
            getattr(
                config,
                "delta_gate_auto_k_edge_ema_alpha",
                self.delta_gate_auto_k_ema_alpha,
            )
        )
        self.delta_gate_auto_k_edge_step_down_low: float = float(
            getattr(config, "delta_gate_auto_k_edge_step_down_low", 0.01)
        )
        self.delta_gate_auto_k_edge_step_down_high: float = float(
            getattr(config, "delta_gate_auto_k_edge_step_down_high", 0.01)
        )
        self.delta_gate_auto_k_min: float = float(
            getattr(config, "delta_gate_auto_k_min", 1.0)
        )
        self.delta_gate_auto_k_max: float = float(
            getattr(config, "delta_gate_auto_k_max", 16.0)
        )
        self.delta_gate_auto_k_width_frac: float = float(
            getattr(config, "delta_gate_auto_k_width_frac", 0.05)
        )
        self.delta_gate_auto_k_edge_frac: float = float(
            getattr(config, "delta_gate_auto_k_edge_frac", 0.02)
        )
        self.delta_gate_auto_k_log_interval: int = int(
            getattr(config, "delta_gate_auto_k_log_interval", 200) or 0
        )
        if self.delta_gate_auto_k_enabled and (
            self.delta_gate_fallback_k_low <= 0.0
            or self.delta_gate_fallback_k_high <= 0.0
        ):
            m = max(float(self.delta_gate_auto_k_min), 1e-6)
            self.delta_gate_fallback_k_low = max(
                self.delta_gate_fallback_k_low, m
            )
            self.delta_gate_fallback_k_high = max(
                self.delta_gate_fallback_k_high, m
            )
        self.delta_gate_fallback_enabled: bool = bool(
            self.delta_gate_fallback_k_low > 0.0
            and self.delta_gate_fallback_k_high > 0.0
        )
        self.delta_gate_tile_size: Optional[int] = getattr(
            config, "delta_gate_tile_size", None
        )
        raw_tile_shape = getattr(config, "delta_gate_tile_shape", None)
        tile_shape: Optional[Tuple[int, ...]]
        try:
            if raw_tile_shape is None:
                tile_shape = None
            else:
                if isinstance(raw_tile_shape, int) and not isinstance(
                    raw_tile_shape, bool
                ):
                    tile_shape = (int(raw_tile_shape),)
                else:
                    tile_shape = tuple(int(v) for v in raw_tile_shape)
                tile_shape = tuple(max(1, int(v)) for v in tile_shape)
                out_ndim = int(len(self.out_shape))
                if out_ndim > 0:
                    if len(tile_shape) == 1:
                        tile_shape = tile_shape * out_ndim
                    elif len(tile_shape) < out_ndim:
                        tile_shape = (1,) * (
                            out_ndim - len(tile_shape)
                        ) + tile_shape
                    elif len(tile_shape) > out_ndim:
                        tile_shape = tile_shape[-out_ndim:]
        except Exception:
            tile_shape = None
        self.delta_gate_tile_shape: Optional[Tuple[int, ...]] = tile_shape
        self.delta_gate_bounds_use_quantile: bool = bool(
            getattr(config, "delta_gate_bounds_use_quantile", False)
        )
        self.delta_gate_bounds_q_low: float = float(
            getattr(config, "delta_gate_bounds_q_low", 0.005)
        )
        self.delta_gate_bounds_q_high: float = float(
            getattr(config, "delta_gate_bounds_q_high", 0.995)
        )
        self.delta_gate_bounds_q_max_samples: int = int(
            getattr(config, "delta_gate_bounds_q_max_samples", 8192) or 0
        )
        self.delta_gate_bounds_clip_to_minmax: bool = bool(
            getattr(config, "delta_gate_bounds_clip_to_minmax", True)
        )
        try:
            self.register_buffer(
                "delta_gate_fallback_k_low_buf",
                torch.tensor(
                    float(self.delta_gate_fallback_k_low), dtype=torch.float32
                ),
                persistent=True,
            )
            self.register_buffer(
                "delta_gate_fallback_k_high_buf",
                torch.tensor(
                    float(self.delta_gate_fallback_k_high), dtype=torch.float32
                ),
                persistent=True,
            )
            self.register_buffer(
                "delta_gate_fallback_k_buf",
                torch.tensor(
                    float(
                        0.5
                        * (
                            self.delta_gate_fallback_k_low
                            + self.delta_gate_fallback_k_high
                        )
                    ),
                    dtype=torch.float32,
                ),
                persistent=True,
            )
            self.register_buffer(
                "delta_gate_auto_k_step_buf",
                torch.tensor(0, dtype=torch.int64),
                persistent=True,
            )
            self.register_buffer(
                "delta_gate_auto_k_ema_low_buf",
                torch.tensor(0.0, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "delta_gate_auto_k_ema_high_buf",
                torch.tensor(0.0, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "delta_gate_auto_k_edge_ema_low_buf",
                torch.tensor(0.0, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "delta_gate_auto_k_edge_ema_high_buf",
                torch.tensor(0.0, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "delta_gate_auto_k_edge_ema_buf",
                torch.tensor(0.0, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "delta_gate_auto_k_ema_buf",
                torch.tensor(0.0, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "delta_gate_auto_k_updates_buf",
                torch.tensor(0, dtype=torch.int64),
                persistent=True,
            )
        except Exception:
            pass
        self.delta_gate = SigmoidGate(
            d_model=int(config.d_model),
            hidden_dim=int(getattr(config, "delta_gate_hidden_dim", 64)),
            detach_inputs=bool(
                getattr(config, "delta_gate_detach_inputs", True)
            ),
            p_floor=float(getattr(config, "delta_gate_p_floor", 0.0)),
            p_ceil=float(getattr(config, "delta_gate_p_ceil", 1.0)),
            tile_size=getattr(config, "delta_gate_tile_size", None),
            tile_shape=self.delta_gate_tile_shape,
            event_shape=self.out_shape,
            clip_eps=float(getattr(config, "delta_gate_clip_eps", 1e-6)),
            eps=float(getattr(config, "delta_gate_eps", 1e-6)),
            stat_width_frac=float(
                getattr(config, "delta_gate_auto_k_width_frac", 0.05)
            ),
            stat_edge_frac=float(
                getattr(config, "delta_gate_auto_k_edge_frac", 0.02)
            ),
        ).to(self._device)
        self.unsup_xx_weight = float(getattr(config, "unsup_xx_weight", 0.0))
        self.unsup_yy_weight = float(getattr(config, "unsup_yy_weight", 0.0))
        self.p_prior_weight = float(getattr(config, "p_prior_weight", 0.0))
        self.p_prior_alpha = float(getattr(config, "p_prior_alpha", 2.0))
        self.p_prior_beta = float(getattr(config, "p_prior_beta", 2.0))
        self.delta_gate_edge_reg_weight = float(
            getattr(config, "delta_gate_edge_reg_weight", 0.0)
        )
        self.delta_gate_edge_reg_frac = float(
            getattr(
                config,
                "delta_gate_edge_reg_frac",
                getattr(config, "delta_gate_auto_k_edge_frac", 0.02),
            )
        )
        self.delta_gate_edge_reg_min_width_frac = float(
            getattr(
                config,
                "delta_gate_edge_reg_min_width_frac",
                getattr(config, "delta_gate_auto_k_width_frac", 0.05),
            )
        )
        self.delta_gate_edge_reg_power = float(
            getattr(config, "delta_gate_edge_reg_power", 2.0)
        )
        self.delta_gate_budget_weight = float(
            getattr(config, "delta_gate_budget_weight", 0.0)
        )
        self.delta_gate_budget_target = float(
            getattr(config, "delta_gate_budget_target", 0.5)
        )
        self.delta_gate_tv_weight = float(
            getattr(config, "delta_gate_tv_weight", 0.0)
        )
        self.delta_gate_tv_power = float(
            getattr(config, "delta_gate_tv_power", 1.0)
        )
        self.delta_gate_teacher_weight = float(
            getattr(config, "delta_gate_teacher_weight", 0.0)
        )
        self.delta_gate_teacher_temp = float(
            getattr(config, "delta_gate_teacher_temp", 0.25)
        )
        self.delta_gate_teacher_tau = float(
            getattr(config, "delta_gate_teacher_tau", 0.0)
        )
        self.delta_gate_teacher_relu = bool(
            getattr(config, "delta_gate_teacher_relu", False)
        )
        try:
            w_low_cfg = getattr(config, "delta_gate_edge_reg_weight_low", None)
            w_high_cfg = getattr(
                config, "delta_gate_edge_reg_weight_high", None
            )
            self.delta_gate_edge_reg_weight_low = (
                float(self.delta_gate_edge_reg_weight)
                if w_low_cfg is None
                else float(w_low_cfg)
            )
            self.delta_gate_edge_reg_weight_high = (
                float(self.delta_gate_edge_reg_weight)
                if w_high_cfg is None
                else float(w_high_cfg)
            )
        except Exception:
            self.delta_gate_edge_reg_weight_low = float(
                self.delta_gate_edge_reg_weight
            )
            self.delta_gate_edge_reg_weight_high = float(
                self.delta_gate_edge_reg_weight
            )
        self.delta_gate_edge_reg_fallback_only = bool(
            getattr(config, "delta_gate_edge_reg_fallback_only", False)
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
        self._runtime_lock = Mutex()
        self._eager_fuser_perceiver = getattr(self.fuser, "perceiver", None)
        self._eager_perceiver_cross: Optional[list[nn.Module]] = None
        self._eager_perceiver_self_blocks: Optional[list[nn.Module]] = None
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
        requested_compile_mode = str(compile_mode_canonical)

        warmup_steps = 0
        if (
            compile_mode_canonical in {"max-autotune", "max-autotune-no-cudagraphs"}
            and getattr(self._device, "type", None) == "cuda"
        ):
            try:
                warmup_steps = int(
                    env_first_int(
                        (
                            "ENN_COMPILE_MAX_AUTOTUNE_WARMUP_STEPS",
                            "ENN_MAX_AUTOTUNE_WARMUP_STEPS",
                        ),
                        default=10,
                    )
                    or 10
                )
            except Exception:
                warmup_steps = 10
            warmup_steps = max(0, int(warmup_steps))
            if warmup_steps > 0:
                compile_mode_canonical = "reduce-overhead"

        self._enn_compile_requested_mode: str = (
            requested_compile_mode if warmup_steps > 0 else ""
        )
        self._enn_compile_upgrade_after_steps: int = int(warmup_steps)
        self._enn_compile_upgrade_done: bool = bool(warmup_steps <= 0)
        self._enn_compile_upgrade_inflight: bool = False
        self._enn_compile_active_mode: str = str(compile_mode_canonical)
        set_runtime_cfg("compile_mode", compile_mode_canonical)
        compile_mode_arg = (
            None
            if compile_mode_canonical == "disabled"
            else compile_mode_canonical
        )
        compile_requested = compile_mode_arg is not None
        compile_available = bool(torch_compiler_supported())
        compile_enabled = bool(compile_requested and compile_available)
        nogil_opt = False
        with contextlib.suppress(Exception):
            nogil_opt = bool(CPU.is_optimized_for_no_gil())
        if nogil_opt and compile_enabled:
            _LOGGER.info(
                "No-GIL optimized mode detected; using conservative torch.compile defaults "
                "(disable cudagraphs and prefer staged compilation for heavy submodules)"
            )
        if compile_requested and not compile_available:
            _LOGGER.warning(
                "torch.compile requested (compile_mode=%r) but torch.compile is unavailable; running eagerly",
                raw_mode,
            )
        compile_dynamic = bool(
            getattr(
                config,
                "compile_dynamic",
                compile_mode_canonical == "reduce-overhead",
            )
        )
        compile_cudagraphs_default = (
            not bool(nogil_opt)
        ) and compile_mode_canonical not in {
            "aot-eager",
            "max-autotune-no-cudagraphs",
        }
        compile_cudagraphs = bool(
            getattr(config, "compile_cudagraphs", compile_cudagraphs_default)
        )
        self._compile_cudagraphs = bool(
            compile_enabled
            and compile_cudagraphs
            and getattr(self._device, "type", None) == "cuda"
        )
        set_runtime_cfg("compile_cudagraphs", bool(self._compile_cudagraphs))
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
        self._enn_compile_dynamic: bool = bool(compile_dynamic)
        self._enn_compile_options: dict[str, Any] | None = (
            dict(compile_options) if isinstance(compile_options, dict) else None
        )

        def _empty_cache() -> None:
            if getattr(self._device, "type", None) == "cuda":
                empty_device_cache(
                    device=self._device, do_gc=True, min_interval_s=0.0
                )

        def _compile_one(
            mod: nn.Module,
            *args: Any,
            label: str,
            options: dict[str, Any] | None = None,
        ) -> nn.Module:
            try:
                return compile_module(
                    mod,
                    mode=compile_mode_arg,
                    fullgraph=False,
                    dynamic=compile_dynamic,
                    backend="inductor",
                    options=(
                        options if options is not None else compile_options
                    ),
                    disable=False,
                )
            except Exception:
                _LOGGER.warning(
                    "torch.compile failed for %s; keeping eager module",
                    label,
                    exc_info=True,
                )
                return mod

        def _modulelist_swap(
            modlist: nn.ModuleList, new_items: Sequence[nn.Module]
        ) -> list[nn.Module]:
            old_items = list(modlist)
            for i, item in enumerate(new_items):
                with contextlib.suppress(Exception):
                    modlist[i] = item
            return old_items

        def _compile_modulelist_inplace(
            modlist: nn.ModuleList, *args: Any, label: str
        ) -> tuple[bool, list[nn.Module], list[nn.Module]]:
            eager_items = list(modlist)
            compiled_items: list[nn.Module] = []
            compiled_any = False
            for i, child in enumerate(eager_items):
                compiled = (
                    _compile_one(child, label=f"{label}[{i}]")
                    if isinstance(child, nn.Module)
                    else child
                )
                compiled_items.append(compiled)
                if compiled is not child:
                    compiled_any = True
                _empty_cache()
            if compiled_any:
                _modulelist_swap(modlist, compiled_items)
            return compiled_any, eager_items, compiled_items

        def _compile_perceiver_piecewise(perceiver: nn.Module) -> bool:
            compiled_any = False
            cross = getattr(perceiver, "cross", None)
            if isinstance(cross, nn.ModuleList):
                c_any, eager_items, _ = _compile_modulelist_inplace(
                    cross, label="perceiver.cross"
                )
                if c_any:
                    compiled_any = True
                    self._eager_perceiver_cross = eager_items
            self_blocks = getattr(perceiver, "self_blocks", None)
            if isinstance(self_blocks, nn.ModuleList):
                s_any, eager_items, _ = _compile_modulelist_inplace(
                    self_blocks, label="perceiver.self_blocks"
                )
                if s_any:
                    compiled_any = True
                    self._eager_perceiver_self_blocks = eager_items
            return compiled_any

        staged_heavy = bool(
            nogil_opt
            or compile_mode_canonical
            in {"max-autotune", "max-autotune-no-cudagraphs"}
        )
        compiled_decode = False
        compiled_perceiver = False
        if compile_enabled:
            with compile_patch_ctx:
                try:
                    from .graph import GraphSequential

                    _decode_mod = (
                        GraphSequential(
                            steps=[
                                GraphSequential.path("norm"),
                                GraphSequential.mean(dim=1),
                                GraphSequential.path("head"),
                            ],
                            out_shape=self.out_shape,
                            name="decode",
                            root=self.processor,
                        )
                        .to(self._device)
                        .bind()
                    )
                    _compiled = _compile_one(_decode_mod, label="decode head")
                    self._decode_compiled = _compiled
                    compiled_decode = _compiled is not _decode_mod
                except Exception:
                    self._decode_compiled = None
                    _LOGGER.warning(
                        "torch.compile failed for decode head; continuing without compilation",
                        exc_info=True,
                    )
                _empty_cache()
                try:
                    perceiver = getattr(self.fuser, "perceiver", None)
                    if isinstance(perceiver, nn.Module):
                        if staged_heavy:
                            compiled_perceiver = _compile_perceiver_piecewise(
                                perceiver
                            )
                        if not compiled_perceiver:
                            orig = getattr(
                                self, "_eager_fuser_perceiver", None
                            )
                            if isinstance(orig, nn.Module):
                                _compiled = _compile_one(
                                    orig, label="fuser.perceiver"
                                )
                                self.fuser.perceiver = _compiled
                                compiled_perceiver = _compiled is not orig
                except Exception:
                    _LOGGER.warning(
                        "torch.compile failed for fuser perceiver; continuing eagerly",
                        exc_info=True,
                    )
                _empty_cache()
        self._compiled_submodules = {
            "decode": bool(compiled_decode),
            "perceiver": bool(compiled_perceiver),
        }
        self._pad_compiled_microbatch = bool(
            compiled_decode or compiled_perceiver
        )
        self._amp_dtype_cache: dict[Tuple[Any, ...], torch.dtype] = {}
        self._amp_dtype_cache_last_key: Tuple[Any, ...] | None = None
        self._amp_dtype_cache_last_dtype: torch.dtype | None = None
        self._amp_dtype_cache_max = 64
        self._amp_dtype_cache_lock = Mutex()
        self._amp_dtype_cache_use_lock = not bool(is_gil_enabled())
        self.__config = config
        self.__enn_instance_config__ = config

    def maybe_upgrade_compile_mode(
        self: Self,
        *,
        step_total: int,
        logger: logging.Logger | None = None,
    ) -> bool:
        req = str(getattr(self, "_enn_compile_requested_mode", "") or "").strip()
        if not req:
            return False
        if bool(getattr(self, "_enn_compile_upgrade_done", False)):
            return False
        try:
            step_i = int(step_total)
        except Exception:
            return False
        after = int(getattr(self, "_enn_compile_upgrade_after_steps", 0) or 0)
        if after <= 0 or step_i < after:
            return False
        if not torch_compiler_supported():
            with self._runtime_lock:
                self._enn_compile_upgrade_done = True
                self._enn_compile_requested_mode = ""
            return False

        log = logger if isinstance(logger, logging.Logger) else _LOGGER

        with self._runtime_lock:
            if self._enn_compile_upgrade_done or self._enn_compile_upgrade_inflight:
                return False
            self._enn_compile_upgrade_inflight = True

        def _empty_cache() -> None:
            if getattr(self._device, "type", None) == "cuda":
                empty_device_cache(device=self._device, do_gc=True, min_interval_s=0.0)

        try:
            target_mode = canonicalize_compile_mode(req)
            if target_mode not in {"max-autotune", "max-autotune-no-cudagraphs"}:
                with self._runtime_lock:
                    self._enn_compile_upgrade_done = True
                    self._enn_compile_requested_mode = ""
                return False

            options = getattr(self, "_enn_compile_options", None)
            dyn = bool(getattr(self, "_enn_compile_dynamic", False))

            def _compile_one(
                mod: nn.Module,
                *,
                label: str,
                options_override: dict[str, Any] | None = None,
            ) -> nn.Module:
                try:
                    return compile_module(
                        mod,
                        mode=target_mode,
                        fullgraph=False,
                        dynamic=bool(dyn),
                        backend="inductor",
                        options=(
                            options_override if options_override is not None else options
                        ),
                        disable=False,
                    )
                except Exception as e:
                    if is_oom_error(e) or ("out of memory" in str(e).lower()):
                        raise
                    log.warning(
                        "torch.compile upgrade failed for %s; keeping eager/previous module",
                        label,
                        exc_info=True,
                    )
                    return mod

            with self._runtime_lock:
                orig = getattr(self, "_eager_fuser_perceiver", None)
                if isinstance(orig, nn.Module):
                    with contextlib.suppress(Exception):
                        self.fuser.perceiver = orig
                perceiver = getattr(self.fuser, "perceiver", None)
                if isinstance(perceiver, nn.Module):
                    cross = getattr(perceiver, "cross", None)
                    eager_cross = getattr(self, "_eager_perceiver_cross", None)
                    if (
                        isinstance(cross, nn.ModuleList)
                        and isinstance(eager_cross, list)
                        and len(eager_cross) == len(cross)
                    ):
                        for i, item in enumerate(eager_cross):
                            with contextlib.suppress(Exception):
                                cross[i] = item
                    self_blocks = getattr(perceiver, "self_blocks", None)
                    eager_self = getattr(self, "_eager_perceiver_self_blocks", None)
                    if (
                        isinstance(self_blocks, nn.ModuleList)
                        and isinstance(eager_self, list)
                        and len(eager_self) == len(self_blocks)
                    ):
                        for i, item in enumerate(eager_self):
                            with contextlib.suppress(Exception):
                                self_blocks[i] = item

            try:
                from .graph import GraphSequential

                decode_graph = (
                    GraphSequential(
                        steps=[
                            GraphSequential.path("norm"),
                            GraphSequential.mean(dim=1),
                            GraphSequential.path("head"),
                        ],
                        out_shape=self.out_shape,
                        name="decode",
                        root=self.processor,
                    )
                    .to(self._device)
                    .bind()
                )
                self._decode_compiled = _compile_one(decode_graph, label="decode head")
            except Exception as e:
                if is_oom_error(e) or ("out of memory" in str(e).lower()):
                    raise
                log.warning(
                    "torch.compile upgrade failed for decode head; keeping existing",
                    exc_info=True,
                )

            _empty_cache()

            perceiver = getattr(self.fuser, "perceiver", None)
            if isinstance(perceiver, nn.Module):
                staged_heavy = bool(
                    CPU.is_optimized_for_no_gil()
                    or target_mode in {"max-autotune", "max-autotune-no-cudagraphs"}
                )
                compiled_any = False
                if staged_heavy:
                    cross = getattr(perceiver, "cross", None)
                    if isinstance(cross, nn.ModuleList):
                        eager_items = list(cross)
                        compiled_items: list[nn.Module] = []
                        any_changed = False
                        for i, child in enumerate(eager_items):
                            compiled = _compile_one(
                                child, label=f"perceiver.cross[{i}]"
                            )
                            compiled_items.append(compiled)
                            any_changed = any_changed or (compiled is not child)
                            _empty_cache()
                        if any_changed:
                            for i, item in enumerate(compiled_items):
                                with contextlib.suppress(Exception):
                                    cross[i] = item
                            self._eager_perceiver_cross = eager_items
                            compiled_any = True
                    self_blocks = getattr(perceiver, "self_blocks", None)
                    if isinstance(self_blocks, nn.ModuleList):
                        eager_items = list(self_blocks)
                        compiled_items = []
                        any_changed = False
                        for i, child in enumerate(eager_items):
                            compiled = _compile_one(
                                child, label=f"perceiver.self_blocks[{i}]"
                            )
                            compiled_items.append(compiled)
                            any_changed = any_changed or (compiled is not child)
                            _empty_cache()
                        if any_changed:
                            for i, item in enumerate(compiled_items):
                                with contextlib.suppress(Exception):
                                    self_blocks[i] = item
                            self._eager_perceiver_self_blocks = eager_items
                            compiled_any = True
                if not compiled_any:
                    compiled_p = _compile_one(perceiver, label="fuser.perceiver")
                    with contextlib.suppress(Exception):
                        self.fuser.perceiver = compiled_p

            with self._runtime_lock:
                self._enn_compile_active_mode = str(target_mode)
                set_runtime_cfg("compile_mode", str(target_mode))
                self._enn_compile_upgrade_done = True
                self._enn_compile_requested_mode = ""
            log.info(
                "[compile] upgraded compile_mode: warmup=reduce-overhead -> %s (after %d steps)",
                str(target_mode),
                int(after),
            )
            return True

        except BaseException as e:
            if is_oom_error(e) or ("out of memory" in str(e).lower()):
                _empty_cache()
                log.warning(
                    "[compile] upgrade aborted due to OOM; keeping reduce-overhead"
                )
            else:
                log.warning(
                    "[compile] upgrade aborted; keeping reduce-overhead (%s)", str(e)
                )
            with self._runtime_lock:
                self._enn_compile_upgrade_done = True
                self._enn_compile_requested_mode = ""
            return False
        finally:
            with self._runtime_lock:
                self._enn_compile_upgrade_inflight = False

    @property
    def config(self: Self) -> ModelConfig:
        return self.__config

    def to(self: Self, *args: Any, **kwargs: Any) -> "Model":
        out = super().to(*args, **kwargs)
        with contextlib.suppress(Exception):
            if isinstance(getattr(self, "logger", None), Recorder):
                self.logger.cpu()
        return cast(Model, out)

    def __getstate__(self: Self) -> dict[str, object]:
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

    def __setstate__(self: Self, state: dict[str, object]) -> None:
        super().__setstate__(state)
        self._runtime_lock = Mutex()
        self._amp_dtype_cache_lock = Mutex()
        self._amp_dtype_cache_use_lock = not bool(is_gil_enabled())
        self._amp_dtype_cache = {}
        self._amp_dtype_cache_last_key = None
        self._amp_dtype_cache_last_dtype = None
        if not hasattr(self, "_amp_dtype_cache_max"):
            self._amp_dtype_cache_max = 64



    def train(self: Self, mode: bool = True) -> "Model":
        self.training = bool(mode)
        for m in nn.Module.children(self):
            m.train(mode)
        return self

    def apply(self: Self, fn: Callable[[nn.Module], Any]) -> "Model":
        for m in nn.Module.children(self):
            m.apply(fn)
        fn(self)
        return self

    def _apply(
        self: Self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> "Model":
        recurse = kwargs.pop("recurse", True)
        if args:
            with contextlib.suppress(Exception):
                recurse = bool(args[0])
        if recurse:
            for m in nn.Module.children(self):
                with contextlib.suppress(TypeError):
                    m._apply(fn, recurse=recurse)
                    continue
                m._apply(fn)
        for key, param in list(self._parameters.items()):
            if param is None:
                continue
            with torch.no_grad():
                applied = fn(param)
            if applied is not param:
                self._parameters[key] = torch.nn.Parameter(
                    applied, requires_grad=param.requires_grad
                )
            if param.grad is not None:
                with torch.no_grad():
                    param.grad = fn(param.grad)
        for key, buf in list(self._buffers.items()):
            if buf is None:
                continue
            self._buffers[key] = fn(buf)
        return self

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
    def eager_for_export(self: Self) -> None:
        proc = getattr(self, "fuser", None)
        if proc is None:
            proc = getattr(self, "processor", None)
        if proc is None:
            yield self
            return
        swaps: list[tuple[object, str, Any]] = []
        list_swaps: list[tuple[nn.ModuleList, list[nn.Module]]] = []
        with contextlib.suppress(Exception):
            eager_perceiver = getattr(self, "_eager_fuser_perceiver", None)
            if isinstance(eager_perceiver, nn.Module):
                cur = getattr(proc, "perceiver", None)
                if cur is not None and cur is not eager_perceiver:
                    swaps.append((proc, "perceiver", cur))
                    proc.perceiver = eager_perceiver
        with contextlib.suppress(Exception):
            perceiver = getattr(proc, "perceiver", None)
            cross = (
                getattr(perceiver, "cross", None)
                if perceiver is not None
                else None
            )
            eager_cross = getattr(self, "_eager_perceiver_cross", None)
            if isinstance(cross, nn.ModuleList) and isinstance(
                eager_cross, list
            ):
                if len(eager_cross) == len(cross):
                    cur_items = list(cross)
                    if any(
                        cur_items[i] is not eager_cross[i]
                        for i in range(len(cross))
                    ):
                        list_swaps.append((cross, cur_items))
                        for i, item in enumerate(eager_cross):
                            cross[i] = item
        with contextlib.suppress(Exception):
            perceiver = getattr(proc, "perceiver", None)
            self_blocks = (
                getattr(perceiver, "self_blocks", None)
                if perceiver is not None
                else None
            )
            eager_self = getattr(self, "_eager_perceiver_self_blocks", None)
            if isinstance(self_blocks, nn.ModuleList) and isinstance(
                eager_self, list
            ):
                if len(eager_self) == len(self_blocks):
                    cur_items = list(self_blocks)
                    if any(
                        cur_items[i] is not eager_self[i]
                        for i in range(len(self_blocks))
                    ):
                        list_swaps.append((self_blocks, cur_items))
                        for i, item in enumerate(eager_self):
                            self_blocks[i] = item
        with contextlib.suppress(Exception):
            dc = getattr(self, "_decode_compiled", None)
            if isinstance(dc, nn.Module):
                swaps.append((self, "_decode_compiled", dc))
                self._decode_compiled = None
        try:
            yield self
        finally:
            for modlist, old_items in reversed(list_swaps):
                for i, item in enumerate(old_items):
                    with contextlib.suppress(Exception):
                        modlist[i] = item
            for target, name, old in swaps:
                with contextlib.suppress(Exception):
                    setattr(target, name, old)

    def _run_forward_core(
        self: Self,
        features: torch.Tensor,
        *args: Any,
        export: bool = False,
        temporal_state: object = None,
        causal_mask: Optional[torch.Tensor] = None,
        sanitize_nan: bool = True,
        calibrate_output: bool = True,
        device: Optional[torch.device] = None,
        base_dtype: Optional[torch.dtype] = None,
    ) -> tuple[
        torch.Tensor,
        object | None,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        diag_pre = bool((not export) and sanitize_nan and env_bool(
            "ENN_DIAG_NONFINITE_PRE_SANITIZE", default=False
        ))
        strict_pre = bool(diag_pre and env_bool("ENN_SANITIZE_NAN_STRICT", default=False))
        nonfinite_log: list[dict[str, object]] | None = None
        if diag_pre:
            nonfinite_log = []
            with contextlib.suppress(Exception):
                self._enn_nonfinite_pre_sanitize = nonfinite_log

        def _diag_nonfinite(tag: str, t: object) -> None:
            if not diag_pre or nonfinite_log is None:
                return
            if not torch.is_tensor(t):
                return
            tt: torch.Tensor = t
            if (not (tt.is_floating_point() or tt.is_complex())) or tt.numel() <= 0:
                return
            with torch.no_grad():
                fin = torch.isfinite(tt)
                nonfinite = int((~fin).sum().item())
                if nonfinite == 0:
                    return
                absmax = float(tt.detach().abs().amax().item())
                dtype = str(getattr(tt, "dtype", ""))
                shape = [int(x) for x in tt.shape]
                entry = {
                    "tag": str(tag),
                    "dtype": dtype,
                    "shape": shape,
                    "device": str(getattr(tt, "device", "")),
                    "nonfinite": nonfinite,
                    "absmax": absmax,
                }
                nonfinite_log.append(entry)
                dump_dir = str(env_str("ENN_NONFINITE_DUMP_DIR") or "").strip()
                if dump_dir:
                    dumped = int(getattr(self, "_enn_pre_sanitize_dumped", 0) or 0)
                    limit = int(env_int("ENN_PRE_SANITIZE_DUMP_LIMIT", default=4) or 4)
                    if (limit < 0) or (dumped < limit):
                        with contextlib.suppress(Exception):
                            import os as _os

                            all_ranks = env_bool("ENN_NONFINITE_DUMP_ALL_RANKS", default=False)
                            do_dump = True
                            if (not all_ranks) and int(_os.environ.get("LOCAL_RANK", "0") or 0) != 0:
                                do_dump = False
                            if do_dump:
                                _os.makedirs(dump_dir, exist_ok=True)
                                safe_tag = "".join(
                                    (c if (c.isalnum() or c in "._-") else "_") for c in str(tag)
                                )
                                rank = int(_os.environ.get("RANK", "0") or 0)
                                fname = f"pre_sanitize.{safe_tag}.rank{rank}.{uuid.uuid4().hex[:8]}.pt"
                                out_path = _os.path.join(dump_dir, fname)

                                samp = tt.detach()
                                if samp.numel() > 0:
                                    with contextlib.suppress(Exception):
                                        if samp.dim() >= 1:
                                            samp = samp[: min(2, int(samp.shape[0]))]
                                        if samp.dim() >= 2:
                                            samp = samp[:, : min(128, int(samp.shape[1]))]
                                        if samp.dim() >= 3:
                                            samp = samp[:, :, : min(128, int(samp.shape[2]))]
                                    with contextlib.suppress(Exception):
                                        samp = samp.contiguous()
                                    samp = samp.to("cpu")
                                payload = {
                                    "tag": str(tag),
                                    "entry": entry,
                                    "sample": samp,
                                }
                                torch.save(payload, out_path)
                                setattr(self, "_enn_pre_sanitize_dumped", dumped + 1)
                                _LOGGER.error(
                                    "[ENN] dumped pre-sanitize non-finite snapshot to: %s",
                                    str(out_path),
                                )
                if strict_pre:
                    raise RuntimeError(
                        f"[ENN] nonfinite detected before sanitize in {tag}: nonfinite={nonfinite} absmax={absmax} dtype={dtype} shape={shape}"
                    )

        x = self._cast_graph_safe(
            features, device or self._device, base_dtype or features.dtype
        )
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
            _diag_nonfinite("fuser.tokens", tokens)
            _diag_nonfinite("fuser.context", context)
            tokens = _coerce_tensor(tokens, enabled=True, inplace=not export)
            context = _coerce_tensor(context, enabled=True, inplace=not export)
        assembled = context.reshape(b, -1)
        if self.is_norm_linear and self.linear_branch is not None:
            assembled = assembled + self.linear_branch(
                self._cast_graph_safe(x, self._device, assembled.dtype)
            )
        mean_tok = tokens.mean(dim=1, keepdim=True)
        if (
            mean_tok.dim() == 3
            and tokens.dim() == 3
            and mean_tok.size(1) == 1
            and tokens.size(1) != 1
        ):
            mean_tok = mean_tok.expand(-1, tokens.size(1), -1)
        tokens_centered = (tokens - mean_tok).contiguous()
        if export:
            refined = self.temporal_token_collector.forward_export(
                tokens_centered
            )
        else:
            refined = self.temporal_token_collector.forward(tokens_centered)[0]
        if sanitize_nan:
            _diag_nonfinite("collector.refined", refined)
            refined = _coerce_tensor(refined, enabled=True, inplace=not export)
        residual = self.fuser.decode(refined, apply_norm=True)
        if sanitize_nan:
            _diag_nonfinite("fuser.residual", residual)
            residual = _coerce_tensor(
                residual, enabled=True, inplace=not export
            )
        enhanced = residual.reshape(b, -1).to(dtype=assembled.dtype)
        delta = enhanced - assembled
        p = None
        if self.delta_gate is not None:
            p = self.delta_gate(
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
                            float(getattr(self.delta_gate, "clip_eps", 1e-6)),
                            float(getattr(self.delta_gate, "eps", 0.0)),
                        ),
                        0.0,
                    ),
                    0.49,
                )
            )
            p = p.clamp(clip, 1.0 - clip)
            if (
                p.dim() == 2
                and delta.dim() == 2
                and p.size(1) == 1
                and p.size(0) == delta.size(0)
                and delta.size(1) != 1
            ):
                p = p.expand(-1, delta.size(1))
            y_hat = assembled + p * delta
        else:
            y_hat = assembled + delta * 0.5
        if sanitize_nan:
            y_hat = _coerce_tensor(y_hat, enabled=True, inplace=not export)
        if calibrate_output:
            y_hat = self.scaler.calibrate(y_hat)
        pred = self.scaler.denormalize_y(y_hat).reshape(b, *self.out_shape)
        if sanitize_nan:
            pred = _coerce_tensor(pred, enabled=True, inplace=not export)
        return pred, next_state, p, assembled, enhanced, delta, tokens, refined

    def forward_export(self: Self, features: torch.Tensor) -> torch.Tensor:
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
        self: Self,
        features: torch.Tensor,
        *args: Any,
        temporal_state: object = None,
        causal_mask: Optional[torch.Tensor] = None,
        calibrate_output: bool = True,
        sanitize_nan: bool = True,
    ) -> Tuple[torch.Tensor, Any]:
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
        self: Self,
        loss_val: torch.Tensor,
        y_hat: torch.Tensor,
        assembled: torch.Tensor,
        enhanced: torch.Tensor,
        tokens: torch.Tensor,
        p: torch.Tensor | None,
        z_true: torch.Tensor,
        features_t: torch.Tensor,
        is_cls_loss: bool,
        edge_reg_low: float,
        edge_reg_high: float,
    ) -> torch.Tensor:
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
                    F.smooth_l1_loss(
                        assembled, y_hat.detach(), reduction="none"
                    ).mean(dim=1)
                    * w
                ).mean()
            else:
                loss_yy = F.smooth_l1_loss(
                    assembled, y_hat.detach(), reduction="mean"
                )
            aux_total += self.unsup_yy_weight * loss_yy.to(aux_total.dtype)
            aux_used = True
        if self.p_prior_weight > 0.0 and p is not None:
            clip_eps = max(
                float(getattr(self.delta_gate, "clip_eps", 1e-6)),
                float(getattr(self.delta_gate, "eps", 0.0)),
            )
            p01 = p.squeeze(-1).clamp(min=clip_eps, max=1.0 - clip_eps)
            loss_p = -(
                ((self.p_prior_alpha - 1.0) * torch.log(p01))
                + ((self.p_prior_beta - 1.0) * torch.log1p(-p01))
            ).mean()
            aux_total += self.p_prior_weight * loss_p.to(aux_total.dtype)
            aux_used = True
        if p is not None and (
            self.delta_gate_budget_weight > 0.0
            or self.delta_gate_tv_weight > 0.0
            or self.delta_gate_teacher_weight > 0.0
        ):
            gate_eps = 1e-6
            p_floor = 0.0
            p_ceil = 1.0
            if self.delta_gate is not None:
                with contextlib.suppress(Exception):
                    gate_eps = float(
                        getattr(self.delta_gate, "clip_eps", gate_eps)
                    )
                with contextlib.suppress(Exception):
                    gate_eps = max(
                        gate_eps, float(getattr(self.delta_gate, "eps", 0.0))
                    )
                with contextlib.suppress(Exception):
                    p_floor = float(
                        getattr(self.delta_gate, "p_floor", p_floor)
                    )
                with contextlib.suppress(Exception):
                    p_ceil = float(getattr(self.delta_gate, "p_ceil", p_ceil))
            gate_eps = max(0.0, float(gate_eps))
            if self.delta_gate_budget_weight > 0.0:
                tgt = float(self.delta_gate_budget_target)
                if p_ceil >= p_floor:
                    tgt = max(p_floor, min(p_ceil, tgt))
                loss_budget = (
                    p.to(dtype=torch.float32).mean() - p.new_tensor(tgt)
                ).square()
                aux_total += self.delta_gate_budget_weight * loss_budget.to(
                    aux_total.dtype
                )
                aux_used = True
            p_grid = None
            w_grid = None
            tile_shape = None
            if (
                self.delta_gate_tv_weight > 0.0
                or self.delta_gate_teacher_weight > 0.0
            ) and self.out_shape:
                tile_shape = _normalize_tile_shape(
                    getattr(self, "delta_gate_tile_shape", None),
                    self.out_shape,
                )
            if (
                self.delta_gate_tv_weight > 0.0
                or self.delta_gate_teacher_weight > 0.0
            ) and self.out_shape:
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
                        p_grid = p.to(dtype=torch.float32).mean(
                            dim=1, keepdim=True
                        )
                except Exception:
                    p_grid = None
                    w_grid = None
            if self.delta_gate_tv_weight > 0.0 and p_grid is not None:
                tv = _tv_loss_grid(
                    p_grid, power=float(self.delta_gate_tv_power), eps=gate_eps
                )
                aux_total += self.delta_gate_tv_weight * tv.to(
                    dtype=aux_total.dtype
                )
                aux_used = True
            if (
                self.delta_gate_teacher_weight > 0.0
                and p_grid is not None
                and z_true is not None
                and (not is_cls_loss)
            ):
                base_det = assembled.detach().to(dtype=torch.float32)
                ref_det = enhanced.detach().to(dtype=torch.float32)
                tgt_det = z_true.detach().to(
                    device=base_det.device, dtype=torch.float32
                )
                err_base = (base_det - tgt_det).square()
                err_ref = (ref_det - tgt_det).square()
                improve = err_base - err_ref
                if bool(self.delta_gate_teacher_relu):
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
                temp = max(
                    float(self.delta_gate_teacher_temp), float(gate_eps)
                )
                score = (
                    (
                        imp_grid / scale_grid
                        - float(self.delta_gate_teacher_tau)
                    )
                    / temp
                ).clamp(min=-20.0, max=20.0)
                p01 = torch.sigmoid(score)
                p_target = p01 * float(p_ceil - p_floor) + float(p_floor)
                lo = float(p_floor) + float(gate_eps)
                hi = float(p_ceil) - float(gate_eps)
                if hi > lo:
                    p_target = p_target.clamp(min=lo, max=hi)
                if w_grid is not None:
                    w = w_grid.to(device=p_grid.device, dtype=torch.float32)
                    loss_teacher = (
                        (p_grid.to(dtype=torch.float32) - p_target).square()
                        * w
                    ).sum() / w.sum().clamp_min(float(gate_eps))
                else:
                    loss_teacher = F.mse_loss(
                        p_grid.to(dtype=torch.float32),
                        p_target,
                        reduction="mean",
                    )
                aux_total += self.delta_gate_teacher_weight * loss_teacher.to(
                    dtype=aux_total.dtype
                )
                aux_used = True
        if (
            self.delta_gate_edge_reg_weight_low > 0.0
            and edge_reg_low is not None
        ):
            aux_total += self.delta_gate_edge_reg_weight_low * edge_reg_low.to(
                dtype=aux_total.dtype
            )
            aux_used = True
        if (
            self.delta_gate_edge_reg_weight_high > 0.0
            and edge_reg_high is not None
        ):
            aux_total += (
                self.delta_gate_edge_reg_weight_high
                * edge_reg_high.to(dtype=aux_total.dtype)
            )
            aux_used = True
        if aux_used:
            return (
                (loss_val + aux_total) if (loss_val is not None) else aux_total
            )
        return loss_val

    def _auto_microbatch(
        self: Self,
        features: torch.Tensor | TensorDictBase,
        device: torch.device,
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
        hard_max = int(env_int("ENN_MICROBATCH_MAX", 64))
        hard_max = max(1, min(hard_max, b))
        per_sample = int(
            env_first_int(
                (
                    "ENN_PER_SAMPLE_MEM_BYTES",
                    "ENN_DEVICE_BYTES_PER_SAMPLE",
                ),
                default=0,
            )
            or 0
        )
        if per_sample <= 0:
            one = X[:1]
            bytes_per_sample = int(one.nelement()) * int(one.element_size())
            per_sample = int(bytes_per_sample * 8)
        stage_div = max(1, int(env_int("ENN_MICROBATCH_STAGE_DIV", 4)))
        per_sample = max(1, int(per_sample // stage_div))
        mb_size = _autofit_microbatch(
            device=device, hard_max=hard_max, per_sample_bytes=per_sample
        )
        return int(mb_size)

    def forward(
        self: Self,
        features: torch.Tensor | TensorDictBase,
        *args: Any,
        labels_flat: Optional[torch.Tensor] = None,
        net_loss: Optional[nn.Module] = None,
        global_loss: Optional[nn.Module] = None,
        local_loss: Optional[nn.Module] = None,
        loss_weights: Optional[
            Union[Tuple[float, float], LossWeightPolicy]
        ] = None,
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
        master_dtype: torch.dtype | None = None
        _req_base = getattr(self, "base_dtype", None) or getattr(self, "_base_dtype", None)
        if isinstance(_req_base, torch.dtype):
            master_dtype = _req_base
        else:
            with contextlib.suppress(Exception):
                master_dtype = next(self.parameters()).dtype
        if master_dtype is None:
            master_dtype = torch.get_default_dtype()
        infer_cuda = bool(
            infer_mode
            and (not self.training)
            and (getattr(device, "type", None) == "cuda")
        )
        pred_disable_cg = bool(
            infer_cuda
            and env_bool("ENN_PRED_DISABLE_CUDAGRAPHS", default=True)
        )
        compile_cg_enabled = bool(getattr(self, "_compile_cudagraphs", False))
        if (not compile_cg_enabled) and (not is_export_or_trace()):
            with contextlib.suppress(Exception):
                compile_cg_enabled = bool(
                    getattr(get_runtime_cfg(), "compile_cudagraphs", False)
                )
        cg_ok = bool(compile_cg_enabled and (not pred_disable_cg))
        x_raw = features
        if (
            isinstance(x_raw, torch.Tensor)
            and x_raw.ndim == 3
            and x_raw.shape[1] == 1
        ):
            x_raw = x_raw.reshape(x_raw.shape[0], -1)
        if isinstance(x_raw, torch.Tensor) and x_raw.ndim == 1:
            n = int(x_raw.numel())
            if n == int(self.in_dim):
                x_raw = x_raw.reshape(1, -1)
            elif int(self.in_dim) == 1 and n > 0:
                x_raw = x_raw.reshape(-1, 1)
            else:
                raise ValueError(
                    f"Expected features shaped (B, {self.in_dim}) (or ({self.in_dim},) for a single sample), "
                    f"got {tuple(x_raw.shape)}"
                )
        if isinstance(x_raw, torch.Tensor) and x_raw.device != device:
            x_raw = x_raw.to(device=device, non_blocking=True)
        x_scaled = self.scaler.normalize_x(x_raw)
        graph_break()
        meta = None
        try:
            meta = StatelessAutocast.coerce_metadata(device)
        except Exception:
            meta = None
            _LOGGER.debug(
                "StatelessAutocast.coerce_metadata failed; falling back to fp32",
                exc_info=True,
            )
        amp_candidates: Tuple[torch.dtype, ...] = ()
        if meta is not None:
            try:
                amp_candidates = tuple(getattr(meta, "float_dtypes", ()))
            except Exception:
                amp_candidates = ()
        if not amp_candidates:
            amp_candidates = (master_dtype,)
        safety_margin_pow2 = 3
        try:
            safety_margin_pow2 = int(
                getattr(self.__config, "safety_margin_pow2", 3)
            )
        except Exception:
            safety_margin_pow2 = 3
        safety_margin_pow2 = max(0, min(30, safety_margin_pow2))
        dev_index = (
            int(device.index)
            if getattr(device, "index", None) is not None
            else -1
        )
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
        if self._amp_dtype_cache_use_lock:
            with self._amp_dtype_cache_lock:
                if self._amp_dtype_cache_last_key == cache_key:
                    amp_dtype = self._amp_dtype_cache_last_dtype
                else:
                    amp_dtype = self._amp_dtype_cache.get(cache_key)
        else:
            last_key = self._amp_dtype_cache_last_key
            if last_key == cache_key:
                amp_dtype = self._amp_dtype_cache_last_dtype
            else:
                amp_dtype = self._amp_dtype_cache.get(cache_key)
        if amp_dtype is None:
            negotiated = StatelessAutocast.negotiate(
                tuple(amp_candidates),
                fallback=master_dtype,
                logger=_LOGGER,
                context="instance.forward",
                device=device,
                meta=meta,
                decision_key=cache_key,
                safety_margin_pow2=int(safety_margin_pow2),
            )
            if self._amp_dtype_cache_use_lock:
                with self._amp_dtype_cache_lock:
                    amp_dtype = self._amp_dtype_cache.get(cache_key)
                    if amp_dtype is None:
                        amp_dtype = negotiated
                        if len(self._amp_dtype_cache) >= int(
                            self._amp_dtype_cache_max
                        ):
                            self._amp_dtype_cache.clear()
                        self._amp_dtype_cache[cache_key] = amp_dtype
                    self._amp_dtype_cache_last_dtype = amp_dtype
                    self._amp_dtype_cache_last_key = cache_key
            else:
                amp_dtype = self._amp_dtype_cache.get(cache_key)
                if amp_dtype is None:
                    amp_dtype = negotiated
                    if len(self._amp_dtype_cache) >= int(
                        self._amp_dtype_cache_max
                    ):
                        self._amp_dtype_cache.clear()
                    self._amp_dtype_cache[cache_key] = amp_dtype
                self._amp_dtype_cache_last_dtype = amp_dtype
                self._amp_dtype_cache_last_key = cache_key
        else:
            if self._amp_dtype_cache_use_lock:
                with self._amp_dtype_cache_lock:
                    self._amp_dtype_cache_last_dtype = amp_dtype
                    self._amp_dtype_cache_last_key = cache_key
            else:
                self._amp_dtype_cache_last_dtype = amp_dtype
                self._amp_dtype_cache_last_key = cache_key
        amp_enabled = bool(
            isinstance(amp_dtype, torch.dtype)
            and amp_dtype is not torch.float64
        )
        is_cls_loss = (
            isinstance(net_loss, (nn.CrossEntropyLoss, nn.NLLLoss))
            if net_loss is not None
            else False
        )
        has_any_loss = (
            (net_loss is not None)
            or (global_loss is not None)
            or (local_loss is not None)
        )
        has_supervision = labels_flat is not None and has_any_loss
        is_train_path = bool(
            self.training and grad_enabled and has_supervision
        )
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
            requested_base = getattr(self, "base_dtype", None) or getattr(
                self, "_base_dtype", None
            )
            if requested_base is not None:
                base_dtype = requested_base
            else:
                try:
                    base_dtype = next(self.parameters()).dtype
                except Exception:
                    base_dtype = master_dtype
            if (
                isinstance(x_scaled, torch.Tensor)
                and x_scaled.device != device
            ):
                x_scaled = x_scaled.to(device=device, non_blocking=True)
            features_t = (
                x_scaled.to(dtype=base_dtype)
                if x_scaled.dtype != base_dtype
                else x_scaled
            )
            if features_t.ndim != 2 or features_t.shape[1] != self.in_dim:
                raise ValueError(
                    f"Expected features shaped (B, {self.in_dim}), got {tuple(features_t.shape)}"
                )
            exporting = bool(is_export_or_trace())

            b = features_t.shape[0] if exporting else int(features_t.shape[0])

            if exporting:
                mb = None
            else:
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
                            self.microbatch = max(
                                1, int(getattr(self, "microbatch", 64) or 64)
                            )
                with self._runtime_lock:
                    mb_cfg = (
                        int(self.microbatch) if self.microbatch else int(b)
                    )
                mb = max(1, min(int(b), mb_cfg))

            def _encode(
                inp: torch.Tensor,
            ) -> Tuple[torch.Tensor, Any]:
                with (
                    StatelessAutocast.float(device, metadata=meta)
                    if amp_enabled
                    else StatelessAutocast.suspend(device)
                ):
                    return self.fuser(inp)

            enc_ctx = (
                inference_mode(self.fuser)
                if infer_mode
                else contextlib.nullcontext()
            )
            with enc_ctx:
                if exporting:
                    tokens, context = cast(
                        Tuple[torch.Tensor, torch.Tensor], _encode(features_t)
                    )
                else:
                    tokens, context = cast(
                        Tuple[torch.Tensor, torch.Tensor],
                        _prealloc_microbatch(
                            features_t,
                            mb,
                            _encode,
                            pad_to=(
                                int(mb)
                                if (
                                    self._pad_compiled_microbatch
                                    and (not pred_disable_cg)
                                )
                                else None
                            ),
                            out_dtype=base_dtype,
                            cast_slice=lambda t: self._cast_graph_safe(
                                t, device, base_dtype
                            ),
                            stage="encoder",
                        ),
                    )
            tokens = _coerce_tensor(
                tokens, enabled=sanitize_enabled, inplace=sanitize_inplace
            )
            context = _coerce_tensor(
                context, enabled=sanitize_enabled, inplace=sanitize_inplace
            )
            if exporting:
                assembled = context.reshape(context.shape[0], -1)
            else:
                assembled = context.reshape(b, -1)
            if (not exporting) and int(tokens.shape[0]) != int(b):
                raise RuntimeError(
                    "Internal error: token batch mismatch after microbatch concat. "
                    f"got={int(tokens.shape[0])}, expected={int(b)}"
                )
            if self.is_norm_linear and self.linear_branch is not None:
                bl = self.linear_branch(
                    self._cast_graph_safe(features_t, device, assembled.dtype)
                )
                assembled = assembled + bl
            mean_dtype = (
                master_dtype if (tokens.is_floating_point() and tokens.dtype != master_dtype) else tokens.dtype
            )
            if mean_dtype != tokens.dtype:
                mean = tokens.to(dtype=mean_dtype).mean(dim=1, keepdim=True)
            else:
                mean = tokens.mean(dim=1, keepdim=True)
            mean_cast = symint_safe_expand_as(
                mean.to(dtype=tokens.dtype), tokens
            )
            tokens_centered = tokens - mean_cast
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
            ctrl_mb = max(
                1,
                min(
                    int(b),
                    int(self.temporal_token_collector.microbatch) or int(b),
                ),
            )
            graph_break()
            processor_ctx = (
                inference_mode(self.fuser)
                if infer_mode
                else contextlib.nullcontext()
            )
            with processor_ctx:
                dc = getattr(self, "_decode_compiled", None)

                def _run_decode_chunk(chunk: torch.Tensor) -> torch.Tensor:
                    with (
                        StatelessAutocast.float(device, metadata=meta)
                        if amp_enabled
                        else StatelessAutocast.suspend(device)
                    ):
                        if dc is not None:
                            if cg_ok and (getattr(device, "type", None) == "cuda"):
                                cudagraph_mark_step_begin()
                                out = cast(torch.Tensor, dc(chunk))
                                cudagraph_mark_step_end()
                                return out
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
                            if (
                                dc is not None
                                and self._pad_compiled_microbatch
                                and (not pred_disable_cg)
                            )
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
                try:
                    y_master_float = PrecisionPolicy.from_metadata(
                        device=device, metadata=meta, logger=_LOGGER
                    ).master_float
                except Exception:
                    y_master_float = torch.float64

                if not y_true_raw.is_floating_point():
                    y_true_raw = y_true_raw.to(dtype=y_master_float)
                else:
                    try:
                        if (
                            torch.finfo(y_true_raw.dtype).bits
                            < torch.finfo(y_master_float).bits
                        ):
                            y_true_raw = y_true_raw.to(dtype=y_master_float)
                        elif (
                            y_true_raw.dtype != y_master_float
                            and y_master_float is torch.float64
                        ):
                            y_true_raw = y_true_raw.to(dtype=y_master_float)
                    except Exception:
                        y_true_raw = y_true_raw.to(dtype=y_master_float)
                z_true = self.scaler.normalize_y(y_true_raw).to(
                    device=assembled.device, dtype=assembled.dtype
                )
            p: Optional[torch.Tensor] = None
            edge_reg_low: Optional[torch.Tensor] = None
            edge_reg_high: Optional[torch.Tensor] = None
            if self.delta_gate is not None:
                z_min: Optional[torch.Tensor] = None
                z_max: Optional[torch.Tensor] = None
                fallback_bounds = False
                try:
                    y_min = getattr(self.scaler, "y_min", None)
                    y_max = getattr(self.scaler, "y_max", None)
                    y_q_low = getattr(self.scaler, "y_q_low", None)
                    y_q_high = getattr(self.scaler, "y_q_high", None)
                    mean = self.scaler.y_mean.to(
                        device=assembled.device, dtype=assembled.dtype
                    )
                    std = self.scaler.y_std.to(
                        device=assembled.device, dtype=assembled.dtype
                    )
                    denom = std + float(self.scaler.eps)

                    def _finite_pair(lo: object, hi: object) -> bool:
                        if not (
                            isinstance(lo, torch.Tensor)
                            and isinstance(hi, torch.Tensor)
                        ):
                            return False
                        if (
                            is_symbolic()
                            or is_export_or_trace()
                            or is_meta_or_fake_tensor(lo)
                            or is_meta_or_fake_tensor(hi)
                        ):
                            return True
                        try:
                            return bool(
                                torch.isfinite(lo).all().item()
                            ) and bool(torch.isfinite(hi).all().item())
                        except Exception:
                            return False

                    have_minmax = _finite_pair(y_min, y_max)
                    have_quant = _finite_pair(y_q_low, y_q_high)
                    use_quant = bool(
                        getattr(self, "delta_gate_bounds_use_quantile", False)
                    )
                    clip_quant = bool(
                        getattr(self, "delta_gate_bounds_clip_to_minmax", True)
                    )
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
                        if bool(
                            getattr(self, "delta_gate_fallback_enabled", False)
                        ):
                            k_low_buf = getattr(
                                self, "delta_gate_fallback_k_low_buf", None
                            )
                            k_high_buf = getattr(
                                self, "delta_gate_fallback_k_high_buf", None
                            )
                            if isinstance(k_low_buf, torch.Tensor):
                                k_low = k_low_buf.to(
                                    device=assembled.device,
                                    dtype=assembled.dtype,
                                )
                            else:
                                k_low = mean.new_tensor(
                                    float(
                                        getattr(
                                            self,
                                            "delta_gate_fallback_k_low",
                                            getattr(
                                                self,
                                                "delta_gate_fallback_k",
                                                0.0,
                                            ),
                                        )
                                    )
                                )
                            if isinstance(k_high_buf, torch.Tensor):
                                k_high = k_high_buf.to(
                                    device=assembled.device,
                                    dtype=assembled.dtype,
                                )
                            else:
                                k_high = mean.new_tensor(
                                    float(
                                        getattr(
                                            self,
                                            "delta_gate_fallback_k_high",
                                            getattr(
                                                self,
                                                "delta_gate_fallback_k",
                                                0.0,
                                            ),
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
                        (self.delta_gate_edge_reg_weight_low > 0.0)
                        or (self.delta_gate_edge_reg_weight_high > 0.0)
                    )
                    and (z_min is not None)
                    and (z_max is not None)
                    and (
                        (not self.delta_gate_edge_reg_fallback_only)
                        or bool(fallback_bounds)
                    )
                )
                if do_edge_reg:
                    p, edge_reg_low, edge_reg_high = self.delta_gate(
                        tokens=tokens,
                        refined_tokens=refined_tokens,
                        base=assembled,
                        residue=delta,
                        z_min=z_min,
                        z_max=z_max,
                        fallback_bounds=bool(fallback_bounds),
                        return_edge_reg_lr=True,
                        edge_reg_frac=float(self.delta_gate_edge_reg_frac),
                        edge_reg_min_width_frac=float(
                            self.delta_gate_edge_reg_min_width_frac
                        ),
                        edge_reg_power=float(self.delta_gate_edge_reg_power),
                    )
                else:
                    p = self.delta_gate(
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
                    float(getattr(self.delta_gate, "clip_eps", 1e-6))
                    if self.delta_gate is not None
                    else 1e-6
                )
                with contextlib.suppress(Exception):
                    p_eps = max(
                        p_eps, float(getattr(self.delta_gate, "eps", 0.0))
                    )
                hi = 1.0 - p_eps
                if hi > p_eps:
                    p = p.clamp(min=p_eps, max=hi)
            if p is None:
                y_hat = assembled + delta * 0.5
            else:
                y_hat = assembled + p * delta
            y_hat = _coerce_tensor(
                y_hat, enabled=sanitize_enabled, inplace=sanitize_inplace
            )
            pred = y_hat.reshape(b, *self.out_shape)
            loss_val: Optional[torch.Tensor] = None
            top_component: Optional[torch.Tensor] = None
            bottom_component: Optional[torch.Tensor] = None
            use_global_local = labels_flat is not None and (
                global_loss is not None or local_loss is not None
            )
            use_net = (
                labels_flat is not None
                and (net_loss is not None)
                and (not use_global_local)
            )
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
                    raise RuntimeError(
                        "Internal error: z_true missing for regression loss path"
                    )
                tgt = z_true.to(device=y_hat.device, dtype=y_hat.dtype)
                total = y_hat.new_tensor(0.0, dtype=y_hat.dtype)
                y_bot = assembled.to(device=y_hat.device, dtype=y_hat.dtype)
                if global_loss is not None:
                    use_base_detach = bool(
                        is_train_path
                        and (local_loss is not None)
                        and (float(weights[1]) > 1e-12)
                    )
                    if use_base_detach:
                        base_det = assembled.detach()
                        delta_det = enhanced - base_det
                        z_top = _coerce_tensor(
                            base_det
                            + (
                                base_det.new_tensor(0.5) * delta_det
                                if p is None
                                else p * delta_det
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
                    top_component = cast(
                        torch.Tensor, global_loss(z_top, z_true)
                    )
                    total = total + weights[0] * top_component
                if local_loss is not None:
                    bottom_component = cast(
                        torch.Tensor, local_loss(y_bot, tgt)
                    )
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
                loss_val = torch.as_tensor(
                    loss_val, device=y_hat.device, dtype=y_hat.dtype
                )
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
                pred = self.scaler.denormalize_y(z_cal).reshape(
                    b, *self.out_shape
                )
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
                        out_td.set("delta_gate", p)
                else:
                    with contextlib.suppress(KeyError):
                        out_td.del_("refined_tokens")
                    with contextlib.suppress(KeyError):
                        out_td.del_("residual_context")
                    with contextlib.suppress(KeyError):
                        out_td.del_("delta_gate")
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
                    if (
                        isinstance(bottom_td, torch.Tensor)
                        and bottom_td.ndim == 0
                    ):
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

    def load_state_dict(self, state_dict, *args, **kwargs):
        def _ensure_scaler_shapes(model, sd):
            if not isinstance(sd, dict):
                return

            def _model_device(mod: torch.nn.Module) -> torch.device:
                with contextlib.suppress(StopIteration):
                    return next(mod.parameters()).device
                with contextlib.suppress(StopIteration):
                    return next(mod.buffers()).device
                return torch.device("cpu")

            def _alloc_replacement(
                *args: Any,
                existing: Optional[torch.Tensor],
                shape: torch.Size,
                fallback_mod: torch.nn.Module,
                dtype: Optional[torch.dtype] = None,
            ) -> torch.Tensor:
                if torch.is_tensor(existing):
                    dev = existing.device
                    dt = existing.dtype if dtype is None else dtype
                else:
                    dev = _model_device(fallback_mod)
                    dt = torch.float32 if dtype is None else dtype
                return torch.empty(shape, device=dev, dtype=dt)

            def _is_scaler_key(k: str) -> bool:
                return k.startswith("scaler.") or ".scaler." in k

            for full_key, val in sd.items():
                if not isinstance(full_key, str) or not _is_scaler_key(
                    full_key
                ):
                    continue
                if not torch.is_tensor(val):
                    continue
                parts = full_key.split(".")
                mod = model
                ok = True
                for p in parts[:-1]:
                    if not hasattr(mod, p):
                        ok = False
                        break
                    mod = getattr(mod, p)
                    if not isinstance(mod, torch.nn.Module):
                        ok = False
                        break
                if not ok:
                    continue
                name = parts[-1]
                tgt = val.detach()
                if hasattr(mod, "_buffers") and name in getattr(
                    mod, "_buffers", {}
                ):
                    buf = mod._buffers.get(name)
                    if torch.is_tensor(buf) and tuple(buf.shape) != tuple(
                        tgt.shape
                    ):
                        mod._buffers[name] = _alloc_replacement(
                            existing=buf,
                            shape=tgt.shape,
                            fallback_mod=mod,
                            dtype=buf.dtype,
                        )
                        setattr(mod, name, mod._buffers[name])
                    continue
                if hasattr(mod, "_parameters") and name in getattr(
                    mod, "_parameters", {}
                ):
                    prm = mod._parameters.get(name)
                    if (
                        prm is not None
                        and torch.is_tensor(prm)
                        and tuple(prm.shape) != tuple(tgt.shape)
                    ):
                        mod._parameters[name] = torch.nn.Parameter(
                            _alloc_replacement(
                                existing=prm,
                                shape=tgt.shape,
                                fallback_mod=mod,
                                dtype=prm.dtype,
                            ),
                            requires_grad=prm.requires_grad,
                        )
                        setattr(mod, name, mod._parameters[name])
                    continue

        try:
            _ensure_scaler_shapes(self, state_dict)
        except Exception:
            pass

        try:
            remap = getattr(
                self.fuser, "remap_legacy_task_ids_in_state_dict", None
            )
            if callable(remap):
                state_dict = remap(state_dict)
        except Exception:
            pass
        return super().load_state_dict(state_dict, *args, **kwargs)

    def list_tasks(self: Self, *args: Any, by: str = "name") -> list[str]:
        del args
        return self.fuser.get_subtree(by=by)

    def list(self: Self, *args: Any, by: str = "name") -> list[str]:
        del args
        return self.fuser.get_subtree(by=by)

    def get(
        self: Self, task_spec: str, default: Optional[nn.Module] = None
    ) -> Optional[nn.Module]:
        return self.fuser.get(task_spec, default=default)

    def node_name(self: Self, node_spec: Optional[object] = None) -> str:
        return self.fuser.node_name(node_spec)

    def get_children(
        self: Self, node_spec: Optional[object] = None
    ) -> list[nn.Module]:
        return self.fuser.get_children(node_spec)

    def get_parent(
        self: Self, node_spec: Optional[object] = None
    ) -> Optional[nn.Module]:
        return self.fuser.get_parent(node_spec)

    def get_subtree(
        self: Self,
        node_spec: Optional[object] = None,
        *args: Any,
        by: str = "name",
        include_self: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        del args, kwargs
        return self.fuser.get_subtree(node_spec, by=by, include_self=include_self)

    def attach(self: Self, node_spec: str, parent: Optional[str] = None) -> str:
        return self.fuser.attach(node_spec, parent=parent)

    def detach(self: Self, node_spec: str) -> str:
        return self.fuser.detach(node_spec)

    def new(self: Self, name: Optional[str] = None, *args: Any, **kwargs: Any) -> str:
        return self.fuser.new(name, *args, **kwargs)

    def update(self: Self, node_spec: str, *args: Any, **kwargs: Any) -> str:
        return self.fuser.update(node_spec, *args, **kwargs)

    def remove(self: Self, node_spec: str, *args: Any, **kwargs: Any) -> None:
        self.fuser.remove(node_spec, *args, **kwargs)

    def spec(self: Self) -> list[dict[str, Any]]:
        return self.fuser.spec()

    def graph(self: Self) -> dict[str, Any]:
        return self.fuser.graph()

    def resolve_task_id(self: Self, task_id_or_name: str) -> str:
        return self.fuser.resolve_task_id(task_id_or_name)

    def get_task_name(self: Self, task_id: str) -> str:
        return self.fuser.get_task_name(task_id)

    def get_submodel(self: Self, task_id_or_name: str) -> Optional[nn.Module]:
        return self.fuser.get_submodel(task_id_or_name)

    def task_specs(self: Self) -> list[dict[str, Any]]:
        return self.fuser.task_specs()

    def rebuild_tasks_from_specs(
        self: Self, specs: Sequence[Mapping[str, Any]]
    ) -> None:
        self.fuser.rebuild_tasks_from_specs(specs)

    def add_taskset(
        self: Self,
        name: Optional[str] = None,
        *args: Any,
        children: Optional[Sequence[str]] = None,
        parent: Optional[str] = None,
        reduction: str = "mean",
        description: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        weight: float = 1.0,
        eps: float = 1e-6,
        tokens: Optional[int] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        return self.fuser.add_taskset(
            name,
            children=children,
            parent=parent,
            reduction=reduction,
            description=description,
            tags=tags,
            weight=weight,
            eps=eps,
            tokens=tokens,
            task_id=task_id,
            *args,
            **kwargs,
        )

    def update_taskset(
        self: Self,
        taskset_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        return self.fuser.update_taskset(taskset_name, *args, **kwargs)

    def remove_taskset(
        self: Self,
        taskset_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.fuser.remove_taskset(taskset_name, *args, **kwargs)

    def add_task(
        self: Self,
        name: Optional[str] = None,
        *args: Any,
        mode: str = "spatial",
        description: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        weight: float = 1.0,
        eps: float = 1e-6,
        submodel: Optional[nn.Module] = None,
        tokens: Optional[int] = None,
        depth: Optional[int] = None,
        task_id: Optional[str] = None,
    ) -> str:
        return self.fuser.add_task(
            name,
            mode=mode,
            description=description,
            tags=tags,
            tokens=tokens,
            depth=depth,
            weight=weight,
            eps=eps,
            submodel=submodel,
            task_id=task_id,
        )

    def update_task(
        self: Self,
        task_name: str,
        *args: Any,
        mode: object = _META_UNSET,
        name: object = _META_UNSET,
        description: object = _META_UNSET,
        tags: object = _META_UNSET,
        submodel: Union[torch.nn.Module, None, object] = _SUBMODEL_UNSET,
        weight: object = _META_UNSET,
        eps: object = _META_UNSET,
    ) -> str:
        return self.fuser.update_task(
            task_name,
            mode=mode,
            name=name,
            description=description,
            tags=tags,
            submodel=submodel,
            weight=weight,
            eps=eps,
        )

    def remove_task(
        self: Self, task_name: str, *args: Any, strict: bool = False
    ) -> None:
        self.fuser.remove_task(task_name, strict=strict)

    def predict(
        self: Self,
        features: torch.Tensor | TensorDictBase,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | TensorDictBase:
        kwargs.setdefault("return_loss", False)
        return self.forward(features, *args, **kwargs)

    def history(self: Self) -> Sequence[Mapping[str, Any]]:
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
