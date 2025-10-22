# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import logging
import os
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

__all__ = [
    "FLOP_PROFILER",
    "FlopCounter",
    "attention_flops_bshd",
    "estimate_attention_flops",
]


def _infer_linear_mkn(
    inp: torch.Tensor, weight: Optional[torch.Tensor]
) -> Tuple[int, int, int]:
    if weight is not None and weight.ndim >= 2:
        k_dim = int(weight.shape[-1])
        n_dim = int(weight.shape[0])
    else:
        k_dim = int(inp.shape[-1])
        n_dim = int(getattr(inp, "shape", [0])[-1])
    m_dim = int(inp.numel() // max(k_dim, 1))
    return (m_dim, k_dim, n_dim)


def _compute_linear_flops(
    inp: torch.Tensor,
    out: Any,
    weight: Optional[torch.Tensor],
    include_bias: bool,
    effective_bwd: float,
) -> float:
    m_dim, k_dim, n_dim = _infer_linear_mkn(inp, weight)
    if isinstance(out, torch.Tensor) and out.numel() > 0 and n_dim > 0:
        m_dim = max(m_dim, int(out.numel() // max(n_dim, 1)))
    if m_dim <= 0 or k_dim <= 0 or n_dim <= 0:
        return 0.0
    bias_cost = m_dim * n_dim if include_bias else 0.0
    fwd = 2.0 * m_dim * k_dim * n_dim + bias_cost
    return float(fwd * (1.0 + max(0.0, float(effective_bwd))))


def _compute_conv_flops(
    inp: torch.Tensor,
    out: Any,
    weight: Optional[torch.Tensor],
    groups: int,
    effective_bwd: float,
) -> float:
    if weight is None or weight.ndim < 3:
        return 0.0
    try:
        out_elems = int(out.numel()) if isinstance(out, torch.Tensor) else 0
        groups = max(1, int(groups))
        if isinstance(inp, torch.Tensor) and inp.ndim >= 2:
            cin_total = int(inp.shape[1])
        else:
            cin_total = int(weight.shape[1] * groups)
        cin_per_group = max(1, cin_total // groups)
        kernel = int(weight[0].numel()) // max(cin_per_group, 1)
        return float(
            out_elems
            * (2.0 * cin_per_group * kernel)
            * (1.0 + max(0.0, float(effective_bwd)))
        )
    except Exception:
        return 0.0


def _linear_forward_hook(
    mod: nn.Module,
    inp: Tuple[Any, ...],
    out: Any,
    *,
    profiler: "_FlopProfiler",
    include_bias: bool,
    effective_bwd: float,
) -> None:
    weight = getattr(mod, "weight", None)
    if weight is None:
        inner_linear = getattr(mod, "linear", None)
        weight = getattr(inner_linear, "weight", None)
    x = inp[0] if inp else None
    if not isinstance(x, torch.Tensor) or weight is None:
        return
    val = _compute_linear_flops(x, out, weight, include_bias, effective_bwd)
    if val > 0.0:
        profiler.add_manual(type(mod).__name__, val)


def _conv_forward_hook(
    mod: nn.Module,
    inp: Tuple[Any, ...],
    out: Any,
    *,
    profiler: "_FlopProfiler",
    effective_bwd: float,
) -> None:
    weight = getattr(mod, "weight", None)
    x = inp[0] if inp else None
    if not isinstance(x, torch.Tensor) or weight is None:
        return
    groups = getattr(mod, "groups", 1)
    val = _compute_conv_flops(x, out, weight, groups, effective_bwd)
    if val > 0.0:
        profiler.add_manual(type(mod).__name__, val)


def estimate_attention_flops(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    *,
    bwd_factor: float,
    dropout_p: float,
    training: bool,
    include_softmax_scale_dropout: bool,
) -> float:
    if batch <= 0 or seq_len <= 0 or num_heads <= 0 or head_dim <= 0:
        return 0.0
    matmul = 4.0 * batch * num_heads * seq_len**2 * head_dim
    misc = 0.0
    if include_softmax_scale_dropout:
        misc_coeff = 6.0
        if training and dropout_p > 0.0:
            misc_coeff += 1.0
        misc = misc_coeff * (batch * num_heads * seq_len**2)
    fwd = matmul + misc
    return float(fwd * (1.0 + max(0.0, float(bwd_factor))))


def _attention_forward_hook(
    mod: nn.Module,
    inp: Tuple[Any, ...],
    out: Any,
    *,
    profiler: "_FlopProfiler",
    metadata: Dict[str, Any],
) -> None:
    if not inp:
        return
    q = inp[0]
    if not isinstance(q, torch.Tensor):
        return
    fmt = str(metadata.get("format", "bshd")).lower()
    try:
        if fmt == "bshd":
            batch = int(q.shape[0])
            seq_len = int(q.shape[1])
            num_heads = int(metadata.get("num_heads", q.shape[2]))
            head_dim = int(metadata.get("head_dim", q.shape[3]))
        elif fmt == "sbd":
            seq_len = int(q.shape[0])
            batch = int(q.shape[1])
            embed_dim = int(q.shape[2])
            num_heads = int(
                metadata.get(
                    "num_heads",
                    getattr(mod, "num_attention_heads", getattr(mod, "num_heads", 0)),
                )
            )
            if num_heads <= 0:
                return
            head_dim = int(metadata.get("head_dim", embed_dim // max(num_heads, 1)))
        else:
            return
    except Exception:
        return
    dropout_attr = metadata.get("dropout_attr")
    dropout_default = float(metadata.get("dropout", 0.0))
    if dropout_attr and hasattr(mod, dropout_attr):
        try:
            dropout_p = float(getattr(mod, dropout_attr))
        except Exception:
            dropout_p = dropout_default
    else:
        dropout_p = dropout_default
    include_softmax = bool(metadata.get("include_softmax_scale_dropout", True))
    bwd_factor = metadata.get("bwd_factor")
    if bwd_factor is None:
        bwd_factor_val = 2.0 if mod.training else 0.0
    else:
        try:
            bwd_factor_val = float(bwd_factor)
        except Exception:
            bwd_factor_val = 2.0 if mod.training else 0.0
    total = estimate_attention_flops(
        batch,
        seq_len,
        num_heads,
        head_dim,
        bwd_factor=bwd_factor_val,
        dropout_p=float(dropout_p),
        training=bool(mod.training),
        include_softmax_scale_dropout=include_softmax,
    )
    if total > 0.0:
        profiler.add_manual("Attention", total)


_LOGGER = logging.getLogger(__name__)


class _FlopProfiler:
    def __init__(self) -> None:
        self._manual_total = 0.0
        self._manual_by_type: Dict[str, float] = {}
        self._tracking_depth = 0
        self._nvtx_soft_counter: float = 0.0
        self._nvtx_getter: Optional[Callable[[], float]] = None

    def is_tracking_active(self) -> bool:
        return self._tracking_depth > 0

    def activate(self) -> None:
        self._tracking_depth += 1

    def deactivate(self) -> None:
        self._tracking_depth = max(0, self._tracking_depth - 1)

    def reset_manual(self) -> None:
        self._manual_total = 0.0
        self._manual_by_type.clear()

    def consume_manual_breakdown(self) -> Tuple[float, Dict[str, float]]:
        total = float(self._manual_total)
        breakdown = {k: float(v) for k, v in self._manual_by_type.items()}
        self.reset_manual()
        return (total, breakdown)

    def consume_manual(self) -> float:
        total, _ = self.consume_manual_breakdown()
        return total

    def add_manual(self, typ: str, value: float) -> None:
        if self.is_tracking_active():
            try:
                fv = float(value)
            except Exception:
                return
            if fv <= 0:
                return
            self._manual_total += fv
            self._manual_by_type[typ] = self._manual_by_type.get(typ, 0.0) + fv

    def _nvtx_soft_add(self, value: float) -> None:
        try:
            self._nvtx_soft_counter += float(value) if float(value) > 0 else 0.0
        except Exception:
            pass

    def _nvtx_soft_getter(self) -> float:
        return float(self._nvtx_soft_counter)

    def ensure_nvtx_getter(self) -> None:
        if self._nvtx_getter is not None:
            return
        hook = os.environ.get("STF_NVTX_FLOPS_FN", "").strip()
        if not hook:
            self._nvtx_getter = self._nvtx_soft_getter
            return
        try:
            module_name, attr = hook.split(":", 1)
        except ValueError:
            self._nvtx_getter = self._nvtx_soft_getter
            return
        try:
            module = __import__(module_name, fromlist=[attr])
            getter = getattr(module, attr)
            if callable(getter):
                self._nvtx_getter = getter
                return
        except Exception as exc:
            _LOGGER.debug("Failed to import NVTX getter %s: %s", hook, exc)
        self._nvtx_getter = self._nvtx_soft_getter

    def record_nvtx_soft(self, value: float) -> None:
        self._nvtx_soft_add(value)

    def make_nvtx_counter(self, device: Optional[torch.device]) -> Any:
        self.ensure_nvtx_getter()
        getter = self._nvtx_getter or self._nvtx_soft_getter
        try:
            import torch.cuda.nvtx as nvtx  # noqa: F401
        except Exception:
            return contextlib.nullcontext()

        class _NvtxScope:
            def __init__(self, device: Optional[torch.device]) -> None:
                self.device = device
                self._entered = False

            def __enter__(self) -> "_NvtxScope":
                if self.device is not None and self.device.type == "cuda":
                    try:
                        torch.cuda.synchronize(self.device)
                    except Exception:
                        pass
                self._entered = True
                return self

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
                self._entered = False
                return False

            def get_total_flops(self) -> float:
                try:
                    return float(getter())
                except Exception:
                    return 0.0

        return _NvtxScope(device)

    def _torch_counter(self, display: bool) -> Any:
        try:
            from torch.profiler import profile

            class _TorchScope:
                def __init__(self, display: bool) -> None:
                    self.display = display
                    self._prof = None

                def __enter__(self) -> "_TorchScope":
                    self._prof = profile(with_flops=True, record_shapes=False)
                    self._prof.__enter__()
                    return self

                def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
                    if self._prof is not None:
                        self._prof.__exit__(exc_type, exc, tb)
                        if self.display:
                            try:
                                self._prof.key_averages().table(sort_by="flops")
                            except Exception:
                                pass
                    return False

                def get_total_flops(self) -> float:
                    if self._prof is None:
                        return 0.0
                    try:
                        events = self._prof.key_averages()
                        return float(sum(e.flops for e in events if hasattr(e, "flops")))
                    except Exception:
                        return 0.0

            return _TorchScope(display)
        except Exception:
            return contextlib.nullcontext()

    def start_hooks(
        self,
        model: nn.Module,
        *,
        mode: str,
        include_bias: bool,
        bwd_factor: Optional[float],
    ) -> List[Any]:
        handles: List[Any] = []
        effective_bwd = 2.0 if mode == "train" else 0.0
        if bwd_factor is not None:
            try:
                effective_bwd = float(bwd_factor)
            except Exception:
                pass
        modules: Iterable[nn.Module] = tuple(model.modules())
        for module in modules:
            hook: Optional[Any] = None
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(
                    partial(
                        _linear_forward_hook,
                        profiler=self,
                        include_bias=include_bias,
                        effective_bwd=effective_bwd,
                    )
                )
            elif isinstance(module, nn.modules.conv._ConvNd):
                hook = module.register_forward_hook(
                    partial(
                        _conv_forward_hook,
                        profiler=self,
                        effective_bwd=effective_bwd,
                    )
                )
            if hook is not None:
                handles.append(hook)
        return handles

    def stop_hooks(self, handles: Sequence[Any]) -> None:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass

    def step_scope(self, device: Optional[torch.device], *, display: bool = False) -> Any:
        instrumentation = self
        self.activate()

        class _StepScope:
            def __init__(self) -> None:
                self.manual_total = 0.0
                self.manual_breakdown: Dict[str, float] = {}
                self.torch_total = 0.0
                self.nvtx_total = 0.0
                self.total = 0.0
                self._torch_scope: Any = None
                self._nvtx_scope: Any = None

            def __enter__(self) -> "_StepScope":
                instrumentation.reset_manual()
                self._torch_scope = instrumentation._torch_counter(display)
                self._nvtx_scope = instrumentation.make_nvtx_counter(device)
                self._torch_scope.__enter__()
                self._nvtx_scope.__enter__()
                return self

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
                manual, breakdown = instrumentation.consume_manual_breakdown()
                if manual > 0.0:
                    instrumentation.record_nvtx_soft(manual)
                self.manual_total = manual
                self.manual_breakdown = breakdown
                if self._torch_scope is not None:
                    self._torch_scope.__exit__(exc_type, exc, tb)
                    try:
                        self.torch_total = float(self._torch_scope.get_total_flops())
                    except Exception:
                        self.torch_total = 0.0
                if self._nvtx_scope is not None:
                    self._nvtx_scope.__exit__(exc_type, exc, tb)
                    try:
                        self.nvtx_total = float(self._nvtx_scope.get_total_flops())
                    except Exception:
                        self.nvtx_total = 0.0
                self.total = max(self.manual_total, self.torch_total, self.nvtx_total)
                instrumentation.deactivate()
                return False

            def get_total_flops(self) -> float:
                return float(self.total)

            def get_manual_breakdown(self) -> Dict[str, float]:
                return dict(self.manual_breakdown)

        return _StepScope()

    def attention_flops_bshd(
        self,
        q: torch.Tensor,
        *args: Any,
        bwd_factor: float = 2.0,
        dropout_p: float = 0.0,
        training: bool = False,
        include_softmax_scale_dropout: bool = True,
        **kwargs: Any,
    ) -> float:
        try:
            batch = int(q.shape[0])
            seq_len = int(q.shape[1])
            num_heads = int(q.shape[2])
            head_dim = int(q.shape[3])
        except Exception:
            return 0.0
        total = estimate_attention_flops(
            batch,
            seq_len,
            num_heads,
            head_dim,
            bwd_factor=bwd_factor,
            dropout_p=float(dropout_p),
            training=bool(training),
            include_softmax_scale_dropout=include_softmax_scale_dropout,
        )
        if total > 0.0:
            self.add_manual("Attention", total)
        return total


FLOP_PROFILER = _FlopProfiler()


class FlopCounter:
    def __init__(
        self,
        model: nn.Module,
        *args: Any,
        mode: str = "train",
        device: Optional[torch.device] = None,
        include_bias: bool = True,
        bwd_factor: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._mode = mode
        self._device = device
        self._include_bias = include_bias
        self._bwd_factor = bwd_factor
        self._handles: List[Any] = []
        self._hook_count = 0
        self._active = False

    @property
    def device(self) -> Optional[torch.device]:
        return self._device

    def __enter__(self) -> "FlopCounter":
        self._handles = FLOP_PROFILER.start_hooks(
            self._model,
            mode=self._mode,
            bwd_factor=self._bwd_factor,
            include_bias=self._include_bias,
        )
        self._hook_count = len(self._handles)
        FLOP_PROFILER.reset_manual()
        self._active = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._active:
            FLOP_PROFILER.stop_hooks(self._handles)
            self._handles = []
            self._active = False
        return False

    def step(self, *args: Any, display: bool = False, **kwargs: Any) -> Any:
        if not self._active:
            raise RuntimeError(
                "FlopCounter hooks are not active. Use `with FlopCounter(...)` before measuring FLOPs."
            )
        return FLOP_PROFILER.step_scope(self._device, display=display)

    @property
    def hook_count(self) -> int:
        return int(self._hook_count)


def attention_flops_bshd(
    q: torch.Tensor,
    *args: Any,
    bwd_factor: float = 2.0,
    dropout_p: float = 0.0,
    training: bool = False,
    include_softmax_scale_dropout: bool = True,
    **kwargs: Any,
) -> float:
    return FLOP_PROFILER.attention_flops_bshd(
        q,
        *args,
        bwd_factor=bwd_factor,
        dropout_p=dropout_p,
        training=training,
        include_softmax_scale_dropout=include_softmax_scale_dropout,
        **kwargs,
    )

