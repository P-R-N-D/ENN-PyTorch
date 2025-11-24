# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import logging
import os
from functools import partial
from types import TracebackType
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn


_LOGGER = logging.getLogger(__name__)


def _compute_mkn(
    inp: torch.Tensor, weight: Optional[torch.Tensor]
) -> Tuple[int, int, int]:
    if not isinstance(inp, torch.Tensor) or inp.numel() == 0:
        return (0, 0, 0)
    if weight is not None and weight.ndim >= 2:
        k_dim = int(weight.shape[-1])
        n_dim = int(weight.shape[0])
    else:
        k_dim = int(inp.shape[-1])
        n_dim = int(getattr(inp, "shape", [0])[-1])
    m_dim = int(inp.numel() // max(k_dim, 1))
    return (m_dim, k_dim, n_dim)


def _flops_linear(
    inp: torch.Tensor,
    out: Any,
    weight: Optional[torch.Tensor],
    include_bias: bool,
    effective_bwd: float,
) -> float:
    m_dim, k_dim, n_dim = _compute_mkn(inp, weight)
    if isinstance(out, torch.Tensor) and out.numel() > 0 and n_dim > 0:
        m_dim = max(m_dim, int(out.numel() // max(n_dim, 1)))
    if m_dim <= 0 or k_dim <= 0 or n_dim <= 0:
        return 0.0
    bias_cost = m_dim * n_dim if include_bias else 0.0
    fwd = 2.0 * m_dim * k_dim * n_dim + bias_cost
    return float(fwd * (1.0 + max(0.0, float(effective_bwd))))


def _flops_conv(
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


def _register_to_linear(
    mod: nn.Module,
    inp: Tuple[Any, ...],
    out: Any,
    *args: Any,
    profiler: "_FlopProfiler",
    include_bias: bool,
    effective_bwd: float,
    **kwargs: Any,
) -> None:
    weight = getattr(mod, "weight", None)
    if weight is None:
        inner_linear = getattr(mod, "linear", None)
        weight = getattr(inner_linear, "weight", None)
    x = inp[0] if inp else None
    if not isinstance(x, torch.Tensor) or weight is None:
        return
    val = _flops_linear(x, out, weight, include_bias, effective_bwd)
    if val > 0.0:
        profiler.add(type(mod).__name__, val)


def _register_to_conv(
    mod: nn.Module,
    inp: Tuple[Any, ...],
    out: Any,
    *args: Any,
    profiler: "_FlopProfiler",
    effective_bwd: float,
    **kwargs: Any,
) -> None:
    weight = getattr(mod, "weight", None)
    x = inp[0] if inp else None
    if not isinstance(x, torch.Tensor) or weight is None:
        return
    groups = getattr(mod, "groups", 1)
    val = _flops_conv(x, out, weight, groups, effective_bwd)
    if val > 0.0:
        profiler.add(type(mod).__name__, val)


def _flops_attention(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    *args: Any,
    bwd_factor: float,
    dropout_p: float,
    training: bool,
    include_softmax_scale_dropout: bool,
    **kwargs: Any,
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


class _FlopProfiler:
    def __init__(self) -> None:
        self._manual_total = 0.0
        self._manual_by_type: Dict[str, float] = {}
        self._tracking_depth = 0
        self._nvtx_soft_counter: float = 0.0
        self._nvtx_getter: Optional[Callable[[], float]] = None

    def is_active(self) -> bool:
        return self._tracking_depth > 0

    def activate(self) -> None:
        self._tracking_depth += 1

    def deactivate(self) -> None:
        self._tracking_depth = max(0, self._tracking_depth - 1)

    def reset(self) -> None:
        self._manual_total = 0.0
        self._manual_by_type.clear()

    def pop(self) -> Tuple[float, Dict[str, float]]:
        total = float(self._manual_total)
        breakdown = {k: float(v) for k, v in self._manual_by_type.items()}
        self.reset()
        return (total, breakdown)

    def get(self) -> float:
        total, _ = self.pop()
        return total

    def sum(self, *, sort: bool = True) -> Tuple[float, Dict[str, float]]:
        total = float(self._manual_total)
        if not sort:
            return total, dict(self._manual_by_type)
        ordered: Dict[str, float] = {}
        for name, value in sorted(
            self._manual_by_type.items(), key=lambda kv: kv[1], reverse=True
        ):
            ordered[name] = float(value)
        return total, ordered

    def add(self, typ: str, value: float) -> None:
        if self.is_active():
            try:
                fv = float(value)
            except Exception:
                return
            if fv <= 0:
                return
            self._manual_total += fv
            self._manual_by_type[typ] = self._manual_by_type.get(typ, 0.0) + fv

    def _add_ntvx(self, value: float) -> None:
        try:
            self._nvtx_soft_counter += float(value) if float(value) > 0 else 0.0
        except Exception:
            pass

    def _get_ntvx(self) -> float:
        return float(self._nvtx_soft_counter)

    def coerce_flops_ntvx(self) -> None:
        if self._nvtx_getter is not None:
            return
        hook = os.getenv("STNET_NVTX_GETTER", "")
        if not hook:
            self._nvtx_getter = self._get_ntvx
            return
        try:
            module_name, attr = hook.split(":", 1)
        except ValueError:
            self._nvtx_getter = self._get_ntvx
            return
        try:
            module = __import__(module_name, fromlist=[attr])
            getter = getattr(module, attr)
            if callable(getter):
                self._nvtx_getter = getter
                return
        except Exception as exc:
            _LOGGER.debug("Failed to import NVTX getter %s: %s", hook, exc)
        self._nvtx_getter = self._get_ntvx

    def capture_ntvx(self, value: float) -> None:
        self._add_ntvx(value)

    def new_flops_ntvx(self, device: Optional[torch.device] = None) -> Any:
        self.coerce_flops_ntvx()
        getter = self._nvtx_getter or self._get_ntvx
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            getattr(torch.cuda, "nvtx")
        except Exception:
            return contextlib.nullcontext()

        class _NvtxScope(contextlib.AbstractContextManager[Any]):
            def __init__(self, dev: Optional[torch.device]) -> None:
                self._dev = dev

            def __enter__(self) -> "_NvtxScope":
                try:
                    if (
                        self._dev is not None
                        and getattr(self._dev, "type", "") == "cuda"
                    ):
                        torch.cuda.synchronize(self._dev)
                except Exception:
                    pass
                return self

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> bool:
                return False

            def get_total_flops(self) -> float:
                try:
                    return float(getter())
                except Exception:
                    return 0.0

        return _NvtxScope(device)

    def _capture_torch(self, display: bool = False) -> Any:
        try:
            from torch.profiler import profile
        except Exception:
            profile = None
        if profile is not None:

            class _TorchFlops(contextlib.AbstractContextManager[Any]):
                def __init__(self, show: bool) -> None:
                    self._show = show
                    self._prof = None

                def __enter__(self) -> "_TorchFlops":
                    self._prof = profile(with_flops=True, record_shapes=False)
                    self._prof.__enter__()
                    return self

                def __exit__(
                    self,
                    exc_type: type[BaseException] | None,
                    exc: BaseException | None,
                    tb: TracebackType | None,
                ) -> bool:
                    if self._prof is not None:
                        self._prof.__exit__(exc_type, exc, tb)
                        if self._show:
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
                        return float(sum(getattr(e, "flops", 0.0) for e in events))
                    except Exception:
                        return 0.0

            return _TorchFlops(bool(display))

        class _TorchFlopsCompat(contextlib.AbstractContextManager[Any]):
            def __init__(self, show: bool) -> None:
                try:
                    from torch.utils.flop_counter import FlopCounterMode as _TorchMode

                    self._impl = _TorchMode(display=show)
                except Exception:
                    self._impl = None

            def __enter__(self) -> "_TorchFlopsCompat":
                if self._impl is not None:
                    self._impl.__enter__()
                return self

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> bool:
                if self._impl is not None:
                    self._impl.__exit__(exc_type, exc, tb)
                return False

            def get_total_flops(self) -> float:
                if self._impl is None:
                    return 0.0
                try:
                    return float(self._impl.get_total_flops())
                except Exception:
                    return 0.0

        return _TorchFlopsCompat(bool(display))

    def start_hooks(
        self,
        model: nn.Module,
        *args: Any,
        mode: str,
        include_bias: bool,
        bwd_factor: Optional[float],
        **kwargs: Any,
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
                        _register_to_linear,
                        profiler=self,
                        include_bias=include_bias,
                        effective_bwd=effective_bwd,
                    )
                )
            elif isinstance(module, nn.modules.conv._ConvNd):
                hook = module.register_forward_hook(
                    partial(
                        _register_to_conv,
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

    def monitoring(
        self, device: Optional[torch.device], *args: Any, display: bool = False, **kwargs: Any
    ) -> Any:
        instrumentation = self

        class _Flops:

            def __init__(self) -> None:
                self.manual_total = 0.0
                self.manual_breakdown: Dict[str, float] = {}
                self.torch_total = 0.0
                self.nvtx_total = 0.0
                self.total = 0.0
                self._torch_scope: Any = None
                self._nvtx_scope: Any = None

            def __enter__(self) -> "_Flops":
                instrumentation.activate()
                instrumentation.reset()
                self._torch_scope = instrumentation._capture_torch(display)
                self._nvtx_scope = instrumentation.new_flops_ntvx(device)
                if self._torch_scope is not None:
                    self._torch_scope.__enter__()
                if self._nvtx_scope is not None:
                    self._nvtx_scope.__enter__()
                return self

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
                manual, breakdown = instrumentation.pop()
                self.manual_total = float(manual)
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
                if self.manual_total > 0.0:
                    self.total = float(self.manual_total)
                else:
                    self.total = max(float(self.torch_total), float(self.nvtx_total))
                instrumentation.deactivate()
                return False

            def get_total_flops(self) -> float:
                return float(self.total)

            def get_manual_breakdown(self) -> Dict[str, float]:
                return dict(self.manual_breakdown)

            def to_dict(self) -> Dict[str, float]:
                return {
                    "manual_total": float(self.manual_total),
                    "torch_total": float(self.torch_total),
                    "nvtx_total": float(self.nvtx_total),
                    "total": float(self.total),
                }

            def verbose(self, top_k: int = 8) -> str:
                lines: list[str] = []
                lines.append(
                    f"total FLOPs: manual={self.manual_total:.3e}, "
                    f"torch={self.torch_total:.3e}, nvtx={self.nvtx_total:.3e}"
                )
                if self.manual_breakdown:
                    lines.append(f"manual breakdown (top {top_k}):")
                    items = sorted(
                        self.manual_breakdown.items(),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )
                    for name, value in items[:top_k]:
                        lines.append(f"  - {name}: {value:.3e}")
                return "\n".join(lines)

        return _Flops()

    def capture(
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
            if not isinstance(q, torch.Tensor) or q.ndim < 4:
                return 0.0
            batch = int(q.shape[0])
            seq_len = int(q.shape[1])
            num_heads = int(q.shape[2])
            head_dim = int(q.shape[3])
        except Exception:
            return 0.0
        total = _flops_attention(
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
            self.add("Attention", total)
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
        FLOP_PROFILER.reset()
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
        return FLOP_PROFILER.monitoring(self._device, display=display)

    @property
    def hook_count(self) -> int:
        return int(self._hook_count)


def capture(
    q: torch.Tensor,
    *args: Any,
    bwd_factor: float = 1.0,
    dropout_p: float = 0.0,
    training: bool = False,
    include_softmax_scale_dropout: bool = True,
    **kwargs: Any,
) -> float:
    return FLOP_PROFILER.capture(
        q,
        *args,
        bwd_factor=bwd_factor,
        dropout_p=dropout_p,
        training=training,
        include_softmax_scale_dropout=include_softmax_scale_dropout,
        **kwargs,
    )



def capture_flops(
    *, sort: bool = True, reset: bool = False
) -> Tuple[float, Dict[str, float]]:
    total, breakdown = FLOP_PROFILER.sum(sort=sort)
    if reset:
        FLOP_PROFILER.reset()
    return total, breakdown


@contextlib.contextmanager
def interval_flops(
    label: str = "",
    *,
    sort: bool = True,
    top_k: int = 8,
) -> Any:

    class _IntervalFlops:
        def __init__(self, label: str) -> None:
            self.label = label
            self.total = 0.0
            self.breakdown: Dict[str, float] = {}

        def verbose(self) -> str:
            lines: list[str] = []
            title = f"[FLOPs] region='{self.label}'" if self.label else "[FLOPs] region"
            lines.append(title)
            lines.append(f"  total: {self.total:.3e}")
            if self.breakdown:
                items = self.breakdown.items()
                if sort:
                    items = sorted(
                        items, key=lambda kv: kv[1], reverse=True
                    )
                lines.append(f"  breakdown (top {top_k}):")
                for name, value in list(items)[:top_k]:
                    lines.append(f"    - {name}: {value:.3e}")
            return "\n".join(lines)

    before_total, before_map = FLOP_PROFILER.sum(sort=False)
    region = _IntervalFlops(label=label)
    try:
        yield region
    finally:
        after_total, after_map = FLOP_PROFILER.sum(sort=False)
        delta_total = float(after_total - before_total)
        delta_map: Dict[str, float] = {}
        keys = set(before_map.keys()) | set(after_map.keys())
        for name in keys:
            v0 = float(before_map.get(name, 0.0))
            v1 = float(after_map.get(name, 0.0))
            dv = v1 - v0
            if dv > 0.0:
                delta_map[name] = dv
        region.total = delta_total
        region.breakdown = delta_map
