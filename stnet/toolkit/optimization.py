# -*- coding: utf-8 -*-
from __future__ import annotations
import contextlib
import importlib
import logging
import os
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from torch import nn, optim

try:
    from torch.distributed.algorithms.join import Join as _TorchJoin
except ImportError:
    _TorchJoin = None

Join: type[AbstractContextManager[None]] | None = _TorchJoin

from .capability import (
    get_device,
    get_runtime_config,
    initialize_sdpa_backends,
    is_cpu_bf16_supported,
    is_cuda_bf16_supported,
    is_float8_supported,
    is_int8_supported,
    optimal_optimizer_params,
)

try:
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Float8WeightOnlyConfig,
        Int4WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        Int8WeightOnlyConfig,
        quantize_,
    )
except ImportError:
    quantize_ = None
QATConfig = None
QATStep = None
try:
    from torchao.quantization.qat import QATConfig, QATStep
except Exception:
    try:
        from torchao.quantization.qat.api import QATConfig, QATStep
    except Exception:
        try:
            from torchao.quantization.qat import (
                FromIntXQuantizationAwareTrainingConfig,
                IntXQuantizationAwareTrainingConfig,
            )

            class _ShimQATStep:
                PREPARE = "prepare"
                CONVERT = "convert"

            class _ShimQATConfig:
                def __init__(
                    self,
                    base_config: Any = None,
                    activation_config: Any = None,
                    weight_config: Any = None,
                    *args: Any,
                    step: Any = "prepare",
                    **kwargs: Any,
                ) -> None:
                    self.base_config = base_config
                    self.activation_config = activation_config
                    self.weight_config = weight_config
                    self.step = step

                def to_legacy(self) -> Any:
                    if self.step == "prepare":
                        return IntXQuantizationAwareTrainingConfig(
                            self.activation_config, self.weight_config
                        )
                    else:
                        return FromIntXQuantizationAwareTrainingConfig()

            QATConfig, QATStep = (_ShimQATConfig, _ShimQATStep)
        except Exception:

            class _NullQATConfig:
                pass

            class _NullQATStep:
                PREPARE = "prepare"
                CONVERT = "convert"

            QATConfig, QATStep = (_NullQATConfig, _NullQATStep)


def joining(*joinables: Optional[object]) -> AbstractContextManager[None]:
    if Join is None:
        return contextlib.nullcontext()
    to_join = []
    for obj in joinables:
        if obj is None:
            continue
        if getattr(obj, "join_hook", None) is not None:
            to_join.append(obj)
    if not to_join:
        return contextlib.nullcontext()
    return Join(to_join, throw_on_early_termination=True)


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

    def consume_manual(self) -> float:
        total = float(self._manual_total)
        self.reset_manual()
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
        if hook:
            try:
                self._nvtx_getter = _import_callable(hook)
                return
            except Exception:
                pass
        self._nvtx_getter = self._nvtx_soft_getter

    def record_nvtx_soft(self, value: float) -> None:
        self._nvtx_soft_add(value)

    def make_nvtx_counter(self, device: Optional[torch.device] = None) -> Any:
        try:
            dev = device if device is not None else get_device()
        except Exception:
            dev = None
        return self._nvtx_counter(dev)

    def start_hooks(
        self,
        model: nn.Module,
        mode: str = "train",
        bwd_factor: Optional[float] = None,
        include_bias: bool = True,
    ) -> List[Any]:
        effective_bwd = 0.0 if mode.lower() == "eval" else 2.0
        _env = os.environ.get("STF_FLOP_BWD_FACTOR", "").strip()
        if _env:
            effective_bwd = float(_env)
        if bwd_factor is not None:
            effective_bwd = float(bwd_factor)
        Linear = nn.Linear
        target_types: List[type] = [Linear]
        try:
            import transformer_engine.pytorch as te

            _te_linear = getattr(te, "Linear", None)
            _te_ln_linear = getattr(te, "LayerNormLinear", None)
        except Exception:
            _te_linear = None
            _te_ln_linear = None
        for _t in (_te_linear, _te_ln_linear):
            if _t is not None and _t not in target_types:
                target_types.append(_t)
        conv_types: List[type] = []
        for _name in ("Conv1d", "Conv2d", "Conv3d"):
            t = getattr(nn, _name, None)
            if t is not None:
                conv_types.append(t)

        def _infer_linear_mkn(
            inp: torch.Tensor, weight: Optional[torch.Tensor]
        ) -> Tuple[int, int, int]:
            K = (
                int(weight.shape[-1])
                if weight is not None and weight.ndim >= 2
                else int(inp.shape[-1])
            )
            M = int(inp.numel() // max(K, 1))
            N = (
                int(weight.shape[0])
                if weight is not None and weight.ndim >= 2
                else int(getattr(inp, "shape", [0])[-1])
            )
            return (M, K, N)

        def linear_flops(
            inp: torch.Tensor, out: torch.Tensor, weight: Optional[torch.Tensor]
        ) -> float:
            M, K, N = _infer_linear_mkn(inp, weight)
            if out is not None and out.numel() > 0 and (N > 0):
                M = max(M, int(out.numel() // max(N, 1)))
            if M <= 0 or K <= 0 or N <= 0:
                return 0.0
            bias_cost = 1.0 * M * N if include_bias else 0.0
            fwd = 2.0 * M * K * N + bias_cost
            return float(fwd * (1.0 + max(0.0, float(effective_bwd))))

        def conv_flops(
            inp: torch.Tensor,
            out: torch.Tensor,
            weight: Optional[torch.Tensor],
            groups: int = 1,
        ) -> float:
            if weight is None or weight.ndim < 3:
                return 0.0
            try:
                out_elems = int(out.numel())
                groups = max(1, int(groups))
                cin_total = (
                    int(inp.shape[1])
                    if isinstance(inp, torch.Tensor) and inp.ndim >= 2
                    else int(weight.shape[1] * groups)
                )
                cin_per_group = max(1, cin_total // groups)
                kernel = int(weight[0].numel()) // max(cin_per_group, 1)
                return float(
                    out_elems
                    * (2.0 * cin_per_group * kernel)
                    * (1.0 + max(0.0, float(effective_bwd)))
                )
            except Exception:
                return 0.0

        handles: List[Any] = []
        self.activate()

        def _hook_linear(mod: nn.Module, inp: Tuple[Any, ...], out: Any) -> None:
            weight = getattr(mod, "weight", None)
            if weight is None:
                inner_linear = getattr(mod, "linear", None)
                weight = getattr(inner_linear, "weight", None)
            x = inp[0] if inp else None
            if not isinstance(x, torch.Tensor) or weight is None:
                return
            val = linear_flops(x, out, weight)
            if val > 0.0:
                self.add_manual(type(mod).__name__, val)

        def _hook_conv(mod: nn.Module, inp: Tuple[Any, ...], out: Any) -> None:
            weight = getattr(mod, "weight", None)
            x = inp[0] if inp else None
            if not isinstance(x, torch.Tensor) or weight is None:
                return
            groups = getattr(mod, "groups", 1)
            val = conv_flops(x, out, weight, groups=groups)
            if val > 0.0:
                self.add_manual(type(mod).__name__, val)

        try:
            for m in model.modules():
                if isinstance(m, tuple(target_types)):
                    handles.append(m.register_forward_hook(_hook_linear))
                elif any(isinstance(m, t) for t in conv_types):
                    handles.append(m.register_forward_hook(_hook_conv))
            return handles
        except Exception:
            self.stop_hooks(handles)
            raise

    def stop_hooks(self, handles: Sequence[Any]) -> None:
        for h in list(handles):
            try:
                h.remove()
            except Exception:
                pass
        self.deactivate()

    def _torch_counter(self, display: bool = False) -> Any:
        class _TorchScope(contextlib.AbstractContextManager):
            def __init__(self, show: bool) -> None:
                try:
                    from torch.utils.flop_counter import FlopCounterMode as _TorchMode

                    self._impl = _TorchMode(display=show)
                except Exception:
                    self._impl = None

            def __enter__(self) -> "_TorchScope":
                if self._impl is not None:
                    self._impl.__enter__()
                return self

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
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

        return _TorchScope(bool(display))

    def _nvtx_counter(self, device: Optional[torch.device]) -> Any:
        class _NvtxScope(contextlib.AbstractContextManager):
            def __init__(self, getter: Optional[Callable[[], float]]) -> None:
                self._getter = getter
                self._start = 0.0
                self._end = 0.0

            def __enter__(self) -> "_NvtxScope":
                if self._getter is None:
                    return self
                try:
                    self._start = float(self._getter())
                except Exception:
                    self._start = 0.0
                return self

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
                if self._getter is None:
                    return False
                try:
                    self._end = float(self._getter())
                except Exception:
                    self._end = self._start
                return False

            def get_total_flops(self) -> float:
                if self._getter is None:
                    return 0.0
                end = self._end
                if not end > self._start:
                    try:
                        end = float(self._getter())
                    except Exception:
                        end = self._start
                return max(0.0, float(end) - float(self._start))

        if device is None or getattr(device, "type", None) != "cuda":
            return _NvtxScope(None)
        self.ensure_nvtx_getter()
        getter = self._nvtx_getter or self._nvtx_soft_getter
        return _NvtxScope(getter)

    def step_scope(self, device: Optional[torch.device], display: bool = False) -> Any:
        instrumentation = self

        class _StepScope(contextlib.AbstractContextManager):
            def __init__(self) -> None:
                self.manual_total = 0.0
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
                manual = instrumentation.consume_manual()
                if manual > 0.0:
                    instrumentation.record_nvtx_soft(manual)
                self.manual_total = manual
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
                return False

            def get_total_flops(self) -> float:
                return float(self.total)

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
            B = int(q.shape[0])
            S = int(q.shape[1])
            H = int(q.shape[2])
            d = int(q.shape[3])
        except Exception:
            return 0.0
        if B <= 0 or S <= 0 or H <= 0 or (d <= 0):
            return 0.0
        matmul = 4.0 * B * H * S**2 * d
        misc = 0.0
        if include_softmax_scale_dropout:
            misc_coeff = 6.0
            if training and float(dropout_p) > 0.0:
                misc_coeff += 1.0
            misc = misc_coeff * (B * H * S**2)
        fwd = matmul + misc
        total = float(fwd * (1.0 + max(0.0, float(bwd_factor))))
        if total > 0.0:
            self.add_manual("Attention", total)
        return total


FLOP_PROFILER = _FlopProfiler()


@dataclass
class LossWeightController:
    momentum: float = 0.9
    min_weight: float = 0.05
    max_weight: float = 0.95
    eps: float = 1e-06
    top_avg: float = 1.0
    bottom_avg: float = 1.0

    def weights(self) -> Tuple[float, float]:
        top = max(self.eps, self.top_avg)
        bottom = max(self.eps, self.bottom_avg)
        total = top + bottom
        if total <= 0.0:
            return (0.5, 0.5)
        ratio_top = top / total
        ratio_bottom = bottom / total
        ratio_top = float(min(max(ratio_top, self.min_weight), self.max_weight))
        ratio_bottom = float(
            min(max(ratio_bottom, self.min_weight), self.max_weight)
        )
        norm = ratio_top + ratio_bottom
        if norm <= 0.0:
            return (0.5, 0.5)
        return (ratio_top / norm, ratio_bottom / norm)

    def update(
        self,
        top_loss: Optional[torch.Tensor],
        bottom_loss: Optional[torch.Tensor],
    ) -> None:
        if top_loss is not None:
            top_val = float(top_loss.detach().abs().mean().item())
            self.top_avg = self.momentum * self.top_avg + (
                1.0 - self.momentum
            ) * max(top_val, self.eps)
        if bottom_loss is not None:
            bottom_val = float(bottom_loss.detach().abs().mean().item())
            self.bottom_avg = self.momentum * self.bottom_avg + (
                1.0 - self.momentum
            ) * max(bottom_val, self.eps)


def _import_callable(spec: str) -> Callable:
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("Empty spec for callable import")
    raw = spec.strip()
    root_pkg = __package__.split(".", 1)[0] if __package__ else "stnet"
    default_module = f"{root_pkg}.toolkit.optimization"
    if ":" in raw:
        mod_part, fn_part = raw.split(":", 1)
    else:
        mod_part, fn_part = ("", raw)
    mod_part = mod_part.strip()
    fn_part = fn_part.strip()
    if not fn_part:
        raise ValueError(f"Missing function in spec: {spec}")
    if not mod_part:
        mod_name = default_module
    elif mod_part.startswith("."):
        mod_name = f"{root_pkg}{mod_part}"
    elif not mod_part.startswith(root_pkg + ".") and mod_part.split(".")[
        0
    ] not in ("importlib", "torch", "math", "sys"):
        mod_name = f"{root_pkg}.{mod_part}"
    else:
        mod_name = mod_part
    module = importlib.import_module(mod_name)
    fn = getattr(module, fn_part, None)
    if not callable(fn):
        raise TypeError(f"{mod_name}:{fn_part} is not callable or not found")
    return fn

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

    def __enter__(self) -> FlopCounter:
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


@contextlib.contextmanager
def no_synchronization(m: torch.nn.Module, enable: bool) -> Any:
    if not enable or not torch.distributed.is_initialized():
        yield
        return
    with contextlib.ExitStack() as stack:
        for mod in m.modules():
            ns = getattr(mod, "no_sync", None)
            if callable(ns):
                stack.enter_context(ns())
        yield


def compile(
    m: nn.Module,
    *args: Any,
    mode: str = (
        "max-autotune" if get_device().type == "cuda"
        else "max-autotune-no-cudagraphs"
    ),
    fullgraph: bool = True,
    dynamic: bool = True,
    backend: str = "inductor",
    **kwargs: Any,
) -> nn.Module:
    m_compile = getattr(m, "compile", None)
    if callable(m_compile):
        try:
            compiled = m_compile(
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
                backend=backend,
            )
            return compiled if isinstance(compiled, nn.Module) else m
        except TypeError as exc:
            _LOGGER.debug(
                "module.compile signature mismatch for %s: %s",
                m.__class__.__name__,
                exc,
                exc_info=True,
            )
            try:
                compiled = m_compile()
                return compiled if isinstance(compiled, nn.Module) else m
            except Exception as inner_exc:
                _LOGGER.debug(
                    "module.compile() without kwargs failed for %s: %s",
                    m.__class__.__name__,
                    inner_exc,
                    exc_info=True,
                )
        except Exception:
            _LOGGER.warning(
                "module.compile failed for %s; returning original module",
                m.__class__.__name__,
                exc_info=True,
            )
    _compile = getattr(torch, "compile", None)
    if callable(_compile):
        try:
            return _compile(
                m,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
                backend=backend,
            )
        except Exception:
            _LOGGER.warning(
                "torch.compile failed for %s; returning original module",
                m.__class__.__name__,
                exc_info=True,
            )
            return m
    return m


def _is_contiguous_bshd(t: torch.Tensor) -> bool:
    if t.dim() != 4:
        return False
    _, S, H, D = t.shape
    st = t.stride()
    return (
        t.is_contiguous()
        and st[-1] == 1
        and (st[-2] == D)
        and (st[-3] == H * D)
        and (st[-4] == S * H * D)
    )


class TunedDPA(torch.nn.Module):
    def __init__(
        self,
        num_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        te_first: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.nh = int(num_heads) if num_heads is not None else None
        self.hd = int(head_dim) if head_dim is not None else None
        cfg = get_runtime_config()
        self.te_first = (
            bool(cfg.te_first) if te_first is None else bool(te_first)
        )
        ok, te = self._is_te_available()
        self._te_ok = (
            ok
            and torch.cuda.is_available()
            and (self.nh is not None)
            and (self.hd is not None)
        )
        self._te_attn: Any = None
        if self._te_ok:
            self._te = te
            self._te_attn = te.DotProductAttention(
                num_attention_heads=self.nh,
                kv_channels=self.hd,
                qkv_format="bshd",
                attention_dropout=0.0,
            )

    @staticmethod
    def _is_te_available() -> Any:
        if os.environ.get("STF_DISABLE_TE", "") == "1":
            return (False, None)
        try:
            with contextlib.ExitStack() as st:
                import warnings as _w

                st.enter_context(_w.catch_warnings())
                _w.filterwarnings(
                    "ignore",
                    message="Detected a Jax installation.*",
                    category=RuntimeWarning,
                )
                import transformer_engine.pytorch as te
            return (True, te)
        except Exception:
            return (False, None)

    @torch._dynamo.disable()
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float | torch.Tensor = 0.0,
        is_causal: bool = False,
        training: bool | None = None,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        training = bool(training if training is not None else self.training)
        if isinstance(dropout_p, torch.Tensor):
            dropout_p = float(dropout_p.item())
        q = self._to_optimal_dtype(q)
        k = self._to_optimal_dtype(k)
        v = self._to_optimal_dtype(v)
        q_bshd = q.contiguous()
        k_bshd = k.contiguous()
        v_bshd = v.contiguous()
        try:
            _bwd = 2.0 if training else 0.0
            attention_flops_bshd(
                q_bshd,
                bwd_factor=_bwd,
                dropout_p=float(dropout_p),
                training=bool(training),
            )
        except Exception:
            pass
        dropout_val = float(dropout_p) if training else 0.0
        use_te = (
            self.te_first
            and self._te_ok
            and (self._te_attn is not None)
            and (attn_mask is None)
            and (not kwargs)
        )
        if use_te:
            q_te = q_bshd.permute(0, 2, 1, 3).contiguous()
            k_te = k_bshd.permute(0, 2, 1, 3).contiguous()
            v_te = v_bshd.permute(0, 2, 1, 3).contiguous()
            try:
                out_te = self._te_attn(
                    q_te,
                    k_te,
                    v_te,
                    attn_mask=None,
                    attention_dropout=dropout_val,
                    is_causal=bool(is_causal),
                    training=training,
                )
            except Exception:
                use_te = False
            else:
                return out_te.permute(0, 2, 1, 3).contiguous()
        sdpa_kwargs = {
            "attn_mask": attn_mask,
            "dropout_p": dropout_val,
            "is_causal": bool(is_causal),
        }
        q_bhsd = q_bshd.permute(0, 2, 1, 3).contiguous()
        k_bhsd = k_bshd.permute(0, 2, 1, 3).contiguous()
        v_bhsd = v_bshd.permute(0, 2, 1, 3).contiguous()
        backends = initialize_sdpa_backends()
        sdpa_out: Optional[torch.Tensor] = None
        if backends:
            try:
                from torch.nn.attention import sdpa_kernel
            except Exception:
                backends = []
        if backends:
            with sdpa_kernel(backends):
                sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                    q_bhsd, k_bhsd, v_bhsd, **sdpa_kwargs
                )
        if sdpa_out is None:
            sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, **sdpa_kwargs
            )
        return sdpa_out.permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def _to_optimal_dtype(x: torch.Tensor) -> torch.Tensor:
        dev = x.device.type
        if dev == "cpu" and x.dtype in (torch.float16, torch.bfloat16):
            return x.float()
        if dev == "mps" and x.dtype == torch.bfloat16:
            return x.to(torch.float16)
        return x


class MSRCompat(nn.Module):
    def __init__(
        self, d_model: int, nhead: int, use_gate: bool = True
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = self.d_model // self.nhead
        self.use_gate = bool(use_gate)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.g_proj = (
            nn.Linear(self.d_model, self.d_model, bias=False)
            if self.use_gate
            else None
        )
        self._beta = nn.Parameter(torch.full((self.nhead,), -0.2))
        self.norm = nn.LayerNorm(self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        *args: Any,
        attn_mask: Optional[torch.Tensor] = None,
        state: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        B, S, D = x.shape
        H, Dh = (self.nhead, self.head_dim)
        q = self.q_proj(x).view(B, S, H, Dh)
        v = self.v_proj(x).view(B, S, H, Dh)
        manual_flops = float(max(S - 1, 0) * B * H * Dh * 2)
        lam = (
            torch.sigmoid(self._beta)
            .view(1, H, 1)
            .to(dtype=v.dtype, device=v.device)
        )
        prev = v[:, 0].clone()
        s_list = [prev]
        for t in range(1, S):
            prev = lam * prev + v[:, t]
            s_list.append(prev)
        s = torch.stack(s_list, dim=1).contiguous()
        manual_flops += float(B * S * D)
        y = (q * s).contiguous().view(B, S, D)
        y = self.norm(y)
        if self.use_gate and self.g_proj is not None:
            gate = torch.nn.functional.silu(self.g_proj(x))
            manual_flops += float(B * S * D)
            y = y * gate
        if manual_flops > 0.0:
            FLOP_PROFILER.add_manual("MSRCompat", manual_flops)
        return self.o_proj(y)


class TunedMSR(nn.Module):
    def __init__(
        self, d_model: int, nhead: int, use_gate: bool = True
    ) -> None:
        super().__init__()
        self.d_model, self.nhead, self.use_gate = (
            int(d_model),
            int(nhead),
            bool(use_gate),
        )
        self._ts_ok = False
        try:
            from torchscale.component.multiscale_retention import (
                MultiScaleRetention as _TSMSR,
            )

            self._ts_msr = _TSMSR(self.d_model, self.nhead)
            self._ts_key_dim = int(
                getattr(self._ts_msr, "key_dim", self.d_model // self.nhead)
            )
            self._ts_ok = True
        except Exception:
            self._ts_ok = False
            self._fallback = MSRCompat(
                self.d_model, self.nhead, use_gate=self.use_gate
            )
        self._rope_theta = 10000.0
        self._decay_init = 5.0
        self._decay_range = 1.0

    def _build_rel_pos(self, seq_len: Any, device: Any, dtype: Any) -> Any:
        kd = int(
            self._ts_key_dim
            if getattr(self, "_ts_key_dim", None) is not None
            else self.d_model // self.nhead
        )
        half = kd // 2
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_freq = 1.0 / self._rope_theta ** torch.linspace(
            0, 1, half, device=device, dtype=torch.float32
        )
        freqs = torch.einsum("n,d->nd", t, inv_freq)

        def _dup(x: Any) -> Any:
            return torch.stack((x, x), dim=-1).reshape(x.shape[0], -1)

        sin = _dup(torch.sin(freqs)).to(dtype)[None, None, :, :]
        cos = _dup(torch.cos(freqs)).to(dtype)[None, None, :, :]
        L = seq_len
        i = torch.arange(L, device=device)
        j = torch.arange(L, device=device)
        diff = (i[:, None] - j[None, :]).to(dtype)
        tril = (i[:, None] >= j[None, :]).to(dtype)
        h = torch.arange(self.nhead, device=device, dtype=dtype)
        gammas = 1.0 - torch.pow(
            2.0,
            -(self._decay_init + self._decay_range * (h / max(self.nhead, 1))),
        )
        gammas = torch.clamp(
            gammas, min=torch.finfo(dtype).tiny, max=1 - 1e-09
        )
        inner_mask = torch.pow(
            gammas.view(1, self.nhead, 1, 1), diff.view(1, 1, L, L)
        ) * tril.view(1, 1, L, L)
        return ((sin, cos), inner_mask)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        state: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self._ts_ok:
            _, S, _ = x.shape
            rel_pos = self._build_rel_pos(S, x.device, x.dtype)
            return self._ts_msr(
                x, rel_pos, chunkwise_recurrent=False, incremental_state=state
            )
        return self._fallback(x, attn_mask=attn_mask, state=state, **kwargs)


class TunedAMP:
    @staticmethod
    def float(device: Union[torch.device, str, None] = None) -> Any:
        dev = torch.device(device) if device is not None else get_device()
        if dev.type == "cuda":
            try:
                from transformer_engine.pytorch import fp8_autocast

                ok_cc, _ = (True, "")
                try:
                    ok_cc, _ = is_float8_supported(dev)
                except Exception:
                    ok_cc = False
                if ok_cc:
                    kwargs = {}
                    try:
                        if (
                            torch.distributed.is_available()
                            and torch.distributed.is_initialized()
                        ):
                            kwargs["fp8_group"] = torch.distributed.group.WORLD
                    except Exception:
                        pass
                    return fp8_autocast(enabled=True, **kwargs)
            except Exception:
                pass
            dtype = (
                torch.bfloat16 if is_cuda_bf16_supported() else torch.float16
            )
            return torch.amp.autocast(
                device_type="cuda", dtype=dtype, enabled=True
            )
        if hasattr(torch, "xpu") and dev.type == "xpu":
            try:
                return torch.amp.autocast(
                    device_type="xpu", dtype=torch.bfloat16, enabled=True
                )
            except Exception:
                from contextlib import nullcontext

                return nullcontext()
        if dev.type == "mps":
            try:
                return torch.amp.autocast(
                    device_type="mps", dtype=torch.float16, enabled=True
                )
            except Exception:
                from contextlib import nullcontext

                return nullcontext()
        if dev.type == "cpu" and is_cpu_bf16_supported():
            return torch.amp.autocast(
                device_type="cpu", dtype=torch.bfloat16, enabled=True
            )
        from contextlib import nullcontext

        return nullcontext()

    @staticmethod
    def integer(device: Union[torch.device, str, None] = None) -> Any:
        dev = torch.device(device) if device is not None else get_device()
        if dev.type in ("cpu", "xpu"):
            try:
                import intel_extension_for_pytorch

                return intel_extension_for_pytorch.quantization.autocast(
                    dtype=torch.int8
                )
            except (ImportError, AttributeError):
                pass
        from contextlib import nullcontext

        return nullcontext()


class TunedAdamW:
    @staticmethod
    def float(
        model_or_params: Union[
            nn.Module, Iterable[nn.Parameter], Sequence[Dict[str, Any]]
        ],
        lr: float,
        *args: Any,
        weight_decay: float = 0.0,
        device: Optional[torch.device] = None,
        use_fp8: bool = True,
        use_foreach: Optional[bool] = False,
        use_fused: bool = False,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> optim.Optimizer:
        params = (
            model_or_params.parameters()
            if hasattr(model_or_params, "parameters")
            else model_or_params
        )
        dev: torch.device = device or get_device()
        if hasattr(dev, "type") and dev.type == "cuda":
            try:
                from transformer_engine.pytorch.optimizers import (
                    FusedAdam as TEFusedAdam,
                )

                opt = TEFusedAdam(params, lr=lr, weight_decay=weight_decay)
                if logger:
                    logger("[OPT] Using FusedAdam (Transformer Engine)")
                return opt
            except Exception as e:
                if logger:
                    logger(f"[OPT] TE FusedAdam unavailable: {e}")
        if use_fp8 and (hasattr(dev, "type") and dev.type == "cuda"):
            ok, why = is_float8_supported(dev)
            if "TE" in str(why):
                try:
                    from transformer_engine.pytorch.optimizers import FusedAdam

                    opt: optim.Optimizer = FusedAdam(
                        params, lr=lr, weight_decay=weight_decay
                    )
                    if logger:
                        logger(
                            f"[OPT] Using FusedAdam (transformer_engine) — {why}"
                        )
                    return opt
                except Exception as e:
                    if logger:
                        logger(
                            f"[OPT] transformer_engine.FusedAdam unavailable: {e}"
                        )
            if "AO" in str(why):
                try:
                    from torchao.optim import AdamWFp8

                    opt = AdamWFp8(params, lr=lr, weight_decay=weight_decay)
                    if logger:
                        logger(f"[OPT] Using AdamW-FP8 (torchao) — {why}")
                    return opt
                except Exception as e:
                    if logger:
                        logger(f"[OPT] torchao.AdamWFp8 unavailable: {e}")
            elif logger:
                logger(
                    f"[OPT] FP8 optimizers not supported ({why}) — fallback"
                )
        flags: Dict[str, bool] = optimal_optimizer_params(
            dev, use_foreach=use_foreach, use_fused=use_fused
        )
        opt = optim.AdamW(params, lr=lr, weight_decay=weight_decay, **flags)
        if logger:
            logger(f"[OPT] Using torch.optim.AdamW (flags={flags})")
        return opt

    @staticmethod
    def integer(
        model_or_params: Union[
            nn.Module, Iterable[nn.Parameter], Sequence[Dict[str, Any]]
        ],
        lr: float,
        *args: Any,
        weight_decay: float = 0.0,
        dtype: Optional[str] = None,
        device: Optional[torch.device] = None,
        use_foreach: Optional[bool] = False,
        use_fused: bool = False,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> optim.Optimizer:
        params = (
            model_or_params.parameters()
            if hasattr(model_or_params, "parameters")
            else model_or_params
        )
        dev: torch.device = device or get_device()
        if hasattr(dev, "type") and dev.type == "cuda":
            try:
                from transformer_engine.pytorch.optimizers import (
                    FusedAdam as TEFusedAdam,
                )

                opt = TEFusedAdam(params, lr=lr, weight_decay=weight_decay)
                if logger:
                    logger("[OPT] Using FusedAdam (Transformer Engine)")
                return opt
            except Exception as e:
                if logger:
                    logger(f"[OPT] TE FusedAdam unavailable: {e}")
        if dtype in {"int8", "int4"}:
            try:
                try:
                    from torchao.optim import AdamW4bit, AdamW8bit
                except ImportError:
                    from torchao.prototype.low_bit_optim import (
                        AdamW4bit,
                        AdamW8bit,
                    )
                if dtype == "int8":
                    ok_int8, reason_int8 = is_int8_supported(dev)
                    if ok_int8:
                        opt = AdamW8bit(
                            params, lr=lr, weight_decay=weight_decay
                        )
                        if logger:
                            note = f" — {reason_int8}" if reason_int8 else ""
                            logger(f"[OPT] TorchAO AdamW8bit{note}")
                        return opt
                    if logger:
                        logger(
                            f"[OPT] INT8 optimizers not supported ({reason_int8}) — fallback"
                        )
                else:
                    opt = AdamW4bit(
                        params, lr=lr, weight_decay=weight_decay
                    )
                    if logger:
                        logger("[OPT] TorchAO AdamW4bit")
                    return opt
            except Exception as e:
                if logger:
                    logger(f"[OPT] TorchAO low-bit optimizer unavailable: {e}")
        flags: Dict[str, bool] = optimal_optimizer_params(
            dev, use_foreach=use_foreach, use_fused=use_fused
        )
        opt = optim.AdamW(params, lr=lr, weight_decay=weight_decay, **flags)
        if logger:
            logger(f"[OPT] Using AdamW (flags={flags})")
        return opt


def ptq(
    model: nn.Module,
    *args: Any,
    mode: str,
    dynamic_activations: bool = True,
    group_size: int = 128,
    logger: Any = None,
    **kwargs: Any,
) -> Tuple[nn.Module, bool, str]:
    if quantize_ is None:
        msg = "torchao.quantization not installed (PTQ disabled)"
        if logger:
            logger(f"[Q] {msg}")
        return (model, False, msg)
    match mode:
        case "fp8":
            try:
                cfg = (
                    Float8DynamicActivationFloat8WeightConfig()
                    if dynamic_activations
                    else Float8WeightOnlyConfig()
                )
            except Exception as e:
                return (
                    model,
                    False,
                    f"torchao Float8 config unavailable: {e}",
                )
        case "int8":
            try:
                cfg = (
                    Int8DynamicActivationInt8WeightConfig()
                    if dynamic_activations
                    else Int8WeightOnlyConfig()
                )
            except Exception as e:
                return (model, False, f"torchao Int8 config unavailable: {e}")
        case "int4":
            try:
                cfg = Int4WeightOnlyConfig(group_size=group_size)
            except Exception as e:
                return (model, False, f"torchao Int4 config unavailable: {e}")
        case _:
            return (model, False, f"unknown mode: {mode}")
    quantize_(model, cfg)
    if logger:
        logger(f"[Q] applied {cfg.__class__.__name__}")
    return (model, True, cfg.__class__.__name__)


class QAT:
    @staticmethod
    def initialize(
        model: nn.Module,
        *args: Any,
        mode: str,
        group_size: int = 128,
        dynamic_activations: bool = True,
        logger: Any = None,
        **kwargs: Any,
    ) -> Any:
        if quantize_ is None:
            raise ImportError(
                "TorchAO not installed: QAT unavailable (missing quantize_)"
            )
        match mode:
            case "qat-int8":
                base_cfg = (
                    Int8DynamicActivationInt8WeightConfig()
                    if dynamic_activations
                    else Int8WeightOnlyConfig()
                )
            case "qat-int4":
                base_cfg = Int4WeightOnlyConfig(group_size=group_size)
            case _:
                raise ValueError("mode must be 'qat-int8' or 'qat-int4'")
        quantize_(model, QATConfig(base_cfg, step=QATStep.PREPARE))
        if logger:
            logger(f"[QAT] prepared with base {base_cfg.__class__.__name__}")
        return base_cfg

    @staticmethod
    def apply(
        model: nn.Module,
        *args: Any,
        base_cfg: Any,
        logger: Any = None,
        **kwargs: Any,
    ) -> None:
        if quantize_ is None:
            raise ImportError(
                "TorchAO not installed: QAT unavailable (missing quantize_)"
            )
        quantize_(model, QATConfig(base_cfg, step=QATStep.CONVERT))
        if logger:
            logger("[QAT] converted to quantized model")


class ModuleTuner:
    @staticmethod
    def _infer_optimal_dtype(dev: torch.device) -> torch.dtype:
        if dev.type == "cuda":
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
            except Exception:
                return torch.float16
        elif dev.type == "cpu" and is_cpu_bf16_supported():
            return torch.bfloat16
        return torch.float32

    @staticmethod
    def _apply_te_module(
        root: nn.Module,
        *args: Any,
        apply_te_linear: bool = True,
        apply_te_layer_norm: bool = True,
        apply_te_rms_norm: bool = True,
        filter_linear: Optional[Callable[[nn.Linear, str], bool]] = None,
        params_dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> Tuple[nn.Module, int]:
        try:
            import transformer_engine.pytorch as te
        except Exception as e:
            raise ImportError(f"Transformer Engine import failed: {e}")
        n_swapped = 0
        for fqname, child in list(root.named_children()):
            replaced = False
            if apply_te_linear and isinstance(child, nn.Linear):
                if filter_linear is None or filter_linear(child, fqname):
                    ok_dim = (
                        child.in_features % 16 == 0
                        and child.out_features % 16 == 0
                    )
                    if ok_dim:
                        new = te.Linear(
                            child.in_features,
                            child.out_features,
                            bias=child.bias is not None,
                            params_dtype=params_dtype,
                        )
                        with torch.no_grad():
                            new.weight.copy_(child.weight)
                            if (
                                child.bias is not None
                                and hasattr(new, "bias")
                                and (new.bias is not None)
                            ):
                                new.bias.copy_(child.bias)
                        setattr(root, fqname, new)
                        n_swapped += 1
                        replaced = True
            if (
                not replaced
                and apply_te_layer_norm
                and isinstance(child, nn.LayerNorm)
            ):
                hidden = (
                    int(child.normalized_shape[0])
                    if isinstance(child.normalized_shape, (tuple, list))
                    else int(child.normalized_shape)
                )
                new = te.LayerNorm(
                    hidden, eps=float(child.eps), params_dtype=params_dtype
                )
                with torch.no_grad():
                    if child.weight is not None:
                        new.weight.copy_(child.weight)
                    if child.bias is not None:
                        new.bias.copy_(child.bias)
                setattr(root, fqname, new)
                n_swapped += 1
                replaced = True
            _is_rms = child.__class__.__name__ == "RMSNorm"
            if not _is_rms and hasattr(nn, "RMSNorm"):
                try:
                    _is_rms = isinstance(child, nn.RMSNorm)
                except Exception:
                    _is_rms = False
            if not replaced and apply_te_rms_norm and _is_rms:
                try:
                    import transformer_engine.pytorch as te

                    hidden = int(child.weight.shape[0])
                    new = te.RMSNorm(
                        hidden,
                        eps=float(getattr(child, "eps", 1e-06)),
                        params_dtype=params_dtype,
                    )
                    with torch.no_grad():
                        new.weight.copy_(child.weight)
                    setattr(root, fqname, new)
                    n_swapped += 1
                    replaced = True
                except Exception:
                    pass
            if not replaced:
                _, k = ModuleTuner._apply_te_module(
                    child,
                    apply_te_linear=apply_te_linear,
                    apply_te_layer_norm=apply_te_layer_norm,
                    apply_te_rms_norm=apply_te_rms_norm,
                    filter_linear=filter_linear,
                    params_dtype=params_dtype,
                )
                n_swapped += k
        return (root, n_swapped)

    @staticmethod
    def _apply_te_attention(
        root: nn.Module,
        *args: Any,
        params_dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> Tuple[nn.Module, int]:
        try:
            import transformer_engine.pytorch as te
        except Exception:
            return (root, 0)
        n_swapped = 0
        for fqname, child in list(root.named_children()):
            new_child, k = ModuleTuner._apply_te_attention(
                child, params_dtype=params_dtype
            )
            if k:
                setattr(root, fqname, new_child)
                n_swapped += k
                child = new_child
            replaced = False
            try:
                if isinstance(child, nn.MultiheadAttention):
                    embed_dim = int(child.embed_dim)
                    num_heads = int(child.num_heads)
                    p_drop = float(getattr(child, "dropout", 0.0))
                    te_mha = te.MultiheadAttention(
                        embed_dim,
                        num_heads,
                        attention_dropout=p_drop,
                        bias=True,
                        params_dtype=params_dtype,
                        qkv_weight_interleaved=False,
                        fuse_qkv_params=True,
                        attn_mask_type="no_mask",
                        window_size=(-1, -1),
                    )
                    with torch.no_grad():
                        if getattr(child, "in_proj_weight", None) is not None:
                            te_mha.in_proj_weight.copy_(child.in_proj_weight)
                        if getattr(child, "in_proj_bias", None) is not None:
                            te_mha.in_proj_bias.copy_(child.in_proj_bias)
                        te_mha.out_proj.weight.copy_(child.out_proj.weight)
                        if (
                            child.out_proj.bias is not None
                            and te_mha.out_proj.bias is not None
                        ):
                            te_mha.out_proj.bias.copy_(child.out_proj.bias)
                    setattr(root, fqname, te_mha)
                    n_swapped += 1
                    replaced = True
            except Exception:
                pass
            if (
                not replaced
                and hasattr(child, "_sdpa")
                and hasattr(child, "nhead")
                and hasattr(child, "head_dim")
            ):
                nhead = int(getattr(child, "nhead"))
                head_dim = int(getattr(child, "head_dim"))
                drop_p = 0.0
                if hasattr(child, "drop") and hasattr(child.dropout, "p"):
                    try:
                        drop_p = float(child.dropout.p)
                    except Exception:
                        drop_p = 0.0
                try:
                    te_dpa = te.DotProductAttention(
                        num_attention_heads=nhead,
                        kv_channels=head_dim,
                        qkv_format="bshd",
                        attention_dropout=drop_p,
                    )
                except TypeError:
                    te_dpa = te.DotProductAttention(
                        nhead,
                        head_dim,
                        qkv_format="bshd",
                        attention_dropout=drop_p,
                    )
                if not hasattr(child, "__sdpa_pt__"):
                    child.__sdpa_pt__ = child._sdpa

                def _te_sdpa(
                    self,
                    q: Any,
                    k: Any,
                    v: Any,
                    p: Any,
                    rope_meta: Any = None,
                    _te_dpa: Any = te_dpa,
                ) -> Any:
                    if rope_meta is not None:
                        cos, sin = rope_meta
                        q = self._apply_rope(q, cos, sin)
                        k = self._apply_rope(k, cos, sin)
                    qs, ks, vs = (
                        q.contiguous().clone(),
                        k.contiguous().clone(),
                        v.contiguous().clone(),
                    )
                    try:
                        out = _te_dpa(
                            qs,
                            ks,
                            vs,
                            attention_mask=None,
                            qkv_format="bshd",
                            attn_mask_type="no_mask",
                            window_size=(-1, -1),
                        )
                    except Exception:
                        return self.__sdpa_pt__(q, k, v, p, rope_meta)
                    if out.dim() == 3:
                        B, S, Hd = out.shape
                        H, d = (int(self.nhead), int(self.head_dim))
                        out = out.view(B, S, H, d)
                    return out.contiguous()

                child._sdpa = _te_sdpa.__get__(child, type(child))
                n_swapped += 1
            if (
                not replaced
                and (not hasattr(child, "_sdpa"))
                and all(
                    (
                        hasattr(child, a)
                        for a in (
                            "nhead",
                            "head_dim",
                            "qkv",
                            "proj",
                            "norm1",
                            "norm2",
                            "forward",
                        )
                    )
                )
                and (not hasattr(child, "__forward_pt__"))
            ):
                try:
                    p_drop = 0.0
                    if hasattr(child, "drop") and hasattr(child, "p"):
                        try:
                            p_drop = float(child.dropout.p)
                        except Exception:
                            p_drop = 0.0
                    te_dpa = te.DotProductAttention(
                        num_attention_heads=int(child.nhead),
                        kv_channels=int(child.head_dim),
                        qkv_format="bshd",
                        attention_dropout=p_drop,
                    )
                    setattr(child, "_te_dpa", te_dpa)
                    child.__forward_pt__ = child.forward

                    def _te_forward(
                        self, x: Any, *args: Any, **kwargs: Any
                    ) -> Any:
                        B, S, D = x.shape
                        H = int(self.nhead)
                        d = int(D // H)
                        h = self.norm1(x)
                        qkv = (
                            self.qkv(h)
                            .view(B, S, 3, H, d)
                            .permute(2, 0, 3, 1, 4)
                        )
                        q, k, v = (qkv[0], qkv[1], qkv[2])
                        try:
                            q_bshd = q.transpose(1, 2).contiguous()
                            k_bshd = k.transpose(1, 2).contiguous()
                            v_bshd = v.transpose(1, 2).contiguous()
                            a_bshd = self._te_dpa(
                                q_bshd,
                                k_bshd,
                                v_bshd,
                                attention_mask=None,
                                qkv_format="bshd",
                                attn_mask_type="no_mask",
                                window_size=(-1, -1),
                            )
                            a = (
                                a_bshd.view(B, S, D)
                                if a_bshd.dim() == 3
                                else a_bshd.transpose(1, 2)
                                .contiguous()
                                .view(B, S, D)
                            )
                        except Exception:
                            return self.__forward_pt__(x, *args, **kwargs)
                        x2 = x + self.dropout(self.proj(a))
                        h2 = self.ffn(self.norm2(x2))
                        return x2 + self.dropout(h2)

                    child.forward = _te_forward.__get__(child, type(child))
                    n_swapped += 1
                    replaced = True
                except Exception:
                    pass
        return (root, n_swapped)

    @staticmethod
    def _fuse_sequential_to_te(
        root: nn.Module,
        *args: Any,
        params_dtype: torch.dtype,
        **kwargs: Any,
    ) -> Tuple[nn.Module, int]:
        try:
            import transformer_engine.pytorch as te
        except Exception:
            return (root, 0)
        n_swapped = 0
        for name, child in list(root.named_children()):
            new_child, k = ModuleTuner._fuse_sequential_to_te(
                child, params_dtype=params_dtype
            )
            if k > 0:
                setattr(root, name, new_child)
                n_swapped += k
                child = new_child
            if not isinstance(child, nn.Sequential) or len(child) < 2:
                continue
            i = 0
            new_seq = []
            changed = False
            while i < len(child):
                cur = child[i]
                nxt = child[i + 1] if i + 1 < len(child) else None
                nxt2 = child[i + 2] if i + 2 < len(child) else None
                nxt3 = child[i + 3] if i + 3 < len(child) else None
                if (
                    isinstance(cur, nn.LayerNorm)
                    or cur.__class__.__name__ == "RMSNorm"
                ) and isinstance(nxt, nn.Linear):
                    ln = cur
                    fc = nxt
                    act_name = (
                        "gelu"
                        if isinstance(nxt2, nn.GELU)
                        else "relu"
                        if isinstance(nxt2, nn.ReLU)
                        else None
                    )
                    if act_name is not None and isinstance(nxt3, nn.Linear):
                        hidden = (
                            int(ln.normalized_shape[0])
                            if isinstance(ln, nn.LayerNorm)
                            else int(getattr(ln, "weight").shape[0])
                        )
                        mlp = te.LayerNormMLP(
                            hidden,
                            int(fc.out_features),
                            eps=float(getattr(ln, "eps", 1e-05)),
                            bias=fc.bias is not None and nxt3.bias is not None,
                            normalization="LayerNorm"
                            if isinstance(ln, nn.LayerNorm)
                            else "RMSNorm",
                            activation=act_name,
                            params_dtype=params_dtype,
                        )
                        try:
                            with torch.no_grad():
                                if isinstance(ln, nn.LayerNorm):
                                    if hasattr(mlp, "layernorm"):
                                        if ln.weight is not None and hasattr(
                                            mlp.layernorm, "weight"
                                        ):
                                            mlp.layernorm.weight.copy_(
                                                ln.weight
                                            )
                                        if ln.bias is not None and hasattr(
                                            mlp.layernorm, "bias"
                                        ):
                                            mlp.layernorm.bias.copy_(ln.bias)
                                elif hasattr(mlp, "layernorm") and hasattr(
                                    ln, "weight"
                                ):
                                    mlp.layernorm.weight.copy_(ln.weight)
                                if hasattr(mlp, "fc1") and hasattr(
                                    fc, "weight"
                                ):
                                    mlp.fc1.weight.copy_(fc.weight)
                                    if (
                                        fc.bias is not None
                                        and hasattr(mlp.fc1, "bias")
                                        and (mlp.fc1.bias is not None)
                                    ):
                                        mlp.fc1.bias.copy_(fc.bias)
                                if (
                                    hasattr(mlp, "fc2")
                                    and isinstance(nxt3, nn.Linear)
                                    and hasattr(nxt3, "weight")
                                ):
                                    mlp.fc2.weight.copy_(nxt3.weight)
                                    if (
                                        nxt3.bias is not None
                                        and hasattr(mlp.fc2, "bias")
                                        and (mlp.fc2.bias is not None)
                                    ):
                                        mlp.fc2.bias.copy_(nxt3.bias)
                        except Exception:
                            pass
                        new_seq.append(mlp)
                        i += 4
                        changed = True
                        n_swapped += 1
                        continue
                    hidden = (
                        int(ln.normalized_shape[0])
                        if isinstance(ln, nn.LayerNorm)
                        else int(getattr(ln, "weight").shape[0])
                    )
                    lnlin = te.LayerNormLinear(
                        int(fc.in_features),
                        int(fc.out_features),
                        eps=float(getattr(ln, "eps", 1e-05)),
                        bias=fc.bias is not None,
                        normalization="LayerNorm"
                        if isinstance(ln, nn.LayerNorm)
                        else "RMSNorm",
                        params_dtype=params_dtype,
                    )
                    try:
                        with torch.no_grad():
                            if isinstance(ln, nn.LayerNorm):
                                if (
                                    ln.weight is not None
                                    and hasattr(lnlin, "layernorm")
                                    and hasattr(lnlin.layernorm, "weight")
                                ):
                                    lnlin.layernorm.weight.copy_(ln.weight)
                                if ln.bias is not None and hasattr(
                                    lnlin.layernorm, "bias"
                                ):
                                    lnlin.layernorm.bias.copy_(ln.bias)
                            elif hasattr(lnlin, "layernorm") and hasattr(
                                ln, "weight"
                            ):
                                lnlin.layernorm.weight.copy_(ln.weight)
                            if hasattr(lnlin, "linear") and hasattr(
                                lnlin.linear, "weight"
                            ):
                                lnlin.linear.weight.copy_(fc.weight)
                                if (
                                    fc.bias is not None
                                    and hasattr(lnlin.linear, "bias")
                                    and (lnlin.linear.bias is not None)
                                ):
                                    lnlin.linear.bias.copy_(fc.bias)
                    except Exception:
                        pass
                    new_seq.append(lnlin)
                    i += 2
                    changed = True
                    n_swapped += 1
                    continue
                new_seq.append(cur)
                i += 1
            if changed:
                setattr(root, name, nn.Sequential(*new_seq))
        return (root, n_swapped)

    @staticmethod
    def use_te_module(
        model: nn.Module,
        device: Optional[Union[torch.device, str]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[nn.Module, bool, str]:
        dev = torch.device(device) if device is not None else get_device()
        if dev.type != "cuda":
            return (model, False, "Non-NVIDIA device; TE not applied")
        try:
            import transformer_engine.pytorch as te
        except Exception:
            return (model, False, "transformer_engine not installed")
        te_backend = getattr(te, "__name__", "transformer_engine.pytorch")
        fp8_ok, why = is_float8_supported(dev)
        if fp8_ok:
            setattr(model, "__te_fp8_default__", True)
        params_dtype = ModuleTuner._infer_optimal_dtype(dev)
        model, n_fused = ModuleTuner._fuse_sequential_to_te(
            model, params_dtype=params_dtype
        )
        model, n_basic = ModuleTuner._apply_te_module(
            model,
            apply_te_linear=True,
            apply_te_layer_norm=True,
            apply_te_rms_norm=True,
            filter_linear=None,
            params_dtype=params_dtype,
        )
        try:
            model, attn_swapped = ModuleTuner._apply_te_attention(
                model, params_dtype=params_dtype
            )
        except Exception:
            attn_swapped = 0
        n_total = (n_fused or 0) + (n_basic or 0) + (attn_swapped or 0)
        if logger:
            logger(
                f"[TE] swapped {n_total} modules (fused:{n_fused}, basic:{n_basic}, attn:{attn_swapped}); params_dtype={str(params_dtype).split('.')[-1]}, fp8={('on' if fp8_ok else 'off')} ({(why if fp8_ok else '')}), backend={te_backend}"
            )
        return (
            model,
            n_total > 0,
            f"TE applied (swapped {n_total}, dtype={params_dtype}, fp8={('on' if fp8_ok else 'off')}, backend={te_backend})",
        )

    @staticmethod
    def enable_float8_training(
        model: nn.Module,
        device: Optional[Union[torch.device, str]] = None,
        logger: Optional[Callable[[str], None]] = None,
        prefer: str = "auto",
    ) -> Tuple[nn.Module, bool, str]:
        ok, reason = is_float8_supported(device)
        if not ok:
            return (model, False, reason)
        _dev_for_dtype = (
            torch.device(device) if device is not None else get_device()
        )
        _prefer_bf16 = (
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            if _dev_for_dtype.type != "cpu"
            else is_cpu_bf16_supported()
        )
        params_dtype = (
            torch.bfloat16
            if _prefer_bf16
            else torch.float16
            if _dev_for_dtype.type == "cuda"
            else torch.float32
        )

        def _try_te() -> Any:
            try:
                swapped_model, n = ModuleTuner._apply_te_module(
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
            except Exception as e:
                return (model, False, f"TE swap failed: {e}")

        def _try_ao() -> Any:
            try:
                from torchao.float8 import convert_to_float8_training

                res = convert_to_float8_training(model)
                m2 = res or model
                setattr(m2, "__fp8_training_ao__", True)
                if logger:
                    logger("[FP8][AO] convert_to_float8_training ok")
                return (m2, True, "torchao.float8")
            except Exception as e:
                return (model, False, f"torchao convert failed: {e}")

        order = (
            ("te", "torchao")
            if prefer in ("auto", "te")
            else ("torchao", "te")
        )
        for backend in order:
            m2, ok2, why = _try_te() if backend == "te" else _try_ao()
            if ok2:
                if logger:
                    logger(f"[FP8] training enabled via {why} ({reason})")
                return (m2, True, why)
            elif logger:
                logger(f"[FP8] {backend} path skipped: {why}")
        return (model, False, "No usable FP8 backend")

    @staticmethod
    def enable_float8_prediction(
        model: nn.Module,
        device: Optional[Union[torch.device, str]] = None,
        logger: Optional[Callable[[str], None]] = None,
        dynamic_activations: bool = False,
        prefer: str = "auto",
        te_swap: bool = True,
    ) -> Tuple[nn.Module, bool, str]:
        ok, reason = is_float8_supported(device)
        if not ok:
            return (model, False, reason)
        _dev_for_dtype = (
            torch.device(device) if device is not None else get_device()
        )
        _prefer_bf16 = (
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            if _dev_for_dtype.type != "cpu"
            else is_cpu_bf16_supported()
        )
        params_dtype = (
            torch.bfloat16
            if _prefer_bf16
            else torch.float16
            if _dev_for_dtype.type == "cuda"
            else torch.float32
        )

        def _try_te_swap() -> Any:
            try:
                m2, n = ModuleTuner._apply_te_module(
                    model,
                    apply_te_linear=True,
                    apply_te_layer_norm=True,
                    apply_te_rms_norm=True,
                    filter_linear=lambda lyr, _: lyr.in_features % 16 == 0
                    and lyr.out_features % 16 == 0,
                    params_dtype=params_dtype,
                )
                if n > 0:
                    setattr(m2, "__fp8_inference_te__", True)
                    if logger:
                        logger(
                            f"[FP8][TE] swapped {n} modules; using te.fp8_autocast"
                        )
                    return (m2, True, f"TE swap ({n})")
                return (model, False, "no eligible Linear (dims%16)")
            except Exception as e:
                return (model, False, f"TE swap failed: {e}")

        def _try_te_present() -> Any:
            te_present = any(
                (
                    getattr(m.__class__, "__module__", "").startswith(
                        "transformer_engine"
                    )
                    for m in model.modules()
                )
            )
            if te_present:
                setattr(model, "__fp8_inference_te__", True)
                if logger:
                    logger(
                        "[FP8][TE] te.* already present; using te.fp8_autocast"
                    )
                return (model, True, "TE present")
            return (model, False, "TE layers not present")

        def _try_ao() -> Any:
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
                if logger:
                    logger(f"[FP8][AO] applied {cfg.__class__.__name__}")
                return (model, True, "torchao")
            except Exception as e:
                return (model, False, f"AO failed: {e}")

        order = (
            ("te_swap", "te_present", "ao")
            if prefer in ("auto", "te")
            else ("ao", "te_present", "te_swap")
        )
        for step in order:
            if step == "te_swap" and (not te_swap):
                continue
            m2, ok2, why = {
                "te_swap": _try_te_swap,
                "te_present": _try_te_present,
                "ao": _try_ao,
            }[step]()
            if ok2:
                if logger:
                    logger(f"[FP8] inference enabled via {why} ({reason})")
                return (m2, True, why)
            elif logger:
                logger(f"[FP8] {step} skipped: {why}")
        return (model, False, "No usable FP8 backend")

    @staticmethod
    def enable_int8_training(
        model: nn.Module,
        device: Optional[Union[torch.device, str]] = None,
        logger: Optional[Callable[[str], None]] = None,
        dynamic_activations: bool = True,
        group_size: int = 128,
        prefer: str = "auto",
    ) -> Tuple[nn.Module, bool, str]:
        if quantize_ is None:
            msg = "torchao.quantization not installed (INT8/QAT disabled)"
            if logger:
                logger(f"[INT8] {msg}")
            return (model, False, msg)
        target_device = torch.device(device) if device is not None else None
        if target_device is not None:
            with contextlib.suppress(Exception):
                model.to(target_device)
        prefer_norm = str(prefer).strip().lower()
        use_qat = prefer_norm in ("auto", "qat")
        use_ptq = prefer_norm in ("auto", "ptq")
        if use_qat:
            try:
                base_cfg = QAT.initialize(
                    model,
                    mode="qat-int8",
                    dynamic_activations=bool(dynamic_activations),
                    group_size=int(group_size),
                    logger=logger,
                )
                setattr(model, "__int8_training_qat__", True)
                if logger:
                    logger(
                        f"[INT8][QAT] prepared with base {base_cfg.__class__.__name__}"
                    )
                return (model, True, "QAT-prepare")
            except Exception as e:
                if logger:
                    logger(f"[INT8][QAT] prepare failed: {e}")
                last_err = e
        else:
            last_err = RuntimeError("QAT disabled by prefer")
        if use_ptq:
            try:
                m2, ok, why = ptq(
                    model,
                    mode="int8",
                    dynamic_activations=bool(dynamic_activations),
                    group_size=int(group_size),
                    logger=logger,
                )
                if ok:
                    setattr(m2, "__int8_training_ptq__", True)
                    return (m2, True, f"PTQ({why})")
                return (model, False, f"PTQ failed: {why}")
            except Exception as e:
                return (model, False, f"INT8 training path unavailable: {e}")
        return (model, False, f"INT8 training disabled: {last_err}")

    @staticmethod
    def enable_int8_prediction(
        model: nn.Module,
        device: Optional[Union[torch.device, str]] = None,
        logger: Optional[Callable[[str], None]] = None,
        dynamic_activations: bool = True,
        prefer: str = "auto",
    ) -> Tuple[nn.Module, bool, str]:
        if quantize_ is None:
            msg = "torchao.quantization not installed (INT8 disabled)"
            if logger:
                logger(f"[INT8] {msg}")
            return (model, False, msg)
        prefer_norm = str(prefer).strip().lower()
        if prefer_norm not in ("auto", "ao"):
            return (
                model,
                False,
                f"INT8 inference prefer={prefer} not supported",
            )
        target_device = torch.device(device) if device is not None else None
        if target_device is not None:
            with contextlib.suppress(Exception):
                model.to(target_device)
        try:
            if dynamic_activations:
                cfg = Int8DynamicActivationInt8WeightConfig()
            else:
                cfg = Int8WeightOnlyConfig()
            quantize_(model, cfg)
            setattr(model, "__int8_inference_ao__", True)
            if logger:
                logger(f"[INT8][AO] applied {cfg.__class__.__name__}")
            return (model, True, "torchao")
        except Exception as e:
            return (model, False, f"AO failed: {e}")
