# -*- coding: utf-8 -*-
from __future__ import annotations
import contextlib
import importlib
import logging
import os
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    Union,
)

import torch
import torch._dynamo
from torch import nn, optim

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
from .compat import patch_torch
from .profiler import FLOP_PROFILER, FlopCounter, attention_flops_bshd

patch_torch()

try:
    from torch.distributed.algorithms.join import Join as _TorchJoin
except ImportError:
    _TorchJoin = None

Join: type[AbstractContextManager[None]] | None = _TorchJoin

if TYPE_CHECKING:
    from torch.distributed._composable.fsdp import FSDPModule
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.nn.parallel import DistributedDataParallel as DDP
else:
    DDP = object
    FSDP = object
    FSDPModule = object

JoinableModel: TypeAlias = Union["DDP", "FSDP", "FSDPModule"]


def _duplicate_last_dim(x: torch.Tensor) -> torch.Tensor:
    return torch.stack((x, x), dim=-1).reshape(*x.shape[:-1], -1)


def _retention_manual_flops(
    batch: int,
    seq_len: int,
    *,
    num_heads: int,
    head_dim: int,
    use_gate: bool,
) -> float:
    if batch <= 0 or seq_len <= 0 or num_heads <= 0 or head_dim <= 0:
        return 0.0
    attn = float(batch) * float(seq_len) * float(num_heads) * float(head_dim)
    gate_cost = attn if use_gate else 0.0
    return 4.0 * attn + gate_cost


def _is_contiguous_bshd(tensor: torch.Tensor) -> bool:
    if tensor.dim() != 4:
        return False
    _, seq_len, num_heads, head_dim = tensor.shape
    stride = tensor.stride()
    return (
        tensor.is_contiguous()
        and stride[-1] == 1
        and stride[-2] == head_dim
        and stride[-3] == num_heads * head_dim
        and stride[-4] == seq_len * num_heads * head_dim
    )


def is_transformer_engine_enabled(model: torch.nn.Module) -> bool:
    te_flags = (
        getattr(model, "__fp8_inference_te__", False),
        getattr(model, "__fp8_training_te__", False),
        getattr(model, "__te_fp8_default__", False),
    )
    if any(te_flags):
        return True

    for module in model.modules():
        mod_name = getattr(module.__class__, "__module__", "")
        if isinstance(mod_name, str) and mod_name.startswith("transformer_engine"):
            return True
    return False


def _is_inference_compiled(model: torch.nn.Module) -> bool:
    compile_attrs = (
        "_is_compiled_for_inference",
        "__is_compiled_for_inference__",
        "__compiled_for_serving__",
        "__serving_compiled__",
        "_is_serialized_for_serving",
    )
    if any(bool(getattr(model, attr, False)) for attr in compile_attrs):
        return True

    jit = getattr(torch, "jit", None)
    script_like_types: List[type] = []
    if jit is not None:
        for name in ("ScriptModule", "RecursiveScriptModule", "TopLevelTracedModule"):
            typ = getattr(jit, name, None)
            if isinstance(typ, type):
                script_like_types.append(typ)
        for mod_name in ("_script", "_trace"):
            submod = getattr(jit, mod_name, None)
            if submod is None:
                continue
            for name in ("RecursiveScriptModule", "TopLevelTracedModule"):
                typ = getattr(submod, name, None)
                if isinstance(typ, type):
                    script_like_types.append(typ)

    if any(isinstance(model, typ) for typ in script_like_types):
        return True

    try:
        modules = tuple(model.modules())
    except Exception:
        modules = ()

    for module in modules:
        if module is model:
            continue
        if any(bool(getattr(module, attr, False)) for attr in compile_attrs):
            return True
        if any(isinstance(module, typ) for typ in script_like_types):
            return True
    return False


def _is_aot_autograd_enabled(model: torch.nn.Module) -> bool:
    indicator_attrs = (
        "_aot_autograd_graph",
        "_aot_autograd_cache",
        "_aot_compiled_autograd",
        "_aot_autograd_traced_module",
        "__aot_autograd__",
        "__compiled_with_aot_autograd__",
    )
    if any(getattr(model, attr, None) for attr in indicator_attrs):
        return True

    try:
        modules = tuple(model.modules())
    except Exception:
        modules = ()

    for module in modules:
        if module is model:
            continue
        if any(getattr(module, attr, None) for attr in indicator_attrs):
            return True
        class_name = module.__class__.__name__
        module_name = getattr(module.__class__, "__module__", "")
        if "AOTAutograd" in class_name or "aot_autograd" in module_name:
            return True
    return False


def inference(model: torch.nn.Module) -> AbstractContextManager[None]:
    if (
        is_transformer_engine_enabled(model)
        or _is_inference_compiled(model)
        or _is_aot_autograd_enabled(model)
    ):
        return torch.no_grad()
    return torch.inference_mode()


def _has_join_hook(obj: Any | None) -> bool:
    if obj is None:
        return False
    return getattr(obj, "join_hook", None) is not None

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


def joining(
    model: JoinableModel,
    optimizer: optim.Optimizer | None = None,
) -> AbstractContextManager[None]:
    if Join is None:
        return contextlib.nullcontext()
    joinables = tuple(obj for obj in (model, optimizer) if _has_join_hook(obj))
    if not joinables:
        return contextlib.nullcontext()
    return Join(joinables, throw_on_early_termination=True)


_LOGGER = logging.getLogger(__name__)


class DotProductAttention(nn.Module):
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
        self.te_first = bool(cfg.te_first) if te_first is None else bool(te_first)
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
            with contextlib.ExitStack() as stack:
                import warnings as _warnings

                stack.enter_context(_warnings.catch_warnings())
                _warnings.filterwarnings(
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
        if _is_contiguous_bshd(q_bshd) and _is_contiguous_bshd(k_bshd):
            try:
                attention_flops_bshd(
                    q_bshd,
                    bwd_factor=2.0 if training else 0.0,
                    dropout_p=float(dropout_p),
                    training=training,
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
        q_bhsd = q_bshd
        k_bhsd = k_bshd
        v_bhsd = v_bshd
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
        return sdpa_out

    @staticmethod
    def _to_optimal_dtype(tensor: torch.Tensor) -> torch.Tensor:
        device_type = tensor.device.type
        if device_type == "cpu" and tensor.dtype in (torch.float16, torch.bfloat16):
            return tensor.float()
        if device_type == "mps" and tensor.dtype == torch.bfloat16:
            return tensor.to(torch.float16)
        return tensor


class MultiScaleRetentionCompat(nn.Module):
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
        del attn_mask, state, kwargs
        batch, seq_len, _ = x.shape
        head_dim = self.head_dim
        q = self.q_proj(x).view(batch, seq_len, self.nhead, head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.nhead, head_dim)
        manual_flops = _retention_manual_flops(
            batch,
            seq_len,
            num_heads=self.nhead,
            head_dim=head_dim,
            use_gate=self.use_gate and self.g_proj is not None,
        )
        lam = (
            torch.sigmoid(self._beta)
            .view(1, self.nhead, 1)
            .to(dtype=v.dtype, device=v.device)
        )
        prev = v[:, 0].clone()
        states = [prev]
        for index in range(1, seq_len):
            prev = lam * prev + v[:, index]
            states.append(prev)
        state_tensor = torch.stack(states, dim=1).contiguous()
        y = (q * state_tensor).contiguous().view(batch, seq_len, self.d_model)
        y = self.norm(y)
        if self.use_gate and self.g_proj is not None:
            gate = torch.nn.functional.silu(self.g_proj(x))
            y = y * gate
        if manual_flops > 0.0:
            FLOP_PROFILER.add_manual("Retention", manual_flops)
        return self.o_proj(y)


class MultiScaleRetention(nn.Module):
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
                MultiScaleRetention as _TorchScaleMSR,
            )

            self._ts_msr = _TorchScaleMSR(self.d_model, self.nhead)
            self._ts_key_dim = int(
                getattr(self._ts_msr, "key_dim", self.d_model // self.nhead)
            )
            self._ts_ok = True
        except Exception:
            self._ts_ok = False
            self._fallback = MultiScaleRetentionCompat(
                self.d_model, self.nhead, use_gate=self.use_gate
            )
        self._rope_theta = 10000.0
        self._decay_init = 5.0
        self._decay_range = 1.0

    def _build_rel_pos(self, seq_len: Any, device: Any, dtype: Any) -> Any:
        key_dim = int(
            self._ts_key_dim
            if getattr(self, "_ts_key_dim", None) is not None
            else self.d_model // self.nhead
        )
        half = key_dim // 2
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_freq = 1.0 / self._rope_theta ** torch.linspace(
            0, 1, half, device=device, dtype=torch.float32
        )
        freqs = torch.einsum("n,d->nd", positions, inv_freq)
        sin = _duplicate_last_dim(torch.sin(freqs)).to(dtype)[None, None, :, :]
        cos = _duplicate_last_dim(torch.cos(freqs)).to(dtype)[None, None, :, :]
        length = seq_len
        idx_i = torch.arange(length, device=device)
        idx_j = torch.arange(length, device=device)
        diff = (idx_i[:, None] - idx_j[None, :]).to(dtype)
        tril = (idx_i[:, None] >= idx_j[None, :]).to(dtype)
        heads = torch.arange(self.nhead, device=device, dtype=dtype)
        gammas = 1.0 - torch.pow(
            2.0,
            -(
                self._decay_init
                + self._decay_range * (heads / max(self.nhead, 1))
            ),
        )
        gammas = torch.clamp(
            gammas, min=torch.finfo(dtype).tiny, max=1 - 1e-09
        )
        inner_mask = torch.pow(
            gammas.view(1, self.nhead, 1, 1), diff.view(1, 1, length, length)
        ) * tril.view(1, 1, length, length)
        return ((sin, cos), inner_mask)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        state: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del attn_mask, kwargs
        try:
            batch, seq_len, dim = x.shape
        except ValueError:
            manual_flops = 0.0
        else:
            manual_flops = _retention_manual_flops(
                batch,
                seq_len,
                num_heads=self.nhead,
                head_dim=max(1, dim // max(self.nhead, 1)),
                use_gate=self.use_gate,
            )
        if self._ts_ok:
            _, seq_len, _ = x.shape
            rel_pos = self._build_rel_pos(seq_len, x.device, x.dtype)
            out = self._ts_msr(
                x, rel_pos, chunkwise_recurrent=False, incremental_state=state
            )
            if manual_flops > 0.0:
                FLOP_PROFILER.add_manual("Retention", manual_flops)
            return out
        return self._fallback(x, attn_mask=None, state=state)


class AdamW:
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
            except Exception as exc:
                if logger:
                    logger(f"[OPT] TE FusedAdam unavailable: {exc}")
        if use_fp8 and (hasattr(dev, "type") and dev.type == "cuda"):
            ok, reason = is_float8_supported(dev)
            if "TE" in str(reason):
                try:
                    from transformer_engine.pytorch.optimizers import FusedAdam

                    opt = FusedAdam(params, lr=lr, weight_decay=weight_decay)
                    if logger:
                        logger(
                            f"[OPT] Using FusedAdam (transformer_engine) — {reason}"
                        )
                    return opt
                except Exception as exc:
                    if logger:
                        logger(
                            f"[OPT] transformer_engine.FusedAdam unavailable: {exc}"
                        )
            if "AO" in str(reason):
                try:
                    from torchao.optim import AdamWFp8

                    opt = AdamWFp8(params, lr=lr, weight_decay=weight_decay)
                    if logger:
                        logger(f"[OPT] Using AdamW-FP8 (torchao) — {reason}")
                    return opt
                except Exception as exc:
                    if logger:
                        logger(f"[OPT] torchao.AdamWFp8 unavailable: {exc}")
            elif logger:
                logger(f"[OPT] FP8 optimizers not supported ({reason}) — fallback")
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
            except Exception as exc:
                if logger:
                    logger(f"[OPT] TE FusedAdam unavailable: {exc}")
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
            except Exception as exc:
                if logger:
                    logger(f"[OPT] TorchAO low-bit optimizer unavailable: {exc}")
        flags: Dict[str, bool] = optimal_optimizer_params(
            dev, use_foreach=use_foreach, use_fused=use_fused
        )
        opt = optim.AdamW(params, lr=lr, weight_decay=weight_decay, **flags)
        if logger:
            logger(f"[OPT] Using AdamW (flags={flags})")
        return opt


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
        params_dtype = Module._infer_optimal_dtype(dev)
        model, n_fused = Module._fuse_sequential_to_te(
            model, params_dtype=params_dtype
        )
        model, n_basic = Module._apply_te_module(
            model,
            apply_te_linear=True,
            apply_te_layer_norm=True,
            apply_te_rms_norm=True,
            filter_linear=None,
            params_dtype=params_dtype,
        )
        try:
            model, attn_swapped = Module._apply_te_attention(
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
    def _try_enable_te_training(
        model: nn.Module,
        params_dtype: torch.dtype,
        logger: Optional[Callable[[str], None]],
    ) -> Tuple[nn.Module, bool, str]:
        try:
            swapped_model, n = Module._apply_te_module(
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
    def _try_enable_ao_training(
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
    def _try_enable_te_inference_swap(
        model: nn.Module,
        params_dtype: torch.dtype,
        logger: Optional[Callable[[str], None]],
    ) -> Tuple[nn.Module, bool, str]:
        try:
            swapped, n = Module._apply_te_module(
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
                    logger(
                        f"[FP8][TE] swapped {n} modules; using te.fp8_autocast"
                    )
                return (swapped, True, f"TE swap ({n})")
            return (model, False, "no eligible Linear (dims%16)")
        except Exception as exc:
            return (model, False, f"TE swap failed: {exc}")

    @staticmethod
    def _try_use_existing_te(
        model: nn.Module,
        logger: Optional[Callable[[str], None]],
    ) -> Tuple[nn.Module, bool, str]:
        te_present = any(
            (
                getattr(module.__class__, "__module__", "").startswith(
                    "transformer_engine"
                )
                for module in model.modules()
            )
        )
        if te_present:
            setattr(model, "__fp8_inference_te__", True)
            if logger:
                logger("[FP8][TE] te.* already present; using te.fp8_autocast")
            return (model, True, "TE present")
        return (model, False, "TE layers not present")

    @staticmethod
    def _try_enable_ao_inference(
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
            if logger:
                logger(f"[FP8][AO] applied {cfg.__class__.__name__}")
            return (model, True, "torchao")
        except Exception as exc:
            return (model, False, f"AO failed: {exc}")

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

        order = (
            ("te", "torchao")
            if prefer in ("auto", "te")
            else ("torchao", "te")
        )
        for backend in order:
            if backend == "te":
                m2, ok2, why = Module._try_enable_te_training(
                    model, params_dtype, logger
                )
            else:
                m2, ok2, why = Module._try_enable_ao_training(model, logger)
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

        order = (
            ("te_swap", "te_present", "ao")
            if prefer in ("auto", "te")
            else ("ao", "te_present", "te_swap")
        )
        for step in order:
            if step == "te_swap" and (not te_swap):
                continue
            if step == "te_swap":
                m2, ok2, why = Module._try_enable_te_inference_swap(
                    model, params_dtype, logger
                )
            elif step == "te_present":
                m2, ok2, why = Module._try_use_existing_te(model, logger)
            else:
                m2, ok2, why = Module._try_enable_ao_inference(
                    model, dynamic_activations, logger
                )
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
    default_module = f"{root_pkg}.utils.optimization"
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
