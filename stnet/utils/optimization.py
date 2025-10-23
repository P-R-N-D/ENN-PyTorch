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

from .platform import (
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
from .profiler import FLOP_PROFILER, attention_flops_bshd

patch_torch()

_LOGGER = logging.getLogger(__name__)


class AutoCast:
    _fp8_backend: Optional[str] = None
    _int_backend: Optional[str] = None
    _last_float_dtype: torch.dtype = torch.float32
    _last_int_dtype: torch.dtype = torch.int64

    @classmethod
    def _resolve_fp8_backend(
        cls,
        preferred: Optional[str],
        *,
        device: Optional[torch.device] = None,
    ) -> Optional[str]:
        dev = device if device is not None else cls._resolve_device(None)
        order: Tuple[str, ...]
        if preferred == "te":
            order = ("te", "ao")
        elif preferred == "ao":
            order = ("ao", "te")
        else:
            order = ("te", "ao")
        for backend in order:
            if backend == "te":
                ok, reason = is_float8_supported(dev)
                if not ok:
                    _LOGGER.debug("AutoCast FP8 TE unavailable: %s", reason)
                    continue
                try:
                    te = importlib.import_module("transformer_engine.pytorch")

                    if getattr(te, "fp8_autocast", None) is None:
                        raise AttributeError(
                            "transformer_engine.fp8_autocast missing"
                        )
                except Exception as exc:
                    _LOGGER.debug("AutoCast FP8 TE import failed: %s", exc)
                    continue
                cls._fp8_backend = "te"
                return "te"
            if backend == "ao":
                try:
                    _float8_mod = importlib.import_module("torchao.float8")

                    if getattr(_float8_mod, "fp8_autocast", None) is None:
                        raise AttributeError("torchao.float8.fp8_autocast missing")
                except Exception as exc:
                    _LOGGER.debug("AutoCast FP8 torchao import failed: %s", exc)
                    continue
                cls._fp8_backend = "ao"
                return "ao"
        cls._fp8_backend = None
        return None

    @classmethod
    def _resolve_int_backend(
        cls,
        preferred: Optional[str],
        *,
        device: Optional[torch.device] = None,
    ) -> Optional[str]:
        dev = device if device is not None else cls._resolve_device(None)
        order: Tuple[str, ...]
        if preferred == "te":
            order = ("te", "ao")
        elif preferred == "ao":
            order = ("ao", "te")
        else:
            order = ("te", "ao")
        for backend in order:
            if backend == "te":
                ok, reason = is_int8_supported(dev)
                if not ok:
                    _LOGGER.debug("AutoCast INT8 TE unavailable: %s", reason)
                    continue
                try:
                    te = importlib.import_module("transformer_engine.pytorch")

                    if getattr(te, "int8_autocast", None) is None:
                        raise AttributeError(
                            "transformer_engine.int8_autocast missing"
                        )
                except Exception as exc:
                    _LOGGER.debug("AutoCast INT8 TE import failed: %s", exc)
                    continue
                cls._int_backend = "te"
                return "te"
            if backend == "ao":
                try:
                    quant_mod = importlib.import_module("torchao.quantization")
                    int8_autocast = getattr(quant_mod, "int8_autocast", None)

                    if not callable(int8_autocast):
                        raise AttributeError("torchao.quantization.int8_autocast missing")
                except Exception as exc:
                    _LOGGER.debug("AutoCast INT8 torchao import failed: %s", exc)
                    continue
                cls._int_backend = "ao"
                return "ao"
        cls._int_backend = None
        return None

    @staticmethod
    def _resolve_device(
        device: Optional[Union[torch.device, str]] = None,
    ) -> torch.device:
        if device is None:
            return get_device()
        if isinstance(device, torch.device):
            return device
        return torch.device(device)

    @staticmethod
    def _coerce_dtype(value: Optional[Union[str, torch.dtype]]) -> Optional[torch.dtype]:
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            candidate = getattr(torch, value, None)
            if isinstance(candidate, torch.dtype):
                return candidate
        return None

    @staticmethod
    def _distinct_dtypes(candidates: Iterable[torch.dtype]) -> Tuple[torch.dtype, ...]:
        seen: Dict[torch.dtype, None] = {}
        for dtype in candidates:
            if isinstance(dtype, torch.dtype) and dtype not in seen:
                seen[dtype] = None
        return tuple(seen.keys())

    @classmethod
    def _float_amp_candidates(cls, device: torch.device) -> Tuple[torch.dtype, ...]:
        dev_type = device.type
        candidates: List[torch.dtype] = []
        if dev_type == "cuda":
            if torch.cuda.is_bf16_supported():
                candidates.append(torch.bfloat16)
            candidates.append(torch.float16)
            candidates.append(torch.float32)
        elif dev_type == "xpu":
            candidates.extend((torch.bfloat16, torch.float32))
        elif dev_type == "mps":
            candidates.extend((torch.float16, torch.float32))
        elif dev_type == "cpu":
            if is_cpu_bf16_supported():
                candidates.append(torch.bfloat16)
            candidates.extend((torch.float32, torch.float64))
        else:
            candidates.append(torch.float32)
        if not candidates:
            candidates.append(torch.float32)
        return cls._distinct_dtypes(candidates)

    @staticmethod
    def _float8_dtypes() -> Tuple[torch.dtype, ...]:
        names = (
            "float8_e4m3fn",
            "float8_e4m3fnuz",
            "float8_e5m2",
            "float8_e5m2fnuz",
        )
        values: List[torch.dtype] = []
        for name in names:
            candidate = getattr(torch, name, None)
            if isinstance(candidate, torch.dtype):
                values.append(candidate)
        return tuple(values)

    @classmethod
    def _integer_candidates(cls, device: torch.device) -> Tuple[torch.dtype, ...]:
        candidates: List[torch.dtype] = []
        int8_ok, _ = is_int8_supported(device)
        if int8_ok:
            candidates.append(torch.int8)
        candidates.extend((torch.int16, torch.int32, torch.int64))
        return cls._distinct_dtypes(candidates or (torch.int64,))

    @staticmethod
    def _select_dtype(
        requested: Optional[torch.dtype],
        candidates: Tuple[torch.dtype, ...],
        *,
        fallback: torch.dtype,
        logger: Optional[logging.Logger] = None,
        context: str = "autocast",
        device: Optional[torch.device] = None,
    ) -> torch.dtype:
        if requested is None:
            return candidates[0] if candidates else fallback
        if requested in candidates:
            return requested
        if logger is not None:
            device_str = f" on {device.type}" if device is not None else ""
            logger.debug(
                "AutoCast %s fallback%s: requested %s not available; using %s",
                context,
                device_str,
                str(requested).split(".")[-1],
                str(candidates[0] if candidates else fallback).split(".")[-1],
            )
        return candidates[0] if candidates else fallback

    @classmethod
    def _te_fp8_context(
        cls, device: torch.device, enabled: bool
    ) -> List[contextlib.AbstractContextManager[None]]:
        contexts: List[contextlib.AbstractContextManager[None]] = []
        if not enabled:
            return contexts
        try:
            te = importlib.import_module("transformer_engine.pytorch")

            fp8_ctx = getattr(te, "fp8_autocast", None)
            if callable(fp8_ctx):
                contexts.append(fp8_ctx(enabled=True))
            else:
                raise AttributeError("transformer_engine.fp8_autocast missing")
        except Exception as exc:
            _LOGGER.debug("AutoCast FP8 TE failed: %s", exc)
            cls._fp8_backend = None
        return contexts

    @classmethod
    def _ao_fp8_context(
        cls, enabled: bool
    ) -> List[contextlib.AbstractContextManager[None]]:
        contexts: List[contextlib.AbstractContextManager[None]] = []
        if not enabled:
            return contexts
        try:
            fp8_mod = importlib.import_module("torchao.float8")
            fp8_autocast = getattr(fp8_mod, "fp8_autocast", None)

            if callable(fp8_autocast):
                contexts.append(fp8_autocast(enabled=True))
            else:
                raise AttributeError("torchao.float8.fp8_autocast missing")
        except Exception as exc:
            _LOGGER.debug("AutoCast FP8 torchao failed: %s", exc)
            cls._fp8_backend = None
        return contexts

    @classmethod
    def _int8_context(
        cls,
        device: torch.device,
        enabled: bool,
    ) -> List[contextlib.AbstractContextManager[None]]:
        contexts: List[contextlib.AbstractContextManager[None]] = []
        if not enabled:
            return contexts
        backend = cls._int_backend
        if backend == "te":
            try:
                te = importlib.import_module("transformer_engine.pytorch")

                int_ctx = getattr(te, "int8_autocast", None)
                if callable(int_ctx):
                    contexts.append(int_ctx(enabled=True))
                else:
                    raise AttributeError("transformer_engine.int8_autocast missing")
            except Exception as exc:
                _LOGGER.debug("AutoCast INT8 TE failed: %s", exc)
                cls._int_backend = None
        elif backend == "ao":
            try:
                quant_mod = importlib.import_module("torchao.quantization")
                int8_autocast = getattr(quant_mod, "int8_autocast", None)

                if callable(int8_autocast):
                    contexts.append(int8_autocast(enabled=True))
                else:
                    raise AttributeError("torchao.quantization.int8_autocast missing")
            except Exception as exc:
                _LOGGER.debug("AutoCast INT8 torchao failed: %s", exc)
                cls._int_backend = None
        return contexts

    @classmethod
    def configure(cls, model: Optional[nn.Module]) -> None:
        backend: Optional[str] = None
        int_backend: Optional[str] = None
        if isinstance(model, nn.Module):
            if any(
                getattr(model, attr, False)
                for attr in ("__fp8_inference_te__", "__fp8_training_te__")
            ):
                backend = "te"
            elif any(
                getattr(model, attr, False)
                for attr in ("__fp8_inference_ao__", "__fp8_training_ao__")
            ):
                backend = "ao"
            if any(
                getattr(model, attr, False)
                for attr in (
                    "__int8_training_te__",
                    "__int8_inference_te__",
                    "__te_int8_default__",
                )
            ):
                int_backend = "te"
            elif any(
                getattr(model, attr, False)
                for attr in (
                    "__int8_training_qat__",
                    "__int8_training_ptq__",
                    "__int8_inference_ao__",
                )
            ):
                int_backend = "ao"
        cls._fp8_backend = backend
        cls._int_backend = int_backend

    @classmethod
    @contextlib.contextmanager
    def float(
        cls,
        device: Optional[Union[torch.device, str]] = None,
        *,
        dtype: Optional[torch.dtype] = None,
        enabled: Optional[bool] = None,
    ) -> contextlib.AbstractContextManager[None]:
        dev = cls._resolve_device(device)
        requested_dtype = cls._coerce_dtype(dtype)
        amp_candidates = cls._float_amp_candidates(dev)
        amp_dtype = cls._select_dtype(
            requested_dtype,
            amp_candidates,
            fallback=torch.float32,
            logger=_LOGGER,
            context="float",
            device=dev,
        )
        enabled = True if enabled is None else bool(enabled)
        contexts: List[contextlib.AbstractContextManager[None]] = []

        backend = cls._resolve_fp8_backend(cls._fp8_backend, device=dev)
        float8_dtypes = cls._float8_dtypes()
        wants_fp8 = False
        if backend is not None and enabled:
            if requested_dtype is None:
                wants_fp8 = True
            else:
                wants_fp8 = requested_dtype in float8_dtypes
        if wants_fp8:
            if backend == "te":
                contexts.extend(cls._te_fp8_context(dev, enabled))
                if not contexts:
                    backend = cls._resolve_fp8_backend("ao", device=dev)
                    if backend == "ao":
                        contexts.extend(cls._ao_fp8_context(enabled))
            elif backend == "ao":
                contexts.extend(cls._ao_fp8_context(enabled))
            else:
                _LOGGER.debug(
                    "AutoCast FP8 backend '%s' unsupported; disabling", backend
                )
                cls._fp8_backend = None

        try:
            ctx = torch.amp.autocast(
                device_type=dev.type,
                dtype=amp_dtype,
                enabled=enabled,
            )
            contexts.append(ctx)
        except (RuntimeError, ValueError) as exc:
            _LOGGER.debug(
                "AutoCast.float torch.amp fallback on %s: %s", dev.type, exc
            )
            contexts.append(contextlib.nullcontext())
            cls._last_float_dtype = amp_dtype
        else:
            cls._last_float_dtype = amp_dtype

        with contextlib.ExitStack() as stack:
            for ctx in contexts:
                stack.enter_context(ctx)
            yield

    @classmethod
    @contextlib.contextmanager
    def integer(
        cls,
        device: Optional[Union[torch.device, str]] = None,
        *,
        dtype: Optional[torch.dtype] = None,
        enabled: Optional[bool] = None,
    ) -> contextlib.AbstractContextManager[None]:
        dev = cls._resolve_device(device)
        requested_dtype = cls._coerce_dtype(dtype)
        int_candidates = cls._integer_candidates(dev)
        int_dtype = cls._select_dtype(
            requested_dtype,
            int_candidates,
            fallback=torch.int64,
            logger=_LOGGER,
            context="int",
            device=dev,
        )
        use_enabled = True if enabled is None else bool(enabled)
        backend = cls._resolve_int_backend(cls._int_backend, device=dev)
        contexts = (
            cls._int8_context(dev, use_enabled) if use_enabled and backend else []
        )
        if not contexts and use_enabled and backend == "te":
            fallback_backend = cls._resolve_int_backend("ao", device=dev)
            if fallback_backend == "ao":
                contexts = cls._int8_context(dev, use_enabled)
        if not contexts:
            contexts.append(contextlib.nullcontext())

        with contextlib.ExitStack() as stack:
            for ctx in contexts:
                stack.enter_context(ctx)
            cls._last_int_dtype = int_dtype
            yield


def compile(
    module: nn.Module,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    fullgraph: Optional[bool] = None,
    dynamic: Optional[bool] = None,
    options: Optional[Dict[str, Any]] = None,
    disable: bool = False,
) -> nn.Module:
    if disable:
        return module
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return module
    kwargs: Dict[str, Any] = {}
    if backend is not None:
        kwargs["backend"] = backend
    if mode is not None:
        kwargs["mode"] = mode
    if fullgraph is not None:
        kwargs["fullgraph"] = bool(fullgraph)
    if dynamic is not None:
        kwargs["dynamic"] = bool(dynamic)
    if options:
        kwargs["options"] = dict(options)
    try:
        return compile_fn(module, **kwargs)
    except Exception as exc:
        _LOGGER.warning("torch.compile failed (%s); returning original module", exc)
        return module


@contextlib.contextmanager
def no_synchronization(
    model: nn.Module,
    *,
    enable: bool = True,
) -> contextlib.AbstractContextManager[None]:
    if not enable:
        yield
        return
    ctx = None
    try:
        no_sync = getattr(model, "no_sync", None)
        if callable(no_sync):
            ctx = no_sync()
    except Exception:
        ctx = None
    if ctx is None:
        yield
        return
    with ctx:
        yield

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

Int8DynamicActivationInt8WeightConfig: Any | None
Int8WeightOnlyConfig: Any | None
quantize_: Any | None
ptq: Callable[..., tuple[nn.Module, bool, str]] | None
QAT: Any | None

try:
    from torchao.quantization import (
        Int8DynamicActivationInt8WeightConfig,
        Int8WeightOnlyConfig,
        quantize_,
    )
    ptq = getattr(quantize_, "ptq", None)
except ImportError:
    quantize_ = None
    Int8DynamicActivationInt8WeightConfig = None
    Int8WeightOnlyConfig = None
    ptq = None
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
try:
    from torchao.quantization import qat as _qat_module

    QAT = _qat_module if hasattr(_qat_module, "initialize") else None
except Exception:
    QAT = None


def _ptq_unavailable(
    model: nn.Module,
    *args: Any,
    **kwargs: Any,
) -> tuple[nn.Module, bool, str]:
    return (model, False, "PTQ backend unavailable")


if ptq is None:
    ptq = _ptq_unavailable


if QAT is None:

    class _QATUnavailable:
        @staticmethod
        def initialize(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("QAT backend unavailable")

    QAT = _QATUnavailable()


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


class Module:
    @staticmethod
    def _infer_optimal_dtype(
        device: Optional[Union[torch.device, str]] = None
    ) -> torch.dtype:
        dev = torch.device(device) if device is not None else get_device()
        if dev.type == "cuda":
            try:
                if is_cuda_bf16_supported(dev):
                    return torch.bfloat16
            except Exception:
                pass
            return torch.float16
        if dev.type == "cpu":
            return torch.bfloat16 if is_cpu_bf16_supported() else torch.float32
        if dev.type in {"xpu", "mps"}:
            return torch.bfloat16 if dev.type == "xpu" else torch.float16
        return torch.float32

    @staticmethod
    def _module_reference_tensor(module: nn.Module) -> Optional[torch.Tensor]:
        with contextlib.suppress(StopIteration):
            return next(module.parameters())
        with contextlib.suppress(StopIteration):
            return next(module.buffers())
        return None

    @staticmethod
    def _align_module_like(
        src: nn.Module,
        dst: nn.Module,
        params_dtype: Optional[torch.dtype],
    ) -> None:
        ref = Module._module_reference_tensor(src)
        if ref is not None:
            with contextlib.suppress(Exception):
                dst.to(device=ref.device)
        if params_dtype is not None:
            with contextlib.suppress(Exception):
                dst.to(dtype=params_dtype)

    @staticmethod
    def _copy_state(
        src: nn.Module, dst: nn.Module, params_dtype: Optional[torch.dtype]
    ) -> None:
        try:
            state = src.state_dict()
        except Exception:
            return
        ref = Module._module_reference_tensor(dst)
        device = ref.device if ref is not None else None
        converted = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                tensor = value.detach()
                if params_dtype is not None and tensor.is_floating_point():
                    tensor = tensor.to(dtype=params_dtype)
                if device is not None:
                    tensor = tensor.to(device=device)
                converted[key] = tensor
            else:
                converted[key] = value
        with contextlib.suppress(Exception):
            dst.load_state_dict(converted, strict=False)

    @staticmethod
    def _make_te_linear(
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
        Module._align_module_like(module, replacement, params_dtype)
        Module._copy_state(module, replacement, params_dtype)
        return replacement

    @staticmethod
    def _make_te_layer_norm(
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
        Module._align_module_like(module, replacement, params_dtype)
        if module.elementwise_affine:
            Module._copy_state(module, replacement, params_dtype)
        return replacement

    @staticmethod
    def _make_te_rms_norm(
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
        Module._align_module_like(module, replacement, params_dtype)
        Module._copy_state(module, replacement, params_dtype)
        return replacement

    @staticmethod
    def _fuse_sequential_to_te(
        model: nn.Module, *, params_dtype: Optional[torch.dtype]
    ) -> Tuple[nn.Module, int]:
        try:
            importlib.import_module("transformer_engine.pytorch")
        except Exception:
            return (model, 0)
        return (model, 0)

    @staticmethod
    def _apply_te_module(
        model: nn.Module,
        *,
        apply_te_linear: bool,
        apply_te_layer_norm: bool,
        apply_te_rms_norm: bool,
        filter_linear: Optional[Callable[[nn.Linear, str], bool]],
        params_dtype: Optional[torch.dtype],
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
                        replacement = Module._make_te_linear(
                            child, params_dtype, te
                        )
                elif apply_te_layer_norm and isinstance(child, nn.LayerNorm):
                    replacement = Module._make_te_layer_norm(
                        child, params_dtype, te
                    )
                else:
                    rms_cls = getattr(torch.nn, "RMSNorm", None)
                    if (
                        apply_te_rms_norm
                        and rms_cls is not None
                        and isinstance(child, rms_cls)
                    ):
                        replacement = Module._make_te_rms_norm(
                            child, params_dtype, te
                        )
                if replacement is not None:
                    setattr(parent, name, replacement)
                    converted += 1
                    continue
                converted += _convert(child)
            return converted

        count = _convert(model)
        return (model, count)

    @staticmethod
    def _apply_te_attention(
        model: nn.Module, *, params_dtype: Optional[torch.dtype]
    ) -> Tuple[nn.Module, int]:
        swapped = 0
        for module in model.modules():
            if isinstance(module, DotProductAttention) and getattr(
                module, "_te_ok", False
            ):
                if not getattr(module, "te_first", False):
                    module.te_first = True
                swapped += 1
        return (model, swapped)

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
            AutoCast.configure(model)
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
                AutoCast.configure(m2)
                return (m2, True, why)
            elif logger:
                logger(f"[FP8] {backend} path skipped: {why}")
        AutoCast.configure(model)
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
            AutoCast.configure(model)
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
                AutoCast.configure(m2)
                return (m2, True, why)
            elif logger:
                logger(f"[FP8] {step} skipped: {why}")
        AutoCast.configure(model)
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
