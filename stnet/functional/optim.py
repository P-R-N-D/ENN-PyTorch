from __future__ import annotations

import contextlib
import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
from torch import nn, optim

from ..data.stats import MetaData
from ..utils.platform import (
    get_device,
    is_cpu_bf16_supported,
    is_cuda_bf16_supported,
    is_float8_supported,
    is_int4_supported,
    is_int8_supported,
    optimal_optimizer_params,
)

from ..model.ops import DotProductAttention

__all__ = [
    "AdamW",
    "Module",
]

class _AutoCastProxy:
    def __getattr__(self, attr: str) -> Any:
        from ..kernels import AutoCast as _AutoCast
        return getattr(_AutoCast, attr)


AutoCast = _AutoCastProxy()


def _supports_scale(
    dtype: torch.dtype, meta: Optional[MetaData[Any]], *, safety_margin: float = 8.0
) -> bool:
    from ..kernels import _supports_scale as _kernel_supports_scale

    return _kernel_supports_scale(dtype, meta, safety_margin=safety_margin)


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


class AdamW:
    @staticmethod
    def float(
        model_or_params: Union[
            nn.Module, Iterable[nn.Parameter], Sequence[Dict[str, Any]]
        ],
        lr: float,
        *args: Any,
        weight_decay: float = 0.0,
        metadata: Optional[MetaData[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> optim.Optimizer:
        params = (
            model_or_params.parameters()
            if hasattr(model_or_params, "parameters")
            else model_or_params
        )
        ref_tensor: Optional[torch.Tensor] = None
        if isinstance(model_or_params, nn.Module):
            ref_tensor = Module._module_reference_tensor(model_or_params)
        dev: torch.device
        if metadata is not None:
            dev = torch.device(metadata.device)
        elif ref_tensor is not None:
            dev = ref_tensor.device
        else:
            dev = get_device()
        meta = AutoCast._ensure_metadata(dev, metadata=metadata)
        dev = torch.device(meta.device)
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
        if hasattr(dev, "type") and dev.type == "cuda":
            fp8_allowed = True
            if getattr(meta, "has_scale", False):
                float8_dtypes = AutoCast._float8_dtypes()
                if not any(
                    _supports_scale(dtype, meta, safety_margin=2.0)
                    for dtype in float8_dtypes
                ):
                    fp8_allowed = False
                    if logger:
                        logger(
                            "[OPT] FP8 optimizers disabled: data scale exceeds float8 range"
                        )
            ok, reason = is_float8_supported(dev)
            if fp8_allowed and ok:
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
            dev, use_foreach=None, use_fused=False
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
        metadata: Optional[MetaData[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> optim.Optimizer:
        params = (
            model_or_params.parameters()
            if hasattr(model_or_params, "parameters")
            else model_or_params
        )
        ref_tensor: Optional[torch.Tensor] = None
        if isinstance(model_or_params, nn.Module):
            ref_tensor = Module._module_reference_tensor(model_or_params)
        if metadata is not None:
            dev = torch.device(metadata.device)
        elif ref_tensor is not None:
            dev = ref_tensor.device
        else:
            dev = get_device()
        meta = AutoCast._ensure_metadata(dev, metadata=metadata)
        dev = torch.device(meta.device)
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
        quant_choice: Optional[str] = None
        quant_reason: Optional[str] = None
        if getattr(meta, "has_scale", False):
            if getattr(meta, "scale_is_integral", None) is False:
                if logger:
                    logger("[OPT] Low-bit optimizers disabled: data is not integral")
            else:
                max_abs = float(abs(getattr(meta, "scale_max_abs", 0.0)))
                candidates: List[Tuple[str, Callable[[Optional[torch.device]], Tuple[bool, str]]]] = []
                if max_abs <= 7.0:
                    candidates.append(("int4", is_int4_supported))
                if max_abs <= 127.0:
                    candidates.append(("int8", is_int8_supported))
                if not candidates:
                    if logger:
                        logger(
                            "[OPT] Low-bit optimizers disabled: magnitude %.3f exceeds int8 range",
                            max_abs,
                        )
                else:
                    for name, checker in candidates:
                        ok, reason = checker(dev)
                        if ok:
                            quant_choice = name
                            quant_reason = reason
                            break
                        if logger:
                            logger(
                                f"[OPT] {name.upper()} optimizers not supported ({reason}) — fallback"
                            )
        if quant_choice in {"int8", "int4"}:
            try:
                try:
                    from torchao.optim import AdamW4bit, AdamW8bit
                except ImportError:
                    from torchao.prototype.low_bit_optim import (
                        AdamW4bit,
                        AdamW8bit,
                    )
                if quant_choice == "int8":
                    opt = AdamW8bit(params, lr=lr, weight_decay=weight_decay)
                    if logger:
                        note = f" — {quant_reason}" if quant_reason else ""
                        logger(f"[OPT] TorchAO AdamW8bit{note}")
                    return opt
                opt = AdamW4bit(params, lr=lr, weight_decay=weight_decay)
                if logger:
                    note = f" — {quant_reason}" if quant_reason else ""
                    logger(f"[OPT] TorchAO AdamW4bit{note}")
                return opt
            except Exception as exc:
                if logger:
                    logger(f"[OPT] TorchAO low-bit optimizer unavailable: {exc}")
        flags: Dict[str, bool] = optimal_optimizer_params(
            dev, use_foreach=None, use_fused=False
        )
        opt = optim.AdamW(params, lr=lr, weight_decay=weight_decay, **flags)
        if logger:
            logger(f"[OPT] Using AdamW (flags={flags})")
        return opt


class Module:
    @staticmethod
    def _infer_optimal_dtype(
        device: Optional[Union[torch.device, str]] = None,
        *,
        metadata: Optional[MetaData[Any]] = None,
    ) -> torch.dtype:
        dev = torch.device(device) if device is not None else get_device()
        candidates: List[torch.dtype] = []
        if dev.type == "cuda":
            try:
                if is_cuda_bf16_supported(dev):
                    candidates.append(torch.bfloat16)
            except Exception:
                pass
            candidates.extend((torch.float16, torch.float32))
        elif dev.type == "cpu":
            if is_cpu_bf16_supported():
                candidates.append(torch.bfloat16)
            candidates.extend((torch.float32, torch.float64))
        elif dev.type == "xpu":
            candidates.extend((torch.bfloat16, torch.float32))
        elif dev.type == "mps":
            candidates.extend((torch.float16, torch.float32))
        else:
            candidates.append(torch.float32)
        for dtype in candidates:
            if _supports_scale(dtype, metadata):
                return dtype
        return (
            torch.float64
            if _supports_scale(torch.float64, metadata)
            else candidates[-1]
        )

    @staticmethod
    def _module_reference_tensor(module: nn.Module) -> Optional[torch.Tensor]:
        with contextlib.suppress(StopIteration):
            return next(module.parameters())
        with contextlib.suppress(StopIteration):
            return next(module.buffers())
        return None

    @staticmethod
    def _metadata_for(
        model: nn.Module, metadata: Optional[MetaData[Any]] = None
    ) -> MetaData[Any]:
        AutoCast.configure(model, metadata=metadata)
        meta = AutoCast._metadata
        if meta is None:
            ref = Module._module_reference_tensor(model)
            dev = ref.device if isinstance(ref, torch.Tensor) else get_device()
            meta = MetaData.for_device(dev)
            AutoCast.configure(model, metadata=meta)
        return meta

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
        *,
        metadata: Optional[MetaData[Any]] = None,
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
        params_dtype = Module._infer_optimal_dtype(dev, metadata=metadata)
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
        metadata: Optional[MetaData[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[nn.Module, bool, str]:
        meta = Module._metadata_for(model, metadata)
        device = torch.device(meta.device)
        ok, reason = is_float8_supported(device)
        if not ok:
            AutoCast.configure(model, metadata=meta)
            return (model, False, reason)
        if getattr(meta, "has_scale", False):
            float8_dtypes = AutoCast._float8_dtypes()
            if not any(
                _supports_scale(dtype, meta, safety_margin=2.0)
                for dtype in float8_dtypes
            ):
                if logger:
                    logger(
                        "[FP8] training disabled: data scale exceeds float8 range"
                    )
                AutoCast.configure(model, metadata=meta)
                return (model, False, "data scale")
        params_dtype = Module._infer_optimal_dtype(device, metadata=meta)

        for backend in ("te", "torchao"):
            if backend == "te":
                m2, ok2, why = Module._try_enable_te_training(
                    model, params_dtype, logger
                )
            else:
                m2, ok2, why = Module._try_enable_ao_training(model, logger)
            if ok2:
                if logger:
                    logger(f"[FP8] training enabled via {why} ({reason})")
                AutoCast.configure(m2, metadata=meta)
                return (m2, True, why)
            elif logger:
                logger(f"[FP8] {backend} path skipped: {why}")
        AutoCast.configure(model, metadata=meta)
        return (model, False, "No usable FP8 backend")

    @staticmethod
    def enable_float8_prediction(
        model: nn.Module,
        metadata: Optional[MetaData[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[nn.Module, bool, str]:
        meta = Module._metadata_for(model, metadata)
        device = torch.device(meta.device)
        ok, reason = is_float8_supported(device)
        if not ok:
            AutoCast.configure(model, metadata=meta)
            return (model, False, reason)
        if getattr(meta, "has_scale", False):
            float8_dtypes = AutoCast._float8_dtypes()
            if not any(
                _supports_scale(dtype, meta, safety_margin=2.0)
                for dtype in float8_dtypes
            ):
                if logger:
                    logger(
                        "[FP8] inference disabled: data scale exceeds float8 range"
                    )
                AutoCast.configure(model, metadata=meta)
                return (model, False, "data scale")
        params_dtype = Module._infer_optimal_dtype(device, metadata=meta)
        dynamic_activations = not (
            getattr(meta, "has_scale", False)
            and getattr(meta, "scale_is_integral", None) is True
        )
        order = ("te_swap", "te_present", "ao")
        for step in order:
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
                AutoCast.configure(m2, metadata=meta)
                return (m2, True, why)
            elif logger:
                logger(f"[FP8] {step} skipped: {why}")
        AutoCast.configure(model, metadata=meta)
        return (model, False, "No usable FP8 backend")

    @staticmethod
    def enable_int8_training(
        model: nn.Module,
        metadata: Optional[MetaData[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[nn.Module, bool, str]:
        if quantize_ is None:
            msg = "torchao.quantization not installed (INT8/QAT disabled)"
            if logger:
                logger(f"[INT8] {msg}")
            return (model, False, msg)
        meta = Module._metadata_for(model, metadata)
        device = torch.device(meta.device)
        with contextlib.suppress(Exception):
            model.to(device)
        dynamic_activations = not (
            getattr(meta, "has_scale", False)
            and getattr(meta, "scale_is_integral", None) is True
        )
        group_size = 128
        try:
            base_cfg = QAT.initialize(
                model,
                mode="qat-int8",
                dynamic_activations=dynamic_activations,
                group_size=group_size,
                logger=logger,
            )
            setattr(model, "__int8_training_qat__", True)
            if logger:
                logger(
                    f"[INT8][QAT] prepared with base {base_cfg.__class__.__name__}"
                )
            AutoCast.configure(model, metadata=meta)
            return (model, True, "QAT-prepare")
        except Exception as exc:
            if logger:
                logger(f"[INT8][QAT] prepare failed: {exc}")
            last_err: Exception = exc
        try:
            m2, ok, why = ptq(
                model,
                mode="int8",
                dynamic_activations=dynamic_activations,
                group_size=group_size,
                logger=logger,
            )
            if ok:
                setattr(m2, "__int8_training_ptq__", True)
                AutoCast.configure(m2, metadata=meta)
                return (m2, True, f"PTQ({why})")
            AutoCast.configure(model, metadata=meta)
            return (model, False, f"PTQ failed: {why}")
        except Exception as exc:
            AutoCast.configure(model, metadata=meta)
            return (model, False, f"INT8 training path unavailable: {exc or last_err}")

    @staticmethod
    def enable_int8_prediction(
        model: nn.Module,
        metadata: Optional[MetaData[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[nn.Module, bool, str]:
        if quantize_ is None:
            msg = "torchao.quantization not installed (INT8 disabled)"
            if logger:
                logger(f"[INT8] {msg}")
            return (model, False, msg)
        meta = Module._metadata_for(model, metadata)
        device = torch.device(meta.device)
        with contextlib.suppress(Exception):
            model.to(device)
        dynamic_activations = not (
            getattr(meta, "has_scale", False)
            and getattr(meta, "scale_is_integral", None) is True
        )
        try:
            if dynamic_activations:
                cfg = Int8DynamicActivationInt8WeightConfig()
            else:
                cfg = Int8WeightOnlyConfig()
            quantize_(model, cfg)
            setattr(model, "__int8_inference_ao__", True)
            if logger:
                logger(f"[INT8][AO] applied {cfg.__class__.__name__}")
            AutoCast.configure(model, metadata=meta)
            return (model, True, "torchao")
        except Exception as e:
            AutoCast.configure(model, metadata=meta)
            return (model, False, f"AO failed: {e}")

