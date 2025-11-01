# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import sys
import time
import warnings
from dataclasses import replace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.distributed
import torch.nn as nn
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load,
    save,
)
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import DTensor, Placement, Replicate
from torch.distributed.fsdp import MixedPrecisionPolicy
from tqdm.auto import tqdm

from ..model import Root
from ..api.config import (
    RuntimeConfig,
    coerce_model_config,
)
from ..model.functional import StandardNormalLoss, StudentsTLoss, TiledLoss
from ..data.collate import dataloader
from ..data.transforms import postprocess, preprocess
from ..data.datatype import to_torch_tensor
from ..api import is_fake_tensor, is_meta_or_fake_tensor
from ..data.stats import MetaData
from ..api.utils import Distributed, System
from ..kernels import (
    AdamW,
    AutoCast,
    inference,
    LossWeightController,
    Module,
)
from .profiler import FlopCounter
from .compat import maybe_mark_cudagraph_step_end
from .distributed import (
    broadcast_model_states,
    distributed_barrier,
    is_dist_avail_and_initialized,
    joining,
    no_synchronization,
    wrap_ddp_if_needed,
    wrap_fsdp_module,
)


try:
    from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
except ImportError:  # pragma: no cover - optional dependency

    def precompute_float8_dynamic_scale_for_fsdp(*args: Any, **kwargs: Any) -> Any:
        return None


ignored_sentences = [
    "torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to load in a single process.*",
    "torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to save in a single process.*",
    "TypedStorage is deprecated.*",
]
# --- tqdm helpers (global bar only) ---
def _infer_num_batches(loader: Any) -> int:
    """Best-effort batch count for loaders that may be stateful."""
    if loader is None:
        return 0
    try:
        n = len(loader)  # type: ignore[arg-type]
        if isinstance(n, int) and n >= 0:
            return n
    except Exception:
        pass
    if hasattr(loader, "state_dict") and hasattr(loader, "load_state_dict"):
        state = None
        with contextlib.suppress(Exception):
            state = loader.state_dict()
        if state is not None:
            count = 0
            try:
                for _ in loader:
                    count += 1
            finally:
                with contextlib.suppress(Exception):
                    loader.load_state_dict(state)
            return count
    return 0


def make_global_bar(
    *, total_epochs: int, train_loader: Any, val_loader: Any, device: torch.device
) -> Tuple[Optional[tqdm], int]:
    """학습 전체 구간(Global) 고정 total 바 생성 (실행 중 total 변경 금지)"""
    try:
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return None, 0  # rank 0만 표시
    except Exception:
        pass
    per_epoch = _infer_num_batches(train_loader) + _infer_num_batches(val_loader)
    total = int(total_epochs) * per_epoch
    if total <= 0:
        return None, per_epoch
    bar = tqdm(
        total=total,
        desc=f"Training [{device.type.upper()}]",
        unit="batch",
        dynamic_ncols=True,
        mininterval=0.3,
        miniters=1,
        leave=True,
        file=sys.stdout,
    )
    bar.set_postfix_str("0.00 MB/s, 0.00 TFLOPS", refresh=False)
    return bar, per_epoch


def tick_bar(bar: Optional[tqdm], *, mbps: Optional[float] = None, tflops: Optional[float] = None) -> None:
    """1 스텝 전진 + 간단 postfix 갱신"""
    if bar is None:
        return
    if (mbps is not None) and (tflops is not None):
        bar.set_postfix_str(f"{mbps:.2f} MB/s, {tflops:.2f} TFLOPS", refresh=False)
    bar.update(1)
ignored_pattern = "|".join((f"({sentence})" for sentence in ignored_sentences))

_DL_STATE_FILE = "dataloader.json"
_FLOAT8_LOG_MESSAGES: set[str] = set()


def dl_state_path(directory: str) -> str:
    return os.path.join(directory, _DL_STATE_FILE)


def _float8_log(msg: str, *, only_main_rank: bool = True) -> None:
    text = str(msg)
    if text in _FLOAT8_LOG_MESSAGES:
        return
    _FLOAT8_LOG_MESSAGES.add(text)
    if not only_main_rank:
        warnings.warn(text)
        return
    try:
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
    except Exception:
        pass
    warnings.warn(text)


# --- meta tensor diagnostics (no-op unless env toggles are set) ---
def _assert_no_meta_tensors(module: torch.nn.Module) -> None:
    hits: list[str] = []
    for name, param in module.named_parameters(recurse=True):
        if is_meta_or_fake_tensor(param):
            hits.append(f"param {name} shape={tuple(param.shape)}")
    for name, buffer in module.named_buffers(recurse=True):
        if is_meta_or_fake_tensor(buffer):
            hits.append(f"buffer {name} shape={tuple(buffer.shape)}")
    if hits:
        raise RuntimeError("Found meta tensors in model:\n" + "\n".join(hits))


def _maybe_install_meta_pre_hook(model: torch.nn.Module) -> None:
    hook_mode = os.environ.get("STNET_META_HOOK", "0").strip().lower()
    if hook_mode in {"0", "", "false", "off"}:
        return

    warn_only = hook_mode in {"warn", "warning"}

    def _pre(module: torch.nn.Module, inputs: Tuple[Any, ...]) -> None:
        for arg in inputs:
            if isinstance(arg, torch.Tensor) and is_meta_or_fake_tensor(arg):
                message = f"[META] {module.__class__.__name__} got meta input"
                if warn_only:
                    warnings.warn(message, stacklevel=3)
                    return
                raise RuntimeError(message)

    for submodule in model.modules():
        submodule.register_forward_pre_hook(_pre, with_kwargs=False)
def _assert_no_fake_dtensor_in_ln(
    root: nn.Module, *, allow_dtensor: bool = False
) -> None:
    try:
        from torch.distributed._tensor import DTensor as _DTensor
    except Exception:  # pragma: no cover - DTensor optional
        dtensor_types: Tuple[type, ...] = tuple()
    else:
        dtensor_types = tuple() if allow_dtensor else (_DTensor,)

    bad: list[str] = []
    for name, module in root.named_modules():
        if not isinstance(module, nn.LayerNorm):
            continue
        for attr in ("weight", "bias"):
            tensor = getattr(module, attr, None)
            if not isinstance(tensor, torch.Tensor):
                continue
            data_attr = getattr(tensor, "data", None)
            is_meta_or_fake = is_meta_or_fake_tensor(tensor)
            is_dtensor = isinstance(tensor, dtensor_types) or isinstance(
                data_attr, dtensor_types
            )
            if is_meta_or_fake or is_dtensor:
                module_name = name or module.__class__.__name__
                bad.append(f"{module_name}.{attr}{tuple(tensor.shape)}")
    if bad:
        raise RuntimeError(
            "LayerNorm params must be real/local tensors:\n  " + "\n  ".join(bad)
        )


def _materialize_all_layernorms_(model: torch.nn.Module, device: torch.device) -> None:
    """Materialize LN γ/β if meta/fake and enforce CPU float32 parameters."""

    def _reset_parameter(
        module: nn.LayerNorm,
        name: str,
        data: torch.Tensor,
        *,
        requires_grad: bool,
    ) -> None:
        setattr(module, name, nn.Parameter(data, requires_grad=requires_grad))

    for module in model.modules():
        if not isinstance(module, nn.LayerNorm):
            continue

        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        requires_grad_w = bool(getattr(weight, "requires_grad", True))
        requires_grad_b = bool(getattr(bias, "requires_grad", True))

        if device.type == "cpu":
            target_dtype = torch.float32
        else:
            target_dtype = None
            for tensor in (weight, bias):
                if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                    if not is_meta_or_fake_tensor(tensor):
                        target_dtype = tensor.dtype
                        break
            if target_dtype is None:
                target_dtype = torch.get_default_dtype()

        if module.elementwise_affine:
            if (
                not isinstance(weight, torch.Tensor)
                or is_meta_or_fake_tensor(weight)
                or isinstance(weight, DTensor)
                or isinstance(getattr(weight, "data", None), DTensor)
            ):
                data = torch.ones(
                    module.normalized_shape, device=device, dtype=target_dtype
                )
                _reset_parameter(module, "weight", data, requires_grad=requires_grad_w)
                weight = module.weight
            if (
                not isinstance(bias, torch.Tensor)
                or is_meta_or_fake_tensor(bias)
                or isinstance(bias, DTensor)
                or isinstance(getattr(bias, "data", None), DTensor)
            ):
                data = torch.zeros(
                    module.normalized_shape, device=device, dtype=target_dtype
                )
                _reset_parameter(module, "bias", data, requires_grad=requires_grad_b)
                bias = module.bias

        if device.type == "cpu":
            if isinstance(weight, torch.Tensor) and weight.dtype != torch.float32:
                data = weight.to(device=device, dtype=torch.float32)
                _reset_parameter(module, "weight", data, requires_grad=requires_grad_w)
                weight = module.weight
            if isinstance(bias, torch.Tensor) and bias.dtype != torch.float32:
                data = bias.to(device=device, dtype=torch.float32)
                _reset_parameter(module, "bias", data, requires_grad=requires_grad_b)
                bias = module.bias
        else:
            if (
                isinstance(weight, torch.Tensor)
                and isinstance(bias, torch.Tensor)
                and weight.is_floating_point()
                and bias.is_floating_point()
                and bias.dtype != weight.dtype
            ):
                data = bias.to(device=device, dtype=weight.dtype)
                _reset_parameter(module, "bias", data, requires_grad=requires_grad_b)
                bias = module.bias


def _validate_layernorm_dtypes(model: torch.nn.Module, device: torch.device) -> None:
    """Ensure LayerNorm γ/β tensors use consistent, device-appropriate dtypes."""

    mismatches: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.LayerNorm):
            continue

        tensors = [
            ("weight", getattr(module, "weight", None)),
            ("bias", getattr(module, "bias", None)),
        ]
        expected: Optional[torch.dtype]
        if device.type == "cpu":
            expected = torch.float32
        else:
            expected = None

        for label, tensor in tensors:
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                continue
            if expected is None:
                expected = tensor.dtype
            elif tensor.dtype != expected:
                module_name = name or module.__class__.__name__
                mismatches.append(
                    f"{module_name}.{label} has dtype {tensor.dtype} (expected {expected})"
                )

        if expected is not None and device.type != "cpu":
            dtypes = {
                tensor.dtype
                for _, tensor in tensors
                if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
            }
            if len(dtypes) > 1:
                module_name = name or module.__class__.__name__
                mismatches.append(
                    f"{module_name} parameters disagree on dtype: {sorted(dtypes)}"
                )

    if mismatches:
        raise RuntimeError(
            "LayerNorm parameter dtype mismatch detected:\n" + "\n".join(mismatches)
        )


def _prune_dcp_state_keys(state: Any) -> Any:
    try:
        keys = []
        for key in state.keys():
            s = str(key)
            if s.endswith("._extra_state") or s.endswith("_extra_state"):
                keys.append(key)
    except (AttributeError, TypeError):
        return state
    for key in keys:
        state.pop(key, None)
    return state


_SIZEOF = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}


def _canonical_dtype(x: torch.dtype | str) -> str:
    if isinstance(x, torch.dtype):
        s = str(x).lower()
    else:
        s = str(x).strip().lower()
    if s.startswith("torch."):
        s = s.split(".", 1)[1]
    s = s.lstrip("<>|=")
    aliases = {
        "float": "float32",
        "double": "float64",
        "half": "float16",
        "halffloat": "float16",
        "boolean": "bool",
        "bool_": "bool",
        "bf16": "bfloat16",
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "u8": "uint8",
    }
    return aliases.get(s, s)


def _size(dtype: torch.dtype | str) -> int:
    try:
        return _SIZEOF[_canonical_dtype(dtype)]
    except KeyError as exc:
        raise TypeError(f"unsupported dtype: {dtype}") from exc


def _clear_device_cache(device: Optional[torch.device] = None) -> None:
    """Release cached device memory across supported accelerator backends."""

    device_types: List[str] = []
    dev_type = getattr(device, "type", None)
    if isinstance(dev_type, str) and dev_type:
        device_types.append(dev_type)
    else:
        device_types.extend(["cuda", "xpu", "mps"])

    for kind in dict.fromkeys(device_types):  # preserve order while de-duplicating
        if kind == "cuda":
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif kind == "xpu":
            xpu = getattr(torch, "xpu", None)
            if xpu is not None:
                is_available = getattr(xpu, "is_available", None)
                empty_cache = getattr(xpu, "empty_cache", None)
                if callable(is_available) and callable(empty_cache):
                    if is_available():
                        empty_cache()
        elif kind == "mps":
            mps = getattr(torch, "mps", None)
            backends_mps = getattr(getattr(torch, "backends", None), "mps", None)
            empty_cache = getattr(mps, "empty_cache", None) if mps is not None else None
            if callable(empty_cache):
                available = None
                if backends_mps is not None:
                    available_fn = getattr(backends_mps, "is_available", None)
                    if callable(available_fn):
                        available = bool(available_fn())
                if available is None and mps is not None:
                    available_fn = getattr(mps, "is_available", None)
                    if callable(available_fn):
                        available = bool(available_fn())
                if available is None:
                    available = True
                if available:
                    empty_cache()
def _backend_type(device: torch.device) -> str:
    if device.type == "cuda":
        return "nccl"
    if device.type == "xpu":
        return "xccl"
    return "gloo"


def _set_backend(device: torch.device) -> None:
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if device.type == "cuda":
        torch.cuda.set_device(rank)
    elif device.type == "xpu":
        torch.xpu.set_device(rank)
    else:
        try:
            import netifaces

            gws = netifaces.gateways()
            iface: str | None = None
            default_gateways = gws.get("default", {}) if isinstance(gws, dict) else {}
            families = []
            with contextlib.suppress(AttributeError):
                families.append(netifaces.AF_INET6)
            families.append(netifaces.AF_INET)
            for family in families:
                info = default_gateways.get(family)
                if info and len(info) >= 2:
                    iface = info[1]
                    if iface:
                        break
            if iface:
                os.environ.setdefault("GLOO_SOCKET_IFNAME", iface)
                os.environ.setdefault("TP_SOCKET_IFNAME", iface)
        except (ImportError, KeyError, OSError):
            pass


def _meta(memmap_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(memmap_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)
def _ensure_uniform_param_dtype(
    model: Any, prefer: Optional[torch.dtype] = None
) -> Optional[torch.dtype]:
    dtypes = set((p.dtype for p in model.parameters() if p is not None))
    if len(dtypes) <= 1:
        return None
    if prefer is not None:
        tgt = prefer
    elif torch.bfloat16 in dtypes:
        tgt = torch.bfloat16
    elif torch.float16 in dtypes:
        tgt = torch.float16
    else:
        tgt = torch.float32
    for mod in model.modules():
        params = getattr(mod, "_parameters", None)
        if not params:
            continue
        for name, p in list(params.items()):
            if p is None or p.dtype == tgt:
                continue
            new_p = torch.nn.Parameter(
                p.detach().to(tgt), requires_grad=p.requires_grad
            )
            setattr(mod, name, new_p)
    return tgt





def epochs(
    *,
    model: Root,
    device: torch.device,
    ops: RuntimeConfig,
    param_dtype: torch.dtype,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    sched: torch.optim.lr_scheduler.LRScheduler,
    loss_controller: LossWeightController,
    top_loss: TiledLoss,
    bottom_loss: TiledLoss,
    grad_accum_steps: int,
    train_loader: Any,
    val_loader: Any,
    total_epochs: int,
    local_rank: int = 0,
) -> None:
    """학습 전체(여러 에폭)를 수행한다."""

    if train_loader is None:
        raise RuntimeError("epochs requires a training dataloader")

    in_dim = int(ops.in_dim)
    use_timer = getattr(device, "type", "cpu") in ("cuda", "xpu", "mps") and hasattr(
        torch, "Event"
    )

    status_bar, _ = make_global_bar(
        total_epochs=int(total_epochs),
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    prev_io_time = 0.0
    prev_comp_time = 0.0
    prev_io_bytes = 0.0
    prev_flops = 0.0

    join_context = joining(model=model, optimizer=optimizer)
    with join_context:
        for epoch_idx in range(int(total_epochs)):
            if status_bar is not None:
                status_bar.set_description(
                    f"Epoch {epoch_idx + 1}/{int(total_epochs)} [{device.type.upper()}]"
                )

            if is_dist_avail_and_initialized():
                target_module = model.module if hasattr(model, "module") else model
                broadcast_model_states(target_module)
                distributed_barrier(device)

            flop_breakdown_epoch: Dict[str, float] = {}
            io_time = torch.tensor(0.0, device=device, dtype=torch.float64)
            comp_time = torch.tensor(0.0, device=device, dtype=torch.float64)
            io_bytes = torch.tensor(0.0, device=device, dtype=torch.float64)
            flops = torch.tensor(0.0, device=device, dtype=torch.float64)

            flop_counter_train = FlopCounter(model, mode="train", device=device)
            with flop_counter_train:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                t_fetch_start = time.perf_counter_ns()

                total_batches = len(train_loader)
                for step_idx, _raw in enumerate(train_loader):
                    feat, label, *_ = preprocess(_raw)
                    X = to_torch_tensor(feat)
                    X = torch.atleast_2d(X)
                    if X.dim() != 2:
                        raise RuntimeError(
                            f"features.ndim={X.dim()} (expect 2). got shape={tuple(X.shape)}"
                        )
                    if X.shape[1] != in_dim:
                        raise RuntimeError(
                            f"feature dim mismatch: X.shape[1]={X.shape[1]} != in_dim={in_dim}"
                        )
                    Y = to_torch_tensor(label)

                    t_ready = time.perf_counter_ns()
                    if use_timer:
                        h2d_s_ev, h2d_e_ev = (
                            torch.Event(device=device, enable_timing=True),
                            torch.Event(device=device, enable_timing=True),
                        )
                        h2d_s_ev.record()
                        X = X.to(device, non_blocking=True)
                        Y = Y.to(device, non_blocking=True)
                        h2d_e_ev.record()
                        h2d_e_ev.synchronize()
                        h2d_s = float(h2d_s_ev.elapsed_time(h2d_e_ev)) / 1000.0
                    else:
                        t_h2d_s = time.perf_counter_ns()
                        X = X.to(device, non_blocking=True)
                        Y = Y.to(device, non_blocking=True)
                        t_h2d_e = time.perf_counter_ns()
                        h2d_s = (t_h2d_e - t_h2d_s) / 1_000_000_000.0

                    wait_s = (t_ready - t_fetch_start) / 1_000_000_000.0
                    io_time += torch.tensor(wait_s + h2d_s, device=device, dtype=torch.float64)
                    with contextlib.suppress(Exception):
                        io_bytes += torch.tensor(
                            X.element_size() * X.nelement()
                            + Y.element_size() * Y.nelement(),
                            device=device,
                            dtype=torch.float64,
                        )

                    should_sync = (
                        (step_idx + 1) % max(1, grad_accum_steps) == 0
                    ) or (step_idx + 1 == total_batches)

                    if use_timer:
                        ev_s, ev_e = (
                            torch.Event(device=device, enable_timing=True),
                            torch.Event(device=device, enable_timing=True),
                        )
                        ev_s.record()
                    else:
                        t_comp_s = time.perf_counter_ns()

                    with no_synchronization(
                        model, enable=(grad_accum_steps > 1 and (not should_sync))
                    ):
                        with flop_counter_train.step(display=False) as train_counter:
                            with AutoCast.float(device):
                                Y_flat = Y.reshape(Y.shape[0], -1).to(
                                    device, dtype=param_dtype
                                )
                                y_hat, loss_val = model(
                                    X,
                                    labels_flat=Y_flat,
                                    global_loss=top_loss,
                                    local_loss=bottom_loss,
                                    loss_weights=loss_controller.weights(),
                                )

                            accum_scale = max(1, grad_accum_steps)
                            loss_for_backprop = loss_val / float(accum_scale)
                            scaler.scale(loss_for_backprop).backward()
                            if should_sync:
                                scaler.unscale_(optimizer)
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad(set_to_none=True)
                                sched.step()

                            with contextlib.suppress(Exception):
                                flops += torch.tensor(
                                    max(0.0, float(train_counter.get_total_flops())),
                                    device=device,
                                    dtype=torch.float64,
                                )
                            breakdown_getter = getattr(
                                train_counter, "get_manual_breakdown", None
                            )
                            if callable(breakdown_getter):
                                for name, value in breakdown_getter().items():
                                    with contextlib.suppress(Exception):
                                        flop_breakdown_epoch[name] = flop_breakdown_epoch.get(
                                            name, 0.0
                                        ) + float(value)

                    if use_timer:
                        ev_e.record()
                        ev_e.synchronize()
                        comp_time += torch.tensor(
                            float(ev_s.elapsed_time(ev_e)) / 1000.0,
                            device=device,
                            dtype=torch.float64,
                        )
                    else:
                        comp_time += torch.tensor(
                            (time.perf_counter_ns() - t_comp_s) / 1_000_000_000.0,
                            device=device,
                            dtype=torch.float64,
                        )

                    with contextlib.suppress(Exception):
                        maybe_mark_cudagraph_step_end()

                    if status_bar is not None:
                        io_elapsed = prev_io_time + float(io_time.item())
                        io_transferred = prev_io_bytes + float(io_bytes.item())
                        comp_elapsed = prev_comp_time + float(comp_time.item())
                        flop_total = prev_flops + float(flops.item())
                        mbps_cur = io_transferred / max(io_elapsed, 1e-06) / 1_000_000.0
                        tflops_cur = (
                            flop_total
                            / max(comp_elapsed, 1e-06)
                            / 1_000_000_000_000.0
                        )
                        tick_bar(status_bar, mbps=mbps_cur, tflops=tflops_cur)

                    t_fetch_start = time.perf_counter_ns()

            if val_loader is not None:
                flop_counter_val = FlopCounter(model, mode="eval", device=device)
                with flop_counter_val:
                    model.eval()
                    with inference(model), AutoCast.float(device):
                        t_fetch_start = time.perf_counter_ns()
                        for _vstep, _raw in enumerate(val_loader):
                            feat, label, *_ = preprocess(_raw)
                            X = to_torch_tensor(feat)
                            X = torch.atleast_2d(X)
                            if X.dim() != 2:
                                raise RuntimeError(
                                    f"features.ndim={X.dim()} (expect 2). got shape={tuple(X.shape)}"
                                )
                            if X.shape[1] != in_dim:
                                raise RuntimeError(
                                    f"feature dim mismatch: X.shape[1]={X.shape[1]} != in_dim={in_dim}"
                                )
                            Y = to_torch_tensor(label)

                            t_ready = time.perf_counter_ns()
                            if use_timer:
                                h2d_s_ev, h2d_e_ev = (
                                    torch.Event(device=device, enable_timing=True),
                                    torch.Event(device=device, enable_timing=True),
                                )
                                h2d_s_ev.record()
                                X = X.to(device, non_blocking=True)
                                Y = Y.to(device, non_blocking=True)
                                h2d_e_ev.record()
                                h2d_e_ev.synchronize()
                                h2d_s = float(h2d_s_ev.elapsed_time(h2d_e_ev)) / 1000.0
                            else:
                                t_h2d_s = time.perf_counter_ns()
                                X = X.to(device, non_blocking=True)
                                Y = Y.to(device, non_blocking=True)
                                t_h2d_e = time.perf_counter_ns()
                                h2d_s = (t_h2d_e - t_h2d_s) / 1_000_000_000.0

                            wait_s = (t_ready - t_fetch_start) / 1_000_000_000.0
                            io_time += torch.tensor(
                                wait_s + h2d_s, device=device, dtype=torch.float64
                            )
                            with contextlib.suppress(Exception):
                                io_bytes += torch.tensor(
                                    X.element_size() * X.nelement()
                                    + Y.element_size() * Y.nelement(),
                                    device=device,
                                    dtype=torch.float64,
                                )

                            if use_timer:
                                ev_s, ev_e = (
                                    torch.Event(device=device, enable_timing=True),
                                    torch.Event(device=device, enable_timing=True),
                                )
                                ev_s.record()
                            else:
                                t_comp_s = time.perf_counter_ns()

                            with flop_counter_val.step(display=False) as val_counter:
                                Yv_flat = Y.reshape(Y.shape[0], -1).to(
                                    device, dtype=param_dtype
                                )
                                _y, _loss_val = model(
                                    X,
                                    labels_flat=Yv_flat,
                                    global_loss=top_loss,
                                    local_loss=bottom_loss,
                                    loss_weights=loss_controller.weights(),
                                )

                            if use_timer:
                                ev_e.record()
                                ev_e.synchronize()
                                comp_time += torch.tensor(
                                    float(ev_s.elapsed_time(ev_e)) / 1000.0,
                                    device=device,
                                    dtype=torch.float64,
                                )
                            else:
                                comp_time += torch.tensor(
                                    (time.perf_counter_ns() - t_comp_s) / 1_000_000_000.0,
                                    device=device,
                                    dtype=torch.float64,
                                )

                            with contextlib.suppress(Exception):
                                flops += torch.tensor(
                                    max(0.0, float(val_counter.get_total_flops())),
                                    device=device,
                                    dtype=torch.float64,
                                )
                            breakdown_getter = getattr(
                                val_counter, "get_manual_breakdown", None
                            )
                            if callable(breakdown_getter):
                                for name, value in breakdown_getter().items():
                                    with contextlib.suppress(Exception):
                                        flop_breakdown_epoch[name] = flop_breakdown_epoch.get(
                                            name, 0.0
                                        ) + float(value)

                            if status_bar is not None:
                                io_elapsed = prev_io_time + float(io_time.item())
                                io_transferred = prev_io_bytes + float(io_bytes.item())
                                comp_elapsed = prev_comp_time + float(comp_time.item())
                                flop_total = prev_flops + float(flops.item())
                                mbps_cur = (
                                    io_transferred / max(io_elapsed, 1e-06) / 1_000_000.0
                                )
                                tflops_cur = (
                                    flop_total
                                    / max(comp_elapsed, 1e-06)
                                    / 1_000_000_000_000.0
                                )
                                tick_bar(status_bar, mbps=mbps_cur, tflops=tflops_cur)

                            t_fetch_start = time.perf_counter_ns()

            if is_dist_avail_and_initialized():
                for t in (comp_time, io_time, flops, io_bytes):
                    torch.distributed.all_reduce(
                        t, op=torch.distributed.ReduceOp.SUM
                    )

                world = max(1, Distributed.get_world_size(device))
                comp_time /= world
                io_time /= world
                flops /= world
                io_bytes /= world

                distributed_barrier(device)

            prev_comp_time += float(comp_time.item())
            prev_io_time += float(io_time.item())
            prev_flops += float(flops.item())
            prev_io_bytes += float(io_bytes.item())

    if status_bar is not None:
        mbps = prev_io_bytes / max(prev_io_time, 1e-06) / 1_000_000.0
        tflops = prev_flops / max(prev_comp_time, 1e-06) / 1_000_000_000_000.0
        status_bar.set_postfix_str(f"{mbps:.2f} MB/s, {tflops:.2f} TFLOPS", refresh=True)
        status_bar.close()


def infer(
    model: torch.nn.Module,
    data_loader: Any,
    *,
    device: torch.device,
    ops: RuntimeConfig,
    status_bar: Optional[tqdm] = None,
    chunk_dir: Optional[str] = None,
    streaming: bool = False,
) -> Optional[Dict[Tuple, torch.Tensor]]:
    """Run inference with optional streaming output aggregation."""

    run_model = wrap_ddp_if_needed(model, device=device)
    run_model.eval()
    module_eval = run_model.module if hasattr(run_model, "module") else run_model
    broadcast_model_states(module_eval)

    chunk_idx = 0
    flop_counter = FlopCounter(run_model, mode="eval", device=device)
    use_timer = getattr(device, "type", "cpu") in ("cuda", "xpu", "mps") and hasattr(
        torch, "Event"
    )
    io_bytes: float = 0.0
    io_time: float = 0.0
    comp_time: float = 0.0
    total_flops: float = 0.0
    t_fetch_start = time.perf_counter_ns()
    preds: List[torch.Tensor] = []
    recovered_streaming = False
    is_dist = is_dist_avail_and_initialized()
    rank = torch.distributed.get_rank() if is_dist else 0

    try:
        with flop_counter, inference(run_model), AutoCast.float(device):
            for _idx, _raw in enumerate(data_loader):
                feat, _label, *_ = preprocess(_raw)
                X = to_torch_tensor(feat)
                X = torch.atleast_2d(X)
                if X.dim() != 2:
                    raise RuntimeError(
                        f"infer: feats.ndim={X.dim()} (expect 2), shape={tuple(X.shape)}"
                    )
                if X.shape[1] != int(ops.in_dim):
                    raise AssertionError(
                        "infer: feature dim mismatch — "
                        f"feats.shape[1]={X.shape[1]} != in_dim={ops.in_dim}."
                    )
                if X.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                    X = X.to(dtype=torch.float32)
                if use_timer:
                    ev_h2d_s, ev_h2d_e = (
                        torch.Event(device=device, enable_timing=True),
                        torch.Event(device=device, enable_timing=True),
                    )
                    ev_h2d_s.record()
                    X = X.to(device, non_blocking=True)
                    ev_h2d_e.record()
                    ev_h2d_e.synchronize()
                    h2d_s = float(ev_h2d_s.elapsed_time(ev_h2d_e)) / 1000.0
                else:
                    t_h2d_s = time.perf_counter_ns()
                    X = X.to(device, non_blocking=True)
                    t_h2d_e = time.perf_counter_ns()
                    h2d_s = (t_h2d_e - t_h2d_s) / 1_000_000_000.0
                wait_s = (time.perf_counter_ns() - t_fetch_start) / 1_000_000_000.0
                io_time += wait_s + h2d_s
                with contextlib.suppress(Exception):
                    io_bytes += float(X.element_size() * X.nelement())
                if use_timer:
                    ev_s, ev_e = (
                        torch.Event(device=device, enable_timing=True),
                        torch.Event(device=device, enable_timing=True),
                    )
                    ev_s.record()
                else:
                    t0 = time.perf_counter_ns()
                with no_synchronization(run_model, enable=True):
                    with flop_counter.step(display=False) as step_counter:
                        with contextlib.suppress(Exception):
                            mark_step = getattr(
                                getattr(torch, "compiler", None),
                                "cudagraph_mark_step_begin",
                                None,
                            )
                            if callable(mark_step):
                                mark_step()
                        y_hat, _ = run_model(
                            X,
                            labels_flat=None,
                            global_loss=None,
                            local_loss=None,
                            loss_weights=None,
                        )
                y_hat_cpu = y_hat.detach().cpu().contiguous()
                if streaming and rank == 0:
                    chunk_path = os.path.join(chunk_dir or "", f"chunk_{chunk_idx:06d}.pt")
                    try:
                        torch.save(y_hat_cpu, chunk_path)
                        chunk_idx += 1
                    except Exception as err:
                        streaming = False
                        with contextlib.suppress(OSError):
                            os.remove(chunk_path)
                        warnings.warn(
                            "Streaming inference disabled after failing to write "
                            "predictions to disk; falling back to in-memory aggregation."
                            f" (error: {err!r})",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        if not recovered_streaming and chunk_idx > 0:
                            for stored_idx in range(chunk_idx):
                                chunk_path = os.path.join(
                                    chunk_dir or "", f"chunk_{stored_idx:06d}.pt"
                                )
                                try:
                                    preds.append(
                                        torch.load(chunk_path, map_location="cpu")
                                    )
                                    with contextlib.suppress(OSError):
                                        os.remove(chunk_path)
                                except Exception as load_err:
                                    warnings.warn(
                                        "Failed to recover streamed predictions from "
                                        f"{chunk_path!r}: {load_err!r}",
                                        RuntimeWarning,
                                        stacklevel=2,
                                    )
                            recovered_streaming = True
                            with contextlib.suppress(OSError):
                                if chunk_dir and not os.listdir(chunk_dir):
                                    os.rmdir(chunk_dir)
                        preds.append(y_hat_cpu)
                else:
                    preds.append(y_hat_cpu)
                if use_timer:
                    ev_e.record()
                    ev_e.synchronize()
                    comp_time += float(ev_s.elapsed_time(ev_e)) / 1000.0
                else:
                    t1 = time.perf_counter_ns()
                    comp_time += (t1 - t0) / 1_000_000_000.0
                with contextlib.suppress(Exception):
                    step_flops = float(step_counter.get_total_flops())
                total_flops += max(0.0, step_flops)
                mbps = io_bytes / max(io_time, 1e-06) / 1_000_000.0
                tflops = total_flops / max(comp_time, 1e-06) / 1_000_000_000_000.0
                if status_bar is not None:
                    status_bar.set_postfix_str(
                        f"{mbps:.2f} MB/s, {tflops:.2f} TFLOPS",
                        refresh=False,
                    )
                    status_bar.update(1 if status_bar.total is not None else 0)
                t_fetch_start = time.perf_counter_ns()
    finally:
        with contextlib.suppress(Exception):
            if status_bar is not None:
                status_bar.close()

    if streaming and rank == 0:
        manifest = {
            "dir": chunk_dir,
            "num_chunks": int(chunk_idx),
            "out_shape": list(ops.out_shape),
        }
        with contextlib.suppress(Exception):
            with open(
                os.path.join(chunk_dir or "", "manifest.json"),
                "w",
                encoding="utf-8",
            ) as manifest_file:
                json.dump(manifest, manifest_file)
    elif not streaming:
        flat = torch.cat(preds, dim=0) if preds else torch.empty(0)
        pred_struct = Root.unflatten_labels(flat, ops.out_shape)
        result = postprocess(ops.keys or [], pred_struct)
    else:
        result = None

    distributed_barrier(device)

    if streaming:
        return None
    if is_dist and rank != 0:
        return None
    return result if not streaming else None

def main(*args: Any) -> Optional[Root]:
    if not args:
        raise TypeError("main requires at least a RuntimeConfig argument")

    System.initialize_python_path()
    # NOTE: avoid mutating CUDA environment variables at runtime; set them before launch.

    if os.environ.get("STNET_DISABLE_MKLDNN", "0") == "1":
        mkldnn_backend = getattr(getattr(torch, "backends", None), "mkldnn", None)
        if mkldnn_backend is not None:
            mkldnn_backend.enabled = False

    ret_sink: Optional[Dict[Any, Any]] = None
    if isinstance(args[0], RuntimeConfig):
        ops = args[0]
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if len(args) >= 2:
            ret_sink = args[1]
    elif len(args) >= 2 and isinstance(args[1], RuntimeConfig):
        local_rank = int(args[0])
        ops = args[1]
        if len(args) >= 3:
            ret_sink = args[2]
    else:
        raise TypeError(
            "main expects (RuntimeConfig,), (RuntimeConfig, ret_sink), "
            "(local_rank, RuntimeConfig), or (local_rank, RuntimeConfig, ret_sink) arguments"
        )

    if ops.mode == "train":
        with contextlib.suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank % max(1, torch.cuda.device_count()))
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.set_device(local_rank % max(1, torch.xpu.device_count()))

        device = System.get_device()
        _set_backend(device)
        backend = _backend_type(device)
        init_kwargs: Dict[str, Any] = {"backend": backend}
        torch.distributed.init_process_group(**init_kwargs)
        # 초기 1회 캐시 비우기는 선택 사항. OOM 디버깅 중일 때만 유지 권장.
        # if device.type in {"cuda", "xpu", "mps"}:
        #     _clear_device_cache(device)
        cfg = coerce_model_config(
            ops.cfg_dict if isinstance(ops.cfg_dict, dict) else ops.cfg_dict
        )
        cfg = replace(cfg, device=device)
        model = Root(ops.in_dim, ops.out_shape, config=cfg)
        if ops.init_ckpt_dir is not None and os.path.isdir(ops.init_ckpt_dir):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=ignored_pattern)
                opts_sd = StateDictOptions(full_state_dict=True, cpu_offload=False)
                m_sd = get_model_state_dict(model, options=opts_sd)
                m_sd = _prune_dcp_state_keys(m_sd)
                load(
                    state_dict={"model": m_sd},
                    storage_reader=FileSystemReader(ops.init_ckpt_dir),
                )
                set_model_state_dict(
                    model, m_sd, options=StateDictOptions(strict=False)
                )
        metadata = MetaData.for_device(device)
        meta_info = _meta(ops.memmap_dir or "")
        meta_feature_dim = int(meta_info.get("feature_dim", ops.in_dim))
        if meta_feature_dim != int(ops.in_dim):
            raise RuntimeError(
                "dataset feature_dim mismatch: "
                f"meta={meta_feature_dim}, expected in_dim={ops.in_dim}"
            )
        meta_label_shape = tuple(
            int(x) for x in meta_info.get("label_shape", list(ops.out_shape))
        )
        if tuple(meta_label_shape) != tuple(ops.out_shape):
            raise RuntimeError(
                "dataset label_shape mismatch: "
                f"meta={meta_label_shape}, expected out_shape={tuple(ops.out_shape)}"
            )
        fractions = meta_info.get("fractions", [1.0, 0.0])
        if isinstance(fractions, (list, tuple)) and len(fractions) >= 2:
            actual_val_frac = float(fractions[-1])
            if not math.isclose(
                actual_val_frac,
                float(ops.val_frac),
                rel_tol=0.001,
                abs_tol=0.001,
            ):
                warnings.warn(
                    "val_frac=%s differs from memmap metadata (%s); "
                    "using metadata value for loaders"
                    % (ops.val_frac, actual_val_frac)
                )
                ops = replace(ops, val_frac=actual_val_frac)
        model, _, _ = Module.use_te_module(model, device=device)
        AutoCast.configure(model, metadata=metadata)
        param_dtype = _ensure_uniform_param_dtype(
            model,
            prefer=(
                torch.bfloat16
                if getattr(device, "type", None) == "cuda"
                and torch.cuda.is_bf16_supported()
                else None
            ),
        )
        if param_dtype is None:
            param_dtype = torch.float32
        fp8_ok, fp8_reason = System.is_float8_supported(device)
        fp8_enabled = False
        fp8_backend: Optional[str] = None
        disable_note: Optional[str] = None
        if fp8_ok:
            model, fp8_enabled, fp8_backend = Module.enable_float8_training(
                model,
                metadata=metadata,
                logger=_float8_log,
            )
            if not fp8_enabled:
                disable_note = fp8_backend
        else:
            disable_note = fp8_reason
        if not fp8_enabled:
            AutoCast.configure(model, metadata=metadata)
            if disable_note:
                _float8_log(f"[FP8] disabled: {disable_note}")
        model.train()
        world = Distributed.get_world_size(device)
        mesh = init_device_mesh(
            "cuda" if device.type == "cuda" else device.type, (world,)
        )
        mp_policy = MixedPrecisionPolicy(
            param_dtype=None,
            reduce_dtype=torch.float64,
            output_dtype=None,
            cast_forward_inputs=False,
        )
        # NOTE: FSDP2에서는 LayerNorm 계열 파라미터도 표준 fully_shard 경로를 타게
        # 하여 랭크 간 일관성을 유지한다.
        ignored_params: List[torch.nn.Parameter] = []
        for module in model.modules():
            for name in ("alpha_t", "alpha_s", "gem_p", "cls_query", "cls"):
                if hasattr(module, name):
                    p = getattr(module, name)
                    if isinstance(p, torch.nn.Parameter):
                        ignored_params.append(p)

        class _IdentityParamSet(Sequence[torch.nn.Parameter]):
            def __init__(self, params: Sequence[torch.nn.Parameter]) -> None:
                self._params = tuple(params)
                self._ids = {id(p) for p in self._params}

            def __len__(self) -> int:
                return len(self._params)

            def __iter__(self) -> Iterator[torch.nn.Parameter]:
                return iter(self._params)

            def __getitem__(self, index: int) -> torch.nn.Parameter:
                return self._params[index]

            def __contains__(self, item: object) -> bool:
                return isinstance(item, torch.nn.Parameter) and (
                    id(item) in self._ids
                )

        ignored_param_registry = _IdentityParamSet(tuple(ignored_params))

        def _per_module_ignored_params(
            module: torch.nn.Module,
        ) -> Optional[_IdentityParamSet]:
            if len(ignored_param_registry) == 0:
                return None
            params = [
                param
                for param in module.parameters(recurse=True)
                if param in ignored_param_registry
            ]
            return _IdentityParamSet(tuple(params)) if params else None

        wrapped: set[int] = set()

        def _ensure_dtensor(
            param: torch.nn.Parameter,
            *,
            placements: Optional[Sequence[Placement]] = None,
        ) -> None:
            if not isinstance(param, torch.nn.Parameter):
                return

            current_dtensor: Optional[DTensor]
            current_dtensor = param.data if isinstance(param.data, DTensor) else None
            placements_tuple: Optional[Tuple[Placement, ...]]
            placements_tuple = tuple(placements) if placements is not None else None

            if current_dtensor is None:
                # FSDP flat parameters already carry sharding metadata which must be
                # preserved.  When we see attributes that indicate FSDP ownership we
                # leave the tensor untouched instead of forcing a replicated layout,
                # otherwise the optimizer/checkpoint state would lose the original
                # sharding information.
                if (
                    getattr(param, "_is_sharded", False)
                    or hasattr(param, "_sharding_spec")
                    or param.__class__.__name__ == "FlatParameter"
                ):
                    return

                placements_tuple = placements_tuple or (Replicate(),)
                try:
                    current_dtensor = DTensor.from_local(
                        param.data,
                        mesh,
                        placements_tuple,
                        run_check=False,
                    )
                except Exception:
                    return
                param.data = current_dtensor
            else:
                # Honor explicitly requested placements when they differ from the
                # existing DTensor layout by attempting a redistribution.  Fallback
                # to the original DTensor layout if redistribution fails so we do
                # not drop gradients entirely.
                if placements_tuple is not None and tuple(current_dtensor.placements) != placements_tuple:
                    try:
                        current_dtensor = current_dtensor.redistribute(
                            device_mesh=current_dtensor.device_mesh,
                            placements=placements_tuple,
                        )
                        param.data = current_dtensor
                    except Exception:
                        placements_tuple = tuple(current_dtensor.placements)

            if placements_tuple is None and current_dtensor is not None:
                placements_tuple = tuple(current_dtensor.placements)

            grad = param.grad
            if grad is not None and not isinstance(grad, DTensor):
                target_mesh = mesh if current_dtensor is None else current_dtensor.device_mesh
                try:
                    param.grad = DTensor.from_local(
                        grad,
                        target_mesh,
                        placements_tuple or (Replicate(),),
                        run_check=False,
                    )
                except Exception:
                    param.grad = None

        def _fsdp_wrap(
            target: Optional[torch.nn.Module],
        ) -> Optional[torch.nn.Module]:
            nonlocal model
            if target is None or id(target) in wrapped:
                return target
            wrapped.add(id(target))
            per_mod_ignored = _per_module_ignored_params(target)
            sharded = wrap_fsdp_module(
                target,
                mesh=mesh,
                mp_policy=mp_policy,
                reshard_after_forward=False,
                sync_module_states=True,
                ignored_params=per_mod_ignored or None,
            )
            if target is model:
                model = sharded
            return sharded

        def _collect_block_modules(
            root: Optional[torch.nn.Module],
        ) -> List[torch.nn.Module]:
            if root is None:
                return []
            blocks: List[torch.nn.Module] = []
            seen: set[int] = set()
            for module in root.modules():
                block_list = getattr(module, "blocks", None)
                if isinstance(block_list, torch.nn.ModuleList):
                    for block in block_list:
                        if isinstance(block, torch.nn.Module) and id(block) not in seen:
                            seen.add(id(block))
                            blocks.append(block)
            return blocks

        _m_pre = model.module if hasattr(model, "module") else model
        # LN 파라미터가 meta면, 학습 시작 전(optimizer 생성 전) 강제 실체화
        _materialize_all_layernorms_(_m_pre, device)
        _validate_layernorm_dtypes(_m_pre, device)
        _assert_no_meta_tensors(_m_pre)
        _assert_no_fake_dtensor_in_ln(_m_pre)

        try:
            for submodule in _collect_block_modules(
                getattr(model, "local_net", None)
            ) + _collect_block_modules(getattr(model, "global_net", None)):
                _fsdp_wrap(submodule)
            _fsdp_wrap(model)
        except (RuntimeError, ValueError, TypeError):
            model = wrap_fsdp_module(
                model,
                mesh=mesh,
                mp_policy=mp_policy,
                ignored_params=(
                    ignored_param_registry if len(ignored_param_registry) > 0 else None
                ),
                reshard_after_forward=False,
                sync_module_states=True,
            )

        for ignored_param in ignored_param_registry:
            _ensure_dtensor(ignored_param, placements=(Replicate(),))

        for parameter in model.parameters():
            _ensure_dtensor(parameter)

        _m_post = model.module if hasattr(model, "module") else model
        _validate_layernorm_dtypes(_m_post, device)
        _assert_no_meta_tensors(_m_post)
        _assert_no_fake_dtensor_in_ln(_m_post, allow_dtensor=True)
        _maybe_install_meta_pre_hook(_m_post)
        broadcast_model_states(_m_post)

        net_params = [p for p in model.parameters()]
        optimizer = AdamW.float(
            net_params,
            lr=ops.base_lr,
            weight_decay=ops.weight_decay,
            metadata=metadata,
            logger=None,
        )
        def _prime_adam_like_state(optim: torch.optim.Optimizer) -> None:
            """Eagerly initialize Adam/AdamW optimizer state to avoid lazy step()."""

            def _as_step_tensor(
                value: Any,
                *,
                param: torch.Tensor,
                capturable: bool,
                fused: bool,
            ) -> torch.Tensor:
                desired_device = (
                    param.device if (capturable or fused) else torch.device("cpu")
                )
                desired_dtype = (
                    param.dtype if torch.is_floating_point(param) else torch.float32
                )
                if isinstance(value, torch.Tensor):
                    step_tensor = value.detach()
                    if step_tensor.ndim != 0:
                        step_tensor = step_tensor.reshape(())
                    if step_tensor.device != desired_device:
                        step_tensor = step_tensor.to(desired_device)
                    if step_tensor.dtype != desired_dtype:
                        step_tensor = step_tensor.to(desired_dtype)
                else:
                    base = float(value) if value is not None else 0.0
                    step_tensor = torch.tensor(
                        base,
                        dtype=desired_dtype,
                        device=desired_device,
                    )
                return step_tensor

            for group in optim.param_groups:
                amsgrad = group.get("amsgrad", False)
                capturable = bool(group.get("capturable", False))
                fused = bool(group.get("fused", False))
                for param in group.get("params", []):
                    if not getattr(param, "requires_grad", False):
                        continue

                    state = optim.state.get(param)
                    state = {} if state is None else state
                    step_value = state.get("step")
                    state["step"] = _as_step_tensor(
                        step_value,
                        param=param,
                        capturable=capturable,
                        fused=fused,
                    )
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(param)
                    if "exp_avg_sq" not in state:
                        state["exp_avg_sq"] = torch.zeros_like(param)
                    if amsgrad and "max_exp_avg_sq" not in state:
                        state["max_exp_avg_sq"] = torch.zeros_like(param)
                    optim.state[param] = state

        if ops.init_ckpt_dir is not None and os.path.isdir(ops.init_ckpt_dir):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=ignored_pattern)
                _prime_adam_like_state(optimizer)
                optim_sd = get_optimizer_state_dict(model, optimizers=optimizer)
                try:
                    load(
                        state_dict={"optimizer": optim_sd},
                        storage_reader=FileSystemReader(ops.init_ckpt_dir),
                    )
                except (
                    FileNotFoundError,
                    ValueError,
                    KeyError,
                    RuntimeError,
                    CheckpointException,
                ) as exc:
                    if "optimizer" not in str(exc).lower():
                        raise
                else:
                    set_optimizer_state_dict(
                        model,
                        optimizer,
                        optim_sd,
                        options=StateDictOptions(strict=False),
                    )
                    _prime_adam_like_state(optimizer)
        _t = StudentsTLoss(
            confidence=0.99,
            metric="t_value",
            two_tailed=True,
            df=4,
            mu_mode="error",
            std_mode="pooled",
            ddof=1,
            clamp_max=8.0,
            detach_stats=True,
            dim=-1,
            reduction="none",
        )
        _z = StandardNormalLoss(
            confidence=0.99,
            metric="z_value",
            two_tailed=True,
            penalty="softplus",
            tau=1.0,
            mu_mode="error",
            std_mode="pooled",
            ddof=1,
            clamp_max=8.0,
            detach_stats=True,
            dim=-1,
            reduction="none",
        )
        top_loss = TiledLoss(
            _t,
            mask_mode=ops.loss_mask_mode,
            mask_value=ops.loss_mask_value,
            tile_dim=ops.loss_tile_dim,
            tile_size=ops.loss_tile_size,
            reduction="mean",
        )
        bottom_loss = TiledLoss(
            _z,
            mask_mode=ops.loss_mask_mode,
            mask_value=ops.loss_mask_value,
            tile_dim=ops.loss_tile_dim,
            tile_size=ops.loss_tile_size,
            reduction="mean",
        )
        loss_controller = LossWeightController()
        ckpt_state_path = dl_state_path(ops.ckpt_dir or "")
        init_state_path = (
            dl_state_path(ops.init_ckpt_dir) if ops.init_ckpt_dir else None
        )
        state_train: Dict[str, Any] = {}
        state_val: Dict[str, Any] = {}
        _dlp = (
            ckpt_state_path
            if os.path.isfile(ckpt_state_path)
            else (
                init_state_path
                if init_state_path and os.path.isfile(init_state_path)
                else None
            )
        )
        restore_dl_state = False
        if _dlp:
            with contextlib.suppress(Exception):
                _dl_json = json.load(open(_dlp, "r", encoding="utf-8"))
                if isinstance(_dl_json, dict):
                    state_train = _dl_json.get("train", {}) or {}
                    state_val = _dl_json.get("val", {}) or {}
                    restore_dl_state = bool(state_train) or bool(state_val)
        train_loader: Any = None
        val_loader: Any = None
        keep: Any = None
        status_bar: Optional[tqdm] = None
        try:
            train_loader, val_loader, keep = dataloader(
                memmap_dir=ops.memmap_dir,
                device=device,
                batch_size=int(ops.batch_size or 128),
                val_frac=float(ops.val_frac),
                prefetch_factor=ops.prefetch_factor,
                non_blocking_copy=bool(ops.overlap_h2d),
            )
            if restore_dl_state:
                with contextlib.suppress(Exception):
                    train_loader.load_state_dict(state_train)
                if val_loader is not None:
                    with contextlib.suppress(Exception):
                        val_loader.load_state_dict(state_val)
                restore_dl_state = False

            train_steps = _infer_num_batches(train_loader)
            val_steps = _infer_num_batches(val_loader)
            steps_per_epoch = max(1, train_steps + val_steps)
            total_steps = max(1, int(ops.epochs) * steps_per_epoch)
            if ops.warmup_ratio > 0.0:
                warmup_steps = max(1, int(total_steps * ops.warmup_ratio))
                main_steps = max(1, total_steps - warmup_steps)
            else:
                warmup_steps = 0
                main_steps = max(1, total_steps)
            base = float(ops.base_lr)
            emin = float(ops.eta_min)
            start_factor = 0.001

            def _scheduler(step: int) -> float:
                if warmup_steps > 0 and step < warmup_steps:
                    return start_factor + (1.0 - start_factor) * (
                        step / max(1, warmup_steps)
                    )
                t = step - warmup_steps
                frac_min = emin / base if base > 0.0 else 0.0
                return frac_min + (1.0 - frac_min) * 0.5 * (
                    1.0 + math.cos(math.pi * t / max(1, main_steps))
                )

            sched = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=_scheduler
            )
            scaler = torch.amp.GradScaler(
                enabled=(
                    device.type == "cuda" and (not torch.cuda.is_bf16_supported())
                )
            )
            epochs(
                model=model,
                device=device,
                ops=ops,
                param_dtype=param_dtype,
                optimizer=optimizer,
                scaler=scaler,
                sched=sched,
                loss_controller=loss_controller,
                top_loss=top_loss,
                bottom_loss=bottom_loss,
                grad_accum_steps=int(ops.grad_accum_steps),
                train_loader=train_loader,
                val_loader=val_loader,
                total_epochs=int(ops.epochs),
                local_rank=local_rank,
            )
            # 스텝마다 캐시/GC 호출 제거 (성능/안정성 저하 방지)
        finally:
            if keep is not None:
                keep.cleanup()
            # 종료 시 전역 캐시 비우기도 보통 불필요. OOM 디버깅 중에만 사용.
            # if getattr(device, "type", None) in {"cuda", "xpu", "mps"}:
            #     _clear_device_cache(device)
            # gc.collect()
        if local_rank == 0:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=ignored_pattern)
                opts_sd = StateDictOptions(full_state_dict=True, cpu_offload=True)
                model_sd = get_model_state_dict(model, options=opts_sd)
                optim_sd = get_optimizer_state_dict(model, optimizers=optimizer)
                writer = FileSystemWriter(
                    ops.ckpt_dir or "", sync_files=True, overwrite=True
                )
                save(
                    state_dict={"model": model_sd, "optimizer": optim_sd},
                    storage_writer=writer,
                )
            with contextlib.suppress(Exception):
                _dl = {
                    "train": (
                        train_loader.state_dict()
                        if train_loader is not None
                        else {}
                    ),
                    "val": (
                        val_loader.state_dict()
                        if val_loader is not None
                        else {}
                    ),
                }
                with open(
                    dl_state_path(ops.ckpt_dir or ""), "w", encoding="utf-8"
                ) as _f:
                    json.dump(_dl, _f)
        torch.distributed.barrier(
            device_ids=[local_rank]
            if device.type in ("cuda", "xpu")
            else None
        )
        with contextlib.suppress(Exception):
            if local_rank == 0 and status_bar is not None:
                status_bar.close()
        torch.distributed.destroy_process_group()
        return None

    if ops.mode in ("predict", "infer"):
        with contextlib.suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank % max(1, torch.cuda.device_count()))
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.set_device(local_rank % max(1, torch.xpu.device_count()))
        device = System.get_device()
        _set_backend(device)
        backend = _backend_type(device)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=backend)

        cfg = coerce_model_config(
            ops.cfg_dict if isinstance(ops.cfg_dict, dict) else ops.cfg_dict
        )
        model = Root(ops.in_dim, ops.out_shape, config=cfg)
        if ops.model_ckpt_dir is not None and os.path.isdir(ops.model_ckpt_dir):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=ignored_pattern)
                opts_sd = StateDictOptions(full_state_dict=True, cpu_offload=True)
                m_sd = get_model_state_dict(model, options=opts_sd)
                m_sd = _prune_dcp_state_keys(m_sd)
                load(
                    state_dict={"model": m_sd},
                    storage_reader=FileSystemReader(ops.model_ckpt_dir),
                )
                set_model_state_dict(
                    model, m_sd, options=StateDictOptions(strict=False)
                )
        model.to(device, non_blocking=True).eval()
        metadata = MetaData.for_device(device)
        model, _, _ = Module.use_te_module(model, device=device)
        _m_eval = model.module if hasattr(model, "module") else model
        _materialize_all_layernorms_(_m_eval, device)
        _validate_layernorm_dtypes(_m_eval, device)
        _assert_no_meta_tensors(_m_eval)
        _assert_no_fake_dtensor_in_ln(_m_eval)
        _maybe_install_meta_pre_hook(_m_eval)
        _ensure_uniform_param_dtype(
            model,
            prefer=(
                torch.bfloat16
                if (
                    getattr(device, "type", None) == "cuda"
                    and torch.cuda.is_bf16_supported()
                )
                else None
            ),
        )
        AutoCast.configure(model, metadata=metadata)
        fp8_infer_ok, fp8_infer_reason = System.is_float8_supported(device)
        if fp8_infer_ok:
            model, _, _ = Module.enable_float8_prediction(
                model,
                metadata=metadata,
                logger=_float8_log,
            )
        else:
            AutoCast.configure(model, metadata=metadata)
            _float8_log(f"[FP8] disabled: {fp8_infer_reason}")
        model.eval()

        data_loader, _, keep = dataloader(
            memmap_dir=ops.memmap_dir or "",
            device=device,
            batch_size=int(ops.batch_size or 512),
            val_frac=0.0,
            prefetch_factor=ops.prefetch_factor,
            non_blocking_copy=True,
        )
        total_batches = _infer_num_batches(data_loader)
        status_bar: Optional[tqdm]
        try:
            if torch.distributed.get_rank() != 0:
                status_bar = None
            else:
                status_bar = tqdm(
                    total=total_batches if total_batches > 0 else None,
                    desc=f"Prediction [{device.type.upper()}]",
                    unit="batch",
                    dynamic_ncols=True,
                    mininterval=0.3,
                    miniters=1,
                    leave=True,
                    file=sys.stdout,
                )
                status_bar.set_postfix_str("0.00 MB/s, 0.00 TFLOPS", refresh=False)
        except Exception:
            status_bar = None

        chunk_dir = os.environ.get("STF_PRED_CHUNK_DIR")
        if not chunk_dir and (ops.ckpt_dir or ""):
            chunk_dir = os.path.join(ops.ckpt_dir, "pred_chunks")
        streaming = False
        if chunk_dir and torch.distributed.get_rank() == 0:
            try:
                os.makedirs(chunk_dir, exist_ok=True)
                streaming = True
            except Exception:
                streaming = False
        else:
            chunk_dir = chunk_dir if streaming else None

        result = infer(
            model,
            data_loader,
            device=device,
            ops=ops,
            status_bar=status_bar,
            chunk_dir=chunk_dir,
            streaming=streaming,
        )
        if result is not None and ret_sink is not None:
            ret_sink.update(result)
        if keep is not None:
            keep.cleanup()

        distributed_barrier(device)
        torch.distributed.destroy_process_group()
        return None

    raise ValueError(f"unsupported ops mode: {ops.mode}")
