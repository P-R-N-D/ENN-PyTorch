# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
import os
import platform
import sys
import threading
import time
import warnings
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed
import torch.nn as nn
from tensordict import TensorDictBase
from torch.distributed.checkpoint import (FileSystemReader, FileSystemWriter,
                                          load, save)
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                     get_model_state_dict,
                                                     get_optimizer_state_dict,
                                                     set_model_state_dict,
                                                     set_optimizer_state_dict)
from torch.distributed.fsdp import MixedPrecisionPolicy
from tqdm.auto import tqdm

_nvml = None
_NVML_READY = False
_NVML_TRIED = False
_NVML_LOCK = threading.Lock()

_NVML_QUERY_LOCK = threading.Lock()
_NVML_HANDLE_CACHE: dict[int, Any] = {}
_NVML_UTIL_CACHE: dict[int, tuple[float, Optional[float], Optional[float]]] = {}
_NVML_MIN_INTERVAL_S: Optional[float] = None

# NVML failure backoff:
# - When NVML is flaky (driver reset, container permissions, MIG changes, etc.),
#   repeated exceptions inside tight training loops become expensive, especially
#   on no-GIL builds where the loop runs much faster.
# - We keep NVML optional and best-effort: on repeated failures, temporarily
#   disable NVML queries and fall back to torch.cuda.mem_get_info().
_NVML_FAIL_COUNT: int = 0
_NVML_BACKOFF_UNTIL: float = 0.0
_NVML_BACKOFF_S: Optional[float] = None
_NVML_FAIL_MAX: Optional[int] = None


def _nvml_disabled() -> bool:
    """Return True if NVML telemetry is explicitly disabled via env vars."""

    def _parse_bool(v: object) -> Optional[bool]:
        if v is None:
            return None
        s = str(v).strip().lower()
        if not s:
            return None
        if s in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
            return True
        if s in {"0", "false", "no", "n", "off", "disable", "disabled"}:
            return False
        return None

    # STNET_NVML_DISABLE=1 disables NVML unconditionally.
    v = _parse_bool(os.environ.get("STNET_NVML_DISABLE"))
    if v is True:
        return True

    # STNET_NVML=0 also disables (mirrors other STNET toggles).
    v = _parse_bool(os.environ.get("STNET_NVML"))
    if v is False:
        return True
    return False


def _nvml_fail_max() -> int:
    """Number of consecutive NVML query failures before entering backoff."""

    global _NVML_FAIL_MAX
    if _NVML_FAIL_MAX is not None:
        return int(_NVML_FAIL_MAX)

    v = os.environ.get("STNET_NVML_FAIL_MAX")
    if v is None:
        v = os.environ.get("STNET_NVML_FAILURES")
    if v is not None and str(v).strip():
        try:
            _NVML_FAIL_MAX = max(1, int(v))
            return int(_NVML_FAIL_MAX)
        except Exception:
            pass
    _NVML_FAIL_MAX = 3
    return int(_NVML_FAIL_MAX)


def _nvml_backoff_s() -> float:
    """Backoff duration (seconds) once NVML is considered unhealthy.

    Override with:
      - STNET_NVML_BACKOFF_S / STNET_NVML_BACKOFF
      - Set to 0 to disable backoff (not recommended).
    """

    global _NVML_BACKOFF_S
    if _NVML_BACKOFF_S is not None:
        return float(_NVML_BACKOFF_S)

    for key in ("STNET_NVML_BACKOFF_S", "STNET_NVML_BACKOFF"):
        v = os.environ.get(key)
        if v is not None and str(v).strip():
            try:
                _NVML_BACKOFF_S = max(0.0, float(v))
                return float(_NVML_BACKOFF_S)
            except Exception:
                pass

    # Default: slightly longer backoff when no-GIL optimizations are enabled,
    # because telemetry is polled much more frequently.
    try:
        from .system import Thread

        _NVML_BACKOFF_S = 30.0 if bool(Thread.nogil_optimizations_enabled()) else 10.0
    except Exception:
        _NVML_BACKOFF_S = 10.0
    return float(_NVML_BACKOFF_S)


def _nvml_in_backoff(now: Optional[float] = None) -> bool:
    """True if NVML queries are currently in backoff."""

    if now is None:
        now = float(time.perf_counter())
    with _NVML_LOCK:
        until = float(_NVML_BACKOFF_UNTIL or 0.0)
    return bool(until > 0.0 and float(now) < until)


def _nvml_min_interval_s() -> float:
    """Minimum seconds between NVML queries per device.

    NVML calls are relatively expensive. On no-GIL builds (or when STNET_NOGIL_OPT is enabled),
    the input pipeline can run at much higher throughput, and querying NVML every step can
    become noticeable overhead.

    - Default: 0.0s (no throttling) on regular builds.
    - Default: 0.10s on no-GIL optimized runs.
    - Override with STNET_NVML_MIN_INTERVAL_S / STNET_NVML_MIN_INTERVAL (float seconds).
    """

    global _NVML_MIN_INTERVAL_S
    if _NVML_MIN_INTERVAL_S is not None:
        return float(_NVML_MIN_INTERVAL_S)

    # Env override (seconds)
    for key in ("STNET_NVML_MIN_INTERVAL_S", "STNET_NVML_MIN_INTERVAL"):
        v = os.environ.get(key)
        if v is not None and str(v).strip():
            try:
                _NVML_MIN_INTERVAL_S = max(0.0, float(v))
                return float(_NVML_MIN_INTERVAL_S)
            except Exception:
                pass

    # Default: throttle only when no-GIL optimizations are enabled.
    try:
        from .system import Thread

        _NVML_MIN_INTERVAL_S = 0.10 if bool(Thread.nogil_optimizations_enabled()) else 0.0
    except Exception:
        _NVML_MIN_INTERVAL_S = 0.0
    return float(_NVML_MIN_INTERVAL_S)


def _ensure_nvml() -> bool:
    """Best-effort NVML init.

    - Keeps NVML optional (no hard dependency).
    - Avoids NVML initialization at import time.
    - Thread-safe (important on free-threaded/no-GIL builds).
    """

    global _nvml, _NVML_READY, _NVML_TRIED
    if _nvml_in_backoff():
        return False
    if _nvml_disabled():
        _NVML_TRIED = True
        _NVML_READY = False
        _nvml = None
        return False
    if _NVML_READY:
        return True
    if _NVML_TRIED:
        return False
    with _NVML_LOCK:
        if _nvml_in_backoff():
            return False
        if _NVML_READY:
            return True
        if _NVML_TRIED:
            return False
        _NVML_TRIED = True
        try:
            with warnings.catch_warnings():
                # The deprecated `pynvml` PyPI package can emit FutureWarnings.
                # The recommended distribution is `nvidia-ml-py`, which still
                # provides the `pynvml` import path.
                warnings.simplefilter("ignore", category=FutureWarning)
                import pynvml as _pynvml

            _nvml = _pynvml
            _nvml.nvmlInit()
            _NVML_READY = True
        except Exception:
            _nvml = None
            _NVML_READY = False
    return bool(_NVML_READY)
try:
    import psutil as _psutil
except Exception:
    _psutil = None

from ..api.config import RuntimeConfig, coerce_model_config
from ..data.collections import Cache, Pool
from ..data.datatype import to_tensordict, to_torch_tensor
from ..data.pipeline import Dataset
# NOTE: Sampler scale is per-session/per-loader now; avoid global Sampler scaling here.
from ..model import fused
from ..model.fused import Autocast, Gradient
from ..functional.losses import (CRPSLoss, DataFidelityLoss,
                                 LinearCombinationLoss, LossWeightController,
                                 StandardNormalLoss, StudentsTLoss, TiledLoss)
from ..functional.optimizers import (SWALR, AdamW, StochasticWeightAverage,
                                     stochastic_weight_average)
from ..model.nn import History, Root, resize_scaler_buffer
from .compat import (cudagraph_step_end, is_meta_or_fake_tensor,
                     torch_compile_safe, torch_no_compile,
                     torch_safe_distributed)
from .distributed import (distributed_barrier, distributed_sync,
                          get_world_size, is_distributed, joining, no_sync,
                          to_ddp, to_fsdp)
from ..functional.profiler import FlopCounter
from .system import (Memory, Thread, empty_device_cache, get_device, get_tlb,
                     initialize_python_path, new_dir, posix_time,
                     set_float32_precision)

if TYPE_CHECKING:
    import numpy as _np
    from numpy.typing import NDArray as _NDArray

    Float64Array = _NDArray[_np.float64]
else:
    Float64Array = Any

_LOGGER = logging.getLogger(__name__)

torch_safe_distributed()

_SAMPLER_SCALE_LOG_LOCK = threading.Lock()
_SAMPLER_SCALE_LOG_LAST_S: Dict[Tuple[int, str], float] = {}

_OOM_RETRY_LOCK = threading.Lock()
_OOM_RETRY_COUNT: Dict[Tuple[int, str, int], int] = {}


def _rt_env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    if not s:
        return bool(default)
    if s in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
        return True
    if s in {"0", "false", "no", "n", "off", "disable", "disabled"}:
        return False
    return bool(default)


def _rt_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


def _rt_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _oom_retry_inc(loader: Any, phase: str, step: int) -> int:
    key = (int(id(loader)), str(phase), int(step))
    with _OOM_RETRY_LOCK:
        cur = int(_OOM_RETRY_COUNT.get(key, 0)) + 1
        _OOM_RETRY_COUNT[key] = int(cur)
        return int(cur)


def _oom_retry_clear(loader: Any, phase: str, step: int) -> None:
    key = (int(id(loader)), str(phase), int(step))
    with _OOM_RETRY_LOCK:
        _OOM_RETRY_COUNT.pop(key, None)


def _oom_max_retries(phase: str) -> int:
    # Defaults chosen to avoid infinite retry storms.
    # Override with:
    #   - STNET_OOM_MAX_RETRIES_TRAIN
    #   - STNET_OOM_MAX_RETRIES_VAL
    #   - STNET_OOM_MAX_RETRIES_PER_BATCH
    phase = str(phase).strip().lower()
    if phase == "train":
        v = _rt_env_int("STNET_OOM_MAX_RETRIES_TRAIN", _rt_env_int("STNET_OOM_MAX_RETRIES_PER_BATCH", 4))
    elif phase in {"val", "valid", "validation"}:
        v = _rt_env_int("STNET_OOM_MAX_RETRIES_VAL", _rt_env_int("STNET_OOM_MAX_RETRIES_PER_BATCH", 2))
    else:
        v = _rt_env_int("STNET_OOM_MAX_RETRIES_PER_BATCH", 3)
    return max(0, int(v))


def _oom_skip_enabled(phase: str) -> bool:
    # Default: skip the batch after retry budget is exhausted (prevents OOM storms).
    # Override:
    #   - STNET_OOM_SKIP_BATCH=0 to raise instead
    #   - STNET_OOM_SKIP_TRAIN / STNET_OOM_SKIP_VAL for per-phase control
    phase = str(phase).strip().lower()
    if phase == "train":
        return _rt_env_flag("STNET_OOM_SKIP_TRAIN", _rt_env_flag("STNET_OOM_SKIP_BATCH", True))
    if phase in {"val", "valid", "validation"}:
        return _rt_env_flag("STNET_OOM_SKIP_VAL", _rt_env_flag("STNET_OOM_SKIP_BATCH", True))
    return _rt_env_flag("STNET_OOM_SKIP_BATCH", True)


def _oom_scale_down_factor(attempt: int) -> float:
    # More aggressive with repeated OOMs.
    # 1st: 0.8, 2nd: 0.7, 3rd: 0.6, 4th+: 0.5
    seq = (0.8, 0.7, 0.6, 0.5)
    i = max(1, int(attempt)) - 1
    if i < 0:
        i = 0
    if i >= len(seq):
        i = len(seq) - 1
    return float(seq[i])


def _log_sampler_scale_rate_limited(
    *,
    logger: logging.Logger,
    scale_ctl: Any,
    tag: str,
    msg: str,
    level: str = "info",
    min_interval_s: Optional[float] = None,
) -> None:
    # Env default:
    #   STNET_SAMPLER_SCALE_LOG_MIN_INTERVAL_S=5
    if min_interval_s is None:
        min_interval_s = _rt_env_float("STNET_SAMPLER_SCALE_LOG_MIN_INTERVAL_S", 5.0)
    try:
        min_interval_s = float(min_interval_s)
    except Exception:
        min_interval_s = 5.0
    if min_interval_s < 0:
        min_interval_s = 0.0

    key = (int(id(scale_ctl)), str(tag))
    now = time.monotonic()
    with _SAMPLER_SCALE_LOG_LOCK:
        last = float(_SAMPLER_SCALE_LOG_LAST_S.get(key, 0.0))
        if min_interval_s and (now - last) < float(min_interval_s):
            return
        _SAMPLER_SCALE_LOG_LAST_S[key] = float(now)

    try:
        if str(level).lower() == "debug":
            logger.debug(msg)
        else:
            logger.info(msg)
    except Exception:
        pass


MB_DIV = 1024.0 * 1024.0

_device_mem_get_info = Memory.device_mem_get_info


try:
    from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
except ImportError:

    def precompute_float8_dynamic_scale_for_fsdp(*args: Any, **kwargs: Any) -> Any:
        return None


_DL_STATE_FILE = "dataloader.json"


def _num_batches(loader: Any) -> int:
    if loader is None:
        return 0
    try:
        n = len(loader)
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


def _float8_log(
    msg: str, *args: Any, only_main_rank: bool = True, **kwargs: Any
) -> None:
    try:
        if only_main_rank and torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return
    except Exception:
        pass
    _LOGGER.info(msg, *args)


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


def _meta_monitor_pre_hook(
    module: torch.nn.Module, inputs: Tuple[Any, ...], warn_only: bool
) -> None:
    for arg in inputs:
        if isinstance(arg, torch.Tensor) and is_meta_or_fake_tensor(arg):
            message = f"[META] {module.__class__.__name__} got meta input"
            if warn_only:
                warnings.warn(message, stacklevel=3)
                return
            raise RuntimeError(message)


def _enable_meta_monitor(model: torch.nn.Module) -> None:
    hook_mode = (
        os.environ.get("STNET_META_MONITOR")
        or os.environ.get("STNET_META_HOOK")
        or "off"
    ).strip().lower()
    if hook_mode in {"0", "", "false", "off"}:
        return
    warn_only = hook_mode in {"warn", "warning"}
    for submodule in model.modules():
        submodule.register_forward_pre_hook(
            partial(_meta_monitor_pre_hook, warn_only=warn_only), with_kwargs=False
        )


def _assert_no_fake_dtensor(
    root: nn.Module, *args: Any, allow_dtensor: bool = False, **kwargs: Any
) -> None:
    bad: list[str] = []
    for name, module in root.named_modules():
        if not isinstance(module, nn.LayerNorm):
            continue
        for attr in ("weight", "bias"):
            tensor = getattr(module, attr, None)
            if tensor is None:
                continue
            is_meta_or_fake = is_meta_or_fake_tensor(tensor)
            if is_meta_or_fake:
                module_name = name or module.__class__.__name__
                bad.append(f"{module_name}.{attr}{tuple(tensor.shape)}")
    if bad:
        raise RuntimeError(
            "LayerNorm parameters must be materialized as a real Tensor: "
            + ", ".join(bad)
        )


def _reset_layernorm_parameter(
    module: nn.LayerNorm,
    name: str,
    data: torch.Tensor,
    *args: Any,
    requires_grad: bool,
) -> None:
    setattr(module, name, nn.Parameter(data, requires_grad=requires_grad))

def _cast_model_fp_dtype(model: Any, dtype: torch.dtype) -> None:
    """Cast *floating* parameters/buffers to dtype, skipping precision-exempt modules.

    NOTE: We intentionally avoid nn.Module.to(dtype=...) here, because it would also
    cast bookkeeping buffers (History, Scaler, etc.) that must remain float64/int64.
    """
    if not isinstance(dtype, torch.dtype):
        return
    try:
        if not torch.is_floating_point(torch.empty((), dtype=dtype)):
            return
    except Exception:
        return

    def _is_exempt(mod: Any) -> bool:
        return bool(getattr(mod, "__stnet_precision_exempt__", False))

    with torch.no_grad():
        for mod in getattr(model, "modules", lambda: [])():
            if _is_exempt(mod):
                continue

            params = getattr(mod, "_parameters", None)
            if params:
                for name, p in list(params.items()):
                    if p is None or not isinstance(p, torch.Tensor):
                        continue
                    if (not p.is_floating_point()) or p.dtype == dtype:
                        continue
                    params[name] = torch.nn.Parameter(
                        p.detach().to(dtype), requires_grad=bool(getattr(p, "requires_grad", True))
                    )
            bufs = getattr(mod, "_buffers", None)
            if bufs:
                for name, b in list(bufs.items()):
                    if b is None or not isinstance(b, torch.Tensor):
                        continue
                    if (not b.is_floating_point()) or b.dtype == dtype:
                        continue
                    bufs[name] = b.detach().to(dtype)


def _cpu_layernorm_param_dtype(device: torch.device) -> torch.dtype:
    try:
        meta = Autocast.coerce_metadata(device)
        cands = tuple(getattr(meta, "float_dtypes", ())) if meta is not None else ()
        if not cands:
            cands = (torch.float32,)
        chosen = Autocast.negotiate(
            tuple(cands),
            fallback=torch.float64,
            context="cpu.layernorm",
            device=device,
            meta=meta,
        )
        return torch.float64 if chosen == torch.float64 else torch.float32
    except Exception:
        return torch.float32


def _preload_layers(model: torch.nn.Module, device: torch.device) -> None:
    for module in model.modules():
        if not isinstance(module, nn.LayerNorm):
            continue
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        requires_grad_w = bool(getattr(weight, "requires_grad", True))
        requires_grad_b = bool(getattr(bias, "requires_grad", True))
        if device.type == "cpu":
            target_dtype = _cpu_layernorm_param_dtype(device)
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
            ):
                data = torch.ones(
                    module.normalized_shape, device=device, dtype=target_dtype
                )
                _reset_layernorm_parameter(
                    module, "weight", data, requires_grad=requires_grad_w
                )
                weight = module.weight
            if (
                not isinstance(bias, torch.Tensor)
                or is_meta_or_fake_tensor(bias)
            ):
                data = torch.zeros(
                    module.normalized_shape, device=device, dtype=target_dtype
                )
                _reset_layernorm_parameter(
                    module, "bias", data, requires_grad=requires_grad_b
                )
                bias = module.bias
        if device.type == "cpu":
            if isinstance(weight, torch.Tensor) and weight.dtype != target_dtype:
                data = weight.to(device=device, dtype=target_dtype)
                _reset_layernorm_parameter(
                    module, "weight", data, requires_grad=requires_grad_w
                )
                weight = module.weight
            if isinstance(bias, torch.Tensor) and bias.dtype != target_dtype:
                data = bias.to(device=device, dtype=target_dtype)
                _reset_layernorm_parameter(
                    module, "bias", data, requires_grad=requires_grad_b
                )
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
                _reset_layernorm_parameter(
                    module, "bias", data, requires_grad=requires_grad_b
                )
                bias = module.bias


def _assert_unified_layer_dtype(model: torch.nn.Module, device: torch.device) -> None:
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
            expected = _cpu_layernorm_param_dtype(device)
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


def _trim_dcp_keys(state: Any) -> Any:
    if isinstance(state, dict):
        keys = []
        for key, value in list(state.items()):
            key_str = str(key)
            if (
                key_str.endswith("._extra_state")
                or key_str.endswith("_extra_state")
                or key_str.endswith("output_baked_flag")
            ):
                keys.append(key)
                continue
            state[key] = _trim_dcp_keys(value)
        for key in keys:
            state.pop(key, None)
    return state


def _backend_type(device: torch.device) -> str:
    if device.type == "cuda":
        return "nccl"
    if device.type == "xpu":
        return "xccl"
    return "gloo"


def _set_backend(device: torch.device) -> None:
    with contextlib.suppress(Exception):
        if device.type == "cuda" and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if device.type == "cuda":
        torch.cuda.set_device(rank)
    elif device.type == "xpu":
        torch.xpu.set_device(rank)
    else:
        # CPU / Gloo: pick a sensible NIC without extra dependencies.
        # Some clusters require GLOO_SOCKET_IFNAME/TP_SOCKET_IFNAME to be set.
        iface: str | None = None

        # 0) Respect explicit user configuration.
        gloo_if = os.environ.get("GLOO_SOCKET_IFNAME")
        tp_if = os.environ.get("TP_SOCKET_IFNAME")
        if gloo_if or tp_if:
            # If the user set only one of the variables, mirror it to the other
            # so both subsystems stay consistent.
            if gloo_if and not tp_if:
                os.environ.setdefault("TP_SOCKET_IFNAME", str(gloo_if))
            elif tp_if and not gloo_if:
                os.environ.setdefault("GLOO_SOCKET_IFNAME", str(tp_if))
            return

        # 1) Linux: default route interface from /proc/net/route.
        try:
            with open("/proc/net/route", "r", encoding="utf-8") as f:
                for line in f.readlines()[1:]:
                    fields = line.strip().split()
                    # Destination == 00000000 means default route.
                    if len(fields) >= 2 and fields[1] == "00000000":
                        iface = fields[0]
                        if iface:
                            break
        except Exception:
            iface = None

        # 2) Cross-platform fallback: infer egress IPv4 and match psutil.net_if_addrs().
        if iface is None and _psutil is not None:
            try:
                import socket

                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    # No packets are sent for UDP connect; this only queries the route.
                    s.connect(("8.8.8.8", 80))
                    ip = s.getsockname()[0]
                finally:
                    s.close()

                if ip:
                    for name, addrs in _psutil.net_if_addrs().items():
                        for a in addrs:
                            if getattr(a, "family", None) == socket.AF_INET and getattr(
                                a, "address", None
                            ) == ip:
                                iface = str(name)
                                break
                        if iface:
                            break
            except Exception:
                iface = None

        if iface:
            os.environ.setdefault("GLOO_SOCKET_IFNAME", iface)
            os.environ.setdefault("TP_SOCKET_IFNAME", iface)


def _iter_source_paths(obj: Any):
    """Yield memmap directory paths from a (possibly nested) ops.sources object."""
    if obj is None:
        return
    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, dict):
        # common form: {"kind": "memmap", "path": "..."}
        if obj.get("kind") == "memmap" and isinstance(obj.get("path"), str):
            yield obj["path"]
            return
        # otherwise: recurse values
        for v in obj.values():
            yield from _iter_source_paths(v)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_source_paths(v)
        return


def _merge_meta_dicts(metas: list[dict]) -> dict:
    """Merge multiple meta.json dicts with conservative scale negotiation.

    - feature_dim / label_shape must match.
    - scale_max_abs is merged by max.
    - scale_min_value is merged by min (ignoring None).
    - scale_max_value is merged by max (ignoring None).
    - scale_min_positive is merged by min (ignoring None).
    - has_scale / has_nonfinite are merged by OR.
    - is_negotiable is merged by AND (if present).
    - underflow_action is merged by strictest: forbid > warn > allow.
    """
    if not metas:
        return {}
    base = dict(metas[0])

    def _strictest_underflow(a: str | None, b: str | None) -> str | None:
        order = {"allow": 0, "warn": 1, "forbid": 2}
        if a is None:
            return b
        if b is None:
            return a
        return a if order.get(a, 1) >= order.get(b, 1) else b

    # sanity dims
    feature_dim = base.get("feature_dim")
    label_shape = base.get("label_shape")

    has_scale = bool(base.get("has_scale", False)) or base.get("scale_max_abs") is not None or base.get("scale_min_value") is not None or base.get("scale_max_value") is not None or base.get("scale_min_positive") is not None
    has_nonfinite = bool(base.get("has_nonfinite", False))
    max_abs = base.get("scale_max_abs")
    min_val = base.get("scale_min_value")
    max_val = base.get("scale_max_value")
    min_pos = base.get("scale_min_positive")
    is_integral = base.get("scale_is_integral")
    is_negotiable = base.get("is_negotiable")
    underflow_action = base.get("underflow_action")

    for m in metas[1:]:
        if feature_dim is not None and m.get("feature_dim") is not None and int(m.get("feature_dim")) != int(feature_dim):
            raise ValueError(f"feature_dim mismatch across sources: {feature_dim} vs {m.get('feature_dim')}")
        if label_shape is not None and m.get("label_shape") is not None and tuple(m.get("label_shape")) != tuple(label_shape):
            raise ValueError(f"label_shape mismatch across sources: {label_shape} vs {m.get('label_shape')}")

        has_scale = has_scale or bool(m.get("has_scale", False)) or m.get("scale_max_abs") is not None or m.get("scale_min_value") is not None or m.get("scale_max_value") is not None or m.get("scale_min_positive") is not None
        has_nonfinite = has_nonfinite or bool(m.get("has_nonfinite", False))

        a = m.get("scale_max_abs")
        if a is not None:
            max_abs = a if max_abs is None else max(float(max_abs), float(a))

        mn = m.get("scale_min_value")
        if mn is not None:
            try:
                min_val = mn if min_val is None else (mn if mn <= min_val else min_val)
            except Exception:
                min_val = mn if min_val is None else min(float(min_val), float(mn))

        mx = m.get("scale_max_value")
        if mx is not None:
            try:
                max_val = mx if max_val is None else (mx if mx >= max_val else max_val)
            except Exception:
                max_val = mx if max_val is None else max(float(max_val), float(mx))

        p = m.get("scale_min_positive")
        if p is not None:
            min_pos = p if min_pos is None else min(float(min_pos), float(p))

        i = m.get("scale_is_integral")
        if i is not None:
            is_integral = bool(i) if is_integral is None else bool(is_integral) and bool(i)

        n = m.get("is_negotiable")
        if n is not None:
            is_negotiable = bool(n) if is_negotiable is None else bool(is_negotiable) and bool(n)

        underflow_action = _strictest_underflow(underflow_action, m.get("underflow_action"))

    base["has_scale"] = has_scale
    base["has_nonfinite"] = has_nonfinite
    base["scale_max_abs"] = max_abs
    base["scale_min_value"] = min_val
    base["scale_max_value"] = max_val
    base["scale_min_positive"] = min_pos
    base["scale_is_integral"] = is_integral
    base["is_negotiable"] = is_negotiable
    base["underflow_action"] = underflow_action
    return base


def _from_meta(memmap_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(memmap_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _unify_param_dtype(
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


def _first_source_path(obj: Any) -> str:
    if isinstance(obj, dict):
        if "path" in obj and "kind" in obj:
            return os.fspath(obj["path"])
        if obj:
            first = next(iter(obj.values()))
            return _first_source_path(first)
    if isinstance(obj, (list, tuple)) and obj:
        return _first_source_path(obj[0])
    raise RuntimeError("sources is empty or invalid")


def _merge_meta_infos(sources: Any) -> Dict[str, Any]:
    metas: list[dict] = []
    for path in _iter_source_paths(sources):
        try:
            metas.append(_from_meta(path))
        except Exception:
            continue
    if not metas:
        return {}
    return _merge_meta_dicts(metas)


def _expand(sources: Any) -> Any:
    def _expand_from_root(spec: Any) -> Tuple[Any, bool]:
        if not isinstance(spec, dict) or "path" not in spec or "kind" not in spec:
            return spec, False
        root = os.fspath(spec.get("path") or "")
        mn_path = os.path.join(root, "multinode.json")
        if not os.path.isfile(mn_path):
            return spec, False
        with open(mn_path, "r", encoding="utf-8") as _f:
            _spec = json.load(_f)
        if isinstance(_spec, dict):
            resolved = {
                str(k): {"kind": "memmap", "path": os.path.join(root, str(v))}
                for k, v in _spec.items()
            }
            return resolved, True
        if isinstance(_spec, list):
            resolved = [
                {"kind": "memmap", "path": os.path.join(root, str(v))} for v in _spec
            ]
            return resolved, True
        return spec, False

    expanded, ok = _expand_from_root(sources)
    if ok:
        return expanded
    if isinstance(sources, (list, tuple)) and len(sources) == 1:
        expanded, ok = _expand_from_root(sources[0])
        if ok:
            return expanded
    return sources


def _calibrate_per_sample_mem(
    model: Root,
    device: torch.device,
    ops: RuntimeConfig,
    dataset: Optional[Dataset] = None,
    max_probe_batch: int = 32,
    with_backward: bool = False,
    global_loss: Optional[nn.Module] = None,
    local_loss: Optional[nn.Module] = None,
    loss_weights: Optional[Any] = None,
) -> None:
    from ..data.nodes import Sampler

    try:
        in_dim = int(getattr(ops, "in_dim", 0) or 0)
    except Exception:
        in_dim = 0
    try:
        out_shape = tuple(getattr(ops, "out_shape", []) or [])
        out_dim = 1
        for d in out_shape:
            out_dim *= int(d)
    except Exception:
        out_dim = 1
    elem_size = torch.empty((), dtype=torch.float64).element_size()
    floor_bytes = int((in_dim + out_dim) * elem_size * 10240) if (in_dim + out_dim) > 0 else 0

    dev_type = getattr(device, "type", "")
    if dev_type not in {"cuda", "xpu", "mps"}:
        return

    try:
        memmap_root = _first_source_path(ops.sources)
        ds = Sampler(
            memmap_root,
            split="train",
            val_frac=float(getattr(ops, "val_frac", 0.0) or 0.0),
        )
        
    except Exception:
        return

    try:
        N = int(len(ds))
    except Exception:
        N = 0
    if N <= 0:
        return

    B0 = max(1, min(int(max_probe_batch), N))

    def _to_device(obj: Any, dev: torch.device) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.to(device=dev, non_blocking=True)
        from tensordict import TensorDictBase
        from collections.abc import Mapping
        if isinstance(obj, TensorDictBase):
            return obj.to(device=dev)
        if isinstance(obj, Mapping):
            return {k: _to_device(v, dev) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            seq = [_to_device(v, dev) for v in obj]
            return type(obj)(seq)
        return obj

    try:
        base_alloc: Optional[int] = None
        peak_api: Optional[Callable[[torch.device], int]] = None

        accel = getattr(torch, "accelerator", None)
        if accel is not None and hasattr(accel, "is_available") and accel.is_available():
            mem_mod = getattr(accel, "memory", None)
            with contextlib.suppress(Exception):
                if mem_mod is not None:
                    alloc_fn = getattr(mem_mod, "allocated", None)
                    reset_fn = getattr(mem_mod, "reset_peak_memory_stats", None)
                    peak_fn = getattr(mem_mod, "max_memory_allocated", None)
                    if callable(alloc_fn) and callable(peak_fn):
                        base_alloc = int(alloc_fn(device))
                        if callable(reset_fn):
                            reset_fn(device)
                        peak_api = lambda d: int(peak_fn(d))

        if base_alloc is None:
            if dev_type == "cuda" and torch.cuda.is_available():
                with contextlib.suppress(Exception):
                    base_alloc = int(torch.cuda.memory_allocated(device))
                    torch.cuda.reset_peak_memory_stats(device)
                    peak_api = lambda d: int(torch.cuda.max_memory_allocated(d))
            elif dev_type == "xpu" and hasattr(torch, "xpu"):
                with contextlib.suppress(Exception):
                    alloc = getattr(torch.xpu, "memory_allocated", None)
                    reset = getattr(torch.xpu, "reset_peak_memory_stats", None)
                    peak = getattr(torch.xpu, "max_memory_allocated", None)
                    if callable(alloc) and callable(peak):
                        base_alloc = int(alloc(device))
                        if callable(reset):
                            reset(device)
                        peak_api = lambda d: int(peak(d))
            elif dev_type == "mps" and hasattr(torch, "mps"):
                with contextlib.suppress(Exception):
                    mps = torch.mps
                    alloc = getattr(mps, "current_allocated_memory", None)
                    peak = getattr(mps, "max_memory_allocated", None)
                    if callable(alloc) and callable(peak):
                        base_alloc = int(alloc())
                        peak_api = lambda d: int(peak())

        if base_alloc is None or peak_api is None:
            return

        batch = ds.get(0, B0)
        forward_ran = False

        training_mode = bool(model.training)

        meta = dataset if isinstance(dataset, Dataset) else Dataset.for_device(device)

        try:
            from ..model.fused import Autocast
            from ..model.fused import Gradient
            from tensordict import TensorDictBase

            feats, labels, *_rest = meta.preprocess(batch)
            X = to_torch_tensor(feats)
            X = torch.atleast_2d(X)

            if X.dim() == 2 and int(X.shape[1]) == int(getattr(ops, "in_dim", X.shape[1])):
                X = X.to(device=device, non_blocking=True)

                if with_backward:
                    td = to_tensordict({"features": X})
                    if labels is not None:
                        Y = to_torch_tensor(labels)
                        Y = torch.atleast_2d(Y).to(device=device, non_blocking=True)
                        Y_flat = Y.reshape(Y.shape[0], -1)
                        td["labels_flat"] = Y_flat

                    model.train()
                    with Autocast.float(device):
                        out = model(
                            td,
                            global_loss=global_loss,
                            local_loss=local_loss,
                            loss_weights=loss_weights,
                            calibrate_output=False,
                        )

                    target: Optional[torch.Tensor] = None
                    if isinstance(out, TensorDictBase):
                        target = out.get("loss_total", None)
                        if target is None:
                            pred = out.get("pred", None)
                            if isinstance(pred, torch.Tensor):
                                target = pred
                    elif isinstance(out, torch.Tensor):
                        target = out
                    elif isinstance(out, (list, tuple)) and len(out) > 0:
                        for v in out:
                            if isinstance(v, torch.Tensor):
                                target = v
                                break
                    elif isinstance(out, dict):
                        for v in out.values():
                            if isinstance(v, torch.Tensor):
                                target = v
                                break

                    if isinstance(target, torch.Tensor):
                        loss = target
                        if loss.ndim != 0:
                            loss = loss.mean()
                        loss.backward()
                        forward_ran = True
                else:
                    td = to_tensordict({"features": X})
                    with Gradient.inference(model), Autocast.float(device):
                        _ = model(
                            td,
                            global_loss=None,
                            local_loss=None,
                            loss_weights=None,
                            calibrate_output=True,
                        )
                    forward_ran = True
        except Exception:
            forward_ran = False
        finally:
            if with_backward:
                with contextlib.suppress(Exception):
                    model.zero_grad(set_to_none=True)
            if not training_mode:
                with contextlib.suppress(Exception):
                    model.eval()

        if not forward_ran:
            batch_dev = _to_device(batch, device)

            def _touch(obj: Any) -> None:
                if isinstance(obj, torch.Tensor):
                    _ = obj.sum()
                elif isinstance(obj, (list, tuple)):
                    for v in obj:
                        _touch(v)
                elif isinstance(obj, dict):
                    for v in obj.values():
                        _touch(v)

            _touch(batch_dev)

        with contextlib.suppress(Exception):
            if dev_type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            elif dev_type == "xpu" and hasattr(torch, "xpu"):
                sync = getattr(torch.xpu, "synchronize", None)
                if callable(sync):
                    sync()
            elif dev_type == "mps" and hasattr(torch, "mps"):
                sync = getattr(torch.mps, "synchronize", None)
                if callable(sync):
                    sync()

        peak_alloc = peak_api(device)
        delta = max(0, int(peak_alloc) - int(base_alloc))
        if delta <= 0:
            return

        per_sample = int(delta // max(B0, 1))
        if floor_bytes > 0:
            per_sample = max(per_sample, floor_bytes)
        margin = 1.5 if with_backward else 1.20
        per_sample = int(per_sample * float(margin))
        if per_sample <= 0:
            return
        with contextlib.suppress(Exception):
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                t = torch.tensor(
                    [int(per_sample)],
                    device=device,
                    dtype=torch.long,
                )
                torch.distributed.all_reduce(
                    t, op=torch.distributed.ReduceOp.MAX
                )
                per_sample = int(t.item())

        try:
            Sampler._per_sample_mem_bytes = int(per_sample)
        except Exception:
            pass
        with contextlib.suppress(Exception):
            os.environ["STNET_PER_SAMPLE_MEM_BYTES"] = str(int(per_sample))

    except Exception:
        return

def _wrap_fsdp(
    target: Optional[torch.nn.Module],
    mesh: Any,
    mp_policy: MixedPrecisionPolicy,
    reshard_after_forward: bool,
    wrapped: set[int],
) -> Optional[torch.nn.Module]:
    if target is None or id(target) in wrapped:
        return target
    wrapped.add(id(target))
    return to_fsdp(
        target,
        mesh=mesh,
        mp_policy=mp_policy,
        reshard_after_forward=bool(reshard_after_forward),
        sync_module_states=True,
    )


def _get_layers(root: Optional[torch.nn.Module]) -> List[torch.nn.Module]:
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


def _initialize_tensor(
    value: Any,
    *args: Any,
    param: torch.Tensor,
    capturable: bool,
    fused: bool,
    **kwargs: Any,
) -> torch.Tensor:
    desired_device = param.device if (capturable or fused) else torch.device("cpu")
    desired_dtype = param.dtype if torch.is_floating_point(param) else torch.float32
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
        step_tensor = torch.tensor(base, dtype=desired_dtype, device=desired_device)
    return step_tensor


def _initialize_adamw(optim: torch.optim.Optimizer) -> None:
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
            state["step"] = _initialize_tensor(
                step_value, param=param, capturable=capturable, fused=fused
            )
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(param)
            if "exp_avg_sq" not in state:
                state["exp_avg_sq"] = torch.zeros_like(param)
            if amsgrad and "max_exp_avg_sq" not in state:
                state["max_exp_avg_sq"] = torch.zeros_like(param)
            optim.state[param] = state


def _scheduler(
    step: int,
    *args: Any,
    warmup_steps: int,
    start_factor: float,
    base: float,
    main_steps: int,
    emin: float,
    **kwargs: Any,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return start_factor + (1.0 - start_factor) * (step / max(1, warmup_steps))
    t = step - warmup_steps
    frac_min = emin / base if base > 0.0 else 0.0
    return frac_min + (1.0 - frac_min) * 0.5 * (
        1.0 + math.cos(math.pi * t / max(1, main_steps))
    )


def _initialize_group(backend: str, device: torch.device, local_rank: int) -> None:
    dev_id: Optional[Union[int, torch.device]] = None
    dev_type = getattr(device, "type", "cpu")
    if dev_type in ("cuda", "xpu", "mps"):
        index = (
            device.index
            if getattr(device, "index", None) is not None
            else int(os.environ.get("LOCAL_RANK", local_rank))
        )
        try:
            dev_id = torch.device(dev_type, index)
        except Exception:
            dev_id = index
    try:
        if dev_id is not None:
            torch.distributed.init_process_group(backend=backend, device_id=dev_id)
        else:
            torch.distributed.init_process_group(backend=backend)
    except TypeError:
        torch.distributed.init_process_group(backend=backend)
def loader_state_path(directory: str) -> str:
    return os.path.join(directory, _DL_STATE_FILE)


def get_tqdm(
    *args: Any, title: str, total: int, device: torch.device, **kwargs: Any
) -> Optional[tqdm]:
    try:
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return None
    except Exception:
        pass
    if int(total) <= 0:
        return None
    bar = tqdm(
        total=int(total),
        desc=f"{title} ({device.type.upper()})",
        unit="I/O < 0.01 MB/s, COM < 0.01 TFLOPS",
        bar_format="{desc}"
        + "{bar} {percentage:3.0f}% "
        + "({unit}) Elapsed: {elapsed}, Remaining: {remaining}",
        colour="green",
        position=0,
        leave=False,
        file=sys.stdout,
    )
    return bar


def _gpu_nvml_utils(device: torch.device) -> Tuple[Optional[float], Optional[float]]:
    if getattr(device, "type", "") != "cuda":
        return None, None

    idx = device.index if device.index is not None else torch.cuda.current_device()
    idx_i = int(idx)

    global _NVML_FAIL_COUNT, _NVML_BACKOFF_UNTIL

    gpu_util: Optional[float] = None
    mem_util: Optional[float] = None

    if _ensure_nvml() and _nvml is not None:
        # NVML queries are relatively expensive; throttle them (optionally) and cache
        # device handles to reduce overhead in tight training loops.
        min_interval = float(_nvml_min_interval_s())
        now = float(time.perf_counter())

        if _nvml_in_backoff(now):
            return None, None

        if min_interval > 0.0:
            cached = _NVML_UTIL_CACHE.get(idx_i)
            if cached is not None:
                ts, cg, cm = cached
                if (now - float(ts)) < min_interval:
                    return cg, cm

        with _NVML_QUERY_LOCK:
            # Re-check backoff after acquiring query lock.
            if _nvml_in_backoff(now):
                return None, None
            if min_interval > 0.0:
                cached = _NVML_UTIL_CACHE.get(idx_i)
                if cached is not None:
                    ts, cg, cm = cached
                    if (now - float(ts)) < min_interval:
                        return cg, cm

            try:
                h = _NVML_HANDLE_CACHE.get(idx_i)
                if h is None:
                    h = _nvml.nvmlDeviceGetHandleByIndex(idx_i)
                    _NVML_HANDLE_CACHE[idx_i] = h

                u = _nvml.nvmlDeviceGetUtilizationRates(h)
                mi = _nvml.nvmlDeviceGetMemoryInfo(h)
                gpu_util = float(getattr(u, "gpu", 0.0))
                if getattr(mi, "total", 0):
                    mem_util = 100.0 * float(mi.used) / float(mi.total)

                # Success: clear failure/backoff state.
                with _NVML_LOCK:
                    _NVML_FAIL_COUNT = 0
                    _NVML_BACKOFF_UNTIL = 0.0
            except Exception:
                # Handle can become invalid after driver reset; clear caches so we can retry.
                with contextlib.suppress(Exception):
                    _NVML_HANDLE_CACHE.pop(idx_i, None)
                with contextlib.suppress(Exception):
                    _NVML_UTIL_CACHE.pop(idx_i, None)

                # Failure backoff: if NVML keeps failing, temporarily disable NVML
                # queries to avoid paying exception cost in tight loops.
                fail_max = int(_nvml_fail_max())
                backoff_s = float(_nvml_backoff_s())
                trigger_backoff = False
                with _NVML_LOCK:
                    _NVML_FAIL_COUNT = int(_NVML_FAIL_COUNT) + 1
                    if backoff_s > 0.0 and int(_NVML_FAIL_COUNT) >= int(fail_max):
                        _NVML_BACKOFF_UNTIL = float(time.perf_counter()) + float(backoff_s)
                        _NVML_FAIL_COUNT = 0
                        trigger_backoff = True

                if trigger_backoff:
                    # Drop all cached handles/util metrics while backing off.
                    with contextlib.suppress(Exception):
                        _NVML_HANDLE_CACHE.clear()
                    with contextlib.suppress(Exception):
                        _NVML_UTIL_CACHE.clear()
                    with contextlib.suppress(Exception):
                        _LOGGER.warning(
                            "[NVML] repeated failures; backing off NVML queries for %.1fs "
                            "(override: STNET_NVML_BACKOFF_S, STNET_NVML_FAIL_MAX).",
                            float(backoff_s),
                        )
                gpu_util = None
                mem_util = None

            if (gpu_util is not None) or (mem_util is not None):
                _NVML_UTIL_CACHE[idx_i] = (now, gpu_util, mem_util)

    if mem_util is None:
        with contextlib.suppress(Exception):
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx_i)
            if total_bytes:
                used_bytes = float(total_bytes - free_bytes)
                mem_util = 100.0 * used_bytes / float(total_bytes)

    return gpu_util, mem_util


def _xpu_mem_util(device: torch.device) -> Optional[float]:
    if getattr(device, "type", "") != "xpu":
        return None
    if not hasattr(torch, "xpu"):
        return None
    try:
        idx = device.index if device.index is not None else torch.xpu.current_device()
        props = torch.xpu.get_device_properties(idx)
        total = getattr(props, "total_memory", None)
        if not total:
            return None
        used = float(torch.xpu.memory_allocated(idx))
        return 100.0 * used / float(total) if total > 0 else None
    except Exception:
        return None


def _mps_mem_util(device: torch.device) -> Optional[float]:
    if getattr(device, "type", "") != "mps":
        return None
    if not hasattr(torch, "mps"):
        return None
    if _psutil is None:
        return None
    try:
        vm = _psutil.virtual_memory()
        total = float(getattr(vm, "total", 0.0))
        if total <= 0.0:
            return None
        used = float(torch.mps.current_allocated_memory())
        return 100.0 * used / total
    except Exception:
        return None


def _sync_int_across_ranks(value: int, device: torch.device, src: int = 0) -> int:
    if not is_distributed():
        return int(value)
    try:
        tensor = torch.tensor([int(value)], device=device, dtype=torch.int32)
        torch.distributed.broadcast(tensor, src=src)
        return int(tensor.item())
    except Exception:
        return int(value)


def _cpu_percent_now() -> Optional[float]:
    if _psutil is None:
        return None
    try:
        return float(_psutil.cpu_percent(interval=0.0))
    except Exception:
        return None


def update_tqdm(
    bar: Optional[tqdm],
    finish: int,
    *args: Any,
    mbps: Optional[float] = None,
    tflops: Optional[float] = None,
    **kwargs: Any,
) -> None:
    if bar is None:
        return
    try:
        mbps_val = float(mbps) if mbps is not None else 0.0
    except Exception:
        mbps_val = 0.0
    try:
        tflops_val = float(tflops) if tflops is not None else 0.0
    except Exception:
        tflops_val = 0.0
    io_expr = f"I/O = {mbps_val:.2f} MB/s" if mbps_val >= 0.01 else "I/O < 0.01 MB/s"
    com_expr = (
        f"COM = {tflops_val:.2f} TFLOPS" if tflops_val >= 0.01 else "COM < 0.01 TFLOPS"
    )
    bar.unit = ", ".join([io_expr, com_expr])
    try:
        inc = int(finish)
    except Exception:
        inc = 1
    if inc > 0:
        bar.update(inc)


def epochs(
    model: Root,
    device: torch.device,
    local_rank: int,
    ops: RuntimeConfig,
    *args: Any,
    param_dtype: torch.dtype,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    sched: torch.optim.lr_scheduler.LRScheduler,
    loss_controller: LossWeightController,
    top_loss: nn.Module,
    bottom_loss: TiledLoss,
    train_loader: Any,
    val_loader: Any,
    total_epochs: int,
    scheduler_step_per_batch: bool = True,
    swa_helper: Optional[StochasticWeightAverage] = None,
    swa_start_epoch: int = 0,
    buffers_dtype: Optional[torch.dtype] = None,
    dataset: Optional[Dataset] = None,
    **kwargs: Any,
) -> None:
    from ..data.nodes import Sampler

    if train_loader is None:
        raise RuntimeError("epochs requires a training dataloader")

    meta = dataset if isinstance(dataset, Dataset) else Dataset.for_device(device)

    autocast_dtype: Optional[torch.dtype] = None
    # Compatibility: older Autocast.resolve_float_dtype may not accept `metadata=`.
    with contextlib.suppress(Exception):
        import inspect

        f = getattr(Autocast, "resolve_float_dtype", None)
        if callable(f):
            try:
                sig = inspect.signature(f)
                if "metadata" in getattr(sig, "parameters", {}):
                    autocast_dtype = f(device, metadata=meta)
                else:
                    autocast_dtype = f(device)
            except Exception:
                autocast_dtype = f(device)
    with contextlib.suppress(Exception):
        set_float32_precision(device, dtype=param_dtype, autocast_dtype=autocast_dtype)

    cpu_pool: Optional[Pool] = None
    pool_capacity: int = 0
    if device.type in {"cuda", "xpu"}:
        with contextlib.suppress(Exception):
            Memory.prefer_local_numa()
        try:
            cpu_pool = Pool(capacity=8)
            pool_capacity = int(getattr(cpu_pool, "capacity", 8))
        except Exception:
            cpu_pool = None
            pool_capacity = 0

    per_batch = getattr(train_loader, "batch_size", None)
    est_bytes_per_sample: Optional[int] = None

    with contextlib.suppress(Exception):
        v = getattr(Sampler, "_per_sample_mem_bytes", 0)
        if isinstance(v, int) and v > 0:
            est_bytes_per_sample = int(v)

    if per_batch is None or int(per_batch) <= 0 or est_bytes_per_sample is None:
        def _accumulate_sample_bytes(obj: Any) -> Tuple[Optional[int], int]:

            batch_dim: Optional[int] = None
            bytes_per_sample = 0

            def handle_tensor(t: torch.Tensor) -> None:
                nonlocal batch_dim, bytes_per_sample
                if not isinstance(t, torch.Tensor) or t.numel() <= 0:
                    return
                b = int(t.shape[0]) if t.ndim >= 1 else 1
                if batch_dim is None:
                    batch_dim = b

                if t.ndim >= 1 and b > 0:
                    one = t[:1]
                else:
                    one = t.reshape(1, -1)
                bytes_per_sample += int(one.nelement()) * int(one.element_size())

            def walk(o: Any) -> None:
                if isinstance(o, torch.Tensor):
                    handle_tensor(o)
                elif isinstance(o, TensorDictBase):
                    for v in o.values():
                        walk(v)
                elif isinstance(o, Mapping):

                    for v in o.values():
                        walk(v)
                elif isinstance(o, (list, tuple)):
                    for v in o:
                        walk(v)


            walk(obj)

            if bytes_per_sample <= 0:
                return None, 0
            return batch_dim, bytes_per_sample

        try:
            it = iter(train_loader)
            sample = next(it)

            bs, bytes_ps = _accumulate_sample_bytes(sample)

            if (per_batch is None or int(per_batch) <= 0) and bs is not None and bs > 0:
                per_batch = int(bs)


            if est_bytes_per_sample is None and bytes_ps > 0:
                est_bytes_per_sample = int(bytes_ps)
        except StopIteration:

            if per_batch is None or int(per_batch) <= 0:
                per_batch = 1
        except Exception:

            if per_batch is None or int(per_batch) <= 0:
                per_batch = 1

    if per_batch is None or per_batch <= 0:
        per_batch = 1

    from ..data.pipeline import BatchPolicy

    fixed_accum = 2 if getattr(device, "type", "cpu") == "cpu" else 4
    min_grad_accum = fixed_accum
    max_grad_accum = fixed_accum
    dev_margin = 0.8
    host_margin = 0.8
    budget_slack = 1.25
    with contextlib.suppress(Exception):
        v = os.environ.get("STNET_BUDGET_SLACK")
        if v is not None and str(v).strip():
            budget_slack = float(v)
    budget_slack = max(1.0, min(4.0, float(budget_slack)))

    dev_budget_ratio = 1.0
    dev_budget_min_bytes = 0
    dev_budget_max_bytes: Optional[int] = None

    host_budget_ratio = 1.0
    host_budget_min_bytes = 0
    host_budget_max_bytes: Optional[int] = None

    with contextlib.suppress(Exception):
        v = os.environ.get("STNET_DEVICE_MARGIN")
        if v is not None and str(v).strip():
            dev_margin = float(v)
    with contextlib.suppress(Exception):
        v = os.environ.get("STNET_HOST_MARGIN")
        if v is not None and str(v).strip():
            host_margin = float(v)

    with contextlib.suppress(Exception):
        v = os.environ.get("STNET_DEVICE_BUDGET_RATIO")
        if v is not None and str(v).strip():
            dev_budget_ratio = float(v)
    with contextlib.suppress(Exception):
        v = os.environ.get("STNET_DEVICE_BUDGET_MIN_BYTES")
        if v is not None and str(v).strip():
            dev_budget_min_bytes = int(v)
    with contextlib.suppress(Exception):
        v = os.environ.get("STNET_DEVICE_BUDGET_MAX_BYTES")
        if v is not None and str(v).strip():
            dev_budget_max_bytes = int(v)

    with contextlib.suppress(Exception):
        v = os.environ.get("STNET_HOST_BUDGET_RATIO")
        if v is not None and str(v).strip():
            host_budget_ratio = float(v)
    with contextlib.suppress(Exception):
        v = os.environ.get("STNET_HOST_BUDGET_MIN_BYTES")
        if v is not None and str(v).strip():
            host_budget_min_bytes = int(v)
    with contextlib.suppress(Exception):
        v = os.environ.get("STNET_HOST_BUDGET_MAX_BYTES")
        if v is not None and str(v).strip():
            host_budget_max_bytes = int(v)

    dev_margin = max(0.0, min(1.0, float(dev_margin)))
    host_margin = max(0.0, min(1.0, float(host_margin)))

    dev_budget_ratio = max(0.0, min(1.0, float(dev_budget_ratio)))
    host_budget_ratio = max(0.0, min(1.0, float(host_budget_ratio)))

    dev_budget_min_bytes = max(0, int(dev_budget_min_bytes))
    host_budget_min_bytes = max(0, int(host_budget_min_bytes))
    dev_budget_max_bytes = (
        None if dev_budget_max_bytes is None else max(0, int(dev_budget_max_bytes))
    )
    host_budget_max_bytes = (
        None if host_budget_max_bytes is None else max(0, int(host_budget_max_bytes))
    )
    if dev_budget_max_bytes is not None and int(dev_budget_max_bytes) <= 0:
        dev_budget_max_bytes = None
    if host_budget_max_bytes is not None and int(host_budget_max_bytes) <= 0:
        host_budget_max_bytes = None

    tpl: Optional[BatchPolicy] = None
    if est_bytes_per_sample is not None and est_bytes_per_sample > 0 and max_grad_accum > 0:
        try:
            effective_streams = 1 + max(0, pool_capacity)

            tpl = BatchPolicy(
                sample_bytes=int(est_bytes_per_sample),
                host_sample_bytes=int(est_bytes_per_sample),
                prebatch=1,
                prefetch_factor=int(
                    os.environ.get("STNET_HOST_PREFETCH_FACTOR") or "4"
                ),
                num_workers=getattr(train_loader, "num_workers", 0),
                num_streams=int(effective_streams),
                max_concurrency=1,
                min_batch=1,
                max_batch=max_grad_accum,
                host_margin=float(host_margin),
                device_margin=float(dev_margin),
                host_budget_ratio=float(host_budget_ratio),
                host_budget_min_bytes=int(host_budget_min_bytes),
                host_budget_max_bytes=(
                    None if host_budget_max_bytes is None else int(host_budget_max_bytes)
                ),
                device_budget_ratio=float(dev_budget_ratio),
                device_budget_min_bytes=int(dev_budget_min_bytes),
                device_budget_max_bytes=(
                    None if dev_budget_max_bytes is None else int(dev_budget_max_bytes)
                ),
            )
        except Exception:
            tpl = None

    safe_host_bytes: Optional[int] = None
    safe_host_total: Optional[int] = None
    safe_dev_bytes: Optional[int] = None
    safe_dev_total: Optional[int] = None
    max_from_mem: Optional[int] = None
    if tpl is not None:
        try:
            host_mem = Memory.available()
            if host_mem is not None and host_mem >= 0:
                safe_host_bytes = int(host_mem)
            with contextlib.suppress(Exception):
                host_total = Memory.total()
                if host_total is not None and host_total > 0:
                    safe_host_total = int(host_total)
            safe_dev_bytes, safe_dev_total = _device_mem_get_info(device)
            if tpl.device_budget_max_bytes is None or tpl.host_budget_max_bytes is None:
                try:
                    target_total_samples = max(1, int(per_batch or 1)) * max(
                        1, int(min_grad_accum)
                    )
                    new_dev_cap: Optional[int] = tpl.device_budget_max_bytes
                    new_host_cap: Optional[int] = tpl.host_budget_max_bytes
                    if new_dev_cap is None and int(tpl.sample_bytes or 0) > 0:
                        base_dev = int(tpl.sample_bytes) * int(target_total_samples)
                        cap_dev = int(float(base_dev) * float(budget_slack))
                        if safe_dev_total is not None and int(safe_dev_total) > 0:
                            cap_dev = min(int(cap_dev), int(safe_dev_total))
                        cap_dev = max(0, int(cap_dev))
                        new_dev_cap = None if cap_dev <= 0 else cap_dev

                    if new_host_cap is None and int(tpl.host_sample_bytes or 0) > 0:
                        inflight = int(tpl.host_inflight_batches_per_proc())
                        lw = max(1, int(getattr(tpl, "local_world_size", 1) or 1))
                        base_host = int(tpl.host_sample_bytes) * max(1, inflight) * max(
                            1, lw
                        ) * int(target_total_samples)
                        cap_host = int(float(base_host) * float(budget_slack))
                        if safe_host_total is not None and int(safe_host_total) > 0:
                            cap_host = min(int(cap_host), int(safe_host_total))
                        cap_host = max(0, int(cap_host))
                        new_host_cap = None if cap_host <= 0 else cap_host

                    if (new_dev_cap != tpl.device_budget_max_bytes) or (new_host_cap != tpl.host_budget_max_bytes):
                        tpl = dataclasses.replace(
                            tpl,
                            device_budget_max_bytes=new_dev_cap,
                            host_budget_max_bytes=new_host_cap,
                        )
                except Exception:
                    pass

            if safe_host_bytes is not None or safe_dev_bytes is not None:
                total_samples_cap = tpl.suggest_batch(
                    dev_free=safe_dev_bytes,
                    host_free=safe_host_bytes,
                    dev_total=safe_dev_total,
                    host_total=safe_host_total,
                )
                if total_samples_cap > 0:
                    max_from_mem = max(
                        1, int(total_samples_cap) // int(per_batch or 1)
                    )
        except Exception:
            safe_host_bytes = None
            safe_host_total = None
            safe_dev_bytes = None
            safe_dev_total = None

    if max_from_mem is not None:
        max_grad_accum = max(
            int(min_grad_accum),
            min(int(max_grad_accum), int(max_from_mem)),
        )

    grad_accum_steps: int = int(min_grad_accum)
    grad_accum_steps = _sync_int_across_ranks(grad_accum_steps, device=device, src=0)

    proc = None
    if _psutil is not None:
        try:
            proc = _psutil.Process(os.getpid())
        except Exception:
            proc = None

    gpu_util_ema: Optional[float] = None
    mem_util_ema: Optional[float] = None
    util_alpha: float = 0.2
    global_step: int = 0
    util_adjust_interval: int = 0
    util_warmup_steps: int = 0

    def _cast_fp_buffers(module: torch.nn.Module, dtype: torch.dtype) -> None:
        # Cast *only* BatchNorm/SyncBatchNorm buffers, and skip precision-exempt modules.
        if dtype is None:
            return

        def _is_exempt(m: torch.nn.Module) -> bool:
            return bool(getattr(m, "__stnet_precision_exempt__", False))

        with torch.no_grad():
            for mod in module.modules():
                if _is_exempt(mod):
                    continue
                if isinstance(
                    mod,
                    (
                        torch.nn.BatchNorm1d,
                        torch.nn.BatchNorm2d,
                        torch.nn.BatchNorm3d,
                        torch.nn.SyncBatchNorm,
                    ),
                ):
                    for name, buf in mod._buffers.items():
                        if buf is None or not isinstance(buf, torch.Tensor):
                            continue
                        if not buf.is_floating_point():
                            continue
                        if buf.dtype == dtype:
                            continue
                        try:
                            mod._buffers[name] = buf.to(dtype=dtype)
                        except Exception:
                            pass

    if buffers_dtype is not None:
        target_for_buffers = model.module if hasattr(model, "module") else model
        _cast_fp_buffers(target_for_buffers, buffers_dtype)

    model_for_hist = model.module if hasattr(model, "module") else model
    hist: Optional[History] = None
    maybe_hist = getattr(model_for_hist, "logger", None)
    if isinstance(maybe_hist, History):
        hist = maybe_hist
    if hist is None:
        maybe_hist = getattr(model_for_hist, "history", None)
        if isinstance(maybe_hist, History):
            hist = maybe_hist
    if hist is None:
        hist = History()
        try:
            setattr(model_for_hist, "logger", hist)
        except Exception:
            pass

    if isinstance(hist, History):
        start_ns = posix_time("Asia/Seoul")
        start_sec = round(float(start_ns) / 1e9, 6)
        hist.start_session(start_sec)
        hist.set_epochs(total_epochs)

        os_name = platform.system()
        match os_name:
            case 'Linux':
                pretty = None
                with contextlib.suppress(Exception):
                    if os.path.exists("/etc/os-release"):
                        with open("/etc/os-release", "r", encoding="utf-8") as f:
                            for line in f:
                                if line.startswith("PRETTY_NAME="):
                                    pretty = line.strip().split("=", 1)[1].strip().strip('"')
                                    break
                os_full = pretty or f"{os_name} {platform.release()}"
            case 'Darwin':
                ver, _, _ = platform.mac_ver()
                os_full = f"macOS {ver or platform.release()}"
            case 'Windows':
                ver = platform.version()
                rel = platform.release()
                os_full = f"Windows {rel} {ver}"
            case _:
                os_full = f"{os_name} {platform.release()}"

        kernel = platform.release()
        arch_list = [platform.machine(), platform.processor() or ""]
        cpu_list: List[str] = []
        proc = platform.processor()
        if proc:
            cpu_list.append(proc)

        try:
            ram_bytes = Memory.total()
            ram_gb = int(round(float(ram_bytes) / (1024 ** 3)))
        except Exception:
            ram_gb = 0

        py_ver = platform.python_version()

        backend_list: List[str] = []
        if torch.cuda.is_available():
            backend_list.append("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            backend_list.append("xpu")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            backend_list.append("mps")
        backend_list.append("cpu")

        hist.set_system_info(
            os_name=os_full,
            kernel=kernel,
            cpu_list=cpu_list,
            arch_list=arch_list,
            ram_gb=ram_gb,
            python_version=py_ver,
            backends=backend_list,
        )

    model_for_scaler = model.module if hasattr(model, "module") else model
    scaler_x_device = model_for_scaler.scaler.x_mean.device
    scaler_y_device = model_for_scaler.scaler.y_mean.device
    with torch.no_grad():
        x_count: int = 0
        x_sum: Optional[torch.Tensor] = None
        x_sum_sq: Optional[torch.Tensor] = None
        y_count: int = 0
        y_sum: Optional[torch.Tensor] = None
        y_sum_sq: Optional[torch.Tensor] = None

        for batch in train_loader:
            feats = batch["features"]
            labs = batch["labels"]
            if feats.ndim == 3 and feats.shape[1] == 1:
                feats = feats.reshape(feats.shape[0], -1)

            xf = feats.to(device=scaler_x_device, dtype=torch.float64)
            n_x = xf.shape[0]
            if n_x > 0:
                x_count += n_x
                sx = xf.sum(dim=0)
                sx2 = (xf * xf).sum(dim=0)
                if x_sum is None:
                    x_sum = sx
                    x_sum_sq = sx2
                else:
                    x_sum += sx
                    x_sum_sq += sx2

            yf = labs.to(device=scaler_y_device, dtype=torch.float64)
            if yf.ndim >= 2:
                yf = yf.reshape(yf.shape[0], -1)
            n_y = yf.shape[0]
            if n_y > 0:
                y_count += n_y
                sy = yf.sum(dim=0)
                sy2 = (yf * yf).sum(dim=0)
                if y_sum is None:
                    y_sum = sy
                    y_sum_sq = sy2
                else:
                    y_sum += sy
                    y_sum_sq += sy2
        if is_distributed():
            x_count_t = torch.tensor(
                float(x_count), device=scaler_x_device, dtype=torch.float64
            )
            torch.distributed.all_reduce(x_count_t, op=torch.distributed.ReduceOp.SUM)
            x_count = int(x_count_t.item())
            if x_sum is not None:
                torch.distributed.all_reduce(x_sum, op=torch.distributed.ReduceOp.SUM)
            if x_sum_sq is not None:
                torch.distributed.all_reduce(x_sum_sq, op=torch.distributed.ReduceOp.SUM)

            y_count_t = torch.tensor(
                float(y_count), device=scaler_y_device, dtype=torch.float64
            )
            torch.distributed.all_reduce(y_count_t, op=torch.distributed.ReduceOp.SUM)
            y_count = int(y_count_t.item())
            if y_sum is not None:
                torch.distributed.all_reduce(y_sum, op=torch.distributed.ReduceOp.SUM)
            if y_sum_sq is not None:
                torch.distributed.all_reduce(y_sum_sq, op=torch.distributed.ReduceOp.SUM)

        eps = float(model_for_scaler.scaler.eps)
        if x_count > 0 and x_sum is not None and x_sum_sq is not None:
            mean_x = x_sum / float(x_count)
            var_x = (x_sum_sq / float(x_count)) - mean_x * mean_x
            std_x = torch.sqrt(var_x.clamp_min(eps))
            if model_for_scaler.scaler.x_mean.shape != mean_x.shape:
                model_for_scaler.scaler.x_mean.resize_(mean_x.shape)
            if model_for_scaler.scaler.x_std.shape != std_x.shape:
                model_for_scaler.scaler.x_std.resize_(std_x.shape)
            model_for_scaler.scaler.x_mean.copy_(mean_x)
            model_for_scaler.scaler.x_std.copy_(std_x)

        if y_count > 0 and y_sum is not None and y_sum_sq is not None:
            mean_y = y_sum / float(y_count)
            var_y = (y_sum_sq / float(y_count)) - mean_y * mean_y
            std_y = torch.sqrt(var_y.clamp_min(eps))
            if model_for_scaler.scaler.y_mean.shape != mean_y.shape:
                model_for_scaler.scaler.y_mean.resize_(mean_y.shape)
            if model_for_scaler.scaler.y_std.shape != std_y.shape:
                model_for_scaler.scaler.y_std.resize_(std_y.shape)
            model_for_scaler.scaler.y_mean.copy_(mean_y)
            model_for_scaler.scaler.y_std.copy_(std_y)

    in_dim = int(ops.in_dim)

    use_timer = (
        (device.type == "cuda" and hasattr(torch.cuda, "Event")) or
        (device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "Event"))
    )
    train_steps = _num_batches(train_loader)
    val_steps = _num_batches(val_loader)
    total_updates = int(total_epochs) * (int(train_steps) + int(val_steps))

    if train_steps > 0:
        util_adjust_interval = max(10, int(train_steps * 0.05))
        util_warmup_steps = max(
            util_adjust_interval,
            min(int(train_steps), max(50, int(train_steps * 0.1))),
        )

    status_bar = (
        get_tqdm(title="Training", total=total_updates, device=device)
        if local_rank == 0
        else None
    )
    scheduler_step_per_batch = bool(scheduler_step_per_batch)
    swa_enabled = swa_helper is not None
    swa_start_epoch = max(0, int(swa_start_epoch))
    swa_has_updated = False
    prev_io_time = 0.0
    prev_comp_time = 0.0
    prev_io_bytes = 0.0
    prev_flops = 0.0
    prev_samples = 0.0

    join_context = joining(model=model, optimizer=optimizer)
    with join_context:

        with contextlib.suppress(Exception):
            get_tlb().pin_thread()
        from typing import Dict

        pool_handles: Dict[int, object] = {}

        def _pin_tensor(*tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            if device.type not in ("cuda", "xpu") or cpu_pool is None:
                return tensors
            from typing import List
            out: List[torch.Tensor] = []
            for t in tensors:
                if torch.is_tensor(t) and t.device.type == "cpu":
                    if hasattr(t, "is_pinned") and t.is_pinned():
                        out.append(t)
                        continue
                    buf, h = cpu_pool.get(tuple(t.shape), t.dtype, return_handle=True)
                    buf.copy_(t, non_blocking=False)
                    if h is not None:
                        pool_handles[id(buf)] = h
                    out.append(buf)
                else:
                    out.append(t)
            return tuple(out)

        def _to_device_with_stream(tensor: torch.Tensor) -> torch.Tensor:
            if not torch.is_tensor(tensor):
                return tensor
            if device.type in ("cuda", "xpu") and tensor.device.type == "cpu":
                try:
                    if not (hasattr(tensor, "is_pinned") and tensor.is_pinned()):
                        pinned = torch.empty_like(tensor, device="cpu", pin_memory=True)
                        pinned.copy_(tensor, non_blocking=False)
                    else:
                        pinned = tensor
                    backend = getattr(torch, device.type, None)
                    if backend is None or not hasattr(backend, "current_stream") or not hasattr(backend, "Event"):
                        tensor_dev = pinned.to(device, non_blocking=False)
                        h = pool_handles.pop(id(tensor), None)
                        if h is not None:
                            with contextlib.suppress(Exception):
                                cpu_pool.release(h)
                        return tensor_dev
                    tensor_dev = pinned.to(device, non_blocking=True)
                    stream = backend.current_stream(device)
                    pinned.record_stream(stream)
                    h = pool_handles.pop(id(tensor), None)
                    if h is not None:
                        with contextlib.suppress(Exception):
                            evt = backend.Event()
                            evt.record(stream)
                            cpu_pool.release_after(h, evt)
                    return tensor_dev
                except Exception:
                    return tensor.to(device, non_blocking=(device.type in ("cuda", "xpu")))
            return tensor.to(device, non_blocking=(device.type in ("cuda", "xpu")))

        def _resolve_train_process_group(meta, model):
            pg = None
            with contextlib.suppress(Exception):
                pg = getattr(meta, "process_group", None)
            with contextlib.suppress(Exception):
                if pg is None:
                    pg = getattr(meta, "distributed_process_group", None)
            with contextlib.suppress(Exception):
                if pg is None:
                    tm = model.module if hasattr(model, "module") else model
                    pg = getattr(tm, "process_group", None)
            with contextlib.suppress(Exception):
                if pg is None:
                    tm = model.module if hasattr(model, "module") else model
                    pg = getattr(tm, "distributed_process_group", None)
            return pg

        def _get_ws_for_pg(pg):
            try:
                if pg is None:
                    return max(1, int(get_world_size()))
                return int(torch.distributed.get_world_size(group=pg))
            except Exception:
                return max(1, int(get_world_size()))

        def _all_reduce_sum_in_pg(t: torch.Tensor, pg):
            if pg is None:
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            else:
                torch.distributed.all_reduce(
                    t,
                    op=torch.distributed.ReduceOp.SUM,
                    group=pg,
                )

        for epoch_idx in range(int(total_epochs)):
            if is_distributed():
                target_module = model.module if hasattr(model, "module") else model
                distributed_sync(target_module, device=device)
            flop_breakdown_epoch: Dict[str, float] = {}
            io_time: float = 0.0
            comp_time: float = 0.0
            io_bytes: float = 0.0
            flops: float = 0.0
            train_samples_epoch: float = 0.0
            flop_counter_train = FlopCounter(model, mode="train", device=device)
            with flop_counter_train:
                model.train()
                train_pg = _resolve_train_process_group(meta, model) if is_distributed() else None
                global_step = 0
                optimizer.zero_grad(set_to_none=True)
                t_fetch_start = time.perf_counter_ns()
                total_batches = len(train_loader)
                train_accum_since_last = 0
                lw_top_sum: Optional[torch.Tensor] = None
                lw_bottom_sum: Optional[torch.Tensor] = None
                lw_count: int = 0
                for step_idx, _raw in enumerate(train_loader):
                    train_accum_since_last += 1
                    while True:
                        try:
                            if device.type in ("cuda", "xpu", "mps"):
                                t_ready = time.perf_counter_ns()
                                if use_timer:
                                    h2d_s_ev, h2d_e_ev = (
                                        torch.Event(device=device, enable_timing=True),
                                        torch.Event(device=device, enable_timing=True),
                                    )
                                    h2d_s_ev.record()
                                    X, Y = meta.batch_to_device(_raw, device)
                                    h2d_e_ev.record()
                                    h2d_e_ev.synchronize()
                                    h2d_s = float(h2d_s_ev.elapsed_time(h2d_e_ev)) / 1000.0
                                else:
                                    t_h2d_s = time.perf_counter_ns()
                                    X, Y = meta.batch_to_device(_raw, device)
                                    t_h2d_e = time.perf_counter_ns()
                                    h2d_s = (t_h2d_e - t_h2d_s) / 1_000_000_000.0
                            else:
                                feat, label, *_ = meta.preprocess(_raw)
                                X = to_torch_tensor(feat)
                                Y = to_torch_tensor(label)
    
                                t_ready = time.perf_counter_ns()
                                X, Y = _pin_tensor(X, Y)
    
                                if use_timer:
                                    h2d_s_ev, h2d_e_ev = (
                                        torch.Event(device=device, enable_timing=True),
                                        torch.Event(device=device, enable_timing=True),
                                    )
                                    h2d_s_ev.record()
                                    X = _to_device_with_stream(X)
                                    Y = _to_device_with_stream(Y)
                                    h2d_e_ev.record()
                                    h2d_e_ev.synchronize()
                                    h2d_s = float(h2d_s_ev.elapsed_time(h2d_e_ev)) / 1000.0
                                else:
                                    t_h2d_s = time.perf_counter_ns()
                                    X = _to_device_with_stream(X)
                                    Y = _to_device_with_stream(Y)
                                    t_h2d_e = time.perf_counter_ns()
                                    h2d_s = (t_h2d_e - t_h2d_s) / 1_000_000_000.0
                            X = torch.atleast_2d(X)
                            if X.dim() != 2:
                                raise RuntimeError(
                                    f"features.ndim={X.dim()} (expect 2). got shape={tuple(X.shape)}"
                                )
                            if X.shape[1] != in_dim:
                                raise RuntimeError(
                                    f"feature dim mismatch: X.shape[1]={X.shape[1]} != in_dim={in_dim}"
                                )
                            train_samples_epoch += float(X.shape[0])
                            wait_s = (t_ready - t_fetch_start) / 1_000_000_000.0
                            io_time += float(wait_s + h2d_s)
                            with contextlib.suppress(Exception):
                                io_bytes += float(
                                    X.element_size() * X.nelement()
                                    + Y.element_size() * Y.nelement()
                                )
                            should_sync = ((step_idx + 1) % max(1, grad_accum_steps) == 0) or (
                                step_idx + 1 == total_batches
                            )
                            
                            if use_timer:
                                ev_s, ev_e = (
                                    torch.Event(device=device, enable_timing=True),
                                    torch.Event(device=device, enable_timing=True),
                                )
                                ev_s.record()
                            else:
                                t_comp_s = time.perf_counter_ns()
                            with no_sync(
                                model, enable=(grad_accum_steps > 1 and (not should_sync))
                            ):
                                with flop_counter_train.step(display=False) as train_counter:
                                    with contextlib.suppress(Exception):
                                        mark_step = getattr(
                                            getattr(torch, "compiler", None),
                                            "cudagraph_mark_step_begin",
                                            None,
                                        )
                                        if callable(mark_step):
                                            mark_step()
                                    with Autocast.float(device):
                                        Y_flat = Y.reshape(Y.shape[0], -1)
                                        if Y_flat.device != device or Y_flat.dtype != param_dtype:
                                            Y_flat = Y_flat.to(device, dtype=param_dtype, non_blocking=True)
                                        td = to_tensordict(
                                            {"features": X, "labels_flat": Y_flat}
                                        )
                                        model_out = model(
                                            td,
                                            global_loss=top_loss,
                                            local_loss=bottom_loss,
                                            loss_weights=loss_controller.weights(),
                                        )
                                    if isinstance(model_out, TensorDictBase):
                                        td = model_out
                                        y_hat = td.get("pred")
                                        loss_val = td.get("loss_total", None)
                                        loss_top_val = td.get("loss_top", None)
                                        loss_bottom_val = td.get("loss_bottom", None)
                                        if (
                                            isinstance(loss_val, torch.Tensor)
                                            and loss_val.ndim > 0
                                        ):
                                            loss_val = loss_val.mean()
                                        if (
                                            isinstance(loss_top_val, torch.Tensor)
                                            and loss_top_val.ndim > 0
                                        ):
                                            loss_top_val = loss_top_val.mean()
                                        if (
                                            isinstance(loss_bottom_val, torch.Tensor)
                                            and loss_bottom_val.ndim > 0
                                        ):
                                            loss_bottom_val = loss_bottom_val.mean()
                                    else:
                                        loss_top_val = None
                                        loss_bottom_val = None
                                        y_hat, loss_val = model_out
                                    if loss_val is None:
                                        raise RuntimeError(
                                            "Model returned no loss value during training. "
                                            "Ensure loss functions are provided and returning valid outputs."
                                        )
                                    if not isinstance(loss_val, torch.Tensor):
                                        loss_val = torch.as_tensor(
                                            loss_val, device=device, dtype=param_dtype
                                        )
                                    else:
                                        loss_val = loss_val.to(
                                            device=device, dtype=param_dtype
                                        )
                                    accum_scale = max(1, grad_accum_steps)
                                    loss_for_backprop = loss_val / float(accum_scale)
                                    loss_for_backprop = loss_for_backprop.clone()
                                    scaler.scale(loss_for_backprop).backward()

                                    if (
                                        loss_top_val is not None
                                        or loss_bottom_val is not None
                                    ):
                                        lw_count += 1
                                        if isinstance(loss_top_val, torch.Tensor):
                                            v = loss_top_val.detach()
                                            lw_top_sum = v if lw_top_sum is None else (lw_top_sum + v)
                                        if isinstance(loss_bottom_val, torch.Tensor):
                                            v = loss_bottom_val.detach()
                                            lw_bottom_sum = (
                                                v if lw_bottom_sum is None else (lw_bottom_sum + v)
                                            )
                                    if should_sync:
                                        scaler.unscale_(optimizer)
                                        scaler.step(optimizer)
                                        scaler.update()
                                        optimizer.zero_grad(set_to_none=True)
                                        if scheduler_step_per_batch:
                                            with contextlib.suppress(Exception):
                                                sched.step()

                                        if lw_count > 0:
                                            top_avg_t: Optional[torch.Tensor] = (
                                                (lw_top_sum / float(lw_count))
                                                if lw_top_sum is not None
                                                else None
                                            )
                                            bottom_avg_t: Optional[torch.Tensor] = (
                                                (lw_bottom_sum / float(lw_count))
                                                if lw_bottom_sum is not None
                                                else None
                                            )
                                            if is_distributed():
                                                ws = _get_ws_for_pg(train_pg)
                                                if top_avg_t is not None:
                                                    _all_reduce_sum_in_pg(top_avg_t, train_pg)
                                                    top_avg_t = top_avg_t / float(ws)
                                                if bottom_avg_t is not None:
                                                    _all_reduce_sum_in_pg(bottom_avg_t, train_pg)
                                                    bottom_avg_t = bottom_avg_t / float(ws)
                                            loss_controller.update(top_avg_t, bottom_avg_t)

                                        lw_top_sum = None
                                        lw_bottom_sum = None
                                        lw_count = 0
                                    with contextlib.suppress(Exception):
                                        flops += max(0.0, float(train_counter.get_total_flops()))
                                    breakdown_getter = getattr(
                                        train_counter, "get_manual_breakdown", None
                                    )
                                    if callable(breakdown_getter):
                                        for name, value in breakdown_getter().items():
                                            with contextlib.suppress(Exception):
                                                flop_breakdown_epoch[name] = (
                                                    flop_breakdown_epoch.get(name, 0.0)
                                                    + float(value)
                                                )
    
                            if should_sync:
                                global_step += 1
    
                                if device.type == "cuda":
                                    util_now, mem_now = _gpu_nvml_utils(device)
                                elif device.type == "xpu":
                                    util_now, mem_now = None, _xpu_mem_util(device)
                                elif device.type == "mps":
                                    util_now, mem_now = None, _mps_mem_util(device)
                                else:
                                    util_now, mem_now = None, None
    
                                if util_now is not None:
                                    util_now = float(util_now)
                                    if gpu_util_ema is None:
                                        gpu_util_ema = util_now
                                    else:
                                        gpu_util_ema = (1.0 - util_alpha) * gpu_util_ema + util_alpha * util_now
                                if mem_now is not None:
                                    mem_now = float(mem_now)
                                    if mem_util_ema is None:
                                        mem_util_ema = mem_now
                                    else:
                                        mem_util_ema = (1.0 - util_alpha) * mem_util_ema + util_alpha * mem_now
    
                                if (
                                    util_adjust_interval > 0
                                    and global_step >= util_warmup_steps
                                    and (global_step % util_adjust_interval == 0)
                                ):
                                    new_grad_accum = grad_accum_steps
    
                                    util_frac: Optional[float] = None
                                    mem_frac: Optional[float] = None
    
                                    if gpu_util_ema is not None:
                                        util_frac = max(0.0, min(1.0, gpu_util_ema / 100.0))
                                    if mem_util_ema is not None:
                                        mem_frac = max(0.0, min(1.0, mem_util_ema / 100.0))
    
                                    if util_frac is None:
                                        total_t_local = float(io_time + comp_time)
                                        if total_t_local > 0.0:
                                            util_frac = max(
                                                0.0,
                                                min(1.0, float(comp_time) / total_t_local),
                                            )
                                        else:
                                            util_frac = 0.0
    
                                    if util_frac is not None:
                                        if mem_frac is not None:
                                            if util_frac < 0.88 and mem_frac < 0.90:
                                                new_grad_accum = min(max_grad_accum, grad_accum_steps + 1)
                                            elif util_frac > 0.97 or mem_frac > 0.92:
                                                new_grad_accum = max(min_grad_accum, grad_accum_steps - 1)
                                        else:
                                            if util_frac < 0.88:
                                                new_grad_accum = min(max_grad_accum, grad_accum_steps + 1)
                                            elif util_frac > 0.97:
                                                new_grad_accum = max(min_grad_accum, grad_accum_steps - 1)
    
                                    host_avail_now: Optional[int] = None
                                    host_total_now: Optional[int] = None
                                    with contextlib.suppress(Exception):
                                        host_avail_now = Memory.available()
                                        host_total_now = Memory.total()
                                    host_low = False
                                    if host_avail_now is not None and host_avail_now > 0:
                                        host_low_abs = host_avail_now < (512 * 1024 * 1024)
                                        host_low_rel = False
                                        if host_total_now is not None and host_total_now > 0:
                                            host_low_rel = float(host_avail_now) / float(host_total_now) < 0.10
                                        host_low = host_low_abs or host_low_rel
                                    if host_low:
                                        if new_grad_accum > grad_accum_steps:
                                            new_grad_accum = grad_accum_steps
                                        if grad_accum_steps > min_grad_accum:
                                            new_grad_accum = min_grad_accum
    
                                    if new_grad_accum != grad_accum_steps:
                                        new_grad_accum = _sync_int_across_ranks(
                                            new_grad_accum, device=device, src=0
                                        )
                                        logging.info(
                                            f"[epochs] adjusted grad_accum_steps={new_grad_accum} "
                                            f"(gpu_util_ema={gpu_util_ema}, mem_util_ema={mem_util_ema})"
                                        )
                                        grad_accum_steps = new_grad_accum
                            if use_timer:
                                ev_e.record()
                                ev_e.synchronize()
                                comp_time += float(ev_s.elapsed_time(ev_e)) / 1000.0
                            else:
                                comp_time += (time.perf_counter_ns() - t_comp_s) / 1_000_000_000.0
                            with contextlib.suppress(Exception):
                                cudagraph_step_end()
                            if local_rank == 0 and should_sync:
                                io_elapsed = prev_io_time + float(io_time)
                                io_transferred = prev_io_bytes + float(io_bytes)
                                comp_elapsed = prev_comp_time + float(comp_time)
                                flop_total = prev_flops + float(flops)
                                mbps_cur = io_transferred / max(io_elapsed, 1e-06) / MB_DIV
                                tflops_cur = (
                                    flop_total / max(comp_elapsed, 1e-06) / 1_000_000_000_000.0
                                )
                                update_tqdm(
                                    status_bar,
                                    finish=train_accum_since_last,
                                    mbps=mbps_cur,
                                    tflops=tflops_cur,
                                )
                                train_accum_since_last = 0
                            if isinstance(hist, History):
                                try:
                                    if train_steps <= 0 or step_idx % max(1, int(train_steps * 0.01)) == 0:
                                        hist.record_batch(X, Y)
                                except Exception:
                                    pass
                            t_fetch_start = time.perf_counter_ns()
                            if cpu_pool is not None and ((step_idx + 1) & 255) == 0:
                                with contextlib.suppress(Exception):
                                    cpu_pool.collect()

                            with contextlib.suppress(Exception):
                                _oom_retry_clear(train_loader, "train", step_idx)

                            break

                        except RuntimeError as e:
                            msg = str(e).lower()
                            if "out of memory" in msg:
                                oom_try = _oom_retry_inc(train_loader, "train", step_idx)
                                max_tries = _oom_max_retries("train")
                                if oom_try <= 1:
                                    _LOGGER.error(
                                        "[epochs] OOM during train step %d (global_step=%d). "
                                        "Trying to reduce microbatch / grad_accum and retry same batch. (try=%d/%d)",
                                        step_idx,
                                        global_step,
                                        oom_try,
                                        max_tries,
                                    )
                                else:
                                    _LOGGER.warning(
                                        "[epochs] OOM during train step %d (global_step=%d). Retrying. (try=%d/%d)",
                                        step_idx,
                                        global_step,
                                        oom_try,
                                        max_tries,
                                    )

                                if max_tries > 0 and oom_try > max_tries:
                                    if _oom_skip_enabled("train"):
                                        _LOGGER.error(
                                            "[epochs] OOM storm: exceeded retry budget (try=%d/%d) at train step %d (global_step=%d). "
                                            "Skipping this batch.",
                                            oom_try,
                                            max_tries,
                                            step_idx,
                                            global_step,
                                        )
                                        with contextlib.suppress(Exception):
                                            _oom_retry_clear(train_loader, "train", step_idx)
                                        with contextlib.suppress(Exception):
                                            empty_device_cache(device=device, do_gc=False, min_interval_s=0.0)
                                        with contextlib.suppress(Exception):
                                            optimizer.zero_grad(set_to_none=True)
                                        break
                                    raise

                                # Also request input batch scale-down for NEXT batches (true runtime recovery).
                                # NOTE: scale controller is per-session/per-loader (see SamplerScale).
                                try:
                                    scale_ctl = None
                                    obj = train_loader
                                    for _ in range(4):
                                        if obj is None:
                                            break
                                        scale_ctl = getattr(obj, "_stnet_sampler_scale", None)
                                        if scale_ctl is not None:
                                            break
                                        obj = getattr(obj, "_src", None) or getattr(obj, "src", None)
                                    if scale_ctl is not None:
                                        prev = None
                                        with contextlib.suppress(Exception):
                                            prev = float(scale_ctl.get())
                                        with contextlib.suppress(Exception):
                                            scale_ctl.request_scale_down(_oom_scale_down_factor(oom_try))
                                        with contextlib.suppress(Exception):
                                            cur = float(scale_ctl.get())
                                            if prev is not None and cur < prev:
                                                _log_sampler_scale_rate_limited(
                                                    logger=_LOGGER,
                                                    scale_ctl=scale_ctl,
                                                    tag="oom-train-scale-down",
                                                    msg=(
                                                        "[epochs] reduced sampler scale from %.4f to %.4f after OOM "
                                                        "(factor=%.2f, try=%d/%d)"
                                                    )
                                                    % (
                                                        prev,
                                                        cur,
                                                        _oom_scale_down_factor(oom_try),
                                                        oom_try,
                                                        max_tries,
                                                    ),
                                                    level="info",
                                                )
                                except Exception:
                                    pass

                                with contextlib.suppress(Exception):
                                    ec_min = 0.0 if oom_try <= 1 else _rt_env_float("STNET_OOM_EMPTY_CACHE_MIN_INTERVAL_S", 0.05)
                                    empty_device_cache(device=device, do_gc=False, min_interval_s=ec_min)

                                with contextlib.suppress(Exception):
                                    optimizer.zero_grad(set_to_none=True)

                                reduced_any = False

                                inst = _unwrap_for_microbatch(model)
                                if inst is not None:
                                    with contextlib.suppress(Exception):
                                        cur_mb = int(getattr(inst, "microbatch", 0) or 0)
                                    if cur_mb > 1:
                                        new_mb = max(1, cur_mb // 2)
                                        if new_mb < cur_mb:
                                            with contextlib.suppress(Exception):
                                                inst.microbatch = new_mb
                                                inst._auto_microbatch_pending = False
                                            _LOGGER.info(
                                                "[epochs] reduced Model.microbatch from %d to %d after OOM",
                                                cur_mb,
                                                new_mb,
                                            )
                                            reduced_any = True

                                if grad_accum_steps > min_grad_accum:
                                    new_grad_accum = max(min_grad_accum, grad_accum_steps // 2)
                                    try:
                                        new_grad_accum = _sync_int_across_ranks(
                                            new_grad_accum, device=device, src=0
                                        )
                                    except Exception:
                                        pass
                                    if new_grad_accum != grad_accum_steps:
                                        _LOGGER.info(
                                            "[epochs] reduced grad_accum_steps from %d to %d after OOM",
                                            grad_accum_steps,
                                            new_grad_accum,
                                        )
                                        grad_accum_steps = new_grad_accum
                                        reduced_any = True

                                if not reduced_any:
                                    if _oom_skip_enabled("train"):
                                        _LOGGER.error(
                                            "[epochs] OOM in train and no more knobs to reduce "
                                            "(microbatch <= 1, grad_accum at min). Skipping this batch."
                                        )
                                        with contextlib.suppress(Exception):
                                            _oom_retry_clear(train_loader, "train", step_idx)
                                        with contextlib.suppress(Exception):
                                            empty_device_cache(device=device, do_gc=False, min_interval_s=0.0)
                                        with contextlib.suppress(Exception):
                                            optimizer.zero_grad(set_to_none=True)
                                        break
                                    _LOGGER.error(
                                        "[epochs] OOM in train and no more knobs to reduce; giving up on recovery."
                                    )
                                    raise

                                continue
                            raise
                        finally:
                            pool_handles.clear()

            if lw_count > 0:
                top_avg_t: Optional[torch.Tensor] = (
                    (lw_top_sum / float(lw_count)) if lw_top_sum is not None else None
                )
                bottom_avg_t: Optional[torch.Tensor] = (
                    (lw_bottom_sum / float(lw_count)) if lw_bottom_sum is not None else None
                )
                if is_distributed():
                    ws = _get_ws_for_pg(train_pg)
                    if top_avg_t is not None:
                        with contextlib.suppress(Exception):
                            _all_reduce_sum_in_pg(top_avg_t, train_pg)
                        top_avg_t = top_avg_t / float(ws)
                    if bottom_avg_t is not None:
                        with contextlib.suppress(Exception):
                            _all_reduce_sum_in_pg(bottom_avg_t, train_pg)
                        bottom_avg_t = bottom_avg_t / float(ws)
                with contextlib.suppress(Exception):
                    loss_controller.update(top_avg_t, bottom_avg_t)
                lw_top_sum = None
                lw_bottom_sum = None
                lw_count = 0
            if val_loader is not None:
                flop_counter_val = FlopCounter(model, mode="eval", device=device)
                with flop_counter_val:
                    model.eval()
                    with Gradient.inference(model), Autocast.float(device):
                        t_fetch_start = time.perf_counter_ns()
                        for _vstep, _raw in enumerate(val_loader):
                            while True:
                                try:
                                    if device.type in ("cuda", "xpu", "mps"):
                                        t_ready = time.perf_counter_ns()
                                        if use_timer:
                                            h2d_s_ev, h2d_e_ev = (
                                                torch.Event(device=device, enable_timing=True),
                                                torch.Event(device=device, enable_timing=True),
                                            )
                                            h2d_s_ev.record()
                                            X, Y = meta.batch_to_device(_raw, device)
                                            h2d_e_ev.record()
                                            h2d_e_ev.synchronize()
                                            h2d_s = float(h2d_s_ev.elapsed_time(h2d_e_ev)) / 1000.0
                                        else:
                                            t_h2d_s = time.perf_counter_ns()
                                            X, Y = meta.batch_to_device(_raw, device)
                                            t_h2d_e = time.perf_counter_ns()
                                            h2d_s = (t_h2d_e - t_h2d_s) / 1_000_000_000.0
                                    else:
                                        feat, label, *_ = meta.preprocess(_raw)
                                        X = to_torch_tensor(feat)
                                        Y = to_torch_tensor(label)

                                        t_ready = time.perf_counter_ns()
                                        X, Y = _pin_tensor(X, Y)

                                        if use_timer:
                                            h2d_s_ev, h2d_e_ev = (
                                                torch.Event(device=device, enable_timing=True),
                                                torch.Event(device=device, enable_timing=True),
                                            )
                                            h2d_s_ev.record()
                                            X = _to_device_with_stream(X)
                                            Y = _to_device_with_stream(Y)
                                            h2d_e_ev.record()
                                            h2d_e_ev.synchronize()
                                            h2d_s = float(h2d_s_ev.elapsed_time(h2d_e_ev)) / 1000.0
                                        else:
                                            t_h2d_s = time.perf_counter_ns()
                                            X = _to_device_with_stream(X)
                                            Y = _to_device_with_stream(Y)
                                            t_h2d_e = time.perf_counter_ns()
                                            h2d_s = (t_h2d_e - t_h2d_s) / 1_000_000_000.0

                                    X = torch.atleast_2d(X)
                                    if X.dim() != 2:
                                        raise RuntimeError(
                                            f"features.ndim={X.dim()} (expect 2). got shape={tuple(X.shape)}"
                                        )
                                    if X.shape[1] != in_dim:
                                        raise RuntimeError(
                                            f"feature dim mismatch: X.shape[1]={X.shape[1]} != in_dim={in_dim}"
                                        )

                                    if Y.ndim < 1:
                                        raise RuntimeError(
                                            f"labels.ndim={Y.ndim} (expect >= 1). got shape={tuple(Y.shape)}"
                                        )

                                    wait_s = (t_ready - t_fetch_start) / 1_000_000_000.0
                                    io_time += float(wait_s + h2d_s)
                                    with contextlib.suppress(Exception):
                                        io_bytes += float(
                                            X.element_size() * X.nelement()
                                            + Y.element_size() * Y.nelement()
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
                                        with contextlib.suppress(Exception):
                                            mark_step = getattr(
                                                getattr(torch, "compiler", None),
                                                "cudagraph_mark_step_begin",
                                                None,
                                            )
                                            if callable(mark_step):
                                                mark_step()
                                        Yv_flat = Y.reshape(Y.shape[0], -1).to(
                                            device, dtype=param_dtype, non_blocking=True
                                        )
                                        tdv = to_tensordict(
                                            {"features": X, "labels_flat": Yv_flat}
                                        )
                                        model_out_val = model(
                                            tdv,
                                            global_loss=top_loss,
                                            local_loss=bottom_loss,
                                            loss_weights=loss_controller.weights(),
                                        )
                                        if isinstance(model_out_val, TensorDictBase):
                                            tdv = model_out_val
                                            _y = tdv.get("pred")
                                            _loss_val = tdv.get("loss_total", None)
                                            if (
                                                isinstance(_loss_val, torch.Tensor)
                                                and _loss_val.ndim > 0
                                            ):
                                                _loss_val = _loss_val.mean()
                                        else:
                                            _y, _loss_val = model_out_val
                                    if _loss_val is None:
                                        raise RuntimeError(
                                            "Model returned no loss value during validation. "
                                            "Ensure loss functions are configured correctly."
                                        )
                                    if not isinstance(_loss_val, torch.Tensor):
                                        _loss_val = torch.as_tensor(
                                            _loss_val, device=device, dtype=param_dtype
                                        )
                                    else:
                                        _loss_val = _loss_val.to(
                                            device=device, dtype=param_dtype
                                        )
                                    if use_timer:
                                        ev_e.record()
                                        ev_e.synchronize()
                                        comp_time += float(ev_s.elapsed_time(ev_e)) / 1000.0
                                    else:
                                        comp_time += (
                                            time.perf_counter_ns() - t_comp_s
                                        ) / 1_000_000_000.0
                                    with contextlib.suppress(Exception):
                                        flops += max(0.0, float(val_counter.get_total_flops()))
                                    breakdown_getter = getattr(
                                        val_counter, "get_manual_breakdown", None
                                    )
                                    if callable(breakdown_getter):
                                        for name, value in breakdown_getter().items():
                                            with contextlib.suppress(Exception):
                                                flop_breakdown_epoch[name] = (
                                                    flop_breakdown_epoch.get(name, 0.0)
                                                    + float(value)
                                                )

                                    if local_rank == 0:
                                        io_elapsed = prev_io_time + float(io_time)
                                        io_transferred = prev_io_bytes + float(io_bytes)
                                        comp_elapsed = prev_comp_time + float(comp_time)
                                        flop_total = prev_flops + float(flops)
                                        mbps_cur = (
                                            io_transferred
                                            / max(io_elapsed, 1e-06)
                                            / MB_DIV
                                        )
                                        tflops_cur = (
                                            flop_total
                                            / max(comp_elapsed, 1e-06)
                                            / 1_000_000_000_000.0
                                        )
                                        update_tqdm(
                                            status_bar,
                                            finish=1,
                                            mbps=mbps_cur,
                                            tflops=tflops_cur,
                                        )

                                    t_fetch_start = time.perf_counter_ns()
                                    if cpu_pool is not None and ((_vstep + 1) & 255) == 0:
                                        with contextlib.suppress(Exception):
                                            cpu_pool.collect()

                                    with contextlib.suppress(Exception):
                                        _oom_retry_clear(val_loader, "val", _vstep)

                                    break

                                except RuntimeError as e:
                                    msg = str(e).lower()
                                    if "out of memory" in msg:
                                        oom_try = _oom_retry_inc(val_loader, "val", _vstep)
                                        max_tries = _oom_max_retries("val")
                                        if oom_try <= 1:
                                            _LOGGER.error(
                                                "[epochs] OOM during validation step %d. "
                                                "Trying to reduce microbatch and retry same batch. (try=%d/%d)",
                                                _vstep,
                                                oom_try,
                                                max_tries,
                                            )
                                        else:
                                            _LOGGER.warning(
                                                "[epochs] OOM during validation step %d. Retrying. (try=%d/%d)",
                                                _vstep,
                                                oom_try,
                                                max_tries,
                                            )

                                        if max_tries > 0 and oom_try > max_tries:
                                            if _oom_skip_enabled("val"):
                                                _LOGGER.error(
                                                    "[epochs] OOM storm: exceeded retry budget (try=%d/%d) at validation step %d. "
                                                    "Skipping this batch.",
                                                    oom_try,
                                                    max_tries,
                                                    _vstep,
                                                )
                                                with contextlib.suppress(Exception):
                                                    _oom_retry_clear(val_loader, "val", _vstep)
                                                with contextlib.suppress(Exception):
                                                    empty_device_cache(device=device, do_gc=False, min_interval_s=0.0)
                                                break
                                            raise
                                        # Also request input batch scale-down for NEXT batches (true runtime recovery).
                                        try:
                                            scale_ctl = None
                                            obj = val_loader
                                            for _ in range(4):
                                                if obj is None:
                                                    break
                                                scale_ctl = getattr(obj, "_stnet_sampler_scale", None)
                                                if scale_ctl is not None:
                                                    break
                                                obj = getattr(obj, "_src", None) or getattr(obj, "src", None)
                                            if scale_ctl is not None:
                                                prev = None
                                                with contextlib.suppress(Exception):
                                                    prev = float(scale_ctl.get())
                                                with contextlib.suppress(Exception):
                                                    scale_ctl.request_scale_down(_oom_scale_down_factor(oom_try))
                                                with contextlib.suppress(Exception):
                                                    cur = float(scale_ctl.get())
                                                    if prev is not None and cur < prev:
                                                        _log_sampler_scale_rate_limited(
                                                            logger=_LOGGER,
                                                            scale_ctl=scale_ctl,
                                                            tag="oom-val-scale-down",
                                                            msg=(
                                                                "[epochs] reduced sampler scale from %.4f to %.4f after OOM (validation) "
                                                                "(factor=%.2f, try=%d/%d)"
                                                            )
                                                            % (
                                                                prev,
                                                                cur,
                                                                _oom_scale_down_factor(oom_try),
                                                                oom_try,
                                                                max_tries,
                                                            ),
                                                            level="info",
                                                        )
                                        except Exception:
                                            pass

                                        with contextlib.suppress(Exception):
                                            ec_min = 0.0 if oom_try <= 1 else _rt_env_float("STNET_OOM_EMPTY_CACHE_MIN_INTERVAL_S", 0.05)
                                            empty_device_cache(device=device, do_gc=False, min_interval_s=ec_min)

                                        reduced_any = False

                                        inst = _unwrap_for_microbatch(model)
                                        if inst is not None:
                                            with contextlib.suppress(Exception):
                                                cur_mb = int(getattr(inst, "microbatch", 0) or 0)
                                            if cur_mb > 1:
                                                new_mb = max(1, cur_mb // 2)
                                                if new_mb < cur_mb:
                                                    with contextlib.suppress(Exception):
                                                        inst.microbatch = new_mb
                                                        inst._auto_microbatch_pending = False
                                                    _LOGGER.info(
                                                        "[epochs] reduced Model.microbatch from %d to %d after OOM in validation",
                                                        cur_mb,
                                                        new_mb,
                                                    )
                                                    reduced_any = True

                                        if not reduced_any:
                                            if _oom_skip_enabled("val"):
                                                _LOGGER.error(
                                                    "[epochs] OOM in validation and no more knobs to reduce "
                                                    "(microbatch <= 1). Skipping this batch."
                                                )
                                                with contextlib.suppress(Exception):
                                                    _oom_retry_clear(val_loader, "val", _vstep)
                                                with contextlib.suppress(Exception):
                                                    empty_device_cache(device=device, do_gc=False, min_interval_s=0.0)
                                                break
                                            _LOGGER.error(
                                                "[epochs] OOM in validation and no more knobs to reduce "
                                                "(microbatch <= 1). Giving up on recovery."
                                            )
                                            raise

                                        continue
                                    raise
                                finally:
                                    pool_handles.clear()
            if is_distributed():
                stats = torch.tensor(
                    [comp_time, io_time, flops, io_bytes, train_samples_epoch],
                    device=device,
                    dtype=torch.float64,
                )
                torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
                world = max(1, get_world_size(device))
                stats /= world
                comp_time, io_time, flops, io_bytes, train_samples_epoch = [float(x) for x in stats.tolist()]
                distributed_barrier(device)
            updated_this_epoch = False
            if swa_enabled and epoch_idx >= swa_start_epoch:
                with contextlib.suppress(Exception):
                    swa_helper.update_weight()
                    updated_this_epoch = True
            if not scheduler_step_per_batch:
                with contextlib.suppress(Exception):
                    sched.step()
            if updated_this_epoch:
                swa_has_updated = True
            prev_comp_time += float(comp_time)
            prev_io_time += float(io_time)
            prev_flops += float(flops)
            prev_io_bytes += float(io_bytes)
            prev_samples += float(train_samples_epoch)
    model_for_scaler = model.module if hasattr(model, "module") else model
    scaler_y_device = model_for_scaler.scaler.y_mean.device
    with torch.no_grad():
        sum_x: Optional[torch.Tensor] = None
        sum_y: Optional[torch.Tensor] = None
        sum_x2: Optional[torch.Tensor] = None
        sum_xy: Optional[torch.Tensor] = None
        total_n: int = 0

        for batch in train_loader:
            x_raw = batch["features"].to(device)
            y_raw = batch["labels"].to(scaler_y_device)
            if y_raw.ndim >= 2:
                y_flat = y_raw.reshape(y_raw.shape[0], -1)
            else:
                y_flat = y_raw
            out = model(
                x_raw,
                labels_flat=None,
                net_loss=None,
                global_loss=None,
                local_loss=None,
                calibrate_output=False,
            )
            if isinstance(out, tuple):
                z_pred_raw, _ = out
            else:
                z_pred_raw = out

            z_pred = z_pred_raw.detach().to(
                device=scaler_y_device, dtype=torch.float64
            )
            if z_pred.ndim >= 2:
                z_pred = z_pred.reshape(z_pred.shape[0], -1)
            else:
                z_pred = z_pred.view(-1, 1)

            z_true = model_for_scaler.scaler.normalize_y(
                y_flat.detach()
            ).to(dtype=torch.float64)
            if z_true.ndim >= 2:
                z_true = z_true.reshape(z_true.shape[0], -1)
            else:
                z_true = z_true.view(-1, 1)

            if z_pred.shape[-1] != z_true.shape[-1]:
                f_pred = z_pred.shape[-1]
                f_true = z_true.shape[-1]
                if f_true % f_pred == 0:
                    group = f_true // f_pred
                    z_true = z_true.view(z_true.shape[0], group, f_pred).mean(dim=1)
                elif f_pred % f_true == 0:
                    group = f_pred // f_true
                    z_true = z_true.repeat_interleave(group, dim=1)
                else:
                    raise RuntimeError(
                        "Calibration: feature dimension mismatch between prediction and target "
                        f"that cannot be reconciled generically. "
                        f"z_pred.shape={tuple(z_pred.shape)}, z_true.shape={tuple(z_true.shape)}"
                    )

            if z_pred.shape[0] != z_true.shape[0]:
                raise RuntimeError(
                    "Calibration: batch dimension mismatch between prediction and target. "
                    f"z_pred.shape={tuple(z_pred.shape)}, z_true.shape={tuple(z_true.shape)}"
                )

            if z_pred.numel() == 0 or z_true.numel() == 0:
                continue

            n_batch = z_pred.shape[0]
            total_n += n_batch

            sx = z_pred.sum(dim=0)
            sy = z_true.sum(dim=0)
            sx2 = (z_pred * z_pred).sum(dim=0)
            sxy = (z_pred * z_true).sum(dim=0)

            if sum_x is None:
                sum_x = sx
                sum_y = sy
                sum_x2 = sx2
                sum_xy = sxy
            else:
                sum_x += sx
                sum_y += sy
                sum_x2 += sx2
                sum_xy += sxy
        if is_distributed():
            n_t = torch.tensor(
                float(total_n), device=scaler_y_device, dtype=torch.float64
            )
            torch.distributed.all_reduce(n_t, op=torch.distributed.ReduceOp.SUM)
            total_n = int(n_t.item())

            if sum_x is not None:
                torch.distributed.all_reduce(sum_x, op=torch.distributed.ReduceOp.SUM)
            if sum_y is not None:
                torch.distributed.all_reduce(sum_y, op=torch.distributed.ReduceOp.SUM)
            if sum_x2 is not None:
                torch.distributed.all_reduce(sum_x2, op=torch.distributed.ReduceOp.SUM)
            if sum_xy is not None:
                torch.distributed.all_reduce(sum_xy, op=torch.distributed.ReduceOp.SUM)

        if total_n > 0 and sum_x is not None and sum_y is not None and sum_x2 is not None and sum_xy is not None:
            N = float(total_n)
            mean_x = sum_x / N
            mean_y = sum_y / N
            Ex2 = sum_x2 / N
            Exy = sum_xy / N
            var_x = Ex2 - mean_x * mean_x
            cov_xy = Exy - mean_x * mean_y

            eps = float(model_for_scaler.scaler.eps)
            denom = var_x.clone()
            tiny_mask = denom.abs() < eps
            denom[tiny_mask] = 1.0

            a = (cov_xy / denom).to(dtype=torch.float32)
            b = (mean_y - a.to(dtype=torch.float64) * mean_x).to(dtype=torch.float32)
            a[tiny_mask] = 1.0
            b[tiny_mask] = 0.0

            model_for_scaler.scaler.set_affine(a, b)

    if local_rank == 0 and status_bar is not None:
        mbps = prev_io_bytes / max(prev_io_time, 1e-06) / MB_DIV
        tflops = prev_flops / max(prev_comp_time, 1e-06) / 1_000_000_000_000.0
        status_bar.set_postfix_str(
            f"{mbps:.2f} MB/s, {tflops:.2f} TFLOPS", refresh=True
        )
        status_bar.close()
    end_kst_ns = posix_time("Asia/Seoul")
    try:
        dev_t = getattr(device, "type", "")
        total_t = prev_io_time + prev_comp_time
        samples_per_sec = 0.0
        util_from_sps = 0.0
        if total_t > 0.0 and prev_samples > 0.0 and prev_comp_time > 0.0:
            samples_per_sec = prev_samples / total_t
            max_samples_per_sec = prev_samples / prev_comp_time
            if max_samples_per_sec > 0.0:
                util_from_sps = samples_per_sec / max_samples_per_sec
        util_fallback = util_from_sps if util_from_sps > 0.0 else (
            (prev_comp_time / total_t) if total_t > 0.0 else 0.0
        )

        gpu_util_frac = None
        mem_util_frac = None
        if gpu_util_ema is not None:
            gpu_util_frac = max(0.0, min(1.0, gpu_util_ema / 100.0))
        if mem_util_ema is not None:
            mem_util_frac = max(0.0, min(1.0, mem_util_ema / 100.0))

        if dev_t != "cpu":
            util_for_cap = gpu_util_frac if gpu_util_frac is not None else util_fallback
            util_for_cap = max(0.0, min(1.0, util_for_cap))

            # Adapt *this loader's* sampler scale (per-session/per-loader; best-effort).
            try:
                if train_loader is not None:
                    scale_ctl = None
                    obj = train_loader
                    for _ in range(4):
                        if obj is None:
                            break
                        scale_ctl = getattr(obj, "_stnet_sampler_scale", None)
                        if scale_ctl is not None:
                            break
                        obj = getattr(obj, "_src", None) or getattr(obj, "src", None)

                    if scale_ctl is not None:
                        if mem_util_frac is not None and mem_util_frac > 0.92:
                            prev = None
                            with contextlib.suppress(Exception):
                                prev = float(scale_ctl.get())
                            with contextlib.suppress(Exception):
                                scale_ctl.request_scale_down(0.95)
                            with contextlib.suppress(Exception):
                                cur = float(scale_ctl.get())
                                if prev is not None and cur < prev:
                                    _log_sampler_scale_rate_limited(
                                        logger=_LOGGER,
                                        scale_ctl=scale_ctl,
                                        tag="auto-scale-down",
                                        msg=("[epochs] auto scale_down (mem_util=%.3f): %.4f -> %.4f")
                                        % (float(mem_util_frac), float(prev), float(cur)),
                                        level="debug",
                                    )
                        elif util_for_cap < 0.90 and (mem_util_frac is None or mem_util_frac < 0.88):
                            prev = None
                            with contextlib.suppress(Exception):
                                prev = float(scale_ctl.get())
                            with contextlib.suppress(Exception):
                                scale_ctl.request_scale_up(1.10)
                            with contextlib.suppress(Exception):
                                cur = float(scale_ctl.get())
                                if prev is not None and cur > prev:
                                    _log_sampler_scale_rate_limited(
                                        logger=_LOGGER,
                                        scale_ctl=scale_ctl,
                                        tag="auto-scale-up",
                                        msg=("[epochs] auto scale_up (util=%.3f): %.4f -> %.4f")
                                        % (float(util_for_cap), float(prev), float(cur)),
                                        level="debug",
                                    )
            except Exception:
                pass
        else:
            cpu_pct = _cpu_percent_now()
            if cpu_pct is not None:
                if cpu_pct > 80.0:
                    time.sleep(min(0.005, 0.001 * (cpu_pct - 80.0)))
            else:
                if util_fallback > 0.80:
                    time.sleep(min(0.005, total_t * (util_fallback - 0.80)))
        if isinstance(hist, History):
            try:
                end_sec = round(float(end_kst_ns) / 1e9, 6)
                world = max(1, get_world_size(device)) if is_distributed() else 1
                hist.end_session(end_sec, peers=world)

                if ops.ckpt_dir and int(local_rank) == 0:
                    history_path = os.path.join(ops.ckpt_dir, "history.json")
                    records = hist.save()

                    meta = {
                        "start_posix": float(round(float(hist.start.item()), 6)),
                        "end_posix": float(round(float(hist.end.item()), 6)),

                        "timezone": hist.timezone,
                        "peers": int(hist.peers.item()),
                        "epochs": int(hist.epochs.item()),
                        "os": hist.os,
                        "kernel": hist.kernel,
                        "cpu": list(hist.cpu),
                        "arch": list(hist.arch),
                        "ram_gb": float(round(float(hist.ram_gb), 2)),
                        "python": hist.python,
                        "backends": list(hist.backends),
                    }

                    payload = {
                        "meta": meta,
                        "records": records,
                    }

                    with open(history_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f)
            except Exception:
                pass
    except Exception:
        pass


def infer(
    model: Root,
    device: torch.device,
    local_rank: int,
    ops: RuntimeConfig,
    *,
    data_loader: Optional[Iterable[TensorDictBase]] = None,
    chunk_dir: Optional[str] = None,
    dataset: Optional[Dataset] = None,
) -> Optional[Dict[Tuple, torch.Tensor]]:

    import gc
    import glob

    if data_loader is None:
        return None

    if dataset is None:
        dataset = Dataset.for_device(str(device) if isinstance(device, torch.device) else "cpu")

    if chunk_dir is None:
        if not ops.ckpt_dir:
            raise RuntimeError("infer: ckpt_dir is required when chunk_dir is not provided")
        chunk_dir = os.path.join(ops.ckpt_dir, "pred_chunks")

    rank = torch.distributed.get_rank() if is_distributed() else 0
    world_size = get_world_size(device) if is_distributed() else 1

    if rank == 0:
        os.makedirs(chunk_dir, exist_ok=True)
    distributed_barrier(device)
    cache = Cache(chunk_dir, max_queue=4)

    try:
        target_rows = int(os.environ.get("STNET_PRED_CHUNK_ROWS", "0") or 0)
    except Exception:
        target_rows = 0

    if target_rows <= 0:
        out_shape = tuple(int(x) for x in (ops.out_shape or ()))
        out_numel = 1
        for d in out_shape:
            out_numel *= max(1, int(d))
        est_row_bytes = max(1, out_numel * 4)
        target_bytes = int(os.environ.get("STNET_PRED_CHUNK_BYTES", str(64 * 1024 * 1024)))
        target_rows = max(256, min(65536, target_bytes // est_row_bytes))

    run_model = to_ddp(model, device=device)
    run_model.eval()
    module_eval = run_model.module if hasattr(run_model, "module") else run_model
    distributed_sync(module_eval, device=device)

    status_bar = (
        get_tqdm(
            title="Prediction",
            total=_num_batches(data_loader),
            device=device,
            leave=False,
        )
        if local_rank == 0
        else None
    )
    pending_rows: list[torch.Tensor] = []
    pending_preds: list[torch.Tensor] = []
    pending_count = 0
    chunk_idx = 0
    row_cursor = 0
    first_tail: Optional[Tuple[int, ...]] = None
    variable_shape = False

    def _flush() -> None:
        nonlocal chunk_idx, pending_count
        if pending_count <= 0:
            return

        rows = torch.cat(pending_rows, dim=0).to(dtype=torch.int64, copy=False).contiguous()
        preds = torch.cat(pending_preds, dim=0).contiguous()
        rows_path = os.path.join(chunk_dir, f"part-r{rank:05d}-c{chunk_idx:06d}-rows.pt")
        pred_path = os.path.join(chunk_dir, f"part-r{rank:05d}-c{chunk_idx:06d}-pred.pt")
        cache.submit(rows, path=rows_path)
        cache.submit(preds, path=pred_path)
        chunk_idx += 1
        pending_rows.clear()
        pending_preds.clear()
        pending_count = 0


        del rows, preds
        gc.collect()

    try:
        with Gradient.inference(run_model), Autocast.float(device):
            for batch in data_loader:
                if batch is None:
                    if status_bar is not None:
                        status_bar.update(1)
                    continue
                row_ids: Optional[torch.Tensor] = None
                try:
                    if isinstance(batch, TensorDictBase):
                        row_ids = batch.get("row_ids", None)
                    elif isinstance(batch, dict):
                        row_ids = batch.get("row_ids", None)
                except Exception:
                    row_ids = None

                X, _Y = dataset.batch_to_device(batch, device=device, non_blocking=True)
                bs = int(getattr(X, "shape", [0])[0]) if hasattr(X, "shape") else 0
                if bs <= 0:
                    if status_bar is not None:
                        status_bar.update(1)
                    continue

                if row_ids is None:
                    row_ids = torch.arange(
                        row_cursor, row_cursor + bs, dtype=torch.int64
                    )
                elif not isinstance(row_ids, torch.Tensor):
                    row_ids = torch.as_tensor(row_ids, dtype=torch.int64)
                else:
                    row_ids = row_ids.to(dtype=torch.int64, copy=False)
                row_ids = row_ids.reshape(-1)
                if row_ids.numel() != bs:
                    raise RuntimeError(f"infer: row_ids length mismatch: row_ids={row_ids.numel()} vs batch={bs}")
                row_cursor += bs
                if row_ids.device.type != "cpu":
                    row_ids = row_ids.to(device="cpu")

                mb = int(getattr(model, "microbatch", 0) or 0)
                if mb <= 0:
                    mb = bs
                mb = max(1, min(bs, mb))

                start = 0
                while start < bs:
                    end = min(bs, start + mb)
                    sl = slice(start, end)

                    Xi = X[sl]
                    rows_i = row_ids[sl]
                    tdp = to_tensordict({"features": Xi}, device=device)

                    try:
                        out = run_model(tdp, calibrate_output=True)
                    except RuntimeError as e:
                        msg = str(e).lower()
                        if "out of memory" in msg and mb > 1:
                            with contextlib.suppress(Exception):
                                empty_device_cache(device=device, do_gc=False, min_interval_s=0.0)
                            mb = max(1, mb // 2)
                            try:
                                setattr(model, "microbatch", mb)
                            except Exception:
                                pass
                            with contextlib.suppress(Exception):
                                del Xi, tdp
                            gc.collect()
                            continue  
                        raise

                    y_hat: Optional[torch.Tensor] = None
                    if isinstance(out, TensorDictBase):
                        y_hat = out.get("pred", None)
                    if y_hat is None and isinstance(tdp, TensorDictBase):
                        y_hat = tdp.get("pred", None)
                    if y_hat is None:
                        raise RuntimeError("infer: model output missing 'pred'")

                    y_hat = y_hat.detach()
                    tail = tuple(int(x) for x in y_hat.shape[1:])
                    if first_tail is None:
                        first_tail = tail
                    elif tail != first_tail:
                        variable_shape = True

                    y_cpu = y_hat.to(device="cpu")
                    rows_cpu = (rows_i if rows_i.device.type == "cpu" else rows_i.to(device="cpu")).to(dtype=torch.int64)

                    pending_rows.append(rows_cpu.reshape(-1))
                    pending_preds.append(y_cpu)
                    pending_count += int(y_cpu.shape[0])

                    if pending_count >= target_rows:
                        _flush()

                    del Xi, rows_i, tdp, out, y_hat, y_cpu, rows_cpu
                    start = end

                if status_bar is not None:
                    status_bar.update(1)

                del X, _Y, batch, row_ids

    finally:
        _flush()
        cache.close()
        exc_type, _, _ = sys.exc_info()
        if exc_type is None:
            with contextlib.suppress(Exception):
                if getattr(cache, "had_error", None) and cache.had_error():
                    raise RuntimeError("infer: prediction writer encountered an error")
        if status_bar is not None:
            status_bar.close()

        distributed_barrier(device)

        if rank == 0:
            parts: list[dict[str, str]] = []
            for rows_path in sorted(glob.glob(os.path.join(chunk_dir, "part-r*-c*-rows.pt"))):
                base = rows_path[: -len("-rows.pt")]
                pred_path = base + "-pred.pt"
                if not os.path.exists(pred_path):
                    # Fail fast: missing paired pred means incomplete output.
                    raise RuntimeError(f"infer: missing pred file for rows part: {rows_path} -> {pred_path}")
                parts.append({"rows": os.path.basename(rows_path), "pred": os.path.basename(pred_path)})

            if not parts:
                raise RuntimeError(f"infer: no prediction parts produced in {chunk_dir}")

            manifest = {
                "format": "stnet.pred.v2",
                "rank_count": int(world_size),
                "out_shape": list(int(x) for x in (ops.out_shape or ())),
                "variable_shape": bool(variable_shape),
                "parts": parts,
            }

            man_path = os.path.join(chunk_dir, "manifest.json")
            with open(man_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

        distributed_barrier(device)

    return None


def main(*args: Any, **kwargs: Any) -> Optional[Root]:
    from ..data.pipeline import Session

    if not args:
        raise TypeError("main requires at least a RuntimeConfig argument")
    initialize_python_path()
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
            "main expects (RuntimeConfig,), (RuntimeConfig, ret_sink), (local_rank, RuntimeConfig), or (local_rank, RuntimeConfig, ret_sink) arguments"
        )

    verbose = bool(getattr(ops, "verbose", False))
    if ops.mode == "train":
        with contextlib.suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank % max(1, torch.cuda.device_count()))
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.set_device(local_rank % max(1, torch.xpu.device_count()))
        device = get_device()
        _set_backend(device)
        backend = _backend_type(device)
        enable_tf32 = bool(getattr(ops, "enable_tf32", True))
        _initialize_group(backend, device, local_rank)
        cfg = coerce_model_config(
            ops.cfg_dict if isinstance(ops.cfg_dict, dict) else ops.cfg_dict
        )
        cfg = replace(cfg, device=device)
        model = Root(ops.in_dim, ops.out_shape, config=cfg)
        if ops.init_ckpt_dir is not None and os.path.isdir(ops.init_ckpt_dir):
            fallback_init = os.path.join(ops.init_ckpt_dir, "model.pt")
            if os.path.isfile(fallback_init):
                cpu_state = torch.load(fallback_init, map_location="cpu")
                resize_scaler_buffer(model, cpu_state)
                model.load_state_dict(cpu_state, strict=False)
            else:
                m_sd = get_model_state_dict(
                    model,
                    options=StateDictOptions(
                        full_state_dict=True, cpu_offload=False
                    ),
                )
                m_sd = _trim_dcp_keys(m_sd)
                load(
                    state_dict={"model": m_sd},
                    storage_reader=FileSystemReader(ops.init_ckpt_dir),
                )
                resize_scaler_buffer(model, m_sd)
                set_model_state_dict(
                    model, m_sd, options=StateDictOptions(strict=False)
                )
        if ops.sources is None:
            raise RuntimeError("RuntimeConfig.sources is required but None")
        metadata = Dataset.for_device(device)
        expanded_sources = _expand(ops.sources)
        if expanded_sources is not ops.sources:
            ops = replace(ops, sources=expanded_sources)
        meta_info = _merge_meta_infos(ops.sources)
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
                actual_val_frac, float(ops.val_frac), rel_tol=0.001, abs_tol=0.001
            ):
                warnings.warn(
                    "val_frac=%s differs from memmap metadata (%s); using metadata value for loaders"
                    % (ops.val_frac, actual_val_frac)
                )
                ops = replace(ops, val_frac=actual_val_frac)
        # Apply scale metadata from memmap meta.json to the in-memory Dataset descriptor.
        # (Older meta.json files may not have these fields.)
        metadata.has_scale = bool(
            meta_info.get("has_scale", False)
            or meta_info.get("scale_max_abs") is not None
            or meta_info.get("scale_min_value") is not None
            or meta_info.get("scale_max_value") is not None
            or meta_info.get("scale_min_positive") is not None
            or meta_info.get("scale_min_abs") is not None
        )
        metadata.has_nonfinite = bool(meta_info.get("has_nonfinite", False))
        metadata.scale_max_abs = meta_info.get("scale_max_abs")
        metadata.scale_min_value = meta_info.get("scale_min_value")
        metadata.scale_max_value = meta_info.get("scale_max_value")
        metadata.scale_min_positive = meta_info.get("scale_min_positive") or meta_info.get("scale_min_abs")
        metadata.scale_is_integral = meta_info.get("scale_is_integral")
        if meta_info.get("is_negotiable") is not None:
            metadata.is_negotiable = bool(meta_info.get("is_negotiable"))
        if meta_info.get("underflow_action") is not None:
            metadata.underflow_action = str(meta_info.get("underflow_action"))

        # If the dataset was *stored* in float64, keep master dtype at float64 to avoid
        # accidental promotion/TE incompatibilities, even if the values would be fp32-castable.
        feat_dtype_name = str(meta_info.get("features_dtype", "")).lower()
        lab_dtype_name = str(meta_info.get("labels_dtype", "")).lower()
        if "float64" in feat_dtype_name or "float64" in lab_dtype_name:
            metadata.is_negotiable = False

        # Resolve a coherent precision policy (master dtype + optional AMP compute dtype) from metadata.
        precision = fused.PrecisionPolicy.from_metadata(device=device, metadata=metadata, logger=_LOGGER)
        param_dtype = precision.master_float

        if device.type != "cuda":
            _LOGGER.warning(
                "Forcing CPU / non-CUDA config: mixed precision + NVIDIA fused layers may be unavailable."
            )

        model, _, _ = fused.ModelPolicy.use_nvidia_layers(
            model,
            device=device,
            metadata=metadata,
            params_dtype=param_dtype,
            verbose=verbose,
        )
        Autocast.configure(model, metadata=metadata)

        # TF32 (when available) is orthogonal: it affects matmul/conv execution, not storage dtypes.
        set_float32_precision(
            device=device,
            autocast_dtype=precision.amp_float or param_dtype,
            enable_tf32=enable_tf32,
        )

        # Optional FP8 (only makes sense when master params are <= FP32 and a CUDA backend exists).
        fp8_ok, fp8_reason = Dataset.is_float8_supported(device)
        fp8_enabled: bool = False
        fp8_backend: Optional[str] = None
        disable_note: Optional[str] = None
        if param_dtype is torch.float64:
            disable_note = "master dtype is float64"
        elif fp8_ok:
            model, fp8_enabled, fp8_backend = fused.ModelPolicy.enable_float8_training(
                model,
                metadata=metadata,
                logger=_float8_log,
            )
            if not fp8_enabled:
                disable_note = fp8_backend or fp8_reason
        else:
            disable_note = fp8_reason
        if not fp8_enabled:
            Autocast.configure(model, metadata=metadata)
            if disable_note:
                _float8_log(f"[FP8] disabled: {disable_note}")

        _cast_model_fp_dtype(model, param_dtype)
        model.train()
        world = get_world_size(device)
        mesh = None
        # FSDP: keep parameters in master dtype; use reduce/output in AMP dtype when enabled.
        fsdp_mp_dtype = precision.fsdp_reduce_dtype
        if device.type == "cpu" and fsdp_mp_dtype is not torch.float64:
            fsdp_mp_dtype = torch.float32
        amp_buffers_dtype = precision.bn_buffers_dtype
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=fsdp_mp_dtype,
            output_dtype=fsdp_mp_dtype,
            cast_forward_inputs=False,
        )
        _m_pre = model.module if hasattr(model, "module") else model
        _preload_layers(_m_pre, device)
        _assert_unified_layer_dtype(_m_pre, device)
        _assert_no_meta_tensors(_m_pre)
        _assert_no_fake_dtensor(_m_pre)
        wrapped = set()
        try:
            for submodule in _get_layers(getattr(model, "processor", None)):
                _wrap_fsdp(
                    submodule,
                    mesh,
                    mp_policy,
                    reshard_after_forward=True,
                    wrapped=wrapped,
                )
            for submodule in _get_layers(getattr(model, "controller", None)):
                _wrap_fsdp(
                    submodule,
                    mesh,
                    mp_policy,
                    reshard_after_forward=True,
                    wrapped=wrapped,
                )
            model = (
                _wrap_fsdp(
                    model,
                    mesh,
                    mp_policy,
                    reshard_after_forward=True,
                    wrapped=wrapped,
                )
                or model
            )
        except (RuntimeError, ValueError, TypeError):
            model = to_fsdp(
                model,
                mesh=mesh,
                mp_policy=mp_policy,
                reshard_after_forward=False,
                sync_module_states=True,
            )
        _m_post = model.module if hasattr(model, "module") else model
        _assert_unified_layer_dtype(_m_post, device)
        _assert_no_meta_tensors(_m_post)
        _assert_no_fake_dtensor(_m_post)
        _enable_meta_monitor(_m_post)
        distributed_sync(_m_post, device=device)
        net_params = [p for p in model.parameters()]
        optimizer = AdamW.float(
            net_params,
            lr=ops.base_lr,
            weight_decay=ops.weight_decay,
            metadata=metadata,
            logger=None,
        )
        if ops.init_ckpt_dir is not None and os.path.isdir(ops.init_ckpt_dir):
            _initialize_adamw(optimizer)
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
                _initialize_adamw(optimizer)
        top_df = DataFidelityLoss(
            out_shape=ops.out_shape,
            reduction="mean",
        )

        top_z = StandardNormalLoss(
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
            reduction="mean",
            skew=ops.loss_skew,
        )
        local_crps = CRPSLoss(
            dim=-1,
            reduction="none",
            detach_stats=True,
        )
        local_t = StudentsTLoss(
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
            skew=ops.loss_skew,
        )
        top_loss = LinearCombinationLoss(
            coefficient=[1.00, 0.00],
            loss=[top_df, top_z],
            reduce_each=True,
            auto_schedule=True,
        )
        bottom_loss = TiledLoss(
            nn.Sequential(),
            mask_mode=ops.loss_mask_mode,
            mask_value=ops.loss_mask_value,
            tile_dim=ops.loss_tile_dim,
            tile_size=ops.loss_tile_size,
            reduction="mean",
        )
        bottom_loss.base = LinearCombinationLoss(
            coefficient=[1.00, 0.00],
            loss=[local_crps, local_t],
            reduce_each=False,
            auto_schedule=True,
        )
        loss_controller = LossWeightController(top_avg=0.5, bottom_avg=0.5)
        ckpt_state_path = loader_state_path(ops.ckpt_dir or "")
        init_state_path = (
            loader_state_path(ops.init_ckpt_dir) if ops.init_ckpt_dir else None
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
        raw_train_loader: Any = None
        raw_val_loader: Any = None
        session: Optional[Session] = None
        try:
            expanded_sources = _expand(ops.sources)
            if expanded_sources is not ops.sources:
                ops = replace(ops, sources=expanded_sources)
            accelerator_types = {"cuda", "xpu", "mps"}
            device_type = getattr(device, "type", None)
            if not device_type:
                device_str = str(device)
                device_type = device_str.split(":", 1)[0]
            non_blocking_copy = device_type in accelerator_types
            with contextlib.suppress(Exception):
                _calibrate_per_sample_mem(
                    model=model,
                    device=device,
                    ops=ops,
                    dataset=metadata,
                    with_backward=True,
                    global_loss=top_loss,
                    local_loss=bottom_loss,
                    loss_weights=loss_controller.weights(),
                )

            os.environ.setdefault("STNET_MICROBATCH_MAX", "64")
            os.environ.setdefault("STNET_MICROBATCH_STAGE_DIV", "4")
            session = Session(
                sources=ops.sources,
                device=device,
                val_frac=float(ops.val_frac),
                non_blocking_copy=non_blocking_copy,
                sanitize=True,
                flatten_features=True,
                labels_dtype=param_dtype,
            ).open(
                train_state=(state_train if restore_dl_state else None),
                val_state=(state_val if restore_dl_state else None),
            )
            train_loader = session.training_loader
            val_loader = session.validation_loader
            raw_train_loader = session.raw_training_loader
            raw_val_loader = session.raw_validation_loader
            train_steps = _num_batches(train_loader)
            val_steps = _num_batches(val_loader)
            steps_per_epoch = max(1, train_steps + val_steps)
            total_epochs = int(ops.epochs)
            total_steps = max(1, total_epochs * steps_per_epoch)
            if ops.warmup_ratio > 0.0:
                warmup_steps = max(1, int(total_steps * ops.warmup_ratio))
                main_steps = max(1, total_steps - warmup_steps)
            else:
                warmup_steps = 0
                main_steps = max(1, total_steps)
            base = float(ops.base_lr)
            emin = float(ops.eta_min)
            start_factor = 0.001
            lr_lambda = partial(
                _scheduler,
                warmup_steps=warmup_steps,
                start_factor=start_factor,
                base=base,
                main_steps=main_steps,
                emin=emin,
            )
            sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            scheduler_step_per_batch = True
            swa_helper: Optional[StochasticWeightAverage] = None
            swa_start_epoch = total_epochs
            enable_swa_cfg = bool(getattr(ops, "swa_enabled", False))
            start_epoch_cfg = getattr(ops, "swa_start_epoch", None)
            enable_swa = (
                enable_swa_cfg or start_epoch_cfg is not None
            ) and SWALR is not None
            if enable_swa:
                tracked_module = model.module if hasattr(model, "module") else model
                use_buffers = True
                try:
                    swa_helper = stochastic_weight_average(
                        tracked_module, use_buffers=use_buffers
                    )
                except Exception:
                    swa_helper = None
                if swa_helper is not None:
                    scheduler_step_per_batch = False
                    if start_epoch_cfg is not None:
                        try:
                            swa_start_epoch = max(0, int(start_epoch_cfg))
                        except (TypeError, ValueError):
                            swa_start_epoch = max(1, total_epochs // 2)
                    else:
                        swa_start_epoch = max(1, total_epochs // 2)
                    eta_min = float(getattr(ops, "eta_min", 0.0) or 0.0)
                    base_lr = float(ops.base_lr)
                    default_swa_lr = max(
                        1e-8, eta_min if eta_min > 0.0 else 0.1 * base_lr
                    )
                    swa_lr = default_swa_lr
                    anneal_epochs = max(1, max(1, total_epochs // 10))
                    try:
                        sched = SWALR(
                            optimizer,
                            swa_lr=swa_lr,
                            anneal_epochs=anneal_epochs,
                            anneal_strategy="cos",
                        )
                    except Exception:
                        scheduler_step_per_batch = True
                        swa_helper = None
                        swa_start_epoch = total_epochs
            # AMP GradScaler should be enabled when *compute* uses FP16.
            #
            # NOTE: Do not infer this from *device capabilities* (e.g. BF16 support).
            # We must check the dtype we actually run the compute path in.
            #
            # - If autocast uses FP16 -> enable GradScaler
            # - If autocast is disabled but parameters/compute are FP16 -> enable GradScaler
            amp_dtype = getattr(precision, "amp_float", None)
            compute_dtype = amp_dtype or param_dtype
            scaler = torch.amp.GradScaler(
                enabled=bool(device.type == "cuda" and compute_dtype == torch.float16)
            )

                                                                                        
            try:
                get_tlb().pin_thread()
                with contextlib.suppress(Exception):
                    Memory.prefer_local_numa()
            except Exception:
                pass
            epochs(
                model=model,
                device=device,
                local_rank=local_rank,
                ops=ops,
                param_dtype=param_dtype,
                optimizer=optimizer,
                scaler=scaler,
                sched=sched,
                loss_controller=loss_controller,
                top_loss=top_loss,
                bottom_loss=bottom_loss,
                train_loader=train_loader,
                val_loader=val_loader,
                total_epochs=total_epochs,
                scheduler_step_per_batch=scheduler_step_per_batch,
                swa_helper=swa_helper,
                swa_start_epoch=swa_start_epoch,
                buffers_dtype=amp_buffers_dtype,
                dataset=metadata,
            )
        finally:
            if session is not None:
                session.close()
        if local_rank == 0:
            model_sd = get_model_state_dict(
                model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            optim_sd = get_optimizer_state_dict(model, optimizers=optimizer)

            writer = FileSystemWriter(
                ops.ckpt_dir or "", sync_files=True, overwrite=True
            )
            save(
                state_dict={"model": model_sd, "optimizer": optim_sd},
                storage_writer=writer,
            )
            if ops.ckpt_dir:
                fallback_path = os.path.join(ops.ckpt_dir, "model.pt")
                model_fallback = dict(model_sd)
                _trim_dcp_keys(model_fallback)
                torch.save(model_fallback, fallback_path)
                with contextlib.suppress(Exception):
                    _dl = {
                        "train": (
                            raw_train_loader.state_dict()
                            if raw_train_loader is not None
                            else {}
                        ),
                        "val": (
                            raw_val_loader.state_dict() if raw_val_loader is not None else {}
                        ),
                    }
                    with open(
                        loader_state_path(ops.ckpt_dir or ""), "w", encoding="utf-8"
                    ) as _f:
                        json.dump(_dl, _f)
        torch.distributed.barrier(
            device_ids=[local_rank] if device.type in ("cuda", "xpu") else None
        )
        torch.distributed.destroy_process_group()
        return None
    if ops.mode in ("predict", "infer"):
        with contextlib.suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank % max(1, torch.cuda.device_count()))
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.set_device(local_rank % max(1, torch.xpu.device_count()))
        device = get_device()
        _set_backend(device)
        backend = _backend_type(device)
        if not torch.distributed.is_initialized():
            _initialize_group(backend, device, local_rank)
        cfg = coerce_model_config(
            ops.cfg_dict if isinstance(ops.cfg_dict, dict) else ops.cfg_dict
        )
        model = Root(ops.in_dim, ops.out_shape, config=cfg)
        if ops.model_ckpt_dir is not None and os.path.isdir(ops.model_ckpt_dir):
            fallback_model = os.path.join(ops.model_ckpt_dir, "model.pt")
            if os.path.isfile(fallback_model):
                cpu_state = torch.load(fallback_model, map_location="cpu")
                resize_scaler_buffer(model, cpu_state)
                model.load_state_dict(cpu_state, strict=False)
            else:
                m_sd = get_model_state_dict(
                    model,
                    options=StateDictOptions(
                        full_state_dict=True, cpu_offload=True
                    ),
                )
                m_sd = _trim_dcp_keys(m_sd)
                load(
                    state_dict={"model": m_sd},
                    storage_reader=FileSystemReader(ops.model_ckpt_dir),
                )
                resize_scaler_buffer(model, m_sd)
                set_model_state_dict(
                    model, m_sd, options=StateDictOptions(strict=False)
                )
        model.to(device, non_blocking=True).eval()
        metadata = Dataset.for_device(device)
        model, _, _ = fused.ModelPolicy.use_nvidia_layers(model, device=device)
        _m_eval = model.module if hasattr(model, "module") else model
        _preload_layers(_m_eval, device)
        _assert_unified_layer_dtype(_m_eval, device)
        _assert_no_meta_tensors(_m_eval)
        _assert_no_fake_dtensor(_m_eval)
        _enable_meta_monitor(_m_eval)
        _unify_param_dtype(
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
        Autocast.configure(model, metadata=metadata)
        fp8_infer_ok, fp8_infer_reason = Dataset.is_float8_supported(device)
        if fp8_infer_ok:
            model, _, _ = fused.ModelPolicy.enable_float8_prediction(
                model, metadata=metadata, logger=_float8_log
            )
        else:
            _float8_log(f"[FP8] disabled: {fp8_infer_reason}")
        if ops.sources is None:
            raise RuntimeError("RuntimeConfig.sources is required but None")
        model.eval()
        with contextlib.suppress(Exception):
            _calibrate_per_sample_mem(
                model=model,
                device=device,
                ops=ops,
                dataset=metadata,
                with_backward=False,
            )

        expanded_sources = _expand(ops.sources)
        if expanded_sources is not ops.sources:
            ops = replace(ops, sources=expanded_sources)
        session: Optional[Session] = None
        session = Session(
            sources=ops.sources,
            device=device,
            val_frac=0.0,
            non_blocking_copy=True,
            sanitize=True,
            flatten_features=True,
        ).open()
        data_loader = session.training_loader
        chunk_dir = (os.path.join(ops.ckpt_dir, "pred_chunks") if (ops.ckpt_dir or "") else None)
        if chunk_dir and torch.distributed.get_rank() == 0:
            with contextlib.suppress(Exception):
                os.makedirs(chunk_dir, exist_ok=True)
        if torch.distributed.is_initialized():
            pass
        if ops.mode in ("predict", "infer"):
            if not chunk_dir:
                raise RuntimeError("predict/infer requires chunk_dir (streaming enforced)")
        try:
                                                                                    
            result = infer(
                model=model,
                device=device,
                local_rank=local_rank,
                ops=ops,
                data_loader=data_loader,
                chunk_dir=chunk_dir,
                dataset=metadata,
            )
            if result is not None and ret_sink is not None:
                ret_sink.update(result)
        finally:
            if session is not None:
                session.close()
        distributed_barrier(device)
        torch.distributed.destroy_process_group()
        return None
    raise ValueError(f"unsupported ops mode: {ops.mode}")


def _unwrap_for_microbatch(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    m: Any = model
    for _ in range(8):
        if hasattr(m, "microbatch") and hasattr(m, "_auto_microbatch_pending"):
            return m
        child = getattr(m, "module", None)
        if child is None or child is m:
            break
        m = child
    return None


torch_compile_safe(runtime_module=sys.modules[__name__])
