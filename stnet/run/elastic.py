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
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple

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

from ..data.pipeline import BatchIterator, extract_xy
from ..core.casting import env_bool, env_first, env_first_int, env_float, env_int, env_str
from ..nn.architecture import ModelPolicy
from ..core.precision import Autocast, PrecisionPolicy
from ..core.graph import inference_mode


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

    # STNET_NVML_DISABLE=1 disables NVML unconditionally.
    if env_bool("STNET_NVML_DISABLE", False):
        return True

    # STNET_NVML=0 also disables (mirrors other STNET toggles).
    if not env_bool("STNET_NVML", True):
        return True
    return False


def _nvml_fail_max() -> int:
    """Number of consecutive NVML query failures before entering backoff."""

    global _NVML_FAIL_MAX
    if _NVML_FAIL_MAX is not None:
        return int(_NVML_FAIL_MAX)

    v = env_first_int(("STNET_NVML_FAIL_MAX", "STNET_NVML_FAILURES"), default=3)
    _NVML_FAIL_MAX = max(1, int(v))
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

    raw = env_first(("STNET_NVML_BACKOFF_S", "STNET_NVML_BACKOFF"))
    if raw is not None:
        with contextlib.suppress(Exception):
            _NVML_BACKOFF_S = max(0.0, float(raw))
            return float(_NVML_BACKOFF_S)

    # Default: slightly longer backoff when no-GIL optimizations are enabled,
    # because telemetry is polled much more frequently.
    try:
        from ..core.system import Thread

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

    raw = env_first(("STNET_NVML_MIN_INTERVAL_S", "STNET_NVML_MIN_INTERVAL"))
    if raw is not None:
        with contextlib.suppress(Exception):
            _NVML_MIN_INTERVAL_S = max(0.0, float(raw))
            return float(_NVML_MIN_INTERVAL_S)

    # Default: throttle only when no-GIL optimizations are enabled.
    try:
        from ..core.system import Thread

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
        except Exception as exc:
            _nvml = None
            _NVML_READY = False
            if env_bool("STNET_DEBUG", False):
                logging.getLogger(__name__).debug("NVML init failed: %s", exc, exc_info=True)
    return bool(_NVML_READY)
try:
    import psutil as _psutil
except Exception:
    _psutil = None

from ..core.config import RuntimeConfig, coerce_model_config
from ..core.casting import to_torch_tensor
from ..core.staging import Cache, Pool
from ..data.pipeline import Dataset
# NOTE: Sampler scale is per-session/per-loader now; avoid global Sampler scaling here.
from ..core.losses import (CRPSLoss, DataFidelityLoss, LinearCombinationLoss,
                           LossWeightController, StandardNormalLoss, StudentsTLoss, TiledLoss)
from ..core.optimizers import (SWALR, AdamW, StochasticWeightAverage,
                               stochastic_weight_average)
from ..nn.architecture import Model
from ..nn.primitives import Recorder, resize_scaler_buffer
from ..core.compat import is_meta_or_fake_tensor
from ..core.graph import (
    cudagraph_step_end,
    torch_compile_safe,
    torch_safe_distributed,
)
from ..core.distributed import (distributed_barrier, distributed_sync,
                                   get_world_size, is_distributed, joining, no_sync,
                                   to_ddp, to_fsdp)
from ..core.profiler import FlopCounter
from ..core.system import (
    Memory,
    empty_device_cache,
    get_device,
    get_tlb,
    initialize_python_path,
    posix_time,
    set_float32_precision,
)

if TYPE_CHECKING:
    import numpy as _np
    from numpy.typing import NDArray as _NDArray

    Float64Array = _NDArray[_np.float64]
else:
    Float64Array = Any

_LOGGER = logging.getLogger(__name__)


_COMPILE_SAFE_DONE: bool = False


def _ensure_torch_compile_safe() -> None:
    """Call torch_compile_safe(...) once, but avoid import-time side effects."""

    global _COMPILE_SAFE_DONE
    if _COMPILE_SAFE_DONE:
        return
    _COMPILE_SAFE_DONE = True
    with contextlib.suppress(Exception):
        torch_compile_safe(runtime_module=sys.modules[__name__])


torch_safe_distributed()

_SAMPLER_SCALE_LOG_LOCK = threading.Lock()
_SAMPLER_SCALE_LOG_LAST_S: Dict[Tuple[int, str], float] = {}

_OOM_RETRY_LOCK = threading.Lock()
_OOM_RETRY_COUNT: Dict[Tuple[int, str, int], int] = {}


def _rt_env_flag(name: str, default: bool) -> bool:
    return env_bool(name, bool(default))


def _rt_env_int(name: str, default: int) -> int:
    return env_int(name, int(default))


def _rt_env_float(name: str, default: float) -> float:
    return env_float(name, float(default))


def _rt_maybe_torch_profiler(
    *,
    enabled: bool,
    tag: str,
    device: torch.device,
    out_dir: Optional[str],
    rank: int = 0,
) -> Optional[Any]:
    """Create a torch.profiler profile object (not started) when enabled.

    Controls (environment variables):
      - STNET_TORCH_PROFILE / STNET_TORCH_PROFILE_TRAIN / STNET_TORCH_PROFILE_INFER
      - STNET_TORCH_PROFILE_DIR
      - STNET_TORCH_PROFILE_WAIT / WARMUP / ACTIVE / REPEAT
      - STNET_TORCH_PROFILE_RECORD_SHAPES / PROFILE_MEMORY / WITH_STACK / WITH_FLOPS
      - STNET_TORCH_PROFILE_GROUP_BY_SHAPE / TOPK
    """
    if not bool(enabled):
        return None

    try:
        import torch.profiler as _tp  # local import to keep default overhead at ~0
    except Exception:
        return None

    activities = [_tp.ProfilerActivity.CPU]
    if device.type == "cuda":
        with contextlib.suppress(Exception):
            activities.append(_tp.ProfilerActivity.CUDA)
    elif device.type == "xpu":
        with contextlib.suppress(Exception):
            activities.append(getattr(_tp.ProfilerActivity, "XPU"))

    wait = max(0, int(_rt_env_int("STNET_TORCH_PROFILE_WAIT", 0)))
    warmup = max(0, int(_rt_env_int("STNET_TORCH_PROFILE_WARMUP", 2)))
    active = max(1, int(_rt_env_int("STNET_TORCH_PROFILE_ACTIVE", _rt_env_int("STNET_TORCH_PROFILE_STEPS", 8))))
    repeat = max(1, int(_rt_env_int("STNET_TORCH_PROFILE_REPEAT", 1)))

    record_shapes = bool(_rt_env_flag("STNET_TORCH_PROFILE_RECORD_SHAPES", False))
    profile_memory = bool(_rt_env_flag("STNET_TORCH_PROFILE_PROFILE_MEMORY", True))
    with_stack = bool(_rt_env_flag("STNET_TORCH_PROFILE_WITH_STACK", False))
    with_flops = bool(_rt_env_flag("STNET_TORCH_PROFILE_WITH_FLOPS", False))
    group_by_shape = bool(_rt_env_flag("STNET_TORCH_PROFILE_GROUP_BY_SHAPE", False))
    row_limit = max(5, int(_rt_env_int("STNET_TORCH_PROFILE_TOPK", 40)))

    if not out_dir:
        out_dir = os.path.join(os.getcwd(), "torch_profiler")
    out_dir = os.path.abspath(str(out_dir))
    with contextlib.suppress(Exception):
        os.makedirs(out_dir, exist_ok=True)

    worker_name = f"{str(tag)}-rank{int(rank)}"
    try:
        schedule = _tp.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        on_trace = _tp.tensorboard_trace_handler(out_dir, worker_name=worker_name)
        prof = _tp.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=on_trace,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
        )
        setattr(prof, "_stnet_row_limit", int(row_limit))
        setattr(prof, "_stnet_group_by_shape", bool(group_by_shape))
        setattr(prof, "_stnet_out_dir", str(out_dir))
        setattr(prof, "_stnet_tag", str(tag))
        return prof
    except Exception:
        return None


def _rt_log_torch_profiler_summary(
    prof: Any, *, device: torch.device, logger: logging.Logger, header: str
) -> None:
    if prof is None:
        return

    row_limit = int(getattr(prof, "_stnet_row_limit", 40) or 40)
    group_by_shape = bool(getattr(prof, "_stnet_group_by_shape", False))
    out_dir = str(getattr(prof, "_stnet_out_dir", ""))
    tag = str(getattr(prof, "_stnet_tag", header))

    try:
        ka = prof.key_averages(group_by_input_shape=group_by_shape)
    except Exception:
        with contextlib.suppress(Exception):
            ka = prof.key_averages()
        if "ka" not in locals():
            return

    table: Optional[str] = None
    for sk in ("self_cuda_time_total", "self_xpu_time_total", "self_cpu_time_total"):
        with contextlib.suppress(Exception):
            table = ka.table(sort_by=str(sk), row_limit=row_limit)
            if table:
                break

    if table:
        logger.info(
            "[torch.profiler] %s (trace dir: %s, tag: %s)\n%s",
            str(header),
            str(out_dir),
            str(tag),
            str(table),
        )




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

# (optional) torchao float8 helpers were removed here because they were unused.

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
    hook_mode = str(env_first(("STNET_META_MONITOR", "STNET_META_HOOK"), default="off") or "off").strip().lower()
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
    cast bookkeeping buffers (Recorder, Scaler, etc.) that must remain float64/int64.
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
                for name, p in params.items():
                    if p is None or not isinstance(p, torch.Tensor):
                        continue
                    if (not p.is_floating_point()) or p.dtype == dtype:
                        continue
                    params[name] = torch.nn.Parameter(
                        p.detach().to(dtype), requires_grad=bool(getattr(p, "requires_grad", True))
                    )
            bufs = getattr(mod, "_buffers", None)
            if bufs:
                for name, b in bufs.items():
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
        for key, value in state.items():
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
    match str(getattr(device, "type", "cpu")):
        case "cuda":
            return "nccl"
        case "xpu":
            return "xccl"
        case _:
            return "gloo"


def _set_backend(device: torch.device) -> None:
    with contextlib.suppress(Exception):
        if device.type == "cuda" and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
    rank = int(_rt_env_int("LOCAL_RANK", 0))
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
        for name, p in params.items():
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
    return BatchIterator.merge_meta_infos(sources)


def _expand(sources: Any) -> Any:
    return BatchIterator.expand_sources(sources)


def _calibrate_per_sample_mem(
    model: Model,
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
            return obj.to(device=dev, non_blocking=(dev.type in {"cuda", "xpu"}))
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
            from ..core.precision import Autocast
            from ..core.graph import inference_mode

            feats, labels, *_rest = meta.preprocess(batch)
            X = to_torch_tensor(feats)
            X = torch.atleast_2d(X)

            if X.dim() == 2 and int(X.shape[1]) == int(getattr(ops, "in_dim", X.shape[1])):
                X = X.to(device=device, non_blocking=(device.type in {"cuda", "xpu"}))

                if with_backward:
                    model.train()
                    with Autocast.float(device):
                        Y_flat = None
                        if labels is not None:
                            Y = to_torch_tensor(labels)
                            Y = torch.atleast_2d(Y).to(device=device, non_blocking=(device.type in {"cuda", "xpu"}))
                            Y_flat = Y.reshape(Y.shape[0], -1)

                        y_hat, loss_val = model(
                            X,
                            labels_flat=Y_flat,
                            global_loss=global_loss,
                            local_loss=local_loss,
                            loss_weights=loss_weights,
                            calibrate_output=False,
                        )
                    target: Optional[torch.Tensor] = None
                    if isinstance(loss_val, torch.Tensor):
                        target = loss_val
                    elif isinstance(y_hat, torch.Tensor):
                        target = y_hat

                    if target is not None:
                        loss = target if target.ndim == 0 else target.mean()
                        loss.backward()
                        forward_ran = True
                else:
                    with inference_mode(model), Autocast.float(device):
                        _ = model(
                            X,
                            global_loss=None,
                            local_loss=None,
                            loss_weights=None,
                            calibrate_output=True,
                            return_loss=False,
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
    dev_id: Optional[int | torch.device] = None
    dev_type = getattr(device, "type", "cpu")
    if dev_type in ("cuda", "xpu", "mps"):
        index = (
            device.index
            if getattr(device, "index", None) is not None
            else env_int("LOCAL_RANK", int(local_rank))
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


def _pin_tensors_with_cpu_pool(
    *tensors: torch.Tensor,
    device: torch.device,
    cpu_pool: Any,
    pool_handles: dict[int, object],
) -> Tuple[torch.Tensor, ...]:
    """Best-effort pinning via the CPU pool (reduces pinned alloc churn).

    Returns a tuple of tensors (same arity as input). Any newly allocated
    pinned buffers are tracked in `pool_handles` so they can be released once
    the async device copy has consumed them.
    """

    if device.type not in ("cuda", "xpu") or cpu_pool is None:
        return tensors

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


def _to_device_with_stream_and_pool(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    cpu_pool: Any,
    pool_handles: dict[int, object],
) -> torch.Tensor:
    """Move a CPU tensor to `device` using non-blocking transfers when possible.

    If `tensor` was allocated from `cpu_pool` we release its handle after the
    transfer has been consumed by the current device stream.
    """

    if not torch.is_tensor(tensor):
        return tensor

    non_blocking_ok = (device.type in ("cuda", "xpu"))

    if device.type in ("cuda", "xpu") and tensor.device.type == "cpu":
        # If this tensor came from the pinned CPU pool, pop its handle now so we
        # do not leak on exceptions. We release it back to the pool only when it is safe.
        h = pool_handles.pop(id(tensor), None)

        try:
            if not (hasattr(tensor, "is_pinned") and tensor.is_pinned()):
                pinned = torch.empty_like(tensor, device="cpu", pin_memory=True)
                pinned.copy_(tensor, non_blocking=False)
            else:
                pinned = tensor

            backend = getattr(torch, device.type, None)
            if backend is None or not hasattr(backend, "current_stream") or not hasattr(backend, "Event"):
                tensor_dev = pinned.to(device, non_blocking=False)
                if h is not None and cpu_pool is not None:
                    with contextlib.suppress(Exception):
                        cpu_pool.release(h)
                return tensor_dev

            # Fast path: async H2D with stream-aware lifetime tracking.
            tensor_dev = pinned.to(device, non_blocking=True)
            stream = backend.current_stream(device)
            with contextlib.suppress(Exception):
                pinned.record_stream(stream)

            if h is not None and cpu_pool is not None:
                try:
                    evt = backend.Event()
                    evt.record(stream)
                    cpu_pool.release_after(h, evt)
                except Exception:
                    # Very rare: if event plumbing fails, fall back to a sync+release to avoid leaks.
                    with contextlib.suppress(Exception):
                        sync = getattr(backend, "synchronize", None)
                        if callable(sync):
                            try:
                                sync(device=device)
                            except TypeError:
                                sync(device)
                    with contextlib.suppress(Exception):
                        cpu_pool.release(h)
            return tensor_dev
        except Exception:
            # Conservative fallback: do a blocking copy, then release the handle immediately.
            tensor_dev = tensor.to(device, non_blocking=False)
            if h is not None and cpu_pool is not None:
                with contextlib.suppress(Exception):
                    cpu_pool.release(h)
            return tensor_dev

    return tensor.to(device, non_blocking=non_blocking_ok)


def _drain_pool_handles(
    pool_handles: dict[int, object],
    *,
    cpu_pool: Any,
    device: torch.device,
) -> None:
    """Best-effort cleanup for any outstanding cpu_pool handles."""

    if not pool_handles or cpu_pool is None:
        pool_handles.clear()
        return

    # Ensure any in-flight transfers consuming pinned buffers have completed before reuse.
    try:
        if device.type == "cuda" and hasattr(torch, "cuda"):
            torch.cuda.synchronize(device=device)
        elif device.type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.synchronize(device=device)
    except Exception:
        pass

    for h in tuple(pool_handles.values()):
        with contextlib.suppress(Exception):
            cpu_pool.release(h)
    pool_handles.clear()


def _preprocess_pin_h2d(
    meta: Any,
    raw: Any,
    *,
    device: torch.device,
    pin_tensor: Callable[..., tuple[Any, ...]],
    to_device: Callable[[torch.Tensor], torch.Tensor],
    use_timer: bool,
    require_labels: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, int, float]:
    """Common batch path for all backends.

    Flow:
        preprocess -> pin(pool) -> async H2D(stream)

    Notes:
      - pin(pool) is only meaningful for CUDA/XPU; it is a no-op elsewhere.
      - For infer we typically pass use_timer=False to avoid per-batch sync.
    """

    feat, label, *_ = meta.preprocess(raw)
    X = to_torch_tensor(feat)

    Y: torch.Tensor | None
    if label is None:
        if require_labels:
            raise RuntimeError("Batch is missing labels.")
        Y = None
    else:
        Y = to_torch_tensor(label)

    # Pin (CUDA/XPU only). For timing consistency, we set t_ready after pinning.
    if Y is None:
        (X,) = pin_tensor(X)
    else:
        X, Y = pin_tensor(X, Y)
    t_ready = time.perf_counter_ns()

    # H2D copy. CUDA/XPU can use events; MPS/CPU path falls back to perf_counter.
    if use_timer:
        h2d_s_ev, h2d_e_ev = (
            torch.Event(device=device, enable_timing=True),
            torch.Event(device=device, enable_timing=True),
        )
        h2d_s_ev.record()
        X = to_device(X)
        if Y is not None:
            Y = to_device(Y)
        h2d_e_ev.record()
        h2d_e_ev.synchronize()
        h2d_s = float(h2d_s_ev.elapsed_time(h2d_e_ev)) / 1000.0
    else:
        t_h2d_s = time.perf_counter_ns()
        X = to_device(X)
        if Y is not None:
            Y = to_device(Y)
        t_h2d_e = time.perf_counter_ns()
        h2d_s = (t_h2d_e - t_h2d_s) / 1_000_000_000.0

    return X, Y, t_ready, h2d_s


def _resolve_train_process_group(meta: Any, model: Any) -> Any:
    """Best-effort process group resolution for training-time reductions."""

    candidates = [
        (meta, "process_group"),
        (meta, "distributed_process_group"),
    ]
    tm = model.module if hasattr(model, "module") else model
    candidates.extend(
        [
            (tm, "process_group"),
            (tm, "distributed_process_group"),
        ]
    )

    for obj, attr in candidates:
        try:
            pg = getattr(obj, attr, None)
        except Exception:
            pg = None
        if pg is not None:
            return pg
    return None


def _get_ws_for_pg(pg: Any) -> int:
    try:
        if pg is None:
            return max(1, int(get_world_size()))
        return int(torch.distributed.get_world_size(group=pg))
    except Exception:
        return max(1, int(get_world_size()))


def _all_reduce_sum_in_pg(t: torch.Tensor, pg: Any) -> None:
    if pg is None:
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    else:
        torch.distributed.all_reduce(
            t,
            op=torch.distributed.ReduceOp.SUM,
            group=pg,
        )


def epochs(
    model: Model,
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
            cpu_pool_cap = max(2, int(_rt_env_int("STNET_RUNTIME_PIN_POOL_CAPACITY", 8)))
            cpu_pool = Pool(capacity=cpu_pool_cap)
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
    # Centralize env parsing through core.casting helpers.
    dev_margin = env_float("STNET_DEVICE_MARGIN", 0.8)
    host_margin = env_float("STNET_HOST_MARGIN", 0.8)

    budget_slack = env_float("STNET_BUDGET_SLACK", 1.25)
    budget_slack = max(1.0, min(4.0, float(budget_slack)))

    dev_budget_ratio = env_float("STNET_DEVICE_BUDGET_RATIO", 1.0)
    dev_budget_min_bytes = env_int("STNET_DEVICE_BUDGET_MIN_BYTES", 0)
    _dev_budget_max_raw = env_int("STNET_DEVICE_BUDGET_MAX_BYTES", 0)
    dev_budget_max_bytes: Optional[int] = None if int(_dev_budget_max_raw) <= 0 else int(_dev_budget_max_raw)

    host_budget_ratio = env_float("STNET_HOST_BUDGET_RATIO", 1.0)
    host_budget_min_bytes = env_int("STNET_HOST_BUDGET_MIN_BYTES", 0)
    _host_budget_max_raw = env_int("STNET_HOST_BUDGET_MAX_BYTES", 0)
    host_budget_max_bytes: Optional[int] = None if int(_host_budget_max_raw) <= 0 else int(_host_budget_max_raw)

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
                prefetch_factor=int(env_int("STNET_HOST_PREFETCH_FACTOR", 4)),
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
    hist: Optional[Recorder] = None
    maybe_hist = getattr(model_for_hist, "logger", None)
    if isinstance(maybe_hist, Recorder):
        hist = maybe_hist
    if hist is None:
        maybe_hist = getattr(model_for_hist, "history", None)
        if isinstance(maybe_hist, Recorder):
            hist = maybe_hist
    if hist is None:
        hist = Recorder()
        try:
            setattr(model_for_hist, "logger", hist)
        except Exception:
            pass

    if isinstance(hist, Recorder):
        # posix_time is epoch-based; tz_name does not affect the value.
        start_ns = posix_time()
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
        used_memmap_stats = False
        x_count: int = 0
        x_sum: Optional[torch.Tensor] = None
        x_sum_sq: Optional[torch.Tensor] = None
        y_count: int = 0
        y_sum: Optional[torch.Tensor] = None
        y_sum_sq: Optional[torch.Tensor] = None

        # Prefer persisted scaler stats from memmap metadata when available.
        # This avoids an expensive pre-epoch full scan of the training loader.
        scaler_stats: Optional[dict] = None
        with contextlib.suppress(Exception):
            scaler_stats = BatchIterator.load_scaler_stats(ops.sources)
        if scaler_stats is not None:
            used_memmap_stats = True
            x_count = int(scaler_stats.get("train_count") or 0)
            y_count = int(x_count)
            x_sum = scaler_stats["x_sum"].to(device=scaler_x_device)
            x_sum_sq = scaler_stats["x_sum_sq"].to(device=scaler_x_device)
            y_sum = scaler_stats["y_sum"].to(device=scaler_y_device)
            y_sum_sq = scaler_stats["y_sum_sq"].to(device=scaler_y_device)
        else:
            # Fallback: compute scaler stats by scanning the training loader once.
            # Hot path: avoid per-batch temporary allocations and large intermediates.
            sx_tmp: Optional[torch.Tensor] = None
            sx2_tmp: Optional[torch.Tensor] = None
            sy_tmp: Optional[torch.Tensor] = None
            sy2_tmp: Optional[torch.Tensor] = None

            for batch in train_loader:
                feats, labs = extract_xy(batch, labels_required=True)

                if feats.ndim > 2:
                    feats = feats.reshape(feats.shape[0], -1)

                with torch.inference_mode():
                    xf = feats.to(device=scaler_x_device, dtype=torch.float64)
                    if xf.ndim > 2:
                        xf = xf.reshape(xf.shape[0], -1)
                    n_x = int(xf.shape[0])
                    if n_x > 0:
                        x_count += n_x
                        if x_sum is None:
                            x_sum = torch.zeros(
                                xf.shape[1], device=scaler_x_device, dtype=torch.float64
                            )
                            x_sum_sq = torch.zeros_like(x_sum)
                            sx_tmp = torch.empty_like(x_sum)
                            sx2_tmp = torch.empty_like(x_sum)
                        assert x_sum is not None and x_sum_sq is not None
                        assert sx_tmp is not None and sx2_tmp is not None

                        torch.sum(xf, dim=0, out=sx_tmp)
                        x_sum.add_(sx_tmp)

                        # Avoid mutating the original batch (feats may share storage).
                        xf_sq = xf * xf
                        torch.sum(xf_sq, dim=0, out=sx2_tmp)
                        x_sum_sq.add_(sx2_tmp)

                    yf = labs.to(device=scaler_y_device, dtype=torch.float64)
                    if yf.ndim > 2:
                        yf = yf.reshape(yf.shape[0], -1)
                    n_y = int(yf.shape[0])
                    if n_y > 0:
                        y_count += n_y
                        if y_sum is None:
                            y_sum = torch.zeros(
                                yf.shape[1], device=scaler_y_device, dtype=torch.float64
                            )
                            y_sum_sq = torch.zeros_like(y_sum)
                            sy_tmp = torch.empty_like(y_sum)
                            sy2_tmp = torch.empty_like(y_sum)
                        assert y_sum is not None and y_sum_sq is not None
                        assert sy_tmp is not None and sy2_tmp is not None

                        torch.sum(yf, dim=0, out=sy_tmp)
                        y_sum.add_(sy_tmp)

                        yf_sq = yf * yf
                        torch.sum(yf_sq, dim=0, out=sy2_tmp)
                        y_sum_sq.add_(sy2_tmp)

            # Only reduce across ranks when stats were computed locally.
            # (When loaded from memmaps, every rank sees the same aggregated stats.)
            if is_distributed() and not used_memmap_stats:
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
        pool_handles: dict[int, object] = {}
        pin_tensor = partial(
            _pin_tensors_with_cpu_pool,
            device=device,
            cpu_pool=cpu_pool,
            pool_handles=pool_handles,
        )
        to_device_with_stream = partial(
            _to_device_with_stream_and_pool,
            device=device,
            cpu_pool=cpu_pool,
            pool_handles=pool_handles,
        )

        # Optional torch.profiler instrumentation (disabled by default).
        torch_prof: Optional[Any] = None
        prof_enabled = _rt_env_flag(
            "STNET_TORCH_PROFILE_TRAIN", _rt_env_flag("STNET_TORCH_PROFILE", False)
        )
        prof_all_ranks = _rt_env_flag("STNET_TORCH_PROFILE_ALL_RANKS", False)
        prof_rank = int(torch.distributed.get_rank()) if is_distributed() else 0
        if prof_enabled and (prof_all_ranks or prof_rank == 0):
            prof_dir = env_str("STNET_TORCH_PROFILE_DIR")
            if not prof_dir:
                prof_dir = os.path.join(str(ops.ckpt_dir or "."), "torch_profiler")
            torch_prof = _rt_maybe_torch_profiler(
                enabled=True,
                tag=f"train-{str(run_id)}",
                device=device,
                out_dir=str(prof_dir),
                rank=prof_rank,
            )
            if torch_prof is not None:
                with contextlib.suppress(Exception):
                    torch_prof.start()

        for epoch_idx in range(int(total_epochs)):
            # Ensure the training sampler uses a different shuffle ordering each epoch.
            # This mirrors PyTorch's DistributedSampler best practice:
            # call sampler.set_epoch(epoch) before creating/iterating the DataLoader iterator.
            #
            # We attach per-epoch-capable sampler objects to the (wrapped) loader as
            # `_stnet_epochables` inside stnet.data.pipeline.fetch()/Session.
            with contextlib.suppress(Exception):
                epochables = getattr(train_loader, "_stnet_epochables", None)
                if epochables is not None:
                    for obj in epochables:
                        fn = getattr(obj, "set_epoch", None)
                        if callable(fn):
                            fn(int(epoch_idx))
                else:
                    # Fallbacks for other loader types (e.g. vanilla DataLoader).
                    fn = getattr(getattr(train_loader, "sampler", None), "set_epoch", None)
                    if callable(fn):
                        fn(int(epoch_idx))
                    fn = getattr(train_loader, "set_epoch", None)
                    if callable(fn):
                        fn(int(epoch_idx))
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
                            X, Y_opt, t_ready, h2d_s = _preprocess_pin_h2d(
                                meta,
                                _raw,
                                device=device,
                                pin_tensor=pin_tensor,
                                to_device=to_device_with_stream,
                                use_timer=use_timer,
                                require_labels=True,
                            )
                            assert Y_opt is not None
                            Y = Y_opt
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
                                            Y_flat = Y_flat.to(
                                                device,
                                                dtype=param_dtype,
                                                non_blocking=(device.type in ("cuda", "xpu")),
                                            )
                                        y_hat, loss_val, loss_top_val, loss_bottom_val = model(
                                            X,
                                            labels_flat=Y_flat,
                                            global_loss=top_loss,
                                            local_loss=bottom_loss,
                                            loss_weights=loss_controller.weights(),
                                            calibrate_output=False,
                                            return_loss_components=True,
                                        )
                                    if isinstance(loss_val, torch.Tensor) and loss_val.ndim > 0:
                                        loss_val = loss_val.mean()
                                    if isinstance(loss_top_val, torch.Tensor) and loss_top_val.ndim > 0:
                                        loss_top_val = loss_top_val.mean()
                                    if (
                                        isinstance(loss_bottom_val, torch.Tensor)
                                        and loss_bottom_val.ndim > 0
                                    ):
                                        loss_bottom_val = loss_bottom_val.mean()
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
                            if isinstance(hist, Recorder):
                                try:
                                    if train_steps <= 0 or step_idx % max(1, int(train_steps * 0.01)) == 0:
                                        hist.record_batch(X, Y)
                                except Exception:
                                    pass
                            if torch_prof is not None:
                                torch_prof.step()
                            t_fetch_start = time.perf_counter_ns()
                            if cpu_pool is not None and ((step_idx + 1) & 255) == 0:
                                with contextlib.suppress(Exception):
                                    cpu_pool.collect()

                            with contextlib.suppress(Exception):
                                _oom_retry_clear(train_loader, "train", step_idx)

                            break

                        except RuntimeError as e:
                            with contextlib.suppress(Exception):
                                _drain_pool_handles(pool_handles, cpu_pool=cpu_pool, device=device)
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
                                # NOTE: scale controller is per-session/per-loader (see BatchState).
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
                    with inference_mode(model), Autocast.float(device):
                        t_fetch_start = time.perf_counter_ns()
                        for _vstep, _raw in enumerate(val_loader):
                            while True:
                                try:
                                    X, Y_opt, t_ready, h2d_s = _preprocess_pin_h2d(
                                        meta,
                                        _raw,
                                        device=device,
                                        pin_tensor=pin_tensor,
                                        to_device=to_device_with_stream,
                                        use_timer=use_timer,
                                        require_labels=True,
                                    )
                                    assert Y_opt is not None
                                    Y = Y_opt

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
                                        with Autocast.float(device):
                                            Yv_flat = Y.reshape(Y.shape[0], -1).to(
                                                device,
                                                dtype=param_dtype,
                                                non_blocking=(device.type in ("cuda", "xpu")),
                                            )
                                            _y, _loss_val = model(
                                                X,
                                                labels_flat=Yv_flat,
                                                global_loss=top_loss,
                                                local_loss=bottom_loss,
                                                loss_weights=loss_controller.weights(),
                                                calibrate_output=False,
                                            )
                                        if isinstance(_loss_val, torch.Tensor) and _loss_val.ndim > 0:
                                            _loss_val = _loss_val.mean()
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

                                    if torch_prof is not None:
                                        torch_prof.step()

                                    t_fetch_start = time.perf_counter_ns()
                                    if cpu_pool is not None and ((_vstep + 1) & 255) == 0:
                                        with contextlib.suppress(Exception):
                                            cpu_pool.collect()

                                    with contextlib.suppress(Exception):
                                        _oom_retry_clear(val_loader, "val", _vstep)

                                    break

                                except RuntimeError as e:
                                    with contextlib.suppress(Exception):
                                        _drain_pool_handles(pool_handles, cpu_pool=cpu_pool, device=device)
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
                stats_cpu = stats.detach().cpu()
                comp_time = float(stats_cpu[0].item())
                io_time = float(stats_cpu[1].item())
                flops = float(stats_cpu[2].item())
                io_bytes = float(stats_cpu[3].item())
                train_samples_epoch = float(stats_cpu[4].item())
                distributed_barrier(device)
            updated_this_epoch = False
            if swa_enabled and epoch_idx >= swa_start_epoch:
                try:
                    swa_helper.update_weight()
                    updated_this_epoch = True
                except Exception:
                    pass
            if not scheduler_step_per_batch:
                try:
                    sched.step()
                except Exception:
                    pass
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
            x_b, y_b = extract_xy(batch, labels_required=True)
            x_raw = x_b.to(device)
            y_raw = y_b.to(scaler_y_device)
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
            denom = var_x
            tiny_mask = denom.abs() < eps
            denom[tiny_mask] = 1.0

            a = (cov_xy / denom).to(dtype=torch.float32)
            b = (mean_y - a.to(dtype=torch.float64) * mean_x).to(dtype=torch.float32)
            a[tiny_mask] = 1.0
            b[tiny_mask] = 0.0

            model_for_scaler.scaler.set_affine(a, b)

    if torch_prof is not None:
        with contextlib.suppress(Exception):
            torch_prof.stop()
        _rt_log_torch_profiler_summary(
            torch_prof, device=device, logger=_LOGGER, header="train/val"
        )

    if local_rank == 0 and status_bar is not None:
        mbps = prev_io_bytes / max(prev_io_time, 1e-06) / MB_DIV
        tflops = prev_flops / max(prev_comp_time, 1e-06) / 1_000_000_000_000.0
        status_bar.set_postfix_str(
            f"{mbps:.2f} MB/s, {tflops:.2f} TFLOPS", refresh=True
        )
        status_bar.close()
    # posix_time is epoch-based; tz_name does not affect the value.
    end_kst_ns = posix_time()
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
        if isinstance(hist, Recorder):
            try:
                end_sec = round(float(end_kst_ns) / 1e9, 6)
                world = max(1, get_world_size(device)) if is_distributed() else 1
                hist.end_session(end_sec, peers=world)

                if ops.ckpt_dir and (not is_distributed() or int(torch.distributed.get_rank()) == 0):
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
                    BatchIterator.atomic_write_json(history_path, payload, indent=2)
            except Exception:
                pass
    except Exception:
        pass


def infer(
    model: Model,
    device: torch.device,
    local_rank: int,
    ops: RuntimeConfig,
    *,
    data_loader: Optional[Iterable[TensorDictBase]] = None,
    chunk_dir: Optional[str] = None,
    dataset: Optional[Dataset] = None,
) -> Optional[Dict[Tuple, torch.Tensor]]:

    import glob

    _ensure_torch_compile_safe()

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

    torch_prof: Optional[Any] = None
    prof_enabled = _rt_env_flag(
        "STNET_TORCH_PROFILE_INFER", _rt_env_flag("STNET_TORCH_PROFILE", False)
    )
    prof_all_ranks = _rt_env_flag("STNET_TORCH_PROFILE_ALL_RANKS", False)
    if prof_enabled and (prof_all_ranks or int(rank) == 0):
        prof_dir = env_str("STNET_TORCH_PROFILE_DIR")
        if not prof_dir:
            prof_dir = os.path.join(str(ops.ckpt_dir or "."), "torch_profiler")
        torch_prof = _rt_maybe_torch_profiler(
            enabled=True,
            tag="infer",
            device=device,
            out_dir=str(prof_dir),
            rank=int(rank),
        )
        if torch_prof is not None:
            with contextlib.suppress(Exception):
                torch_prof.start()

    if rank == 0:
        os.makedirs(chunk_dir, exist_ok=True)
    distributed_barrier(device)
    cache = Cache(chunk_dir, max_queue=4)

    target_rows = int(_rt_env_int("STNET_PRED_CHUNK_ROWS", 0))

    if target_rows <= 0:
        out_shape = tuple(int(x) for x in (ops.out_shape or ()))
        out_numel = 1
        for d in out_shape:
            out_numel *= max(1, int(d))
        est_row_bytes = max(1, out_numel * 4)
        target_bytes = int(_rt_env_int("STNET_PRED_CHUNK_BYTES", 64 * 1024 * 1024))
        target_rows = max(256, min(65536, target_bytes // est_row_bytes))

    run_model = to_ddp(model, device=device)
    run_model.eval()
    module_eval = run_model.module if hasattr(run_model, "module") else run_model
    distributed_sync(module_eval, device=device)

    # Stage inputs through the same preprocess -> pin(pool) -> async H2D path
    # used by training. For infer we intentionally avoid event timing (which
    # would introduce per-batch synchronization).
    cpu_pool: Any | None = None
    if device.type in {"cuda", "xpu"} and Pool is not None:
        with contextlib.suppress(Exception):
            Memory.prefer_local_numa()
        with contextlib.suppress(Exception):
            cpu_pool_cap = max(2, int(_rt_env_int("STNET_RUNTIME_PIN_POOL_CAPACITY", 8)))
            cpu_pool = Pool(capacity=cpu_pool_cap)

    # Dedicated (small) pinned CPU pool for prediction chunks (D2H staging).
    # This lets us use non_blocking=True for GPU->CPU copies and avoid re-allocating
    # large pinned buffers every flush.
    pred_pool: Any | None = None
    if device.type == "cuda" and Pool is not None and _rt_env_flag("STNET_PRED_PINNED", True):
        with contextlib.suppress(Exception):
            pred_pool_cap = max(2, int(_rt_env_int("STNET_PRED_PIN_POOL_CAPACITY", 2)))
            pred_pool = Pool(capacity=pred_pool_cap, pin_memory=True)

    pool_handles: dict[int, object] = {}
    pin_tensor = partial(
        _pin_tensors_with_cpu_pool,
        device=device,
        cpu_pool=cpu_pool,
        pool_handles=pool_handles,
    )
    to_device_with_stream = partial(
        _to_device_with_stream_and_pool,
        device=device,
        cpu_pool=cpu_pool,
        pool_handles=pool_handles,
    )

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
    # Fast path: stage into preallocated chunk buffers to avoid repeated
    # concatenations and reduce allocator churn.
    use_buffer = True
    rows_buf: Optional[torch.Tensor] = None
    pred_buf: Optional[torch.Tensor] = None
    pred_handle: Any | None = None
    buf_needs_wait_evt = False
    buf_fill = 0

    # Fallback path for variable-shaped outputs.
    pending_rows: list[torch.Tensor] = []
    pending_preds: list[torch.Tensor] = []
    pending_count = 0
    chunk_idx = 0
    row_cursor = 0
    first_tail: Optional[Tuple[int, ...]] = None
    pending_tail: Optional[Tuple[int, ...]] = None
    variable_shape = False

    def _flush() -> None:
        nonlocal chunk_idx, pending_count, buf_fill, pending_tail, pred_buf, pred_handle, buf_needs_wait_evt
        if use_buffer:
            if buf_fill <= 0:
                return
            assert rows_buf is not None and pred_buf is not None
            # rows_buf is reused immediately; clone to avoid the writer thread
            # observing mutated contents (race).
            rows = rows_buf[:buf_fill].clone()
            preds = pred_buf[:buf_fill]

            # Hand off this pred buffer to the async writer. Do not reuse until
            # release_cb fires.
            local_handle = pred_handle
            need_wait_evt = bool(buf_needs_wait_evt)

            buf_fill = 0
            pred_buf = None
            pred_handle = None
            buf_needs_wait_evt = False
        else:
            if pending_count <= 0:
                return
            rows = torch.cat(pending_rows, dim=0).to(dtype=torch.int64, copy=False)
            if not bool(rows.is_contiguous()):
                rows = rows.contiguous()
            preds = torch.cat(pending_preds, dim=0)
            if not bool(preds.is_contiguous()):
                preds = preds.contiguous()
            local_handle = None
            need_wait_evt = False

        rows_path = os.path.join(chunk_dir, f"part-r{rank:05d}-c{chunk_idx:06d}-rows.pt")

        # B approach:
        # - Fixed-shape path (use_buffer=True): write preds as .mmt
        # - Variable-shape path (use_buffer=False): keep .pt for safety/compat
        if use_buffer:
            pred_path = os.path.join(chunk_dir, f"part-r{rank:05d}-c{chunk_idx:06d}-pred.mmt")
        else:
            pred_path = os.path.join(chunk_dir, f"part-r{rank:05d}-c{chunk_idx:06d}-pred.pt")

        cache.submit(rows, path=rows_path)
        if use_buffer:
            # If we used non_blocking GPU->CPU copies into a pinned buffer, wait for
            # the final D2H transfer to complete before writing.
            wait_evt = None
            if need_wait_evt and device.type == "cuda":
                with contextlib.suppress(Exception):
                    wait_evt = torch.cuda.Event()
                    wait_evt.record()

            release_cb = None
            if local_handle is not None and pred_pool is not None:
                release_cb = lambda h=local_handle: pred_pool.release(h)

            cache.submit(preds, path=pred_path, wait_event=wait_evt, release_cb=release_cb)
        else:
            cache.submit(preds, path=pred_path)

        chunk_idx += 1
        if not use_buffer:
            pending_rows.clear()
            pending_preds.clear()
            pending_count = 0
            pending_tail = None

        del rows, preds

    def _append(rows_cpu: torch.Tensor, preds: torch.Tensor) -> None:
        nonlocal use_buffer, rows_buf, pred_buf, pred_handle, buf_fill, pending_count, first_tail, variable_shape, pending_tail, buf_needs_wait_evt

        if preds.ndim < 1:
            return
        b = int(preds.shape[0])
        if b <= 0:
            return

        rows_cpu = rows_cpu.reshape(-1).to(dtype=torch.int64, device="cpu", copy=False)
        if rows_cpu.numel() != b:
            raise RuntimeError(f"infer: rows/preds batch mismatch rows={rows_cpu.numel()} preds={b}")

        preds = preds.detach()

        tail = tuple(int(x) for x in preds.shape[1:])
        if first_tail is None:
            first_tail = tail
        elif tail != first_tail:
            variable_shape = True
            if use_buffer:
                # Flush current fixed-shape buffer as .mmt, then switch to variable-shape path.
                _flush()
                use_buffer = False
                pending_tail = None

        if not use_buffer:
            # Ensure pending chunks are homogeneous in tail shape; otherwise, flush.
            if pending_tail is None:
                pending_tail = tail
            elif tail != pending_tail:
                _flush()
                pending_tail = tail

            pending_rows.append(rows_cpu.clone())
            if preds.device.type != "cpu":
                preds = preds.to(device="cpu")
            pending_preds.append(preds)
            pending_count += b
            if pending_count >= target_rows:
                _flush()
            return

        # Fixed-shape buffer path.
        if rows_buf is None:
            rows_buf = torch.empty((target_rows,), dtype=torch.int64)

        # Allocate/validate a pinned CPU staging buffer for predictions.
        if pred_buf is None:
            if pred_pool is not None:
                pred_buf, pred_handle = pred_pool.get((target_rows, *tail), dtype=preds.dtype, return_handle=True)
            else:
                pred_buf = torch.empty((target_rows, *tail), dtype=preds.dtype)
                pred_handle = None
        else:
            if pred_buf.dtype != preds.dtype or tuple(int(x) for x in pred_buf.shape[1:]) != tail:
                if buf_fill > 0:
                    _flush()
                if pred_pool is not None and pred_handle is not None:
                    with contextlib.suppress(Exception):
                        pred_pool.release(pred_handle)
                pred_buf = None
                pred_handle = None

        start = 0
        while start < b:
            # _flush() hands off and clears pred_buf; reallocate as needed.
            if pred_buf is None:
                if pred_pool is not None:
                    pred_buf, pred_handle = pred_pool.get((target_rows, *tail), dtype=preds.dtype, return_handle=True)
                else:
                    pred_buf = torch.empty((target_rows, *tail), dtype=preds.dtype)
                    pred_handle = None

            space = target_rows - buf_fill
            if space <= 0:
                _flush()
                continue
            n = min(space, b - start)
            rows_buf[buf_fill : buf_fill + n].copy_(rows_cpu[start : start + n])

            # GPU->(pinned)CPU D2H copies can be scheduled asynchronously.
            non_blocking = bool(pred_buf.is_pinned()) and preds.device.type != "cpu"
            pred_buf[buf_fill : buf_fill + n].copy_(preds[start : start + n], non_blocking=non_blocking)
            if non_blocking:
                buf_needs_wait_evt = True

            buf_fill += n
            start += n
            if buf_fill >= target_rows:
                _flush()


    try:
        with inference_mode(run_model), Autocast.float(device):
            row_ids_buf: Optional[torch.Tensor] = None
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

                try:
                    X, _Y, _t_ready, _h2d_s = _preprocess_pin_h2d(
                        dataset,
                        batch,
                        device=device,
                        pin_tensor=pin_tensor,
                        to_device=to_device_with_stream,
                        use_timer=False,
                        require_labels=False,
                    )
                except Exception:
                    with contextlib.suppress(Exception):
                        _drain_pool_handles(pool_handles, cpu_pool=cpu_pool, device=device)
                    raise
                finally:
                    pool_handles.clear()

                X = torch.atleast_2d(X)
                bs = int(getattr(X, "shape", [0])[0]) if hasattr(X, "shape") else 0
                if bs <= 0:
                    if status_bar is not None:
                        status_bar.update(1)
                    continue

                if row_ids is None:
                    # Reuse a CPU buffer to avoid per-batch arange allocations.
                    if row_ids_buf is None or int(row_ids_buf.numel()) < bs:
                        row_ids_buf = torch.empty((bs,), device="cpu", dtype=torch.int64)
                    view = row_ids_buf[:bs]
                    torch.arange(
                        int(row_cursor),
                        int(row_cursor) + int(bs),
                        dtype=torch.int64,
                        device=view.device,
                        out=view,
                    )
                    row_ids = view
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

                    try:
                        # Request tensor output to keep inference memory use
                        # predictable (no TensorDict copies / aux tensors).
                        out = run_model(Xi, calibrate_output=True, return_loss=False)
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
                                del Xi
                            continue  
                        raise

                    if isinstance(out, tuple):
                        # Backward-compat: some wrappers may still return (pred, loss).
                        y_hat = out[0]
                    else:
                        y_hat = out

                    if not isinstance(y_hat, torch.Tensor):
                        raise RuntimeError("infer: unexpected model output type")

                    preds = y_hat.detach()
                    rows_cpu = rows_i if rows_i.device.type == "cpu" else rows_i.to(device="cpu")
                    _append(rows_cpu, preds)

                    del Xi, rows_i, out, y_hat, preds, rows_cpu
                    start = end

                if torch_prof is not None:
                    torch_prof.step()

                if status_bar is not None:
                    status_bar.update(1)

                del X, _Y, batch, row_ids

    finally:
        _flush()
        if torch_prof is not None:
            with contextlib.suppress(Exception):
                torch_prof.stop()
            _rt_log_torch_profiler_summary(
                torch_prof, device=device, logger=_LOGGER, header="infer"
            )
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
                pred_mmt = base + "-pred.mmt"
                pred_pt = base + "-pred.pt"

                if os.path.exists(pred_mmt):
                    pred_path = pred_mmt
                    meta_path = BatchIterator.mmt_meta_path(pred_mmt)
                    if not os.path.exists(meta_path):
                        raise RuntimeError(f"infer: missing pred meta for memmap part: {pred_mmt} -> {meta_path}")
                elif os.path.exists(pred_pt):
                    pred_path = pred_pt
                else:
                    raise RuntimeError(
                        f"infer: missing pred file for rows part: {rows_path} -> ({pred_mmt} or {pred_pt})"
                    )

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
            BatchIterator.atomic_write_json(man_path, manifest, indent=2)

        distributed_barrier(device)

    return None


def main(*args: Any, **kwargs: Any) -> Optional[Model]:
    from ..data.pipeline import Session

    if not args:
        raise TypeError("main requires at least a RuntimeConfig argument")
    initialize_python_path()
    _ensure_torch_compile_safe()
    ret_sink: Optional[Dict[Any, Any]] = None
    if isinstance(args[0], RuntimeConfig):
        ops = args[0]
        local_rank = env_int("LOCAL_RANK", 0)
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

    # Determinism + seeding are applied *inside* the runtime worker.
    det = bool(getattr(ops, "deterministic", False))
    seed_base = int(getattr(ops, "seed", 42))
    seed_value = int(seed_base) + int(local_rank)

    with contextlib.suppress(Exception):
        import random as _random
        _random.seed(seed_value)
    with contextlib.suppress(Exception):
        import numpy as _np
        _np.random.seed(seed_value)
    with contextlib.suppress(Exception):
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)

    with contextlib.suppress(Exception):
        torch.use_deterministic_algorithms(det, warn_only=False)
    with contextlib.suppress(Exception):
        torch.backends.cudnn.deterministic = det
        torch.backends.cudnn.benchmark = not det
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
        model = Model(ops.in_dim, ops.out_shape, config=cfg)
        if ops.init_ckpt_dir is not None and os.path.isdir(ops.init_ckpt_dir):
            fallback_init = os.path.join(ops.init_ckpt_dir, "model.pt")
            if os.path.isfile(fallback_init):
                from .io import _torch_load_checkpoint as _torch_load_checkpoint

                cpu_state = _torch_load_checkpoint(
                    fallback_init,
                    map_location="cpu",
                    weights_only=True,
                )
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
        precision = PrecisionPolicy.from_metadata(device=device, metadata=metadata, logger=_LOGGER)
        param_dtype = precision.master_float

        if device.type != "cuda":
            _LOGGER.warning(
                "Forcing CPU / non-CUDA config: mixed precision + NVIDIA fused layers may be unavailable."
            )

        model, _, _ = ModelPolicy.use_nvidia_layers(
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
            model, fp8_enabled, fp8_backend = ModelPolicy.enable_float8_training(
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
                train_shuffle=bool(getattr(ops, "shuffle", True)),
                seed=int(getattr(ops, "seed", 42)),
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
                    BatchIterator.atomic_write_json(
                        loader_state_path(ops.ckpt_dir or ""),
                        _dl,
                        indent=2,
                    )
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
        model = Model(ops.in_dim, ops.out_shape, config=cfg)
        # Fail-fast: predict/infer must load a trained checkpoint.
        if not ops.model_ckpt_dir:
            raise RuntimeError(
                'predict/infer requires model_ckpt_dir (checkpoint directory). '
                'Set RuntimeConfig.model_ckpt_dir to a directory produced by train().'
            )
        if not os.path.isdir(ops.model_ckpt_dir):
            raise RuntimeError(
                f'predict/infer: model_ckpt_dir does not exist or is not a directory: {ops.model_ckpt_dir!r}'
            )

        if ops.model_ckpt_dir is not None and os.path.isdir(ops.model_ckpt_dir):
            fallback_model = os.path.join(ops.model_ckpt_dir, "model.pt")
            if os.path.isfile(fallback_model):
                from .io import _torch_load_checkpoint as _torch_load_checkpoint

                cpu_state = _torch_load_checkpoint(
                    fallback_model,
                    map_location="cpu",
                    weights_only=True,
                )
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
        model.to(device, non_blocking=(device.type in ("cuda", "xpu"))).eval()
        metadata = Dataset.for_device(device)
        model, _, _ = ModelPolicy.use_nvidia_layers(model, device=device)
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
            model, _, _ = ModelPolicy.enable_float8_prediction(
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
            train_shuffle=bool(getattr(ops, "shuffle", False)),
            seed=int(getattr(ops, "seed", 7)),
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
