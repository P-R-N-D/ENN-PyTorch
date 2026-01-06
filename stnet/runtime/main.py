# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import glob
import random
import re
import platform
import socket
import logging
import math
import os
import sys
import threading
import time
import warnings
from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, MutableMapping, TypeAlias, overload

from tqdm.auto import tqdm
import numpy
import torch
import torch.distributed
import torch.nn as nn
from tensordict import TensorDictBase
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load,
    save
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from ..config import RuntimeConfig, coerce_model_config
from ..core.system import (
    Memory,
    empty_device_cache,
    get_affinity,
    get_device,
    init_python_path,
    is_cuda_bf16_supported,
    set_float32_precision,
)
from ..core.system import (
    accelerator_max_allocated_memory,
    accelerator_stream,
    accelerator_type,
    allocated_accelerator_memory,
    available_device_memory,
    flush_accelerator_memory_stats,
    get_accelerator_index,
    get_num_accelerators,
    is_accelerator_available,
    is_accelerator_timer_supported,
    is_pin_supported,
    new_accelerator_event,
    set_accelerator_index,
    set_accelerator_seed,
    sync_accelerator,
    posix_time,
)
from ..core.casting import (
    env_bool,
    env_first,
    env_first_int,
    env_float,
    env_int,
    env_str,
    to_torch_tensor
)
from ..core.compat import is_meta_or_fake_tensor
from ..core.staging import Cache, Pool
from ..core.distributed import (
    distributed_barrier,
    distributed_sync,
    joining,
    no_sync,
    get_world_size,
    is_distributed,
    to_ddp,
    to_fsdp,
)
from ..core.graph import (
    inference_mode,
    compile_safe,
    compile_distributed_safe,
    cudagraph_mark_step_begin,
    cudagraph_mark_step_end,
)
from ..core.profiler import FlopCounter
from ..core.precision import Autocast, PrecisionPolicy
from ..nn.architecture import Model, ModelPolicy
from ..nn.layers import Recorder, resize_scaler_buffer
from .losses import (
    CRPSLoss,
    DataFidelityLoss,
    LinearCombinationLoss,
    LossWeightController,
    StandardNormalLoss,
    StudentsTLoss,
    TiledLoss,
)
from .optimizers import SWALR, AdamW, stochastic_weight_average
from ..data.nodes import Storage
from ..data.pipeline import (
    Dataset,
    get_row
)
from ..data.schemas import read_json
from .io import _filtered_warnings, _torch_load_checkpoint

try:
    import psutil
except Exception:
    psutil = None

try:
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy
except Exception:
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy
    except Exception:
        MixedPrecisionPolicy = None


PathLike: TypeAlias = str | os.PathLike[str] | Path
JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
TorchDeviceLike: TypeAlias = torch.device | str | int
ReturnSink: TypeAlias = MutableMapping[str, object]

_LOGGER = logging.getLogger(__name__)

_nvml = None
_NVML_READY = False
_NVML_TRIED = False
_NVML_LOCK = threading.Lock()
_NVML_QUERY_LOCK = threading.Lock()
_NVML_HANDLE_CACHE = {}
_NVML_UTIL_CACHE = {}
_NVML_MIN_INTERVAL_S = None
_NVML_FAIL_COUNT = 0
_NVML_BACKOFF_UNTIL = 0.0
_NVML_BACKOFF_S = None
_NVML_FAIL_MAX = None

_COMPILE_SAFE_DONE = False
_COMPILE_SAFE_LOCK = threading.Lock()

_SAMPLER_SCALE_LOG_LOCK = threading.Lock()
_SAMPLER_SCALE_LOG_LAST_S = {}

_OOM_RETRY_LOCK = threading.Lock()
_OOM_RETRY_COUNT = {}

_TIMING_EVENT_TLS = threading.local()
_TIMING_EVENTS_UNSUPPORTED = object()

_IGNORED_WARNING_PATTERNS: tuple[str, ...] = (
    "torch.distributed is disabled, unavailable or uninitialized",
    "TypedStorage is deprecated",
    "Found a non-scalar tensor with numel=1 and ndim!=0",
    "distributed_broadcast: coalesced broadcast failed",
    "distributed_broadcast: per-tensor broadcast failed",
    "found no DeviceMesh from dtensor args",
    "mixed precision.*may be unavailable",
    "Either mode or options can be specified, but both can't be specified at the same time\\.",
)

_IGNORED_WARNING_MESSAGE_RE = re.compile(
    r".*(?:" + "|".join((f"(?:{p})" for p in _IGNORED_WARNING_PATTERNS)) + r").*"
)

MB_DIV = 1024.0 * 1024.0

_DL_STATE_FILE = "dataloader.json"


@lru_cache(maxsize=4)
def _is_nvml_disabled() -> bool:
    if env_bool("STNET_NVML_DISABLE", False):
        return True
    if not env_bool("STNET_NVML", True):
        return True
    return False


def _nvml_max_retry() -> object:
    global _NVML_FAIL_MAX
    if _NVML_FAIL_MAX is not None:
        return int(_NVML_FAIL_MAX)
    v = env_first_int(("STNET_NVML_FAIL_MAX", "STNET_NVML_FAILURES"), default=3)
    _NVML_FAIL_MAX = max(1, int(v))
    return int(_NVML_FAIL_MAX)


def _nvml_blocking_interval() -> object:
    global _NVML_BACKOFF_S
    if _NVML_BACKOFF_S is not None:
        return float(_NVML_BACKOFF_S)
    raw = env_first(("STNET_NVML_BACKOFF_S", "STNET_NVML_BACKOFF"))
    if raw is not None:
        with contextlib.suppress(Exception):
            _NVML_BACKOFF_S = max(0.0, float(raw))
            return float(_NVML_BACKOFF_S)
    try:
        from ..core.system import Thread
        _NVML_BACKOFF_S = 30.0 if bool(Thread.is_optimized_for_no_gil()) else 10.0
    except Exception:
        _NVML_BACKOFF_S = 10.0
    return float(_NVML_BACKOFF_S)


def _is_nvml_blocked(now: object | None = None) -> object:
    if now is None:
        now = float(time.perf_counter())
    with _NVML_LOCK:
        until = float(_NVML_BACKOFF_UNTIL or 0.0)
    return bool(until > 0.0 and float(now) < until)


def _nvml_min_interval_s() -> object:
    global _NVML_MIN_INTERVAL_S
    if _NVML_MIN_INTERVAL_S is not None:
        return float(_NVML_MIN_INTERVAL_S)
    raw = env_first(("STNET_NVML_MIN_INTERVAL_S", "STNET_NVML_MIN_INTERVAL"))
    if raw is not None:
        with contextlib.suppress(Exception):
            _NVML_MIN_INTERVAL_S = max(0.0, float(raw))
            return float(_NVML_MIN_INTERVAL_S)
    try:
        from ..core.system import Thread
        _NVML_MIN_INTERVAL_S = 0.1 if bool(Thread.is_optimized_for_no_gil()) else 0.0
    except Exception:
        _NVML_MIN_INTERVAL_S = 0.0
    return float(_NVML_MIN_INTERVAL_S)


def _is_nvml_available() -> object:
    global _nvml, _NVML_READY, _NVML_TRIED
    if _is_nvml_blocked():
        return False
    if _is_nvml_disabled():
        _NVML_TRIED = True
        _NVML_READY = False
        _nvml = None
        return False
    if _NVML_READY:
        return True
    if _NVML_TRIED:
        return False
    with _NVML_LOCK:
        if _is_nvml_blocked():
            return False
        if _NVML_READY:
            return True
        if _NVML_TRIED:
            return False
        _NVML_TRIED = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                import pynvml
            _nvml = pynvml
            _nvml.nvmlInit()
            _NVML_READY = True
        except Exception as exc:
            _nvml = None
            _NVML_READY = False
            if env_bool("STNET_DEBUG", False):
                logging.getLogger(__name__).debug("NVML init failed: %s", exc, exc_info=True)
    return bool(_NVML_READY)


def _validate_compile_safe() -> None:
    global _COMPILE_SAFE_DONE
    if _COMPILE_SAFE_DONE:
        return
    with _COMPILE_SAFE_LOCK:
        if _COMPILE_SAFE_DONE:
            return
        _COMPILE_SAFE_DONE = True
    with contextlib.suppress(Exception):
        compile_safe(runtime_module=sys.modules[__name__])


@lru_cache(maxsize=8)
def _is_clock_synchronized(dev_type: str) -> bool:
    if _env_flag("STNET_TIMER_SYNC", False) or _env_flag("STNET_WALLCLOCK_TIMER_SYNC", False):
        return True
    dt = str(dev_type or "cpu")
    if dt == "mps":
        return bool(
            _env_flag("STNET_MPS_TIMER_SYNC", _env_flag("STNET_MPS_SYNC_TIMER", False))
        )
    if dt == "xpu":
        return bool(_env_flag("STNET_XPU_TIMER_SYNC", False))
    if dt == "cuda":
        return bool(_env_flag("STNET_CUDA_TIMER_SYNC", False))
    return False


def _is_event_timer_available(device: torch.device) -> bool:
    try:
        return bool(is_accelerator_timer_supported(str(getattr(device, "type", "cpu"))))
    except Exception:
        return False


def _new_event_timer(device: torch.device) -> object:
    try:
        dev_type = str(getattr(device, "type", "cpu"))
    except Exception:
        dev_type = "cpu"
    if not is_accelerator_timer_supported(dev_type):
        return None
    return new_accelerator_event(device, enable_timing=True)


def _get_thread_events(device: torch.device, slot: str) -> object:
    try:
        dev_type = str(getattr(device, "type", "cpu"))
    except Exception:
        dev_type = "cpu"
    if not is_accelerator_timer_supported(dev_type):
        return None
    tls = _TIMING_EVENT_TLS
    d = getattr(tls, "events", None)
    if d is None:
        d = {}
        setattr(tls, "events", d)
    try:
        dev_idx = int(device.index) if device.index is not None else -1
    except Exception:
        dev_idx = -1
    key = (str(slot), str(dev_type), int(dev_idx))
    cached = d.get(key, _TIMING_EVENTS_UNSUPPORTED)
    if cached is _TIMING_EVENTS_UNSUPPORTED:
        ev_s = _new_event_timer(device)
        ev_e = _new_event_timer(device)
        if ev_s is None or ev_e is None:
            d[key] = None
            return None
        cached = (ev_s, ev_e)
        d[key] = cached
    return cached


@lru_cache(maxsize=256)
def _env_flag(name: object, default: object) -> object:
    return env_bool(str(name), bool(default))


@lru_cache(maxsize=256)
def _env_int(name: object, default: object) -> object:
    return env_int(str(name), int(default))


@lru_cache(maxsize=256)
def _env_float(name: object, default: object) -> object:
    return env_float(str(name), float(default))


def _get_torch_profiler(
    *args: Any,
    enabled: object,
    tag: object,
    device: TorchDeviceLike,
    out_dir: PathLike,
    rank: int = 0,
) -> object:
    if not bool(enabled):
        return None
    try:
        import torch.profiler
    except Exception:
        return None
    tp = torch.profiler
    activities = [tp.ProfilerActivity.CPU]
    if device.type == "cuda":
        with contextlib.suppress(Exception):
            activities.append(tp.ProfilerActivity.CUDA)
    elif device.type == "xpu":
        with contextlib.suppress(Exception):
            activities.append(getattr(tp.ProfilerActivity, "XPU"))
    elif device.type == "mps":
        with contextlib.suppress(Exception):
            mps_act = getattr(tp.ProfilerActivity, "MPS", None)
            if mps_act is not None:
                activities.append(mps_act)
    wait = max(0, int(_env_int("STNET_TORCH_PROFILE_WAIT", 0)))
    warmup = max(0, int(_env_int("STNET_TORCH_PROFILE_WARMUP", 2)))
    active = max(
        1,
        int(_env_int("STNET_TORCH_PROFILE_ACTIVE", _env_int("STNET_TORCH_PROFILE_STEPS", 8))),
    )
    repeat = max(1, int(_env_int("STNET_TORCH_PROFILE_REPEAT", 1)))
    record_shapes = bool(_env_flag("STNET_TORCH_PROFILE_RECORD_SHAPES", False))
    profile_memory = bool(_env_flag("STNET_TORCH_PROFILE_PROFILE_MEMORY", True))
    with_stack = bool(_env_flag("STNET_TORCH_PROFILE_WITH_STACK", False))
    with_flops = bool(_env_flag("STNET_TORCH_PROFILE_WITH_FLOPS", False))
    group_by_shape = bool(_env_flag("STNET_TORCH_PROFILE_GROUP_BY_SHAPE", False))
    row_limit = max(5, int(_env_int("STNET_TORCH_PROFILE_TOPK", 40)))
    if not out_dir:
        out_dir = os.path.join(os.getcwd(), "torch_profiler")
    out_dir = os.path.abspath(str(out_dir))
    with contextlib.suppress(Exception):
        os.makedirs(out_dir, exist_ok=True)
    worker_name = f"{str(tag)}-rank{int(rank)}"
    try:
        schedule = tp.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        on_trace = tp.tensorboard_trace_handler(out_dir, worker_name=worker_name)
        prof = tp.profile(
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


def _get_profiler_summary(
    prof: object,
    *args: Any,
    device: TorchDeviceLike,
    logger: logging.Logger,
    header: object,
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
    table = None
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


def _oom_retries(loader: object, phase: object, step: int) -> object:
    key = (int(id(loader)), str(phase), int(step))
    with _OOM_RETRY_LOCK:
        cur = int(_OOM_RETRY_COUNT.get(key, 0)) + 1
        _OOM_RETRY_COUNT[key] = int(cur)
        return int(cur)


def _clear_oom_retries(loader: object, phase: object, step: int) -> None:
    key = (int(id(loader)), str(phase), int(step))
    with _OOM_RETRY_LOCK:
        _OOM_RETRY_COUNT.pop(key, None)


def _oom_max_retries(phase: object) -> object:
    phase = str(phase).strip().lower()
    if phase == "train":
        v = _env_int(
            "STNET_OOM_MAX_RETRIES_TRAIN", _env_int("STNET_OOM_MAX_RETRIES_PER_BATCH", 4)
        )
    elif phase in {"val", "valid", "validation"}:
        v = _env_int(
            "STNET_OOM_MAX_RETRIES_VAL", _env_int("STNET_OOM_MAX_RETRIES_PER_BATCH", 2)
        )
    else:
        v = _env_int("STNET_OOM_MAX_RETRIES_PER_BATCH", 3)
    return max(0, int(v))


def _is_batch_skippable(phase: object) -> bool:
    phase = str(phase).strip().lower()
    if phase == "train":
        return _env_flag("STNET_OOM_SKIP_TRAIN", _env_flag("STNET_OOM_SKIP_BATCH", True))
    elif phase in {"val", "valid", "validation"}:
        return _env_flag("STNET_OOM_SKIP_VAL", _env_flag("STNET_OOM_SKIP_BATCH", True))
    return _env_flag("STNET_OOM_SKIP_BATCH", True)


def _get_scale_rate_down(attempt: object) -> object:
    seq = (0.8, 0.7, 0.6, 0.5)
    i = max(1, int(attempt)) - 1
    if i < 0:
        i = 0
    if i >= len(seq):
        i = len(seq) - 1
    return float(seq[i])


def _is_scale_rate_logged(
    *args: Any,
    logger: logging.Logger,
    scale_ctl: object,
    tag: object,
    msg: object,
    level: str = "info",
    min_interval_s: float | None = None,
) -> None:
    if min_interval_s is None:
        min_interval_s = _env_float("STNET_SAMPLER_SCALE_LOG_MIN_INTERVAL_S", 5.0)
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
        if min_interval_s and now - last < float(min_interval_s):
            return
        _SAMPLER_SCALE_LOG_LAST_S[key] = float(now)
    try:
        if str(level).lower() == "debug":
            logger.debug(msg)
        else:
            logger.info(msg)
    except Exception:
        pass


def _get_sampler_scaler(loader: object, *args: Any, max_depth: int = 4) -> object:
    obj = loader
    try:
        depth = max(1, int(max_depth))
    except Exception:
        depth = 4
    for _ in range(depth):
        if obj is None:
            break
        ctl = getattr(obj, "_stnet_sampler_scale", None)
        if ctl is not None:
            return ctl
        obj = getattr(obj, "_src", None) or getattr(obj, "src", None)
    return None


def _get_oom_blocking_time(oom_try: int, phase: str | None = None) -> float:
    try:
        base_ms = float(_env_float("STNET_OOM_BACKOFF_BASE_MS", 0.0))
    except Exception:
        base_ms = 0.0
    if base_ms <= 0.0:
        return 0.0
    try:
        max_ms = float(_env_float("STNET_OOM_BACKOFF_MAX_MS", 50.0))
    except Exception:
        max_ms = 50.0
    if max_ms < 0.0:
        max_ms = 0.0
    p = max(0, int(oom_try) - 2)
    try:
        sleep_ms = min(float(max_ms), float(base_ms) * (2.0 ** float(p)))
    except Exception:
        sleep_ms = float(base_ms)
    return max(0.0, float(sleep_ms) / 1000.0)


def _recover_oom(
    *args: Any,
    phase: str,
    loader: object,
    step_idx: int,
    device: torch.device,
    model: object,
    optimizer: Any | None = None,
    global_step: int | None = None,
    grad_accum_steps: int | None = None,
    min_grad_accum: int = 1,
) -> tuple[str, int | None]:
    ph = str(phase).strip().lower()
    oom_try = _oom_retries(loader, ph, int(step_idx))
    max_tries = _oom_max_retries(ph)
    if ph == "train":
        if global_step is not None and oom_try <= 1:
            _LOGGER.error(
                "[epochs] OOM during train step %d (global_step=%d). Trying to reduce microbatch / grad_accum and retry same batch. (try=%d/%d)",
                int(step_idx),
                int(global_step),
                int(oom_try),
                int(max_tries),
            )
        elif global_step is not None:
            _LOGGER.warning(
                "[epochs] OOM during train step %d (global_step=%d). Retrying. (try=%d/%d)",
                int(step_idx),
                int(global_step),
                int(oom_try),
                int(max_tries),
            )
        elif oom_try <= 1:
            _LOGGER.error(
                "[epochs] OOM during train step %d. Trying to reduce microbatch / grad_accum and retry same batch. (try=%d/%d)",
                int(step_idx),
                int(oom_try),
                int(max_tries),
            )
        else:
            _LOGGER.warning(
                "[epochs] OOM during train step %d. Retrying. (try=%d/%d)",
                int(step_idx),
                int(oom_try),
                int(max_tries),
            )
    else:
        if oom_try <= 1:
            _LOGGER.error(
                "[epochs] OOM during %s step %d. Trying to reduce microbatch and retry same batch. (try=%d/%d)",
                str(ph),
                int(step_idx),
                int(oom_try),
                int(max_tries),
            )
        else:
            _LOGGER.warning(
                "[epochs] OOM during %s step %d. Retrying. (try=%d/%d)",
                str(ph),
                int(step_idx),
                int(oom_try),
                int(max_tries),
            )
    if max_tries > 0 and oom_try > max_tries:
        if _is_batch_skippable(ph):
            _LOGGER.error(
                "[epochs] OOM storm: exceeded retry budget (try=%d/%d) at %s step %d. Skipping this batch.",
                int(oom_try),
                int(max_tries),
                str(ph),
                int(step_idx),
            )
            with contextlib.suppress(Exception):
                _clear_oom_retries(loader, ph, int(step_idx))
            with contextlib.suppress(Exception):
                empty_device_cache(device=device, do_gc=False, min_interval_s=0.0)
            if optimizer is not None:
                with contextlib.suppress(Exception):
                    optimizer.zero_grad(set_to_none=True)
            return ("skip", grad_accum_steps)
        return ("raise", grad_accum_steps)
    try:
        scale_ctl = _get_sampler_scaler(loader)
        if scale_ctl is not None:
            prev = None
            with contextlib.suppress(Exception):
                prev = float(scale_ctl.get())
            with contextlib.suppress(Exception):
                scale_ctl.request_scale_down(_get_scale_rate_down(oom_try))
            with contextlib.suppress(Exception):
                cur = float(scale_ctl.get())
                if prev is not None and cur < prev:
                    _is_scale_rate_logged(
                        logger=_LOGGER,
                        scale_ctl=scale_ctl,
                        tag=f"oom-{ph}-scale-down",
                        msg="[epochs] reduced sampler scale from %.4f to %.4f after OOM (factor=%.2f, try=%d/%d)"
                        % (prev, cur, _get_scale_rate_down(oom_try), oom_try, max_tries),
                        level="info",
                    )
    except Exception:
        pass
    with contextlib.suppress(Exception):
        ec_min = (
            0.0 if oom_try <= 1 else _env_float("STNET_OOM_EMPTY_CACHE_MIN_INTERVAL_S", 0.05)
        )
        empty_device_cache(device=device, do_gc=False, min_interval_s=ec_min)
    if optimizer is not None:
        with contextlib.suppress(Exception):
            optimizer.zero_grad(set_to_none=True)

    inst_pressure = _to_submodule(model) or (model.module if hasattr(model, "module") else model)
    if inst_pressure is not None:
        cur_step_total = int(getattr(inst_pressure, "_stnet_step_total", 0) or 0)
        if int(oom_try) <= 1:
            applied = _to_checkpoint(
                model,
                device=device,
                step_total=cur_step_total,
                ttl_steps=64,
                min_bytes=0,
            )
            if applied:
                sleep_s = _get_oom_blocking_time(oom_try, ph)
                if sleep_s > 0.0:
                    with contextlib.suppress(Exception):
                        time.sleep(float(sleep_s))
                return ("retry", grad_accum_steps)

    reduced_any = False
    inst = _to_submodule(model)
    if inst is not None:
        cur_mb = 0
        with contextlib.suppress(Exception):
            cur_mb = int(getattr(inst, "microbatch", 0) or 0)
        if cur_mb > 1:
            new_mb = max(1, cur_mb // 2)
            try:
                new_mb = _sync_tensor(new_mb, device=device, src=0)
            except Exception:
                pass
            if new_mb < cur_mb:
                with contextlib.suppress(Exception):
                    inst.microbatch = int(new_mb)
                    inst._auto_microbatch_pending = False
                _LOGGER.info(
                    "[epochs] reduced Model.microbatch from %d to %d after OOM%s",
                    int(cur_mb),
                    int(new_mb),
                    " in validation" if ph != "train" else "",
                )
                reduced_any = True
    if ph == "train" and grad_accum_steps is not None:
        try:
            cur_ga = int(grad_accum_steps)
        except Exception:
            cur_ga = int(grad_accum_steps or 1)
        if cur_ga > int(min_grad_accum):
            new_ga = max(int(min_grad_accum), cur_ga // 2)
            try:
                new_ga = _sync_tensor(new_ga, device=device, src=0)
            except Exception:
                pass
            if int(new_ga) != int(cur_ga):
                _LOGGER.info(
                    "[epochs] reduced grad_accum_steps from %d to %d after OOM",
                    int(cur_ga),
                    int(new_ga),
                )
                grad_accum_steps = int(new_ga)
                reduced_any = True
    if not reduced_any:
        if _is_batch_skippable(ph):
            _LOGGER.error(
                "[epochs] OOM in %s and no more knobs to reduce. Skipping this batch.", str(ph)
            )
            with contextlib.suppress(Exception):
                _clear_oom_retries(loader, ph, int(step_idx))
            with contextlib.suppress(Exception):
                empty_device_cache(device=device, do_gc=False, min_interval_s=0.0)
            if optimizer is not None:
                with contextlib.suppress(Exception):
                    optimizer.zero_grad(set_to_none=True)
            return ("skip", grad_accum_steps)
        _LOGGER.error(
            "[epochs] OOM in %s and no more knobs to reduce; giving up on recovery.", str(ph)
        )
        return ("raise", grad_accum_steps)
    sleep_s = _get_oom_blocking_time(oom_try, ph)
    if sleep_s > 0.0:
        with contextlib.suppress(Exception):
            time.sleep(float(sleep_s))
    return ("retry", grad_accum_steps)


def _get_batch_length(loader: object) -> object:
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


def _float8_log(msg: str, *args: Any, only_main_rank: bool = True, **kwargs: Any) -> None:
    try:
        if (
            only_main_rank
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            if torch.distributed.get_rank() != 0:
                return
    except Exception:
        pass
    _LOGGER.info(msg, *args)


def _validate_no_meta_tensors(module: object) -> None:
    hits = []
    for name, param in module.named_parameters(recurse=True):
        if is_meta_or_fake_tensor(param):
            hits.append(f"param {name} shape={tuple(param.shape)}")
    for name, buffer in module.named_buffers(recurse=True):
        if is_meta_or_fake_tensor(buffer):
            hits.append(f"buffer {name} shape={tuple(buffer.shape)}")
    if hits:
        raise RuntimeError("Found meta tensors in model:\n" + "\n".join(hits))


def _hook_meta_monitor(module: object, inputs: object, warn_only: object) -> None:
    for arg in inputs:
        if isinstance(arg, torch.Tensor) and is_meta_or_fake_tensor(arg):
            message = f"[META] {module.__class__.__name__} got meta input"
            if warn_only:
                warnings.warn(message, stacklevel=3)
                return
            raise RuntimeError(message)


def _enable_meta_monitor(model: object) -> None:
    hook_mode = (
        str(env_first(("STNET_META_MONITOR", "STNET_META_HOOK"), default="off") or "off")
        .strip()
        .lower()
    )
    if hook_mode in {"0", "", "false", "off"}:
        return
    warn_only = hook_mode in {"warn", "warning"}
    for submodule in model.modules():
        submodule.register_forward_pre_hook(
            partial(_hook_meta_monitor, warn_only=warn_only), with_kwargs=False
        )


def _validate_no_fake_dtensor(root: nn.Module, *args: Any, **kwargs: Any) -> None:
    bad = []
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
            "LayerNorm parameters must be materialized as a real Tensor: " + ", ".join(bad)
        )


def _set_requires_grad(
    module: nn.Module,
    name: str,
    data: torch.Tensor,
    *args: Any,
    requires_grad: bool,
) -> None:
    setattr(module, name, nn.Parameter(data, requires_grad=requires_grad))


def _is_precision_exempted(module: object) -> bool:
    return bool(getattr(module, "__stnet_precision_exempt__", False))


def _cast_float_dtype(model: object, dtype: torch.dtype) -> object:
    if not isinstance(dtype, torch.dtype):
        return
    try:
        if not torch.is_floating_point(torch.empty((), dtype=dtype)):
            return
    except Exception:
        return

    with torch.no_grad():
        for mod in getattr(model, "modules", lambda: [])():
            if _is_precision_exempted(mod):
                continue
            params = getattr(mod, "_parameters", None)
            if params:
                for name, p in params.items():
                    if p is None or not isinstance(p, torch.Tensor):
                        continue
                    if not p.is_floating_point() or p.dtype == dtype:
                        continue
                    params[name] = torch.nn.Parameter(
                        p.detach().to(dtype), requires_grad=bool(getattr(p, "requires_grad", True))
                    )
            bufs = getattr(mod, "_buffers", None)
            if bufs:
                for name, b in bufs.items():
                    if b is None or not isinstance(b, torch.Tensor):
                        continue
                    if not b.is_floating_point() or b.dtype == dtype:
                        continue
                    bufs[name] = b.detach().to(dtype)


def _to_device_recursive(obj: object, dev: object) -> object:
    if isinstance(obj, torch.Tensor):
        dev_type = getattr(dev, "type", None)
        non_blocking = bool(dev_type and is_pin_supported(str(dev_type)))
        try:
            return obj.to(device=dev, non_blocking=non_blocking)
        except TypeError:
            return obj.to(device=dev)
    if isinstance(obj, TensorDictBase):
        return obj.to(device=dev)
    if isinstance(obj, Mapping):
        return {k: _to_device_recursive(v, dev) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_to_device_recursive(v, dev) for v in obj]
        return type(obj)(seq)
    return obj


def _touch_tensors(obj: object) -> None:
    if isinstance(obj, torch.Tensor):
        _ = obj.sum()
        return
    if isinstance(obj, TensorDictBase):
        for v in obj.values():
            _touch_tensors(v)
        return
    if isinstance(obj, Mapping):
        for v in obj.values():
            _touch_tensors(v)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _touch_tensors(v)
        return


def _cast_batchnorm_buffers_dtype(module: object, dtype: torch.dtype | None) -> None:
    if dtype is None:
        return
    if not isinstance(dtype, torch.dtype):
        return
    with torch.no_grad():
        for mod in getattr(module, "modules", lambda: [])():
            if _is_precision_exempted(mod):
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
                for name, buf in getattr(mod, "_buffers", {}).items():
                    if buf is None or not isinstance(buf, torch.Tensor):
                        continue
                    if not buf.is_floating_point() or buf.dtype == dtype:
                        continue
                    with contextlib.suppress(Exception):
                        mod._buffers[name] = buf.to(dtype=dtype)


def _get_layernorm_dtype(device: TorchDeviceLike) -> object:
    try:
        meta = Autocast.coerce_metadata(device)
        cands = tuple(getattr(meta, "float_dtypes", ())) if meta is not None else ()
        if not cands:
            cands = (torch.float32,)
        chosen = Autocast.negotiate(
            tuple(cands), fallback=torch.float64, context="cpu.layernorm", device=device, meta=meta
        )
        return torch.float64 if chosen == torch.float64 else torch.float32
    except Exception:
        return torch.float32


def _preload_layers(model: object, device: TorchDeviceLike) -> None:
    for module in model.modules():
        if not isinstance(module, nn.LayerNorm):
            continue
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        requires_grad_w = bool(getattr(weight, "requires_grad", True))
        requires_grad_b = bool(getattr(bias, "requires_grad", True))
        if device.type == "cpu":
            target_dtype = _get_layernorm_dtype(device)
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
            if not isinstance(weight, torch.Tensor) or is_meta_or_fake_tensor(weight):
                data = torch.ones(module.normalized_shape, device=device, dtype=target_dtype)
                _set_requires_grad(module, "weight", data, requires_grad=requires_grad_w)
                weight = module.weight
            if not isinstance(bias, torch.Tensor) or is_meta_or_fake_tensor(bias):
                data = torch.zeros(module.normalized_shape, device=device, dtype=target_dtype)
                _set_requires_grad(module, "bias", data, requires_grad=requires_grad_b)
                bias = module.bias
        if device.type == "cpu":
            if isinstance(weight, torch.Tensor) and weight.dtype != target_dtype:
                data = weight.to(device=device, dtype=target_dtype)
                _set_requires_grad(module, "weight", data, requires_grad=requires_grad_w)
                weight = module.weight
            if isinstance(bias, torch.Tensor) and bias.dtype != target_dtype:
                data = bias.to(device=device, dtype=target_dtype)
                _set_requires_grad(module, "bias", data, requires_grad=requires_grad_b)
                bias = module.bias
        elif (
            isinstance(weight, torch.Tensor)
            and isinstance(bias, torch.Tensor)
            and weight.is_floating_point()
            and bias.is_floating_point()
            and (bias.dtype != weight.dtype)
        ):
            data = bias.to(device=device, dtype=weight.dtype)
            _set_requires_grad(module, "bias", data, requires_grad=requires_grad_b)
            bias = module.bias


def _validate_model_dtype_unity(model: object, device: TorchDeviceLike) -> None:
    mismatches = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.LayerNorm):
            continue
        tensors = [
            ("weight", getattr(module, "weight", None)),
            ("bias", getattr(module, "bias", None)),
        ]
        expected = None
        if device.type == "cpu":
            expected = _get_layernorm_dtype(device)
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
                mismatches.append(f"{module_name} parameters disagree on dtype: {sorted(dtypes)}")
    if mismatches:
        raise RuntimeError("LayerNorm parameter dtype mismatch detected:\n" + "\n".join(mismatches))


def _coerce_dcp_keys(state: object) -> object:
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
            state[key] = _coerce_dcp_keys(value)
        for key in keys:
            state.pop(key, None)
    return state


def _get_backend_type(device: TorchDeviceLike) -> object:
    match str(getattr(device, "type", "cpu")):
        case "cuda":
            return "nccl"
        case "xpu":
            return "xccl"
        case _:
            return "gloo"


def _init_backend(device: TorchDeviceLike) -> None:
    with contextlib.suppress(Exception):
        if device.type == "cuda" and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
    rank = int(_env_int("LOCAL_RANK", 0))
    if device.type in {"cuda", "xpu"}:
        n = max(1, int(get_num_accelerators(device.type) or 1))
        set_accelerator_index(device.type, int(rank) % int(n))
    else:
        iface = None
        gloo_if = os.environ.get("GLOO_SOCKET_IFNAME")
        tp_if = os.environ.get("TP_SOCKET_IFNAME")
        if gloo_if or tp_if:
            if gloo_if and (not tp_if):
                os.environ.setdefault("TP_SOCKET_IFNAME", str(gloo_if))
            elif tp_if and (not gloo_if):
                os.environ.setdefault("GLOO_SOCKET_IFNAME", str(tp_if))
            return
        try:
            with open("/proc/net/route", "r", encoding="utf-8") as f:
                for line in f.readlines()[1:]:
                    fields = line.strip().split()
                    if len(fields) >= 2 and fields[1] == "00000000":
                        iface = fields[0]
                        if iface:
                            break
        except Exception:
            iface = None
        if iface is None and psutil is not None:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    s.connect(("8.8.8.8", 80))
                    ip = s.getsockname()[0]
                finally:
                    s.close()
                if ip:
                    for name, addrs in psutil.net_if_addrs().items():
                        for a in addrs:
                            if (
                                getattr(a, "family", None) == socket.AF_INET
                                and getattr(a, "address", None) == ip
                            ):
                                iface = str(name)
                                break
                        if iface:
                            break
            except Exception:
                iface = None
        if iface:
            os.environ.setdefault("GLOO_SOCKET_IFNAME", iface)
            os.environ.setdefault("TP_SOCKET_IFNAME", iface)


def _unify_model_dtype(model: object, prefer: object | None = None) -> object:
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
            new_p = torch.nn.Parameter(p.detach().to(tgt), requires_grad=p.requires_grad)
            setattr(mod, name, new_p)
    return tgt


def _get_source_path(obj: object) -> object:
    if isinstance(obj, dict):
        if "path" in obj and "kind" in obj:
            return os.fspath(obj["path"])
        if obj:
            first = next(iter(obj.values()))
            return _get_source_path(first)
    if isinstance(obj, (list, tuple)) and obj:
        return _get_source_path(obj[0])
    raise RuntimeError("sources is empty or invalid")


def _get_sample_size(
    model: object,
    device: TorchDeviceLike,
    ops: object,
    dataset: object | None = None,
    max_probe_batch: int = 32,
    with_backward: bool = False,
    global_loss: object | None = None,
    local_loss: object | None = None,
    loss_weights: object | None = None,
) -> object:
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
    floor_bytes = int((in_dim + out_dim) * elem_size * 10240) if in_dim + out_dim > 0 else 0
    dev_type = getattr(device, "type", "")
    if dev_type not in {"cuda", "xpu", "mps"}:
        return
    try:
        memmap_root = _get_source_path(ops.sources)
        ds = Sampler(
            memmap_root, split="train", val_frac=float(getattr(ops, "val_frac", 0.0) or 0.0)
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

    try:
        base_alloc = allocated_accelerator_memory(device)
        if base_alloc is None:
            return
        flush_accelerator_memory_stats(device)
        batch = ds.get(0, B0)
        forward_ran = False
        training_mode = bool(model.training)
        meta = dataset if isinstance(dataset, Dataset) else Dataset.for_device(device)
        try:
            from ..core.graph import inference_mode
            from ..core.precision import Autocast
            feats, labels, *_rest = meta.preprocess(batch, return_keys=False)
            X = to_torch_tensor(feats)
            X = torch.atleast_2d(X)
            if X.dim() == 2 and int(X.shape[1]) == int(getattr(ops, "in_dim", X.shape[1])):
                X = X.to(
                    device=device, non_blocking=is_pin_supported(device.type)
                )
                if with_backward:
                    model.train()
                    with Autocast.float(device):
                        Y_flat = None
                        if labels is not None:
                            Y = to_torch_tensor(labels)
                            Y = torch.atleast_2d(Y).to(
                                device=device,
                                non_blocking=is_pin_supported(device.type),
                            )
                            Y_flat = Y.reshape(Y.shape[0], -1)
                        y_hat, loss_val = model(
                            X,
                            labels_flat=Y_flat,
                            global_loss=global_loss,
                            local_loss=local_loss,
                            loss_weights=loss_weights,
                            calibrate_output=False,
                        )
                    target = None
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
                        warmup_iters = int(_env_int("STNET_SERVE_WARMUP_ITERS", 0) or 0)
                        if (
                            warmup_iters <= 0
                            and str(getattr(ops, "mode", "") or "") in ("predict", "infer")
                            and _env_flag("STNET_MAX_PERF", True)
                        ):
                            m_eval = model.module if hasattr(model, "module") else model
                            compiled = getattr(m_eval, "_compiled_submodules", None)
                            warmup_iters = (
                                3
                                if (isinstance(compiled, dict) and any(bool(v) for v in compiled.values()))
                                else 1
                            )
                        warmup_iters = max(1, min(16, int(warmup_iters)))
                        for _i in range(warmup_iters):
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
            batch_dev = _to_device_recursive(batch, device)

            _touch_tensors(batch_dev)
        with contextlib.suppress(Exception):
            sync_accelerator(device)
        peak_alloc = accelerator_max_allocated_memory(device)
        if peak_alloc is None:
            peak_alloc = allocated_accelerator_memory(device)
        if peak_alloc is None:
            return
        delta = max(0, int(peak_alloc) - int(base_alloc))
        if delta <= 0:
            return
        per_sample = int(delta // max(B0, 1))
        if floor_bytes > 0:
            per_sample = max(per_sample, floor_bytes)
        margin = 1.5 if with_backward else 1.2
        per_sample = int(per_sample * float(margin))
        if per_sample <= 0:
            return
        with contextlib.suppress(Exception):
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                t = torch.tensor([int(per_sample)], device=device, dtype=torch.long)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
                per_sample = int(t.item())
        try:
            Sampler._per_sample_mem_bytes = int(per_sample)
        except Exception:
            pass
        with contextlib.suppress(Exception):
            os.environ["STNET_PER_SAMPLE_MEM_BYTES"] = str(int(per_sample))
    except Exception:
        return


def _get_fsdp_module(
    target: PathLike,
    mesh: object,
    mp_policy: object,
    reshard_after_forward: object,
    wrapped: object,
) -> object:
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


def _get_layers(root: object) -> object:
    if root is None:
        return []
    blocks = []
    seen = set()
    for module in root.modules():
        block_list = getattr(module, "blocks", None)
        if isinstance(block_list, torch.nn.ModuleList):
            for block in block_list:
                if isinstance(block, torch.nn.Module) and id(block) not in seen:
                    seen.add(id(block))
                    blocks.append(block)
    return blocks


def _init_tensor(
    value: object,
    *args: Any,
    param: torch.Tensor,
    capturable: bool,
    fused: bool,
    **kwargs: Any,
) -> torch.Tensor:
    desired_device = param.device if capturable or fused else torch.device("cpu")
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


def _init_optimizer(optim: object) -> None:
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
            state["step"] = _init_tensor(
                step_value, param=param, capturable=capturable, fused=fused
            )
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(param)
            if "exp_avg_sq" not in state:
                state["exp_avg_sq"] = torch.zeros_like(param)
            if amsgrad and "max_exp_avg_sq" not in state:
                state["max_exp_avg_sq"] = torch.zeros_like(param)
            optim.state[param] = state


def _schedule(
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
    return frac_min + (1.0 - frac_min) * 0.5 * (1.0 + math.cos(math.pi * t / max(1, main_steps)))


def _init_distributed_group(backend: object, device: TorchDeviceLike, local_rank: int) -> None:
    dev_id = None
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


def _gpu_nvml_utils(device: TorchDeviceLike) -> object:
    if getattr(device, "type", "") != "cuda":
        return (None, None)
    idx = device.index if device.index is not None else get_accelerator_index("cuda")
    idx_i = int(idx)
    global _NVML_FAIL_COUNT, _NVML_BACKOFF_UNTIL
    gpu_util = None
    mem_util = None
    if _is_nvml_available() and _nvml is not None:
        min_interval = float(_nvml_min_interval_s())
        now = float(time.perf_counter())
        if _is_nvml_blocked(now):
            return (None, None)
        if min_interval > 0.0:
            cached = _NVML_UTIL_CACHE.get(idx_i)
            if cached is not None:
                ts, cg, cm = cached
                if now - float(ts) < min_interval:
                    return (cg, cm)
        with _NVML_QUERY_LOCK:
            if _is_nvml_blocked(now):
                return (None, None)
            if min_interval > 0.0:
                cached = _NVML_UTIL_CACHE.get(idx_i)
                if cached is not None:
                    ts, cg, cm = cached
                    if now - float(ts) < min_interval:
                        return (cg, cm)
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
                with _NVML_LOCK:
                    _NVML_FAIL_COUNT = 0
                    _NVML_BACKOFF_UNTIL = 0.0
            except Exception:
                with contextlib.suppress(Exception):
                    _NVML_HANDLE_CACHE.pop(idx_i, None)
                with contextlib.suppress(Exception):
                    _NVML_UTIL_CACHE.pop(idx_i, None)
                fail_max = int(_nvml_max_retry())
                backoff_s = float(_nvml_blocking_interval())
                trigger_backoff = False
                with _NVML_LOCK:
                    _NVML_FAIL_COUNT = int(_NVML_FAIL_COUNT) + 1
                    if backoff_s > 0.0 and int(_NVML_FAIL_COUNT) >= int(fail_max):
                        _NVML_BACKOFF_UNTIL = float(time.perf_counter()) + float(backoff_s)
                        _NVML_FAIL_COUNT = 0
                        trigger_backoff = True
                if trigger_backoff:
                    with contextlib.suppress(Exception):
                        _NVML_HANDLE_CACHE.clear()
                    with contextlib.suppress(Exception):
                        _NVML_UTIL_CACHE.clear()
                    with contextlib.suppress(Exception):
                        _LOGGER.warning(
                            "[NVML] repeated failures; backing off NVML queries for %.1fs (override: STNET_NVML_BACKOFF_S, STNET_NVML_FAIL_MAX).",
                            float(backoff_s),
                        )
                gpu_util = None
                mem_util = None
            if gpu_util is not None or mem_util is not None:
                _NVML_UTIL_CACHE[idx_i] = (now, gpu_util, mem_util)
    if mem_util is None:
        with contextlib.suppress(Exception):
            mem_util = available_device_memory(torch.device("cuda", idx_i))
    return (gpu_util, mem_util)


def _xpu_mem_util(device: TorchDeviceLike) -> object:
    if getattr(device, "type", "") != "xpu":
        return None
    with contextlib.suppress(Exception):
        return available_device_memory(device)
    return None


def _mps_mem_util(device: TorchDeviceLike) -> object:
    if getattr(device, "type", "") != "mps":
        return None
    with contextlib.suppress(Exception):
        return available_device_memory(device)
    return None


def _sync_tensor(value: object, device: TorchDeviceLike, src: int = 0) -> object:
    if not is_distributed():
        return int(value)
    try:
        tensor = torch.tensor([int(value)], device=device, dtype=torch.int32)
        torch.distributed.broadcast(tensor, src=src)
        return int(tensor.item())
    except Exception:
        return int(value)


def _get_cpu_load() -> object:
    if psutil is None:
        return None
    try:
        return float(psutil.cpu_percent(interval=0.0))
    except Exception:
        return None


def _pool_tensor(
    tensor: torch.Tensor,
    *args: Any,
    dtype: torch.dtype,
    device: torch.device,
    cpu_pool: Pool | None,
    dev_type: str | None = None,
    pinned_ok: bool | None = None,
) -> tuple[torch.Tensor, Pool.Token | None, bool]:
    if not torch.is_tensor(tensor):
        raise TypeError(f"stage_tensor expects a torch.Tensor, got {type(tensor)}")
    if dev_type is None:
        dev_type = str(getattr(device, "type", "cpu"))
    if pinned_ok is None:
        pinned_ok = bool(is_pin_supported(dev_type))
    if tensor.device.type != "cpu":
        if tensor.dtype != dtype:
            with contextlib.suppress(Exception):
                tensor = tensor.to(dtype=dtype, copy=False)
        return tensor, None, False
    with contextlib.suppress(Exception):
        is_pinned = getattr(tensor, "is_pinned", None)
        if callable(is_pinned) and bool(is_pinned()) and tensor.dtype == dtype:
            return tensor, None, True
    if cpu_pool is not None and bool(pinned_ok):
        buf, token = cpu_pool.get(tuple(tensor.shape), dtype, return_handle=True)
        buf.copy_(tensor, non_blocking=False)
        pinned = False
        with contextlib.suppress(Exception):
            is_pinned = getattr(buf, "is_pinned", None)
            if callable(is_pinned):
                pinned = bool(is_pinned())
        return buf, token, pinned
    out = tensor
    if out.dtype != dtype:
        out = out.to(dtype=dtype, copy=False)
    pinned = False
    with contextlib.suppress(Exception):
        is_pinned = getattr(out, "is_pinned", None)
        if callable(is_pinned):
            pinned = bool(is_pinned())
    return out, None, pinned


def _stream_tensor(
    tensor: object,
    *args: Any,
    device: TorchDeviceLike,
    cpu_pool: object,
    handle: Pool.Token | None = None,
    pinned: bool | None = None,
    dev_type: object | None = None,
    non_blocking_ok: object | None = None,
    backend: object | None = None,
    stream_fn: object | None = None,
    Event: object | None = None,
    fence_event_factory: object | None = None,
    can_stream_release: object | None = None,
) -> object:
    if not torch.is_tensor(tensor):
        return tensor
    if dev_type is None:
        dev_type = str(getattr(device, "type", "cpu"))
    if non_blocking_ok is None:
        non_blocking_ok = bool(dev_type in ("cuda", "xpu"))
    pinned_ok = bool(is_pin_supported(dev_type))
    if pinned is None:
        pinned = False
        with contextlib.suppress(Exception):
            is_pinned = getattr(tensor, "is_pinned", None)
            if callable(is_pinned):
                pinned = bool(is_pinned())
    if tensor.device.type != "cpu" or (not bool(non_blocking_ok)):
        out = tensor.to(device, non_blocking=bool(non_blocking_ok))
        if handle is not None and cpu_pool is not None:
            with contextlib.suppress(Exception):
                cpu_pool.release(handle)
        return out
    if handle is None:
        return tensor.to(device, non_blocking=bool(non_blocking_ok and pinned and pinned_ok))
    if backend is None:
        backend = accelerator_type(dev_type)
    if stream_fn is None and backend is not None:
        stream_fn = getattr(backend, "current_stream", None)
    if Event is None and backend is not None:
        Event = getattr(backend, "Event", None)
    if can_stream_release is None:
        can_stream_release = bool(
            pinned and pinned_ok and callable(stream_fn) and (Event is not None)
        )
    if (not bool(pinned)) or (not bool(can_stream_release)):
        out = tensor.to(device, non_blocking=False)
        if cpu_pool is not None:
            with contextlib.suppress(Exception):
                cpu_pool.release(handle)
        return out
    stream = None
    if callable(stream_fn):
        with contextlib.suppress(Exception):
            try:
                stream = stream_fn(device=device)
            except TypeError:
                try:
                    stream = stream_fn(device)
                except TypeError:
                    stream = stream_fn()
    try:
        if stream is not None:
            with accelerator_stream(stream, dev_type):
                out = tensor.to(device, non_blocking=True)
        else:
            out = tensor.to(device, non_blocking=True)
        if stream is not None:
            rec = getattr(tensor, "record_stream", None)
            if callable(rec):
                with contextlib.suppress(Exception):
                    rec(stream)
        if cpu_pool is not None:
            try:
                evt = None
                fe = getattr(cpu_pool, "fence_event", None)
                if callable(fe) and fence_event_factory is not None:
                    with contextlib.suppress(Exception):
                        evt = fe(handle, fence_event_factory)
                if evt is None:
                    if fence_event_factory is not None:
                        with contextlib.suppress(Exception):
                            evt = fence_event_factory()
                    elif Event is not None:
                        with contextlib.suppress(Exception):
                            evt = Event()
                if evt is not None:
                    if stream is not None:
                        try:
                            evt.record(stream)
                        except TypeError:
                            evt.record()
                    else:
                        evt.record()
                    cpu_pool.release_after(handle, evt)
                else:
                    with contextlib.suppress(Exception):
                        sync_accelerator(device)
                    with contextlib.suppress(Exception):
                        cpu_pool.release(handle)
            except Exception:
                with contextlib.suppress(Exception):
                    sync_accelerator(device)
                with contextlib.suppress(Exception):
                    cpu_pool.release(handle)
        return out
    except Exception:
        out = tensor.to(device, non_blocking=False)
        if cpu_pool is not None:
            with contextlib.suppress(Exception):
                cpu_pool.release(handle)
        return out


def _move_staged_pair_to_device(
    X_st: object,
    x_tok: Pool.Token | None,
    x_pinned: bool,
    Y_st: object | None,
    y_tok: Pool.Token | None,
    y_pinned: bool,
    to_device: object,
) -> tuple[object, object | None, Pool.Token | None, Pool.Token | None]:
    X_dev = to_device(X_st, handle=x_tok, pinned=x_pinned)
    x_tok = None
    Y_dev = None
    if Y_st is not None:
        Y_dev = to_device(Y_st, handle=y_tok, pinned=y_pinned)
        y_tok = None
    return X_dev, Y_dev, x_tok, y_tok


def _pin(
    meta: object,
    raw: object,
    *args: Any,
    device: TorchDeviceLike,
    stage_tensor: object,
    to_device: TorchDeviceLike,
    cpu_pool: object,
    use_timer: bool,
    timer_sync: bool = False,
    require_labels: bool = True,
) -> object:
    feat, label, *_ = meta.preprocess(raw, return_keys=False, cast=False)
    X_src = feat if torch.is_tensor(feat) else to_torch_tensor(feat)
    if label is None:
        if require_labels:
            raise RuntimeError("Batch is missing labels.")
        Y_src = None
    else:
        Y_src = label if torch.is_tensor(label) else to_torch_tensor(label)
    x_tok: Pool.Token | None = None
    y_tok: Pool.Token | None = None
    try:
        x_dtype = getattr(meta, "feature_dtype", X_src.dtype)
        X_st, x_tok, x_pinned = stage_tensor(X_src, dtype=x_dtype)
        Y_st = None
        y_pinned = False
        if Y_src is not None:
            y_dtype = getattr(meta, "label_float_dtype", Y_src.dtype)
            Y_st, y_tok, y_pinned = stage_tensor(Y_src, dtype=y_dtype)
        t_ready = time.perf_counter_ns()

        if use_timer:
            pair = _get_thread_events(device, slot="h2d")
            if pair is not None:
                try:
                    h2d_s_ev, h2d_e_ev = pair
                    h2d_s_ev.record()
                    X_dev, Y_dev, x_tok, y_tok = _move_staged_pair_to_device(
                        X_st,
                        x_tok,
                        x_pinned,
                        Y_st,
                        y_tok,
                        y_pinned,
                        to_device=to_device,
                    )
                    h2d_e_ev.record()
                    h2d_e_ev.synchronize()
                    h2d_s = float(h2d_s_ev.elapsed_time(h2d_e_ev)) / 1000.0
                except Exception:
                    t_h2d_s = time.perf_counter_ns()
                    X_dev, Y_dev, x_tok, y_tok = _move_staged_pair_to_device(
                        X_st,
                        x_tok,
                        x_pinned,
                        Y_st,
                        y_tok,
                        y_pinned,
                        to_device=to_device,
                    )
                    if timer_sync:
                        sync_accelerator(device)
                    t_h2d_e = time.perf_counter_ns()
                    h2d_s = (t_h2d_e - t_h2d_s) / 1000000000.0
            else:
                t_h2d_s = time.perf_counter_ns()
                X_dev, Y_dev, x_tok, y_tok = _move_staged_pair_to_device(
                    X_st,
                    x_tok,
                    x_pinned,
                    Y_st,
                    y_tok,
                    y_pinned,
                    to_device=to_device,
                )
                if timer_sync:
                    sync_accelerator(device)
                t_h2d_e = time.perf_counter_ns()
                h2d_s = (t_h2d_e - t_h2d_s) / 1000000000.0
        else:
            t_h2d_s = time.perf_counter_ns()
            X_dev, Y_dev, x_tok, y_tok = _move_staged_pair_to_device(
                X_st,
                x_tok,
                x_pinned,
                Y_st,
                y_tok,
                y_pinned,
                to_device=to_device,
            )
            if timer_sync:
                sync_accelerator(device)
            t_h2d_e = time.perf_counter_ns()
            h2d_s = (t_h2d_e - t_h2d_s) / 1000000000.0
        return (X_dev, Y_dev, t_ready, h2d_s)
    finally:
        if x_tok is not None and cpu_pool is not None:
            with contextlib.suppress(Exception):
                cpu_pool.release(x_tok)
        if y_tok is not None and cpu_pool is not None:
            with contextlib.suppress(Exception):
                cpu_pool.release(y_tok)


def _validate_distributed_group(meta: object, model: object) -> object:
    candidates = [(meta, "process_group"), (meta, "distributed_process_group")]
    tm = model.module if hasattr(model, "module") else model
    candidates.extend([(tm, "process_group"), (tm, "distributed_process_group")])
    for obj, attr in candidates:
        try:
            pg = getattr(obj, attr, None)
        except Exception:
            pg = None
        if pg is not None:
            return pg
    return None


def _get_world_size(pg: object) -> object:
    try:
        if pg is None:
            return max(1, int(get_world_size()))
        return int(torch.distributed.get_world_size(group=pg))
    except Exception:
        return max(1, int(get_world_size()))


def _reduce_sum(t: object, pg: object) -> None:
    if pg is None:
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    else:
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM, group=pg)


@torch.no_grad()
def _set_gate_factor(
    target_module: object,
    *args: Any,
    step: int,
    pg: object | None = None,
    local_rank: int = 0,
) -> None:
    if target_module is None:
        return
    if not bool(getattr(target_module, "p_gate_auto_k_enabled", False)):
        return
    gate = getattr(target_module, "p_gate", None)
    if gate is None or not hasattr(gate, "consume_fallback_stats"):
        return
    interval = int(getattr(target_module, "p_gate_auto_k_interval", 0) or 0)
    if interval <= 0:
        return
    warmup = int(getattr(target_module, "p_gate_auto_k_warmup", 0) or 0)
    if int(step) < int(warmup):
        if int(step) % max(1, int(interval)) == 0:
            with contextlib.suppress(Exception):
                gate.consume_fallback_stats()
        return
    if int(step) % int(interval) != 0:
        return
    step_buf = getattr(target_module, "p_gate_auto_k_step_buf", None)
    if isinstance(step_buf, torch.Tensor):
        with contextlib.suppress(Exception):
            step_buf.fill_(int(step))
    stats = gate.consume_fallback_stats()
    if not isinstance(stats, torch.Tensor) or stats.numel() < 6:
        return
    if bool(is_distributed()):
        with contextlib.suppress(Exception):
            _reduce_sum(stats, pg)

    count = float(stats[0].item())
    if not math.isfinite(count) or count <= 0.0:
        return
    active_low_rate = float((stats[1] / stats[0]).item())
    active_high_rate = float((stats[2] / stats[0]).item())
    width_mean = float((stats[3] / stats[0]).item())
    edge_low_rate = float((stats[4] / stats[0]).item())
    edge_high_rate = float((stats[5] / stats[0]).item())
    alpha = float(getattr(target_module, "p_gate_auto_k_ema_alpha", 0.1))
    alpha = max(0.0, min(1.0, alpha))
    ema_low_buf = getattr(target_module, "p_gate_auto_k_ema_low_buf", None)
    ema_high_buf = getattr(target_module, "p_gate_auto_k_ema_high_buf", None)
    ema_low_prev = float(ema_low_buf.item()) if isinstance(ema_low_buf, torch.Tensor) else 0.0
    ema_high_prev = float(ema_high_buf.item()) if isinstance(ema_high_buf, torch.Tensor) else 0.0
    ema_low_new = (1.0 - alpha) * ema_low_prev + alpha * active_low_rate
    ema_high_new = (1.0 - alpha) * ema_high_prev + alpha * active_high_rate
    if isinstance(ema_low_buf, torch.Tensor):
        with contextlib.suppress(Exception):
            ema_low_buf.fill_(float(ema_low_new))
    if isinstance(ema_high_buf, torch.Tensor):
        with contextlib.suppress(Exception):
            ema_high_buf.fill_(float(ema_high_new))
    ema_overall = 0.5 * (float(ema_low_new) + float(ema_high_new))
    ema_buf = getattr(target_module, "p_gate_auto_k_ema_buf", None)
    if isinstance(ema_buf, torch.Tensor):
        with contextlib.suppress(Exception):
            ema_buf.fill_(float(ema_overall))
    edge_enabled = bool(getattr(target_module, "p_gate_auto_k_edge_enabled", False))
    edge_alpha = float(getattr(target_module, "p_gate_auto_k_edge_ema_alpha", alpha))
    edge_alpha = max(0.0, min(1.0, edge_alpha))
    edge_ema_low_buf = getattr(target_module, "p_gate_auto_k_edge_ema_low_buf", None)
    edge_ema_high_buf = getattr(target_module, "p_gate_auto_k_edge_ema_high_buf", None)
    edge_ema_buf = getattr(target_module, "p_gate_auto_k_edge_ema_buf", None)
    edge_ema_low_prev = (
        float(edge_ema_low_buf.item()) if isinstance(edge_ema_low_buf, torch.Tensor) else 0.0
    )
    edge_ema_high_prev = (
        float(edge_ema_high_buf.item()) if isinstance(edge_ema_high_buf, torch.Tensor) else 0.0
    )
    edge_ema_low_new = (1.0 - edge_alpha) * edge_ema_low_prev + edge_alpha * edge_low_rate
    edge_ema_high_new = (1.0 - edge_alpha) * edge_ema_high_prev + edge_alpha * edge_high_rate
    if isinstance(edge_ema_low_buf, torch.Tensor):
        with contextlib.suppress(Exception):
            edge_ema_low_buf.fill_(float(edge_ema_low_new))
    if isinstance(edge_ema_high_buf, torch.Tensor):
        with contextlib.suppress(Exception):
            edge_ema_high_buf.fill_(float(edge_ema_high_new))
    if isinstance(edge_ema_buf, torch.Tensor):
        with contextlib.suppress(Exception):
            edge_ema_buf.fill_(float(0.5 * (edge_ema_low_new + edge_ema_high_new)))
    k_low_buf = getattr(target_module, "p_gate_fallback_k_low_buf", None)
    k_high_buf = getattr(target_module, "p_gate_fallback_k_high_buf", None)
    k_legacy_buf = getattr(target_module, "p_gate_fallback_k_buf", None)
    use_legacy = not (isinstance(k_low_buf, torch.Tensor) and isinstance(k_high_buf, torch.Tensor))
    if use_legacy:
        if not isinstance(k_legacy_buf, torch.Tensor):
            return
        k_low_buf = k_legacy_buf
        k_high_buf = k_legacy_buf
    k_low_prev = float(k_low_buf.item()) if isinstance(k_low_buf, torch.Tensor) else 0.0
    k_high_prev = float(k_high_buf.item()) if isinstance(k_high_buf, torch.Tensor) else 0.0
    k_low_new = k_low_prev
    k_high_new = k_high_prev
    target = float(getattr(target_module, "p_gate_auto_k_target_tight", 0.02))
    tol = float(getattr(target_module, "p_gate_auto_k_tolerance", 0.5))
    hi = target * (1.0 + tol)
    lo = max(0.0, target * (1.0 - tol))
    step_up = float(getattr(target_module, "p_gate_auto_k_step_up", 0.1))
    step_down = float(getattr(target_module, "p_gate_auto_k_step_down", 0.02))
    step_up_low = float(getattr(target_module, "p_gate_auto_k_step_up_low", step_up))
    step_down_low = float(getattr(target_module, "p_gate_auto_k_step_down_low", step_down))
    step_up_high = float(getattr(target_module, "p_gate_auto_k_step_up_high", step_up))
    step_down_high = float(getattr(target_module, "p_gate_auto_k_step_down_high", step_down))
    edge_target = float(getattr(target_module, "p_gate_auto_k_target_edge", 0.05))
    edge_tol = float(getattr(target_module, "p_gate_auto_k_edge_tolerance", 0.5))
    edge_hi = edge_target * (1.0 + edge_tol)
    edge_step_down_low = float(getattr(target_module, "p_gate_auto_k_edge_step_down_low", 0.01))
    edge_step_down_high = float(getattr(target_module, "p_gate_auto_k_edge_step_down_high", 0.01))
    k_min = float(getattr(target_module, "p_gate_auto_k_min", 1.0))
    k_max = float(getattr(target_module, "p_gate_auto_k_max", 16.0))
    if k_max < k_min:
        k_max = k_min
    edge_low_eff = (
        float(edge_ema_low_new) if math.isfinite(edge_ema_low_new) else float(edge_low_rate)
    )
    edge_high_eff = (
        float(edge_ema_high_new) if math.isfinite(edge_ema_high_new) else float(edge_high_rate)
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
    k_low_changed = bool(math.isfinite(k_low_new) and abs(k_low_new - k_low_prev) > 1e-12)
    k_high_changed = bool(math.isfinite(k_high_new) and abs(k_high_new - k_high_prev) > 1e-12)
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
                target_module, "p_gate_fallback_enabled", bool(k_low_new > 0.0 and k_high_new > 0.0)
            )
        upd_buf = getattr(target_module, "p_gate_auto_k_updates_buf", None)
        if isinstance(upd_buf, torch.Tensor):
            with contextlib.suppress(Exception):
                upd_buf.add_(1)
    log_interval = int(getattr(target_module, "p_gate_auto_k_log_interval", 0) or 0)
    log_due = bool(k_changed) if log_interval <= 0 else bool(int(step) % int(log_interval) == 0)
    if int(local_rank) == 0 and log_due:
        _LOGGER.info(
            "[p_gate] auto_k step=%d seen=%d activeL_sma=%.4f activeH_sma=%.4f width_mean=%.4f edgeL_sma=%.4f edgeH_sma=%.4f activeL_ema=%.4f activeH_ema=%.4f edgeL_ema=%.4f edgeH_ema=%.4f kL=%.4f -> %.4f kH=%.4f -> %.4f",
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


def _to_submodule(model: object) -> object:
    m = model
    for _ in range(8):
        if hasattr(m, "microbatch") and hasattr(m, "_auto_microbatch_pending"):
            return m
        child = getattr(m, "module", None)
        if child is None or child is m:
            break
        m = child
    return None


def _iter_checkpoint(root: object):
    try:
        import torch.nn as nn
        if isinstance(root, nn.Module):
            for mod in root.modules():
                if hasattr(mod, "_ckpt_min_bytes") and hasattr(mod, "_ckpt_enabled"):
                    yield mod
    except Exception:
        return


def _to_checkpoint(
    model: object,
    *,
    device: torch.device,
    step_total: int,
    ttl_steps: int,
    min_bytes: int,
) -> bool:
    inst = _to_submodule(model) or (model.module if hasattr(model, "module") else model)
    if inst is None:
        return False
    try:
        ttl_steps = max(1, int(ttl_steps))
        min_bytes = max(0, int(min_bytes))
        step_total = max(0, int(step_total))
    except Exception:
        return False

    until = step_total + ttl_steps
    try:
        until = int(_sync_tensor(until, device=device, src=0))
        min_bytes = int(_sync_tensor(min_bytes, device=device, src=0))
    except Exception:
        pass

    cur_until = int(getattr(inst, "_stnet_ckpt_pressure_until", 0) or 0)
    if cur_until >= until and int(getattr(inst, "_stnet_ckpt_pressure_min_bytes", 0) or 0) <= min_bytes:
        return False

    changed = False
    for mod in _iter_checkpoint(inst):
        if not hasattr(mod, "_stnet_ckpt_saved_min_bytes"):
            with contextlib.suppress(Exception):
                setattr(mod, "_stnet_ckpt_saved_min_bytes", int(getattr(mod, "_ckpt_min_bytes", 0) or 0))
                setattr(mod, "_stnet_ckpt_saved_enabled", bool(getattr(mod, "_ckpt_enabled", True)))
        try:
            cur = int(getattr(mod, "_ckpt_min_bytes", 0) or 0)
            if min_bytes < cur:
                setattr(mod, "_ckpt_min_bytes", int(min_bytes))
                changed = True
            if not bool(getattr(mod, "_ckpt_enabled", True)):
                setattr(mod, "_ckpt_enabled", True)
                changed = True
        except Exception:
            pass
    with contextlib.suppress(Exception):
        setattr(inst, "_stnet_ckpt_pressure_until", int(max(cur_until, until)))
        prev_mb = int(getattr(inst, "_stnet_ckpt_pressure_min_bytes", 0) or 0)
        if prev_mb <= 0:
            setattr(inst, "_stnet_ckpt_pressure_min_bytes", int(min_bytes))
        else:
            setattr(inst, "_stnet_ckpt_pressure_min_bytes", int(min(prev_mb, min_bytes)))
    return bool(changed)


def _from_checkpoint(model: object, *, step_total: int) -> None:
    inst = _to_submodule(model) or (model.module if hasattr(model, "module") else model)
    if inst is None:
        return
    try:
        step_total = int(step_total)
    except Exception:
        return
    until = int(getattr(inst, "_stnet_ckpt_pressure_until", 0) or 0)
    if until <= 0 or step_total < until:
        return
    for mod in _iter_checkpoint(inst):
        try:
            if hasattr(mod, "_stnet_ckpt_saved_min_bytes"):
                setattr(mod, "_ckpt_min_bytes", int(getattr(mod, "_stnet_ckpt_saved_min_bytes", 0) or 0))
            if hasattr(mod, "_stnet_ckpt_saved_enabled"):
                setattr(mod, "_ckpt_enabled", bool(getattr(mod, "_stnet_ckpt_saved_enabled", True)))
            for k in ("_stnet_ckpt_saved_min_bytes", "_stnet_ckpt_saved_enabled"):
                with contextlib.suppress(Exception):
                    delattr(mod, k)
        except Exception:
            pass
    with contextlib.suppress(Exception):
        setattr(inst, "_stnet_ckpt_pressure_until", 0)
        setattr(inst, "_stnet_ckpt_pressure_min_bytes", 0)


def _compute_batch_bytes_per_sample(obj: object) -> tuple[int | None, int]:
    batch_dim: int | None = None
    bytes_per_sample = 0
    stack: list[object] = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, torch.Tensor):
            if o.numel() <= 0:
                continue
            b = int(o.shape[0]) if o.ndim >= 1 else 1
            if batch_dim is None:
                batch_dim = b
            if o.ndim >= 1 and b > 0:
                one = o[:1]
            else:
                one = o.reshape(1, -1)
            bytes_per_sample += int(one.nelement()) * int(one.element_size())
        elif isinstance(o, TensorDictBase):
            stack.extend(list(o.values()))
        elif isinstance(o, Mapping):
            stack.extend(list(o.values()))
        elif isinstance(o, (list, tuple)):
            stack.extend(list(o))
    if bytes_per_sample <= 0:
        return (None, 0)
    return (batch_dim, bytes_per_sample)


def get_loader_state(directory: PathLike) -> object:
    return os.path.join(directory, _DL_STATE_FILE)


def get_progress_bar(
    *args: Any,
    title: str,
    total: int,
    device: torch.device,
    **kwargs: Any,
) -> object:
    try:
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return None
    except Exception:
        pass
    if int(total) <= 0:
        return None
    bar = tqdm(
        total=int(total),
        desc=f"{title} ({device.type.upper()}) ",
        unit="I/O < 0.01 MB/s, COM < 0.01 TFLOPS",
        bar_format="{desc}"
        + "{bar} {percentage:3.0f}% "
        + "({unit}) Elapsed: {elapsed}, Remaining: {remaining}",
        colour="green",
        ascii=True,
        position=0,
        leave=False,
        file=sys.stdout,
    )
    return bar


def update_progress_bar(
    bar: object,
    finish: bool,
    *args: Any,
    mbps: float | None = None,
    tflops: float | None = None,
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
    com_expr = f"COM = {tflops_val:.2f} TFLOPS" if tflops_val >= 0.01 else "COM < 0.01 TFLOPS"
    bar.unit = io_expr + ", " + com_expr
    try:
        inc = int(finish)
    except Exception:
        inc = 1
    if inc > 0:
        bar.update(inc)


def epochs(
    model: nn.Module,
    device: torch.device,
    local_rank: int,
    ops: object,
    *args: Any,
    param_dtype: torch.dtype,
    optimizer: object,
    scaler: object,
    sched: object,
    loss_controller: object,
    top_loss: object,
    bottom_loss: object,
    train_loader: object,
    val_loader: object,
    total_epochs: int,
    scheduler_step_per_batch: bool = True,
    swa_helper: object | None = None,
    swa_start_epoch: int = 0,
    buffers_dtype: torch.dtype | None = None,
    dataset: object | None = None,
    **kwargs: Any,
) -> object:
    from ..data.nodes import Sampler
    if train_loader is None:
        raise RuntimeError("epochs requires a training dataloader")
    meta = dataset if isinstance(dataset, Dataset) else Dataset.for_device(device)
    autocast_dtype = None
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
    cpu_pool = None
    pool_capacity = 0
    if is_pin_supported(str(getattr(device, "type", "cpu"))):
        with contextlib.suppress(Exception):
            Memory.prefer_local_numa()
        try:
            cpu_pool_cap = max(2, int(_env_int("STNET_RUNTIME_PIN_POOL_CAPACITY", 8)))
            cpu_pool = Pool(capacity=cpu_pool_cap)
            pool_capacity = int(getattr(cpu_pool, "capacity", 8))
        except Exception:
            cpu_pool = None
            pool_capacity = 0
    per_batch = getattr(train_loader, "batch_size", None)
    est_bytes_per_sample = None
    with contextlib.suppress(Exception):
        v = getattr(Sampler, "_per_sample_mem_bytes", 0)
        if isinstance(v, int) and v > 0:
            est_bytes_per_sample = int(v)
    if per_batch is None or int(per_batch) <= 0 or est_bytes_per_sample is None:


        try:
            it = iter(train_loader)
            sample = next(it)
            bs, bytes_ps = _compute_batch_bytes_per_sample(sample)
            if (per_batch is None or int(per_batch) <= 0) and bs is not None and (bs > 0):
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
    dev_margin = env_float("STNET_DEVICE_MARGIN", 0.8)
    host_margin = env_float("STNET_HOST_MARGIN", 0.8)
    budget_slack = env_float("STNET_BUDGET_SLACK", 1.25)
    budget_slack = max(1.0, min(4.0, float(budget_slack)))
    dev_budget_ratio = env_float("STNET_DEVICE_BUDGET_RATIO", 1.0)
    dev_budget_min_bytes = env_int("STNET_DEVICE_BUDGET_MIN_BYTES", 0)
    _dev_budget_max_raw = env_int("STNET_DEVICE_BUDGET_MAX_BYTES", 0)
    dev_budget_max_bytes = None if int(_dev_budget_max_raw) <= 0 else int(_dev_budget_max_raw)
    host_budget_ratio = env_float("STNET_HOST_BUDGET_RATIO", 1.0)
    host_budget_min_bytes = env_int("STNET_HOST_BUDGET_MIN_BYTES", 0)
    _host_budget_max_raw = env_int("STNET_HOST_BUDGET_MAX_BYTES", 0)
    host_budget_max_bytes = None if int(_host_budget_max_raw) <= 0 else int(_host_budget_max_raw)
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
    tpl = None
    if est_bytes_per_sample is not None and est_bytes_per_sample > 0 and (max_grad_accum > 0):
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
                host_budget_max_bytes=None
                if host_budget_max_bytes is None
                else int(host_budget_max_bytes),
                device_budget_ratio=float(dev_budget_ratio),
                device_budget_min_bytes=int(dev_budget_min_bytes),
                device_budget_max_bytes=None
                if dev_budget_max_bytes is None
                else int(dev_budget_max_bytes),
            )
        except Exception:
            tpl = None
    safe_host_bytes = None
    safe_host_total = None
    safe_dev_bytes = None
    safe_dev_total = None
    max_from_mem = None
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
                    target_total_samples = max(1, int(per_batch or 1)) * max(1, int(min_grad_accum))
                    new_dev_cap = tpl.device_budget_max_bytes
                    new_host_cap = tpl.host_budget_max_bytes
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
                        base_host = (
                            int(tpl.host_sample_bytes)
                            * max(1, inflight)
                            * max(1, lw)
                            * int(target_total_samples)
                        )
                        cap_host = int(float(base_host) * float(budget_slack))
                        if safe_host_total is not None and int(safe_host_total) > 0:
                            cap_host = min(int(cap_host), int(safe_host_total))
                        cap_host = max(0, int(cap_host))
                        new_host_cap = None if cap_host <= 0 else cap_host
                    if (
                        new_dev_cap != tpl.device_budget_max_bytes
                        or new_host_cap != tpl.host_budget_max_bytes
                    ):
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
                    max_from_mem = max(1, int(total_samples_cap) // int(per_batch or 1))
        except Exception:
            safe_host_bytes = None
            safe_host_total = None
            safe_dev_bytes = None
            safe_dev_total = None
    if max_from_mem is not None:
        max_grad_accum = max(int(min_grad_accum), min(int(max_grad_accum), int(max_from_mem)))
    grad_accum_steps = int(min_grad_accum)
    grad_accum_steps = _sync_tensor(grad_accum_steps, device=device, src=0)
    proc = None
    if psutil is not None:
        try:
            proc = psutil.Process(os.getpid())
        except Exception:
            proc = None
    gpu_util_ema = None
    mem_util_ema = None
    util_alpha = 0.2
    global_step = 0
    p_gate_auto_step_total = 0
    with contextlib.suppress(Exception):
        target_for_autok = model.module if hasattr(model, "module") else model
        step_buf = getattr(target_for_autok, "p_gate_auto_k_step_buf", None)
        if isinstance(step_buf, torch.Tensor):
            p_gate_auto_step_total = int(step_buf.item())
    util_adjust_interval = 0
    util_warmup_steps = 0


    if buffers_dtype is not None:
        target_for_buffers = model.module if hasattr(model, "module") else model
        _cast_batchnorm_buffers_dtype(target_for_buffers, buffers_dtype)
    model_for_hist = model.module if hasattr(model, "module") else model
    hist = None
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
        start_ns = posix_time()
        start_sec = round(float(start_ns) / 1000000000.0, 6)
        hist.start_session(start_sec)
        hist.set_epochs(total_epochs)
        os_name = platform.system()
        match os_name:
            case "Linux":
                pretty = None
                with contextlib.suppress(Exception):
                    if os.path.exists("/etc/os-release"):
                        with open("/etc/os-release", "r", encoding="utf-8") as f:
                            for line in f:
                                if line.startswith("PRETTY_NAME="):
                                    pretty = line.strip().split("=", 1)[1].strip().strip('"')
                                    break
                os_full = pretty or f"{os_name} {platform.release()}"
            case "Darwin":
                ver, _, _ = platform.mac_ver()
                os_full = f"macOS {ver or platform.release()}"
            case "Windows":
                ver = platform.version()
                rel = platform.release()
                os_full = f"Windows {rel} {ver}"
            case _:
                os_full = f"{os_name} {platform.release()}"
        kernel = platform.release()
        arch_list = [platform.machine(), platform.processor() or ""]
        cpu_list = []
        proc = platform.processor()
        if proc:
            cpu_list.append(proc)
        try:
            ram_bytes = Memory.total()
            ram_gb = int(round(float(ram_bytes) / 1024**3))
        except Exception:
            ram_gb = 0
        py_ver = platform.python_version()
        backend_list = []
        if is_accelerator_available("cuda"):
            backend_list.append("cuda")
        if is_accelerator_available("xpu"):
            backend_list.append("xpu")
        if is_accelerator_available("mps"):
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
    scaler_x_dtype = (
        model_for_scaler.scaler.x_mean.dtype
        if model_for_scaler.scaler.x_mean.dtype in (torch.float64, torch.int64)
        else torch.float64
    )
    scaler_y_dtype = (
        model_for_scaler.scaler.y_mean.dtype
        if model_for_scaler.scaler.y_mean.dtype in (torch.float64, torch.int64)
        else torch.float64
    )
    with torch.no_grad():
        used_memmap_stats = False
        x_count = 0
        x_sum = None
        x_sum_sq = None
        y_count = 0
        y_sum = None
        y_sum_sq = None
        y_min = None
        y_max = None
        y_q_low = None
        y_q_high = None
        want_y_quantiles = bool(getattr(model_for_scaler, "p_gate_bounds_use_quantile", False))
        q_low = float(getattr(model_for_scaler, "p_gate_bounds_q_low", 0.005))
        q_high = float(getattr(model_for_scaler, "p_gate_bounds_q_high", 0.995))
        q_max = int(getattr(model_for_scaler, "p_gate_bounds_q_max_samples", 8192) or 0)
        want_y_quantiles = bool(want_y_quantiles and q_max > 0 and (0.0 < q_low < q_high < 1.0))
        y_q_samples = None
        scaler_stats = None
        with contextlib.suppress(Exception):
            scaler_stats = Storage.load_scaler_stats(ops.sources)
        if scaler_stats is not None:
            used_memmap_stats = True
            x_count = int(scaler_stats.get("train_count") or 0)
            y_count = int(x_count)
            x_sum = scaler_stats["x_sum"].to(device=scaler_x_device)
            x_sum_sq = scaler_stats["x_sum_sq"].to(device=scaler_x_device)
            y_sum = scaler_stats["y_sum"].to(device=scaler_y_device)
            y_sum_sq = scaler_stats["y_sum_sq"].to(device=scaler_y_device)
            y_min = scaler_stats.get("y_min")
            y_max = scaler_stats.get("y_max")
            y_q_low = scaler_stats.get("y_q_low")
            y_q_high = scaler_stats.get("y_q_high")
            if y_min is not None and y_max is not None:
                y_min = y_min.to(device=scaler_y_device)
                y_max = y_max.to(device=scaler_y_device)
            if y_q_low is not None and y_q_high is not None:
                y_q_low = y_q_low.to(device=scaler_y_device)
                y_q_high = y_q_high.to(device=scaler_y_device)
        else:
            sx_tmp = None
            sx2_tmp = None
            sy_tmp = None
            sy2_tmp = None
            sy_min_tmp = None
            sy_max_tmp = None
            scaler_x_dtype = model_for_scaler.scaler.x_mean.dtype
            scaler_y_dtype = model_for_scaler.scaler.y_mean.dtype
            for batch in train_loader:
                feats, labs = get_row(batch, labels_required=True)
                if feats.ndim > 2:
                    feats = feats.reshape(feats.shape[0], -1)
                with inference_mode(torch.nn.Identity()):
                    xf = feats.to(device=scaler_x_device, dtype=scaler_x_dtype)
                    if xf.ndim > 2:
                        xf = xf.reshape(xf.shape[0], -1)
                    n_x = int(xf.shape[0])
                    if n_x > 0:
                        x_count += n_x
                        if x_sum is None:
                            x_sum = torch.zeros(
                                xf.shape[1], device=scaler_x_device, dtype=scaler_x_dtype
                            )
                            x_sum_sq = torch.zeros_like(x_sum)
                            sx_tmp = torch.empty_like(x_sum)
                            sx2_tmp = torch.empty_like(x_sum)
                        assert x_sum is not None and x_sum_sq is not None
                        assert sx_tmp is not None and sx2_tmp is not None
                        torch.sum(xf, dim=0, out=sx_tmp)
                        x_sum.add_(sx_tmp)
                        sx2_tmp.copy_(torch.einsum("ni,ni->i", xf, xf))
                        x_sum_sq.add_(sx2_tmp)
                    yf = labs.to(device=scaler_y_device, dtype=scaler_y_dtype)
                    if yf.ndim > 2:
                        yf = yf.reshape(yf.shape[0], -1)
                    n_y = int(yf.shape[0])
                    if n_y > 0:
                        y_count += n_y
                        if y_sum is None:
                            y_sum = torch.zeros(
                                yf.shape[1], device=scaler_y_device, dtype=scaler_y_dtype
                            )
                            y_sum_sq = torch.zeros_like(y_sum)
                            sy_tmp = torch.empty_like(y_sum)
                            sy2_tmp = torch.empty_like(y_sum)
                            sy_min_tmp = torch.empty_like(y_sum)
                            sy_max_tmp = torch.empty_like(y_sum)
                            y_min = torch.full_like(y_sum, float("inf"))
                            y_max = torch.full_like(y_sum, float("-inf"))
                        assert y_sum is not None and y_sum_sq is not None
                        assert sy_tmp is not None and sy2_tmp is not None
                        torch.sum(yf, dim=0, out=sy_tmp)
                        y_sum.add_(sy_tmp)
                        sy2_tmp.copy_(torch.einsum("ni,ni->i", yf, yf))
                        y_sum_sq.add_(sy2_tmp)
                        if y_min is not None and y_max is not None:
                            assert sy_min_tmp is not None and sy_max_tmp is not None
                            torch.amin(yf, dim=0, out=sy_min_tmp)
                            torch.minimum(y_min, sy_min_tmp, out=y_min)
                            torch.amax(yf, dim=0, out=sy_max_tmp)
                            torch.maximum(y_max, sy_max_tmp, out=y_max)
                        if want_y_quantiles and q_max > 0:

                            y_take = yf
                            if int(y_take.shape[0]) > int(q_max):
                                idx = torch.randperm(int(y_take.shape[0]), device=y_take.device)[
                                    : int(q_max)
                                ]
                                y_take = y_take.index_select(0, idx)
                            if y_q_samples is None:
                                y_q_samples = y_take
                            else:
                                y_q_samples = torch.cat([y_q_samples, y_take], dim=0)
                            if int(y_q_samples.shape[0]) > int(q_max):
                                idx = torch.randperm(
                                    int(y_q_samples.shape[0]), device=y_q_samples.device
                                )[: int(q_max)]
                                y_q_samples = y_q_samples.index_select(0, idx)
            if is_distributed() and (not used_memmap_stats):
                x_count_t = torch.tensor(
                    float(x_count), device=scaler_x_device, dtype=scaler_x_dtype
                )
                torch.distributed.all_reduce(x_count_t, op=torch.distributed.ReduceOp.SUM)
                x_count = int(x_count_t.item())
                if x_sum is not None:
                    torch.distributed.all_reduce(x_sum, op=torch.distributed.ReduceOp.SUM)
                if x_sum_sq is not None:
                    torch.distributed.all_reduce(x_sum_sq, op=torch.distributed.ReduceOp.SUM)
                y_count_t = torch.tensor(
                    float(y_count), device=scaler_y_device, dtype=scaler_y_dtype
                )
                torch.distributed.all_reduce(y_count_t, op=torch.distributed.ReduceOp.SUM)
                y_count = int(y_count_t.item())
                if y_sum is not None:
                    torch.distributed.all_reduce(y_sum, op=torch.distributed.ReduceOp.SUM)
                if y_sum_sq is not None:
                    torch.distributed.all_reduce(y_sum_sq, op=torch.distributed.ReduceOp.SUM)
                if y_min is not None:
                    torch.distributed.all_reduce(y_min, op=torch.distributed.ReduceOp.MIN)
                if y_max is not None:
                    torch.distributed.all_reduce(y_max, op=torch.distributed.ReduceOp.MAX)
        if want_y_quantiles and (y_q_low is None or y_q_high is None):
            if y_q_samples is None:
                for batch in train_loader:
                    _, labs = get_row(batch, labels_required=True)
                    with inference_mode(torch.nn.Identity()):
                        yf = labs.to(device=scaler_y_device, dtype=torch.float64)
                        if yf.ndim > 2:
                            yf = yf.reshape(yf.shape[0], -1)
                        if int(yf.shape[0]) <= 0:
                            continue
                        y_take = yf
                        if int(y_take.shape[0]) > int(q_max):
                            idx2 = torch.randperm(int(y_take.shape[0]), device=y_take.device)[
                                : int(q_max)
                            ]
                            y_take = y_take.index_select(0, idx2)
                        if y_q_samples is None:
                            y_q_samples = y_take
                        else:
                            y_q_samples = torch.cat([y_q_samples, y_take], dim=0)
                        if int(y_q_samples.shape[0]) > int(q_max):
                            idx2 = torch.randperm(
                                int(y_q_samples.shape[0]), device=y_q_samples.device
                            )[: int(q_max)]
                            y_q_samples = y_q_samples.index_select(0, idx2)
            if y_q_samples is not None and int(y_q_samples.shape[0]) > 0:
                y_all = y_q_samples
                if is_distributed():
                    world = int(torch.distributed.get_world_size())
                    n_local = torch.tensor(
                        [int(y_all.shape[0])], device=y_all.device, dtype=torch.int64
                    )
                    counts = [torch.zeros_like(n_local) for _ in range(world)]
                    torch.distributed.all_gather(counts, n_local)
                    counts_i = [int(c.item()) for c in counts]
                    n_max = int(max(counts_i) if counts_i else 0)
                    if n_max > 0:
                        if int(y_all.shape[0]) < n_max:
                            pad = torch.zeros(
                                (n_max - int(y_all.shape[0]), int(y_all.shape[1])),
                                device=y_all.device,
                                dtype=y_all.dtype,
                            )
                            y_pad = torch.cat([y_all, pad], dim=0)
                        else:
                            y_pad = y_all
                        gathered = [torch.empty_like(y_pad) for _ in range(world)]
                        torch.distributed.all_gather(gathered, y_pad)
                        parts = []
                        for gi, c in zip(gathered, counts_i):
                            if int(c) > 0:
                                parts.append(gi[: int(c)])
                        if len(parts) > 0:
                            y_all = torch.cat(parts, dim=0)
                try:
                    y_q_low = torch.quantile(y_all, float(q_low), dim=0)
                    y_q_high = torch.quantile(y_all, float(q_high), dim=0)
                except Exception:
                    y_q_low = None
                    y_q_high = None
        eps = float(model_for_scaler.scaler.eps)
        if x_count > 0 and x_sum is not None and (x_sum_sq is not None):
            mean_x = x_sum / float(x_count)
            var_x = x_sum_sq / float(x_count) - mean_x * mean_x
            std_x = torch.sqrt(var_x.clamp_min(eps))
            if model_for_scaler.scaler.x_mean.shape != mean_x.shape:
                model_for_scaler.scaler.x_mean.resize_(mean_x.shape)
            if model_for_scaler.scaler.x_std.shape != std_x.shape:
                model_for_scaler.scaler.x_std.resize_(std_x.shape)
            model_for_scaler.scaler.x_mean.copy_(mean_x)
            model_for_scaler.scaler.x_std.copy_(std_x)
        if y_count > 0 and y_sum is not None and (y_sum_sq is not None):
            mean_y = y_sum / float(y_count)
            var_y = y_sum_sq / float(y_count) - mean_y * mean_y
            std_y = torch.sqrt(var_y.clamp_min(eps))
            if model_for_scaler.scaler.y_mean.shape != mean_y.shape:
                model_for_scaler.scaler.y_mean.resize_(mean_y.shape)
            if model_for_scaler.scaler.y_std.shape != std_y.shape:
                model_for_scaler.scaler.y_std.resize_(std_y.shape)
            model_for_scaler.scaler.y_mean.copy_(mean_y)
            model_for_scaler.scaler.y_std.copy_(std_y)
            if y_min is not None and y_max is not None:
                if model_for_scaler.scaler.y_min.shape != y_min.shape:
                    model_for_scaler.scaler.y_min.resize_(y_min.shape)
                if model_for_scaler.scaler.y_max.shape != y_max.shape:
                    model_for_scaler.scaler.y_max.resize_(y_max.shape)
                model_for_scaler.scaler.y_min.copy_(y_min)
                model_for_scaler.scaler.y_max.copy_(y_max)
            if y_q_low is not None and y_q_high is not None:
                if model_for_scaler.scaler.y_q_low.shape != y_q_low.shape:
                    model_for_scaler.scaler.y_q_low.resize_(y_q_low.shape)
                if model_for_scaler.scaler.y_q_high.shape != y_q_high.shape:
                    model_for_scaler.scaler.y_q_high.resize_(y_q_high.shape)
                model_for_scaler.scaler.y_q_low.copy_(y_q_low)
                model_for_scaler.scaler.y_q_high.copy_(y_q_high)
    in_dim = int(ops.in_dim)
    dev_type = str(getattr(device, "type", "cpu"))
    non_blocking_ok = bool(dev_type in ("cuda", "xpu"))
    pinned_ok = bool(is_pin_supported(dev_type))
    backend = accelerator_type(dev_type)
    stream_fn = getattr(backend, "current_stream", None) if backend is not None else None
    Event = getattr(backend, "Event", None) if backend is not None else None
    can_stream_release = bool(
        pinned_ok and non_blocking_ok and callable(stream_fn) and (Event is not None)
    )
    make_fence_event = partial(new_accelerator_event, device, enable_timing=False)
    use_timer = _is_event_timer_available(device)
    timer_sync = (not use_timer) and bool(_is_clock_synchronized(dev_type))
    train_steps = _get_batch_length(train_loader)
    val_steps = _get_batch_length(val_loader)
    total_updates = int(total_epochs) * (int(train_steps) + int(val_steps))
    if train_steps > 0:
        util_adjust_interval = max(10, int(train_steps * 0.05))
        util_warmup_steps = max(
            util_adjust_interval, min(int(train_steps), max(50, int(train_steps * 0.1)))
        )
    status_bar = (
        get_progress_bar(title="Training", total=total_updates, device=device) if local_rank == 0 else None
    )
    scheduler_step_per_batch = bool(scheduler_step_per_batch)
    swa_enabled = swa_helper is not None
    swa_start_epoch = max(0, int(swa_start_epoch))
    prev_io_time = 0.0
    prev_comp_time = 0.0
    prev_io_bytes = 0.0
    prev_flops = 0.0
    prev_samples = 0.0
    join_context = joining(model=model, optimizer=optimizer)
    with join_context:
        with contextlib.suppress(Exception):
            get_affinity().pin_thread()
        stage_tensor = partial(
            _pool_tensor,
            device=device,
            dev_type=dev_type,
            pinned_ok=pinned_ok,
            cpu_pool=cpu_pool,
        )
        to_device_with_stream = partial(
            _stream_tensor,
            device=device,
            dev_type=dev_type,
            non_blocking_ok=non_blocking_ok,
            backend=backend,
            stream_fn=stream_fn,
            Event=Event,
            fence_event_factory=make_fence_event,
            can_stream_release=can_stream_release,
            cpu_pool=cpu_pool,
        )
        comp_ev_s = None
        comp_ev_e = None
        if use_timer:
            pair = _get_thread_events(device, slot="comp")
            if pair is not None:
                comp_ev_s, comp_ev_e = pair
        torch_prof = None
        prof_enabled = _env_flag(
            "STNET_TORCH_PROFILE_TRAIN", _env_flag("STNET_TORCH_PROFILE", False)
        )
        prof_all_ranks = _env_flag("STNET_TORCH_PROFILE_ALL_RANKS", False)
        prof_rank = int(torch.distributed.get_rank()) if is_distributed() else 0
        if prof_enabled and (prof_all_ranks or prof_rank == 0):
            prof_dir = env_str("STNET_TORCH_PROFILE_DIR")
            if not prof_dir:
                prof_dir = os.path.join(str(ops.ckpt_dir or "."), "torch_profiler")
            torch_prof = _get_torch_profiler(
                enabled=True,
                tag=f"train-{str(run_id)}",
                device=device,
                out_dir=str(prof_dir),
                rank=prof_rank,
            )
            if torch_prof is not None:
                with contextlib.suppress(Exception):
                    torch_prof.start()
        flop_counter_train = FlopCounter(model, mode="train", device=device)
        flop_counter_val = FlopCounter(model, mode="eval", device=device) if val_loader is not None else None
        for epoch_idx in range(int(total_epochs)):
            with contextlib.suppress(Exception):
                epochables = getattr(train_loader, "_stnet_epochables", None)
                if epochables is not None:
                    for obj in epochables:
                        fn = getattr(obj, "set_epoch", None)
                        if callable(fn):
                            fn(int(epoch_idx))
                else:
                    fn = getattr(getattr(train_loader, "sampler", None), "set_epoch", None)
                    if callable(fn):
                        fn(int(epoch_idx))
                    fn = getattr(train_loader, "set_epoch", None)
                    if callable(fn):
                        fn(int(epoch_idx))
            if is_distributed():
                target_module = model.module if hasattr(model, "module") else model
                distributed_sync(target_module, device=device)
            flop_breakdown_epoch = {}
            io_time = 0.0
            comp_time = 0.0
            io_bytes = 0.0
            flops = 0.0
            train_samples_epoch = 0.0
            with flop_counter_train:
                model.train()
                train_pg = _validate_distributed_group(meta, model) if is_distributed() else None
                global_step = 0
                optimizer.zero_grad(set_to_none=True)
                t_fetch_start = time.perf_counter_ns()
                total_batches = len(train_loader)
                train_accum_since_last = 0
                lw_top_sum = None
                lw_bottom_sum = None
                lw_count = 0
                for step_idx, _raw in enumerate(train_loader):
                    train_accum_since_last += 1
                    while True:
                        try:
                            with contextlib.suppress(Exception):
                                flush_accelerator_memory_stats(device)
                            X, Y_opt, t_ready, h2d_s = _pin(
                                meta,
                                _raw,
                                device=device,
                                stage_tensor=stage_tensor,
                                to_device=to_device_with_stream,
                                cpu_pool=cpu_pool,
                                use_timer=use_timer,
                                timer_sync=timer_sync,
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
                            wait_s = (t_ready - t_fetch_start) / 1000000000.0
                            io_time += float(wait_s + h2d_s)
                            with contextlib.suppress(Exception):
                                io_bytes += float(
                                    X.element_size() * X.nelement()
                                    + Y.element_size() * Y.nelement()
                                )
                            should_sync = (step_idx + 1) % max(
                                1, grad_accum_steps
                            ) == 0 or step_idx + 1 == total_batches
                            if use_timer and comp_ev_s is not None and comp_ev_e is not None:
                                comp_ev_s.record()
                            else:
                                t_comp_s = time.perf_counter_ns()
                            with no_sync(model, enable=grad_accum_steps > 1 and (not should_sync)):
                                with flop_counter_train.step(display=False) as train_counter:
                                    if (
                                        getattr(device, "type", None) == "cuda"
                                        and bool(getattr(model, "_compile_cudagraphs", False))
                                    ):
                                        cudagraph_mark_step_begin()
                                    with Autocast.float(device):
                                        Y_flat = Y.reshape(Y.shape[0], -1)
                                        if Y_flat.device != device or Y_flat.dtype != param_dtype:
                                            Y_flat = Y_flat.to(
                                                device,
                                                dtype=param_dtype,
                                                non_blocking=non_blocking_ok,
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
                                    if loss_val is None:
                                        raise RuntimeError(
                                            "Model returned no loss value during training. Ensure loss functions are provided and returning valid outputs."
                                        )
                                    if not isinstance(loss_val, torch.Tensor):
                                        loss_val = torch.as_tensor(
                                            loss_val, device=device, dtype=param_dtype
                                        )
                                    else:
                                        loss_val = loss_val.to(device=device, dtype=param_dtype)
                                    accum_scale = max(1, grad_accum_steps)
                                    loss_for_backprop = loss_val / float(accum_scale)
                                    scaler.scale(loss_for_backprop).backward()
                                    if loss_top_val is not None or loss_bottom_val is not None:
                                        lw_count += 1
                                        if isinstance(loss_top_val, torch.Tensor):
                                            v = loss_top_val.detach()
                                            lw_top_sum = v if lw_top_sum is None else lw_top_sum + v
                                        if isinstance(loss_bottom_val, torch.Tensor):
                                            v = loss_bottom_val.detach()
                                            lw_bottom_sum = (
                                                v if lw_bottom_sum is None else lw_bottom_sum + v
                                            )
                                    if should_sync:
                                        scaler.unscale_(optimizer)
                                        scaler.step(optimizer)
                                        scaler.update()
                                        optimizer.zero_grad(set_to_none=True)

                                        target_for_step = model.module if hasattr(model, "module") else model
                                        inst_step = _to_submodule(target_for_step) or target_for_step
                                        with contextlib.suppress(Exception):
                                            setattr(inst_step, "_stnet_step_total", int(p_gate_auto_step_total))

                                        peak = None
                                        free = None
                                        total = None
                                        with contextlib.suppress(Exception):
                                            peak = int(accelerator_max_allocated_memory(device))
                                        with contextlib.suppress(Exception):
                                            free, total = _device_mem_get_info(device)
                                        if isinstance(peak, int) and peak > 0 and isinstance(total, int) and total and total > 0:
                                            frac = float(peak) / float(total)
                                            prev_ema = float(getattr(inst_step, "_stnet_peak_ema", 0.0) or 0.0)
                                            ema = frac if prev_ema <= 0.0 else (0.9 * prev_ema + 0.1 * frac)
                                            with contextlib.suppress(Exception):
                                                setattr(inst_step, "_stnet_peak_ema", float(ema))
                                            spike = (prev_ema > 0.0 and frac > max(0.92, prev_ema * 1.25))
                                            if frac > 0.92 or spike:
                                                _to_checkpoint(
                                                    model,
                                                    device=device,
                                                    step_total=int(p_gate_auto_step_total),
                                                    ttl_steps=128,
                                                    min_bytes=16 * 1024 * 1024,
                                                )

                                        _from_checkpoint(model, step_total=int(p_gate_auto_step_total))

                                        if scheduler_step_per_batch:
                                            with contextlib.suppress(Exception):
                                                sched.step()
                                        if lw_count > 0:
                                            top_avg_t = (
                                                lw_top_sum / float(lw_count)
                                                if lw_top_sum is not None
                                                else None
                                            )
                                            bottom_avg_t = (
                                                lw_bottom_sum / float(lw_count)
                                                if lw_bottom_sum is not None
                                                else None
                                            )
                                            if is_distributed():
                                                ws = _get_world_size(train_pg)
                                                if top_avg_t is not None:
                                                    _reduce_sum(top_avg_t, train_pg)
                                                    top_avg_t = top_avg_t / float(ws)
                                                if bottom_avg_t is not None:
                                                    _reduce_sum(bottom_avg_t, train_pg)
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
                                p_gate_auto_step_total += 1
                                with contextlib.suppress(Exception):
                                    target_for_autok = (
                                        model.module if hasattr(model, "module") else model
                                    )
                                    _set_gate_factor(
                                        target_for_autok,
                                        step=p_gate_auto_step_total,
                                        pg=train_pg,
                                        local_rank=local_rank,
                                    )
                                match device.type:
                                    case "cuda":
                                        util_now, mem_now = _gpu_nvml_utils(device)
                                    case "xpu":
                                        util_now, mem_now = (None, _xpu_mem_util(device))
                                    case "mps":
                                        util_now, mem_now = (None, _mps_mem_util(device))
                                    case _:
                                        util_now, mem_now = (None, None)
                                if util_now is not None:
                                    util_now = float(util_now)
                                    if gpu_util_ema is None:
                                        gpu_util_ema = util_now
                                    else:
                                        gpu_util_ema = (
                                            1.0 - util_alpha
                                        ) * gpu_util_ema + util_alpha * util_now
                                if mem_now is not None:
                                    mem_now = float(mem_now)
                                    if mem_util_ema is None:
                                        mem_util_ema = mem_now
                                    else:
                                        mem_util_ema = (
                                            1.0 - util_alpha
                                        ) * mem_util_ema + util_alpha * mem_now
                                if (
                                    util_adjust_interval > 0
                                    and global_step >= util_warmup_steps
                                    and (global_step % util_adjust_interval == 0)
                                ):
                                    new_grad_accum = grad_accum_steps
                                    util_frac = None
                                    mem_frac = None
                                    if gpu_util_ema is not None:
                                        util_frac = max(0.0, min(1.0, gpu_util_ema / 100.0))
                                    if mem_util_ema is not None:
                                        mem_frac = max(0.0, min(1.0, mem_util_ema / 100.0))
                                    if util_frac is None:
                                        total_t_local = float(io_time + comp_time)
                                        if total_t_local > 0.0:
                                            util_frac = max(
                                                0.0, min(1.0, float(comp_time) / total_t_local)
                                            )
                                        else:
                                            util_frac = 0.0
                                    if util_frac is not None:
                                        if mem_frac is not None:
                                            if util_frac < 0.88 and mem_frac < 0.9:
                                                new_grad_accum = min(
                                                    max_grad_accum, grad_accum_steps + 1
                                                )
                                            elif util_frac > 0.97 or mem_frac > 0.92:
                                                new_grad_accum = max(
                                                    min_grad_accum, grad_accum_steps - 1
                                                )
                                        elif util_frac < 0.88:
                                            new_grad_accum = min(
                                                max_grad_accum, grad_accum_steps + 1
                                            )
                                        elif util_frac > 0.97:
                                            new_grad_accum = max(
                                                min_grad_accum, grad_accum_steps - 1
                                            )
                                    host_avail_now = None
                                    host_total_now = None
                                    with contextlib.suppress(Exception):
                                        host_avail_now = Memory.available()
                                        host_total_now = Memory.total()
                                    host_low = False
                                    if host_avail_now is not None and host_avail_now > 0:
                                        host_low_abs = host_avail_now < 512 * 1024 * 1024
                                        host_low_rel = False
                                        if host_total_now is not None and host_total_now > 0:
                                            host_low_rel = (
                                                float(host_avail_now) / float(host_total_now) < 0.1
                                            )
                                        host_low = host_low_abs or host_low_rel
                                    if host_low:
                                        if new_grad_accum > grad_accum_steps:
                                            new_grad_accum = grad_accum_steps
                                        if grad_accum_steps > min_grad_accum:
                                            new_grad_accum = min_grad_accum
                                    if new_grad_accum != grad_accum_steps:
                                        new_grad_accum = _sync_tensor(
                                            new_grad_accum, device=device, src=0
                                        )
                                        _LOGGER.info(
                                            "[epochs] adjusted grad_accum_steps=%d (gpu_util_ema=%s, mem_util_ema=%s)",
                                            int(new_grad_accum),
                                            str(gpu_util_ema),
                                            str(mem_util_ema),
                                        )
                                        grad_accum_steps = new_grad_accum
                            if use_timer and comp_ev_s is not None and comp_ev_e is not None:
                                comp_ev_e.record()
                                comp_ev_e.synchronize()
                                comp_time += float(comp_ev_s.elapsed_time(comp_ev_e)) / 1000.0
                            else:
                                if timer_sync:
                                    sync_accelerator(device)
                                comp_time += (time.perf_counter_ns() - t_comp_s) / 1000000000.0
                            if (
                                getattr(device, "type", None) == "cuda"
                                and bool(getattr(model, "_compile_cudagraphs", False))
                            ):
                                cudagraph_mark_step_end()
                            if local_rank == 0 and should_sync:
                                io_elapsed = prev_io_time + float(io_time)
                                io_transferred = prev_io_bytes + float(io_bytes)
                                comp_elapsed = prev_comp_time + float(comp_time)
                                flop_total = prev_flops + float(flops)
                                mbps_cur = io_transferred / max(io_elapsed, 1e-06) / MB_DIV
                                tflops_cur = flop_total / max(comp_elapsed, 1e-06) / 1000000000000.0
                                update_progress_bar(
                                    status_bar,
                                    finish=train_accum_since_last,
                                    mbps=mbps_cur,
                                    tflops=tflops_cur,
                                )
                                train_accum_since_last = 0
                            if isinstance(hist, Recorder):
                                try:
                                    if (
                                        train_steps <= 0
                                        or step_idx % max(1, int(train_steps * 0.01)) == 0
                                    ):
                                        x_rec = X
                                        y_rec = Y
                                        with contextlib.suppress(Exception):
                                            model_for_scaler = model.module if hasattr(model, "module") else model
                                            data_scaler = getattr(model_for_scaler, "scaler", None)
                                            if data_scaler is not None:
                                                dy = getattr(data_scaler, "denormalize_y", None)
                                                if callable(dy):
                                                    y_flat = y_rec
                                                    if isinstance(y_flat, torch.Tensor) and y_flat.ndim != 2:
                                                        y_flat = y_flat.reshape(y_flat.shape[0], -1)
                                                    need_denorm = True
                                                    scale_max = (
                                                        getattr(meta, "scale_max_value", None)
                                                        or getattr(meta, "scale_max_abs", None)
                                                    )
                                                    if isinstance(y_flat, torch.Tensor) and scale_max is not None:
                                                        with contextlib.suppress(Exception):
                                                            max_obs = float(y_flat.detach().abs().max().item())
                                                            if max_obs >= float(scale_max) * 0.5:
                                                                need_denorm = False
                                                    if need_denorm:
                                                        y_rec = dy(y_flat).view_as(y_rec)
                                        hist.record_batch(x_rec, y_rec)
                                except Exception:
                                    pass
                            if torch_prof is not None:
                                torch_prof.step()
                            t_fetch_start = time.perf_counter_ns()
                            if cpu_pool is not None and step_idx + 1 & 255 == 0:
                                with contextlib.suppress(Exception):
                                    cpu_pool.collect()
                            with contextlib.suppress(Exception):
                                _clear_oom_retries(train_loader, "train", step_idx)
                            break
                        except RuntimeError as e:
                            msg = str(e).lower()
                            if "out of memory" in msg:
                                decision, grad_accum_steps = _recover_oom(
                                    phase="train",
                                    loader=train_loader,
                                    step_idx=step_idx,
                                    global_step=global_step,
                                    device=device,
                                    model=model,
                                    optimizer=optimizer,
                                    grad_accum_steps=grad_accum_steps,
                                    min_grad_accum=min_grad_accum,
                                )
                                if decision == "retry":
                                    continue
                                if decision == "skip":
                                    break
                            raise
            if lw_count > 0:
                top_avg_t = lw_top_sum / float(lw_count) if lw_top_sum is not None else None
                bottom_avg_t = (
                    lw_bottom_sum / float(lw_count) if lw_bottom_sum is not None else None
                )
                if is_distributed():
                    ws = _get_world_size(train_pg)
                    if top_avg_t is not None:
                        with contextlib.suppress(Exception):
                            _reduce_sum(top_avg_t, train_pg)
                        top_avg_t = top_avg_t / float(ws)
                    if bottom_avg_t is not None:
                        with contextlib.suppress(Exception):
                            _reduce_sum(bottom_avg_t, train_pg)
                        bottom_avg_t = bottom_avg_t / float(ws)
                with contextlib.suppress(Exception):
                    loss_controller.update(top_avg_t, bottom_avg_t)
                lw_top_sum = None
                lw_bottom_sum = None
                lw_count = 0
            if val_loader is not None and flop_counter_val is not None:
                with flop_counter_val:
                    model.eval()
                    with inference_mode(model), Autocast.float(device):
                        t_fetch_start = time.perf_counter_ns()
                        for _vstep, _raw in enumerate(val_loader):
                            while True:
                                try:
                                    X, Y_opt, t_ready, h2d_s = _pin(
                                        meta,
                                        _raw,
                                        device=device,
                                        stage_tensor=stage_tensor,
                                        to_device=to_device_with_stream,
                                        cpu_pool=cpu_pool,
                                        use_timer=use_timer,
                                        timer_sync=timer_sync,
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
                                    wait_s = (t_ready - t_fetch_start) / 1000000000.0
                                    io_time += float(wait_s + h2d_s)
                                    with contextlib.suppress(Exception):
                                        io_bytes += float(
                                            X.element_size() * X.nelement()
                                            + Y.element_size() * Y.nelement()
                                        )
                                    if (
                                        use_timer
                                        and comp_ev_s is not None
                                        and comp_ev_e is not None
                                    ):
                                        comp_ev_s.record()
                                    else:
                                        t_comp_s = time.perf_counter_ns()
                                    with flop_counter_val.step(display=False) as val_counter:
                                        if (
                                            getattr(device, "type", None) == "cuda"
                                            and bool(getattr(model, "_compile_cudagraphs", False))
                                        ):
                                            cudagraph_mark_step_begin()
                                        with Autocast.float(device):
                                            Yv_flat = Y.reshape(Y.shape[0], -1).to(
                                                device,
                                                dtype=param_dtype,
                                                non_blocking=non_blocking_ok,
                                            )
                                            _y, _loss_val = model(
                                                X,
                                                labels_flat=Yv_flat,
                                                global_loss=top_loss,
                                                local_loss=bottom_loss,
                                                loss_weights=loss_controller.weights(),
                                                calibrate_output=False,
                                            )
                                        if (
                                            isinstance(_loss_val, torch.Tensor)
                                            and _loss_val.ndim > 0
                                        ):
                                            _loss_val = _loss_val.mean()
                                        if (
                                            getattr(device, "type", None) == "cuda"
                                            and bool(getattr(model, "_compile_cudagraphs", False))
                                        ):
                                            cudagraph_mark_step_end()
                                    if _loss_val is None:
                                        raise RuntimeError(
                                            "Model returned no loss value during validation. Ensure loss functions are configured correctly."
                                        )
                                    if not isinstance(_loss_val, torch.Tensor):
                                        _loss_val = torch.as_tensor(
                                            _loss_val, device=device, dtype=param_dtype
                                        )
                                    else:
                                        _loss_val = _loss_val.to(device=device, dtype=param_dtype)
                                    if (
                                        use_timer
                                        and comp_ev_s is not None
                                        and comp_ev_e is not None
                                    ):
                                        comp_ev_e.record()
                                        comp_ev_e.synchronize()
                                        comp_time += (
                                            float(comp_ev_s.elapsed_time(comp_ev_e)) / 1000.0
                                        )
                                    else:
                                        if timer_sync:
                                            sync_accelerator(device)
                                        comp_time += (
                                            time.perf_counter_ns() - t_comp_s
                                        ) / 1000000000.0
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
                                        mbps_cur = io_transferred / max(io_elapsed, 1e-06) / MB_DIV
                                        tflops_cur = (
                                            flop_total / max(comp_elapsed, 1e-06) / 1000000000000.0
                                        )
                                        update_progress_bar(
                                            status_bar, finish=1, mbps=mbps_cur, tflops=tflops_cur
                                        )
                                    if torch_prof is not None:
                                        torch_prof.step()
                                    t_fetch_start = time.perf_counter_ns()
                                    if cpu_pool is not None and _vstep + 1 & 255 == 0:
                                        with contextlib.suppress(Exception):
                                            cpu_pool.collect()
                                    with contextlib.suppress(Exception):
                                        _clear_oom_retries(val_loader, "val", _vstep)
                                    break
                                except RuntimeError as e:
                                    msg = str(e).lower()
                                    if "out of memory" in msg:
                                        decision, _ = _recover_oom(
                                            phase="val",
                                            loader=val_loader,
                                            step_idx=_vstep,
                                            device=device,
                                            model=model,
                                            optimizer=None,
                                            global_step=None,
                                            grad_accum_steps=None,
                                            min_grad_accum=1,
                                        )
                                        if decision == "retry":
                                            continue
                                        if decision == "skip":
                                            break
                                    raise
            if is_distributed():
                stats_dtype = (
                    param_dtype
                    if isinstance(param_dtype, torch.dtype) and param_dtype.is_floating_point
                    else torch.float64
                )
                stats = torch.tensor(
                    [comp_time, io_time, flops, io_bytes, train_samples_epoch],
                    device=device,
                    dtype=stats_dtype,
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
            if swa_enabled and epoch_idx >= swa_start_epoch:
                try:
                    swa_helper.update_weight()
                except Exception:
                    pass
            if not scheduler_step_per_batch:
                try:
                    sched.step()
                except Exception:
                    pass
            prev_comp_time += float(comp_time)
            prev_io_time += float(io_time)
            prev_flops += float(flops)
            prev_io_bytes += float(io_bytes)
            prev_samples += float(train_samples_epoch)
    model_for_scaler = model.module if hasattr(model, "module") else model
    scaler_y_device = model_for_scaler.scaler.y_mean.device
    with torch.no_grad():
        sum_x = None
        sum_y = None
        sum_x2 = None
        sum_xy = None
        total_n = 0
        for batch in train_loader:
            x_b, y_b = get_row(batch, labels_required=True)
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
            z_pred = z_pred_raw.detach().to(device=scaler_y_device, dtype=scaler_y_dtype)
            if z_pred.ndim >= 2:
                z_pred = z_pred.reshape(z_pred.shape[0], -1)
            else:
                z_pred = z_pred.view(-1, 1)
            z_true = model_for_scaler.scaler.normalize_y(y_flat.detach()).to(dtype=scaler_y_dtype)
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
                        f"Calibration: feature dimension mismatch between prediction and target that cannot be reconciled generically. z_pred.shape={tuple(z_pred.shape)}, z_true.shape={tuple(z_true.shape)}"
                    )
            if z_pred.shape[0] != z_true.shape[0]:
                raise RuntimeError(
                    f"Calibration: batch dimension mismatch between prediction and target. z_pred.shape={tuple(z_pred.shape)}, z_true.shape={tuple(z_true.shape)}"
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
            n_t = torch.tensor(float(total_n), device=scaler_y_device, dtype=scaler_y_dtype)
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
        if (
            total_n > 0
            and sum_x is not None
            and (sum_y is not None)
            and (sum_x2 is not None)
            and (sum_xy is not None)
        ):
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
            affine_dtype = model_for_scaler.scaler.affine_a.dtype
            a = (cov_xy / denom).to(dtype=affine_dtype)
            b = (mean_y - a.to(dtype=affine_dtype) * mean_x).to(dtype=affine_dtype)
            a[tiny_mask] = 1.0
            b[tiny_mask] = 0.0
            model_for_scaler.scaler.set_affine(a, b)
    if torch_prof is not None:
        with contextlib.suppress(Exception):
            torch_prof.stop()
        _get_profiler_summary(
            torch_prof, device=device, logger=_LOGGER, header="train/val"
        )
    if local_rank == 0 and status_bar is not None:
        mbps = prev_io_bytes / max(prev_io_time, 1e-06) / MB_DIV
        tflops = prev_flops / max(prev_comp_time, 1e-06) / 1000000000000.0
        status_bar.set_postfix_str(f"{mbps:.2f} MB/s, {tflops:.2f} TFLOPS", refresh=True)
        status_bar.close()
    end_kst_ns = posix_time()
    try:
        dev_t = getattr(device, "type", "")
        total_t = prev_io_time + prev_comp_time
        samples_per_sec = 0.0
        util_from_sps = 0.0
        if total_t > 0.0 and prev_samples > 0.0 and (prev_comp_time > 0.0):
            samples_per_sec = prev_samples / total_t
            max_samples_per_sec = prev_samples / prev_comp_time
            if max_samples_per_sec > 0.0:
                util_from_sps = samples_per_sec / max_samples_per_sec
        util_fallback = (
            util_from_sps
            if util_from_sps > 0.0
            else prev_comp_time / total_t
            if total_t > 0.0
            else 0.0
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
            try:
                if train_loader is not None:
                    scale_ctl = _get_sampler_scaler(train_loader)
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
                                    _is_scale_rate_logged(
                                        logger=_LOGGER,
                                        scale_ctl=scale_ctl,
                                        tag="auto-scale-down",
                                        msg="[epochs] auto scale_down (mem_util=%.3f): %.4f -> %.4f"
                                        % (float(mem_util_frac), float(prev), float(cur)),
                                        level="debug",
                                    )
                        elif util_for_cap < 0.9 and (mem_util_frac is None or mem_util_frac < 0.88):
                            prev = None
                            with contextlib.suppress(Exception):
                                prev = float(scale_ctl.get())
                            with contextlib.suppress(Exception):
                                scale_ctl.request_scale_up(1.1)
                            with contextlib.suppress(Exception):
                                cur = float(scale_ctl.get())
                                if prev is not None and cur > prev:
                                    _is_scale_rate_logged(
                                        logger=_LOGGER,
                                        scale_ctl=scale_ctl,
                                        tag="auto-scale-up",
                                        msg="[epochs] auto scale_up (util=%.3f): %.4f -> %.4f"
                                        % (float(util_for_cap), float(prev), float(cur)),
                                        level="debug",
                                    )
            except Exception:
                pass
        else:
            cpu_pct = _get_cpu_load()
            if cpu_pct is not None:
                if cpu_pct > 80.0:
                    time.sleep(min(0.005, 0.001 * (cpu_pct - 80.0)))
            elif util_fallback > 0.8:
                time.sleep(min(0.005, total_t * (util_fallback - 0.8)))
        if isinstance(hist, Recorder):
            try:
                end_sec = round(float(end_kst_ns) / 1000000000.0, 6)
                world = max(1, get_world_size(device)) if is_distributed() else 1
                hist.end_session(end_sec, peers=world)
                if ops.ckpt_dir and (
                    not is_distributed() or int(torch.distributed.get_rank()) == 0
                ):
                    history_path = os.path.join(ops.ckpt_dir, "history.json")
                    records = hist.save()
                    sampled_n_total = None
                    with contextlib.suppress(Exception):
                        sampled_n_total = int(round(float(prev_samples) * float(max(1, world))))
                    meta = {
                        "start_posix": float(round(float(hist.start.item()), 6)),
                        "end_posix": float(round(float(hist.end.item()), 6)),
                        "timezone": hist.timezone,
                        "peers": int(hist.peers.item()),
                        "epochs": int(hist.epochs.item()),
                        "sampled_n": int(sampled_n_total or 0),
                        "os": hist.os,
                        "kernel": hist.kernel,
                        "cpu": list(hist.cpu),
                        "arch": list(hist.arch),
                        "ram_gb": float(round(float(hist.ram_gb), 2)),
                        "python": hist.python,
                        "backends": list(hist.backends),
                    }
                    payload = {"meta": meta, "records": records}
                    Storage.write_json(history_path, payload, indent=2)
            except Exception:
                pass
    except Exception:
        pass


def infer(
    model: object,
    device: TorchDeviceLike,
    local_rank: int,
    ops: object,
    *args: Any,
    data_loader: object | None = None,
    chunk_dir: PathLike | None = None,
    dataset: object | None = None,
) -> object:
    _validate_compile_safe()
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
    torch_prof = None
    prof_enabled = _env_flag(
        "STNET_TORCH_PROFILE_INFER", _env_flag("STNET_TORCH_PROFILE", False)
    )
    prof_all_ranks = _env_flag("STNET_TORCH_PROFILE_ALL_RANKS", False)
    if prof_enabled and (prof_all_ranks or int(rank) == 0):
        prof_dir = env_str("STNET_TORCH_PROFILE_DIR")
        if not prof_dir:
            prof_dir = os.path.join(str(ops.ckpt_dir or "."), "torch_profiler")
        torch_prof = _get_torch_profiler(
            enabled=True, tag="infer", device=device, out_dir=str(prof_dir), rank=int(rank)
        )
        if torch_prof is not None:
            with contextlib.suppress(Exception):
                torch_prof.start()
    if rank == 0:
        os.makedirs(chunk_dir, exist_ok=True)
    distributed_barrier(device)
    _nogil_opt = False
    with contextlib.suppress(Exception):
        from ..core.system import Thread
        _nogil_opt = bool(Thread.is_optimized_for_no_gil())
    _cache_default = 16 if _nogil_opt else 4
    cache_q = max(
        1,
        int(
            env_first_int(
                (
                    "STNET_PRED_CACHE_MAX_QUEUE",
                    "STNET_PRED_WRITE_QUEUE",
                    "STNET_CACHE_MAX_QUEUE",
                ),
                default=_cache_default,
            )
        ),
    )
    dev_type = str(getattr(device, "type", "cpu"))
    use_async_write = bool(_env_flag("STNET_PRED_ASYNC_WRITE", True))
    use_mmt_pred_parts = bool(_env_flag("STNET_PRED_MMT_PARTS", dev_type != "cpu"))
    if not use_async_write:
        use_mmt_pred_parts = False
    cache = Cache(chunk_dir, max_queue=cache_q) if use_async_write else None
    target_rows = int(_env_int("STNET_PRED_CHUNK_ROWS", 0))
    if target_rows <= 0:
        out_shape = tuple((int(x) for x in ops.out_shape or ()))
        out_numel = 1
        for d in out_shape:
            out_numel *= max(1, int(d))
        est_row_bytes = max(1, out_numel * 4)
        target_bytes = int(_env_int("STNET_PRED_CHUNK_BYTES", 64 * 1024 * 1024))
        target_rows = max(256, min(65536, target_bytes // est_row_bytes))
    run_model = to_ddp(model, device=device)
    run_model.eval()
    module_eval = run_model.module if hasattr(run_model, "module") else run_model
    distributed_sync(module_eval, device=device)
    non_blocking_ok = bool(dev_type in ("cuda", "xpu"))
    pinned_ok = bool(is_pin_supported(dev_type))
    backend = accelerator_type(dev_type)
    stream_fn = getattr(backend, "current_stream", None) if backend is not None else None
    Event = getattr(backend, "Event", None) if backend is not None else None
    can_stream_release = bool(
        pinned_ok and non_blocking_ok and callable(stream_fn) and (Event is not None)
    )
    make_fence_event = partial(new_accelerator_event, device, enable_timing=False)
    cpu_pool = None
    if pinned_ok and Pool is not None:
        with contextlib.suppress(Exception):
            Memory.prefer_local_numa()
        with contextlib.suppress(Exception):
            _nogil = False
            with contextlib.suppress(Exception):
                from ..core.system import Thread
                _nogil = bool(Thread.is_optimized_for_no_gil())
            _cpu_default = 8
            if _nogil:
                try:
                    _cpu_default = max(8, min(64, int(os.cpu_count() or 8)))
                except Exception:
                    _cpu_default = 16
            cpu_pool_cap = max(2, int(_env_int("STNET_RUNTIME_PIN_POOL_CAPACITY", _cpu_default)))
            cpu_pool = Pool(capacity=cpu_pool_cap)
    pred_pool = None
    if non_blocking_ok and Pool is not None and _env_flag("STNET_PRED_PINNED", True):
        with contextlib.suppress(Exception):
            _nogil = False
            with contextlib.suppress(Exception):
                from ..core.system import Thread
                _nogil = bool(Thread.is_optimized_for_no_gil())
            _pred_default = 2 if not _nogil else 4
            pred_pool_cap = max(2, int(_env_int("STNET_PRED_PIN_POOL_CAPACITY", _pred_default)))
            pred_pool = Pool(capacity=pred_pool_cap, pin_memory=True)
    stage_tensor = partial(
        _pool_tensor,
        device=device,
        dev_type=dev_type,
        pinned_ok=pinned_ok,
        cpu_pool=cpu_pool,
    )
    to_device_with_stream = partial(
        _stream_tensor,
        device=device,
        dev_type=dev_type,
        non_blocking_ok=non_blocking_ok,
        backend=backend,
        stream_fn=stream_fn,
        Event=Event,
        fence_event_factory=make_fence_event,
        can_stream_release=can_stream_release,
        cpu_pool=cpu_pool,
    )
    status_bar = (
        get_progress_bar(title="Prediction", total=_get_batch_length(data_loader), device=device, leave=False)
        if local_rank == 0
        else None
    )
    row_cursor = 0
    writer = _PredictionWriter(
        chunk_dir=str(chunk_dir),
        rank=int(rank),
        use_mmt_pred_parts=bool(use_mmt_pred_parts),
        cache=cache,
        pred_pool=pred_pool,
        target_rows=int(target_rows),
        make_fence_event=make_fence_event,
    )

    try:
        with inference_mode(run_model), Autocast.float(device):
            row_ids_buf = None
            for batch in data_loader:
                if batch is None:
                    if status_bar is not None:
                        status_bar.update(1)
                    continue
                row_ids = None
                try:
                    if isinstance(batch, TensorDictBase):
                        row_ids = batch.get("row_ids", None)
                    elif isinstance(batch, dict):
                        row_ids = batch.get("row_ids", None)
                except Exception:
                    row_ids = None
                try:
                    X, _Y, _t_ready, _h2d_s = _pin(
                        dataset,
                        batch,
                        device=device,
                        stage_tensor=stage_tensor,
                        to_device=to_device_with_stream,
                        cpu_pool=cpu_pool,
                        use_timer=False,
                        require_labels=False,
                    )
                except Exception:
                    raise
                X = torch.atleast_2d(X)
                bs = int(getattr(X, "shape", [0])[0]) if hasattr(X, "shape") else 0
                if bs <= 0:
                    if status_bar is not None:
                        status_bar.update(1)
                    continue
                if row_ids is None:
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
                    raise RuntimeError(
                        f"infer: row_ids length mismatch: row_ids={row_ids.numel()} vs batch={bs}"
                    )
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
                        y_hat = out[0]
                    else:
                        y_hat = out
                    if not isinstance(y_hat, torch.Tensor):
                        raise RuntimeError("infer: unexpected model output type")
                    preds = y_hat.detach()
                    rows_cpu = rows_i if rows_i.device.type == "cpu" else rows_i.to(device="cpu")
                    writer.append(rows_cpu, preds)
                    del Xi, rows_i, out, y_hat, preds, rows_cpu
                    start = end
                if torch_prof is not None:
                    torch_prof.step()
                if status_bar is not None:
                    status_bar.update(1)
                del X, _Y, batch, row_ids
    finally:
        writer.flush()
        if torch_prof is not None:
            with contextlib.suppress(Exception):
                torch_prof.stop()
            _get_profiler_summary(
                torch_prof, device=device, logger=_LOGGER, header="infer"
            )
        if cache is not None:
            cache.close()
        exc_type, _, _ = sys.exc_info()
        if exc_type is None and cache is not None:
            had_error = False
            with contextlib.suppress(Exception):
                had_error = bool(getattr(cache, "had_error", None) and cache.had_error())
            if had_error:
                err = getattr(cache, "_err", None)
                if isinstance(err, BaseException):
                    raise RuntimeError(
                        f"infer: prediction writer encountered an error: {type(err).__name__}: {err}"
                    ) from err
                raise RuntimeError("infer: prediction writer encountered an error")
        if status_bar is not None:
            status_bar.close()
        with contextlib.suppress(Exception):
            distributed_barrier(device)
        if exc_type is None and rank == 0:
            parts = []
            for rows_path in sorted(glob.glob(os.path.join(chunk_dir, "part-r*-c*-rows.pt"))):
                base = rows_path[: -len("-rows.pt")]
                pred_mmt = base + "-pred.mmt"
                pred_pt = base + "-pred.pt"
                if os.path.exists(pred_mmt):
                    pred_path = pred_mmt
                    meta_path = Storage.mmt_meta_path(pred_mmt)
                    if not os.path.exists(meta_path):
                        raise RuntimeError(
                            f"infer: missing pred meta for memmap part: {pred_mmt} -> {meta_path}"
                        )
                elif os.path.exists(pred_pt):
                    pred_path = pred_pt
                else:
                    raise RuntimeError(
                        f"infer: missing pred file for rows part: {rows_path} -> ({pred_mmt} or {pred_pt})"
                    )
                parts.append(
                    {"rows": os.path.basename(rows_path), "pred": os.path.basename(pred_path)}
                )
            if not parts:
                raise RuntimeError(f"infer: no prediction parts produced in {chunk_dir}")
            manifest = {
                "format": "stnet.pred.v2",
                "rank_count": int(world_size),
                "out_shape": list((int(x) for x in ops.out_shape or ())),
                "variable_shape": bool(writer.variable_shape),
                "parts": parts,
            }
            man_path = os.path.join(chunk_dir, "manifest.json")
            Storage.write_json(man_path, manifest, indent=2)
        if exc_type is None:
            with contextlib.suppress(Exception):
                distributed_barrier(device)
    return None


@overload
def process(ops: RuntimeConfig, ret_sink: ReturnSink | None = None) -> object: ...


@overload
def process(local_rank: int, ops: RuntimeConfig, ret_sink: ReturnSink | None = None) -> object: ...


def process(*args: Any, **kwargs: Any) -> object:
    from ..data.pipeline import Session
    if not args:
        raise TypeError("process requires at least a RuntimeConfig argument")
    init_python_path()
    _validate_compile_safe()
    ret_sink = None
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
            "process expects (RuntimeConfig,), (RuntimeConfig, ret_sink), (local_rank, RuntimeConfig), or (local_rank, RuntimeConfig, ret_sink) arguments"
        )
    verbose = bool(getattr(ops, "verbose", False))
    det = bool(getattr(ops, "deterministic", False))
    seed_base = int(getattr(ops, "seed", 42))
    seed_value = int(seed_base) + int(local_rank)
    with contextlib.suppress(Exception):
        warnings.filterwarnings(
            "ignore",
            message=_IGNORED_WARNING_MESSAGE_RE.pattern,
            category=UserWarning,
        )
    with contextlib.suppress(Exception):
        random.seed(seed_value)
    with contextlib.suppress(Exception):
        numpy.random.seed(seed_value)
    with contextlib.suppress(Exception):
        torch.manual_seed(seed_value)
    with contextlib.suppress(Exception):
        set_accelerator_seed(seed_value)
    with contextlib.suppress(Exception):
        torch.use_deterministic_algorithms(det, warn_only=False)
    with contextlib.suppress(Exception):
        torch.backends.cudnn.deterministic = det
        torch.backends.cudnn.benchmark = not det
    if ops.mode == "train":
        with contextlib.suppress(Exception):
            if is_accelerator_available("cuda"):
                n = max(1, int(get_num_accelerators("cuda") or 1))
                set_accelerator_index("cuda", int(local_rank) % int(n))
            elif is_accelerator_available("xpu"):
                n = max(1, int(get_num_accelerators("xpu") or 1))
                set_accelerator_index("xpu", int(local_rank) % int(n))
        device = get_device()
        _init_backend(device)
        backend = _get_backend_type(device)
        enable_tf32 = bool(getattr(ops, "enable_tf32", True))
        _init_distributed_group(backend, device, local_rank)
        cfg = coerce_model_config(ops.cfg_dict if isinstance(ops.cfg_dict, dict) else ops.cfg_dict)
        cfg = replace(cfg, device=device)
        model = Model(ops.in_dim, ops.out_shape, config=cfg)
        if ops.init_ckpt_dir is not None and os.path.isdir(ops.init_ckpt_dir):
            fallback_init = os.path.join(ops.init_ckpt_dir, "model.pt")
            if os.path.isfile(fallback_init):
                cpu_state = _torch_load_checkpoint(
                    fallback_init, map_location="cpu", weights_only=True
                )
                resize_scaler_buffer(model, cpu_state)
                model.load_state_dict(cpu_state, strict=False)
            else:
                m_sd = get_model_state_dict(
                    model, options=StateDictOptions(full_state_dict=True, cpu_offload=False)
                )
                m_sd = _coerce_dcp_keys(m_sd)
                with _filtered_warnings():
                    load(state_dict={"model": m_sd}, storage_reader=FileSystemReader(ops.init_ckpt_dir))
                resize_scaler_buffer(model, m_sd)
                set_model_state_dict(model, m_sd, options=StateDictOptions(strict=False))
        if ops.sources is None:
            raise RuntimeError("RuntimeConfig.sources is required but None")
        metadata = Dataset.for_device(device)
        expanded_sources = Storage.expand_source(ops.sources)
        if expanded_sources is not ops.sources:
            ops = replace(ops, sources=expanded_sources)
        meta_info = Storage.merge_meta_info(ops.sources)
        meta_feature_dim = int(meta_info.get("feature_dim", ops.in_dim))
        if meta_feature_dim != int(ops.in_dim):
            raise RuntimeError(
                f"dataset feature_dim mismatch: meta={meta_feature_dim}, expected in_dim={ops.in_dim}"
            )
        meta_label_shape = tuple(
            (int(x) for x in meta_info.get("label_shape", list(ops.out_shape)))
        )
        if tuple(meta_label_shape) != tuple(ops.out_shape):
            raise RuntimeError(
                f"dataset label_shape mismatch: meta={meta_label_shape}, expected out_shape={tuple(ops.out_shape)}"
            )
        fractions = meta_info.get("fractions", [1.0, 0.0])
        if isinstance(fractions, (list, tuple)) and len(fractions) >= 2:
            actual_val_frac = float(fractions[-1])
            if not math.isclose(actual_val_frac, float(ops.val_frac), rel_tol=0.001, abs_tol=0.001):
                warnings.warn(
                    "val_frac=%s differs from memmap metadata (%s); using metadata value for loaders"
                    % (ops.val_frac, actual_val_frac)
                )
                ops = replace(ops, val_frac=actual_val_frac)
        metadata.has_scale = bool(
            meta_info.get("has_scale", False)
            or meta_info.get("scale_max_abs") is not None
            or meta_info.get("scale_min_value") is not None
            or (meta_info.get("scale_max_value") is not None)
            or (meta_info.get("scale_min_positive") is not None)
            or (meta_info.get("scale_min_abs") is not None)
        )
        metadata.has_nonfinite = bool(meta_info.get("has_nonfinite", False))
        metadata.scale_max_abs = meta_info.get("scale_max_abs")
        metadata.scale_min_value = meta_info.get("scale_min_value")
        metadata.scale_max_value = meta_info.get("scale_max_value")
        metadata.scale_min_positive = meta_info.get("scale_min_positive") or meta_info.get(
            "scale_min_abs"
        )
        metadata.scale_is_integral = meta_info.get("scale_is_integral")
        if meta_info.get("is_negotiable") is not None:
            metadata.is_negotiable = bool(meta_info.get("is_negotiable"))
        if meta_info.get("underflow_action") is not None:
            metadata.underflow_action = str(meta_info.get("underflow_action"))
        feat_dtype_name = str(meta_info.get("features_dtype", "")).lower()
        lab_dtype_name = str(meta_info.get("labels_dtype", "")).lower()
        if "float64" in feat_dtype_name or "float64" in lab_dtype_name:
            metadata.is_negotiable = False
        precision = PrecisionPolicy.from_metadata(device=device, metadata=metadata, logger=_LOGGER)
        param_dtype = precision.master_float
        if device.type != "cuda":
            _LOGGER.warning(
                "Forcing CPU / non-CUDA config: mixed precision + NVIDIA fused layers may be unavailable."
            )
        model, _, _ = ModelPolicy.use_nvidia_layers(
            model, device=device, metadata=metadata, params_dtype=param_dtype, verbose=verbose
        )
        Autocast.configure(model, metadata=metadata)
        set_float32_precision(
            device=device,
            autocast_dtype=precision.amp_float or param_dtype,
            enable_tf32=enable_tf32,
        )
        fp8_ok, fp8_reason = Dataset.is_float8_supported(device)
        fp8_enabled = False
        fp8_backend = None
        disable_note = None
        if param_dtype is torch.float64:
            disable_note = "master dtype is float64"
        elif fp8_ok:
            model, fp8_enabled, fp8_backend = ModelPolicy.enable_float8_training(
                model, metadata=metadata, logger=_float8_log
            )
            if not fp8_enabled:
                disable_note = fp8_backend or fp8_reason
        else:
            disable_note = fp8_reason
        if not fp8_enabled:
            Autocast.configure(model, metadata=metadata)
        if disable_note:
            _float8_log(f"[FP8] disabled: {disable_note}")
        _cast_float_dtype(model, param_dtype)
        model.train()
        mesh = None
        fsdp_mp_dtype = precision.fsdp_reduce_dtype
        if device.type == "cpu" and fsdp_mp_dtype is not torch.float64:
            fsdp_mp_dtype = torch.float32
        amp_buffers_dtype = precision.bn_buffers_dtype
        mp_policy = None
        if MixedPrecisionPolicy is not None:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=fsdp_mp_dtype,
                output_dtype=fsdp_mp_dtype,
                cast_forward_inputs=False,
            )
        elif verbose:
            _LOGGER.warning(
                "MixedPrecisionPolicy is not available in this PyTorch build; "
                "continuing without explicit FSDP mixed-precision policy."
            )
        _m_pre = model.module if hasattr(model, "module") else model
        _preload_layers(_m_pre, device)
        _validate_model_dtype_unity(_m_pre, device)
        _validate_no_meta_tensors(_m_pre)
        _validate_no_fake_dtensor(_m_pre)
        wrapped = set()
        try:
            for submodule in _get_layers(getattr(model, "processor", None)):
                _get_fsdp_module(submodule, mesh, mp_policy, reshard_after_forward=True, wrapped=wrapped)
            for submodule in _get_layers(getattr(model, "controller", None)):
                _get_fsdp_module(submodule, mesh, mp_policy, reshard_after_forward=True, wrapped=wrapped)
            model = (
                _get_fsdp_module(model, mesh, mp_policy, reshard_after_forward=True, wrapped=wrapped)
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
        _validate_model_dtype_unity(_m_post, device)
        _validate_no_meta_tensors(_m_post)
        _validate_no_fake_dtensor(_m_post)
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
        _init_optimizer(optimizer)
        top_df = DataFidelityLoss(out_shape=ops.out_shape, reduction="mean")
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
        local_crps = CRPSLoss(dim=-1, reduction="none", detach_stats=True)
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
            coefficient=[1.0, 0.0], loss=[top_df, top_z], reduce_each=True, auto_schedule=True
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
            coefficient=[1.0, 0.0],
            loss=[local_crps, local_t],
            reduce_each=False,
            auto_schedule=True,
        )
        loss_controller = LossWeightController(top_avg=0.5, bottom_avg=0.5)
        ckpt_state_path = get_loader_state(ops.ckpt_dir or "")
        init_state_path = get_loader_state(ops.init_ckpt_dir) if ops.init_ckpt_dir else None
        state_train = {}
        state_val = {}
        _dlp = (
            ckpt_state_path
            if os.path.isfile(ckpt_state_path)
            else init_state_path
            if init_state_path and os.path.isfile(init_state_path)
            else None
        )
        restore_dl_state = False
        if _dlp:
            with contextlib.suppress(Exception):
                _dl_json = read_json(_dlp)
                if isinstance(_dl_json, dict):
                    state_train = _dl_json.get("train", {}) or {}
                    state_val = _dl_json.get("val", {}) or {}
                    restore_dl_state = bool(state_train) or bool(state_val)
        train_loader = None
        val_loader = None
        raw_train_loader = None
        raw_val_loader = None
        session = None
        try:
            expanded_sources = Storage.expand_source(ops.sources)
            if expanded_sources is not ops.sources:
                ops = replace(ops, sources=expanded_sources)
            accelerator_types = {"cuda", "xpu", "mps"}
            device_type = getattr(device, "type", None)
            if not device_type:
                device_str = str(device)
                device_type = device_str.split(":", 1)[0]
            non_blocking_copy = device_type in accelerator_types
            with contextlib.suppress(Exception):
                _get_sample_size(
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
                train_weights=getattr(ops, "train_weights", None),
                val_weights=getattr(ops, "val_weights", None),
                labels_dtype=param_dtype,
            ).open(
                train_state=state_train if restore_dl_state else None,
                val_state=state_val if restore_dl_state else None,
            )
            train_loader = session.training_loader
            val_loader = session.validation_loader
            raw_train_loader = session.raw_training_loader
            raw_val_loader = session.raw_validation_loader
            train_steps = _get_batch_length(train_loader)
            val_steps = _get_batch_length(val_loader)
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
                _schedule,
                warmup_steps=warmup_steps,
                start_factor=start_factor,
                base=base,
                main_steps=main_steps,
                emin=emin,
            )
            sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            scheduler_step_per_batch = True
            swa_helper = None
            swa_start_epoch = total_epochs
            enable_swa_cfg = bool(getattr(ops, "swa_enabled", False))
            start_epoch_cfg = getattr(ops, "swa_start_epoch", None)
            enable_swa = (enable_swa_cfg or start_epoch_cfg is not None) and SWALR is not None
            if enable_swa:
                tracked_module = model.module if hasattr(model, "module") else model
                use_buffers = True
                try:
                    swa_helper = stochastic_weight_average(tracked_module, use_buffers=use_buffers)
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
                    default_swa_lr = max(1e-08, eta_min if eta_min > 0.0 else 0.1 * base_lr)
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
            amp_dtype = getattr(precision, "amp_float", None)
            compute_dtype = amp_dtype or param_dtype
            scaler = torch.amp.GradScaler(
                enabled=bool(device.type == "cuda" and compute_dtype == torch.float16)
            )
            try:
                get_affinity().pin_thread()
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
                model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
            )
            optim_sd = get_optimizer_state_dict(model, optimizers=optimizer)
            writer = FileSystemWriter(ops.ckpt_dir or "", sync_files=True, overwrite=True)
            with _filtered_warnings():
                save(state_dict={"model": model_sd, "optimizer": optim_sd}, storage_writer=writer)
            if ops.ckpt_dir:
                fallback_path = os.path.join(ops.ckpt_dir, "model.pt")
                model_fallback = dict(model_sd)
                _coerce_dcp_keys(model_fallback)
                torch.save(model_fallback, fallback_path)
                with contextlib.suppress(Exception):
                    _dl = {
                        "train": raw_train_loader.state_dict()
                        if raw_train_loader is not None
                        else {},
                        "val": raw_val_loader.state_dict() if raw_val_loader is not None else {},
                    }
                    Storage.write_json(get_loader_state(ops.ckpt_dir or ""), _dl, indent=2)
        torch.distributed.barrier(
            device_ids=[local_rank] if device.type in ("cuda", "xpu") else None
        )
        torch.distributed.destroy_process_group()
        return None
    if ops.mode in ("predict", "infer"):
        with contextlib.suppress(Exception):
            if is_accelerator_available("cuda"):
                n = max(1, int(get_num_accelerators("cuda") or 1))
                set_accelerator_index("cuda", int(local_rank) % int(n))
            elif is_accelerator_available("xpu"):
                n = max(1, int(get_num_accelerators("xpu") or 1))
                set_accelerator_index("xpu", int(local_rank) % int(n))
        device = get_device()
        _init_backend(device)
        backend = _get_backend_type(device)
        if not torch.distributed.is_initialized():
            _init_distributed_group(backend, device, local_rank)
        cfg = coerce_model_config(ops.cfg_dict if isinstance(ops.cfg_dict, dict) else ops.cfg_dict)
        cfg = replace(cfg, device=device)
        if _env_flag("STNET_MAX_PERF", True):
            cm = str(getattr(cfg, "compile_mode", "disabled") or "").strip().lower()
            if cm in ("", "none", "disabled", "disable", "off", "false", "0"):
                default_cm = env_str("STNET_SERVE_COMPILE_MODE")
                if not default_cm:
                    default_cm = "max-autotune" if getattr(device, "type", None) == "cuda" else "max-autotune-no-cudagraphs"
                cfg = replace(cfg, compile_mode=str(default_cm))
                with contextlib.suppress(Exception):
                    setattr(cfg, "compile_heavy_submodules", bool(_env_flag("STNET_COMPILE_HEAVY", True)))
                with contextlib.suppress(Exception):
                    setattr(cfg, "compile_dynamic", bool(_env_flag("STNET_COMPILE_DYNAMIC", False)))
                with contextlib.suppress(Exception):
                    setattr(cfg, "compile_cudagraphs", bool(_env_flag("STNET_COMPILE_CUDAGRAPHS", True)))
        model = Model(ops.in_dim, ops.out_shape, config=cfg)
        if not ops.model_ckpt_dir:
            raise RuntimeError(
                "predict/infer requires model_ckpt_dir (checkpoint directory). Set RuntimeConfig.model_ckpt_dir to a directory produced by train()."
            )
        if not os.path.isdir(ops.model_ckpt_dir):
            raise RuntimeError(
                f"predict/infer: model_ckpt_dir does not exist or is not a directory: {ops.model_ckpt_dir!r}"
            )
        if ops.model_ckpt_dir is not None and os.path.isdir(ops.model_ckpt_dir):
            fallback_model = os.path.join(ops.model_ckpt_dir, "model.pt")
            if os.path.isfile(fallback_model):
                cpu_state = _torch_load_checkpoint(
                    fallback_model, map_location="cpu", weights_only=True
                )
                resize_scaler_buffer(model, cpu_state)
                model.load_state_dict(cpu_state, strict=False)
            else:
                m_sd = get_model_state_dict(
                    model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
                )
                m_sd = _coerce_dcp_keys(m_sd)
                load(
                    state_dict={"model": m_sd}, storage_reader=FileSystemReader(ops.model_ckpt_dir)
                )
                resize_scaler_buffer(model, m_sd)
                set_model_state_dict(model, m_sd, options=StateDictOptions(strict=False))
        model.to(device, non_blocking=device.type in ("cuda", "xpu")).eval()
        metadata = Dataset.for_device(device)
        model, _, _ = ModelPolicy.use_nvidia_layers(model, device=device)
        _m_eval = model.module if hasattr(model, "module") else model
        _preload_layers(_m_eval, device)
        _validate_model_dtype_unity(_m_eval, device)
        _validate_no_meta_tensors(_m_eval)
        _validate_no_fake_dtensor(_m_eval)
        _enable_meta_monitor(_m_eval)
        _unify_model_dtype(
            model,
            prefer=torch.bfloat16
            if getattr(device, "type", None) == "cuda" and is_cuda_bf16_supported(device)
            else None,
        )
        Autocast.configure(model, metadata=metadata)
        enable_tf32 = bool(getattr(ops, "enable_tf32", True))
        with contextlib.suppress(Exception):
            param_dtype = next(
                (p.dtype for p in (model.module if hasattr(model, "module") else model).parameters()),
                None,
            )
        with contextlib.suppress(Exception):
            set_float32_precision(
                device=device,
                dtype=param_dtype,
                autocast_dtype=param_dtype,
                enable_tf32=enable_tf32,
            )
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
            _get_sample_size(
                model=model, device=device, ops=ops, dataset=metadata, with_backward=False
            )
        expanded_sources = Storage.expand_source(ops.sources)
        if expanded_sources is not ops.sources:
            ops = replace(ops, sources=expanded_sources)
        session = None
        session = Session(
            sources=ops.sources,
            device=device,
            val_frac=0.0,
            non_blocking_copy=True,
            sanitize=True,
            flatten_features=True,
            train_shuffle=bool(getattr(ops, "shuffle", False)),
            seed=int(getattr(ops, "seed", 7)),
            train_weights=getattr(ops, "train_weights", None),
            val_weights=getattr(ops, "val_weights", None),
        ).open()
        data_loader = session.training_loader
        chunk_dir = os.path.join(ops.ckpt_dir, "pred_chunks") if ops.ckpt_dir or "" else None
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


class _PredictionWriter:
    __slots__ = (
        "chunk_dir",
        "rank",
        "use_mmt_pred_parts",
        "cache",
        "pred_pool",
        "target_rows",
        "make_fence_event",
        "use_buffer",
        "rows_buf",
        "pred_buf",
        "pred_handle",
        "pred_buf_is_pinned",
        "buf_needs_wait_evt",
        "buf_fill",
        "pending_rows",
        "pending_preds",
        "pending_count",
        "chunk_idx",
        "first_tail",
        "pending_tail",
        "variable_shape",
    )

    def __init__(
        self,
        *args: Any,
        chunk_dir: str,
        rank: int,
        use_mmt_pred_parts: bool,
        cache: Cache | None,
        pred_pool: Pool | None,
        target_rows: int,
        make_fence_event: callable,
    ) -> None:
        self.chunk_dir = str(chunk_dir)
        self.rank = int(rank)
        self.use_mmt_pred_parts = bool(use_mmt_pred_parts)
        self.cache = cache
        self.pred_pool = pred_pool
        self.target_rows = max(1, int(target_rows))
        self.make_fence_event = make_fence_event
        self.use_buffer = True
        self.rows_buf: torch.Tensor | None = None
        self.pred_buf: torch.Tensor | None = None
        self.pred_handle: Pool.Token | None = None
        self.pred_buf_is_pinned = False
        self.buf_needs_wait_evt = False
        self.buf_fill = 0
        self.pending_rows: list[torch.Tensor] = []
        self.pending_preds: list[torch.Tensor] = []
        self.pending_count = 0
        self.chunk_idx = 0
        self.first_tail: tuple[int, ...] | None = None
        self.pending_tail: tuple[int, ...] | None = None
        self.variable_shape = False

    def flush(self) -> None:
        if self.use_buffer:
            if self.buf_fill <= 0:
                return
            if self.rows_buf is None or self.pred_buf is None:
                return
            rows = self.rows_buf[: self.buf_fill].clone()
            preds = self.pred_buf[: self.buf_fill]
            local_handle = self.pred_handle
            need_wait_evt = bool(self.buf_needs_wait_evt)
            self.buf_fill = 0
            self.pred_buf = None
            self.pred_handle = None
            self.pred_buf_is_pinned = False
            self.buf_needs_wait_evt = False
        else:
            if self.pending_count <= 0:
                return
            rows = torch.cat(self.pending_rows, dim=0).to(dtype=torch.int64, copy=False)
            if not bool(rows.is_contiguous()):
                rows = rows.contiguous()
            preds = torch.cat(self.pending_preds, dim=0)
            if not bool(preds.is_contiguous()):
                preds = preds.contiguous()
            local_handle = None
            need_wait_evt = False

        rows_path = os.path.join(
            self.chunk_dir, f"part-r{self.rank:05d}-c{self.chunk_idx:06d}-rows.pt"
        )
        if self.use_buffer and self.use_mmt_pred_parts:
            pred_path = os.path.join(
                self.chunk_dir, f"part-r{self.rank:05d}-c{self.chunk_idx:06d}-pred.mmt"
            )
        else:
            pred_path = os.path.join(
                self.chunk_dir, f"part-r{self.rank:05d}-c{self.chunk_idx:06d}-pred.pt"
            )

        if self.cache is not None:
            self.cache.submit(rows, path=rows_path)
        else:
            Storage.atomic_torch_save(rows_path, rows)

        wait_evt = None
        release_cb = None
        if self.use_buffer:
            if need_wait_evt:
                try:
                    if local_handle is not None and self.pred_pool is not None:
                        fe = getattr(self.pred_pool, "fence_event", None)
                        if callable(fe):
                            wait_evt = fe(local_handle, self.make_fence_event)
                    if wait_evt is None:
                        wait_evt = self.make_fence_event()
                    if wait_evt is not None:
                        with contextlib.suppress(Exception):
                            wait_evt.record()
                except Exception:
                    wait_evt = None

            if local_handle is not None and self.pred_pool is not None:
                release_cb = partial(self.pred_pool.release, local_handle)

        if self.cache is not None:
            self.cache.submit(preds, path=pred_path, wait_event=wait_evt, release_cb=release_cb)
        else:
            preds_cpu = preds.detach()
            if getattr(preds_cpu, "device", None) is not None and preds_cpu.device.type != "cpu":
                preds_cpu = preds_cpu.to(device="cpu")
            Storage.atomic_torch_save(pred_path, preds_cpu)
            if release_cb is not None:
                with contextlib.suppress(Exception):
                    release_cb()

        self.chunk_idx += 1
        if not self.use_buffer:
            self.pending_rows.clear()
            self.pending_preds.clear()
            self.pending_count = 0
            self.pending_tail = None
        del rows, preds

    def append(self, rows_cpu: object, preds: object) -> None:
        if not isinstance(preds, torch.Tensor) or preds.ndim < 1:
            return
        b = int(preds.shape[0])
        if b <= 0:
            return
        if not isinstance(rows_cpu, torch.Tensor):
            rows_cpu = torch.as_tensor(rows_cpu, dtype=torch.int64)
        rows_cpu = rows_cpu.reshape(-1).to(dtype=torch.int64, device="cpu", copy=False)
        if rows_cpu.numel() != b:
            raise RuntimeError(
                f"infer: rows/preds batch mismatch rows={rows_cpu.numel()} preds={b}"
            )
        preds = preds.detach()
        tail = tuple((int(x) for x in preds.shape[1:]))
        if self.first_tail is None:
            self.first_tail = tail
        elif tail != self.first_tail:
            self.variable_shape = True
            if self.use_buffer:
                self.flush()
                self.use_buffer = False
                self.pending_tail = None

        if not self.use_buffer:
            if self.pending_tail is None:
                self.pending_tail = tail
            elif tail != self.pending_tail:
                self.flush()
                self.pending_tail = tail
            self.pending_rows.append(rows_cpu.clone())
            if preds.device.type != "cpu":
                preds = preds.to(device="cpu")
            self.pending_preds.append(preds)
            self.pending_count += b
            if self.pending_count >= self.target_rows:
                self.flush()
            return

        if self.rows_buf is None:
            self.rows_buf = torch.empty((self.target_rows,), dtype=torch.int64)

        if self.pred_buf is None:
            if self.pred_pool is not None:
                self.pred_buf, self.pred_handle = self.pred_pool.get(
                    (self.target_rows, *tail), dtype=preds.dtype, return_handle=True
                )
            else:
                self.pred_buf = torch.empty((self.target_rows, *tail), dtype=preds.dtype)
                self.pred_handle = None
            self.pred_buf_is_pinned = False
            with contextlib.suppress(Exception):
                is_pinned = getattr(self.pred_buf, "is_pinned", None)
                if callable(is_pinned):
                    self.pred_buf_is_pinned = bool(is_pinned())
        elif self.pred_buf.dtype != preds.dtype or tuple((int(x) for x in self.pred_buf.shape[1:])) != tail:
            if self.buf_fill > 0:
                self.flush()
            if self.pred_pool is not None and self.pred_handle is not None:
                with contextlib.suppress(Exception):
                    self.pred_pool.release(self.pred_handle)
            self.pred_buf = None
            self.pred_handle = None
            self.pred_buf_is_pinned = False

        start = 0
        while start < b:
            if self.pred_buf is None:
                if self.pred_pool is not None:
                    self.pred_buf, self.pred_handle = self.pred_pool.get(
                        (self.target_rows, *tail), dtype=preds.dtype, return_handle=True
                    )
                else:
                    self.pred_buf = torch.empty((self.target_rows, *tail), dtype=preds.dtype)
                    self.pred_handle = None
                self.pred_buf_is_pinned = False
                with contextlib.suppress(Exception):
                    is_pinned = getattr(self.pred_buf, "is_pinned", None)
                    if callable(is_pinned):
                        self.pred_buf_is_pinned = bool(is_pinned())
            space = int(self.target_rows) - int(self.buf_fill)
            if space <= 0:
                self.flush()
                continue
            n = min(space, b - start)
            assert self.rows_buf is not None and self.pred_buf is not None
            self.rows_buf[self.buf_fill : self.buf_fill + n].copy_(rows_cpu[start : start + n])
            non_blocking = bool(self.pred_buf_is_pinned) and preds.device.type != "cpu"
            self.pred_buf[self.buf_fill : self.buf_fill + n].copy_(
                preds[start : start + n], non_blocking=non_blocking
            )
            if non_blocking:
                self.buf_needs_wait_evt = True
            self.buf_fill += n
            start += n
            if self.buf_fill >= int(self.target_rows):
                self.flush()


compile_distributed_safe()
