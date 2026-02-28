# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import json
import datetime
import gc
import glob
import importlib
import inspect
import itertools
import logging
import math
import os
import platform
import re
import socket
import sys
import threading
import tempfile
import time
import warnings
from collections.abc import Mapping, MutableMapping
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Optional, TypeAlias

import torch
import torch.distributed
import torch.nn as nn
from tensordict import TensorDictBase
from torch.distributed.checkpoint import FileSystemReader, load
from torch.distributed.elastic.control_plane import worker_main
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

from ..core.concurrency import (
    Mutex,
    TensorPagePool,
    TensorSpooler,
    move_staged_pair_to_device as _move_staged_pair_to_device,
    new_affinity,
    pool_tensor as _pool_tensor,
    stream_tensor as _stream_tensor,
)
from ..core.config import RuntimeConfig, coerce_model_config
from ..core.datatypes import (
    env_bool,
    env_first,
    env_first_int,
    env_float,
    env_int,
    env_str,
    read_json,
)
from ..core.policies import DistributedPolicy, ModelPolicy, PrecisionPolicy
from ..core.precision import (
    StatelessAutocast,
    cast_batchnorm_buffers_dtype as _cast_batchnorm_buffers_dtype,
    cast_float_dtype as _cast_float_dtype,
    preload_layers as _preload_layers,
    unify_model_dtype as _unify_model_dtype,
    validate_model_dtype_unity as _validate_model_dtype_unity,
)
from ..core.system import (
    CPU,
    Memory,
    Monitor,
    accelerator_max_allocated_memory,
    accelerator_stream,
    accelerator_type,
    allocated_accelerator_memory,
    available_device_memory,
    empty_device_cache,
    flush_accelerator_memory_stats,
    get_accelerator_index,
    get_device,
    get_num_accelerators,
    init_python_path,
    is_accelerator_available,
    is_accelerator_timer_supported,
    is_cuda_bf16_supported,
    is_oom_error,
    is_pin_supported,
    new_accelerator_event,
    posix_time,
    set_accelerator_index,
    set_float32_precision,
    sync_accelerator,
)
from ..core.tensor import (
    compute_batch_bytes_per_sample as _compute_batch_bytes_per_sample,
    enable_meta_monitor as _enable_meta_monitor,
    is_meta_or_fake_tensor,
    to_device_recursive as _to_device_recursive,
    to_torch_tensor,
    touch_tensors as _touch_tensors,
    validate_no_fake_dtensor as _validate_no_fake_dtensor,
    validate_no_meta_tensors as _validate_no_meta_tensors,
)
from ..data import collate
from ..data.collate import Unsharder, warmup_scaler_stats as _warmup_scaler_stats
from ..data.pipeline import Dataset, get_batch_length as _get_batch_length
from ..nn.graph import (
    canonicalize_compile_mode,
    compile_distributed_safe,
    compile_safe,
    cudagraph_mark_step_begin,
    cudagraph_mark_step_end,
    from_checkpoint,
    inference_mode,
    to_checkpoint,
    to_submodule,
)
from ..nn.layers import Recorder, resize_scaler_buffer
from ..nn.profiler import (
    FlopCounter,
    get_torch_profiler as _get_torch_profiler,
    log_profiler_summary as _get_profiler_summary,
)
from ..nn.wrappers import Model, update_delta_gate_auto_k as _set_gate_factor
from .autobatch import (
    clear_oom_retries as _clear_oom_retries,
    get_sampler_scaler as _get_sampler_scaler,
    log_scale_rate_throttled as _is_scale_rate_logged,
    probe_per_sample_mem_bytes as _get_sample_size,
    recover_oom as _recover_oom,
)
from .distributed import (
    Checkpointer,
    ProcessBroker,
    broadcast_scalar,
    distributed_all_reduce_grads,
    distributed_all_reduce_sum as _reduce_sum,
    distributed_barrier,
    distributed_sync,
    get_distributed_mesh,
    get_group_world_size as _get_world_size,
    get_accel_group,
    get_world_size,
    is_distributed,
    joining,
    no_sync,
    resolve_process_group as _validate_distributed_group,
    to_hsdp_module,
)
from .io import _filtered_warnings, _torch_load_checkpoint
from .losses import (
    CRPSLoss,
    DataFidelityLoss,
    LinearCombinationLoss,
    LossWeightController,
    StandardNormalLoss,
    StudentsTLoss,
    TiledLoss,
)
from .optimizers import (
    AdamW,
    ExponentialMovingAverage,
    StochasticWeightAverage,
    init_optimizer_state as _init_optimizer,
)

try:
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy
except Exception:
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy
    except Exception:
        MixedPrecisionPolicy = None

try:
    from tensordict.nn import CudaGraphModule as TD_CudaGraphModule
except Exception:
    TD_CudaGraphModule = None

try:
    from tensordict.nn.functional_modules import _exclude_td_from_pytree
    _exclude_td_from_pytree().set()
except Exception:
    pass

os.environ.setdefault("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS", "1")

_COMPILE_SAFE_DONE = False
_COMPILE_SAFE_LOCK = Mutex()
_LOGGER = logging.getLogger(__name__)
_float8_log = ProcessBroker.rank0_logger(_LOGGER)
_EXPORT_RETURN_LOCK = threading.Lock()
JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = (
    JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
)
MB_DIV = 1024.0 * 1024.0
PathLike: TypeAlias = str | os.PathLike[str] | Path
ReturnSink: TypeAlias = MutableMapping[str, object]
TorchDeviceLike: TypeAlias = torch.device | str | int


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key, None)
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off", ""):
        return False
    return bool(default)


def _sync_torchinductor_cache_globals(cache_dir: str) -> None:
    with contextlib.suppress(Exception):
        import torch._inductor.config as _icfg

        if hasattr(_icfg, "global_cache_dir"):
            _icfg.global_cache_dir = cache_dir
        if hasattr(_icfg, "cache_dir"):
            _icfg.cache_dir = cache_dir
    with contextlib.suppress(Exception):
        import torch._inductor.codecache as _cc

        if hasattr(_cc, "_cache_dir"):
            _cc._cache_dir = cache_dir
        for _name in ("global_cache_dir", "cache_dir", "inductor_cache_dir"):
            if hasattr(_cc, _name):
                v = getattr(_cc, _name)
                if isinstance(v, str):
                    setattr(_cc, _name, cache_dir)
        if callable(getattr(_cc, "cache_dir", None)):
            _cc.cache_dir()


def _normalize_model_averaging(
    x: object, *, default: str = "auto"
) -> str | None:
    if default is None:
        raise ValueError("default model_averaging must not be None")
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("auto", "ema", "swa"):
            return s
        if s in ("none", "null", "off", "false", "0", ""):
            return None
    raise ValueError(
        f"Invalid model_averaging={x!r}. Expected None|'auto'|'ema'|'swa'."
    )


def _has_bn_modules(model: Any) -> bool:
    try:
        import torch
        from torch import nn

        if not isinstance(model, nn.Module):
            return False
        bn_types = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            getattr(nn, "SyncBatchNorm", nn.BatchNorm1d),
        )
        for m in model.modules():
            if isinstance(m, bn_types):
                return True
    except Exception:
        return False
    return False


def _resolve_model_averaging(
    model: Any, requested: object
) -> tuple[str | None, bool]:
    req = _normalize_model_averaging(requested, default="auto")
    has_bn = _has_bn_modules(model)
    if req is None:
        return None, has_bn
    if req == "auto":
        return ("ema" if has_bn else "swa"), has_bn
    return req, has_bn


def _atomic_torch_save(obj: object, path: str) -> None:
    import torch

    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=d
    )
    os.close(fd)
    try:
        try:
            torch.save(obj, tmp, _use_new_zipfile_serialization=False)
        except Exception:
            torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _mark_ephemeral_ckpt_dir(d: str) -> None:
    try:
        os.makedirs(d, exist_ok=True)
        with open(
            os.path.join(d, ".enn_ephemeral_ckpt"), "w", encoding="utf-8"
        ) as f:
            f.write("1\n")
    except Exception:
        pass


def _force_final_avg_update(
    ema_helper: object | None,
    swa_helper: object | None,
    model: Any,
    optimizer: object | None,
) -> None:
    try:
        if ema_helper is not None and hasattr(ema_helper, "update"):
            prev = getattr(ema_helper, "update_every", None)
            try:
                if prev is not None:
                    ema_helper.update_every = 1
                ema_helper.update(model, optimizer=optimizer)
            finally:
                if prev is not None:
                    ema_helper.update_every = prev
    except Exception:
        pass
    try:
        if swa_helper is not None and hasattr(swa_helper, "update"):
            prev = getattr(swa_helper, "_update_every", None)
            try:
                if prev is not None:
                    swa_helper._update_every = 1
                swa_helper.update(model)
            finally:
                if prev is not None:
                    swa_helper._update_every = prev
    except Exception:
        pass


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
        return start_factor + (1.0 - start_factor) * (
            step / max(1, warmup_steps)
        )
    t = step - warmup_steps
    frac_min = emin / base if base > 0.0 else 0.0
    return frac_min + (1.0 - frac_min) * 0.5 * (
        1.0 + math.cos(math.pi * t / max(1, main_steps))
    )


def _export_return_model_pt(
    model: Any,
    ckpt_dir: str,
    *,
    ema_helper: object | None = None,
    swa_helper: object | None = None,
    model_averaging: str | None = "auto",
) -> None:
    import torch

    out_dir_env = os.environ.get("ENN_RETURN_DIR", None)
    out_dir = (out_dir_env or "").strip() or str(ckpt_dir or "").strip()
    if not out_dir_env:
        try:
            from .distributed import _is_tmpfs_path, _pick_disk_cache_base

            if out_dir and _is_tmpfs_path(out_dir):
                out_dir = ""
            if not out_dir:
                base = _pick_disk_cache_base()
                if base and (not _is_tmpfs_path(base)):
                    out_dir = os.path.join(str(base), "enn_return")
                    _mark_ephemeral_ckpt_dir(out_dir)
        except Exception:
            pass
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)

    model_averaging = _normalize_model_averaging(model_averaging, default="auto")
    if model_averaging == "auto":
        env_ma = os.environ.get("ENN_MODEL_AVERAGING", None)
        if env_ma is not None:
            with contextlib.suppress(Exception):
                model_averaging = _normalize_model_averaging(env_ma, default="auto")
        if model_averaging == "auto":
            model_averaging = "ema" if _has_bn_modules(model) else "swa"

    try:
        p0 = next((p for p in model.parameters() if torch.is_tensor(p)), None)
        if torch.is_tensor(p0) and p0.device.type == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass

    avg_params = None
    if model_averaging == "swa" and swa_helper is not None:
        shadow = getattr(swa_helper, "shadow", None)
        n_avg = int(getattr(swa_helper, "n_averaged", 0) or 0)
        if n_avg > 0 and isinstance(shadow, dict) and shadow:
            avg_params = shadow
    if model_averaging == "ema" and avg_params is None and ema_helper is not None:
        shadow = getattr(ema_helper, "shadow", None)
        if isinstance(shadow, dict) and shadow:
            avg_params = shadow

    def _canon_key(k: str) -> str:
        kk = str(k)
        kk = kk.replace("._enn_inner._orig_mod", "")
        kk = kk.replace("._enn_inner", "")
        if kk.startswith("processor."):
            kk = "fuser." + kk[len("processor.") :]
        if kk.startswith("controller."):
            kk = "temporal_token_collector." + kk[len("controller.") :]
        if kk.startswith("fuser.perceiver._orig_mod."):
            kk = "fuser.perceiver." + kk[len("fuser.perceiver._orig_mod.") :]
        return kk

    with _EXPORT_RETURN_LOCK:
        eager_ctx = getattr(model, "eager_for_export", None)
        cm = eager_ctx() if callable(eager_ctx) else contextlib.nullcontext()
        with cm:
            with torch.no_grad():
                sd = dict(model.state_dict())

        if isinstance(avg_params, dict) and avg_params:
            for name, v in avg_params.items():
                if not isinstance(name, str) or not torch.is_tensor(v):
                    continue
                cur = sd.get(name, None)
                if not torch.is_tensor(cur):
                    continue
                try:
                    vv = v.detach()
                    if tuple(vv.shape) != tuple(cur.shape):
                        continue
                    if vv.dtype != cur.dtype:
                        vv = vv.to(dtype=cur.dtype)
                    sd[name] = vv
                except Exception:
                    continue

        sd_out: dict[str, object] = {}
        collisions = 0
        for k, v in sd.items():
            if not isinstance(k, str):
                k = str(k)
            kk = _canon_key(k)
            if torch.is_tensor(v):
                t = v.detach()
                if getattr(t, "is_meta", False) or t.device.type == "meta":
                    continue
                if t.device.type != "cpu":
                    t = t.to("cpu")
                if kk in sd_out and torch.is_tensor(sd_out.get(kk)):
                    try:
                        cur = sd_out[kk]
                        if torch.is_tensor(cur) and tuple(cur.shape) != tuple(t.shape):
                            collisions += 1
                            continue
                    except Exception:
                        collisions += 1
                        continue
                sd_out[kk] = t
            else:
                sd_out[kk] = v

        with contextlib.suppress(Exception):
            from .distributed import _coerce_dcp_keys

            _coerce_dcp_keys(sd_out)

        if collisions > 0:
            with contextlib.suppress(Exception):
                _LOGGER.warning(
                    "[export_return_model_pt] key canonicalization collisions=%d (kept first matching tensor)",
                    int(collisions),
                )

        out_path = os.path.join(out_dir, "model.pt")
        _atomic_torch_save(sd_out, out_path)

        sd.clear()
        sd_out.clear()
        del sd
        del sd_out
        gc.collect()


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
    x_tok: TensorPagePool.Token | None = None
    y_tok: TensorPagePool.Token | None = None
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
            pair = Monitor.get_thread_events(device, slot="h2d")
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
    calibration_loader: object | None = None,
    total_epochs: int,
    scheduler_step_per_batch: bool = True,
    swa_helper: object | None = None,
    swa_start_epoch: int = 0,
    ema_helper: object | None = None,
    checkpointer: Checkpointer | None = None,
    buffers_dtype: torch.dtype | None = None,
    dataset: object | None = None,
    **kwargs: Any,
) -> object:
    from ..data.nodes import Sampler

    ddp_fallback: bool = bool(kwargs.pop("ddp_fallback", False))
    if train_loader is None:
        raise RuntimeError("epochs requires a training dataloader")
    model_for_grads = model.module if hasattr(model, "module") else model
    world_sz = int(get_world_size(device)) if is_distributed() else 1
    meta = (
        dataset if isinstance(dataset, Dataset) else Dataset.for_device(device)
    )

    optim_diag = env_bool("ENN_OPTIMIZER_DIAG_NONFINITE", default=False)
    optim_diag_every = max(1, int(env_int("ENN_OPTIMIZER_DIAG_EVERY", 1) or 1))
    optim_diag_scope = str(env_str("ENN_OPTIMIZER_DIAG_SCOPE") or "").strip()
    nonfinite_fail_fast = env_bool("ENN_FAIL_FAST_NONFINITE", default=False)
    nonfinite_skip_step = env_bool("ENN_SKIP_STEP_ON_NONFINITE", default=False)
    nonfinite_dump_dir = str(env_str("ENN_NONFINITE_DUMP_DIR") or "").strip()
    nonfinite_dump_limit = max(
        0, int(env_int("ENN_NONFINITE_DUMP_LIMIT", 4) or 4)
    )
    nonfinite_dump_all_ranks = env_bool(
        "ENN_NONFINITE_DUMP_ALL_RANKS", default=False
    )
    nonfinite_dumped = 0
    strict_sanitize = env_bool("ENN_SANITIZE_NAN_STRICT", default=False)

    def _opt_scope_ok(name: str) -> bool:
        return (not optim_diag_scope) or (optim_diag_scope in name)

    def _first_nonfinite_grad(mod: nn.Module) -> str | None:
        for n, p in mod.named_parameters(recurse=True):
            if not _opt_scope_ok(n):
                continue
            g = getattr(p, "grad", None)
            if torch.is_tensor(g) and g.numel() > 0:
                with contextlib.suppress(Exception):
                    if not bool(torch.isfinite(g).all().item()):
                        return n
        return None

    def _first_nonfinite_param(mod: nn.Module) -> str | None:
        for n, p in mod.named_parameters(recurse=True):
            if not _opt_scope_ok(n):
                continue
            if not (torch.is_tensor(p) and p.is_floating_point() and p.numel() > 0):
                continue
            with contextlib.suppress(Exception):
                if not bool(torch.isfinite(p).all().item()):
                    return n
        return None

    def _first_nonfinite_optim_state(opt: object) -> str | None:
        st = getattr(opt, "state", None)
        if not isinstance(st, dict):
            return None
        for _, v in st.items():
            if not isinstance(v, dict):
                continue
            for k2, t in v.items():
                if torch.is_tensor(t) and t.numel() > 0:
                    with contextlib.suppress(Exception):
                        if not bool(torch.isfinite(t).all().item()):
                            return str(k2)
        return None

    def _maybe_dump_nonfinite(
        *,
        epoch: int,
        step_idx: int,
        step_total: int,
        bad: str,
        X: torch.Tensor | None = None,
        Y: torch.Tensor | None = None,
        Y_flat: torch.Tensor | None = None,
        loss: torch.Tensor | None = None,
        extra: dict[str, object] | None = None,
    ) -> str | None:
        nonlocal nonfinite_dumped
        if not nonfinite_dump_dir:
            return None
        if nonfinite_dump_limit > 0 and int(nonfinite_dumped) >= int(
            nonfinite_dump_limit
        ):
            return None
        if is_distributed() and (not nonfinite_dump_all_ranks) and local_rank != 0:
            return None
        try:
            os.makedirs(nonfinite_dump_dir, exist_ok=True)
        except Exception:
            return None

        payload: dict[str, object] = {
            "epoch": int(epoch),
            "step_idx": int(step_idx),
            "step_total": int(step_total),
            "bad": str(bad),
        }
        if isinstance(extra, dict) and extra:
            payload["extra"] = extra
        try:
            if torch.is_tensor(loss):
                with contextlib.suppress(Exception):
                    payload["loss"] = loss.detach().float().cpu()
        except Exception:
            pass
        for k, t in (
            ("X", X),
            ("Y", Y),
            ("Y_flat", Y_flat),
        ):
            if not torch.is_tensor(t):
                continue
            try:
                tt = t.detach()
                if tt.ndim >= 1:
                    tt = tt[: min(16, int(tt.shape[0]))]
                payload[k] = tt.to("cpu")
                with contextlib.suppress(Exception):
                    payload[f"{k}_shape"] = tuple(t.shape)
                    payload[f"{k}_dtype"] = str(t.dtype)
                    payload[f"{k}_device"] = str(t.device)
            except Exception:
                pass

        path = os.path.join(
            nonfinite_dump_dir,
            f"nonfinite_e{int(epoch)}_s{int(step_idx)}_t{int(step_total)}.pt",
        )
        try:
            torch.save(payload, path)
            nonfinite_dumped += 1
            return str(path)
        except Exception:
            return None
    autocast_dtype = None
    with contextlib.suppress(Exception):
        import inspect

        f = getattr(StatelessAutocast, "resolve_float_dtype", None)
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
        set_float32_precision(
            device, dtype=param_dtype, autocast_dtype=autocast_dtype
        )
    cpu_pool = None
    pool_capacity = 0
    if is_pin_supported(str(getattr(device, "type", "cpu"))):
        with contextlib.suppress(Exception):
            Memory.prefer_local_numa()
        try:
            cpu_pool_cap = max(
                2, int(env_int("ENN_RUNTIME_PIN_POOL_CAPACITY", 8))
            )
            cpu_pool = TensorPagePool(capacity=cpu_pool_cap)
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
    if (
        per_batch is None
        or int(per_batch) <= 0
        or est_bytes_per_sample is None
    ):
        try:
            it = iter(train_loader)
            sample = next(it)
            bs, bytes_ps = _compute_batch_bytes_per_sample(sample)
            if (
                (per_batch is None or int(per_batch) <= 0)
                and bs is not None
                and (bs > 0)
            ):
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
    from ..core.policies import BatchPolicy

    fixed_accum = 2 if getattr(device, "type", "cpu") == "cpu" else 4
    min_grad_accum = fixed_accum
    max_grad_accum = fixed_accum
    dev_margin = env_float("ENN_DEVICE_MARGIN", 0.8)
    host_margin = env_float("ENN_HOST_MARGIN", 0.8)
    budget_slack = env_float("ENN_BUDGET_SLACK", 1.25)
    budget_slack = max(1.0, min(4.0, float(budget_slack)))
    dev_budget_ratio = env_float("ENN_DEVICE_BUDGET_RATIO", 1.0)
    dev_budget_min_bytes = env_int("ENN_DEVICE_BUDGET_MIN_BYTES", 0)
    _dev_budget_max_raw = env_int("ENN_DEVICE_BUDGET_MAX_BYTES", 0)
    dev_budget_max_bytes = (
        None if int(_dev_budget_max_raw) <= 0 else int(_dev_budget_max_raw)
    )
    host_budget_ratio = env_float("ENN_HOST_BUDGET_RATIO", 1.0)
    host_budget_min_bytes = env_int("ENN_HOST_BUDGET_MIN_BYTES", 0)
    _host_budget_max_raw = env_int("ENN_HOST_BUDGET_MAX_BYTES", 0)
    host_budget_max_bytes = (
        None if int(_host_budget_max_raw) <= 0 else int(_host_budget_max_raw)
    )
    dev_margin = max(0.0, min(1.0, float(dev_margin)))
    host_margin = max(0.0, min(1.0, float(host_margin)))
    dev_budget_ratio = max(0.0, min(1.0, float(dev_budget_ratio)))
    host_budget_ratio = max(0.0, min(1.0, float(host_budget_ratio)))
    dev_budget_min_bytes = max(0, int(dev_budget_min_bytes))
    host_budget_min_bytes = max(0, int(host_budget_min_bytes))
    dev_budget_max_bytes = (
        None
        if dev_budget_max_bytes is None
        else max(0, int(dev_budget_max_bytes))
    )
    host_budget_max_bytes = (
        None
        if host_budget_max_bytes is None
        else max(0, int(host_budget_max_bytes))
    )
    if dev_budget_max_bytes is not None and int(dev_budget_max_bytes) <= 0:
        dev_budget_max_bytes = None
    if host_budget_max_bytes is not None and int(host_budget_max_bytes) <= 0:
        host_budget_max_bytes = None
    tpl = None
    if (
        est_bytes_per_sample is not None
        and est_bytes_per_sample > 0
        and (max_grad_accum > 0)
    ):
        try:
            effective_streams = 1 + max(0, pool_capacity)
            tpl = BatchPolicy(
                sample_bytes=int(est_bytes_per_sample),
                host_sample_bytes=int(est_bytes_per_sample),
                prebatch=1,
                prefetch_factor=int(env_int("ENN_HOST_PREFETCH_FACTOR", 4)),
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
                    None
                    if host_budget_max_bytes is None
                    else int(host_budget_max_bytes)
                ),
                device_budget_ratio=float(dev_budget_ratio),
                device_budget_min_bytes=int(dev_budget_min_bytes),
                device_budget_max_bytes=(
                    None
                    if dev_budget_max_bytes is None
                    else int(dev_budget_max_bytes)
                ),
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
            if (
                tpl.device_budget_max_bytes is None
                or tpl.host_budget_max_bytes is None
            ):
                try:
                    target_total_samples = max(1, int(per_batch or 1)) * max(
                        1, int(min_grad_accum)
                    )
                    new_dev_cap = tpl.device_budget_max_bytes
                    new_host_cap = tpl.host_budget_max_bytes
                    if new_dev_cap is None and int(tpl.sample_bytes or 0) > 0:
                        base_dev = int(tpl.sample_bytes) * int(
                            target_total_samples
                        )
                        cap_dev = int(float(base_dev) * float(budget_slack))
                        if (
                            safe_dev_total is not None
                            and int(safe_dev_total) > 0
                        ):
                            cap_dev = min(int(cap_dev), int(safe_dev_total))
                        cap_dev = max(0, int(cap_dev))
                        new_dev_cap = None if cap_dev <= 0 else cap_dev
                    if (
                        new_host_cap is None
                        and int(tpl.host_sample_bytes or 0) > 0
                    ):
                        inflight = int(tpl.host_inflight_batches_per_proc())
                        lw = max(
                            1, int(getattr(tpl, "local_world_size", 1) or 1)
                        )
                        base_host = (
                            int(tpl.host_sample_bytes)
                            * max(1, inflight)
                            * max(1, lw)
                            * int(target_total_samples)
                        )
                        cap_host = int(float(base_host) * float(budget_slack))
                        if (
                            safe_host_total is not None
                            and int(safe_host_total) > 0
                        ):
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
            int(min_grad_accum), min(int(max_grad_accum), int(max_from_mem))
        )
    grad_accum_steps = int(min_grad_accum)
    grad_accum_steps = broadcast_scalar(grad_accum_steps, device=device, src=0)
    gpu_util_ema = None
    mem_util_ema = None
    util_alpha = 0.2
    global_step = 0
    delta_gate_auto_step_total = 0
    with contextlib.suppress(Exception):
        target_for_autok = model.module if hasattr(model, "module") else model
        step_buf = getattr(
            target_for_autok, "delta_gate_auto_k_step_buf", None
        )
        if isinstance(step_buf, torch.Tensor):
            delta_gate_auto_step_total = int(step_buf.item())
    util_adjust_interval = 0
    util_warmup_steps = 0
    if buffers_dtype is not None:
        target_for_buffers = (
            model.module if hasattr(model, "module") else model
        )
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
                        with open(
                            "/etc/os-release", "r", encoding="utf-8"
                        ) as f:
                            for line in f:
                                if line.startswith("PRETTY_NAME="):
                                    pretty = (
                                        line.strip()
                                        .split("=", 1)[1]
                                        .strip()
                                        .strip('"')
                                    )
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
    with torch.no_grad():
        _warmup_scaler_stats(model, train_loader, ops)
    in_dim = int(ops.in_dim)
    dev_type = str(getattr(device, "type", "cpu"))
    non_blocking_ok = bool(dev_type in ("cuda", "xpu"))
    pinned_ok = bool(is_pin_supported(dev_type))
    backend = accelerator_type(dev_type)
    stream_fn = (
        getattr(backend, "current_stream", None)
        if backend is not None
        else None
    )
    Event = getattr(backend, "Event", None) if backend is not None else None
    can_stream_release = bool(
        pinned_ok
        and non_blocking_ok
        and callable(stream_fn)
        and (Event is not None)
    )
    make_fence_event = partial(
        new_accelerator_event, device, enable_timing=False
    )
    use_timer = Monitor.is_event_timer_available(device)
    timer_sync = (not use_timer) and bool(
        Monitor.is_clock_synchronized(str(dev_type or "cpu"))
    )
    train_steps = _get_batch_length(train_loader)
    val_steps = _get_batch_length(val_loader)
    total_updates = int(total_epochs) * (int(train_steps) + int(val_steps))
    if train_steps > 0:
        util_adjust_interval = max(10, int(train_steps * 0.05))
        util_warmup_steps = max(
            util_adjust_interval,
            min(int(train_steps), max(50, int(train_steps * 0.1))),
        )
    status_bar = (
        ProcessBroker.get_progress_bar(
            title="Training",
            total=total_updates,
            device=device,
        )
        if local_rank == 0
        else None
    )
    scheduler_step_per_batch = bool(scheduler_step_per_batch)
    swa_start_epoch = max(0, int(swa_start_epoch))
    prev_io_time = 0.0
    prev_comp_time = 0.0
    prev_kern_time = 0.0
    prev_io_bytes = 0.0
    prev_flops = 0.0
    prev_samples = 0.0
    tflops_warmup = int(env_int("ENN_TFLOPS_WARMUP_ITERS", 0) or 0)
    if tflops_warmup <= 0 and getattr(device, "type", "cpu") == "cuda":
        tflops_warmup = 5
    tflops_seen = 0
    prev_flops_tflops = 0.0
    prev_kern_tflops = 0.0
    join_context = joining(model=model, optimizer=optimizer)
    with join_context:
        with contextlib.suppress(Exception):
            new_affinity().pin_thread()
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
        kern_ev_s = None
        kern_ev_e = None
        if use_timer:
            pair = Monitor.get_thread_events(device, slot="comp")
            if pair is not None:
                comp_ev_s, comp_ev_e = pair
        if getattr(device, "type", "cpu") == "cuda":
            with contextlib.suppress(Exception):
                kern_ev_s = torch.cuda.Event(enable_timing=True)
                kern_ev_e = torch.cuda.Event(enable_timing=True)
        torch_prof = None
        prof_enabled = env_bool(
            "ENN_TORCH_PROFILE_TRAIN",
            env_bool("ENN_TORCH_PROFILE", False),
        )
        prof_all_ranks = env_bool("ENN_TORCH_PROFILE_ALL_RANKS", False)
        prof_rank = (
            int(torch.distributed.get_rank()) if is_distributed() else 0
        )
        if prof_enabled and (prof_all_ranks or prof_rank == 0):
            prof_dir = env_str("ENN_TORCH_PROFILE_DIR")
            if not prof_dir:
                prof_dir = os.path.join(
                    str(ops.ckpt_dir or "."), "torch_profiler"
                )
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
        flop_counter_val = (
            FlopCounter(model, mode="eval", device=device)
            if val_loader is not None
            else None
        )
        for epoch_idx in range(int(total_epochs)):
            with contextlib.suppress(Exception):
                epochables = getattr(train_loader, "_enn_epochables", None)
                if epochables is not None:
                    for obj in epochables:
                        fn = getattr(obj, "set_epoch", None)
                        if callable(fn):
                            fn(int(epoch_idx))
                else:
                    fn = getattr(
                        getattr(train_loader, "sampler", None),
                        "set_epoch",
                        None,
                    )
                    if callable(fn):
                        fn(int(epoch_idx))
                    fn = getattr(train_loader, "set_epoch", None)
                    if callable(fn):
                        fn(int(epoch_idx))
            if is_distributed() and env_bool("ENN_DIST_SYNC_EPOCH", False):
                target_module = (
                    model.module if hasattr(model, "module") else model
                )
                distributed_sync(target_module, device=device)
            flop_breakdown_epoch = {}
            io_time = 0.0
            comp_time = 0.0
            kern_time = 0.0
            io_bytes = 0.0
            flops = 0.0
            train_samples_epoch = 0.0
            flops_tflops = 0.0
            kern_tflops = 0.0
            with flop_counter_train:
                model.train()
                train_pg = (
                    _validate_distributed_group(meta, model)
                    if is_distributed()
                    else None
                )
                global_step = 0
                optimizer.zero_grad(set_to_none=True)
                t_fetch_start = time.perf_counter_ns()
                total_batches = 0
                with contextlib.suppress(Exception):
                    if int(train_steps) > 0:
                        total_batches = int(train_steps)
                if int(total_batches) <= 0:
                    with contextlib.suppress(Exception):
                        total_batches = int(len(train_loader))
                if int(total_batches) < 0:
                    total_batches = 0
                train_accum_since_last = 0
                lw_top_sum = None
                lw_bottom_sum = None
                lw_count = 0
                train_iter = iter(train_loader)
                try:
                    _first_raw = next(train_iter)
                except StopIteration:
                    dl_type = type(train_loader).__name__
                    dl_len = None
                    with contextlib.suppress(Exception):
                        dl_len = len(train_loader)
                    r = (
                        int(torch.distributed.get_rank())
                        if is_distributed()
                        else 0
                    )
                    w = int(get_world_size(device)) if is_distributed() else 1
                    with contextlib.suppress(Exception):
                        chain = []
                        cur = train_loader
                        for _ in range(8):
                            if cur is None:
                                break
                            try:
                                clen = len(cur)
                            except Exception:
                                clen = None
                            chain.append(
                                f"{type(cur).__module__}.{type(cur).__name__}(len={clen})"
                            )
                            nxt = getattr(cur, "_src", None)
                            if nxt is None and hasattr(cur, "_base_iterable"):
                                nxt = getattr(cur, "_base_iterable", None)
                            if nxt is None or nxt is cur:
                                break
                            cur = nxt
                        _LOGGER.error(
                            "[DIAG] train: loader chain: %s",
                            " -> ".join(chain),
                        )
                        base = getattr(train_loader, "_base_iterable", None)
                        if base is not None:
                            try:
                                next(iter(base))
                                _LOGGER.error(
                                    "[DIAG] train: _base_iterable yields batches; Stream layer likely blocking/dropping."
                                )
                            except StopIteration:
                                _LOGGER.error(
                                    "[DIAG] train: _base_iterable also produced 0 batches (source pipeline empty)."
                                )
                            except Exception as e:
                                _LOGGER.exception(
                                    "[DIAG] train: probing _base_iterable failed: %s",
                                    e,
                                )
                        src = getattr(train_loader, "_src", None)
                        if src is not None:
                            try:
                                next(iter(src))
                                _LOGGER.error(
                                    "[DIAG] train: underlying _src yields batches; wrapper/preload layer likely blocking."
                                )
                            except StopIteration:
                                _LOGGER.error(
                                    "[DIAG] train: underlying _src also produced 0 batches."
                                )
                            except Exception as e:
                                _LOGGER.exception(
                                    "[DIAG] train: probing underlying _src failed: %s",
                                    e,
                                )
                    raise RuntimeError(
                        "train: data_loader produced 0 batches. "
                        f"(rank={r}/{w}, device={device}, loader={dl_type}, len={dl_len}) "
                        "If you cancelled a prior run or reused the same iterator, recreate the Loader/Dataset. "
                        "On accelerators, also check Stream/prefetch backpressure and dataset filtering."
                    )
                for step_idx, _raw in enumerate(
                    itertools.chain([_first_raw], train_iter)
                ):
                    train_accum_since_last += 1
                    while True:
                        mark_cudagraph = False
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
                            should_sync = ((step_idx + 1) % max(
                                1, grad_accum_steps
                            ) == 0) or (
                                (int(total_batches) > 0)
                                and ((step_idx + 1) == int(total_batches))
                            )
                            if (
                                use_timer
                                and comp_ev_s is not None
                                and comp_ev_e is not None
                            ):
                                comp_ev_s.record()
                            else:
                                t_comp_s = time.perf_counter_ns()
                            with no_sync(
                                model,
                                enable=grad_accum_steps > 1
                                and (not should_sync),
                            ):
                                with flop_counter_train.step(
                                    display=False
                                ) as train_counter:
                                    if getattr(device, "type", None) == "cuda":
                                        cudagraph_mark_step_begin()
                                        mark_cudagraph = True
                                    with StatelessAutocast.float(device):
                                        Y_flat = Y.reshape(Y.shape[0], -1)
                                        if (
                                            Y_flat.device != device
                                            or Y_flat.dtype != param_dtype
                                        ):
                                            Y_flat = Y_flat.to(
                                                device,
                                                dtype=param_dtype,
                                                non_blocking=non_blocking_ok,
                                            )

                                        t_kern_s = 0
                                        if kern_ev_s is not None:
                                            with contextlib.suppress(Exception):
                                                kern_ev_s.record()
                                        else:
                                            t_kern_s = time.perf_counter_ns()

                                        (
                                            y_hat,
                                            loss_val,
                                            loss_top_val,
                                            loss_bottom_val,
                                        ) = model(
                                            X,
                                            labels_flat=Y_flat,
                                            global_loss=top_loss,
                                            local_loss=bottom_loss,
                                            loss_weights=loss_controller.weights(),
                                            calibrate_output=False,
                                            return_loss_components=True,
                                        )
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
                                        isinstance(
                                            loss_bottom_val, torch.Tensor
                                        )
                                        and loss_bottom_val.ndim > 0
                                    ):
                                        loss_bottom_val = (
                                            loss_bottom_val.mean()
                                        )

                                    if loss_val is None:
                                        raise RuntimeError(
                                            "Model returned no loss value during training. Ensure loss functions are provided and returning valid outputs."
                                        )
                                    if not isinstance(loss_val, torch.Tensor):
                                        loss_val = torch.as_tensor(
                                            loss_val,
                                            device=device,
                                            dtype=param_dtype,
                                        )
                                    else:
                                        loss_val = loss_val.to(
                                            device=device, dtype=param_dtype
                                        )
                                    accum_scale = max(1, grad_accum_steps)
                                    loss_for_backprop = loss_val / float(
                                        accum_scale
                                    )
                                    scaler.scale(loss_for_backprop).backward()

                                    t_kern_step = 0.0
                                    if kern_ev_e is not None:
                                        with contextlib.suppress(Exception):
                                            kern_ev_e.record()
                                    else:
                                        t_kern_e = time.perf_counter_ns()
                                        if timer_sync:
                                            sync_accelerator(device)
                                        t_kern_step = (
                                            float(t_kern_e - t_kern_s)
                                            / 1000000000.0
                                        )
                                    if (
                                        loss_top_val is not None
                                        or loss_bottom_val is not None
                                    ):
                                        lw_count += 1
                                        if isinstance(
                                            loss_top_val, torch.Tensor
                                        ):
                                            v = loss_top_val.detach()
                                            lw_top_sum = (
                                                v
                                                if lw_top_sum is None
                                                else lw_top_sum + v
                                            )
                                        if isinstance(
                                            loss_bottom_val, torch.Tensor
                                        ):
                                            v = loss_bottom_val.detach()
                                            lw_bottom_sum = (
                                                v
                                                if lw_bottom_sum is None
                                                else lw_bottom_sum + v
                                            )
                                    if should_sync:
                                        if ddp_fallback and world_sz > 1:
                                            distributed_all_reduce_grads(
                                                model_for_grads,
                                                average=True,
                                                policy=dist_policy.collective,
                                            )
                                        scaler.unscale_(optimizer)
                                        diag_now = (
                                            int(delta_gate_auto_step_total)
                                            % int(optim_diag_every)
                                        ) == 0
                                        if (
                                            optim_diag
                                            or nonfinite_fail_fast
                                            or nonfinite_skip_step
                                            or bool(nonfinite_dump_dir)
                                        ) and (
                                            diag_now
                                            or nonfinite_fail_fast
                                            or nonfinite_skip_step
                                            or bool(nonfinite_dump_dir)
                                        ):
                                            bad_p0 = _first_nonfinite_param(model_for_grads)
                                            if bad_p0 is not None:
                                                dump_path = _maybe_dump_nonfinite(
                                                    epoch=int(epoch_idx),
                                                    step_idx=int(step_idx),
                                                    step_total=int(delta_gate_auto_step_total),
                                                    bad=str("param:" + str(bad_p0)),
                                                    X=X,
                                                    Y=Y,
                                                    Y_flat=Y_flat,
                                                    loss=loss_val,
                                                )
                                                _LOGGER.error(
                                                    "[OPTIM][nonfinite] pre-step param non-finite: %s (epoch=%d, step_idx=%d, step=%d, opt=%s)",
                                                    bad_p0,
                                                    int(epoch_idx),
                                                    int(step_idx),
                                                    int(delta_gate_auto_step_total),
                                                    type(optimizer).__name__,
                                                )
                                                if dump_path:
                                                    _LOGGER.error(
                                                        "[OPTIM][nonfinite] dumped to: %s",
                                                        str(dump_path),
                                                    )
                                                if nonfinite_fail_fast:
                                                    raise RuntimeError(
                                                        f"Non-finite parameters detected (param={bad_p0}, epoch={int(epoch_idx)}, step_idx={int(step_idx)}, step={int(delta_gate_auto_step_total)})."
                                                    )
                                                if nonfinite_skip_step:
                                                    _LOGGER.error(
                                                        "[OPTIM][nonfinite] skipping optimizer step due to non-finite params (epoch=%d, step_idx=%d, step=%d)",
                                                        int(epoch_idx),
                                                        int(step_idx),
                                                        int(delta_gate_auto_step_total),
                                                    )
                                                    optimizer.zero_grad(set_to_none=True)
                                                    with contextlib.suppress(Exception):
                                                        scaler.update()
                                                    train_accum_since_last = 0
                                                    break
                                            bad = _first_nonfinite_grad(model_for_grads)
                                            if bad is not None:
                                                dump_path = _maybe_dump_nonfinite(
                                                    epoch=int(epoch_idx),
                                                    step_idx=int(step_idx),
                                                    step_total=int(
                                                        delta_gate_auto_step_total
                                                    ),
                                                    bad=str(bad),
                                                    X=X,
                                                    Y=Y,
                                                    Y_flat=Y_flat,
                                                    loss=loss_val,
                                                )
                                                _LOGGER.error(
                                                    "[OPTIM][nonfinite] pre-step grad non-finite: %s (epoch=%d, step_idx=%d, step=%d, opt=%s)",
                                                    bad,
                                                    int(epoch_idx),
                                                    int(step_idx),
                                                    int(
                                                        delta_gate_auto_step_total
                                                    ),
                                                    type(optimizer).__name__,
                                                )
                                                if dump_path:
                                                    _LOGGER.error(
                                                        "[OPTIM][nonfinite] dumped to: %s",
                                                        str(dump_path),
                                                    )
                                                if nonfinite_fail_fast:
                                                    raise RuntimeError(
                                                        f"Non-finite gradients detected (param={bad}, epoch={int(epoch_idx)}, step_idx={int(step_idx)}, step={int(delta_gate_auto_step_total)})."
                                                    )
                                                if nonfinite_skip_step:
                                                    _LOGGER.error(
                                                        "[OPTIM][nonfinite] skipping optimizer step (epoch=%d, step_idx=%d, step=%d)",
                                                        int(epoch_idx),
                                                        int(step_idx),
                                                        int(
                                                            delta_gate_auto_step_total
                                                        ),
                                                    )
                                                    optimizer.zero_grad(
                                                        set_to_none=True
                                                    )
                                                    with contextlib.suppress(
                                                        Exception
                                                    ):
                                                        scaler.update()
                                                    train_accum_since_last = 0
                                                    break
                                        scaler.step(optimizer)
                                        if strict_sanitize or nonfinite_fail_fast or bool(nonfinite_dump_dir):
                                            bad_p_post = _first_nonfinite_param(model_for_grads)
                                            if bad_p_post is not None:
                                                bad_s = _first_nonfinite_optim_state(optimizer)
                                                p_t = None
                                                with contextlib.suppress(Exception):
                                                    for n, p in model_for_grads.named_parameters(recurse=True):
                                                        if n == bad_p_post:
                                                            p_t = p
                                                            break
                                                extra = {"where": "post_step_param", "param": str(bad_p_post)}
                                                if bad_s is not None:
                                                    extra["optim_state_first_bad"] = str(bad_s)
                                                if torch.is_tensor(p_t):
                                                    with torch.no_grad():
                                                        tt = p_t.detach()
                                                        extra["dtype"] = str(tt.dtype)
                                                        with contextlib.suppress(Exception):
                                                            extra["shape"] = [int(x) for x in tuple(tt.shape)]
                                                        if tt.is_floating_point():
                                                            with contextlib.suppress(Exception):
                                                                extra["nan"] = int(torch.isnan(tt).sum().item())
                                                            with contextlib.suppress(Exception):
                                                                extra["inf"] = int(torch.isinf(tt).sum().item())
                                                dump_path = _maybe_dump_nonfinite(
                                                    epoch=int(epoch_idx),
                                                    step_idx=int(step_idx),
                                                    step_total=int(delta_gate_auto_step_total),
                                                    bad=str("post_step_param:" + str(bad_p_post)),
                                                    X=X,
                                                    Y=Y,
                                                    Y_flat=Y_flat,
                                                    loss=loss_val,
                                                    extra=extra,
                                                )
                                                _LOGGER.error(
                                                    "[OPTIM][nonfinite] post-step param non-finite: %s (epoch=%d, step_idx=%d, step=%d, opt=%s, state=%s)",
                                                    str(bad_p_post),
                                                    int(epoch_idx),
                                                    int(step_idx),
                                                    int(delta_gate_auto_step_total),
                                                    type(optimizer).__name__,
                                                    str(bad_s),
                                                )
                                                if dump_path:
                                                    _LOGGER.error("[OPTIM][nonfinite] dumped to: %s", str(dump_path))
                                                if strict_sanitize or nonfinite_fail_fast:
                                                    raise RuntimeError(
                                                        f"Non-finite parameter after optimizer.step: {bad_p_post} (epoch={int(epoch_idx)}, step_idx={int(step_idx)}, step={int(delta_gate_auto_step_total)})"
                                                    )
                                        if optim_diag and (int(delta_gate_auto_step_total) % int(optim_diag_every) == 0):
                                            bad_p = _first_nonfinite_param(model_for_grads)
                                            if bad_p is not None:
                                                bad_s = _first_nonfinite_optim_state(optimizer)
                                                _LOGGER.error("[OPTIM][nonfinite] post-step param non-finite: %s (step=%d, opt=%s, state=%s)", bad_p, int(delta_gate_auto_step_total), type(optimizer).__name__, str(bad_s))
                                        scaler.update()
                                        optimizer.zero_grad(set_to_none=True)
                                        ema_target = (
                                            model.module
                                            if hasattr(model, "module")
                                            else model
                                        )
                                        if ema_helper is not None:
                                            try:
                                                ema_helper.update(
                                                    ema_target,
                                                    optimizer=optimizer,
                                                )
                                            except Exception:
                                                pass
                                        if swa_helper is not None:
                                            try:
                                                swa_helper.update(ema_target)
                                            except Exception:
                                                pass
                                        target_for_step = (
                                            model.module
                                            if hasattr(model, "module")
                                            else model
                                        )
                                        inst_step = (
                                            to_submodule(target_for_step)
                                            or target_for_step
                                        )
                                        with contextlib.suppress(Exception):
                                            setattr(
                                                inst_step,
                                                "_enn_step_total",
                                                int(
                                                    delta_gate_auto_step_total
                                                ),
                                            )
                                        peak = None
                                        free = None
                                        total = None
                                        with contextlib.suppress(Exception):
                                            peak = int(
                                                accelerator_max_allocated_memory(
                                                    device
                                                )
                                            )
                                        with contextlib.suppress(Exception):
                                            free, total = _device_mem_get_info(
                                                device
                                            )
                                        if (
                                            isinstance(peak, int)
                                            and peak > 0
                                            and isinstance(total, int)
                                            and total
                                            and total > 0
                                        ):
                                            frac = float(peak) / float(total)
                                            prev_ema = float(
                                                getattr(
                                                    inst_step,
                                                    "_enn_peak_ema",
                                                    0.0,
                                                )
                                                or 0.0
                                            )
                                            ema = (
                                                frac
                                                if prev_ema <= 0.0
                                                else (
                                                    0.9 * prev_ema + 0.1 * frac
                                                )
                                            )
                                            with contextlib.suppress(
                                                Exception
                                            ):
                                                setattr(
                                                    inst_step,
                                                    "_enn_peak_ema",
                                                    float(ema),
                                                )
                                            spike = (
                                                prev_ema > 0.0
                                                and frac
                                                > max(0.92, prev_ema * 1.25)
                                            )
                                            if frac > 0.92 or spike:
                                                to_checkpoint(
                                                    model,
                                                    device=device,
                                                    step_total=int(
                                                        delta_gate_auto_step_total
                                                    ),
                                                    ttl_steps=128,
                                                    min_bytes=16 * 1024 * 1024,
                                                )
                                        from_checkpoint(
                                            model,
                                            step_total=int(
                                                delta_gate_auto_step_total
                                            ),
                                        )
                                        if scheduler_step_per_batch:
                                            with contextlib.suppress(
                                                Exception
                                            ):
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
                                                    _reduce_sum(
                                                        top_avg_t, train_pg
                                                    )
                                                    top_avg_t = (
                                                        top_avg_t / float(ws)
                                                    )
                                                if bottom_avg_t is not None:
                                                    _reduce_sum(
                                                        bottom_avg_t, train_pg
                                                    )
                                                    bottom_avg_t = (
                                                        bottom_avg_t
                                                        / float(ws)
                                                    )
                                            loss_controller.update(
                                                top_avg_t, bottom_avg_t
                                            )
                                        lw_top_sum = None
                                        lw_bottom_sum = None
                                        lw_count = 0
                                    step_flops = 0.0
                                    with contextlib.suppress(Exception):
                                        step_flops = max(
                                            0.0,
                                            float(
                                                train_counter.get_total_flops()
                                            ),
                                        )
                                    flops += float(step_flops)
                                    tflops_seen += 1
                                    if int(tflops_seen) > int(tflops_warmup):
                                        flops_tflops += float(step_flops)
                                        kern_tflops += float(t_kern_step)
                                    breakdown_getter = getattr(
                                        train_counter,
                                        "get_manual_breakdown",
                                        None,
                                    )
                                    if callable(breakdown_getter):
                                        for (
                                            name,
                                            value,
                                        ) in breakdown_getter().items():
                                            with contextlib.suppress(
                                                Exception
                                            ):
                                                flop_breakdown_epoch[name] = (
                                                    flop_breakdown_epoch.get(
                                                        name, 0.0
                                                    )
                                                    + float(value)
                                                )
                            if should_sync:
                                global_step += 1
                                delta_gate_auto_step_total += 1
                                with contextlib.suppress(Exception):
                                    target_for_autok = (
                                        model.module
                                        if hasattr(model, "module")
                                        else model
                                    )
                                    _set_gate_factor(
                                        target_for_autok,
                                        step=delta_gate_auto_step_total,
                                        pg=train_pg,
                                        local_rank=local_rank,
                                    )
                                with contextlib.suppress(Exception):
                                    maybe_upgrade = getattr(
                                        inst_step, "maybe_upgrade_compile_mode", None
                                    )
                                    if callable(maybe_upgrade):
                                        maybe_upgrade(
                                            step_total=int(
                                                delta_gate_auto_step_total
                                            ),
                                            logger=_LOGGER,
                                        )
                                match device.type:
                                    case "cuda":
                                        util_now, mem_now = (
                                            Monitor.gpu_nvml_utils(device)
                                        )
                                    case "xpu":
                                        util_now, mem_now = (
                                            None,
                                            Monitor.xpu_mem_util(device),
                                        )
                                    case "mps":
                                        util_now, mem_now = (
                                            None,
                                            Monitor.mps_mem_util(device),
                                        )
                                    case _:
                                        util_now, mem_now = (None, None)
                                if util_now is not None:
                                    util_now = float(util_now)
                                    if gpu_util_ema is None:
                                        gpu_util_ema = util_now
                                    else:
                                        gpu_util_ema = (
                                            (1.0 - util_alpha) * gpu_util_ema
                                            + util_alpha * util_now
                                        )
                                if mem_now is not None:
                                    mem_now = float(mem_now)
                                    if mem_util_ema is None:
                                        mem_util_ema = mem_now
                                    else:
                                        mem_util_ema = (
                                            (1.0 - util_alpha) * mem_util_ema
                                            + util_alpha * mem_now
                                        )
                                if (
                                    util_adjust_interval > 0
                                    and global_step >= util_warmup_steps
                                    and (
                                        global_step % util_adjust_interval == 0
                                    )
                                ):
                                    new_grad_accum = grad_accum_steps
                                    util_frac = None
                                    mem_frac = None
                                    if gpu_util_ema is not None:
                                        util_frac = max(
                                            0.0, min(1.0, gpu_util_ema / 100.0)
                                        )
                                    if mem_util_ema is not None:
                                        mem_frac = max(
                                            0.0, min(1.0, mem_util_ema / 100.0)
                                        )
                                    if util_frac is None:
                                        total_t_local = float(
                                            io_time + comp_time
                                        )
                                        if total_t_local > 0.0:
                                            util_frac = max(
                                                0.0,
                                                min(
                                                    1.0,
                                                    float(comp_time)
                                                    / total_t_local,
                                                ),
                                            )
                                        else:
                                            util_frac = 0.0
                                    if util_frac is not None:
                                        if mem_frac is not None:
                                            if (
                                                util_frac < 0.88
                                                and mem_frac < 0.9
                                            ):
                                                new_grad_accum = min(
                                                    max_grad_accum,
                                                    grad_accum_steps + 1,
                                                )
                                            elif (
                                                util_frac > 0.97
                                                or mem_frac > 0.92
                                            ):
                                                new_grad_accum = max(
                                                    min_grad_accum,
                                                    grad_accum_steps - 1,
                                                )
                                        elif util_frac < 0.88:
                                            new_grad_accum = min(
                                                max_grad_accum,
                                                grad_accum_steps + 1,
                                            )
                                        elif util_frac > 0.97:
                                            new_grad_accum = max(
                                                min_grad_accum,
                                                grad_accum_steps - 1,
                                            )
                                    host_avail_now = None
                                    host_total_now = None
                                    with contextlib.suppress(Exception):
                                        host_avail_now = Memory.available()
                                        host_total_now = Memory.total()
                                    host_low = False
                                    if (
                                        host_avail_now is not None
                                        and host_avail_now > 0
                                    ):
                                        raw_min_free = str(
                                            os.environ.get(
                                                "ENN_HOST_MIN_FREE_MB", ""
                                            )
                                            or ""
                                        ).strip()
                                        min_free_mb = None
                                        if raw_min_free:
                                            with contextlib.suppress(Exception):
                                                min_free_mb = int(raw_min_free)
                                        ratio = 0.30
                                        with contextlib.suppress(Exception):
                                            ratio = float(
                                                os.environ.get(
                                                    "ENN_HOST_MIN_FREE_RATIO",
                                                    ratio,
                                                )
                                                or ratio
                                            )
                                        ratio = max(0.05, min(0.50, float(ratio)))
                                        cap_mb = 4096
                                        with contextlib.suppress(Exception):
                                            cap_mb = int(
                                                os.environ.get(
                                                    "ENN_HOST_MIN_FREE_MB_CAP",
                                                    cap_mb,
                                                )
                                                or cap_mb
                                            )
                                        cap_mb = max(1024, int(cap_mb))
                                        if min_free_mb is None:
                                            if (
                                                host_total_now is not None
                                                and host_total_now > 0
                                            ):
                                                total_mb = int(
                                                    int(host_total_now)
                                                    // (1024 * 1024)
                                                )
                                                cand = int(total_mb * ratio)
                                                min_free_mb = max(
                                                    1024,
                                                    min(int(cap_mb), int(cand)),
                                                )
                                            else:
                                                min_free_mb = 1024
                                        host_low = (
                                            int(host_avail_now)
                                            < int(min_free_mb) * 1024 * 1024
                                        )
                                    if host_low:
                                        with contextlib.suppress(Exception):
                                            if cpu_pool is not None and hasattr(
                                                cpu_pool, "collect"
                                            ):
                                                cpu_pool.collect()
                                        with contextlib.suppress(Exception):
                                            time.sleep(
                                                float(
                                                    os.environ.get(
                                                        "ENN_HOST_PRESSURE_YIELD_S",
                                                        "0.001",
                                                    )
                                                    or 0.001
                                                )
                                            )
                                        try:
                                            cd = float(
                                                os.environ.get(
                                                    "ENN_HOST_PRESSURE_HEAVY_COOLDOWN_SEC",
                                                    "30",
                                                )
                                                or 30.0
                                            )
                                        except Exception:
                                            cd = 30.0
                                        cd = max(1.0, float(cd))
                                        last = float(
                                            getattr(
                                                ops,
                                                "_enn_host_pressure_last_heavy",
                                                0.0,
                                            )
                                            or 0.0
                                        )
                                        now = time.monotonic()
                                        if now - last >= cd:
                                            with contextlib.suppress(Exception):
                                                setattr(
                                                    ops,
                                                    "_enn_host_pressure_last_heavy",
                                                    float(now),
                                                )
                                            if env_bool(
                                                "ENN_HOST_PRESSURE_GC",
                                                default=False,
                                            ):
                                                with contextlib.suppress(Exception):
                                                    import gc

                                                    gc.collect()
                                            if env_bool(
                                                "ENN_HOST_PRESSURE_MALLOC_TRIM",
                                                default=False,
                                            ):
                                                with contextlib.suppress(Exception):
                                                    import ctypes
                                                    import platform as _platform

                                                    sysname = _platform.system()
                                                    if sysname == "Linux":
                                                        libc = ctypes.CDLL(
                                                            "libc.so.6"
                                                        )
                                                        trim = getattr(
                                                            libc, "malloc_trim", None
                                                        )
                                                        if callable(trim):
                                                            trim(0)
                                                    elif sysname == "Windows":
                                                        msvcrt = ctypes.CDLL(
                                                            "msvcrt.dll"
                                                        )
                                                        heapmin = getattr(
                                                            msvcrt, "_heapmin", None
                                                        )
                                                        if callable(heapmin):
                                                            heapmin()
                                                    elif sysname == "Darwin":
                                                        libsys = ctypes.CDLL(
                                                            "libsystem_malloc.dylib"
                                                        )
                                                        fn = getattr(
                                                            libsys,
                                                            "malloc_zone_pressure_relief",
                                                            None,
                                                        )
                                                        if callable(fn):
                                                            fn(None, 0)
                                            if env_bool(
                                                "ENN_HOST_PRESSURE_FADVISE_OPEN_FDS",
                                                default=False,
                                            ):
                                                with contextlib.suppress(Exception):
                                                    import os as _os
                                                    import stat as _stat

                                                    if (
                                                        hasattr(_os, "posix_fadvise")
                                                        and hasattr(
                                                            _os,
                                                            "POSIX_FADV_DONTNEED",
                                                        )
                                                        and _os.path.isdir(
                                                            "/proc/self/fd"
                                                        )
                                                    ):
                                                        for _ent in _os.listdir(
                                                            "/proc/self/fd"
                                                        ):
                                                            if not _ent.isdigit():
                                                                continue
                                                            _fd = int(_ent)
                                                            if _fd <= 2:
                                                                continue
                                                            with contextlib.suppress(
                                                                Exception
                                                            ):
                                                                st = _os.fstat(_fd)
                                                                if not _stat.S_ISREG(
                                                                    int(st.st_mode)
                                                                ):
                                                                    continue
                                                                _os.posix_fadvise(
                                                                    _fd,
                                                                    0,
                                                                    0,
                                                                    _os.POSIX_FADV_DONTNEED,
                                                                )
                                        if new_grad_accum > grad_accum_steps:
                                            new_grad_accum = grad_accum_steps
                                        if grad_accum_steps > min_grad_accum:
                                            new_grad_accum = min_grad_accum
                                    if new_grad_accum != grad_accum_steps:
                                        new_grad_accum = broadcast_scalar(
                                            new_grad_accum,
                                            device=device,
                                            src=0,
                                        )
                                        _LOGGER.info(
                                            "[epochs] adjusted grad_accum_steps=%d (gpu_util_ema=%s, mem_util_ema=%s)",
                                            int(new_grad_accum),
                                            str(gpu_util_ema),
                                            str(mem_util_ema),
                                        )
                                        grad_accum_steps = new_grad_accum
                            if (
                                use_timer
                                and comp_ev_s is not None
                                and comp_ev_e is not None
                            ):
                                comp_ev_e.record()
                                comp_ev_e.synchronize()
                                comp_time += (
                                    float(comp_ev_s.elapsed_time(comp_ev_e))
                                    / 1000.0
                                )
                                if (
                                    kern_ev_s is not None
                                    and kern_ev_e is not None
                                ):
                                    with contextlib.suppress(Exception):
                                        t_kern_step = float(
                                            kern_ev_s.elapsed_time(kern_ev_e)
                                        ) / 1000.0
                            else:
                                if timer_sync:
                                    sync_accelerator(device)
                                comp_time += (
                                    time.perf_counter_ns() - t_comp_s
                                ) / 1000000000.0
                            kern_time += float(t_kern_step)
                            if local_rank == 0 and should_sync:
                                io_elapsed = prev_io_time + float(io_time)
                                io_transferred = prev_io_bytes + float(
                                    io_bytes
                                )
                                comp_elapsed = prev_comp_time + float(
                                    comp_time
                                )
                                mbps_cur = (
                                    io_transferred
                                    / max(io_elapsed, 1e-06)
                                    / MB_DIV
                                )
                                flops_used = prev_flops_tflops + float(
                                    flops_tflops
                                )
                                kern_used = prev_kern_tflops + float(
                                    kern_tflops
                                )
                                denom = (
                                    kern_used
                                    if kern_used > 0.0
                                    else (
                                        prev_kern_time + float(kern_time)
                                    )
                                )
                                if denom <= 0.0:
                                    denom = comp_elapsed
                                tflops_cur = (
                                    flops_used
                                    / max(float(denom), 1e-06)
                                    / 1000000000000.0
                                )
                                ProcessBroker.update_progress_bar(
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
                                        or step_idx
                                        % max(1, int(train_steps * 0.01))
                                        == 0
                                    ):
                                        x_rec = X
                                        y_rec = Y
                                        with contextlib.suppress(Exception):
                                            model_for_scaler = (
                                                model.module
                                                if hasattr(model, "module")
                                                else model
                                            )
                                            data_scaler = getattr(
                                                model_for_scaler,
                                                "scaler",
                                                None,
                                            )
                                            if data_scaler is not None:
                                                dy = getattr(
                                                    data_scaler,
                                                    "denormalize_y",
                                                    None,
                                                )
                                                if callable(dy):
                                                    y_flat = y_rec
                                                    if (
                                                        isinstance(
                                                            y_flat,
                                                            torch.Tensor,
                                                        )
                                                        and y_flat.ndim != 2
                                                    ):
                                                        y_flat = (
                                                            y_flat.reshape(
                                                                y_flat.shape[
                                                                    0
                                                                ],
                                                                -1,
                                                            )
                                                        )
                                                    need_denorm = True
                                                    scale_max = getattr(
                                                        meta,
                                                        "scale_max_value",
                                                        None,
                                                    ) or getattr(
                                                        meta,
                                                        "scale_max_abs",
                                                        None,
                                                    )
                                                    if (
                                                        isinstance(
                                                            y_flat,
                                                            torch.Tensor,
                                                        )
                                                        and scale_max
                                                        is not None
                                                    ):
                                                        with contextlib.suppress(
                                                            Exception
                                                        ):
                                                            max_obs = float(
                                                                y_flat.detach()
                                                                .abs()
                                                                .max()
                                                                .item()
                                                            )
                                                            if (
                                                                max_obs
                                                                >= float(
                                                                    scale_max
                                                                )
                                                                * 0.5
                                                            ):
                                                                need_denorm = (
                                                                    False
                                                                )
                                                    if need_denorm:
                                                        y_rec = dy(
                                                            y_flat
                                                        ).view_as(y_rec)
                                        hist.record_batch(x_rec, y_rec)
                                except Exception:
                                    pass
                            if torch_prof is not None:
                                torch_prof.step()
                            t_fetch_start = time.perf_counter_ns()
                            if (
                                cpu_pool is not None
                                and step_idx + 1 & 255 == 0
                            ):
                                with contextlib.suppress(Exception):
                                    cpu_pool.collect()
                            with contextlib.suppress(Exception):
                                _clear_oom_retries(
                                    train_loader, "train", step_idx
                                )
                            break
                        except RuntimeError as e:
                            if is_oom_error(e):
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
                        finally:
                            if mark_cudagraph:
                                cudagraph_mark_step_end()
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
            if train_accum_since_last > 0:
                if (
                    is_distributed()
                    and world_sz > 1
                    and (ddp_fallback or (max(1, grad_accum_steps) > 1))
                ):
                    distributed_all_reduce_grads(
                        model_for_grads,
                        average=True,
                        policy=dist_policy.collective,
                    )
                scaler.unscale_(optimizer)
                diag_now = (
                    int(delta_gate_auto_step_total) % int(optim_diag_every)
                ) == 0
                bad = None
                if (
                    optim_diag
                    or nonfinite_fail_fast
                    or nonfinite_skip_step
                    or bool(nonfinite_dump_dir)
                ) and (
                    diag_now
                    or nonfinite_fail_fast
                    or nonfinite_skip_step
                    or bool(nonfinite_dump_dir)
                ):
                    bad_p0 = _first_nonfinite_param(model_for_grads)
                    if bad_p0 is not None:
                        dump_path = _maybe_dump_nonfinite(
                            epoch=int(epoch_idx),
                            step_idx=int(step_idx),
                            step_total=int(delta_gate_auto_step_total),
                            bad=str("param:" + str(bad_p0)),
                        )
                        _LOGGER.error(
                            "[OPTIM][nonfinite] pre-step param non-finite: %s (epoch=%d, step_idx=%d, step=%d, opt=%s)",
                            bad_p0,
                            int(epoch_idx),
                            int(step_idx),
                            int(delta_gate_auto_step_total),
                            type(optimizer).__name__,
                        )
                        if dump_path:
                            _LOGGER.error("[OPTIM][nonfinite] dumped to: %s", str(dump_path))
                        if nonfinite_fail_fast:
                            raise RuntimeError(
                                f"Non-finite parameters detected (param={bad_p0}, epoch={int(epoch_idx)}, step_idx={int(step_idx)}, step={int(delta_gate_auto_step_total)})."
                            )
                        if nonfinite_skip_step:
                            optimizer.zero_grad(set_to_none=True)
                            with contextlib.suppress(Exception):
                                scaler.update()
                            train_accum_since_last = 0
                            bad = str("param:" + str(bad_p0))
                    bad_g = _first_nonfinite_grad(model_for_grads)
                    if bad_g is not None:
                        if bad is None:
                            bad = bad_g
                        dump_path = _maybe_dump_nonfinite(
                            epoch=int(epoch_idx),
                            step_idx=int(step_idx),
                            step_total=int(delta_gate_auto_step_total),
                            bad=str(bad_g),
                        )
                        _LOGGER.error(
                            "[OPTIM][nonfinite] pre-step grad non-finite: %s (epoch=%d, step_idx=%d, step=%d, opt=%s)",
                            bad_g,
                            int(epoch_idx),
                            int(step_idx),
                            int(delta_gate_auto_step_total),
                            type(optimizer).__name__,
                        )
                        if dump_path:
                            _LOGGER.error(
                                "[OPTIM][nonfinite] dumped to: %s",
                                str(dump_path),
                            )
                        if nonfinite_fail_fast:
                            raise RuntimeError(
                                f"Non-finite gradients detected (param={bad_g}, epoch={int(epoch_idx)}, step_idx={int(step_idx)}, step={int(delta_gate_auto_step_total)})."
                            )
                        if nonfinite_skip_step:
                            _LOGGER.error(
                                "[OPTIM][nonfinite] skipping optimizer step (epoch=%d, step_idx=%d, step=%d)",
                                int(epoch_idx),
                                int(step_idx),
                                int(delta_gate_auto_step_total),
                            )
                            optimizer.zero_grad(set_to_none=True)
                            with contextlib.suppress(Exception):
                                scaler.update()
                            train_accum_since_last = 0
                if bad is None or (not nonfinite_skip_step):
                    scaler.step(optimizer)
                    if strict_sanitize or nonfinite_fail_fast or bool(nonfinite_dump_dir):
                        bad_p_post = _first_nonfinite_param(model_for_grads)
                        if bad_p_post is not None:
                            bad_s = _first_nonfinite_optim_state(optimizer)
                            extra = {"where": "post_step_param_tail", "param": str(bad_p_post)}
                            if bad_s is not None:
                                extra["optim_state_first_bad"] = str(bad_s)
                            dump_path = _maybe_dump_nonfinite(
                                epoch=int(epoch_idx),
                                step_idx=int(step_idx),
                                step_total=int(delta_gate_auto_step_total),
                                bad=str("post_step_param:" + str(bad_p_post)),
                                extra=extra,
                            )
                            _LOGGER.error(
                                "[OPTIM][nonfinite] post-step param non-finite (tail): %s (epoch=%d, step_idx=%d, step=%d, opt=%s, state=%s)",
                                str(bad_p_post),
                                int(epoch_idx),
                                int(step_idx),
                                int(delta_gate_auto_step_total),
                                type(optimizer).__name__,
                                str(bad_s),
                            )
                            if dump_path:
                                _LOGGER.error("[OPTIM][nonfinite] dumped to: %s", str(dump_path))
                            if strict_sanitize or nonfinite_fail_fast:
                                raise RuntimeError(
                                    f"Non-finite parameter after optimizer.step (tail): {bad_p_post} (epoch={int(epoch_idx)}, step_idx={int(step_idx)}, step={int(delta_gate_auto_step_total)})"
                                )
                if optim_diag and (int(delta_gate_auto_step_total) % int(optim_diag_every) == 0):
                    bad_p = _first_nonfinite_param(model_for_grads)
                    if bad_p is not None:
                        bad_s = _first_nonfinite_optim_state(optimizer)
                        _LOGGER.error("[OPTIM][nonfinite] post-step param non-finite: %s (step=%d, opt=%s, state=%s)", bad_p, int(delta_gate_auto_step_total), type(optimizer).__name__, str(bad_s))
                if bad is None or (not nonfinite_skip_step):
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler_step_per_batch:
                        with contextlib.suppress(Exception):
                            sched.step()
                    if local_rank == 0 and status_bar is not None:
                        io_elapsed = prev_io_time + float(io_time)
                        io_transferred = prev_io_bytes + float(io_bytes)
                        comp_elapsed = prev_comp_time + float(comp_time)
                        mbps_cur = (
                            io_transferred / max(io_elapsed, 1e-06) / MB_DIV
                        )
                        flops_used = prev_flops_tflops + float(flops_tflops)
                        kern_used = prev_kern_tflops + float(kern_tflops)
                        denom = (
                            kern_used
                            if kern_used > 0.0
                            else (prev_kern_time + float(kern_time))
                        )
                        if denom <= 0.0:
                            denom = comp_elapsed
                        tflops_cur = (
                            flops_used
                            / max(float(denom), 1e-06)
                            / 1000000000000.0
                        )
                        ProcessBroker.update_progress_bar(
                            status_bar,
                            finish=train_accum_since_last,
                            mbps=mbps_cur,
                            tflops=tflops_cur,
                        )
                    train_accum_since_last = 0
            if val_loader is not None and flop_counter_val is not None:
                with flop_counter_val:
                    model.eval()
                    with inference_mode(model), StatelessAutocast.float(device):
                        t_fetch_start = time.perf_counter_ns()
                        for _vstep, _raw in enumerate(val_loader):
                            while True:
                                mark_cudagraph = False
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
                                    wait_s = (
                                        t_ready - t_fetch_start
                                    ) / 1000000000.0
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
                                    with flop_counter_val.step(
                                        display=False
                                    ) as val_counter:
                                        if getattr(device, "type", None) == "cuda":
                                            cudagraph_mark_step_begin()
                                            mark_cudagraph = True
                                        with StatelessAutocast.float(device):
                                            Yv_flat = Y.reshape(
                                                Y.shape[0], -1
                                            ).to(
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
                                    if _loss_val is None:
                                        raise RuntimeError(
                                            "Model returned no loss value during validation. Ensure loss functions are configured correctly."
                                        )
                                    if not isinstance(_loss_val, torch.Tensor):
                                        _loss_val = torch.as_tensor(
                                            _loss_val,
                                            device=device,
                                            dtype=param_dtype,
                                        )
                                    else:
                                        _loss_val = _loss_val.to(
                                            device=device, dtype=param_dtype
                                        )
                                    if (
                                        use_timer
                                        and comp_ev_s is not None
                                        and comp_ev_e is not None
                                    ):
                                        comp_ev_e.record()
                                        comp_ev_e.synchronize()
                                        comp_time += (
                                            float(
                                                comp_ev_s.elapsed_time(
                                                    comp_ev_e
                                                )
                                            )
                                            / 1000.0
                                        )
                                    else:
                                        if timer_sync:
                                            sync_accelerator(device)
                                        comp_time += (
                                            time.perf_counter_ns() - t_comp_s
                                        ) / 1000000000.0
                                    with contextlib.suppress(Exception):
                                        flops += max(
                                            0.0,
                                            float(
                                                val_counter.get_total_flops()
                                            ),
                                        )
                                    breakdown_getter = getattr(
                                        val_counter,
                                        "get_manual_breakdown",
                                        None,
                                    )
                                    if callable(breakdown_getter):
                                        for (
                                            name,
                                            value,
                                        ) in breakdown_getter().items():
                                            with contextlib.suppress(
                                                Exception
                                            ):
                                                flop_breakdown_epoch[name] = (
                                                    flop_breakdown_epoch.get(
                                                        name, 0.0
                                                    )
                                                    + float(value)
                                                )
                                    if local_rank == 0:
                                        io_elapsed = prev_io_time + float(
                                            io_time
                                        )
                                        io_transferred = prev_io_bytes + float(
                                            io_bytes
                                        )
                                        comp_elapsed = prev_comp_time + float(
                                            comp_time
                                        )
                                        flop_total = prev_flops + float(flops)
                                        mbps_cur = (
                                            io_transferred
                                            / max(io_elapsed, 1e-06)
                                            / MB_DIV
                                        )
                                        tflops_cur = (
                                            flop_total
                                            / max(comp_elapsed, 1e-06)
                                            / 1000000000000.0
                                        )
                                        ProcessBroker.update_progress_bar(
                                            status_bar,
                                            finish=1,
                                            mbps=mbps_cur,
                                            tflops=tflops_cur,
                                        )
                                    if torch_prof is not None:
                                        torch_prof.step()
                                    t_fetch_start = time.perf_counter_ns()
                                    if (
                                        cpu_pool is not None
                                        and _vstep + 1 & 255 == 0
                                    ):
                                        with contextlib.suppress(Exception):
                                            cpu_pool.collect()
                                    with contextlib.suppress(Exception):
                                        _clear_oom_retries(
                                            val_loader, "val", _vstep
                                        )
                                    break
                                except RuntimeError as e:
                                    if is_oom_error(e):
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
                                finally:
                                    if mark_cudagraph:
                                        cudagraph_mark_step_end()
            if is_distributed():
                stats_dtype = (
                    param_dtype
                    if isinstance(param_dtype, torch.dtype)
                    and param_dtype.is_floating_point
                    else torch.float64
                )
                stats = torch.tensor(
                    [comp_time, io_time, flops, io_bytes, train_samples_epoch],
                    device=device,
                    dtype=stats_dtype,
                )
                torch.distributed.all_reduce(
                    stats, op=torch.distributed.ReduceOp.SUM
                )
                world = max(1, get_world_size(device))
                stats /= world
                stats_cpu = stats.detach().cpu()
                comp_time = float(stats_cpu[0].item())
                io_time = float(stats_cpu[1].item())
                flops = float(stats_cpu[2].item())
                io_bytes = float(stats_cpu[3].item())
                train_samples_epoch = float(stats_cpu[4].item())
                (
                    distributed_barrier(
                        device, group=get_accel_group(device), lane="accelerator"
                    )
                    if get_world_size(device) > 1
                    else None
                )
            if not scheduler_step_per_batch:
                try:
                    sched.step()
                except Exception:
                    pass

            if checkpointer is not None:
                checkpointer.poll()
                ckpt_participate = True
                ckpt_pg = None
                use_collectives = bool(is_distributed())
                if is_distributed():
                    try:
                        mesh, mesh_kind = get_distributed_mesh(device)
                        if mesh_kind == "hsdp2" and mesh is not None:
                            coord = None
                            with contextlib.suppress(Exception):
                                coord = mesh.get_coordinate()
                            if not coord or int(coord[0]) != 0:
                                ckpt_participate = False
                            else:
                                with contextlib.suppress(Exception):
                                    ckpt_pg = mesh.get_group(mesh_dim="dp_shard")
                                if ckpt_pg is None:
                                    with contextlib.suppress(Exception):
                                        ckpt_pg = mesh.get_group("dp_shard")
                    except Exception:
                        pass
                    if ckpt_pg is None:
                        ckpt_pg = train_pg

                if ckpt_participate:
                    did_start_ckpt = bool(
                        checkpointer.try_request_save_epoch_collective(
                            epoch=int(epoch_idx + 1),
                            model=model,
                            optimizer=optimizer,
                            save_optimizer=getattr(ops, "ckpt_save_optimizer", None),
                            extra_state={"epoch": int(epoch_idx + 1)},
                            block_if_busy=False,
                            device=device,
                            group=ckpt_pg,
                            process_group=ckpt_pg,
                            use_collectives=use_collectives,
                        )
                    )
                    if did_start_ckpt:
                        checkpointer.await_staging()
            prev_comp_time += float(comp_time)
            prev_kern_time += float(kern_time)
            prev_io_time += float(io_time)
            prev_flops += float(flops)
            prev_io_bytes += float(io_bytes)
            prev_samples += float(train_samples_epoch)
            prev_flops_tflops += float(flops_tflops)
            prev_kern_tflops += float(kern_tflops)
    if torch_prof is not None:
        with contextlib.suppress(Exception):
            torch_prof.stop()
        _get_profiler_summary(
            torch_prof, device=device, logger=_LOGGER, header="train/val"
        )
    if local_rank == 0 and status_bar is not None:
        mbps = prev_io_bytes / max(prev_io_time, 1e-06) / MB_DIV
        denom = prev_kern_tflops if prev_kern_tflops > 0.0 else prev_kern_time
        if denom <= 0.0:
            denom = prev_comp_time
        tflops = prev_flops_tflops / max(float(denom), 1e-06) / 1000000000000.0
        status_bar.set_postfix_str(
            f"{mbps:.2f} MB/s, {tflops:.2f} TFLOPS", refresh=True
        )
        status_bar.close()

    model_for_scaler = model.module if hasattr(model, "module") else model
    scaler_y_device = model_for_scaler.scaler.y_mean.device
    calib_src = calibration_loader if calibration_loader is not None else (val_loader or train_loader)
    if calib_src is None:
        end_kst_ns = posix_time()
        return None

    max_batches = int(
        env_first_int(("ENN_CALIB_MAX_BATCHES", "ENN_CALIB_MAX_STEPS"), default=32)
        or 32
    )
    max_samples = int(env_first_int(("ENN_CALIB_MAX_SAMPLES",), default=2048) or 2048)
    if max_batches < 0:
        max_batches = 0
    if max_samples < 0:
        max_samples = 0

    def _iter_raw(loader_obj: object):
        node_obj = getattr(loader_obj, "_node", None)
        if node_obj is not None:
            try:
                import torchdata.nodes as _tdn

                base = (
                    node_obj
                    if isinstance(node_obj, _tdn.Loader)
                    else _tdn.Loader(node_obj)
                )
                with contextlib.suppress(Exception):
                    base.reset(None)
                return iter(base)
            except Exception:
                pass
        base_it = getattr(loader_obj, "_base_iterable", None)
        if base_it is not None:
            with contextlib.suppress(Exception):
                base_it.reset(None)
            return iter(base_it)
        return iter(loader_obj)

    def _coerce_targets_to_B(y_raw: torch.Tensor, B: int) -> torch.Tensor:
        if y_raw.ndim == 0:
            return y_raw.view(1, 1).expand(max(1, int(B)), 1)
        if y_raw.ndim == 1:
            if int(y_raw.shape[0]) == int(B):
                return y_raw.view(int(B), 1)
            if int(B) == 1:
                return y_raw.view(1, -1)
            raise RuntimeError(
                "Calibration: 1D target length does not match batch size. "
                f"target.shape={tuple(y_raw.shape)}, batch={int(B)}"
            )
        if int(y_raw.shape[0]) == int(B):
            return y_raw
        if int(B) == 1:
            return y_raw.unsqueeze(0)
        candidates = [
            dim_idx
            for dim_idx, dim_size in enumerate(y_raw.shape[1:], start=1)
            if int(dim_size) == int(B)
        ]
        if not candidates:
            raise RuntimeError(
                "Calibration: could not infer batch axis for target tensor. "
                f"target.shape={tuple(y_raw.shape)}, batch={int(B)}"
            )
        if len(candidates) > 1:
            raise RuntimeError(
                "Calibration: ambiguous batch axis for target tensor "
                f"(candidates={candidates}). Provide targets in (B, ...) layout. "
                f"target.shape={tuple(y_raw.shape)}, batch={int(B)}"
            )
        return y_raw.movedim(int(candidates[0]), 0)

    def _finalize_affine(total_n: int, sum_x, sum_y, sum_x2, sum_xy) -> None:
        if (
            int(total_n) <= 0
            or sum_x is None
            or sum_y is None
            or sum_x2 is None
            or sum_xy is None
        ):
            return
        N = float(int(total_n))
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

    planned = int(_get_batch_length(calib_src) or 0)
    if int(max_batches) > 0 and planned > int(max_batches):
        planned = int(max_batches)
    calib_bar = (
        ProcessBroker.get_progress_bar(
            title="Calibration", total=int(planned), device=device, leave=False
        )
        if local_rank == 0 and int(planned) > 0
        else None
    )

    sum_x = sum_y = sum_x2 = sum_xy = None
    total_n = 0
    target_chunk_bytes = 16 * 1024 * 1024
    with contextlib.suppress(Exception):
        target_chunk_bytes = int(
            env_first_int(
                ("ENN_CALIB_CHUNK_BYTES", "ENN_CALIB_TARGET_CHUNK_BYTES"),
                default=target_chunk_bytes,
            )
            or target_chunk_bytes
        )
    target_chunk_bytes = int(target_chunk_bytes)
    if target_chunk_bytes < 0:
        target_chunk_bytes = 0

    accum_spec = str(env_str("ENN_CALIB_ACCUM_DEVICE") or "cpu").strip().lower()
    if accum_spec in {"cpu", ""}:
        accum_device = torch.device("cpu")
    elif accum_spec in {"scaler", "scaler_y", "y"}:
        accum_device = scaler_y_device
    elif accum_spec in {"device", "train", "accel"}:
        accum_device = device
    else:
        with contextlib.suppress(Exception):
            accum_device = torch.device(accum_spec)
        if "accum_device" not in locals():
            accum_device = torch.device("cpu")

    def _to_accum(v: torch.Tensor) -> torch.Tensor:
        if v.dtype is not torch.float64 or v.device != accum_device:
            return v.to(device=accum_device, dtype=torch.float64)
        return v

    disable_calib_compile = env_bool(
        "ENN_CALIB_DISABLE_COMPILE", default=bool(CPU.is_optimized_for_no_gil())
    )
    _dynamo_disable = None
    if disable_calib_compile:
        with contextlib.suppress(Exception):
            comp = getattr(torch, "compiler", None)
            cand = getattr(comp, "disable", None) if comp is not None else None
            if callable(cand):
                _dynamo_disable = cand
        if _dynamo_disable is None:
            with contextlib.suppress(Exception):
                torch_dynamo = importlib.import_module("torch._dynamo")
                cand = getattr(torch_dynamo, "disable", None)
                if callable(cand):
                    _dynamo_disable = cand

    def _dist_all_reduce_sum_(t: torch.Tensor) -> None:
        if not is_distributed():
            return
        try:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            return
        except Exception:
            pass
        try:
            if t.device.type == "cpu":
                td = t.to(device=device)
                torch.distributed.all_reduce(td, op=torch.distributed.ReduceOp.SUM)
                t.copy_(td.to(device="cpu"))
                return
        except Exception:
            raise

    seen_batches = 0
    seen_samples = 0

    def _dist_all_reduce_min_(t: torch.Tensor) -> None:
        if not is_distributed():
            return
        exc0: Exception | None = None
        try:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
            return
        except Exception as exc:
            exc0 = exc

        td = t
        moved = False
        try:
            backend = str(torch.distributed.get_backend() or "").lower()
        except Exception:
            backend = ""
        if t.device.type == "cpu" and backend == "nccl":
            try:
                dev = device
                if getattr(dev, "type", "cpu") != "cuda" and torch.cuda.is_available():
                    dev = torch.device("cuda", torch.cuda.current_device())
                td = t.to(device=dev)
                moved = True
                torch.distributed.all_reduce(td, op=torch.distributed.ReduceOp.MIN)
                t.copy_(td.to(device="cpu"))
                return
            except Exception as exc:
                exc0 = exc

        try:
            if t.device.type == "cpu" and backend == "nccl" and not moved:
                dev = device
                if getattr(dev, "type", "cpu") != "cuda" and torch.cuda.is_available():
                    dev = torch.device("cuda", torch.cuda.current_device())
                td = t.to(device=dev)
                moved = True
            world = int(torch.distributed.get_world_size())
            bufs = [torch.empty_like(td) for _ in range(world)]
            torch.distributed.all_gather(bufs, td)
            out = torch.stack(bufs, dim=0).min(dim=0).values
            if moved:
                t.copy_(out.to(device="cpu"))
            else:
                t.copy_(out)
            return
        except Exception as exc:
            raise RuntimeError(
                "[ENN] distributed calibration: failed to synchronize observed label MIN across ranks"
            ) from (exc0 if exc0 is not None else exc)

    def _dist_all_reduce_max_(t: torch.Tensor) -> None:
        if not is_distributed():
            return
        exc0: Exception | None = None
        try:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
            return
        except Exception as exc:
            exc0 = exc

        td = t
        moved = False
        try:
            backend = str(torch.distributed.get_backend() or "").lower()
        except Exception:
            backend = ""
        if t.device.type == "cpu" and backend == "nccl":
            try:
                dev = device
                if getattr(dev, "type", "cpu") != "cuda" and torch.cuda.is_available():
                    dev = torch.device("cuda", torch.cuda.current_device())
                td = t.to(device=dev)
                moved = True
                torch.distributed.all_reduce(td, op=torch.distributed.ReduceOp.MAX)
                t.copy_(td.to(device="cpu"))
                return
            except Exception as exc:
                exc0 = exc

        try:
            if t.device.type == "cpu" and backend == "nccl" and not moved:
                dev = device
                if getattr(dev, "type", "cpu") != "cuda" and torch.cuda.is_available():
                    dev = torch.device("cuda", torch.cuda.current_device())
                td = t.to(device=dev)
                moved = True
            world = int(torch.distributed.get_world_size())
            bufs = [torch.empty_like(td) for _ in range(world)]
            torch.distributed.all_gather(bufs, td)
            out = torch.stack(bufs, dim=0).max(dim=0).values
            if moved:
                t.copy_(out.to(device="cpu"))
            else:
                t.copy_(out)
            return
        except Exception as exc:
            raise RuntimeError(
                "[ENN] distributed calibration: failed to synchronize observed label MAX across ranks"
            ) from (exc0 if exc0 is not None else exc)

    def _run_calibration(
        sum_x, sum_y, sum_x2, sum_xy, total_n: int, seen_batches: int, seen_samples: int
    ):
        model.eval()
        with inference_mode(model), StatelessAutocast.float(device):
            for batch in _iter_raw(calib_src):
                if int(max_batches) > 0 and int(seen_batches) >= int(max_batches):
                    break
                if int(max_samples) > 0 and int(seen_samples) >= int(max_samples):
                    break

                x_b, y_b = collate.get_row(batch, labels_required=True)
                x_raw = torch.atleast_2d(x_b.to(device))
                B = int(x_raw.shape[0])

                y_raw = (
                    y_b.to(scaler_y_device)
                    if isinstance(y_b, torch.Tensor)
                    else torch.as_tensor(y_b, device=scaler_y_device)
                )
                y_raw = _coerce_targets_to_B(y_raw, B)
                y_flat = y_raw.reshape(int(B), -1)

                y_for_loss = y_flat.to(
                    device=device, dtype=param_dtype, non_blocking=non_blocking_ok
                )
                out = model(
                    x_raw,
                    labels_flat=y_for_loss,
                    global_loss=top_loss,
                    local_loss=bottom_loss,
                    loss_weights=loss_controller.weights(),
                    calibrate_output=False,
                )
                z_pred_raw = out[0] if isinstance(out, tuple) else out
                z_pred = z_pred_raw.detach()
                if z_pred.device != scaler_y_device:
                    z_pred = z_pred.to(device=scaler_y_device)
                z_pred = (
                    z_pred.reshape(z_pred.shape[0], -1)
                    if z_pred.ndim >= 2
                    else z_pred.view(-1, 1)
                )

                z_true = model_for_scaler.scaler.normalize_y(y_flat.detach())
                if z_true.device != scaler_y_device:
                    z_true = z_true.to(device=scaler_y_device)
                z_true = (
                    z_true.reshape(z_true.shape[0], -1)
                    if z_true.ndim >= 2
                    else z_true.view(-1, 1)
                )

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
                            "Calibration: feature dim mismatch cannot be reconciled. "
                            f"z_pred.shape={tuple(z_pred.shape)}, z_true.shape={tuple(z_true.shape)}"
                        )
                if z_pred.shape[0] != z_true.shape[0]:
                    raise RuntimeError(
                        "Calibration: batch dim mismatch. "
                        f"z_pred.shape={tuple(z_pred.shape)}, z_true.shape={tuple(z_true.shape)}"
                    )
                if z_pred.numel() == 0 or z_true.numel() == 0:
                    seen_batches += 1
                    seen_samples += int(B)
                    if calib_bar is not None:
                        ProcessBroker.update_progress_bar(calib_bar, finish=1)
                    continue

                n_batch = int(z_pred.shape[0])
                total_n += n_batch
                feat = int(z_pred.shape[1])
                if sum_x is None:
                    sum_x = torch.zeros((feat,), device=accum_device, dtype=torch.float64)
                    sum_y = torch.zeros((feat,), device=accum_device, dtype=torch.float64)
                    sum_x2 = torch.zeros((feat,), device=accum_device, dtype=torch.float64)
                    sum_xy = torch.zeros((feat,), device=accum_device, dtype=torch.float64)

                if int(target_chunk_bytes) <= 0:
                    chunk_f = feat
                else:
                    denom = max(1, n_batch * int(z_pred.element_size()))
                    chunk_f = max(1, int(int(target_chunk_bytes) // denom))
                    chunk_f = min(chunk_f, feat)

                if chunk_f >= feat:
                    sum_x += _to_accum(z_pred.sum(dim=0, dtype=torch.float64))
                    sum_y += _to_accum(z_true.sum(dim=0, dtype=torch.float64))
                    sum_x2 += _to_accum((z_pred * z_pred).sum(dim=0, dtype=torch.float64))
                    sum_xy += _to_accum((z_pred * z_true).sum(dim=0, dtype=torch.float64))
                else:
                    for j in range(0, feat, chunk_f):
                        j2 = j + chunk_f
                        zp = z_pred[:, j:j2]
                        zt = z_true[:, j:j2]
                        sum_x[j:j2] += _to_accum(zp.sum(dim=0, dtype=torch.float64))
                        sum_y[j:j2] += _to_accum(zt.sum(dim=0, dtype=torch.float64))
                        sum_x2[j:j2] += _to_accum((zp * zp).sum(dim=0, dtype=torch.float64))
                        sum_xy[j:j2] += _to_accum((zp * zt).sum(dim=0, dtype=torch.float64))

                seen_batches += 1
                seen_samples += int(B)
                if calib_bar is not None:
                    ProcessBroker.update_progress_bar(calib_bar, finish=1)

        return sum_x, sum_y, sum_x2, sum_xy, total_n, seen_batches, seen_samples

    try:
        run_fn = _run_calibration
        dyn_ctx = None
        if _dynamo_disable is not None:
            wrapped_run_fn = None
            with contextlib.suppress(Exception):
                cand = _dynamo_disable(run_fn)
                if callable(cand):
                    wrapped_run_fn = cand
            if wrapped_run_fn is not None:
                run_fn = wrapped_run_fn
            else:
                with contextlib.suppress(Exception):
                    cand_ctx = _dynamo_disable()
                    if hasattr(cand_ctx, "__enter__") and hasattr(cand_ctx, "__exit__"):
                        dyn_ctx = cand_ctx

        if dyn_ctx is not None:
            try:
                dyn_ctx.__enter__()
            except Exception:
                if local_rank == 0:
                    _LOGGER.debug(
                        "[calibration] disable() context enter failed; continuing unwrapped",
                        exc_info=True,
                    )
                dyn_ctx = None

        if dyn_ctx is not None:
            try:
                (
                    sum_x,
                    sum_y,
                    sum_x2,
                    sum_xy,
                    total_n,
                    seen_batches,
                    seen_samples,
                ) = run_fn(sum_x, sum_y, sum_x2, sum_xy, total_n, seen_batches, seen_samples)
            except BaseException as e:
                suppress_exc = False
                with contextlib.suppress(Exception):
                    suppress_exc = bool(dyn_ctx.__exit__(type(e), e, e.__traceback__))
                if not suppress_exc:
                    raise
            else:
                with contextlib.suppress(Exception):
                    dyn_ctx.__exit__(None, None, None)
        else:
            (
                sum_x,
                sum_y,
                sum_x2,
                sum_xy,
                total_n,
                seen_batches,
                seen_samples,
            ) = run_fn(sum_x, sum_y, sum_x2, sum_xy, total_n, seen_batches, seen_samples)
    finally:
        if calib_bar is not None:
            calib_bar.close()

    if is_distributed():
        n_t = torch.tensor(int(total_n), device="cpu", dtype=torch.int64)
        _dist_all_reduce_sum_(n_t)
        total_n = int(n_t.item())
        if sum_x is not None:
            _dist_all_reduce_sum_(sum_x)
        if sum_y is not None:
            _dist_all_reduce_sum_(sum_y)
        if sum_x2 is not None:
            _dist_all_reduce_sum_(sum_x2)
        if sum_xy is not None:
            _dist_all_reduce_sum_(sum_xy)

    with torch.inference_mode():
        _finalize_affine(total_n, sum_x, sum_y, sum_x2, sum_xy)

    try:
        enable_out_ab = bool(
            env_bool(
                ("ENN_OUTPUT_AB_ENABLE", "ENN_SCALER_OUTPUT_AB_ENABLE"),
                default=True,
            )
        )
    except Exception:
        enable_out_ab = True
    if enable_out_ab and getattr(model_for_scaler, "scaler", None) is not None:
        scaler = model_for_scaler.scaler
        if not env_bool("ENN_DISABLE_OUTPUT_AB", default=False):
            mix_alpha = float(os.environ.get("ENN_OUTPUT_AB_MIX_ALPHA", "") or 0.9)
            if not (0.0 <= mix_alpha <= 1.0):
                mix_alpha = max(0.0, min(1.0, mix_alpha))
            scale_clamp = float(os.environ.get("ENN_OUTPUT_AB_SCALE_CLAMP", "") or 4.0)
            if scale_clamp <= 0.0:
                scale_clamp = 0.0

            with torch.no_grad():
                with contextlib.suppress(Exception):
                    scaler.disable_output_ab()

            sum_pz = sum_pz2 = None
            sum_tz = sum_tz2 = None
            z_min_obs = z_max_obs = None
            y_min_obs = y_max_obs = None
            n_z = 0
            seen_batches2 = 0
            seen_samples2 = 0
            _pred_is_z_env = os.environ.get("ENN_OUTPUT_AB_PRED_IS_Z", None)
            pred_is_z_fixed = (
                bool(env_bool("ENN_OUTPUT_AB_PRED_IS_Z", default=False))
                if _pred_is_z_env is not None
                else None
            )
            need_pred_is_z_sync = bool(is_distributed() and (_pred_is_z_env is None))
            pred_is_z_synced = False

            model.eval()
            with inference_mode(model), StatelessAutocast.float(device):
                for batch in _iter_raw(calib_src):
                    if int(max_batches) > 0 and int(seen_batches2) >= int(max_batches):
                        break
                    if int(max_samples) > 0 and int(seen_samples2) >= int(max_samples):
                        break
                    x_b, _y_b = collate.get_row(batch, labels_required=True)
                    x_raw = torch.atleast_2d(x_b.to(device))
                    b2 = int(x_raw.shape[0])
                    if b2 <= 0:
                        seen_batches2 += 1
                        continue

                    y_pred = model(
                        x_raw,
                        calibrate_output=False,
                        sanitize_nan=True,
                        return_loss=False,
                        return_aux=False,
                    )
                    if isinstance(y_pred, tuple):
                        y_pred = y_pred[0]
                    if not isinstance(y_pred, torch.Tensor):
                        y_pred = torch.as_tensor(y_pred, device=device)

                    ypf = y_pred.reshape(b2, -1)
                    y_true = torch.atleast_2d(_y_b.to(device)).reshape(b2, -1)
                    z_true = scaler.normalize_y(y_true)
                    zt64 = z_true.to(device=accum_device, dtype=torch.float64)

                    local_vote = bool(pred_is_z_fixed) if pred_is_z_fixed is not None else False
                    vote_valid = 0.0
                    local_err_z_sum = 0.0
                    local_err_y_sum = 0.0
                    local_err_n = 0.0

                    if pred_is_z_fixed is None:
                        vote_valid = 1.0
                        with contextlib.suppress(Exception):
                            yp0 = ypf.detach().to(dtype=torch.float32)
                            if yp0.numel() > 0:
                                max_abs = float(yp0.abs().amax().item())
                                std0 = float(yp0.std(unbiased=False).item())
                                local_vote = (max_abs <= 30.0) and (std0 <= 15.0)

                        with contextlib.suppress(Exception):
                            zt0 = z_true.detach().to(dtype=torch.float32)
                            z_as_z = ypf.detach().to(dtype=torch.float32)
                            z_as_y = scaler.normalize_y(ypf).detach().to(dtype=torch.float32)
                            valid = torch.isfinite(zt0) & torch.isfinite(z_as_z) & torch.isfinite(z_as_y)
                            if valid.any():
                                dz0 = z_as_z - zt0
                                dz1 = z_as_y - zt0
                                e0 = (dz0[valid] * dz0[valid]).sum()
                                e1 = (dz1[valid] * dz1[valid]).sum()
                                n0 = int(valid.sum().item())
                                if n0 > 0 and torch.isfinite(e0).all() and torch.isfinite(e1).all():
                                    local_err_z_sum = float(e0.item())
                                    local_err_y_sum = float(e1.item())
                                    local_err_n = float(n0)
                                    local_vote = local_err_z_sum <= local_err_y_sum

                        pred_is_z_fixed = bool(local_vote)

                    if need_pred_is_z_sync and (not pred_is_z_synced):
                        pack = torch.tensor(
                            [
                                float(local_err_z_sum),
                                float(local_err_y_sum),
                                float(local_err_n),
                                float(int(bool(local_vote))),
                                float(vote_valid),
                            ],
                            device="cpu",
                            dtype=torch.float64,
                        )
                        _dist_all_reduce_sum_(pack)
                        g_err_z = float(pack[0].item())
                        g_err_y = float(pack[1].item())
                        g_n = float(pack[2].item())
                        g_vote_sum = float(pack[3].item())
                        g_vote_n = float(pack[4].item())
                        if g_n > 0.0 and math.isfinite(g_err_z) and math.isfinite(g_err_y):
                            pred_is_z_fixed = bool(g_err_z <= g_err_y)
                        elif g_vote_n > 0.0:
                            pred_is_z_fixed = bool((g_vote_sum * 2.0) >= g_vote_n)
                        else:
                            pred_is_z_fixed = False
                        pred_is_z_synced = True

                    pred_is_z = bool(pred_is_z_fixed)
                    z_hat = ypf if pred_is_z else scaler.normalize_y(ypf)
                    z_pred = scaler.calibrate(z_hat)
                    zp64 = z_pred.to(device=accum_device, dtype=torch.float64)

                    yt64 = y_true.to(device=accum_device, dtype=torch.float64)
                    allow_neg_targets = bool(env_bool("ENN_OUTPUT_AB_ALLOW_NEGATIVE_TARGETS", default=False))
                    valid = torch.isfinite(yt64)
                    if not allow_neg_targets:
                        valid = valid & (yt64 >= 0.0)
                    valid = valid & torch.isfinite(zt64)

                    pos_inf = torch.full_like(yt64, float("inf"))
                    neg_inf = torch.full_like(yt64, float("-inf"))
                    yb_min = torch.where(valid, yt64, pos_inf).amin(dim=0)
                    yb_max = torch.where(valid, yt64, neg_inf).amax(dim=0)

                    z_pos_inf = torch.full_like(zt64, float("inf"))
                    z_neg_inf = torch.full_like(zt64, float("-inf"))
                    zb_min = torch.where(valid, zt64, z_pos_inf).amin(dim=0)
                    zb_max = torch.where(valid, zt64, z_neg_inf).amax(dim=0)
                    if z_min_obs is None:
                        z_min_obs = zb_min
                        z_max_obs = zb_max
                        y_min_obs = yb_min
                        y_max_obs = yb_max
                    else:
                        z_min_obs = torch.minimum(z_min_obs, zb_min)
                        z_max_obs = torch.maximum(z_max_obs, zb_max)
                        y_min_obs = torch.minimum(y_min_obs, yb_min)
                        y_max_obs = torch.maximum(y_max_obs, yb_max)

                    if sum_pz is None:
                        feat_y = int(zp64.shape[1])
                        sum_pz = torch.zeros((feat_y,), device=accum_device, dtype=torch.float64)
                        sum_pz2 = torch.zeros((feat_y,), device=accum_device, dtype=torch.float64)
                        sum_tz = torch.zeros((feat_y,), device=accum_device, dtype=torch.float64)
                        sum_tz2 = torch.zeros((feat_y,), device=accum_device, dtype=torch.float64)

                    sum_pz += zp64.sum(dim=0, dtype=torch.float64)
                    sum_pz2 += (zp64 * zp64).sum(dim=0, dtype=torch.float64)
                    sum_tz += zt64.sum(dim=0, dtype=torch.float64)
                    sum_tz2 += (zt64 * zt64).sum(dim=0, dtype=torch.float64)
                    n_z += int(b2)
                    seen_batches2 += 1
                    seen_samples2 += int(b2)

            if need_pred_is_z_sync and (not pred_is_z_synced):
                pack = torch.zeros((5,), device="cpu", dtype=torch.float64)
                _dist_all_reduce_sum_(pack)
                g_err_z = float(pack[0].item())
                g_err_y = float(pack[1].item())
                g_n = float(pack[2].item())
                g_vote_sum = float(pack[3].item())
                g_vote_n = float(pack[4].item())
                if g_n > 0.0 and math.isfinite(g_err_z) and math.isfinite(g_err_y):
                    pred_is_z_fixed = bool(g_err_z <= g_err_y)
                elif g_vote_n > 0.0:
                    pred_is_z_fixed = bool((g_vote_sum * 2.0) >= g_vote_n)
                else:
                    pred_is_z_fixed = False
                pred_is_z_synced = True

            if is_distributed():
                feat_local = int(sum_pz.shape[0]) if isinstance(sum_pz, torch.Tensor) else 0
                big_feat = float(1.0e30)
                feat_t_max = torch.tensor(float(feat_local), device="cpu", dtype=torch.float64)
                feat_t_min = torch.tensor(
                    float(feat_local) if int(feat_local) > 0 else big_feat,
                    device="cpu",
                    dtype=torch.float64,
                )
                _dist_all_reduce_max_(feat_t_max)
                _dist_all_reduce_min_(feat_t_min)
                feat_max = int(round(float(feat_t_max.item())))
                feat_min_f = float(feat_t_min.item())
                feat_min = 0 if feat_min_f >= big_feat * 0.9 else int(round(feat_min_f))
                if feat_min > 0 and feat_max != feat_min:
                    raise RuntimeError(
                        f"[ENN] distributed calibration: inconsistent output feature dim across ranks (min={feat_min}, max={feat_max})"
                    )

                feat_y = int(feat_max)
                if feat_y <= 0:
                    with contextlib.suppress(Exception):
                        feat_y = int(
                            getattr(scaler, "y_mean", torch.zeros(1))
                            .reshape(-1)
                            .numel()
                        )
                if feat_y <= 0:
                    feat_y = 1

                if sum_pz is None or sum_pz2 is None or sum_tz is None or sum_tz2 is None:
                    sum_pz = torch.zeros((feat_y,), device=accum_device, dtype=torch.float64)
                    sum_pz2 = torch.zeros((feat_y,), device=accum_device, dtype=torch.float64)
                    sum_tz = torch.zeros((feat_y,), device=accum_device, dtype=torch.float64)
                    sum_tz2 = torch.zeros((feat_y,), device=accum_device, dtype=torch.float64)

                if z_min_obs is None or z_max_obs is None or y_min_obs is None or y_max_obs is None:
                    z_min_obs = torch.full((feat_y,), float("inf"), device=accum_device, dtype=torch.float64)
                    z_max_obs = torch.full((feat_y,), float("-inf"), device=accum_device, dtype=torch.float64)
                    y_min_obs = torch.full((feat_y,), float("inf"), device=accum_device, dtype=torch.float64)
                    y_max_obs = torch.full((feat_y,), float("-inf"), device=accum_device, dtype=torch.float64)

                n_t2 = torch.tensor(int(n_z), device="cpu", dtype=torch.int64)
                _dist_all_reduce_sum_(n_t2)
                n_z = int(n_t2.item())
                _dist_all_reduce_sum_(sum_pz)
                _dist_all_reduce_sum_(sum_pz2)
                _dist_all_reduce_sum_(sum_tz)
                _dist_all_reduce_sum_(sum_tz2)
                _dist_all_reduce_min_(z_min_obs)
                _dist_all_reduce_max_(z_max_obs)
                _dist_all_reduce_min_(y_min_obs)
                _dist_all_reduce_max_(y_max_obs)

            if int(n_z) > 0 and sum_pz is not None and sum_pz2 is not None and sum_tz is not None and sum_tz2 is not None:
                n2 = float(int(n_z))
                pred_mean = sum_pz / n2
                pred_var = (sum_pz2 / n2 - pred_mean * pred_mean).clamp_min(
                    float(getattr(scaler, "eps", 1e-6))
                )
                pred_std = pred_var.sqrt()

                ref_mean = sum_tz / n2
                ref_var = (sum_tz2 / n2 - ref_mean * ref_mean).clamp_min(
                    float(getattr(scaler, "eps", 1e-6))
                )
                ref_std = ref_var.sqrt()

                clip_low_y = getattr(scaler, "y_min", None)
                clip_high_y = getattr(scaler, "y_max", None)
                clip_low = clip_high = None
                clip_src = "unset"

                try:
                    use_quant = bool(getattr(model_for_scaler, "delta_gate_bounds_use_quantile", False))
                except Exception:
                    use_quant = False
                if use_quant:
                    ql = getattr(scaler, "y_q_low", None)
                    qh = getattr(scaler, "y_q_high", None)
                    if isinstance(ql, torch.Tensor) and isinstance(qh, torch.Tensor):
                        with contextlib.suppress(Exception):
                            if bool(torch.isfinite(ql).all().item()) and bool(torch.isfinite(qh).all().item()):
                                clip_low_y, clip_high_y = ql, qh
                                clip_src = "quantile"

                try:
                    ext_abs = float(os.environ.get("ENN_SCALER_BOUNDS_EXTREME_ABS", "") or 1e307)

                    def _ok_bound(t: object) -> bool:
                        if not isinstance(t, torch.Tensor):
                            return False
                        if is_symbolic() or is_meta_or_fake_tensor(t):
                            return True
                        return bool(torch.isfinite(t).all().item()) and bool((t.abs() < ext_abs).all().item())

                    low_ok = _ok_bound(clip_low_y)
                    high_ok = _ok_bound(clip_high_y)
                    if not bool(env_bool("ENN_OUTPUT_AB_ALLOW_NEGATIVE_TARGETS", default=False)):
                        with contextlib.suppress(Exception):
                            if isinstance(clip_low_y, torch.Tensor) and bool((clip_low_y < 0).any().item()):
                                low_ok = False
                            if isinstance(clip_high_y, torch.Tensor) and bool((clip_high_y < 0).any().item()):
                                high_ok = False
                except Exception:
                    low_ok = False
                    high_ok = False
                if (low_ok and high_ok) and isinstance(clip_low_y, torch.Tensor) and isinstance(clip_high_y, torch.Tensor):
                    with contextlib.suppress(Exception):
                        clip_low = scaler.normalize_y(clip_low_y)
                        clip_high = scaler.normalize_y(clip_high_y)
                        if clip_src == "quantile":
                            clip_src = "quantile_y"
                        else:
                            clip_src = "y_minmax"

                if (clip_low is None or clip_high is None) and isinstance(z_min_obs, torch.Tensor) and isinstance(z_max_obs, torch.Tensor):
                    clip_low = z_min_obs
                    clip_high = z_max_obs
                    clip_src = "z_obs"

                if clip_low is None or clip_high is None:
                    k = float(os.environ.get("ENN_OUTPUT_AB_FALLBACK_K", "") or 8.0)
                    with contextlib.suppress(Exception):
                        k = max(2.0, min(12.0, k))
                    clip_low = ref_mean - k * ref_std
                    clip_high = ref_mean + k * ref_std
                    clip_src = "ref_k"

                with contextlib.suppress(Exception):
                    if isinstance(clip_low, torch.Tensor) and isinstance(clip_high, torch.Tensor):
                        clip_low = clip_low.to(device=accum_device, dtype=torch.float64).reshape(-1)
                        clip_high = clip_high.to(device=accum_device, dtype=torch.float64).reshape(-1)
                        lo2 = torch.minimum(clip_low, clip_high)
                        hi2 = torch.maximum(clip_low, clip_high)
                        clip_low, clip_high = lo2, hi2

                        bad = (~torch.isfinite(clip_low)) | (~torch.isfinite(clip_high))
                        if bad.any():
                            k = float(os.environ.get("ENN_OUTPUT_AB_FALLBACK_K", "") or 8.0)
                            k = max(2.0, min(12.0, k))
                            rm = ref_mean.to(device=accum_device, dtype=torch.float64).reshape(-1)
                            rs = ref_std.to(device=accum_device, dtype=torch.float64).reshape(-1)
                            fb_lo = rm - k * rs
                            fb_hi = rm + k * rs
                            clip_low = torch.where(bad, fb_lo, clip_low)
                            clip_high = torch.where(bad, fb_hi, clip_high)

                        rm = ref_mean.to(device=accum_device, dtype=torch.float64).reshape(-1)
                        ok = (rm >= clip_low) & (rm <= clip_high)
                        frac_ok = float(ok.float().mean().item()) if ok.numel() else 1.0
                        if frac_ok < 0.90:
                            k = float(os.environ.get("ENN_OUTPUT_AB_FALLBACK_K", "") or 8.0)
                            k = max(2.0, min(12.0, k))
                            rs = ref_std.to(device=accum_device, dtype=torch.float64).reshape(-1)
                            clip_low = rm - k * rs
                            clip_high = rm + k * rs
                            clip_src = "sanity_ref_k"

                scaler.fit_output_ab(
                    pred_mean,
                    pred_std,
                    ref_mean,
                    ref_std,
                    clip_low=clip_low,
                    clip_high=clip_high,
                    mix_alpha=mix_alpha,
                    scale_clamp=scale_clamp,
                    enable=True,
                )

                if env_bool(("ENN_PRED_DIAG_OVERWRITE", "ENN_OUTPUT_AB_DIAG"), default=False):
                    with contextlib.suppress(Exception):
                        def _q3(x: torch.Tensor) -> dict[str, float]:
                            xf = x.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
                            if xf.numel() == 0:
                                return {"p0": float("nan"), "p50": float("nan"), "p100": float("nan")}
                            q = torch.quantile(xf, torch.tensor([0.0, 0.5, 1.0]))
                            return {"p0": float(q[0].item()), "p50": float(q[1].item()), "p100": float(q[2].item())}

                        batch0 = None
                        for _b in _iter_raw(calib_src):
                            batch0 = _b
                            break
                        if batch0 is not None:
                            xb0, yb0 = collate.get_row(batch0, labels_required=True)
                            x0 = torch.atleast_2d(xb0.to(device))
                            y0 = torch.atleast_2d(yb0.to(device)).reshape(int(x0.shape[0]), -1)

                            with inference_mode(model), StatelessAutocast.float(device):
                                z0 = model(
                                    x0,
                                    calibrate_output=False,
                                    sanitize_nan=True,
                                    return_loss=False,
                                    return_aux=False,
                                )
                                if isinstance(z0, tuple):
                                    z0 = z0[0]
                                z0 = torch.as_tensor(z0, device=device).reshape(int(x0.shape[0]), -1)
                                z0_cal = scaler.calibrate(z0)
                                y0_hat = scaler.denormalize_y(z0_cal)

                            clip_y_lo = clip_y_hi = None
                            with contextlib.suppress(Exception):
                                if isinstance(clip_low, torch.Tensor):
                                    clip_y_lo = scaler.denormalize_y(clip_low.to(device=device, dtype=torch.float64))
                                if isinstance(clip_high, torch.Tensor):
                                    clip_y_hi = scaler.denormalize_y(clip_high.to(device=device, dtype=torch.float64))

                            payload = {
                                "tag": "output_ab_diag",
                                "n_z": int(n_z),
                                "pred_is_z": bool(pred_is_z),
                                "clip_src": str(clip_src),
                                "y_true": _q3(y0),
                                "y_pred": _q3(y0_hat),
                                "z_pred": _q3(z0),
                                "z_pred_cal": _q3(z0_cal),
                                "pred_mean": _q3(pred_mean),
                                "pred_std": _q3(pred_std),
                                "ref_mean": _q3(ref_mean),
                                "ref_std": _q3(ref_std),
                                "clip_low_z": _q3(clip_low) if isinstance(clip_low, torch.Tensor) else None,
                                "clip_high_z": _q3(clip_high) if isinstance(clip_high, torch.Tensor) else None,
                                "clip_low_y": _q3(clip_y_lo) if isinstance(clip_y_lo, torch.Tensor) else None,
                                "clip_high_y": _q3(clip_y_hi) if isinstance(clip_y_hi, torch.Tensor) else None,
                                "y_min_obs": _q3(y_min_obs) if isinstance(y_min_obs, torch.Tensor) else None,
                                "y_max_obs": _q3(y_max_obs) if isinstance(y_max_obs, torch.Tensor) else None,
                                "z_min_obs": _q3(z_min_obs) if isinstance(z_min_obs, torch.Tensor) else None,
                                "z_max_obs": _q3(z_max_obs) if isinstance(z_max_obs, torch.Tensor) else None,
                                "y_out_scale": _q3(getattr(scaler, "y_out_scale")),
                                "y_out_bias": _q3(getattr(scaler, "y_out_bias")),
                            }
                            dump_dir = os.environ.get("ENN_PRED_OVERWRITE_DUMP_DIR", "") or os.environ.get("ENN_NONFINITE_DUMP_DIR", "")
                            if dump_dir:
                                os.makedirs(dump_dir, exist_ok=True)
                                rid = str(os.environ.get("ENN_RUN_ID", "") or "run").strip() or "run"
                                path = os.path.join(
                                    dump_dir,
                                    f"output_ab_diag.rank{int(get_rank() or 0)}.{rid}.json",
                                )
                                collate.write_json(path, payload, indent=2)
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
            else prev_comp_time / total_t if total_t > 0.0 else 0.0
        )
        gpu_util_frac = None
        mem_util_frac = None
        if gpu_util_ema is not None:
            gpu_util_frac = max(0.0, min(1.0, gpu_util_ema / 100.0))
        if mem_util_ema is not None:
            mem_util_frac = max(0.0, min(1.0, mem_util_ema / 100.0))
        if dev_t != "cpu":
            util_for_cap = (
                gpu_util_frac if gpu_util_frac is not None else util_fallback
            )
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
                                        % (
                                            float(mem_util_frac),
                                            float(prev),
                                            float(cur),
                                        ),
                                        level="debug",
                                    )
                        elif util_for_cap < 0.9 and (
                            mem_util_frac is None or mem_util_frac < 0.88
                        ):
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
                                        % (
                                            float(util_for_cap),
                                            float(prev),
                                            float(cur),
                                        ),
                                        level="debug",
                                    )
            except Exception:
                pass
        else:
            cpu_pct = Monitor.cpu_load()
            if cpu_pct is not None:
                if cpu_pct > 80.0:
                    time.sleep(min(0.005, 0.001 * (cpu_pct - 80.0)))
            elif util_fallback > 0.8:
                time.sleep(min(0.005, total_t * (util_fallback - 0.8)))
        if isinstance(hist, Recorder):
            try:
                end_sec = round(float(end_kst_ns) / 1000000000.0, 6)
                world = (
                    max(1, get_world_size(device)) if is_distributed() else 1
                )
                hist.end_session(end_sec, peers=world)

                try:
                    if ops.ckpt_dir and (
                        not is_distributed()
                        or int(torch.distributed.get_rank()) == 0
                    ):
                        hist_path = os.path.join(
                            str(ops.ckpt_dir), "history.json"
                        )
                        collate.write_json(hist_path, hist.save(), indent=2)
                        ret_dir = os.environ.get("ENN_RETURN_DIR") or ""
                        if ret_dir and str(ret_dir) != str(ops.ckpt_dir):
                            try:
                                os.makedirs(ret_dir, exist_ok=True)
                                collate.write_json(
                                    os.path.join(ret_dir, "history.json"),
                                    hist.save(),
                                    indent=2,
                                )
                            except Exception:
                                pass
                except Exception:
                    pass
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
        dataset = Dataset.for_device(
            str(device) if isinstance(device, torch.device) else "cpu"
        )
    if chunk_dir is None:
        if not ops.ckpt_dir:
            raise RuntimeError(
                "infer: ckpt_dir is required when chunk_dir is not provided"
            )
        chunk_dir = os.path.join(ops.ckpt_dir, "pred_chunks")
    rank = torch.distributed.get_rank() if is_distributed() else 0
    world_size = get_world_size(device) if is_distributed() else 1
    torch_prof = None
    prof_enabled = env_bool(
        "ENN_TORCH_PROFILE_INFER", env_bool("ENN_TORCH_PROFILE", False)
    )
    prof_all_ranks = env_bool("ENN_TORCH_PROFILE_ALL_RANKS", False)
    if prof_enabled and (prof_all_ranks or int(rank) == 0):
        prof_dir = env_str("ENN_TORCH_PROFILE_DIR")
        if not prof_dir:
            prof_dir = os.path.join(str(ops.ckpt_dir or "."), "torch_profiler")
        torch_prof = _get_torch_profiler(
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
    with contextlib.suppress(Exception):
        os.makedirs(chunk_dir, exist_ok=True)
    _nogil_opt = bool(CPU.is_optimized_for_no_gil())
    _cache_default = 16 if _nogil_opt else 4
    cache_q = max(
        1,
        int(
            env_first_int(
                (
                    "ENN_PRED_CACHE_MAX_QUEUE",
                    "ENN_PRED_WRITE_QUEUE",
                    "ENN_CACHE_MAX_QUEUE",
                ),
                default=_cache_default,
            )
        ),
    )
    dev_type = str(getattr(device, "type", "cpu"))
    use_async_write = bool(env_bool("ENN_PRED_ASYNC_WRITE", True))
    use_mmt_pred_parts = bool(
        env_bool("ENN_PRED_MMT_PARTS", dev_type != "cpu")
    )
    if not use_async_write:
        use_mmt_pred_parts = False
    cache = (
        TensorSpooler(chunk_dir, max_queue=cache_q)
        if use_async_write
        else None
    )
    target_rows = int(env_int("ENN_PRED_CHUNK_ROWS", 0))
    if target_rows <= 0:
        out_shape = tuple((int(x) for x in ops.out_shape or ()))
        out_numel = 1
        for d in out_shape:
            out_numel *= max(1, int(d))
        est_row_bytes = max(1, out_numel * 4)
        target_bytes = int(env_int("ENN_PRED_CHUNK_BYTES", 64 * 1024 * 1024))
        target_rows = max(256, min(65536, target_bytes // est_row_bytes))
    dev_obj = (
        device if isinstance(device, torch.device) else torch.device(device)
    )
    run_model: torch.nn.Module = model
    if (
        is_distributed()
        and get_world_size(dev_obj) > 1
        and str(getattr(dev_obj, "type", "cpu")) in ("cuda", "xpu")
    ):
        mesh, mesh_kind = get_distributed_mesh(dev_obj)
        if mesh_kind != "hsdp2":
            _LOGGER.warning(
                f"HSDP2 mesh not available (e.g. heterogeneous nodes). Using fallback: {mesh_kind}"
            )
        run_model = to_hsdp_module(
            model,
            mesh=mesh,
            mp_policy=None,
            reshard_after_forward=True,
            sync_module_states=True,
        )
    run_model.eval()
    module_eval = (
        run_model.module if hasattr(run_model, "module") else run_model
    )
    distributed_sync(module_eval, device=dev_obj)
    def _unwrap_uncompiled_model_handle(m: torch.nn.Module) -> torch.nn.Module:
        cur = m
        seen: set[int] = set()
        for _ in range(16):
            cur_id = id(cur)
            if cur_id in seen:
                break
            seen.add(cur_id)
            next_m = None
            for attr in ("_orig_mod", "_original_module", "_uncompiled_module"):
                cand = getattr(cur, attr, None)
                if isinstance(cand, torch.nn.Module) and cand is not cur:
                    next_m = cand
                    break
            if next_m is None:
                child = getattr(cur, "module", None)
                if isinstance(child, torch.nn.Module) and child is not cur:
                    next_m = child
            if next_m is None:
                break
            cur = next_m
        return cur

    run_model_uncompiled_base = _unwrap_uncompiled_model_handle(run_model)
    run_model_uncompiled = (
        to_submodule(run_model_uncompiled_base) or run_model_uncompiled_base
    )
    if run_model_uncompiled is not run_model:
        with contextlib.suppress(Exception):
            run_model_uncompiled.eval()

    eager_ctx_factory = getattr(module_eval, "eager_for_export", None)
    force_eager = bool(env_bool("ENN_PRED_FORCE_EAGER", False))
    eager_on_broadcast = bool(env_bool("ENN_PRED_EAGER_ON_BROADCAST", True))
    force_uncompiled = bool(env_bool("ENN_PRED_FORCE_UNCOMPILED", False))
    collapse_force_uncompiled = bool(
        env_bool("ENN_PRED_COLLAPSE_FORCE_UNCOMPILED", default=True)
    )

    cg_enabled = bool(
        dev_type == "cuda"
        and getattr(module_eval, "_compile_cudagraphs", False)
    )
    _pred_compile_mode = ""
    with contextlib.suppress(Exception):
        _pred_compile_mode = str(
            getattr(module_eval, "_enn_compile_active_mode", "")
            or getattr(module_eval, "_enn_compile_requested_mode", "")
            or ""
        )
    if not _pred_compile_mode:
        with contextlib.suppress(Exception):
            _pred_compile_mode = str(
                getattr(model, "_enn_compile_active_mode", "")
                or getattr(model, "_enn_compile_requested_mode", "")
                or ""
            )
    _pred_compile_mode = str(_pred_compile_mode).strip().lower()
    _td_cg_default = True
    if "no-cudagraph" in _pred_compile_mode:
        _td_cg_default = False
    td_cg_candidate = bool(
        (not cg_enabled)
        and dev_type == "cuda"
        and bool(env_bool("ENN_PRED_TD_CUDAGRAPH", default=_td_cg_default))
        and (TD_CudaGraphModule is not None)
        and bool(getattr(torch.cuda, "is_available", lambda: False)())
        and hasattr(torch.cuda, "CUDAGraph")
        and (not bool(force_eager))
    )
    non_blocking_ok = bool(dev_type in ("cuda", "xpu"))
    pinned_ok = bool(is_pin_supported(dev_type))
    backend = accelerator_type(dev_type)
    stream_fn = (
        getattr(backend, "current_stream", None)
        if backend is not None
        else None
    )
    Event = getattr(backend, "Event", None) if backend is not None else None
    can_stream_release = bool(
        pinned_ok
        and non_blocking_ok
        and callable(stream_fn)
        and (Event is not None)
    )
    make_fence_event = partial(
        new_accelerator_event, device, enable_timing=False
    )
    cpu_pool = None
    if pinned_ok and TensorPagePool is not None:
        with contextlib.suppress(Exception):
            Memory.prefer_local_numa()
        with contextlib.suppress(Exception):
            _nogil = bool(CPU.is_optimized_for_no_gil())
            _cpu_default = 8
            if _nogil:
                try:
                    _cpu_default = max(8, min(64, int(os.cpu_count() or 8)))
                except Exception:
                    _cpu_default = 16
            cpu_pool_cap = max(
                2,
                int(env_int("ENN_RUNTIME_PIN_POOL_CAPACITY", _cpu_default)),
            )
            cpu_pool = TensorPagePool(capacity=cpu_pool_cap)
    pred_pool = None
    if (
        non_blocking_ok
        and TensorPagePool is not None
        and env_bool("ENN_PRED_PINNED", True)
    ):
        with contextlib.suppress(Exception):
            _nogil = bool(CPU.is_optimized_for_no_gil())
            _pred_default = 2 if not _nogil else 4
            pred_pool_cap = max(
                2, int(env_int("ENN_PRED_PIN_POOL_CAPACITY", _pred_default))
            )
            pred_pool = TensorPagePool(capacity=pred_pool_cap, pin_memory=True)
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
        ProcessBroker.get_progress_bar(
            title="Prediction",
            total=_get_batch_length(data_loader),
            device=device,
            leave=False,
        )
        if local_rank == 0
        else None
    )
    row_cursor = 0
    writer = Unsharder(
        chunk_dir=str(chunk_dir),
        rank=int(rank),
        use_mmt_pred_parts=bool(use_mmt_pred_parts),
        cache=cache,
        pred_pool=pred_pool,
        target_rows=int(target_rows),
        make_fence_event=make_fence_event,
    )
    try:
        with inference_mode(run_model), StatelessAutocast.float(device):
            td_cg_active = False
            td_cg_disabled = not bool(td_cg_candidate)
            td_cg_mb = None
            td_cg_mod = None
            td_cg_pad_buf = None
            td_cg_seen = 0
            td_cg_max_bs = 0
            td_cg_target = None
            td_cg_x_inner_shape = None
            force_single = False
            broadcast_checked = False
            detect_broadcast = bool(env_bool("ENN_PRED_DETECT_BATCH_BROADCAST", True))
            broadcast_atol = float(env_float("ENN_PRED_BROADCAST_ATOL", 1e-6))
            broadcast_match_frac = float(env_float("ENN_PRED_BROADCAST_MATCH_FRAC", 0.995))
            broadcast_rel_mean = float(env_float("ENN_PRED_BROADCAST_REL_MEAN", 1e-5))
            broadcast_sample_max = max(0, int(env_int("ENN_PRED_BROADCAST_SAMPLE_MAX", 16384) or 16384))

            def _broadcast_like(
                y0: torch.Tensor,
                y1: torch.Tensor,
                *,
                atol: float,
            ) -> tuple[bool, dict[str, float]]:
                st: dict[str, float] = {}
                try:
                    diff = (y0 - y1).abs()
                    numel = int(diff.numel())
                    st["y_numel"] = float(numel)
                    if numel <= 0:
                        st.update({"y_max": 0.0, "y_mean": 0.0, "y_match_frac": 1.0, "y_scale": 1.0, "y_rel_mean": 0.0, "sample_n": 0.0, "sampled": 0.0})
                        return True, st

                    if int(broadcast_sample_max) > 0 and numel > int(broadcast_sample_max):
                        flat = diff.reshape(-1)
                        sample_n = int(broadcast_sample_max)
                        if sample_n <= 1:
                            if numel <= 1:
                                sample = flat[:1].clone()
                            else:
                                idx = torch.tensor([0, numel - 1], device=flat.device, dtype=torch.long)
                                sample = flat.index_select(0, idx)
                        else:
                            idx = torch.arange(sample_n, device=flat.device, dtype=torch.long)
                            idx = (idx * (numel - 1)) // (sample_n - 1)
                            sample = flat.index_select(0, idx)
                        st["sampled"] = 1.0
                        st["sample_n"] = float(int(sample.numel()))
                        mean_abs = float(sample.mean().item())
                        match_frac = float((sample <= float(atol)).to(dtype=torch.float32).mean().item())
                    else:
                        st["sampled"] = 0.0
                        st["sample_n"] = float(numel)
                        mean_abs = float(diff.mean().item())
                        match_frac = float((diff <= float(atol)).to(dtype=torch.float32).mean().item())

                    max_abs = float(diff.max().item())
                    st["y_max"] = float(max_abs)
                    st["y_mean"] = float(mean_abs)
                    st["y_match_frac"] = float(match_frac)

                    y_scale = float(torch.maximum(y0.abs().mean(), y1.abs().mean()).item())
                    if (not math.isfinite(y_scale)) or y_scale <= 0.0:
                        y_scale = 1.0
                    st["y_scale"] = float(y_scale)
                    rel_mean = float(mean_abs) / float(max(float(y_scale), 1e-12))
                    st["y_rel_mean"] = float(rel_mean)

                    if float(max_abs) <= float(atol):
                        return True, st
                    if float(match_frac) >= float(broadcast_match_frac) and float(rel_mean) <= float(broadcast_rel_mean):
                        return True, st
                    return False, st
                except Exception:
                    st["error"] = 1.0
                    return False, st
            collapse_stage_diag_enabled = bool(
                env_bool("ENN_PRED_COLLAPSE_STAGE_DIAG", default=True)
            )
            collapse_stage_diag_once = bool(
                env_bool("ENN_PRED_COLLAPSE_STAGE_DIAG_ONCE", default=True)
            )
            _collapse_stage_diag_done = False
            diag_overwrite = bool(env_bool("ENN_PRED_DIAG_OVERWRITE", default=False))
            diag_overwrite_abort = bool(
                env_bool(
                    "ENN_PRED_DIAG_OVERWRITE_ABORT",
                    default=env_bool("ENN_SANITIZE_NAN_STRICT", default=False),
                )
            )
            diag_overwrite_n = max(1, int(env_int("ENN_PRED_DIAG_OVERWRITE_SAMPLE", 256)))
            prev_pred_ref: torch.Tensor | None = None
            prev_pred_sample_cpu: torch.Tensor | None = None
            prev_pred_ptr: int | None = None
            prev_pred_step: int | None = None
            prev_pred_slice: tuple[int, int] | None = None
            prev_pred_rows_cpu: torch.Tensor | None = None
            prev_pred_X_cpu: torch.Tensor | None = None
            prev_pred_X_run_cpu: torch.Tensor | None = None
            prev_pred_X_run_ptr: int | None = None
            prev_pred_X_run_source: str | None = None
            prev_pred_pad_n: int | None = None
            prev_pred_mb: int | None = None
            prev_pred_n_i: int | None = None

            def _ids_diag(ids: object, *, head: int = 8) -> object:
                if ids is None:
                    return None
                try:
                    if torch.is_tensor(ids):
                        x = ids.detach()
                        try:
                            x = x.reshape(-1)
                        except Exception:
                            x = x.flatten()
                        if getattr(getattr(x, 'device', None), 'type', None) != 'cpu':
                            x = x.to(device='cpu')
                        n = int(min(int(head), int(x.numel())))
                        vals = x[:n].tolist()
                        out_vals: list[object] = []
                        for v in vals:
                            try:
                                out_vals.append(int(v))
                            except Exception:
                                out_vals.append(v)
                        return {'type': 'tensor', 'n': int(x.numel()), 'head': out_vals}
                    if isinstance(ids, (list, tuple)):
                        vals = list(ids)[: int(head)]
                        out_vals: list[object] = []
                        for v in vals:
                            try:
                                out_vals.append(int(v))
                            except Exception:
                                out_vals.append(v)
                        return {'type': type(ids).__name__, 'n': int(len(ids)), 'head': out_vals}
                    if isinstance(ids, dict):
                        keys = list(ids.keys())
                        return {'type': 'dict', 'keys_head': keys[: int(head)]}
                except Exception:
                    pass
                r = repr(ids)
                if len(r) > 200:
                    r = r[:200] + '...'
                return {'type': type(ids).__name__, 'repr': r}

            def _x_diag(X: object, *, head_rows: int = 4, head_cols: int = 8) -> object:
                if not torch.is_tensor(X):
                    return {'type': type(X).__name__}
                x = X.detach()
                info: dict[str, object] = {
                    'shape': [int(v) for v in tuple(x.shape)],
                    'dtype': str(x.dtype),
                    'device': str(x.device),
                }
                try:
                    if x.ndim >= 2:
                        rr = int(min(int(head_rows), int(x.shape[0])))
                        if rr <= 0:
                            return info
                        xr = x[:rr]
                        if getattr(getattr(xr, 'device', None), 'type', None) != 'cpu':
                            xr = xr.to('cpu')
                        if isinstance(xr, torch.Tensor) and xr.dtype != torch.float32:
                            xr = xr.to(dtype=torch.float32)
                        cc = int(min(int(head_cols), int(xr.shape[1]))) if xr.ndim >= 2 else 0
                        rows: list[dict[str, object]] = []
                        for i in range(rr):
                            row_full = xr[i].reshape(-1)
                            s = float(row_full.sum().item()) if int(row_full.numel()) > 0 else 0.0
                            mean = float(row_full.mean().item()) if int(row_full.numel()) > 0 else 0.0
                            std = float(row_full.std(unbiased=False).item()) if int(row_full.numel()) > 1 else 0.0
                            vals = [float(v) for v in row_full[:cc].tolist()] if cc > 0 else []
                            rows.append({'row': int(i), 'vals': vals, 'stats': {'sum': s, 'mean': mean, 'std': std, 'numel': int(row_full.numel())}})
                        info['head'] = rows
                    else:
                        flat = x.reshape(-1)
                        if getattr(getattr(flat, 'device', None), 'type', None) != 'cpu':
                            flat = flat.to('cpu')
                        if isinstance(flat, torch.Tensor) and flat.dtype != torch.float32:
                            flat = flat.to(dtype=torch.float32)
                        n = int(min(32, int(flat.numel())))
                        info['head'] = [float(v) for v in flat[:n].tolist()]
                        s = float(flat.sum().item()) if int(flat.numel()) > 0 else 0.0
                        mean = float(flat.mean().item()) if int(flat.numel()) > 0 else 0.0
                        std = float(flat.std(unbiased=False).item()) if int(flat.numel()) > 1 else 0.0
                        info['stats'] = {'sum': s, 'mean': mean, 'std': std, 'numel': int(flat.numel())}
                except Exception:
                    pass
                return info

            def _row_stats(xrow: object) -> object:
                if not torch.is_tensor(xrow):
                    return None
                try:
                    v = xrow.detach().reshape(-1)
                    if getattr(getattr(v, 'device', None), 'type', None) != 'cpu':
                        v = v.to('cpu')
                    if isinstance(v, torch.Tensor) and v.dtype != torch.float32:
                        v = v.to(dtype=torch.float32)
                    n = int(v.numel())
                    if n <= 0:
                        return {'sum': 0.0, 'mean': 0.0, 'std': 0.0, 'numel': 0}
                    s = float(v.sum().item())
                    mean = float(v.mean().item())
                    std = float(v.std(unbiased=False).item()) if n > 1 else 0.0
                    return {'sum': s, 'mean': mean, 'std': std, 'numel': n}
                except Exception:
                    return None

            def _unravel_index(flat_idx: int, shape: list[int]) -> list[int]:
                idx = int(flat_idx)
                out: list[int] = []
                for dim in reversed(shape):
                    d = int(dim)
                    if d <= 0:
                        out.append(0)
                        continue
                    out.append(int(idx % d))
                    idx //= d
                return list(reversed(out))

            def _overwrite_dump_dir() -> str:
                d = env_str('ENN_PRED_OVERWRITE_DUMP_DIR')
                if d:
                    return str(d)
                d = env_str('ENN_NONFINITE_DUMP_DIR')
                if d:
                    return os.path.join(str(d), 'pred_overwrite')
                return os.path.join(str(chunk_dir), 'pred_overwrite')

            def _dump_overwrite_diag(payload: dict[str, object]) -> str | None:
                try:
                    ddir = _overwrite_dump_dir()
                    os.makedirs(ddir, exist_ok=True)
                    fn = payload.get('filename')
                    if not isinstance(fn, str) or not fn:
                        ts = int(time.time() * 1000)
                        fn = f'overwrite_rank{int(rank)}_{ts}.json'
                    path = os.path.join(ddir, fn)
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
                    return path
                except Exception:
                    with contextlib.suppress(Exception):
                        _LOGGER.exception('[infer][overwrite-diag] failed to write overwrite dump')
                    return None

            def _tensor_diag(t: object, *, sample_n: int = 64) -> dict[str, object]:
                if not torch.is_tensor(t):
                    return {"type": type(t).__name__}
                x = t.detach()
                out: dict[str, object] = {
                    "shape": [int(v) for v in tuple(x.shape)],
                    "dtype": str(x.dtype),
                    "device": str(x.device),
                    "numel": int(x.numel()),
                }
                if x.numel() <= 0:
                    out["finite"] = True
                    out["abs_max"] = 0.0
                    out["min"] = 0.0
                    out["max"] = 0.0
                    out["std_mean"] = 0.0
                    out["sample"] = []
                    return out
                try:
                    if x.is_floating_point() or x.is_complex():
                        out["finite"] = bool(torch.isfinite(x).all().item())
                    else:
                        out["finite"] = True
                except Exception:
                    out["finite"] = False
                try:
                    xf = x
                    if not (xf.is_floating_point() or xf.is_complex()):
                        xf = xf.to(dtype=torch.float32)
                    out["abs_max"] = float(xf.abs().max().item())
                    out["std_mean"] = float(xf.reshape(xf.shape[0], -1).std(dim=1, unbiased=False).mean().item()) if xf.dim() >= 2 else float(xf.std(unbiased=False).item())
                    out["min"] = float(xf.min().item())
                    out["max"] = float(xf.max().item())
                except Exception:
                    out["abs_max"] = None
                    out["std_mean"] = None
                    out["min"] = None
                    out["max"] = None
                try:
                    flat = x.reshape(-1)
                    n = int(min(int(sample_n), int(flat.numel())))
                    samp = flat[:n]
                    if samp.is_floating_point() or samp.is_complex():
                        samp = samp.to(dtype=torch.float32)
                    out["sample"] = [float(v) for v in samp.cpu().tolist()]
                except Exception:
                    out["sample"] = []
                return out

            def _pair_diff_max(t: object) -> float | None:
                if not torch.is_tensor(t):
                    return None
                x = t.detach()
                if x.dim() < 1 or int(x.shape[0]) < 2:
                    return None
                try:
                    a = x[0]
                    b = x[1]
                    if not (a.is_floating_point() or a.is_complex()):
                        a = a.to(dtype=torch.float32)
                        b = b.to(dtype=torch.float32)
                    else:
                        a = a.to(dtype=torch.float32)
                        b = b.to(dtype=torch.float32)
                    return float((a - b).abs().max().item())
                except Exception:
                    return None

            def _brief_tensor(t: object, *, max_elems: int = 64) -> dict[str, object] | None:
                try:
                    if not isinstance(t, torch.Tensor):
                        return None
                    d: dict[str, object] = {}
                    d["shape"] = [int(x) for x in t.shape]
                    d["dtype"] = str(t.dtype)
                    d["device"] = str(t.device)
                    d["numel"] = int(t.numel())
                    if t.numel() > 0:
                        td = t.detach()
                        if td.is_floating_point():
                            with contextlib.suppress(Exception):
                                d["finite"] = bool(torch.isfinite(td).all().item())
                            with contextlib.suppress(Exception):
                                d["abs_max"] = float(td.abs().max().item())
                            with contextlib.suppress(Exception):
                                d["min"] = float(td.min().item())
                            with contextlib.suppress(Exception):
                                d["max"] = float(td.max().item())
                        flat = td.reshape(-1)
                        n = int(min(int(max_elems), int(flat.numel())))
                        with contextlib.suppress(Exception):
                            d["sample"] = [float(x) for x in flat[:n].to("cpu")]
                    return d
                except Exception:
                    return None

            def _unwrap_to_model_core(m: object) -> object:
                cur = m
                seen: set[int] = set()
                for _ in range(16):
                    if cur is None:
                        break
                    cid = id(cur)
                    if cid in seen:
                        break
                    seen.add(cid)
                    nxt = getattr(cur, "module", None)
                    if isinstance(nxt, torch.nn.Module) and nxt is not cur:
                        cur = nxt
                        continue
                    break
                return cur

            def _dump_collapse_stage_diag_from_model(
                Xi2: torch.Tensor,
                *,
                where: str,
                x_diff: float,
                y_diff_cal: float,
            ) -> None:
                nonlocal _collapse_stage_diag_done
                if not collapse_stage_diag_enabled:
                    return
                if collapse_stage_diag_once and _collapse_stage_diag_done:
                    return
                if not isinstance(Xi2, torch.Tensor) or Xi2.ndim < 2 or int(Xi2.shape[0]) < 2:
                    return
                try:
                    m0 = _unwrap_to_model_core(run_model_uncompiled if "run_model_uncompiled" in locals() else run_model)
                    m0 = _unwrap_to_model_core(m0)
                    fn = getattr(m0, "_run_forward_core", None)
                    if not callable(fn):
                        return
                    base_dtype = None
                    with contextlib.suppress(Exception):
                        p0 = next(iter(m0.parameters()))
                        if isinstance(p0, torch.Tensor):
                            base_dtype = p0.dtype
                    kwargs = dict(
                        export=False,
                        temporal_state=None,
                        causal_mask=None,
                        sanitize_nan=True,
                        calibrate_output=bool(calibrate_pred_output),
                        device=None,
                        base_dtype=base_dtype,
                    )
                    with contextlib.suppress(Exception):
                        sig = inspect.signature(fn)
                        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
                    _prev_diag_env = os.environ.get("ENN_PRED_COLLAPSE_STAGE_DIAG", None)
                    os.environ["ENN_PRED_COLLAPSE_STAGE_DIAG"] = "1"
                    try:
                        with torch.no_grad():
                            out = fn(Xi2, **kwargs)
                    finally:
                        if _prev_diag_env is None:
                            with contextlib.suppress(Exception):
                                os.environ.pop("ENN_PRED_COLLAPSE_STAGE_DIAG", None)
                        else:
                            os.environ["ENN_PRED_COLLAPSE_STAGE_DIAG"] = _prev_diag_env
                    nonfinite_pre_sanitize = getattr(m0, "_enn_nonfinite_pre_sanitize", None)
                    fuser_diag = None
                    perceiver_diag = None
                    with contextlib.suppress(Exception):
                        f0 = getattr(m0, "fuser", None)
                        fuser_diag = getattr(f0, "_enn_last_fuser_diag", None)
                        p0 = getattr(f0, "perceiver", None)
                        perceiver_diag = getattr(p0, "_enn_last_perceiver_diag", None)
                    if not (isinstance(out, tuple) and len(out) >= 8):
                        return
                    pred, _next_state, p, assembled, enhanced, delta, tokens, refined = out[:8]
                    y_hat = assembled + (delta * 0.5 if (p is None) else (p.to(dtype=assembled.dtype) * delta))
                    pred_denorm_uncal = None
                    with contextlib.suppress(Exception):
                        scaler = getattr(m0, "scaler", None)
                        if scaler is not None and hasattr(scaler, "denormalize_y"):
                            pred_denorm_uncal = scaler.denormalize_y(y_hat).reshape(int(y_hat.shape[0]), *tuple(getattr(m0, "out_shape", ())))
                    tokenize_pre: dict[str, object] = {}
                    tokenize_post_blk0: dict[str, object] = {}
                    tokenize_meta: dict[str, object] = {}
                    template_out: dict[str, object] = {}
                    template_weight_raw: dict[str, object] = {}
                    template_token_count: dict[str, object] = {}
                    template_logit_bias: dict[str, object] = {}
                    template_prior_mass: dict[str, object] = {}
                    fuser_attn_bias_segments: dict[str, object] = {}
                    try:
                        fuser = getattr(m0, "fuser", None)
                        tasks = getattr(fuser, "tasks", None) if fuser is not None else None
                        if isinstance(tasks, torch.nn.ModuleDict):
                            B2 = int(Xi2.shape[0])
                            import math as _math
                            w_eff_map: dict[str, float] = {}
                            for name, tmpl in tasks.items():
                                tok = getattr(tmpl, "tokenizer", None)
                                t_tokens = getattr(tmpl, "tokens", None)
                                t_dmodel = getattr(tmpl, "d_model", None)
                                if callable(tok) and isinstance(t_tokens, int) and isinstance(t_dmodel, int):
                                    with torch.no_grad():
                                        t = tok(Xi2.contiguous())
                                    t = t.reshape(B2, int(t_tokens), int(t_dmodel)).contiguous()
                                    tokenize_pre[str(name)] = _brief_tensor(t)
                                    tokenize_meta[str(name)] = {
                                        "mode": str(getattr(tmpl, "mode", "")),
                                        "in_dim": int(getattr(tmpl, "in_dim", -1) or -1),
                                        "tokens": int(getattr(tmpl, "tokens", -1) or -1),
                                        "d_model": int(getattr(tmpl, "d_model", -1) or -1),
                                        "nhead": int(getattr(tmpl, "nhead", -1) or -1),
                                        "depth": int(getattr(tmpl, "depth", -1) or -1),
                                    }
                                    with contextlib.suppress(Exception):
                                        blocks = getattr(tmpl, "blocks", None)
                                        if isinstance(blocks, torch.nn.ModuleList) and len(blocks) > 0:
                                            blk0 = blocks[0]
                                            mode0 = str(getattr(tmpl, "mode", "spatial"))
                                            kw0: dict[str, object] = {}
                                            with contextlib.suppress(Exception):
                                                ps = inspect.signature(blk0).parameters
                                                if "causal_mask" in ps:
                                                    kw0["causal_mask"] = None
                                                if "state" in ps:
                                                    kw0["state"] = None
                                                if "mode" in ps:
                                                    kw0["mode"] = mode0
                                            with torch.no_grad():
                                                y0 = blk0(t, **kw0)
                                            if isinstance(y0, tuple):
                                                y0 = y0[0]
                                            tokenize_post_blk0[str(name)] = _brief_tensor(y0)
                                    with contextlib.suppress(Exception):
                                        kwt: dict[str, object] = {"causal_mask": None}
                                        with contextlib.suppress(Exception):
                                            ps_t = inspect.signature(tmpl.forward).parameters
                                            if "state" in ps_t:
                                                kwt["state"] = None
                                            if "return_state" in ps_t:
                                                kwt["return_state"] = False
                                        with torch.no_grad():
                                            tout = tmpl(Xi2, **kwt)
                                        if isinstance(tout, tuple):
                                            tout = tout[0]
                                        if torch.is_tensor(tout):
                                            template_out[str(name)] = _brief_tensor(tout)
                                            n_tok = int(tout.shape[1]) if tout.dim() >= 2 else int(getattr(tmpl, "tokens", 1) or 1)
                                            template_token_count[str(name)] = int(n_tok)
                                            w = getattr(tmpl, "weight", None)
                                            eps = getattr(tmpl, "eps", None)
                                            wv = 1.0
                                            ev = 1e-6
                                            with contextlib.suppress(Exception):
                                                if torch.is_tensor(w):
                                                    wv = float(w.detach().float().item())
                                            with contextlib.suppress(Exception):
                                                if torch.is_tensor(eps):
                                                    ev = float(eps.detach().float().item())
                                            template_weight_raw[str(name)] = float(wv)
                                            w_eff = max(float(wv), float(ev))
                                            w_eff_map[str(name)] = w_eff
                                            template_logit_bias[str(name)] = float(_math.log(w_eff / float(max(1, n_tok))))
                            s = float(sum(w_eff_map.values())) if w_eff_map else 0.0
                            if s > 0.0:
                                for k, w_eff in w_eff_map.items():
                                    template_prior_mass[k] = float(w_eff / s)
                            with contextlib.suppress(Exception):
                                if fuser is not None and callable(getattr(fuser, "_build_attn_bias", None)):
                                    token_sets = []
                                    names = []
                                    for name, _ in tasks.items():
                                        if name in template_out:
                                            kwt = {"causal_mask": None}
                                            with torch.no_grad():
                                                tout = tasks[name](Xi2, **kwt)
                                            if isinstance(tout, tuple):
                                                tout = tout[0]
                                            if torch.is_tensor(tout):
                                                token_sets.append(tout)
                                                names.append(str(name))
                                    if token_sets:
                                        am = fuser._build_attn_bias(names, token_sets, device=token_sets[0].device, dtype=token_sets[0].dtype)
                                        if torch.is_tensor(am) and am.numel() > 0:
                                            last = am.reshape(-1, am.shape[-1])[0]
                                            off = 0
                                            for nm, ts in zip(names, token_sets):
                                                n_tok = int(ts.shape[1])
                                                seg = last[off:off + n_tok]
                                                off += n_tok
                                                fuser_attn_bias_segments[nm] = {
                                                    "min": float(seg.min().item()),
                                                    "max": float(seg.max().item()),
                                                    "mean": float(seg.mean().item()),
                                                }
                    except Exception:
                        pass
                    diag: dict[str, object] = {
                        "where": str(where),
                        "nonfinite_pre_sanitize": nonfinite_pre_sanitize,
                        "fuser_diag": fuser_diag,
                        "perceiver_diag": perceiver_diag,
                        "rank": int(rank),
                        "seen_batches": int(seen_batches),
                        "x_diff": float(x_diff),
                        "y_diff_calibrated": float(y_diff_cal),
                        "atol": float(broadcast_atol),
                        "calibrate_pred_output": bool(calibrate_pred_output),
                        "diff_tokens": float(_pair_diff_max(tokens) or 0.0),
                        "diff_assembled_z": float(_pair_diff_max(assembled) or 0.0),
                        "diff_refined": float(_pair_diff_max(refined) or 0.0),
                        "diff_delta": float(_pair_diff_max(delta) or 0.0),
                        "diff_p": float(_pair_diff_max(p) if isinstance(p, torch.Tensor) else 0.0),
                        "diff_y_hat_z": float(_pair_diff_max(y_hat) or 0.0),
                        "diff_pred_final": float(_pair_diff_max(pred) or 0.0),
                        "diff_pred_denorm_uncal": float(_pair_diff_max(pred_denorm_uncal) if isinstance(pred_denorm_uncal, torch.Tensor) else 0.0),
                        "X": _brief_tensor(Xi2),
                        "tokenize_pre": tokenize_pre,
                        "tokenize_meta": tokenize_meta,
                        "tokenize_post_blk0": tokenize_post_blk0,
                        "template_out": template_out,
                        "template_weight_raw": template_weight_raw,
                        "template_token_count": template_token_count,
                        "template_logit_bias": template_logit_bias,
                        "template_prior_mass": template_prior_mass,
                        "fuser_attn_bias_segments": fuser_attn_bias_segments,
                        "tokens": _brief_tensor(tokens),
                        "assembled_z": _brief_tensor(assembled),
                        "refined": _brief_tensor(refined),
                        "y_hat_z": _brief_tensor(y_hat),
                        "pred_final": _brief_tensor(pred),
                        "pred_denorm_uncal": _brief_tensor(pred_denorm_uncal),
                    }
                    with contextlib.suppress(Exception):
                        nf = getattr(m0, "_enn_nonfinite_pre_sanitize", None)
                        if isinstance(nf, list) and nf:
                            diag["nonfinite_pre_sanitize"] = nf
                    fname = f"collapse_stage_diag.rank{int(rank)}.batch{int(seen_batches):06d}.json"
                    out_path = os.path.join(str(chunk_dir), fname)
                    with contextlib.suppress(Exception):
                        os.makedirs(str(chunk_dir), exist_ok=True)
                    collate.write_json(out_path, diag, indent=2)
                    with contextlib.suppress(Exception):
                        _maybe_write_diag_copy(fname, diag)
                    _LOGGER.warning(
                        "[infer][collapse-stage] where=%s diffs: tokens=%.6g assembled_z=%.6g refined=%.6g delta=%.6g p=%.6g y_hat_z=%.6g pred_final=%.6g pred_denorm_uncal=%.6g (saved %s)",
                        str(where),
                        float(diag["diff_tokens"]),
                        float(diag["diff_assembled_z"]),
                        float(diag["diff_refined"]),
                        float(diag["diff_delta"]),
                        float(diag["diff_p"]),
                        float(diag["diff_y_hat_z"]),
                        float(diag["diff_pred_final"]),
                        float(diag["diff_pred_denorm_uncal"]),
                        str(out_path),
                    )
                    _collapse_stage_diag_done = True
                except Exception:
                    return

            def _unwrap_for_stage_diag(m: object) -> torch.nn.Module | None:
                if not isinstance(m, torch.nn.Module):
                    return None
                cur: torch.nn.Module = m
                seen: set[int] = set()
                for _ in range(16):
                    cid = id(cur)
                    if cid in seen:
                        break
                    seen.add(cid)
                    nxt = None
                    for attr in ("_orig_mod", "_original_module", "_uncompiled_module"):
                        cand = getattr(cur, attr, None)
                        if isinstance(cand, torch.nn.Module) and cand is not cur:
                            nxt = cand
                            break
                    if nxt is None:
                        cand = getattr(cur, "module", None)
                        if isinstance(cand, torch.nn.Module) and cand is not cur:
                            nxt = cand
                    if nxt is None:
                        break
                    cur = nxt
                return cur

            def _dump_collapse_stage_diag(
                *,
                where: str,
                Xi2: torch.Tensor,
                seen_batches: int,
                x_diff: float,
                y_diff_cal: float,
            ) -> None:
                nonlocal _collapse_stage_diag_done
                if not collapse_stage_diag_enabled:
                    return
                if collapse_stage_diag_once and _collapse_stage_diag_done:
                    return
                if str(dev_type) != "cuda":
                    return
                try:
                    m0 = _unwrap_for_stage_diag(locals().get("run_model_uncompiled", None))
                    m1 = _unwrap_for_stage_diag(locals().get("run_model", None))
                    mod = m0 if (m0 is not None and hasattr(m0, "_run_forward_core")) else m1
                    if mod is None or (not hasattr(mod, "_run_forward_core")):
                        return
                    fn = getattr(mod, "_run_forward_core", None)
                    if not callable(fn):
                        return

                    with torch.no_grad():
                        pred_denorm, _st, p, assembled, enhanced, delta, tokens, refined = fn(
                            Xi2,
                            export=False,
                            temporal_state=None,
                            causal_mask=None,
                            sanitize_nan=False,
                            calibrate_output=False,
                        )
                        if p is None:
                            y_hat = assembled + delta * 0.5
                        else:
                            y_hat = assembled + p.to(dtype=assembled.dtype) * delta

                        out_shape = tuple(int(x) for x in (ops.out_shape or ()))
                        out_dim = 1
                        for d in out_shape:
                            out_dim *= max(1, int(d))
                        if bool(calibrate_pred_output):
                            sc = getattr(mod, "scaler", None)
                            if sc is not None and hasattr(sc, "calibrate") and hasattr(sc, "denormalize_y"):
                                y_cal = sc.calibrate(y_hat)
                                pred_runtime = sc.denormalize_y(y_cal).reshape(int(y_hat.shape[0]), *out_shape)
                            else:
                                pred_runtime = y_hat.reshape(int(y_hat.shape[0]), out_dim).reshape(int(y_hat.shape[0]), *out_shape)
                        else:
                            pred_runtime = y_hat.reshape(int(y_hat.shape[0]), out_dim).reshape(int(y_hat.shape[0]), *out_shape)

                    diag: dict[str, object] = {
                        "where": str(where),
                        "rank": int(rank),
                        "seen_batches": int(seen_batches),
                        "x_diff": float(x_diff),
                        "y_diff_calibrated": float(y_diff_cal),
                        "calibrate_pred_output": bool(calibrate_pred_output),
                        "broadcast_atol": float(broadcast_atol),
                        "X": _tensor_diag(Xi2),
                        "tokens": _tensor_diag(tokens),
                        "context_z": _tensor_diag(assembled),
                        "refined": _tensor_diag(refined),
                        "delta_z": _tensor_diag(delta),
                        "p": _tensor_diag(p) if torch.is_tensor(p) else {"type": "None"},
                        "y_hat_z": _tensor_diag(y_hat),
                        "pred_runtime": _tensor_diag(pred_runtime),
                        "pred_denorm_uncal": _tensor_diag(pred_denorm),
                        "diff_tokens": _pair_diff_max(tokens),
                        "diff_context_z": _pair_diff_max(assembled),
                        "diff_refined": _pair_diff_max(refined),
                        "diff_delta_z": _pair_diff_max(delta),
                        "diff_p": _pair_diff_max(p) if torch.is_tensor(p) else None,
                        "diff_y_hat_z": _pair_diff_max(y_hat),
                        "diff_pred_runtime": _pair_diff_max(pred_runtime),
                        "diff_pred_denorm_uncal": _pair_diff_max(pred_denorm),
                    }

                    if env_bool("ENN_PRED_COLLAPSE_STAGE_DIAG", default=False):
                        with contextlib.suppress(Exception):
                            fdiag = None
                            pdiag = None
                            mm = run_model.module if hasattr(run_model, "module") else run_model
                            fuser = getattr(mm, "fuser", None) or getattr(mm, "processor", None)
                            if fuser is not None:
                                fdiag = getattr(fuser, "_enn_last_fuser_diag", None)
                                perceiver = getattr(fuser, "perceiver", None)
                                if perceiver is not None:
                                    pdiag = getattr(perceiver, "_enn_last_perceiver_diag", None)

                            raw_stage = None
                            with contextlib.suppress(Exception):
                                core = getattr(mm, "_run_forward_core", None)
                                if callable(core):
                                    Xi_raw = Xi2[:2].detach()
                                    _raw_pred, _raw_st, _raw_p, raw_assembled, _raw_enhanced, _raw_delta, raw_tokens, _raw_refined = core(
                                        Xi_raw,
                                        export=False,
                                        sanitize_nan=False,
                                        calibrate_output=bool(calibrate_pred_output),
                                    )
                                    raw_stage = {
                                        "raw_tokens_nonfinite": int((~torch.isfinite(raw_tokens)).sum().item()) if raw_tokens.is_floating_point() else 0,
                                        "raw_context_nonfinite": int((~torch.isfinite(raw_assembled)).sum().item()) if raw_assembled.is_floating_point() else 0,
                                        "raw_tokens_absmax": float(raw_tokens.abs().max().item()) if raw_tokens.numel() else 0.0,
                                        "raw_context_absmax": float(raw_assembled.abs().max().item()) if raw_assembled.numel() else 0.0,
                                    }

                            diag.setdefault("hooks", {})
                            hooks = diag.get("hooks")
                            if isinstance(hooks, dict):
                                hooks["fuser"] = fdiag
                                hooks["perceiver"] = pdiag
                                hooks["raw_pass"] = raw_stage

                    with contextlib.suppress(Exception):
                        z2 = _try_normalize_x(Xi2)
                        if isinstance(z2, torch.Tensor):
                            diag["Z_norm"] = _tensor_diag(z2)
                            diag["diff_z_norm"] = _pair_diff_max(z2)

                    out_path = os.path.join(
                        str(chunk_dir),
                        f"collapse_stage_diag.rank{int(rank)}.batch{int(seen_batches):06d}.json",
                    )
                    with contextlib.suppress(Exception):
                        os.makedirs(str(chunk_dir), exist_ok=True)
                    collate.write_json(out_path, diag, indent=2)
                    _maybe_write_diag_copy(
                        f"collapse_stage_diag.rank{int(rank)}.batch{int(seen_batches):06d}.json",
                        diag,
                    )
                    _LOGGER.warning(
                        "[infer][collapse-stage] diff: "
                        "Z_norm=%.6g tokens=%.6g ctx_z=%.6g refined=%.6g y_hat_z=%.6g pred=%.6g pred_denorm_uncal=%.6g "
                        "(dumped %s ; copy_dir=%s)",
                        float(diag.get("diff_z_norm") or 0.0),
                        float(diag.get("diff_tokens") or 0.0),
                        float(diag.get("diff_context_z") or 0.0),
                        float(diag.get("diff_refined") or 0.0),
                        float(diag.get("diff_y_hat_z") or 0.0),
                        float(diag.get("diff_pred_runtime") or 0.0),
                        float(diag.get("diff_pred_denorm_uncal") or 0.0),
                        str(out_path),
                        str(_collapse_diag_out_dir),
                    )
                    _collapse_stage_diag_done = True
                except Exception:
                    return
            collapse_force_fp32 = bool(
                env_bool("ENN_PRED_COLLAPSE_FORCE_FP32", default=False)
            )
            collapse_force_fp32_persist = bool(
                env_bool("ENN_PRED_COLLAPSE_FORCE_FP32_PERSIST", default=False)
            )
            collapse_fp32_active = False
            collapse_diag = bool(env_bool("ENN_PRED_COLLAPSE_DIAG", True))
            collapse_diag_save = bool(env_bool("ENN_PRED_COLLAPSE_DIAG_SAVE", True))
            collapse_diag_max_elems = int(env_int("ENN_PRED_COLLAPSE_DIAG_MAX_ELEMS", 64) or 64)
            collapse_diag_once = bool(env_bool("ENN_PRED_COLLAPSE_DIAG_ONCE", True))
            _collapse_diag_done = False

            _collapse_diag_out_dir = (
                str(os.environ.get("ENN_PRED_COLLAPSE_DIAG_DIR") or "").strip()
            )
            if not _collapse_diag_out_dir:
                _collapse_diag_out_dir = str(os.environ.get("ENN_RETURN_DIR") or "").strip()
            if not _collapse_diag_out_dir:
                _collapse_diag_out_dir = os.path.join(tempfile.gettempdir(), "enn_collapse_diag")

            def _maybe_write_diag_copy(filename: str, payload: dict[str, object]) -> None:
                with contextlib.suppress(Exception):
                    out_dir = str(_collapse_diag_out_dir or "").strip()
                    if not out_dir:
                        return
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, filename)
                    collate.write_json(out_path, payload, indent=2)

            def _brief_stats(t: torch.Tensor) -> dict[str, object]:
                try:
                    d: dict[str, object] = {}
                    if not isinstance(t, torch.Tensor):
                        return {"type": str(type(t))}
                    d["shape"] = [int(x) for x in t.shape]
                    d["dtype"] = str(t.dtype)
                    d["device"] = str(t.device)
                    if t.numel() == 0:
                        d["numel"] = 0
                        return d
                    d["numel"] = int(t.numel())
                    if t.is_floating_point():
                        with contextlib.suppress(Exception):
                            d["finite"] = bool(torch.isfinite(t).all().item())
                        tt = t.detach()
                        with contextlib.suppress(Exception):
                            d["abs_max"] = float(tt.abs().max().item())
                        with contextlib.suppress(Exception):
                            v = tt
                            if v.dim() >= 2:
                                s2 = v.to(dtype=torch.float32).std(dim=-1, unbiased=False).mean()
                            else:
                                s2 = v.to(dtype=torch.float32).std(unbiased=False)
                            d["std_mean"] = float(s2.item())
                        with contextlib.suppress(Exception):
                            d["min"] = float(tt.min().item())
                            d["max"] = float(tt.max().item())
                    else:
                        with contextlib.suppress(Exception):
                            d["min"] = int(t.min().item())
                            d["max"] = int(t.max().item())
                    with contextlib.suppress(Exception):
                        flat = t.detach().reshape(-1)
                        k = int(min(int(collapse_diag_max_elems), int(flat.numel())))
                        if k > 0:
                            sample = flat[:k]
                            if sample.is_floating_point():
                                d["sample"] = [float(x) for x in sample.to(dtype=torch.float32).cpu().tolist()]
                            else:
                                d["sample"] = [int(x) for x in sample.cpu().tolist()]
                    return d
                except Exception as e:
                    return {"error": f"{type(e).__name__}: {e}"}

            def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
                try:
                    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
                        return float("nan")
                    if a.numel() == 0 or b.numel() == 0:
                        return 0.0
                    if a.shape != b.shape:
                        if a.dim() >= 1 and b.dim() >= 1 and a.shape[1:] == b.shape[1:]:
                            n = min(int(a.shape[0]), int(b.shape[0]))
                            if n <= 0:
                                return float("nan")
                            a = a[:n]
                            b = b[:n]
                        else:
                            return float("nan")
                    d = (a.detach() - b.detach()).abs()
                    return float(d.max().item())
                except Exception:
                    return float("nan")

            def _try_normalize_x(x_in: torch.Tensor) -> torch.Tensor | None:
                try:
                    sc = getattr(module_eval, "scaler", None)
                    if sc is None:
                        return None
                    fn = getattr(sc, "normalize_x", None)
                    if not callable(fn):
                        return None
                    return fn(x_in)
                except Exception:
                    return None

            def _diag_collapse_once(*, Xi2: torch.Tensor, preds2: torch.Tensor, x_diff: float, y_diff: float, where: str) -> None:
                nonlocal _collapse_diag_done
                if not collapse_diag:
                    return
                if collapse_diag_once and _collapse_diag_done:
                    return
                _collapse_diag_done = True
                diag: dict[str, object] = {}
                try:
                    diag["where"] = str(where)
                    diag["rank"] = int(rank)
                    diag["seen_batches"] = int(seen_batches)
                    diag["x_diff"] = float(x_diff)
                    diag["y_diff_calibrated"] = float(y_diff)
                    diag["X"] = _brief_stats(Xi2)
                    diag["Y_cal"] = _brief_stats(preds2)
                    z2 = _try_normalize_x(Xi2)
                    if z2 is not None and isinstance(z2, torch.Tensor):
                        diag["Z_norm"] = _brief_stats(z2)
                        if int(z2.shape[0]) >= 2:
                            diag["z_diff"] = _max_abs_diff(z2[0], z2[1])
                        else:
                            diag["z_diff"] = 0.0
                    try:
                        raw2 = _td_predict(Xi2, calibrate_output=False)
                        if isinstance(raw2, torch.Tensor):
                            diag["Y_raw"] = _brief_stats(raw2)
                            if int(raw2.shape[0]) >= 2:
                                diag["y_diff_raw"] = _max_abs_diff(raw2[0], raw2[1])
                    except Exception as e:
                        diag["Y_raw_error"] = f"{type(e).__name__}: {e}"
                    try:
                        if bool(force_uncompiled) and (run_model_uncompiled is not None) and (run_model_uncompiled is not run_model):
                            out_uc = run_model_uncompiled(Xi2, calibrate_output=True, return_loss=False)
                            if isinstance(out_uc, tuple):
                                out_uc = out_uc[0]
                            if isinstance(out_uc, torch.Tensor):
                                diag["Y_uncompiled"] = _brief_stats(out_uc)
                                if int(out_uc.shape[0]) >= 2:
                                    diag["y_diff_uncompiled"] = _max_abs_diff(out_uc[0], out_uc[1])
                    except Exception as e:
                        diag["Y_uncompiled_error"] = f"{type(e).__name__}: {e}"
                finally:
                    with contextlib.suppress(Exception):
                        _LOGGER.warning(
                            "[infer][collapse-diag] where=%s x_diff=%.6g y_diff_cal=%.6g diag_keys=%s",
                            str(where),
                            float(x_diff),
                            float(y_diff),
                            ",".join(sorted(list(diag.keys()))),
                        )
                    if collapse_diag_save:
                        with contextlib.suppress(Exception):
                            os.makedirs(str(chunk_dir), exist_ok=True)
                            fname = f"collapse_diag.rank{int(rank)}.batch{int(seen_batches):06d}.json"
                            fp = os.path.join(str(chunk_dir), fname)
                            with open(fp, "w", encoding="utf-8") as f:
                                json.dump(diag, f, ensure_ascii=False, indent=2)
                            _maybe_write_diag_copy(fname, diag)

            collapse_switched_uncompiled = False
            calibrate_pred_output = bool(env_bool("ENN_PRED_CALIBRATE_OUTPUT", True))
            collapse_fallback_raw = bool(env_bool("ENN_PRED_COLLAPSE_FALLBACK_RAW", True))
            collapse_abort = bool(
                env_bool(
                    "ENN_PRED_COLLAPSE_ABORT",
                    default=env_bool("ENN_SANITIZE_NAN_STRICT", default=False),
                )
            )

            class _InferCollapseAbort(RuntimeError):
                pass

            collapse_switched_raw = False
            pred_cg_strict_sync = bool(
                env_bool(
                    "ENN_PRED_CUDAGRAPH_STRICT_SYNC",
                    default=bool(dev_type == "cuda" and (cg_enabled or td_cg_candidate)),
                )
            )

            _pred_disable_decorator = None
            with contextlib.suppress(Exception):
                comp = getattr(torch, "compiler", None)
                cand = getattr(comp, "disable", None) if comp is not None else None
                if callable(cand) and getattr(cand, "__name__", "") == "disable":
                    _pred_disable_decorator = cand
            if _pred_disable_decorator is None:
                with contextlib.suppress(Exception):
                    torch_dynamo = importlib.import_module("torch._dynamo")
                    cand = getattr(torch_dynamo, "disable", None)
                    if callable(cand) and getattr(cand, "__name__", "") == "disable":
                        _pred_disable_decorator = cand

            def _run_model_predict(x: torch.Tensor):
                return run_model(
                    x,
                    calibrate_output=bool(calibrate_pred_output),
                    return_loss=False,
                )

            def _select_pred_model(use_uncompiled: bool) -> torch.nn.Module:
                if bool(use_uncompiled) and (run_model_uncompiled is not run_model):
                    return run_model_uncompiled
                return run_model

            def _run_model_predict_with_calibration(
                x: torch.Tensor,
                *,
                calibrate_output: bool,
                use_uncompiled: bool = False,
                force_fp32: bool = False,
            ):
                nonlocal force_uncompiled, force_eager

                def _invoke_predict(mm: torch.nn.Module):
                    ctx = (
                        eager_ctx_factory()
                        if (force_eager and callable(eager_ctx_factory))
                        else contextlib.nullcontext()
                    )
                    with ctx:
                        if bool(force_fp32) or bool(collapse_fp32_active):
                            with StatelessAutocast.suspend(device):
                                x_fp32 = x
                                if torch.is_tensor(x_fp32) and x_fp32.dtype != torch.float32:
                                    x_fp32 = x_fp32.to(dtype=torch.float32)
                                return mm(
                                    x_fp32,
                                    calibrate_output=bool(calibrate_output),
                                    return_loss=False,
                                )
                        return mm(
                            x,
                            calibrate_output=bool(calibrate_output),
                            return_loss=False,
                        )

                m = _select_pred_model(bool(use_uncompiled))
                try:
                    return _invoke_predict(m)
                except AssertionError:
                    tb = ""
                    is_cg_trees = False
                    try:
                        import traceback as _tb
                        tb = _tb.format_exc()
                        is_cg_trees = "torch/_inductor/cudagraph_trees.py" in tb
                    except Exception:
                        tb = ""
                        is_cg_trees = False
                    if is_cg_trees and env_bool(
                        "ENN_PRED_RETRY_ON_CUDAGRAPH_TREES_ASSERT", default=True
                    ):
                        try:
                            import torch._inductor.config as _icfg
                            if env_bool("ENN_PRED_LOG_CUDAGRAPH_RECOVERY", default=False):
                                _LOGGER.error(
                                    "[infer] inductor cudagraph_trees assertion hit; retrying with triton.cudagraph_trees=0 (keep cudagraphs). "
                                    "Set ENN_PRED_RETRY_ON_CUDAGRAPH_TREES_ASSERT=0 to disable."
                                )
                            with _icfg.patch({"triton.cudagraph_trees": False}):
                                return _invoke_predict(m)
                        except Exception:
                            pass
                    if is_cg_trees and env_bool(
                        "ENN_PRED_RETRY_ON_CUDAGRAPH_DISABLE_ASSERT", default=True
                    ):
                        try:
                            import torch._inductor.config as _icfg
                            if env_bool("ENN_PRED_LOG_CUDAGRAPH_RECOVERY", default=False):
                                _LOGGER.error(
                                    "[infer] inductor cudagraph_trees assertion persists; retrying with triton.cudagraphs=0 (no cudagraph capture, keep compile). "
                                    "Set ENN_PRED_RETRY_ON_CUDAGRAPH_DISABLE_ASSERT=0 to disable."
                                )
                            with _icfg.patch(
                                {"triton.cudagraph_trees": False, "triton.cudagraphs": False}
                            ):
                                return _invoke_predict(m)
                        except Exception:
                            pass
                    if not env_bool(
                        "ENN_PRED_FALLBACK_ON_CUDAGRAPH_ASSERT",
                        default=is_cg_trees
                        or (not env_bool("ENN_SANITIZE_NAN_STRICT", default=False)),
                    ):
                        raise
                    if is_cg_trees:
                        if bool(force_uncompiled) and bool(force_eager):
                            raise
                        if env_bool("ENN_PRED_LOG_CUDAGRAPH_RECOVERY", default=False):
                            _LOGGER.error(
                                "[infer] inductor cudagraph assertion hit; falling back to eager (disable compiled submodules). "
                                "Set ENN_PRED_FALLBACK_ON_CUDAGRAPH_ASSERT=0 to disable."
                            )
                        force_uncompiled = True
                        force_eager = True
                        mm_fb = (
                            run_model_uncompiled
                            if (run_model_uncompiled is not None and run_model_uncompiled is not run_model)
                            else m
                        )
                        return _invoke_predict(mm_fb)
                    raise

                except Exception as e:
                    err_name = type(e).__name__
                    err_msg = ""
                    with contextlib.suppress(Exception):
                        err_msg = str(e)
                    is_backend_failed = bool(
                        ("BackendCompilerFailed" in err_name)
                        or ("backend='inductor'" in err_msg)
                        or ("Offset increment outside graph capture" in err_msg)
                    )
                    if is_backend_failed and env_bool(
                        "ENN_PRED_FALLBACK_ON_COMPILER_ERROR", default=True
                    ):
                        if bool(force_uncompiled) and bool(force_eager):
                            raise
                        _LOGGER.error(
                            "[infer] compiler/capture failure (%s); falling back to eager (disable compiled submodules). "
                            "Set ENN_PRED_FALLBACK_ON_COMPILER_ERROR=0 to disable.",
                            err_name,
                        )
                        force_uncompiled = True
                        force_eager = True
                        mm_fb = (
                            run_model_uncompiled
                            if (run_model_uncompiled is not None and run_model_uncompiled is not run_model)
                            else m
                        )
                        return _invoke_predict(mm_fb)
                    raise

            def _maybe_enable_fp32_collapse_fallback() -> bool:
                nonlocal collapse_fp32_active, force_uncompiled, force_eager
                if (not bool(collapse_force_fp32)) or bool(collapse_fp32_active):
                    return False
                if str(dev_type) != "cuda":
                    return False
                cast_compiled = env_bool(
                    "ENN_PRED_COLLAPSE_CAST_COMPILED_MODEL", default=False
                )
                mods: list[torch.nn.Module] = []
                for cand in (run_model_uncompiled, run_model):
                    if not isinstance(cand, torch.nn.Module):
                        continue
                    if (
                        (cand is run_model)
                        and (run_model_uncompiled is not run_model)
                        and (not bool(cast_compiled))
                    ):
                        continue
                    if cand not in mods:
                        mods.append(cand)
                if not mods:
                    return False
                try:
                    for mm in mods:
                        mm.to(dtype=torch.float32)
                    force_uncompiled = True
                    force_eager = True
                    collapse_fp32_active = bool(collapse_force_fp32_persist)
                    _LOGGER.warning(
                        "[infer] collapse persisted; enabled fp32+no-autocast fallback (ENN_PRED_COLLAPSE_FORCE_FP32=1)."
                    )
                    return True
                except RuntimeError as e:
                    if is_oom_error(e):
                        _LOGGER.warning(
                            "[infer] fp32 upcast fallback skipped due to OOM; keeping current precision path."
                        )
                        return False
                    raise

            _run_model_predict_disabled = None
            if callable(_pred_disable_decorator):
                with contextlib.suppress(Exception):
                    wrapped = _pred_disable_decorator(_run_model_predict)
                    if callable(wrapped):
                        _run_model_predict_disabled = wrapped

            def _td_predict(
                x: torch.Tensor,
                *,
                calibrate_output: bool | None = None,
                use_uncompiled: bool | None = None,
                force_fp32: bool = False,
            ) -> torch.Tensor:
                ctx = (
                    eager_ctx_factory()
                    if (force_eager and callable(eager_ctx_factory))
                    else contextlib.nullcontext()
                )
                uc = bool(force_uncompiled) if use_uncompiled is None else bool(use_uncompiled)
                with ctx:
                    if (
                        force_eager
                        and callable(_run_model_predict_disabled)
                        and calibrate_output is None
                        and use_uncompiled is None
                        and (not force_fp32)
                    ):
                        out = _run_model_predict_disabled(x)
                    elif calibrate_output is not None:
                        out = _run_model_predict_with_calibration(
                            x,
                            calibrate_output=bool(calibrate_output),
                            use_uncompiled=uc,
                            force_fp32=bool(force_fp32),
                        )
                    else:
                        out = _run_model_predict_with_calibration(
                            x, calibrate_output=bool(calibrate_pred_output), use_uncompiled=uc
                        )
                if isinstance(out, tuple):
                    out = out[0]
                if not isinstance(out, torch.Tensor):
                    raise RuntimeError("infer: unexpected model output type")
                return out.detach()

            def _pred_reuse_active(cur_fn: object) -> bool:
                if bool(cg_enabled):
                    return True
                if (
                    bool(td_cg_active)
                    and (td_cg_mod is not None)
                    and (cur_fn is td_cg_mod)
                    and (not bool(td_cg_disabled))
                    and (td_cg_pad_buf is not None)
                    and (td_cg_mb is not None)
                    and (not bool(force_eager))
                    and (not bool(force_single))
                ):
                    return True
                return False

            def _td_benchmark(bs_now: int) -> int:
                mb_cfg = int(getattr(model, "microbatch", 0) or 0)
                dl_bs = int(getattr(data_loader, "batch_size", 0) or 0)
                mb_target = (
                    int(mb_cfg)
                    if int(mb_cfg) > 0
                    else (int(dl_bs) if int(dl_bs) > 0 else int(bs_now))
                )
                if int(dl_bs) > 0:
                    mb_target = min(int(mb_target), int(dl_bs))
                return int(max(1, int(mb_target)))

            def _td_cudagraph(bs_now: int, X_now: torch.Tensor) -> None:
                nonlocal td_cg_active, td_cg_disabled, td_cg_mb, td_cg_mod
                nonlocal td_cg_pad_buf, td_cg_seen, td_cg_max_bs, td_cg_target, td_cg_x_inner_shape
                if (
                    td_cg_disabled
                    or td_cg_active
                    or (TD_CudaGraphModule is None)
                    or (td_cg_mod is not None)
                ):
                    return
                td_cg_seen += 1
                td_cg_max_bs = max(int(td_cg_max_bs), int(bs_now))
                if td_cg_target is None:
                    td_cg_target = _td_benchmark(int(bs_now))
                mb_cap = None
                if int(bs_now) >= int(td_cg_target):
                    mb_cap = int(td_cg_target)
                elif int(td_cg_seen) >= 4 and int(td_cg_max_bs) > 0:
                    mb_cap = int(min(int(td_cg_target), int(td_cg_max_bs)))
                if mb_cap is None:
                    return
                mb_cap = int(max(1, int(mb_cap)))
                td_cg_x_inner_shape = tuple(
                    int(d) for d in tuple(X_now.shape[1:])
                )
                td_cg_mb = int(mb_cap)
                td_cg_pad_buf = X_now.new_empty(
                    (int(td_cg_mb),) + tuple(td_cg_x_inner_shape)
                )
                n0 = int(min(int(bs_now), int(td_cg_mb)))
                td_cg_pad_buf[:n0].copy_(X_now[:n0])
                if n0 < int(td_cg_mb):
                    td_cg_pad_buf[n0:].copy_(
                        X_now[n0 - 1 : n0].expand(
                            int(td_cg_mb) - n0, *tuple(td_cg_x_inner_shape)
                        )
                    )
                try:
                    prewarm = int(env_int("ENN_PRED_TD_CUDAGRAPH_PREWARM", 2) or 0)
                    prewarm = max(0, min(8, int(prewarm)))
                    if prewarm > 0:
                        for _ in range(int(prewarm)):
                            _ = _td_predict(td_cg_pad_buf)

                    td_cg_mod = TD_CudaGraphModule(
                        _td_predict, warmup=0, device=device
                    )

                    _compile_disable = getattr(getattr(torch, "compiler", None), "disable", None)
                    _compile_disable_ctx = (
                        _compile_disable() if callable(_compile_disable) else contextlib.nullcontext()
                    )
                    with _compile_disable_ctx:
                        for _ in range(3):
                            _ = td_cg_mod(td_cg_pad_buf)
                    td_cg_active = True
                    with contextlib.suppress(Exception):
                        setattr(model, "microbatch", int(td_cg_mb))
                except Exception:
                    td_cg_disabled = True
                    td_cg_active = False
                    td_cg_mb = None
                    td_cg_mod = None
                    td_cg_pad_buf = None
                    td_cg_x_inner_shape = None

            row_ids_buf = None
            pad_buf = None
            dl_type = type(data_loader).__name__
            dl_len = None
            with contextlib.suppress(Exception):
                dl_len = len(data_loader)
            seen_batches = 0
            none_batches = 0
            empty_x_batches = 0
            appended_rows_total = 0
            first_batch_info = {}
            data_iter = iter(data_loader)
            try:
                _first_batch = next(data_iter)
            except StopIteration:
                r = (
                    int(torch.distributed.get_rank())
                    if is_distributed()
                    else 0
                )
                w = int(get_world_size(device)) if is_distributed() else 1
                with contextlib.suppress(Exception):
                    chain = []
                    cur = data_loader
                    for _ in range(8):
                        if cur is None:
                            break
                        try:
                            clen = len(cur)
                        except Exception:
                            clen = None
                        chain.append(
                            f"{type(cur).__module__}.{type(cur).__name__}(len={clen})"
                        )
                        nxt = getattr(cur, "_src", None)
                        if nxt is None and hasattr(cur, "_base_iterable"):
                            nxt = getattr(cur, "_base_iterable", None)
                        if nxt is None or nxt is cur:
                            break
                        cur = nxt
                    _LOGGER.error(
                        "[DIAG] infer: loader chain: %s", " -> ".join(chain)
                    )
                    base = getattr(data_loader, "_base_iterable", None)
                    if base is not None:
                        try:
                            next(iter(base))
                            _LOGGER.error(
                                "[DIAG] infer: _base_iterable yields batches; Stream layer likely blocking/dropping."
                            )
                        except StopIteration:
                            _LOGGER.error(
                                "[DIAG] infer: _base_iterable also produced 0 batches (source pipeline empty)."
                            )
                        except Exception as e:
                            _LOGGER.exception(
                                "[DIAG] infer: probing _base_iterable failed: %s",
                                e,
                            )
                    src = getattr(data_loader, "_src", None)
                    if src is not None:
                        try:
                            next(iter(src))
                            _LOGGER.error(
                                "[DIAG] infer: underlying _src yields batches; wrapper/preload layer likely blocking."
                            )
                        except StopIteration:
                            _LOGGER.error(
                                "[DIAG] infer: underlying _src also produced 0 batches."
                            )
                        except Exception as e:
                            _LOGGER.exception(
                                "[DIAG] infer: probing underlying _src failed: %s",
                                e,
                            )
                raise RuntimeError(
                    "infer: data_loader produced 0 batches. "
                    f"(rank={r}/{w}, device={device}, loader={dl_type}, len={dl_len}, chunk_dir={chunk_dir})"
                )
            for batch in itertools.chain([_first_batch], data_iter):
                seen_batches += 1
                if batch is None:
                    none_batches += 1
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
                bs = (
                    int(getattr(X, "shape", [0])[0])
                    if hasattr(X, "shape")
                    else 0
                )
                if (not first_batch_info) and seen_batches == 1:
                    with contextlib.suppress(Exception):
                        first_batch_info["batch_type"] = type(batch).__name__
                        first_batch_info["X_shape"] = [
                            int(x) for x in getattr(X, "shape", ())
                        ]
                        first_batch_info["X_dtype"] = str(
                            getattr(X, "dtype", None)
                        )
                        first_batch_info["X_device"] = str(
                            getattr(X, "device", None)
                        )
                        if row_ids is not None and hasattr(row_ids, "shape"):
                            first_batch_info["row_ids_shape"] = [
                                int(x) for x in getattr(row_ids, "shape", ())
                            ]
                            first_batch_info["row_ids_device"] = str(
                                getattr(row_ids, "device", None)
                            )
                if bs <= 0:
                    empty_x_batches += 1
                    if status_bar is not None:
                        status_bar.update(1)
                    continue
                if (not force_single) and (not force_eager) and (not td_cg_disabled) and (not td_cg_active):
                    _td_cudagraph(int(bs), X)
                if row_ids is None:
                    if row_ids_buf is None or int(row_ids_buf.numel()) < bs:
                        row_ids_buf = torch.empty(
                            (bs,), device="cpu", dtype=torch.int64
                        )
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
                mb_cfg = int(getattr(model, "microbatch", 0) or 0)
                if mb_cfg <= 0:
                    mb_eager = bs
                else:
                    mb_eager = min(bs, int(mb_cfg))
                mb_eager = max(1, int(mb_eager))
                use_td_cg = bool(
                    td_cg_active
                    and (td_cg_mod is not None)
                    and (td_cg_mb is not None)
                    and (not bool(force_eager))
                )
                if force_single:
                    use_td_cg = False
                if use_td_cg and (td_cg_x_inner_shape is not None):
                    if tuple(int(d) for d in tuple(X.shape[1:])) != tuple(
                        td_cg_x_inner_shape
                    ):
                        td_cg_active = False
                        td_cg_disabled = True
                        td_cg_mb = None
                        td_cg_mod = None
                        td_cg_pad_buf = None
                        td_cg_x_inner_shape = None
                        use_td_cg = False
                mb = (
                    1 if force_single else (int(td_cg_mb) if use_td_cg else int(mb_eager))
                )
                predict_fn = td_cg_mod if use_td_cg else _td_predict
                start = 0
                while start < bs:
                    end = min(bs, start + mb)
                    sl = slice(start, end)
                    Xi = X[sl]
                    rows_i = row_ids[sl]
                    n_i = int(end - start)
                    Xi_pad = None
                    pad_n = 0
                    Xi_run = Xi
                    if use_td_cg:
                        if (
                            (td_cg_pad_buf is not None)
                            and (td_cg_x_inner_shape is not None)
                            and tuple(int(d) for d in tuple(Xi.shape[1:]))
                            != tuple(td_cg_x_inner_shape)
                        ):
                            raise RuntimeError(
                                "infer: input shape changed during td-cudagraph run"
                            )
                        if n_i < mb:
                            pad_n = int(mb - n_i)
                        try:
                            td_cg_pad_buf[:n_i].copy_(Xi, non_blocking=True)
                            if n_i < mb:
                                td_cg_pad_buf[n_i:].zero_()
                            Xi_run = td_cg_pad_buf
                        except Exception:
                            Xi_pad = None
                            pad_n = 0
                            Xi_run = Xi
                    elif cg_enabled and n_i < mb:
                        pad_n = int(mb - n_i)
                        try:
                            want_shape = (int(mb),) + tuple(Xi.shape[1:])
                            if (
                                pad_buf is None
                                or pad_buf.shape != want_shape
                                or pad_buf.dtype != Xi.dtype
                                or pad_buf.device != Xi.device
                            ):
                                pad_buf = Xi.new_empty(want_shape)
                            Xi_pad = pad_buf
                            Xi_pad[:n_i].copy_(Xi)
                            Xi_pad[n_i:].copy_(
                                Xi[-1:].expand(pad_n, *tuple(Xi.shape[1:]))
                            )
                            Xi_run = Xi_pad
                        except Exception:
                            Xi_pad = None
                            pad_n = 0
                            Xi_run = Xi
                    elif cg_enabled:
                        pad_n = int(max(0, int(mb) - int(n_i)))
                        try:
                            want_shape = (int(mb),) + tuple(Xi.shape[1:])
                            if (
                                pad_buf is None
                                or pad_buf.shape != want_shape
                                or pad_buf.dtype != Xi.dtype
                                or pad_buf.device != Xi.device
                            ):
                                pad_buf = Xi.new_empty(want_shape)
                            Xi_pad = pad_buf
                            Xi_pad[:n_i].copy_(Xi)
                            if pad_n > 0:
                                Xi_pad[n_i:].copy_(
                                    Xi[-1:].expand(pad_n, *tuple(Xi.shape[1:]))
                                )
                            Xi_run = Xi_pad
                        except Exception:
                            Xi_pad = None
                            pad_n = 0
                            Xi_run = Xi
                    try:
                        reuse_risk = bool(cg_enabled) or bool(use_td_cg)
                        if (
                            pred_cg_strict_sync
                            and reuse_risk
                            and getattr(getattr(Xi_run, "device", None), "type", None) == "cuda"
                        ):
                            sync_accelerator(dev_obj)
                        if dev_type == "cuda":
                            cudagraph_mark_step_begin()
                        out = predict_fn(Xi_run)
                        if (
                            pred_cg_strict_sync
                            and reuse_risk
                            and getattr(getattr(out, "device", None), "type", None) == "cuda"
                        ):
                            sync_accelerator(dev_obj)
                        if isinstance(out, torch.Tensor):
                            if int(n_i) < int(mb) and int(out.shape[0]) == int(mb):
                                out = out[: int(n_i)]
                    except RuntimeError as e:
                        if is_oom_error(e) and mb > 1:
                            with contextlib.suppress(Exception):
                                empty_device_cache(
                                    device=device,
                                    do_gc=False,
                                    min_interval_s=0.0,
                                )
                            mb = max(1, mb // 2)
                            try:
                                setattr(model, "microbatch", mb)
                            except Exception:
                                pass
                            with contextlib.suppress(Exception):
                                del Xi, Xi_pad
                            if td_cg_active:
                                td_cg_active = False
                                td_cg_disabled = True
                                td_cg_mb = None
                                td_cg_mod = None
                                td_cg_pad_buf = None
                                td_cg_x_inner_shape = None
                                predict_fn = _td_predict
                                use_td_cg = False
                            continue
                        raise
                    finally:
                        if dev_type == "cuda":
                            cudagraph_mark_step_end()
                    preds = out
                    if not isinstance(preds, torch.Tensor):
                        raise RuntimeError(
                            "infer: unexpected model output type"
                        )
                    if pad_n > 0 and int(preds.shape[0]) == int(mb):
                        preds = preds[:n_i]
                    if diag_overwrite and isinstance(preds, torch.Tensor) and getattr(getattr(preds, "device", None), "type", None) == "cuda":
                        reuse_risk_diag = bool(cg_enabled) or bool(use_td_cg)
                        if reuse_risk_diag:
                            cur_batch = int(seen_batches)
                            cur_slice = (int(start), int(end))
                            cur_ptr = None
                            with contextlib.suppress(Exception):
                                cur_ptr = int(preds.data_ptr())

                            if prev_pred_ptr is not None and cur_ptr is not None and int(cur_ptr) == int(prev_pred_ptr):
                                _LOGGER.warning(
                                    "[infer][overwrite-diag] output data_ptr reused: prev_batch=%s prev_slice=%s curr_batch=%s curr_slice=%s ptr=0x%x cg=%s td_cg=%s",
                                    prev_pred_step,
                                    prev_pred_slice,
                                    cur_batch,
                                    cur_slice,
                                    int(cur_ptr),
                                    bool(cg_enabled),
                                    bool(use_td_cg),
                                )

                            overwrite_max = None
                            overwrite_idx = None
                            cur_prev = None
                            prev0 = prev_pred_sample_cpu
                            if prev_pred_ref is not None and prev0 is not None:
                                try:
                                    cur_prev = prev_pred_ref.detach().reshape(-1)[: int(diag_overwrite_n)]
                                    if getattr(getattr(cur_prev, "device", None), "type", None) != "cpu":
                                        cur_prev = cur_prev.to(device="cpu")
                                    if isinstance(cur_prev, torch.Tensor) and cur_prev.dtype != torch.float32:
                                        cur_prev = cur_prev.to(dtype=torch.float32)
                                    if isinstance(prev0, torch.Tensor) and prev0.dtype != torch.float32:
                                        prev0 = prev0.to(dtype=torch.float32)
                                    if int(cur_prev.numel()) == int(prev0.numel()) and int(cur_prev.numel()) > 0:
                                        diff = (cur_prev - prev0).abs()
                                        dmax, imax = diff.max(dim=0)
                                        overwrite_max = float(dmax.item())
                                        overwrite_idx = int(imax.item())
                                        if overwrite_max <= 0.0:
                                            overwrite_max = None
                                            overwrite_idx = None
                                except Exception as e:
                                    prev_d = None
                                    cur_d = None
                                    with contextlib.suppress(Exception):
                                        prev_d = _tensor_diag(prev_pred_ref, sample_n=16)
                                    with contextlib.suppress(Exception):
                                        cur_d = _tensor_diag(preds, sample_n=16)
                                    _LOGGER.exception(
                                        "[infer][overwrite-diag] diagnostic FAILED: prev_batch=%s prev_slice=%s curr_batch=%s curr_slice=%s prev_ptr=%s curr_ptr=%s cg=%s td_cg=%s prev_row_ids=%s curr_row_ids=%s prev=%s curr=%s",
                                        prev_pred_step,
                                        prev_pred_slice,
                                        cur_batch,
                                        cur_slice,
                                        (f"0x{int(prev_pred_ptr):x}" if prev_pred_ptr is not None else None),
                                        (f"0x{int(cur_ptr):x}" if cur_ptr is not None else None),
                                        bool(cg_enabled),
                                        bool(use_td_cg),
                                        _ids_diag(prev_pred_rows_cpu),
                                        _ids_diag(rows_i),
                                        prev_d,
                                        cur_d,
                                    )
                                    if diag_overwrite_abort:
                                        raise RuntimeError(
                                            "infer: overwrite diagnostic failed while ENN_PRED_DIAG_OVERWRITE_ABORT=1. Cannot guarantee cudagraph safety; aborting."
                                        ) from e
                                if overwrite_max is not None:
                                    prev_shape = [int(v) for v in tuple(getattr(prev_pred_ref, "shape", ()))] if prev_pred_ref is not None else []
                                    multi = _unravel_index(int(overwrite_idx or 0), prev_shape) if prev_shape else []
                                    row_in_prev = int(multi[0]) if multi else 0
                                    axis1_idx = int(multi[1]) if len(multi) >= 2 else None
                                    axis2_idx = int(multi[2]) if len(multi) >= 3 else None
                                    row_id = None
                                    x_row_head = None
                                    x_row_stats = None
                                    x_run_row_head = None
                                    x_run_row_stats = None

                                    curr_x_run_ptr = None
                                    with contextlib.suppress(Exception):
                                        if torch.is_tensor(Xi_run):
                                            curr_x_run_ptr = int(Xi_run.data_ptr())
                                    curr_x_run_source = None
                                    with contextlib.suppress(Exception):
                                        if Xi_run is Xi:
                                            curr_x_run_source = 'Xi'
                                        elif use_td_cg and (td_cg_pad_buf is not None) and (Xi_run is td_cg_pad_buf):
                                            curr_x_run_source = 'td_cg_pad_buf'
                                        elif Xi_pad is not None and (Xi_run is Xi_pad):
                                            curr_x_run_source = 'pad_buf'
                                        else:
                                            curr_x_run_source = 'other'

                                    with contextlib.suppress(Exception):
                                        if prev_pred_rows_cpu is not None and int(prev_pred_rows_cpu.numel()) > row_in_prev:
                                            row_id = int(prev_pred_rows_cpu[row_in_prev].item())
                                    with contextlib.suppress(Exception):
                                        if prev_pred_X_cpu is not None and int(prev_pred_X_cpu.shape[0]) > row_in_prev:
                                            xrow = prev_pred_X_cpu[row_in_prev]
                                            xflat = xrow.detach().reshape(-1)
                                            n = int(min(16, int(xflat.numel())))
                                            x_row_head = [float(v) for v in xflat[:n].tolist()]
                                            x_row_stats = _row_stats(xrow)
                                    with contextlib.suppress(Exception):
                                        if prev_pred_X_run_cpu is not None and int(prev_pred_X_run_cpu.shape[0]) > row_in_prev:
                                            xrow = prev_pred_X_run_cpu[row_in_prev]
                                            xflat = xrow.detach().reshape(-1)
                                            n = int(min(16, int(xflat.numel())))
                                            x_run_row_head = [float(v) for v in xflat[:n].tolist()]
                                            x_run_row_stats = _row_stats(xrow)

                                    prev_val = None
                                    cur_val = None
                                    with contextlib.suppress(Exception):
                                        if prev0 is not None and overwrite_idx is not None and int(prev0.numel()) > int(overwrite_idx):
                                            prev_val = float(prev0[int(overwrite_idx)].item())
                                    with contextlib.suppress(Exception):
                                        if cur_prev is not None and overwrite_idx is not None and int(cur_prev.numel()) > int(overwrite_idx):
                                            cur_val = float(cur_prev[int(overwrite_idx)].item())

                                    prev_d = _tensor_diag(prev_pred_ref, sample_n=min(32, int(diag_overwrite_n)))
                                    cur_d = _tensor_diag(preds, sample_n=min(32, int(diag_overwrite_n)))
                                    payload: dict[str, object] = {
                                        "kind": "pred_overwrite",
                                        "rank": int(rank),
                                        "timestamp_ms": int(time.time() * 1000),
                                        "cg_enabled": bool(cg_enabled),
                                        "td_cg": bool(use_td_cg),
                                        "prev_batch": prev_pred_step,
                                        "prev_slice": prev_pred_slice,
                                        "curr_batch": cur_batch,
                                        "curr_slice": cur_slice,
                                        "prev_ptr": (f"0x{int(prev_pred_ptr):x}" if prev_pred_ptr is not None else None),
                                        "curr_ptr": (f"0x{int(cur_ptr):x}" if cur_ptr is not None else None),
                                        "row_id": row_id,
                                        "row_in_prev": row_in_prev,
                                        "multi_index": multi,
                                        "axis1_index": axis1_idx,
                                        "axis2_index": axis2_idx,
                                        "x_row_head": x_row_head,
                                        "x_row_stats": x_row_stats,
                                        "x_run_row_head": x_run_row_head,
                                        "x_run_row_stats": x_run_row_stats,
                                        "prev_pad_n": int(prev_pred_pad_n) if prev_pred_pad_n is not None else None,
                                        "curr_pad_n": int(pad_n),
                                        "prev_mb": int(prev_pred_mb) if prev_pred_mb is not None else None,
                                        "curr_mb": int(mb),
                                        "prev_n_i": int(prev_pred_n_i) if prev_pred_n_i is not None else None,
                                        "curr_n_i": int(n_i),
                                        "prev_X_run_ptr": (f"0x{int(prev_pred_X_run_ptr):x}" if prev_pred_X_run_ptr is not None else None),
                                        "curr_X_run_ptr": (f"0x{int(curr_x_run_ptr):x}" if curr_x_run_ptr is not None else None),
                                        "prev_X_run_source": prev_pred_X_run_source,
                                        "curr_X_run_source": curr_x_run_source,
                                        "prev_X_run_diag": _x_diag(prev_pred_X_run_cpu) if prev_pred_X_run_cpu is not None else None,
                                        "curr_X_run_diag": _x_diag(Xi_run),
                                        "overwrite_max": float(overwrite_max),
                                        "overwrite_sample_index": int(overwrite_idx or 0),
                                        "prev_value_at_max": prev_val,
                                        "curr_value_at_max": cur_val,
                                        "prev_row_ids_diag": _ids_diag(prev_pred_rows_cpu),
                                        "curr_row_ids_diag": _ids_diag(rows_i),
                                        "prev_X_diag": _x_diag(prev_pred_X_cpu) if prev_pred_X_cpu is not None else None,
                                        "curr_X_diag": _x_diag(Xi),
                                        "prev_pred_diag": prev_d,
                                        "curr_pred_diag": cur_d,
                                    }

                                    _row_part = f"row{row_id}" if row_id is not None else "rowNA"
                                    _a1_part = f"a1{int(axis1_idx)}" if axis1_idx is not None else "a1NA"
                                    _a2_part = f"a2{int(axis2_idx)}" if axis2_idx is not None else "a2NA"
                                    payload["filename"] = (
                                        f"overwrite_rank{int(rank)}_prevb{prev_pred_step}_prevsl{prev_pred_slice}_curb{cur_batch}_cursl{cur_slice}_"
                                        f"prevptr{int(prev_pred_ptr) if prev_pred_ptr is not None else 0:x}_curptr{int(cur_ptr) if cur_ptr is not None else 0:x}_"
                                        f"{_row_part}_{_a1_part}_{_a2_part}.json"
                                    )
                                    dump_path = _dump_overwrite_diag(payload)

                                    _LOGGER.error(
                                        "[infer][overwrite-diag] OUTPUT OVERWRITE DETECTED: max|prev-prev_snapshot|=%.6g prev_batch=%s prev_slice=%s curr_batch=%s curr_slice=%s prev_ptr=%s curr_ptr=%s row_id=%s multi=%s a1=%s a2=%s x_head=%s x_stats=%s x_run_head=%s x_run_stats=%s x_run_src_prev=%s x_run_src_curr=%s dump=%s",
                                        float(overwrite_max),
                                        prev_pred_step,
                                        prev_pred_slice,
                                        cur_batch,
                                        cur_slice,
                                        (f"0x{int(prev_pred_ptr):x}" if prev_pred_ptr is not None else None),
                                        (f"0x{int(cur_ptr):x}" if cur_ptr is not None else None),
                                        row_id,
                                        multi,
                                        axis1_idx,
                                        axis2_idx,
                                        x_row_head,
                                        x_row_stats,
                                        x_run_row_head,
                                        x_run_row_stats,
                                        prev_pred_X_run_source,
                                        curr_x_run_source,
                                        dump_path,
                                    )
                                    if diag_overwrite_abort:
                                        raise RuntimeError(
                                            "infer: output overwrite detected (likely cudagraph buffer reuse). "
                                            f"prev={prev_pred_step}/{prev_pred_slice}, curr={cur_batch}/{cur_slice}, prev_ptr={prev_pred_ptr}, curr_ptr={cur_ptr}, "
                                            f"row_id={row_id}, multi={multi}, max_abs={overwrite_max}. dump={dump_path}"
                                        )
                            prev_pred_ref = preds
                            prev_pred_step = cur_batch
                            prev_pred_slice = cur_slice
                            prev_pred_ptr = int(cur_ptr) if cur_ptr is not None else None
                            with contextlib.suppress(Exception):
                                prev_pred_rows_cpu = rows_i.detach().to(device='cpu', dtype=torch.int64).reshape(-1).clone()
                            with contextlib.suppress(Exception):
                                xcpu = Xi.detach()
                                if getattr(getattr(xcpu, 'device', None), 'type', None) != 'cpu':
                                    xcpu = xcpu.to('cpu')
                                if isinstance(xcpu, torch.Tensor) and xcpu.dtype != torch.float32:
                                    xcpu = xcpu.to(dtype=torch.float32)
                                prev_pred_X_cpu = xcpu.clone() if isinstance(xcpu, torch.Tensor) else None
                                prev_pred_pad_n = int(pad_n)
                                prev_pred_mb = int(mb)
                                prev_pred_n_i = int(n_i)
                                with contextlib.suppress(Exception):
                                    if torch.is_tensor(Xi_run):
                                        prev_pred_X_run_ptr = int(Xi_run.data_ptr())
                                with contextlib.suppress(Exception):
                                    if Xi_run is Xi:
                                        prev_pred_X_run_source = 'Xi'
                                    elif use_td_cg and (td_cg_pad_buf is not None) and (Xi_run is td_cg_pad_buf):
                                        prev_pred_X_run_source = 'td_cg_pad_buf'
                                    elif Xi_pad is not None and (Xi_run is Xi_pad):
                                        prev_pred_X_run_source = 'pad_buf'
                                    else:
                                        prev_pred_X_run_source = 'other'
                                with contextlib.suppress(Exception):
                                    xrun_cpu = Xi_run.detach()
                                    if getattr(getattr(xrun_cpu, 'device', None), 'type', None) != 'cpu':
                                        xrun_cpu = xrun_cpu.to('cpu')
                                    if isinstance(xrun_cpu, torch.Tensor) and xrun_cpu.dtype != torch.float32:
                                        xrun_cpu = xrun_cpu.to(dtype=torch.float32)
                                    if isinstance(xrun_cpu, torch.Tensor) and int(xrun_cpu.shape[0]) >= int(n_i):
                                        xrun_cpu = xrun_cpu[: int(n_i)]
                                    prev_pred_X_run_cpu = xrun_cpu.clone() if isinstance(xrun_cpu, torch.Tensor) else None
                            with contextlib.suppress(Exception):
                                samp = preds.detach().reshape(-1)[: int(diag_overwrite_n)]
                                if getattr(getattr(samp, "device", None), "type", None) != "cpu":
                                    samp = samp.to(device="cpu")
                                if isinstance(samp, torch.Tensor) and samp.dtype != torch.float32:
                                    samp = samp.to(dtype=torch.float32)
                                if isinstance(samp, torch.Tensor) and (not samp.is_contiguous()):
                                    samp = samp.contiguous()
                                prev_pred_sample_cpu = samp.clone() if isinstance(samp, torch.Tensor) else None
                    if (
                        (not force_single)
                        and bool(detect_broadcast)
                        and (not broadcast_checked)
                        and int(n_i) >= 2
                        and hasattr(preds, "shape")
                        and int(getattr(preds, "shape", (0,))[0]) >= 2
                    ):
                        broadcast_checked = True
                        try:
                            x0 = Xi[0].detach()
                            x1 = Xi[1].detach()
                            x_diff = (x0 - x1).abs().max()
                            if float(x_diff.item()) > 0.0:
                                y0 = preds[0].detach()
                                y1 = preds[1].detach()
                                is_broadcast_like, bstats = _broadcast_like(
                                    y0, y1, atol=float(broadcast_atol)
                                )
                                y_diff = float(bstats.get("y_max", float("nan")))
                                if bool(is_broadcast_like):
                                    _dump_collapse_stage_diag_from_model(
                                        Xi,
                                        where="broadcast_trigger",
                                        x_diff=float(x_diff.item()),
                                        y_diff_cal=float(y_diff),
                                    )
                                    with contextlib.suppress(Exception):
                                        _diag_collapse_once(
                                            Xi2=Xi[:2].detach(),
                                            preds2=preds[:2].detach(),
                                            x_diff=float(x_diff.item()),
                                            y_diff=float(y_diff),
                                            where="broadcast_trigger",
                                        )
                                    _LOGGER.warning(
                                        "[infer] detected batch-broadcasted predictions (inputs differ but outputs are ~equal). "
                                        "max|Y0-Y1|=%.6g match_frac=%.5f rel_mean=%.3e (atol=%.3e, sample_n=%.0f). "
                                        "Falling back to per-sample inference (microbatch=1) for correctness.",
                                        float(y_diff),
                                        float(bstats.get("y_match_frac", float("nan"))),
                                        float(bstats.get("y_rel_mean", float("nan"))),
                                        float(broadcast_atol),
                                        float(bstats.get("sample_n", float("nan"))),
                                    )
                                    if eager_on_broadcast:
                                        force_eager = True

                                    force_single = True
                                    td_cg_active = False
                                    td_cg_disabled = True
                                    td_cg_mb = None
                                    td_cg_mod = None
                                    td_cg_pad_buf = None
                                    td_cg_x_inner_shape = None
                                    use_td_cg = False
                                    mb = 1
                                    predict_fn = _td_predict
                                    with contextlib.suppress(Exception):
                                        setattr(model, "microbatch", 1)
                                    with contextlib.suppress(Exception):
                                        if torch.is_tensor(preds) and getattr(preds.device, "type", None) == "cuda":
                                            _preds_tmp = preds
                                            preds = None
                                            del _preds_tmp
                                            empty_device_cache(
                                                device=dev_obj, do_gc=False, min_interval_s=0.0
                                            )
                                    x1_buf = Xi.new_empty((1,) + tuple(Xi.shape[1:]))
                                    preds_fix_cpu: torch.Tensor | None = None
                                    for j in range(int(n_i)):
                                        x1_buf.copy_(Xi[j : j + 1])
                                        if (
                                            pred_cg_strict_sync
                                            and getattr(x1_buf.device, "type", None) == "cuda"
                                        ):
                                            sync_accelerator(dev_obj)
                                        if dev_type == "cuda":
                                            cudagraph_mark_step_begin()
                                        pj = _td_predict(
                                            x1_buf,
                                            calibrate_output=(bool(calibrate_pred_output) if "calibrate_pred_output" in locals() else None),
                                        )
                                        if dev_type == "cuda":
                                            cudagraph_mark_step_end()
                                        if (
                                            pred_cg_strict_sync
                                            and getattr(getattr(pj, "device", None), "type", None) == "cuda"
                                        ):
                                            sync_accelerator(dev_obj)
                                        pj_cpu = pj.detach()
                                        if getattr(pj_cpu.device, "type", None) != "cpu":
                                            pj_cpu = pj_cpu.to(device="cpu")
                                        with contextlib.suppress(Exception):
                                            pj_cpu = pj_cpu.contiguous()
                                        pj_cpu = pj_cpu.clone()
                                        if preds_fix_cpu is None:
                                            preds_fix_cpu = pj_cpu.new_empty(
                                                (int(n_i),) + tuple(pj_cpu.shape[1:])
                                            )
                                        preds_fix_cpu[j : j + 1].copy_(pj_cpu[:1])
                                    preds = preds_fix_cpu if preds_fix_cpu is not None else preds
                                    with contextlib.suppress(Exception):
                                        if int(preds.shape[0]) >= 2:
                                            y0b = preds[0].detach()
                                            y1b = preds[1].detach()
                                            is_broadcast_like2, bstats2 = _broadcast_like(
                                                y0b, y1b, atol=float(broadcast_atol)
                                            )
                                            ydiff2 = float(bstats2.get("y_max", float("nan")))
                                            if bool(is_broadcast_like2):
                                                with contextlib.suppress(Exception):
                                                    _diag_collapse_once(
                                                        Xi2=Xi[:2].detach(),
                                                        preds2=preds[:2].detach(),
                                                        x_diff=float(x_diff.item()) if "x_diff" in locals() else float("nan"),
                                                        y_diff=float(ydiff2),
                                                        where="after_per_sample_fix",
                                                    )
                                                _LOGGER.warning(
                                                    "[infer] outputs remain batch-broadcasted even after per-sample fallback; this indicates a model/preprocess collapse (not just cudagraph batching)."
                                                )
                                                _dump_collapse_stage_diag(
                                                    where="broadcast_trigger",
                                                    Xi2=Xi[:2].detach(),
                                                    seen_batches=int(seen_batches),
                                                    x_diff=float(x_diff.item()),
                                                    y_diff_cal=float(ydiff2),
                                                )
                                                if _maybe_enable_fp32_collapse_fallback():
                                                    preds_fix_fp32: torch.Tensor | None = None
                                                    for j in range(int(n_i)):
                                                        x1_buf.copy_(Xi[j : j + 1])
                                                        if cg_enabled:
                                                            cudagraph_mark_step_begin()
                                                        pj_fp32 = _td_predict(
                                                            x1_buf,
                                                            calibrate_output=True,
                                                            use_uncompiled=True,
                                                            force_fp32=True,
                                                        )
                                                        if cg_enabled:
                                                            cudagraph_mark_step_end()
                                                        pj_fp32_cpu = pj_fp32.detach()
                                                        if getattr(pj_fp32_cpu.device, "type", None) != "cpu":
                                                            pj_fp32_cpu = pj_fp32_cpu.to(device="cpu")
                                                        with contextlib.suppress(Exception):
                                                            pj_fp32_cpu = pj_fp32_cpu.contiguous()
                                                        pj_fp32_cpu = pj_fp32_cpu.clone()
                                                        if preds_fix_fp32 is None:
                                                            preds_fix_fp32 = pj_fp32_cpu.new_empty(
                                                                (int(n_i),) + tuple(pj_fp32_cpu.shape[1:])
                                                            )
                                                        preds_fix_fp32[j : j + 1].copy_(pj_fp32_cpu[:1])
                                                    if preds_fix_fp32 is not None and int(preds_fix_fp32.shape[0]) >= 2:
                                                        is_like_fp32, st_fp32 = _broadcast_like(
                                                            preds_fix_fp32[0],
                                                            preds_fix_fp32[1],
                                                            atol=float(broadcast_atol),
                                                        )
                                                        dy_fp32 = float(st_fp32.get("y_max", float("nan")))
                                                        _LOGGER.warning(
                                                            "[infer] fp32 fallback sanity (this batch): max|Y0-Y1|=%.6g match_frac=%.5f rel_mean=%.3e",
                                                            float(dy_fp32),
                                                            float(st_fp32.get("y_match_frac", float("nan"))),
                                                            float(st_fp32.get("y_rel_mean", float("nan"))),
                                                        )
                                                        if not bool(is_like_fp32):
                                                            preds = preds_fix_fp32
                                                            if not bool(collapse_force_fp32_persist):
                                                                collapse_fp32_active = False
                                                        else:
                                                            if bool(collapse_abort):
                                                                raise RuntimeError(
                                                                    "infer: collapse persisted even after fp32 upcast retry; "
                                                                    "this indicates a true model/preprocess collapse. "
                                                                    "Set ENN_PRED_COLLAPSE_ABORT=0 to ignore and write outputs anyway."
                                                                )
                                                if (
                                                    (not collapse_switched_uncompiled)
                                                    and bool(collapse_force_uncompiled)
                                                    and (not bool(force_uncompiled))
                                                    and (run_model_uncompiled is not run_model)
                                                ):
                                                    collapse_switched_uncompiled = True
                                                    _LOGGER.warning(
                                                        "[infer] collapse persisted; retrying this batch per-sample on uncompiled model (ENN_PRED_COLLAPSE_FORCE_UNCOMPILED=1)."
                                                    )
                                                    prev_force_uncompiled = bool(force_uncompiled)
                                                    force_uncompiled = True

                                                    preds_fix_uc: Optional[torch.Tensor] = None
                                                    for j in range(int(n_i)):
                                                        x1_buf.copy_(Xi[j : j + 1])
                                                        if dev_type == "cuda":
                                                            cudagraph_mark_step_begin()
                                                        pj_uc = _td_predict(x1_buf)
                                                        if dev_type == "cuda":
                                                            cudagraph_mark_step_end()
                                                        if getattr(pj_uc.device, "type", None) == "cuda":
                                                            sync_accelerator(dev_obj)
                                                        pj_uc_cpu = pj_uc.detach()
                                                        if getattr(pj_uc_cpu.device, "type", None) != "cpu":
                                                            pj_uc_cpu = pj_uc_cpu.to(device="cpu")
                                                        with contextlib.suppress(Exception):
                                                            pj_uc_cpu = pj_uc_cpu.contiguous()
                                                        pj_uc_cpu = pj_uc_cpu.clone()
                                                        if preds_fix_uc is None:
                                                            preds_fix_uc = pj_uc_cpu.new_empty(
                                                                (int(n_i),) + tuple(pj_uc_cpu.shape[1:])
                                                            )
                                                        preds_fix_uc[j : j + 1].copy_(pj_uc_cpu[:1])

                                                    if preds_fix_uc is not None:
                                                        preds = preds_fix_uc
                                                        with contextlib.suppress(Exception):
                                                            if int(preds.shape[0]) >= 2:
                                                                is_like_uc, st_uc = _broadcast_like(
                                                                    preds[0], preds[1], atol=float(broadcast_atol)
                                                                )
                                                                dy_uc = float(st_uc.get("y_max", float("nan")))
                                                                if not bool(is_like_uc):
                                                                    _LOGGER.warning(
                                                                        "[infer] uncompiled per-sample resolved collapse for this batch; keeping force_eager=1 and force_uncompiled=1."
                                                                    )
                                                                    force_eager = True
                                                                else:
                                                                    force_uncompiled = bool(prev_force_uncompiled)
                                                if (
                                                    (not collapse_switched_raw)
                                                    and bool(collapse_fallback_raw)
                                                    and bool(calibrate_pred_output)
                                                ):
                                                    collapse_switched_raw = True
                                                    _LOGGER.warning(
                                                        "[infer] collapse persisted; retrying this batch per-sample with calibrate_output=False (ENN_PRED_COLLAPSE_FALLBACK_RAW=1)."
                                                    )
                                                    preds_fix2: torch.Tensor | None = None
                                                    for j in range(int(n_i)):
                                                        x1_buf.copy_(Xi[j : j + 1])
                                                        if dev_type == "cuda":
                                                            cudagraph_mark_step_begin()
                                                        pj2 = _td_predict(x1_buf, calibrate_output=False)
                                                        if dev_type == "cuda":
                                                            cudagraph_mark_step_end()
                                                        pj2_cpu = pj2.detach()
                                                        if getattr(pj2_cpu.device, "type", None) != "cpu":
                                                            pj2_cpu = pj2_cpu.to(device="cpu")
                                                        with contextlib.suppress(Exception):
                                                            pj2_cpu = pj2_cpu.contiguous()
                                                        pj2_cpu = pj2_cpu.clone()
                                                        if preds_fix2 is None:
                                                            preds_fix2 = pj2_cpu.new_empty(
                                                                (int(n_i),) + tuple(pj2_cpu.shape[1:])
                                                            )
                                                        preds_fix2[j : j + 1].copy_(pj2_cpu[:1])
                                                    if preds_fix2 is not None and int(preds_fix2.shape[0]) >= 2:
                                                        is_like_raw, st_raw = _broadcast_like(
                                                            preds_fix2[0],
                                                            preds_fix2[1],
                                                            atol=float(broadcast_atol),
                                                        )
                                                        dy2 = float(st_raw.get("y_max", float("nan")))
                                                        _LOGGER.warning(
                                                            "[infer] calibrate_output=False sanity (this batch only): max|Y0-Y1|=%.6g match_frac=%.5f rel_mean=%.3e",
                                                            float(dy2),
                                                            float(st_raw.get("y_match_frac", float("nan"))),
                                                            float(st_raw.get("y_rel_mean", float("nan"))),
                                                        )
                                                        if bool(is_like_raw) and bool(collapse_abort):
                                                            raise _InferCollapseAbort(
                                                                "infer: collapse persisted even in calibrate_output=False path; "
                                                                "this indicates a true model/preprocess collapse. "
                                                                "Set ENN_PRED_COLLAPSE_ABORT=0 to ignore and write outputs anyway."
                                                            )
                        except Exception as exc:
                            if isinstance(exc, _InferCollapseAbort):
                                raise
                            pass

                    reuse_risk = bool(_pred_reuse_active(predict_fn))

                    rows_cpu = (
                        rows_i
                        if rows_i.device.type == "cpu"
                        else rows_i.to(device="cpu")
                    )
                    force_cpu_default = bool(use_async_write) and bool(dev_type in ("cuda", "xpu"))
                    force_cpu = env_bool("ENN_PRED_FORCE_CPU_COPY", default=force_cpu_default)
                    need_cpu_copy = bool(force_cpu) or bool(reuse_risk)
                    with contextlib.suppress(Exception):
                        rows_cpu = rows_cpu.clone()
                    if need_cpu_copy:
                        preds_cpu = preds.detach()
                        if getattr(preds_cpu, "device", None) is not None and preds_cpu.device.type != "cpu":
                            preds_cpu = preds_cpu.to(device="cpu")
                        if isinstance(preds_cpu, torch.Tensor) and (not preds_cpu.is_contiguous()):
                            preds_cpu = preds_cpu.contiguous()
                        with contextlib.suppress(Exception):
                            preds_cpu = preds_cpu.clone()
                        writer.append(rows_cpu, preds_cpu)
                    else:
                        if reuse_risk and getattr(preds.device, "type", None) == "cuda":
                            preds = preds.clone()
                        writer.append(rows_cpu, preds)
                    appended_rows_total += int(
                        getattr(preds, "shape", (0,))[0]
                    )
                    del Xi, Xi_pad, rows_i, out, preds, rows_cpu
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
                had_error = bool(
                    getattr(cache, "had_error", None) and cache.had_error()
                )
            if had_error:
                err = getattr(cache, "_err", None)
                if isinstance(err, BaseException):
                    raise RuntimeError(
                        f"infer: prediction writer encountered an error: {type(err).__name__}: {err}"
                    ) from err
                raise RuntimeError(
                    "infer: prediction writer encountered an error"
                )
        if status_bar is not None:
            status_bar.close()
        if exc_type is None:
            with contextlib.suppress(Exception):
                done_path = os.path.join(str(chunk_dir), f".rankdone.{int(rank):06d}")
                Path(done_path).write_text(f"ok{os.linesep}", encoding="utf-8")

        if exc_type is None and rank == 0:
            timeout_s = int(env_int("ENN_PRED_MANIFEST_WAIT_SEC", 600) or 600)
            strict = bool(env_bool("ENN_PRED_MANIFEST_STRICT", default=True))
            t0 = time.monotonic()
            expected = int(world_size)
            got = 0
            while True:
                got = 0
                for i in range(expected):
                    p = os.path.join(str(chunk_dir), f".rankdone.{i:06d}")
                    if os.path.exists(p):
                        got += 1
                if got >= expected:
                    break
                if time.monotonic() - t0 >= float(timeout_s):
                    msg = f"infer: manifest wait timeout ({timeout_s}s): got {got}/{expected} rankdone markers"
                    if strict:
                        raise RuntimeError(msg)
                    _LOGGER.warning(msg)
                    break
                time.sleep(0.2)
        if exc_type is None and rank == 0:
            parts = []
            for rows_path in sorted(
                glob.glob(os.path.join(chunk_dir, "part-r*-c*-rows.pt"))
            ):
                base = rows_path[: -len("-rows.pt")]
                pred_mmt = base + "-pred.mmt"
                pred_pt = base + "-pred.pt"
                if os.path.exists(pred_mmt):
                    pred_path = pred_mmt
                    meta_path = collate.get_meta_path(pred_mmt)
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
                    {
                        "rows": os.path.basename(rows_path),
                        "pred": os.path.basename(pred_path),
                    }
                )
            if not parts:
                diag = {
                    "rank": int(rank),
                    "world_size": int(world_size),
                    "seen_batches": int(seen_batches),
                    "none_batches": int(none_batches),
                    "empty_x_batches": int(empty_x_batches),
                    "appended_rows_total": int(appended_rows_total),
                    "loader_type": dl_type,
                    "loader_len": dl_len,
                    "first_batch": first_batch_info,
                }
                raise RuntimeError(
                    f"infer: no prediction parts produced in {chunk_dir}. diag={diag}"
                )
            manifest = {
                "format": "enn_torch.pred.v2",
                "rank_count": int(world_size),
                "out_shape": list((int(x) for x in ops.out_shape or ())),
                "variable_shape": bool(writer.variable_shape),
                "parts": parts,
            }
            man_path = os.path.join(chunk_dir, "manifest.json")
            collate.write_json(man_path, manifest, indent=2)
            with contextlib.suppress(Exception):
                for i in range(int(world_size)):
                    p = os.path.join(str(chunk_dir), f".rankdone.{i:06d}")
                    with contextlib.suppress(Exception):
                        if os.path.exists(p):
                            os.remove(p)
    return None


@worker_main()
def process(*args: Any, **kwargs: Any) -> object:
    import signal
    import faulthandler

    from ..data.pipeline import Session

    current_pid = os.getpid()
    print(f"PyTorch Elastic has been launched. (PID: {current_pid})", flush=True)
    path = os.path.join(tempfile.gettempdir(), f"pytrace.{os.getpid()}.log")
    _log = open(path, "a", buffering=1, encoding="utf-8")
    registered_signals: list[signal.Signals] = []
    faulthandler.enable(all_threads=True, file=_log)
    for name in ("SIGUSR1", "SIGUSR2", "SIGBREAK", "SIGQUIT"):
        sig = getattr(signal, name, None)
        try:
            if sig is not None:
                faulthandler.register(sig, all_threads=True, file=_log, chain=True)
                registered_signals.append(sig)
        except Exception:
            pass
    def _cleanup_on_success() -> None:
        for sig in registered_signals:
            with contextlib.suppress(Exception):
                faulthandler.unregister(sig)
        with contextlib.suppress(Exception):
            faulthandler.disable()
        with contextlib.suppress(Exception):
            _log.close()
        with contextlib.suppress(Exception):
            if os.path.exists(path):
                os.remove(path)

    if not args:
        raise TypeError("process requires at least a RuntimeConfig argument")
    _MA_SENTINEL = object()
    kw_model_averaging = kwargs.pop("model_averaging", _MA_SENTINEL)
    kw_ret_sink = kwargs.pop("ret_sink", _MA_SENTINEL)
    if kwargs:
        raise TypeError(
            f"process got unexpected keyword arguments: {', '.join(sorted(kwargs))}"
        )
    init_python_path()
    _validate_compile_safe()
    ret_sink: ReturnSink | None = None
    if kw_ret_sink is not _MA_SENTINEL:
        if not isinstance(kw_ret_sink, MutableMapping):
            raise TypeError(
                f"process ret_sink must be a mutable mapping, got {type(kw_ret_sink).__name__}"
            )
        ret_sink = kw_ret_sink
    pos_model_averaging = _MA_SENTINEL
    tail: tuple[Any, ...]
    if isinstance(args[0], RuntimeConfig):
        ops = args[0]
        local_rank = env_int("LOCAL_RANK", 0)
        tail = tuple(args[1:])
    elif len(args) >= 2 and isinstance(args[1], RuntimeConfig):
        local_rank = int(args[0])
        ops = args[1]
        tail = tuple(args[2:])
    else:
        raise TypeError(
            "process expects (RuntimeConfig,), (RuntimeConfig, ret_sink), (local_rank, RuntimeConfig), or (local_rank, RuntimeConfig, ret_sink) arguments"
        )
    extras: list[Any] = []
    for v in tail:
        if isinstance(v, MutableMapping):
            if ret_sink is None:
                ret_sink = v
                continue
            extras.append(v)
            continue
        if pos_model_averaging is _MA_SENTINEL and (v is None or isinstance(v, str)):
            pos_model_averaging = v
            continue
        extras.append(v)
    if extras:
        raise TypeError(
            "process got unexpected positional arguments: "
            + ", ".join(type(x).__name__ for x in extras)
        )

    ops_model_averaging = getattr(ops, "model_averaging", _MA_SENTINEL)
    if kw_model_averaging is not _MA_SENTINEL:
        model_averaging = kw_model_averaging
        model_averaging_provided = True
    elif pos_model_averaging is not _MA_SENTINEL:
        model_averaging = pos_model_averaging
        model_averaging_provided = True
    elif ops_model_averaging is not _MA_SENTINEL:
        model_averaging = ops_model_averaging
        model_averaging_provided = True
    else:
        model_averaging = "auto"
        model_averaging_provided = False
    use_env = False
    try:
        if not model_averaging_provided:
            use_env = True
        else:
            use_env = (
                _normalize_model_averaging(model_averaging, default="auto") == "auto"
            )
    except Exception:
        use_env = False
    if use_env:
        env_ma = os.environ.get("ENN_MODEL_AVERAGING", None)
        if env_ma is not None:
            model_averaging = env_ma
    try:
        if int(local_rank) == 0 and getattr(ops, "ckpt_dir", None):
            _mark_ephemeral_ckpt_dir(str(ops.ckpt_dir))
    except Exception:
        pass
    verbose = bool(getattr(ops, "verbose", False))
    det = bool(getattr(ops, "deterministic", False))
    seed_value = int(getattr(ops, "seed", 42)) + int(local_rank)
    ProcessBroker.apply_warning_filters()
    ProcessBroker.set_seed(seed_value)
    with contextlib.suppress(Exception):
        torch.use_deterministic_algorithms(det, warn_only=False)
    with contextlib.suppress(Exception):
        torch.backends.cudnn.deterministic = det
        torch.backends.cudnn.benchmark = not det
    strict_cache = env_bool(
        ("ENN_PRED_UNIQUE_INDUCTOR_CACHE_STRICT", "ENN_UNIQUE_INDUCTOR_CACHE_STRICT"),
        default=env_bool("ENN_SANITIZE_NAN_STRICT", default=False),
    )
    try:
        if ops.mode != "train" and env_bool(
            ("ENN_PRED_UNIQUE_INDUCTOR_CACHE", "ENN_UNIQUE_INDUCTOR_CACHE"),
            default=True,
        ):
            root = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
            if not root:
                root = env_str("ENN_PRED_INDUCTOR_CACHE_ROOT")
            if isinstance(root, str):
                root = root.strip()
                if "\n" in root or "\r" in root:
                    root = root.splitlines()[0].strip()
            if not root:
                root = os.path.join(tempfile.gettempdir(), "torchinductor_enn")
            os.makedirs(root, exist_ok=True)
            rid = os.environ.get("ENN_RUN_ID") or os.urandom(4).hex()
            rank = os.environ.get("RANK") or str(local_rank)
            cache_dir = os.path.join(
                root,
                f"pred_{rid}_pid{os.getpid()}_rank{rank}",
            )
            os.makedirs(cache_dir, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
            _sync_torchinductor_cache_globals(cache_dir)
            if verbose or env_bool("ENN_LOG_INDUCTOR_CACHE_DIR", default=True):
                print(f"[ENN] TORCHINDUCTOR_CACHE_DIR={cache_dir}", flush=True)
    except Exception as e:
        if strict_cache:
            raise
        if verbose or env_bool("ENN_LOG_INDUCTOR_CACHE_DIR_ERRORS", default=True):
            print(
                f"[ENN] WARNING: failed to set unique TORCHINDUCTOR_CACHE_DIR: {e!r}",
                flush=True,
            )
    if ops.mode == "train":
        resolved_ma: str | None = None
        has_bn = False
        device = get_device()
        ProcessBroker.init_backend(device, local_rank=int(local_rank))
        backend = ProcessBroker.get_backend_type(device)
        ProcessBroker.configure_backend_env(backend, device)
        enable_tf32 = bool(getattr(ops, "enable_tf32", True))
        ProcessBroker.init_process_group(
            backend, device, local_rank=int(local_rank)
        )
        cfg = coerce_model_config(
            ops.cfg_dict if isinstance(ops.cfg_dict, dict) else ops.cfg_dict
        )
        cfg = replace(cfg, device=device)
        model: Model
        if ops.init_ckpt_dir is not None and os.path.exists(ops.init_ckpt_dir):
            from .workflows import load_model as _api_load_model

            init_path = str(ops.init_ckpt_dir)
            if os.path.isdir(init_path):
                for _name in ("model.pt", "model.pth", "model.safetensors"):
                    fp = os.path.join(init_path, _name)
                    if os.path.isfile(fp):
                        init_path = fp
                        break

            model = _api_load_model(
                init_path,
                in_dim=ops.in_dim,
                out_shape=ops.out_shape,
                config=cfg,
                map_location="cpu",
            )
        else:
            model = Model(ops.in_dim, ops.out_shape, config=cfg)
        if ops.sources is None:
            raise RuntimeError("RuntimeConfig.sources is required but None")
        metadata = Dataset.for_device(device)
        expanded_sources = collate.expand_source(ops.sources)
        if expanded_sources is not ops.sources:
            ops = replace(ops, sources=expanded_sources)
        meta_info = collate.merge_meta_info(ops.sources)
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
            if not math.isclose(
                actual_val_frac,
                float(ops.val_frac),
                rel_tol=0.001,
                abs_tol=0.001,
            ):
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
        metadata.scale_min_positive = meta_info.get(
            "scale_min_positive"
        ) or meta_info.get("scale_min_abs")
        metadata.scale_is_integral = meta_info.get("scale_is_integral")
        if meta_info.get("is_negotiable") is not None:
            metadata.is_negotiable = bool(meta_info.get("is_negotiable"))
        if meta_info.get("underflow_action") is not None:
            metadata.underflow_action = str(meta_info.get("underflow_action"))
        feat_dtype_name = str(meta_info.get("features_dtype", "")).lower()
        lab_dtype_name = str(meta_info.get("labels_dtype", "")).lower()
        if "float64" in feat_dtype_name or "float64" in lab_dtype_name:
            metadata.is_negotiable = False
        precision = PrecisionPolicy.from_metadata(
            device=device, metadata=metadata, logger=_LOGGER
        )
        param_dtype = precision.master_float
        model, _, _ = ModelPolicy.use_nvidia_layers(
            model,
            device=device,
            metadata=metadata,
            params_dtype=param_dtype,
            verbose=verbose,
        )
        StatelessAutocast.configure(model, metadata=metadata)
        set_float32_precision(
            device=device,
            autocast_dtype=precision.amp_float or param_dtype,
            enable_tf32=enable_tf32,
        )
        fp8_quantize = env_bool(("ENN_MODEL_ENABLE_FP8_QUANTIZE", "ENN_ENABLE_FP8_QUANTIZE"), default=False)
        fp8_enabled = False
        fp8_backend = None
        disable_note = None
        if fp8_quantize:
            fp8_ok, fp8_reason = Dataset.is_float8_supported(device)
            if param_dtype is torch.float64:
                disable_note = "master dtype is float64"
            elif fp8_ok:
                model, fp8_enabled, fp8_backend = (
                    ModelPolicy.quantize_for_float8_training(
                        model, metadata=metadata, logger=_float8_log
                    )
                )
                if not fp8_enabled:
                    disable_note = fp8_backend or fp8_reason
            else:
                disable_note = fp8_reason
            if disable_note:
                _float8_log(f"[FP8][quantize] disabled: {disable_note}")
        else:
            _float8_log(
                "[FP8][quantize] skipped (default AMP-only). "
                "Set ENN_MODEL_ENABLE_FP8_QUANTIZE=1 to run model quantization."
            )
        model.train()
        fsdp_mp_dtype = precision.fsdp_reduce_dtype
        if device.type == "cpu" and fsdp_mp_dtype is not torch.float64:
            fsdp_mp_dtype = torch.float32
        amp_buffers_dtype = precision.bn_buffers_dtype
        mp_policy = None
        if MixedPrecisionPolicy is not None:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=param_dtype,
                output_dtype=param_dtype,
                cast_forward_inputs=False,
            )
        elif verbose:
            _LOGGER.warning(
                "MixedPrecisionPolicy is not available in this PyTorch build; "
                "continuing without explicit FSDP mixed-precision policy."
            )
        _m_pre = model.module if hasattr(model, "module") else model
        _preload_layers(_m_pre, device)
        _cast_float_dtype(model, param_dtype)
        _validate_model_dtype_unity(_m_pre, device)
        _validate_no_meta_tensors(_m_pre)
        _validate_no_fake_dtensor(_m_pre)
        dist_policy = DistributedPolicy.from_env()
        hsdp_wrapped = False
        if (
            is_distributed()
            and get_world_size(device) > 1
            and dist_policy.prefer_hsdp
            and device.type in ("cuda", "xpu")
        ):
            mesh, mesh_kind = get_distributed_mesh(device)
            if mesh_kind != "hsdp2":
                _LOGGER.warning(
                    f"HSDP2 mesh not available (e.g. heterogeneous nodes). Using fallback: {mesh_kind}"
                )
            try:
                wrapped_ids: set[int] = set()

                def _wrap_once(
                    mod: torch.nn.Module = None, *args: Any, is_root: bool
                ) -> None:
                    if mod is None or id(mod) in wrapped_ids:
                        return
                    wrapped_ids.add(id(mod))
                    to_hsdp_module(
                        mod,
                        mesh=mesh,
                        mp_policy=mp_policy,
                        reshard_after_forward=(not is_root),
                        sync_module_states=bool(is_root),
                    )

                for root_mod in (
                    getattr(model, "processor", None),
                    getattr(model, "controller", None),
                ):
                    if isinstance(root_mod, torch.nn.Module):
                        blocks = getattr(root_mod, "blocks", None)
                        if isinstance(blocks, torch.nn.ModuleList):
                            for blk in blocks:
                                if isinstance(blk, torch.nn.Module):
                                    _wrap_once(blk, is_root=False)
                        _wrap_once(root_mod, is_root=False)
                model = to_hsdp_module(
                    model,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=False,
                    sync_module_states=True,
                )
            except Exception as e:
                _LOGGER.warning(
                    f"HSDP2 wrapping fallback to root-only FSDP: {e}"
                )
                model = to_hsdp_module(
                    model,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=False,
                    sync_module_states=True,
                )
            hsdp_wrapped = True
        _m_post = model.module if hasattr(model, "module") else model
        _validate_model_dtype_unity(_m_post, device)
        _validate_no_meta_tensors(_m_post)
        _validate_no_fake_dtensor(_m_post)
        _enable_meta_monitor(_m_post)
        ddp_fallback = bool(
            is_distributed()
            and get_world_size(device) > 1
            and (not hsdp_wrapped)
            and dist_policy.prefer_ddp
        )
        if dist_policy.sync_state:
            distributed_sync(
                _m_post, device=device, policy=dist_policy.collective
            )
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
            coefficient=[0.5, 0.5],
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
            coefficient=[0.5, 0.5],
            loss=[local_crps, local_t],
            reduce_each=False,
            auto_schedule=True,
        )
        loss_controller = LossWeightController(top_avg=0.5, bottom_avg=0.5)
        ckpt_state_path = ProcessBroker.get_loader_state(ops.ckpt_dir or "")
        init_state_path = (
            ProcessBroker.get_loader_state(ops.init_ckpt_dir)
            if ops.init_ckpt_dir
            else None
        )
        state_train = {}
        state_val = {}
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
            expanded_sources = collate.expand_source(ops.sources)
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
            os.environ.setdefault("ENN_MICROBATCH_MAX", "64")
            os.environ.setdefault("ENN_MICROBATCH_STAGE_DIV", "4")
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

            calibration_loader = None
            try:
                if raw_val_loader is not None and int(_get_batch_length(raw_val_loader)) > 0:
                    calibration_loader = raw_val_loader
            except Exception:
                calibration_loader = None
            if calibration_loader is None:
                try:
                    if raw_train_loader is not None and int(_get_batch_length(raw_train_loader)) > 0:
                        calibration_loader = raw_train_loader
                except Exception:
                    calibration_loader = None
            if calibration_loader is None:
                try:
                    if val_loader is not None and int(_get_batch_length(val_loader)) > 0:
                        calibration_loader = val_loader
                except Exception:
                    calibration_loader = None
            if calibration_loader is None:
                calibration_loader = train_loader

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
            sched = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda
            )
            scheduler_step_per_batch = True
            tracked_module = (
                model.module if hasattr(model, "module") else model
            )
            ema_helper = None
            swa_helper = None
            swa_start_epoch = int(total_epochs)
            resolved_ma, has_bn = _resolve_model_averaging(
                tracked_module, model_averaging
            )
            use_swa = resolved_ma == "swa"
            if local_rank == 0:
                if use_swa:
                    swa_start_epoch = 0
                    _swa_env = os.environ.get("ENN_SWA_START_EPOCH")
                    if _swa_env is not None and str(_swa_env).strip() != "":
                        try:
                            swa_start_epoch = max(
                                0, int(str(_swa_env).strip())
                            )
                        except Exception:
                            swa_start_epoch = 0
                    swa_helper = StochasticWeightAverage(
                        tracked_module, metadata=metadata
                    )
                elif resolved_ma == "ema":
                    ema_helper = ExponentialMovingAverage(
                        tracked_module, decay=0.9999, metadata=metadata
                    )
            amp_dtype = getattr(precision, "amp_float", None)
            compute_dtype = amp_dtype or param_dtype
            scaler = torch.amp.GradScaler(
                enabled=bool(
                    device.type == "cuda" and compute_dtype == torch.float16
                )
            )
            checkpointer: Checkpointer | None = None
            checkpoint_enabled = bool(getattr(ops, "checkpoint", True))
            if ops.ckpt_dir and checkpoint_enabled:
                keep_last = max(
                    1,
                    int(
                        env_first_int(
                            ("ENN_DCP_KEEP_LAST", "ENN_CKPT_KEEP_LAST"),
                            default=1,
                        )
                    ),
                )
                use_async = bool(
                    env_bool(("ENN_DCP_ASYNC", "ENN_CKPT_ASYNC"), default=True)
                )
                mmap_load = (
                    bool(env_bool("ENN_DCP_MMAP_LOAD", default=False))
                    if "ENN_DCP_MMAP_LOAD" in os.environ
                    else None
                )
                checkpointer = Checkpointer(
                    ops.ckpt_dir,
                    keep_last=keep_last,
                    use_async=use_async,
                    mmap_load=mmap_load,
                    device=device,
                    cpu_offload=getattr(ops, "ckpt_cpu_offload", None),
                )
            try:
                new_affinity().pin_thread()
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
                calibration_loader=calibration_loader,
                total_epochs=total_epochs,
                scheduler_step_per_batch=scheduler_step_per_batch,
                swa_helper=swa_helper,
                ema_helper=ema_helper,
                swa_start_epoch=swa_start_epoch,
                checkpointer=checkpointer,
                buffers_dtype=amp_buffers_dtype,
                dataset=metadata,
                ddp_fallback=ddp_fallback,
            )
        finally:
            if session is not None:
                session.close()
        with contextlib.suppress(Exception):
            if checkpointer is not None:
                checkpointer.close(abort_inflight=True)
                checkpointer = None
        with contextlib.suppress(Exception):
            import gc

            gc.collect()
        with contextlib.suppress(Exception):
            if env_bool("ENN_FINALIZE_MALLOC_TRIM", default=False):
                import ctypes
                import platform as _platform

                sysname = _platform.system()
                if sysname == "Linux":
                    libc = ctypes.CDLL("libc.so.6")
                    trim = getattr(libc, "malloc_trim", None)
                    if callable(trim):
                        trim(0)
                elif sysname == "Windows":
                    msvcrt = ctypes.CDLL("msvcrt.dll")
                    heapmin = getattr(msvcrt, "_heapmin", None)
                    if callable(heapmin):
                        heapmin()
                elif sysname == "Darwin":
                    libsys = ctypes.CDLL("libsystem_malloc.dylib")
                    fn = getattr(libsys, "malloc_zone_pressure_relief", None)
                    if callable(fn):
                        fn(None, 0)
        with contextlib.suppress(Exception):
            time.sleep(float(os.environ.get("ENN_FINALIZE_YIELD_S", "0.02") or 0.02))
        with contextlib.suppress(Exception):
            if int(local_rank) == 0 and getattr(ops, "ckpt_dir", None):
                target = model.module if hasattr(model, "module") else model
                _force_final_avg_update(
                    ema_helper, swa_helper, target, optimizer
                )
                if (
                    swa_helper is not None
                    and has_bn
                    and _env_bool("ENN_SWA_UPDATE_BN", default=True)
                ):
                    try:
                        dev = None
                        try:
                            p0 = next(
                                (
                                    p
                                    for p in target.parameters()
                                    if torch.is_tensor(p)
                                ),
                                None,
                            )
                            if torch.is_tensor(p0):
                                dev = p0.device
                        except Exception:
                            dev = None
                        fn = getattr(
                            swa_helper, "apply_and_update_batch_norm", None
                        )
                        if callable(fn):
                            fn(train_loader, model=target, device=dev)
                    except Exception:
                        pass
                _export_return_model_pt(
                    target,
                    str(ops.ckpt_dir),
                    ema_helper=ema_helper,
                    swa_helper=swa_helper,
                    model_averaging=resolved_ma,
                )
                try:
                    if swa_helper is not None and hasattr(swa_helper, "close"):
                        swa_helper.close()
                except Exception:
                    pass
                gc.collect()
        with contextlib.suppress(Exception):
            if swa_helper is not None and hasattr(swa_helper, "close"):
                swa_helper.close()
        torch.distributed.barrier(
            device_ids=[local_rank] if device.type in ("cuda", "xpu") else None
        )
        torch.distributed.destroy_process_group()
        _cleanup_on_success()
        return None
    if ops.mode in ("predict", "infer"):
        device = get_device()
        ProcessBroker.init_backend(device, local_rank=int(local_rank))
        backend = ProcessBroker.get_backend_type(device)
        ProcessBroker.configure_backend_env(backend, device)
        if not torch.distributed.is_initialized():
            ProcessBroker.init_process_group(
                backend, device, local_rank=int(local_rank)
            )
        cfg = coerce_model_config(
            ops.cfg_dict if isinstance(ops.cfg_dict, dict) else ops.cfg_dict
        )
        cfg = replace(cfg, device=device)
        if env_bool("ENN_MAX_PERF", True):
            cm = canonicalize_compile_mode(getattr(cfg, "compile_mode", None))
            if (
                cm == "reduce-overhead"
                and getattr(cfg, "compile_dynamic", None) is None
            ):
                with contextlib.suppress(Exception):
                    setattr(cfg, "compile_dynamic", False)
            if "ENN_COMPILE_DYNAMIC" in os.environ:
                with contextlib.suppress(Exception):
                    setattr(
                        cfg,
                        "compile_dynamic",
                        bool(env_bool("ENN_COMPILE_DYNAMIC", False)),
                    )
            if cm == "disabled":
                default_cm = env_str("ENN_SERVE_COMPILE_MODE")
                if not default_cm:
                    default_cm = (
                        "max-autotune"
                        if getattr(device, "type", None) == "cuda"
                        else "max-autotune-no-cudagraphs"
                    )
                cfg = replace(cfg, compile_mode=str(default_cm))
                with contextlib.suppress(Exception):
                    setattr(
                        cfg,
                        "compile_dynamic",
                        bool(env_bool("ENN_COMPILE_DYNAMIC", False)),
                    )
                with contextlib.suppress(Exception):
                    setattr(
                        cfg,
                        "compile_cudagraphs",
                        bool(env_bool("ENN_COMPILE_CUDAGRAPHS", True)),
                    )
        if not ops.model_ckpt_dir:
            raise RuntimeError(
                "predict/infer requires model_ckpt_dir (checkpoint directory). Set RuntimeConfig.model_ckpt_dir to a directory produced by train()."
            )
        if not os.path.isdir(ops.model_ckpt_dir):
            raise RuntimeError(
                f"predict/infer: model_ckpt_dir does not exist or is not a directory: {ops.model_ckpt_dir!r}"
            )

        from .workflows import load_model as _api_load_model

        ckpt_source: str = str(ops.model_ckpt_dir)
        for _name in ("model.pt", "model.pth", "model.safetensors"):
            fp = os.path.join(ckpt_source, _name)
            if os.path.isfile(fp):
                ckpt_source = fp
                break
        model = _api_load_model(
            ckpt_source,
            in_dim=ops.in_dim,
            out_shape=ops.out_shape,
            config=cfg,
            map_location="cpu",
        )
        model.to(device, non_blocking=device.type in ("cuda", "xpu")).eval()
        if ops.sources is None:
            raise RuntimeError("RuntimeConfig.sources is required but None")
        expanded_sources = collate.expand_source(ops.sources)
        if expanded_sources is not ops.sources:
            ops = replace(ops, sources=expanded_sources)
        metadata = Dataset.for_device(device)
        meta_info = collate.merge_meta_info(ops.sources)
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
        metadata.scale_min_positive = meta_info.get(
            "scale_min_positive"
        ) or meta_info.get("scale_min_abs")
        metadata.scale_is_integral = meta_info.get("scale_is_integral")
        if meta_info.get("is_negotiable") is not None:
            metadata.is_negotiable = bool(meta_info.get("is_negotiable"))
        if meta_info.get("underflow_action") is not None:
            metadata.underflow_action = str(meta_info.get("underflow_action"))
        feat_dtype_name = str(meta_info.get("features_dtype", "")).lower()
        lab_dtype_name = str(meta_info.get("labels_dtype", "")).lower()
        if "float64" in feat_dtype_name or "float64" in lab_dtype_name:
            metadata.is_negotiable = False
        precision = PrecisionPolicy.from_metadata(
            device=device, metadata=metadata, logger=_LOGGER
        )
        param_dtype = precision.master_float
        model, _, _ = ModelPolicy.use_nvidia_layers(
            model,
            device=device,
            metadata=metadata,
            params_dtype=param_dtype,
        )
        _m_eval = model.module if hasattr(model, "module") else model
        _preload_layers(_m_eval, device)
        _cast_float_dtype(model, param_dtype)
        _validate_model_dtype_unity(_m_eval, device)
        _validate_no_meta_tensors(_m_eval)
        _validate_no_fake_dtensor(_m_eval)
        _enable_meta_monitor(_m_eval)
        StatelessAutocast.configure(model, metadata=metadata)
        enable_tf32 = bool(getattr(ops, "enable_tf32", True))
        with contextlib.suppress(Exception):
            set_float32_precision(
                device=device,
                dtype=param_dtype,
                autocast_dtype=precision.amp_float or param_dtype,
                enable_tf32=enable_tf32,
            )
        fp8_quantize = env_bool(("ENN_MODEL_ENABLE_FP8_QUANTIZE", "ENN_ENABLE_FP8_QUANTIZE"), default=False)
        fp8_enabled = False
        fp8_backend = None
        disable_note = None
        if fp8_quantize:
            fp8_infer_ok, fp8_infer_reason = Dataset.is_float8_supported(device)
            if fp8_infer_ok:
                model, fp8_enabled, fp8_backend = ModelPolicy.quantize_for_float8_prediction(
                    model, metadata=metadata, logger=_float8_log
                )
                if not fp8_enabled:
                    disable_note = fp8_backend or fp8_infer_reason
            else:
                disable_note = fp8_infer_reason
            if disable_note:
                _float8_log(f"[FP8][quantize] disabled: {disable_note}")
        else:
            _float8_log(
                "[FP8][quantize] skipped (default AMP-only). "
                "Set ENN_MODEL_ENABLE_FP8_QUANTIZE=1 to run model quantization."
            )
        model.eval()
        with contextlib.suppress(Exception):
            _get_sample_size(
                model=model,
                device=device,
                ops=ops,
                dataset=metadata,
                with_backward=False,
            )
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
        chunk_dir = (
            os.path.join(ops.ckpt_dir, "pred_chunks")
            if ops.ckpt_dir or ""
            else None
        )
        if chunk_dir and torch.distributed.get_rank() == 0:
            with contextlib.suppress(Exception):
                os.makedirs(chunk_dir, exist_ok=True)
        if torch.distributed.is_initialized():
            pass
        if ops.mode in ("predict", "infer"):
            if not chunk_dir:
                raise RuntimeError(
                    "predict/infer requires chunk_dir (streaming enforced)"
                )
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
        distributed_barrier(device, group=get_accel_group(device), lane="auto")
        torch.distributed.destroy_process_group()
        _cleanup_on_success()
        return None
    raise ValueError(f"unsupported ops mode: {ops.mode}")


compile_distributed_safe()
