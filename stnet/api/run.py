# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import json
import math
import os
import random
import shutil
import warnings
import numpy as np
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, Sequence, Mapping

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, load, save
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

try:
    from torch.distributed.run import LaunchConfig, elastic_launch
except ImportError:
    from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from ..data.nodes import SampleReader
from ..data.transforms import preprocess, set_scaler, get_scaler, drop_scaler
from ..model import Root
from .config import (
    ModelConfig,
    OpsMode,
    RuntimeConfig,
    coerce_model_config,
    runtime_config,
)
from ..backend.distributed import (
    get_available_host,
    get_preferred_ip,
    initialize_master_addr,
)
from ..backend.environment import (
    initialize_python_path,
    new_dir,
    optimize_threads,
    optimal_procs,
    optimal_start_method,
    set_multiprocessing_env,
)
from ..backend.runtime import _trim_dcp_keys, ignored_pattern, main


_DTENSOR_TYPE = getattr(getattr(torch.distributed, "_tensor", None), "DTensor", None)


def _coerce_buffer(
    model: Root,
    name: str,
    value: torch.Tensor | float | int,
    *args: Any,
    persistent: bool = True,
    ref: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    with torch.no_grad():
        if ref is None:
            ref = next(model.parameters(), None)
            if ref is None:
                ref = next(model.buffers(), None)
        ref_dtype = ref.dtype if isinstance(ref, torch.Tensor) else torch.float32
        ref_device = ref.device if isinstance(ref, torch.Tensor) else torch.device("cpu")

        new_t = torch.as_tensor(value, dtype=ref_dtype, device=ref_device)

        if hasattr(model, name) and isinstance(getattr(model, name), torch.Tensor):
            buf = getattr(model, name)
            if tuple(buf.shape) != tuple(new_t.shape):
                try:
                    delattr(model, name)
                except Exception:
                    pass
                model.register_buffer(name, new_t, persistent=persistent)
                return getattr(model, name)
            if buf.dtype != ref_dtype or buf.device != ref_device:
                buf.data = buf.data.to(dtype=ref_dtype, device=ref_device)
            buf.copy_(new_t)
            return buf
        else:
            model.register_buffer(name, new_t, persistent=persistent)
            return getattr(model, name)


def _attach_buffer(
    model: Root,
    /,
    *args: Any,
    persistent: bool = True,
    ref: torch.Tensor | None = None,
    **named_values: torch.Tensor | float | int,
) -> None:
    for k, v in named_values.items():
        _coerce_buffer(model, k, v, persistent=persistent, ref=ref)


def _attach_scaler(
    model: Root, mean: torch.Tensor | float, std: torch.Tensor | float
) -> None:
    try:
        ref = torch.empty((), dtype=torch.float64, device="cpu")
        _attach_buffer(
            model,
            target_mean=torch.as_tensor(mean, dtype=torch.float64, device="cpu"),
            target_std=torch.as_tensor(std, dtype=torch.float64, device="cpu"),
            ref=ref,
        )
    except Exception:
        pass


def _preload_state(value: Any) -> Any:
    if _DTENSOR_TYPE is not None and isinstance(value, _DTENSOR_TYPE):
        return value.to_local()
    if isinstance(value, dict):
        return {k: _preload_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_preload_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_preload_state(v) for v in value)
    return value


def train(
    model: Root,
    data: Dict[Tuple, torch.Tensor]
    | Sequence[Dict[Tuple, torch.Tensor]]
    | Mapping[str, Dict[Tuple, torch.Tensor]],
    *args: Any,
    epochs: int = 5,
    batch_size: int = 128,
    val_frac: float = 0.1,
    shuffle: bool = False,
    base_lr: float = 0.001,
    weight_decay: float = 0.0001,
    warmup_ratio: float = 0.0,
    eta_min: float = 0.0,
    run_id: str = "torch",
    seed: int = 42,
    max_nodes: int = 1,
    rdzv_backend: Optional[str] = "c10d",
    rdzv_endpoint: Optional[str] = None,
    prefetch_factor: Optional[int] = 1,
    grad_accum_steps: int = 1,
    overlap_h2d: bool = True,
    loss_tile_dim: Optional[int] = None,
    loss_tile_size: Optional[int] = None,
    loss_mask_mode: str = "none",
    loss_mask_value: Optional[float] = None,
    target_scaler: str = "standard",
    robust_q: Tuple[float, float] = (25.0, 75.0),
    robust_cap: int = 200_000,
    scale_non_floating: bool = False,
    **kwargs: Any,
) -> Root:
    try:
        val_frac = float(val_frac)
        val_frac = 0.0 if val_frac < 0.0 else (1.0 if val_frac > 1.0 else val_frac)
    except Exception:
        val_frac = 0.1

    try:
        seed_value = int(seed)
    except Exception:
        seed_value = None

    if seed_value is not None:
        try:
            torch.manual_seed(seed_value)
        except Exception:
            pass
        try:
            torch.cuda.manual_seed_all(seed_value)
        except Exception:
            pass
        try:
            random.seed(seed_value)
        except Exception:
            pass
        try:
            np.random.seed(seed_value)
        except Exception:
            pass
    with contextlib.suppress(Exception):
        torch.use_deterministic_algorithms(True, warn_only=True)
    with contextlib.suppress(Exception):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    drop_scaler()
    robust_cap = max(1, int(robust_cap))

    mode = str(target_scaler).lower().strip()
    use_none = mode == "none"
    use_robust = mode == "robust"
    q_lo = float(robust_q[0]) / 100.0
    q_hi = float(robust_q[1]) / 100.0
    q_lo = max(0.0, min(q_lo, 1.0))
    q_hi = max(0.0, min(q_hi, 1.0))
    if q_hi <= q_lo:
        q_hi = min(1.0, q_lo + 1e-6)

    def _coerce_scaler(
        labels: torch.Tensor, *, fit_count: int | None = None
    ) -> torch.Tensor:
        scaler = get_scaler()
        if scaler is not None:
            mean_t = torch.as_tensor(
                scaler["mean"], dtype=torch.float64, device=labels.device
            )
            std_t = torch.as_tensor(
                scaler["std"], dtype=torch.float64, device=labels.device
            )
            std_t = torch.clamp(std_t, min=1e-6)
            scaled = (labels.to(torch.float64) - mean_t) / std_t
            if torch.is_floating_point(labels):
                scaled = scaled.to(labels.dtype)
            return scaled
        with torch.no_grad():
            safe = labels.detach().to(torch.float64)
            if use_none:
                return labels
            if safe.numel() == 0:
                return labels
            fit = (
                safe[:fit_count]
                if isinstance(fit_count, int)
                and fit_count > 0
                and fit_count <= int(safe.shape[0])
                else safe
            )
            fit = torch.where(torch.isfinite(fit), fit, torch.nan)
            if fit.dim() == 1:
                mean_t = torch.nanmean(fit)
                std_t = torch.nanstd(fit, unbiased=False)
            else:
                mean_t = torch.nanmean(fit, dim=0)
                std_t = torch.nanstd(fit, dim=0, unbiased=False)
            std_t = torch.nan_to_num(std_t, nan=0.0)
            std_t = torch.clamp(std_t, min=1e-6)
        set_scaler(mean=mean_t.cpu(), std=std_t.cpu())
        _attach_scaler(model, mean_t, std_t)
        scaled = (labels.to(torch.float64) - mean_t) / std_t
        if torch.is_floating_point(labels):
            scaled = scaled.to(labels.dtype)
        return scaled

    def _mat_one(d: Any, out_dir: str) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        fx, lb, _, lshape = preprocess(d)
        if (not torch.is_floating_point(lb)) and (not bool(scale_non_floating)):
            SampleReader.preload(
                {"features": fx, "labels": lb},
                memmap_dir=out_dir,
                train_frac=1.0 - float(val_frac),
                val_frac=float(val_frac),
                shuffle=bool(shuffle),
                seed=seed_value,
                target_scaler=mode,
                robust_q=tuple(robust_q),
                robust_cap=int(robust_cap),
                scale_non_floating=bool(scale_non_floating),
            )
            return fx, tuple(lshape)
        did_manual_shuffle = False
        if shuffle:
            n_total_ps = int(lb.shape[0]) if hasattr(lb, "shape") and lb.ndim > 0 else 0
            if n_total_ps > 0:
                g = torch.Generator(device="cpu")
                if seed_value is not None:
                    g.manual_seed(seed_value)
                perm = torch.randperm(n_total_ps, generator=g)
                fx = fx.index_select(0, perm)
                lb = lb.index_select(0, perm)
                did_manual_shuffle = True
        n_total = int(lb.shape[0]) if hasattr(lb, "shape") and lb.ndim > 0 else 0
        val_count = int(round(n_total * float(val_frac)))
        n_train = n_total - val_count
        if (not use_none) and (get_scaler() is None):
            lb = _coerce_scaler(lb, fit_count=n_train if n_train > 0 else None)
        shuffle_for_preload = bool(shuffle and not did_manual_shuffle)
        SampleReader.preload(
            {"features": fx, "labels": lb},
            memmap_dir=out_dir,
            train_frac=1.0 - float(val_frac),
            val_frac=float(val_frac),
            shuffle=shuffle_for_preload,
            seed=seed_value,
            target_scaler=mode,
            robust_q=tuple(robust_q),
            robust_cap=int(robust_cap),
            scale_non_floating=bool(scale_non_floating),
        )
        return fx, tuple(lshape)

    initialize_python_path()
    mp.allow_connection_pickling()
    set_multiprocessing_env()

    def _iter_data(dsrc: Any):
        if isinstance(dsrc, Mapping) and dsrc and all(
            isinstance(v, Mapping) for v in dsrc.values()
        ):
            for _d in dsrc.values():
                yield _d
        elif isinstance(dsrc, Sequence) and dsrc and all(
            isinstance(_d, Mapping) for _d in dsrc
        ):
            for _d in dsrc:
                yield _d
        else:
            yield dsrc

    sum_t: torch.Tensor | None = None
    sumsq_t: torch.Tensor | None = None
    cnt_t: torch.Tensor | None = None
    robust_samples: list[torch.Tensor] = []
    robust_rows = 0

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    if not use_none:
        g_stats = torch.Generator(device="cpu")
        g_samp = torch.Generator(device="cpu")
        if seed_value is not None:
            with contextlib.suppress(Exception):
                g_stats.manual_seed(seed_value)
                g_samp.manual_seed(seed_value + 1337)

        iter_data = _iter_data(data) if (not is_dist or rank == 0) else []

        for _d in iter_data:
            fx_, lb_, _, _ = preprocess(_d)
            if not torch.is_floating_point(lb_) and not bool(scale_non_floating):
                continue
            if shuffle:
                n_total_ps = (
                    int(lb_.shape[0]) if hasattr(lb_, "shape") and lb_.ndim > 0 else 0
                )
                if n_total_ps > 0:
                    if seed_value is not None:
                        with contextlib.suppress(Exception):
                            g_stats.manual_seed(seed_value)
                    perm_ = torch.randperm(n_total_ps, generator=g_stats)
                    fx_ = fx_.index_select(0, perm_)
                    lb_ = lb_.index_select(0, perm_)
            n_total_ = int(lb_.shape[0]) if hasattr(lb_, "shape") and lb_.ndim > 0 else 0
            val_count_ = int(round(n_total_ * float(val_frac)))
            n_train_ = n_total_ - val_count_
            if n_train_ <= 0:
                continue
            y_ = lb_[:n_train_].detach().to(torch.float64)
            if y_.ndim == 1:
                y_ = y_.unsqueeze(1)
            if use_robust:
                if robust_rows < int(robust_cap):
                    remain = int(robust_cap) - robust_rows
                    if y_.shape[0] > remain:
                        if seed_value is not None:
                            with contextlib.suppress(Exception):
                                g_samp.manual_seed(seed_value + robust_rows)
                        idx = torch.randperm(y_.shape[0], generator=g_samp)[:remain]
                        y_ = y_.index_select(0, idx)
                    robust_samples.append(y_)
                    robust_rows += y_.shape[0]
            else:
                finite_ = torch.isfinite(y_)
                zero_fill = torch.zeros((), dtype=torch.float64, device=y_.device)
                y_masked_ = torch.where(finite_, y_, zero_fill)
                s_ = y_masked_.sum(dim=0)
                ss_ = (y_masked_ * y_masked_).sum(dim=0)
                c_ = finite_.sum(dim=0).to(torch.float64)
                if sum_t is None:
                    sum_t, sumsq_t, cnt_t = s_, ss_, c_
                else:
                    sum_t = sum_t + s_
                    sumsq_t = sumsq_t + ss_
                    cnt_t = cnt_t + c_

    if use_none:
        mean_fit = None
        std_fit = None
    elif use_robust:
        if len(robust_samples) == 0:
            mean_fit = torch.zeros(1, dtype=torch.float64)
            std_fit = torch.ones(1, dtype=torch.float64)
        else:
            Y = torch.cat(robust_samples, dim=0)
            med = torch.nanmedian(Y, dim=0).values
            nanquantile = getattr(torch, "nanquantile", None)
            if nanquantile is not None:
                q1 = torch.nanquantile(Y, q_lo, dim=0)
                q3 = torch.nanquantile(Y, q_hi, dim=0)
            else:
                cols_q1 = []
                cols_q3 = []
                for c in range(Y.shape[1]):
                    col = Y[:, c]
                    col = col[torch.isfinite(col)]
                    if col.numel() == 0:
                        cols_q1.append(
                            torch.tensor(float("nan"), dtype=Y.dtype, device=Y.device)
                        )
                        cols_q3.append(
                            torch.tensor(float("nan"), dtype=Y.dtype, device=Y.device)
                        )
                    else:
                        cols_q1.append(torch.quantile(col, q_lo))
                        cols_q3.append(torch.quantile(col, q_hi))
                q1 = torch.stack(cols_q1)
                q3 = torch.stack(cols_q3)
            iqr = q3 - q1
            iqr = torch.nan_to_num(iqr, nan=0.0).clamp(min=1e-6)
            med = torch.nan_to_num(med, nan=0.0)
            mean_fit = med
            std_fit = iqr
    else:
        if sum_t is None or sumsq_t is None or cnt_t is None:
            mean_fit = torch.zeros(1, dtype=torch.float64)
            std_fit = torch.ones(1, dtype=torch.float64)
        else:
            cnt_safe = torch.clamp(cnt_t, min=1.0)
            mean_fit = sum_t / cnt_safe
            var_fit = torch.clamp(sumsq_t / cnt_safe - mean_fit * mean_fit, min=0.0)
            std_fit = torch.sqrt(var_fit)
            std_fit = torch.clamp(std_fit, min=1e-6)
            zero_cnt_mask = (cnt_t == 0) if isinstance(cnt_t, torch.Tensor) else None
            if isinstance(zero_cnt_mask, torch.Tensor) and torch.any(zero_cnt_mask):
                mean_fit = torch.where(zero_cnt_mask, torch.zeros_like(mean_fit), mean_fit)
                std_fit = torch.where(zero_cnt_mask, torch.ones_like(std_fit), std_fit)

    if not use_none:
        set_scaler(mean=mean_fit.cpu(), std=std_fit.cpu())
        _attach_scaler(model, mean_fit, std_fit)
        with contextlib.suppress(Exception):
            if dist.is_available() and dist.is_initialized():
                try:
                    backend = dist.get_backend()
                except Exception:
                    backend = None
                for name_ in ("target_mean", "target_std"):
                    buf = getattr(model, name_, None)
                    if not isinstance(buf, torch.Tensor):
                        continue
                    if backend == "nccl":
                        if torch.cuda.is_available():
                            dev = torch.device("cuda", torch.cuda.current_device())
                            tmp = buf.to(dev, dtype=buf.dtype)
                            dist.broadcast(tmp, src=0)
                            buf.copy_(tmp.to("cpu", dtype=buf.dtype))
                        else:
                            dist.broadcast(buf, src=0)
                    else:
                        dist.broadcast(buf, src=0)
                mean_buf = getattr(model, "target_mean", None)
                std_buf = getattr(model, "target_std", None)
                if isinstance(mean_buf, torch.Tensor) and isinstance(std_buf, torch.Tensor):
                    set_scaler(mean=mean_buf.detach().cpu(), std=std_buf.detach().cpu())

    memmap_dir = new_dir("memmap_ds")

    first_feats: Optional[torch.Tensor] = None
    label_shape: Tuple[int, ...] = ()
    manifest: Optional[Dict[str, str] | Sequence[str]] = None

    try:
        if isinstance(data, Mapping) and data and all(isinstance(v, Mapping) for v in data.values()):
            manifest = {}
            for k, d in data.items():
                sub = os.path.join(memmap_dir, str(k))
                os.makedirs(sub, exist_ok=True)
                fx, lshape = _mat_one(d, sub)
                if first_feats is None:
                    first_feats, label_shape = fx, lshape
                else:
                    if int(fx.shape[1]) != int(first_feats.shape[1]) or tuple(lshape) != tuple(label_shape):
                        raise RuntimeError("inconsistent feature/label shapes across datasets")
                manifest[str(k)] = str(k)
        elif isinstance(data, Sequence) and data and all(isinstance(d, Mapping) for d in data):
            manifest = []
            for i, d in enumerate(data):
                key = str(i)
                sub = os.path.join(memmap_dir, key)
                os.makedirs(sub, exist_ok=True)
                fx, lshape = _mat_one(d, sub)
                if first_feats is None:
                    first_feats, label_shape = fx, lshape
                else:
                    if int(fx.shape[1]) != int(first_feats.shape[1]) or tuple(lshape) != tuple(label_shape):
                        raise RuntimeError("inconsistent feature/label shapes across datasets")
                manifest.append(key)
        else:
            fx, lb, _, lshape = preprocess(data)
            if shuffle:
                n_total_ps = int(lb.shape[0]) if hasattr(lb, "shape") and lb.ndim > 0 else 0
                if n_total_ps > 0:
                    g = torch.Generator(device="cpu")
                    if seed_value is not None:
                        g.manual_seed(seed_value)
                    perm = torch.randperm(n_total_ps, generator=g)
                    fx = fx.index_select(0, perm)
                    lb = lb.index_select(0, perm)
            n_total = int(lb.shape[0]) if hasattr(lb, "shape") and lb.ndim > 0 else 0
            if (not torch.is_floating_point(lb)) and (not bool(scale_non_floating)):
                SampleReader.preload(
                    {"features": fx, "labels": lb},
                    memmap_dir=memmap_dir,
                    train_frac=1.0 - float(val_frac),
                    val_frac=float(val_frac),
                    shuffle=bool(shuffle),
                    seed=seed_value,
                    target_scaler=mode,
                    robust_q=tuple(robust_q),
                    robust_cap=int(robust_cap),
                    scale_non_floating=bool(scale_non_floating),
                )
                first_feats, label_shape = fx, tuple(lshape)
            else:
                val_count = int(round(n_total * float(val_frac)))
                n_train = n_total - val_count
                if (not use_none) and (get_scaler() is None):
                    lb = _coerce_scaler(lb, fit_count=n_train if n_train > 0 else None)
                SampleReader.preload(
                    {"features": fx, "labels": lb},
                    memmap_dir=memmap_dir,
                    train_frac=1.0 - float(val_frac),
                    val_frac=float(val_frac),
                    shuffle=False,
                    seed=seed_value,
                    target_scaler=mode,
                    robust_q=tuple(robust_q),
                    robust_cap=int(robust_cap),
                    scale_non_floating=bool(scale_non_floating),
                )
                first_feats, label_shape = fx, tuple(lshape)

        if first_feats is None or not label_shape:
            raise RuntimeError("no training data provided to train()")

        if manifest is not None:
            with open(os.path.join(memmap_dir, "multinode.json"), "w", encoding="utf-8") as f:
                payload = manifest if isinstance(manifest, dict) else list(manifest)
                json.dump(payload, f)
        ckpt_dir = new_dir("ckpt_dcp")
        init_dir = new_dir("init_dcp")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=ignored_pattern)
            opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
            m_sd = get_model_state_dict(model, options=opts)
            save(
                state_dict={"model": m_sd},
                storage_writer=FileSystemWriter(init_dir, sync_files=True, overwrite=True),
            )
        torch.save(
            {k: v.detach().cpu() for k, v in model.state_dict().items()},
            os.path.join(init_dir, "model.pt"),
        )
        default_rdzv_host = get_preferred_ip(allow_loopback=True) or "127.0.0.1"
        resolved_rdzv = rdzv_endpoint if rdzv_endpoint else default_rdzv_host
        rdzv_endpoint = get_available_host(resolved_rdzv)
        master_addr, _master_port = initialize_master_addr(rdzv_endpoint)
        optimize_threads()
        nprocs = optimal_procs()["nproc_per_node"]
        cfg_obj = getattr(model, "_Root__config", None)
        if isinstance(cfg_obj, (ModelConfig, dict)):
            cfg_model = coerce_model_config(cfg_obj)
        else:
            cfg_model = ModelConfig()
        cfg_dict: Dict[str, Any] = asdict(cfg_model)
        lc = LaunchConfig(
            min_nodes=1,
            max_nodes=max_nodes,
            nproc_per_node=nprocs,
            rdzv_backend=rdzv_backend,
            rdzv_endpoint=rdzv_endpoint,
            run_id=run_id,
            max_restarts=0,
            monitor_interval=5,
            start_method=optimal_start_method(),
            local_addr=master_addr,
        )
        base = dict(
            memmap_dir=memmap_dir,
            ckpt_dir=ckpt_dir,
            init_ckpt_dir=init_dir,
            in_dim=int(first_feats.shape[1]),
            out_shape=tuple(label_shape),
            cfg_dict=cfg_dict,
        )
        default_kwargs = {
            "epochs": epochs,
            "batch_size": batch_size,
            "val_frac": val_frac,
            "base_lr": base_lr,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "eta_min": eta_min,
            "seed": seed,
            "prefetch_factor": prefetch_factor,
            "grad_accum_steps": grad_accum_steps,
            "overlap_h2d": overlap_h2d,
            "loss_tile_dim": loss_tile_dim,
            "loss_tile_size": loss_tile_size,
            "loss_mask_mode": loss_mask_mode,
            "loss_mask_value": loss_mask_value,
        }
        positional_names = RuntimeConfig.TRAIN_POS_ORDER[: len(args)]
        for key in list(default_kwargs):
            if key in positional_names or key in kwargs:
                default_kwargs.pop(key, None)
        ops = runtime_config(
            "train",
            base,
            *args,
            **default_kwargs,
            **kwargs,
        )
        elastic_launch(lc, main)(ops)
        fallback = os.path.join(ckpt_dir, "model.pt")
        if os.path.isfile(fallback):
            cpu_state = torch.load(fallback, map_location="cpu")
            cpu_state = _preload_state(cpu_state)
            model.load_state_dict(cpu_state, strict=False)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=ignored_pattern)
                opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
                m_sd = get_model_state_dict(model, options=opts)
                m_sd = _trim_dcp_keys(m_sd)
                load(
                    state_dict={"model": m_sd},
                    storage_reader=FileSystemReader(ckpt_dir),
                )
                set_model_state_dict(
                    model, m_sd, options=StateDictOptions(strict=False)
                )
        shutil.rmtree(memmap_dir, ignore_errors=True)
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        shutil.rmtree(init_dir, ignore_errors=True)
        return model
    finally:
        drop_scaler()


def predict(
    model: Root,
    data: Dict[Tuple, torch.Tensor],
    *args: Any,
    batch_size: int = 512,
    seed: int = 7,
    prefetch_factor: Optional[int] = 1,
    mode: OpsMode = "predict",
    max_nodes: Optional[int] = None,
    rdzv_backend: Optional[str] = None,
    **kwargs: Any,
) -> Dict[Tuple, torch.Tensor]:

    initialize_python_path()
    set_multiprocessing_env()
    tmp_dir = new_dir("infer")
    dcp_dir = os.path.join(tmp_dir, "dcp")
    memmap_dir = os.path.join(tmp_dir, "memmap")
    mp.allow_connection_pickling()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=ignored_pattern)
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        m_sd = get_model_state_dict(model, options=opts)
        save(
            state_dict={"model": m_sd},
            storage_writer=FileSystemWriter(dcp_dir, sync_files=True, overwrite=True),
        )
    torch.save(
        {k: v.detach().cpu() for k, v in model.state_dict().items()},
        os.path.join(dcp_dir, "model.pt"),
    )
    cfg_obj = getattr(model, "_Root__config", None)
    if isinstance(cfg_obj, (ModelConfig, dict)):
        cfg_model = coerce_model_config(cfg_obj)
    else:
        cfg_model = ModelConfig()
    cfg_dict = asdict(cfg_model)
    try:
        seed_value = int(seed)
    except Exception:
        seed_value = None
    if any((v is None for v in data.values())):
        dummy_shape = tuple(model.out_shape)
        data = {
            k: (
                torch.zeros(dummy_shape)
                if v is None
                else torch.as_tensor(v).view(*dummy_shape)
            )
            for k, v in data.items()
        }
    with contextlib.suppress(Exception):
        mean_buf = getattr(model, "target_mean", None)
        std_buf = getattr(model, "target_std", None)
        if mean_buf is not None and std_buf is not None:
            mean_t = (
                torch.as_tensor(mean_buf, dtype=torch.float64)
                .detach()
                .cpu()
            )
            std_t = (
                torch.as_tensor(std_buf, dtype=torch.float64)
                .detach()
                .cpu()
            )
            std_t = torch.clamp(std_t, min=1e-6)
            set_scaler(mean=mean_t, std=std_t)

    feats, labels, keys, label_shape = preprocess(data)
    SampleReader.preload(
        {"features": feats, "labels": labels},
        memmap_dir=memmap_dir,
        train_frac=1.0,
        val_frac=0.0,
        shuffle=False,
        seed=seed_value,
    )
    base = dict(
        model_ckpt_dir=dcp_dir,
        memmap_dir=memmap_dir,
        in_dim=int(feats.shape[1]),
        out_shape=tuple(label_shape),
        cfg_dict=cfg_dict,
        keys=list(keys),
    )
    mode = mode if mode in ("predict", "infer") else "predict"
    default_kwargs = {
        "batch_size": batch_size,
        "seed": seed,
        "prefetch_factor": prefetch_factor,
    }
    positional_names = RuntimeConfig.PRED_POS_ORDER[: len(args)]
    for key in list(default_kwargs):
        if key in positional_names or key in kwargs:
            default_kwargs.pop(key, None)
    ops = runtime_config(
        mode,
        base,
        *args,
        **default_kwargs,
        **kwargs,
    )
    default_rdzv_host = get_preferred_ip(allow_loopback=True) or "127.0.0.1"
    rdzv_endpoint = get_available_host(default_rdzv_host)
    master_addr, _ = initialize_master_addr(rdzv_endpoint)
    optimize_threads()
    nprocs = int(optimal_procs()["nproc_per_node"])
    manager = mp.Manager()
    ret_dict = manager.dict()
    resolved_max_nodes = int(max_nodes) if max_nodes is not None else 1
    resolved_rdzv_backend = rdzv_backend or "c10d"
    lc = LaunchConfig(
        min_nodes=1,
        max_nodes=resolved_max_nodes,
        nproc_per_node=nprocs,
        rdzv_backend=resolved_rdzv_backend,
        rdzv_endpoint=rdzv_endpoint,
        run_id="predict",
        max_restarts=0,
        monitor_interval=5,
        start_method=optimal_start_method(),
        local_addr=master_addr,
    )
    elastic_launch(lc, main)(ops, ret_dict)
    result: Dict[Tuple, torch.Tensor] = dict(ret_dict)
    try:
        return result
    finally:
        with contextlib.suppress(Exception):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        drop_scaler()
