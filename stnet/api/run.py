# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import json
import os
import shutil
import warnings
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, Sequence, Mapping

import torch
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
        _attach_buffer(model, target_mean=mean, target_std=std)
    except Exception:
        pass


def _materialize_state(value: Any) -> Any:
    if _DTENSOR_TYPE is not None and isinstance(value, _DTENSOR_TYPE):
        return value.to_local()
    if isinstance(value, dict):
        return {k: _materialize_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_materialize_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_materialize_state(v) for v in value)
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
    **kwargs: Any,
) -> Root:

    def _coerce_scaler(
        labels: torch.Tensor, *, fit_count: int | None = None
    ) -> torch.Tensor:
        if get_scaler() is not None:
            return labels
        with torch.no_grad():
            safe = labels.detach().to(torch.float64)
            if safe.numel() == 0:
                return labels
            fit = (
                safe[:fit_count]
                if isinstance(fit_count, int)
                and fit_count > 0
                and fit_count <= int(safe.shape[0])
                else safe
            )
            if fit.dim() == 1:
                mean_t = fit.mean()
                std_t = fit.std(unbiased=False)
            else:
                mean_t = fit.mean(dim=0)
                std_t = fit.std(dim=0, unbiased=False)
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
        did_manual_shuffle = False
        if shuffle:
            n_total_ps = int(lb.shape[0]) if hasattr(lb, "shape") and lb.ndim > 0 else 0
            if n_total_ps > 0:
                g = torch.Generator(device="cpu")
                g.manual_seed(int(seed))
                perm = torch.randperm(n_total_ps, generator=g)
                fx = fx.index_select(0, perm)
                lb = lb.index_select(0, perm)
                did_manual_shuffle = True
        n_total = int(lb.shape[0]) if hasattr(lb, "shape") and lb.ndim > 0 else 0
        n_train = int(n_total * (1.0 - float(val_frac)))
        lb = _coerce_scaler(lb, fit_count=n_train if n_train > 0 else None)
        shuffle_for_preload = bool(shuffle and not did_manual_shuffle)
        SampleReader.preload(
            {"features": fx, "labels": lb},
            memmap_dir=out_dir,
            train_frac=1.0 - float(val_frac),
            val_frac=float(val_frac),
            shuffle=shuffle_for_preload,
        )
        return fx, tuple(lshape)

    initialize_python_path()
    mp.allow_connection_pickling()
    set_multiprocessing_env()
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
                    g.manual_seed(int(seed))
                    perm = torch.randperm(n_total_ps, generator=g)
                    fx = fx.index_select(0, perm)
                    lb = lb.index_select(0, perm)
            n_total = int(lb.shape[0]) if hasattr(lb, "shape") and lb.ndim > 0 else 0
            n_train = int(n_total * (1.0 - float(val_frac)))
            lb = _coerce_scaler(lb, fit_count=n_train if n_train > 0 else None)
            SampleReader.preload(
                {"features": fx, "labels": lb},
                memmap_dir=memmap_dir,
                train_frac=1.0 - float(val_frac),
                val_frac=float(val_frac),
                shuffle=shuffle,
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
            cpu_state = _materialize_state(cpu_state)
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
