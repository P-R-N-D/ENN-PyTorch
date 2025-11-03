# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import os
import shutil
import warnings
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

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
from ..data.transforms import BatchLike, preprocess
from ..model import Root
from .config import (
    ModelConfig,
    OpsMode,
    RuntimeConfig,
    coerce_model_config,
    runtime_config,
)
from ..backend.distributed import Distributed, Network
from ..backend.environment import System
from ..backend.runtime import _prune_dcp_state_keys, ignored_pattern, main


def train(
    model: Root,
    data: BatchLike,
    *args: Any,
    epochs: int = 5,
    batch_size: int = 128,
    val_frac: float = 0.1,
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
    """Train ``model`` using in-memory samples.

    ``data`` may be a mapping, a :class:`TensorDict`, or a sequence of those
    structures (e.g., ``List[dict]``) to accommodate multi-node sampling
    workflows.
    """

    System.initialize_python_path()
    feats, labels, _, label_shape = preprocess(data)
    mp.allow_connection_pickling()
    System.set_multiprocessing_env()
    memmap_dir = System.new_dir("memmap_ds")
    SampleReader.materialize(
        {"features": feats, "labels": labels},
        memmap_dir=memmap_dir,
        train_frac=1.0 - float(val_frac),
        val_frac=float(val_frac),
        shuffle=False,
    )
    ckpt_dir = System.new_dir("ckpt_dcp")
    init_dir = System.new_dir("init_dcp")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=ignored_pattern)
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        m_sd = get_model_state_dict(model, options=opts)
        save(
            state_dict={"model": m_sd},
            storage_writer=FileSystemWriter(init_dir, sync_files=True, overwrite=True),
        )
    default_rdzv_host = Network.get_preferred_ip(allow_loopback=True) or "127.0.0.1"
    resolved_rdzv = rdzv_endpoint if rdzv_endpoint else default_rdzv_host
    rdzv_endpoint = Network.get_available_addr(resolved_rdzv)
    master_addr, _master_port = Distributed.initialize_master_addr(rdzv_endpoint)
    System.optimize_threads()
    nprocs = System.optimal_procs()["nproc_per_node"]
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
        start_method=System.optimal_start_method(),
        local_addr=master_addr,
    )
    base = dict(
        memmap_dir=memmap_dir,
        ckpt_dir=ckpt_dir,
        init_ckpt_dir=init_dir,
        in_dim=int(feats.shape[1]),
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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=ignored_pattern)
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        m_sd = get_model_state_dict(model, options=opts)
        m_sd = _prune_dcp_state_keys(m_sd)
        load(state_dict={"model": m_sd}, storage_reader=FileSystemReader(ckpt_dir))
        set_model_state_dict(model, m_sd, options=StateDictOptions(strict=False))
    shutil.rmtree(memmap_dir, ignore_errors=True)
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    shutil.rmtree(init_dir, ignore_errors=True)
    return model


def predict(
    model: Root,
    data: BatchLike,
    *args: Any,
    batch_size: int = 512,
    seed: int = 7,
    prefetch_factor: Optional[int] = 1,
    mode: OpsMode = "predict",
    max_nodes: Optional[int] = None,
    rdzv_backend: Optional[str] = None,
    **kwargs: Any,
) -> Dict[Tuple, torch.Tensor]:
    """Run batched inference and return predictions keyed by sample index."""

    System.initialize_python_path()
    System.set_multiprocessing_env()
    tmp_dir = System.new_dir("infer")
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
    cfg_obj = getattr(model, "_Root__config", None)
    if isinstance(cfg_obj, (ModelConfig, dict)):
        cfg_model = coerce_model_config(cfg_obj)
    else:
        cfg_model = ModelConfig()
    cfg_dict = asdict(cfg_model)
    def _materialize_missing(batch: Mapping[str, Any]) -> Mapping[str, Any]:
        if not any(value is None for value in batch.values()):
            return batch
        dummy_shape = tuple(model.out_shape)
        return {
            key: (
                torch.zeros(dummy_shape)
                if value is None
                else torch.as_tensor(value).view(*dummy_shape)
            )
            for key, value in batch.items()
        }

    if isinstance(data, Mapping):
        data = _materialize_missing(data)
    elif isinstance(data, Sequence) and not isinstance(data, (bytes, str)):
        data = [
            _materialize_missing(item) if isinstance(item, Mapping) else item
            for item in data
        ]
    feats, labels, keys, label_shape = preprocess(data)
    SampleReader.materialize(
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
    default_rdzv_host = Network.get_preferred_ip(allow_loopback=True) or "127.0.0.1"
    rdzv_endpoint = Network.get_available_addr(default_rdzv_host)
    master_addr, _ = Distributed.initialize_master_addr(rdzv_endpoint)
    System.optimize_threads()
    nprocs = int(System.optimal_procs()["nproc_per_node"])
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
        start_method=System.optimal_start_method(),
        local_addr=master_addr,
    )
    elastic_launch(lc, main)(ops, ret_dict)
    result: Dict[Tuple, torch.Tensor] = dict(ret_dict)
    try:
        return result
    finally:
        with contextlib.suppress(Exception):
            shutil.rmtree(tmp_dir, ignore_errors=True)

launch = SimpleNamespace(train=train, predict=predict)
