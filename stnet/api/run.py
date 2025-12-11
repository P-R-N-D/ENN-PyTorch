# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import gc
import json
import os
import random
import shutil
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tensordict import MemoryMappedTensor, TensorDictBase
import torch.multiprocessing as mp
from torch.distributed.checkpoint import (FileSystemReader, FileSystemWriter,
                                          load, save)
from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                     get_model_state_dict,
                                                     set_model_state_dict)

try:
    from torch.distributed.run import LaunchConfig, elastic_launch
except ImportError:
    from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from ..backend.distributed import (get_available_host, get_preferred_ip,
                                   initialize_master_addr)
from ..backend.runtime import _trim_dcp_keys, main
from ..backend.system import (initialize_python_path, new_dir,
                              optimal_start_method, set_multiprocessing_env)
from .templates import WorkerPolicy
from ..data.nodes import preload_memmap
from ..data.transforms import preprocess
from ..model.layers import History, Instance
from .config import (ModelConfig, OpsMode, RuntimeConfig, coerce_model_config,
                     runtime_config)

_DTENSOR_TYPE = getattr(getattr(torch.distributed, "_tensor", None), "DTensor", None)


def _clear_device_caches() -> None:
    with contextlib.suppress(Exception):
        gc.collect()

    with contextlib.suppress(Exception):
        accelerator = getattr(torch, "accelerator", None)
        memory_mod = getattr(accelerator, "memory", None) if accelerator is not None else None
        empty_cache = getattr(memory_mod, "empty_cache", None) if memory_mod is not None else None
        if callable(empty_cache):
            empty_cache()

    with contextlib.suppress(Exception):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with contextlib.suppress(Exception):
        mps_mod = getattr(torch, "mps", None)
        empty_cache = getattr(mps_mod, "empty_cache", None) if mps_mod is not None else None
        if callable(empty_cache):
            empty_cache()

    with contextlib.suppress(Exception):
        xpu_mod = getattr(torch, "xpu", None)
        memory_mod = getattr(xpu_mod, "memory", None) if xpu_mod is not None else None
        empty_cache = getattr(memory_mod, "empty_cache", None) if memory_mod is not None else None
        if callable(empty_cache):
            empty_cache()


def _reset_process_group() -> None:
    if not torch.distributed.is_available():
        return
    if torch.distributed.is_initialized():
        with contextlib.suppress(Exception):
            torch.distributed.destroy_process_group()


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


def _resize_scaler_buffers(model: Any, state: Mapping[str, torch.Tensor]) -> None:
    scaler = getattr(model, "scaler", None)
    if scaler is None:
        return

    tensor_keys = (
        "x_mean",
        "x_std",
        "y_mean",
        "y_std",
        "affine_a",
        "affine_b",
        "pw_x",
        "pw_y",
    )
    for name in tensor_keys:
        key = f"scaler.{name}"
        if key not in state:
            continue
        buf = getattr(scaler, name, None)
        tensor = state[key]
        if not isinstance(buf, torch.Tensor) or not isinstance(tensor, torch.Tensor):
            continue
        if buf.shape == tensor.shape:
            continue
        with contextlib.suppress(Exception):
            buf.resize_(tensor.shape)


def _ensure_seed(seed: Optional[int]) -> Optional[int]:
    if seed is None:
        return None
    try:
        return int(seed)
    except (TypeError, ValueError):
        return None


def _seed_everything(seed_value: Optional[int]) -> None:
    if seed_value is None:
        return
    with contextlib.suppress(Exception):
        torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.manual_seed_all(seed_value)
    with contextlib.suppress(Exception):
        random.seed(seed_value)
    with contextlib.suppress(Exception):
        np.random.seed(seed_value)


def train(
    model: Instance,
    data: (
        Dict[Tuple, torch.Tensor]
        | Sequence[Dict[Tuple, torch.Tensor]]
        | Mapping[str, Dict[Tuple, torch.Tensor]]
    ),
    *args: Any,
    epochs: int = 5,
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
    loss_tile_dim: Optional[int] = None,
    loss_tile_size: Optional[int] = None,
    loss_mask_mode: str = "none",
    loss_mask_value: Optional[float] = None,
    **kwargs: Any,
) -> Instance:
    _reset_process_group()
    try:
        val_frac = float(val_frac)
        val_frac = 0.0 if val_frac < 0.0 else (1.0 if val_frac > 1.0 else val_frac)
    except (TypeError, ValueError):
        val_frac = 0.1

    seed_value = _ensure_seed(seed)
    _seed_everything(seed_value)

    with contextlib.suppress(Exception):
        torch.use_deterministic_algorithms(False, warn_only=True)
    with contextlib.suppress(Exception):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    def _check_shapes(
        first_in_dim: Optional[int],
        in_dim: int,
        first_label_shape: Tuple[int, ...],
        lshape: Tuple[int, ...],
    ) -> Tuple[Optional[int], Tuple[int, ...]]:
        if first_in_dim is None:
            return int(in_dim), tuple(lshape)
        if int(in_dim) != int(first_in_dim) or tuple(lshape) != tuple(first_label_shape):
            raise RuntimeError(
                f"Shape mismatch across datasets: expected X_dim={first_in_dim}, y_shape={first_label_shape}, "
                f"got X_dim={in_dim}, y_shape={lshape}"
            )
        return first_in_dim, first_label_shape

    def _mat_one(d: Any, out_dir: str) -> Tuple[int, Tuple[int, ...], int]:

        from collections.abc import Mapping as _Mapping

        if isinstance(d, TensorDictBase):
            td = d
            if td.batch_size is None or len(td.batch_size) == 0:
                raise ValueError("TensorDict input to train() must have a batch dimension.")

            count = int(td.batch_size[0])
            if count <= 0:
                raise ValueError("Empty TensorDict provided to train().")

            chunk_size = 32
            chunk_size = min(chunk_size, count)
            first_td = td[:chunk_size]
            fx0, lb0, _, _ = preprocess(first_td)
            fx0 = fx0.contiguous()
            lb0 = lb0.contiguous()

            if fx0.ndim < 2:
                fx0 = fx0.reshape(fx0.shape[0], -1)

            n0 = int(fx0.shape[0])
            in_dim = int(fx0.reshape(n0, -1).shape[1])
            label_shape = tuple(lb0.shape[1:])

            os.makedirs(out_dir, exist_ok=True)
            features_path = os.path.join(out_dir, "features.mmt")
            labels_path = os.path.join(out_dir, "labels.mmt")

            features_mmt = MemoryMappedTensor.empty(
                (count, in_dim),
                dtype=fx0.dtype,
                filename=features_path,
                existsok=True,
            )
            labels_mmt = MemoryMappedTensor.empty(
                (count, *label_shape),
                dtype=lb0.dtype,
                filename=labels_path,
                existsok=True,
            )

            features_mmt[0:n0].copy_(fx0.view(n0, -1))
            labels_mmt[0:n0].copy_(lb0.view(n0, *label_shape))
            written = n0

            idx = chunk_size
            while idx < count:
                end = min(idx + chunk_size, count)
                td_chunk = td[idx:end]

                fx, lb, _, _ = preprocess(td_chunk)
                fx = fx.contiguous()
                lb = lb.contiguous()

                if tuple(lb.shape[1:]) != label_shape:
                    raise RuntimeError(
                        f"label shape mismatch: expected {label_shape}, "
                        f"got {tuple(lb.shape[1:])}"
                    )

                n = int(fx.shape[0])
                if int(fx.reshape(n, -1).shape[1]) != in_dim:
                    raise RuntimeError(
                        f"feature dim mismatch: expected {in_dim}, "
                        f"got {int(fx.reshape(n, -1).shape[1])}"
                    )

                features_mmt[idx : idx + n].copy_(fx.view(n, -1))
                labels_mmt[idx : idx + n].copy_(lb.view(n, *label_shape))

                written += n
                idx = end

            if written != count:
                raise RuntimeError(f"memmap written={written}, expected={count}")

            val_count = max(0, min(count, int(round(count * float(val_frac)))))
            train_count = max(0, count - val_count)
            train_start, train_end = 0, train_count
            val_start, val_end = train_end, train_end + val_count

            meta = {
                "N": int(count),
                "feature_dim": int(in_dim),
                "features_path": "features.mmt",
                "labels_path": "labels.mmt",
                "label_shape": list(label_shape),
                "features_dtype": str(fx0.dtype).replace("torch.", ""),
                "labels_dtype": str(lb0.dtype).replace("torch.", ""),
                "fractions": [float(1.0 - float(val_frac)), float(val_frac)],
                "shuffled": False,
                "shuffle_seed": int(seed_value) if seed_value is not None else None,
                "shuffle_mode": "none",
                "train_start": int(train_start),
                "train_end": int(train_end),
                "val_start": int(val_start),
                "val_end": int(val_end),
            }

            with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f)

            return in_dim, label_shape, count

        if (
            isinstance(d, _Mapping)
            and not isinstance(d, TensorDictBase)
            and d
            and all(not isinstance(v, _Mapping) for v in d.values())
        ):
            items = list(d.items())
            count = len(items)
            if count <= 0:
                raise ValueError("Empty dataset provided to train().")

            chunk_size = 32
            chunk_size = min(chunk_size, count)
            first_keys = [k for k, _ in items[:chunk_size]]
            first_batch = {k: d[k] for k in first_keys}

            fx0, lb0, _, lshape0 = preprocess(first_batch)
            fx0 = fx0.contiguous()
            lb0 = lb0.contiguous()

            if fx0.ndim < 2:
                fx0 = fx0.reshape(fx0.shape[0], -1)

            n0 = int(fx0.shape[0])
            in_dim = int(fx0.reshape(n0, -1).shape[1])
            label_shape = tuple(lb0.shape[1:])

            os.makedirs(out_dir, exist_ok=True)
            features_path = os.path.join(out_dir, "features.mmt")
            labels_path = os.path.join(out_dir, "labels.mmt")
            features_mmt = MemoryMappedTensor.empty(
                (count, in_dim),
                dtype=fx0.dtype,
                filename=features_path,
                existsok=True,
            )
            labels_mmt = MemoryMappedTensor.empty(
                (count, *label_shape),
                dtype=lb0.dtype,
                filename=labels_path,
                existsok=True,
            )
            features_mmt[0:n0].copy_(fx0.view(n0, -1))
            labels_mmt[0:n0].copy_(lb0.view(n0, *label_shape))
            written = n0
            idx = chunk_size
            while idx < count:
                end = min(idx + chunk_size, count)
                batch_items = items[idx:end]
                batch_dict = {k: v for (k, v) in batch_items}

                fx, lb, _, lshape = preprocess(batch_dict)
                fx = fx.contiguous()
                lb = lb.contiguous()

                if tuple(lb.shape[1:]) != label_shape:
                    raise RuntimeError(
                        f"label shape mismatch: expected {label_shape}, "
                        f"got {tuple(lb.shape[1:])}"
                    )

                n = int(fx.shape[0])
                if int(fx.reshape(n, -1).shape[1]) != in_dim:
                    raise RuntimeError(
                        f"feature dim mismatch: expected {in_dim}, "
                        f"got {int(fx.reshape(n, -1).shape[1])}"
                    )

                features_mmt[idx : idx + n].copy_(fx.view(n, -1))
                labels_mmt[idx : idx + n].copy_(lb.view(n, *label_shape))

                written += n
                idx = end

            if written != count:
                raise RuntimeError(f"memmap written={written}, expected={count}")
            val_count = max(0, min(count, int(round(count * float(val_frac)))))
            train_count = max(0, count - val_count)
            train_start, train_end = 0, train_count
            val_start, val_end = train_end, train_end + val_count

            meta = {
                "N": int(count),
                "feature_dim": int(in_dim),
                "features_path": "features.mmt",
                "labels_path": "labels.mmt",
                "label_shape": list(label_shape),
                "features_dtype": str(fx0.dtype).replace("torch.", ""),
                "labels_dtype": str(lb0.dtype).replace("torch.", ""),
                "fractions": [float(1.0 - float(val_frac)), float(val_frac)],
                "shuffled": False,
                "shuffle_seed": int(seed_value) if seed_value is not None else None,
                "shuffle_mode": "none",
                "train_start": int(train_start),
                "train_end": int(train_end),
                "val_start": int(val_start),
                "val_end": int(val_end),
            }

            with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f)

            return in_dim, label_shape, count

        fx, lb, _, lshape = preprocess(d)
        fx = fx.contiguous()
        count = int(fx.shape[0])
        if count <= 0:
            raise ValueError("Empty dataset provided to train().")
        in_dim = int(fx.reshape(count, -1).shape[1])

        preload_memmap(
            {"features": fx, "labels": lb},
            memmap_dir=out_dir,
            train_frac=1.0 - float(val_frac),
            val_frac=float(val_frac),
            shuffle=False,
            seed=seed_value,
        )
        del fx, lb
        return in_dim, tuple(lshape), count

    initialize_python_path()
    mp.allow_connection_pickling()
    set_multiprocessing_env()

    memmap_dir = new_dir("memmap_ds")

    num_samples = 0

    first_in_dim: Optional[int] = None
    label_shape: Tuple[int, ...] = ()
    manifest: Optional[Dict[str, str] | Sequence[str]] = None
    ckpt_dir: Optional[str] = None
    init_dir: Optional[str] = None

    try:
        if isinstance(data, TensorDictBase):
            in_dim, lshape, n = _mat_one(data, memmap_dir)
            first_in_dim, label_shape = _check_shapes(
                first_in_dim, in_dim, label_shape, lshape
            )
            num_samples += n
        elif (
            isinstance(data, Mapping)
            and data
            and all(isinstance(v, Mapping) for v in data.values())
        ):
            manifest = {}
            for k, d in data.items():
                sub = os.path.join(memmap_dir, str(k))
                os.makedirs(sub, exist_ok=True)
                in_dim, lshape, n = _mat_one(d, sub)
                first_in_dim, label_shape = _check_shapes(
                    first_in_dim, in_dim, label_shape, lshape
                )
                num_samples += n
                manifest[str(k)] = str(k)
        elif (
            isinstance(data, Sequence)
            and data
            and all(isinstance(d, Mapping) for d in data)
        ):
            manifest = []
            for i, d in enumerate(data):
                key = str(i)
                sub = os.path.join(memmap_dir, key)
                os.makedirs(sub, exist_ok=True)
                in_dim, lshape, n = _mat_one(d, sub)
                first_in_dim, label_shape = _check_shapes(
                    first_in_dim, in_dim, label_shape, lshape
                )
                num_samples += n
                manifest.append(key)
        else:
            in_dim, lshape, n = _mat_one(data, memmap_dir)
            first_in_dim, label_shape = in_dim, tuple(lshape)
            num_samples += n

        if first_in_dim is None or not label_shape:
            raise RuntimeError("no training data provided to train()")

        if manifest is not None:
            with open(
                os.path.join(memmap_dir, "multinode.json"), "w", encoding="utf-8"
            ) as f:
                payload = manifest if isinstance(manifest, dict) else list(manifest)
                json.dump(payload, f)
        ckpt_dir = new_dir("ckpt_dcp")
        init_dir = new_dir("init_dcp")
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        m_sd = get_model_state_dict(model, options=opts)
        save(
            state_dict={"model": m_sd},
            storage_writer=FileSystemWriter(
                init_dir, sync_files=True, overwrite=True
            ),
        )
        torch.save(
            {k: v.detach().cpu() for k, v in model.state_dict().items()},
            os.path.join(init_dir, "model.pt"),
        )
        default_rdzv_host = get_preferred_ip(allow_loopback=True) or "127.0.0.1"
        resolved_rdzv = rdzv_endpoint if rdzv_endpoint else default_rdzv_host
        rdzv_endpoint = get_available_host(resolved_rdzv)
        master_addr, _master_port = initialize_master_addr(rdzv_endpoint)
        # Keep local world-size & threads consistent with WorkerPolicy used by loaders.
        _wp = WorkerPolicy.autotune()
        _wp.apply_torch_threads()
        nprocs = int(_wp.nproc_per_node)
        cfg_obj = getattr(model, "_Instance__config", None)
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
            sources={"kind": "memmap", "path": memmap_dir},
            ckpt_dir=ckpt_dir,
            init_ckpt_dir=init_dir,
            in_dim=int(first_in_dim),
            out_shape=tuple(label_shape),
            cfg_dict=cfg_dict,
        )
        default_kwargs = {
            "epochs": epochs,
            "val_frac": val_frac,
            "base_lr": base_lr,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "eta_min": eta_min,
            "seed": seed,
            "loss_tile_dim": loss_tile_dim,
            "loss_tile_size": loss_tile_size,
            "loss_mask_mode": loss_mask_mode,
            "loss_mask_value": loss_mask_value,
        }

        num_samples = int(num_samples)
        this_run_samples = int(num_samples)
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
        with contextlib.suppress(Exception):
            model.to("cpu")
        _clear_device_caches()
        elastic_launch(lc, main)(ops)
        fallback = os.path.join(ckpt_dir, "model.pt")
        if os.path.isfile(fallback):
            cpu_state = torch.load(fallback, map_location="cpu")
            cpu_state = _preload_state(cpu_state)
            _resize_scaler_buffers(model, cpu_state)
            model.load_state_dict(cpu_state, strict=False)
        else:
            opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
            m_sd = get_model_state_dict(model, options=opts)
            m_sd = _trim_dcp_keys(m_sd)
            load(
                state_dict={"model": m_sd},
                storage_reader=FileSystemReader(ckpt_dir),
            )
            _resize_scaler_buffers(model, m_sd)
            set_model_state_dict(
                model, m_sd, options=StateDictOptions(strict=False)
            )

        try:
            if ckpt_dir is not None:
                history_path = os.path.join(ckpt_dir, "history.json")
                if os.path.isfile(history_path):
                    with open(history_path, "r", encoding="utf-8") as f:
                        raw = json.load(f)

                    if isinstance(raw, dict):
                        records = raw.get("records", []) or []
                        meta = raw.get("meta", {}) or {}
                    else:
                        records = raw if isinstance(raw, list) else []
                        meta = {}

                    logger = getattr(model, "logger", None)

                    if isinstance(meta, dict):
                        setattr(model, "_train_history_meta", dict(meta))

                    def _aggregate_run_stats(
                        recs: List[Mapping[str, Any]],
                    ) -> Optional[Dict[str, float]]:
                        if not isinstance(recs, list) or len(recs) == 0:
                            return None
                        total_bs = 0
                        sum_x = 0.0
                        sum_x2 = 0.0
                        sum_y = 0.0
                        sum_y2 = 0.0
                        x_min = float("inf")
                        x_max = float("-inf")
                        y_min = float("inf")
                        y_max = float("-inf")
                        for r in recs:
                            if not isinstance(r, Mapping):
                                continue
                            bs = int(r.get("batch_size", 0))
                            if bs <= 0:
                                continue
                            bxm = float(r.get("batch_x_mean", 0.0))
                            bxv = float(r.get("batch_x_var", 0.0))
                            bym = float(r.get("batch_y_mean", 0.0))
                            byv = float(r.get("batch_y_var", 0.0))
                            bxmin = float(r.get("batch_x_min", float("inf")))
                            bxmax = float(r.get("batch_x_max", float("-inf")))
                            bymin = float(r.get("batch_y_min", float("inf")))
                            bymax = float(r.get("batch_y_max", float("-inf")))

                            total_bs += bs
                            sum_x += bxm * bs
                            sum_x2 += (bxv + bxm * bxm) * bs
                            sum_y += bym * bs
                            sum_y2 += (byv + bym * bym) * bs

                            x_min = min(x_min, bxmin)
                            x_max = max(x_max, bxmax)
                            y_min = min(y_min, bymin)
                            y_max = max(y_max, bymax)

                        if total_bs <= 0:
                            return None

                        mean_x = sum_x / total_bs
                        mean_y = sum_y / total_bs
                        var_x = max(sum_x2 / total_bs - mean_x * mean_x, 0.0)
                        var_y = max(sum_y2 / total_bs - mean_y * mean_y, 0.0)

                        return {
                            "sampled_x_mean": mean_x,
                            "sampled_x_var": var_x,
                            "sampled_x_min": x_min,
                            "sampled_x_max": x_max,
                            "sampled_y_mean": mean_y,
                            "sampled_y_var": var_y,
                            "sampled_y_min": y_min,
                            "sampled_y_max": y_max,
                        }

                    if isinstance(records, list) and len(records) > 0:
                        run_stats = _aggregate_run_stats(records)
                    else:
                        run_stats = None

                    prev_total = int(getattr(model, "_history_total_samples", 0))
                    inc_samples = this_run_samples
                    new_total = prev_total + inc_samples

                    prev_cum = getattr(model, "_history_cum_stats", None)

                    def _update_cum_stats(
                        prev: Optional[Dict[str, float]],
                        n_prev: int,
                        inc: Optional[Dict[str, float]],
                        n_inc: int,
                    ) -> Optional[Dict[str, float]]:
                        if inc is None or n_inc <= 0:
                            return prev
                        if prev is None or n_prev <= 0:
                            out = {}
                            for key, val in inc.items():
                                if key.startswith("sampled_"):
                                    out["reduced_" + key[len("sampled_") :]] = float(val)
                            return out

                        out: Dict[str, float] = {}
                        for axis in ("x", "y"):
                            m_key = f"{axis}_mean"
                            v_key = f"{axis}_var"
                            lo_key = f"{axis}_min"
                            hi_key = f"{axis}_max"

                            m_prev = float(prev.get("reduced_" + m_key, 0.0))
                            v_prev = float(prev.get("reduced_" + v_key, 0.0))
                            lo_prev = float(prev.get("reduced_" + lo_key, float("inf")))
                            hi_prev = float(prev.get("reduced_" + hi_key, float("-inf")))

                            m_inc = float(inc.get(f"sampled_{m_key}", 0.0))
                            v_inc = float(inc.get(f"sampled_{v_key}", 0.0))
                            lo_inc = float(inc.get(f"sampled_{lo_key}", float("inf")))
                            hi_inc = float(inc.get(f"sampled_{hi_key}", float("-inf")))

                            sum_prev = m_prev * n_prev
                            sum2_prev = (v_prev + m_prev * m_prev) * n_prev
                            sum_inc = m_inc * n_inc
                            sum2_inc = (v_inc + m_inc * m_inc) * n_inc

                            n_new = n_prev + n_inc
                            sum_new = sum_prev + sum_inc
                            sum2_new = sum2_prev + sum2_inc

                            m_new = sum_new / n_new
                            v_new = max(sum2_new / n_new - m_new * m_new, 0.0)

                            lo_new = min(lo_prev, lo_inc)
                            hi_new = max(hi_prev, hi_inc)

                            out["reduced_" + m_key] = m_new
                            out["reduced_" + v_key] = v_new
                            out["reduced_" + lo_key] = lo_new
                            out["reduced_" + hi_key] = hi_new

                        return out

                    cum_stats = _update_cum_stats(prev_cum, prev_total, run_stats, inc_samples)

                    setattr(model, "_history_total_samples", new_total)
                    if cum_stats is not None:
                        setattr(model, "_history_cum_stats", cum_stats)

                    run_hist_prev = getattr(model, "_train_history", None)
                    run_index = len(run_hist_prev) if isinstance(run_hist_prev, list) else 0

                    run_record: Dict[str, Any] = {
                        "run_index": run_index,
                        "sampled_n": inc_samples,
                        "reduced_n": new_total,
                    }
                    if run_stats is not None:
                        run_record.update(run_stats)
                    if cum_stats is not None:
                        run_record.update(cum_stats)
                    if isinstance(meta, dict) and meta:
                        run_record["env"] = dict(meta)

                    if isinstance(run_hist_prev, list):
                        new_run_hist = run_hist_prev + [run_record]
                    else:
                        new_run_hist = [run_record]

                    setattr(model, "_train_history", new_run_hist)

                    if isinstance(logger, History):
                        logger._records = new_run_hist
        except Exception:
            pass

        return model
    finally:
        shutil.rmtree(memmap_dir, ignore_errors=True)
        if ckpt_dir is not None:
            shutil.rmtree(ckpt_dir, ignore_errors=True)
        if init_dir is not None:
            shutil.rmtree(init_dir, ignore_errors=True)


def predict(
    model: Instance,
    data: Dict[Tuple, torch.Tensor],
    *args: Any,
    seed: int = 7,
    mode: OpsMode = "predict",
    max_nodes: Optional[int] = None,
    rdzv_backend: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:

    _reset_process_group()
    initialize_python_path()
    set_multiprocessing_env()
    tmp_dir = new_dir("infer")
    dcp_dir = os.path.join(tmp_dir, "dcp")
    memmap_dir = os.path.join(tmp_dir, "memmap")
    ckpt_dir = os.path.join(tmp_dir, "pred_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    mp.allow_connection_pickling()
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
    cfg_obj = getattr(model, "_Instance__config", None)
    if isinstance(cfg_obj, (ModelConfig, dict)):
        cfg_model = coerce_model_config(cfg_obj)
    else:
        cfg_model = ModelConfig()
    cfg_dict = asdict(cfg_model)
    seed_value = _ensure_seed(seed)
    _seed_everything(seed_value)

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

    from collections.abc import Mapping as _Mapping

    if isinstance(data, TensorDictBase):
        td = data
        if td.batch_size is None or len(td.batch_size) == 0:
            raise ValueError("TensorDict input to predict() must have a batch dimension.")

        count = int(td.batch_size[0])
        if count <= 0:
            return {}

        chunk_size = 32
        chunk_size = min(chunk_size, count)

        keys = list(range(count))

        first_td = td[:chunk_size]
        feats0, labels0, _, label_shape = preprocess(first_td)
        feats0 = feats0.contiguous()
        labels0 = labels0.contiguous()

        if feats0.ndim < 2:
            feats0 = feats0.reshape(feats0.shape[0], -1)

        n0 = int(feats0.shape[0])
        in_dim = int(feats0.reshape(n0, -1).shape[1])
        label_shape = tuple(labels0.shape[1:])

        os.makedirs(memmap_dir, exist_ok=True)
        features_path = os.path.join(memmap_dir, "features.mmt")
        labels_path = os.path.join(memmap_dir, "labels.mmt")

        features_mmt = MemoryMappedTensor.empty(
            (count, in_dim),
            dtype=feats0.dtype,
            filename=features_path,
            existsok=True,
        )
        labels_mmt = MemoryMappedTensor.empty(
            (count, *label_shape),
            dtype=labels0.dtype,
            filename=labels_path,
            existsok=True,
        )

        features_mmt[0:n0].copy_(feats0.view(n0, -1))
        labels_mmt[0:n0].copy_(labels0.view(n0, *label_shape))
        written = n0

        idx = chunk_size
        while idx < count:
            end = min(idx + chunk_size, count)
            td_chunk = td[idx:end]

            fx, lb, _, _ = preprocess(td_chunk)
            fx = fx.contiguous()
            lb = lb.contiguous()

            if tuple(lb.shape[1:]) != label_shape:
                raise RuntimeError(
                    f"label shape mismatch: expected {label_shape}, "
                    f"got {tuple(lb.shape[1:])}"
                )

            n = int(fx.shape[0])
            if int(fx.reshape(n, -1).shape[1]) != in_dim:
                raise RuntimeError(
                    f"feature dim mismatch: expected {in_dim}, "
                    f"got {int(fx.reshape(n, -1).shape[1])}"
                )

            features_mmt[idx : idx + n].copy_(fx.view(n, -1))
            labels_mmt[idx : idx + n].copy_(lb.view(n, *label_shape))

            written += n
            idx = end

        if written != count:
            raise RuntimeError(f"memmap written={written}, expected={count}")

    elif (
        isinstance(data, _Mapping)
        and not isinstance(data, TensorDictBase)
        and data
        and all(not isinstance(v, _Mapping) for v in data.values())
    ):
        items = list(data.items())
        count = len(items)
        if count <= 0:
            return {}

        chunk_size = 32
        chunk_size = min(chunk_size, count)

        keys = [k for (k, _) in items]

        first_items = items[:chunk_size]
        first_batch = {k: v for (k, v) in first_items}

        feats0, labels0, _, label_shape = preprocess(first_batch)
        feats0 = feats0.contiguous()
        labels0 = labels0.contiguous()

        if feats0.ndim < 2:
            feats0 = feats0.reshape(feats0.shape[0], -1)

        n0 = int(feats0.shape[0])
        in_dim = int(feats0.reshape(n0, -1).shape[1])
        label_shape = tuple(labels0.shape[1:])

        os.makedirs(memmap_dir, exist_ok=True)
        features_path = os.path.join(memmap_dir, "features.mmt")
        labels_path = os.path.join(memmap_dir, "labels.mmt")

        features_mmt = MemoryMappedTensor.empty(
            (count, in_dim),
            dtype=feats0.dtype,
            filename=features_path,
            existsok=True,
        )
        labels_mmt = MemoryMappedTensor.empty(
            (count, *label_shape),
            dtype=labels0.dtype,
            filename=labels_path,
            existsok=True,
        )

        features_mmt[0:n0].copy_(feats0.view(n0, -1))
        labels_mmt[0:n0].copy_(labels0.view(n0, *label_shape))
        written = n0

        idx = chunk_size
        while idx < count:
            end = min(idx + chunk_size, count)
            batch_items = items[idx:end]
            batch_dict = {k: v for (k, v) in batch_items}

            fx, lb, _, _ = preprocess(batch_dict)
            fx = fx.contiguous()
            lb = lb.contiguous()

            if tuple(lb.shape[1:]) != label_shape:
                raise RuntimeError(
                    f"label shape mismatch: expected {label_shape}, "
                    f"got {tuple(lb.shape[1:])}"
                )

            n = int(fx.shape[0])
            if int(fx.reshape(n, -1).shape[1]) != in_dim:
                raise RuntimeError(
                    f"feature dim mismatch: expected {in_dim}, "
                    f"got {int(fx.reshape(n, -1).shape[1])}"
                )

            features_mmt[idx : idx + n].copy_(fx.view(n, -1))
            labels_mmt[idx : idx + n].copy_(lb.view(n, *label_shape))

            written += n
            idx = end

        if written != count:
            raise RuntimeError(f"memmap written={written}, expected={count}")

    else:
        feats, labels, keys, label_shape = preprocess(data)
        feats = feats.contiguous()
        labels = labels.contiguous()

        if feats.ndim < 2:
            feats = feats.reshape(feats.shape[0], -1)

        count = int(feats.shape[0])
        if count <= 0:
            return {}

        in_dim = int(feats.reshape(count, -1).shape[1])

        os.makedirs(memmap_dir, exist_ok=True)
        features_path = os.path.join(memmap_dir, "features.mmt")
        labels_path = os.path.join(memmap_dir, "labels.mmt")

        features_mmt = MemoryMappedTensor.empty(
            (count, in_dim),
            dtype=feats.dtype,
            filename=features_path,
            existsok=True,
        )
        labels_mmt = MemoryMappedTensor.empty(
            (count, *label_shape),
            dtype=labels.dtype,
            filename=labels_path,
            existsok=True,
        )

        features_mmt[0:count].copy_(feats.view(count, -1))
        labels_mmt[0:count].copy_(labels.view(count, *label_shape))

        feats0, labels0 = feats, labels

    val_count = 0
    train_count = count
    train_start, train_end = 0, train_count
    val_start, val_end = train_end, train_end + val_count

    meta = {
        "N": int(count),
        "feature_dim": int(in_dim),
        "features_path": "features.mmt",
        "labels_path": "labels.mmt",
        "label_shape": list(label_shape),
        "features_dtype": str(feats0.dtype).replace("torch.", ""),
        "labels_dtype": str(labels0.dtype).replace("torch.", ""),
        "fractions": [1.0, 0.0],
        "shuffled": False,
        "shuffle_seed": int(seed_value) if seed_value is not None else None,
        "shuffle_mode": "none",
        "train_start": int(train_start),
        "train_end": int(train_end),
        "val_start": int(val_start),
        "val_end": int(val_end),
    }

    with open(os.path.join(memmap_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    base = dict(
        model_ckpt_dir=dcp_dir,
        sources={"kind": "memmap", "path": memmap_dir},
        in_dim=int(in_dim),
        out_shape=tuple(label_shape),
        cfg_dict=cfg_dict,
        keys=list(keys),
        ckpt_dir=ckpt_dir,
    )
    mode = mode if mode in ("predict", "infer") else "predict"
    default_kwargs = {"seed": seed}
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
    with contextlib.suppress(Exception):
        model.to("cpu")
    _clear_device_caches()
    default_rdzv_host = get_preferred_ip(allow_loopback=True) or "127.0.0.1"
    rdzv_endpoint = get_available_host(default_rdzv_host)
    master_addr, _ = initialize_master_addr(rdzv_endpoint)
    _wp = WorkerPolicy.autotune()
    _wp.apply_torch_threads()
    nprocs = int(_wp.nproc_per_node)
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
    elastic_launch(lc, main)(ops)
    try:
        chunks_dir = os.path.join(ckpt_dir, "pred_chunks")
        if os.path.isdir(chunks_dir):
            with open(os.path.join(chunks_dir, "keys.json"), "w", encoding="utf-8") as f:
                json.dump(list(keys), f)
            final_dir = new_dir("predictions")
            moved_dir = shutil.move(chunks_dir, final_dir)
            chunk_root = moved_dir if os.path.isdir(moved_dir) else os.path.join(
                final_dir, os.path.basename(chunks_dir)
            )
            manifest_path = os.path.join(chunk_root, "manifest.json")
            if not os.path.isfile(manifest_path):
                return {"chunks_dir": chunk_root, "out_shape": tuple(label_shape)}

            with open(manifest_path, "r", encoding="utf-8") as mf:
                manifest = json.load(mf)

            out_shape = tuple(manifest.get("out_shape") or tuple(label_shape))
            variable_shape = bool(manifest.get("variable_shape"))

            if variable_shape:
                return {
                    "chunks_dir": chunk_root,
                    "out_shape": out_shape,
                    "variable_shape": True,
                }

            num_chunks = int(manifest.get("num_chunks", 0))

            chunks: List[torch.Tensor] = []
            for idx in range(num_chunks):
                base_mmt = os.path.join(chunk_root, f"chunk_{idx:06d}.mmt")
                tensor = None
                if os.path.exists(base_mmt):
                    try:
                        tensor = MemoryMappedTensor.from_filename(base_mmt)
                    except Exception:
                        with contextlib.suppress(Exception):
                            tensor = torch.load(base_mmt, map_location="cpu")
                else:
                    alt_pt = os.path.join(chunk_root, f"chunk_{idx:06d}.pt")
                    if os.path.exists(alt_pt):
                        tensor = torch.load(alt_pt, map_location="cpu")
                if tensor is not None:
                    chunks.append(
                        tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
                    )

            if chunks:
                flat = torch.cat(chunks, dim=0)
            else:
                tail = tuple(out_shape[1:]) if len(out_shape) > 1 else ()
                flat = torch.empty((0, *tail), dtype=torch.float64)

            pred_tensor = Instance.unflatten_y(flat, out_shape)

            result: Dict[Tuple, torch.Tensor] = {}
            for i, key in enumerate(keys):
                if i >= pred_tensor.shape[0]:
                    break
                result[key] = pred_tensor[i].detach().cpu().to(dtype=torch.float64)

            return result
        return {}
    finally:
        with contextlib.suppress(Exception):
            shutil.rmtree(tmp_dir, ignore_errors=True)
