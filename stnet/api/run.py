# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import shutil
import time
from functools import lru_cache, wraps
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from tensordict import MemoryMappedTensor, TensorDict, TensorDictBase, PersistentTensorDict
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, load, save
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

try:
    from torch.distributed.run import LaunchConfig, Std, elastic_launch
except ImportError:  # pragma: no cover
    from torch.distributed.launcher.api import LaunchConfig, Std, elastic_launch

from ..backend.distributed import get_available_host, get_preferred_ip, initialize_master_addr
from ..backend.runtime import _trim_dcp_keys, main
from ..backend.system import (
    WorkerPolicy,
    initialize_python_path,
    new_dir,
    optimal_start_method,
    remove_dir,
    set_multiprocessing_env,
)
from ..data.datatype import dtype_from_name, env_bool, parse_torch_dtype
from ..data.nodes import preload_memmap
from ..data.pipeline import (
    BatchIterator,
    Dataset,
    default_underflow_action,
    extract_xy,
    normalize_underflow_action,
    resolve_feature_key,
)
from ..model.fused import Gradient
from ..model.nn import History, Root, resize_scaler_buffer
from .io import _to_cpu, _torch_load_checkpoint
from .config import ModelConfig, OpsMode, RuntimeConfig, coerce_model_config, model_config_to_dict, runtime_config


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _timing_logs_enabled() -> bool:
    # Best-effort timing logs (disabled by default). Users can opt-in with any of these flags.
    return env_bool(("STNET_LOG_TIMINGS", "STNET_TIMINGS", "STNET_DEBUG_TIMINGS"), default=False)


def catchtime(log: logging.Logger, *, fn_name: str | None = None):
    """Best-effort timing decorator.

    - Safe at import time (never raises)
    - When disabled, adds minimal overhead
    """

    def deco(fn):
        name = fn_name or getattr(fn, "__name__", "<fn>")

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not _timing_logs_enabled():
                return fn(*args, **kwargs)
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                with contextlib.suppress(Exception):
                    log.info("%s took %.3fs", name, float(dt))

        return wrapper

    return deco


# -----------------------------
# Process / device housekeeping
# -----------------------------

def _reset_process_group() -> None:
    """Best-effort cleanup for a potentially initialized process group.

    In interactive/iterative workflows it is easy to end up with an initialized
    process group from a previous run. That can break subsequent elastic runs.

    This function is intentionally conservative: it never raises.
    """
    try:
        import torch.distributed as _dist

        if _dist.is_available() and _dist.is_initialized():
            with contextlib.suppress(Exception):
                _dist.barrier()
            with contextlib.suppress(Exception):
                _dist.destroy_process_group()
    except Exception:
        pass


def _clear_device_caches() -> None:
    """Release best-effort accelerator caches (CUDA/XPU/MPS) and run GC."""
    with contextlib.suppress(Exception):
        import gc

        gc.collect()

    # Prefer the project helper when available (rate-limited).
    with contextlib.suppress(Exception):
        from ..backend.system import empty_device_cache, get_device

        empty_device_cache(device=get_device(), do_gc=False, min_interval_s=0.0)

    with contextlib.suppress(Exception):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            with contextlib.suppress(Exception):
                torch.cuda.ipc_collect()

    with contextlib.suppress(Exception):
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and callable(getattr(xpu, "empty_cache", None)):
            if (not callable(getattr(xpu, "is_available", None))) or bool(xpu.is_available()):
                xpu.empty_cache()

    with contextlib.suppress(Exception):
        mps = getattr(torch, "mps", None)
        if mps is not None and callable(getattr(mps, "empty_cache", None)):
            mps.empty_cache()


# -----------------------------
# Seeding
# -----------------------------

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
    try:
        torch.manual_seed(seed_value)
    except (TypeError, ValueError, RuntimeError):
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed_value)
        except (TypeError, ValueError, RuntimeError):
            pass
    try:
        random.seed(seed_value)
    except (TypeError, ValueError):
        pass
    try:
        np.random.seed(seed_value)
    except (TypeError, ValueError):
        pass


# -----------------------------
# Checkpoint helpers (DCP + PT)
# -----------------------------

def _maybe_save_model_checkpoint(
    model: Root,
    out_dir: str,
    *,
    save_dcp: bool,
    save_pt: bool,
    overwrite: bool = True,
) -> Optional[Dict[str, Any]]:
    """Materialize model weights on disk for worker loading.

    Returns the (possibly computed) DCP state_dict for reuse.
    """
    m_sd: Optional[Dict[str, Any]] = None

    if not (save_dcp or save_pt):
        return None

    os.makedirs(out_dir, exist_ok=True)

    # Only build the expensive full state_dict when required.
    if save_dcp:
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        m_sd = get_model_state_dict(model, options=opts)

    if save_dcp and m_sd is not None:
        save(
            state_dict={"model": m_sd},
            storage_writer=FileSystemWriter(out_dir, sync_files=True, overwrite=bool(overwrite)),
        )

    if save_pt:
        if m_sd is not None:
            pt_state: Dict[str, Any] = dict(m_sd)
            _trim_dcp_keys(pt_state)
        else:
            # Keep this best-effort and CPU-only.
            pt_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(pt_state, os.path.join(out_dir, "model.pt"))

    return m_sd


# -----------------------------
# JSON helpers + MemoryMappedTensor sidecar
# -----------------------------

def read_json(path: str) -> Any:
    """Read and parse JSON payload from ``path``.

    This thin wrapper exists for symmetry with ``write_json_atomic`` and to avoid
    sprinkling ``json.load`` calls throughout prediction assembly helpers. Using
    ``utf-8-sig`` tolerates BOM-prefixed files (common on Windows) when manifests
    are created or edited outside Python.
    """

    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json_atomic(path: str, payload: Any) -> None:
    """Atomically write JSON using BatchIterator's shared writer."""

    BatchIterator.atomic_write_json(path, payload, indent=None)


def _open_pred_memmap(mmt_path: str) -> Optional[MemoryMappedTensor]:
    """Open an existing MemoryMappedTensor using the sidecar meta file."""
    meta_path = BatchIterator.mmt_meta_path(mmt_path)
    if not (os.path.isfile(mmt_path) and os.path.isfile(meta_path)):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f) or {}
        dtype = parse_torch_dtype(meta.get("dtype")) or torch.float32
        shape = tuple(int(x) for x in (meta.get("shape") or ()))
        if not isinstance(dtype, torch.dtype) or not shape:
            return None
        return MemoryMappedTensor.from_filename(filename=mmt_path, dtype=dtype, shape=torch.Size(shape))
    except Exception:
        return None



# -----------------------------
# Data normalization helpers
# -----------------------------

def _iter_datasets(data: Any) -> tuple[list[tuple[str, Any]], Optional[dict[str, str] | list[str]]]:
    """Normalize data input into an iterable of (key, dataset_obj) plus optional manifest."""
    from collections.abc import Mapping as _Mapping

    if isinstance(data, TensorDictBase):
        return [("0", data)], None

    elif isinstance(data, _Mapping) and data and all(isinstance(v, _Mapping) for v in data.values()):
        man: dict[str, str] = {}
        items: list[tuple[str, Any]] = []
        for k, d in data.items():
            key = str(k)
            items.append((key, d))
            man[key] = key
        return items, man

    elif isinstance(data, Sequence) and data and all(isinstance(d, _Mapping) for d in data):
        man2: list[str] = []
        items2: list[tuple[str, Any]] = []
        for i, d in enumerate(data):
            key = str(i)
            items2.append((key, d))
            man2.append(key)
        return items2, man2

    return [("0", data)], None


# -----------------------------
# train() helpers (kept module-level to avoid nested defs)
# -----------------------------


def _check_shapes(
    first_in_dim: Optional[int],
    in_dim: int,
    first_label_shape: Tuple[int, ...],
    lshape: Tuple[int, ...],
) -> Tuple[Optional[int], Tuple[int, ...]]:
    """Validate that all datasets share the same feature/label shapes."""
    if first_in_dim is None:
        return int(in_dim), tuple(lshape)
    if int(in_dim) != int(first_in_dim) or tuple(lshape) != tuple(first_label_shape):
        raise RuntimeError(
            f"Shape mismatch across datasets: expected X_dim={first_in_dim}, y_shape={first_label_shape}, "
            f"got X_dim={in_dim}, y_shape={lshape}"
        )
    return first_in_dim, first_label_shape



def _mat_one(
    d: Any,
    out_dir: str,
    *,
    ds: Dataset,
    val_frac: float,
    seed_value: Optional[int],
    underflow_action: Any,
    shuffle: bool,
) -> Tuple[int, Tuple[int, ...], int]:
    """Materialize one dataset into the on-disk memmap format used by runtime workers."""
    from collections.abc import Mapping as _Mapping

    if isinstance(d, TensorDictBase):
        td = d
        if td.batch_size is None or len(td.batch_size) == 0:
            raise ValueError("TensorDict input to train() must have a batch dimension.")
        count = int(td.batch_size[0])
        if count <= 0:
            raise ValueError("Empty TensorDict provided to train().")

        in_dim, label_shape = BatchIterator.write_memmap_streaming_two_pass(
            ds=ds,
            out_dir=out_dir,
            count=count,
            get_batch=lambda s, e: td[s:e],
            get_by_indices=lambda idx: td[idx],
            val_frac=float(val_frac),
            seed_value=seed_value,
            underflow_action=underflow_action,
            shuffle=bool(shuffle),
            allow_missing_labels=False,
            chunk_size=0,
        )
        return int(in_dim), tuple(label_shape), int(count)

    if (
        isinstance(d, _Mapping)
        and d
        and all(not isinstance(v, _Mapping) for v in d.values())
        and not BatchIterator.is_feature_label_batch_mapping(d)
    ):
        # Key-index mappings can be huge; do not materialize keys.
        # Shuffle should be handled by the sampler, not the data writer.
        count, _get_batch = BatchIterator.key_index_mapping_getters(d)
        if count <= 0:
            raise ValueError("Empty dataset provided to train().")

        in_dim, label_shape = BatchIterator.write_memmap_streaming_two_pass(
            ds=ds,
            out_dir=out_dir,
            count=count,
            get_batch=_get_batch,
            get_by_indices=None,
            val_frac=float(val_frac),
            seed_value=seed_value,
            underflow_action=underflow_action,
            shuffle=False,
            allow_missing_labels=False,
            chunk_size=0,
        )
        return int(in_dim), tuple(label_shape), int(count)

    fx, lb, _, lshape = ds.preprocess(d)
    if not fx.is_contiguous():
        fx = fx.contiguous()
    if lb is None:
        raise ValueError("train() requires labels")
    count = int(fx.shape[0])
    if count <= 0:
        raise ValueError("Empty dataset provided to train().")
    in_dim = int(fx.reshape(count, -1).shape[1])

    preload_memmap(
        {"features": fx, "labels": lb},
        memmap_dir=out_dir,
        train_frac=1.0 - float(val_frac),
        val_frac=float(val_frac),
        shuffle=bool(shuffle),
        seed=seed_value,
        underflow_action=underflow_action,
    )
    del fx, lb
    return int(in_dim), tuple(lshape), int(count)



def _aggregate_run_stats(recs: List[Mapping[str, Any]]) -> Optional[Dict[str, float]]:
    """Reduce per-batch statistics into a single aggregate (weighted by batch_size)."""
    if not isinstance(recs, list) or not recs:
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
        "processed_n": float(total_bs),
        "sampled_x_mean": mean_x,
        "sampled_x_var": var_x,
        "sampled_x_min": x_min,
        "sampled_x_max": x_max,
        "sampled_y_mean": mean_y,
        "sampled_y_var": var_y,
        "sampled_y_min": y_min,
        "sampled_y_max": y_max,
    }



def _update_cum_stats(
    prev: Optional[Dict[str, float]],
    n_prev: int,
    inc: Optional[Dict[str, float]],
    n_inc: int,
) -> Optional[Dict[str, float]]:
    """Combine previous reduced stats with a new run's sampled stats."""
    if inc is None or n_inc <= 0:
        return prev
    if prev is None or n_prev <= 0:
        out: Dict[str, float] = {}
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


# -----------------------------
# Public API
# -----------------------------

def train(
    model: Root,
    data: (
        Dict[Tuple, torch.Tensor]
        | Sequence[Dict[Tuple, torch.Tensor]]
        | Mapping[str, Dict[Tuple, torch.Tensor]]
    ),
    *args: Any,
    epochs: int = 5,
    val_frac: float = 0.1,
    shuffle: bool = True,
    deterministic: bool = False,
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
) -> Root:
    _reset_process_group()

    try:
        val_frac = float(val_frac)
        val_frac = 0.0 if val_frac < 0.0 else (1.0 if val_frac > 1.0 else val_frac)
    except (TypeError, ValueError):
        val_frac = 0.1

    seed_value = _ensure_seed(seed)
    _seed_everything(seed_value)

    underflow_action = normalize_underflow_action(
        kwargs.pop("underflow_action", None),
        default=default_underflow_action(),
    )

    # Determinism settings are applied inside runtime workers.
    ds_meta = Dataset.for_device("cpu", feature_dtype=torch.float64, label_float_dtype=torch.float64)
    ds_meta.underflow_action = underflow_action

    initialize_python_path()
    mp.allow_connection_pickling()
    set_multiprocessing_env()

    memmap_dir = new_dir("memmap_ds")

    num_samples_dataset = 0
    first_in_dim: Optional[int] = None
    label_shape: Tuple[int, ...] = ()
    manifest: Optional[Dict[str, str] | Sequence[str]] = None
    ckpt_dir: Optional[str] = None
    init_dir: Optional[str] = None

    try:
        datasets, manifest = _iter_datasets(data)
        multi = manifest is not None

        for key, d in datasets:
            sub = memmap_dir if (not multi) else os.path.join(memmap_dir, key)
            if multi:
                os.makedirs(sub, exist_ok=True)
            in_dim, lshape, n = _mat_one(
                d,
                sub,
                ds=ds_meta,
                val_frac=float(val_frac),
                seed_value=seed_value,
                underflow_action=underflow_action,
                shuffle=bool(shuffle),
            )
            first_in_dim, label_shape = _check_shapes(first_in_dim, in_dim, label_shape, lshape)
            num_samples_dataset += int(n)

        if first_in_dim is None or not label_shape:
            raise RuntimeError("no training data provided to train()")

        if manifest is not None:
            with open(os.path.join(memmap_dir, "multinode.json"), "w", encoding="utf-8") as f:
                payload = manifest if isinstance(manifest, dict) else list(manifest)
                json.dump(payload, f)

        ckpt_dir = new_dir("ckpt_dcp")

        save_dcp = env_bool("STNET_SAVE_DCP", True)
        save_pt = env_bool("STNET_SAVE_MODEL_PT", True)
        if not (save_dcp or save_pt):
            save_pt = True

        m_sd: Optional[Dict[str, Any]] = None
        if save_dcp or save_pt:
            init_dir = new_dir("init_dcp")
            m_sd = _maybe_save_model_checkpoint(model, init_dir, save_dcp=save_dcp, save_pt=save_pt, overwrite=True)

        default_rdzv_host = get_preferred_ip(allow_loopback=True) or "127.0.0.1"
        resolved_rdzv = rdzv_endpoint if rdzv_endpoint else default_rdzv_host
        rdzv_endpoint = get_available_host(resolved_rdzv)
        master_addr, _master_port = initialize_master_addr(rdzv_endpoint)

        _wp = WorkerPolicy.autotune()
        _wp.apply_torch_threads()
        nprocs = int(_wp.nproc_per_node)

        cfg_obj = getattr(model, "_Root__config", None)
        if isinstance(cfg_obj, (ModelConfig, dict)):
            cfg_model = coerce_model_config(cfg_obj)
        else:
            cfg_model = ModelConfig()
        cfg_dict: Dict[str, Any] = model_config_to_dict(cfg_model)

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
            in_dim=int(first_in_dim),
            out_shape=tuple(label_shape),
            cfg_dict=cfg_dict,
        )
        if init_dir is not None:
            base["init_ckpt_dir"] = init_dir

        default_kwargs = {
            "epochs": epochs,
            "val_frac": val_frac,
            "shuffle": shuffle,
            "deterministic": deterministic,
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

        positional_names = RuntimeConfig.TRAIN_POS_ORDER[: len(args)]
        for key in list(default_kwargs):
            if key in positional_names or key in kwargs:
                default_kwargs.pop(key, None)

        ops = runtime_config("train", base, *args, **default_kwargs, **kwargs)

        with contextlib.suppress(Exception):
            model.to("cpu")
        _clear_device_caches()

        elastic_launch(lc, main)(ops)

        # Load final model weights back into this process.
        fallback = os.path.join(ckpt_dir, "model.pt")
        if os.path.isfile(fallback):
            cpu_state = _torch_load_checkpoint(
                fallback,
                map_location="cpu",
                weights_only=True,
            )
            cpu_state = _to_cpu(cpu_state)
            resize_scaler_buffer(model, cpu_state)
            model.load_state_dict(cpu_state, strict=False)
        else:
            opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
            m_sd = get_model_state_dict(model, options=opts)
            m_sd = _trim_dcp_keys(m_sd)
            load(state_dict={"model": m_sd}, storage_reader=FileSystemReader(ckpt_dir))
            resize_scaler_buffer(model, m_sd)
            set_model_state_dict(model, m_sd, options=StateDictOptions(strict=False))

        # Attach training history (best-effort).
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

                    run_stats = _aggregate_run_stats(records) if isinstance(records, list) and records else None

                    prev_total = int(getattr(model, "_history_total_samples", 0))
                    # Prefer recorded processed sample count; fall back to dataset size for compatibility.
                    inc_samples = int(run_stats.get("processed_n", 0)) if run_stats else int(num_samples_dataset)
                    if inc_samples <= 0:
                        inc_samples = int(num_samples_dataset)
                    new_total = prev_total + inc_samples

                    prev_cum = getattr(model, "_history_cum_stats", None)

                    cum_stats = _update_cum_stats(prev_cum, prev_total, run_stats, inc_samples)

                    setattr(model, "_history_total_samples", new_total)
                    setattr(model, "_history_dataset_n", int(num_samples_dataset))
                    if cum_stats is not None:
                        setattr(model, "_history_cum_stats", cum_stats)

                    run_hist_prev = getattr(model, "_train_history", None)
                    run_index = len(run_hist_prev) if isinstance(run_hist_prev, list) else 0

                    run_record: Dict[str, Any] = {
                        "run_index": run_index,
                        "dataset_n": int(num_samples_dataset),
                        "processed_n": int(inc_samples),
                        "reduced_n": new_total,
                    }
                    if run_stats is not None:
                        # do not duplicate processed_n twice
                        for k, v in run_stats.items():
                            if k != "processed_n":
                                run_record[k] = v
                    if cum_stats is not None:
                        run_record.update(cum_stats)
                    if isinstance(meta, dict) and meta:
                        run_record["env"] = dict(meta)

                    new_run_hist = (run_hist_prev + [run_record]) if isinstance(run_hist_prev, list) else [run_record]
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


def _normalize_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    p = str(path).strip()
    if not p:
        return None
    # Treat common "null" strings as None.
    if p.lower() in ("none", "null", "nil"):
        return None
    return os.path.abspath(os.path.expanduser(p))


def _torch_dtype_to_numpy(dtype: torch.dtype):
    import numpy as _np

    # Prefer not to upcast unknown float dtypes to float64: it is expensive and
    # often doubles memory use with no benefit.
    np_bfloat16 = getattr(_np, "bfloat16", _np.float32)

    mapping = {
        torch.float16: _np.float16,
        torch.float32: _np.float32,
        torch.float64: _np.float64,
        torch.bfloat16: np_bfloat16,
        torch.int8: _np.int8,
        torch.uint8: _np.uint8,
        torch.int16: _np.int16,
        torch.int32: _np.int32,
        torch.int64: _np.int64,
        torch.bool: _np.bool_,
    }
    return mapping.get(dtype, _np.float32)


def _read_memmap_meta(memmap_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(memmap_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"memmap meta.json not found: {meta_path}")
    meta = read_json(meta_path)
    if not isinstance(meta, dict):
        raise ValueError(f"memmap meta.json malformed: {meta_path}")
    return meta


def _open_features_mmt(memmap_dir: str) -> "MemoryMappedTensor":
    if MemoryMappedTensor is None:
        raise ImportError(
            "tensordict is required for MemoryMappedTensor-backed inference outputs. "
            "Please install 'tensordict'."
        )

    meta = _read_memmap_meta(memmap_dir)
    N = int(meta.get("N", 0))
    if N <= 0:
        raise ValueError(f"memmap meta.json under {memmap_dir} has non-positive N={N}")

    feat_rel = str(meta.get("features_path", "features.mmt"))
    feat_path = os.path.join(memmap_dir, feat_rel)

    fdim = int(meta.get("feature_dim", 0))
    if fdim <= 0:
        raise ValueError(f"memmap meta.json under {memmap_dir} has non-positive feature_dim={fdim}")

    f_dtype = dtype_from_name(meta.get("features_dtype", "float64"), torch.float64)

    return MemoryMappedTensor.from_filename(
        feat_path,
        dtype=f_dtype,
        shape=torch.Size([N, fdim]),
    )


def _is_writable_file_path(path: str) -> bool:
    try:
        path = os.path.abspath(os.path.expanduser(path))
        parent = os.path.dirname(path) or os.getcwd()
        os.makedirs(parent, exist_ok=True)
        test_path = path + ".__stnet_write_test__"
        with open(test_path, "wb") as f:
            f.write(b"0")
        os.remove(test_path)
        return True
    except Exception:
        return False


def _infer_pred_master_dtype(chunks_dir: str, *, default: torch.dtype = torch.float32) -> torch.dtype:
    """Infer the prediction dtype from the first available chunk.

    Fixed-shape inference writes .mmt parts with a small .meta.json sidecar.
    Variable-shape inference uses .pt parts; we then load one part to inspect dtype.
    """

    manifest_path = os.path.join(chunks_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return default

    try:
        manifest = read_json(manifest_path)
    except Exception:
        return default

    parts = manifest.get("parts") if isinstance(manifest, dict) else None
    if not isinstance(parts, list) or not parts:
        return default

    for part in parts:
        if not isinstance(part, dict):
            continue
        pred_rel = part.get("pred")
        if not pred_rel:
            continue

        pred_path = os.path.join(chunks_dir, str(pred_rel))

        # Fast path: memmap chunk with sidecar meta.
        if pred_path.endswith(".mmt"):
            meta_path = BatchIterator.mmt_meta_path(pred_path)
            if os.path.isfile(meta_path):
                try:
                    meta = read_json(meta_path)
                except Exception:
                    meta = None
                if isinstance(meta, dict):
                    dt = parse_torch_dtype(meta.get("dtype"))
                    if isinstance(dt, torch.dtype):
                        return dt

        # Fallback: torch chunk (loads into RAM).
        if os.path.isfile(pred_path):
            try:
                preds_t = _torch_load_checkpoint(
                    pred_path,
                    map_location="cpu",
                    weights_only=True,
                )
            except Exception:
                preds_t = None
            if isinstance(preds_t, torch.Tensor):
                return preds_t.dtype
            try:
                return torch.as_tensor(preds_t).dtype
            except Exception:
                continue

    return default


def _assemble_predictions_to_memmap(
    chunks_dir: str,
    out_path: str,
    *,
    count: int,
    out_shape: Sequence[int],
    store_float: torch.dtype,
) -> "MemoryMappedTensor":
    if MemoryMappedTensor is None:
        raise ImportError(
            "tensordict is required for MemoryMappedTensor-backed prediction assembly. "
            "Please install 'tensordict'."
        )

    out_shape_t = tuple(int(x) for x in out_shape)
    full_shape = torch.Size([int(count), *out_shape_t])

    Y_out = MemoryMappedTensor.empty(full_shape, dtype=store_float, filename=out_path, existsok=True)

    manifest_path = os.path.join(chunks_dir, "manifest.json")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid manifest: {manifest_path}")

    variable_shape = bool(manifest.get("variable_shape", False))
    if variable_shape:
        raise NotImplementedError(
            "Variable-shaped predictions cannot be assembled into a single dense MemoryMappedTensor. "
            "Please rerun with a fixed output shape, or set lazy=False and handle per-sample tensors."
        )

    parts = list(manifest.get("parts", []))
    for part in parts:
        rows_file = os.path.join(chunks_dir, str(part["rows"]))
        pred_file = os.path.join(chunks_dir, str(part["pred"]))

        rows_t = _torch_load_checkpoint(
            rows_file,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(rows_t, torch.Tensor):
            rows_t = torch.as_tensor(rows_t, device="cpu")
        rows_t = rows_t.to(dtype=torch.int64, device="cpu")

        if pred_file.endswith(".mmt"):
            preds_t = _open_pred_memmap(pred_file)
        else:
            preds_t = _torch_load_checkpoint(
                pred_file,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(preds_t, torch.Tensor):
                preds_t = torch.as_tensor(preds_t, device="cpu")

        preds_t = preds_t.to(dtype=store_float, device="cpu")

        if preds_t.shape[0] != rows_t.shape[0]:
            raise ValueError(
                f"Pred/rows mismatch in {pred_file}: preds[0]={preds_t.shape[0]} vs rows={rows_t.shape[0]}"
            )

        Y_out.index_copy_(0, rows_t, preds_t)

    pred_meta_path = BatchIterator.mmt_meta_path(out_path)
    write_json_atomic(
        pred_meta_path,
        {
            "dtype": str(store_float).replace("torch.", ""),
            "shape": list(map(int, full_shape)),
        },
    )

    return Y_out


def _assemble_predictions_to_tensor(
    chunks_dir: str,
    *,
    count: int,
    out_shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    out_shape_t = tuple(int(x) for x in out_shape)
    Y_out = torch.empty((int(count), *out_shape_t), dtype=dtype, device="cpu")

    manifest_path = os.path.join(chunks_dir, "manifest.json")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid manifest: {manifest_path}")

    variable_shape = bool(manifest.get("variable_shape", False))
    if variable_shape:
        raise NotImplementedError(
            "Variable-shaped predictions cannot be returned as a single dense Tensor. "
            "Please rerun with a fixed output shape."
        )

    parts = list(manifest.get("parts", []))
    for part in parts:
        rows_file = os.path.join(chunks_dir, str(part["rows"]))
        pred_file = os.path.join(chunks_dir, str(part["pred"]))

        rows_t = _torch_load_checkpoint(
            rows_file,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(rows_t, torch.Tensor):
            rows_t = torch.as_tensor(rows_t, device="cpu")
        rows_t = rows_t.to(dtype=torch.int64, device="cpu")

        if pred_file.endswith(".mmt"):
            preds_t = _open_pred_memmap(pred_file)
        else:
            preds_t = _torch_load_checkpoint(
                pred_file,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(preds_t, torch.Tensor):
                preds_t = torch.as_tensor(preds_t, device="cpu")

        preds_t = preds_t.to(dtype=dtype, device="cpu")

        if preds_t.shape[0] != rows_t.shape[0]:
            raise ValueError(
                f"Pred/rows mismatch in {pred_file}: preds[0]={preds_t.shape[0]} vs rows={rows_t.shape[0]}"
            )

        Y_out.index_copy_(0, rows_t, preds_t)

    return Y_out


def _write_predictions_h5_from_chunks(
    out_path: str,
    *,
    memmap_dir: str,
    chunks_dir: str,
    count: int,
    out_shape: Sequence[int],
    store_float: torch.dtype,
    chunk_size: int = 8192,
) -> "PersistentTensorDict":
    if PersistentTensorDict is None:
        raise ImportError(
            "tensordict is required for PersistentTensorDict outputs. Please install 'tensordict'."
        )

    import h5py
    import numpy as np

    X_mmt = _open_features_mmt(memmap_dir)

    out_shape_t = tuple(int(x) for x in out_shape)

    os.makedirs(os.path.dirname(out_path) or os.getcwd(), exist_ok=True)

    np_float = _torch_dtype_to_numpy(store_float)
    cast_dtype = store_float
    # NumPy may not support bfloat16 depending on version.
    if store_float == torch.bfloat16 and np_float == np.float32:
        cast_dtype = torch.float32

    with h5py.File(out_path, "w") as f:
        dset_X = f.create_dataset("X", shape=tuple(X_mmt.shape), dtype=_torch_dtype_to_numpy(X_mmt.dtype))
        dset_Y = f.create_dataset("Y", shape=(int(count), *out_shape_t), dtype=np_float)

        chunk = int(chunk_size)
        for s in range(0, int(count), chunk):
            e = min(int(count), s + chunk)
            x_slice = X_mmt[s:e]
            dset_X[s:e] = x_slice.detach().to(device="cpu").numpy()

        manifest = read_json(os.path.join(chunks_dir, "manifest.json"))
        if not isinstance(manifest, dict):
            raise ValueError(f"Invalid manifest under: {chunks_dir}")

        if bool(manifest.get("variable_shape", False)):
            raise NotImplementedError(
                "Variable-shaped predictions cannot be stored as a dense HDF5 dataset. "
                "Please rerun with a fixed output shape."
            )

        parts = list(manifest.get("parts", []))
        for part in parts:
            rows_file = os.path.join(chunks_dir, str(part["rows"]))
            pred_file = os.path.join(chunks_dir, str(part["pred"]))

            rows_t = _torch_load_checkpoint(
                rows_file,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(rows_t, torch.Tensor):
                rows_t = torch.as_tensor(rows_t, device="cpu")
            rows_np = rows_t.detach().to(device="cpu", dtype=torch.int64).numpy()

            if pred_file.endswith(".mmt"):
                preds_t = _open_pred_memmap(pred_file)
            else:
                preds_t = _torch_load_checkpoint(
                    pred_file,
                    map_location="cpu",
                    weights_only=True,
                )
                if not isinstance(preds_t, torch.Tensor):
                    preds_t = torch.as_tensor(preds_t, device="cpu")

            preds_np = preds_t.detach().to(device="cpu", dtype=cast_dtype).numpy()

            if preds_np.shape[0] != rows_np.shape[0]:
                raise ValueError(
                    f"Pred/rows mismatch in {pred_file}: preds[0]={preds_np.shape[0]} vs rows={rows_np.shape[0]}"
                )

            dset_Y[rows_np] = preds_np

    return PersistentTensorDict(filename=out_path, batch_size=[int(count)], mode="r")


def _write_predictions_h5_from_memmaps(
    out_path: str,
    *,
    memmap_dir: str,
    pred_path: str,
    count: Optional[int] = None,
    chunk_size: int = 8192,
) -> PersistentTensorDict:
    """Write an HDF5 PersistentTensorDict from assembled memmaps."""

    if not _is_writable_file_path(out_path):
        raise ValueError(f"persist_path is not a writable file path: {out_path}")

    import numpy as _np
    import h5py

    X_mmt = _open_features_mmt(memmap_dir)
    Y_mmt = _open_pred_memmap(pred_path)
    if Y_mmt is None:
        raise FileNotFoundError(f"missing prediction memmap: {pred_path}")

    n = int(Y_mmt.shape[0]) if count is None else int(count)
    if n <= 0:
        raise ValueError(f"non-positive prediction count: {n}")

    x_np_dtype = _torch_dtype_to_numpy(X_mmt.dtype)
    y_np_dtype = _torch_dtype_to_numpy(Y_mmt.dtype)

    # If numpy doesn't support bfloat16, write float32 instead.
    y_cast_dtype = Y_mmt.dtype
    if Y_mmt.dtype == torch.bfloat16 and y_np_dtype == _np.float32:
        y_cast_dtype = torch.float32

    out_parent = os.path.dirname(out_path) or "."
    os.makedirs(out_parent, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        dset_X = f.create_dataset(
            "X",
            shape=(n, int(X_mmt.shape[1])),
            dtype=x_np_dtype,
            chunks=(min(n, int(chunk_size)), int(X_mmt.shape[1])),
        )
        dset_Y = f.create_dataset(
            "Y",
            shape=(n, *[int(x) for x in Y_mmt.shape[1:]]),
            dtype=y_np_dtype,
            chunks=(min(n, int(chunk_size)), *[int(x) for x in Y_mmt.shape[1:]]),
        )

        for s in range(0, n, int(chunk_size)):
            e = min(n, s + int(chunk_size))
            dset_X[s:e] = X_mmt[s:e].detach().to(device="cpu", dtype=X_mmt.dtype).numpy()
            dset_Y[s:e] = Y_mmt[s:e].detach().to(device="cpu", dtype=y_cast_dtype).numpy()

    return PersistentTensorDict(filename=out_path, batch_size=[int(n)], mode="r")

@catchtime(logger, fn_name="predict")
def predict(
    data: Any,
    *args: Any,
    model: Optional[torch.nn.Module] = None,
    mode: OpsMode = "predict",
    seed: int = 0,
    shuffle: bool = False,
    max_nodes: Optional[int] = None,
    rdzv_endpoint: Optional[str] = None,
    rdzv_backend: Optional[str] = None,
    persist_path: Optional[str] = None,
    **kwargs: Any,
) -> TensorDictBase | PersistentTensorDict:
    """Run distributed inference and return predictions.

    By default this returns a lazy TensorDict backed by memmaps. Set ``lazy=False``
    to materialize tensors in-memory. If ``persist_path`` is provided, an HDF5
    PersistentTensorDict is written instead.
    """

    if model is None:
        raise ValueError("predict: model must not be None")

    # Ensure a clean distributed state.
    _reset_process_group()
    initialize_python_path()
    set_multiprocessing_env()

    # Required fixed output shape (fixed-shape inference uses .mmt parts).
    out_shape = kwargs.pop("out_shape", getattr(model, "out_shape", None))
    if out_shape is None:
        raise ValueError(
            "predict: out_shape is required (pass out_shape=... or set model.out_shape)"
        )
    out_shape_t = tuple(int(x) for x in out_shape)
    if not out_shape_t or any(int(x) <= 0 for x in out_shape_t):
        raise ValueError(f"predict: invalid out_shape={out_shape!r}")

    # Input materialization settings.
    underflow_action = kwargs.pop("underflow_action", default_underflow_action())
    chunk_size = kwargs.pop("chunk_size", None)
    lazy = bool(kwargs.pop("lazy", True))
    writer_chunk_size = int(chunk_size) if chunk_size is not None else 8192

    def _infer_master_float_dtype(obj: Any) -> torch.dtype:
        """Best-effort dtype inference for predict() materialization.

        We prefer to preserve float64 inputs (avoid silent precision loss) while
        keeping the default path on float32 to reduce CPU memory and copy cost.
        """

        def _coerce(dt: Any) -> torch.dtype | None:
            try:
                if isinstance(dt, torch.dtype):
                    return dt
                if dt is None:
                    return None
                # numpy dtype
                ndt = np.dtype(dt)
                if ndt == np.float64:
                    return torch.float64
                if ndt == np.float32:
                    return torch.float32
                if ndt == np.float16:
                    return torch.float16
            except Exception:
                return None
            return None

        try:
            if TensorDictBase is not None and isinstance(obj, TensorDictBase):
                X_td, _ = extract_xy(obj, labels_required=False)
                if X_td is None:
                    return torch.float32
                if not bool(torch.is_tensor(X_td)):
                    X_td = torch.as_tensor(X_td)
                dt = _coerce(getattr(X_td, "dtype", None))
                return torch.float64 if dt == torch.float64 else torch.float32

            if isinstance(obj, Mapping) and BatchIterator.is_feature_label_batch_mapping(obj):
                f_key = resolve_feature_key(obj)
                if f_key is not None and f_key in obj:
                    X_all = obj.get(f_key)
                    dt = _coerce(getattr(X_all, "dtype", None))
                    if dt is not None:
                        return torch.float64 if dt == torch.float64 else torch.float32
                    if isinstance(X_all, (list, tuple)) and X_all:
                        dt0 = _coerce(getattr(X_all[0], "dtype", None))
                        return torch.float64 if dt0 == torch.float64 else torch.float32

            if torch.is_tensor(obj):
                dt = _coerce(obj.dtype)
                return torch.float64 if dt == torch.float64 else torch.float32

            if isinstance(obj, np.ndarray):
                dt = _coerce(obj.dtype)
                return torch.float64 if dt == torch.float64 else torch.float32
        except Exception:
            pass
        return torch.float32

    master_dtype = _infer_master_float_dtype(data)
    ds = Dataset.for_device("cpu", feature_dtype=master_dtype, label_float_dtype=master_dtype)
    ds.underflow_action = underflow_action

    tmp_dir = new_dir("infer")
    ckpt_dir = os.path.join(tmp_dir, "ckpt")
    memmap_dir = os.path.join(tmp_dir, "memmap")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(memmap_dir, exist_ok=True)

    inference_ctx = Gradient.inference(model)
    with inference_ctx:
        try:
            # Save model checkpoint for the worker process group.
            dcp_dir = os.path.join(ckpt_dir, "model")
            save_dcp = env_bool("STNET_SAVE_DCP", True)
            save_pt = env_bool("STNET_SAVE_MODEL_PT", True)
            if not (save_dcp or save_pt):
                save_pt = True
            _maybe_save_model_checkpoint(
                model,
                dcp_dir,
                save_dcp=save_dcp,
                save_pt=save_pt,
                overwrite=True,
            )

            # Materialize input data to memmap (features-only).
            count: int
            in_dim: int

            # 1) TensorDict input
            if TensorDictBase is not None and isinstance(data, TensorDictBase):
                X_td, _ = extract_xy(data, labels_required=False)
                if X_td is None:
                    raise ValueError("predict: failed to extract features from TensorDict")
                if not bool(torch.is_tensor(X_td)):
                    X_td = torch.as_tensor(X_td)
                if X_td.ndim < 2:
                    X_td = X_td.view(-1, 1)
                count = int(X_td.shape[0])

                def _get_batch(s: int, e: int) -> Mapping[str, Any]:
                    return data[s:e]

                in_dim, _ = BatchIterator.write_memmap_streaming_two_pass(
                    ds=ds,
                    out_dir=memmap_dir,
                    count=count,
                    get_batch=_get_batch,
                    get_by_indices=None,
                    val_frac=0.0,
                    seed_value=int(seed),
                    underflow_action=underflow_action,
                    shuffle=False,
                    allow_missing_labels=True,
                    features_only=True,
                    chunk_size=writer_chunk_size,
                )

            # 2) Feature/label mapping input
            elif isinstance(data, Mapping) and BatchIterator.is_feature_label_batch_mapping(data):
                f_key = resolve_feature_key(data)
                if f_key is None:
                    raise ValueError("predict: could not resolve feature key from mapping")
                X_all = data[f_key]
                try:
                    count = int(len(X_all))
                except Exception:
                    count = int(getattr(X_all, "shape", [0])[0] or 0)
                if count <= 0:
                    raise ValueError("predict: empty input")

                def _get_batch(s: int, e: int) -> Mapping[str, Any]:
                    batch: dict[str, Any] = {}
                    for k, v in data.items():
                        # Slice per-sample tensors/arrays, keep scalars as-is.
                        try:
                            if hasattr(v, "__len__") and int(len(v)) == count:
                                batch[k] = v[s:e]
                            else:
                                batch[k] = v
                        except Exception:
                            batch[k] = v
                    return batch

                in_dim, _ = BatchIterator.write_memmap_streaming_two_pass(
                    ds=ds,
                    out_dir=memmap_dir,
                    count=count,
                    get_batch=_get_batch,
                    get_by_indices=None,
                    val_frac=0.0,
                    seed_value=int(seed),
                    underflow_action=underflow_action,
                    shuffle=False,
                    allow_missing_labels=True,
                    features_only=True,
                    chunk_size=writer_chunk_size,
                )

            # 3) Compact key-index mapping input (no pre-materialization of keys)
            elif (
                isinstance(data, Mapping)
                and data
                and all(not isinstance(v, Mapping) for v in data.values())
                and not BatchIterator.is_feature_label_batch_mapping(data)
            ):
                count, _get_batch = BatchIterator.key_index_mapping_getters(data)
                in_dim, _ = BatchIterator.write_memmap_streaming_two_pass(
                    ds=ds,
                    out_dir=memmap_dir,
                    count=count,
                    get_batch=_get_batch,
                    get_by_indices=None,
                    val_frac=0.0,
                    seed_value=int(seed),
                    underflow_action=underflow_action,
                    shuffle=False,
                    allow_missing_labels=True,
                    features_only=True,
                    chunk_size=writer_chunk_size,
                )

            # 4) Fallback: preprocess then preload to memmap
            else:
                fx, _lb, _keys, _lshape = ds.preprocess(data)
                if fx is None:
                    raise ValueError("predict: preprocess returned no features")
                if not bool(torch.is_tensor(fx)):
                    fx = torch.as_tensor(fx)
                if fx.ndim < 2:
                    fx = fx.view(-1, 1)
                count = int(fx.shape[0])
                in_dim = int(fx.shape[1])
                preload_memmap(
                    {"features": fx},
                    memmap_dir=memmap_dir,
                    val_frac=0.0,
                    shuffle=False,
                    seed=int(seed),
                    chunk_size=chunk_size,
                    allow_missing_labels=True,
                    features_only=True,
                    underflow_action=underflow_action,
                )

            if count <= 0:
                raise ValueError("predict: empty input")

            # Model config dict (optional).
            cfg_obj = getattr(model, "_Root__config", None)
            if isinstance(cfg_obj, ModelConfig):
                cfg_dict: dict[str, Any] = cfg_obj.to_dict()
            elif isinstance(cfg_obj, Mapping):
                cfg_dict = dict(cfg_obj)
            else:
                cfg_dict = {}

            base: dict[str, Any] = {
                "sources": {"kind": "memmap", "path": memmap_dir},
                "ckpt_dir": ckpt_dir,
                "model_ckpt_dir": dcp_dir,
                "in_dim": int(in_dim),
                "out_shape": out_shape_t,
                "cfg_dict": cfg_dict,
            }

            # Only pass supported runtime-config keys.
            keys = kwargs.pop("keys", None)
            loss_skew = kwargs.pop("loss_skew", None)
            ops_kwargs: dict[str, Any] = {"seed": int(seed)}
            if keys is not None:
                ops_kwargs["keys"] = keys
            if loss_skew is not None:
                ops_kwargs["loss_skew"] = float(loss_skew)

            # Launch config knobs (not part of RuntimeConfig).
            run_id = str(kwargs.pop("run_id", f"predict-{os.getpid()}-{int(seed)}"))

            # Fail fast on any leftover kwargs (previously these would have crashed inside runtime_config).
            if kwargs:
                raise ValueError(f"predict: unsupported kwargs: {sorted(kwargs.keys())}")

            ops = runtime_config(mode, base, *args, **ops_kwargs)

            # Distributed rendezvous.
            _wp = WorkerPolicy.autotune()
            _wp.apply_torch_threads()
            nprocs = int(_wp.nproc_per_node)
            max_nodes_i = max(1, int(max_nodes) if max_nodes is not None else 1)

            resolved_rdzv = rdzv_endpoint or get_preferred_ip()
            resolved_rdzv = get_available_host(resolved_rdzv)
            master_addr, _ = initialize_master_addr(resolved_rdzv)

            lc = LaunchConfig(
                min_nodes=1,
                max_nodes=max_nodes_i,
                nproc_per_node=nprocs,
                rdzv_backend=str(rdzv_backend or "c10d"),
                rdzv_endpoint=resolved_rdzv,
                run_id=run_id,
                max_restarts=0,
                monitor_interval=5,
                start_method=optimal_start_method(),
                local_addr=master_addr,
                redirects=Std.NONE,
                tee=Std.NONE,
            )

            # Reduce peak GPU/CPU memory before forking workers.
            with contextlib.suppress(Exception):
                model.to("cpu")
            _clear_device_caches()

            elastic_launch(lc, main)(ops)

            chunks_dir = os.path.join(ckpt_dir, "pred_chunks")
            if not os.path.isdir(chunks_dir):
                raise RuntimeError(f"predict: missing pred_chunks at {chunks_dir!r}")

            master_dtype = _infer_pred_master_dtype(chunks_dir)

            # Persist to HDF5.
            if persist_path is not None:
                persist_path_n = _normalize_path(persist_path)
                if persist_path_n is None:
                    raise ValueError("predict: persist_path is empty/None after normalization")
                if not _is_writable_file_path(persist_path_n):
                    raise ValueError(f"predict: persist_path is not writable: {persist_path_n!r}")
                return _write_predictions_h5_from_chunks(
                    persist_path_n,
                    chunks_dir=chunks_dir,
                    memmap_dir=memmap_dir,
                    count=count,
                    out_shape=out_shape_t,
                    store_float=master_dtype,
                    chunk_size=int(chunk_size or 8192),
                )

            # Lazy memmap output.
            if lazy:
                final_dir = new_dir("predictions")
                moved_memmap_dir = os.path.join(final_dir, "memmap")
                shutil.move(memmap_dir, moved_memmap_dir)
                pred_mmt_path = os.path.join(final_dir, "pred.mmt")
                _assemble_predictions_to_memmap(
                    chunks_dir=chunks_dir,
                    out_path=pred_mmt_path,
                    count=count,
                    out_shape=out_shape_t,
                    store_float=master_dtype,
                )

                X_mmt = _open_features_mmt(moved_memmap_dir)
                Y_mmt = _open_pred_memmap(pred_mmt_path)
                if Y_mmt is None:
                    raise RuntimeError("predict: failed to open assembled pred.mmt")
                return TensorDict({"X": X_mmt[:count], "Y": Y_mmt[:count]}, batch_size=[count])

            # In-memory output.
            Y_t = _assemble_predictions_to_tensor(
                chunks_dir=chunks_dir,
                count=count,
                out_shape=out_shape_t,
                dtype=master_dtype,
            )
            X_mmt = _open_features_mmt(memmap_dir)
            X_t = X_mmt[:count].detach().cpu().clone()
            return TensorDict({"X": X_t, "Y": Y_t}, batch_size=[count])

        finally:
            remove_dir(tmp_dir)
@catchtime(logger, fn_name="get_prediction")
def get_prediction(
    source: str,
    *,
    lazy: bool = True,
    persist_path: Optional[str] = None,
) -> TensorDictBase | PersistentTensorDict:
    """Load prediction artifacts.

    Supported sources:
      - A .h5 file created by STNet: returns PersistentTensorDict (read-only).
      - A directory containing `memmap/` and either:
          * `pred.mmt` (+ sidecar `.meta.json`), or
          * `pred_chunks/manifest.json` (and per-part tensors).

    If `persist_path` is provided, an HDF5 PersistentTensorDict is written and
    returned.
    """

    with Gradient.inference(torch.nn.Identity()):
        if not bool(source):
            raise ValueError("get_prediction: 'source' must be a non-empty path")

        src = _normalize_path(source)

        if src is None:
            raise ValueError("get_prediction: 'source' is empty/None after normalization")

        if persist_path is not None:
            persist_path = _normalize_path(persist_path)
            if persist_path is None:
                raise ValueError("get_prediction: persist_path is empty/None after normalization")
            if not _is_writable_file_path(persist_path):
                raise ValueError(f"persist_path is not writable: {persist_path!r}")

        # Direct .h5 load.
        if src.endswith(".h5") and os.path.isfile(src):
            return PersistentTensorDict(filename=src, mode="r")

        if not os.path.isdir(src):
            raise FileNotFoundError(f"source must be a directory or .h5 file: {src!r}")

        memmap_dir = os.path.join(src, "memmap")
        chunks_dir = os.path.join(src, "pred_chunks")
        pred_path = os.path.join(src, "pred.mmt")

        if not os.path.isdir(memmap_dir):
            raise FileNotFoundError(f"missing memmap dir: {memmap_dir!r}")

        # Persist to .h5 if requested.
        if persist_path is not None:
            if os.path.isfile(pred_path):
                return _write_predictions_h5_from_memmaps(
                    persist_path,
                    memmap_dir=memmap_dir,
                    pred_path=pred_path,
                )

            if os.path.isdir(chunks_dir):
                manifest = read_json(os.path.join(chunks_dir, "manifest.json"))
                out_shape = tuple(int(x) for x in (manifest.get("out_shape", []) or []))
                X_mmt = _open_features_mmt(memmap_dir)
                count = int(X_mmt.shape[0])
                store_float = _infer_pred_master_dtype(chunks_dir)
                return _write_predictions_h5_from_chunks(
                    persist_path,
                    chunks_dir=chunks_dir,
                    memmap_dir=memmap_dir,
                    count=count,
                    out_shape=out_shape,
                    store_float=store_float,
                )

            raise FileNotFoundError(f"missing prediction artifacts under: {src!r}")

        # No persist: return lazy memmaps (or materialize if requested).
        X_mmt = _open_features_mmt(memmap_dir)
        count = int(X_mmt.shape[0])

        if os.path.isfile(pred_path):
            Y_mmt = _open_pred_memmap(pred_path)
            if Y_mmt is None:
                raise RuntimeError(f"failed to open prediction memmap: {pred_path!r}")
            td_mm = TensorDict({"X": X_mmt, "Y": Y_mmt}, batch_size=[count])
            if not bool(lazy):
                return TensorDict(
                    {
                        "X": X_mmt.detach().cpu().clone(),
                        "Y": Y_mmt.detach().cpu().clone(),
                    },
                    batch_size=[count],
                )
            return td_mm

        if os.path.isdir(chunks_dir):
            manifest = read_json(os.path.join(chunks_dir, "manifest.json"))
            out_shape = tuple(int(x) for x in (manifest.get("out_shape", []) or []))
            store_float = _infer_pred_master_dtype(chunks_dir)
            _assemble_predictions_to_memmap(
                chunks_dir=chunks_dir,
                out_path=pred_path,
                count=count,
                out_shape=out_shape,
                store_float=store_float,
            )
            Y_mmt = _open_pred_memmap(pred_path)
            if Y_mmt is None:
                raise RuntimeError(f"failed to open prediction memmap: {pred_path!r}")
            td_mm = TensorDict({"X": X_mmt, "Y": Y_mmt}, batch_size=[count])
            if not bool(lazy):
                return TensorDict(
                    {
                        "X": X_mmt.detach().cpu().clone(),
                        "Y": Y_mmt.detach().cpu().clone(),
                    },
                    batch_size=[count],
                )
            return td_mm

        raise FileNotFoundError(f"No predictions found under: {src!r}")
