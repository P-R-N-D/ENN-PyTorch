# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import json
import os
import random
import shutil
import tempfile
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from tensordict import MemoryMappedTensor, TensorDictBase
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, load, save
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

try:
    from torch.distributed.run import LaunchConfig, elastic_launch
except ImportError:  # pragma: no cover
    from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from ..backend.distributed import get_available_host, get_preferred_ip, initialize_master_addr
from ..backend.runtime import _trim_dcp_keys, main
from ..backend.system import (
    WorkerPolicy,
    initialize_python_path,
    new_dir,
    optimal_start_method,
    set_multiprocessing_env,
)
from ..data.collections import LazyTensor
from ..data.datatype import env_bool
from ..data.nodes import preload_memmap
from ..data.pipeline import Dataset, default_underflow_action, normalize_underflow_action
from ..model.nn import History, Root, resize_scaler_buffer
from .config import ModelConfig, OpsMode, RuntimeConfig, coerce_model_config, model_config_to_dict, runtime_config


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
# Safer / more compatible loads
# -----------------------------

def _torch_load_cpu(path: str, *, weights_only: Optional[bool] = None) -> Any:
    """torch.load wrapper with CPU map + best-effort weights_only support."""
    # weights_only exists on modern PyTorch and is recommended for loading weights-only artifacts.
    # On older versions, passing the kwarg raises TypeError.
    if weights_only is not None:
        try:
            return torch.load(path, map_location="cpu", weights_only=bool(weights_only))
        except TypeError:
            return torch.load(path, map_location="cpu")
        except Exception:
            # Some files are not compatible with weights_only=True (e.g., keys lists).
            if weights_only:
                with contextlib.suppress(Exception):
                    return torch.load(path, map_location="cpu", weights_only=False)
            raise
    return torch.load(path, map_location="cpu")


def _preload_state(state: Any) -> Any:
    """Normalize a loaded state_dict for safe CPU-side use.

    - Ensures tensors are detached and resident on CPU.
    - Converts meta/fake tensors to real CPU tensors when encountered.
    - Makes tensors contiguous when possible.

    This is useful when user code provides a state dict (or torch.load result)
    that may contain non-standard tensor types.
    """
    from collections.abc import Mapping as _Mapping

    if isinstance(state, _Mapping):
        return {k: _preload_state(v) for k, v in state.items()}
    if isinstance(state, list):
        return [_preload_state(v) for v in state]
    if isinstance(state, tuple):
        seq = tuple(_preload_state(v) for v in state)
        if type(state) is tuple:
            return seq
        # namedtuple: constructor expects positional args.
        if hasattr(state, "_fields"):
            try:
                return type(state)(*seq)
            except Exception:
                return seq
        try:
            return type(state)(seq)
        except Exception:
            return seq

    if isinstance(state, torch.Tensor):
        t = state
        with contextlib.suppress(Exception):
            from ..backend.compat import is_meta_or_fake_tensor

            if is_meta_or_fake_tensor(t):
                t = torch.zeros(tuple(t.shape), dtype=t.dtype, device="cpu")

        if t.device.type != "cpu":
            t = t.detach().to(device="cpu")
        else:
            t = t.detach()

        with contextlib.suppress(Exception):
            t = t.contiguous()
        return t

    return state


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
# Prediction part memmap (B approach)
# -----------------------------

_PRED_MEMMAP_LOCK = threading.Lock()
_PRED_MEMMAP_PATH_LOCKS: dict[str, threading.Lock] = {}

def _pred_memmap_lock_for(path: str) -> threading.Lock:
    with _PRED_MEMMAP_LOCK:
        lk = _PRED_MEMMAP_PATH_LOCKS.get(path)
        if lk is None:
            lk = threading.Lock()
            _PRED_MEMMAP_PATH_LOCKS[path] = lk
        return lk


def _derive_mmt_path(pred_path: str) -> str:
    p = str(pred_path)
    if p.endswith(".mmt"):
        return p
    if p.endswith(".pt"):
        return p[:-3] + ".mmt"
    return p + ".mmt"


def _mmt_meta_path(mmt_path: str) -> str:
    return str(mmt_path) + ".meta.json"


def _mmt_lock_path(mmt_path: str) -> str:
    return str(mmt_path) + ".lock"


@contextlib.contextmanager
def _process_file_lock(lock_path: str) -> Any:
    """Best-effort cross-process exclusive lock.

    Uses:
    - POSIX: fcntl.flock
    - Windows: msvcrt.locking
    - Fallback: atomic mkdir spin-lock (only if neither file-lock API is available)
    """
    path = str(lock_path)
    parent = os.path.dirname(path) or "."
    with contextlib.suppress(Exception):
        os.makedirs(parent, exist_ok=True)

    # 1) POSIX: fcntl.flock
    try:
        import fcntl  # type: ignore

        f = open(path, "a+b")
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(Exception):
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            with contextlib.suppress(Exception):
                f.close()
        return
    except Exception:
        pass

    # 2) Windows: msvcrt.locking
    try:
        import msvcrt  # type: ignore

        f = open(path, "a+b")
        try:
            # Ensure file has at least 1 byte; msvcrt.locking requires a positive length.
            with contextlib.suppress(Exception):
                f.seek(0, os.SEEK_END)
                if f.tell() < 1:
                    f.write(b"0")
                    f.flush()
            with contextlib.suppress(Exception):
                f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            with contextlib.suppress(Exception):
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            with contextlib.suppress(Exception):
                f.close()
        return
    except Exception:
        pass

    # 3) Fallback: mkdir spin lock (best-effort).
    import time

    lock_dir = path + ".d"
    while True:
        try:
            os.mkdir(lock_dir)
            break
        except FileExistsError:
            time.sleep(0.05)
        except Exception:
            # If we cannot lock, proceed unlocked rather than hanging forever.
            yield
            return

    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            os.rmdir(lock_dir)


def _atomic_write_json(path: str, payload: Any) -> None:
    """Atomically write JSON to `path` (best-effort).

    This prevents corrupt sidecar metadata files when multiple processes/threads race or the process
    is interrupted mid-write.
    """
    p = str(path)
    parent = os.path.dirname(p) or "."
    os.makedirs(parent, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix=os.path.basename(p) + ".", suffix=".tmp", dir=parent)
    os.close(fd)
    try:
        with open(tmp_name, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_name, p)
    finally:
        with contextlib.suppress(Exception):
            os.remove(tmp_name)


def _parse_dtype(dtype_s: Any) -> Optional[torch.dtype]:
    if isinstance(dtype_s, torch.dtype):
        return dtype_s
    s = str(dtype_s or "").strip()
    if not s:
        return None
    if s.startswith("torch."):
        s = s.split(".", 1)[1]
    # Common aliases
    if s == "float":
        s = "float32"
    if s == "half":
        s = "float16"
    return getattr(torch, s, None)


def _ensure_pred_memmap_from_pt(pt_path: str, mmt_path: str) -> tuple[str, torch.dtype, Tuple[int, ...]]:
    """Create a MemoryMappedTensor file from a .pt Tensor if needed.

    - Thread-safe within a process.
    - Cross-process safe via a best-effort file lock.
    - Writes the sidecar meta.json atomically.
    - Writes the .mmt via a temp file + atomic replace to avoid half-written files.
    """
    lock = _pred_memmap_lock_for(mmt_path)
    with lock:
        # Cross-process lock around conversion + sidecar writes.
        with _process_file_lock(_mmt_lock_path(mmt_path)):
            meta_path = _mmt_meta_path(mmt_path)

            if os.path.isfile(mmt_path) and os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f) or {}
                    dtype = _parse_dtype(meta.get("dtype")) or torch.float32
                    shape = tuple(int(x) for x in (meta.get("shape") or ()))
                    if shape:
                        return mmt_path, dtype, shape
                except Exception:
                    # Fall through to re-create metadata if needed.
                    pass

            preds = _torch_load_cpu(pt_path, weights_only=True)
            if not isinstance(preds, torch.Tensor):
                preds = torch.as_tensor(preds)
            preds = preds.detach().cpu().contiguous()

            parent = os.path.dirname(str(mmt_path)) or "."
            os.makedirs(parent, exist_ok=True)

            fd, tmp_name = tempfile.mkstemp(
                prefix=os.path.basename(str(mmt_path)) + ".",
                suffix=".tmp",
                dir=parent,
            )
            os.close(fd)
            try:
                MemoryMappedTensor.from_tensor(preds, filename=tmp_name, existsok=True)
                os.replace(tmp_name, str(mmt_path))
            finally:
                with contextlib.suppress(Exception):
                    os.remove(tmp_name)

            meta = {"dtype": str(preds.dtype), "shape": list(preds.shape)}
            _atomic_write_json(meta_path, meta)

            return mmt_path, preds.dtype, tuple(int(x) for x in preds.shape)


def _open_pred_memmap(mmt_path: str) -> Optional[MemoryMappedTensor]:
    """Open an existing MemoryMappedTensor using the sidecar meta file."""
    meta_path = _mmt_meta_path(mmt_path)
    if not (os.path.isfile(mmt_path) and os.path.isfile(meta_path)):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f) or {}
        dtype = _parse_dtype(meta.get("dtype"))
        shape = tuple(int(x) for x in (meta.get("shape") or ()))
        if dtype is None or not shape:
            return None
        return MemoryMappedTensor.from_filename(filename=mmt_path, dtype=dtype, shape=torch.Size(shape))
    except Exception:
        return None





def _parse_size_bytes(value: Any) -> Optional[int]:
    """Parse a human-friendly byte size from an env var.

    Accepts:
      - plain integers (bytes)
      - suffixes: K, M, G, T (binary multiples, 1024^n)
        and also KB/MB/GB/TB (same)
      - underscores are ignored (e.g., 1_073_741_824)

    Returns None if the value is empty or cannot be parsed.
    """
    if value is None:
        return None
    s = str(value).strip().lower().replace("_", "")
    if not s:
        return None

    mult = 1
    for suf, m in (
        ("tb", 1024**4),
        ("t", 1024**4),
        ("gb", 1024**3),
        ("g", 1024**3),
        ("mb", 1024**2),
        ("m", 1024**2),
        ("kb", 1024**1),
        ("k", 1024**1),
        ("b", 1),
    ):
        if s.endswith(suf):
            mult = m
            s = s[: -len(suf)]
            break

    try:
        base = int(s)
    except Exception:
        return None
    if base < 0:
        return None
    try:
        return int(base * mult)
    except Exception:
        return None

def _alloc_row_map_arrays(nkeys: int, *, chunk_root: str) -> tuple[np.ndarray, np.ndarray]:
    """Allocate row->(part,offset) mapping arrays for get_prediction(lazy=True).

    Default: in-RAM int32 arrays (fastest).

    Optional sparse/out-of-core mode via env vars:
      - STNET_PRED_LAZY_ROW_MAP:
          * "sparse" | "memmap" | "disk" | "file" | "1" | "true" | ... -> force np.memmap-backed arrays on disk
          * "dense"  | "0" | "false" | "off" -> force in-RAM arrays (may OOM)
          * "auto" (default when unset) -> choose based on STNET_PRED_LAZY_ROW_MAP_MAX_BYTES, if provided

      - STNET_PRED_LAZY_ROW_MAP_MAX_BYTES:
          * only used when STNET_PRED_LAZY_ROW_MAP is "auto"/unset
          * if estimated bytes for (row_to_part,row_to_off) exceed this threshold -> use memmap
          * supports plain integers (bytes) or suffixes K/M/G/T (binary, 1024^n), e.g. "512M", "2G"

    Location for memmap row maps:
      - STNET_PRED_LAZY_ROW_MAP_DIR (directory). Defaults to chunk_root; falls back to system temp.
    """
    mode = str(os.environ.get("STNET_PRED_LAZY_ROW_MAP", "")).strip().lower()
    if not mode:
        mode = "auto"

    force_dense = mode in {"dense", "0", "false", "no", "n", "off"}
    want_sparse = (not force_dense) and mode in {"sparse", "memmap", "disk", "file", "1", "true", "yes", "y", "on"}

    if (not force_dense) and (not want_sparse) and mode in {"auto"}:
        max_bytes = _parse_size_bytes(os.environ.get("STNET_PRED_LAZY_ROW_MAP_MAX_BYTES", ""))
        if max_bytes is not None:
            need_bytes = int(2) * int(nkeys) * int(np.dtype(np.int32).itemsize)
            if need_bytes > int(max_bytes):
                want_sparse = True

    if not want_sparse:
        try:
            return (
                np.full((nkeys,), -1, dtype=np.int32),
                np.full((nkeys,), -1, dtype=np.int32),
            )
        except MemoryError:
            if force_dense:
                raise
            want_sparse = True

    # Disk-backed (np.memmap): avoids large contiguous heap allocations.
    dir_env = str(os.environ.get("STNET_PRED_LAZY_ROW_MAP_DIR", "")).strip()
    base_dir = dir_env if dir_env else str(chunk_root)
    try:
        os.makedirs(base_dir, exist_ok=True)
    except Exception:
        base_dir = tempfile.gettempdir()

    prefix = f"stnet_rowmap_{os.getpid()}_{threading.get_ident()}_"
    fd1, part_path = tempfile.mkstemp(prefix=prefix + "part_", suffix=".i32", dir=base_dir)
    os.close(fd1)
    fd2, off_path = tempfile.mkstemp(prefix=prefix + "off_", suffix=".i32", dir=base_dir)
    os.close(fd2)

    row_to_part = np.memmap(part_path, dtype=np.int32, mode="w+", shape=(nkeys,))
    row_to_off = np.memmap(off_path, dtype=np.int32, mode="w+", shape=(nkeys,))
    row_to_part.fill(-1)
    row_to_off.fill(-1)
    return row_to_part, row_to_off


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

        in_dim, label_shape = LazyTensor.write_memmap_streaming_two_pass(
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
        and not LazyTensor.is_feature_label_batch_mapping(d)
    ):
        keys_t, _get_batch, _get_by_indices = LazyTensor.key_index_mapping_getters(d)
        count = len(keys_t)
        if count <= 0:
            raise ValueError("Empty dataset provided to train().")

        in_dim, label_shape = LazyTensor.write_memmap_streaming_two_pass(
            ds=ds,
            out_dir=out_dir,
            count=count,
            get_batch=_get_batch,
            get_by_indices=_get_by_indices,
            val_frac=float(val_frac),
            seed_value=seed_value,
            underflow_action=underflow_action,
            shuffle=bool(shuffle),
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
            cpu_state = _torch_load_cpu(fallback, weights_only=True)
            cpu_state = _preload_state(cpu_state)
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


def predict(
    model: Root,
    data: Dict[Tuple, torch.Tensor],
    *args: Any,
    seed: int = 7,
    mode: OpsMode = "predict",
    max_nodes: Optional[int] = None,
    rdzv_backend: Optional[str] = None,
    output: str = "tensor",
    lazy: bool = False,
    **kwargs: Any,
) -> Any:
    _reset_process_group()
    initialize_python_path()
    set_multiprocessing_env()

    tmp_dir = new_dir("infer")
    dcp_dir: Optional[str] = None
    memmap_dir = os.path.join(tmp_dir, "memmap")
    ckpt_dir = os.path.join(tmp_dir, "pred_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    mp.allow_connection_pickling()

    save_dcp = env_bool("STNET_SAVE_DCP", True)
    save_pt = env_bool("STNET_SAVE_MODEL_PT", True)
    if not (save_dcp or save_pt):
        save_pt = True

    try:
        if save_dcp or save_pt:
            dcp_dir = os.path.join(tmp_dir, "dcp")
            _maybe_save_model_checkpoint(model, dcp_dir, save_dcp=save_dcp, save_pt=save_pt, overwrite=True)

        cfg_obj = getattr(model, "_Root__config", None)
        if isinstance(cfg_obj, (ModelConfig, dict)):
            cfg_model = coerce_model_config(cfg_obj)
        else:
            cfg_model = ModelConfig()
        cfg_dict = model_config_to_dict(cfg_model)

        seed_value = _ensure_seed(seed)
        _seed_everything(seed_value)

        underflow_action = normalize_underflow_action(
            kwargs.pop("underflow_action", None),
            default=default_underflow_action(),
        )

        ds = Dataset.for_device("cpu", feature_dtype=torch.float64, label_float_dtype=torch.float64)
        ds.underflow_action = underflow_action

        output_mode = str(kwargs.pop("output", output) or "tensor").strip().lower()
        if output_mode not in {"tensor", "file"}:
            raise ValueError(f"predict: output must be 'tensor' or 'file', got: {output_mode!r}")

        lazy_flag = bool(kwargs.pop("lazy", lazy)) if output_mode == "tensor" else False
        shuffle = bool(kwargs.pop("shuffle", True))

        from collections.abc import Mapping as _Mapping

        default_out_shape = tuple(getattr(model, "out_shape", ()))
        if not default_out_shape:
            default_out_shape = (1,)

        if isinstance(data, TensorDictBase):
            td = data
            if td.batch_size is None or len(td.batch_size) == 0:
                raise ValueError("TensorDict input to predict() must have a batch dimension.")

            count = int(td.batch_size[0])
            if count <= 0:
                return {}

            in_dim, label_shape = LazyTensor.write_memmap_streaming_two_pass(
                ds=ds,
                out_dir=memmap_dir,
                count=count,
                get_batch=lambda s, e: td[s:e],
                get_by_indices=lambda idx: td[idx],
                val_frac=0.0,
                seed_value=seed_value,
                underflow_action=underflow_action,
                shuffle=bool(shuffle),
                default_label_shape=default_out_shape,
                allow_missing_labels=True,
                features_only=True,
                chunk_size=0,
            )
            keys = range(count)

        elif (
            isinstance(data, _Mapping)
            and data
            and all(not isinstance(v, _Mapping) for v in data.values())
            and not LazyTensor.is_feature_label_batch_mapping(data)
        ):
            keys_t, _get_batch, _get_by_indices = LazyTensor.key_index_mapping_getters(data)
            keys = list(keys_t)
            count = len(keys)
            if count <= 0:
                return {}

            in_dim, label_shape = LazyTensor.write_memmap_streaming_two_pass(
                ds=ds,
                out_dir=memmap_dir,
                count=count,
                get_batch=_get_batch,
                get_by_indices=_get_by_indices,
                val_frac=0.0,
                seed_value=seed_value,
                underflow_action=underflow_action,
                shuffle=bool(shuffle),
                default_label_shape=default_out_shape,
                allow_missing_labels=True,
                features_only=True,
                chunk_size=0,
            )

        else:
            feats, labels, keys, label_shape = ds.preprocess(data)
            if not feats.is_contiguous():
                feats = feats.contiguous()

            count = int(feats.shape[0])
            if count <= 0:
                return {}

            in_dim = int(feats.reshape(count, -1).shape[1])

            preload_memmap(
                {"features": feats, "labels": labels},
                memmap_dir=memmap_dir,
                train_frac=1.0,
                val_frac=0.0,
                shuffle=bool(shuffle),
                seed=seed_value,
                underflow_action=underflow_action,
                allow_missing_labels=True,
                default_label_shape=default_out_shape,
                features_only=True,
            )
            if keys is None:
                keys = range(count)
            if not label_shape:
                label_shape = tuple(default_out_shape)

        base = dict(
            sources={"kind": "memmap", "path": memmap_dir},
            in_dim=int(in_dim),
            out_shape=tuple(label_shape),
            cfg_dict=cfg_dict,
            ckpt_dir=ckpt_dir,
        )
        if dcp_dir is not None:
            base["model_ckpt_dir"] = dcp_dir

        mode = mode if mode in ("predict", "infer") else "predict"

        default_kwargs = {"seed": seed}
        positional_names = RuntimeConfig.PRED_POS_ORDER[: len(args)]
        for key in list(default_kwargs):
            if key in positional_names or key in kwargs:
                default_kwargs.pop(key, None)

        ops = runtime_config(mode, base, *args, **default_kwargs, **kwargs)

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
                log = __import__("logging").getLogger(__name__)

                nkeys = 0
                with contextlib.suppress(Exception):
                    nkeys = int(len(keys))

                keys_kind = "range" if isinstance(keys, range) else "list"
                keys_meta_path = os.path.join(chunks_dir, "keys.meta.json")
                try:
                    meta = {"N": int(nkeys), "kind": keys_kind}
                    if isinstance(keys, range):
                        meta.update({"start": int(keys.start), "stop": int(keys.stop), "step": int(keys.step)})
                    with open(keys_meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f)
                except Exception as e:
                    log.warning("predict: failed to write keys.meta.json (ignored): %r", e, exc_info=True)

                if not isinstance(keys, range):
                    try:
                        # keys is list-like python object => weights_only must be False on modern torch
                        torch.save(keys, os.path.join(chunks_dir, "keys.pt"))
                    except Exception as e:
                        log.warning("predict: failed to write keys.pt (ignored): %r", e, exc_info=True)

                final_dir = new_dir("predictions")
                moved_dir = shutil.move(chunks_dir, final_dir)
                chunk_root = moved_dir if os.path.isdir(moved_dir) else os.path.join(final_dir, os.path.basename(chunks_dir))
                return get_prediction(chunk_root, output=output_mode, lazy=lazy_flag)

            return {}
        finally:
            with contextlib.suppress(Exception):
                shutil.rmtree(tmp_dir, ignore_errors=True)
    finally:
        # If we early-returned with moved predictions, tmp_dir may already be gone.
        with contextlib.suppress(Exception):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _load_legacy_flat(
    chunk_root: str,
    *,
    num_chunks: int,
    out_shape: Tuple[int, ...],
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for idx in range(int(num_chunks)):
        base_mmt = os.path.join(chunk_root, f"chunk_{idx:06d}.mmt")
        tensor = None
        if os.path.exists(base_mmt):
            try:
                # NOTE: legacy path has no stored dtype/shape metadata; MemoryMappedTensor encodes it in the file name.
                tensor = MemoryMappedTensor.from_filename(base_mmt)
            except Exception:
                with contextlib.suppress(Exception):
                    tensor = _torch_load_cpu(base_mmt, weights_only=True)
        else:
            alt_pt = os.path.join(chunk_root, f"chunk_{idx:06d}.pt")
            if os.path.exists(alt_pt):
                tensor = _torch_load_cpu(alt_pt, weights_only=True)

        if tensor is not None:
            chunks.append(tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor))

    if chunks:
        return torch.cat(chunks, dim=0)

    tail = tuple(out_shape[1:]) if len(out_shape) > 1 else ()
    return torch.empty((0, *tail), dtype=torch.float64)


def get_prediction(
    pred_or_dir: Any,
    *,
    output: str = "tensor",
    lazy: bool = False,
) -> Any:
    from collections.abc import Mapping as _Mapping

    if isinstance(pred_or_dir, _Mapping) and "chunks_dir" in pred_or_dir:
        chunk_root = str(pred_or_dir.get("chunks_dir") or "")
    else:
        chunk_root = str(pred_or_dir or "")
    if not chunk_root:
        raise ValueError("get_prediction: chunks_dir is empty")

    out_mode = str(output or "tensor").strip().lower()
    if out_mode not in {"tensor", "file"}:
        raise ValueError(f"get_prediction: output must be 'tensor' or 'file', got: {out_mode!r}")
    lazy = bool(lazy) if out_mode == "tensor" else False

    manifest_path = os.path.join(chunk_root, "manifest.json")
    if not os.path.isfile(manifest_path):
        if out_mode == "file":
            return {"chunks_dir": chunk_root, "format": "stnet.pred"}
        raise FileNotFoundError(f"get_prediction: missing manifest.json: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    out_shape = tuple(manifest.get("out_shape") or ())
    variable_shape = bool(manifest.get("variable_shape"))
    file_result: Dict[str, Any] = {
        "chunks_dir": chunk_root,
        "out_shape": out_shape,
        "format": manifest.get("format") or "stnet.pred",
    }
    if variable_shape:
        file_result["variable_shape"] = True

    if out_mode == "file":
        return file_result

    keys_meta_path = os.path.join(chunk_root, "keys.meta.json")
    if not os.path.isfile(keys_meta_path):
        raise FileNotFoundError(f"get_prediction: missing keys.meta.json: {keys_meta_path}")
    with open(keys_meta_path, "r", encoding="utf-8") as f:
        kmeta = json.load(f) if f is not None else {}
    kind = str((kmeta or {}).get("kind") or "list").strip().lower()
    nkeys = int((kmeta or {}).get("N") or 0)

    keys: Any
    if kind == "range":
        start = int((kmeta or {}).get("start") or 0)
        stop = int((kmeta or {}).get("stop") or nkeys)
        step = int((kmeta or {}).get("step") or 1)
        keys = range(start, stop, step)
        nkeys = int(len(keys))
    else:
        keys_pt = os.path.join(chunk_root, "keys.pt")
        if not os.path.isfile(keys_pt):
            raise FileNotFoundError(f"get_prediction: missing keys.pt for kind=list: {keys_pt}")
        # keys.pt is a python object => weights_only must be False.
        keys = _torch_load_cpu(keys_pt, weights_only=False)
        if not isinstance(keys, list):
            try:
                keys = list(keys)
            except Exception:
                raise TypeError(f"get_prediction: keys.pt must be list-like, got {type(keys)!r}")
        nkeys = int(len(keys))

    parts = manifest.get("parts")
    has_parts = isinstance(parts, list) and bool(parts)

    def _as_tuple_key(k: Any) -> Tuple[Any, ...]:
        if isinstance(k, tuple):
            return k
        try:
            return tuple(k)
        except TypeError:
            return (k,)

    fixed_keys: Optional[list[Tuple[Any, ...]]] = None
    key_to_row: Optional[dict[Tuple[Any, ...], int]] = None

    if isinstance(keys, range):

        class _TupleKeyRange:
            __slots__ = ("_r",)

            def __init__(self, r: range) -> None:
                self._r = r

            def __len__(self) -> int:
                return int(len(self._r))

            def __iter__(self):
                for i in self._r:
                    yield (int(i),)

            def __getitem__(self, idx: int):
                return (int(self._r[idx]),)

        key_seq: Any = _TupleKeyRange(keys)

        def _row_for_key(k: Any) -> int:
            if isinstance(k, tuple) and len(k) == 1:
                k = k[0]
            rid = int(k)
            if rid < 0 or rid >= nkeys:
                raise KeyError(k)
            return rid

    else:
        fixed_keys = []
        key_to_row = {}
        seen: set[Tuple[Any, ...]] = set()
        for rid, k in enumerate(keys):
            kt = _as_tuple_key(k)
            kout = kt if kt not in seen else (kt + (rid,))
            if kout in seen:
                kout = kt + (rid, len(seen))
            seen.add(kout)
            fixed_keys.append(kout)
            key_to_row[kout] = int(rid)
        key_seq = fixed_keys

        def _row_for_key(k: Any) -> int:
            kt = _as_tuple_key(k)
            rid = key_to_row.get(kt) if key_to_row is not None else None
            if rid is None:
                raise KeyError(k)
            return int(rid)

    if has_parts:
        parts_list = parts

        # Lazy path: build a fast row->(part,offset) map and use memmap-backed part access.
        if lazy:
            from ..data.collections import LazyDict

            row_to_part, row_to_off = _alloc_row_map_arrays(nkeys, chunk_root=chunk_root)

            # Store per-part pred file paths.
            pred_pt_paths: list[Optional[str]] = [None] * len(parts_list)
            pred_mmt_paths: list[Optional[str]] = [None] * len(parts_list)

            _dup_env = str(os.environ.get("STNET_PRED_CHECK_ROWIDS_DUP", "")).strip().lower()
            check_dups = _dup_env in {"1", "true", "yes", "y", "on"}

            for p_idx, part in enumerate(parts_list):
                rows_name = (part or {}).get("rows")
                pred_name = (part or {}).get("pred")
                if not rows_name or not pred_name:
                    continue

                rows_path = os.path.join(chunk_root, rows_name)
                pred_path = os.path.join(chunk_root, pred_name)

                pred_pt_paths[p_idx] = pred_path
                pred_mmt_paths[p_idx] = _derive_mmt_path(pred_path)

                rows = _torch_load_cpu(rows_path, weights_only=True)
                if not isinstance(rows, torch.Tensor):
                    rows = torch.as_tensor(rows)
                rows = rows.to(dtype=torch.int64).reshape(-1).contiguous()
                if rows.numel() == 0:
                    continue
                rows_np = rows.numpy()

                if check_dups:
                    u = np.unique(rows_np)
                    if u.size != rows_np.size:
                        raise RuntimeError(f"get_prediction: duplicate row_ids within part {rows_name} (idx={p_idx})")
                    if np.any(row_to_part[u] != -1):
                        raise RuntimeError(f"get_prediction: duplicate row_ids across parts (current={rows_name}, idx={p_idx})")
                    del u
                else:
                    if np.any(row_to_part[rows_np] != -1):
                        raise RuntimeError(f"get_prediction: duplicate row_ids across parts (current={rows_name}, idx={p_idx})")

                row_to_part[rows_np] = int(p_idx)
                row_to_off[rows_np] = np.arange(rows_np.size, dtype=np.int32)

            if np.any(row_to_part < 0):
                missing = int(np.sum(row_to_part < 0))
                raise RuntimeError(f"get_prediction: missing predictions for {missing}/{nkeys} rows")

            _tls = threading.local()

            def _open_part_pred(p: int):
                # Thread-local last-part cache to avoid locks on no-GIL builds.
                cache = getattr(_tls, "pred_cache", None)
                if cache is not None and cache[0] == p and cache[1] is not None:
                    return cache[1]

                pt_path = pred_pt_paths[p]
                mmt_path = pred_mmt_paths[p]

                pred_obj: Any = None
                # Prefer memmap if available, or convert .pt -> .mmt once (B approach).
                if mmt_path is not None:
                    pred_obj = _open_pred_memmap(mmt_path)
                    if pred_obj is None and pt_path is not None and os.path.isfile(pt_path):
                        try:
                            _ensure_pred_memmap_from_pt(pt_path, mmt_path)
                            pred_obj = _open_pred_memmap(mmt_path)
                        except Exception:
                            pred_obj = None

                if pred_obj is None:
                    if pt_path is None:
                        raise FileNotFoundError(f"get_prediction: missing pred file for part idx={p}")
                    pred_obj = _torch_load_cpu(pt_path, weights_only=True)
                    if not isinstance(pred_obj, torch.Tensor):
                        pred_obj = torch.as_tensor(pred_obj)

                _tls.pred_cache = (p, pred_obj)
                return pred_obj

            def _pred_for_row(row_id: int) -> torch.Tensor:
                p = int(row_to_part[int(row_id)])
                if p < 0:
                    raise KeyError(row_id)
                off = int(row_to_off[int(row_id)])
                pred = _open_part_pred(p)
                try:
                    return pred[off].detach()
                except Exception:
                    # Safety: if pred is not indexable, coerce.
                    t = pred if isinstance(pred, torch.Tensor) else torch.as_tensor(pred)
                    return t[off].detach()

            def _getter(key: Any) -> torch.Tensor:
                rid = _row_for_key(key)
                return _pred_for_row(rid)

            return LazyDict(key_seq, _getter, name="predictions", cache=False)

        # Non-lazy path: materialize into a dict (can be large).
        out: Dict[Tuple[Any, ...], torch.Tensor] = {}
        out_set = out.__setitem__
        keys_is_range = isinstance(keys, range)
        fk = fixed_keys
        int_ = int
        for part in parts_list:
            rows_name = (part or {}).get("rows")
            pred_name = (part or {}).get("pred")
            if not rows_name or not pred_name:
                continue
            rows_path = os.path.join(chunk_root, rows_name)
            pred_path = os.path.join(chunk_root, pred_name)

            rows = _torch_load_cpu(rows_path, weights_only=True)
            if not isinstance(rows, torch.Tensor):
                rows = torch.as_tensor(rows)
            rows = rows.to(dtype=torch.int64).reshape(-1).contiguous()

            preds: Any
            if pred_path.endswith(".mmt"):
                mm = _open_pred_memmap(pred_path)
                if mm is None:
                    raise FileNotFoundError(f"get_prediction: missing/invalid memmap meta for {pred_path}")
                preds = mm
            else:
                preds = _torch_load_cpu(pred_path, weights_only=True)

            if not isinstance(preds, torch.Tensor):
                preds = torch.as_tensor(preds)
            preds = preds.detach()

            if int(preds.shape[0]) != int(rows.shape[0]):
                raise RuntimeError(
                    f"get_prediction: part size mismatch rows={int(rows.shape[0])} preds={int(preds.shape[0])} ({rows_name},{pred_name})"
                )

            rows_np = rows.numpy()
            preds_get = preds.__getitem__

            if keys_is_range:
                for j, rid in enumerate(rows_np):
                    rid_i = int_(rid)
                    out_set((rid_i,), preds_get(j))
            elif fk is not None:
                for j, rid in enumerate(rows_np):
                    out_set(fk[int_(rid)], preds_get(j))
            else:
                for j, rid in enumerate(rows_np):
                    rid_i = int_(rid)
                    out_set((rid_i,), preds_get(j))
        return out

    # Legacy flat chunks path.
    num_chunks = int(manifest.get("num_chunks", 0) or 0)
    flat = _load_legacy_flat(chunk_root, num_chunks=num_chunks, out_shape=out_shape)

    pred_tensor = Root.unflatten_y(flat, out_shape)
    out_legacy: Dict[Tuple[Any, ...], torch.Tensor] = {}
    out_set = out_legacy.__setitem__
    n_out = int(pred_tensor.shape[0])
    for i, k in enumerate(key_seq):
        if i >= n_out:
            break
        out_set(k, pred_tensor[i].detach().cpu().to(dtype=torch.float64))
    return out_legacy
