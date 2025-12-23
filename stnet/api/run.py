# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import json
import os
import random
import shutil
import tempfile
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

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
    from torch.distributed.run import LaunchConfig, elastic_launch
except ImportError:  # pragma: no cover
    from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from ..backend.distributed import get_available_host, get_preferred_ip, initialize_master_addr
from ..backend.runtime import _torch_load_cpu, _trim_dcp_keys, main
from ..backend.system import (
    WorkerPolicy,
    get_master_port,
    initialize_python_path,
    new_dir,
    optimal_start_method,
    remove_dir,
    set_multiprocessing_env,
)
from ..data.collections import LazyTensor
from ..data.datatype import env_bool, env_int
from ..data.nodes import preload_memmap
from ..data.pipeline import (
    Dataset,
    default_underflow_action,
    extract_xy,
    normalize_underflow_action,
    resolve_feature_key,
    resolve_label_key,
)
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
            if hasattr(t, "is_contiguous") and not bool(t.is_contiguous()):
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


def read_json(path: str) -> Any:
    """Read and parse JSON payload from ``path``.

    This thin wrapper exists for symmetry with ``write_json_atomic`` and to avoid
    sprinkling ``json.load`` calls throughout prediction assembly helpers.
    """

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: str, payload: Any) -> None:
    """Atomically write JSON using :func:`_atomic_write_json`."""

    _atomic_write_json(path, payload)


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

    mapping = {
        torch.float16: _np.float16,
        torch.float32: _np.float32,
        torch.float64: _np.float64,
        torch.int8: _np.int8,
        torch.uint8: _np.uint8,
        torch.int16: _np.int16,
        torch.int32: _np.int32,
        torch.int64: _np.int64,
        torch.bool: _np.bool_,
    }
    return mapping.get(dtype, _np.float64)


def _dtype_from_name(name: str, default: torch.dtype) -> torch.dtype:
    n = str(name).strip().lower().replace("torch.", "")
    if n in ("float", "float32", "fp32"):
        return torch.float32
    if n in ("float64", "double", "fp64"):
        return torch.float64
    if n in ("float16", "half", "fp16"):
        return torch.float16
    if n in ("bfloat16", "bf16"):
        return torch.bfloat16
    if n in ("int64", "long"):
        return torch.int64
    if n in ("int32", "int"):
        return torch.int32
    if n in ("int16", "short"):
        return torch.int16
    if n in ("int8", "char"):
        return torch.int8
    if n in ("uint8", "byte"):
        return torch.uint8
    if n in ("bool", "boolean"):
        return torch.bool
    return default


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

    f_dtype = _dtype_from_name(meta.get("features_dtype", "float64"), torch.float64)

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

        rows_t = _torch_load_cpu(rows_file, weights_only=True)
        if not isinstance(rows_t, torch.Tensor):
            rows_t = torch.as_tensor(rows_t, device="cpu")
        rows_t = rows_t.to(dtype=torch.int64, device="cpu")

        if pred_file.endswith(".mmt"):
            preds_t = _open_pred_memmap(pred_file)
        else:
            preds_t = _torch_load_cpu(pred_file, weights_only=True)
            if not isinstance(preds_t, torch.Tensor):
                preds_t = torch.as_tensor(preds_t, device="cpu")

        preds_t = preds_t.to(dtype=store_float, device="cpu")

        if preds_t.shape[0] != rows_t.shape[0]:
            raise ValueError(
                f"Pred/rows mismatch in {pred_file}: preds[0]={preds_t.shape[0]} vs rows={rows_t.shape[0]}"
            )

        Y_out.index_copy_(0, rows_t, preds_t)

    pred_meta_path = out_path + ".meta.json"
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

        rows_t = _torch_load_cpu(rows_file, weights_only=True)
        if not isinstance(rows_t, torch.Tensor):
            rows_t = torch.as_tensor(rows_t, device="cpu")
        rows_t = rows_t.to(dtype=torch.int64, device="cpu")

        if pred_file.endswith(".mmt"):
            preds_t = _open_pred_memmap(pred_file)
        else:
            preds_t = _torch_load_cpu(pred_file, weights_only=True)
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

    with h5py.File(out_path, "w") as f:
        dset_X = f.create_dataset("X", shape=tuple(X_mmt.shape), dtype=_torch_dtype_to_numpy(X_mmt.dtype))
        dset_Y = f.create_dataset("Y", shape=(int(count), *out_shape_t), dtype=np_float)

        chunk = 8192
        for s in range(0, int(count), chunk):
            e = min(int(count), s + chunk)
            x_slice = X_mmt[s:e]
            dset_X[s:e] = x_slice.detach().cpu().numpy()

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

            rows_t = _torch_load_cpu(rows_file, weights_only=True)
            if not isinstance(rows_t, torch.Tensor):
                rows_t = torch.as_tensor(rows_t, device="cpu")
            rows_np = rows_t.to(dtype=torch.int64, device="cpu").numpy()

            if pred_file.endswith(".mmt"):
                preds_t = _open_pred_memmap(pred_file)
            else:
                preds_t = _torch_load_cpu(pred_file, weights_only=True)
                if not isinstance(preds_t, torch.Tensor):
                    preds_t = torch.as_tensor(preds_t, device="cpu")

            preds_np = preds_t.to(dtype=store_float, device="cpu").detach().cpu().numpy()

            if preds_np.shape[0] != rows_np.shape[0]:
                raise ValueError(
                    f"Pred/rows mismatch in {pred_file}: preds[0]={preds_np.shape[0]} vs rows={rows_np.shape[0]}"
                )

            dset_Y[rows_np] = preds_np

    return PersistentTensorDict(filename=out_path, batch_size=[int(count)], mode="r")


def _write_predictions_h5(
    filename: str,
    X: torch.Tensor,
    parts: list[tuple[int, int, str]],
    *,
    y_tail_shape: tuple[int, ...],
    y_dtype: torch.dtype = torch.float64,
    chunk_rows: int = 16384,
) -> PersistentTensorDict:
    """Write X/Y prediction results to an HDF5-backed PersistentTensorDict."""
    import h5py

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    count = int(X.shape[0])
    x_shape = tuple(int(x) for x in X.shape)
    y_shape = (count, *tuple(int(x) for x in y_tail_shape))

    with h5py.File(filename, "w") as f:
        dset_x = f.create_dataset("X", shape=x_shape, dtype=_torch_dtype_to_numpy(X.dtype), chunks=True)
        dset_y = f.create_dataset("Y", shape=y_shape, dtype=_torch_dtype_to_numpy(y_dtype), chunks=True)

        if count:
            step = max(1, min(chunk_rows, count))
            for s in range(0, count, step):
                e = min(count, s + step)
                xb = X[s:e]
                if xb.dtype is torch.bfloat16:
                    xb = xb.to(torch.float32)
                dset_x[s:e] = xb.detach().cpu().numpy()

        for part_start, part_end, part_path in parts:
            part_start = int(part_start)
            part_end = int(part_end)
            part_tensor = None
            if part_path.endswith(".mmt") and os.path.isfile(part_path):
                part_tensor = _open_pred_memmap(part_path)
            elif part_path.endswith(".pt") and os.path.isfile(part_path):
                mmt_path = os.path.splitext(part_path)[0] + ".mmt"
                _ensure_pred_memmap_from_pt(part_path, mmt_path, existsok=True)
                part_tensor = _open_pred_memmap(mmt_path)
            else:
                raise FileNotFoundError(f"Missing prediction part: {part_path}")

            preds = part_tensor.detach().cpu()
            if preds.ndim == 1:
                preds = preds.unsqueeze(-1)

            expected = (part_end - part_start, *y_tail_shape)
            if tuple(preds.shape) != expected:
                try:
                    preds_unflat = Root.unflatten_y(preds, y_tail_shape)
                except Exception:
                    preds_unflat = None
                if preds_unflat is not None and tuple(preds_unflat.shape) == expected:
                    preds = preds_unflat
                else:
                    raise ValueError(
                        "Prediction part shape mismatch: "
                        f"expected {expected} but got {tuple(preds.shape)} for {part_path}"
                    )

            if preds.dtype != y_dtype:
                preds = preds.to(y_dtype)

            dset_y[part_start:part_end] = preds.numpy()

    return PersistentTensorDict(filename=filename, batch_size=[count], mode="r")


@torch.inference_mode()
@catchtime(logger, fn_name="predict")
def predict(
    model: Root,
    data: Any,
    *args,
    seed: int = 7,
    mode: OpsMode = "predict",
    max_nodes: Optional[int] = None,
    rdzv_backend: Optional[str] = None,
    lazy: bool = True,
    path: Optional[str] = None,
    shuffle: bool = False,
    **kwargs,
) -> "TensorDictBase":
    """Distributed prediction with a single internal streaming path.

    Internal path (always):
      1) Write input features to a memmap dataset (MemoryMappedTensor on disk).
      2) elastic_launch() workers consume memmaps and write sharded prediction parts.
      3) Driver assembles a single pred.mmt (MemoryMappedTensor on disk).

    Return types (only these three):
      * lazy=False -> TensorDict with in-memory torch.Tensor leaves
      * lazy=True and valid `path` -> PersistentTensorDict (HDF5) written to `path`
      * lazy=True and no/invalid `path` -> TensorDict with MemoryMappedTensor leaves

    Notes:
      - We intentionally do NOT wrap the output in a LazyStackedTensorDict.
        When X/Y are already memmap tensors, TensorDict indexing/slicing is already
        lazy without creating N Python objects (better for GIL/no-GIL).
    """

    _reset_process_group()
    initialize_python_path()
    set_multiprocessing_env()

    persist_path = _normalize_path(path)
    shuffle = bool(kwargs.pop("shuffle", shuffle))

    underflow_action = kwargs.pop("underflow_action", default_underflow_action())
    ds = Dataset.for_device("cpu", feature_dtype=torch.float64, label_float_dtype=torch.float64)
    ds.underflow_action = underflow_action

    out_shape = kwargs.pop("out_shape", getattr(model, "out_shape", ()))
    out_shape = tuple(int(x) for x in (out_shape or ()))

    tmp_dir = new_dir("infer")
    ckpt_dir = os.path.join(tmp_dir, "ckpt")
    memmap_dir = os.path.join(tmp_dir, "memmap")
    os.makedirs(ckpt_dir, exist_ok=True)

    dcp_dir = os.path.join(ckpt_dir, "model")
    _maybe_save_model_checkpoint(model, dcp_dir)

    key0: Any = 0
    keys: Any = range(0)
    count: int = 0
    feature_dim: int = 0

    if TensorDictBase is not None and isinstance(data, TensorDictBase):
        try:
            X_raw, _Y_raw = extract_xy(data, labels_required=False)
            if X_raw is None or not hasattr(X_raw, "shape"):
                raise ValueError("TensorDict input must contain a feature column ('X'/'x' etc).")
            count = int(X_raw.shape[0])
        except Exception:
            count = int(getattr(data, "batch_size", [0])[0] or 0)

        key0 = 0
        keys = range(count)

        def _get_batch(s: int, e: int):
            return data[s:e]

        def _get_by_indices(idxs: Sequence[int]):
            return data[torch.as_tensor(idxs, device="cpu")]

        count, feature_dim, _label_shape, write_labels = LazyTensor.write_memmap_streaming_two_pass(
            ds,
            count,
            memmap_dir,
            get_batch=_get_batch,
            get_by_indices=_get_by_indices,
            key0=key0,
            keys=keys,
            seed_value=seed,
            shuffle=shuffle,
        )

    elif isinstance(data, Mapping) and data:
        if LazyTensor.is_feature_label_batch_mapping(data):
            fkey = resolve_feature_key(data) or "X"
            lkey = resolve_label_key(data)
            X_all = data.get(fkey)
            if X_all is None or not hasattr(X_all, "shape"):
                raise ValueError("Feature/label mapping must include a feature column ('X'/'x' etc).")
            Y_all = data.get(lkey) if lkey is not None else None

            count = int(X_all.shape[0])
            key0 = 0
            keys = range(count)

            def _get_batch(s: int, e: int):
                out = {fkey: X_all[s:e]}
                if Y_all is not None:
                    out[lkey] = Y_all[s:e]
                return out

            def _get_by_indices(idxs):
                idx_t = torch.as_tensor(list(idxs), device="cpu")
                out = {fkey: X_all.index_select(0, idx_t)}
                if Y_all is not None:
                    out[lkey] = Y_all.index_select(0, idx_t)
                return out

            count, feature_dim, _label_shape, write_labels = LazyTensor.write_memmap_streaming_two_pass(
                ds,
                count,
                memmap_dir,
                get_batch=_get_batch,
                get_by_indices=_get_by_indices,
                key0=key0,
                keys=keys,
                seed_value=seed,
                shuffle=shuffle,
            )

        else:
            keys = list(data.keys())
            count = len(keys)
            key0 = keys[0] if keys else 0

            def _get_batch(s: int, e: int):
                return {keys[i]: data[keys[i]] for i in range(s, e)}

            def _get_by_indices(idxs):
                return {keys[i]: data[keys[i]] for i in idxs}

            count, feature_dim, _label_shape, write_labels = LazyTensor.write_memmap_streaming_two_pass(
                ds,
                count,
                memmap_dir,
                get_batch=_get_batch,
                get_by_indices=_get_by_indices,
                key0=key0,
                keys=keys,
                seed_value=seed,
                shuffle=shuffle,
            )

    else:
        X, Y, count, key0, keys, _label_shape = ds.preprocess(data, device="cpu", split_frac=0.0)
        if X is None:
            raise ValueError("Could not infer features from input data.")
        feature_dim = int(X.shape[-1])
        preload_memmap(
            {"X": X, "Y": Y},
            memmap_dir,
            device="cpu",
            feature_dtype=torch.float64,
            label_float_dtype=torch.float64,
            allow_missing_labels=True,
        )

    if count <= 0 or feature_dim <= 0:
        raise ValueError(f"Invalid inference dataset: count={count}, feature_dim={feature_dim}")

    cfg_dict = getattr(model, "__config", None)
    if cfg_dict is None:
        cfg_dict = getattr(model, "config", None)
    cfg_dict = cfg_dict or {}

    base = {
        "sources": [memmap_dir],
        "feature_dim": int(feature_dim),
        "out_shape": out_shape,
        "cfg_dict": cfg_dict,
        "ckpt_dir": ckpt_dir,
        "model_ckpt_dir": dcp_dir,
    }

    if mode not in ("predict", "infer"):
        raise ValueError(f"predict only supports mode='predict'/'infer' (got: {mode})")

    default_kwargs = {
        "log_interval": 0,
        "progress": False,
        "seed": seed,
    }

    ops = runtime_config(mode, base, *args, **default_kwargs, **kwargs)

    model.to(device="cpu")
    torch.cuda.empty_cache()

    master_addr = initialize_master_addr(seed)
    master_port = get_master_port(seed)

    wp = WorkerPolicy.autotune(
        max_nodes=max_nodes,
        nproc_per_node=env_int("LOCAL_WORLD_SIZE", 0),
        omp_num_threads=env_int("OMP_NUM_THREADS", 0),
    )

    lc = LaunchConfig(
        min_nodes=wp.nnodes,
        max_nodes=wp.nnodes,
        nproc_per_node=wp.nproc,
        run_id=ops.run_id,
        rdzv_backend=rdzv_backend or "c10d",
        rdzv_endpoint=f"{master_addr}:{master_port}",
        max_restarts=0,
        monitor_interval=0,
        start_method=wp.start_method,
    )

    elastic_launch(lc, main)(ops)

    chunks_dir = os.path.join(ckpt_dir, "pred_chunks")
    if not os.path.isdir(chunks_dir):
        remove_dir(tmp_dir)
        return TensorDict({"X": torch.empty((0, feature_dim)), "Y": torch.empty((0, *out_shape))}, batch_size=[0])

    if lazy and persist_path and _is_writable_file_path(persist_path):
        out_td = _write_predictions_h5_from_chunks(
            persist_path,
            memmap_dir=memmap_dir,
            chunks_dir=chunks_dir,
            count=int(count),
            out_shape=out_shape,
            store_float=torch.float64,
        )
        remove_dir(tmp_dir)
        return out_td

    if lazy:
        final_dir = new_dir("predictions")
        os.makedirs(final_dir, exist_ok=True)

        moved_memmap_dir = os.path.join(final_dir, "memmap")
        os.makedirs(os.path.dirname(moved_memmap_dir), exist_ok=True)
        shutil.move(memmap_dir, moved_memmap_dir)

        X_mmt = _open_features_mmt(moved_memmap_dir)

        pred_mmt_path = os.path.join(final_dir, "pred.mmt")
        Y_mmt = _assemble_predictions_to_memmap(
            chunks_dir,
            pred_mmt_path,
            count=int(count),
            out_shape=out_shape,
            store_float=torch.float64,
        )

        remove_dir(tmp_dir)

        td_mm = TensorDict({"X": X_mmt[: int(count)], "Y": Y_mmt[: int(count)]}, batch_size=[int(count)])
        return td_mm

    X_mmt = _open_features_mmt(memmap_dir)
    Y_t = _assemble_predictions_to_tensor(
        chunks_dir,
        count=int(count),
        out_shape=out_shape,
        dtype=torch.float64,
    )

    X_t = X_mmt[: int(count)].detach().cpu().clone()

    remove_dir(tmp_dir)

    return TensorDict({"X": X_t, "Y": Y_t}, batch_size=[int(count)])


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



@torch.inference_mode()
@catchtime(logger, fn_name="get_prediction")
def get_prediction(
    source: str,
    mode: str = "predict",
    *,
    lazy: bool = True,
    path: Optional[str] = None,
) -> "TensorDictBase":
    """Load predictions produced by :func:`predict`.

    Supported sources:
      - A persistent HDF5 file written by predict(lazy=True, path=...)
      - A directory produced by predict(lazy=True, path=None) containing:
            <dir>/memmap/meta.json
            <dir>/memmap/features.mmt
            <dir>/pred.mmt (+ pred.mmt.meta.json)

    Returns:
      * lazy=False: a TensorDict with in-memory torch.Tensor leaves
      * lazy=True & path is valid: a PersistentTensorDict
      * lazy=True & path is None/invalid: a memmap-backed TensorDict (MemoryMappedTensor leaves)
    """
    _ = mode

    source = _normalize_path(source) or source
    persist_path = _normalize_path(path)

    if os.path.isfile(source) and os.path.splitext(source)[1].lower() in (".h5", ".hdf5"):
        if PersistentTensorDict is None:
            raise ImportError("tensordict is required to read PersistentTensorDict outputs.")
        td = PersistentTensorDict(filename=source, batch_size=None, mode="r")
        if lazy:
            return td
        X = torch.as_tensor(td["X"], device="cpu").clone()
        Y = torch.as_tensor(td["Y"], device="cpu").clone()
        return TensorDict({"X": X, "Y": Y}, batch_size=[int(X.shape[0])])

    if not os.path.isdir(source):
        raise FileNotFoundError(f"Prediction source not found: {source}")

    memmap_dir = os.path.join(source, "memmap")
    pred_path = os.path.join(source, "pred.mmt")
    chunks_dir = os.path.join(source, "pred_chunks")

    if lazy and persist_path and _is_writable_file_path(persist_path):
        if os.path.isfile(pred_path):
            if PersistentTensorDict is None:
                raise ImportError("tensordict is required for PersistentTensorDict outputs.")
            X_mmt = _open_features_mmt(memmap_dir)

            meta = read_json(pred_path + ".meta.json")
            shape = tuple(int(x) for x in meta.get("shape", []))
            if not shape:
                raise ValueError(f"Missing pred meta: {pred_path}.meta.json")
            count = int(shape[0])
            out_shape = shape[1:]
            store_float = _dtype_from_name(meta.get("dtype", "float64"), torch.float64)

            out_td = _write_predictions_h5_from_chunks(
                persist_path,
                memmap_dir=memmap_dir,
                chunks_dir=chunks_dir if os.path.isdir(chunks_dir) else os.path.dirname(pred_path),
                count=count,
                out_shape=out_shape,
                store_float=store_float,
            )
            return out_td

        if os.path.isdir(chunks_dir):
            manifest = read_json(os.path.join(chunks_dir, "manifest.json"))
            out_shape = tuple(int(x) for x in manifest.get("out_shape", []) or [])
            X_mmt = _open_features_mmt(memmap_dir)
            count = int(X_mmt.shape[0])
            return _write_predictions_h5_from_chunks(
                persist_path,
                memmap_dir=memmap_dir,
                chunks_dir=chunks_dir,
                count=count,
                out_shape=out_shape,
                store_float=torch.float64,
            )

    if os.path.isfile(pred_path):
        X_mmt = _open_features_mmt(memmap_dir)
        Y_mmt = _open_pred_memmap(pred_path)
        td_mm = TensorDict({"X": X_mmt, "Y": Y_mmt}, batch_size=[int(X_mmt.shape[0])])
        if not lazy:
            return TensorDict({
                "X": X_mmt.detach().cpu().clone(),
                "Y": Y_mmt.detach().cpu().clone(),
            }, batch_size=[int(X_mmt.shape[0])])
        return td_mm

    if os.path.isdir(chunks_dir):
        manifest = read_json(os.path.join(chunks_dir, "manifest.json"))
        out_shape = tuple(int(x) for x in manifest.get("out_shape", []) or [])
        X_mmt = _open_features_mmt(memmap_dir)
        count = int(X_mmt.shape[0])
        Y_mmt = _assemble_predictions_to_memmap(
            chunks_dir,
            pred_path,
            count=count,
            out_shape=out_shape,
            store_float=torch.float64,
        )
        td_mm = TensorDict({"X": X_mmt, "Y": Y_mmt}, batch_size=[count])
        if not lazy:
            return TensorDict({
                "X": X_mmt.detach().cpu().clone(),
                "Y": Y_mmt.detach().cpu().clone(),
            }, batch_size=[count])
        return td_mm

    raise FileNotFoundError(f"No predictions found under: {source}")
