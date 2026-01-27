# -*- coding: utf-8 -*-
from __future__ import annotations

import collections.abc
import contextlib
import logging
import os
import shutil
import tempfile
import time
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Self,
)

import h5py
import numpy
import torch
import torch.utils.data
from tensordict import (
    MemoryMappedTensor,
    PersistentTensorDict,
    TensorDict,
    TensorDictBase,
)

from ..schema import canonicalize_keys_, get_row
from ..core.concurrency import (
    TensorPagePool,
    TensorSpooler,
)
from ..core.datatypes import (
    PathLike,
    dtype_from_name,
    env_first_int,
    env_str,
    get_meta_path,
    parse_torch_dtype,
    read_json,
    save_temp,
    write_json,
)
from ..core.system import (
    Memory,
    is_accelerator_available,
)


_LOGGER = logging.getLogger(__name__)


def _strictest_underflow_action(
    v1: Optional[str], v2: Optional[str]
) -> Optional[str]:
    if v1 is None or v2 is None:
        return v1 or v2
    order = {"allow": 0, "warn": 1, "forbid": 2}
    return v1 if order.get(str(v1), 0) >= order.get(str(v2), 0) else v2


def _meta_has_scale(meta: Any) -> bool:
    if not isinstance(meta, Mapping):
        return False
    keys = (
        "scale_max_abs",
        "scale_min_value",
        "scale_max_value",
        "scale_min_positive",
    )
    if meta.get("has_scale") or any(meta.get(k) is not None for k in keys):
        return True
    return isinstance((ss := meta.get("scale_stats")), Mapping) and (
        ss.get("has_scale") or any(ss.get(k) is not None for k in keys)
    )


def _to_safe_tensor(
    obj: Any, dtype: Optional[torch.dtype] = None
) -> Optional[torch.Tensor]:
    if obj is None:
        return None
    t = obj if isinstance(obj, torch.Tensor) else torch.as_tensor(obj)
    if dtype is None:
        return t
    return (
        t.to(dtype=dtype, copy=False)
        if isinstance(t, torch.Tensor) and t.dtype != dtype
        else t
    )


def _td_batch_size_from_X(x: Any) -> list[int]:
    return (
        [int(x.shape[0])]
        if isinstance(x, torch.Tensor) and x.ndim >= 1
        else []
    )


def _coerce_path(path: PathLike) -> Optional[str]:
    if path is None:
        return None
    p = str(path).replace("\r", "").replace("\n", "").strip()
    if not p or p.lower() in ("none", "null", "nil"):
        return None
    return os.path.abspath(os.path.expanduser(p))


def _coerce_prediction_output(output: object) -> str:
    if isinstance(output, str) and output.strip().lower() in {
        "file",
        "disk",
        "lazy",
        "h5",
        "hdf5",
    }:
        return "file"
    return "memory"


def _coerce_prediction_overwrite(overwrite: object) -> str:
    if isinstance(overwrite, str):
        ow = overwrite.strip().lower()
        if ow == "resume":
            return "resume"
        if ow in {"replace", "overwrite", "force"}:
            return "replace"
        if ow in {"ignore", "skip"}:
            return "ignore"
    return "error"


def _coerce_prediction_path(
    path: PathLike, *args: Any, run_id: str
) -> Optional[str]:
    del args
    p = Path(str(path))
    if p.suffix.lower() in {".h5", ".hdf5"}:
        return os.fspath(p)
    if p.is_dir():
        return os.fspath(p / f"{run_id}.h5")
    return None


def _is_path_writable(path: PathLike) -> bool:
    try:
        p = Path(path)
        if p.is_dir():
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".probe"
            probe.write_text("", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return True
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8"):
            pass
        return True
    except Exception:
        return False


def _get_prediction_dtype(
    chunks_dir: PathLike,
    *args: Any,
    default: torch.dtype = torch.float32,
) -> torch.dtype:
    del args
    manifest_path = os.path.join(os.fspath(chunks_dir), "manifest.json")
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
        pred_path = os.path.join(os.fspath(chunks_dir), str(pred_rel))
        if pred_path.endswith(".mmt"):
            meta_path = get_meta_path(pred_path)
            if os.path.isfile(meta_path):
                try:
                    meta = read_json(meta_path)
                except Exception:
                    meta = None
                if isinstance(meta, dict):
                    dt = parse_torch_dtype(meta.get("dtype"))
                    if isinstance(dt, torch.dtype):
                        return dt
        if os.path.isfile(pred_path):
            try:
                from ..runtime.io import _torch_load_checkpoint

                preds_t = _torch_load_checkpoint(
                    pred_path, map_location="cpu", weights_only=True
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


def _remove_safe(path: str) -> None:
    with suppress(FileNotFoundError):
        os.remove(path)


def _node_state_key(node: Any, attr: str, fallback: str) -> str:
    k = getattr(node, attr, None) or getattr(type(node), attr, None)
    return k if isinstance(k, str) else fallback


def _expand_multinode_sources(spec: Any) -> tuple[Any, bool]:
    if not (isinstance(spec, dict) and "path" in spec and "kind" in spec):
        return spec, False
    root = os.fspath(spec.get("path") or "")
    mn_path = os.path.join(root, "multinode.json")
    if not os.path.isfile(mn_path):
        return spec, False
    payload = read_json(mn_path)
    if isinstance(payload, dict):
        return {
            str(k): {"kind": "memmap", "path": os.path.join(root, str(v))}
            for k, v in payload.items()
        }, True
    if isinstance(payload, list):
        return [
            {"kind": "memmap", "path": os.path.join(root, str(v))}
            for v in payload
        ], True
    return spec, False


def _is_accelerator_available() -> bool:
    return any(is_accelerator_available(a) for a in ("cuda", "xpu", "mps"))


def _device_guard_ok(device: torch.device, guard_bytes: int) -> bool:
    if guard_bytes <= 0:
        return True
    try:
        return (
            free_b := Memory.mem_get_info(device)[0]
        ) is None or free_b >= guard_bytes
    except Exception:
        return True


def _host_guard_ok(guard_bytes: int) -> bool:
    return (
        guard_bytes <= 0
        or getattr(Memory, "available", lambda: guard_bytes)() >= guard_bytes
    )


def _accel_event_poll_params() -> tuple[float, float, float]:
    start_us = int(
        env_first_int(
            (
                "ENN_ACCEL_EVENT_POLL_START_US",
                "ENN_CUDA_EVENT_POLL_START_US",
            ),
            default=500,
        )
        or 500
    )
    max_ms = int(
        env_first_int(
            ("ENN_ACCEL_EVENT_POLL_MAX_MS", "ENN_CUDA_EVENT_POLL_MAX_MS"),
            default=50,
        )
        or 50
    )
    stop_min_ms = int(
        env_first_int(
            (
                "ENN_ACCEL_EVENT_POLL_STOP_MIN_MS",
                "ENN_CUDA_EVENT_POLL_STOP_MIN_MS",
            ),
            default=5,
        )
        or 5
    )
    base_s = max(0.0, float(start_us) / 1_000_000.0)
    max_s = max(base_s, float(max_ms) / 1000.0)
    stop_min_s = max(0.0, float(stop_min_ms) / 1000.0)
    return base_s, max_s, stop_min_s


def _wait_accel_event_done(
    ev: Any,
    *args: Any,
    stopped: Callable[[], bool] | None = None,
    base_sleep_s: float | None = None,
    max_sleep_s: float | None = None,
    stop_min_sleep_s: float | None = None,
) -> None:
    stop_fn = stopped if stopped is not None else (lambda: False)
    d_base, d_max, d_stop_min = _accel_event_poll_params()
    base_sleep_s = base_sleep_s if base_sleep_s is not None else d_base
    max_sleep_s = max_sleep_s if max_sleep_s is not None else d_max
    stop_min_sleep_s = (
        stop_min_sleep_s if stop_min_sleep_s is not None else d_stop_min
    )
    sleep_s = max(0.0, float(base_sleep_s))
    max_s = max(sleep_s, float(max_sleep_s))
    stop_min_s = max(0.0, float(stop_min_sleep_s))
    while True:
        try:
            if ev.query():
                return
        except Exception:
            with suppress(Exception):
                ev.synchronize()
            return
        if stop_fn():
            sleep_s = max(float(sleep_s), stop_min_s)
        time.sleep(sleep_s)
        sleep_s = min(float(sleep_s) * 2.0, max_s)


def _preload_slice_any(
    obj: object | None, s: int, e: int, *args: object, name: str
) -> object | None:
    if obj is None:
        return None
    try:
        return obj[s:e]
    except Exception:
        return [obj[i] for i in range(s, e)]


def _preload_gather_any_preconverted(
    obj: object | None,
    idx_cpu: torch.Tensor,
    idx_np: object | None,
    *args: object,
    name: str,
) -> object | None:
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj[idx_cpu]
    if isinstance(obj, numpy.ndarray):
        return obj[idx_np if idx_np is not None else idx_cpu.numpy()]
    try:
        return obj[idx_np if idx_np is not None else idx_cpu.numpy()]
    except Exception:
        return [obj[int(i)] for i in idx_cpu.tolist()]


def _normalize_device_spec(
    device: torch.device | str | Sequence[torch.device | str],
) -> torch.device | list[torch.device]:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    if isinstance(device, Sequence) and not isinstance(
        device, (str, bytes, bytearray)
    ):
        devs: list[torch.device] = []
        for d in device:
            devs.append(
                d if isinstance(d, torch.device) else torch.device(str(d))
            )
        return devs if devs else torch.device("cpu")
    return torch.device(device)


def _primary_device(
    device_spec: torch.device | list[torch.device],
) -> torch.device:
    return (
        device_spec[0]
        if isinstance(device_spec, list) and device_spec
        else device_spec
    )


def _resolve_memmap_store_float(*args: Any, negotiable: bool) -> torch.dtype:
    req = str(env_str("ENN_MEMMAP_FLOAT_DTYPE") or "").strip()
    if req.startswith("torch."):
        req = req.split(".", 1)[1]
    req_dtype = getattr(torch, req, None) if req else None
    if not isinstance(req_dtype, torch.dtype):
        req_dtype = torch.float32
    try:
        if not torch.is_floating_point(torch.empty((), dtype=req_dtype)):
            req_dtype = torch.float32
    except Exception:
        req_dtype = torch.float32
    return (
        torch.float32
        if (bool(negotiable) and req_dtype != torch.float64)
        else torch.float64
    )


def _to_cpu_contig(t: torch.Tensor) -> torch.Tensor:
    t = t.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    if not t.is_contiguous():
        t = t.contiguous()
    return t


def _flat2d_cpu_contig(t: torch.Tensor, n: int) -> torch.Tensor:
    t_cpu = _to_cpu_contig(t)
    if t_cpu.ndim == 0:
        t_cpu = t_cpu.reshape(1)
    return t_cpu.reshape(int(n), -1)


def _batch_n(x: torch.Tensor) -> int:
    xd = int(getattr(x, "ndim", 0) or 0)
    return int(x.shape[0]) if xd > 0 else 1


def _idx_to_cpu_int64(idx: Any) -> torch.Tensor:
    if not isinstance(idx, torch.Tensor):
        idx = torch.as_tensor(idx)
    if idx.device.type != "cpu":
        idx = idx.detach().cpu()
    if idx.dtype not in (torch.int64, torch.int32):
        idx = idx.to(dtype=torch.int64, copy=False)
    idx = idx.reshape(-1)
    return idx.to(dtype=torch.int64, copy=False)


def _validate_row_contiguity(rows: torch.Tensor) -> tuple[bool, int, int]:
    rows = rows.reshape(-1)
    n = int(rows.numel())
    if n <= 0:
        return (True, 0, 0)
    start = int(rows[0].item())
    if n == 1:
        return (True, start, start + 1)
    last = int(rows[-1].item())
    if last != start + n - 1:
        return (False, 0, 0)
    if not bool(torch.all(rows[1:] == rows[:-1] + 1)):
        return (False, 0, 0)
    return (True, start, start + n)


def _index_copy_rows(
    dst: object,
    rows_t: torch.Tensor,
    preds_t: torch.Tensor,
    *args: Any,
    count: int,
) -> None:
    if preds_t.shape[0] != rows_t.shape[0]:
        raise ValueError(
            f"Pred/rows mismatch: preds[0]={preds_t.shape[0]} vs rows={rows_t.shape[0]}"
        )
    is_contig, start, end = _validate_row_contiguity(rows_t)
    if is_contig:
        if start < 0 or end > int(count):
            raise ValueError(
                f"Row indices out of bounds: [{start}, {end}) vs count={int(count)}"
            )
        dst[start:end].copy_(preds_t)
        return
    if rows_t.numel() > 0:
        rmin = int(rows_t.min().item())
        rmax = int(rows_t.max().item())
        if rmin < 0 or rmax >= int(count):
            raise ValueError(
                f"Row indices out of bounds: min={rmin}, max={rmax}, count={int(count)}"
            )
    dst.index_copy_(0, rows_t, preds_t)


def _h5_write_rows(
    dset_Y: object,
    rows_t: torch.Tensor,
    preds_np: object,
    *args: Any,
    count: int,
) -> None:
    is_contig, start, end = _validate_row_contiguity(rows_t)
    if is_contig:
        if start < 0 or end > int(count):
            raise ValueError(
                f"Row indices out of bounds: [{start}, {end}) vs count={int(count)}"
            )
        dset_Y[start:end] = preds_np
        return
    rows_np = rows_t.detach().to(device="cpu", dtype=torch.int64).numpy()
    if rows_np.size:
        rmin = int(rows_np.min())
        rmax = int(rows_np.max())
        if rmin < 0 or rmax >= int(count):
            raise ValueError(
                f"Row indices out of bounds: min={rmin}, max={rmax}, count={int(count)}"
            )
    dset_Y[rows_np] = preds_np


def _torch_load_cpu(path: str) -> object:
    return torch.load(path, map_location="cpu", weights_only=True)


def _load_row(rows_file: str) -> torch.Tensor:
    rows_t = _torch_load_cpu(os.fspath(rows_file))
    if not isinstance(rows_t, torch.Tensor):
        rows_t = torch.as_tensor(rows_t, device="cpu")
    rows_t = rows_t.reshape(-1).to(dtype=torch.int64, device="cpu", copy=False)
    if not bool(rows_t.is_contiguous()):
        rows_t = rows_t.contiguous()
    return rows_t


def _load_prediction(
    pred_file: str, *args: Any, dtype: torch.dtype
) -> torch.Tensor:
    _ = args
    pf = os.fspath(pred_file)
    if pf.endswith(".mmt"):
        preds_t = open_memory_mapped_tensor(pf)
        if preds_t is None:
            raise FileNotFoundError(
                f"missing prediction memmap or meta: {pf!r}"
            )
    else:
        preds_t = _torch_load_cpu(pf)
    if not isinstance(preds_t, torch.Tensor):
        preds_t = torch.as_tensor(preds_t, device="cpu")
    if preds_t.device.type != "cpu":
        preds_t = preds_t.to(device="cpu")
    return preds_t.to(dtype=dtype, copy=False)


def _to_numpy_dtype(dtype: torch.dtype) -> type[numpy.generic]:
    np_bfloat16 = getattr(numpy, "bfloat16", numpy.float32)
    mapping = {
        torch.float16: numpy.float16,
        torch.float32: numpy.float32,
        torch.float64: numpy.float64,
        torch.bfloat16: np_bfloat16,
        torch.int8: numpy.int8,
        torch.uint8: numpy.uint8,
        torch.int16: numpy.int16,
        torch.int32: numpy.int32,
        torch.int64: numpy.int64,
        torch.bool: numpy.bool_,
    }
    return mapping.get(dtype, numpy.float32)


def _atomic_h5_op(
    out_path: str | os.PathLike[str],
    overwrite: str | None,
    op_fn: Callable[[str], None],
) -> PersistentTensorDict:
    out_path = os.fspath(out_path)
    ow = str(overwrite or "replace").strip().lower()
    if ow == "resume" and os.path.isfile(out_path):
        return PersistentTensorDict(filename=out_path, mode="r")
    if ow == "error" and os.path.exists(out_path):
        raise FileExistsError(f"Exists: {out_path}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=os.path.basename(out_path) + ".",
        suffix=".tmp",
        dir=os.path.dirname(out_path),
    )
    os.close(fd)
    try:
        op_fn(tmp)
        if ow == "replace":
            os.replace(tmp, out_path)
        elif ow in ("error", "resume"):
            if not os.path.exists(out_path):
                os.link(tmp, out_path)
            os.remove(tmp)
        else:
            raise ValueError(f"Invalid overwrite={overwrite}")
    finally:
        with suppress(Exception):
            os.remove(tmp)
    return PersistentTensorDict(filename=out_path, mode="r")


def preprocess(
    batch: Any,
    *args: Any,
    flatten_features: bool,
    labels_dtype: Optional[torch.dtype],
    sanitize: bool,
    **kwargs: Any,
) -> Dict[str, Any]:
    features: Any = None
    labels: Any = None
    if isinstance(batch, (Mapping, TensorDictBase)):
        with contextlib.suppress(Exception):
            features, labels = get_row(batch, labels_required=False)
    if features is None and isinstance(batch, Mapping):
        features = batch.get("X")
        labels = batch.get("Y", None)
    if (
        flatten_features
        and isinstance(features, torch.Tensor)
        and (features.dim() >= 2)
    ):
        features = features.reshape(features.shape[0], -1)
    if labels_dtype is not None and isinstance(labels, torch.Tensor):
        labels = labels.to(dtype=labels_dtype, non_blocking=True, copy=False)
    if (
        sanitize
        and isinstance(labels, torch.Tensor)
        and labels.is_floating_point()
    ):
        torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0, out=labels)
    out: Dict[str, Any] = {"X": features}
    if labels is not None:
        out["Y"] = labels
    if isinstance(batch, (Mapping, TensorDictBase)):
        with contextlib.suppress(Exception):
            if "row_ids" in batch:
                out["row_ids"] = batch.get("row_ids")
    return out


def column_cursor(
    data: Mapping[Any, Any],
    *args: Any,
    keys: Optional[Sequence[Any]] = None,
) -> Tuple[int, Callable[[int, int], _KeyView]]:
    count = int(len(data) if keys is None else len(keys))
    if count <= 0:
        raise ValueError("Empty mapping: no keys")
    return count, _KeyCursor(data, keys)


def stream_memmap(
    *args: Any,
    ds: Any,
    out_dir: str,
    count: int,
    get_batch: Any,
    val_frac: float,
    seed_value: Any,
    underflow_action: str,
    shuffle: bool = False,
    get_by_indices: Any = None,
    default_label_shape: Any = None,
    allow_missing_labels: bool = False,
    features_only: bool = False,
    chunk_size: int = 32,
) -> Tuple[int, Tuple[int, ...]]:
    from .pipeline import Dataset

    out_dir = os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    count_i = int(count)
    if count_i <= 0:
        raise ValueError("count must be > 0")
    env_chunk = env_first_int(
        ("ENN_MEMMAP_CHUNK_SIZE", "ENN_MEMMAP_CHUNK"), 0
    )
    if int(env_chunk) > 0:
        chunk_size = int(env_chunk)
    req_chunk = int(chunk_size or 0)
    auto_chunk = req_chunk <= 0
    if not auto_chunk:
        chunk_first = max(1, min(count_i, req_chunk))
    else:
        chunk_first = 1
    allow_missing = bool(allow_missing_labels) or bool(features_only)
    default_lshape = (
        tuple(default_label_shape) if default_label_shape is not None else (1,)
    )
    stats: Dict[str, Any] = {
        "has_scale": False,
        "has_nonfinite": False,
        "scale_max_abs": None,
        "scale_min_value": None,
        "scale_max_value": None,
        "scale_min_positive": None,
        "scale_is_integral": None,
    }
    in_dim: Optional[int] = None
    label_shape: Optional[Tuple[int, ...]] = None
    s_i = 0
    auto_chunk_adjusted = not bool(auto_chunk)
    while int(s_i) < int(count_i):
        e_i = min(int(count_i), int(s_i) + int(chunk_first))
        batch = get_batch(int(s_i), int(e_i))
        fx, lb, _, _ = ds.preprocess(batch, return_keys=False)
        n = _batch_n(fx)
        if n <= 0:
            s_i = int(e_i)
            continue
        expected = int(e_i) - int(s_i)
        if int(n) != int(expected):
            raise RuntimeError(
                f"Pass1 batch size mismatch for out_dir={out_dir!r}: expected {expected}, got {n} (s={s_i}, e={e_i})."
            )
        fx_flat = _flat2d_cpu_contig(fx, n)
        cur_in_dim = int(fx_flat.shape[1])
        if in_dim is None:
            in_dim = cur_in_dim
        elif cur_in_dim != int(in_dim):
            raise RuntimeError(
                f"feature dim mismatch: expected {in_dim}, got {cur_in_dim}"
            )
        if lb is None:
            if not allow_missing:
                raise RuntimeError("memmap writer requires labels (got None)")
            cur_label_shape = tuple(default_lshape)
            lb_flat = None
        else:
            cur_label_shape = tuple(lb.shape[1:])
            lb_flat = _flat2d_cpu_contig(lb, n)
        if label_shape is None:
            label_shape = cur_label_shape
        elif tuple(label_shape) != tuple(cur_label_shape):
            raise RuntimeError(
                f"label shape mismatch: expected {label_shape}, got {cur_label_shape}"
            )
        if not bool(auto_chunk_adjusted):
            try:
                target_bytes = int(
                    env_first_int(("ENN_MEMMAP_CHUNK_BYTES",), 0)
                )
                if target_bytes <= 0:
                    target_mb = int(
                        env_first_int(("ENN_MEMMAP_CHUNK_MB",), 64)
                    )
                    target_bytes = int(target_mb) * 1024 * 1024
                avail = int(Memory.available() or 0)
                if avail > 0:
                    target_bytes = int(
                        min(
                            int(target_bytes),
                            max(8 * 1024 * 1024, int(avail) // 16),
                        )
                    )
                elem_size = int(fx_flat.element_size())
                if lb_flat is not None:
                    elem_size = max(elem_size, int(lb_flat.element_size()))
                label_numel = (
                    0 if bool(features_only) else int(numpy.prod(label_shape))
                )
                row_bytes = max(
                    1, (int(in_dim) + int(label_numel)) * elem_size
                )
                new_chunk = int(
                    max(
                        1,
                        min(int(count_i), int(target_bytes) // int(row_bytes)),
                    )
                )
                pass1_max = int(
                    env_first_int(("ENN_MEMMAP_PASS1_CHUNK_MAX",), 256)
                )
                if pass1_max > 0:
                    new_chunk = min(new_chunk, pass1_max)
                chunk_first = int(max(1, new_chunk))
            except Exception:
                pass
            auto_chunk_adjusted = True
        f_stats = Dataset.tensor_scale_stats(fx_flat)
        if bool(features_only):
            stats = Dataset.merge_scale_stats(stats, f_stats)
        else:
            if lb_flat is None:
                l_stats = {
                    "has_scale": True,
                    "has_nonfinite": False,
                    "scale_max_abs": 0.0,
                    "scale_min_value": 0.0,
                    "scale_max_value": 0.0,
                    "scale_min_positive": None,
                    "scale_is_integral": None,
                }
            else:
                l_stats = Dataset.tensor_scale_stats(lb_flat)
            stats = Dataset.merge_scale_stats(
                stats, Dataset.merge_scale_stats(f_stats, l_stats)
            )
        s_i = int(e_i)
    if in_dim is None or label_shape is None:
        raise RuntimeError("Failed to infer in_dim/label_shape from data")
    negotiable = Dataset.is_fp32_castable(
        stats,
        underflow_action=underflow_action,
        safety_margin=1.0,
    )
    store_float = _resolve_memmap_store_float(negotiable=bool(negotiable))
    if auto_chunk:
        elem_size = int(torch.empty((), dtype=store_float).element_size())
        label_numel = (
            0 if bool(features_only) else int(numpy.prod(label_shape))
        )
        row_bytes = max(1, (int(in_dim) + int(label_numel)) * int(elem_size))
        target_bytes = env_first_int(("ENN_MEMMAP_CHUNK_BYTES",), 0)
        if int(target_bytes) <= 0:
            target_mb = env_first_int(("ENN_MEMMAP_CHUNK_MB",), 64)
            target_bytes = int(target_mb) * 1024 * 1024
        try:
            avail = int(Memory.available() or 0)
            if avail > 0:
                target_bytes = int(
                    min(int(target_bytes), max(8 * 1024 * 1024, avail // 16))
                )
        except Exception:
            pass
        chunk_second = int(
            max(1, min(count_i, int(target_bytes) // int(row_bytes)))
        )
    else:
        chunk_second = int(max(1, min(count_i, req_chunk)))
    val_count = max(0, min(count_i, int(round(count_i * float(val_frac)))))
    train_count = max(0, count_i - val_count)
    train_start, train_end = 0, int(train_count)
    val_start, val_end = int(train_end), int(train_end) + int(val_count)
    features_path = os.path.join(out_dir, "features.mmt")
    labels_path = os.path.join(out_dir, "labels.mmt")
    features_mmt = MemoryMappedTensor.empty(
        (count_i, int(in_dim)),
        dtype=store_float,
        filename=features_path,
        existsok=True,
    )
    write_labels = not bool(features_only)
    labels_mmt = None
    if write_labels:
        labels_mmt = MemoryMappedTensor.empty(
            (count_i, *tuple(label_shape)),
            dtype=store_float,
            filename=labels_path,
            existsok=True,
        )
    shuffle_indexer = None
    shuffle_impl = "none"
    order: Optional[torch.Tensor] = None
    shuffle_seed: Optional[int] = None
    if bool(shuffle):
        if get_by_indices is None:
            raise ValueError("shuffle=True requires get_by_indices")
        max_elems = env_first_int(
            (
                "ENN_MEMMAP_RANDPERM_MAX_ELEMS",
                "ENN_MEMMAP_SHUFFLE_MAX_ELEMS",
            ),
            5_000_000,
        )
        use_full = (max_elems is not None) and (count_i <= int(max_elems))
        seed_i = (
            None
            if seed_value is None
            else (int(seed_value) & 0x7FFFFFFFFFFFFFFF)
        )
        if use_full:
            g = None
            if seed_i is not None:
                g = torch.Generator(device="cpu")
                g.manual_seed(seed_i)
            shuffle_seed = seed_i
            order = torch.randperm(count_i, generator=g, dtype=torch.int64)

            def _idx(s: int, e: int) -> torch.Tensor:
                assert order is not None
                return order[int(s) : int(e)]

            shuffle_indexer = _idx
            shuffle_impl = "randperm"
        else:
            if seed_i is None:
                seed_i = int(
                    torch.randint(0, 2**63 - 1, (1,), dtype=torch.int64).item()
                )
            shuffle_seed = seed_i
            k = max(1, int((count_i - 1)).bit_length())
            if (k % 2) == 1:
                k += 1
            half = k // 2
            mask = (1 << half) - 1
            domain_mask = (1 << k) - 1 if k < 64 else 0xFFFFFFFFFFFFFFFF
            seed_u = torch.tensor(
                seed_i & 0xFFFFFFFFFFFFFFFF, dtype=torch.uint64
            )
            mask_u = torch.tensor(mask, dtype=torch.uint64)
            domain_u = torch.tensor(domain_mask, dtype=torch.uint64)
            count_u = torch.tensor(count_i, dtype=torch.uint64)
            k0 = seed_u ^ torch.tensor(0x9E3779B97F4A7C15, dtype=torch.uint64)
            k1 = seed_u ^ torch.tensor(0xBF58476D1CE4E5B9, dtype=torch.uint64)
            k2 = seed_u ^ torch.tensor(0x94D049BB133111EB, dtype=torch.uint64)
            k3 = seed_u ^ torch.tensor(0xD6E8FEB86659FD93, dtype=torch.uint64)
            round_keys = (k0, k1, k2, k3)
            _c_mul1 = torch.tensor(0x9E3779B97F4A7C15, dtype=torch.uint64)
            _c_mul2 = torch.tensor(0xC2B2AE3D27D4EB4F, dtype=torch.uint64)

            def _round_fn(r: torch.Tensor, rk: torch.Tensor) -> torch.Tensor:
                x = (r ^ rk) & mask_u
                x = (x * _c_mul1) & domain_u
                x = (x ^ (x >> 33)) & domain_u
                x = (x * _c_mul2) & domain_u
                x = (x ^ (x >> 29)) & domain_u
                return x & mask_u

            def _feistel(x: torch.Tensor) -> torch.Tensor:
                x = x & domain_u
                l = (x >> half) & mask_u
                r = x & mask_u
                for rk in round_keys:
                    f = _round_fn(r, rk)
                    l, r = r, (l ^ f) & mask_u
                return ((l << half) | r) & domain_u

            def _gcd(a: int, b: int) -> int:
                while b:
                    a, b = b, a % b
                return abs(int(a))

            a0 = (seed_i | 1) % count_i
            if a0 == 0:
                a0 = 1
            while _gcd(a0, count_i) != 1:
                a0 = (a0 + 2) % count_i
                if a0 == 0:
                    a0 = 1
            b0 = seed_i % count_i

            def _affine(pos: torch.Tensor) -> torch.Tensor:
                p = pos.to(dtype=torch.int64)
                return ((p * int(a0) + int(b0)) % int(count_i)).to(
                    dtype=torch.int64
                )

            def _permute(pos: torch.Tensor) -> torch.Tensor:
                x = pos.to(dtype=torch.uint64)
                y = _feistel(x)
                bad = y >= count_u
                max_iter = 64
                it = 0
                while bool(bad.any()):
                    it += 1
                    if it > max_iter:
                        return _affine(pos)
                    y_bad = _feistel(y[bad])
                    y[bad] = y_bad
                    bad = y >= count_u
                return y.to(dtype=torch.int64)

            def _idx(s: int, e: int) -> torch.Tensor:
                pos = torch.arange(
                    int(s), int(e), device="cpu", dtype=torch.int64
                )
                return _permute(pos)

            shuffle_indexer = _idx
            shuffle_impl = "prp"
    compute_scaler_stats = bool(write_labels) and (
        not bool(allow_missing_labels)
    )
    x_sum: Optional[torch.Tensor] = None
    x_sum_sq: Optional[torch.Tensor] = None
    x_tmp: Optional[torch.Tensor] = None
    x2_tmp: Optional[torch.Tensor] = None
    y_sum: Optional[torch.Tensor] = None
    y_sum_sq: Optional[torch.Tensor] = None
    y_tmp: Optional[torch.Tensor] = None
    y2_tmp: Optional[torch.Tensor] = None
    x_min: Optional[torch.Tensor] = None
    x_max: Optional[torch.Tensor] = None
    x_min_tmp: Optional[torch.Tensor] = None
    x_max_tmp: Optional[torch.Tensor] = None
    y_min: Optional[torch.Tensor] = None
    y_max: Optional[torch.Tensor] = None
    y_min_tmp: Optional[torch.Tensor] = None
    y_max_tmp: Optional[torch.Tensor] = None
    if compute_scaler_stats and int(train_end) > 0:
        x_sum = torch.zeros(
            (int(in_dim),), dtype=torch.float64, device=torch.device("cpu")
        )
        x_sum_sq = torch.zeros(
            (int(in_dim),), dtype=torch.float64, device=torch.device("cpu")
        )
        x_tmp = torch.empty_like(x_sum)
        x2_tmp = torch.empty_like(x_sum)
        out_dim = int(numpy.prod(label_shape))
        y_sum = torch.zeros(
            (int(out_dim),), dtype=torch.float64, device=torch.device("cpu")
        )
        y_sum_sq = torch.zeros(
            (int(out_dim),), dtype=torch.float64, device=torch.device("cpu")
        )
        y_tmp = torch.empty_like(y_sum)
        y2_tmp = torch.empty_like(y_sum)
        x_min = torch.full(
            (int(in_dim),),
            float("inf"),
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        x_max = torch.full(
            (int(in_dim),),
            float("-inf"),
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        x_min_tmp = torch.empty_like(x_sum)
        x_max_tmp = torch.empty_like(x_sum)
        y_min = torch.full(
            (int(out_dim),),
            float("inf"),
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        y_max = torch.full(
            (int(out_dim),),
            float("-inf"),
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        y_min_tmp = torch.empty_like(y_sum)
        y_max_tmp = torch.empty_like(y_sum)
    written = 0
    features_cast_copy_ok: Optional[bool] = None
    labels_cast_copy_ok: Optional[bool] = None
    for s in range(0, count_i, int(chunk_second)):
        e = min(count_i, s + int(chunk_second))
        if shuffle_indexer is None:
            batch = get_batch(int(s), int(e))
        else:
            idx = shuffle_indexer(int(s), int(e))
            batch = get_by_indices(idx)
        fx, lb, _, _ = ds.preprocess(batch, return_keys=False)
        n = _batch_n(fx)
        if n <= 0:
            continue
        expected = int(e) - int(s)
        if n != expected:
            raise RuntimeError(
                f"Pass2 batch size mismatch for out_dir={out_dir!r}: expected {expected}, got {n} (s={s}, e={e})."
            )
        fx_flat = _flat2d_cpu_contig(fx, n)
        if int(fx_flat.shape[1]) != int(in_dim):
            raise RuntimeError(
                f"feature dim mismatch: expected {in_dim}, got {int(fx_flat.shape[1])}"
            )
        if (
            x_sum is not None
            and x_sum_sq is not None
            and x_tmp is not None
            and x2_tmp is not None
        ):
            end_pos = int(s) + int(n)
            overlap = max(0, min(end_pos, int(train_end)) - int(s))
            if overlap > 0:
                fx_stats = fx_flat[:overlap]
                if fx_stats.dtype != torch.float64:
                    fx_stats = fx_stats.to(dtype=torch.float64)
                torch.sum(fx_stats, dim=0, out=x_tmp)
                x_sum.add_(x_tmp)
                x2_tmp.copy_(torch.einsum("ni,ni->i", fx_stats, fx_stats))
                x_sum_sq.add_(x2_tmp)
                if (
                    x_min is not None
                    and x_max is not None
                    and x_min_tmp is not None
                    and x_max_tmp is not None
                ):
                    torch.amin(fx_stats, dim=0, out=x_min_tmp)
                    torch.minimum(x_min, x_min_tmp, out=x_min)
                    torch.amax(fx_stats, dim=0, out=x_max_tmp)
                    torch.maximum(x_max, x_max_tmp, out=x_max)
        dst_f = features_mmt[int(s) : int(s) + int(n)]
        if features_cast_copy_ok is None:
            try:
                dst_f.copy_(fx_flat)
                features_cast_copy_ok = True
            except Exception:
                features_cast_copy_ok = False
                dst_f.copy_(fx_flat.to(dtype=store_float))
        elif features_cast_copy_ok:
            dst_f.copy_(fx_flat)
        else:
            dst_f.copy_(fx_flat.to(dtype=store_float))
        if write_labels:
            assert labels_mmt is not None
            if lb is None:
                if not allow_missing:
                    raise RuntimeError(
                        "memmap writer requires labels (got None)"
                    )
                labels_mmt[int(s) : int(s) + int(n)].zero_()
            else:
                if tuple(lb.shape[1:]) != tuple(label_shape):
                    raise RuntimeError(
                        f"label shape mismatch: expected {label_shape}, got {tuple(lb.shape[1:])}"
                    )
                lb_cpu = _to_cpu_contig(lb)
                dst_l = labels_mmt[int(s) : int(s) + int(n)]
                if labels_cast_copy_ok is None:
                    try:
                        dst_l.copy_(lb_cpu)
                        labels_cast_copy_ok = True
                    except Exception:
                        labels_cast_copy_ok = False
                        dst_l.copy_(lb_cpu.to(dtype=store_float))
                elif labels_cast_copy_ok:
                    dst_l.copy_(lb_cpu)
                else:
                    dst_l.copy_(lb_cpu.to(dtype=store_float))
                if (
                    y_sum is not None
                    and y_sum_sq is not None
                    and y_tmp is not None
                    and y2_tmp is not None
                ):
                    end_pos = int(s) + int(n)
                    overlap = max(0, min(end_pos, int(train_end)) - int(s))
                    if overlap > 0:
                        lb_stats = lb_cpu[:overlap].reshape(int(overlap), -1)
                        if lb_stats.dtype != torch.float64:
                            lb_stats = lb_stats.to(dtype=torch.float64)
                        torch.sum(lb_stats, dim=0, out=y_tmp)
                        y_sum.add_(y_tmp)
                        y2_tmp.copy_(
                            torch.einsum("ni,ni->i", lb_stats, lb_stats)
                        )
                        y_sum_sq.add_(y2_tmp)
                        if (
                            y_min is not None
                            and y_max is not None
                            and y_min_tmp is not None
                            and y_max_tmp is not None
                        ):
                            torch.amin(lb_stats, dim=0, out=y_min_tmp)
                            torch.minimum(y_min, y_min_tmp, out=y_min)
                            torch.amax(lb_stats, dim=0, out=y_max_tmp)
                            torch.maximum(y_max, y_max_tmp, out=y_max)
        written += int(n)
    if int(written) != int(count_i):
        raise RuntimeError(f"memmap written={written}, expected={count_i}")
    scaler_stats_path: Optional[str] = None
    if (
        compute_scaler_stats
        and int(train_end) > 0
        and x_sum is not None
        and x_sum_sq is not None
        and y_sum is not None
        and y_sum_sq is not None
    ):
        payload = {
            "version": 1,
            "train_count": int(train_end),
            "x_sum": x_sum,
            "x_sum_sq": x_sum_sq,
            "x_min": x_min,
            "x_max": x_max,
            "y_sum": y_sum,
            "y_sum_sq": y_sum_sq,
            "y_min": y_min,
            "y_max": y_max,
        }
        scaler_stats_path = "scaler_stats.pt"
        try:
            save_temp(os.path.join(out_dir, scaler_stats_path), payload)
        except Exception:
            scaler_stats_path = None
    meta_json: Dict[str, Any] = {
        "N": int(count_i),
        "feature_dim": int(in_dim),
        "features_path": "features.mmt",
        "labels_path": ("labels.mmt" if write_labels else None),
        "label_shape": list(label_shape),
        "features_dtype": str(store_float).replace("torch.", ""),
        "labels_dtype": (
            str(store_float).replace("torch.", "") if write_labels else None
        ),
        "fractions": [float(1.0 - float(val_frac)), float(val_frac)],
        "shuffled": bool(shuffle),
        "shuffle_seed": (
            int(shuffle_seed) if shuffle_seed is not None else None
        ),
        "shuffle_mode": "physical" if bool(shuffle) else "none",
        "shuffle_impl": shuffle_impl,
        "train_start": int(train_start),
        "train_end": int(train_end),
        "val_start": int(val_start),
        "val_end": int(val_end),
        "scaler_stats_path": scaler_stats_path,
        "has_scale": bool(stats.get("has_scale")),
        "has_nonfinite": bool(stats.get("has_nonfinite")),
        "scale_max_abs": stats.get("scale_max_abs"),
        "scale_min_value": stats.get("scale_min_value"),
        "scale_max_value": stats.get("scale_max_value"),
        "scale_min_positive": stats.get("scale_min_positive"),
        "scale_is_integral": stats.get("scale_is_integral"),
        "is_negotiable": bool(negotiable),
        "underflow_action": str(underflow_action),
        "features_only": bool(features_only),
    }
    write_json(os.path.join(out_dir, "meta.json"), meta_json, indent=2)
    return int(in_dim), tuple(label_shape)


def iter_source_path(obj: Any) -> None:
    if obj is None:
        return
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        if obj.get("kind") == "memmap" and isinstance(obj.get("path"), str):
            yield obj["path"]
        else:
            for v in obj.values():
                yield from iter_source_path(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from iter_source_path(v)


def from_meta(memmap_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(os.fspath(memmap_dir), "meta.json")
    raw = read_json(meta_path)
    return raw if isinstance(raw, dict) else {}


def merge_meta_info(metas: Any) -> Dict[str, Any]:
    def _merge_dicts(items: list[dict]) -> Dict[str, Any]:
        if not items:
            return {}
        base = dict(items[0])

        def _upd_min(k: str, v: float) -> None:
            base[k] = (
                v if base.get(k) is None else min(float(base[k]), float(v))
            )

        def _upd_max(k: str, v: float) -> None:
            base[k] = (
                v if base.get(k) is None else max(float(base[k]), float(v))
            )

        for m in items[1:]:
            if (fd := m.get("feature_dim")) and fd != base.get("feature_dim"):
                raise ValueError("feature_dim mismatch")
            if "label_shape" in m:
                ls = m.get("label_shape")
                if tuple(ls) != tuple(base.get("label_shape", [])):
                    raise ValueError("label_shape mismatch")
            base["has_scale"] |= _meta_has_scale(m)
            base["has_nonfinite"] |= bool(m.get("has_nonfinite"))
            if (v := m.get("scale_max_abs")) is not None:
                _upd_max("scale_max_abs", v)
            if (v := m.get("scale_min_value")) is not None:
                _upd_min("scale_min_value", v)
            if (v := m.get("scale_max_value")) is not None:
                _upd_max("scale_max_value", v)
            if (v := m.get("scale_min_positive")) is not None:
                _upd_min("scale_min_positive", v)
            if (v := m.get("scale_is_integral")) is not None:
                base["scale_is_integral"] = bool(v) and base.get(
                    "scale_is_integral", True
                )
            if (v := m.get("is_negotiable")) is not None:
                base["is_negotiable"] = bool(v) and base.get(
                    "is_negotiable", True
                )
            base["underflow_action"] = _strictest_underflow_action(
                base.get("underflow_action"), m.get("underflow_action")
            )
        return base

    if metas is None:
        return {}
    if isinstance(metas, list) and metas and isinstance(metas[0], dict):
        return _merge_dicts(metas)
    collected: list[dict] = []
    for path in iter_source_path(metas):
        try:
            meta = from_meta(path)
        except Exception:
            meta = None
        if isinstance(meta, dict):
            collected.append(meta)
    return _merge_dicts(collected)


def load_scaler_stats(sources: Any) -> Optional[Dict[str, Any]]:
    expanded = expand_source(sources)
    total = 0
    x_sum: Optional[torch.Tensor] = None
    x_sum_sq: Optional[torch.Tensor] = None
    y_sum: Optional[torch.Tensor] = None
    y_sum_sq: Optional[torch.Tensor] = None
    x_min: Optional[torch.Tensor] = None
    x_max: Optional[torch.Tensor] = None
    y_min: Optional[torch.Tensor] = None
    y_max: Optional[torch.Tensor] = None
    have_bounds: Optional[bool] = None
    y_q_low: Optional[torch.Tensor] = None
    y_q_high: Optional[torch.Tensor] = None
    have_qbounds: Optional[bool] = None
    for path in iter_source_path(expanded):
        try:
            meta = from_meta(path)
        except Exception:
            return None
        rel = meta.get("scaler_stats_path")
        if not rel:
            return None
        stats_path = os.path.join(os.fspath(path), os.fspath(rel))
        if not os.path.isfile(stats_path):
            return None
        try:
            payload = torch.load(
                stats_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            payload = torch.load(stats_path, map_location="cpu")
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if int(payload.get("version") or 0) != 1:
            return None
        c = int(payload.get("train_count") or 0)
        if c <= 0:
            return None
        xs = payload.get("x_sum")
        xs2 = payload.get("x_sum_sq")
        ys = payload.get("y_sum")
        ys2 = payload.get("y_sum_sq")
        if xs is None or xs2 is None or ys is None or ys2 is None:
            return None
        xs = (
            xs.detach().to(dtype=torch.float64, device="cpu")
            if isinstance(xs, torch.Tensor)
            else torch.as_tensor(xs, dtype=torch.float64)
        )
        xs2 = (
            xs2.detach().to(dtype=torch.float64, device="cpu")
            if isinstance(xs2, torch.Tensor)
            else torch.as_tensor(xs2, dtype=torch.float64)
        )
        ys = (
            ys.detach().to(dtype=torch.float64, device="cpu")
            if isinstance(ys, torch.Tensor)
            else torch.as_tensor(ys, dtype=torch.float64)
        )
        ys2 = (
            ys2.detach().to(dtype=torch.float64, device="cpu")
            if isinstance(ys2, torch.Tensor)
            else torch.as_tensor(ys2, dtype=torch.float64)
        )
        local_xmin = payload.get("x_min")
        local_xmax = payload.get("x_max")
        local_ymin = payload.get("y_min")
        local_ymax = payload.get("y_max")
        local_yq_low = payload.get("y_q_low")
        local_yq_high = payload.get("y_q_high")
        local_have_bounds = (
            local_xmin is not None
            and local_xmax is not None
            and local_ymin is not None
            and local_ymax is not None
        )
        if have_bounds is None:
            have_bounds = bool(local_have_bounds)
        elif have_bounds and not local_have_bounds:
            have_bounds = False
            x_min = x_max = y_min = y_max = None
        local_have_q = local_yq_low is not None and local_yq_high is not None
        if have_qbounds is None:
            have_qbounds = bool(local_have_q)
        elif have_qbounds and not local_have_q:
            have_qbounds = False
            y_q_low = y_q_high = None
        if have_bounds:
            local_xmin = (
                local_xmin.detach().to(dtype=torch.float64, device="cpu")
                if isinstance(local_xmin, torch.Tensor)
                else torch.as_tensor(local_xmin, dtype=torch.float64)
            )
            local_xmax = (
                local_xmax.detach().to(dtype=torch.float64, device="cpu")
                if isinstance(local_xmax, torch.Tensor)
                else torch.as_tensor(local_xmax, dtype=torch.float64)
            )
            local_ymin = (
                local_ymin.detach().to(dtype=torch.float64, device="cpu")
                if isinstance(local_ymin, torch.Tensor)
                else torch.as_tensor(local_ymin, dtype=torch.float64)
            )
            local_ymax = (
                local_ymax.detach().to(dtype=torch.float64, device="cpu")
                if isinstance(local_ymax, torch.Tensor)
                else torch.as_tensor(local_ymax, dtype=torch.float64)
            )
        if have_qbounds:
            local_yq_low = (
                local_yq_low.detach().to(dtype=torch.float64, device="cpu")
                if isinstance(local_yq_low, torch.Tensor)
                else torch.as_tensor(local_yq_low, dtype=torch.float64)
            )
            local_yq_high = (
                local_yq_high.detach().to(dtype=torch.float64, device="cpu")
                if isinstance(local_yq_high, torch.Tensor)
                else torch.as_tensor(local_yq_high, dtype=torch.float64)
            )
        if x_sum is None:
            x_sum = xs.clone()
            x_sum_sq = xs2.clone()
            y_sum = ys.clone()
            y_sum_sq = ys2.clone()
            if have_bounds:
                x_min = local_xmin.clone()
                x_max = local_xmax.clone()
                y_min = local_ymin.clone()
                y_max = local_ymax.clone()
            if have_qbounds:
                y_q_low = local_yq_low.clone()
                y_q_high = local_yq_high.clone()
        else:
            if xs.shape != x_sum.shape or xs2.shape != x_sum_sq.shape:
                return None
            if ys.shape != y_sum.shape or ys2.shape != y_sum_sq.shape:
                return None
            x_sum += xs
            x_sum_sq += xs2
            y_sum += ys
            y_sum_sq += ys2
            if have_bounds:
                assert (
                    x_min is not None
                    and x_max is not None
                    and y_min is not None
                    and y_max is not None
                )
                if (
                    local_xmin.shape != x_min.shape
                    or local_xmax.shape != x_max.shape
                ):
                    return None
                if (
                    local_ymin.shape != y_min.shape
                    or local_ymax.shape != y_max.shape
                ):
                    return None
                torch.minimum(x_min, local_xmin, out=x_min)
                torch.maximum(x_max, local_xmax, out=x_max)
                torch.minimum(y_min, local_ymin, out=y_min)
                torch.maximum(y_max, local_ymax, out=y_max)
            if have_qbounds:
                assert y_q_low is not None and y_q_high is not None
                if (
                    local_yq_low.shape != y_q_low.shape
                    or local_yq_high.shape != y_q_high.shape
                ):
                    return None
                torch.minimum(y_q_low, local_yq_low, out=y_q_low)
                torch.maximum(y_q_high, local_yq_high, out=y_q_high)
        total += c
    if (
        total <= 0
        or x_sum is None
        or x_sum_sq is None
        or y_sum is None
        or y_sum_sq is None
    ):
        return None
    out: Dict[str, Any] = {
        "train_count": int(total),
        "x_sum": x_sum,
        "x_sum_sq": x_sum_sq,
        "y_sum": y_sum,
        "y_sum_sq": y_sum_sq,
    }
    if (
        have_bounds
        and x_min is not None
        and x_max is not None
        and y_min is not None
        and y_max is not None
    ):
        out.update(
            {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
            }
        )
    if have_qbounds and y_q_low is not None and y_q_high is not None:
        out.update(
            {
                "y_q_low": y_q_low,
                "y_q_high": y_q_high,
            }
        )
    return out


def expand_source(sources: Any) -> Any:
    expanded, ok = _expand_multinode_sources(sources)
    if ok:
        return expanded
    if isinstance(sources, (list, tuple)) and len(sources) == 1:
        expanded, ok = _expand_multinode_sources(sources[0])
        if ok:
            return expanded
    return sources


def load_memmap_meta(memmap_dir: str) -> Mapping[str, Any]:
    meta_path = os.path.join(os.fspath(memmap_dir), "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"memmap meta.json not found: {meta_path}")
    meta = read_json(meta_path)
    if not isinstance(meta, dict):
        raise ValueError(f"memmap meta.json malformed: {meta_path}")
    return meta


def open_memory_mapped_tensor(mmt_path: str) -> Optional[MemoryMappedTensor]:
    meta_path = get_meta_path(mmt_path)
    if not (os.path.isfile(mmt_path) and os.path.isfile(meta_path)):
        return None
    try:
        meta = read_json(meta_path)
        if not isinstance(meta, dict):
            return None
        dtype = parse_torch_dtype(meta.get("dtype")) or torch.float32
        shape = tuple(int(x) for x in (meta.get("shape") or ()))
        if not isinstance(dtype, torch.dtype) or not shape:
            return None
        return MemoryMappedTensor.from_filename(
            filename=mmt_path, dtype=dtype, shape=torch.Size(shape)
        )
    except Exception:
        return None


def load_memmap_features(memmap_dir: str) -> MemoryMappedTensor:
    meta = load_memmap_meta(memmap_dir)
    n = int(meta.get("N", 0) or 0)
    if n <= 0:
        raise ValueError(
            f"memmap meta.json under {memmap_dir} has non-positive N={n}"
        )
    feat_rel = str(meta.get("features_path", "features.mmt"))
    feat_path = os.path.join(os.fspath(memmap_dir), feat_rel)
    fdim = int(meta.get("feature_dim", 0) or 0)
    if fdim <= 0:
        raise ValueError(
            f"memmap meta.json under {memmap_dir} has non-positive feature_dim={fdim}"
        )
    f_dtype = dtype_from_name(
        meta.get("features_dtype", "float64"), torch.float64
    )
    return MemoryMappedTensor.from_filename(
        feat_path, dtype=f_dtype, shape=torch.Size([n, fdim])
    )


def copy_mmt_to_cpu_tensor(
    mmt: object,
    *args: Any,
    count: object | None = None,
    chunk_size: int = 8192,
) -> torch.Tensor:
    _ = args
    if mmt is None:
        raise ValueError("copy_mmt_to_cpu_tensor: mmt must not be None")
    n = (
        int(count)
        if count is not None
        else int(getattr(mmt, "shape", [0])[0] or 0)
    )
    if n < 0:
        raise ValueError(f"copy_mmt_to_cpu_tensor: invalid count={count!r}")
    shape = tuple(int(x) for x in getattr(mmt, "shape", (n,)))
    if not shape:
        raise ValueError("copy_mmt_to_cpu_tensor: failed to infer shape")
    out_shape = (n,) + tuple(shape[1:])
    dtype = getattr(mmt, "dtype", None)
    out = (
        torch.empty(out_shape, dtype=dtype, device="cpu")
        if dtype is not None
        else torch.empty(out_shape, device="cpu")
    )
    step = max(1, int(chunk_size))
    for s in range(0, n, step):
        e = min(n, s + step)
        chunk = mmt[s:e]
        if not bool(torch.is_tensor(chunk)):
            chunk = torch.as_tensor(chunk)
        chunk_cpu = chunk.detach().to(device="cpu", dtype=out.dtype)
        out[s:e].copy_(chunk_cpu)
    return out


def load_predictions_h5(path: str) -> TensorDict:
    p = os.fspath(path)
    if not p or not os.path.isfile(p):
        raise FileNotFoundError(f"predictions .h5 not found: {path!r}")
    with h5py.File(p, "r") as f:
        if "X" not in f or "Y" not in f:
            raise KeyError(f"predictions file missing X/Y datasets: {p!r}")
        x_np = f["X"][...]
        y_np = f["Y"][...]
    x_t = torch.as_tensor(x_np).clone()
    y_t = torch.as_tensor(y_np).clone()
    return TensorDict({"X": x_t, "Y": y_t}, batch_size=[int(x_t.shape[0])])


def validate_predictions_h5(
    path: str,
    *args: Any,
    out_shape: object | None = None,
    in_dim: object | None = None,
) -> int:
    _ = args
    p = os.fspath(path)
    if not p or not os.path.isfile(p):
        raise FileNotFoundError(f"predictions file not found: {path!r}")
    with h5py.File(p, "r") as f:
        if "X" not in f or "Y" not in f:
            raise KeyError(f"predictions file missing X/Y datasets: {p!r}")
        dX = f["X"]
        dY = f["Y"]
        x_shape = tuple(int(x) for x in (getattr(dX, "shape", ()) or ()))
        y_shape = tuple(int(x) for x in (getattr(dY, "shape", ()) or ()))
        try:
            x_kind = getattr(getattr(dX, "dtype", None), "kind", "")
            y_kind = getattr(getattr(dY, "dtype", None), "kind", "")
        except Exception:
            x_kind = y_kind = ""
        if x_kind in ("O", "S", "U", "V") or y_kind in ("O", "S", "U", "V"):
            raise ValueError(
                f"predictions file has unsupported dtypes: X.dtype={getattr(dX, 'dtype', None)!r}, "
                f"Y.dtype={getattr(dY, 'dtype', None)!r} ({p!r})"
            )
        if len(x_shape) != 2 or len(y_shape) < 1:
            raise ValueError(
                f"predictions file has invalid shapes: X={x_shape}, Y={y_shape} ({p!r})"
            )
        if in_dim is not None and int(x_shape[1]) != int(in_dim):
            raise ValueError(
                f"predictions file has unexpected X feature_dim: got {int(x_shape[1])}, expected {int(in_dim)} ({p!r})"
            )
        if int(x_shape[0]) != int(y_shape[0]):
            raise ValueError(
                f"predictions file has mismatched row counts: X[0]={x_shape[0]}, Y[0]={y_shape[0]} ({p!r})"
            )
        n = int(x_shape[0])
        if n <= 0:
            raise ValueError(
                f"predictions file has non-positive row count: {n} ({p!r})"
            )
        if out_shape is None:
            if len(y_shape) < 2:
                raise ValueError(
                    f"predictions file has invalid Y shape: Y={y_shape} ({p!r})"
                )
        else:
            out_shape_t = tuple(int(d) for d in out_shape)
            if not out_shape_t or any(int(d) <= 0 for d in out_shape_t):
                raise ValueError(
                    f"validate_predictions_h5: invalid out_shape={out_shape!r}"
                )
            if len(y_shape) != 1 + len(out_shape_t):
                raise ValueError(
                    f"predictions file has unexpected Y rank: got {len(y_shape)}, expected {1 + len(out_shape_t)} ({p!r})"
                )
            if tuple(int(d) for d in y_shape[1:]) != out_shape_t:
                raise ValueError(
                    f"predictions file has unexpected Y shape: got {tuple(int(d) for d in y_shape[1:])}, expected {out_shape_t} ({p!r})"
                )
    return int(n)


def concat_memory_mapped_tensor(
    chunks_dir: str,
    out_path: str,
    *args: Any,
    count: object,
    out_shape: object,
    store_float: object,
) -> MemoryMappedTensor:
    _ = args
    out_shape_t = tuple(int(x) for x in out_shape)
    full_shape = torch.Size([int(count), *out_shape_t])
    y_out = MemoryMappedTensor.empty(
        full_shape,
        dtype=store_float,
        filename=os.fspath(out_path),
        existsok=True,
    )
    manifest_path = os.path.join(os.fspath(chunks_dir), "manifest.json")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid manifest: {manifest_path}")
    if bool(manifest.get("variable_shape", False)):
        raise NotImplementedError(
            "Variable-shaped predictions cannot be assembled into a single dense MemoryMappedTensor. "
            "Please rerun with a fixed output shape."
        )
    parts = list(manifest.get("parts", []) or [])
    for part in parts:
        rows_file = os.path.join(chunks_dir, str(part["rows"]))
        pred_file = os.path.join(chunks_dir, str(part["pred"]))
        rows_t = _load_row(rows_file)
        preds_t = _load_prediction(pred_file, dtype=store_float)
        _index_copy_rows(y_out, rows_t, preds_t, count=int(count))
    pred_meta_path = get_meta_path(os.fspath(out_path))
    write_json(
        pred_meta_path,
        {
            "dtype": str(store_float).replace("torch.", ""),
            "shape": list(map(int, full_shape)),
        },
        indent=None,
    )
    return y_out


def concat_tensor(
    chunks_dir: str,
    *args: Any,
    count: object,
    out_shape: object,
    dtype: torch.dtype,
) -> torch.Tensor:
    _ = args
    out_shape_t = tuple(int(x) for x in out_shape)
    y_out = torch.empty((int(count), *out_shape_t), dtype=dtype, device="cpu")
    manifest_path = os.path.join(os.fspath(chunks_dir), "manifest.json")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid manifest: {manifest_path}")
    if bool(manifest.get("variable_shape", False)):
        raise NotImplementedError(
            "Variable-shaped predictions cannot be returned as a single dense Tensor. Please rerun with a fixed output shape."
        )
    parts = list(manifest.get("parts", []) or [])
    for part in parts:
        rows_file = os.path.join(chunks_dir, str(part["rows"]))
        pred_file = os.path.join(chunks_dir, str(part["pred"]))
        rows_t = _load_row(rows_file)
        preds_t = _load_prediction(pred_file, dtype=dtype)
        _index_copy_rows(y_out, rows_t, preds_t, count=int(count))
    return y_out


def _h5_filter_kwargs(
    h5_compression: str | None,
    h5_compression_opts: int | None,
    h5_shuffle: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if h5_compression:
        kwargs["compression"] = str(h5_compression)
        if h5_compression_opts is not None:
            kwargs["compression_opts"] = int(h5_compression_opts)
    if h5_shuffle:
        kwargs["shuffle"] = True
    return kwargs


def concat_segment_h5(
    out_path: str,
    *args: Any,
    memmap_dir: str,
    chunks_dir: str,
    count: object,
    out_shape: object,
    store_float: object,
    chunk_size: int = 8192,
    h5_compression: str | None = None,
    h5_compression_opts: int | None = None,
    h5_shuffle: bool = False,
) -> PersistentTensorDict:
    _ = args
    x_mmt = load_memmap_features(memmap_dir)
    out_shape_t = tuple(int(x) for x in out_shape)
    os.makedirs(os.path.dirname(out_path) or os.getcwd(), exist_ok=True)
    np_float = _to_numpy_dtype(store_float)
    cast_dtype = store_float
    if store_float == torch.bfloat16 and np_float == numpy.float32:
        cast_dtype = torch.float32
    step = int(chunk_size)
    h5_kwargs = _h5_filter_kwargs(
        h5_compression, h5_compression_opts, h5_shuffle
    )
    use_filters = bool(h5_kwargs)
    with h5py.File(out_path, "w") as f:
        dset_x_kwargs = dict(h5_kwargs)
        dset_y_kwargs = dict(h5_kwargs)
        if use_filters:
            dset_x_kwargs["chunks"] = (
                min(int(count), step),
                *[int(x) for x in x_mmt.shape[1:]],
            )
            dset_y_kwargs["chunks"] = (
                min(int(count), step),
                *out_shape_t,
            )
        dset_X = f.create_dataset(
            "X",
            shape=tuple(x_mmt.shape),
            dtype=_to_numpy_dtype(x_mmt.dtype),
            **dset_x_kwargs,
        )
        dset_Y = f.create_dataset(
            "Y",
            shape=(int(count), *out_shape_t),
            dtype=np_float,
            **dset_y_kwargs,
        )
        for s in range(0, int(count), step):
            e = min(int(count), s + step)
            dset_X[s:e] = x_mmt[s:e].detach().to(device="cpu").numpy()
        manifest = read_json(os.path.join(chunks_dir, "manifest.json"))
        if not isinstance(manifest, dict):
            raise ValueError(f"Invalid manifest under: {chunks_dir}")
        if bool(manifest.get("variable_shape", False)):
            raise NotImplementedError(
                "Variable-shaped predictions cannot be stored as a dense HDF5 dataset. Please rerun with a fixed output shape."
            )
        parts = list(manifest.get("parts", []) or [])
        for part in parts:
            rows_file = os.path.join(chunks_dir, str(part["rows"]))
            pred_file = os.path.join(chunks_dir, str(part["pred"]))
            rows_t = _load_row(rows_file)
            preds_t = _load_prediction(pred_file, dtype=cast_dtype)
            preds_np = (
                preds_t.detach().to(device="cpu", dtype=cast_dtype).numpy()
            )
            if preds_np.shape[0] != int(rows_t.numel()):
                raise ValueError(
                    f"Pred/rows mismatch in {pred_file}: preds[0]={preds_np.shape[0]} vs rows={int(rows_t.numel())}"
                )
            _h5_write_rows(dset_Y, rows_t, preds_np, count=int(count))
    return PersistentTensorDict(
        filename=out_path, batch_size=[int(count)], mode="r"
    )


def write_predictions_h5_from_memmap(
    out_path: str,
    *args: Any,
    memmap_dir: str,
    pred_path: str,
    count: object | None = None,
    chunk_size: int = 8192,
    h5_compression: str | None = None,
    h5_compression_opts: int | None = None,
    h5_shuffle: bool = False,
) -> PersistentTensorDict:
    _ = args
    x_mmt = load_memmap_features(memmap_dir)
    y_mmt = open_memory_mapped_tensor(os.fspath(pred_path))
    if y_mmt is None:
        raise FileNotFoundError(f"missing prediction memmap: {pred_path}")
    n = int(y_mmt.shape[0]) if count is None else int(count)
    if n <= 0:
        raise ValueError(f"non-positive prediction count: {n}")
    x_np_dtype = _to_numpy_dtype(x_mmt.dtype)
    y_np_dtype = _to_numpy_dtype(y_mmt.dtype)
    y_cast_dtype = y_mmt.dtype
    if y_mmt.dtype == torch.bfloat16 and y_np_dtype == numpy.float32:
        y_cast_dtype = torch.float32
    out_parent = os.path.dirname(out_path) or "."
    os.makedirs(out_parent, exist_ok=True)
    step = int(chunk_size)
    h5_kwargs = _h5_filter_kwargs(
        h5_compression, h5_compression_opts, h5_shuffle
    )
    with h5py.File(out_path, "w") as f:
        dset_X = f.create_dataset(
            "X",
            shape=(n, int(x_mmt.shape[1])),
            dtype=x_np_dtype,
            chunks=(min(n, step), int(x_mmt.shape[1])),
            **h5_kwargs,
        )
        dset_Y = f.create_dataset(
            "Y",
            shape=(n, *[int(x) for x in y_mmt.shape[1:]]),
            dtype=y_np_dtype,
            chunks=(min(n, step), *[int(x) for x in y_mmt.shape[1:]]),
            **h5_kwargs,
        )
        for s in range(0, n, step):
            e = min(n, s + step)
            dset_X[s:e] = (
                x_mmt[s:e].detach().to(device="cpu", dtype=x_mmt.dtype).numpy()
            )
            dset_Y[s:e] = (
                y_mmt[s:e]
                .detach()
                .to(device="cpu", dtype=y_cast_dtype)
                .numpy()
            )
    return PersistentTensorDict(
        filename=out_path, batch_size=[int(n)], mode="r"
    )


def write_predictions_h5_atomic(
    out_path: str,
    *args: Any,
    memmap_dir: str,
    pred_path: str,
    chunk_size: int = 8192,
    h5_compression: str | None = None,
    h5_compression_opts: int | None = None,
    h5_shuffle: bool = False,
    overwrite: str = "replace",
) -> PersistentTensorDict:
    return _atomic_h5_op(
        out_path,
        overwrite,
        lambda tmp: write_predictions_h5_from_memmap(
            tmp,
            memmap_dir=memmap_dir,
            pred_path=pred_path,
            chunk_size=chunk_size,
            h5_compression=h5_compression,
            h5_compression_opts=h5_compression_opts,
            h5_shuffle=h5_shuffle,
        ),
    )


def copy_predictions_h5_atomic(
    src_path: str,
    dst_path: str,
    *args: Any,
    overwrite: str = "replace",
    out_shape: object | None = None,
) -> PersistentTensorDict:
    validate_predictions_h5(src_path, out_shape=out_shape)
    res = _atomic_h5_op(
        dst_path, overwrite, lambda tmp: shutil.copy2(src_path, tmp)
    )
    validate_predictions_h5(dst_path, out_shape=out_shape)
    return res


def remove_prediction_artifacts(
    *args: Any, memmap_dir: str, pred_path: str
) -> None:
    _ = args
    try:
        meta = load_memmap_meta(memmap_dir)
        feat_rel = str(meta.get("features_path", "features.mmt"))
        feat_path = os.path.join(os.fspath(memmap_dir), feat_rel)
    except Exception:
        feat_path = None

    if feat_path:
        _remove_safe(os.fspath(feat_path))
        with suppress(Exception):
            _remove_safe(get_meta_path(os.fspath(feat_path)))
    _remove_safe(os.fspath(pred_path))
    with suppress(Exception):
        _remove_safe(get_meta_path(os.fspath(pred_path)))
    _remove_safe(os.path.join(os.fspath(memmap_dir), "meta.json"))
    with suppress(OSError):
        os.rmdir(os.fspath(memmap_dir))


def postprocess(
    source: PathLike,
    *args: Any,
    output: str | None = "memory",
    path: PathLike | None = None,
    overwrite: str = "error",
) -> Any:
    del args
    with torch.inference_mode():
        if not bool(source):
            raise ValueError("postprocess: 'source' must be a non-empty path")
        src = _coerce_path(source)
        if src is None:
            raise ValueError(
                "postprocess: 'source' is empty/None after normalization"
            )
        output_mode = _coerce_prediction_output(output)
        overwrite_mode = _coerce_prediction_overwrite(overwrite)
        out_path = None
        path_n = _coerce_path(path) if path is not None else None
        if output_mode == "file":
            if path_n is not None:
                run_id = (
                    os.path.basename(src.rstrip(os.sep))
                    or f"prediction-{os.getpid()}"
                )
                out_path = _coerce_prediction_path(path_n, run_id=run_id)
                if out_path is None:
                    _LOGGER.warning(
                        "postprocess: output=%r requires path to be a .h5/.hdf5 file. Got path=%r; falling back to output='memory'.",
                        output,
                        path,
                    )
                    output_mode = "memory"
                elif not _is_path_writable(out_path):
                    _LOGGER.warning(
                        "postprocess: output=%r path is not writable: %r; falling back to output='memory'.",
                        output,
                        out_path,
                    )
                    out_path = None
                    output_mode = "memory"
            else:
                output_mode = "memory"

        if (
            output_mode == "file"
            and out_path is not None
            and os.path.exists(out_path)
        ):
            if overwrite_mode == "resume" and os.path.isfile(out_path):
                validate_predictions_h5(os.fspath(out_path))
                return PersistentTensorDict(filename=out_path, mode="r")
            if overwrite_mode == "error":
                raise FileExistsError(
                    f"postprocess: destination already exists: {out_path!r}"
                )

        if (src.endswith(".h5") or src.endswith(".hdf5")) and os.path.isfile(
            src
        ):
            if output_mode == "file":
                if out_path is None:
                    return PersistentTensorDict(filename=src, mode="r")
                return copy_predictions_h5_atomic(
                    os.fspath(src),
                    os.fspath(out_path),
                    overwrite=str(overwrite_mode or "replace"),
                )
            return load_predictions_h5(os.fspath(src))

        if not os.path.isdir(src):
            raise FileNotFoundError(
                f"source must be a directory or .h5 file: {src!r}"
            )
        memmap_dir = os.path.join(src, "memmap")
        chunks_dir = os.path.join(src, "pred_chunks")
        pred_path = os.path.join(src, "pred.mmt")
        if not os.path.isdir(memmap_dir):
            raise FileNotFoundError(f"missing memmap dir: {memmap_dir!r}")
        count = None
        out_shape = None
        if os.path.isdir(chunks_dir):
            man_path = os.path.join(chunks_dir, "manifest.json")
            if os.path.isfile(man_path):
                try:
                    man = read_json(man_path)
                    if isinstance(man, dict):
                        count = man.get("count", None)
                        out_shape = man.get("out_shape", None)
                except Exception:
                    pass
        if not os.path.isfile(pred_path):
            if not os.path.isdir(chunks_dir):
                raise FileNotFoundError(
                    f"missing pred_chunks dir: {chunks_dir!r}"
                )
            if count is None or out_shape is None:
                raise FileNotFoundError(
                    f"missing/invalid manifest.json in pred_chunks: {chunks_dir!r}"
                )
            count = int(count)
            out_shape_t = tuple(int(x) for x in (out_shape or ()))
            if (
                count <= 0
                or (not out_shape_t)
                or any(int(d) <= 0 for d in out_shape_t)
            ):
                raise ValueError(
                    f"postprocess: invalid manifest metadata: count={count!r}, out_shape={out_shape!r}"
                )
            store_float = _get_prediction_dtype(chunks_dir)
            concat_memory_mapped_tensor(
                os.fspath(chunks_dir),
                os.fspath(pred_path),
                count=count,
                out_shape=out_shape_t,
                store_float=store_float,
            )
        X_mmt = load_memmap_features(os.fspath(memmap_dir))
        Y_mmt = open_memory_mapped_tensor(os.fspath(pred_path))
        if Y_mmt is None:
            raise RuntimeError("postprocess: failed to open pred.mmt")
        if count is None:
            try:
                count = int(X_mmt.shape[0])
            except Exception:
                count = None
        if count is None:
            raise RuntimeError("postprocess: failed to infer count")

        if out_path is not None:
            if os.path.exists(out_path):
                if overwrite_mode == "resume" and os.path.isfile(out_path):
                    validate_predictions_h5(
                        os.fspath(out_path),
                        out_shape=tuple(int(d) for d in Y_mmt.shape[1:]),
                        in_dim=(
                            int(X_mmt.shape[1])
                            if hasattr(X_mmt, "shape") and len(X_mmt.shape) > 1
                            else None
                        ),
                    )
                    return PersistentTensorDict(filename=out_path, mode="r")
                if overwrite_mode == "error":
                    raise FileExistsError(
                        f"postprocess: destination already exists: {out_path!r}"
                    )
            out_td = write_predictions_h5_atomic(
                os.fspath(out_path),
                memmap_dir=os.fspath(memmap_dir),
                pred_path=os.fspath(pred_path),
                chunk_size=8192,
                overwrite=str(overwrite_mode or "replace"),
            )
            if not os.path.isfile(out_path):
                raise RuntimeError(
                    f"postprocess: persistent output missing after write: {out_path!r}"
                )
            validate_predictions_h5(
                os.fspath(out_path),
                out_shape=tuple(int(d) for d in Y_mmt.shape[1:]),
                in_dim=(
                    int(X_mmt.shape[1])
                    if hasattr(X_mmt, "shape") and len(X_mmt.shape) > 1
                    else None
                ),
            )
            remove_prediction_artifacts(
                memmap_dir=os.fspath(memmap_dir),
                pred_path=os.fspath(pred_path),
            )
            return out_td

        X_t = copy_mmt_to_cpu_tensor(X_mmt, count=int(count), chunk_size=8192)
        Y_t = copy_mmt_to_cpu_tensor(Y_mmt, count=int(count), chunk_size=8192)
        td_out = TensorDict({"X": X_t, "Y": Y_t}, batch_size=[int(count)])
        remove_prediction_artifacts(
            memmap_dir=os.fspath(memmap_dir),
            pred_path=os.fspath(pred_path),
        )
        return td_out


class _BatchSliceGetter:
    def __init__(
        self: Self, raw_X: Any, raw_Y: Any, *args: Any, features_only: bool
    ) -> None:
        self.raw_X, self.raw_Y, self.features_only = (
            raw_X,
            raw_Y,
            bool(features_only),
        )

    def __call__(self: Self, s: int, e: int) -> Mapping[str, Any]:
        out = {
            "features": _preload_slice_any(self.raw_X, s, e, name="features")
        }
        if self.raw_Y is not None and not self.features_only:
            out["labels"] = _preload_slice_any(self.raw_Y, s, e, name="labels")
        return out


class _BatchIndexGetter:
    def __init__(
        self: Self, raw_X: Any, raw_Y: Any, *args: Any, features_only: bool
    ) -> None:
        self.raw_X, self.raw_Y, self.features_only = (
            raw_X,
            raw_Y,
            bool(features_only),
        )

    def __call__(self: Self, idx: torch.Tensor) -> Mapping[str, Any]:
        idx_cpu = _idx_to_cpu_int64(idx)
        idx_np = idx_cpu.numpy() if hasattr(idx_cpu, "numpy") else None

        out: Dict[str, Any] = {
            "features": _preload_gather_any_preconverted(
                self.raw_X, idx_cpu, idx_np, name="features"
            )
        }
        if self.raw_Y is not None and not self.features_only:
            out["labels"] = _preload_gather_any_preconverted(
                self.raw_Y, idx_cpu, idx_np, name="labels"
            )
        return out


class _KeyView(collections.abc.Mapping):
    __slots__ = ("_data", "_keys")

    def __init__(
        self: Self, data: Mapping[Any, Any], keys: Sequence[Any]
    ) -> None:
        self._data = data
        self._keys = keys

    def __len__(self: Self) -> int:
        return int(len(self._keys))

    def __iter__(self: Self) -> Iterator[Any]:
        return iter(self._keys)

    def __getitem__(self: Self, k: Any) -> Any:
        return self._data[k]


class _KeyCursor:
    __slots__ = ("_data", "_keys_source", "_it", "_pos")

    def __init__(
        self: Self,
        data: Mapping[Any, Any],
        keys: Optional[Sequence[Any]] = None,
    ) -> None:
        self._data = data
        self._keys_source = data.keys() if keys is None else keys
        self._it = iter(self._keys_source)
        self._pos = 0

    def _reset(self: Self) -> None:
        self._it = iter(self._keys_source)
        self._pos = 0

    def __call__(self: Self, s: int, e: int) -> _KeyView:
        s_i = int(s)
        e_i = int(e)
        if s_i < 0 or e_i < s_i:
            raise ValueError(f"invalid batch slice: s={s_i}, e={e_i}")
        if s_i == 0 and self._pos != 0:
            self._reset()
        if s_i != self._pos:
            raise RuntimeError(
                "key_index_mapping_getters: non-sequential access requested "
                f"(expected s={self._pos}, got s={s_i}). "
                "Disable writer-side shuffle; let the sampler handle shuffling."
            )
        need = e_i - s_i
        if need <= 0:
            return _KeyView(self._data, ())
        batch_keys: list[Any] = []
        for _ in range(int(need)):
            try:
                k = next(self._it)
            except StopIteration:
                break
            batch_keys.append(k)
        self._pos += int(len(batch_keys))
        return _KeyView(self._data, batch_keys)


class MappingSlicer:
    __slots__ = ("const_items", "slice_items")

    def __init__(
        self: Self,
        const_items: Mapping[Any, Any],
        slice_items: Tuple[Any, ...],
    ) -> None:
        self.const_items = dict(const_items)
        self.slice_items = tuple(slice_items)

    def __call__(self: Self, s: object, e: object) -> Mapping[Any, Any]:
        batch = dict(self.const_items)
        for k, v in self.slice_items:
            try:
                batch[k] = v[s:e]
            except Exception:
                batch[k] = v
        return batch


class TensorDictSlicer:
    __slots__ = ("td",)

    def __init__(self: Self, td: TensorDictBase) -> None:
        self.td = td

    def __call__(self: Self, s: object, e: object) -> TensorDictBase:
        return self.td[s:e]


class Unsharder:
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
        self: Self,
        *args: Any,
        chunk_dir: str,
        rank: int,
        use_mmt_pred_parts: bool,
        cache: TensorSpooler | None,
        pred_pool: TensorPagePool | None,
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
        self.pred_handle: TensorPagePool.Token | None = None
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

    def flush(self: Self) -> None:
        if self.use_buffer:
            if (
                self.buf_fill <= 0
                or self.rows_buf is None
                or self.pred_buf is None
            ):
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
            rows = torch.cat(self.pending_rows, dim=0).to(
                dtype=torch.int64, copy=False
            )
            if not bool(rows.is_contiguous()):
                rows = rows.contiguous()
            preds = torch.cat(self.pending_preds, dim=0)
            if not bool(preds.is_contiguous()):
                preds = preds.contiguous()
            local_handle = None
            need_wait_evt = False

        rows_path = os.path.join(
            self.chunk_dir,
            f"part-r{self.rank:05d}-c{self.chunk_idx:06d}-rows.pt",
        )
        pred_ext = (
            "mmt" if self.use_buffer and self.use_mmt_pred_parts else "pt"
        )
        pred_path = os.path.join(
            self.chunk_dir,
            f"part-r{self.rank:05d}-c{self.chunk_idx:06d}-pred.{pred_ext}",
        )

        if self.cache is not None:
            self.cache.submit(rows, path=rows_path)
        else:
            atomic_torch_save(rows_path, rows)

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
            self.cache.submit(
                preds,
                path=pred_path,
                wait_event=wait_evt,
                release_cb=release_cb,
            )
        else:
            preds_cpu = preds.detach()
            if (
                getattr(preds_cpu, "device", None) is not None
                and preds_cpu.device.type != "cpu"
            ):
                preds_cpu = preds_cpu.to(device="cpu")
            atomic_torch_save(pred_path, preds_cpu)
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

    def append(self: Self, rows_cpu: object, preds: object) -> None:
        if not isinstance(preds, torch.Tensor) or preds.ndim < 1:
            return
        b = int(preds.shape[0])
        if b <= 0:
            return
        if not isinstance(rows_cpu, torch.Tensor):
            rows_cpu = torch.as_tensor(rows_cpu, dtype=torch.int64)
        rows_cpu = rows_cpu.reshape(-1).to(
            dtype=torch.int64, device="cpu", copy=False
        )
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
                    (self.target_rows, *tail),
                    dtype=preds.dtype,
                    return_handle=True,
                )
            else:
                self.pred_buf = torch.empty(
                    (self.target_rows, *tail), dtype=preds.dtype
                )
                self.pred_handle = None
            self.pred_buf_is_pinned = False
            with contextlib.suppress(Exception):
                is_pinned = getattr(self.pred_buf, "is_pinned", None)
                if callable(is_pinned):
                    self.pred_buf_is_pinned = bool(is_pinned())
        elif (
            self.pred_buf.dtype != preds.dtype
            or tuple((int(x) for x in self.pred_buf.shape[1:])) != tail
        ):
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
                        (self.target_rows, *tail),
                        dtype=preds.dtype,
                        return_handle=True,
                    )
                else:
                    self.pred_buf = torch.empty(
                        (self.target_rows, *tail), dtype=preds.dtype
                    )
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
            self.rows_buf[self.buf_fill : self.buf_fill + n].copy_(
                rows_cpu[start : start + n]
            )
            non_blocking = (
                bool(self.pred_buf_is_pinned) and preds.device.type != "cpu"
            )
            self.pred_buf[self.buf_fill : self.buf_fill + n].copy_(
                preds[start : start + n], non_blocking=non_blocking
            )
            if non_blocking:
                self.buf_needs_wait_evt = True
            self.buf_fill += n
            start += n
            if self.buf_fill >= int(self.target_rows):
                self.flush()


class Collator:
    labels_dtype: Optional[torch.dtype] = None
    sanitize: bool = False
    flatten_features: bool = False

    def __init__(
        self: Self,
        *args: Any,
        labels_dtype: Optional[torch.dtype] = None,
        sanitize: bool = False,
        flatten_features: bool = False,
    ) -> None:
        self.labels_dtype = labels_dtype
        self.sanitize = bool(sanitize)
        self.flatten_features = bool(flatten_features)

    def __call__(self: Self, batch: Any) -> Any:
        labels_dtype = self.labels_dtype
        sanitize = bool(self.sanitize)
        flatten = bool(self.flatten_features)
        if isinstance(batch, (list, tuple)):
            if not batch:
                return batch

            def _standardize_batch(
                items: Sequence[object],
            ) -> TensorDictBase | dict[str, object]:
                if all(isinstance(elem, TensorDictBase) for elem in items):
                    return torch.stack(items, dim=0)

                def _stack_key(
                    key: str, default: object | None = None
                ) -> torch.Tensor | list[object] | None:
                    vals = [
                        (
                            x.get(key, default)
                            if isinstance(x, Mapping)
                            else default
                        )
                        for x in items
                    ]
                    if all(v is None for v in vals):
                        return None
                    return _to_safe_tensor(
                        torch.stack(
                            [_to_safe_tensor(v) for v in vals if v is not None]
                        )
                        if any(isinstance(v, torch.Tensor) for v in vals)
                        else vals
                    )

                X = _stack_key("X") or _stack_key("x")
                Y = _stack_key("Y") or _stack_key("y")
                rows = _stack_key("row_ids")

                data = {"X": X}
                if Y is not None:
                    data["Y"] = Y
                if rows is not None:
                    data["row_ids"] = rows
                return (
                    TensorDict(data, batch_size=_td_batch_size_from_X(X))
                    if TensorDict
                    else data
                )

            stacked = _standardize_batch(batch)
            if isinstance(stacked, TensorDictBase):
                with contextlib.suppress(Exception):
                    canonicalize_keys_(stacked, allow_missing_labels=True)

            try:
                conv = preprocess(
                    stacked,
                    flatten_features=flatten,
                    labels_dtype=labels_dtype,
                    sanitize=sanitize,
                )
            except Exception:
                return stacked

            if isinstance(conv, Mapping) and isinstance(
                stacked, TensorDictBase
            ):
                for k in ["X", "Y", "row_ids"]:
                    if k in conv and conv[k] is not None:
                        stacked.set(k, conv[k])
            return stacked
        if isinstance(batch, Mapping):
            try:
                conv = preprocess(
                    batch,
                    flatten_features=flatten,
                    labels_dtype=labels_dtype,
                    sanitize=sanitize,
                )
            except Exception:
                conv = batch
            X = conv.get("X", None) if isinstance(conv, Mapping) else None
            Y = conv.get("Y", None) if isinstance(conv, Mapping) else None
            if X is None:
                with contextlib.suppress(Exception):
                    X, Y = get_row(batch, labels_required=False)
            if X is None:
                X = batch.get("X")
            if Y is None:
                Y = batch.get("Y")
            row_ids = (
                conv.get("row_ids")
                if isinstance(conv, Mapping)
                else batch.get("row_ids")
            )
            data: dict[str, Any] = {"X": X}
            if isinstance(Y, torch.Tensor):
                data["Y"] = Y
            if row_ids is not None:
                data["row_ids"] = row_ids
            return (
                TensorDict(data, batch_size=_td_batch_size_from_X(X))
                if TensorDict
                else data
            )
        return batch
