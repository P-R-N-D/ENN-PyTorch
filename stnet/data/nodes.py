# -*- coding: utf-8 -*-
from __future__ import annotations

import collections.abc
import contextlib
import logging
import math
import multiprocessing
import os
import queue
import shutil
import tempfile
import threading
import time
import traceback
from contextlib import suppress
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
)

import h5py
import numpy
import torch
import torch.utils.data
import torchdata.nodes
from tensordict import MemoryMappedTensor, PersistentTensorDict, TensorDict
from torchdata.nodes import (
    BaseNode,
    MultiNodeWeightedSampler,
    ParallelMapper,
    SamplerWrapper,
)

from ..core.casting import (
    dtype_from_name,
    env_bool,
    env_first_int,
    env_str,
    parse_torch_dtype,
)
from ..core.graph import inference_mode
from ..core.concurrency import Buffer, Pool, ProducerError, close, get_affinity, new_thread
from ..core.system import (
    CPU,
    Memory,
    WorkerPolicy,
    accelerator_stream,
    accelerator_type,
    current_accelerator_stream,
    get_accelerator_index,
    get_num_accelerators,
    is_accelerator_available,
    is_stream_supported,
    new_accelerator_event,
    new_accelerator_stream,
)
from . import schemas
from .pipeline import Dataset
from .schemas import (
    _FEATURE_KEY_ALIASES,
    _LABEL_KEY_ALIASES,
    default_underflow_action,
    normalize_underflow_action,
)


_LOGGER = logging.getLogger(__name__)

SourceType = Literal["memmap"]


def _strictest_underflow_action(v1: Optional[str], v2: Optional[str]) -> Optional[str]:
    if v1 is None or v2 is None:
        return v1 or v2
    order = {"allow": 0, "warn": 1, "forbid": 2}
    return v1 if order.get(str(v1), 0) >= order.get(str(v2), 0) else v2


def _meta_has_scale(meta: Any) -> bool:
    if not isinstance(meta, Mapping):
        return False
    keys = ("scale_max_abs", "scale_min_value", "scale_max_value", "scale_min_positive")
    if meta.get("has_scale") or any(meta.get(k) is not None for k in keys):
        return True
    return isinstance((ss := meta.get("scale_stats")), Mapping) and (
        ss.get("has_scale") or any(ss.get(k) is not None for k in keys)
    )


def _remove_safe(path: str) -> None:
    with suppress(FileNotFoundError):
        os.remove(path)


def _flatten_args(items: Sequence[Any]) -> Iterator[Any]:
    for item in items:
        if isinstance(item, Mapping):
            yield from _flatten_args(list(item.values()))
        elif isinstance(item, (list, tuple, set)):
            yield from _flatten_args(list(item))
        else:
            yield item


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
    payload = schemas.read_json(mn_path)
    if isinstance(payload, dict):
        return {
            str(k): {"kind": "memmap", "path": os.path.join(root, str(v))}
            for k, v in payload.items()
        }, True
    if isinstance(payload, list):
        return [{"kind": "memmap", "path": os.path.join(root, str(v))} for v in payload], True
    return spec, False


def _is_accelerator_available() -> bool:
    return any(is_accelerator_available(a) for a in ("cuda", "xpu", "mps"))


def _device_guard_ok(device: torch.device, guard_bytes: int) -> bool:
    if guard_bytes <= 0:
        return True
    try:
        return (free_b := Memory.mem_get_info(device)[0]) is None or free_b >= guard_bytes
    except Exception:
        return True


def _host_guard_ok(guard_bytes: int) -> bool:
    return guard_bytes <= 0 or getattr(Memory, "available", lambda: guard_bytes)() >= guard_bytes


@lru_cache(maxsize=1)
def _accel_event_poll_params() -> tuple[float, float, float]:
    start_us = int(
        env_first_int(
            ("STNET_ACCEL_EVENT_POLL_START_US", "STNET_CUDA_EVENT_POLL_START_US"),
            default=500,
        )
        or 500
    )
    max_ms = int(
        env_first_int(
            ("STNET_ACCEL_EVENT_POLL_MAX_MS", "STNET_CUDA_EVENT_POLL_MAX_MS"),
            default=50,
        )
        or 50
    )
    stop_min_ms = int(
        env_first_int(
            ("STNET_ACCEL_EVENT_POLL_STOP_MIN_MS", "STNET_CUDA_EVENT_POLL_STOP_MIN_MS"),
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
    stop_min_sleep_s = stop_min_sleep_s if stop_min_sleep_s is not None else d_stop_min
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


def _preload_len0(obj: Any) -> int:
    return (
        (obj.shape[0] if getattr(obj, "ndim", 0) > 0 else 1)
        if isinstance(obj, torch.Tensor)
        else len(obj)
    )


def _preload_slice_any(obj: Any, s: int, e: int, *args, name: str) -> Any:
    if obj is None:
        return None
    try:
        return obj[s:e]
    except Exception:
        return [obj[i] for i in range(s, e)]


def _preload_gather_any_preconverted(
    obj: Any,
    idx_cpu: torch.Tensor,
    idx_np: Any,
    *args,
    name: str,
) -> Any:
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
    if isinstance(device, Sequence) and not isinstance(device, (str, bytes, bytearray)):
        devs: list[torch.device] = []
        for d in device:
            devs.append(d if isinstance(d, torch.device) else torch.device(str(d)))
        return devs if devs else torch.device("cpu")
    return torch.device(device)


def _primary_device(device_spec: torch.device | list[torch.device]) -> torch.device:
    return device_spec[0] if isinstance(device_spec, list) and device_spec else device_spec


class _RowSlicer:
    def __init__(self, raw_X: Any, raw_Y: Any, *args: Any, features_only: bool) -> None:
        self.raw_X, self.raw_Y, self.features_only = raw_X, raw_Y, bool(features_only)

    def __call__(self, s: int, e: int) -> Mapping[str, Any]:
        out = {"features": _preload_slice_any(self.raw_X, s, e, name="features")}
        if self.raw_Y is not None and not self.features_only:
            out["labels"] = _preload_slice_any(self.raw_Y, s, e, name="labels")
        return out


class _RowIndexer:
    def __init__(self, raw_X: Any, raw_Y: Any, *args: Any, features_only: bool) -> None:
        self.raw_X, self.raw_Y, self.features_only = raw_X, raw_Y, bool(features_only)

    def __call__(self, idx: torch.Tensor) -> Mapping[str, Any]:
        idx_cpu = Storage._idx_to_cpu_int64(idx)
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


class _ColumnView(collections.abc.Mapping):
    __slots__ = ("_data", "_keys")

    def __init__(self, data: Mapping[Any, Any], keys: Sequence[Any]) -> None:
        self._data = data
        self._keys = keys

    def __len__(self) -> int:
        return int(len(self._keys))

    def __iter__(self):
        return iter(self._keys)

    def __getitem__(self, k: Any) -> Any:
        return self._data[k]


class _ColumnCursor:
    __slots__ = ("_data", "_keys_source", "_it", "_pos")

    def __init__(self, data: Mapping[Any, Any], keys: Optional[Sequence[Any]] = None) -> None:
        self._data = data
        self._keys_source = data.keys() if keys is None else keys
        self._it = iter(self._keys_source)
        self._pos = 0

    def _reset(self) -> None:
        self._it = iter(self._keys_source)
        self._pos = 0

    def __call__(self, s: int, e: int) -> _ColumnView:
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
            return _ColumnView(self._data, ())
        batch_keys: list[Any] = []
        for _ in range(int(need)):
            try:
                k = next(self._it)
            except StopIteration:
                break
            batch_keys.append(k)
        self._pos += int(len(batch_keys))
        return _ColumnView(self._data, batch_keys)


class Storage:
    get_meta_path = staticmethod(schemas.get_meta_path)
    write_json = staticmethod(schemas.write_json)
    save_temp = staticmethod(schemas.save_temp)

    @staticmethod
    def _resolve_memmap_store_float(*args: Any, negotiable: bool) -> torch.dtype:
        req = str(env_str("STNET_MEMMAP_FLOAT_DTYPE") or "").strip()
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
        return torch.float32 if (bool(negotiable) and req_dtype != torch.float64) else torch.float64

    @staticmethod
    def _to_cpu_contig(t: torch.Tensor) -> torch.Tensor:
        t = t.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        if not t.is_contiguous():
            t = t.contiguous()
        return t

    @staticmethod
    def _flat2d_cpu_contig(t: torch.Tensor, n: int) -> torch.Tensor:
        t_cpu = Storage._to_cpu_contig(t)
        if t_cpu.ndim == 0:
            t_cpu = t_cpu.reshape(1)
        return t_cpu.reshape(int(n), -1)

    @staticmethod
    def _batch_n(x: torch.Tensor) -> int:
        xd = int(getattr(x, "ndim", 0) or 0)
        return int(x.shape[0]) if xd > 0 else 1

    @staticmethod
    def _idx_to_cpu_int64(idx: Any) -> torch.Tensor:
        if not isinstance(idx, torch.Tensor):
            idx = torch.as_tensor(idx)
        if idx.device.type != "cpu":
            idx = idx.detach().cpu()
        if idx.dtype not in (torch.int64, torch.int32):
            idx = idx.to(dtype=torch.int64, copy=False)
        idx = idx.reshape(-1)
        return idx.to(dtype=torch.int64, copy=False)

    @staticmethod
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

    @staticmethod
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
        is_contig, start, end = Storage._validate_row_contiguity(rows_t)
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

    @staticmethod
    def _h5_write_rows(
        dset_Y: object,
        rows_t: torch.Tensor,
        preds_np: object,
        *args: Any,
        count: int,
    ) -> None:
        is_contig, start, end = Storage._validate_row_contiguity(rows_t)
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

    @staticmethod
    def _torch_load_cpu(path: str) -> object:
        return torch.load(path, map_location="cpu", weights_only=True)

    @staticmethod
    def _load_row(rows_file: str) -> torch.Tensor:
        rows_t = Storage._torch_load_cpu(os.fspath(rows_file))
        if not isinstance(rows_t, torch.Tensor):
            rows_t = torch.as_tensor(rows_t, device="cpu")
        rows_t = rows_t.reshape(-1).to(dtype=torch.int64, device="cpu", copy=False)
        if not bool(rows_t.is_contiguous()):
            rows_t = rows_t.contiguous()
        return rows_t

    @staticmethod
    def _load_prediction(pred_file: str, *args: Any, dtype: torch.dtype) -> torch.Tensor:
        _ = args
        pf = os.fspath(pred_file)
        if pf.endswith(".mmt"):
            preds_t = Storage.open_memory_mapped_tensor(pf)
            if preds_t is None:
                raise FileNotFoundError(f"missing prediction memmap or meta: {pf!r}")
        else:
            preds_t = Storage._torch_load_cpu(pf)
        if not isinstance(preds_t, torch.Tensor):
            preds_t = torch.as_tensor(preds_t, device="cpu")
        if preds_t.device.type != "cpu":
            preds_t = preds_t.to(device="cpu")
        return preds_t.to(dtype=dtype, copy=False)

    @staticmethod
    def _to_numpy_dtype(dtype: torch.dtype):
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

    @staticmethod
    def column_cursor(
        data: Mapping[Any, Any],
        *args: Any,
        keys: Optional[Sequence[Any]] = None,
    ) -> Tuple[int, Callable[[int, int], _ColumnView]]:
        count = int(len(data) if keys is None else len(keys))
        if count <= 0:
            raise ValueError("Empty mapping: no keys")
        return count, _ColumnCursor(data, keys)

    @staticmethod
    def is_feature_label_batch_mapping(obj: Any) -> bool:
        if not isinstance(obj, Mapping) or not obj:
            return False
        for k in obj.keys():
            if not isinstance(k, str):
                continue
            ck = k.casefold()
            if ck in _FEATURE_KEY_ALIASES or ck in _LABEL_KEY_ALIASES:
                return True
        return False

    @staticmethod
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
        out_dir = os.fspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        count_i = int(count)
        if count_i <= 0:
            raise ValueError("count must be > 0")
        env_chunk = env_first_int(("STNET_MEMMAP_CHUNK_SIZE", "STNET_MEMMAP_CHUNK"), 0)
        if int(env_chunk) > 0:
            chunk_size = int(env_chunk)
        req_chunk = int(chunk_size or 0)
        auto_chunk = req_chunk <= 0
        chunk_first = max(1, min(count_i, req_chunk if req_chunk > 0 else min(count_i, 256)))
        allow_missing = bool(allow_missing_labels) or bool(features_only)
        default_lshape = tuple(default_label_shape) if default_label_shape is not None else (1,)
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
        for s in range(0, count_i, int(chunk_first)):
            e = min(count_i, s + int(chunk_first))
            batch = get_batch(int(s), int(e))
            fx, lb, _, _ = ds.preprocess(batch, return_keys=False)
            n = Storage._batch_n(fx)
            if n <= 0:
                continue
            expected = int(e) - int(s)
            if n != expected:
                raise RuntimeError(
                    f"Pass1 batch size mismatch for out_dir={out_dir!r}: expected {expected}, got {n} (s={s}, e={e})."
                )
            fx_flat = Storage._flat2d_cpu_contig(fx, n)
            cur_in_dim = int(fx_flat.shape[1])
            if in_dim is None:
                in_dim = cur_in_dim
            elif cur_in_dim != int(in_dim):
                raise RuntimeError(f"feature dim mismatch: expected {in_dim}, got {cur_in_dim}")
            if lb is None:
                if not allow_missing:
                    raise RuntimeError("memmap writer requires labels (got None)")
                cur_label_shape = tuple(default_lshape)
                lb_flat = None
            else:
                cur_label_shape = tuple(lb.shape[1:])
                lb_flat = Storage._flat2d_cpu_contig(lb, n)
            if label_shape is None:
                label_shape = cur_label_shape
            elif tuple(label_shape) != tuple(cur_label_shape):
                raise RuntimeError(
                    f"label shape mismatch: expected {label_shape}, got {cur_label_shape}"
                )
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
        if in_dim is None or label_shape is None:
            raise RuntimeError("Failed to infer in_dim/label_shape from data")
        negotiable = Dataset.is_fp32_castable(
            stats,
            underflow_action=underflow_action,
            safety_margin=1.0,
        )
        store_float = Storage._resolve_memmap_store_float(negotiable=bool(negotiable))
        if auto_chunk:
            elem_size = int(torch.empty((), dtype=store_float).element_size())
            label_numel = 0 if bool(features_only) else int(numpy.prod(label_shape))
            row_bytes = max(1, (int(in_dim) + int(label_numel)) * int(elem_size))
            target_bytes = env_first_int(("STNET_MEMMAP_CHUNK_BYTES",), 0)
            if int(target_bytes) <= 0:
                target_mb = env_first_int(("STNET_MEMMAP_CHUNK_MB",), 64)
                target_bytes = int(target_mb) * 1024 * 1024
            try:
                avail = int(Memory.available() or 0)
                if avail > 0:
                    target_bytes = int(min(int(target_bytes), max(8 * 1024 * 1024, avail // 16)))
            except Exception:
                pass
            chunk_second = int(max(1, min(count_i, max(32, int(target_bytes) // int(row_bytes)))))
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
                ("STNET_MEMMAP_RANDPERM_MAX_ELEMS", "STNET_MEMMAP_SHUFFLE_MAX_ELEMS"),
                5_000_000,
            )
            use_full = (max_elems is not None) and (count_i <= int(max_elems))
            seed_i = None if seed_value is None else (int(seed_value) & 0x7FFFFFFFFFFFFFFF)
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
                    seed_i = int(torch.randint(0, 2**63 - 1, (1,), dtype=torch.int64).item())
                shuffle_seed = seed_i
                k = max(1, int((count_i - 1)).bit_length())
                if (k % 2) == 1:
                    k += 1
                half = k // 2
                mask = (1 << half) - 1
                domain_mask = (1 << k) - 1 if k < 64 else 0xFFFFFFFFFFFFFFFF
                seed_u = torch.tensor(seed_i & 0xFFFFFFFFFFFFFFFF, dtype=torch.uint64)
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
                    return ((p * int(a0) + int(b0)) % int(count_i)).to(dtype=torch.int64)

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
                    pos = torch.arange(int(s), int(e), device="cpu", dtype=torch.int64)
                    return _permute(pos)

                shuffle_indexer = _idx
                shuffle_impl = "prp"
        compute_scaler_stats = bool(write_labels) and (not bool(allow_missing_labels))
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
            x_sum = torch.zeros((int(in_dim),), dtype=torch.float64, device=torch.device("cpu"))
            x_sum_sq = torch.zeros((int(in_dim),), dtype=torch.float64, device=torch.device("cpu"))
            x_tmp = torch.empty_like(x_sum)
            x2_tmp = torch.empty_like(x_sum)
            out_dim = int(numpy.prod(label_shape))
            y_sum = torch.zeros((int(out_dim),), dtype=torch.float64, device=torch.device("cpu"))
            y_sum_sq = torch.zeros((int(out_dim),), dtype=torch.float64, device=torch.device("cpu"))
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
            n = Storage._batch_n(fx)
            if n <= 0:
                continue
            expected = int(e) - int(s)
            if n != expected:
                raise RuntimeError(
                    f"Pass2 batch size mismatch for out_dir={out_dir!r}: expected {expected}, got {n} (s={s}, e={e})."
                )
            fx_flat = Storage._flat2d_cpu_contig(fx, n)
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
                        raise RuntimeError("memmap writer requires labels (got None)")
                    labels_mmt[int(s) : int(s) + int(n)].zero_()
                else:
                    if tuple(lb.shape[1:]) != tuple(label_shape):
                        raise RuntimeError(
                            f"label shape mismatch: expected {label_shape}, got {tuple(lb.shape[1:])}"
                        )
                    lb_cpu = Storage._to_cpu_contig(lb)
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
                            y2_tmp.copy_(torch.einsum("ni,ni->i", lb_stats, lb_stats))
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
                Storage.save_temp(os.path.join(out_dir, scaler_stats_path), payload)
            except Exception:
                scaler_stats_path = None
        meta_json: Dict[str, Any] = {
            "N": int(count_i),
            "feature_dim": int(in_dim),
            "features_path": "features.mmt",
            "labels_path": ("labels.mmt" if write_labels else None),
            "label_shape": list(label_shape),
            "features_dtype": str(store_float).replace("torch.", ""),
            "labels_dtype": (str(store_float).replace("torch.", "") if write_labels else None),
            "fractions": [float(1.0 - float(val_frac)), float(val_frac)],
            "shuffled": bool(shuffle),
            "shuffle_seed": int(shuffle_seed) if shuffle_seed is not None else None,
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
        Storage.write_json(os.path.join(out_dir, "meta.json"), meta_json, indent=2)
        return int(in_dim), tuple(label_shape)

    @staticmethod
    def preload_memmap(
        data: Mapping[str, Any],
        *args: Any,
        memmap_dir: str,
        val_frac: float = 0.0,
        shuffle: bool = False,
        seed: Optional[int] = None,
        underflow_action: Optional[str] = None,
        chunk_size: int = 4096,
        allow_missing_labels: bool = False,
        features_only: bool = False,
        default_label_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        if not isinstance(data, Mapping):
            raise TypeError("preload_memmap expects a Mapping with at least 'features'")
        if "features" not in data:
            raise ValueError("preload_memmap expects 'features'")
        raw_X = data["features"]
        raw_Y = data.get("labels")
        count = _preload_len0(raw_X)
        if count <= 0:
            raise ValueError("cannot create memmap with zero samples")
        if not bool(features_only):
            if raw_Y is None:
                if not bool(allow_missing_labels):
                    raise ValueError(
                        "preload_memmap expects 'labels' unless allow_missing_labels=True"
                    )
            else:
                if _preload_len0(raw_Y) != int(count):
                    raise ValueError("features and labels must have the same length")
        ua = normalize_underflow_action(underflow_action, default=default_underflow_action())
        ds = Dataset.for_device("cpu", feature_dtype=torch.float64, label_float_dtype=torch.float64)
        ds.underflow_action = ua
        get_batch = _RowSlicer(raw_X, raw_Y, features_only=bool(features_only))
        get_by_indices = (
            _RowIndexer(raw_X, raw_Y, features_only=bool(features_only)) if bool(shuffle) else None
        )
        Storage.stream_memmap(
            ds=ds,
            out_dir=os.fspath(memmap_dir),
            count=int(count),
            get_batch=get_batch,
            get_by_indices=get_by_indices,
            val_frac=float(val_frac),
            seed_value=int(seed) if seed is not None else None,
            underflow_action=str(ua),
            shuffle=bool(shuffle),
            allow_missing_labels=bool(allow_missing_labels),
            features_only=bool(features_only),
            default_label_shape=(
                tuple(default_label_shape) if default_label_shape is not None else None
            ),
            chunk_size=int(chunk_size),
        )
        return None

    @staticmethod
    def iter_source_path(obj: Any):
        if obj is None:
            return
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, dict):
            if obj.get("kind") == "memmap" and isinstance(obj.get("path"), str):
                yield obj["path"]
            else:
                for v in obj.values():
                    yield from Storage.iter_source_path(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from Storage.iter_source_path(v)

    @staticmethod
    def from_meta(memmap_dir: str) -> Dict[str, Any]:
        meta_path = os.path.join(os.fspath(memmap_dir), "meta.json")
        raw = schemas.read_json(meta_path)
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def merge_meta_info(metas: Any) -> Dict[str, Any]:
        def _merge_dicts(items: list[dict]) -> Dict[str, Any]:
            if not items:
                return {}
            base = dict(items[0])

            def _upd_min(k, v):
                base[k] = v if base.get(k) is None else min(float(base[k]), float(v))

            def _upd_max(k, v):
                base[k] = v if base.get(k) is None else max(float(base[k]), float(v))

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
                    base["scale_is_integral"] = bool(v) and base.get("scale_is_integral", True)
                if (v := m.get("is_negotiable")) is not None:
                    base["is_negotiable"] = bool(v) and base.get("is_negotiable", True)
                base["underflow_action"] = _strictest_underflow_action(
                    base.get("underflow_action"), m.get("underflow_action")
                )
            return base

        if metas is None:
            return {}
        if isinstance(metas, list) and metas and isinstance(metas[0], dict):
            return _merge_dicts(metas)
        collected: list[dict] = []
        for path in Storage.iter_source_path(metas):
            try:
                meta = Storage.from_meta(path)
            except Exception:
                meta = None
            if isinstance(meta, dict):
                collected.append(meta)
        return _merge_dicts(collected)

    @staticmethod
    def load_scaler_stats(sources: Any) -> Optional[Dict[str, Any]]:
        expanded = Storage.expand_source(sources)
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
        for path in Storage.iter_source_path(expanded):
            try:
                meta = Storage.from_meta(path)
            except Exception:
                return None
            rel = meta.get("scaler_stats_path")
            if not rel:
                return None
            stats_path = os.path.join(os.fspath(path), os.fspath(rel))
            if not os.path.isfile(stats_path):
                return None
            try:
                payload = torch.load(stats_path, map_location="cpu", weights_only=True)
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
                    if local_xmin.shape != x_min.shape or local_xmax.shape != x_max.shape:
                        return None
                    if local_ymin.shape != y_min.shape or local_ymax.shape != y_max.shape:
                        return None
                    torch.minimum(x_min, local_xmin, out=x_min)
                    torch.maximum(x_max, local_xmax, out=x_max)
                    torch.minimum(y_min, local_ymin, out=y_min)
                    torch.maximum(y_max, local_ymax, out=y_max)
                if have_qbounds:
                    assert y_q_low is not None and y_q_high is not None
                    if local_yq_low.shape != y_q_low.shape or local_yq_high.shape != y_q_high.shape:
                        return None
                    torch.minimum(y_q_low, local_yq_low, out=y_q_low)
                    torch.maximum(y_q_high, local_yq_high, out=y_q_high)
            total += c
        if total <= 0 or x_sum is None or x_sum_sq is None or y_sum is None or y_sum_sq is None:
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

    @staticmethod
    def expand_source(sources: Any) -> Any:
        expanded, ok = _expand_multinode_sources(sources)
        if ok:
            return expanded
        if isinstance(sources, (list, tuple)) and len(sources) == 1:
            expanded, ok = _expand_multinode_sources(sources[0])
            if ok:
                return expanded
        return sources

    @staticmethod
    def load_memmap_meta(memmap_dir: str) -> Mapping[str, Any]:
        meta_path = os.path.join(os.fspath(memmap_dir), "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"memmap meta.json not found: {meta_path}")
        meta = schemas.read_json(meta_path)
        if not isinstance(meta, dict):
            raise ValueError(f"memmap meta.json malformed: {meta_path}")
        return meta

    @staticmethod
    def open_memory_mapped_tensor(mmt_path: str) -> Optional[MemoryMappedTensor]:
        meta_path = Storage.get_meta_path(mmt_path)
        if not (os.path.isfile(mmt_path) and os.path.isfile(meta_path)):
            return None
        try:
            meta = schemas.read_json(meta_path)
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

    @staticmethod
    def load_memmap_features(memmap_dir: str) -> MemoryMappedTensor:
        meta = Storage.load_memmap_meta(memmap_dir)
        n = int(meta.get("N", 0) or 0)
        if n <= 0:
            raise ValueError(f"memmap meta.json under {memmap_dir} has non-positive N={n}")
        feat_rel = str(meta.get("features_path", "features.mmt"))
        feat_path = os.path.join(os.fspath(memmap_dir), feat_rel)
        fdim = int(meta.get("feature_dim", 0) or 0)
        if fdim <= 0:
            raise ValueError(
                f"memmap meta.json under {memmap_dir} has non-positive feature_dim={fdim}"
            )
        f_dtype = dtype_from_name(meta.get("features_dtype", "float64"), torch.float64)
        return MemoryMappedTensor.from_filename(
            feat_path, dtype=f_dtype, shape=torch.Size([n, fdim])
        )

    @staticmethod
    def copy_mmt_to_cpu_tensor(
        mmt: object,
        *args: Any,
        count: object | None = None,
        chunk_size: int = 8192,
    ) -> torch.Tensor:
        _ = args
        if mmt is None:
            raise ValueError("copy_mmt_to_cpu_tensor: mmt must not be None")
        n = int(count) if count is not None else int(getattr(mmt, "shape", [0])[0] or 0)
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

    @staticmethod
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

    @staticmethod
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
                raise ValueError(f"predictions file has non-positive row count: {n} ({p!r})")
            if out_shape is None:
                if len(y_shape) < 2:
                    raise ValueError(f"predictions file has invalid Y shape: Y={y_shape} ({p!r})")
            else:
                out_shape_t = tuple(int(d) for d in out_shape)
                if not out_shape_t or any(int(d) <= 0 for d in out_shape_t):
                    raise ValueError(f"validate_predictions_h5: invalid out_shape={out_shape!r}")
                if len(y_shape) != 1 + len(out_shape_t):
                    raise ValueError(
                        f"predictions file has unexpected Y rank: got {len(y_shape)}, expected {1 + len(out_shape_t)} ({p!r})"
                    )
                if tuple(int(d) for d in y_shape[1:]) != out_shape_t:
                    raise ValueError(
                        f"predictions file has unexpected Y shape: got {tuple(int(d) for d in y_shape[1:])}, expected {out_shape_t} ({p!r})"
                    )
        return int(n)

    @staticmethod
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
            full_shape, dtype=store_float, filename=os.fspath(out_path), existsok=True
        )
        manifest_path = os.path.join(os.fspath(chunks_dir), "manifest.json")
        manifest = schemas.read_json(manifest_path)
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
            rows_t = Storage._load_row(rows_file)
            preds_t = Storage._load_prediction(pred_file, dtype=store_float)
            Storage._index_copy_rows(y_out, rows_t, preds_t, count=int(count))
        pred_meta_path = Storage.get_meta_path(os.fspath(out_path))
        Storage.write_json(
            pred_meta_path,
            {
                "dtype": str(store_float).replace("torch.", ""),
                "shape": list(map(int, full_shape)),
            },
            indent=None,
        )
        return y_out

    @staticmethod
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
        manifest = schemas.read_json(manifest_path)
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
            rows_t = Storage._load_row(rows_file)
            preds_t = Storage._load_prediction(pred_file, dtype=dtype)
            Storage._index_copy_rows(y_out, rows_t, preds_t, count=int(count))
        return y_out

    @staticmethod
    def concat_segment_h5(
        out_path: str,
        *args: Any,
        memmap_dir: str,
        chunks_dir: str,
        count: object,
        out_shape: object,
        store_float: object,
        chunk_size: int = 8192,
    ) -> PersistentTensorDict:
        _ = args
        x_mmt = Storage.load_memmap_features(memmap_dir)
        out_shape_t = tuple(int(x) for x in out_shape)
        os.makedirs(os.path.dirname(out_path) or os.getcwd(), exist_ok=True)
        np_float = Storage._to_numpy_dtype(store_float)
        cast_dtype = store_float
        if store_float == torch.bfloat16 and np_float == numpy.float32:
            cast_dtype = torch.float32
        with h5py.File(out_path, "w") as f:
            dset_X = f.create_dataset(
                "X",
                shape=tuple(x_mmt.shape),
                dtype=Storage._to_numpy_dtype(x_mmt.dtype),
            )
            dset_Y = f.create_dataset("Y", shape=(int(count), *out_shape_t), dtype=np_float)
            step = int(chunk_size)
            for s in range(0, int(count), step):
                e = min(int(count), s + step)
                dset_X[s:e] = x_mmt[s:e].detach().to(device="cpu").numpy()
            manifest = schemas.read_json(os.path.join(chunks_dir, "manifest.json"))
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
                rows_t = Storage._load_row(rows_file)
                preds_t = Storage._load_prediction(pred_file, dtype=cast_dtype)
                preds_np = preds_t.detach().to(device="cpu", dtype=cast_dtype).numpy()
                if predsnp.shape[0] != int(rows_t.numel()):
                    raise ValueError(
                        f"Pred/rows mismatch in {pred_file}: preds[0]={predsnp.shape[0]} vs rows={int(rows_t.numel())}"
                    )
                Storage._h5_write_rows(dset_Y, rows_t, preds_np, count=int(count))
        return PersistentTensorDict(filename=out_path, batch_size=[int(count)], mode="r")

    @staticmethod
    def write_predictions_h5_from_memmap(
        out_path: str,
        *args: Any,
        memmap_dir: str,
        pred_path: str,
        count: object | None = None,
        chunk_size: int = 8192,
    ) -> PersistentTensorDict:
        _ = args
        x_mmt = Storage.load_memmap_features(memmap_dir)
        y_mmt = Storage.open_memory_mapped_tensor(os.fspath(pred_path))
        if y_mmt is None:
            raise FileNotFoundError(f"missing prediction memmap: {pred_path}")
        n = int(y_mmt.shape[0]) if count is None else int(count)
        if n <= 0:
            raise ValueError(f"non-positive prediction count: {n}")
        x_np_dtype = Storage._to_numpy_dtype(x_mmt.dtype)
        y_np_dtype = Storage._to_numpy_dtype(y_mmt.dtype)
        y_cast_dtype = y_mmt.dtype
        if y_mmt.dtype == torch.bfloat16 and y_np_dtype == numpy.float32:
            y_cast_dtype = torch.float32
        out_parent = os.path.dirname(out_path) or "."
        os.makedirs(out_parent, exist_ok=True)
        step = int(chunk_size)
        with h5py.File(out_path, "w") as f:
            dset_X = f.create_dataset(
                "X",
                shape=(n, int(x_mmt.shape[1])),
                dtype=x_np_dtype,
                chunks=(min(n, step), int(x_mmt.shape[1])),
            )
            dset_Y = f.create_dataset(
                "Y",
                shape=(n, *[int(x) for x in y_mmt.shape[1:]]),
                dtype=y_np_dtype,
                chunks=(min(n, step), *[int(x) for x in y_mmt.shape[1:]]),
            )
            for s in range(0, n, step):
                e = min(n, s + step)
                dset_X[s:e] = x_mmt[s:e].detach().to(device="cpu", dtype=x_mmt.dtype).numpy()
                dset_Y[s:e] = y_mmt[s:e].detach().to(device="cpu", dtype=y_cast_dtype).numpy()
        return PersistentTensorDict(filename=out_path, batch_size=[int(n)], mode="r")

    @staticmethod
    def write_predictions_h5_atomic(
        out_path: str,
        *args: Any,
        memmap_dir: str,
        pred_path: str,
        chunk_size: int = 8192,
        overwrite: str = "replace",
    ) -> PersistentTensorDict:
        return Storage._atomic_h5_op(
            out_path,
            overwrite,
            lambda tmp: Storage.write_predictions_h5_from_memmap(
                tmp,
                memmap_dir=memmap_dir,
                pred_path=pred_path,
                chunk_size=chunk_size,
            ),
        )

    @staticmethod
    def copy_predictions_h5_atomic(
        src_path: str,
        dst_path: str,
        *args: Any,
        overwrite: str = "replace",
        out_shape: object | None = None,
    ) -> PersistentTensorDict:
        Storage.validate_predictions_h5(src_path, out_shape=out_shape)
        res = Storage._atomic_h5_op(dst_path, overwrite, lambda tmp: shutil.copy2(src_path, tmp))
        Storage.validate_predictions_h5(dst_path, out_shape=out_shape)
        return res

    @staticmethod
    def _atomic_h5_op(out_path, overwrite, op_fn):
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

    @staticmethod
    def remove_prediction_artifacts(*args: Any, memmap_dir: str, pred_path: str) -> None:
        _ = args
        try:
            meta = Storage.load_memmap_meta(memmap_dir)
            feat_rel = str(meta.get("features_path", "features.mmt"))
            feat_path = os.path.join(os.fspath(memmap_dir), feat_rel)
        except Exception:
            feat_path = None

        if feat_path:
            _remove_safe(os.fspath(feat_path))
            with suppress(Exception):
                _remove_safe(Storage.get_meta_path(os.fspath(feat_path)))
        _remove_safe(os.fspath(pred_path))
        with suppress(Exception):
            _remove_safe(Storage.get_meta_path(os.fspath(pred_path)))
        _remove_safe(os.path.join(os.fspath(memmap_dir), "meta.json"))
        with suppress(OSError):
            os.rmdir(os.fspath(memmap_dir))


class Source(TypedDict):
    format: SourceType
    path: str


class Disposable:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._keep: list[Any] = list(_flatten_args(list(args)))
        if kwargs:
            self._keep.extend(list(_flatten_args(list(kwargs.values()))))

    def add(self, *args: Any, **kwargs: Any) -> None:
        self._keep.extend(list(_flatten_args(list(args))))
        if kwargs:
            self._keep.extend(list(_flatten_args(list(kwargs.values()))))

    def cleanup(self) -> None:
        for obj in self._keep:
            close(obj)

    def close(self) -> None:
        self.cleanup()

    def __iter__(self) -> Iterator[Any]:
        return iter(self._keep)


class BatchScaler:
    __slots__ = ("_v", "_min_scale", "_max_scale")

    def __init__(
        self,
        scale: float = 1.0,
        *args: Any,
        min_scale: float = 0.5,
        max_scale: float = 2.0,
    ) -> None:
        self._min_scale = float(min_scale)
        self._max_scale = float(max_scale)
        self._v = multiprocessing.Value("d", 1.0, lock=True)
        self.reset(scale)

    def __getstate__(self):
        return (self._v, float(self._min_scale), float(self._max_scale))

    def __setstate__(self, state):
        try:
            v, mn, mx = state
        except Exception:
            v, mn, mx = None, 0.5, 2.0
        self._min_scale = float(mn) if isinstance(mn, (int, float, str)) else 0.5
        self._max_scale = float(mx) if isinstance(mx, (int, float, str)) else 2.0
        if v is None:
            self._v = multiprocessing.Value("d", 1.0, lock=True)
        else:
            self._v = v
        try:
            if not hasattr(self._v, "get_lock"):
                scale = float(v)
                self._v = multiprocessing.Value("d", 1.0, lock=True)
                self.reset(scale)
        except Exception:
            self._v = multiprocessing.Value("d", 1.0, lock=True)

    def get(self) -> float:
        mn = float(self._min_scale)
        mx = float(self._max_scale)
        try:
            v = float(self._v.value)
        except Exception:
            v = float("nan")
        if math.isfinite(v) and (v > 0.0) and (mn <= v <= mx):
            return float(v)
        try:
            with self._v.get_lock():
                v = float(self._v.value)
        except Exception:
            with suppress(Exception):
                v = float(self._v.value)
            if not isinstance(v, (int, float)):
                v = 1.0
        if (not math.isfinite(v)) or (not (v > 0.0)):
            v = 1.0
        if v < mn:
            v = mn
        elif v > mx:
            v = mx
        return float(v)

    def reset(self, value: float = 1.0) -> None:
        try:
            v = float(value)
        except Exception:
            v = 1.0
        if not (v > 0.0):
            v = 1.0
        v = max(float(self._min_scale), min(float(self._max_scale), float(v)))
        try:
            with self._v.get_lock():
                self._v.value = float(v)
        except Exception:
            with suppress(Exception):
                self._v.value = float(v)

    def request_scale_up(self, factor: float) -> None:
        try:
            f = float(factor)
        except Exception:
            f = 1.0
        if not (f > 0.0):
            return
        try:
            with self._v.get_lock():
                cur = float(self._v.value)
                self._v.value = float(min(float(self._max_scale), cur * float(f)))
        except Exception:
            pass

    def request_scale_down(self, factor: float) -> None:
        try:
            f = float(factor)
        except Exception:
            f = 1.0
        if not (f > 0.0):
            return
        try:
            with self._v.get_lock():
                cur = float(self._v.value)
                self._v.value = float(max(float(self._min_scale), cur * float(f)))
        except Exception:
            pass


class BatchQueue(Buffer):
    def __init__(
        self,
        iterable: Any,
        *args: Any,
        max_batches: int = 4,
        name: str = "buffer",
        daemon: bool = True,
        _session: bool = False,
    ) -> None:
        super().__init__(max_batches=max_batches)
        self._src = iterable
        self._name = str(name or "buffer")
        self._daemon = bool(daemon)
        self._session = bool(_session)
        self._join_timeout_s = 0.5
        with suppress(Exception):
            jt_ms = int(env_first_int(("STNET_THREAD_JOIN_TIMEOUT_MS",), default=500) or 500)
            self._join_timeout_s = max(0.0, float(jt_ms) / 1000.0)

    def __len__(self) -> int:
        try:
            return int(len(self._src))
        except Exception:
            return 1

    def __iter__(self) -> Iterator[Any]:
        if not bool(self._session):
            return iter(
                BatchQueue(
                    self._src,
                    max_batches=int(self.max_batches),
                    name=self._name,
                    daemon=self._daemon,
                    _session=True,
                )
            )
        return self._iter_session()

    def _producer_loop(self, it: Iterator[Any], sentinel: object) -> None:
        try:
            while not self.is_stopped():
                if not self.block(timeout=None):
                    break
                try:
                    item = next(it)
                except StopIteration:
                    break
                if self.is_stopped():
                    break
                if not self.put(item, timeout=0.0):
                    break
        except BaseException as exc:
            with suppress(Exception):
                self.put(ProducerError(exc=exc, tb=traceback.format_exc()))
        finally:
            with suppress(Exception):
                self.put(sentinel)

    def _iter_session(self) -> Iterator[Any]:
        src_iter = iter(self._src)
        sentinel = object()
        t = threading.Thread(
            target=self._producer_loop,
            name=f"{self._name}-producer",
            daemon=self._daemon,
            args=(src_iter, sentinel),
        )
        t.start()
        try:
            while True:
                try:
                    item = self.get(timeout=None)
                except queue.Empty:
                    break
                if item is sentinel:
                    break
                if isinstance(item, ProducerError):
                    if isinstance(item.exc, (KeyboardInterrupt, SystemExit)):
                        raise item.exc
                    raise RuntimeError(
                        f"BatchQueue producer crashed: {item.exc}\n{item.tb}"
                    ) from item.exc
                yield item
        finally:
            self.stop()
            with suppress(Exception):
                self.clear()
            close(src_iter)
            with suppress(Exception):
                if t.is_alive():
                    t.join(timeout=float(getattr(self, "_join_timeout_s", 0.5)))
            close(t)


class Sampler(torch.utils.data.Sampler):
    _per_sample_mem_bytes: int = 0

    def __init__(
        self,
        memmap_dir: str,
        *args: Any,
        split: str = "train",
        val_frac: float = 0.0,
        sampler_scale: Optional["BatchScaler"] = None,
        **kwargs: Any,
    ) -> None:
        self.dir = os.fspath(memmap_dir)
        self.split = str(split)
        self._meta: Mapping[str, Any] = self._load_meta(self.dir)
        self._sampler_scale = sampler_scale if sampler_scale is not None else BatchScaler()
        self._S_B_cap = 0
        self._N = int(self._meta.get("N", 0))
        if self._N <= 0:
            raise ValueError(f"meta.json under {self.dir} has non-positive N={self._N}")
        feat_rel = str(self._meta.get("features_path", "features.mmt"))
        feat_path = os.path.join(self.dir, feat_rel)
        lab_rel_raw = self._meta.get("labels_path", "labels.mmt")
        features_only = bool(self._meta.get("features_only", False))
        lab_rel = ""
        lab_path = ""
        if lab_rel_raw not in (None, "", False):
            with suppress(Exception):
                lab_rel = str(lab_rel_raw)
        if lab_rel.strip().lower() in ("none", "null"):
            lab_rel = ""
        if lab_rel:
            lab_path = os.path.join(self.dir, lab_rel)
            if not os.path.isfile(lab_path):
                if features_only:
                    lab_rel = ""
                    lab_path = ""
                else:
                    raise FileNotFoundError(f"labels.mmt not found under: {lab_path}")
        fdim = int(self._meta.get("feature_dim", 0))
        lshape_meta = list(self._meta.get("label_shape") or [])
        f_dtype = dtype_from_name(self._meta.get("features_dtype", "float64"), torch.float64)
        l_dtype = dtype_from_name(self._meta.get("labels_dtype", "int64"), torch.int64)
        self._include_row_ids = env_bool("STNET_INCLUDE_ROW_IDS", default=True)
        self._feat_path = feat_path
        self._lab_path = lab_path if lab_path else None
        self._feat_dtype = f_dtype
        self._label_dtype = l_dtype
        self._feat_shape = torch.Size([self._N, fdim])
        if MemoryMappedTensor is None:
            raise ImportError(
                "tensordict is required for MemoryMappedTensor-backed pipelines. "
                "Please install 'tensordict' (or install stnet-pytorch with its default dependencies)."
            )
        self._features = MemoryMappedTensor.from_filename(
            filename=feat_path, dtype=f_dtype, shape=torch.Size([self._N, fdim])
        )
        lshape = tuple(lshape_meta)
        self._label_shape_full = torch.Size([self._N, *lshape])
        self._labels = (
            MemoryMappedTensor.from_filename(
                filename=str(self._lab_path),
                dtype=l_dtype,
                shape=self._label_shape_full,
            )
            if self._lab_path
            else None
        )
        self._mmap_tls, self._mmap_limit_lock = None, threading.Lock()
        self._mmap_thread_local = False
        self._mmap_thread_local_max = 0
        self._mmap_thread_local_created = 0
        self._mmap_thread_local_overflow_warned = False
        default_tl = False
        with suppress(Exception):
            default_tl = bool(CPU.is_optimized_for_no_gil())
        self._mmap_thread_local = env_bool(
            (
                "STNET_MEMMAP_THREAD_LOCAL_HANDLES",
                "STNET_MEMMAP_THREAD_LOCAL",
                "STNET_NOGIL",
            ),
            default=default_tl,
        )
        if self._mmap_thread_local:
            cpu = int(CPU.count() or 8)
            default_max = max(8, min(64, cpu))
            self._mmap_thread_local_max = env_first_int(
                ("STNET_MEMMAP_THREAD_LOCAL_MAX", "STNET_MEMMAP_TL_MAX"),
                default=default_max,
            )
            self._mmap_thread_local_max = int(self._mmap_thread_local_max)
        self._memmap_features = self._features
        self._memmap_labels = self._labels
        self._X = self._memmap_features
        self._Y = self._memmap_labels
        self._S_B_base = 1
        self._S_B = 1
        self._S_shuffle = True
        self._S_seed = 0
        self._S_epoch = 0
        self._len_epoch = -1
        self._len_B_snapshot: Optional[int] = None
        self._num_shards = 1
        self._shard_id = 0
        self._key = ""
        self._label_shape: Tuple[int, ...] = tuple(lshape) if lshape else tuple()
        train_start = int(self._meta.get("train_start", 0))
        train_end = int(self._meta.get("train_end", self._N))
        val_start = int(self._meta.get("val_start", 0))
        val_end = int(self._meta.get("val_end", 0))
        if val_frac and not (val_end > val_start):
            vf = float(val_frac)
            vc = max(0, min(self._N, int(self._N * vf)))
            val_start, val_end = max(0, self._N - vc), self._N
            train_start, train_end = 0, val_start
        if self.split == "val":
            self._start, self._end = (val_start, val_end) if val_end > val_start else (0, 0)
        else:
            self._start, self._end = (train_start, train_end)

    @classmethod
    def _load_meta(cls: type["Sampler"], memmap_dir: str) -> Mapping[str, Any]:
        meta_path = os.path.join(os.fspath(memmap_dir), "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"meta.json not found under: {memmap_dir}")
        meta = schemas.read_json(meta_path)
        if not isinstance(meta, Mapping):
            raise ValueError(f"meta.json under {memmap_dir} must contain a mapping")
        return meta

    def _effective_batch_size(self) -> int:
        base = self.base_batch_size
        scale = 1.0
        with suppress(Exception):
            scale = float(self._sampler_scale.get())
        if not (scale > 0.0):
            scale = 1.0
        eff = int(round(float(base) * float(scale)))
        eff = max(1, int(eff))
        cap = 0
        with suppress(Exception):
            cap = int(getattr(self, "_S_B_cap", 0) or 0)
        if cap > 0:
            eff = min(int(eff), int(cap))
        return max(1, int(eff))

    def _len_batch_size(self) -> int:
        epoch = int(getattr(self, "_S_epoch", 0) or 0)
        snap_epoch = int(getattr(self, "_len_epoch", -1) or -1)
        snap = getattr(self, "_len_B_snapshot", None)
        if snap is None or snap_epoch != epoch:
            snap = max(1, int(self._effective_batch_size()))
            self._len_B_snapshot = int(snap)
            self._len_epoch = int(epoch)
        return max(1, int(snap))

    @property
    def _S_B(self) -> int:
        return self._effective_batch_size()

    @_S_B.setter
    def _S_B(self, value: int) -> None:
        try:
            v = int(value)
        except Exception:
            v = 1
        if v <= 0:
            v = 1
        setattr(self, "_S_B_base", int(v))

    def __getstate__(self):
        state = dict(self.__dict__)
        for key in (
            "_features",
            "_labels",
            "_memmap_features",
            "_memmap_labels",
            "_X",
            "_Y",
            "_mmap_tls",
            "_mmap_limit_lock",
        ):
            state.pop(key, None)
        state["_mmap_tls"] = None
        state["_mmap_limit_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._mmap_tls = None
        self._mmap_limit_lock = threading.Lock()
        if MemoryMappedTensor is None:
            raise ImportError(
                "tensordict is required for MemoryMappedTensor-backed pipelines. "
                "Please install 'tensordict' (or install stnet-pytorch with its default dependencies)."
            )
        self._features = MemoryMappedTensor.from_filename(
            filename=str(self._feat_path),
            dtype=self._feat_dtype,
            shape=self._feat_shape,
        )
        self._labels = None
        if getattr(self, "_lab_path", None):
            self._labels = MemoryMappedTensor.from_filename(
                filename=str(self._lab_path),
                dtype=self._label_dtype,
                shape=self._label_shape_full,
            )
        self._memmap_features = self._features
        self._memmap_labels = self._labels
        self._X = self._memmap_features
        self._Y = self._memmap_labels

    def _get_mmaps(self):
        if not getattr(self, "_mmap_thread_local", False):
            return self._features, self._labels
        has_labels = bool(self._labels is not None and self._lab_path)
        tls = self._mmap_tls or threading.local()
        self._mmap_tls = tls
        if getattr(tls, "features", None) is not None:
            return tls.features, getattr(tls, "labels", None)
        max_pairs = int(getattr(self, "_mmap_thread_local_max", 0) or 0)
        if max_pairs > 0:
            with self._mmap_limit_lock:
                created = self._mmap_thread_local_created
                if created >= max_pairs:
                    if not self._mmap_thread_local_overflow_warned:
                        self._mmap_thread_local_overflow_warned = True
                    return self._features, self._labels
                self._mmap_thread_local_created += 1
        try:
            f_new = MemoryMappedTensor.from_filename(
                filename=str(self._feat_path),
                dtype=self._feat_dtype,
                shape=self._feat_shape,
            )
            l_new = (
                MemoryMappedTensor.from_filename(
                    filename=str(self._lab_path),
                    dtype=self._label_dtype,
                    shape=self._label_shape_full,
                )
                if has_labels
                else None
            )
            tls.features, tls.labels = f_new, l_new
            return f_new, l_new
        except Exception:
            if max_pairs > 0:
                with self._mmap_limit_lock:
                    self._mmap_thread_local_created = max(0, self._mmap_thread_local_created - 1)
            return self._features, self._labels

    def _slice(self, start: int, end: int) -> Mapping[str, torch.Tensor]:
        start = int(start)
        end = int(end)
        features, labels = self._get_mmaps()
        x = features[start:end]
        xt = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        out: Dict[str, torch.Tensor] = {"X": xt}
        if bool(getattr(self, "_include_row_ids", True)):
            out["row_ids"] = torch.arange(start, end, dtype=torch.int64)
        if labels is not None:
            y = labels[start:end]
            if self._label_shape:
                y = y.reshape(end - start, *self._label_shape)
            yt = y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
            out["Y"] = yt
        return out

    def _gather(
        self, idx_tensor: torch.Tensor, features: Any, labels: Any
    ) -> Mapping[str, torch.Tensor]:
        idx_tensor = idx_tensor.to(dtype=torch.long, copy=False)
        try:
            x = features.index_select(0, idx_tensor)
        except Exception:
            x = (
                features[idx_tensor]
                if hasattr(features, "__getitem__")
                else torch.as_tensor(features)[idx_tensor]
            )
        out: Dict[str, torch.Tensor] = {"X": x}
        if bool(getattr(self, "_include_row_ids", True)):
            out["row_ids"] = idx_tensor.to(dtype=torch.int64, copy=False)
        if labels is not None:
            try:
                y = labels.index_select(0, idx_tensor)
            except Exception:
                y = (
                    labels[idx_tensor]
                    if hasattr(labels, "__getitem__")
                    else torch.as_tensor(labels)[idx_tensor]
                )
            if self._label_shape:
                y = y.reshape(y.shape[0], *self._label_shape)
            out["Y"] = y
        return out

    def __getitem__(
        self, idx: int | Tuple[int, int] | Sequence[int] | torch.Tensor
    ) -> Mapping[str, torch.Tensor]:
        features, labels = self._get_mmaps()
        base = int(self._start)
        match idx:
            case tuple() as t if len(t) == 2:
                s, e = int(t[0]), int(t[1])
                return self._slice(base + s, base + e)
            case torch.Tensor() as t:
                idx_t = t.detach()
                if idx_t.device.type != "cpu":
                    idx_t = idx_t.to("cpu")
                if idx_t.ndim == 0:
                    return self.__getitem__(int(idx_t.item()))
                idx_tensor = idx_t.to(dtype=torch.long, copy=False).reshape(-1)
                if idx_tensor.numel() == 0:
                    return self._slice(base, base)
                idx_tensor = idx_tensor + base
                return self._gather(idx_tensor, features, labels)
            case seq if isinstance(seq, Sequence) and not isinstance(seq, (str, bytes, bytearray)):
                if len(seq) == 0:
                    return self._slice(base, base)
                idx_tensor = torch.as_tensor(seq, dtype=torch.long).reshape(-1) + base
                return self._gather(idx_tensor, features, labels)
            case _:
                i = base + int(idx)
                out = self._slice(i, i + 1)
                with suppress(Exception):
                    x = out.get("X", None)
                    if torch.is_tensor(x):
                        out["X"] = x.squeeze(0)
                    y = out.get("Y", None)
                    if torch.is_tensor(y):
                        out["Y"] = y.squeeze(0)
                    r = out.get("row_ids", None)
                    if torch.is_tensor(r):
                        out["row_ids"] = r.squeeze(0)
                return out

    def _shard(self) -> None:
        try:
            dist = getattr(torch, "distributed", None)
            if dist is not None and dist.is_available() and dist.is_initialized():
                self._num_shards = max(1, int(dist.get_world_size()))
                self._shard_id = max(0, int(dist.get_rank()))
                return
        except Exception:
            pass
        world = env_first_int(("WORLD_SIZE",), default=1)
        rank = env_first_int(("RANK",), default=0)
        if world > 1:
            self._num_shards = max(1, int(world))
            self._shard_id = max(0, min(int(world) - 1, int(rank)))
        else:
            self._num_shards = 1
            self._shard_id = 0

    def __iter__(self) -> Iterator[tuple[str, tuple[int, int]]]:
        start = int(getattr(self, "start", 0))
        end = int(getattr(self, "end", 0))
        if end <= start:
            return
        self._shard()
        self._len_epoch = int(getattr(self, "_S_epoch", 0) or 0)
        self._len_B_snapshot = max(1, int(self._effective_batch_size()))
        ns = int(getattr(self, "_num_shards", 1) or 1)
        si = int(getattr(self, "_shard_id", 0) or 0)
        base = int(self.base_batch_size)
        max_scale = 2.0
        with suppress(Exception):
            max_scale = float(getattr(self._sampler_scale, "_max_scale", 2.0))
        block = max(1, int(math.ceil(float(base) * float(max_scale))))
        with suppress(Exception):
            cap = int(getattr(self, "_S_B_cap", 0) or 0)
            if cap > 0:
                block = min(int(block), int(cap))
        total = int(end - start)
        n_blocks = max(1, int((total + block - 1) // block))
        if bool(getattr(self, "_S_shuffle", True)):
            g = torch.Generator(device="cpu")
            g.manual_seed(int(getattr(self, "_S_seed", 0)) + int(getattr(self, "_S_epoch", 0)))
            order = torch.randperm(n_blocks, generator=g, dtype=torch.int64)
        else:
            order = torch.arange(n_blocks, dtype=torch.int64)
        if ns > 1:
            order = order[si::ns]
        for b in order:
            bs = start + int(b) * int(block)
            be = min(end, bs + int(block))
            cur = int(bs)
            while cur < int(be):
                B = int(self._effective_batch_size())
                nxt = min(int(be), cur + max(1, int(B)))
                if nxt <= cur:
                    break
                yield (self._key, (int(cur), int(nxt)))
                cur = int(nxt)

    def __len__(self) -> int:
        start = int(getattr(self, "start", 0))
        end = int(getattr(self, "end", 0))
        B = max(1, int(self._len_batch_size()))
        total = max(0, (end - start + B - 1) // B)
        ns = int(getattr(self, "_num_shards", 1) or 1)
        si = int(getattr(self, "_shard_id", 0) or 0)
        if ns <= 1:
            return int(total)
        return max(0, (int(total) - si + ns - 1) // ns)

    @property
    def sampler_scale(self) -> "BatchScaler":
        return self._sampler_scale

    @property
    def base_batch_size(self) -> int:
        try:
            b = int(getattr(self, "_S_B_base", 1) or 1)
        except Exception:
            b = 1
        return max(1, int(b))

    @property
    def start(self) -> int:
        return int(self._start)

    @property
    def end(self) -> int:
        return int(self._end)

    @property
    def meta(self) -> Mapping[str, Any]:
        return dict(self._meta or {})

    def compose(
        self,
        *args: Any,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        key: str = "",
        **kwargs: Any,
    ) -> "BaseNode":
        self._S_B = max(1, int(batch_size))
        self._S_shuffle = bool(shuffle)
        self._S_seed = int(seed)
        self._S_epoch = 0
        self._key = str(key)
        self._len_epoch = int(self._S_epoch)
        self._len_B_snapshot = max(1, int(self._effective_batch_size()))
        self._shard()
        if SamplerWrapper is None:
            raise RuntimeError("torchdata.nodes.SamplerWrapper is required")
        return SamplerWrapper(self)

    def set_epoch(self, epoch: int) -> None:
        self._S_epoch = int(epoch)
        self._len_epoch = int(self._S_epoch)
        self._len_B_snapshot = max(1, int(self._effective_batch_size()))

    def get(self, start: int, end: int) -> Mapping[str, Any]:
        s = int(start)
        e = int(end)
        n = max(0, e - s)
        features, labels = self._get_mmaps()
        with inference_mode(torch.nn.Module()):
            if n <= 0:
                X0 = features.narrow(0, 0, 0)
                out0: Dict[str, Any] = {"X": X0}
                if labels is not None:
                    out0["Y"] = labels.narrow(0, 0, 0)
                return out0
            X = features.narrow(0, s, n)
            out: Dict[str, Any] = {"X": X}
            if labels is not None:
                Y = labels.narrow(0, s, n)
                if self._label_shape:
                    Y = Y.reshape(n, *self._label_shape)
                out["Y"] = Y
            pin_in_dataset = env_bool("STNET_PIN_IN_DATASET", default=False)
            if pin_in_dataset and _is_accelerator_available():
                with suppress(Exception):
                    out["X"] = out["X"].pin_memory()
                    if out.get("Y") is not None:
                        out["Y"] = out["Y"].pin_memory()
            return out


class Multiplexer:
    def __init__(
        self,
        *args: Any,
        stop_criteria: str = "ALL_DATASETS_EXHAUSTED",
        weights: Optional[Mapping[str, float] | Sequence[float] | float | int] = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        _ = args, kwargs
        self.stop_criteria = str(stop_criteria)
        self._raw_weights = weights
        self.seed = int(seed)
        self._epoch = 0
        self._node: Optional[Any] = None
        self._source_keys: list[str] = []

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        node = getattr(self, "_node", None)
        if node is None:
            return
        reset = getattr(node, "reset", None)
        if not callable(reset):
            return

        epoch_key = _node_state_key(node, "EPOCH_KEY", "epoch")
        ws_key = _node_state_key(node, "WEIGHTED_SAMPLER_STATE_KEY", "weighted_sampler_state")
        ny_key = _node_state_key(node, "NUM_YIELDED_KEY", "num_yielded")
        ex_key = _node_state_key(node, "DATASETS_EXHAUSTED_KEY", "datasets_exhausted")
        dns_key = _node_state_key(node, "DATASET_NODE_STATES_KEY", "dataset_node_states")
        keys = list(getattr(self, "_source_keys", []) or [])
        initial_state: Dict[str, Any] = {
            epoch_key: int(self._epoch),
            ny_key: 0,
            ws_key: None,
        }
        if keys:
            initial_state[ex_key] = {k: False for k in keys}
            initial_state[dns_key] = {k: None for k in keys}
        try:
            reset(initial_state)
            return
        except Exception:
            pass
        with suppress(Exception):
            setattr(node, "seed", int(self.seed) + int(self._epoch))
        with suppress(Exception):
            reset(None)

    def compose(
        self, sources: Mapping[str, "BaseNode"] | Sequence["BaseNode"] | "BaseNode"
    ) -> "BaseNode":
        sources_kind: str
        if isinstance(sources, BaseNode):
            sources_kind = "single"
            sources_map = {"0": sources}
        elif isinstance(sources, (list, tuple)):
            sources_kind = "sequence"
            if len(sources) < 1:
                raise ValueError("sources must be non-empty")
            sources_map = (
                {"0": sources[0]}
                if len(sources) == 1
                else {str(i): n for i, n in enumerate(sources)}
            )
        elif isinstance(sources, Mapping):
            sources_kind = "mapping"
            if len(sources) < 1:
                raise ValueError("sources must be non-empty")
            sources_map: Dict[str, BaseNode] = {}
            for k, v in dict(sources).items():
                kk = str(k)
                if kk in sources_map:
                    raise ValueError(f"sources has duplicate key after str(): {kk!r}")
                sources_map[kk] = v
        else:
            raise TypeError(
                "sources must be a BaseNode, Sequence[BaseNode], or Mapping[str, BaseNode]"
            )
        if len(sources_map) <= 1:
            w = {k: 1.0 for k in sources_map.keys()}
            self._source_keys = list(sources_map.keys())
            node = MultiNodeWeightedSampler(
                sources_map,
                w,
                stop_criteria=self.stop_criteria,
                seed=int(self.seed),
            )
            self._node = node
            return node
        raw = getattr(self, "_raw_weights", None)

        def _coerce_weight(v: Any, *args: Any, where: str) -> float:
            try:
                fv = float(v)
            except Exception as exc:
                raise TypeError(f"weights entry must be numeric (float/int): {where}") from exc
            if not math.isfinite(fv):
                raise ValueError(f"weights entry must be finite: {where}")
            if fv < 0.0:
                raise ValueError(f"weights entry must be >= 0: {where}")
            return float(fv)

        w: Dict[str, float]
        if raw is None:
            w = {k: 1.0 for k in sources_map.keys()}
        elif isinstance(raw, Mapping):
            if sources_kind != "mapping":
                raise TypeError("weights must be a Mapping[str, float] when sources is a Mapping")
            w_in: Dict[str, float] = {}
            for k, v in dict(raw).items():
                kk = str(k)
                if kk in w_in:
                    raise ValueError(f"weights has duplicate key after str(): {kk!r}")
                w_in[kk] = _coerce_weight(v, where=f"weights[{kk!r}]")
            if not w_in:
                raise ValueError("weights mapping must be non-empty (use None for uniform)")
            missing = set(sources_map.keys()) - set(w_in.keys())
            extra = set(w_in.keys()) - set(sources_map.keys())
            if missing or extra:
                raise ValueError(
                    "weights mapping keys must match sources keys exactly; "
                    f"missing={sorted(missing)} extra={sorted(extra)} sources={sorted(sources_map.keys())}"
                )
            if not any((float(v) > 0.0) for v in w_in.values()):
                raise ValueError("weights mapping must contain at least one positive weight")
            w = {k: float(w_in[k]) for k in sources_map.keys()}
        elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
            raise TypeError(
                "scalar weights are only supported for a single source (omit weights for multi-source)"
            )
        elif isinstance(raw, collections.abc.Sequence) and not isinstance(
            raw, (str, bytes, bytearray)
        ):
            if sources_kind != "sequence":
                raise TypeError("weights must be a Sequence[float] when sources is a Sequence")
            seq = list(raw)
            expected = len(sources_map)
            if len(seq) != expected:
                raise ValueError(
                    f"weights sequence length mismatch: expected {expected}, got {len(seq)}"
                )
            w_seq = [_coerce_weight(v, where=f"weights[{i}]") for i, v in enumerate(seq)]
            if not any((float(v) > 0.0) for v in w_seq):
                raise ValueError("weights sequence must contain at least one positive weight")
            if not all(str(k).isdigit() for k in sources_map.keys()):
                raise ValueError("sequence weights require digit-only source keys ('0','1',...)")
            w = {k: float(w_seq[int(k)]) for k in sources_map.keys()}
        else:
            raise TypeError(
                "weights must be a Mapping[str, float] or Sequence[float] (or omitted for uniform)"
            )
        if not any(v > 0.0 for v in w.values()):
            raise ValueError("At least one weight must be > 0")
        self._source_keys = list(sources_map.keys())
        node = MultiNodeWeightedSampler(
            sources_map, w, stop_criteria=self.stop_criteria, seed=int(self.seed)
        )
        self._node = node
        return node


class Mapper:
    def __init__(
        self,
        *args: Any,
        map_fn: Callable[[Any], Any],
        io_workers: Optional[int] = None,
        prebatch: Optional[int] = None,
        prefetch_factor: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
        non_blocking: bool = True,
        pin_memory: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        self.map_fn = map_fn
        wp = WorkerPolicy.optimize()
        wp.set_thread_setting()
        self.io_workers = (
            int(io_workers) if io_workers is not None else int(getattr(wp, "num_workers", 1))
        )
        self.io_workers = max(1, self.io_workers)
        self.prebatch = (
            int(prebatch)
            if prebatch is not None
            else int(getattr(wp, "prebatch", max(1, self.io_workers * 2)))
        )
        with suppress(Exception):
            self.prebatch = max(1, int(self.prebatch))
        pf = (
            int(prefetch_factor)
            if prefetch_factor is not None
            else int(getattr(wp, "prefetch_factor", 1))
        )
        with suppress(Exception):
            pf = max(1, int(pf))
        self._prefetch_factor = pf
        self.prefetch_factor = self._prefetch_factor
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.non_blocking = bool(non_blocking)
        pin = (
            bool(pin_memory)
            if pin_memory is not None
            else (getattr(self.device, "type", "cpu") in {"cuda", "xpu"})
        )
        self._pin_memory = pin
        self.pin_memory = self._pin_memory
        self.max_concurrency = max(1, int(self.io_workers))
        try:
            get_affinity(io_workers=self.io_workers)
        except Exception:
            pass

    def compose(self, source: "BaseNode") -> "BaseNode":
        node: BaseNode = source
        mapper = self.map_fn
        if (self.prebatch or 0) and int(self.prebatch) > 1:
            B = max(1, int(self.prebatch))
            node = torchdata.nodes.Batcher(node, batch_size=B, drop_last=False)
            pm_map_fn = new_thread(mapper)
        else:
            pm_map_fn = new_thread(mapper)
        node = ParallelMapper(
            node,
            map_fn=pm_map_fn,
            num_workers=self.io_workers,
            in_order=False,
            method="thread",
            max_concurrent=int(self.max_concurrency),
        )
        if (self.prebatch or 0) and int(self.prebatch) > 1:
            node = torchdata.nodes.Unbatcher(node)
        return node


class Loader:
    def __init__(
        self,
        device: torch.device | str | Sequence[torch.device | str],
        *args: Any,
        node: BaseNode | None = None,
        dataset: BaseNode | None = None,
        prefetch_factor: int = 2,
        non_blocking: bool = True,
        length: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        node_obj = node or dataset
        if not isinstance(node_obj, BaseNode):
            raise TypeError("Loader supports only torchdata.nodes.BaseNode instances.")
        self._device = _normalize_device_spec(device)
        self._non_blocking = bool(non_blocking)
        self._length = int(length) if length is not None else None
        depth = max(1, int(prefetch_factor))
        with suppress(Exception):
            depth_env = int(env_first_int(("STNET_PREFETCH_DEPTH",), default=0) or 0)
            if depth_env > 0:
                depth = int(depth_env)
        self._prefetch_depth = max(1, min(32, int(depth)))
        prim = _primary_device(self._device)
        dev_t = getattr(prim, "type", "cpu")
        default_pin = dev_t in {"cuda", "xpu"}
        self._pin_host = bool(pin_memory) if pin_memory is not None else bool(default_pin)
        if dev_t == "cuda" and self._non_blocking:
            gpu_guard_mb = 2048
        elif dev_t in {"xpu"} and self._non_blocking:
            gpu_guard_mb = 512
        else:
            gpu_guard_mb = 0
        host_guard_mb = 1024 if self._non_blocking else 0
        with suppress(Exception):
            gpu_guard_mb = int(
                env_first_int(("STNET_GPU_GUARD_MB",), default=gpu_guard_mb) or gpu_guard_mb
            )
        with suppress(Exception):
            host_guard_mb = int(
                env_first_int(("STNET_HOST_GUARD_MB",), default=host_guard_mb) or host_guard_mb
            )
        self._gpu_guard_bytes = int(max(0, gpu_guard_mb) * (1 << 20))
        self._host_guard_bytes = int(max(0, host_guard_mb) * (1 << 20))
        self._node = node_obj
        self._base_iterable = (
            node_obj
            if isinstance(node_obj, torchdata.nodes.Loader)
            else torchdata.nodes.Loader(node_obj)
        )
        self._thread2dev: Dict[int, torch.device] = {}
        self._threads_hint = self._infer_mapper_threads(node_obj) if node_obj is not None else 1
        self._num_shards = 1
        self._shard_id = 0
        try:
            acc = max(1, int(get_num_accelerators("cuda") or 0))
            if acc <= 0:
                acc = max(1, int(get_num_accelerators("xpu") or 0))
            thr = max(1, int(self._threads_hint))
            self._num_shards = acc * thr
            dev_idx = self._local_device_index()
            self._shard_id = max(0, min(self._num_shards - 1, int(dev_idx * thr)))
        except Exception:
            pass

    def __iter__(self) -> Iterator[Any]:
        dev = self._device_for_current_thread()
        dev_t = getattr(dev, "type", "cpu")
        use_accel = dev_t in {"cuda", "xpu", "mps"}
        use_prefetch = bool(use_accel and self._non_blocking)
        iterable: Any = self._base_iterable
        if use_prefetch:
            iterable = Prefetcher(
                iterable,
                device=dev,
                depth=int(self._prefetch_depth),
                non_blocking=True,
                memory_backpressure=True,
                gpu_guard_bytes=int(self._gpu_guard_bytes),
                host_guard_bytes=int(self._host_guard_bytes),
                pin_host=bool(self._pin_host and dev_t in {"cuda", "xpu"}),
            )
        return iter(iterable)

    def __len__(self) -> int:
        if self._length is not None:
            return int(self._length)
        try:
            return int(len(self._base_iterable))
        except Exception:
            return 1

    def _device_for_current_thread(self) -> torch.device:
        if isinstance(self._device, list):
            tid = threading.get_ident()
            dev = self._thread2dev.get(tid)
            if dev is None:
                idx = len(self._thread2dev) % max(1, len(self._device))
                dev = self._device[idx]
                self._thread2dev[tid] = dev
            return dev
        return self._device

    def _infer_mapper_threads(self, node: Any) -> int:
        if node is None:
            return 1
        if hasattr(node, "num_workers"):
            with suppress(Exception):
                return int(getattr(node, "num_workers"))
        for key in ("child", "source", "node", "_node"):
            sub = getattr(node, key, None)
            if sub is not None:
                count = self._infer_mapper_threads(sub)
                if count > 0:
                    return count
        return 1

    def _local_device_index(self) -> int:
        try:
            if is_accelerator_available("cuda"):
                return int(get_accelerator_index("cuda"))
            if is_accelerator_available("xpu"):
                return int(get_accelerator_index("xpu"))
        except Exception:
            pass
        return 0

    @staticmethod
    def compose(
        source: "BaseNode",
        *args: Any,
        device: torch.device | str | Sequence[torch.device | str],
        prefetch_factor: int = 2,
        non_blocking: bool = True,
        length: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        **kwargs: Any,
    ) -> "Loader":
        if not isinstance(source, BaseNode):
            raise TypeError("Loader.compose expects a torchdata.nodes.BaseNode source.")
        return Loader(
            device=device,
            node=source,
            prefetch_factor=int(prefetch_factor),
            non_blocking=bool(non_blocking),
            length=length,
            pin_memory=pin_memory,
        )


class Prefetcher(Buffer):
    def __init__(
        self,
        iterable: Any,
        *args: Any,
        device: torch.device | str,
        depth: int = 2,
        non_blocking: bool = True,
        oom_safe: bool = True,
        memory_backpressure: bool | None = None,
        gpu_guard_bytes: int | None = None,
        host_guard_bytes: int | None = None,
        _session: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_batches=depth)
        self._src = iterable
        self._device = torch.device(device) if not isinstance(device, torch.device) else device
        self._depth = max(1, int(depth))
        self._non_blocking = bool(non_blocking)
        self._backpressure = (
            bool(memory_backpressure) if memory_backpressure is not None else bool(oom_safe)
        )
        self._gpu_guard_bytes = int(gpu_guard_bytes or 0)
        self._host_guard_bytes = int(host_guard_bytes or 0)
        use_accel = isinstance(self._device, torch.device) and self._device.type in (
            "cuda",
            "xpu",
            "mps",
        )
        self._pin = bool(kwargs.get("pin_host", use_accel))
        self._pin_pool = False
        self._host_pool: Optional[Pool] = None
        if self._pin and self._non_blocking and is_stream_supported(self._device.type):
            use_pool = env_bool("STNET_PREFETCH_PIN_POOL", default=True)
            cap_default = max(8, max(2, int(self._depth) * 2))
            cap = env_first_int(("STNET_PREFETCH_PIN_POOL_CAPACITY",), default=cap_default)
            if use_pool and int(cap) > 0:
                self._host_pool = Pool(capacity=int(cap), pin_memory=True)
                self._pin_pool = True
        self._accel_stream: Optional[object] = None
        self._accel_event_pool: Optional[queue.SimpleQueue] = None
        self._session = bool(_session)
        ttl_ms = int(env_first_int(("STNET_PREFETCH_GUARD_TTL_MS",), default=10) or 10)
        self._guard_ttl_s = max(0.0, float(ttl_ms) / 1000.0)
        self._join_timeout_s = 0.5
        with suppress(Exception):
            jt_ms = int(env_first_int(("STNET_THREAD_JOIN_TIMEOUT_MS",), default=500) or 500)
            self._join_timeout_s = max(0.0, float(jt_ms) / 1000.0)

    def _spawn_session(self) -> "Prefetcher":
        return Prefetcher(
            self._src,
            device=self._device,
            depth=int(self._depth),
            non_blocking=bool(self._non_blocking),
            memory_backpressure=bool(self._backpressure),
            gpu_guard_bytes=int(self._gpu_guard_bytes),
            host_guard_bytes=int(self._host_guard_bytes),
            pin_host=bool(self._pin),
            _session=True,
        )

    def _apply_structure(self, obj, func):
        if isinstance(obj, list):
            return [self._apply_structure(x, func) for x in obj]
        if isinstance(obj, tuple):
            return type(obj)(*(self._apply_structure(x, func) for x in obj))
        if isinstance(obj, dict):
            return {
                k: (v if k in ("row_ids", "keys") else self._apply_structure(v, func))
                for k, v in obj.items()
            }
        return func(obj)

    def _to_device(self, x: Any, device: torch.device) -> Any:
        def _f(t):
            if not torch.is_tensor(t) or t.device == device:
                return t
            nb = self._non_blocking and (
                t.device.type != "cpu" or (hasattr(t, "is_pinned") and t.is_pinned())
            )
            return t.to(device, non_blocking=nb)

        return self._apply_structure(x, _f)

    def _pin_memory(self, x: Any) -> Any:
        if not self._pin:
            return x

        def _f(t):
            if (
                torch.is_tensor(t)
                and t.device.type == "cpu"
                and not (hasattr(t, "is_pinned") and t.is_pinned())
            ):
                return t.pin_memory()
            return t

        return self._apply_structure(x, _f)

    def _stage_with_pool(self, obj: Any, pool: Pool, tokens: list[Optional[Pool.Token]]) -> Any:
        def _f(t):
            if not (torch.is_tensor(t) and t.device.type == "cpu") or (
                hasattr(t, "is_pinned") and t.is_pinned()
            ):
                return t
            buf, tok = pool.get_like(obj, return_handle=True, block=False)
            buf.copy_(t, non_blocking=False)
            tokens.append(tok)
            return buf

        return self._apply_structure(obj, _f)

    def _pin_batch(self, x: Any) -> tuple[Any, list[Optional[Pool.Token]]]:
        if not self._pin:
            return x, []
        pool = self._host_pool
        if pool is None:
            return self._pin_memory(x), []
        tokens: list[Optional[Pool.Token]] = []
        return self._stage_with_pool(x, pool, tokens), tokens

    def _producer_loop(
        self,
        it: Iterator[Any],
        sentinel: object,
        *args: Any,
        device: torch.device,
        use_device: bool,
        use_accel_stream: bool,
        gpu_guard_bytes: int,
        host_guard_bytes: int,
    ) -> None:
        last_check_t = 0.0
        last_guards_ok = True
        ttl_s = float(getattr(self, "_guard_ttl_s", 0.0) or 0.0)
        try:
            if use_device and isinstance(device, torch.device):
                backend = accelerator_type(device.type)
                set_dev = getattr(backend, "set_device", None) if backend is not None else None
                with suppress(Exception):
                    if callable(set_dev) and device.index is not None:
                        set_dev(int(device.index))
            while True:
                if self.is_stopped():
                    break
                if not self.block(timeout=None):
                    break
                try:
                    batch = next(it)
                except StopIteration:
                    break
                if self.is_stopped():
                    break
                batch, pool_tokens = self._pin_batch(batch)
                if self._backpressure:
                    sleep_s = 0.001
                    while not self.is_stopped():
                        now = time.monotonic()
                        if ttl_s <= 0.0 or (now - last_check_t) >= ttl_s:
                            last_check_t = now
                            host_ok = _host_guard_ok(host_guard_bytes)
                            dev_ok = _device_guard_ok(device, gpu_guard_bytes)
                            last_guards_ok = bool(host_ok and dev_ok)
                        if bool(last_guards_ok):
                            break
                        time.sleep(sleep_s)
                        sleep_s = min(float(sleep_s) * 2.0, 0.05)
                    if self.is_stopped():
                        if self._host_pool is not None and pool_tokens:
                            for tok in pool_tokens:
                                with suppress(Exception):
                                    self._host_pool.release(tok)
                        break
                if use_device:
                    if use_accel_stream and self._accel_stream is not None:
                        ev = None
                        pool = self._accel_event_pool
                        if pool is not None:
                            while ev is None and not self.is_stopped():
                                try:
                                    ev = pool.get(timeout=0.05)
                                except queue.Empty:
                                    continue
                        if ev is not None:
                            _wait_accel_event_done(ev, stopped=self.is_stopped)
                        try:
                            with accelerator_stream(self._accel_stream, device.type):
                                batch_dev = self._to_device(batch, device)
                                if ev is not None:
                                    try:
                                        ev.record(self._accel_stream)
                                    except TypeError:
                                        ev.record()
                            if self._host_pool is not None and pool_tokens:
                                if ev is not None:
                                    for tok in pool_tokens:
                                        self._host_pool.release_after(tok, ev)
                                else:
                                    for tok in pool_tokens:
                                        self._host_pool.release(tok)
                            if not self.put((batch_dev, ev), timeout=0.0):
                                if ev is not None and pool is not None:
                                    _wait_accel_event_done(ev, stopped=self.is_stopped)
                                    with suppress(Exception):
                                        pool.put(ev)
                                break
                        except BaseException:
                            if self._host_pool is not None and pool_tokens:
                                for tok in pool_tokens:
                                    with suppress(Exception):
                                        self._host_pool.release(tok)
                            if ev is not None and pool is not None:
                                _wait_accel_event_done(ev, stopped=self.is_stopped)
                                with suppress(Exception):
                                    pool.put(ev)
                            raise
                    else:
                        batch_dev = self._to_device(batch, device)
                        if not self.put((batch_dev, None), timeout=0.0):
                            break
                else:
                    if not self.put((batch, None), timeout=0.0):
                        break
        except BaseException as exc:
            with suppress(Exception):
                self.put(ProducerError(exc=exc, tb=traceback.format_exc()))
        finally:
            with suppress(Exception):
                self.put(sentinel)

    def __iter__(self) -> Iterator[Any]:
        if not bool(self._session):
            return iter(self._spawn_session())
        device = getattr(self, "_device", torch.device("cpu"))
        use_device = device.type in {"cuda", "mps", "xpu"}
        use_accel_stream = bool(self._non_blocking and is_stream_supported(device.type))
        iterable = getattr(self, "_iterable", self._src)
        gpu_guard_bytes = int(getattr(self, "_gpu_guard_bytes", 0) or 0)
        host_guard_bytes = int(getattr(self, "_host_guard_bytes", 0) or 0)
        if use_accel_stream and self._accel_stream is None:
            self._accel_stream = new_accelerator_stream(device)
        if use_accel_stream and self._accel_stream is None:
            use_accel_stream = False
        if use_accel_stream:
            pool: queue.SimpleQueue = queue.SimpleQueue()
            created = 0
            for _ in range(max(1, int(getattr(self, "_depth", 2) or 2))):
                ev = new_accelerator_event(device, enable_timing=False)
                if ev is not None:
                    pool.put(ev)
                    created += 1
            if created <= 0:
                use_accel_stream = False
                self._accel_event_pool = None
            else:
                self._accel_event_pool = pool
        else:
            self._accel_event_pool = None
        sentinel = object()
        it = iter(iterable)
        t = threading.Thread(
            target=self._producer_loop,
            name="PrefetcherProducer",
            daemon=True,
            args=(it, sentinel),
            kwargs={
                "device": device,
                "use_device": use_device,
                "use_accel_stream": use_accel_stream,
                "gpu_guard_bytes": gpu_guard_bytes,
                "host_guard_bytes": host_guard_bytes,
            },
        )
        t.start()
        try:
            while True:
                try:
                    item = self.get(timeout=None)
                except queue.Empty:
                    break
                if item is sentinel:
                    break
                if isinstance(item, ProducerError):
                    if isinstance(item.exc, (KeyboardInterrupt, SystemExit)):
                        raise item.exc
                    raise RuntimeError(
                        f"Prefetcher producer crashed: {item.exc}\n{item.tb}"
                    ) from item.exc
                batch, ev = item
                if use_accel_stream and ev is not None:
                    cs = current_accelerator_stream(device)
                    if cs is not None:
                        with suppress(Exception):
                            cs.wait_event(ev)
                        with suppress(Exception):
                            try:
                                ev.record(cs)
                            except TypeError:
                                ev.record()
                    pool = self._accel_event_pool
                    if pool is not None:
                        with suppress(Exception):
                            pool.put(ev)
                yield batch
        finally:
            self.stop()
            close(it)
            with suppress(Exception):
                if t.is_alive():
                    t.join(timeout=float(getattr(self, "_join_timeout_s", 0.5)))
            close(t)
            pool = self._accel_event_pool
            if use_accel_stream and pool is not None:
                while True:
                    try:
                        leftover = self.get(block=False)
                    except queue.Empty:
                        break
                    if leftover is sentinel or isinstance(leftover, ProducerError):
                        continue
                    if isinstance(leftover, tuple) and len(leftover) == 2:
                        _batch, ev = leftover
                        if ev is not None:
                            _wait_accel_event_done(ev, stopped=self.is_stopped)
                            with suppress(Exception):
                                pool.put(ev)
                while True:
                    try:
                        ev = pool.get_nowait()
                    except queue.Empty:
                        break
                    _wait_accel_event_done(ev, stopped=self.is_stopped)
                self._accel_event_pool = None
                self._accel_stream = None
            with suppress(Exception):
                self.clear()
