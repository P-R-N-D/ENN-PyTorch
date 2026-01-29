# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import concurrent.futures
import dataclasses
import glob
import importlib.util
import inspect
import json
import logging
import os
import pickle
import re
import sys
import tempfile
import shutil
import subprocess
import threading
import time
import types
import warnings
import weakref
from base64 import b64decode, b64encode
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Self, Sequence

import torch
from torch import nn

from ..core.concurrency import Mutex
from ..core.tensor import extract_tensor, from_buffer
from ..core.datatypes import PathLike, coerce_json, env_bool, save_temp, write_json
from .distributed import distributed_barrier, is_rank0

try:
    from torch.serialization import add_safe_globals
except ImportError:
    add_safe_globals = None



_IGNORED_WARNINGS = (
    "torch.distributed is disabled",
    "TypedStorage is deprecated",
)
_IGNORED_RE = r".*(?:" + "|".join(re.escape(s) for s in _IGNORED_WARNINGS) + r").*"
_SAVE_LOCK_GUARD = Mutex()
_SAVE_PATH_LOCKS = weakref.WeakValueDictionary()
_WARNINGS_FILTER_LOCK = Mutex()
_OPENZL_MAGIC = "enn-openzl-ckpt-v1"
_OPENZL_TENSOR_MARKER = "__ozl_tensor__"
_OPENZL_DICT_MARKER = "__ozl_dict__"
_OPENZL_TUPLE_MARKER = "__ozl_tuple__"
_OPENZL_PICKLE_MARKER = "__ozl_pickle__"
_OPENZL_COMPRESSOR: object | None = None
_OPENZL_FALLBACK_COMPRESSOR: object | None = None
_OPENZL_LOCK = Mutex()
_OPENZL_DEFAULT_BUILD_LOGGED = False
_OPENZL_FALLBACK_LOGGED = False
_LOGGER = logging.getLogger(__name__)


def _is_openzl_temporarily_unavailable(exc: BaseException) -> bool:
    msg = str(exc)
    return (
        "OpenZL error code: 81" in msg
        or "Temporary OpenZL library limitation" in msg
        or "temporary library limitation" in msg.lower()
    )


def _register_safe_globals() -> None:
    with contextlib.suppress(Exception):
        if add_safe_globals:
            from torch.torch_version import TorchVersion

            add_safe_globals([TorchVersion])


@contextlib.contextmanager
def _filtered_warnings(
    sentences: Sequence[str] | None = None,
) -> Iterator[None]:
    msg_re = (
        _IGNORED_RE
        if sentences is None
        else (
            r".*(?:" + "|".join(re.escape(str(s)) for s in sentences) + r").*"
            if sentences
            else ""
        )
    )
    if not msg_re:
        yield
        return
    guard = _WARNINGS_FILTER_LOCK
    if getattr(sys.flags, "context_aware_warnings", False):
        guard = contextlib.nullcontext()
    with guard:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=msg_re)
            yield


@contextlib.contextmanager
def _temp_environ(
    updates: dict[str, str | None], *args: Any, only_if_unset: bool = True
) -> Iterator[None]:
    prev: dict[str, str | None] = {}
    for key, val in updates.items():
        if only_if_unset and key in os.environ:
            continue
        prev[key] = os.environ.get(key)
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(val)
    try:
        yield
    finally:
        for key, val in prev.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


def _torch_dtype_from_str(dtype_str: str) -> torch.dtype:
    s = str(dtype_str)
    if s.startswith("torch."):
        s = s.split(".", 1)[1]
    dt = getattr(torch, s, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"Unsupported dtype string: {dtype_str!r}")
    return dt


def _save_lock(path: PathLike | None = None) -> Mutex:
    try:
        key = str(Path(path).expanduser().resolve()) if path else "__global__"
    except Exception:
        key = str(path)
    with _SAVE_LOCK_GUARD:
        return _SAVE_PATH_LOCKS.setdefault(key, Mutex(reentrant=True))


@contextlib.contextmanager
def _save_sync(
    path: PathLike | None = None, *args: Any, barrier: bool = False
) -> Iterator[None]:
    with _save_lock(path):
        if barrier:
            distributed_barrier()
        try:
            yield
        finally:
            if barrier:
                distributed_barrier()


def _load_model_config(model: object) -> object:
    try:
        from ..core.config import _extract_model_config_dict

        return _extract_model_config_dict(model)
    except Exception:
        return {}


def _coerce_dcp_keys(state: object) -> object:
    if isinstance(state, dict):
        keys_to_drop: list[object] = []
        for key, value in state.items():
            key_str = str(key)
            if (
                key_str.endswith("._extra_state")
                or key_str.endswith("_extra_state")
                or key_str.endswith("output_baked_flag")
            ):
                keys_to_drop.append(key)
                continue
            state[key] = _coerce_dcp_keys(value)
        for key in keys_to_drop:
            with contextlib.suppress(Exception):
                state.pop(key, None)
    return state


def _torch_load_checkpoint(
    path: PathLike,
    *args: Any,
    map_location: object = None,
    weights_only: bool = True,
    mmap: bool | None = None,
) -> object:
    load_args = args
    load_kwargs: dict[str, Any] = {"map_location": map_location or "cpu"}
    if mmap is not None:
        load_kwargs["mmap"] = bool(mmap)
    if weights_only is not None:
        load_kwargs["weights_only"] = bool(weights_only)
    try:
        return torch.load(str(path), *load_args, **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only", None)
        try:
            return torch.load(str(path), *load_args, **load_kwargs)
        except TypeError:
            load_kwargs.pop("mmap", None)
            return torch.load(str(path), *load_args, **load_kwargs)
    except Exception as exc:
        if weights_only:
            raise RuntimeError("weights_only=True failed") from exc
        raise


def _openzl_import() -> Any:
    is_required("openzl.ext", "pip install openzl")
    
    import openzl.ext as zl

    return zl


def _openzl_get_default_compressor() -> Any:
    global _OPENZL_COMPRESSOR
    cached = _OPENZL_COMPRESSOR
    if cached is not None:
        return cached
    with _OPENZL_LOCK:
        cached = _OPENZL_COMPRESSOR
        if cached is not None:
            return cached
        zl = _openzl_import()
        compressor = zl.Compressor()
        compress_graph = zl.graphs.Compress()
        entropy_graph = zl.graphs.Entropy()
        compress_id = compress_graph(compressor)
        entropy_id = entropy_graph(compressor)
        f16_graph: Any | None = None
        bf16_graph: Any | None = None
        f32_graph: Any | None = None
        with contextlib.suppress(Exception):
            f16_graph = zl.nodes.Float16Deconstruct()(
                compressor, sign_frac=compress_id, exponent=entropy_id
            )
        with contextlib.suppress(Exception):
            bf16_graph = zl.nodes.BFloat16Deconstruct()(
                compressor, sign_frac=compress_id, exponent=entropy_id
            )
        with contextlib.suppress(Exception):
            f32_graph = zl.nodes.Float32Deconstruct()(
                compressor, sign_frac=compress_id, exponent=entropy_id
            )
        f64_graph: Any | None = None
        with contextlib.suppress(Exception):
            f64_graph = zl.nodes.ConvertNumToStructLE()(
                compressor,
                successor=zl.nodes.TransposeSplit()(
                    compressor,
                    successor=compress_id,
                ),
            )
        int_fieldlz_id: Any = compress_id
        int_rangepack_fieldlz_id: Any | None = None
        int_zigzag_rangepack_fieldlz_id: Any | None = None
        with contextlib.suppress(Exception):
            int_fieldlz_id = zl.graphs.FieldLz()(compressor)
        with contextlib.suppress(Exception):
            int_rangepack_fieldlz_id = zl.nodes.RangePack()(
                compressor, successor=int_fieldlz_id
            )
        with contextlib.suppress(Exception):
            int_zigzag_rangepack_fieldlz_id = zl.nodes.Zigzag()(
                compressor, successor=(int_rangepack_fieldlz_id or int_fieldlz_id)
            )

        class _OpenZLGraph(zl.Selector):
            def __init__(
                self,
                *args: Any,
                serial_graph: Any,
                default_numeric_graph: Any,
                f16_graph: Any | None,
                bf16_graph: Any | None,
                f32_graph: Any | None,
                f64_graph: Any | None,
                int_fieldlz_graph: Any,
                int_rangepack_graph: Any | None,
                int_zigzag_rangepack_graph: Any | None,
            ) -> None:
                super().__init__()
                self._serial_graph = serial_graph
                self._default_numeric_graph = default_numeric_graph
                self._float_graphs: dict[torch.dtype, Any] = {}
                if f16_graph is not None:
                    self._float_graphs[torch.float16] = f16_graph
                if bf16_graph is not None:
                    self._float_graphs[torch.bfloat16] = bf16_graph
                if f32_graph is not None:
                    self._float_graphs[torch.float32] = f32_graph
                self._f64_graph = f64_graph
                self._int_fieldlz_graph = int_fieldlz_graph
                self._int_rangepack_graph = int_rangepack_graph
                self._int_zigzag_rangepack_graph = int_zigzag_rangepack_graph

            def selector_description(self) -> Any:
                return zl.SelectorDescription(
                    name="enn_ckpt_selector",
                    input_type_mask=zl.TypeMask.Serial | zl.TypeMask.Numeric,
                )

            @staticmethod
            def _min_width_bytes_unsigned(v: int) -> int:
                if v <= 0xFF:
                    return 1
                if v <= 0xFFFF:
                    return 2
                if v <= 0xFFFFFFFF:
                    return 4
                return 8

            def _pick_int_graph(self, t: torch.Tensor) -> Any:
                base = self._int_fieldlz_graph

                def _env_int(name: str, default: int) -> int:
                    with contextlib.suppress(Exception):
                        v = os.environ.get(name, "")
                        if v is None:
                            return default
                        v = v.strip()
                        if not v:
                            return default
                        return int(v)
                    return default

                try:
                    n = int(t.numel())
                except Exception:
                    return base
                min_n = _env_int("ENN_OPENZL_INT_AUTO_MIN_N", 128)
                if min_n < 0:
                    min_n = 0
                if n < min_n:
                    return base
                try:
                    flat = t.reshape(-1)
                    max_sample = _env_int("ENN_OPENZL_INT_AUTO_SAMPLE_N", 4096)
                    if max_sample <= 0:
                        max_sample = 4096
                    sample_n = max_sample if n > max_sample else n
                    sample = flat[:sample_n]
                    min_v = int(sample.min().item())
                    max_v = int(sample.max().item())
                except Exception:
                    return base
                try:
                    elem_bytes = int(t.element_size())
                except Exception:
                    elem_bytes = 0
                if elem_bytes <= 1:
                    return base
                has_neg = min_v < 0
                if has_neg and self._int_zigzag_rangepack_graph is not None:
                    absmax = max(abs(min_v), abs(max_v))
                    max_map = absmax * 2 + 1
                    need = self._min_width_bytes_unsigned(int(max_map))
                    if need < elem_bytes:
                        return self._int_zigzag_rangepack_graph
                if self._int_rangepack_graph is not None:
                    rng = max_v - min_v
                    if rng < 0:
                        rng = -rng
                    need = self._min_width_bytes_unsigned(int(rng))
                    if need < elem_bytes:
                        return self._int_rangepack_graph
                return base

            @staticmethod
            def _dominant_byte_ratio_u8(x: torch.Tensor) -> float:
                try:
                    n = int(x.numel())
                    if n <= 0:
                        return 0.0
                    bc = torch.bincount(x.to(torch.int64), minlength=256)
                    return float(bc.max().item()) / float(n)
                except Exception:
                    return 0.0

            def _pick_f64_graph(self, t: torch.Tensor) -> Any:
                g = self._f64_graph
                if g is None:
                    return self._default_numeric_graph

                def _env_int(name: str, default: int) -> int:
                    with contextlib.suppress(Exception):
                        v = os.environ.get(name, "")
                        if v is None:
                            return default
                        v = v.strip()
                        if not v:
                            return default
                        return int(v)
                    return default

                def _env_float(name: str, default: float) -> float:
                    with contextlib.suppress(Exception):
                        v = os.environ.get(name, "")
                        if v is None:
                            return default
                        v = v.strip()
                        if not v:
                            return default
                        return float(v)
                    return default

                strat = (
                    os.environ.get("ENN_OPENZL_FP64_STRATEGY", "auto").strip().lower()
                )
                if strat in ("compress", "generic", "off", "0", "false"):
                    return self._default_numeric_graph
                if strat in ("transpose", "on", "1", "true"):
                    return g
                try:
                    n = int(t.numel())
                except Exception:
                    return g
                min_n = _env_int("ENN_OPENZL_FP64_AUTO_MIN_N", 256)
                if min_n < 0:
                    min_n = 0
                if n < min_n:
                    return self._default_numeric_graph
                try:
                    flat = t.reshape(-1)
                    max_sample = _env_int("ENN_OPENZL_FP64_AUTO_SAMPLE_N", 4096)
                    if max_sample <= 0:
                        max_sample = 4096
                    sample_n = max_sample if n > max_sample else n
                    sample = flat[:sample_n].contiguous()
                    b = sample.view(torch.uint8).view(sample_n, 8)
                    lane7 = b[:, 7]
                    lane6 = b[:, 6]
                    u7 = int(torch.unique(lane7).numel())
                    u6 = int(torch.unique(lane6).numel())
                    u7_thr = _env_int("ENN_OPENZL_FP64_AUTO_UNIQUE_LANE7", 64)
                    u6_thr = _env_int("ENN_OPENZL_FP64_AUTO_UNIQUE_LANE6", 128)
                    if u7_thr < 1:
                        u7_thr = 1
                    if u6_thr < 1:
                        u6_thr = 1
                    if u7 <= u7_thr or u6 <= u6_thr:
                        return g
                    dom_thr = _env_float("ENN_OPENZL_FP64_AUTO_DOMINANT_RATIO", 0.55)
                    if dom_thr < 0.0:
                        dom_thr = 0.0
                    if dom_thr > 1.0:
                        dom_thr = 1.0
                    if self._dominant_byte_ratio_u8(lane7) >= dom_thr:
                        return g
                except Exception:
                    return self._default_numeric_graph
                return self._default_numeric_graph

            def select(self, state: Any, input: Any) -> Any:
                if input.type == zl.Type.Serial:
                    return self._serial_graph
                if input.type == zl.Type.Numeric:
                    t: torch.Tensor | None = None
                    with contextlib.suppress(Exception):
                        t = input.content.as_pytensor()
                    if t is not None:
                        dt = getattr(t, "dtype", None)
                        if isinstance(dt, torch.dtype):
                            if dt == torch.float64:
                                return self._pick_f64_graph(t)
                            g = self._float_graphs.get(dt)
                            if g is not None:
                                return g
                        if not torch.is_floating_point(t):
                            with contextlib.suppress(Exception):
                                if t.is_complex():
                                    return self._default_numeric_graph
                            if t.dtype == torch.bool:
                                return self._pick_int_graph(t)
                            try:
                                torch.iinfo(t.dtype)
                            except Exception:
                                return self._default_numeric_graph
                            return self._pick_int_graph(t)
                    return self._default_numeric_graph
                return self._default_numeric_graph

        selector_graph = compressor.register_selector_graph(
            _OpenZLGraph(
                serial_graph=compress_id,
                default_numeric_graph=compress_id,
                f16_graph=f16_graph,
                bf16_graph=bf16_graph,
                f32_graph=f32_graph,
                f64_graph=f64_graph,
                int_fieldlz_graph=int_fieldlz_id,
                int_rangepack_graph=int_rangepack_fieldlz_id,
                int_zigzag_rangepack_graph=int_zigzag_rangepack_fieldlz_id,
            )
        )
        compressor.select_starting_graph(selector_graph)
        _OPENZL_COMPRESSOR = compressor
        return compressor


def _openzl_compat_compressor() -> Any:
    global _OPENZL_FALLBACK_COMPRESSOR
    cached = _OPENZL_FALLBACK_COMPRESSOR
    if cached is not None:
        return cached
    with _OPENZL_LOCK:
        cached = _OPENZL_FALLBACK_COMPRESSOR
        if cached is not None:
            return cached
        zl = _openzl_import()
        compressor = zl.Compressor()
        compress_graph = zl.graphs.Compress()
        compress_id = compress_graph(compressor)
        compressor.select_starting_graph(compress_id)
        _OPENZL_FALLBACK_COMPRESSOR = compressor
        return compressor


def _openzl_jsonify(
    obj: object,
    *args: Any,
    tensors: list[torch.Tensor],
    tensor_table: list[dict[str, Any]],
) -> object:
    from ..core.tensor import coerce_tensor

    if torch.is_tensor(obj):
        t = coerce_tensor(obj)
        tid = len(tensors)
        tensors.append(t.reshape(-1))
        tensor_table.append(
            {
                "dtype": str(t.dtype),
                "shape": list(t.shape),
                "numel": int(t.numel()),
            }
        )
        return {_OPENZL_TENSOR_MARKER: tid}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        seq = [
            _openzl_jsonify(v, tensors=tensors, tensor_table=tensor_table) for v in obj
        ]
        if isinstance(obj, tuple):
            return {_OPENZL_TUPLE_MARKER: seq}
        return seq
    if isinstance(obj, dict):
        items = [
            [
                _openzl_jsonify(k, tensors=tensors, tensor_table=tensor_table),
                _openzl_jsonify(v, tensors=tensors, tensor_table=tensor_table),
            ]
            for k, v in obj.items()
        ]
        return {_OPENZL_DICT_MARKER: items}
    try:
        return coerce_json(obj)
    except Exception:
        try:
            return {_OPENZL_PICKLE_MARKER: b64encode(pickle.dumps(obj)).decode("ascii")}
        except Exception:
            return repr(obj)


def _openzl_unjsonify(
    obj: object,
    *args: Any,
    tensors: list[torch.Tensor],
    tensor_table: Sequence[Mapping[str, Any]],
    allow_pickle: bool = True,
) -> object:
    if isinstance(obj, list):
        return [
            _openzl_unjsonify(
                v, tensors=tensors, tensor_table=tensor_table, allow_pickle=allow_pickle
            )
            for v in obj
        ]
    if isinstance(obj, dict):
        if _OPENZL_TENSOR_MARKER in obj:
            tid = int(obj[_OPENZL_TENSOR_MARKER])
            entry = tensor_table[tid]
            shape = tuple(int(x) for x in (entry.get("shape") or ()))
            t = tensors[tid]
            return t.view(shape) if shape else t.reshape(())
        if _OPENZL_TUPLE_MARKER in obj:
            vals = obj.get(_OPENZL_TUPLE_MARKER) or []
            return tuple(
                _openzl_unjsonify(
                    v,
                    tensors=tensors,
                    tensor_table=tensor_table,
                    allow_pickle=allow_pickle,
                )
                for v in vals
            )
        if _OPENZL_DICT_MARKER in obj:
            out: dict[Any, Any] = {}
            pairs = obj.get(_OPENZL_DICT_MARKER) or []
            for k_raw, v_raw in pairs:
                k = _openzl_unjsonify(
                    k_raw,
                    tensors=tensors,
                    tensor_table=tensor_table,
                    allow_pickle=allow_pickle,
                )
                v = _openzl_unjsonify(
                    v_raw,
                    tensors=tensors,
                    tensor_table=tensor_table,
                    allow_pickle=allow_pickle,
                )
                if isinstance(k, list):
                    k = tuple(k)
                out[k] = v
            return out
        if _OPENZL_PICKLE_MARKER in obj:
            if not allow_pickle:
                raise RuntimeError(
                    "OpenZL pickle deserialization is disabled when weights_only=True."
                )
            try:
                return pickle.loads(b64decode(str(obj[_OPENZL_PICKLE_MARKER])))
            except Exception:
                return None
        return {
            k: _openzl_unjsonify(
                v, tensors=tensors, tensor_table=tensor_table, allow_pickle=allow_pickle
            )
            for k, v in obj.items()
        }
    return obj


def _openzl_pack_tensors_by_dtype(
    tensors: Sequence[torch.Tensor], tensor_table: list[dict[str, Any]]
) -> tuple[list[str], list[torch.Tensor]]:
    groups: dict[torch.dtype, list[tuple[int, torch.Tensor]]] = {}
    for tid, t in enumerate(tensors):
        groups.setdefault(t.dtype, []).append((tid, t))
    dtype_order = sorted(groups.keys(), key=lambda d: str(d))
    dtype_buffers: list[torch.Tensor] = []
    dtype_names: list[str] = []
    for buf_idx, dt in enumerate(dtype_order):
        items = groups[dt]
        total = int(sum(int(t.numel()) for _, t in items))
        buf = torch.empty((total,), dtype=dt, device="cpu")
        offset = 0
        for tid, t in items:
            n = int(t.numel())
            if n:
                buf[offset : offset + n].copy_(t, non_blocking=False)
            tensor_table[tid]["buffer"] = int(buf_idx)
            tensor_table[tid]["offset"] = int(offset)
            offset += n
        dtype_names.append(str(dt))
        dtype_buffers.append(buf)
    return dtype_names, dtype_buffers


def _openzl_compress_payload(
    payload: object,
    *args: Any,
    openzl_level: int | None = None,
    openzl_format_version: int | None = None,
    openzl_min_stream_size: int | None = None,
    openzl_content_checksum: bool | None = None,
    openzl_compressed_checksum: bool | None = None,
    openzl_permissive: bool | None = None,
    openzl_pack_by_dtype: bool = True,
) -> bytes:
    zl = _openzl_import()
    if openzl_permissive is None:
        openzl_permissive = not env_bool("ENN_OPENZL_STRICT", False)
    compat_only = env_bool("ENN_OPENZL_COMPAT_ONLY", False)
    no_compat_fallback = env_bool("ENN_OPENZL_NO_COMPAT_FALLBACK", False)
    compressors: list[Any] = []
    default_build_exc: Exception | None = None
    if not compat_only:
        try:
            compressors.append(_openzl_get_default_compressor())
        except Exception as exc:
            default_build_exc = exc
            if no_compat_fallback:
                raise
            global _OPENZL_DEFAULT_BUILD_LOGGED
            if not _OPENZL_DEFAULT_BUILD_LOGGED:
                _LOGGER.warning(
                    "OpenZL default compressor graph build failed; falling back to compat compressor. "
                    "Set ENN_OPENZL_STRICT=1 and ENN_OPENZL_NO_COMPAT_FALLBACK=1 to debug. Error: %s",
                    exc,
                )
                _OPENZL_DEFAULT_BUILD_LOGGED = True
    if compat_only or (not no_compat_fallback):
        with contextlib.suppress(Exception):
            fallback = _openzl_compat_compressor()
            if fallback is not None:
                if compat_only:
                    compressors = [fallback]
                elif fallback not in compressors:
                    compressors.append(fallback)
    if not compressors:
        if default_build_exc is not None:
            raise default_build_exc
        raise RuntimeError("OpenZL compressor initialization failed")
    tensors: list[torch.Tensor] = []
    tensor_table: list[dict[str, Any]] = []
    payload_meta = _openzl_jsonify(payload, tensors=tensors, tensor_table=tensor_table)
    if openzl_pack_by_dtype:
        dtype_names, dtype_buffers = _openzl_pack_tensors_by_dtype(
            tensors, tensor_table
        )
    else:
        dtype_names = [str(t.dtype) for t in tensors]
        dtype_buffers = list(tensors)
        for tid, t in enumerate(tensors):
            tensor_table[tid]["buffer"] = int(tid)
            tensor_table[tid]["offset"] = 0
    meta = {
        "magic": _OPENZL_MAGIC,
        "payload": payload_meta,
        "tensor_table": tensor_table,
        "dtype_buffers": dtype_names,
    }
    meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    inputs: list[Any] = [zl.Input(zl.Type.Serial, meta_bytes)]
    for buf in dtype_buffers:
        inputs.append(zl.Input(zl.Type.Numeric, buf))
    last_exc: Exception | None = None
    for idx, compressor in enumerate(compressors):
        cctx = zl.CCtx()
        cctx.ref_compressor(compressor)
        fmt = (
            int(openzl_format_version)
            if openzl_format_version is not None
            else int(getattr(zl, "MAX_FORMAT_VERSION", 0))
        )
        with contextlib.suppress(Exception):
            cctx.set_parameter(zl.CParam.FormatVersion, fmt)
        if openzl_level is not None:
            with contextlib.suppress(Exception):
                cctx.set_parameter(zl.CParam.CompressionLevel, int(openzl_level))
        if openzl_min_stream_size is not None:
            with contextlib.suppress(Exception):
                cctx.set_parameter(zl.CParam.MinStreamSize, int(openzl_min_stream_size))
        if openzl_content_checksum is not None:
            with contextlib.suppress(Exception):
                cctx.set_parameter(
                    zl.CParam.ContentChecksum, int(bool(openzl_content_checksum))
                )
        if openzl_compressed_checksum is not None:
            with contextlib.suppress(Exception):
                cctx.set_parameter(
                    zl.CParam.CompressedChecksum, int(bool(openzl_compressed_checksum))
                )
        with contextlib.suppress(Exception):
            cctx.set_parameter(
                zl.CParam.PermissiveCompression, int(bool(openzl_permissive))
            )
        try:
            out = bytes(cctx.compress(inputs))
            if idx > 0:
                global _OPENZL_FALLBACK_LOGGED
                if (
                    env_bool("ENN_OPENZL_LOG_FALLBACK", True)
                    and not _OPENZL_FALLBACK_LOGGED
                ):
                    _LOGGER.warning(
                        "OpenZL compression used compat fallback compressor due to prior failure: %s",
                        last_exc,
                    )
                    _OPENZL_FALLBACK_LOGGED = True
            return out
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("OpenZL compression failed without an error")


def _openzl_decompress_payload(
    blob: object,
    *args: Any,
    openzl_lazy_tensors: bool = True,
    openzl_check_content_checksum: bool | None = None,
    openzl_check_compressed_checksum: bool | None = None,
    weights_only: bool = True,
) -> object:
    zl = _openzl_import()
    dctx = zl.DCtx()
    if openzl_check_content_checksum is not None:
        with contextlib.suppress(Exception):
            dctx.set_parameter(
                zl.DParam.CheckContentChecksum,
                int(bool(openzl_check_content_checksum)),
            )
    if openzl_check_compressed_checksum is not None:
        with contextlib.suppress(Exception):
            dctx.set_parameter(
                zl.DParam.CheckCompressedChecksum,
                int(bool(openzl_check_compressed_checksum)),
            )
    outs = dctx.decompress(blob)
    if not outs:
        raise RuntimeError("OpenZL decompression returned no outputs")
    meta_bytes = outs[0].content.as_bytes()
    meta = json.loads(meta_bytes.decode("utf-8"))
    if meta.get("magic") != _OPENZL_MAGIC:
        raise ValueError("Not an ENN OpenZL checkpoint")
    tensor_table = meta.get("tensor_table") or []
    dtype_names = meta.get("dtype_buffers") or []
    dtype_bufs: list[torch.Tensor] = []
    for i, dt_name in enumerate(dtype_names):
        target = _torch_dtype_from_str(dt_name)
        out = outs[i + 1].content
        raw: torch.Tensor | None = None
        try:
            raw = out.as_pytensor()
        except Exception:
            raw = None
        if raw is not None and getattr(raw, "dtype", None) != target:
            with contextlib.suppress(Exception):
                raw = raw.view(target)
        if raw is None or getattr(raw, "dtype", None) != target:
            b = out.as_bytes()
            raw = torch.frombuffer(b, dtype=target)
        dtype_bufs.append(raw.reshape(-1))
    if not openzl_lazy_tensors:
        dtype_bufs = [b.clone() for b in dtype_bufs]
    tensors: list[torch.Tensor] = [torch.empty(0)] * int(len(tensor_table))
    for tid, entry in enumerate(tensor_table):
        dt_name = entry.get("dtype")
        shape = tuple(int(x) for x in (entry.get("shape") or ()))
        numel = int(entry.get("numel") or 0)
        buf_idx = int(entry.get("buffer") or 0)
        offset = int(entry.get("offset") or 0)
        if numel <= 0:
            tensors[tid] = torch.empty(shape, dtype=_torch_dtype_from_str(dt_name))
            continue
        buf = dtype_bufs[buf_idx]
        view = buf[offset : offset + numel]
        target = _torch_dtype_from_str(dt_name)
        if getattr(view, "dtype", None) != target:
            with contextlib.suppress(Exception):
                view = view.view(target)
        tensors[tid] = view.view(shape) if shape else view.reshape(())
    payload_meta = meta.get("payload")
    return _openzl_unjsonify(
        payload_meta,
        tensors=tensors,
        tensor_table=tensor_table,
        allow_pickle=not weights_only,
    )


def _openzl_load_checkpoint(
    path: PathLike,
    *args: Any,
    openzl_memmap: bool = True,
    openzl_lazy_tensors: bool = True,
    openzl_check_content_checksum: bool | None = None,
    openzl_check_compressed_checksum: bool | None = None,
    weights_only: bool = True,
) -> object:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    if not openzl_memmap:
        data = p.read_bytes()
        return _openzl_decompress_payload(
            data,
            openzl_lazy_tensors=openzl_lazy_tensors,
            openzl_check_content_checksum=openzl_check_content_checksum,
            openzl_check_compressed_checksum=openzl_check_compressed_checksum,
            weights_only=weights_only,
        )

    import mmap

    with p.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            mv = memoryview(mm)
            try:
                return _openzl_decompress_payload(
                    mv,
                    openzl_lazy_tensors=openzl_lazy_tensors,
                    openzl_check_content_checksum=openzl_check_content_checksum,
                    openzl_check_compressed_checksum=openzl_check_compressed_checksum,
                    weights_only=weights_only,
                )
            except TypeError:
                return _openzl_decompress_payload(
                    bytes(mv),
                    openzl_lazy_tensors=openzl_lazy_tensors,
                    openzl_check_content_checksum=openzl_check_content_checksum,
                    openzl_check_compressed_checksum=openzl_check_compressed_checksum,
                    weights_only=weights_only,
                )
        finally:
            with contextlib.suppress(Exception):
                mm.close()


def _openzl_save_checkpoint(
    path: PathLike,
    payload: object,
    *args: Any,
    openzl_level: int | None = None,
    openzl_format_version: int | None = None,
    openzl_min_stream_size: int | None = None,
    openzl_content_checksum: bool | None = None,
    openzl_compressed_checksum: bool | None = None,
    openzl_permissive: bool | None = None,
    openzl_pack_by_dtype: bool = True,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = _openzl_compress_payload(
        payload,
        openzl_level=openzl_level,
        openzl_format_version=openzl_format_version,
        openzl_min_stream_size=openzl_min_stream_size,
        openzl_content_checksum=openzl_content_checksum,
        openzl_compressed_checksum=openzl_compressed_checksum,
        openzl_permissive=openzl_permissive,
        openzl_pack_by_dtype=openzl_pack_by_dtype,
    )
    fd, tmp_name = tempfile.mkstemp(
        prefix=p.name + ".",
        suffix=p.suffix + ".tmp",
        dir=str(p.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("wb") as f:
            f.write(blob)
        tmp_path.replace(p)
    finally:
        with contextlib.suppress(Exception):
            tmp_path.unlink() if tmp_path.exists() else None


def is_required(module: str, pip_hint: str | None = None) -> None:
    try:
        __import__(module)
    except ImportError as err:
        hint = f" (try: {pip_hint})" if pip_hint else ""
        raise ImportError(f"{module} is required for this operation{hint}") from err


class Builder:
    NATIVE_EXTS = {".pt", ".pth", ".safetensors"}

    @staticmethod
    def is_target_native(path: PathLike) -> bool:
        suffix = Path(path).suffix.lower()
        return not suffix or suffix in Builder.NATIVE_EXTS

    @staticmethod
    def save(
        model: object,
        path: PathLike,
        optimizer: object | None = None,
        extra: object | None = None,
        state_dict: object | None = None,
        **opts: Any,
    ) -> object:
        p = Path(path)

        def _make_meta() -> dict[str, Any]:
            meta: dict[str, Any] = {
                "version": 1,
                "in_dim": int(getattr(model, "in_dim", 0)),
                "out_shape": tuple(int(x) for x in getattr(model, "out_shape", ())),
                "config": _load_model_config(model),
                "pytorch_version": torch.__version__,
                "extra": coerce_json(extra or {}),
            }
            with contextlib.suppress(Exception):
                task_specs = getattr(model, "task_specs", None)
                if callable(task_specs):
                    meta["tasks"] = task_specs()
            return meta

        if not p.suffix and p.exists() and p.is_dir():
            from torch.distributed.checkpoint import FileSystemWriter
            from torch.distributed.checkpoint import save as dcp_save
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
            )
            with _save_sync(p, barrier=True):
                dcp_save(
                    state_dict={
                        "model": get_model_state_dict(
                            model,
                            options=StateDictOptions(full_state_dict=True),
                        )
                    },
                    storage_writer=FileSystemWriter(str(p)),
                )
                if is_rank0():
                    write_json(
                        p / "meta.json",
                        {**_make_meta(), "format": "dcp-dir-v1"},
                        indent=2,
                    )
            return p
        if not p.suffix:
            p = p.with_suffix(".pt")
        p.parent.mkdir(parents=True, exist_ok=True)
        with _save_sync(p, barrier=False):
            if not is_rank0():
                return p
            sd = state_dict if state_dict is not None else model.state_dict()
            if p.suffix == ".ozl":
                raise ValueError(
                    "OpenZL checkpoint writing has been removed. Use .pt/.pth/.safetensors instead."
                )
            if p.suffix == ".safetensors":
                is_required("safetensors", "pip install safetensors")
                
                from safetensors.torch import save_file as save_tensors
                from ..core.tensor import coerce_tensor

                fd, tmp_name = tempfile.mkstemp(
                    prefix=p.name + ".",
                    suffix=p.suffix + ".tmp",
                    dir=str(p.parent),
                )
                os.close(fd)
                tmp_path = Path(tmp_name)
                try:
                    save_tensors(
                        {
                            k: coerce_tensor(v)
                            for k, v in (sd.items() if hasattr(sd, "items") else {})
                        },
                        str(tmp_path),
                        metadata={"format": "safetensors-v1"},
                    )
                    tmp_path.replace(p)
                finally:
                    with contextlib.suppress(Exception):
                        tmp_path.unlink() if tmp_path.exists() else None
                meta = _make_meta()
                write_json(p.with_name(p.name + ".json"), meta, indent=2)
                with contextlib.suppress(Exception):
                    legacy = p.with_suffix(".json")
                    if legacy != p.with_name(p.name + ".json"):
                        write_json(legacy, meta, indent=2)
                return p
            payload = {**_make_meta(), "state_dict": sd}
            if optimizer is not None:
                with contextlib.suppress(Exception):
                    payload["optimizer_state_dict"] = optimizer.state_dict()
            for key in list(opts):
                if str(key).startswith("openzl_"):
                    opts.pop(key)
            save_temp(p, payload, **opts)
            return p


@dataclasses.dataclass
class _PendingOp:
    kind: str
    epoch: int
    future: object
    started_monotonic: float


def _future_result(fut: object) -> object:
    if fut is None:
        return None
    result_fn = getattr(fut, "result", None)
    if callable(result_fn):
        return result_fn()
    wait_fn = getattr(fut, "wait", None)
    if callable(wait_fn):
        return wait_fn()
    return fut


def _future_done(fut: object) -> bool:
    if fut is None:
        return True
    done_fn = getattr(fut, "done", None)
    if callable(done_fn):
        try:
            return bool(done_fn())
        except Exception:
            return False
    return False


def _add_future_callback(fut: object, fn: Callable[[], None]) -> None:
    if fut is None:
        return
    then = getattr(fut, "then", None)
    if callable(then):
        with contextlib.suppress(Exception):
            then(lambda _: fn())
            return
    add_done = getattr(fut, "add_done_callback", None)
    if callable(add_done):
        with contextlib.suppress(Exception):
            add_done(lambda _: fn())


class Checkpointer:
    def __init__(
        self,
        ckpt_dir: PathLike,
        *,
        keep_last: int = 3,
        max_in_flight: int = 1,
        use_async: bool = True,
        dcp_subdir: str = "dcp_epochs",
        avg_subdir: str = "avg",
        avg_ext: str = ".pt",
        mmap_load: bool | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.root = Path(ckpt_dir)
        self.dcp_root = self.root / dcp_subdir
        self.avg_root = self.root / avg_subdir
        self.avg_ext = avg_ext if str(avg_ext).startswith(".") else f".{avg_ext}"
        self.keep_last = max(1, int(keep_last))
        self.max_in_flight = max(1, int(max_in_flight))
        self.use_async = bool(use_async)
        self.mmap_load = mmap_load

        self._pending_dcp: deque[_PendingOp] = deque()
        self._pending_avg: deque[_PendingOp] = deque()
        self._pending_lock = Mutex(reentrant=True)

        self._avg_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="enn-avg-ckpt"
        )

        self._stager: object | None = None
        self._stager_lock = Mutex(reentrant=True)
        self._stager_owner_thread: int | None = None
        self._stager_closed = False

        try:
            import torch.distributed as dist

            self._dist = dist
            self._rank = dist.get_rank() if dist.is_initialized() else 0
            self._world = dist.get_world_size() if dist.is_initialized() else 1
        except Exception:
            self._dist = None
            self._rank = 0
            self._world = 1

        self._local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
        self._local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1") or 1)
        if self._local_world < 1:
            self._local_world = 1
        self._node_rank = self._rank // self._local_world if self._local_world else 0

        self._dcp_process_group: object | None = None
        self._dcp_should_participate: bool = True
        if device is not None and self._is_distributed():
            try:
                from .distributed import get_distributed_mesh

                mesh, kind = get_distributed_mesh(device)
                if kind == "hsdp2" and mesh is not None:
                    coord = None
                    with contextlib.suppress(Exception):
                        coord = mesh.get_coordinate()
                    if isinstance(coord, tuple) and len(coord) >= 1:
                        self._dcp_should_participate = int(coord[0]) == 0

                    pg = None
                    with contextlib.suppress(Exception):
                        pg = mesh.get_group("dp_shard")
                    if pg is not None:
                        self._dcp_process_group = pg
            except Exception:
                self._dcp_process_group = None
                self._dcp_should_participate = True

        self.dcp_root.mkdir(parents=True, exist_ok=True)
        self.avg_root.mkdir(parents=True, exist_ok=True)

    def _is_distributed(self) -> bool:
        return bool(self._dist is not None and self._dist.is_initialized())

    def _is_global_rank0(self) -> bool:
        return (not self._is_distributed()) or self._rank == 0

    def _epoch_dir(self, epoch: int) -> Path:
        return self.dcp_root / f"epoch_{epoch:06d}"

    def _epoch_done_file(self, epoch_dir: Path) -> Path:
        return epoch_dir / "done.json"

    def _avg_node_dir(self, node_rank: int | None = None) -> Path:
        nr = self._node_rank if node_rank is None else int(node_rank)
        return self.avg_root / f"node_{nr:04d}"

    def _avg_epoch_path(self, epoch: int, node_rank: int | None = None) -> Path:
        d = self._avg_node_dir(node_rank)
        return d / f"epoch_{epoch:06d}{self.avg_ext}"

    def _avg_latest_file(self, node_rank: int | None = None) -> Path:
        return self._avg_node_dir(node_rank) / "latest.json"

    def _ensure_stager(self) -> object | None:
        if self._stager_closed:
            return None
        tid = threading.get_ident()
        with self._stager_lock:
            if self._stager is not None and self._stager_owner_thread == tid:
                return self._stager
            if self._stager is not None:
                with contextlib.suppress(Exception):
                    close = getattr(self._stager, "close", None)
                    if callable(close):
                        close()
                self._stager = None
                self._stager_owner_thread = None

            try:
                from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions

                opts = StagingOptions(
                    use_pinned_memory=True,
                    use_shared_memory=True,
                    use_async_staging=True,
                    use_non_blocking_copy=True,
                )
                self._stager = DefaultStager(config=opts)
                self._stager_owner_thread = tid
            except Exception:
                self._stager = None
                self._stager_owner_thread = None
            return self._stager

    def _maybe_wait_for_budget(self) -> None:
        with self._pending_lock:
            while self._pending_dcp and _future_done(self._pending_dcp[0].future):
                self._pending_dcp.popleft()
            while self._pending_avg and _future_done(self._pending_avg[0].future):
                self._pending_avg.popleft()

            while len(self._pending_dcp) >= self.max_in_flight:
                op = self._pending_dcp[0]
                fut = op.future
                self._pending_lock.release()
                try:
                    _future_result(fut)
                finally:
                    self._pending_lock.acquire()
                while self._pending_dcp and _future_done(self._pending_dcp[0].future):
                    self._pending_dcp.popleft()

            while len(self._pending_avg) >= 1:
                op = self._pending_avg[0]
                fut = op.future
                self._pending_lock.release()
                try:
                    _future_result(fut)
                finally:
                    self._pending_lock.acquire()
                while self._pending_avg and _future_done(self._pending_avg[0].future):
                    self._pending_avg.popleft()

    def _register_pending(self, kind: str, epoch: int, fut: object) -> None:
        with self._pending_lock:
            q = self._pending_dcp if kind == "dcp" else self._pending_avg
            q.append(
                _PendingOp(
                    kind=kind,
                    epoch=int(epoch),
                    future=fut,
                    started_monotonic=time.monotonic(),
                )
            )

    def _finalize_dcp_epoch(self, epoch: int, epoch_dir: Path, extra_meta: dict[str, Any]) -> None:
        if not self._is_global_rank0():
            return
        try:
            done_file = self._epoch_done_file(epoch_dir)
            payload = {
                "format": "enn-dcp-epoch-v1",
                "epoch": int(epoch),
                "created_time": time.time(),
                "rank": int(self._rank),
                "world_size": int(self._world),
                "node_rank": int(self._node_rank),
                **(extra_meta or {}),
            }
            write_json(done_file, payload, indent=2)
        finally:
            with contextlib.suppress(Exception):
                self._prune_dcp()

    def _prune_dcp(self) -> None:
        if not self._is_global_rank0():
            return
        try:
            epoch_dirs: list[Path] = []
            for p in self.dcp_root.glob("epoch_*"):
                if not p.is_dir():
                    continue
                if self._epoch_done_file(p).is_file():
                    epoch_dirs.append(p)
            epoch_dirs.sort(key=lambda x: x.name)
            if len(epoch_dirs) <= self.keep_last:
                return
            for p in epoch_dirs[: max(0, len(epoch_dirs) - self.keep_last)]:
                with contextlib.suppress(Exception):
                    shutil.rmtree(p)
        except Exception:
            return

    def _prune_avg(self, node_rank: int | None = None) -> None:
        d = self._avg_node_dir(node_rank)
        if not d.is_dir():
            return
        try:
            files = sorted(d.glob(f"epoch_*{self.avg_ext}"), key=lambda x: x.name)
            if len(files) <= self.keep_last:
                return
            for p in files[: max(0, len(files) - self.keep_last)]:
                with contextlib.suppress(Exception):
                    p.unlink()
        except Exception:
            return

    def request_save_epoch(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: object | None = None,
        avg_state_dict: Mapping[str, Any] | None = None,
        extra_state: dict[str, Any] | None = None,
        force_sync: bool = False,
    ) -> None:
        epoch_i = int(epoch)

        self._maybe_wait_for_budget()

        if avg_state_dict is not None and self._is_local_rank0():
            self._schedule_avg_save(epoch_i, avg_state_dict)

        if not self._dcp_should_participate:
            return

        dcp_future: object | None = None
        try:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint import FileSystemWriter
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

            epoch_dir = self._epoch_dir(epoch_i)
            epoch_dir.mkdir(parents=True, exist_ok=True)

            opts = StateDictOptions(full_state_dict=False)
            model_sd, optim_sd = get_state_dict(model, optimizer or [], options=opts)
            dcp_state: dict[str, Any] = {"model": _coerce_dcp_keys(model_sd)}
            if optimizer is not None:
                dcp_state["optimizer"] = _coerce_dcp_keys(optim_sd)
            if extra_state is not None:
                dcp_state["extra"] = _coerce_dcp_keys(extra_state)

            writer = FileSystemWriter(str(epoch_dir))
            planner: object | None = None
            with contextlib.suppress(Exception):
                from torch.distributed.checkpoint import DefaultSavePlanner

                planner = DefaultSavePlanner(
                    dedup_replicated_tensors=True, dedup_save_to_lowest_rank=True
                )

            if self.use_async and hasattr(dcp, "async_save"):
                stager = self._ensure_stager()
                kwargs: dict[str, Any] = {
                    "state_dict": dcp_state,
                    "checkpoint_id": str(epoch_dir),
                    "storage_writer": writer,
                    "planner": planner,
                    "process_group": self._dcp_process_group,
                }
                if stager is not None:
                    kwargs["async_stager"] = stager

                with contextlib.suppress(Exception):
                    from torch.distributed.checkpoint.state_dict_saver import (
                        AsyncCheckpointerType,
                    )

                    async_type = (
                        os.environ.get("ENN_DCP_ASYNC_TYPE", "thread").strip().lower()
                    )
                    if async_type == "process":
                        kwargs["async_checkpointer_type"] = AsyncCheckpointerType.PROCESS
                    elif async_type == "thread":
                        kwargs["async_checkpointer_type"] = AsyncCheckpointerType.THREAD

                try:
                    sig = inspect.signature(dcp.async_save)
                    supported = set(sig.parameters.keys())
                    kwargs = {
                        k: v for k, v in kwargs.items() if k in supported and v is not None
                    }
                except Exception:
                    kwargs = {k: v for k, v in kwargs.items() if v is not None}

                dcp_future = dcp.async_save(**kwargs)
            else:
                dcp.save(
                    state_dict=dcp_state,
                    checkpoint_id=str(epoch_dir),
                    storage_writer=writer,
                    planner=planner,
                    process_group=self._dcp_process_group,
                )
                dcp_future = None

            if dcp_future is not None:
                if self._is_global_rank0():
                    _add_future_callback(
                        dcp_future,
                        lambda: self._finalize_dcp_epoch(
                            epoch_i,
                            epoch_dir,
                            extra_meta={"has_optimizer": optimizer is not None},
                        ),
                    )
                self._register_pending("dcp", epoch_i, dcp_future)
            else:
                self._finalize_dcp_epoch(
                    epoch_i,
                    epoch_dir,
                    extra_meta={"has_optimizer": optimizer is not None},
                )
        except Exception as exc:
            _LOGGER.exception("DCP epoch checkpoint failed: %s", exc)
            if force_sync:
                raise

    def wait(self) -> None:
        pending: list[_PendingOp]
        with self._pending_lock:
            pending = list(self._pending_dcp) + list(self._pending_avg)
        for op in pending:
            with contextlib.suppress(Exception):
                _future_result(op.future)
        with self._pending_lock:
            self._pending_dcp.clear()
            self._pending_avg.clear()

    def close(self) -> None:
        self.wait()
        with contextlib.suppress(Exception):
            import torch.distributed.checkpoint.state_dict_saver as sds

            close_fn = getattr(sds, "close", None)
            if callable(close_fn):
                close_fn()
        with self._stager_lock:
            self._stager_closed = True
            if self._stager is not None:
                with contextlib.suppress(Exception):
                    close = getattr(self._stager, "close", None)
                    if callable(close):
                        close()
                self._stager = None
                self._stager_owner_thread = None
        with contextlib.suppress(Exception):
            self._avg_executor.shutdown(wait=True)

    def find_latest_dcp_epoch(self) -> int | None:
        try:
            completed: list[int] = []
            for p in self.dcp_root.glob("epoch_*"):
                if not p.is_dir():
                    continue
                if self._epoch_done_file(p).is_file():
                    m = re.match(r"epoch_(\\d+)", p.name)
                    if m:
                        completed.append(int(m.group(1)))
            return max(completed) if completed else None
        except Exception:
            return None

    def load_latest_dcp(
        self,
        *,
        model: nn.Module,
        optimizer: object | None = None,
        strict: bool = False,
    ) -> int | None:
        latest = self.find_latest_dcp_epoch()
        if latest is None:
            return None
        epoch_dir = self._epoch_dir(latest)
        if not self._epoch_done_file(epoch_dir).is_file():
            return None
        try:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint import FileSystemReader
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
                get_optimizer_state_dict,
                set_model_state_dict,
                set_optimizer_state_dict,
            )

            opts = StateDictOptions(full_state_dict=False, strict=bool(strict))
            model_sd = get_model_state_dict(model, options=opts)
            state: dict[str, Any] = {"model": model_sd}
            optim_sd: Any | None = None
            if optimizer is not None:
                optim_sd = get_optimizer_state_dict(model, optimizer, options=opts)
                state["optimizer"] = optim_sd
            dcp.load(state_dict=state, storage_reader=FileSystemReader(str(epoch_dir)))
            set_model_state_dict(model, model_sd, options=opts)
            if optimizer is not None and optim_sd is not None:
                set_optimizer_state_dict(model, optimizer, optim_sd, options=opts)
            return latest
        except Exception as exc:
            _LOGGER.exception("DCP load failed: %s", exc)
            if strict:
                raise
            return None

    def load_model_from_torchsave_broadcast(
        self,
        *,
        model: nn.Module,
        torch_save_path: PathLike,
        strict: bool = False,
    ) -> None:
        try:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.format_utils import (
                BroadcastingTorchSaveReader,
                DynamicMetaLoadPlanner,
            )
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
                set_model_state_dict,
            )

            opts = StateDictOptions(full_state_dict=False, strict=bool(strict))
            model_sd = get_model_state_dict(model, options=opts)
            dcp.load(
                state_dict={"model": model_sd},
                storage_reader=BroadcastingTorchSaveReader(),
                planner=DynamicMetaLoadPlanner(),
                checkpoint_id=str(torch_save_path),
            )
            set_model_state_dict(model, model_sd, options=opts)
        except Exception as exc:
            _LOGGER.exception("Broadcasting TorchSave load failed: %s", exc)
            if strict:
                raise



class Exporter:
    _by_name: dict[str, Format] = {}
    _ext_map: dict[str, str] = {}
    _defaults_registered: bool = False
    _defaults_lock = Mutex()
    _ONNXExporter: Any = None
    _ORTBuilder: Any = None
    _export_sig_cache: object | None = None
    _export_sig_lock = Mutex()

    @classmethod
    def _export_sig(cls: type[Self]) -> object:
        cached = getattr(cls, "_export_sig_cache", None)
        if cached is not None:
            return cached
        with cls._export_sig_lock:
            cached = getattr(cls, "_export_sig_cache", None)
            if cached is not None:
                return cached
            try:
                sig = inspect.signature(torch.onnx.export)
            except Exception:
                sig = None
            cls._export_sig_cache = sig
            return sig

    @classmethod
    def _export_sig_keys(cls: type[Self]) -> set[str]:
        sig = cls._export_sig()
        if sig is None:
            return set()
        try:
            params = getattr(sig, "parameters", None)
            if isinstance(params, dict):
                return set(params.keys())
        except Exception:
            pass
        try:
            return set(sig.parameters.keys())
        except Exception:
            return set()

    @classmethod
    def register(
        cls: type[Self], name: str, exts: tuple[str, ...], impl: Format
    ) -> None:
        with cls._defaults_lock:
            cls._register_unlocked(name, exts, impl)

    @classmethod
    def _register_unlocked(
        cls: type[Self], name: str, exts: tuple[str, ...], impl: Format
    ) -> None:
        cls._by_name[name] = impl
        for ext in exts:
            cls._ext_map[ext.lower()] = name

    @classmethod
    def _ensure_defaults_registered(cls: type[Self]) -> None:
        if cls._defaults_registered:
            return
        with cls._defaults_lock:
            if cls._defaults_registered:
                return
            cls._ONNXExporter = _ONNXExporter
            cls._ORTBuilder = _ORTBuilder
            cls._register_unlocked("onnx", (".onnx",), ONNX())
            cls._register_unlocked("ort", (".ort",), ORT())
            cls._register_unlocked("tensorrt", (".engine", ".plan"), TensorRT())
            cls._register_unlocked("coreml", (".mlmodel", ".mlpackage"), CoreML())
            cls._register_unlocked("litert", (".tflite",), LiteRT())
            cls._register_unlocked("pt2", (".pt2", ".export"), TorchExport())
            cls._register_unlocked("aoti", (".aoti",), TorchInductor())
            cls._register_unlocked("executorch", (".pte",), ExecuTorch())
            cls._register_unlocked("tensorflow", (".savedmodel", ".pb", ".tf"), TensorFlow())
            cls._defaults_registered = True

    @classmethod
    def for_export(cls: type[Self], ext: str) -> Format | None:
        cls._ensure_defaults_registered()
        with cls._defaults_lock:
            name = cls._ext_map.get(ext.lower())
            return cls._by_name.get(name) if name else None


_EXPORT_SIG_CACHE: object | None = None
_EXPORT_SIG_LOCK = Mutex()
_EXPORT_WARN_FILTERS_INSTALLED = False
_FORWARD_PARAM_CACHE: dict[object, object] = {}
_FORWARD_PARAM_CACHE_LOCK = Mutex()
_ONNX2TF_HELP_CACHE: str | None = None
_ONNX2TF_HELP_LOCK = Mutex(reentrant=True)


def _export_strip_slots_enabled() -> bool:
    v = os.environ.get("ENN_EXPORT_STRIP_SLOTS", "1").strip().lower()
    return v not in ("0", "false", "off", "no", "n")


def _gil_disabled() -> bool:
    fn = getattr(sys, "_is_gil_enabled", None)
    if callable(fn):
        try:
            return not bool(fn())
        except Exception:
            return False
    return False


def _export_strip_locks_enabled() -> bool:
    v = os.environ.get("ENN_EXPORT_STRIP_LOCKS", "").strip().lower()
    if not v:
        return not _gil_disabled()
    return v not in ("0", "false", "off", "no", "n")


def _is_lock_like(v: object) -> bool:
    if isinstance(v, (Mutex,)):
        return True
    try:
        tn = type(v).__name__.lower()
        tm = type(v).__module__
        if tm in ("_thread", "threading") and "lock" in tn:
            return True
    except Exception:
        pass
    return False


def _is_export_problem_attr(v: object) -> bool:
    if v is None:
        return False
    if isinstance(v, (torch.Tensor, nn.Parameter, nn.Module)):
        return False
    if isinstance(v, (str, bytes, int, float, bool)):
        return False
    if isinstance(v, (torch.dtype, torch.device)):
        return False
    if isinstance(
        v,
        (types.FunctionType, types.BuiltinFunctionType, types.MethodType),
    ):
        return False
    if _is_lock_like(v):
        return True
    if isinstance(
        v,
        (
            weakref.ReferenceType,
            weakref.WeakSet,
            weakref.WeakKeyDictionary,
            weakref.WeakValueDictionary,
        ),
    ):
        return True
    if _export_strip_slots_enabled():
        try:
            t = type(v)
            if getattr(t, "__module__", "").startswith("torch"):
                return False
            slots = getattr(t, "__slots__", None)
            if slots:
                return True
        except Exception:
            pass
    return False


@contextlib.contextmanager
def _strip_for_export(model: object) -> Iterator[None]:
    if not isinstance(model, nn.Module):
        yield
        return
    removed: list[tuple[object, str, object]] = []
    try:
        allow_strip_locks = _export_strip_locks_enabled()

        def _lock_guard_enabled(mod: nn.Module, lock_attr: str) -> bool:
            if not lock_attr.endswith("_lock"):
                return False
            guard = lock_attr[: -len("_lock")] + "_use_lock"
            with contextlib.suppress(Exception):
                return bool(getattr(mod, guard))
            return False

        for module in model.modules():
            d = getattr(module, "__dict__", None)
            if not isinstance(d, dict) or not d:
                continue
            for k, v in list(d.items()):
                if k in ("_modules", "_parameters", "_buffers"):
                    continue
                if _is_lock_like(v):
                    if (not allow_strip_locks) or _lock_guard_enabled(module, k):
                        continue
                if _is_export_problem_attr(v):
                    with contextlib.suppress(Exception):
                        removed.append((module, k, v))
                        delattr(module, k)
        yield
    finally:
        for obj, k, v in reversed(removed):
            with contextlib.suppress(Exception):
                setattr(obj, k, v)


@contextlib.contextmanager
def _no_empty_tensor(root: nn.Module) -> Iterator[None]:
    patched: list[tuple[nn.Module, str, torch.Tensor]] = []
    try:
        for module in root.modules():
            for name in ("pw_x", "pw_y"):
                tensor = getattr(module, name, None)
                if isinstance(tensor, torch.Tensor) and tensor.numel() == 0:
                    patched.append((module, name, tensor))
                    placeholder = tensor.detach().new_zeros((1, 1))
                    setattr(module, name, placeholder)
        yield
    finally:
        for module, name, old in patched:
            with contextlib.suppress(Exception):
                setattr(module, name, old)


def _suppress_export_warnings() -> None:
    global _EXPORT_WARN_FILTERS_INSTALLED
    if _EXPORT_WARN_FILTERS_INSTALLED:
        return
    _EXPORT_WARN_FILTERS_INSTALLED = True
    warnings.filterwarnings(
        "ignore",
        message=r".*Converting a tensor to a Python boolean.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*LeafSpec.*deprecated.*",
    )
    with contextlib.suppress(Exception):
        tw = getattr(torch.jit, "TracerWarning", None)
        if tw is not None:
            warnings.filterwarnings(
                "ignore",
                category=tw,
                message=r".*Converting a tensor to a Python boolean.*",
            )


@contextlib.contextmanager
def _onnx_model(model: object) -> Iterator[object]:
    _suppress_export_warnings()
    was_training = getattr(model, "training", False)
    removed_top, removed_sub = {}, []
    model.eval()
    for name in (
        "optimizer",
        "optimizer_state",
        "optim",
        "training_history",
        "history",
        "logger",
        "metrics",
        "_training_history",
    ):
        if hasattr(model, name):
            with contextlib.suppress(Exception):
                removed_top[name] = getattr(model, name)
                delattr(model, name)
    RecorderCls = None
    with contextlib.suppress(Exception):
        from ..nn.layers import Recorder as _Recorder
        RecorderCls = _Recorder
    if RecorderCls is not None:
        with contextlib.suppress(Exception):
            for module in model.modules():
                for attr in ("logger", "history"):
                    if not hasattr(module, attr):
                        continue
                    with contextlib.suppress(Exception):
                        v = getattr(module, attr)
                        if isinstance(v, RecorderCls):
                            removed_sub.append((module, attr, v))
                            delattr(module, attr)
    try:
        with _temp_environ(
            {
                "ENN_MSR_FORCE_TORCH": "1",
                "ENN_DISABLE_FLEX_ATTENTION": "1",
                "ENN_DISABLE_SDPA": "1",
                "ENN_DISABLE_PIECEWISE_CALIB": "1",
            },
            only_if_unset=True,
        ):
            eager_ctx = getattr(model, "eager_for_export", None)
            with eager_ctx() if callable(eager_ctx) else contextlib.nullcontext():
                yield model
    finally:
        for module, attr, v in removed_sub:
            with contextlib.suppress(Exception):
                setattr(module, attr, v)
        for name, v in removed_top.items():
            with contextlib.suppress(Exception):
                setattr(model, name, v)
        if was_training:
            with contextlib.suppress(Exception):
                model.train(True)


def _get_forward_parameters(model_cls: object) -> object:
    with _FORWARD_PARAM_CACHE_LOCK:
        cached = _FORWARD_PARAM_CACHE.get(model_cls)
        if cached is not None:
            return cached
    try:
        sig = inspect.signature(model_cls.forward)
    except Exception:
        return None
    info = (
        set(sig.parameters.keys()),
        any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()),
    )
    with _FORWARD_PARAM_CACHE_LOCK:
        if len(_FORWARD_PARAM_CACHE) > 512:
            _FORWARD_PARAM_CACHE.clear()
        return _FORWARD_PARAM_CACHE.setdefault(model_cls, info)


def _forward(model: object, x: object) -> object:
    fwd_export = getattr(model, "forward_export", None)
    if callable(fwd_export) and isinstance(x, torch.Tensor):
        return fwd_export(x)
    info = _get_forward_parameters(type(model))
    if info is None:
        return model(x)
    names, accepts_kwargs = info
    kwargs = {}
    for key in ("labels_flat", "net_loss"):
        if accepts_kwargs or key in names:
            kwargs[key] = None
    return model(x, **kwargs) if kwargs else model(x)


def _get_tensor_shape(model: object, sample_input: object) -> object:
    in_dim = int(getattr(model, "in_dim")) if hasattr(model, "in_dim") else None
    out_shape = (
        tuple(int(x) for x in getattr(model, "out_shape"))
        if hasattr(model, "out_shape")
        else None
    )
    if in_dim is not None and out_shape is not None:
        return (int(in_dim), tuple(int(x) for x in out_shape))
    if sample_input is not None:
        if not isinstance(sample_input, torch.Tensor):
            raise TypeError("sample_input must be a torch.Tensor")
        dev = next(
            (p.device for p in model.parameters() if p is not None),
            torch.device("cpu"),
        )
        sample = sample_input.to(dev)
        model.eval()
        with torch.no_grad():
            sample_batched = sample.unsqueeze(0) if sample.ndim == 1 else sample
            y_flat = extract_tensor(_forward(model, sample_batched))
        if in_dim is None:
            in_dim = (
                int(sample.numel())
                if sample.ndim == 1
                else int(sample.numel() // sample.shape[0])
            )
        if out_shape is None:
            out_shape = tuple(int(x) for x in tuple(y_flat.shape[1:]))
    if in_dim is None or out_shape is None:
        raise RuntimeError("Failed to infer shapes.")
    return (int(in_dim), tuple(int(x) for x in out_shape))


def _pad_sample(
    model: object, sample_input: object, *args: Any, batch: int = 1
) -> object:
    if sample_input is not None:
        return sample_input
    in_dim, _ = _get_tensor_shape(model, sample_input)
    param = next(model.parameters(), None)
    dtype, device = (
        (param.dtype, param.device)
        if param is not None
        else (torch.float32, torch.device("cpu"))
    )
    b = max(1, int(batch))
    return torch.zeros(b, in_dim, dtype=dtype, device=device)


def _onnx_options(kwargs: object, *args: Any, target: str = "onnx") -> object:
    target_l = str(target or "onnx").strip().lower()
    defaults = {
        "tensorrt": (18, True, True, True, []),
        "tensorflow": (18, False, True, True, []),
        "default": (18, False, True, False, []),
    }
    key = target_l.replace("-", "").replace("_", "")
    if key == "trt":
        key = "tensorrt"
    d_opset, d_dyn, d_pref, d_simp, d_fb = defaults.get(
        target_l, defaults.get(key, defaults["default"])
    )
    opset = int(kwargs.get("opset_version", d_opset))
    fb = kwargs.get("opset_fallback", d_fb)
    if not fb:
        fallback: list[int] = []
    elif isinstance(fb, str):
        fallback = [int(x) for x in re.split(r"[\s,]+", fb) if x]
    elif isinstance(fb, (list, tuple)):
        fallback = [int(x) for x in fb]
    else:
        fallback = [int(fb)]
    clean_fb: list[int] = []
    for v in fallback:
        iv = int(v)
        if iv == int(opset):
            continue
        if iv not in clean_fb:
            clean_fb.append(iv)
    return {
        "sample_input": kwargs.get("sample_input"),
        "opset_version": opset,
        "opset_fallback": clean_fb,
        "dynamic_batch": kwargs.get("dynamic_batch", d_dyn),
        "prefer_dynamo": kwargs.get("prefer_dynamo", kwargs.get("dynamo", d_pref)),
        "simplify": kwargs.get("simplify_onnx", kwargs.get("onnx_simplify", d_simp)),
        "optimize_onnx": kwargs.get("optimize_onnx", kwargs.get("onnx_optimize", True)),
        "onnxoptimizer_passes": kwargs.get(
            "onnxoptimizer_passes", kwargs.get("onnx_optimizer_passes")
        ),
    }


def _coerce_onnx_path(dst: PathLike, kwargs: object) -> object:
    dst_p = Path(dst)
    onnx_override = None
    with contextlib.suppress(Exception):
        if isinstance(kwargs, dict):
            onnx_override = kwargs.get("onnx_path")
        else:
            onnx_override = getattr(kwargs, "onnx_path", None)
    if onnx_override:
        return Path(onnx_override)
    if dst_p.suffix.lower() == ".onnx":
        return dst_p
    return dst_p.with_name(dst_p.name + ".onnx")


def _sidecar_json_path(dst: PathLike) -> Path:
    p = Path(dst)
    return p.with_name(p.name + ".json")


def _write_export_meta(
    model: nn.Module,
    dst: PathLike,
    *args: Any,
    format_name: str,
    extra: dict[str, Any] | None = None,
) -> None:
    p = Path(dst)
    out_shape = getattr(model, "out_shape", None) or ()
    if not isinstance(out_shape, (list, tuple)):
        out_shape = (out_shape,)
    payload: dict[str, Any] = {
        "version": 1,
        "format": str(format_name),
        "in_dim": int(getattr(model, "in_dim", 0) or 0),
        "out_shape": tuple(int(x) for x in out_shape),
        "config": _load_model_config(model),
        "pytorch_version": torch.__version__,
    }
    if extra:
        payload["extra"] = extra
    write_json(_sidecar_json_path(p), payload, indent=2)


def _pad_to_batch(sample: torch.Tensor, min_batch: int) -> torch.Tensor:
    if not isinstance(sample, torch.Tensor):
        return sample
    if sample.ndim == 0:
        return sample
    b = int(sample.shape[0]) if sample.shape else 1
    if b >= int(min_batch):
        return sample
    pad_shape = (int(min_batch) - b,) + tuple(sample.shape[1:])
    pad = torch.zeros(pad_shape, dtype=sample.dtype, device=sample.device)
    return torch.cat([sample, pad], dim=0)


def _onnx2tf_help_text() -> str:
    global _ONNX2TF_HELP_CACHE
    cached = _ONNX2TF_HELP_CACHE
    if cached is not None:
        return cached
    with _ONNX2TF_HELP_LOCK:
        cached = _ONNX2TF_HELP_CACHE
        if cached is not None:
            return cached
        try:
            out = subprocess.check_output(
                [sys.executable, "-m", "onnx2tf", "-h"],
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception:
            out = ""
        _ONNX2TF_HELP_CACHE = out
        return out


def _onnx2tf_supports(flag: str) -> bool:
    try:
        return flag in _onnx2tf_help_text()
    except Exception:
        return False


def _find_latest_onnx2tf_auto_json(out_dir: Path) -> Path | None:
    try:
        candidates = list(Path(out_dir).rglob("*_auto.json"))
    except Exception:
        return None
    if not candidates:
        return None
    try:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        pass
    return candidates[0]


def _run_onnx2tf(
    onnx_path: Path,
    out_dir: Path,
    *extra_args: str,
    dynamic_batch: bool = True,
) -> None:
    onnx_path = onnx_path.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base_extra = [str(x) for x in extra_args if str(x)]

    def _cmd(*more: str) -> list[str]:
        cmd: list[str] = [
            sys.executable,
            "-m",
            "onnx2tf",
            "-i",
            str(onnx_path),
            "-o",
            str(out_dir),
            "-nuo",
        ]
        cmd.extend(base_extra)
        cmd.extend([str(x) for x in more if str(x)])
        return cmd

    def _try(*more: str) -> bool:
        try:
            _in_console(_cmd(*more), "onnx2tf", cwd=str(out_dir))
            return True
        except RuntimeError:
            return False

    if _try():
        return

    auto_json = _find_latest_onnx2tf_auto_json(out_dir)
    agj_flag: str | None = None
    for cand in ("-agj", "--auto_generate_json"):
        if _onnx2tf_supports(cand):
            agj_flag = cand
            break
    if agj_flag is not None:
        _try(agj_flag)
        auto_json = _find_latest_onnx2tf_auto_json(out_dir)
    if _onnx2tf_supports("-prf") and auto_json is not None:
        if _try("-prf", str(auto_json)):
            return
    if dynamic_batch and _onnx2tf_supports("-b"):
        if _try("-b", "1"):
            return
        if _onnx2tf_supports("-prf") and auto_json is not None:
            if _try("-b", "1", "-prf", str(auto_json)):
                return
        if agj_flag is not None:
            _try("-b", "1", agj_flag)
            auto_json = _find_latest_onnx2tf_auto_json(out_dir)
            if _onnx2tf_supports("-prf") and auto_json is not None:
                if _try("-b", "1", "-prf", str(auto_json)):
                    return
    raise RuntimeError(
        "onnx2tf conversion failed after multiple retries; "
        "consider running onnx2tf manually to inspect the failing op"
    )


def _torch_export_program(
    torch_export: Callable[..., Any],
    wrapper: nn.Module,
    sample: torch.Tensor,
    *args: Any,
    dynamic_batch: bool = True,
    dynamic_seq: bool = False,
    strict: bool = True,
    tag: str = "torch.export",
) -> object:
    with contextlib.suppress(Exception):
        import torch._functorch.predispatch as _pd

        _pd.lazy_load_decompositions()
    if isinstance(sample, torch.Tensor) and sample.ndim == 1:
        sample = sample.unsqueeze(0)
    if dynamic_batch and isinstance(sample, torch.Tensor) and sample.ndim >= 1:
        mode = os.environ.get("ENN_EXPORT_BATCH_DIM", "dynamic").strip().lower()
        if mode in {"0", "false", "off", "none"}:
            dynamic_batch = False
        elif mode not in {"static", "fixed"}:
            sample = _pad_to_batch(sample, 2)
    dynamic_shapes = None
    if (
        (dynamic_batch or dynamic_seq)
        and hasattr(torch, "export")
        and hasattr(torch.export, "Dim")
    ):
        spec: dict[int, object] = {}
        Dim = torch.export.Dim

        def _dim_from_mode(name: str, env_key: str) -> object | None:
            mode = os.environ.get(env_key, "dynamic").strip().lower()
            if mode in {"0", "false", "off", "none"}:
                return None
            if mode in {"auto", "adaptive"}:
                return getattr(Dim, "AUTO", None)
            if mode in {"static", "fixed"}:
                return getattr(Dim, "STATIC", None)
            try:
                return Dim(name, min=1)
            except Exception:
                try:
                    return Dim(name)
                except Exception:
                    return getattr(Dim, "AUTO", None)

        if dynamic_batch:
            bd = _dim_from_mode("batch", "ENN_EXPORT_BATCH_DIM")
            if bd is not None:
                spec[0] = bd
        if dynamic_seq and isinstance(sample, torch.Tensor) and sample.ndim >= 2:
            sd = _dim_from_mode("seq", "ENN_EXPORT_SEQ_DIM")
            if sd is not None:
                spec[1] = sd
        spec = {k: v for k, v in spec.items() if v is not None}
        if spec:
            dynamic_shapes = (spec,)
    call_kw: dict[str, Any] = {}
    try:
        sig = inspect.signature(torch_export)
        params = sig.parameters
        if dynamic_shapes is not None and "dynamic_shapes" in params:
            call_kw["dynamic_shapes"] = dynamic_shapes
        if "strict" in params:
            call_kw["strict"] = bool(strict)
    except Exception:
        if dynamic_shapes is not None:
            call_kw["dynamic_shapes"] = dynamic_shapes
        call_kw["strict"] = bool(strict)

    def _call(**kw: Any) -> Any:
        with _strip_for_export(wrapper), torch.no_grad():
            return torch_export(wrapper, (sample,), **kw)

    def _is_constraint_violation(exc: BaseException) -> bool:
        msg = str(exc)
        return (
            "Constraints violated" in msg
            or "specialized it to be a constant" in msg
            or "marked batch as dynamic" in msg
            or "marked seq as dynamic" in msg
        )

    default_allow_non_strict = "0" if sys.version_info >= (3, 12) else "1"
    allow_non_strict = os.environ.get(
        "ENN_EXPORT_ALLOW_NON_STRICT", default_allow_non_strict
    ).strip().lower() not in ("0", "false", "off", "no", "n")
    try:
        return _call(**call_kw)
    except TypeError as exc:
        msg = str(exc)
        stripped = dict(call_kw)
        retry = False
        for k in ("dynamic_shapes", "strict"):
            if k in stripped and f"'{k}'" in msg:
                stripped.pop(k, None)
                retry = True
        if retry:
            return _call(**stripped)
        raise
    except Exception as exc:
        if (
            call_kw.get("strict", False)
            and call_kw.get("dynamic_shapes") is not None
            and _is_constraint_violation(exc)
        ):
            try:
                Dim = torch.export.Dim
                auto_hint = getattr(Dim, "AUTO", None)
                static_hint = getattr(Dim, "STATIC", None)
            except Exception:
                auto_hint, static_hint = None, None
            if auto_hint is not None:
                try:
                    spec0 = call_kw["dynamic_shapes"][0]
                    auto_spec = {
                        k: (
                            v
                            if (static_hint is not None and v == static_hint)
                            else auto_hint
                        )
                        for k, v in dict(spec0).items()
                    }
                    auto_kw = dict(call_kw)
                    auto_kw["dynamic_shapes"] = (auto_spec,)
                    return _call(**auto_kw)
                except Exception:
                    pass
            retry_kw = dict(call_kw)
            retry_kw.pop("dynamic_shapes", None)
            return _call(**retry_kw)
        if call_kw.get("strict", False):
            if not allow_non_strict:
                raise
            call_kw["strict"] = False
            return _call(**call_kw)
        raise


def _sanitize_exported_program(exported: object) -> object:
    gm = getattr(exported, "graph_module", None)
    g = getattr(gm, "graph", None) if gm is not None else None
    if g is None:
        return exported
    with contextlib.suppress(Exception):
        g.eliminate_dead_code()
    for node in list(getattr(g, "nodes", ())):
        try:
            if node.op != "call_function":
                continue
            tgt = getattr(node, "target", None)
            name = getattr(tgt, "__name__", "")
            mod = getattr(tgt, "__module__", "")
            if (
                name == "lazy_load_decompositions"
                and "torch._functorch.predispatch" in mod
            ):
                if not getattr(node, "users", {}):
                    g.erase_node(node)
        except Exception:
            continue
    with contextlib.suppress(Exception):
        g.lint()
    with contextlib.suppress(Exception):
        gm.recompile()
    return exported


def _in_console(cmd: object, desc: object, *args: Any, cwd: object = None) -> None:
    try:
        subprocess.run(list(cmd), check=True, cwd=cwd)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"{desc} failed with error: {exc}") from exc


def _export_sig() -> object:
    global _EXPORT_SIG_CACHE
    if _EXPORT_SIG_CACHE is not None:
        return _EXPORT_SIG_CACHE
    with _EXPORT_SIG_LOCK:
        if _EXPORT_SIG_CACHE is not None:
            return _EXPORT_SIG_CACHE
        try:
            sig = inspect.signature(torch.onnx.export)
        except Exception:
            sig = None
        _EXPORT_SIG_CACHE = sig
        return sig


def _export_sig_keys() -> set[str]:
    sig = _export_sig()
    if sig is None:
        return set()
    try:
        params = getattr(sig, "parameters", None)
        if isinstance(params, dict):
            return set(params.keys())
    except Exception:
        pass
    try:
        return set(sig.parameters.keys())
    except Exception:
        return set()


class _TensorDictPack(nn.Module):
    def __init__(self: Self, averaged_module: nn.Module, key: str) -> None:
        super().__init__()
        self._averaged_module = averaged_module
        self._key = str(key)

    def forward(self: Self, x: torch.Tensor) -> Any:
        try:
            from tensordict import TensorDict
        except Exception as exc:
            raise ImportError("tensordict is required for TensorDict-based models") from exc
        bs = int(x.shape[0]) if (hasattr(x, "ndim") and x.ndim >= 1) else 1
        td = TensorDict({self._key: x}, batch_size=[bs], device=x.device)
        return self._averaged_module(td)


class _TensorOutputModule(nn.Module):
    def __init__(self: Self, net: object) -> None:
        super().__init__()
        self.net = net

    def forward(self: Self, x: object) -> object:
        return extract_tensor(_forward(self.net, x))


class _ONNXExporter:
    @staticmethod
    def export(
        model: nn.Module,
        onnx_path: PathLike,
        *args: Any,
        sample_input: object | None = None,
        opset_version: int = 18,
        opset_fallback: Sequence[int] | None = None,
        dynamic_batch: bool = False,
        prefer_dynamo: bool = False,
        simplify: bool = False,
        optimize_onnx: bool = True,
        onnxoptimizer_passes: Sequence[str] | None = None,
    ) -> object:
        is_required("onnx", "pip install onnx")
        wrapper = _TensorOutputModule(model).eval()
        onnx_path = Path(onnx_path)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        input_names = ["features"]
        dyn_axes = None
        dyn_shapes = None
        if dynamic_batch:
            dyn_axes = {"features": {0: "batch"}, "preds_flat": {0: "batch"}}
            if hasattr(torch, "export") and hasattr(torch.export, "Dim"):
                mode = os.environ.get("ENN_EXPORT_BATCH_DIM", "auto").strip().lower()
                batch_dim = None
                if mode == "explicit":
                    try:
                        batch_dim = torch.export.Dim("batch", min=1)
                    except Exception:
                        try:
                            batch_dim = torch.export.Dim("batch")
                        except Exception:
                            batch_dim = None
                else:
                    try:
                        batch_dim = getattr(torch.export.Dim, "AUTO")
                    except Exception:
                        batch_dim = None
                if batch_dim is not None:
                    dyn_shapes = ({0: batch_dim},)
        min_export_batch = 2 if dynamic_batch else 1
        sample = _pad_sample(model, sample_input, batch=min_export_batch)
        if isinstance(sample, torch.Tensor) and sample.ndim == 1:
            sample = sample.unsqueeze(0)
        if (
            dynamic_batch
            and isinstance(sample, torch.Tensor)
            and sample.ndim >= 2
            and int(sample.shape[0]) < min_export_batch
        ):
            pad = torch.zeros(
                (min_export_batch - int(sample.shape[0]),) + tuple(sample.shape[1:]),
                device=sample.device,
                dtype=sample.dtype,
            )
            sample = torch.cat([sample, pad], dim=0)
        training = None
        with contextlib.suppress(Exception):
            training = torch.onnx.TrainingMode.EVAL
        sig_keys = _export_sig_keys()
        has_dynamo = "dynamo" in sig_keys
        allow_dynamo_fallback = False
        with contextlib.suppress(Exception):
            v = os.environ.get("ENN_ONNX_TRY_DYNAMO", "").strip().lower()
            allow_dynamo_fallback = v in ("1", "true", "yes", "on")
        if has_dynamo:
            if prefer_dynamo:
                exporters = [True, False]
            elif allow_dynamo_fallback:
                exporters = [False, True]
            else:
                exporters = [False]
        else:
            exporters = [False]
        errors: list[str] = []
        seen: set[int] = set()
        for opset in (opset_version, *(opset_fallback or ())):
            if opset in seen or opset < 1:
                continue
            seen.add(opset)
            base_kw: dict[str, Any] = {
                "export_params": True,
                "f": str(onnx_path),
                "opset_version": int(opset),
                "do_constant_folding": False,
                "keep_initializers_as_inputs": False,
                "input_names": input_names,
                "output_names": ["preds_flat"],
            }
            if training is not None:
                base_kw["training"] = training
            if dyn_axes:
                base_kw["dynamic_axes"] = dyn_axes
            valid_kw = {k: v for k, v in base_kw.items() if k in sig_keys}
            for use_dyn in exporters:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=r"torchvision|Setting ONNX exporter to use operator set version",
                        )
                        call_kw = dict(valid_kw)
                        if use_dyn and dyn_shapes and "dynamic_shapes" in sig_keys:
                            call_kw["dynamic_shapes"] = dyn_shapes
                        if has_dynamo:
                            call_kw["dynamo"] = use_dyn
                        call_kw.pop("model", None)
                        call_kw.pop("args", None)
                        if isinstance(sample, tuple):
                            args = sample
                        elif isinstance(sample, list):
                            args = tuple(sample)
                        else:
                            args = (sample,)
                        torch.onnx.export(model=wrapper, args=args, **call_kw)
                    if optimize_onnx:
                        if (
                            importlib.util.find_spec("onnx") is not None
                            and importlib.util.find_spec("onnxoptimizer") is not None
                        ):
                            try:
                                import onnx
                                import onnxoptimizer

                                model_onnx = onnx.load(str(onnx_path))
                                passes = (
                                    list(onnxoptimizer_passes)
                                    if onnxoptimizer_passes is not None
                                    else None
                                )
                                model_opt = onnxoptimizer.optimize(model_onnx, passes)
                                onnx.save(model_opt, str(onnx_path))
                            except Exception as exc:
                                warnings.warn(
                                    f"ONNX optimization failed; keeping the exported model. ({exc})"
                                )
                    if simplify:
                        with contextlib.suppress(Exception):
                            import onnx
                            import onnxsim

                            model_onnx = onnx.load(str(onnx_path))
                            model_simp, ok = onnxsim.simplify(model_onnx)
                            if ok:
                                onnx.save(model_simp, str(onnx_path))
                    return onnx_path
                except Exception as exc:
                    errors.append(f"opset={opset} dynamo={use_dyn} -> {exc}")
        raise RuntimeError("ONNX export failed. " + "; ".join(errors[-6:]))

    @staticmethod
    def coerce(
        model: nn.Module, onnx_path: PathLike, *args: Any, **kwargs: Any
    ) -> object:
        if not onnx_path.exists():
            return _ONNXExporter.export(model, onnx_path, **kwargs)
        return onnx_path


class _ORTBuilder:
    @staticmethod
    def to_ort(
        onnx_path: PathLike,
        ort_path: PathLike,
        *args: Any,
        optimization_level: str = "all",
        optimization_style: str = "fixed",
        target_platform: object | None = None,
        save_optimized_onnx_model: bool = False,
        **kwargs: Any,
    ) -> object:
        is_required("onnxruntime", "pip install onnxruntime")
        import onnxruntime as ort

        opt_map = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        level = opt_map.get(
            optimization_level.lower(),
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        )
        disabled_optimizers: list[str] | None = None
        if level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL:
            disabled_optimizers = ["NchwcTransformer"]
        optimized_onnx_path = None
        if save_optimized_onnx_model:
            optimized_onnx_path = ort_path.with_suffix(".optimized.onnx")
            try:
                so_onnx = ort.SessionOptions()
                so_onnx.optimized_model_filepath = str(optimized_onnx_path)
                so_onnx.graph_optimization_level = level
                if optimization_style.lower() == "runtime":
                    so_onnx.add_session_config_entry(
                        "optimization.minimal_build_optimizations", "apply"
                    )
                ort.InferenceSession(
                    str(onnx_path),
                    sess_options=so_onnx,
                    providers=["CPUExecutionProvider"],
                    disabled_optimizers=disabled_optimizers,
                )
            except Exception as exc:
                warnings.warn(
                    f"ORT optimized ONNX save failed; continuing without it. ({exc})",
                    RuntimeWarning,
                )
                optimized_onnx_path = None
        levels_to_try = [level]
        if level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL:
            levels_to_try.extend(
                [
                    ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                    ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                ]
            )
        elif level == ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED:
            levels_to_try.extend(
                [
                    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                    ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                ]
            )
        elif level == ort.GraphOptimizationLevel.ORT_ENABLE_BASIC:
            levels_to_try.append(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
        last_exc: Exception | None = None
        for lvl in levels_to_try:
            try:
                so = ort.SessionOptions()
                so.optimized_model_filepath = str(ort_path)
                so.graph_optimization_level = lvl
                so.add_session_config_entry("session.save_model_format", "ORT")
                if optimization_style.lower() == "runtime":
                    so.add_session_config_entry(
                        "optimization.minimal_build_optimizations", "save"
                    )
                dis = (
                    disabled_optimizers
                    if lvl == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    else None
                )
                ort.InferenceSession(
                    str(onnx_path),
                    sess_options=so,
                    providers=["CPUExecutionProvider"],
                    disabled_optimizers=dis,
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise RuntimeError("ORT export failed.") from last_exc
        return (ort_path, optimized_onnx_path)

class Format(Protocol):
    name: str | None

    def save(
        self: Self, model: nn.Module, dst: PathLike, *args: Any, **kwargs: Any
    ) -> object: ...


class TorchInductor(Format):
    name = "aoti"

    def save(
        self: Self,
        model: nn.Module,
        dst: PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        del args
        try:
            import torch.export

            torch_export = torch.export.export
        except Exception as exc:
            raise ImportError(
                "torch.export is required for TorchInductor export (PyTorch 2.0+)."
            ) from exc
        try:
            from torch._inductor import aoti_compile_and_package
        except Exception as exc:
            raise ImportError(
                "torch._inductor (TorchInductor) is required for AOT compilation. "
                "Install a PyTorch build with torch.compile/TorchInductor support."
            ) from exc
        with _onnx_model(model) as serving_model:
            sample = kwargs.get("sample_input")
            sample = _pad_sample(serving_model, sample)
            if isinstance(sample, torch.Tensor) and sample.ndim == 1:
                sample = sample.unsqueeze(0)
            wrapper = _TensorOutputModule(serving_model).eval()

            exported = _torch_export_program(
                torch_export,
                wrapper,
                sample,
                dynamic_batch=bool(kwargs.get("dynamic_batch", True)),
                dynamic_seq=bool(kwargs.get("dynamic_seq", False)),
                strict=bool(kwargs.get("strict", True)),
                tag="TorchInductor export",
            )
            dst = Path(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            inductor_configs = kwargs.get("inductor_configs")
            aoti_kw: dict[str, Any] = {}
            try:
                sig = inspect.signature(aoti_compile_and_package)
                params = getattr(sig, "parameters", None)
                if isinstance(params, dict):
                    if "package_path" in params:
                        aoti_kw["package_path"] = str(dst)
                    if inductor_configs is not None and "inductor_configs" in params:
                        aoti_kw["inductor_configs"] = inductor_configs
            except Exception:
                aoti_kw["package_path"] = str(dst)
                if inductor_configs is not None:
                    aoti_kw["inductor_configs"] = inductor_configs
            try:
                out_path = aoti_compile_and_package(exported, **aoti_kw)
            except TypeError as exc:
                msg = str(exc)
                stripped = dict(aoti_kw)
                for k in ("package_path", "inductor_configs"):
                    if k in stripped and f"'{k}'" in msg:
                        stripped.pop(k, None)
                out_path = aoti_compile_and_package(exported, **stripped)
            out = Path(str(out_path))
            if out != dst:
                with contextlib.suppress(Exception):
                    shutil.copyfile(out, dst)
                out = dst
            with contextlib.suppress(Exception):
                _write_export_meta(model, out, format_name=self.name or "aoti")
        return (out,)


class TorchExport(Format):
    name = "pt2"

    def save(
        self: Self,
        model: nn.Module,
        dst: PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        del args
        try:
            import torch.export

            torch_save = torch.export.save
        except Exception as exc:
            raise ImportError(
                "torch.export is required for PT2 export (PyTorch 2.0+)."
            ) from exc
        with _onnx_model(model) as serving_model:
            sample = kwargs.get("sample_input")
            sample = _pad_sample(serving_model, sample)
            if isinstance(sample, torch.Tensor) and sample.ndim == 1:
                sample = sample.unsqueeze(0)
            wrapper = _TensorOutputModule(serving_model).eval()
            with _no_empty_tensor(serving_model):
                exported = self._export_program(wrapper, sample, **kwargs)
                exported = _sanitize_exported_program(exported)
            dst = Path(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            with from_buffer():
                torch_save(exported, str(dst))
            with contextlib.suppress(Exception):
                _write_export_meta(model, dst, format_name=self.name or "pt2")
        return (dst,)

    def _export_program(
        self: Self, wrapper: nn.Module, sample: torch.Tensor, **kwargs: Any
    ) -> object:
        try:
            import torch.export

            torch_export = torch.export.export
        except Exception as exc:
            raise ImportError(
                "torch.export is required for PT2 export (PyTorch 2.0+)."
            ) from exc
        return _torch_export_program(
            torch_export,
            wrapper,
            sample,
            dynamic_batch=bool(kwargs.get("dynamic_batch", True)),
            dynamic_seq=bool(kwargs.get("dynamic_seq", False)),
            strict=bool(kwargs.get("strict", True)),
            tag="PT2 export",
        )


class ExecuTorch(Format):
    name = "executorch"

    @staticmethod
    def _truthy_env(name: str) -> bool:
        v = os.environ.get(name, "")
        return v.strip().lower() in ("1", "true", "yes", "y", "on")

    def save(
        self: Self,
        model: nn.Module,
        dst: PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        is_required("executorch", "pip install executorch")
        try:
            from torch.export import export as torch_export
        except ImportError as exc:
            raise ImportError(
                "torch.export is required for ExecuTorch export (PyTorch 2.0+)."
            ) from exc
        import executorch.exir as exir

        dst = Path(dst)
        with _onnx_model(model) as serving_model:
            sample = kwargs.get("sample_input")
            sample = _pad_sample(serving_model, sample)
            if isinstance(sample, torch.Tensor) and sample.ndim == 1:
                sample = sample.unsqueeze(0)
            wrapper = _TensorOutputModule(serving_model).eval()
            dst.parent.mkdir(parents=True, exist_ok=True)
            dyn_batch = bool(kwargs.get("dynamic_batch", True))
            dyn_seq = bool(kwargs.get("dynamic_seq", False))
            strict = bool(kwargs.get("strict", True))
            silent_fallback = bool(
                kwargs.get("silent_fallback", False)
            ) or self._truthy_env("ENN_EXPORT_SILENT_FALLBACK")

            def _do_export(db: bool, ds: bool) -> object:
                return _torch_export_program(
                    torch_export,
                    wrapper,
                    sample,
                    dynamic_batch=db,
                    dynamic_seq=ds,
                    strict=strict,
                    tag="ExecuTorch export"
                    + (" (static)" if (not db and not ds) else ""),
                )

            try:
                exported = _do_export(dyn_batch, dyn_seq)
            except Exception:
                if dyn_batch or dyn_seq:
                    if os.environ.get("ENN_EXPORT_WARNINGS", "").strip().lower() in (
                        "1",
                        "true",
                        "yes",
                        "y",
                        "on",
                    ):
                        warnings.warn(
                            "ExecuTorch export failed with dynamic shapes; retrying with dynamic_batch=False, dynamic_seq=False",
                            RuntimeWarning,
                        )
                    exported = _do_export(False, False)
                else:
                    raise

            def _to_executorch_program(exported_program: object) -> object:
                edge = exir.to_edge(exported_program)
                if hasattr(exir, "to_executorch"):
                    return exir.to_executorch(edge)
                if hasattr(edge, "to_executorch"):
                    return edge.to_executorch()
                if hasattr(edge, "to_executorch_program"):
                    return edge.to_executorch_program()
                raise AttributeError(
                    "ExecuTorch export: could not find `exir.to_executorch(edge)` nor `edge.to_executorch()`."
                )

            used_dyn_batch, used_dyn_seq = dyn_batch, dyn_seq
            fallback_error: str | None = None
            with _temp_environ(
                {"ET_EXIR_SAVE_FLATC_INPUTS_ON_FAILURE": "1"},
                only_if_unset=True,
            ):
                try:
                    exec_prog = _to_executorch_program(exported)
                    with open(dst, "wb") as fh:
                        exec_prog.write_to_file(fh)
                except Exception as exc:
                    if dyn_batch or dyn_seq:
                        used_dyn_batch, used_dyn_seq = False, False
                        fallback_error = repr(exc)
                        if not silent_fallback:
                            warnings.warn(
                                "ExecuTorch export with dynamic shapes failed; "
                                "falling back to a static program (dynamic_batch=False, dynamic_seq=False). "
                                "This static artifact may not accept variable-length inputs. "
                                "To silence this warning, pass silent_fallback=True or set "
                                "ENN_EXPORT_SILENT_FALLBACK=1.",
                                RuntimeWarning,
                            )
                        exported_static = _torch_export_program(
                            torch_export,
                            wrapper,
                            sample,
                            dynamic_batch=False,
                            dynamic_seq=False,
                            strict=strict,
                            tag="ExecuTorch export (static fallback)",
                        )
                        exec_prog = _to_executorch_program(exported_static)
                        with open(dst, "wb") as fh:
                            exec_prog.write_to_file(fh)
                    else:
                        raise
            with contextlib.suppress(Exception):
                _write_export_meta(
                    model,
                    dst,
                    format_name=self.name or "executorch",
                    extra={
                        "requested_dynamic_batch": dyn_batch,
                        "requested_dynamic_seq": dyn_seq,
                        "exported_dynamic_batch": used_dyn_batch,
                        "exported_dynamic_seq": used_dyn_seq,
                        "static_fallback": bool(
                            (dyn_batch or dyn_seq)
                            and (not used_dyn_batch and not used_dyn_seq)
                        ),
                        "fallback_error": fallback_error,
                    },
                )
        return (dst,)


class ONNX(Format):
    name = "onnx"

    def save(
        self: Self,
        model: nn.Module,
        dst: PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        dst = Path(dst)
        with _onnx_model(model) as serving:
            out = _ONNXExporter.export(
                serving, dst, **_onnx_options(kwargs, target="onnx")
            )
        with contextlib.suppress(Exception):
            _write_export_meta(model, out, format_name=self.name or "onnx")
        return (out,)


class ORT(Format):
    name = "ort"

    def save(
        self: Self,
        model: nn.Module,
        dst: PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        dst = Path(dst)
        with _onnx_model(model) as serving:
            onnx_path = _ONNXExporter.coerce(
                serving,
                _coerce_onnx_path(dst, kwargs),
                **_onnx_options(kwargs, target="onnx"),
            )
            ort_path, optimized = _ORTBuilder.to_ort(
                onnx_path,
                dst,
                optimization_level=str(kwargs.get("optimization_level", "all")),
                optimization_style=str(kwargs.get("optimization_style", "fixed")),
                target_platform=kwargs.get("target_platform"),
                save_optimized_onnx_model=bool(
                    kwargs.get("save_optimized_onnx_model", False)
                ),
            )
        with contextlib.suppress(Exception):
            _write_export_meta(
                model,
                ort_path,
                format_name=self.name or "ort",
                extra={
                    "onnx_path": str(onnx_path),
                    "optimized_onnx": (
                        str(optimized) if optimized is not None else None
                    ),
                },
            )
        return (ort_path, optimized) if optimized is not None else (ort_path,)


class TensorRT(Format):
    name = "tensorrt"

    def save(
        self: Self,
        model: nn.Module,
        dst: PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        del args
        dst = Path(dst)
        if not torch.cuda.is_available():
            raise ImportError(
                "TensorRT export requires CUDA-enabled PyTorch (torch.cuda.is_available() is False)."
            )
        try:
            torch.cuda.current_device()
        except Exception as cuda_exc:
            raise ImportError(
                "CUDA runtime/driver is not available or incompatible for TensorRT export."
            ) from cuda_exc
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError("TensorRT is required for this export.") from exc
        with _onnx_model(model) as serving_model:
            onnx_path = _ONNXExporter.coerce(
                serving_model,
                _coerce_onnx_path(dst, kwargs),
                **_onnx_options(kwargs, target="tensorrt"),
            )
            if bool(kwargs.get("graphsurgeon", True)):
                if (
                    importlib.util.find_spec("onnx") is not None
                    and importlib.util.find_spec("onnx_graphsurgeon") is not None
                ):
                    try:
                        import onnx
                        import onnx_graphsurgeon as gs

                        gs_model = onnx.load(str(onnx_path))
                        graph = gs.import_onnx(gs_model)
                        graph.cleanup().toposort()
                        onnx.save(gs.export_onnx(graph), str(onnx_path))
                    except Exception as exc:
                        warnings.warn(
                            f"TensorRT graphsurgeon optimization failed; using unoptimized ONNX. ({exc})"
                        )
            trt_logger = trt.Logger(trt.Logger.WARNING)
            explicit_batch_flag = 0
            with contextlib.suppress(Exception):
                explicit_batch_flag = 1 << int(
                    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
                )
            with contextlib.ExitStack() as stack:
                builder = stack.enter_context(trt.Builder(trt_logger))
                create_network = getattr(builder, "create_network", None)
                if not callable(create_network):
                    raise RuntimeError(
                        "TensorRT builder.create_network is not available."
                    )
                try:
                    network = stack.enter_context(create_network(explicit_batch_flag))
                except TypeError:
                    network = stack.enter_context(create_network())
                parser = stack.enter_context(trt.OnnxParser(network, trt_logger))
                config = stack.enter_context(builder.create_builder_config())
                workspace_size_bytes = int(kwargs.get("workspace_size_bytes", 1 << 30))
                if hasattr(config, "set_memory_pool_limit"):
                    config.set_memory_pool_limit(
                        trt.MemoryPoolType.WORKSPACE, workspace_size_bytes
                    )
                else:
                    config.max_workspace_size = workspace_size_bytes
                with open(onnx_path, "rb") as handle:
                    if not parser.parse(handle.read()):
                        for i in range(parser.num_errors):
                            print(parser.get_error(i))
                        raise RuntimeError("TensorRT could not parse the ONNX model.")
                use_fp16 = bool(kwargs.get("fp16", True))
                use_int8 = bool(kwargs.get("int8", False))
                if use_fp16 and bool(getattr(builder, "platform_has_fast_fp16", False)):
                    with contextlib.suppress(Exception):
                        config.set_flag(trt.BuilderFlag.FP16)
                if use_int8:
                    if not bool(getattr(builder, "platform_has_fast_int8", False)):
                        warnings.warn(
                            "INT8 precision is not supported on this platform; ignoring request."
                        )
                    else:
                        calibrator = kwargs.get("calibrator")
                        if calibrator is not None:
                            with contextlib.suppress(Exception):
                                config.set_int8_calibrator(calibrator)
                        with contextlib.suppress(Exception):
                            config.set_flag(trt.BuilderFlag.INT8)
                input_tensor = network.get_input(0)
                sample = _pad_sample(serving_model, kwargs.get("sample_input"))
                shape = tuple((int(x) for x in sample.shape))
                min_batch = max(1, int(kwargs.get("min_batch", 1)))
                opt_batch = max(min_batch, int(kwargs.get("opt_batch", shape[0])))
                max_batch = max(opt_batch, int(kwargs.get("max_batch", 8)))
                min_shape = (min_batch, *shape[1:])
                opt_shape = (opt_batch, *shape[1:])
                max_shape = (max_batch, *shape[1:])
                profile = builder.create_optimization_profile()
                profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
                config.add_optimization_profile(profile)
                engine_blob = None
                build_serialized = getattr(builder, "build_serialized_network", None)
                if callable(build_serialized):
                    engine_blob = build_serialized(network, config)
                else:
                    build_engine = getattr(builder, "build_engine", None) or getattr(
                        builder, "build_cuda_engine", None
                    )
                    if callable(build_engine):
                        engine = build_engine(network, config)
                        if engine is not None:
                            engine_blob = getattr(engine, "serialize", lambda: None)()
                if engine_blob is None:
                    raise RuntimeError("Failed to build the TensorRT engine.")
                try:
                    engine_bytes = bytes(engine_blob)
                except Exception:
                    engine_bytes = engine_blob
                if not isinstance(engine_bytes, (bytes, bytearray)):
                    buf = getattr(engine_blob, "buffer", None)
                    if buf is not None:
                        engine_bytes = buf
                if not isinstance(engine_bytes, (bytes, bytearray)):
                    raise RuntimeError(
                        f"TensorRT engine serialization returned unexpected type: {type(engine_blob)}"
                    )
                dst.parent.mkdir(parents=True, exist_ok=True)
                with open(dst, "wb") as handle:
                    handle.write(engine_bytes)

                with contextlib.suppress(Exception):
                    _write_export_meta(
                        model,
                        dst,
                        format_name=self.name or "tensorrt",
                        extra={"onnx_path": str(onnx_path)},
                    )
        return (dst,)


class CoreML(Format):
    name = "coreml"

    def save(
        self: Self,
        model: nn.Module,
        dst: PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        if sys.platform != "darwin":
            raise ImportError(
                "coremltools is required for this operation (try: pip install coremltools on macOS)"
            )
        spec = importlib.util.find_spec("coremltools")
        if spec is None or not getattr(spec, "submodule_search_locations", None):
            raise ImportError(
                "coremltools is required for this operation (try: pip install coremltools on macOS)"
            )
        pkg_dirs = list(spec.submodule_search_locations or [])
        has_native = any(
            glob.glob(os.path.join(d, "libcoremlpython*")) for d in pkg_dirs
        )
        if not has_native:
            raise ImportError(
                "coremltools is required for this operation (try: pip install coremltools on macOS)"
            )
        import coremltools as ct

        dst = Path(dst)
        with _onnx_model(model) as serving_model:
            sample = _pad_sample(serving_model, kwargs.get("sample_input"))
            wrapper = _TensorOutputModule(serving_model).eval()
            with torch.no_grad():
                scripted = torch.jit.trace(
                    wrapper,
                    sample,
                    check_trace=False,
                    strict=False,
                )
            with contextlib.suppress(Exception):
                scripted = torch.jit.freeze(scripted)
            with contextlib.suppress(Exception):
                scripted = torch.jit.optimize_for_inference(scripted)
            cu_map = {
                "ALL": getattr(ct.ComputeUnit, "ALL", None),
                "CPU_ONLY": getattr(ct.ComputeUnit, "CPU_ONLY", None),
                "CPU_AND_GPU": getattr(ct.ComputeUnit, "CPU_AND_GPU", None),
                "CPU_AND_NE": getattr(ct.ComputeUnit, "CPU_AND_NE", None),
            }
            convert_to = str(kwargs.get("convert_to", "mlprogram"))
            compute_units = cu_map.get(
                str(kwargs.get("compute_units", "ALL")).upper(), cu_map["ALL"]
            )
            kwargs_dict = {
                "inputs": [ct.TensorType(shape=tuple((int(x) for x in sample.shape)))],
                "convert_to": convert_to,
                "compute_units": compute_units,
            }
            deployment_target = kwargs.get("minimum_deployment_target")
            if deployment_target:
                target = getattr(ct.target, deployment_target, None)
                if target is not None:
                    kwargs_dict["minimum_deployment_target"] = target
            convert_to_try = [str(convert_to)]
            ct_to_l = str(convert_to).strip().lower()
            if ct_to_l != "neuralnetwork":
                convert_to_try.append("neuralnetwork")
            if ct_to_l != "mlprogram":
                convert_to_try.append("mlprogram")
            mlmodel = None
            last_exc: Exception | None = None
            for ct_to in convert_to_try:
                kwargs_dict["convert_to"] = ct_to
                try:
                    mlmodel = ct.convert(scripted, **kwargs_dict)
                    break
                except Exception as exc:
                    last_exc = exc
                    mlmodel = None

            if mlmodel is None:
                raise RuntimeError(
                    f"CoreML conversion failed. Tried convert_to={convert_to_try}."
                ) from last_exc
            dst.parent.mkdir(parents=True, exist_ok=True)
            mlmodel.save(str(dst))
            with contextlib.suppress(Exception):
                _write_export_meta(model, dst, format_name=self.name or "coreml")
        return (dst,)


class LiteRT(Format):
    name = "litert"

    def save(
        self: Self,
        model: nn.Module,
        dst: PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        del args
        dst = Path(dst)
        prefer_ai_edge = bool(
            kwargs.get("prefer_ai_edge_torch", kwargs.get("prefer_ai_edge", True))
        )
        have_ai_edge = importlib.util.find_spec("ai_edge_torch") is not None
        have_onnx2tf = importlib.util.find_spec("onnx2tf") is not None
        with _onnx_model(model) as serving_model:
            if prefer_ai_edge and have_ai_edge:
                try:
                    import ai_edge_torch

                    sample = kwargs.get("sample_input")
                    sample = _pad_sample(serving_model, sample, batch=1)
                    if isinstance(sample, torch.Tensor) and sample.ndim == 1:
                        sample = sample.unsqueeze(0)
                    wrapper = _TensorOutputModule(serving_model).eval()
                    edge_model = ai_edge_torch.convert(wrapper, (sample,))
                    exporter = getattr(edge_model, "export", None)
                    if not callable(exporter):
                        raise AttributeError(
                            "ai-edge-torch conversion did not return an object with .export()"
                        )
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    exporter(str(dst))
                    with contextlib.suppress(Exception):
                        _write_export_meta(
                            model,
                            dst,
                            format_name=self.name or "litert",
                            extra={"backend": "ai-edge-torch"},
                        )
                    return (dst,)
                except Exception as exc:
                    if have_onnx2tf and bool(kwargs.get("fallback_onnx2tf", True)):
                        warnings.warn(
                            f"ai-edge-torch conversion failed; retrying via onnx2tf: {exc}",
                            RuntimeWarning,
                        )
                    else:
                        raise
            if not have_onnx2tf:
                raise ImportError(
                    "LiteRT export requires ai-edge-torch (preferred) or onnx2tf (fallback). "
                    "Try: pip install ai-edge-torch (or: pip install onnx2tf)"
                )
            is_required("onnx2tf", "pip install onnx2tf")
            onnx_path = _ONNXExporter.coerce(
                serving_model,
                _coerce_onnx_path(dst, kwargs),
                **_onnx_options(kwargs, target="litert"),
            )
            work_dir = dst.with_name(dst.name + ".onnx2tf")
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
            work_dir.mkdir(parents=True, exist_ok=True)
            _run_onnx2tf(
                Path(onnx_path),
                work_dir,
                "--copy_onnx_input_output_names_to_tflite",
                dynamic_batch=bool(kwargs.get("dynamic_batch", True)),
            )
            tflites = list(work_dir.rglob("*.tflite"))
            if not tflites:
                raise RuntimeError("onnx2tf did not produce a .tflite model.")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tflites[0], dst)
            with contextlib.suppress(Exception):
                _write_export_meta(
                    model,
                    dst,
                    format_name=self.name or "litert",
                    extra={"backend": "onnx2tf", "onnx_path": str(onnx_path)},
                )
            return (dst,)


class TensorFlow(Format):
    name = "tensorflow"

    def save(
        self: Self,
        model: object,
        dst: PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        del args
        is_required("onnx2tf", "pip install onnx2tf")
        dst = Path(dst)
        suffix = dst.suffix.lower()
        if suffix == ".savedmodel":
            saved_model_dir = dst
        elif suffix in {".pb", ".tf"}:
            saved_model_dir = dst.with_suffix("")
        else:
            saved_model_dir = dst
        with _onnx_model(model) as serving_model:
            onnx_path = _ONNXExporter.coerce(
                serving_model,
                _coerce_onnx_path(saved_model_dir, kwargs),
                **_onnx_options(kwargs, target="tensorflow"),
            )
            with tempfile.TemporaryDirectory() as tmpd:
                tmp_out = Path(tmpd) / "onnx2tf_out"
                tmp_out.mkdir(parents=True, exist_ok=True)
                _run_onnx2tf(
                    Path(onnx_path),
                    tmp_out,
                    dynamic_batch=bool(kwargs.get("dynamic_batch", True)),
                )
                pb_files = list(tmp_out.rglob("saved_model.pb"))
                if not pb_files:
                    raise RuntimeError(
                        "onnx2tf did not produce a TensorFlow SavedModel (saved_model.pb)"
                    )
                src_dir = pb_files[0].parent
                if saved_model_dir.exists():
                    shutil.rmtree(saved_model_dir)
                saved_model_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src_dir, saved_model_dir)
        with contextlib.suppress(Exception):
            _write_export_meta(
                model,
                saved_model_dir,
                format_name=self.name or "tensorflow",
                extra={"backend": "onnx2tf", "onnx_path": str(onnx_path)},
            )
        return (saved_model_dir,)


_register_safe_globals()
