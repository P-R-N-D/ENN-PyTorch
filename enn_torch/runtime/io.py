# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import inspect
import json
import os
import pickle
import re
import sys
import tempfile
import warnings
import weakref
from base64 import b64decode, b64encode
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence, Self

import torch
from torch import nn

from ..core.concurrency import Mutex
from ..core.datatypes import PathLike, coerce_json, save_temp, write_json
from ..core.distributed import distributed_barrier, is_rank0

if TYPE_CHECKING:
    from .wrappers import Format

_IGNORED_WARNINGS = (
    "torch.distributed is disabled",
    "TypedStorage is deprecated",
)
_IGNORED_RE = (
    r".*(?:" + "|".join(re.escape(s) for s in _IGNORED_WARNINGS) + r").*"
)
_SAVE_LOCK_GUARD = Mutex()
_SAVE_PATH_LOCKS = weakref.WeakValueDictionary()
_WARNINGS_FILTER_LOCK = Mutex()

_OPENZL_MAGIC = "enn-openzl-ckpt-v1"
_OPENZL_TENSOR_MARKER = "__ozl_tensor__"
_OPENZL_DICT_MARKER = "__ozl_dict__"
_OPENZL_TUPLE_MARKER = "__ozl_tuple__"
_OPENZL_PICKLE_MARKER = "__ozl_pickle__"

_OPENZL_COMPRESSOR: object | None = None
_OPENZL_LOCK = Mutex()

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
        from ..config import _extract_model_config_dict

        return _extract_model_config_dict(model)
    except Exception:
        return {}


def _torch_load_checkpoint(
    path: PathLike,
    *args: Any,
    map_location: object = None,
    weights_only: bool = True,
) -> object:
    try:
        return torch.load(
            str(path),
            map_location=map_location or "cpu",
            weights_only=weights_only,
        )
    except TypeError:
        return torch.load(str(path), map_location=map_location or "cpu")
    except Exception as exc:
        if weights_only:
            raise RuntimeError("weights_only=True failed") from exc
        raise


def is_required(module: str, pip_hint: str | None = None) -> None:
    try:
        __import__(module)
    except ImportError as err:
        hint = f" (try: {pip_hint})" if pip_hint else ""
        raise ImportError(
            f"{module} is required for this operation{hint}"
        ) from err

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
                *,
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
                        v = os.environ.get(name, '')
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

                min_n = _env_int('ENN_OPENZL_INT_AUTO_MIN_N', 128)
                if min_n < 0:
                    min_n = 0
                if n < min_n:
                    return base

                try:
                    flat = t.reshape(-1)
                    max_sample = _env_int('ENN_OPENZL_INT_AUTO_SAMPLE_N', 4096)
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
                        v = os.environ.get(name, '')
                        if v is None:
                            return default
                        v = v.strip()
                        if not v:
                            return default
                        return int(v)
                    return default

                def _env_float(name: str, default: float) -> float:
                    with contextlib.suppress(Exception):
                        v = os.environ.get(name, '')
                        if v is None:
                            return default
                        v = v.strip()
                        if not v:
                            return default
                        return float(v)
                    return default

                strat = os.environ.get('ENN_OPENZL_FP64_STRATEGY', 'auto').strip().lower()
                if strat in ('compress', 'generic', 'off', '0', 'false'):
                    return self._default_numeric_graph
                if strat in ('transpose', 'on', '1', 'true'):
                    return g

                try:
                    n = int(t.numel())
                except Exception:
                    return g

                min_n = _env_int('ENN_OPENZL_FP64_AUTO_MIN_N', 256)
                if min_n < 0:
                    min_n = 0
                if n < min_n:
                    return self._default_numeric_graph

                try:
                    flat = t.reshape(-1)
                    max_sample = _env_int('ENN_OPENZL_FP64_AUTO_SAMPLE_N', 4096)
                    if max_sample <= 0:
                        max_sample = 4096
                    sample_n = max_sample if n > max_sample else n
                    sample = flat[:sample_n].contiguous()
                    b = sample.view(torch.uint8).view(sample_n, 8)
                    lane7 = b[:, 7]
                    lane6 = b[:, 6]

                    u7 = int(torch.unique(lane7).numel())
                    u6 = int(torch.unique(lane6).numel())

                    u7_thr = _env_int('ENN_OPENZL_FP64_AUTO_UNIQUE_LANE7', 64)
                    u6_thr = _env_int('ENN_OPENZL_FP64_AUTO_UNIQUE_LANE6', 128)
                    if u7_thr < 1:
                        u7_thr = 1
                    if u6_thr < 1:
                        u6_thr = 1
                    if u7 <= u7_thr or u6 <= u6_thr:
                        return g

                    dom_thr = _env_float('ENN_OPENZL_FP64_AUTO_DOMINANT_RATIO', 0.55)
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


def _torch_dtype_from_str(dtype_str: str) -> torch.dtype:
    s = str(dtype_str)
    if s.startswith("torch."):
        s = s.split(".", 1)[1]
    dt = getattr(torch, s, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"Unsupported dtype string: {dtype_str!r}")
    return dt


def _openzl_jsonify(obj: object, *, tensors: list[torch.Tensor], tensor_table: list[dict[str, Any]]) -> object:

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
        seq = [_openzl_jsonify(v, tensors=tensors, tensor_table=tensor_table) for v in obj]
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
    *,
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
    *,
    openzl_level: int | None = None,
    openzl_format_version: int | None = None,
    openzl_min_stream_size: int | None = None,
    openzl_content_checksum: bool | None = None,
    openzl_compressed_checksum: bool | None = None,
    openzl_permissive: bool | None = None,
    openzl_pack_by_dtype: bool = True,
) -> bytes:

    zl = _openzl_import()
    compressor = _openzl_get_default_compressor()

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
    if openzl_permissive is not None:
        with contextlib.suppress(Exception):
            cctx.set_parameter(
                zl.CParam.PermissiveCompression, int(bool(openzl_permissive))
            )

    return bytes(cctx.compress(inputs))


def _openzl_decompress_payload(
    blob: object,
    *,
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

    outs = dctx.decompress(blob)  # bytes-like
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
        raw = outs[i + 1].content.as_pytensor()
        if getattr(raw, "dtype", None) != target:
            with contextlib.suppress(Exception):
                raw = raw.view(target)
        if getattr(raw, "dtype", None) != target:
            b = outs[i + 1].content.as_bytes()
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
    *,
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
    *,
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


class Builder:
    NATIVE_EXTS = {".pt", ".pth", ".safetensors", ".ozl"}

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
            return {
                "version": 1,
                "in_dim": int(getattr(model, "in_dim", 0)),
                "out_shape": tuple(
                    int(x) for x in getattr(model, "out_shape", ())
                ),
                "config": _load_model_config(model),
                "pytorch_version": torch.__version__,
                "extra": coerce_json(extra or {}),
            }

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
                payload = {**_make_meta(), "state_dict": sd, "format": "openzl-ckpt-v1"}
                if optimizer is not None:
                    with contextlib.suppress(Exception):
                        payload["optimizer_state_dict"] = optimizer.state_dict()

                _openzl_save_checkpoint(
                    p,
                    payload,
                    openzl_level=opts.pop("openzl_level", None),
                    openzl_format_version=opts.pop("openzl_format_version", None),
                    openzl_min_stream_size=opts.pop("openzl_min_stream_size", None),
                    openzl_content_checksum=opts.pop("openzl_content_checksum", None),
                    openzl_compressed_checksum=opts.pop(
                        "openzl_compressed_checksum", None
                    ),
                    openzl_permissive=opts.pop("openzl_permissive", None),
                    openzl_pack_by_dtype=bool(opts.pop("openzl_pack_by_dtype", True)),
                )
                meta = _make_meta()
                meta["format"] = "openzl-ckpt-v1"
                write_json(p.with_name(p.name + ".json"), meta, indent=2)
                with contextlib.suppress(Exception):
                    legacy = p.with_suffix(".json")
                    if legacy != p.with_name(p.name + ".json"):
                        write_json(legacy, meta, indent=2)
                return p
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
            try:
                from . import wrappers as _w
            except Exception as exc:
                raise RuntimeError(
                    "Exporter backends live in enn_torch.runtime.wrappers, but it could not be imported. "
                    "Install the optional export dependencies (e.g. tensordict) or avoid calling Exporter.for_export()."
                ) from exc

            cls._ONNXExporter = getattr(_w, "_ONNXExporter", None)
            cls._ORTBuilder = getattr(_w, "_ORTBuilder", None)

            cls._register_unlocked("onnx", (".onnx",), _w.ONNX())
            cls._register_unlocked("ort", (".ort",), _w.ORT())
            cls._register_unlocked(
                "tensorrt", (".engine", ".plan"), _w.TensorRT()
            )
            cls._register_unlocked(
                "coreml", (".mlmodel", ".mlpackage"), _w.CoreML()
            )
            cls._register_unlocked("litert", (".tflite",), _w.LiteRT())
            cls._register_unlocked(
                "pt2", (".pt2", ".export"), _w.TorchExport()
            )
            cls._register_unlocked("aoti", (".aoti",), _w.TorchInductor())
            cls._register_unlocked("executorch", (".pte",), _w.ExecuTorch())
            cls._register_unlocked(
                "tensorflow", (".savedmodel", ".pb", ".tf"), _w.TensorFlow()
            )
            cls._defaults_registered = True

    @classmethod
    def for_export(cls: type[Self], ext: str) -> Format | None:
        cls._ensure_defaults_registered()
        with cls._defaults_lock:
            name = cls._ext_map.get(ext.lower())
            return cls._by_name.get(name) if name else None


try:
    from torch.serialization import add_safe_globals
except ImportError:
    add_safe_globals = None
_register_safe_globals()
