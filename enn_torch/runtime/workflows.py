# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import collections
import gc
import inspect
import json
import logging
import os
import shutil
import tempfile
import time
from functools import partial, update_wrapper
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
    ParamSpec,
    Sequence,
    Tuple,
    TypeAlias,
    TypeVar,
    cast,
)

import numpy
import torch
import torch.multiprocessing
from tensordict import (
    MemoryMappedTensor,
    PersistentTensorDict,
    TensorDict,
    TensorDictBase,
)
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, load, save
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from ..core.config import (
    ModelConfig,
    RuntimeConfig,
    _extract_model_config_dict,
    coerce_model_config,
    runtime_config,
)
from ..core.datatypes import env_bool, read_json
from ..core.policies import WorkerPolicy
from ..core.system import _start_context, new_dir, optimal_start_method
from ..core.tensor import coerce_tensor, is_meta_or_fake_tensor
from ..data import collate
from ..data.collate import MappingSlicer, TensorDictSlicer
from ..data.pipeline import (
    Dataset,
    default_underflow_action,
    iter_dataset,
    normalize_underflow_action,
    preload_memmap,
)
from ..nn.graph import inference_mode
from ..nn.layers import Recorder, Scaler, resize_scaler_buffer
from ..nn.wrappers import Model
from .distributed import (
    ProcessBroker,
    _coerce_dcp_keys,
    ensure_dcp_process_group,
    get_available_host,
    get_preferred_ip,
    init_master_addr,
)
from .io import _filtered_warnings, _torch_load_checkpoint, is_required
from .main import process

try:
    from tensordict.nn.functional_modules import _exclude_td_from_pytree
    _exclude_td_from_pytree().set()
except Exception:
    pass

logger = logging.getLogger(__name__)


def _rewrite_state_dict_key(k: str) -> str:
    if k.startswith("module."):
        return k[len("module.") :]
    if k.startswith("model."):
        return k[len("model.") :]
    if k.startswith("m.") and ".module." in k:
        parts = k.split(".")
        if len(parts) >= 4 and parts[1].isdigit() and parts[2] == "module":
            return ".".join(parts[3:])
    return k


def _coerce_state_dict(sd: Mapping[str, Any]) -> Mapping[str, Any]:
    if not any(_rewrite_state_dict_key(k) != k for k in sd):
        return sd

    meta = getattr(sd, "_metadata", None)
    try:
        out: Any = type(sd)()
    except Exception:
        out = collections.OrderedDict()
    for k, v in sd.items():
        out[_rewrite_state_dict_key(k)] = v
    if meta is not None:
        with contextlib.suppress(Exception):
            setattr(out, "_metadata", meta)
    return out


def _drop_runtime_only_state_keys(sd: Mapping[str, Any]) -> Mapping[str, Any]:
    drop: list[str] = []
    for k in sd.keys():
        if not isinstance(k, str):
            continue
        if k == "output_baked_flag" or k.split(".")[-1] == "output_baked_flag":
            drop.append(k)
    if not drop:
        return sd

    meta = getattr(sd, "_metadata", None)
    try:
        out: Any = type(sd)()
        if hasattr(out, "update"):
            out.update(sd)
        else:
            out = collections.OrderedDict(sd)
    except Exception:
        out = collections.OrderedDict(sd)
    for k in drop:
        with contextlib.suppress(Exception):
            out.pop(k, None)
    if meta is not None:
        with contextlib.suppress(Exception):
            setattr(out, "_metadata", meta)

    if env_bool(
        "ENN_LOG_DROPPED_RUNTIME_KEYS",
        default=env_bool("ENN_SANITIZE_NAN_STRICT", default=False),
    ):
        ex = drop[0] if drop else None
        logger.warning(
            "[ENN] load_weights: dropped %d runtime-only key(s) from checkpoint state (e.g. %s)",
            int(len(drop)),
            str(ex),
        )
    return out


def _normalize_windows_paste(value: PathLike) -> PathLike:
    if isinstance(value, str):
        value = (
            value.replace("\\r\\n", "\n")
            .replace("\\n", "\n")
            .replace("\\r", "\n")
        )
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        value = value.strip()
        if "\n" in value:
            lines = [
                line.strip() for line in value.split("\n") if line.strip()
            ]
            value = lines[0] if lines else ""
    return value


def _read_safetensors_embedded_meta(p: Path) -> Mapping[str, Any] | None:
    is_required(
        "safetensors",
        "pip install 'enn-torch[safetensors]'  # or: pip install safetensors",
    )
    try:
        from safetensors import safe_open
    except Exception:
        return None
    try:
        with safe_open(str(p), framework="pt", device="cpu") as f:
            md = f.metadata() or {}
    except Exception:
        return None
    if not isinstance(md, dict):
        return None
    raw = md.get("enn_meta_json") or md.get("enn.meta_json") or md.get("enn_meta")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def export_safetensors_single(
    model: torch.nn.Module,
    path: PathLike,
    *,
    meta: Mapping[str, Any] | None = None,
    overwrite: bool = True,
) -> str:
    is_required(
        "safetensors",
        "pip install 'enn-torch[safetensors]'  # or: pip install safetensors",
    )
    from safetensors.torch import save_file as save_tensors

    p = Path(_normalize_windows_paste(path))
    if not p.suffix:
        p = p.with_suffix(".safetensors")
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not bool(overwrite):
        raise FileExistsError(f"Export destination already exists: {str(p)!r}")

    sd = model.state_dict()
    tensors: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        t = v.detach()
        if getattr(t, "device", None) is not None and t.device.type != "cpu":
            t = t.to("cpu", non_blocking=False)
        if not t.is_contiguous():
            t = t.contiguous()
        tensors[str(k)] = t

    md: dict[str, str] | None = None
    if isinstance(meta, Mapping) and meta:
        md = {"enn_meta_json": json.dumps(meta, ensure_ascii=False, separators=(",", ":"))}

    tmp = p.with_name(p.name + ".tmp")
    try:
        save_tensors(tensors, str(tmp), metadata=md)
        os.replace(tmp, p)
    finally:
        with contextlib.suppress(Exception):
            if tmp.exists():
                tmp.unlink()
    return str(p)


def _find_latest_dcp_epoch_dir(ckpt_dir: str | None) -> str | None:
    if not ckpt_dir:
        return None
    dcp_root = os.path.join(ckpt_dir, "dcp_epochs")
    if not os.path.isdir(dcp_root):
        return None
    epoch_dirs: list[str] = []
    for entry in os.scandir(dcp_root):
        if not entry.is_dir():
            continue
        if not entry.name.startswith("epoch_"):
            continue
        done_path = os.path.join(entry.path, ".done")
        failed_path = os.path.join(entry.path, ".failed")
        if os.path.exists(failed_path):
            continue
        if not os.path.isfile(done_path):
            continue
        ok = False
        with contextlib.suppress(Exception):
            raw = Path(done_path).read_text(encoding="utf-8").strip()
            raw = raw.replace("\\n", "\n")
            obj = json.loads(raw) if raw else None
            if isinstance(obj, dict) and obj.get("status") == "ok":
                if obj.get("epoch_dir") in (None, "", entry.name):
                    ok = True
        if not ok:
            continue
        with contextlib.suppress(Exception):
            entries = os.listdir(entry.path)
            non_marker = [n for n in entries if n not in {".done", ".failed"} and not n.startswith(".done.tmp")]
            if len(non_marker) == 0:
                continue
        epoch_dirs.append(entry.path)
    if not epoch_dirs:
        return None
    epoch_dirs.sort()
    return epoch_dirs[-1]


def _resize_scaler_buffers_for_shape(
    model: torch.nn.Module,
    in_dim: int | None,
    label_shape: Sequence[int] | tuple[int, ...],
) -> None:
    if in_dim is None and not label_shape:
        return
    x_numel = int(in_dim) if in_dim is not None else None
    y_numel = int(numpy.prod(label_shape)) if label_shape else None

    def _resize_buffer(
        module: Scaler,
        name: str,
        shape: tuple[int, ...],
        *,
        fill: float | None = None,
    ) -> None:
        buf = getattr(module, name, None)
        if not isinstance(buf, torch.Tensor):
            return
        if tuple(buf.shape) == tuple(shape):
            return
        if fill is not None:
            try:
                module._buffers[name] = buf.detach().new_full(shape, fill_value=float(fill))
            except Exception:
                setattr(module, name, buf.detach().new_full(shape, fill_value=float(fill)))
            return
        try:
            buf.resize_(shape)
        except Exception:
            module._buffers[name] = buf.detach().new_zeros(shape)

    for module in model.modules():
        if not isinstance(module, Scaler):
            continue
        if x_numel and x_numel > 0:
            _resize_buffer(module, "x_mean", (x_numel,))
            _resize_buffer(module, "x_std", (x_numel,))
            _resize_buffer(module, "pw_x", (x_numel,))
        if y_numel and y_numel > 0:
            for name in (
                "y_mean",
                "y_std",
                "y_min",
                "y_max",
                "y_q_low",
                "y_q_high",
                "affine_a",
                "affine_b",
                "pw_y",
            ):
                _resize_buffer(module, name, (y_numel,))
            big32_min = float(torch.finfo(torch.float32).min)
            big32_max = float(torch.finfo(torch.float32).max)
            _resize_buffer(module, "y_out_scale", (y_numel,), fill=1.0)
            _resize_buffer(module, "y_out_bias", (y_numel,), fill=0.0)
            _resize_buffer(module, "y_out_clip_low", (y_numel,), fill=big32_min)
            _resize_buffer(module, "y_out_clip_high", (y_numel,), fill=big32_max)
            with contextlib.suppress(Exception):
                module.output_ab_enabled = False
        continue


def _resize_scaler_buffers_from_metadata(
    model: torch.nn.Module,
    metadata: object | None,
) -> bool:
    if metadata is None:
        return False
    state_meta = getattr(metadata, "state_dict_metadata", None)
    if not isinstance(state_meta, Mapping):
        return False
    suffix_map = {
        "scaler.x_mean": "x_mean",
        "scaler.x_std": "x_std",
        "scaler.y_mean": "y_mean",
        "scaler.y_std": "y_std",
        "scaler.y_min": "y_min",
        "scaler.y_max": "y_max",
        "scaler.y_q_low": "y_q_low",
        "scaler.y_q_high": "y_q_high",
        "scaler.affine_a": "affine_a",
        "scaler.affine_b": "affine_b",
        "scaler.pw_x": "pw_x",
        "scaler.pw_y": "pw_y",
        "scaler.y_out_scale": "y_out_scale",
        "scaler.y_out_bias": "y_out_bias",
        "scaler.y_out_clip_low": "y_out_clip_low",
        "scaler.y_out_clip_high": "y_out_clip_high",
    }

    def _resize_buffer(
        module: Scaler,
        name: str,
        shape: tuple[int, ...],
        *,
        fill: float | None = None,
    ) -> None:
        buf = getattr(module, name, None)
        if not isinstance(buf, torch.Tensor):
            return
        if tuple(buf.shape) == tuple(shape):
            return
        old_shape = tuple(buf.shape)
        if fill is not None:
            try:
                module._buffers[name] = buf.detach().new_full(shape, fill_value=float(fill))
            except Exception:
                setattr(module, name, buf.detach().new_full(shape, fill_value=float(fill)))
            logger.debug("scaler pre-load resize(init): %s %s -> %s", name, old_shape, tuple(shape))
            return
        try:
            buf.resize_(shape)
        except Exception:
            module._buffers[name] = buf.detach().new_zeros(shape)
        logger.debug("scaler pre-load resize: %s %s -> %s", name, old_shape, tuple(shape))

    big32_min = float(torch.finfo(torch.float32).min)
    big32_max = float(torch.finfo(torch.float32).max)
    fill_defaults: dict[str, float] = {
        "y_out_scale": 1.0,
        "y_out_bias": 0.0,
        "y_out_clip_low": big32_min,
        "y_out_clip_high": big32_max,
    }

    resized_any = False
    for module in model.modules():
        if not isinstance(module, Scaler):
            continue
        resized_local = False
        for key, value in state_meta.items():
            k = str(key)
            for suffix, buf_name in suffix_map.items():
                if not k.endswith(suffix):
                    continue
                size = getattr(value, "size", None)
                if size is None:
                    continue
                try:
                    shape = tuple(int(dim) for dim in size)
                except Exception:
                    continue
                _resize_buffer(module, buf_name, shape, fill=fill_defaults.get(buf_name))
                resized_local = True
        if resized_local:
            with contextlib.suppress(Exception):
                module.output_ab_enabled = False
        resized_any = resized_any or resized_local
    return resized_any


def _coerce_scaler_buffers_to_shape(
    model: torch.nn.Module,
    in_dim: int | None,
    label_shape: Sequence[int] | tuple[int, ...],
) -> bool:
    if not env_bool("ENN_SCALER_COMPAT_RESIZE", default=True):
        return False
    if in_dim is None and not label_shape:
        return False

    x_numel = int(in_dim) if in_dim is not None else None
    y_numel = int(numpy.prod(label_shape)) if label_shape else None
    if (not x_numel or x_numel <= 0) and (not y_numel or y_numel <= 0):
        return False

    nonscalar_default = env_bool("ENN_SCALER_COMPAT_RESIZE_NONSCALAR", default=False)
    x_allow_nonscalar = env_bool("ENN_SCALER_COMPAT_RESIZE_X_NONSCALAR", default=nonscalar_default)
    y_allow_nonscalar = env_bool("ENN_SCALER_COMPAT_RESIZE_Y_NONSCALAR", default=nonscalar_default)

    fill_default = (os.environ.get("ENN_SCALER_COMPAT_RESIZE_FILL") or "edge").strip().lower()
    x_fill = (os.environ.get("ENN_SCALER_COMPAT_RESIZE_X_FILL") or fill_default).strip().lower()
    y_fill = (os.environ.get("ENN_SCALER_COMPAT_RESIZE_Y_FILL") or fill_default).strip().lower()
    allowed_fill = {"edge", "mean", "zero", "tile"}
    if x_fill not in allowed_fill:
        x_fill = "edge"
    if y_fill not in allowed_fill:
        y_fill = "edge"

    name_by_module: dict[object, str] = {}
    with contextlib.suppress(Exception):
        name_by_module = {m: n for n, m in model.named_modules()}

    def _qual(module: Scaler, buf_name: str) -> str:
        prefix = name_by_module.get(module, "") or ""
        if prefix:
            return f"{prefix}.{buf_name}"
        return buf_name

    def _make_new_1d_like(src: torch.Tensor, n: int) -> torch.Tensor:
        return src.detach().reshape(-1).new_empty((int(n),))

    def _fill_tail(dst: torch.Tensor, src_flat: torch.Tensor, start: int, mode: str) -> None:
        if start >= dst.numel():
            return
        if mode == "zero":
            dst[start:] = 0
            return
        if src_flat.numel() <= 0:
            dst[start:] = 0
            return
        if mode == "mean":
            dst[start:] = src_flat.mean()
            return
        if mode == "tile":
            reps = int((dst.numel() + src_flat.numel() - 1) // src_flat.numel())
            tiled = src_flat.repeat(reps)[: dst.numel()]
            dst[start:] = tiled[start:]
            return
        dst[start:] = src_flat[-1]

    def _coerce_one(
        module: Scaler,
        name: str,
        target_numel: int,
        *,
        allow_nonscalar: bool,
        fill_mode: str,
    ) -> bool:
        buf = getattr(module, name, None)
        if not isinstance(buf, torch.Tensor):
            return False
        if is_meta_or_fake_tensor(buf):
            return False
        tgt = int(target_numel)
        if tgt <= 0:
            return False
        cur_shape = tuple(buf.shape)
        cur = int(buf.numel())
        if cur == tgt:
            if int(buf.ndim) == 1:
                return False
            try:
                new = buf.detach().reshape(-1).contiguous()
                module._buffers[name] = new
                logger.debug("scaler post-load compat: %s %s -> %s (reshape)", _qual(module, name), cur_shape, tuple(new.shape))
                return True
            except Exception:
                return False

        src_flat = buf.detach().reshape(-1).contiguous()
        if cur == 1 and tgt > 1:
            new = _make_new_1d_like(src_flat, tgt)
            new[:] = src_flat[0]
            module._buffers[name] = new
            logger.debug(
                "scaler post-load compat: %s %s -> %s (scalar->vec fill=scalar)",
                _qual(module, name), cur_shape, tuple(new.shape)
            )
            return True

        if not allow_nonscalar:
            return False

        new = _make_new_1d_like(src_flat, tgt)
        copy_n = min(int(src_flat.numel()), int(new.numel()))
        if copy_n > 0:
            new[:copy_n] = src_flat[:copy_n]
        _fill_tail(new, src_flat, copy_n, fill_mode)
        module._buffers[name] = new
        logger.debug(
            "scaler post-load compat: %s %s -> %s (nonscalar copy=%d fill=%s)",
            _qual(module, name), cur_shape, tuple(new.shape), copy_n, fill_mode
        )
        return True

    changed = False
    x_names = ("x_mean", "x_std", "pw_x")
    y_names = ("y_mean", "y_std", "y_min", "y_max", "y_q_low", "y_q_high", "affine_a", "affine_b", "pw_y")
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, Scaler):
                continue
            if x_numel and x_numel > 0:
                for n in x_names:
                    changed = _coerce_one(
                        module, n, int(x_numel),
                        allow_nonscalar=bool(x_allow_nonscalar),
                        fill_mode=str(x_fill),
                    ) or changed
            if y_numel and y_numel > 0:
                for n in y_names:
                    changed = _coerce_one(
                        module, n, int(y_numel),
                        allow_nonscalar=bool(y_allow_nonscalar),
                        fill_mode=str(y_fill),
                    ) or changed
    return changed


def _validate_scaler_buffers(model: torch.nn.Module, *, strict: bool) -> None:
    if not env_bool("ENN_SCALER_VALIDATE", default=bool(strict)):
        return
    allow_default = env_bool("ENN_SCALER_ALLOW_DEFAULT", default=False)
    log_only = env_bool("ENN_SCALER_VALIDATE_LOG_ONLY", default=False)
    try:
        tol = float(os.environ.get("ENN_SCALER_VALIDATE_TOL", "") or 0.0)
    except Exception:
        tol = 0.0
    if tol <= 0.0:
        tol = 1e-12

    def _amax_abs(t: torch.Tensor) -> float:
        if t.numel() <= 0:
            return 0.0
        try:
            return float(t.detach().abs().amax().item())
        except Exception:
            return float(t.detach().abs().max().item())

    def _amax_abs_diff1(t: torch.Tensor) -> float:
        if t.numel() <= 0:
            return 0.0
        try:
            return float((t.detach() - 1.0).abs().amax().item())
        except Exception:
            return float((t.detach() - 1.0).abs().max().item())

    def _tmin(t: torch.Tensor) -> float:
        if t.numel() <= 0:
            return float("inf")
        return float(t.detach().min().item())

    def _tmax(t: torch.Tensor) -> float:
        if t.numel() <= 0:
            return float("-inf")
        return float(t.detach().max().item())

    for mod_name, mod in model.named_modules():
        if not isinstance(mod, Scaler):
            continue
        x_mean = getattr(mod, "x_mean", None)
        x_std = getattr(mod, "x_std", None)
        y_mean = getattr(mod, "y_mean", None)
        y_std = getattr(mod, "y_std", None)
        if not all(isinstance(t, torch.Tensor) for t in (x_mean, x_std, y_mean, y_std)):
            continue

        issues: list[str] = []
        with torch.no_grad():
            for nm, t in (("x_mean", x_mean), ("x_std", x_std), ("y_mean", y_mean), ("y_std", y_std)):
                if (t.is_floating_point() or t.is_complex()) and t.numel() > 0:
                    try:
                        ok = bool(torch.isfinite(t).all().item())
                    except Exception:
                        ok = True
                    if not ok:
                        issues.append(f"{nm}:nonfinite")
            for nm, t in (("x_std", x_std), ("y_std", y_std)):
                if t.is_floating_point() and t.numel() > 0:
                    mn = _tmin(t)
                    if not (mn > 0.0):
                        issues.append(f"{nm}:min<=0({mn:.3g})")

            default_x = (_amax_abs(x_mean) <= tol) and (_amax_abs_diff1(x_std) <= tol)
            default_y = (_amax_abs(y_mean) <= tol) and (_amax_abs_diff1(y_std) <= tol)
            if default_x and default_y and (not allow_default):
                issues.append("stats_default(zeros/ones)")

            xm = _amax_abs(x_mean)
            xs_min = _tmin(x_std)
            xs_max = _tmax(x_std)
            ym = _amax_abs(y_mean)
            ys_min = _tmin(y_std)
            ys_max = _tmax(y_std)
            cal = str(getattr(mod, "calib_mode", ""))

        if issues:
            loc = mod_name or "<root>"
            msg = (
                f"[ENN] scaler validate: module={loc} calib_mode={cal} issues={issues} "
                f"x_mean_absmax={xm:.6g} x_std[min,max]=[{xs_min:.6g},{xs_max:.6g}] "
                f"y_mean_absmax={ym:.6g} y_std[min,max]=[{ys_min:.6g},{ys_max:.6g}]"
            )
            logger.warning(msg)
            if strict and (not log_only):
                raise RuntimeError(msg)


def _parse_meta(p: PathLike) -> Mapping[str, Any]:
    base = Path(p)
    for d in (base, base.parent, base.parent.parent, base.parent.parent.parent):
        meta_path = d / "meta.json"
        try:
            if meta_path.exists():
                out = read_json(meta_path)
                return out if isinstance(out, Mapping) else {}
        except Exception as exc:
            raise RuntimeError(f"Metadata parse failed: {meta_path}") from exc
    done_path = base / "done.json"
    with contextlib.suppress(Exception):
        if done_path.exists():
            done_json = read_json(done_path)
            if isinstance(done_json, Mapping):
                extra_meta = done_json.get("extra_meta")
                if isinstance(extra_meta, Mapping):
                    return extra_meta
    return {}


def _read_pt_sidecar_meta(p: Path) -> Mapping[str, Any] | None:
    try:
        cand1 = p.with_suffix(".meta.json")
    except Exception:
        cand1 = None
    cand2 = None
    with contextlib.suppress(Exception):
        cand2 = p.with_name(p.name + ".meta.json")
    for cand in (cand1, cand2):
        if cand is None:
            continue
        try:
            if cand.exists():
                meta = read_json(cand)
                if isinstance(meta, Mapping):
                    return meta
        except Exception:
            continue
    return None


def _dcp_strict_load_enabled() -> bool:
    return env_bool(
        ("ENN_DCP_STRICT_LOAD", "ENN_CHECKPOINT_STRICT", "ENN_STRICT_LOAD"),
        default=False,
    )


def _dcp_allowed_keys_and_shapes(
    metadata: object | None,
    *,
    root_prefix: str = "model.",
) -> tuple[set[str], dict[str, tuple[int, ...]]]:
    allowed: set[str] = set()
    shapes: dict[str, tuple[int, ...]] = {}
    if metadata is None:
        return allowed, shapes

    state_meta = getattr(metadata, "state_dict_metadata", None)
    if not isinstance(state_meta, Mapping) or not state_meta:
        return allowed, shapes

    for k, v in state_meta.items():
        key = str(k)
        if root_prefix and key.startswith(root_prefix):
            key = key[len(root_prefix) :]
        allowed.add(key)

        size = getattr(v, "size", None)
        if size is None:
            continue
        try:
            shapes[key] = tuple(int(x) for x in size)
        except Exception:
            continue
    return allowed, shapes


def _dcp_filter_model_state_dict(
    m_sd: Mapping[str, Any],
    *,
    allowed_keys: set[str] | None,
    shapes: Mapping[str, tuple[int, ...]] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    allowed = allowed_keys or set()
    for k, v in m_sd.items():
        kk = str(k)
        if allowed and kk not in allowed:
            continue
        if shapes and kk in shapes and torch.is_tensor(v):
            try:
                if tuple(v.shape) != tuple(shapes[kk]):
                    continue
            except Exception:
                continue
        out[kk] = v
    return out


def _make_dcp_load_planner(*, allow_partial_load: bool) -> object | None:
    Planner = None
    with contextlib.suppress(Exception):
        from torch.distributed.checkpoint import DefaultLoadPlanner as _DLP

        Planner = _DLP
    if Planner is None:
        with contextlib.suppress(Exception):
            from torch.distributed.checkpoint.planner import DefaultLoadPlanner as _DLP2

            Planner = _DLP2
    if Planner is None:
        return None

    kwargs: dict[str, Any] = {}
    if allow_partial_load:
        try:
            sig = inspect.signature(Planner)
            params = getattr(sig, "parameters", None)
            if isinstance(params, Mapping):
                if "allow_partial_load" in params:
                    kwargs["allow_partial_load"] = True
                elif "allow_partial" in params:
                    kwargs["allow_partial"] = True
        except Exception:
            kwargs["allow_partial_load"] = True

    if kwargs:
        try:
            return Planner(**kwargs)
        except Exception:
            pass
    with contextlib.suppress(Exception):
        return Planner()
    return None


def _dcp_load_model_state(
    *,
    reader: FileSystemReader,
    model_state: dict[str, Any],
    planner: object | None,
) -> None:
    with ensure_dcp_process_group(torch.device("cpu")):
        if planner is not None:
            try:
                load(
                    state_dict={"model": model_state},
                    storage_reader=reader,
                    planner=planner,
                )
                return
            except TypeError:
                pass
        load(state_dict={"model": model_state}, storage_reader=reader)


def _raise_if_empty_dcp_model_state(
    model_state: Mapping[str, Any],
    *,
    checkpoint_path: Path,
) -> None:
    if model_state:
        return
    raise RuntimeError(
        "Directory checkpoint load resolved zero model parameters after metadata "
        f"filtering at {str(checkpoint_path)!r}; refusing partial load from an "
        "empty parameter set."
    )


def _try_load_dir_checkpoint_fallback_pt(
    model: torch.nn.Module,
    *,
    pt_path: Path,
    map_location: TorchDeviceLike | None,
    mmap: bool | None,
    rebuild_tasks: bool = False,
) -> bool:
    if not pt_path.is_file():
        return False
    obj = _torch_load_checkpoint(
        pt_path,
        map_location=map_location or "cpu",
        weights_only=True,
        mmap=mmap,
    )

    if rebuild_tasks:
        side_meta: Mapping[str, Any] | None = None
        with contextlib.suppress(Exception):
            side_meta = _read_pt_sidecar_meta(pt_path)
        if isinstance(side_meta, Mapping):
            with contextlib.suppress(Exception):
                tasks = side_meta.get("tasks")
                if tasks:
                    tgt = model
                    wrapped = getattr(model, "module", None)
                    if isinstance(wrapped, torch.nn.Module):
                        tgt = wrapped
                    rebuild = getattr(tgt, "rebuild_tasks_from_specs", None)
                    if callable(rebuild):
                        rebuild(tasks)

    sd = None
    if isinstance(obj, dict):
        for kk in (
            "state_dict",
            "model_state_dict",
            "model",
            "model_sd",
            "weights",
            "model_weights",
        ):
            vv = obj.get(kk)
            if isinstance(vv, dict):
                sd = vv
                break
        if sd is None:
            try:
                if obj and all(
                    isinstance(k, str) and torch.is_tensor(v)
                    for k, v in obj.items()
                ):
                    sd = obj
            except Exception:
                sd = None
    else:
        sd = obj

    if not isinstance(sd, dict):
        return False
    sd = _coerce_state_dict(sd)
    sd = _drop_runtime_only_state_keys(sd)
    with contextlib.suppress(Exception):
        resize_scaler_buffer(model, sd)
    if _model_has_meta_or_fake_tensors(model):
        _materialize_module_to_device(model, map_location or "cpu")
    with contextlib.suppress(Exception):
        buf = getattr(model, "output_baked_flag", None)
        if torch.is_tensor(buf):
            buf.zero_()

    sd_for_load: object = sd
    if isinstance(sd, Mapping) and env_bool("ENN_LOAD_ALIAS_PERCEIVER_KEYS", default=True):
        try:
            sd_map = dict(sd)
            model_names = [str(n) for n, _ in model.named_parameters(recurse=True)]
            model_keys = set(model_names)

            proc_root = "fuser"
            if any(k.startswith("fuser.perceiver.") for k in model_keys):
                proc_root = "fuser"
            elif any(k.startswith("processor.perceiver.") for k in model_keys):
                proc_root = "processor"

            ctrl_root = "temporal_token_collector"
            if any(k.startswith("temporal_token_collector.") for k in model_keys):
                ctrl_root = "temporal_token_collector"
            elif any(k.startswith("controller.") for k in model_keys):
                ctrl_root = "controller"

            perceiver_plain_prefix = f"{proc_root}.perceiver."
            perceiver_orig_prefix = f"{proc_root}.perceiver._orig_mod."
            has_perceiver_origmod = any(k.startswith(perceiver_orig_prefix) for k in model_keys)
            has_perceiver_plain = any(
                k.startswith(perceiver_plain_prefix) and (not k.startswith(perceiver_orig_prefix))
                for k in model_keys
            )

            def _alias_key(k: str) -> str:
                kk = str(k)
                kk = kk.replace("._enn_inner._orig_mod", "")
                kk = kk.replace("._enn_inner", "")

                if proc_root == "fuser" and kk.startswith("processor."):
                    kk = "fuser." + kk[len("processor.") :]
                elif proc_root == "processor" and kk.startswith("fuser."):
                    kk = "processor." + kk[len("fuser.") :]

                if ctrl_root == "temporal_token_collector" and kk.startswith("controller."):
                    kk = "temporal_token_collector." + kk[len("controller.") :]
                elif ctrl_root == "controller" and kk.startswith("temporal_token_collector."):
                    kk = "controller." + kk[len("temporal_token_collector.") :]

                if has_perceiver_origmod and kk.startswith(perceiver_plain_prefix) and (not kk.startswith(perceiver_orig_prefix)):
                    kk = perceiver_orig_prefix + kk[len(perceiver_plain_prefix) :]
                elif has_perceiver_plain and kk.startswith(perceiver_orig_prefix):
                    kk = perceiver_plain_prefix + kk[len(perceiver_orig_prefix) :]
                return kk

            added = 0
            examples: list[str] = []
            try:
                ex_lim = int(os.environ.get("ENN_LOAD_ALIAS_PERCEIVER_LOG_LIMIT", "10") or "10")
            except Exception:
                ex_lim = 10
            for k, v in list(sd_map.items()):
                if not isinstance(k, str):
                    continue
                kk = _alias_key(k)
                if kk != k and kk in model_keys and kk not in sd_map:
                    sd_map[kk] = v
                    added += 1
                    if len(examples) < int(ex_lim):
                        examples.append(f"{k} -> {kk}")

            if added > 0 and env_bool("ENN_LOAD_ALIAS_PERCEIVER_LOG", default=False):
                msg = ""
                if examples:
                    msg = " Example: " + "; ".join(examples[:10])
                logger.warning(
                    "[ENN] load_weights: added %d aliased keys for perceiver wrapper compatibility.%s",
                    int(added),
                    msg,
                )
            sd_for_load = sd_map
        except Exception:
            sd_for_load = sd

    eager_ctx = getattr(model, "eager_for_export", None)
    cm = eager_ctx() if callable(eager_ctx) else contextlib.nullcontext()
    with cm:
        incompat = _load_state_dict_compat(model, sd_for_load, strict=False)
    warn_incompat = env_bool(
        "ENN_LOAD_ALIAS_PERCEIVER_WARN_MISSING",
        default=env_bool("ENN_SANITIZE_NAN_STRICT", default=False),
    )
    fail_incompat = env_bool(
        "ENN_LOAD_FAIL_ON_INCOMPAT",
        default=env_bool("ENN_SANITIZE_NAN_STRICT", default=False),
    )
    if (warn_incompat or fail_incompat) and incompat is not None:
        miss = getattr(incompat, "missing_keys", None) or []
        unexp = getattr(incompat, "unexpected_keys", None) or []
        if miss or unexp:
            miss_p = [k for k in miss if str(k).startswith("fuser.perceiver.") or str(k).startswith("processor.perceiver.")]
            sm_miss = (miss_p[:10] if miss_p else miss[:10])
            sm_unexp = unexp[:10]
            try:
                sm_miss_s = ", ".join(str(x) for x in sm_miss)
            except Exception:
                sm_miss_s = str(sm_miss)
            try:
                sm_unexp_s = ", ".join(str(x) for x in sm_unexp)
            except Exception:
                sm_unexp_s = str(sm_unexp)
            logger.warning(
                "[ENN] load_weights: state_dict incompat: missing=%d (perceiver=%d), unexpected=%d. "
                "missing_example=%s; unexpected_example=%s",
                int(len(miss)),
                int(len(miss_p)),
                int(len(unexp)),
                sm_miss_s,
                sm_unexp_s,
            )
            if fail_incompat:
                raise RuntimeError(
                    f"[ENN] load_weights: state_dict incompat (missing={len(miss)}, unexpected={len(unexp)}); "
                    "set ENN_LOAD_FAIL_ON_INCOMPAT=0 to ignore."
                )
    with contextlib.suppress(Exception):
        _coerce_scaler_buffers_to_shape(
            model,
            getattr(model, "in_dim", None),
            getattr(model, "out_shape", ()) or (),
        )
    _validate_scaler_buffers(model, strict=env_bool("ENN_SANITIZE_NAN_STRICT", default=False))
    return True


def _is_execution_time_logged() -> bool:
    return env_bool(
        ("ENN_LOG_TIMINGS", "ENN_TIMINGS", "ENN_DEBUG_TIMINGS"),
        default=False,
    )


def _timed_invoke(
    fn: Callable[P, R],
    log: logging.Logger,
    fn_name: str,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    if not _is_execution_time_logged():
        return fn(*args, **kwargs)
    t0 = time.perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        dt = time.perf_counter() - t0
        log.info("%s executed in %.3f seconds", fn_name, dt)


def _clear_device_caches() -> None:
    with contextlib.suppress(Exception):
        import gc

        gc.collect()
    with contextlib.suppress(Exception):
        from ..core.system import collect_accelerator_ipc
        from ..core.system import empty_device_cache
        from ..core.system import get_device

        empty_device_cache(
            device=get_device(), do_gc=False, min_interval_s=0.0
        )
        collect_accelerator_ipc()


def _coerce_seed(seed: int) -> Optional[int]:
    try:
        return int(seed) if seed is not None else None
    except (TypeError, ValueError):
        return None




def _model_has_meta_or_fake_tensors(m: torch.nn.Module) -> bool:
    try:
        for t in m.parameters(recurse=True):
            if is_meta_or_fake_tensor(t):
                return True
        for t in m.buffers(recurse=True):
            if is_meta_or_fake_tensor(t):
                return True
    except Exception:
        return False
    return False


def _materialize_module_to_device(m: torch.nn.Module, device: object = "cpu") -> None:
    dev = device
    try:
        if not isinstance(dev, torch.device):
            dev = torch.device(dev)
    except Exception:
        dev = torch.device("cpu")
    if hasattr(m, "to_empty"):
        try:
            m.to_empty(device=dev)
            return
        except TypeError:
            m.to_empty(device=dev)
            return
    raise RuntimeError(
        "Model contains meta/fake tensors but Module.to_empty() is unavailable. "
        "Upgrade PyTorch or avoid moving the parent model to meta before load."
    )


def _load_state_dict_compat(m: torch.nn.Module, sd: Mapping[str, Any], *, strict: bool) -> object:
    strict_b = bool(strict)
    if _model_has_meta_or_fake_tensors(m):
        try:
            return m.load_state_dict(sd, strict=strict_b, assign=True)
        except TypeError:
            pass
    return m.load_state_dict(sd, strict=strict_b)


def _save_model_checkpoint(
    model: Model,
    out_dir: PathLike,
    *args: Any,
    save_dcp: bool,
    save_pt: bool,
    overwrite: bool = True,
) -> Mapping[str, Any] | None:
    m_sd = None
    if not (save_dcp or save_pt):
        return None
    os.makedirs(out_dir, exist_ok=True)

    def _first_nonfinite_state_tensor(
        m: torch.nn.Module,
    ) -> tuple[str | None, torch.Tensor | None]:
        allow_scaler_inf = env_bool("ENN_SAVE_ALLOW_SCALER_INF", default=True)
        allow_logger_inf = env_bool("ENN_SAVE_ALLOW_LOGGER_INF", default=True)
        scaler_inf_suffixes = (
            "scaler.y_min",
            "scaler.y_max",
            "scaler.y_q_low",
            "scaler.y_q_high",
        )

        def _allowed_nonfinite(name: str, t: torch.Tensor) -> bool:
            if allow_scaler_inf and any(str(name).endswith(suf) for suf in scaler_inf_suffixes):
                with torch.no_grad():
                    if bool(torch.isnan(t).any().item()):
                        return False
                    return bool(torch.isinf(t).all().item())
            if allow_logger_inf:
                n = str(name)
                if (("logger.sampled_" in n) or ("logger.reduced_" in n)) and (
                    n.endswith("_min") or n.endswith("_max")
                ):
                    with torch.no_grad():
                        if t.numel() <= 16:
                            if bool(torch.isnan(t).any().item()):
                                return False
                            return bool(torch.isinf(t).all().item())
            return False

        def _scan_named_tensors(
            named_tensors: Iterable[tuple[str, torch.Tensor]],
        ) -> tuple[str | None, torch.Tensor | None]:
            for k, v in named_tensors:
                if not isinstance(v, torch.Tensor):
                    continue
                if (not v.is_floating_point()) and (not v.is_complex()):
                    continue
                if v.numel() <= 0:
                    continue
                try:
                    a = v.detach().abs().amax()
                    ok = bool(torch.isfinite(a).item())
                except Exception:
                    ok = bool(torch.isfinite(v).all().item())
                if not ok:
                    if _allowed_nonfinite(str(k), v):
                        continue
                    return str(k), v
            return None, None

        bad_k, bad_t = _scan_named_tensors(m.named_parameters(recurse=True))
        if bad_k is not None:
            return bad_k, bad_t
        bad_k, bad_t = _scan_named_tensors(m.named_buffers(recurse=True))
        if bad_k is not None:
            return bad_k, bad_t
        return None, None

    if save_dcp:
        dcp_cpu_offload = env_bool("ENN_DCP_CPU_OFFLOAD", default=False)
        dcp_sync_files = env_bool("ENN_DCP_SYNC_FILES", default=True)
        dump_dir = str(os.environ.get("ENN_NONFINITE_DUMP_DIR", "") or "").strip()
        strict_save = env_bool(
            "ENN_FAIL_ON_SAVE_NONFINITE",
            default=env_bool("ENN_SANITIZE_NAN_STRICT", default=False),
        )

        if dump_dir or strict_save:
            bad_k, bad_t = _first_nonfinite_state_tensor(model)
            if bad_k is not None:
                logger.error(
                    "[ENN] save_checkpoint: non-finite model state tensor detected before save_dcp: %s",
                    str(bad_k),
                )
                if dump_dir:
                    with contextlib.suppress(Exception):
                        os.makedirs(dump_dir, exist_ok=True)
                        rid = os.urandom(4).hex()
                        rank = str(os.environ.get("RANK", "0") or "0")
                        path = os.path.join(
                            dump_dir, f"save_nonfinite.dcp.rank{rank}.{rid}.pt"
                        )
                        payload = {
                            "where": "save_dcp_state",
                            "first_bad": str(bad_k),
                        }
                        if isinstance(bad_t, torch.Tensor):
                            with contextlib.suppress(Exception):
                                payload["shape"] = [int(x) for x in tuple(bad_t.shape)]
                                payload["dtype"] = str(bad_t.dtype)
                                payload["device"] = str(bad_t.device)
                        torch.save(payload, path)
                        logger.error("[ENN] save_checkpoint: dumped to: %s", str(path))
                if strict_save:
                    raise RuntimeError(
                        "[ENN] save_checkpoint: non-finite model state tensor detected before "
                        f"save_dcp: {bad_k}"
                    )

        with _filtered_warnings():
            m_sd = get_model_state_dict(
                model,
                options=StateDictOptions(
                    full_state_dict=False, cpu_offload=bool(dcp_cpu_offload)
                ),
            )
            use_collectives = False
            with contextlib.suppress(Exception):
                import torch.distributed as _dist
                if _dist.is_available() and _dist.is_initialized():
                    use_collectives = int(_dist.get_world_size()) > 1
            with ensure_dcp_process_group(torch.device("cpu")):
                try:
                    save(
                        state_dict={"model": m_sd},
                        storage_writer=FileSystemWriter(
                            out_dir,
                            sync_files=bool(dcp_sync_files),
                            overwrite=bool(overwrite),
                        ),
                        use_collectives=bool(use_collectives),
                    )
                except TypeError:
                    save(
                        state_dict={"model": m_sd},
                        storage_writer=FileSystemWriter(
                            out_dir,
                            sync_files=bool(dcp_sync_files),
                            overwrite=bool(overwrite),
                        ),
                    )
    if save_pt:
        eager_ctx = getattr(model, "eager_for_export", None)
        cm = eager_ctx() if callable(eager_ctx) else contextlib.nullcontext()
        with cm:
            with _filtered_warnings():
                pt_state = dict(model.state_dict())
        _coerce_dcp_keys(pt_state)
        if any(
            torch.is_tensor(v) and is_meta_or_fake_tensor(v)
            for v in pt_state.values()
        ):
            raise NotImplementedError(
                "Cannot save checkpoint with fake/meta tensors (no data)."
            )
        pt_state = {
            k: (v.detach().to(device="cpu") if torch.is_tensor(v) else v)
            for k, v in pt_state.items()
        }
        dump_dir = str(os.environ.get("ENN_NONFINITE_DUMP_DIR", "") or "").strip()
        strict_save = env_bool(
            "ENN_FAIL_ON_SAVE_NONFINITE",
            default=env_bool("ENN_SANITIZE_NAN_STRICT", default=False),
        )
        if dump_dir or strict_save:
            bad_k, bad_t = _first_nonfinite_state_tensor(model)
            if bad_k is not None:
                logger.error(
                    "[ENN] save_checkpoint: non-finite model state tensor detected before save_pt: %s",
                    str(bad_k),
                )
                if dump_dir:
                    with contextlib.suppress(Exception):
                        os.makedirs(dump_dir, exist_ok=True)
                        rid = os.urandom(4).hex()
                        rank = str(os.environ.get("RANK", "0") or "0")
                        path = os.path.join(
                            dump_dir, f"save_nonfinite.pt.rank{rank}.{rid}.pt"
                        )
                        payload = {"where": "save_pt_state", "first_bad": str(bad_k)}
                        if isinstance(bad_t, torch.Tensor):
                            with contextlib.suppress(Exception):
                                payload["shape"] = [int(x) for x in tuple(bad_t.shape)]
                                payload["dtype"] = str(bad_t.dtype)
                                payload["device"] = str(bad_t.device)
                        torch.save(payload, path)
                        logger.error("[ENN] save_checkpoint: dumped to: %s", str(path))
                if strict_save:
                    raise RuntimeError(
                        "[ENN] save_checkpoint: non-finite model state tensor detected before "
                        f"save_pt: {bad_k}"
                    )
        torch.save(pt_state, os.path.join(out_dir, "model.pt"))
        with contextlib.suppress(Exception):
            base = model
            wrapped = getattr(model, "module", None)
            if isinstance(wrapped, torch.nn.Module):
                base = wrapped
            meta_payload: dict[str, Any] = {
                "format": "enn-model-meta-v1",
                "created_time": float(time.time()),
            }
            in_dim0 = getattr(base, "in_dim", None)
            out_shape0 = getattr(base, "out_shape", None)
            if in_dim0 is not None:
                with contextlib.suppress(Exception):
                    meta_payload["in_dim"] = int(in_dim0)
            if out_shape0 is not None:
                with contextlib.suppress(Exception):
                    meta_payload["out_shape"] = [int(x) for x in tuple(out_shape0)]
            cfg0 = _extract_model_config_dict(base)
            if isinstance(cfg0, dict) and cfg0:
                meta_payload["config"] = cfg0
            ts_fn = getattr(base, "task_specs", None)
            if callable(ts_fn):
                with contextlib.suppress(Exception):
                    meta_payload["tasks"] = ts_fn()
            collate.write_json(
                os.path.join(out_dir, "model.meta.json"),
                meta_payload,
                indent=2,
            )
    return m_sd


def _is_wrapped_or_distributed_model(model: torch.nn.Module) -> bool:
    wrapped = getattr(model, "module", None)
    if isinstance(wrapped, torch.nn.Module):
        return True
    with contextlib.suppress(Exception):
        import torch.distributed as _dist

        if _dist.is_available() and _dist.is_initialized():
            return int(_dist.get_world_size()) > 1
    return False


def _get_label_shape(
    first_in_dim: int,
    in_dim: int,
    first_label_shape: object,
    lshape: object,
) -> Tuple[Any]:
    if first_in_dim is None:
        return (int(in_dim), tuple(lshape))
    if int(in_dim) != int(first_in_dim) or tuple(lshape) != tuple(
        first_label_shape
    ):
        raise RuntimeError(
            f"Shape mismatch: {first_in_dim}/{first_label_shape} vs {in_dim}/{lshape}"
        )
    return (first_in_dim, first_label_shape)


def _adapt_source(
    d: Any, allow_columns: bool = True
) -> Tuple[int, Optional[Callable], bool]:
    def _value_len(value: object) -> Optional[int]:
        if isinstance(value, (str, bytes, bytearray)):
            return None
        if isinstance(value, Mapping) and not isinstance(
            value, TensorDictBase
        ):
            return None
        if hasattr(value, "__len__"):
            with contextlib.suppress(Exception):
                return int(len(value))
        with contextlib.suppress(Exception):
            return int(value.shape[0])
        return None

    if isinstance(d, TensorDictBase):
        if not d.batch_size:
            raise ValueError("TensorDict must have batch dimension")
        return int(d.batch_size[0]), TensorDictSlicer(d), False
    if (
        allow_columns
        and isinstance(d, Mapping)
        and all(not isinstance(v, Mapping) for v in d.values())
        and not collate.is_feature_label_batch_mapping(d)
    ):
        count, getter = collate.column_cursor(d)
        return count, getter, False
    if isinstance(d, Mapping) and collate.is_feature_label_batch_mapping(d):
        f_key = collate.get_feature_key(d)
        if f_key is None:
            return 0, None, True
        count = _value_len(d.get(f_key))
        if count is None:
            return 0, None, True
        constants: dict[object, object] = {}
        slices = []
        for key, value in d.items():
            item_len = _value_len(value)
            if item_len is None or item_len != count:
                constants[key] = value
            else:
                slices.append((key, value))
        slices_t = tuple(slices)
        constants = {
            key: value
            for key, value in constants.items()
            if key not in dict(slices_t)
        }
        return count, MappingSlicer(constants, slices_t), False
    if isinstance(d, (list, tuple)):
        return len(d), (lambda s, e: {"features": d[int(s) : int(e)]}), False
    return 0, None, True


def _save_dataset(
    d: object,
    out_dir: PathLike,
    *args: Any,
    ds: object,
    val_frac: object,
    seed_value: int,
    underflow_action: object,
    shuffle: object,
) -> Tuple[int, Tuple[Any], int]:
    count, getter, needs_preprocess = _adapt_source(d, allow_columns=True)
    if not needs_preprocess:
        if count <= 0:
            raise ValueError("Empty dataset")
        shuffle_enabled = bool(shuffle)
        get_by_indices = None
        if shuffle_enabled:
            if isinstance(getter, TensorDictSlicer):
                td = getter.td

                def _td_indexer(indices: object) -> TensorDictBase:
                    return td[indices]

                get_by_indices = _td_indexer
            elif isinstance(getter, MappingSlicer):

                def _can_index(value: object) -> bool:
                    if isinstance(value, (list, tuple)):
                        return True
                    try:
                        value[[0]]
                    except Exception:
                        return False
                    return True

                if not all(_can_index(v) for _, v in getter.slice_items):
                    shuffle_enabled = False
                else:
                    const_items = dict(getter.const_items)
                    slice_items = tuple(getter.slice_items)

                    def _map_indexer(indices: object) -> Mapping[Any, Any]:
                        batch = dict(const_items)
                        idx_list = None
                        if isinstance(indices, torch.Tensor):
                            idx_list = indices.tolist()
                        for k, v in slice_items:
                            if isinstance(v, (list, tuple)):
                                if idx_list is None:
                                    idx_list = list(indices)
                                batch[k] = [v[i] for i in idx_list]
                            else:
                                batch[k] = v[indices]
                        return batch

                    get_by_indices = _map_indexer
        if shuffle_enabled and get_by_indices is None:
            shuffle_enabled = False
        in_dim, label_shape = collate.stream_memmap(
            ds=ds,
            out_dir=out_dir,
            count=count,
            get_batch=getter,
            val_frac=float(val_frac),
            seed_value=seed_value,
            underflow_action=underflow_action,
            shuffle=shuffle_enabled,
            get_by_indices=get_by_indices,
            allow_missing_labels=False,
            chunk_size=0,
        )
        return (int(in_dim), tuple(label_shape), int(count))
    fx, lb, _, lshape = ds.preprocess(d, return_keys=False)
    if not fx.is_contiguous():
        fx = fx.contiguous()
    if lb is None:
        raise ValueError("Labels required")
    preload_memmap(
        {"features": fx, "labels": lb},
        memmap_dir=out_dir,
        val_frac=float(val_frac),
        shuffle=bool(shuffle),
        seed=seed_value,
        underflow_action=underflow_action,
    )
    count = int(fx.shape[0])
    in_dim = int(fx.reshape(count, -1).shape[1])
    return (int(in_dim), tuple(lshape), int(count))


def _reduce_batch_stats(recs: object) -> Optional[Mapping[str, Any]]:
    if not isinstance(recs, list) or not recs:
        return None
    last = recs[-1]
    if isinstance(last, Mapping):
        rxm = last.get("reduced_x_mean")
        if rxm is not None and last.get("reduced_y_var") is not None:
            return {
                f"sampled_{k}_{s}": float(
                    last.get(
                        f"reduced_{k}_{s}",
                        (
                            0.0
                            if s in ("mean", "var")
                            else float("inf") if s == "min" else float("-inf")
                        ),
                    )
                )
                for k in ("x", "y")
                for s in ("mean", "var", "min", "max")
            }
    sums = {k: 0.0 for k in ("bs", "x", "x2", "y", "y2")}
    ext = {
        f"{axis}_{suffix}": float("inf") if suffix == "min" else float("-inf")
        for axis in ("x", "y")
        for suffix in ("min", "max")
    }
    for r in recs:
        if not isinstance(r, Mapping):
            continue
        bs = int(r.get("batch_size", 0))
        if bs <= 0:
            continue
        sums["bs"] += bs
        for axis in ("x", "y"):
            mean = float(r.get(f"batch_{axis}_mean", 0.0))
            var = float(r.get(f"batch_{axis}_var", 0.0))
            sums[axis] += mean * bs
            sums[f"{axis}2"] += (var + mean * mean) * bs
            ext[f"{axis}_min"] = min(
                ext[f"{axis}_min"],
                float(r.get(f"batch_{axis}_min", float("inf"))),
            )
            ext[f"{axis}_max"] = max(
                ext[f"{axis}_max"],
                float(r.get(f"batch_{axis}_max", float("-inf"))),
            )
    if sums["bs"] <= 0:
        return None
    out: dict[str, float] = {}
    for axis in ("x", "y"):
        mean = sums[axis] / sums["bs"]
        out.update(
            {
                f"sampled_{axis}_mean": mean,
                f"sampled_{axis}_var": max(
                    0.0, sums[f"{axis}2"] / sums["bs"] - mean * mean
                ),
                f"sampled_{axis}_min": ext[f"{axis}_min"],
                f"sampled_{axis}_max": ext[f"{axis}_max"],
            }
        )
    return out


def _update_batch_stats(
    prev: object, n_prev: object, inc: object, n_inc: object
) -> Any:
    if inc is None or n_inc <= 0:
        return prev
    if prev is None or n_prev <= 0:
        return {
            f"reduced_{key[len('sampled_') :]}": float(val)
            for key, val in inc.items()
            if key.startswith("sampled_")
        }
    out = {}
    for axis in ("x", "y"):

        def _get(
            dct: Mapping[str, Any], prefix: str, suffix: str, default: float
        ) -> float:
            return float(dct.get(f"{prefix}_{axis}_{suffix}", default))

        m_prev = _get(prev, "reduced", "mean", 0.0)
        v_prev = _get(prev, "reduced", "var", 0.0)
        m_inc = _get(inc, "sampled", "mean", 0.0)
        v_inc = _get(inc, "sampled", "var", 0.0)
        n_new = n_prev + n_inc
        m_new = (m_prev * n_prev + m_inc * n_inc) / n_new
        v_new = max(
            0.0,
            (
                (v_prev + m_prev * m_prev) * n_prev
                + (v_inc + m_inc * m_inc) * n_inc
            )
            / n_new
            - m_new * m_new,
        )
        out.update(
            {
                f"reduced_{axis}_mean": m_new,
                f"reduced_{axis}_var": v_new,
                f"reduced_{axis}_min": min(
                    _get(prev, "reduced", "min", float("inf")),
                    _get(inc, "sampled", "min", float("inf")),
                ),
                f"reduced_{axis}_max": max(
                    _get(prev, "reduced", "max", float("-inf")),
                    _get(inc, "sampled", "max", float("-inf")),
                ),
            }
        )
    return out


def _update_history(
    model: object,
    ckpt_dir: str | None,
    epochs: int,
    val_frac: float,
    num_samples_dataset: int,
    train_device: str | None = None,
) -> None:
    if not ckpt_dir:
        return
    history_path = os.path.join(ckpt_dir, "history.json")
    if not os.path.isfile(history_path):
        return
    try:
        raw = read_json(history_path)
        if isinstance(raw, dict):
            records = raw.get("records", []) or []
            meta = raw.get("meta", {}) or {}
        else:
            records = raw if isinstance(raw, list) else []
            meta = {}
        run_stats = _reduce_batch_stats(records)
        epochs_val = (
            int(meta.get("epochs", epochs))
            if isinstance(meta, dict)
            else int(epochs)
        )
        frac_val = (
            float(meta.get("val_frac", val_frac))
            if isinstance(meta, dict)
            else float(val_frac)
        )
        sampled_n = (
            int(meta.get("sampled_n", 0)) if isinstance(meta, dict) else 0
        )
        train_split_n_est = int(
            round(num_samples_dataset * max(0.0, 1.0 - frac_val))
        )
        sampled_n_est = int(round(train_split_n_est * max(1, epochs_val)))
        if sampled_n <= 0:
            frac_val = max(0.0, min(1.0, frac_val))
            sampled_n = (
                int(
                    round(
                        num_samples_dataset
                        * max(0.0, 1.0 - frac_val)
                        * max(1, epochs_val)
                    )
                )
                or num_samples_dataset
            )
        prev_n = int(getattr(model, "_history_total_samples", 0))
        cum_stats = _update_batch_stats(
            getattr(model, "_history_cum_stats", None),
            prev_n,
            run_stats,
            sampled_n,
        )
        setattr(model, "_history_total_samples", prev_n + sampled_n)
        setattr(model, "_history_dataset_n", int(num_samples_dataset))
        if cum_stats:
            setattr(model, "_history_cum_stats", cum_stats)
        history = getattr(model, "_train_history", []) or []
        model_dev_str: str | None = None
        with contextlib.suppress(Exception):
            import torch as _torch

            params = list(getattr(model, "parameters", lambda: [])())
            if params:
                model_dev_str = str(params[0].device)
            else:
                bufs = list(getattr(model, "buffers", lambda: [])())
                if bufs:
                    model_dev_str = str(bufs[0].device)
        train_dev_str: str | None = None
        if train_device is not None:
            with contextlib.suppress(Exception):
                train_dev_str = str(train_device)
        if not train_dev_str:
            with contextlib.suppress(Exception):
                from ..core.system import get_device as _get_device

                train_dev_str = str(_get_device())
        if not train_dev_str:
            train_dev_str = model_dev_str
        posix_time = None
        with contextlib.suppress(Exception):
            posix_time = float(time.time())
        record = {
            "run_index": len(history),
            "posix_time": posix_time,
            "epochs": int(epochs_val),
            "device": train_dev_str,
            "dataset_n": int(num_samples_dataset),
            "val_frac": float(frac_val),
            "train_split_n_est": int(train_split_n_est),
            "sampled_n_est": int(sampled_n_est),
            "sampled_n": sampled_n,
            "reduced_n": prev_n + sampled_n,
            **(run_stats or {}),
            **(cum_stats or {}),
        }
        if (
            model_dev_str
            and train_dev_str
            and str(model_dev_str) != str(train_dev_str)
        ):
            record["model_device"] = model_dev_str
        env_meta: Dict[str, Any] = dict(meta) if isinstance(meta, dict) else {}
        for k in (
            "epochs",
            "val_frac",
            "dataset_n",
            "train_split_n_est",
            "sampled_n_est",
            "device",
            "posix_time",
        ):
            env_meta.pop(k, None)
        with contextlib.suppress(Exception):
            import sys as _sys

            env_meta.setdefault("python", _sys.version.split()[0])
        with contextlib.suppress(Exception):
            import torch as _torch

            env_meta.setdefault("torch", getattr(_torch, "__version__", None))
            env_meta.setdefault(
                "cuda", getattr(getattr(_torch, "version", None), "cuda", None)
            )
        if env_meta:
            record["env"] = env_meta
        setattr(model, "_train_history", history + [record])
        logger_obj = getattr(model, "logger", None)
        if isinstance(logger_obj, Recorder):
            logger_obj._records = getattr(model, "_train_history")
    except Exception:
        pass


def _to_torch_dtype(dt: object) -> Optional[torch.dtype]:
    try:
        if isinstance(dt, torch.dtype):
            return dt
        if dt is None:
            return None
        ndt = numpy.dtype(dt)
        if ndt == numpy.float64:
            return torch.float64
        if ndt == numpy.float32:
            return torch.float32
        if ndt == numpy.float16:
            return torch.float16
    except Exception:
        return None
    return None


def _get_float_precision(obj: object) -> torch.dtype:
    try:
        if TensorDictBase is not None and isinstance(obj, TensorDictBase):
            X_td, _ = collate.get_row(obj, labels_required=False)
            if X_td is None:
                return torch.float32
            if not bool(torch.is_tensor(X_td)):
                X_td = torch.as_tensor(X_td)
            dt = _to_torch_dtype(getattr(X_td, "dtype", None))
            return torch.float64 if dt == torch.float64 else torch.float32
        if isinstance(obj, Mapping) and collate.is_feature_label_batch_mapping(
            obj
        ):
            f_key = collate.get_feature_key(obj)
            if f_key is not None and f_key in obj:
                X_all = obj.get(f_key)
                dt = _to_torch_dtype(getattr(X_all, "dtype", None))
                if dt is not None:
                    return (
                        torch.float64 if dt == torch.float64 else torch.float32
                    )
                if isinstance(X_all, (list, tuple)) and X_all:
                    dt0 = _to_torch_dtype(getattr(X_all[0], "dtype", None))
                    return (
                        torch.float64
                        if dt0 == torch.float64
                        else torch.float32
                    )
        if torch.is_tensor(obj):
            dt = _to_torch_dtype(getattr(obj, "dtype", None))
            return torch.float64 if dt == torch.float64 else torch.float32
        if isinstance(obj, numpy.ndarray):
            dt = _to_torch_dtype(getattr(obj, "dtype", None))
            return torch.float64 if dt == torch.float64 else torch.float32
    except Exception:
        pass
    return torch.float32


def get_execution_time(
    log: logging.Logger,
    fn_name: str = "",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def _decorator(fn: Callable[P, R]) -> Callable[P, R]:
        name = fn_name or getattr(fn, "__name__", "call")
        wrapped = partial(_timed_invoke, fn, log, str(name))
        update_wrapper(wrapped, fn)
        return cast(Callable[P, R], wrapped)

    return _decorator


def new_model(
    in_dim: int,
    out_shape: Sequence[int],
    config: ModelConfig | Mapping[str, object] | None,
) -> Model:
    cfg = coerce_model_config(config)
    core = Model(in_dim, tuple((int(x) for x in out_shape)), config=cfg)
    return core


def load_weights(
    model: Model,
    checkpoint_path: PathLike,
    *,
    map_location: TorchDeviceLike | None = None,
    weights_only: bool = True,
    mmap: bool | None = True,
    rebuild_tasks: bool = False,
) -> Mapping[str, Any] | None:
    p = Path(_normalize_windows_paste(checkpoint_path))

    side_meta: Mapping[str, Any] | None = None
    with contextlib.suppress(Exception):
        if p.is_file() and p.suffix.lower() in (".pt", ".pth"):
            side_meta = _read_pt_sidecar_meta(p)

    if p.is_dir():
        if _model_has_meta_or_fake_tensors(model):
            _materialize_module_to_device(model, map_location or "cpu")
        meta = _parse_meta(p)
        if rebuild_tasks and isinstance(meta, dict):
            with contextlib.suppress(Exception):
                tasks = meta.get("tasks")
                if tasks:
                    model.rebuild_tasks_from_specs(tasks)
        reader = FileSystemReader(str(p))
        meta_data = None
        with ensure_dcp_process_group(torch.device("cpu")):
            with contextlib.suppress(Exception):
                meta_data = reader.read_metadata()
        if not _resize_scaler_buffers_from_metadata(model, meta_data):
            _resize_scaler_buffers_for_shape(
                model,
                getattr(model, "in_dim", None),
                getattr(model, "out_shape", ()) or (),
            )
        opts = StateDictOptions(full_state_dict=False)
        eager_ctx = getattr(model, "eager_for_export", None)
        cm = eager_ctx() if callable(eager_ctx) else contextlib.nullcontext()
        with cm:
            strict = _dcp_strict_load_enabled()
            m_sd = dict(get_model_state_dict(model, options=opts))
            _coerce_dcp_keys(m_sd)
            if not strict:
                allowed, shapes = _dcp_allowed_keys_and_shapes(meta_data)
                if allowed:
                    m_sd = _dcp_filter_model_state_dict(
                        m_sd, allowed_keys=allowed, shapes=shapes
                    )
            planner = _make_dcp_load_planner(allow_partial_load=not strict)
            try:
                if not strict:
                    _raise_if_empty_dcp_model_state(m_sd, checkpoint_path=p)
                _dcp_load_model_state(reader=reader, model_state=m_sd, planner=planner)
            except Exception:
                if (not strict) and _try_load_dir_checkpoint_fallback_pt(
                    model,
                    pt_path=(p / "model.pt"),
                    map_location=map_location,
                    mmap=mmap,
                    rebuild_tasks=bool(rebuild_tasks),
                ):
                    return meta if isinstance(meta, dict) else None
                raise
            resize_scaler_buffer(model, m_sd)
            set_model_state_dict(
                model,
                m_sd,
                options=StateDictOptions(full_state_dict=False, strict=False),
            )
        with contextlib.suppress(Exception):
            _coerce_scaler_buffers_to_shape(
                model,
                getattr(model, "in_dim", None),
                getattr(model, "out_shape", ()) or (),
            )
        return meta if isinstance(meta, dict) else None

    if not p.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {str(p)!r}")

    suffix = p.suffix.lower()
    if suffix == ".safetensors":
        meta_path = p.with_name(p.name + ".json")
        if not meta_path.exists():
            meta_path = p.with_suffix(".json")
        meta = None
        if meta_path.exists():
            meta = read_json(meta_path)
        if not isinstance(meta, dict):
            meta = _read_safetensors_embedded_meta(p)
        if rebuild_tasks:
            with contextlib.suppress(Exception):
                if isinstance(meta, dict):
                    tasks = meta.get("tasks")
                    if tasks:
                        model.rebuild_tasks_from_specs(tasks)
        is_required(
            "safetensors",
            "pip install 'enn-torch[safetensors]'  # or: pip install safetensors",
        )
        from safetensors.torch import load_file as load_tensors
        if map_location is None:
            dev = "cpu"
        elif isinstance(map_location, torch.device):
            dev = str(map_location)
        else:
            dev = str(map_location)
        sd = load_tensors(str(p), device=dev)
        sd = _coerce_state_dict(sd)
        sd = _drop_runtime_only_state_keys(sd)
        resize_scaler_buffer(model, sd)
        if _model_has_meta_or_fake_tensors(model):
            _materialize_module_to_device(model, dev)
        _load_state_dict_compat(model, sd, strict=False)
        with contextlib.suppress(Exception):
            _coerce_scaler_buffers_to_shape(
                model, getattr(model, "in_dim", None), getattr(model, "out_shape", ()) or ()
            )
        return meta if isinstance(meta, dict) else None

    if suffix and suffix not in (".pt", ".pth"):
        raise ValueError(
            f"Unsupported checkpoint extension {suffix!r}. Use .pt/.pth/.safetensors or a directory checkpoint instead."
        )
    obj = _torch_load_checkpoint(
        p,
        map_location=map_location or "cpu",
        weights_only=weights_only,
        mmap=mmap,
    )

    dump_dir = str(os.environ.get("ENN_NONFINITE_DUMP_DIR", "") or "").strip()
    strict_nf = env_bool(
        "ENN_FAIL_ON_LOAD_NONFINITE",
        default=env_bool("ENN_SANITIZE_NAN_STRICT", default=False),
    )

    def _allowed_nonfinite_key(name: str, t: torch.Tensor) -> bool:
        n = str(name)
        allow_scaler_inf = env_bool("ENN_SAVE_ALLOW_SCALER_INF", default=True)
        allow_logger_inf = env_bool("ENN_SAVE_ALLOW_LOGGER_INF", default=True)
        if bool(torch.isnan(t).any().item()):
            return False
        if allow_scaler_inf:
            if (
                n.endswith("scaler.y_min")
                or n.endswith("scaler.y_max")
                or n.endswith("scaler.y_q_low")
                or n.endswith("scaler.y_q_high")
                or n.endswith("scaler.y_out_clip_low")
                or n.endswith("scaler.y_out_clip_high")
            ):
                return bool(torch.isinf(t).all().item())
        if allow_logger_inf:
            if (("logger.sampled_" in n) or ("logger.reduced_" in n)) and (n.endswith("_min") or n.endswith("_max")):
                if t.numel() <= 16:
                    return bool(torch.isinf(t).all().item())
        return False

    def _first_nonfinite_in_state(sd: Mapping[str, Any]) -> str | None:
        with torch.no_grad():
            for k, v in sd.items():
                if not torch.is_tensor(v):
                    continue
                if (not v.is_floating_point()) and (not v.is_complex()):
                    continue
                if v.numel() <= 0:
                    continue
                try:
                    a = v.detach().abs().amax()
                    ok = bool(torch.isfinite(a).item())
                except Exception:
                    ok = bool(torch.isfinite(v).all().item())
                if not ok:
                    if _allowed_nonfinite_key(str(k), v):
                        continue
                    return str(k)
        return None

    def _first_nonfinite_param(m: torch.nn.Module) -> str | None:
        with torch.no_grad():
            for n, p0 in m.named_parameters(recurse=True):
                if not torch.is_tensor(p0) or (not p0.is_floating_point()) or p0.numel() <= 0:
                    continue
                try:
                    a = p0.detach().abs().amax()
                    ok = bool(torch.isfinite(a).item())
                except Exception:
                    ok = bool(torch.isfinite(p0).all().item())
                if not ok:
                    return str(n)
        return None

    bad_sd: str | None = None

    meta: Mapping[str, Any] | None = obj if isinstance(obj, dict) else None
    sd = None
    if isinstance(obj, dict):
        if rebuild_tasks:
            with contextlib.suppress(Exception):
                tasks = obj.get("tasks")
                if (not tasks) and isinstance(side_meta, Mapping):
                    tasks = side_meta.get("tasks")
                if tasks:
                    model.rebuild_tasks_from_specs(tasks)
        for k in ("state_dict", "model_state_dict", "model", "model_sd", "weights", "model_weights"):
            v = obj.get(k)
            if isinstance(v, dict):
                sd = v
                break
        if sd is None:
            try:
                if obj and all(isinstance(k, str) and torch.is_tensor(v) for k, v in obj.items()):
                    sd = obj
            except Exception:
                sd = None
        if sd is None:
            raise RuntimeError(f"Checkpoint did not contain a recognizable state_dict: {str(p)!r}")
    else:
        sd = obj
    sd = _coerce_state_dict(sd) if isinstance(sd, dict) else sd
    if isinstance(sd, Mapping):
        sd = _drop_runtime_only_state_keys(cast(Mapping[str, Any], sd))

    if (dump_dir or strict_nf) and isinstance(sd, Mapping):
        bad_sd = _first_nonfinite_in_state(cast(Mapping[str, Any], sd))
        if bad_sd is not None:
            logger.error("[ENN] load_weights: non-finite tensor detected in checkpoint state: %s", str(bad_sd))
            if dump_dir:
                with contextlib.suppress(Exception):
                    os.makedirs(dump_dir, exist_ok=True)
                    rid = os.urandom(4).hex()
                    rank = str(os.environ.get("RANK", "0") or "0")
                    path = os.path.join(dump_dir, f"load_nonfinite.state.rank{rank}.{rid}.pt")
                    torch.save({"where": "load_weights_state", "first_bad": str(bad_sd), "path": str(p)}, path)
                    logger.error("[ENN] load_weights: dumped to: %s", str(path))
            if strict_nf:
                raise RuntimeError(f"[ENN] load_weights: non-finite tensor in checkpoint state: {bad_sd}")

    with contextlib.suppress(Exception):
        resize_scaler_buffer(model, sd)
    if _model_has_meta_or_fake_tensors(model):
        _materialize_module_to_device(model, map_location or "cpu")
    with contextlib.suppress(Exception):
        buf = getattr(model, "output_baked_flag", None)
        if torch.is_tensor(buf):
            buf.zero_()

    sd_for_load: object = sd
    if isinstance(sd, Mapping) and env_bool("ENN_LOAD_ALIAS_PERCEIVER_KEYS", default=True):
        try:
            sd_map = dict(sd)
            model_names = [str(n) for n, _ in model.named_parameters(recurse=True)]
            model_keys = set(model_names)

            proc_root = "fuser"
            if any(k.startswith("fuser.perceiver.") for k in model_keys):
                proc_root = "fuser"
            elif any(k.startswith("processor.perceiver.") for k in model_keys):
                proc_root = "processor"

            ctrl_root = "temporal_token_collector"
            if any(k.startswith("temporal_token_collector.") for k in model_keys):
                ctrl_root = "temporal_token_collector"
            elif any(k.startswith("controller.") for k in model_keys):
                ctrl_root = "controller"

            perceiver_plain_prefix = f"{proc_root}.perceiver."
            perceiver_orig_prefix = f"{proc_root}.perceiver._orig_mod."
            has_perceiver_origmod = any(k.startswith(perceiver_orig_prefix) for k in model_keys)
            has_perceiver_plain = any(
                k.startswith(perceiver_plain_prefix) and (not k.startswith(perceiver_orig_prefix))
                for k in model_keys
            )

            def _alias_key(k: str) -> str:
                kk = str(k)
                kk = kk.replace("._enn_inner._orig_mod", "")
                kk = kk.replace("._enn_inner", "")

                if proc_root == "fuser" and kk.startswith("processor."):
                    kk = "fuser." + kk[len("processor.") :]
                elif proc_root == "processor" and kk.startswith("fuser."):
                    kk = "processor." + kk[len("fuser.") :]

                if ctrl_root == "temporal_token_collector" and kk.startswith("controller."):
                    kk = "temporal_token_collector." + kk[len("controller.") :]
                elif ctrl_root == "controller" and kk.startswith("temporal_token_collector."):
                    kk = "controller." + kk[len("temporal_token_collector.") :]

                if has_perceiver_origmod and kk.startswith(perceiver_plain_prefix) and (not kk.startswith(perceiver_orig_prefix)):
                    kk = perceiver_orig_prefix + kk[len(perceiver_plain_prefix) :]
                elif has_perceiver_plain and kk.startswith(perceiver_orig_prefix):
                    kk = perceiver_plain_prefix + kk[len(perceiver_orig_prefix) :]
                return kk

            added = 0
            examples: list[str] = []
            try:
                ex_lim = int(os.environ.get("ENN_LOAD_ALIAS_PERCEIVER_LOG_LIMIT", "10") or "10")
            except Exception:
                ex_lim = 10
            for k, v in list(sd_map.items()):
                if not isinstance(k, str):
                    continue
                kk = _alias_key(k)
                if kk != k and kk in model_keys and kk not in sd_map:
                    sd_map[kk] = v
                    added += 1
                    if len(examples) < int(ex_lim):
                        examples.append(f"{k} -> {kk}")

            if added > 0:
                if (dump_dir or strict_nf) or env_bool("ENN_LOAD_ALIAS_PERCEIVER_LOG", default=False):
                    msg = ""
                    if examples:
                        msg = " Example: " + "; ".join(examples[:10])
                    logger.warning(
                        "[ENN] load_weights: added %d aliased keys for perceiver wrapper compatibility.%s",
                        int(added),
                        msg,
                    )
            sd_for_load = sd_map
        except Exception:
            sd_for_load = sd

    eager_ctx = getattr(model, "eager_for_export", None)
    cm = eager_ctx() if callable(eager_ctx) else contextlib.nullcontext()
    with cm:
        incompat = _load_state_dict_compat(model, sd_for_load, strict=False)

    warn_incompat = bool(dump_dir or strict_nf) or env_bool(
        "ENN_LOAD_ALIAS_PERCEIVER_WARN_MISSING",
        default=env_bool("ENN_SANITIZE_NAN_STRICT", default=False),
    )
    fail_incompat = env_bool(
        "ENN_LOAD_FAIL_ON_INCOMPAT",
        default=env_bool("ENN_SANITIZE_NAN_STRICT", default=False),
    )
    if (warn_incompat or fail_incompat) and incompat is not None:
        miss0 = getattr(incompat, "missing_keys", None) or []
        unexp0 = getattr(incompat, "unexpected_keys", None) or []

        def _is_ignorable_incompat_key(k: object) -> bool:
            try:
                s = str(k)
            except Exception:
                return False
            return s == "output_baked_flag" or s.endswith(".output_baked_flag")

        ign_miss = [k for k in miss0 if _is_ignorable_incompat_key(k)]
        ign_unexp = [k for k in unexp0 if _is_ignorable_incompat_key(k)]
        miss = [k for k in miss0 if not _is_ignorable_incompat_key(k)]
        unexp = [k for k in unexp0 if not _is_ignorable_incompat_key(k)]

        if ign_miss:
            with contextlib.suppress(Exception):
                buf = getattr(model, "output_baked_flag", None)
                if torch.is_tensor(buf):
                    buf.zero_()

        if (ign_miss or ign_unexp) and (dump_dir or strict_nf or env_bool("ENN_LOAD_ALIAS_PERCEIVER_LOG", default=False)):
            with contextlib.suppress(Exception):
                logger.info(
                    "[ENN] load_weights: ignoring state_dict incompat keys: missing=%s; unexpected=%s",
                    ", ".join(str(x) for x in ign_miss[:10]) if ign_miss else "",
                    ", ".join(str(x) for x in ign_unexp[:10]) if ign_unexp else "",
                )

        if miss or unexp:
            miss_p = [k for k in miss if str(k).startswith("fuser.perceiver.") or str(k).startswith("processor.perceiver.")]
            sm_miss = (miss_p[:10] if miss_p else miss[:10])
            sm_unexp = unexp[:10]
            try:
                sm_miss_s = ", ".join(str(x) for x in sm_miss)
            except Exception:
                sm_miss_s = str(sm_miss)
            try:
                sm_unexp_s = ", ".join(str(x) for x in sm_unexp)
            except Exception:
                sm_unexp_s = str(sm_unexp)
            logger.warning(
                "[ENN] load_weights: state_dict incompat: missing=%d (perceiver=%d), unexpected=%d. "
                "missing_example=%s; unexpected_example=%s",
                int(len(miss)),
                int(len(miss_p)),
                int(len(unexp)),
                sm_miss_s,
                sm_unexp_s,
            )
            if fail_incompat:
                raise RuntimeError(
                    f"[ENN] load_weights: state_dict incompat (missing={len(miss)}, unexpected={len(unexp)})"
                )

    if dump_dir or strict_nf:
        bad_model = _first_nonfinite_param(model)
        if bad_model is not None:
            logger.error("[ENN] load_weights: non-finite parameter detected after load: %s", str(bad_model))
            if dump_dir:
                with contextlib.suppress(Exception):
                    os.makedirs(dump_dir, exist_ok=True)
                    rid = os.urandom(4).hex()
                    rank = str(os.environ.get("RANK", "0") or "0")
                    path = os.path.join(dump_dir, f"load_nonfinite.param.rank{rank}.{rid}.pt")
                    payload = {"where": "load_weights_param", "first_bad": str(bad_model), "path": str(p), "bad_sd": str(bad_sd) if bad_sd else None}
                    sd_used = sd_for_load if isinstance(sd_for_load, Mapping) else sd
                    if isinstance(sd_used, Mapping):
                        try:
                            sd_keys = set(str(k) for k in sd_used.keys())
                            missing = [n for n, _ in model.named_parameters(recurse=True) if str(n) not in sd_keys]
                            payload["missing_param_keys"] = missing[:100]
                            payload["missing_param_count"] = int(len(missing))
                        except Exception:
                            pass
                    torch.save(payload, path)
                    logger.error("[ENN] load_weights: dumped to: %s", str(path))
            if strict_nf:
                raise RuntimeError(f"[ENN] load_weights: non-finite parameter after load: {bad_model} (checkpoint_bad={bad_sd})")

    with contextlib.suppress(Exception):
        _coerce_scaler_buffers_to_shape(
            model, getattr(model, "in_dim", None), getattr(model, "out_shape", ()) or ()
        )
    _validate_scaler_buffers(model, strict=bool(strict_nf))
    return meta


def load_model(
    checkpoint_path: PathLike,
    in_dim: int | None = None,
    out_shape: Sequence[int] | None = None,
    config: ModelConfig | Mapping[str, object] | None = None,
    map_location: TorchDeviceLike | None = None,
    weights_only: bool = True,
    mmap: bool | None = True,
) -> Model:
    def _diag_nonfinite_after_load(m: nn.Module, *, where: str) -> None:
        dump_dir = str(os.environ.get("ENN_NONFINITE_DUMP_DIR", "") or "").strip()
        strict = env_bool(
            "ENN_FAIL_ON_LOAD_NONFINITE",
            default=env_bool("ENN_SANITIZE_NAN_STRICT", default=False),
        )
        if (not dump_dir) and (not strict):
            return

        scope = str(os.environ.get("ENN_DIAG_LOAD_NONFINITE_SCOPE", "perceiver") or "perceiver").strip().lower()
        tgt: nn.Module = m
        with contextlib.suppress(Exception):
            if scope in {"perceiver", "fuser.perceiver"}:
                f0 = getattr(m, "fuser", None)
                p0 = getattr(f0, "perceiver", None) if f0 is not None else None
                if isinstance(p0, nn.Module):
                    tgt = p0

        first_bad = None
        with torch.no_grad():
            for n, p in tgt.named_parameters(recurse=True):
                if not isinstance(p, torch.Tensor) or (not p.is_floating_point()) or p.numel() <= 0:
                    continue
                try:
                    a = p.detach().abs().amax()
                    ok = bool(torch.isfinite(a).item())
                except Exception:
                    ok = bool(torch.isfinite(p).all().item())
                if not ok:
                    first_bad = str(n)
                    break

        if first_bad is None:
            return

        logger.error("[ENN] load_model: non-finite parameters detected (%s): %s", str(where), str(first_bad))
        if dump_dir:
            with contextlib.suppress(Exception):
                os.makedirs(dump_dir, exist_ok=True)
                rid = os.urandom(4).hex()
                rank = str(os.environ.get("RANK", "0") or "0")
                path = os.path.join(dump_dir, f"load_nonfinite.{scope}.rank{rank}.{rid}.pt")
                torch.save({"where": where, "scope": scope, "first_bad": first_bad}, path)
                logger.error("[ENN] load_model: dumped to: %s", str(path))

        if strict:
            raise RuntimeError(f"[ENN] load_model: non-finite parameters detected ({where}): {first_bad}")

    p = Path(_normalize_windows_paste(checkpoint_path))
    side_meta: Mapping[str, Any] | None = None
    with contextlib.suppress(Exception):
        if p.is_file() and p.suffix.lower() in (".pt", ".pth"):
            side_meta = _read_pt_sidecar_meta(p)
    load_dev = (
        torch.device(map_location)
        if map_location is not None
        else torch.device("cpu")
    )
    if p.is_dir():
        meta = _parse_meta(p)
        use_in_dim = int(
            in_dim if in_dim is not None else meta.get("in_dim") or 0
        )
        out_shape_meta = (
            out_shape if out_shape is not None else meta.get("out_shape") or ()
        )
        use_out_shape = (
            tuple((int(x) for x in out_shape_meta)) if out_shape_meta else ()
        )
        user_provided_config = config is not None
        raw_cfg = config if user_provided_config else meta.get("config")
        use_config = coerce_model_config(raw_cfg)
        if not user_provided_config:
            use_config.device = load_dev
        elif map_location is not None and use_config.device is None:
            use_config.device = load_dev
        if use_in_dim <= 0 or not use_out_shape:
            raise ValueError(
                "Loading from a checkpoint directory requires in_dim and out_shape, or a valid meta.json inside the directory."
            )
        model = new_model(use_in_dim, use_out_shape, use_config)
        if _model_has_meta_or_fake_tensors(model):
            _materialize_module_to_device(model, load_dev)
        with contextlib.suppress(Exception):
            tasks = meta.get("tasks") if isinstance(meta, dict) else None
            if tasks:
                model.rebuild_tasks_from_specs(tasks)
        reader = FileSystemReader(str(p))
        meta_data = None
        with ensure_dcp_process_group(torch.device("cpu")):
            with contextlib.suppress(Exception):
                meta_data = reader.read_metadata()
        if not _resize_scaler_buffers_from_metadata(model, meta_data):
            _resize_scaler_buffers_for_shape(model, use_in_dim, use_out_shape)
        opts = StateDictOptions(full_state_dict=False)
        eager_ctx = getattr(model, "eager_for_export", None)
        cm = eager_ctx() if callable(eager_ctx) else contextlib.nullcontext()
        with cm:
            strict = _dcp_strict_load_enabled()
            m_sd = dict(get_model_state_dict(model, options=opts))
            _coerce_dcp_keys(m_sd)
            if not strict:
                allowed, shapes = _dcp_allowed_keys_and_shapes(meta_data)
                if allowed:
                    m_sd = _dcp_filter_model_state_dict(
                        m_sd, allowed_keys=allowed, shapes=shapes
                    )
            planner = _make_dcp_load_planner(allow_partial_load=not strict)
            try:
                if not strict:
                    _raise_if_empty_dcp_model_state(m_sd, checkpoint_path=p)
                _dcp_load_model_state(reader=reader, model_state=m_sd, planner=planner)
            except Exception:
                if (not strict) and _try_load_dir_checkpoint_fallback_pt(
                    model,
                    pt_path=(p / "model.pt"),
                    map_location=map_location,
                    mmap=mmap,
                    rebuild_tasks=True,
                ):
                    _diag_nonfinite_after_load(model, where="dcp_dir_fallback_pt")
                    return model
                raise
            resize_scaler_buffer(model, m_sd)
            set_model_state_dict(
                model,
                m_sd,
                options=StateDictOptions(full_state_dict=False, strict=False),
            )
        with contextlib.suppress(Exception):
            _coerce_scaler_buffers_to_shape(model, use_in_dim, use_out_shape)
        _diag_nonfinite_after_load(model, where="dcp_dir")
        return model
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {str(p)!r}")
    suffix = p.suffix.lower()
    if suffix == ".safetensors":
        meta_path = p.with_name(p.name + ".json")
        if not meta_path.exists():
            meta_path = p.with_suffix(".json")
        meta = None
        if meta_path.exists():
            meta = read_json(meta_path)
        if not isinstance(meta, dict):
            meta = _read_safetensors_embedded_meta(p)
        if not isinstance(meta, dict):
            if in_dim is None or out_shape is None:
                raise RuntimeError(
                    "Missing metadata for safetensors checkpoint. Provide in_dim and out_shape, "
                    "or export with embedded metadata (workflows.train export_path=...)."
                )
            meta = {}
        use_in_dim = int(in_dim if in_dim is not None else meta.get("in_dim"))
        out_shape_meta = (
            out_shape if out_shape is not None else meta.get("out_shape") or ()
        )
        use_out_shape = (
            tuple((int(x) for x in out_shape_meta)) if out_shape_meta else ()
        )
        user_provided_config = config is not None
        use_config = coerce_model_config(
            config if user_provided_config else meta.get("config")
        )
        if not user_provided_config:
            use_config.device = load_dev
        elif map_location is not None and use_config.device is None:
            use_config.device = load_dev
        if use_in_dim <= 0 or not use_out_shape:
            raise RuntimeError(
                f"Invalid in_dim/out_shape metadata in {str(meta_path)!r}: in_dim={use_in_dim}, out_shape={use_out_shape}"
            )
        model = new_model(use_in_dim, use_out_shape, use_config)
        with contextlib.suppress(Exception):
            tasks = meta.get("tasks") if isinstance(meta, dict) else None
            if tasks:
                model.rebuild_tasks_from_specs(tasks)
        is_required(
            "safetensors",
            "pip install 'enn-torch[safetensors]'  # or: pip install safetensors",
        )
        from safetensors.torch import load_file as load_tensors

        dev = None
        if map_location is None:
            dev = "cpu"
        elif isinstance(map_location, torch.device):
            dev = str(map_location)
        else:
            dev = str(map_location)
        sd = load_tensors(str(p), device=dev)
        sd = _coerce_state_dict(sd)
        resize_scaler_buffer(model, sd)
        if _model_has_meta_or_fake_tensors(model):
            _materialize_module_to_device(model, dev)
        _load_state_dict_compat(model, sd, strict=False)
        with contextlib.suppress(Exception):
            _coerce_scaler_buffers_to_shape(model, use_in_dim, use_out_shape)
        _diag_nonfinite_after_load(model, where="safetensors")
        return model

    if suffix and suffix not in (".pt", ".pth"):
        raise ValueError(
            f"Unsupported checkpoint extension {suffix!r}. Use .pt/.pth/.safetensors or a directory checkpoint instead."
        )

    obj = _torch_load_checkpoint(
        p,
        map_location=map_location or "cpu",
        weights_only=weights_only,
        mmap=mmap,
    )
    if isinstance(obj, dict):
        meta_in_dim = obj.get("in_dim")
        meta_out_shape = obj.get("out_shape")
        meta_cfg = obj.get("config")
        sd = None
        for k in (
            "state_dict",
            "model_state_dict",
            "model",
            "model_sd",
            "weights",
            "model_weights",
        ):
            v = obj.get(k)
            if isinstance(v, dict):
                sd = v
                break
        if sd is None:
            try:
                if obj and all(
                    isinstance(k, str) and torch.is_tensor(v)
                    for k, v in obj.items()
                ):
                    sd = obj
            except Exception:
                sd = None
        if sd is None:
            raise RuntimeError(
                f"Checkpoint did not contain a recognizable state_dict: {str(p)!r}"
            )
    else:
        meta_in_dim = None
        meta_out_shape = None
        meta_cfg = None
        sd = obj
    use_in_dim = int(in_dim if in_dim is not None else meta_in_dim)
    out_shape_meta = (
        out_shape if out_shape is not None else meta_out_shape or ()
    )
    use_out_shape = tuple((int(x) for x in out_shape_meta))
    user_provided_config = config is not None
    use_config = coerce_model_config(
        config if user_provided_config else meta_cfg
    )
    if not user_provided_config:
        use_config.device = load_dev
    elif map_location is not None and use_config.device is None:
        use_config.device = load_dev
    if use_in_dim <= 0 or not use_out_shape:
        raise RuntimeError(
            f"Invalid or missing in_dim/out_shape when loading checkpoint {str(p)!r}: in_dim={use_in_dim}, out_shape={use_out_shape}"
        )
    model = new_model(use_in_dim, use_out_shape, use_config)
    with contextlib.suppress(Exception):
        tasks = obj.get("tasks") if isinstance(obj, dict) else None
        if (not tasks) and isinstance(side_meta, Mapping):
            tasks = side_meta.get("tasks")
        if tasks:
            model.rebuild_tasks_from_specs(tasks)
    sd = _coerce_state_dict(sd) if isinstance(sd, dict) else sd
    with contextlib.suppress(Exception):
        resize_scaler_buffer(model, sd)
    if _model_has_meta_or_fake_tensors(model):
        _materialize_module_to_device(model, load_dev)
    _load_state_dict_compat(model, sd, strict=False)
    with contextlib.suppress(Exception):
        _coerce_scaler_buffers_to_shape(model, use_in_dim, use_out_shape)
    _diag_nonfinite_after_load(model, where="file")
    return model


def save_model(
    model: torch.nn.Module,
    path: PathLike,
    optimizer: torch.optim.Optimizer | None = None,
    extra: Mapping[str, object] | None = None,
    *args: Any,
    state_dict: Mapping[str, torch.Tensor] | None = None,
    ema_averager: object | None = None,
    swa_averager: object | None = None,
    **kwargs: Any,
) -> str:
    from .io import Builder
    from .io import Exporter

    p = Path(_normalize_windows_paste(path))
    if Builder.is_target_native(p):
        if args:
            raise TypeError(
                "Positional args are only supported for export converters; use keyword arguments for TorchIO.save()."
            )
        merged_extra = dict(extra or {})
        if ema_averager is not None and hasattr(ema_averager, "state_dict"):
            with contextlib.suppress(Exception):
                merged_extra["ema_averager_state"] = ema_averager.state_dict()
        if swa_averager is not None and hasattr(swa_averager, "state_dict"):
            with contextlib.suppress(Exception):
                merged_extra["swa_averager_state"] = swa_averager.state_dict()
        out = Builder.save(
            model,
            p,
            optimizer=optimizer,
            extra=merged_extra or None,
            state_dict=state_dict,
            **kwargs,
        )
        return str(out)
    conv = Exporter.for_export(p.suffix)
    if conv is None:
        raise ValueError(f"Unknown export format for path '{path}'.")
    conv.save(model, p, *args, **kwargs)
    return str(p)


def train(
    model: torch.nn.Module | PathLike,
    data: TrainData,
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
    rdzv_backend: str = "c10d",
    rdzv_endpoint: str | None = None,
    loss_tile_dim: int | None = None,
    loss_tile_size: int | None = None,
    loss_mask_mode: str = "none",
    loss_mask_value: float | None = None,
    export_path: PathLike | None = None,
    export_overwrite: bool = True,
    checkpoint: bool = True,
    model_averaging: Optional[str] = "auto",
    **kwargs: Any,
) -> Model:
    ProcessBroker.bootstrap()
    val_frac = max(0.0, min(1.0, float(val_frac)))
    seed_value = _coerce_seed(seed)
    ProcessBroker.set_seed(seed_value)
    underflow_action = normalize_underflow_action(
        kwargs.pop("underflow_action", None),
        default=default_underflow_action(),
    )
    ds_meta = Dataset.for_device(
        "cpu", feature_dtype=torch.float64, label_float_dtype=torch.float64
    )
    ds_meta.underflow_action = underflow_action
    memmap_dir = new_dir("memmap_ds", large=True)
    ckpt_dir = new_dir("ckpt_dcp", large=True)
    init_dir = None
    init_ckpt_path: str | None = None
    trained_loaded: bool = False
    num_samples_dataset = 0
    first_in_dim = None
    label_shape = ()
    try:
        datasets, manifest = iter_dataset(data)
        for key, d in datasets:
            sub = os.path.join(memmap_dir, key) if manifest else memmap_dir
            if manifest:
                os.makedirs(sub, exist_ok=True)
            in_dim, lshape, n = _save_dataset(
                d,
                sub,
                ds=ds_meta,
                val_frac=val_frac,
                seed_value=seed_value,
                underflow_action=underflow_action,
                shuffle=bool(shuffle),
            )
            first_in_dim, label_shape = _get_label_shape(
                first_in_dim, in_dim, label_shape, lshape
            )
            num_samples_dataset += int(n)
        if first_in_dim is None or not label_shape:
            raise RuntimeError("No training data")
        if manifest is not None:
            payload = (
                manifest if isinstance(manifest, dict) else list(manifest)
            )
            collate.write_json(
                os.path.join(memmap_dir, "multinode.json"),
                payload,
                indent=None,
            )

        _max_nodes = int(max_nodes) if max_nodes is not None else 1
        use_local_init = env_bool(
            "ENN_INIT_CKPT_LOCAL", default=(_max_nodes <= 1)
        )
        if use_local_init:
            init_dir = tempfile.mkdtemp(prefix=f"enn_init_ckpt_{run_id}_")
        else:
            init_dir = new_dir("init_ckpt", large=True)
        init_ckpt_path = os.fspath(init_dir)
        save_init_pt = env_bool(
            ("ENN_INIT_SAVE_PT", "ENN_INIT_CKPT_SAVE_PT"),
            default=True,
        )
        _save_model_checkpoint(
            model,
            init_dir,
            save_dcp=True,
            save_pt=bool(save_init_pt),
            overwrite=True,
        )
        cfg_raw = _extract_model_config_dict(model)
        cfg_dict = (
            coerce_model_config(cfg_raw).to_dict()
            if cfg_raw
            else ModelConfig().to_dict()
        )
        parent_to_meta = False
        if isinstance(model, torch.nn.Module) and env_bool(
            "ENN_PARENT_MODEL_TO_META", default=True
        ):
            with contextlib.suppress(Exception):
                model.to("meta")
                parent_to_meta = True
            with contextlib.suppress(Exception):
                import gc

                gc.collect()
        else:
            with contextlib.suppress(Exception):
                if isinstance(model, torch.nn.Module):
                    model.to("cpu")
        rdzv = get_available_host(
            rdzv_endpoint
            or get_preferred_ip(allow_loopback=True)
            or "127.0.0.1"
        )
        master_addr, _master_port = init_master_addr(rdzv)
        _wp = WorkerPolicy.optimize()
        _wp.set_thread_setting()
        lc = LaunchConfig(
            min_nodes=1,
            max_nodes=max_nodes,
            nproc_per_node=int(_wp.nproc_per_node),
            rdzv_backend=rdzv_backend,
            rdzv_endpoint=rdzv,
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
            "checkpoint": checkpoint,
            "model_averaging": model_averaging,
        }
        for key in RuntimeConfig.TRAIN_POS_ORDER[: len(args)]:
            default_kwargs.pop(key, None)
        default_kwargs.update(
            {
                key: value
                for key, value in kwargs.items()
                if key in default_kwargs
            }
        )
        ops = runtime_config("train", base, *args, **default_kwargs, **kwargs)

        def _has_meta_tensors(m: torch.nn.Module) -> bool:
            try:
                for t in m.parameters(recurse=True):
                    if is_meta_or_fake_tensor(t):
                        return True
                for t in m.buffers(recurse=True):
                    if is_meta_or_fake_tensor(t):
                        return True
            except Exception:
                return False
            return False

        def _materialize_to_cpu(m: torch.nn.Module) -> None:
            if hasattr(m, "to_empty"):
                m.to_empty(device="cpu")
            else:
                m.to("cpu")

        _clear_device_caches()
        with _start_context():
            elastic_launch(lc, process)(ops)
        fallback: str | None = None
        ret_dir = os.environ.get("ENN_RETURN_DIR") or ""
        for fname in ("model.pt",):
            fp = os.path.join(ckpt_dir, fname)
            if ret_dir:
                alt = os.path.join(str(ret_dir), fname)
                if os.path.isfile(alt):
                    fp = alt
            if os.path.isfile(fp):
                fallback = fp
                break
        if fallback is not None:
            if isinstance(model, torch.nn.Module) and _has_meta_tensors(model):
                with contextlib.suppress(Exception):
                    _materialize_to_cpu(model)
            try:
                load_weights(
                    model, fallback, map_location="cpu", rebuild_tasks=True
                )
            finally:
                with contextlib.suppress(Exception):
                    os.remove(fallback)
                with contextlib.suppress(Exception):
                    meta_path = os.path.join(
                        os.path.dirname(fallback), "model.meta.json"
                    )
                    if os.path.isfile(meta_path):
                        os.remove(meta_path)
                with contextlib.suppress(Exception):
                    d = os.path.dirname(fallback)
                    for name in os.listdir(d):
                        if name.endswith(".tmp") and "model.pt" in name:
                            with contextlib.suppress(Exception):
                                os.remove(os.path.join(d, name))
            _update_history(model, ckpt_dir, epochs, val_frac, num_samples_dataset)
            if export_path is not None and isinstance(model, torch.nn.Module):
                export_meta: dict[str, Any] = {
                    "format": "enn-export-safetensors-v1",
                    "created_time": float(time.time()),
                    "run_id": str(run_id),
                    "in_dim": int(first_in_dim) if first_in_dim is not None else None,
                    "out_shape": list(int(x) for x in (label_shape or ())),
                    "config": cfg_dict if isinstance(cfg_dict, dict) else None,
                }
                with contextlib.suppress(Exception):
                    tasks = getattr(model, "tasks", None)
                    if tasks is not None:
                        export_meta["tasks"] = tasks
                export_safetensors_single(
                    model,
                    export_path,
                    meta=export_meta,
                    overwrite=bool(export_overwrite),
                )
            cleanup = os.environ.get("ENN_CKPT_CLEANUP_ON_SUCCESS", "1")
            do_cleanup = cleanup.strip().lower() not in (
                "0",
                "false",
                "off",
                "no",
                "n",
            )
            if do_cleanup:
                cand = os.path.dirname(fallback)
                marker = os.path.join(cand, ".enn_ephemeral_ckpt")
                if os.path.isfile(marker):
                    with contextlib.suppress(Exception):
                        shutil.rmtree(cand, ignore_errors=True)
            gc.collect()
            return model
        else:
            if isinstance(model, torch.nn.Module) and _has_meta_tensors(model):
                with contextlib.suppress(Exception):
                    _materialize_to_cpu(model)
            dcp_dir = _find_latest_dcp_epoch_dir(ckpt_dir) or ckpt_dir
            reader = FileSystemReader(dcp_dir)
            meta = None
            with ensure_dcp_process_group(torch.device("cpu")):
                with contextlib.suppress(Exception):
                    meta = reader.read_metadata()
            if not _resize_scaler_buffers_from_metadata(model, meta):
                _resize_scaler_buffers_for_shape(
                    model, first_in_dim, label_shape
                )
            m_sd = _coerce_dcp_keys(
                get_model_state_dict(
                    model,
                    options=StateDictOptions(
                        full_state_dict=False, cpu_offload=True
                    ),
                )
            )
            state_meta = getattr(meta, "state_dict_metadata", None)
            if isinstance(state_meta, Mapping) and state_meta:
                allowed_keys: set[str] = set()
                for key in state_meta.keys():
                    if key.startswith("model."):
                        allowed_keys.add(key[len("model.") :])
                    else:
                        allowed_keys.add(key)
                if allowed_keys:
                    m_sd = {k: v for k, v in m_sd.items() if k in allowed_keys}
            with ensure_dcp_process_group(torch.device("cpu")):
                load(
                    state_dict={"model": m_sd},
                    storage_reader=reader,
                )
            resize_scaler_buffer(model, m_sd)
            set_model_state_dict(
                model, m_sd, options=StateDictOptions(full_state_dict=False, strict=False)
            )
            with contextlib.suppress(Exception):
                _coerce_scaler_buffers_to_shape(model, first_in_dim, label_shape)
        _update_history(model, ckpt_dir, epochs, val_frac, num_samples_dataset)
        if export_path is not None and isinstance(model, torch.nn.Module):
            export_meta: dict[str, Any] = {
                "format": "enn-export-safetensors-v1",
                "created_time": float(time.time()),
                "run_id": str(run_id),
                "in_dim": int(first_in_dim) if first_in_dim is not None else None,
                "out_shape": list(int(x) for x in (label_shape or ())),
                "config": cfg_dict if isinstance(cfg_dict, dict) else None,
            }
            with contextlib.suppress(Exception):
                tasks = getattr(model, "tasks", None)
                if tasks is not None:
                    export_meta["tasks"] = tasks
            export_safetensors_single(
                model,
                export_path,
                meta=export_meta,
                overwrite=bool(export_overwrite),
            )
        return model
    finally:
        restore_path: str | None = None
        with contextlib.suppress(Exception):
            for _name in ("model.pt",):
                fp = os.path.join(str(ckpt_dir or ""), _name)
                if fp and os.path.isfile(fp):
                    restore_path = fp
                    break
        if (
            restore_path is None
            and init_ckpt_path
            and os.path.exists(init_ckpt_path)
        ):
            restore_path = init_ckpt_path
        if isinstance(model, torch.nn.Module) and restore_path:
            try:
                if _has_meta_tensors(model):
                    with contextlib.suppress(Exception):
                        if hasattr(model, "to_empty"):
                            model.to_empty(device="cpu")
                    load_weights(
                        model,
                        restore_path,
                        map_location="cpu",
                        weights_only=True,
                        rebuild_tasks=True,
                    )
            except Exception:
                pass
        do_async = (
            str(os.environ.get("ENN_ASYNC_CLEANUP", "1") or "1")
            .strip()
            .lower()
            not in ("0", "false", "off", "no", "n")
        )
        if do_async:
            import threading

            paths = [memmap_dir]
            if ckpt_dir is not None:
                paths.append(ckpt_dir)
            if init_dir is not None:
                paths.append(init_dir)

            def _cleanup(ps: list[str]) -> None:
                for p in ps:
                    with contextlib.suppress(Exception):
                        shutil.rmtree(p, ignore_errors=True)

            threading.Thread(
                target=_cleanup,
                args=(paths,),
                daemon=True,
                name="enn-cleanup",
            ).start()
        else:
            shutil.rmtree(memmap_dir, ignore_errors=True)
            if ckpt_dir is not None:
                shutil.rmtree(ckpt_dir, ignore_errors=True)
            if init_dir is not None:
                shutil.rmtree(init_dir, ignore_errors=True)


def predict(
    model: torch.nn.Module | PathLike,
    data: PredictData,
    *args: Any,
    mode: str = "predict",
    seed: int = 7,
    shuffle: bool = False,
    max_nodes: int | None = None,
    rdzv_endpoint: str | None = None,
    rdzv_backend: str = None,
    output: str | None = "memory",
    path: PathLike | None = None,
    overwrite: str = "error",
    h5_compression: str | None = None,
    h5_compression_opts: int | None = None,
    h5_shuffle: bool = False,
    **kwargs: Any,
) -> PredictionOutput:
    if model is None:
        raise ValueError("predict: model must not be None")
    ProcessBroker.bootstrap()
    out_shape = tuple(
        int(x)
        for x in (
            kwargs.pop("out_shape", getattr(model, "out_shape", None)) or ()
        )
    )
    if not out_shape or any(x <= 0 for x in out_shape):
        raise ValueError(f"Invalid out_shape {out_shape}")
    multi_sources: dict[str, TensorDictBase] | None = None
    if not isinstance(data, TensorDictBase) and isinstance(
        data, (Mapping, Sequence)
    ):
        items = data.items() if isinstance(data, Mapping) else enumerate(data)
        if all(isinstance(v, TensorDictBase) for _, v in items):
            multi_sources = {str(k): v for k, v in items}
    if multi_sources is not None:
        base_kwargs = dict(kwargs)
        base_run_id = str(
            base_kwargs.get("run_id", f"predict-{os.getpid()}-{int(seed)}")
        )
        output_mode0 = collate._coerce_prediction_output(output)
        path_n0 = collate._coerce_path(path) if path is not None else None
        out_multi: dict[str, TensorDictBase] = {}
        for k, td in multi_sources.items():
            key = str(k)
            safe = (
                str(k).replace(os.sep, "_").replace(os.altsep or os.sep, "_")
                or "0"
            )
            per_run_id = f"{base_run_id}-{safe}" if safe else base_run_id
            per_path: PathLike | None = path
            if output_mode0 == "file" and path_n0 is not None:
                try:
                    p = Path(path_n0)
                    suf = str(p.suffix or "").lower()
                    if suf in {".h5", ".hdf5"} and bool(p.name):
                        per_path = p.with_name(f"{p.stem}-{safe}{p.suffix}")
                    else:
                        per_path = path_n0
                except Exception:
                    per_path = path
            out_multi[key] = cast(
                TensorDictBase,
                predict(
                    model,
                    td,
                    *args,
                    mode=mode,
                    seed=seed,
                    shuffle=shuffle,
                    max_nodes=max_nodes,
                    rdzv_endpoint=rdzv_endpoint,
                    rdzv_backend=rdzv_backend,
                    output=output,
                    path=per_path,
                    overwrite=overwrite,
                    h5_compression=h5_compression,
                    h5_compression_opts=h5_compression_opts,
                    h5_shuffle=h5_shuffle,
                    **{
                        **base_kwargs,
                        "run_id": per_run_id,
                        "out_shape": out_shape,
                    },
                ),
            )
        return out_multi
    underflow_action = kwargs.pop(
        "underflow_action", default_underflow_action()
    )
    chunk_size = kwargs.pop("chunk_size", None)
    output_mode = collate._coerce_prediction_output(output)
    overwrite_mode = collate._coerce_prediction_overwrite(overwrite)
    run_id = str(kwargs.pop("run_id", f"predict-{os.getpid()}-{int(seed)}"))
    out_path = None
    path_n = collate._coerce_path(path) if path is not None else None
    if output_mode == "file":
        if path_n is not None:
            out_path = collate._coerce_prediction_path(path_n, run_id=run_id)
            if out_path is None:
                logger.warning(
                    "predict: output=%r requires path to be a .h5/.hdf5 file. Got path=%r; falling back to output='memory'.",
                    output,
                    path,
                )
                output_mode = "memory"
            elif not collate._is_path_writable(out_path):
                logger.warning(
                    "predict: output=%r path is not writable: %r; falling back to output='memory'.",
                    output,
                    out_path,
                )
                out_path = None
                output_mode = "memory"
        else:
            output_mode = "memory"
    if output_mode == "file" and out_path and os.path.exists(out_path):
        if overwrite_mode == "resume" and os.path.isfile(out_path):
            collate.validate_predictions_h5(
                os.fspath(out_path), out_shape=out_shape
            )
            return PersistentTensorDict(filename=out_path, mode="r")
        if overwrite_mode == "error":
            raise FileExistsError(
                f"predict: destination already exists: {out_path!r}"
            )
    writer_chunk_size = int(chunk_size) if chunk_size is not None else 8192
    master_dtype = _get_float_precision(data)
    ds = Dataset.for_device(
        "cpu", feature_dtype=master_dtype, label_float_dtype=master_dtype
    )
    ds.underflow_action = underflow_action
    tmp_dir = new_dir("infer", large=True)
    ckpt_dir = os.path.join(tmp_dir, "ckpt")
    memmap_dir = os.path.join(tmp_dir, "memmap")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(memmap_dir, exist_ok=True)
    cleanup_ok = False
    inference_ctx = inference_mode(model)
    with inference_ctx:
        try:
            save_dcp = kwargs.pop("save_dcp", True)
            save_pt = kwargs.pop("save_pt", False)
            strict = env_bool("ENN_SANITIZE_NAN_STRICT", default=False)
            pt_only = env_bool("ENN_PRED_SAVE_PT_ONLY", default=False) or (
                strict and env_bool("ENN_PRED_SAVE_PT_ONLY_IF_STRICT", default=True)
            )
            wrapped_or_distributed = _is_wrapped_or_distributed_model(model)
            if pt_only:
                save_pt = True
                if wrapped_or_distributed:
                    logger.info(
                        "predict: keeping DCP checkpointing for wrapped/distributed model in strict PT-only mode"
                    )
                else:
                    save_dcp = False
            elif strict and env_bool("ENN_PRED_SAVE_PT_ALSO_IF_STRICT", default=False):
                save_pt = True
            dcp_dir = os.path.join(ckpt_dir, "dcp")
            if not (save_dcp or save_pt):
                save_dcp = True
            _save_model_checkpoint(
                model,
                dcp_dir,
                save_dcp=save_dcp,
                save_pt=save_pt,
                overwrite=True,
            )
            count, getter, _needs_preprocess = _adapt_source(data)
            if count <= 0:
                raise ValueError("Empty input")
            in_dim, _ = collate.stream_memmap(
                ds=ds,
                out_dir=memmap_dir,
                count=count,
                get_batch=getter,
                val_frac=0.0,
                seed_value=seed,
                underflow_action=underflow_action,
                shuffle=False,
                allow_missing_labels=True,
                features_only=True,
                chunk_size=writer_chunk_size,
            )
            cfg_raw = _extract_model_config_dict(model)
            cfg_dict = (
                coerce_model_config(cfg_raw).to_dict()
                if cfg_raw
                else ModelConfig().to_dict()
            )
            base = {
                "sources": {"kind": "memmap", "path": memmap_dir},
                "ckpt_dir": ckpt_dir,
                "model_ckpt_dir": dcp_dir,
                "in_dim": int(in_dim),
                "out_shape": out_shape,
                "cfg_dict": cfg_dict,
            }
            ops_kwargs = dict(kwargs)
            ops_kwargs.setdefault("seed", seed)
            ops_kwargs.setdefault("shuffle", shuffle)
            ops = runtime_config(mode, base, *args, **ops_kwargs)
            _wp = WorkerPolicy.optimize()
            _wp.set_thread_setting()
            rdzv = get_available_host(rdzv_endpoint or get_preferred_ip())
            master_addr, _ = init_master_addr(rdzv)
            lc = LaunchConfig(
                min_nodes=1,
                max_nodes=(
                    int(max_nodes)
                    if max_nodes is not None
                    else int(_wp.nproc_per_node)
                ),
                nproc_per_node=int(_wp.nproc_per_node),
                rdzv_backend=str(rdzv_backend or "c10d"),
                rdzv_endpoint=rdzv,
                run_id=run_id,
                max_restarts=0,
                monitor_interval=5,
                start_method=optimal_start_method(),
                local_addr=master_addr,
            )
            with contextlib.suppress(Exception):
                model.to("cpu")
            _clear_device_caches()
            with _start_context():
                elastic_launch(lc, process)(ops)
            chunks_dir = os.path.join(ckpt_dir, "pred_chunks")
            if not os.path.isdir(chunks_dir):
                raise RuntimeError(
                    f"predict: missing pred_chunks at {chunks_dir!r}"
                )
            store_float = collate._get_prediction_dtype(chunks_dir)
            pred_mmt_path = os.path.join(tmp_dir, "pred.mmt")
            collate.concat_memory_mapped_tensor(
                os.fspath(chunks_dir),
                os.fspath(pred_mmt_path),
                count=count,
                out_shape=out_shape,
                store_float=store_float,
            )
            X_mmt = collate.load_memmap_features(os.fspath(memmap_dir))
            Y_mmt = collate.open_memory_mapped_tensor(os.fspath(pred_mmt_path))
            if Y_mmt is None:
                raise RuntimeError(
                    "predict: failed to open assembled pred.mmt"
                )
            if out_path is not None:
                out_td = collate.write_predictions_h5_atomic(
                    os.fspath(out_path),
                    memmap_dir=os.fspath(memmap_dir),
                    pred_path=os.fspath(pred_mmt_path),
                    chunk_size=int(chunk_size or 8192),
                    overwrite=str(overwrite_mode or "replace"),
                    h5_compression=h5_compression,
                    h5_compression_opts=h5_compression_opts,
                    h5_shuffle=h5_shuffle,
                )
                collate.validate_predictions_h5(
                    os.fspath(out_path),
                    out_shape=out_shape,
                    in_dim=(int(in_dim) if in_dim else None),
                )
                cleanup_ok = True
                return out_td
            X_t = collate.copy_mmt_to_cpu_tensor(
                X_mmt, count=count, chunk_size=writer_chunk_size
            )
            Y_t = collate.copy_mmt_to_cpu_tensor(
                Y_mmt, count=count, chunk_size=writer_chunk_size
            )
            td_out = TensorDict({"X": X_t, "Y": Y_t}, batch_size=[int(count)])
            cleanup_ok = True
            return td_out
        finally:
            if cleanup_ok:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                logger.info("predict debug: preserving tmp_dir=%s", tmp_dir)
