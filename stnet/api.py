# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import logging
import os
import random
import shutil
import time
from functools import lru_cache, partial, update_wrapper
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
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
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load,
    save,
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

from .config import (
    ModelConfig,
    RuntimeConfig,
    _extract_model_config_dict,
    coerce_model_config,
    runtime_config,
)
from .core.casting import env_bool, parse_torch_dtype
from .core.distributed import (
    get_available_host,
    get_preferred_ip,
    initialize_master_addr
)
from .core.graph import inference_mode
from .core.system import (
    WorkerPolicy,
    init_python_path,
    init_start_method,
    new_dir,
    optimal_start_method,
)
from .data.nodes import Storage
from .data.pipeline import (
    Dataset,
    default_underflow_action,
    get_row,
    iter_dataset,
    normalize_underflow_action,
    get_feature_key,
)
from .data.schemas import read_json
from .nn.architecture import Model
from .nn.layers import Recorder, resize_scaler_buffer
from .runtime.io import (
    _filtered_warnings,
    _to_tensor,
    _torch_load_checkpoint,
    is_required
)
from .runtime.main import _coerce_dcp_keys, process

from torch.distributed.run import LaunchConfig, elastic_launch

P = ParamSpec("P")
R = TypeVar("R")
PathLike: TypeAlias = str | os.PathLike[str] | Path
TorchDeviceLike: TypeAlias = torch.device | str | int
TensorLike: TypeAlias = torch.Tensor | MemoryMappedTensor
TrainData: TypeAlias = (
    TensorDictBase
    | Mapping[str, object]
    | Sequence[Mapping[str, object]]
    | Mapping[str, Mapping[str, object]]
    | object
)
PredictData: TypeAlias = TrainData
PredictionOutput: TypeAlias = TensorDictBase | PersistentTensorDict | Mapping[str, TensorDictBase] | Mapping[str, torch.Tensor]
logger = logging.getLogger(__name__)


def _rewrite_state_dict_key(k: str) -> str:
    if not (k.startswith("m.") and ".module." in k):
        return k
    parts = k.split(".")
    if len(parts) >= 4 and parts[0] == "m" and parts[1].isdigit() and parts[2] == "module":
        return ".".join(parts[3:])
    return k


def _coerce_state_dict(sd: Mapping[str, Any]) -> Mapping[str, Any]:
    new_sd = None
    for k in sd.keys():
        nk = _rewrite_state_dict_key(k)
        if nk != k:
            new_sd = type(sd)()
            break
    if new_sd is None:
        return sd
    for k, v in sd.items():
        new_sd[_rewrite_state_dict_key(k)] = v
    return new_sd


def _parse_meta(p: PathLike) -> Mapping[str, Any]:
    meta_path = p / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        meta = read_json(meta_path)
        return meta if isinstance(meta, dict) else {}
    except Exception as exc:
        raise RuntimeError(f"Failed to parse checkpoint metadata at {str(meta_path)!r}") from exc


@lru_cache(maxsize=1)
def _is_execution_time_logged() -> bool:
    return env_bool(("STNET_LOG_TIMINGS", "STNET_TIMINGS", "STNET_DEBUG_TIMINGS"), default=False)


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


def _clear_process_group() -> None:
    try:
        import torch.distributed
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            with contextlib.suppress(Exception):
                torch.distributed.barrier()
            with contextlib.suppress(Exception):
                torch.distributed.destroy_process_group()
    except Exception:
        pass


def _init_distributed() -> None:
    _clear_process_group()
    init_python_path()
    with contextlib.suppress(Exception):
        torch.multiprocessing.allow_connection_pickling()
    init_start_method()


def _clear_device_caches() -> None:
    with contextlib.suppress(Exception):
        import gc
        gc.collect()
    with contextlib.suppress(Exception):
        from .core.system import empty_device_cache, get_device
        empty_device_cache(device=get_device(), do_gc=False, min_interval_s=0.0)
    with contextlib.suppress(Exception):
        from .core.system import collect_accelerator_ipc
        collect_accelerator_ipc()


def _coerce_seed(seed: int) -> Optional[int]:
    if seed is None:
        return None
    try:
        return int(seed)
    except (TypeError, ValueError):
        return None


def _set_seed(seed_value: int) -> None:
    if seed_value is None:
        return
    try:
        torch.manual_seed(seed_value)
    except (TypeError, ValueError, RuntimeError):
        pass
    with contextlib.suppress(Exception):
        from .core.system import set_accelerator_seed
        set_accelerator_seed(int(seed_value))
    try:
        random.seed(seed_value)
    except (TypeError, ValueError):
        pass
    try:
        numpy.random.seed(seed_value)
    except (TypeError, ValueError):
        pass


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
    if save_dcp:
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        m_sd = get_model_state_dict(model, options=opts)
    if save_dcp and m_sd is not None:
        with _filtered_warnings():
            save(
                state_dict={"model": m_sd},
                storage_writer=FileSystemWriter(out_dir, sync_files=True, overwrite=bool(overwrite)),
            )
    if save_pt:
        if m_sd is not None:
            pt_state = dict(m_sd)
            _coerce_dcp_keys(pt_state)
        else:
            pt_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(pt_state, os.path.join(out_dir, "model.pt"))
    return m_sd


def _get_label_shape(
    first_in_dim: int,
    in_dim: int,
    first_label_shape: object,
    lshape: object,
) -> Tuple[Any]:
    if first_in_dim is None:
        return (int(in_dim), tuple(lshape))
    if int(in_dim) != int(first_in_dim) or tuple(lshape) != tuple(first_label_shape):
        raise RuntimeError(
            f"Shape mismatch across datasets: expected X_dim={first_in_dim}, y_shape={first_label_shape}, got X_dim={in_dim}, y_shape={lshape}"
        )
    return (first_in_dim, first_label_shape)


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
    if isinstance(d, TensorDictBase):
        td = d
        if td.batch_size is None or len(td.batch_size) == 0:
            raise ValueError("TensorDict input to train() must have a batch dimension.")
        count = int(td.batch_size[0])
        if count <= 0:
            raise ValueError("Empty TensorDict provided to train().")
        in_dim, label_shape = Storage.stream_memmap(
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
        return (int(in_dim), tuple(label_shape), int(count))
    if (
        isinstance(d, Mapping)
        and d
        and all((not isinstance(v, Mapping) for v in d.values()))
        and (not Storage.is_feature_label_batch_mapping(d))
    ):
        count, _get_batch = Storage.column_cursor(d)
        if count <= 0:
            raise ValueError("Empty dataset provided to train().")
        in_dim, label_shape = Storage.stream_memmap(
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
        return (int(in_dim), tuple(label_shape), int(count))
    fx, lb, _, lshape = ds.preprocess(d, return_keys=False)
    if not fx.is_contiguous():
        fx = fx.contiguous()
    if lb is None:
        raise ValueError("train() requires labels")
    count = int(fx.shape[0])
    if count <= 0:
        raise ValueError("Empty dataset provided to train().")
    in_dim = int(fx.reshape(count, -1).shape[1])
    Storage.preload_memmap(
        {"features": fx, "labels": lb},
        memmap_dir=out_dir,
        val_frac=float(val_frac),
        shuffle=bool(shuffle),
        seed=seed_value,
        underflow_action=underflow_action,
    )
    del fx, lb
    return (int(in_dim), tuple(lshape), int(count))


def _reduce_batch_stats(recs: object) -> Optional[Mapping[str, Any]]:
    if not isinstance(recs, list) or not recs:
        return None
    last = recs[-1]
    if isinstance(last, Mapping):
        rxm = last.get("reduced_x_mean")
        ryv = last.get("reduced_y_var")
        if rxm is not None and ryv is not None:
            return {
                "sampled_x_mean": float(last.get("reduced_x_mean", 0.0)),
                "sampled_x_var": float(last.get("reduced_x_var", 0.0)),
                "sampled_x_min": float(last.get("reduced_x_min", float("inf"))),
                "sampled_x_max": float(last.get("reduced_x_max", float("-inf"))),
                "sampled_y_mean": float(last.get("reduced_y_mean", 0.0)),
                "sampled_y_var": float(last.get("reduced_y_var", 0.0)),
                "sampled_y_min": float(last.get("reduced_y_min", float("inf"))),
                "sampled_y_max": float(last.get("reduced_y_max", float("-inf"))),
            }
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


def _update_batch_stats(prev: object, n_prev: object, inc: object, n_inc: object) -> Any:
    if inc is None or n_inc <= 0:
        return prev
    if prev is None or n_prev <= 0:
        out = {}
        for key, val in inc.items():
            if key.startswith("sampled_"):
                out["reduced_" + key[len("sampled_") :]] = float(val)
        return out
    out = {}
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


def _coerce_path(path: PathLike) -> Optional[PathLike]:
    if path is None:
        return None
    p = str(path).strip()
    if not p:
        return None
    if p.lower() in ("none", "null", "nil"):
        return None
    return os.path.abspath(os.path.expanduser(p))


def _coerce_prediction_output(output: object) -> str:
    if isinstance(output, str):
        out = output.strip().lower()
        if out in {"file", "disk", "lazy", "h5", "hdf5"}:
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


def _coerce_prediction_path(path: PathLike, *args: Any, run_id: str) -> Optional[str]:
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
            probe = p / ".stnet_probe"
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
        if pred_path.endswith(".mmt"):
            meta_path = Storage.mmt_meta_path(pred_path)
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
                preds_t = _torch_load_checkpoint(pred_path, map_location="cpu", weights_only=True)
            except Exception:
                preds_t = None
            if isinstance(preds_t, torch.Tensor):
                return preds_t.dtype
            try:
                return torch.as_tensor(preds_t).dtype
            except Exception:
                continue
    return default


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
            X_td, _ = get_row(obj, labels_required=False)
            if X_td is None:
                return torch.float32
            if not bool(torch.is_tensor(X_td)):
                X_td = torch.as_tensor(X_td)
            dt = _to_torch_dtype(getattr(X_td, "dtype", None))
            return torch.float64 if dt == torch.float64 else torch.float32
        if isinstance(obj, Mapping) and Storage.is_feature_label_batch_mapping(obj):
            f_key = get_feature_key(obj)
            if f_key is not None and f_key in obj:
                X_all = obj.get(f_key)
                dt = _to_torch_dtype(getattr(X_all, "dtype", None))
                if dt is not None:
                    return torch.float64 if dt == torch.float64 else torch.float32
                if isinstance(X_all, (list, tuple)) and X_all:
                    dt0 = _to_torch_dtype(getattr(X_all[0], "dtype", None))
                    return torch.float64 if dt0 == torch.float64 else torch.float32
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


def load_model(
    checkpoint_path: PathLike,
    in_dim: int | None = None,
    out_shape: Sequence[int] | None = None,
    config: ModelConfig | Mapping[str, object] | None = None,
    map_location: TorchDeviceLike | None = None,
    weights_only: bool = True,
) -> Model:
    p = Path(checkpoint_path)
    load_dev = torch.device(map_location) if map_location is not None else torch.device("cpu")
    if p.is_dir():
        meta = _parse_meta(p)
        use_in_dim = int(in_dim if in_dim is not None else meta.get("in_dim") or 0)
        out_shape_meta = out_shape if out_shape is not None else meta.get("out_shape") or ()
        use_out_shape = tuple((int(x) for x in out_shape_meta)) if out_shape_meta else ()
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
        opts = StateDictOptions(full_state_dict=True)
        m_sd = get_model_state_dict(model, options=opts)
        load(state_dict={"model": m_sd}, storage_reader=FileSystemReader(str(p)))
        resize_scaler_buffer(model, m_sd)
        set_model_state_dict(model, m_sd, options=StateDictOptions(strict=False))
        return model
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {str(p)!r}")
    suffix = p.suffix.lower()
    if suffix == ".safetensors":
        meta_path = p.with_suffix(".json")
        if not meta_path.exists():
            raise RuntimeError("Missing sidecar JSON file for the safetensors checkpoint.")
        meta = read_json(meta_path)
        if not isinstance(meta, dict):
            raise RuntimeError(f"Invalid sidecar JSON file for the safetensors checkpoint: {str(meta_path)!r}")
        use_in_dim = int(in_dim if in_dim is not None else meta.get("in_dim"))
        out_shape_meta = out_shape if out_shape is not None else meta.get("out_shape") or ()
        use_out_shape = tuple((int(x) for x in out_shape_meta)) if out_shape_meta else ()
        user_provided_config = config is not None
        use_config = coerce_model_config(config if user_provided_config else meta.get("config"))
        if not user_provided_config:
            use_config.device = load_dev
        elif map_location is not None and use_config.device is None:
            use_config.device = load_dev
        if use_in_dim <= 0 or not use_out_shape:
            raise RuntimeError(
                f"Invalid in_dim/out_shape metadata in {str(meta_path)!r}: in_dim={use_in_dim}, out_shape={use_out_shape}"
            )
        model = new_model(use_in_dim, use_out_shape, use_config)
        is_required("safetensors", "pip install safetensors")
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
        model.load_state_dict(sd, strict=False)
        return model
    obj = _torch_load_checkpoint(p, map_location=map_location or "cpu", weights_only=weights_only)
    if isinstance(obj, dict):
        meta_in_dim = obj.get("in_dim")
        meta_out_shape = obj.get("out_shape")
        meta_cfg = obj.get("config")
        sd = obj["state_dict"] if "state_dict" in obj else obj
    else:
        meta_in_dim = None
        meta_out_shape = None
        meta_cfg = None
        sd = obj
    use_in_dim = int(in_dim if in_dim is not None else meta_in_dim)
    out_shape_meta = out_shape if out_shape is not None else meta_out_shape or ()
    use_out_shape = tuple((int(x) for x in out_shape_meta))
    user_provided_config = config is not None
    use_config = coerce_model_config(config if user_provided_config else meta_cfg)
    if not user_provided_config:
        use_config.device = load_dev
    elif map_location is not None and use_config.device is None:
        use_config.device = load_dev
    if use_in_dim <= 0 or not use_out_shape:
        raise RuntimeError(
            f"Invalid or missing in_dim/out_shape when loading checkpoint {str(p)!r}: in_dim={use_in_dim}, out_shape={use_out_shape}"
        )
    model = new_model(use_in_dim, use_out_shape, use_config)
    sd = _coerce_state_dict(sd) if isinstance(sd, dict) else sd
    with contextlib.suppress(Exception):
        resize_scaler_buffer(model, sd)
    model.load_state_dict(sd, strict=False)
    return model


def save_model(
    model: torch.nn.Module,
    path: PathLike,
    optimizer: torch.optim.Optimizer | None = None,
    extra: Mapping[str, object] | None = None,
    *args: Any,
    ema_averager: object | None = None,
    swa_averager: object | None = None,
    **kwargs: Any,
) -> str:
    from .runtime.io import Exporter, Builder
    p = Path(path)
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
        out = Builder.save(model, p, optimizer=optimizer, extra=merged_extra or None, **kwargs)
        return str(out)
    conv = Exporter.for_export(p.suffix)
    if conv is None:
        raise ValueError(f"Unknown export format for path '{path}'.")
    conv.save(model, p, *args, **kwargs)
    return str(p)


@get_execution_time(logger, fn_name="train")
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
    **kwargs: Any,
) -> Model:
    _init_distributed()
    try:
        val_frac = float(val_frac)
        val_frac = 0.0 if val_frac < 0.0 else 1.0 if val_frac > 1.0 else val_frac
    except (TypeError, ValueError):
        val_frac = 0.1
    seed_value = _coerce_seed(seed)
    _set_seed(seed_value)
    underflow_action = normalize_underflow_action(
        kwargs.pop("underflow_action", None), default=default_underflow_action()
    )
    ds_meta = Dataset.for_device(
        "cpu", feature_dtype=torch.float64, label_float_dtype=torch.float64
    )
    ds_meta.underflow_action = underflow_action
    memmap_dir = new_dir("memmap_ds")
    num_samples_dataset = 0
    first_in_dim = None
    label_shape = ()
    manifest = None
    ckpt_dir = None
    init_dir = None
    try:
        datasets, manifest = iter_dataset(data)
        multi = manifest is not None
        for key, d in datasets:
            sub = memmap_dir if not multi else os.path.join(memmap_dir, key)
            if multi:
                os.makedirs(sub, exist_ok=True)
            in_dim, lshape, n = _save_dataset(
                d,
                sub,
                ds=ds_meta,
                val_frac=float(val_frac),
                seed_value=seed_value,
                underflow_action=underflow_action,
                shuffle=bool(shuffle),
            )
            first_in_dim, label_shape = _get_label_shape(first_in_dim, in_dim, label_shape, lshape)
            num_samples_dataset += int(n)
        if first_in_dim is None or not label_shape:
            raise RuntimeError("no training data provided to train()")
        if manifest is not None:
            payload = manifest if isinstance(manifest, dict) else list(manifest)
            Storage.write_json(os.path.join(memmap_dir, "multinode.json"), payload, indent=None)
        ckpt_dir = new_dir("ckpt_dcp")
        save_dcp = env_bool("STNET_SAVE_DCP", True)
        save_pt = env_bool("STNET_SAVE_MODEL_PT", True)
        if not (save_dcp or save_pt):
            save_pt = True
        m_sd = None
        if save_dcp or save_pt:
            init_dir = new_dir("init_dcp")
            m_sd = _save_model_checkpoint(
                model, init_dir, save_dcp=save_dcp, save_pt=save_pt, overwrite=True
            )
        default_rdzv_host = get_preferred_ip(allow_loopback=True) or "127.0.0.1"
        resolved_rdzv = rdzv_endpoint if rdzv_endpoint else default_rdzv_host
        rdzv_endpoint = get_available_host(resolved_rdzv)
        master_addr, _master_port = initialize_master_addr(rdzv_endpoint)
        _wp = WorkerPolicy.optimize()
        _wp.set_thread_setting()
        nprocs = int(_wp.nproc_per_node)
        cfg_dict = _extract_model_config_dict(model)
        cfg_model = coerce_model_config(cfg_dict) if cfg_dict else ModelConfig()
        cfg_dict = cfg_model.to_dict()
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
        elastic_launch(lc, process)(ops)
        fallback = os.path.join(ckpt_dir, "model.pt")
        if os.path.isfile(fallback):
            cpu_state = _torch_load_checkpoint(fallback, map_location="cpu", weights_only=True)
            cpu_state = _to_tensor(cpu_state)
            resize_scaler_buffer(model, cpu_state)
            model.load_state_dict(cpu_state, strict=False)
        else:
            opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
            m_sd = get_model_state_dict(model, options=opts)
            m_sd = _coerce_dcp_keys(m_sd)
            load(state_dict={"model": m_sd}, storage_reader=FileSystemReader(ckpt_dir))
            resize_scaler_buffer(model, m_sd)
            set_model_state_dict(model, m_sd, options=StateDictOptions(strict=False))
        try:
            if ckpt_dir is not None:
                history_path = os.path.join(ckpt_dir, "history.json")
                if os.path.isfile(history_path):
                    raw = read_json(history_path)
                    if isinstance(raw, dict):
                        records = raw.get("records", []) or []
                        meta = raw.get("meta", {}) or {}
                    else:
                        records = raw if isinstance(raw, list) else []
                        meta = {}
                    logger = getattr(model, "logger", None)
                    if isinstance(meta, dict):
                        setattr(model, "_train_history_meta", dict(meta))
                    run_stats = _reduce_batch_stats(records) if isinstance(records, list) and records else None
                    sampled_n = None
                    if isinstance(meta, dict):
                        with contextlib.suppress(Exception):
                            sampled_n = int(meta.get("sampled_n")) if meta.get("sampled_n") is not None else None
                    if sampled_n is None or sampled_n <= 0:
                        try:
                            e = int(meta.get("epochs")) if isinstance(meta, dict) and meta.get("epochs") is not None else int(epochs)
                        except Exception:
                            e = int(epochs)
                        e = max(1, int(e))
                        try:
                            vf = float(meta.get("val_frac")) if isinstance(meta, dict) and meta.get("val_frac") is not None else float(val_frac)
                        except Exception:
                            vf = float(val_frac)
                        vf = 0.0 if vf < 0.0 else 1.0 if vf > 1.0 else vf
                        train_frac = max(0.0, min(1.0, 1.0 - vf))
                        sampled_n = int(round(float(num_samples_dataset) * float(train_frac) * float(e)))
                    if sampled_n is None or sampled_n <= 0:
                        sampled_n = int(num_samples_dataset)
                    prev_total = int(getattr(model, "_history_total_samples", 0))
                    new_total = prev_total + int(sampled_n)
                    prev_cum = getattr(model, "_history_cum_stats", None)
                    cum_stats = _update_batch_stats(prev_cum, prev_total, run_stats, int(sampled_n))
                    setattr(model, "_history_total_samples", new_total)
                    setattr(model, "_history_dataset_n", int(num_samples_dataset))
                    if cum_stats is not None:
                        setattr(model, "_history_cum_stats", cum_stats)
                    run_hist_prev = getattr(model, "_train_history", None)
                    run_index = len(run_hist_prev) if isinstance(run_hist_prev, list) else 0
                    run_record = {
                        "run_index": run_index,
                        "sampled_n": int(sampled_n),
                        "reduced_n": int(new_total),
                    }
                    if run_stats is not None:
                        run_record.update(run_stats)
                    if cum_stats is not None:
                        run_record.update(cum_stats)
                    if isinstance(meta, dict) and meta:
                        run_record["env"] = dict(meta)
                    new_run_hist = (
                        run_hist_prev + [run_record]
                        if isinstance(run_hist_prev, list)
                        else [run_record]
                    )
                    setattr(model, "_train_history", new_run_hist)
                    if isinstance(logger, Recorder):
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


@get_execution_time(logger, fn_name="predict")
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
    **kwargs: Any,
) -> PredictionOutput:
    if model is None:
        raise ValueError("predict: model must not be None")
    _init_distributed()
    out_shape = kwargs.pop("out_shape", getattr(model, "out_shape", None))
    if out_shape is None:
        raise ValueError(
            "predict: out_shape is required (pass out_shape=... or set model.out_shape)"
        )
    out_shape_t = tuple((int(x) for x in out_shape))
    if not out_shape_t or any((int(x) <= 0 for x in out_shape_t)):
        raise ValueError(f"predict: invalid out_shape={out_shape!r}")
    multi_sources: dict[str, TensorDictBase] | None = None
    if not isinstance(data, TensorDictBase):
        if isinstance(data, Mapping) and data and all((isinstance(v, TensorDictBase) for v in data.values())):
            multi_sources = {str(k): v for k, v in data.items()}
        elif isinstance(data, Sequence) and data and all((isinstance(v, TensorDictBase) for v in data)):
            multi_sources = {str(i): v for i, v in enumerate(data)}
    if multi_sources is not None:

        def _safe_key(k: str) -> str:
            s = str(k)
            for sep in (os.sep, os.altsep):
                if sep:
                    s = s.replace(sep, "_")
            return s or "0"

        base_kwargs = dict(kwargs)
        base_run_id = str(base_kwargs.get("run_id", f"predict-{os.getpid()}-{int(seed)}"))
        output_mode0 = _coerce_prediction_output(output)
        path_n0 = _coerce_path(path) if path is not None else None
        out_multi: dict[str, TensorDictBase] = {}
        for k, td in multi_sources.items():
            key = str(k)
            safe = _safe_key(key)
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
            inner_kwargs = dict(base_kwargs)
            inner_kwargs["run_id"] = per_run_id
            inner_kwargs.setdefault("out_shape", out_shape_t)
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
                    **inner_kwargs,
                ),
            )
        return out_multi
    underflow_action = kwargs.pop("underflow_action", default_underflow_action())
    chunk_size = kwargs.pop("chunk_size", None)
    output_mode = _coerce_prediction_output(output)
    overwrite_mode = _coerce_prediction_overwrite(overwrite)
    run_id = str(kwargs.pop("run_id", f"predict-{os.getpid()}-{int(seed)}"))
    out_path = None
    path_n = _coerce_path(path) if path is not None else None
    if output_mode == "file":
        if path_n is not None:
            out_path = _coerce_prediction_path(path_n, run_id=run_id)
            if out_path is None:
                logger.warning(
                    "predict: output=%r requires path to be a .h5/.hdf5 file. Got path=%r; falling back to output='memory'.",
                    output,
                    path,
                )
                output_mode = "memory"
            elif not _is_path_writable(out_path):
                logger.warning(
                    "predict: output=%r path is not writable: %r; falling back to output='memory'.",
                    output,
                    out_path,
                )
                out_path = None
                output_mode = "memory"
        else:
            output_mode = "memory"
    if output_mode == "file" and out_path is None:
        output_mode = "memory"
    if output_mode == "file" and out_path is not None and os.path.exists(out_path):
        if overwrite_mode == "resume" and os.path.isfile(out_path):
            Storage.validate_predictions_h5(os.fspath(out_path), out_shape=out_shape_t)
            return PersistentTensorDict(filename=out_path, mode="r")
        if overwrite_mode == "error":
            raise FileExistsError(f"predict: destination already exists: {out_path!r}")
    writer_chunk_size = int(chunk_size) if chunk_size is not None else 8192
    master_dtype = _get_float_precision(data)
    ds = Dataset.for_device("cpu", feature_dtype=master_dtype, label_float_dtype=master_dtype)
    ds.underflow_action = underflow_action
    tmp_dir = new_dir("infer")
    ckpt_dir = os.path.join(tmp_dir, "ckpt")
    memmap_dir = os.path.join(tmp_dir, "memmap")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(memmap_dir, exist_ok=True)
    cleanup_ok = False
    inference_ctx = inference_mode(model)
    with inference_ctx:
        try:
            save_dcp = kwargs.pop("save_dcp", False)
            save_pt = kwargs.pop("save_pt", False)
            dcp_dir = os.path.join(ckpt_dir, "dcp")
            if not (save_dcp or save_pt):
                save_pt = True
            _save_model_checkpoint(
                model, dcp_dir, save_dcp=save_dcp, save_pt=save_pt, overwrite=True
            )
            count = None
            in_dim = None
            if TensorDictBase is not None and isinstance(data, TensorDictBase):
                X_td, _ = get_row(data, labels_required=False)
                if X_td is None:
                    raise ValueError("predict: failed to extract features from TensorDict")
                try:
                    count = int(getattr(X_td, "shape", [0])[0] or 0)
                except Exception:
                    count = int(len(X_td)) if hasattr(X_td, "__len__") else 0
                if count <= 0:
                    raise ValueError("predict: empty input")
                get_batch = _TensorDictSlicer(data)
                in_dim, _ = Storage.stream_memmap(
                    ds=ds,
                    out_dir=memmap_dir,
                    count=int(count),
                    get_batch=get_batch,
                    get_by_indices=None,
                    val_frac=0.0,
                    seed_value=int(seed),
                    underflow_action=underflow_action,
                    shuffle=False,
                    allow_missing_labels=True,
                    features_only=True,
                    chunk_size=writer_chunk_size,
                )
            elif isinstance(data, Mapping) and Storage.is_feature_label_batch_mapping(data):
                f_key = get_feature_key(data)
                if f_key is None:
                    raise ValueError("predict: could not resolve feature key from mapping")
                X_all = data.get(f_key)
                try:
                    count = int(len(X_all))
                except Exception:
                    count = int(getattr(X_all, "shape", [0])[0] or 0)
                if count <= 0:
                    raise ValueError("predict: empty input")
                slice_items = []
                const_items = {}
                for k, v in data.items():
                    if isinstance(v, (str, bytes, bytearray)):
                        const_items[k] = v
                        continue
                    if isinstance(v, Mapping) and (
                        not (TensorDictBase is not None and isinstance(v, TensorDictBase))
                    ):
                        const_items[k] = v
                        continue
                    try:
                        if len(v) == count:
                            slice_items.append((k, v))
                        else:
                            const_items[k] = v
                    except Exception:
                        const_items[k] = v
                get_batch = _MappingSlicer(const_items, tuple(slice_items))
                in_dim, _ = Storage.stream_memmap(
                    ds=ds,
                    out_dir=memmap_dir,
                    count=int(count),
                    get_batch=get_batch,
                    get_by_indices=None,
                    val_frac=0.0,
                    seed_value=int(seed),
                    underflow_action=underflow_action,
                    shuffle=False,
                    allow_missing_labels=True,
                    features_only=True,
                    chunk_size=writer_chunk_size,
                )
            elif isinstance(data, Mapping):
                count, get_batch = Storage.column_cursor(data)
                if int(count) <= 0:
                    raise ValueError("predict: empty input")
                in_dim, _ = Storage.stream_memmap(
                    ds=ds,
                    out_dir=memmap_dir,
                    count=int(count),
                    get_batch=get_batch,
                    get_by_indices=None,
                    val_frac=0.0,
                    seed_value=int(seed),
                    underflow_action=underflow_action,
                    shuffle=False,
                    allow_missing_labels=True,
                    features_only=True,
                    chunk_size=writer_chunk_size,
                )
            else:
                try:
                    count = int(len(data))
                except Exception as e:
                    raise TypeError(
                        "predict: unsupported data type. Expected TensorDict, Mapping, or sized sequence."
                    ) from e
                if count <= 0:
                    raise ValueError("predict: empty input")

                def _seq_get_batch(s: object, e: object) -> object:
                    s_i = int(s)
                    e_i = int(e)
                    try:
                        sl = data[s_i:e_i]
                    except Exception:
                        sl = [data[i] for i in range(s_i, e_i)]

                    return {"features": sl}

                in_dim, _ = Storage.stream_memmap(
                    ds=ds,
                    out_dir=memmap_dir,
                    count=int(count),
                    get_batch=_seq_get_batch,
                    get_by_indices=None,
                    val_frac=0.0,
                    seed_value=int(seed),
                    underflow_action=underflow_action,
                    shuffle=False,
                    allow_missing_labels=True,
                    features_only=True,
                    chunk_size=writer_chunk_size,
                )
            if count is None or in_dim is None:
                raise RuntimeError("predict: failed to infer count/in_dim from input data")
            cfg_obj = None
            with contextlib.suppress(Exception):
                cfg_obj = getattr(model, "config", None)
            if cfg_obj is None:
                with contextlib.suppress(Exception):
                    cfg_obj = getattr(model, "__stnet_instance_config__", None)
            if cfg_obj is None:
                for submodule in model.modules():
                    with contextlib.suppress(Exception):
                        cfg_obj = getattr(submodule, "config", None)
                    if cfg_obj is None:
                        cfg_obj = getattr(submodule, "__stnet_instance_config__", None)
                    if cfg_obj is not None:
                        break
            if isinstance(cfg_obj, ModelConfig):
                cfg_dict = cfg_obj.to_dict()
            elif isinstance(cfg_obj, Mapping):
                cfg_dict = dict(cfg_obj)
            else:
                cfg_dict = {}
            base = {
                "sources": {"kind": "memmap", "path": memmap_dir},
                "ckpt_dir": ckpt_dir,
                "model_ckpt_dir": dcp_dir,
                "in_dim": int(in_dim),
                "out_shape": out_shape_t,
                "cfg_dict": cfg_dict,
            }
            ops_kwargs = dict(kwargs)
            ops_kwargs.setdefault("seed", seed)
            ops_kwargs.setdefault("shuffle", shuffle)
            ops = runtime_config(mode, base, *args, **ops_kwargs)
            _wp = WorkerPolicy.optimize()
            _wp.set_thread_setting()
            nprocs = int(_wp.nproc_per_node)
            max_nodes_i = int(max_nodes) if max_nodes is not None else nprocs
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
            )
            with contextlib.suppress(Exception):
                model.to("cpu")
            _clear_device_caches()
            elastic_launch(lc, process)(ops)
            chunks_dir = os.path.join(ckpt_dir, "pred_chunks")
            if not os.path.isdir(chunks_dir):
                raise RuntimeError(f"predict: missing pred_chunks at {chunks_dir!r}")
            store_float = _get_prediction_dtype(chunks_dir)
            pred_mmt_path = os.path.join(tmp_dir, "pred.mmt")
            Storage.concat_memory_mapped_tensor(
                os.fspath(chunks_dir),
                os.fspath(pred_mmt_path),
                count=count,
                out_shape=out_shape_t,
                store_float=store_float,
            )
            X_mmt = Storage.load_memmap_features(os.fspath(memmap_dir))
            Y_mmt = Storage.open_memory_mapped_tensor(os.fspath(pred_mmt_path))
            if Y_mmt is None:
                raise RuntimeError("predict: failed to open assembled pred.mmt")
            if out_path is not None:
                if os.path.exists(out_path):
                    if overwrite_mode == "resume" and os.path.isfile(out_path):
                        Storage.validate_predictions_h5(os.fspath(out_path), out_shape=out_shape_t)
                        return PersistentTensorDict(filename=out_path, mode="r")
                    if overwrite_mode == "error":
                        raise FileExistsError(f"predict: destination already exists: {out_path!r}")
                out_td = Storage.write_predictions_h5_atomic(
                    os.fspath(out_path),
                    memmap_dir=os.fspath(memmap_dir),
                    pred_path=os.fspath(pred_mmt_path),
                    chunk_size=int(chunk_size or 8192),
                    overwrite=str(overwrite_mode or "replace"),
                )
                if not os.path.isfile(out_path):
                    raise RuntimeError(
                        f"predict: persistent output missing after write: {out_path!r}"
                    )
                Storage.validate_predictions_h5(
                    os.fspath(out_path),
                    out_shape=out_shape_t,
                    in_dim=(int(in_dim) if in_dim is not None else None),
                )
                cleanup_ok = True
                return out_td
            X_t = Storage.copy_mmt_to_cpu_tensor(X_mmt, count=count, chunk_size=writer_chunk_size)
            Y_t = Storage.copy_mmt_to_cpu_tensor(Y_mmt, count=count, chunk_size=writer_chunk_size)
            td_out = TensorDict({"X": X_t, "Y": Y_t}, batch_size=[int(count)])
            cleanup_ok = True
            return td_out
        finally:
            if cleanup_ok:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                with contextlib.suppress(Exception):
                    logger.info("predict debug: preserving tmp_dir=%s", tmp_dir)


@get_execution_time(logger, fn_name="get_prediction")
def get_prediction(
    source: PathLike,
    *args: Any,
    output: str | None = "memory",
    path: PathLike | None = None,
    overwrite: str = "error",
) -> PredictionOutput:
    with inference_mode(torch.nn.Identity()):
        if not bool(source):
            raise ValueError("get_prediction: 'source' must be a non-empty path")
        src = _coerce_path(source)
        if src is None:
            raise ValueError("get_prediction: 'source' is empty/None after normalization")
        output_mode = _coerce_prediction_output(output)
        overwrite_mode = _coerce_prediction_overwrite(overwrite)
        out_path = None
        path_n = _coerce_path(path) if path is not None else None
        if output_mode == "file":
            if path_n is not None:
                run_id = os.path.basename(src.rstrip(os.sep)) or f"prediction-{os.getpid()}"
                out_path = _coerce_prediction_path(path_n, run_id=run_id)
                if out_path is None:
                    logger.warning(
                        "get_prediction: output=%r requires path to be a .h5/.hdf5 file. Got path=%r; falling back to output='memory'.",
                        output,
                        path,
                    )
                    output_mode = "memory"
                elif not _is_path_writable(out_path):
                    logger.warning(
                        "get_prediction: output=%r path is not writable: %r; falling back to output='memory'.",
                        output,
                        out_path,
                    )
                    out_path = None
                    output_mode = "memory"
            else:
                output_mode = "memory"
        if output_mode == "file" and out_path is not None and os.path.exists(out_path):
            if overwrite_mode == "resume" and os.path.isfile(out_path):
                Storage.validate_predictions_h5(os.fspath(out_path))
                return PersistentTensorDict(filename=out_path, mode="r")
            if overwrite_mode == "error":
                raise FileExistsError(f"get_prediction: destination already exists: {out_path!r}")
        if (src.endswith(".h5") or src.endswith(".hdf5")) and os.path.isfile(src):
            if output_mode == "file":
                if out_path is None:
                    return PersistentTensorDict(filename=src, mode="r")
                return Storage.copy_predictions_h5_atomic(
                    os.fspath(src),
                    os.fspath(out_path),
                    overwrite=str(overwrite_mode or "replace"),
                )
            return Storage.load_predictions_h5(os.fspath(src))
        if not os.path.isdir(src):
            raise FileNotFoundError(f"source must be a directory or .h5 file: {src!r}")
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
        if os.path.isfile(pred_path):
            pass
        else:
            if not os.path.isdir(chunks_dir):
                raise FileNotFoundError(f"missing pred_chunks dir: {chunks_dir!r}")
            if count is None or out_shape is None:
                raise FileNotFoundError(
                    f"missing/invalid manifest.json in pred_chunks: {chunks_dir!r}"
                )
            count = int(count)
            out_shape_t = tuple(int(x) for x in (out_shape or ()))
            if count <= 0 or (not out_shape_t) or any(int(d) <= 0 for d in out_shape_t):
                raise ValueError(
                    f"get_prediction: invalid manifest metadata: count={count!r}, out_shape={out_shape!r}"
                )
            store_float = _get_prediction_dtype(chunks_dir)
            Storage.concat_memory_mapped_tensor(
                os.fspath(chunks_dir),
                os.fspath(pred_path),
                count=count,
                out_shape=out_shape_t,
                store_float=store_float,
            )
        X_mmt = Storage.load_memmap_features(os.fspath(memmap_dir))
        Y_mmt = Storage.open_memory_mapped_tensor(os.fspath(pred_path))
        if Y_mmt is None:
            raise RuntimeError("get_prediction: failed to open pred.mmt")
        if count is None:
            try:
                count = int(X_mmt.shape[0])
            except Exception:
                count = None
        if count is None:
            raise RuntimeError("get_prediction: failed to infer count")
        if out_path is not None:
            if os.path.exists(out_path):
                if overwrite_mode == "resume" and os.path.isfile(out_path):
                    Storage.validate_predictions_h5(
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
                        f"get_prediction: destination already exists: {out_path!r}"
                    )
            out_td = Storage.write_predictions_h5_atomic(
                os.fspath(out_path),
                memmap_dir=os.fspath(memmap_dir),
                pred_path=os.fspath(pred_path),
                chunk_size=8192,
                overwrite=str(overwrite_mode or "replace"),
            )
            if not os.path.isfile(out_path):
                raise RuntimeError(
                    f"get_prediction: persistent output missing after write: {out_path!r}"
                )
            Storage.validate_predictions_h5(
                os.fspath(out_path),
                out_shape=tuple(int(d) for d in Y_mmt.shape[1:]),
                in_dim=(
                    int(X_mmt.shape[1])
                    if hasattr(X_mmt, "shape") and len(X_mmt.shape) > 1
                    else None
                ),
            )
            Storage.remove_prediction_artifacts(
                memmap_dir=os.fspath(memmap_dir),
                pred_path=os.fspath(pred_path),
            )
            return out_td
        X_t = Storage.copy_mmt_to_cpu_tensor(X_mmt, count=int(count), chunk_size=8192)
        Y_t = Storage.copy_mmt_to_cpu_tensor(Y_mmt, count=int(count), chunk_size=8192)
        td_out = TensorDict({"X": X_t, "Y": Y_t}, batch_size=[int(count)])
        Storage.remove_prediction_artifacts(
            memmap_dir=os.fspath(memmap_dir),
            pred_path=os.fspath(pred_path),
        )
        return td_out


class _MappingSlicer:
    __slots__ = ("const_items", "slice_items")

    def __init__(self, const_items: Mapping[Any, Any], slice_items: Tuple[Any, ...]) -> None:
        self.const_items = dict(const_items)
        self.slice_items = tuple(slice_items)

    def __call__(self, s: object, e: object) -> Mapping[Any, Any]:
        batch = dict(self.const_items)
        for k, v in self.slice_items:
            try:
                batch[k] = v[s:e]
            except Exception:
                batch[k] = v
        return batch


class _TensorDictSlicer:
    __slots__ = ("td",)

    def __init__(self, td: TensorDictBase) -> None:
        self.td = td

    def __call__(self, s: object, e: object) -> TensorDictBase:
        return self.td[s:e]
