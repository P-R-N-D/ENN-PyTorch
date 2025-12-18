# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import gc
import json
import os
import random
import shutil
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from collections.abc import Mapping as _MappingABC

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
from ..backend.system import (
    WorkerPolicy,
    initialize_python_path,
    new_dir,
    optimal_start_method,
    set_multiprocessing_env,
)
from ..data.pipeline import Dataset, default_underflow_action, normalize_underflow_action
from ..data.nodes import preload_memmap
from ..model.nn import History, Root
from .config import (ModelConfig, OpsMode, RuntimeConfig, coerce_model_config,
                     runtime_config)


def _resolve_memmap_store_float(*, negotiable: bool) -> torch.dtype:
    # Match nodes.preload_memmap behavior:
    # - default float32 when negotiable
    # - allow forcing float64 via STNET_MEMMAP_FLOAT_DTYPE=torch.float64
    req = str(os.environ.get("STNET_MEMMAP_FLOAT_DTYPE", "") or "").strip()
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
    # (필수) dtype는 identity(is) 비교 보장 없음 → 값 비교 사용
    return torch.float32 if (negotiable and req_dtype != torch.float64) else torch.float64


class _KeySliceMappingView(_MappingABC):
    __slots__ = ("_data", "_keys", "_s", "_e")

    def __init__(self, data: Mapping[Any, Any], keys: Sequence[Any], s: int, e: int):
        self._data = data
        self._keys = keys
        self._s = int(s)
        self._e = int(e)

    def __len__(self) -> int:
        n = self._e - self._s
        return n if n > 0 else 0

    def __iter__(self):
        for i in range(self._s, self._e):
            yield self._keys[i]

    def __getitem__(self, k):
        return self._data[k]


def _write_memmap_streaming_two_pass(
    *,
    ds: Dataset,
    out_dir: str,
    count: int,
    get_batch: Callable[[int, int], Any],
    val_frac: float,
    seed_value: Optional[int],
    underflow_action: str,
    default_label_shape: Optional[Tuple[int, ...]] = None,
    allow_missing_labels: bool = False,
    chunk_size: int = 32,
) -> Tuple[int, Tuple[int, ...]]:
    os.makedirs(out_dir, exist_ok=True)
    if count <= 0:
        raise ValueError("count must be > 0")

    chunk = max(1, min(int(chunk_size), int(count)))

    # 중복 제거: 공통 CPU/contiguous 정규화
    def _to_cpu_contig(t: torch.Tensor) -> torch.Tensor:
        t = t.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        if not t.is_contiguous():
            t = t.contiguous()
        return t

    def _flat2d_cpu_contig(t: torch.Tensor, n: int) -> torch.Tensor:
        # 핫픽스: view는 비연속 텐서에서 터질 수 있음 → CPU+contig 보장 후 reshape 사용
        t_cpu = _to_cpu_contig(t)
        if t_cpu.ndim == 0:
            # 핫픽스2: 극단 케이스에서도 reshape가 안전하게 되도록
            t_cpu = t_cpu.reshape(1)
        return t_cpu.reshape(int(n), -1)

    # 중복 제거: batch 길이 계산을 한 군데로
    def _batch_n(x: torch.Tensor) -> int:
        xd = int(getattr(x, "ndim", 0) or 0)
        return int(x.shape[0]) if xd > 0 else 1

    stats: Dict[str, Any] = {
        "has_scale": False,
        "has_nonfinite": False,
        "scale_max_abs": None,
        "scale_min_positive": None,
        "scale_is_integral": None,
    }
    in_dim: Optional[int] = None
    label_shape: Optional[Tuple[int, ...]] = None

    for s in range(0, count, chunk):
        e = min(count, s + chunk)
        batch = get_batch(s, e)
        fx, lb, _, _ = ds.preprocess(batch)
        n = _batch_n(fx)
        if n <= 0:
            continue
        fx_flat = _flat2d_cpu_contig(fx, n)
        cur_in_dim = int(fx_flat.shape[1])
        if in_dim is None:
            in_dim = cur_in_dim
        elif cur_in_dim != int(in_dim):
            raise RuntimeError(f"feature dim mismatch: expected {in_dim}, got {cur_in_dim}")

        if lb is None:
            if not allow_missing_labels:
                raise RuntimeError("streaming memmap writer requires labels tensor (non-None)")
            if default_label_shape is None:
                raise RuntimeError("labels are missing and default_label_shape was not provided")
            cur_label_shape = tuple(default_label_shape)
            # For missing labels, treat scale stats as all-zeros (safe, no allocation).
            l_stats = {
                "has_scale": True,
                "has_nonfinite": False,
                "scale_max_abs": 0.0,
                "scale_min_positive": None,
                "scale_is_integral": None,
            }
        else:
            cur_label_shape = tuple(lb.shape[1:])
            lb_flat = _flat2d_cpu_contig(lb, n)
            l_stats = Dataset.tensor_scale_stats(lb_flat)
        if label_shape is None:
            label_shape = cur_label_shape
        elif tuple(label_shape) != tuple(cur_label_shape):
            raise RuntimeError(f"label shape mismatch: expected {label_shape}, got {cur_label_shape}")

        f_stats = Dataset.tensor_scale_stats(fx_flat)
        stats = Dataset.merge_scale_stats(stats, Dataset.merge_scale_stats(f_stats, l_stats))

    if in_dim is None or label_shape is None:
        raise RuntimeError("Failed to infer in_dim/label_shape from data")

    # decide storage dtype (float32 if negotiable, else float64)
    negotiable = Dataset.is_fp32_castable(stats, underflow_action=underflow_action, safety_margin=1.0)
    store_float = _resolve_memmap_store_float(negotiable=bool(negotiable))

    features_path = os.path.join(out_dir, "features.mmt")
    labels_path = os.path.join(out_dir, "labels.mmt")

    features_mmt = MemoryMappedTensor.empty((count, int(in_dim)), dtype=store_float, filename=features_path, existsok=True)
    labels_mmt = MemoryMappedTensor.empty((count, *tuple(label_shape)), dtype=store_float, filename=labels_path, existsok=True)

    # (권장1) missing-label 경로에서 chunk마다 zeros를 새로 만들지 말고 재사용
    zeros_label_buf: Optional[torch.Tensor] = None
    if allow_missing_labels:
        # (권장3) device를 명시적으로 torch.device("cpu")로
        zeros_label_buf = torch.zeros(
            (chunk, *tuple(label_shape)),
            dtype=store_float,
            device=torch.device("cpu"),
        )

    # ---- pass 2: write chunks
    written = 0
    for s in range(0, count, chunk):
        e = min(count, s + chunk)
        batch = get_batch(s, e)
        fx, lb, _, _ = ds.preprocess(batch)
        n = _batch_n(fx)
        if n <= 0:
            continue
        fx_flat = _flat2d_cpu_contig(fx, n)
        if int(fx_flat.shape[1]) != int(in_dim):
            raise RuntimeError(f"feature dim mismatch: expected {in_dim}, got {int(fx_flat.shape[1])}")

        # dtype 변환은 필요할 때만 (미세 최적화)
        fx_out = fx_flat if fx_flat.dtype == store_float else fx_flat.to(dtype=store_float)
        if lb is None:
            if not allow_missing_labels:
                raise RuntimeError("streaming memmap writer requires labels tensor (non-None)")
            assert zeros_label_buf is not None
            lb_out = zeros_label_buf[:n]
        else:
            if tuple(lb.shape[1:]) != tuple(label_shape):
                raise RuntimeError(f"label shape mismatch: expected {label_shape}, got {tuple(lb.shape[1:])}")
            lb_cpu = _to_cpu_contig(lb)
            lb_out = lb_cpu if lb_cpu.dtype == store_float else lb_cpu.to(dtype=store_float)
        features_mmt[s : s + n].copy_(fx_out)
        labels_mmt[s : s + n].copy_(lb_out)
        written += n

    if written != int(count):
        raise RuntimeError(f"memmap written={written}, expected={count}")

    # split metadata (no physical shuffle; sampler should shuffle)
    val_count = max(0, min(count, int(round(count * float(val_frac)))))
    train_count = max(0, count - val_count)
    train_start, train_end = 0, train_count
    val_start, val_end = train_end, train_end + val_count

    meta_json: Dict[str, Any] = {
        "N": int(count),
        "feature_dim": int(in_dim),
        "features_path": "features.mmt",
        "labels_path": "labels.mmt",
        "label_shape": list(label_shape),
        "features_dtype": str(store_float).replace("torch.", ""),
        "labels_dtype": str(store_float).replace("torch.", ""),
        "fractions": [float(1.0 - float(val_frac)), float(val_frac)],
        "shuffled": False,
        "shuffle_seed": int(seed_value) if seed_value is not None else None,
        "shuffle_mode": "none",
        "train_start": int(train_start),
        "train_end": int(train_end),
        "val_start": int(val_start),
        "val_end": int(val_end),
        # scale negotiation metadata
        "has_scale": bool(stats.get("has_scale")),
        "has_nonfinite": bool(stats.get("has_nonfinite")),
        "scale_max_abs": stats.get("scale_max_abs"),
        "scale_min_positive": stats.get("scale_min_positive"),
        "scale_is_integral": stats.get("scale_is_integral"),
        "is_negotiable": bool(negotiable),
        "underflow_action": str(underflow_action),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_json, f)

    return int(in_dim), tuple(label_shape)


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
    model: Root,
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
) -> Root:
    _reset_process_group()
    try:
        val_frac = float(val_frac)
        val_frac = 0.0 if val_frac < 0.0 else (1.0 if val_frac > 1.0 else val_frac)
    except (TypeError, ValueError):
        val_frac = 0.1

    seed_value = _ensure_seed(seed)
    _seed_everything(seed_value)

    underflow_action = normalize_underflow_action(kwargs.pop("underflow_action", None), default=default_underflow_action())

    with contextlib.suppress(Exception):
        torch.use_deterministic_algorithms(False, warn_only=True)
    with contextlib.suppress(Exception):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    ds_meta = Dataset.for_device("cpu", feature_dtype=torch.float64, label_float_dtype=torch.float64)
    ds_meta.underflow_action = underflow_action

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

            in_dim, label_shape = _write_memmap_streaming_two_pass(
                ds=ds_meta,
                out_dir=out_dir,
                count=count,
                get_batch=lambda s, e: td[s:e],
                val_frac=float(val_frac),
                seed_value=seed_value,
                underflow_action=underflow_action,
                allow_missing_labels=False,
                chunk_size=32,
            )

            return int(in_dim), tuple(label_shape), int(count)

        if (
            isinstance(d, _Mapping)
            and not isinstance(d, TensorDictBase)
            and d
            and all(not isinstance(v, _Mapping) for v in d.values())
        ):
            keys = list(d.keys())
            count = len(keys)
            if count <= 0:
                raise ValueError("Empty dataset provided to train().")
            def _get_batch(s: int, e: int):
                return _KeySliceMappingView(d, keys, s, e)

            in_dim, label_shape = _write_memmap_streaming_two_pass(
                ds=ds_meta,
                out_dir=out_dir,
                count=count,
                get_batch=_get_batch,
                val_frac=float(val_frac),
                seed_value=seed_value,
                underflow_action=underflow_action,
                allow_missing_labels=False,
                chunk_size=32,
            )

            return int(in_dim), tuple(label_shape), int(count)

        fx, lb, _, lshape = ds_meta.preprocess(d)
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
            shuffle=False,
            seed=seed_value,
            underflow_action=underflow_action,
        )
        del fx, lb
        return int(in_dim), tuple(lshape), int(count)

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
        _wp = WorkerPolicy.autotune()
        _wp.apply_torch_threads()
        nprocs = int(_wp.nproc_per_node)
        cfg_obj = getattr(model, "_Root__config", None)
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
    cfg_obj = getattr(model, "_Root__config", None)
    if isinstance(cfg_obj, (ModelConfig, dict)):
        cfg_model = coerce_model_config(cfg_obj)
    else:
        cfg_model = ModelConfig()
    cfg_dict = asdict(cfg_model)
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

        in_dim, label_shape = _write_memmap_streaming_two_pass(
            ds=ds,
            out_dir=memmap_dir,
            count=count,
            get_batch=lambda s, e: td[s:e],
            val_frac=0.0,
            seed_value=seed_value,
            underflow_action=underflow_action,
            default_label_shape=default_out_shape,
            allow_missing_labels=True,
            chunk_size=32,
        )

        keys = list(range(count))

    elif (
        isinstance(data, _Mapping)
        and not isinstance(data, TensorDictBase)
        and data
        and all(not isinstance(v, _Mapping) for v in data.values())
    ):
        keys = list(data.keys())
        count = len(keys)
        if count <= 0:
            return {}

        def _get_batch(s: int, e: int):
            return _KeySliceMappingView(data, keys, s, e)

        in_dim, label_shape = _write_memmap_streaming_two_pass(
            ds=ds,
            out_dir=memmap_dir,
            count=count,
            get_batch=_get_batch,
            val_frac=0.0,
            seed_value=seed_value,
            underflow_action=underflow_action,
            default_label_shape=default_out_shape,
            allow_missing_labels=True,
            chunk_size=32,
        )

    else:
        feats, labels, keys, label_shape = ds.preprocess(data)
        feats = feats.contiguous()
        if labels is None:
            out_shape = tuple(getattr(model, "out_shape", ()))
            if not out_shape:
                out_shape = (1,)
            labels = torch.zeros((int(feats.shape[0]), *tuple(out_shape)), dtype=torch.float64)
        labels = labels.contiguous()

        count = int(feats.shape[0])
        if count <= 0:
            return {}

        in_dim = int(feats.reshape(count, -1).shape[1])

        preload_memmap(
            {"features": feats, "labels": labels},
            memmap_dir=memmap_dir,
            train_frac=1.0,
            val_frac=0.0,
            shuffle=False,
            seed=seed_value,
            underflow_action=underflow_action,
        )
        if keys is None:
            keys = list(range(count))
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
            import logging

            log = logging.getLogger(__name__)
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
                    torch.save(keys, os.path.join(chunks_dir, "keys.pt"))
                except Exception as e:
                    log.warning("predict: failed to write keys.pt (ignored): %r", e, exc_info=True)
            final_dir = new_dir("predictions")
            moved_dir = shutil.move(chunks_dir, final_dir)
            chunk_root = moved_dir if os.path.isdir(moved_dir) else os.path.join(
                final_dir, os.path.basename(chunks_dir)
            )
            return get_prediction(chunk_root, output=output_mode, lazy=lazy_flag)

        return {}
    finally:
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
                tensor = MemoryMappedTensor.from_filename(base_mmt)
            except Exception:
                with contextlib.suppress(Exception):
                    tensor = torch.load(base_mmt, map_location="cpu")
        else:
            alt_pt = os.path.join(chunk_root, f"chunk_{idx:06d}.pt")
            if os.path.exists(alt_pt):
                tensor = torch.load(alt_pt, map_location="cpu")
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
        keys = torch.load(keys_pt, map_location="cpu")
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
        if lazy:
            from ..data.collections import LazyDict

            row_to_part = np.full((nkeys,), -1, dtype=np.int32)
            row_to_off = np.full((nkeys,), -1, dtype=np.int32)
            pred_paths: list[Optional[str]] = [None] * len(parts_list)

            _dup_env = str(os.environ.get("STNET_PRED_CHECK_ROWIDS_DUP", "")).strip().lower()
            check_dups = _dup_env in {"1", "true", "yes", "y", "on"}

            for p_idx, part in enumerate(parts_list):
                rows_name = (part or {}).get("rows")
                pred_name = (part or {}).get("pred")
                if not rows_name or not pred_name:
                    continue
                rows_path = os.path.join(chunk_root, rows_name)
                pred_paths[p_idx] = os.path.join(chunk_root, pred_name)

                rows = torch.load(rows_path, map_location="cpu")
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

            _cache_part = {"idx": -1, "pred": None}

            def _pred_for_row(row_id: int) -> torch.Tensor:
                p = int(row_to_part[int(row_id)])
                if p < 0:
                    raise KeyError(row_id)
                off = int(row_to_off[int(row_id)])
                if _cache_part["idx"] != p or _cache_part["pred"] is None:
                    path = pred_paths[p]
                    if not path:
                        raise KeyError(p)
                    _cache_part["pred"] = torch.load(path, map_location="cpu")
                    _cache_part["idx"] = p
                return _cache_part["pred"][off].detach()

            def _getter(key: Any) -> torch.Tensor:
                rid = _row_for_key(key)
                return _pred_for_row(rid)

            return LazyDict(key_seq, _getter, name="predictions", cache=False)

        out: Dict[Tuple[Any, ...], torch.Tensor] = {}
        for part in parts_list:
            rows_name = (part or {}).get("rows")
            pred_name = (part or {}).get("pred")
            if not rows_name or not pred_name:
                continue
            rows_path = os.path.join(chunk_root, rows_name)
            pred_path = os.path.join(chunk_root, pred_name)
            rows = torch.load(rows_path, map_location="cpu")
            preds = torch.load(pred_path, map_location="cpu")
            if not isinstance(rows, torch.Tensor):
                rows = torch.as_tensor(rows)
            if not isinstance(preds, torch.Tensor):
                preds = torch.as_tensor(preds)
            rows = rows.to(dtype=torch.int64).reshape(-1).contiguous()
            if int(preds.shape[0]) != int(rows.shape[0]):
                raise RuntimeError(
                    f"get_prediction: part size mismatch rows={int(rows.shape[0])} preds={int(preds.shape[0])} ({rows_name},{pred_name})"
                )
            rows_np = rows.numpy()
            for j, rid in enumerate(rows_np):
                rid_i = int(rid)
                k = (rid_i,) if isinstance(keys, range) else (fixed_keys[rid_i] if fixed_keys is not None else (rid_i,))
                out[k] = preds[j].detach()
        return out

    num_chunks = int(manifest.get("num_chunks", 0) or 0)
    flat = _load_legacy_flat(chunk_root, num_chunks=num_chunks, out_shape=out_shape)

    pred_tensor = Root.unflatten_y(flat, out_shape)
    out_legacy: Dict[Tuple[Any, ...], torch.Tensor] = {}
    for i, k in enumerate(key_seq):
        if i >= int(pred_tensor.shape[0]):
            break
        out_legacy[k] = pred_tensor[i].detach().cpu().to(dtype=torch.float64)
    return out_legacy
