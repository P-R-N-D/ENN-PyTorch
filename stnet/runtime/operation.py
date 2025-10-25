# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import sys
import time
import warnings
from collections import deque
from dataclasses import asdict, replace
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
import torch.distributed
import torch.multiprocessing as mp
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load,
    save,
)
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh

try:  # pragma: no cover - optional distributed helpers
    from torch.distributed import all_reduce as _all_reduce
    from torch.distributed import get_world_size as _get_world_size
    from torch.distributed import is_initialized as _dist_is_initialized
    from torch.distributed import ReduceOp as _ReduceOp
except Exception:  # pragma: no cover - CPU-only or minimal builds
    _all_reduce = None
    _get_world_size = lambda: 1  # type: ignore[assignment]
    _dist_is_initialized = lambda: False  # type: ignore[assignment]
    _ReduceOp = None

try:  # pragma: no cover - optional dependency
    from torch.distributed.tensor import DTensor  # type: ignore
except (ImportError, AttributeError):  # pragma: no cover - environment without DTensor
    DTensor = ()  # type: ignore


def _safe_clip_grad_norm(
    parameters: Sequence[torch.nn.Parameter],
    max_norm: float,
    norm_type: float = 2.0,
    eps: float = 1e-6,
) -> float:
    """Clip gradients while avoiding DTensor/Tensor mixing errors."""

    grads: List[torch.Tensor] = []
    original_grads: List[torch.Tensor] = []
    for param in parameters:
        if param is None:
            continue
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        original_grads.append(grad)
        if DTensor and isinstance(grad, DTensor):
            grads.append(grad.to_local())
        else:
            grads.append(grad)

    if not grads:
        return 0.0

    norm_vals = []
    if norm_type == float("inf"):
        for grad in grads:
            norm_vals.append(
                grad.detach()
                .abs()
                .max()
                .to(device="cpu", dtype=torch.float32)
            )
        total_norm_tensor = torch.stack(norm_vals).max()
    else:
        for grad in grads:
            norm_vals.append(
                grad.detach()
                .norm(norm_type)
                .to(device="cpu", dtype=torch.float32)
            )
        total_norm_tensor = torch.stack(norm_vals).norm(norm_type)

    total_norm = float(total_norm_tensor)
    clip_coef = max_norm / (total_norm + eps)
    clip_coef = min(clip_coef, 1.0)

    if clip_coef < 1.0:
        for grad in original_grads:
            grad.mul_(clip_coef)

    return total_norm
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from tqdm.auto import tqdm

from ..model import Root
from ..config import (
    ModelConfig,
    OpsMode,
    RuntimeConfig,
    coerce_model_config,
    runtime_config,
)
from ..model.functional import StandardNormalLoss, StudentsTLoss, TiledLoss
from ..data.collate import dataloader
from ..data.transforms import postprocess, preprocess
from ..data.stats import compute_y_range
from ..utils.dtypes import to_torch
from ..data.dataset import SampleReader
from ..utils.platform import Distributed, Network, System
from ..utils.optimization import (
    AdamW,
    AutoCast,
    LossWeightController,
    DataScale,
    Module,
    _supports_scale,
    inference,
    joining,
    no_synchronization,
)
from ..utils.profiler import FlopCounter


def _float8_scale_supported(scale: Optional[DataScale]) -> bool:
    if scale is None:
        return True
    try:
        candidates = AutoCast._float8_dtypes()
    except Exception:
        return False
    if not candidates:
        return False
    for dtype in candidates:
        if isinstance(dtype, torch.dtype) and _supports_scale(
            dtype, scale, safety_margin=2.0
        ):
            return True
    return False


def _resolve_data_scale_hint(value: Any, *, _depth: int = 0) -> Optional[DataScale]:
    if value is None or _depth > 4:
        return None
    if isinstance(value, DataScale):
        return value
    candidate = getattr(value, "__data_scale_hint__", None)
    if isinstance(candidate, DataScale):
        return candidate
    for attr in ("_fsdp_wrapped_module", "_orig_module", "module"):
        nested = None
        with contextlib.suppress(Exception):
            nested = getattr(value, attr)
        if nested is None or nested is value:
            continue
        resolved = _resolve_data_scale_hint(nested, _depth=_depth + 1)
        if resolved is not None:
            return resolved
    return None


def _resolve_module_device(value: Any, *, _depth: int = 0) -> Optional[torch.device]:
    if value is None or _depth > 4:
        return None
    if isinstance(value, torch.device):
        return value
    if isinstance(value, torch.Tensor):
        return value.device
    candidate = getattr(value, "device", None)
    if isinstance(candidate, torch.device):
        return candidate
    if isinstance(candidate, torch.Tensor):
        return candidate.device
    if isinstance(value, torch.nn.Module):
        with contextlib.suppress(Exception):
            param = next(value.parameters(recurse=True), None)
            if isinstance(param, torch.nn.Parameter):
                return param.device
        with contextlib.suppress(Exception):
            buffer = next(value.buffers(recurse=True), None)
            if isinstance(buffer, torch.Tensor):
                return buffer.device
    for attr in ("_fsdp_wrapped_module", "_orig_module", "module"):
        nested = None
        with contextlib.suppress(Exception):
            nested = getattr(value, attr)
        if nested is None or nested is value:
            continue
        resolved = _resolve_module_device(nested, _depth=_depth + 1)
        if resolved is not None:
            return resolved
    return None


class _QuantileState:
    __slots__ = (
        "q_lo",
        "q_hi",
        "win_lo",
        "win_hi",
        "beta",
        "mode",
        "warmup",
        "count",
        "sync",
    )

    def __init__(
        self,
        mode: str,
        beta: float,
        win_size: int,
        warmup_steps: int,
        sync: str,
    ) -> None:
        self.q_lo: Optional[torch.Tensor] = None
        self.q_hi: Optional[torch.Tensor] = None
        self.win_lo: deque[float] = deque(maxlen=max(1, int(win_size)))
        self.win_hi: deque[float] = deque(maxlen=max(1, int(win_size)))
        self.beta = float(beta)
        self.mode = str(mode)
        self.warmup = int(max(0, warmup_steps))
        self.count = 0
        self.sync = str(sync)


def _finite_quantiles(y: torch.Tensor, probs: Sequence[float]) -> List[torch.Tensor]:
    vals = torch.nan_to_num(y.detach()).to(dtype=torch.float64)
    finite_mask = torch.isfinite(vals)
    if not bool(finite_mask.any()):
        return [
            torch.tensor(0.0, device=vals.device, dtype=torch.float64) for _ in probs
        ]
    qs = torch.as_tensor(list(probs), device=vals.device, dtype=torch.float64)
    quantiles = torch.quantile(vals[finite_mask], qs)
    return [q.to(dtype=torch.float64) for q in quantiles.unbind()]


def _sync_scalar(value: torch.Tensor, how: str) -> torch.Tensor:
    if _all_reduce is None or _ReduceOp is None or not _dist_is_initialized():
        return value.to(dtype=torch.float64)
    result = value.to(dtype=torch.float64)
    if how == "mean":
        _all_reduce(result, op=_ReduceOp.SUM)
        world = max(1, int(_get_world_size()))
        result /= world
    elif how == "min":
        _all_reduce(result, op=_ReduceOp.MIN)
    elif how == "max":
        _all_reduce(result, op=_ReduceOp.MAX)
    # "first" keeps local value (rank 0 broadcast could be added later)
    return result


def _update_quantiles(
    state: _QuantileState,
    target: torch.Tensor,
    p_lo: float,
    p_hi: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_lo_batch, q_hi_batch = _finite_quantiles(target, [p_lo, p_hi])
    q_lo_batch = q_lo_batch.to(dtype=torch.float64)
    q_hi_batch = q_hi_batch.to(dtype=torch.float64)
    q_lo_batch = _sync_scalar(q_lo_batch, state.sync)
    q_hi_batch = _sync_scalar(q_hi_batch, state.sync)

    if state.count < state.warmup or state.mode == "none":
        q_lo = q_lo_batch
        q_hi = q_hi_batch
    elif state.mode == "ema":
        if state.q_lo is None:
            state.q_lo = q_lo_batch
        if state.q_hi is None:
            state.q_hi = q_hi_batch
        q_lo = state.beta * state.q_lo + (1.0 - state.beta) * q_lo_batch
        q_hi = state.beta * state.q_hi + (1.0 - state.beta) * q_hi_batch
    else:  # window
        state.win_lo.append(float(q_lo_batch.detach().cpu()))
        state.win_hi.append(float(q_hi_batch.detach().cpu()))
        q_lo = torch.tensor(
            np.median(np.asarray(state.win_lo, dtype=np.float64)),
            device=target.device,
            dtype=torch.float64,
        )
        q_hi = torch.tensor(
            np.median(np.asarray(state.win_hi, dtype=np.float64)),
            device=target.device,
            dtype=torch.float64,
        )

    state.q_lo = q_lo
    state.q_hi = q_hi
    state.count += 1
    return q_lo, q_hi


try:
    from torchao.float8 import (
        precompute_float8_dynamic_scale_for_fsdp as _torchao_precompute_float8_dynamic_scale_for_fsdp,
    )
except ImportError:

    def precompute_float8_dynamic_scale_for_fsdp(*args: Any, **kwargs: Any) -> Any:
        return None
else:

    def precompute_float8_dynamic_scale_for_fsdp(*args: Any, **kwargs: Any) -> Any:
        if not args and "module" not in kwargs:
            return None
        module = args[0] if args else kwargs.get("module")
        device_hint = kwargs.get("device")
        device: Optional[torch.device]
        with contextlib.suppress(Exception):
            device = torch.device(device_hint) if device_hint is not None else None
        if device is None:
            device = _resolve_module_device(module)
        ok, _ = System.is_float8_supported(device) if device is not None else System.is_float8_supported()
        if not ok:
            return None
        scale_hint = _resolve_data_scale_hint(kwargs.pop("data_scale", None))
        if scale_hint is None:
            scale_hint = _resolve_data_scale_hint(module)
        if scale_hint is not None and not _float8_scale_supported(scale_hint):
            return None
        return _torchao_precompute_float8_dynamic_scale_for_fsdp(*args, **kwargs)


try:
    from torch.distributed.run import LaunchConfig, elastic_launch
except ImportError:
    from torch.distributed.launcher.api import LaunchConfig, elastic_launch



ignored_sentences = [
    "torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to load in a single process.*",
    "torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to save in a single process.*",
    "TypedStorage is deprecated.*",
]
ignored_pattern = "|".join((f"({sentence})" for sentence in ignored_sentences))

_DL_STATE_FILE = "dataloader.json"
_FLOAT8_LOG_MESSAGES: set[str] = set()


def dl_state_path(directory: str) -> str:
    return os.path.join(directory, _DL_STATE_FILE)


def _float8_log(msg: str, *, only_main_rank: bool = True) -> None:
    text = str(msg)
    if text in _FLOAT8_LOG_MESSAGES:
        return
    _FLOAT8_LOG_MESSAGES.add(text)
    if not only_main_rank:
        warnings.warn(text)
        return
    try:
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
    except Exception:
        pass
    warnings.warn(text)


def _accumulate_data_scale(
    scale: Optional[DataScale], *values: Any
) -> Optional[DataScale]:
    current = scale
    for value in values:
        if value is None:
            continue
        tensor: Optional[torch.Tensor]
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            try:
                tensor = to_torch(value)
            except Exception:
                tensor = None
        if tensor is None:
            continue
        current = DataScale.accumulate(current, tensor)
    return current


def _prune_dcp_state_keys(state: Any) -> Any:
    try:
        keys = []
        for key in state.keys():
            s = str(key)
            if s.endswith("._extra_state") or s.endswith("_extra_state"):
                keys.append(key)
    except (AttributeError, TypeError):
        return state
    for key in keys:
        state.pop(key, None)
    return state


_SIZEOF = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}


def _canonical_dtype(x: torch.dtype | str) -> str:
    if isinstance(x, torch.dtype):
        s = str(x).lower()
    else:
        s = str(x).strip().lower()
    if s.startswith("torch."):
        s = s.split(".", 1)[1]
    s = s.lstrip("<>|=")
    aliases = {
        "float": "float32",
        "double": "float64",
        "half": "float16",
        "halffloat": "float16",
        "boolean": "bool",
        "bool_": "bool",
        "bf16": "bfloat16",
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "u8": "uint8",
    }
    return aliases.get(s, s)


def _size(dtype: torch.dtype | str) -> int:
    try:
        return _SIZEOF[_canonical_dtype(dtype)]
    except KeyError as exc:
        raise TypeError(f"unsupported dtype: {dtype}") from exc


def _status_bar(activity: str, total: int, dev: torch.device) -> tqdm:
    device_label = dev.type.upper()
    bar = tqdm(
        total=total,
        desc=f"{activity} ({device_label})",
        unit="step",
        bar_format=(
            "{desc}{bar} {percentage:3.0f}% {postfix} Elapsed: {elapsed}, Remaining: {remaining}"
        ),
        colour="green",
        position=0,
        leave=False,
    )
    bar.set_postfix_str("0.00 MB/s, 0.00 TFLOPS", refresh=True)
    return bar


def _loader_length(loader: Any) -> int:
    if loader is None:
        return 0
    try:
        length = len(loader)
    except Exception:
        return 0
    if isinstance(length, int) and length >= 0:
        return length
    try:
        return int(length)
    except Exception:
        return 0


def _format_metrics_postfix(
    mbps: float,
    tflops: float,
    *,
    comp_elapsed: Optional[float] = None,
    flop_breakdown: Optional[Dict[str, float]] = None,
) -> str:
    postfix = f"{mbps:.2f} MB/s, {tflops:.2f} TFLOPS"
    if comp_elapsed is None or not flop_breakdown:
        return postfix
    manual_total = 0.0
    attn_total = 0.0
    ret_total = 0.0
    for name, value in flop_breakdown.items():
        try:
            fv = float(value)
        except Exception:
            continue
        if fv <= 0.0:
            continue
        manual_total += fv
        if name == "Attention":
            attn_total += fv
        if name in {"Retention", "MSRCompat"}:
            ret_total += fv
    if manual_total <= 0.0:
        return postfix
    attn_pct = (attn_total / manual_total) * 100.0 if attn_total > 0.0 else 0.0
    ret_pct = (ret_total / manual_total) * 100.0 if ret_total > 0.0 else 0.0
    comp_sec = max(float(comp_elapsed), 1e-06)
    attn_rate = attn_total / comp_sec / 1_000_000_000_000.0
    ret_rate = ret_total / comp_sec / 1_000_000_000_000.0
    postfix += (
        f" | Attn {attn_pct:.0f}%/{attn_rate:.2f}T | Ret {ret_pct:.0f}%/{ret_rate:.2f}T"
    )
    return postfix


def _advance_status_bar(
    status_bar: Optional[tqdm],
    increment: int,
    mbps: float,
    tflops: float,
    *,
    comp_elapsed: Optional[float] = None,
    flop_breakdown: Optional[Dict[str, float]] = None,
) -> None:
    if status_bar is None or increment <= 0:
        return
    target_total = status_bar.n + increment
    current_total = status_bar.total or 0
    if target_total > current_total:
        status_bar.total = target_total
    postfix = _format_metrics_postfix(
        mbps,
        tflops,
        comp_elapsed=comp_elapsed,
        flop_breakdown=flop_breakdown,
    )
    status_bar.set_postfix_str(postfix, refresh=False)
    status_bar.update(increment)


def _backend_type(device: torch.device) -> str:
    if device.type == "cuda":
        return "nccl"
    if device.type == "xpu":
        return "xccl"
    return "gloo"


def _set_backend(device: torch.device) -> None:
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if device.type == "cuda":
        torch.cuda.set_device(rank)
    elif device.type == "xpu":
        torch.xpu.set_device(rank)
    else:
        try:
            import netifaces

            gws = netifaces.gateways()
            iface: str | None = None
            default_gateways = gws.get("default", {}) if isinstance(gws, dict) else {}
            families = []
            with contextlib.suppress(AttributeError):
                families.append(netifaces.AF_INET6)
            families.append(netifaces.AF_INET)
            for family in families:
                info = default_gateways.get(family)
                if info and len(info) >= 2:
                    iface = info[1]
                    if iface:
                        break
            if iface:
                os.environ.setdefault("GLOO_SOCKET_IFNAME", iface)
                os.environ.setdefault("TP_SOCKET_IFNAME", iface)
        except (ImportError, KeyError, OSError):
            pass


def _meta(memmap_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(memmap_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)
def _ensure_uniform_param_dtype(
    model: Any, prefer: Optional[torch.dtype] = None
) -> Optional[torch.dtype]:
    dtypes = set((p.dtype for p in model.parameters() if p is not None))
    if len(dtypes) <= 1:
        return None
    if prefer is not None:
        tgt = prefer
    elif torch.bfloat16 in dtypes:
        tgt = torch.bfloat16
    elif torch.float16 in dtypes:
        tgt = torch.float16
    else:
        tgt = torch.float32
    for mod in model.modules():
        params = getattr(mod, "_parameters", None)
        if not params:
            continue
        for name, p in list(params.items()):
            if p is None or p.dtype == tgt:
                continue
            new_p = torch.nn.Parameter(
                p.detach().to(tgt), requires_grad=p.requires_grad
            )
            setattr(mod, name, new_p)
    return tgt


def train(
    model: Root,
    data: Dict[Tuple, torch.Tensor],
    *args: Any,
    epochs: int = 5,
    batch_size: int = 128,
    val_frac: float = 0.1,
    base_lr: float = 0.001,
    weight_decay: float = 0.0001,
    warmup_ratio: float = 0.0,
    eta_min: float = 0.0,
    run_id: str = "torch",
    seed: int = 42,
    max_nodes: int = 1,
    rdzv_backend: Optional[str] = "c10d",
    rdzv_endpoint: Optional[str] = None,
    prefetch_factor: Optional[int] = 1,
    grad_accum_steps: int = 1,
    overlap_h2d: bool = True,
    loss_tile_dim: Optional[int] = None,
    loss_tile_size: Optional[int] = None,
    loss_mask_mode: str = "none",
    loss_mask_value: Optional[float] = None,
    **kwargs: Any,
) -> Root:
    System.initialize_python_path()
    feats, labels, _, label_shape = preprocess(data)
    mp.allow_connection_pickling()
    System.set_multiprocessing_env()
    memmap_dir = System.new_dir("memmap_ds")
    SampleReader.materialize(
        {"features": feats, "labels": labels},
        memmap_dir=memmap_dir,
        train_frac=1.0 - float(val_frac),
        val_frac=float(val_frac),
        shuffle=False,
    )
    ckpt_dir = System.new_dir("ckpt_dcp")
    init_dir = System.new_dir("init_dcp")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=ignored_pattern)
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        m_sd = get_model_state_dict(model, options=opts)
        save(
            state_dict={"model": m_sd},
            storage_writer=FileSystemWriter(init_dir, sync_files=True, overwrite=True),
        )
    default_rdzv_host = Network.get_preferred_ip(allow_loopback=True) or "127.0.0.1"
    resolved_rdzv = rdzv_endpoint if rdzv_endpoint else default_rdzv_host
    rdzv_endpoint = Network.get_available_addr(resolved_rdzv)
    master_addr, _master_port = Distributed.initialize_master_addr(rdzv_endpoint)
    System.optimize_threads()
    nprocs = System.optimal_procs()["nproc_per_node"]
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
        start_method=System.optimal_start_method(),
        local_addr=master_addr,
    )
    base = dict(
        memmap_dir=memmap_dir,
        ckpt_dir=ckpt_dir,
        init_ckpt_dir=init_dir,
        in_dim=int(feats.shape[1]),
        out_shape=tuple(label_shape),
        cfg_dict=cfg_dict,
    )
    default_kwargs = {
        "epochs": epochs,
        "batch_size": batch_size,
        "val_frac": val_frac,
        "base_lr": base_lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "eta_min": eta_min,
        "seed": seed,
        "prefetch_factor": prefetch_factor,
        "grad_accum_steps": grad_accum_steps,
        "overlap_h2d": overlap_h2d,
        "loss_tile_dim": loss_tile_dim,
        "loss_tile_size": loss_tile_size,
        "loss_mask_mode": loss_mask_mode,
        "loss_mask_value": loss_mask_value,
    }
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
    elastic_launch(lc, main)(ops)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=ignored_pattern)
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        m_sd = get_model_state_dict(model, options=opts)
        m_sd = _prune_dcp_state_keys(m_sd)
        load(state_dict={"model": m_sd}, storage_reader=FileSystemReader(ckpt_dir))
        set_model_state_dict(model, m_sd, options=StateDictOptions(strict=False))
    shutil.rmtree(memmap_dir, ignore_errors=True)
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    shutil.rmtree(init_dir, ignore_errors=True)
    return model


def predict(
    model: Root,
    data: Dict[Tuple, torch.Tensor],
    *args: Any,
    batch_size: int = 512,
    seed: int = 7,
    prefetch_factor: Optional[int] = 1,
    mode: OpsMode = "predict",
    **kwargs: Any,
) -> Dict[Tuple, torch.Tensor]:
    System.initialize_python_path()
    System.set_multiprocessing_env()
    tmp_dir = System.new_dir("infer")
    dcp_dir = os.path.join(tmp_dir, "dcp")
    memmap_dir = os.path.join(tmp_dir, "memmap")
    device = System.get_device()
    mp.allow_connection_pickling()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=ignored_pattern)
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        m_sd = get_model_state_dict(model, options=opts)
        save(
            state_dict={"model": m_sd},
            storage_writer=FileSystemWriter(dcp_dir, sync_files=True, overwrite=True),
        )
    cfg_obj = getattr(model, "_Root__config", None)
    if isinstance(cfg_obj, (ModelConfig, dict)):
        cfg_model = coerce_model_config(cfg_obj)
    else:
        cfg_model = ModelConfig()
    cfg_dict = asdict(cfg_model)
    # (선택) 추론 전 퍼센타일 경계 캘리브레이션을 하려면, 별도 샘플 로더로
    # _QuantileState를 초기화하고 _update_quantiles를 호출하는 훅을 추가할 수 있다.
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
    feats, labels, keys, label_shape = preprocess(data)
    SampleReader.materialize(
        {"features": feats, "labels": labels},
        memmap_dir=memmap_dir,
        train_frac=1.0,
        val_frac=0.0,
        shuffle=False,
    )
    base = dict(
        model_ckpt_dir=dcp_dir,
        memmap_dir=memmap_dir,
        in_dim=int(feats.shape[1]),
        out_shape=tuple(label_shape),
        cfg_dict=cfg_dict,
        keys=keys,
    )
    mode = mode if mode in ("predict", "infer") else "predict"
    default_kwargs = {
        "batch_size": batch_size,
        "seed": seed,
        "prefetch_factor": prefetch_factor,
    }
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
    nprocs = (
        Distributed.get_world_size(device)
        if device.type in ("cuda", "xpu")
        else 1
    )
    manager = mp.Manager()
    ret_dict = manager.dict()
    mp.start_processes(
        main,
        args=(ops, ret_dict),
        nprocs=nprocs,
        join=True,
        daemon=False,
        start_method=System.optimal_start_method(),
    )
    try:
        return dict(ret_dict)
    finally:
        with contextlib.suppress(Exception):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def epoch(
    *,
    model: Root,
    device: torch.device,
    ops: RuntimeConfig,
    param_dtype: torch.dtype,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    sched: torch.optim.lr_scheduler.LRScheduler,
    loss_controller: LossWeightController,
    epoch_idx: int,
    total_epochs: int,
    top_loss: TiledLoss,
    bottom_loss: TiledLoss,
    status_bar: Optional[tqdm],
    grad_accum_steps: int,
    train_loader: Any,
    val_loader: Any,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Dict[str, float],
]:
    if train_loader is None:
        raise RuntimeError("epoch requires a training dataloader")
    in_dim = int(ops.in_dim)
    cfg_source = getattr(ops, "cfg_dict", None)
    if isinstance(cfg_source, ModelConfig):
        cfg_dict = asdict(cfg_source)
    elif isinstance(cfg_source, dict):
        cfg_dict = dict(cfg_source)
    else:
        cfg_dict = {}
    flop_breakdown_epoch: Dict[str, float] = {}
    std_mode = str(cfg_dict.get("loss_std_mode", "pooled")).lower()
    loss_ddof = int(cfg_dict.get("loss_ddof", 1))
    detach_stats = bool(cfg_dict.get("loss_detach_stats", True))
    clamp_max = float(cfg_dict.get("loss_clamp_max", 6.0))
    loss_t_conf = float(cfg_dict.get("loss_t_confidence", 0.995))
    loss_z_penalty = str(cfg_dict.get("loss_z_penalty", "huber")).lower()
    if loss_z_penalty not in {"softplus", "huber"}:
        loss_z_penalty = "huber"
    loss_z_tau = float(cfg_dict.get("loss_z_tau", 1.5))
    df_start = float(cfg_dict.get("loss_t_df_start", 3.0))
    df_end = float(cfg_dict.get("loss_t_df_end", 6.0))
    total_epochs = max(1, int(total_epochs))
    epoch_progress = (int(epoch_idx) + 1) / total_epochs
    df_now = df_start + (df_end - df_start) * epoch_progress
    df_now = float(max(df_now, 1e-06))
    top_loss.base = StudentsTLoss(
        confidence=loss_t_conf,
        metric="t_value",
        two_tailed=True,
        df=df_now,
        mu_mode="error",
        std_mode=std_mode,
        ddof=loss_ddof,
        clamp_max=clamp_max,
        detach_stats=detach_stats,
        dim=-1,
        reduction="none",
    )
    bottom_loss.base = StandardNormalLoss(
        confidence=0.99,
        metric="z_value",
        two_tailed=True,
        penalty=loss_z_penalty,
        tau=loss_z_tau,
        mu_mode="error",
        std_mode=std_mode,
        ddof=loss_ddof,
        clamp_max=clamp_max,
        detach_stats=detach_stats,
        dim=-1,
        reduction="none",
    )
    io_time = torch.tensor(0.0, device=device, dtype=torch.float64)
    comp_time = torch.tensor(0.0, device=device, dtype=torch.float64)
    io_bytes = torch.tensor(0.0, device=device, dtype=torch.float64)
    flops = torch.tensor(0.0, device=device, dtype=torch.float64)

    flop_counter_train = FlopCounter(model, mode="train", device=device)
    use_timer = getattr(device, "type", "cpu") in ("cuda", "xpu", "mps") and hasattr(
        torch, "Event"
    )
    with flop_counter_train:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        t_fetch_start = time.perf_counter_ns()
        with joining(model=model, optimizer=optimizer):
            total_batches = len(train_loader)
            alpha = float(cfg_dict.get("w_global", 1.0))
            beta0 = float(cfg_dict.get("w_local", 0.2))
            decay_local = bool(cfg_dict.get("local_decay", True))
            beta = (
                beta0 * (1.0 - 0.7 * epoch_progress)
                if decay_local
                else beta0
            )
            p_lo = float(cfg_dict.get("aux_region_lo_pct", 0.2))
            p_hi = float(cfg_dict.get("aux_region_hi_pct", 0.8))
            q_mode = str(cfg_dict.get("aux_quantile_smooth", "ema")).lower()
            q_beta = float(cfg_dict.get("aux_quantile_ema_beta", 0.9))
            q_win = int(cfg_dict.get("aux_quantile_win_size", 64))
            q_warm = int(cfg_dict.get("aux_quantile_warmup_steps", 16))
            q_sync = str(cfg_dict.get("aux_quantile_sync", "mean")).lower()
            qstate = _QuantileState(q_mode, q_beta, q_win, q_warm, q_sync)
            quantile_logged = False
            for step_idx, _raw in enumerate(train_loader):
                feat, label, *_ = preprocess(_raw)
                X = to_torch(feat)
                X = torch.atleast_2d(X)
                if X.dim() != 2:
                    raise RuntimeError(
                        f"features.ndim={X.dim()} (expect 2). got shape={tuple(X.shape)}"
                    )
                if X.shape[1] != in_dim:
                    raise RuntimeError(
                        f"feature dim mismatch: X.shape[1]={X.shape[1]} != in_dim={in_dim}"
                    )
                Y = to_torch(label)
                t_ready = time.perf_counter_ns()
                if use_timer:
                    h2d_s_ev, h2d_e_ev = (
                        torch.Event(device=device, enable_timing=True),
                        torch.Event(device=device, enable_timing=True),
                    )
                    h2d_s_ev.record()
                    X = X.to(device, non_blocking=True)
                    Y = Y.to(device, non_blocking=True)
                    h2d_e_ev.record()
                    h2d_e_ev.synchronize()
                    h2d_s = float(h2d_s_ev.elapsed_time(h2d_e_ev)) / 1000.0
                else:
                    t_h2d_s = time.perf_counter_ns()
                    X = X.to(device, non_blocking=True)
                    Y = Y.to(device, non_blocking=True)
                    t_h2d_e = time.perf_counter_ns()
                    h2d_s = (t_h2d_e - t_h2d_s) / 1_000_000_000.0
                wait_s = (t_ready - t_fetch_start) / 1_000_000_000.0
                io_time += torch.tensor(
                    wait_s + h2d_s, device=device, dtype=torch.float64
                )
                with contextlib.suppress(Exception):
                    io_bytes += torch.tensor(
                        X.element_size() * X.nelement()
                        + Y.element_size() * Y.nelement(),
                        device=device,
                        dtype=torch.float64,
                    )
                should_sync = ((step_idx + 1) % max(1, grad_accum_steps) == 0) or (
                    step_idx + 1 == total_batches
                )
                if use_timer:
                    ev_s, ev_e = (
                        torch.Event(device=device, enable_timing=True),
                        torch.Event(device=device, enable_timing=True),
                    )
                    ev_s.record()
                else:
                    t_comp_s = time.perf_counter_ns()
                with no_synchronization(
                    model, enable=(grad_accum_steps > 1 and (not should_sync))
                ):
                    with flop_counter_train.step(display=False) as train_counter:
                        with AutoCast.float(device):
                            Y_flat = Y.reshape(Y.shape[0], -1).to(
                                device, dtype=param_dtype
                            )
                            if step_idx == 0:
                                try:
                                    _float8_log(
                                        f"[loss] epoch={epoch_idx + 1}/{total_epochs} "
                                        f"df_now={df_now:.3f} std_mode={std_mode} "
                                        f"z_penalty={loss_z_penalty}"
                                    )
                                except Exception:
                                    pass
                            y_hat, loss_val = model(
                                X,
                                labels_flat=Y_flat,
                                global_loss=top_loss,
                                local_loss=bottom_loss,
                                loss_weights=(alpha, beta),
                            )
                            if step_idx == 0:
                                try:
                                    _float8_log(
                                        f"[train] space={getattr(cfg, 'loss_space', 'z')} "
                                        f"calib={getattr(cfg, 'calibrate_output', True)} "
                                        f"z_reg={getattr(cfg, 'z_reg_lambda', 1e-2)}"
                                    )
                                except Exception:
                                    pass
                            quantile_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

                            def _ensure_quantiles(
                                target_tensor: torch.Tensor,
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
                                nonlocal quantile_cache
                                if quantile_cache is None:
                                    q_lo_raw, q_hi_raw = _update_quantiles(
                                        qstate, target_tensor, p_lo, p_hi
                                    )
                                    quantile_cache = (
                                        q_lo_raw.to(dtype=target_tensor.dtype),
                                        q_hi_raw.to(dtype=target_tensor.dtype),
                                    )
                                return quantile_cache

                            if (
                                loss_val is not None
                                and bool(cfg_dict.get("aux_q_enable", True))
                            ):
                                w_aux_override = float(cfg_dict.get("w_aux", 0.0))
                                if w_aux_override > 0.0:
                                    w_aux = w_aux_override
                                else:
                                    w_aux = float(cfg_dict.get("aux_q_weight", 0.05))
                                if w_aux > 0.0:
                                    pred = y_hat.reshape_as(Y_flat)
                                    target = Y_flat.detach()
                                    err = target - pred
                                    tau_low = float(
                                        cfg_dict.get(
                                            "aux_tau_lo_region",
                                            cfg_dict.get("aux_q_tau_lowspd", 0.7),
                                        )
                                    )
                                    tau_high = float(
                                        cfg_dict.get(
                                            "aux_tau_hi_region",
                                            cfg_dict.get("aux_q_tau_highspd", 0.45),
                                        )
                                    )
                                    tau_mid = pred.new_full(pred.shape, 0.5)
                                    tau_low_tensor = pred.new_full(pred.shape, tau_low)
                                    tau_high_tensor = pred.new_full(pred.shape, tau_high)
                                    q_lo, q_hi = _ensure_quantiles(target)
                                    tau_tensor = torch.where(
                                        target <= q_lo,
                                        tau_low_tensor,
                                        tau_mid,
                                    )
                                    tau_tensor = torch.where(
                                        target >= q_hi,
                                        tau_high_tensor,
                                        tau_tensor,
                                    )
                                    q_pin = torch.maximum(
                                        tau_tensor * err,
                                        (tau_tensor - 1.0) * err,
                                    ).mean()
                                    damp = 1.0 - 0.5 * epoch_progress
                                    if step_idx == 0 and not quantile_logged:
                                        try:
                                            _float8_log(
                                                f"[loss] epoch={epoch_idx + 1}/{total_epochs} "
                                                f"quantile_smooth={q_mode} beta={q_beta:.2f} "
                                                f"win={q_win} warm={q_warm} sync={q_sync} "
                                                f"pct(lo={p_lo:.2f},hi={p_hi:.2f}) "
                                                f"q_lo={float(q_lo):.3f} q_hi={float(q_hi):.3f}"
                                            )
                                            quantile_logged = True
                                        except Exception:
                                            pass
                                    loss_val = loss_val + pred.new_tensor(w_aux * damp) * q_pin
                            if loss_val is not None and isinstance(loss_val, torch.Tensor):
                                w_low = float(
                                    cfg_dict.get(
                                        "aux_weight_lo_region",
                                        cfg_dict.get("loss_w_low", 1.0),
                                    )
                                )
                                w_high = float(
                                    cfg_dict.get(
                                        "aux_weight_hi_region",
                                        cfg_dict.get("loss_w_high", 1.0),
                                    )
                                )
                                if (w_low != 1.0) or (w_high != 1.0):
                                    with torch.no_grad():
                                        weights = torch.ones_like(Y_flat)
                                        q_lo, q_hi = _ensure_quantiles(Y_flat.detach())
                                        if step_idx == 0 and not quantile_logged:
                                            try:
                                                _float8_log(
                                                    f"[loss] epoch={epoch_idx + 1}/{total_epochs} "
                                                    f"quantile_smooth={q_mode} beta={q_beta:.2f} "
                                                    f"win={q_win} warm={q_warm} sync={q_sync} "
                                                    f"pct(lo={p_lo:.2f},hi={p_hi:.2f}) "
                                                    f"q_lo={float(q_lo):.3f} q_hi={float(q_hi):.3f}"
                                                )
                                                quantile_logged = True
                                            except Exception:
                                                pass
                                        weights = torch.where(
                                            Y_flat <= q_lo,
                                            weights * w_low,
                                            weights,
                                        )
                                        weights = torch.where(
                                            Y_flat >= q_hi,
                                            weights * w_high,
                                            weights,
                                        )
                                    weights = weights.to(
                                        device=loss_val.device, dtype=loss_val.dtype
                                    )
                                    if loss_val.ndim == 0:
                                        loss_val = loss_val * weights.mean()
                                    else:
                                        weight_view = weights
                                        if weights.shape != loss_val.shape:
                                            try:
                                                weight_view = weights.view_as(loss_val)
                                            except Exception:
                                                weight_view = weights.expand_as(loss_val)
                                        loss_val = (loss_val * weight_view).mean()
                        accum_scale = max(1, grad_accum_steps)
                        loss_for_backprop = loss_val / float(accum_scale)
                        scaler.scale(loss_for_backprop).backward()
                        if should_sync:
                            scaler.unscale_(optimizer)
                            clip_max_norm = 1.0
                            clip_fn = getattr(model, "clip_grad_norm_", None)
                            if callable(clip_fn):
                                try:
                                    clip_fn(clip_max_norm)
                                except (TypeError, RuntimeError):
                                    _safe_clip_grad_norm(
                                        list(model.parameters()),
                                        max_norm=clip_max_norm,
                                    )
                            else:
                                _safe_clip_grad_norm(
                                    list(model.parameters()),
                                    max_norm=clip_max_norm,
                                )
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                            sched.step()
                        with contextlib.suppress(Exception):
                            step_flops = float(train_counter.get_total_flops())
                        flops += torch.tensor(
                            max(0.0, step_flops), device=device, dtype=torch.float64
                        )
                        breakdown_getter = getattr(
                            train_counter, "get_manual_breakdown", None
                        )
                        if callable(breakdown_getter):
                            for name, value in breakdown_getter().items():
                                try:
                                    flop_breakdown_epoch[name] = flop_breakdown_epoch.get(
                                        name, 0.0
                                    ) + float(value)
                                except Exception:
                                    continue
                if use_timer:
                    ev_e.record()
                    ev_e.synchronize()
                    comp_time += torch.tensor(
                        float(ev_s.elapsed_time(ev_e)) / 1000.0,
                        device=device,
                        dtype=torch.float64,
                    )
                else:
                    comp_time += torch.tensor(
                        (time.perf_counter_ns() - t_comp_s) / 1_000_000_000.0,
                        device=device,
                        dtype=torch.float64,
                    )
                with contextlib.suppress(Exception):
                    mark_step = getattr(
                        getattr(torch, "compiler", None),
                        "cudagraph_mark_step_end",
                        None,
                    )
                    if callable(mark_step):
                        mark_step()
                if status_bar is not None:
                    io_elapsed = float(io_time.item())
                    io_transferred = float(io_bytes.item())
                    comp_elapsed = float(comp_time.item())
                    flop_total = float(flops.item())
                    mbps_cur = io_transferred / max(io_elapsed, 1e-06) / 1_000_000.0
                    tflops_cur = (
                        flop_total / max(comp_elapsed, 1e-06) / 1_000_000_000_000.0
                    )
                    _advance_status_bar(
                        status_bar,
                        1,
                        mbps_cur,
                        tflops_cur,
                        comp_elapsed=comp_elapsed,
                        flop_breakdown=flop_breakdown_epoch,
                    )
                t_fetch_start = time.perf_counter_ns()

    if val_loader is not None:
        flop_counter_val = FlopCounter(model, mode="eval", device=device)
        with flop_counter_val:
            model.eval()
            with inference(model), AutoCast.float(device):
                t_fetch_start = time.perf_counter_ns()
                with joining(model=model, optimizer=optimizer):
                    for step_idx, _raw in enumerate(val_loader):
                        feat, label, *_ = preprocess(_raw)
                        X = to_torch(feat)
                        X = torch.atleast_2d(X)
                        if X.dim() != 2:
                            raise RuntimeError(
                                f"features.ndim={X.dim()} (expect 2). got shape={tuple(X.shape)}"
                            )
                        if X.shape[1] != in_dim:
                            raise RuntimeError(
                                f"feature dim mismatch: X.shape[1]={X.shape[1]} != in_dim={in_dim}"
                            )
                        Y = to_torch(label)
                        t_ready = time.perf_counter_ns()
                        if use_timer:
                            h2d_s_ev, h2d_e_ev = (
                                torch.Event(device=device, enable_timing=True),
                                torch.Event(device=device, enable_timing=True),
                            )
                            h2d_s_ev.record()
                            X = X.to(device, non_blocking=True)
                            Y = Y.to(device, non_blocking=True)
                            h2d_e_ev.record()
                            h2d_e_ev.synchronize()
                            h2d_s = float(h2d_s_ev.elapsed_time(h2d_e_ev)) / 1000.0
                        else:
                            t_h2d_s = time.perf_counter_ns()
                            X = X.to(device, non_blocking=True)
                            Y = Y.to(device, non_blocking=True)
                            t_h2d_e = time.perf_counter_ns()
                            h2d_s = (t_h2d_e - t_h2d_s) / 1_000_000_000.0
                        wait_s = (t_ready - t_fetch_start) / 1_000_000_000.0
                        io_time += torch.tensor(
                            wait_s + h2d_s, device=device, dtype=torch.float64
                        )
                        with contextlib.suppress(Exception):
                            io_bytes += torch.tensor(
                                X.element_size() * X.nelement()
                                + Y.element_size() * Y.nelement(),
                                device=device,
                                dtype=torch.float64,
                            )
                        if use_timer:
                            ev_s, ev_e = (
                                torch.Event(device=device, enable_timing=True),
                                torch.Event(device=device, enable_timing=True),
                            )
                            ev_s.record()
                        else:
                            t_comp_s = time.perf_counter_ns()
                        with flop_counter_val.step(display=False) as val_counter:
                            Yv_flat = Y.reshape(Y.shape[0], -1).to(
                                device, dtype=param_dtype
                            )
                            _y, _loss_val = model(
                                X,
                                labels_flat=Yv_flat,
                                global_loss=top_loss,
                                local_loss=bottom_loss,
                                loss_weights=(alpha, beta),
                            )
                        if use_timer:
                            ev_e.record()
                            ev_e.synchronize()
                            comp_time += torch.tensor(
                                float(ev_s.elapsed_time(ev_e)) / 1000.0,
                                device=device,
                                dtype=torch.float64,
                            )
                        else:
                            comp_time += torch.tensor(
                                (time.perf_counter_ns() - t_comp_s) / 1_000_000_000.0,
                                device=device,
                                dtype=torch.float64,
                            )
                        with contextlib.suppress(Exception):
                            v_step_flops = float(val_counter.get_total_flops())
                        flops += torch.tensor(
                            max(0.0, v_step_flops), device=device, dtype=torch.float64
                        )
                        breakdown_getter = getattr(
                            val_counter, "get_manual_breakdown", None
                        )
                        if callable(breakdown_getter):
                            for name, value in breakdown_getter().items():
                                try:
                                    flop_breakdown_epoch[name] = flop_breakdown_epoch.get(
                                        name, 0.0
                                    ) + float(value)
                                except Exception:
                                    continue
                        if status_bar is not None:
                            io_elapsed = float(io_time.item())
                            io_transferred = float(io_bytes.item())
                            comp_elapsed = float(comp_time.item())
                            flop_total = float(flops.item())
                            mbps_cur = (
                                io_transferred / max(io_elapsed, 1e-06) / 1_000_000.0
                            )
                            tflops_cur = (
                                flop_total / max(comp_elapsed, 1e-06)
                                / 1_000_000_000_000.0
                            )
                            _advance_status_bar(
                                status_bar,
                                1,
                                mbps_cur,
                                tflops_cur,
                                comp_elapsed=comp_elapsed,
                                flop_breakdown=flop_breakdown_epoch,
                            )
                        t_fetch_start = time.perf_counter_ns()
    return (
        io_time,
        comp_time,
        io_bytes,
        flops,
        flop_breakdown_epoch,
    )


def main(*args: Any) -> Optional[Root]:
    if not args:
        raise TypeError("main requires at least a RuntimeConfig argument")

    System.initialize_python_path()

    ret_sink: Optional[Dict[Any, Any]] = None
    if len(args) == 1 and isinstance(args[0], RuntimeConfig):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        ops = args[0]
    elif len(args) >= 2 and isinstance(args[1], RuntimeConfig):
        local_rank = int(args[0])
        ops = args[1]
        if len(args) >= 3:
            ret_sink = args[2]
    else:
        raise TypeError(
            "main expects (RuntimeConfig,), (local_rank, RuntimeConfig), or "
            "(local_rank, RuntimeConfig, ret_sink) arguments"
        )

    if ops.mode == "train":
        with contextlib.suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank % max(1, torch.cuda.device_count()))
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.set_device(local_rank % max(1, torch.xpu.device_count()))

        device = System.get_device()
        _set_backend(device)
        backend = _backend_type(device)
        init_kwargs: Dict[str, Any] = {"backend": backend}
        torch.distributed.init_process_group(**init_kwargs)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        cfg = coerce_model_config(
            ops.cfg_dict if isinstance(ops.cfg_dict, dict) else ops.cfg_dict
        )
        cfg = replace(cfg, device=device)
        model = Root(ops.in_dim, ops.out_shape, config=cfg)
        if ops.init_ckpt_dir is not None and os.path.isdir(ops.init_ckpt_dir):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=ignored_pattern)
                opts_sd = StateDictOptions(full_state_dict=True, cpu_offload=False)
                m_sd = get_model_state_dict(model, options=opts_sd)
                m_sd = _prune_dcp_state_keys(m_sd)
                load(
                    state_dict={"model": m_sd},
                    storage_reader=FileSystemReader(ops.init_ckpt_dir),
                )
                set_model_state_dict(
                    model, m_sd, options=StateDictOptions(strict=False)
                )
        meta_info = _meta(ops.memmap_dir or "")
        meta_feature_dim = int(meta_info.get("feature_dim", ops.in_dim))
        if meta_feature_dim != int(ops.in_dim):
            raise RuntimeError(
                "dataset feature_dim mismatch: "
                f"meta={meta_feature_dim}, expected in_dim={ops.in_dim}"
            )
        meta_label_shape = tuple(
            int(x) for x in meta_info.get("label_shape", list(ops.out_shape))
        )
        if tuple(meta_label_shape) != tuple(ops.out_shape):
            raise RuntimeError(
                "dataset label_shape mismatch: "
                f"meta={meta_label_shape}, expected out_shape={tuple(ops.out_shape)}"
            )
        fractions = meta_info.get("fractions", [1.0, 0.0])
        if isinstance(fractions, (list, tuple)) and len(fractions) >= 2:
            actual_val_frac = float(fractions[-1])
            if not math.isclose(
                actual_val_frac,
                float(ops.val_frac),
                rel_tol=0.001,
                abs_tol=0.001,
            ):
                warnings.warn(
                    "val_frac=%s differs from memmap metadata (%s); "
                    "using metadata value for loaders"
                    % (ops.val_frac, actual_val_frac)
                )
                ops = replace(ops, val_frac=actual_val_frac)
        data_scale: Optional[DataScale] = None
        train_loader0, val_loader0, keep0 = dataloader(
            memmap_dir=ops.memmap_dir,
            device=device,
            batch_size=int(ops.batch_size or 128),
            val_frac=float(ops.val_frac),
            prefetch_factor=ops.prefetch_factor,
            non_blocking_copy=bool(ops.overlap_h2d),
            io_backend="auto",
        )
        if (
            getattr(cfg, "auto_y_range", False)
            and str(getattr(cfg, "loss_space", "")).lower() == "logit"
            and hasattr(model, "set_y_range")
            and train_loader0 is not None
        ):
            q_low = float(getattr(cfg, "y_range_q_low", 0.001))
            q_high = float(getattr(cfg, "y_range_q_high", 0.999))
            eps = float(getattr(cfg, "y_eps_range", 1e-3))
            try:
                lo, hi = compute_y_range(train_loader0, q_low=q_low, q_high=q_high)
            except Exception as exc:  # pragma: no cover - defensive path
                warnings.warn(f"auto_y_range failed: {exc}")
            else:
                span = float(hi - lo)
                m_low = float(getattr(cfg, "y_range_margin_low", 0.10)) * span
                m_high = float(getattr(cfg, "y_range_margin_high", 0.05)) * span
                base_low = float(getattr(cfg, "y_low", 0.0))
                base_high = float(getattr(cfg, "y_high", 100.0))
                if (
                    not math.isfinite(base_low)
                    or not math.isfinite(base_high)
                    or base_high <= base_low
                ):
                    base_low, base_high = 0.0, 100.0
                A = max(base_low, float(lo) - m_low)
                B = min(base_high, float(hi) + m_high)
                if B <= A:
                    center = 0.5 * (float(lo) + float(hi))
                    delta = max(1e-3, span * 0.5 or 1.0)
                    A = center - delta
                    B = center + delta
                A = min(base_high, max(base_low, A))
                B = min(base_high, max(base_low, B))
                domain_span = max(1e-3, base_high - base_low)
                target_width = min(domain_span, max(1e-3, eps))
                if B - A < target_width:
                    mid = min(base_high, max(base_low, 0.5 * (A + B)))
                    half = 0.5 * target_width
                    A = mid - half
                    B = mid + half
                    if A < base_low:
                        shift = base_low - A
                        A = base_low
                        B = min(base_high, B + shift)
                    if B > base_high:
                        shift = B - base_high
                        B = base_high
                        A = max(base_low, A - shift)
                    if B - A < target_width:
                        if A <= base_low:
                            B = min(base_high, A + target_width)
                        elif B >= base_high:
                            A = max(base_low, B - target_width)
                        else:
                            B = min(base_high, A + target_width)
                    A = min(base_high, max(base_low, A))
                    B = min(base_high, max(base_low, B))
                try:
                    model.set_y_range(A, B, eps)
                    try:
                        _float8_log(
                            f"[y-range] domain=[{base_low:.3f},{base_high:.3f}] "
                            f"A={A:.3f}, B={B:.3f}, eps_abs={eps:.4g}, "
                            f"eps_rel={getattr(cfg, 'y_eps_rel', 0.0)}"
                        )
                    except Exception:
                        pass
                except Exception as exc:  # pragma: no cover - defensive path
                    warnings.warn(f"set_y_range failed: {exc}")
        train_step_count = 0
        for train_step_count, _raw in enumerate(train_loader0, start=1):
            _feat0, _label0, *_ = preprocess(_raw)
            data_scale = _accumulate_data_scale(data_scale, _feat0, _label0)
            if hasattr(model, "update_x_stats"):
                with contextlib.suppress(Exception):
                    model.update_x_stats(_feat0)
            _label0 = to_torch(_label0)
            _Y0_flat = _label0.view(_label0.shape[0], -1)
            model.update_y_stats(_Y0_flat)
        model.finalize_y_stats()
        if hasattr(model, "finalize_x_stats"):
            model.finalize_x_stats()
        if keep0 is not None:
            keep0.cleanup()
        if data_scale is not None:
            setattr(model, "__data_scale_hint__", data_scale)
        model, _, _ = Module.use_te_module(
            model, device=device, scale=data_scale
        )
        param_dtype = _ensure_uniform_param_dtype(
            model,
            prefer=(
                torch.bfloat16
                if getattr(device, "type", None) == "cuda"
                and torch.cuda.is_bf16_supported()
                else None
            ),
        )
        if param_dtype is None:
            param_dtype = torch.float32
        fp8_ok, fp8_reason = System.is_float8_supported(device)
        fp8_enabled = False
        fp8_backend: Optional[str] = None
        disable_note: Optional[str] = None
        if fp8_ok:
            model, fp8_enabled, fp8_backend = Module.enable_float8_training(
                model,
                device=device,
                prefer="te",
                logger=_float8_log,
                scale=data_scale,
            )
            if not fp8_enabled:
                disable_note = fp8_backend
        else:
            disable_note = fp8_reason
        if not fp8_enabled:
            AutoCast.configure(model, scale=data_scale)
            if disable_note:
                _float8_log(f"[FP8] disabled: {disable_note}")
        model.train()
        world = Distributed.get_world_size(device)
        mesh = init_device_mesh(
            "cuda" if device.type == "cuda" else device.type, (world,)
        )
        mp_policy = MixedPrecisionPolicy(
            param_dtype=None,
            reduce_dtype=torch.float64,
            output_dtype=None,
            cast_forward_inputs=False,
        )
        ignored_params: List[torch.nn.Parameter] = []
        for module in model.modules():
            if isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
                for p in module.parameters(recurse=False):
                    ignored_params.append(p)
            for name in ("alpha_t", "alpha_s", "gem_p", "cls_query", "cls"):
                if hasattr(module, name):
                    p = getattr(module, name)
                    if isinstance(p, torch.nn.Parameter):
                        ignored_params.append(p)

        class _IdentityParamSet(Sequence[torch.nn.Parameter]):
            def __init__(self, params: Sequence[torch.nn.Parameter]) -> None:
                self._params = tuple(params)
                self._ids = {id(p) for p in self._params}

            def __len__(self) -> int:
                return len(self._params)

            def __iter__(self) -> Iterator[torch.nn.Parameter]:
                return iter(self._params)

            def __getitem__(self, index: int) -> torch.nn.Parameter:
                return self._params[index]

            def __contains__(self, item: object) -> bool:
                return isinstance(item, torch.nn.Parameter) and (
                    id(item) in self._ids
                )

        ignored_param_registry = _IdentityParamSet(tuple(ignored_params))

        def _per_module_ignored_params(
            module: torch.nn.Module,
        ) -> Optional[_IdentityParamSet]:
            if len(ignored_param_registry) == 0:
                return None
            params = [
                param
                for param in module.parameters(recurse=True)
                if param in ignored_param_registry
            ]
            return _IdentityParamSet(tuple(params)) if params else None

        wrapped: set[int] = set()

        def _fsdp_wrap(
            target: Optional[torch.nn.Module],
        ) -> Optional[torch.nn.Module]:
            nonlocal model
            if target is None or id(target) in wrapped:
                return target
            wrapped.add(id(target))
            per_mod_ignored = _per_module_ignored_params(target)
            sharded = fully_shard(
                target,
                mesh=mesh,
                mp_policy=mp_policy,
                reshard_after_forward=False,
                ignored_params=per_mod_ignored or None,
            )
            sharded.set_requires_gradient_sync(True)
            if target is model:
                model = sharded
            return sharded

        def _collect_block_modules(
            root: Optional[torch.nn.Module],
        ) -> List[torch.nn.Module]:
            if root is None:
                return []
            blocks: List[torch.nn.Module] = []
            seen: set[int] = set()
            for module in root.modules():
                block_list = getattr(module, "blocks", None)
                if isinstance(block_list, torch.nn.ModuleList):
                    for block in block_list:
                        if isinstance(block, torch.nn.Module) and id(block) not in seen:
                            seen.add(id(block))
                            blocks.append(block)
            return blocks

        try:
            for submodule in _collect_block_modules(
                getattr(model, "local_net", None)
            ) + _collect_block_modules(getattr(model, "global_net", None)):
                _fsdp_wrap(submodule)
            _fsdp_wrap(model)
        except (RuntimeError, ValueError, TypeError):
            model = fully_shard(
                model,
                mesh=mesh,
                mp_policy=mp_policy,
                ignored_params=(
                    ignored_param_registry if len(ignored_param_registry) > 0 else None
                ),
                reshard_after_forward=False,
            )
            model.set_requires_gradient_sync(True)
        if fp8_enabled and _float8_scale_supported(data_scale):
            with contextlib.suppress(Exception):
                precompute_float8_dynamic_scale_for_fsdp(
                    model, data_scale=data_scale
                )
        net_params = [p for p in model.parameters()]
        optimizer = AdamW.float(
            net_params,
            lr=ops.base_lr,
            weight_decay=ops.weight_decay,
            use_fp8=(device.type == "cuda"),
            use_foreach=False,
            use_fused=False,
            scale=data_scale,
            logger=None,
        )
        if ops.init_ckpt_dir is not None and os.path.isdir(ops.init_ckpt_dir):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=ignored_pattern)
                optim_sd = get_optimizer_state_dict(model, optimizers=optimizer)
                try:
                    load(
                        state_dict={"optimizer": optim_sd},
                        storage_reader=FileSystemReader(ops.init_ckpt_dir),
                    )
                except (
                    FileNotFoundError,
                    ValueError,
                    KeyError,
                    RuntimeError,
                    CheckpointException,
                ) as exc:
                    if "optimizer" not in str(exc).lower():
                        raise
                else:
                    set_optimizer_state_dict(
                        model,
                        optimizer,
                        optim_sd,
                        options=StateDictOptions(strict=False),
                    )
        std_mode = str(getattr(cfg, "loss_std_mode", "pooled")).lower()
        loss_ddof = int(getattr(cfg, "loss_ddof", 1))
        detach_stats = bool(getattr(cfg, "loss_detach_stats", True))
        clamp_max = float(getattr(cfg, "loss_clamp_max", 6.0))
        loss_t_conf = float(getattr(cfg, "loss_t_confidence", 0.995))
        loss_z_penalty = str(getattr(cfg, "loss_z_penalty", "softplus")).lower()
        loss_z_tau = float(getattr(cfg, "loss_z_tau", 1.5))
        df_start = float(getattr(cfg, "loss_t_df_start", 3.0))
        df_end = float(getattr(cfg, "loss_t_df_end", 6.0))

        def _build_students_t(df: float) -> StudentsTLoss:
            return StudentsTLoss(
                confidence=loss_t_conf,
                metric="t_value",
                two_tailed=True,
                df=float(df),
                mu_mode="error",
                std_mode=std_mode,
                ddof=loss_ddof,
                clamp_max=clamp_max,
                detach_stats=detach_stats,
                dim=-1,
                reduction="none",
            )

        _t = _build_students_t(df_start)
        _z = StandardNormalLoss(
            confidence=0.99,
            metric="z_value",
            two_tailed=True,
            penalty=loss_z_penalty,
            tau=loss_z_tau,
            mu_mode="error",
            std_mode=std_mode,
            ddof=loss_ddof,
            clamp_max=clamp_max,
            detach_stats=detach_stats,
            dim=-1,
            reduction="none",
        )
        top_loss = TiledLoss(
            _t,
            mask_mode=ops.loss_mask_mode,
            mask_value=ops.loss_mask_value,
            tile_dim=ops.loss_tile_dim,
            tile_size=ops.loss_tile_size,
            reduction="mean",
        )
        bottom_loss = TiledLoss(
            _z,
            mask_mode=ops.loss_mask_mode,
            mask_value=ops.loss_mask_value,
            tile_dim=ops.loss_tile_dim,
            tile_size=ops.loss_tile_size,
            reduction="mean",
        )
        loss_controller = LossWeightController()
        ckpt_state_path = dl_state_path(ops.ckpt_dir or "")
        init_state_path = (
            dl_state_path(ops.init_ckpt_dir) if ops.init_ckpt_dir else None
        )
        state_train: Dict[str, Any] = {}
        state_val: Dict[str, Any] = {}
        _dlp = (
            ckpt_state_path
            if os.path.isfile(ckpt_state_path)
            else (
                init_state_path
                if init_state_path and os.path.isfile(init_state_path)
                else None
            )
        )
        restore_dl_state = False
        if _dlp:
            with contextlib.suppress(Exception):
                _dl_json = json.load(open(_dlp, "r", encoding="utf-8"))
                if isinstance(_dl_json, dict):
                    state_train = _dl_json.get("train", {}) or {}
                    state_val = _dl_json.get("val", {}) or {}
                    restore_dl_state = bool(state_train) or bool(state_val)
        train_loader: Any = None
        val_loader: Any = None
        keep: Any = None
        status_bar: Optional[tqdm] = None
        try:
            train_loader, val_loader, keep = dataloader(
                memmap_dir=ops.memmap_dir,
                device=device,
                batch_size=int(ops.batch_size or 128),
                val_frac=float(ops.val_frac),
                prefetch_factor=ops.prefetch_factor,
                non_blocking_copy=bool(ops.overlap_h2d),
                io_backend="auto",
            )
            if restore_dl_state:
                with contextlib.suppress(Exception):
                    train_loader.load_state_dict(state_train)
                if val_loader is not None:
                    with contextlib.suppress(Exception):
                        val_loader.load_state_dict(state_val)
                restore_dl_state = False

            train_steps = _loader_length(train_loader)
            if train_step_count > 0:
                train_steps = max(train_steps, train_step_count)
            val_steps = _loader_length(val_loader)
            steps_per_epoch = max(1, train_steps + val_steps)
            total_steps = max(1, int(ops.epochs) * steps_per_epoch)
            if ops.warmup_ratio > 0.0:
                warmup_steps = max(1, int(total_steps * ops.warmup_ratio))
                main_steps = max(1, total_steps - warmup_steps)
            else:
                warmup_steps = 0
                main_steps = max(1, total_steps)
            base = float(ops.base_lr)
            emin = float(ops.eta_min)
            start_factor = 0.001

            def _scheduler(step: int) -> float:
                if warmup_steps > 0 and step < warmup_steps:
                    return start_factor + (1.0 - start_factor) * (
                        step / max(1, warmup_steps)
                    )
                t = step - warmup_steps
                frac_min = emin / base if base > 0.0 else 0.0
                return frac_min + (1.0 - frac_min) * 0.5 * (
                    1.0 + math.cos(math.pi * t / max(1, main_steps))
                )

            sched = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=_scheduler
            )
            scaler = torch.amp.GradScaler(
                enabled=(
                    device.type == "cuda" and (not torch.cuda.is_bf16_supported())
                )
            )
            status_bar = (
                _status_bar("Training", total_steps, device)
                if local_rank == 0
                else None
            )

            for epoch_idx in range(int(ops.epochs)):
                (
                    io_time,
                    comp_time,
                    io_bytes,
                    flops,
                    flop_breakdown_epoch,
                ) = epoch(
                    model=model,
                    device=device,
                    ops=ops,
                    param_dtype=param_dtype,
                    optimizer=optimizer,
                    scaler=scaler,
                    sched=sched,
                    loss_controller=loss_controller,
                    top_loss=top_loss,
                    bottom_loss=bottom_loss,
                    status_bar=status_bar,
                    grad_accum_steps=int(ops.grad_accum_steps),
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epoch_idx=epoch_idx,
                    total_epochs=int(ops.epochs),
                )
                torch.distributed.barrier(
                    device_ids=[local_rank]
                    if device.type in ("cuda", "xpu")
                    else None
                )
                for t in (comp_time, io_time, flops, io_bytes):
                    torch.distributed.all_reduce(
                        t, op=torch.distributed.ReduceOp.SUM
                    )
                world = max(1, Distributed.get_world_size(device))
                comp_time /= world
                io_time /= world
                flops /= world
                io_bytes /= world
                if torch.distributed.is_initialized():
                    gathered: List[Dict[str, float]] = [dict() for _ in range(world)]
                    torch.distributed.all_gather_object(
                        gathered, flop_breakdown_epoch
                    )
                    merged: Dict[str, float] = {}
                    for entry in gathered:
                        if not isinstance(entry, dict):
                            continue
                        for key, value in entry.items():
                            try:
                                merged[key] = merged.get(key, 0.0) + float(value)
                            except Exception:
                                continue
                    aggregated_breakdown = merged
                else:
                    aggregated_breakdown = dict(flop_breakdown_epoch)
                if world > 0:
                    aggregated_breakdown = {
                        key: value / world
                        for key, value in aggregated_breakdown.items()
                    }
                if local_rank == 0 and status_bar is not None:
                    mbps = float(io_bytes / io_time.clamp_min(1e-06) / 1_000_000.0)
                    tflops = float(
                        flops / comp_time.clamp_min(1e-06) / 1_000_000_000_000.0
                    )
                    comp_elapsed_mean = float(comp_time.item())
                    postfix = _format_metrics_postfix(
                        mbps,
                        tflops,
                        comp_elapsed=comp_elapsed_mean,
                        flop_breakdown=aggregated_breakdown,
                    )
                    status_bar.set_postfix_str(postfix, refresh=False)
                torch.distributed.barrier(
                    device_ids=[local_rank]
                    if device.type in ("cuda", "xpu")
                    else None
                )
        finally:
            if keep is not None:
                keep.cleanup()
        if local_rank == 0:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=ignored_pattern)
                opts_sd = StateDictOptions(full_state_dict=True, cpu_offload=True)
                model_sd = get_model_state_dict(model, options=opts_sd)
                optim_sd = get_optimizer_state_dict(model, optimizers=optimizer)
                writer = FileSystemWriter(
                    ops.ckpt_dir or "", sync_files=True, overwrite=True
                )
                save(
                    state_dict={"model": model_sd, "optimizer": optim_sd},
                    storage_writer=writer,
                )
            with contextlib.suppress(Exception):
                _dl = {
                    "train": (
                        train_loader.state_dict()
                        if train_loader is not None
                        else {}
                    ),
                    "val": (
                        val_loader.state_dict()
                        if val_loader is not None
                        else {}
                    ),
                }
                with open(
                    dl_state_path(ops.ckpt_dir or ""), "w", encoding="utf-8"
                ) as _f:
                    json.dump(_dl, _f)
        torch.distributed.barrier(
            device_ids=[local_rank]
            if device.type in ("cuda", "xpu")
            else None
        )
        with contextlib.suppress(Exception):
            if local_rank == 0 and status_bar is not None:
                status_bar.close()
        torch.distributed.destroy_process_group()
        return None

    if ops.mode in ("predict", "infer"):
        with contextlib.suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank % max(1, torch.cuda.device_count()))
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.set_device(local_rank % max(1, torch.xpu.device_count()))
        device = System.get_device()
        cfg = coerce_model_config(
            ops.cfg_dict if isinstance(ops.cfg_dict, dict) else ops.cfg_dict
        )
        model = Root(ops.in_dim, ops.out_shape, config=cfg)
        if ops.model_ckpt_dir is not None and os.path.isdir(ops.model_ckpt_dir):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=ignored_pattern)
                opts_sd = StateDictOptions(full_state_dict=True, cpu_offload=True)
                m_sd = get_model_state_dict(model, options=opts_sd)
                m_sd = _prune_dcp_state_keys(m_sd)
                load(
                    state_dict={"model": m_sd},
                    storage_reader=FileSystemReader(ops.model_ckpt_dir),
                )
                set_model_state_dict(
                    model, m_sd, options=StateDictOptions(strict=False)
                )
        model.to(device, non_blocking=True).eval()
        data_scale: Optional[DataScale] = None
        scale_loader, _, scale_keep = dataloader(
            memmap_dir=ops.memmap_dir or "",
            device=device,
            batch_size=int(ops.batch_size or 512),
            val_frac=0.0,
            prefetch_factor=ops.prefetch_factor,
            non_blocking_copy=False,
            io_backend="auto",
        )
        if scale_loader is not None:
            for _raw in scale_loader:
                feat_s, label_s, *_ = preprocess(_raw)
                data_scale = _accumulate_data_scale(data_scale, feat_s, label_s)
        if scale_keep is not None:
            scale_keep.cleanup()
        if data_scale is not None:
            setattr(model, "__data_scale_hint__", data_scale)
        model, _, _ = Module.use_te_module(
            model, device=device, scale=data_scale
        )
        _ensure_uniform_param_dtype(
            model,
            prefer=(
                torch.bfloat16
                if (
                    getattr(device, "type", None) == "cuda"
                    and torch.cuda.is_bf16_supported()
                )
                else None
            ),
        )
        fp8_infer_ok, fp8_infer_reason = System.is_float8_supported(device)
        if fp8_infer_ok:
            model, _, _ = Module.enable_float8_prediction(
                model,
                device=device,
                prefer="te",
                logger=_float8_log,
                dynamic_activations=True,
                scale=data_scale,
            )
        else:
            AutoCast.configure(model, scale=data_scale)
            _float8_log(f"[FP8] disabled: {fp8_infer_reason}")
        model.eval()
        data_loader, _, keep = dataloader(
            memmap_dir=ops.memmap_dir or "",
            device=device,
            batch_size=int(ops.batch_size or 512),
            val_frac=0.0,
            prefetch_factor=ops.prefetch_factor,
            non_blocking_copy=True,
            io_backend="auto",
        )
        status_bar = _status_bar("Prediction", len(data_loader), device)
        flop_counter = FlopCounter(model, mode="eval", device=device)
        use_timer = getattr(device, "type", "cpu") in ("cuda", "xpu", "mps") and hasattr(
            torch, "Event"
        )
        io_bytes: float = 0.0
        io_time: float = 0.0
        comp_time: float = 0.0
        total_flops: float = 0.0
        t_fetch_start = time.perf_counter_ns()
        preds: List[torch.Tensor] = []
        with flop_counter, inference(model), AutoCast.float(device):
            for _idx, _raw in enumerate(data_loader):
                feat, _label, *_ = preprocess(_raw)
                X = to_torch(feat)
                X = torch.atleast_2d(X)
                if X.dim() != 2:
                    raise RuntimeError(
                        f"infer: feats.ndim={X.dim()} (expect 2), shape={tuple(X.shape)}"
                    )
                if X.shape[1] != int(ops.in_dim):
                    raise AssertionError(
                        "infer: feature dim mismatch — "
                        f"feats.shape[1]={X.shape[1]} != in_dim={ops.in_dim}."
                    )
                if X.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                    X = X.to(dtype=torch.float32)
                if use_timer:
                    ev_h2d_s, ev_h2d_e = (
                        torch.Event(device=device, enable_timing=True),
                        torch.Event(device=device, enable_timing=True),
                    )
                    ev_h2d_s.record()
                    X = X.to(device, non_blocking=True)
                    ev_h2d_e.record()
                    ev_h2d_e.synchronize()
                    h2d_s = float(ev_h2d_s.elapsed_time(ev_h2d_e)) / 1000.0
                else:
                    t_h2d_s = time.perf_counter_ns()
                    X = X.to(device, non_blocking=True)
                    t_h2d_e = time.perf_counter_ns()
                    h2d_s = (t_h2d_e - t_h2d_s) / 1_000_000_000.0
                wait_s = (time.perf_counter_ns() - t_fetch_start) / 1_000_000_000.0
                io_time += wait_s + h2d_s
                with contextlib.suppress(Exception):
                    io_bytes += float(X.element_size() * X.nelement())
                if use_timer:
                    ev_s, ev_e = (
                        torch.Event(device=device, enable_timing=True),
                        torch.Event(device=device, enable_timing=True),
                    )
                    ev_s.record()
                else:
                    t0 = time.perf_counter_ns()
                with no_synchronization(model, enable=True):
                    with flop_counter.step(display=False) as step_counter:
                        with contextlib.suppress(Exception):
                            mark_step = getattr(
                                getattr(torch, "compiler", None),
                                "cudagraph_mark_step_begin",
                                None,
                            )
                            if callable(mark_step):
                                mark_step()
                        y_hat, _ = model(
                            X,
                            labels_flat=None,
                            global_loss=None,
                            local_loss=None,
                            loss_weights=None,
                        )
                clip_outputs = bool(getattr(cfg, "clip_output_on_serialize", False))
                loss_space = str(
                    getattr(model, "_loss_space", getattr(cfg, "loss_space", ""))
                ).lower()
                if (
                    clip_outputs
                    and loss_space not in {"logit"}
                    and hasattr(model, "y_low_buf")
                    and hasattr(model, "y_high_buf")
                ):
                    low = float(model.y_low_buf.item())
                    high = float(model.y_high_buf.item())
                    y_hat = y_hat.clamp(min=low, max=high)
                preds.append(y_hat.detach().cpu())
                if use_timer:
                    ev_e.record()
                    ev_e.synchronize()
                    comp_time += float(ev_s.elapsed_time(ev_e)) / 1000.0
                else:
                    t1 = time.perf_counter_ns()
                    comp_time += (t1 - t0) / 1_000_000_000.0
                with contextlib.suppress(Exception):
                    step_flops = float(step_counter.get_total_flops())
                total_flops += max(0.0, step_flops)
                mbps = io_bytes / max(io_time, 1e-06) / 1_000_000.0
                tflops = total_flops / max(comp_time, 1e-06) / 1_000_000_000_000.0
                _advance_status_bar(status_bar, 1, mbps, tflops)
                t_fetch_start = time.perf_counter_ns()
        with contextlib.suppress(Exception):
            status_bar.close()
        flat = torch.cat(preds, dim=0)
        pred_struct = Root.unflatten_labels(flat, ops.out_shape)
        ret = postprocess(ops.keys or [], pred_struct)
        if ret_sink is not None:
            ret_sink.update(ret)
        if keep is not None:
            keep.cleanup()
        return None

    raise ValueError(f"unsupported ops mode: {ops.mode}")
