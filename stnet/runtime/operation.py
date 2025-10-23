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
from ..data.collate import dataloader, postprocess, preprocess
from ..utils.datatype import to_torch
from ..data.dataset import SampleReader
from ..utils.platform import Distributed, Network, System
from ..utils.optimization import (
    AdamW,
    AutoCast,
    FlopCounter,
    LossWeightController,
    Module,
    inference,
    joining,
    no_synchronization,
)

try:
    from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
except ImportError:

    def precompute_float8_dynamic_scale_for_fsdp(*args: Any, **kwargs: Any) -> Any:
        return None


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


def dl_state_path(directory: str) -> str:
    return os.path.join(directory, _DL_STATE_FILE)


def _float8_log(msg: str, *, only_main_rank: bool = True) -> None:
    if not only_main_rank:
        warnings.warn(msg)
        return
    try:
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
    except Exception:
        pass
    warnings.warn(msg)
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
        file=sys.stdout,
    )
    bar.set_postfix_str("0.00 MB/s, 0.00 TFLOPS", refresh=False)
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


def recompute_y_stats(model: Any, loader: Any) -> None:
    dev = next(model.parameters()).device
    model.y_min.fill_(float("inf"))
    model.y_max.fill_(float("-inf"))
    model.y_sum.zero_()
    model.y_sum2.zero_()
    model.y_count.zero_()
    model.y_stats_ready.fill_(False)
    model.eval()
    with inference(model):
        for X, Y in loader:
            if Y is None:
                continue
            model.update_y_stats(Y.to(dev))
        model.finalize_y_stats()


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
    nprocs = get_world_size(device) if device.type in ("cuda", "xpu") else 1
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
    flop_breakdown_epoch: Dict[str, float] = {}
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
                            y_hat, loss_val = model(
                                X,
                                labels_flat=Y_flat,
                                global_loss=top_loss,
                                local_loss=bottom_loss,
                                loss_weights=loss_controller.weights(),
                            )
                        accum_scale = max(1, grad_accum_steps)
                        loss_for_backprop = loss_val / float(accum_scale)
                        scaler.scale(loss_for_backprop).backward()
                        if should_sync:
                            scaler.unscale_(optimizer)
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
                                loss_weights=loss_controller.weights(),
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
        model, _, _ = Module.use_te_module(model, device=device)
        _ensure_uniform_param_dtype(
            model,
            prefer=(
                torch.bfloat16
                if getattr(device, "type", None) == "cuda"
                and torch.cuda.is_bf16_supported()
                else None
            ),
        )
        model, _, _ = Module.enable_float8_training(
            model, device=device, prefer="te", logger=_float8_log
        )
        model.train()
        world = get_world_size(device)
        mesh = init_device_mesh(
            "cuda" if device.type == "cuda" else device.type, (world,)
        )
        match device.type:
            case "cuda":
                param_dtype = (
                    torch.bfloat16 if System.is_cuda_bf16_supported() else torch.float16
                )
                reduce_dtype = torch.float32
                cast_forward_inputs = True
            case "xpu":
                param_dtype = torch.bfloat16
                reduce_dtype = torch.float32
                cast_forward_inputs = False
            case "mps":
                param_dtype = torch.float16
                reduce_dtype = param_dtype
                cast_forward_inputs = False
            case "cpu":
                param_dtype = (
                    torch.bfloat16 if System.is_cpu_bf16_supported() else torch.float32
                )
                reduce_dtype = torch.float32
                cast_forward_inputs = False if System.is_cpu_bf16_supported() else True
            case _:
                param_dtype = torch.float32
                reduce_dtype = torch.float32
                cast_forward_inputs = True
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            output_dtype=None,
            cast_forward_inputs=cast_forward_inputs,
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
        net_params = [p for p in model.parameters()]
        optimizer = AdamW.float(
            net_params,
            lr=ops.base_lr,
            weight_decay=ops.weight_decay,
            use_fp8=(device.type == "cuda"),
            use_foreach=False,
            use_fused=False,
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
        train_loader0, val_loader0, keep0 = dataloader(
            memmap_dir=ops.memmap_dir,
            device=device,
            batch_size=int(ops.batch_size or 128),
            val_frac=float(ops.val_frac),
            prefetch_factor=ops.prefetch_factor,
            non_blocking_copy=bool(ops.overlap_h2d),
            io_backend="auto",
        )
        train_step_count = 0
        for train_step_count, _raw in enumerate(train_loader0, start=1):
            _feat0, _label0, *_ = preprocess(_raw)
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
        _t = StudentsTLoss(
            confidence=0.99,
            metric="t_value",
            two_tailed=True,
            df=4,
            mu_mode="error",
            std_mode="pooled",
            ddof=1,
            clamp_max=8.0,
            detach_stats=True,
            dim=-1,
            reduction="none",
        )
        _z = StandardNormalLoss(
            confidence=0.99,
            metric="z_value",
            two_tailed=True,
            penalty="softplus",
            tau=1.0,
            mu_mode="error",
            std_mode="pooled",
            ddof=1,
            clamp_max=8.0,
            detach_stats=True,
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

            for _ in range(int(ops.epochs)):
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
                world = max(1, get_world_size(device))
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
        model, _, _ = Module.use_te_module(model, device=device)
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
        model, _, _ = Module.enable_float8_prediction(
            model,
            device=device,
            prefer="te",
            logger=_float8_log,
            dynamic_activations=True,
        )
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
