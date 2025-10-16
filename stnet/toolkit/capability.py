# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import math
import os

import torch

def is_cpu_bf16_supported() -> bool:
    try:
        mkldnn_ops = getattr(torch.ops, "mkldnn", None)
        if mkldnn_ops is not None and hasattr(mkldnn_ops, "_is_mkldnn_bf16_supported"):
            return bool(torch.ops.mkldnn._is_mkldnn_bf16_supported())
    except Exception:
        pass
    return False

def is_cuda_bf16_supported() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        f = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(f):
            return bool(f())
        major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
        return major >= 8
    except Exception:
        return False

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif hasattr(torch, "is_vulkan_available") and torch.is_vulkan_available():
        device = torch.device("vulkan")
    else:
        device = torch.device("cpu")
    return device

def optimizer_flags(
    device: torch.device,
    use_foreach: Optional[bool],
    use_fused: bool,
) -> Dict[str, bool]:
    devt = device.type
    flags: Dict[str, bool] = {}
    flags["foreach"] = (devt in {"cuda", "xpu"}) if use_foreach is None else bool(use_foreach)
    if use_fused and devt in {"cuda", "xpu"}:
        flags["fused"] = True
        flags["foreach"] = False
    return flags

def _is_cuda_cc_equal_or_over_90(
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[bool, str]:
    dev = torch.device(device) if device is not None else get_device()
    if dev.type != "cuda" or not torch.cuda.is_available():
        return (False, "CUDA unavailable")
    major, minor = torch.cuda.get_device_capability(dev)
    ok = major >= 9
    return (ok, f"sm_{major}{minor}")

def is_float8_supported(
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[bool, str]:
    dev = torch.device(device) if device is not None else get_device()
    ok_cc, cc = _is_cuda_cc_equal_or_over_90(dev)
    if dev.type != "cuda" or not ok_cc:
        return (False, f"FP8 requires sm_90+ (found {dev.type}, cc={cc})")
    try:
        import transformer_engine.pytorch as te
        backend = getattr(te, "__name__", "transformer_engine.pytorch")
        return (True, backend)
    except Exception:
        return (False, "transformer_engine not found")

def optimal_procs() -> dict:
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return {"nproc_per_node": (n_gpu or 1), "device": ("cuda" if n_gpu else "cpu")}

def _cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1

def optimal_threads() -> dict:
    n_cpu = _cpu_count()
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    intra = max(1, min(n_cpu, int(round(0.8 * n_cpu))))
    inter = max(1, min(4, int(math.sqrt(intra))))
    workers = (
        max(2, min(8 * n_gpu, n_cpu // max(1, n_gpu))) if n_gpu > 0 else max(2, min(8, n_cpu // 2))
    )
    return {
        "intraop": intra,
        "interop": inter,
        "dataloader_workers": workers,
        "prefetch_factor": 2,
        "pin_memory": bool(n_gpu > 0),
    }

def apply_threading_defaults() -> dict:
    threads = optimal_threads()
    os.environ.setdefault("OMP_NUM_THREADS", str(threads["intraop"]))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads["intraop"]))
    try:
        torch.set_num_threads(int(threads["intraop"]))
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(int(threads["interop"]))
    except Exception:
        pass
    return threads
