from __future__ import annotations

import contextlib

import torch


def autocast_disabled(device: torch.device):
    device_type = torch.device(device).type
    if device_type in {"cuda", "cpu", "xpu", "mps", "hpu"}:
        return torch.autocast(device_type=device_type, enabled=False)
    return contextlib.nullcontext()


def stable_work_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float64:
        return torch.float64
    if dtype in {torch.float16, torch.bfloat16, torch.float32}:
        return torch.float32
    raise TypeError(f"Unsupported stable work dtype: {dtype}")
