from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import util
from typing import Any, Iterable, Iterator

import torch
from torch import nn

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:
    class _SDPEnum:
        MATH = object()
        FLASH_ATTENTION = object()
        EFFICIENT_ATTENTION = object()
        CUDNN_ATTENTION = object()

    SDPBackend = _SDPEnum

    @contextmanager
    def sdpa_kernel(*_args: Any, **_kwargs: Any) -> Iterator[None]:
        yield


def secure_torch() -> None:
    if not hasattr(nn, "RMSNorm"):
        class _RMSNorm(nn.Module):
            def __init__(self, d_model: int, eps: float = 1e-6) -> None:
                super().__init__()
                self.eps = float(eps)
                self.weight = nn.Parameter(torch.ones(d_model))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                inv_rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
                return x * inv_rms * self.weight

        setattr(nn, "RMSNorm", _RMSNorm)

    if not hasattr(torch, "fmin"):
        def _fmin(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            a, b = torch.broadcast_tensors(a, b)
            a_nan = torch.isnan(a)
            b_nan = torch.isnan(b)
            return torch.where(a_nan & ~b_nan, b, torch.where(b_nan & ~a_nan, a, torch.minimum(a, b)))

        setattr(torch, "fmin", _fmin)

    if not hasattr(torch, "nanmin"):
        def _nanmin(x: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> Any:
            mask = torch.isfinite(x)
            xp = torch.where(mask, x, torch.full_like(x, float("inf")))
            if dim is None:
                return xp.min()
            values, indices = xp.min(dim=dim, keepdim=keepdim)
            return values, indices

        setattr(torch, "nanmin", _nanmin)

    if not hasattr(torch, "nanmax"):
        def _nanmax(x: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> Any:
            mask = torch.isfinite(x)
            xp = torch.where(mask, x, torch.full_like(x, float("-inf")))
            if dim is None:
                return xp.max()
            values, indices = xp.max(dim=dim, keepdim=keepdim)
            return values, indices

        setattr(torch, "nanmax", _nanmax)

    if not hasattr(torch, "nansum"):
        def _nansum(
            x: torch.Tensor,
            dim: int | tuple[int, ...] | None = None,
            keepdim: bool = False,
            *,
            dtype: torch.dtype | None = None,
        ) -> torch.Tensor:
            target_dtype = dtype if dtype is not None else x.dtype
            x_cast = x.to(target_dtype)
            mask = torch.isfinite(x_cast)
            zero = torch.zeros((), device=x_cast.device, dtype=x_cast.dtype)
            x_masked = torch.where(mask, x_cast, zero)
            return torch.sum(x_masked, dim=dim, keepdim=keepdim, dtype=dtype)

        setattr(torch, "nansum", _nansum)


def lazy_import(module_name: str) -> Any:
    spec = util.find_spec(module_name)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(module_name)
    loader = util.LazyLoader(spec.loader)
    spec.loader = loader
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    loader.exec_module(module)
    return module


def has_fsdp2() -> bool:
    try:
        __import__("torch.distributed.fsdp.fully_shard")
        return True
    except Exception:
        return False


def has_arrow_flight() -> bool:
    try:
        return util.find_spec("pyarrow.flight") is not None
    except Exception:
        return False


def has_zero_mq() -> bool:
    try:
        return util.find_spec("zmq") is not None
    except Exception:
        return False


def has_cuda_ipc() -> bool:
    try:
        available = bool(getattr(torch.cuda, "is_available", lambda: False)())
        return available and hasattr(torch.cuda, "ipc_collect")
    except Exception:
        return False


def has_gds() -> bool:
    try:
        if util.find_spec("kvikio") is not None:
            return True
    except Exception:
        pass
    try:
        if os.path.exists("/proc/modules"):
            with open("/proc/modules", "r", encoding="utf-8") as handle:
                if "nvidia_fs" in handle.read():
                    return True
    except Exception:
        pass
    return os.path.exists("/etc/cufile.json")


@dataclass(frozen=True)
class EnvInfo:
    torch_version: str
    cuda_available: bool
    cuda_device_count: int
    has_fsdp2: bool
    has_arrow_flight: bool
    has_zero_mq: bool
    has_cuda_ipc: bool
    has_gds: bool


def env_info() -> EnvInfo:
    cuda_available = bool(getattr(torch.cuda, "is_available", lambda: False)())
    cuda_devices = int(getattr(torch.cuda, "device_count", lambda: 0)())
    return EnvInfo(
        torch_version=str(getattr(torch, "__version__", "unknown")),
        cuda_available=cuda_available,
        cuda_device_count=cuda_devices,
        has_fsdp2=has_fsdp2(),
        has_arrow_flight=has_arrow_flight(),
        has_zero_mq=has_zero_mq(),
        has_cuda_ipc=has_cuda_ipc(),
        has_gds=has_gds(),
    )


def _to_sdpa_backends(backends: Iterable[Any] | None = None) -> list[Any]:
    if backends is None:
        candidates: Iterable[Any] = (
            "FLASH_ATTENTION",
            "EFFICIENT_ATTENTION",
            "CUDNN_ATTENTION",
            "MATH",
        )
    else:
        candidates = backends
    resolved: list[Any] = []
    for candidate in candidates:
        if isinstance(candidate, str):
            attr = candidate.upper()
            if hasattr(SDPBackend, attr):
                resolved.append(getattr(SDPBackend, attr))
            continue
        if candidate is not None:
            resolved.append(candidate)
    return resolved


__all__ = [
    "secure_torch",
    "lazy_import",
    "has_fsdp2",
    "has_arrow_flight",
    "has_zero_mq",
    "has_cuda_ipc",
    "has_gds",
    "env_info",
    "EnvInfo",
    "sdpa_kernel",
    "SDPBackend",
    "_to_sdpa_backends",
]
