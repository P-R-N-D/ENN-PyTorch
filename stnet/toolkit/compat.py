from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module, util
from typing import Any, Iterable, Iterator, Sequence

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
    def sdpa_kernel(*args: Any, **kwargs: Any) -> Iterator[None]:
        if args or kwargs:
            _ = (args, kwargs)
            del _
        yield


_ARROW_COMPAT: "ArrowCompat | None" = None


def patch_torch() -> None:
    if not hasattr(nn, "RMSNorm"):

        class _RMSNorm(nn.Module):
            def __init__(self, d_model: int, eps: float = 1e-06) -> None:
                super().__init__()
                self.eps = float(eps)
                self.weight = nn.Parameter(torch.ones(d_model))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                inv_rms = (
                    x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
                )
                return x * inv_rms * self.weight

        setattr(nn, "RMSNorm", _RMSNorm)
    if not hasattr(torch, "fmin"):

        def _fmin(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            a, b = torch.broadcast_tensors(a, b)
            a_nan = torch.isnan(a)
            b_nan = torch.isnan(b)
            return torch.where(
                a_nan & ~b_nan,
                b,
                torch.where(b_nan & ~a_nan, a, torch.minimum(a, b)),
            )

        setattr(torch, "fmin", _fmin)
    if not hasattr(torch, "nanmin"):

        def _nanmin(
            x: torch.Tensor, dim: int | None = None, keepdim: bool = False
        ) -> Any:
            mask = torch.isfinite(x)
            xp = torch.where(mask, x, torch.full_like(x, float("inf")))
            if dim is None:
                return xp.min()
            values, indices = xp.min(dim=dim, keepdim=keepdim)
            return (values, indices)

        setattr(torch, "nanmin", _nanmin)
    if not hasattr(torch, "nanmax"):

        def _nanmax(
            x: torch.Tensor, dim: int | None = None, keepdim: bool = False
        ) -> Any:
            mask = torch.isfinite(x)
            xp = torch.where(mask, x, torch.full_like(x, float("-inf")))
            if dim is None:
                return xp.max()
            values, indices = xp.max(dim=dim, keepdim=keepdim)
            return (values, indices)

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


class ArrowCompat:
    def __init__(self, module: Any, flight: Any) -> None:
        self.module = module
        self.flight = flight

    def to_numpy(self, array: Any, *, zero_copy_only: bool = True) -> Any:
        try:
            return array.to_numpy(zero_copy_only=zero_copy_only)
        except TypeError:
            return array.to_numpy()

    def fixed_shape_list_from_arrays(
        self, values: Any, shape_or_size: int | Sequence[int]
    ) -> Any:
        errors: list[BaseException] = []
        array_cls = getattr(self.module, "FixedShapeArrayList", None)
        if array_cls is not None:
            try:
                return array_cls.from_arrays(values, shape_or_size)
            except TypeError as exc:
                errors.append(exc)
        fallback_cls = getattr(self.module, "FixedSizeListArray", None)
        if fallback_cls is not None:
            try:
                length = 1
                if isinstance(shape_or_size, (list, tuple)):
                    dims = list(shape_or_size)
                    if not dims:
                        length = 1
                    else:
                        for dim in dims:
                            length *= int(dim)
                else:
                    length = int(shape_or_size)
                return fallback_cls.from_arrays(values, length)
            except TypeError as exc:
                errors.append(exc)
        if errors:
            raise errors[-1]
        raise AttributeError(
            "pyarrow does not provide a fixed-shape list array implementation"
        )


def patch_arrow(module: Any | None = None) -> ArrowCompat:
    global _ARROW_COMPAT
    if _ARROW_COMPAT is not None:
        return _ARROW_COMPAT
    if module is None:
        import pyarrow as pa  # type: ignore
    else:
        pa = module
    try:
        flight = import_module("pyarrow.flight")
    except Exception:
        flight = None
    if not hasattr(pa, "FixedShapeArrayList") and hasattr(pa, "FixedSizeListArray"):
        setattr(pa, "FixedShapeArrayList", pa.FixedSizeListArray)
    compat = ArrowCompat(pa, flight)
    _ARROW_COMPAT = compat
    return compat


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
    "patch_torch",
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
