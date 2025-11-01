# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from importlib import import_module, util
from typing import Any, Iterable, Iterator, Sequence

import torch
from torch import nn

try:  # pragma: no cover - optional dependency
    from torchdistx.fake import is_fake as _tdx_is_fake  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - torchdistx not installed
    _tdx_is_fake = None  # type: ignore[assignment]

try:  # pragma: no cover - private API best-effort
    from torch._subclasses.fake_tensor import FakeTensor  # type: ignore
except Exception:  # pragma: no cover - fallback when private API unavailable
    FakeTensor = tuple()  # type: ignore[assignment]


try:
    # Prefer torch.compiler.disable (PyTorch ≥2.5)
    _torch_compile_disable = torch.compiler.disable  # type: ignore[attr-defined]
except Exception:
    try:
        # Fallback for PyTorch 2.0–2.4
        import torch._dynamo as _dynamo  # type: ignore

        _torch_compile_disable = _dynamo.disable  # type: ignore[attr-defined]
    except Exception:

        def _torch_compile_disable(fn=None, *, recursive=False):  # type: ignore[no-untyped-def]
            if fn is None:
                return lambda real_fn: real_fn
            return fn


if not hasattr(torch, "compiler"):
    class _TorchCompilerNamespace:
        @staticmethod
        def disable(fn=None, *, recursive=False):  # type: ignore[no-untyped-def]
            return _torch_compile_disable(fn, recursive=recursive)


    torch.compiler = _TorchCompilerNamespace()  # type: ignore[attr-defined]
elif not hasattr(torch.compiler, "disable"):

    def _compiler_disable_passthrough(fn=None, *, recursive=False):  # type: ignore[no-untyped-def]
        return _torch_compile_disable(fn, recursive=recursive)


    torch.compiler.disable = _compiler_disable_passthrough  # type: ignore[attr-defined]


if hasattr(nn, "RMSNorm"):
    RMSNorm = torch.compiler.disable(nn.RMSNorm, recursive=True)  # type: ignore[attr-defined]
else:
    RMSNorm = None  # type: ignore[assignment]

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
    def sdpa_kernel(*backends: Any) -> Iterator[None]:
        _ = backends
        del _
        yield


def _fmin_impl(torch_mod: Any, a: Any, b: Any) -> Any:
    a, b = torch_mod.broadcast_tensors(a, b)
    a_nan = torch_mod.isnan(a)
    b_nan = torch_mod.isnan(b)
    return torch_mod.where(
        a_nan & ~b_nan,
        b,
        torch_mod.where(b_nan & ~a_nan, a, torch_mod.minimum(a, b)),
    )


def _nanmin_impl(
    torch_mod: Any,
    x: Any,
    dim: int | None = None,
    keepdim: bool = False,
) -> Any:
    mask = torch_mod.isfinite(x)
    xp = torch_mod.where(mask, x, torch_mod.full_like(x, float("inf")))
    if dim is None:
        if not bool(mask.any()):
            return torch_mod.full((), float("nan"), device=x.device, dtype=x.dtype)
        return xp.min()
    values, indices = xp.min(dim=dim, keepdim=keepdim)
    any_valid = mask.any(dim=dim, keepdim=keepdim)
    values = torch_mod.where(
        any_valid,
        values,
        torch_mod.full_like(values, float("nan")),
    )
    indices = torch_mod.where(
        any_valid,
        indices,
        torch_mod.zeros_like(indices),
    )
    return (values, indices)


def _nanmax_impl(
    torch_mod: Any,
    x: Any,
    dim: int | None = None,
    keepdim: bool = False,
) -> Any:
    mask = torch_mod.isfinite(x)
    xp = torch_mod.where(mask, x, torch_mod.full_like(x, float("-inf")))
    if dim is None:
        if not bool(mask.any()):
            return torch_mod.full((), float("nan"), device=x.device, dtype=x.dtype)
        return xp.max()
    values, indices = xp.max(dim=dim, keepdim=keepdim)
    any_valid = mask.any(dim=dim, keepdim=keepdim)
    values = torch_mod.where(
        any_valid,
        values,
        torch_mod.full_like(values, float("nan")),
    )
    indices = torch_mod.where(
        any_valid,
        indices,
        torch_mod.zeros_like(indices),
    )
    return (values, indices)


def _nansum_impl(
    torch_mod: Any,
    x: Any,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    *args: Any,
    dtype: Any = None,
    **kwargs: Any,
) -> Any:
    _ = args, kwargs
    x_cast = x.to(dtype) if dtype is not None else x
    mask = torch_mod.isfinite(x_cast)
    zero = torch_mod.zeros((), device=x_cast.device, dtype=x_cast.dtype)
    x_masked = torch_mod.where(mask, x_cast, zero)
    return torch_mod.sum(x_masked, dim=dim, keepdim=keepdim)


_TORCH_COMPAT: TorchCompat | None = None


class TorchCompat:
    def __init__(
        self,
        module: Any | None = None,
        nn_module: Any | None = None,
    ) -> None:
        self.module = module if module is not None else torch
        self.nn_module = nn_module if nn_module is not None else getattr(
            self.module, "nn", nn
        )

    def apply(self) -> None:
        self._ensure_rmsnorm()
        self._ensure_fmin()
        self._ensure_nanmin()
        self._ensure_nanmax()
        self._ensure_nansum()

    def _ensure_rmsnorm(self) -> None:
        global RMSNorm
        if hasattr(self.nn_module, "RMSNorm"):
            if RMSNorm is None:
                RMSNorm = torch.compiler.disable(self.nn_module.RMSNorm, recursive=True)  # type: ignore[attr-defined]
            return
        torch_mod = self.module
        nn_mod = self.nn_module

        class _RMSNorm(nn_mod.Module):
            def __init__(self, d_model: int, eps: float = 1e-06) -> None:
                super().__init__()
                self.eps = float(eps)
                self.weight = nn_mod.Parameter(torch_mod.ones(d_model))

            def forward(self, x: Any) -> Any:
                inv_rms = (
                    x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
                )
                return x * inv_rms * self.weight

        setattr(self.nn_module, "RMSNorm", _RMSNorm)
        RMSNorm = torch.compiler.disable(self.nn_module.RMSNorm, recursive=True)  # type: ignore[attr-defined]

    def _ensure_fmin(self) -> None:
        if hasattr(self.module, "fmin"):
            return
        setattr(self.module, "fmin", partial(_fmin_impl, self.module))

    def _ensure_nanmin(self) -> None:
        if hasattr(self.module, "nanmin"):
            return
        setattr(self.module, "nanmin", partial(_nanmin_impl, self.module))

    def _ensure_nanmax(self) -> None:
        if hasattr(self.module, "nanmax"):
            return
        setattr(self.module, "nanmax", partial(_nanmax_impl, self.module))

    def _ensure_nansum(self) -> None:
        if hasattr(self.module, "nansum"):
            return
        setattr(self.module, "nansum", partial(_nansum_impl, self.module))


def patch_torch(
    module: Any | None = None,
    nn_module: Any | None = None,
) -> TorchCompat:
    global _TORCH_COMPAT
    if _TORCH_COMPAT is None or module is not None or nn_module is not None:
        compat = TorchCompat(module=module, nn_module=nn_module)
    else:
        compat = _TORCH_COMPAT
    compat.apply()
    _TORCH_COMPAT = compat
    return compat


def maybe_mark_cudagraph_step_end() -> None:
    """Call torch.compiler.cudagraph_mark_step_end() if present (no-op otherwise)."""
    try:
        mark_step = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_end", None)
        if callable(mark_step):
            mark_step()
    except Exception:
        pass
def is_fake_tensor(value: Any) -> bool:
    """Return ``True`` when ``value`` references a FakeTensor placeholder."""

    if not isinstance(value, torch.Tensor):
        return False
    if _tdx_is_fake is not None:
        try:
            return bool(_tdx_is_fake(value))
        except Exception:
            pass
    return isinstance(value, FakeTensor) or getattr(value, "fake_mode", None) is not None


def is_meta_tensor(value: Any) -> bool:
    """Check whether a tensor is backed by the meta device placeholder."""

    return isinstance(value, torch.Tensor) and getattr(value, "is_meta", False)


def is_meta_or_fake_tensor(value: Any) -> bool:
    """Return ``True`` when ``value`` is either a meta tensor or a fake tensor."""

    return is_meta_tensor(value) or is_fake_tensor(value)


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


patch_torch()
