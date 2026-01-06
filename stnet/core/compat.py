# -*- coding: utf-8 -*-
from __future__ import annotations

import threading
from contextlib import contextmanager, suppress
from functools import partial
from typing import Any, Iterator

import torch
from torch import nn

from .graph import compile_distributed_safe

try:
    import torchdistx.fake
except Exception:
    _tdx_is_fake = None
else:
    _tdx_is_fake = getattr(torchdistx.fake, "is_fake", None)

try:
    from torch._subclasses.fake_tensor import FakeTensor
except Exception:
    FakeTensor = tuple()


_PATCH_LOCK = threading.RLock()

_TORCH_COMPAT: TorchCompat | None = None

RMSNorm = getattr(nn, "RMSNorm", None)


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
    torch_mod: Any, x: Any, dim: int | None = None, keepdim: bool = False
) -> Any:
    if isinstance(x, torch.Tensor) and not torch_mod.is_floating_point(x):
        return x.min() if dim is None else x.min(dim=dim, keepdim=keepdim)
    mask = torch_mod.isfinite(x)
    xp = torch_mod.where(mask, x, torch_mod.full_like(x, float("inf")))
    if dim is None:
        values = xp.min()
        any_valid = mask.any()
        return torch_mod.where(
            any_valid, values, torch_mod.full_like(values, float("nan"))
        )
    values, indices = xp.min(dim=dim, keepdim=keepdim)
    any_valid = mask.any(dim=dim, keepdim=keepdim)
    values = torch_mod.where(
        any_valid, values, torch_mod.full_like(values, float("nan"))
    )
    indices = torch_mod.where(any_valid, indices, torch_mod.zeros_like(indices))
    return (values, indices)


def _nanmax_impl(
    torch_mod: Any, x: Any, dim: int | None = None, keepdim: bool = False
) -> Any:
    if isinstance(x, torch.Tensor) and not torch_mod.is_floating_point(x):
        return x.max() if dim is None else x.max(dim=dim, keepdim=keepdim)
    mask = torch_mod.isfinite(x)
    xp = torch_mod.where(mask, x, torch_mod.full_like(x, float("-inf")))
    if dim is None:
        values = xp.max()
        any_valid = mask.any()
        return torch_mod.where(
            any_valid, values, torch_mod.full_like(values, float("nan"))
        )
    values, indices = xp.max(dim=dim, keepdim=keepdim)
    any_valid = mask.any(dim=dim, keepdim=keepdim)
    values = torch_mod.where(
        any_valid, values, torch_mod.full_like(values, float("nan"))
    )
    indices = torch_mod.where(any_valid, indices, torch_mod.zeros_like(indices))
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
    _ = args
    if dtype is not None and not isinstance(dtype, torch.dtype):
        with suppress(Exception):
            dtype = getattr(torch, str(dtype).split(".")[-1], None)
    x_cast = x.to(dtype) if dtype is not None else x
    if isinstance(x_cast, torch.Tensor) and not torch_mod.is_floating_point(x_cast):
        return torch_mod.sum(x_cast, dim=dim, keepdim=keepdim, **kwargs)
    nan_to_num = getattr(torch_mod, "nan_to_num", None)
    if callable(nan_to_num):
        with suppress(Exception):
            x_norm = nan_to_num(x_cast, nan=0.0, posinf=0.0, neginf=0.0)
            return torch_mod.sum(x_norm, dim=dim, keepdim=keepdim, **kwargs)
    mask = torch_mod.isfinite(x_cast)
    zero = torch_mod.zeros((), device=x_cast.device, dtype=x_cast.dtype)
    x_masked = torch_mod.where(mask, x_cast, zero)
    return torch_mod.sum(x_masked, dim=dim, keepdim=keepdim, **kwargs)


def is_fake_tensor(value: Any) -> bool:
    if not isinstance(value, torch.Tensor):
        return False
    if _tdx_is_fake is not None:
        with suppress(Exception):
            return bool(_tdx_is_fake(value))
    return (
        isinstance(value, FakeTensor) or getattr(value, "fake_mode", None) is not None
    )


def is_meta_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and getattr(value, "is_meta", False)


def is_meta_or_fake_tensor(value: Any) -> bool:
    return is_meta_tensor(value) or is_fake_tensor(value)


def torch_compat(
    module: Any | None = None, nn_module: Any | None = None
) -> TorchCompat:
    global _TORCH_COMPAT
    with _PATCH_LOCK:
        if _TORCH_COMPAT is None or module is not None or nn_module is not None:
            compat = TorchCompat(module=module, nn_module=nn_module)
        else:
            compat = _TORCH_COMPAT
        compat.apply()
        _TORCH_COMPAT = compat
        return compat


class _RMSNormFallback(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-06) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(d_model)))

    def forward(self, x: Any) -> Any:
        inv_rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * inv_rms * self.weight


class _StochasticDepthFallback(nn.Module):
    def __init__(self, p: float = 0.0, mode: str = "row") -> None:
        super().__init__()
        self.p = float(p)
        self.mode = str(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        if keep <= 0.0:
            return torch.zeros_like(x)
        if self.mode == "row" and x.dim() >= 2:
            noise_shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        else:
            noise_shape = x.shape
        noise = x.new_empty(noise_shape).bernoulli_(keep).div_(keep)
        return x * noise


class _SDPBackendFallback:
    MATH = object()
    FLASH_ATTENTION = object()
    EFFICIENT_ATTENTION = object()
    CUDNN_ATTENTION = object()


class TorchCompat:
    def __init__(self, module: Any | None = None, nn_module: Any | None = None) -> None:
        self.module = module if module is not None else torch
        self.nn_module = (
            nn_module if nn_module is not None else getattr(self.module, "nn", nn)
        )

    def _patch_rmsnorm(self) -> None:
        global RMSNorm
        if hasattr(self.nn_module, "RMSNorm"):
            RMSNorm = getattr(self.nn_module, "RMSNorm", None)
            return
        setattr(self.nn_module, "RMSNorm", _RMSNormFallback)
        RMSNorm = getattr(self.nn_module, "RMSNorm", None)

    def _patch_fmin(self) -> None:
        if hasattr(self.module, "fmin"):
            return
        setattr(self.module, "fmin", partial(_fmin_impl, self.module))

    def _patch_nanmin(self) -> None:
        if hasattr(self.module, "nanmin"):
            return
        setattr(self.module, "nanmin", partial(_nanmin_impl, self.module))

    def _patch_nanmax(self) -> None:
        if hasattr(self.module, "nanmax"):
            return
        setattr(self.module, "nanmax", partial(_nanmax_impl, self.module))

    def _patch_nansum(self) -> None:
        if hasattr(self.module, "nansum"):
            return
        setattr(self.module, "nansum", partial(_nansum_impl, self.module))

    def apply(self) -> None:
        with _PATCH_LOCK:
            self._patch_rmsnorm()
            self._patch_fmin()
            self._patch_nanmin()
            self._patch_nanmax()
            self._patch_nansum()
            compile_distributed_safe()


try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:
    SDPBackend = _SDPBackendFallback

    @contextmanager
    def sdpa_kernel(*backends: Any) -> Iterator[None]:
        _ = backends
        yield

StochasticDepth = getattr(nn, "StochasticDepth", None)
if StochasticDepth is None:
    StochasticDepth = _StochasticDepthFallback
