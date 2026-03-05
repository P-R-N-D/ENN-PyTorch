# -*- coding: utf-8 -*-
from __future__ import annotations

from contextlib import contextmanager, suppress
from functools import partial
from types import ModuleType
from typing import Any, Iterator, Self

import torch
from torch import nn

from ..nn.graph import compile_distributed_safe
from .concurrency import Mutex

_PATCH_LOCK = Mutex(reentrant=True)
_TORCH_COMPAT: TorchCompat | None = None


def _fmin_impl(
    tm: ModuleType, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    a, b = tm.broadcast_tensors(a, b)
    an, bn = tm.isnan(a), tm.isnan(b)
    return tm.where(an & ~bn, b, tm.where(bn & ~an, a, tm.minimum(a, b)))


def _nan_mm_impl(
    tm: ModuleType,
    x: torch.Tensor,
    dim: int | None,
    keepdim: bool,
    op: str,
    fill: float | str,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    kwargs: dict[str, Any] = (
        {"dim": dim, "keepdim": keepdim} if dim is not None else {}
    )
    if isinstance(x, torch.Tensor) and not tm.is_floating_point(x):
        return getattr(x, op)(**kwargs)

    match op:
        case "min" | "max":
            pass
        case _:
            msg = f"Unsupported reduction operator: {op}"
            raise ValueError(msg)

    mask = tm.isfinite(x)
    xp = tm.where(mask, x, tm.full_like(x, float(fill)))

    if dim is None:
        res = getattr(xp, op)()
        return tm.where(mask.any(), res, tm.full_like(res, float("nan")))

    val, idx = getattr(xp, op)(dim=dim, keepdim=keepdim)
    valid = mask.any(dim=dim, keepdim=keepdim)
    return tm.where(valid, val, tm.full_like(val, float("nan"))), tm.where(
        valid, idx, tm.zeros_like(idx)
    )


def _nanmin_impl(
    tm: ModuleType,
    x: torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return _nan_mm_impl(tm, x, dim, keepdim, "min", "inf")


def _nanmax_impl(
    tm: ModuleType,
    x: torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return _nan_mm_impl(tm, x, dim, keepdim, "max", "-inf")


def _nansum_impl(
    tm: ModuleType,
    x: torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *args: Any,
    dtype: torch.dtype | str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    _ = args
    if dtype and not isinstance(dtype, torch.dtype):
        with suppress(Exception):
            dtype = getattr(torch, str(dtype).split(".")[-1], None)
    x_cast = x.to(dtype) if dtype is not None else x
    if isinstance(x_cast, torch.Tensor) and not tm.is_floating_point(x_cast):
        return tm.sum(x_cast, dim=dim, keepdim=keepdim, **kwargs)
    if callable(n2n := getattr(tm, "nan_to_num", None)):
        with suppress(Exception):
            return tm.sum(
                n2n(x_cast, nan=0.0, posinf=0.0, neginf=0.0),
                dim=dim,
                keepdim=keepdim,
                **kwargs,
            )
    return tm.sum(
        tm.where(
            tm.isfinite(x_cast),
            x_cast,
            tm.zeros((), device=x_cast.device, dtype=x_cast.dtype),
        ),
        dim=dim,
        keepdim=keepdim,
        **kwargs,
    )


def torch_compat(
    module: Any | None = None, nn_module: Any | None = None
) -> TorchCompat:
    global _TORCH_COMPAT
    with _PATCH_LOCK:
        compat = (
            TorchCompat(module=module, nn_module=nn_module)
            if _TORCH_COMPAT is None or module or nn_module
            else _TORCH_COMPAT
        )
        compat.apply()
        _TORCH_COMPAT = compat
        return compat


class _RMSNormFallback(nn.Module):
    def __init__(self: Self, d_model: int, eps: float = 1e-06) -> None:
        super().__init__()
        self.eps, self.weight = (
            float(eps),
            nn.Parameter(torch.ones(int(d_model))),
        )

    def forward(self: Self, x: Any) -> Any:
        inv_rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * inv_rms * self.weight


class _StochasticDepthFallback(nn.Module):
    def __init__(self: Self, p: float = 0.0, mode: str = "row") -> None:
        super().__init__()
        self.p, self.mode = float(p), str(mode)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        elif (keep := 1.0 - self.p) <= 0.0:
            return torch.zeros_like(x)
        shape = (
            (x.shape[0],) + (1,) * (x.dim() - 1)
            if self.mode == "row" and x.dim() >= 2
            else x.shape
        )
        return x * x.new_empty(shape).bernoulli_(keep).div_(keep)


class _SDPBackendFallback:
    MATH = FLASH_ATTENTION = EFFICIENT_ATTENTION = CUDNN_ATTENTION = object()


class TorchCompat:
    def __init__(
        self: Self, module: Any | None = None, nn_module: Any | None = None
    ) -> None:
        self.module = module if module is not None else torch
        self.nn_module = (
            nn_module
            if nn_module is not None
            else getattr(self.module, "nn", nn)
        )

    def apply(self: Self) -> None:
        with _PATCH_LOCK:
            global RMSNorm
            if not hasattr(self.nn_module, "RMSNorm"):
                setattr(self.nn_module, "RMSNorm", _RMSNormFallback)
            RMSNorm = getattr(self.nn_module, "RMSNorm", None)

            patches: tuple[tuple[str, Any], ...] = (
                ("fmin", _fmin_impl),
                ("nanmin", _nanmin_impl),
                ("nanmax", _nanmax_impl),
                ("nansum", _nansum_impl),
            )
            for name, impl in patches:
                if not hasattr(self.module, name):
                    setattr(self.module, name, partial(impl, self.module))
            compile_distributed_safe()


RMSNorm = getattr(nn, "RMSNorm", None)

StochasticDepth = (
    getattr(nn, "StochasticDepth", None) or _StochasticDepthFallback
)

try:
    from torch.nn.attention import SDPBackend
    from torch.nn.attention import sdpa_kernel
except Exception:
    SDPBackend = _SDPBackendFallback

    @contextmanager
    def sdpa_kernel(*backends: Any) -> Iterator[None]:
        _ = backends
        yield
