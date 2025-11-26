# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
from contextlib import contextmanager, suppress
from functools import partial
from typing import Any, Callable, Iterator

import torch
from torch import nn

try:
    from torchdistx.fake import is_fake as _tdx_is_fake
except Exception:
    _tdx_is_fake = None

try:
    from torch._subclasses.fake_tensor import FakeTensor
except Exception:
    FakeTensor = tuple()

try:
    from torch import compiler as _TORCH_COMPILER
except Exception:
    _TORCH_COMPILER = None

try:
    import torch._dynamo as _TORCH_DYNAMO
except Exception:
    _TORCH_DYNAMO = None

_TORCH_COMPILE_DISABLE = None
if _TORCH_COMPILER is not None:
    _TORCH_COMPILE_DISABLE = getattr(_TORCH_COMPILER, "disable", None)
if _TORCH_COMPILE_DISABLE is None and _TORCH_DYNAMO is not None:
    _TORCH_COMPILE_DISABLE = getattr(_TORCH_DYNAMO, "disable", None)

_COLLECTIVE_NAMES: tuple[str, ...] = (
    "all_gather",
    "all_gather_into_tensor",
    "all_reduce",
    "reduce_scatter_tensor",
    "broadcast",
    "barrier",
)

if hasattr(nn, "RMSNorm"):
    RMSNorm = nn.RMSNorm
else:
    RMSNorm = None

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
    if dtype is not None and not isinstance(dtype, torch.dtype):
        with suppress(Exception):
            dtype = getattr(torch, str(dtype).split(".")[-1], None)
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
        self.nn_module = (
            nn_module if nn_module is not None else getattr(self.module, "nn", nn)
        )

    def apply(self) -> None:
        self._patch_rmsnorm()
        self._patch_fmin()
        self._patch_nanmin()
        self._patch_nanmax()
        self._patch_nansum()
        torch_safe_distributed()

    def _patch_rmsnorm(self) -> None:
        global RMSNorm
        if hasattr(self.nn_module, "RMSNorm"):
            RMSNorm = self.nn_module.RMSNorm
            return
        torch_mod = self.module
        nn_mod = self.nn_module

        class _RMSNorm(nn_mod.Module):
            def __init__(self, d_model: int, eps: float = 1e-06) -> None:
                super().__init__()
                self.eps = float(eps)
                self.weight = nn_mod.Parameter(torch_mod.ones(d_model))

            def forward(self, x: Any) -> Any:
                inv_rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
                return x * inv_rms * self.weight

        setattr(self.nn_module, "RMSNorm", _RMSNorm)
        RMSNorm = self.nn_module.RMSNorm

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


def cudagraph_step_end() -> None:
    mark_step = getattr(_TORCH_COMPILER, "cudagraph_mark_step_end", None)
    if callable(mark_step):
        with suppress(Exception):
            mark_step()


def is_fake_tensor(value: Any) -> bool:
    if not isinstance(value, torch.Tensor):
        return False
    if _tdx_is_fake is not None:
        try:
            return bool(_tdx_is_fake(value))
        except Exception:
            pass
    return (
        isinstance(value, FakeTensor) or getattr(value, "fake_mode", None) is not None
    )


def is_meta_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and getattr(value, "is_meta", False)


def is_meta_or_fake_tensor(value: Any) -> bool:
    return is_meta_tensor(value) or is_fake_tensor(value)


def torch_no_compile(
    *,
    reason: str | None = None,
    recursive: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if _TORCH_COMPILE_DISABLE is None:

        def _identity(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn

        return _identity

    kwargs: dict[str, Any] = {}
    if reason is not None:
        kwargs["reason"] = reason
    if recursive is not None:
        kwargs["recursive"] = recursive

    attempts = [kwargs]
    if "reason" in kwargs:
        attempts.append({k: v for k, v in kwargs.items() if k != "recursive"})
    if "recursive" in kwargs:
        attempts.append({k: v for k, v in kwargs.items() if k != "reason"})
    attempts.append({})

    for opts in attempts:
        try:
            return _TORCH_COMPILE_DISABLE(**opts)
        except TypeError:
            continue

    def _identity(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    return _identity


def torch_safe_distributed(
    *, collectives: tuple[str, ...] = _COLLECTIVE_NAMES
) -> bool:
    if _TORCH_DYNAMO is None or not hasattr(_TORCH_DYNAMO, "disallow_in_graph"):
        return False
    try:
        import torch.distributed as dist
    except Exception:
        return False

    disallow = getattr(_TORCH_DYNAMO, "disallow_in_graph", None)
    if disallow is None:
        return False

    updated = False
    for name in collectives:
        fn = getattr(dist, name, None)
        if fn is None:
            continue
        with suppress(Exception):
            disallow(fn)
            updated = True
    return updated


def torch_disable_compile(
    target: Any,
    attr: str,
    *,
    reason: str | None = None,
    recursive: bool = True,
) -> bool:
    if target is None or not hasattr(target, attr):
        return False
    fn = getattr(target, attr)
    decorator = torch_no_compile(reason=reason, recursive=recursive)
    try:
        wrapped = decorator(fn)
    except Exception:
        return False
    setattr(target, attr, wrapped)
    return True


def torch_compile_safe(
    *,
    runtime_module: Any | None = None,
    layers_module: Any | None = None,
) -> None:
    if layers_module is None:
        with suppress(Exception):
            layers_module = importlib.import_module("stnet.model.layers")
    if layers_module is not None:
        torch_disable_compile(
            getattr(layers_module, "Normal", None),
            "commit_training_success",
            reason="history/BN sync – eager",
        )
        torch_disable_compile(
            getattr(layers_module, "StudentsT", None),
            "commit_training_success",
            reason="history – eager",
        )
        torch_disable_compile(
            getattr(layers_module, "History", None),
            "forward",
        )

    if runtime_module is None:
        with suppress(Exception):
            runtime_module = importlib.import_module("stnet.backend.runtime")
    if runtime_module is not None:
        torch_disable_compile(
            runtime_module,
            "push_metrics",
            reason="metric aggregation – eager",
        )
        torch_disable_compile(
            runtime_module,
            "_reduce_metrics",
            reason="distributed collectives – eager",
        )
