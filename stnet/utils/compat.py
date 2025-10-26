# -*- coding: utf-8 -*-
from __future__ import annotations

import numbers
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
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
_ARROW_COMPAT: ArrowCompat | None = None


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
        if hasattr(self.nn_module, "RMSNorm"):
            return
        torch_mod = self.module
        nn_mod = self.nn_module

        class _RMSNorm(nn_mod.Module):
            __constants__ = ("normalized_shape", "eps", "elementwise_affine")

            def __init__(
                self,
                normalized_shape: Any,
                eps: float | None = None,
                elementwise_affine: bool = True,
                device: Any = None,
                dtype: Any = None,
            ) -> None:
                super().__init__()
                if isinstance(normalized_shape, numbers.Integral):
                    normalized_shape = (int(normalized_shape),)
                else:
                    normalized_shape = tuple(int(dim) for dim in normalized_shape)
                if not normalized_shape:
                    raise ValueError("normalized_shape must be non-empty")

                factory_kwargs = {}
                if device is not None:
                    factory_kwargs["device"] = device
                if dtype is not None:
                    factory_kwargs["dtype"] = dtype

                self.normalized_shape = normalized_shape
                self.eps = float(eps) if eps is not None else None
                self.elementwise_affine = bool(elementwise_affine)
                if self.elementwise_affine:
                    self.weight = nn_mod.Parameter(
                        torch_mod.empty(self.normalized_shape, **factory_kwargs)
                    )
                else:
                    self.register_parameter("weight", None)
                self.reset_parameters()

            def reset_parameters(self) -> None:
                if self.elementwise_affine and self.weight is not None:
                    with torch_mod.no_grad():
                        self.weight.fill_(1.0)

            def forward(
                self,
                input: Any | None = None,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                if input is None and args:
                    input = args[0]
                    args = args[1:]
                if input is None:
                    for key in ("input", "hidden_states", "x"):
                        if key in kwargs:
                            input = kwargs.pop(key)
                            break
                if input is None:
                    raise TypeError("RMSNorm.forward() missing required argument 'input'")

                _ = args, kwargs

                x = input
                if x.dim() < len(self.normalized_shape):
                    raise RuntimeError(
                        "input.dim() must be >= len(normalized_shape) but got {} < {}".format(
                            x.dim(),
                            len(self.normalized_shape),
                        )
                    )
                dims = tuple(range(-len(self.normalized_shape), 0))
                if dims and any(
                    x.shape[dim] != expected
                    for dim, expected in zip(range(-len(self.normalized_shape), 0), self.normalized_shape)
                ):
                    raise RuntimeError(
                        "Given normalized_shape={}, expected input with shape of the form "
                        "(*, {}) but got {}".format(
                            self.normalized_shape,
                            self.normalized_shape,
                            tuple(x.shape),
                        )
                    )

                eps = self.eps
                if eps is None:
                    eps = torch_mod.finfo(x.dtype).eps

                mean_square = x.pow(2).mean(dim=dims, keepdim=True)
                inv_rms = (mean_square + eps).rsqrt()
                output = x * inv_rms

                if self.elementwise_affine and self.weight is not None:
                    view_shape = (1,) * (x.dim() - len(self.normalized_shape)) + self.normalized_shape
                    output = output * self.weight.view(view_shape)

                return output

        setattr(self.nn_module, "RMSNorm", _RMSNorm)

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


class ArrowCompat:
    def __init__(self, module: Any, flight: Any) -> None:
        self.module = module
        self.flight = flight

    def to_numpy(self, array: Any, *args: Any, zero_copy_only: bool = True, **kwargs: Any) -> Any:
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
        import pyarrow as pa
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
