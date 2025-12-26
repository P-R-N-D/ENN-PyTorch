# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import re
import threading
from contextlib import contextmanager, suppress
from functools import partial
from typing import Any, Callable, Iterator, NoReturn

import torch
from torch import nn

# -----------------------------------------------------------------------------
# Optional imports / version-dependent APIs
# -----------------------------------------------------------------------------

try:
    from torchdistx.fake import is_fake as _tdx_is_fake
except Exception:
    _tdx_is_fake = None

try:
    from torch._subclasses.fake_tensor import FakeTensor  # type: ignore
except Exception:
    FakeTensor = tuple()  # empty tuple-of-types works with isinstance()

try:
    from torch import compiler as _TORCH_COMPILER  # type: ignore
except Exception:
    _TORCH_COMPILER = None

try:
    import torch._dynamo as _TORCH_DYNAMO  # type: ignore
except Exception:
    _TORCH_DYNAMO = None


def _resolve_compile_disable() -> Any | None:
    if _TORCH_COMPILER is not None:
        fn = getattr(_TORCH_COMPILER, "disable", None)
        if callable(fn):
            return fn
    if _TORCH_DYNAMO is not None:
        fn = getattr(_TORCH_DYNAMO, "disable", None)
        if callable(fn):
            return fn
    return None


_TORCH_COMPILE_DISABLE = _resolve_compile_disable()

_COLLECTIVE_NAMES: tuple[str, ...] = (
    "all_gather",
    "all_gather_into_tensor",
    "all_reduce",
    "reduce_scatter_tensor",
    "broadcast",
    "barrier",
)

RMSNorm = getattr(nn, "RMSNorm", None)

StochasticDepth = getattr(nn, "StochasticDepth", None)
if StochasticDepth is None:

    class StochasticDepth(nn.Module):
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


# SDP backend + sdpa_kernel (ensure both names exist)
try:
    from torch.nn.attention import SDPBackend as SDPBackend  # type: ignore
    from torch.nn.attention import sdpa_kernel as sdpa_kernel  # type: ignore
except Exception:

    class SDPBackend:
        MATH = object()
        FLASH_ATTENTION = object()
        EFFICIENT_ATTENTION = object()
        CUDNN_ATTENTION = object()

    @contextmanager
    def sdpa_kernel(*backends: Any) -> Iterator[None]:
        _ = backends
        yield


# -----------------------------------------------------------------------------
# graph_break (cache resolved callable; avoid repeated imports/lookups)
# -----------------------------------------------------------------------------

_GRAPH_BREAK_FN: Callable[[], None] | None = None
_GRAPH_BREAK_LOCK = threading.Lock()


def _resolve_graph_break_fn() -> Callable[[], None] | None:
    # Prefer inductor.graph_break if present; fallback to dynamo.graph_break
    try:
        import torch._inductor as _inductor  # type: ignore

        gb = getattr(_inductor, "graph_break", None)
        if callable(gb):
            return gb
    except Exception:
        pass

    if _TORCH_DYNAMO is not None:
        gb = getattr(_TORCH_DYNAMO, "graph_break", None)
        if callable(gb):
            return gb
    return None


def graph_break() -> None:
    """Break torch.compile graphs only when tracing (safe no-op otherwise)."""
    dyn = _TORCH_DYNAMO
    if dyn is None:
        return

    try:
        if not dyn.is_compiling():
            return
    except Exception:
        return

    global _GRAPH_BREAK_FN
    fn = _GRAPH_BREAK_FN
    if fn is None:
        with _GRAPH_BREAK_LOCK:
            if _GRAPH_BREAK_FN is None:
                _GRAPH_BREAK_FN = _resolve_graph_break_fn()
            fn = _GRAPH_BREAK_FN

    if fn is None:
        return
    with suppress(Exception):
        fn()


# -----------------------------------------------------------------------------
# torchdata check
# -----------------------------------------------------------------------------

MIN_TORCHDATA_VERSION = "0.11.0"
_VERSION_RE = re.compile(r"\d+")


def _parse_version(v: str) -> tuple[int, int, int]:
    parts = _VERSION_RE.findall(str(v))
    nums = [int(x) for x in parts[:3]]
    while len(nums) < 3:
        nums.append(0)
    return (nums[0], nums[1], nums[2])


def ensure_torchdata(
    *, min_version: str = MIN_TORCHDATA_VERSION, err: Exception | None = None, context: str = "stnet"
) -> NoReturn:
    """Fail-fast if torchdata is missing/too old, or torchdata.nodes API is unavailable.

    Intended usage: call inside an `except Exception as _e:` that failed to import torchdata.nodes APIs.
    This function ALWAYS raises ImportError.
    """
    try:
        import torchdata  # type: ignore

        v = getattr(torchdata, "__version__", "") or ""
        if not v:
            try:
                from importlib.metadata import version as _md_version

                v = _md_version("torchdata")
            except Exception:
                v = "0.0.0"

        if _parse_version(v) < _parse_version(min_version):
            raise ImportError(
                f"torchdata>={min_version} required (found {v}). "
                f"Upgrade: pip install -U 'torchdata>={min_version}'"
            )
    except Exception as e:
        raise ImportError(
            f"{context}: torchdata>={min_version} is required. "
            f"Install/upgrade: pip install -U 'torchdata>={min_version}'"
        ) from (err or e)

    raise ImportError(
        f"{context}: torchdata.nodes APIs required (torchdata>={min_version}). "
        f"Install/upgrade: pip install -U 'torchdata>={min_version}'"
    ) from err


# -----------------------------------------------------------------------------
# Compat implementations for missing torch APIs
# -----------------------------------------------------------------------------

def _fmin_impl(torch_mod: Any, a: Any, b: Any) -> Any:
    a, b = torch_mod.broadcast_tensors(a, b)
    a_nan = torch_mod.isnan(a)
    b_nan = torch_mod.isnan(b)
    return torch_mod.where(
        a_nan & ~b_nan,
        b,
        torch_mod.where(b_nan & ~a_nan, a, torch_mod.minimum(a, b)),
    )


def _nanmin_impl(torch_mod: Any, x: Any, dim: int | None = None, keepdim: bool = False) -> Any:
    # Avoid inf/nan fill on non-floating dtypes (int/bool), which can error.
    if isinstance(x, torch.Tensor) and not torch_mod.is_floating_point(x):
        return x.min() if dim is None else x.min(dim=dim, keepdim=keepdim)

    mask = torch_mod.isfinite(x)
    xp = torch_mod.where(mask, x, torch_mod.full_like(x, float("inf")))

    if dim is None:
        values = xp.min()
        any_valid = mask.any()
        return torch_mod.where(any_valid, values, torch_mod.full_like(values, float("nan")))

    values, indices = xp.min(dim=dim, keepdim=keepdim)
    any_valid = mask.any(dim=dim, keepdim=keepdim)
    values = torch_mod.where(any_valid, values, torch_mod.full_like(values, float("nan")))
    indices = torch_mod.where(any_valid, indices, torch_mod.zeros_like(indices))
    return (values, indices)


def _nanmax_impl(torch_mod: Any, x: Any, dim: int | None = None, keepdim: bool = False) -> Any:
    if isinstance(x, torch.Tensor) and not torch_mod.is_floating_point(x):
        return x.max() if dim is None else x.max(dim=dim, keepdim=keepdim)

    mask = torch_mod.isfinite(x)
    xp = torch_mod.where(mask, x, torch_mod.full_like(x, float("-inf")))

    if dim is None:
        values = xp.max()
        any_valid = mask.any()
        return torch_mod.where(any_valid, values, torch_mod.full_like(values, float("nan")))

    values, indices = xp.max(dim=dim, keepdim=keepdim)
    any_valid = mask.any(dim=dim, keepdim=keepdim)
    values = torch_mod.where(any_valid, values, torch_mod.full_like(values, float("nan")))
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
    # Keep signature-flexibility but try to forward common kwargs (e.g., out=)
    _ = args

    if dtype is not None and not isinstance(dtype, torch.dtype):
        with suppress(Exception):
            dtype = getattr(torch, str(dtype).split(".")[-1], None)

    x_cast = x.to(dtype) if dtype is not None else x

    # If not floating, NaN/inf isn't meaningful; behave like normal sum.
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


# -----------------------------------------------------------------------------
# Patching and compile-safety helpers (thread-safe & idempotent)
# -----------------------------------------------------------------------------

_PATCH_LOCK = threading.RLock()
_SAFE_DIST_LOCK = threading.Lock()
_SAFE_DIST_PATCHED: set[str] = set()

_TORCH_COMPAT: TorchCompat | None = None
_NO_COMPILE_SENTINEL = "__stnet_no_compile_wrapped__"


class TorchCompat:
    def __init__(self, module: Any | None = None, nn_module: Any | None = None) -> None:
        self.module = module if module is not None else torch
        self.nn_module = nn_module if nn_module is not None else getattr(self.module, "nn", nn)

    def apply(self) -> None:
        with _PATCH_LOCK:
            self._patch_rmsnorm()
            self._patch_fmin()
            self._patch_nanmin()
            self._patch_nanmax()
            self._patch_nansum()
            torch_safe_distributed()

    def _patch_rmsnorm(self) -> None:
        global RMSNorm
        if hasattr(self.nn_module, "RMSNorm"):
            RMSNorm = getattr(self.nn_module, "RMSNorm", None)
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


def patch_torch(module: Any | None = None, nn_module: Any | None = None) -> TorchCompat:
    global _TORCH_COMPAT
    with _PATCH_LOCK:
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
        with suppress(Exception):
            return bool(_tdx_is_fake(value))
    return isinstance(value, FakeTensor) or getattr(value, "fake_mode", None) is not None


def is_meta_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and getattr(value, "is_meta", False)


def is_meta_or_fake_tensor(value: Any) -> bool:
    return is_meta_tensor(value) or is_fake_tensor(value)


def torch_no_compile(*, reason: str | None = None, recursive: bool = True) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if _TORCH_COMPILE_DISABLE is None:
        def _identity(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn
        return _identity

    kwargs: dict[str, Any] = {}
    if reason is not None:
        kwargs["reason"] = reason
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


def torch_safe_distributed(*, collectives: tuple[str, ...] = _COLLECTIVE_NAMES) -> bool:
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
    with _SAFE_DIST_LOCK:
        for name in collectives:
            if name in _SAFE_DIST_PATCHED:
                continue
            fn = getattr(dist, name, None)
            if fn is None:
                continue
            with suppress(Exception):
                disallow(fn)
                _SAFE_DIST_PATCHED.add(name)
                updated = True
    return updated


def torch_disable_compile(target: Any, attr: str, *, reason: str | None = None, recursive: bool = True) -> bool:
    if target is None or not hasattr(target, attr):
        return False

    fn = getattr(target, attr)
    if getattr(fn, _NO_COMPILE_SENTINEL, False):
        return True

    decorator = torch_no_compile(reason=reason, recursive=recursive)
    try:
        wrapped = decorator(fn)
    except Exception:
        return False

    with suppress(Exception):
        setattr(wrapped, _NO_COMPILE_SENTINEL, True)

    try:
        setattr(target, attr, wrapped)
    except Exception:
        return False
    return True


def torch_compile_safe(*, runtime_module: Any | None = None, layers_module: Any | None = None) -> None:
    if layers_module is None:
        with suppress(Exception):
            layers_module = importlib.import_module("stnet.model.nn")

    if layers_module is not None:
        torch_disable_compile(getattr(layers_module, "Normal", None), "commit_training_success",
                             reason="history/BN sync – eager")
        torch_disable_compile(getattr(layers_module, "StudentsT", None), "commit_training_success",
                             reason="history – eager")
        torch_disable_compile(getattr(layers_module, "History", None), "forward")

    if runtime_module is None:
        with suppress(Exception):
            runtime_module = importlib.import_module("stnet.run.elastic")

    if runtime_module is not None:
        # NOTE: Runtime metric aggregation helpers used to live in stnet.core.runtime.
        # They were removed/merged; keep the hook here to avoid stale attribute references.
        pass
