from __future__ import annotations

import contextlib
import importlib
import inspect
import warnings
from collections.abc import Mapping, Callable
from typing import Any, Iterator, TypeVar

import torch


_T = TypeVar("_T")

_tdx_is_fake = None

FakeTensor = ()
TensorDictBase = ()


def _optional_attr(
    module: str,
    attr: str,
    default: _T,
    *args: Any,
    predicate: Callable[[Any], bool] | None = None,
) -> Any | _T:
    try:
        if importlib.util.find_spec(module) is None:
            return default
    except Exception:
        return default
    try:
        mod = importlib.import_module(module)
    except Exception:
        return default
    try:
        val = getattr(mod, attr)
    except Exception:
        return default
    if predicate is not None and not predicate(val):
        return default
    return val


def _call_from_buffer(
    fn: Any,
    buffer: Any,
    *args: Any,
    dtype: torch.dtype,
    count: int = -1,
    offset: int = 0,
    requires_grad: bool = False,
) -> torch.Tensor:
    del args
    kw = {"buffer": buffer, "dtype": dtype, "count": count, "offset": offset}
    try:
        return fn(**kw, requires_grad=requires_grad)
    except TypeError:
        return fn(**kw)


def _to_local(t: torch.Tensor) -> torch.Tensor:
    try:
        return t.to_local() if hasattr(t, "to_local") else t
    except Exception:
        return t


def to_torch_tensor(obj: Any) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    for attr in ("to_torch_tensor", "to_torch", "to_tensor", "as_tensor"):
        try:
            fn = getattr(obj, attr, None)
        except Exception:
            fn = None
        if callable(fn):
            try:
                t = fn()
            except TypeError:
                continue
            except Exception:
                continue
            if isinstance(t, torch.Tensor):
                return t
            try:
                return torch.as_tensor(t)
            except Exception:
                continue
    return torch.as_tensor(obj)


def is_fake_tensor(value: Any) -> bool:
    if not isinstance(value, torch.Tensor):
        return False
    if _tdx_is_fake and (res := _tdx_is_fake(value)):
        return bool(res)
    return isinstance(value, FakeTensor)


def is_meta_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and getattr(value, "is_meta", False)


def is_meta_or_fake_tensor(value: Any) -> bool:
    return is_meta_tensor(value) or is_fake_tensor(value)


def coerce_tensor(
    value: object,
    *args: Any,
    materialize_meta: bool = True,
    make_contiguous: bool = True,
) -> object:
    if isinstance(value, torch.Tensor):
        t = value.to_local() if hasattr(value, "to_local") else value
        if materialize_meta and is_meta_or_fake_tensor(t):
            t = torch.zeros(t.shape, dtype=t.dtype, device="cpu")
        t = t.detach()
        if t.device.type != "cpu":
            t = t.to(device="cpu")
        if make_contiguous and not t.is_contiguous():
            t = t.contiguous()
        return t
    if isinstance(value, (list, tuple)):
        out = [
            coerce_tensor(
                v,
                materialize_meta=materialize_meta,
                make_contiguous=make_contiguous,
            )
            for v in value
        ]
        return (
            type(value)(*out)
            if hasattr(value, "_fields")
            else type(value)(out)
        )
    if isinstance(value, Mapping):
        return type(value)(
            (
                k,
                coerce_tensor(
                    v,
                    materialize_meta=materialize_meta,
                    make_contiguous=make_contiguous,
                ),
            )
            for k, v in value.items()
        )
    return value


def extract_tensor(out: object) -> torch.Tensor:
    def _to_plain(t: torch.Tensor) -> torch.Tensor:
        try:
            if hasattr(t, "to_local"):
                tl = t.to_local()
                if isinstance(tl, torch.Tensor):
                    t = tl
        except Exception:
            pass
        try:
            from torch._subclasses.functional_tensor import (
                disable_functional_mode,
                mb_unwrap_functional_tensor,
            )

            with disable_functional_mode():
                u = mb_unwrap_functional_tensor(t)
                if isinstance(u, torch.Tensor):
                    t = u
        except Exception:
            pass
        return t

    if isinstance(out, TensorDictBase):
        y = out.get("pred", None)
        if not isinstance(y, torch.Tensor):
            y = next(
                (v for v in out.values() if isinstance(v, torch.Tensor)), None
            )
        if isinstance(y, torch.Tensor):
            return _to_plain(y)
        raise RuntimeError("TensorDict output missing tensors")
    if isinstance(out, torch.Tensor):
        return _to_plain(out)
    if isinstance(out, (tuple, list)) and len(out) > 0:
        if isinstance(out[0], torch.Tensor):
            return _to_plain(out[0])
        y = next((v for v in out if isinstance(v, torch.Tensor)), None)
        if isinstance(y, torch.Tensor):
            return _to_plain(y)
        raise RuntimeError("Sequence output missing tensors")
    raise RuntimeError(f"Unsupported output type: {type(out)}")


def to_tensor_like(x: Any, ref: torch.Tensor) -> torch.Tensor:
    return (
        x.to(device=ref.device, dtype=ref.dtype)
        if torch.is_tensor(x)
        else torch.tensor(x, device=ref.device, dtype=ref.dtype)
    )


@contextlib.contextmanager
def from_buffer(
    *args: Any, coerce_requires_grad: bool = True
) -> Iterator[None]:
    if not hasattr(torch, "frombuffer"):
        yield
        return
    _original = torch.frombuffer

    def _patched(
        buffer: Any,
        dtype: torch.dtype,
        count: int = -1,
        offset: int = 0,
        requires_grad: bool = False,
    ):
        if coerce_requires_grad:
            requires_grad = False
        try:
            mv = memoryview(buffer)
            nbytes = int(getattr(mv, "nbytes", len(mv)))
            off = max(0, int(offset))
            if int(count) == 0:
                return torch.zeros((0,), dtype=dtype)
            if nbytes <= off:
                n = (
                    int(count)
                    if isinstance(count, int) and int(count) > 0
                    else 0
                )
                return torch.zeros((n,), dtype=dtype)
            readonly = bool(getattr(mv, "readonly", False))
        except Exception:
            readonly = False
        if readonly:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=r".*buffer is not writable.*"
                )
                return _call_from_buffer(
                    _original,
                    buffer,
                    dtype=dtype,
                    count=count,
                    offset=offset,
                    requires_grad=requires_grad,
                )
        return _call_from_buffer(
            _original,
            buffer,
            dtype=dtype,
            count=count,
            offset=offset,
            requires_grad=requires_grad,
        )

    setattr(torch, "frombuffer", _patched)
    try:
        yield
    finally:
        setattr(torch, "frombuffer", _original)


def symint_safe_expand(
    t: torch.Tensor,
    target_shape: tuple[object, ...] | list[object] | torch.Size,
) -> torch.Tensor:
    target = tuple(target_shape)
    if tuple(t.shape) == target:
        return t

    src = tuple(t.shape)
    if len(target) < len(src):
        return t.expand(target)

    src_aligned = (1,) * (len(target) - len(src)) + src

    sizes: list[object] = []
    for s_dim, t_dim in zip(src_aligned, target):
        sizes.append(-1 if s_dim == t_dim else t_dim)

    return t.expand(tuple(sizes))


def symint_safe_expand_as(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return symint_safe_expand(t, ref.shape)


_tdx_is_fake = _optional_attr(
    "torchdistx.fake", "is_fake", None, predicate=callable
)
FakeTensor = _optional_attr(
    "torch._subclasses.fake_tensor", "FakeTensor", (), predicate=inspect.isclass
)
TensorDictBase = _optional_attr(
    "tensordict", "TensorDictBase", (), predicate=inspect.isclass
)
