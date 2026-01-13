from __future__ import annotations

import contextlib
import importlib
import inspect
from collections.abc import Mapping
from typing import Any, Iterator

import torch


def _safe_find_spec(name: str) -> object | None:
    try:
        return importlib.util.find_spec(name)
    except (ModuleNotFoundError, ImportError):
        return None
    except Exception:
        return None


spec = _safe_find_spec("torchdistx.fake")
if spec is not None:
    try:
        torchdistx_fake = importlib.import_module("torchdistx.fake")
        _tdx_is_fake = getattr(torchdistx_fake, "is_fake", None)
    except Exception:
        _tdx_is_fake = None
else:
    _tdx_is_fake = None

spec = _safe_find_spec("torch._subclasses.fake_tensor")
if spec is not None:
    try:
        from torch._subclasses.fake_tensor import FakeTensor
    except Exception:
        FakeTensor = tuple()
else:
    FakeTensor = tuple()

spec = _safe_find_spec("tensordict")
if spec is not None:
    try:
        from tensordict import TensorDictBase
    except Exception:
        TensorDictBase = ()
else:
    TensorDictBase = ()


def _call_from_buffer(
    fn: Any,
    buffer: Any,
    *args: Any,
    dtype: torch.dtype,
    count: int = -1,
    offset: int = 0,
    requires_grad: bool = False,
) -> torch.Tensor:
    s = inspect.signature(fn)
    args = {"buffer": buffer, "dtype": dtype, "count": count, "offset": offset}
    if "requires_grad" in s.parameters:
        args["requires_grad"] = requires_grad
    else:
        try:
            if requires_grad:
                args["requires_grad"] = requires_grad
        except Exception:
            pass
    return fn(**args)


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
    *,
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
            coerce_tensor(v, materialize_meta=materialize_meta, make_contiguous=make_contiguous)
            for v in value
        ]
        return type(value)(*out) if hasattr(value, "_fields") else type(value)(out)
    if isinstance(value, Mapping):
        return type(value)(
            (k, coerce_tensor(v, materialize_meta=materialize_meta, make_contiguous=make_contiguous))
            for k, v in value.items()
        )
    return value


def extract_tensor(out: object) -> torch.Tensor:
    if isinstance(out, TensorDictBase):
        y = out.get("pred", None)
        if not isinstance(y, torch.Tensor):
            y = next((v for v in out.values() if isinstance(v, torch.Tensor)), None)
        if isinstance(y, torch.Tensor):
            return y
        raise RuntimeError("TensorDict output missing tensors")
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and len(out) > 0:
        if isinstance(out[0], torch.Tensor):
            return out[0]
        y = next((v for v in out if isinstance(v, torch.Tensor)), None)
        if isinstance(y, torch.Tensor):
            return y
        raise RuntimeError("Sequence output missing tensors")
    raise RuntimeError(f"Unsupported output type: {type(out)}")


def to_tensor_like(x: Any, ref: torch.Tensor) -> torch.Tensor:
    return (
        x.to(device=ref.device, dtype=ref.dtype)
        if torch.is_tensor(x)
        else torch.tensor(x, device=ref.device, dtype=ref.dtype)
    )


@contextlib.contextmanager
def from_buffer(*, coerce_requires_grad: bool = True) -> Iterator[None]:
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
