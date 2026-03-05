# -*- coding: utf-8 -*-
from __future__ import annotations

# =============================================================================
# 1. Standard Library Imports
# =============================================================================
import contextlib
import importlib
import inspect
import warnings
from collections.abc import Callable, Iterator, Mapping
from functools import partial
from typing import Any, TypeVar

# =============================================================================
# 2. Third-Party Imports
# =============================================================================
import torch

# =============================================================================
# 3. Local Imports
# =============================================================================
from .system import is_pin_supported


# =============================================================================
# Globals & Constants
# =============================================================================
_T = TypeVar("_T")

try:
    import torch._dynamo as _dynamo
except ImportError:
    _dynamo = None


# =============================================================================
# Internal Helpers
# =============================================================================
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


def _dynamo_is_compiling() -> bool:
    try:
        return bool(_dynamo is not None and _dynamo.is_compiling())
    except Exception:
        return False


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


# =============================================================================
# Core Tensor Operations
# =============================================================================
def to_torch_tensor(obj: Any) -> torch.Tensor:
    match obj:
        case torch.Tensor():
            return obj
        case _:
            for attr in ("to_torch_tensor", "to_torch", "to_tensor", "as_tensor"):
                with contextlib.suppress(Exception):
                    fn = getattr(obj, attr, None)
                    if callable(fn):
                        t = fn()
                        if isinstance(t, torch.Tensor):
                            return t
                        return torch.as_tensor(t)
            return torch.as_tensor(obj)


def is_fake_tensor(value: Any) -> bool:
    match value:
        case torch.Tensor():
            if callable(_tdx_is_fake) and (res := _tdx_is_fake(value)):
                return bool(res)
            return isinstance(value, FakeTensor)
        case _:
            return False


def is_meta_tensor(value: Any) -> bool:
    match value:
        case torch.Tensor():
            return bool(getattr(value, "is_meta", False))
        case _:
            return False


def is_meta_or_fake_tensor(value: Any) -> bool:
    return is_meta_tensor(value) or is_fake_tensor(value)


def validate_no_meta_tensors(module: object) -> None:
    hits: list[str] = []
    for name, param in getattr(module, "named_parameters", lambda **_: [])(recurse=True):
        if is_meta_or_fake_tensor(param):
            hits.append(f"param {name} shape={tuple(param.shape)}")
            
    for name, buffer in getattr(module, "named_buffers", lambda **_: [])(recurse=True):
        if is_meta_or_fake_tensor(buffer):
            hits.append(f"buffer {name} shape={tuple(buffer.shape)}")
            
    if hits:
        raise RuntimeError("Found meta tensors in model:\n" + "\n".join(hits))


def hook_meta_monitor(module: object, inputs: object, warn_only: object) -> None:
    try:
        iterator = iter(inputs)
    except TypeError:
        iterator = iter(())
        
    for arg in iterator:
        match arg:
            case torch.Tensor() if is_meta_or_fake_tensor(arg):
                message = f"[META] {module.__class__.__name__} got meta input"
                if warn_only:
                    warnings.warn(message, stacklevel=3)
                    return
                raise RuntimeError(message)
            case _:
                pass


def enable_meta_monitor(model: object) -> None:
    try:
        from .datatypes import env_first
    except ImportError:
        env_first = None

    mode = "off"
    if callable(env_first):
        mode = str(env_first(("ENN_META_MONITOR", "ENN_META_HOOK"), default="off") or "off")
        
    mode = mode.strip().lower()
    if mode in {"0", "", "false", "off"}:
        return
        
    warn_only = mode in {"warn", "warning"}
    
    try:
        mods = getattr(model, "modules", lambda: [])()
    except Exception:
        mods = ()
        
    for submodule in mods:
        try:
            submodule.register_forward_pre_hook(
                partial(hook_meta_monitor, warn_only=warn_only),
                with_kwargs=False,
            )
        except TypeError:
            submodule.register_forward_pre_hook(
                partial(hook_meta_monitor, warn_only=warn_only)
            )


def validate_no_fake_dtensor(root: object, *args: object, **kwargs: object) -> None:
    del args, kwargs
    try:
        import torch.nn as nn
    except ImportError:
        nn = None

    bad: list[str] = []
    for name, module in getattr(root, "named_modules", lambda: [])():
        if nn is not None and not isinstance(module, nn.LayerNorm):
            continue
        if nn is None and module.__class__.__name__ != "LayerNorm":
            continue
            
        for attr in ("weight", "bias"):
            tensor = getattr(module, attr, None)
            if tensor is None:
                continue
            if is_meta_or_fake_tensor(tensor):
                module_name = name or module.__class__.__name__
                bad.append(f"{module_name}.{attr}{tuple(tensor.shape)}")
                
    if bad:
        raise RuntimeError("LayerNorm parameters must be materialized as a real Tensor: " + ", ".join(bad))


def coerce_tensor(
    value: object,
    *args: Any,
    materialize_meta: bool = True,
    make_contiguous: bool = True,
) -> object:
    match value:
        case torch.Tensor():
            t = value.to_local() if hasattr(value, "to_local") else value
            if materialize_meta and is_meta_or_fake_tensor(t):
                t = torch.zeros(t.shape, dtype=t.dtype, device="cpu")
            t = t.detach()
            if t.device.type != "cpu":
                t = t.to(device="cpu")
            if make_contiguous and not t.is_contiguous():
                t = t.contiguous()
            return t
        case list() | tuple():
            out = [
                coerce_tensor(v, materialize_meta=materialize_meta, make_contiguous=make_contiguous)
                for v in value
            ]
            return type(value)(*out) if hasattr(value, "_fields") else type(value)(out)
        case Mapping():
            return type(value)(
                (k, coerce_tensor(v, materialize_meta=materialize_meta, make_contiguous=make_contiguous))
                for k, v in value.items()
            )
        case _:
            return value


def extract_tensor(out: object) -> torch.Tensor:
    def _to_plain(t: torch.Tensor) -> torch.Tensor:
        if _dynamo_is_compiling():
            return t
        with contextlib.suppress(Exception):
            if hasattr(t, "to_local"):
                tl = t.to_local()
                if isinstance(tl, torch.Tensor):
                    t = tl
                    
        fn_disable = getattr(sys.modules[__name__], "_disable_functional_mode", None)
        fn_unwrap = getattr(sys.modules[__name__], "_mb_unwrap_functional_tensor", None)
        
        if callable(fn_disable) and callable(fn_unwrap):
            with contextlib.suppress(Exception):
                with fn_disable():
                    u = fn_unwrap(t)
                    if isinstance(u, torch.Tensor):
                        return u
        return t

    match out:
        case _ if isinstance(out, TensorDictBase):
            y = out.get("pred", None)
            if not isinstance(y, torch.Tensor):
                y = next((v for v in out.values() if isinstance(v, torch.Tensor)), None)
            if isinstance(y, torch.Tensor):
                return _to_plain(y)
            raise RuntimeError("TensorDict output missing tensors")
        case torch.Tensor():
            return _to_plain(out)
        case tuple() | list() if len(out) > 0:
            if isinstance(out[0], torch.Tensor):
                return _to_plain(out[0])
            y = next((v for v in out if isinstance(v, torch.Tensor)), None)
            if isinstance(y, torch.Tensor):
                return _to_plain(y)
            raise RuntimeError("Sequence output missing tensors")
        case _:
            raise RuntimeError(f"Unsupported output type: {type(out)}")


def to_tensor_like(x: Any, ref: torch.Tensor) -> torch.Tensor:
    match x:
        case torch.Tensor():
            return x.to(device=ref.device, dtype=ref.dtype)
        case _:
            return torch.tensor(x, device=ref.device, dtype=ref.dtype)


@contextlib.contextmanager
def from_buffer(*args: Any, coerce_requires_grad: bool = True) -> Iterator[None]:
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
    ) -> torch.Tensor:
        if coerce_requires_grad:
            requires_grad = False
        try:
            mv = memoryview(buffer)
            nbytes = int(getattr(mv, "nbytes", len(mv)))
            off = max(0, int(offset))
            if int(count) == 0:
                return torch.zeros((0,), dtype=dtype)
            if nbytes <= off:
                n = int(count) if isinstance(count, int) and int(count) > 0 else 0
                return torch.zeros((n,), dtype=dtype)
            readonly = bool(getattr(mv, "readonly", False))
        except Exception:
            readonly = False
            
        if readonly:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r".*buffer is not writable.*")
                return _call_from_buffer(
                    _original, buffer, dtype=dtype, count=count, offset=offset, requires_grad=requires_grad
                )
                
        return _call_from_buffer(
            _original, buffer, dtype=dtype, count=count, offset=offset, requires_grad=requires_grad
        )

    setattr(torch, "frombuffer", _patched)
    try:
        yield
    finally:
        setattr(torch, "frombuffer", _original)


def symint_safe_expand(
    t: torch.Tensor, target_shape: tuple[object, ...] | list[object] | torch.Size
) -> torch.Tensor:
    target = tuple(target_shape)
    src = tuple(t.shape)
    if src == target:
        return t
    if len(target) < len(src):
        return t.expand(target)
        
    src_aligned = (1,) * (len(target) - len(src)) + src
    sizes: list[object] = [-1 if s_dim == t_dim else t_dim for s_dim, t_dim in zip(src_aligned, target)]
    return t.expand(tuple(sizes))


def symint_safe_expand_as(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return symint_safe_expand(t, ref.shape)


def to_device_recursive(obj: object, dev: object) -> object:
    match obj:
        case torch.Tensor():
            dev_type = getattr(dev, "type", None)
            non_blocking = bool(dev_type and is_pin_supported(str(dev_type)))
            try:
                return obj.to(device=dev, non_blocking=non_blocking)
            except TypeError:
                return obj.to(device=dev)
        case _ if isinstance(obj, TensorDictBase):
            return obj.to(device=dev)
        case Mapping():
            return {k: to_device_recursive(v, dev) for k, v in obj.items()}
        case list() | tuple():
            seq = [to_device_recursive(v, dev) for v in obj]
            return type(obj)(seq)
        case _:
            return obj


def touch_tensors(obj: object) -> None:
    match obj:
        case torch.Tensor():
            _ = obj.sum()
        case _ if isinstance(obj, TensorDictBase):
            for v in obj.values():
                touch_tensors(v)
        case Mapping():
            for v in obj.values():
                touch_tensors(v)
        case list() | tuple():
            for v in obj:
                touch_tensors(v)
        case _:
            pass


def compute_batch_bytes_per_sample(obj: object) -> tuple[int | None, int]:
    batch_dim: int | None = None
    bytes_per_sample = 0
    stack: list[object] = [obj]

    while stack:
        o = stack.pop()
        match o:
            case torch.Tensor():
                if o.numel() <= 0:
                    continue
                b = int(o.shape[0]) if o.ndim >= 1 else 1
                if batch_dim is None:
                    batch_dim = b
                    
                one = o[:1] if (o.ndim >= 1 and b > 0) else o.reshape(1, -1)
                bytes_per_sample += int(one.nelement()) * int(one.element_size())
            case _ if isinstance(o, TensorDictBase):
                stack.extend(list(o.values()))
            case Mapping():
                stack.extend(list(o.values()))
            case list() | tuple():
                stack.extend(list(o))
            case _:
                pass

    if bytes_per_sample <= 0:
        return (None, 0)
    return (batch_dim, bytes_per_sample)


# =============================================================================
# Deferred & Lazy Declarations
# =============================================================================
_disable_functional_mode = _optional_attr(
    "torch._subclasses.functional_tensor",
    "disable_functional_mode",
    None,
    predicate=callable,
)

_mb_unwrap_functional_tensor = _optional_attr(
    "torch._subclasses.functional_tensor",
    "mb_unwrap_functional_tensor",
    None,
    predicate=callable,
)

_tdx_is_fake = _optional_attr(
    "torchdistx.fake", "is_fake", None, predicate=callable
)

FakeTensor = _optional_attr(
    "torch._subclasses.fake_tensor",
    "FakeTensor",
    (),
    predicate=inspect.isclass,
)

TensorDictBase = _optional_attr(
    "tensordict", "TensorDictBase", (), predicate=inspect.isclass
)
