# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import importlib
import threading
from contextlib import AbstractContextManager, suppress
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn

from .casting import env_first, env_first_int
from .system import process_cpu_count

try:
    from torch import compiler as _TORCH_COMPILER  # type: ignore
except Exception:
    _TORCH_COMPILER = None

try:
    import torch._dynamo as _TORCH_DYNAMO  # type: ignore
except Exception:
    _TORCH_DYNAMO = None


# torch._inductor config is process-global; guard concurrent mutation.
_INDUCTOR_CONFIG_LOCK = threading.Lock()
_SAFE_DIST_LOCK = threading.Lock()
_SAFE_DIST_PATCHED: set[str] = set()

_COLLECTIVE_NAMES: tuple[str, ...] = (
    "all_gather",
    "all_gather_into_tensor",
    "all_reduce",
    "reduce_scatter_tensor",
    "broadcast",
    "barrier",
)

_NO_COMPILE_SENTINEL = "__stnet_no_compile_wrapped__"


def invalidate_model_introspection_caches(model: Optional[nn.Module]) -> None:
    """Invalidate cached runtime-introspection hints on `model`.

    We cache a few expensive checks (TorchScript/compile/AOT/TE probing).
    Any time we swap modules (TE, QAT wrappers) or quantize weights, these
    hints can become stale.
    """
    if not isinstance(model, nn.Module):
        return
    for attr in (
        "__stnet_cached_is_compiled_for_inference__",
        "__stnet_cached_is_aot_autograd_enabled__",
        "__stnet_cached_is_nvidia_te_available__",
    ):
        with contextlib.suppress(Exception):
            delattr(model, attr)


# Backward-compatible private alias.
_invalidate_model_introspection_caches = invalidate_model_introspection_caches


def _is_compiled_for_inference(model: torch.nn.Module) -> bool:
    cached = getattr(model, "__stnet_cached_is_compiled_for_inference__", None)
    if isinstance(cached, bool):
        return cached

    compile_attrs = (
        "_is_compiled_for_inference",
        "__is_compiled_for_inference__",
        "__compiled_for_serving__",
        "__serving_compiled__",
        "_is_serialized_for_serving",
    )
    if any(bool(getattr(model, attr, False)) for attr in compile_attrs):
        try:
            setattr(model, "__stnet_cached_is_compiled_for_inference__", True)
        except Exception:
            pass
        return True

    jit = getattr(torch, "jit", None)
    script_like_types: List[type] = []
    if jit is not None:
        for name in ("ScriptModule", "RecursiveScriptModule", "TopLevelTracedModule"):
            typ = getattr(jit, name, None)
            if isinstance(typ, type):
                script_like_types.append(typ)
        for mod_name in ("_script", "_trace"):
            submod = getattr(jit, mod_name, None)
            if submod is None:
                continue
            for name in ("RecursiveScriptModule", "TopLevelTracedModule"):
                typ = getattr(submod, name, None)
                if isinstance(typ, type):
                    script_like_types.append(typ)

    if any(isinstance(model, typ) for typ in script_like_types):
        try:
            setattr(model, "__stnet_cached_is_compiled_for_inference__", True)
        except Exception:
            pass
        return True

    try:
        modules = tuple(model.modules())
    except (RuntimeError, AttributeError, TypeError):
        modules = ()

    for module in modules:
        if module is model:
            continue
        if any(bool(getattr(module, attr, False)) for attr in compile_attrs):
            return True
        if any(isinstance(module, typ) for typ in script_like_types):
            return True
    try:
        setattr(model, "__stnet_cached_is_compiled_for_inference__", False)
    except Exception:
        pass
    return False


# Public alias (callers should prefer the non-underscored name).
is_compiled_for_inference = _is_compiled_for_inference


def _is_aot_autograd_enabled(model: torch.nn.Module) -> bool:
    cached = getattr(model, "__stnet_cached_is_aot_autograd_enabled__", None)
    if isinstance(cached, bool):
        return cached

    indicator_attrs = (
        "_aot_autograd_graph",
        "_aot_autograd_cache",
        "_aot_compiled_autograd",
        "_aot_autograd_traced_module",
        "__aot_autograd__",
        "__compiled_with_aot_autograd__",
    )
    if any(getattr(model, attr, None) for attr in indicator_attrs):
        try:
            setattr(model, "__stnet_cached_is_aot_autograd_enabled__", True)
        except Exception:
            pass
        return True

    try:
        modules = tuple(model.modules())
    except (RuntimeError, AttributeError, TypeError):
        modules = ()

    for module in modules:
        if module is model:
            continue
        if any(getattr(module, attr, None) for attr in indicator_attrs):
            return True
        class_name = module.__class__.__name__
        module_name = getattr(module.__class__, "__module__", "")
        if "AOTAutograd" in class_name or "aot_autograd" in module_name:
            return True
    try:
        setattr(model, "__stnet_cached_is_aot_autograd_enabled__", False)
    except Exception:
        pass
    return False


# Public alias.
is_aot_autograd_enabled = _is_aot_autograd_enabled


def is_nvidia_te_available(model: torch.nn.Module) -> bool:
    cached = getattr(model, "__stnet_cached_is_nvidia_te_available__", None)
    if isinstance(cached, bool):
        return cached

    te_flags = (
        getattr(model, "__fp8_inference_te__", False),
        getattr(model, "__fp8_training_te__", False),
        getattr(model, "__te_fp8_default__", False),
    )
    if any(te_flags):
        try:
            setattr(model, "__stnet_cached_is_nvidia_te_available__", True)
        except Exception:
            pass
        return True

    for module in model.modules():
        mod_name = getattr(module.__class__, "__module__", "")
        if isinstance(mod_name, str) and mod_name.startswith("transformer_engine"):
            try:
                setattr(model, "__stnet_cached_is_nvidia_te_available__", True)
            except Exception:
                pass
            return True
    try:
        setattr(model, "__stnet_cached_is_nvidia_te_available__", False)
    except Exception:
        pass
    return False


def inference_mode(model: torch.nn.Module) -> AbstractContextManager[None]:
    """Return the right inference context for `model`.

    - If the model is compiled / AOT / TransformerEngine, prefer `no_grad()`.
      (inference_mode() can break some fused/compiled paths)
    - Otherwise, use `inference_mode()` for extra speed.
    """
    if (
        is_nvidia_te_available(model)
        or _is_compiled_for_inference(model)
        or _is_aot_autograd_enabled(model)
    ):
        return torch.no_grad()
    return torch.inference_mode()


def _is_for_cuda(module: nn.Module) -> bool:
    """Best-effort check for whether module is intended to run on CUDA."""
    try:
        p0 = next(module.parameters(), None)
        if isinstance(p0, torch.Tensor):
            return p0.device.type == "cuda"
    except Exception:
        pass
    try:
        b0 = next(module.buffers(), None)
        if isinstance(b0, torch.Tensor):
            return b0.device.type == "cuda"
    except Exception:
        pass
    # Fall back to runtime capability.
    return bool(torch.cuda.is_available())


def compile(
    module: nn.Module,
    *args: Any,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    fullgraph: Optional[bool] = None,
    dynamic: Optional[bool] = None,
    options: Optional[Dict[str, Any]] = None,
    disable: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """Best-effort wrapper around `torch.compile`.

    This is intentionally conservative:
    - No-op when torch.compile is unavailable.
    - No-op when mode is empty/disabled.
    - Applies a small amount of per-process inductor config guarding to avoid
      oversubscription when multiple local ranks compile.

    NOTE: signature matches the historical `Gradient.compile` API.
    """
    normalized_mode = ""
    if mode is not None:
        normalized_mode = str(mode).strip().lower()
    if disable or normalized_mode in {"", "disabled", "none"}:
        return module
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return module

    normalized_alias = normalized_mode.replace("_", "-").replace(" ", "-")
    if "-" in normalized_alias:
        normalized_alias = "-".join(part for part in normalized_alias.split("-") if part)

    recognized_modes = {
        "default",
        "aot_eager",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    }

    canonical_mode = normalized_alias or normalized_mode

    # TorchInductor compilation can be very CPU hungry. When running multiple local
    # ranks (DDP/FSDP), each process compiling with a large thread pool can easily
    # oversubscribe the node and slow down *both* compilation and the input pipeline.
    #
    # Keep this conservative and fully overridable:
    #   - STNET_INDUCTOR_COMPILE_THREADS / STNET_COMPILE_THREADS
    #   - TORCHINDUCTOR_COMPILE_THREADS
    try:
        from torch._inductor import config as _inductor_config  # type: ignore
    except Exception:
        _inductor_config = None

    if _inductor_config is not None:
        with _INDUCTOR_CONFIG_LOCK:
            try:
                if getattr(_inductor_config, "compile_threads", None) is not None:
                    override = env_first(
                        ("STNET_INDUCTOR_COMPILE_THREADS", "STNET_COMPILE_THREADS"),
                        None,
                    )
                    if override is None:
                        override = env_first(("TORCHINDUCTOR_COMPILE_THREADS",), None)

                    if override is not None:
                        _inductor_config.compile_threads = max(1, int(override))
                    else:
                        local_world = env_first_int(
                            (
                                "STNET_LOCAL_WORLD_SIZE",
                                "LOCAL_WORLD_SIZE",
                                "SLURM_NTASKS_PER_NODE",
                            ),
                            1,
                        )
                        if int(local_world) > 1:
                            # Pick a small per-rank compile thread count. Too many threads
                            # across multiple ranks oversubscribes the node and hurts both
                            # compilation latency and the data pipeline.
                            cpu_count = int(process_cpu_count() or 1)
                            per_rank = max(1, int(cpu_count) // max(1, int(local_world)))
                            _inductor_config.compile_threads = max(1, min(4, int(per_rank) // 2))
            except Exception:
                pass

    if canonical_mode == "max-autotune" and not _is_for_cuda(module):
        canonical_mode = "max-autotune-no-cudagraphs"

    if canonical_mode == "max-autotune-no-cudagraphs":
        try:
            _opt = dict(options or {})
            _opt.setdefault("triton.cudagraphs", False)
            options = _opt
        except Exception:
            pass

    if canonical_mode in {"max-autotune", "max-autotune-no-cudagraphs"}:
        try:
            from torch._inductor import config as _inductor_config
        except Exception:
            _inductor_config = None

        if _inductor_config is not None:
            with _INDUCTOR_CONFIG_LOCK:
                try:
                    _inductor_config.autotune_in_subproc = True
                except Exception:
                    pass

                try:
                    _inductor_config.autotune_local_cache = True
                except Exception:
                    pass
                try:
                    _inductor_config.autotune_remote_cache = None
                except Exception:
                    pass

                try:
                    if getattr(_inductor_config, "max_autotune_gemm_search_space", None) is not None:
                        _inductor_config.max_autotune_gemm_search_space = "DEFAULT"
                except Exception:
                    pass

                try:
                    if getattr(_inductor_config, "compile_threads", None) is not None:
                        # max-autotune can be very memory hungry; default to serial compilation
                        # unless the user explicitly overrides the thread count.
                        override_raw = env_first(
                            (
                                "STNET_INDUCTOR_COMPILE_THREADS",
                                "STNET_COMPILE_THREADS",
                                "TORCHINDUCTOR_COMPILE_THREADS",
                            )
                        )
                        override_valid = False
                        if override_raw is not None:
                            with contextlib.suppress(Exception):
                                int(override_raw)
                                override_valid = True

                        if not override_valid:
                            _inductor_config.compile_threads = 1
                except Exception:
                    pass

                # max-autotune is extremely memory hungry. Keep GEMM tuning but disable
                # pointwise autotune by default to reduce peak memory (RAM/VRAM).
                try:
                    if getattr(_inductor_config, "max_autotune_pointwise", None) is not None:
                        _inductor_config.max_autotune_pointwise = False
                except Exception:
                    pass
                try:
                    if getattr(_inductor_config, "max_autotune_gemm", None) is not None:
                        _inductor_config.max_autotune_gemm = True
                except Exception:
                    pass

    backend_value = backend
    mode_value: Optional[str] = None
    if canonical_mode in {"aot_eager", "aot-eager"}:
        backend_value = "aot_eager"
    elif canonical_mode in recognized_modes:
        mode_value = canonical_mode
    elif mode is not None:
        mode_value = str(mode)

    compile_kwargs: Dict[str, Any] = dict(kwargs)
    if backend_value is not None:
        compile_kwargs["backend"] = backend_value
    if mode_value is not None:
        compile_kwargs["mode"] = mode_value
    if fullgraph is not None:
        compile_kwargs["fullgraph"] = bool(fullgraph)
    if dynamic is not None:
        compile_kwargs["dynamic"] = bool(dynamic)

    # Allow setting torch._inductor.config.* when user passes options in compile.
    if options and mode_value is not None:
        try:
            from torch._inductor import config as _inductor_config  # type: ignore
        except Exception:
            _inductor_config = None
        if _inductor_config is not None:
            for _k, _v in dict(options).items():
                if not isinstance(_k, str):
                    continue
                try:
                    _obj = _inductor_config
                    _parts = _k.split(".")
                    for _p in _parts[:-1]:
                        _obj = getattr(_obj, _p)
                    setattr(_obj, _parts[-1], _v)
                except Exception:
                    pass
        options = None

    if options:
        existing = compile_kwargs.get("options", {})
        if isinstance(existing, dict):
            merged = {**dict(options), **existing}
        else:
            merged = dict(options)
        compile_kwargs["options"] = merged

    return compile_fn(module, **compile_kwargs)


# -----------------------------------------------------------------------------
# Graph / compile helpers
# -----------------------------------------------------------------------------

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


def torch_compile_supported() -> bool:
    """Return True when torch.compile is available in this runtime."""
    return callable(getattr(torch, "compile", None))


def cudagraph_step_end() -> None:
    mark_step = getattr(_TORCH_COMPILER, "cudagraph_mark_step_end", None)
    if callable(mark_step):
        with suppress(Exception):
            mark_step()


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


def _compile_disable_decorator(
    *, reason: str | None = None, recursive: bool = True
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a decorator that disables torch.compile/Dynamo tracing for a function.

    Prefers `torch.compiler.disable` (newer) and falls back to `torch._dynamo.disable`
    (older). When the API is unavailable, this becomes a no-op identity decorator.

    The `disable()` signature has changed across PyTorch releases; this helper tries a
    few keyword combinations to stay compatible.
    """
    if _TORCH_COMPILE_DISABLE is None:
        def _identity(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn
        return _identity

    kwargs: dict[str, Any] = {}
    if reason is not None:
        kwargs["reason"] = reason
    # Some versions accept this keyword, others don't.
    kwargs["recursive"] = bool(recursive)

    for opts in (
        kwargs,
        {k: v for k, v in kwargs.items() if k != "recursive"},
        {k: v for k, v in kwargs.items() if k != "reason"},
        {},
    ):
        try:
            dec = _TORCH_COMPILE_DISABLE(**opts)
            if callable(dec):
                return dec
        except TypeError:
            continue
        except Exception:
            break

    def _identity(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    return _identity


def torch_compile_disable(
    target: Any | None = None,
    attr: str | None = None,
    /,
    *,
    reason: str | None = None,
    recursive: bool = True,
) -> Any:
    """Disable torch.compile for a function, or patch an attribute in-place.

    Supports both decorator and "patch an attribute" call styles:

    Decorator (with or without args):
        @torch_compile_disable
        def fn(...): ...

        @torch_compile_disable(reason="...", recursive=True)
        def fn(...): ...

    Attribute patching:
        torch_compile_disable(SomeClass, "method", reason="...", recursive=False) -> bool
    """
    if attr is not None:
        if target is None or (not hasattr(target, attr)):
            return False

        fn = getattr(target, attr)
        if getattr(fn, _NO_COMPILE_SENTINEL, False):
            return True

        decorator = _compile_disable_decorator(reason=reason, recursive=recursive)
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

    decorator = _compile_disable_decorator(reason=reason, recursive=recursive)
    if callable(target):
        return decorator(target)
    return decorator


# Backward-compatible aliases (prefer torch_compile_disable).
def torch_no_compile(
    *, reason: str | None = None, recursive: bool = True
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return torch_compile_disable(reason=reason, recursive=recursive)


def torch_disable_compile(
    target: Any, attr: str, *, reason: str | None = None, recursive: bool = True
) -> bool:
    return bool(torch_compile_disable(target, attr, reason=reason, recursive=recursive))


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


def torch_compile_safe(*, runtime_module: Any | None = None, layers_module: Any | None = None) -> None:
    """Patch known Python-side helpers to be eager under torch.compile.

    This function is intentionally best-effort and idempotent.

    Why this exists:
    - Some modules (e.g., Scaler) keep small Python caches guarded by locks.
      Those are great for eager execution, but tend to confuse Dynamo/AOT.
    - We prefer compiling the pure tensor math while keeping bookkeeping eager.
    """

    if not torch_compile_supported():
        return

    # Disallow tracing distributed collectives if possible.
    with suppress(Exception):
        torch_safe_distributed()

    # Resolve a reasonable default layers module (project layout changed over time).
    if layers_module is None:
        for mod_name in ("stnet.model.primitives", "stnet.model.blocks", "stnet.model.architecture"):
            with suppress(Exception):
                layers_module = importlib.import_module(mod_name)
                break

    # Scaler: uses Python dict + locks for dtype/device stats caching.
    scaler_cls = getattr(layers_module, "Scaler", None) if layers_module is not None else None
    if scaler_cls is None:
        for mod_name in ("stnet.model.primitives", "stnet.model.blocks"):
            with suppress(Exception):
                mod = importlib.import_module(mod_name)
                scaler_cls = getattr(mod, "Scaler", None)
                if scaler_cls is not None:
                    break

    if scaler_cls is not None:
        for attr in (
            "normalize_x",
            "denormalize_x",
            "normalize_y",
            "denormalize_y",
            "calibrate",
            "_piecewise",
        ):
            torch_compile_disable(
                scaler_cls,
                attr,
                reason="Scaler uses Python-side caches/loops; keep eager",
                recursive=False,
            )

    # History: logging / metadata buffers; never performance-critical.
    history_cls = getattr(layers_module, "History", None) if layers_module is not None else None
    if history_cls is None:
        for mod_name in ("stnet.model.primitives", "stnet.model.blocks"):
            with suppress(Exception):
                mod = importlib.import_module(mod_name)
                history_cls = getattr(mod, "History", None)
                if history_cls is not None:
                    break

    if history_cls is not None:
        for attr in (
            "start_session",
            "end_session",
            "set_epochs",
            "set_system_info",
            "record_batch",
            "_append",
            "save",
            "clear",
        ):
            torch_compile_disable(
                history_cls,
                attr,
                reason="History is logging/bookkeeping; keep eager",
                recursive=False,
            )
