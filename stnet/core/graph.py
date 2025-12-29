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
from .system import accel_is_available, process_cpu_count

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
    if not isinstance(model, nn.Module):
        return
    for attr in (
        "__stnet_cached_is_compiled_for_inference__",
        "__stnet_cached_is_aot_autograd_enabled__",
        "__stnet_cached_is_nvidia_te_available__",
    ):
        with contextlib.suppress(Exception):
            delattr(model, attr)


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
    if (
        is_nvidia_te_available(model)
        or _is_compiled_for_inference(model)
        or _is_aot_autograd_enabled(model)
    ):
        return torch.no_grad()
    return torch.inference_mode()


def _is_for_cuda(module: nn.Module) -> bool:
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
    return bool(accel_is_available("cuda"))


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
    mode_given = str(mode) if mode is not None else None
    mode_raw = str(mode_given or "").strip().lower()

    canonical_mode = mode_raw.replace("_", "-").replace(" ", "-")
    if "-" in canonical_mode:
        canonical_mode = "-".join(part for part in canonical_mode.split("-") if part)

    # Map common synonyms / separator variants to canonical torch.compile modes.
    mode_compact = canonical_mode.replace("-", "")
    match canonical_mode:
        case "" | "none" | "disabled" | "disable" | "off" | "false" | "0":
            canonical_mode = "disabled"
        case (
            "default"
            | "reduce-overhead"
            | "max-autotune"
            | "max-autotune-no-cudagraphs"
            | "aot-eager"
        ):
            pass
        case _:
            match mode_compact:
                case "reduceoverhead":
                    canonical_mode = "reduce-overhead"
                case "maxautotune":
                    canonical_mode = "max-autotune"
                case "maxautotunenocudagraphs" | "maxautotunenocudagraph":
                    canonical_mode = "max-autotune-no-cudagraphs"
                case "aoteager":
                    canonical_mode = "aot-eager"
                case _:
                    # Keep normalized string; may be a custom/forward-compatible torch.compile mode.
                    pass

    if disable:
        return module

    match canonical_mode:
        case "disabled":
            return module
        case _:
            pass

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return module

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
    match canonical_mode:
        case "aot-eager":
            # AOT eager uses a backend name rather than a mode string.
            backend_value = "aot_eager"
        case "default" | "reduce-overhead" | "max-autotune" | "max-autotune-no-cudagraphs":
            mode_value = canonical_mode
        case _:
            mode_value = mode_given

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
            from torch._inductor import \
                config as _inductor_config  # type: ignore
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
    return callable(getattr(torch, "compile", None))


def cudagraph_step_begin() -> None:
    mark_step = getattr(_TORCH_COMPILER, "cudagraph_mark_step_begin", None)
    if callable(mark_step):
        with suppress(Exception):
            mark_step()


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

    if not torch_compile_supported():
        return

    # Disallow tracing distributed collectives if possible.
    with suppress(Exception):
        torch_safe_distributed()

    # Resolve a reasonable default layers module (project layout changed over time).
    if layers_module is None:
        for mod_name in ("stnet.nn.primitives", "stnet.nn.blocks", "stnet.nn.architecture"):
            with suppress(Exception):
                layers_module = importlib.import_module(mod_name)
                break

    # Scaler: uses Python dict + locks for dtype/device stats caching.
    scaler_cls = getattr(layers_module, "Scaler", None) if layers_module is not None else None
    if scaler_cls is None:
        for mod_name in ("stnet.nn.primitives", "stnet.nn.blocks"):
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

    # Recorder: logging / metadata buffers; never performance-critical.
    history_cls = getattr(layers_module, "Recorder", None) if layers_module is not None else None
    if history_cls is None:
        for mod_name in ("stnet.nn.primitives", "stnet.nn.blocks"):
            with suppress(Exception):
                mod = importlib.import_module(mod_name)
                history_cls = getattr(mod, "Recorder", None)
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
                reason="Recorder is logging/bookkeeping; keep eager",
                recursive=False,
            )
