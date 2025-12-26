# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import threading
from contextlib import AbstractContextManager
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from .casting import env_first, env_first_int
from .system import process_cpu_count


# torch._inductor config is process-global; guard concurrent mutation.
_INDUCTOR_CONFIG_LOCK = threading.Lock()


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
