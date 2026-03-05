# -*- coding: utf-8 -*-
from __future__ import annotations

# =============================================================================
# 1. Standard Library Imports
# =============================================================================
import contextlib
import importlib
import logging
import sys
import threading
import traceback
import warnings
import weakref
from contextlib import AbstractContextManager, nullcontext, suppress
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Self, Sequence

# =============================================================================
# 2. Third-Party Imports
# =============================================================================
import torch
from torch import nn

# =============================================================================
# 3. Local Imports
# =============================================================================
from ..core.concurrency import Mutex
from ..core.datatypes import env_bool, env_first, env_first_int
from ..core.system import (
    CPU,
    get_runtime_cfg,
    is_accelerator_available,
    runtime_cfg_override,
)
from ..core.tensor import is_meta_or_fake_tensor
from ..runtime.distributed import broadcast_scalar, is_dtensor_active


# =============================================================================
# Globals & Constants
# =============================================================================
_COLLECTIVE_NAMES: tuple[str, ...] = (
    "all_gather",
    "all_gather_into_tensor",
    "all_reduce",
    "reduce_scatter_tensor",
    "broadcast",
    "barrier",
)
_GRAPH_BREAK_FN: Callable[[], None] | None = None
_GRAPH_BREAK_LOCK = Mutex()
_INDUCTOR_CONFIG_LOCK = Mutex(reentrant=True)
_INDUCTOR_WARN_FILTER_LOCK = Mutex(reentrant=True)
_INDUCTOR_MAX_AUTOTUNE_SMS_FILTERED = False
_SAFE_DIST_LOCK = Mutex()
_SAFE_DIST_PATCHED: set[str] = set()
_TORCH_COMPILER = getattr(torch, "compiler", None)
_TORCH_COMPILE_LOCK = Mutex(reentrant=True)
_NO_COMPILE_SENTINEL = "__enn_no_compile_wrapped__"

_CKPT_TL = threading.local()

try:
    import torch.utils.checkpoint
    _TORCH_CHECKPOINT = torch.utils.checkpoint.checkpoint
except Exception:
    _TORCH_CHECKPOINT = None

try:
    _TORCH_DYNAMO = importlib.import_module("torch._dynamo")
except Exception:
    _TORCH_DYNAMO = None


# =============================================================================
# Internal Environment & Lazy Init Helpers
# =============================================================================
def _is_in_jupyter() -> bool:
    try:
        return "ipykernel" in sys.modules or "google.colab" in sys.modules
    except Exception:
        return False


def _suppress_inductor_max_autotune_sms_warning() -> None:
    global _INDUCTOR_MAX_AUTOTUNE_SMS_FILTERED
    if _INDUCTOR_MAX_AUTOTUNE_SMS_FILTERED:
        return
    with _INDUCTOR_WARN_FILTER_LOCK:
        if _INDUCTOR_MAX_AUTOTUNE_SMS_FILTERED:
            return

        needle = "Not enough SMs to use max_autotune_gemm mode"

        class _DropInductorSMWarning(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                try:
                    return needle not in record.getMessage()
                except Exception:
                    return True

        flt = _DropInductorSMWarning()
        for logger_name in ("torch._inductor.utils", "torch._inductor", "torch", ""):
            with suppress(Exception):
                logging.getLogger(logger_name).addFilter(flt)

        _INDUCTOR_MAX_AUTOTUNE_SMS_FILTERED = True


def _is_compiled_for_inference(model: torch.nn.Module) -> bool:
    cached = getattr(model, "__enn_cached_is_compiled_for_inference__", None)
    if isinstance(cached, bool):
        return cached
        
    compile_attrs = (
        "_is_compiled_for_inference", "__is_compiled_for_inference__",
        "__compiled_for_serving__", "__serving_compiled__", "_is_serialized_for_serving",
    )
    if any(bool(getattr(model, attr, False)) for attr in compile_attrs):
        with suppress(Exception): setattr(model, "__enn_cached_is_compiled_for_inference__", True)
        return True
        
    jit = getattr(torch, "jit", None)
    script_like_types: List[type] = []
    if jit is not None:
        for name in ("ScriptModule", "RecursiveScriptModule", "TopLevelTracedModule"):
            typ = getattr(jit, name, None)
            if isinstance(typ, type): script_like_types.append(typ)
        for mod_name in ("_script", "_trace"):
            submod = getattr(jit, mod_name, None)
            if submod is None: continue
            for name in ("RecursiveScriptModule", "TopLevelTracedModule"):
                typ = getattr(submod, name, None)
                if isinstance(typ, type): script_like_types.append(typ)
                
    if any(isinstance(model, typ) for typ in script_like_types):
        with suppress(Exception): setattr(model, "__enn_cached_is_compiled_for_inference__", True)
        return True
        
    try:
        modules = tuple(model.modules())
    except (RuntimeError, AttributeError, TypeError):
        modules = ()
        
    for module in modules:
        if module is model: continue
        if any(bool(getattr(module, attr, False)) for attr in compile_attrs) or any(isinstance(module, typ) for typ in script_like_types):
            return True
            
    with suppress(Exception): setattr(model, "__enn_cached_is_compiled_for_inference__", False)
    return False


def _is_aot_autograd_enabled(model: torch.nn.Module) -> bool:
    cached = getattr(model, "__enn_cached_is_aot_autograd_enabled__", None)
    if isinstance(cached, bool):
        return cached
        
    indicator_attrs = (
        "_aot_autograd_graph", "_aot_autograd_cache", "_aot_compiled_autograd",
        "_aot_autograd_traced_module", "__aot_autograd__", "__compiled_with_aot_autograd__",
    )
    if any(getattr(model, attr, None) for attr in indicator_attrs):
        with suppress(Exception): setattr(model, "__enn_cached_is_aot_autograd_enabled__", True)
        return True
        
    try:
        modules = tuple(model.modules())
    except (RuntimeError, AttributeError, TypeError):
        modules = ()
        
    for module in modules:
        if module is model: continue
        if any(getattr(module, attr, None) for attr in indicator_attrs): return True
        class_name = module.__class__.__name__
        module_name = getattr(module.__class__, "__module__", "")
        if "AOTAutograd" in class_name or "aot_autograd" in module_name:
            return True
            
    with suppress(Exception): setattr(model, "__enn_cached_is_aot_autograd_enabled__", False)
    return False


def _is_for_cuda(module: nn.Module) -> bool:
    try:
        if isinstance(p0 := next(module.parameters(), None), torch.Tensor):
            return p0.device.type == "cuda"
    except Exception: pass
    try:
        if isinstance(b0 := next(module.buffers(), None), torch.Tensor):
            return b0.device.type == "cuda"
    except Exception: pass
    return bool(is_accelerator_available("cuda"))


def _resolve_compiler_disable() -> Any | None:
    if _TORCH_COMPILER is not None and callable(fn := getattr(_TORCH_COMPILER, "disable", None)):
        return fn
    if _TORCH_DYNAMO is not None and callable(fn := getattr(_TORCH_DYNAMO, "disable", None)):
        return fn
    return None


def _resolve_graph_break() -> Callable[[], None] | None:
    with suppress(Exception):
        inductor = importlib.import_module("torch._inductor")
        if callable(gb := getattr(inductor, "graph_break", None)): return gb
    if _TORCH_DYNAMO is not None and callable(gb := getattr(_TORCH_DYNAMO, "graph_break", None)):
        return gb
    return None


def _get_inductor_config() -> Any | None:
    with suppress(Exception):
        from torch._inductor import config
        return config
    return None


def _identity(fn: Callable[..., Any]) -> Callable[..., Any]:
    return fn


def _decorate_compiler_disable(*args: Any, reason: str | None = None, recursive: bool = True) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    fn_disable = _resolve_compiler_disable()
    if fn_disable is None: return _identity
    
    kwargs: dict[str, Any] = {"recursive": bool(recursive)}
    if reason is not None: kwargs["reason"] = reason
        
    for opts in (kwargs, {k: v for k, v in kwargs.items() if k != "recursive"}, {k: v for k, v in kwargs.items() if k != "reason"}, {}):
        try:
            if callable(dec := fn_disable(**opts)): return dec
        except TypeError: continue
        except Exception: break
    return _identity


def _dispatch_mode_stack() -> list[Any]:
    pd = getattr(getattr(torch, "utils", None), "_python_dispatch", None)
    if callable(fn_stack := getattr(pd, "_get_current_dispatch_mode_stack", None)):
        stack = fn_stack()
        return list(stack) if stack is not None else []
    if callable(fn_mode := getattr(pd, "_get_current_dispatch_mode", None)):
        mode = fn_mode()
        return [mode] if mode is not None else []
    return []


# =============================================================================
# Torch Compile & Dispatch Utilities
# =============================================================================
@contextlib.contextmanager
def skip_non_infra_dispatch_mode() -> Iterator[None]:
    try:
        from torch.utils._python_dispatch import _disable_current_modes, is_in_torch_dispatch_mode
    except Exception:
        yield
        return

    active_non_infra = False
    for kw in ({"include_infra_modes": False}, {False}, {}):
        with suppress(Exception):
            active_non_infra = bool(is_in_torch_dispatch_mode(**kw) if isinstance(kw, dict) else is_in_torch_dispatch_mode(*kw))
            if active_non_infra: break
            
    if not active_non_infra:
        yield
        return

    try:
        cm = _disable_current_modes()
        cm.__enter__()
    except Exception:
        yield
        return

    exc_type = exc = tb = None
    try:
        yield
    except Exception:
        exc_type, exc, tb = sys.exc_info()
        raise
    finally:
        cm.__exit__(exc_type, exc, tb)


def is_dynamo_compiling() -> bool:
    for mod in (getattr(torch, "compiler", None), getattr(torch, "_dynamo", None)):
        if callable(fn := getattr(mod, "is_dynamo_compiling", None)) and bool(fn()): return True
    return False


def is_compiling() -> bool:
    dyn = getattr(torch, "_dynamo", None)
    for fn_name in ("is_compiling", "is_dynamo_compiling"):
        if callable(fn := getattr(dyn, fn_name, None)) and bool(fn()): return True
    if callable(fn := getattr(getattr(torch, "compiler", None), "is_compiling", None)) and bool(fn()): return True
    return False


def is_fake_tensor_mode_active() -> bool:
    FakeTensorMode = getattr(getattr(getattr(torch, "_subclasses", None), "fake_tensor", None), "FakeTensorMode", None)
    if FakeTensorMode is None: return False
    return any(isinstance(mode, FakeTensorMode) for mode in _dispatch_mode_stack() if mode is not None)


def is_tracing_or_exporting() -> bool:
    if (jit := getattr(torch, "jit", None)) is not None and (torch.jit.is_tracing() or torch.jit.is_scripting()): return True
    if callable(fn := getattr(getattr(torch, "compiler", None), "is_exporting", None)) and bool(fn()): return True
    if callable(fn := getattr(getattr(torch, "onnx", None), "is_in_onnx_export", None)) and bool(fn()): return True
    return is_fake_tensor_mode_active()


def is_export_or_trace() -> bool:
    return bool(is_tracing_or_exporting() or is_compiling() or is_fake_tensor_mode_active())


def is_symbolic() -> bool:
    return is_export_or_trace()


def assert_trace(condition: object, message: str = "") -> None:
    if callable(fn := getattr(torch, "_assert_scalar", None)):
        fn(condition, message)
        return

    ok = False
    try:
        match condition:
            case torch.Tensor() if is_meta_or_fake_tensor(condition):
                return
            case torch.Tensor() if condition.numel() == 1:
                ok = bool(condition.item())
            case torch.Tensor():
                ok = bool(condition.all().item())
            case _:
                ok = bool(condition)
    except Exception:
        ok = False
        
    if not ok:
        raise RuntimeError(str(message))


def canonicalize_compile_mode(mode: object | None) -> str:
    if not isinstance(mode, str): return "disabled"
    compact_mode = mode.lower().replace("_", "").replace("-", "").replace(" ", "")
    
    match compact_mode:
        case "default": return "default"
        case "aoteager" | "debug": return "aot-eager"
        case "reduceoverhead" | "stable": return "reduce-overhead"
        case "maxautotune": return "max-autotune"
        case "maxautotunenocudagraphs" | "maxautotunenocudagraph": return "max-autotune-no-cudagraphs"
        case _: return "disabled"


def clear_model_cache(model: Optional[nn.Module]) -> None:
    if not isinstance(model, nn.Module): return
    for attr in ("__enn_cached_is_compiled_for_inference__", "__enn_cached_is_aot_autograd_enabled__", "__enn_cached_is_nvidia_te_available__"):
        with suppress(Exception): delattr(model, attr)


def is_nvidia_te_available(model: torch.nn.Module) -> bool:
    if isinstance(cached := getattr(model, "__enn_cached_is_nvidia_te_available__", None), bool): return cached
    
    if any(getattr(model, attr, False) for attr in ("__fp8_inference_te__", "__fp8_training_te__", "__te_fp8_default__")):
        with suppress(Exception): setattr(model, "__enn_cached_is_nvidia_te_available__", True)
        return True
        
    for module in model.modules():
        if isinstance(mod_name := getattr(module.__class__, "__module__", ""), str) and mod_name.startswith("transformer_engine"):
            with suppress(Exception): setattr(model, "__enn_cached_is_nvidia_te_available__", True)
            return True
            
    with suppress(Exception): setattr(model, "__enn_cached_is_nvidia_te_available__", False)
    return False


def inference_mode(model: torch.nn.Module) -> AbstractContextManager[None]:
    if is_symbolic() or is_nvidia_te_available(model) or _is_compiled_for_inference(model) or _is_aot_autograd_enabled(model):
        return torch.no_grad()
    return torch.inference_mode()


# =============================================================================
# Compile Core
# =============================================================================
def compile(
    module: nn.Module, *args: Any, backend: Optional[str] = None, mode: Optional[str] = None,
    fullgraph: Optional[bool] = None, dynamic: Optional[bool] = None, options: Optional[Dict[str, Any]] = None,
    disable: bool = False, **kwargs: Any,
) -> nn.Module:
    del args
    if disable: return module
    
    canonical_mode = canonicalize_compile_mode(mode)
    if canonical_mode == "disabled" or not torch_compiler_supported(): return module
    if not callable(compile_fn := getattr(torch, "compile", None)): return module
    
    _suppress_inductor_max_autotune_sms_warning()
    if canonical_mode == "max-autotune" and not _is_for_cuda(module):
        canonical_mode = "max-autotune-no-cudagraphs"
        
    opt: Dict[str, Any] = dict(options or {})
    if canonical_mode == "max-autotune-no-cudagraphs": opt.setdefault("triton.cudagraphs", False)
    options_merged: Dict[str, Any] | None = opt or None
    
    _inductor_config = _get_inductor_config()
    _restore_inductor: Dict[str, Any] | None = None
    _scoped_inductor_overrides: Dict[str, Any] | None = None
    
    if _inductor_config is not None:
        with _INDUCTOR_CONFIG_LOCK:
            with suppress(Exception):
                if getattr(_inductor_config, "compile_threads", None) is not None:
                    override = env_first(("ENN_INDUCTOR_COMPILE_THREADS", "ENN_COMPILE_THREADS", "TORCHINDUCTOR_COMPILE_THREADS"), None)
                    if override is not None:
                        _inductor_config.compile_threads = max(1, int(override))
                    else:
                        local_world = env_first_int(("ENN_LOCAL_WORLD_SIZE", "LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE"), 1)
                        if int(local_world) > 1:
                            _inductor_config.compile_threads = max(1, min(2, max(1, int(CPU.count() or 1) // max(1, int(local_world))) // 2))

            if canonical_mode in {"max-autotune", "max-autotune-no-cudagraphs"}:
                _restore_inductor, _scoped_inductor_overrides = {}, {}

                def _snapshot(attr: str) -> None:
                    if hasattr(_inductor_config, attr):
                        with suppress(Exception): _restore_inductor[attr] = getattr(_inductor_config, attr)

                for _attr in ("autotune_in_subproc", "autotune_local_cache", "autotune_remote_cache", "max_autotune_gemm_search_space", "max_autotune_pointwise", "max_autotune_gemm", "compile_threads"):
                    _snapshot(_attr)

                def _want(attr: str, value: Any) -> None:
                    if hasattr(_inductor_config, attr): _scoped_inductor_overrides[attr] = value

                want_subproc = env_bool(("ENN_INDUCTOR_AUTOTUNE_IN_SUBPROC", "ENN_AUTOTUNE_IN_SUBPROC", "TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC"), default=not _is_in_jupyter())
                with suppress(Exception): _inductor_config.autotune_in_subproc = bool(want_subproc)
                _want("autotune_in_subproc", bool(want_subproc))
                with suppress(Exception): _inductor_config.autotune_local_cache = True
                _want("autotune_local_cache", True)
                with suppress(Exception): _inductor_config.autotune_remote_cache = None
                _want("autotune_remote_cache", None)

                for attr, val in (("max_autotune_gemm_search_space", "DEFAULT"), ("max_autotune_pointwise", False), ("max_autotune_gemm", True)):
                    with suppress(Exception):
                        if getattr(_inductor_config, attr, None) is not None:
                            setattr(_inductor_config, attr, val)
                            _want(attr, val)
                            
                with suppress(Exception):
                    if getattr(_inductor_config, "compile_threads", None) is not None:
                        override_raw = env_first(("ENN_INDUCTOR_COMPILE_THREADS", "ENN_COMPILE_THREADS", "TORCHINDUCTOR_COMPILE_THREADS"))
                        override_valid = False
                        if override_raw is not None:
                            with suppress(Exception):
                                int(override_raw)
                                override_valid = True
                        if not override_valid:
                            threads = 1
                            if env_first_int(("ENN_LOCAL_WORLD_SIZE", "LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE"), 1) <= 1 and _is_in_jupyter():
                                threads = max(1, min(2, int(CPU.count() or 1) // 2))
                            _inductor_config.compile_threads = int(threads)
                            _want("compile_threads", int(threads))
    try:
        backend_value = backend
        mode_value: Optional[str] = None
        match canonical_mode:
            case "aot-eager": backend_value = "aot_eager"
            case "reduce-overhead" | "max-autotune" | "max-autotune-no-cudagraphs": mode_value = canonical_mode
            case _: mode_value = str(mode) if mode is not None else None
            
        compile_kwargs: Dict[str, Any] = dict(kwargs)
        if backend_value is not None: compile_kwargs["backend"] = backend_value
        if mode_value is not None: compile_kwargs["mode"] = mode_value
        if fullgraph is not None: compile_kwargs["fullgraph"] = bool(fullgraph)
        if dynamic is not None: compile_kwargs["dynamic"] = bool(dynamic)
        if options_merged is not None:
            if isinstance(existing := compile_kwargs.get("options", {}), dict): options_merged = {**options_merged, **existing}
            compile_kwargs["options"] = options_merged
            
        inductor_cfg = _get_inductor_config()
        patch = getattr(inductor_cfg, "patch", None) if inductor_cfg is not None else None
        patchable: Dict[str, Any] = {}

        def _has_cfg_key(cfg: Any, key: str) -> bool:
            obj = cfg
            for part in key.split("."):
                if not hasattr(obj, part): return False
                obj = getattr(obj, part)
            return True

        options_dict = dict(compile_kwargs.get("options") or {}) if isinstance(compile_kwargs.get("options", None), dict) else {}
        compile_time_keys = {"triton.cudagraphs", "triton.cudagraph_trees"}
        compile_time_opts = {k: v for k, v in options_dict.items() if isinstance(k, str) and k in compile_time_keys}
        
        if callable(patch) and options_dict:
            patchable = {k: v for k, v in options_dict.items() if isinstance(k, str) and k not in compile_time_keys and _has_cfg_key(inductor_cfg, k)}
            
        strip_options = bool(mode_value is not None)
        compile_time_patch: Dict[str, Any] = {}
        
        if strip_options and compile_time_opts:
            if callable(patch): compile_time_patch = dict(compile_time_opts)
            else:
                compile_kwargs.pop("mode", None)
                strip_options = False

        if strip_options: compile_kwargs.pop("options", None)
        elif patchable:
            if compile_time_opts: compile_kwargs["options"] = dict(compile_time_opts)
            else: compile_kwargs.pop("options", None)

        with _TORCH_COMPILE_LOCK:
            cm_compile = patch(dict(compile_time_patch)) if callable(patch) and compile_time_patch else nullcontext()
            with cm_compile:
                compiled = compile_fn(module, **compile_kwargs)

        if not (bool(_scoped_inductor_overrides) or bool(patchable)):
            return compiled

        class _ScopedInductorCompiled(nn.Module):
            def __init__(self, inner: nn.Module, cfg: Any, overrides: Dict[str, Any] | None, restore: Dict[str, Any] | None, patch_fn: Any, patch_dict: Dict[str, Any]) -> None:
                super().__init__()
                self._enn_inner = inner
                self._enn_cfg = cfg
                self._enn_overrides = dict(overrides or {})
                self._enn_restore = dict(restore or {})
                self._enn_patch_fn = patch_fn
                self._enn_patch_dict = dict(patch_dict or {})

            def forward(self, *f_args: Any, **f_kwargs: Any) -> Any:
                if self._enn_cfg is None or (not self._enn_overrides and not self._enn_patch_dict): return self._enn_inner(*f_args, **f_kwargs)
                with _INDUCTOR_CONFIG_LOCK:
                    for k, v in self._enn_overrides.items():
                        with suppress(Exception): setattr(self._enn_cfg, k, v)
                    cm = self._enn_patch_fn(self._enn_patch_dict) if callable(self._enn_patch_fn) and self._enn_patch_dict else nullcontext()
                    try:
                        with cm: return self._enn_inner(*f_args, **f_kwargs)
                    finally:
                        for k, v in self._enn_restore.items():
                            with suppress(Exception): setattr(self._enn_cfg, k, v)

            def __getattr__(self, name: str) -> Any:
                try: return super().__getattr__(name)
                except AttributeError: return getattr(self._enn_inner, name)

            def state_dict(self, *sd_args: Any, **sd_kwargs: Any) -> Any: return self._enn_inner.state_dict(*sd_args, **sd_kwargs)
            def load_state_dict(self, *ls_args: Any, **ls_kwargs: Any) -> Any: return self._enn_inner.load_state_dict(*ls_args, **ls_kwargs)

        return _ScopedInductorCompiled(compiled, inductor_cfg, _scoped_inductor_overrides, _restore_inductor, patch, patchable)
    finally:
        if _restore_inductor and _inductor_config is not None:
            with _INDUCTOR_CONFIG_LOCK:
                for k, v in _restore_inductor.items():
                    with suppress(Exception): setattr(_inductor_config, k, v)


def torch_compiler_supported() -> bool:
    if env_bool("ENN_TORCH_COMPILE", default=True) is False: return False
    if not callable(getattr(torch, "compile", None)): return False
    with suppress(Exception):
        if getattr(torch, "jit", None) is not None and (torch.jit.is_tracing() or torch.jit.is_scripting()): return False
    with suppress(Exception):
        if callable(is_exporting := getattr(getattr(torch, "compiler", None), "is_exporting", None)) and bool(is_exporting()): return False
    return True


def cudagraph_mark_step_begin() -> None:
    if is_export_or_trace(): return
    with suppress(Exception):
        if callable(mark_step := getattr(_TORCH_COMPILER, "cudagraph_mark_step_begin", None)): mark_step()


def cudagraph_mark_step_end() -> None:
    if is_export_or_trace(): return
    with suppress(Exception):
        if callable(mark_step := getattr(_TORCH_COMPILER, "cudagraph_mark_step_end", None)): mark_step()


def graph_break() -> None:
    if _TORCH_DYNAMO is None: return
    with suppress(Exception):
        if callable(is_exporting := getattr(getattr(torch, "compiler", None), "is_exporting", None)) and bool(is_exporting()): return
    with suppress(Exception):
        if getattr(torch, "jit", None) is not None and (torch.jit.is_tracing() or torch.jit.is_scripting()): return
    with suppress(Exception):
        if callable(is_onnx_export := getattr(getattr(torch, "onnx", None), "is_in_onnx_export", None)) and bool(is_onnx_export()): return
    with suppress(Exception):
        if not _TORCH_DYNAMO.is_compiling(): return
        
    global _GRAPH_BREAK_FN
    if _GRAPH_BREAK_FN is None:
        with _GRAPH_BREAK_LOCK:
            if _GRAPH_BREAK_FN is None: _GRAPH_BREAK_FN = _resolve_graph_break()
    if _GRAPH_BREAK_FN is not None:
        with suppress(Exception): _GRAPH_BREAK_FN()


def torch_compiler_disable(target: Any | None = None, attr: str | None = None, /, *args: Any, reason: str | None = None, recursive: bool = True) -> Any:
    if attr is not None:
        if target is None or not hasattr(target, attr): return False
        fn = getattr(target, attr)
        if getattr(fn, _NO_COMPILE_SENTINEL, False): return True
        decorator = _decorate_compiler_disable(reason=reason, recursive=recursive)
        try: non_export_wrapped = decorator(fn)
        except Exception: return False

        if non_export_wrapped is fn:
            wrapped = fn
        else:
            import functools
            @functools.wraps(fn)
            def wrapped(*a: Any, **kw: Any) -> Any:
                return fn(*a, **kw) if is_export_or_trace() else non_export_wrapped(*a, **kw)

        with suppress(Exception): setattr(wrapped, _NO_COMPILE_SENTINEL, True)
        try: setattr(target, attr, wrapped)
        except Exception: return False
        return True
        
    decorator = _decorate_compiler_disable(reason=reason, recursive=recursive)
    if callable(target):
        fn = target
        non_export_wrapped = decorator(fn)
        if non_export_wrapped is fn: return fn
        import functools
        @functools.wraps(fn)
        def wrapped(*a: Any, **kw: Any) -> Any:
            return fn(*a, **kw) if is_export_or_trace() else non_export_wrapped(*a, **kw)
        return wrapped
    return decorator


def compile_distributed_safe(*args: Any, collectives: tuple[str, ...] = _COLLECTIVE_NAMES) -> bool:
    if _TORCH_DYNAMO is None or (disallow := getattr(_TORCH_DYNAMO, "disallow_in_graph", None)) is None: return False
    try: import torch.distributed as dist
    except Exception: return False
    
    updated = False
    with _SAFE_DIST_LOCK:
        for name in collectives:
            if name in _SAFE_DIST_PATCHED: continue
            if (fn := getattr(dist, name, None)) is None: continue
            with suppress(Exception):
                disallow(fn)
                _SAFE_DIST_PATCHED.add(name)
                updated = True
    return updated


def compile_safe(*args: Any, runtime_module: Any | None = None, layers_module: Any | None = None) -> None:
    if not torch_compiler_supported(): return
    with suppress(Exception): compile_distributed_safe()
    
    if layers_module is None:
        for mod_name in ("enn_torch.nn.layers", "enn_torch.nn.blocks", "enn_torch.nn.wrappers"):
            with suppress(Exception):
                layers_module = importlib.import_module(mod_name)
                break
                
    scaler_cls = getattr(layers_module, "Scaler", None) if layers_module is not None else None
    if scaler_cls is None:
        for mod_name in ("enn_torch.nn.layers", "enn_torch.nn.blocks"):
            with suppress(Exception):
                scaler_cls = getattr(importlib.import_module(mod_name), "Scaler", None)
                if scaler_cls is not None: break
                
    if scaler_cls is not None:
        for attr in ("normalize_x", "denormalize_x", "normalize_y", "denormalize_y", "calibrate", "_piecewise"):
            torch_compiler_disable(scaler_cls, attr, reason="Scaler uses Python-side caches/loops; keep eager", recursive=False)
            
    history_cls = getattr(layers_module, "Recorder", None) if layers_module is not None else None
    if history_cls is None:
        for mod_name in ("enn_torch.nn.layers", "enn_torch.nn.blocks"):
            with suppress(Exception):
                history_cls = getattr(importlib.import_module(mod_name), "Recorder", None)
                if history_cls is not None: break
                
    if history_cls is not None:
        for attr in ("start_session", "end_session", "set_epochs", "set_system_info", "record_batch", "_append", "save", "clear"):
            torch_compiler_disable(history_cls, attr, reason="Recorder is logging/bookkeeping; keep eager", recursive=False)


# =============================================================================
# Graph Checkpointing & Submodules
# =============================================================================
def to_submodule(model: nn.Module) -> Optional[nn.Module]:
    m = model
    for _ in range(8):
        if hasattr(m, "microbatch") and hasattr(m, "_auto_microbatch_pending"): return m
        child = getattr(m, "module", None)
        if child is None or child is m: break
        m = child
    return None


def _raised_from_checkpointed_fn(err: BaseException) -> bool:
    tb = err.__traceback__
    if tb is None: return False
    for frame, _ in traceback.walk_tb(tb):
        if frame.f_code.co_name == "_state" and frame.f_globals.get("__name__") == __name__: return True
    return False


def iter_checkpoint(root: nn.Module) -> Iterator[nn.Module]:
    if not isinstance(root, nn.Module): return
    for mod in root.modules():
        if hasattr(mod, "_ckpt_min_bytes") and hasattr(mod, "_ckpt_enabled"): yield mod


def to_checkpoint(model: object, *args: Any, device: torch.device, step_total: int, ttl_steps: int, min_bytes: int) -> bool:
    inst = to_submodule(model) or (model.module if hasattr(model, "module") else model)
    if inst is None: return False
    try:
        ttl_steps, min_bytes, step_total = max(1, int(ttl_steps)), max(0, int(min_bytes)), max(0, int(step_total))
    except Exception: return False
    until = step_total + ttl_steps
    try:
        until = int(broadcast_scalar(until, device=device, src=0))
        min_bytes = int(broadcast_scalar(min_bytes, device=device, src=0))
    except Exception: pass
    
    cur_until = int(getattr(inst, "_enn_ckpt_pressure_until", 0) or 0)
    if cur_until >= until and int(getattr(inst, "_enn_ckpt_pressure_min_bytes", 0) or 0) <= min_bytes: return False
    
    changed = False
    for mod in iter_checkpoint(inst):
        if not hasattr(mod, "_enn_ckpt_saved_min_bytes"):
            with suppress(Exception):
                setattr(mod, "_enn_ckpt_saved_min_bytes", int(getattr(mod, "_ckpt_min_bytes", 0) or 0))
                setattr(mod, "_enn_ckpt_saved_enabled", bool(getattr(mod, "_ckpt_enabled", True)))
        try:
            if min_bytes < int(getattr(mod, "_ckpt_min_bytes", 0) or 0):
                setattr(mod, "_ckpt_min_bytes", int(min_bytes))
                changed = True
            if not bool(getattr(mod, "_ckpt_enabled", True)):
                setattr(mod, "_ckpt_enabled", True)
                changed = True
        except Exception: pass
        
    with suppress(Exception):
        setattr(inst, "_enn_ckpt_pressure_until", int(max(cur_until, until)))
        prev_mb = int(getattr(inst, "_enn_ckpt_pressure_min_bytes", 0) or 0)
        setattr(inst, "_enn_ckpt_pressure_min_bytes", int(min_bytes) if prev_mb <= 0 else int(min(prev_mb, min_bytes)))
    return bool(changed)


def from_checkpoint(model: nn.Module, *args: Any, step_total: int) -> None:
    inst = to_submodule(model) or (model.module if hasattr(model, "module") else model)
    if inst is None: return
    try: step_total = int(step_total)
    except Exception: return
    
    until = int(getattr(inst, "_enn_ckpt_pressure_until", 0) or 0)
    if until <= 0 or step_total < until: return
    
    for mod in iter_checkpoint(inst):
        with suppress(Exception):
            if hasattr(mod, "_enn_ckpt_saved_min_bytes"): setattr(mod, "_ckpt_min_bytes", int(getattr(mod, "_enn_ckpt_saved_min_bytes", 0) or 0))
            if hasattr(mod, "_enn_ckpt_saved_enabled"): setattr(mod, "_ckpt_enabled", bool(getattr(mod, "_enn_ckpt_saved_enabled", True)))
            for k in ("_enn_ckpt_saved_min_bytes", "_enn_ckpt_saved_enabled"): delattr(mod, k)
            
    with suppress(Exception):
        setattr(inst, "_enn_ckpt_pressure_until", 0)
        setattr(inst, "_enn_ckpt_pressure_min_bytes", 0)


def is_checkpoint() -> bool:
    return bool(getattr(_CKPT_TL, "depth", 0) or 0)


def coerce_checkpoint(fn: Callable[..., Any], *args: Any, **ckpt_kwargs: Any) -> Any:
    if _TORCH_CHECKPOINT is None or is_export_or_trace() or not any(isinstance(a, torch.Tensor) and getattr(a, "requires_grad", False) for a in args):
        return fn(*args)
        
    force_reentrant = env_first(("ENN_CKPT_REQUIRE_REENTRANT",), default=None)
    require_reentrant = env_bool("ENN_CKPT_REQUIRE_REENTRANT", default=False) if force_reentrant is not None else bool(is_dtensor_active())
    
    use_reentrant = ckpt_kwargs.pop("use_reentrant", True if require_reentrant else True)
    preserve_rng_state = ckpt_kwargs.pop("preserve_rng_state", True)
    determinism_check = ckpt_kwargs.pop("determinism_check", None)
    
    ck_opts = {k: v for k, v in [("use_reentrant", use_reentrant), ("preserve_rng_state", preserve_rng_state), ("determinism_check", determinism_check)] if v is not None}
    tried: set[tuple[tuple[str, object], ...]] = set()
    last_type_error: TypeError | None = None
    
    opts_list: list[dict[str, object]] = [ck_opts, {k: v for k, v in ck_opts.items() if k != "determinism_check"}]
    opts_list.extend([{k: v for k, v in ck_opts.items() if k == "use_reentrant"}] if require_reentrant else [{k: v for k, v in ck_opts.items() if k not in ("determinism_check", "use_reentrant")}, {k: v for k, v in ck_opts.items() if k != "use_reentrant"}, {}])
    
    for opts in opts_list:
        key = tuple(sorted(opts.items()))
        if key in tried: continue
        tried.add(key)
        try: return checkpoint(fn, *args, **opts, **ckpt_kwargs)
        except TypeError as e:
            if require_reentrant and _raised_from_checkpointed_fn(e): raise
            last_type_error = e
            continue
            
    if require_reentrant:
        raise TypeError("DTensor/FSDP2 checkpointing requires `use_reentrant=True`, but torch.utils.checkpoint.checkpoint did not accept a compatible signature in this runtime. Upgrade PyTorch or set ENN_CKPT_REQUIRE_REENTRANT=0 to override.") from last_type_error
    return checkpoint(fn, *args, **ckpt_kwargs)


def checkpoint(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    if _TORCH_CHECKPOINT is None: return fn(*args, **kwargs)

    def _state(*a: Any, **k: Any) -> Any:
        depth = int(getattr(_CKPT_TL, "depth", 0) or 0)
        setattr(_CKPT_TL, "depth", depth + 1)
        try:
            disable_cg = False
            with suppress(Exception):
                cfg = get_runtime_cfg()
                if (prev_cg := getattr(cfg, "compile_cudagraphs", None)) is None:
                    cg_mode = canonicalize_compile_mode(getattr(cfg, "compile_mode", "disabled")) not in {"disabled", "aot-eager", "max-autotune-no-cudagraphs"}

                    def _has_cuda_tensor(obj: Any, _depth: int = 0) -> bool:
                        if _depth > 4: return False
                        match obj:
                            case torch.Tensor(): return bool(getattr(obj.device, "type", None) == "cuda")
                            case tuple() | list(): return any(_has_cuda_tensor(x, _depth + 1) for x in obj)
                            case dict(): return any(_has_cuda_tensor(x, _depth + 1) for x in obj.values())
                            case _: return False

                    prev_cg = bool(cg_mode and (_has_cuda_tensor(a) or _has_cuda_tensor(k)))
                disable_cg = bool(env_bool("ENN_CKPT_DISABLE_CUDAGRAPHS", default=True) and prev_cg and is_accelerator_available("cuda"))
                
            if disable_cg:
                cudagraph_mark_step_begin()
                fn_no_compile = torch_compiler_disable(fn, reason="checkpoint region: disable cudagraphs/compile for safety", recursive=False)
                with runtime_cfg_override(compile_cudagraphs=False): return fn_no_compile(*a, **k)
            return fn(*a, **k)
        finally:
            setattr(_CKPT_TL, "depth", depth)

    return _TORCH_CHECKPOINT(_state, *args, **kwargs)


# =============================================================================
# GraphSequential Types & Class
# =============================================================================
@dataclass
class BorrowedModule:
    module: nn.Module
    name: str | None = None

@dataclass
class OwnedModule:
    module: nn.Module
    name: str | None = None

@dataclass
class ModulePath:
    path: str
    name: str | None = None

@dataclass
class CallArguments:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

class ReduceMean(nn.Module):
    def __init__(self: Self, dim: int = 1, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = int(dim)
        self.keepdim = bool(keepdim)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim, keepdim=self.keepdim)


class GraphSequential(nn.Module):
    _CONTROL_ATTR = "__enn_subgraph_control_op__"

    def __init__(self: Self, steps: Sequence[object], *args: Any, out_shape: object | None = None, name: str | None = None, root: nn.Module | None = None) -> None:
        super().__init__()
        self._name = str(name or "subgraph")
        self._owned = nn.ModuleList()
        self._refs_materialized = False
        self._root_ref: weakref.ReferenceType[nn.Module] | None = weakref.ref(root) if root is not None else None
        self._path_cache: dict[str, weakref.ReferenceType[nn.Module]] = {}
        self._path_cache_lock = threading.Lock()
        self._out_shape_kind, self._out_shape_spec = self._normalize_out_shape(out_shape)
        
        compiled_steps: list[tuple[object, ...]] = []
        for raw in steps:
            step, extra_args, extra_kwargs = self._parse_step(raw)
            meta: dict[str, Any] | None = None

            match step:
                case BorrowedModule(mod, n):
                    if n: meta = {"name": str(n)}
                    compiled_steps.append(("ref", weakref.ref(mod), extra_args, extra_kwargs, meta))
                case ModulePath(p, n):
                    meta = {"path": str(p), "name": str(n) if n else None}
                    compiled_steps.append(("path", str(p), extra_args, extra_kwargs, meta))
                case OwnedModule(mod, n):
                    if n: meta = {"name": str(n)}
                    self._owned.append(mod)
                    compiled_steps.append(("owned", len(self._owned) - 1, extra_args, extra_kwargs, meta))
                case nn.Module():
                    compiled_steps.append(("ref", weakref.ref(step), extra_args, extra_kwargs, meta))
                case _ if callable(step):
                    if (tag := getattr(step, self._CONTROL_ATTR, None)) is not None: meta = {"control": str(tag)}
                    compiled_steps.append(("fn", step, extra_args, extra_kwargs, meta))
                case _:
                    raise TypeError(f"Unsupported GraphSequential step: {type(step)!r}")
                    
        if not compiled_steps: raise ValueError("GraphSequential requires at least one step.")
        self._steps: list[tuple[object, ...]] = compiled_steps

    @staticmethod
    def ref(module: nn.Module, *args: Any, name: str | None = None) -> BorrowedModule:
        return BorrowedModule(module=module, name=name)

    @staticmethod
    def own(module: nn.Module, *args: Any, name: str | None = None) -> OwnedModule:
        return OwnedModule(module=module, name=name)

    @staticmethod
    def path(path: str, *args: Any, name: str | None = None) -> ModulePath:
        return ModulePath(path=str(path), name=name)

    @staticmethod
    def mean(dim: int = 1, *args: Any, keepdim: bool = False) -> OwnedModule:
        return OwnedModule(module=ReduceMean(dim=int(dim), keepdim=bool(keepdim)), name="mean")

    @staticmethod
    def io(*args: Any, **kwargs: Any) -> CallArguments:
        return CallArguments(args=tuple(args), kwargs=dict(kwargs))

    @staticmethod
    def _tag_control(fn: Callable[..., Any], tag: str) -> Callable[..., Any]:
        with suppress(Exception): setattr(fn, GraphSequential._CONTROL_ATTR, str(tag))
        return fn

    @staticmethod
    def break_graph() -> Callable[..., Any]:
        def _op(*a: Any, **kw: Any) -> Any:
            graph_break()
            return CallArguments(args=tuple(a), kwargs=dict(kw)) if kw else (a[0] if len(a) == 1 else tuple(a))
        return GraphSequential._tag_control(_op, "graph_break")

    @staticmethod
    def cudagraph_begin(*args: Any, disable_compile: bool = True) -> Callable[..., Any]:
        def _op(*a: Any, **kw: Any) -> Any:
            cudagraph_mark_step_begin()
            return CallArguments(args=tuple(a), kwargs=dict(kw)) if kw else (a[0] if len(a) == 1 else tuple(a))
        _op = GraphSequential._tag_control(_op, "cudagraph_begin")
        return torch_compiler_disable(_op, reason="subgraph:cudagraph_begin", recursive=False) if disable_compile else _op

    @staticmethod
    def cudagraph_end(*args: Any, disable_compile: bool = True) -> Callable[..., Any]:
        def _op(*a: Any, **kw: Any) -> Any:
            cudagraph_mark_step_end()
            return CallArguments(args=tuple(a), kwargs=dict(kw)) if kw else (a[0] if len(a) == 1 else tuple(a))
        _op = GraphSequential._tag_control(_op, "cudagraph_end")
        return torch_compiler_disable(_op, reason="subgraph:cudagraph_end", recursive=False) if disable_compile else _op

    @staticmethod
    def no_compile(step: nn.Module | Callable[..., Any], *args: Any, reason: str | None = None, recursive: bool = False) -> Callable[..., Any]:
        if isinstance(step, nn.Module):
            ref = weakref.ref(step)
            def _call(*a: Any, **kw: Any) -> Any:
                if (mod := ref()) is None: raise RuntimeError("A shared submodule reference was cleared before GraphSequential.forward().")
                return mod(*a, **kw)
        else:
            def _call(*a: Any, **kw: Any) -> Any: return step(*a, **kw)
            
        wrapped = torch_compiler_disable(_call, reason=str(reason or "subgraph:no_compile"), recursive=bool(recursive))
        return GraphSequential._tag_control(wrapped, "no_compile")

    @staticmethod
    def checkpoint(step: nn.Module | Callable[..., Any], *args: Any, use_reentrant: bool | None = None, preserve_rng_state: bool | None = None, determinism_check: str | None = None) -> Callable[..., Any]:
        if isinstance(step, nn.Module):
            ref = weakref.ref(step)
            def _call(*a: Any, **kw: Any) -> Any:
                if (mod := ref()) is None: raise RuntimeError("A shared submodule reference was cleared before GraphSequential.forward().")
                def _inner(*aa: Any) -> Any: return mod(*aa, **kw)
                return coerce_checkpoint(_inner, *a, use_reentrant=use_reentrant, preserve_rng_state=preserve_rng_state, determinism_check=determinism_check)
        else:
            def _call(*a: Any, **kw: Any) -> Any:
                def _inner(*aa: Any) -> Any: return step(*aa, **kw)
                return coerce_checkpoint(_inner, *a, use_reentrant=use_reentrant, preserve_rng_state=preserve_rng_state, determinism_check=determinism_check)
        return GraphSequential._tag_control(_call, "checkpoint")

    def set_root(self: Self, root: nn.Module | None) -> "GraphSequential":
        self._root_ref = weakref.ref(root) if root is not None else None
        with self._path_cache_lock: self._path_cache.clear()
        return self

    def bind(self: Self, root: nn.Module | None = None, *args: Any, strict: bool = True) -> "GraphSequential":
        if root is not None: self.set_root(root)
        self._refs_materialized = True
        rebound: list[tuple[object, ...]] = []
        
        for item in list(self._steps):
            kind, payload, extra_args, extra_kwargs, meta = self._split_step(item)
            match kind:
                case "path":
                    m = dict(meta) if isinstance(meta, dict) else {}
                    m["path"] = str(payload)
                    rebound.append(("ref", self._resolve_path(str(payload)), extra_args, extra_kwargs, m))
                case "ref" if payload is None:
                    path = meta.get("path") if isinstance(meta, dict) else None
                    if isinstance(path, str):
                        rebound.append(("ref", self._resolve_path(path), extra_args, extra_kwargs, meta))
                    elif strict:
                        raise RuntimeError("GraphSequential.bind() encountered an unresolved ref without a path hint.")
                case "ref" if isinstance(payload, weakref.ReferenceType):
                    with contextlib.suppress(Exception):
                        if isinstance(mod := payload(), nn.Module):
                            rebound.append(("ref", mod, extra_args, extra_kwargs, meta))
                            continue
                    rebound.append((kind, payload, extra_args, extra_kwargs, meta))
                case _:
                    rebound.append((kind, payload, extra_args, extra_kwargs, meta))
                    
        self._steps = rebound
        return self

    def forward(self: Self, *args: Any, **kwargs: Any) -> Any:
        cur: Any = CallArguments(args=tuple(args), kwargs=dict(kwargs)) if kwargs else (args[0] if len(args) == 1 else tuple(args))
        for item in self._steps:
            kind, payload, extra_args, extra_kwargs, meta = self._split_step(item)
            cur = self._apply_step(kind, payload, cur, extra_args, extra_kwargs, meta=meta)
        return self._apply_out_shape(cur)

    def extra_repr(self: Self) -> str:
        return f"name={self._name!r}, out_shape={self._out_shape_spec!r}, steps={len(self._steps)}"

    def __getstate__(self: Self) -> dict[str, object]:
        state = super().__getstate__()
        sanitized: list[tuple[object, ...]] = []
        if isinstance(steps := state.get("_steps", []), list):
            for item in steps:
                kind, payload, extra_args, extra_kwargs, meta = self._split_step(item)
                sanitized.append(("ref", None, extra_args, extra_kwargs, meta) if kind == "ref" else (kind, payload, extra_args, extra_kwargs, meta))
        state.update({"_steps": sanitized, "_root_ref": None, "_path_cache": {}, "_path_cache_lock": None})
        return state

    def __setstate__(self: Self, state: dict[str, object]) -> None:
        super().__setstate__(state)
        if getattr(self, "_path_cache", None) is None: self._path_cache = {}
        if getattr(self, "_path_cache_lock", None) is None: self._path_cache_lock = threading.Lock()
        if getattr(self, "_root_ref", None) is None: self._root_ref = None

    @staticmethod
    def _parse_step(raw: object) -> tuple[object, tuple[Any, ...], dict[str, Any]]:
        match raw:
            case (step, dict() as kw) if len(raw) == 2:
                return step, (), dict(kw)
            case (step, (tuple() | list()) as arg, dict() as kw) if len(raw) == 3:
                return step, tuple(arg), dict(kw)
            case _:
                return raw, (), {}

    @staticmethod
    def _split_step(item: object) -> tuple[str, object, tuple[Any, ...], dict[str, Any], object | None]:
        if not isinstance(item, tuple) or len(item) < 4: raise TypeError("Invalid GraphSequential internal step format.")
        
        raw_args = item[2]
        extra_args = () if raw_args is None else (raw_args if isinstance(raw_args, tuple) else tuple(raw_args) if isinstance(raw_args, list) else (raw_args,))
        
        raw_kw = item[3]
        extra_kwargs = {} if raw_kw is None else (dict(raw_kw) if isinstance(raw_kw, dict) else dict(raw_kw) if hasattr(raw_kw, "items") else {})
        
        return str(item[0]), item[1], extra_args, extra_kwargs, item[4] if len(item) >= 5 else None

    @staticmethod
    def _normalize_out_shape(out_shape: object | None) -> tuple[str | None, object | None]:
        match out_shape:
            case None:
                return None, None
            case dict():
                return "dict", {str(k): (None if v is None else tuple(int(x) for x in v)) for k, v in out_shape.items()}
            case list() | tuple() if out_shape and isinstance(out_shape[0], (list, tuple, type(None))):
                return "seq", tuple((None if s is None else tuple(int(x) for x in s)) for s in out_shape)
            case list() | tuple():
                return "single", tuple(int(x) for x in out_shape)
            case _:
                return "single", tuple(int(x) for x in (out_shape,))

    @staticmethod
    def _unpack(value: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        match value:
            case CallArguments(args, kwargs): return tuple(args), dict(kwargs)
            case tuple(): return value, {}
            case list(): return tuple(value), {}
            case dict(): return (), value
            case _: return (value,), {}

    def _resolve_path(self: Self, path: str) -> nn.Module:
        with self._path_cache_lock:
            if (ref := self._path_cache.get(path)) is not None and (mod := ref()) is not None: return mod
            
        if (root := self._root_ref() if self._root_ref is not None else None) is None:
            raise RuntimeError("GraphSequential requires `root=` (or set_root()) when using ModulePath steps.")
            
        mod: nn.Module | None = getattr(root, "get_submodule", lambda x: None)(path)
        if mod is None:
            cur: nn.Module = root
            for part in str(path).split("."):
                child = getattr(cur, "_modules", None)
                if isinstance(child, dict) and part in child: nxt = child.get(part)
                else: nxt = getattr(cur, part, None)
                if not isinstance(nxt, nn.Module): raise AttributeError(f"Failed to resolve submodule path {path!r} at {part!r}.")
                cur = nxt
            mod = cur
            
        if not isinstance(mod, nn.Module): raise TypeError(f"get_submodule({path!r}) did not return an nn.Module")
        with self._path_cache_lock: self._path_cache[path] = weakref.ref(mod)
        return mod

    def _resolve_path_nocache(self: Self, path: str) -> nn.Module:
        if (root := self._root_ref() if self._root_ref is not None else None) is None:
            raise RuntimeError("GraphSequential requires `root=` (or set_root()) when using ModulePath steps.")
            
        mod: nn.Module | None = getattr(root, "get_submodule", lambda x: None)(path)
        if mod is None:
            cur: nn.Module = root
            for part in str(path).split("."):
                child = getattr(cur, "_modules", None)
                if isinstance(child, dict) and part in child: nxt = child.get(part)
                else: nxt = getattr(cur, part, None)
                if not isinstance(nxt, nn.Module): raise AttributeError(f"Failed to resolve submodule path {path!r} at {part!r}.")
                cur = nxt
            mod = cur
            
        if not isinstance(mod, nn.Module): raise TypeError(f"get_submodule({path!r}) did not return an nn.Module")
        return mod

    def _apply_step(self: Self, kind: str, payload: object, cur: Any, extra_args: tuple[Any, ...], extra_kwargs: dict[str, Any], *args: Any, meta: object | None = None) -> Any:
        args, kwargs = self._unpack(cur)
        if extra_args: args = tuple(args) + tuple(extra_args)
        if extra_kwargs: kwargs = {**kwargs, **extra_kwargs}
        
        match kind:
            case "ref":
                mod: nn.Module | None = None
                path = meta.get("path") if isinstance(meta, dict) else None
                match payload:
                    case nn.Module(): mod = payload
                    case weakref.ReferenceType() if not is_compiling() or not isinstance(path, str): mod = payload()
                if mod is None:
                    if isinstance(path, str): mod = self._resolve_path_nocache(path) if is_compiling() else self._resolve_path(path)
                    else: raise RuntimeError("A shared submodule reference was cleared (or not bound) before GraphSequential.forward().")
                return mod(*args, **kwargs)
            case "owned": return self._owned[int(payload)](*args, **kwargs)
            case "path": return self._resolve_path(str(payload))(*args, **kwargs)
            case _: return payload(*args, **kwargs)

    def _apply_out_shape(self: Self, out: Any) -> Any:
        kind, spec = self._out_shape_kind, self._out_shape_spec
        if kind is None or spec is None: return out

        def _reshape_one(t: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
            if t.ndim == 0: raise RuntimeError("Cannot reshape a scalar output in GraphSequential.")
            return t.reshape(t.shape[0], *shape)

        match kind:
            case "single":
                if not isinstance(out, torch.Tensor): raise RuntimeError("out_shape is set but the pipeline output is not a Tensor.")
                return _reshape_one(out, spec)
            case "seq":
                if not isinstance(out, (tuple, list)): raise RuntimeError("out_shape expects tuple/list output but got a different type.")
                shapes = list(spec)
                if len(out) != len(shapes): raise RuntimeError("out_shape length does not match tuple/list output length.")
                out_list = list(out)
                for i, sh in enumerate(shapes):
                    if sh is None: continue
                    if not isinstance(out_list[i], torch.Tensor): raise RuntimeError("out_shape expects Tensor outputs in tuple/list.")
                    out_list[i] = _reshape_one(out_list[i], sh)
                return tuple(out_list) if isinstance(out, tuple) else out_list
            case "dict":
                if not isinstance(out, dict): raise RuntimeError("out_shape expects dict output but got a different type.")
                out_dict = dict(out)
                for k, sh in spec.items():
                    if sh is None: continue
                    if k not in out_dict: raise RuntimeError(f"out_shape missing key in output dict: {k!r}")
                    if not isinstance(out_dict[k], torch.Tensor): raise RuntimeError("out_shape expects Tensor values in dict output.")
                    out_dict[k] = _reshape_one(out_dict[k], sh)
                return out_dict
            case _:
                return out

    def extract_for_serving(self: Self, *args: Any, root: nn.Module | None = None, clone_modules: bool = True, strip_control_ops: bool = True, name: str | None = None) -> "GraphSequential":
        import copy
        if root is not None: self.set_root(root)
        
        steps_out: list[object] = []
        for item in list(self._steps):
            kind, payload, extra_args, extra_kwargs, meta = self._split_step(item)
            match kind:
                case "fn":
                    if strip_control_ops and bool(getattr(payload, self._CONTROL_ATTR, "")): continue
                    step_obj: object = payload
                case _:
                    mod: nn.Module | None = None
                    match kind:
                        case "owned": mod = self._owned[int(payload)]
                        case "path": mod = self._resolve_path(str(payload))
                        case "ref":
                            path = meta.get("path") if isinstance(meta, dict) else None
                            match payload:
                                case nn.Module(): mod = payload
                                case weakref.ReferenceType(): mod = payload()
                                case _: mod = None
                            if mod is None and isinstance(path, str): mod = self._resolve_path(str(path))
                        case _: raise TypeError(f"Unknown GraphSequential step kind: {kind!r}")
                            
                    if not isinstance(mod, nn.Module): raise RuntimeError(f"extract_for_serving could not resolve module for step kind={kind!r}")
                    if clone_modules:
                        try: mod = copy.deepcopy(mod)
                        except Exception as e:
                            warnings.warn(f"GraphSequential.extract_for_serving: deepcopy failed for {mod.__class__.__name__}: {e}. Falling back to sharing the original module object.", RuntimeWarning)
                    step_obj = OwnedModule(module=mod)
                    
            if extra_args and extra_kwargs: steps_out.append((step_obj, extra_args, extra_kwargs))
            elif extra_args: steps_out.append((step_obj, extra_args, {}))
            elif extra_kwargs: steps_out.append((step_obj, extra_kwargs))
            else: steps_out.append(step_obj)
            
        out = GraphSequential(steps=steps_out, out_shape=(self._out_shape_spec if self._out_shape_kind is not None else None), name=str(name or f"{self._name}_serving"), root=None)
        out.eval()
        with suppress(Exception): out.requires_grad_(False)
        with suppress(Exception): setattr(out, "__compiled_for_serving__", True)
        return out
