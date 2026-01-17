# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import importlib
import threading
import traceback
from contextlib import AbstractContextManager, suppress, nullcontext
from typing import Any, Callable, Dict, List, Optional, Iterator

import torch
from torch import nn

from .concurrency import Mutex
from .datatypes import env_bool, env_first, env_first_int
from .distributed import broadcast_scalar, is_dtensor_active
from .system import CPU, is_accelerator_available
from .tensor import is_meta_or_fake_tensor

try:
    from torch.utils.checkpoint import checkpoint as _torch_checkpoint
except Exception:
    _torch_checkpoint = None


try:
    _TORCH_DYNAMO = importlib.import_module("torch._dynamo")
except Exception:
    _TORCH_DYNAMO = None

_INDUCTOR_CONFIG_LOCK = Mutex(reentrant=True)

_TORCH_COMPILER = getattr(torch, "compiler", None)
_TORCH_COMPILE_LOCK = Mutex(reentrant=True)

_CKPT_TL = threading.local()

_SAFE_DIST_LOCK = Mutex()
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

_GRAPH_BREAK_FN: Callable[[], None] | None = None
_GRAPH_BREAK_LOCK = Mutex()


def _raised_from_checkpointed_fn(err: BaseException) -> bool:
    tb = err.__traceback__
    if tb is None:
        return False
    for frame, _ in traceback.walk_tb(tb):
        if (
            frame.f_code.co_name == "_state"
            and frame.f_globals.get("__name__") == __name__
        ):
            return True
    return False


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
        for name in (
            "ScriptModule",
            "RecursiveScriptModule",
            "TopLevelTracedModule",
        ):
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
    return bool(is_accelerator_available("cuda"))


def _resolve_compiler_disable() -> Any | None:
    if _TORCH_COMPILER is not None:
        fn = getattr(_TORCH_COMPILER, "disable", None)
        if callable(fn):
            return fn
    if _TORCH_DYNAMO is not None:
        fn = getattr(_TORCH_DYNAMO, "disable", None)
        if callable(fn):
            return fn
    return None


def _resolve_graph_break() -> Callable[[], None] | None:
    try:
        inductor = importlib.import_module("torch._inductor")
        gb = getattr(inductor, "graph_break", None)
        if callable(gb):
            return gb
    except Exception:
        pass
    if _TORCH_DYNAMO is not None:
        gb = getattr(_TORCH_DYNAMO, "graph_break", None)
        if callable(gb):
            return gb
    return None


def _get_inductor_config() -> Any | None:
    try:
        from torch._inductor import config
    except Exception:
        return None
    return config


def _identity(fn: Callable[..., Any]) -> Callable[..., Any]:
    return fn


def _decorate_compiler_disable(
    *args: Any, reason: str | None = None, recursive: bool = True
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if _resolve_compiler_disable() is None:
        return _identity
    kwargs: dict[str, Any] = {}
    if reason is not None:
        kwargs["reason"] = reason
    kwargs["recursive"] = bool(recursive)
    for opts in (
        kwargs,
        {k: v for k, v in kwargs.items() if k != "recursive"},
        {k: v for k, v in kwargs.items() if k != "reason"},
        {},
    ):
        try:
            dec = _resolve_compiler_disable()(**opts)
            if callable(dec):
                return dec
        except TypeError:
            continue
        except Exception:
            break
    return _identity


def _dispatch_mode_stack() -> list[Any]:
    try:
        from torch.utils._python_dispatch import (
            _get_current_dispatch_mode_stack,
        )

        stack = _get_current_dispatch_mode_stack()
        return list(stack) if stack is not None else []
    except Exception:
        pass
    try:
        from torch.utils._python_dispatch import _get_current_dispatch_mode

        mode = _get_current_dispatch_mode()
        return [mode] if mode is not None else []
    except Exception:
        return []


def is_compiling() -> bool:
    with suppress(Exception):
        dyn = getattr(torch, "_dynamo", None)
        if dyn is not None and callable(getattr(dyn, "is_compiling", None)):
            if bool(dyn.is_compiling()):
                return True
        if dyn is not None and callable(
            getattr(dyn, "is_dynamo_compiling", None)
        ):
            if bool(dyn.is_dynamo_compiling()):
                return True
    with suppress(Exception):
        comp = getattr(torch, "compiler", None)
        fn = getattr(comp, "is_compiling", None)
        if callable(fn) and bool(fn()):
            return True
    return False


def is_fake_tensor_mode_active() -> bool:
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
    except Exception:
        return False
    for mode in _dispatch_mode_stack():
        if mode is None:
            continue
        try:
            if isinstance(mode, FakeTensorMode):
                return True
        except Exception:
            continue
    return False


def is_tracing_or_exporting() -> bool:
    with suppress(Exception):
        jit = getattr(torch, "jit", None)
        if jit is not None and (
            torch.jit.is_tracing() or torch.jit.is_scripting()
        ):
            return True
    with suppress(Exception):
        comp = getattr(torch, "compiler", None)
        fn = getattr(comp, "is_exporting", None)
        if callable(fn) and bool(fn()):
            return True
    with suppress(Exception):
        onnx = getattr(torch, "onnx", None)
        fn = getattr(onnx, "is_in_onnx_export", None)
        if callable(fn) and bool(fn()):
            return True
    with suppress(Exception):
        if is_fake_tensor_mode_active():
            return True
    return False


def is_export_or_trace() -> bool:
    return bool(is_tracing_or_exporting() or is_fake_tensor_mode_active())


def is_symbolic() -> bool:
    return bool(
        is_tracing_or_exporting()
        or is_compiling()
        or is_fake_tensor_mode_active()
    )


def assert_trace(condition: object, message: str = "") -> None:
    fn = getattr(torch, "_assert_scalar", None)
    if callable(fn):
        fn(condition, message)
        return

    try:
        if isinstance(condition, torch.Tensor):
            if is_meta_or_fake_tensor(condition):
                return
            if condition.numel() == 1:
                ok = bool(condition.item())
            else:
                ok = bool(condition.all().item())
        else:
            ok = bool(condition)
    except Exception:
        ok = False
    if not ok:
        raise RuntimeError(str(message))


def canonicalize_compile_mode(mode: object | None) -> str:
    if not isinstance(mode, str):
        return "disabled"
    compact_mode = (
        mode.lower().replace("_", "").replace("-", "").replace(" ", "")
    )
    mode_map = {
        "default": "default",
        "aoteager": "aot-eager",
        "reduceoverhead": "reduce-overhead",
        "maxautotune": "max-autotune",
        "maxautotunenocudagraphs": "max-autotune-no-cudagraphs",
        "maxautotunenocudagraph": "max-autotune-no-cudagraphs",
        "debug": "aot-eager",
        "stable": "reduce-overhead",
    }
    return mode_map.get(compact_mode, "disabled")


def clear_model_cache(model: Optional[nn.Module]) -> None:
    if not isinstance(model, nn.Module):
        return
    for attr in (
        "__stnet_cached_is_compiled_for_inference__",
        "__stnet_cached_is_aot_autograd_enabled__",
        "__stnet_cached_is_nvidia_te_available__",
    ):
        with suppress(Exception):
            delattr(model, attr)


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
        if isinstance(mod_name, str) and mod_name.startswith(
            "transformer_engine"
        ):
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
        is_symbolic()
        or is_nvidia_te_available(model)
        or _is_compiled_for_inference(model)
        or _is_aot_autograd_enabled(model)
    ):
        return torch.no_grad()
    return torch.inference_mode()


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
    del args
    if disable:
        return module
    canonical_mode = canonicalize_compile_mode(mode)
    if canonical_mode == "disabled":
        return module
    if not torch_compiler_supported():
        return module
    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        return module
    if canonical_mode == "max-autotune" and not _is_for_cuda(module):
        canonical_mode = "max-autotune-no-cudagraphs"
    opt: Dict[str, Any] = dict(options or {})
    if CPU.is_free_threaded_build():
        opt.setdefault("compile_threads", 1)
        opt.setdefault("triton.cudagraphs", False)
    if canonical_mode == "max-autotune-no-cudagraphs":
        opt.setdefault("triton.cudagraphs", False)
    options_merged: Dict[str, Any] | None = opt or None
    _inductor_config = _get_inductor_config()
    if _inductor_config is not None:
        with _INDUCTOR_CONFIG_LOCK:
            try:
                if (
                    getattr(_inductor_config, "compile_threads", None)
                    is not None
                ):
                    override = env_first(
                        (
                            "STNET_INDUCTOR_COMPILE_THREADS",
                            "STNET_COMPILE_THREADS",
                        ),
                        None,
                    )
                    if override is None:
                        override = env_first(
                            ("TORCHINDUCTOR_COMPILE_THREADS",), None
                        )
                    if override is not None:
                        _inductor_config.compile_threads = max(
                            1, int(override)
                        )
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
                            cpu_count = int(CPU.count() or 1)
                            per_rank = max(
                                1, int(cpu_count) // max(1, int(local_world))
                            )
                            _inductor_config.compile_threads = max(
                                1, min(4, int(per_rank) // 2)
                            )
            except Exception:
                pass
            if canonical_mode in {
                "max-autotune",
                "max-autotune-no-cudagraphs",
            }:
                with suppress(Exception):
                    _inductor_config.autotune_in_subproc = True
                with suppress(Exception):
                    _inductor_config.autotune_local_cache = True
                with suppress(Exception):
                    _inductor_config.autotune_remote_cache = None
                with suppress(Exception):
                    if (
                        getattr(
                            _inductor_config,
                            "max_autotune_gemm_search_space",
                            None,
                        )
                        is not None
                    ):
                        _inductor_config.max_autotune_gemm_search_space = (
                            "DEFAULT"
                        )
                with suppress(Exception):
                    if (
                        getattr(
                            _inductor_config, "max_autotune_pointwise", None
                        )
                        is not None
                    ):
                        _inductor_config.max_autotune_pointwise = False
                with suppress(Exception):
                    if (
                        getattr(_inductor_config, "max_autotune_gemm", None)
                        is not None
                    ):
                        _inductor_config.max_autotune_gemm = True
                with suppress(Exception):
                    if (
                        getattr(_inductor_config, "compile_threads", None)
                        is not None
                    ):
                        override_raw = env_first(
                            (
                                "STNET_INDUCTOR_COMPILE_THREADS",
                                "STNET_COMPILE_THREADS",
                                "TORCHINDUCTOR_COMPILE_THREADS",
                            )
                        )
                        override_valid = False
                        if override_raw is not None:
                            with suppress(Exception):
                                int(override_raw)
                                override_valid = True
                        if not override_valid:
                            _inductor_config.compile_threads = 1
    backend_value = backend
    mode_value: Optional[str] = None
    match canonical_mode:
        case "aot-eager":
            backend_value = "aot_eager"
        case "reduce-overhead" | "max-autotune" | "max-autotune-no-cudagraphs":
            mode_value = canonical_mode
        case _:
            mode_value = str(mode) if mode is not None else None
    compile_kwargs: Dict[str, Any] = dict(kwargs)
    if backend_value is not None:
        compile_kwargs["backend"] = backend_value
    if mode_value is not None:
        compile_kwargs["mode"] = mode_value
    if fullgraph is not None:
        compile_kwargs["fullgraph"] = bool(fullgraph)
    if dynamic is not None:
        compile_kwargs["dynamic"] = bool(dynamic)
    if options_merged is not None:
        existing = compile_kwargs.get("options", {})
        if isinstance(existing, dict):
            options_merged = {**options_merged, **existing}
        compile_kwargs["options"] = options_merged
    if isinstance(compile_kwargs.get("options", None), dict):
        inductor_cfg = _get_inductor_config()
        patch = (
            getattr(inductor_cfg, "patch", None)
            if inductor_cfg is not None
            else None
        )

        def _has_cfg_key(cfg: Any, key: str) -> bool:
            obj = cfg
            for part in key.split("."):
                if not hasattr(obj, part):
                    return False
                obj = getattr(obj, part)
            return True

        options_dict = dict(compile_kwargs.get("options") or {})
        if callable(patch) and options_dict:
            patchable = {
                k: v
                for k, v in options_dict.items()
                if isinstance(k, str) and _has_cfg_key(inductor_cfg, k)
            }
            strip_options = mode_value is not None
            with _TORCH_COMPILE_LOCK, _INDUCTOR_CONFIG_LOCK:
                with patch(patchable) if patchable else nullcontext():
                    if strip_options or patchable:
                        compile_kwargs.pop("options", None)
                    return compile_fn(module, **compile_kwargs)
        if mode_value is not None:
            compile_kwargs.pop("options", None)
    with _TORCH_COMPILE_LOCK:
        return compile_fn(module, **compile_kwargs)


def torch_compiler_supported() -> bool:
    if env_bool("STNET_TORCH_COMPILE", default=True) is False:
        return False
    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        return False
    try:
        if getattr(torch, "jit", None) is not None:
            if torch.jit.is_tracing() or torch.jit.is_scripting():
                return False
    except Exception:
        pass
    try:
        comp = getattr(torch, "compiler", None)
        is_exporting = (
            getattr(comp, "is_exporting", None) if comp is not None else None
        )
        if callable(is_exporting) and bool(is_exporting()):
            return False
    except Exception:
        pass
    return True


def cudagraph_mark_step_begin() -> None:
    mark_step = getattr(_TORCH_COMPILER, "cudagraph_mark_step_begin", None)
    if callable(mark_step):
        with suppress(Exception):
            mark_step()


def cudagraph_mark_step_end() -> None:
    mark_step = getattr(_TORCH_COMPILER, "cudagraph_mark_step_end", None)
    if callable(mark_step):
        with suppress(Exception):
            mark_step()


def graph_break() -> None:
    dyn = _TORCH_DYNAMO
    if dyn is None:
        return
    try:
        comp = getattr(torch, "compiler", None)
        is_exporting = getattr(comp, "is_exporting", None)
        if callable(is_exporting) and bool(is_exporting()):
            return
    except Exception:
        pass
    try:
        if getattr(torch, "jit", None) is not None:
            if torch.jit.is_tracing() or torch.jit.is_scripting():
                return
    except Exception:
        pass
    try:
        onnx = getattr(torch, "onnx", None)
        is_onnx_export = getattr(onnx, "is_in_onnx_export", None)
        if callable(is_onnx_export) and bool(is_onnx_export()):
            return
    except Exception:
        pass
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
                _GRAPH_BREAK_FN = _resolve_graph_break()
            fn = _GRAPH_BREAK_FN
    if fn is None:
        return
    with suppress(Exception):
        fn()


def torch_compiler_disable(
    target: Any | None = None,
    attr: str | None = None,
    /,
    *args: Any,
    reason: str | None = None,
    recursive: bool = True,
) -> Any:
    if attr is not None:
        if target is None or (not hasattr(target, attr)):
            return False
        fn = getattr(target, attr)
        if getattr(fn, _NO_COMPILE_SENTINEL, False):
            return True
        decorator = _decorate_compiler_disable(
            reason=reason, recursive=recursive
        )
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
    decorator = _decorate_compiler_disable(reason=reason, recursive=recursive)
    if callable(target):
        return decorator(target)
    return decorator


def compile_distributed_safe(
    *args: Any, collectives: tuple[str, ...] = _COLLECTIVE_NAMES
) -> bool:
    if _TORCH_DYNAMO is None or not hasattr(
        _TORCH_DYNAMO, "disallow_in_graph"
    ):
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


def compile_safe(
    *args: Any,
    runtime_module: Any | None = None,
    layers_module: Any | None = None,
) -> None:
    if not torch_compiler_supported():
        return
    with suppress(Exception):
        compile_distributed_safe()
    if layers_module is None:
        for mod_name in (
            "stnet.nn.layers",
            "stnet.nn.blocks",
            "stnet.nn.architecture",
        ):
            with suppress(Exception):
                layers_module = importlib.import_module(mod_name)
                break
    scaler_cls = (
        getattr(layers_module, "Scaler", None)
        if layers_module is not None
        else None
    )
    if scaler_cls is None:
        for mod_name in ("stnet.nn.layers", "stnet.nn.blocks"):
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
            torch_compiler_disable(
                scaler_cls,
                attr,
                reason="Scaler uses Python-side caches/loops; keep eager",
                recursive=False,
            )
    history_cls = (
        getattr(layers_module, "Recorder", None)
        if layers_module is not None
        else None
    )
    if history_cls is None:
        for mod_name in ("stnet.nn.layers", "stnet.nn.blocks"):
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
            torch_compiler_disable(
                history_cls,
                attr,
                reason="Recorder is logging/bookkeeping; keep eager",
                recursive=False,
            )


def to_submodule(model: nn.Module) -> Optional[nn.Module]:
    m = model
    for _ in range(8):
        if hasattr(m, "microbatch") and hasattr(m, "_auto_microbatch_pending"):
            return m
        child = getattr(m, "module", None)
        if child is None or child is m:
            break
        m = child
    return None


def iter_checkpoint(root: nn.Module) -> Iterator[nn.Module]:
    try:
        import torch.nn as nn

        if isinstance(root, nn.Module):
            for mod in root.modules():
                if hasattr(mod, "_ckpt_min_bytes") and hasattr(
                    mod, "_ckpt_enabled"
                ):
                    yield mod
    except Exception:
        return


def to_checkpoint(
    model: object,
    *args: Any,
    device: torch.device,
    step_total: int,
    ttl_steps: int,
    min_bytes: int,
) -> bool:
    inst = to_submodule(model) or (
        model.module if hasattr(model, "module") else model
    )
    if inst is None:
        return False
    try:
        ttl_steps = max(1, int(ttl_steps))
        min_bytes = max(0, int(min_bytes))
        step_total = max(0, int(step_total))
    except Exception:
        return False
    until = step_total + ttl_steps
    try:
        until = int(broadcast_scalar(until, device=device, src=0))
        min_bytes = int(broadcast_scalar(min_bytes, device=device, src=0))
    except Exception:
        pass
    cur_until = int(getattr(inst, "_stnet_ckpt_pressure_until", 0) or 0)
    if (
        cur_until >= until
        and int(getattr(inst, "_stnet_ckpt_pressure_min_bytes", 0) or 0)
        <= min_bytes
    ):
        return False
    changed = False
    for mod in iter_checkpoint(inst):
        if not hasattr(mod, "_stnet_ckpt_saved_min_bytes"):
            with contextlib.suppress(Exception):
                setattr(
                    mod,
                    "_stnet_ckpt_saved_min_bytes",
                    int(getattr(mod, "_ckpt_min_bytes", 0) or 0),
                )
                setattr(
                    mod,
                    "_stnet_ckpt_saved_enabled",
                    bool(getattr(mod, "_ckpt_enabled", True)),
                )
        try:
            cur = int(getattr(mod, "_ckpt_min_bytes", 0) or 0)
            if min_bytes < cur:
                setattr(mod, "_ckpt_min_bytes", int(min_bytes))
                changed = True
            if not bool(getattr(mod, "_ckpt_enabled", True)):
                setattr(mod, "_ckpt_enabled", True)
                changed = True
        except Exception:
            pass
    with contextlib.suppress(Exception):
        setattr(inst, "_stnet_ckpt_pressure_until", int(max(cur_until, until)))
        prev_mb = int(getattr(inst, "_stnet_ckpt_pressure_min_bytes", 0) or 0)
        if prev_mb <= 0:
            setattr(inst, "_stnet_ckpt_pressure_min_bytes", int(min_bytes))
        else:
            setattr(
                inst,
                "_stnet_ckpt_pressure_min_bytes",
                int(min(prev_mb, min_bytes)),
            )
    return bool(changed)


def from_checkpoint(model: nn.Module, *args: Any, step_total: int) -> None:
    inst = to_submodule(model) or (
        model.module if hasattr(model, "module") else model
    )
    if inst is None:
        return
    try:
        step_total = int(step_total)
    except Exception:
        return
    until = int(getattr(inst, "_stnet_ckpt_pressure_until", 0) or 0)
    if until <= 0 or step_total < until:
        return
    for mod in iter_checkpoint(inst):
        try:
            if hasattr(mod, "_stnet_ckpt_saved_min_bytes"):
                setattr(
                    mod,
                    "_ckpt_min_bytes",
                    int(getattr(mod, "_stnet_ckpt_saved_min_bytes", 0) or 0),
                )
            if hasattr(mod, "_stnet_ckpt_saved_enabled"):
                setattr(
                    mod,
                    "_ckpt_enabled",
                    bool(getattr(mod, "_stnet_ckpt_saved_enabled", True)),
                )
            for k in (
                "_stnet_ckpt_saved_min_bytes",
                "_stnet_ckpt_saved_enabled",
            ):
                with contextlib.suppress(Exception):
                    delattr(mod, k)
        except Exception:
            pass
    with contextlib.suppress(Exception):
        setattr(inst, "_stnet_ckpt_pressure_until", 0)
        setattr(inst, "_stnet_ckpt_pressure_min_bytes", 0)


def is_checkpoint() -> bool:
    return bool(getattr(_CKPT_TL, "depth", 0) or 0)


def coerce_checkpoint(
    fn: Callable[..., Any],
    *args: Any,
    **ckpt_kwargs: Any,
) -> Any:
    if _torch_checkpoint is None:
        return fn(*args)
    if is_export_or_trace() or not any(
        isinstance(a, torch.Tensor) and a.requires_grad for a in args
    ):
        return fn(*args)
    force_reentrant = env_first(
        ("STNET_CKPT_REQUIRE_REENTRANT",), default=None
    )
    require_reentrant = (
        env_bool("STNET_CKPT_REQUIRE_REENTRANT", default=False)
        if force_reentrant is not None
        else bool(is_dtensor_active())
    )

    use_reentrant = ckpt_kwargs.pop("use_reentrant", None)
    preserve_rng_state = ckpt_kwargs.pop("preserve_rng_state", None)
    determinism_check = ckpt_kwargs.pop("determinism_check", None)

    if use_reentrant is None:
        use_reentrant = True
    if require_reentrant:
        use_reentrant = True
    if preserve_rng_state is None:
        preserve_rng_state = True

    ck_opts = {
        k: v
        for k, v in [
            ("use_reentrant", use_reentrant),
            ("preserve_rng_state", preserve_rng_state),
            ("determinism_check", determinism_check),
        ]
        if v is not None
    }

    tried: set[tuple[tuple[str, object], ...]] = set()
    last_type_error: TypeError | None = None

    opts_list: list[dict[str, object]] = [
        ck_opts,
        {k: v for k, v in ck_opts.items() if k != "determinism_check"},
    ]
    if require_reentrant:
        opts_list.extend(
            [{k: v for k, v in ck_opts.items() if k == "use_reentrant"}]
        )
    else:
        opts_list.extend(
            [
                {
                    k: v
                    for k, v in ck_opts.items()
                    if k not in ("determinism_check", "use_reentrant")
                },
                {k: v for k, v in ck_opts.items() if k != "use_reentrant"},
                {},
            ]
        )

    for opts in opts_list:
        key = tuple(sorted(opts.items()))
        if key in tried:
            continue
        tried.add(key)
        try:
            return checkpoint(fn, *args, **opts, **ckpt_kwargs)
        except TypeError as e:
            if require_reentrant and _raised_from_checkpointed_fn(e):
                raise
            last_type_error = e
            continue

    if require_reentrant:
        raise TypeError(
            "DTensor/FSDP2 checkpointing requires `use_reentrant=True`, but torch.utils.checkpoint.checkpoint did not accept a compatible signature in this runtime. Upgrade PyTorch or set STNET_CKPT_REQUIRE_REENTRANT=0 to override."
        ) from last_type_error
    return checkpoint(fn, *args, **ckpt_kwargs)


@torch_compiler_disable(reason="torch.utils.checkpoint", recursive=False)
def checkpoint(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    if _torch_checkpoint is None:
        return fn(*args, **kwargs)
    tl = _CKPT_TL

    def _state(*a: Any, **k: Any) -> Any:
        depth = int(getattr(tl, "depth", 0) or 0)
        setattr(tl, "depth", depth + 1)
        try:
            return fn(*a, **k)
        finally:
            setattr(tl, "depth", depth)

    return _torch_checkpoint(_state, *args, **kwargs)
