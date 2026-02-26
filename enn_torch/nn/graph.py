# -*- coding: utf-8 -*-
from __future__ import annotations

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

import torch
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
from torch import nn
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


def _is_in_jupyter() -> bool:
    try:
        if "ipykernel" in sys.modules:
            return True
        if "google.colab" in sys.modules:
            return True
    except Exception:
        return False
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
        for logger_name in (
            "torch._inductor.utils",
            "torch._inductor",
            "torch",
            "",
        ):
            try:
                logging.getLogger(logger_name).addFilter(flt)
            except Exception:
                pass

        _INDUCTOR_MAX_AUTOTUNE_SMS_FILTERED = True


def _is_compiled_for_inference(model: torch.nn.Module) -> bool:
    cached = getattr(model, "__enn_cached_is_compiled_for_inference__", None)
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
            setattr(model, "__enn_cached_is_compiled_for_inference__", True)
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
            setattr(model, "__enn_cached_is_compiled_for_inference__", True)
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
        setattr(model, "__enn_cached_is_compiled_for_inference__", False)
    except Exception:
        pass
    return False


def _is_aot_autograd_enabled(model: torch.nn.Module) -> bool:
    cached = getattr(model, "__enn_cached_is_aot_autograd_enabled__", None)
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
            setattr(model, "__enn_cached_is_aot_autograd_enabled__", True)
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
        setattr(model, "__enn_cached_is_aot_autograd_enabled__", False)
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
    utils = getattr(torch, "utils", None)
    pd = (
        getattr(utils, "_python_dispatch", None) if utils is not None else None
    )
    fn_stack = getattr(pd, "_get_current_dispatch_mode_stack", None)
    if callable(fn_stack):
        stack = fn_stack()
        return list(stack) if stack is not None else []
    fn_mode = getattr(pd, "_get_current_dispatch_mode", None)
    if callable(fn_mode):
        mode = fn_mode()
        return [mode] if mode is not None else []
    return []


@contextlib.contextmanager
def skip_non_infra_dispatch_mode() -> Iterator[None]:
    try:
        from torch.utils._python_dispatch import _disable_current_modes
        from torch.utils._python_dispatch import is_in_torch_dispatch_mode
    except Exception:
        yield
        return

    active_non_infra = False
    with suppress(Exception):
        active_non_infra = bool(
            is_in_torch_dispatch_mode(include_infra_modes=False)
        )
    if not active_non_infra:
        with suppress(Exception):
            active_non_infra = bool(is_in_torch_dispatch_mode(False))
    if not active_non_infra:
        with suppress(Exception):
            active_non_infra = bool(is_in_torch_dispatch_mode())
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
    comp = getattr(torch, "compiler", None)
    fn = (
        getattr(comp, "is_dynamo_compiling", None)
        if comp is not None
        else None
    )
    if callable(fn) and bool(fn()):
        return True
    dyn = getattr(torch, "_dynamo", None)
    fn = getattr(dyn, "is_dynamo_compiling", None) if dyn is not None else None
    if callable(fn) and bool(fn()):
        return True
    return False


def is_compiling() -> bool:
    dyn = getattr(torch, "_dynamo", None)
    fn = getattr(dyn, "is_compiling", None) if dyn is not None else None
    if callable(fn) and bool(fn()):
        return True
    fn = getattr(dyn, "is_dynamo_compiling", None) if dyn is not None else None
    if callable(fn) and bool(fn()):
        return True

    comp = getattr(torch, "compiler", None)
    fn = getattr(comp, "is_compiling", None) if comp is not None else None
    if callable(fn) and bool(fn()):
        return True
    return False


def is_fake_tensor_mode_active() -> bool:
    subclasses = getattr(torch, "_subclasses", None)
    ft_mod = (
        getattr(subclasses, "fake_tensor", None)
        if subclasses is not None
        else None
    )
    FakeTensorMode = getattr(ft_mod, "FakeTensorMode", None)
    if FakeTensorMode is None:
        return False
    for mode in _dispatch_mode_stack():
        if mode is not None and isinstance(mode, FakeTensorMode):
            return True
    return False


def is_tracing_or_exporting() -> bool:
    jit = getattr(torch, "jit", None)
    if jit is not None and (
        torch.jit.is_tracing() or torch.jit.is_scripting()
    ):
        return True

    comp = getattr(torch, "compiler", None)
    fn = getattr(comp, "is_exporting", None) if comp is not None else None
    if callable(fn) and bool(fn()):
        return True

    onnx = getattr(torch, "onnx", None)
    fn = getattr(onnx, "is_in_onnx_export", None) if onnx is not None else None
    if callable(fn) and bool(fn()):
        return True

    if is_fake_tensor_mode_active():
        return True
    return False


def is_export_or_trace() -> bool:
    return bool(
        is_tracing_or_exporting()
        or is_compiling()
        or is_fake_tensor_mode_active()
    )


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
        "__enn_cached_is_compiled_for_inference__",
        "__enn_cached_is_aot_autograd_enabled__",
        "__enn_cached_is_nvidia_te_available__",
    ):
        with suppress(Exception):
            delattr(model, attr)


def is_nvidia_te_available(model: torch.nn.Module) -> bool:
    cached = getattr(model, "__enn_cached_is_nvidia_te_available__", None)
    if isinstance(cached, bool):
        return cached
    te_flags = (
        getattr(model, "__fp8_inference_te__", False),
        getattr(model, "__fp8_training_te__", False),
        getattr(model, "__te_fp8_default__", False),
    )
    if any(te_flags):
        try:
            setattr(model, "__enn_cached_is_nvidia_te_available__", True)
        except Exception:
            pass
        return True
    for module in model.modules():
        mod_name = getattr(module.__class__, "__module__", "")
        if isinstance(mod_name, str) and mod_name.startswith(
            "transformer_engine"
        ):
            try:
                setattr(model, "__enn_cached_is_nvidia_te_available__", True)
            except Exception:
                pass
            return True
    try:
        setattr(model, "__enn_cached_is_nvidia_te_available__", False)
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
    _suppress_inductor_max_autotune_sms_warning()
    if canonical_mode == "max-autotune" and not _is_for_cuda(module):
        canonical_mode = "max-autotune-no-cudagraphs"
    opt: Dict[str, Any] = dict(options or {})
    if canonical_mode == "max-autotune-no-cudagraphs":
        opt.setdefault("triton.cudagraphs", False)
    options_merged: Dict[str, Any] | None = opt or None
    _inductor_config = _get_inductor_config()
    _restore_inductor: Dict[str, Any] | None = None
    _scoped_inductor_overrides: Dict[str, Any] | None = None
    if _inductor_config is not None:
        with _INDUCTOR_CONFIG_LOCK:
            try:
                if (
                    getattr(_inductor_config, "compile_threads", None)
                    is not None
                ):
                    override = env_first(
                        (
                            "ENN_INDUCTOR_COMPILE_THREADS",
                            "ENN_COMPILE_THREADS",
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
                                "ENN_LOCAL_WORLD_SIZE",
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
                                1, min(2, int(per_rank) // 2)
                            )
            except Exception:
                pass
            if canonical_mode in {
                "max-autotune",
                "max-autotune-no-cudagraphs",
            }:
                _restore_inductor = {}
                _scoped_inductor_overrides = {}

                def _snapshot(attr: str) -> None:
                    if hasattr(_inductor_config, attr):
                        with suppress(Exception):
                            _restore_inductor[attr] = getattr(
                                _inductor_config, attr
                            )

                for _attr in (
                    "autotune_in_subproc",
                    "autotune_local_cache",
                    "autotune_remote_cache",
                    "max_autotune_gemm_search_space",
                    "max_autotune_pointwise",
                    "max_autotune_gemm",
                    "compile_threads",
                ):
                    _snapshot(_attr)

                def _want(attr: str, value: Any) -> None:
                    if hasattr(_inductor_config, attr):
                        _scoped_inductor_overrides[attr] = value

                default_subproc = not _is_in_jupyter()
                want_subproc = env_bool(
                    (
                        "ENN_INDUCTOR_AUTOTUNE_IN_SUBPROC",
                        "ENN_AUTOTUNE_IN_SUBPROC",
                        "TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC",
                    ),
                    default=default_subproc,
                )
                with suppress(Exception):
                    _inductor_config.autotune_in_subproc = bool(want_subproc)
                _want("autotune_in_subproc", bool(want_subproc))
                with suppress(Exception):
                    _inductor_config.autotune_local_cache = True
                _want("autotune_local_cache", True)
                with suppress(Exception):
                    _inductor_config.autotune_remote_cache = None
                _want("autotune_remote_cache", None)
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
                        _want("max_autotune_gemm_search_space", "DEFAULT")
                with suppress(Exception):
                    if (
                        getattr(
                            _inductor_config, "max_autotune_pointwise", None
                        )
                        is not None
                    ):
                        _inductor_config.max_autotune_pointwise = False
                        _want("max_autotune_pointwise", False)
                with suppress(Exception):
                    if (
                        getattr(_inductor_config, "max_autotune_gemm", None)
                        is not None
                    ):
                        _inductor_config.max_autotune_gemm = True
                        _want("max_autotune_gemm", True)
                with suppress(Exception):
                    if (
                        getattr(_inductor_config, "compile_threads", None)
                        is not None
                    ):
                        override_raw = env_first(
                            (
                                "ENN_INDUCTOR_COMPILE_THREADS",
                                "ENN_COMPILE_THREADS",
                                "TORCHINDUCTOR_COMPILE_THREADS",
                            )
                        )
                        override_valid = False
                        if override_raw is not None:
                            with suppress(Exception):
                                int(override_raw)
                                override_valid = True
                        if not override_valid:
                            local_world = env_first_int(
                                (
                                    "ENN_LOCAL_WORLD_SIZE",
                                    "LOCAL_WORLD_SIZE",
                                    "SLURM_NTASKS_PER_NODE",
                                ),
                                1,
                            )
                            threads = 1
                            if int(local_world) <= 1 and _is_in_jupyter():
                                cpu_count = int(CPU.count() or 1)
                                threads = max(1, min(2, int(cpu_count) // 2))
                            _inductor_config.compile_threads = int(threads)
                            _want("compile_threads", int(threads))
    try:
        backend_value = backend
        mode_value: Optional[str] = None
        match canonical_mode:
            case "aot-eager":
                backend_value = "aot_eager"
            case (
                "reduce-overhead"
                | "max-autotune"
                | "max-autotune-no-cudagraphs"
            ):
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
        inductor_cfg = _get_inductor_config()
        patch = (
            getattr(inductor_cfg, "patch", None)
            if inductor_cfg is not None
            else None
        )
        patchable: Dict[str, Any] = {}

        def _has_cfg_key(cfg: Any, key: str) -> bool:
            obj = cfg
            for part in key.split("."):
                if not hasattr(obj, part):
                    return False
                obj = getattr(obj, part)
            return True

        options_dict = (
            dict(compile_kwargs.get("options") or {})
            if isinstance(compile_kwargs.get("options", None), dict)
            else {}
        )
        compile_time_keys = {"triton.cudagraphs", "triton.cudagraph_trees"}
        compile_time_opts = {
            k: v
            for k, v in options_dict.items()
            if isinstance(k, str) and k in compile_time_keys
        }
        if callable(patch) and options_dict:
            patchable = {
                k: v
                for k, v in options_dict.items()
                if isinstance(k, str)
                and (k not in compile_time_keys)
                and _has_cfg_key(inductor_cfg, k)
            }
        strip_options = bool(mode_value is not None)

        if strip_options or patchable:
            if compile_time_opts:
                compile_kwargs["options"] = dict(compile_time_opts)
            else:
                compile_kwargs.pop("options", None)

        with _TORCH_COMPILE_LOCK:
            compiled = compile_fn(module, **compile_kwargs)

        need_scope = bool(_scoped_inductor_overrides) or bool(patchable)
        if not need_scope:
            return compiled

        class _ScopedInductorCompiled(nn.Module):
            def __init__(
                self,
                inner: nn.Module,
                cfg: Any,
                overrides: Dict[str, Any] | None,
                restore: Dict[str, Any] | None,
                patch_fn: Any,
                patch_dict: Dict[str, Any],
            ) -> None:
                super().__init__()
                self._enn_inner = inner
                self._enn_cfg = cfg
                self._enn_overrides = dict(overrides or {})
                self._enn_restore = dict(restore or {})
                self._enn_patch_fn = patch_fn
                self._enn_patch_dict = dict(patch_dict or {})

            def forward(self, *f_args: Any, **f_kwargs: Any) -> Any:
                cfg = self._enn_cfg
                if cfg is None or (
                    not self._enn_overrides and not self._enn_patch_dict
                ):
                    return self._enn_inner(*f_args, **f_kwargs)

                with _INDUCTOR_CONFIG_LOCK:
                    for k, v in self._enn_overrides.items():
                        with suppress(Exception):
                            setattr(cfg, k, v)

                    cm = (
                        self._enn_patch_fn(self._enn_patch_dict)
                        if callable(self._enn_patch_fn)
                        and self._enn_patch_dict
                        else nullcontext()
                    )
                    try:
                        with cm:
                            return self._enn_inner(*f_args, **f_kwargs)
                    finally:
                        for k, v in self._enn_restore.items():
                            with suppress(Exception):
                                setattr(cfg, k, v)

            def __getattr__(self, name: str) -> Any:
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self._enn_inner, name)

            def state_dict(self, *sd_args: Any, **sd_kwargs: Any) -> Any:
                return self._enn_inner.state_dict(*sd_args, **sd_kwargs)

            def load_state_dict(self, *ls_args: Any, **ls_kwargs: Any) -> Any:
                return self._enn_inner.load_state_dict(*ls_args, **ls_kwargs)

        return _ScopedInductorCompiled(
            compiled,
            inductor_cfg,
            _scoped_inductor_overrides,
            _restore_inductor,
            patch,
            patchable,
        )
    finally:
        if _restore_inductor and _inductor_config is not None:
            with _INDUCTOR_CONFIG_LOCK:
                for k, v in _restore_inductor.items():
                    with suppress(Exception):
                        setattr(_inductor_config, k, v)


def torch_compiler_supported() -> bool:
    if env_bool("ENN_TORCH_COMPILE", default=True) is False:
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
    if is_export_or_trace():
        return
    mark_step = getattr(_TORCH_COMPILER, "cudagraph_mark_step_begin", None)
    if callable(mark_step):
        try:
            mark_step()
        except Exception:
            pass


def cudagraph_mark_step_end() -> None:
    if is_export_or_trace():
        return
    mark_step = getattr(_TORCH_COMPILER, "cudagraph_mark_step_end", None)
    if callable(mark_step):
        try:
            mark_step()
        except Exception:
            pass


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
            non_export_wrapped = decorator(fn)
        except Exception:
            return False

        if non_export_wrapped is fn:
            wrapped = fn
        else:
            import functools

            @functools.wraps(fn)
            def wrapped(*a: Any, **kw: Any) -> Any:
                if is_export_or_trace():
                    return fn(*a, **kw)
                return non_export_wrapped(*a, **kw)

        with suppress(Exception):
            setattr(wrapped, _NO_COMPILE_SENTINEL, True)
        try:
            setattr(target, attr, wrapped)
        except Exception:
            return False
        return True
    decorator = _decorate_compiler_disable(reason=reason, recursive=recursive)
    if callable(target):
        fn = target
        non_export_wrapped = decorator(fn)
        if non_export_wrapped is fn:
            return fn
        import functools

        @functools.wraps(fn)
        def wrapped(*a: Any, **kw: Any) -> Any:
            if is_export_or_trace():
                return fn(*a, **kw)
            return non_export_wrapped(*a, **kw)

        return wrapped
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
            "enn_torch.nn.layers",
            "enn_torch.nn.blocks",
            "enn_torch.nn.wrappers",
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
        for mod_name in ("enn_torch.nn.layers", "enn_torch.nn.blocks"):
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
        for mod_name in ("enn_torch.nn.layers", "enn_torch.nn.blocks"):
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


try:
    import torch.utils.checkpoint
except Exception:
    _TORCH_CHECKPOINT = None
else:
    _TORCH_CHECKPOINT = torch.utils.checkpoint.checkpoint

_CKPT_TL = threading.local()


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


def iter_checkpoint(root: nn.Module) -> Iterator[nn.Module]:
    if not isinstance(root, nn.Module):
        return
    for mod in root.modules():
        if hasattr(mod, "_ckpt_min_bytes") and hasattr(mod, "_ckpt_enabled"):
            yield mod


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
    cur_until = int(getattr(inst, "_enn_ckpt_pressure_until", 0) or 0)
    if (
        cur_until >= until
        and int(getattr(inst, "_enn_ckpt_pressure_min_bytes", 0) or 0)
        <= min_bytes
    ):
        return False
    changed = False
    for mod in iter_checkpoint(inst):
        if not hasattr(mod, "_enn_ckpt_saved_min_bytes"):
            with suppress(Exception):
                setattr(
                    mod,
                    "_enn_ckpt_saved_min_bytes",
                    int(getattr(mod, "_ckpt_min_bytes", 0) or 0),
                )
                setattr(
                    mod,
                    "_enn_ckpt_saved_enabled",
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
    with suppress(Exception):
        setattr(inst, "_enn_ckpt_pressure_until", int(max(cur_until, until)))
        prev_mb = int(getattr(inst, "_enn_ckpt_pressure_min_bytes", 0) or 0)
        if prev_mb <= 0:
            setattr(inst, "_enn_ckpt_pressure_min_bytes", int(min_bytes))
        else:
            setattr(
                inst,
                "_enn_ckpt_pressure_min_bytes",
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
    until = int(getattr(inst, "_enn_ckpt_pressure_until", 0) or 0)
    if until <= 0 or step_total < until:
        return
    for mod in iter_checkpoint(inst):
        try:
            if hasattr(mod, "_enn_ckpt_saved_min_bytes"):
                setattr(
                    mod,
                    "_ckpt_min_bytes",
                    int(getattr(mod, "_enn_ckpt_saved_min_bytes", 0) or 0),
                )
            if hasattr(mod, "_enn_ckpt_saved_enabled"):
                setattr(
                    mod,
                    "_ckpt_enabled",
                    bool(getattr(mod, "_enn_ckpt_saved_enabled", True)),
                )
            for k in (
                "_enn_ckpt_saved_min_bytes",
                "_enn_ckpt_saved_enabled",
            ):
                with suppress(Exception):
                    delattr(mod, k)
        except Exception:
            pass
    with suppress(Exception):
        setattr(inst, "_enn_ckpt_pressure_until", 0)
        setattr(inst, "_enn_ckpt_pressure_min_bytes", 0)


def is_checkpoint() -> bool:
    return bool(getattr(_CKPT_TL, "depth", 0) or 0)


def coerce_checkpoint(
    fn: Callable[..., Any],
    *args: Any,
    **ckpt_kwargs: Any,
) -> Any:
    if _TORCH_CHECKPOINT is None:
        return fn(*args)
    if is_export_or_trace() or not any(
        isinstance(a, torch.Tensor) and a.requires_grad for a in args
    ):
        return fn(*args)
    force_reentrant = env_first(("ENN_CKPT_REQUIRE_REENTRANT",), default=None)
    require_reentrant = (
        env_bool("ENN_CKPT_REQUIRE_REENTRANT", default=False)
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
            "DTensor/FSDP2 checkpointing requires `use_reentrant=True`, but torch.utils.checkpoint.checkpoint did not accept a compatible signature in this runtime. Upgrade PyTorch or set ENN_CKPT_REQUIRE_REENTRANT=0 to override."
        ) from last_type_error
    return checkpoint(fn, *args, **ckpt_kwargs)


def checkpoint(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    if _TORCH_CHECKPOINT is None:
        return fn(*args, **kwargs)
    tl = _CKPT_TL

    def _state(*a: Any, **k: Any) -> Any:
        depth = int(getattr(tl, "depth", 0) or 0)
        setattr(tl, "depth", depth + 1)
        try:
            disable_cg = False
            try:
                cfg = get_runtime_cfg()
                prev_cg = getattr(cfg, "compile_cudagraphs", None)
                if prev_cg is None:
                    try:
                        mode = canonicalize_compile_mode(
                            getattr(cfg, "compile_mode", "disabled")
                        )
                    except Exception:
                        mode = "disabled"
                    cg_mode = mode not in {
                        "disabled",
                        "aot-eager",
                        "max-autotune-no-cudagraphs",
                    }

                    def _has_cuda_tensor(obj: Any, _depth: int = 0) -> bool:
                        if _depth > 4:
                            return False
                        if torch.is_tensor(obj):
                            return bool(getattr(obj.device, "type", None) == "cuda")
                        if isinstance(obj, (list, tuple)):
                            return any(_has_cuda_tensor(x, _depth + 1) for x in obj)
                        if isinstance(obj, dict):
                            return any(
                                _has_cuda_tensor(x, _depth + 1)
                                for x in obj.values()
                            )
                        return False

                    prev_cg = bool(cg_mode and (_has_cuda_tensor(a) or _has_cuda_tensor(k)))
                disable_cg = bool(
                    env_bool("ENN_CKPT_DISABLE_CUDAGRAPHS", default=True)
                    and bool(prev_cg)
                    and bool(is_accelerator_available("cuda"))
                )
            except Exception:
                disable_cg = False
            if disable_cg:
                cudagraph_mark_step_begin()
                fn_no_compile = torch_compiler_disable(
                    fn,
                    reason="checkpoint region: disable cudagraphs/compile for safety",
                    recursive=False,
                )
                with runtime_cfg_override(compile_cudagraphs=False):
                    return fn_no_compile(*a, **k)
            return fn(*a, **k)
        finally:
            setattr(tl, "depth", depth)

    return _TORCH_CHECKPOINT(_state, *args, **kwargs)


try:
    _TORCH_DYNAMO = importlib.import_module("torch._dynamo")
except Exception:
    _TORCH_DYNAMO = None


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

    def __init__(
        self: Self,
        steps: Sequence[object],
        *args: Any,
        out_shape: object | None = None,
        name: str | None = None,
        root: nn.Module | None = None,
    ) -> None:
        super().__init__()
        del args
        self._name = str(name or "subgraph")
        self._owned = nn.ModuleList()
        self._refs_materialized = False
        self._root_ref: weakref.ReferenceType[nn.Module] | None = (
            weakref.ref(root) if root is not None else None
        )
        self._path_cache: dict[str, weakref.ReferenceType[nn.Module]] = {}
        self._path_cache_lock = threading.Lock()
        self._out_shape_kind, self._out_shape_spec = self._normalize_out_shape(
            out_shape
        )
        compiled_steps: list[tuple[object, ...]] = []
        for raw in steps:
            step, extra_args, extra_kwargs = self._parse_step(raw)
            meta: dict[str, Any] | None = None

            if isinstance(step, BorrowedModule):
                if step.name:
                    meta = {"name": str(step.name)}
                compiled_steps.append(
                    (
                        "ref",
                        weakref.ref(step.module),
                        extra_args,
                        extra_kwargs,
                        meta,
                    )
                )
                continue
            if isinstance(step, ModulePath):
                meta = {
                    "path": str(step.path),
                    "name": (str(step.name) if step.name else None),
                }
                compiled_steps.append(
                    ("path", str(step.path), extra_args, extra_kwargs, meta)
                )
                continue
            if isinstance(step, OwnedModule):
                if step.name:
                    meta = {"name": str(step.name)}
                self._owned.append(step.module)
                compiled_steps.append(
                    (
                        "owned",
                        len(self._owned) - 1,
                        extra_args,
                        extra_kwargs,
                        meta,
                    )
                )
                continue
            if isinstance(step, nn.Module):
                compiled_steps.append(
                    ("ref", weakref.ref(step), extra_args, extra_kwargs, meta)
                )
                continue
            if callable(step):
                tag = getattr(step, self._CONTROL_ATTR, None)
                if tag is not None:
                    meta = {"control": str(tag)}
                compiled_steps.append(
                    ("fn", step, extra_args, extra_kwargs, meta)
                )
                continue
            raise TypeError(
                f"Unsupported GraphSequential step: {type(step)!r}"
            )
        if not compiled_steps:
            raise ValueError("GraphSequential requires at least one step.")
        self._steps: list[tuple[object, ...]] = compiled_steps

    @staticmethod
    def ref(
        module: nn.Module, *args: Any, name: str | None = None
    ) -> BorrowedModule:
        del args
        return BorrowedModule(module=module, name=name)

    @staticmethod
    def own(
        module: nn.Module, *args: Any, name: str | None = None
    ) -> OwnedModule:
        del args
        return OwnedModule(module=module, name=name)

    @staticmethod
    def path(path: str, *args: Any, name: str | None = None) -> ModulePath:
        return ModulePath(path=str(path), name=name)

    @staticmethod
    def mean(dim: int = 1, *args: Any, keepdim: bool = False) -> OwnedModule:
        del args
        return OwnedModule(
            module=ReduceMean(dim=int(dim), keepdim=bool(keepdim)), name="mean"
        )

    @staticmethod
    def io(*args: Any, **kwargs: Any) -> CallArguments:
        return CallArguments(args=tuple(args), kwargs=dict(kwargs))

    @staticmethod
    def _tag_control(fn: Callable[..., Any], tag: str) -> Callable[..., Any]:
        try:
            setattr(fn, GraphSequential._CONTROL_ATTR, str(tag))
        except Exception:
            pass
        return fn

    @staticmethod
    def break_graph() -> Callable[..., Any]:
        def _op(*a: Any, **kw: Any) -> Any:
            graph_break()
            if kw:
                return CallArguments(args=tuple(a), kwargs=dict(kw))
            if len(a) == 1:
                return a[0]
            return tuple(a)

        return GraphSequential._tag_control(_op, "graph_break")

    @staticmethod
    def cudagraph_begin(
        *args: Any, disable_compile: bool = True
    ) -> Callable[..., Any]:
        def _op(*a: Any, **kw: Any) -> Any:
            cudagraph_mark_step_begin()
            if kw:
                return CallArguments(args=tuple(a), kwargs=dict(kw))
            if len(a) == 1:
                return a[0]
            return tuple(a)

        _op = GraphSequential._tag_control(_op, "cudagraph_begin")
        return (
            torch_compiler_disable(
                _op, reason="subgraph:cudagraph_begin", recursive=False
            )
            if disable_compile
            else _op
        )

    @staticmethod
    def cudagraph_end(
        *args: Any, disable_compile: bool = True
    ) -> Callable[..., Any]:
        def _op(*a: Any, **kw: Any) -> Any:
            cudagraph_mark_step_end()
            if kw:
                return CallArguments(args=tuple(a), kwargs=dict(kw))
            if len(a) == 1:
                return a[0]
            return tuple(a)

        _op = GraphSequential._tag_control(_op, "cudagraph_end")
        return (
            torch_compiler_disable(
                _op, reason="subgraph:cudagraph_end", recursive=False
            )
            if disable_compile
            else _op
        )

    @staticmethod
    def no_compile(
        step: nn.Module | Callable[..., Any],
        *args: Any,
        reason: str | None = None,
        recursive: bool = False,
    ) -> Callable[..., Any]:
        if isinstance(step, nn.Module):
            ref = weakref.ref(step)

            def _call(*a: Any, **kw: Any) -> Any:
                mod = ref()
                if mod is None:
                    raise RuntimeError(
                        "A shared submodule reference was cleared before GraphSequential.forward()."
                    )
                return mod(*a, **kw)

        else:

            def _call(*a: Any, **kw: Any) -> Any:
                return step(*a, **kw)

        wrapped = torch_compiler_disable(
            _call,
            reason=str(reason or "subgraph:no_compile"),
            recursive=bool(recursive),
        )
        return GraphSequential._tag_control(wrapped, "no_compile")

    @staticmethod
    def checkpoint(
        step: nn.Module | Callable[..., Any],
        *args: Any,
        use_reentrant: bool | None = None,
        preserve_rng_state: bool | None = None,
        determinism_check: str | None = None,
    ) -> Callable[..., Any]:
        if isinstance(step, nn.Module):
            ref = weakref.ref(step)

            def _call(*a: Any, **kw: Any) -> Any:
                mod = ref()
                if mod is None:
                    raise RuntimeError(
                        "A shared submodule reference was cleared before GraphSequential.forward()."
                    )

                def _inner(*aa: Any) -> Any:
                    return mod(*aa, **kw)

                return coerce_checkpoint(
                    _inner,
                    *a,
                    use_reentrant=use_reentrant,
                    preserve_rng_state=preserve_rng_state,
                    determinism_check=determinism_check,
                )

        else:

            def _call(*a: Any, **kw: Any) -> Any:
                def _inner(*aa: Any) -> Any:
                    return step(*aa, **kw)

                return coerce_checkpoint(
                    _inner,
                    *a,
                    use_reentrant=use_reentrant,
                    preserve_rng_state=preserve_rng_state,
                    determinism_check=determinism_check,
                )

        return GraphSequential._tag_control(_call, "checkpoint")

    def set_root(self: Self, root: nn.Module | None) -> "GraphSequential":
        self._root_ref = weakref.ref(root) if root is not None else None
        with self._path_cache_lock:
            self._path_cache.clear()
        return self

    def bind(
        self: Self,
        root: nn.Module | None = None,
        *args: Any,
        strict: bool = True,
    ) -> "GraphSequential":
        if root is not None:
            self.set_root(root)
        self._refs_materialized = True
        rebound: list[tuple[object, ...]] = []
        for item in list(self._steps):
            kind, payload, extra_args, extra_kwargs, meta = self._split_step(
                item
            )
            if kind == "path":
                path = str(payload)
                mod = self._resolve_path(path)
                m = dict(meta) if isinstance(meta, dict) else {}
                m["path"] = path
                rebound.append(("ref", mod, extra_args, extra_kwargs, m))
                continue
            if kind == "ref" and payload is None:
                path = meta.get("path") if isinstance(meta, dict) else None
                if isinstance(path, str):
                    mod = self._resolve_path(path)
                    rebound.append(
                        (
                            "ref",
                            mod,
                            extra_args,
                            extra_kwargs,
                            meta,
                        )
                    )
                    continue
                if strict:
                    raise RuntimeError(
                        "GraphSequential.bind() encountered an unresolved ref without a path hint."
                    )
            if kind == "ref" and isinstance(payload, weakref.ReferenceType):
                with contextlib.suppress(Exception):
                    mod = payload()
                    if isinstance(mod, nn.Module):
                        rebound.append(
                            ("ref", mod, extra_args, extra_kwargs, meta)
                        )
                        continue
            rebound.append((kind, payload, extra_args, extra_kwargs, meta))
        self._steps = rebound
        return self

    def forward(self: Self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            cur: Any = CallArguments(args=tuple(args), kwargs=dict(kwargs))
        else:
            cur = args[0] if len(args) == 1 else tuple(args)
        for item in self._steps:
            kind, payload, extra_args, extra_kwargs, meta = self._split_step(
                item
            )
            cur = self._apply_step(
                kind, payload, cur, extra_args, extra_kwargs, meta=meta
            )
        return self._apply_out_shape(cur)

    def extra_repr(self: Self) -> str:
        return f"name={self._name!r}, out_shape={self._out_shape_spec!r}, steps={len(self._steps)}"

    def __getstate__(self: Self) -> dict[str, object]:
        state = super().__getstate__()
        steps = state.get("_steps", [])
        sanitized: list[tuple[object, ...]] = []
        if isinstance(steps, list):
            for item in steps:
                kind, payload, extra_args, extra_kwargs, meta = (
                    self._split_step(item)
                )
                if kind == "ref":
                    sanitized.append(
                        (kind, None, extra_args, extra_kwargs, meta)
                    )
                else:
                    sanitized.append(
                        (kind, payload, extra_args, extra_kwargs, meta)
                    )
            state["_steps"] = sanitized
        state["_root_ref"] = None
        state["_path_cache"] = {}
        state["_path_cache_lock"] = None
        return state

    def __setstate__(self: Self, state: dict[str, object]) -> None:
        super().__setstate__(state)
        if getattr(self, "_path_cache", None) is None:
            self._path_cache = {}
        if getattr(self, "_path_cache_lock", None) is None:
            self._path_cache_lock = threading.Lock()
        if getattr(self, "_root_ref", None) is None:
            self._root_ref = None

    @staticmethod
    def _parse_step(
        raw: object,
    ) -> tuple[object, tuple[Any, ...], dict[str, Any]]:
        if isinstance(raw, (tuple, list)):
            if len(raw) == 2 and isinstance(raw[1], dict):
                return raw[0], (), dict(raw[1])
            if (
                len(raw) == 3
                and isinstance(raw[1], (tuple, list))
                and isinstance(raw[2], dict)
            ):
                return raw[0], tuple(raw[1]), dict(raw[2])
        return raw, (), {}

    @staticmethod
    def _split_step(
        item: object,
    ) -> tuple[str, object, tuple[Any, ...], dict[str, Any], object | None]:
        if not isinstance(item, tuple) or len(item) < 4:
            raise TypeError("Invalid GraphSequential internal step format.")
        kind = str(item[0])
        payload = item[1]
        raw_args = item[2]
        if raw_args is None:
            extra_args: tuple[Any, ...] = ()
        elif isinstance(raw_args, tuple):
            extra_args = raw_args
        else:
            try:
                extra_args = tuple(raw_args)
            except TypeError:
                extra_args = (raw_args,)
        raw_kwargs = item[3]
        if raw_kwargs is None:
            extra_kwargs: dict[str, Any] = {}
        elif isinstance(raw_kwargs, dict):
            extra_kwargs = dict(raw_kwargs)
        else:
            extra_kwargs = (
                dict(raw_kwargs) if hasattr(raw_kwargs, "items") else {}
            )
        meta = item[4] if len(item) >= 5 else None
        return kind, payload, extra_args, extra_kwargs, meta

    @staticmethod
    def _normalize_out_shape(
        out_shape: object | None,
    ) -> tuple[str | None, object | None]:
        if out_shape is None:
            return None, None
        if isinstance(out_shape, dict):
            spec: dict[str, object] = {}
            for k, v in out_shape.items():
                if v is None:
                    spec[str(k)] = None
                else:
                    spec[str(k)] = tuple(int(x) for x in v)
            return "dict", spec
        if (
            isinstance(out_shape, (list, tuple))
            and out_shape
            and isinstance(out_shape[0], (list, tuple, type(None)))
        ):
            shapes: list[object] = []
            for s in out_shape:
                if s is None:
                    shapes.append(None)
                else:
                    shapes.append(tuple(int(x) for x in s))
            return "seq", tuple(shapes)
        return "single", tuple(int(x) for x in out_shape)

    @staticmethod
    def _unpack(value: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if isinstance(value, CallArguments):
            return tuple(value.args), dict(value.kwargs)
        if isinstance(value, tuple):
            return value, {}
        if isinstance(value, list):
            return tuple(value), {}
        if isinstance(value, dict):
            return (), value
        return (value,), {}

    def _resolve_path(self: Self, path: str) -> nn.Module:
        with self._path_cache_lock:
            ref = self._path_cache.get(path)
        if ref is not None:
            mod = ref()
            if mod is not None:
                return mod
        root = self._root_ref() if self._root_ref is not None else None
        if root is None:
            raise RuntimeError(
                "GraphSequential requires `root=` (or set_root()) when using ModulePath steps."
            )
        mod: nn.Module | None = None
        if hasattr(root, "get_submodule"):
            try:
                mod = root.get_submodule(path)
            except Exception:
                mod = None
        if mod is None:
            cur: nn.Module = root
            for part in str(path).split("."):
                child = getattr(cur, "_modules", None)
                if isinstance(child, dict) and part in child:
                    nxt = child.get(part)
                else:
                    nxt = getattr(cur, part, None)
                if not isinstance(nxt, nn.Module):
                    raise AttributeError(
                        f"Failed to resolve submodule path {path!r} at {part!r}."
                    )
                cur = nxt
            mod = cur
        if not isinstance(mod, nn.Module):
            raise TypeError(
                f"get_submodule({path!r}) did not return an nn.Module"
            )
        with self._path_cache_lock:
            self._path_cache[path] = weakref.ref(mod)
        return mod

    def _resolve_path_nocache(self: Self, path: str) -> nn.Module:
        root = self._root_ref() if self._root_ref is not None else None
        if root is None:
            raise RuntimeError(
                "GraphSequential requires `root=` (or set_root()) when using ModulePath steps."
            )
        mod: nn.Module | None = None
        if hasattr(root, "get_submodule"):
            try:
                mod = root.get_submodule(path)
            except Exception:
                mod = None
        if mod is None:
            cur: nn.Module = root
            for part in str(path).split("."):
                child = getattr(cur, "_modules", None)
                if isinstance(child, dict) and part in child:
                    nxt = child.get(part)
                else:
                    nxt = getattr(cur, part, None)
                if not isinstance(nxt, nn.Module):
                    raise AttributeError(
                        f"Failed to resolve submodule path {path!r} at {part!r}."
                    )
                cur = nxt
            mod = cur
        if not isinstance(mod, nn.Module):
            raise TypeError(
                f"get_submodule({path!r}) did not return an nn.Module"
            )
        return mod

    def _apply_step(
        self: Self,
        kind: str,
        payload: object,
        cur: Any,
        extra_args: tuple[Any, ...],
        extra_kwargs: dict[str, Any],
        *args: Any,
        meta: object | None = None,
    ) -> Any:
        args, kwargs = self._unpack(cur)
        if extra_args:
            args = tuple(args) + tuple(extra_args)
        if extra_kwargs:
            merged = dict(kwargs)
            merged.update(extra_kwargs)
            kwargs = merged
        if kind == "ref":
            mod: nn.Module | None = None
            path = meta.get("path") if isinstance(meta, dict) else None
            if isinstance(payload, nn.Module):
                mod = payload
            elif isinstance(payload, weakref.ReferenceType) and (
                (not is_compiling()) or not isinstance(path, str)
            ):
                mod = payload()
            if mod is None:
                if isinstance(path, str):
                    mod = (
                        self._resolve_path_nocache(path)
                        if is_compiling()
                        else self._resolve_path(path)
                    )
                else:
                    raise RuntimeError(
                        "A shared submodule reference was cleared (or not bound) before GraphSequential.forward()."
                    )
            return mod(*args, **kwargs)
        if kind == "owned":
            return self._owned[int(payload)](*args, **kwargs)
        if kind == "path":
            return self._resolve_path(str(payload))(*args, **kwargs)
        return payload(*args, **kwargs)

    def _apply_out_shape(self: Self, out: Any) -> Any:
        kind = self._out_shape_kind
        spec = self._out_shape_spec
        if kind is None or spec is None:
            return out

        def _reshape_one(
            t: torch.Tensor, shape: tuple[int, ...]
        ) -> torch.Tensor:
            if t.ndim == 0:
                raise RuntimeError(
                    "Cannot reshape a scalar output in GraphSequential."
                )
            return t.reshape(t.shape[0], *shape)

        if kind == "single":
            if not isinstance(out, torch.Tensor):
                raise RuntimeError(
                    "out_shape is set but the pipeline output is not a Tensor."
                )
            return _reshape_one(out, spec)
        if kind == "seq":
            if not isinstance(out, (tuple, list)):
                raise RuntimeError(
                    "out_shape expects tuple/list output but got a different type."
                )
            shapes = list(spec)
            if len(out) != len(shapes):
                raise RuntimeError(
                    "out_shape length does not match tuple/list output length."
                )
            out_list = list(out)
            for i, sh in enumerate(shapes):
                if sh is None:
                    continue
                if not isinstance(out_list[i], torch.Tensor):
                    raise RuntimeError(
                        "out_shape expects Tensor outputs in tuple/list."
                    )
                out_list[i] = _reshape_one(out_list[i], sh)
            return tuple(out_list) if isinstance(out, tuple) else out_list
        if not isinstance(out, dict):
            raise RuntimeError(
                "out_shape expects dict output but got a different type."
            )
        out_dict = dict(out)
        for k, sh in spec.items():
            if sh is None:
                continue
            if k not in out_dict:
                raise RuntimeError(
                    f"out_shape missing key in output dict: {k!r}"
                )
            if not isinstance(out_dict[k], torch.Tensor):
                raise RuntimeError(
                    "out_shape expects Tensor values in dict output."
                )
            out_dict[k] = _reshape_one(out_dict[k], sh)
        return out_dict

    def extract_for_serving(
        self: Self,
        *args: Any,
        root: nn.Module | None = None,
        clone_modules: bool = True,
        strip_control_ops: bool = True,
        name: str | None = None,
    ) -> "GraphSequential":
        import copy

        if root is not None:
            self.set_root(root)
        steps_out: list[object] = []
        for item in list(self._steps):
            kind, payload, extra_args, extra_kwargs, meta = self._split_step(
                item
            )
            if kind == "fn":
                fn = payload
                if strip_control_ops and bool(
                    getattr(fn, self._CONTROL_ATTR, "")
                ):
                    continue
                step_obj: object = fn
            else:
                mod: nn.Module | None = None
                if kind == "owned":
                    mod = self._owned[int(payload)]
                elif kind == "path":
                    mod = self._resolve_path(str(payload))
                elif kind == "ref":
                    path = meta.get("path") if isinstance(meta, dict) else None
                    if isinstance(payload, nn.Module):
                        mod = payload
                    elif isinstance(payload, weakref.ReferenceType):
                        mod = payload()
                    else:
                        mod = None
                    if mod is None and isinstance(path, str):
                        mod = self._resolve_path(str(path))
                else:
                    raise TypeError(
                        f"Unknown GraphSequential step kind: {kind!r}"
                    )
                if not isinstance(mod, nn.Module):
                    raise RuntimeError(
                        f"extract_for_serving could not resolve module for step kind={kind!r}"
                    )
                if clone_modules:
                    try:
                        mod = copy.deepcopy(mod)
                    except Exception as e:
                        warnings.warn(
                            f"GraphSequential.extract_for_serving: deepcopy failed for {mod.__class__.__name__}: {e}. "
                            "Falling back to sharing the original module object.",
                            RuntimeWarning,
                        )
                step_obj = OwnedModule(module=mod)
            if extra_args and extra_kwargs:
                steps_out.append((step_obj, extra_args, extra_kwargs))
            elif extra_args:
                steps_out.append((step_obj, extra_args, {}))
            elif extra_kwargs:
                steps_out.append((step_obj, extra_kwargs))
            else:
                steps_out.append(step_obj)
        out = GraphSequential(
            steps=steps_out,
            out_shape=(
                self._out_shape_spec
                if self._out_shape_kind is not None
                else None
            ),
            name=str(name or f"{self._name}_serving"),
            root=None,
        )
        out.eval()
        with contextlib.suppress(Exception):
            out.requires_grad_(False)
        with contextlib.suppress(Exception):
            setattr(out, "__compiled_for_serving__", True)
        return out
