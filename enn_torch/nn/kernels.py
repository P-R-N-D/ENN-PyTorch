# -*- coding: utf-8 -*-
from __future__ import annotations

# =============================================================================
# 1. Standard Library Imports
# =============================================================================
import contextlib
import inspect
import math
import os
import threading
import time
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Self, Tuple

# =============================================================================
# 2. Third-Party Imports
# =============================================================================
import torch
import torch._dynamo
from torch import nn

# =============================================================================
# 3. Local Imports
# =============================================================================
from ..core.datatypes import env_bool, env_int, env_str
from ..core.system import (
    get_device,
    get_dpa_backends,
    get_runtime_cfg,
    get_runtime_config,
)
from ..core.tensor import is_meta_or_fake_tensor
from .graph import (
    assert_trace,
    canonicalize_compile_mode,
    compile as _model_compile,
    cudagraph_mark_step_begin,
    is_checkpoint,
    is_compiling,
    is_dynamo_compiling,
    is_export_or_trace,
    is_symbolic,
    is_tracing_or_exporting,
    skip_non_infra_dispatch_mode,
    torch_compiler_disable,
    torch_compiler_supported,
)
from .profiler import FLOP_PROFILER, capture


# =============================================================================
# Lazy Imports & Environment Stubs
# =============================================================================
try:
    import triton
    import triton.language as tl

    _HAS_TRITON_LIB = True
    _HAS_TRITON_MSR = bool(torch.cuda.is_available())
except Exception:
    _HAS_TRITON_LIB = False
    _HAS_TRITON_MSR = False

    class _TritonStub:
        def jit(
            self: Self, fn: Callable[..., object] | None = None, **kwargs: object
        ) -> Callable[..., object]:
            return fn if fn else lambda f: f

        @staticmethod
        def cdiv(a: int, b: int) -> int:
            return (int(a) + int(b) - 1) // max(1, int(b))

    class _TLStub:
        constexpr = object()

    triton = _TritonStub()
    tl = _TLStub()

try:
    _FLEX_KWARGS: set[str] = set()
    _FLEX_KWARGS_LOCK = threading.Lock()
    from torch.nn.attention.flex_attention import create_mask as _torch_create_mask
    from torch.nn.attention.flex_attention import flex_attention as _torch_flex_attention

    _HAS_TORCH_FLEX = True
    with contextlib.suppress(Exception):
        _FLEX_KWARGS = set(inspect.signature(_torch_flex_attention).parameters.keys())
except Exception:
    _torch_create_mask = None
    _torch_flex_attention = None
    _HAS_TORCH_FLEX = False
    _FLEX_KWARGS = set()

_HAS_TE = False
te = None
if torch.cuda.is_available() and getattr(get_device(), "type", "cpu") == "cuda":
    try:
        import transformer_engine.pytorch as te
        _HAS_TE = True
    except Exception:
        _HAS_TE = False


# =============================================================================
# Kernel State Management
# =============================================================================
class KernelFailure(RuntimeError):
    pass


@dataclass
class _KernelState:
    dead: bool = False
    fail_count: int = 0
    first_ts: float = 0.0
    last_ts: float = 0.0
    last_exc_type: str = ""
    last_exc_msg: str = ""


def _now() -> float:
    try:
        return time.time()
    except Exception:
        return 0.0


def _sanitize(out: Any) -> Any:
    match out:
        case torch.Tensor() if out.is_floating_point() and out.numel() > 0:
            try:
                return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                return out
        case _:
            return out


class KernelManager:
    def __init__(self: Self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, _KernelState] = {}

    def enabled(self: Self) -> bool:
        return bool(env_bool("ENN_KERNEL_MANAGER_ENABLE", default=True))

    def _fail_threshold(self: Self) -> int:
        return max(1, int(env_int("ENN_KERNEL_MANAGER_FAIL_THRESHOLD", 3)))

    def is_dead(self: Self, key: str) -> bool:
        if (not key) or (not self.enabled()):
            return False
        with self._lock:
            st = self._state.get(key)
            return bool(st.dead) if st is not None else False

    def mark_dead(
        self: Self, key: str, *, exc: BaseException | None = None, reason: str | None = None
    ) -> None:
        if (not key) or (not self.enabled()):
            return
        ts = _now()
        with self._lock:
            st = self._state.setdefault(key, _KernelState())
            st.dead = True
            st.fail_count = max(1, int(st.fail_count) + 1)
            st.last_ts = ts
            if st.first_ts <= 0.0:
                st.first_ts = ts
            if exc is not None:
                st.last_exc_type = type(exc).__name__
                st.last_exc_msg = str(exc)
            elif reason is not None:
                st.last_exc_type = "KernelFailure"
                st.last_exc_msg = str(reason)

    def note_failure(
        self: Self, key: str, exc: BaseException, *, sticky: bool = True
    ) -> None:
        if (not key) or (not self.enabled()):
            return
        ts = _now()
        with self._lock:
            st = self._state.setdefault(key, _KernelState())
            st.fail_count += 1
            st.last_ts = ts
            if st.first_ts <= 0.0:
                st.first_ts = ts
            st.last_exc_type = type(exc).__name__
            st.last_exc_msg = str(exc)
            if sticky or st.fail_count >= self._fail_threshold():
                st.dead = True

    def run(
        self: Self,
        key: str,
        fn_main: Callable[..., Any],
        *args: Any,
        device: torch.device | None = None,
        master_dtype: torch.dtype | None = None,
        fn_safe: Optional[Callable[..., Any]] = None,
        validate: Optional[Callable[[Any], bool]] = None,
        sticky: bool = True,
        safe_on_exception: bool = True,
        sanitize_dead: bool = False,
        **kwargs: Any,
    ) -> Any:
        del device, master_dtype, kwargs

        if (not key) or (not self.enabled()):
            return fn_main(*args)

        if self.is_dead(key):
            if fn_safe is None:
                raise KernelFailure(f"Kernel disabled: {key}")
            out = fn_safe(*args)
            return _sanitize(out) if sanitize_dead else out

        try:
            out = fn_main(*args)
        except Exception as exc:
            self.note_failure(key, exc, sticky=sticky)
            if safe_on_exception and fn_safe is not None:
                try:
                    out2 = fn_safe(*args)
                except Exception as exc2:
                    self.note_failure(key, exc2, sticky=True)
                    raise KernelFailure(f"Kernel '{key}' failed and safe fallback also failed: {type(exc2).__name__}: {exc2}") from exc2
                return _sanitize(out2) if sanitize_dead else out2
            raise

        if validate is not None:
            ok = False
            try:
                ok = bool(validate(out))
            except Exception as exc:
                self.note_failure(key, exc, sticky=sticky)
            if not ok:
                self.note_failure(key, RuntimeError("output validation failed"), sticky=sticky)
                if fn_safe is not None:
                    try:
                        out2 = fn_safe(*args)
                    except Exception as exc2:
                        self.note_failure(key, exc2, sticky=True)
                        raise KernelFailure(f"Kernel '{key}' produced invalid output and safe fallback failed: {type(exc2).__name__}: {exc2}") from exc2
                    return _sanitize(out2) if sanitize_dead else out2
                if sanitize_dead:
                    return _sanitize(out)
                raise KernelFailure(f"Kernel '{key}' produced invalid output")

        return out

    def try_run(
        self: Self,
        key: str,
        fn_main: Callable[..., Any],
        *args: Any,
        validate: Optional[Callable[[Any], bool]] = None,
        sticky: bool = True,
    ) -> tuple[bool, Any]:
        try:
            out = self.run(key, fn_main, *args, validate=validate, sticky=sticky, safe_on_exception=False, fn_safe=None)
            return True, out
        except Exception:
            return False, None

    def dead_keys(self: Self) -> list[str]:
        if not self.enabled():
            return []
        with self._lock:
            return sorted([k for (k, st) in self._state.items() if st.dead])


_KERNEL_MANAGER_LOCK = threading.Lock()
_KERNEL_MANAGER_SINGLETON: KernelManager | None = None


def get_kernel_manager() -> KernelManager:
    global _KERNEL_MANAGER_SINGLETON
    if _KERNEL_MANAGER_SINGLETON is not None:
        return _KERNEL_MANAGER_SINGLETON
    with _KERNEL_MANAGER_LOCK:
        if _KERNEL_MANAGER_SINGLETON is None:
            _KERNEL_MANAGER_SINGLETON = KernelManager()
    return _KERNEL_MANAGER_SINGLETON


_FLEX_KERNEL_SINGLETON_LOCK = threading.Lock()
_FLEX_KERNEL_SINGLETON = None


def get_flex_kernel() -> "FlexAttention":
    global _FLEX_KERNEL_SINGLETON
    if _FLEX_KERNEL_SINGLETON is not None:
        return _FLEX_KERNEL_SINGLETON
    with _FLEX_KERNEL_SINGLETON_LOCK:
        if _FLEX_KERNEL_SINGLETON is None:
            _FLEX_KERNEL_SINGLETON = FlexAttention(prefer_torch=True)
    return _FLEX_KERNEL_SINGLETON


# =============================================================================
# FlexAttention Global State & Helpers
# =============================================================================
_FLEX_ATTN_COMPILED: dict[str, Any] = {}
_FLEX_ATTN_COMPILE_LOCK = threading.Lock()
_FLEX_ATTN_BASE_COMPILED: dict[tuple[Any, ...], Any] = {}
_FLEX_ATTN_BASE_COMPILE_LOCK = threading.Lock()
_FLEX_ATTN_VERIFIED: set[tuple[Any, ...]] = set()
_FLEX_ATTN_VERIFIED_LOCK = threading.Lock()
_FLEX_ATTN_UNCOMPILED_NEEDLE = "flex_attention called without torch.compile"
_FLEX_UNCOMPILED_WARN_RE = r"(?s).*flex_attention called without torch\.compile.*"
_FLEX_UNCOMPILED_WARN_MODULE_RE = r"torch\.nn\.attention\.flex_attention"
_FLEX_ATTN_SPECIALIZED: dict[tuple[Any, ...], Any] = {}
_FLEX_ATTN_SPECIALIZE_LOCK = threading.Lock()
_FLEX_ATTN_FAILED: dict[tuple[Any, ...], str] = {}
_FLEX_ATTN_WARNED: set[str] = set()
_FLEX_ATTN_RESOURCE_KOPTS: dict[tuple[Any, ...], dict[str, Any]] = {}
_FLEX_ATTN_RESOURCE_KOPTS_LOCK = threading.Lock()
_FLEX_ATTN_FUSED_OK_KEYS: set[str] = set()
_FLEX_ATTN_FUSED_OK_LOCK = threading.Lock()
_FLEX_UNCOMPILED_SUPPRESS_LOCK = threading.Lock()
_FLEX_UNCOMPILED_SUPPRESS_INSTALLED = False
_FLEX_UNCOMPILED_SUPPRESS_REPORTED = False


def _flex_attention_disabled() -> bool:
    return bool(env_bool("ENN_DISABLE_FLEX_ATTENTION", False))


def _flex_attention_compile_mode() -> str:
    cfg = get_runtime_cfg()
    global_mode = canonicalize_compile_mode(getattr(cfg, "compile_mode", "disabled"))
    if not bool(getattr(cfg, "compile_cudagraphs", True)):
        if global_mode in {"max-autotune", "reduce-overhead"}:
            global_mode = "max-autotune-no-cudagraphs"
            
    match global_mode:
        case "disabled" | "aot-eager": return "aot-eager"
        case "reduce-overhead": return "reduce-overhead"
        case "max-autotune" | "max-autotune-no-cudagraphs": return global_mode
        case _: return "reduce-overhead"


def _warn_once(key: str, message: str) -> None:
    if key in _FLEX_ATTN_WARNED: return
    _FLEX_ATTN_WARNED.add(key)
    with contextlib.suppress(Exception): warnings.warn(str(message), stacklevel=3)


def _warn_fused_ok_throttled(*, mode_key: str, q: torch.Tensor, flex_kwargs: dict[str, Any], dyn_key: Any) -> None:
    if not _flex_debug_enabled(): return
    bm = flex_kwargs.get("block_mask", None)
    mm = getattr(bm, "mask_mod", None)
    ko = flex_kwargs.get("kernel_options", None)
    key = f"{mode_key}|{str(getattr(q,'dtype',None))}|{tuple(getattr(q,'shape',()))}|{repr(ko)}|{type(mm).__name__}"
    
    with _FLEX_ATTN_FUSED_OK_LOCK:
        if key in _FLEX_ATTN_FUSED_OK_KEYS: return
        _FLEX_ATTN_FUSED_OK_KEYS.add(key)
        
    smod = flex_kwargs.get("score_mod", None)
    _warn_flex_debug_once(
        f"flexattn-fused-ok-{hash(key)}",
        f"FlexAttention debug: compiled+FUSED OK; mode={mode_key!r} dynamic={dyn_key!r} "
        f"q={tuple(getattr(q,'shape',()))} dtype={getattr(q,'dtype',None)} passed_keys={sorted(list(flex_kwargs.keys()))} "
        f"kernel_options={ko} block_mask={type(bm).__name__}@{id(bm):x} score_mod={type(smod).__name__}@{id(smod):x} "
        f"mask_mod={type(mm).__name__}@{id(mm):x} dilation={getattr(mm,'dilation',None)} win={getattr(mm,'win',None)} causal={getattr(mm,'causal',None)}",
    )


def _ensure_pythonwarnings_suppresses_flex_uncompiled() -> None:
    if env_bool("ENN_FLEX_SUPPRESS_UNCOMPILED_WARNING", True) is False: return
    rule = "ignore:.*flex_attention called without torch\\.compile.*:UserWarning:torch\\.nn\\.attention\\.flex_attention"
    cur = os.environ.get("PYTHONWARNINGS", "")
    if rule in cur: return
    os.environ["PYTHONWARNINGS"] = rule if not cur else f"{cur},{rule}"
    if env_bool("ENN_FLEX_DEBUG_KOPTS", False):
        _warn_once("flexattn-debug-pythonwarnings", f"FlexAttention debug: appended PYTHONWARNINGS filter for flex_attention uncompiled warning (PYTHONWARNINGS={os.environ.get('PYTHONWARNINGS','')!r})")


def _stack_hint(limit: int = 32, keep_last: int = 14) -> str:
    try: st = traceback.format_stack(limit=limit)
    except Exception: return ""
    keep = [s.rstrip() for s in st if any(k in s for k in ("enn_torch", "flex_attention", "torch.compile", "torch/_inductor"))]
    keep = keep[-keep_last:] if keep else [x.rstrip() for x in st[-keep_last:]]
    return "\n".join(keep).rstrip()


def _flex_strict_fused_enabled() -> bool:
    return bool(env_bool("ENN_FLEX_STRICT_FUSED", False) or env_bool("ENN_FLEX_ASSERT_FUSED", False))


def _install_flex_uncompiled_warning_suppression() -> None:
    global _FLEX_UNCOMPILED_SUPPRESS_INSTALLED, _FLEX_UNCOMPILED_SUPPRESS_REPORTED
    if env_bool("ENN_FLEX_SUPPRESS_UNCOMPILED_WARNING", True) is False or _FLEX_UNCOMPILED_SUPPRESS_INSTALLED:
        return
    with _FLEX_UNCOMPILED_SUPPRESS_LOCK:
        if _FLEX_UNCOMPILED_SUPPRESS_INSTALLED: return
        _ensure_pythonwarnings_suppresses_flex_uncompiled()
        with contextlib.suppress(Exception):
            warnings.filterwarnings("ignore", message=_FLEX_UNCOMPILED_WARN_RE, category=UserWarning, module=_FLEX_UNCOMPILED_WARN_MODULE_RE)

        orig = warnings.showwarning
        def _showwarning(message: Any, category: type[Warning], filename: str, lineno: int, file: Any = None, line: str | None = None) -> None:
            nonlocal orig
            global _FLEX_UNCOMPILED_SUPPRESS_REPORTED
            msg = str(message) if message else ""
            if _FLEX_ATTN_UNCOMPILED_NEEDLE in msg.lower():
                if env_bool("ENN_FLEX_DEBUG_KOPTS", False) and not _FLEX_UNCOMPILED_SUPPRESS_REPORTED:
                    _FLEX_UNCOMPILED_SUPPRESS_REPORTED = True
                    hint, flag, phase = _stack_hint(), _torch_flex_disable_compile_debug_value(), "compile" if is_dynamo_compiling() else "runtime"
                    with contextlib.suppress(Exception):
                        text = f"FlexAttention debug: suppressed PyTorch 'called without torch.compile' warning (phase={phase}, origin={filename}:{lineno}, torch_flag={flag!r})."
                        orig(UserWarning(f"{text}\n{hint}" if hint else text), UserWarning, __file__, 0, file=file, line=None)
                return
            return orig(message, category, filename, lineno, file=file, line=line)
            
        warnings.showwarning = _showwarning
        _FLEX_UNCOMPILED_SUPPRESS_INSTALLED = True


def _torch_flex_attention_module() -> Any | None:
    with contextlib.suppress(Exception):
        import torch.nn.attention.flex_attention as _fa
        return _fa
    return None


def _torch_flex_disable_compile_debug_value() -> Any:
    fa = _torch_flex_attention_module()
    if fa is None: return None
    with contextlib.suppress(Exception): return getattr(fa, "_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG", None)
    return None


def _force_enable_torch_flex_compile() -> None:
    if env_bool("ENN_FLEX_RESPECT_TORCH_DISABLE_COMPILE_DEBUG", False): return
    fa = _torch_flex_attention_module()
    if fa is None: return
    before = _torch_flex_disable_compile_debug_value()
    with contextlib.suppress(Exception):
        if bool(getattr(fa, "_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG", False)): setattr(fa, "_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG", False)
    after = _torch_flex_disable_compile_debug_value()
    if env_bool("ENN_FLEX_DEBUG_KOPTS", False) and before != after:
        _warn_once("flexattn-debug-torch-flag", f"FlexAttention debug: flipped torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG from {before!r} to {after!r}")


def _flex_debug_enabled() -> bool:
    return bool(env_bool("ENN_FLEX_DEBUG_KOPTS", False))


def _flex_retry_failed_enabled() -> bool:
    try: dbg = bool(env_bool("ENN_FLEX_DEBUG_KOPTS", False))
    except Exception: dbg = False
    return bool(env_bool("ENN_FLEX_RETRY_FAILED", default=dbg))


def _ensure_flex_kwargs_initialized() -> None:
    if _FLEX_KWARGS or _torch_flex_attention is None: return
    with _FLEX_KWARGS_LOCK:
        if _FLEX_KWARGS: return
        with contextlib.suppress(Exception):
            _FLEX_KWARGS.update(inspect.signature(_torch_flex_attention).parameters.keys())
    if not _FLEX_KWARGS and _flex_debug_enabled():
        _warn_once("flexattn-debug-empty-kwargs", "FlexAttention debug: _FLEX_KWARGS is empty; kernel_options/env tuning will not apply.")


def _env_bool_optional(name: str) -> Optional[bool]:
    if name not in os.environ: return None
    try: return bool(env_bool(name, False))
    except Exception: return None


def _flex_attention_dynamic_flag(mode: str) -> Optional[bool]:
    for key in ("ENN_FLEX_ATTENTION_DYNAMIC", "ENN_FLEX_COMPILE_DYNAMIC", "ENN_FLEXATTN_DYNAMIC"):
        if isinstance(v := _env_bool_optional(key), bool): return v
    if isinstance(dyn := getattr(get_runtime_cfg(), "compile_dynamic", None), bool): return dyn
    if str(mode) in {"max-autotune", "max-autotune-no-cudagraphs"}: return True
    return None


def _flex_attention_fallback_modes(mode: str) -> tuple[str, ...]:
    match str(mode):
        case "max-autotune": return ("max-autotune-no-cudagraphs", "reduce-overhead")
        case "max-autotune-no-cudagraphs": return ()
        case _: return ()


def _coerce_flex_fallback_mode(mode: str) -> str:
    fallback_mode = canonicalize_compile_mode(str(mode))
    if fallback_mode in {"disabled", "aot-eager"}: return "max-autotune-no-cudagraphs"
    if not bool(getattr(get_runtime_cfg(), "compile_cudagraphs", True)) and fallback_mode == "reduce-overhead": return "max-autotune-no-cudagraphs"
    return fallback_mode


def _flex_attention_cache_key(*args: Any, mode: str, dynamic: Optional[bool], device: torch.device, dtype: torch.dtype, flex_kwargs: Mapping[str, Any]) -> tuple[Any, ...]:
    sm = flex_kwargs.get("score_mod", None)
    bm = flex_kwargs.get("block_mask", None)
    ko = flex_kwargs.get("kernel_options", None)
    scale = flex_kwargs.get("scale", None)
    gqa = flex_kwargs.get("enable_gqa", None)
    lse = flex_kwargs.get("return_lse", None)
    drop = flex_kwargs.get("dropout_p", flex_kwargs.get("dropout", None))
    keys = tuple(sorted(str(k) for k in flex_kwargs.keys()))
    return (
        "flexattn", str(mode), bool(dynamic) if dynamic is not None else dynamic, str(device), str(dtype), keys,
        int(id(sm)) if sm is not None else 0, int(id(bm)) if bm is not None else 0, int(id(ko)) if ko is not None else 0,
        float(scale) if isinstance(scale, (int, float)) else None, float(drop) if isinstance(drop, (int, float)) else None,
        bool(gqa) if gqa is not None else None, bool(lse) if lse is not None else None,
    )


def _compile_flex_attention_wrapper(*args: Any, mode: str, dynamic: Optional[bool], flex_kwargs: dict[str, Any]) -> Any:
    if _torch_flex_attention is None: raise RuntimeError("Flex Attention is not available")
    _install_flex_uncompiled_warning_suppression()
    frozen = dict(flex_kwargs)
    _force_enable_torch_flex_compile()
    dyn_key = bool(dynamic) if dynamic is not None else dynamic
    base_key = ("flexattn-base", str(mode), dyn_key)
    
    base = _FLEX_ATTN_BASE_COMPILED.get(base_key)
    if base is None:
        with _FLEX_ATTN_BASE_COMPILE_LOCK:
            if (base := _FLEX_ATTN_BASE_COMPILED.get(base_key)) is None:
                with skip_non_infra_dispatch_mode(), warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=_FLEX_UNCOMPILED_WARN_RE, category=UserWarning, module=_FLEX_UNCOMPILED_WARN_MODULE_RE)
                    base, saw = _call_with_flex_warn_guard(lambda: _model_compile(_torch_flex_attention, mode=mode, dynamic=dynamic, fullgraph=False))
                    if saw and _flex_debug_enabled():
                        _warn_once(f"flexattn-debug-compile-warn-{hash(base_key)}", f"FlexAttention debug: PyTorch emitted an 'uncompiled' warning during base compile (mode={mode!r}, dynamic={dynamic!r}, torch_flag={_torch_flex_disable_compile_debug_value()!r}).")
                _FLEX_ATTN_BASE_COMPILED[base_key] = base
                
    if base is _torch_flex_attention: return _torch_flex_attention

    def _wrapped(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Any:
        if bool(is_checkpoint()) and getattr(q.device, "type", None) == "cuda" and _flex_attention_compile_mode() in {"max-autotune", "reduce-overhead"}:
            cudagraph_mark_step_begin()
        with skip_non_infra_dispatch_mode():
            return _flex_ckpt_always_clone_out(base(q, k, v, **frozen))

    return _wrapped


def _is_compile_failure(exc: BaseException) -> bool:
    qual = f"{getattr(type(exc), '__module__', '')}.{getattr(type(exc), '__name__', '')}"
    msg = str(exc)
    needles = ("torch._dynamo", "torch._inductor", "BackendCompilerFailed", "LoweringException", "Unsupported", "CompileError")
    return any(n in qual for n in needles) or any(n in msg for n in needles)


def _call_with_flex_warn_guard(fn: Callable[[], Any]) -> tuple[Any, bool]:
    saw_uncompiled = False
    orig_showwarning = warnings.showwarning

    def _showwarning(message: Any, category: type[Warning], filename: str, lineno: int, file: Any = None, line: str | None = None) -> None:
        nonlocal saw_uncompiled
        if _FLEX_ATTN_UNCOMPILED_NEEDLE in (str(message) if message else "").lower():
            saw_uncompiled = True
            return
        return orig_showwarning(message, category, filename, lineno, file=file, line=line)

    with warnings.catch_warnings():
        warnings.showwarning = _showwarning
        try: out = fn()
        finally: warnings.showwarning = orig_showwarning
    return out, saw_uncompiled


def _flex_ckpt_always_clone_enabled(out: Any) -> bool:
    if not bool(is_checkpoint()): return False
    if env_bool("ENN_CKPT_DISABLE_CUDAGRAPHS", default=True): return False
    if bool(torch.is_grad_enabled()) and not env_bool("ENN_CKPT_CUDAGRAPH_CLONE_RECOMPUTE", default=False): return False
    if _flex_attention_compile_mode() not in {"max-autotune", "reduce-overhead"}: return False
    
    match out:
        case torch.Tensor(): t0 = out
        case list() | tuple() if out:
            t0 = next((v for v in out if torch.is_tensor(v)), None)
        case _: t0 = None
        
    return bool(getattr(t0.device, "type", None) == "cuda") if torch.is_tensor(t0) else False


@torch_compiler_disable(recursive=False, reason="FlexAttention: always clone outputs under reentrant checkpoint to avoid CUDAGraph overwrite")
def _flex_ckpt_always_clone_out(out: Any) -> Any:
    if not _flex_ckpt_always_clone_enabled(out): return out
    match out:
        case torch.Tensor(): return out.clone()
        case tuple(): return tuple(v.clone() if torch.is_tensor(v) else v for v in out)
        case list(): return [v.clone() if torch.is_tensor(v) else v for v in out]
        case _: return out


def _call_torch_flex_attention_eager(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args: Any, flex_kwargs: dict[str, Any]) -> Any:
    if _torch_flex_attention is None: raise RuntimeError("Flex Attention is not available")
    _install_flex_uncompiled_warning_suppression()
    _force_enable_torch_flex_compile()
    
    with skip_non_infra_dispatch_mode(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_FLEX_UNCOMPILED_WARN_RE, category=UserWarning, module=_FLEX_UNCOMPILED_WARN_MODULE_RE)
        if bool(is_checkpoint()) and getattr(q.device, "type", None) == "cuda" and not env_bool("ENN_CKPT_DISABLE_CUDAGRAPHS", default=True) and _flex_attention_compile_mode() in {"max-autotune", "reduce-overhead"}:
            cudagraph_mark_step_begin()
        return _flex_ckpt_always_clone_out(_torch_flex_attention(q, k, v, **flex_kwargs))


def _get_compiled_flex_attention_for_kwargs(q: torch.Tensor, flex_kwargs: dict[str, Any]) -> tuple[Any, tuple[Any, ...]]:
    if not _HAS_TORCH_FLEX or _torch_flex_attention is None: raise RuntimeError("Flex Attention is not available")
    if is_dynamo_compiling() or is_tracing_or_exporting() or not torch_compiler_supported(): return _torch_flex_attention, ("flexattn", "raw")
    
    mode = _flex_attention_compile_mode()
    dynamic = _flex_attention_dynamic_flag(mode)
    key = _flex_attention_cache_key(mode=mode, dynamic=dynamic, device=q.device, dtype=q.dtype, flex_kwargs=flex_kwargs)
    
    if (cached := _FLEX_ATTN_SPECIALIZED.get(key)) is not None: return cached, key
    if _FLEX_ATTN_FAILED.get(key) is not None:
        if _flex_retry_failed_enabled() and _flex_env_overrides_present():
            with _FLEX_ATTN_SPECIALIZE_LOCK: _FLEX_ATTN_FAILED.pop(key, None)
        else:
            return _torch_flex_attention, key
            
    with _FLEX_ATTN_SPECIALIZE_LOCK:
        if (cached := _FLEX_ATTN_SPECIALIZED.get(key)) is not None: return cached, key
        if _FLEX_ATTN_FAILED.get(key) is not None:
            if _flex_retry_failed_enabled() and _flex_env_overrides_present(): _FLEX_ATTN_FAILED.pop(key, None)
            else: return _torch_flex_attention, key
        try:
            compiled = _compile_flex_attention_wrapper(mode=mode, dynamic=dynamic, flex_kwargs=flex_kwargs)
            _FLEX_ATTN_SPECIALIZED[key] = compiled
            return compiled, key
        except Exception as exc:
            _FLEX_ATTN_FAILED[key] = f"{type(exc).__name__}: {exc}"
            _warn_once(f"flexattn-compile-failed-{hash(key)}", f"FlexAttention: torch.compile() failed; falling back to eager. (mode={mode!r}, dynamic={dynamic!r})\n{type(exc).__name__}: {exc}")
            return _torch_flex_attention, key


def _exporting_boundary() -> bool:
    return bool(is_export_or_trace())


def _coerce_block_mask_to_dense(mask: Any, *args: Any, device: torch.device) -> Optional[torch.Tensor]:
    match mask:
        case None: return None
        case torch.Tensor(): return mask.to(device)
        case _ if hasattr(mask, "to_dense"):
            with contextlib.suppress(Exception):
                if torch.is_tensor(dense := mask.to_dense()): return dense.to(device)
            return None
        case _: return None


def _int_or_none(x: Any) -> Optional[int]:
    try: return int(x)
    except Exception: return None


def _env_int(name: str, default: int) -> int:
    try: return int(os.environ.get(name) or default)
    except Exception: return int(default)


def _flex_env_overrides_present() -> bool:
    return any(k in os.environ for k in ("ENN_FLEX_BLOCK_M", "ENN_FLEX_BLOCK_N", "ENN_FLEX_NUM_STAGES", "ENN_FLEX_NUM_WARPS", "ENN_FLEX_BWD_NUM_STAGES", "ENN_FLEX_BWD_NUM_WARPS", "ENN_FLEX_BWD_BLOCK_M1", "ENN_FLEX_BWD_BLOCK_N1", "ENN_FLEX_BWD_WRITE_DQ"))


def _looks_like_triton_resource_error(exc: BaseException) -> bool:
    msg = str(exc)
    return "No valid triton configs" in msg and any(n in msg for n in ("out of resource", "OutOfResources", "Hardware limit", "Reducing block sizes", "num_stages", "num_warps"))


def _warn_flex_debug_once(key: str, msg: str) -> None:
    if _flex_debug_enabled(): _warn_once(key, msg)


def _cuda_sm_for_flex_defaults(device: Optional[torch.device] = None) -> Optional[int]:
    if (device is not None and getattr(device, "type", "cpu") != "cuda") or not torch.cuda.is_available(): return None
    try:
        major, minor = torch.cuda.get_device_capability(getattr(device, "index", None) or torch.cuda.current_device())
        return int(major) * 10 + int(minor)
    except Exception: return None


def _pos_int(v: Any, default: int) -> int:
    try: return i if (i := int(v)) > 0 else int(default)
    except Exception: return int(default)


def _clamp_max(opts: dict[str, Any], key: str, cap: int) -> None:
    opts[key] = min(_pos_int(opts.get(key), cap_i := _pos_int(cap, 1)), cap_i)


def _resource_safe_kernel_options(existing: Any, *, device: Optional[torch.device] = None, sm: Optional[int] = None) -> dict[str, Any]:
    existing = existing if isinstance(existing, Mapping) else None
    sm = sm if sm is not None else _cuda_sm_for_flex_defaults(device=device)
    
    fw_b, fw_w = (32, 2) if sm is not None and sm <= 75 else (64, 4)
    bw_b, bw_w = (16, 2) if sm is not None and sm <= 75 else (32, 4)

    bm, bn = _env_int("ENN_FLEX_BLOCK_M", fw_b), _env_int("ENN_FLEX_BLOCK_N", fw_b)
    ns, nw = _env_int("ENN_FLEX_NUM_STAGES", 1), _env_int("ENN_FLEX_NUM_WARPS", fw_w)
    bns, bnw = _env_int("ENN_FLEX_BWD_NUM_STAGES", 1), _env_int("ENN_FLEX_BWD_NUM_WARPS", bw_w)
    bm1, bn1 = _env_int("ENN_FLEX_BWD_BLOCK_M1", bw_b), _env_int("ENN_FLEX_BWD_BLOCK_N1", bw_b)
    bwd_wdq = 1 if env_bool("ENN_FLEX_BWD_WRITE_DQ", False) else 0

    key = ("flex-kopts", int(id(existing)) if existing else 0, int(sm) if sm is not None else -1, int(bm), int(bn), int(ns), int(nw), int(bns), int(bnw), int(bm1), int(bn1), int(bwd_wdq))
    if (cached := _FLEX_ATTN_RESOURCE_KOPTS.get(key)) is not None: return cached
    
    base: dict[str, Any] = {str(k): v for k, v in existing.items()} if existing else {}
    for k, v in (("BLOCK_M", bm), ("BLOCK_N", bn), ("num_stages", ns), ("num_warps", nw), ("bwd_num_stages", bns), ("bwd_num_warps", bnw), ("bwd_BLOCK_M1", bm1), ("bwd_BLOCK_N1", bn1)):
        _clamp_max(base, k, int(v))
    if "WRITE_DQ" not in base and "bwd_WRITE_DQ" not in base: base["bwd_WRITE_DQ"] = bool(bwd_wdq)
    
    with _FLEX_ATTN_RESOURCE_KOPTS_LOCK:
        return _FLEX_ATTN_RESOURCE_KOPTS.setdefault(key, base)


def _python_token_mask_from_mask_mod(mask_mod: Any, *args: Any, B: int, H: int, Lq: int, Lk: int, device: torch.device, max_elems: int) -> Optional[torch.Tensor]:
    if (total := B * H * Lq * Lk) <= 0 or total > max_elems: return None
    try:
        out = torch.empty((B, H, Lq, Lk), dtype=torch.bool, device=device)
        for b in range(B):
            for h in range(H):
                for qi in range(Lq):
                    for ki in range(Lk):
                        out[b, h, qi, ki] = bool(mask_mod(b, h, qi, ki))
        return out
    except Exception: return None


def _blockmask_to_token_mask(mask: Any, *args: Any, B: int, H: int, Lq: int, Lk: int, device: torch.device) -> Optional[torch.Tensor]:
    if mask is None: return None
    if (mask_mod := getattr(mask, "mask_mod", None)) is not None:
        if _torch_create_mask is not None and all(isinstance(v, int) for v in (B, H, Lq, Lk)):
            with contextlib.suppress(Exception):
                if torch.is_tensor(tok := _torch_create_mask(mask_mod, B, H, Lq, Lk, device=str(device))): return tok.to(device)
        if _exporting_boundary():
            if all(v is not None for v in (b_i := _int_or_none(B), h_i := _int_or_none(H), lq_i := _int_or_none(Lq), lk_i := _int_or_none(Lk))):
                if py_mask := _python_token_mask_from_mask_mod(mask_mod, B=b_i, H=h_i, Lq=lq_i, Lk=lk_i, device=device, max_elems=_env_int("ENN_FLEX_PY_MASK_MAX_ELEMS", 2_000_000)): return py_mask
                
    if (dense := _coerce_block_mask_to_dense(mask, device=device)) is None or dense.dim() < 2 or (dense.shape[-2] == Lq and dense.shape[-1] == Lk): return dense
    
    q_block, k_block = None, None
    match getattr(mask, "BLOCK_SIZE", None):
        case (q, k) if isinstance(q, int) and isinstance(k, int): q_block, k_block = q, k
        case int(sz): q_block, k_block = sz, sz
        
    if q_block is None or k_block is None:
        q_block = max(1, int(math.ceil(Lq / dense.shape[-2]))) if dense.shape[-2] > 0 else None
        k_block = max(1, int(math.ceil(Lk / dense.shape[-1]))) if dense.shape[-1] > 0 else None
        
    if q_block is None or k_block is None: return dense
    return dense.repeat_interleave(q_block, dim=-2).repeat_interleave(k_block, dim=-1)[..., :Lq, :Lk]


def _apply_allowed_mask(scores: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    return scores.masked_fill((allowed != 0).logical_not() if allowed.dtype != torch.bool else allowed.logical_not(), torch.finfo(scores.dtype).min)


def _flatten_attn_mask(mask: torch.Tensor, *args: Any, device: torch.device, B: int, H: int, L: int, S: int) -> tuple[torch.Tensor, int, int, int]:
    del args
    mask = mask.to(device)
    trace_like = bool(is_symbolic())
    
    match mask.dim():
        case 0: return mask.view(1, 1, 1, 1).expand(1, 1, 1, S), 1, 1, 1
        case 1:
            if trace_like: assert_trace(mask.size(0) == S, "attn_mask S mismatch")
            elif mask.shape[0] != S: raise RuntimeError(f"attn_mask S mismatch: {mask.shape} != {S}")
            return mask.view(1, 1, 1, mask.shape[0]), 1, 1, 1
        case 2:
            a, b = mask.shape
            if trace_like:
                assert_trace(mask.size(1) == S, "attn_mask S mismatch")
                assert_trace(mask.size(0) == 1, "2D attn_mask under symbolic shapes must be (1,S). Use 4D attn_mask for batch/len-specific masks.")
                return mask.reshape(1, 1, 1, S), 1, 1, 1
            if b != S: raise RuntimeError(f"attn_mask S mismatch: {b} != {S}")
            if a == L: return mask.view(1, 1, L, S), 1, 1, L
            if a == 1: return mask.view(1, 1, 1, S), 1, 1, 1
            if a == B: return mask.view(B, 1, 1, S), B, 1, 1
            raise RuntimeError(f"Unsupported 2D mask {mask.shape} for B={B},L={L}")
        case 3:
            a, b, c = mask.shape
            if trace_like:
                assert_trace(mask.size(2) == S, "attn_mask S mismatch")
                assert_trace((mask.size(0) == 1) & (mask.size(1) == 1), "3D attn_mask under symbolic shapes must be (1,1,S). Use 4D attn_mask for batch/head/len-specific masks.")
                return mask.reshape(1, 1, 1, S), 1, 1, 1
            if c != S: raise RuntimeError(f"attn_mask S mismatch: {c} != {S}")
            if a == B and b == L: return mask.view(B, 1, L, S), B, 1, L
            if a == B and b == 1: return mask.view(B, 1, 1, S), B, 1, 1
            if a == H and b == L: return mask.view(1, H, L, S), 1, H, L
            if a == B and b == H: return mask.view(B, H, 1, S), B, H, 1
            raise RuntimeError(f"Unsupported 3D mask {mask.shape}")
        case 4:
            b0, h0, l0, s0 = mask.shape
            if trace_like: assert_trace(mask.size(3) == S, "attn_mask S mismatch")
            else:
                if s0 != S: raise RuntimeError(f"attn_mask S mismatch: {s0} != {S}")
                if b0 not in (1, B): raise RuntimeError(f"Batch mismatch {b0} != {B}")
                if h0 not in (1, H): raise RuntimeError(f"Head mismatch {h0} != {H}")
                if l0 not in (1, L): raise RuntimeError(f"Len mismatch {l0} != {L}")
            return mask, b0, h0, l0
        case _: raise RuntimeError(f"attn_mask rank {mask.dim()} not supported")


def _compute_flops_msr(batch: int, seq_len: int, *args: Any, num_heads: int, head_dim: int, use_gate: bool, **kwargs: Any) -> float:
    if any(x <= 0 for x in (batch, seq_len, num_heads, head_dim)): return 0.0
    attn = float(batch) * float(seq_len) * float(num_heads) * float(head_dim)
    return 4.0 * attn + (attn if use_gate else 0.0)


def _compute_flops_mha(query: torch.Tensor, key: torch.Tensor, num_heads: int, embed_dim: int, batch_first: bool, include_projections: bool = True, label: str = "MultiHeadAttention") -> None:
    try:
        if is_symbolic() or is_tracing_or_exporting() or is_meta_or_fake_tensor(query) or is_meta_or_fake_tensor(key) or FLOP_PROFILER is None or torch.jit.is_tracing() or torch.jit.is_scripting() or torch.compiler.is_compiling(): return
        if query.dim() < 3 or key.dim() < 3: return
        B, Lq, Sk = (query.size(0), query.size(1), key.size(1)) if batch_first else (query.size(1), query.size(0), key.size(0))
        H, E = int(num_heads), int(embed_dim)
        if any(x <= 0 for x in (B, Lq, Sk, H, E)) or E % H != 0: return
        
        Dh = E // H
        core = 2.0 * B * H * Lq * Dh * Sk + 2.0 * B * H * Lq * Sk * Dh
        proj = 2.0 * float(float(B) * torch.tensor([Lq, Sk, Sk, Lq], dtype=torch.float32).sum()) * E * E if include_projections else 0.0
        FLOP_PROFILER.add(label, float(core + proj))
    except Exception: pass


def _call_mha_compat(mha: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args: Any, attn_mask: Optional[torch.Tensor], is_causal: Optional[bool], kwargs: dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    call_kwargs: dict[str, Any] = dict(kwargs)
    if attn_mask is not None: call_kwargs["attn_mask"] = attn_mask
    kw_variants = (call_kwargs, {k: v for k, v in call_kwargs.items() if k != "average_attn_weights"}) if "average_attn_weights" in call_kwargs else (call_kwargs,)
    is_causal_arg = {"is_causal": is_causal} if is_causal is not None else {}
    
    for kw in kw_variants:
        with contextlib.suppress(TypeError): return mha(q, k, v, **is_causal_arg, **kw)
        with contextlib.suppress(TypeError): return mha(q, k, v, **kw)
    return mha(q, k, v, **call_kwargs)


def _mha_export_safe(mha: torch.nn.MultiheadAttention, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: Any, attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor], need_weights: bool, average_attn_weights: bool, is_causal: Optional[bool], batch_first: bool, training: bool) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    unbatched = query.dim() == 2
    if unbatched: query, key, value = query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0)
    elif not batch_first: query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
    
    B, Lq, Lk, E, H = query.shape[0], query.shape[1], key.shape[1], mha.embed_dim, mha.num_heads
    Dh = E // H
    w, b = mha.in_proj_weight, mha.in_proj_bias
    
    if query is key and key is value:
        q, k, v = torch.nn.functional.linear(query, w, b).split(E, dim=-1)
    else:
        wq, wk, wv = w.split(E, dim=0)
        bq, bk, bv = b.split(E, dim=0) if b is not None else (None, None, None)
        q, k, v = torch.nn.functional.linear(query, wq, bq), torch.nn.functional.linear(key, wk, bk), torch.nn.functional.linear(value, wv, bv)
        
    q, k, v = [x.reshape(B, -1, H, Dh).transpose(1, 2) for x in (q, k, v)]
    scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(float(Dh)))
    
    if is_causal:
        with contextlib.suppress(Exception):
            scores = scores.masked_fill(torch.ones((Lq, Lk), dtype=torch.bool, device=scores.device).tril().logical_not(), torch.finfo(scores.dtype).min)
            
    if key_padding_mask is not None:
        scores = scores.masked_fill(key_padding_mask.to(dtype=torch.bool, device=scores.device)[:, None, None, :], torch.finfo(scores.dtype).min)
        
    if attn_mask is not None:
        m = attn_mask.to(device=scores.device)
        if m.dtype == torch.bool:
            scores = scores.masked_fill(m[None, None, :, :] if m.dim() == 2 else (m[:, None, :, :] if m.dim() == 3 else m), torch.finfo(scores.dtype).min)
        else:
            scores = scores + (m[None, None, :, :] if m.dim() == 2 else (m[:, None, :, :] if m.dim() == 3 else m))
            
    attn = torch.nn.functional.dropout(torch.softmax(scores, dim=-1), p=float(mha.dropout), training=True) if training and float(mha.dropout) > 0.0 else torch.softmax(scores, dim=-1)
    out = mha.out_proj(torch.matmul(attn, v).transpose(1, 2).reshape(B, Lq, E))
    weights = (attn.mean(dim=1) if average_attn_weights else attn) if need_weights else None
    
    if unbatched:
        out = out.squeeze(0)
        if weights is not None: weights = weights.squeeze(0)
    elif not batch_first: out = out.transpose(0, 1)
    
    return out, weights


def _call_sdpa_fallback(fallback: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: Any, attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor], need_weights: bool, average_attn_weights: bool, is_causal: Optional[bool]) -> tuple[torch.Tensor, torch.Tensor | None]:
    call_kwargs = dict(attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=need_weights, average_attn_weights=average_attn_weights, is_causal=is_causal)
    if backends := get_dpa_backends():
        try:
            from torch.nn.attention import sdpa_kernel
            with sdpa_kernel(backends): return fallback(query, key, value, **call_kwargs)
        except Exception: pass
    return fallback(query, key, value, **call_kwargs)


def _attention_math_bshd(q_bshd: torch.Tensor, k_bshd: torch.Tensor, v_bshd: torch.Tensor, *args: Any, attn_mask: torch.Tensor | None, is_causal: bool, dropout_p: float, training: bool) -> torch.Tensor:
    _, _, q_len, head_dim = q_bshd.shape
    k_len = k_bshd.shape[2]
    scores = torch.matmul(q_bshd, k_bshd.transpose(-2, -1)) * (1.0 / math.sqrt(float(head_dim))) if isinstance(head_dim, int) and head_dim > 0 else torch.matmul(q_bshd, k_bshd.transpose(-2, -1))
    
    if is_causal:
        with contextlib.suppress(Exception):
            scores = scores.masked_fill(torch.ones((q_len, k_len), dtype=torch.bool, device=scores.device).tril().logical_not(), torch.finfo(scores.dtype).min)
            
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask.logical_not(), torch.finfo(scores.dtype).min) if attn_mask.dtype == torch.bool else scores + attn_mask
        
    probs = torch.nn.functional.dropout(torch.softmax(scores, dim=-1), p=float(dropout_p), training=True) if training and float(dropout_p) > 0.0 else torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v_bshd)


def _is_bshd_contiguous(tensor: torch.Tensor) -> bool:
    if tensor.dim() != 4: return False
    _, seq_len, num_heads, head_dim = tensor.shape
    stride = tensor.stride()
    return tensor.is_contiguous() and stride[-1] == 1 and stride[-2] == head_dim and stride[-3] == num_heads * head_dim and stride[-4] == seq_len * num_heads * head_dim


def _is_nvidia_te_supported() -> bool:
    if not torch.cuda.is_available() or getattr(get_device(), "type", "cpu") != "cuda": return False
    with contextlib.suppress(Exception):
        if torch._dynamo.is_compiling(): return False
    return True


def _is_nvidia_mha_preferred() -> bool:
    return bool(_HAS_TE and _is_nvidia_te_supported() and torch.cuda.is_available() and getattr(get_device(), "type", "cpu") == "cuda")


def _triton_retention(V: Any, LAMBDA: Any, OUT: Any, B: tl.constexpr, L: tl.constexpr, H: tl.constexpr, DH: tl.constexpr, SVB: Any, SVL: Any, SVH: Any, SVD: Any, SOB: Any, SOL: Any, SOH: Any, SOD: Any, BLOCK_DH: tl.constexpr) -> None:
    pid_bh, pid_d = tl.program_id(0), tl.program_id(1)
    b, h = pid_bh // H, pid_bh % H
    dh_off = pid_d * BLOCK_DH + tl.arange(0, BLOCK_DH)
    mask_d = dh_off < DH
    lam = tl.load(LAMBDA + h)
    state = tl.zeros([BLOCK_DH], dtype=tl.float32)
    for t in range(0, L):
        state = lam * state + tl.load(V + b * SVB + t * SVL + h * SVH + dh_off * SVD, mask=mask_d, other=0.0).to(tl.float32)
        tl.store(OUT + b * SOB + t * SOL + h * SOH + dh_off * SOD, state, mask=mask_d)


def reshape_for_mha(x: torch.Tensor, batch: int, heads: int, head_dim: int) -> torch.Tensor:
    if x.dim() != 3: raise ValueError(f"reshape_for_mha expects a 3D tensor (B,N,E), got shape={tuple(x.shape)}")
    return x.reshape(batch, -1, heads, head_dim).transpose(1, 2).contiguous()


# =============================================================================
# Submodules
# =============================================================================
class _MultiHeadAttentionNvidia(nn.Module):
    def __init__(self: Self, embed_dim: int, num_heads: int, *args: Any, bias: bool = True, dropout: float = 0.0, batch_first: bool = True, **kwargs: Any) -> None:
        super().__init__()
        self.batch_first = bool(batch_first)
        self.num_heads = int(num_heads)
        self._fallback = _MultiHeadAttentionCompat(embed_dim, num_heads, bias=bias, dropout=dropout, batch_first=batch_first, **kwargs)
        self._te_mha = self._nvidia_mha(embed_dim, num_heads, float(dropout), kwargs)
        self._force_pt: bool = self._te_mha is None
        self._te_forward_signature: inspect.Signature | None = None
        self._te_mask_param: str | None = None
        self._te_mask_type_param: str | None = None
        self._te_supports_is_causal: bool = False
        self._te_supports_training: bool = False
        self._te_supports_tuple_mask: bool = True
        
        if self._te_mha is not None:
            try: self._te_forward_signature = inspect.signature(getattr(self._te_mha, "forward", getattr(self._te_mha, "__call__", None)))
            except Exception: self._te_forward_signature = None
            params = self._te_forward_signature.parameters if self._te_forward_signature else {}
            
            self._te_mask_param = "attention_mask" if "attention_mask" in params else ("attn_mask" if "attn_mask" in params else None)
            self._te_mask_type_param = "attn_mask_type" if "attn_mask_type" in params else ("attention_mask_type" if "attention_mask_type" in params else None)
            self._te_supports_is_causal = "is_causal" in params
            self._te_supports_training = "training" in params

    @staticmethod
    def _nvidia_mha(embed_dim: int, num_heads: int, dropout: float, kwargs: dict[str, object]) -> nn.Module | None:
        if not (_HAS_TE and _is_nvidia_te_supported()): return None
        for cls in (getattr(te, n) for n in ("MultiHeadAttention", "MultiheadAttention") if hasattr(te, n)):
            for ckw in ({"hidden_size": embed_dim, "num_attention_heads": num_heads, "attention_dropout": dropout}, {"hidden_size": embed_dim, "num_heads": num_heads, "attention_dropout": dropout}, {"embed_dim": embed_dim, "num_heads": num_heads, "dropout": dropout}):
                with contextlib.suppress(Exception): return cls(**{**ckw, **kwargs})
        return None

    def forward(self: Self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None, need_weights: bool = False, is_causal: Optional[bool] = None, average_attn_weights: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        def _fallback_call() -> tuple[torch.Tensor, torch.Tensor | None]:
            return _call_sdpa_fallback(self._fallback, query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=need_weights, average_attn_weights=average_attn_weights, is_causal=is_causal)

        if _exporting_boundary() or self._force_pt or self._te_mha is None or need_weights or attn_mask is not None or not query.is_cuda or query.dtype not in (torch.float16, torch.bfloat16):
            return _fallback_call()
        with contextlib.suppress(Exception):
            if torch._dynamo.is_compiling(): return _fallback_call()
            
        bf = bool(self.batch_first)
        _q, _k, _v = (query, key, value) if bf else (query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1))
        te_kwargs, te_mask, mask_type = {}, None, None
        
        if key_padding_mask is not None:
            B0, Lq, Lk = int(_q.shape[0]), int(_q.shape[1]), int(_k.shape[1])
            if key_padding_mask.shape != (B0, Lk): return _fallback_call()
            kv_mask = (key_padding_mask if key_padding_mask.dtype is torch.bool else key_padding_mask.to(torch.bool)).to(device=_q.device, non_blocking=True).contiguous().view(B0, 1, 1, Lk)
            if Lq == Lk: te_mask = kv_mask
            else:
                if not self._te_supports_tuple_mask: return _fallback_call()
                te_mask = (torch.zeros((B0, 1, 1, Lq), device=_q.device, dtype=torch.bool), kv_mask)
            mask_type = "padding_causal" if bool(is_causal) else "padding"
        else:
            mask_type = "causal" if bool(is_causal) else "no_mask"
            
        if te_mask is not None and mask_type and mask_type.startswith("padding") and self._te_mask_type_param is None: return _fallback_call()
        if te_mask is not None:
            if self._te_mask_param is None: return _fallback_call()
            te_kwargs[self._te_mask_param] = te_mask
        if mask_type and self._te_mask_type_param: te_kwargs[self._te_mask_type_param] = mask_type
        elif self._te_supports_is_causal and is_causal is not None: te_kwargs["is_causal"] = bool(is_causal)
        if self._te_supports_training: te_kwargs["training"] = bool(self.training)
        
        try: out = self._te_mha(_q, _k, _v, **te_kwargs)
        except Exception as e:
            if isinstance(e, TypeError) and isinstance(te_mask, tuple): self._te_supports_tuple_mask = False
            self._force_pt = True
            return _fallback_call()
            
        y, w = (out[0] if isinstance(out, tuple) and len(out) >= 1 else out), None
        if not bf and isinstance(y, torch.Tensor) and y.dim() >= 2: y = y.transpose(0, 1)
        _compute_flops_mha(query, key, num_heads=self.num_heads, embed_dim=int(query.shape[-1]), batch_first=self.batch_first, include_projections=True)
        return y, w


class _MultiHeadAttentionCompat(nn.Module):
    def __init__(self: Self, embed_dim: int, num_heads: int, *args: Any, bias: bool = True, dropout: float = 0.0, batch_first: bool = True, **kwargs: Any) -> None:
        super().__init__()
        self.batch_first = bool(batch_first)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=float(dropout), bias=bool(bias), batch_first=self.batch_first)

    def forward(self: Self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None, need_weights: bool = False, is_causal: Optional[bool] = None, average_attn_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if _exporting_boundary():
            out, w = _mha_export_safe(self.mha, query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=need_weights, average_attn_weights=bool(average_attn_weights), is_causal=is_causal, batch_first=self.batch_first, training=bool(self.training))
        else:
            out, w = _call_mha_compat(self.mha, query, key, value, attn_mask=attn_mask, is_causal=is_causal, kwargs={"key_padding_mask": key_padding_mask, "need_weights": need_weights, "average_attn_weights": bool(average_attn_weights)} if need_weights else {"key_padding_mask": key_padding_mask, "need_weights": need_weights})
        _compute_flops_mha(query, key, num_heads=self.mha.num_heads, embed_dim=self.mha.embed_dim, batch_first=self.batch_first, include_projections=True)
        return out, w


class DotProductAttention(nn.Module):
    def __init__(self: Self, num_heads: Optional[int] = None, head_dim: Optional[int] = None, te_first: Optional[bool] = None) -> None:
        super().__init__()
        self.nh = int(num_heads) if num_heads is not None else None
        self.hd = int(head_dim) if head_dim is not None else None
        self.te_first = bool(get_runtime_config().te_first) if te_first is None else bool(te_first)
        self._te_ok = bool(_HAS_TE and torch.cuda.is_available() and _is_nvidia_te_supported() and self.nh and self.hd)
        self._force_pt: bool = False
        self._disable_te: bool = bool(env_bool("ENN_TE_DPA_DISABLE", False))
        self._auto_disable_te_on_zero: bool = bool(env_bool("ENN_TE_DPA_AUTO_DISABLE_ON_ZERO", True))
        self._auto_disable_te_check_every: int = max(1, _env_int("ENN_TE_DPA_AUTO_DISABLE_CHECK_EVERY", 128))
        self._auto_disable_te_check_count: int = 0
        self._auto_disable_te_logged: bool = False
        self._te_require_contig: bool = bool(env_bool("ENN_TE_DPA_REQUIRE_CONTIGUOUS", True))
        self._te_force_contig: bool = bool(env_bool("ENN_TE_DPA_FORCE_CONTIGUOUS", True))
        self._te_attn, self._te_mask_param, self._te_mask_type_param = None, None, None
        
        if self._te_ok:
            try: self._te_attn = te.DotProductAttention(num_attention_heads=self.nh, kv_channels=self.hd, qkv_format="bshd", attention_dropout=0.0)
            except Exception: self._te_attn, self._force_pt = None, True
            if self._te_attn is not None:
                try: self._te_forward_signature = inspect.signature(getattr(self._te_attn, "forward", getattr(self._te_attn, "__call__", None)))
                except Exception: self._te_forward_signature = None
                params = self._te_forward_signature.parameters if self._te_forward_signature else {}
                self._te_mask_param = "attention_mask" if "attention_mask" in params else ("attn_mask" if "attn_mask" in params else None)
                self._te_mask_type_param = "attn_mask_type" if "attn_mask_type" in params else ("attention_mask_type" if "attention_mask_type" in params else None)
                self._te_supports_mask = self._te_mask_param is not None
                self._te_supports_mask_type = self._te_mask_type_param is not None
                self._te_supports_attention_dropout = "attention_dropout" in params
                self._te_supports_is_causal = "is_causal" in params
                self._te_supports_training = "training" in params

    def forward(self: Self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args: Any, attn_mask: Optional[torch.Tensor] = None, dropout_p: float = 0.0, is_causal: bool = False, training: Optional[bool] = None, **kwargs: Any) -> torch.Tensor:
        del kwargs
        training = bool(training) if training is not None else self.training
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4: raise ValueError(f"DPA expects 4D inputs, got {q.shape}, {k.shape}, {v.shape}")
        
        tracing = bool(is_symbolic() or is_meta_or_fake_tensor(q) or is_meta_or_fake_tensor(k) or is_meta_or_fake_tensor(v))
        if tracing:
            assert_trace(q.size(0) == k.size(0), "Batch mismatch")
            assert_trace(q.size(0) == v.size(0), "Batch mismatch")
            assert_trace(q.size(1) == k.size(1), "Head mismatch")
            assert_trace(q.size(1) == v.size(1), "Head mismatch")
            assert_trace(k.size(2) == v.size(2), "K/V length mismatch")
            assert_trace(q.size(3) == k.size(3), "Embed dim mismatch")
            assert_trace(k.size(3) == v.size(3), "Embed dim mismatch")
        elif not (q.size(0) == k.size(0) == v.size(0) and q.size(1) == k.size(1) == v.size(1)) or k.size(2) != v.size(2) or q.size(3) != k.size(3) or k.size(3) != v.size(3):
            raise ValueError("Batch/Head/Len/Dim mismatch")
            
        q_bshd, k_bshd, v_bshd = [self._negotiate_dtype(t).contiguous() for t in (q, k, v)]
        if not tracing and _is_bshd_contiguous(q_bshd) and _is_bshd_contiguous(k_bshd):
            with contextlib.suppress(Exception): capture(q_bshd, bwd_factor=2.0 if training else 0.0, dropout_p=float(dropout_p), training=training)
            
        dropout_val = float(dropout_p) if training else 0.0
        B, H, L, D = q_bshd.shape
        S = k_bshd.shape[2]
        mask_bool, bias_float, mb, mh, mL = None, None, 0, 0, 0
        
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool: mask_bool = attn_mask
            else: bias_float = attn_mask
            
        if mask_bool is not None:
            mask_bool = mask_bool.to(device=q_bshd.device, dtype=torch.bool, non_blocking=True)
            if not _exporting_boundary():
                mask_bool, mb, mh, mL = _flatten_attn_mask(mask_bool, device=q_bshd.device, B=B, H=H, L=L, S=S)
                if not tracing and int(mh) == 1 and int(mL) == int(L):
                    with contextlib.suppress(Exception):
                        if int(mask_bool.stride(-2)) == 0: mask_bool, mL = mask_bool[..., :1, :], 1
                        
        if bias_float is not None:
            bias_float = bias_float.to(device=q_bshd.device, dtype=q_bshd.dtype, non_blocking=True)
            if not _exporting_boundary(): bias_float, _, _, _ = _flatten_attn_mask(bias_float, device=q_bshd.device, B=B, H=H, L=L, S=S)

        try: is_compiling = torch.compiler.is_compiling()
        except Exception: is_compiling = False
        
        exporting_boundary = _exporting_boundary()
        km, kkey_base, kkey_te = None, "", ""
        if not exporting_boundary and not is_compiling and not tracing:
            km = get_kernel_manager()
            site = getattr(self, "_enn_kernel_site", None) or f"{self.__class__.__name__}@{id(self):x}"
            setattr(self, "_enn_kernel_site", site)
            kkey_base = f"dpa:{site}@{q_bshd.device.type}:{int(q_bshd.device.index) if q_bshd.device.index is not None else 0}"
            kkey_te = f"{kkey_base}:te"

        def _is_finite_out(out: Any) -> bool:
            return True if not isinstance(out, torch.Tensor) or not out.is_floating_point() or out.numel() <= 0 else bool(torch.isfinite(out).all().item())

        use_te = bool(self._te_ok and self.te_first and not self._force_pt and not self._disable_te and not exporting_boundary and km is not None and not km.is_dead(kkey_te) and self._te_attn is not None and not is_compiling and not tracing and getattr(q_bshd.device, "type", "cpu") == "cuda" and q_bshd.dtype in (torch.float16, torch.bfloat16))
        te_mask, te_mask_type = None, None
        
        if use_te:
            if bias_float is not None: use_te = False
            elif mask_bool is None: te_mask_type = "causal" if bool(is_causal) else "no_mask"
            elif int(mh) == 1 and int(mL) == 1:
                te_mask = mask_bool.expand(int(B), 1, 1, int(S)).contiguous() if int(mb) != int(B) else mask_bool.contiguous()
                te_mask_type = "padding_causal" if bool(is_causal) else "padding"
            else: use_te = False
            
            if use_te and te_mask is not None and not self._te_supports_mask: use_te = False
            if use_te and te_mask is not None and te_mask_type and te_mask_type.startswith("padding") and not (self._te_supports_mask_type and self._te_mask_type_param): use_te = False
            if use_te and te_mask is None and bool(is_causal) and not (self._te_supports_mask_type or self._te_supports_is_causal): use_te = False

        if use_te:
            q_te, k_te, v_te = q_bshd.transpose(1, 2), k_bshd.transpose(1, 2), v_bshd.transpose(1, 2)
            if self._te_require_contig:
                ok = q_te.is_contiguous() and k_te.is_contiguous() and v_te.is_contiguous() and int(q_te.stride(-1)) == 1 and int(k_te.stride(-1)) == 1 and int(v_te.stride(-1)) == 1
                if not ok and self._te_force_contig:
                    q_te, k_te, v_te = q_te.contiguous(), k_te.contiguous(), v_te.contiguous()
                    ok = q_te.is_contiguous() and k_te.is_contiguous() and v_te.is_contiguous() and int(q_te.stride(-1)) == 1 and int(k_te.stride(-1)) == 1 and int(v_te.stride(-1)) == 1
                if not ok: use_te = False
                
            te_kwargs: dict[str, Any] = {}
            if self._te_supports_attention_dropout: te_kwargs["attention_dropout"] = dropout_val
            if self._te_supports_mask_type and self._te_mask_type_param is not None and te_mask_type is not None: te_kwargs[self._te_mask_type_param] = te_mask_type
            elif self._te_supports_is_causal: te_kwargs["is_causal"] = bool(is_causal)
            if self._te_supports_training: te_kwargs["training"] = training
            if te_mask is not None and self._te_supports_mask and self._te_mask_param: te_kwargs[self._te_mask_param] = te_mask
            
            try:
                def _call_te() -> torch.Tensor: return self._te_attn(q_te, k_te, v_te, **te_kwargs)
                out_te = km.run(kkey_te, _call_te, validate=_is_finite_out, sticky=True, safe_on_exception=False)
            except Exception:
                self._force_pt, use_te = True, False
            else:
                out_te = out_te.transpose(1, 2).contiguous()
                if self._auto_disable_te_on_zero and isinstance(out_te, torch.Tensor) and out_te.numel() > 0 and getattr(out_te.device, "type", None) == "cuda":
                    self._auto_disable_te_check_count += 1
                    if self._auto_disable_te_check_count == 1 or (self._auto_disable_te_check_count % int(self._auto_disable_te_check_every)) == 0:
                        with contextlib.suppress(Exception):
                            if float(out_te.detach().abs().max().item()) == 0.0 and max(float(q_bshd.detach().abs().max().item()), float(k_bshd.detach().abs().max().item()), float(v_bshd.detach().abs().max().item())) > 0.0:
                                if not self._auto_disable_te_logged:
                                    self._auto_disable_te_logged = True
                                    warnings.warn("[ENN] TE DPA produced all-zero output on CUDA despite non-zero inputs; disabling TE DPA for this process and falling back to PyTorch attention.", UserWarning, stacklevel=2)
                                with contextlib.suppress(Exception): km.mark_dead(kkey_te, reason="all-zero output")
                                self._disable_te, self._force_pt, use_te = True, True, False
                                
                if use_te:
                    try:
                        if FLOP_PROFILER is not None and not tracing: FLOP_PROFILER.add("DotProductAttention", float(4.0 * q_bshd.shape[0] * q_bshd.shape[1] * q_bshd.shape[2] * k_bshd.shape[2] * q_bshd.shape[3]))
                    except Exception: pass
                    return out_te
                    
        final_mask: torch.Tensor | None = None
        sdpa_is_causal = bool(is_causal)
        if bias_float is None:
            if mask_bool is not None:
                final_mask, sdpa_is_causal = ~mask_bool, False
        else:
            final_mask = bias_float if mask_bool is None else (torch.where(mask_bool, torch.full((), torch.finfo(q_bshd.dtype).min, dtype=q_bshd.dtype, device=q_bshd.device), torch.zeros((), dtype=q_bshd.dtype, device=q_bshd.device)) + bias_float).contiguous()
            sdpa_is_causal = False
            
        if (exporting_boundary and not env_bool(("ENN_FORCE_SDPA",), default=False)) or env_bool(("ENN_DISABLE_SDPA",), default=False) or not q_bshd.is_cuda:
            if final_mask is not None:
                final_mask = final_mask.to(device=q_bshd.device, dtype=torch.bool if final_mask.dtype is torch.bool else q_bshd.dtype, non_blocking=True)
                if final_mask.dim() != 4: final_mask, _, _, _ = _flatten_attn_mask(final_mask, device=q_bshd.device, B=B, H=H, L=q_bshd.shape[2], S=k_bshd.shape[2])
            sdpa_out = _attention_math_bshd(q_bshd, k_bshd, v_bshd, attn_mask=final_mask, is_causal=bool(sdpa_is_causal), dropout_p=float(dropout_val), training=bool(training))
        else:
            sdpa_kwargs = {"attn_mask": final_mask, "dropout_p": dropout_val, "is_causal": bool(sdpa_is_causal)}
            if final_mask is not None:
                final_mask = final_mask.to(device=q_bshd.device, dtype=q_bshd.dtype if not final_mask.dtype is torch.bool else torch.bool, non_blocking=True)
                sdpa_kwargs["attn_mask"], bd, hd, _ = _flatten_attn_mask(final_mask, device=q_bshd.device, B=B, H=H, L=q_bshd.shape[2], S=k_bshd.shape[2])
                if not tracing and (bd not in (1, B) or hd not in (1, H)): raise RuntimeError("Attn mask mismatch")
            sdpa_out: torch.Tensor | None = None
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel
                def _be_name(be: Any) -> str:
                    try: return str(getattr(be, "name", None)).lower() if isinstance(getattr(be, "name", None), str) and getattr(be, "name", None) else str(be).replace("SDPBackend.", "").lower()
                    except Exception: return str(be).replace("SDPBackend.", "").lower()

                def _sdpa_call(backends: list[Any] | None) -> torch.Tensor:
                    with sdpa_kernel(backends) if backends else contextlib.nullcontext(), warnings.catch_warnings():
                        if env_bool("ENN_SDPA_FILTER_WARNINGS", default=True):
                            for msg in (".*Memory efficient kernel not used because:.*", ".*Memory Efficient attention has been runtime disabled.*", ".*Flash attention kernel not used because:.*", ".*Flash Attention does not support non-null attn_mask.*", ".*cuDNN attention kernel not used because:.*", ".*cuDNN attention has been runtime disabled.*"):
                                warnings.filterwarnings("ignore", message=msg, category=UserWarning)
                        return torch.nn.functional.scaled_dot_product_attention(q_bshd, k_bshd, v_bshd, **sdpa_kwargs)

                if env_bool("ENN_DPA_WARN_ALL_FALSE_MASK", default=False) and isinstance(am := sdpa_kwargs.get("attn_mask", None), torch.Tensor) and am.dtype == torch.bool and am.numel() > 0:
                    with contextlib.suppress(Exception):
                        if not bool(am.any().item()): _warn_once("dpa-all-false-mask", "[ENN] DotProductAttention: attn_mask is all-False (no allowed positions). Preserving mask so SDPA keeps fully masked queries zeroed.")

                cfg_backends = [be for be in list(get_dpa_backends()) if not (be == SDPBackend.FLASH_ATTENTION and (sdpa_kwargs.get("attn_mask") is not None or (isinstance(sdpa_kwargs.get("attn_mask"), torch.Tensor) and sdpa_kwargs.get("attn_mask").dtype != torch.bool))) and not (km is not None and kkey_base and km.is_dead(f"{kkey_base}:sdpa:{_be_name(be)}"))]
                try:
                    if _is_finite_out(out0 := _sdpa_call(cfg_backends if cfg_backends else None)): sdpa_out = out0
                except Exception: sdpa_out = None

                if sdpa_out is None:
                    candidates = cfg_backends or [getattr(SDPBackend, nm) for nm in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH") if hasattr(SDPBackend, nm)]
                    candidates = [be for be in candidates if not (be == SDPBackend.FLASH_ATTENTION and (sdpa_kwargs.get("attn_mask") is not None or (isinstance(sdpa_kwargs.get("attn_mask"), torch.Tensor) and sdpa_kwargs.get("attn_mask").dtype != torch.bool))) and not (km is not None and kkey_base and km.is_dead(f"{kkey_base}:sdpa:{_be_name(be)}"))]
                    if km is not None and kkey_base:
                        for be in candidates:
                            try:
                                sdpa_out = km.run(f"{kkey_base}:sdpa:{_be_name(be)}", lambda be_=be: _sdpa_call([be_]), validate=_is_finite_out, sticky=True, safe_on_exception=False)
                                break
                            except Exception: sdpa_out = None
                if sdpa_out is None: sdpa_out = _attention_math_bshd(q_bshd, k_bshd, v_bshd, attn_mask=sdpa_kwargs.get("attn_mask", None), is_causal=bool(sdpa_kwargs.get("is_causal", False)), dropout_p=float(dropout_val), training=bool(training))
            except Exception: sdpa_out = _attention_math_bshd(q_bshd, k_bshd, v_bshd, attn_mask=sdpa_kwargs.get("attn_mask", None), is_causal=bool(sdpa_kwargs.get("is_causal", False)), dropout_p=float(dropout_val), training=bool(training))
            
        with contextlib.suppress(Exception):
            if FLOP_PROFILER is not None and not tracing: FLOP_PROFILER.add("DotProductAttention", float(4.0 * B * H * q_bshd.shape[2] * k_bshd.shape[2] * D))
        return sdpa_out

    @staticmethod
    def _negotiate_dtype(tensor: torch.Tensor) -> torch.Tensor:
        if getattr(tensor.device, "type", "cpu") == "cpu" and tensor.dtype in (torch.float16, torch.bfloat16): return tensor.float()
        if getattr(tensor.device, "type", "cpu") == "mps" and tensor.dtype == torch.bfloat16: return tensor.to(torch.float16)
        return tensor


class MultiScaleRetention(nn.Module):
    def __init__(self: Self, d_model: int, nhead: int, use_gate: bool = True) -> None:
        super().__init__()
        self.d_model, self.nhead = int(d_model), int(nhead)
        if self.d_model % max(1, self.nhead) != 0: raise ValueError(f"MultiScaleRetention: d_model={self.d_model} must be divisible by nhead={self.nhead}")
        self.head_dim = int(self.d_model // max(1, self.nhead))
        self.use_gate = bool(use_gate)
        self.q_proj, self.v_proj, self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False), nn.Linear(self.d_model, self.d_model, bias=False), nn.Linear(self.d_model, self.d_model, bias=False)
        self.g_proj = nn.Linear(self.d_model, self.d_model, bias=False) if self.use_gate else None
        self.norm = nn.LayerNorm(self.d_model)
        self._triton_ok = bool(_HAS_TRITON_MSR and torch.cuda.is_available())
        self._decay_init, self._decay_range = 5.0, 1.0
        self._beta = nn.Parameter(float(self._decay_init) + float(self._decay_range) * (torch.arange(self.nhead, dtype=torch.float32) / float(max(self.nhead, 1))))

    def _decay_lambda(self: Self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        H, calc_dtype = int(self.nhead), dtype if dtype in (torch.float32, torch.float64) else torch.float32
        beta = getattr(self, "_beta", None)
        if not (isinstance(beta, torch.Tensor) and (is_tracing_or_exporting() or beta.numel() == H)):
            beta = float(self._decay_init) + float(self._decay_range) * (torch.arange(H, device=device, dtype=calc_dtype) / float(max(H, 1)))
        gammas = (1.0 - torch.pow(2.0, -beta.to(device=device, dtype=calc_dtype))).clamp(min=torch.finfo(calc_dtype).tiny, max=1.0 - 1e-9)
        return gammas.to(dtype=dtype) if calc_dtype != dtype else gammas

    @staticmethod
    def _apply_kpm_to_v(v: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        if not isinstance(attn_mask, torch.Tensor) or attn_mask.dim() != 2 or attn_mask.dtype is not torch.bool: return v
        if is_symbolic():
            assert_trace(attn_mask.shape[0] == v.shape[0], "attn_mask batch mismatch")
            assert_trace(attn_mask.shape[1] == v.shape[1], "attn_mask length mismatch")
        elif tuple(attn_mask.shape) != (int(v.shape[0]), int(v.shape[1])): return v
        return torch.where(attn_mask.to(device=v.device, non_blocking=True).unsqueeze(-1).unsqueeze(-1), torch.zeros_like(v), v)

    @staticmethod
    def _extract_state_tensor(state: Any, *args: Any, B: int, H: int) -> Optional[torch.Tensor]:
        match state:
            case torch.Tensor(): st = state
            case Mapping(): st = next((state.get(k) for k in ("state", "msr_state", "retention_state") if isinstance(state.get(k), torch.Tensor)), None)
            case _: return None
        if st is None: return None
        if st.dim() == 4: st = st.squeeze(2)
        if st.dim() != 3: return None
        if is_symbolic():
            assert_trace(st.shape[0] == B, "state batch mismatch")
            assert_trace(st.shape[1] == H, "state head mismatch")
        elif tuple(st.shape[:2]) != (int(B), int(H)): return None
        return st

    @staticmethod
    def _select_last_state(state_tensor: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        if state_tensor.dim() != 4: raise ValueError(f"_select_last_state expects (B,L,H,Dh), got {tuple(state_tensor.shape)}")
        B, L, H, Dh = state_tensor.shape
        if is_symbolic(): assert_trace(L > 0, "empty sequence")
        elif L <= 0: return state_tensor.new_zeros((B, H, Dh))
        if isinstance(attn_mask, torch.Tensor) and attn_mask.dim() == 2 and attn_mask.dtype is torch.bool:
            if is_symbolic():
                assert_trace(attn_mask.shape[0] == B, "attn_mask batch mismatch")
                assert_trace(attn_mask.shape[1] == L, "attn_mask length mismatch")
            elif tuple(attn_mask.shape) != (B, L): return state_tensor[:, -1]
            return torch.gather(state_tensor, dim=1, index=((~attn_mask).to(dtype=torch.int64).sum(dim=1).clamp(min=1) - 1).clamp(min=0, max=L - 1).view(B, 1, 1, 1).expand(-1, -1, H, Dh)).squeeze(1)
        return state_tensor[:, -1]

    @staticmethod
    def _scan_causal_torch(v: torch.Tensor, lam_h: torch.Tensor) -> torch.Tensor:
        B, L, H, Dh = v.shape
        if is_symbolic(): assert_trace(L > 0, "empty sequence")
        elif L <= 0: return v.new_zeros(v.shape)
        calc_dtype = torch.float32 if v.dtype in (torch.float16, torch.bfloat16) else v.dtype
        p = torch.pow(lam_h.to(dtype=calc_dtype, device=v.device).view(1, 1, H, 1), torch.arange(L, device=v.device, dtype=calc_dtype).view(1, L, 1, 1)).clamp_min(torch.finfo(calc_dtype).tiny)
        cumsum_scaled = torch.cumsum(v.to(dtype=calc_dtype) * torch.reciprocal(p), dim=1)
        return (p * (v[:, 0].to(dtype=calc_dtype).unsqueeze(1) + (cumsum_scaled - cumsum_scaled[:, :1]))).to(dtype=v.dtype).contiguous()

    @torch_compiler_disable(recursive=False, reason="Triton retention scan")
    def _scan_causal_triton(self: Self, v: torch.Tensor, lam_h: torch.Tensor) -> torch.Tensor:
        if v.dim() != 4: raise ValueError(f"_scan_causal_triton expects (B,L,H,Dh), got {tuple(v.shape)}")
        if getattr(v.device, "type", "cpu") != "cuda": raise RuntimeError("_scan_causal_triton requires CUDA tensor")
        B, L, H, Dh = v.shape
        out = torch.empty_like(v, dtype=v.dtype)
        SVB, SVL, SVH, SVD = v.stride()
        SOB, SOL, SOH, SOD = out.stride()
        try: BLOCK_DH = int(env_str("ENN_MSR_TRITON_BLOCK_DH") or (64 if Dh >= 64 else 32))
        except Exception: BLOCK_DH = 64 if Dh >= 64 else 32
        try: num_warps = int(env_str("ENN_MSR_TRITON_NUM_WARPS") or (8 if BLOCK_DH >= 64 else 4))
        except Exception: num_warps = 8 if BLOCK_DH >= 64 else 4
        
        _triton_retention[(B * H, (Dh + BLOCK_DH - 1) // BLOCK_DH)](v, lam_h, out, B, L, H, Dh, SVB, SVL, SVH, SVD, SOB, SOL, SOH, SOD, BLOCK_DH=BLOCK_DH, num_warps=num_warps)
        return out

    def _scan_causal(self: Self, v: torch.Tensor, lam_h: torch.Tensor) -> torch.Tensor:
        if not _HAS_TRITON_MSR or _triton_retention is None or env_bool("ENN_MSR_FORCE_TORCH", default=False) or _exporting_boundary() or not self._triton_ok or not v.is_cuda or not torch.cuda.is_available() or not env_bool("ENN_ENABLE_MSR_TRITON", default=False): return self._scan_causal_torch(v, lam_h)
        km = get_kernel_manager()
        site = getattr(self, "_enn_kernel_site", None) or f"{self.__class__.__name__}@{id(self):x}"
        setattr(self, "_enn_kernel_site", site)
        k_triton = f"msr:{site}@{v.device.type}:{int(v.device.index) if v.device.index is not None else 0}:scan_triton"
        if km.is_dead(k_triton): return self._scan_causal_torch(v, lam_h)

        def _is_finite_out(out: Any) -> bool:
            return True if not isinstance(out, torch.Tensor) or not out.is_floating_point() or out.numel() <= 0 else bool(torch.isfinite(out).all().item())

        try:
            out = km.run(k_triton, lambda: self._scan_causal_triton(v.contiguous(), lam_h), validate=_is_finite_out, sticky=True, safe_on_exception=False)
            if env_bool("ENN_MSR_DEBUG_ZERO_OUTPUT", default=False):
                with contextlib.suppress(Exception):
                    if float(out.detach().abs().sum().item()) == 0.0 and float(v.detach().abs().sum().item()) > 0.0:
                        _warn_once("msr-triton-zero-output", "[ENN] MultiScaleRetention(triton): got all-zeros output with non-zero inputs; disabling triton scan for this process.")
                        with contextlib.suppress(Exception): km.mark_dead(k_triton, reason="zero output")
                        self._triton_ok = False
                        return self._scan_causal_torch(v, lam_h)
            return out
        except Exception: return self._scan_causal_torch(v, lam_h)

    def forward(self: Self, x: torch.Tensor, *args: Any, decay: Any = None, attn_mask: torch.Tensor | None = None, state: Any = None, return_state: bool = False, **kwargs: Any) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        restore_dtype = x.dtype if getattr(x.device, "type", "cpu") == "mps" and x.dtype == torch.bfloat16 else None
        x_in = x.to(torch.float16) if restore_dtype else x
        if x_in.dim() != 3: raise ValueError(f"MultiScaleRetention expects (B,L,D), got {tuple(x_in.shape)}")
        
        B, L, D = x_in.shape
        trace_like = bool(is_symbolic())
        if not trace_like and L <= 0:
            out0 = x_in.new_zeros(x_in.shape)
            return (out0.to(restore_dtype) if restore_dtype else out0, x_in.new_zeros((B, self.nhead, int(self.head_dim))).to(restore_dtype) if restore_dtype else x_in.new_zeros((B, self.nhead, int(self.head_dim)))) if return_state else (out0.to(restore_dtype) if restore_dtype else out0)
        if not trace_like and D != int(self.d_model): raise ValueError(f"Last dimension {D} must equal d_model={int(self.d_model)}")
        
        decay_arg = args[0] if args else decay
        v = self._apply_kpm_to_v(self.v_proj(x_in).view(B, L, self.nhead, int(self.head_dim)), attn_mask)
        lam_h = self._decay_lambda(v.device, v.dtype).to(dtype=v.dtype, device=v.device)
        
        if isinstance(decay_arg, torch.Tensor):
            if decay_arg.dim() == 1:
                if trace_like: assert_trace(decay_arg.shape[0] == self.nhead, "decay[H] shape mismatch")
                elif int(decay_arg.shape[0]) != int(self.nhead): decay_arg = None
                if decay_arg is not None: lam_h = decay_arg.to(dtype=v.dtype, device=v.device)
            elif decay_arg.dim() == 3:
                if trace_like: assert_trace(decay_arg.shape[0] == self.nhead, "decay[H,*,*] shape mismatch")
                elif int(decay_arg.shape[0]) != int(self.nhead): decay_arg = None
                if decay_arg is not None: lam_h = decay_arg[:, 1, 0].to(dtype=v.dtype, device=v.device)
                
        if (st_bhd := self._extract_state_tensor(state, B=B, H=int(self.nhead))) is not None:
            v = torch.cat([v[:, :1] + lam_h.view(1, 1, self.nhead, 1) * st_bhd.to(dtype=v.dtype, device=v.device).unsqueeze(1), v[:, 1:]], dim=1)
            
        state_tensor = self._scan_causal(v, lam_h)
        y = self.norm((self.q_proj(x_in).view(B, L, self.nhead, int(self.head_dim)) * state_tensor).contiguous().view(B, L, self.d_model))
        if self.use_gate and self.g_proj is not None: y = y * torch.nn.functional.silu(self.g_proj(x_in))
        
        out = self.o_proj(y)
        last_state = (self._select_last_state(state_tensor, attn_mask).contiguous().detach() if not torch.is_grad_enabled() else self._select_last_state(state_tensor, attn_mask).contiguous()) if return_state else None
        
        with contextlib.suppress(Exception):
            if FLOP_PROFILER is not None and not trace_like and (fl := _compute_flops_msr(B, L, num_heads=int(self.nhead), head_dim=int(self.head_dim), use_gate=bool(self.use_gate and self.g_proj is not None))) > 0.0:
                FLOP_PROFILER.add("MultiScaleRetention", float(fl))
                
        if restore_dtype is not None:
            out = out.to(restore_dtype)
            if last_state is not None: last_state = last_state.to(restore_dtype)
        return (out, last_state if last_state is not None else state_tensor.new_zeros((B, self.nhead, int(self.head_dim)))) if return_state else out


class MultiHeadAttention(nn.Module):
    def __init__(self: Self, embed_dim: int, num_heads: int, *args: Any, bias: bool = True, dropout: float = 0.0, batch_first: bool = True, **kwargs: Any) -> None:
        super().__init__()
        self._backend = "torch"
        self.impl = _MultiHeadAttentionCompat(embed_dim, num_heads, bias=bias, dropout=dropout, batch_first=batch_first, **kwargs)
        if _is_nvidia_mha_preferred():
            with contextlib.suppress(Exception):
                if (impl := _MultiHeadAttentionNvidia(embed_dim, num_heads, bias=bias, dropout=dropout, batch_first=batch_first, **kwargs)) and impl._te_mha is not None:
                    self.impl, self._backend = impl, "te"
        if isinstance(self.impl, _MultiHeadAttentionNvidia):
            self.impl._fallback = _MultiHeadAttentionCompat(embed_dim, num_heads, bias=bias, dropout=dropout, batch_first=batch_first, **kwargs)

    @property
    def backend(self: Self) -> str:
        return self._backend

    def forward(self: Self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None, need_weights: bool = False, is_causal: Optional[bool] = None, average_attn_weights: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.impl(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=need_weights, average_attn_weights=average_attn_weights, is_causal=is_causal)


if _flex_attention_disabled():
    _HAS_TORCH_FLEX = False
