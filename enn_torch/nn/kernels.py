# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import inspect
import math
import os
import threading
import warnings
from typing import Any, Callable, Mapping, Optional, Self, Tuple

import torch
import torch._dynamo
from ..core.datatypes import env_bool, env_str
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
from torch import nn
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
            self: Self,
            fn: Callable[..., object] | None = None,
            **kwargs: object,
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
    from torch.nn.attention.flex_attention import (
        create_mask as _torch_create_mask,
    )
    from torch.nn.attention.flex_attention import (
        flex_attention as _torch_flex_attention,
    )

    _HAS_TORCH_FLEX = True
    with contextlib.suppress(Exception):
        _FLEX_KWARGS = set(
            inspect.signature(_torch_flex_attention).parameters.keys()
        )
except Exception:
    _torch_create_mask = None
    _torch_flex_attention = None
    _HAS_TORCH_FLEX = False
    pass

_HAS_TE = False
te = None

if (
    torch.cuda.is_available()
    and getattr(get_device(), "type", "cpu") == "cuda"
):
    try:
        import transformer_engine.pytorch as te

        _HAS_TE = True
    except Exception:
        _HAS_TE = False
        pass

_FLEX_ATTN_COMPILED: dict[str, Any] = {}
_FLEX_ATTN_COMPILE_LOCK = threading.Lock()
_FLEX_ATTN_BASE_COMPILED: dict[tuple[Any, ...], Any] = {}
_FLEX_ATTN_BASE_COMPILE_LOCK = threading.Lock()
_FLEX_ATTN_VERIFIED: set[tuple[Any, ...]] = set()
_FLEX_ATTN_VERIFIED_LOCK = threading.Lock()
_FLEX_ATTN_UNCOMPILED_NEEDLE = "flex_attention called without torch.compile"
_FLEX_KWARGS: set[str] = set()
_FLEX_ATTN_SPECIALIZED: dict[tuple[Any, ...], Any] = {}
_FLEX_ATTN_SPECIALIZE_LOCK = threading.Lock()
_FLEX_ATTN_FAILED: dict[tuple[Any, ...], str] = {}
_FLEX_ATTN_WARNED: set[str] = set()
_FLEX_ATTN_RESOURCE_KOPTS: dict[int, dict[str, Any]] = {}
_FLEX_ATTN_RESOURCE_KOPTS_LOCK = threading.Lock()


def _flex_attention_disabled() -> bool:
    return bool(env_bool("ENN_DISABLE_FLEX_ATTENTION", False))


def _flex_attention_compile_mode() -> str:
    cfg = get_runtime_cfg()
    global_mode = getattr(cfg, "compile_mode", "disabled")
    global_mode = canonicalize_compile_mode(global_mode)
    if global_mode in {"disabled", "aot-eager"}:
        return "aot-eager"
    if global_mode == "reduce-overhead":
        return "reduce-overhead"
    if global_mode in {"max-autotune", "max-autotune-no-cudagraphs"}:
        return global_mode
    return "reduce-overhead"


def _warn_once(key: str, message: str) -> None:
    if key in _FLEX_ATTN_WARNED:
        return
    _FLEX_ATTN_WARNED.add(key)
    with contextlib.suppress(Exception):
        warnings.warn(str(message), stacklevel=3)


def _env_bool_optional(name: str) -> Optional[bool]:
    if name not in os.environ:
        return None
    try:
        return bool(env_bool(name, False))
    except Exception:
        return None


def _flex_attention_dynamic_flag(mode: str) -> Optional[bool]:
    for key in (
        "ENN_FLEX_ATTENTION_DYNAMIC",
        "ENN_FLEX_COMPILE_DYNAMIC",
        "ENN_FLEXATTN_DYNAMIC",
    ):
        v = _env_bool_optional(key)
        if isinstance(v, bool):
            return v
    cfg = get_runtime_cfg()
    dyn = getattr(cfg, "compile_dynamic", None)
    if isinstance(dyn, bool):
        return dyn
    if str(mode) in {"max-autotune", "max-autotune-no-cudagraphs"}:
        return True
    return None


def _flex_attention_fallback_modes(mode: str) -> tuple[str, ...]:
    m = str(mode)
    if m == "max-autotune":
        return ("max-autotune-no-cudagraphs", "reduce-overhead")
    if m == "max-autotune-no-cudagraphs":
        return ("reduce-overhead",)
    return ()


def _flex_attention_cache_key(
    *args: Any,
    mode: str,
    dynamic: Optional[bool],
    device: torch.device,
    dtype: torch.dtype,
    flex_kwargs: Mapping[str, Any],
) -> tuple[Any, ...]:
    score_mod = flex_kwargs.get("score_mod", None)
    block_mask = flex_kwargs.get("block_mask", None)
    kernel_options = flex_kwargs.get("kernel_options", None)
    scale = flex_kwargs.get("scale", None)
    enable_gqa = flex_kwargs.get("enable_gqa", None)
    return_lse = flex_kwargs.get("return_lse", None)
    drop = flex_kwargs.get("dropout_p", None)
    if drop is None:
        drop = flex_kwargs.get("dropout", None)
    keys = tuple(sorted(str(k) for k in flex_kwargs.keys()))
    return (
        "flexattn",
        str(mode),
        (dynamic if dynamic is None else bool(dynamic)),
        str(device),
        str(dtype),
        keys,
        int(id(score_mod)) if score_mod is not None else 0,
        int(id(block_mask)) if block_mask is not None else 0,
        int(id(kernel_options)) if kernel_options is not None else 0,
        float(scale) if isinstance(scale, (int, float)) else None,
        float(drop) if isinstance(drop, (int, float)) else None,
        bool(enable_gqa) if enable_gqa is not None else None,
        bool(return_lse) if return_lse is not None else None,
    )


def _compile_flex_attention_wrapper(
    *args: Any,
    mode: str,
    dynamic: Optional[bool],
    flex_kwargs: dict[str, Any],
) -> Any:
    if _torch_flex_attention is None:
        raise RuntimeError("Flex Attention is not available")
    frozen = dict(flex_kwargs)
    dyn_key = dynamic if dynamic is None else bool(dynamic)
    base_key = ("flexattn-base", str(mode), dyn_key)
    base = _FLEX_ATTN_BASE_COMPILED.get(base_key)
    if base is None:
        with _FLEX_ATTN_BASE_COMPILE_LOCK:
            base = _FLEX_ATTN_BASE_COMPILED.get(base_key)
            if base is None:
                with skip_non_infra_dispatch_mode():
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=r"flex_attention called without torch\.compile",
                            category=UserWarning,
                        )
                        base = _model_compile(
                            _torch_flex_attention,
                            mode=mode,
                            dynamic=dynamic,
                            fullgraph=False,
                        )
                _FLEX_ATTN_BASE_COMPILED[base_key] = base
    if base is _torch_flex_attention:
        return _torch_flex_attention

    def _wrapped(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Any:
        return base(q, k, v, **frozen)

    return _wrapped


def _is_compile_failure(exc: BaseException) -> bool:
    t = type(exc)
    qual = f"{getattr(t, '__module__', '')}.{getattr(t, '__name__', '')}"
    msg = str(exc)
    needles = (
        "torch._dynamo",
        "torch._inductor",
        "BackendCompilerFailed",
        "LoweringException",
        "Unsupported",
        "CompileError",
    )
    if any(n in qual for n in needles):
        return True
    if any(n in msg for n in needles):
        return True
    return False


def _call_with_flex_warn_guard(fn: Callable[[], Any]) -> tuple[Any, bool]:
    saw_uncompiled = False
    orig_showwarning = warnings.showwarning

    def _showwarning(
        message: Any,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Any = None,
        line: str | None = None,
    ) -> None:
        nonlocal saw_uncompiled
        try:
            msg = str(message)
        except Exception:
            msg = ""
        if _FLEX_ATTN_UNCOMPILED_NEEDLE in msg:
            saw_uncompiled = True
            return
        return orig_showwarning(
            message, category, filename, lineno, file=file, line=line
        )

    with warnings.catch_warnings():
        warnings.showwarning = _showwarning
        try:
            out = fn()
        finally:
            warnings.showwarning = orig_showwarning
    return out, saw_uncompiled


def _call_torch_flex_attention_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *args: Any,
    flex_kwargs: dict[str, Any],
) -> Any:
    if _torch_flex_attention is None:
        raise RuntimeError("Flex Attention is not available")
    with skip_non_infra_dispatch_mode():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"flex_attention called without torch\.compile",
                category=UserWarning,
            )
            return _torch_flex_attention(q, k, v, **flex_kwargs)


def _get_compiled_flex_attention_for_kwargs(
    q: torch.Tensor, flex_kwargs: dict[str, Any]
) -> tuple[Any, tuple[Any, ...]]:
    if not _HAS_TORCH_FLEX or _torch_flex_attention is None:
        raise RuntimeError("Flex Attention is not available")
    if is_dynamo_compiling() or is_tracing_or_exporting():
        return _torch_flex_attention, ("flexattn", "raw")
    if not torch_compiler_supported():
        return _torch_flex_attention, ("flexattn", "raw")
    mode = _flex_attention_compile_mode()
    dynamic = _flex_attention_dynamic_flag(mode)
    key = _flex_attention_cache_key(
        mode=mode,
        dynamic=dynamic,
        device=q.device,
        dtype=q.dtype,
        flex_kwargs=flex_kwargs,
    )
    cached = _FLEX_ATTN_SPECIALIZED.get(key)
    if cached is not None:
        return cached, key
    failed = _FLEX_ATTN_FAILED.get(key)
    if failed is not None:
        return _torch_flex_attention, key
    with _FLEX_ATTN_SPECIALIZE_LOCK:
        cached = _FLEX_ATTN_SPECIALIZED.get(key)
        if cached is not None:
            return cached, key
        failed = _FLEX_ATTN_FAILED.get(key)
        if failed is not None:
            return _torch_flex_attention, key
        try:
            compiled = _compile_flex_attention_wrapper(
                mode=mode, dynamic=dynamic, flex_kwargs=flex_kwargs
            )
            _FLEX_ATTN_SPECIALIZED[key] = compiled
            return compiled, key
        except Exception as exc:
            _FLEX_ATTN_FAILED[key] = f"{type(exc).__name__}: {exc}"
            _warn_once(
                f"flexattn-compile-failed-{hash(key)}",
                "FlexAttention: torch.compile() failed; falling back to eager. "
                f"(mode={mode!r}, dynamic={dynamic!r})\n"
                f"{type(exc).__name__}: {exc}",
            )
            return _torch_flex_attention, key


def _exporting_boundary() -> bool:
    return bool(is_export_or_trace())


def _coerce_block_mask_to_dense(
    mask: Any, *args: Any, device: torch.device
) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if torch.is_tensor(mask):
        return mask.to(device)
    if hasattr(mask, "to_dense"):
        with contextlib.suppress(Exception):
            dense = mask.to_dense()
            if torch.is_tensor(dense):
                return dense.to(device)
    return None


def _int_or_none(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _looks_like_triton_resource_error(exc: BaseException) -> bool:
    msg = str(exc)
    if "No valid triton configs" not in msg:
        return False
    needles = (
        "out of resource",
        "OutOfResources",
        "Hardware limit",
        "Reducing block sizes",
        "num_stages",
        "num_warps",
    )
    return any(n in msg for n in needles)


def _cuda_sm_for_flex_defaults() -> Optional[int]:
    try:
        dev = get_device()
    except Exception:
        dev = None
    if (
        dev is None
        or getattr(dev, "type", None) != "cuda"
        or (not torch.cuda.is_available())
    ):
        return None
    try:
        idx = (
            dev.index
            if getattr(dev, "index", None) is not None
            else torch.cuda.current_device()
        )
        major, minor = torch.cuda.get_device_capability(idx)
        return int(major) * 10 + int(minor)
    except Exception:
        return None


def _pos_int(v: Any, default: int) -> int:
    try:
        i = int(v)
        return i if i > 0 else int(default)
    except Exception:
        return int(default)


def _clamp_max(opts: dict[str, Any], key: str, cap: int) -> None:
    cap_i = _pos_int(cap, 1)
    cur = opts.get(key, None)
    if cur is None:
        opts[key] = cap_i
        return
    opts[key] = min(_pos_int(cur, cap_i), cap_i)


def _resource_safe_kernel_options(existing: Any) -> dict[str, Any]:
    if (existing is not None) and (not isinstance(existing, Mapping)):
        existing = None
    key = int(id(existing)) if existing is not None else 0
    cached = _FLEX_ATTN_RESOURCE_KOPTS.get(key)
    if cached is not None:
        return cached
    base: dict[str, Any] = {}
    if isinstance(existing, Mapping):
        for k, v in existing.items():
            base[str(k)] = v
    sm = _cuda_sm_for_flex_defaults()
    fwd_block_def = 32 if (sm is not None and sm <= 75) else 64
    fwd_warps_def = 2 if (sm is not None and sm <= 75) else 4
    bwd_block_def = 16 if (sm is not None and sm <= 75) else 32
    bwd_warps_def = 2 if (sm is not None and sm <= 75) else 4

    _clamp_max(base, "BLOCK_M", _env_int("ENN_FLEX_BLOCK_M", fwd_block_def))
    _clamp_max(base, "BLOCK_N", _env_int("ENN_FLEX_BLOCK_N", fwd_block_def))
    _clamp_max(base, "num_stages", _env_int("ENN_FLEX_NUM_STAGES", 1))
    _clamp_max(base, "num_warps", _env_int("ENN_FLEX_NUM_WARPS", fwd_warps_def))

    _clamp_max(base, "bwd_num_stages", _env_int("ENN_FLEX_BWD_NUM_STAGES", 1))
    _clamp_max(base, "bwd_num_warps", _env_int("ENN_FLEX_BWD_NUM_WARPS", bwd_warps_def))
    _clamp_max(base, "bwd_BLOCK_M1", _env_int("ENN_FLEX_BWD_BLOCK_M1", bwd_block_def))
    _clamp_max(base, "bwd_BLOCK_N1", _env_int("ENN_FLEX_BWD_BLOCK_N1", bwd_block_def))
    if "WRITE_DQ" not in base and "bwd_WRITE_DQ" not in base:
        base["bwd_WRITE_DQ"] = bool(env_bool("ENN_FLEX_BWD_WRITE_DQ", False))
    with _FLEX_ATTN_RESOURCE_KOPTS_LOCK:
        cached = _FLEX_ATTN_RESOURCE_KOPTS.get(key)
        if cached is None:
            _FLEX_ATTN_RESOURCE_KOPTS[key] = base
            cached = base
    return cached


def _python_token_mask_from_mask_mod(
    mask_mod: Any,
    *args: Any,
    B: int,
    H: int,
    Lq: int,
    Lk: int,
    device: torch.device,
    max_elems: int,
) -> Optional[torch.Tensor]:
    total = B * H * Lq * Lk
    if total <= 0 or total > max_elems:
        return None
    try:
        out = torch.empty((B, H, Lq, Lk), dtype=torch.bool, device=device)
        for b in range(B):
            for h in range(H):
                for qi in range(Lq):
                    for ki in range(Lk):
                        out[b, h, qi, ki] = bool(mask_mod(b, h, qi, ki))
        return out
    except Exception:
        return None


def _blockmask_to_token_mask(
    mask: Any,
    *args: Any,
    B: int,
    H: int,
    Lq: int,
    Lk: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    mask_mod = getattr(mask, "mask_mod", None)
    if mask_mod is not None and _torch_create_mask is not None:
        if (
            isinstance(B, int)
            and isinstance(H, int)
            and isinstance(Lq, int)
            and isinstance(Lk, int)
        ):
            with contextlib.suppress(Exception):
                tok = _torch_create_mask(
                    mask_mod, B, H, Lq, Lk, device=str(device)
                )
                if torch.is_tensor(tok):
                    return tok.to(device)
    if _exporting_boundary() and (mask_mod is not None):
        b_i = _int_or_none(B)
        h_i = _int_or_none(H)
        lq_i = _int_or_none(Lq)
        lk_i = _int_or_none(Lk)
        if (
            (b_i is not None)
            and (h_i is not None)
            and (lq_i is not None)
            and (lk_i is not None)
        ):
            max_elems = _env_int("ENN_FLEX_PY_MASK_MAX_ELEMS", 2_000_000)
            py_mask = _python_token_mask_from_mask_mod(
                mask_mod,
                B=b_i,
                H=h_i,
                Lq=lq_i,
                Lk=lk_i,
                device=device,
                max_elems=max_elems,
            )
            if py_mask is not None:
                return py_mask
    dense = _coerce_block_mask_to_dense(mask, device=device)
    if dense is None:
        return None
    if dense.dim() < 2:
        return dense
    if dense.shape[-2] == Lq and dense.shape[-1] == Lk:
        return dense
    q_block: Optional[int] = None
    k_block: Optional[int] = None
    block_size = getattr(mask, "BLOCK_SIZE", None)
    if isinstance(block_size, tuple) and len(block_size) == 2:
        q_block, k_block = int(block_size[0]), int(block_size[1])
    elif isinstance(block_size, int):
        q_block = int(block_size)
        k_block = int(block_size)
    if q_block is None or k_block is None:
        if dense.shape[-2] > 0:
            q_block = max(1, int(math.ceil(Lq / dense.shape[-2])))
        if dense.shape[-1] > 0:
            k_block = max(1, int(math.ceil(Lk / dense.shape[-1])))
    if q_block is None or k_block is None:
        return dense
    expanded = dense.repeat_interleave(q_block, dim=-2).repeat_interleave(
        k_block, dim=-1
    )
    return expanded[..., :Lq, :Lk]


def _apply_allowed_mask(
    scores: torch.Tensor, allowed: torch.Tensor
) -> torch.Tensor:
    if allowed.dtype != torch.bool:
        allowed = allowed != 0
    return scores.masked_fill(
        allowed.logical_not(), torch.finfo(scores.dtype).min
    )


def _flatten_attn_mask(
    mask: torch.Tensor,
    *args: Any,
    device: torch.device,
    B: int,
    H: int,
    L: int,
    S: int,
) -> tuple[torch.Tensor, int, int, int]:
    del args
    trace_like = bool(is_symbolic())
    mask = mask.to(device)
    dim = mask.dim()
    if dim == 0:
        return mask.view(1, 1, 1, 1).expand(1, 1, 1, S), 1, 1, 1
    if dim == 1:
        if trace_like:
            assert_trace(
                mask.size(0) == S,
                "attn_mask S mismatch",
            )
        else:
            if mask.shape[0] != S:
                raise RuntimeError(
                    f"attn_mask S mismatch: {mask.shape} != {S}"
                )
        return mask.view(1, 1, 1, mask.shape[0]), 1, 1, 1
    if dim == 2:
        a, b = mask.shape
        if trace_like:
            assert_trace(
                mask.size(1) == S,
                "attn_mask S mismatch",
            )
            assert_trace(
                mask.size(0) == 1,
                "2D attn_mask under symbolic shapes must be (1,S). Use 4D attn_mask for batch/len-specific masks.",
            )
            out = mask.reshape(1, 1, 1, S)
            return out, 1, 1, 1
        if b != S:
            raise RuntimeError(f"attn_mask S mismatch: {b} != {S}")
        if a == L:
            return mask.view(1, 1, L, S), 1, 1, L
        if a == 1:
            return mask.view(1, 1, 1, S), 1, 1, 1
        if a == B:
            return mask.view(B, 1, 1, S), B, 1, 1
        raise RuntimeError(f"Unsupported 2D mask {mask.shape} for B={B},L={L}")
    if dim == 3:
        a, b, c = mask.shape
        if trace_like:
            assert_trace(
                mask.size(2) == S,
                "attn_mask S mismatch",
            )
            assert_trace(
                (mask.size(0) == 1) & (mask.size(1) == 1),
                "3D attn_mask under symbolic shapes must be (1,1,S). Use 4D attn_mask for batch/head/len-specific masks.",
            )
            out = mask.reshape(1, 1, 1, S)
            return out, 1, 1, 1
        if c != S:
            raise RuntimeError(f"attn_mask S mismatch: {c} != {S}")
        if a == B and b == L:
            return mask.view(B, 1, L, S), B, 1, L
        if a == B and b == 1:
            return mask.view(B, 1, 1, S), B, 1, 1
        if a == H and b == L:
            return mask.view(1, H, L, S), 1, H, L
        if a == B and b == H:
            return mask.view(B, H, 1, S), B, H, 1
        raise RuntimeError(f"Unsupported 3D mask {mask.shape}")
    if dim == 4:
        b0, h0, l0, s0 = mask.shape
        if trace_like:
            assert_trace(
                mask.size(3) == S,
                "attn_mask S mismatch",
            )
        else:
            if s0 != S:
                raise RuntimeError(f"attn_mask S mismatch: {s0} != {S}")
            if b0 not in (1, B):
                raise RuntimeError(f"Batch mismatch {b0} != {B}")
            if h0 not in (1, H):
                raise RuntimeError(f"Head mismatch {h0} != {H}")
            if l0 not in (1, L):
                raise RuntimeError(f"Len mismatch {l0} != {L}")
        return mask, b0, h0, l0
    raise RuntimeError(f"attn_mask rank {dim} not supported")


def _compute_flops_msr(
    batch: int,
    seq_len: int,
    *args: Any,
    num_heads: int,
    head_dim: int,
    use_gate: bool,
    **kwargs: Any,
) -> float:
    if batch <= 0 or seq_len <= 0 or num_heads <= 0 or head_dim <= 0:
        return 0.0
    attn = float(batch) * float(seq_len) * float(num_heads) * float(head_dim)
    gate_cost = attn if use_gate else 0.0
    return 4.0 * attn + gate_cost


def _compute_flops_mha(
    query: torch.Tensor,
    key: torch.Tensor,
    num_heads: int,
    embed_dim: int,
    batch_first: bool,
    include_projections: bool = True,
    label: str = "MultiHeadAttention",
) -> None:
    try:
        if (
            is_symbolic()
            or is_tracing_or_exporting()
            or is_meta_or_fake_tensor(query)
            or is_meta_or_fake_tensor(key)
        ):
            return
        if (
            FLOP_PROFILER is None
            or torch.jit.is_tracing()
            or torch.jit.is_scripting()
        ):
            return
        if torch.compiler.is_compiling():
            return
    except:
        return
    try:
        if query.dim() < 3 or key.dim() < 3:
            return
        B, Lq, Sk = (
            (query.size(0), query.size(1), key.size(1))
            if batch_first
            else (query.size(1), query.size(0), key.size(0))
        )
        H, E = int(num_heads), int(embed_dim)
        if B <= 0 or Lq <= 0 or Sk <= 0 or H <= 0 or E <= 0 or E % H != 0:
            return
        Dh = E // H
        core = 2.0 * B * H * Lq * Dh * Sk + 2.0 * B * H * Lq * Sk * Dh
        proj = 0.0
        if include_projections:
            tokens = (
                float(B)
                * torch.tensor([Lq, Sk, Sk, Lq], dtype=torch.float32).sum()
            )
            proj = 2.0 * float(tokens) * E * E
        FLOP_PROFILER.add(label, float(core + proj))
    except Exception:
        pass


def _call_mha_compat(
    mha: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *args: Any,
    attn_mask: Optional[torch.Tensor],
    is_causal: Optional[bool],
    kwargs: dict[str, Any],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    call_kwargs: dict[str, Any] = dict(kwargs)
    if attn_mask is not None:
        call_kwargs["attn_mask"] = attn_mask
    kw_variants: tuple[dict[str, Any], ...] = (call_kwargs,)
    if "average_attn_weights" in call_kwargs:
        no_avg = dict(call_kwargs)
        del no_avg["average_attn_weights"]
        kw_variants = (call_kwargs, no_avg)
    is_causal_arg = {"is_causal": is_causal} if is_causal is not None else {}
    for kw in kw_variants:
        with contextlib.suppress(TypeError):
            return mha(q, k, v, **is_causal_arg, **kw)
        with contextlib.suppress(TypeError):
            return mha(q, k, v, **kw)
    return mha(q, k, v, **call_kwargs)


def _mha_export_safe(
    mha: torch.nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *args: Any,
    attn_mask: Optional[torch.Tensor],
    key_padding_mask: Optional[torch.Tensor],
    need_weights: bool,
    average_attn_weights: bool,
    is_causal: Optional[bool],
    batch_first: bool,
    training: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    unbatched = False
    if query.dim() == 2:
        unbatched = True
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
    elif not batch_first:
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
    B = query.shape[0]
    Lq = query.shape[1]
    Lk = key.shape[1]
    E = mha.embed_dim
    H = mha.num_heads
    Dh = E // H
    w = mha.in_proj_weight
    b = mha.in_proj_bias
    if (query is key) and (key is value):
        qkv = torch.nn.functional.linear(query, w, b)
        q, k, v = qkv.split(E, dim=-1)
    else:
        wq, wk, wv = w.split(E, dim=0)
        if b is not None:
            bq, bk, bv = b.split(E, dim=0)
        else:
            bq = bk = bv = None
        q = torch.nn.functional.linear(query, wq, bq)
        k = torch.nn.functional.linear(key, wk, bk)
        v = torch.nn.functional.linear(value, wv, bv)
    q = q.reshape(B, Lq, H, Dh).transpose(1, 2)
    k = k.reshape(B, Lk, H, Dh).transpose(1, 2)
    v = v.reshape(B, Lk, H, Dh).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (
        1.0 / math.sqrt(float(Dh))
    )
    if is_causal:
        try:
            causal = torch.ones(
                (Lq, Lk), dtype=torch.bool, device=scores.device
            ).tril()
            scores = scores.masked_fill(
                causal.logical_not(), torch.finfo(scores.dtype).min
            )
        except Exception:
            pass
    if key_padding_mask is not None:
        kpm = key_padding_mask.to(dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(
            kpm[:, None, None, :], torch.finfo(scores.dtype).min
        )
    if attn_mask is not None:
        m = attn_mask.to(device=scores.device)
        if m.dtype == torch.bool:
            if m.dim() == 2:
                scores = scores.masked_fill(
                    m[None, None, :, :], torch.finfo(scores.dtype).min
                )
            elif m.dim() == 3:
                scores = scores.masked_fill(
                    m[:, None, :, :], torch.finfo(scores.dtype).min
                )
            elif m.dim() == 4:
                scores = scores.masked_fill(m, torch.finfo(scores.dtype).min)
        else:
            if m.dim() == 2:
                scores = scores + m[None, None, :, :]
            elif m.dim() == 3:
                scores = scores + m[:, None, :, :]
            else:
                scores = scores + m
    attn = torch.softmax(scores, dim=-1)
    if training and float(mha.dropout) > 0.0:
        attn = torch.nn.functional.dropout(
            attn, p=float(mha.dropout), training=True
        )
    out = torch.matmul(attn, v)
    out = out.transpose(1, 2).reshape(B, Lq, E)
    out = mha.out_proj(out)
    weights: Optional[torch.Tensor] = None
    if need_weights:
        if average_attn_weights:
            weights = attn.mean(dim=1)
        else:
            weights = attn
    if unbatched:
        out = out.squeeze(0)
        if weights is not None:
            weights = weights.squeeze(0)
    elif not batch_first:
        out = out.transpose(0, 1)
    return out, weights


def _call_sdpa_fallback(
    fallback: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *args: Any,
    attn_mask: Optional[torch.Tensor],
    key_padding_mask: Optional[torch.Tensor],
    need_weights: bool,
    average_attn_weights: bool,
    is_causal: Optional[bool],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    call_kwargs = dict(
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        need_weights=need_weights,
        average_attn_weights=average_attn_weights,
        is_causal=is_causal,
    )
    backends = get_dpa_backends()
    if backends:
        try:
            from torch.nn.attention import sdpa_kernel

            with sdpa_kernel(backends):
                return fallback(query, key, value, **call_kwargs)
        except Exception:
            pass
    return fallback(query, key, value, **call_kwargs)


def _attention_math_bshd(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    v_bshd: torch.Tensor,
    *args: Any,
    attn_mask: torch.Tensor | None,
    is_causal: bool,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    _, _, q_len, head_dim = q_bshd.shape
    k_len = k_bshd.shape[2]
    scores = torch.matmul(q_bshd, k_bshd.transpose(-2, -1))
    if isinstance(head_dim, int) and head_dim > 0:
        scores = scores * (1.0 / math.sqrt(float(head_dim)))
    if is_causal:
        try:
            causal = torch.ones(
                (q_len, k_len), dtype=torch.bool, device=scores.device
            ).tril()
            scores = scores.masked_fill(
                causal.logical_not(), torch.finfo(scores.dtype).min
            )
        except Exception:
            pass
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(
                attn_mask.logical_not(), torch.finfo(scores.dtype).min
            )
        else:
            scores = scores + attn_mask
    probs = torch.softmax(scores, dim=-1)
    if training and float(dropout_p) > 0.0:
        probs = torch.nn.functional.dropout(
            probs, p=float(dropout_p), training=True
        )
    return torch.matmul(probs, v_bshd)


def _is_bshd_contiguous(tensor: torch.Tensor) -> bool:
    if tensor.dim() != 4:
        return False
    _, seq_len, num_heads, head_dim = tensor.shape
    stride = tensor.stride()
    return (
        tensor.is_contiguous()
        and stride[-1] == 1
        and stride[-2] == head_dim
        and stride[-3] == num_heads * head_dim
        and stride[-4] == seq_len * num_heads * head_dim
    )


def _is_nvidia_te_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        device = get_device()
    except Exception:
        device = torch.device("cuda", 0)
    if device.type != "cuda":
        return False
    try:
        if torch._dynamo.is_compiling():
            return False
    except Exception:
        pass
    return True


def _is_nvidia_mha_preferred() -> bool:
    if not _HAS_TE:
        return False
    if not _is_nvidia_te_supported():
        return False
    try:
        device = get_device()
    except Exception:
        device = torch.device("cuda", 0)
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    return True


def _triton_retention(
    V: Any,
    LAMBDA: Any,
    OUT: Any,
    B: tl.constexpr,
    L: tl.constexpr,
    H: tl.constexpr,
    DH: tl.constexpr,
    SVB: Any,
    SVL: Any,
    SVH: Any,
    SVD: Any,
    SOB: Any,
    SOL: Any,
    SOH: Any,
    SOD: Any,
    BLOCK_DH: tl.constexpr,
) -> None:
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    dh_off = pid_d * BLOCK_DH + tl.arange(0, BLOCK_DH)
    mask_d = dh_off < DH
    lam = tl.load(LAMBDA + h)
    state = tl.zeros([BLOCK_DH], dtype=tl.float32)
    for t in range(0, L):
        v_ptr = V + b * SVB + t * SVL + h * SVH + dh_off * SVD
        v_t = tl.load(v_ptr, mask=mask_d, other=0.0).to(tl.float32)
        state = lam * state + v_t
        out_ptr = OUT + b * SOB + t * SOL + h * SOH + dh_off * SOD
        tl.store(out_ptr, state, mask=mask_d)


def reshape_for_mha(
    x: torch.Tensor,
    batch: int,
    heads: int,
    head_dim: int,
) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(
            f"reshape_for_mha expects a 3D tensor (B,N,E), got shape={tuple(x.shape)}"
        )
    return (
        x.reshape(int(batch), -1, int(heads), int(head_dim))
        .transpose(1, 2)
        .contiguous()
    )


class _MultiHeadAttentionNvidia(nn.Module):
    def __init__(
        self: Self,
        embed_dim: int,
        num_heads: int,
        *args: Any,
        bias: bool = True,
        dropout: float = 0.0,
        batch_first: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.batch_first = batch_first
        self.num_heads = int(num_heads)
        self._fallback = _MultiHeadAttentionCompat(
            embed_dim,
            num_heads,
            bias=bias,
            dropout=dropout,
            batch_first=batch_first,
            **kwargs,
        )
        self._te_mha = self._nvidia_mha(embed_dim, num_heads, dropout, kwargs)
        self._force_pt: bool = self._te_mha is None
        self._te_forward_signature: inspect.Signature | None = None
        self._te_mask_param: str | None = None
        self._te_mask_type_param: str | None = None
        self._te_supports_is_causal: bool = False
        self._te_supports_training: bool = False
        self._te_supports_tuple_mask: bool = True
        if self._te_mha is not None:
            _fwd = getattr(
                self._te_mha,
                "forward",
                getattr(self._te_mha, "__call__", None),
            )
            try:
                self._te_forward_signature = (
                    inspect.signature(_fwd) if _fwd else None
                )
            except:
                self._te_forward_signature = None
            params = (
                self._te_forward_signature.parameters
                if self._te_forward_signature
                else {}
            )
            if "attention_mask" in params:
                self._te_mask_param = "attention_mask"
            elif "attn_mask" in params:
                self._te_mask_param = "attn_mask"
            if "attn_mask_type" in params:
                self._te_mask_type_param = "attn_mask_type"
            elif "attention_mask_type" in params:
                self._te_mask_type_param = "attention_mask_type"
            self._te_supports_is_causal = "is_causal" in params
            self._te_supports_training = "training" in params

    @staticmethod
    def _nvidia_mha(
        embed_dim: int,
        num_heads: int,
        dropout: float,
        kwargs: dict[str, object],
    ) -> nn.Module | None:
        if not (_HAS_TE and _is_nvidia_te_supported()):
            return None
        candidates = [
            getattr(te, n)
            for n in ("MultiHeadAttention", "MultiheadAttention")
            if hasattr(te, n)
        ]
        for cls in candidates:
            ctor_variants = (
                dict(
                    hidden_size=embed_dim,
                    num_attention_heads=num_heads,
                    attention_dropout=dropout,
                ),
                dict(
                    hidden_size=embed_dim,
                    num_heads=num_heads,
                    attention_dropout=dropout,
                ),
                dict(
                    embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
                ),
            )
            for ckw in ctor_variants:
                with contextlib.suppress(Exception):
                    return cls(**{**ckw, **kwargs})
        return None

    def forward(
        self: Self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        is_causal: Optional[bool] = None,
        average_attn_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if _exporting_boundary():
            return _call_sdpa_fallback(
                self._fallback,
                query,
                key,
                value,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )

        def _fallback_call() -> tuple[torch.Tensor, torch.Tensor | None]:
            return _call_sdpa_fallback(
                self._fallback,
                query,
                key,
                value,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )

        if (
            self._force_pt
            or self._te_mha is None
            or need_weights
            or attn_mask is not None
            or not query.is_cuda
            or query.dtype not in (torch.float16, torch.bfloat16)
        ):
            return _fallback_call()
        with contextlib.suppress(Exception):
            if torch._dynamo.is_compiling():
                return _fallback_call()
        bf = bool(self.batch_first)
        _q, _k, _v = query, key, value
        if not bf:
            _q, _k, _v = (
                query.transpose(0, 1),
                key.transpose(0, 1),
                value.transpose(0, 1),
            )
        te_kwargs, te_mask, mask_type = {}, None, None
        if key_padding_mask is not None:
            B0, Lq, Lk = int(_q.shape[0]), int(_q.shape[1]), int(_k.shape[1])
            if key_padding_mask.shape != (B0, Lk):
                return _fallback_call()
            kpm = key_padding_mask
            if kpm.dtype is not torch.bool:
                kpm = kpm.to(torch.bool)
            kpm = kpm.to(device=_q.device, non_blocking=True).contiguous()
            kv_mask = kpm.view(B0, 1, 1, Lk)
            if Lq == Lk:
                te_mask = kv_mask
            else:
                if not self._te_supports_tuple_mask:
                    return _fallback_call()
                q_mask = torch.zeros(
                    (B0, 1, 1, Lq), device=_q.device, dtype=torch.bool
                )
                te_mask = (q_mask, kv_mask)
            mask_type = "padding_causal" if bool(is_causal) else "padding"
        else:
            mask_type = "causal" if bool(is_causal) else "no_mask"
        if (
            te_mask is not None
            and mask_type
            and mask_type.startswith("padding")
            and self._te_mask_type_param is None
        ):
            return _fallback_call()
        if te_mask is not None:
            if self._te_mask_param is None:
                return _fallback_call()
            te_kwargs[self._te_mask_param] = te_mask
        if mask_type and self._te_mask_type_param:
            te_kwargs[self._te_mask_type_param] = mask_type
        elif self._te_supports_is_causal and (is_causal is not None):
            te_kwargs["is_causal"] = bool(is_causal)
        if self._te_supports_training:
            te_kwargs["training"] = bool(self.training)
        try:
            out = self._te_mha(_q, _k, _v, **te_kwargs)
        except Exception as e:
            if isinstance(e, TypeError) and isinstance(te_mask, tuple):
                self._te_supports_tuple_mask = False
            self._force_pt = True
            return _fallback_call()
        y, w = (
            (out[0] if isinstance(out, tuple) and len(out) >= 1 else out),
            None,
        )
        if not bf and isinstance(y, torch.Tensor) and y.dim() >= 2:
            y = y.transpose(0, 1)
        _compute_flops_mha(
            query,
            key,
            num_heads=self.num_heads,
            embed_dim=embed_dim,
            batch_first=self.batch_first,
            include_projections=True,
        )
        return y, w


class _MultiHeadAttentionCompat(nn.Module):
    def __init__(
        self: Self,
        embed_dim: int,
        num_heads: int,
        *args: Any,
        bias: bool = True,
        dropout: float = 0.0,
        batch_first: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.batch_first = batch_first
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )

    def forward(
        self: Self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        is_causal: Optional[bool] = None,
        average_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        kwargs = dict(
            key_padding_mask=key_padding_mask, need_weights=need_weights
        )
        if need_weights:
            kwargs["average_attn_weights"] = bool(average_attn_weights)
        if _exporting_boundary():
            out, w = _mha_export_safe(
                self.mha,
                query,
                key,
                value,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                average_attn_weights=bool(average_attn_weights),
                is_causal=is_causal,
                batch_first=self.batch_first,
                training=bool(self.training),
            )
        else:
            out, w = _call_mha_compat(
                self.mha,
                query,
                key,
                value,
                attn_mask=attn_mask,
                is_causal=is_causal,
                kwargs=kwargs,
            )
        _compute_flops_mha(
            query,
            key,
            num_heads=self.mha.num_heads,
            embed_dim=self.mha.embed_dim,
            batch_first=self.batch_first,
            include_projections=True,
        )
        return out, w


class DotProductAttention(nn.Module):
    def __init__(
        self: Self,
        num_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        te_first: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.nh = int(num_heads) if num_heads is not None else None
        self.hd = int(head_dim) if head_dim is not None else None
        cfg = get_runtime_config()
        self.te_first = (
            bool(cfg.te_first) if te_first is None else bool(te_first)
        )
        self._te_ok = bool(
            _HAS_TE
            and torch.cuda.is_available()
            and _is_nvidia_te_supported()
            and self.nh
            and self.hd
        )
        self._force_pt: bool = False
        self._te_attn, self._te_mask_param, self._te_mask_type_param = (
            None,
            None,
            None,
        )
        if self._te_ok:
            self._te = te
            try:
                self._te_attn = te.DotProductAttention(
                    num_attention_heads=self.nh,
                    kv_channels=self.hd,
                    qkv_format="bshd",
                    attention_dropout=0.0,
                )
            except Exception:
                self._te_attn, self._force_pt = None, True
            if self._te_attn is not None:
                _forward = getattr(
                    self._te_attn,
                    "forward",
                    getattr(self._te_attn, "__call__", None),
                )
                try:
                    self._te_forward_signature = (
                        inspect.signature(_forward) if _forward else None
                    )
                except:
                    self._te_forward_signature = None
                params = (
                    self._te_forward_signature.parameters
                    if self._te_forward_signature
                    else {}
                )
                if "attention_mask" in params:
                    self._te_mask_param = "attention_mask"
                elif "attn_mask" in params:
                    self._te_mask_param = "attn_mask"
                if "attn_mask_type" in params:
                    self._te_mask_type_param = "attn_mask_type"
                elif "attention_mask_type" in params:
                    self._te_mask_type_param = "attention_mask_type"
                self._te_supports_mask = self._te_mask_param is not None
                self._te_supports_mask_type = (
                    self._te_mask_type_param is not None
                )
                self._te_supports_attention_dropout = (
                    "attention_dropout" in params
                )
                self._te_supports_is_causal = "is_causal" in params
                self._te_supports_training = "training" in params

    def forward(
        self: Self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args: Any,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        training: Optional[bool] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        training = bool(training) if training is not None else self.training
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise ValueError(
                f"DPA expects 4D inputs, got {q.shape}, {k.shape}, {v.shape}"
            )
        tracing = bool(
            is_symbolic()
            or is_meta_or_fake_tensor(q)
            or is_meta_or_fake_tensor(k)
            or is_meta_or_fake_tensor(v)
        )
        if tracing:
            assert_trace(
                q.size(0) == k.size(0),
                "Batch mismatch",
            )
            assert_trace(
                q.size(0) == v.size(0),
                "Batch mismatch",
            )
            assert_trace(
                q.size(1) == k.size(1),
                "Head mismatch",
            )
            assert_trace(
                q.size(1) == v.size(1),
                "Head mismatch",
            )
            assert_trace(
                k.size(2) == v.size(2),
                "K/V length mismatch",
            )
            assert_trace(
                q.size(3) == k.size(3),
                "Embed dim mismatch",
            )
            assert_trace(
                k.size(3) == v.size(3),
                "Embed dim mismatch",
            )
        else:
            if not (
                q.size(0) == k.size(0) == v.size(0)
                and q.size(1) == k.size(1) == v.size(1)
            ):
                raise ValueError("Batch/Head mismatch")
            if k.size(2) != v.size(2):
                raise ValueError("K/V length mismatch")
            if q.size(3) != k.size(3) or k.size(3) != v.size(3):
                raise ValueError("Embed dim mismatch")
        q_bshd, k_bshd, v_bshd = [
            self._negotiate_dtype(t).contiguous() for t in (q, k, v)
        ]
        if (
            not tracing
            and _is_bshd_contiguous(q_bshd)
            and _is_bshd_contiguous(k_bshd)
        ):
            with contextlib.suppress(Exception):
                capture(
                    q_bshd,
                    bwd_factor=2.0 if training else 0.0,
                    dropout_p=float(dropout_p),
                    training=training,
                )
        dropout_val = float(dropout_p) if training else 0.0
        B, H, L, D = q_bshd.shape
        S = k_bshd.shape[2]
        mask_bool, bias_float = None, None
        if attn_mask is not None:
            m = attn_mask
            if m.dtype == torch.bool:
                mask_bool = m
            else:
                bias_float = m
        mb = mh = mL = 0
        if mask_bool is not None:
            mask_bool = mask_bool.to(
                device=q_bshd.device, dtype=torch.bool, non_blocking=True
            )
            if not _exporting_boundary():
                mask_bool, mb, mh, mL = _flatten_attn_mask(
                    mask_bool, device=q_bshd.device, B=B, H=H, L=L, S=S
                )
                if not tracing and int(mh) == 1 and int(mL) == int(L):
                    with contextlib.suppress(Exception):
                        if int(mask_bool.stride(-2)) == 0:
                            mask_bool, mL = mask_bool[..., :1, :], 1
        if bias_float is not None:
            bias_float = bias_float.to(
                device=q_bshd.device, dtype=q_bshd.dtype, non_blocking=True
            )
            if not _exporting_boundary():
                bias_float, _, _, _ = _flatten_attn_mask(
                    bias_float, device=q_bshd.device, B=B, H=H, L=L, S=S
                )
        try:
            is_compiling = torch.compiler.is_compiling()
        except:
            is_compiling = False
        use_te = (
            self._te_ok
            and self.te_first
            and not self._force_pt
            and self._te_attn is not None
            and not is_compiling
            and not tracing
            and q_bshd.is_cuda
            and q_bshd.dtype in (torch.float16, torch.bfloat16)
        )
        te_mask, te_mask_type = None, None
        if use_te:
            if bias_float is not None:
                use_te = False
            elif mask_bool is None:
                te_mask_type = "causal" if bool(is_causal) else "no_mask"
            else:
                if not (int(mh) == 1 and int(mL) == 1):
                    use_te = False
                else:
                    te_mask = mask_bool
                    if int(mb) != int(B):
                        te_mask = te_mask.expand(int(B), 1, 1, int(S))
                    te_mask = te_mask.contiguous()
                    te_mask_type = (
                        "padding_causal" if bool(is_causal) else "padding"
                    )
            if use_te and te_mask is not None and not self._te_supports_mask:
                use_te = False
            if (
                use_te
                and te_mask is not None
                and te_mask_type
                and te_mask_type.startswith("padding")
                and not (
                    self._te_supports_mask_type and self._te_mask_type_param
                )
            ):
                use_te = False
            if use_te and (te_mask is None) and bool(is_causal):
                if not (
                    self._te_supports_mask_type or self._te_supports_is_causal
                ):
                    use_te = False
        if use_te:
            q_te = q_bshd.transpose(1, 2).contiguous()
            k_te = k_bshd.transpose(1, 2).contiguous()
            v_te = v_bshd.transpose(1, 2).contiguous()
            te_kwargs: dict[str, Any] = {}
            if self._te_supports_attention_dropout:
                te_kwargs["attention_dropout"] = dropout_val
            if (
                self._te_supports_mask_type
                and self._te_mask_type_param is not None
                and te_mask_type is not None
            ):
                te_kwargs[self._te_mask_type_param] = te_mask_type
            elif self._te_supports_is_causal:
                te_kwargs["is_causal"] = bool(is_causal)
            if self._te_supports_training:
                te_kwargs["training"] = training
            if (
                te_mask is not None
                and self._te_supports_mask
                and self._te_mask_param
            ):
                te_kwargs[self._te_mask_param] = te_mask
            try:
                out_te = self._te_attn(q_te, k_te, v_te, **te_kwargs)
            except Exception:
                self._force_pt = True
                use_te = False
            else:
                out_te = out_te.transpose(1, 2).contiguous()
                try:
                    B_, H_, L_, D_ = q_bshd.shape
                    flops = 4.0 * B_ * H_ * L_ * k_bshd.shape[2] * D_
                    if FLOP_PROFILER is not None and not tracing:
                        FLOP_PROFILER.add("DotProductAttention", float(flops))
                except:
                    pass
                return out_te
        final_mask: torch.Tensor | None = None
        sdpa_is_causal = bool(is_causal)
        if bias_float is None:
            if mask_bool is None:
                final_mask = None
            else:
                final_mask = ~mask_bool
                sdpa_is_causal = False
        else:
            if mask_bool is None:
                final_mask = bias_float
            else:
                mask_bias = torch.where(
                    mask_bool,
                    torch.full(
                        (),
                        torch.finfo(q_bshd.dtype).min,
                        dtype=q_bshd.dtype,
                        device=q_bshd.device,
                    ),
                    torch.zeros((), dtype=q_bshd.dtype, device=q_bshd.device),
                )
                final_mask = (mask_bias + bias_float).contiguous()
            sdpa_is_causal = False
        exporting = _exporting_boundary()
        disable_sdpa = env_bool(("ENN_DISABLE_SDPA",), default=False)
        force_sdpa = env_bool(("ENN_FORCE_SDPA",), default=False)
        if (
            (exporting and not force_sdpa)
            or disable_sdpa
            or (not q_bshd.is_cuda)
        ):
            fm = final_mask
            if fm is not None:
                fm = fm.to(
                    device=q_bshd.device,
                    dtype=(
                        torch.bool if fm.dtype is torch.bool else q_bshd.dtype
                    ),
                    non_blocking=True,
                )
                if fm.dim() != 4:
                    fm, _, _, _ = _flatten_attn_mask(
                        fm,
                        device=q_bshd.device,
                        B=B,
                        H=H,
                        L=q_bshd.shape[2],
                        S=k_bshd.shape[2],
                    )
            sdpa_out = _attention_math_bshd(
                q_bshd,
                k_bshd,
                v_bshd,
                attn_mask=fm,
                is_causal=bool(sdpa_is_causal),
                dropout_p=float(dropout_val),
                training=bool(training),
            )
        else:
            sdpa_kwargs = {
                "attn_mask": final_mask,
                "dropout_p": dropout_val,
                "is_causal": bool(sdpa_is_causal),
            }
            fm = final_mask
            if fm is not None:
                fm = fm.to(
                    device=q_bshd.device,
                    dtype=(
                        q_bshd.dtype
                        if not fm.dtype is torch.bool
                        else torch.bool
                    ),
                    non_blocking=True,
                )
                sdpa_kwargs["attn_mask"], bd, hd, _ = _flatten_attn_mask(
                    fm,
                    device=q_bshd.device,
                    B=B,
                    H=H,
                    L=q_bshd.shape[2],
                    S=k_bshd.shape[2],
                )
                if not tracing and (bd not in (1, B) or hd not in (1, H)):
                    raise RuntimeError("Attn mask mismatch")
                sdpa_kwargs["is_causal"] = False
            sdpa_out = None
            backends = get_dpa_backends() or []
            try:
                from torch.nn.attention import sdpa_kernel

                if backends:
                    with sdpa_kernel(backends):
                        sdpa_out = (
                            torch.nn.functional.scaled_dot_product_attention(
                                q_bshd, k_bshd, v_bshd, **sdpa_kwargs
                            )
                        )
            except Exception:
                pass
            if sdpa_out is None:
                sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                    q_bshd, k_bshd, v_bshd, **sdpa_kwargs
                )
        try:
            flops = 4.0 * B * H * q_bshd.shape[2] * k_bshd.shape[2] * D
            if FLOP_PROFILER is not None and not tracing:
                FLOP_PROFILER.add("DotProductAttention", float(flops))
        except:
            pass
        return sdpa_out

    @staticmethod
    def _negotiate_dtype(tensor: torch.Tensor) -> torch.Tensor:
        device_type = tensor.device.type
        if device_type == "cpu" and tensor.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            return tensor.float()
        if device_type == "mps" and tensor.dtype == torch.bfloat16:
            return tensor.to(torch.float16)
        return tensor


class FlexAttention(nn.Module):
    def __init__(self: Self, *args: Any, prefer_torch: bool = True) -> None:
        super().__init__()
        self.prefer_torch = bool(prefer_torch)

    @property
    def has_torch_backend(self: Self) -> bool:
        return bool(_HAS_TORCH_FLEX and _torch_flex_attention is not None)

    def _reference(
        self: Self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args: Any,
        block_mask: Any = None,
        scale: float | None = None,
        dropout_p: float = 0.0,
        training: bool = False,
        is_causal: bool = False,
        score_mod: Any = None,
    ) -> torch.Tensor:
        if score_mod is not None:
            raise RuntimeError(
                "FlexAttention reference path does not support score_mod"
            )
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise ValueError(
                f"FlexAttention expects (B,H,L,D), got {tuple(q.shape)}"
            )
        q_bshd, k_bshd, v_bshd = q, k, v
        _, _, Lq, Dh = q_bshd.shape
        Lk = k_bshd.shape[2]
        sc = (
            float(scale) if scale is not None else (1.0 / math.sqrt(float(Dh)))
        )
        scores = torch.matmul(q_bshd, k_bshd.transpose(-2, -1)) * sc
        if is_causal:
            with contextlib.suppress(Exception):
                causal = torch.ones(
                    (Lq, Lk), dtype=torch.bool, device=scores.device
                ).tril()
                scores = scores.masked_fill(
                    causal.logical_not(), torch.finfo(scores.dtype).min
                )
        dense = _blockmask_to_token_mask(
            block_mask,
            B=q_bshd.shape[0],
            H=q_bshd.shape[1],
            Lq=Lq,
            Lk=Lk,
            device=scores.device,
        )
        if dense is not None:
            if dense.dim() == 2:
                dense = dense[None, None, :, :]
            elif dense.dim() == 3:
                dense = dense[:, None, :, :]
            if dense.dtype == torch.bool or not torch.is_floating_point(dense):
                scores = _apply_allowed_mask(scores, dense)
            else:
                scores = scores + dense
        attn = torch.softmax(scores, dim=-1)
        if training and float(dropout_p) > 0.0:
            attn = torch.nn.functional.dropout(
                attn, p=float(dropout_p), training=True
            )
        return torch.matmul(attn, v_bshd)

    def forward(
        self: Self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args: Any,
        score_mod: Any = None,
        block_mask: Any = None,
        scale: float | None = None,
        enable_gqa: bool = False,
        return_lse: bool = False,
        kernel_options: Any = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        training: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        del args, kwargs
        training = bool(training) if training is not None else self.training
        if _exporting_boundary():
            return self._reference(
                q,
                k,
                v,
                block_mask=block_mask,
                scale=scale,
                dropout_p=float(dropout_p),
                training=training,
                is_causal=bool(is_causal),
                score_mod=score_mod,
            )
        if (
            self.prefer_torch
            and self.has_torch_backend
            and (not _flex_attention_disabled())
        ):
            if (not q.is_cuda) or _torch_flex_attention is None:
                return self._reference(
                    q,
                    k,
                    v,
                    block_mask=block_mask,
                    scale=scale,
                    dropout_p=float(dropout_p),
                    training=training,
                    is_causal=bool(is_causal),
                    score_mod=score_mod,
                )
            flex_kwargs: dict[str, Any] = {}
            if "score_mod" in _FLEX_KWARGS and score_mod is not None:
                flex_kwargs["score_mod"] = score_mod
            if "block_mask" in _FLEX_KWARGS and block_mask is not None:
                flex_kwargs["block_mask"] = block_mask
            if "scale" in _FLEX_KWARGS and scale is not None:
                flex_kwargs["scale"] = float(scale)
            if "enable_gqa" in _FLEX_KWARGS:
                flex_kwargs["enable_gqa"] = bool(enable_gqa)
            if "return_lse" in _FLEX_KWARGS:
                flex_kwargs["return_lse"] = bool(return_lse)
            if float(dropout_p) > 0.0:
                if "dropout_p" in _FLEX_KWARGS:
                    flex_kwargs["dropout_p"] = float(dropout_p)
                elif "dropout" in _FLEX_KWARGS:
                    flex_kwargs["dropout"] = float(dropout_p)
            if "kernel_options" in _FLEX_KWARGS and kernel_options is not None:
                flex_kwargs["kernel_options"] = kernel_options
            flex_fn, flex_key = _get_compiled_flex_attention_for_kwargs(
                q, flex_kwargs
            )
            try:
                if flex_fn is _torch_flex_attention:
                    return _call_torch_flex_attention_eager(
                        q, k, v, flex_kwargs=flex_kwargs
                    )

                def _run(fn: Callable[[], Any]) -> Any:
                    if not is_dynamo_compiling():
                        with skip_non_infra_dispatch_mode():
                            return fn()
                    return fn()

                mode_key = (
                    str(flex_key[1])
                    if isinstance(flex_key, tuple) and len(flex_key) > 1
                    else ""
                )
                needs_verify = (
                    mode_key in {"max-autotune", "max-autotune-no-cudagraphs"}
                ) and (not is_dynamo_compiling())
                verified = False
                if needs_verify:
                    with _FLEX_ATTN_VERIFIED_LOCK:
                        verified = flex_key in _FLEX_ATTN_VERIFIED
                if not needs_verify or verified:
                    return _run(lambda: flex_fn(q, k, v))
                out, saw_uncompiled = _call_with_flex_warn_guard(
                    lambda: _run(lambda: flex_fn(q, k, v))
                )
                if not saw_uncompiled:
                    with _FLEX_ATTN_VERIFIED_LOCK:
                        _FLEX_ATTN_VERIFIED.add(flex_key)
                    return out
                for fb_mode in _flex_attention_fallback_modes(mode_key):
                    try:
                        dyn_fb = _flex_attention_dynamic_flag(fb_mode)
                        fb_fn = _compile_flex_attention_wrapper(
                            mode=fb_mode,
                            dynamic=dyn_fb,
                            flex_kwargs=flex_kwargs,
                        )
                        if fb_fn is _torch_flex_attention:
                            continue
                        _FLEX_ATTN_SPECIALIZED[flex_key] = fb_fn
                        out2, saw2 = _call_with_flex_warn_guard(
                            lambda: _run(lambda: fb_fn(q, k, v))
                        )
                        if not saw2:
                            with _FLEX_ATTN_VERIFIED_LOCK:
                                _FLEX_ATTN_VERIFIED.add(flex_key)
                            return out2
                    except Exception:
                        pass
                with contextlib.suppress(Exception):
                    _FLEX_ATTN_FAILED[flex_key] = "uncompiled-warning"
                    _FLEX_ATTN_SPECIALIZED.pop(flex_key, None)
                return _call_torch_flex_attention_eager(
                    q, k, v, flex_kwargs=flex_kwargs
                )
            except Exception as exc:
                if _is_compile_failure(exc):
                    with contextlib.suppress(Exception):
                        _FLEX_ATTN_FAILED[flex_key] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        _FLEX_ATTN_SPECIALIZED.pop(flex_key, None)
                    if _looks_like_triton_resource_error(exc) and (
                        "kernel_options" in _FLEX_KWARGS
                    ):
                        try:
                            flex_kwargs2 = dict(flex_kwargs)
                            existing = flex_kwargs2.get("kernel_options", None)
                            flex_kwargs2["kernel_options"] = (
                                _resource_safe_kernel_options(existing)
                            )
                            flex_fn2, flex_key2 = (
                                _get_compiled_flex_attention_for_kwargs(
                                    q, flex_kwargs2
                                )
                            )
                            if flex_fn2 is not _torch_flex_attention:
                                try:
                                    return flex_fn2(q, k, v)
                                except Exception as exc2:
                                    if _is_compile_failure(exc2):
                                        with contextlib.suppress(Exception):
                                            _FLEX_ATTN_FAILED[flex_key2] = (
                                                f"{type(exc2).__name__}: {exc2}"
                                            )
                                            _FLEX_ATTN_SPECIALIZED.pop(
                                                flex_key2, None
                                            )
                                    else:
                                        raise
                            return _call_torch_flex_attention_eager(
                                q, k, v, flex_kwargs=flex_kwargs2
                            )
                        except Exception:
                            pass
                    _warn_once(
                        f"flexattn-runtime-failed-{hash(flex_key)}",
                        "FlexAttention: compiled execution failed; falling back to eager.\n"
                        f"{type(exc).__name__}: {exc}",
                    )
                    return _call_torch_flex_attention_eager(
                        q, k, v, flex_kwargs=flex_kwargs
                    )
                raise
        return self._reference(
            q,
            k,
            v,
            block_mask=block_mask,
            scale=scale,
            dropout_p=float(dropout_p),
            training=training,
            is_causal=bool(is_causal),
            score_mod=score_mod,
        )


class MultiScaleRetention(nn.Module):
    def __init__(
        self: Self, d_model: int, nhead: int, use_gate: bool = True
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        if self.d_model % max(1, self.nhead) != 0:
            raise ValueError(
                f"MultiScaleRetention: d_model={self.d_model} must be divisible by nhead={self.nhead}"
            )
        self.head_dim = int(self.d_model // max(1, self.nhead))
        self.use_gate = bool(use_gate)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.g_proj = (
            nn.Linear(self.d_model, self.d_model, bias=False)
            if self.use_gate
            else None
        )
        self.norm = nn.LayerNorm(self.d_model)
        self._triton_ok = bool(_HAS_TRITON_MSR and torch.cuda.is_available())
        self._decay_init = 5.0
        self._decay_range = 1.0
        heads = torch.arange(self.nhead, dtype=torch.float32)
        beta_init = float(self._decay_init) + float(self._decay_range) * (
            heads / float(max(self.nhead, 1))
        )
        self._beta = nn.Parameter(beta_init)

    def _decay_lambda(
        self: Self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        H = int(self.nhead)
        calc_dtype = (
            dtype if dtype in (torch.float32, torch.float64) else torch.float32
        )
        beta = getattr(self, "_beta", None)
        if not (
            isinstance(beta, torch.Tensor)
            and (is_tracing_or_exporting() or beta.numel() == H)
        ):
            beta = float(self._decay_init) + float(self._decay_range) * (
                torch.arange(H, device=device, dtype=calc_dtype)
                / float(max(H, 1))
            )
        gammas = (
            1.0 - torch.pow(2.0, -beta.to(device=device, dtype=calc_dtype))
        ).clamp(min=torch.finfo(calc_dtype).tiny, max=1.0 - 1e-9)
        if calc_dtype != dtype:
            gammas = gammas.to(dtype=dtype)
        return gammas

    @staticmethod
    def _apply_kpm_to_v(
        v: torch.Tensor, attn_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if not isinstance(attn_mask, torch.Tensor):
            return v
        if attn_mask.dim() != 2 or attn_mask.dtype is not torch.bool:
            return v
        trace_like = bool(is_symbolic())
        if not trace_like:
            B, L = int(v.shape[0]), int(v.shape[1])
            if attn_mask.shape != (B, L):
                return v
        else:
            assert_trace(
                attn_mask.shape[0] == v.shape[0],
                "attn_mask batch mismatch",
            )
            assert_trace(
                attn_mask.shape[1] == v.shape[1],
                "attn_mask length mismatch",
            )
        mask = (
            attn_mask.to(device=v.device, non_blocking=True)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        return torch.where(mask, torch.zeros_like(v), v)

    @staticmethod
    def _extract_state_tensor(
        state: Any, *args: Any, B: int, H: int
    ) -> Optional[torch.Tensor]:
        if state is None:
            return None
        st: Optional[torch.Tensor] = None
        if isinstance(state, torch.Tensor):
            st = state
        elif isinstance(state, Mapping):
            for key in ("state", "msr_state", "retention_state"):
                v = state.get(key, None)
                if isinstance(v, torch.Tensor):
                    st = v
                    break
        if st is None:
            return None
        trace_like = bool(is_symbolic())
        if st.dim() == 4:
            st = st.squeeze(2)
        if st.dim() != 3:
            return None
        if trace_like:
            assert_trace(
                st.shape[0] == B,
                "state batch mismatch",
            )
            assert_trace(
                st.shape[1] == H,
                "state head mismatch",
            )
        else:
            if tuple(st.shape[:2]) != (int(B), int(H)):
                return None
        return st

    @staticmethod
    def _select_last_state(
        state_tensor: torch.Tensor, attn_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if state_tensor.dim() != 4:
            raise ValueError(
                f"_select_last_state expects (B,L,H,Dh), got {tuple(state_tensor.shape)}"
            )
        B, L, H, Dh = state_tensor.shape
        trace_like = bool(is_symbolic())
        if trace_like:
            assert_trace(
                L > 0,
                "empty sequence",
            )
        else:
            if L <= 0:
                return state_tensor.new_zeros((B, H, Dh))
        if (
            isinstance(attn_mask, torch.Tensor)
            and attn_mask.dim() == 2
            and attn_mask.dtype is torch.bool
        ):
            if trace_like:
                assert_trace(
                    attn_mask.shape[0] == B,
                    "attn_mask batch mismatch",
                )
                assert_trace(
                    attn_mask.shape[1] == L,
                    "attn_mask length mismatch",
                )
            else:
                if tuple(attn_mask.shape) != (B, L):
                    return state_tensor[:, -1]
            lengths = (
                (~attn_mask).to(dtype=torch.int64).sum(dim=1).clamp(min=1)
            )
            idx = (lengths - 1).clamp(min=0, max=L - 1)
            gather_idx = idx.view(B, 1, 1, 1).expand(-1, -1, H, Dh)
            return torch.gather(state_tensor, dim=1, index=gather_idx).squeeze(
                1
            )
        return state_tensor[:, -1]

    @staticmethod
    def _scan_causal_torch(
        v: torch.Tensor, lam_h: torch.Tensor
    ) -> torch.Tensor:
        B, L, H, Dh = v.shape
        trace_like = bool(is_symbolic())
        if trace_like:
            assert_trace(
                L > 0,
                "empty sequence",
            )
        else:
            if L <= 0:
                return v.new_zeros(v.shape)
        calc_dtype = (
            torch.float32
            if v.dtype in (torch.float16, torch.bfloat16)
            else v.dtype
        )
        lam_calc = lam_h.to(dtype=calc_dtype, device=v.device).view(1, 1, H, 1)
        t = torch.arange(L, device=v.device, dtype=calc_dtype).view(1, L, 1, 1)
        p = torch.pow(lam_calc, t)
        tiny = torch.finfo(calc_dtype).tiny
        p = p.clamp_min(tiny)
        inv_p = torch.reciprocal(p)
        v_scaled = v.to(dtype=calc_dtype) * inv_p
        v_scaled[:, 0].zero_()
        cumsum_scaled = torch.cumsum(v_scaled, dim=1)
        prev_scaled = v[:, 0].to(dtype=calc_dtype).unsqueeze(1)
        state = p * (prev_scaled + cumsum_scaled)
        return state.to(dtype=v.dtype).contiguous()

    @torch_compiler_disable(recursive=False, reason="Triton retention scan")
    def _scan_causal_triton(
        self: Self, v: torch.Tensor, lam_h: torch.Tensor
    ) -> torch.Tensor:
        if v.dim() != 4:
            raise ValueError(
                f"_scan_causal_triton expects (B,L,H,Dh), got {tuple(v.shape)}"
            )
        if v.device.type != "cuda":
            raise RuntimeError("_scan_causal_triton requires CUDA tensor")
        B, L, H, Dh = v.shape
        out = torch.empty_like(v, dtype=v.dtype)
        SVB, SVL, SVH, SVD = v.stride()
        SOB, SOL, SOH, SOD = out.stride()
        env_block = env_str("ENN_MSR_TRITON_BLOCK_DH") or ""
        if env_block:
            try:
                BLOCK_DH = int(env_block)
                if BLOCK_DH <= 0:
                    raise ValueError
            except Exception:
                BLOCK_DH = 64 if Dh >= 64 else 32
        else:
            BLOCK_DH = 64 if Dh >= 64 else 32
        env_warps = env_str("ENN_MSR_TRITON_NUM_WARPS") or ""
        if env_warps:
            try:
                num_warps = int(env_warps)
                if num_warps <= 0:
                    raise ValueError
            except Exception:
                num_warps = 8 if BLOCK_DH >= 64 else 4
        else:
            num_warps = 8 if BLOCK_DH >= 64 else 4
        grid = (B * H, (Dh + BLOCK_DH - 1) // BLOCK_DH)
        _triton_retention[grid](
            v,
            lam_h,
            out,
            B,
            L,
            H,
            Dh,
            SVB,
            SVL,
            SVH,
            SVD,
            SOB,
            SOL,
            SOH,
            SOD,
            BLOCK_DH=BLOCK_DH,
            num_warps=num_warps,
        )
        return out

    def _scan_causal(
        self: Self, v: torch.Tensor, lam_h: torch.Tensor
    ) -> torch.Tensor:
        disable_triton = env_bool("ENN_MSR_FORCE_TORCH", default=False)
        use_triton = bool(
            (not disable_triton)
            and self._triton_ok
            and v.is_cuda
            and torch.cuda.is_available()
            and (not _exporting_boundary())
        )
        if use_triton:
            try:
                return self._scan_causal_triton(v.contiguous(), lam_h)
            except Exception:
                pass
        return self._scan_causal_torch(v, lam_h)

    def forward(
        self: Self,
        x: torch.Tensor,
        *args: Any,
        decay: Any = None,
        attn_mask: torch.Tensor | None = None,
        state: Any = None,
        return_state: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        restore_dtype: torch.dtype | None = None
        x_in = x
        if (
            getattr(x_in.device, "type", "cpu") == "mps"
            and x_in.dtype == torch.bfloat16
        ):
            restore_dtype = x_in.dtype
            x_in = x.to(torch.float16)
        if x_in.dim() != 3:
            raise ValueError(
                f"MultiScaleRetention expects (B,L,D), got {tuple(x_in.shape)}"
            )
        B, L, D = x_in.shape
        trace_like = bool(is_symbolic())
        if (not trace_like) and L <= 0:
            out0 = x_in.new_zeros(x_in.shape)
            if bool(return_state):
                st0 = x_in.new_zeros((B, self.nhead, int(self.head_dim)))
                if restore_dtype is not None:
                    out0 = out0.to(restore_dtype)
                    st0 = st0.to(restore_dtype)
                return out0, st0
            return (
                out0.to(restore_dtype) if restore_dtype is not None else out0
            )
        if (not trace_like) and D != int(self.d_model):
            raise ValueError(
                f"Last dimension {D} must equal d_model={int(self.d_model)}"
            )
        decay_arg = None
        if args:
            decay_arg = args[0]
        else:
            decay_arg = decay
        head_dim = int(self.head_dim)
        q = self.q_proj(x_in).view(B, L, self.nhead, head_dim)
        v = self.v_proj(x_in).view(B, L, self.nhead, head_dim)
        v = self._apply_kpm_to_v(v, attn_mask)
        lam_h = self._decay_lambda(v.device, v.dtype).to(
            dtype=v.dtype, device=v.device
        )
        if isinstance(decay_arg, torch.Tensor):
            if decay_arg.dim() == 1:
                if trace_like:
                    assert_trace(
                        decay_arg.shape[0] == self.nhead,
                        "decay[H] shape mismatch",
                    )
                else:
                    if int(decay_arg.shape[0]) != int(self.nhead):
                        decay_arg = None
                if decay_arg is not None:
                    lam_h = decay_arg.to(dtype=v.dtype, device=v.device)
            elif decay_arg.dim() == 3:
                if trace_like:
                    assert_trace(
                        decay_arg.shape[0] == self.nhead,
                        "decay[H,*,*] shape mismatch",
                    )
                else:
                    if int(decay_arg.shape[0]) != int(self.nhead):
                        decay_arg = None
                if decay_arg is not None:
                    lam_h = decay_arg[:, 1, 0].to(
                        dtype=v.dtype, device=v.device
                    )
        st_bhd = self._extract_state_tensor(state, B=B, H=int(self.nhead))
        if st_bhd is not None:
            v = v.clone()
            v[:, 0] = v[:, 0] + lam_h.view(1, self.nhead, 1) * st_bhd.to(
                dtype=v.dtype, device=v.device
            )
        state_tensor = self._scan_causal(v, lam_h)
        y = (q * state_tensor).contiguous().view(B, L, self.d_model)
        y = self.norm(y)
        if self.use_gate and self.g_proj is not None:
            gate = torch.nn.functional.silu(self.g_proj(x_in))
            y = y * gate
        out = self.o_proj(y)
        last_state: Optional[torch.Tensor] = None
        if bool(return_state):
            last_state = self._select_last_state(
                state_tensor, attn_mask
            ).contiguous()
            if not torch.is_grad_enabled():
                last_state = last_state.detach()
        if (FLOP_PROFILER is not None) and (not trace_like):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    fl = _compute_flops_msr(
                        B,
                        L,
                        num_heads=int(self.nhead),
                        head_dim=int(head_dim),
                        use_gate=bool(
                            self.use_gate and self.g_proj is not None
                        ),
                    )
                    if fl > 0.0:
                        FLOP_PROFILER.add("MultiScaleRetention", float(fl))
                except Exception:
                    pass
        if restore_dtype is not None:
            out = out.to(restore_dtype)
            if last_state is not None:
                last_state = last_state.to(restore_dtype)
        if bool(return_state):
            if last_state is None:
                last_state = state_tensor.new_zeros((B, self.nhead, head_dim))
            return out, last_state
        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self: Self,
        embed_dim: int,
        num_heads: int,
        *args: Any,
        bias: bool = True,
        dropout: float = 0.0,
        batch_first: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._backend = "torch"
        self.impl = _MultiHeadAttentionCompat(
            embed_dim,
            num_heads,
            bias=bias,
            dropout=dropout,
            batch_first=batch_first,
            **kwargs,
        )
        if _is_nvidia_mha_preferred():
            try:
                impl = _MultiHeadAttentionNvidia(
                    embed_dim,
                    num_heads,
                    bias=bias,
                    dropout=dropout,
                    batch_first=batch_first,
                    **kwargs,
                )
                if impl._te_mha is not None:
                    self.impl, self._backend = impl, "te"
            except Exception:
                pass
        if isinstance(self.impl, _MultiHeadAttentionNvidia):
            self.impl._fallback = _MultiHeadAttentionCompat(
                embed_dim,
                num_heads,
                bias=bias,
                dropout=dropout,
                batch_first=batch_first,
                **kwargs,
            )

    @property
    def backend(self: Self) -> str:
        return self._backend

    def forward(
        self: Self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        is_causal: Optional[bool] = None,
        average_attn_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.impl(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )


if _flex_attention_disabled():
    _HAS_TORCH_FLEX = False
