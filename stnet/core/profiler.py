# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import contextvars
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache, partial
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "FLOP_PROFILER",
    "FlopCounter",
    "capture",
]


# ==============================================================================
# Utilities
# ==============================================================================

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if v == v else default  # NaN guard
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _prod_int(xs: Sequence[int]) -> int:
    p = 1
    for v in xs:
        p *= int(v)
    return int(p)


def _get_out_tensor(out: Any) -> Optional[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for v in out:
            if isinstance(v, torch.Tensor):
                return v
    return None


def _first_tensors_from_args(args: Tuple[Any, ...], max_n: int = 4) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for a in args:
        if isinstance(a, torch.Tensor):
            out.append(a)
            if len(out) >= max_n:
                break
        elif isinstance(a, (tuple, list)):
            for v in a:
                if isinstance(v, torch.Tensor):
                    out.append(v)
                    if len(out) >= max_n:
                        return out
    return out


def _infer_bhsd(x: torch.Tensor) -> Tuple[int, int, int, int]:
    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        return (0, 0, 0, 0)
    b = int(x.shape[0])
    d = int(x.shape[-1])
    a1 = int(x.shape[1])
    a2 = int(x.shape[2])
    if a1 <= a2:
        h = a1
        s = a2
    else:
        s = a1
        h = a2
    return (b, h, s, d)


def _infer_bhsd_shape(shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    if not shape or len(shape) != 4:
        return (0, 0, 0, 0)
    b = int(shape[0])
    d = int(shape[-1])
    a1 = int(shape[1])
    a2 = int(shape[2])
    if a1 <= a2:
        h = a1
        s = a2
    else:
        s = a1
        h = a2
    return (b, h, s, d)


# ==============================================================================
# FLOP estimation primitives (approximate but consistent)
# ==============================================================================

def _linear_mkn(inp: torch.Tensor, out: Any, weight: Optional[torch.Tensor]) -> Tuple[int, int, int]:
    if not isinstance(inp, torch.Tensor) or inp.numel() == 0:
        return (0, 0, 0)

    if isinstance(weight, torch.Tensor) and weight.ndim >= 2:
        # [out_features, in_features]
        n_dim = _safe_int(weight.shape[0], 0)
        k_dim = _safe_int(weight.shape[-1], 0)
    else:
        k_dim = _safe_int(inp.shape[-1], 0)
        if isinstance(out, torch.Tensor) and out.numel() > 0 and out.ndim >= 1:
            n_dim = _safe_int(out.shape[-1], 0)
        else:
            n_dim = 0

    if k_dim <= 0 or n_dim <= 0:
        return (0, 0, 0)

    m_dim = _safe_int(inp.numel() // max(k_dim, 1), 0)
    if isinstance(out, torch.Tensor) and out.numel() > 0:
        m_dim = max(m_dim, _safe_int(out.numel() // max(n_dim, 1), 0))
    return (m_dim, k_dim, n_dim)


def _flops_linear(
    inp: torch.Tensor,
    out: Any,
    weight: Optional[torch.Tensor],
    *,
    include_bias: bool,
    has_bias: bool,
    effective_bwd: float,
) -> float:
    m_dim, k_dim, n_dim = _linear_mkn(inp, out, weight)
    if m_dim <= 0 or k_dim <= 0 or n_dim <= 0:
        return 0.0
    bias_cost = float(m_dim * n_dim) if (include_bias and has_bias) else 0.0
    fwd = 2.0 * m_dim * k_dim * n_dim + bias_cost
    return float(fwd * (1.0 + max(0.0, _safe_float(effective_bwd, 0.0))))


def _flops_conv(
    inp: torch.Tensor,
    out: Any,
    weight: Optional[torch.Tensor],
    *,
    groups: int,
    include_bias: bool,
    has_bias: bool,
    effective_bwd: float,
) -> float:
    if not isinstance(weight, torch.Tensor) or weight.ndim < 3:
        return 0.0
    out_t = _get_out_tensor(out)
    if out_t is None or out_t.numel() == 0:
        return 0.0

    try:
        out_elems = int(out_t.numel())
        g = max(1, int(groups))

        if isinstance(inp, torch.Tensor) and inp.ndim >= 2:
            cin_total = int(inp.shape[1])
        else:
            cin_total = int(weight.shape[1] * g)

        cin_per_group = max(1, cin_total // g)
        kernel_spatial = int(weight[0].numel()) // max(cin_per_group, 1)

        fwd = out_elems * (2.0 * cin_per_group * kernel_spatial)
        if include_bias and has_bias:
            fwd += float(out_elems)
        return float(fwd * (1.0 + max(0.0, _safe_float(effective_bwd, 0.0))))
    except Exception:
        return 0.0


def _flops_elementwise(out: Any, *, coeff: float, effective_bwd: float) -> float:
    out_t = _get_out_tensor(out)
    if out_t is None or out_t.numel() == 0:
        return 0.0
    fwd = float(out_t.numel()) * float(coeff)
    return float(fwd * (1.0 + max(0.0, _safe_float(effective_bwd, 0.0))))


def _flops_softmax(inp: torch.Tensor, out: Any, *, dim: int, effective_bwd: float) -> float:
    out_t = _get_out_tensor(out)
    if not isinstance(inp, torch.Tensor) or inp.numel() == 0:
        return 0.0
    if out_t is None or out_t.numel() == 0:
        return 0.0
    nd = int(inp.ndim)
    if nd <= 0:
        return 0.0

    d = int(dim)
    if d < 0:
        d += nd
    if d < 0 or d >= nd:
        return 0.0

    cols = int(inp.shape[d])
    if cols <= 0:
        return 0.0
    rows = int(inp.numel() // max(cols, 1))

    # stable softmax approx per row:
    # max-reduction (cols comps), sub (cols), exp (cols), sum (cols adds), div (cols)
    fwd = float(rows) * (5.0 * float(cols))
    return float(fwd * (1.0 + max(0.0, _safe_float(effective_bwd, 0.0))))


def _flops_layernorm(
    inp: torch.Tensor,
    out: Any,
    *,
    normalized_shape: Sequence[int],
    elementwise_affine: bool,
    has_bias: bool,
    effective_bwd: float,
) -> float:
    out_t = _get_out_tensor(out)
    if not isinstance(inp, torch.Tensor) or inp.numel() == 0:
        return 0.0
    if out_t is None or out_t.numel() == 0:
        return 0.0

    n_norm = _prod_int([int(x) for x in normalized_shape]) if normalized_shape else int(inp.shape[-1])
    if n_norm <= 0:
        return 0.0
    groups = int(out_t.numel() // max(n_norm, 1))

    # Rough:
    # mean ~ N, var ~ 3N, normalize ~ 2N => 6N
    # affine: weight mul (N) + bias add (N if bias)
    affine = 0.0
    if elementwise_affine:
        affine += 1.0
        if has_bias:
            affine += 1.0

    fwd = float(groups) * float(n_norm) * (6.0 + affine)
    return float(fwd * (1.0 + max(0.0, _safe_float(effective_bwd, 0.0))))


def _flops_attention_general(
    *,
    batch: int,
    num_heads: int,
    q_len: int,
    k_len: int,
    head_dim: int,
    effective_bwd: float,
    dropout_p: float,
    training: bool,
    include_softmax_scale_dropout: bool,
) -> float:
    if batch <= 0 or q_len <= 0 or k_len <= 0 or num_heads <= 0 or head_dim <= 0:
        return 0.0

    # QK^T: 2*B*H*Q*K*D, AV: 2*B*H*Q*K*D => 4*...
    matmul = 4.0 * batch * num_heads * q_len * k_len * head_dim

    misc = 0.0
    if include_softmax_scale_dropout:
        misc_coeff = 6.0
        if training and dropout_p > 0.0:
            misc_coeff += 1.0
        misc = misc_coeff * (batch * num_heads * q_len * k_len)

    fwd = matmul + misc
    return float(fwd * (1.0 + max(0.0, _safe_float(effective_bwd, 0.0))))


def _flops_attention_from_qkv(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    *,
    effective_bwd: float,
    dropout_p: float,
    training: bool,
    include_softmax_scale_dropout: bool,
) -> float:
    if not isinstance(q, torch.Tensor) or q.ndim != 4:
        return 0.0
    b, h, qlen, d = _infer_bhsd(q)
    klen = qlen
    if isinstance(k, torch.Tensor) and k.ndim == 4:
        _, _, klen, _ = _infer_bhsd(k)
    return _flops_attention_general(
        batch=b,
        num_heads=h,
        q_len=qlen,
        k_len=klen,
        head_dim=d,
        effective_bwd=effective_bwd,
        dropout_p=dropout_p,
        training=training,
        include_softmax_scale_dropout=include_softmax_scale_dropout,
    )


# ==============================================================================
# Manual hook registry
# ==============================================================================

@dataclass(frozen=True)
class _HookConfig:
    include_bias: bool
    effective_bwd: float
    count_activations: bool
    count_norms: bool
    count_softmax: bool
    count_dropout: bool
    count_embedding: bool


_ACT_COEFF: Dict[type, float] = {
    nn.ReLU: 1.0,
    nn.ReLU6: 1.0,
    nn.Sigmoid: 4.0,
    nn.Tanh: 4.0,
    nn.GELU: 8.0,
    nn.SiLU: 6.0,
    nn.ELU: 4.0,
    nn.LeakyReLU: 1.0,
    nn.Hardswish: 6.0,
    nn.Hardsigmoid: 4.0,
}

_ACT_CLASSES: Tuple[type, ...] = tuple(_ACT_COEFF.keys())


@lru_cache(maxsize=128)
def _act_coeff_for_type(t: type) -> Optional[float]:
    coeff = _ACT_COEFF.get(t)
    if coeff is not None:
        return float(coeff)
    # Handle subclasses (rare, but possible).
    for cls, c in _ACT_COEFF.items():
        try:
            if issubclass(t, cls):
                return float(c)
        except Exception:
            continue
    return None


def _is_te_module(mod: nn.Module) -> bool:
    return "transformer_engine" in getattr(type(mod), "__module__", "")


def _register_linear(mod: nn.Module, inp: Tuple[Any, ...], out: Any, *, profiler: "_FlopProfiler", cfg: _HookConfig) -> None:
    x = inp[0] if inp else None
    if not isinstance(x, torch.Tensor):
        return

    weight = getattr(mod, "weight", None)
    if weight is None:
        inner = getattr(mod, "linear", None)
        weight = getattr(inner, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return

    has_bias = getattr(mod, "bias", None) is not None
    val = _flops_linear(x, out, weight, include_bias=cfg.include_bias, has_bias=has_bias, effective_bwd=cfg.effective_bwd)
    if val > 0.0:
        typ = type(mod).__name__
        if _is_te_module(mod):
            typ = f"TE.{typ}"
        profiler.add(typ, val)


def _register_conv(mod: nn.Module, inp: Tuple[Any, ...], out: Any, *, profiler: "_FlopProfiler", cfg: _HookConfig) -> None:
    x = inp[0] if inp else None
    if not isinstance(x, torch.Tensor):
        return
    weight = getattr(mod, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return

    groups = getattr(mod, "groups", 1)
    has_bias = getattr(mod, "bias", None) is not None
    val = _flops_conv(
        x, out, weight,
        groups=int(groups),
        include_bias=cfg.include_bias,
        has_bias=has_bias,
        effective_bwd=cfg.effective_bwd,
    )
    if val > 0.0:
        profiler.add(type(mod).__name__, val)


def _register_activation(mod: nn.Module, inp: Tuple[Any, ...], out: Any, *, profiler: "_FlopProfiler", cfg: _HookConfig) -> None:
    if not cfg.count_activations:
        return
    coeff = _act_coeff_for_type(type(mod))
    if coeff is None:
        return
    val = _flops_elementwise(out, coeff=float(coeff), effective_bwd=cfg.effective_bwd)
    if val > 0.0:
        profiler.add(type(mod).__name__, val)


def _register_dropout(mod: nn.Module, inp: Tuple[Any, ...], out: Any, *, profiler: "_FlopProfiler", cfg: _HookConfig) -> None:
    if not cfg.count_dropout:
        return
    training = bool(getattr(mod, "training", False))
    if not training:
        return
    val = _flops_elementwise(out, coeff=2.0, effective_bwd=cfg.effective_bwd)
    if val > 0.0:
        profiler.add(type(mod).__name__, val)


def _register_softmax(mod: nn.Module, inp: Tuple[Any, ...], out: Any, *, profiler: "_FlopProfiler", cfg: _HookConfig) -> None:
    if not cfg.count_softmax:
        return
    x = inp[0] if inp else None
    if not isinstance(x, torch.Tensor):
        return
    dim = getattr(mod, "dim", -1)
    val = _flops_softmax(x, out, dim=int(dim), effective_bwd=cfg.effective_bwd)
    if val > 0.0:
        profiler.add(type(mod).__name__, val)


def _register_layernorm(mod: nn.Module, inp: Tuple[Any, ...], out: Any, *, profiler: "_FlopProfiler", cfg: _HookConfig) -> None:
    if not cfg.count_norms:
        return
    x = inp[0] if inp else None
    if not isinstance(x, torch.Tensor):
        return
    normalized_shape = getattr(mod, "normalized_shape", ())
    elementwise_affine = bool(getattr(mod, "elementwise_affine", True))
    has_bias = getattr(mod, "bias", None) is not None
    val = _flops_layernorm(
        x, out,
        normalized_shape=list(normalized_shape) if normalized_shape is not None else (),
        elementwise_affine=elementwise_affine,
        has_bias=has_bias,
        effective_bwd=cfg.effective_bwd,
    )
    if val > 0.0:
        profiler.add(type(mod).__name__, val)


def _register_embedding(mod: nn.Module, inp: Tuple[Any, ...], out: Any, *, profiler: "_FlopProfiler", cfg: _HookConfig) -> None:
    # Gather is mostly memory traffic; FLOPs ~ 0
    return


def _register_mha(mod: nn.MultiheadAttention, inp: Tuple[Any, ...], out: Any, *, profiler: "_FlopProfiler", cfg: _HookConfig) -> None:
    # Best-effort: count in-proj + attention core + out-proj
    if not inp:
        return
    q = inp[0]
    if not isinstance(q, torch.Tensor) or q.numel() == 0:
        return
    if q.ndim != 3:
        return

    batch_first = bool(getattr(mod, "batch_first", False))
    if batch_first:
        bsz, seq_len, embed_dim = q.shape
    else:
        seq_len, bsz, embed_dim = q.shape

    num_heads = int(getattr(mod, "num_heads", 0))
    if num_heads <= 0:
        return
    head_dim = int(embed_dim // num_heads) if int(embed_dim) % num_heads == 0 else int(getattr(mod, "head_dim", 0))
    if head_dim <= 0:
        return

    m = int(bsz) * int(seq_len)
    E = int(embed_dim)

    # QKV projection
    inproj_has_bias = getattr(mod, "in_proj_bias", None) is not None
    inproj_fwd = 2.0 * m * E * (3.0 * E)
    if cfg.include_bias and inproj_has_bias:
        inproj_fwd += float(m * (3 * E))
    inproj = float(inproj_fwd * (1.0 + max(0.0, cfg.effective_bwd)))

    p = float(getattr(mod, "dropout", 0.0))
    training = bool(getattr(mod, "training", False))
    attn = _flops_attention_general(
        batch=int(bsz),
        num_heads=int(num_heads),
        q_len=int(seq_len),
        k_len=int(seq_len),
        head_dim=int(head_dim),
        effective_bwd=cfg.effective_bwd,
        dropout_p=p,
        training=training,
        include_softmax_scale_dropout=True,
    )

    outproj_fwd = 2.0 * m * E * E
    outproj = float(outproj_fwd * (1.0 + max(0.0, cfg.effective_bwd)))

    total = float(inproj + attn + outproj)
    if total > 0.0:
        profiler.add(type(mod).__name__, total)


# ==============================================================================
# Aggressive TE weight discovery (cached)
# ==============================================================================

def _get_tensor_attr(mod: nn.Module, name: str) -> Optional[torch.Tensor]:
    try:
        v = getattr(mod, name)
        return v if isinstance(v, torch.Tensor) else None
    except Exception:
        return None


def _find_2d_weight_entries(mod: nn.Module) -> List[Tuple[str, torch.Tensor, bool]]:
    entries: List[Tuple[str, torch.Tensor, bool]] = []
    seen: set[int] = set()

    params: Dict[str, torch.Tensor] = {}
    try:
        for n, p in mod.named_parameters(recurse=False):
            if isinstance(p, torch.Tensor):
                params[n] = p
    except Exception:
        pass

    buffers: Dict[str, torch.Tensor] = {}
    try:
        for n, b in mod.named_buffers(recurse=False):
            if isinstance(b, torch.Tensor):
                buffers[n] = b
    except Exception:
        pass

    def has_bias_for_weight_name(wname: str) -> bool:
        cand = []
        cand.append(wname.replace("weight", "bias"))
        cand.append(wname.replace("_weight", "_bias"))
        cand.append(wname.replace(".weight", ".bias"))
        for bn in cand:
            b = params.get(bn) or buffers.get(bn) or _get_tensor_attr(mod, bn)
            if isinstance(b, torch.Tensor):
                return True
        return isinstance(getattr(mod, "bias", None), torch.Tensor)

    def add(name: str, t: Any, has_bias: bool = False) -> None:
        if not (isinstance(t, torch.Tensor) and t.ndim == 2):
            return
        tid = id(t)
        if tid in seen:
            return
        seen.add(tid)
        entries.append((name, t, bool(has_bias)))

    # 1) named_parameters recurse=False
    for n, p in params.items():
        add(f"param:{n}", p, has_bias=has_bias_for_weight_name(n))

    # 2) attribute hints
    candidates = [
        "weight",
        "linear_weight",
        "fc1_weight",
        "fc2_weight",
        "w1",
        "w2",
        "weight1",
        "weight2",
        "in_proj_weight",
        "out_proj_weight",
        "qkv_weight",
        "proj_weight",
        "dense_h_to_4h_weight",
        "dense_4h_to_h_weight",
    ]
    for n in candidates:
        t = _get_tensor_attr(mod, n)
        add(f"attr:{n}", t, has_bias=has_bias_for_weight_name(n))

    # 3) scan immediate children (recurse=True, capped)
    try:
        for name, sm in mod.named_modules():
            if name == "":
                continue
            w = getattr(sm, "weight", None)
            b = getattr(sm, "bias", None)
            add(f"sub:{name}.weight", w, has_bias=isinstance(b, torch.Tensor))
            if len(entries) >= 16:
                break
    except Exception:
        pass

    # 4) fallback: vars(mod) and dir(mod) lightly
    try:
        d = vars(mod)
        for n, v in d.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                add(f"var:{n}", v, has_bias=has_bias_for_weight_name(n))
            if len(entries) >= 16:
                break
    except Exception:
        pass

    try:
        for n in list(dir(mod))[:256]:
            if n.startswith("_"):
                continue
            t = _get_tensor_attr(mod, n)
            if isinstance(t, torch.Tensor) and t.ndim == 2:
                add(f"dir:{n}", t, has_bias=has_bias_for_weight_name(n))
            if len(entries) >= 16:
                break
    except Exception:
        pass

    return entries


def _te_cached_weight_entries(mod: nn.Module) -> List[Tuple[str, torch.Tensor, bool]]:
    try:
        cache = getattr(mod, "_stnet_te_weight_cache", None)
        if isinstance(cache, dict) and cache.get("v") == 1 and isinstance(cache.get("entries"), list):
            return cache["entries"]  # type: ignore[return-value]
    except Exception:
        cache = None

    entries = _find_2d_weight_entries(mod)
    try:
        setattr(mod, "_stnet_te_weight_cache", {"v": 1, "entries": entries})
    except Exception:
        pass
    return entries


def _te_ln_affine(mod: nn.Module) -> Tuple[bool, bool]:
    # (elementwise_affine, has_bias)
    has_weight = False
    has_bias = False
    for n in ("weight", "ln_weight", "layer_norm_weight", "gamma"):
        if isinstance(_get_tensor_attr(mod, n), torch.Tensor):
            has_weight = True
            break
    for n in ("bias", "ln_bias", "layer_norm_bias", "beta"):
        if isinstance(_get_tensor_attr(mod, n), torch.Tensor):
            has_bias = True
            break
    return (has_weight or has_bias), has_bias


def _register_te_any(mod: nn.Module, inp: Tuple[Any, ...], out: Any, *, profiler: "_FlopProfiler", cfg: _HookConfig) -> None:
    x = inp[0] if inp else None
    cname = type(mod).__name__.lower()

    # Attention-ish
    if "attention" in cname:
        ts = _first_tensors_from_args(inp, max_n=3)
        q = ts[0] if len(ts) >= 1 else None
        k = ts[1] if len(ts) >= 2 else None
        v = ts[2] if len(ts) >= 3 else None
        if isinstance(q, torch.Tensor) and q.ndim == 4:
            dropout_p = _safe_float(getattr(mod, "dropout", 0.0), 0.0)
            training = bool(getattr(mod, "training", False))
            val = _flops_attention_from_qkv(
                q, k if isinstance(k, torch.Tensor) else None, v if isinstance(v, torch.Tensor) else None,
                effective_bwd=cfg.effective_bwd,
                dropout_p=dropout_p,
                training=training,
                include_softmax_scale_dropout=True,
            )
            if val > 0.0:
                profiler.add(f"TE.{type(mod).__name__}", val)
        return

    if not isinstance(x, torch.Tensor):
        return

    w_entries = _te_cached_weight_entries(mod)

    # LayerNormLinear: LN + Linear
    if "layernormlinear" in cname or ("layernorm" in cname and "linear" in cname):
        elementwise_affine, ln_has_bias = _te_ln_affine(mod)
        ln = _flops_layernorm(
            x, x,
            normalized_shape=[int(x.shape[-1])],
            elementwise_affine=elementwise_affine,
            has_bias=ln_has_bias,
            effective_bwd=cfg.effective_bwd,
        )

        in_feat = int(x.shape[-1])
        hinted = [(n, w, hb) for (n, w, hb) in w_entries if ("linear" in n.lower() or "proj" in n.lower() or "fc" in n.lower())]
        candidates = hinted if hinted else list(w_entries)
        matches = [(n, w, hb) for (n, w, hb) in candidates if int(w.shape[-1]) == in_feat]
        pick_from = matches if matches else candidates
        if not pick_from:
            return
        pick_from.sort(key=lambda t: int(t[1].numel()), reverse=True)
        _, lin_w, lin_has_bias = pick_from[0]

        lin = _flops_linear(
            x, out, lin_w,
            include_bias=cfg.include_bias,
            has_bias=lin_has_bias,
            effective_bwd=cfg.effective_bwd,
        )
        total = float(ln + lin)
        if total > 0.0:
            profiler.add(f"TE.{type(mod).__name__}", total)
        return

    # LayerNormMLP: LN + (Linear1 + Act + Linear2)
    if "layernormmlp" in cname or ("mlp" in cname and "layernorm" in cname):
        elementwise_affine, ln_has_bias = _te_ln_affine(mod)
        ln = _flops_layernorm(
            x, x,
            normalized_shape=[int(x.shape[-1])],
            elementwise_affine=elementwise_affine,
            has_bias=ln_has_bias,
            effective_bwd=cfg.effective_bwd,
        )

        if not w_entries:
            return

        in_feat = int(x.shape[-1])

        def hint_score(name: str, for_w2: bool) -> int:
            s = name.lower()
            if for_w2:
                if "fc2" in s or "w2" in s or "linear2" in s or "dense_4h_to_h" in s:
                    return 3
                if "out" in s or "proj" in s:
                    return 2
            else:
                if "fc1" in s or "w1" in s or "linear1" in s or "dense_h_to_4h" in s:
                    return 3
                if "in" in s or "qkv" in s:
                    return 2
            return 1

        cand1 = sorted(w_entries, key=lambda t: (hint_score(t[0], False), int(t[1].numel())), reverse=True)
        match1 = [t for t in cand1 if int(t[1].shape[-1]) == in_feat]
        n1, w1, b1 = (match1[0] if match1 else cand1[0])
        hidden = int(w1.shape[0])

        rest = [t for t in w_entries if id(t[1]) != id(w1)]
        cand2 = sorted(rest, key=lambda t: (hint_score(t[0], True), int(t[1].numel())), reverse=True)
        match2 = [t for t in cand2 if int(t[1].shape[-1]) == hidden]
        w2_entry = (match2[0] if match2 else (cand2[0] if cand2 else None))
        w2 = w2_entry[1] if w2_entry is not None else None
        b2 = bool(w2_entry[2]) if w2_entry is not None else False

        # linear1
        lin1 = _flops_linear(
            x, out, w1,
            include_bias=cfg.include_bias,
            has_bias=b1,
            effective_bwd=cfg.effective_bwd,
        )

        # activation: assume GELU-ish on [m, hidden]
        m_dim = int(x.numel() // max(int(w1.shape[-1]), 1))
        act_out_elems = int(m_dim * max(hidden, 0))
        act = float(act_out_elems) * 8.0 * (1.0 + max(0.0, cfg.effective_bwd)) if act_out_elems > 0 else 0.0

        # linear2 direct estimate
        lin2 = 0.0
        if isinstance(w2, torch.Tensor) and w2.ndim == 2 and hidden > 0 and m_dim > 0:
            n2 = int(w2.shape[0])
            k2 = int(w2.shape[-1])
            if k2 > 0:
                fwd2 = 2.0 * float(m_dim) * float(k2) * float(n2)
                if cfg.include_bias and b2:
                    fwd2 += float(m_dim) * float(n2)
                lin2 = float(fwd2 * (1.0 + max(0.0, cfg.effective_bwd)))

        total = float(ln + lin1 + act + lin2)
        if total > 0.0:
            profiler.add(f"TE.{type(mod).__name__}", total)
        return

    # Fallback: linear-like
    if w_entries:
        in_feat = int(x.shape[-1])
        matches = [t for t in w_entries if int(t[1].shape[-1]) == in_feat]
        pick = matches if matches else list(w_entries)
        pick.sort(key=lambda t: int(t[1].numel()), reverse=True)
        _, w, hb = pick[0]
        val = _flops_linear(
            x, out, w,
            include_bias=cfg.include_bias,
            has_bias=hb,
            effective_bwd=cfg.effective_bwd,
        )
        if val > 0.0:
            profiler.add(f"TE.{type(mod).__name__}", val)


# ==============================================================================
# Operator-level (dispatch) mode
# ==============================================================================

try:
    from torch.utils._python_dispatch import TorchDispatchMode
except Exception:  # pragma: no cover
    TorchDispatchMode = None  # type: ignore


def _op_name(func: Any) -> str:
    try:
        return str(func).replace("torch.ops.aten.", "aten.")
    except Exception:
        return "op"


class _OpFlopDispatchMode(TorchDispatchMode):  # type: ignore[misc]

    def __init__(
        self,
        profiler: "_FlopProfiler",
        *,
        include_bias: bool,
        effective_bwd: float,
        count_elementwise: bool,
    ) -> None:
        super().__init__()
        self._profiler = profiler
        self._include_bias = bool(include_bias)
        self._effective_bwd = float(effective_bwd)
        self._count_elementwise = bool(count_elementwise)

        self._aten = torch.ops.aten  # shorthand

        handlers: Dict[Any, Callable[[Tuple[Any, ...], Dict[str, Any], Any], float]] = {}

        # Map specific ops to aggregated breakdown tags (e.g., Elementwise.exp)
        self._tag_overrides: Dict[Any, str] = {}

        def _maybe(op: Any) -> Any:
            return op if op is not None else None

        # GEMM-like
        handlers[_maybe(self._aten.mm.default)] = self._h_mm
        handlers[_maybe(self._aten.bmm.default)] = self._h_bmm
        handlers[_maybe(self._aten.matmul.default)] = self._h_matmul
        handlers[_maybe(self._aten.addmm.default)] = self._h_addmm
        if getattr(self._aten, "linear", None) is not None:
            handlers[_maybe(self._aten.linear.default)] = self._h_linear
        handlers[_maybe(self._aten.convolution.default)] = self._h_convolution

        # Norms
        handlers[_maybe(self._aten.native_layer_norm.default)] = self._h_native_layer_norm
        handlers[_maybe(self._aten.layer_norm.default)] = self._h_layer_norm

        # Dropout / Softmax
        if getattr(self._aten, "dropout", None) is not None:
            handlers[_maybe(self._aten.dropout.default)] = self._h_dropout

        if getattr(self._aten, "_softmax", None) is not None:
            handlers[_maybe(self._aten._softmax.default)] = self._h_softmax
        if getattr(self._aten, "softmax", None) is not None and hasattr(self._aten.softmax, "int"):
            handlers[_maybe(self._aten.softmax.int)] = self._h_softmax

        # SDPA + attention variants (flash / efficient / cudnn)
        if getattr(self._aten, "scaled_dot_product_attention", None) is not None:
            handlers[_maybe(self._aten.scaled_dot_product_attention.default)] = self._h_sdpa
        for name in (
            "_scaled_dot_product_flash_attention",
            "_scaled_dot_product_flash_attention_for_cpu",
            "_scaled_dot_product_efficient_attention",
            "_scaled_dot_product_cudnn_attention",
            "_flash_attention_forward",
            "_efficient_attention_forward",
        ):
            op = getattr(self._aten, name, None)
            if op is not None and hasattr(op, "default"):
                handlers[_maybe(op.default)] = self._h_sdpa_like

        # Embedding (FLOPs ~ 0)
        if getattr(self._aten, "embedding", None) is not None:
            handlers[_maybe(self._aten.embedding.default)] = lambda a, k, o: 0.0

        # Elementwise ops
        if self._count_elementwise:
            # binary ops: 1 flop/elem
            for base in ("add", "sub", "mul", "div"):
                obj = getattr(self._aten, base, None)
                obj_ = getattr(self._aten, base + "_", None)
                if obj is not None:
                    for overload in ("Tensor", "Scalar", "default", "self", "ScalarSelf", "ScalarOther", "Tensor_out", "Scalar_out", "self_out"):
                        f = getattr(obj, overload, None)
                        if f is not None:
                            op = _maybe(f)
                            handlers[op] = self._h_binop
                            if op is not None:
                                self._tag_overrides[op] = f"Elementwise.{base}"
                if obj_ is not None:
                    for overload in ("Tensor", "Scalar", "default"):
                        f = getattr(obj_, overload, None)
                        if f is not None:
                            op = _maybe(f)
                            handlers[op] = self._h_binop
                            if op is not None:
                                self._tag_overrides[op] = f"Elementwise.{base}"

            # ternary fused patterns (fma-like): ~3 flops/elem
            for base, handler in (("addcmul", self._h_addcmul), ("addcdiv", self._h_addcdiv)):
                obj = getattr(self._aten, base, None)
                obj_ = getattr(self._aten, base + "_", None)
                if obj is not None:
                    for overload in ("Tensor", "Scalar", "default"):
                        f = getattr(obj, overload, None)
                        if f is not None:
                            op = _maybe(f)
                            handlers[op] = handler
                            if op is not None:
                                self._tag_overrides[op] = f"Elementwise.{base}"
                if obj_ is not None:
                    for overload in ("Tensor", "Scalar", "default"):
                        f = getattr(obj_, overload, None)
                        if f is not None:
                            op = _maybe(f)
                            handlers[op] = handler
                            if op is not None:
                                self._tag_overrides[op] = f"Elementwise.{base}"

            # unary coefficient table (rough relative cost)
            unary_coeff = {
                # cheap-ish
                "abs": 1.0,
                "neg": 1.0,
                # cheap rounding/sign
                "floor": 1.0,
                "ceil": 1.0,
                "round": 1.0,
                "trunc": 1.0,
                "frac": 1.0,
                "sign": 1.0,
                "reciprocal": 4.0,
                # roots
                "sqrt": 4.0,
                "rsqrt": 6.0,
                # exp/log family
                "exp": 10.0,
                "exp2": 10.0,
                "expm1": 10.0,
                "log": 10.0,
                "log2": 10.0,
                "log10": 10.0,
                "log1p": 10.0,
                # trig/hyperbolic
                "sin": 12.0,
                "cos": 12.0,
                "tan": 12.0,
                "asin": 14.0,
                "acos": 14.0,
                "atan": 14.0,
                "sinh": 14.0,
                "cosh": 14.0,
                "tanh": 4.0,
                "asinh": 14.0,
                "acosh": 14.0,
                "atanh": 14.0,
                # special
                "erf": 14.0,
                "erfc": 14.0,
                "erfinv": 18.0,
                "sigmoid": 4.0,
                # common fused activations
                "relu": 1.0,
                "gelu": 8.0,
                "silu": 6.0,
            }
            for name, coeff in unary_coeff.items():
                obj = getattr(self._aten, name, None)
                obj_ = getattr(self._aten, name + "_", None)
                if obj is not None and hasattr(obj, "default"):
                    op = _maybe(obj.default)
                    handlers[op] = (lambda a, k, o, c=coeff: _flops_elementwise(o, coeff=c, effective_bwd=self._effective_bwd))
                    if op is not None:
                        self._tag_overrides[op] = f"Elementwise.{name}"
                if obj_ is not None and hasattr(obj_, "default"):
                    op = _maybe(obj_.default)
                    handlers[op] = (lambda a, k, o, c=coeff: _flops_elementwise(o, coeff=c, effective_bwd=self._effective_bwd))
                    if op is not None:
                        self._tag_overrides[op] = f"Elementwise.{name}"

            
            # binary special elementwise ops (coeff table)
            binary_coeff = {
                "atan2": 14.0,
                "minimum": 1.0,
                "maximum": 1.0,
                "fmin": 1.0,
                "fmax": 1.0,
                "clamp": 2.0,
                "clamp_min": 1.0,
                "clamp_max": 1.0,
                "where": 1.0,
                "lerp": 3.0,
            }
            for name, coeff in binary_coeff.items():
                obj = getattr(self._aten, name, None)
                obj_ = getattr(self._aten, name + "_", None)
                for pobj in (obj, obj_):
                    if pobj is None:
                        continue
                    for overload in ("Tensor", "Scalar", "default", "self", "ScalarSelf", "ScalarOther", "Tensor_out", "Scalar_out", "self_out"):
                        f = getattr(pobj, overload, None)
                        if f is not None:
                            op = _maybe(f)
                            handlers[op] = (lambda a, k, o, c=coeff: _flops_elementwise(o, coeff=c, effective_bwd=self._effective_bwd))
                            if op is not None:
                                self._tag_overrides[op] = f"Elementwise.{name}"

# pow: heuristic based on exponent
            pow_obj = getattr(self._aten, "pow", None)
            pow_obj_ = getattr(self._aten, "pow_", None)
            for pobj in (pow_obj, pow_obj_):
                if pobj is None:
                    continue
                for overload in ("Tensor_Tensor", "Tensor_Scalar", "Scalar", "default"):
                    f = getattr(pobj, overload, None)
                    if f is not None:
                        op = _maybe(f)
                        handlers[op] = self._h_pow
                        if op is not None:
                            self._tag_overrides[op] = "Elementwise.pow"

            # fma: fused multiply-add (if present) ~2 flops/elem
            fma_obj = getattr(self._aten, "fma", None)
            if fma_obj is not None and hasattr(fma_obj, "default"):
                op = _maybe(fma_obj.default)
                handlers[op] = self._h_fma
                if op is not None:
                    self._tag_overrides[op] = "Elementwise.fma"

        self._handlers = {k: v for k, v in handlers.items() if k is not None}

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        out = func(*args, **kwargs)

        handler = self._handlers.get(func, None)
        if handler is None:
            # Best-effort for Triton / Flex / TE custom ops.
            name = _op_name(func).lower()
            val = 0.0
            if ("triton" in name) or ("flex" in name) or ("transformer_engine" in name) or ("xformers" in name):
                val = self._h_custom(args, kwargs, out, name=name)
                if val > 0.0:
                    tag = (
                        "Triton"
                        if "triton" in name
                        else ("FlexAttention" if "flex" in name else ("TE" if "transformer_engine" in name else "Custom"))
                    )
                    self._profiler.add(tag, val)
            return out

        try:
            val = float(handler(args, kwargs, out))
        except Exception:
            val = 0.0

        if val > 0.0:
            tag = self._tag_overrides.get(func)
            self._profiler.add(tag if tag is not None else _op_name(func), val)

        return out

    # ---- handlers

    def _h_binop(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        return _flops_elementwise(out, coeff=1.0, effective_bwd=self._effective_bwd)

    def _h_addcmul(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        # out = input + value * t1 * t2  (mul + mul + add)
        return _flops_elementwise(out, coeff=3.0, effective_bwd=self._effective_bwd)

    def _h_addcdiv(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        # out = input + value * t1 / t2  (div + mul + add)
        return _flops_elementwise(out, coeff=3.0, effective_bwd=self._effective_bwd)

    def _h_pow(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        exp = args[1] if len(args) >= 2 else kwargs.get("exponent", None)
        coeff = 12.0
        try:
            if isinstance(exp, (int, float)):
                e = float(exp)
                if e == 0.5:
                    coeff = 4.0
                elif e == -0.5:
                    coeff = 6.0
                elif e.is_integer():
                    ei = int(e)
                    if 2 <= ei <= 4:
                        coeff = float(ei - 1)
                    elif ei in (0, 1):
                        coeff = 1.0
        except Exception:
            coeff = 12.0
        return _flops_elementwise(out, coeff=float(coeff), effective_bwd=self._effective_bwd)

    def _h_fma(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        # fma(a, b, c) ~= a*b + c
        return _flops_elementwise(out, coeff=2.0, effective_bwd=self._effective_bwd)

    def _h_mm(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        a, b = args[0], args[1]
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return 0.0
        if a.ndim != 2 or b.ndim != 2:
            return 0.0
        m, k = a.shape
        k2, n = b.shape
        if int(k) != int(k2):
            return 0.0
        fwd = 2.0 * int(m) * int(n) * int(k)
        return float(fwd * (1.0 + max(0.0, self._effective_bwd)))

    def _h_bmm(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        a, b = args[0], args[1]
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return 0.0
        if a.ndim != 3 or b.ndim != 3:
            return 0.0
        batch, m, k = a.shape
        batch2, k2, n = b.shape
        if int(batch) != int(batch2) or int(k) != int(k2):
            return 0.0
        fwd = 2.0 * int(batch) * int(m) * int(n) * int(k)
        return float(fwd * (1.0 + max(0.0, self._effective_bwd)))

    def _h_matmul(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        a, b = args[0], args[1]
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return 0.0
        if a.ndim == 2 and b.ndim == 2:
            return self._h_mm(args, kwargs, out)
        if a.ndim == 3 and b.ndim == 3:
            return self._h_bmm(args, kwargs, out)
        return 0.0

    def _h_addmm(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        if len(args) < 3:
            return 0.0
        input_, mat1, mat2 = args[0], args[1], args[2]
        if not (isinstance(mat1, torch.Tensor) and isinstance(mat2, torch.Tensor)):
            return 0.0
        if mat1.ndim != 2 or mat2.ndim != 2:
            return 0.0
        m, k = mat1.shape
        k2, n = mat2.shape
        if int(k) != int(k2):
            return 0.0
        fwd = 2.0 * int(m) * int(n) * int(k)
        if self._include_bias and isinstance(input_, torch.Tensor) and input_.numel() > 0:
            fwd += float(int(m) * int(n))
        return float(fwd * (1.0 + max(0.0, self._effective_bwd)))

    def _h_linear(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        if len(args) < 2:
            return 0.0
        x, w = args[0], args[1]
        b = args[2] if len(args) >= 3 else None
        if not (isinstance(x, torch.Tensor) and isinstance(w, torch.Tensor)):
            return 0.0
        has_bias = isinstance(b, torch.Tensor)
        return _flops_linear(
            x, out, w,
            include_bias=self._include_bias,
            has_bias=has_bias,
            effective_bwd=self._effective_bwd,
        )

    def _h_convolution(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        if len(args) < 3:
            return 0.0
        x, w, b = args[0], args[1], args[2]
        groups = args[8] if len(args) >= 9 else kwargs.get("groups", 1)
        has_bias = isinstance(b, torch.Tensor)
        return _flops_conv(
            x, out, w,
            groups=int(groups),
            include_bias=self._include_bias,
            has_bias=has_bias,
            effective_bwd=self._effective_bwd,
        )

    def _h_native_layer_norm(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        if len(args) < 2:
            return 0.0
        x = args[0]
        normalized_shape = args[1]
        w = args[2] if len(args) >= 3 else None
        b = args[3] if len(args) >= 4 else None
        y = out[0] if isinstance(out, (tuple, list)) and out else out
        return _flops_layernorm(
            x, y,
            normalized_shape=list(normalized_shape) if isinstance(normalized_shape, (tuple, list)) else (),
            elementwise_affine=isinstance(w, torch.Tensor),
            has_bias=isinstance(b, torch.Tensor),
            effective_bwd=self._effective_bwd,
        )

    def _h_layer_norm(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        if len(args) < 2:
            return 0.0
        x = args[0]
        normalized_shape = args[1]
        w = args[2] if len(args) >= 3 else None
        b = args[3] if len(args) >= 4 else None
        return _flops_layernorm(
            x, out,
            normalized_shape=list(normalized_shape) if isinstance(normalized_shape, (tuple, list)) else (),
            elementwise_affine=isinstance(w, torch.Tensor),
            has_bias=isinstance(b, torch.Tensor),
            effective_bwd=self._effective_bwd,
        )

    def _h_dropout(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        if len(args) < 3:
            return 0.0
        train = bool(args[2])
        if not train:
            return 0.0
        return _flops_elementwise(out, coeff=2.0, effective_bwd=self._effective_bwd)

    def _h_softmax(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        if len(args) < 2:
            return 0.0
        x = args[0]
        dim = int(args[1])
        if not isinstance(x, torch.Tensor):
            return 0.0
        return _flops_softmax(x, out, dim=dim, effective_bwd=self._effective_bwd)

    def _h_sdpa(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        if len(args) < 3:
            return 0.0
        q = args[0]
        k = args[1]
        v = args[2]
        dropout_p = _safe_float(args[4], 0.0) if len(args) >= 5 else _safe_float(kwargs.get("dropout_p", 0.0), 0.0)
        training = True if dropout_p > 0.0 else False
        if not isinstance(q, torch.Tensor):
            return 0.0
        return _flops_attention_from_qkv(
            q,
            k if isinstance(k, torch.Tensor) else None,
            v if isinstance(v, torch.Tensor) else None,
            effective_bwd=self._effective_bwd,
            dropout_p=float(dropout_p),
            training=training,
            include_softmax_scale_dropout=True,
        )

    def _h_sdpa_like(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        ts = _first_tensors_from_args(args, max_n=3)
        if not ts:
            return 0.0
        q = ts[0]
        k = ts[1] if len(ts) >= 2 else None
        v = ts[2] if len(ts) >= 3 else None
        dropout_p = _safe_float(kwargs.get("dropout_p", 0.0), 0.0)
        training = True if dropout_p > 0.0 else False
        return _flops_attention_from_qkv(
            q, k, v,
            effective_bwd=self._effective_bwd,
            dropout_p=float(dropout_p),
            training=training,
            include_softmax_scale_dropout=True,
        )

    def _h_custom(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any, *, name: str) -> float:
        # Heuristic: attention if any 4D tensor, else GEMM if 2D+2D, else elementwise-ish.
        ts = _first_tensors_from_args(args, max_n=4)
        for t in ts:
            if isinstance(t, torch.Tensor) and t.ndim == 4:
                q = t
                k = ts[1] if len(ts) >= 2 else None
                v = ts[2] if len(ts) >= 3 else None
                return _flops_attention_from_qkv(
                    q, k, v,
                    effective_bwd=self._effective_bwd,
                    dropout_p=0.0,
                    training=False,
                    include_softmax_scale_dropout=True,
                )

        if len(ts) >= 2 and ts[0].ndim == 2 and ts[1].ndim == 2:
            a, b = ts[0], ts[1]
            m, k = a.shape
            k2, n = b.shape
            if int(k) == int(k2):
                fwd = 2.0 * int(m) * int(n) * int(k)
                return float(fwd * (1.0 + max(0.0, self._effective_bwd)))

        return _flops_elementwise(out, coeff=1.0, effective_bwd=self._effective_bwd)


# ==============================================================================
# FX / Dynamo export estimator (compile-safe, CUDA graphs safe)
# ==============================================================================

class _ShapeTensor:

    def __init__(self, shape: Tuple[int, ...]) -> None:
        self.shape = tuple(int(x) for x in shape)
        self.ndim = len(self.shape)

    def numel(self) -> int:
        n = 1
        for v in self.shape:
            n *= int(v)
        return int(n)


def _is_tensorlike(x: Any) -> bool:
    return isinstance(x, (torch.Tensor, _ShapeTensor))


def _shape_from_meta(meta: Any) -> Optional[Tuple[int, ...]]:
    try:
        if meta is None:
            return None
        shp = getattr(meta, "shape", None)
        if shp is None:
            return None
        return tuple(int(s) for s in shp)
    except Exception:
        return None


class _FxGraphFlopEstimator:

    def __init__(self, *, include_bias: bool, effective_bwd: float, count_elementwise: bool) -> None:
        self._include_bias = bool(include_bias)
        self._effective_bwd = float(effective_bwd)
        self._count_elementwise = bool(count_elementwise)
        self._aten = torch.ops.aten

        # binary ops
        def _ops_for(base: str) -> List[Any]:
            ops: List[Any] = []
            obj = getattr(self._aten, base, None)
            obj_ = getattr(self._aten, base + "_", None)
            for pobj in (obj, obj_):
                if pobj is None:
                    continue
                for o in ("Tensor", "Scalar", "default", "self", "ScalarSelf", "ScalarOther", "Tensor_out", "Scalar_out", "self_out"):
                    f = getattr(pobj, o, None)
                    if f is not None:
                        ops.append(f)
            return ops

        self._add_ops = _ops_for("add")
        self._sub_ops = _ops_for("sub")
        self._mul_ops = _ops_for("mul")
        self._div_ops = _ops_for("div")
        self._atan2_ops = _ops_for("atan2")
        self._minimum_ops = _ops_for("minimum")
        self._maximum_ops = _ops_for("maximum")
        self._fmin_ops = _ops_for("fmin")
        self._fmax_ops = _ops_for("fmax")
        self._clamp_ops = _ops_for("clamp")
        self._clamp_min_ops = _ops_for("clamp_min")
        self._clamp_max_ops = _ops_for("clamp_max")
        self._where_ops = _ops_for("where")
        self._lerp_ops = _ops_for("lerp")

        # addcmul/addcdiv
        self._addcmul_ops: List[Any] = []
        self._addcdiv_ops: List[Any] = []
        for base, slot in (("addcmul", self._addcmul_ops), ("addcdiv", self._addcdiv_ops)):
            obj = getattr(self._aten, base, None)
            obj_ = getattr(self._aten, base + "_", None)
            for pobj in (obj, obj_):
                if pobj is None:
                    continue
                for o in ("Tensor", "Scalar", "default"):
                    f = getattr(pobj, o, None)
                    if f is not None:
                        slot.append(f)

        # unary op table
        self._unary_ops: Dict[Any, Tuple[str, float]] = {}
        unary_coeff = {
            "abs": 1.0,
            "neg": 1.0,
            "floor": 1.0,
            "ceil": 1.0,
            "round": 1.0,
            "trunc": 1.0,
            "frac": 1.0,
            "sign": 1.0,
            "reciprocal": 4.0,
            "sqrt": 4.0,
            "rsqrt": 6.0,
            "exp": 10.0,
            "exp2": 10.0,
            "expm1": 10.0,
            "log": 10.0,
            "log2": 10.0,
            "log10": 10.0,
            "log1p": 10.0,
            "sin": 12.0,
            "cos": 12.0,
            "tan": 12.0,
            "asin": 14.0,
            "acos": 14.0,
            "atan": 14.0,
            "sinh": 14.0,
            "cosh": 14.0,
            "tanh": 4.0,
            "asinh": 14.0,
            "acosh": 14.0,
            "atanh": 14.0,
            "erf": 14.0,
            "erfc": 14.0,
            "erfinv": 18.0,
            "sigmoid": 4.0,
            "relu": 1.0,
            "gelu": 8.0,
            "silu": 6.0,
        }
        for name, coeff in unary_coeff.items():
            obj = getattr(self._aten, name, None)
            obj_ = getattr(self._aten, name + "_", None)
            if obj is not None and hasattr(obj, "default"):
                self._unary_ops[obj.default] = (name, float(coeff))
            if obj_ is not None and hasattr(obj_, "default"):
                self._unary_ops[obj_.default] = (name, float(coeff))

        # pow ops
        self._pow_ops: List[Any] = []
        for pobj in (getattr(self._aten, "pow", None), getattr(self._aten, "pow_", None)):
            if pobj is None:
                continue
            for o in ("Tensor_Tensor", "Tensor_Scalar", "Scalar", "default"):
                f = getattr(pobj, o, None)
                if f is not None:
                    self._pow_ops.append(f)

        # fma ops
        self._fma_ops: List[Any] = []
        fma_obj = getattr(self._aten, "fma", None)
        if fma_obj is not None and hasattr(fma_obj, "default"):
            self._fma_ops.append(fma_obj.default)

        # SDPA variants
        self._sdpa_like_ops: List[Any] = []
        for name in (
            "scaled_dot_product_attention",
            "_scaled_dot_product_flash_attention",
            "_scaled_dot_product_flash_attention_for_cpu",
            "_scaled_dot_product_efficient_attention",
            "_scaled_dot_product_cudnn_attention",
            "_flash_attention_forward",
            "_efficient_attention_forward",
        ):
            op = getattr(self._aten, name, None)
            if op is not None and hasattr(op, "default"):
                self._sdpa_like_ops.append(op.default)

    def estimate(self, gm: torch.fx.GraphModule) -> Tuple[float, Dict[str, float]]:
        def resolve(obj: Any) -> Any:
            if isinstance(obj, torch.fx.Node):
                return resolve_node(obj)
            if isinstance(obj, (tuple, list)):
                return type(obj)(resolve(x) for x in obj)
            if isinstance(obj, dict):
                return {k: resolve(v) for k, v in obj.items()}
            return obj

        def resolve_node(n: torch.fx.Node) -> Any:
            v = n.meta.get("val", None)
            if v is not None:
                return v
            tm = n.meta.get("tensor_meta", None)
            shp = _shape_from_meta(tm)
            if shp is not None:
                return _ShapeTensor(shp)
            if n.op == "get_attr":
                try:
                    return getattr(gm, str(n.target))
                except Exception:
                    return None
            return None

        total = 0.0
        by: Dict[str, float] = {}

        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue

            target = node.target
            args = resolve(node.args)
            kwargs = resolve(node.kwargs or {})
            out = resolve_node(node)

            flops, typ = self._estimate_call(target, args, kwargs, out)
            if flops <= 0.0:
                continue
            total += float(flops)
            by[typ] = by.get(typ, 0.0) + float(flops)

        return float(total), by

    def _as_tensor(self, out: Any) -> Any:
        if isinstance(out, (tuple, list)) and out:
            return out[0]
        return out

    def _out_numel(self, out: Any) -> int:
        y = self._as_tensor(out)
        if y is None:
            return 0
        try:
            numel = getattr(y, "numel", None)
            return int(numel()) if callable(numel) else int(numel)
        except Exception:
            return 0

    def _eltwise(self, out: Any, coeff: float) -> float:
        n = self._out_numel(out)
        if n <= 0:
            return 0.0
        fwd = float(n) * float(coeff)
        return float(fwd * (1.0 + max(0.0, self._effective_bwd)))

    def _pow_coeff(self, exp: Any) -> float:
        try:
            if isinstance(exp, (int, float)):
                e = float(exp)
                if e == 0.5:
                    return 4.0
                if e == -0.5:
                    return 6.0
                if e.is_integer():
                    ei = int(e)
                    if 2 <= ei <= 4:
                        return float(ei - 1)
                    if ei in (0, 1):
                        return 1.0
        except Exception:
            pass
        return 12.0

    def _estimate_call(self, target: Any, args: Any, kwargs: Dict[str, Any], out: Any) -> Tuple[float, str]:
        try:
            if target == self._aten.mm.default:
                return self._mm(args), "MatMul"
            if target == self._aten.bmm.default:
                return self._bmm(args), "MatMul"
            if target == self._aten.matmul.default:
                return self._matmul(args), "MatMul"
            if target == self._aten.addmm.default:
                return self._addmm(args), "Linear"
            if getattr(self._aten, "linear", None) is not None and target == self._aten.linear.default:
                return self._linear(args, out), "Linear"
            if target == self._aten.convolution.default:
                return self._convolution(args, out), "Conv"
            if target == self._aten.native_layer_norm.default:
                return self._native_layer_norm(args, out), "LayerNorm"
            if target == self._aten.layer_norm.default:
                return self._layer_norm(args, out), "LayerNorm"
            if getattr(self._aten, "dropout", None) is not None and target == self._aten.dropout.default:
                return self._dropout(args, out), "Dropout"
            if getattr(self._aten, "_softmax", None) is not None and target == self._aten._softmax.default:
                return self._softmax(args, out), "Softmax"
            if getattr(self._aten, "softmax", None) is not None and hasattr(self._aten.softmax, "int") and target == self._aten.softmax.int:
                return self._softmax(args, out), "Softmax"
            if target in self._sdpa_like_ops:
                return self._sdpa_like(args, kwargs), "Attention"

            if self._count_elementwise and target in self._unary_ops:
                name, coeff = self._unary_ops[target]
                return (self._eltwise(out, float(coeff)), f"Elementwise.{name}")
            if self._count_elementwise and target in self._pow_ops:
                exp = args[1] if isinstance(args, (tuple, list)) and len(args) >= 2 else kwargs.get("exponent", None)
                return (self._eltwise(out, self._pow_coeff(exp)), "Elementwise.pow")
            if self._count_elementwise and target in self._fma_ops:
                return (self._eltwise(out, 2.0), "Elementwise.fma")
            if self._count_elementwise and target in getattr(self, "_atan2_ops", []):
                return (self._eltwise(out, 14.0), "Elementwise.atan2")
            if self._count_elementwise and target in getattr(self, "_minimum_ops", []):
                return (self._eltwise(out, 1.0), "Elementwise.minimum")
            if self._count_elementwise and target in getattr(self, "_maximum_ops", []):
                return (self._eltwise(out, 1.0), "Elementwise.maximum")
            if self._count_elementwise and target in getattr(self, "_fmin_ops", []):
                return (self._eltwise(out, 1.0), "Elementwise.fmin")
            if self._count_elementwise and target in getattr(self, "_fmax_ops", []):
                return (self._eltwise(out, 1.0), "Elementwise.fmax")
            if self._count_elementwise and target in getattr(self, "_clamp_min_ops", []):
                return (self._eltwise(out, 1.0), "Elementwise.clamp_min")
            if self._count_elementwise and target in getattr(self, "_clamp_max_ops", []):
                return (self._eltwise(out, 1.0), "Elementwise.clamp_max")
            if self._count_elementwise and target in getattr(self, "_clamp_ops", []):
                return (self._eltwise(out, 2.0), "Elementwise.clamp")
            if self._count_elementwise and target in getattr(self, "_where_ops", []):
                return (self._eltwise(out, 1.0), "Elementwise.where")
            if self._count_elementwise and target in getattr(self, "_lerp_ops", []):
                return (self._eltwise(out, 3.0), "Elementwise.lerp")
            if self._count_elementwise and (target in self._add_ops or target in self._sub_ops or target in self._mul_ops or target in self._div_ops):
                if target in self._add_ops:
                    return (self._eltwise(out, 1.0), "Elementwise.add")
                if target in self._sub_ops:
                    return (self._eltwise(out, 1.0), "Elementwise.sub")
                if target in self._mul_ops:
                    return (self._eltwise(out, 1.0), "Elementwise.mul")
                return (self._eltwise(out, 1.0), "Elementwise.div")
            if self._count_elementwise and target in self._addcmul_ops:
                return (self._eltwise(out, 3.0), "Elementwise.addcmul")
            if self._count_elementwise and target in self._addcdiv_ops:
                return (self._eltwise(out, 3.0), "Elementwise.addcdiv")
        except Exception:
            return (0.0, "Unknown")

        name = str(target).lower()
        if ("triton" in name) or ("flex" in name) or ("transformer_engine" in name) or ("xformers" in name):
            return (self._custom(args, out, name=name), "Custom")

        return (0.0, "Other")

    def _mm(self, args: Any) -> float:
        a, b = args[0], args[1]
        if not (_is_tensorlike(a) and _is_tensorlike(b)):
            return 0.0
        if a.ndim != 2 or b.ndim != 2:
            return 0.0
        m, k = a.shape
        k2, n = b.shape
        if int(k) != int(k2):
            return 0.0
        fwd = 2.0 * int(m) * int(n) * int(k)
        return float(fwd * (1.0 + max(0.0, self._effective_bwd)))

    def _bmm(self, args: Any) -> float:
        a, b = args[0], args[1]
        if not (_is_tensorlike(a) and _is_tensorlike(b)):
            return 0.0
        if a.ndim != 3 or b.ndim != 3:
            return 0.0
        batch, m, k = a.shape
        batch2, k2, n = b.shape
        if int(batch) != int(batch2) or int(k) != int(k2):
            return 0.0
        fwd = 2.0 * int(batch) * int(m) * int(n) * int(k)
        return float(fwd * (1.0 + max(0.0, self._effective_bwd)))

    def _matmul(self, args: Any) -> float:
        a, b = args[0], args[1]
        if not (_is_tensorlike(a) and _is_tensorlike(b)):
            return 0.0
        if a.ndim == 2 and b.ndim == 2:
            return self._mm(args)
        if a.ndim == 3 and b.ndim == 3:
            return self._bmm(args)
        return 0.0

    def _addmm(self, args: Any) -> float:
        if len(args) < 3:
            return 0.0
        input_, mat1, mat2 = args[0], args[1], args[2]
        if not (_is_tensorlike(mat1) and _is_tensorlike(mat2)):
            return 0.0
        if mat1.ndim != 2 or mat2.ndim != 2:
            return 0.0
        m, k = mat1.shape
        k2, n = mat2.shape
        if int(k) != int(k2):
            return 0.0
        fwd = 2.0 * int(m) * int(n) * int(k)
        if self._include_bias and _is_tensorlike(input_) and int(getattr(input_, "numel", lambda: 0)()) > 0:
            fwd += float(int(m) * int(n))
        return float(fwd * (1.0 + max(0.0, self._effective_bwd)))

    def _linear(self, args: Any, out: Any) -> float:
        if len(args) < 2:
            return 0.0
        x, w = args[0], args[1]
        b = args[2] if len(args) >= 3 else None
        if not (_is_tensorlike(x) and _is_tensorlike(w)):
            return 0.0
        return _flops_linear(
            x if isinstance(x, torch.Tensor) else _to_real_tensor(x),
            self._as_tensor(out),
            w if isinstance(w, torch.Tensor) else _to_real_tensor(w),
            include_bias=self._include_bias,
            has_bias=_is_tensorlike(b),
            effective_bwd=self._effective_bwd,
        )

    def _convolution(self, args: Any, out: Any) -> float:
        if len(args) < 3:
            return 0.0
        x, w, b = args[0], args[1], args[2]
        groups = args[8] if len(args) >= 9 else 1
        return _flops_conv(
            _to_real_tensor(x),
            self._as_tensor(out),
            _to_real_tensor(w),
            groups=int(groups),
            include_bias=self._include_bias,
            has_bias=_is_tensorlike(b),
            effective_bwd=self._effective_bwd,
        )

    def _native_layer_norm(self, args: Any, out: Any) -> float:
        if len(args) < 2:
            return 0.0
        x = _to_real_tensor(args[0])
        normalized_shape = args[1]
        w = args[2] if len(args) >= 3 else None
        b = args[3] if len(args) >= 4 else None
        y = self._as_tensor(out)
        return _flops_layernorm(
            x, y,
            normalized_shape=list(normalized_shape) if isinstance(normalized_shape, (tuple, list)) else (),
            elementwise_affine=_is_tensorlike(w),
            has_bias=_is_tensorlike(b),
            effective_bwd=self._effective_bwd,
        )

    def _layer_norm(self, args: Any, out: Any) -> float:
        if len(args) < 2:
            return 0.0
        x = _to_real_tensor(args[0])
        normalized_shape = args[1]
        w = args[2] if len(args) >= 3 else None
        b = args[3] if len(args) >= 4 else None
        return _flops_layernorm(
            x, self._as_tensor(out),
            normalized_shape=list(normalized_shape) if isinstance(normalized_shape, (tuple, list)) else (),
            elementwise_affine=_is_tensorlike(w),
            has_bias=_is_tensorlike(b),
            effective_bwd=self._effective_bwd,
        )

    def _dropout(self, args: Any, out: Any) -> float:
        if len(args) < 3:
            return 0.0
        train = bool(args[2])
        if not train:
            return 0.0
        return _flops_elementwise(self._as_tensor(out), coeff=2.0, effective_bwd=self._effective_bwd)

    def _softmax(self, args: Any, out: Any) -> float:
        if len(args) < 2:
            return 0.0
        x = _to_real_tensor(args[0])
        dim = int(args[1])
        return _flops_softmax(x, self._as_tensor(out), dim=dim, effective_bwd=self._effective_bwd)

    def _sdpa_like(self, args: Any, kwargs: Dict[str, Any]) -> float:
        # first 3 args usually q,k,v
        q = args[0] if isinstance(args, (tuple, list)) and len(args) >= 1 else None
        k = args[1] if isinstance(args, (tuple, list)) and len(args) >= 2 else None
        v = args[2] if isinstance(args, (tuple, list)) and len(args) >= 3 else None
        if not _is_tensorlike(q):
            return 0.0
        dropout_p = float(kwargs.get("dropout_p", 0.0))
        training = True if dropout_p > 0.0 else False

        if isinstance(q, _ShapeTensor):
            b, h, s, d = _infer_bhsd_shape(q.shape)
            klen = s
            if isinstance(k, _ShapeTensor) and k.ndim == 4:
                _, _, klen, _ = _infer_bhsd_shape(k.shape)
            return _flops_attention_general(
                batch=b,
                num_heads=h,
                q_len=s,
                k_len=klen,
                head_dim=d,
                effective_bwd=self._effective_bwd,
                dropout_p=dropout_p,
                training=training,
                include_softmax_scale_dropout=True,
            )

        q_t = _to_real_tensor(q)
        k_t = _to_real_tensor(k) if _is_tensorlike(k) else None
        v_t = _to_real_tensor(v) if _is_tensorlike(v) else None
        return _flops_attention_from_qkv(
            q_t, k_t, v_t,
            effective_bwd=self._effective_bwd,
            dropout_p=dropout_p,
            training=training,
            include_softmax_scale_dropout=True,
        )

    def _custom(self, args: Any, out: Any, *, name: str) -> float:
        # attention if any 4D tensorlike, else GEMM if 2D+2D, else elementwise
        if isinstance(args, (tuple, list)):
            ts = [a for a in args if _is_tensorlike(a)]
        else:
            ts = []
        for t in ts:
            if int(getattr(t, "ndim", 0)) == 4:
                q = t
                k = ts[1] if len(ts) >= 2 else None
                v = ts[2] if len(ts) >= 3 else None
                if isinstance(q, _ShapeTensor):
                    b, h, s, d = _infer_bhsd_shape(q.shape)
                    klen = s
                    if isinstance(k, _ShapeTensor) and k.ndim == 4:
                        _, _, klen, _ = _infer_bhsd_shape(k.shape)
                    return _flops_attention_general(
                        batch=b,
                        num_heads=h,
                        q_len=s,
                        k_len=klen,
                        head_dim=d,
                        effective_bwd=self._effective_bwd,
                        dropout_p=0.0,
                        training=False,
                        include_softmax_scale_dropout=True,
                    )
                if isinstance(q, torch.Tensor):
                    return _flops_attention_from_qkv(
                        q,
                        k if isinstance(k, torch.Tensor) else None,
                        v if isinstance(v, torch.Tensor) else None,
                        effective_bwd=self._effective_bwd,
                        dropout_p=0.0,
                        training=False,
                        include_softmax_scale_dropout=True,
                    )

        if len(ts) >= 2 and int(getattr(ts[0], "ndim", 0)) == 2 and int(getattr(ts[1], "ndim", 0)) == 2:
            a, b = ts[0], ts[1]
            m, k = a.shape
            k2, n = b.shape
            if int(k) == int(k2):
                fwd = 2.0 * int(m) * int(n) * int(k)
                return float(fwd * (1.0 + max(0.0, self._effective_bwd)))

        return self._eltwise(out, 1.0)


def _to_real_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, _ShapeTensor):
        # allocate smallest placeholder; only used for non-attention ops where shape doesn't matter much
        return torch.empty((1, 1), device="cpu")
    return torch.empty((1, 1), device="cpu")


# ==============================================================================
# Profiler core (nested-safe accumulator)
# ==============================================================================

@dataclass
class _Acc:
    total: float = 0.0
    by_type: Dict[str, float] = field(default_factory=dict)


class _FlopProfiler:
    _stack_var: contextvars.ContextVar[Tuple[_Acc, ...]] = contextvars.ContextVar("stnet_flops_stack", default=())
    _nvtx_getter: Optional[Callable[[], float]] = None

    def _stack(self) -> Tuple[_Acc, ...]:
        return self._stack_var.get()

    def is_active(self) -> bool:
        return len(self._stack()) > 0

    def activate(self) -> None:
        stack = self._stack()
        self._stack_var.set(stack + (_Acc(),))

    def deactivate(self) -> None:
        stack = self._stack()
        if not stack:
            return
        self._stack_var.set(stack[:-1])

    def reset(self) -> None:
        stack = self._stack()
        if not stack:
            return
        acc = stack[-1]
        acc.total = 0.0
        acc.by_type.clear()

    def pop(self) -> Tuple[float, Dict[str, float]]:
        stack = self._stack()
        if not stack:
            return (0.0, {})
        acc = stack[-1]
        total = float(acc.total)
        breakdown = {k: float(v) for k, v in acc.by_type.items()}
        acc.total = 0.0
        acc.by_type.clear()
        return (total, breakdown)

    def sum(self, *, sort: bool = True) -> Tuple[float, Dict[str, float]]:
        stack = self._stack()
        if not stack:
            return (0.0, {})
        acc = stack[-1]
        total = float(acc.total)
        if not sort:
            return total, {k: float(v) for k, v in acc.by_type.items()}
        ordered: Dict[str, float] = {}
        for name, value in sorted(acc.by_type.items(), key=lambda kv: kv[1], reverse=True):
            ordered[name] = float(value)
        return total, ordered

    def add(self, typ: str, value: float) -> None:
        fv = _safe_float(value, 0.0)
        if fv <= 0.0:
            return
        stack = self._stack()
        if not stack:
            return
        for acc in stack:
            acc.total += fv
            acc.by_type[typ] = acc.by_type.get(typ, 0.0) + fv

    # ---- NVTX getter (optional)

    def coerce_flops_nvtx(self) -> None:
        if self._nvtx_getter is not None:
            return
        hook = os.getenv("STNET_NVTX_GETTER", "")
        if not hook:
            self._nvtx_getter = None
            return
        try:
            module_name, attr = hook.split(":", 1)
        except ValueError:
            self._nvtx_getter = None
            return
        getter: Optional[Callable[[], float]] = None
        try:
            module = __import__(module_name, fromlist=[attr])
            candidate = getattr(module, attr)
            if callable(candidate):
                getter = candidate
        except Exception as exc:
            _LOGGER.debug("Failed to import NVTX getter %s: %s", hook, exc)
        self._nvtx_getter = getter

    def coerce_flops_ntvx(self) -> None:
        self.coerce_flops_nvtx()

    def new_flops_nvtx(self, device: Optional[torch.device] = None) -> Any:
        self.coerce_flops_nvtx()
        getter = self._nvtx_getter
        try:
            if getter is None:
                raise RuntimeError("NVTX getter not set")
            from .system import accel_is_available
            if not accel_is_available("cuda"):
                raise RuntimeError("CUDA not available")
            getattr(torch.cuda, "nvtx")
        except Exception:
            return contextlib.nullcontext()

        class _NvtxScope(contextlib.AbstractContextManager[Any]):
            def __init__(self, dev: Optional[torch.device]) -> None:
                self._dev = dev
                self._base = 0.0

            def __enter__(self) -> "_NvtxScope":
                try:
                    if self._dev is not None and getattr(self._dev, "type", "") == "cuda":
                        from .system import accel_synchronize
                        accel_synchronize(self._dev)
                except Exception:
                    pass
                try:
                    self._base = float(getter())
                except Exception:
                    self._base = 0.0
                return self

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> bool:
                return False

            def get_total_flops(self) -> float:
                try:
                    cur = float(getter())
                    return max(0.0, cur - float(self._base))
                except Exception:
                    return 0.0

        return _NvtxScope(device)

    def new_flops_ntvx(self, device: Optional[torch.device] = None) -> Any:
        return self.new_flops_nvtx(device)

    # ---- torch.profiler capture (optional)

    def _capture_torch(self, display: bool = False) -> Any:
        try:
            from torch.profiler import profile as _profile
        except Exception:
            _profile = None

        if _profile is not None:

            class _TorchFlops(contextlib.AbstractContextManager[Any]):
                def __init__(self, show: bool) -> None:
                    self._show = show
                    self._prof: Any = None

                def __enter__(self) -> "_TorchFlops":
                    self._prof = _profile(with_flops=True, record_shapes=False)
                    self._prof.__enter__()
                    return self

                def __exit__(
                    self,
                    exc_type: type[BaseException] | None,
                    exc: BaseException | None,
                    tb: TracebackType | None,
                ) -> bool:
                    if self._prof is not None:
                        self._prof.__exit__(exc_type, exc, tb)
                        if self._show:
                            try:
                                table = self._prof.key_averages().table(sort_by="flops")
                                _LOGGER.info("%s", table)
                            except Exception:
                                pass
                    return False

                def get_total_flops(self) -> float:
                    if self._prof is None:
                        return 0.0
                    try:
                        events = self._prof.key_averages()
                        return float(sum(getattr(e, "flops", 0.0) for e in events))
                    except Exception:
                        return 0.0

            return _TorchFlops(bool(display))

        class _TorchFlopsCompat(contextlib.AbstractContextManager[Any]):
            def __init__(self, show: bool) -> None:
                try:
                    from torch.utils.flop_counter import \
                        FlopCounterMode as _TorchMode
                    self._impl = _TorchMode(display=show)
                except Exception:
                    self._impl = None

            def __enter__(self) -> "_TorchFlopsCompat":
                if self._impl is not None:
                    self._impl.__enter__()
                return self

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> bool:
                if self._impl is not None:
                    self._impl.__exit__(exc_type, exc, tb)
                return False

            def get_total_flops(self) -> float:
                if self._impl is None:
                    return 0.0
                try:
                    return float(self._impl.get_total_flops())
                except Exception:
                    return 0.0

        return _TorchFlopsCompat(bool(display))

    # ---- hook management

    def start_hooks(self, model: nn.Module, *, cfg: _HookConfig) -> List[Any]:
        handles: List[Any] = []
        skip: set[int] = set()

        # Avoid double-counting under TE fused modules by skipping their children.
        for module in model.modules():
            if id(module) in skip:
                continue

            hook = None

            if _is_te_module(module):
                try:
                    children = list(module.modules())[1:]
                    for c in children:
                        skip.add(id(c))
                except Exception:
                    pass
                hook = module.register_forward_hook(partial(_register_te_any, profiler=self, cfg=cfg))
            elif isinstance(module, nn.Linear):
                hook = module.register_forward_hook(partial(_register_linear, profiler=self, cfg=cfg))
            elif isinstance(module, nn.modules.conv._ConvNd):
                hook = module.register_forward_hook(partial(_register_conv, profiler=self, cfg=cfg))
            elif isinstance(module, nn.LayerNorm):
                hook = module.register_forward_hook(partial(_register_layernorm, profiler=self, cfg=cfg))
            elif cfg.count_softmax and isinstance(module, nn.Softmax):
                hook = module.register_forward_hook(partial(_register_softmax, profiler=self, cfg=cfg))
            elif cfg.count_dropout and isinstance(module, nn.Dropout):
                hook = module.register_forward_hook(partial(_register_dropout, profiler=self, cfg=cfg))
            elif cfg.count_embedding and isinstance(module, nn.Embedding):
                hook = module.register_forward_hook(partial(_register_embedding, profiler=self, cfg=cfg))
            elif cfg.count_activations and isinstance(module, _ACT_CLASSES):
                hook = module.register_forward_hook(partial(_register_activation, profiler=self, cfg=cfg))
            elif isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(partial(_register_mha, profiler=self, cfg=cfg))

            if hook is not None:
                handles.append(hook)

        return handles

    def stop_hooks(self, handles: Sequence[Any]) -> None:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    # ---- monitoring context

    def monitoring(
        self,
        device: Optional[torch.device],
        *,
        display: bool = False,
        use_torch_profiler: bool = True,
        use_nvtx: bool = True,
        dispatch_mode: Optional[Any] = None,
    ) -> Any:
        profiler = self

        class _Flops(contextlib.AbstractContextManager[Any]):
            def __init__(self) -> None:
                self.manual_total = 0.0
                self.manual_breakdown: Dict[str, float] = {}
                self.torch_total = 0.0
                self.nvtx_total = 0.0
                self.total = 0.0

                self._torch_scope: Any = None
                self._nvtx_scope: Any = None
                self._dispatch_scope: Any = None
                self._outer_active = False

            def __enter__(self) -> "_Flops":
                self._outer_active = profiler.is_active()
                profiler.activate()
                profiler.reset()

                if not self._outer_active:
                    if dispatch_mode is not None:
                        self._dispatch_scope = dispatch_mode
                        try:
                            self._dispatch_scope.__enter__()
                        except Exception:
                            self._dispatch_scope = None

                    if use_torch_profiler:
                        self._torch_scope = profiler._capture_torch(display)
                        if self._torch_scope is not None:
                            self._torch_scope.__enter__()

                    if use_nvtx:
                        self._nvtx_scope = profiler.new_flops_nvtx(device)
                        if self._nvtx_scope is not None:
                            self._nvtx_scope.__enter__()

                return self

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
                manual, breakdown = profiler.pop()
                self.manual_total = float(manual)
                self.manual_breakdown = breakdown

                if not self._outer_active:
                    if self._nvtx_scope is not None:
                        self._nvtx_scope.__exit__(exc_type, exc, tb)
                        try:
                            self.nvtx_total = float(self._nvtx_scope.get_total_flops())
                        except Exception:
                            self.nvtx_total = 0.0

                    if self._torch_scope is not None:
                        self._torch_scope.__exit__(exc_type, exc, tb)
                        try:
                            self.torch_total = float(self._torch_scope.get_total_flops())
                        except Exception:
                            self.torch_total = 0.0

                    if self._dispatch_scope is not None:
                        try:
                            self._dispatch_scope.__exit__(exc_type, exc, tb)
                        except Exception:
                            pass

                # Union-ish total:
                base = float(max(self.torch_total, self.nvtx_total))
                manual_total = float(self.manual_total)
                self.total = float(max(base, manual_total, 0.0))

                profiler.deactivate()
                return False

            def get_total_flops(self) -> float:
                if not profiler.is_active():
                    return float(self.total)
                cur_total, _ = profiler.sum(sort=False)
                return float(cur_total)

            def get_manual_breakdown(self) -> Dict[str, float]:
                if profiler.is_active():
                    _, b = profiler.sum(sort=False)
                    return dict(b)
                return dict(self.manual_breakdown)

            def to_dict(self) -> Dict[str, float]:
                return {
                    "manual_total": float(self.manual_total),
                    "torch_total": float(self.torch_total),
                    "nvtx_total": float(self.nvtx_total),
                    "total": float(self.total),
                }

            def verbose(self, top_k: int = 12) -> str:
                lines: list[str] = []
                lines.append(
                    f"total FLOPs: manual={self.manual_total:.3e}, "
                    f"torch={self.torch_total:.3e}, nvtx={self.nvtx_total:.3e}, total={self.total:.3e}"
                )
                if self.manual_breakdown:
                    lines.append(f"manual breakdown (top {top_k}):")
                    items = sorted(self.manual_breakdown.items(), key=lambda kv: kv[1], reverse=True)
                    for name, value in items[:top_k]:
                        lines.append(f"  - {name}: {value:.3e}")
                return "\n".join(lines)

        return _Flops()

    # ---- Attention helper for legacy capture(q)

    def capture(
        self,
        q: torch.Tensor,
        *,
        bwd_factor: float = 2.0,
        dropout_p: float = 0.0,
        training: bool = False,
        include_softmax_scale_dropout: bool = True,
    ) -> float:
        if not isinstance(q, torch.Tensor) or q.ndim < 4:
            return 0.0
        b, h, s, d = _infer_bhsd(q)
        total = _flops_attention_general(
            batch=b,
            num_heads=h,
            q_len=s,
            k_len=s,
            head_dim=d,
            effective_bwd=float(bwd_factor),
            dropout_p=float(dropout_p),
            training=bool(training),
            include_softmax_scale_dropout=bool(include_softmax_scale_dropout),
        )
        if total > 0.0:
            self.add("Attention", total)
        return float(total)


FLOP_PROFILER = _FlopProfiler()


# ==============================================================================
# User-facing API
# ==============================================================================

class _StaticFlops(contextlib.AbstractContextManager[Any]):
    def __init__(self, total: float, breakdown: Dict[str, float]) -> None:
        self.manual_total = float(total)
        self.manual_breakdown = dict(breakdown)
        self.torch_total = 0.0
        self.nvtx_total = 0.0
        self.total = float(total)

    def __enter__(self) -> "_StaticFlops":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def get_total_flops(self) -> float:
        return float(self.total)

    def get_manual_breakdown(self) -> Dict[str, float]:
        return dict(self.manual_breakdown)

    def to_dict(self) -> Dict[str, float]:
        return {"manual_total": float(self.manual_total), "torch_total": 0.0, "nvtx_total": 0.0, "total": float(self.total)}

    def verbose(self, top_k: int = 12) -> str:
        lines: list[str] = []
        lines.append(f"total FLOPs (static): {self.total:.3e}")
        if self.manual_breakdown:
            lines.append(f"breakdown (top {top_k}):")
            items = sorted(self.manual_breakdown.items(), key=lambda kv: kv[1], reverse=True)
            for name, value in items[:top_k]:
                lines.append(f"  - {name}: {value:.3e}")
        return "\n".join(lines)


class _HybridFlops(contextlib.AbstractContextManager[Any]):

    def __init__(
        self,
        inner: Any,
        *,
        static_total: float,
        static_breakdown: Dict[str, float],
        cache_slot: Optional[Dict[Any, Tuple[float, Dict[str, float]]]] = None,
        cache_key: Any = None,
    ) -> None:
        self._inner = inner
        self._static_total = float(static_total)
        self._static_breakdown = dict(static_breakdown)
        self._cache_slot = cache_slot
        self._cache_key = cache_key

        self.manual_total = 0.0
        self.manual_breakdown: Dict[str, float] = {}
        self.torch_total = 0.0
        self.nvtx_total = 0.0
        self.total = 0.0

    def __enter__(self) -> "_HybridFlops":
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self._inner.__exit__(exc_type, exc, tb)

        self.manual_total = float(getattr(self._inner, "manual_total", 0.0))
        self.manual_breakdown = dict(getattr(self._inner, "manual_breakdown", {}) or {})
        self.torch_total = float(getattr(self._inner, "torch_total", 0.0))
        self.nvtx_total = float(getattr(self._inner, "nvtx_total", 0.0))
        self.total = float(getattr(self._inner, "total", 0.0))

        if self.total > 0.0 and self._cache_slot is not None and self._cache_key is not None:
            self._cache_slot[self._cache_key] = (float(self.total), dict(self.manual_breakdown))

        if self.total <= 0.0 and self._static_total > 0.0:
            self.manual_total = float(self._static_total)
            self.manual_breakdown = dict(self._static_breakdown)
            self.total = float(self._static_total)

        return False

    def get_total_flops(self) -> float:
        try:
            return float(self._inner.get_total_flops())
        except Exception:
            return float(self.total)

    def get_manual_breakdown(self) -> Dict[str, float]:
        try:
            return dict(self._inner.get_manual_breakdown())
        except Exception:
            return dict(self.manual_breakdown)

    def to_dict(self) -> Dict[str, float]:
        return {"manual_total": float(self.manual_total), "torch_total": float(self.torch_total), "nvtx_total": float(self.nvtx_total), "total": float(self.total)}

    def verbose(self, top_k: int = 12) -> str:
        try:
            return str(self._inner.verbose(top_k=top_k))
        except Exception:
            return _StaticFlops(self.total, self.manual_breakdown).verbose(top_k=top_k)


class FlopCounter:

    def __init__(
        self,
        model: nn.Module,
        *,
        mode: str = "train",
        device: Optional[torch.device] = None,
        include_bias: bool = True,
        bwd_factor: Optional[float] = None,
        backend: str = "auto",  # auto|hooks|dispatch|dynamo|torch
        estimate_bwd: bool = True,
        count_activations: bool = True,
        count_norms: bool = True,
        count_softmax: bool = True,
        count_dropout: bool = True,
        count_embedding: bool = True,
        count_elementwise: bool = True,
        static_fallback_on_zero: bool = True,
    ) -> None:
        self._model = model
        self._mode = str(mode)
        self._device = device
        self._include_bias = bool(include_bias)
        self._bwd_factor = bwd_factor
        self._backend = str(backend)
        self._estimate_bwd = bool(estimate_bwd)

        self._count_activations = bool(count_activations)
        self._count_norms = bool(count_norms)
        self._count_softmax = bool(count_softmax)
        self._count_dropout = bool(count_dropout)
        self._count_embedding = bool(count_embedding)
        self._count_elementwise = bool(count_elementwise)
        self._static_fallback_on_zero = bool(static_fallback_on_zero)

        self._handles: List[Any] = []
        self._hook_count = 0
        self._active = False

        self._static_cache: Dict[Any, Tuple[float, Dict[str, float]]] = {}
        self._runtime_cache: Dict[Any, Tuple[float, Dict[str, float]]] = {}

        self._fx_estimator: Optional[_FxGraphFlopEstimator] = None
        self._effective_backend = self._backend.lower()

    @property
    def device(self) -> Optional[torch.device]:
        return self._device

    @property
    def hook_count(self) -> int:
        return int(self._hook_count)

    def _effective_bwd(self) -> float:
        eff = 2.0 if self._mode == "train" else 0.0
        if self._bwd_factor is not None:
            eff = _safe_float(self._bwd_factor, eff)
        return float(eff if self._estimate_bwd else 0.0)

    def _is_compiled(self) -> bool:
        try:
            from torch._dynamo.eval_frame import OptimizedModule
            return isinstance(self._model, OptimizedModule)
        except Exception:
            return hasattr(self._model, "_orig_mod")

    def _orig_model(self) -> nn.Module:
        m = self._model
        if hasattr(m, "_orig_mod"):
            try:
                return getattr(m, "_orig_mod")
            except Exception:
                return m
        return m

    def __enter__(self) -> "FlopCounter":
        eff_bwd = self._effective_bwd()
        cfg = _HookConfig(
            include_bias=self._include_bias,
            effective_bwd=eff_bwd,
            count_activations=self._count_activations,
            count_norms=self._count_norms,
            count_softmax=self._count_softmax,
            count_dropout=self._count_dropout,
            count_embedding=self._count_embedding,
        )

        backend = self._backend.lower()
        if backend == "auto":
            backend = "dynamo" if self._is_compiled() else ("dispatch" if TorchDispatchMode is not None else "hooks")
        self._effective_backend = backend

        if backend == "hooks":
            self._handles = FLOP_PROFILER.start_hooks(self._model, cfg=cfg)
            self._hook_count = len(self._handles)
        else:
            self._handles = []
            self._hook_count = 0

        self._active = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._active:
            if self._handles:
                FLOP_PROFILER.stop_hooks(self._handles)
            self._handles = []
            self._active = False
        return False

    def _sig_key(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        def k_of(x: Any) -> Any:
            if isinstance(x, torch.Tensor):
                return ("T", tuple(int(s) for s in x.shape), str(x.dtype), str(x.device.type))
            if isinstance(x, (tuple, list)):
                return tuple(k_of(v) for v in x)
            if isinstance(x, dict):
                return tuple(sorted((kk, k_of(vv)) for kk, vv in x.items()))
            return ("O", type(x).__name__, repr(x)[:64])

        return (tuple(k_of(a) for a in args), k_of(kwargs))

    def prepare(self, *example_args: Any, **example_kwargs: Any) -> Tuple[float, Dict[str, float]]:
        key = self._sig_key(example_args, example_kwargs)
        if key in self._static_cache:
            return self._static_cache[key]

        eff_bwd = self._effective_bwd()
        self._fx_estimator = _FxGraphFlopEstimator(
            include_bias=self._include_bias,
            effective_bwd=eff_bwd,
            count_elementwise=self._count_elementwise,
        )

        gm = _export_fx(self._orig_model(), example_args, example_kwargs)
        if gm is None:
            res = (0.0, {})
            self._static_cache[key] = res
            return res

        _shape_propagate(gm, example_args, example_kwargs)
        total, breakdown = self._fx_estimator.estimate(gm)

        res = (float(total), dict(breakdown))
        self._static_cache[key] = res
        return res

    def step(
        self,
        *,
        display: bool = False,
        example_args: Optional[Tuple[Any, ...]] = None,
        example_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if not self._active:
            raise RuntimeError("FlopCounter is not active. Use `with FlopCounter(...)` before measuring FLOPs.")

        backend = self._effective_backend.lower()
        eff_bwd = self._effective_bwd()

        ex_args = example_args or ()
        ex_kwargs = example_kwargs or {}
        key = self._sig_key(ex_args, ex_kwargs) if (example_args is not None or example_kwargs is not None) else None

        cached_total = 0.0
        cached_breakdown: Dict[str, float] = {}
        if key is not None and key in self._runtime_cache:
            cached_total, cached_breakdown = self._runtime_cache[key]

        static_total = 0.0
        static_breakdown: Dict[str, float] = {}

        if key is not None and (cached_total <= 0.0) and (backend in ("dynamo", "hooks", "dispatch")):
            static_total, static_breakdown = self.prepare(*ex_args, **ex_kwargs)

        match backend:
            case "dynamo":
                if key is not None and static_total > 0.0:
                    return _StaticFlops(static_total, static_breakdown)
                return FLOP_PROFILER.monitoring(
                    self._device,
                    display=display,
                    use_torch_profiler=True,
                    use_nvtx=True,
                    dispatch_mode=None,
                )

            case "dispatch":
                if TorchDispatchMode is None:
                    inner = FLOP_PROFILER.monitoring(
                        self._device,
                        display=display,
                        use_torch_profiler=True,
                        use_nvtx=True,
                        dispatch_mode=None,
                    )
                else:
                    dispatch = _OpFlopDispatchMode(
                        FLOP_PROFILER,
                        include_bias=self._include_bias,
                        effective_bwd=eff_bwd,
                        count_elementwise=self._count_elementwise,
                    )
                    inner = FLOP_PROFILER.monitoring(
                        self._device,
                        display=display,
                        use_torch_profiler=False,
                        use_nvtx=True,
                        dispatch_mode=dispatch,
                    )

                if self._static_fallback_on_zero and key is not None and (static_total > 0.0 or cached_total > 0.0):
                    fallback_total = float(cached_total) if cached_total > 0.0 else float(static_total)
                    fallback_breakdown = dict(cached_breakdown) if cached_total > 0.0 else dict(static_breakdown)
                    return _HybridFlops(
                        inner,
                        static_total=fallback_total,
                        static_breakdown=fallback_breakdown,
                        cache_slot=self._runtime_cache,
                        cache_key=key,
                    )

                return inner

            case "hooks":
                inner = FLOP_PROFILER.monitoring(
                    self._device,
                    display=display,
                    use_torch_profiler=True,
                    use_nvtx=True,
                    dispatch_mode=None,
                )
                if self._static_fallback_on_zero and key is not None and (static_total > 0.0 or cached_total > 0.0):
                    fallback_total = float(cached_total) if cached_total > 0.0 else float(static_total)
                    fallback_breakdown = dict(cached_breakdown) if cached_total > 0.0 else dict(static_breakdown)
                    return _HybridFlops(
                        inner,
                        static_total=fallback_total,
                        static_breakdown=fallback_breakdown,
                        cache_slot=self._runtime_cache,
                        cache_key=key,
                    )
                return inner

            case _:
                # backend == "torch" or unknown
                return FLOP_PROFILER.monitoring(
                    self._device,
                    display=display,
                    use_torch_profiler=True,
                    use_nvtx=True,
                    dispatch_mode=None,
                )


def _export_fx(model: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[torch.fx.GraphModule]:
    try:
        import torch._dynamo as dynamo
        exported = dynamo.export(model, aten_graph=True)
        res = exported(*args, **kwargs)
        return res.graph_module
    except Exception as exc:
        _LOGGER.debug("dynamo.export failed: %s", exc)
        return None


def _shape_propagate(gm: torch.fx.GraphModule, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    try:
        from torch.fx.passes.shape_prop import ShapeProp
        ShapeProp(gm).propagate(*args, **kwargs)
    except Exception as exc:
        _LOGGER.debug("ShapeProp failed: %s", exc)


# ==============================================================================
# Functional helpers (backward-compatible)
# ==============================================================================

def capture(
    q: torch.Tensor,
    *,
    bwd_factor: float = 1.0,
    dropout_p: float = 0.0,
    training: bool = False,
    include_softmax_scale_dropout: bool = True,
) -> float:
    return FLOP_PROFILER.capture(
        q,
        bwd_factor=bwd_factor,
        dropout_p=dropout_p,
        training=training,
        include_softmax_scale_dropout=include_softmax_scale_dropout,
    )
