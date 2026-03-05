# -*- coding: utf-8 -*-
from __future__ import annotations

# =============================================================================
# 1. Standard Library Imports
# =============================================================================
import contextlib
import contextvars
import logging
import math
import os
from dataclasses import dataclass, field
from functools import partial
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Self, Sequence, Tuple

# =============================================================================
# 2. Third-Party Imports
# =============================================================================
import torch
from torch import nn

# =============================================================================
# 3. Local Imports
# =============================================================================
from ..core.datatypes import env_bool, env_int


# =============================================================================
# Lazy Declarations & Fallbacks
# =============================================================================
try:
    from torch.utils._python_dispatch import TorchDispatchMode
except Exception:
    TorchDispatchMode = None

try:
    from torch._ops import OpOverload, OpOverloadPacket
except Exception:
    OpOverload = tuple()
    OpOverloadPacket = tuple()

try:
    from torch._higher_order_ops.higher_order_operator import HigherOrderOperator
except Exception:
    HigherOrderOperator = tuple()


# =============================================================================
# Globals & Constants
# =============================================================================
_LOGGER = logging.getLogger(__name__)

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


# =============================================================================
# Internal Helpers
# =============================================================================
def _float_safe(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if v == v else default
    except Exception:
        return default


def _int_safe(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _prod_int(xs: Sequence[int]) -> int:
    return int(math.prod(int(v) for v in xs)) if xs else 1


def _coerce(obj: Any) -> Any:
    if obj is None:
        return None
    with contextlib.suppress(Exception):
        if isinstance(obj, OpOverload):
            return obj
        return getattr(obj, "default", obj)
    return obj


def _get_forward(out: Any) -> Optional[torch.Tensor]:
    match out:
        case torch.Tensor():
            return out
        case tuple() | list():
            return next((v for v in out if isinstance(v, torch.Tensor)), None)
        case _:
            return None


def _coerce_tensor_sequence(args: Tuple[Any, ...], max_n: int = 4) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for a in args:
        match a:
            case torch.Tensor():
                out.append(a)
            case tuple() | list():
                out.extend(v for v in a if isinstance(v, torch.Tensor))
        if len(out) >= max_n:
            return out[:max_n]
    return out


def _infer_bhsd_shape(shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    if not shape or len(shape) != 4:
        return (0, 0, 0, 0)
    return (
        (int(shape[0]), int(shape[1]), int(shape[2]), int(shape[-1]))
        if shape[1] <= shape[2]
        else (int(shape[0]), int(shape[2]), int(shape[1]), int(shape[-1]))
    )


def _bhsd_shape(x: torch.Tensor) -> Tuple[int, int, int, int]:
    match x:
        case torch.Tensor() if x.ndim == 4:
            return _infer_bhsd_shape(x.shape)
        case _:
            return (0, 0, 0, 0)


def _te_layernormmlp_name_score(name: str, for_w2: bool) -> int:
    s = name.lower()
    if for_w2:
        if any(k in s for k in ("fc2", "w2", "linear2", "dense_4h_to_h")):
            return 3
        if "out" in s or "proj" in s:
            return 2
    else:
        if any(k in s for k in ("fc1", "w1", "linear1", "dense_h_to_4h")):
            return 3
        if "in" in s or "qkv" in s:
            return 2
    return 1


def _register_op_handler(handlers: Dict[Any, Any], op: Any, handler: Any) -> None:
    if op is not None:
        handlers[op] = handler


def _aten_ops_from(aten: Any, base: str) -> List[Any]:
    ops: List[Any] = []
    for pobj in (getattr(aten, base, None), getattr(aten, base + "_", None)):
        if pobj is None:
            continue
        for o in (
            "Tensor", "Scalar", "default", "self", "ScalarSelf",
            "ScalarOther", "Tensor_out", "Scalar_out", "self_out",
        ):
            f = getattr(pobj, o, None)
            if f is not None:
                ops.append(f)
    return ops


def _fx_resolve(obj: Any, gm: torch.fx.GraphModule) -> Any:
    match obj:
        case torch.fx.Node():
            return _fx_resolve_node(obj, gm)
        case tuple() | list():
            return type(obj)(_fx_resolve(x, gm) for x in obj)
        case dict():
            return {k: _fx_resolve(v, gm) for k, v in obj.items()}
        case _:
            return obj


def _fx_resolve_node(n: torch.fx.Node, gm: torch.fx.GraphModule) -> Any:
    v = n.meta.get("val", None)
    if v is not None:
        return v
    tm = n.meta.get("tensor_meta", None)
    if (shp := _meta_shape(tm)) is not None:
        return _TensorShape(shp)
    if n.op == "get_attr":
        return getattr(gm, str(n.target), None)
    return None


def _flop_sig_key_of(x: Any) -> Any:
    match x:
        case torch.Tensor():
            return ("T", tuple(int(s) for s in x.shape), str(x.dtype), str(x.device.type))
        case tuple() | list():
            return tuple(_flop_sig_key_of(v) for v in x)
        case dict():
            return tuple(sorted((kk, _flop_sig_key_of(vv)) for kk, vv in x.items()))
        case _:
            return ("O", type(x).__name__, repr(x)[:64])


def _op_name(func: Any) -> str:
    try:
        return str(func).replace("torch.ops.aten.", "aten.")
    except Exception:
        return "op"


def _is_tensorlike(x: Any) -> bool:
    return isinstance(x, (torch.Tensor, _TensorShape))


def _meta_shape(meta: Any) -> Optional[Tuple[int, ...]]:
    try:
        return tuple(int(s) for s in meta.shape) if meta is not None and getattr(meta, "shape", None) else None
    except Exception:
        return None


def _to_tensor(x: Any) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.empty((1, 1), device="cpu")


# =============================================================================
# FLOPs Math Formulations
# =============================================================================
def _linear_mkn_shape(inp: torch.Tensor, out: Any, weight: Optional[torch.Tensor]) -> Tuple[int, int, int]:
    if not isinstance(inp, torch.Tensor) or inp.numel() == 0:
        return (0, 0, 0)
    if isinstance(weight, torch.Tensor) and weight.ndim >= 2:
        n_dim, k_dim = _int_safe(weight.shape[0], 0), _int_safe(weight.shape[-1], 0)
    else:
        k_dim = _int_safe(inp.shape[-1], 0)
        n_dim = _int_safe(out.shape[-1], 0) if isinstance(out, torch.Tensor) and out.numel() > 0 and out.ndim >= 1 else 0
        
    if k_dim <= 0 or n_dim <= 0:
        return (0, 0, 0)
        
    m_dim = _int_safe(inp.numel() // max(k_dim, 1), 0)
    if isinstance(out, torch.Tensor) and out.numel() > 0:
        m_dim = max(m_dim, _int_safe(out.numel() // max(n_dim, 1), 0))
    return (m_dim, k_dim, n_dim)


def _flops_linear(
    inp: torch.Tensor, out: Any, weight: Optional[torch.Tensor], *args: Any,
    include_bias: bool, has_bias: bool, effective_bwd: float,
) -> float:
    m_dim, k_dim, n_dim = _linear_mkn_shape(inp, out, weight)
    if m_dim <= 0 or k_dim <= 0 or n_dim <= 0:
        return 0.0
    fwd = 2.0 * m_dim * k_dim * n_dim + (float(m_dim * n_dim) if (include_bias and has_bias) else 0.0)
    return float(fwd * (1.0 + max(0.0, _float_safe(effective_bwd, 0.0))))


def _flops_conv(
    inp: torch.Tensor, out: Any, weight: Optional[torch.Tensor], *args: Any,
    groups: int, include_bias: bool, has_bias: bool, effective_bwd: float,
) -> float:
    if not isinstance(weight, torch.Tensor) or weight.ndim < 3:
        return 0.0
    out_t = _get_forward(out)
    if out_t is None or out_t.numel() == 0:
        return 0.0
    try:
        out_elems, g = int(out_t.numel()), max(1, int(groups))
        cin_total = int(inp.shape[1]) if isinstance(inp, torch.Tensor) and inp.ndim >= 2 else int(weight.shape[1] * g)
        cin_per_group = max(1, cin_total // g)
        fwd = out_elems * (2.0 * cin_per_group * (int(weight[0].numel()) // max(cin_per_group, 1)))
        if include_bias and has_bias:
            fwd += float(out_elems)
        return float(fwd * (1.0 + max(0.0, _float_safe(effective_bwd, 0.0))))
    except Exception:
        return 0.0


def _flops_elementwise(out: Any, *args: Any, coeff: float, effective_bwd: float) -> float:
    out_t = _get_forward(out)
    if out_t is None or out_t.numel() == 0:
        return 0.0
    fwd = float(out_t.numel()) * float(coeff)
    return float(fwd * (1.0 + max(0.0, _float_safe(effective_bwd, 0.0))))


def _flops_softmax(inp: torch.Tensor, out: Any, *args: Any, dim: int, effective_bwd: float) -> float:
    out_t = _get_forward(out)
    if not isinstance(inp, torch.Tensor) or inp.numel() == 0 or out_t is None or out_t.numel() == 0:
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
    fwd = float(int(inp.numel() // max(cols, 1))) * (5.0 * float(cols)) if cols > 0 else 0.0
    return float(fwd * (1.0 + max(0.0, _float_safe(effective_bwd, 0.0))))


def _flops_layernorm(
    inp: torch.Tensor, out: Any, *args: Any,
    normalized_shape: Sequence[int], elementwise_affine: bool, has_bias: bool, effective_bwd: float,
) -> float:
    out_t = _get_forward(out)
    if not isinstance(inp, torch.Tensor) or inp.numel() == 0 or out_t is None or out_t.numel() == 0:
        return 0.0
    n_norm = _prod_int([int(x) for x in normalized_shape]) if normalized_shape else int(inp.shape[-1])
    if n_norm <= 0:
        return 0.0
    groups = int(out_t.numel() // max(n_norm, 1))
    affine = 0.0
    if elementwise_affine:
        affine += 1.0 + (1.0 if has_bias else 0.0)
    fwd = float(groups) * float(n_norm) * (6.0 + affine)
    return float(fwd * (1.0 + max(0.0, _float_safe(effective_bwd, 0.0))))


def _flops_attention_generics(
    *args: Any,
    batch: int, num_heads: int, q_len: int, k_len: int, head_dim: int,
    effective_bwd: float, dropout_p: float, training: bool, include_softmax_scale_dropout: bool,
) -> float:
    if any(x <= 0 for x in (batch, q_len, k_len, num_heads, head_dim)):
        return 0.0
    matmul = 4.0 * batch * num_heads * q_len * k_len * head_dim
    misc = (6.0 + (1.0 if training and dropout_p > 0.0 else 0.0)) * (batch * num_heads * q_len * k_len) if include_softmax_scale_dropout else 0.0
    fwd = matmul + misc
    return float(fwd * (1.0 + max(0.0, _float_safe(effective_bwd, 0.0))))


def _flops_attention_qkv(
    q: torch.Tensor, k: Optional[torch.Tensor], v: Optional[torch.Tensor], *args: Any,
    effective_bwd: float, dropout_p: float, training: bool, include_softmax_scale_dropout: bool,
) -> float:
    if not isinstance(q, torch.Tensor) or q.ndim != 4:
        return 0.0
    b, h, qlen, d = _bhsd_shape(q)
    klen = qlen
    if isinstance(k, torch.Tensor) and k.ndim == 4:
        _, _, klen, _ = _bhsd_shape(k)
    return _flops_attention_generics(
        batch=b, num_heads=h, q_len=qlen, k_len=klen, head_dim=d,
        effective_bwd=effective_bwd, dropout_p=dropout_p, training=training,
        include_softmax_scale_dropout=include_softmax_scale_dropout,
    )


# =============================================================================
# Graph & Module Analyzers
# =============================================================================
def _export_graph(model: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[torch.fx.GraphModule]:
    try:
        import torch._dynamo as dynamo
        exported = dynamo.export(model, aten_graph=True)
        res = exported(*args, **kwargs)
        return res.graph_module
    except Exception as exc:
        _LOGGER.debug("dynamo.export failed: %s", exc)
        return None


def _forward_shape(gm: torch.fx.GraphModule, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    try:
        from torch.fx.passes.shape_prop import ShapeProp
        ShapeProp(gm).propagate(*args, **kwargs)
    except Exception as exc:
        _LOGGER.debug("ShapeProp failed: %s", exc)


def _activation_coefficients(t: type) -> Optional[float]:
    if (coeff := _ACT_COEFF.get(t)) is not None:
        return float(coeff)
    for cls, c in _ACT_COEFF.items():
        with contextlib.suppress(Exception):
            if issubclass(t, cls):
                return float(c)
    return None


def _is_te_module(mod: nn.Module) -> bool:
    return "transformer_engine" in getattr(type(mod), "__module__", "")


# =============================================================================
# Hooks Registers
# =============================================================================
def _register_linear(mod: nn.Module, inp: Tuple[Any, ...], out: Any, *args: Any, profiler: "_FlopProfiler", cfg: _ProfilerConfig) -> None:
    x = inp[0] if inp else None
    if not isinstance(x, torch.Tensor): return
    weight = getattr(mod, "weight", None) or getattr(getattr(mod, "linear", None), "weight", None)
    if not isinstance(weight, torch.Tensor): return
    has_bias = getattr(mod, "bias", None) is not None
    val = _flops_linear(x, out, weight, include_bias=cfg.include_bias, has_bias=has_bias, effective_bwd=cfg.effective_bwd)
    if val > 0.0:
        typ = f"TE.{type(mod).__name__}" if _is_te_module(mod) else type(mod).__name__
        profiler.add(typ, val)


def _register_conv(mod: nn.Module, inp: Tuple[Any, ...], out: Any, *args: Any, profiler: "_FlopProfiler", cfg: _ProfilerConfig) -> None:
    x = inp[0] if inp else None
    if not isinstance(x, torch.Tensor): return
    weight = getattr(mod, "weight", None)
    if not isinstance(weight, torch.Tensor): return
    has_bias = getattr(mod, "bias", None) is not None
    val = _flops_conv(x, out, weight, groups=int(getattr(mod, "groups", 1)), include_bias=cfg.include_bias, has_bias=has_bias, effective_bwd=cfg.effective_bwd)
    if val > 0.0: profiler.add(type(mod).__name__, val)


def _register_activation(mod: nn.Module, inp: Tuple[Any, ...], out: Any, *args: Any, profiler: "_FlopProfiler", cfg: _ProfilerConfig) -> None:
    if not cfg.count_activations: return
    if (coeff := _activation_coefficients(type(mod))) is None: return
    val = _flops_elementwise(out, coeff=float(coeff), effective_bwd=cfg.effective_bwd)
    if val > 0.0: profiler.add(type(mod).__name__, val)


def _register_dropout(mod: nn.Module, inp: Tuple[Any, ...], out: Any, *args: Any, profiler: "_FlopProfiler", cfg: _ProfilerConfig) -> None:
    if not cfg.count_dropout or not bool(getattr(mod, "training", False)): return
    val = _flops_elementwise(out, coeff=2.0, effective_bwd=cfg.effective_bwd)
    if val > 0.0: profiler.add(type(mod).__name__, val)


def _register_softmax(mod: nn.Module, inp: Tuple[Any, ...], out: Any, *args: Any, profiler: "_FlopProfiler", cfg: _ProfilerConfig) -> None:
    if not cfg.count_softmax: return
    x = inp[0] if inp else None
    if not isinstance(x, torch.Tensor): return
    val = _flops_softmax(x, out, dim=int(getattr(mod, "dim", -1)), effective_bwd=cfg.effective_bwd)
    if val > 0.0: profiler.add(type(mod).__name__, val)


def _register_layernorm(mod: nn.Module, inp: Tuple[Any, ...], out: Any, *args: Any, profiler: "_FlopProfiler", cfg: _ProfilerConfig) -> None:
    if not cfg.count_norms: return
    x = inp[0] if inp else None
    if not isinstance(x, torch.Tensor): return
    ns = getattr(mod, "normalized_shape", ())
    val = _flops_layernorm(x, out, normalized_shape=(list(ns) if ns is not None else ()), elementwise_affine=bool(getattr(mod, "elementwise_affine", True)), has_bias=getattr(mod, "bias", None) is not None, effective_bwd=cfg.effective_bwd)
    if val > 0.0: profiler.add(type(mod).__name__, val)


def _register_embedding(mod: nn.Module, inp: Tuple[Any, ...], out: Any, *args: Any, profiler: "_FlopProfiler", cfg: _ProfilerConfig) -> None:
    return


def _register_mha(mod: nn.MultiheadAttention, inp: Tuple[Any, ...], out: Any, *args: Any, profiler: "_FlopProfiler", cfg: _ProfilerConfig) -> None:
    if not inp: return
    q = inp[0]
    if not isinstance(q, torch.Tensor) or q.numel() == 0 or q.ndim != 3: return
    
    batch_first = bool(getattr(mod, "batch_first", False))
    bsz, seq_len, embed_dim = q.shape if batch_first else (q.shape[1], q.shape[0], q.shape[2])
    num_heads = int(getattr(mod, "num_heads", 0))
    if num_heads <= 0: return
    
    head_dim = int(embed_dim // num_heads) if int(embed_dim) % num_heads == 0 else int(getattr(mod, "head_dim", 0))
    if head_dim <= 0: return
    
    m, E = int(bsz) * int(seq_len), int(embed_dim)
    inproj_fwd = 2.0 * m * E * (3.0 * E)
    if cfg.include_bias and getattr(mod, "in_proj_bias", None) is not None:
        inproj_fwd += float(m * (3 * E))
        
    inproj = float(inproj_fwd * (1.0 + max(0.0, cfg.effective_bwd)))
    attn = _flops_attention_generics(
        batch=int(bsz), num_heads=int(num_heads), q_len=int(seq_len), k_len=int(seq_len), head_dim=int(head_dim),
        effective_bwd=cfg.effective_bwd, dropout_p=float(getattr(mod, "dropout", 0.0)), training=bool(getattr(mod, "training", False)),
        include_softmax_scale_dropout=True,
    )
    outproj = float((2.0 * m * E * E) * (1.0 + max(0.0, cfg.effective_bwd)))
    total = float(inproj + attn + outproj)
    if total > 0.0: profiler.add(type(mod).__name__, total)


def _get_tensor_attr(mod: nn.Module, name: str) -> Optional[torch.Tensor]:
    try:
        return v if isinstance(v := getattr(mod, name), torch.Tensor) else None
    except Exception:
        return None


def _get_2d_weights(mod: nn.Module) -> List[Tuple[str, torch.Tensor, bool]]:
    entries: List[Tuple[str, torch.Tensor, bool]] = []
    seen: set[int] = set()
    params: Dict[str, torch.Tensor] = {n: p for n, p in getattr(mod, "named_parameters", lambda **kw: [])(recurse=False) if isinstance(p, torch.Tensor)}
    buffers: Dict[str, torch.Tensor] = {n: b for n, b in getattr(mod, "named_buffers", lambda **kw: [])(recurse=False) if isinstance(b, torch.Tensor)}

    def has_weight_bias(wname: str) -> bool:
        cand = [wname.replace("weight", "bias"), wname.replace("_weight", "_bias"), wname.replace(".weight", ".bias")]
        for bn in cand:
            if isinstance(params.get(bn) or buffers.get(bn) or _get_tensor_attr(mod, bn), torch.Tensor): return True
        return isinstance(getattr(mod, "bias", None), torch.Tensor)

    def add(name: str, t: Any, has_bias: bool = False) -> None:
        if not (isinstance(t, torch.Tensor) and t.ndim == 2): return
        tid = id(t)
        if tid in seen: return
        seen.add(tid)
        entries.append((name, t, bool(has_bias)))

    for n, p in params.items(): add(f"param:{n}", p, has_bias=has_weight_bias(n))
    
    candidates = ["weight", "linear_weight", "fc1_weight", "fc2_weight", "w1", "w2", "weight1", "weight2", "in_proj_weight", "out_proj_weight", "qkv_weight", "proj_weight", "dense_h_to_4h_weight", "dense_4h_to_h_weight"]
    for n in candidates: add(f"attr:{n}", _get_tensor_attr(mod, n), has_bias=has_weight_bias(n))
        
    with contextlib.suppress(Exception):
        for name, sm in mod.named_modules():
            if name == "": continue
            add(f"sub:{name}.weight", getattr(sm, "weight", None), has_bias=isinstance(getattr(sm, "bias", None), torch.Tensor))
            if len(entries) >= 16: break
            
    for d_src, prefix in ((vars(mod), "var"), (list(dir(mod))[:256], "dir")):
        with contextlib.suppress(Exception):
            for n in d_src:
                if isinstance(n, str) and not n.startswith("_"):
                    v = d_src[n] if isinstance(d_src, dict) else _get_tensor_attr(mod, n)
                    if isinstance(v, torch.Tensor) and v.ndim == 2: add(f"{prefix}:{n}", v, has_bias=has_weight_bias(n))
                if len(entries) >= 16: break
    return entries


def _get_te_weights(mod: nn.Module) -> List[Tuple[str, torch.Tensor, bool]]:
    try:
        if isinstance(cache := getattr(mod, "_enn_te_weight_cache", None), dict) and cache.get("v") == 1 and isinstance(cache.get("entries"), list):
            return cache["entries"]
    except Exception: pass
    
    entries = _get_2d_weights(mod)
    with contextlib.suppress(Exception): setattr(mod, "_enn_te_weight_cache", {"v": 1, "entries": entries})
    return entries


def _get_te_norm(mod: nn.Module) -> Tuple[bool, bool]:
    has_weight = any(isinstance(_get_tensor_attr(mod, n), torch.Tensor) for n in ("weight", "ln_weight", "layer_norm_weight", "gamma"))
    has_bias = any(isinstance(_get_tensor_attr(mod, n), torch.Tensor) for n in ("bias", "ln_bias", "layer_norm_bias", "beta"))
    return (has_weight or has_bias), has_bias


def _register_te_module(mod: nn.Module, inp: Tuple[Any, ...], out: Any, *args: Any, profiler: "_FlopProfiler", cfg: _ProfilerConfig) -> None:
    x = inp[0] if inp else None
    cname = type(mod).__name__.lower()
    
    match cname:
        case _ if "attention" in cname:
            ts = _coerce_tensor_sequence(inp, max_n=3)
            q = ts[0] if len(ts) >= 1 else None
            if isinstance(q, torch.Tensor) and q.ndim == 4:
                val = _flops_attention_qkv(
                    q, ts[1] if len(ts) >= 2 else None, ts[2] if len(ts) >= 3 else None,
                    effective_bwd=cfg.effective_bwd, dropout_p=_float_safe(getattr(mod, "dropout", 0.0), 0.0),
                    training=bool(getattr(mod, "training", False)), include_softmax_scale_dropout=True,
                )
                if val > 0.0: profiler.add(f"TE.{type(mod).__name__}", val)
            return
            
        case _ if "layernormlinear" in cname or ("layernorm" in cname and "linear" in cname):
            if not isinstance(x, torch.Tensor): return
            elementwise_affine, ln_has_bias = _get_te_norm(mod)
            ln = _flops_layernorm(x, x, normalized_shape=[int(x.shape[-1])], elementwise_affine=elementwise_affine, has_bias=ln_has_bias, effective_bwd=cfg.effective_bwd)
            in_feat = int(x.shape[-1])
            w_entries = _get_te_weights(mod)
            hinted = [(n, w, hb) for (n, w, hb) in w_entries if any(k in n.lower() for k in ("linear", "proj", "fc"))]
            candidates = hinted if hinted else list(w_entries)
            pick_from = [t for t in candidates if int(t[1].shape[-1]) == in_feat] or candidates
            if not pick_from: return
            pick_from.sort(key=lambda t: int(t[1].numel()), reverse=True)
            lin = _flops_linear(x, out, pick_from[0][1], include_bias=cfg.include_bias, has_bias=pick_from[0][2], effective_bwd=cfg.effective_bwd)
            total = float(ln + lin)
            if total > 0.0: profiler.add(f"TE.{type(mod).__name__}", total)
            return
            
        case _ if "layernormmlp" in cname or ("mlp" in cname and "layernorm" in cname):
            if not isinstance(x, torch.Tensor): return
            w_entries = _get_te_weights(mod)
            if not w_entries: return
            elementwise_affine, ln_has_bias = _get_te_norm(mod)
            ln = _flops_layernorm(x, x, normalized_shape=[int(x.shape[-1])], elementwise_affine=elementwise_affine, has_bias=ln_has_bias, effective_bwd=cfg.effective_bwd)
            
            in_feat = int(x.shape[-1])
            cand1 = sorted(w_entries, key=lambda t: (_te_layernormmlp_name_score(t[0], False), int(t[1].numel())), reverse=True)
            match1 = [t for t in cand1 if int(t[1].shape[-1]) == in_feat]
            n1, w1, b1 = match1[0] if match1 else cand1[0]
            hidden = int(w1.shape[0])
            
            rest = [t for t in w_entries if id(t[1]) != id(w1)]
            cand2 = sorted(rest, key=lambda t: (_te_layernormmlp_name_score(t[0], True), int(t[1].numel())), reverse=True)
            match2 = [t for t in cand2 if int(t[1].shape[-1]) == hidden]
            w2_entry = match2[0] if match2 else (cand2[0] if cand2 else None)
            w2, b2 = (w2_entry[1], bool(w2_entry[2])) if w2_entry else (None, False)
            
            lin1 = _flops_linear(x, out, w1, include_bias=cfg.include_bias, has_bias=b1, effective_bwd=cfg.effective_bwd)
            m_dim = int(x.numel() // max(int(w1.shape[-1]), 1))
            act = float(int(m_dim * max(hidden, 0))) * 8.0 * (1.0 + max(0.0, cfg.effective_bwd)) if int(m_dim * max(hidden, 0)) > 0 else 0.0
            
            lin2 = 0.0
            if isinstance(w2, torch.Tensor) and w2.ndim == 2 and hidden > 0 and m_dim > 0:
                n2, k2 = int(w2.shape[0]), int(w2.shape[-1])
                if k2 > 0:
                    fwd2 = 2.0 * float(m_dim) * float(k2) * float(n2) + (float(m_dim) * float(n2) if cfg.include_bias and b2 else 0.0)
                    lin2 = float(fwd2 * (1.0 + max(0.0, cfg.effective_bwd)))
                    
            total = float(ln + lin1 + act + lin2)
            if total > 0.0: profiler.add(f"TE.{type(mod).__name__}", total)
            return
            
        case _:
            if not isinstance(x, torch.Tensor): return
            if w_entries := _get_te_weights(mod):
                in_feat = int(x.shape[-1])
                pick = [t for t in w_entries if int(t[1].shape[-1]) == in_feat] or list(w_entries)
                pick.sort(key=lambda t: int(t[1].numel()), reverse=True)
                val = _flops_linear(x, out, pick[0][1], include_bias=cfg.include_bias, has_bias=pick[0][2], effective_bwd=cfg.effective_bwd)
                if val > 0.0: profiler.add(f"TE.{type(mod).__name__}", val)


# =============================================================================
# Core Profiling Data Structures
# =============================================================================
class _TensorShape:
    def __init__(self: Self, shape: Tuple[int, ...]) -> None:
        self.shape = tuple(int(x) for x in shape)
        self.ndim = len(self.shape)

    def numel(self: Self) -> int:
        n = 1
        for v in self.shape: n *= int(v)
        return int(n)


@dataclass
class _Acc:
    total: float = 0.0
    by_type: Dict[str, float] = field(default_factory=dict)


@dataclass
class _ProfilerConfig:
    include_bias: bool
    effective_bwd: float
    count_activations: bool
    count_norms: bool
    count_softmax: bool
    count_dropout: bool
    count_embedding: bool


class _OpFlopDispatchMode(TorchDispatchMode):
    supports_higher_order_operators = True

    def __init__(
        self: Self, profiler: "_FlopProfiler", *args: Any,
        include_bias: bool, effective_bwd: float, count_elementwise: bool,
    ) -> None:
        super().__init__()
        self._profiler = profiler
        self._include_bias = bool(include_bias)
        self._effective_bwd = float(effective_bwd)
        self._count_elementwise = bool(count_elementwise)
        self._aten = torch.ops.aten
        
        handlers: Dict[Any, Callable] = {}
        self._tag_overrides: Dict[Any, str] = {}

        def _reg(op: object | None, handler: Callable[..., float]) -> None:
            _register_op_handler(handlers, op, handler)

        _reg(getattr(self._aten.mm, "default", None), self._h_mm)
        _reg(getattr(self._aten.bmm, "default", None), self._h_bmm)
        _reg(getattr(self._aten.matmul, "default", None), self._h_matmul)
        _reg(getattr(self._aten.addmm, "default", None), self._h_addmm)
        if getattr(self._aten, "linear", None): _reg(getattr(self._aten.linear, "default", None), self._h_linear)
        _reg(getattr(self._aten.convolution, "default", None), self._h_convolution)
        _reg(getattr(self._aten.native_layer_norm, "default", None), self._h_native_layer_norm)
        _reg(getattr(self._aten.layer_norm, "default", None), self._h_layer_norm)
        if getattr(self._aten, "dropout", None): _reg(getattr(self._aten.dropout, "default", None), self._h_dropout)
        
        if getattr(self._aten, "_softmax", None): _reg(getattr(self._aten._softmax, "default", None), self._h_softmax)
        if getattr(self._aten, "softmax", None) and hasattr(self._aten.softmax, "int"):
            _reg(getattr(self._aten.softmax, "int", None), self._h_softmax)
            
        if getattr(self._aten, "scaled_dot_product_attention", None):
            _reg(getattr(self._aten.scaled_dot_product_attention, "default", None), self._h_sdpa)

        for name in ("_scaled_dot_product_flash_attention", "_scaled_dot_product_flash_attention_for_cpu", "_scaled_dot_product_efficient_attention", "_scaled_dot_product_cudnn_attention", "_flash_attention_forward", "_efficient_attention_forward"):
            if op := getattr(self._aten, name, None):
                if hasattr(op, "default"): handlers[_coerce(op.default)] = self._h_sdpa_like
                
        if getattr(self._aten, "embedding", None): handlers[_coerce(self._aten.embedding.default)] = lambda a, k, o: 0.0

        if self._count_elementwise:
            def _reg_ops(bases: Sequence[str], handler: Callable[..., float], tag_prefix: str = "Elementwise") -> None:
                for base in bases:
                    for pobj in [getattr(self._aten, base, None), getattr(self._aten, base + "_", None)]:
                        if not pobj: continue
                        for overload in ("Tensor", "Scalar", "default", "self", "ScalarSelf", "ScalarOther", "Tensor_out", "Scalar_out", "self_out"):
                            if (f := getattr(pobj, overload, None)) and (op := _coerce(f)):
                                handlers[op], self._tag_overrides[op] = handler, f"{tag_prefix}.{base}"

            _reg_ops(("add", "sub", "mul", "div"), self._h_binop)
            _reg_ops(("addcmul", "addcdiv"), lambda a, k, o: _flops_elementwise(o, coeff=3.0, effective_bwd=self._effective_bwd))

            unary_coeff = {
                "abs": 1.0, "neg": 1.0, "floor": 1.0, "ceil": 1.0, "round": 1.0, "trunc": 1.0, "frac": 1.0, "sign": 1.0,
                "reciprocal": 4.0, "sqrt": 4.0, "rsqrt": 6.0, "exp": 10.0, "exp2": 10.0, "expm1": 10.0, "log": 10.0,
                "log2": 10.0, "log10": 10.0, "log1p": 10.0, "sin": 12.0, "cos": 12.0, "tan": 12.0, "asin": 14.0, "acos": 14.0,
                "atan": 14.0, "sinh": 14.0, "cosh": 14.0, "tanh": 4.0, "asinh": 14.0, "acosh": 14.0, "atanh": 14.0, "erf": 14.0,
                "erfc": 14.0, "erfinv": 18.0, "sigmoid": 4.0, "relu": 1.0, "gelu": 8.0, "silu": 6.0,
            }
            for name, coeff in unary_coeff.items():
                _reg_ops((name,), lambda a, k, o, c=coeff: _flops_elementwise(o, coeff=c, effective_bwd=self._effective_bwd))

            binary_coeff = {"atan2": 14.0, "minimum": 1.0, "maximum": 1.0, "fmin": 1.0, "fmax": 1.0, "clamp": 2.0, "clamp_min": 1.0, "clamp_max": 1.0, "where": 1.0, "lerp": 3.0}
            for name, coeff in binary_coeff.items():
                _reg_ops((name,), lambda a, k, o, c=coeff: _flops_elementwise(o, coeff=c, effective_bwd=self._effective_bwd))

            for pobj in (getattr(self._aten, "pow", None), getattr(self._aten, "pow_", None)):
                if not pobj: continue
                for overload in ("Tensor_Tensor", "Tensor_Scalar", "Scalar", "default"):
                    if (f := getattr(pobj, overload, None)) and (op := _coerce(f)):
                        handlers[op], self._tag_overrides[op] = self._h_pow, "Elementwise.pow"

            if fma_obj := getattr(self._aten, "fma", None):
                if hasattr(fma_obj, "default") and (op := _coerce(fma_obj.default)):
                    handlers[op], self._tag_overrides[op] = self._h_fma, "Elementwise.fma"
                    
        self._handlers = {k: v for k, v in handlers.items() if k is not None}

    def __torch_dispatch__(self: Self, func: Any, types: Any, args: Tuple[Any, ...] = (), kwargs: Optional[Dict[str, Any]] = None) -> Any:
        kwargs = kwargs or {}
        if HigherOrderOperator and isinstance(func, HigherOrderOperator): return NotImplemented

        out = func(*args, **kwargs)
        if (handler := self._handlers.get(func, None)) is None:
            name = _op_name(func).lower()
            if any(k in name for k in ("triton", "flex", "transformer_engine", "xformers")):
                val = self._h_custom(args, kwargs, out, name=name)
                if val > 0.0:
                    tag = "Triton" if "triton" in name else ("FlexAttention" if "flex" in name else ("TE" if "transformer_engine" in name else "Custom"))
                    self._profiler.add(tag, val)
            return out
            
        try: val = float(handler(args, kwargs, out))
        except Exception: val = 0.0
        
        if val > 0.0:
            self._profiler.add(self._tag_overrides.get(func) or _op_name(func), val)
        return out

    def _h_binop(self: Self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        return _flops_elementwise(out, coeff=1.0, effective_bwd=self._effective_bwd)

    def _h_pow(self: Self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        exp = args[1] if len(args) >= 2 else kwargs.get("exponent", None)
        coeff = 12.0
        try:
            if isinstance(exp, (int, float)):
                e = float(exp)
                if e == 0.5: coeff = 4.0
                elif e == -0.5: coeff = 6.0
                elif e.is_integer():
                    ei = int(e)
                    if 2 <= ei <= 4: coeff = float(ei - 1)
                    elif ei in (0, 1): coeff = 1.0
        except Exception: pass
        return _flops_elementwise(out, coeff=float(coeff), effective_bwd=self._effective_bwd)

    def _h_fma(self: Self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        return _flops_elementwise(out, coeff=2.0, effective_bwd=self._effective_bwd)

    def _h_mm(self: Self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        a, b = args[0], args[1]
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)) or a.ndim != 2 or b.ndim != 2: return 0.0
        m, k = a.shape
        k2, n = b.shape
        if int(k) != int(k2): return 0.0
        fwd = 2.0 * int(m) * int(n) * int(k)
        return float(fwd * (1.0 + max(0.0, self._effective_bwd)))

    def _h_bmm(self: Self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        a, b = args[0], args[1]
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)) or a.ndim != 3 or b.ndim != 3: return 0.0
        batch, m, k = a.shape
        batch2, k2, n = b.shape
        if int(batch) != int(batch2) or int(k) != int(k2): return 0.0
        fwd = 2.0 * int(batch) * int(m) * int(n) * int(k)
        return float(fwd * (1.0 + max(0.0, self._effective_bwd)))

    def _h_matmul(self: Self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        a, b = args[0], args[1]
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)): return 0.0
        if a.ndim == 2 and b.ndim == 2: return self._h_mm(args, kwargs, out)
        if a.ndim == 3 and b.ndim == 3: return self._h_bmm(args, kwargs, out)
        return 0.0

    def _h_addmm(self: Self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        if len(args) < 3: return 0.0
        input_, mat1, mat2 = args[0], args[1], args[2]
        if not (isinstance(mat1, torch.Tensor) and isinstance(mat2, torch.Tensor)) or mat1.ndim != 2 or mat2.ndim != 2: return 0.0
        m, k = mat1.shape
        k2, n = mat2.shape
        if int(k) != int(k2): return 0.0
        fwd = 2.0 * int(m) * int(n) * int(k)
        if self._include_bias and isinstance(input_, torch.Tensor) and input_.numel() > 0: fwd += float(int(m) * int(n))
        return float(fwd * (1.0 + max(0.0, self._effective_bwd)))

    def _h_linear(self: Self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        if len(args) < 2: return 0.0
        x, w = args[0], args[1]
        b = args[2] if len(args) >= 3 else None
        if not (isinstance(x, torch.Tensor) and isinstance(w, torch.Tensor)): return 0.0
        return _flops_linear(x, out, w, include_bias=self._include_bias, has_bias=isinstance(b, torch.Tensor), effective_bwd=self._effective_bwd)

    def _h_convolution(self: Self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        if len(args) < 3: return 0.0
        x, w, b = args[0], args[1], args[2]
        groups = args[8] if len(args) >= 9 else kwargs.get("groups", 1)
        return _flops_conv(x, out, w, groups=int(groups), include_bias=self._include_bias, has_bias=isinstance(b, torch.Tensor), effective_bwd=self._effective_bwd)

    def _h_native_layer_norm(self: Self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        if len(args) < 2: return 0.0
        x, normalized_shape = args[0], args[1]
        w = args[2] if len(args) >= 3 else None
        b = args[3] if len(args) >= 4 else None
        y = out[0] if isinstance(out, (tuple, list)) and out else out
        return _flops_layernorm(x, y, normalized_shape=(list(normalized_shape) if isinstance(normalized_shape, (tuple, list)) else ()), elementwise_affine=isinstance(w, torch.Tensor), has_bias=isinstance(b, torch.Tensor), effective_bwd=self._effective_bwd)

    def _h_layer_norm(self: Self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        if len(args) < 2: return 0.0
        x, normalized_shape = args[0], args[1]
        w = args[2] if len(args) >= 3 else None
        b = args[3] if len(args) >= 4 else None
        return _flops_layernorm(x, out, normalized_shape=(list(normalized_shape) if isinstance(normalized_shape, (tuple, list)) else ()), elementwise_affine=isinstance(w, torch.Tensor), has_bias=isinstance(b, torch.Tensor), effective_bwd=self._effective_bwd)

    def _h_dropout(self: Self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        if len(args) < 3 or not bool(args[2]): return 0.0
        return _flops_elementwise(out, coeff=2.0, effective_bwd=self._effective_bwd)

    def _h_softmax(self: Self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        if len(args) < 2: return 0.0
        x, dim = args[0], int(args[1])
        if not isinstance(x, torch.Tensor): return 0.0
        return _flops_softmax(x, out, dim=dim, effective_bwd=self._effective_bwd)

    def _h_sdpa(self: Self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        if len(args) < 3: return 0.0
        q, k, v = args[0], args[1], args[2]
        dropout_p = _float_safe(args[4], 0.0) if len(args) >= 5 else _float_safe(kwargs.get("dropout_p", 0.0), 0.0)
        if not isinstance(q, torch.Tensor): return 0.0
        return _flops_attention_qkv(q, k if isinstance(k, torch.Tensor) else None, v if isinstance(v, torch.Tensor) else None, effective_bwd=self._effective_bwd, dropout_p=float(dropout_p), training=bool(dropout_p > 0.0), include_softmax_scale_dropout=True)

    def _h_sdpa_like(self: Self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any) -> float:
        ts = _coerce_tensor_sequence(args, max_n=3)
        if not ts: return 0.0
        q = ts[0]
        k = ts[1] if len(ts) >= 2 else None
        v = ts[2] if len(ts) >= 3 else None
        dropout_p = _float_safe(kwargs.get("dropout_p", 0.0), 0.0)
        return _flops_attention_qkv(q, k, v, effective_bwd=self._effective_bwd, dropout_p=float(dropout_p), training=bool(dropout_p > 0.0), include_softmax_scale_dropout=True)

    def _h_custom(self: Self, args: Tuple[Any, ...], kwargs: Dict[str, Any], out: Any, name: str) -> float:
        ts = _coerce_tensor_sequence(args, max_n=4)
        for t in ts:
            if isinstance(t, torch.Tensor) and t.ndim == 4:
                return _flops_attention_qkv(t, ts[1] if len(ts) >= 2 else None, ts[2] if len(ts) >= 3 else None, effective_bwd=self._effective_bwd, dropout_p=0.0, training=False, include_softmax_scale_dropout=True)
        if len(ts) >= 2 and ts[0].ndim == 2 and ts[1].ndim == 2:
            a, b = ts[0], ts[1]
            if int(a.shape[1]) == int(b.shape[1]):
                return float(2.0 * int(a.shape[0]) * int(b.shape[0]) * int(a.shape[1]) * (1.0 + max(0.0, self._effective_bwd)))
        return _flops_elementwise(out, coeff=1.0, effective_bwd=self._effective_bwd)


class _GraphProfiler:
    def __init__(self: Self, *args: Any, include_bias: bool, effective_bwd: float, count_elementwise: bool) -> None:
        self._include_bias = bool(include_bias)
        self._effective_bwd = float(effective_bwd)
        self._count_elementwise = bool(count_elementwise)
        self._aten = torch.ops.aten
        
        handlers: Dict[Any, Tuple[Callable, str]] = {}
        def _reg(op: object | None, handler: Callable[..., float], tag: str) -> None:
            if op is not None: handlers[op] = (handler, tag)

        _reg(getattr(self._aten.mm, "default", None), self._mm, "MatMul")
        _reg(getattr(self._aten.bmm, "default", None), self._bmm, "MatMul")
        _reg(getattr(self._aten.matmul, "default", None), self._matmul, "MatMul")
        _reg(getattr(self._aten.addmm, "default", None), self._addmm, "Linear")
        if getattr(self._aten, "linear", None): _reg(getattr(self._aten.linear, "default", None), self._linear, "Linear")
        _reg(getattr(self._aten.convolution, "default", None), self._convolution, "Conv")
        _reg(getattr(self._aten.native_layer_norm, "default", None), self._native_layer_norm, "LayerNorm")
        _reg(getattr(self._aten.layer_norm, "default", None), self._layer_norm, "LayerNorm")
        if getattr(self._aten, "dropout", None): _reg(getattr(self._aten.dropout, "default", None), self._dropout, "Dropout")
        if getattr(self._aten, "_softmax", None): _reg(getattr(self._aten._softmax, "default", None), self._softmax, "Softmax")
        if getattr(self._aten, "softmax", None) and hasattr(self._aten.softmax, "int"):
            _reg(getattr(self._aten.softmax, "int", None), self._softmax, "Softmax")

        for name in ("scaled_dot_product_attention", "_scaled_dot_product_flash_attention", "_scaled_dot_product_flash_attention_for_cpu", "_scaled_dot_product_efficient_attention", "_scaled_dot_product_cudnn_attention", "_flash_attention_forward", "_efficient_attention_forward"):
            if (op := getattr(self._aten, name, None)) and hasattr(op, "default"):
                _reg(op.default, self._sdpa_like, "Attention")

        if self._count_elementwise:
            def _reg_ops(bases: Sequence[str], coeff: float, tag_prefix: str = "Elementwise") -> None:
                for base in bases:
                    for pobj in (getattr(self._aten, base, None), getattr(self._aten, base + "_", None)):
                        if not pobj: continue
                        for o in ("Tensor", "Scalar", "default", "self", "ScalarSelf", "ScalarOther", "Tensor_out", "Scalar_out", "self_out"):
                            if (f := getattr(pobj, o, None)):
                                _reg(f, lambda a, k, o, c=coeff: self._eltwise(o, c), f"{tag_prefix}.{base}")

            _reg_ops(("add", "sub", "mul", "div"), 1.0)
            _reg_ops(("addcmul", "addcdiv"), 3.0)
            unary_coeff = {"abs": 1.0, "neg": 1.0, "floor": 1.0, "ceil": 1.0, "round": 1.0, "trunc": 1.0, "frac": 1.0, "sign": 1.0, "reciprocal": 4.0, "sqrt": 4.0, "rsqrt": 6.0, "exp": 10.0, "exp2": 10.0, "expm1": 10.0, "log": 10.0, "log2": 10.0, "log10": 10.0, "log1p": 10.0, "sin": 12.0, "cos": 12.0, "tan": 12.0, "asin": 14.0, "acos": 14.0, "atan": 14.0, "sinh": 14.0, "cosh": 14.0, "tanh": 4.0, "asinh": 14.0, "acosh": 14.0, "atanh": 14.0, "erf": 14.0, "erfc": 14.0, "erfinv": 18.0, "sigmoid": 4.0, "relu": 1.0, "gelu": 8.0, "silu": 6.0}
            for name, c in unary_coeff.items(): _reg_ops((name,), float(c))
            binary_coeff = {"atan2": 14.0, "minimum": 1.0, "maximum": 1.0, "fmin": 1.0, "fmax": 1.0, "clamp": 2.0, "clamp_min": 1.0, "clamp_max": 1.0, "where": 1.0, "lerp": 3.0}
            for name, c in binary_coeff.items(): _reg_ops((name,), float(c))
            for pobj in (getattr(self._aten, "pow", None), getattr(self._aten, "pow_", None)):
                if not pobj: continue
                for o in ("Tensor_Tensor", "Tensor_Scalar", "Scalar", "default"):
                    if f := getattr(pobj, o, None): _reg(f, self._pow, "Elementwise.pow")
            if (fma_obj := getattr(self._aten, "fma", None)) and hasattr(fma_obj, "default"):
                _reg(fma_obj.default, lambda a, k, o: self._eltwise(o, 2.0), "Elementwise.fma")

        self._handlers = handlers

    def _as_tensor(self: Self, out: Any) -> Any:
        return out[0] if isinstance(out, (tuple, list)) and out else out

    def _out_numel(self: Self, out: Any) -> int:
        y = self._as_tensor(out)
        if y is None: return 0
        try:
            numel = getattr(y, "numel", None)
            return int(numel()) if callable(numel) else int(numel)
        except Exception: return 0

    def _eltwise(self: Self, out: Any, coeff: float) -> float:
        n = self._out_numel(out)
        return float(float(n) * float(coeff) * (1.0 + max(0.0, self._effective_bwd))) if n > 0 else 0.0

    def _pow_coeff(self: Self, exp: Any) -> float:
        try:
            if isinstance(exp, (int, float)):
                e = float(exp)
                if e == 0.5: return 4.0
                if e == -0.5: return 6.0
                if e.is_integer():
                    ei = int(e)
                    if 2 <= ei <= 4: return float(ei - 1)
                    if ei in (0, 1): return 1.0
        except Exception: pass
        return 12.0

    def _call(self: Self, target: Any, args: Any, kwargs: Dict[str, Any], out: Any) -> Tuple[float, str]:
        if (handler_info := self._handlers.get(target)) is not None:
            handler, tag = handler_info
            try: return float(handler(args, kwargs, out)), tag
            except Exception: return 0.0, tag
            
        name = str(target).lower()
        if any(k in name for k in ("triton", "flex", "transformer_engine", "xformers")):
            return self._custom(args, out, name=name), "Custom"
        return 0.0, "Other"

    def _pow(self: Self, args: Any, kwargs: Dict[str, Any], out: Any) -> float:
        exp = args[1] if isinstance(args, (tuple, list)) and len(args) >= 2 else kwargs.get("exponent", None)
        return self._eltwise(out, self._pow_coeff(exp))

    def _mm(self: Self, args: Any, kwargs: Dict[str, Any] = {}, out: Any = None) -> float:
        a, b = args[0], args[1]
        if not (_is_tensorlike(a) and _is_tensorlike(b)) or a.ndim != 2 or b.ndim != 2: return 0.0
        m, k = a.shape
        k2, n = b.shape
        if int(k) != int(k2): return 0.0
        return float(2.0 * int(m) * int(n) * int(k) * (1.0 + max(0.0, self._effective_bwd)))

    def _bmm(self: Self, args: Any, kwargs: Dict[str, Any] = {}, out: Any = None) -> float:
        a, b = args[0], args[1]
        if not (_is_tensorlike(a) and _is_tensorlike(b)) or a.ndim != 3 or b.ndim != 3: return 0.0
        batch, m, k = a.shape
        batch2, k2, n = b.shape
        if int(batch) != int(batch2) or int(k) != int(k2): return 0.0
        return float(2.0 * int(batch) * int(m) * int(n) * int(k) * (1.0 + max(0.0, self._effective_bwd)))

    def _matmul(self: Self, args: Any, kwargs: Dict[str, Any] = {}, out: Any = None) -> float:
        a, b = args[0], args[1]
        if not (_is_tensorlike(a) and _is_tensorlike(b)): return 0.0
        if a.ndim == 2 and b.ndim == 2: return self._mm(args)
        if a.ndim == 3 and b.ndim == 3: return self._bmm(args)
        return 0.0

    def _addmm(self: Self, args: Any, kwargs: Dict[str, Any] = {}, out: Any = None) -> float:
        if len(args) < 3: return 0.0
        input_, mat1, mat2 = args[0], args[1], args[2]
        if not (_is_tensorlike(mat1) and _is_tensorlike(mat2)) or mat1.ndim != 2 or mat2.ndim != 2: return 0.0
        m, k = mat1.shape
        k2, n = mat2.shape
        if int(k) != int(k2): return 0.0
        fwd = 2.0 * int(m) * int(n) * int(k) + (float(int(m) * int(n)) if self._include_bias and _is_tensorlike(input_) and int(getattr(input_, "numel", lambda: 0)()) > 0 else 0.0)
        return float(fwd * (1.0 + max(0.0, self._effective_bwd)))

    def _linear(self: Self, args: Any, kwargs: Dict[str, Any], out: Any) -> float:
        if len(args) < 2: return 0.0
        x, w = args[0], args[1]
        if not (_is_tensorlike(x) and _is_tensorlike(w)): return 0.0
        return _flops_linear(x if isinstance(x, torch.Tensor) else _to_tensor(x), self._as_tensor(out), w if isinstance(w, torch.Tensor) else _to_tensor(w), include_bias=self._include_bias, has_bias=_is_tensorlike(args[2] if len(args) >= 3 else None), effective_bwd=self._effective_bwd)

    def _convolution(self: Self, args: Any, kwargs: Dict[str, Any], out: Any) -> float:
        if len(args) < 3: return 0.0
        x, w, b = args[0], args[1], args[2]
        return _flops_conv(_to_tensor(x), self._as_tensor(out), _to_tensor(w), groups=int(args[8] if len(args) >= 9 else 1), include_bias=self._include_bias, has_bias=_is_tensorlike(b), effective_bwd=self._effective_bwd)

    def _native_layer_norm(self: Self, args: Any, kwargs: Dict[str, Any], out: Any) -> float:
        if len(args) < 2: return 0.0
        x, normalized_shape = _to_tensor(args[0]), args[1]
        w = args[2] if len(args) >= 3 else None
        b = args[3] if len(args) >= 4 else None
        return _flops_layernorm(x, self._as_tensor(out), normalized_shape=(list(normalized_shape) if isinstance(normalized_shape, (tuple, list)) else ()), elementwise_affine=_is_tensorlike(w), has_bias=_is_tensorlike(b), effective_bwd=self._effective_bwd)

    def _layer_norm(self: Self, args: Any, kwargs: Dict[str, Any], out: Any) -> float:
        if len(args) < 2: return 0.0
        x, normalized_shape = _to_tensor(args[0]), args[1]
        w = args[2] if len(args) >= 3 else None
        b = args[3] if len(args) >= 4 else None
        return _flops_layernorm(x, self._as_tensor(out), normalized_shape=(list(normalized_shape) if isinstance(normalized_shape, (tuple, list)) else ()), elementwise_affine=_is_tensorlike(w), has_bias=_is_tensorlike(b), effective_bwd=self._effective_bwd)

    def _dropout(self: Self, args: Any, kwargs: Dict[str, Any], out: Any) -> float:
        if len(args) < 3 or not bool(args[2]): return 0.0
        return self._eltwise(out, 2.0)

    def _softmax(self: Self, args: Any, kwargs: Dict[str, Any], out: Any) -> float:
        if len(args) < 2: return 0.0
        return _flops_softmax(_to_tensor(args[0]), self._as_tensor(out), dim=int(args[1]), effective_bwd=self._effective_bwd)

    def _sdpa_like(self: Self, args: Any, kwargs: Dict[str, Any], out: Any = None) -> float:
        q = args[0] if isinstance(args, (tuple, list)) and len(args) >= 1 else None
        k = args[1] if isinstance(args, (tuple, list)) and len(args) >= 2 else None
        v = args[2] if isinstance(args, (tuple, list)) and len(args) >= 3 else None
        if not _is_tensorlike(q): return 0.0
        dropout_p = float(kwargs.get("dropout_p", 0.0))
        
        match q:
            case _TensorShape():
                b, h, s, d = _infer_bhsd_shape(q.shape)
                klen = _infer_bhsd_shape(k.shape)[2] if isinstance(k, _TensorShape) and k.ndim == 4 else s
                return _flops_attention_generics(batch=b, num_heads=h, q_len=s, k_len=klen, head_dim=d, effective_bwd=self._effective_bwd, dropout_p=dropout_p, training=bool(dropout_p > 0.0), include_softmax_scale_dropout=True)
            case _:
                return _flops_attention_qkv(_to_tensor(q), _to_tensor(k) if _is_tensorlike(k) else None, _to_tensor(v) if _is_tensorlike(v) else None, effective_bwd=self._effective_bwd, dropout_p=dropout_p, training=bool(dropout_p > 0.0), include_softmax_scale_dropout=True)

    def _custom(self: Self, args: Any, out: Any, *extra: Any, name: str) -> float:
        ts = [a for a in args if _is_tensorlike(a)] if isinstance(args, (tuple, list)) else []
        for t in ts:
            if int(getattr(t, "ndim", 0)) == 4:
                q = t
                k = ts[1] if len(ts) >= 2 else None
                v = ts[2] if len(ts) >= 3 else None
                if isinstance(q, _TensorShape):
                    b, h, s, d = _infer_bhsd_shape(q.shape)
                    klen = _infer_bhsd_shape(k.shape)[2] if isinstance(k, _TensorShape) and k.ndim == 4 else s
                    return _flops_attention_generics(batch=b, num_heads=h, q_len=s, k_len=klen, head_dim=d, effective_bwd=self._effective_bwd, dropout_p=0.0, training=False, include_softmax_scale_dropout=True)
                if isinstance(q, torch.Tensor):
                    return _flops_attention_qkv(q, k if isinstance(k, torch.Tensor) else None, v if isinstance(v, torch.Tensor) else None, effective_bwd=self._effective_bwd, dropout_p=0.0, training=False, include_softmax_scale_dropout=True)
                    
        if len(ts) >= 2 and int(getattr(ts[0], "ndim", 0)) == 2 and int(getattr(ts[1], "ndim", 0)) == 2:
            a, b = ts[0], ts[1]
            m, k = a.shape
            k2, n = b.shape
            if int(k) == int(k2):
                return float(2.0 * int(m) * int(n) * int(k) * (1.0 + max(0.0, self._effective_bwd)))
        return self._eltwise(out, 1.0)

    def estimate(self: Self, gm: torch.fx.GraphModule) -> Tuple[float, Dict[str, float]]:
        total = 0.0
        by: Dict[str, float] = {}
        for node in gm.graph.nodes:
            if node.op != "call_function": continue
            flops, typ = self._call(node.target, _fx_resolve(node.args, gm), _fx_resolve(node.kwargs or {}, gm), _fx_resolve_node(node, gm))
            if flops > 0.0:
                total += float(flops)
                by[typ] = by.get(typ, 0.0) + float(flops)
        return float(total), by


class _FlopProfiler:
    _stack_var: contextvars.ContextVar[Tuple[_Acc, ...]] = contextvars.ContextVar("enn_flops_stack", default=())
    _nvtx_getter: Optional[Callable[[], float]] = None

    def _stack(self: Self) -> Tuple[_Acc, ...]:
        return self._stack_var.get()

    def _capture_torch(self: Self, display: bool = False) -> Any:
        try: from torch.profiler import profile as profile_fn
        except Exception: profile_fn = None
        return _TorchFlops(profile_fn, show=bool(display)) if profile_fn is not None else _TorchFlopsCompat(show=bool(display))

    def is_active(self: Self) -> bool:
        return len(self._stack()) > 0

    def activate(self: Self) -> None:
        self._stack_var.set(self._stack() + (_Acc(),))

    def deactivate(self: Self) -> None:
        if stack := self._stack(): self._stack_var.set(stack[:-1])

    def reset(self: Self) -> None:
        if stack := self._stack():
            stack[-1].total = 0.0
            stack[-1].by_type.clear()

    def pop(self: Self) -> Tuple[float, Dict[str, float]]:
        if not (stack := self._stack()): return (0.0, {})
        acc = stack[-1]
        total = float(acc.total)
        breakdown = {k: float(v) for k, v in acc.by_type.items()}
        acc.total = 0.0
        acc.by_type.clear()
        return (total, breakdown)

    def sum(self: Self, *args: Any, sort: bool = True) -> Tuple[float, Dict[str, float]]:
        if not (stack := self._stack()): return (0.0, {})
        acc = stack[-1]
        total = float(acc.total)
        if not sort: return total, {k: float(v) for k, v in acc.by_type.items()}
        return total, {k: float(v) for k, v in sorted(acc.by_type.items(), key=lambda kv: kv[1], reverse=True)}

    def add(self: Self, typ: str, value: float) -> None:
        fv = _float_safe(value, 0.0)
        if fv <= 0.0 or not (stack := self._stack()): return
        for acc in stack:
            acc.total += fv
            acc.by_type[typ] = acc.by_type.get(typ, 0.0) + fv

    def coerce_flops_nvtx(self: Self) -> None:
        if self._nvtx_getter is not None: return
        hook = os.getenv("ENN_NVTX_GETTER", "")
        if not hook:
            self._nvtx_getter = None
            return
        try: module_name, attr = hook.split(":", 1)
        except ValueError:
            self._nvtx_getter = None
            return
        getter: Optional[Callable[[], float]] = None
        try:
            if callable(candidate := getattr(__import__(module_name, fromlist=[attr]), attr)):
                getter = candidate
        except Exception as exc: _LOGGER.debug("Failed to import NVTX getter %s: %s", hook, exc)
        self._nvtx_getter = getter

    def new_flops_nvtx(self: Self, device: Optional[torch.device] = None) -> Any:
        self.coerce_flops_nvtx()
        if self._nvtx_getter is None: return contextlib.nullcontext()
        try:
            from .system import is_accelerator_available
            if not is_accelerator_available("cuda"): raise RuntimeError("CUDA not available")
            getattr(torch.cuda, "nvtx")
        except Exception: return contextlib.nullcontext()
        return _NvtxFlops(device, self._nvtx_getter)

    def start(self: Self, model: nn.Module, *args: Any, cfg: _ProfilerConfig) -> List[Any]:
        handles: List[Any] = []
        skip: set[int] = set()
        for module in model.modules():
            if id(module) in skip: continue
            hook = None
            match module:
                case _ if _is_te_module(module):
                    with contextlib.suppress(Exception): skip.update(id(c) for c in list(module.modules())[1:])
                    hook = module.register_forward_hook(partial(_register_te_module, profiler=self, cfg=cfg))
                case nn.Linear(): hook = module.register_forward_hook(partial(_register_linear, profiler=self, cfg=cfg))
                case nn.modules.conv._ConvNd(): hook = module.register_forward_hook(partial(_register_conv, profiler=self, cfg=cfg))
                case nn.LayerNorm(): hook = module.register_forward_hook(partial(_register_layernorm, profiler=self, cfg=cfg))
                case nn.Softmax() if cfg.count_softmax: hook = module.register_forward_hook(partial(_register_softmax, profiler=self, cfg=cfg))
                case nn.Dropout() if cfg.count_dropout: hook = module.register_forward_hook(partial(_register_dropout, profiler=self, cfg=cfg))
                case nn.Embedding() if cfg.count_embedding: hook = module.register_forward_hook(partial(_register_embedding, profiler=self, cfg=cfg))
                case _ if cfg.count_activations and isinstance(module, _ACT_CLASSES): hook = module.register_forward_hook(partial(_register_activation, profiler=self, cfg=cfg))
                case nn.MultiheadAttention(): hook = module.register_forward_hook(partial(_register_mha, profiler=self, cfg=cfg))
            if hook is not None: handles.append(hook)
        return handles

    def stop(self: Self, handles: Sequence[Any]) -> None:
        for h in handles:
            with contextlib.suppress(Exception): h.remove()

    def reduce(
        self: Self, device: Optional[torch.device], *args: Any,
        display: bool = False, use_torch_profiler: bool = True, use_nvtx: bool = True, dispatch_mode: Optional[Any] = None,
    ) -> Any:
        return _Flops(profiler=self, device=device, display=bool(display), use_torch_profiler=bool(use_torch_profiler), use_nvtx=bool(use_nvtx), dispatch_mode=dispatch_mode)

    def capture(
        self: Self, q: torch.Tensor, *args: Any, bwd_factor: float = 2.0, dropout_p: float = 0.0,
        training: bool = False, include_softmax_scale_dropout: bool = True,
    ) -> float:
        if not isinstance(q, torch.Tensor) or q.ndim < 4: return 0.0
        b, h, s, d = _bhsd_shape(q)
        total = _flops_attention_generics(batch=b, num_heads=h, q_len=s, k_len=s, head_dim=d, effective_bwd=float(bwd_factor), dropout_p=float(dropout_p), training=bool(training), include_softmax_scale_dropout=bool(include_softmax_scale_dropout))
        if total > 0.0: self.add("Attention", total)
        return float(total)


# =============================================================================
# Context Managers
# =============================================================================
class _NvtxFlops(contextlib.AbstractContextManager[Any]):
    def __init__(self: Self, dev: Optional[torch.device], getter: Callable[[], float]) -> None:
        self._dev = dev
        self._getter = getter
        self._base = 0.0

    def __enter__(self: Self) -> "_NvtxFlops":
        with contextlib.suppress(Exception):
            if self._dev is not None and getattr(self._dev, "type", "") == "cuda":
                from .system import sync_accelerator
                sync_accelerator(self._dev)
        try: self._base = float(self._getter())
        except Exception: self._base = 0.0
        return self

    def __exit__(self: Self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> bool:
        return False

    def get_total_flops(self: Self) -> float:
        try: return max(0.0, float(self._getter()) - float(self._base))
        except Exception: return 0.0


class _TorchFlops(contextlib.AbstractContextManager[Any]):
    def __init__(self: Self, profile_fn: Callable[..., Any], show: bool) -> None:
        self._profile_fn = profile_fn
        self._show = bool(show)
        self._prof: Any = None

    def __enter__(self: Self) -> "_TorchFlops":
        self._prof = self._profile_fn(with_flops=True, record_shapes=False)
        self._prof.__enter__()
        return self

    def __exit__(self: Self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> bool:
        if self._prof is not None:
            self._prof.__exit__(exc_type, exc, tb)
            if self._show:
                with contextlib.suppress(Exception):
                    _LOGGER.info("%s", self._prof.key_averages().table(sort_by="flops"))
        return False

    def get_total_flops(self: Self) -> float:
        if self._prof is None: return 0.0
        try: return float(sum(getattr(e, "flops", 0.0) for e in self._prof.key_averages()))
        except Exception: return 0.0


class _TorchFlopsCompat(contextlib.AbstractContextManager[Any]):
    def __init__(self: Self, show: bool) -> None:
        try:
            from torch.utils.flop_counter import FlopCounterMode as TorchMode
            self._impl = TorchMode(display=show)
        except Exception: self._impl = None

    def __enter__(self: Self) -> "_TorchFlopsCompat":
        if self._impl is not None: self._impl.__enter__()
        return self

    def __exit__(self: Self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> bool:
        if self._impl is not None: self._impl.__exit__(exc_type, exc, tb)
        return False

    def get_total_flops(self: Self) -> float:
        if self._impl is None: return 0.0
        try: return float(self._impl.get_total_flops())
        except Exception: return 0.0


class _Flops(contextlib.AbstractContextManager[Any]):
    def __init__(
        self: Self, *args: Any, profiler: "_FlopProfiler", device: Optional[torch.device],
        display: bool, use_torch_profiler: bool, use_nvtx: bool, dispatch_mode: Optional[Any],
    ) -> None:
        self._profiler = profiler
        self._device = device
        self._display = bool(display)
        self._use_torch_profiler = bool(use_torch_profiler)
        self._use_nvtx = bool(use_nvtx)
        self._dispatch_mode = dispatch_mode
        self.manual_total = 0.0
        self.manual_breakdown: Dict[str, float] = {}
        self.torch_total = 0.0
        self.nvtx_total = 0.0
        self.total = 0.0
        self._torch_scope: Any = None
        self._nvtx_scope: Any = None
        self._dispatch_scope: Any = None
        self._outer_active = False

    def __enter__(self: Self) -> "_Flops":
        self._outer_active = self._profiler.is_active()
        self._profiler.activate()
        self._profiler.reset()
        if not self._outer_active:
            if self._dispatch_mode is not None:
                self._dispatch_scope = self._dispatch_mode
                with contextlib.suppress(Exception): self._dispatch_scope.__enter__()
            if self._use_torch_profiler:
                self._torch_scope = self._profiler._capture_torch(self._display)
                if self._torch_scope is not None: self._torch_scope.__enter__()
            if self._use_nvtx:
                self._nvtx_scope = self._profiler.new_flops_nvtx(self._device)
                if self._nvtx_scope is not None: self._nvtx_scope.__enter__()
        return self

    def __exit__(self: Self, exc_type: Any, exc: Any, tb: Any) -> bool:
        manual, breakdown = self._profiler.pop()
        self.manual_total, self.manual_breakdown = float(manual), breakdown
        
        if not self._outer_active:
            if self._nvtx_scope is not None:
                self._nvtx_scope.__exit__(exc_type, exc, tb)
                try: self.nvtx_total = float(self._nvtx_scope.get_total_flops())
                except Exception: self.nvtx_total = 0.0
            if self._torch_scope is not None:
                self._torch_scope.__exit__(exc_type, exc, tb)
                try: self.torch_total = float(self._torch_scope.get_total_flops())
                except Exception: self.torch_total = 0.0
            if self._dispatch_scope is not None:
                with contextlib.suppress(Exception): self._dispatch_scope.__exit__(exc_type, exc, tb)
                
        self.total = float(max(max(self.torch_total, self.nvtx_total), self.manual_total, 0.0))
        self._profiler.deactivate()
        return False

    def get_total_flops(self: Self) -> float:
        return float(self._profiler.sum(sort=False)[0]) if self._profiler.is_active() else float(self.total)

    def get_manual_breakdown(self: Self) -> Dict[str, float]:
        return dict(self._profiler.sum(sort=False)[1]) if self._profiler.is_active() else dict(self.manual_breakdown)

    def to_dict(self: Self) -> Dict[str, float]:
        return {"manual_total": float(self.manual_total), "torch_total": float(self.torch_total), "nvtx_total": float(self.nvtx_total), "total": float(self.total)}

    def verbose(self: Self, top_k: int = 12) -> str:
        lines = [f"total FLOPs: manual={self.manual_total:.3e}, torch={self.torch_total:.3e}, nvtx={self.nvtx_total:.3e}, total={self.total:.3e}"]
        if self.manual_breakdown:
            lines.append(f"manual breakdown (top {top_k}):")
            for name, value in sorted(self.manual_breakdown.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
                lines.append(f"  - {name}: {value:.3e}")
        return "\n".join(lines)


class _StaticFlops(contextlib.AbstractContextManager[Any]):
    def __init__(self: Self, total: float, breakdown: Dict[str, float]) -> None:
        self.manual_total = float(total)
        self.manual_breakdown = dict(breakdown)
        self.torch_total, self.nvtx_total = 0.0, 0.0
        self.total = float(total)

    def __enter__(self: Self) -> "_StaticFlops": return self
    def __exit__(self: Self, exc_type: Any, exc: Any, tb: Any) -> bool: return False
    def get_total_flops(self: Self) -> float: return float(self.total)
    def get_manual_breakdown(self: Self) -> Dict[str, float]: return dict(self.manual_breakdown)
    def to_dict(self: Self) -> Dict[str, float]: return {"manual_total": float(self.manual_total), "torch_total": 0.0, "nvtx_total": 0.0, "total": float(self.total)}

    def verbose(self: Self, top_k: int = 12) -> str:
        lines = [f"total FLOPs (static): {self.total:.3e}"]
        if self.manual_breakdown:
            lines.append(f"breakdown (top {top_k}):")
            for name, value in sorted(self.manual_breakdown.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
                lines.append(f"  - {name}: {value:.3e}")
        return "\n".join(lines)


class _DynamicFlops(contextlib.AbstractContextManager[Any]):
    def __init__(
        self: Self, inner: Any, *args: Any, static_total: float, static_breakdown: Dict[str, float],
        cache_slot: Optional[Dict[Any, Tuple[float, Dict[str, float]]]] = None, cache_key: Any = None,
    ) -> None:
        self._inner = inner
        self._static_total = float(static_total)
        self._static_breakdown = dict(static_breakdown)
        self._cache_slot = cache_slot
        self._cache_key = cache_key
        self.manual_total, self.torch_total, self.nvtx_total, self.total = 0.0, 0.0, 0.0, 0.0
        self.manual_breakdown: Dict[str, float] = {}

    def __enter__(self: Self) -> "_DynamicFlops":
        self._inner.__enter__()
        return self

    def __exit__(self: Self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self._inner.__exit__(exc_type, exc, tb)
        self.manual_total = float(getattr(self._inner, "manual_total", 0.0))
        self.manual_breakdown = dict(getattr(self._inner, "manual_breakdown", {}) or {})
        self.torch_total = float(getattr(self._inner, "torch_total", 0.0))
        self.nvtx_total = float(getattr(self._inner, "nvtx_total", 0.0))
        self.total = float(getattr(self._inner, "total", 0.0))
        
        if self.total > 0.0 and self._cache_slot is not None and self._cache_key is not None:
            self._cache_slot[self._cache_key] = (float(self.total), dict(self.manual_breakdown))
        elif self.total <= 0.0 and self._static_total > 0.0:
            self.manual_total, self.total = float(self._static_total), float(self._static_total)
            self.manual_breakdown = dict(self._static_breakdown)
        return False

    def get_total_flops(self: Self) -> float:
        try: return float(self._inner.get_total_flops())
        except Exception: return float(self.total)

    def get_manual_breakdown(self: Self) -> Dict[str, float]:
        try: return dict(self._inner.get_manual_breakdown())
        except Exception: return dict(self.manual_breakdown)

    def to_dict(self: Self) -> Dict[str, float]:
        return {"manual_total": float(self.manual_total), "torch_total": float(self.torch_total), "nvtx_total": float(self.nvtx_total), "total": float(self.total)}

    def verbose(self: Self, top_k: int = 12) -> str:
        try: return str(self._inner.verbose(top_k=top_k))
        except Exception: return _StaticFlops(self.total, self.manual_breakdown).verbose(top_k=top_k)


# =============================================================================
# Public API
# =============================================================================
def get_torch_profiler(
    *args: Any, enabled: object, tag: object, device: torch.device, out_dir: object, rank: int = 0, **kwargs: Any,
) -> object:
    if not bool(enabled): return None
    try: import torch.profiler as tp
    except Exception: return None
    
    dev = device if isinstance(device, torch.device) else torch.device(device)
    activities = [tp.ProfilerActivity.CPU]
    
    match dev.type:
        case "cuda":
            with contextlib.suppress(Exception): activities.append(tp.ProfilerActivity.CUDA)
        case "xpu":
            with contextlib.suppress(Exception): activities.append(getattr(tp.ProfilerActivity, "XPU"))
        case "mps":
            with contextlib.suppress(Exception):
                if (mps_act := getattr(tp.ProfilerActivity, "MPS", None)) is not None: activities.append(mps_act)

    wait = max(0, int(env_int("ENN_TORCH_PROFILE_WAIT", 0)))
    warmup = max(0, int(env_int("ENN_TORCH_PROFILE_WARMUP", 2)))
    active = max(1, int(env_int("ENN_TORCH_PROFILE_ACTIVE", env_int("ENN_TORCH_PROFILE_STEPS", 8))))
    repeat = max(1, int(env_int("ENN_TORCH_PROFILE_REPEAT", 1)))
    
    out_dir_s = os.path.abspath(str(os.fspath(out_dir) if out_dir else os.path.join(os.getcwd(), "torch_profiler")))
    with contextlib.suppress(Exception): os.makedirs(out_dir_s, exist_ok=True)
        
    try:
        prof = tp.profile(
            activities=activities,
            schedule=tp.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=tp.tensorboard_trace_handler(out_dir_s, worker_name=f"{str(tag)}-rank{int(rank)}"),
            record_shapes=bool(env_bool("ENN_TORCH_PROFILE_RECORD_SHAPES", False)),
            profile_memory=bool(env_bool("ENN_TORCH_PROFILE_PROFILE_MEMORY", True)),
            with_stack=bool(env_bool("ENN_TORCH_PROFILE_WITH_STACK", False)),
            with_flops=bool(env_bool("ENN_TORCH_PROFILE_WITH_FLOPS", False)),
        )
        setattr(prof, "_enn_row_limit", max(5, int(env_int("ENN_TORCH_PROFILE_TOPK", 40))))
        setattr(prof, "_enn_group_by_shape", bool(env_bool("ENN_TORCH_PROFILE_GROUP_BY_SHAPE", False)))
        setattr(prof, "_enn_out_dir", str(out_dir_s))
        setattr(prof, "_enn_tag", str(tag))
        return prof
    except Exception: return None


def log_profiler_summary(prof: object, *args: Any, device: torch.device, logger: logging.Logger, header: object, **kwargs: Any) -> None:
    if prof is None: return
    try:
        ka = prof.key_averages(group_by_input_shape=bool(getattr(prof, "_enn_group_by_shape", False)))
    except Exception:
        with contextlib.suppress(Exception): ka = prof.key_averages()
        if "ka" not in locals(): return
        
    table = None
    for sk in ("self_cuda_time_total", "self_xpu_time_total", "self_cpu_time_total"):
        with contextlib.suppress(Exception):
            if table := ka.table(sort_by=str(sk), row_limit=int(getattr(prof, "_enn_row_limit", 40) or 40)): break
            
    if table:
        logger.info("[torch.profiler] %s (trace dir: %s, tag: %s)\n%s", str(header), str(getattr(prof, "_enn_out_dir", "")), str(getattr(prof, "_enn_tag", header)), str(table))


class FlopCounter:
    def __init__(
        self: Self, model: nn.Module, *args: Any, mode: str = "train", device: Optional[torch.device] = None,
        include_bias: bool = True, bwd_factor: Optional[float] = None, backend: str = "auto",
        estimate_bwd: bool = True, count_activations: bool = True, count_norms: bool = True,
        count_softmax: bool = True, count_dropout: bool = True, count_embedding: bool = True,
        count_elementwise: bool = True, static_fallback_on_zero: bool = True,
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
        self._fx_estimator: Optional[_GraphProfiler] = None
        self._effective_backend = self._backend.lower()

    def _effective_bwd(self: Self) -> float:
        eff = 2.0 if self._mode == "train" else 0.0
        if self._bwd_factor is not None: eff = _float_safe(self._bwd_factor, eff)
        return float(eff if self._estimate_bwd else 0.0)

    def _is_compiled(self: Self) -> bool:
        try:
            from torch._dynamo.eval_frame import OptimizedModule
            return isinstance(self._model, OptimizedModule)
        except Exception: return hasattr(self._model, "_orig_mod")

    def _orig_model(self: Self) -> nn.Module:
        return getattr(self._model, "_orig_mod", self._model)

    def __enter__(self: Self) -> "FlopCounter":
        cfg = _ProfilerConfig(
            include_bias=self._include_bias, effective_bwd=self._effective_bwd(),
            count_activations=self._count_activations, count_norms=self._count_norms,
            count_softmax=self._count_softmax, count_dropout=self._count_dropout, count_embedding=self._count_embedding,
        )
        backend = self._backend.lower()
        if backend == "auto":
            backend = "dynamo" if self._is_compiled() else ("dispatch" if TorchDispatchMode is not None else "hooks")
            
        self._effective_backend = backend
        if backend == "hooks":
            self._handles = FLOP_PROFILER.start(self._model, cfg=cfg)
            self._hook_count = len(self._handles)
        else:
            self._handles, self._hook_count = [], 0
            
        self._active = True
        return self

    def __exit__(self: Self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._active:
            if self._handles: FLOP_PROFILER.stop(self._handles)
            self._handles, self._active = [], False
        return False

    def _sig_key(self: Self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        return (tuple(_flop_sig_key_of(a) for a in args), _flop_sig_key_of(kwargs))

    @property
    def device(self: Self) -> Optional[torch.device]:
        return self._device

    @property
    def hook_count(self: Self) -> int:
        return int(self._hook_count)

    def prepare(self: Self, *example_args: Any, **example_kwargs: Any) -> Tuple[float, Dict[str, float]]:
        key = self._sig_key(example_args, example_kwargs)
        if key in self._static_cache: return self._static_cache[key]
        
        self._fx_estimator = _GraphProfiler(include_bias=self._include_bias, effective_bwd=self._effective_bwd(), count_elementwise=self._count_elementwise)
        if (gm := _export_graph(self._orig_model(), example_args, example_kwargs)) is None:
            self._static_cache[key] = (0.0, {})
            return (0.0, {})
            
        _forward_shape(gm, example_args, example_kwargs)
        total, breakdown = self._fx_estimator.estimate(gm)
        self._static_cache[key] = (float(total), dict(breakdown))
        return self._static_cache[key]

    def step(
        self: Self, *args: Any, display: bool = False,
        example_args: Optional[Tuple[Any, ...]] = None, example_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if not self._active: raise RuntimeError("FlopCounter is not active. Use `with FlopCounter(...)` before measuring FLOPs.")
        
        backend = self._effective_backend.lower()
        ex_args, ex_kwargs = example_args or (), example_kwargs or {}
        key = self._sig_key(ex_args, ex_kwargs) if (example_args is not None or example_kwargs is not None) else None
        
        cached_total, cached_breakdown = self._runtime_cache.get(key, (0.0, {})) if key is not None else (0.0, {})
        static_total, static_breakdown = 0.0, {}
        
        if key is not None and cached_total <= 0.0 and backend in ("dynamo", "hooks", "dispatch"):
            static_total, static_breakdown = self.prepare(*ex_args, **ex_kwargs)

        def _reduce_and_wrap(dispatch_mode: TorchDispatchMode | None = None, use_tp: bool = True) -> contextlib.AbstractContextManager[Any]:
            inner = FLOP_PROFILER.reduce(self._device, display=display, use_torch_profiler=use_tp, use_nvtx=True, dispatch_mode=dispatch_mode)
            if self._static_fallback_on_zero and key and (static_total > 0.0 or cached_total > 0.0):
                ft = float(cached_total) if cached_total > 0.0 else float(static_total)
                fb = dict(cached_breakdown) if cached_total > 0.0 else dict(static_breakdown)
                return _DynamicFlops(inner, static_total=ft, static_breakdown=fb, cache_slot=self._runtime_cache, cache_key=key)
            return inner

        if backend == "dynamo": return _StaticFlops(static_total, static_breakdown) if key and static_total > 0.0 else _reduce_and_wrap()
        if backend == "dispatch" and TorchDispatchMode: return _reduce_and_wrap(_OpFlopDispatchMode(FLOP_PROFILER, include_bias=self._include_bias, effective_bwd=self._effective_bwd(), count_elementwise=self._count_elementwise), use_tp=False)
        if backend == "hooks" or (backend == "dispatch" and not TorchDispatchMode): return _reduce_and_wrap()
        return FLOP_PROFILER.reduce(self._device, display=display, use_torch_profiler=True, use_nvtx=True, dispatch_mode=None)


FLOP_PROFILER = _FlopProfiler()
