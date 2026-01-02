# -*- coding: utf-8 -*-
from __future__ import annotations

import inspect
import warnings
from typing import Any, Optional, Tuple

import torch
import torch._dynamo
from torch import nn

from ..core.casting import env_str
from ..core.graph import torch_compiler_disable
from ..core.profiler import FLOP_PROFILER, capture
from ..core.system import get_device, get_dpa_backends, get_runtime_config

try:
    import triton
    import triton.language as tl
    _HAS_TRITON_LIB = True
except Exception:
    _HAS_TRITON_LIB = False

    class _TritonStub:
        def jit(self, fn=None, **kwargs):
            if fn is None:
                return (lambda f: f)
            return fn

        @staticmethod
        def cdiv(a: int, b: int) -> int:
            a_i = int(a)
            b_i = int(b)
            return (a_i + b_i - 1) // max(1, b_i)

    class _TLStub:
        constexpr = object()

    triton = _TritonStub()
    tl = _TLStub()

_HAS_TRITON_MSR = bool(_HAS_TRITON_LIB and torch.cuda.is_available())
_HAS_TE: bool

te = None
_should_import_te = True

if not torch.cuda.is_available():
    _should_import_te = False
else:
    try:
        device = get_device()
    except Exception:
        device = torch.device("cuda", 0)
    if getattr(device, "type", "cpu") != "cuda":
        _should_import_te = False

if _should_import_te:
    try:
        import transformer_engine.pytorch as te

        _HAS_TE = True
    except Exception:
        te = None
        _HAS_TE = False
else:
    _HAS_TE = False


def _flatten_attn_mask(
    mask: torch.Tensor,
    *args: Any,
    device: torch.device,
    B: int,
    H: int,
    L: int,
    S: int,
) -> tuple[torch.Tensor, int, int, int]:
    if mask.dim() == 0:
        m = mask.to(device=device).view(1, 1, 1, 1)
        m = m.expand(1, 1, 1, int(S))
        return m, 1, 1, 1
    if mask.dim() == 1:
        if int(mask.shape[0]) != int(S):
            raise RuntimeError(
                f"attn_mask shape {tuple(mask.shape)} incompatible with key length S={int(S)}"
            )
        m = mask.to(device=device).view(1, 1, 1, int(S))
        return m, 1, 1, 1
    if mask.dim() == 2:
        a, b = int(mask.shape[0]), int(mask.shape[1])
        if b != int(S):
            raise RuntimeError(
                f"attn_mask trailing dim {b} does not match expected S={int(S)}"
            )
        if a == int(L):
            m = mask.to(device=device).view(1, 1, int(L), int(S))
            return m, 1, 1, int(L)
        if a == 1:
            m = mask.to(device=device).view(1, 1, 1, int(S))
            return m, 1, 1, 1
        if a == int(B):
            m = mask.to(device=device).view(int(B), 1, 1, int(S))
            return m, int(B), 1, 1
        raise RuntimeError(
            f"unsupported 2D attn_mask shape {tuple(mask.shape)} for (B={int(B)}, L={int(L)}, S={int(S)})"
        )
    if mask.dim() == 3:
        a, b, c = int(mask.shape[0]), int(mask.shape[1]), int(mask.shape[2])
        if c != int(S):
            raise RuntimeError(
                f"attn_mask trailing dim {c} does not match expected S={int(S)}"
            )
        if (a == int(B)) and (b == int(L)):
            m = mask.to(device=device).view(int(B), 1, int(L), int(S))
            return m, int(B), 1, int(L)
        if (a == int(B)) and (b == 1):
            # (B,1,S) -> per-batch key-only
            m = mask.to(device=device).view(int(B), 1, 1, int(S))
            return m, int(B), 1, 1
        if (a == int(H)) and (b == int(L)):
            # (H,L,S) -> head-specific, broadcast batch
            m = mask.to(device=device).view(1, int(H), int(L), int(S))
            return m, 1, int(H), int(L)
        if (a == int(B)) and (b == int(H)):
            # (B,H,S) -> per-(B,H) key-only
            m = mask.to(device=device).view(int(B), int(H), 1, int(S))
            return m, int(B), int(H), 1
        raise RuntimeError(
            f"unsupported 3D attn_mask shape {tuple(mask.shape)} for (B={int(B)}, H={int(H)}, L={int(L)}, S={int(S)})"
        )
    if mask.dim() == 4:
        b0, h0, l0, s0 = map(int, mask.shape)
        if s0 != int(S):
            raise RuntimeError(
                f"attn_mask trailing dim {s0} does not match expected S={int(S)}"
            )
        if b0 not in (1, int(B)):
            raise RuntimeError(f"attn_mask batch dim {b0} incompatible with B={int(B)}")
        if h0 not in (1, int(H)):
            raise RuntimeError(f"attn_mask head dim {h0} incompatible with H={int(H)}")
        if l0 not in (1, int(L)):
            raise RuntimeError(
                f"attn_mask query dim {l0} incompatible with L={int(L)} (broadcast 1 or L allowed)"
            )
        m = mask.to(device=device)
        return m, b0, h0, l0
    raise RuntimeError(f"attn_mask rank {int(mask.dim())} not supported")


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
        if batch_first:
            if query.dim() < 3 or key.dim() < 3:
                return
            B = int(query.shape[0])
            Lq = int(query.shape[1])
            Sk = int(key.shape[1])
        else:
            if query.dim() < 3 or key.dim() < 3:
                return
            Lq = int(query.shape[0])
            B = int(query.shape[1])
            Sk = int(key.shape[0])
        H = int(num_heads)
        E = int(embed_dim)
        if B <= 0 or Lq <= 0 or Sk <= 0 or H <= 0 or E <= 0:
            return
        if E % H != 0:
            return
        Dh = E // H
        core = 2.0 * B * H * Lq * Dh * Sk + 2.0 * B * H * Lq * Sk * Dh
        proj = 0.0
        if include_projections:
            q_tokens = float(B * Lq)
            k_tokens = float(B * Sk)
            v_tokens = float(B * Sk)
            out_tokens = float(B * Lq)
            proj = 2.0 * q_tokens * E * E
            proj += 2.0 * k_tokens * E * E
            proj += 2.0 * v_tokens * E * E
            proj += 2.0 * out_tokens * E * E
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
        no_avg.pop("average_attn_weights", None)
        kw_variants = (call_kwargs, no_avg)
    if is_causal is not None:
        for kw in kw_variants:
            try:
                return mha(q, k, v, is_causal=is_causal, **kw)
            except TypeError:
                pass
    for kw in kw_variants:
        try:
            return mha(q, k, v, **kw)
        except TypeError:
            pass
    return mha(q, k, v, **call_kwargs)


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


@triton.jit
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
    pid_d  = tl.program_id(1)
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
        raise ValueError(f"reshape_for_mha expects a 3D tensor (B,N,E), got shape={tuple(x.shape)}")
    return (
        x.reshape(int(batch), -1, int(heads), int(head_dim))
        .transpose(1, 2)
        .contiguous()
    )


class DotProductAttention(nn.Module):
    def __init__(
        self,
        num_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        te_first: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.nh = int(num_heads) if num_heads is not None else None
        self.hd = int(head_dim) if head_dim is not None else None
        cfg = get_runtime_config()
        self.te_first = bool(cfg.te_first) if te_first is None else bool(te_first)
        ok, te_mod = self._is_nvidia_te_available()
        self._te_ok = bool(
            ok
            and torch.cuda.is_available()
            and _is_nvidia_te_supported()
            and (self.nh is not None)
            and (self.hd is not None)
        )
        self._force_pt: bool = False
        self._te_attn: Any = None
        self._te_forward_signature: inspect.Signature | None = None
        self._te_mask_param: str | None = None
        self._te_mask_type_param: str | None = None
        self._te_core_bias_param: str | None = None
        self._te_core_bias_type_param: str | None = None
        self._te_supports_mask = False
        self._te_supports_mask_type = False
        self._te_supports_core_bias = False
        self._te_supports_core_bias_type = False
        self._te_supports_attention_dropout = False
        self._te_supports_is_causal = False
        self._te_supports_training = False
        if self._te_ok:
            self._te = te_mod
            try:
                self._te_attn = te_mod.DotProductAttention(
                    num_attention_heads=self.nh,
                    kv_channels=self.hd,
                    qkv_format="bshd",
                    attention_dropout=0.0,
                )
            except Exception:
                self._te_attn = None
                self._force_pt = True
            if self._te_attn is not None:
                _forward = getattr(
                    self._te_attn,
                    "forward",
                    getattr(self._te_attn, "__call__", None),
                )
                if _forward is not None:
                    try:
                        self._te_forward_signature = inspect.signature(_forward)
                    except (TypeError, ValueError):
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
                if "core_attention_bias" in params:
                    self._te_core_bias_param = "core_attention_bias"
                if "core_attention_bias_type" in params:
                    self._te_core_bias_type_param = "core_attention_bias_type"
                self._te_supports_mask = self._te_mask_param is not None
                self._te_supports_mask_type = self._te_mask_type_param is not None
                self._te_supports_core_bias = self._te_core_bias_param is not None
                self._te_supports_core_bias_type = (
                    self._te_core_bias_type_param is not None
                )
                self._te_supports_attention_dropout = "attention_dropout" in params
                self._te_supports_is_causal = "is_causal" in params
                self._te_supports_training = "training" in params

    @staticmethod
    def _is_nvidia_te_available() -> Any:
        try:
            import transformer_engine.pytorch as te_mod
            return (True, te_mod)
        except Exception:
            return (False, None)

    def forward(
        self,
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
                f"DotProductAttention expects q/k/v to be 4D (B,H,L,D), got "
                f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
            )
        Bq, Hq, Lq, Dq = q.shape
        Bk, Hk, Lk, Dk = k.shape
        Bv, Hv, Lv, Dv = v.shape
        if Bq != Bk or Hq != Hk or Bq != Bv or Hq != Hv:
            raise ValueError(
                "DotProductAttention expects matching batch and head dims for q/k/v, "
                f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
            )
        if Lk != Lv:
            raise ValueError(
                "DotProductAttention expects matching sequence length for k and v, "
                f"got k={tuple(k.shape)}, v={tuple(v.shape)}"
            )
        if Dq != Dk:
            raise ValueError(
                "DotProductAttention expects matching embedding dims for q and k, "
                f"got q={tuple(q.shape)}, k={tuple(k.shape)}"
            )
        if Dk != Dv:
            raise ValueError(
                "DotProductAttention expects matching embedding dims for k and v, "
                f"got k={tuple(k.shape)}, v={tuple(v.shape)}"
            )
        q = self._negotiate_dtype(q)
        k = self._negotiate_dtype(k)
        v = self._negotiate_dtype(v)
        q_bshd = q.contiguous()
        k_bshd = k.contiguous()
        v_bshd = v.contiguous()
        if _is_bshd_contiguous(q_bshd) and _is_bshd_contiguous(k_bshd):
            try:
                capture(
                    q_bshd,
                    bwd_factor=2.0 if training else 0.0,
                    dropout_p=float(dropout_p),
                    training=training,
                )
            except Exception:
                pass
        dropout_val = float(dropout_p) if training else 0.0
        B, H, L, D = q_bshd.shape
        S = k_bshd.shape[2]
        mask_bool: torch.Tensor | None = None
        bias_float: torch.Tensor | None = None
        if attn_mask is not None:
            m = attn_mask
            if m.dtype == torch.bool:
                mask_bool = m
            elif torch.is_floating_point(m):
                bias_float = m
            elif m.dtype in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                try:
                    sampled = m if m.numel() <= 4096 else m.reshape(-1)[:4096]
                    uniq = torch.unique(sampled)
                except Exception:
                    uniq = torch.tensor([], device=m.device, dtype=m.dtype)
                if uniq.numel() <= 2 and bool(((uniq == 0) | (uniq == 1)).all().item()):
                    mask_bool = (m != 0)
                else:
                    bias_float = m.to(dtype=q_bshd.dtype)
            else:
                mask_bool = m.to(torch.bool)
        mb = mh = mL = 0
        bb = bh = bL = 0
        if mask_bool is not None:
            mask_bool = mask_bool.to(device=q_bshd.device, dtype=torch.bool, non_blocking=True)
            mask_bool, mb, mh, mL = _flatten_attn_mask(
                mask_bool,
                device=q_bshd.device,
                B=B,
                H=H,
                L=L,
                S=S,
            )
            if int(mh) == 1 and int(mL) == int(L):
                try:
                    if int(mask_bool.stride(-2)) == 0:
                        mask_bool = mask_bool[..., :1, :]
                        mL = 1
                except Exception:
                    pass
        if bias_float is not None:
            bias_float = bias_float.to(device=q_bshd.device, dtype=q_bshd.dtype, non_blocking=True)
            bias_float, bb, bh, bL = _flatten_attn_mask(
                bias_float,
                device=q_bshd.device,
                B=B,
                H=H,
                L=L,
                S=S,
            )
        try:
            is_compiling = torch.compiler.is_compiling()
        except Exception:
            is_compiling = False
        use_te = (
            self._te_ok
            and self.te_first
            and not self._force_pt
            and (self._te_attn is not None)
            and not is_compiling
            and q_bshd.is_cuda
            and q_bshd.dtype in (torch.float16, torch.bfloat16)
        )
        te_mask: torch.Tensor | None = None
        te_mask_type: str | None = None
        if use_te:
            if bias_float is not None:
                use_te = False
            elif mask_bool is None:
                te_mask_type = "causal" if bool(is_causal) else "no_mask"
            else:
                is_padding_like = (int(mh) == 1) and (int(mL) == 1)
                if not is_padding_like:
                    use_te = False
                else:
                    te_mask = mask_bool
                    if int(mb) != int(B):
                        te_mask = te_mask.expand(int(B), 1, 1, int(S))
                    te_mask = te_mask.contiguous()
                    te_mask_type = "padding_causal" if bool(is_causal) else "padding"
            if use_te and (te_mask is not None):
                if not self._te_supports_mask:
                    use_te = False
                elif te_mask_type is not None and te_mask_type.startswith("padding"):
                    if not (self._te_supports_mask_type and (self._te_mask_type_param is not None)):
                        use_te = False
            if use_te and (te_mask is None) and bool(is_causal):
                if not (self._te_supports_mask_type or self._te_supports_is_causal):
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
            if te_mask is not None and self._te_supports_mask and self._te_mask_param:
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
                    S_ = k_bshd.shape[2]
                    flops = 2.0 * B_ * H_ * L_ * D_ * S_ + 2.0 * B_ * H_ * L_ * S_ * D_
                    FLOP_PROFILER.add("DotProductAttention", float(flops))
                except Exception:
                    pass
                return out_te
        final_mask: torch.Tensor | None = None
        sdpa_is_causal = bool(is_causal)
        if bias_float is None:
            if mask_bool is None:
                final_mask = None
            else:
                final_mask = (~mask_bool)
                sdpa_is_causal = False
        else:
            if mask_bool is None:
                final_mask = bias_float
            else:
                finfo = torch.finfo(q_bshd.dtype)
                zero = torch.zeros((), dtype=q_bshd.dtype, device=q_bshd.device)
                neg_inf = torch.full((), finfo.min, dtype=q_bshd.dtype, device=q_bshd.device)
                mask_bias = torch.where(mask_bool, neg_inf, zero)
                final_mask = (mask_bias + bias_float).contiguous()
            sdpa_is_causal = False
        sdpa_kwargs = {
            "attn_mask": final_mask,
            "dropout_p": dropout_val,
            "is_causal": bool(sdpa_is_causal),
        }
        q_bhsd = q_bshd.contiguous()
        k_bhsd = k_bshd.contiguous()
        v_bhsd = v_bshd.contiguous()
        B, H, _, _ = q_bhsd.shape
        L2 = int(q_bhsd.shape[2])
        S2 = int(k_bhsd.shape[2])
        fm = sdpa_kwargs["attn_mask"]
        if fm is not None:
            if fm.dtype is torch.bool:
                fm = fm.to(device=q_bhsd.device, non_blocking=True)
            else:
                fm = fm.to(device=q_bhsd.device, dtype=q_bhsd.dtype, non_blocking=True)
            fm, batch_dim, head_count, _qdim = _flatten_attn_mask(
                fm,
                device=q_bhsd.device,
                B=B,
                H=H,
                L=L2,
                S=S2,
            )
            if batch_dim not in (1, B):
                raise RuntimeError(
                    f"attn_mask batch dimension {batch_dim} incompatible with batch {B}"
                )
            if head_count not in (1, H):
                raise RuntimeError(
                    f"attn_mask head count {head_count} incompatible with num_heads {H}"
                )
            sdpa_kwargs["attn_mask"] = fm
            sdpa_kwargs["is_causal"] = False
        backends = get_dpa_backends()
        sdpa_out: Optional[torch.Tensor] = None
        if backends:
            try:
                from torch.nn.attention import sdpa_kernel
            except Exception:
                backends = []
        if backends:
            with sdpa_kernel(backends):
                sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                    q_bhsd, k_bhsd, v_bhsd, **sdpa_kwargs
                )
        if sdpa_out is None:
            sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, **sdpa_kwargs
            )
        try:
            B_, H_, L_, D_ = q_bhsd.shape
            S_ = k_bhsd.shape[2]
            flops = 2.0 * B_ * H_ * L_ * D_ * S_ + 2.0 * B_ * H_ * L_ * S_ * D_
            FLOP_PROFILER.add("DotProductAttention", float(flops))
        except Exception:
            pass
        return sdpa_out

    @staticmethod
    def _negotiate_dtype(tensor: torch.Tensor) -> torch.Tensor:
        device_type = tensor.device.type
        if device_type == "cpu" and tensor.dtype in (torch.float16, torch.bfloat16):
            return tensor.float()
        if device_type == "mps" and tensor.dtype == torch.bfloat16:
            return tensor.to(torch.float16)
        return tensor


class MultiScaleRetention(nn.Module):
    def __init__(self, d_model: int, nhead: int, use_gate: bool = True) -> None:
        super().__init__()
        self.d_model, self.nhead, self.use_gate = (
            int(d_model),
            int(nhead),
            bool(use_gate),
        )
        self._fallback = MultiScaleRetentionCompat(
            self.d_model, self.nhead, use_gate=self.use_gate
        )
        self._triton_ok = False
        self._triton_msr: Optional[MultiScaleRetentionTriton]
        self._triton_msr = None
        if _HAS_TRITON_MSR:
            try:
                self._triton_msr = MultiScaleRetentionTriton(
                    self.d_model, self.nhead, use_gate=self.use_gate
                )
                self._triton_ok = True
            except Exception:
                self._triton_ok = False
        self._decay_init = 5.0
        self._decay_range = 1.0

    def _decay_lambda(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        H = int(self.nhead)
        calc_dtype = dtype if dtype in (torch.float32, torch.float64) else torch.float32
        heads = torch.arange(H, device=device, dtype=calc_dtype)
        denom = float(max(H, 1))
        gammas = 1.0 - torch.pow(
            2.0,
            -(
                float(self._decay_init)
                + float(self._decay_range) * (heads / denom)
            ),
        )
        gammas = gammas.clamp(min=torch.finfo(calc_dtype).tiny, max=1.0 - 1e-9)
        if calc_dtype != dtype:
            gammas = gammas.to(dtype=dtype)
        return gammas

    def forward(
        self,
        x: torch.Tensor,
        *args: Any,
        attn_mask: Optional[torch.Tensor] = None,
        state: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        seq_len = x.shape[1]
        dtype = x.dtype
        device = x.device
        lam = self._decay_lambda(device, dtype)
        if (
            self._triton_ok
            and self._triton_msr is not None
            and device.type == "cuda"
            and torch.cuda.is_available()
        ):
            try:
                out_tr = self._triton_msr(
                    x,
                    lam,
                    attn_mask=attn_mask,
                    state=state,
                    **kwargs,
                )
                try:
                    B_ = x.shape[0]
                    fl = _compute_flops_msr(
                        B_,
                        seq_len,
                        num_heads=self.nhead,
                        head_dim=self.d_model // self.nhead,
                        use_gate=self.use_gate,
                    )
                    FLOP_PROFILER.add("MultiScaleRetention", float(fl))
                except Exception:
                    pass
                return out_tr
            except Exception:
                pass
        return self._fallback(
            x,
            attn_mask=attn_mask,
            state=state,
            decay=lam,
            **kwargs,
        )


class MultiScaleRetentionTriton(nn.Module):
    def __init__(self, d_model: int, nhead: int, use_gate: bool = True) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = self.d_model // self.nhead
        self.use_gate = bool(use_gate)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.g_proj = (
            nn.Linear(self.d_model, self.d_model, bias=False) if self.use_gate else None
        )
        self._beta = nn.Parameter(torch.full((self.nhead,), -0.2))
        self.norm = nn.LayerNorm(self.d_model)
        if not (_HAS_TRITON_MSR and torch.cuda.is_available()):
            raise RuntimeError("Triton MSR backend is not available (Triton+CUDA required).")

    @torch_compiler_disable(recursive=False, reason='Unable to be compiled')
    def forward(
        self,
        x: torch.Tensor,
        decay: Any,
        *args: Any,
        attn_mask: Optional[torch.Tensor] = None,
        state: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del args, kwargs
        if x.dim() != 3:
            raise ValueError(f"MultiScaleRetentionTriton expects (B,L,D), got {tuple(x.shape)}")
        if x.device.type != "cuda":
            raise RuntimeError("MultiScaleRetentionTriton only supports CUDA tensors.")
        B, L, D = x.shape
        if D != self.d_model:
            raise ValueError(f"Last dimension {D} must equal d_model={self.d_model}")
        x_in = x
        head_dim = self.head_dim
        q = self.q_proj(x_in).view(B, L, self.nhead, head_dim)
        v = self.v_proj(x_in).view(B, L, self.nhead, head_dim)
        lam_h: torch.Tensor
        if isinstance(decay, torch.Tensor):
            if decay.dim() == 1 and int(decay.shape[0]) == int(self.nhead):
                lam_h = decay.to(dtype=v.dtype, device=v.device)
            elif decay.dim() == 3:
                H = self.nhead
                if decay.shape[0] == H and decay.shape[1] >= 2 and decay.shape[2] >= 1:
                    lam_h = decay[:, 1, 0].to(dtype=v.dtype, device=v.device)
                else:
                    lam_h = torch.sigmoid(self._beta).to(dtype=v.dtype, device=v.device)
            else:
                lam_h = torch.sigmoid(self._beta).to(dtype=v.dtype, device=v.device)
        else:
            lam_h = torch.sigmoid(self._beta).to(dtype=v.dtype, device=v.device)
        lam = lam_h
        if isinstance(attn_mask, torch.Tensor) and attn_mask.dim() == 2:
            if attn_mask.shape == (B, L) and attn_mask.dtype == torch.bool:
                mask = attn_mask.to(device=v.device).unsqueeze(-1).unsqueeze(-1)
                v = torch.where(mask, torch.zeros_like(v), v)
        if isinstance(state, torch.Tensor):
            st = state
            if st.dim() == 4 and st.shape[2] == 1:
                st = st[:, :, 0, :]
            if st.dim() == 3 and st.shape[:2] == (B, self.nhead):
                v = v.clone()
                v[:, 0] = v[:, 0] + lam.view(1, self.nhead, 1) * st.to(
                    dtype=v.dtype, device=v.device
                )
        v_blhc = v.contiguous()
        out_state = torch.empty_like(v_blhc, dtype=v_blhc.dtype)
        SVB, SVL, SVH, SVD = v_blhc.stride()
        SOB, SOL, SOH, SOD = out_state.stride()
        BLOCK_DH: int
        num_warps: int
        env_block = env_str("STNET_MSR_TRITON_BLOCK_DH") or ""
        if env_block:
            try:
                _b = int(env_block)
                if _b in (16, 32, 64, 128):
                    BLOCK_DH = _b
                else:
                    raise ValueError
            except Exception:
                BLOCK_DH = 64 if head_dim >= 64 else 32
        else:
            BLOCK_DH = 64 if head_dim >= 64 else 32
        env_warps = env_str("STNET_MSR_TRITON_NUM_WARPS") or ""
        if env_warps:
            try:
                _w = int(env_warps)
                if _w in (1, 2, 4, 8, 16):
                    num_warps = _w
                else:
                    raise ValueError
            except Exception:
                num_warps = 8 if BLOCK_DH >= 64 else 4
        else:
            num_warps = 8 if BLOCK_DH >= 64 else 4
        grid = (B * self.nhead, triton.cdiv(head_dim, BLOCK_DH))
        _triton_retention[grid](
            v_blhc,
            lam,
            out_state,
            B,
            L,
            self.nhead,
            head_dim,
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
        state_tensor = out_state
        y = (q * state_tensor).contiguous().view(B, L, self.d_model)
        y = self.norm(y)
        if self.use_gate and self.g_proj is not None:
            gate = torch.nn.functional.silu(self.g_proj(x_in))
            y = y * gate
        out = self.o_proj(y)
        return out


class MultiScaleRetentionCompat(nn.Module):
    def __init__(self, d_model: int, nhead: int, use_gate: bool = True) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = self.d_model // self.nhead
        self.use_gate = bool(use_gate)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.g_proj = (
            nn.Linear(self.d_model, self.d_model, bias=False) if self.use_gate else None
        )
        self._beta = nn.Parameter(torch.full((self.nhead,), -0.2))
        self.norm = nn.LayerNorm(self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        *args: Any,
        decay: Any = None,
        attn_mask: Optional[torch.Tensor] = None,
        state: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        restore_dtype: Optional[torch.dtype] = None
        x_in = x
        if getattr(x.device, "type", "cpu") == "mps" and x.dtype == torch.bfloat16:
            restore_dtype = x.dtype
            x_in = x.to(torch.float16)
        del args, kwargs
        batch, seq_len, _ = x_in.shape
        if seq_len == 0:
            return x_in.new_zeros(x_in.shape)
        head_dim = self.head_dim
        q = self.q_proj(x_in).view(batch, seq_len, self.nhead, head_dim)
        v = self.v_proj(x_in).view(batch, seq_len, self.nhead, head_dim)
        manual_flops = _compute_flops_msr(
            batch,
            seq_len,
            num_heads=self.nhead,
            head_dim=head_dim,
            use_gate=self.use_gate and self.g_proj is not None,
        )
        lam_h: torch.Tensor
        if isinstance(decay, torch.Tensor):
            if decay.dim() == 1 and int(decay.shape[0]) == int(self.nhead):
                lam_h = decay.to(dtype=v.dtype, device=v.device)
            elif decay.dim() == 3:
                H = self.nhead
                if decay.shape[0] == H and decay.shape[1] >= 2 and decay.shape[2] >= 1:
                    lam_h = decay[:, 1, 0].to(dtype=v.dtype, device=v.device)
                else:
                    lam_h = torch.sigmoid(self._beta).to(dtype=v.dtype, device=v.device)
            else:
                lam_h = torch.sigmoid(self._beta).to(dtype=v.dtype, device=v.device)
        else:
            lam_h = torch.sigmoid(self._beta).to(dtype=v.dtype, device=v.device)
        lam = lam_h.view(1, self.nhead, 1)
        if isinstance(attn_mask, torch.Tensor) and attn_mask.dim() == 2:
            if attn_mask.shape == (batch, seq_len) and attn_mask.dtype == torch.bool:
                mask = attn_mask.to(device=v.device).unsqueeze(-1).unsqueeze(-1)
                v = torch.where(mask, torch.zeros_like(v), v)
        st_bhd: Optional[torch.Tensor] = None
        if isinstance(state, torch.Tensor):
            st = state
            if st.dim() == 4 and st.shape[2] == 1:
                st = st[:, :, 0, :]
            if st.dim() == 3 and st.shape[:2] == (batch, self.nhead):
                st_bhd = st.to(dtype=v.dtype, device=v.device)
        if st_bhd is not None:
            v = v.clone()
            v[:, 0] = v[:, 0] + lam * st_bhd
        calc_dtype = (
            torch.float32 if v.dtype in (torch.float16, torch.bfloat16) else v.dtype
        )
        lam_calc = lam_h.to(dtype=calc_dtype, device=v.device).view(1, 1, self.nhead, 1)
        t = torch.arange(seq_len, device=v.device, dtype=calc_dtype).view(1, seq_len, 1, 1)
        p = torch.pow(lam_calc, t)  # (1, T, H, 1)
        tiny = torch.finfo(calc_dtype).tiny
        p = p.clamp_min(tiny)
        inv_p = torch.reciprocal(p)
        v_scaled = v.to(dtype=calc_dtype) * inv_p
        v_scaled[:, 0].zero_()
        cumsum_scaled = torch.cumsum(v_scaled, dim=1)
        prev_scaled = v[:, 0].to(dtype=calc_dtype).unsqueeze(1)
        state_tensor = p * (prev_scaled + cumsum_scaled)
        state_tensor = state_tensor.to(dtype=v.dtype).contiguous()
        y = (q * state_tensor).contiguous().view(batch, seq_len, self.d_model)
        y = self.norm(y)
        if self.use_gate and self.g_proj is not None:
            gate = torch.nn.functional.silu(self.g_proj(x_in))
            y = y * gate
        if manual_flops > 0.0:
            FLOP_PROFILER.add("Retention", manual_flops)
        out = self.o_proj(y)
        if restore_dtype is not None:
            out = out.to(restore_dtype)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *args: Any,
        bias: bool = True,
        dropout: float = 0.0,
        batch_first: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
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
                if (
                    isinstance(impl, _MultiHeadAttentionNvidia)
                    and impl._te_mha is not None
                ):
                    self.impl = impl
                    self._backend = "te"
                else:
                    self.impl = _MultiHeadAttentionCompat(
                        embed_dim,
                        num_heads,
                        bias=bias,
                        dropout=dropout,
                        batch_first=batch_first,
                        **kwargs,
                    )
                    self._backend = "torch"
            except Exception:
                self.impl = _MultiHeadAttentionCompat(
                    embed_dim,
                    num_heads,
                    bias=bias,
                    dropout=dropout,
                    batch_first=batch_first,
                    **kwargs,
                )
                self._backend = "torch"
        else:
            self.impl = _MultiHeadAttentionCompat(
                embed_dim,
                num_heads,
                bias=bias,
                dropout=dropout,
                batch_first=batch_first,
                **kwargs,
            )
            self._backend = "torch"
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
    def backend(self) -> str:
        return self._backend

    def forward(
        self,
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


class _MultiHeadAttentionNvidia(nn.Module):
    def __init__(
        self,
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
            _fwd = getattr(self._te_mha, "forward", getattr(self._te_mha, "__call__", None))
            with torch.no_grad():
                if _fwd is not None:
                    try:
                        self._te_forward_signature = inspect.signature(_fwd)
                    except (TypeError, ValueError):
                        self._te_forward_signature = None
            params = self._te_forward_signature.parameters if self._te_forward_signature else {}
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
        if self._force_pt:
            warnings.warn(
                "Unable to use Transformer Engine multi-head attention; falling back to torch.nn.MultiheadAttention.",
                RuntimeWarning,
            )

    @staticmethod
    def _nvidia_mha(
        embed_dim: int, num_heads: int, dropout: float, kwargs: dict[str, Any]
    ) -> Any | None:
        if not _HAS_TE:
            return None
        if not _is_nvidia_te_supported():
            return None
        candidates = []
        for name in ("MultiHeadAttention", "MultiheadAttention"):
            if hasattr(te, name):
                candidates.append(getattr(te, name))
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
                dict(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout),
            )
            for ckw in ctor_variants:
                try:
                    return cls(**{**ckw, **kwargs})
                except TypeError:
                    continue
                except Exception:
                    continue
        return None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        is_causal: Optional[bool] = None,
        average_attn_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        embed_dim = int(query.shape[-1])

        if self._force_pt or (self._te_mha is None):
            return _call_sdpa_fallback(self._fallback, query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=need_weights, average_attn_weights=average_attn_weights, is_causal=is_causal)
        if need_weights:
            return _call_sdpa_fallback(self._fallback, query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=need_weights, average_attn_weights=average_attn_weights, is_causal=is_causal)
        if attn_mask is not None:
            return _call_sdpa_fallback(self._fallback, query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=need_weights, average_attn_weights=average_attn_weights, is_causal=is_causal)
        if (not query.is_cuda) or (query.dtype not in (torch.float16, torch.bfloat16)):
            return _call_sdpa_fallback(self._fallback, query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=need_weights, average_attn_weights=average_attn_weights, is_causal=is_causal)
        try:
            if torch._dynamo.is_compiling():
                return _call_sdpa_fallback(self._fallback, query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=need_weights, average_attn_weights=average_attn_weights, is_causal=is_causal)
        except Exception:
            pass
        bf = bool(self.batch_first)
        _q, _k, _v = query, key, value
        if not bf:
            _q, _k, _v = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        te_kwargs: dict[str, Any] = {}
        te_mask: Any = None
        mask_type: str | None = None
        if key_padding_mask is not None:
            B0 = int(_q.shape[0])
            Lq = int(_q.shape[1])
            Lk = int(_k.shape[1])
            if key_padding_mask.shape != (B0, Lk):
                return _call_sdpa_fallback(self._fallback, query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=need_weights, average_attn_weights=average_attn_weights, is_causal=is_causal)
            kpm = key_padding_mask
            if kpm.dtype is not torch.bool:
                kpm = kpm.to(torch.bool)
            kpm = kpm.to(device=_q.device, non_blocking=True).contiguous()
            kv_mask = kpm.view(B0, 1, 1, Lk)
            if Lq == Lk:
                te_mask = kv_mask
            else:
                if not self._te_supports_tuple_mask:
                    return _call_sdpa_fallback(self._fallback, query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=need_weights, average_attn_weights=average_attn_weights, is_causal=is_causal)
                q_mask = torch.zeros((B0, 1, 1, Lq), device=_q.device, dtype=torch.bool)
                te_mask = (q_mask, kv_mask)
            mask_type = "padding_causal" if bool(is_causal) else "padding"
        else:
            mask_type = "causal" if bool(is_causal) else "no_mask"
        if te_mask is not None and mask_type is not None and mask_type.startswith("padding"):
            if self._te_mask_type_param is None:
                return _call_sdpa_fallback(self._fallback, query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=need_weights, average_attn_weights=average_attn_weights, is_causal=is_causal)
        if te_mask is not None:
            if self._te_mask_param is None:
                return _call_sdpa_fallback(self._fallback, query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=need_weights, average_attn_weights=average_attn_weights, is_causal=is_causal)
            te_kwargs[self._te_mask_param] = te_mask
        if (mask_type is not None) and (self._te_mask_type_param is not None):
            te_kwargs[self._te_mask_type_param] = mask_type
        elif self._te_supports_is_causal and (is_causal is not None):
            te_kwargs["is_causal"] = bool(is_causal)
        if self._te_supports_training:
            te_kwargs["training"] = bool(self.training)
        try:
            out = self._te_mha(_q, _k, _v, **te_kwargs)
        except TypeError:
            if isinstance(te_mask, tuple):
                self._te_supports_tuple_mask = False
                return _call_sdpa_fallback(self._fallback, query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=need_weights, average_attn_weights=average_attn_weights, is_causal=is_causal)
            self._force_pt = True
            return _call_sdpa_fallback(self._fallback, query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=need_weights, average_attn_weights=average_attn_weights, is_causal=is_causal)
        except Exception:
            self._force_pt = True
            return _call_sdpa_fallback(self._fallback, query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=need_weights, average_attn_weights=average_attn_weights, is_causal=is_causal)
        if isinstance(out, tuple) and len(out) >= 1:
            y, w = out[0], None
        else:
            y, w = out, None
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
        self,
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
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        is_causal: Optional[bool] = None,
        average_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        kwargs = dict(key_padding_mask=key_padding_mask, need_weights=need_weights)
        if need_weights:
            kwargs["average_attn_weights"] = bool(average_attn_weights)

        out, w = _call_mha_compat(self.mha, query, key, value, attn_mask=attn_mask, is_causal=is_causal, kwargs=kwargs)
        _compute_flops_mha(
            query,
            key,
            num_heads=self.mha.num_heads,
            embed_dim=self.mha.embed_dim,
            batch_first=self.batch_first,
            include_projections=True,
        )
        return out, w
