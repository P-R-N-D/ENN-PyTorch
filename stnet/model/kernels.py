# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import inspect
import math
import warnings
from typing import Any, Optional, Tuple

import torch
import torch._dynamo
from torch import nn

from ..backend.profiler import FLOP_PROFILER, capture
from ..backend.system import (
    cuda_compute_capability,
    get_device,
    get_runtime_config,
    get_dpa_backends,
)

try:
    import triton
    import triton.language as tl

    _HAS_TRITON_LIB = True
except Exception:
    _HAS_TRITON_LIB = False

_HAS_TRITON_MSR = bool(_HAS_TRITON_LIB and torch.cuda.is_available())


def _clone_last_dim(x: torch.Tensor) -> torch.Tensor:
    return torch.stack((x, x), dim=-1).reshape(*x.shape[:-1], -1)


def _estimate_flops_msr(
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


def _add_mha_flops(
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


def _dpa_sequence_length(
    query: torch.Tensor, key: torch.Tensor, batch_first: bool
) -> tuple[int, int, int]:

    if batch_first:
        if query.dim() < 2 or key.dim() < 2:
            raise ValueError(
                "expected query/key tensors with at least 2 dims when batch_first=True"
            )
        batch = int(query.shape[0])
        seq_q = int(query.shape[1])
        seq_k = int(key.shape[1])
    else:
        if query.dim() < 2 or key.dim() < 2:
            raise ValueError(
                "expected query/key tensors with at least 2 dims when batch_first=False"
            )
        batch = int(query.shape[1])
        seq_q = int(query.shape[0])
        seq_k = int(key.shape[0])
    return batch, seq_q, seq_k


def _expand_for_mha(
    mask: torch.Tensor,
    *args: Any,
    batch: int,
    heads: int,
    seq_q: int,
    seq_k: int,
    device: torch.device,
    **kwargs: Any,
) -> torch.Tensor:

    if mask.dtype is not torch.bool:
        raise TypeError("expected boolean mask")
    if mask.dim() == 2:
        if mask.shape != (seq_q, seq_k):
            raise ValueError(
                f"mask shape {tuple(mask.shape)} incompatible with ({seq_q}, {seq_k})"
            )
        expanded = mask.view(1, 1, seq_q, seq_k).expand(batch, heads, seq_q, seq_k)
    elif mask.dim() == 3:
        if mask.shape == (batch, seq_q, seq_k):
            expanded = mask.view(batch, 1, seq_q, seq_k).expand(
                batch, heads, seq_q, seq_k
            )
        else:
            raise ValueError(f"unsupported 3D mask shape {tuple(mask.shape)}")
    elif mask.dim() == 4:
        b, h, sq, sk = mask.shape
        if b != batch or sq != seq_q or sk != seq_k:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} incompatible with (batch={batch}, seq_q={seq_q}, seq_k={seq_k})"
            )
        if h == 1:
            expanded = mask.expand(batch, heads, seq_q, seq_k)
        elif h == heads:
            expanded = mask
        else:
            raise ValueError(
                f"mask head dimension {h} does not match expected heads {heads}"
            )
    else:
        raise ValueError(f"unsupported mask rank {mask.dim()}")
    return expanded.to(device=device, dtype=torch.bool, non_blocking=True)


def to_additive_mask(
    attn_mask: torch.Tensor | None,
    *args: Any,
    batch: int,
    heads: int,
    seq_q: int,
    seq_k: int,
    dtype: torch.dtype,
    device: torch.device,
    **kwargs: Any,
) -> torch.Tensor:

    if attn_mask is None:
        return torch.zeros((batch, heads, seq_q, seq_k), dtype=dtype, device=device)
    elif attn_mask.dtype is torch.bool:
        expanded = _expand_for_mha(
            attn_mask, batch=batch, heads=heads, seq_q=seq_q, seq_k=seq_k, device=device
        )
        neg_inf = _negative_inf(dtype, device)
        zero = torch.zeros((), dtype=neg_inf.dtype, device=device)
        return torch.where(expanded, neg_inf, zero).to(dtype)

    am = attn_mask.to(device=device, dtype=dtype, non_blocking=True)
    match am.dim():
        case 2:
            if am.shape != (seq_q, seq_k):
                raise ValueError(
                    f"mask shape {tuple(am.shape)} incompatible with ({seq_q}, {seq_k})"
                )
            return (
                am.view(1, 1, seq_q, seq_k)
                .expand(batch, heads, seq_q, seq_k)
                .contiguous()
            )
        case 3:
            if am.shape != (batch, seq_q, seq_k):
                raise ValueError(
                    f"mask shape {tuple(am.shape)} incompatible with (batch={batch}, seq_q={seq_q}, seq_k={seq_k})"
                )
            return (
                am.view(batch, 1, seq_q, seq_k)
                .expand(batch, heads, seq_q, seq_k)
                .contiguous()
            )
        case 4:
            b, h, sq, sk = am.shape
            if b != batch or sq != seq_q or sk != seq_k:
                raise ValueError(
                    f"mask shape {tuple(am.shape)} incompatible with (batch={batch}, seq_q={seq_q}, seq_k={seq_k})"
                )
            if h == 1:
                return am.expand(batch, heads, seq_q, seq_k).contiguous()
            if h == heads:
                return am.contiguous()
            raise ValueError(
                f"mask head dimension {h} does not match expected heads {heads}"
            )
        case _:
            raise ValueError(f"unsupported mask rank {am.dim()}")


def _negative_inf(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if not torch.is_floating_point(torch.empty((), dtype=dtype)):
        dtype = torch.float32
    return torch.tensor(float("-inf"), dtype=dtype, device=device)


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
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
    except Exception:
        index = 0
    try:
        props = torch.cuda.get_device_properties(index)
        maj = int(getattr(props, "major", 0))
        minr = int(getattr(props, "minor", 0))
    except Exception:
        try:
            maj, minr = torch.cuda.get_device_capability(index)
        except Exception:
            return False
    min_major, min_minor = 8, 0
    if maj < min_major or (maj == min_major and minr < min_minor):
        return False
    try:
        if torch._dynamo.is_compiling():
            return False
    except Exception:
        pass
    return True


def _to_nvidia_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    key_padding_mask: Optional[torch.Tensor],
    *args: Any,
    num_heads: int,
    batch_first: bool,
    **kwargs: Any,
) -> Optional[torch.Tensor]:

    if attn_mask is None and key_padding_mask is None:
        return None

    batch, seq_q, seq_k = _dpa_sequence_length(query, key, batch_first)
    device = query.device
    dtype = query.dtype
    heads = int(num_heads)
    neg_inf = _negative_inf(dtype, device)
    mask_dtype = neg_inf.dtype
    zero_scalar = torch.zeros((), dtype=mask_dtype, device=device)
    float_mask: Optional[torch.Tensor] = None

    try:
        if attn_mask is not None:
            if attn_mask.dtype is torch.bool:
                expanded = _expand_for_mha(
                    attn_mask,
                    batch=batch,
                    heads=heads,
                    seq_q=seq_q,
                    seq_k=seq_k,
                    device=device,
                )
                float_mask = torch.where(expanded, neg_inf, zero_scalar)
            elif torch.is_floating_point(attn_mask):
                am = attn_mask.to(device=device, dtype=mask_dtype, non_blocking=True)
                if am.dim() == 2:
                    if am.shape != (seq_q, seq_k):
                        return None
                    float_mask = (
                        am.view(1, 1, seq_q, seq_k)
                        .expand(batch, heads, seq_q, seq_k)
                        .clone()
                    )
                elif am.dim() == 3:
                    if am.shape != (batch, seq_q, seq_k):
                        return None
                    float_mask = (
                        am.view(batch, 1, seq_q, seq_k)
                        .expand(batch, heads, seq_q, seq_k)
                        .clone()
                    )
                elif am.dim() == 4:
                    if (
                        am.shape[0] != batch
                        or am.shape[2] != seq_q
                        or am.shape[3] != seq_k
                    ):
                        return None
                    if am.shape[1] == 1:
                        float_mask = am.expand(batch, heads, seq_q, seq_k).clone()
                    elif am.shape[1] == heads:
                        float_mask = am.clone()
                    else:
                        return None
                else:
                    return None
            else:
                return None

        if key_padding_mask is not None:
            if key_padding_mask.dtype is not torch.bool:
                key_padding_mask = key_padding_mask.to(
                    device=device, dtype=torch.bool, non_blocking=True
                )
            else:
                key_padding_mask = key_padding_mask.to(device=device, non_blocking=True)
            if key_padding_mask.dim() != 2 or key_padding_mask.shape != (batch, seq_k):
                return None
            padding = key_padding_mask.view(batch, 1, 1, seq_k)
            pad_values = torch.where(
                padding.expand(batch, heads, seq_q, seq_k),
                neg_inf,
                zero_scalar,
            )
            float_mask = pad_values if float_mask is None else float_mask + pad_values

        return float_mask.contiguous() if float_mask is not None else None
    except Exception:
        return None


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
    else:
        try:
            index = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
        except Exception:
            index = 0
        try:
            props = torch.cuda.get_device_properties(index)
            major = int(getattr(props, "major", 0))
        except Exception:
            try:
                major, _ = torch.cuda.get_device_capability(index)
            except Exception:
                major = 0
        min_major = 8
        if min_major >= 10:
            min_major //= 10
        if major < min_major:
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


def _is_nvidia_mha_preferred(min_cc: Tuple[int, int] = (8, 0)) -> bool:
    if not _HAS_TE:
        return False
    if not _is_nvidia_te_supported():
        return False
    device = get_device()
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    cc = cuda_compute_capability(device)
    return cc >= min_cc


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
            with contextlib.ExitStack() as stack:
                import warnings as _warnings

                stack.enter_context(_warnings.catch_warnings())
                _warnings.filterwarnings(
                    "ignore",
                    message="Detected a Jax installation.*",
                    category=RuntimeWarning,
                )
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
                if uniq.numel() <= 2 and set(uniq.tolist()).issubset({0, 1}):
                    mask_bool = m != 0
                else:
                    bias_float = m.to(dtype=q_bshd.dtype)
            else:
                mask_bool = m.to(torch.bool)

            def _flatten_mask(
                mask: torch.Tensor,
            ) -> tuple[torch.Tensor, int, int]:
                if mask.dim() == 0:
                    shaped = mask.to(device=q_bshd.device).view(1, 1, 1, 1)
                    shaped = shaped.expand(1, 1, L, S)
                    return shaped.contiguous(), 1, 1
                if mask.dim() < 2:
                    raise RuntimeError(
                        f"attn_mask rank {mask.dim()} not supported; expected at least 2 dimensions"
                    )
                if mask.shape[-2:] != (L, S):
                    raise RuntimeError(
                        "attn_mask trailing dims {} do not match expected (L={}, S={})".format(
                            tuple(mask.shape[-2:]), L, S
                        )
                    )
                mask = mask.to(device=q_bshd.device).contiguous()
                while True:
                    leading = mask.shape[:-2]
                    if not leading:
                        return mask.view(1, 1, L, S).contiguous(), 1, 1
                    batch_dim = leading[0]
                    if batch_dim in (B, 1):
                        head_dims = leading[1:]
                        break
                    if batch_dim == H:
                        mask = mask.unsqueeze(0)
                        continue
                    raise RuntimeError(
                        f"attn_mask batch dimension {batch_dim} incompatible with batch {B}"
                    )
                head_dims = tuple(head_dims)
                head_count = 1 if not head_dims else math.prod(head_dims)
                mask = mask.view(batch_dim, head_count, L, S)
                if head_count not in (1, H):
                    raise RuntimeError(
                        "attn_mask head dims {} collapse to {} which is not compatible with num_heads {}".format(
                            head_dims, head_count, H
                        )
                    )
                return mask.contiguous(), int(batch_dim), int(head_count)

            def _to_bhls(
                x: torch.Tensor | None, *, dtype: torch.dtype | None = None
            ) -> torch.Tensor | None:
                if x is None:
                    return None
                mask, batch_dim, head_count = _flatten_mask(x)
                if batch_dim != B:
                    mask = mask.expand(B, head_count, L, S)
                    batch_dim = B
                if head_count == 1:
                    mask = mask.expand(batch_dim, H, L, S)
                elif head_count == H and batch_dim != B:
                    mask = mask.expand(B, head_count, L, S)
                if dtype is not None and mask.dtype != dtype:
                    mask = mask.to(dtype=dtype)
                return mask.contiguous()

            mask_bool = _to_bhls(mask_bool)
            if bias_float is not None:
                bias_float = _to_bhls(bias_float, dtype=q_bshd.dtype)

            if (
                mask_bool is not None
                and bias_float is None
                and self._te_ok
                and self._te_attn is not None
                and not self._te_supports_mask
                and self._te_supports_core_bias
            ):
                finfo = torch.finfo(q_bshd.dtype)
                zero = torch.zeros((), dtype=q_bshd.dtype, device=q_bshd.device)
                neg_inf = torch.full(
                    (), finfo.min, dtype=q_bshd.dtype, device=q_bshd.device
                )
                bias_float = torch.where(mask_bool, neg_inf, zero)
                mask_bool = None

        try:
            is_compiling = torch._dynamo.is_compiling()
        except Exception:
            is_compiling = False

        use_te = (
            self._te_ok
            and self.te_first
            and not self._force_pt
            and (self._te_attn is not None)
            and (attn_mask is None or mask_bool is None or self._te_supports_mask)
            and not is_compiling
            and q_bshd.is_cuda
            and q_bshd.dtype in (torch.float16, torch.bfloat16)
            and ((mask_bool is None) or self._te_supports_mask)
            and ((bias_float is None) or self._te_supports_core_bias)
        )
        if use_te:
            q_te = q_bshd.transpose(1, 2).contiguous()
            k_te = k_bshd.transpose(1, 2).contiguous()
            v_te = v_bshd.transpose(1, 2).contiguous()
            te_kwargs: dict[str, Any] = {}
            if self._te_supports_attention_dropout:
                te_kwargs["attention_dropout"] = dropout_val
            if self._te_supports_is_causal:
                te_kwargs["is_causal"] = bool(is_causal)
            if self._te_supports_training:
                te_kwargs["training"] = training
            if mask_bool is not None and self._te_mask_param:
                te_kwargs[self._te_mask_param] = mask_bool
                if self._te_supports_mask_type and self._te_mask_type_param:
                    te_kwargs[self._te_mask_type_param] = "arbitrary"
            if bias_float is not None and self._te_core_bias_param:
                te_kwargs[self._te_core_bias_param] = bias_float
                if self._te_supports_core_bias_type and self._te_core_bias_type_param:
                    te_kwargs[self._te_core_bias_type_param] = "post_scale_bias"
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
        sdpa_bias: torch.Tensor | None = None
        if mask_bool is not None:
            finfo = torch.finfo(q_bshd.dtype)
            zero = torch.zeros((), dtype=q_bshd.dtype, device=q_bshd.device)
            neg_inf = torch.full(
                (), finfo.min, dtype=q_bshd.dtype, device=q_bshd.device
            )
            sdpa_bias = torch.where(mask_bool, neg_inf, zero).expand(B, H, L, S)
        if bias_float is not None:
            base = (
                sdpa_bias
                if sdpa_bias is not None
                else torch.zeros(B, H, L, S, device=q_bshd.device, dtype=q_bshd.dtype)
            )
            sdpa_bias = base + bias_float
        final_mask = (
            attn_mask if (attn_mask is not None and sdpa_bias is None) else sdpa_bias
        )
        sdpa_kwargs = {
            "attn_mask": final_mask,
            "dropout_p": dropout_val,
            "is_causal": bool(is_causal),
        }
        q_bhsd = q_bshd.contiguous()
        k_bhsd = k_bshd.contiguous()
        v_bhsd = v_bshd.contiguous()

        B, H, _, _ = q_bhsd.shape
        fm = sdpa_kwargs["attn_mask"]
        if fm is not None:
            if fm.dtype is torch.bool:
                fm = fm.to(device=q_bhsd.device, non_blocking=True)
            else:
                fm = fm.to(device=q_bhsd.device, dtype=q_bhsd.dtype, non_blocking=True)

            fm, batch_dim, head_count = _flatten_mask(fm)
            if batch_dim not in (1, B):
                raise RuntimeError(
                    f"attn_mask batch dimension {batch_dim} incompatible with batch {B}"
                )
            if head_count not in (1, H):
                raise RuntimeError(
                    f"attn_mask head count {head_count} incompatible with num_heads {H}"
                )
            sdpa_kwargs["attn_mask"] = fm.contiguous()
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
        attn_mask: Optional[torch.Tensor] = None,
        state: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        restore_dtype: Optional[torch.dtype] = None
        x_in = x
        if getattr(x.device, "type", "cpu") == "mps" and x.dtype == torch.bfloat16:
            restore_dtype = x.dtype
            x_in = x.to(torch.float16)

        sin = kwargs.pop("sin", None)
        cos = kwargs.pop("cos", None)
        decay = kwargs.pop("decay", None)

        batch, seq_len, _ = x_in.shape
        if seq_len == 0:
            return x_in.new_zeros(x_in.shape)
        head_dim = self.head_dim
        q = self.q_proj(x_in).view(batch, seq_len, self.nhead, head_dim)
        v = self.v_proj(x_in).view(batch, seq_len, self.nhead, head_dim)
        manual_flops = _estimate_flops_msr(
            batch,
            seq_len,
            num_heads=self.nhead,
            head_dim=head_dim,
            use_gate=self.use_gate and self.g_proj is not None,
        )

        if isinstance(decay, torch.Tensor) and decay.dim() == 3:
            H = self.nhead
            if decay.shape[0] == H and decay.shape[1] >= 2 and decay.shape[2] >= 1:
                lam_h = decay[:, 1, 0].to(dtype=v.dtype, device=v.device)
            else:
                lam_h = torch.sigmoid(self._beta).to(dtype=v.dtype, device=v.device)
        else:
            lam_h = torch.sigmoid(self._beta).to(dtype=v.dtype, device=v.device)
        lam = lam_h.view(1, self.nhead, 1)

        if isinstance(state, torch.Tensor):
            st = state
            if st.dim() == 4 and st.shape[2] == 1:
                st = st[:, :, 0, :]
            if st.dim() == 3 and st.shape[:2] == (batch, self.nhead):
                prev = st.to(dtype=v.dtype, device=v.device)
            else:
                prev = v[:, 0].clone()
        else:
            prev = v[:, 0].clone()

        if isinstance(attn_mask, torch.Tensor) and attn_mask.dim() == 2:
            if attn_mask.shape == (batch, seq_len) and attn_mask.dtype == torch.bool:
                mask = attn_mask.to(device=v.device).unsqueeze(-1).unsqueeze(-1)
                v = torch.where(mask, torch.zeros_like(v), v)

        states = [prev]
        for index in range(1, seq_len):
            prev = lam * prev + v[:, index]
            states.append(prev)
        state_tensor = torch.stack(states, dim=1).contiguous()
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
        self._ts_ok = False
        self._triton_ok = False
        try:
            from torchscale.component.multiscale_retention import (
                MultiScaleRetention as _TorchScaleMSR,
            )

            self._ts_msr = _TorchScaleMSR(self.d_model, self.nhead)
            self._msr_dev_tag: Optional[str] = None
            self._msr_compiled: bool = False
            self._msr_ipex_infer: bool = False
            self._ts_key_dim = int(
                getattr(self._ts_msr, "key_dim", self.d_model // self.nhead)
            )
            self._ts_ok = True
        except Exception:
            self._ts_ok = False
            self._ts_msr = None  # type: ignore[assignment]

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
        self._rope_theta = 10000.0
        self._decay_init = 5.0
        self._decay_range = 1.0

    def _get_coords(self, seq_len: Any, device: Any, dtype: Any) -> Any:
        key_dim = int(
            self._ts_key_dim
            if getattr(self, "_ts_key_dim", None) is not None
            else self.d_model // self.nhead
        )
        half = key_dim // 2
        coord_dtype = (
            torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        )
        positions = torch.arange(seq_len, device=device, dtype=coord_dtype)
        inv_freq = 1.0 / self._rope_theta ** torch.linspace(
            0, 1, half, device=device, dtype=coord_dtype
        )
        freqs = torch.einsum("n,d->nd", positions, inv_freq)
        sin = _clone_last_dim(torch.sin(freqs)).to(dtype)[None, None, :, :]
        cos = _clone_last_dim(torch.cos(freqs)).to(dtype)[None, None, :, :]
        length = seq_len
        idx_i = torch.arange(length, device=device)
        idx_j = torch.arange(length, device=device)
        diff = (idx_i[:, None] - idx_j[None, :]).to(dtype)
        tril = (idx_i[:, None] >= idx_j[None, :]).to(dtype)
        heads = torch.arange(self.nhead, device=device, dtype=dtype)
        gammas = 1.0 - torch.pow(
            2.0,
            -(self._decay_init + self._decay_range * (heads / max(self.nhead, 1))),
        )
        gammas = torch.clamp(gammas, min=torch.finfo(dtype).tiny, max=1 - 1e-09)
        powers = torch.pow(gammas.view(self.nhead, 1, 1), diff.abs().to(dtype))
        rel = tril * powers
        sin = sin.expand(self.nhead, -1, -1, -1)
        cos = cos.expand(self.nhead, -1, -1, -1)
        return sin, cos, rel

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
        sin, cos, rel = self._get_coords(seq_len, device, dtype)

        if self._ts_ok:
            try:
                outputs = self._ts_msr(  # type: ignore[call-arg]
                    x,
                    sin=sin,
                    cos=cos,
                    decay=rel,
                    attn_mask=attn_mask,
                    state=state,
                    **kwargs,
                )
            except TypeError:
                outputs = self._ts_msr(x, sin=sin, cos=cos, decay=rel)  # type: ignore[call-arg]
            except Exception:
                outputs = None

            if isinstance(outputs, torch.Tensor):
                try:
                    B_ = x.shape[0]
                    fl = _estimate_flops_msr(B_, seq_len, num_heads=self.nhead, head_dim=self.d_model // self.nhead, use_gate=self.use_gate)
                    FLOP_PROFILER.add("MultiScaleRetention", float(fl))
                except Exception:
                    pass
                return outputs
            if isinstance(outputs, tuple) and outputs:
                return outputs[0]

        if (
            self._triton_ok
            and self._triton_msr is not None
            and device.type == "cuda"
            and torch.cuda.is_available()
        ):
            try:
                out_tr = self._triton_msr(
                    x,
                    sin,
                    cos,
                    rel,
                    attn_mask=attn_mask,
                    state=state,
                    **kwargs,
                )
                try:
                    B_ = x.shape[0]
                    fl = _estimate_flops_msr(B_, seq_len, num_heads=self.nhead, head_dim=self.d_model // self.nhead, use_gate=self.use_gate)
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
            sin=sin,
            cos=cos,
            decay=rel,
            **kwargs,
        )


@triton.jit
def _triton_retention(
    V,
    LAMBDA,
    OUT,
    B: tl.constexpr,
    L: tl.constexpr,
    H: tl.constexpr,
    DH: tl.constexpr,
    SVB, SVL, SVH, SVD,
    SOB, SOL, SOH, SOD,
    BLOCK_DH: tl.constexpr,
):
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

    def forward(
        self,
        x: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        decay: torch.Tensor,
        *args: Any,
        attn_mask: Optional[torch.Tensor] = None,
        state: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del sin, cos, kwargs

        if x.dim() != 3:
            raise ValueError(f"MultiScaleRetentionTriton expects (B,L,D), got {tuple(x.shape)}")
        if x.device.type != "cuda":
            raise RuntimeError("MultiScaleRetentionTriton only supports CUDA tensors.")

        B, L, D = x.shape
        if D != self.d_model:
            raise ValueError(f"Last dimension {D} must equal d_model={self.d_model}")

        x_in = x
        restore_dtype: Optional[torch.dtype] = None
        if x_in.device.type == "mps" and x_in.dtype == torch.bfloat16:
            restore_dtype = x_in.dtype
            x_in = x_in.to(torch.float16)

        head_dim = self.head_dim
        q = self.q_proj(x_in).view(B, L, self.nhead, head_dim)
        v = self.v_proj(x_in).view(B, L, self.nhead, head_dim)

        if isinstance(decay, torch.Tensor) and decay.dim() == 3:
            H = self.nhead
            if decay.shape[0] == H and decay.shape[1] >= 2 and decay.shape[2] >= 1:
                lam_h = decay[:, 1, 0].to(dtype=v.dtype, device=v.device)
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

        BLOCK_DH = 32
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
        )

        state_tensor = out_state.to(v.dtype)
        y = (q * state_tensor).contiguous().view(B, L, self.d_model)
        y = self.norm(y)
        if self.use_gate and self.g_proj is not None:
            gate = torch.nn.functional.silu(self.g_proj(x_in))
            y = y * gate
        out = self.o_proj(y)
        if restore_dtype is not None:
            out = out.to(restore_dtype)
        return out


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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        kwargs = dict(key_padding_mask=key_padding_mask, need_weights=need_weights)

        def _call_mha(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            try:
                if is_causal is not None:
                    return self.mha(
                        q,
                        k,
                        v,
                        attn_mask=attn_mask,
                        is_causal=is_causal,
                        **kwargs,
                    )
            except TypeError:
                pass
            return self.mha(
                q,
                k,
                v,
                attn_mask=attn_mask,
                **kwargs,
            )

        out, w = _call_mha(query, key, value)
        _add_mha_flops(
            query,
            key,
            num_heads=self.mha.num_heads,
            embed_dim=self.mha.embed_dim,
            batch_first=self.batch_first,
            include_projections=True,
        )
        return out, w


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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        embed_dim = int(query.shape[-1])
        if self._force_pt or (self._te_mha is None):
            return self._fallback(
                query,
                key,
                value,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                is_causal=is_causal,
            )
        te_attn_mask = attn_mask
        if attn_mask is not None or key_padding_mask is not None:
            te_attn_mask = _to_nvidia_mask(
                query,
                key,
                attn_mask,
                key_padding_mask,
                num_heads=self.num_heads,
                batch_first=self.batch_first,
            )
            if te_attn_mask is None:
                self._force_pt = True
                return self._fallback(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    is_causal=is_causal,
                )
        bf = bool(self.batch_first)
        _q, _k, _v = query, key, value
        if not bf:
            _q, _k, _v = (
                query.transpose(0, 1),
                key.transpose(0, 1),
                value.transpose(0, 1),
            )
        for variant in (
            dict(
                query=_q,
                key=_k,
                value=_v,
                attn_mask=te_attn_mask,
                need_weights=need_weights,
                is_causal=is_causal,
            ),
            dict(query=_q, attn_mask=te_attn_mask, need_weights=need_weights),
            dict(
                query=_q,
                key=_k,
                value=_v,
                attention_mask=te_attn_mask,
                need_weights=need_weights,
            ),
        ):
            try:
                out = self._te_mha(**variant)
                if isinstance(out, tuple) and len(out) >= 1:
                    y, w = out[0], (out[1] if need_weights and len(out) > 1 else None)
                else:
                    y, w = out, None
                if not bf and isinstance(y, torch.Tensor) and y.dim() >= 2:
                    y = y.transpose(0, 1)
                _add_mha_flops(
                    query,
                    key,
                    num_heads=self.num_heads,
                    embed_dim=embed_dim,
                    batch_first=self.batch_first,
                    include_projections=True,
                )
                return y, w
            except TypeError:
                continue
            except Exception:
                continue
        self._force_pt = True
        return self._fallback(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            is_causal=is_causal,
        )


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *args: Any,
        bias: bool = True,
        dropout: float = 0.0,
        batch_first: bool = True,
        prefer_te_min_cc: Tuple[int, int] = (8, 0),
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if _is_nvidia_mha_preferred(prefer_te_min_cc):
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.impl(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            is_causal=is_causal,
        )
