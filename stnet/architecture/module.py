# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from dataclasses import dataclass
from math import prod
from typing import Any, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Normal, StudentT

from ..toolkit.compat import patch_torch
from ..toolkit.optimization import GatedMultiScaleRetention, ScaledDotProductAttention


class StochasticDepth(nn.Module):
    def __init__(self, p: float = 0.0, mode: str = "row") -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = float(p)
        self.mode = str(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        if keep_prob <= 0.0:
            return torch.zeros_like(x)
        if self.mode == "row":
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        else:
            shape = (1,) * x.ndim
        noise = x.new_empty(shape).bernoulli_(keep_prob).div_(keep_prob)
        return x * noise


def _norm(norm_type: str, d_model: int) -> nn.Module:
    kind = str(norm_type).lower()
    if kind in {"layernorm", "layer_norm", "ln"}:
        return nn.LayerNorm(d_model)
    if kind in {"rmsnorm", "rms_norm", "rms"} and hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(d_model)
    if kind in {"batchnorm", "batchnorm1d", "bn", "bn1d"}:
        return nn.BatchNorm1d(d_model)
    return nn.LayerNorm(d_model)


def _stochastic_depth_scheduler(max_rate: float, depth: int) -> list[float]:
    if depth <= 0:
        return []
    if max_rate <= 0.0:
        return [0.0 for _ in range(depth)]
    step = float(max_rate) / max(1, depth)
    return [step * float(index + 1) for index in range(depth)]

patch_torch()


if TYPE_CHECKING:
    from .network import Config


def _canon_dims(x: torch.Tensor, dims: Any, keep_batch: bool = False) -> Tuple[int, ...]:
    nd = x.ndim
    if dims is None:
        return tuple(range(1, nd)) if nd > 1 else (0,)
    if isinstance(dims, int):
        dims = (dims,)
    pos = []
    for d in dims:
        d0 = d + nd if d < 0 else d
        if 0 <= d0 < nd and (keep_batch or d0 != 0):
            pos.append(d0)
    out = tuple(sorted(set(pos)))
    if not out:
        return (1,) if nd > 1 else (0,)
    return out


def _normalize_data_definition(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode in {"ss", "spatial", "sxs"}:
        return "sxs"
    if mode in {"tt", "temporal", "txt"}:
        return "txt"
    if mode in {"txs"}:
        return "txs"
    if mode in {
        "st",
        "ts",
        "sxt",
        "spatiotemporal",
        "spatio-temporal",
        "temporospatial",
        "temporo-spatial",
    }:
        return "sxt"
    raise ValueError(f"Unsupported data definition '{value}'")


def _stable_std(
    x: torch.Tensor,
    dim: Tuple[int, ...] | int | None,
    ddof: int,
    eps: float,
) -> torch.Tensor:
    if dim is None:
        dim_tuple: Tuple[int, ...] = tuple()
    elif isinstance(dim, int):
        dim_tuple = (dim,)
    else:
        dim_tuple = tuple(dim)
    dims = tuple(d if d >= 0 else x.dim() + d for d in dim_tuple)
    sample = 1
    for d in dims:
        if 0 <= d < x.dim():
            sample *= max(1, int(x.shape[d]))
    if sample <= 1:
        base = x.mean(dim=dim_tuple, keepdim=True)
        return torch.full_like(base, fill_value=eps)
    correction = min(int(ddof), max(sample - 1, 0))
    try:
        var = torch.var(x, dim=dim_tuple, correction=correction, keepdim=True)
        std = torch.sqrt(torch.clamp(var, min=eps * eps))
    except TypeError:
        std = torch.std(x, dim=dim_tuple, unbiased=correction == 1, keepdim=True)
        std = torch.clamp(std, min=eps)
    return std


def _normal_cdf_loc_scale(
    x: torch.Tensor,
    loc: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    z = (x - loc) / torch.clamp(scale, min=1e-12)
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def _student_t_cdf_loc_scale(
    x: torch.Tensor,
    df: torch.Tensor,
    loc: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    t = (x - loc) / torch.clamp(scale, min=1e-12)
    v = df.to(dtype=t.dtype, device=t.device)
    x2 = v / torch.clamp(v + t * t, min=1e-12)
    x2 = torch.clamp(x2, 0.0, 1.0)
    if hasattr(torch.special, "betainc"):
        a = v / 2.0
        b = torch.full_like(v, 0.5)
        ib = torch.special.betainc(a, b, x2)
        return torch.where(t >= 0, 1.0 - 0.5 * ib, 0.5 * ib)
    v_clamped = torch.clamp(v, min=3.0)
    z = t * torch.sqrt((v_clamped - 2.0) / v_clamped)
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        *args: Any,
        in_channels: int = 1,
        d_model: int = 128,
        ndim: int = 2,
        patch: Tuple[int, ...] = (4, 4, 4),
        stride: Optional[Tuple[int, ...]] = None,
        grid: Optional[Tuple[int, ...]] = None,
        grid_3d: Optional[Tuple[int, int, int]] = None,
        dropout: float = 0.0,
        pad_to_multiple: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__()
        if ndim not in (1, 2, 3):
            raise ValueError(f"ndim must be 1, 2, or 3, got {ndim!r}")
        self.ndim = int(ndim)
        self.grid = grid
        self.grid_3d = grid_3d
        if any((p <= 0 for p in patch)):
            raise ValueError(f"patch sizes must be positive, got {patch}")
        if stride is not None and (len(stride) < self.ndim or any((s <= 0 for s in stride[: self.ndim]))):
            raise ValueError(f"stride must have length >= {self.ndim} with positive values, got {stride}")
        match self.ndim:
            case 1:
                if grid is not None and len(grid) not in (0, 1):
                    raise ValueError(f"1D grid must be None or (S,), got {grid}")
            case 2:
                if grid is not None and len(grid) != 2:
                    raise ValueError(f"2D grid must be (H,W), got {grid}")
            case 3:
                if grid_3d is not None and len(grid_3d) != 3:
                    raise ValueError(f"3D grid must be (T,H,W), got {grid_3d}")
        self.dropout = nn.Dropout(dropout)
        self.d_model = int(d_model)
        self.patch = patch
        self.pad_to_multiple = bool(pad_to_multiple)
        stride = patch if stride is None else stride
        match self.ndim:
            case 1:
                self.proj = nn.Conv1d(in_channels, d_model, kernel_size=(patch[0],), stride=(stride[0],))
            case 2:
                self.proj = nn.Conv2d(
                    in_channels, d_model, kernel_size=(patch[0], patch[1]), stride=(stride[0], stride[1])
                )
            case 3:
                self.proj = nn.Conv3d(
                    in_channels,
                    d_model,
                    kernel_size=(patch[0], patch[1], patch[2]),
                    stride=(stride[0], stride[1], stride[2]),
                )

    def _normalize_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            b, fdim = x.shape
            match self.ndim:
                case 1:
                    if self.grid is None:
                        l = fdim
                        k = self.patch[0]
                        need = (l + k - 1) // k * k
                        if fdim < need:
                            x = torch.nn.functional.pad(x, (0, need - fdim))
                        return x.view(b, 1, -1)
                    (l,) = self.grid
                    if fdim < l:
                        x = torch.nn.functional.pad(x, (0, l - fdim))
                    elif fdim > l:
                        raise ValueError(f"[B,F] grid(L={l}) but F={fdim} > L.")
                    return x.view(b, 1, l)
                case 2:
                    if self.grid is None:
                        side = int(math.ceil(math.sqrt(fdim)))
                        h = w = side
                    else:
                        h, w = self.grid
                        need_hw = h * w
                        if fdim > need_hw:
                            raise ValueError(f"[B,F] grid({h}x{w}) but F={fdim} > H*W={need_hw}.")
                    need = h * w
                    if fdim < need:
                        x = torch.nn.functional.pad(x, (0, need - fdim))
                    x = x.view(b, 1, h, w)
                    if self.pad_to_multiple:
                        ht = int(math.ceil(h / self.patch[0]) * self.patch[0])
                        wt = int(math.ceil(w / self.patch[1]) * self.patch[1])
                        if ht != h or wt != w:
                            x = torch.nn.functional.pad(x, (0, wt - w, 0, ht - h))
                    return x.contiguous(memory_format=torch.channels_last)
                case 3:
                    if self.grid_3d is None:
                        raise ValueError("Provide grid_3d=(T,H,W) for 3D with [B,F].")
                    t, h, w = self.grid_3d
                    need = t * h * w
                    if fdim < need:
                        x = torch.nn.functional.pad(x, (0, need - fdim))
                    elif fdim > need:
                        raise ValueError(f"[B,F] grid_3d product {need}, but F={fdim} > product.")
                    x = x.view(b, 1, t, h, w)
                    if self.pad_to_multiple:
                        tt = int(math.ceil(t / self.patch[0]) * self.patch[0])
                        ht = int(math.ceil(h / self.patch[1]) * self.patch[1])
                        wt = int(math.ceil(w / self.patch[2]) * self.patch[2])
                        if tt != t or ht != h or wt != w:
                            x = torch.nn.functional.pad(x, (0, wt - w, 0, ht - h, 0, tt - t))
                    return x.contiguous(memory_format=torch.channels_last_3d)
        match self.ndim:
            case 1:
                return x if x.ndim == 3 else x.unsqueeze(1)
            case 2:
                x = x if x.ndim == 4 else x.unsqueeze(1)
                return x.contiguous(memory_format=torch.channels_last)
            case 3:
                x = x if x.ndim == 5 else x.unsqueeze(1)
                return x.contiguous(memory_format=torch.channels_last_3d)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        x = self._normalize_shape(x)
        y = self.proj(x)
        match self.ndim:
            case 1:
                b, d, l = y.shape
                tokens = y.transpose(1, 2).contiguous().view(b, l, d)
                meta = (l, 1, 1)
            case 2:
                b, d, h, w = y.shape
                tokens = y.permute(0, 2, 3, 1).contiguous().view(b, h * w, d)
                meta = (1, h, w)
            case 3:
                b, d, t, h, w = y.shape
                tokens = y.permute(0, 2, 3, 4, 1).contiguous().view(b, t * h * w, d)
                meta = (t, h, w)
        return (self.dropout(tokens), meta)


class SinusoidalEncoding(nn.Module):
    def __init__(self, d_model: int, axes: Tuple[str, ...]) -> None:
        super().__init__()
        self.axes = axes
        assert len(axes) >= 1 and all((a in ("l", "t", "t_score", "h", "w") for a in axes))
        assert d_model % len(axes) == 0
        self.d_axis = d_model // len(axes)
        assert self.d_axis % 2 == 0
        self.register_buffer("_cache_meta", torch.tensor([-1, -1, -1], dtype=torch.int64), persistent=False)
        self.register_buffer("_cache_pe", torch.empty(0, 0), persistent=False)
        self._cache_device = None
        self._cache_dtype = None

    @staticmethod
    def _to_1d(n: int, dim: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(n, dim, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(
        self,
        meta: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        t, h, w = meta
        if (
            self._cache_device == device
            and self._cache_dtype == dtype
            and (self._cache_meta.numel() == 3)
            and (tuple((int(x) for x in self._cache_meta.tolist())) == (t, h, w))
        ):
            return self._cache_pe
        chunks: List[torch.Tensor] = []
        if "t" in self.axes:
            pt = self._to_1d(t, self.d_axis, device).view(t, 1, 1, self.d_axis)
            chunks.append(pt.expand(t, h, w, self.d_axis))
        if "h" in self.axes:
            ph = self._to_1d(h, self.d_axis, device).view(1, h, 1, self.d_axis)
            chunks.append(ph.expand(t, h, w, self.d_axis))
        if "w" in self.axes:
            pw = self._to_1d(w, self.d_axis, device).view(1, 1, w, self.d_axis)
            chunks.append(pw.expand(t, h, w, self.d_axis))
        if "l" in self.axes:
            l = t
            pl = self._to_1d(l, self.d_axis, device).view(l, 1, 1, self.d_axis)
            chunks.append(pl.expand(l, 1, 1, self.d_axis))
        pe = torch.cat(chunks, dim=-1).view(-1, self.d_axis * len(self.axes)).to(dtype=dtype)
        self._cache_meta = torch.tensor([t, h, w], dtype=torch.int64)
        self._cache_pe = pe
        self._cache_device = device
        self._cache_dtype = dtype
        return pe


class GeGLU(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hid = int(hidden_dim)
        self.out_dim = int(out_dim if out_dim is not None else in_dim)
        self.in_proj = nn.Linear(self.in_dim, 2 * self.hid, bias=bias)
        self.dropout = nn.Dropout(float(dropout))
        self.out_proj = nn.Linear(self.hid, self.out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.in_proj(x).chunk(2, dim=-1)
        y = a * F.gelu(b)
        y = self.dropout(y)
        return self.out_proj(y)


class SwiGLU(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hid = int(hidden_dim)
        self.out_dim = int(out_dim if out_dim is not None else in_dim)
        self.in_proj = nn.Linear(self.in_dim, 2 * self.hid, bias=bias)
        self.dropout = nn.Dropout(float(dropout))
        self.out_proj = nn.Linear(self.hid, self.out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.in_proj(x).chunk(2, dim=-1)
        y = a * F.silu(b)
        y = self.dropout(y)
        return self.out_proj(y)


class MultipleQuantileLoss(nn.Module):
    def __init__(
        self,
        quantiles: Sequence[float],
        weights: Optional[Sequence[float]] = None,
        quantile_dim: Optional[int] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        q = torch.tensor(list(quantiles), dtype=torch.float32)
        assert q.ndim == 1 and torch.all((q > 0) & (q < 1))
        self.register_buffer("q", q)
        if weights is None:
            w = torch.ones_like(q)
        else:
            w = torch.tensor(list(weights), dtype=torch.float32)
            assert w.shape == q.shape
        self.register_buffer("w", w / (w.sum() + 1e-12))
        self.quantile_dim = None if quantile_dim is None else int(quantile_dim)
        self.reduction = str(reduction)

    def _resolve_qdim(self, shape: torch.Size) -> int:
        Q = int(self.q.numel())
        if self.quantile_dim is not None:
            assert 0 <= self.quantile_dim < len(shape)
            assert shape[self.quantile_dim] == Q
            return self.quantile_dim
        candidates = [i for i, s in enumerate(shape) if s == Q]
        if 1 in candidates:
            return 1
        if 0 in candidates:
            return 0
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError("cannot infer quantile_dim")

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        if preds.shape != target.shape:
            raise ValueError("shape mismatch")
        qdim = self._resolve_qdim(preds.shape)
        if qdim != 0:
            preds = preds.transpose(0, qdim)
            target = target.transpose(0, qdim)
        errors = target - preds
        q = self.q.view(-1, *[1] * (errors.ndim - 1))
        losses = torch.maximum((q - 1) * errors, q * errors)
        if self.reduction == "none":
            return losses.transpose(0, qdim) if qdim != 0 else losses
        dims = tuple(range(1, losses.ndim))
        per_q = losses.mean(dim=dims) if self.reduction == "mean" else losses.sum(dim=dims)
        return (per_q * self.w).sum()


class StandardNormalLoss(nn.Module):
    _Number = Union[float, int]
    _TensorLike = Union[_Number, torch.Tensor]

    def __init__(
        self,
        *args: Any,
        confidence: float = 0.95,
        metric: str = "p_value",
        penalty: str = "soft",
        tau: float = 2.0,
        hinge_power: float = 2.0,
        dim: Optional[Tuple[int, ...]] = None,
        eps: float = 1e-06,
        reduction: str = "mean",
        mu_mode: str = "target",
        mu: Optional[_TensorLike] = None,
        std_mode: str = "target",
        std: Optional[_TensorLike] = None,
        two_tailed: bool = True,
        ddof: int = 0,
        clamp_max: Optional[float] = None,
        detach_stats: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__()
        assert 0.0 < confidence < 1.0
        assert metric.lower() in {"z", "z_score", "z_value", "zscore", "zvalue", "p", "p_value", "pvalue"}
        assert penalty.lower() in {"hinge", "tau", "soft", "softplus"}
        assert reduction.lower() in {"mean", "sum", "none"}
        self.confidence = float(confidence)
        self.metric = metric.lower()
        self.penalty = penalty.lower()
        self.tau = float(tau)
        self.hinge_power = float(hinge_power)
        self.dim = dim
        self.eps = float(eps)
        self.reduction = reduction.lower()
        self.mu_mode = mu_mode.lower()
        self.mu = mu
        self.std_mode = std_mode.lower()
        self.std = std
        self.two_tailed = bool(two_tailed)
        self.ddof = int(ddof)
        self.clamp_max = clamp_max
        self.detach_stats = bool(detach_stats)
        self._std_normal = Normal(loc=0.0, scale=1.0)

    @staticmethod
    def _to_tensor_like(x: _TensorLike, ref: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(x):
            t = x.to(device=ref.device, dtype=ref.dtype)
        else:
            t = torch.tensor(x, device=ref.device, dtype=ref.dtype)
        return t

    @staticmethod
    def _broadcast_param(x: _TensorLike, ref: torch.Tensor) -> torch.Tensor:
        t = StandardNormalLoss._to_tensor_like(x, ref)
        if t.ndim < ref.ndim:
            t = t.view(*[1] * (ref.ndim - t.ndim), *t.shape)
        return t

    @staticmethod
    def _safe_std(
        x: torch.Tensor,
        dim: Tuple[int, ...],
        ddof: int,
        eps: float,
    ) -> torch.Tensor:
        return _stable_std(x, dim, ddof, eps)

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        match self.reduction:
            case "mean":
                return x.mean()
            case "sum":
                return x.sum()
            case "none":
                return x
        return x

    def _get_dims(self, pred: torch.Tensor) -> Tuple[int, ...]:
        return _canon_dims(pred, self.dim, keep_batch=False)

    def _compute_mu(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        dims: Tuple[int, ...],
    ) -> torch.Tensor:
        dims = _canon_dims(pred, dims)
        match self.mu_mode:
            case "target":
                mu = target.mean(dim=dims, keepdim=True)
            case "pred":
                mu = pred.mean(dim=dims, keepdim=True)
            case "error":
                mu = (pred - target).mean(dim=dims, keepdim=True)
            case "provided":
                if self.mu is None:
                    raise ValueError("mu required")
                mu = self._broadcast_param(self.mu, pred)
            case "none":
                mu = torch.zeros(1, device=pred.device, dtype=pred.dtype)
            case _:
                raise ValueError("invalid mu_mode")
        if self.detach_stats:
            mu = mu.detach()
        return mu

    def _compute_std(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        dims: Tuple[int, ...],
    ) -> torch.Tensor:
        match self.std_mode:
            case "target":
                std = self._safe_std(target, dim=dims, ddof=self.ddof, eps=self.eps)
            case "pred":
                std = self._safe_std(pred, dim=dims, ddof=self.ddof, eps=self.eps)
            case "pooled":
                std_t = self._safe_std(target, dim=dims, ddof=self.ddof, eps=self.eps)
                std_p = self._safe_std(pred, dim=dims, ddof=self.ddof, eps=self.eps)
                std = torch.sqrt(torch.clamp(0.5 * (std_t**2 + std_p**2), min=self.eps * self.eps))
            case "provided":
                if self.std is None:
                    raise ValueError("std required")
                std = self._broadcast_param(self.std, pred)
                if self.detach_stats:
                    std = std.detach()
                return torch.clamp(std, min=self.eps)
            case "none":
                std = torch.ones(1, device=pred.device, dtype=pred.dtype)
            case _:
                raise ValueError("invalid std_mode")
        if self.detach_stats:
            std = std.detach()
        return torch.clamp(std, min=self.eps)

    def _z_threshold(self, device: Any, dtype: Any) -> torch.Tensor:
        q = 0.5 + 0.5 * self.confidence if self.two_tailed else self.confidence
        return self._std_normal.icdf(torch.tensor(q, device=device, dtype=dtype))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError("shape mismatch")
        dims = self._get_dims(pred)
        mu = self._compute_mu(pred, target, dims)
        std = self._compute_std(pred, target, dims)
        z_abs = ((pred - target - mu) / std).abs()
        if self.clamp_max is not None:
            z_abs = torch.clamp(z_abs, max=float(self.clamp_max))
        match self.metric:
            case "z" | "z_score" | "z_value" | "zscore" | "zvalue":
                margin = z_abs - self._z_threshold(pred.device, pred.dtype)
            case "p" | "p_value" | "pvalue":
                x = z_abs
                try:
                    one_tail = torch.clamp(1.0 - Normal(loc=0.0, scale=1.0).cdf(x), min=self.eps)
                except NotImplementedError:
                    one_tail = torch.clamp(1.0 - _normal_cdf_loc_scale(x, torch.tensor(0.0, device=x.device, dtype=x.dtype), torch.tensor(1.0, device=x.device, dtype=x.dtype)), min=self.eps)
                p = 2.0 * one_tail if self.two_tailed else one_tail
                alpha = max(1.0 - self.confidence, self.eps)
                margin = -torch.log(torch.clamp(p, min=self.eps)) + math.log(alpha)
            case _:
                raise ValueError("Invalid metric")
        match self.penalty:
            case "hinge":
                pen = torch.clamp(margin, min=0.0).pow(self.hinge_power)
            case "tau":
                tau = max(self.tau, self.eps)
                pen = torch.nn.functional.softplus(margin / tau) * tau
            case "soft" | "softplus":
                beta = max(self.tau, self.eps)
                pen = torch.nn.functional.softplus(beta * margin) / beta
            case _:
                raise ValueError("Invalid penalty")
        return self._reduce(pen)


class StudentsTLoss(nn.Module):
    _Number = Union[float, int]
    _TensorLike = Union[_Number, torch.Tensor]

    def __init__(
        self,
        *args: Any,
        confidence: float = 0.95,
        metric: str = "p_value",
        penalty: str = "soft",
        tau: float = 2.0,
        hinge_power: float = 2.0,
        dim: Optional[Tuple[int, ...]] = None,
        eps: float = 1e-06,
        reduction: str = "mean",
        mu_mode: str = "target",
        mu: Optional[_TensorLike] = None,
        std_mode: str = "target",
        std: Optional[_TensorLike] = None,
        df: _TensorLike = 3.0,
        two_tailed: bool = True,
        ddof: int = 0,
        clamp_max: Optional[float] = None,
        detach_stats: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__()
        assert 0.0 < confidence < 1.0
        assert metric.lower() in {"t", "t_score", "t_value", "tscore", "tvalue", "p", "p_value", "pvalue"}
        assert penalty.lower() in {"hinge", "tau", "soft", "softplus"}
        assert reduction.lower() in {"mean", "sum", "none"}
        self.confidence = float(confidence)
        self.metric = metric.lower()
        self.penalty = penalty.lower()
        self.tau = float(tau)
        self.hinge_power = float(hinge_power)
        self.dim = dim
        self.eps = float(eps)
        self.reduction = reduction.lower()
        self.mu_mode = mu_mode.lower()
        self.mu = mu
        self.std_mode = std_mode.lower()
        self.std = std
        self.df = df
        self.two_tailed = bool(two_tailed)
        self.ddof = int(ddof)
        self.clamp_max = clamp_max
        self.detach_stats = bool(detach_stats)

    @staticmethod
    def _to_tensor_like(x: _TensorLike, ref: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(x):
            t = x.to(device=ref.device, dtype=ref.dtype)
        else:
            t = torch.tensor(x, device=ref.device, dtype=ref.dtype)
        return t

    @staticmethod
    def _broadcast_param(x: _TensorLike, ref: torch.Tensor) -> torch.Tensor:
        t = StudentsTLoss._to_tensor_like(x, ref)
        if t.ndim < ref.ndim:
            t = t.view(*[1] * (ref.ndim - t.ndim), *t.shape)
        return t

    @staticmethod
    def _safe_std(
        x: torch.Tensor,
        dim: Tuple[int, ...],
        ddof: int,
        eps: float,
    ) -> torch.Tensor:
        return _stable_std(x, dim, ddof, eps)

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        match self.reduction:
            case "mean":
                return x.mean()
            case "sum":
                return x.sum()
            case "none":
                return x
        return x

    def _get_dims(self, pred: torch.Tensor) -> Tuple[int, ...]:
        if self.dim is None:
            if pred.ndim <= 1:
                return tuple()
            return tuple(range(1, pred.ndim))
        return self.dim

    def _compute_mu(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        dims: Tuple[int, ...],
    ) -> torch.Tensor:
        dims = _canon_dims(pred, dims)
        match self.mu_mode:
            case "target":
                mu = target.mean(dim=dims, keepdim=True)
            case "pred":
                mu = pred.mean(dim=dims, keepdim=True)
            case "pooled":
                mu = 0.5 * (target.mean(dim=dims, keepdim=True) + pred.mean(dim=dims, keepdim=True))
            case "provided":
                if self.mu is None:
                    raise ValueError("mu required when mu_mode='provided'")
                mu = self._broadcast_param(self.mu, pred)
            case "error":
                mu = (pred - target).mean(dim=dims, keepdim=True)
            case "none":
                mu = torch.zeros(1, device=pred.device, dtype=pred.dtype)
            case _:
                raise ValueError("invalid mu_mode")
        if self.detach_stats:
            mu = mu.detach()
        return mu

    def _compute_std(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        dims: Tuple[int, ...],
    ) -> torch.Tensor:
        match self.std_mode:
            case "target":
                std = self._safe_std(target, dim=dims, ddof=self.ddof, eps=self.eps)
            case "pred":
                std = self._safe_std(pred, dim=dims, ddof=self.ddof, eps=self.eps)
            case "pooled":
                std_t = self._safe_std(target, dim=dims, ddof=self.ddof, eps=self.eps)
                std_p = self._safe_std(pred, dim=dims, ddof=self.ddof, eps=self.eps)
                std = torch.sqrt(torch.clamp(0.5 * (std_t**2 + std_p**2), min=self.eps * self.eps))
            case "provided":
                if self.std is None:
                    raise ValueError("std required when std_mode='provided'")
                std = self._broadcast_param(self.std, pred)
                if self.detach_stats:
                    std = std.detach()
                return torch.clamp(std, min=self.eps)
            case "none":
                std = torch.ones(1, device=pred.device, dtype=pred.dtype)
            case _:
                raise ValueError("invalid std_mode")
        if self.detach_stats:
            std = std.detach()
        return torch.clamp(std, min=self.eps)

    def _t_threshold(self, device: Any, dtype: Any) -> torch.Tensor:
        q = 0.5 + 0.5 * self.confidence if self.two_tailed else self.confidence
        df = self._to_tensor_like(self.df, torch.empty((), device=device, dtype=dtype))
        try:
            dist = StudentT(df=df)
            return dist.icdf(torch.tensor(q, device=device, dtype=dtype))
        except NotImplementedError:
            target = torch.full_like(df, float(q), dtype=dtype, device=device)
            loc = torch.zeros_like(df, dtype=dtype, device=device)
            scale = torch.ones_like(df, dtype=dtype, device=device)
            lo = torch.full_like(df, -50.0, dtype=dtype, device=device)
            hi = torch.full_like(df, 50.0, dtype=dtype, device=device)
            for _ in range(32):
                mid = (lo + hi) / 2.0
                cdf_mid = _student_t_cdf_loc_scale(mid, df, loc, scale)
                mask = cdf_mid < target
                lo = torch.where(mask, mid, lo)
                hi = torch.where(mask, hi, mid)
            return hi

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError("shape mismatch")
        dims = self._get_dims(pred)
        mu = self._compute_mu(pred, target, dims)
        std = self._compute_std(pred, target, dims)
        t_abs = ((pred - target - mu) / std).abs()
        if self.clamp_max is not None:
            t_abs = torch.clamp(t_abs, max=float(self.clamp_max))
        match self.metric:
            case "t" | "t_score" | "t_value" | "tscore" | "tvalue":
                margin = t_abs - self._t_threshold(pred.device, pred.dtype)
            case "p" | "p_value" | "pvalue":
                df = self._broadcast_param(self.df, t_abs)
                df_safe = torch.clamp(df, min=2.0 + self.eps)
                scale = std * torch.sqrt(torch.clamp((df_safe - 2.0) / df_safe, min=self.eps))
                x = mu + t_abs * scale
                try:
                    one_tail = torch.clamp(1.0 - StudentT(df=df, loc=mu, scale=scale).cdf(x), min=self.eps)
                except NotImplementedError:
                    one_tail = torch.clamp(1.0 - _student_t_cdf_loc_scale(x, df, mu, scale), min=self.eps)
                p = 2.0 * one_tail if self.two_tailed else one_tail
                alpha = max(1.0 - self.confidence, self.eps)
                margin = -torch.log(torch.clamp(p, min=self.eps)) + math.log(alpha)
            case _:
                raise ValueError("Invalid metric")
        match self.penalty:
            case "hinge":
                pen = torch.clamp(margin, min=0.0).pow(self.hinge_power)
            case "tau":
                tau = max(self.tau, self.eps)
                pen = torch.nn.functional.softplus(margin / tau) * tau
            case "soft" | "softplus":
                beta = max(self.tau, self.eps)
                pen = torch.nn.functional.softplus(beta * margin) / beta
            case _:
                raise ValueError("Invalid penalty")
        return self._reduce(pen)


def _as_tuple(x: Any) -> Tuple[int, ...]:
    return tuple((int(v) for v in x))


def _fftn_nd(
    x: torch.Tensor,
    shape: Sequence[int],
    *args: Any,
    real_input: bool = False,
    inverse: bool = False,
    norm: Optional[str] = "ortho",
    **kwargs: Any
) -> torch.Tensor:
    dims = tuple(range(-len(shape), 0))
    if not inverse:
        if real_input:
            return torch.fft.rfftn(x, s=_as_tuple(shape), dim=dims, norm=norm)
        return torch.fft.fftn(x, s=_as_tuple(shape), dim=dims, norm=norm)
    else:
        if real_input:
            return torch.fft.irfftn(x, s=_as_tuple(shape), dim=dims, norm=norm)
        return torch.fft.ifftn(x, s=_as_tuple(shape), dim=dims, norm=norm)


def _nufft_nd_cufinufft(
    x_cplx: torch.Tensor,
    omega: torch.Tensor,
    shape: Sequence[int],
    *args: Any,
    nufft_type: int = 2,
    eps: float = 1e-06,
    **kwargs: Any
) -> torch.Tensor:
    try:
        import cufinufft
    except Exception as e:
        raise RuntimeError("cuFINUFFT not available") from e
    B = x_cplx.shape[0]
    ndim = len(shape)
    out_list = []
    if omega.dim() == 2:
        pts = [omega[i].contiguous() for i in range(ndim)]
        plan = cufinufft.Plan(nufft_type, _as_tuple(shape), n_trans=1, eps=eps, dtype="complex64")
        plan.setpts(*pts)
        for b in range(B):
            fk = plan.execute(x_cplx[b])
            out_list.append(fk.unsqueeze(0))
    else:
        for b in range(B):
            pts = [omega[b, i].contiguous() for i in range(ndim)]
            plan = cufinufft.Plan(nufft_type, _as_tuple(shape), n_trans=1, eps=eps, dtype="complex64")
            plan.setpts(*pts)
            fk = plan.execute(x_cplx[b])
            out_list.append(fk.unsqueeze(0))
    return torch.cat(out_list, dim=0)


class DataFidelityLoss(nn.Module):
    def __init__(
        self,
        out_shape: Sequence[int],
        *args: Any,
        mode: str = "fft",
        ktraj: Optional[torch.Tensor] = None,
        backend: Optional[str] = None,
        weight: float = 1.0,
        fft_norm: Optional[str] = "ortho",
        reduction: str = "mean",
        nufft_eps: float = 1e-06,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.out_shape = _as_tuple(out_shape)
        self.ndim = len(self.out_shape)
        self.mode = str(mode).lower()
        self.backend = None if backend is None else str(backend).lower()
        self.register_buffer("ktraj", ktraj if ktraj is not None else None, persistent=False)
        self.weight = float(weight)
        self.fft_norm = fft_norm
        self.reduction = reduction
        self.nufft_eps = float(nufft_eps)
        if self.mode not in ("fft", "nufft"):
            raise ValueError("mode must be 'fft' or 'nufft'")
        if self.mode == "nufft" and self.ktraj is None:
            raise ValueError("ktraj is required for NUFFT mode")

    def _mse(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = (a - b).abs() if torch.is_complex(a) or torch.is_complex(b) else a - b
        sq = diff * diff
        match self.reduction:
            case "mean":
                val = sq.mean()
            case "sum":
                val = sq.sum()
            case _:
                B = int(a.shape[0])
                val = sq.reshape(B, -1).mean(dim=1)
        return val * self.weight

    def forward(
        self,
        pred_flat: torch.Tensor,
        target_flat: torch.Tensor,
    ) -> torch.Tensor:
        B = int(pred_flat.shape[0])
        shape = self.out_shape
        x = pred_flat.view(B, *shape)
        y = target_flat.view(B, *shape)
        match self.mode:
            case "fft":
                Xk = _fftn_nd(x.to(torch.complex64) if not torch.is_complex(x) else x, shape, real_input=False, inverse=False, norm=self.fft_norm)
                Yk = _fftn_nd(y.to(torch.complex64) if not torch.is_complex(y) else y, shape, real_input=False, inverse=False, norm=self.fft_norm)
            case "nufft":
                match self.backend:
                    case None | "cufinufft":
                        try:
                            Xk = _nufft_nd_cufinufft(x.to(torch.complex64).unsqueeze(1), self.ktraj, shape, nufft_type=2, eps=self.nufft_eps)
                            Yk = _nufft_nd_cufinufft(y.to(torch.complex64).unsqueeze(1), self.ktraj, shape, nufft_type=2, eps=self.nufft_eps)
                        except Exception:
                            Xk = _fftn_nd(x.to(torch.complex64), shape, real_input=False, inverse=False, norm=self.fft_norm)
                            Yk = _fftn_nd(y.to(torch.complex64), shape, real_input=False, inverse=False, norm=self.fft_norm)
                    case "finufft":
                        raise NotImplementedError("FINUFFT path: wire with finufft.nufft*d* or Plan if you need CPU NUFFT.")
                    case _:
                        raise ValueError(f"Unknown NUFFT backend: {self.backend}")
            case _:
                raise ValueError(f"Invalid mode: {self.mode}")
        return self._mse(Xk, Yk)


class LinearCombinationLoss(nn.Module):
    def __init__(
        self,
        *args: Any,
        coefficient: Sequence[float],
        loss: Sequence[nn.Module],
        offset: float = 0.0,
        **kwargs: Any
    ) -> None:
        super().__init__()
        if not (isinstance(coefficient, (list, tuple)) and isinstance(loss, (list, tuple))):
            raise TypeError("coefficient/loss must be sequences")
        if len(coefficient) != len(loss) or len(loss) == 0:
            raise ValueError("invalid coefficient/loss length")
        self.register_buffer("coefficient", torch.tensor(list(coefficient), dtype=torch.float32))
        self.losses = nn.ModuleList(list(loss))
        self.offset = float(offset)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        total = pred.new_tensor(self.offset, dtype=pred.dtype)
        for w, L in zip(self.coefficient, self.losses):
            v = L(pred, target)
            if v.dim() > 0:
                v = v.mean()
            total = total + w.to(device=pred.device, dtype=pred.dtype) * v
        return total


class TiledLoss(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        *args: Any,
        mask_mode: str = "none",
        mask_value: Optional[float] = None,
        tile_dim: Optional[int] = None,
        tile_size: Optional[int] = None,
        reduction: str = "mean",
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.base = base
        self.mask_mode = str(mask_mode).lower()
        self.mask_value = mask_value
        self.tile_dim = tile_dim
        self.tile_size = int(tile_size) if tile_size is not None else None
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    def _make_mask(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        match self.mask_mode:
            case "none":
                return None
            case "finite":
                try:
                    return torch.isfinite(target)
                except Exception:
                    return None
            case "neq":
                if self.mask_value is None:
                    return None
                try:
                    return target != target.new_tensor(self.mask_value, dtype=target.dtype, device=target.device)
                except Exception:
                    return None
            case _: 
                return None

    def _reduce(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            x = x.masked_select(mask)
        match self.reduction:
            case "mean":
                return x.mean()
            case "sum":
                return x.sum()
            case "none":
                return x
        return x 

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = self._make_mask(pred, target)
        if self.tile_dim is None or self.tile_size is None:
            v = self.base(pred, target)
            return self._reduce(v, mask)
        nd = pred.ndim
        td = self.tile_dim + nd if self.tile_dim < 0 else self.tile_dim
        td = max(0, min(td, nd - 1))
        N = pred.shape[td]
        start = 0
        total_sum = pred.new_tensor(0.0, dtype=pred.dtype)
        total_count = pred.new_tensor(0.0, dtype=pred.dtype)
        parts: List[torch.Tensor] = []
        while start < N:
            end = min(N, start + self.tile_size)
            sl = [slice(None)] * nd
            sl[td] = slice(start, end)
            pv = pred[tuple(sl)]
            tv = target[tuple(sl)]
            mv = mask[tuple(sl)] if mask is not None else None
            elem = self.base(pv, tv)
            if self.reduction == "none":
                parts.append(elem if mv is None else elem.masked_select(mv))
            elif mv is not None:
                total_sum = total_sum + elem.masked_select(mv).sum()
                total_count = total_count + mv.to(dtype=pred.dtype).sum()
            else:
                total_sum = total_sum + elem.sum()
                total_count = total_count + elem.numel()
            start = end
        match self.reduction:
            case "none":
                return torch.cat(parts, dim=td) if parts else pred.new_zeros(())
            case "sum":
                return total_sum
            case "mean":
                denom = torch.clamp(total_count, min=1.0)
                return total_sum / denom
        return pred.new_zeros(()) 


class GatedCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.norm_q = _norm(norm_type, d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.kv_proj = nn.Linear(d_model, 2*d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(1))
        self.sdpa = ScaledDotProductAttention()

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Nq, D = q.shape
        qn = self.q_proj(self.norm_q(q))
        kv = self.kv_proj(kv)
        k, v = kv.chunk(2, dim=-1)
        def split(x: torch.Tensor) -> torch.Tensor:
            return x.view(B, -1, self.nhead, self.head_dim).transpose(1, 2)

        qh, kh, vh = split(qn), split(k), split(v)
        yh = self.sdpa(qh, kh, vh, attn_mask=attn_mask)
        y = yh.transpose(1, 2).contiguous().view(B, Nq, D)
        y = self.out_proj(self.dropout(y))
        return q + torch.sigmoid(self.gate) * y


class PatchAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        *,
        coord_dim: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead for PatchAttention")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = self.d_model // self.nhead
        self.coord_dim = int(coord_dim)
        self.norm1 = _norm(norm_type, self.d_model)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=True)
        self.rel_bias = nn.Sequential(
            nn.Linear(self.coord_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.nhead),
        )
        self.rel_value = nn.Sequential(
            nn.Linear(self.coord_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(p=drop_path, mode="row")
        self.norm2 = _norm(norm_type, self.d_model)
        hid = int(self.d_model * mlp_ratio * (2.0 / 3.0))
        self.ffn = SwiGLU(self.d_model, hid, out_dim=self.d_model, dropout=dropout)

    def _apply_attn_mask(self, scores: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attn_mask is None:
            return scores
        mask = attn_mask
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[-2:] != scores.shape[-2:]:
            raise ValueError("Attention mask shape mismatch in PatchAttention")
        return scores.masked_fill(~mask.to(dtype=torch.bool), float('-inf'))

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        if coords.shape[:2] != (B, N):
            raise ValueError("coords must have shape (B, N, C)")
        qkv = self.qkv(self.norm1(x))
        q, k, v = qkv.chunk(3, dim=-1)

        def _split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, N, self.nhead, self.head_dim).transpose(1, 2)

        qh, kh, vh = _split(q), _split(k), _split(v)
        rel = coords.unsqueeze(2) - coords.unsqueeze(1)
        rel_bias = self.rel_bias(rel).permute(0, 3, 1, 2)
        rel_value = self.rel_value(rel).view(B, N, N, self.nhead, self.head_dim).permute(0, 3, 1, 2, 4)
        scores = torch.einsum('bhid,bhjd->bhij', qh, kh) / math.sqrt(float(self.head_dim))
        scores = scores + rel_bias
        scores = self._apply_attn_mask(scores, attn_mask)
        weights = torch.softmax(scores, dim=-1)
        value = vh.unsqueeze(2).expand(-1, -1, N, -1, -1) + rel_value
        y = torch.einsum('bhij,bhijd->bhid', weights, value)
        y = y.transpose(1, 2).contiguous().view(B, N, D)
        x = x + self.drop_path(self.dropout(y))
        x = x + self.drop_path(self.dropout(self.ffn(self.norm2(x))))
        return x


class SpatialSubnet(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        depth: int,
        *,
        coord_dim: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        drops = _stochastic_depth_scheduler(drop_path, depth)
        self.blocks = nn.ModuleList(
            [
                PatchAttention(
                    d_model,
                    nhead,
                    coord_dim=coord_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drops[i],
                    norm_type=norm_type,
                )
                for i in range(depth)
            ]
        )
        self.norm = _norm(norm_type, d_model)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, coords, attn_mask=attn_mask)
        return self.norm(x)


class TemporalRetNet(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        self.norm1 = _norm(norm_type, d_model)
        self.msr = GatedMultiScaleRetention(d_model, nhead)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(p=drop_path, mode="row")
        self.norm2 = _norm(norm_type, d_model)
        hid = int(d_model * mlp_ratio * (2.0 / 3.0))
        self.ffn = SwiGLU(d_model, hid, out_dim=d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        h = self.msr(self.norm1(x), attn_mask=causal_mask, state=state)
        x = x + self.drop_path(self.dropout(h))
        x = x + self.drop_path(self.dropout(self.ffn(self.norm2(x))))
        return x, state


class TemporalSubnet(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        depth: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        drops = _stochastic_depth_scheduler(drop_path, depth)
        self.blocks = nn.ModuleList(
            [
                TemporalRetNet(
                    d_model,
                    nhead,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drops[i],
                    norm_type=norm_type,
                )
                for i in range(depth)
            ]
        )
        self.norm = _norm(norm_type, d_model)

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        state = None
        for blk in self.blocks:
            x, state = blk(x, causal_mask=causal_mask, state=state)
        return self.norm(x)


class CrossTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        *,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.cross_s = GatedCrossAttention(d_model, nhead, dropout=dropout, norm_type=norm_type)
        self.cross_t = GatedCrossAttention(d_model, nhead, dropout=dropout, norm_type=norm_type)
        self.mix_norm = _norm(norm_type, 2 * d_model)
        hid = int(2 * d_model * mlp_ratio * (2.0 / 3.0))
        self.mix = SwiGLU(2 * d_model, hid, out_dim=d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(p=drop_path, mode="row")

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        temporal_tokens: torch.Tensor,
        mode: str = "spatiotemporal",
    ) -> torch.Tensor:
        mode_l = _normalize_data_definition(mode)
        s_context = self.cross_s(spatial_tokens, temporal_tokens)
        t_context = self.cross_t(temporal_tokens, spatial_tokens)
        if mode_l == "sxs":
            return s_context
        if mode_l == "txt":
            return t_context
        if mode_l == "txs":
            base = torch.cat(
                [t_context, s_context.mean(dim=1, keepdim=True).expand(-1, t_context.size(1), -1)],
                dim=-1,
            )
            fused = self.mix(self.mix_norm(base))
            return t_context + self.drop_path(self.dropout(fused))
        base = torch.cat(
            [s_context, t_context.mean(dim=1, keepdim=True).expand(-1, s_context.size(1), -1)],
            dim=-1,
        )
        fused = self.mix(self.mix_norm(base))
        return s_context + self.drop_path(self.dropout(fused))


@dataclass
class Meta:
    tokens: torch.Tensor
    context: torch.Tensor
    flat: torch.Tensor
    offset: torch.Tensor
    context_shape: Tuple[int, ...]


class MetaNet(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        depth: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        drops = _stochastic_depth_scheduler(drop_path, depth)
        self.blocks = nn.ModuleList(
            [
                TemporalRetNet(
                    d_model,
                    nhead,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drops[i],
                    norm_type=norm_type,
                )
                for i in range(depth)
            ]
        )
        self.norm = _norm(norm_type, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        state = None
        for blk in self.blocks:
            tokens, state = blk(tokens, causal_mask=None, state=state)
        return self.norm(tokens)


class SpatioTemporalNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_shape: Sequence[int],
        *,
        config: Config,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_shape = tuple(int(v) for v in out_shape)
        self.out_dim = int(prod(self.out_shape))
        self.d_model = int(config.depth)
        self.nhead = int(config.heads)
        self.data_definition = _normalize_data_definition(config.data_definition)
        self.spatial_tokens = max(1, int(config.spatial_latent_tokens))
        self.temporal_tokens = max(1, int(config.temporal_latent_tokens))
        self.mlp_ratio = float(config.mlp_ratio)
        self.dropout = float(config.dropout)
        self.drop_path = float(config.drop_path)
        self.norm_type = str(config.normalize_method)

        self.spatial_tokenizer = nn.Linear(self.in_dim, self.spatial_tokens * self.d_model)
        self.temporal_tokenizer = nn.Linear(self.in_dim, self.temporal_tokens * self.d_model)
        self.register_buffer(
            "spatial_coords_template",
            self._build_spatial_coords(self.spatial_tokens, device=torch.device("cpu")),
            persistent=False,
        )

        self.spatial_subnet = SpatialSubnet(
            self.d_model,
            self.nhead,
            depth=max(1, int(config.spatial_depth)),
            coord_dim=self.spatial_coords_template.shape[-1],
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            drop_path=self.drop_path,
            norm_type=self.norm_type,
        )
        self.temporal_subnet = TemporalSubnet(
            self.d_model,
            self.nhead,
            depth=max(1, int(config.temporal_depth)),
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            drop_path=self.drop_path,
            norm_type=self.norm_type,
        )
        self.perception = CrossTransformer(
            self.d_model,
            self.nhead,
            dropout=self.dropout,
            norm_type=self.norm_type,
            mlp_ratio=self.mlp_ratio,
            drop_path=self.drop_path,
        )
        self.norm = _norm(self.norm_type, self.d_model)
        hid = int(self.d_model * max(1.0, self.mlp_ratio))
        self.head = nn.Sequential(
            _norm(self.norm_type, self.d_model),
            nn.Linear(self.d_model, hid),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hid, self.out_dim),
        )

    @staticmethod
    def _build_spatial_coords(n_tokens: int, device: torch.device) -> torch.Tensor:
        side = max(1, int(round(n_tokens ** (1.0 / 3.0))))
        coords: List[Tuple[float, float, float]] = []
        for idx in range(n_tokens):
            z = idx // (side * side)
            rem = idx % (side * side)
            y = rem // side
            x = rem % side
            if side == 1:
                coords.append((0.0, 0.0, 0.0))
            else:
                coords.append((x / (side - 1 if side > 1 else 1), y / (side - 1 if side > 1 else 1), z / (side - 1 if side > 1 else 1)))
        return torch.tensor(coords, dtype=torch.float32, device=device)

    def _spatial_coords(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        coords = self.spatial_coords_template.to(device=device, dtype=dtype)
        return coords.unsqueeze(0).expand(batch, -1, -1)

    def forward(self, x: torch.Tensor) -> Meta:
        B = x.shape[0]
        spatial_tokens = self.spatial_tokenizer(x).view(B, self.spatial_tokens, self.d_model)
        temporal_tokens = self.temporal_tokenizer(x).view(B, self.temporal_tokens, self.d_model)
        coords = self._spatial_coords(B, x.device, spatial_tokens.dtype)
        spatial_out = self.spatial_subnet(spatial_tokens, coords)
        temporal_out = self.temporal_subnet(temporal_tokens)
        mode = self.data_definition
        if mode == "sxs":
            tokens = spatial_out
        elif mode == "txt":
            tokens = temporal_out
        elif mode == "txs":
            tokens = self.perception(temporal_out, spatial_out, mode="txs")
        elif mode == "sxt":
            tokens = self.perception(spatial_out, temporal_out, mode="sxt")
        else:
            raise RuntimeError(f"Unhandled data definition '{mode}'")
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        flat = self.head(pooled)
        context = flat.view(B, *self.out_shape)
        dims = tuple(range(1, context.ndim))
        offset = context.mean(dim=dims, keepdim=True)
        return Meta(
            tokens=tokens,
            context=context,
            flat=flat,
            offset=offset,
            context_shape=self.out_shape,
        )

    def decode(self, tokens: torch.Tensor, *, apply_norm: bool = False) -> torch.Tensor:
        if apply_norm:
            tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        flat = self.head(pooled)
        return flat.view(tokens.shape[0], *self.out_shape)
