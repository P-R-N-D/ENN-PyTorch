# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.distributions import Normal, StudentT


def _expand_to_pred(mask: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    try:
        if mask.shape != prediction.shape:
            mask = mask.expand_as(prediction)
    except Exception:
        pass
    return mask


def _canonize_dims(
    x: torch.Tensor, dims: Any, keep_batch: bool = False
) -> Tuple[int, ...]:
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


def _coerce_std(
    x: torch.Tensor, dim: Tuple[int, ...] | int | None, ddof: int, eps: float
) -> torch.Tensor:
    if dim is None:
        dim_tuple: Tuple[int, ...] = tuple()
    elif isinstance(dim, int):
        dim_tuple = (dim,)
    else:
        dim_tuple = tuple(dim)
    dims = tuple((d if d >= 0 else x.dim() + d for d in dim_tuple))
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


def normal_cdf(x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    z = (x - loc) / torch.clamp(scale, min=1e-12)
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def students_t_cdf(
    x: torch.Tensor, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
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

    def _resolve_q_dim(self, shape: torch.Size) -> int:
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
        qdim = self._resolve_q_dim(preds.shape)
        if qdim != 0:
            preds = preds.transpose(0, qdim)
            target = target.transpose(0, qdim)
        errors = target - preds
        q = self.q.view(-1, *[1] * (errors.ndim - 1))
        losses = torch.maximum((q - 1) * errors, q * errors)
        if self.reduction == "none":
            return losses.transpose(0, qdim) if qdim != 0 else losses
        dims = tuple(range(1, losses.ndim))
        per_q = (
            losses.mean(dim=dims) if self.reduction == "mean" else losses.sum(dim=dims)
        )
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
        **kwargs: Any,
    ) -> None:
        super().__init__()
        assert 0.0 < confidence < 1.0
        assert metric.lower() in {
            "z",
            "z_score",
            "z_value",
            "zscore",
            "zvalue",
            "p",
            "p_value",
            "pvalue",
        }
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
    def _expand_params(x: _TensorLike, ref: torch.Tensor) -> torch.Tensor:
        t = StandardNormalLoss._to_tensor_like(x, ref)
        if t.ndim < ref.ndim:
            t = t.view(*[1] * (ref.ndim - t.ndim), *t.shape)
        return t

    @staticmethod
    def _safe_std(
        x: torch.Tensor, dim: Tuple[int, ...], ddof: int, eps: float
    ) -> torch.Tensor:
        return _coerce_std(x, dim, ddof, eps)

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        match self.reduction:
            case "mean":
                return x.mean()
            case "sum":
                return x.sum()
            case "none":
                return x
        return x

    def _dims(self, pred: torch.Tensor) -> Tuple[int, ...]:
        return _canonize_dims(pred, self.dim, keep_batch=False)

    def compute_mu(
        self, pred: torch.Tensor, target: torch.Tensor, dims: Tuple[int, ...]
    ) -> torch.Tensor:
        dims = _canonize_dims(pred, dims)
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
                mu = self._expand_params(self.mu, pred)
            case "none":
                mu = torch.zeros(1, device=pred.device, dtype=pred.dtype)
            case _:
                raise ValueError("invalid mu_mode")
        if self.detach_stats:
            mu = mu.detach()
        return mu

    def compute_std(
        self, pred: torch.Tensor, target: torch.Tensor, dims: Tuple[int, ...]
    ) -> torch.Tensor:
        match self.std_mode:
            case "target":
                std = self._safe_std(target, dim=dims, ddof=self.ddof, eps=self.eps)
            case "pred":
                std = self._safe_std(pred, dim=dims, ddof=self.ddof, eps=self.eps)
            case "pooled":
                std_t = self._safe_std(target, dim=dims, ddof=self.ddof, eps=self.eps)
                std_p = self._safe_std(pred, dim=dims, ddof=self.ddof, eps=self.eps)
                std = torch.sqrt(
                    torch.clamp(
                        0.5 * (std_t**2 + std_p**2),
                        min=self.eps * self.eps,
                    )
                )
            case "provided":
                if self.std is None:
                    raise ValueError("std required")
                std = self._expand_params(self.std, pred)
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
        dims = self._dims(pred)
        mu = self.compute_mu(pred, target, dims)
        std = self.compute_std(pred, target, dims)
        z_abs = ((pred - target - mu) / std).abs()
        if self.clamp_max is not None:
            z_abs = torch.clamp(z_abs, max=float(self.clamp_max))
        match self.metric:
            case "z" | "z_score" | "z_value" | "zscore" | "zvalue":
                margin = z_abs - self._z_threshold(pred.device, pred.dtype)
            case "p" | "p_value" | "pvalue":
                x = z_abs
                try:
                    one_tail = torch.clamp(
                        1.0 - Normal(loc=0.0, scale=1.0).cdf(x), min=self.eps
                    )
                except NotImplementedError:
                    one_tail = torch.clamp(
                        1.0
                        - normal_cdf(
                            x,
                            torch.tensor(0.0, device=x.device, dtype=x.dtype),
                            torch.tensor(1.0, device=x.device, dtype=x.dtype),
                        ),
                        min=self.eps,
                    )
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
        return self.reduce(pen)


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
        **kwargs: Any,
    ) -> None:
        super().__init__()
        assert 0.0 < confidence < 1.0
        assert metric.lower() in {
            "t",
            "t_score",
            "t_value",
            "tscore",
            "tvalue",
            "p",
            "p_value",
            "pvalue",
        }
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
    def _expand_params(x: _TensorLike, ref: torch.Tensor) -> torch.Tensor:
        t = StudentsTLoss._to_tensor_like(x, ref)
        if t.ndim < ref.ndim:
            t = t.view(*[1] * (ref.ndim - t.ndim), *t.shape)
        return t

    @staticmethod
    def _safe_std(
        x: torch.Tensor, dim: Tuple[int, ...], ddof: int, eps: float
    ) -> torch.Tensor:
        return _coerce_std(x, dim, ddof, eps)

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        match self.reduction:
            case "mean":
                return x.mean()
            case "sum":
                return x.sum()
            case "none":
                return x
        return x

    def _dims(self, pred: torch.Tensor) -> Tuple[int, ...]:
        if self.dim is None:
            if pred.ndim <= 1:
                return tuple()
            return tuple(range(1, pred.ndim))
        return self.dim

    def compute_mu(
        self, pred: torch.Tensor, target: torch.Tensor, dims: Tuple[int, ...]
    ) -> torch.Tensor:
        dims = _canonize_dims(pred, dims)
        match self.mu_mode:
            case "target":
                mu = target.mean(dim=dims, keepdim=True)
            case "pred":
                mu = pred.mean(dim=dims, keepdim=True)
            case "pooled":
                mu = 0.5 * (
                    target.mean(dim=dims, keepdim=True)
                    + pred.mean(dim=dims, keepdim=True)
                )
            case "provided":
                if self.mu is None:
                    raise ValueError("mu required when mu_mode='provided'")
                mu = self._expand_params(self.mu, pred)
            case "error":
                mu = (pred - target).mean(dim=dims, keepdim=True)
            case "none":
                mu = torch.zeros(1, device=pred.device, dtype=pred.dtype)
            case _:
                raise ValueError("invalid mu_mode")
        if self.detach_stats:
            mu = mu.detach()
        return mu

    def compute_std(
        self, pred: torch.Tensor, target: torch.Tensor, dims: Tuple[int, ...]
    ) -> torch.Tensor:
        match self.std_mode:
            case "target":
                std = self._safe_std(target, dim=dims, ddof=self.ddof, eps=self.eps)
            case "pred":
                std = self._safe_std(pred, dim=dims, ddof=self.ddof, eps=self.eps)
            case "pooled":
                std_t = self._safe_std(target, dim=dims, ddof=self.ddof, eps=self.eps)
                std_p = self._safe_std(pred, dim=dims, ddof=self.ddof, eps=self.eps)
                std = torch.sqrt(
                    torch.clamp(
                        0.5 * (std_t**2 + std_p**2),
                        min=self.eps * self.eps,
                    )
                )
            case "provided":
                if self.std is None:
                    raise ValueError("std required when std_mode='provided'")
                std = self._expand_params(self.std, pred)
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
                cdf_mid = students_t_cdf(mid, df, loc, scale)
                mask = cdf_mid < target
                lo = torch.where(mask, mid, lo)
                hi = torch.where(mask, hi, mid)
            return hi

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError("shape mismatch")
        dims = self._dims(pred)
        mu = self.compute_mu(pred, target, dims)
        std = self.compute_std(pred, target, dims)
        t_abs = ((pred - target - mu) / std).abs()
        if self.clamp_max is not None:
            t_abs = torch.clamp(t_abs, max=float(self.clamp_max))
        match self.metric:
            case "t" | "t_score" | "t_value" | "tscore" | "tvalue":
                margin = t_abs - self._t_threshold(pred.device, pred.dtype)
            case "p" | "p_value" | "pvalue":
                df = self._expand_params(self.df, t_abs)
                df_safe = torch.clamp(df, min=2.0 + self.eps)
                scale = std * torch.sqrt(
                    torch.clamp((df_safe - 2.0) / df_safe, min=self.eps)
                )
                x = mu + t_abs * scale
                try:
                    one_tail = torch.clamp(
                        1.0 - StudentT(df=df, loc=mu, scale=scale).cdf(x),
                        min=self.eps,
                    )
                except NotImplementedError:
                    one_tail = torch.clamp(
                        1.0 - students_t_cdf(x, df, mu, scale),
                        min=self.eps,
                    )
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
        return self.reduce(pen)


def _to_tuple(x: Any) -> Tuple[int, ...]:
    return tuple((int(v) for v in x))


def fft_nd(
    x: torch.Tensor,
    shape: Sequence[int],
    *args: Any,
    real_input: bool = False,
    inverse: bool = False,
    norm: Optional[str] = "ortho",
    **kwargs: Any,
) -> torch.Tensor:
    dims = tuple(range(-len(shape), 0))
    if not inverse:
        if real_input:
            return torch.fft.rfftn(x, s=_to_tuple(shape), dim=dims, norm=norm)
        return torch.fft.fftn(x, s=_to_tuple(shape), dim=dims, norm=norm)
    else:
        if real_input:
            return torch.fft.irfftn(x, s=_to_tuple(shape), dim=dims, norm=norm)
        return torch.fft.ifftn(x, s=_to_tuple(shape), dim=dims, norm=norm)


def nufft_nd(
    x_cplx: torch.Tensor,
    omega: torch.Tensor,
    shape: Sequence[int],
    *args: Any,
    nufft_type: int = 2,
    eps: float = 1e-06,
    **kwargs: Any,
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
        plan = cufinufft.Plan(
            nufft_type, _to_tuple(shape), n_trans=1, eps=eps, dtype="complex64"
        )
        plan.setpts(*pts)
        for b in range(B):
            fk = plan.execute(x_cplx[b])
            out_list.append(fk.unsqueeze(0))
    else:
        for b in range(B):
            pts = [omega[b, i].contiguous() for i in range(ndim)]
            plan = cufinufft.Plan(
                nufft_type,
                _to_tuple(shape),
                n_trans=1,
                eps=eps,
                dtype="complex64",
            )
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
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.out_shape = _to_tuple(out_shape)
        self.ndim = len(self.out_shape)
        self.mode = str(mode).lower()
        self.backend = None if backend is None else str(backend).lower()
        self.register_buffer(
            "ktraj", ktraj if ktraj is not None else None, persistent=False
        )
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
        self, pred_flat: torch.Tensor, target_flat: torch.Tensor
    ) -> torch.Tensor:
        B = int(pred_flat.shape[0])
        shape = self.out_shape
        x = pred_flat.view(B, *shape)
        y = target_flat.view(B, *shape)
        match self.mode:
            case "fft":
                Xk = fft_nd(
                    x.to(torch.complex64) if not torch.is_complex(x) else x,
                    shape,
                    real_input=False,
                    inverse=False,
                    norm=self.fft_norm,
                )
                Yk = fft_nd(
                    y.to(torch.complex64) if not torch.is_complex(y) else y,
                    shape,
                    real_input=False,
                    inverse=False,
                    norm=self.fft_norm,
                )
            case "nufft":
                match self.backend:
                    case None | "cufinufft":
                        try:
                            Xk = nufft_nd(
                                x.to(torch.complex64).unsqueeze(1),
                                self.ktraj,
                                shape,
                                nufft_type=2,
                                eps=self.nufft_eps,
                            )
                            Yk = nufft_nd(
                                y.to(torch.complex64).unsqueeze(1),
                                self.ktraj,
                                shape,
                                nufft_type=2,
                                eps=self.nufft_eps,
                            )
                        except Exception:
                            Xk = fft_nd(
                                x.to(torch.complex64),
                                shape,
                                real_input=False,
                                inverse=False,
                                norm=self.fft_norm,
                            )
                            Yk = fft_nd(
                                y.to(torch.complex64),
                                shape,
                                real_input=False,
                                inverse=False,
                                norm=self.fft_norm,
                            )
                    case "finufft":
                        raise NotImplementedError(
                            (
                                "FINUFFT path: wire with finufft.nufft*d* "
                                "or Plan if you need CPU NUFFT."
                            )
                        )
                    case _:
                        raise ValueError(f"Unknown NUFFT backend: {self.backend}")
            case _:
                raise ValueError(f"Invalid mode: {self.mode}")
        return self._mse(Xk, Yk)


class LinearCombinationLoss(nn.Module):
    def __init__(
        self,
        coefficient: Sequence[float],
        loss: Sequence[nn.Module],
        *args: Any,
        offset: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if not (
            isinstance(coefficient, (list, tuple)) and isinstance(loss, (list, tuple))
        ):
            raise TypeError("coefficient/loss must be sequences")
        if len(coefficient) != len(loss) or len(loss) == 0:
            raise ValueError("invalid coefficient/loss length")
        self.register_buffer(
            "coefficient", torch.tensor(list(coefficient), dtype=torch.float32)
        )
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
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.base = base
        self.mask_mode = str(mask_mode).lower()
        self.mask_value = mask_value
        self.tile_dim = tile_dim
        self.tile_size = int(tile_size) if tile_size is not None else None
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    def _mask(self, pred: torch.Tensor, target: torch.Tensor) -> Optional[torch.Tensor]:
        match self.mask_mode:
            case "none":
                return None
            case "finite":
                try:
                    return _expand_to_pred(torch.isfinite(target), pred)
                except Exception:
                    return None
            case "neq":
                if self.mask_value is None:
                    return None
                try:
                    base = target.new_tensor(
                        self.mask_value,
                        dtype=target.dtype,
                        device=target.device,
                    )
                    return _expand_to_pred(target != base, pred)
                except Exception:
                    return None
            case _:
                return None

    def reduce(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
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
        mask = self._mask(pred, target)
        if self.tile_dim is None or self.tile_size is None:
            v = self.base(pred, target)
            return self.reduce(v, mask)
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


@dataclass
class LossWeightController:
    momentum: float = 0.9
    min_weight: float = 0.05
    max_weight: float = 0.95
    eps: float = 1e-06
    top_avg: float = 1.0
    bottom_avg: float = 1.0

    def weights(self) -> Tuple[float, float]:
        top = max(self.eps, self.top_avg)
        bottom = max(self.eps, self.bottom_avg)
        total = top + bottom
        if total <= 0.0:
            return (0.5, 0.5)
        ratio_top = top / total
        ratio_bottom = bottom / total
        ratio_top = float(min(max(ratio_top, self.min_weight), self.max_weight))
        ratio_bottom = float(min(max(ratio_bottom, self.min_weight), self.max_weight))
        norm = ratio_top + ratio_bottom
        if norm <= 0.0:
            return (0.5, 0.5)
        return (ratio_top / norm, ratio_bottom / norm)

    def update(
        self,
        top_loss: Optional[torch.Tensor],
        bottom_loss: Optional[torch.Tensor],
    ) -> None:
        if top_loss is not None:
            top_val = float(top_loss.detach().abs().mean().item())
            self.top_avg = self.momentum * self.top_avg + (1.0 - self.momentum) * max(
                top_val, self.eps
            )
        if bottom_loss is not None:
            bottom_val = float(bottom_loss.detach().abs().mean().item())
            self.bottom_avg = self.momentum * self.bottom_avg + (
                1.0 - self.momentum
            ) * max(bottom_val, self.eps)
