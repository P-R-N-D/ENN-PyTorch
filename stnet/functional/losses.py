# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.distributions import Normal, StudentT

Number = Union[float, int]
TensorLike = Union[Number, torch.Tensor]


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


def _median_over_dims(x: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
    dims = _canonize_dims(x, dims)
    m = x
    for d in dims:
        m, _ = m.median(dim=d, keepdim=True)
    return m


def _mad_std(
    x: torch.Tensor,
    dims: Tuple[int, ...],
    eps: float,
) -> torch.Tensor:
    mu = _median_over_dims(x, dims)
    dev = (x - mu).abs()
    mad = _median_over_dims(dev, dims)
    c = 1.482602218505602
    std = torch.clamp(mad * c, min=eps)
    return std


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
    x_work = x
    if torch.is_floating_point(x_work) and x_work.dtype != torch.float64:
        x_work = x_work.to(torch.float64)
    try:
        var = torch.var(x_work, dim=dim_tuple, correction=correction, keepdim=True)
        std = torch.sqrt(torch.clamp(var, min=float(eps) * float(eps)))
    except TypeError:
        std = torch.std(x_work, dim=dim_tuple, unbiased=correction == 1, keepdim=True)
        std = torch.clamp(std, min=float(eps))
    if torch.is_floating_point(x) and std.dtype != x.dtype:
        std = std.to(dtype=x.dtype)
    return std


def _to_tuple(x: Any) -> Tuple[int, ...]:
    return tuple((int(v) for v in x))


def _normal_cdf(
    x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    z = (x - loc) / torch.clamp(scale, min=1e-12)
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def _students_t_cdf(
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


def _fft_nd(
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
    if real_input:
        return torch.fft.irfftn(x, s=_to_tuple(shape), dim=dims, norm=norm)
    return torch.fft.ifftn(x, s=_to_tuple(shape), dim=dims, norm=norm)


def _nufft_nd(
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
    except ImportError as e:
        raise RuntimeError("cuFINUFFT not available") from e

    if not torch.is_complex(x_cplx):
        raise TypeError("x_cplx must be a complex-valued tensor")

    B = int(x_cplx.shape[0])
    ndim = len(shape)
    out_list: List[torch.Tensor] = []

    device = x_cplx.device
    if device.type != "cuda":
        raise RuntimeError(f"cuFINUFFT requires CUDA tensors, got device={device}")

    dtype_str = "complex128" if x_cplx.dtype == torch.complex128 else "complex64"

    if omega.dim() == 2:
        pts = [omega[i].contiguous() for i in range(ndim)]
        plan = cufinufft.Plan(
            nufft_type,
            _to_tuple(shape),
            n_trans=1,
            eps=eps,
            dtype=dtype_str,
        )
        plan.setpts(*pts)
        for b in range(B):
            fk = plan.execute(x_cplx[b].contiguous())
            out_list.append(torch.as_tensor(fk, device=device).unsqueeze(0))
    elif omega.dim() == 3:
        for b in range(B):
            pts = [omega[b, i].contiguous() for i in range(ndim)]
            plan = cufinufft.Plan(
                nufft_type,
                _to_tuple(shape),
                n_trans=1,
                eps=eps,
                dtype=dtype_str,
            )
            plan.setpts(*pts)
            fk = plan.execute(x_cplx[b].contiguous())
            out_list.append(torch.as_tensor(fk, device=device).unsqueeze(0))
    else:
        raise ValueError(
            f"omega must have shape (ndim, npts) or (B, ndim, npts), got {omega.shape}"
        )

    return torch.cat(out_list, dim=0)


def expand_to_pred(mask: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    if mask.shape == prediction.shape:
        return mask
    try:
        return mask.expand_as(prediction)
    except RuntimeError as e:
        raise ValueError(
            f"Mask shape {mask.shape} cannot be broadcast to prediction shape {prediction.shape}"
        ) from e


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


class CRPSLoss(nn.Module):

    def __init__(
        self,
        dim: Optional[Tuple[int, ...]] = None,
        eps: float = 1e-6,
        reduction: str = "mean",
        ddof: int = 0,
        std: Optional[TensorLike] = None,
        detach_stats: bool = True,
        mode: str = "normal",
        sample_dim: Optional[int] = None,
        max_z: float = 10.0,
    ) -> None:
        super().__init__()
        if reduction.lower() not in {"mean", "sum", "none"}:
            raise ValueError(f"invalid reduction={reduction}")
        mode_l = str(mode).lower()
        if mode_l not in {"normal", "energy"}:
            raise ValueError(f"CRPSLoss.mode must be 'normal' or 'energy', got {mode}")
        self.dim = dim
        self.eps = float(eps)
        self.reduction = reduction.lower()
        self.ddof = int(ddof)
        self.std_param = std
        self.detach_stats = bool(detach_stats)
        self.mode = mode_l
        self.sample_dim = None if sample_dim is None else int(sample_dim)
        self.max_z = float(max_z)
        self._skew_samples: int = 32
        self._skew_pair_samples: int = 8

    @staticmethod
    def _expand_params(x: TensorLike, ref: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(x):
            t = x.to(device=ref.device, dtype=ref.dtype)
        else:
            t = torch.tensor(x, device=ref.device, dtype=ref.dtype)
        if t.ndim < ref.ndim:
            t = t.view(*[1] * (ref.ndim - t.ndim), *t.shape)
        return t

    def _dims(self, pred: torch.Tensor) -> Tuple[int, ...]:
        return _canonize_dims(pred, self.dim, keep_batch=False)

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        match self.reduction:
            case "mean":
                return x.mean()
            case "sum":
                return x.sum()
            case "none":
                return x
        return x

    def _std_from_error(self, err: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
        std = _coerce_std(err, dim=dims, ddof=self.ddof, eps=self.eps)
        if self.detach_stats:
            std = std.detach()
        return torch.clamp(std, min=self.eps)

    def _crps_normal(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: pred={pred.shape}, target={target.shape}")
        dims = self._dims(pred)
        err = target - pred

        if self.std_param is not None:
            sigma = self._expand_params(self.std_param, pred)
            if self.detach_stats:
                sigma = sigma.detach()
            sigma = torch.clamp(sigma, min=self.eps)
        else:
            sigma = self._std_from_error(err, dims=dims)

        z_obs = err / sigma
        if self.max_z > 0.0:
            z_obs = torch.clamp(z_obs, min=-self.max_z, max=self.max_z)

        z_mean = z_obs.mean(dim=dims, keepdim=True)
        z_centered = z_obs - z_mean
        m2 = (z_centered**2).mean(dim=dims, keepdim=True).clamp(min=self.eps)
        m3 = (z_centered**3).mean(dim=dims, keepdim=True)
        skew_emp = (m3 / (m2.sqrt() ** 3 + self.eps))

        alpha = torch.clamp(skew_emp, min=-5.0, max=5.0)
        if self.detach_stats:
            alpha = alpha.detach()

        device, dtype = z_obs.device, z_obs.dtype
        n_samples = self._skew_samples

        delta = alpha / torch.sqrt(1.0 + alpha * alpha)
        z0 = torch.randn((n_samples, *z_obs.shape), device=device, dtype=dtype)
        z1 = torch.randn((n_samples, *z_obs.shape), device=device, dtype=dtype)
        while delta.ndim < z_obs.ndim:
            delta = delta.unsqueeze(0)
        base = delta * z0.abs()
        tail = torch.sqrt(torch.clamp(1.0 - delta * delta, min=0.0)) * z1
        z_samp = base + tail

        z_obs_exp = z_obs.unsqueeze(0)
        e1 = (z_samp - z_obs_exp).abs().mean(dim=0)

        pair_k = min(self._skew_pair_samples, n_samples)
        if pair_k >= 2:
            z_pair = z_samp[:pair_k]
            z1_s = z_pair.unsqueeze(1)
            z2_s = z_pair.unsqueeze(0)
            diff_pair = (z1_s - z2_s).abs()
            e2 = 0.5 * diff_pair.mean(dim=(0, 1))
        else:
            e2 = torch.zeros_like(e1)

        crps = sigma * (e1 - e2)
        crps = torch.clamp(crps, min=0.0)
        return self._reduce(crps)

    def _crps_energy(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.sample_dim is None:
            raise ValueError("CRPSLoss(mode='energy') requires sample_dim to be set")
        if target.ndim > pred.ndim:
            raise ValueError(
                f"target.ndim ({target.ndim}) cannot exceed pred.ndim ({pred.ndim}) in energy mode"
            )

        nd = pred.ndim
        sd = self.sample_dim if self.sample_dim >= 0 else self.sample_dim + nd
        if not (0 <= sd < nd):
            raise ValueError(f"invalid sample_dim={self.sample_dim} for pred.ndim={nd}")

        if sd != 1:
            perm = list(range(nd))
            perm[1], perm[sd] = perm[sd], perm[1]
            pred = pred.permute(*perm)

        B = int(pred.shape[0])
        S = int(pred.shape[1])
        if S <= 1:
            raise ValueError("energy mode requires at least 2 samples along sample_dim")

        V = int(pred[0, 0].numel())
        samples = pred.reshape(B, S, V)

        if target.ndim == pred.ndim - 1:
            target_flat = target.reshape(B, 1, V)
        else:
            raise ValueError(
                "In energy mode, target must be [B, ...] with the same tail shape as pred without sample_dim"
            )

        diff1 = samples - target_flat
        dist1 = diff1.norm(dim=-1)
        term1 = dist1.mean(dim=1)

        diff2 = samples.unsqueeze(2) - samples.unsqueeze(1)
        dist2 = diff2.norm(dim=-1)
        eye = torch.eye(S, device=dist2.device, dtype=torch.bool)
        mask = ~eye
        dist2 = dist2[:, mask].view(B, S * (S - 1))
        term2 = dist2.mean(dim=1)

        es = term1 - 0.5 * term2
        es = torch.clamp(es, min=0.0)
        return self._reduce(es)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.mode == "normal":
            return self._crps_normal(pred, target)
        return self._crps_energy(pred, target)


class DistributionLoss(nn.Module):
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
        mu: Optional[TensorLike] = None,
        std_mode: str = "target",
        std: Optional[TensorLike] = None,
        two_tailed: bool = True,
        ddof: int = 0,
        clamp_max: Optional[float] = None,
        detach_stats: bool = True,
        skew: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if not (0.0 < confidence < 1.0):
            raise ValueError(f"confidence must be in (0,1), got {confidence}")
        if penalty.lower() not in {"hinge", "tau", "soft", "softplus"}:
            raise ValueError(f"invalid penalty={penalty}")
        if reduction.lower() not in {"mean", "sum", "none"}:
            raise ValueError(f"invalid reduction={reduction}")

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
        self.skew = bool(skew)

    @staticmethod
    def _to_tensor_like(x: TensorLike, ref: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(x):
            t = x.to(device=ref.device, dtype=ref.dtype)
        else:
            t = torch.tensor(x, device=ref.device, dtype=ref.dtype)
        return t

    @staticmethod
    def _expand_params(x: TensorLike, ref: torch.Tensor) -> torch.Tensor:
        t = DistributionLoss._to_tensor_like(x, ref)
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
                x = target
                reducer = torch.median if self.skew else torch.mean
            case "pred":
                x = pred
                reducer = torch.median if self.skew else torch.mean
            case "pooled":
                if self.skew:
                    mu_t = _median_over_dims(target, dims)
                    mu_p = _median_over_dims(pred, dims)
                    mu = 0.5 * (mu_t + mu_p)
                    if self.detach_stats:
                        mu = mu.detach()
                    return mu
                else:
                    mu = 0.5 * (
                        target.mean(dim=dims, keepdim=True)
                        + pred.mean(dim=dims, keepdim=True)
                    )
                    if self.detach_stats:
                        mu = mu.detach()
                    return mu
            case "error":
                x = pred - target
                reducer = torch.median if self.skew else torch.mean
            case "provided":
                if self.mu is None:
                    raise ValueError("mu required when mu_mode='provided'")
                mu = self._expand_params(self.mu, pred)
                if self.detach_stats:
                    mu = mu.detach()
                return mu
            case "none":
                mu = torch.zeros(1, device=pred.device, dtype=pred.dtype)
                if self.detach_stats:
                    mu = mu.detach()
                return mu
            case _:
                raise ValueError("invalid mu_mode")

        if self.skew:
            mu = _median_over_dims(x, dims)
        else:
            mu = reducer(x, dim=dims, keepdim=True)
        if self.detach_stats:
            mu = mu.detach()
        return mu

    def compute_std(
        self, pred: torch.Tensor, target: torch.Tensor, dims: Tuple[int, ...]
    ) -> torch.Tensor:
        dims = _canonize_dims(pred, dims)
        match self.std_mode:
            case "target":
                std = (
                    _mad_std(target, dims=dims, eps=self.eps)
                    if self.skew
                    else DistributionLoss._safe_std(
                        target, dim=dims, ddof=self.ddof, eps=self.eps
                    )
                )
            case "pred":
                std = (
                    _mad_std(pred, dims=dims, eps=self.eps)
                    if self.skew
                    else DistributionLoss._safe_std(
                        pred, dim=dims, ddof=self.ddof, eps=self.eps
                    )
                )
            case "pooled":
                std_t = (
                    _mad_std(target, dims=dims, eps=self.eps)
                    if self.skew
                    else DistributionLoss._safe_std(
                        target, dim=dims, ddof=self.ddof, eps=self.eps
                    )
                )
                std_p = (
                    _mad_std(pred, dims=dims, eps=self.eps)
                    if self.skew
                    else DistributionLoss._safe_std(
                        pred, dim=dims, ddof=self.ddof, eps=self.eps
                    )
                )
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
                std = torch.clamp(std, min=self.eps)
                return std
            case "none":
                std = torch.ones(1, device=pred.device, dtype=pred.dtype)
            case _:
                raise ValueError("invalid std_mode")
        if self.detach_stats:
            std = std.detach()
        return torch.clamp(std, min=self.eps)

    def _statistic_abs(
        self, pred: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return ((pred - target - mu) / std).abs()

    def _compute_margin(
        self,
        stat_abs: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _apply_penalty(self, margin: torch.Tensor) -> torch.Tensor:
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
        return pen

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError("shape mismatch")
        dims = self._dims(pred)
        mu = self.compute_mu(pred, target, dims)
        std = self.compute_std(pred, target, dims)
        stat_abs = self._statistic_abs(pred, target, mu, std)
        if self.clamp_max is not None:
            stat_abs = torch.clamp(stat_abs, max=float(self.clamp_max))
        margin = self._compute_margin(stat_abs, pred, target, mu, std)
        pen = self._apply_penalty(margin)
        return self.reduce(pen)


class StandardNormalLoss(DistributionLoss):
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
        mu: Optional[TensorLike] = None,
        std_mode: str = "target",
        std: Optional[TensorLike] = None,
        two_tailed: bool = True,
        ddof: int = 0,
        clamp_max: Optional[float] = None,
        detach_stats: bool = True,
        skew: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            confidence=confidence,
            metric=metric,
            penalty=penalty,
            tau=tau,
            hinge_power=hinge_power,
            dim=dim,
            eps=eps,
            reduction=reduction,
            mu_mode=mu_mode,
            mu=mu,
            std_mode=std_mode,
            std=std,
            two_tailed=two_tailed,
            ddof=ddof,
            clamp_max=clamp_max,
            detach_stats=detach_stats,
            skew=skew,
            **kwargs,
        )
        valid = {
            "z",
            "z_score",
            "z_value",
            "zscore",
            "zvalue",
            "p",
            "p_value",
            "pvalue",
        }
        if self.metric not in valid:
            raise ValueError(f"Invalid metric for StandardNormalLoss: {self.metric}")
        self._std_normal = Normal(loc=0.0, scale=1.0)

    def _z_threshold(self, device: Any, dtype: Any) -> torch.Tensor:
        q = 0.5 + 0.5 * self.confidence if self.two_tailed else self.confidence
        return self._std_normal.icdf(torch.tensor(q, device=device, dtype=dtype))

    def _compute_margin(
        self,
        stat_abs: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        if self.skew:
            z = (pred - target - mu) / std
            if self.clamp_max is not None:
                c = float(self.clamp_max)
                z = torch.clamp(z, min=-c, max=c)
            dims = self._dims(pred)
            z_mean = z.mean(dim=dims, keepdim=True)
            z_centered = z - z_mean
            m2 = (z_centered**2).mean(dim=dims, keepdim=True).clamp(min=self.eps)
            m3 = (z_centered**3).mean(dim=dims, keepdim=True)
            gamma = m3 / (m2.sqrt() ** 3 + self.eps)
            if self.detach_stats:
                gamma = gamma.detach()
            z_eff = z + (gamma / 6.0) * (z * z - 1.0)
            z_abs = z_eff.abs()
        else:
            z_abs = stat_abs
        match self.metric:
            case "z" | "z_score" | "z_value" | "zscore" | "zvalue":
                return z_abs - self._z_threshold(pred.device, pred.dtype)
            case "p" | "p_value" | "pvalue":
                x = z_abs
                try:
                    one_tail = torch.clamp(
                        1.0 - Normal(loc=0.0, scale=1.0).cdf(x), min=self.eps
                    )
                except NotImplementedError:
                    one_tail = torch.clamp(
                        1.0
                        - _normal_cdf(
                            x,
                            torch.tensor(0.0, device=x.device, dtype=x.dtype),
                            torch.tensor(1.0, device=x.device, dtype=x.dtype),
                        ),
                        min=self.eps,
                    )
                p = 2.0 * one_tail if self.two_tailed else one_tail
                alpha = max(1.0 - self.confidence, self.eps)
                return -torch.log(torch.clamp(p, min=self.eps)) + math.log(alpha)
            case _:
                raise ValueError("Invalid metric")


class StudentsTLoss(DistributionLoss):
    _Number = Number
    _TensorLike = TensorLike

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
        skew: bool = False,
        **kwargs: Any,
    ) -> None:
        self.df = df
        super().__init__(
            confidence=confidence,
            metric=metric,
            penalty=penalty,
            tau=tau,
            hinge_power=hinge_power,
            dim=dim,
            eps=eps,
            reduction=reduction,
            mu_mode=mu_mode,
            mu=mu,
            std_mode=std_mode,
            std=std,
            two_tailed=two_tailed,
            ddof=ddof,
            clamp_max=clamp_max,
            detach_stats=detach_stats,
            skew=skew,
            **kwargs,
        )
        valid = {
            "t",
            "t_score",
            "t_value",
            "tscore",
            "tvalue",
            "p",
            "p_value",
            "pvalue",
        }
        if self.metric not in valid:
            raise ValueError(f"Invalid metric for StudentsTLoss: {self.metric}")
        self.df = df

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

    def _dims(self, pred: torch.Tensor) -> Tuple[int, ...]:
        if self.dim is None:
            if pred.ndim <= 1:
                return tuple()
            return tuple(range(1, pred.ndim))
        return self.dim

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
                cdf_mid = _students_t_cdf(mid, df, loc, scale)
                mask = cdf_mid < target
                lo = torch.where(mask, mid, lo)
                hi = torch.where(mask, hi, mid)
            return hi

    def _compute_margin(
        self,
        stat_abs: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        if self.skew:
            t = (pred - target - mu) / std
            if self.clamp_max is not None:
                c = float(self.clamp_max)
                t = torch.clamp(t, min=-c, max=c)
            dims = self._dims(pred)
            t_mean = t.mean(dim=dims, keepdim=True)
            t_centered = t - t_mean
            m2 = (t_centered**2).mean(dim=dims, keepdim=True).clamp(min=self.eps)
            m3 = (t_centered**3).mean(dim=dims, keepdim=True)
            gamma = m3 / (m2.sqrt() ** 3 + self.eps)
            if self.detach_stats:
                gamma = gamma.detach()
            t_eff = t + (gamma / 6.0) * (t * t - 1.0)
            t_abs = t_eff.abs()
        else:
            t_abs = stat_abs
        match self.metric:
            case "t" | "t_score" | "t_value" | "tscore" | "tvalue":
                return t_abs - self._t_threshold(pred.device, pred.dtype)
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
                        1.0 - _students_t_cdf(x, df, mu, scale),
                        min=self.eps,
                    )
                p = 2.0 * one_tail if self.two_tailed else one_tail
                alpha = max(1.0 - self.confidence, self.eps)
                return -torch.log(torch.clamp(p, min=self.eps)) + math.log(alpha)
            case _:
                raise ValueError("Invalid metric")


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
                Xk = _fft_nd(
                    x.to(torch.complex64) if not torch.is_complex(x) else x,
                    shape,
                    real_input=False,
                    inverse=False,
                    norm=self.fft_norm,
                )
                Yk = _fft_nd(
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
                            Xk = _nufft_nd(
                                x.to(torch.complex64).unsqueeze(1),
                                self.ktraj,
                                shape,
                                nufft_type=2,
                                eps=self.nufft_eps,
                            )
                            Yk = _nufft_nd(
                                y.to(torch.complex64).unsqueeze(1),
                                self.ktraj,
                                shape,
                                nufft_type=2,
                                eps=self.nufft_eps,
                            )
                        except RuntimeError as e:
                            if "cuFINUFFT not available" in str(e) or "requires CUDA tensors" in str(e):
                                Xk = _fft_nd(
                                    x.to(torch.complex64),
                                    shape,
                                    real_input=False,
                                    inverse=False,
                                    norm=self.fft_norm,
                                )
                                Yk = _fft_nd(
                                    y.to(torch.complex64),
                                    shape,
                                    real_input=False,
                                    inverse=False,
                                    norm=self.fft_norm,
                                )
                            else:
                                raise
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
        reduce_each: bool = True,
        auto_schedule: bool = False,
        schedule_momentum: float = 0.9,
        min_coeff: float = 0.05,
        max_coeff: float = 0.95,
        eps: float = 1e-06,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if not isinstance(coefficient, Sequence):
            raise TypeError(f"coefficient must be a Sequence, got {type(coefficient)}")
        if not isinstance(loss, Sequence):
            raise TypeError(f"loss must be a Sequence[nn.Module], got {type(loss)}")
        if len(coefficient) == 0 or len(coefficient) != len(loss):
            raise ValueError(
                f"invalid coefficient/loss length: "
                f"len(coefficient)={len(coefficient)}, len(loss)={len(loss)}"
            )

        coeff_tensor = torch.as_tensor(coefficient, dtype=torch.float32)
        if coeff_tensor.ndim != 1 or coeff_tensor.numel() != len(loss):
            raise ValueError(
                f"coefficient must be 1D of length {len(loss)}, got shape {coeff_tensor.shape}"
            )

        self.register_buffer("coefficient", coeff_tensor)
        self.losses = nn.ModuleList(list(loss))
        self.offset = float(offset)
        self.reduce_each = bool(reduce_each)

        self.auto_schedule = bool(auto_schedule)
        self.schedule_momentum = float(schedule_momentum)
        self.min_coeff = float(min_coeff)
        self.max_coeff = float(max_coeff)
        self.eps = float(eps)

        loss_avg_init = torch.full_like(coeff_tensor, fill_value=1.0)
        self.register_buffer("loss_avg", loss_avg_init)

    def _update_coefficients(self, per_loss_vals: List[torch.Tensor]) -> None:
        if not self.auto_schedule or not per_loss_vals:
            return

        with torch.no_grad():
            vals: List[float] = []
            for v in per_loss_vals:
                try:
                    val = float(v.detach().abs().mean().item())
                except Exception:
                    val = float(self.eps)
                vals.append(max(val, self.eps))

            avg = self.loss_avg
            device = avg.device
            dtype = avg.dtype
            vals_t = torch.tensor(vals, device=device, dtype=dtype)

            m = max(0.0, min(1.0, float(self.schedule_momentum)))
            avg.mul_(m).add_(vals_t * (1.0 - m))

            base = self.coefficient.detach().to(device=device, dtype=dtype)
            base_sum = float(base.sum().item())
            if not math.isfinite(base_sum) or base_sum <= 0.0:
                base = torch.ones_like(base)
                base_sum = float(base.numel())
            base = base / base_sum

            inv = base / torch.clamp(avg, min=self.eps)
            inv_sum = float(inv.sum().item())
            if not math.isfinite(inv_sum) or inv_sum <= 0.0:
                new_w = torch.full_like(inv, fill_value=1.0 / float(inv.numel()))
            else:
                new_w = inv / inv_sum

            if self.min_coeff > 0.0 or self.max_coeff < 1.0:
                lo = float(self.min_coeff)
                hi = float(self.max_coeff)
                new_w = torch.clamp(new_w, min=lo, max=hi)
                denom = float(new_w.sum().item())
                if denom > 0.0 and math.isfinite(denom):
                    new_w = new_w / denom

            self.coefficient.copy_(new_w.to(self.coefficient.device))

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        weights = self.coefficient.clone().to(device=pred.device, dtype=pred.dtype)
        total = pred.new_tensor(self.offset, dtype=pred.dtype)

        per_loss_vals: List[torch.Tensor] = []

        for w, L in zip(weights, self.losses):
            v = L(pred, target)
            if not torch.is_tensor(v):
                raise TypeError(
                    f"Loss module {L.__class__.__name__} must return a Tensor, "
                    f"got {type(v)}"
                )
            v_eff = v
            if self.reduce_each and v_eff.dim() > 0:
                v_eff = v_eff.mean()
            per_loss_vals.append(v_eff)
            total = total + w * v_eff

        self._update_coefficients(per_loss_vals)

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
                    return expand_to_pred(torch.isfinite(target), pred)
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
                    return expand_to_pred(target != base, pred)
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
                flat = elem.reshape(-1)
                if mv is not None:
                    flat_mask = mv.reshape(-1)
                    flat = flat[flat_mask]
                parts.append(flat)
            elif mv is not None:
                selected = elem.masked_select(mv)
                total_sum = total_sum + selected.sum()
                total_count = total_count + selected.numel()
            else:
                total_sum = total_sum + elem.sum()
                total_count = total_count + elem.numel()
            start = end
        match self.reduction:
            case "none":
                return torch.cat(parts, dim=0) if parts else pred.new_zeros(())
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
    top_avg: float = 0.8
    bottom_avg: float = 0.2

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
