# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.distributions import Normal, StudentT
from torch.nn import functional as F

from ..core.graph import is_compiling


Number = Union[float, int]
TensorLike = Union[Number, torch.Tensor]


def _canonize_dims(x: torch.Tensor, dims: Any, keep_batch: bool = False) -> Tuple[int, ...]:
    nd = int(x.ndim)
    if dims is None:
        return tuple(range(1, nd)) if nd > 1 else (0,)
    d_tup = (dims,) if isinstance(dims, int) else tuple(int(d) for d in dims)
    pos = {d + nd if d < 0 else d for d in d_tup}
    out = tuple(sorted(d for d in pos if 0 <= d < nd and (keep_batch or d != 0)))
    return out or ((1,) if nd > 1 else (0,))


def _median_over_dims(x: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
    for d in _canonize_dims(x, dims):
        x = x.median(dim=d, keepdim=True).values
    return x


def _mad_std(x: torch.Tensor, dims: Tuple[int, ...], eps: float) -> torch.Tensor:
    return torch.clamp(
        _median_over_dims((x - _median_over_dims(x, dims)).abs(), dims) * 1.482602218505602,
        min=float(eps),
    )


def _master_float_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def _coerce_std(
    x: torch.Tensor, dim: Tuple[int, ...] | int | None, ddof: int, eps: float
) -> torch.Tensor:
    dim_tuple = (
        () if dim is None else (int(dim),) if isinstance(dim, int) else tuple(int(d) for d in dim)
    )
    nd = int(x.dim())
    sample = math.prod(
        max(1, int(x.shape[d if d >= 0 else nd + d]))
        for d in dim_tuple
        if 0 <= (d if d >= 0 else nd + d) < nd
    )
    if sample <= 1:
        return torch.full_like(
            x.mean(dim=dim_tuple, keepdim=True),
            float(eps),
            dtype=_master_float_dtype(x),
        )
    correction = min(int(ddof), max(sample - 1, 0))
    x_work = x.to(dtype=_master_float_dtype(x))
    try:
        var = torch.var(x_work, dim=dim_tuple, correction=correction, keepdim=True)
    except TypeError:
        var = torch.var(x_work, dim=dim_tuple, unbiased=False, keepdim=True) * (
            float(sample) / float(sample - correction) if correction > 0 else 1.0
        )
    return torch.sqrt(var.clamp(min=float(eps) ** 2)).clamp(min=float(eps))


def _to_tuple(x: Any) -> Tuple[int, ...]:
    return tuple(int(v) for v in x)


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _normal_pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


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
    return _normal_cdf(z)


def _fft_nd(
    x: torch.Tensor,
    shape: Sequence[int],
    *args: Any,
    real_input: bool = False,
    inverse: bool = False,
    norm: Optional[str] = "ortho",
    **kwargs: Any,
) -> torch.Tensor:
    _ = args, kwargs
    d, s = tuple(range(-len(shape), 0)), _to_tuple(shape)
    if not inverse:
        return (
            torch.fft.rfftn(x, s=s, dim=d, norm=norm)
            if real_input
            else torch.fft.fftn(x, s=s, dim=d, norm=norm)
        )
    return (
        torch.fft.irfftn(x, s=s, dim=d, norm=norm)
        if real_input
        else torch.fft.ifftn(x, s=s, dim=d, norm=norm)
    )


def _nufft_nd(
    x_cplx: torch.Tensor,
    omega: torch.Tensor,
    shape: Sequence[int],
    *args: Any,
    nufft_type: int = 2,
    eps: float = 1e-06,
    **kwargs: Any,
) -> torch.Tensor:
    _ = args, kwargs
    try:
        import cufinufft
    except ImportError as e:
        raise RuntimeError("cuFINUFFT not available") from e
    if not torch.is_complex(x_cplx):
        raise TypeError("x_cplx must be complex")
    if x_cplx.device.type != "cuda":
        raise RuntimeError("cuFINUFFT requires CUDA")

    B, ndim = int(x_cplx.shape[0]), len(shape)
    dtype_str = "complex128" if x_cplx.dtype == torch.complex128 else "complex64"

    def _exec_plan(n_trans: int, x: torch.Tensor, pts: Sequence[torch.Tensor]) -> torch.Tensor:
        plan = cufinufft.Plan(
            nufft_type,
            _to_tuple(shape),
            n_trans=n_trans,
            eps=eps,
            dtype=dtype_str,
        )
        plan.setpts(*pts)
        return torch.as_tensor(plan.execute(x.contiguous()), device=x.device)

    if omega.dim() == 2:
        pts = [omega[i].contiguous() for i in range(ndim)]
        try:
            out = _exec_plan(B, x_cplx, pts)
            if out.shape[0] != B and out.shape[-1] == B:
                return out.movedim(-1, 0)
            return out
        except Exception:
            pass

    return torch.stack(
        [
            _exec_plan(
                1,
                x_cplx[b],
                [omega[b if omega.dim() == 3 else 0, i].contiguous() for i in range(ndim)],
            )
            for b in range(B)
        ]
    )


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
        if q.ndim != 1 or not ((q > 0) & (q < 1)).all():
            raise ValueError("Invalid quantiles")
        self.register_buffer("q", q)
        w = torch.tensor(list(weights), dtype=torch.float32) if weights else torch.ones_like(q)
        if w.shape != q.shape:
            raise ValueError("Weights shape mismatch")
        self.register_buffer("w", w / (w.sum() + 1e-12))
        self.quantile_dim = None if quantile_dim is None else int(quantile_dim)
        self.reduction = str(reduction).lower()
        if self.reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"invalid reduction={reduction}")

    def _resolve_q_dim(self, shape: torch.Size) -> int:
        Q = int(self.q.numel())
        if (qd := self.quantile_dim) is not None:
            if not (0 <= qd < len(shape) and shape[qd] == Q):
                raise ValueError(f"Invalid quantile_dim {qd}")
            return qd
        candidates = [i for i, s in enumerate(shape) if int(s) == Q]
        return (
            1
            if 1 in candidates
            else 0
            if 0 in candidates
            else candidates[0]
            if len(candidates) == 1
            else -1
        )

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        if preds.shape != target.shape:
            raise ValueError("shape mismatch")
        qdim = self._resolve_q_dim(preds.shape)
        if qdim < 0:
            raise ValueError("cannot infer quantile_dim")
        if qdim != 0:
            preds, target = preds.transpose(0, qdim), target.transpose(0, qdim)
        errors = target - preds
        q = self.q.view(-1, *[1] * (errors.ndim - 1))
        losses = torch.maximum((q - 1) * errors, q * errors)
        if self.reduction == "none":
            return losses.transpose(0, qdim) if qdim != 0 else losses
        per_q = (
            losses.mean(dim=tuple(range(1, losses.ndim)))
            if self.reduction == "mean"
            else losses.sum(dim=tuple(range(1, losses.ndim)))
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
        skew: bool = False,
        skew_samples: int = 32,
        skew_pair_samples: int = 8,
        energy_cdist_max_bytes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.reduction = str(reduction).lower()
        self.mode = str(mode).lower()
        if self.reduction not in ("mean", "sum", "none") or self.mode not in (
            "normal",
            "energy",
        ):
            raise ValueError("Invalid reduction/mode")
        self.dim = dim
        self.eps = float(eps)
        self.ddof = int(ddof)
        self.std_param = std
        self.detach_stats = bool(detach_stats)
        self.sample_dim = None if sample_dim is None else int(sample_dim)
        self.max_z = float(max_z)
        self.skew = bool(skew)
        self._skew_samples = max(0, int(skew_samples))
        self._skew_pair_samples = max(0, int(skew_pair_samples))
        self.energy_cdist_max_bytes = (
            int(energy_cdist_max_bytes) if energy_cdist_max_bytes else None
        )
        self._energy_cdist_cache_key, self._energy_cdist_cache_val = None, None

    @staticmethod
    def _expand_params(x: TensorLike, ref: torch.Tensor) -> torch.Tensor:
        t = (
            x.to(device=ref.device, dtype=ref.dtype)
            if torch.is_tensor(x)
            else torch.tensor(x, device=ref.device, dtype=ref.dtype)
        )
        return t.view(*[1] * (ref.ndim - t.ndim), *t.shape) if t.ndim < ref.ndim else t

    def _dims(self, pred: torch.Tensor) -> Tuple[int, ...]:
        return _canonize_dims(pred, self.dim, keep_batch=False)

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean() if self.reduction == "mean" else (x.sum() if self.reduction == "sum" else x)

    def _std_from_error(self, err: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
        std = _coerce_std(err, dim=dims, ddof=self.ddof, eps=self.eps)
        return torch.clamp(std.detach() if self.detach_stats else std, min=self.eps)

    def _crps_normal_analytic(self, err: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        z = err / torch.clamp(sigma, min=self.eps)
        if self.max_z > 0.0:
            z = z.clamp(min=-self.max_z, max=self.max_z)
        return (
            sigma
            * (z * (2.0 * _normal_cdf(z) - 1.0) + 2.0 * _normal_pdf(z) - (1.0 / math.sqrt(math.pi)))
        ).clamp(min=0.0)

    def _crps_normal_skew_sampled(
        self, err: torch.Tensor, sigma: torch.Tensor, dims: Tuple[int, ...]
    ) -> torch.Tensor:
        z_obs = (
            (err / sigma.clamp(min=self.eps)).clamp(min=-self.max_z, max=self.max_z)
            if self.max_z > 0
            else err / sigma.clamp(min=self.eps)
        )
        z_centered = z_obs - z_obs.mean(dim=dims, keepdim=True)
        m2 = (z_centered**2).mean(dim=dims, keepdim=True).clamp(min=self.eps)
        alpha = ((z_centered**3).mean(dim=dims, keepdim=True) / (m2.sqrt() ** 3 + self.eps)).clamp(
            -5.0, 5.0
        )
        if self.detach_stats:
            alpha = alpha.detach()
        n_samples = int(self._skew_samples)
        if n_samples <= 0:
            return self._crps_normal_analytic(err, sigma)
        delta = (alpha / torch.sqrt(1.0 + alpha**2)).expand_as(z_obs)
        tail = (1.0 - delta**2).clamp(min=0.0).sqrt()
        sum_abs = 0.0
        chunk = 8
        for start in range(0, n_samples, chunk):
            k = min(chunk, n_samples - start)
            z0, z1 = torch.randn(2, k, *z_obs.shape, device=z_obs.device, dtype=z_obs.dtype)
            sum_abs += (
                (delta.unsqueeze(0) * z0.abs() + tail.unsqueeze(0) * z1 - z_obs.unsqueeze(0))
                .abs()
                .sum(dim=0)
            )
        e1 = sum_abs / float(n_samples)
        if (pair_k := min(int(self._skew_pair_samples), n_samples)) >= 2:
            z0, z1 = torch.randn(2, pair_k, *z_obs.shape, device=z_obs.device, dtype=z_obs.dtype)
            z_pair = delta.unsqueeze(0) * z0.abs() + tail.unsqueeze(0) * z1
            m_pairs = min(int(pair_k * (pair_k - 1)), 64)
            idx = torch.randint(0, pair_k, (m_pairs, 2), device=z_obs.device)
            e2 = 0.5 * (z_pair[idx[:, 0]] - z_pair[idx[:, 1]]).abs().mean(dim=0)
        else:
            e2 = torch.zeros_like(e1)
        return (sigma * (e1 - e2)).clamp(min=0.0)

    def _crps_normal(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError("shape mismatch")
        dims = self._dims(pred)
        err = target - pred
        if self.std_param is not None:
            sigma = self._expand_params(self.std_param, pred).clamp(min=self.eps)
            if self.detach_stats:
                sigma = sigma.detach()
        else:
            sigma = self._std_from_error(err, dims=dims)
        return self._reduce(
            self._crps_normal_skew_sampled(err, sigma, dims)
            if self.skew
            else self._crps_normal_analytic(err, sigma)
        )

    @staticmethod
    def _move_sample_dim_to_1(pred: torch.Tensor, sample_dim: int) -> torch.Tensor:
        nd = int(pred.ndim)
        sd = sample_dim if sample_dim >= 0 else sample_dim + nd
        if not (0 <= sd < nd):
            raise ValueError(f"Invalid sample_dim {sample_dim}")
        return pred if sd == 1 else pred.permute(0, sd, *[i for i in range(1, nd) if i != sd])

    def _recommend_energy_cdist_max_bytes(
        self, B: int, S: int, device: torch.device, out_elem_size: int
    ) -> int:
        candidates = (32 << 20, 64 << 20, 128 << 20)
        if device.type in {"cuda", "xpu", "mps"}:
            free: Optional[int] = None
            total: Optional[int] = None
            if not is_compiling():
                with contextlib.suppress(Exception):
                    from .system import Memory

                    free, total = map(lambda x: int(x) if x else None, Memory.mem_get_info(device))
            if not total:
                with contextlib.suppress(Exception):
                    from .system import available_accelerator_memory

                    total = available_accelerator_memory(device)
            cap = (
                int(min((free or total or 0) * 0.06, (total or 0) * 0.03))
                if total
                else candidates[0]
            )
        else:
            cap = candidates[-1]
        allowed = sorted([c for c in candidates if c <= cap] or [candidates[0]])
        row_bytes = max(1, int(B) * int(S) * int(out_elem_size))
        tgt = 1 if S <= 256 else (2 if S <= 512 else (4 if S <= 1024 else 8))
        req = min(S, max(1, (S + tgt - 1) // tgt))
        for budget in allowed:
            if max(1, min(S, int(budget // row_bytes))) >= req:
                return int(budget)
        return int(allowed[-1])

    def _energy_cdist_budget_bytes(self, B: int, S: int, samples: torch.Tensor) -> int:
        if self.energy_cdist_max_bytes:
            return int(self.energy_cdist_max_bytes)
        out_elem_size = max(4, int(samples.element_size()))
        dev = samples.device
        dev_idx = -1
        if dev.type == "cuda":
            from .system import get_accelerator_index

            dev_idx = (
                int(dev.index) if dev.index is not None else int(get_accelerator_index("cuda"))
            )
        key = (dev.type, dev_idx, int(B), int(S), int(out_elem_size))
        if self._energy_cdist_cache_key == key and self._energy_cdist_cache_val:
            return int(self._energy_cdist_cache_val)
        budget = self._recommend_energy_cdist_max_bytes(int(B), int(S), dev, out_elem_size)
        self._energy_cdist_cache_key, self._energy_cdist_cache_val = key, int(budget)
        return int(budget)

    def _crps_energy(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.sample_dim is None:
            raise ValueError("energy mode needs sample_dim")
        if target.ndim != pred.ndim - 1:
            raise ValueError("target ndim mismatch")
        pred_m = self._move_sample_dim_to_1(pred, self.sample_dim)
        nd = int(pred.ndim)
        sd = self.sample_dim if self.sample_dim >= 0 else self.sample_dim + nd
        target_m = target.unsqueeze(sd)
        if sd != 1:
            perm = [0, sd] + [i for i in range(1, nd) if i != sd]
            target_m = target_m.permute(*perm)
        B, S = int(pred_m.shape[0]), int(pred_m.shape[1])
        if S <= 1:
            raise ValueError("energy mode needs >1 samples")
        samples = pred_m.reshape(B, S, -1)
        term1 = torch.linalg.vector_norm(samples - target_m.reshape(B, 1, -1), dim=-1).mean(dim=1)
        budget_bytes = self._energy_cdist_budget_bytes(B, S, samples)
        max_elems = max(1, int(budget_bytes // max(4, int(samples.element_size()))))
        chunk = max(1, min(S, int(max_elems // max(B * S, 1))))
        acc_dt = torch.float32 if term1.dtype in (torch.float16, torch.bfloat16) else term1.dtype
        sum_d = sum(
            torch.cdist(samples[:, i : i + chunk], samples).sum(dim=(1, 2)).to(acc_dt)
            for i in range(0, S, chunk)
        )
        term2 = sum_d / float(S * (S - 1))
        return self._reduce((term1.to(acc_dt) - 0.5 * term2).clamp(min=0.0))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self._crps_normal(pred, target)
            if self.mode == "normal"
            else self._crps_energy(pred, target)
        )


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
        _ = args, kwargs
        if not (0.0 < confidence < 1.0):
            raise ValueError("Invalid confidence")
        if penalty.lower() not in ("hinge", "tau", "soft", "softplus"):
            raise ValueError("Invalid penalty")
        if reduction.lower() not in ("mean", "sum", "none"):
            raise ValueError("Invalid reduction")
        self.confidence = float(confidence)
        self.metric, self.penalty, self.reduction = (
            metric.lower(),
            penalty.lower(),
            reduction.lower(),
        )
        self.tau, self.hinge_power, self.eps = (
            float(tau),
            float(hinge_power),
            float(eps),
        )
        self.dim, self.mu, self.std, self.ddof, self.clamp_max = (
            dim,
            mu,
            std,
            int(ddof),
            clamp_max,
        )
        self.mu_mode, self.std_mode = mu_mode.lower(), std_mode.lower()
        self.two_tailed, self.detach_stats, self.skew = (
            bool(two_tailed),
            bool(detach_stats),
            bool(skew),
        )

    @staticmethod
    def _to_tensor_like(x: TensorLike, ref: torch.Tensor) -> torch.Tensor:
        return (
            x.to(device=ref.device, dtype=ref.dtype)
            if torch.is_tensor(x)
            else torch.tensor(x, device=ref.device, dtype=ref.dtype)
        )

    @staticmethod
    def _expand_params(x: TensorLike, ref: torch.Tensor) -> torch.Tensor:
        t = DistributionLoss._to_tensor_like(x, ref)
        return t.view(*[1] * (ref.ndim - t.ndim), *t.shape) if t.ndim < ref.ndim else t

    @staticmethod
    def _safe_std(x: torch.Tensor, dim: Tuple[int, ...], ddof: int, eps: float) -> torch.Tensor:
        return _coerce_std(x, dim, ddof, eps)

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean() if self.reduction == "mean" else (x.sum() if self.reduction == "sum" else x)

    def _dims(self, pred: torch.Tensor) -> Tuple[int, ...]:
        return _canonize_dims(pred, self.dim, keep_batch=False)

    def compute_mu(
        self, pred: torch.Tensor, target: torch.Tensor, dims: Tuple[int, ...]
    ) -> torch.Tensor:
        dims = _canonize_dims(pred, dims)
        red = torch.median if self.skew else torch.mean

        if self.mu_mode == "target":
            x = target
        elif self.mu_mode == "pred":
            x = pred
        elif self.mu_mode == "error":
            x = pred - target
        elif self.mu_mode == "pooled":
            mu = (
                0.5 * (_median_over_dims(target, dims) + _median_over_dims(pred, dims))
                if self.skew
                else 0.5 * (target.mean(dim=dims, keepdim=True) + pred.mean(dim=dims, keepdim=True))
            )
            return mu.detach() if self.detach_stats else mu
        elif self.mu_mode == "provided":
            if self.mu is None:
                raise ValueError("mu required")
            mu = self._expand_params(self.mu, pred)
            return mu.detach() if self.detach_stats else mu
        elif self.mu_mode == "none":
            mu = torch.zeros(1, device=pred.device, dtype=pred.dtype)
            return mu.detach() if self.detach_stats else mu
        else:
            raise ValueError(f"Invalid mu_mode {self.mu_mode}")

        mu = _median_over_dims(x, dims) if self.skew else red(x, dim=dims, keepdim=True)
        return mu.detach() if self.detach_stats else mu

    def compute_std(
        self, pred: torch.Tensor, target: torch.Tensor, dims: Tuple[int, ...]
    ) -> torch.Tensor:
        dims = _canonize_dims(pred, dims)

        def _get_s(t: torch.Tensor) -> torch.Tensor:
            return (
                _mad_std(t, dims, self.eps)
                if self.skew
                else self._safe_std(t, dims, self.ddof, self.eps)
            )

        if self.std_mode == "target":
            std = _get_s(target)
        elif self.std_mode == "pred":
            std = _get_s(pred)
        elif self.std_mode == "pooled":
            st, sp = _get_s(target), _get_s(pred)
            std = torch.sqrt((0.5 * (st**2 + sp**2)).clamp(min=self.eps**2))
        elif self.std_mode == "provided":
            if self.std is None:
                raise ValueError("std required")
            std = self._expand_params(self.std, pred)
        elif self.std_mode == "none":
            std = torch.ones(1, device=pred.device, dtype=pred.dtype)
        else:
            raise ValueError(f"Invalid std_mode {self.std_mode}")
        if self.detach_stats:
            std = std.detach()
        return torch.clamp(std, min=self.eps)

    def _statistic_abs(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        std: torch.Tensor,
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
        if self.penalty == "hinge":
            return torch.clamp(margin, min=0.0).pow(self.hinge_power)
        if self.penalty == "tau":
            tau = max(self.tau, self.eps)
            return F.softplus(margin / tau) * tau
        if self.penalty in ("soft", "softplus"):
            beta = max(self.tau, self.eps)
            return F.softplus(beta * margin) / beta
        raise ValueError("Invalid penalty")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError("shape mismatch")
        dims = self._dims(pred)
        mu = self.compute_mu(pred, target, dims)
        std = self.compute_std(pred, target, dims)
        stat_abs = self._statistic_abs(pred, target, mu, std)
        if self.clamp_max is not None:
            stat_abs = torch.clamp(stat_abs, max=float(self.clamp_max))
        return self.reduce(
            self._apply_penalty(self._compute_margin(stat_abs, pred, target, mu, std))
        )


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
        if self.metric not in {
            "z",
            "z_score",
            "z_value",
            "zscore",
            "zvalue",
            "p",
            "p_value",
            "pvalue",
        }:
            raise ValueError(f"Invalid metric {self.metric}")
        self._std_normal = Normal(loc=0.0, scale=1.0)
        q = 0.5 + 0.5 * self.confidence if self.two_tailed else self.confidence
        try:
            self._z_threshold_f64 = float(
                self._std_normal.icdf(torch.tensor(q, dtype=torch.float64)).item()
            )
        except Exception:
            self._z_threshold_f64 = float(self._std_normal.icdf(torch.tensor(q)).item())

    def _z_threshold(self, device: Any, dtype: Any) -> torch.Tensor:
        return torch.tensor(self._z_threshold_f64, device=device, dtype=dtype)

    def _compute_margin(
        self,
        stat_abs: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        z_abs = stat_abs
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
        if self.metric.startswith("z"):
            return z_abs - self._z_threshold(pred.device, pred.dtype)
        if self.metric.startswith("p"):
            try:
                one_tail = torch.clamp(1.0 - Normal(0.0, 1.0).cdf(z_abs), min=self.eps)
            except NotImplementedError:
                one_tail = torch.clamp(1.0 - _normal_cdf(z_abs), min=self.eps)
            p = 2.0 * one_tail if self.two_tailed else one_tail
            alpha = max(1.0 - self.confidence, self.eps)
            return -torch.log(torch.clamp(p, min=self.eps)) + math.log(alpha)
        raise ValueError("Invalid metric")


class StudentsTLoss(DistributionLoss):
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
        df: TensorLike = 3.0,
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
        if self.metric not in {
            "t",
            "t_score",
            "t_value",
            "tscore",
            "tvalue",
            "p",
            "p_value",
            "pvalue",
        }:
            raise ValueError(f"Invalid metric {self.metric}")
        self._cached_t_threshold_f64: Optional[float] = None
        self._cached_t_df: Optional[float] = None
        self._cached_t_q: Optional[float] = None

    def _t_threshold(self, device: Any, dtype: Any) -> torch.Tensor:
        q = 0.5 + 0.5 * self.confidence if self.two_tailed else self.confidence
        df_scalar = (
            float(self.df.detach().cpu().item())
            if torch.is_tensor(self.df) and self.df.numel() == 1
            else float(self.df)
            if not torch.is_tensor(self.df)
            else None
        )
        df_requires_grad = torch.is_tensor(self.df) and self.df.requires_grad
        if (
            df_scalar is not None
            and not df_requires_grad
            and self._cached_t_threshold_f64
            and self._cached_t_df == df_scalar
            and self._cached_t_q == float(q)
        ):
            return torch.tensor(self._cached_t_threshold_f64, device=device, dtype=dtype)
        df = self._to_tensor_like(self.df, torch.empty((), device=device, dtype=dtype))
        try:
            dist = StudentT(df=df)
            thr = dist.icdf(torch.tensor(q, device=device, dtype=dtype))
        except NotImplementedError:
            target = torch.full_like(df, float(q), dtype=dtype, device=device)
            loc = torch.zeros_like(df, dtype=dtype, device=device)
            scale = torch.ones_like(df, dtype=dtype, device=device)
            lo = torch.full_like(df, -50.0, dtype=dtype, device=device)
            hi = torch.full_like(df, 50.0, dtype=dtype, device=device)
            for _ in range(64):
                mid = 0.5 * (lo + hi)
                mask = _students_t_cdf(mid, df=df, loc=loc, scale=scale) >= target
                hi, lo = torch.where(mask, mid, hi), torch.where(~mask, mid, lo)
            thr = 0.5 * (lo + hi)
        if df_scalar is not None and not df_requires_grad and thr.numel() == 1:
            try:
                self._cached_t_threshold_f64 = float(thr.detach().double().cpu().item())
                self._cached_t_df = float(df_scalar)
                self._cached_t_q = float(q)
                return torch.tensor(self._cached_t_threshold_f64, device=device, dtype=dtype)
            except Exception:
                pass
        return thr

    def _compute_margin(
        self,
        stat_abs: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        t_abs = stat_abs
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
        if self.metric.startswith("t"):
            return t_abs - self._t_threshold(pred.device, pred.dtype)
        if self.metric.startswith("p"):
            df = self._expand_params(self.df, t_abs)
            df_safe = torch.clamp(df, min=2.0 + self.eps)
            scale = std * torch.sqrt(torch.clamp((df_safe - 2.0) / df_safe, min=self.eps))
            x = mu + t_abs * scale
            try:
                one_tail = torch.clamp(
                    1.0 - StudentT(df=df, loc=mu, scale=scale).cdf(x), min=self.eps
                )
            except NotImplementedError:
                one_tail = torch.clamp(1.0 - _students_t_cdf(x, df, mu, scale), min=self.eps)
            p = 2.0 * one_tail if self.two_tailed else one_tail
            alpha = max(1.0 - self.confidence, self.eps)
            return -torch.log(torch.clamp(p, min=self.eps)) + math.log(alpha)
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
        _ = args, kwargs
        self.out_shape = _to_tuple(out_shape)
        self.ndim = len(self.out_shape)
        self.mode = str(mode).lower()
        self.backend = None if backend is None else str(backend).lower()
        self.register_buffer("ktraj", ktraj if ktraj is not None else None, persistent=False)
        self.weight = float(weight)
        self.fft_norm = fft_norm
        self.reduction = str(reduction).lower()
        self.nufft_eps = float(nufft_eps)
        if self.mode not in ("fft", "nufft"):
            raise ValueError("mode must be 'fft' or 'nufft'")
        if self.mode == "nufft" and self.ktraj is None:
            raise ValueError("ktraj is required for NUFFT mode")

    def _mse(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = (a - b).abs() if (torch.is_complex(a) or torch.is_complex(b)) else (a - b)
        sq = diff * diff
        if self.reduction == "mean":
            val = sq.mean()
        elif self.reduction == "sum":
            val = sq.sum()
        elif self.reduction == "none":
            val = sq.reshape(int(a.shape[0]), -1).mean(dim=1)
        else:
            raise ValueError(f"invalid reduction={self.reduction}")
        return val * self.weight

    def forward(self, pred_flat: torch.Tensor, target_flat: torch.Tensor) -> torch.Tensor:
        B = int(pred_flat.shape[0])
        shape = self.out_shape
        x = pred_flat.view(B, *shape)
        y = target_flat.view(B, *shape)
        cdtype = (
            torch.complex128
            if x.dtype == torch.float64 or y.dtype == torch.float64
            else torch.complex64
        )

        if self.mode == "fft":
            Xk = _fft_nd(
                x.to(cdtype) if not torch.is_complex(x) else x,
                shape,
                norm=self.fft_norm,
            )
            Yk = _fft_nd(
                y.to(cdtype) if not torch.is_complex(y) else y,
                shape,
                norm=self.fft_norm,
            )
        elif self.mode == "nufft":
            if self.backend in (None, "cufinufft"):
                try:
                    Xk = _nufft_nd(
                        x.to(cdtype).unsqueeze(1),
                        self.ktraj,
                        shape,
                        nufft_type=2,
                        eps=self.nufft_eps,
                    )
                    Yk = _nufft_nd(
                        y.to(cdtype).unsqueeze(1),
                        self.ktraj,
                        shape,
                        nufft_type=2,
                        eps=self.nufft_eps,
                    )
                except RuntimeError as e:
                    if "cuFINUFFT" not in str(e) and "CUDA" not in str(e):
                        raise
                    Xk = _fft_nd(x.to(cdtype), shape, norm=self.fft_norm)
                    Yk = _fft_nd(y.to(cdtype), shape, norm=self.fft_norm)
            elif self.backend == "finufft":
                raise NotImplementedError("FINUFFT not impl")
            else:
                raise ValueError(f"Unknown backend {self.backend}")
        else:
            raise ValueError(f"Invalid mode {self.mode}")
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
        schedule_momentum: float = 0.95,
        min_coeff: float = 1e-6,
        max_coeff: float = 1.0 - 1e-6,
        eps: float = 1e-06,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _ = args, kwargs
        if not isinstance(coefficient, Sequence):
            raise TypeError(f"coefficient must be a Sequence, got {type(coefficient)}")
        if not isinstance(loss, Sequence):
            raise TypeError(f"loss must be a Sequence[nn.Module], got {type(loss)}")
        if len(coefficient) == 0 or len(coefficient) != len(loss):
            raise ValueError(
                f"invalid coefficient/loss length: len(coefficient)={len(coefficient)}, len(loss)={len(loss)}"
            )
        coeff_tensor = torch.as_tensor(coefficient, dtype=torch.float32)
        if coeff_tensor.ndim != 1 or int(coeff_tensor.numel()) != len(loss):
            raise ValueError(
                f"coefficient must be 1D of length {len(loss)}, got shape {tuple(coeff_tensor.shape)}"
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
        self.register_buffer("loss_avg", torch.full_like(coeff_tensor, fill_value=1.0))

    def _update_coefficients(self, per_loss_vals: List[torch.Tensor]) -> None:
        if (not self.auto_schedule) or (not per_loss_vals) or (not self.training):
            return
        with torch.no_grad():
            device = self.loss_avg.device
            dtype = self.loss_avg.dtype
            vals_t = (
                torch.stack([v.detach().abs().mean() for v in per_loss_vals])
                .to(device=device, dtype=dtype)
                .clamp(min=self.eps)
            )
            m = float(max(0.0, min(1.0, self.schedule_momentum)))
            self.loss_avg.mul_(m).add_(vals_t * (1.0 - m))
            base = self.coefficient.detach().to(device=device, dtype=dtype)
            base = base / base.sum().clamp(min=self.eps)
            inv = base / self.loss_avg.clamp(min=self.eps)
            new_w = inv / torch.clamp(inv.sum(), min=self.eps)
            if self.min_coeff > 0.0 or self.max_coeff < 1.0:
                new_w = new_w.clamp(min=float(self.min_coeff), max=float(self.max_coeff))
                new_w = new_w / torch.clamp(new_w.sum(), min=self.eps)
            self.coefficient.copy_(new_w)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        weights = self.coefficient.detach().clone().to(device=pred.device, dtype=pred.dtype)
        total = pred.new_tensor(self.offset, dtype=pred.dtype)
        per_loss_vals: List[torch.Tensor] = []
        for w, L in zip(weights, self.losses):
            v = L(pred, target)
            if not torch.is_tensor(v):
                raise TypeError("Loss must return tensor")
            v_eff = v.mean() if (self.reduce_each and v.dim() > 0) else v
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
        base_reduction: str = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _ = args, kwargs
        self.base = base
        self.mask_mode = str(mask_mode).lower()
        self.mask_value = mask_value
        self.tile_dim = tile_dim
        self.tile_size = int(tile_size) if tile_size is not None else None
        reduction_v = str(reduction).lower()
        if reduction_v not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be one of ('mean', 'sum', 'none'), got {reduction!r}")
        self.reduction = reduction_v
        base_red = str(base_reduction).lower()
        if base_red not in ("auto", "none", "mean", "sum"):
            raise ValueError(
                f"base_reduction must be one of ('auto', 'none', 'mean', 'sum'), got {base_reduction!r}"
            )
        self.base_reduction = base_red

    def _mask(self, pred: torch.Tensor, target: torch.Tensor) -> Optional[torch.Tensor]:
        try:
            if self.mask_mode == "finite":
                return expand_to_pred(torch.isfinite(target), pred)
            if self.mask_mode == "neq" and self.mask_value is not None:
                return expand_to_pred(target != target.new_tensor(self.mask_value), pred)
        except Exception:
            pass
        return None

    @staticmethod
    def _masked_select_safe(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        return x.masked_select(mask) if mask is not None and x.shape == mask.shape else x

    def _infer_base_reduction(self, loss: torch.Tensor, tile_pred: torch.Tensor) -> str:
        if self.base_reduction != "auto":
            return self.base_reduction
        red = getattr(self.base, "reduction", None)
        if isinstance(red, str) and red.lower() in ("mean", "sum", "none"):
            return red.lower()
        return (
            "none"
            if loss.shape == tile_pred.shape
            else "mean"
            if loss.numel() == 1 or loss.ndim == 0
            else "none"
        )

    def reduce(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x2 = self._masked_select_safe(x, mask)
        return (
            x2.mean() if self.reduction == "mean" else x2.sum() if self.reduction == "sum" else x2
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = self._mask(pred, target)
        if self.tile_dim is None or self.tile_size is None:
            v = self.base(pred, target)
            base_red = self._infer_base_reduction(v, pred)
            if self.reduction == "none":
                return self._masked_select_safe(v, mask)
            if base_red == "none" or v.shape == pred.shape:
                return self.reduce(v, mask)
            if mask is not None and mask.shape == pred.shape:
                if (n := int(pred[mask].numel())) == 0:
                    return pred.new_zeros(())
                v = self.base(pred[mask], target[mask])
            else:
                n = int(pred.numel())
            if v.numel() == 1:
                total_sum = v.reshape(()) * (float(n) if base_red != "sum" else 1.0)
                return total_sum if self.reduction == "sum" else total_sum / float(max(n, 1))
            return v.sum() if self.reduction == "sum" else v.mean()
        nd = int(pred.ndim)
        td = int(self.tile_dim) + nd if int(self.tile_dim) < 0 else int(self.tile_dim)
        td = max(0, min(td, nd - 1))
        N = int(pred.shape[td])
        acc_dtype = torch.float32 if pred.dtype in (torch.float16, torch.bfloat16) else pred.dtype
        total_sum = pred.new_tensor(0.0, dtype=acc_dtype)
        total_count: int = 0
        parts: List[torch.Tensor] = []
        start = 0
        while start < N:
            end = min(N, start + int(self.tile_size))
            sl = [slice(None)] * nd
            sl[td] = slice(start, end)
            pv, tv = pred[tuple(sl)], target[tuple(sl)]
            mv = mask[tuple(sl)] if mask is not None else None
            elem = self.base(pv, tv)
            if self.reduction == "none":
                flat = elem.reshape(-1)
                if mv is not None and elem.shape == mv.shape:
                    flat = flat[mv.reshape(-1)]
                parts.append(flat)
                start = end
                continue
            base_red = self._infer_base_reduction(elem, pv)
            if base_red == "none" or elem.shape == pv.shape:
                if mv is not None and elem.shape == mv.shape:
                    selected = elem.masked_select(mv)
                    total_sum = total_sum + selected.sum()
                    total_count += int(selected.numel())
                else:
                    total_sum = total_sum + elem.sum()
                    total_count += int(elem.numel())
                start = end
                continue
            if mv is not None and mv.shape == pv.shape:
                if (n := int(pv[mv].numel())) == 0:
                    start = end
                    continue
                elem2 = self.base(pv[mv], tv[mv])
            else:
                n = int(pv.numel())
                elem2 = elem
            if elem2.numel() == 1:
                total_sum = total_sum + elem2.reshape(()) * (float(n) if base_red != "sum" else 1.0)
                total_count += int(n)
            else:
                total_sum = total_sum + elem2.sum()
                total_count += int(elem2.numel())
            start = end
        if self.reduction == "none":
            return torch.cat(parts, dim=0) if parts else pred.new_zeros(())
        return total_sum if self.reduction == "sum" else total_sum / float(max(total_count, 1))


@dataclass
class LossWeightController:
    momentum: float = 0.95
    min_weight: float = 1e-6
    max_weight: float = 1.0 - 1e-6
    eps: float = 1e-06
    top_avg: float = 0.75
    bottom_avg: float = 0.25

    def weights(self) -> Tuple[float, float]:
        top, bottom = max(self.eps, self.top_avg), max(self.eps, self.bottom_avg)
        if (total := top + bottom) <= 0.0:
            return (0.5, 0.5)
        ratio_top = max(self.min_weight, min(top / total, self.max_weight))
        ratio_bottom = max(self.min_weight, min(bottom / total, self.max_weight))
        norm = ratio_top + ratio_bottom
        return (0.5, 0.5) if norm <= 0.0 else (ratio_top / norm, ratio_bottom / norm)

    def update(self, top_loss: Optional[torch.Tensor], bottom_loss: Optional[torch.Tensor]) -> None:
        for loss, attr in [(top_loss, "top_avg"), (bottom_loss, "bottom_avg")]:
            if loss is not None:
                try:
                    val = float(loss.detach().abs().mean().item())
                except Exception:
                    val = float(self.eps)
                setattr(
                    self,
                    attr,
                    self.momentum * getattr(self, attr)
                    + (1.0 - self.momentum) * max(val, self.eps),
                )
