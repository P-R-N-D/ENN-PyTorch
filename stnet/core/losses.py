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

Number = Union[float, int]
TensorLike = Union[Number, torch.Tensor]


# ----------------------------
# Helpers
# ----------------------------
def _canonize_dims(
    x: torch.Tensor, dims: Any, keep_batch: bool = False
) -> Tuple[int, ...]:
    nd = int(x.ndim)
    if dims is None:
        return tuple(range(1, nd)) if nd > 1 else (0,)

    if isinstance(dims, int):
        dims = (dims,)
    pos: List[int] = []
    for d in dims:
        try:
            d_int = int(d)
        except Exception:
            continue
        d0 = d_int + nd if d_int < 0 else d_int
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


def _mad_std(x: torch.Tensor, dims: Tuple[int, ...], eps: float) -> torch.Tensor:
    mu = _median_over_dims(x, dims)
    dev = (x - mu).abs()
    mad = _median_over_dims(dev, dims)
    # Consistency constant for normal distribution.
    c = 1.482602218505602
    return torch.clamp(mad * c, min=float(eps))


def _master_float_dtype(x: torch.Tensor) -> torch.dtype:
    # Master dtypes are float64/float32; ints are handled by casting to float32.
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def _coerce_std(
    x: torch.Tensor, dim: Tuple[int, ...] | int | None, ddof: int, eps: float
) -> torch.Tensor:
    if dim is None:
        dim_tuple: Tuple[int, ...] = tuple()
    elif isinstance(dim, int):
        dim_tuple = (int(dim),)
    else:
        dim_tuple = tuple(int(d) for d in dim)

    # Compute sample size over valid dims for ddof correction.
    nd = int(x.dim())
    dims_pos = tuple((d if d >= 0 else nd + d) for d in dim_tuple)
    sample = 1
    for d in dims_pos:
        if 0 <= d < nd:
            sample *= max(1, int(x.shape[d]))

    # Degenerate: std undefined; return eps with correct broadcast shape.
    if sample <= 1:
        base = x.mean(dim=dim_tuple, keepdim=True)
        return torch.full_like(
            base, fill_value=float(eps), dtype=_master_float_dtype(base)
        )

    correction = min(int(ddof), max(sample - 1, 0))
    work_dtype = _master_float_dtype(x)

    x_work = x
    if (not torch.is_floating_point(x_work)) or (x_work.dtype != work_dtype):
        x_work = x_work.to(dtype=work_dtype)

    eps_f = float(eps)
    try:
        # Newer PyTorch: `correction` supports general ddof.
        var = torch.var(x_work, dim=dim_tuple, correction=correction, keepdim=True)
    except TypeError:
        # Older PyTorch: only unbiased (ddof=1) supported via `unbiased=True`.
        # Compute var0 with ddof=0 and apply correction factor: n/(n-ddof).
        var0 = torch.var(x_work, dim=dim_tuple, unbiased=False, keepdim=True)
        if correction > 0 and sample > correction:
            var = var0 * (float(sample) / float(sample - correction))
        else:
            var = var0

    std = torch.sqrt(torch.clamp(var, min=eps_f * eps_f))
    return torch.clamp(std, min=eps_f)


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
    # Approximate with normal CDF for df >= 3
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
    _ = args, kwargs
    try:
        import cufinufft
    except ImportError as e:
        raise RuntimeError("cuFINUFFT not available") from e

    if not torch.is_complex(x_cplx):
        raise TypeError("x_cplx must be a complex-valued tensor")

    B = int(x_cplx.shape[0])
    ndim = len(shape)

    device = x_cplx.device
    if device.type != "cuda":
        raise RuntimeError(f"cuFINUFFT requires CUDA tensors, got device={device}")

    dtype_str = "complex128" if x_cplx.dtype == torch.complex128 else "complex64"

    if omega.dim() == 2:
        # Shared points for all batch elements: setpts once, reuse plan.
        pts = [omega[i].contiguous() for i in range(ndim)]

        # Best-effort vectorized path (n_trans=B).
        if B > 1:
            try:
                plan = cufinufft.Plan(
                    nufft_type,
                    _to_tuple(shape),
                    n_trans=B,
                    eps=eps,
                    dtype=dtype_str,
                )
                plan.setpts(*pts)
                fk = plan.execute(x_cplx.contiguous())
                out = torch.as_tensor(fk, device=device)

                # Normalize to (B, ...).
                if out.ndim >= 1 and int(out.shape[0]) != B and int(out.shape[-1]) == B:
                    out = out.movedim(-1, 0)
                if out.ndim == 0 or int(out.shape[0]) != B:
                    raise RuntimeError("unexpected cuFINUFFT output layout for n_trans")

                return out
            except Exception:
                pass

        plan = cufinufft.Plan(
            nufft_type,
            _to_tuple(shape),
            n_trans=1,
            eps=eps,
            dtype=dtype_str,
        )
        plan.setpts(*pts)

        fk0 = torch.as_tensor(plan.execute(x_cplx[0].contiguous()), device=device)
        out = fk0.new_empty((B, *fk0.shape))
        out[0] = fk0
        for b in range(1, B):
            out[b] = torch.as_tensor(plan.execute(x_cplx[b].contiguous()), device=device)
        return out

    if omega.dim() == 3:
        try:
            plan = cufinufft.Plan(
                nufft_type,
                _to_tuple(shape),
                n_trans=1,
                eps=eps,
                dtype=dtype_str,
            )

            pts0 = [omega[0, i].contiguous() for i in range(ndim)]
            plan.setpts(*pts0)
            fk0 = torch.as_tensor(plan.execute(x_cplx[0].contiguous()), device=device)
            out = fk0.new_empty((B, *fk0.shape))
            out[0] = fk0

            for b in range(1, B):
                pts = [omega[b, i].contiguous() for i in range(ndim)]
                plan.setpts(*pts)
                out[b] = torch.as_tensor(plan.execute(x_cplx[b].contiguous()), device=device)

            return out
        except Exception:
            out_list: List[torch.Tensor] = []
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
            return torch.cat(out_list, dim=0)

    raise ValueError(
        f"omega must have shape (ndim, npts) or (B, ndim, npts), got {omega.shape}"
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


# ----------------------------
# Losses
# ----------------------------
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
        if q.ndim != 1 or not bool(torch.all((q > 0) & (q < 1))):
            raise ValueError(
                "quantiles must be a 1D sequence with all values in the open interval (0, 1)"
            )
        self.register_buffer("q", q)
        if weights is None:
            w = torch.ones_like(q)
        else:
            w = torch.tensor(list(weights), dtype=torch.float32)
            if w.shape != q.shape:
                raise ValueError(
                    f"weights must have the same shape as quantiles: expected {tuple(q.shape)}, got {tuple(w.shape)}"
                )
        self.register_buffer("w", w / (w.sum() + 1e-12))
        self.quantile_dim = None if quantile_dim is None else int(quantile_dim)
        self.reduction = str(reduction).lower()
        if self.reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"invalid reduction={reduction}")

    def _resolve_q_dim(self, shape: torch.Size) -> int:
        Q = int(self.q.numel())
        if self.quantile_dim is not None:
            if not (0 <= self.quantile_dim < len(shape)):
                raise ValueError(
                    f"quantile_dim={self.quantile_dim} is out of bounds for shape {tuple(shape)}"
                )
            if int(shape[self.quantile_dim]) != Q:
                raise ValueError(
                    f"quantile_dim={self.quantile_dim} expects size {Q}, got {int(shape[self.quantile_dim])}"
                )
            return self.quantile_dim
        candidates = [i for i, s in enumerate(shape) if int(s) == Q]
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
        per_q = losses.mean(dim=dims) if self.reduction == "mean" else losses.sum(dim=dims)
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
        # Skew correction is optional and sampling-based (best-effort).
        skew: bool = False,
        skew_samples: int = 32,
        skew_pair_samples: int = 8,
        energy_cdist_max_bytes: Optional[int] = None,
    ) -> None:
        super().__init__()
        reduction_l = str(reduction).lower()
        if reduction_l not in {"mean", "sum", "none"}:
            raise ValueError(f"invalid reduction={reduction}")
        mode_l = str(mode).lower()
        if mode_l not in {"normal", "energy"}:
            raise ValueError(f"CRPSLoss.mode must be 'normal' or 'energy', got {mode}")
        self.dim = dim
        self.eps = float(eps)
        self.reduction = reduction_l
        self.ddof = int(ddof)
        self.std_param = std
        self.detach_stats = bool(detach_stats)
        self.mode = mode_l
        self.sample_dim = None if sample_dim is None else int(sample_dim)
        self.max_z = float(max_z)

        self.skew = bool(skew)
        self._skew_samples = max(0, int(skew_samples))
        self._skew_pair_samples = max(0, int(skew_pair_samples))

        # Energy-mode pairwise term uses `torch.cdist` in chunks.
        # If `energy_cdist_max_bytes` is None, auto-pick among {32,64,128} MiB
        # based on (B,S) and device memory. Otherwise, use the provided budget.
        self.energy_cdist_max_bytes = (
            None if energy_cdist_max_bytes is None else int(energy_cdist_max_bytes)
        )
        self._energy_cdist_cache_key = None
        self._energy_cdist_cache_val = None

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

    def _crps_normal_analytic(
        self, err: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        z = err / torch.clamp(sigma, min=self.eps)
        if self.max_z > 0.0:
            z = torch.clamp(z, min=-self.max_z, max=self.max_z)

        # CRPS = sigma * [ z(2Φ(z)-1) + 2φ(z) - 1/sqrt(pi) ]
        cdf = _normal_cdf(z)
        pdf = _normal_pdf(z)
        term = z * (2.0 * cdf - 1.0) + 2.0 * pdf - (1.0 / math.sqrt(math.pi))
        crps = sigma * term
        return torch.clamp(crps, min=0.0)

    def _crps_normal_skew_sampled(
        self, err: torch.Tensor, sigma: torch.Tensor, dims: Tuple[int, ...]
    ) -> torch.Tensor:
        # Standardized residuals for skew estimation.
        z_obs = err / torch.clamp(sigma, min=self.eps)
        if self.max_z > 0.0:
            z_obs = torch.clamp(z_obs, min=-self.max_z, max=self.max_z)

        z_mean = z_obs.mean(dim=dims, keepdim=True)
        z_centered = z_obs - z_mean
        m2 = (z_centered**2).mean(dim=dims, keepdim=True).clamp(min=self.eps)
        m3 = (z_centered**3).mean(dim=dims, keepdim=True)
        skew_emp = m3 / (m2.sqrt() ** 3 + self.eps)

        alpha = torch.clamp(skew_emp, min=-5.0, max=5.0)
        if self.detach_stats:
            alpha = alpha.detach()

        n_samples = int(self._skew_samples)
        if n_samples <= 0:
            # Fallback to analytic when skew sampling disabled.
            return self._crps_normal_analytic(err, sigma)

        # Skew-normal sampling parameter.
        delta = alpha / torch.sqrt(1.0 + alpha * alpha)  # same shape as z_obs (keepdim)
        # Ensure broadcast to z_obs.
        delta = delta.expand_as(z_obs)
        tail = torch.sqrt(torch.clamp(1.0 - delta * delta, min=0.0))

        device, dtype = z_obs.device, z_obs.dtype
        z_obs0 = z_obs.unsqueeze(0)

        # e1 = E|Z - z_obs|
        sum_abs = torch.zeros_like(z_obs)
        # Chunked sampling to cap peak memory.
        chunk = 8
        for start in range(0, n_samples, chunk):
            k = min(chunk, n_samples - start)
            z0 = torch.randn((k, *z_obs.shape), device=device, dtype=dtype)
            z1 = torch.randn((k, *z_obs.shape), device=device, dtype=dtype)
            z_samp = delta.unsqueeze(0) * z0.abs() + tail.unsqueeze(0) * z1
            sum_abs = sum_abs + (z_samp - z_obs0).abs().sum(dim=0)
        e1 = sum_abs / float(n_samples)

        # e2 = 0.5 E|Z - Z'| (estimated from a small subset)
        pair_k = min(int(self._skew_pair_samples), n_samples)
        if pair_k >= 2:
            z0 = torch.randn((pair_k, *z_obs.shape), device=device, dtype=dtype)
            z1 = torch.randn((pair_k, *z_obs.shape), device=device, dtype=dtype)
            z_pair = delta.unsqueeze(0) * z0.abs() + tail.unsqueeze(0) * z1
            # Avoid O(pair_k^2) materialization: sample random pairs instead.
            m_pairs = min(int(pair_k * (pair_k - 1)), 64)
            idx_i = torch.randint(0, pair_k, (m_pairs,), device=device)
            idx_j = (idx_i + torch.randint(1, pair_k, (m_pairs,), device=device)) % pair_k
            diff = (z_pair[idx_i] - z_pair[idx_j]).abs()
            e2 = 0.5 * diff.mean(dim=0)
        else:
            e2 = torch.zeros_like(e1)

        crps = sigma * (e1 - e2)
        return torch.clamp(crps, min=0.0)

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

        if not self.skew:
            crps = self._crps_normal_analytic(err, sigma)
        else:
            crps = self._crps_normal_skew_sampled(err, sigma, dims=dims)

        return self._reduce(crps)

    @staticmethod
    def _move_sample_dim_to_1(pred: torch.Tensor, sample_dim: int) -> torch.Tensor:
        nd = int(pred.ndim)
        sd = sample_dim if sample_dim >= 0 else sample_dim + nd
        if not (0 <= sd < nd):
            raise ValueError(f"invalid sample_dim={sample_dim} for pred.ndim={nd}")
        if sd == 1:
            return pred
        perm = [0, sd] + [i for i in range(1, nd) if i != sd]
        return pred.permute(*perm)


    @staticmethod
    def _is_compiling() -> bool:
        try:
            import torch._dynamo  # type: ignore

            return bool(torch._dynamo.is_compiling())
        except Exception:
            return False

    def _recommend_energy_cdist_max_bytes(
        self, B: int, S: int, device: torch.device, out_elem_size: int
    ) -> int:
        _MIB = 1024 * 1024
        candidates = (32 * _MIB, 64 * _MIB, 128 * _MIB)

        # Determine a conservative cap from device memory.
        if device.type in {"cuda", "xpu", "mps"}:
            free: Optional[int] = None
            total: Optional[int] = None

            if not self._is_compiling():
                with contextlib.suppress(Exception):
                    from .system import Memory

                    free_i, total_i = Memory.device_mem_get_info(device)
                    free = int(free_i) if free_i is not None else None
                    total = int(total_i) if total_i is not None else None

            # Total-only fallback when mem_get_info is unavailable.
            if total is None or total <= 0:
                with contextlib.suppress(Exception):
                    from .system import accel_device_total_memory_bytes

                    total = accel_device_total_memory_bytes(device)

            if total is None or total <= 0:
                cap = candidates[0]
            else:
                if free is None:
                    # Use total as a proxy when free-memory query is unavailable.
                    free = total
                cap = int(min(int(free) * 0.06, int(total) * 0.03))
                if cap <= 0:
                    cap = candidates[0]
        else:
            # CPU/other backends: default to the largest candidate.
            cap = candidates[-1]

        allowed = [c for c in candidates if c <= cap] or [candidates[0]]
        allowed.sort()

        row_bytes = max(1, int(B) * int(S) * int(out_elem_size))  # bytes for chunk_rows=1 output

        # Target number of Python-loop iterations over row chunks.
        # (Larger chunks reduce loop overhead but increase peak memory.)
        if S <= 256:
            target_iters = 1
        elif S <= 512:
            target_iters = 2
        elif S <= 1024:
            target_iters = 4
        else:
            target_iters = 8

        required_rows = max(1, int((S + target_iters - 1) // target_iters))
        required_rows = min(required_rows, S)

        # Pick the smallest allowed candidate that reaches required_rows,
        # else fall back to the largest allowed.
        for budget in allowed:
            rows = max(1, min(S, int(budget // row_bytes)))
            if rows >= required_rows:
                return int(budget)
        return int(allowed[-1])

    def _energy_cdist_budget_bytes(self, B: int, S: int, samples: torch.Tensor) -> int:
        if self.energy_cdist_max_bytes is not None:
            return int(self.energy_cdist_max_bytes)

        # Output of cdist is typically float32 for fp16/bf16 inputs; be conservative.
        out_elem_size = max(4, int(samples.element_size()))
        dev = samples.device
        dev_idx = -1
        if dev.type == "cuda":
            from .system import accel_current_device_index
            dev_idx = int(dev.index) if dev.index is not None else int(accel_current_device_index("cuda"))

        key = (dev.type, dev_idx, int(B), int(S), int(out_elem_size))
        if self._energy_cdist_cache_key == key and self._energy_cdist_cache_val is not None:
            return int(self._energy_cdist_cache_val)

        budget = self._recommend_energy_cdist_max_bytes(int(B), int(S), dev, out_elem_size)
        self._energy_cdist_cache_key = key
        self._energy_cdist_cache_val = int(budget)
        return int(budget)
    def _crps_energy(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.sample_dim is None:
            raise ValueError("CRPSLoss(mode='energy') requires sample_dim to be set")
        if target.ndim != pred.ndim - 1:
            raise ValueError(
                "In energy mode, target must be [B, ...] with the same tail shape as pred without sample_dim"
            )

        pred_m = self._move_sample_dim_to_1(pred, self.sample_dim)

        # Align target to the same tail-dim order as pred_m.
        nd = int(pred.ndim)
        sd = self.sample_dim if self.sample_dim >= 0 else self.sample_dim + nd
        target_m = target.unsqueeze(sd)  # insert singleton sample dim
        if sd != 1:
            perm = [0, sd] + [i for i in range(1, nd) if i != sd]
            target_m = target_m.permute(*perm)

        B = int(pred_m.shape[0])
        S = int(pred_m.shape[1])
        if S <= 1:
            raise ValueError("energy mode requires at least 2 samples along sample_dim")

        samples = pred_m.reshape(B, S, -1)
        target_flat = target_m.reshape(B, 1, -1)

        # term1 = E||X - y||
        dist1 = torch.linalg.vector_norm(samples - target_flat, dim=-1)
        term1 = dist1.mean(dim=1)

        # term2 = E||X - X'||
        # Use cdist to avoid allocating [B,S,S,V].
        # Chunk rows based on an auto-tuned (or user-provided) max-bytes budget.
        budget_bytes = self._energy_cdist_budget_bytes(B, S, samples)
        out_elem_size = max(4, int(samples.element_size()))
        max_elems = max(1, int(budget_bytes // out_elem_size))
        chunk = max(1, min(S, int(max_elems // max(B * S, 1))))

        acc_dtype = (
            torch.float32
            if dist1.dtype in (torch.float16, torch.bfloat16)
            else dist1.dtype
        )
        term1 = term1.to(dtype=acc_dtype)
        sum_d = samples.new_zeros((B,), dtype=acc_dtype)
        for i in range(0, S, chunk):
            a = samples[:, i : i + chunk]
            d = torch.cdist(a, samples)  # [B,chunk,S]
            sum_d = sum_d + d.sum(dim=(1, 2)).to(acc_dtype)

        denom = float(S * (S - 1))
        term2 = sum_d / denom

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
        _ = args, kwargs
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
            return x.to(device=ref.device, dtype=ref.dtype)
        return torch.tensor(x, device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _expand_params(x: TensorLike, ref: torch.Tensor) -> torch.Tensor:
        t = DistributionLoss._to_tensor_like(x, ref)
        if t.ndim < ref.ndim:
            t = t.view(*[1] * (ref.ndim - t.ndim), *t.shape)
        return t

    @staticmethod
    def _safe_std(x: torch.Tensor, dim: Tuple[int, ...], ddof: int, eps: float) -> torch.Tensor:
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

    def compute_mu(self, pred: torch.Tensor, target: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
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
                else:
                    mu = 0.5 * (
                        target.mean(dim=dims, keepdim=True) + pred.mean(dim=dims, keepdim=True)
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
                return mu.detach() if self.detach_stats else mu
            case _:
                raise ValueError("invalid mu_mode")

        mu = _median_over_dims(x, dims) if self.skew else reducer(x, dim=dims, keepdim=True)
        return mu.detach() if self.detach_stats else mu

    def compute_std(self, pred: torch.Tensor, target: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
        dims = _canonize_dims(pred, dims)
        match self.std_mode:
            case "target":
                std = _mad_std(target, dims=dims, eps=self.eps) if self.skew else self._safe_std(target, dim=dims, ddof=self.ddof, eps=self.eps)
            case "pred":
                std = _mad_std(pred, dims=dims, eps=self.eps) if self.skew else self._safe_std(pred, dim=dims, ddof=self.ddof, eps=self.eps)
            case "pooled":
                std_t = _mad_std(target, dims=dims, eps=self.eps) if self.skew else self._safe_std(target, dim=dims, ddof=self.ddof, eps=self.eps)
                std_p = _mad_std(pred, dims=dims, eps=self.eps) if self.skew else self._safe_std(pred, dim=dims, ddof=self.ddof, eps=self.eps)
                std = torch.sqrt(torch.clamp(0.5 * (std_t * std_t + std_p * std_p), min=self.eps * self.eps))
            case "provided":
                if self.std is None:
                    raise ValueError("std required when std_mode='provided'")
                std = self._expand_params(self.std, pred)
            case "none":
                std = torch.ones(1, device=pred.device, dtype=pred.dtype)
            case _:
                raise ValueError("invalid std_mode")
        if self.detach_stats:
            std = std.detach()
        return torch.clamp(std, min=self.eps)

    def _statistic_abs(self, pred: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return ((pred - target - mu) / std).abs()

    def _compute_margin(self, stat_abs: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _apply_penalty(self, margin: torch.Tensor) -> torch.Tensor:
        match self.penalty:
            case "hinge":
                pen = torch.clamp(margin, min=0.0).pow(self.hinge_power)
            case "tau":
                tau = max(self.tau, self.eps)
                pen = F.softplus(margin / tau) * tau
            case "soft" | "softplus":
                beta = max(self.tau, self.eps)
                pen = F.softplus(beta * margin) / beta
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
        valid = {"z", "z_score", "z_value", "zscore", "zvalue", "p", "p_value", "pvalue"}
        if self.metric not in valid:
            raise ValueError(f"Invalid metric for StandardNormalLoss: {self.metric}")
        self._std_normal = Normal(loc=0.0, scale=1.0)
        # Cache scalar z-threshold once (device/dtype independent).
        q = 0.5 + 0.5 * self.confidence if self.two_tailed else self.confidence
        try:
            self._z_threshold_f64 = float(self._std_normal.icdf(torch.tensor(q, dtype=torch.float64)).item())
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
                    one_tail = torch.clamp(1.0 - Normal(loc=0.0, scale=1.0).cdf(x), min=self.eps)
                except NotImplementedError:
                    one_tail = torch.clamp(1.0 - _normal_cdf(x), min=self.eps)
                p = 2.0 * one_tail if self.two_tailed else one_tail
                alpha = max(1.0 - self.confidence, self.eps)
                return -torch.log(torch.clamp(p, min=self.eps)) + math.log(alpha)
            case _:
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
        valid = {"t", "t_score", "t_value", "tscore", "tvalue", "p", "p_value", "pvalue"}
        if self.metric not in valid:
            raise ValueError(f"Invalid metric for StudentsTLoss: {self.metric}")

        # Cache scalar t-threshold for fixed df (avoids repeated icdf/bisection in forward).
        self._cached_t_threshold_f64: Optional[float] = None
        self._cached_t_df: Optional[float] = None
        self._cached_t_q: Optional[float] = None

    def _t_threshold(self, device: Any, dtype: Any) -> torch.Tensor:
        q = 0.5 + 0.5 * self.confidence if self.two_tailed else self.confidence

        df_scalar: Optional[float] = None
        try:
            if torch.is_tensor(self.df):
                if int(self.df.numel()) == 1:
                    df_scalar = float(self.df.detach().cpu().item())
            else:
                df_scalar = float(self.df)
        except Exception:
            df_scalar = None

        df_requires_grad = torch.is_tensor(self.df) and self.df.requires_grad

        if (
            df_scalar is not None
            and not df_requires_grad
            and self._cached_t_threshold_f64 is not None
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
                c = _students_t_cdf(mid, df=df, loc=loc, scale=scale)
                hi = torch.where(c >= target, mid, hi)
                lo = torch.where(c < target, mid, lo)
            thr = 0.5 * (lo + hi)

        # Cache only when df is a fixed scalar.
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
                scale = std * torch.sqrt(torch.clamp((df_safe - 2.0) / df_safe, min=self.eps))
                x = mu + t_abs * scale
                try:
                    one_tail = torch.clamp(1.0 - StudentT(df=df, loc=mu, scale=scale).cdf(x), min=self.eps)
                except NotImplementedError:
                    one_tail = torch.clamp(1.0 - _students_t_cdf(x, df, mu, scale), min=self.eps)
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
        match self.reduction:
            case "mean":
                val = sq.mean()
            case "sum":
                val = sq.sum()
            case "none":
                B = int(a.shape[0])
                val = sq.reshape(B, -1).mean(dim=1)
            case _:
                raise ValueError(f"invalid reduction={self.reduction}")
        return val * self.weight

    def forward(self, pred_flat: torch.Tensor, target_flat: torch.Tensor) -> torch.Tensor:
        B = int(pred_flat.shape[0])
        shape = self.out_shape
        x = pred_flat.view(B, *shape)
        y = target_flat.view(B, *shape)

        # Pick complex dtype based on master float.
        base_dtype = torch.float64 if (x.dtype == torch.float64 or y.dtype == torch.float64) else torch.float32
        cplx_dtype = torch.complex128 if base_dtype == torch.float64 else torch.complex64

        match self.mode:
            case "fft":
                Xk = _fft_nd(x.to(cplx_dtype) if not torch.is_complex(x) else x, shape, real_input=False, inverse=False, norm=self.fft_norm)
                Yk = _fft_nd(y.to(cplx_dtype) if not torch.is_complex(y) else y, shape, real_input=False, inverse=False, norm=self.fft_norm)
            case "nufft":
                match self.backend:
                    case None | "cufinufft":
                        try:
                            Xk = _nufft_nd(
                                x.to(cplx_dtype).unsqueeze(1),
                                self.ktraj,
                                shape,
                                nufft_type=2,
                                eps=self.nufft_eps,
                            )
                            Yk = _nufft_nd(
                                y.to(cplx_dtype).unsqueeze(1),
                                self.ktraj,
                                shape,
                                nufft_type=2,
                                eps=self.nufft_eps,
                            )
                        except RuntimeError as e:
                            msg = str(e)
                            if ("cuFINUFFT not available" in msg) or ("requires CUDA tensors" in msg):
                                Xk = _fft_nd(x.to(cplx_dtype), shape, real_input=False, inverse=False, norm=self.fft_norm)
                                Yk = _fft_nd(y.to(cplx_dtype), shape, real_input=False, inverse=False, norm=self.fft_norm)
                            else:
                                raise
                    case "finufft":
                        raise NotImplementedError(
                            "FINUFFT path: wire with finufft.nufft*d* or Plan if you need CPU NUFFT."
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

        # Avoid compilation/tracing side-effects.
        with torch.no_grad():
            # Keep updates on-device (avoid .item() GPU sync).
            device = self.loss_avg.device
            dtype = self.loss_avg.dtype
            vals_t = torch.stack([v.detach().abs().mean() for v in per_loss_vals]).to(device=device, dtype=dtype)
            vals_t = torch.clamp(vals_t, min=self.eps)

            m = float(max(0.0, min(1.0, self.schedule_momentum)))
            self.loss_avg.mul_(m).add_(vals_t * (1.0 - m))

            base = self.coefficient.detach().to(device=device, dtype=dtype)
            base = base / torch.clamp(base.sum(), min=self.eps)

            inv = base / torch.clamp(self.loss_avg, min=self.eps)
            new_w = inv / torch.clamp(inv.sum(), min=self.eps)

            if self.min_coeff > 0.0 or self.max_coeff < 1.0:
                lo = float(self.min_coeff)
                hi = float(self.max_coeff)
                new_w = torch.clamp(new_w, min=lo, max=hi)
                new_w = new_w / torch.clamp(new_w.sum(), min=self.eps)

            self.coefficient.copy_(new_w.to(self.coefficient.device, dtype=self.coefficient.dtype))

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        weights = self.coefficient.detach().clone().to(device=pred.device, dtype=pred.dtype)
        total = pred.new_tensor(self.offset, dtype=pred.dtype)

        per_loss_vals: List[torch.Tensor] = []
        for w, L in zip(weights, self.losses):
            v = L(pred, target)
            if not torch.is_tensor(v):
                raise TypeError(
                    f"Loss module {L.__class__.__name__} must return a Tensor, got {type(v)}"
                )
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
                    base = target.new_tensor(self.mask_value, dtype=target.dtype, device=target.device)
                    return expand_to_pred(target != base, pred)
                except Exception:
                    return None
            case _:
                return None

    @staticmethod
    def _masked_select_safe(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return x
        # Only apply elementwise mask when shapes match.
        if x.shape == mask.shape:
            return x.masked_select(mask)
        # If loss is scalar or per-sample, masking is ill-defined; leave unchanged.
        if x.numel() == 1 or x.ndim == 0:
            return x
        return x

    def _infer_base_reduction(self, loss: torch.Tensor, tile_pred: torch.Tensor) -> str:
        if self.base_reduction != "auto":
            return self.base_reduction
        red = getattr(self.base, "reduction", None)
        if isinstance(red, str) and red.lower() in ("mean", "sum", "none"):
            return red.lower()
        if loss.shape == tile_pred.shape:
            return "none"
        if loss.numel() == 1 or loss.ndim == 0:
            return "mean"
        return "none"

    def reduce(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x2 = self._masked_select_safe(x, mask)
        match self.reduction:
            case "mean":
                return x2.mean()
            case "sum":
                return x2.sum()
            case "none":
                return x2
        return x2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = self._mask(pred, target)

        # Fast path: no tiling (still handle scalar base reductions correctly).
        if self.tile_dim is None or self.tile_size is None:
            v = self.base(pred, target)
            base_red = self._infer_base_reduction(v, pred)

            if self.reduction == "none":
                return self._masked_select_safe(v, mask)

            if base_red == "none" or v.shape == pred.shape:
                return self.reduce(v, mask)

            # Reduced output: compute global sum/mean consistently.
            if mask is not None and mask.shape == pred.shape:
                pv = pred[mask]
                tv = target[mask]
                n = int(pv.numel())
                if n == 0:
                    return pred.new_zeros(())
                v = self.base(pv, tv)
            else:
                n = int(pred.numel())

            if v.numel() == 1:
                v_scalar = v.reshape(())
                if base_red == "sum":
                    total_sum = v_scalar
                else:  # assume mean
                    total_sum = v_scalar * float(n)
                if self.reduction == "sum":
                    return total_sum
                return total_sum / float(max(n, 1))

            # Fallback: treat as elementwise-ish.
            if self.reduction == "sum":
                return v.sum()
            return v.mean()

        # --- Tiled path ---
        nd = int(pred.ndim)
        td = int(self.tile_dim) + nd if int(self.tile_dim) < 0 else int(self.tile_dim)
        td = max(0, min(td, nd - 1))
        N = int(pred.shape[td])

        # Accumulate in fp32 when inputs are low precision.
        acc_dtype = torch.float32 if pred.dtype in (torch.float16, torch.bfloat16) else pred.dtype
        total_sum = pred.new_tensor(0.0, dtype=acc_dtype)
        total_count: int = 0
        parts: List[torch.Tensor] = []

        start = 0
        while start < N:
            end = min(N, start + int(self.tile_size))
            sl = [slice(None)] * nd
            sl[td] = slice(start, end)
            sl_tuple = tuple(sl)

            pv = pred[sl_tuple]
            tv = target[sl_tuple]
            mv = mask[sl_tuple] if mask is not None else None

            elem = self.base(pv, tv)

            if self.reduction == "none":
                flat = elem.reshape(-1)
                if mv is not None and elem.shape == mv.shape:
                    flat = flat[mv.reshape(-1)]
                parts.append(flat)
                start = end
                continue

            base_red = self._infer_base_reduction(elem, pv)

            # Elementwise loss: can be masked on the output.
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

            # Reduced loss (typically scalar). For correctness under masking, recompute on masked selection.
            if mv is not None and mv.shape == pv.shape:
                pv2 = pv[mv]
                tv2 = tv[mv]
                n = int(pv2.numel())
                if n == 0:
                    start = end
                    continue
                elem2 = self.base(pv2, tv2)
            else:
                n = int(pv.numel())
                elem2 = elem

            if elem2.numel() == 1:
                v_scalar = elem2.reshape(())
                if base_red == "sum":
                    total_sum = total_sum + v_scalar
                else:  # assume mean
                    total_sum = total_sum + v_scalar * float(n)
                total_count += int(n)
            else:
                total_sum = total_sum + elem2.sum()
                total_count += int(elem2.numel())

            start = end

        match self.reduction:
            case "none":
                return torch.cat(parts, dim=0) if parts else pred.new_zeros(())
            case "sum":
                return total_sum
            case "mean":
                denom = float(max(total_count, 1))
                return total_sum / denom
        return pred.new_zeros(())


@dataclass
class LossWeightController:
    momentum: float = 0.95
    min_weight: float = 1e-6
    max_weight: float = 1.0 - 1e-6
    eps: float = 1e-06
    top_avg: float = 0.75
    bottom_avg: float = 0.25

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

    def update(self, top_loss: Optional[torch.Tensor], bottom_loss: Optional[torch.Tensor]) -> None:
        # Controller is pure-Python and expected to run on CPU-side scalars.
        if top_loss is not None:
            try:
                top_val = float(top_loss.detach().abs().mean().item())
            except Exception:
                top_val = float(self.eps)
            self.top_avg = self.momentum * self.top_avg + (1.0 - self.momentum) * max(top_val, self.eps)
        if bottom_loss is not None:
            try:
                bottom_val = float(bottom_loss.detach().abs().mean().item())
            except Exception:
                bottom_val = float(self.eps)
            self.bottom_avg = self.momentum * self.bottom_avg + (1.0 - self.momentum) * max(bottom_val, self.eps)
