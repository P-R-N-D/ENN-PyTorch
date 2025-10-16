from __future__ import annotations

from typing import Any, Callable, Iterable

__all__ = [
    'Autocast',
    'ScaledDotProductAttention',
    'GatedMultiScaleRetention',
    'AdamW',
    'attention_flops_bshd',
    'register_flop_hooks',
    'register_nvtx_flops_getter',
    'get_total_flops',
    'NVTXCounterMode',
    'nvtx_soft_add',
    'get_device',
    'secure_torch',
    'SDPBackend',
    'sdpa_kernel',
    '_to_sdpa_backends',
    'VarianceThreshold',
]

import torch
from torch import Tensor

from .compat import secure_torch

secure_torch()

class VarianceThreshold:
    def __init__(
        self,
        threshold: float = 0.0,
        unbiased: bool = True,
        device: str = 'cpu',
        dtype: Any = torch.float64,
    ) -> None:
        self.threshold = float(threshold)
        self.unbiased = bool(unbiased)
        self.device = torch.device(device)
        self.dtype = dtype
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean = None
        self.M2 = None

    @torch.no_grad()
    def partial_fit(self, X: Tensor) -> 'VarianceThreshold':
        X = X.to(self.device, self.dtype)
        if X.numel() == 0:
            return self
        b, _ = X.shape
        batch_mean = X.mean(dim=0)
        Xc = X - batch_mean
        batch_M2 = (Xc * Xc).sum(dim=0)
        if self.n == 0:
            self.mean, self.M2, self.n = (batch_mean, batch_M2, b)
        else:
            n_a, n_b = (self.n, b)
            delta = batch_mean - self.mean
            n = n_a + n_b
            self.mean = self.mean + delta * (n_b / n)
            self.M2 = self.M2 + batch_M2 + delta * delta * (n_a * n_b / n)
            self.n = n
        return self

    @torch.no_grad()
    def finalize(self) -> 'VarianceThreshold':
        if self.n <= 1:
            raise ValueError('Not enough samples to estimate variance.')
        denom = self.n - 1 if self.unbiased else self.n
        self.variances_ = self.M2 / denom
        self.feature_mask_ = self.variances_ > self.threshold
        self.n_features_in_ = int(self.variances_.numel())
        self.n_features_out_ = int(self.feature_mask_.sum().item())
        return self

    @torch.no_grad()
    def transform(self, X: Tensor) -> Tensor:
        if not hasattr(self, 'feature_mask_'):
            raise RuntimeError('Call finalize() before transform().')
        return X[..., self.feature_mask_.to(X.device)]

class StandardScaler:
    def __init__(
        self,
        with_mean: bool = True,
        with_std: bool = True,
        eps: float = 1e-8,
        device: str = 'cpu',
        dtype: Any = torch.float64,
    ) -> None:
        self.with_mean = bool(with_mean)
        self.with_std = bool(with_std)
        self.eps = float(eps)
        self.device = torch.device(device)
        self.dtype = dtype
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean = None
        self.M2 = None
        self.var_ = None
        self.mean_ = None
        self.scale_ = None
        self.min = None
        self.max = None
        self.min_ = None
        self.max_ = None

    @torch.no_grad()
    def partial_fit(self, X: Tensor) -> 'StandardScaler':
        X = X.to(self.device, self.dtype)
        if X.numel() == 0:
            return self
        b, _ = X.shape
        batch_mean = X.mean(dim=0)
        Xc = X - batch_mean
        batch_M2 = (Xc * Xc).sum(dim=0)
        _min_res = torch.nanmin(X, dim=0)
        _max_res = torch.nanmax(X, dim=0)
        bmin = getattr(_min_res, 'values', _min_res[0] if isinstance(_min_res, (tuple, list)) else _min_res)
        bmax = getattr(_max_res, 'values', _max_res[0] if isinstance(_max_res, (tuple, list)) else _max_res)
        if self.n == 0:
            self.mean, self.M2, self.n = (batch_mean, batch_M2, b)
            self.min, self.max = (bmin, bmax)
        else:
            n_a, n_b = (self.n, b)
            delta = batch_mean - self.mean
            n = n_a + n_b
            self.mean = self.mean + delta * (n_b / n)
            self.M2 = self.M2 + batch_M2 + delta * delta * (n_a * n_b / n)
            self.n = n
            self.min = torch.minimum(self.min, bmin) if self.min is not None else bmin
            self.max = torch.maximum(self.max, bmax) if self.max is not None else bmax
        return self

    @torch.no_grad()
    def finalize(self) -> 'StandardScaler':
        if self.n <= 1:
            raise ValueError('Not enough samples to estimate variance.')
        denom = self.n - 1
        var = self.M2 / denom
        self.mean_ = self.mean.clone() if self.with_mean else torch.zeros_like(var)
        if self.with_std:
            scale = torch.sqrt(torch.clamp(var, min=0.0))
            scale[scale < self.eps] = 1.0
            self.scale_ = scale
        else:
            self.scale_ = torch.ones_like(var)
        self.var_ = var
        self.min_ = self.min.clone() if self.min is not None else torch.full_like(var, float('nan'))
        self.max_ = self.max.clone() if self.max is not None else torch.full_like(var, float('nan'))
        return self

    @torch.no_grad()
    def transform(
        self,
        X: Tensor,
        *args: Any,
        clip: bool = False,
        clip_sigma: float | None = None,
        **kwargs: Any
    ) -> Tensor:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError('Call finalize() before transform().')
        mu = self.mean_.to(X.device, X.dtype)
        sc = self.scale_.to(X.device, X.dtype)
        Z = (X - mu) / sc
        if clip:
            if clip_sigma is not None and clip_sigma > 0:
                zmin = torch.full_like(Z, -float(clip_sigma))
                zmax = torch.full_like(Z, float(clip_sigma))
            else:
                mn = self.min_.to(X.device, X.dtype) if self.min_ is not None else None
                mx = self.max_.to(X.device, X.dtype) if self.max_ is not None else None
                if mn is not None and mx is not None:
                    zmin = (mn - mu) / sc
                    zmax = (mx - mu) / sc
                else:
                    zmin = zmax = None
            if zmin is not None and zmax is not None:
                Z = torch.maximum(Z, zmin.expand_as(Z))
                Z = torch.minimum(Z, zmax.expand_as(Z))
        return Z

    @torch.no_grad()
    def inverse_transform(self, X: Tensor) -> Tensor:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError('Call finalize() before inverse_transform().')
        mu = self.mean_.to(X.device, X.dtype)
        sc = self.scale_.to(X.device, X.dtype)
        return X * sc + mu

class IncrementalPCA:
    def __init__(
        self,
        n_components: int,
        method: str = 'cov',
        center: bool = True,
        lr: float = 0.1,
        oja_epochs: int = 1,
        device: str = 'cpu',
        stats_dtype: Any = torch.float64,
    ) -> None:
        self.k = int(n_components)
        self.method = str(method)
        self.center = bool(center)
        self.lr = float(lr)
        self.oja_epochs = int(oja_epochs)
        self.device = torch.device(device)
        self.stats_dtype = stats_dtype
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n = 0
        self.mu = None
        self.S = None

    @torch.no_grad()
    def partial_fit(self, X: Tensor) -> 'IncrementalPCA':
        if self.method != 'cov':
            raise RuntimeError("partial_fit is only for method='cov'. Use fit_oja.")
        X = X.to(self.device, self.stats_dtype)
        if X.numel() == 0:
            return self
        b, _ = X.shape
        batch_mu = X.mean(dim=0)
        Xc = X - batch_mu
        batch_S = Xc.T @ Xc
        if self.n == 0:
            self.mu, self.S, self.n = (batch_mu, batch_S, b)
        else:
            n_a, n_b = (self.n, b)
            delta = (batch_mu - self.mu).unsqueeze(1)
            n = n_a + n_b
            self.S = self.S + batch_S + n_a * n_b / n * (delta @ delta.T)
            self.mu = self.mu + (batch_mu - self.mu) * (n_b / n)
            self.n = n
        return self

    @torch.no_grad()
    def finalize_cov(self) -> 'IncrementalPCA':
        if self.n <= 1:
            raise ValueError('Not enough samples for covariance PCA.')
        cov = self.S / (self.n - 1)
        evals, evecs = torch.linalg.eigh(cov)
        idx = torch.argsort(evals, descending=True)
        evals = evals[idx][:self.k]
        comps = evecs[:, idx[:self.k]].T.contiguous()
        self.components_ = comps.to(self.device)
        self.explained_variance_ = evals.to(self.device)
        self.explained_variance_ratio_ = (evals / torch.clamp(evals.sum(), min=1e-12)).to(self.device)
        self.mean_ = self.mu.to(self.device)
        return self

    @torch.no_grad()
    def fit_oja(
        self,
        data_iter: Callable[[], Iterable[Tensor]],
        d: int,
    ) -> 'IncrementalPCA':
        device = self.device
        W = torch.randn(d, self.k, device=device)
        W, _ = torch.linalg.qr(W, mode='reduced')
        mu = torch.zeros(d, dtype=self.stats_dtype, device=device) if self.center else None
        n_seen = 0
        for _ in range(self.oja_epochs):
            for X in data_iter():
                X = X.to(device)
                if self.center:
                    b = X.shape[0]
                    n_a, n_b = (n_seen, b)
                    batch_mu = X.mean(dim=0).to(mu.dtype)
                    mu = batch_mu if n_seen == 0 else mu + (batch_mu - mu) * (n_b / (n_a + n_b))
                    n_seen += b
                    Xc = X - mu.to(X.dtype)
                else:
                    Xc = X
                Y = Xc @ W
                grad = Xc.T @ Y / max(1, Xc.shape[0])
                W = W + self.lr * grad
                W, _ = torch.linalg.qr(W, mode='reduced')
        self.components_ = W.T.contiguous()
        self.mean_ = mu if self.center else torch.zeros(d, device=device)
        return self

    @torch.no_grad()
    def transform(self, X: Tensor) -> Tensor:
        if self.components_ is None:
            raise RuntimeError('Call finalize_cov() or fit_oja() first.')
        Xc = X - self.mean_.to(X.device, X.dtype) if self.center else X
        return Xc @ self.components_.T

from .optimization import (
    Autocast,
    ScaledDotProductAttention,
    GatedMultiScaleRetention,
    AdamW,
    attention_flops_bshd,
    register_flop_hooks,
    register_nvtx_flops_getter,
    get_total_flops,
    NVTXCounterMode,
    nvtx_soft_add,
)
from .capability import get_device
from .compat import SDPBackend, sdpa_kernel, _to_sdpa_backends
