# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

__all__ = ["VarianceThreshold", "StandardScaler", "IncrementalPCA"]


class _ScalerBase:
    def __init__(self, *, device: str = "cpu", dtype: Any = torch.float64) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.reset()

    def reset(self) -> None:
        raise NotImplementedError

    def _prepare(self, X: Tensor) -> Tensor:
        return X.to(self.device, self.dtype)


class VarianceThreshold(_ScalerBase):
    def __init__(self, threshold: float = 0.0, unbiased: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.threshold = float(threshold)
        self.unbiased = bool(unbiased)

    def reset(self) -> None:
        self.n = 0
        self.mean = None
        self.M2 = None

    @torch.no_grad()
    def partial_fit(self, X: Tensor) -> "VarianceThreshold":
        X = self._prepare(X)
        if X.numel() == 0:
            return self
        b, _ = X.shape
        batch_mean = X.mean(dim=0)
        Xc = X - batch_mean
        batch_M2 = (Xc * Xc).sum(dim=0)
        if self.n == 0:
            self.mean, self.M2, self.n = (batch_mean, batch_M2, b)
        else:
            delta = batch_mean - self.mean
            n = self.n + b
            self.mean = self.mean + delta * (b / n)
            self.M2 = self.M2 + batch_M2 + delta * delta * (self.n * b / n)
            self.n = n
        return self

    @torch.no_grad()
    def finalize(self) -> "VarianceThreshold":
        if self.n <= 1:
            raise ValueError("Not enough samples to estimate variance.")
        denom = self.n - 1 if self.unbiased else self.n
        self.variances_ = self.M2 / denom
        self.feature_mask_ = self.variances_ > self.threshold
        self.n_features_in_ = int(self.variances_.numel())
        self.n_features_out_ = int(self.feature_mask_.sum().item())
        return self

    @torch.no_grad()
    def transform(self, X: Tensor) -> Tensor:
        if not hasattr(self, "feature_mask_"):
            raise RuntimeError("Call finalize() before transform().")
        return X[..., self.feature_mask_.to(X.device)]


class StandardScaler(_ScalerBase):
    def __init__(
        self,
        *,
        with_mean: bool = True,
        with_std: bool = True,
        eps: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.with_mean = bool(with_mean)
        self.with_std = bool(with_std)
        self.eps = float(eps)

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
    def partial_fit(self, X: Tensor) -> "StandardScaler":
        X = self._prepare(X)
        if X.numel() == 0:
            return self
        b, _ = X.shape
        batch_mean = X.mean(dim=0)
        Xc = X - batch_mean
        batch_M2 = (Xc * Xc).sum(dim=0)
        _min_res = torch.nanmin(X, dim=0)
        _max_res = torch.nanmax(X, dim=0)
        bmin = getattr(
            _min_res,
            "values",
            _min_res[0] if isinstance(_min_res, (tuple, list)) else _min_res,
        )
        bmax = getattr(
            _max_res,
            "values",
            _max_res[0] if isinstance(_max_res, (tuple, list)) else _max_res,
        )
        if self.n == 0:
            self.mean, self.M2, self.n = (batch_mean, batch_M2, b)
            self.min, self.max = (bmin, bmax)
        else:
            delta = batch_mean - self.mean
            n = self.n + b
            self.mean = self.mean + delta * (b / n)
            self.M2 = self.M2 + batch_M2 + delta * delta * (self.n * b / n)
            self.n = n
            self.min = (
                torch.minimum(self.min, bmin)
                if self.min is not None
                else bmin
            )
            self.max = (
                torch.maximum(self.max, bmax)
                if self.max is not None
                else bmax
            )
        return self

    @torch.no_grad()
    def finalize(self) -> "StandardScaler":
        if self.n <= 1:
            raise ValueError("Not enough samples to estimate variance.")
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
        self.min_ = (
            self.min.clone()
            if self.min is not None
            else torch.full_like(var, float("nan"))
        )
        self.max_ = (
            self.max.clone()
            if self.max is not None
            else torch.full_like(var, float("nan"))
        )
        return self

    @torch.no_grad()
    def transform(
        self,
        X: Tensor,
        *,
        clip: bool = False,
        clip_sigma: float | None = None,
    ) -> Tensor:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Call finalize() before transform().")
        mu = self.mean_.to(X.device, X.dtype)
        sc = self.scale_.to(X.device, X.dtype)
        Z = (X - mu) / sc
        if clip:
            if clip_sigma is not None and clip_sigma > 0:
                zmin = torch.full_like(Z, -float(clip_sigma))
                zmax = torch.full_like(Z, float(clip_sigma))
            else:
                mn = (
                    self.min_.to(X.device, X.dtype)
                    if self.min_ is not None
                    else None
                )
                mx = (
                    self.max_.to(X.device, X.dtype)
                    if self.max_ is not None
                    else None
                )
                if mn is not None and mx is not None:
                    zmin = (mn - mu) / sc
                    zmax = (mx - mu) / sc
                else:
                    zmin = None
                    zmax = None
            if zmin is not None and zmax is not None:
                Z = torch.maximum(Z, zmin.expand_as(Z))
                Z = torch.minimum(Z, zmax.expand_as(Z))
        return Z

    @torch.no_grad()
    def inverse_transform(self, X: Tensor) -> Tensor:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Call finalize() before inverse_transform().")
        mu = self.mean_.to(X.device, X.dtype)
        sc = self.scale_.to(X.device, X.dtype)
        return X * sc + mu


class IncrementalPCA(_ScalerBase):
    def __init__(
        self,
        n_components: int,
        *,
        method: str = "cov",
        center: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_components = int(n_components)
        self.method = str(method)
        self.center = bool(center)
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")
        self._components: Tensor | None = None
        self._mean: Tensor | None = None
        self._eigvals: Tensor | None = None

    def reset(self) -> None:
        self._components = None
        self._mean = None
        self._eigvals = None
        self._count = 0

    @torch.no_grad()
    def partial_fit(self, X: Tensor) -> "IncrementalPCA":
        X = self._prepare(X)
        if X.numel() == 0:
            return self
        if self.method == "cov":
            return self._partial_fit_cov(X)
        if self.method == "oja":
            return self._partial_fit_oja(X)
        raise ValueError(f"Unsupported method: {self.method}")

    def _partial_fit_cov(self, X: Tensor) -> "IncrementalPCA":
        if self.center:
            batch_mean = X.mean(dim=0, keepdim=True)
            Xc = X - batch_mean
        else:
            batch_mean = torch.zeros(1, X.shape[1], device=X.device, dtype=X.dtype)
            Xc = X
        denom = max(1, X.shape[0] - (1 if self.center else 0))
        cov = Xc.T @ Xc / denom
        eigvals, eigvecs = torch.linalg.eigh(cov)
        top_vals = eigvals.flip(0)[: self.n_components]
        top_vecs = eigvecs.flip(1)[:, : self.n_components]
        if self._components is None:
            self._components = top_vecs.to(self.device, self.dtype)
            self._eigvals = top_vals.to(self.device, self.dtype)
            self._mean = batch_mean.mean(dim=0) if self.center else torch.zeros_like(top_vals)
        else:
            stacked = torch.cat([self._components, top_vecs.to(self.device, self.dtype)], dim=1)
            q, _ = torch.linalg.qr(stacked, mode="reduced")
            self._components = q[:, : self.n_components]
            self._eigvals = top_vals.to(self.device, self.dtype)
            if self.center and self._mean is not None:
                total = self._count + X.shape[0]
                self._mean = (self._mean * self._count + batch_mean.sum(dim=0)) / total
        self._count += X.shape[0]
        return self

    def _partial_fit_oja(self, X: Tensor) -> "IncrementalPCA":
        if self._components is None:
            dim = X.shape[1]
            comps = torch.randn(dim, self.n_components, device=X.device, dtype=X.dtype)
            comps = torch.linalg.qr(comps).Q
            self._components = comps.to(self.device, self.dtype)
            self._eigvals = torch.zeros(self.n_components, device=self.device, dtype=self.dtype)
            self._mean = torch.zeros(dim, device=self.device, dtype=self.dtype)
        eta0 = 1.0
        for t, x in enumerate(X, start=1 + self._count):
            if self.center and self._mean is not None:
                self._mean = (self._mean * (t - 1) + x) / t
                x = x - self._mean
            eta = eta0 / float(t)
            for i in range(self.n_components):
                w = self._components[:, i]
                proj = torch.dot(w, x)
                update = w + eta * proj * (x - proj * w)
                self._components[:, i] = torch.nn.functional.normalize(update, dim=0)
                self._eigvals[i] = (1 - eta) * self._eigvals[i] + eta * proj * proj
        self._count += X.shape[0]
        return self

    @torch.no_grad()
    def finalize(self) -> "IncrementalPCA":
        if self._components is None:
            raise RuntimeError("Call partial_fit() before finalize().")
        return self

    @torch.no_grad()
    def transform(self, X: Tensor) -> Tensor:
        if self._components is None:
            raise RuntimeError("Call finalize() before transform().")
        centered = X.to(self.device, self.dtype)
        if self.center and self._mean is not None:
            centered = centered - self._mean
        return centered @ self._components
