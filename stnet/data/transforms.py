# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from contextlib import suppress
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from tensordict import TensorDictBase

from .datatype import to_tuple, to_torch_tensor

class _ScalerBase:
    def __init__(
        self,
        *args: Any,
        device: str = "cpu",
        dtype: Any = torch.float64,
        **kwargs: Any,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.reset()

    def reset(self) -> None:
        raise NotImplementedError

    def _prepare(self, X: Tensor) -> Tensor:
        return X.to(self.device, self.dtype)

class VarianceThreshold(_ScalerBase):
    def __init__(
        self,
        threshold: float = 0.0,
        unbiased: bool = True,
        *args: Any,
        device: str = "cpu",
        dtype: Any = torch.float64,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.threshold = float(threshold)
        self.unbiased = bool(unbiased)

    def reset(self) -> None:
        self.n = 0
        self.mean: Tensor | None = None
        self.M2: Tensor | None = None

    @torch.no_grad()
    def partial_fit(self, X: Tensor) -> VarianceThreshold:
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
            if self.mean is None or self.M2 is None:
                raise RuntimeError("VarianceThreshold state corrupted")
            delta = batch_mean - self.mean
            n = self.n + b
            self.mean = self.mean + delta * (b / n)
            self.M2 = self.M2 + batch_M2 + delta * delta * (self.n * b / n)
            self.n = n
        return self

    @torch.no_grad()
    def finalize(self) -> VarianceThreshold:
        if self.n <= 1 or self.M2 is None:
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
        with_mean: bool = True,
        with_std: bool = True,
        eps: float = 1e-08,
        *args: Any,
        device: str = "cpu",
        dtype: Any = torch.float64,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.with_mean = bool(with_mean)
        self.with_std = bool(with_std)
        self.eps = float(eps)

    def reset(self) -> None:
        self.n = 0
        self.mean: Tensor | None = None
        self.M2: Tensor | None = None
        self.var_: Tensor | None = None
        self.mean_: Tensor | None = None
        self.scale_: Tensor | None = None
        self.min: Tensor | None = None
        self.max: Tensor | None = None
        self.min_: Tensor | None = None
        self.max_: Tensor | None = None

    @torch.no_grad()
    def partial_fit(self, X: Tensor) -> StandardScaler:
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
            if self.mean is None or self.M2 is None:
                raise RuntimeError("StandardScaler state corrupted")
            delta = batch_mean - self.mean
            n = self.n + b
            self.mean = self.mean + delta * (b / n)
            self.M2 = self.M2 + batch_M2 + delta * delta * (self.n * b / n)
            self.n = n
            self.min = (
                torch.minimum(self.min, bmin) if self.min is not None else bmin
            )
            self.max = (
                torch.maximum(self.max, bmax) if self.max is not None else bmax
            )
        return self

    @torch.no_grad()
    def finalize(self) -> StandardScaler:
        if self.n <= 1 or self.M2 is None or self.mean is None:
            raise ValueError("Not enough samples to estimate variance.")
        denom = self.n - 1
        var = self.M2 / denom
        self.mean_ = (
            self.mean.clone() if self.with_mean else torch.zeros_like(var)
        )
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
        *args: Any,
        clip: bool = False,
        clip_sigma: float | None = None,
        **kwargs: Any,
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
        method: str = "cov",
        center: bool = True,
        *args: Any,
        device: str = "cpu",
        dtype: Any = torch.float64,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
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
    def partial_fit(self, X: Tensor) -> IncrementalPCA:
        X = self._prepare(X)
        if X.numel() == 0:
            return self
        if self.method == "cov":
            return self._partial_fit_cov(X)
        if self.method == "oja":
            return self._partial_fit_oja(X)
        raise ValueError(f"Unsupported method: {self.method}")

    def _partial_fit_cov(self, X: Tensor) -> IncrementalPCA:
        if self.center:
            batch_mean = X.mean(dim=0, keepdim=True)
            Xc = X - batch_mean
        else:
            batch_mean = torch.zeros(
                1, X.shape[1], device=X.device, dtype=X.dtype
            )
            Xc = X
        denom = max(1, X.shape[0] - (1 if self.center else 0))
        cov = Xc.T @ Xc / denom
        eigvals, eigvecs = torch.linalg.eigh(cov)
        top_vals = eigvals.flip(0)[: self.n_components]
        top_vecs = eigvecs.flip(1)[:, : self.n_components]
        if self._components is None:
            self._components = top_vecs.to(self.device, self.dtype)
            self._eigvals = top_vals.to(self.device, self.dtype)
            self._mean = (
                batch_mean.mean(dim=0)
                if self.center
                else torch.zeros_like(top_vals)
            )
        else:
            stacked = torch.cat(
                [self._components, top_vecs.to(self.device, self.dtype)], dim=1
            )
            q, _ = torch.linalg.qr(stacked, mode="reduced")
            self._components = q[:, : self.n_components]
            self._eigvals = top_vals.to(self.device, self.dtype)
            if self.center and self._mean is not None:
                total = self._count + X.shape[0]
                self._mean = (
                    self._mean * self._count + batch_mean.sum(dim=0)
                ) / total
        self._count += X.shape[0]
        return self

    def _partial_fit_oja(self, X: Tensor) -> IncrementalPCA:
        if self._components is None:
            dim = X.shape[1]
            comps = torch.randn(
                dim, self.n_components, device=X.device, dtype=X.dtype
            )
            comps = torch.linalg.qr(comps).Q
            self._components = comps.to(self.device, self.dtype)
            self._eigvals = torch.zeros(
                self.n_components, device=self.device, dtype=self.dtype
            )
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
                self._components[:, i] = F.normalize(update, dim=0)
                self._eigvals[i] = (1 - eta) * self._eigvals[
                    i
                ] + eta * proj * proj
        self._count += X.shape[0]
        return self

    @torch.no_grad()
    def finalize(self) -> IncrementalPCA:
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

def _assert_finites(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if torch.is_floating_point(tensor) or torch.is_complex(tensor):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} tensor contains non-finite values")
    return tensor

def _preprocess_x(x_tuple: Any) -> torch.Tensor:
    try:
        values = [float(v) for v in to_tuple(x_tuple)]
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "preprocess: feature tuples must contain only numeric values. "
            f"Invalid value={x_tuple!r}"
        ) from exc
    for value in values:
        if not math.isfinite(value):
            raise ValueError("preprocess: feature tuples must be finite")
    tensor = torch.as_tensor(values, dtype=torch.float64)
    return _assert_finites(tensor, "feature")

def _preprocess_batch(
    x_value: Any, y_value: Any
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple], Tuple[int, ...]] | None:
    if isinstance(x_value, torch.Tensor):
        feature_tensor: Any = x_value
    else:
        feature_tensor = None
        for attr in ("to_torch_tensor", "to_torch", "to_tensor", "as_tensor"):
            if not hasattr(x_value, attr):
                continue
            with suppress(Exception):
                candidate = getattr(x_value, attr)()
            if candidate is not None:
                feature_tensor = candidate
                if isinstance(feature_tensor, torch.Tensor):
                    break
                break
        if not isinstance(feature_tensor, torch.Tensor):
            try:
                feature_tensor = torch.as_tensor(x_value)
            except Exception:
                feature_tensor = None
    if not isinstance(feature_tensor, torch.Tensor):
        return None
    try:
        label_tensor = to_torch_tensor(y_value)
    except Exception:
        return None
    if not isinstance(label_tensor, torch.Tensor):
        return None
    feature_tensor = _assert_finites(
        feature_tensor.detach().to(dtype=torch.float64), "feature"
    )
    if feature_tensor.dim() == 0:
        feature_tensor = feature_tensor.reshape(1, 1)
    elif feature_tensor.dim() == 1:
        feature_tensor = feature_tensor.reshape(-1, 1)
    else:
        batch_dim = int(feature_tensor.shape[0]) if feature_tensor.shape else 1
        feature_tensor = feature_tensor.reshape(batch_dim, -1)
    batch_size = int(feature_tensor.shape[0])
    label_tensor = label_tensor.detach()
    label_tensor = _assert_finites(label_tensor, "label")
    if label_tensor.dim() == 0:
        label_tensor = label_tensor.unsqueeze(0)
    if label_tensor.dim() == 1 and label_tensor.shape[0] == batch_size:
        label_tensor = label_tensor.unsqueeze(-1)
    if label_tensor.shape[0] != batch_size:
        return None
    label_shape = tuple(label_tensor.shape[1:])
    batch_keys = [(int(index),) for index in range(batch_size)]
    return (feature_tensor, label_tensor, batch_keys, label_shape)

def _preprocess_y(value: Any) -> torch.Tensor:
    tensor = to_torch_tensor(value)
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(value)
    return tensor.detach()


def preprocess(
    data: Union[Dict[Tuple, torch.Tensor], TensorDictBase]
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple], Tuple[int, ...]]:
    if isinstance(data, TensorDictBase):
        if "features" not in data.keys():
            raise ValueError("preprocess(TensorDict): missing 'features'")
        feats = torch.as_tensor(data.get("features"))
        if feats.ndim == 1:
            feats = feats.unsqueeze(0)
        if "labels" in data.keys():
            labels = torch.as_tensor(data.get("labels"))
        elif "labels_flat" in data.keys():
            labels = torch.as_tensor(data.get("labels_flat"))
        else:
            raise ValueError("preprocess(TensorDict): missing 'labels' or 'labels_flat'")
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        label_shape = tuple(labels.shape[1:]) if labels.dim() > 1 else (1,)
        keys = [(int(i),) for i in range(int(feats.shape[0]))]
        return (feats, labels, keys, label_shape)
    if isinstance(data, dict) and "X" in data and ("Y" in data):
        x, y = (data["X"], data["Y"])
        batch_result = _preprocess_batch(x, y)
        if batch_result is not None:
            return batch_result
        xr, yt = (
            _preprocess_x(x).unsqueeze(0),
            _assert_finites(_preprocess_y(y), "label"),
        )
        if yt.dim() == 0 or yt.dim() == 1:
            yt = yt.unsqueeze(0)
        keys = [to_tuple(x)]
        label_shape = tuple(yt.shape[1:])
        return (xr, yt, keys, label_shape)
    elif isinstance(data, (tuple, list)) and len(data) >= 2:
        x, y = (data[0], data[1])
        batch_result = _preprocess_batch(x, y)
        if batch_result is not None:
            return batch_result
        xr = _preprocess_x(x).unsqueeze(0)
        yt = _assert_finites(_preprocess_y(y), "label")
        if yt.dim() == 0:
            yt = yt.unsqueeze(0)
        elif yt.shape[0] != 1:
            yt = yt.unsqueeze(0)
        keys = [to_tuple(x)]
        label_shape = tuple(yt.shape[1:])
        return (xr, yt, keys, label_shape)
    elif isinstance(data, dict) and len(data) > 0:
        items = list(data.items())
        if any((isinstance(k, str) for k, _ in items)):
            raise TypeError(
                "preprocess: keys in a multi-sample dict must be tuples. "
                "Provide single samples as {'X': ..., 'Y': ...}."
            )
        keys: List[Tuple] = [to_tuple(k) for k, _ in items]
        feats = torch.stack([_preprocess_x(k) for k in keys], dim=0)
        lbl_list = [
            _assert_finites(_preprocess_y(v), "label")
            for _, v in items
        ]
        if all((t.shape == lbl_list[0].shape for t in lbl_list)):
            labels = torch.stack(lbl_list, dim=0)
        else:
            labels = torch.cat([t.unsqueeze(0) for t in lbl_list], dim=0)
        labels = _assert_finites(labels, "label")
        label_shape = tuple(labels.shape[1:])
        return (feats, labels, keys, label_shape)
    else:
        raise ValueError(
            "preprocess: unsupported input format. Provide a dict or an (X, Y) pair."
        )

def postprocess(
    keys: List[Tuple], preds: torch.Tensor | Sequence[torch.Tensor]
) -> Dict[Tuple, torch.Tensor]:
    if isinstance(preds, torch.Tensor):
        if preds.dim() == 0:
            preds = preds.unsqueeze(0)
        if preds.shape[0] != len(keys):
            raise ValueError(
                f"preds batch={preds.shape[0]} != len(keys)={len(keys)}"
            )
        rows = [preds[i].detach().cpu() for i in range(len(keys))]
    else:
        if len(preds) != len(keys):
            raise ValueError(
                f"len(preds)={len(preds)} != len(keys)={len(keys)}"
            )
        rows = [
            p.detach().cpu()
            if isinstance(p, torch.Tensor)
            else torch.as_tensor(p)
            for p in preds
        ]
    fixed_keys: List[Tuple] = []
    seen = set()
    for i, k in enumerate(keys):
        if not isinstance(k, tuple):
            try:
                k = tuple(k)
            except TypeError:
                k = (k,)
        k_out = k
        if k in seen:
            k_out = k + (i,)
        seen.add(k_out)
        fixed_keys.append(k_out)
    return {k: v for k, v in zip(fixed_keys, rows)}
