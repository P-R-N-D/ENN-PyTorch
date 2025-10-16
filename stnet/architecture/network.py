# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Sequence, Tuple, Union, cast

from math import prod

import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True)
class PatchParameters:
    is_square: bool = False
    patch_size_1d: int = 16
    grid_size_2d: Optional[Union[int, Tuple[int, int], List[int]]] = None
    patch_size_2d: Union[int, Tuple[int, int], List[int]] = 4
    is_cube: bool = False
    grid_size_3d: Optional[Union[int, Tuple[int, int, int], List[int]]] = None
    patch_size_3d: Union[int, Tuple[int, int, int], List[int]] = (2, 2, 2)
    dropout: float = 0.0
    use_padding: bool = True


@dataclass
class Config:
    device: Optional[torch.device | str] = None
    microbatch: int = 64
    dropout: float = 0.1
    normalize_method: str = 'layernorm'
    depth: int = 128
    heads: int = 4
    spatial_depth: int = 4
    temporal_depth: int = 4
    mlp_ratio: float = 4.0
    drop_path: float = 0.0
    spatial_latent_tokens: int = 64
    temporal_latent_tokens: int = 64
    data_definition: str = 'spatiotemporal'
    patch: PatchParameters = field(default_factory=PatchParameters)
    use_linear_branch: bool = False
    use_compilation: bool = False
    compile_mode: str = 'default'
    loss_space: str = 'z'

from .module import SpatioTemporalNet, Meta, MetaNet
from ..toolkit.optimization import Autocast, compile
from ..toolkit.compat import patch_torch

patch_torch()


class LossWeightController(Protocol):
    def weights(self) -> Tuple[float, float]:
        ...

    def update(
        self,
        top_loss: Optional[torch.Tensor],
        bottom_loss: Optional[torch.Tensor],
    ) -> None:
        ...


class Model(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_shape: Sequence[int],
        *args: Any,
        config: Config,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_shape = tuple((int(x) for x in out_shape))
        self.out_dim = int(prod(self.out_shape))
        self._loss_space = str(getattr(config, 'loss_space', 'z')).lower()
        if config.device is not None:
            self._device = torch.device(config.device)
        else:
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                dev = 'cuda'
            elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                dev = 'mps'
            elif hasattr(torch, 'is_vulkan_available') and torch.is_vulkan_available():
                dev = 'vulkan'
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                dev = 'xpu'
            else:
                dev = 'cpu'
            self._device = torch.device(dev)
        self.is_norm_linear = bool(getattr(config, 'use_linear_branch', False))
        self.linear_branch = nn.Linear(self.in_dim, self.out_dim).to(self._device) if self.is_norm_linear else None
        self._local = SpatioTemporalNet(
            self.in_dim,
            self.out_shape,
            config=config,
        ).to(self._device)
        global_net = MetaNet(
            int(config.depth),
            int(config.heads),
            depth=max(1, int(getattr(config, 'temporal_depth', 1))),
            mlp_ratio=float(getattr(config, 'mlp_ratio', 4.0)),
            dropout=float(getattr(config, 'dropout', 0.0)),
            drop_path=float(getattr(config, 'drop_path', 0.0)),
            norm_type=str(getattr(config, 'normalize_method', 'layernorm')),
        ).to(self._device)
        self._global = global_net
        setattr(self, 'global', global_net)
        self.microbatch = int(config.microbatch)
        if self.microbatch <= 0:
            raise ValueError(f'config.microbatch must be >= 1, got {config.microbatch}')
        try:
            self.register_buffer('output_baked_flag', torch.tensor(0, dtype=torch.uint8), persistent=True)
        except Exception:
            pass
        mode = str(getattr(config, 'compile_mode', 'default'))
        try:
            if bool(getattr(config, 'use_compilation', False)):
                self._local = compile(self._local, mode=mode, fullgraph=False, dynamic=False, backend='inductor')
            if bool(getattr(config, 'use_compilation', False)):
                compiled_global = compile(self._global, mode=mode, fullgraph=False, dynamic=False, backend='inductor')
                self._global = compiled_global
                setattr(self, 'global', compiled_global)
        except Exception:
            pass
        self.__config = config
        self._label_dim = int(prod(out_shape))
        self.register_buffer('y_stats_ready', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('y_eps', torch.tensor(1e-06, dtype=torch.float32))
        d = self._label_dim
        dev = getattr(self, '_device', torch.device('cpu'))
        self.register_buffer('y_min', torch.full((d,), float('inf'), device=dev, dtype=torch.float32))
        self.register_buffer('y_max', torch.full((d,), float('-inf'), device=dev, dtype=torch.float32))
        self.register_buffer('y_sum', torch.zeros(d, device=dev, dtype=torch.float64))
        self.register_buffer('y_sum2', torch.zeros(d, device=dev, dtype=torch.float64))
        self.register_buffer('y_count', torch.zeros(d, device=dev, dtype=torch.float64))
        self.register_buffer('y_mean', torch.zeros(d, device=dev, dtype=torch.float32))
        self.register_buffer('y_std', torch.ones(d, device=dev, dtype=torch.float32))
        self.register_buffer('x_seen_elems', torch.zeros((), device=dev, dtype=torch.float64))
        self.register_buffer('x_mean', torch.zeros(self.in_dim, dtype=torch.float32), persistent=True)
        self.register_buffer('x_std', torch.ones(self.in_dim, dtype=torch.float32), persistent=True)
        self.register_buffer('x_stats_ready', torch.tensor(False, dtype=torch.bool), persistent=True)
        self._x_eps: float = 1e-06
        self._input_scale_method: str = 'standard'
        self._x_sum: torch.Tensor | None = None
        self._x_sum2: torch.Tensor | None = None
        self._x_count: torch.Tensor | None = None

    def set_input_scale_method(self, method: str = 'standard') -> None:
        self._input_scale_method = 'standard' if str(method).lower() == 'standard' else 'none'

    @torch.no_grad()
    def update_x_stats(self, X: torch.Tensor) -> None:
        if X is None:
            return
        x = torch.as_tensor(X).detach()
        x = torch.atleast_2d(x)
        if x.dim() != 2:
            x = x.view(x.shape[0], -1)
        x64 = x.to(dtype=torch.float64, device='cpu')
        if self._x_sum is None or self._x_sum.shape[0] != x64.shape[1]:
            D = int(x64.shape[1])
            self._x_sum = torch.zeros(D, dtype=torch.float64)
            self._x_sum2 = torch.zeros(D, dtype=torch.float64)
            self._x_count = torch.zeros((), dtype=torch.int64)
        xnz = torch.nan_to_num(x64, nan=0.0)
        self._x_sum += xnz.sum(dim=0)
        self._x_sum2 += (xnz * xnz).sum(dim=0)
        self._x_count += int(x64.shape[0])

    @torch.no_grad()
    def finalize_x_stats(self) -> None:
        if self._x_sum is None or self._x_sum2 is None or self._x_count is None or (int(self._x_count.item()) == 0):
            self.x_stats_ready.fill_(False)
            return
        dev = next(self.parameters()).device
        s = self._x_sum.to(device=dev)
        s2 = self._x_sum2.to(device=dev)
        c = torch.tensor(float(int(self._x_count.item())), dtype=torch.float64, device=dev)
        if dist.is_initialized():
            dist.all_reduce(s, op=dist.ReduceOp.SUM)
            dist.all_reduce(s2, op=dist.ReduceOp.SUM)
            dist.all_reduce(c, op=dist.ReduceOp.SUM)
        s = s.cpu()
        s2 = s2.cpu()
        c = float(c.cpu().item())
        c = max(1.0, c)
        mean = (s / c).to(torch.float32)
        var = s2 / c - mean.to(torch.float64).pow(2)
        std = torch.sqrt(var.clamp_min(self._x_eps ** 2)).to(torch.float32)
        self.x_mean.data.copy_(mean)
        self.x_std.data.copy_(std)
        self.x_stats_ready.data.fill_(True)
        self._x_sum = None
        self._x_sum2 = None
        self._x_count = None

    def _normalize_inputs(self, X: torch.Tensor) -> torch.Tensor:
        if self._input_scale_method != 'standard' or not bool(self.x_stats_ready.item()):
            return X
        mu = self.x_mean.to(device=X.device, dtype=X.dtype)
        sd = self.x_std.to(device=X.device, dtype=X.dtype).clamp_min(self._x_eps)
        return (X - mu) / sd

    @torch.no_grad()
    def update_y_stats(self, y_raw: torch.Tensor) -> None:
        y = y_raw.detach().view(y_raw.shape[0], -1).to(device=self.y_min.device, dtype=torch.float32)
        _, d = y.shape
        if d != self._label_dim:
            raise ValueError(f'Target flattened dim {d} != model label_dim {self._label_dim}')
        _min_res = torch.nanmin(y, dim=0)
        _max_res = torch.nanmax(y, dim=0)
        batch_min = getattr(_min_res, 'values', _min_res[0] if isinstance(_min_res, (tuple, list)) else _min_res)
        batch_max = getattr(_max_res, 'values', _max_res[0] if isinstance(_max_res, (tuple, list)) else _max_res)
        batch_sum = torch.nansum(y, dim=0, dtype=torch.float64)
        batch_sum2 = torch.nansum(y.to(torch.float64) ** 2, dim=0)
        batch_cnt = torch.sum(torch.isfinite(y), dim=0, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_min, op=dist.ReduceOp.MIN)
            dist.all_reduce(batch_max, op=dist.ReduceOp.MAX)
            dist.all_reduce(batch_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_sum2, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_cnt, op=dist.ReduceOp.SUM)
        self.y_min.copy_(torch.fmin(self.y_min, batch_min))
        self.y_max.copy_(torch.fmax(self.y_max, batch_max))
        self.y_sum.add_(batch_sum)
        self.y_sum2.add_(batch_sum2)
        self.y_count.add_(batch_cnt)

    @torch.no_grad()
    def finalize_y_stats(self) -> None:
        valid = self.y_count > 0
        y_mean = torch.zeros_like(self.y_sum, dtype=torch.float64, device=self.y_sum.device)
        var = torch.zeros_like(self.y_sum2, device=self.y_sum2.device)
        y_mean[valid] = self.y_sum[valid] / self.y_count[valid]
        var[valid] = self.y_sum2[valid] / self.y_count[valid] - y_mean[valid] ** 2
        y_std = torch.sqrt(torch.clamp(var, min=float(self.y_eps.item()) ** 2 if hasattr(self, 'y_eps') else 1e-12))
        self.y_mean.copy_(y_mean.to(torch.float32))
        self.y_std.copy_(y_std.to(torch.float32))
        self.y_stats_ready.fill_(bool((self.y_count > 0).any().item()))

    def has_valid_y_stats(self) -> bool:
        try:
            ready = bool(self.y_stats_ready.item())
        except Exception:
            ready = False
        if not ready:
            return False
        if hasattr(self, 'y_std'):
            try:
                return bool((self.y_std > 0).any().item())
            except Exception:
                return False
        try:
            return bool((self.y_max > self.y_min).any().item())
        except Exception:
            return False

    def forward(
        self,
        features: torch.Tensor,
        *args: Any,
        labels_flat: Optional[torch.Tensor] = None,
        net_loss: Optional[nn.Module] = None,
        global_loss: Optional[nn.Module] = None,
        local_loss: Optional[nn.Module] = None,
        loss_weights: Optional[
            Union[Tuple[float, float], LossWeightController]
        ] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        features = self._normalize_inputs(features)
        if features.ndim == 3 and features.shape[1] == 1:
            features = features.view(features.shape[0], -1)
        assert features.ndim == 2 and features.shape[1] == self.in_dim
        b = features.shape[0]
        device = self._device
        base_dtype = next(self._local.parameters()).dtype
        infer_mode = labels_flat is None or (net_loss is None and global_loss is None and (local_loss is None))
        try:
            self.x_seen_elems += torch.tensor(features.numel(), device=self.x_seen_elems.device, dtype=self.x_seen_elems.dtype)
        except Exception:
            pass
        num_slices = (b + self.microbatch - 1) // self.microbatch
        token_chunks: List[torch.Tensor] = []
        context_chunks: List[torch.Tensor] = []
        if not infer_mode:
            self._local.train()
            self._global.train()
            for idx in range(num_slices):
                s = idx * self.microbatch
                e = min(b, (idx + 1) * self.microbatch)
                x_slice = features[s:e].to(device, dtype=base_dtype, non_blocking=True)
                with Autocast.float(device):
                    out: Meta = self._local(x_slice)
                token_chunks.append(out.tokens)
                context_chunks.append(out.context)
        else:
            self._local.eval()
            self._global.eval()
            for idx in range(num_slices):
                s = idx * self.microbatch
                e = min(b, (idx + 1) * self.microbatch)
                x_slice = features[s:e].to(device, dtype=base_dtype, non_blocking=True)
                with torch.no_grad(), Autocast.float(device):
                    out = self._local(x_slice)
                token_chunks.append(out.tokens if out.tokens.dtype == base_dtype else out.tokens.to(base_dtype))
                context_chunks.append(out.context if out.context.dtype == base_dtype else out.context.to(base_dtype))
        tokens = torch.cat(token_chunks, dim=0).to(device=device, dtype=base_dtype)
        context = torch.cat(context_chunks, dim=0).to(device=device, dtype=base_dtype)
        assembled = context.view(b, -1)
        if self.is_norm_linear and self.linear_branch is not None:
            bl = self.linear_branch(features.to(device, dtype=assembled.dtype))
            assembled = assembled + bl
        tokens_centered = tokens - tokens.mean(dim=1, keepdim=True)
        with (torch.no_grad() if infer_mode else torch.enable_grad()):
            with Autocast.float(device):
                refined_tokens = self._global(tokens_centered)
        residual_context = self._local.decode(refined_tokens, apply_norm=True)
        residual = residual_context.view(b, -1)
        y_hat_z = assembled + residual
        if residual.dtype != assembled.dtype:
            residual = residual.to(dtype=assembled.dtype)
            y_hat_z = assembled + residual
        is_cls_loss = isinstance(net_loss, (nn.CrossEntropyLoss, nn.NLLLoss)) if net_loss is not None else False
        y_hat_out = y_hat_z
        if self.has_valid_y_stats() and (not is_cls_loss):
            mu = self.y_mean.to(device=y_hat_z.device, dtype=y_hat_z.dtype)
            sd = self.y_std.to(device=y_hat_z.device, dtype=y_hat_z.dtype)
            eps = float(self.y_eps.item()) if hasattr(self, 'y_eps') else 1e-06
            sd = torch.clamp(sd, min=eps)
            y_hat_out = y_hat_z * sd + mu
        loss_val: Optional[torch.Tensor] = None
        if labels_flat is not None and (global_loss is not None or local_loss is not None):
            controller: Optional[LossWeightController] = None
            weights: Tuple[float, float]
            if loss_weights is None:
                weights = (1.0, 0.0)
            elif isinstance(loss_weights, (tuple, list)):
                seq = list(loss_weights)
                if len(seq) != 2:
                    raise ValueError("loss_weights requires two values")
                weights = (float(seq[0]), float(seq[1]))
            else:
                controller = cast(LossWeightController, loss_weights)
                weights = controller.weights()
            total = y_hat_out.new_tensor(0.0, dtype=y_hat_out.dtype)
            top_component: Optional[torch.Tensor] = None
            bottom_component: Optional[torch.Tensor] = None
            if self._loss_space == 'z' and self.has_valid_y_stats():
                mu_lbl = self.y_mean.to(device=y_hat_z.device, dtype=y_hat_z.dtype)
                sd_lbl = self.y_std.to(device=y_hat_z.device, dtype=y_hat_z.dtype)
                eps_lbl = float(self.y_eps.item()) if hasattr(self, 'y_eps') else 1e-06
                sd_lbl = torch.clamp(sd_lbl, min=eps_lbl)
                tgt_z = labels_flat.to(device=y_hat_z.device, dtype=y_hat_z.dtype)
                tgt_z = (tgt_z - mu_lbl) / sd_lbl
                y_top = y_hat_z
                y_bot = assembled
                if global_loss is not None:
                    top_component = global_loss(y_top, tgt_z)
                    total = total + weights[0] * top_component
                if local_loss is not None:
                    bottom_component = local_loss(y_bot, tgt_z)
                    total = total + weights[1] * bottom_component
            else:
                tgt_y = labels_flat.to(device=y_hat_out.device, dtype=y_hat_out.dtype)
                y_top = y_hat_out
                if self.has_valid_y_stats():
                    mu_lbl = self.y_mean.to(device=assembled.device, dtype=assembled.dtype)
                    sd_lbl = self.y_std.to(device=assembled.device, dtype=assembled.dtype)
                    eps_lbl = float(self.y_eps.item()) if hasattr(self, 'y_eps') else 1e-06
                    sd_lbl = torch.clamp(sd_lbl, min=eps_lbl)
                    y_bot = assembled * sd_lbl + mu_lbl
                else:
                    y_bot = assembled
                if global_loss is not None:
                    top_component = global_loss(y_top, tgt_y)
                    total = total + weights[0] * top_component
                if local_loss is not None:
                    bottom_component = local_loss(y_bot, tgt_y)
                    total = total + weights[1] * bottom_component
            if controller is not None:
                controller.update(top_component, bottom_component)
            loss_val = total
        elif net_loss is not None and labels_flat is not None:
            if is_cls_loss:
                tgt = labels_flat.to(device=y_hat_out.device).long()
                loss_val = net_loss(y_hat_out, tgt)
            elif self._loss_space == 'z' and self.has_valid_y_stats():
                mu_lbl = self.y_mean.to(device=y_hat_z.device, dtype=y_hat_z.dtype)
                sd_lbl = self.y_std.to(device=y_hat_z.device, dtype=y_hat_z.dtype)
                eps_lbl = float(self.y_eps.item()) if hasattr(self, 'y_eps') else 1e-06
                sd_lbl = torch.clamp(sd_lbl, min=eps_lbl)
                tgt = labels_flat.to(device=y_hat_z.device, dtype=y_hat_z.dtype)
                tgt = (tgt - mu_lbl) / sd_lbl
                loss_val = net_loss(y_hat_z, tgt)
            else:
                loss_val = net_loss(y_hat_out, labels_flat.to(device=y_hat_out.device, dtype=y_hat_out.dtype))
        return (y_hat_out.view(b, *self.out_shape), loss_val)

    def stats(self) -> dict:
        try:
            x_seen = float(self.x_seen_elems.item())
        except Exception:
            x_seen = 0.0
        y_cnt = self.y_count.to(torch.float64)
        y_seen = float(y_cnt.sum().item())
        denom = y_cnt.sum().clamp_min(1.0)
        y_avg = float((self.y_sum.sum() / denom).item())
        return {
            'accumulated_x': x_seen,
            'accumulated_y': y_seen,
            'y_min': self.y_min.clone(),
            'y_max': self.y_max.clone(),
            'y_mean': self.y_mean.clone(),
            'y_std': self.y_std.clone(),
            'y_avg': y_avg,
            'y_count': self.y_count.clone(),
        }

    @staticmethod
    def flatten_labels(
        labels: Sequence[torch.Tensor],
        *,
        dtype: Optional[torch.dtype] = None,
        pin_memory: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        out = torch.stack([l.reshape(-1) for l in labels], dim=0)
        if dtype is not None:
            out = out.to(dtype=dtype)
        if pin_memory and out.device.type == 'cpu':
            out = out.pin_memory()
        return (out.contiguous(), tuple(labels[0].shape))

    @staticmethod
    def unflatten_labels(flat: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
        return flat.view(flat.shape[0], *shape)

    @property
    def local(self) -> SpatioTemporalNet:
        return self._local
