# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import math
import warnings
from dataclasses import dataclass
from importlib import import_module
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    cast,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.nn.functional import scaled_dot_product_attention as _sdpa
    _HAS_SDPA = True
                                                          
except Exception:
    _HAS_SDPA = False

                                                                
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

@torch.no_grad()
def _norm_vector(coords: torch.Tensor, eps: float = 1e-6):
    B, N, C = coords.shape
    if C == 2:
        z = torch.zeros(B, N, 1, dtype=coords.dtype, device=coords.device)
        coords = torch.cat([coords, z], dim=-1)
    mins = coords.amin(dim=1, keepdim=True)
    maxs = coords.amax(dim=1, keepdim=True)
    rng = (maxs - mins).clamp_min(eps)
    x = (coords - mins) / rng
    return x.clamp_(0.0, 1.0 - 1e-7)

def _expand_bits(v: torch.Tensor):
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8))  & 0x0300F00F
    v = (v | (v << 4))  & 0x030C30C3
    v = (v | (v << 2))  & 0x09249249
    return v

@torch.no_grad()
def _to_z_index(coords01: torch.Tensor, bits: int = 10) -> torch.Tensor:
    B, N, _ = coords01.shape
    maxv = (1 << bits) - 1
    xyz = (coords01 * maxv).to(torch.int32)
    x, y, z = xyz.unbind(dim=-1)
    xx = _expand_bits(x)
    yy = _expand_bits(y) << 1
    zz = _expand_bits(z) << 2
    return (xx | yy | zz)

@torch.no_grad()
def _serialize_z_index(coords: torch.Tensor, *, bits: int, patch: int, shift_order: bool, block_index: int):
    """Return perm, invperm for PTv3 serialization (+ optional half-patch shift)."""
    coords01 = _norm_vector(coords)
    keys = _to_z_index(coords01, bits=bits)
    perm = keys.argsort(dim=-1, stable=True)                               
    if shift_order and (block_index % 2 == 1):
        B, N = perm.shape
        shift = (patch // 2) % max(N, 1)
        if shift:
            roll = torch.arange(N, device=perm.device).roll(shift)
            perm = perm.gather(1, roll.unsqueeze(0).expand(B, N))
    invperm = torch.empty_like(perm)
    scatter_src = torch.arange(perm.size(1), device=perm.device)
    if scatter_src.dim() < perm.dim():
        scatter_src = scatter_src.view(1, -1).expand_as(perm)
    invperm.scatter_(1, perm, scatter_src)
    return perm, invperm

def _block_ranges(N: int, patch: int):
    for s in range(0, N, patch):
        e = min(s + patch, N)
        yield s, e

                                
                           
                                
if _HAS_TRITON:
                                                                                                                                                           
    _RV_CONFIGS = [
        triton.Config({'BLOCK_J':  64, 'BLOCK_DH':  32}, num_warps=2,  num_stages=1),
        triton.Config({'BLOCK_J': 128, 'BLOCK_DH':  64}, num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_J':  32, 'BLOCK_DH':  32}, num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_J':  32, 'BLOCK_DH':  64}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_J':  64, 'BLOCK_DH':  32}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_J':  64, 'BLOCK_DH':  64}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_J': 128, 'BLOCK_DH':  32}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_J': 128, 'BLOCK_DH':  64}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_J': 128, 'BLOCK_DH': 128}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_J': 256, 'BLOCK_DH':  64}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_J': 256, 'BLOCK_DH': 128}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_J': 512, 'BLOCK_DH':  64}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_J': 512, 'BLOCK_DH': 128}, num_warps=16, num_stages=3),
    ]

                                          
    @triton.autotune(configs=_RV_CONFIGS, key=['J', 'DH', 'K'])
    @triton.jit
    def _reduce_weighted_sum(
        W, RV, O,
        B: tl.constexpr, H: tl.constexpr, K: tl.constexpr, J: tl.constexpr, DH: tl.constexpr,
                                                               
        SWB, SWH, SWK, SWJ,
        SRVB, SRVH, SRVK, SRVJ, SRVDH,
        SOB, SOH, SOK, SODH,
        BLOCK_J: tl.constexpr, BLOCK_DH: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)                 
        pid_k  = tl.program_id(1)               
        pid_d  = tl.program_id(2)                       

        b = pid_bh // H
        h = pid_bh %  H
        k = pid_k

        dh_off = pid_d * BLOCK_DH + tl.arange(0, BLOCK_DH)
        acc = tl.zeros([BLOCK_DH], dtype=tl.float32)

        j0 = 0
        while j0 < J:
            j_off = j0 + tl.arange(0, BLOCK_J)
            mask_j = j_off < J
                                          
            w_ptr = W + b*SWB + h*SWH + k*SWK + j_off*SWJ
            w_val = tl.load(w_ptr, mask=mask_j, other=0.0).to(tl.float32)        
                                                        
            rv_ptr = RV + b*SRVB + h*SRVH + k*SRVK + j_off[:, None]*SRVJ + dh_off[None, :]*SRVDH
            mask_rv = (mask_j[:, None]) & (dh_off[None, :] < DH)
            rv_val = tl.load(rv_ptr, mask=mask_rv, other=0.0).to(tl.float32)
                                
            acc += tl.sum(rv_val * w_val[:, None], axis=0)
            j0 += BLOCK_J

                                                                                            
        o_ptr = O + b*SOB + h*SOH + k*SOK + dh_off*SODH
        mask_o = dh_off < DH
        old = tl.load(o_ptr, mask=mask_o, other=0.0)
        tl.store(o_ptr, old + acc, mask=mask_o)
from tensordict import TensorDict, TensorDictBase

try:
    import numpy as _np
except Exception:
    _np = None

try:
    import scipy.stats as _sps
except Exception:
    _sps = None
try:
    from torch.utils.checkpoint import checkpoint as activation_checkpoint
except Exception:
    activation_checkpoint = None

from ..api import is_meta_or_fake_tensor
from ..backend.compat import patch_torch
from ..backend.distributed import get_world_size
from ..backend.system import cpu_info, system_info, posix_time
from ..functional.fx import Autocast, Gradient, reshape_for_mha
from .activations import SwiGLU
from .kernels import (
    DotProductAttention,
    MultiHeadAttention,
    MultiScaleRetention,
    to_additive_mask,
)

try:
    from torch._inductor import config as _inductor_config
except Exception:
    _inductor_config = None
else:
    try:
        triton_cfg = getattr(_inductor_config, "triton")
        if hasattr(triton_cfg, "cudagraph_skip_dynamic_graphs"):
            triton_cfg.cudagraph_skip_dynamic_graphs = True
    except Exception:
        pass

try:
    from ..backend.compat import RMSNorm as _Norm
except Exception:
    _Norm = nn.LayerNorm


patch_torch()


if TYPE_CHECKING:
    import numpy as np

    from ..api.config import ModelConfig


LayerNorm = nn.LayerNorm


_NORM_MODES = {"norm", "normalize", "normalization"}
_DENORM_MODES = {"denorm", "denormalize", "denormalization"}


_LEN_OS, _LEN_KERNEL, _LEN_ARCH, _LEN_ACCEL, _LEN_TZ = 96, 96, 32, 256, 32
_LEN_CPU = 2048


def _get_device_from(module: nn.Module) -> str:
    try:
        dev = next((p.device for p in module.parameters() if p is not None), None)
        if dev is None:
            dev = next(
                (b.device for _, b in module.named_buffers()), torch.device("cpu")
            )
        return str(getattr(dev, "type", "cpu"))
    except Exception:
        return "cpu"


def _fixed_bytes(s: str, L: int) -> torch.Tensor:
    b = s.encode("utf-8", errors="ignore")[:L]
    out = torch.zeros(L, dtype=torch.uint8)
    if b:
        out[: len(b)] = torch.as_tensor(list(b), dtype=torch.uint8)
    return out


@torch.jit.ignore
def _as_utf8(row: torch.Tensor) -> str:
    if not isinstance(row, torch.Tensor) or row.dtype != torch.uint8 or row.dim() != 1:
        raise TypeError("expected 1D torch.uint8 tensor")
    data = bytes(row.detach().cpu().tolist())
    data = data.split(b"\x00", 1)[0]
    return data.decode("utf-8", errors="ignore")


@torch.jit.ignore
def _as_utf8_list(col: torch.Tensor) -> list[str]:
    if not isinstance(col, torch.Tensor) or col.dtype != torch.uint8 or col.dim() != 2:
        raise TypeError("expected [T, L] torch.uint8 tensor")
    return [_as_utf8(col[i]) for i in range(col.shape[0])]


def _get_sys_info(tz_name: Optional[str]) -> Tuple[str, str, str, str, str, str]:
    os_name, kernel, arch, accel = system_info()
    tzlabel = tz_name or "GMT"
    cpu_label = cpu_info(max_bytes=_LEN_CPU)
    return os_name, kernel, arch, accel, tzlabel, cpu_label


@torch.jit.ignore
def _accumulate_moments(
    x: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    x64 = x.to(dtype=torch.float64)
    n = int(x64.shape[0])
    if n == 0:
        d = x64.shape[-1]
        z = x64.new_zeros(d)
        return (0, z, z, z)
    m = x64.mean(dim=0)
    xc = x64 - m
    M2 = (xc * xc).sum(dim=0)
    M3 = (xc * xc * xc).sum(dim=0)
    return (n, m, M2, M3)


@torch.jit.ignore
def _reduce_moments(
    n1: int,
    m1: torch.Tensor,
    M2_1: torch.Tensor,
    M3_1: torch.Tensor,
    n2: int,
    m2: torch.Tensor,
    M2_2: torch.Tensor,
    M3_2: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    if n1 == 0:
        return (n2, m2, M2_2, M3_2)
    if n2 == 0:
        return (n1, m1, M2_1, M3_1)
    n = n1 + n2
    delta = m2 - m1
    n1f = float(n1)
    n2f = float(n2)
    nf = float(n)
    m = m1 + delta * (n2f / nf)
    M2 = M2_1 + M2_2 + (delta * delta) * (n1f * n2f / nf)
    M3 = (
        M3_1
        + M3_2
        + (delta * delta * delta) * (n1f * n2f * (n1f - n2f) / (nf * nf))
        + 3.0 * delta * (n1f * M2_2 - n2f * M2_1) / nf
    )
    return (int(n), m, M2, M3)


def _sample_skewness(
    N: int, M2: torch.Tensor, M3: torch.Tensor, eps: float
) -> torch.Tensor:
    M2s = M2.clamp_min(eps)
    return (math.sqrt(max(N, 1)) * M3) / (M2s.sqrt() * M2s)


def _skew_normal_delta(gamma1: torch.Tensor) -> torch.Tensor:
    c = ((4.0 - math.pi) / 2.0) ** (2.0 / 3.0)
    a = gamma1.abs().clamp_max(0.9952717464).pow(2.0 / 3.0)
    num = (math.pi / 2.0) * a
    den = (a + c).clamp_min(1e-12)
    delta = (num / den).sqrt()
    return delta.copysign(gamma1)


def _skew_normal_vars(
    mu: torch.Tensor, var: torch.Tensor, gamma1: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta = _skew_normal_delta(gamma1).clamp(-0.999999, 0.999999)
    one = var.new_tensor(1.0)
    omega = (var / (one - 2.0 * delta * delta / math.pi).clamp_min(1e-12)).sqrt()
    xi = mu - omega * delta * math.sqrt(2.0 / math.pi)
    alpha = delta / (one - delta * delta).clamp_min(1e-12).sqrt()
    return xi, omega, alpha


class Affine(nn.Module):
    def __init__(
        self, n_features: int, init_weight: float = 1.0, init_bias: float = 0.0
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((int(n_features),), float(init_weight)))
        self.bias = nn.Parameter(torch.full((int(n_features),), float(init_bias)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight.view(1, -1) + self.bias.view(1, -1)


class PowerTransform(nn.Module):
    def __init__(
        self,
        n_features: int,
        method: str = "yeojohnson",
        mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.method = str(method or "yeojohnson").lower()
        if self.method not in ("yeojohnson", "boxcox"):
            raise ValueError("method must be 'yeojohnson' or 'boxcox'")
        self.eps = float(eps)
        m = (
            torch.ones(self.n_features, dtype=torch.bool)
            if mask is None
            else torch.as_tensor(mask, dtype=torch.bool)
        )
        self.register_buffer("mask", m, persistent=True)
        self.register_buffer(
            "lmbda", torch.zeros(self.n_features, dtype=torch.float64), persistent=True
        )
        self.register_buffer(
            "shift", torch.zeros(self.n_features, dtype=torch.float64), persistent=True
        )

    def _bc(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return v.view(1, -1).to(dtype=x.dtype, device=x.device).expand_as(x)

    def _yj(self, x: torch.Tensor) -> torch.Tensor:
        lam = self._bc(x, self.lmbda.to(x.dtype))
        pos = x >= 0
        out = torch.empty_like(x)
        if pos.any():
            xp, lp = x[pos], lam[pos]
            out[pos] = torch.where(
                torch.isclose(lp, torch.zeros_like(lp)),
                torch.log1p(xp),
                ((xp + 1.0).clamp_min(self.eps).pow(lp) - 1.0) / lp,
            )
        if (~pos).any():
            xn, ln = x[~pos], lam[~pos]
            two = x.new_tensor(2.0)
            out[~pos] = torch.where(
                torch.isclose(ln, two),
                -torch.log1p((-xn).clamp_min(self.eps)),
                -(((1.0 - xn).clamp_min(self.eps)).pow(two - ln) - 1.0) / (two - ln),
            )
        return out

    def _inv_yj(self, y: torch.Tensor) -> torch.Tensor:
        lam = self._bc(y, self.lmbda.to(y.dtype))
        out = torch.empty_like(y)
        pos = y >= 0
        if pos.any():
            yp, lp = y[pos], lam[pos]
            out[pos] = torch.where(
                torch.isclose(lp, torch.zeros_like(lp)),
                torch.expm1(yp),
                (lp * yp + 1.0).clamp_min(self.eps).pow(1.0 / lp) - 1.0,
            )
        if (~pos).any():
            yn, ln = y[~pos], lam[~pos]
            two = y.new_tensor(2.0)
            out[~pos] = torch.where(
                torch.isclose(ln, two),
                1.0 - torch.exp(-yn),
                1.0 - (1.0 - (two - ln) * yn).clamp_min(self.eps).pow(1.0 / (two - ln)),
            )
        return out

    def _bcx(self, x: torch.Tensor) -> torch.Tensor:
        z = x + self._bc(x, self.shift.to(x.dtype))
        lam = self._bc(x, self.lmbda.to(x.dtype))
        z = z.clamp_min(self.eps)
        return torch.where(
            torch.isclose(lam, torch.zeros_like(lam)),
            torch.log(z),
            (z.pow(lam) - 1.0) / lam,
        )

    def _inv_bcx(self, y: torch.Tensor) -> torch.Tensor:
        lam = self._bc(y, self.lmbda.to(y.dtype))
        z = torch.where(
            torch.isclose(lam, torch.zeros_like(lam)),
            torch.exp(y),
            (lam * y + 1.0).clamp_min(self.eps).pow(1.0 / lam),
        )
        return z - self._bc(y, self.shift.to(y.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.mask.any()):
            return x
        m = self._bc(x, self.mask)
        y = self._yj(x) if self.method.startswith("yeo") else self._bcx(x)
        return torch.where(m, y, x)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if not bool(self.mask.any()):
            return y
        m = self._bc(y, self.mask)
        x = self._inv_yj(y) if self.method.startswith("yeo") else self._inv_bcx(y)
        return torch.where(m, x, y)

    @torch.no_grad()
    def set_params(
        self,
        lmbda: torch.Tensor,
        shift: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        lam = torch.as_tensor(lmbda, dtype=torch.float64, device=self.lmbda.device)
        if lam.numel() == 1:
            lam = lam.expand_as(self.lmbda)
        self.lmbda.copy_(lam)
        if shift is not None:
            sh = torch.as_tensor(shift, dtype=torch.float64, device=self.shift.device)
            if sh.numel() == 1:
                sh = sh.expand_as(self.shift)
            self.shift.copy_(sh)
        if mask is not None:
            self.mask.copy_(
                torch.as_tensor(mask, dtype=torch.bool, device=self.mask.device)
            )

    @torch.no_grad()
    def fit_params_numpy(
        self,
        X: "np.ndarray",
        boxcox_shift: str = "auto",
    ) -> Tuple["np.ndarray", "np.ndarray"]:

        if _np is None:
            raise RuntimeError("NumPy is required; install it with 'pip install numpy'.")
        if _sps is None:
            raise RuntimeError("SciPy is required; install it with 'pip install scipy'.")
        X = _np.asarray(X, dtype=_np.float64)
        D = X.shape[-1]
        lam = _np.zeros(D, dtype=_np.float64)
        shf = _np.zeros(D, dtype=_np.float64)
        for j in range(D):
            xj = X[:, j]
            if not self.mask[j].item():
                continue
            if self.method.startswith("yeo"):
                _, lj = _sps.yeojohnson(xj, lmbda=None)
                lam[j] = lj
            else:
                s = 0.0
                if boxcox_shift == "auto":
                    mn = _np.nanmin(xj)
                    if mn <= 0:
                        s = -mn + 1e-6
                lam[j] = _sps.boxcox_normmax(xj + s)
                shf[j] = s
        self.set_params(torch.from_numpy(lam), torch.from_numpy(shf))
        return lam, shf


class Normal(nn.Module):
    def __init__(
        self,
        n_features: int,
        *,
        standardize: bool = True,
        skew: bool = True,
        mode: str = "norm",
        momentum: float = 0.1,
        eps: float = 1e-5,
        power: Optional[str] = None,
        power_mask: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        mode_l = str(mode or "").lower()
        if mode_l not in _NORM_MODES and mode_l not in _DENORM_MODES:
            raise ValueError(f"mode must be one of {_NORM_MODES | _DENORM_MODES}")
        self.mode = mode_l
        self.standardize = bool(standardize)
        self.skew = bool(skew)
        self.eps = float(eps)
        self.bn = nn.BatchNorm1d(
            self.n_features,
            affine=False,
            momentum=float(momentum),
            eps=float(eps),
            track_running_stats=True,
        )
        self.power = None if power is None else str(power).lower()
        self.pt: Optional[PowerTransform] = None
        if self.skew and self.power in ("yeojohnson", "boxcox"):
            self.pt = PowerTransform(
                self.n_features,
                method=self.power,
                mask=power_mask,
                eps=self.eps,
            )

        self.register_buffer(
            "stg_count", torch.tensor(0, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "stg_mean",
            torch.zeros(self.n_features, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "stg_M2",
            torch.zeros(self.n_features, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "stg_M3",
            torch.zeros(self.n_features, dtype=torch.float64),
            persistent=True,
        )

        self.register_buffer(
            "gs_count", torch.tensor(0, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "gs_mean",
            torch.zeros(self.n_features, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "gs_M2",
            torch.zeros(self.n_features, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "gs_M3",
            torch.zeros(self.n_features, dtype=torch.float64),
            persistent=True,
        )

        self.register_buffer(
            "last_train_start_ns", torch.tensor(0, dtype=torch.int64), persistent=True
        )
        self.register_buffer(
            "last_train_end_ns", torch.tensor(0, dtype=torch.int64), persistent=True
        )
        self.register_buffer(
            "train_runs", torch.tensor(0, dtype=torch.long), persistent=True
        )

        self._hist_maxlen = 2000
        self.register_buffer(
            "hist_step", torch.empty(0, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "hist_seen", torch.empty(0, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "hist_start_ns", torch.empty(0, dtype=torch.int64), persistent=True
        )
        self.register_buffer(
            "hist_end_ns", torch.empty(0, dtype=torch.int64), persistent=True
        )
        self.register_buffer(
            "hist_device_code",
            torch.empty(0, dtype=torch.int8),
            persistent=True,
        )
        self.register_buffer(
            "hist_world_size", torch.empty(0, dtype=torch.int32), persistent=True
        )
        self.register_buffer(
            "hist_os", torch.empty(0, _LEN_OS, dtype=torch.uint8), persistent=True
        )
        self.register_buffer(
            "hist_kernel",
            torch.empty(0, _LEN_KERNEL, dtype=torch.uint8),
            persistent=True,
        )
        self.register_buffer(
            "hist_arch", torch.empty(0, _LEN_ARCH, dtype=torch.uint8), persistent=True
        )
        self.register_buffer(
            "hist_accel", torch.empty(0, _LEN_ACCEL, dtype=torch.uint8), persistent=True
        )
        self.register_buffer(
            "hist_tz", torch.empty(0, _LEN_TZ, dtype=torch.uint8), persistent=True
        )
        self.register_buffer(
            "hist_cpu", torch.empty(0, _LEN_CPU, dtype=torch.uint8), persistent=True
        )

        self.register_buffer(
            "t_df",
            torch.full((self.n_features,), float("nan"), dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "t_loc",
            torch.full((self.n_features,), float("nan"), dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "t_scale",
            torch.full((self.n_features,), float("nan"), dtype=torch.float64),
            persistent=True,
        )

    @staticmethod
    def _encode_device(t: str) -> int:
        m = {"cpu": 0, "cuda": 1, "mps": 2, "xpu": 3, "vulkan": 4}
        return int(m.get(str(t), 0))

    @torch.no_grad()
    def _accumulate_batch(self, x: torch.Tensor) -> None:
        n2, m2, M2_2, M3_2 = _accumulate_moments(x)
        n1 = int(self.stg_count.item())
        n, m, M2, M3 = _reduce_moments(
            n1,
            self.stg_mean,
            self.stg_M2,
            self.stg_M3,
            n2,
            m2,
            M2_2,
            M3_2,
        )
        self.stg_count.fill_(n)
        self.stg_mean.copy_(m)
        self.stg_M2.copy_(M2)
        self.stg_M3.copy_(M3)

    @torch.no_grad()
    def _commit_moments(self) -> None:
        n2 = int(self.stg_count.item())
        if n2 <= 0:
            return
        n1 = int(self.gs_count.item())
        n, m, M2, M3 = _reduce_moments(
            n1,
            self.gs_mean,
            self.gs_M2,
            self.gs_M3,
            n2,
            self.stg_mean,
            self.stg_M2,
            self.stg_M3,
        )
        self.gs_count.fill_(n)
        self.gs_mean.copy_(m)
        self.gs_M2.copy_(M2)
        self.gs_M3.copy_(M3)

        self.stg_count.zero_()
        self.stg_mean.zero_()
        self.stg_M2.zero_()
        self.stg_M3.zero_()

    @torch.no_grad()
    def export_history(self) -> TensorDict:
        T = int(self.hist_step.numel())
        return TensorDict(
            {
                "step": self.hist_step.detach().clone(),
                "seen": self.hist_seen.detach().clone(),
                "t_start_ns": self.hist_start_ns.detach().clone(),
                "t_end_ns": self.hist_end_ns.detach().clone(),
                "device_code": self.hist_device_code.detach().clone(),
                "world_size": self.hist_world_size.detach().clone(),
                "os": self.hist_os.detach().clone(),
                "kernel": self.hist_kernel.detach().clone(),
                "arch": self.hist_arch.detach().clone(),
                "accel": self.hist_accel.detach().clone(),
                "tz": self.hist_tz.detach().clone(),
                "cpu": self.hist_cpu.detach().clone(),
            },
            batch_size=[T],
        )

    @torch.jit.ignore
    @torch.no_grad()
    def export_history_text(self) -> list[dict[str, object]]:
        T = int(self.hist_step.numel())
        dev_map = {0: "cpu", 1: "cuda", 2: "mps", 3: "xpu", 4: "vulkan"}
        out: list[dict[str, object]] = []
        for i in range(T):
            out.append(
                {
                    "step": int(self.hist_step[i].item()),
                    "seen": int(self.hist_seen[i].item()),
                    "t_start_ns": int(self.hist_start_ns[i].item()),
                    "t_end_ns": int(self.hist_end_ns[i].item()),
                    "device": dev_map.get(
                        int(self.hist_device_code[i].item()), "unknown"
                    ),
                    "world_size": int(self.hist_world_size[i].item()),
                    "os": _as_utf8(self.hist_os[i]),
                    "kernel": _as_utf8(self.hist_kernel[i]),
                    "arch": _as_utf8(self.hist_arch[i]),
                    "accelerators": _as_utf8(self.hist_accel[i]),
                    "tz": _as_utf8(self.hist_tz[i]),
                    "cpu": _as_utf8(self.hist_cpu[i]),
                }
            )
        return out

    @torch.no_grad()
    def set_power_params(
        self,
        lmbda: torch.Tensor,
        shift: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        if self.pt is None:
            raise RuntimeError("Power transform is not enabled (power=None).")
        self.pt.set_params(lmbda, shift, mask)

    @torch.no_grad()
    def fit_power_params_numpy(
        self, X: "np.ndarray", boxcox_shift: str = "auto"
    ) -> Tuple["np.ndarray", "np.ndarray"]:
        if self.pt is None:
            raise RuntimeError("Power transform is not enabled (power=None).")
        return self.pt.fit_params_numpy(X, boxcox_shift=boxcox_shift)

    @torch.no_grad()
    def commit_training_success(
        self,
        start_ns: int | None = None,
        end_ns: int | None = None,
        tz_name: str | None = "GMT",
    ) -> None:

        self._commit_moments()

        s = int(start_ns if start_ns is not None else posix_time(tz_name))
        e = int(end_ns if end_ns is not None else posix_time(tz_name))
        self.last_train_start_ns.fill_(s)
        self.last_train_end_ns.fill_(e)
        runs = int(self.train_runs.item()) + 1
        self.train_runs.fill_(runs)

        dev_code = torch.tensor(
            [self._encode_device(_get_device_from(self))],
            dtype=torch.int8,
            device=self.hist_device_code.device,
        )
        ws = torch.tensor(
            [get_world_size()],
            dtype=torch.int32,
            device=self.hist_world_size.device,
        )
        step = torch.tensor([runs - 1], dtype=torch.long, device=self.hist_step.device)
        seen = torch.tensor(
            [int(self.gs_count.item())], dtype=torch.long, device=self.hist_seen.device
        )
        self.hist_step = torch.cat([self.hist_step, step], dim=0)[-self._hist_maxlen :]
        self.hist_seen = torch.cat([self.hist_seen, seen], dim=0)[-self._hist_maxlen :]
        self.hist_start_ns = torch.cat(
            [
                self.hist_start_ns,
                torch.tensor([s], dtype=torch.int64, device=self.hist_start_ns.device),
            ],
            dim=0,
        )[-self._hist_maxlen :]
        self.hist_end_ns = torch.cat(
            [
                self.hist_end_ns,
                torch.tensor([e], dtype=torch.int64, device=self.hist_end_ns.device),
            ],
            dim=0,
        )[-self._hist_maxlen :]
        self.hist_device_code = torch.cat([self.hist_device_code, dev_code], dim=0)[
            -self._hist_maxlen :
        ]
        self.hist_world_size = torch.cat([self.hist_world_size, ws], dim=0)[
            -self._hist_maxlen :
        ]
        os_name, kernel, arch, accel, tzlabel, cpu_label = _get_sys_info(tz_name)
        self.hist_os = torch.cat(
            [
                self.hist_os,
                _fixed_bytes(os_name, _LEN_OS)
                .unsqueeze(0)
                .to(device=self.hist_os.device),
            ],
            dim=0,
        )[-self._hist_maxlen :]
        self.hist_kernel = torch.cat(
            [
                self.hist_kernel,
                _fixed_bytes(kernel, _LEN_KERNEL)
                .unsqueeze(0)
                .to(device=self.hist_kernel.device),
            ],
            dim=0,
        )[-self._hist_maxlen :]
        self.hist_arch = torch.cat(
            [
                self.hist_arch,
                _fixed_bytes(arch, _LEN_ARCH)
                .unsqueeze(0)
                .to(device=self.hist_arch.device),
            ],
            dim=0,
        )[-self._hist_maxlen :]
        self.hist_accel = torch.cat(
            [
                self.hist_accel,
                _fixed_bytes(accel, _LEN_ACCEL)
                .unsqueeze(0)
                .to(device=self.hist_accel.device),
            ],
            dim=0,
        )[-self._hist_maxlen :]
        self.hist_tz = torch.cat(
            [
                self.hist_tz,
                _fixed_bytes(tzlabel, _LEN_TZ)
                .unsqueeze(0)
                .to(device=self.hist_tz.device),
            ],
            dim=0,
        )[-self._hist_maxlen :]
        self.hist_cpu = torch.cat(
            [
                self.hist_cpu,
                _fixed_bytes(cpu_label, _LEN_CPU)
                .unsqueeze(0)
                .to(device=self.hist_cpu.device),
            ],
            dim=0,
        )[-self._hist_maxlen :]

        N = int(self.gs_count.item())
        if N > 0:
            mu = self.gs_mean.to(dtype=torch.float64)
            var = (self.gs_M2 / max(N, 1)).clamp_min(1e-12)
            if self.skew:
                gamma1 = _sample_skewness(N, self.gs_M2, self.gs_M3, self.eps)
                xi, omega, _alpha = _skew_normal_vars(mu, var, gamma1)
                rm = xi.to(dtype=self.bn.running_mean.dtype)
                rv = (omega * omega).to(dtype=self.bn.running_var.dtype)
            else:
                rm = mu.to(dtype=self.bn.running_mean.dtype)
                rv = var.to(dtype=self.bn.running_var.dtype)
            self.bn.running_mean.copy_(rm)
            self.bn.running_var.copy_(rv)

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        hist_names = (
            "hist_step",
            "hist_seen",
            "hist_start_ns",
            "hist_end_ns",
            "hist_device_code",
            "hist_world_size",
            "hist_os",
            "hist_kernel",
            "hist_arch",
            "hist_accel",
            "hist_tz",
            "hist_cpu",
        )
        hist_state: dict[str, torch.Tensor] = {}
        for name in hist_names:
            key = prefix + name
            value = state_dict.pop(key, None)
            if isinstance(value, torch.Tensor):
                hist_state[name] = value.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for key in list(hist_state):
            full = prefix + key
            if full in missing_keys:
                missing_keys.remove(full)
        for name, tensor in hist_state.items():
            current = getattr(self, name, None)
            if isinstance(current, torch.Tensor):
                updated = tensor.to(device=current.device, dtype=current.dtype)
                self._buffers[name] = updated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.standardize:
            return x
        bn_dtype = (
            self.bn.running_mean.dtype
            if isinstance(self.bn.running_mean, torch.Tensor)
            else x.dtype
        )
        orig_dtype = x.dtype
        if self.mode in _NORM_MODES:
            if self.training:
                z = self.pt(x) if self.pt is not None else x
                self._accumulate_batch(z)
                if z.dtype != bn_dtype:
                    z = z.to(dtype=bn_dtype)
                out = self.bn(z)
                if out.dtype != orig_dtype:
                    out = out.to(dtype=orig_dtype)
                return out
            else:
                z = self.pt(x) if self.pt is not None else x
                if z.dtype != bn_dtype:
                    z = z.to(dtype=bn_dtype)
                out = self.bn(z)
                if out.dtype != orig_dtype:
                    out = out.to(dtype=orig_dtype)
                return out
        elif self.mode in _DENORM_MODES:
            x_cast = x if x.dtype == bn_dtype else x.to(dtype=bn_dtype)
            std = (self.bn.running_var + self.bn.eps).sqrt().view(1, -1)
            mean = self.bn.running_mean.view(1, -1)
            y = x_cast * std + mean
            y = self.pt.inverse(y) if self.pt is not None else y
            if y.dtype != orig_dtype:
                y = y.to(dtype=orig_dtype)
            return y
        else:
            raise ValueError(f"invalid mode: {self.mode}")


class StudentsT(nn.Module):
    def __init__(
        self,
        n_features: int,
        *,
        standardize: bool = True,
        mode: str = "norm",
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        mode_l = str(mode or "").lower()
        if mode_l not in _NORM_MODES and mode_l not in _DENORM_MODES:
            raise ValueError(f"mode must be one of {_NORM_MODES | _DENORM_MODES}")
        self.mode = mode_l
        self.standardize = bool(standardize)
        self.eps = float(eps)
        self.bn = nn.BatchNorm1d(
            self.n_features,
            affine=False,
            momentum=float(momentum),
            eps=float(eps),
            track_running_stats=True,
        )

        self.register_buffer(
            "stg_count", torch.tensor(0, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "stg_mean",
            torch.zeros(self.n_features, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "stg_M2", torch.zeros(self.n_features, dtype=torch.float64), persistent=True
        )
        self.register_buffer(
            "stg_M3", torch.zeros(self.n_features, dtype=torch.float64), persistent=True
        )
        self.register_buffer(
            "gs_count", torch.tensor(0, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "gs_mean",
            torch.zeros(self.n_features, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "gs_M2", torch.zeros(self.n_features, dtype=torch.float64), persistent=True
        )
        self.register_buffer(
            "gs_M3", torch.zeros(self.n_features, dtype=torch.float64), persistent=True
        )
        self.register_buffer(
            "last_train_start_ns", torch.tensor(0, dtype=torch.int64), persistent=True
        )
        self.register_buffer(
            "last_train_end_ns", torch.tensor(0, dtype=torch.int64), persistent=True
        )
        self.register_buffer(
            "train_runs", torch.tensor(0, dtype=torch.long), persistent=True
        )
        self._hist_maxlen = 2000
        self.register_buffer(
            "hist_step", torch.empty(0, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "hist_seen", torch.empty(0, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "hist_start_ns", torch.empty(0, dtype=torch.int64), persistent=True
        )
        self.register_buffer(
            "hist_end_ns", torch.empty(0, dtype=torch.int64), persistent=True
        )
        self.register_buffer(
            "hist_device_code", torch.empty(0, dtype=torch.int8), persistent=True
        )
        self.register_buffer(
            "hist_world_size", torch.empty(0, dtype=torch.int32), persistent=True
        )
        self.register_buffer(
            "hist_os", torch.empty(0, _LEN_OS, dtype=torch.uint8), persistent=True
        )
        self.register_buffer(
            "hist_kernel",
            torch.empty(0, _LEN_KERNEL, dtype=torch.uint8),
            persistent=True,
        )
        self.register_buffer(
            "hist_arch", torch.empty(0, _LEN_ARCH, dtype=torch.uint8), persistent=True
        )
        self.register_buffer(
            "hist_accel", torch.empty(0, _LEN_ACCEL, dtype=torch.uint8), persistent=True
        )
        self.register_buffer(
            "hist_tz", torch.empty(0, _LEN_TZ, dtype=torch.uint8), persistent=True
        )
        self.register_buffer(
            "hist_cpu", torch.empty(0, _LEN_CPU, dtype=torch.uint8), persistent=True
        )

    @torch.no_grad()
    def _accumulate_batch(self, x: torch.Tensor) -> None:
        n2, m2, M2_2, M3_2 = _accumulate_moments(x)
        n1 = int(self.stg_count.item())
        n, m, M2, M3 = _reduce_moments(
            n1,
            self.stg_mean,
            self.stg_M2,
            self.stg_M3,
            n2,
            m2,
            M2_2,
            M3_2,
        )
        self.stg_count.fill_(n)
        self.stg_mean.copy_(m)
        self.stg_M2.copy_(M2)
        self.stg_M3.copy_(M3)

    @torch.no_grad()
    def _commit_moments(self) -> None:
        n2 = int(self.stg_count.item())
        if n2 <= 0:
            return
        n1 = int(self.gs_count.item())
        n, m, M2, M3 = _reduce_moments(
            n1,
            self.gs_mean,
            self.gs_M2,
            self.gs_M3,
            n2,
            self.stg_mean,
            self.stg_M2,
            self.stg_M3,
        )
        self.gs_count.fill_(n)
        self.gs_mean.copy_(m)
        self.gs_M2.copy_(M2)
        self.gs_M3.copy_(M3)
        self.stg_count.zero_()
        self.stg_mean.zero_()
        self.stg_M2.zero_()
        self.stg_M3.zero_()

    @torch.no_grad()
    def commit_training_success(
        self,
        start_ns: int | None = None,
        end_ns: int | None = None,
        tz_name: str | None = "GMT",
    ) -> None:
        self._commit_moments()
        s = int(start_ns if start_ns is not None else posix_time(tz_name))
        e = int(end_ns if end_ns is not None else posix_time(tz_name))
        self.last_train_start_ns.fill_(s)
        self.last_train_end_ns.fill_(e)
        runs = int(self.train_runs.item()) + 1
        self.train_runs.fill_(runs)
        dev_code = torch.tensor(
            [Normal._encode_device(_get_device_from(self))],
            dtype=torch.int8,
            device=self.hist_device_code.device,
        )
        ws = torch.tensor(
            [get_world_size()], dtype=torch.int32, device=self.hist_world_size.device
        )
        step = torch.tensor([runs - 1], dtype=torch.long, device=self.hist_step.device)
        seen = torch.tensor(
            [int(self.gs_count.item())], dtype=torch.long, device=self.hist_seen.device
        )
        self.hist_step = torch.cat([self.hist_step, step], dim=0)[-self._hist_maxlen :]
        self.hist_seen = torch.cat([self.hist_seen, seen], dim=0)[-self._hist_maxlen :]
        self.hist_start_ns = torch.cat(
            [
                self.hist_start_ns,
                torch.tensor([s], dtype=torch.int64, device=self.hist_start_ns.device),
            ],
            dim=0,
        )[-self._hist_maxlen :]
        self.hist_end_ns = torch.cat(
            [
                self.hist_end_ns,
                torch.tensor([e], dtype=torch.int64, device=self.hist_end_ns.device),
            ],
            dim=0,
        )[-self._hist_maxlen :]
        self.hist_device_code = torch.cat([self.hist_device_code, dev_code], dim=0)[
            -self._hist_maxlen :
        ]
        self.hist_world_size = torch.cat([self.hist_world_size, ws], dim=0)[
            -self._hist_maxlen :
        ]
        os_name, kernel, arch, accel, tzlabel, cpu_label = _get_sys_info(tz_name)
        self.hist_os = torch.cat(
            [
                self.hist_os,
                _fixed_bytes(os_name, _LEN_OS)
                .unsqueeze(0)
                .to(device=self.hist_os.device),
            ],
            dim=0,
        )[-self._hist_maxlen :]
        self.hist_kernel = torch.cat(
            [
                self.hist_kernel,
                _fixed_bytes(kernel, _LEN_KERNEL)
                .unsqueeze(0)
                .to(device=self.hist_kernel.device),
            ],
            dim=0,
        )[-self._hist_maxlen :]
        self.hist_arch = torch.cat(
            [
                self.hist_arch,
                _fixed_bytes(arch, _LEN_ARCH)
                .unsqueeze(0)
                .to(device=self.hist_arch.device),
            ],
            dim=0,
        )[-self._hist_maxlen :]
        self.hist_accel = torch.cat(
            [
                self.hist_accel,
                _fixed_bytes(accel, _LEN_ACCEL)
                .unsqueeze(0)
                .to(device=self.hist_accel.device),
            ],
            dim=0,
        )[-self._hist_maxlen :]
        self.hist_tz = torch.cat(
            [
                self.hist_tz,
                _fixed_bytes(tzlabel, _LEN_TZ)
                .unsqueeze(0)
                .to(device=self.hist_tz.device),
            ],
            dim=0,
        )[-self._hist_maxlen :]
        self.hist_cpu = torch.cat(
            [
                self.hist_cpu,
                _fixed_bytes(cpu_label, _LEN_CPU)
                .unsqueeze(0)
                .to(device=self.hist_cpu.device),
            ],
            dim=0,
        )[-self._hist_maxlen :]
        N = int(self.gs_count.item())
        if N > 0:
            mu = self.gs_mean.to(dtype=self.bn.running_mean.dtype)
            var = (
                (self.gs_M2 / max(N, 1))
                .clamp_min(1e-12)
                .to(dtype=self.bn.running_var.dtype)
            )
            self.bn.running_mean.copy_(mu)
            self.bn.running_var.copy_(var)

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        hist_names = (
            "hist_step",
            "hist_seen",
            "hist_start_ns",
            "hist_end_ns",
            "hist_device_code",
            "hist_world_size",
            "hist_os",
            "hist_kernel",
            "hist_arch",
            "hist_accel",
            "hist_tz",
            "hist_cpu",
        )
        for name in hist_names:
            state_dict.pop(prefix + name, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.standardize:
            return x
        bn_dtype = (
            self.bn.running_mean.dtype if hasattr(self.bn, "running_mean") else x.dtype
        )
        orig_dtype = x.dtype
        if x.dtype != bn_dtype:
            x = x.to(dtype=bn_dtype)
        if self.mode in _NORM_MODES:
            if self.training:
                out = self.bn(x)
                self._accumulate_batch(x)
                return out.to(dtype=orig_dtype)
            return self.bn(x).to(dtype=orig_dtype)
        elif self.mode in _DENORM_MODES:
            std = (self.bn.running_var + self.bn.eps).sqrt().view(1, -1)
            mean = self.bn.running_mean.view(1, -1)
            out = x * std + mean
            if out.dtype != orig_dtype:
                out = out.to(dtype=orig_dtype)
            return out
        raise ValueError(f"invalid mode: {self.mode}")

    @torch.jit.ignore
    def fit_numpy(
        self,
        X: "np.ndarray",
        *,
        floc: Optional[Union[float, "np.ndarray"]] = None,
        fscale: Optional[Union[float, "np.ndarray"]] = None,
        apply_bn: bool = False,
    ) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:

        if _np is None:
            raise RuntimeError("NumPy is required; install it with 'pip install numpy'.")
        if _sps is None:
            raise RuntimeError("SciPy is required; install it with 'pip install scipy'.")
        X = _np.asarray(X, dtype=_np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"X shape must be (N, {self.n_features})")
        D = self.n_features
        df = _np.full(D, _np.nan, dtype=_np.float64)
        loc = _np.full(D, _np.nan, dtype=_np.float64)
        sc = _np.full(D, _np.nan, dtype=_np.float64)

        def _sel(v: float | "np.ndarray" | None, j: int) -> float | None:
            if v is None:
                return None
            return float(v if _np.isscalar(v) else v[j])

        for j in range(D):
            xj = _np.asarray(X[:, j], dtype=_np.float64)
            xj = xj[_np.isfinite(xj)]
            if xj.size == 0:
                continue
            kw = {}
            fl = _sel(floc, j)
            fs = _sel(fscale, j)
            if fl is not None:
                kw["floc"] = fl
            if fs is not None:
                kw["fscale"] = fs
            try:
                dfj, locj, scj = _sps.t.fit(xj, **kw)
            except Exception:

                locj = _np.nanmean(xj)
                s2 = _np.nanvar(xj)
                dfj = 10.0
                scj = _np.sqrt(max(s2 * (dfj - 2.0) / dfj, 1e-12))
            df[j], loc[j], sc[j] = dfj, locj, scj

        self.t_df.copy_(torch.from_numpy(df))
        self.t_loc.copy_(torch.from_numpy(loc))
        self.t_scale.copy_(torch.from_numpy(sc))

        if apply_bn:
            rm = (
                torch.from_numpy(loc)
                .to(self.bn.running_mean.dtype)
                .to(self.bn.running_mean.device)
            )
            var_np = _np.where(
                df > 2.0,
                sc**2 * df / (df - 2.0),
                _np.asarray(self.bn.running_var.detach().cpu(), dtype=_np.float64),
            )
            rv = (
                torch.from_numpy(var_np)
                .to(self.bn.running_var.dtype)
                .to(self.bn.running_var.device)
            )
            self.bn.running_mean.copy_(rm)
            self.bn.running_var.copy_(rv)
        return df, loc, sc


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, axes: Tuple[str, ...]) -> None:
        super().__init__()
        self.axes = axes
        if len(axes) < 1 or any(a not in ("l", "t", "t_score", "h", "w") for a in axes):
            raise ValueError("axes must be drawn from {'l','t','t_score','h','w'}")
        if d_model % len(axes) != 0:
            raise ValueError("d_model must be divisible by number of axes")
        self.d_axis = d_model // len(axes)
        if self.d_axis % 2 != 0:
            raise ValueError("per-axis dimension must be even")
        self.register_buffer(
            "_cache_meta",
            torch.tensor([-1, -1, -1], dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer("_cache_pe", torch.empty(0, 0), persistent=False)
        self._cache_device: torch.device | None = None
        self._cache_dtype: torch.dtype | None = None

    @staticmethod
    def _flatten(n: int, dim: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / dim)
        )
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
            and self._cache_meta.numel() == 3
            and tuple(int(x) for x in self._cache_meta.tolist()) == (t, h, w)
        ):
            return self._cache_pe
        chunks: List[torch.Tensor] = []
        if "t" in self.axes:
            pt = self._flatten(t, self.d_axis, device).view(t, 1, 1, self.d_axis)
            chunks.append(pt.expand(t, h, w, self.d_axis))
        if "h" in self.axes:
            ph = self._flatten(h, self.d_axis, device).view(1, h, 1, self.d_axis)
            chunks.append(ph.expand(t, h, w, self.d_axis))
        if "w" in self.axes:
            pw = self._flatten(w, self.d_axis, device).view(1, 1, w, self.d_axis)
            chunks.append(pw.expand(t, h, w, self.d_axis))
        if "l" in self.axes:
            pl = self._flatten(t, self.d_axis, device).view(t, 1, 1, self.d_axis)
            chunks.append(pl.expand(t, 1, 1, self.d_axis))
        if "t_score" in self.axes:
            pt_score = self._flatten(t, self.d_axis, device).view(t, 1, 1, self.d_axis)
            chunks.append(pt_score.expand(t, h, w, self.d_axis))
        pe = torch.cat(chunks, dim=-1).view(-1, self.d_axis * len(self.axes))
        pe = pe.to(dtype=dtype)
        self._cache_meta = torch.tensor([t, h, w], dtype=torch.int64)
        self._cache_pe = pe
        self._cache_device = device
        self._cache_dtype = dtype
        return pe


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


def norm_layer(norm_type: str, d_model: int) -> nn.Module:
    kind = str(norm_type).lower()
    if kind in {"layernorm", "layer_norm", "ln"}:
        return LayerNorm(d_model)
    if kind in {"rmsnorm", "rms_norm", "rms"} and hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(d_model)
    if kind in {"batchnorm", "batchnorm1d", "bn", "bn1d"}:
        return nn.BatchNorm1d(d_model)
    return LayerNorm(d_model)


def stochastic_depth_schedule(max_rate: float, depth: int) -> list[float]:
    if depth <= 0:
        return []
    if max_rate <= 0.0:
        return [0.0 for _ in range(depth)]
    step = float(max_rate) / max(1, depth)
    return [step * float(index + 1) for index in range(depth)]


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
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if ndim not in (1, 2, 3):
            raise ValueError(f"ndim must be 1, 2, or 3, got {ndim!r}")
        self.ndim = int(ndim)
        self.grid = grid
        self.grid_3d = grid_3d
        if any((p <= 0 for p in patch)):
            raise ValueError(f"patch sizes must be positive, got {patch}")
        if stride is not None and (
            len(stride) < self.ndim or any((s <= 0 for s in stride[: self.ndim]))
        ):
            raise ValueError(
                (
                    "stride must have length >= "
                    f"{self.ndim} with positive values, got {stride}"
                )
            )
        self.dropout = nn.Dropout(dropout)
        self.d_model = int(d_model)
        self.patch = patch
        self.pad_to_multiple = bool(pad_to_multiple)
        self.static_spatial: Optional[Tuple[int, ...]] = getattr(
            self, "static_spatial", None
        )
        if self.static_spatial is None:
            hw = getattr(self, "static_hw", None)
            if hw is not None and self.ndim == 2:
                self.static_spatial = (int(hw[0]), int(hw[1]))
        stride = patch if stride is None else stride
        match self.ndim:
            case 1:
                self.proj = nn.Conv1d(
                    in_channels,
                    d_model,
                    kernel_size=(patch[0],),
                    stride=(stride[0],),
                )
            case 2:
                self.proj = nn.Conv2d(
                    in_channels,
                    d_model,
                    kernel_size=(patch[0], patch[1]),
                    stride=(stride[0], stride[1]),
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
                        length = fdim
                        kernel = self.patch[0]
                        need = (length + kernel - 1) // kernel * kernel
                        if fdim < need:
                            x = torch.nn.functional.pad(x, (0, need - fdim))
                        return x.view(b, 1, -1)
                    (grid_length,) = self.grid
                    if fdim < grid_length:
                        x = torch.nn.functional.pad(x, (0, grid_length - fdim))
                    elif fdim > grid_length:
                        raise ValueError(
                            f"[B,F] grid(L={grid_length}) but F={fdim} > L."
                        )
                    return x.view(b, 1, grid_length)
                case 2:
                    if self.grid is None:
                        side = int(math.ceil(math.sqrt(fdim)))
                        h = w = side
                    else:
                        h, w = self.grid
                    need = h * w
                    if fdim < need:
                        x = torch.nn.functional.pad(x, (0, need - fdim))
                    elif fdim > need:
                        raise ValueError(f"[B,F] grid(H={h},W={w}) but F={fdim} > H*W.")
                    return x.view(b, h, w)
                case 3:
                    if self.grid_3d is None:
                        side = int(round(fdim ** (1.0 / 3.0)))
                        t = h = w = max(1, side)
                    else:
                        t, h, w = self.grid_3d
                    need = t * h * w
                    if fdim < need:
                        x = torch.nn.functional.pad(x, (0, need - fdim))
                    elif fdim > need:
                        raise ValueError(
                            f"[B,F] grid(T={t},H={h},W={w}) but F={fdim} > T*H*W."
                        )
                    return x.view(b, t, h, w)
        return x

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        match self.ndim:
            case 1:
                length = x.shape[-1]
                kernel = self.patch[0]
                need = (length + kernel - 1) // kernel * kernel
                if length < need:
                    x = F.pad(x, (0, need - length))
            case 2:
                h, w = x.shape[-2:]
                kh, kw = self.patch[:2]
                need_h = (h + kh - 1) // kh * kh
                need_w = (w + kw - 1) // kw * kw
                if h < need_h or w < need_w:
                    x = F.pad(x, (0, need_w - w, 0, need_h - h))
            case 3:
                t, h, w = x.shape[-3:]
                kt, kh, kw = self.patch
                need_t = (t + kt - 1) // kt * kt
                need_h = (h + kh - 1) // kh * kh
                need_w = (w + kw - 1) // kw * kw
                if t < need_t or h < need_h or w < need_w:
                    x = F.pad(
                        x,
                        (0, need_w - w, 0, need_h - h, 0, need_t - t),
                    )
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        x = self._normalize_shape(x)
        x = torch.atleast_3d(x)
        if self.static_spatial is not None:
            x = self._pad_or_crop_to_nd(x, self.static_spatial)
        elif x.shape[1] != self.patch[0] and self.pad_to_multiple:
            x = self._dynamic_pad(x)
        y = self.proj(x)
        match self.ndim:
            case 1:
                b, d, length = y.shape
                tokens = y.transpose(1, 2).contiguous().view(b, length, d)
                meta = (length, 1, 1)
            case 2:
                b, d, h, w = y.shape
                tokens = y.permute(0, 2, 3, 1).contiguous().view(b, h * w, d)
                meta = (1, h, w)
            case 3:
                b, d, t, h, w = y.shape
                tokens = y.permute(0, 2, 3, 4, 1).contiguous().view(b, t * h * w, d)
                meta = (t, h, w)
            case _:
                raise RuntimeError("Unsupported ndim for PatchEmbedding")
        return (self.dropout(tokens), meta)

    def _pad_or_crop_to_nd(
        self, x: torch.Tensor, target: Tuple[int, ...]
    ) -> torch.Tensor:
        if len(target) != self.ndim:
            raise ValueError(
                f"static_spatial must have length {self.ndim}, got {len(target)}"
            )
        tgt = [int(v) for v in target]
        if any(v <= 0 for v in tgt):
            raise ValueError("static_spatial values must be positive")
        spatial = list(x.shape[-self.ndim :])
        pads: list[int] = []
        for cur, want in reversed(list(zip(spatial, tgt))):
            pad_right = max(want - cur, 0)
            pads.extend([0, pad_right])
        if any(pads):
            x = F.pad(x, tuple(pads))
        slices = [slice(None)] * x.ndim
        base = x.ndim - self.ndim
        for offset, want in enumerate(tgt):
            slices[base + offset] = slice(0, want)
        return x[tuple(slices)]

    def _dynamic_pad(self, x: torch.Tensor) -> torch.Tensor:
        return self._pad(x)


class CrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        bias: bool = True,
        *args: Any,
        use_gate: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead for attention")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.norm_q = norm_layer(norm_type, d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.use_gate = bool(use_gate)
        if self.use_gate:
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("gate", None)
        self.sdpa = DotProductAttention()

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
        qh, kh, vh = (
            reshape_for_mha(qn, B, self.nhead, self.head_dim),
            reshape_for_mha(k, B, self.nhead, self.head_dim),
            reshape_for_mha(v, B, self.nhead, self.head_dim),
        )
        yh = self.sdpa(qh, kh, vh, attn_mask=attn_mask)
        y = yh.transpose(1, 2).contiguous().view(B, Nq, D)
        y = self.out_proj(self.dropout(y))
        if self.use_gate:
            y = torch.sigmoid(self.gate) * y
        return q + y


class PatchAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        *args: Any,
        coord_dim: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead for PatchAttention")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = self.d_model // self.nhead
        self.coord_dim = int(coord_dim)
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
        self.attn = DotProductAttention(num_heads=self.nhead, head_dim=self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        if coords.shape[:2] != (B, N):
            raise ValueError("coords must have shape (B, N, C)")
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        qh, kh, vh = (
            reshape_for_mha(q, B, self.nhead, self.head_dim),
            reshape_for_mha(k, B, self.nhead, self.head_dim),
            reshape_for_mha(v, B, self.nhead, self.head_dim),
        )
        P = N           
                                               
                                                                 
        coords = coords.contiguous()
        coords_f32 = coords if coords.dtype == torch.float32 else coords.float()
        attn_bias = None
        try:
                                    
            rel = (coords_f32.unsqueeze(2) - coords_f32.unsqueeze(1))                    
                                                  
            attn_bias = self.rel_bias(rel.to(x.dtype)).permute(0, 3, 1, 2).contiguous()
        except Exception:
            attn_bias = None

                                                      
        additive = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                                                                   
                _mask = ~attn_mask
                additive = to_additive_mask(
                    _mask,
                    batch=B,
                    heads=self.nhead,
                    seq_q=P,
                    seq_k=P,
                    dtype=(attn_bias.dtype if attn_bias is not None else qh.dtype),
                    device=qh.device,
                )
            else:
                                                                          
                additive = attn_mask
                if additive.dim() == 3:
                    additive = additive.unsqueeze(1)                     
                if additive.size(1) == 1 and self.nhead > 1:
                    additive = additive.expand(B, self.nhead, P, P)
                additive = additive.to(
                    dtype=(attn_bias.dtype if attn_bias is not None else qh.dtype),
                    device=qh.device,
                )
        if attn_bias is not None and additive is not None:
            attn_bias = attn_bias + additive
        elif additive is not None:
            attn_bias = additive
                                                          

                                                           
                                                        
        use_shared_weights = bool(getattr(self, "reuse_weights_for_base", False))
        if use_shared_weights:
            base = torch.zeros((B, self.nhead, P, self.head_dim), dtype=vh.dtype, device=vh.device)
        else:
            base = self.attn(
                qh,
                kh,
                vh,
                attn_mask=(attn_bias.to(dtype=qh.dtype) if attn_bias is not None else None),
                training=self.training,
            )
                                                                      
                                                                 
        H, Dh = self.nhead, self.head_dim
        scale = 1.0 / math.sqrt(float(self.head_dim))
                                      
        rtile = min(P, int(getattr(self, "rel_rtile", getattr(self, "rel_tile", 64))))
        ctile = min(P, int(getattr(self, "rel_ctile", getattr(self, "rel_tile", 64))))
        use_triton = (_HAS_TRITON and qh.is_cuda and (not torch.is_grad_enabled()))

                        
        rel_ctx = torch.zeros((B, H, P, Dh), dtype=qh.dtype, device=qh.device)

                                                                                          
        for s in range(0, P, rtile):
            e = min(s + rtile, P)
            q_blk = qh[:, :, s:e, :]              

                                                                              
            m = torch.full((B, H, e - s, 1), -float("inf"), dtype=torch.float32, device=qh.device)
            sum_exp = torch.zeros_like(m)             
            for t in range(0, P, ctile):
                u = min(t + ctile, P)
                k_blk = kh[:, :, t:u, :]              
                                         
                sc = torch.einsum("bhid,bhjd->bhij", q_blk, k_blk) * scale
                if attn_bias is not None:
                    sc = sc + attn_bias[:, :, s:e, t:u].to(dtype=sc.dtype)
                sc = sc.float()
                                        
                tile_max = sc.amax(dim=-1, keepdim=True)             
                m_new = torch.maximum(m, tile_max)
                                                                             
                sum_exp = sum_exp * torch.exp(m - m_new) + torch.exp(sc - m_new).sum(dim=-1, keepdim=True)
                m = m_new

                                                                             
            all_masked = torch.isneginf(m) | (~torch.isfinite(sum_exp)) | (sum_exp <= 0)
            if all_masked.any():
                                                                            
                m = torch.where(all_masked, torch.zeros_like(m), m)
                sum_exp = torch.where(all_masked, torch.ones_like(sum_exp), sum_exp)

                                                                   
                                                           
            if use_triton:
                o_slice = torch.zeros_like(rel_ctx[:, :, s:e, :], dtype=torch.float32)

            for t in range(0, P, ctile):
                u = min(t + ctile, P)
                k_blk = kh[:, :, t:u, :]              
                sc = torch.einsum("bhid,bhjd->bhij", q_blk, k_blk) * scale             
                if attn_bias is not None:
                    sc = sc + attn_bias[:, :, s:e, t:u].to(dtype=sc.dtype)
                sc = sc.float()
                                                             
                w_t = torch.exp(sc - m) / (sum_exp + 1e-12)

                                                                                            
                drop_p = 0.0
                _attn_p = getattr(self.attn, "dropout_p", None)
                if _attn_p is None:
                    _attn_p = getattr(self.attn, "dropout", 0.0)
                try:
                    drop_p = float(_attn_p or 0.0)
                except Exception:
                    drop_p = 0.0
                if self.training and drop_p > 0.0:
                    w_t = F.dropout(w_t, p=drop_p, training=True)

                                                            
                rel_chunk = (coords_f32[:, s:e, :].unsqueeze(2) - coords_f32[:, t:u, :].unsqueeze(1))
                #debug
                if rel_chunk.numel() == 0:
                    raise RuntimeError("rel_chunk is empty?!")

                print(
                    "[DEBUG] rel_chunk shape:", tuple(rel_chunk.shape),
                    "numel:", rel_chunk.numel(),
                    "dtype:", rel_chunk.dtype,
                    "device:", rel_chunk.device,
                )
                                                                    
                rv = self.rel_value(rel_chunk.to(x.dtype))             
                                           
                rv = rv.view(B, (e - s), (u - t), H, Dh).permute(0, 3, 1, 2, 4).contiguous()
                if use_shared_weights:
                                                              
                    v_blk = vh[:, :, t:u, :]              
                    _base_acc32 = (w_t.unsqueeze(-1) * v_blk.to(torch.float32).unsqueeze(2)).sum(dim=-2)
                    base[:, :, s:e, :] += _base_acc32.to(base.dtype)
                                                          
                if use_triton:
                    w_ctg = w_t.contiguous()                                  
                                                           
                    B_, H_, K_, J_, DH_ = B, H, (e - s), (u - t), Dh
                                        
                    SWB, SWH, SWK, SWJ = w_ctg.stride()
                    SRVB, SRVH, SRVK, SRVJ, SRVDH = rv.stride()
                    SOB, SOH, SOK, SODH = o_slice.stride()
                                                                     
                    _reduce_weighted_sum[lambda META: (B_*H_, K_, triton.cdiv(DH_, META['BLOCK_DH']))](
                        w_ctg, rv, o_slice,
                        B_, H_, K_, J_, DH_,
                        SWB, SWH, SWK, SWJ,
                        SRVB, SRVH, SRVK, SRVJ, SRVDH,
                        SOB, SOH, SOK, SODH,
                    )
                else:
                    _acc32 = (w_t.unsqueeze(-1) * rv.to(torch.float32)).sum(dim=-2)           
                    rel_ctx[:, :, s:e, :] += _acc32.to(qh.dtype)

            if use_triton:
                                                          
                rel_ctx[:, :, s:e, :] += o_slice.to(dtype=qh.dtype)

        context = base + rel_ctx
        return context.transpose(1, 2).contiguous().view(B, N, self.d_model)


class Retention(nn.Module):
    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()
        self.msr = MultiScaleRetention(d_model, nhead)

    def forward(
        self,
        x: torch.Tensor,
        *args: Any,
        attn_mask: Optional[torch.Tensor] = None,
        state: Optional[dict] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        h = self.msr(x, attn_mask=attn_mask, state=state)
        if isinstance(h, tuple):
            out, new_state = h
            if new_state is None:
                new_state = state
        else:
            out, new_state = h, state
        return out, new_state


def _get_dilated_mask(
    seq_len: int,
    *args: Any,
    dilation: int = 1,
    window_size: Optional[int] = None,
    causal: bool = False,
    **kwargs: Any,
) -> torch.Tensor:

    if dilation < 1:
        raise ValueError(f"dilation must be >= 1, got {dilation}")
    L = int(seq_len)
    device = torch.device("cpu")
    i = torch.arange(L, device=device).unsqueeze(1).expand(L, L)
    j = torch.arange(L, device=device).unsqueeze(0).expand(L, L)
    dist = (i - j).abs()
    congruent = ((i - j) % dilation) == 0
    within = (
        torch.ones_like(congruent, dtype=torch.bool)
        if window_size is None
        else (dist <= window_size)
    )
    not_future = (j <= i) if causal else torch.ones_like(congruent, dtype=torch.bool)
    allowed = congruent & within & not_future
    return (~allowed).contiguous()


class DilatedAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *args: Any,
        dilation: int = 1,
        window_size: Optional[int] = None,
        causal: bool = False,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        batch_first: bool = True,
        bias: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dilation = int(dilation)
        self.window_size = window_size
        self.causal = causal
        self.batch_first = batch_first
        self.nhead = int(num_heads)
        self.head_dim = int(embed_dim // max(self.nhead, 1))
        self.dropout_p = float(dropout)
        self.__stf_attention_profile__ = {
            "format": "xs",
            "num_heads": self.nhead,
            "head_dim": self.head_dim,
            "dropout_attr": "dropout_p",
            "effective_window_attr": ["window_size"],
            "include_softmax_scale_dropout": True,
        }

        self.norm1 = _Norm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = _Norm(embed_dim)
        hidden = int(mlp_ratio * embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim, bias=True),
        )

        self._mask_cache_cpu: Dict[int, torch.Tensor] = {}
        self._mask_cache_gpu: Dict[Tuple[int, int], torch.Tensor] = {}

    def _get_mask(self, L: int, device: torch.device) -> torch.Tensor:
        key = (L, device.index if device.type == "cuda" else -1)
        mask_gpu = self._mask_cache_gpu.get(key)
        if mask_gpu is not None and mask_gpu.device == device:
            return mask_gpu
        mask_cpu = self._mask_cache_cpu.get(L)
        if mask_cpu is None:
            mask_cpu = (
                _get_dilated_mask(
                    L,
                    dilation=self.dilation,
                    window_size=self.window_size,
                    causal=self.causal,
                )
                .detach()
                .contiguous()
                .cpu()
            )
            self._mask_cache_cpu[L] = mask_cpu
        mask_gpu = mask_cpu.to(device=device, non_blocking=True)
        self._mask_cache_gpu[key] = mask_gpu
        return mask_gpu

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.batch_first:
            x = x.transpose(0, 1)
        B, L, _ = x.shape
        residual = x
        x = self.norm1(x)
        mask = self._get_mask(L, x.device)
        attn_out, attn_w = self.attn(
            x,
            x,
            x,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            is_causal=self.causal,
        )
        x = residual + self.dropout(attn_out)
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        if not self.batch_first:
            x = x.transpose(0, 1)
        return x, (attn_w if need_weights else None)


class PointTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        *args: Any,
        coord_dim: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_type: str = "layernorm",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.coord_dim = int(coord_dim)
        self.d_model = int(d_model)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(p=drop_path, mode="row")
        self.norm1 = norm_layer(norm_type, self.d_model)
        self.attn = PatchAttention(self.d_model, nhead, coord_dim=self.coord_dim)
        self.norm2 = norm_layer(norm_type, self.d_model)
        hid = int(self.d_model * mlp_ratio * (2.0 / 3.0))
        self.ffn = SwiGLU(self.d_model, hid, out_dim=self.d_model, dropout=dropout)
        self._ln_materialized = False

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if is_meta_or_fake_tensor(x):
            raise RuntimeError("meta/fake tensor reached PointTransformer.forward (x)")
        if is_meta_or_fake_tensor(coords):
            raise RuntimeError("meta/fake tensor reached PointTransformer.forward")
        if coords.shape[:2] != x.shape[:2] or coords.size(-1) != self.coord_dim:
            raise ValueError(
                f"coords must be (B, N, {self.coord_dim}), got {tuple(coords.shape)} vs x {tuple(x.shape)}"
            )
        x = x.contiguous()
        coords = coords.contiguous()
        if attn_mask is not None:
            attn_mask = attn_mask.contiguous()
            if is_meta_or_fake_tensor(attn_mask):
                raise RuntimeError("attn_mask is meta before attention")

        def _materialize_ln_(ln: nn.LayerNorm, ref: torch.Tensor) -> None:
            if not isinstance(ln, nn.LayerNorm):
                return
            dev = ref.device
            target_dtype = torch.float32 if dev.type == "cpu" else ref.dtype
            w = getattr(ln, "weight", None)
            b = getattr(ln, "bias", None)
            try:
                from torch.distributed._tensor import DTensor as _DTensor
            except Exception:
                dtensor_types: Tuple[type, ...] = tuple()
            else:
                dtensor_types = (_DTensor,)
            w_data = getattr(w, "data", None)
            if isinstance(w, torch.Tensor) and (
                is_meta_or_fake_tensor(w)
                or isinstance(w, dtensor_types)
                or isinstance(w_data, dtensor_types)
            ):
                ln.weight = nn.Parameter(
                    torch.ones(ln.normalized_shape, device=dev, dtype=target_dtype)
                )
            b_data = getattr(b, "data", None)
            if isinstance(b, torch.Tensor) and (
                is_meta_or_fake_tensor(b)
                or isinstance(b, dtensor_types)
                or isinstance(b_data, dtensor_types)
            ):
                ln.bias = nn.Parameter(
                    torch.zeros(ln.normalized_shape, device=dev, dtype=target_dtype)
                )

        _materialize_ln_(self.norm1, x)
        _materialize_ln_(self.norm2, x)
        self._ln_materialized = True
        _x = x
        if isinstance(_x, torch.Tensor) and is_meta_or_fake_tensor(_x):
            raise RuntimeError("x is meta before LayerNorm")
        if (
            _x.device.type == "cpu"
            and _x.is_floating_point()
            and _x.dtype != torch.float32
        ):
            _x = _x.float()
        _x = self.norm1(_x)
        if isinstance(_x, torch.Tensor) and is_meta_or_fake_tensor(_x):
            raise RuntimeError("x is meta after LayerNorm")
        if (
            _x.device.type == "cpu"
            and x.is_floating_point()
            and x.dtype != torch.float32
        ):
            _x = _x.to(x.dtype)
        y = self.attn(_x, coords, attn_mask=attn_mask)
        x = x + self.drop_path(self.dropout(y))
        _x2 = x
        if is_meta_or_fake_tensor(_x2):
            raise RuntimeError("x is meta before LayerNorm(norm2)")
        if (
            _x2.device.type == "cpu"
            and _x2.is_floating_point()
            and _x2.dtype != torch.float32
        ):
            _x2 = _x2.float()
        _x2 = self.norm2(_x2)
        if is_meta_or_fake_tensor(_x2):
            raise RuntimeError("x is meta after LayerNorm(norm2)")
        if (
            _x2.device.type == "cpu"
            and x.is_floating_point()
            and x.dtype != torch.float32
        ):
            _x2 = _x2.to(x.dtype)
        x = x + self.drop_path(self.dropout(self.ffn(_x2)))
        return x


_MODELING_TYPE_ALIASES: dict[str, str] = {
    "ss": "ss",
    "spatial": "ss",
    "sxs": "ss",
    "tt": "tt",
    "temporal": "tt",
    "txt": "tt",
    "ts": "ts",
    "txs": "ts",
    "temporal-spatial": "ts",
    "temporo-spatial": "ts",
    "temporospatial": "ts",
    "st": "st",
    "sxt": "st",
    "spatiotemporal": "st",
    "spatio-temporal": "st",
}


def _coerce_modeling_types(value: Any) -> str:
    mode = str(value).strip().lower()
    normalized = _MODELING_TYPE_ALIASES.get(mode)
    if normalized is None:
        raise ValueError(f"Unsupported modeling type '{value}'")
    return normalized


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        depth: int,
        *args: Any,
        coord_dim: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_type: str = "layernorm",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        drops = stochastic_depth_schedule(drop_path, depth)
        self.blocks = nn.ModuleList(
            [
                PointTransformer(
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
        self.norm = norm_layer(norm_type, d_model)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if is_meta_or_fake_tensor(x):
            raise RuntimeError("x is meta/fake before SpatialEncoder.forward")
        if is_meta_or_fake_tensor(coords):
            raise RuntimeError("coords is meta/fake before SpatialEncoder.forward")
        if x.dim() != 3:
            raise ValueError(
                f"SpatialEncoder expects (B, N, C) tokens, got shape {tuple(x.shape)}"
            )
        if coords.dim() != 3:
            raise ValueError(
                f"SpatialEncoder expects (B, N, D) coords, got shape {tuple(coords.shape)}"
            )
        if x.shape[:2] != coords.shape[:2]:
            raise ValueError(
                "tokens/coords batch or length mismatch: "
                f"tokens={tuple(x.shape)} vs coords={tuple(coords.shape)}"
            )
        x = x.contiguous()
        coords = coords.contiguous()
        if coords.dtype != torch.float32:
            coords = coords.float()
        if attn_mask is not None:
            if is_meta_or_fake_tensor(attn_mask):
                raise RuntimeError(
                    "attn_mask is meta/fake before SpatialEncoder.forward"
                )
            attn_mask = attn_mask.contiguous()
        for i, blk in enumerate(self.blocks):
                                                               
            B, N, D = x.shape
            _patch = getattr(self, "patch_size", 512)
            _shift = getattr(self, "shift_order", True)
            _bits  = getattr(self, "morton_bits", 10)
            perm, invperm = _serialize_z_index(coords, bits=_bits, patch=_patch,
                                                   shift_order=_shift, block_index=i)
            x_s = x.gather(1, perm.unsqueeze(-1).expand(B, N, D))
            c_s = coords.gather(1, perm.unsqueeze(-1).expand(B, N, coords.size(-1)))
            out_s = torch.empty_like(x_s)
                                                
            m_s = None
            if attn_mask is not None:
                                                          
                if attn_mask.dim() == 3:
                    m_s = attn_mask.gather(
                        1, perm.unsqueeze(-1).expand(B, N, N)
                    ).gather(
                        2, perm.unsqueeze(1).expand(B, N, N)
                    )
                elif attn_mask.dim() == 4:
                    if attn_mask.size(1) != 1:
                        raise ValueError("attn_mask with per-head shape not supported here")
                    m_s = attn_mask.squeeze(1).gather(
                        1, perm.unsqueeze(-1).expand(B, N, N)
                    ).gather(
                        2, perm.unsqueeze(1).expand(B, N, N)
                    )
                else:
                    raise ValueError("attn_mask must be (B,N,N)")
            for s, e in _block_ranges(N, _patch):
                xb = x_s[:, s:e, :]
                cb = c_s[:, s:e, :]
                mb = None if m_s is None else m_s[:, s:e, s:e]
                out_s[:, s:e, :] = blk(xb, cb, attn_mask=mb)
            x = out_s.gather(1, invperm.unsqueeze(-1).expand(B, N, D))
        out = self.norm(x)
        if is_meta_or_fake_tensor(out):
            raise RuntimeError("SpatialEncoder produced meta/fake tensor")
        return out.contiguous()


class RetNet(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        *args: Any,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_type: str = "layernorm",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(norm_type, d_model)
        self.retention = Retention(d_model, nhead)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(p=drop_path, mode="row")
        self.norm2 = norm_layer(norm_type, d_model)
        hid = int(d_model * mlp_ratio * (2.0 / 3.0))
        self.ffn = SwiGLU(d_model, hid, out_dim=d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        if is_meta_or_fake_tensor(x):
            raise RuntimeError("meta/fake tensor reached RetNet.forward")
        x = x.contiguous()
        if causal_mask is not None:
            causal_mask = causal_mask.contiguous()
        h, state = self.retention(self.norm1(x), attn_mask=causal_mask, state=state)
        x = x + self.drop_path(self.dropout(h))
        x = x + self.drop_path(self.dropout(self.ffn(self.norm2(x))))
        return x, state


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        depth: int,
        *args: Any,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_type: str = "layernorm",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        drops = stochastic_depth_schedule(drop_path, depth)
        self.blocks = nn.ModuleList(
            [
                RetNet(
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
        self.norm = norm_layer(norm_type, d_model)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        state: Optional[dict] = None,
        *args: Any,
        return_state: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[dict]]]:
        next_state = state
        for blk in self.blocks:
            x, next_state = blk(x, causal_mask=causal_mask, state=next_state)
        x = self.norm(x)
        if return_state:
            return x, next_state
        return x


class CrossTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        *args: Any,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.cross_s = CrossAttention(
            d_model, nhead, dropout=dropout, norm_type=norm_type
        )
        self.cross_t = CrossAttention(
            d_model, nhead, dropout=dropout, norm_type=norm_type
        )
        self.mix_norm = norm_layer(norm_type, 2 * d_model)
        hid = int(2 * d_model * mlp_ratio * (2.0 / 3.0))
        self.mix = SwiGLU(2 * d_model, hid, out_dim=d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(p=drop_path, mode="row")
        self._fixed_mode: Optional[str] = getattr(self, "modeling_type", None)

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        temporal_tokens: torch.Tensor,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        if self._fixed_mode is not None:
            mode_l = _coerce_modeling_types(self._fixed_mode)
            if mode_l == "ss":
                return self.cross_s(spatial_tokens, temporal_tokens)
            if mode_l == "tt":
                return self.cross_t(temporal_tokens, spatial_tokens)
            if mode_l == "ts":
                s_context = self.cross_s(spatial_tokens, temporal_tokens)
                t_context = self.cross_t(temporal_tokens, spatial_tokens)
                base = torch.cat(
                    [
                        t_context,
                        s_context.mean(dim=1, keepdim=True).expand_as(t_context),
                    ],
                    dim=-1,
                )
                fused = self.mix(self.mix_norm(base))
                return t_context + self.drop_path(self.dropout(fused))
            if mode_l == "st":
                s_context = self.cross_s(spatial_tokens, temporal_tokens)
                t_context = self.cross_t(temporal_tokens, spatial_tokens)
                base = torch.cat(
                    [
                        s_context,
                        t_context.mean(dim=1, keepdim=True).expand_as(s_context),
                    ],
                    dim=-1,
                )
                fused = self.mix(self.mix_norm(base))
                return s_context + self.drop_path(self.dropout(fused))
        requested = mode if mode is not None else "spatiotemporal"
        return self._forward_dynamically(spatial_tokens, temporal_tokens, requested)

    def _forward_dynamically(
        self,
        spatial_tokens: torch.Tensor,
        temporal_tokens: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        mode_l = _coerce_modeling_types(mode)
        s_context = self.cross_s(spatial_tokens, temporal_tokens)
        t_context = self.cross_t(temporal_tokens, spatial_tokens)
        if mode_l == "ss":
            return s_context
        if mode_l == "tt":
            return t_context
        if mode_l == "ts":
            base = torch.cat(
                [
                    t_context,
                    s_context.mean(dim=1, keepdim=True).expand_as(t_context),
                ],
                dim=-1,
            )
            fused = self.mix(self.mix_norm(base))
            return t_context + self.drop_path(self.dropout(fused))
        if mode_l == "st":
            base = torch.cat(
                [
                    s_context,
                    t_context.mean(dim=1, keepdim=True).expand_as(s_context),
                ],
                dim=-1,
            )
            fused = self.mix(self.mix_norm(base))
            return s_context + self.drop_path(self.dropout(fused))
        raise RuntimeError(f"Unhandled mode: {mode_l}")


@dataclass
class Payload:
    tokens: torch.Tensor
    context: torch.Tensor
    flat: torch.Tensor
    offset: torch.Tensor
    context_shape: Tuple[int, ...]


class LongNet(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int,
        *args: Any,
        dilation_growth: int = 2,
        base_dilation: int = 1,
        window_size: Optional[int] = None,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        causal: bool = False,
        batch_first: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.nhead = int(num_heads)
        self.head_dim = int(embed_dim // max(self.nhead, 1))
        self.dropout_p = float(dropout)
        self.__stf_attention_profile__ = {
            "format": "xs",
            "num_heads": self.nhead,
            "head_dim": self.head_dim,
            "dropout_attr": "dropout_p",
            "effective_window_attr": ["window_size", "block_size"],
            "include_softmax_scale_dropout": True,
        }
        self.batch_first = batch_first
        self._impl: Optional[nn.Module] = None
        try:
            ts_longnet = import_module("torchscale.net.longnet")
        except Exception:
            self._impl = None
        else:
            ctor_variants = (
                {
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "depth": depth,
                    "dropout": dropout,
                },
                {
                    "d_model": embed_dim,
                    "nhead": num_heads,
                    "num_layers": depth,
                    "dropout": dropout,
                },
            )
            longnet_ctor = getattr(ts_longnet, "LongNet", None)
            impl = None
            if callable(longnet_ctor):
                for kw in ctor_variants:
                    try:
                        impl = longnet_ctor(**kw)
                        break
                    except Exception:
                        continue
            self._impl = impl

        if self._impl is not None:
            self._using = "torchscale"
            self._impl_batch_first = getattr(self._impl, "batch_first", True)
        else:
            self._using = "fallback"
            self._impl_batch_first = True
            layers: List[nn.Module] = []
            dilation = base_dilation
            for _ in range(depth):
                layers.append(
                    DilatedAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        dilation=dilation,
                        window_size=window_size,
                        dropout=dropout,
                        mlp_ratio=mlp_ratio,
                        causal=causal,
                        batch_first=True,
                    )
                )
                dilation = max(1, dilation * max(1, int(dilation_growth)))
            self.layers = nn.ModuleList(layers)
            self.norm = nn.LayerNorm(embed_dim)

    @property
    def using(self) -> str:
        return self._using

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        **_: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._impl is not None:
            out = x
            if (
                self._impl_batch_first
                and out.dim() == 3
                and out.shape[0] != out.shape[1]
                and not self._impl_batch_first
            ):
                out = out.transpose(0, 1)
            try:
                out = self._impl(out)
            except Exception as exc:
                warnings.warn(
                    f"torchscale LongNet call failed: {exc}. Returning the input tensor.",
                    RuntimeWarning,
                )
            if self._impl_batch_first is not True:
                out = out.transpose(0, 1)
            return out, None
        attn_w: Optional[torch.Tensor] = None
        out = x
        for layer in self.layers:
            out, attn_w = layer(
                out,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
            )
        out = self.norm(out)
        return out, attn_w


class GlobalEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int,
        *args: Any,
        dilation_growth: int = 2,
        base_dilation: int = 1,
        window_size: Optional[int] = None,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        causal: bool = False,
        batch_first: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.backbone = LongNet(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            dilation_growth=dilation_growth,
            base_dilation=base_dilation,
            window_size=window_size,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            causal=causal,
            batch_first=batch_first,
        )

    @property
    def using(self) -> str:
        return self.backbone.using

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        **_: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.backbone(
            x,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )


class LocalProcessor(nn.Module):
    def __init__(
        self, in_dim: int, out_shape: Sequence[int], config: ModelConfig
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_shape = tuple((int(v) for v in out_shape))
        self.out_dim = int(math.prod(self.out_shape) if self.out_shape else 1)
        self.d_model = int(config.depth)
        self.nhead = int(config.heads)
        self.modeling_type = _coerce_modeling_types(config.modeling_type)
        self.spatial_tokens = max(1, int(config.spatial_latents))
        self.temporal_tokens = max(1, int(config.temporal_latents))
        self.mlp_ratio = float(config.mlp_ratio)
        self.dropout = float(config.dropout)
        self.drop_path = float(config.drop_path)
        self.norm_type = str(config.normalization_method)
        self.spatial_tokenizer = nn.Linear(
            self.in_dim, self.spatial_tokens * self.d_model
        )
        self.temporal_tokenizer = nn.Linear(
            self.in_dim, self.temporal_tokens * self.d_model
        )
        self.register_buffer(
            "spatial_coords_template",
            self._get_spatial_coords(self.spatial_tokens, device=torch.device("cpu")),
            persistent=False,
        )
        self.spatial_encoder = SpatialEncoder(
            self.d_model,
            self.nhead,
            depth=max(1, int(config.spatial_depth)),
            coord_dim=self.spatial_coords_template.shape[-1],
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            drop_path=self.drop_path,
            norm_type=self.norm_type,
        )
        self.temporal_encoder = TemporalEncoder(
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
        self._fixed_mode: Optional[str] = getattr(self, "modeling_type", None)
        self.norm = norm_layer(self.norm_type, self.d_model)
        hid = int(self.d_model * max(1.0, self.mlp_ratio))
        self.head = nn.Sequential(
            norm_layer(self.norm_type, self.d_model),
            nn.Linear(self.d_model, hid),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hid, self.out_dim),
        )

    @staticmethod
    def _get_spatial_coords(n_tokens: int, device: torch.device) -> torch.Tensor:
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
                coords.append(
                    (
                        x / (side - 1 if side > 1 else 1),
                        y / (side - 1 if side > 1 else 1),
                        z / (side - 1 if side > 1 else 1),
                    )
                )
        return torch.tensor(coords, dtype=torch.float32, device=device)

    def _spatial_coords(
        self, batch: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        coords = self.spatial_coords_template.to(device=device, dtype=dtype)
        return coords.unsqueeze(0).expand(batch, -1, -1)

    def forward(self, x: torch.Tensor) -> Payload:
        B = x.shape[0]
        spatial_raw = self.spatial_tokenizer(x)
        expected_spatial = B * self.spatial_tokens * self.d_model
        if spatial_raw.numel() != expected_spatial:
            raise RuntimeError(
                "spatial tokenizer output has unexpected numel: "
                f"got {spatial_raw.numel()} vs expected {expected_spatial}"
            )
        spatial_tokens = spatial_raw.reshape(
            B, self.spatial_tokens, self.d_model
        ).contiguous()
        temporal_raw = self.temporal_tokenizer(x)
        expected_temporal = B * self.temporal_tokens * self.d_model
        if temporal_raw.numel() != expected_temporal:
            raise RuntimeError(
                "temporal tokenizer output has unexpected numel: "
                f"got {temporal_raw.numel()} vs expected {expected_temporal}"
            )
        temporal_tokens = temporal_raw.reshape(
            B, self.temporal_tokens, self.d_model
        ).contiguous()
        coords = self._spatial_coords(B, x.device, spatial_tokens.dtype)
        spatial_out = self.spatial_encoder(spatial_tokens, coords)
        temporal_out = self.temporal_encoder(temporal_tokens)
        mode = self._fixed_mode
        if mode is not None:
            mode_l = _coerce_modeling_types(mode)
            if mode_l == "ss":
                tokens = spatial_out
            elif mode_l == "tt":
                tokens = temporal_out
            elif mode_l == "ts":
                if hasattr(self.perception, "_forward_dynamically"):
                    tokens = self.perception._forward_dynamically(
                        temporal_out, spatial_out, "ts"
                    )
                else:
                    tokens = self.perception(temporal_out, spatial_out, mode="ts")
            elif mode_l == "st":
                if hasattr(self.perception, "_forward_dynamically"):
                    tokens = self.perception._forward_dynamically(
                        spatial_out, temporal_out, "st"
                    )
                else:
                    tokens = self.perception(spatial_out, temporal_out, mode="st")
            else:
                raise RuntimeError(f"Unhandled modeling type '{mode}'")
        else:
            tokens = self._forward_dynamically(spatial_out, temporal_out)
        tokens = self.norm(tokens)
        tokens = tokens.contiguous()
        pooled = tokens.mean(dim=1)
        flat = self.head(pooled)
        flat = flat.contiguous()
        context = flat.reshape(B, *self.out_shape).contiguous()
        dims = tuple(range(1, context.ndim))
        offset = context.mean(dim=dims, keepdim=True).contiguous()
        return Payload(
            tokens=tokens,
            context=context,
            flat=flat,
            offset=offset,
            context_shape=self.out_shape,
        )

    def _forward_dynamically(
        self, spatial_out: torch.Tensor, temporal_out: torch.Tensor
    ) -> torch.Tensor:
        mode = getattr(self, "modeling_type", None)
        if mode is None:
            raise RuntimeError("modeling_type is not set")
        mode_l = _coerce_modeling_types(mode)
        if mode_l == "ss":
            return spatial_out
        if mode_l == "tt":
            return temporal_out
        if mode_l == "ts":
            return self.perception(temporal_out, spatial_out, mode="ts")
        if mode_l == "st":
            return self.perception(spatial_out, temporal_out, mode="st")
        raise RuntimeError(f"Unhandled modeling type '{mode_l}'")

    def decode(
        self, tokens: torch.Tensor, *args: Any, apply_norm: bool = False, **kwargs: Any
    ) -> torch.Tensor:
        if apply_norm:
            tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        flat = self.head(pooled)
        return flat.reshape(tokens.shape[0], *self.out_shape)


class LossWeightPolicy(Protocol):
    def weights(self) -> Tuple[float, float]: ...

    def update(
        self,
        top_loss: Optional[torch.Tensor],
        bottom_loss: Optional[torch.Tensor],
    ) -> None:
        raise NotImplementedError


class Root(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_shape: Sequence[int],
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_shape = tuple((int(x) for x in out_shape))
        self.out_dim = int(math.prod(self.out_shape))

        self.input_norm = Normal(self.in_dim, standardize=True, skew=True, mode="norm")

        self.output_denorm = Normal(
            self.out_dim, standardize=True, skew=True, mode="denorm"
        )
        self.output_affine = Affine(self.out_dim)
        if config.device is not None:
            self._device = torch.device(config.device)
        else:
            if torch.cuda.is_available():
                device_name = "cuda"
            elif (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                device_name = "mps"
            elif (
                getattr(torch, "is_vulkan_available", None)
                and torch.is_vulkan_available()
            ):
                device_name = "vulkan"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                device_name = "xpu"
            else:
                device_name = "cpu"
            self._device = torch.device(device_name)
        self.is_norm_linear = bool(getattr(config, "use_linear_branch", False))
        self.linear_branch = (
            nn.Linear(self.in_dim, self.out_dim).to(self._device)
            if self.is_norm_linear
            else None
        )
        self.local_net = LocalProcessor(self.in_dim, self.out_shape, config=config).to(
            self._device
        )
        global_net = GlobalEncoder(
            int(config.depth),
            int(config.heads),
            depth=max(1, int(getattr(config, "temporal_depth", 1))),
            mlp_ratio=float(getattr(config, "mlp_ratio", 4.0)),
            dropout=float(getattr(config, "dropout", 0.0)),
            batch_first=True,
        ).to(self._device)
        self.global_net = global_net
        try:
            self.microbatch = int(getattr(config, "microbatch", 0))
        except Exception:
            self.microbatch = 0
        if self.microbatch < 0:
            raise ValueError(f"config.microbatch must be >= 0, got {self.microbatch}")
        self._auto_microbatch_pending = self.microbatch == 0
        self._activation_checkpoint = bool(
            getattr(config, "activation_checkpoint", False)
        )
        try:
            self.register_buffer(
                "output_baked_flag",
                torch.tensor(0, dtype=torch.uint8),
                persistent=True,
            )
        except Exception:
            pass
        raw_mode = getattr(config, "compile_mode", "disabled")
        mode = str(raw_mode or "").strip()
        normalized_mode = mode.lower()
        disable_compile = normalized_mode in {"", "disabled", "none"}
        compile_mode_arg = normalized_mode if not disable_compile else None
        self.local_net = Gradient.compile(
            self.local_net,
            mode=compile_mode_arg,
            fullgraph=False,
            dynamic=False,
            backend="inductor",
            disable=disable_compile,
        )
        self.global_net = Gradient.compile(
            self.global_net,
            mode=compile_mode_arg,
            fullgraph=False,
            dynamic=False,
            backend="inductor",
            disable=disable_compile,
        )

        self.input_norm = self.input_norm.to(self._device)
        self.output_denorm = self.output_denorm.to(self._device)
        self.output_affine = self.output_affine.to(self._device)
        self.__config = config
        self._base_dtype: Optional[torch.dtype] = getattr(self, "base_dtype", None)

    @staticmethod
    def _cast_graph_safe(
        x: torch.Tensor, device: torch.device, dtype: Optional[torch.dtype]
    ) -> torch.Tensor:
        target_dtype = dtype or x.dtype
        if x.device != device:
            return x.to(device=device, dtype=target_dtype, non_blocking=True)
        if x.dtype != target_dtype:
            return x.to(dtype=target_dtype)
        return x

    def _auto_microbatch(self, features: torch.Tensor | TensorDictBase, device: torch.device) -> int:
        """가용 가속기 메모리 90% 리밋을 목표로 보수적 microbatch를 산정."""
        dev_t = getattr(device, "type", "cpu")
        if isinstance(features, TensorDictBase):
            X = features.get("features")
        else:
            X = features
        if not isinstance(X, torch.Tensor):
            return 64
        one = X[:1]
        bytes_per_sample = int(one.nelement()) * int(one.element_size())
        fudge = 8
        max_batch = 1 << 14
        free_bytes = None
        if dev_t == "cuda":
            with contextlib.suppress(Exception):
                free, _ = torch.cuda.mem_get_info(device)
                free_bytes = int(free)
        elif dev_t == "xpu":
            with contextlib.suppress(Exception):
                props = getattr(torch.xpu, "get_device_properties", None)
                mem_alloc = getattr(torch.xpu, "memory_allocated", None)
                if callable(props) and callable(mem_alloc):
                    total = int(props(device).total_memory); used = int(mem_alloc(device))
                    free_bytes = max(0, total - used)
        elif dev_t == "mps":
            with contextlib.suppress(Exception):
                from ..backend.system import Memory
                free_bytes = int(Memory.available() * 0.25)
        if not free_bytes or free_bytes <= 0:
            return 64
        budget = int(free_bytes * 0.90)
        denom = max(1, bytes_per_sample * fudge)
        mb = max(1, min(max_batch, budget // denom))
        return int(mb)

    def forward(
        self,
        features: torch.Tensor | TensorDictBase,
        *args: Any,
        labels_flat: Optional[torch.Tensor] = None,
        net_loss: Optional[nn.Module] = None,
        global_loss: Optional[nn.Module] = None,
        local_loss: Optional[nn.Module] = None,
        loss_weights: Optional[Union[Tuple[float, float], LossWeightPolicy]] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]] | TensorDictBase:
        td_input: TensorDictBase | None = None
        if isinstance(features, TensorDictBase):
            td_input = features
            td_labels = td_input.get("labels_flat", None)
            td_features = td_input.get("features")
            if td_features is None:
                raise KeyError("TensorDict input requires a 'features' key")
            features = td_features
            if labels_flat is None and td_labels is not None:
                labels_flat = td_labels
            td_net_loss = td_input.get("net_loss", None)
            td_global_loss = td_input.get("global_loss", None)
            td_local_loss = td_input.get("local_loss", None)
            if net_loss is None and td_net_loss is not None:
                net_loss = td_net_loss
            if global_loss is None and td_global_loss is not None:
                global_loss = td_global_loss
            if local_loss is None and td_local_loss is not None:
                local_loss = td_local_loss
            td_loss_weights = td_input.get("loss_weights", None)
            if loss_weights is None and td_loss_weights is not None:
                loss_weights = td_loss_weights
        device = self._device
        if features.ndim == 3 and features.shape[1] == 1:
            features = features.reshape(features.shape[0], -1)

        if (
            self.training
            and labels_flat is not None
            and not isinstance(net_loss, (nn.CrossEntropyLoss, nn.NLLLoss))
            and self.output_denorm.standardize
        ):
            target_stats = labels_flat.reshape(labels_flat.shape[0], -1)
            bn = self.output_denorm.bn
            stats_device = (
                bn.running_mean.device
                if isinstance(bn.running_mean, torch.Tensor)
                else self._device
            )
            stats_dtype = (
                bn.running_mean.dtype
                if isinstance(bn.running_mean, torch.Tensor)
                else target_stats.dtype
            )
            target_stats = target_stats.to(device=stats_device, dtype=stats_dtype)
            self.output_denorm._accumulate_batch(target_stats)

        norm_dtype = self.input_norm.bn.running_mean.dtype
        features = features.to(device=device, dtype=norm_dtype)
        features = self.input_norm(features)
        assert features.ndim == 2 and features.shape[1] == self.in_dim
        b = features.shape[0]
        amp_enabled = device.type != "cpu"
        base_param = next(self.local_net.parameters())
        base_dtype = self._base_dtype or base_param.dtype
        infer_mode = labels_flat is None or (
            net_loss is None and global_loss is None and (local_loss is None)
        )
        if self._auto_microbatch_pending:
            try:
                mb = self._auto_microbatch(features, device)
                self.microbatch = max(1, int(mb))
            except Exception:
                self.microbatch = max(1, int(getattr(self, "microbatch", 64) or 64))
            self._auto_microbatch_pending = False
        num_slices = (b + self.microbatch - 1) // self.microbatch
        token_chunks: List[torch.Tensor] = []
        context_chunks: List[torch.Tensor] = []
        use_activation_checkpoint = bool(self._activation_checkpoint and not infer_mode)
        if not infer_mode:
            self.local_net.train()
            self.global_net.train()
            for idx in range(num_slices):
                s = idx * self.microbatch
                e = min(b, (idx + 1) * self.microbatch)
                x_slice = self._cast_graph_safe(features[s:e], device, base_dtype)
                ctx = (
                    Autocast.float(device) if amp_enabled else Autocast.suspend(device)
                )
                with ctx:
                    out: Payload = self.local_net(x_slice)
                out_tokens = torch.nan_to_num(
                    out.tokens, nan=0.0, posinf=0.0, neginf=0.0
                ).to(dtype=base_dtype)
                out_context = torch.nan_to_num(
                    out.context, nan=0.0, posinf=0.0, neginf=0.0
                ).to(dtype=base_dtype)
                token_chunks.append(out_tokens)
                context_chunks.append(out_context)
        else:
            self.local_net.eval()
            self.global_net.eval()
            for idx in range(num_slices):
                s = idx * self.microbatch
                e = min(b, (idx + 1) * self.microbatch)
                x_slice = self._cast_graph_safe(features[s:e], device, base_dtype)
                with contextlib.ExitStack() as stack:
                    stack.enter_context(Gradient.inference(self.local_net))
                    stack.enter_context(
                        Autocast.float(device)
                        if amp_enabled
                        else Autocast.suspend(device)
                    )
                    out = self.local_net(x_slice)
                out_tokens = torch.nan_to_num(
                    out.tokens, nan=0.0, posinf=0.0, neginf=0.0
                ).to(dtype=base_dtype)
                out_context = torch.nan_to_num(
                    out.context, nan=0.0, posinf=0.0, neginf=0.0
                ).to(dtype=base_dtype)
                token_chunks.append(out_tokens)
                context_chunks.append(out_context)
        tokens = torch.cat(token_chunks, dim=0).to(device=device, dtype=base_dtype)
        context = torch.cat(context_chunks, dim=0).to(device=device, dtype=base_dtype)
        tokens = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
        context = torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        assembled = torch.nan_to_num(
            context.reshape(b, -1), nan=0.0, posinf=0.0, neginf=0.0
        )
        if self.is_norm_linear and self.linear_branch is not None:
            bl = self.linear_branch(
                self._cast_graph_safe(features, self._device, assembled.dtype)
            )
            assembled = assembled + bl
        tokens = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
        t32 = tokens.to(torch.float32)
        tokens_centered = (t32 - t32.mean(dim=1, keepdim=True)).to(dtype=tokens.dtype)
        if infer_mode:
            with Gradient.inference(self.global_net):
                with (
                    Autocast.float(device) if amp_enabled else Autocast.suspend(device)
                ):
                    refined_tokens, _ = self.global_net(tokens_centered)
            refined_tokens = torch.nan_to_num(
                refined_tokens, nan=0.0, posinf=0.0, neginf=0.0
            )
            decode_tokens = refined_tokens.detach().clone()
            with Gradient.inference(self.local_net):
                with (
                    Autocast.float(device) if amp_enabled else Autocast.suspend(device)
                ):
                    residual_context = self.local_net.decode(
                        decode_tokens, apply_norm=True
                    )
            residual_context = torch.nan_to_num(
                residual_context, nan=0.0, posinf=0.0, neginf=0.0
            )
        else:
            with torch.enable_grad():

                def _global_tokens(inp: torch.Tensor) -> torch.Tensor:
                    with (
                        Autocast.float(device)
                        if amp_enabled
                        else Autocast.suspend(device)
                    ):
                        out, _ = self.global_net(inp)
                    return out

                if use_activation_checkpoint and activation_checkpoint is not None:
                    refined_tokens = activation_checkpoint(
                        _global_tokens, tokens_centered
                    )
                else:
                    refined_tokens = _global_tokens(tokens_centered)
                refined_tokens = torch.nan_to_num(
                    refined_tokens, nan=0.0, posinf=0.0, neginf=0.0
                )

                def _decode_tokens(inp: torch.Tensor) -> torch.Tensor:
                    with (
                        Autocast.float(device)
                        if amp_enabled
                        else Autocast.suspend(device)
                    ):
                        return self.local_net.decode(inp, apply_norm=True)

                if use_activation_checkpoint and activation_checkpoint is not None:
                    residual_context = activation_checkpoint(
                        _decode_tokens, refined_tokens
                    )
                else:
                    residual_context = _decode_tokens(refined_tokens)
                residual_context = torch.nan_to_num(
                    residual_context, nan=0.0, posinf=0.0, neginf=0.0
                )
        residual = torch.nan_to_num(
            residual_context.reshape(b, -1), nan=0.0, posinf=0.0, neginf=0.0
        )
        if residual.dtype != assembled.dtype:
            residual = residual.to(dtype=assembled.dtype)
        y_hat = torch.nan_to_num(assembled + residual, nan=0.0, posinf=0.0, neginf=0.0)
        is_cls_loss = (
            isinstance(net_loss, (nn.CrossEntropyLoss, nn.NLLLoss))
            if net_loss is not None
            else False
        )
        y_hat_out = y_hat
        if not is_cls_loss:
            y_hat_for_loss = self.output_affine(self.output_denorm(y_hat_out))
        else:
            y_hat_for_loss = y_hat_out
        loss_val: Optional[torch.Tensor] = None
        if labels_flat is not None and (
            global_loss is not None or local_loss is not None
        ):
            controller: Optional[LossWeightPolicy] = None
            weights: Tuple[float, float]
            if loss_weights is None:
                weights = (1.0, 0.0)
            elif isinstance(loss_weights, (tuple, list)):
                seq = list(loss_weights)
                if len(seq) != 2:
                    raise ValueError("loss_weights requires two values")
                weights = (float(seq[0]), float(seq[1]))
            else:
                controller = cast(LossWeightPolicy, loss_weights)
                weights = controller.weights()
            tgt = labels_flat.to(
                device=y_hat_for_loss.device, dtype=y_hat_for_loss.dtype
            )
            total = y_hat_for_loss.new_tensor(0.0, dtype=y_hat_for_loss.dtype)
            top_component: Optional[torch.Tensor] = None
            bottom_component: Optional[torch.Tensor] = None
            y_top = y_hat_for_loss
            y_bot = assembled.to(
                device=y_hat_for_loss.device, dtype=y_hat_for_loss.dtype
            )
            if global_loss is not None:
                top_component = global_loss(y_top, tgt)
                total = total + weights[0] * top_component
            if local_loss is not None:
                bottom_component = local_loss(y_bot, tgt)
                total = total + weights[1] * bottom_component
            if controller is not None:
                controller.update(top_component, bottom_component)
            loss_val = total
        elif net_loss is not None and labels_flat is not None:
            if is_cls_loss:
                tgt = labels_flat.to(device=y_hat_for_loss.device).long()
                loss_val = net_loss(y_hat_for_loss, tgt)
            else:
                tgt = labels_flat.to(
                    device=y_hat_for_loss.device, dtype=y_hat_for_loss.dtype
                )
                loss_val = net_loss(y_hat_for_loss, tgt)

        y_hat_out = y_hat_for_loss
        pred = y_hat_out.reshape(b, *self.out_shape)
        if td_input is not None:
            out_td = td_input.clone()
            out_td.set("pred", pred)
            out_td.set("refined_tokens", refined_tokens)
            out_td.set("residual_context", residual_context)
            if loss_val is not None:
                loss_td = loss_val
                if isinstance(loss_td, torch.Tensor) and loss_td.ndim == 0:
                    batch_size = tuple(out_td.batch_size)
                    if len(batch_size):
                        loss_td = loss_td.expand(batch_size)
                out_td.set("loss_total", loss_td)
            else:
                with contextlib.suppress(KeyError):
                    out_td.del_("loss_total")
            return out_td
        return (pred, loss_val)

    @staticmethod
    def flatten_y(
        labels: Sequence[torch.Tensor],
        *args: Any,
        dtype: Optional[torch.dtype] = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        out = torch.stack([label.reshape(-1) for label in labels], dim=0)
        if dtype is not None:
            out = out.to(dtype=dtype)
        if pin_memory and out.device.type == "cpu":
            out = out.pin_memory()
        return (out.contiguous(), tuple(labels[0].shape))

    @staticmethod
    def unflatten_y(flat: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
        return flat.reshape(flat.shape[0], *shape).contiguous()
