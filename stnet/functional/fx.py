"""General functional utilities for STNet modules."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "reshape_for_heads",
    "PositionalEncoding",
    "GeGLU",
    "SwiGLU",
]


def reshape_for_heads(
    tensor: torch.Tensor, batch_size: int, head_count: int, head_dim: int
) -> torch.Tensor:
    """Reshape a projection to multi-head layout."""

    return tensor.view(batch_size, -1, head_count, head_dim).transpose(1, 2)


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
    def _to_1d(n: int, dim: int, device: torch.device) -> torch.Tensor:
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
            pt = self._to_1d(t, self.d_axis, device).view(t, 1, 1, self.d_axis)
            chunks.append(pt.expand(t, h, w, self.d_axis))
        if "h" in self.axes:
            ph = self._to_1d(h, self.d_axis, device).view(1, h, 1, self.d_axis)
            chunks.append(ph.expand(t, h, w, self.d_axis))
        if "w" in self.axes:
            pw = self._to_1d(w, self.d_axis, device).view(1, 1, w, self.d_axis)
            chunks.append(pw.expand(t, h, w, self.d_axis))
        if "l" in self.axes:
            pl = self._to_1d(t, self.d_axis, device).view(t, 1, 1, self.d_axis)
            chunks.append(pl.expand(t, 1, 1, self.d_axis))
        if "t_score" in self.axes:
            pt_score = self._to_1d(t, self.d_axis, device).view(t, 1, 1, self.d_axis)
            chunks.append(pt_score.expand(t, h, w, self.d_axis))
        pe = torch.cat(chunks, dim=-1).view(-1, self.d_axis * len(self.axes))
        pe = pe.to(dtype=dtype)
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
