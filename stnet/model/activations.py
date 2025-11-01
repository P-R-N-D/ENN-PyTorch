# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


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
