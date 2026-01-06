# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Any

import torch
from torch import nn


class GLU(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        *args: Any,
        activation: nn.Module,
        check_input: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hid = int(hidden_dim)
        self.out_dim = int(out_dim if out_dim is not None else self.in_dim)
        if self.in_dim <= 0:
            raise ValueError(f"{self.__class__.__name__}: in_dim must be > 0, got {self.in_dim}")
        if self.hid <= 0:
            raise ValueError(f"{self.__class__.__name__}: hidden_dim must be > 0, got {self.hid}")
        if self.out_dim <= 0:
            raise ValueError(f"{self.__class__.__name__}: out_dim must be > 0, got {self.out_dim}")
        p = float(dropout)
        if p < 0.0 or p > 1.0:
            raise ValueError(f"{self.__class__.__name__}: dropout must be in [0, 1], got {p}")
        if not isinstance(activation, nn.Module):
            raise TypeError(
                f"{self.__class__.__name__}: activation must be an nn.Module, got {type(activation)!r}"
            )
        self.check_input = bool(check_input)
        self.activation = activation
        self.in_proj = nn.Linear(self.in_dim, 2 * self.hid, bias=bias)
        self.dropout = nn.Identity() if p == 0.0 else nn.Dropout(p)
        self.out_proj = nn.Linear(self.hid, self.out_dim, bias=bias)

    @property
    def hidden_dim(self) -> int:
        return self.hid

    def extra_repr(self) -> str:
        act_name = self.activation.__class__.__name__
        drop = 0.0 if isinstance(self.dropout, nn.Identity) else getattr(self.dropout, "p", None)
        return (
            f"in_dim={self.in_dim}, hidden_dim={self.hid}, out_dim={self.out_dim}, "
            f"activation={act_name}, dropout={drop}, check_input={self.check_input}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.check_input and x.size(-1) != self.in_dim:
            raise ValueError(
                f"{self.__class__.__name__} expected last dimension {self.in_dim}, got {x.size(-1)}"
            )
        a, b = self.in_proj(x).chunk(2, dim=-1)
        y = self.activation(a) * b
        y = self.dropout(y)
        return self.out_proj(y)


class GeGLU(GLU):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        *args: Any,
        check_input: bool = True,
    ) -> None:
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
            bias=bias,
            activation=nn.GELU(),
            check_input=check_input,
        )


class SwiGLU(GLU):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        *args: Any,
        check_input: bool = True,
    ) -> None:
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
            bias=bias,
            activation=nn.SiLU(),
            check_input=check_input,
        )