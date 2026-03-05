# -*- coding: utf-8 -*-
from __future__ import annotations

# =============================================================================
# 1. Standard Library Imports
# =============================================================================
from typing import Any, Optional, Self

# =============================================================================
# 2. Third-Party Imports
# =============================================================================
import torch
from torch import nn


# =============================================================================
# Core Activation Modules
# =============================================================================
class GLU(nn.Module):
    def __init__(
        self: Self,
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
        
        if self.in_dim <= 0 or self.hid <= 0 or self.out_dim <= 0:
            raise ValueError(
                f"Dims must be > 0: in_dim={self.in_dim}, hidden_dim={self.hid}, out_dim={self.out_dim}"
            )
            
        p = float(dropout)
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Dropout probability {p} is invalid; must be in [0.0, 1.0]")
            
        if not isinstance(activation, nn.Module):
            raise TypeError(f"Invalid activation type: expected nn.Module, got {type(activation).__name__}")
            
        self.check_input = bool(check_input)
        self.activation = activation
        
        self.in_proj = nn.Linear(self.in_dim, 2 * self.hid, bias=bool(bias))
        
        match p:
            case 0.0:
                self.dropout = nn.Identity()
            case _:
                self.dropout = nn.Dropout(p=p)
                
        self.out_proj = nn.Linear(self.hid, self.out_dim, bias=bool(bias))

    @property
    def hidden_dim(self: Self) -> int:
        return self.hid

    def extra_repr(self: Self) -> str:
        p = getattr(self.dropout, "p", 0.0)
        return (
            f"in_dim={self.in_dim}, hidden_dim={self.hid}, out_dim={self.out_dim}, "
            f"activation={type(self.activation).__name__}, "
            f"dropout={p}, "
            f"check_input={self.check_input}"
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        if self.check_input and not torch.jit.is_tracing():
            if x.size(-1) != self.in_dim:
                raise ValueError(f"Expected input last dim to be {self.in_dim}, got {x.size(-1)}")
                
        a, b = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(self.dropout(self.activation(a) * b))


class GeGLU(GLU):
    def __init__(
        self: Self,
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
        self: Self,
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
