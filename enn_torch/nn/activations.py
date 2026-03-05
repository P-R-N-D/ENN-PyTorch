# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Self

import torch
from torch import nn


class GLU(nn.Module):
    def __init__(
        self: Self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        *args: Any,
        activation: nn.Module,
        check_input: bool = True,
    ) -> None:
        super().__init__()
        _ = args

        self.in_dim = int(in_dim)
        self.hid = int(hidden_dim)
        self.out_dim = int(out_dim if out_dim is not None else self.in_dim)

        if any(d <= 0 for d in (self.in_dim, self.hid, self.out_dim)):
            raise ValueError(
                f"Dims must be > 0: {self.in_dim}, {self.hid}, {self.out_dim}"
            )
        elif not 0.0 <= float(dropout) <= 1.0:
            raise ValueError(f"Dropout {dropout} invalid")
        elif not isinstance(activation, nn.Module):
            raise TypeError(f"Invalid activation type: {type(activation)}")

        self.check_input = bool(check_input)
        self.activation = activation
        self.in_proj = nn.Linear(self.in_dim, 2 * self.hid, bias=bias)
        self.dropout = nn.Identity() if dropout == 0.0 else nn.Dropout(dropout)
        self.out_proj = nn.Linear(self.hid, self.out_dim, bias=bias)

    @property
    def hidden_dim(self: Self) -> int:
        return self.hid

    def extra_repr(self: Self) -> str:
        return (
            f"in_dim={self.in_dim}, hidden_dim={self.hid}, out_dim={self.out_dim}, "
            f"activation={type(self.activation).__name__}, "
            f"dropout={getattr(self.dropout, 'p', 0.0)}, "
            f"check_input={self.check_input}"
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.check_input
            and (not torch.jit.is_tracing())
            and x.size(-1) != self.in_dim
        ):
            raise ValueError(f"Expected dim {self.in_dim}, got {x.size(-1)}")

        a, b = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(self.dropout(self.activation(a) * b))


class GeGLU(GLU):
    def __init__(
        self: Self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        *args: Any,
        check_input: bool = True,
    ) -> None:
        _ = args
        super().__init__(
            in_dim,
            hidden_dim,
            out_dim,
            dropout,
            bias,
            activation=nn.GELU(),
            check_input=check_input,
        )


class SwiGLU(GLU):
    def __init__(
        self: Self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        *args: Any,
        check_input: bool = True,
    ) -> None:
        _ = args
        super().__init__(
            in_dim,
            hidden_dim,
            out_dim,
            dropout,
            bias,
            activation=nn.SiLU(),
            check_input=check_input,
        )
