from __future__ import annotations

from numbers import Real
import math

import torch
from torch import Tensor, nn


def _validate_positive_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return value


def _validate_dropout(value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError("dropout must be a real number.")
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("dropout must be between 0.0 and 1.0.")
    return value


def _validate_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool.")
    return value


class RecurrentContextHead(nn.Module):
    """
    GRU-based context head for fused or graph-level sequence outputs.

    The head is intentionally independent from tile/stream executors. It can be
    attached as a normal ``nn.Module`` graph node, or called after
    ``GlobalLocalPipeline`` fusion.

    By default, input and output use ``(B, N, C)`` layout. ``hidden_dim`` may
    differ from ``input_dim``; an output projection maps the GRU output back to
    ``input_dim`` so residual connections and downstream graph nodes see a
    stable feature dimension.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = True,
        residual: bool = True,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = _validate_positive_int(input_dim, "input_dim")
        self.hidden_dim = (
            self.input_dim
            if hidden_dim is None
            else _validate_positive_int(hidden_dim, "hidden_dim")
        )
        self.num_layers = _validate_positive_int(num_layers, "num_layers")
        self.dropout = _validate_dropout(dropout)
        if self.num_layers == 1 and self.dropout > 0.0:
            raise ValueError(
                "dropout requires num_layers > 1 because GRU applies dropout "
                "only between stacked recurrent layers."
            )
        self.batch_first = _validate_bool(batch_first, "batch_first")
        self.residual = _validate_bool(residual, "residual")
        self.use_norm = _validate_bool(norm, "norm")

        self.input_norm = (
            nn.LayerNorm(self.input_dim) if self.use_norm else nn.Identity()
        )
        self.rnn = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.output_proj = (
            nn.Identity()
            if self.hidden_dim == self.input_dim
            else nn.Linear(self.hidden_dim, self.input_dim)
        )
        self.output_norm = (
            nn.LayerNorm(self.input_dim) if self.use_norm else nn.Identity()
        )

    def _batch_size_for(self, x: Tensor) -> int:
        return int(x.shape[0] if self.batch_first else x.shape[1])

    def _validate_input(self, x: Tensor) -> None:
        if not isinstance(x, Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)!r}")
        if x.ndim != 3:
            raise ValueError(f"x must be a 3D tensor, got ndim={x.ndim}.")
        if int(x.shape[-1]) != self.input_dim:
            raise ValueError(
                f"x last dimension must match input_dim: "
                f"{int(x.shape[-1])} != {self.input_dim}"
            )
        if not torch.is_floating_point(x):
            raise TypeError("x must use a floating dtype.")

    def _validate_state(self, state: Tensor | None, x: Tensor) -> None:
        if state is None:
            return
        if not isinstance(state, Tensor):
            raise TypeError(f"state must be a torch.Tensor, got {type(state)!r}")
        expected_shape = (
            self.num_layers,
            self._batch_size_for(x),
            self.hidden_dim,
        )
        if tuple(state.shape) != expected_shape:
            raise ValueError(
                f"state must have shape {expected_shape}, got {tuple(state.shape)}."
            )
        if state.device != x.device:
            raise ValueError("state must be on the same device as x.")
        if state.dtype != x.dtype:
            raise ValueError("state must have the same dtype as x.")

    def forward(
        self,
        x: Tensor,
        state: Tensor | None = None,
        *,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        self._validate_input(x)
        self._validate_state(state, x)
        return_state = _validate_bool(return_state, "return_state")

        h = self.input_norm(x)
        y, next_state = self.rnn(h, state)
        y = self.output_proj(y)
        if self.residual:
            y = y + x
        y = self.output_norm(y)

        if return_state:
            return y, next_state
        return y
