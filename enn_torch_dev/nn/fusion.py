from __future__ import annotations

from numbers import Real

import torch
from torch import Tensor, nn


class LocalGlobalFusion(nn.Module):
    """
    Learnable scalar gate for fusing global and local/tiled outputs.

    ``gate = sigmoid(logit)`` controls the local contribution:

    ``output = gate * local_out + (1 - gate) * global_out``.

    This module is intentionally small. Spatial, channel-wise, context-aware, or
    loss-conditioned gates should build on top of this baseline.
    """

    def __init__(
        self,
        *,
        init_logit: float = 0.0,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(init_logit, Real) or isinstance(init_logit, bool):
            raise TypeError("init_logit must be a real number.")
        if not isinstance(learnable, bool):
            raise TypeError("learnable must be a bool.")

        logit = torch.tensor(float(init_logit), dtype=torch.float32)
        if learnable:
            self.logit = nn.Parameter(logit)
        else:
            self.register_buffer("logit", logit)

    @property
    def gate(self) -> Tensor:
        return torch.sigmoid(self.logit)

    def forward(self, global_out: Tensor, local_out: Tensor) -> Tensor:
        if not isinstance(global_out, Tensor):
            raise TypeError(
                f"global_out must be a torch.Tensor, got {type(global_out)!r}"
            )
        if not isinstance(local_out, Tensor):
            raise TypeError(
                f"local_out must be a torch.Tensor, got {type(local_out)!r}"
            )
        if global_out.shape != local_out.shape:
            raise ValueError(
                "global_out and local_out must have the same shape: "
                f"{tuple(global_out.shape)} != {tuple(local_out.shape)}"
            )
        if global_out.device != local_out.device:
            raise ValueError("global_out and local_out must be on the same device.")
        if global_out.dtype != local_out.dtype:
            raise ValueError("global_out and local_out must have the same dtype.")

        gate = self.gate.to(device=global_out.device, dtype=global_out.dtype)
        return gate * local_out + (1.0 - gate) * global_out
