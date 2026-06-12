from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from enn_torch_dev.executor import KVStore


class StepStatus(Enum):
    SUCCESS = "success"
    OOM_FAULT = "oom_fault"
    NONFINITE_FAULT = "nonfinite_fault"
    DATA_FAULT = "data_fault"
    RUNTIME_FAULT = "runtime_fault"


class RuntimePhase(Enum):
    TO_STORE = "to_store"
    FORWARD = "forward"
    LOSS = "loss"
    BACKWARD = "backward"
    OPTIMIZER = "optimizer"


@dataclass(frozen=True, slots=True)
class StepResult:
    status: StepStatus
    phase: RuntimePhase | None
    batch_size: int
    row_ids: torch.Tensor
    loss: torch.Tensor | None = None
    store: KVStore | None = None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is StepStatus.SUCCESS
