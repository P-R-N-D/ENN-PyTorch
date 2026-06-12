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
class ResourceSample:
    timestamp_ns: int
    phase: str
    cpu_rss_bytes: int | None = None
    cuda_available: bool = False
    cuda_device_index: int | None = None
    cuda_allocated_bytes: int | None = None
    cuda_reserved_bytes: int | None = None
    cuda_max_allocated_bytes: int | None = None
    cuda_max_reserved_bytes: int | None = None


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
    resource_samples: tuple[ResourceSample, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status is StepStatus.SUCCESS
