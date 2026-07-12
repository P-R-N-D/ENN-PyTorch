from __future__ import annotations

import os
import time
from pathlib import Path

import torch

from .faults import ResourceSample, RuntimePhase
from .pressure import ResourceCapacity


def _phase_name(phase: RuntimePhase | str) -> str:
    if isinstance(phase, RuntimePhase):
        return phase.value
    if not isinstance(phase, str):
        raise TypeError("resource sample phase must be a RuntimePhase or string.")
    if not phase:
        raise ValueError("resource sample phase must be non-empty.")
    return phase


def _read_cpu_total_bytes() -> int | None:
    try:
        physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        return None
    if physical_pages <= 0 or page_size <= 0:
        return None
    return physical_pages * page_size


def _read_cpu_rss_bytes() -> int | None:
    statm = Path("/proc/self/statm")
    try:
        resident_pages = int(statm.read_text(encoding="utf-8").split()[1])
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        return None
    return resident_pages * page_size


class ResourceMonitor:
    """Small CPU/CUDA resource snapshot helper.

    The monitor is deliberately observational. It does not choose batch sizes,
    recover from OOM, move tensors between devices, or write telemetry files.
    """

    def __init__(self, *, cuda_device: int | None = None) -> None:
        if cuda_device is not None and (
            not isinstance(cuda_device, int) or isinstance(cuda_device, bool)
        ):
            raise TypeError("ResourceMonitor.cuda_device must be an integer or None.")
        self.cuda_device = cuda_device

    def _device_index(self) -> int | None:
        if not torch.cuda.is_available():
            return None
        if self.cuda_device is not None:
            return self.cuda_device
        try:
            return int(torch.cuda.current_device())
        except Exception:
            return None

    def capacity(self) -> ResourceCapacity:
        cpu_total_bytes = _read_cpu_total_bytes()
        device_index = self._device_index()
        if device_index is None:
            return ResourceCapacity(cpu_total_bytes=cpu_total_bytes)

        try:
            total_memory = int(
                torch.cuda.get_device_properties(device_index).total_memory
            )
        except Exception:
            return ResourceCapacity(cpu_total_bytes=cpu_total_bytes)
        if total_memory <= 0:
            return ResourceCapacity(cpu_total_bytes=cpu_total_bytes)

        return ResourceCapacity(
            cpu_total_bytes=cpu_total_bytes,
            cuda_total_bytes=total_memory,
            cuda_device_index=device_index,
        )

    def reset_peak_memory_stats(self) -> None:
        device_index = self._device_index()
        if device_index is None:
            return
        try:
            torch.cuda.reset_peak_memory_stats(device_index)
        except Exception:
            return

    def sample(self, phase: RuntimePhase | str) -> ResourceSample:
        phase_value = _phase_name(phase)
        timestamp_ns = time.time_ns()
        cpu_rss_bytes = _read_cpu_rss_bytes()
        cuda_available = bool(torch.cuda.is_available())
        device_index = self._device_index()

        if not cuda_available or device_index is None:
            return ResourceSample(
                timestamp_ns=timestamp_ns,
                phase=phase_value,
                cpu_rss_bytes=cpu_rss_bytes,
                cuda_available=cuda_available,
                cuda_device_index=device_index,
            )

        try:
            allocated = int(torch.cuda.memory_allocated(device_index))
            reserved = int(torch.cuda.memory_reserved(device_index))
            max_allocated = int(torch.cuda.max_memory_allocated(device_index))
            max_reserved = int(torch.cuda.max_memory_reserved(device_index))
        except Exception:
            allocated = reserved = max_allocated = max_reserved = None

        return ResourceSample(
            timestamp_ns=timestamp_ns,
            phase=phase_value,
            cpu_rss_bytes=cpu_rss_bytes,
            cuda_available=cuda_available,
            cuda_device_index=device_index,
            cuda_allocated_bytes=allocated,
            cuda_reserved_bytes=reserved,
            cuda_max_allocated_bytes=max_allocated,
            cuda_max_reserved_bytes=max_reserved,
        )
