from __future__ import annotations

import os
import time
from pathlib import Path

import torch

from .faults import ResourceSample, RuntimePhase
from .pressure import ResourceCapacity

_PROC_SELF_CGROUP = Path("/proc/self/cgroup")
_CGROUP_V2_ROOT = Path("/sys/fs/cgroup")
_CGROUP_V1_MEMORY_ROOT = Path("/sys/fs/cgroup/memory")
_CGROUP_V1_UNLIMITED_THRESHOLD = 1 << 60


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


def _safe_relative_cgroup_path(value: str) -> Path | None:
    parts = tuple(part for part in value.strip().split("/") if part)
    if any(part in {".", ".."} for part in parts):
        return None
    return Path(*parts)


def _read_cgroup_membership() -> tuple[str | None, str | None]:
    try:
        lines = _PROC_SELF_CGROUP.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None, None

    v2_path: str | None = None
    v1_memory_path: str | None = None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controller_text, relative_path = parts
        if controller_text == "":
            v2_path = relative_path
            continue
        if "memory" in controller_text.split(","):
            v1_memory_path = relative_path
    return v2_path, v1_memory_path


def _parse_cgroup_limit(value: str, *, version: int) -> int | None:
    stripped = value.strip()
    if not stripped or stripped == "max":
        return None
    try:
        limit = int(stripped)
    except ValueError:
        return None
    if limit <= 0:
        return None
    if version == 1 and limit >= _CGROUP_V1_UNLIMITED_THRESHOLD:
        return None
    return limit


def _read_cgroup_limit_file(
    path: Path,
    *,
    version: int,
) -> int | None:
    try:
        value = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    return _parse_cgroup_limit(value, version=version)


def _relative_path_hierarchy(relative: Path) -> tuple[Path, ...]:
    paths = [relative]
    paths.extend(parent for parent in relative.parents if parent != Path(".."))
    return tuple(paths)


def _read_cgroup_v2_hierarchy_limits(relative_text: str | None) -> tuple[int, ...]:
    if relative_text is None:
        relative = Path(".")
    else:
        relative = _safe_relative_cgroup_path(relative_text)
        if relative is None:
            return ()

    limits: list[int] = []
    for path in _relative_path_hierarchy(relative):
        limit = _read_cgroup_limit_file(
            _CGROUP_V2_ROOT / path / "memory.max",
            version=2,
        )
        if limit is not None:
            limits.append(limit)
    return tuple(limits)


def _read_v1_hierarchical_memory_limit(relative: Path) -> int | None:
    stat_path = _CGROUP_V1_MEMORY_ROOT / relative / "memory.stat"
    try:
        lines = stat_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None

    for line in lines:
        key, _, value = line.partition(" ")
        if key == "hierarchical_memory_limit":
            return _parse_cgroup_limit(value, version=1)
    return None


def _read_cgroup_v1_hierarchy_limits(relative_text: str | None) -> tuple[int, ...]:
    if relative_text is None:
        relative = Path(".")
    else:
        relative = _safe_relative_cgroup_path(relative_text)
        if relative is None:
            return ()

    hierarchical_limit = _read_v1_hierarchical_memory_limit(relative)
    if hierarchical_limit is not None:
        return (hierarchical_limit,)

    limits: list[int] = []
    for path in _relative_path_hierarchy(relative):
        limit = _read_cgroup_limit_file(
            _CGROUP_V1_MEMORY_ROOT / path / "memory.limit_in_bytes",
            version=1,
        )
        if limit is not None:
            limits.append(limit)
    return tuple(limits)


def _read_cgroup_memory_limit_bytes() -> int | None:
    v2_relative, v1_relative = _read_cgroup_membership()

    limits: list[int] = []
    if v2_relative is not None:
        limits.extend(_read_cgroup_v2_hierarchy_limits(v2_relative))
    if v1_relative is not None:
        limits.extend(_read_cgroup_v1_hierarchy_limits(v1_relative))

    if v2_relative is None and v1_relative is None:
        limits.extend(_read_cgroup_v2_hierarchy_limits(None))
        limits.extend(_read_cgroup_v1_hierarchy_limits(None))

    return min(limits) if limits else None


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
        cpu_limit_bytes = _read_cgroup_memory_limit_bytes()
        device_index = self._device_index()
        if device_index is None:
            return ResourceCapacity(
                cpu_total_bytes=cpu_total_bytes,
                cpu_limit_bytes=cpu_limit_bytes,
            )

        try:
            total_memory = int(
                torch.cuda.get_device_properties(device_index).total_memory
            )
        except Exception:
            return ResourceCapacity(
                cpu_total_bytes=cpu_total_bytes,
                cpu_limit_bytes=cpu_limit_bytes,
            )
        if total_memory <= 0:
            return ResourceCapacity(
                cpu_total_bytes=cpu_total_bytes,
                cpu_limit_bytes=cpu_limit_bytes,
            )

        return ResourceCapacity(
            cpu_total_bytes=cpu_total_bytes,
            cpu_limit_bytes=cpu_limit_bytes,
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
