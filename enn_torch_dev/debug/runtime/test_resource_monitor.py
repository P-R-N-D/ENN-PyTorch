from __future__ import annotations

import pytest

from enn_torch_dev.runtime import ResourceMonitor, RuntimePhase


def test_resource_monitor_creates_cpu_safe_sample() -> None:
    monitor = ResourceMonitor()

    sample = monitor.sample("before_step")

    assert sample.phase == "before_step"
    assert isinstance(sample.timestamp_ns, int)
    assert sample.timestamp_ns > 0
    if sample.cpu_rss_bytes is not None:
        assert sample.cpu_rss_bytes > 0
    assert isinstance(sample.cuda_available, bool)


def test_resource_monitor_accepts_runtime_phase() -> None:
    monitor = ResourceMonitor()

    sample = monitor.sample(RuntimePhase.FORWARD)

    assert sample.phase == "forward"


def test_resource_monitor_reset_peak_is_safe_without_cuda() -> None:
    monitor = ResourceMonitor()

    monitor.reset_peak_memory_stats()


def test_resource_monitor_rejects_bad_phase() -> None:
    monitor = ResourceMonitor()

    with pytest.raises(TypeError):
        monitor.sample(object())  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        monitor.sample("")


def test_resource_monitor_rejects_bool_cuda_device() -> None:
    with pytest.raises(TypeError):
        ResourceMonitor(cuda_device=True)  # type: ignore[arg-type]
