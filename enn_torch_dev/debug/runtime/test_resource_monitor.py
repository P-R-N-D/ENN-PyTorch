from __future__ import annotations

from types import SimpleNamespace

import pytest

import enn_torch_dev.runtime.resources as resources_module
from enn_torch_dev.runtime import ResourceCapacity, ResourceMonitor, RuntimePhase


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


def test_resource_monitor_capacity_is_cpu_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        resources_module.os,
        "sysconf",
        lambda name: {
            "SC_PHYS_PAGES": 100,
            "SC_PAGE_SIZE": 4096,
        }[name],
    )
    monkeypatch.setattr(resources_module.torch.cuda, "is_available", lambda: False)

    capacity = ResourceMonitor().capacity()

    assert capacity == ResourceCapacity(cpu_total_bytes=409_600)


def test_resource_monitor_capacity_falls_back_when_cpu_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_lookup_error(name: str) -> int:
        del name
        raise RuntimeError("sysconf unavailable")

    monkeypatch.setattr(resources_module.os, "sysconf", raise_lookup_error)
    monkeypatch.setattr(resources_module.torch.cuda, "is_available", lambda: False)

    assert ResourceMonitor().capacity() == ResourceCapacity()


def test_resource_monitor_capacity_rejects_invalid_cpu_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resources_module.os, "sysconf", lambda name: -1)
    monkeypatch.setattr(resources_module.torch.cuda, "is_available", lambda: False)

    assert ResourceMonitor().capacity() == ResourceCapacity()


def test_resource_monitor_capacity_reads_cuda_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resources_module.os,
        "sysconf",
        lambda name: {
            "SC_PHYS_PAGES": 200,
            "SC_PAGE_SIZE": 4096,
        }[name],
    )
    monkeypatch.setattr(resources_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(resources_module.torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(
        resources_module.torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(total_memory=8_192) if index == 2 else None,
    )

    capacity = ResourceMonitor().capacity()

    assert capacity == ResourceCapacity(
        cpu_total_bytes=819_200,
        cuda_total_bytes=8_192,
        cuda_device_index=2,
    )


def test_resource_monitor_capacity_falls_back_when_cuda_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resources_module.os,
        "sysconf",
        lambda name: {
            "SC_PHYS_PAGES": 50,
            "SC_PAGE_SIZE": 4096,
        }[name],
    )
    monkeypatch.setattr(resources_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(resources_module.torch.cuda, "current_device", lambda: 1)

    def raise_lookup_error(index: int) -> object:
        del index
        raise RuntimeError("device properties unavailable")

    monkeypatch.setattr(
        resources_module.torch.cuda,
        "get_device_properties",
        raise_lookup_error,
    )

    assert ResourceMonitor().capacity() == ResourceCapacity(
        cpu_total_bytes=204_800
    )
