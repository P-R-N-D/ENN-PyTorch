from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import enn_torch_dev.runtime.resources as resources_module
from enn_torch_dev.runtime import ResourceCapacity, ResourceMonitor, RuntimePhase


def _disable_cgroup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        resources_module,
        "_read_cgroup_memory_limit_bytes",
        lambda: None,
    )


def _configure_cgroup_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    membership: str,
) -> tuple[Path, Path]:
    proc_self_cgroup = tmp_path / "self.cgroup"
    proc_self_cgroup.write_text(membership, encoding="utf-8")
    cgroup_v2_root = tmp_path / "cgroup-v2"
    cgroup_v1_memory_root = tmp_path / "cgroup-v1-memory"
    cgroup_v2_root.mkdir()
    cgroup_v1_memory_root.mkdir()
    monkeypatch.setattr(resources_module, "_PROC_SELF_CGROUP", proc_self_cgroup)
    monkeypatch.setattr(resources_module, "_CGROUP_V2_ROOT", cgroup_v2_root)
    monkeypatch.setattr(
        resources_module,
        "_CGROUP_V1_MEMORY_ROOT",
        cgroup_v1_memory_root,
    )
    return cgroup_v2_root, cgroup_v1_memory_root


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
    _disable_cgroup(monkeypatch)
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
    _disable_cgroup(monkeypatch)

    def raise_lookup_error(name: str) -> int:
        del name
        raise RuntimeError("sysconf unavailable")

    monkeypatch.setattr(resources_module.os, "sysconf", raise_lookup_error)
    monkeypatch.setattr(resources_module.torch.cuda, "is_available", lambda: False)

    assert ResourceMonitor().capacity() == ResourceCapacity()


def test_resource_monitor_capacity_rejects_invalid_cpu_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_cgroup(monkeypatch)
    monkeypatch.setattr(resources_module.os, "sysconf", lambda name: -1)
    monkeypatch.setattr(resources_module.torch.cuda, "is_available", lambda: False)

    assert ResourceMonitor().capacity() == ResourceCapacity()


def test_resource_monitor_capacity_reads_cuda_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_cgroup(monkeypatch)
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
    _disable_cgroup(monkeypatch)
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

    assert ResourceMonitor().capacity() == ResourceCapacity(cpu_total_bytes=204_800)


def test_resource_monitor_capacity_reads_nested_cgroup_v2_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cgroup_v2_root, _ = _configure_cgroup_paths(
        monkeypatch, tmp_path, "0::/workloads/demo\n"
    )
    limit_path = cgroup_v2_root / "workloads" / "demo" / "memory.max"
    limit_path.parent.mkdir(parents=True)
    limit_path.write_text("4096\n", encoding="utf-8")
    monkeypatch.setattr(
        resources_module.os,
        "sysconf",
        lambda name: {"SC_PHYS_PAGES": 4, "SC_PAGE_SIZE": 4096}[name],
    )
    monkeypatch.setattr(resources_module.torch.cuda, "is_available", lambda: False)
    capacity = ResourceMonitor().capacity()
    assert capacity.cpu_total_bytes == 16_384
    assert capacity.cpu_limit_bytes == 4_096
    assert capacity.effective_cpu_bytes == 4_096


def test_resource_monitor_capacity_uses_cgroup_limit_when_physical_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cgroup_v2_root, _ = _configure_cgroup_paths(
        monkeypatch, tmp_path, "0::/workloads/demo\n"
    )
    limit_path = cgroup_v2_root / "workloads" / "demo" / "memory.max"
    limit_path.parent.mkdir(parents=True)
    limit_path.write_text("4096\n", encoding="utf-8")

    def raise_lookup_error(name: str) -> int:
        del name
        raise RuntimeError("sysconf unavailable")

    monkeypatch.setattr(resources_module.os, "sysconf", raise_lookup_error)
    monkeypatch.setattr(resources_module.torch.cuda, "is_available", lambda: False)
    capacity = ResourceMonitor().capacity()
    assert capacity.cpu_total_bytes is None
    assert capacity.cpu_limit_bytes == 4_096
    assert capacity.effective_cpu_bytes == 4_096


def test_resource_monitor_capacity_treats_cgroup_v2_max_as_unlimited(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cgroup_v2_root, _ = _configure_cgroup_paths(
        monkeypatch, tmp_path, "0::/workloads/demo\n"
    )
    limit_path = cgroup_v2_root / "workloads" / "demo" / "memory.max"
    limit_path.parent.mkdir(parents=True)
    limit_path.write_text("max\n", encoding="utf-8")
    monkeypatch.setattr(
        resources_module.os,
        "sysconf",
        lambda name: {"SC_PHYS_PAGES": 4, "SC_PAGE_SIZE": 4096}[name],
    )
    monkeypatch.setattr(resources_module.torch.cuda, "is_available", lambda: False)
    capacity = ResourceMonitor().capacity()
    assert capacity.cpu_limit_bytes is None
    assert capacity.effective_cpu_bytes == 16_384


@pytest.mark.parametrize("value", ["invalid\n", "0\n", "-1\n"])
def test_resource_monitor_capacity_ignores_invalid_cgroup_v2_limits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: str,
) -> None:
    cgroup_v2_root, _ = _configure_cgroup_paths(
        monkeypatch, tmp_path, "0::/workloads/demo\n"
    )
    limit_path = cgroup_v2_root / "workloads" / "demo" / "memory.max"
    limit_path.parent.mkdir(parents=True)
    limit_path.write_text(value, encoding="utf-8")
    monkeypatch.setattr(
        resources_module.os,
        "sysconf",
        lambda name: {"SC_PHYS_PAGES": 4, "SC_PAGE_SIZE": 4096}[name],
    )
    monkeypatch.setattr(resources_module.torch.cuda, "is_available", lambda: False)
    capacity = ResourceMonitor().capacity()
    assert capacity.cpu_limit_bytes is None
    assert capacity.effective_cpu_bytes == 16_384


def test_resource_monitor_capacity_reads_nested_cgroup_v1_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, cgroup_v1_root = _configure_cgroup_paths(
        monkeypatch, tmp_path, "5:cpu,memory:/containers/demo\n"
    )
    limit_path = cgroup_v1_root / "containers" / "demo" / "memory.limit_in_bytes"
    limit_path.parent.mkdir(parents=True)
    limit_path.write_text("8192\n", encoding="utf-8")
    monkeypatch.setattr(
        resources_module.os,
        "sysconf",
        lambda name: {"SC_PHYS_PAGES": 2, "SC_PAGE_SIZE": 4096}[name],
    )
    monkeypatch.setattr(resources_module.torch.cuda, "is_available", lambda: False)
    capacity = ResourceMonitor().capacity()
    assert capacity.cpu_total_bytes == 8_192
    assert capacity.cpu_limit_bytes == 8_192
    assert capacity.effective_cpu_bytes == 8_192


@pytest.mark.parametrize("value", ["invalid\n", "0\n", "-1\n", f"{1 << 62}\n"])
def test_resource_monitor_capacity_ignores_invalid_or_unlimited_v1_limits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: str,
) -> None:
    _, cgroup_v1_root = _configure_cgroup_paths(
        monkeypatch, tmp_path, "5:memory:/containers/demo\n"
    )
    limit_path = cgroup_v1_root / "containers" / "demo" / "memory.limit_in_bytes"
    limit_path.parent.mkdir(parents=True)
    limit_path.write_text(value, encoding="utf-8")
    monkeypatch.setattr(
        resources_module.os,
        "sysconf",
        lambda name: {"SC_PHYS_PAGES": 4, "SC_PAGE_SIZE": 4096}[name],
    )
    monkeypatch.setattr(resources_module.torch.cuda, "is_available", lambda: False)
    capacity = ResourceMonitor().capacity()
    assert capacity.cpu_limit_bytes is None
    assert capacity.effective_cpu_bytes == 16_384


def test_resource_monitor_capacity_handles_missing_cgroup_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_cgroup_paths(monkeypatch, tmp_path, "0::/missing\n")
    monkeypatch.setattr(
        resources_module.os,
        "sysconf",
        lambda name: {"SC_PHYS_PAGES": 4, "SC_PAGE_SIZE": 4096}[name],
    )
    monkeypatch.setattr(resources_module.torch.cuda, "is_available", lambda: False)
    capacity = ResourceMonitor().capacity()
    assert capacity.cpu_limit_bytes is None
    assert capacity.effective_cpu_bytes == 16_384


def test_resource_monitor_capacity_falls_back_to_v1_in_hybrid_cgroup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, cgroup_v1_root = _configure_cgroup_paths(
        monkeypatch,
        tmp_path,
        "0::/unified/demo\n5:memory:/legacy/demo\n",
    )
    limit_path = cgroup_v1_root / "legacy" / "demo" / "memory.limit_in_bytes"
    limit_path.parent.mkdir(parents=True)
    limit_path.write_text("4096\n", encoding="utf-8")
    monkeypatch.setattr(
        resources_module.os,
        "sysconf",
        lambda name: {"SC_PHYS_PAGES": 4, "SC_PAGE_SIZE": 4096}[name],
    )
    monkeypatch.setattr(resources_module.torch.cuda, "is_available", lambda: False)

    capacity = ResourceMonitor().capacity()

    assert capacity.cpu_limit_bytes == 4_096
    assert capacity.effective_cpu_bytes == 4_096


def test_resource_monitor_capacity_reads_v2_parent_limit_after_leaf_max(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cgroup_v2_root, _ = _configure_cgroup_paths(
        monkeypatch,
        tmp_path,
        "0::/workloads/demo/task\n",
    )
    leaf_path = cgroup_v2_root / "workloads" / "demo" / "task" / "memory.max"
    leaf_path.parent.mkdir(parents=True)
    leaf_path.write_text("max\n", encoding="utf-8")
    parent_path = cgroup_v2_root / "workloads" / "demo" / "memory.max"
    parent_path.write_text("8192\n", encoding="utf-8")
    (cgroup_v2_root / "memory.max").write_text("max\n", encoding="utf-8")
    monkeypatch.setattr(
        resources_module.os,
        "sysconf",
        lambda name: {"SC_PHYS_PAGES": 4, "SC_PAGE_SIZE": 4096}[name],
    )
    monkeypatch.setattr(resources_module.torch.cuda, "is_available", lambda: False)

    capacity = ResourceMonitor().capacity()

    assert capacity.cpu_limit_bytes == 8_192
    assert capacity.effective_cpu_bytes == 8_192


def test_resource_monitor_capacity_uses_lowest_v2_parent_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cgroup_v2_root, _ = _configure_cgroup_paths(
        monkeypatch,
        tmp_path,
        "0::/workloads/demo/task\n",
    )
    leaf_path = cgroup_v2_root / "workloads" / "demo" / "task" / "memory.max"
    leaf_path.parent.mkdir(parents=True)
    leaf_path.write_text("16384\n", encoding="utf-8")
    (cgroup_v2_root / "workloads" / "demo" / "memory.max").write_text(
        "8192\n",
        encoding="utf-8",
    )
    (cgroup_v2_root / "workloads" / "memory.max").write_text(
        "12288\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        resources_module.os,
        "sysconf",
        lambda name: {"SC_PHYS_PAGES": 8, "SC_PAGE_SIZE": 4096}[name],
    )
    monkeypatch.setattr(resources_module.torch.cuda, "is_available", lambda: False)

    capacity = ResourceMonitor().capacity()

    assert capacity.cpu_limit_bytes == 8_192
    assert capacity.effective_cpu_bytes == 8_192


def test_resource_monitor_capacity_prefers_v1_hierarchical_memory_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, cgroup_v1_root = _configure_cgroup_paths(
        monkeypatch,
        tmp_path,
        "5:memory:/containers/demo/task\n",
    )
    leaf_dir = cgroup_v1_root / "containers" / "demo" / "task"
    leaf_dir.mkdir(parents=True)
    (leaf_dir / "memory.stat").write_text(
        "cache 0\nhierarchical_memory_limit 4096\nrss 0\n",
        encoding="utf-8",
    )
    (leaf_dir / "memory.limit_in_bytes").write_text("16384\n", encoding="utf-8")
    monkeypatch.setattr(
        resources_module.os,
        "sysconf",
        lambda name: {"SC_PHYS_PAGES": 4, "SC_PAGE_SIZE": 4096}[name],
    )
    monkeypatch.setattr(resources_module.torch.cuda, "is_available", lambda: False)

    capacity = ResourceMonitor().capacity()

    assert capacity.cpu_limit_bytes == 4_096
    assert capacity.effective_cpu_bytes == 4_096


def test_resource_monitor_capacity_uses_lowest_v1_parent_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, cgroup_v1_root = _configure_cgroup_paths(
        monkeypatch,
        tmp_path,
        "5:memory:/containers/demo/task\n",
    )
    leaf_limit = (
        cgroup_v1_root / "containers" / "demo" / "task" / "memory.limit_in_bytes"
    )
    leaf_limit.parent.mkdir(parents=True)
    leaf_limit.write_text("16384\n", encoding="utf-8")
    (cgroup_v1_root / "containers" / "demo" / "memory.limit_in_bytes").write_text(
        "4096\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        resources_module.os,
        "sysconf",
        lambda name: {"SC_PHYS_PAGES": 4, "SC_PAGE_SIZE": 4096}[name],
    )
    monkeypatch.setattr(resources_module.torch.cuda, "is_available", lambda: False)

    capacity = ResourceMonitor().capacity()

    assert capacity.cpu_limit_bytes == 4_096
    assert capacity.effective_cpu_bytes == 4_096


@pytest.mark.parametrize("v2_value", ["max\n", "invalid\n", None])
def test_resource_monitor_capacity_uses_v1_after_non_finite_v2_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    v2_value: str | None,
) -> None:
    cgroup_v2_root, cgroup_v1_root = _configure_cgroup_paths(
        monkeypatch,
        tmp_path,
        "0::/unified/demo\n5:memory:/legacy/demo\n",
    )
    if v2_value is not None:
        v2_limit = cgroup_v2_root / "unified" / "demo" / "memory.max"
        v2_limit.parent.mkdir(parents=True)
        v2_limit.write_text(v2_value, encoding="utf-8")
    v1_limit = cgroup_v1_root / "legacy" / "demo" / "memory.limit_in_bytes"
    v1_limit.parent.mkdir(parents=True)
    v1_limit.write_text("4096\n", encoding="utf-8")
    monkeypatch.setattr(
        resources_module.os,
        "sysconf",
        lambda name: {"SC_PHYS_PAGES": 4, "SC_PAGE_SIZE": 4096}[name],
    )
    monkeypatch.setattr(resources_module.torch.cuda, "is_available", lambda: False)

    capacity = ResourceMonitor().capacity()

    assert capacity.cpu_limit_bytes == 4_096
    assert capacity.effective_cpu_bytes == 4_096


def test_resource_monitor_capacity_uses_lowest_limit_across_v2_and_v1(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cgroup_v2_root, cgroup_v1_root = _configure_cgroup_paths(
        monkeypatch,
        tmp_path,
        "0::/unified/demo\n5:memory:/legacy/demo\n",
    )
    v2_limit = cgroup_v2_root / "unified" / "demo" / "memory.max"
    v2_limit.parent.mkdir(parents=True)
    v2_limit.write_text("8192\n", encoding="utf-8")
    v1_limit = cgroup_v1_root / "legacy" / "demo" / "memory.limit_in_bytes"
    v1_limit.parent.mkdir(parents=True)
    v1_limit.write_text("4096\n", encoding="utf-8")
    monkeypatch.setattr(
        resources_module.os,
        "sysconf",
        lambda name: {"SC_PHYS_PAGES": 4, "SC_PAGE_SIZE": 4096}[name],
    )
    monkeypatch.setattr(resources_module.torch.cuda, "is_available", lambda: False)

    capacity = ResourceMonitor().capacity()

    assert capacity.cpu_limit_bytes == 4_096
    assert capacity.effective_cpu_bytes == 4_096
