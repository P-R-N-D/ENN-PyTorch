# Runtime Resource Pressure

This document describes the resource-capacity and pressure-assessment boundary
for the active-development runtime under `enn_torch_dev.runtime`.

## Goal

`ResourceMonitor` and `ResourceSample` already expose observed process and CUDA
memory usage. This slice adds the missing denominator and a pure interpretation
layer:

```text
ResourceMonitor.capacity()
  -> ResourceCapacity

Iterable[ResourceSample] + ResourceCapacity
  -> assess_resource_pressure(...)
  -> ResourcePressureSummary
```

The assessment layer observes and summarizes. It does not choose a `BatchBudget`,
retry work, or execute a model. A separate opt-in governor policy may consume the
summary only to suppress success-driven budget growth.

## Public Objects

`enn_torch_dev.runtime` exports:

- `ResourceCapacity`
- `ResourcePressureSummary`
- `assess_resource_pressure`

The stable `enn_torch` namespace is unchanged.

## Capacity Contract

`ResourceCapacity` records:

- total physical CPU memory bytes when available;
- cgroup CPU memory limit bytes when available;
- total CUDA device memory bytes when available;
- the CUDA device index associated with that capacity.

Capacity values are `None` or positive integers. CUDA total bytes and device
index must either both be configured or both be `None`.

`ResourceCapacity.effective_cpu_bytes` is the smaller of the known physical CPU
capacity and cgroup memory limit. If only one is known, that value is used. If
neither is known, the effective CPU capacity is `None`.

`ResourceMonitor.capacity()` uses:

- `SC_PHYS_PAGES * SC_PAGE_SIZE` for physical CPU memory;
- cgroup v2 `memory.max` candidates from the process cgroup leaf through
  the unified hierarchy root;
- cgroup v1 `memory.stat` `hierarchical_memory_limit` when available,
  otherwise `memory.limit_in_bytes` candidates from the process memory
  controller leaf through the v1 memory hierarchy root;
- `torch.cuda.get_device_properties(index).total_memory` for CUDA capacity.

The monitor resolves nested process cgroup paths from `/proc/self/cgroup`.
When both cgroup v2 and v1 memory memberships are present, both hierarchies are
evaluated independently and the smallest finite candidate is used as
`cpu_limit_bytes`. `memory.max=max`, invalid/non-positive limits, and cgroup v1
unlimited sentinel values are ignored as individual candidates while parent or
other-hierarchy discovery continues. Missing files and lookup failures return
`None`; they do not become runtime faults.

## Pressure Contract

`assess_resource_pressure(samples, capacity)` streams `ResourceSample` objects
once and returns peak ratios for:

- process CPU RSS;
- CUDA allocated memory;
- CUDA reserved memory;
- CUDA max allocated memory;
- CUDA max reserved memory.

Each ratio is:

```text
CPU: observed RSS bytes / effective CPU capacity bytes
CUDA: observed bytes / CUDA total capacity bytes
```

CPU pressure uses `effective_cpu_bytes`, so a container limit smaller than host
physical memory becomes the denominator. CUDA pressure semantics are unchanged.

Missing capacity or observation values produce `None`, not zero. Ratios are not
clamped to `1.0`; values above one remain visible so inconsistent measurements
or overhead are not hidden. Ratio records reject negative, NaN, and infinite
values.

When CUDA values are present and CUDA capacity is configured, sample and
capacity device indices must match. Device-mismatched samples are rejected
instead of being silently combined.

The function returns only scalar ratios and does not retain the input sample
objects. `ResourcePressureSummary.max_observed_ratio` returns the highest known
ratio, or `None` when every ratio is unknown.

## Optional Governor Growth Guard

`GovernorPolicy.max_pressure_ratio_for_growth` is `None` by default, preserving
the existing governor behavior. When configured, callers may pass a
`ResourcePressureSummary` to `ConservativeRuntimeGovernor.observe_results(...)`.

- A known maximum ratio below the configured limit allows normal success-streak
  accumulation and growth.
- A ratio equal to or above the limit suppresses growth and resets the success
  streak.
- A missing summary or an all-unknown summary also suppresses growth.
- OOM and retry-recovered OOM retain priority and continue to shrink the budget.
- Pressure never directly shrinks a budget in this slice.

The governor does not build the summary itself. A
`ConservativeRuntimeOrchestrator` configured with an explicit fixed
`ResourceCapacity` can assess all raw-attempt samples for a finite pass and pass
the summary to the governor.

## Example

```python
from enn_torch_dev.runtime import ResourceMonitor, assess_resource_pressure

monitor = ResourceMonitor(cuda_device=0)
capacity = monitor.capacity()
samples = (
    monitor.sample("before_step"),
    monitor.sample("after_step"),
)
pressure = assess_resource_pressure(samples, capacity)
print(pressure.peak_cuda_reserved_ratio)
```

## Safety Boundary

Pressure summaries are observational unless an explicit governor growth guard is
configured. Even with that guard, a high or unavailable ratio does not:

- shrink a runtime budget;
- trigger retry;
- reserve memory;
- move tensors;
- change precision;
- stop a session.

The only supported feedback in this slice is opt-in suppression of
success-driven growth. Orchestration may produce the summary only when a caller
provides an explicit fixed capacity; automatic monitoring and capacity discovery
remain outside this contract.

## Out of Scope

- Pressure-triggered budget shrink.
- Automatic capacity sampling or pressure-summary construction in the governor.
- Automatic `ResourceMonitor` creation or capacity refresh in orchestration.
- Real-time free-memory reservation.
- Memory admission control.
- Persistent telemetry or dashboards.
- Checkpoint/resume.
- Distributed aggregation.
- Windows Job Object or non-cgroup container limits.
- Slurm memory-limit discovery.
- Kubernetes API integration.
- Hardware presets or model-specific tuning.

## Test Commands

```bash
python -m pytest enn_torch_dev/debug/runtime/test_runtime_pressure.py -q
python -m pytest enn_torch_dev/debug/runtime/test_resource_monitor.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_governor.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```
