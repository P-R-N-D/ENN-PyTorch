# Development Initial Batch Budget Recommendation

This document describes the development-only initial budget recommendation helper
under `enn_torch_dev.runtime`.

## Goal

The helper computes one conservative starting `BatchBudget` before a runtime
session begins:

```text
ResourceCapacity
+ model and optimizer tensor footprints by device
+ reference BatchCost and explicit device-byte provenance
+ explicit utilization, reserve, and item limits
  -> recommend_initial_batch_budget(...)
  -> BatchBudgetRecommendation
```

It is a pure recommendation boundary. It does not execute a model, consume a
source, mutate governor or history state, admit a pass, retry a batch, or persist
calibration data.

## Device provenance

`ModelFootprint` and `OptimizerFootprint` retain append-only `bytes_by_device`
maps. The reference batch's aggregate `BatchCost.device_bytes` is accompanied by
the separate `reference_device_bytes_by_device` recommender input. The mapping
must sum exactly to the aggregate and bind every non-zero byte to the configured
`cuda:<index>` capacity. Bare `cuda`, a different CUDA index, MPS, XPU, and
multiple non-zero devices are rejected; aggregate non-CPU bytes are never
assigned to an arbitrary CUDA capacity.

The recommender applies the same concrete-device rule to reference costs and
static footprints. It uses only:

- `cpu` bytes against `ResourceCapacity.effective_cpu_bytes`;
- the exact matching `cuda:<index>` bytes against the configured CUDA capacity.

Bare `cuda` footprint keys are rejected because they do not identify an index;
they are never assigned to the currently configured CUDA device. Non-zero
footprint bytes on unsupported or different CUDA devices are also rejected. A
manually constructed non-empty footprint without device provenance is rejected
rather than assigned to CPU or CUDA by assumption.

## Calculation

For each known dimension:

```text
usable bytes
= floor(capacity bytes * utilization ratio)
- explicit reserve bytes
- fixed model/optimizer bytes
```

Reference `BatchCost` byte totals are converted to per-item costs with ceiling
division:

```text
bytes per item = ceil(reference bytes / reference num_items)
item limit = floor(usable bytes / bytes per item)
```

The final item recommendation is the minimum of all known CPU, CUDA, and policy
limits. A limit below `min_items` is an error; it is never clamped upward.

`None` means unknown and is never treated as zero. A zero total byte cost is
explicitly non-limiting even when the reference item count is `None` or zero.
Positive totals with an unavailable item count have unknown per-item cost.
Unknown dimensions require `fallback_max_items`; otherwise the helper raises
`BatchBudgetRecommendationError`.

## Result contract

`BatchBudgetRecommendation` includes:

- the recommended `BatchBudget`;
- the original `ResourceCapacity`, reference `BatchCost`, and resolved policy;
- sorted immutable reference device-byte provenance;
- limiting dimensions;
- capacity, fixed-footprint, usable-byte, per-item, and item-limit values;
- whether fallback was used;
- deterministic warning strings.

The preserved inputs make utilization, reserve, original totals, and physical or
cgroup CPU-capacity provenance auditable from the result. The returned byte
budgets represent variable batch headroom after utilization,
reserve, and known static tensor footprints have been removed.

## Known limits

The static recommendation does not estimate unobserved activations, allocator
fragmentation, framework overhead, asynchronous transfer buffers, or future
optimizer state that has not yet been materialized. It is not proof that a pass
is admissible. A separate pre-pass admission boundary and observed-cost
calibration remain outside this slice.

## Validation

```bash
python -m pytest enn_torch_dev/debug/runtime/test_budget_recommendation.py -q
python -m pytest enn_torch_dev/debug/runtime/test_model_footprint.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```
