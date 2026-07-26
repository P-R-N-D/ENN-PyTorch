# Development Observed Cost Calibration

This document describes the development-only observed runtime-cost calibration
boundary under `enn_torch_dev.runtime`.

## Goal

The initial batch-budget recommender uses static capacity, batch cost, and fixed
model/optimizer footprints. It intentionally cannot account for execution-time
activation growth, allocator reservation behavior, or phase-specific peaks.

This slice turns already observed `ModelCost` records into one bounded,
deterministic cost envelope:

```text
StepResult.resource_samples
  -> ModelCostProbe
  -> ModelCost
  -> ObservedCostCalibrator.observe(...)
  -> ObservedCostProfile
```

The calibrator does not execute a model or consume a source. Callers decide when
to probe a completed `StepResult` and submit its `ModelCost`.

## Accepted observations

Only `StepStatus.SUCCESS` observations contribute numeric values. OOM,
non-finite, data, and runtime faults are ignored and counted by status for
diagnostics. A successful observation with `batch_size == 0` is ignored because
no per-item cost can be derived. Negative batch sizes and inconsistent
`batch_size` / `row_count` values are rejected.

For each available byte delta:

```text
normalized delta = max(observed delta, 0)
bytes per item = ceil(normalized delta / batch_size)
profile envelope = max(bytes per item over accepted samples)
```

Negative deltas are clamped to zero rather than treated as memory credits. The
metric profile records unknown, observed-zero, and negative-clamped counts
separately, so `None` remains distinguishable from a known zero envelope.

## CUDA provenance

`ModelCost` retains an append-only `cuda_device_index` resolved by
`ModelCostProbe` only when every CUDA-bearing resource sample supplies the same
bool-excluding, non-negative integer index. Any missing or invalid index leaves
the model cost unbound. CUDA deltas likewise require both endpoints to supply
the same concrete index; `None == None` is not a device match, and the current
CUDA device is never inferred. Known CUDA total or phase metrics require this
concrete provenance.

One `ObservedCostCalibrator` profile may contain CUDA observations from only one
device. `ObservedCostCalibrationPolicy.expected_cuda_device_index` can bind the
profile to a specific device before the first observation. Missing or mismatched
CUDA provenance is rejected; observations are never assigned to a device by
assumption.

## Bounded state

The calibrator stores only scalar counters, maxima, device provenance, and one
accumulator per distinct adjacent phase pair. It does not retain `ModelCost`,
`StepResult`, `ResourceSample`, tensors, stores, or losses.

`ObservedCostCalibrationPolicy.max_phase_pairs` bounds the number of retained
phase-pair accumulators. An observation that would exceed that bound is rejected
before its numeric values are applied.

## Result contract

`ObservedCostProfile` includes:

- the applied calibration policy;
- total, successful, ignored, and rejected observation counts;
- ignored fault counts and zero-batch count;
- observed batch-size range;
- concrete CUDA device provenance when CUDA metrics were accepted;
- total CPU RSS and CUDA allocated/reserved/peak per-item envelopes;
- per-phase metric envelopes in deterministic phase-pair order;
- known, unknown, observed-zero, and negative-clamped counts for every metric.

`profile()` raises until `min_successful_samples` has been reached.

## Safety boundary

This profile is evidence from previous executions, not proof that a future pass
is admissible. It does not account for workload distribution shifts, new graph
paths, external allocations, concurrent CUDA activity, or capacity changes.
Pre-pass admission, automatic governor wiring, persistence, learned tuning, and
multi-device profile merging remain outside this slice.

## Validation

```bash
python -m pytest enn_torch_dev/debug/runtime/test_observed_cost_calibration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_model_cost_probe.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```
