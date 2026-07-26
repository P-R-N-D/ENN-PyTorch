# Development Pre-Pass Admission Assessment

This document describes the development-only, side-effect-free pre-pass memory
admission boundary under `enn_torch_dev.runtime`.

## Goal

The static initial-budget recommendation accounts for capacity, a reference
`BatchCost`, and model/optimizer footprints. Observed-cost calibration reduces
completed runtime measurements to conservative per-item envelopes. This slice
combines the current resource baseline with those observed envelopes to assess a
single candidate batch size without executing it:

```text
ResourceCapacity
+ execution-immediate ResourceSample
+ ObservedCostProfile
+ candidate batch size
  -> assess_prepass_admission(...)
  -> ADMIT / REJECT / UNKNOWN
```

The assessor does not consume a source, split a batch, run a model, invoke retry,
or mutate governor, history, session, or calibration state.

## Inputs

`assess_prepass_admission(...)` accepts:

- one `ResourceCapacity`;
- one execution-immediate baseline `ResourceSample`;
- one immutable `ObservedCostProfile`;
- a positive candidate `batch_size`;
- an optional `PrePassAdmissionPolicy`.

`BatchCost` is not an input. `BudgetedBatcher` remains responsible for static
payload and item-budget enforcement. Admission adds current RSS/CUDA usage and
observed execution-time memory growth instead of repeating static batching.

## Policy

`PrePassAdmissionPolicy` provides:

- host and device utilization ratios in `(0, 1]`;
- non-negative host and device reserve bytes;
- a positive minimum accepted calibration-sample count.

Usable bytes are calculated in this order:

```text
usable = max(0, floor(capacity * utilization ratio) - reserve)
```

## Projection

CPU projection uses current RSS plus the calibrated CPU RSS delta envelope:

```text
projected CPU RSS
= baseline cpu_rss_bytes
+ profile cpu_rss.max_bytes_per_item * batch_size
```

CUDA allocated and reserved projections use current allocated/reserved bytes.
For each dimension, the larger known direct or peak-delta envelope is used:

```text
allocated increment per item
= max(cuda_allocated, cuda_max_allocated)

reserved increment per item
= max(cuda_reserved, cuda_max_reserved)
```

Baseline `cuda_max_*` values are not added to the projection because they may
contain a prior peak. They are used only to identify whether the baseline sample
is CUDA-bearing and therefore requires concrete device provenance.

Known CUDA metrics in a phase profile also establish that CUDA is relevant and
therefore trigger CUDA capacity and provenance validation. Phase metrics are not
summed or otherwise used as total projection increments. If the current CUDA
usage or total CUDA increment remains unknown, the applicable CUDA dimension is
`UNKNOWN`; phase-only evidence is never downgraded to non-applicable `ADMIT`.

## Status rules

Each applicable dimension is assessed independently in deterministic order:

```text
cpu_rss
cuda_allocated
cuda_reserved
```

- current or projected usage above usable capacity: `REJECT`;
- no overage, but required capacity/current/profile evidence is unavailable:
  `UNKNOWN`;
- all required values are known and within usable capacity: `ADMIT`.

Overall precedence is:

```text
REJECT > UNKNOWN > ADMIT
```

A known zero per-item cost is non-limiting. `None` remains unknown. The assessor
never replaces missing evidence with zero. If the profile has fewer successful
samples than `min_profile_samples`, its per-item costs are treated as unknown,
although a baseline that already exceeds usable capacity still rejects.

## CUDA provenance

When CUDA is relevant, the following concrete indices must agree:

- `ResourceCapacity.cuda_device_index`;
- `ResourceSample.cuda_device_index` for CUDA-bearing baseline values;
- `ObservedCostProfile.cuda_device_index` for known CUDA envelopes.

Missing, invalid, or mismatched CUDA provenance raises
`PrePassAdmissionError`. The assessor does not infer the current CUDA device,
normalize bare device names, or merge devices.

## Result contract

`PrePassAdmissionAssessment` includes:

- the overall status and candidate batch size;
- the applied policy and profile sample count;
- fixed-order structured dimension calculations;
- rejected and unknown dimension names;
- the minimum finite known item limit across applicable dimensions;
- deterministic warning strings.

A dimension records capacity, usable bytes, current bytes, per-item increment,
projected bytes, headroom, item limit, and a structured reason. A `None` overall
item limit means either no applicable dimension supplies a finite limit or all
known limiting dimensions are unbounded; `unknown_dimensions` distinguishes the
unknown case.

## Safety boundary

This assessment is a pure calculation, not an execution gate. It does not prove
that unobserved graph paths, allocator fragmentation, concurrent allocations, or
workload distribution shifts are safe. Orchestrator wiring, fail-open/fail-closed
handling, automatic split/skip behavior, persistence, multi-GPU, and distributed
admission remain outside this slice.

## Validation

```bash
python -m pytest enn_torch_dev/debug/runtime/test_prepass_admission.py -q
python -m pytest enn_torch_dev/debug/runtime/test_observed_cost_calibration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```
