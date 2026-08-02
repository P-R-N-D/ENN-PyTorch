# Development Runtime Workflow

This document describes the current supported composition for the bounded
single-node development runtime under `enn_torch_dev.runtime`.

The APIs in this document are active-development APIs. They are not exported by
the stable `enn_torch` namespace.

## Supported flow

```text
static footprints + reference BatchCost + ResourceCapacity
  -> optional recommend_initial_batch_budget
finite KVBatch pass source
  -> BudgetedBatcher
  -> RuntimeRetryRunner
  -> optional per-attempt PrePassAdmissionGate
  -> optional bounded reject split and child reassessment
  -> configured RuntimeStep
  -> optional fixed or pass-scoped capacity pressure assessment
  -> ConservativeRuntimeGovernor
  -> ConservativeRuntimeOrchestrator
  -> RuntimePassSummary
  -> RuntimePassHistory
  -> ConservativeRuntimeSession
  -> optional RuntimePassSourceFactory for fresh per-pass sources

completed StepResult.resource_samples
  -> optional ModelCostProbe
  -> optional ObservedCostCalibrator -> ObservedCostProfile
  -> optional assess_prepass_admission with current ResourceSample
```

`ConservativeRuntimeSession` connects existing components across multiple finite
passes. It does not replace the retry, governor, summary, or history contracts.

## Minimal composition

```python
from enn_torch_dev.runtime import (
    BatchBudget,
    ConservativeRuntimeGovernor,
    ConservativeRuntimeOrchestrator,
    ConservativeRuntimeSession,
    GovernorPolicy,
    RetryPolicy,
    RuntimePassHistory,
)

runtime_step = ...  # object providing run(KVBatch) -> StepResult

governor = ConservativeRuntimeGovernor(
    BatchBudget(max_items=8),
    policy=GovernorPolicy(
        shrink_factor=0.5,
        grow_factor=2.0,
        grow_after_successes=3,
        min_items=1,
        max_items=32,
    ),
)

orchestrator = ConservativeRuntimeOrchestrator(
    runtime_step,
    governor,
    retry_policy=RetryPolicy(
        max_retry_depth=3,
        min_items=1,
        split_factor=2,
    ),
)

history = RuntimePassHistory(max_records=10)

session = ConservativeRuntimeSession(
    orchestrator,
    history,
    max_passes=20,
)

for record in session.run_passes(pass_sources):
    print(record.pass_index, record.pass_summary)
```

## Optional initial budget recommendation

Use `recommend_initial_batch_budget(...)` when the first governor budget should be
derived from known static facts instead of supplied manually:

```python
from enn_torch_dev.runtime import (
    InitialBatchBudgetPolicy,
    ModelFootprint,
    OptimizerFootprint,
    ResourceCapacity,
    recommend_initial_batch_budget,
)

recommendation = recommend_initial_batch_budget(
    ResourceCapacity(
        cpu_total_bytes=host_bytes,
        cuda_total_bytes=device_bytes,
        cuda_device_index=0,
    ),
    reference_batch_cost,
    reference_device_bytes_by_device={
        "cuda:0": reference_batch_cost.device_bytes or 0,
    },
    model_footprint=ModelFootprint.from_module(model),
    optimizer_footprint=OptimizerFootprint.from_optimizer(optimizer),
    policy=InitialBatchBudgetPolicy(
        max_items=32,
        fallback_max_items=1,
    ),
)

governor = ConservativeRuntimeGovernor(recommendation.recommended_budget)
```

The helper is deterministic and side-effect free. It does not execute the model,
consume a source, mutate governor or history state, or decide whether a pass is
admissible. Model and optimizer footprints retain device-resolved byte maps so
CPU and the configured CUDA device are accounted independently. Positive
reference device bytes likewise require an explicit mapping to the matching
`cuda:<index>`; aggregate non-CPU cost is not assigned to an arbitrary CUDA
capacity. Zero byte totals remain non-limiting without an item count. The result
preserves the original capacity, reference cost, resolved policy, and normalized
reference device mapping for audit and reproduction. Unknown capacity or
positive per-item cost remains unknown; an explicit `fallback_max_items` is
required when a finite limit cannot otherwise be derived. See
[`dev_initial_batch_budget.md`](dev_initial_batch_budget.md) for formulas and
boundaries.

## Optional observed-cost calibration

Use `ObservedCostCalibrator` when completed runtime observations should be reduced
to a bounded per-item cost envelope for later inspection or a separately reviewed
admission layer:

```python
from enn_torch_dev.runtime import (
    ModelCostProbe,
    ObservedCostCalibrationPolicy,
    ObservedCostCalibrator,
)

cost_probe = ModelCostProbe()
calibrator = ObservedCostCalibrator(
    ObservedCostCalibrationPolicy(
        min_successful_samples=3,
        max_phase_pairs=16,
        expected_cuda_device_index=0,
    )
)

for pass_result in completed_pass_results:
    for step_result in pass_result.results:
        calibrator.observe(cost_probe.estimate_step(step_result))

observed_profile = calibrator.profile()
```

Only successful positive-batch observations contribute numeric values. Faults and
zero-batch successes are counted but ignored. Available deltas are converted to
per-item costs with ceiling division, negative deltas are clamped to zero, and
the maximum value observed for each total and adjacent-phase metric is retained.
Unknown values remain distinct from observed zero.

One profile accepts CUDA metrics from only one concrete device. The policy may
bind that device before the first observation. Phase-pair state is bounded by
`max_phase_pairs`. The calibrator retains scalar accumulators and phase names,
not raw `ModelCost`, `StepResult`, `ResourceSample`, tensor, store, or loss
objects.

Calibration is explicit and side-effect free. It does not execute the model,
consume a source, mutate a governor, persist a profile, or decide whether a future
pass is admissible. See
[`dev_observed_cost_calibration.md`](dev_observed_cost_calibration.md) for the
full contract.

## Optional pre-pass admission assessment

Use `assess_prepass_admission(...)` to compare one candidate batch size against
an execution-immediate resource baseline and an observed per-item cost profile:

```python
from enn_torch_dev.runtime import (
    PrePassAdmissionPolicy,
    PrePassAdmissionStatus,
    assess_prepass_admission,
)

assessment = assess_prepass_admission(
    capacity,
    monitor.sample("before_admission"),
    observed_profile,
    batch_size=candidate_batch_size,
    policy=PrePassAdmissionPolicy(
        host_utilization_ratio=0.9,
        device_utilization_ratio=0.9,
        min_profile_samples=3,
    ),
)

if assessment.status is PrePassAdmissionStatus.REJECT:
    inspect(assessment.rejected_dimensions)
```

The assessor returns `ADMIT`, `REJECT`, or `UNKNOWN`; it does not execute or
block the candidate. CPU RSS and CUDA allocated/reserved projections are reported
separately, with `REJECT` taking precedence over `UNKNOWN`. Known zero cost is
non-limiting, while missing capacity, current usage, or per-item cost remains
unknown. CUDA-bearing capacity, baseline, and profile provenance must identify the
same concrete device. See
[`dev_prepass_admission.md`](dev_prepass_admission.md) for formulas and boundaries.

## Optional pre-pass admission gate

Use the gate only when a completed observed profile should be enforced immediately
before every runtime execution attempt:

```python
from enn_torch_dev.runtime import (
    AdmissionSplitPolicy,
    AdmissionUnknownAction,
    ConservativeRuntimeOrchestrator,
    PrePassAdmissionPolicy,
    ResourceMonitor,
)

monitor = ResourceMonitor(cuda_device=0)
orchestrator = ConservativeRuntimeOrchestrator(
    runtime_step,
    governor,
    retry_policy=retry_policy,
    resource_capacity_provider=monitor,
    admission_profile=observed_profile,
    admission_sample_provider=monitor,
    admission_policy=PrePassAdmissionPolicy(min_profile_samples=3),
    admission_unknown_action=AdmissionUnknownAction.BLOCK,
    admission_split_policy=AdmissionSplitPolicy(
        max_split_depth=3,
        min_items=1,
        max_split_parts=16,
    ),
)
```

The gate is disabled unless `admission_profile` is configured. When enabled, it
requires an admission sample provider and either fixed or provider-backed
capacity. Capacity is resolved once before the pass source is consumed and stays
fixed for that pass. The sample provider is called once with `"before_admission"` before every
original, admission child, or OOM retry-split candidate assessment.

`REJECT` always raises `PrePassAdmissionBlocked`. `UNKNOWN` also blocks by default;
`AdmissionUnknownAction.ALLOW` permits only unknown assessments and never permits
a rejection. The exception custom payload stores the immutable assessment, while
its ordinary Python traceback may still reference frame-local runtime objects.
Admission blocking is not a `StepStatus` because no runtime step completed.

A completed `RuntimePassResult` records attempt-ordered
`admission_assessments`. Retry-consumed OOM attempts may therefore make this tuple
longer than the final result tuple. If a later candidate blocks, earlier
candidates in the same pass may already have executed, but no pass result is
created and the governor is not updated. When `admission_split_policy` is
configured, only a `REJECT` with a positive smaller `max_admissible_items` may be
recovered by bounded identity-preserving split and fresh child assessments.
`UNKNOWN`, invalid limits, exhausted depth, or excessive parts remain terminal.
Recovery applies only to the orchestrator wrapper's private pre-execution request;
a public `PrePassAdmissionBlocked` from a generic runtime step is terminal.
See [`dev_prepass_admission_gate.md`](dev_prepass_admission_gate.md) and
[`dev_prepass_admission_split.md`](dev_prepass_admission_split.md).

## Optional pressure-aware composition

The governor pressure guard becomes operational when the caller supplies either
a fixed `ResourceCapacity` or a pass-scoped `ResourceCapacityProvider`.
`ResourceMonitor` already implements the provider contract, so one monitor can
produce both runtime samples and pass-start capacity snapshots:

```python
from enn_torch_dev.runtime import ResourceMonitor, RuntimeStep

governor = ConservativeRuntimeGovernor(
    BatchBudget(max_items=8),
    policy=GovernorPolicy(
        grow_after_successes=3,
        max_pressure_ratio_for_growth=0.8,
    ),
)

monitor = ResourceMonitor(cuda_device=0)
runtime_step = RuntimeStep(..., resource_monitor=monitor)

orchestrator = ConservativeRuntimeOrchestrator(
    runtime_step,
    governor,
    retry_policy=RetryPolicy(max_retry_depth=3, min_items=1, split_factor=2),
    resource_capacity_provider=monitor,
)
```

The provider is called exactly once at each pass start, before the pass source is
consumed. Its result remains fixed for that pass, including retry and split
attempts. A caller may still use `resource_capacity=...` for a fixed snapshot, but
fixed capacity and a provider cannot be configured together.

If neither fixed capacity nor a provider is supplied, the orchestrator passes no
pressure summary and the existing governor contract applies. Pressure may suppress
success-driven growth when the opt-in guard is enabled. Separately configured
sustained high pressure may shrink only the next-pass budget after its configured
pass count is reached; a single non-OOM pressure sample cannot shrink a budget.

## Factory composition

Use a `RuntimePassSourceFactory` when each pass needs a newly constructed
one-shot loader or generator:

```python
from enn_torch_dev.runtime import RuntimePassSourceFactory


class PassFactory:
    def create_pass_source(self, pass_index: int):
        return build_finite_pass_source(pass_index)


for record in session.run_factory(PassFactory()):
    print(record.pass_index, record.pass_summary)
```

`run_factory(...)` is lazy and uses the same `max_passes` bound as
`run_passes(...)`. It calls the factory at most once per yielded record and does
not call it for an extra pass after the limit. Factory exceptions propagate, and
a pass whose source was not created is not appended to history.

## Source contract

`pass_sources` is an outer iterable of inner pass sources.

- The outer session is bounded by `max_passes`.
- Every inner source must be a finite iterable of `KVBatch`.
- The session does not fetch an extra outer source after reaching `max_passes`.
- Each `next()` call on the session iterator executes at most one finite pass.

`run_passes(...)` does not recreate or replay a source. `run_factory(...)` asks
caller-defined code to construct a fresh finite source for each pass, but it does
not cache sources or replay a consumed iterator.

## OOM and budget behavior

`RuntimeRetryRunner` may split a retryable OOM batch into smaller subbatches.
`ConservativeRuntimeOrchestrator` reports a retry-recovered OOM to the governor
even when the final yielded results are successful.

When the opt-in admission gate is enabled, the original attempt and every retry
subbatch are sampled and assessed independently before execution. The admission
wrapper forwards the configured runtime step's optimizer attribute so the
existing training-time retry restriction is unchanged. Admission split depth and
OOM retry depth are independent: on the trusted admission-wrapper path, splitting
is pre-execution and may run with an optimizer, while post-execution OOM retry keeps its existing restriction.
Recovered admission rejection does not directly affect governor feedback.

The conservative governor then:

- shrinks configured budget fields after a yielded or retry-recovered OOM;
- keeps the current budget after non-OOM faults;
- grows configured fields only after the configured clean-success threshold;
- optionally suppresses success growth when an explicit pressure summary is
  missing or reaches the configured growth limit;
- optionally shrinks the next-pass budget only after a configured sustained
  high-pressure streak, tracking CPU and CUDA persistence independently and
  selecting host bytes for CPU pressure and device bytes for CUDA pressure;
- allows CPU and CUDA shrink thresholds, required pass counts, and sustained-
  pressure shrink factors to override the common policy independently;
- applies each dimension's effective factor to its matching byte budget;
- falls back to `max_items` only when no matching triggered byte budget is
  configured, using the triggered dimension's factor or the smaller factor when
  both dimensions share the fallback;
- keeps yielded and retry-recovered OOM shrink on the common `shrink_factor`;
- preserves an incomplete streak for one dimension when only the other dimension
  reaches its effective shrink threshold and required pass count;
- applies configured minimum and maximum bounds.

The next pass uses the governor's current budget.

## Inspection and retention

A yielded `RuntimeSessionRecord` contains the finite pass result for immediate
caller inspection. The caller controls how long that record remains alive.

The session itself does not retain prior pass results after the generator resumes.
Longer-lived in-memory retention is limited to lightweight `RuntimePassSummary`
objects in `RuntimePassHistory`, which requires a positive `max_records` bound.
Pass summaries expose scalar pressure ratios, growth-suppression decisions, and
sustained-pressure shrink feedback. They also expose structured high/triggered
dimensions, selected adjustment fields, field-level applied factors, and actual
changed fields without requiring `decision_reason` parsing. History aggregation
uses that structured provenance to count retained CPU/CUDA high and trigger
passes, adjustment attempts, full no-ops, triggers without matching budgets, and
actual host/device/items shrink passes. CPU and CUDA can each contribute for one
pass, while attempt/no-op/trigger-without-budget counts increment at most once per
pass; a partial change is not a full no-op. Existing pressure-assessed,
pressure-suppressed, actual pressure-shrink, and peak-ratio aggregates remain. All
counts are recomputed only within the currently retained summary window, and OOM
status or ratios never substitute for empty structured provenance. Each pass
summary also records the scalar capacity used for normalization. Raw
`ResourceSample` records are not retained by summary or history.

## Fault and exception semantics

A `StepStatus` fault is a completed runtime result. It does not automatically stop
the session. The governor observes it according to its documented policy.

Provider, source-iteration, and pass-execution exceptions are not suppressed. A
provider failure occurs before source consumption and before governor updates.
The failing pass is not added to history when execution or summary construction
fails before history append. Previously completed history records remain intact.

`PrePassAdmissionBlocked` is an execution-gate exception rather than a completed
runtime result. The blocked candidate is not executed, later candidates are not
consumed, and the governor is not updated. Candidates completed earlier in the
same pass are not rolled back and no partial `RuntimePassResult` is returned.

## Safety boundary

This workflow is intended for bounded, single-node development execution.

It does not provide:

- persistent logging, JSONL, or CSV export;
- dashboards or telemetry backends;
- checkpoint/resume;
- automatic source replay or source caching;
- distributed execution or aggregation;
- AutoGovernor or learned tuning;
- automatic `ResourceMonitor` creation;
- mid-pass capacity refresh;
- proof that an initial recommendation is safe for unobserved activation or allocator costs;
- persistent observed-cost profile storage;
- admission-driven skip, replay, rollback, or heuristic split sizes;
- admission-based governor, summary, or history feedback;
- automatic use of an `ObservedCostProfile` for governor updates;
- learned field weights;
- stable `enn_torch` API exposure.

Use small synthetic inputs for baseline validation. Do not use this workflow as an
unbounded production streaming runner.

## Validation

```bash
python -m pytest enn_torch_dev/debug/runtime/test_prepass_admission_gate.py -q
python -m pytest enn_torch_dev/debug/runtime/test_prepass_admission_split.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_capacity_provider.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_source_factory.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```
