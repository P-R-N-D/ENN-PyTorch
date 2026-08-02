# Development Opt-In Pre-Pass Admission Gate

This document describes the development-only enforcement layer that applies a
completed `ObservedCostProfile` immediately before each orchestrated runtime
execution attempt.

## Goal

The pure `assess_prepass_admission(...)` helper returns structured evidence but
does not block execution. `PrePassAdmissionGate` turns that evidence into an
explicit opt-in execution boundary:

```text
pass-scoped ResourceCapacity
+ fixed ObservedCostProfile
+ execution-immediate ResourceSample
+ candidate batch size
  -> PrePassAdmissionGate.check(...)
  -> ADMIT: execute
  -> UNKNOWN: block by default, or execute only with explicit ALLOW
  -> REJECT: always block
```

The gate itself does not choose a smaller batch, skip a candidate, replay a
source, change a governor, update calibration, or persist state. A separately
configured bounded split policy may recover a rejected candidate outside the gate.

## Public development API

The gate slice adds the following under `enn_torch_dev.runtime`:

- `ResourceSampleProvider`;
- `AdmissionUnknownAction`;
- `AdmissionSplitPolicy`;
- `PrePassAdmissionBlocked`;
- `PrePassAdmissionGate`.

These APIs are not exported by stable `enn_torch`.

`ResourceSampleProvider` is a structural protocol:

```python
class ResourceSampleProvider(Protocol):
    def sample(self, phase: str) -> ResourceSample:
        ...
```

`ResourceMonitor` satisfies this protocol. Tests may use deterministic synthetic
providers without allocating CUDA memory.

## Gate behavior

`PrePassAdmissionGate.check(batch_size)` validates the positive batch size before
sampling, calls `sample("before_admission")` exactly once, validates the returned
`ResourceSample`, and delegates all capacity, projection, sample-floor, and CUDA
provenance calculations to `assess_prepass_admission(...)`.

Status handling is fixed:

```text
ADMIT                 -> return assessment
UNKNOWN + BLOCK       -> raise PrePassAdmissionBlocked
UNKNOWN + ALLOW       -> return assessment
REJECT                -> raise PrePassAdmissionBlocked
```

`AdmissionUnknownAction.ALLOW` never overrides `REJECT`. The default is `BLOCK`.

`PrePassAdmissionBlocked` stores only the immutable
`PrePassAdmissionAssessment` in its custom payload and custom attributes. As with
ordinary Python exceptions, however, its `__traceback__` may reference execution
frames whose local variables include the candidate `KVBatch`, baseline
`ResourceSample`, source, or runtime wrappers. The exception object's transitive
object graph is therefore not limited to the assessment. Admission blocking is
not represented as `StepStatus` because no runtime step completed.

Callers that need long-lived diagnostics should store `exc.assessment` separately
and use a lightweight standard-library traceback representation when textual
traceback details are required. They should not cache the exception object or its
traceback for long periods, especially in memory-sensitive runtimes. The gate does
not clear tracebacks or provide transactional rollback.

## Orchestration placement

The opt-in orchestration path is:

```text
BudgetedBatcher
  -> RuntimeRetryRunner
       -> admission-aware RuntimeStep wrapper
            -> sample and assess
            -> OOM-tracking RuntimeStep wrapper
                 -> configured RuntimeStep
```

This placement is intentional. `RuntimeRetryRunner` calls the wrapper for the
original batch and for every OOM retry subbatch, so no retry attempt bypasses the
gate. The wrapper forwards the configured runtime step's `optimizer` attribute so
existing training-time retry restrictions remain unchanged.

The orchestrator resolves fixed or provider-backed `ResourceCapacity` once before
the pass source is consumed. That capacity remains fixed for every split and retry
attempt in the pass. The sample provider is called once immediately before each candidate assessment,
including rejected parents and every recursively produced child.

## Configuration

The gate is disabled unless `admission_profile` is supplied to
`ConservativeRuntimeOrchestrator`.

When enabled, the caller must also supply:

- `admission_sample_provider`;
- either `resource_capacity` or `resource_capacity_provider`.

Optional configuration includes:

- `admission_policy`;
- `admission_unknown_action`, defaulting to `AdmissionUnknownAction.BLOCK`;
- `admission_split_policy`, disabled by default.

Admission-only options without a profile are rejected instead of being silently
ignored. Existing fixed-capacity/provider mutual exclusion remains unchanged.

## Optional bounded reject recovery

`AdmissionSplitPolicy` allows `RuntimeRetryRunner` to recover only a trusted
private request created by the orchestrator admission wrapper when its gate blocks
before configured-step execution. A public `PrePassAdmissionBlocked` raised by a
generic runtime step is terminal and is not proof of preflight provenance. An
eligible request contains a `REJECT` whose assessment supplies a positive finite `max_admissible_items` below
the current batch size. `UNKNOWN` is never split. The policy bounds:

- recursive admission split depth;
- minimum child item count;
- maximum child parts produced by one rejected parent.

The split target comes directly from `assessment.max_admissible_items`; the runner
does not guess a half size or reuse the OOM `split_factor`. It computes the minimum
part count needed to keep every child at or below the target, then distributes rows
as evenly as possible. Recovery is refused unless every child is at least
`min_items` and no larger than the assessed target.

Admission depth and OOM retry depth are independent. On the trusted wrapper path,
admission splitting occurs before model execution and remains available when the wrapped step has an
optimizer. OOM retry still follows its existing phase and optimizer restrictions.
A recoverable private request and its underlying block have their tracebacks
cleared before child recursion; terminal blocks keep their normal traceback and propagate.

Recovered admission rejection does not create a fault result or directly change
the governor. The governor observes only the final `StepResult` objects. Skip,
source replay, rollback, and admission-based budget feedback remain outside this
slice. See [`dev_prepass_admission_split.md`](dev_prepass_admission_split.md).

## Pass result and failure semantics

A successfully completed `RuntimePassResult` appends
`admission_assessments`, ordered by execution attempt. The tuple is empty when the
gate is disabled.

The assessment count may exceed the final `StepResult` count. A rejected parent
may be recorded before admitted split children, or an admitted original batch may
execute with an OOM result that is discarded by retry. All parent, child, original,
and retry assessments remain visible in a completed recovered pass result.

When the gate blocks:

- the blocked candidate is not passed to the runtime step;
- no `RuntimePassResult` is created for that pass;
- `ConservativeRuntimeGovernor.observe_results(...)` is not called;
- later source candidates are not consumed;
- earlier candidates in the same pass may already have executed and are not rolled
  back.

The block exception carries the assessment for the failed candidate. This is a
bounded stop policy, not transactional pass execution.

## Out of scope

This slice does not provide:

- skip-and-continue behavior or skipped-row records;
- heuristic splitting that ignores `max_admissible_items`;
- source replay or rollback;
- automatic profile refresh or persistence;
- admission-driven governor changes;
- raw admission assessment retention or persistent summary/history telemetry
  export;
- per-dimension unknown policies;
- multi-GPU or distributed admission;
- stable `enn_torch` exposure.

## Validation

```bash
python -m pytest enn_torch_dev/debug/runtime/test_admission_observability.py -q
python -m pytest enn_torch_dev/debug/runtime/test_prepass_admission_gate.py -q
python -m pytest enn_torch_dev/debug/runtime/test_prepass_admission_split.py -q
python -m pytest enn_torch_dev/debug/runtime/test_prepass_admission.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_orchestration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```
