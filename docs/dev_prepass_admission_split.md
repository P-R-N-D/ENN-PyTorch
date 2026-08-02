# Development Bounded Admission Reject Splitting

This document describes the development-only recovery path for a candidate that
is blocked by pre-pass admission with a concrete smaller item limit.

## Goal

The opt-in admission gate stops a candidate before execution when the assessment
is `REJECT` or when `UNKNOWN` is configured to block. This slice optionally
recovers only a `REJECT` whose assessment provides a usable smaller batch size:

```text
candidate KVBatch
  -> pre-pass admission REJECT
  -> assessment.max_admissible_items
  -> bounded identity-preserving split
  -> fresh admission assessment for every child
  -> admitted children execute
```

The runner does not skip rows, invent a split factor, replay the source, roll back
previous work, update calibration, or persist recovery state.

## Development API

`AdmissionSplitPolicy` is exported from `enn_torch_dev.runtime`:

```python
@dataclass(frozen=True, slots=True)
class AdmissionSplitPolicy:
    max_split_depth: int = 3
    min_items: int = 1
    max_split_parts: int = 16
```

- `max_split_depth` is a non-negative bound independent from OOM retry depth.
- `min_items` is the positive lower bound for every child.
- `max_split_parts` is at least two and bounds the children produced by one
  rejected parent.

The policy is disabled when omitted. Existing gate behavior therefore remains a
terminal block unless the caller explicitly opts into recovery.

## Recovery eligibility

A blocked candidate is recoverable only when all of the following are true:

```text
assessment.status == REJECT
assessment.batch_size == current batch size
max_admissible_items is a bool-excluding positive integer
max_admissible_items < current batch size
max_admissible_items >= policy.min_items
current admission depth < policy.max_split_depth
required parts <= policy.max_split_parts
all rows can be assigned to valid children
```

`UNKNOWN` is never split, even if a malformed or manually constructed assessment
contains a numeric item limit. A zero, negative, missing, non-integer, boolean,
non-reducing, or candidate-mismatched limit remains terminal.

## Split calculation

The assessed item limit is authoritative:

```text
target = assessment.max_admissible_items
parts = ceil(batch_size / target)
```

The runner does not halve the batch and does not reuse `RetryPolicy.split_factor`.
Rows are distributed as evenly as possible across the minimum required part count.
Recovery proceeds only when every resulting child satisfies:

```text
policy.min_items <= child.batch_size <= target
```

For example, a batch of ten with target three and minimum two becomes:

```text
3, 3, 2, 2
```

A batch of five with target two and minimum two cannot cover every row without a
child below the minimum, so it remains blocked.

Splitting uses `slice_kvbatch(...)`, preserving row, source, and sample identity,
schema metadata, shard metadata, and source order. Each child materializes the
same identity boundary already used by OOM retry splitting.

## Interaction with OOM retry

Admission recovery and OOM retry have separate counters:

```text
admission rejection -> admission_split_depth + 1
runtime OOM result   -> retry_count + 1
```

An admission split does not consume OOM retry depth because no model execution
occurred. An OOM split does not consume admission depth. Therefore:

- an admission child may later use ordinary OOM retry;
- an OOM retry subbatch may itself be admission-split before execution;
- each recursively produced candidate is reassessed with a fresh resource sample.

Admission splitting occurs before execution and is allowed when the wrapped step
has an optimizer. Existing OOM retry remains disabled for optimizer-backed steps
because that restriction concerns side effects after execution begins.

## Assessment order and exception retention

The orchestration admission wrapper records an assessment whether the gate returns
or raises. A completed recovered pass therefore preserves attempt order such as:

```text
REJECT parent
ADMIT child 1
ADMIT child 2
```

A recovered internal `PrePassAdmissionBlocked` is consumed by the retry runner.
Its assessment is already recorded by the wrapper, and its traceback is cleared
before child recursion so the recovered exception does not continue retaining the
parent execution frames. This is an internal cleanup only; it does not change the
documented behavior of terminal block exceptions.

If recovery is impossible or a child is terminally blocked, the block propagates,
no `RuntimePassResult` is created for that pass, and the governor is not updated.
Earlier candidates from the same pass may already have executed and are not rolled
back.

## Governor and inspection boundary

A successfully recovered admission rejection does not become a `StepStatus` and
does not directly shrink, grow, or suppress the governor budget. The governor
observes only the final `StepResult` sequence and existing OOM/pressure signals.

`RuntimePassResult.admission_assessments` exposes the parent and child evidence for
immediate caller inspection. This slice does not add admission recovery fields to
`RuntimePassSummary` or `RuntimePassHistory`.

## Out of scope

This slice does not provide:

- skip-and-continue behavior or skipped-row records;
- source replay or transactional rollback;
- heuristic or learned split sizes;
- admission-based governor feedback;
- summary/history admission aggregation;
- automatic profile refresh or persistence;
- per-candidate capacity refresh;
- multi-GPU or distributed coordination;
- stable `enn_torch` exposure.

## Validation

```bash
python -m pytest enn_torch_dev/debug/runtime/test_prepass_admission_split.py -q
python -m pytest enn_torch_dev/debug/runtime/test_prepass_admission_gate.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_retry.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_orchestration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```
