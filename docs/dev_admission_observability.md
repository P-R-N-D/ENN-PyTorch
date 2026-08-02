# Development Admission Summary and History Observability

This document describes the development-only inspection layer that reduces
completed pre-pass admission assessments into bounded scalar provenance.

## Goal

`RuntimePassResult.admission_assessments` preserves attempt-ordered evidence for a
completed pass, including admitted candidates, explicitly allowed unknowns, and
rejected parents recovered through bounded splitting. This slice makes that
evidence visible through the existing summary and history layers:

```text
RuntimePassResult.admission_assessments
  -> summarize_runtime_pass(...)
  -> scalar RuntimePassSummary admission fields
  -> bounded RuntimePassHistory retained-window aggregation
```

The inspection layer does not alter admission, retry, governor, source, or model
execution behavior.

## Pass summary contract

`RuntimePassSummary` appends:

```text
admission_assessment_count
admission_admit_assessment_count
admission_recovered_reject_count
admission_allowed_unknown_count
admission_recovery_occurred
minimum_recovered_admissible_items
```

The fields mean:

- `admission_assessment_count`: every completed-pass admission assessment;
- `admission_admit_assessment_count`: assessments with status `ADMIT`;
- `admission_recovered_reject_count`: completed-pass `REJECT` assessments;
- `admission_allowed_unknown_count`: completed-pass `UNKNOWN` assessments;
- `admission_recovery_occurred`: whether at least one recovered reject exists;
- `minimum_recovered_admissible_items`: the smallest positive reducing
  `max_admissible_items` among recovered rejects, or `None`.

A terminal block never produces a `RuntimePassResult`, so a valid completed-pass
`REJECT` represents a parent recovered through the trusted bounded split path. A
completed-pass `UNKNOWN` represents an explicitly allowed unknown assessment.

`summarize_runtime_pass(...)` validates every entry as a
`PrePassAdmissionAssessment`. A completed-pass reject must carry a bool-excluding
positive integer target smaller than its assessed batch size. Invalid manually
constructed pass results are rejected rather than silently described as recovered.

## Reference safety

The summary stores only integer counts, one boolean, and one optional integer. It
does not retain:

- `PrePassAdmissionAssessment` objects;
- dimension records or warning tuples;
- `PrePassAdmissionBlocked` or private split requests;
- `KVBatch`, tensors, sources, stores, losses, or `ResourceSample` objects.

The original `RuntimePassResult` remains caller-owned. The summary is a separate
lightweight record.

## History contract

`RuntimeHistorySummary` appends retained-window aggregates:

```text
admission_assessed_passes
admission_recovery_passes
admission_total_assessments
admission_admit_assessments
admission_recovered_rejects
admission_allowed_unknowns
minimum_recovered_admissible_items
```

`admission_assessed_passes` counts retained summaries with at least one admission
assessment. `admission_recovery_passes` counts retained summaries with one or more
recovered rejects. Assessment counts are summed, while the minimum recovered item
limit is the minimum known value in the retained window.

`RuntimePassHistory` continues to recompute aggregates from the currently retained
summary records. When `max_records` trimming removes a record, all of that record's
admission contributions disappear from the aggregate.

## Governor boundary

Admission observability remains lightweight evidence. An optional governor growth
guard may consume the minimum recovered item limit to reset clean-success growth,
but it does not retain raw assessments or reinterpret recovery as `StepStatus` or
OOM.

The guard does not directly cap or shrink `max_items`, create an admission streak,
or override OOM and pressure decisions. See
[`dev_admission_governor_growth_guard.md`](dev_admission_governor_growth_guard.md).

## Failure boundary

A terminal admission block still prevents creation of a pass result. Therefore it
is not summarized or appended to history. Earlier work in the same pass is not
rolled back, and no partial summary is created.

## Out of scope

- admission-driven governor feedback;
- skipped-row records or reprocessing;
- source replay or rollback;
- raw assessment history retention;
- persistent telemetry, JSONL, CSV, or dashboard export;
- profile refresh or persistence;
- multi-GPU or distributed aggregation;
- stable `enn_torch` exposure;
- new dependencies.

## Validation

```bash
python -m pytest enn_torch_dev/debug/runtime/test_admission_observability.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_summary.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_history.py -q
python -m pytest enn_torch_dev/debug/runtime/test_prepass_admission_split.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```
