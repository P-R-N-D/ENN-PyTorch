# Development Admission Recovery Growth Guard

This document describes the opt-in governor feedback that prevents a recovered
pre-pass admission rejection from being counted as clean success evidence.

## Goal

A pass can complete successfully only after a rejected parent is split into
smaller admitted children. Without an explicit guard, the final successful
`StepResult` sequence can advance the ordinary success streak and grow the next
budget back toward the rejected size.

The guard adds bounded feedback:

```text
completed recovered REJECT assessments
  -> minimum recovered max_admissible_items
  -> ConservativeRuntimeGovernor.observe_results(...)
  -> optional clean-success growth suppression
```

This slice does not cap or shrink `max_items` to the recovered limit.

## Policy

`GovernorPolicy.suppress_growth_after_admission_recovery` is a bool and defaults
to `False`. The default preserves existing governor behavior.

When enabled, a positive `admission_recovery_max_items` supplied with an otherwise
successful observation:

- resets `consecutive_successes` to zero;
- cancels success-threshold growth for that observation;
- keeps the existing budget unless pressure independently shrinks it;
- records structured decision provenance.

## Priority

OOM remains highest priority. Yielded or retry-recovered OOM follows the existing
shrink path and is never replaced or undone by the admission guard.

Sustained-pressure shrink also remains intact. A successful pass may report both
pressure-based and admission-based growth suppression, but admission feedback does
not alter pressure streaks, selected fields, factors, or actual shrink results.

Non-OOM faults and empty observations retain their existing behavior and do not
report admission growth suppression. Explicitly allowed `UNKNOWN` assessments do
not activate the guard because no recovered reject limit exists.

## Provenance

`GovernorDecision` appends:

```text
admission_recovery_max_items
growth_suppressed_by_admission_recovery
```

`RuntimePassSummary` copies that decision evidence separately from the assessment
minimum:

```text
minimum_recovered_admissible_items
governor_admission_recovery_max_items
growth_suppressed_by_admission_recovery
```

The first value is derived from completed admission assessments. The second is the
value actually supplied to the governor. Normal orchestrator results should make
them equal.

`RuntimeHistorySummary.admission_growth_suppressed_passes` counts retained pass
summaries where the guard actually operated.

## Safety boundary

The orchestrator derives the governor input only from completed-pass `REJECT`
assessments produced by the trusted admission wrapper. Terminal blocks still
propagate before governor observation and produce no pass result or history entry.

This slice does not:

- set `BatchBudget.max_items` to the recovered limit;
- activate an unset `max_items` field;
- resolve conflicts with `GovernorPolicy.min_items`;
- create an admission recovery streak;
- skip rows, replay sources, or roll back completed work;
- expose a stable `enn_torch` API;
- add dependencies.

## Validation

```bash
python -m pytest enn_torch_dev/debug/runtime/test_admission_growth_guard.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_governor.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_orchestration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_admission_observability.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_summary.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_history.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```
