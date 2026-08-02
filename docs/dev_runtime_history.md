# Runtime Pass History

This document describes the thirteenth data/runtime rewrite slice for
`enn_torch_dev`.

## Goal

`RuntimePassSummary` provides a compact inspection record for one finite runtime
pass. This slice adds an in-memory history layer for multiple summaries:

```text
RuntimePassSummary records
  -> RuntimePassHistory
  -> RuntimeHistorySummary
  -> format_runtime_history_summary(...)
```

The history layer exists for local inspection across finite passes. It does not
write files, persist logs, export JSONL/CSV, run dashboards, execute models,
retry batches, or adjust budgets.

## Public Objects

`enn_torch_dev.runtime` exports:

- `RuntimeHistorySummary`
- `RuntimePassHistory`
- `format_runtime_history_summary`

The stable `enn_torch` namespace does not expose this development history API.

## Contract

`RuntimePassHistory` is a small mutable in-memory container with bounded
retention. It requires `max_records` as a positive integer; `None`, booleans,
non-integers, and non-positive integers are rejected. Appending a new summary
trims the oldest records first so only the latest `max_records` summaries are
retained. Unbounded retention is not supported.

The history accepts two append paths:

- `append_summary(summary)` accepts an existing `RuntimePassSummary`.
- `append_pass_result(pass_result)` first calls `summarize_runtime_pass(...)` and
  stores only the resulting summary.

`records` returns a tuple snapshot of currently retained summaries. The snapshot
is not a live mutable view of the history internals.

`RuntimeHistorySummary` aggregates retained records into:

- total pass count;
- total result count;
- total batch size;
- total row count;
- status counts across retained summaries;
- recovered-OOM pass count;
- yielded-OOM pass count;
- budget-changed pass count;
- pressure-assessed pass count;
- pressure-growth-suppressed pass count;
- highest known pressure ratio across retained summaries;
- count of retained passes whose budget actually shrank because of sustained pressure;
- CPU and CUDA high-pressure pass counts;
- CPU and CUDA trigger pass counts;
- pressure adjustment-attempt pass count, including minimum-bound no-ops;
- full pressure adjustment no-op pass count;
- trigger-without-matching-budget pass count;
- actual host-byte, device-byte, and `max_items` fallback shrink pass counts;
- admission-assessed and admission-recovery pass counts;
- total retained admission assessments;
- retained admitted, recovered-reject, and allowed-unknown assessment counts;
- the smallest recovered admissible item limit in the retained window;
- latest retained `RuntimePassSummary`, or `None` for an empty history.

`format_runtime_history_summary(summary)` returns stable human-readable text for
runtime history inspection, including retained-window pressure counts,
dimension-specific high/trigger counts, attempted/no-op adjustment counts,
field-specific actual shrink counts, peak ratio, and latest-pass pressure state.
It also reports retained-window admission counts, the minimum recovered item
limit, and latest-pass recovery state. It is not a stable machine interchange
format.

Every aggregate is recomputed only from the currently retained summaries. When
`max_records` trimming removes an older summary, that summary no longer
contributes to any dimension, attempt, no-op, trigger-without-budget, or
field-specific shrink counter. A pass can increment both CPU and CUDA dimension
counters, but each pass increments the adjustment-attempt, full-no-op, and
trigger-without-budget counters at most once. A selected pass with no actually
changed field is a full adjustment no-op; a pass with at least one changed
selected field is not counted as a no-op. History does not infer structured
provenance from pressure ratios or OOM status when provenance tuples are empty.
Admission aggregates follow the same retained-window rule: trimming removes every
assessment count, recovery-pass contribution, and recovered-limit contribution
from the discarded summary.

## Reference Safety

`RuntimePassHistory.append_pass_result(...)` stores a `RuntimePassSummary`, not
the original `RuntimePassResult` or `StepResult` objects. Since
`RuntimePassSummary` is lightweight, the history does not retain `StepResult`,
raw `ResourceSample`, `loss`, or `store` references through that append path.
Pressure aggregation uses only scalar ratios and immutable structured provenance
from each retained summary. That provenance includes high and triggered
dimensions, selected budget fields, and fields that actually shrank; it does not
require retaining any additional raw runtime objects. Admission aggregation uses
only scalar summary fields and does not retain raw admission assessments,
dimension or warning tuples, exceptions, batches, samples, sources, stores,
losses, or tensors.

## Relationship to Session

`docs/dev_runtime_session.md` describes a bounded lazy coordinator that appends
one `RuntimePassSummary` after each completed pass. The session reuses the
history's required `max_records` bound and does not replace or bypass its
retention policy.

## Out of Scope

- Running `RuntimeStep`.
- Retrying or splitting batches.
- Budget shrink/grow decisions.
- Persistent logging, JSONL, CSV, or dashboard export.
- Unbounded retention or streaming summarization.
- AutoGovernor behavior.
- Learned or model-specific tuning.
- ResourceMonitor feedback loops.
- Admission-driven governor feedback or next-pass budget changes.
- Checkpoint/resume.
- Distributed coordination.

## Test Commands

```bash
python -m pytest enn_torch_dev/debug/runtime/test_admission_observability.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_history.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_summary.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_orchestration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_governor.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug/data -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```
