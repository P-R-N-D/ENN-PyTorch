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

`RuntimePassHistory` is a small mutable in-memory container. It accepts an
optional positive `max_records` bound. When the bound is set, appending a new
summary trims the oldest records so only the latest `max_records` summaries are
retained.

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
- latest retained `RuntimePassSummary`, or `None` for an empty history.

`format_runtime_history_summary(summary)` returns stable human-readable text for
runtime history inspection. It is not a stable machine interchange format.

## Reference Safety

`RuntimePassHistory.append_pass_result(...)` stores a `RuntimePassSummary`, not
the original `RuntimePassResult` or `StepResult` objects. Since
`RuntimePassSummary` is lightweight, the history does not retain `StepResult`,
`loss`, or `store` references through that append path.

## Out of Scope

- Running `RuntimeStep`.
- Retrying or splitting batches.
- Budget shrink/grow decisions.
- Persistent logging, JSONL, CSV, or dashboard export.
- Unbounded streaming summarization.
- AutoGovernor behavior.
- Learned or model-specific tuning.
- ResourceMonitor feedback loops.
- Checkpoint/resume.
- Distributed coordination.

## Test Commands

```bash
python -m pytest enn_torch_dev/debug/runtime/test_runtime_history.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_summary.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_orchestration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_governor.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug/data -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```
