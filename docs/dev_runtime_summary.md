# Runtime Pass Summary

This document describes the twelfth data/runtime rewrite slice for
`enn_torch_dev`.

## Goal

`ConservativeRuntimeOrchestrator` returns a finite `RuntimePassResult` containing
pass results, the governor decision, and the recovered-OOM flag. This slice adds
a lightweight inspection layer for that record:

```text
RuntimePassResult
  -> summarize_runtime_pass(...)
  -> RuntimePassSummary
  -> format_runtime_pass_summary(...)
```

The summary exists for inspection and reporting. It does not execute models,
retry batches, adjust budgets, persist logs, or stream unbounded workloads.

## Public Objects

`enn_torch_dev.runtime` exports:

- `RuntimePassSummary`
- `summarize_runtime_pass`
- `format_runtime_pass_summary`

The stable `enn_torch` namespace does not expose this development summary API.

## Contract

`RuntimePassSummary` records compact pass-level facts:

- total result count;
- yielded statuses and status counts;
- total `StepResult.batch_size` across the pass;
- total yielded row identity count from `StepResult.row_ids`;
- whether the pass reported `recovered_oom`;
- whether the yielded results include `StepStatus.OOM_FAULT`;
- previous and next governor budgets;
- whether the budget changed;
- governor decision reason;
- governor success/OOM streak counters;
- resource peak values copied from `GovernorDecision`.

`summarize_runtime_pass(pass_result)` accepts only a finite `RuntimePassResult`.
It scans the pass result tuple and stores only lightweight summary fields. It does
not retain `StepResult` objects, `store`, or `loss` references.

`format_runtime_pass_summary(summary)` returns stable human-readable text for a
`RuntimePassSummary`. The formatter is intended for debug output and PR/report
inspection, not for a stable machine interchange format.

## Relationship to Orchestration

The orchestrator owns finite pass execution and may materialize
`RuntimePassResult.results`. The summary layer only inspects that finished result.
It does not change orchestration state or feed decisions back into the governor.

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
python -m pytest enn_torch_dev/debug/runtime/test_runtime_summary.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_orchestration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_governor.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug/data -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```
