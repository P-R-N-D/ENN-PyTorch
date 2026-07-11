# Bounded Runtime Session

This document describes the fourteenth data/runtime rewrite slice for
`enn_torch_dev`.

## Goal

The development runtime already has separate components for one finite pass,
single-pass inspection, and bounded in-memory history. This slice adds a bounded
session layer that connects them across multiple finite pass sources:

```text
Iterable[finite KVBatch pass sources]
  -> ConservativeRuntimeSession.run_passes(...)
  -> ConservativeRuntimeOrchestrator.run_pass(...)
  -> RuntimePassResult
  -> summarize_runtime_pass(...)
  -> RuntimePassHistory.append_summary(...)
  -> RuntimeSessionRecord
```

The session is deliberately small. It coordinates existing components but does
not absorb their execution, retry, governor, summary, or retention
responsibilities.

## Public Objects

`enn_torch_dev.runtime` exports:

- `RuntimeSessionRecord`
- `ConservativeRuntimeSession`

The stable `enn_torch` namespace does not expose this development session API.

## Constructor Contract

`ConservativeRuntimeSession` requires:

- a `ConservativeRuntimeOrchestrator`;
- a bounded `RuntimePassHistory`;
- `max_passes` as a positive integer.

`max_passes` is required. `None`, booleans, non-integers, and non-positive
integers are rejected. The session therefore has an explicit upper bound even
when the outer pass-source iterable is longer or unbounded.

The session reuses the provided history. It does not clear existing retained
summaries when a new `run_passes(...)` or `run_factory(...)` call begins.

## Lazy Pass Execution

`run_passes(pass_sources)` accepts an iterable whose items are finite iterables of
`KVBatch`.

Calling `run_passes(...)` returns an iterator without executing a pass. Each
`next()` call executes at most one pass and yields one `RuntimeSessionRecord`.
The session stops when either:

1. the outer iterable is exhausted; or
2. `max_passes` records have been yielded.

The implementation does not fetch one extra outer source after the pass limit is
reached.

`run_factory(source_factory)` provides the same lazy, bounded execution contract
while asking a `RuntimePassSourceFactory` to create one fresh finite source for
each pass index. The factory is not called until the returned iterator advances,
and it is not called after `max_passes` is reached.

Each `RuntimeSessionRecord` contains:

- `pass_index`, starting at zero for each `run_passes(...)` invocation;
- the current `RuntimePassResult`;
- the derived `RuntimePassSummary`;
- the current bounded `RuntimeHistorySummary`.

## Relationship to Source Factory

`docs/dev_runtime_source_factory.md` describes the source-construction protocol
used by `run_factory(...)`. The factory owns fresh source creation; the session
continues to own bounded iteration, pass execution, summary creation, and history
append. It does not cache or replay created sources.

## Error and Fault Semantics

Python exceptions from source iteration, pass execution, summary construction,
or history append are not suppressed. If pass execution or summary construction
raises before history append completes, that pass is not added to history.

A yielded `StepStatus` fault is data in a completed `RuntimePassResult`; it is not
treated as a session exception and does not automatically stop later passes.
Retry and OOM behavior remain owned by `RuntimeRetryRunner` and
`ConservativeRuntimeOrchestrator`.

## Memory and Reference Boundary

The session does not materialize all session records and does not retain previous
`RuntimeSessionRecord` or `RuntimePassResult` objects internally. The currently
yielded record may contain the finite pass result for caller inspection, and the
caller may keep that record for as long as needed.

When the generator resumes after a yield, it releases the previous pass's
`source`, `RuntimePassResult`, `RuntimePassSummary`, and
`RuntimeHistorySummary` locals before consuming the next outer source. The
session frame therefore does not keep a previous pass's `RuntimePassResult`,
`StepResult.store`, or `loss` alive into the next pass execution window.

Longer-lived retention remains limited to lightweight `RuntimePassSummary`
objects inside the provided bounded `RuntimePassHistory`.

## Out of Scope

- Persistent logging or JSONL/CSV export.
- Dashboards or telemetry backends.
- Checkpoint/resume.
- Automatic source replay or caching of consumed sources.
- Exception suppression or automatic continuation after Python exceptions.
- Distributed execution or aggregation.
- AutoGovernor behavior.
- Learned or model-specific tuning.
- ResourceMonitor feedback loops.
- Materializing all session results.
- Changing the finite materialization boundary inside one orchestrator pass.

## Test Commands

```bash
python -m pytest enn_torch_dev/debug/runtime/test_runtime_source_factory.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_session.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_history.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_summary.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_orchestration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_governor.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug/data -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```
