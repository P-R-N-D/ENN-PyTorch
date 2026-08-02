# ENN-PyTorch Change Checklist

Every change must include an AI-facing documentation impact review.

If a change affects repository structure, architecture, public APIs, package boundaries, configuration, dependencies, test commands, compatibility contracts, runtime safety rules, artifact handling, documented workflows, or current-state classification, update the affected AI-facing documents in the same PR.

Do not defer required documentation updates to a follow-up task. Do not edit unrelated AI-facing documents merely to create churn.

## Checklist by change type

| Change type | Code paths to check | Test paths to check | AI-facing docs likely to update | Compatibility risks |
|---|---|---|---|---|
| public API / `__all__` | `enn_torch/__init__.py`, package `__init__.py` files, `enn_torch_dev/*/__init__.py` | Stable `enn_torch` targeted import/API smoke checks; `enn_torch_dev/debug/executor/test_public_api_exports.py` for executor public exports; relevant area tests when covered | `docs/CONTEXT.md`, `docs/CURRENT_STATE.md`, `docs/TESTING.md` | Breaking imports, exposing active-development APIs as stable, changing lazy import behavior |
| configuration | `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `MANIFEST.in`, config modules | Relevant debug tests plus packaging review | `docs/CONTEXT.md`, `docs/CURRENT_STATE.md`, `docs/TESTING.md` | Python version drift, dependency mismatch, package data omissions |
| checkpoint and save/load | `enn_torch/runtime/io.py`, `enn_torch/runtime/workflows.py`, related runtime modules | Targeted runtime tests if present; otherwise document untested scope | `docs/RUNTIME_SAFETY.md`, `docs/CHANGE_CHECKLIST.md`, `docs/CONTEXT.md` | Overwriting user artifacts, adding generated files, changing formats |
| data schema and manifest | `enn_torch_dev/data/schema.py`, `manifest.py`, `batch.py`, `staging.py`, `readers.py` | `enn_torch_dev/debug/data -q` | `docs/CURRENT_STATE.md`, `docs/TESTING.md`, `docs/RUNTIME_SAFETY.md` | Breaking staged data compatibility, identity tensor semantics, schema validation changes |
| executor graph/node/tile/stream/state | `enn_torch_dev/executor/**` | `enn_torch_dev/debug/executor -q` | `docs/CURRENT_STATE.md`, `docs/TESTING.md`, executor docs if behavior changes | Execution ordering, state routing, public exports, tile/stream terminology |
| runtime step, retry, orchestration, session, source factory, and fault classification | `enn_torch_dev/runtime/step.py`, `retry.py`, `orchestration.py`, `session.py`, `source_factory.py`, `faults.py`, loader/runtime integration | `enn_torch_dev/debug/runtime/test_runtime_integration.py -q`; `enn_torch_dev/debug/runtime -q` | `docs/CURRENT_STATE.md`, `docs/TESTING.md`, `docs/RUNTIME_SAFETY.md`, relevant `docs/dev_*.md` | Incorrect success/fault reporting, retry loops, orchestration materialization scope, session over-consumption, factory over-invocation, source caching, exception suppression, recovered-OOM signaling, row-order changes, unverified failure modes, optimizer/loss behavior changes, cross-layer contract drift |
| resource monitoring, capacity/pressure assessment, cost/batching, runtime governor, orchestration, pass summaries, pass history, sessions, and source factories | `enn_torch_dev/runtime/cost.py`, `footprint.py`, `resources.py`, `capacity_provider.py`, `pressure.py`, `budget_recommendation.py`, `calibration.py`, `batching.py`, `governor.py`, `orchestration.py`, `summary.py`, `history.py`, `session.py`, `source_factory.py` | `enn_torch_dev/debug/runtime/test_budget_recommendation.py -q`; `enn_torch_dev/debug/runtime/test_observed_cost_calibration.py -q`; `enn_torch_dev/debug/runtime/test_runtime_pressure.py -q`; `enn_torch_dev/debug/runtime/test_runtime_integration.py -q`; `enn_torch_dev/debug/runtime -q` | `docs/CURRENT_STATE.md`, `docs/TESTING.md`, `docs/RUNTIME_SAFETY.md`, relevant `docs/dev_*.md` | CPU/CUDA assumptions, cgroup v1/v2 path parsing, hierarchy-effective cgroup limit discovery, physical-vs-effective CPU capacity, capacity lookup fallback, device mismatch, unknown-vs-zero pressure semantics, unclamped ratios, common-vs-dimension pressure-threshold fallback, per-dimension required-pass validation, common-vs-dimension pressure-factor fallback, shared max-items fallback factor selection, growth-limit ordering against every active effective shrink threshold, missing-summary growth suppression, OOM priority over pressure, fixed/provider capacity mutual exclusion, pass-start provider call count, provider failure before source consumption, invalid provider returns, per-pass capacity provenance, retry-attempt sample inclusion, CUDA capacity/sample mismatch propagation, memory accounting semantics, footprint device-provenance drift, ModelCost CUDA-provenance drift, unknown-vs-zero observed-cost semantics, negative-delta handling, fault-sample contamination, cross-device calibration contamination, unbounded phase-pair retention, raw observation retention, unknown-vs-zero recommendation cost semantics, unsafe upward min-item clamping, fallback recommendation drift, reserve/utilization ordering, oversized batch behavior, budget shrink/grow bounds, dimension-aware pressure-to-budget mapping, structured high/trigger/selection/factor provenance drift, cross-dimension streak contamination, legacy aggregate streak compatibility, max-items fallback drift, actual-vs-attempted adjusted-field reporting, accidental execution/retry coupling, unbounded pass materialization, summary/history retaining `StepResult`, raw `ResourceSample`, `store`, or `loss` references, pressure summary field-order compatibility, sustained-pressure streak reset and priority errors, retained-window dimension/attempt/no-op/field-shrink aggregation drift, retained-window pressure shrink aggregation drift, unbounded history retention, retention-bound regression, unbounded session execution, session retaining previous pass results, factory returning cached or non-finite sources, unexpected persistent logging, stable namespace leakage, identity/order regression across split and retry paths |
| dependency and optional extras | `pyproject.toml`, `requirements.txt`, `requirements-dev.txt` | Targeted import/test for affected dependency; report missing optional packages | `docs/CONTEXT.md`, `docs/TESTING.md`, `docs/RUNTIME_SAFETY.md` | Unrequested dependency changes, optional backend availability, platform markers |
| test layout and commands | `enn_torch_dev/debug/**`, pytest paths | New or moved tests themselves | `docs/TESTING.md`, `docs/CONTEXT.md` | Documenting non-existent commands, hiding skipped dependency or CUDA coverage |
| runtime safety and artifacts | Runtime IO/export/checkpoint code, staging writers, loaders, notebooks | Targeted tests using temp directories | `docs/RUNTIME_SAFETY.md`, `docs/CONTEXT.md` | Data loss, committed artifacts, secret leakage, long-running validation |
| repository structure | Top-level files, package directories, `docs/`, `.github/` | `git diff --check`, path existence checks | `docs/CONTEXT.md`, `docs/CURRENT_STATE.md`, `docs/TESTING.md`, this file | Broken relative links, package discovery changes, forbidden AI-doc paths |
| documentation-only changes | Changed Markdown files | `git diff --check`; link/path review | Affected docs only | Churn, stale instructions, treating plans as implementation |
| Notion reference alignment | Repository docs that mention external Notion context | No Notion edits; verify against current repository implementation and tests | `docs/CONTEXT.md`, `docs/CURRENT_STATE.md` | Treating Notion as implementation source of truth, classifying external reference material as historical implementation, presenting unapproved roadmap as current state |

## Pre-pass admission assessment changes

- Check `enn_torch_dev/runtime/admission.py` and
  `enn_torch_dev/debug/runtime/test_prepass_admission.py`.
- Verify unknown-vs-zero semantics, current-vs-peak CUDA baseline handling,
  `REJECT` precedence over `UNKNOWN`, concrete CUDA provenance agreement, and
  finite known item-limit calculation.
- Confirm the assessor remains pure and is not wired into batching, retry,
  governor, orchestration, source consumption, or model execution.

## Pre-pass admission gate changes

- Check `enn_torch_dev/runtime/admission_gate.py`, the admission wrapper in
  `enn_torch_dev/runtime/orchestration.py`, and
  `enn_torch_dev/debug/runtime/test_prepass_admission_gate.py`.
- Verify `REJECT` always blocks, `UNKNOWN` blocks by default, and explicit allow
  applies only to `UNKNOWN`.
- Verify the gate samples once per original or retry execution attempt, while a
  capacity provider remains pass-scoped and is called once before source
  consumption.
- Confirm blocked attempts do not call the runtime step, do not become a
  `StepStatus`, and do not update governor state. Document that earlier candidates
  in the same pass may already have executed before a later block.
- Confirm optimizer passthrough preserves retry restrictions and completed
  `RuntimePassResult.admission_assessments` retains only immutable assessments.
  Separately confirm that a block exception's custom payload stores only its
  assessment; do not confuse this with the exception traceback's transitive frame
  references. Confirm stable `enn_torch` exports remain unchanged.

## Bounded admission split recovery changes

- Check `AdmissionSplitPolicy`, `RuntimeRetryRunner` admission exception handling,
  the orchestration wrapper assessment order, and
  `test_prepass_admission_split.py`.
- Confirm only the orchestrator's private admission-aware pre-execution request can
  trigger recovery. A generic runtime step's public `PrePassAdmissionBlocked` must
  remain terminal, with and without an `optimizer` attribute, to prevent duplicate
  side effects.
- Split only `REJECT` with a matching candidate batch size and a positive finite
  `max_admissible_items` smaller than the current batch. Never split `UNKNOWN`,
  zero/unknown limits, or malformed/mismatched assessments.
- Verify every child is between `min_items` and the assessed target, part count and
  recursive depth are bounded, and identity/order are preserved.
- Keep admission split depth independent from OOM retry depth. Admission splitting
  may precede execution with an optimizer, while existing optimizer-based OOM retry
  restrictions must remain unchanged.
- Record the rejected parent before child assessments in completed recovered pass
  results. Terminal blocks must not update governor state or return partial pass
  results. Recovered rejection must not create governor feedback; summary and
  history may retain only bounded scalar admission provenance.
- Confirm internal recovered block tracebacks are cleared before recursion, while
  terminal block traceback behavior and the documented custom-payload contract are
  unchanged.

## Admission summary and history observability changes

- Check `summary.py`, `history.py`, and `test_admission_observability.py`.
- Count completed-pass `ADMIT`, recovered `REJECT`, and explicitly allowed
  `UNKNOWN` assessments separately. A terminal block must not create a summary or
  history record.
- Require completed-pass rejects to carry a bool-excluding positive reducing
  `max_admissible_items`; reject malformed manually constructed pass results.
- Keep summary fields append-only and scalar. Do not retain raw assessment,
  dimension, warning, exception, batch, source, sample, tensor, store, or loss
  objects.
- Recompute history counts and the minimum recovered item limit only from the
  currently retained window; trimming must remove every discarded contribution.
- Confirm scalar admission provenance remains reference-safe and retained-window bounded.
- Confirm stable `enn_torch` exports remain unchanged.

## Admission recovery governor growth guard changes

- Check `GovernorPolicy`, `GovernorDecision`, `observe_results(...)`, orchestration
  limit extraction, and `test_admission_growth_guard.py`.
- Keep the guard opt-in and validate a bool policy plus a positive non-bool integer
  recovery limit.
- Ensure yielded and retry-recovered OOM retain priority. Preserve pressure streaks,
  selected fields, factors, and actual pressure shrink.
- On an otherwise successful recovered pass, reset the success streak and cancel
  only success-driven growth. Do not directly cap or create `max_items`.
- Copy decision provenance into pass summaries and count actual suppression only
  inside the retained history window.
- Confirm terminal admission blocks still bypass governor, summary, and history.

## Required final-report result

Every final report must include exactly one of:

```text
AI docs updated:
- <documents updated>
```

or

```text
AI docs impact: none
Reason: <concrete reason>
```
