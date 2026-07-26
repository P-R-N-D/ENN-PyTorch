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
