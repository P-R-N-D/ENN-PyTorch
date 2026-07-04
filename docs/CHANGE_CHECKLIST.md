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
| runtime step, retry, orchestration, and fault classification | `enn_torch_dev/runtime/step.py`, `retry.py`, `orchestration.py`, `faults.py`, loader/runtime integration | `enn_torch_dev/debug/runtime -q` | `docs/CURRENT_STATE.md`, `docs/TESTING.md`, `docs/RUNTIME_SAFETY.md`, relevant `docs/dev_*.md` | Incorrect success/fault reporting, retry loops, orchestration materialization scope, recovered-OOM signaling, row-order changes, unverified failure modes, optimizer/loss behavior changes |
| resource monitoring, cost/batching, runtime governor, orchestration, and pass summaries | `enn_torch_dev/runtime/cost.py`, `footprint.py`, `resources.py`, `batching.py`, `governor.py`, `orchestration.py`, `summary.py` | `enn_torch_dev/debug/runtime -q` | `docs/CURRENT_STATE.md`, `docs/TESTING.md`, `docs/RUNTIME_SAFETY.md`, relevant `docs/dev_*.md` | CPU/CUDA assumptions, memory accounting semantics, oversized batch behavior, budget shrink/grow bounds, accidental execution/retry coupling, unbounded pass materialization, summary retaining `StepResult.store` or `loss` references |
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
