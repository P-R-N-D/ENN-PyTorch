# ENN-PyTorch Agent Context

## Purpose

This document is the canonical source of truth for AI-facing repository context and instructions in ENN-PyTorch. Use the current repository implementation, tests, and configuration as primary evidence. Prefer verified code, tests, and package metadata over guesses.

## Instruction precedence

1. Direct user, system, and developer instructions.
2. Root `AGENTS.md` and this `docs/CONTEXT.md` file.
3. Focused AI-facing documents in `docs/`.
4. Existing repository documentation and implementation notes.
5. External references, only as supporting context.

When sources conflict, prefer the current GitHub repository implementation and tests over external references or older design notes.

## Document roles

- `AGENTS.md`: thin root entry point for coding agents.
- `CLAUDE.md`: thin Claude Code entry point.
- `docs/CONTEXT.md`: canonical AI-facing repository context and instructions.
- `docs/CURRENT_STATE.md`: current-state classification for major areas.
- `docs/TESTING.md`: verification commands by change scope.
- `docs/RUNTIME_SAFETY.md`: runtime, artifact, backend, GPU, and secret safety rules.
- `docs/CHANGE_CHECKLIST.md`: change-type checklist and required AI documentation impact review.
- `docs/dev_*.md`: implementation-slice design records. Do not treat their `Next Step`, `Follow-up`, or `Out of Scope` sections as current instructions or an approved roadmap.
- `docs/executor_modes.md` and `docs/executor_builders.md`: executor terminology and user-facing builder guidance.

## External references

Connected Notion ENN-PyTorch technical documents may be used only as supporting references for structure and terminology. Do not modify Notion documents from this repository task flow. If Notion and the repository conflict, the current GitHub implementation, tests, and configuration win.

## Repository boundaries

Do not change production Python code, dependencies, lockfiles, existing README bodies, LICENSE body, secrets, credentials, or generated artifacts unless explicitly requested. Do not create `docs/ai/`, `docs/AGENT_*.md`, `docs/HANDOFF_PROMPT.ko.md`, or Markdown instruction files inside `enn_torch/`, `enn_torch_dev/`, or `enn_torch_dev/debug/`.

Preserve README and LICENSE policy. Do not create or store OpenAI Platform files, API keys, tokens, credentials, or `.env` files.

## Package guidance

### enn_torch

`enn_torch` is the stable, user-facing package namespace described by the root package metadata. Its top-level API lazily exposes `core`, `data`, `nn`, and `runtime`, plus workflow helpers such as `new_model`, `load_model`, `save_model`, `train`, and `predict`. Do not remove or change existing public API behavior without explicit approval.

### enn_torch_dev

`enn_torch_dev` is the active-development namespace for the tensor data contract, executor, neural-network components, and runtime rewrite slices. Do not automatically expose `enn_torch_dev` implementations through the stable `enn_torch` namespace. Treat integration, replacement, or migration policy between `enn_torch_dev` and `enn_torch` as maintainer-owned unless the repository clearly states otherwise.

### enn_torch_dev/debug

`enn_torch_dev/debug` contains pytest-based debug tests for the active-development namespace. It is a test area, not a location for agent instruction Markdown files.

## Stable API and active-development boundaries

Distinguish current implementation, active development, future plan, historical note, and maintainer decision required. Do not present future plans as current behavior. Do not remove existing public APIs, change `__all__`, or alter package boundaries unless explicitly requested and tested.

## Development workflow

Before editing, inspect relevant code, tests, package metadata, and existing docs. Keep changes scoped to the request. For documentation-only work, do not modify production Python code, dependencies, lockfiles, generated artifacts, existing README bodies, or LICENSE body.

## Compatibility expectations

Respect the current package metadata and optional extras in `pyproject.toml`. Do not add dependencies or change lockfiles unless explicitly requested. Do not assume optional backends such as TensorRT, CoreML, TensorFlow, ExecuTorch, NVIDIA Transformer Engine, Intel Extension for PyTorch, pandas, polars, Spark, or safetensors are installed.

Do not assume GPU, CUDA, distributed execution, or child-process execution is available. Check availability before running environment-specific commands.

## Testing baseline

Use `docs/TESTING.md` for scope-specific commands. Report tests that were run, tests not run, dependency-related failures, and CUDA-related skips separately. Do not claim unrun behavior is verified. Do not invent formatter, linter, type-checker, or CI requirements that are not configured in the repository.

## Runtime and artifact safety

Use `docs/RUNTIME_SAFETY.md` before executing code that can write files, allocate accelerator memory, spawn workers, download data, export models, or create checkpoints. Prefer small synthetic tensors. Do not overwrite real data or repository artifacts. Do not add checkpoints, exports, predictions, memmaps, temporary artifacts, secrets, tokens, API keys, credentials, or `.env` files to Git.

## AI-facing documentation maintenance

AI-facing documentation must be kept current.

For every task, before finishing, review whether the change affects any AI-facing documentation. This review is always required, even for small changes.

If the change affects repository structure, architecture, public APIs, package boundaries, configuration, dependencies, test commands, compatibility contracts, runtime safety rules, artifact handling, documented workflows, or current-state classification, update the affected AI-facing documents in the same PR.

Do not defer required AI documentation updates to a follow-up task. Do not edit unrelated AI-facing documents merely to create churn.

## Final reporting requirements

Final reports must include tests or checks that were run and exactly one AI documentation impact result:

```text
AI docs updated:
- <documents updated>
```

or

```text
AI docs impact: none
Reason: <concrete reason>
```
