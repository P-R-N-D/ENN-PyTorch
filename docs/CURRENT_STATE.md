# ENN-PyTorch Current State

Status categories:

- **CURRENT**: implemented in the current repository.
- **ACTIVE DEVELOPMENT**: implemented under development-focused paths with current code evidence such as implemented modules, public exports, or tests. A design document alone is not enough to classify an area as active development.
- **PLANNED**: explicitly described as future work, not current behavior.
- **HISTORICAL**: design-record or background information.
- **EXTERNAL REFERENCE**: supporting technical reference, not implementation source of truth.
- **MAINTAINER DECISION REQUIRED**: not settled by the current repository.

| Area | Status | Current evidence and notes |
|---|---|---|
| `enn_torch` public API | CURRENT | Stable user-facing namespace with lazy modules `core`, `data`, `nn`, `runtime` and workflow helpers including `new_model`, `load_model`, `load_weights`, `new_embedding`, `load_embedding`, `save_model`, `save_embedding`, `train`, and `predict`. |
| `enn_torch/core` | CURRENT | Stable package area for configuration, compatibility, concurrency, datatypes, policies, precision, system, and tensor utilities. |
| `enn_torch/data` | CURRENT | Stable package area exposing `collate`, `nodes`, and `pipeline`. |
| `enn_torch/nn` | CURRENT | Stable package area exposing wrappers, activations, kernels, layers, blocks, graph, and profiler modules. |
| `enn_torch/runtime` | CURRENT | Stable package area exposing autobatch, distributed, IO, main runtime, losses, optimizers, and workflows. Treat long-running, distributed, and artifact-writing behavior with runtime safety rules. |
| `enn_torch_dev/data` | ACTIVE DEVELOPMENT | Development namespace for `DataSchema`, `FieldSpec`, `KeyMapping`, `KVBatch`, `BatchCost`, manifests, TensorDict staging/reading, and SPDL tensor adapter boundaries. |
| `enn_torch_dev/executor` | ACTIVE DEVELOPMENT | Development namespace for `KVStore`, node/subgraph/graph execution, plans, runners, model specs, graph/model/branch builders, tile, stream, global-local, and state routing. |
| `enn_torch_dev/nn` | ACTIVE DEVELOPMENT | Development namespace for blocks, attention, fusion, layers, recurrent components, and related types. |
| `enn_torch_dev/runtime` | ACTIVE DEVELOPMENT | Development namespace for loaders, runtime steps, fault/status records, model/resource cost probes, bounded observed-cost calibration, pure pre-pass admission assessment, opt-in per-attempt pre-pass admission enforcement, device-resolved model/optimizer footprint, resource monitoring, resource capacity and pressure assessment, pure initial batch-budget recommendation, budgeted batching, OOM-class runtime retry, conservative StepResult-observing budget governance with opt-in pressure growth suppression, independent CPU/CUDA sustained-pressure streaks, per-dimension threshold/pass-count/factor overrides, and structured high/trigger/selection/factor provenance driving dimension-aware shrink, finite pass-level runtime orchestration with fixed or pass-scoped provider capacity assessment of all raw-attempt resource samples and per-pass capacity provenance, runtime pass summary inspection with pressure-feedback, structured provenance, pressure-shrink, adjusted-field, and per-dimension streak visibility, bounded in-memory runtime history aggregation of pressure assessment, growth suppression, structured CPU/CUDA high/trigger provenance, adjustment attempts/no-ops, and field-specific actual pressure shrink, bounded lazy multi-pass runtime sessions, fresh per-pass source factories, and end-to-end integration coverage for retry, budget transitions, history retention, identity order, and API boundaries. |
| `enn_torch_dev/debug` | CURRENT | Pytest debug suite for development data, executor, nn, and runtime areas. Do not add agent instruction Markdown files here. |
| Existing `docs/dev_*.md` | HISTORICAL | Implementation-slice design records for `enn_torch_dev`. Their `Next Step`, `Follow-up`, and `Out of Scope` sections are not current task instructions or confirmed roadmap items. |
| Runtime development workflow | CURRENT | `docs/runtime_development_workflow.md` documents the supported bounded single-node composition and its safety boundaries. |
| Existing executor documents | CURRENT | `docs/executor_modes.md` defines tile/stream terminology. `docs/executor_builders.md` is user-facing guidance for executor builder APIs. |
| Notion ENN-PyTorch technical docs | EXTERNAL REFERENCE | Supporting technical reference for structure and terminology only, not implementation source of truth. Current GitHub implementation and tests remain primary evidence. |
| Relationship between `enn_torch` and `enn_torch_dev` | MAINTAINER DECISION REQUIRED | The repository does not clearly settle whether `enn_torch_dev` replaces, runs parallel to, or partially migrates into `enn_torch`. Do not decide this in agent work. |

## Planned items noted in design records

Existing development design records mention future concepts such as ingestion plugins, SPDL pipeline construction, dynamic batching, OOM recovery, AutoGovernor, persistent calibration cache, sharding, distributed resume. Treat these as **PLANNED** only when explicitly requested or implemented; otherwise do not describe them as current functionality.
