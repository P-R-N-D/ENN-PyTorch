# ENN-PyTorch Runtime and Artifact Safety

Use this document before running code that can allocate accelerator memory, write files, spawn workers, download data, train, export, checkpoint, or create artifacts.

## Default execution rules

- Use small synthetic tensors as default test inputs.
- Do not overwrite real data or artifacts in the repository.
- Run staging tests only in temporary directories.
- Use `overwrite=True` only for temporary paths created by the agent for the current task.
- Do not use full notebook execution as baseline validation.
- Do not use long-running training as baseline validation.
- Do not run CUDA code without checking CUDA availability first.
- Clean up distributed workers and child processes after tests.
- Do not automatically download external data or models.
- Do not assume TensorRT, CoreML, TensorFlow, ExecuTorch, NVIDIA Transformer Engine, Intel Extension for PyTorch, pandas, polars, Spark, safetensors, or other optional backends are installed.
- Do not add checkpoints, exports, predictions, memmaps, or temporary artifacts to Git.
- Do not use `sudo`, administrator elevation, or destructive root-level operations for repository tasks.
- Root-based container workspaces are allowed for safe repository-local checks such as `git diff --check`, `python -m py_compile`, and `python -m pytest`, as long as the task does not change system paths, install global packages, modify ownership/permissions, or touch data outside the workspace.
- Do not record secrets, tokens, API keys, or real credentials.
- Do not create `.env` files.

## Development runtime integration

- Use finite inner `KVBatch` pass sources and a positive `max_passes` session bound.
- Use a positive `RuntimePassHistory.max_records` bound; do not add unbounded in-memory history.
- A `RuntimePassSourceFactory` must create a fresh finite source per call; do not cache or silently replay consumed iterators inside the session.
- Keep factory creation lazy and bounded by `ConservativeRuntimeSession.max_passes`; factory exceptions must remain visible to the caller.
- Consume session records incrementally. Avoid collecting long sessions into a list outside small synthetic tests.
- Keep baseline integration tests CPU-only with small synthetic tensors.
- Treat retry-recovered OOM as a budget signal, not proof that an unrestricted workload is safe.
- Treat missing capacity or observation values as unknown, not zero utilization.
- Keep initial batch-budget recommendation pure: it must not execute a model, consume a source, mutate governor/history state, or admit a pass.
- Require concrete device-resolved model and optimizer footprint provenance: use exact `cpu` and `cuda:<index>` keys, reject bare `cuda`, and do not guess which CUDA device owns non-zero static tensor bytes.
- Require positive reference device cost to be explicitly bound to the matching `cuda:<index>` capacity; never apply aggregate non-CPU bytes to an arbitrary CUDA device, and reject unsupported, bare, mismatched, or multiple non-zero device provenance.
- Treat a zero total byte cost as known and non-limiting even when the reference item count is missing or zero, while preserving positive totals with unknown item count as unknown per-item cost.
- Use ceiling division for reference per-item costs, never clamp an insufficient computed limit upward to `min_items`, and require an explicit `fallback_max_items` when a relevant dimension remains unknown.
- Preserve the original capacity, reference batch cost, resolved policy, and normalized reference device provenance in the recommendation so its static calculation can be audited.
- Treat a recommended initial budget as a conservative starting point, not proof that unobserved activations, allocator overhead, or the next pass are admissible.
- Calibrate only completed `ModelCost` records; the calibrator must not execute a model, consume a source, mutate governor/history state, or retain `StepResult`, `ResourceSample`, tensor, store, or loss objects.
- Use only successful positive-batch observations for numeric calibration. Count fault and zero-batch observations separately, clamp negative deltas to zero, and preserve unknown values instead of fabricating costs.
- Keep one observed-cost profile bound to at most one concrete CUDA device and cap retained phase-pair accumulators with `max_phase_pairs`; do not merge mismatched devices or grow calibration state without a bound.
- Resolve `ModelCost` CUDA provenance only when every CUDA-bearing sample has the same bool-excluding, non-negative integer index, and compute CUDA deltas only between endpoints with that same concrete index; never treat `None == None` as a device match or infer the current CUDA device for missing or invalid provenance.
- Treat observed-cost envelopes as prior execution evidence, not admission proof or permission to bypass retry, pressure, or capacity checks.
- Keep pre-pass admission assessment pure: it may combine one fixed `ResourceCapacity`, one execution-immediate `ResourceSample`, one `ObservedCostProfile`, and a candidate batch size, but it must not consume a source, execute a model, split or skip a batch, invoke retry, or mutate runtime state.
- Preserve `REJECT`, `UNKNOWN`, and `ADMIT` as distinct outcomes with `REJECT` precedence; never treat missing capacity, baseline usage, profile evidence, or profile sample floor as a known zero.
- Require matching concrete CUDA provenance across capacity, every CUDA-bearing baseline value, and every known CUDA profile envelope; do not infer the current device or merge devices.
- Use current CUDA allocated/reserved values as baselines and the larger known direct/peak calibrated delta as the per-item increment; do not add historical baseline max counters as current usage.
- Treat known phase-profile CUDA metrics as CUDA relevance and provenance evidence, but never add them to total projection costs; keep an applicable dimension `UNKNOWN` when its current usage or total increment is missing rather than treating phase-only evidence as non-applicable.
- Treat a pre-pass assessment as structured evidence only until a separately reviewed opt-in execution gate defines fail-open/fail-closed and split/skip behavior.
- Keep admission enforcement explicitly opt-in. Resolve capacity once per pass, then sample and assess immediately before every original or retry-split execution attempt.
- Always block `REJECT`. Block `UNKNOWN` by default; allow it only through an explicit `AdmissionUnknownAction.ALLOW` configuration. Never allow that option to override `REJECT`.
- Represent an admission block with `PrePassAdmissionBlocked`, not a synthetic `StepStatus`; store only the immutable assessment in the exception's custom payload. Do not claim that the exception object's transitive object graph is limited to the assessment, because a normal Python traceback may reference execution frames and their batch, baseline sample, source, or runtime-wrapper locals. Memory-sensitive callers should extract `exc.assessment`, preserve only a lightweight textual traceback when needed, and release the exception and traceback rather than caching them.
- Preserve retry safety by passing the wrapped runtime step optimizer through the admission wrapper so training-time OOM retry restrictions remain unchanged.
- A blocked pass must not update governor state or produce a `RuntimePassResult`. Earlier candidates from the same pass may already have executed before a later candidate blocks; do not claim rollback or atomic pass execution.
- Trust split provenance only from the orchestrator's admission-aware pre-execution wrapper. Never infer an admission split request from a public `PrePassAdmissionBlocked` raised by an arbitrary runtime step; that exception is terminal because execution side effects may already have occurred.
- Keep admission reject splitting disabled unless an explicit `AdmissionSplitPolicy` is supplied. Split only `REJECT`, never `UNKNOWN`, and use only a positive finite `max_admissible_items` that is smaller than the current candidate.
- Bound admission recovery independently from OOM retry with a non-negative split depth, positive `min_items`, and maximum parts per rejected parent. Refuse recovery when all rows cannot be partitioned into children within both the assessed target and `min_items`.
- Preserve row identity and order across admission splits. Reassess every child with a fresh baseline while keeping pass-scoped capacity fixed. Do not reuse OOM `split_factor` as an admission guess.
- Clear the traceback only for an internal block that is successfully converted into child recursion. Terminal blocks retain normal traceback behavior and must remain visible.
- Do not automatically skip, replay, rollback, or tune a blocked candidate. Do not feed recovered admission rejection into the governor by default; summary and history may record only bounded scalar provenance.
- Reduce completed-pass admission evidence to scalar counts, a recovery flag, and an optional minimum recovered item limit before storing summaries or history. Do not retain raw admission assessments, dimensions, warnings, exceptions, batches, samples, sources, stores, losses, or tensors in summary/history records.
- Validate that every completed-pass `REJECT` has a bool-excluding positive `max_admissible_items` smaller than its assessed batch size; terminal or malformed rejects must not be silently reported as recovered.
- Recompute admission history aggregates only from the currently retained summary window.
- Keep admission governor feedback opt-in. It may reset clean-success growth using a positive recovered item limit, but must not undo OOM shrink, pressure shrink, or directly cap/create `max_items` in this slice.
- Normalize CPU RSS against the smallest known physical or hierarchy-effective cgroup capacity; do not assume host physical memory or a leaf cgroup file is the process limit in containers.
- Do not clamp pressure ratios to `1.0`; ratios above one must remain visible for diagnosis.
- Feed pressure summaries into governor decisions through explicit opt-in policies. Missing or high pressure may suppress growth; pressure may shrink a future budget only when the effective sustained-pressure threshold and required pass count for that dimension are both met.
- Dimension-specific shrink thresholds, required pass counts, and pressure shrink factors override the common policy independently; unset overrides fall back to the common values, and a dimension with no effective threshold remains disabled.
- Keep `max_pressure_ratio_for_growth` less than or equal to every active effective CPU/CUDA shrink threshold so growth suppression cannot begin above a shrink trigger.
- Never let a single non-OOM pressure sample trigger shrink, and keep OOM/recovered-OOM shrink higher priority than pressure streak handling.
- Map sustained CPU pressure only to `max_host_bytes` and sustained CUDA pressure only to `max_device_bytes`; apply each triggered dimension's effective factor to its matching byte budget.
- Use `max_items` only when no matching triggered byte budget is configured; use the triggered dimension's factor, or the smaller factor when CPU and CUDA share the same fallback.
- Track CPU and CUDA pressure persistence independently; alternating pressure dimensions must not combine into one sustained streak.
- Reset only the dimension observed as low or unknown during a successful assessed pass, while preserving the other dimension's incomplete high-pressure streak.
- Reset both dimension streaks after fully unavailable pressure, empty passes, non-OOM faults, yielded OOM, or retry-recovered OOM.
- Use structured pressure provenance for automation and inspection; do not parse
  `GovernorDecision.reason` to recover high dimensions, trigger dimensions,
  selected fields, or applied factors.
- Keep OOM and retry-recovered OOM pressure-specific provenance tuples empty.
- Recompute history provenance counters only from the currently retained
  `RuntimePassSummary` window; trimming must remove every contribution from the
  discarded summary.
- Count dimension high/trigger events per dimension but count adjustment attempts,
  full no-ops, and triggers without budgets at most once per pass. A partial
  adjustment is not a full no-op. Do not infer structured provenance from
  pressure ratios or OOM status when the provenance tuples are empty.
- Record only fields whose values actually changed, and do not label minimum-bound no-ops as pressure shrink.
- Keep yielded or retry-recovered OOM behavior unchanged: it uses the common `shrink_factor`, shrinks every configured budget field and leaves pressure-specific field metadata empty.
- When orchestration is given fixed or provider-resolved `ResourceCapacity`, include samples from retry-consumed attempts as well as final results.
- Resolve a `ResourceCapacityProvider` exactly once before consuming each pass source; provider failures and invalid return types must remain visible and must not update governor state.
- Keep the resolved capacity fixed within a pass; do not hide CUDA device mismatches or refresh capacity during retry/split execution.
- Do not add persistent logs, checkpoints, exports, source replay, distributed workers, or automatic tuning to the bounded development workflow without a separately reviewed safety contract.
- Confirm the stable `enn_torch` namespace is unchanged when adding development runtime helpers.

## Artifact handling

Keep generated outputs outside tracked repository paths when possible. If a test must write files, use a temporary directory and remove it when finished. Before final reporting, check `git status --short` for accidental artifacts.
