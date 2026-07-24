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
- Normalize CPU RSS against the smallest known physical or hierarchy-effective cgroup capacity; do not assume host physical memory or a leaf cgroup file is the process limit in containers.
- Do not clamp pressure ratios to `1.0`; ratios above one must remain visible for diagnosis.
- Feed pressure summaries into governor decisions through explicit opt-in policies. Missing or high pressure may suppress growth; pressure may shrink a future budget only when a separately configured sustained-pressure threshold and pass count are both met.
- Never let a single non-OOM pressure sample trigger shrink, and keep OOM/recovered-OOM shrink higher priority than pressure streak handling.
- Map sustained CPU pressure only to `max_host_bytes` and sustained CUDA pressure only to `max_device_bytes`; use `max_items` only when no matching pressured byte budget is configured.
- Record only fields whose values actually changed, and do not label minimum-bound no-ops as pressure shrink.
- Keep yielded or retry-recovered OOM behavior unchanged: it shrinks every configured budget field and leaves pressure-specific field metadata empty.
- When orchestration is given fixed or provider-resolved `ResourceCapacity`, include samples from retry-consumed attempts as well as final results.
- Resolve a `ResourceCapacityProvider` exactly once before consuming each pass source; provider failures and invalid return types must remain visible and must not update governor state.
- Keep the resolved capacity fixed within a pass; do not hide CUDA device mismatches or refresh capacity during retry/split execution.
- Do not add persistent logs, checkpoints, exports, source replay, distributed workers, or automatic tuning to the bounded development workflow without a separately reviewed safety contract.
- Confirm the stable `enn_torch` namespace is unchanged when adding development runtime helpers.

## Artifact handling

Keep generated outputs outside tracked repository paths when possible. If a test must write files, use a temporary directory and remove it when finished. Before final reporting, check `git status --short` for accidental artifacts.
