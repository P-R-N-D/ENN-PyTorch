# Conservative Runtime Governor

`ConservativeRuntimeGovernor` is a small observation/decision layer for the
active-development runtime namespace. It watches an already-produced stream of
`StepResult` records and chooses the `BatchBudget` to use for the next runtime
attempt.

The governor is intentionally conservative. It is not AutoGovernor, does not run
models, does not split batches, does not retry failed batches, and does not probe
costs. It should remain separate from `BudgetedBatcher`, `RuntimeRetryRunner`,
and `RuntimeStep`.

## Public Objects

`enn_torch_dev.runtime` exports:

- `GovernorPolicy`
- `GovernorDecision`
- `RuntimeGovernorState`
- `ConservativeRuntimeGovernor`

The stable `enn_torch` namespace does not expose this development governor.

## Budget Field Semantics

The governor only adjusts `BatchBudget` fields that are already configured.
A configured field is a field whose current value is not `None`:

- `max_items`
- `max_host_bytes`
- `max_device_bytes`

Fields that are `None` stay `None`. Bounds in `GovernorPolicy` do not activate a
previously unconfigured budget field.

Active configured budget values must be positive when used by the governor. This
is stricter than the lower-level `BatchBudget` validation because the governor
needs a positive value to shrink or grow.

When a `ConservativeRuntimeGovernor` is created, the active budget from either
`budget` or `state.current_budget` must already be inside the configured policy
bounds. Configured fields below a policy minimum or above a policy maximum are
rejected at construction time. `None` fields are not validated against bounds and
are not activated by bounds. This prevents an OOM shrink from increasing a budget
or a success grow from decreasing one through clamping.

## Policy Validation

`GovernorPolicy` validates static adjustment parameters when it is created:

- `shrink_factor` must be finite and satisfy `0 < shrink_factor < 1`.
- `grow_factor` must be finite and satisfy `grow_factor > 1`.
- `grow_after_successes` must be a positive integer.
- `min_items`, `max_items`, `min_host_bytes`, `max_host_bytes`,
  `min_device_bytes`, and `max_device_bytes` must be `None` or positive
  integers.
- For each field family, `min_*` must not exceed `max_*`.
- `max_pressure_ratio_for_growth` is optional; when configured it must be finite and satisfy `0 < value <= 1`.
- `min_pressure_ratio_for_shrink` is optional; when configured it must be finite and satisfy `0 < value <= 1`.
- `shrink_after_pressure_passes` must be a positive integer.
- When both pressure thresholds are configured, the growth threshold must not exceed the shrink threshold.

## Adjustment Rules

OOM shrink and success growth are applied uniformly to every configured budget
field. The governor does not choose host or device bytes based on resource peaks.

- Shrink uses `floor(current * shrink_factor)`.
- Grow uses `ceil(current * grow_factor)`.
- Configured fields are never allowed to fall below `1`.
- Bounds are applied only to configured fields after the shrink/grow calculation.

## Decision Rules

`observe_results(results, *, recovered_oom=False, pressure_summary=None)` consumes an iterable of
`StepResult` objects and returns a `GovernorDecision`. The method streams the
iterable once and accumulates only statuses, resource peaks, and small decision
flags. It does not keep a list or tuple of `StepResult` objects, so `store` and
`loss` references from results are not preserved in the decision/state records.

`recovered_oom=True` is an explicit signal from an outer retry layer that an
OOM-class failure was recovered before the governor observed final results. It is
treated as conservative budget pressure rather than success evidence.

Decision priority is deliberately small and predictable:

1. Empty streams keep the current budget unless `recovered_oom=True`.
2. If any result has `StepStatus.OOM_FAULT`, OOM wins and all configured budget
   fields shrink.
3. `recovered_oom=True` also shrinks all configured budget fields, even when all
   observed statuses are `SUCCESS` or the observed stream is empty.
4. Mixed streams containing OOM and success or non-OOM faults still shrink.
5. If every observed result is `StepStatus.SUCCESS` and `recovered_oom=False`,
   an optional sustained-pressure shrink guard is evaluated before success evidence is
   accumulated.
6. With `max_pressure_ratio_for_growth=None`, pressure does not change the existing
   success-growth behavior.
7. With the guard enabled, a missing pressure summary, an all-unknown summary, or
   `pressure_summary.max_observed_ratio >= max_pressure_ratio_for_growth` keeps the
   budget and resets the success streak to zero.
8. A known maximum pressure ratio below the configured limit allows the success
   streak to increase by one for the observe call.
9. With `min_pressure_ratio_for_shrink` configured, a known maximum ratio at or
   above that threshold increments a high-pressure streak, suppresses growth, and
   shrinks the next budget only when `shrink_after_pressure_passes` is reached.
   Low, unavailable, empty, or faulted passes reset this streak.
10. Retry-recovered OOM is not success evidence and does not increase the success
   streak.
11. The success streak does not increase per successful `StepResult`.
12. When `grow_after_successes` is reached, all configured budget fields grow and
   the success streak resets to zero.
13. Non-OOM faults keep the current budget and reset success/OOM/high-pressure streaks.

## State and Decision Records

`RuntimeGovernorState` is a frozen reusable state record with:

- `current_budget`
- `consecutive_successes`
- `consecutive_ooms`
- `last_decision`
- `consecutive_high_pressure_passes`

`ConservativeRuntimeGovernor` replaces its state with a new
`RuntimeGovernorState` after each observation. It does not mutate a state object
that was passed in from outside.

`GovernorDecision` records the previous and next budget, reason text, observed
statuses, updated streak counters, resource peaks, the supplied pressure summary,
whether pressure suppressed success growth, and whether sustained pressure actually
changed the next budget.

## Resource Samples

The governor records peak values from `ResourceSample` objects into the decision
record and reason text:

- `peak_cpu_rss_bytes`
- `peak_cuda_allocated_bytes`
- `peak_cuda_reserved_bytes`
- `peak_cuda_max_allocated_bytes`
- `peak_cuda_max_reserved_bytes`

Raw byte peaks remain observational. A separately supplied
`ResourcePressureSummary` may suppress success-driven growth when the opt-in policy
limit is configured. It may shrink only a future budget when the separately opt-in
sustained-pressure threshold and pass count are met; OOM and recovered OOM retain
higher priority.

## Relationship to Orchestration

A finite pass-level orchestration helper can feed `GovernorDecision.next_budget`
back into `BudgetedBatcher` on a later pass. That orchestration boundary is
described in `docs/dev_runtime_orchestration.md`. When callers explicitly
configure a fixed `ResourceCapacity`, the orchestrator can assess all raw-attempt
resource samples and supply the resulting summary to the governor. The governor
itself remains an observation/decision object and does not run batches, retry,
execute models, discover capacity, or construct a pressure summary.

## Out of Scope

- Full AutoGovernor behavior.
- Learned or model-specific tuning.
- Persistent calibration caches or history databases.
- Field-specific tuning.
- Automatic pressure-summary construction inside the governor.
- Automatic capacity discovery or refresh inside the orchestrator.
- `ModelCostProbe`-driven policy changes.
- SPDL queue-depth tuning.
- Device transfer.
- AMP or precision fallback.
- Optimizer rollback or training semantic recovery.
- Distributed coordination.

## Test Commands

```bash
python -m pytest enn_torch_dev/debug/runtime/test_runtime_governor.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_retry.py -q
python -m pytest enn_torch_dev/debug/runtime/test_budgeted_batcher.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug/data -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```
