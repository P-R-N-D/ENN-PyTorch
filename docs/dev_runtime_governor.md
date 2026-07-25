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
- `cpu_pressure_shrink_factor` and `cuda_pressure_shrink_factor` must be `None` or finite and satisfy `0 < value < 1`; unset overrides fall back to `shrink_factor`.
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
6. When `max_pressure_ratio_for_growth`, the common shrink threshold, and both
   dimension-specific shrink thresholds are `None`, pressure does not change the
   existing success-growth behavior.
7. With the guard enabled, a missing pressure summary, an all-unknown summary, or
   `pressure_summary.max_observed_ratio >= max_pressure_ratio_for_growth` keeps the
   budget and resets the success streak to zero.
8. A known maximum pressure ratio below the configured limit allows the success
   streak to increase by one for the observe call.
9. The CPU and CUDA shrink threshold and required pass count resolve independently.
   `min_cpu_pressure_ratio_for_shrink` and
   `min_cuda_pressure_ratio_for_shrink` override the common
   `min_pressure_ratio_for_shrink` value when configured.
   `cpu_shrink_after_pressure_passes` and
   `cuda_shrink_after_pressure_passes` override the common
   `shrink_after_pressure_passes` value.
   `cpu_pressure_shrink_factor` and `cuda_pressure_shrink_factor` override the
   common `shrink_factor` only for sustained-pressure adjustments. A dimension
   with no effective threshold is disabled. A ratio at or above an effective threshold increments only the
   matching pressure streak, suppresses growth, and shrinks the next budget only
   when that dimension reaches its effective required pass count.
   CPU pressure selects `max_host_bytes`; any CUDA allocated/reserved/max ratio
   selects `max_device_bytes`. Each matching byte budget uses its dimension's
   effective shrink factor. If no matching byte budget is configured, `max_items`
   is the fallback and uses the triggered dimension's factor; when both dimensions
   trigger the shared fallback, the smaller factor is used. When at least one
   matching byte budget is configured, `max_items` remains unchanged. A low or unknown observation resets
   only that dimension's streak; a fully unavailable summary, empty pass, fault,
   yielded OOM, or retry-recovered OOM resets both dimension streaks.
   A dimension that reaches the threshold resets after its shrink decision without
   clearing the other dimension's incomplete streak.
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
- `consecutive_cpu_pressure_passes`
- `consecutive_cuda_pressure_passes`

`ConservativeRuntimeGovernor` replaces its state with a new
`RuntimeGovernorState` after each observation. It does not mutate a state object
that was passed in from outside.

`GovernorDecision` records the previous and next budget, reason text, observed
statuses, updated streak counters, resource peaks, the supplied pressure summary,
whether pressure suppressed success growth, whether sustained pressure actually
changed the next budget, and the ordered tuple of budget fields whose values
actually changed because of pressure. `consecutive_high_pressure_passes` remains a
compatibility aggregate equal to the maximum of the CPU and CUDA streaks in every
new decision and state.

For compatibility with state constructed before dimension-specific streak fields
existed, a positive legacy `consecutive_high_pressure_passes` value is inherited
only when exactly one currently high dimension is observed and both new streak
fields are zero. When CPU and CUDA are both high, the aggregate cannot identify its
source dimension, so neither inherits it and both streaks start at one for the
current pass. After the next observation, the governor emits explicit CPU/CUDA
streaks and recomputes the compatibility aggregate.

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

Sustained-pressure shrink is dimension-aware. CPU RSS pressure maps to the host
byte budget, while every CUDA pressure ratio maps to the device byte budget.
`max_items` is used only when none of the triggered dimensions has a configured
matching byte budget. OOM and retry-recovered OOM continue to shrink every
configured budget field and do not populate the pressure-specific field tuple.
OOM paths always use the common `shrink_factor`; pressure-specific overrides apply
only to sustained-pressure decisions.
CPU and CUDA persistence, effective thresholds, required pass counts, and
pressure shrink factors are
resolved independently. An unset dimension override falls back to the common
policy value; when neither a dimension override nor the common threshold is set,
that dimension's sustained-pressure shrink is disabled. Alternating CPU-only and
CUDA-only high-pressure passes cannot combine into a sustained-pressure shrink.
When both dimensions are continuously high, each can reach its own threshold and
required pass count independently.
Triggered decision reasons report only the current ratios, effective threshold
policies, and effective pressure shrink factors for dimensions that reached the threshold; they do not describe the
summary-wide maximum ratio as having persisted for the full streak.

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
- Learned field weights.
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
