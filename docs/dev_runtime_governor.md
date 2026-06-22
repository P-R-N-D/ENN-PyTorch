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

## Policy Validation

`GovernorPolicy` validates static adjustment parameters when it is created:

- `shrink_factor` must be finite and satisfy `0 < shrink_factor < 1`.
- `grow_factor` must be finite and satisfy `grow_factor > 1`.
- `grow_after_successes` must be a positive integer.
- `min_items`, `max_items`, `min_host_bytes`, `max_host_bytes`,
  `min_device_bytes`, and `max_device_bytes` must be `None` or positive
  integers.
- For each field family, `min_*` must not exceed `max_*`.

## Adjustment Rules

OOM shrink and success growth are applied uniformly to every configured budget
field. The governor does not choose host or device bytes based on resource peaks.

- Shrink uses `floor(current * shrink_factor)`.
- Grow uses `ceil(current * grow_factor)`.
- Configured fields are never allowed to fall below `1`.
- Bounds are applied only to configured fields after the shrink/grow calculation.

## Decision Rules

`observe_results(results)` consumes an iterable of `StepResult` objects and
returns a `GovernorDecision`.

Decision priority is deliberately small and predictable:

1. Empty streams keep the current budget.
2. If any result has `StepStatus.OOM_FAULT`, OOM wins and all configured budget
   fields shrink.
3. Mixed streams containing OOM and success or non-OOM faults still shrink.
4. If every observed result is `StepStatus.SUCCESS`, the success streak increases
   by one for the observe call.
5. The success streak does not increase per successful `StepResult`.
6. When `grow_after_successes` is reached, all configured budget fields grow and
   the success streak resets to zero.
7. Non-OOM faults keep the current budget and reset success/OOM streaks.

## State and Decision Records

`RuntimeGovernorState` is a frozen reusable state record with:

- `current_budget`
- `consecutive_successes`
- `consecutive_ooms`
- `last_decision`

`ConservativeRuntimeGovernor` replaces its state with a new
`RuntimeGovernorState` after each observation. It does not mutate a state object
that was passed in from outside.

`GovernorDecision` records the previous and next budget, reason text, observed
statuses, updated streak counters, and resource peaks observed in the result
stream.

## Resource Samples

The governor records peak values from `ResourceSample` objects into the decision
record and reason text:

- `peak_cpu_rss_bytes`
- `peak_cuda_allocated_bytes`
- `peak_cuda_reserved_bytes`
- `peak_cuda_max_allocated_bytes`
- `peak_cuda_max_reserved_bytes`

These peaks are observational only in this slice. They do not drive field
selection, learned tuning, or feedback-loop policy changes.

## Out of Scope

- Full AutoGovernor behavior.
- Learned or model-specific tuning.
- Persistent calibration caches or history databases.
- `ResourceMonitor` feedback-loop tuning.
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
