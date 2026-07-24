# Runtime Resource Capacity Provider

`ResourceCapacityProvider` is the active-development protocol used to resolve one
`ResourceCapacity` snapshot at the start of a finite runtime pass.

## Public Object

`enn_torch_dev.runtime` exports `ResourceCapacityProvider`. The stable
`enn_torch` namespace does not expose it.

## Contract

A provider implements:

```python
class ResourceCapacityProvider(Protocol):
    def capacity(self) -> ResourceCapacity:
        ...
```

`ConservativeRuntimeOrchestrator` accepts either:

- `resource_capacity=<fixed snapshot>`; or
- `resource_capacity_provider=<provider>`.

The two options are mutually exclusive. When a provider is configured:

1. `capacity()` is called exactly once for each `run_pass(...)` call;
2. the call occurs after source type validation but before source consumption;
3. the returned capacity remains fixed for every batch, retry, and split attempt
   within that pass;
4. the resolved capacity is recorded in `RuntimePassResult` and
   `RuntimePassSummary` as scalar provenance;
5. provider exceptions and invalid return types propagate without updating the
   governor or consuming the pass source.

`ResourceMonitor` already provides a compatible `capacity()` method and therefore
satisfies this protocol without an adapter.

## Safety Boundary

The provider contract does not:

- create a `ResourceMonitor` automatically;
- refresh capacity during a pass;
- inspect real-time free memory;
- reserve memory or perform admission control;
- trigger pressure-based budget shrink;
- persist telemetry.

## Test Commands

```bash
python -m pytest enn_torch_dev/debug/runtime/test_runtime_capacity_provider.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_orchestration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_summary.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```
