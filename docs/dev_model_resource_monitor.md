# Development Model Footprint and Resource Monitor

This document describes the fourth data/runtime rewrite slice for
`enn_torch_dev`.

## Goal

Previous slices established:

```text
DataSchema / KVBatch
  -> TensorDict staging / reader
  -> PlainLoader
  -> RuntimeStep
  -> StepResult / fault classification
```

This slice adds the first observation layer:

```text
GraphExecutor / nn.Module
  -> ModelFootprint

RuntimeStep
  -> ResourceMonitor
  -> StepResult.resource_samples
```

The goal is not dynamic batching yet. The goal is to measure stable facts that
future `ModelCostProbe`, `BudgetedBatcher`, `AutoGovernor`, and OOM recovery can
use.

## ModelFootprint

`ModelFootprint.from_module(module)` computes static model size information for
any `torch.nn.Module`. `GraphExecutor` is also an `nn.Module`, so it can be
measured directly.

Recorded values include:

- parameter count;
- trainable parameter count;
- buffer count;
- parameter bytes;
- trainable parameter bytes;
- buffer bytes;
- total model bytes;
- dtype-grouped parameter, buffer, and byte counts.

Shared parameter or buffer objects are counted once.

## OptimizerFootprint

`OptimizerFootprint.from_optimizer(optimizer)` computes optimizer state tensor
size.

Optimizer state can be empty before the first backward/step. That is valid and
must not be treated as an error.

Recorded values include:

- state tensor count;
- state bytes;
- parameter group count;
- dtype-grouped state tensor and byte counts.

## ResourceMonitor

`ResourceMonitor` creates lightweight CPU/CUDA memory snapshots.

Each `ResourceSample` records:

- timestamp in nanoseconds;
- phase label;
- process CPU RSS bytes when available;
- CUDA availability;
- CUDA device index;
- CUDA allocated bytes;
- CUDA reserved bytes;
- CUDA max allocated bytes;
- CUDA max reserved bytes.

CPU RSS is read without adding a hard `psutil` dependency. If the platform does
not expose the current process RSS through `/proc/self/statm`, the value is
`None`.

CUDA fields are safe on CPU-only machines. If CUDA is unavailable, CUDA-specific
memory fields are `None`.

`ResourceMonitor.capacity()` returns a `ResourceCapacity` snapshot with total
physical CPU memory when `os.sysconf(...)` exposes it and total CUDA device memory
when `torch.cuda.get_device_properties(...)` succeeds. Capacity lookup failures
are represented as `None`; they are not execution faults.

Capacity and usage samples remain separate records. The pure
`assess_resource_pressure(...)` helper described in
`docs/dev_runtime_pressure.md` combines them without changing governor policy.

## RuntimeStep Integration

`RuntimeStep` accepts an optional `resource_monitor`:

```python
RuntimeStep(
    executor,
    schema=schema,
    resource_monitor=ResourceMonitor(),
)
```

When provided, `RuntimeStep` records samples into
`StepResult.resource_samples`.

Current sample phase labels are intentionally minimal:

- `before_step`;
- `after_to_store`;
- `after_zero_grad`;
- `after_forward`;
- `after_loss`;
- `after_backward`;
- `after_optimizer`.

On faults, samples collected up to the failing phase are preserved in the
returned `StepResult`.

## Out of Scope

- Dynamic batch size selection.
- OOM recovery.
- Batch split retry.
- AMP fallback.
- Precision policy.
- SPDL integration.
- DataCostProbe.
- ModelCostProbe.
- Automatic governor changes based on pressure.
- AutoGovernor.
- Telemetry JSON writer.
- `run_profile.json`.
- GPU-specific presets.
- T4/L40S/B200/GB10 hardcoding.

## Test Commands

```bash
python -m pytest enn_torch_dev/debug/runtime/test_model_footprint.py -q
python -m pytest enn_torch_dev/debug/runtime/test_resource_monitor.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_pressure.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_step_resources.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug -q
```

## Next Step

The next slice should add `DataCostProbe` and `ModelCostProbe`.

Those probes can consume model footprint and resource samples, then estimate
per-row data cost and per-batch activation/runtime cost without hardcoding a GPU
profile.
