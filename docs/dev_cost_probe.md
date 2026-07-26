# Development Cost Probe

This document describes the fifth data/runtime rewrite slice for
`enn_torch_dev`.

## Goal

Previous slices established:

```text
DataSchema / KVBatch
  -> TensorDict staging / reader
  -> PlainLoader
  -> RuntimeStep
  -> StepResult / fault classification
  -> ModelFootprint / ResourceMonitor
```

This slice adds the first cost probe layer:

```text
KVBatch / TensorDict
  -> DataCostProbe

StepResult.resource_samples
  -> ModelCostProbe
```

The goal is not automatic batch sizing yet. The goal is to turn tensor batches
and runtime resource samples into stable cost records that future
`BudgetedBatcher`, `AutoGovernor`, and calibration cache components can consume.

## DataCostProbe

`DataCostProbe` estimates tensor memory cost for hot-path tensor batches.

Supported inputs are:

- `KVBatch` through `estimate_kvbatch(batch)`;
- `TensorDictBase` through `estimate_tensordict(td)`;
- a small tensor mapping through `estimate_mapping(mapping, batch_size=...)`.

Recorded values include:

- batch size;
- unique tensor count;
- total tensor bytes;
- bytes per row;
- per-tensor key, dtype, shape, numel, element size, bytes, and device;
- dtype-grouped byte counts;
- device-grouped byte counts.

Tensor storage aliases are counted once. Shared storage is charged by the full
backing storage bytes for the first occurrence, so sliced views do not
underestimate the storage cost and repeated aliases do not double-count it.

Nested `TensorDict` values are traversed recursively using dotted keys such as
`nested.mask`.

When `batch_size == 0`, `bytes_per_row` is `None` so callers do not accidentally
hide division-by-zero semantics behind a synthetic value.

Non-tensor values in mappings are ignored. They are not part of the tensor hot
path cost model.

## ModelCostProbe

`ModelCostProbe` estimates runtime memory deltas from a `StepResult`.

It reads `StepResult.resource_samples` and computes deltas between adjacent
samples. For example:

```text
before_step -> after_to_store
after_to_store -> after_zero_grad
after_zero_grad -> after_forward
after_forward -> after_loss
after_loss -> after_backward
after_backward -> after_optimizer
```

Forward-only steps naturally contain fewer samples:

```text
before_step -> after_to_store
after_to_store -> after_forward
```

Faulted steps are allowed to contain a truncated sample sequence. The probe only
computes deltas for available adjacent pairs.

Recorded values include:

- step status;
- batch size;
- row count;
- total CPU RSS delta;
- total CUDA allocated delta;
- total CUDA reserved delta;
- total CUDA max allocated delta;
- total CUDA max reserved delta;
- per-phase resource deltas;
- one concrete `cuda_device_index` when all CUDA-bearing samples identify the same device.

If a field is unavailable in either endpoint sample, the corresponding delta is
`None`. If CUDA-bearing samples identify different devices, CUDA deltas that cross
devices remain unknown and `ModelCost.cuda_device_index` is `None`.

## Out of Scope

- Dynamic batch size selection.
- BudgetedBatcher.
- AutoGovernor.
- OOM recovery.
- Batch split retry.
- AMP or precision fallback.
- SPDL integration.
- GPU-specific profile presets.
- Calibration cache.
- Telemetry JSON writer.
- `run_profile.json`.
- DataFrame, Arrow, WebDataset, or Hugging Face ingestion cost models.

## Test Commands

```bash
python -m pytest enn_torch_dev/debug/runtime/test_data_cost_probe.py -q
python -m pytest enn_torch_dev/debug/runtime/test_model_cost_probe.py -q
python -m pytest enn_torch_dev/debug/runtime/test_cost_probe_integration.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug/data -q
python -m pytest enn_torch_dev/debug -q
```

## Follow-up

`ObservedCostCalibrator` now consumes completed `ModelCost` records and reduces
successful observations to a bounded per-item envelope without retaining raw
`StepResult` or `ResourceSample` objects. Persistent calibration caches and
automatic admission remain outside the cost probe layer.
