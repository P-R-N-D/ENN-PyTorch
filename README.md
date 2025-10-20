# STNet-PyTorch

## Overview
This repository provides a PyTorch implementation of the STNet architecture for joint spatial and temporal modeling, exposing a high-level workflow API for model construction, training, inference, and export utilities. The workflow manages dataset materialization, adaptive loss balancing, FLOP accounting, and throughput reporting so you can focus on configuration and feature preparation.

## Key components
- **Configurable architecture** – `stnet.architecture.network.ModelConfig` defines depth, attention heads, patching strategy, compilation options, and other hyperparameters that tailor the spatio-temporal transformer. Helper schemas such as `PatchConfig` and the `BuildConfig` alias keep patch extraction and compiler hints organized.
- **Modeling type aliases** – `_normalize_modeling_type` interprets spatial (`ss`, `spatial`), temporal (`tt`, `temporal`), and spatio-temporal (`st`, `ts`, `spatiotemporal`, etc.) shorthands so configuration files and user input remain ergonomic.
- **Workflow facade** – `stnet.workflow` re-exports lifecycle helpers such as `new_model`, `save_model`, `load_model`, `train`, `predict`, and multiple export utilities (TorchScript, ONNX, TensorRT, Core ML, ExecuTorch, TensorFlow, LiteRT). Configuration dataclasses (`ModelConfig`, `PatchConfig`) and runtime orchestration (`OpsConfig`) are available from the same namespace for ergonomic imports.
- **Architecture utilities** – `stnet.architecture.module` houses reusable building blocks including `StochasticDepth`, `norm_layer`, `schedule_stochastic_depth`, and the model definitions so dependents import everything from a single entry point.

## Installation
1. Create and activate a Python 3.10+ environment.
2. Install PyTorch that matches your hardware (CUDA, ROCm, XPU, or CPU) by following the official [PyTorch instructions](https://pytorch.org/get-started/locally/).
3. Install STNet-PyTorch and its dependencies:
   ```bash
   pip install -e .
   ```
   Optional exporter extras are available via:
   ```bash
   pip install -e .[export]
   ```
   Additional extras include `ao`, `gds`, `te`, `intel`, `ucx`, `arrow_cuda`, and `scale` as defined in `pyproject.toml`.

## Dependencies
The core runtime depends on:

- `torch>=2.1`
- `torchvision>=0.16`
- `torchdata>=0.11`
- `tensordict>=0.10.0`
- `numpy>=1.24`
- `netifaces>=0.11`
- `pyarrow[flight]>=10`
  - add `pyarrow[cuda]` (or install the `arrow_cuda` extra) when GPU-accelerated Arrow Flight is required
- `tqdm>=4.66`

Optional extras listed in `pyproject.toml` cover exporter stacks (`export`), advanced optimization toolchains (`ao`, `te`, `scale`),
vendor accelerators (`intel`, `ucx`), storage pipelines (`gds`), Arrow GPU acceleration (`arrow_cuda`), and queue backends (`queue`).
Install `stnet-pytorch[queue]` or `pyzmq` manually when the ZeroMQ-based message queue helpers are required.

## Quick start
```python
import torch
from stnet.workflow import (
    ModelConfig,
    new_model,
    save_model,
    load_model,
    train,
    predict,
    to_script,
)

config = ModelConfig(modeling_type="spatiotemporal", depth=64, heads=4)
model = new_model(in_dim=1024, out_shape=(10,), config=config)

features = torch.randn(32, model.in_dim)
labels = torch.randn(32, *model.out_shape)

dataset = {"X": features, "Y": labels}
trained = train(model, dataset, epochs=1, batch_size=8)

infer_batch = {"X": features, "Y": torch.zeros_like(labels)}
predictions = predict(trained, infer_batch)

save_path = save_model(trained, "checkpoints/stnet.pt")
restored = load_model(save_path)

scripted = to_script(restored)  # TorchScript export
```
During training and inference the progress bar reports MB/s, TFLOPS, elapsed time, and completion percentage while distributed workers stay synchronized through the join context. FLOP counters and adaptive loss weights update automatically, and the pipeline keeps dataset schemas and scaling statistics in sync with the provided tensors.

The workflow helpers manage distributed checkpoints, mixed precision, exporter requirements, and memory-mapped datasets internally, letting you focus on preparing feature tensors and configuration hyperparameters.

### Sample workbook configuration

When following `notebook.ipynb` to materialize features from `raw_data.xlsx`, the CUDA profile keeps the model depth at 1152 with larger microbatches while the CPU path dials the depth back to 512 and halves the microbatch size to remain memory efficient. Both flows share the same tokenizer geometry so predictions remain shape-compatible across devices.

## Exporting for inference
Exporter helpers automatically check for optional dependencies and raise informative errors if a backend such as ONNX, TensorFlow, Core ML, TensorRT, LiteRT, or ExecuTorch is unavailable. Install the `export` extra to enable the full conversion toolkit.

## License
**Code** is licensed under **PolyForm Noncommercial 1.0.0**
(SPDX: `PolyForm-Noncommercial-1.0.0`). See `LICENSE`.

**Model weights / datasets** (and other non-code artifacts) are provided
under **CC BY-NC 4.0**. See the "Creative Commons Attribution-NonCommercial 4.0 International" section in `LICENSE`.

Commercial use requires a separate license. Please contact the author.
