# STNet-PyTorch

## Overview
This repository provides a PyTorch implementation of the STNet architecture for joint spatial and temporal modeling, exposing a high-level runtime API for model construction, training, inference, and export utilities. The runtime manages dataset materialization, adaptive loss balancing, FLOP accounting, and throughput reporting so you can focus on configuration and feature preparation.

## Key components
- **Configurable architecture** – `stnet.config.ModelConfig` defines depth, attention heads, patching strategy, and compiler hints through a single `compile_mode` string (default: `"disabled"`) instead of a boolean flag. Helper schemas such as `PatchConfig` and the `build_config`/`coerce_build_config` aliases keep patch extraction and compilation options organized. Runtime helpers re-export these dataclasses so existing imports from `stnet.runtime` continue to function.
- **Modeling type aliases** – `_normalize_modeling_type` interprets spatial (`ss`, `spatial`), temporal (`tt`, `temporal`), and spatio-temporal (`st`, `ts`, `spatiotemporal`, etc.) shorthands so configuration files and user input remain ergonomic.
- **Runtime facade** – `stnet.runtime` provides lifecycle helpers such as `new_model`, `save_model`, `load_model`, `train`/`learn`, `predict`/`infer`, and exporter shims (TorchScript, ONNX, TensorRT, Core ML, ExecuTorch, TensorFlow, LiteRT). The runtime consumes `ModelConfig`, `PatchConfig`, and `RuntimeConfig` from the unified `stnet.config` module for consistent configuration management, while orchestration internals live under `stnet.runtime.launch` for direct use when needed.
- **Architecture utilities** – `stnet.model` contains the `Root` model, encoder blocks, and building blocks such as `norm_layer`, `CrossAttention`, and `PatchAttention`, while lower-level primitives live in `stnet.model.layers` for reuse across modules.
- **Data transforms** – reusable preprocessing helpers are located under `stnet.data.transforms`, consolidating the former utilities in a single data namespace.
- **Thread load balancer** – dataloader workers automatically pin to allowed CPUs, request OpenMP `proc_bind(spread)` when available, and dynamically retune PyTorch intra/inter-op thread counts to avoid oversubscription.

## Installation
1. Create and activate a Python 3.10+ environment.
2. Install PyTorch that matches your hardware (CUDA, ROCm, XPU, or CPU) by following the official [PyTorch instructions](https://pytorch.org/get-started/locally/).
3. Install STNet-PyTorch and its dependencies:
   ```bash
   pip install -e .
   ```
   Optional exporter extras are available via:
   ```bash
   pip install -e .[servable]
   ```
   Additional extras include `pandas`, `polars`, `excel`, `spark`, `optimization`, `nvidia_gds_cu12`, `nvidia_gds_cu13`, `nvidia_te_cu12`, `nvidia_te_cu13`, `nvidia_rdma_cu12`, `nvidia_rdma_cu13`, `intel_ai`, `compiler`, and `torchscale` as defined in `pyproject.toml`.

## Dependencies
The core runtime depends on:

- `torch>=2.7.0`
- `torchdata>=0.11.0`
- `tensordict>=0.8.0`
- `numpy>=2.2.5`
- `netifaces>=0.11.0`
- `tqdm>=4.67.1`

Optional extras listed in `pyproject.toml` include:
- dataframe integrations (`pandas`, `polars`)
- spreadsheet tooling (`excel`)
- Spark pipelines (`spark`)
- advanced optimization toolchains (`optimization`)
- vendor accelerators (`intel_ai`, `nvidia_te_cu12`, `nvidia_te_cu13`)
- storage pipelines (`nvidia_gds_cu12`, `nvidia_gds_cu13`)
- distributed fabrics (`nvidia_rdma_cu12`, `nvidia_rdma_cu13`)
- retention-focused research modules (`torchscale`)
- compiler niceties (`compiler`, which installs Triton kernels on Linux/x86_64 hosts)

Install the `service` extra to enable the exporter stack (ONNX, TensorRT, Core ML, ExecuTorch, TensorFlow, LiteRT).

### Compiler configuration

`ModelConfig.compile_mode` accepts the same modes as `torch.compile` (for example `"default"`, `"reduce-overhead"`, or `"max-autotune"`).
The runtime now treats `"disabled"`, `"none"`, or an empty string as an explicit request to skip compilation, so you no longer need the
legacy `enable_compilation` flag. The helper in `stnet.utils.optimization.compile` normalizes the value, trims whitespace, and avoids
calling `torch.compile` when compilation is disabled or unsupported.

## Quick start
```python
import torch
from stnet import (
    PatchConfig,
    build_config,
    new_model,
    load_model,
    save_model,
    learn,
    infer,
)

patch = PatchConfig(is_cube=True, grid_size_3d=(10, 10, 1), patch_size_3d=(1, 1, 1))
config = build_config(
    modeling_type="spatiotemporal",
    depth=64,
    heads=4,
    patch=patch,
    compile_mode="default",  # set to "disabled" (default) to keep eager execution
)
model = new_model(in_dim=1024, out_shape=(10,), config=config)

features = torch.randn(32, model.in_dim)
labels = torch.randn(32, *model.out_shape)

dataset = {"X": features, "Y": labels}
trained = learn(model, dataset, epochs=1, batch_size=8)

infer_batch = {"X": features, "Y": torch.zeros_like(labels)}
predictions = infer(trained, infer_batch)

save_path = save_model(trained, "checkpoints/stnet.pt")
restored = load_model(save_path)

restored.eval()
with torch.inference_mode():
    scripted_output, _ = restored(features)
```
During training and inference the progress bar reports MB/s, TFLOPS, elapsed time, and completion percentage while distributed workers stay synchronized through the join context. FLOP counters and adaptive loss weights update automatically, and the pipeline keeps dataset schemas and scaling statistics in sync with the provided tensors.

The runtime helpers manage distributed checkpoints, mixed precision, exporter requirements, and memory-mapped datasets internally, letting you focus on preparing feature tensors and configuration hyperparameters.

## Debugging runtime tensor issues
- Enable meta/fake tensor diagnostics by setting `STNET_META_HOOK=1` to raise immediately when a module receives a meta/FakeTensor input. Use `STNET_META_HOOK=warn` during inference services to log a warning instead of aborting execution.
- Toggle the oneDNN (MKLDNN) backend with `STNET_DISABLE_MKLDNN=1`. When set, the runtime will call `torch.backends.mkldnn.enabled = False` before model construction so you can confirm whether a backend-specific kernel is responsible for anomalous behavior.

### Sample workbook configuration

When following `notebook.ipynb` to materialize features from `raw_data.xlsx`, the CUDA profile keeps the model depth at 1152 with larger microbatches while the CPU path dials the depth back to 512 and halves the microbatch size to remain memory efficient. Both flows share the same tokenizer geometry so predictions remain shape-compatible across devices.

## Exporting for inference
Exporter helpers automatically check for optional dependencies and raise informative errors if a backend such as ONNX, TensorFlow, Core ML, TensorRT, LiteRT, or ExecuTorch is unavailable. Install the `servable` extra to enable the full conversion toolkit.

## License
**Code** is licensed under **PolyForm Noncommercial 1.0.0**
(SPDX: `PolyForm-Noncommercial-1.0.0`). See `LICENSE`.

**Model weights / datasets** (and other non-code artifacts) are provided
under **CC BY-NC 4.0**. See the "Creative Commons Attribution-NonCommercial 4.0 International" section in `LICENSE`.

Commercial use requires a separate license. Please contact the author.
