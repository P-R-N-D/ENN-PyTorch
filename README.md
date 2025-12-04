# STNet-PyTorch

STNet-PyTorch is a PyTorch implementation of the STNet architecture for joint spatial and temporal modeling. The package ships a high-level backend that takes care of configuration management, training, inference, export utilities, and runtime diagnostics so you can focus on preparing tensors and tuning hyperparameters.

## Features
- **Unified configuration layer**: `stnet.api.config` exposes `ModelConfig`, `PatchConfig`, and `RuntimeConfig` along with helpers such as `build_config`/`coerce_build_config` and modeling aliases that normalize spatial (`ss`), temporal (`tt`), and spatio-temporal (`st`) shorthands.
- **Backend facade**: `stnet.backend` re-exports lifecycle helpers (`new_model`, `save_model`, `load_model`, `learn`/`train`, `infer`/`predict`) and exporter shims (TorchScript, ONNX, TensorRT, Core ML, ExecuTorch, TensorFlow, LiteRT) while using the same configuration dataclasses.
- **Precision-aware modules**: normalization layers and Student’s t components preserve native dtypes for statistics while casting inputs/outputs for safe AMP, BF16, and FP8 execution.
- **Architecture utilities**: `stnet.model` contains the `Root` model, encoder blocks, `CrossAttention`, `PatchAttention`, and shared primitives under `stnet.model.layers`.
- **Data transforms**: reusable preprocessing lives under `stnet.data.transforms` for consistent feature handling.
- **Thread load balancer**: dataloader workers pin to allowed CPUs, request OpenMP `proc_bind(spread)` when available, and tune intra/inter-op threads to avoid oversubscription.

## Requirements and dependencies
- Python 3.10+
- PyTorch built for your hardware (CUDA, ROCm, XPU, or CPU-only) installed prior to the editable install.

Core runtime dependencies:
- `netifaces>=0.11.0`
- `numpy>=2.2.5`
- `psutil>=7.0.0`
- `py-cpuinfo>=9.0.0`
- `scipy>=1.14.1`
- `tensordict>=0.10.0`
- `torch>=2.8.0`
- `torchdata>=0.11.0`
- `torchrl>=0.8.1`
- `tqdm>=4.67.1`
- `triton>=3.2.0`

Optional extras (install with `pip install -e .[extra]`):
- `pandas`: pandas dataframe integration.
- `polars`: Polars dataframe integration.
- `excel`: spreadsheet helpers via `pandas`, `openpyxl`, and `fastexcel`.
- `spark`: Spark pipelines (`pyspark[pandas_on_spark]`).
- `thread`: explicit installation of `psutil` for thread affinity helpers.
- `torchao`: advanced optimization toolchain.
- `nvidia_te_cu12` / `nvidia_te_cu13`: NVIDIA Transformer Engine builds for CUDA 12/13.
- `intel_ai`: Intel Extension for PyTorch.
- `service`: exporter stack (ONNX, ONNX-TF, ONNX2TF, Core ML, TensorRT, ExecuTorch).
- `telemetry`: GPU telemetry (`pynvml`).

## Installation
1. Create and activate a Python 3.10+ environment.
2. Install PyTorch that matches your hardware by following the official [PyTorch instructions](https://pytorch.org/get-started/locally/).
3. Install STNet-PyTorch in editable mode:
   ```bash
   pip install -e .
   ```
4. Add extras as needed, for example the exporter stack:
   ```bash
   pip install -e .[service]
   ```

## Quickstart
```python
import torch
from stnet import (
    PatchConfig,
    build_config,
    infer,
    learn,
    load_model,
    new_model,
    save_model,
)

patch = PatchConfig(is_cube=True, grid_size_3d=(10, 10, 1), patch_size_3d=(1, 1, 1))
config = build_config(
    modeling_type="spatiotemporal",
    depth=64,
    heads=4,
    patch=patch,
    compile_mode="default",
)

model = new_model(in_dim=1024, out_shape=(10,), config=config)
features = torch.randn(32, model.in_dim)
labels = torch.randn(32, *model.out_shape)

train_ds = {"X": features, "Y": labels}
trained = learn(model, train_ds, epochs=1, batch_size=8)

infer_batch = {"X": features, "Y": torch.zeros_like(labels)}
predictions = infer(trained, infer_batch)

save_path = save_model(trained, "checkpoints/stnet.pt")
restored = load_model(save_path)

restored.eval()
with torch.inference_mode():
    scripted_output, _ = restored(features)
```

During training the progress bar reports MB/s, TFLOPS, elapsed time, and completion percentage while distributed workers stay synchronized. FLOP counters and adaptive loss weights update automatically, and dataset schemas remain aligned with provided tensors.

## Configuration and compilation
`ModelConfig.compile_mode` accepts the same modes as `torch.compile` (e.g., `"default"`, `"reduce-overhead"`, `"max-autotune"`). The helper in `stnet.functional.fx.compile` trims whitespace, normalizes disabled options (`"disabled"`, `"none"`, empty string), and skips compilation when unsupported.

## Diagnostics and troubleshooting
- Set `STNET_META_HOOK=1` to raise immediately when a module receives a meta/FakeTensor. Use `STNET_META_HOOK=warn` during inference services to log without aborting.
- Set `STNET_DISABLE_MKLDNN=1` to disable the oneDNN (MKLDNN) backend before model construction.

## Exporting for inference
Exporter helpers automatically check for optional dependencies and raise informative errors if ONNX, TensorFlow, Core ML, TensorRT, LiteRT, or ExecuTorch backends are missing. Install the `service` extra to enable the full conversion toolkit.

## License
**Code** is licensed under **PolyForm Noncommercial 1.0.0** (SPDX: `PolyForm-Noncommercial-1.0.0`). See `LICENSE`.

**Model weights / datasets** (and other non-code artifacts) are provided under **CC BY-NC 4.0**. See the "Creative Commons Attribution-NonCommercial 4.0 International" section in `LICENSE`.

Commercial use requires a separate license. Please contact the author.
