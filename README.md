# STNet-PyTorch

Spatio‑temporal modular neural network (MNN) model for PyTorch with pragmatic utilities for data pipelines, training, export, and system tuning. The package is structured to keep the *model code* small while providing a clean façade over configuration, I/O, distributed runtime, and export helpers.

This repository also includes a worked example notebook (`notebook.ipynb`, Korean) and a small sample dataset (`raw_data.xlsx`) demonstrating a tabular‑to‑grid regression workflow.

> **License**: PolyForm Noncommercial 1.0.0

## Requirements

### Mandatory
- **Python**: >= 3.10 (uses `match` / `case`)
- **PyTorch**: >= 2.8.0
- **torchdata**: >= 0.11.0 (`torchdata.nodes`-based pipeline)
- **tensordict**: >= 0.10.0

### Recommended (throughput)
- **Python 3.13t** (free-threading / no-GIL build)
  - The data pipeline uses thread-parallel execution via `torchdata.nodes` and is designed to reduce GIL contention on standard CPython.
  - 3.14t can further improve throughput by removing the GIL, but it is not required.

## Features
- **Typed configuration** (`stnet.api.config`): dataclass-based configs with sensible defaults and validation/coercion.
- **I/O helpers** (`stnet.api.io`): create models from config and save/load checkpoints with device‑safe tensor handling.
- **Runtime utilities** (`stnet.backend.runtime`, `stnet.backend.system`): thread/NUMA tuning, mixed-precision friendly components, and training-time helpers.
- **Distributed** (`stnet.backend.distributed`): utilities to bootstrap and coordinate multi‑process training.
- **Export** (`stnet.backend.export`): ONNX / ONNX Runtime (ORT) / TensorRT / CoreML / ExecuTorch conversion helpers (optional `deployment` extra; some backends are platform-specific).
- **Data pipeline** (`stnet.data`): `torchdata`-driven nodes with memmap-friendly flows.
- **Functional blocks** (`stnet.functional`): robust losses (e.g., Student’s t), optimizer/SWA helpers, and lightweight profiling utilities.
- **Model library** (`stnet.model`): attention variants and spatio‑temporal layers (e.g., `Root`, `History`).
- **AMP negotiation margin** (`ModelConfig.safety_margin_pow2`): sets the conservative overflow guard band used when selecting mixed-precision dtypes (margin = 2**n).

## Installation

1. Install the appropriate **PyTorch** build for your accelerator first (CUDA/ROCm/XPU/CPU).
2. Install STNet-PyTorch (editable for development is recommended):
   ```bash
   pip install --upgrade pip
   # Example (CUDA users should pick the right index-url/wheel for their CUDA version)
   # pip install --index-url https://download.pytorch.org/whl/cu121 'torch>=2.8.0'
   pip install -e .
   ```

Optional extras:
```bash
# ONNX / ORT / CoreML / TensorRT / ExecuTorch export helpers
pip install -e .[deployment]

# Dataframe integrations
pip install -e .[pandas]      # or: .[polars]

# Pandas on Spark (with pandas-on-Spark)
pip install -e .[pandas_on_spark]

# NVIDIA TE / Intel IPEX / TorchAO (hardware-specific)
pip install -e .[nvidia_te_cu12]   # or .[nvidia_te_cu13]
pip install -e .[intel_ai]
pip install -e .[torchao]

# Telemetry (NVIDIA GPU info)
pip install -e .[telemetry]

# NUFFT losses (CUDA, optional)
pip install -e .[nufft]

# Dev tooling (lint/test)
pip install -e .[dev]
```

> **Note**: Do **not** install `triton` manually; the correct Triton build is pulled automatically by PyTorch.

> **Platform note**: some `deployment` backends are OS/driver dependent (e.g., CoreML on macOS, TensorRT on Linux with NVIDIA CUDA). If you only need ONNX export, installing `onnx` (and optionally `onnxruntime`) is usually sufficient.

## Quickstart

Minimal forward/backward loop:

```python
import torch

from stnet.api.config import ModelConfig
from stnet.api.io import new_model
from stnet.functional.losses import StudentsTLoss
from stnet.backend.system import optimize_threads

# 1) Build a config and model
cfg = ModelConfig(
    depth=4,
    heads=4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    # AMP dtype negotiation guard band: safety_margin = 2**n (default n=3 -> margin=8)
    safety_margin_pow2=3,
)
model = new_model(in_dim=16, out_shape=(1,), config=cfg)

# 2) Synthetic batch (B x C_in) -> (B x *out_shape)
device = next(model.parameters()).device
x = torch.randn(32, 16, device=device)
y = torch.randn(32, 1, device=device)
labels_flat = y.reshape(y.shape[0], -1)

# 3) Loss & optimizer
net_loss = StudentsTLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 4) Optional: autotune thread settings for this machine
optimize_threads()

model.train()
for step in range(200):
    pred, loss = model(x, labels_flat=labels_flat, net_loss=net_loss)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

# Inference
model.eval()
with torch.no_grad():
    pred = model(x, return_loss=False)
    # or: pred = model.predict(x)
```

Checkpointing:
```python
from stnet.api.io import save_model, load_model

save_model(model, "ckpt.pth")
model2 = load_model("ckpt.pth", map_location="cuda")

# Directory checkpoints (torch.distributed.checkpoint + meta.json)
from pathlib import Path
Path("ckpt_dir").mkdir(exist_ok=True)
save_model(model, "ckpt_dir")
model3 = load_model("ckpt_dir", map_location="cpu")
```

Prediction outputs:
```python
from stnet.api.run import predict, get_prediction

# Write streaming prediction chunks to disk without holding tensors in memory
file_info = predict(model, data=my_data, output="file")

# Later (or on another machine), reopen chunks as a mapping
lazy_preds = get_prediction(file_info, output="tensor", lazy=True)
for key, tensor in lazy_preds.items():
    ...
```
`output="file"` streams per-rank parts to `chunks_dir` with stable key metadata; use `get_prediction` to reload eagerly or lazily without rematerializing keys.

Notebook demo:
- Open `notebook.ipynb` for an end-to-end workflow using `raw_data.xlsx`.

> API names above reflect the current package layout. If you have local changes, adjust imports accordingly.

## Project layout

```
stnet/
  __init__.py
  api/
    __init__.py
    config.py
    io.py
    run.py
  backend/
    __init__.py
    compat.py
    distributed.py
    export.py
    runtime.py
    system.py
  data/
    __init__.py
    collections.py
    datatype.py
    nodes.py
    pipeline.py
  functional/
    __init__.py
    losses.py
    optimizers.py
    profiler.py
  model/
    __init__.py
    activations.py
    fused.py
    kernels.py
    layers.py
    nn.py

```

## Environment variables

- `STNET_META_MONITOR` (alias: `STNET_META_HOOK`): set to `1` to fail fast on meta-tensor inputs during model wiring/forward; set to `warn` to log only.
- `STNET_DISABLE_MKLDNN`: set to `1` to disable oneDNN (MKLDNN) before model construction if it causes issues for your CPU build.

- **Free-threading / no-GIL**
  - `STNET_NOGIL_OPT` (aliases: `STNET_NO_GIL_OPT`, `STNET_FREE_THREADING_OPT`): enable/disable no-GIL-specific tuning. Default: auto-detect (enabled only when running on a free-threaded build *and* the GIL is disabled).
  - `STNET_MEMMAP_THREAD_LOCAL`: use per-thread `MemoryMappedTensor` handles (`1`/`0`). Default: auto-enabled when no-GIL optimizations are enabled.
  - `STNET_TLB_FLUSH_EVERY`, `STNET_TLB_SAMPLE_EVERY`: reduce thread-local telemetry overhead in high-throughput thread pipelines (defaults are higher when no-GIL optimizations are enabled).

- **NVML telemetry** (optional extra: `.[telemetry]`)
  - `STNET_NVML_DISABLE=1`: disable NVML telemetry unconditionally.
  - `STNET_NVML_MIN_INTERVAL_S` (alias: `STNET_NVML_MIN_INTERVAL`): throttle NVML utilization queries (seconds).

- **TorchInductor compilation**
  - `STNET_INDUCTOR_COMPILE_THREADS` (alias: `STNET_COMPILE_THREADS`): cap Inductor compile threads per process to avoid oversubscription on multi-rank nodes.

## Version & compatibility notes

- Python ≥ 3.10 is required.
- PyTorch is expected to be **≥ 2.8.0**. If you rely on features that landed later (e.g., newer Inductor options), bump the constraint accordingly.
- `torchdata` (`torchdata.nodes`) powers the data pipeline.
- `tensordict` provides TensorDict integration and memory-mapped tensor utilities.
- `triton` is intentionally **not** pinned here; the correct binary is installed as a transitive dependency of PyTorch.
- Export backends (TensorRT/CoreML/ExecuTorch/ORT) have additional system requirements; install only what you need.

## License

**Code** is licensed under **PolyForm Noncommercial 1.0.0**.  
**Weights/datasets/other artifacts** remain under the terms described in your `LICENSE` file. For commercial use, obtain a separate license
