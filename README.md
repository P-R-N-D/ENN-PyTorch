# STNet-PyTorch

Spatio‑temporal neural network building blocks for PyTorch with pragmatic utilities for data pipelines, training, export, and system tuning. The package is structured to keep the *model code* small while providing a clean façade over configuration, I/O, distributed runtime, and export helpers.

> **Python**: 3.10+ · **License**: PolyForm Noncommercial 1.0.0

## Features
- **Typed configuration** (`stnet.api.config`): dataclass-based configs with sensible defaults and validation.
- **I/O helpers** (`stnet.api.io`): create models from config and save/load checkpoints with device‑safe tensor handling.
- **Runtime utilities** (`stnet.backend.runtime`, `stnet.backend.system`): thread/NUMA tuning, mixed-precision friendly components, and training-time helpers.
- **Distributed** (`stnet.backend.distributed`): utilities to bootstrap and coordinate multi‑process training.
- **Export** (`stnet.backend.export`): ONNX and serving-oriented conversion helpers (optional `service` extra).
- **Data pipeline** (`stnet.data`): transforms, simple stats, and `torchdata`-driven nodes for scalable input pipelines.
- **Functional blocks** (`stnet.functional`): robust losses (e.g., Student’s t), FX utils, and optimizers/SWA.
- **Model library** (`stnet.model`): attention variants and spatio‑temporal layers (e.g., `History`, `Instance`).

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
# ONNX/CoreML/TensorRT/ExecuTorch export helpers
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
```

> **Note**: Do **not** install `triton` manually; the correct Triton build is pulled automatically by PyTorch.

## Quickstart

Minimal forward/backward loop:

```python
import torch
from stnet.api.config import ModelConfig
from stnet.model.layers import Instance
from stnet.functional.losses import StudentsTLoss
from stnet.backend.system import optimize_threads

# 1) Build a config and model
cfg = ModelConfig(
    in_dim=16,
    out_shape=(1,),
    depth=4,
    heads=4,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
model = Instance(cfg.in_dim, cfg.out_shape, cfg)

# 2) Synthetic batch (B x T x C_in) -> (B x T x *out_shape)
x = torch.randn(32, 12, cfg.in_dim, device=next(model.parameters()).device)
y = torch.randn(32, 12, *cfg.out_shape, device=x.device)

# 3) Loss & optimizer
loss_fn = StudentsTLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 4) Optional: autotune thread settings for this machine
optimize_threads()

model.train()
for step in range(200):
    pred = model(x)                 # forward
    loss = loss_fn(pred, y)         # compute loss
    loss.backward()                 # backward
    opt.step(); opt.zero_grad()     # update
```

Checkpointing:
```python
from stnet.api.io import save_model, load_model

save_model(model, "ckpt.pth")
model2 = load_model("ckpt.pth", device="cuda")
```

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
    profiler.py
    runtime.py
    system.py
  data/
    __init__.py
    datatype.py
    nodes.py
    pipeline.py
    stats.py
    transforms.py
  functional/
    __init__.py
    fx.py
    losses.py
    optimizers.py
  model/
    __init__.py
    activations.py
    kernels.py
    layers.py

```

## Environment variables

- `STNET_META_HOOK`: set to `1` to fail fast on metadata shape/type mismatches during module wiring; set to `warn` to log only.
- `STNET_DISABLE_MKLDNN`: set to `1` to disable oneDNN (MKLDNN) before model construction if it causes issues for your CPU build.

## Version & compatibility notes

- Python ≥ 3.10 is required.
- PyTorch is expected to be **≥ 2.8.0**. If you rely on features that landed later (e.g., advanced Inductor options), bump the constraint accordingly.
- `torchdata` is used by the data pipeline nodes; it is listed as a dependency to avoid runtime import errors.
- `triton` is intentionally **not** pinned here; the correct binary is installed as a transitive dependency of PyTorch.

## License

**Code** is licensed under **PolyForm Noncommercial 1.0.0**.  
**Weights/datasets/other artifacts** remain under the terms described in your `LICENSE` file. For commercial use, obtain a separate license.
