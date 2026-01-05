# STNet-PyTorch

Spatio‑temporal modular neural network (MNN) model for PyTorch with pragmatic utilities for data pipelines, training, export, and system tuning. The package is structured to keep the *model code* small while providing a clean façade over configuration, I/O, distributed runtime, and export helpers.

This repository also includes a worked example notebook (`notebook.ipynb`) and a sample data (`raw_data.xlsx`) demonstrating a tabular‑to‑grid regression workflow.

## Requirements

### Mandatory
- **Python**: >= 3.10 (uses `match` / `case`)
- **PyTorch**: >= 2.8.0
- **Triton**: >= 3.4.0 (core JIT backend; typically installed alongside PyTorch)
- **torchao**: >= 0.14.0
- **torchdata**: >= 0.11.0 (`torchdata.nodes`-based pipeline)
- **tensordict**: >= 0.10.0
- **numpy**: >= 2.2.5
- **psutil**: >= 7.0.0
- **py-cpuinfo**: >= 9.0.0
- **tqdm**: >= 4.67.1
- **h5py**: >= 3.11.0 (required for persisted prediction outputs)

### Recommended
- **Python**: >= 3.13t (free-threading / no-GIL build)
  - The data pipeline uses thread-parallel execution via `torchdata.nodes` and is designed to reduce GIL contention on standard CPython.
  - Free-threaded Python can further improve throughput by removing the GIL, but it is not necessary.
- **PyTorch**: >= 2.9.1
- DataFrame integrations (pandas, pandas-on-Spark, polars) are optional. install the corresponding extra (e.g., `pip install -e .[pandas]`) when needed.

## Features
- **APIs** (`stnet.api`): build/load models, elastic train/predict entrypoints (uses `torch.distributed.elastic`), and checkpoint/export helpers.
- **Templated configurations** (`stnet.config`): dataclass configs with coercion/validation and string canonicalizers for modeling type, normalization, and compile options.
- **Neural network stacks** (`stnet.nn`): spatio-temporal TokenFuser/TokenCollector blocks, attention variants, scaler + recorder modules, AMP negotiation guard band (`ModelConfig.safety_margin_pow2`).
- **Data pipeline** (`stnet.data`): `torchdata.nodes`-driven memmap pipeline with TensorDict support, prefetch/pin/pool options, and scale-aware dataset metadata.
- **Runnable tasks** (`stnet.runtime`): thread/NUMA tuning, free-threaded/no-GIL optimizations, mixed-precision helpers, history recorder, and OOM recovery hooks. ONNX/ORT/TorchScript out of the box; optional platform-dependent backends (TensorRT/CoreML/ExecuTorch/onnx-tf) via extras. elastic launch wiring and group setup for multi-process CPU/GPU runs.
- **Losses/optimizers/profiling** (`stnet.core`): Student’s t losses, SWA helpers, FLOP/IO timing.

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
# Minimal ONNX / ORT export helpers
pip install -e .[deployment]

# Broader export stack (platform/python constraints; some wheels may be unavailable on py3.13/py3.14t)
# Includes onnx-tf/onnx2tf (py<3.13), CoreML (macOS), TensorRT (Linux/CUDA), ExecuTorch (Linux, py<3.12)
pip install -e .[deployment_full]

# Dataframe integrations
pip install -e .[pandas]      # or: .[polars]

# Pandas on Spark (with pandas-on-Spark)
pip install -e .[pandas_on_spark]

# NVIDIA TE / Intel IPEX (hardware-specific)
pip install -e .[nvidia_te_cu12]   # or .[nvidia_te_cu13]
pip install -e .[intel_ai]

# Telemetry (NVIDIA GPU info)
pip install -e .[telemetry]

# NUFFT losses (CUDA, optional)
pip install -e .[nufft]

# Dev tooling (lint/test)
pip install -e .[dev]
```

> **Note**: Triton >= 3.4.0 is required, but a matching build is normally installed alongside PyTorch—avoid overriding it with a mismatched wheel.

> **Platform note**: many `deployment_full` backends are OS/driver/python dependent (e.g., CoreML on macOS, TensorRT on Linux/CUDA, ExecuTorch often lacks wheels for Python ≥3.12). If you only need ONNX export, installing `onnx` (and optionally `onnxruntime`) is usually sufficient.

## Quickstart

Minimal forward/backward loop:

```python
import torch

import stnet

from stnet.config import ModelConfig
from stnet.runtime.losses import StudentsTLoss
from stnet.core.system import optimize_threads

# 1) Build a config and model
cfg = ModelConfig(
    d_model=128,
    heads=4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    # AMP dtype negotiation guard band: safety_margin = 2**n (default n=3 -> margin=8)
    safety_margin_pow2=3,
)
model = stnet.new_model(in_dim=16, out_shape=(1,), config=cfg)

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
from stnet import load_model, save_model

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
import stnet

# In-memory (default): returns a CPU TensorDict with keys {"X", "Y"}.
# NOTE: output='eager' is accepted as an alias of output='memory'.
td = stnet.predict(model, my_data, output="memory", path=None)
print(td[0]["X"], td[0]["Y"])

# Persistent: writes an HDF5 file and returns a PersistentTensorDict.
# NOTE: output='lazy' is accepted as an alias of output='file'.
td_persistent = stnet.predict(
    model,
    my_data,
    output="file",
    path="predictions.h5",
    overwrite="replace",
)
print(td_persistent[0]["X"], td_persistent[0]["Y"])

# IMPORTANT: PersistentTensorDict does not auto-close. Close it when you're done.
td_persistent.close()
```

Notebook demo:
- Open `notebook.ipynb` for an end-to-end workflow using `raw_data.xlsx`.

> API names above reflect the current package layout. If you have local changes, adjust imports accordingly.

## Project layout

```
stnet/
  __init__.py
  api.py
  config.py
  core/
    __init__.py
    casting.py            # dtype helpers, env parsing, safe tensor coercions
    compat.py             # accelerator/memory helpers, meta/fake tensor guards
    distributed.py        # elastic launch + process group utilities
    graph.py              # torch.compile helpers, graph break utilities
    precision.py          # AMP negotiation/autocast policies, dtype guards
    profiler.py           # lightweight FLOP/IO timers
    staging.py            # pinned-memory staging pool
    system.py             # thread/NUMA tuning, device detection, temp dirs
  data/
    __init__.py
    nodes.py              # torchdata nodes, Sampler/Loader, memmap writer/reader
    pipeline.py           # dataset fetch, collate, session orchestration
    schemas.py            # key resolution, JSON helpers, underflow handling
  runtime/
    __init__.py
    io.py                 # exporters (ONNX/ORT/TorchScript/etc.), checkpoint save/load
    main.py               # training loop, predict path, elastic worker entrypoint
    losses.py             # Student’s t, regression losses, mask utils
    optimizers.py         # SGD/AdamW wrappers, SWA helper
  nn/
    __init__.py
    activations.py
    architecture.py
    blocks.py
    kernels.py
    layers.py
```

## Configuration notes

### ModelConfig string options

`stnet.config.coerce_model_config()` normalizes common separator variants in a few string fields
to reduce "almost-right" config bugs:

- `modeling_type`: canonical values `{ss, tt, st}`.
  - Examples accepted: `spatial`, `temporal`, `spatiotemporal`, `spatio-temporal`, `spatio_temporal`, `temporal-spatial`, …
- `normalization_method`: canonical values `layernorm`, `batchnorm`, `rmsnorm`.
  - Examples accepted: `ln`, `layer_norm`, `layer-norm`, `bn`, `batch_norm`, `rms_norm`, …
- `compile_mode`: canonical values `disabled` (default), `default`, `reduce-overhead`, `max-autotune`,
  `max-autotune-no-cudagraphs`, `aot-eager`.
  - Examples accepted: `max_autotune`, `max-autotune`, `reduce_overhead`, `aot_eager`, …

### RuntimeConfig string options (training)

- `loss_mask_mode`: must be one of `none`, `finite`, or `neq`.
  - Examples accepted: `isfinite`, `not_equal`, `!=`, …


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

- **Prediction assembly (predict)**
  - `STNET_PRED_ASSEMBLE_MAX_SEGMENTS`: threshold for the "segment slice-copy" fast path when assembling chunked predictions.
    - Default: `64`.
    - If `<= 0`, the segment fast path is disabled (always uses index-based writes).
  - `STNET_PRED_ASSEMBLE_TUNE`: enable best-effort tuning logs for prediction assembly.
    - Default: `1`.
    - When enabled, assembly reports how often the segment-copy path was applicable and may suggest a better `STNET_PRED_ASSEMBLE_MAX_SEGMENTS`.
  - `STNET_PRED_ASSEMBLE_TUNE_LOG_ONCE`: log tuning guidance at most once per process.
    - Default: `1`.
  - `STNET_PRED_ASSEMBLE_TUNE_MIN_PARTS`: minimum number of manifest parts required before emitting tuning logs.
    - Default: `4`.

- **Prefetch (CUDA event polling)**
  - `STNET_CUDA_EVENT_POLL_START_US`: initial sleep (microseconds) for CUDA event completion polling.
    - Default: `500` (0.5 ms).
  - `STNET_CUDA_EVENT_POLL_MAX_MS`: max sleep (milliseconds) for CUDA event polling backoff.
    - Default: `50`.
  - `STNET_CUDA_EVENT_POLL_STOP_MIN_MS`: minimum sleep (milliseconds) once shutdown/stop is requested.
    - Default: `5`.

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
