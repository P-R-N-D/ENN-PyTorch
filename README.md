# ENN-PyTorch

Elastic Neural Networks (ENN), modular neural networks for high-performance computing (HPC) — flexible and efficient by design, adaptive and stable at runtime, implemented in PyTorch. This features pragmatic utilities for data pipelines, training, export, and system tuning. The package is structured to keep the *model code* small while providing a clean façade over configuration, I/O, distributed runtime, and export helpers.

This repository also includes a worked example notebook (`notebook.ipynb`) and a sample data (`raw_data.xlsx`) demonstrating a tabular‑to‑grid regression workflow.

## Requirements

### Mandatory
- **python**: >= 3.12
- **torch**: >= 2.9.1
- **torchvision**: >= 0.24.1
- **torchao**: >= 0.15.0
- **torchdata**: >= 0.11.0 (`torchdata.nodes`-based pipeline)
- **tensordict**: >= 0.10.0
- **triton**: >= 3.5.1 (core JIT backend; typically installed alongside PyTorch)
- **numpy**: >= 2.4.1
- **onnx**: >= 1.20.1
- **onnxruntime**: >= 1.23.2
- **onnxscript**: >= 0.5.7
- **onnx_ir**: >= 0.1.14
- **openzl**: >= 0.1.0 (If error happens when building wheel, install this using the official GitHub repository by Meta Platforms, Inc. (e.g., `pip install "openzl @ git+https://github.com/facebook/openzl.git#subdirectory=py"`)
- **ml_dtypes**: >= 0.5.4
- **psutil**: >= 7.2.1
- **py-cpuinfo**: >= 9.0.0
- **tqdm**: >= 4.67.1
- **h5py**: >= 3.15.1 (required for persisted prediction outputs)

### Recommended
- **python**: >= 3.13t (free-threading / no-GIL build)
  - The data pipeline uses thread-parallel execution via `torchdata.nodes` and is designed to reduce GIL contention on standard CPython.
  - Free-threaded Python can further improve throughput by removing the GIL, but it is not necessary.
- DataFrame integrations (pandas, pandas-on-Spark, polars) are optional. install the corresponding extra (e.g., `pip install -e .[pandas]`) when needed.

## Features
- **APIs** (`enn_torch.runtime.workflow`): build/load models, elastic train/predict entrypoints (uses `torch.distributed.elastic`), and checkpoint/export helpers.
- **Templated configurations** (`enn_torch.config`): dataclass configs with coercion/validation and string canonicalizers for modeling type, normalization, and compile options.
- **Neural network stacks** (`enn_torch.nn`): spatio-temporal TokenFuser/TokenCollector blocks, attention variants, scaler + recorder modules, AMP negotiation guard band (`ModelConfig.safety_margin_pow2`).
- **Data pipeline** (`enn_torch.data`): `torchdata.nodes`-driven memmap pipeline with TensorDict support, prefetch/pin/pool options, and scale-aware dataset metadata.
- **Runnable tasks** (`enn_torch.runtime`): thread/NUMA tuning, free-threaded/no-GIL optimizations, mixed-precision helpers, history recorder, and OOM recovery hooks. ONNX/ORT/onnxscript/onnx_ir/torch.export (PT2) out of the box; optional platform-dependent backends (TensorRT/CoreML/ExecuTorch/onnx-tf) via extras. elastic launch wiring and group setup for multi-process CPU/GPU runs.
- **Losses/optimizers/profiling** (`enn_torch.core`): Student’s t losses, SWA helpers, FLOP/IO timing.

## Model layout (current)

High-level flow used by `enn_torch.nn.architecture.Model`:

```
Input features (B x C_in)
  → Scaler (feature normalization)
  → TokenFuser
      → TokenizedView (spatial extractor)
      → TokenizedView (temporal extractor)
      → CrossTransformer fusion (spatial|temporal)
      → Aggregation + MLP head → output vector
  → TokenCollector (temporal controller head)
  → Optional linear branch (configurable)
```

Key building blocks:
- **Scaler** for input feature normalization, with an optional linear branch when `use_linear_branch` is enabled in the config.【F:enn_torch/nn/architecture.py†L1032-L1049】
- **TokenFuser** builds spatial/temporal TokenizedViews and fuses them via a CrossTransformer before projecting through the head MLP to the output dimension.【F:enn_torch/nn/architecture.py†L659-L836】
- **TokenCollector** acts as the temporal controller head for the model instance.【F:enn_torch/nn/architecture.py†L1051-L1061】

## Installation
1. Install the appropriate **PyTorch** build for your accelerator first (CUDA/ROCm/XPU/CPU).
2. Install ENN-PyTorch (editable for development is recommended):
   ```bash
   pip install --upgrade pip
   # Example (CUDA users should pick the right index-url/wheel for their CUDA version)
   # pip install --index-url https://download.pytorch.org/whl/cu121 'torch>=2.9.1'
   pip install -e .
   ```

Optional extras:
```bash
# Broader export stack (platform/python constraints; some wheels may be unavailable on py3.13/py3.14t)
# Includes onnx-tf/onnx2tf (py<3.13; requires tf-keras), CoreML (macOS), TensorRT (Linux/CUDA), ExecuTorch (Linux, py<3.12)
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

> **Note**: Triton >= 3.5.1 is required, but a matching build is normally installed alongside PyTorch—avoid overriding it with a mismatched wheel.

> **Platform note**: many `deployment_full` backends are OS/driver/python dependent (e.g., CoreML on macOS, TensorRT on Linux/CUDA, ExecuTorch often lacks wheels for Python ≥3.12). If you only need ONNX export, the mandatory `onnx`/`onnxruntime`/`onnxscript`/`onnx_ir` set is sufficient.

## Distributed training & inference (HSDP)

ENN uses **FSDP2 (composable `fully_shard`)** for both training and inference on GPU/XPU.
When running multi-node, ENN will prefer an **HSDP-style 2D DeviceMesh**:

- **Shard** within a node (across local GPUs)
- **Replicate** across nodes (no cross-node parameter sharding)

### Asymmetric nodes (different GPU counts)

HSDP requires a rectangular 2D mesh (i.e., the same number of local ranks/GPUs per node). When nodes have different GPU counts, ENN will **automatically fall back to a 1D FSDP2 mesh** across all ranks. This keeps the run working (no hang/crash), but it does mean **parameters can be sharded across nodes** in the fallback mode.

### Mesh selection

- Default: **implicit mesh** via `torch.distributed.device_mesh.init_device_mesh`.
- If your rank ordering is not node-contiguous, you can force explicit mesh construction:
  - `ENN_HSDP_EXPLICIT_MESH=1`
- If your launcher does not set `LOCAL_WORLD_SIZE`, you can override the assumed per-node size:
  - `ENN_HSDP_SHARD_SIZE=<int>`

## Export / serving notes

- ONNX export defaults to **opset 18**.
- Some graphs contain batch-dependent control flow; the export utilities avoid common `batch==1` specialization pitfalls by using safe defaults (and disabling dynamic batching when the current `torch.export.Dim` implementation cannot express constraints).

## Quickstart

Minimal forward/backward loop:

```python
import torch

import enn_torch

from enn_torch.config import ModelConfig
from enn_torch.runtime.losses import StudentsTLoss
from enn_torch.core.policies import optimize_threads

# 1) Build a config and model
cfg = ModelConfig(
    d_model=128,
    heads=4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    # AMP dtype negotiation guard band: safety_margin = 2**n (default n=3 -> margin=8)
    safety_margin_pow2=3,
)
model = enn_torch.new_model(in_dim=16, out_shape=(1,), config=cfg)

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
from enn_torch import load_model, save_model

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
import enn_torch

# In-memory (default): returns a CPU TensorDict with keys {"X", "Y"}.
# NOTE: output='eager' is accepted as an alias of output='memory'.
td = enn_torch.predict(model, my_data, output="memory", path=None)
print(td[0]["X"], td[0]["Y"])

# Persistent: writes an HDF5 file and returns a PersistentTensorDict.
# NOTE: output='lazy' is accepted as an alias of output='file'.
td_persistent = enn_torch.predict(
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
enn_torch/
  __init__.py
  config.py
  core/
    __init__.py
    compat.py             # accelerator/memory helpers, meta/fake tensor guards
    concurrency.py        # threading/affinity helpers
    datatypes.py          # env parsing + small type utilities
    distributed.py        # elastic launch + process group utilities
    graph.py              # torch.compile helpers, graph break utilities
  runtime/
    workflow.py           # build/load models, elastic train/predict entrypoints
    policies.py           # thread/data policy heuristics
    precision.py          # dtype/precision helpers
    profiler.py           # lightweight FLOP/IO timers
    system.py             # thread/NUMA tuning, device detection, temp dirs
    tensor.py             # tensor helpers + from_buffer context manager
  data/
    __init__.py
    nodes.py              # torchdata nodes, Sampler/Loader, memmap writer/reader
    pipeline.py           # dataset fetch, collate, session orchestration
    collate.py            # dataset storage helpers
  runtime/
    __init__.py
    io.py                 # exporters (ONNX/ORT/torch.export (PT2)/etc.), checkpoint save/load
    main.py               # training loop, predict path, elastic worker entrypoint
    losses.py             # Student’s t, regression losses, mask utils
    optimizers.py         # SGD/AdamW wrappers, SWA helper
    wrappers.py           # export/runtime wrappers
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

`enn_torch.config.coerce_model_config()` normalizes common separator variants in a few string fields
to reduce "almost-right" config bugs:

- `modeling_type`: canonical values `{ss, tt, st}`.
  - Examples accepted: `spatial`, `temporal`, `spatiotemporal`, `spatio-temporal`, `spatio_temporal`, `temporal-spatial`, …
- `normalization_method`: canonical values `layernorm`, `batchnorm`, `rmsnorm`.
  - Examples accepted: `ln`, `layer_norm`, `layer-norm`, `bn`, `batch_norm`, `rms_norm`, …
- `compile_mode`: canonical values `disabled` (default), `reduce-overhead` (alias: `stable`),
  `max-autotune`, `max-autotune-no-cudagraphs`, `aot-eager` (alias: `debug`).
  - Examples accepted (case-insensitive; `-`/`_` treated the same): `max_autotune`, `max-autotune`,
    `stable`, `reduce_overhead`, `debug`, `aot_eager`, …
  - Any other value is treated as `disabled`.

### RuntimeConfig string options (training)

- `loss_mask_mode`: must be one of `none`, `finite`, or `neq`.
  - Examples accepted: `isfinite`, `not_equal`, `!=`, …


## Environment variables

- `ENN_META_MONITOR` (alias: `ENN_META_HOOK`): set to `1` to fail fast on meta-tensor inputs during model wiring/forward; set to `warn` to log only.
- `ENN_DISABLE_MKLDNN`: set to `1` to disable oneDNN (MKLDNN) before model construction if it causes issues for your CPU build.

- **Free-threading / no-GIL**
  - `ENN_NOGIL_OPT` (aliases: `ENN_NO_GIL_OPT`, `ENN_FREE_THREADING_OPT`): enable/disable no-GIL-specific tuning. Default: auto-detect (enabled only when running on a free-threaded build *and* the GIL is disabled).
  - `ENN_MEMMAP_THREAD_LOCAL`: use per-thread `MemoryMappedTensor` handles (`1`/`0`). Default: auto-enabled when no-GIL optimizations are enabled.
  - `ENN_TLB_FLUSH_EVERY`, `ENN_TLB_SAMPLE_EVERY`: reduce thread-local telemetry overhead in high-throughput thread pipelines (defaults are higher when no-GIL optimizations are enabled).

- **NVML telemetry** (optional extra: `.[telemetry]`)
  - `ENN_NVML_DISABLE=1`: disable NVML telemetry unconditionally.
  - `ENN_NVML_MIN_INTERVAL_S` (alias: `ENN_NVML_MIN_INTERVAL`): throttle NVML utilization queries (seconds).

- **TorchInductor compilation**
  - `ENN_INDUCTOR_COMPILE_THREADS` (alias: `ENN_COMPILE_THREADS`): cap Inductor compile threads per process to avoid oversubscription on multi-rank nodes.

- **Prediction assembly (predict)**
  - `ENN_PRED_ASSEMBLE_MAX_SEGMENTS`: threshold for the "segment slice-copy" fast path when assembling chunked predictions.
    - Default: `64`.
    - If `<= 0`, the segment fast path is disabled (always uses index-based writes).
  - `ENN_PRED_ASSEMBLE_TUNE`: enable best-effort tuning logs for prediction assembly.
    - Default: `1`.
    - When enabled, assembly reports how often the segment-copy path was applicable and may suggest a better `ENN_PRED_ASSEMBLE_MAX_SEGMENTS`.
  - `ENN_PRED_ASSEMBLE_TUNE_LOG_ONCE`: log tuning guidance at most once per process.
    - Default: `1`.
  - `ENN_PRED_ASSEMBLE_TUNE_MIN_PARTS`: minimum number of manifest parts required before emitting tuning logs.
    - Default: `4`.

- **Prefetch (CUDA event polling)**
  - `ENN_CUDA_EVENT_POLL_START_US`: initial sleep (microseconds) for CUDA event completion polling.
    - Default: `500` (0.5 ms).
  - `ENN_CUDA_EVENT_POLL_MAX_MS`: max sleep (milliseconds) for CUDA event polling backoff.
    - Default: `50`.
  - `ENN_CUDA_EVENT_POLL_STOP_MIN_MS`: minimum sleep (milliseconds) once shutdown/stop is requested.
    - Default: `5`.

## Version & compatibility notes

- `python` ≥ 3.12 is mandatory among GIL-enabled builds (or `abi3`-complicants), while `python` ≥ 3.13t is reqiured among free-threading builds (or `abi3t`-complicants).
- `torch` ≥ 2.9.1 is mandatory.
- `torchdata` ≥ 0.11.0 is mandatory bacause `torchdata.nodes` is required.
- `tensordict` provides TensorDict integration and memory-mapped tensor utilities.
- `triton` ≥ 3.5.1 is necessary.
- Export backends (TensorRT/CoreML/ExecuTorch/ORT) have additional system requirements; install only what you need.

## License

**Code** is licensed under **PolyForm Noncommercial 1.0.0**.  
**Weights/datasets/other artifacts** remain under the terms described in your `LICENSE` file. For commercial use, obtain a separate license
