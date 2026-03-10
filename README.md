# ENN-PyTorch

> A PyTorch-based Elastic Neural Networks framework for training, inference, export, and runtime-aware AI deployment workflows.

## Overview

ENN-PyTorch is a research-to-deployment deep learning framework designed to keep model development, execution control, and artifact export inside one consistent workflow.

The project focuses on a practical engineering problem: a model is only useful when it can be trained, evaluated, executed repeatedly, and exported in a way that remains stable under real runtime conditions. ENN-PyTorch addresses that problem by combining configurable model composition, data pipelines, distributed execution, runtime safeguards, and deployment-oriented export support.

## Core Goals

- unify training, inference, checkpointing, and export in one workflow
- support repeatable execution under changing runtime conditions
- reduce friction between model iteration and deployment preparation
- improve resilience under memory pressure and unstable kernel behavior
- keep export and runtime control as first-class parts of development

## Core Capabilities

### Unified Workflow
ENN-PyTorch provides one integrated path for:

- model construction
- training
- prediction
- checkpointing
- artifact export

This makes it easier to move from experimentation to validation and deployment without rebuilding the surrounding runtime every time.

### Elastic and Distributed Execution
The framework supports elastic and distributed execution patterns for larger-scale environments, including multi-process and hardware-aware runtime behavior.

### Configurable Model Composition
ENN-PyTorch is designed for flexible model construction rather than a single fixed architecture. It supports:

- task-based model composition
- BYOM-style extension
- DAG-oriented workflow structure
- spatio-temporal modeling blocks

### Data Pipeline Support
The data layer is built for practical workloads, including:

- TensorDict-aware processing
- memory-mapped data handling
- `torchdata.nodes`-based pipelines
- structured prediction output handling

### Deployment-Oriented Export
ENN-PyTorch supports export workflows for:

- ONNX
- ONNX Runtime
- `torch.export`
- optional deployment targets through extras

The framework is built so that export is part of the normal workflow, not a separate afterthought.

### Runtime Resilience
A major focus of the project is runtime stability. ENN-PyTorch includes mechanisms for:

- memory-aware batch scaling
- OOM retry and recovery
- gradient accumulation adjustment under pressure
- kernel fallback handling
- output validation and sanitization
- precision-aware execution control

This makes the repository relevant not only as a model project, but also as a runtime engineering project.

## Engineering Principles

ENN-PyTorch is built around the following principles:

- **Reproducibility**  
  The same project should support repeated training, inference, and export without ad hoc rewrites.

- **Operational Stability**  
  Runtime behavior matters. A model that runs once is not enough.

- **Deployment Readiness**  
  Export and downstream integration should be part of the design.

- **Extensibility**  
  New model blocks and workflows should be added without rebuilding the runtime layer.

## Technology Stack

- Python 3.12+
- PyTorch
- Triton
- TensorDict
- torchdata
- ONNX
- ONNX Runtime
- `torch.export`
- optional dataframe integrations such as pandas and polars

## Installation

Install the appropriate PyTorch build for your hardware first, then install the project:

```bash
pip install --upgrade pip
pip install -e .
```

Optional extras can be installed depending on the workflow, such as dataframe integrations or broader deployment backends.

## Minimal Example

```python
import torch
import enn_torch

from enn_torch.core.config import ModelConfig
from enn_torch.runtime.losses import StudentsTLoss

cfg = ModelConfig(
    d_model=128,
    heads=4,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

model = enn_torch.new_model(in_dim=16, out_shape=(1,), config=cfg)

x = torch.randn(32, 16, device=next(model.parameters()).device)
y = torch.randn(32, 1, device=next(model.parameters()).device)

loss_fn = StudentsTLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

model.train()
for _ in range(10):
    pred, loss = model(
        x,
        labels_flat=y.reshape(y.shape[0], -1),
        net_loss=loss_fn,
    )
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

model.eval()
with torch.no_grad():
    pred = model(x, return_loss=False)
```

## Included Examples

- `notebook.ipynb` — worked example workflow
- `raw_data.xlsx` — sample input data for the notebook flow

## Repository Layout

```text
enn_torch/
  core/
  data/
  nn/
  runtime/
README.md
pyproject.toml
notebook.ipynb
raw_data.xlsx
```

## Example Use Cases

ENN-PyTorch is a strong fit for:

- PyTorch model development with export requirements
- distributed or elastic training workflows
- tabular or spatio-temporal learning pipelines
- runtime-aware experimentation on CPU/GPU systems
- deployment-oriented model validation

## Why This Repository Matters

ENN-PyTorch is more than a model implementation repository. It is a runtime and deployment engineering project that combines:

- model construction
- data pipeline design
- execution control
- exportability
- resilience under unstable runtime conditions

This makes it relevant for AI engineering, MLOps, systems-oriented deep learning, and deployment-aware model development.

## License

Source code is licensed under the PolyForm Noncommercial License 1.0.0. Please review the repository license file for full details.
