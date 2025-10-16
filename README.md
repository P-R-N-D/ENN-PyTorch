# STNet-PyTorch

## Overview
This repository provides a PyTorch implementation of the STNet architecture for joint spatial and temporal modeling, exposing a high-level workflow API for model construction, training, inference, and export utilities.

## Key components
- **Configurable architecture** – `stnet.architecture.network.Config` defines depth, attention heads, patching strategy, compilation options, and other hyperparameters that tailor the spatio-temporal transformer.
- **Data definition aliases** – `_normalize_data_definition` interprets spatial (`ss`, `spatial`), temporal (`tt`, `temporal`), and spatio-temporal (`st`, `ts`, `spatiotemporal`, etc.) shorthands so configuration files and user input remain ergonomic.
- **Workflow facade** – `stnet.workflow` re-exports lifecycle helpers such as `new_model`, `save_model`, `load_model`, `train`, `predict`, and multiple export utilities (TorchScript, ONNX, TensorRT, Core ML, ExecuTorch, TensorFlow, LiteRT).

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
   Additional extras include `ao`, `gds`, `te`, `intel`, `ucx`, and `scale` as defined in `pyproject.toml`.

## Quick start
```python
from stnet.workflow import Config, new_model, save_model, load_model, train, predict, to_script

config = Config(data_definition="spatiotemporal", depth=64, heads=4)
model = new_model(in_dim=1024, out_shape=(10,), config=config)

# Prepare a mapping from sample keys to feature/label tensors.
# Features must flatten to the configured input dimension; labels match `out_shape`.
training_batch = {
    ("segment", 0): (features_tensor, labels_tensor),
    # ... add more key -> (features, labels) pairs ...
}

trained = train(model, training_batch, epochs=1, batch_size=8)
preds = predict(trained, {("segment", 0): (features_tensor, None)})

save_path = save_model(trained, "checkpoints/stnet.pt")
restored = load_model(save_path)

scripted = to_script(restored)  # TorchScript export
```
The workflow helpers manage distributed checkpoints, mixed precision, exporter requirements, and memory-mapped datasets internally, letting you focus on preparing feature tensors and configuration hyperparameters.

## Exporting for inference
Exporter helpers automatically check for optional dependencies and raise informative errors if a backend such as ONNX, TensorFlow, Core ML, TensorRT, LiteRT, or ExecuTorch is unavailable. Install the `export` extra to enable the full conversion toolkit.

## License
The project is distributed under a proprietary license as declared in `pyproject.toml`.
