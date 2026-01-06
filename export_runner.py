from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import onnxruntime as ort

from wsl_nogil_train import build_dataset
from stnet.api import new_model, train
from stnet.config import ModelConfig, PatchConfig
from stnet.runtime.io import Exporter


def export_and_validate(
    model: torch.nn.Module, sample: torch.Tensor, out_dir: Path
) -> Dict[str, Any]:
    out_dir.mkdir(exist_ok=True)
    targets = {
        "torchscript": out_dir / "model.ts",
        "onnx": out_dir / "model.onnx",
        "executorch": out_dir / "model.pte",
        "nnef": out_dir / "model.nnef",
        "tensorflow": out_dir / "model.savedmodel",
        "litert": out_dir / "model.tflite",
    }
    results: Dict[str, Any] = {}
    for name, path in targets.items():
        fmt = Exporter.for_export(path.suffix if path.suffix else str(path))
        try:
            if fmt is None:
                raise RuntimeError("no exporter registered")
            save_kwargs: Dict[str, Any] = {"sample_input": sample}
            if name == "torchscript":
                # TorchScript scripting dislikes **kwargs in Model.forward; trace is safer here.
                save_kwargs["method"] = "trace"
            out = fmt.save(model, path, **save_kwargs)
            results[name] = {
                "status": "ok",
                "path": str(out if out is not None else path),
            }
        except Exception as exc:  # pragma: no cover - diagnostic output
            results[name] = {"status": "error", "error": repr(exc)}

    # TorchScript validation
    validations: Dict[str, Any] = {}
    ts_path = targets["torchscript"]
    if ts_path.exists():
        try:
            ts_model = torch.jit.load(str(ts_path))
            with torch.no_grad():
                ts_out = ts_model(sample)
            torch_out = model(sample, return_loss=False)
            ts_mae = float(torch.mean(torch.abs(ts_out - torch_out)).item())
            validations["torchscript_mae"] = ts_mae
        except Exception as exc:
            validations["torchscript_error"] = repr(exc)

    onnx_path = targets["onnx"]
    if onnx_path.exists():
        try:
            sess = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            inp_name = sess.get_inputs()[0].name
            onnx_out = sess.run(
                None, {inp_name: sample.detach().cpu().numpy().astype(np.float32)}
            )[0]
            torch_out = model(sample, return_loss=False).detach().cpu().numpy()
            validations["onnx_mae"] = float(np.mean(np.abs(torch_out - onnx_out)))
        except Exception as exc:
            validations["onnx_error"] = repr(exc)
    return {"exports": results, "validation": validations}


def main() -> None:
    os.environ.setdefault("STNET_PREBATCH", "1")
    os.environ.setdefault("STNET_PREFETCH_FACTOR", "1")
    data = build_dataset("raw_data.xlsx")
    td_train = data["td_train"]
    S = data["S"]
    T = data["T"]
    print(f"[export] dataset groups={data['B']} grid={S}x{T}")
    device = torch.device("cpu")
    patch = PatchConfig(
        is_cube=True, grid_size_3d=(S, T, 1), patch_size_3d=(1, 1, 1), use_padding=True
    )
    cfg = ModelConfig(
        device=device,
        patch=patch,
        normalization_method="layernorm",
        d_model=192,
        heads=2,
        mlp_ratio=2.0,
        dropout=0.05,
        drop_path=0.05,
        spatial_depth=2,
        temporal_depth=2,
        spatial_latents=24,
        temporal_latents=24,
        modeling_type="spatiotemporal",
        compile_mode="disabled",
    )
    model = new_model(in_dim=td_train["X"].shape[1], out_shape=(S, T), config=cfg).to(
        device
    )
    print("[export] training short run (epochs=2) for exportable weights")
    train(
        model,
        td_train,
        epochs=2,
        base_lr=3e-3,
        weight_decay=1e-4,
        val_frac=0.1,
        max_nodes=1,
    )
    model.eval()
    sample = td_train["X"][:4].to(device)
    stats = export_and_validate(model, sample, Path("export_artifacts"))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
