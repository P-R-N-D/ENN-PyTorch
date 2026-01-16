from __future__ import annotations

import contextlib
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from stnet.core.tensor import extract_tensor, from_buffer
from wsl_nogil_train import build_dataset
from stnet.api import new_model, train
from stnet.config import ModelConfig, PatchConfig
from stnet.runtime.io import Exporter


def _as_path_list(out: Any, fallback: Path) -> list[str]:
    if out is None:
        return [str(fallback)]
    if isinstance(out, (str, Path)):
        return [str(out)]
    if isinstance(out, (tuple, list)):
        flat: list[str] = []
        for item in out:
            if item is None:
                continue
            if isinstance(item, (str, Path)):
                flat.append(str(item))
            else:
                flat.append(str(item))
        return flat if flat else [str(fallback)]
    return [str(out)]


def _stats_np(arr: np.ndarray) -> dict[str, float | list[int]]:
    arr = np.asarray(arr)
    return {
        "shape": list(arr.shape),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


@contextlib.contextmanager
def _temp_env(k: str, v: str):
    old = os.environ.get(k)
    os.environ[k] = v
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = old


def export_and_validate(
    model: torch.nn.Module, sample: torch.Tensor, td_train, out_dir: Path
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*LeafSpec.*",
    )
    targets = {
        "pt2": out_dir / "model.pt2",
        "onnx": out_dir / "model.onnx",
        "ort": out_dir / "model.ort",
        "executorch": out_dir / "model.pte",
        "tensorflow": out_dir / "model.savedmodel",
        "litert": out_dir / "model.tflite",
        "coreml": out_dir / "model.mlmodel",
        "tensorrt": out_dir / "model.engine",
    }
    results: Dict[str, Any] = {}
    with _temp_env("STNET_DISABLE_PIECEWISE_CALIB", "1"):
        for name, path in targets.items():
            fmt = Exporter.for_export(path.suffix if path.suffix else str(path))
            try:
                if fmt is None:
                    raise RuntimeError("no exporter registered")
                save_kwargs: Dict[str, Any] = {
                    "sample_input": sample,
                    "dynamic_batch": True,
                }
                out = fmt.save(model, path, **save_kwargs)
                results[name] = {
                    "status": "ok",
                    "paths": _as_path_list(out, path),
                }
            except ImportError as exc:
                if name in ("pt2", "onnx"):
                    results[name] = {"status": "error", "error": repr(exc)}
                else:
                    results[name] = {
                        "status": "skipped",
                        "reason": "missing_optional_dependency",
                        "error": repr(exc),
                    }
            except Exception as exc:  # pragma: no cover - diagnostic output
                results[name] = {"status": "error", "error": repr(exc)}

    validation: Dict[str, Any] = {}
    try:
        y = td_train["Y"].detach().cpu().numpy()
        validation["label_stats"] = _stats_np(y)
    except Exception as exc:
        validation["label_stats_error"] = repr(exc)
    pt2_path = targets["pt2"]
    if pt2_path.exists():
        with _temp_env("STNET_DISABLE_PIECEWISE_CALIB", "1"):
            try:
                with from_buffer():
                    ep = torch.export.load(str(pt2_path))
                with torch.no_grad():
                    pt2_out = extract_tensor(ep.module()(sample))
                    torch_out = extract_tensor(
                        model.forward_export(sample)
                        if hasattr(model, "forward_export")
                        else model(sample, return_loss=False)
                    )
                pt2_np = pt2_out.detach().cpu().numpy()
                torch_np = torch_out.detach().cpu().numpy()
                validation["pt2_mae"] = float(np.mean(np.abs(pt2_np - torch_np)))
                validation["pt2_out_stats"] = _stats_np(pt2_np)
            except Exception as exc:
                validation["pt2_error"] = repr(exc)
        def _truthy(v: str) -> bool:
            return v.strip().lower() in ("1", "true", "yes", "y", "on")

        do_alt = _truthy(os.environ.get("STNET_VALIDATE_ALT_BATCH", "0"))
        if (
            "pt2_error" not in validation
            and do_alt
            and isinstance(sample, torch.Tensor)
            and sample.ndim >= 2
            and int(sample.shape[0]) > 1
        ):
            try:
                alt_n = max(1, int(sample.shape[0]) // 2)
                alt = sample[:alt_n]
                with torch.no_grad():
                    alt_pt2 = extract_tensor(ep.module()(alt))
                    alt_torch = extract_tensor(
                        model.forward_export(alt)
                        if hasattr(model, "forward_export")
                        else model(alt, return_loss=False)
                    )
                validation["pt2_mae_alt"] = float(
                    np.mean(
                        np.abs(
                            alt_pt2.detach().cpu().numpy() - alt_torch.detach().cpu().numpy()
                        )
                    )
                )
            except Exception as exc:
                validation["pt2_mae_alt_error"] = repr(exc)

    for name in ("onnx", "ort"):
        path = targets[name]
        if path.exists():
            with _temp_env("STNET_DISABLE_PIECEWISE_CALIB", "1"):
                try:
                    import onnxruntime as ort
                    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                    inp_name = sess.get_inputs()[0].name
                    out = sess.run(
                        None, {inp_name: sample.detach().cpu().numpy().astype(np.float32)}
                    )[0]
                    validation[f"{name}_out_stats"] = _stats_np(out)
                except Exception as exc:
                    validation[f"{name}_error"] = repr(exc)
    return {"exports": results, "validation": validation}


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
    model = new_model(in_dim=td_train["X"].shape[1], out_shape=(S, T), config=cfg).to(device)
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
    stats = export_and_validate(model, sample, td_train, Path("export_artifacts"))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
