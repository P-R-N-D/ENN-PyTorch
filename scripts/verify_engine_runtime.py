#!/usr/bin/env python
"""Smoke-test stnet.runtime.train/predict using a raw_data.xlsx subset."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stnet import ModelConfig, PatchConfig, new_model
from stnet.runtime import predict, train

FeatureKey = Tuple[float, ...]
Dataset = Mapping[FeatureKey, torch.Tensor]


def _parse_sheet_tokens(sheet_name: str) -> Tuple[int, int]:
    month_match = re.search(r"(\d+)", sheet_name)
    month = int(month_match.group(1)) if month_match else 0
    weekend = 1 if ("주말" in sheet_name or "공휴일" in sheet_name) else 0
    return month, weekend


def _ensure_tensor(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(list(values), dtype=torch.float32)
    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)
    return tensor


def load_subset(workbook: Path, *, rows_per_sheet: int = 2) -> Dict[FeatureKey, torch.Tensor]:
    if rows_per_sheet <= 0:
        raise ValueError("rows_per_sheet must be >= 1")
    book = pd.ExcelFile(str(workbook))
    label_columns_cache: Dict[str, List[str]] = {}
    route_ids: Dict[str, int] = {}
    segment_ids: Dict[str, int] = {}
    direction_ids: Dict[str, int] = {}
    dataset: Dict[FeatureKey, torch.Tensor] = {}
    for sheet_name in book.sheet_names:
        frame = book.parse(sheet_name)
        if frame.empty:
            continue
        if sheet_name not in label_columns_cache:
            label_columns_cache[sheet_name] = [
                col
                for col in frame.columns
                if isinstance(col, str) and col.strip().endswith("시")
            ]
        label_columns = label_columns_cache[sheet_name]
        if not label_columns:
            continue
        month, weekend_flag = _parse_sheet_tokens(sheet_name)
        subset = frame.head(rows_per_sheet)
        for row_idx, row in subset.iterrows():
            route = str(row.get("노선", "")).strip()
            segment = str(row.get("구간", "")).strip()
            direction = str(row.get("방향", "")).strip()
            route_id = route_ids.setdefault(route, len(route_ids))
            segment_id = segment_ids.setdefault(segment, len(segment_ids))
            direction_id = direction_ids.setdefault(direction, len(direction_ids))
            label_values = row[label_columns].to_numpy(dtype="float32", na_value=0.0)
            if not np.isfinite(label_values).all():
                raise ValueError(f"Found non-finite label row in sheet {sheet_name!r}")
            feature = (
                float(month),
                float(weekend_flag),
                float(route_id),
                float(segment_id),
                float(direction_id),
                float(row_idx),
            )
            dataset[feature] = _ensure_tensor(label_values)
    if not dataset:
        raise RuntimeError("No samples extracted from workbook")
    return dataset


def build_model(in_dim: int, *, out_dim: int = 24) -> torch.nn.Module:
    config = ModelConfig(
        microbatch=8,
        depth=32,
        heads=4,
        spatial_depth=1,
        temporal_depth=1,
        spatial_latents=4,
        temporal_latents=4,
        mlp_ratio=2.0,
        dropout=0.1,
        drop_path=0.0,
        modeling_type="spatiotemporal",
        use_linear_branch=True,
        patch=PatchConfig(),
    )
    return new_model(in_dim=in_dim, out_shape=(out_dim,), config=config)


def run_smoke_test(dataset: Dataset, *, epochs: int = 1) -> Dict[FeatureKey, torch.Tensor]:
    sample_key = next(iter(dataset))
    in_dim = len(sample_key)
    label_shape = next(iter(dataset.values())).shape
    if label_shape != (24,):
        raise RuntimeError(f"Expected label shape (24,), got {label_shape}")
    model = build_model(in_dim)
    trained = train(
        model,
        dataset,
        epochs=epochs,
        batch_size=max(2, len(dataset) // 4),
        val_frac=0.2,
        base_lr=5e-4,
        weight_decay=1e-4,
        warmup_ratio=0.0,
        grad_accum_steps=1,
        overlap_h2d=False,
    )
    predictions = predict(
        trained,
        dataset,
        batch_size=max(2, len(dataset) // 4),
        seed=123,
    )
    if set(predictions.keys()) != set(dataset.keys()):
        raise AssertionError("Prediction keys do not match input keys")
    for key, value in predictions.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Prediction for {key} is not a tensor: {type(value)!r}")
        if value.shape != (24,):
            raise ValueError(f"Prediction for {key} has unexpected shape {tuple(value.shape)}")
    return predictions


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows-per-sheet",
        type=int,
        default=2,
        help="Number of rows to sample from each worksheet",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs to run",
    )
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    workbook = repo_root / "raw_data.xlsx"
    if not workbook.is_file():
        raise FileNotFoundError(f"Could not locate workbook at {workbook}")
    dataset = load_subset(workbook, rows_per_sheet=args.rows_per_sheet)
    predictions = run_smoke_test(dataset, epochs=args.epochs)
    first_key = next(iter(sorted(predictions)))
    print("Sample key:", first_key)
    print("Sample prediction:", predictions[first_key][:5])
    print(f"Validated {len(predictions)} predictions with shape (24,)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
