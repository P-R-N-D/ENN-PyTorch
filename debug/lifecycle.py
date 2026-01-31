# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import importlib
import os
import re
import sys
import threading
import time
from types import ModuleType
from typing import Any, Callable, Dict, List, Sequence, Tuple, TypeVar

import numpy as np
import psutil
import torch
from tensordict import TensorDict

from enn_torch.core.config import ModelConfig, PatchConfig
from enn_torch.core.system import get_device
from enn_torch.runtime.workflows import new_model, predict, train

COL_DIR = "방향"
COL_ROUTE = "노선"
COL_SECTION = "구간"
DAY_MAP = {
    "월요일": 0,
    "화요일": 1,
    "수요일": 2,
    "목요일": 3,
    "금요일": 4,
    "토요일": 5,
    "일요일": 6,
}
DIR_DOWN = "하행"
DIR_UP = "상행"
HOUR_SUFFIX = "시"
T = TypeVar("T")


def _canonical_section(val: object) -> str:
    parts = [p.strip() for p in str(val).split("-") if str(p).strip()]
    if len(parts) <= 1:
        return str(val).strip()
    parts.sort()
    return "-".join(parts)


def _require_tabular_deps() -> tuple[ModuleType, Callable[..., object]]:
    if importlib.util.find_spec("pandas") is None:
        raise ImportError(
            "pandas is required for dataset scripts; install with `pip install -e .[pandas]`"
        )
    if importlib.util.find_spec("openpyxl") is None:
        raise ImportError(
            "openpyxl is required for dataset scripts; install with `pip install -e .[pandas]`"
        )
    pd = importlib.import_module("pandas")
    load_workbook = importlib.import_module("openpyxl").load_workbook
    return pd, load_workbook


def parse_sheet_name(name: str) -> tuple[int, str]:
    m = re.search(r"(\d+)\uc6d4", name)
    if not m:
        raise ValueError(f"Could not find month in sheet name: {name}")
    month = int(m.group(1))
    name_clean = name.replace(m.group(0), "")
    day_kind = None
    for k in DAY_MAP.keys():
        if k in name_clean:
            day_kind = k
            break
    if day_kind is None:
        raise ValueError(f"Could not find weekday in sheet name: {name}")
    return month, day_kind


def build_dataset(xlsx_path: str) -> Dict[str, Any]:
    pd, load_workbook = _require_tabular_deps()
    HOURS = [f"{h:02d}{HOUR_SUFFIX}" for h in range(24)]
    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    sheet_names: Sequence[str] = list(wb.sheetnames)
    wb.close()
    frames: list[pd.DataFrame] = []
    for sh in sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sh)
        df.rename(columns=lambda c: str(c).strip(), inplace=True)
        required = {COL_SECTION, COL_DIR}
        if not required.issubset(df.columns):
            print(f"[warn] skipping sheet missing required columns: {sh}")
            continue
        hour_cols = [c for c in df.columns if c in HOURS]
        if not hour_cols:
            print(f"[warn] skipping sheet with no hour columns: {sh}")
            continue
        try:
            month, day_kind = parse_sheet_name(sh)
        except Exception as e:
            print(f"[warn] skipping sheet with bad name {sh!r}: {e}")
            continue
        if COL_ROUTE not in df.columns:
            df[COL_ROUTE] = "수도권제1순환선"
        df.insert(0, "row_in_sheet", np.arange(len(df), dtype=np.int64))
        long_df = (
            df.melt(
                id_vars=["row_in_sheet", COL_ROUTE, COL_SECTION, COL_DIR],
                value_vars=hour_cols,
                var_name="시간",
                value_name="지표",
            )
            .dropna(subset=["지표"])
            .copy()
        )
        long_df["월"] = month
        long_df["일종"] = day_kind
        frames.append(long_df)
    if not frames:
        raise RuntimeError("No valid sheets found in workbook")
    long_df = pd.concat(frames, axis=0, ignore_index=True)
    long_df["시간"] = (
        long_df["시간"]
        .astype(str)
        .str.replace(HOUR_SUFFIX, "", regex=False)
        .astype(int)
    )
    long_df["지표"] = long_df["지표"].astype(float)
    long_df["요일타입_id"] = long_df["일종"].map(DAY_MAP).astype(int)
    long_df["방향_id"] = (
        long_df[COL_DIR].map({DIR_UP: 0, DIR_DOWN: 1}).astype(int)
    )
    long_df["canonical_section"] = long_df[COL_SECTION].apply(
        _canonical_section
    )
    long_df["seg_key"] = (
        long_df[COL_ROUTE].astype(str).str.strip()
        + "|"
        + long_df["canonical_section"]
    )
    seg_meta = (
        long_df[["seg_key", COL_ROUTE, "canonical_section"]]
        .drop_duplicates()
        .sort_values("seg_key")
        .reset_index(drop=True)
    )
    seg_meta["seg_idx"] = np.arange(len(seg_meta), dtype=np.int64)
    long_df = long_df.merge(
        seg_meta[["seg_key", "seg_idx"]], on="seg_key", how="left"
    )
    S_orig = int(seg_meta.shape[0])
    T_orig = 24
    group_cols = ["월", "요일타입_id", "방향_id"]
    groups_df = (
        long_df[group_cols]
        .drop_duplicates()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    X_keys: List[Tuple[int, int, int]] = [
        tuple(map(int, row)) for row in groups_df.to_numpy()
    ]
    B = len(X_keys)
    full_grid = groups_df.merge(seg_meta[["seg_idx"]], how="cross")
    pivot = (
        long_df.pivot_table(
            index=group_cols + ["seg_idx"],
            columns="시간",
            values="지표",
            aggfunc="first",
        )
        .reindex(columns=list(range(T_orig)))
        .fillna(0.0)
        .reset_index()
    )
    y_full = (
        full_grid.merge(pivot, on=group_cols + ["seg_idx"], how="left")
        .fillna(0.0)
        .sort_values(group_cols + ["seg_idx"])
    )
    y_vals = y_full[list(range(T_orig))].to_numpy(dtype=np.float32)
    Y_np = y_vals.reshape(B, S_orig, T_orig)
    row_map = (
        long_df.groupby(group_cols + ["seg_idx"])["row_in_sheet"]
        .min()
        .reset_index()
    )
    row_full = (
        full_grid.merge(row_map, on=group_cols + ["seg_idx"], how="left")
        .fillna(-1)
        .sort_values(group_cols + ["seg_idx"])
    )
    row_ids_np = (
        row_full["row_in_sheet"].to_numpy(dtype=np.int64).reshape(B, S_orig)
    )
    grid_dim = max(S_orig, T_orig)
    Y_pad = np.zeros((B, grid_dim, grid_dim), dtype=np.float32)
    Y_pad[:, :S_orig, :T_orig] = Y_np
    row_ids_pad = -np.ones((B, grid_dim), dtype=np.int64)
    row_ids_pad[:, :S_orig] = row_ids_np
    X_tensor = torch.tensor(X_keys, dtype=torch.float32)
    Y_tensor = torch.from_numpy(Y_pad)
    row_ids_tensor = torch.from_numpy(row_ids_pad)
    td_train = TensorDict(
        {"X": X_tensor, "Y": Y_tensor, "row_ids": row_ids_tensor},
        batch_size=[B],
    )
    return {
        "td_train": td_train,
        "S": grid_dim,
        "T": grid_dim,
        "B": B,
        "seg_meta": seg_meta,
        "group_keys": X_keys,
        "S_orig": S_orig,
        "T_orig": T_orig,
    }


def monitor_run(fn: Callable[[], T]) -> tuple[T, dict[str, object]]:
    cpu_samples: List[List[float]] = []
    mem_series: List[int] = []
    mem_peak = 0
    stop = threading.Event()
    proc = psutil.Process()
    psutil.cpu_percent(interval=None, percpu=True)

    def sampler() -> None:
        nonlocal mem_peak
        while not stop.is_set():
            cpu_vals = psutil.cpu_percent(interval=1.0, percpu=True)
            cpu_samples.append(cpu_vals)
            mem = 0
            procs = [proc]
            with contextlib.suppress(Exception):
                procs += proc.children(recursive=True)
            for p in procs:
                try:
                    mem += p.memory_info().rss
                except Exception:
                    pass
            mem_series.append(mem)
            mem_peak = max(mem_peak, mem)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    start = time.time()
    result = fn()
    stop.set()
    t.join()
    duration = time.time() - start
    if cpu_samples:
        arr = np.array(cpu_samples)
        cpu_avg = arr.mean(axis=0).tolist()
        cpu_peak = arr.max(axis=0).tolist()
    else:
        cpu_avg = []
        cpu_peak = []
    return result, {
        "duration_s": duration,
        "cpu_avg": cpu_avg,
        "cpu_peak": cpu_peak,
        "mem_peak": mem_peak,
        "cpu_samples": len(cpu_samples),
        "mem_samples": len(mem_series),
    }


def main() -> None:
    print("PYTHON_GIL env:", os.environ.get("PYTHON_GIL"))
    print("sys._is_gil_enabled available:", hasattr(sys, "_is_gil_enabled"))
    print("GIL enabled?:", getattr(sys, "_is_gil_enabled", lambda: None)())
    os.environ.setdefault("ENN_PREBATCH", "1")
    excel_path = os.path.abspath("raw_data.xlsx")
    if not os.path.isfile(excel_path):
        raise FileNotFoundError(excel_path)
    print("Excel path:", excel_path)
    print("[load] building dataset...")
    info = build_dataset(excel_path)
    td_train = info["td_train"]
    S, T, B = info["S"], info["T"], info["B"]
    S_orig, T_orig = info["S_orig"], info["T_orig"]
    print(
        f"Dataset built: B={B} groups, S_orig={S_orig}, T_orig={T_orig}, padded_grid={S}x{T}"
    )
    print(
        f"td_train batch_size={td_train.batch_size}, X shape={tuple(td_train['X'].shape)}, Y shape={tuple(td_train['Y'].shape)}"
    )
    device = get_device()
    print("Device:", device)
    patch = PatchConfig(
        is_cube=True,
        grid_size_3d=(S, T, 1),
        patch_size_3d=(1, 1, 1),
        use_padding=True,
    )
    config = ModelConfig(
        device=device,
        patch=patch,
        normalization_method="layernorm",
        d_model=256,
        heads=2,
        mlp_ratio=2.0,
        dropout=0.05,
        drop_path=0.05,
        spatial_depth=2,
        temporal_depth=2,
        spatial_latents=32,
        temporal_latents=32,
        modeling_type="spatiotemporal",
        compile_mode="disabled",
    )
    model = new_model(
        in_dim=td_train["X"].shape[1], out_shape=(S, T), config=config
    ).to(device)
    with contextlib.suppress(Exception):
        model.add_task("extra_spatial", mode="spatial", weight=0.25)
        model.update_task("extra_spatial", weight=0.5)
        print("[lifecycle] tasks:", model.list_tasks())
    train_epochs = 6
    print(
        "[train] starting... (elastic_launch inside enn_torch.runtime.workflows.train)"
    )
    trained_model, train_metrics = monitor_run(
        lambda: train(
            model,
            td_train,
            epochs=train_epochs,
            base_lr=3e-3,
            weight_decay=1e-4,
            val_frac=0.1,
            max_nodes=1,
        )
    )
    print("[train] done")
    print("train duration (s):", train_metrics["duration_s"])
    print("train CPU avg per core:", train_metrics["cpu_avg"])
    print("train CPU peak per core:", train_metrics["cpu_peak"])
    print(
        "train peak RSS MB:", round(train_metrics["mem_peak"] / (1024**2), 2)
    )
    hist = []
    with contextlib.suppress(Exception):
        hist = trained_model.history()
    if hist:
        print("train history entries:", len(hist))
        print("last history entry:", hist[-1])
    else:
        print("no train history recorded")
    infer_td = TensorDict({"X": td_train["X"]}, batch_size=td_train.batch_size)
    pred_path = os.path.abspath("predictions.h5")
    if os.path.exists(pred_path):
        os.remove(pred_path)
    print("[predict] starting...")
    pred_result, pred_metrics = monitor_run(
        lambda: predict(
            trained_model,
            infer_td,
            output="file",
            path=pred_path,
            overwrite="replace",
            max_nodes=1,
            out_shape=(S, T),
        )
    )
    print("[predict] done ->", pred_path)
    print("predict duration (s):", pred_metrics["duration_s"])
    print("predict CPU avg per core:", pred_metrics["cpu_avg"])
    print("predict CPU peak per core:", pred_metrics["cpu_peak"])
    print(
        "predict peak RSS MB:", round(pred_metrics["mem_peak"] / (1024**2), 2)
    )
    Y_pred = pred_result["Y"]
    if hasattr(Y_pred, "detach"):
        Y_pred_t = Y_pred.detach().cpu()
    else:
        Y_pred_t = torch.as_tensor(Y_pred)
    print("Prediction tensor shape:", tuple(Y_pred_t.shape))
    print(
        "Prediction stats: min=%.4f max=%.4f mean=%.4f std=%.4f"
        % (
            float(Y_pred_t.min().item()),
            float(Y_pred_t.max().item()),
            float(Y_pred_t.mean().item()),
            float(Y_pred_t.std().item()),
        )
    )
    if hasattr(pred_result, "close"):
        pred_result.close()
    print("[done]")


if __name__ == "__main__":
    main()
