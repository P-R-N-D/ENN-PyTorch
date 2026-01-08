#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

import torch


def _dtype_from_str(s: str) -> torch.dtype:
    s = s.strip().lower()
    if s in {"fp32", "float32", "f32"}:
        return torch.float32
    if s in {"fp16", "float16", "f16"}:
        return torch.float16
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {s}")


def _torch_load(path: Path, map_location: str | torch.device):
    kw = {}
    try:
        if "weights_only" in inspect.signature(torch.load).parameters:
            kw["weights_only"] = False
    except Exception:
        pass
    return torch.load(str(path), map_location=map_location, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="pickle_smoke_out", help="Output directory")
    ap.add_argument("--device", type=str, default="cpu", help="cpu|cuda")
    ap.add_argument("--dtype", type=str, default="fp32", help="fp32|fp16|bf16")
    ap.add_argument("--compile-mode", type=str, default="disabled")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--in-dim", type=int, default=64)
    ap.add_argument("--out-shape", type=str, default="8,8")
    ap.add_argument(
        "--force-torch-msr",
        action="store_true",
        help="Disable Triton MSR for the run via STNET_MSR_FORCE_TORCH=1",
    )
    args = ap.parse_args()

    if args.force_torch_msr:
        os.environ["STNET_MSR_FORCE_TORCH"] = "1"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stnet_model.pkl"

    device = torch.device(args.device)
    dtype = _dtype_from_str(args.dtype)
    out_shape = tuple(int(p.strip()) for p in str(args.out_shape).split(",") if p.strip())
    if not out_shape:
        raise ValueError("out-shape must be like '8,8'")

    from stnet.config import ModelConfig
    from stnet.nn.architecture import Model

    cfg = ModelConfig(
        device=str(device),
        d_model=64,
        heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        drop_path=0.0,
        spatial_depth=1,
        temporal_depth=1,
        spatial_latents=8,
        temporal_latents=8,
        modeling_type="st",
        compile_mode=str(getattr(args, "compile_mode", "disabled")),
    )

    model = Model(int(args.in_dim), out_shape, cfg).to(device=device, dtype=dtype)
    model.eval()

    x = torch.randn(int(args.batch), int(args.seq), int(args.in_dim), device=device, dtype=dtype)

    with torch.no_grad():
        y_ref = model.forward_export(x)
    if not isinstance(y_ref, torch.Tensor):
        raise RuntimeError("forward_export did not return a Tensor")

    try:
        torch.save(model, str(path))
    except Exception as exc:
        print(f"[fail] torch.save(model): {type(exc).__name__}: {exc}")
        return 2
    print(f"[ok] torch.save(model) -> {path}")

    try:
        m2 = _torch_load(path, map_location="cpu")
    except Exception as exc:
        print(f"[fail] torch.load(model): {type(exc).__name__}: {exc}")
        return 2

    if not isinstance(m2, torch.nn.Module):
        print(f"[fail] torch.load returned {type(m2).__name__}, expected nn.Module")
        return 2

    m2.eval()
    try:
        m2.to(device=device, dtype=dtype)
    except Exception:
        m2.to("cpu", dtype=torch.float32)

    x2 = x.detach().to(next(m2.parameters()).device, dtype=next(m2.parameters()).dtype)
    with torch.no_grad():
        y2 = m2.forward_export(x2) if hasattr(m2, "forward_export") else m2(x2)

    if not isinstance(y2, torch.Tensor):
        print("[fail] loaded model forward did not return Tensor")
        return 2

    print(
        f"[ok] torch.load(model) and run: out={tuple(y2.shape)} dtype={y2.dtype} device={y2.device}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
