#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Quick export smoke test. Skips missing backends with a message.

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
from pathlib import Path

import torch

try:
    from stnet.core.graph import from_buffer
except Exception:
    from_buffer = contextlib.nullcontext


def _parse_out_shape(s: str) -> tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("out_shape must be a comma-separated list like '8,8'")
    return tuple(int(p) for p in parts)


def _parse_formats(s: str) -> list[str]:
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    if not parts:
        return []
    return parts


def _dtype_from_str(s: str) -> torch.dtype:
    s = s.strip().lower()
    if s in {"fp32", "float32", "f32"}:
        return torch.float32
    if s in {"fp16", "float16", "f16"}:
        return torch.float16
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {s}")


def _bool_verify(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "verify", False))


def _onnx_check(path: Path) -> bool:
    spec = importlib.util.find_spec("onnx")
    if spec is None:
        print("[skip] onnx checker: package 'onnx' not installed")
        return True
    import onnx  # type: ignore

    try:
        m = onnx.load(str(path))
        onnx.checker.check_model(m)
        print("[ok] onnx checker")
        return True
    except Exception as exc:
        print(f"[fail] onnx checker: {type(exc).__name__}: {exc}")
        return False


def _ort_run(path: Path, x: torch.Tensor, prefer_cuda: bool) -> bool:
    np_spec = importlib.util.find_spec("numpy")
    if np_spec is None:
        print("[skip] onnxruntime run: numpy not installed")
        return True
    ort_spec = importlib.util.find_spec("onnxruntime")
    if ort_spec is None:
        print("[skip] onnxruntime: package 'onnxruntime' not installed")
        return True
    import numpy as _np  # type: ignore
    import onnxruntime as ort  # type: ignore

    providers = ["CPUExecutionProvider"]
    if prefer_cuda:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(str(path), providers=providers)
    except Exception as exc:
        print(f"[fail] onnxruntime load: {type(exc).__name__}: {exc}")
        return False
    try:
        inp = sess.get_inputs()[0]
        inp_name = inp.name
        inp_type = str(getattr(inp, "type", ""))
        x_cpu = x.detach().cpu()
        if "float16" in inp_type:
            x_np = x_cpu.to(torch.float16).numpy()
        elif "bfloat16" in inp_type or "bf16" in inp_type:
            print(
                "[skip] onnxruntime run: input is bfloat16 (numpy dtype unsupported in many envs)"
            )
            return True
        else:
            x_np = x_cpu.to(torch.float32).numpy()
        if not isinstance(x_np, _np.ndarray):
            print("[skip] onnxruntime run: failed to convert input to numpy")
            return True
        outs = sess.run(None, {inp_name: x_np})
        shapes = [getattr(o, "shape", None) for o in outs]
        print(f"[ok] onnxruntime run: outputs={shapes}")
        return True
    except Exception as exc:
        print(f"[fail] onnxruntime run: {type(exc).__name__}: {exc}")
        return False


def _pt2_run(path: Path, x: torch.Tensor, device: torch.device) -> bool:
    try:
        with from_buffer():
            ep = torch.export.load(str(path))
    except Exception as exc:
        print(f"[fail] pt2 load: {type(exc).__name__}: {exc}")
        return False
    try:
        m = ep.module()
    except Exception:
        m = ep
    m.eval()
    x_run = x.detach()
    try:
        if hasattr(m, "to"):
            m = m.to(device=device)
    except Exception:
        try:
            m = m.to("cpu")
        except Exception:
            pass
        x_run = x_run.to("cpu", dtype=torch.float32)
    try:
        x_run = x_run.to(device=device)
    except Exception:
        x_run = x_run.to("cpu", dtype=torch.float32)
    try:
        with torch.no_grad():
            y = m(x_run)
        y0 = y[0] if isinstance(y, (tuple, list)) and y else y
        shape = tuple(y0.shape) if isinstance(y0, torch.Tensor) else type(y0)
        print(f"[ok] pt2 run: output={shape}")
        return True
    except Exception as exc:
        print(f"[fail] pt2 run: {type(exc).__name__}: {exc}")
        return False


def _aoti_run(path: Path, x: torch.Tensor, device: torch.device) -> bool:
    try:
        from torch._inductor import aoti_load_package
    except Exception as exc:
        print(f"[skip] aoti run: torch._inductor not available ({type(exc).__name__}: {exc})")
        return True
    try:
        dev_index = int(device.index or 0) if device.type == "cuda" else -1
        m = aoti_load_package(str(path), device_index=dev_index)
    except Exception as exc:
        print(f"[fail] aoti load: {type(exc).__name__}: {exc}")
        return False
    x_run = x.detach()
    try:
        x_run = x_run.to(device=device)
    except Exception:
        x_run = x_run.to("cpu", dtype=torch.float32)
    try:
        with torch.no_grad():
            y = m(x_run)
        y0 = y[0] if isinstance(y, (tuple, list)) and y else y
        shape = tuple(y0.shape) if isinstance(y0, torch.Tensor) else type(y0)
        print(f"[ok] aoti run: output={shape}")
        return True
    except Exception as exc:
        print(f"[fail] aoti run: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="export_smoke_out", help="Output directory")
    ap.add_argument("--device", type=str, default="cpu", help="cpu|cuda")
    ap.add_argument("--dtype", type=str, default="fp32", help="fp32|fp16|bf16")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--in-dim", type=int, default=64)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--out-shape", type=str, default="8,8")
    ap.add_argument(
        "--formats",
        type=str,
        default="onnx,pt2",
        help="comma-separated: onnx,ort,pt2,aoti,executorch,litert,coreml,tensorrt,tensorflow",
    )
    ap.add_argument(
        "--force-torch-msr",
        action="store_true",
        help="Disable Triton MSR for the run via STNET_MSR_FORCE_TORCH=1",
    )
    bool_opt = getattr(argparse, "BooleanOptionalAction", None)
    if bool_opt is not None:
        ap.add_argument(
            "--verify",
            action=bool_opt,
            default=True,
            help="Verify exported artifacts when possible",
        )
    else:
        ap.set_defaults(verify=True)
        ap.add_argument(
            "--no-verify",
            dest="verify",
            action="store_false",
            help="Disable post-export verification",
        )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.force_torch_msr:
        os.environ["STNET_MSR_FORCE_TORCH"] = "1"

    device = torch.device(args.device)
    dtype = _dtype_from_str(args.dtype)
    out_shape = _parse_out_shape(args.out_shape)
    formats = _parse_formats(args.formats)

    from stnet.config import ModelConfig
    from stnet.nn.architecture import Model
    from stnet.runtime.io import Exporter

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
        compile_mode="disabled",
    )

    model = Model(args.in_dim, out_shape, cfg).to(device=device)
    model.eval()

    x = torch.randn(args.batch, args.seq, args.in_dim, device=device, dtype=dtype)

    print(f"Model: in_dim={args.in_dim}, out_shape={out_shape}, device={device}, dtype={dtype}")
    print(f"Input: {tuple(x.shape)}")

    verify = _bool_verify(args)

    y_ref = None
    if verify:
        try:
            with torch.no_grad():
                y_ref = model.forward_export(x)
            if isinstance(y_ref, torch.Tensor):
                print(f"Ref output: {tuple(y_ref.shape)} dtype={y_ref.dtype} device={y_ref.device}")
            else:
                y_ref = None
        except Exception as exc:
            print(f"[warn] failed to compute reference output: {type(exc).__name__}: {exc}")
            y_ref = None

    ok = True
    for fmt_name in formats:
        ext = {
            "onnx": ".onnx",
            "ort": ".ort",
            "pt2": ".pt2",
            "aoti": ".aoti",
            "executorch": ".pte",
            "litert": ".tflite",
            "tensorflow": ".savedmodel",
            "coreml": ".mlmodel",
            "tensorrt": ".engine",
        }.get(fmt_name)
        if ext is None:
            print(f"[skip] unknown format: {fmt_name}")
            ok = False
            continue
        fmt = Exporter.for_export(ext)
        if fmt is None:
            print(f"[skip] no exporter registered for {fmt_name}")
            ok = False
            continue
        dst = out_dir / f"stnet_smoke{ext}"
        print(f"[export] {fmt_name} -> {dst}")
        try:
            res = fmt.save(model, dst, sample_input=x)
        except Exception as exc:
            ok = False
            print(f"[fail] {fmt_name}: {type(exc).__name__}: {exc}")
            continue
        if res is not None:
            _ = res
        print(f"[ok] {fmt_name}")
        if verify:
            prefer_cuda = bool(device.type == "cuda")
            if fmt_name == "onnx":
                ok = bool(_onnx_check(dst)) and ok
                ok = bool(_ort_run(dst, x, prefer_cuda=prefer_cuda)) and ok
            elif fmt_name == "ort":
                ok = bool(_ort_run(dst, x, prefer_cuda=prefer_cuda)) and ok
            elif fmt_name == "pt2":
                ok = bool(_pt2_run(dst, x, device=device)) and ok
            elif fmt_name == "aoti":
                ok = bool(_aoti_run(dst, x, device=device)) and ok
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
