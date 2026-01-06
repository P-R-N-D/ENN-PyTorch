#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Streaming state smoke test for forward_stream().

from __future__ import annotations

import argparse
import os
import threading
from dataclasses import dataclass

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


@dataclass
class _Result:
    ok: bool = True
    err: str = ""


def _worker(
    idx: int,
    model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
    seq: int,
    in_dim: int,
    iters: int,
    barrier: threading.Barrier | None,
    out: list[_Result],
) -> None:
    state = None
    try:
        if barrier is not None:
            barrier.wait()
        for _ in range(int(iters)):
            x = torch.randn(
                int(batch), int(seq), int(in_dim), device=device, dtype=dtype
            )
            y, state = model.forward_stream(x, temporal_state=state)
            if not isinstance(y, torch.Tensor):
                raise RuntimeError("forward_stream did not return Tensor output")
            if not isinstance(state, torch.Tensor):
                raise RuntimeError("forward_stream did not return Tensor state")
            if state.dim() != 4:
                raise RuntimeError(
                    f"state dim mismatch: expected 4D, got {state.dim()}D"
                )
            if not state.is_contiguous():
                raise RuntimeError("state is not contiguous")
            if state.device != device:
                raise RuntimeError(f"state device mismatch: {state.device} vs {device}")
            if state.dtype != dtype:
                raise RuntimeError(f"state dtype mismatch: {state.dtype} vs {dtype}")
            if not y.is_contiguous():
                raise RuntimeError("output is not contiguous")
    except Exception as exc:
        out[idx].ok = False
        out[idx].err = f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu", help="cpu|cuda")
    ap.add_argument("--dtype", type=str, default="fp32", help="fp32|fp16|bf16")
    ap.add_argument(
        "--compile-mode",
        type=str,
        default="disabled",
        help="torch.compile mode: disabled|reduce-overhead|max-autotune|max-autotune-no-cudagraphs|aot-eager",
    )
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--iters", type=int, default=20)
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

    device = torch.device(args.device)
    dtype = _dtype_from_str(args.dtype)
    out_shape = tuple(
        int(p.strip()) for p in str(args.out_shape).split(",") if p.strip()
    )
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

    model = Model(int(args.in_dim), out_shape, cfg).to(device=device)
    model.eval()
    model.to(dtype=dtype)

    n = max(1, int(args.threads))
    barrier = threading.Barrier(n) if n > 1 else None
    results = [_Result() for _ in range(n)]
    threads: list[threading.Thread] = []
    for i in range(n):
        t = threading.Thread(
            target=_worker,
            args=(
                i,
                model,
                device,
                dtype,
                int(args.batch),
                int(args.seq),
                int(args.in_dim),
                int(args.iters),
                barrier,
                results,
            ),
            daemon=True,
        )
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    ok = all(r.ok for r in results)
    if ok:
        print(
            f"[ok] forward_stream smoke: threads={n} iters={int(args.iters)} device={device} dtype={dtype}"
        )
        return 0
    for i, r in enumerate(results):
        if not r.ok:
            print(f"[fail] thread#{i}: {r.err}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
