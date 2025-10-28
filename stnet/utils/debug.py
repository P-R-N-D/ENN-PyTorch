# -*- coding: utf-8 -*-
"""Debug utilities shared across runtime and model modules."""
from __future__ import annotations

from typing import Any
import os, time, contextlib, signal, faulthandler

import torch
import torch.nn.functional as F


__all__ = ["is_fake_tensor"]

try:  # pragma: no cover - optional dependency
    from torchdistx.fake import is_fake as _tdx_is_fake  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - torchdistx not installed
    _tdx_is_fake = None  # type: ignore

try:  # pragma: no cover - private API best-effort
    from torch._subclasses.fake_tensor import FakeTensor  # type: ignore
except Exception:  # pragma: no cover - fallback when private API unavailable
    FakeTensor = tuple()  # type: ignore


def is_fake_tensor(value: Any) -> bool:
    """Return ``True`` when ``value`` references a FakeTensor placeholder."""
    if not isinstance(value, torch.Tensor):
        return False
    if _tdx_is_fake is not None:
        try:
            return bool(_tdx_is_fake(value))
        except Exception:
            # torchdistx is optional; fall back to local heuristics when it errors.
            pass
    return isinstance(value, FakeTensor) or getattr(value, "fake_mode", None) is not None

def log_error() -> None:
    # 0) basic env (inherit from parent; ensure sane defaults)
    os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
    # disable fragile fused SDPA paths by default (math fallback like CPU)
    os.environ.setdefault("PYTORCH_SDP_DISABLE_FLASH_ATTENTION", "1")
    os.environ.setdefault("PYTORCH_SDP_DISABLE_MEM_EFFICIENT",   "1")
    os.environ.setdefault("PYTORCH_SDP_DISABLE_FAST_PATH",       "1")
    # TE가 설치돼 있어도 Pytorch 모드로만 (JAX 경로 차단)
    os.environ.setdefault("NVTE_FRAMEWORK", "pytorch")

    # 1) per-rank signal dump file
    log_dir = os.environ.get("STF_ELASTIC_LOGDIR", "/tmp")
    try:
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    except Exception:
        rank = 0
    sig_path = os.path.join(log_dir, f"signal_rank{rank}.log")
    try:
        fh = open(sig_path, "w", buffering=1)
        faulthandler.enable(all_threads=True, file=fh)
        for sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS):
            with contextlib.suppress(Exception):
                faulthandler.register(sig, file=fh, all_threads=True, chain=True)
        print(f"[worker dbg] signal dump -> {sig_path}", flush=True)
    except Exception:
        pass

    # 2) optional SDPA debug hook (q/k/v/mask shapes)
    if os.environ.get("STF_SDPA_HOOK", "1") == "1":
        if not hasattr(F, "_orig_sdpa"):
            F._orig_sdpa = F.scaled_dot_product_attention  # type: ignore[attr-defined]
            def _dbg_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
                try:
                    qm, km = tuple(q.shape), tuple(k.shape)
                    vm     = tuple(v.shape)
                    mm     = None if attn_mask is None else tuple(attn_mask.shape)
                    print(f"[SDPA] dev={q.device} q={qm} k={km} v={vm} mask={mm} causal={is_causal}",
                          flush=True)
                except Exception:
                    pass
                return F._orig_sdpa(q, k, v, attn_mask=attn_mask,
                                    dropout_p=dropout_p, is_causal=is_causal)
            F.scaled_dot_product_attention = _dbg_sdpa
