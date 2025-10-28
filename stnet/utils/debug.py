# -*- coding: utf-8 -*-
"""Debug utilities shared across runtime and model modules."""
from __future__ import annotations

from typing import Any
import os, time, contextlib, signal, faulthandler, time

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

def install_compile_monitor():
    if os.environ.get("STF_VERBOSE_CGR", "0") != "1":
        return

    # 0) 파이토치 내부 로거를 stdio로 내보내도록 (TORCH_LOGS는 노트북 셀에서 이미 셋)
    with contextlib.suppress(Exception):
        import torch._logging as _tlog
        _tlog.set_logs(dynamo=True, aot=True, aot_graphs=True, aot_runtime=True,
                       inductor=True, graph_breaks=True, recompiles=True)

    # 1) Inductor CUDA graphs 경로만 얇게 래핑
    try:
        import torch._inductor.cudagraph_trees as _cgr

        def wrap(fn, tag):
            def inner(*a, **kw):
                try:
                    print(f"[CGR][enter] {tag}", flush=True)
                except Exception:
                    pass
                try:
                    return fn(*a, **kw)
                finally:
                    try:
                        print(f"[CGR][exit ] {tag}", flush=True)
                    except Exception:
                        pass
            return inner

        # 버전마다 존재하는 심볼만 선택적으로 래핑
        for name in ("run", "run_eager", "_run", "cudagraphify", "add_function"):
            if hasattr(_cgr, name):
                setattr(_cgr, name, wrap(getattr(_cgr, name), f"cudagraph_trees.{name}"))
    except Exception:
        pass

    # 2) AOTAutograd backward 진입/탈출만 로그
    try:
        import torch._functorch._aot_autograd.runtime_wrappers as _rw
        if hasattr(_rw, "_backward_impl"):
            _orig_bwd = _rw._backward_impl
            def _wrapped_bwd(*a, **kw):
                print("[AOT][enter] backward_impl", flush=True)
                try:
                    return _orig_bwd(*a, **kw)
                finally:
                    print("[AOT][exit ] backward_impl", flush=True)
            _rw._backward_impl = _wrapped_bwd
    except Exception:
        pass

def log_error() -> None:
    # 안전 기본값: GPU에서도 fused SDPA/그래프를 꺼서 CPU와 비슷한 math 경로로
    os.environ.setdefault("PYTORCH_SDP_DISABLE_FLASH_ATTENTION", "1")
    os.environ.setdefault("PYTORCH_SDP_DISABLE_MEM_EFFICIENT",   "1")
    os.environ.setdefault("PYTORCH_SDP_DISABLE_FAST_PATH",       "1")
    os.environ.setdefault("NVTE_FRAMEWORK", "pytorch")  # TE가 있더라도 JAX 경로 차단
    os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")

    # 1) 워커별 시그널 덤프 파일 등록
    log_dir = os.environ.get("STF_ELASTIC_LOGDIR", "/tmp")
    try:
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    except Exception:
        rank = 0
    sig_path = os.path.join(log_dir, f"signal_rank{rank}.log")
    try:
        fh = open(sig_path, "a", buffering=1)  # append
        faulthandler.enable(all_threads=True, file=fh)
        for sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS):
            with contextlib.suppress(Exception):
                faulthandler.register(sig, file=fh, all_threads=True, chain=True)
        print(f"[worker dbg] signal dump -> {sig_path}", flush=True)
    except Exception:
        pass

    # 2) SDPA 호출 모양을 로그(옵션: STF_SDPA_HOOK=0이면 비활)
    if os.environ.get("STF_SDPA_HOOK", "1") == "1":
        if not hasattr(F, "_orig_sdpa"):
            F._orig_sdpa = F.scaled_dot_product_attention
            def _dbg_sdpa(q,k,v, attn_mask=None, dropout_p=0.0, is_causal=False):
                try:
                    qm, km, vm = tuple(q.shape), tuple(k.shape), tuple(v.shape)
                    mm = None if attn_mask is None else tuple(attn_mask.shape)
                    print(f"[SDPA] dev={q.device} q={qm} k={km} v={vm} mask={mm} causal={is_causal}",
                          flush=True)
                except Exception:
                    pass
                return F._orig_sdpa(q,k,v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
            F.scaled_dot_product_attention = _dbg_sdpa
    
    install_compile_monitor()
