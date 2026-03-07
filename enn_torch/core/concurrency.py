# -*- coding: utf-8 -*-
from __future__ import annotations

import collections
import concurrent.futures as futures
import contextlib
import ctypes
import hashlib
import importlib
import itertools
import logging
import math
import os
import queue
import re
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from types import ModuleType, TracebackType
from typing import Any, Callable, Optional, Protocol, Self, Tuple

import torch
from ..core.datatypes import (
    env_first,
    env_first_float,
    env_first_int,
    env_flag,
    get_meta_path,
    save_temp,
    write_json,
)
from .system import (
    CPU,
    _default_thread_limit,
    _optimal_local_worlds,
    _optimal_threads,
    accelerator_stream,
    accelerator_type,
    is_pin_supported,
    sync_accelerator,
)
_ENV_INNER_BOOL_VARS: dict[str, str] = {
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
    "KMP_BLOCKTIME": "0",
    "OMP_PROC_BIND": "TRUE",
}
_ENV_INNER_THREAD_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_EXECUTOR_ORDINAL = itertools.count(0)
_EXECUTOR_ORDINAL_LOCK = threading.Lock()
_TLB_SINGLETON: Optional["Thread"] = None
_TLB_SINGLETON_LOCK = threading.Lock()
_TORCH_THREAD_CFG_LOCK = threading.Lock()
_LAST_TORCH_NUM_THREADS: Optional[int] = None
_LAST_TORCH_INTEROP_THREADS: Optional[int] = None


def _flatten_args(items: Sequence[Any]) -> Iterator[Any]:
    stack: list[Iterator[Any]] = [iter(items)]
    while stack:
        try:
            item = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue

        match item:
            case dict():
                stack.append(iter(item.values()))
            case list() | tuple() | set():
                stack.append(iter(item))
            case _:
                yield item


def _get_throttle_state() -> str:
    state = (
        str(
            env_first(
                (
                    "ENN_CACHE_BACKPRESSURE_MODE",
                    "ENN_CACHE_BACKPRESSURE",
                    "ENN_CACHE_MODE",
                ),
                default="block",
            )
            or "block"
        )
        .strip()
        .lower()
    )
    match state:
        case "sync" | "synchronous":
            return "sync"
        case "raise" | "error":
            return "raise"
        case _:
            return "block"


def _get_throttle_timeout() -> float:
    return max(
        0.0,
        float(
            env_first_float(
                (
                    "ENN_CACHE_BACKPRESSURE_TIMEOUT_S",
                    "ENN_CACHE_SUBMIT_TIMEOUT_S",
                ),
                default=0.05,
            )
            or 0.05
        ),
    )


def _is_early_release_enabled() -> bool:
    return bool(
        env_flag(
            "ENN_CACHE_EARLY_RELEASE",
            "ENN_CACHE_RELEASE_EARLY",
            default=True,
        )
    )


def _is_force_unpin_enabled() -> bool:
    return bool(
        env_flag("ENN_CACHE_FORCE_UNPIN", "ENN_CACHE_UNPIN", default=False)
    )


def _prod_int(shape: Sequence[int]) -> int:
    return int(max(1, math.prod(int(s) for s in shape)))


def _is_affinity_enabled() -> bool:
    return bool(
        env_flag("ENN_EXECUTOR_AFFINITY", "ENN_AFFINITY", default=True)
    )


def _is_affinity_strict() -> bool:
    return bool(
        env_flag(
            "ENN_EXECUTOR_AFFINITY_STRICT",
            "ENN_AFFINITY_STRICT",
            default=False,
        )
    )


def _next_ordinal() -> int:
    with _EXECUTOR_ORDINAL_LOCK:
        with contextlib.suppress(Exception):
            return int(next(_EXECUTOR_ORDINAL))
    return 0


def _is_inner_thread_limited(wl: str) -> bool:
    wl = str(wl or "").strip().lower()
    if wl not in {"cpu", "compute"}:
        return False
    if not bool(env_flag("ENN_EXECUTOR_LIMIT_INNER_THREADS", default=True)):
        return False
    if not bool(
        env_flag("ENN_EXECUTOR_TORCH_OUTER_PARALLELISM", default=True)
    ):
        return False
    return True


def _set_concurrency_env(
    key: str, value: str, *args: Any, force: bool, cap_down: bool = True
) -> None:
    try:
        k = str(key)
        v = str(value)
        if force:
            os.environ[k] = v
            return
        cur = os.environ.get(k, None)
        if cur is None or str(cur).strip() == "":
            os.environ[k] = v
            return
        if not cap_down:
            return
        try:
            cur_i = int(str(cur).strip())
            tgt_i = int(str(v).strip())
        except Exception:
            return
        if cur_i > tgt_i:
            os.environ[k] = str(max(1, tgt_i))
    except Exception:
        return


def _init_env(key: str, value: str, *args: Any, force: bool) -> None:
    try:
        if (
            force
            or (key not in os.environ)
            or (str(os.environ.get(key, "")).strip() == "")
        ):
            os.environ[str(key)] = str(value)
    except Exception:
        return


def _limit_inner_threads(threads: int, *args: Any, force: bool = False) -> int:
    global _LAST_TORCH_NUM_THREADS, _LAST_TORCH_INTEROP_THREADS
    t = max(1, int(threads))
    ov = env_first_int(("ENN_EXECUTOR_INNER_THREADS",), default=0) or 0
    if int(ov) > 0:
        t = max(1, int(ov))
    force = bool(force) or bool(
        env_flag("ENN_EXECUTOR_FORCE_INNER_THREADS", default=False)
    )
    cap_down = bool(
        env_flag("ENN_EXECUTOR_CAP_DOWN_INNER_THREADS", default=True)
    )
    for k in _ENV_INNER_THREAD_VARS:
        _set_concurrency_env(k, str(t), force=force, cap_down=cap_down)
    for k, v in _ENV_INNER_BOOL_VARS.items():
        _init_env(k, v, force=force)
    interop = env_first_int(("ENN_EXECUTOR_INTEROP_THREADS",), default=1) or 1
    interop = max(1, int(interop))
    with _TORCH_THREAD_CFG_LOCK:
        if force or (_LAST_TORCH_NUM_THREADS is None) or int(_LAST_TORCH_NUM_THREADS) != int(t):
            with contextlib.suppress(Exception):
                torch.set_num_threads(int(t))
            _LAST_TORCH_NUM_THREADS = int(t)
        if force or (_LAST_TORCH_INTEROP_THREADS is None) or int(_LAST_TORCH_INTEROP_THREADS) != int(interop):
            with contextlib.suppress(Exception):
                torch.set_num_interop_threads(int(interop))
            _LAST_TORCH_INTEROP_THREADS = int(interop)
    return int(t)


def _is_outer_concurrency_limited() -> bool:
    return bool(env_flag("ENN_EXECUTOR_LIMIT_OUTER_CONCURRENCY", default=True))


def _max_outer_concurrency() -> int:
    return int(
        env_first_int(
            ("ENN_EXECUTOR_OUTER_CONCURRENCY", "ENN_EXECUTOR_OUTER_LIMIT"),
            default=0,
        )
        or 0
    )


def _outer_concurrency_mode() -> str:
    s = (
        str(
            env_first(("ENN_EXECUTOR_OUTER_CONCURRENCY_MODE",), default="auto")
            or "auto"
        )
        .strip()
        .lower()
    )
    return s if s in {"auto", "logical", "physical"} else "auto"


def _are_processes_limited() -> bool:
    return bool(env_flag("ENN_EXECUTOR_CAP_PROCESS_WORKERS", default=True))


def _target_process_workers() -> int:
    return int(
        env_first_int(("ENN_EXECUTOR_PROCESS_TARGET_WORKERS",), default=0) or 0
    )


def _outer_concurrency_limit(
    wl: str, executor_kind: str, nlogical: int, nphysical: int, mw: int
) -> int:
    wl = str(wl or "").strip().lower()
    if wl not in {"cpu", "compute"}:
        return 0
    if str(executor_kind) not in {"thread", "interpreter"}:
        return 0
    if not _is_outer_concurrency_limited():
        return 0
    ov = int(_max_outer_concurrency())
    if ov < 0:
        return 0
    if ov > 0:
        return max(1, min(int(ov), int(mw)))
    mode = _outer_concurrency_mode()
    nlog = max(1, int(nlogical))
    nphy = max(1, int(nphysical))
    if mode == "logical":
        base = nlog
    elif mode == "physical":
        base = nphy
    else:
        base = nphy if wl == "compute" else nlog
    return max(1, min(int(mw), int(base)))


def _parse_cpu_list(spec: str) -> list[int]:
    out: list[int] = []
    s = str(spec or "").strip()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            with contextlib.suppress(Exception):
                lo = int(a.strip())
                hi = int(b.strip())
                if hi < lo:
                    lo, hi = hi, lo
                out.extend(range(lo, hi + 1))
        else:
            with contextlib.suppress(Exception):
                out.append(int(part))
    return out


def _linux_thread_sibling_groups(
    cpus_key: tuple[int, ...],
) -> list[tuple[int, ...]]:
    if not sys.platform.startswith("linux"):
        return []
    groups: dict[tuple[int, ...], None] = {}
    for c in cpus_key:
        p = f"/sys/devices/system/cpu/cpu{int(c)}/topology/thread_siblings_list"
        try:
            with open(p, "r", encoding="utf-8") as f:
                raw = f.read().strip()
        except Exception:
            continue
        sibs = sorted({int(x) for x in _parse_cpu_list(raw)})
        if not sibs:
            continue
        groups[tuple(sibs)] = None
    return sorted(groups.keys(), key=lambda g: (g[0], len(g), g))


def _executor_scatter_cpus(cpus: Sequence[int]) -> list[int]:
    base = sorted({int(x) for x in (cpus or [])})
    if not base:
        return []
    groups = _linux_thread_sibling_groups(tuple(base))
    if not groups:
        return base
    primary = [int(min(g)) for g in groups]
    secondary: list[int] = []
    for g in groups:
        for x in g:
            if int(x) not in primary:
                secondary.append(int(x))
    allowed = set(base)
    out = [c for c in primary if c in allowed] + [
        c for c in secondary if c in allowed
    ]
    return out if out else base


def _executor_prefer_smt_lane(
    cpus: Sequence[int], *args: Any, prefer_primary: bool
) -> list[int]:
    base = sorted({int(x) for x in (cpus or [])})
    if not base:
        return []
    groups = _linux_thread_sibling_groups(tuple(base))
    if not groups:
        return base
    primary = [int(min(g)) for g in groups]
    secondary: list[int] = []
    for g in groups:
        for x in g:
            if int(x) not in primary:
                secondary.append(int(x))
    allowed = set(base)
    prim = [c for c in primary if c in allowed]
    sec = [c for c in secondary if c in allowed]
    out = (prim + sec) if prefer_primary else (sec + prim)
    return out if out else base


def _executor_allowed_cpus() -> list[int]:
    with contextlib.suppress(Exception):
        return sorted({int(x) for x in CPU.allowed()})
    with contextlib.suppress(Exception):
        return sorted({int(x) for x in os.sched_getaffinity(0)})
    return []


def _hash32(s: str) -> int:
    with contextlib.suppress(Exception):
        b = hashlib.md5(s.encode("utf-8", errors="ignore")).digest()
        return int.from_bytes(b[:4], byteorder="little", signed=False)
    return 0


def _pick_coprime_stride(n: int, hint: int, salt: int) -> int:
    n = max(1, int(n))
    if n == 1:
        return 1
    cand = max(1, int(hint)) + max(0, int(salt))
    if n % 2 == 0 and cand % 2 == 0:
        cand += 1
    for k in range(cand, cand + n + 2):
        if math.gcd(int(k), int(n)) == 1:
            return int(k)
    return 1


def _executor_scope_start(
    role: str, wl: str, prefix: str, mw: int, cpw: int, ordinal: int, ncpu: int
) -> int:
    seed = env_first(
        ("ENN_EXECUTOR_AFFINITY_SEED", "ENN_AFFINITY_SEED"), default="0"
    )
    key = f"{seed}|pid={os.getpid()}|role={role}|wl={wl}|pfx={prefix}|mw={mw}|cpw={cpw}|ord={ordinal}"
    h = _hash32(key)
    return int(h % max(1, int(ncpu)))


def _pick_cores_balanced(
    cpus: Sequence[int],
    start: int,
    idx: int,
    mw: int,
    cpw: int,
    salt: int,
) -> list[int]:
    base = list(cpus or [])
    n = len(base)
    if n <= 0:
        return []
    i = int(idx) % max(1, int(mw))
    cpw_i = max(1, min(int(cpw), n))
    if cpw_i == 1:
        return [int(base[(int(start) + i) % n])]
    stride1 = _pick_coprime_stride(
        n, hint=max(1, n // max(1, int(mw))), salt=salt
    )
    base_pos = (int(start) + i * stride1) % n
    stride2 = _pick_coprime_stride(n, hint=max(1, n // cpw_i), salt=salt + 7)
    out: list[int] = []
    seen: set[int] = set()
    for j in range(cpw_i):
        c = int(base[(base_pos + j * stride2) % n])
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out or [int(base[base_pos])]


def _thread_worker_index(max_workers: int) -> int:
    try:
        nm = str(threading.current_thread().name or "")
    except Exception:
        nm = ""
    m = re.search(r"(?:_|-)(\d+)$", nm)
    if m:
        with contextlib.suppress(Exception):
            return int(m.group(1)) % max(1, int(max_workers))
    with contextlib.suppress(Exception):
        nid = int(threading.get_native_id())
        return int(nid) % max(1, int(max_workers))
    return 0


def _process_worker_index(max_workers: int) -> int:
    with contextlib.suppress(Exception):
        import multiprocessing as _mp

        ident = getattr(_mp.current_process(), "_identity", None)
        if ident and len(ident) >= 1:
            return max(0, int(ident[0]) - 1) % max(1, int(max_workers))
    with contextlib.suppress(Exception):
        return int(os.getpid()) % max(1, int(max_workers))
    return 0


def _set_current_thread_affinity(cores: Sequence[int]) -> None:
    if not cores:
        return
    if sys.platform.startswith("linux"):
        with contextlib.suppress(Exception):
            tid = int(threading.get_native_id())
            if tid > 0:
                os.sched_setaffinity(tid, {int(c) for c in cores})


def _set_current_process_affinity(cores: Sequence[int]) -> None:
    if not cores:
        return
    if sys.platform.startswith("linux"):
        with contextlib.suppress(Exception):
            os.sched_setaffinity(0, {int(c) for c in cores})


def _executor_thread_initializer(
    role: str,
    wl: str,
    cpus: Sequence[int],
    start: int,
    mw: int,
    cpw: int,
    salt: int,
) -> None:
    idx = _thread_worker_index(max_workers=int(mw))
    cores = _pick_cores_balanced(
        cpus,
        start=int(start),
        idx=int(idx),
        mw=int(mw),
        cpw=int(cpw),
        salt=int(salt),
    )
    if cores:
        _set_current_thread_affinity(cores)
    with contextlib.suppress(Exception):
        tlb = new_affinity()
        tlb._tls.pinned = True


def _executor_process_initializer(
    wl: str,
    cpus: Sequence[int],
    start: int,
    mw: int,
    cpw: int,
    salt: int,
) -> None:
    idx = _process_worker_index(max_workers=int(mw))
    cores = _pick_cores_balanced(
        cpus,
        start=int(start),
        idx=int(idx),
        mw=int(mw),
        cpw=int(cpw),
        salt=int(salt),
    )
    if cores:
        _set_current_process_affinity(cores)
    if cores and _is_inner_thread_limited(wl):
        _limit_inner_threads(max(1, int(len(cores))))
    with contextlib.suppress(Exception):
        tlb = new_affinity()
        tlb._tls.pinned = True


def is_free_threading_build() -> bool:
    with contextlib.suppress(Exception):
        import sysconfig
        v = sysconfig.get_config_var("Py_GIL_DISABLED")
        if v is not None:
            return bool(int(v))
    with contextlib.suppress(Exception):
        return bool(CPU.is_free_threaded_build())
    tag = getattr(getattr(sys, "implementation", None), "cache_tag", "") or ""
    return bool(isinstance(tag, str) and tag.endswith("t"))


def is_gil_enabled() -> bool:
    with contextlib.suppress(Exception):
        return bool(CPU.is_gil_enabled())
    fn = getattr(sys, "_is_gil_enabled", None)
    if callable(fn):
        with contextlib.suppress(Exception):
            return bool(fn())
    return True


def python_build_tag() -> str:
    major, minor = sys.version_info[:2]
    return f"{major}.{minor}{'t' if is_free_threading_build() else ''}"


def is_interpreter_pool_supported() -> bool:
    return getattr(futures, "InterpreterPoolExecutor", None) is not None


def new_prefetcher(
    iterable: Any,
    *args: Any,
    max_batches: int = 4,
    name: str = "buffer",
    daemon: bool = True,
) -> Prefetcher:
    return Prefetcher(
        iterable, max_batches=max_batches, name=name, daemon=daemon
    )


def new_executor(
    max_workers: int,
    *args: Any,
    workload: str = "io",
    name: str = "enn_torch",
    prefer_interpreters: bool | None = None,
) -> futures.Executor:
    mw = max(1, int(max_workers))
    wl = str(workload or "io").strip().lower()
    prefix = str(name or "enn_torch").strip() or "enn_torch"
    executor_kind = "thread"
    if wl in {"cpu", "compute"}:
        if not is_gil_enabled():
            executor_kind = "thread"
        else:
            if prefer_interpreters is None:
                prefer_interpreters = bool(
                    env_flag("ENN_PREFER_INTERPRETER_POOL", default=True)
                )
            if bool(prefer_interpreters) and is_interpreter_pool_supported():
                executor_kind = "interpreter"
            else:
                executor_kind = "process"
    if executor_kind == "process":
        thread_prefix = f"{prefix}-proc"
    elif executor_kind == "interpreter":
        thread_prefix = f"{prefix}-interp"
    else:
        thread_prefix = (
            f"{prefix}-thr" if wl in {"cpu", "compute"} else f"{prefix}-io"
        )
    init_fn: Callable[..., Any] | None = None
    init_args: tuple[Any, ...] = ()
    nlogical_for_limit = 0
    nphysical_for_limit = 0
    mw_eff = int(mw)
    if _is_affinity_enabled():
        try:
            allowed = _executor_allowed_cpus()
            cpus_scatter = _executor_scatter_cpus(allowed)
            if cpus_scatter:
                groups = _linux_thread_sibling_groups(tuple(cpus_scatter))
                nphysical_for_limit = int(
                    len(groups) if groups else len(cpus_scatter)
                )
                ordinal = _next_ordinal()
                if executor_kind == "process" and wl in {"cpu", "compute"}:
                    mw_eff = int(mw)
                    if _are_processes_limited():
                        tgt = int(_target_process_workers())
                        if tgt > 0:
                            mw_eff = min(int(mw_eff), max(1, int(tgt)))
                        else:
                            mw_eff = min(
                                int(mw_eff),
                                max(
                                    1,
                                    int(
                                        nphysical_for_limit
                                        or len(cpus_scatter)
                                    ),
                                ),
                            )
                    mw_eff = max(1, int(mw_eff))
                if executor_kind == "process":
                    default_cpw = max(
                        1, int(len(cpus_scatter) // max(1, mw_eff))
                    )
                    cpw = int(
                        env_first_int(
                            (
                                "ENN_EXECUTOR_CORES_PER_WORKER_CPU",
                                "ENN_AFFINITY_CORES_PER_WORKER_CPU",
                            ),
                            default=default_cpw,
                        )
                        or default_cpw
                    )
                    cpw = max(1, min(int(cpw), max(1, len(cpus_scatter))))
                    start = _executor_scope_start(
                        role="process",
                        wl=wl,
                        prefix=thread_prefix,
                        mw=int(mw_eff),
                        cpw=cpw,
                        ordinal=ordinal,
                        ncpu=len(cpus_scatter),
                    )
                    init_fn = _executor_process_initializer
                    init_args = (
                        wl,
                        cpus_scatter,
                        int(start),
                        int(mw_eff),
                        int(cpw),
                        11,
                    )
                elif executor_kind == "interpreter":
                    cpus_pref = _executor_prefer_smt_lane(
                        cpus_scatter, prefer_primary=True
                    )
                    nlogical_for_limit = int(len(cpus_pref))
                    start = _executor_scope_start(
                        role="interpreter",
                        wl=wl,
                        prefix=thread_prefix,
                        mw=int(mw_eff),
                        cpw=1,
                        ordinal=ordinal,
                        ncpu=max(1, len(cpus_pref)),
                    )
                    init_fn = _executor_thread_initializer
                    init_args = (
                        "interpreter",
                        wl,
                        cpus_pref,
                        int(start),
                        int(mw),
                        1,
                        23,
                    )
                else:
                    cpus_pref = _executor_prefer_smt_lane(
                        cpus_scatter, prefer_primary=False
                    )
                    nlogical_for_limit = int(len(cpus_pref))
                    start = _executor_scope_start(
                        role="thread",
                        wl=wl,
                        prefix=thread_prefix,
                        mw=int(mw_eff),
                        cpw=1,
                        ordinal=ordinal,
                        ncpu=max(1, len(cpus_pref)),
                    )
                    init_fn = _executor_thread_initializer
                    init_args = (
                        "thread",
                        wl,
                        cpus_pref,
                        int(start),
                        int(mw),
                        1,
                        37,
                    )
        except Exception:
            if _is_affinity_strict():
                raise
            init_fn = None
            init_args = ()
            nlogical_for_limit = 0
            nphysical_for_limit = 0
            mw_eff = int(mw)
    if executor_kind in {"thread", "interpreter"} and _is_inner_thread_limited(
        wl
    ):
        _limit_inner_threads(1)
    if nlogical_for_limit <= 0:
        with contextlib.suppress(Exception):
            nlogical_for_limit = int(len(_executor_allowed_cpus()))
    if nphysical_for_limit <= 0:
        nphysical_for_limit = int(nlogical_for_limit or mw)
    outer_limit = _outer_concurrency_limit(
        wl,
        executor_kind,
        int(nlogical_for_limit or mw),
        int(nphysical_for_limit or (nlogical_for_limit or mw)),
        int(mw),
    )
    mw_outer = int(mw)
    if (
        executor_kind in {"thread", "interpreter"}
        and outer_limit
        and int(outer_limit) < int(mw_outer)
    ):
        mw_outer = int(outer_limit)
        if (
            init_fn is _executor_thread_initializer
            and init_args
            and len(init_args) >= 5
        ):
            init_args = (*init_args[:4], int(mw_outer), *init_args[5:])
    if executor_kind == "thread":
        ex = futures.ThreadPoolExecutor(
            max_workers=mw_outer,
            thread_name_prefix=thread_prefix,
            initializer=init_fn,
            initargs=init_args,
        )
        return (
            BoundedExecutor(ex, outer_limit)
            if (outer_limit and outer_limit < mw)
            else ex
        )
    if executor_kind == "interpreter":
        cls = getattr(futures, "InterpreterPoolExecutor", None)
        if cls is None:
            ex = futures.ThreadPoolExecutor(
                max_workers=mw_outer,
                thread_name_prefix=thread_prefix,
                initializer=init_fn,
                initargs=init_args,
            )
            return (
                BoundedExecutor(ex, outer_limit)
                if (outer_limit and outer_limit < mw)
                else ex
            )
        try:
            ex = cls(
                max_workers=mw_outer,
                thread_name_prefix=thread_prefix,
                initializer=init_fn,
                initargs=init_args,
            )
        except TypeError:
            ex = cls(max_workers=mw_outer, thread_name_prefix=thread_prefix)
        return (
            BoundedExecutor(ex, outer_limit)
            if (outer_limit and outer_limit < mw)
            else ex
        )
    try:
        return futures.ProcessPoolExecutor(
            max_workers=int(mw_eff),
            initializer=init_fn,
            initargs=init_args,
        )
    except TypeError:
        return futures.ProcessPoolExecutor(max_workers=int(mw_eff))


def new_affinity(io_workers: Optional[int] = None) -> "Thread":
    global _TLB_SINGLETON
    if _TLB_SINGLETON is None:
        with _TLB_SINGLETON_LOCK:
            if _TLB_SINGLETON is None:
                default_workers = (
                    int(io_workers)
                    if io_workers is not None
                    else max(1, int(CPU.count()) // 2)
                )
                _TLB_SINGLETON = Thread(io_workers=int(default_workers))
    elif io_workers is not None:
        _TLB_SINGLETON.tune(io_workers=int(io_workers))
    return _TLB_SINGLETON


def new_thread(
    fn: Callable[[Any], Any], *args: Any, io_workers: Optional[int] = None
) -> Callable[[Any], Any]:
    return new_affinity(io_workers=io_workers).new_thread(fn)


def close(obj: Any, *args: Any, join_timeout: float | None = 1.0) -> None:
    for name in (
        "cleanup",
        "close",
        "shutdown",
        "stop",
        "terminate",
        "disconnect",
        "release",
        "join",
    ):
        if callable(fn := getattr(obj, name, None)):
            with contextlib.suppress(Exception):
                match name:
                    case "join" if join_timeout is not None:
                        fn(timeout=float(join_timeout))
                    case _:
                        fn()
            return
    with contextlib.suppress(Exception):
        obj() if callable(obj) else None


class _QueryEvent(Protocol):
    def query(self: Self) -> bool: ...


class _SyncEvent(Protocol):
    def synchronize(self: Self) -> Any: ...


class _WaitEvent(Protocol):
    def wait(self: Self, timeout: float | None = None) -> Any: ...


@dataclass(slots=True)
class _PoolToken:
    i: int
    g: int


@dataclass(slots=True)
class _PoolEntry:
    page: TensorPage
    busy: bool = False
    fence: object | None = None
    fence_evt: object | None = None
    gen: int = 0


@dataclass(slots=True)
class ProducerError:
    exc: BaseException
    tb: str = ""


class Affinity:
    __slots__ = (
        "_parent",
        "_fn",
        "_pin_thread",
        "_tls",
        "_lock",
        "_tune",
        "_sample_every",
        "_flush_every",
        "_perf_counter_ns",
        "_thread_time_ns",
    )

    def __init__(
        self: Self,
        parent: "Thread",
        fn: Callable[[Any], Any],
        pin_thread: Callable[[], None],
        tls: Any,
        lock: Any,
        tune: Callable[..., None],
        sample_every: int,
        flush_every: int,
        perf_counter_ns: Callable[[], int],
        thread_time_ns: Optional[Callable[[], int]],
    ) -> None:
        self._parent = parent
        self._fn = fn
        self._pin_thread = pin_thread
        self._tls = tls
        self._lock = lock
        self._tune = tune
        self._sample_every = sample_every
        self._flush_every = flush_every
        self._perf_counter_ns = perf_counter_ns
        self._thread_time_ns = thread_time_ns

    def __call__(self: Self, *args: Any, **kwargs: Any) -> Any:
        if not getattr(self._tls, "pinned", False):
            self._pin_thread()
        count = getattr(self._tls, "count", 0) + 1
        self._tls.count = count
        do_sample = count % self._sample_every == 0
        t0 = self._perf_counter_ns() if do_sample else 0
        c0 = (
            self._thread_time_ns()
            if do_sample and callable(self._thread_time_ns)
            else 0
        )
        out = self._fn(*args, **kwargs)
        if do_sample:
            t1 = self._perf_counter_ns()
            c1 = (
                self._thread_time_ns()
                if callable(self._thread_time_ns)
                else None
            )
            try:
                with self._lock:
                    self._parent._total_time += max(0, int(t1) - int(t0))
                    if c1 is not None:
                        self._parent._total_cpu += max(0, int(c1) - int(c0))
            except Exception:
                pass
        if count % self._flush_every == 0:
            try:
                self._tune(initial=False)
            except Exception:
                pass
        return out


class Disposable:
    def __init__(self: Self, *args: Any, **kwargs: Any) -> None:
        self._keep: list[Any] = list(_flatten_args(list(args)))
        if kwargs:
            self._keep.extend(list(_flatten_args(list(kwargs.values()))))

    def add(self: Self, *args: Any, **kwargs: Any) -> None:
        self._keep.extend(list(_flatten_args(list(args))))
        if kwargs:
            self._keep.extend(list(_flatten_args(list(kwargs.values()))))

    def cleanup(self: Self) -> None:
        for obj in self._keep:
            close(obj)

    def close(self: Self) -> None:
        self.cleanup()

    def __iter__(self: Self) -> Iterator[Any]:
        return iter(self._keep)


class Prefetcher:
    __slots__ = (
        "_src",
        "_max_batches",
        "_name",
        "_daemon",
        "_session",
        "_join_timeout_s",
    )

    def __init__(
        self: Self,
        iterable: Any,
        *args: Any,
        max_batches: int = 4,
        name: str = "buffer",
        daemon: bool = True,
        _session: bool = False,
    ) -> None:
        self._src = iterable
        self._max_batches = int(max_batches)
        self._name = str(name or "buffer")
        self._daemon = bool(daemon)
        self._session = bool(_session)
        self._join_timeout_s = 0.5
        with contextlib.suppress(Exception):
            jt_ms = int(
                env_first_int(("ENN_THREAD_JOIN_TIMEOUT_MS",), default=500)
                or 500
            )
            self._join_timeout_s = max(0.0, float(jt_ms) / 1000.0)

    @property
    def max_batches(self: Self) -> int:
        return int(self._max_batches)

    def __len__(self: Self) -> int:
        try:
            return int(len(self._src))
        except Exception:
            return 1

    def __iter__(self: Self) -> Iterator[Any]:
        if not bool(self._session):
            return iter(
                Prefetcher(
                    self._src,
                    max_batches=int(self._max_batches),
                    name=self._name,
                    daemon=self._daemon,
                    _session=True,
                )
            )
        return self._iter_session()

    def __getattr__(self: Self, name: str) -> Any:
        return getattr(self._src, name)

    def _producer_loop(
        self: Self, src: Any, buf: "BufferQueue", sentinel: object
    ) -> None:
        it: Iterator[Any] | None = None
        try:
            it = iter(src)
            while not buf.is_stopped():
                if not buf.block(timeout=None):
                    break
                try:
                    item = next(it)
                except StopIteration:
                    break
                if buf.is_stopped():
                    break
                if not buf.put(item, timeout=0.0):
                    break
        except BaseException as exc:
            with contextlib.suppress(Exception):
                buf.put(ProducerError(exc=exc, tb=traceback.format_exc()))
        finally:
            with contextlib.suppress(Exception):
                if it is not None:
                    close(it)
            with contextlib.suppress(Exception):
                buf.put(sentinel)

    def _iter_session(self: Self) -> Iterator[Any]:
        buf = BufferQueue(max_batches=int(self._max_batches))
        sentinel = object()
        ex = new_executor(1, workload="io", name=f"{self._name}-producer")
        fut = ex.submit(self._producer_loop, self._src, buf, sentinel)
        try:
            while True:
                try:
                    item = buf.get(timeout=None)
                except queue.Empty:
                    break
                if item is sentinel:
                    break
                if isinstance(item, ProducerError):
                    if isinstance(item.exc, (KeyboardInterrupt, SystemExit)):
                        raise item.exc
                    raise RuntimeError(
                        f"{self._name} producer crashed: {item.exc}\n{item.tb}"
                    ) from item.exc
                yield item
        finally:
            buf.stop()
            with contextlib.suppress(Exception):
                buf.clear()
            with contextlib.suppress(Exception):
                try:
                    fut.result(timeout=float(self._join_timeout_s))
                except futures.TimeoutError:
                    pass
            with contextlib.suppress(Exception):
                try:
                    ex.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    ex.shutdown(wait=False)


class TensorPage:
    __slots__ = ("_buf", "_numel", "_dtype", "_pinned")

    def __init__(
        self: Self,
        numel: int,
        dtype: torch.dtype,
        *args: Any,
        pin_memory: bool = True,
    ) -> None:
        self._numel = int(max(1, int(numel)))
        self._dtype = dtype
        self._pinned = bool(pin_memory)
        self._buf = torch.empty(
            self._numel,
            dtype=self._dtype,
            device="cpu",
            pin_memory=bool(self._pinned),
        )

    @property
    def numel(self: Self) -> int:
        return int(self._numel)

    @property
    def dtype(self: Self) -> torch.dtype:
        return self._dtype

    @property
    def pinned(self: Self) -> bool:
        return bool(self._pinned)

    def ensure(self: Self, numel: int) -> None:
        need = int(max(1, int(numel)))
        if need <= int(self._numel):
            return
        self._numel = need
        self._buf = torch.empty(
            self._numel,
            dtype=self._dtype,
            device="cpu",
            pin_memory=bool(self._pinned),
        )

    def view(self: Self, *shape: int) -> torch.Tensor:
        need = _prod_int(shape)
        self.ensure(need)
        return self._buf[:need].view(*shape)


class TensorPagePool:
    Token = _PoolToken
    _Entry = _PoolEntry

    def __init__(
        self: Self, capacity: int = 4, *args: Any, pin_memory: bool = True
    ) -> None:
        self._cap = max(1, int(capacity))
        self._pin = bool(pin_memory)
        self._pages: list[TensorPagePool._Entry] = []
        self._rr = 0
        self._cv = threading.Condition()

    def _event_finished(self: Self, evt: object | None) -> bool:
        if evt is None:
            return True
        with contextlib.suppress(Exception):
            if isinstance(evt, _QueryEvent):
                return bool(evt.query())
            elif callable(is_set := getattr(evt, "is_set", None)):
                return bool(is_set())
        return False

    def _scavenge_lock(self: Self) -> int:
        freed = 0
        for e in self._pages:
            if (
                e.busy
                and e.fence is not None
                and self._event_finished(e.fence)
            ):
                e.busy = False
                e.fence = None
                freed += 1
        return freed

    @property
    def capacity(self: Self) -> int:
        return int(self._cap)

    def ensure_capacity(self: Self, capacity: int) -> int:
        want = max(1, int(capacity))
        with self._cv:
            if want > self._cap:
                self._cap = int(want)
                self._cv.notify_all()
            return int(self._cap)

    def get(
        self: Self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        *args: Any,
        return_handle: bool = False,
        block: bool = False,
        timeout: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, TensorPagePool.Token | None]:
        shape_t = tuple(int(s) for s in shape)
        need = _prod_int(shape_t)
        deadline = (
            (time.monotonic() + float(timeout))
            if timeout is not None
            else None
        )
        check_interval = 0.01
        while True:
            idx, need_new, grow = None, False, False
            with self._cv:
                self._scavenge_lock()
                n = len(self._pages)
                if n:
                    for k in range(n):
                        j = (self._rr + k) % n
                        e = self._pages[j]
                        if not e.busy:
                            e.busy, e.fence, idx, self._rr = (
                                True,
                                None,
                                j,
                                (j + 1) % max(1, n),
                            )
                            need_new = (
                                (e.page.dtype != dtype)
                                or (e.page.numel < need)
                                or (e.page.pinned != self._pin)
                            )
                            break
                if idx is None:
                    if n < self._cap:
                        grow = True
                    elif block:
                        if (
                            deadline
                            and (wait := deadline - time.monotonic()) <= 0
                        ):
                            break
                        self._cv.wait(
                            timeout=(
                                min(check_interval, wait)
                                if deadline
                                else check_interval
                            )
                        )
                        continue
                    else:
                        break
            if idx is not None:
                new_page = (
                    TensorPage(numel=need, dtype=dtype, pin_memory=self._pin)
                    if need_new
                    else None
                )
                with self._cv:
                    e = self._pages[idx]
                    if new_page:
                        e.page, e.gen = new_page, e.gen + 1
                    view, token = (
                        e.page.view(*shape_t),
                        TensorPagePool.Token(int(idx), int(e.gen)),
                    )
                return (view, token) if return_handle else view
            if grow:
                entry = TensorPagePool._Entry(
                    page=TensorPage(
                        numel=need, dtype=dtype, pin_memory=self._pin
                    ),
                    busy=True,
                    fence=None,
                    gen=0,
                )
                with self._cv:
                    if len(self._pages) < self._cap:
                        self._pages.append(entry)
                        self._rr = len(self._pages) % self._cap
                        view, token = (
                            entry.page.view(*shape_t),
                            TensorPagePool.Token(len(self._pages) - 1, 0),
                        )
                        return (view, token) if return_handle else view
                continue
            break
        view = torch.empty(
            need, dtype=dtype, device="cpu", pin_memory=False
        ).view(*shape_t)
        return (view, None) if return_handle else view

    def get_like(
        self: Self,
        t: torch.Tensor,
        *args: Any,
        return_handle: bool = False,
        block: bool = False,
        timeout: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, TensorPagePool.Token | None]:
        return self.get(
            tuple(int(s) for s in t.shape),
            t.dtype,
            return_handle=return_handle,
            block=block,
            timeout=timeout,
        )

    def fence_event(
        self: Self,
        token: TensorPagePool.Token | None,
        factory: Callable[[], object] | None,
    ) -> object | None:
        if token is None or factory is None:
            return None
        i = int(getattr(token, "i", -1))
        g = int(getattr(token, "g", -1))
        if i < 0:
            return None
        with self._cv:
            if (
                0 <= i < len(self._pages)
                and self._pages[i].gen == g
                and self._pages[i].fence_evt
            ):
                return self._pages[i].fence_evt
        try:
            ev_new = factory()
        except:
            return None
        if not ev_new:
            return None
        with self._cv:
            if 0 <= i < len(self._pages) and self._pages[i].gen == g:
                if not self._pages[i].fence_evt:
                    self._pages[i].fence_evt = ev_new
                return self._pages[i].fence_evt
        return ev_new

    def release_after(
        self: Self,
        token: TensorPagePool.Token | None,
        wait_event: object | None,
    ) -> None:
        if token is None:
            return
        with self._cv:
            i = int(getattr(token, "i", -1))
            g = int(getattr(token, "g", -1))
            if 0 <= i < len(self._pages) and self._pages[i].gen == g:
                self._pages[i].busy, self._pages[i].fence = True, wait_event
                self._cv.notify()

    def release(self: Self, token: TensorPagePool.Token | None) -> None:
        if token is None:
            return
        with self._cv:
            i = int(getattr(token, "i", -1))
            g = int(getattr(token, "g", -1))
            if 0 <= i < len(self._pages) and self._pages[i].gen == g:
                self._pages[i].busy, self._pages[i].fence = False, None
                self._cv.notify()

    def collect(self: Self) -> None:
        with self._cv:
            if self._scavenge_lock():
                self._cv.notify_all()


def pool_tensor(
    tensor: torch.Tensor,
    *args: Any,
    dtype: torch.dtype,
    device: torch.device,
    cpu_pool: "TensorPagePool | None",
    dev_type: str | None = None,
    pinned_ok: bool | None = None,
) -> tuple[torch.Tensor, "TensorPagePool.Token | None", bool]:
    _ = args
    if not torch.is_tensor(tensor):
        raise TypeError(
            f"pool_tensor expects a torch.Tensor, got {type(tensor)}"
        )
    if dev_type is None:
        dev_type = str(getattr(device, "type", "cpu"))
    if pinned_ok is None:
        pinned_ok = bool(is_pin_supported(str(dev_type)))

    if tensor.device.type != "cpu":
        if tensor.dtype != dtype:
            with contextlib.suppress(Exception):
                tensor = tensor.to(dtype=dtype, copy=False)
        return tensor, None, False

    with contextlib.suppress(Exception):
        is_pinned = getattr(tensor, "is_pinned", None)
        if callable(is_pinned) and bool(is_pinned()) and tensor.dtype == dtype:
            return tensor, None, True

    if cpu_pool is not None and bool(pinned_ok):
        pin_wait_ms = max(
            0,
            int(
                env_first_int(
                    ("ENN_RUNTIME_PIN_POOL_WAIT_MS", "ENN_PIN_POOL_WAIT_MS"),
                    default=(2 if CPU.is_optimized_for_no_gil() else 0),
                )
                or 0
            ),
        )
        pin_wait_s = (
            max(0.0, float(pin_wait_ms) / 1000.0)
            if int(pin_wait_ms) > 0
            else None
        )
        buf, token = cpu_pool.get(
            tuple(tensor.shape),
            dtype,
            return_handle=True,
            block=bool(int(pin_wait_ms) > 0),
            timeout=pin_wait_s,
        )
        buf.copy_(tensor, non_blocking=False)
        pinned = False
        with contextlib.suppress(Exception):
            is_pinned = getattr(buf, "is_pinned", None)
            if callable(is_pinned):
                pinned = bool(is_pinned())
        return buf, token, pinned

    out = tensor
    if out.dtype != dtype:
        out = out.to(dtype=dtype, copy=False)
    pinned = False
    with contextlib.suppress(Exception):
        is_pinned = getattr(out, "is_pinned", None)
        if callable(is_pinned):
            pinned = bool(is_pinned())
    return out, None, pinned


def stream_tensor(
    tensor: object,
    *args: Any,
    device: torch.device | str | int,
    cpu_pool: object,
    handle: "TensorPagePool.Token | None" = None,
    pinned: bool | None = None,
    dev_type: object | None = None,
    non_blocking_ok: object | None = None,
    backend: object | None = None,
    stream_fn: object | None = None,
    Event: object | None = None,
    fence_event_factory: object | None = None,
    can_stream_release: object | None = None,
) -> object:
    _ = args
    if not torch.is_tensor(tensor):
        return tensor
    device = (
        torch.device(device)
        if not isinstance(device, torch.device)
        else device
    )
    if dev_type is None:
        dev_type = str(getattr(device, "type", "cpu"))
    if non_blocking_ok is None:
        non_blocking_ok = bool(dev_type in ("cuda", "xpu"))
    pinned_ok = bool(is_pin_supported(str(dev_type)))

    if pinned is None:
        pinned = False
        with contextlib.suppress(Exception):
            is_pinned = getattr(tensor, "is_pinned", None)
            if callable(is_pinned):
                pinned = bool(is_pinned())

    if tensor.device.type != "cpu" or (not bool(non_blocking_ok)):
        out = tensor.to(device, non_blocking=bool(non_blocking_ok))
        if handle is not None and cpu_pool is not None:
            with contextlib.suppress(Exception):
                cpu_pool.release(handle)
        return out

    if handle is None:
        return tensor.to(
            device, non_blocking=bool(non_blocking_ok and pinned and pinned_ok)
        )

    if backend is None:
        backend = accelerator_type(str(dev_type))
    if stream_fn is None and backend is not None:
        stream_fn = getattr(backend, "current_stream", None)
    if Event is None and backend is not None:
        Event = getattr(backend, "Event", None)
    if can_stream_release is None:
        can_stream_release = bool(
            pinned
            and pinned_ok
            and callable(stream_fn)
            and (Event is not None)
        )

    if (not bool(pinned)) or (not bool(can_stream_release)):
        out = tensor.to(device, non_blocking=False)
        if cpu_pool is not None:
            with contextlib.suppress(Exception):
                cpu_pool.release(handle)
        return out

    stream = None
    if callable(stream_fn):
        with contextlib.suppress(Exception):
            try:
                stream = stream_fn(device=device)
            except TypeError:
                try:
                    stream = stream_fn(device)
                except TypeError:
                    stream = stream_fn()

    try:
        if stream is not None:
            with accelerator_stream(stream, str(dev_type)):
                out = tensor.to(device, non_blocking=True)
        else:
            out = tensor.to(device, non_blocking=True)

        if stream is not None:
            rec = getattr(tensor, "record_stream", None)
            if callable(rec):
                with contextlib.suppress(Exception):
                    rec(stream)

        if cpu_pool is not None:
            try:
                evt = None
                fe = getattr(cpu_pool, "fence_event", None)
                if callable(fe) and fence_event_factory is not None:
                    with contextlib.suppress(Exception):
                        evt = fe(handle, fence_event_factory)
                if evt is None:
                    if fence_event_factory is not None:
                        with contextlib.suppress(Exception):
                            evt = fence_event_factory()
                    elif Event is not None:
                        with contextlib.suppress(Exception):
                            evt = Event()
                if evt is not None:
                    if stream is not None:
                        try:
                            evt.record(stream)
                        except TypeError:
                            evt.record()
                    else:
                        evt.record()
                    cpu_pool.release_after(handle, evt)
                else:
                    with contextlib.suppress(Exception):
                        sync_accelerator(device)
                    with contextlib.suppress(Exception):
                        cpu_pool.release(handle)
            except Exception:
                with contextlib.suppress(Exception):
                    sync_accelerator(device)
                with contextlib.suppress(Exception):
                    cpu_pool.release(handle)
        return out
    except Exception:
        out = tensor.to(device, non_blocking=False)
        if cpu_pool is not None:
            with contextlib.suppress(Exception):
                cpu_pool.release(handle)
        return out


def move_staged_pair_to_device(
    X_st: object,
    x_tok: "TensorPagePool.Token | None",
    x_pinned: bool,
    Y_st: object | None,
    y_tok: "TensorPagePool.Token | None",
    y_pinned: bool,
    to_device: Callable[..., object],
) -> tuple[
    object,
    object | None,
    "TensorPagePool.Token | None",
    "TensorPagePool.Token | None",
]:
    X_dev = to_device(X_st, handle=x_tok, pinned=x_pinned)
    x_tok = None
    Y_dev = None
    if Y_st is not None:
        Y_dev = to_device(Y_st, handle=y_tok, pinned=y_pinned)
        y_tok = None
    return X_dev, Y_dev, x_tok, y_tok


class TensorSpooler:
    def __init__(self: Self, root: str, max_queue: int = 8) -> None:
        self._root = os.fspath(root)
        os.makedirs(self._root, exist_ok=True)
        max_q = max(1, int(max_queue))
        self._sem = threading.Semaphore(max_q)
        self._q: "queue.SimpleQueue[tuple[Any, Any, Any, Any]]" = (
            queue.SimpleQueue()
        )
        self._bp_mode = str(_get_throttle_state() or "block")
        self._bp_timeout_s = float(_get_throttle_timeout() or 0.0)
        self._early_release = bool(_is_early_release_enabled())
        self._force_unpin = bool(_is_force_unpin_enabled())
        self._err: BaseException | None = None
        self._err_event = threading.Event()
        self._closed = threading.Event()
        self._executor = new_executor(1, workload="io", name="CacheWriter")
        self._future = self._executor.submit(self._run)

    def _wait(self: Self, evt: object | None) -> None:
        if evt is None:
            return
        with contextlib.suppress(Exception):
            if isinstance(evt, _SyncEvent):
                evt.synchronize()
                return
            if isinstance(evt, _WaitEvent):
                evt.wait()
                return

    @staticmethod
    def _is_cpu_pinned(t: torch.Tensor) -> bool:
        if not torch.is_tensor(t):
            return False
        if getattr(t, "device", None) is None or t.device.type != "cpu":
            return False
        with contextlib.suppress(Exception):
            return bool(t.is_pinned()) if hasattr(t, "is_pinned") else False
        return False

    def _init_tensor(
        self: Self,
        tensor: torch.Tensor,
        *args: Any,
        release_cb: Callable[[], Any] | None = None,
        early_release: bool | None = None,
        force_unpin: bool | None = None,
    ) -> tuple[torch.Tensor, bool]:
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        buf = tensor.detach()
        if hasattr(buf, "to_local"):
            buf = buf.to_local()
        if buf.device.type != "cpu":
            buf = buf.to(device="cpu", non_blocking=False)
        early = (
            early_release if early_release is not None else self._early_release
        )
        unpin = force_unpin if force_unpin is not None else self._force_unpin
        released = False
        if TensorSpooler._is_cpu_pinned(buf) and (
            unpin or (early and callable(release_cb))
        ):
            try:
                tmp = torch.empty_like(buf, device="cpu", pin_memory=False)
                tmp.copy_(buf, non_blocking=False)
                buf, released = tmp, bool(early and callable(release_cb))
                if released:
                    with contextlib.suppress(Exception):
                        release_cb()
            except Exception:
                released = False
        if not buf.is_contiguous():
            buf = buf.contiguous()
        return buf, released

    def _save_tensor(self: Self, tensor: torch.Tensor, path: str) -> None:
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        buf = tensor.detach()
        if hasattr(buf, "to_local"):
            buf = buf.to_local()
        if buf.device.type != "cpu":
            buf = buf.to(device="cpu", non_blocking=False)
        if not buf.is_contiguous():
            buf = buf.contiguous()
        if str(path).endswith(".mmt"):
            from tensordict import MemoryMappedTensor

            parent = os.path.dirname(path) or "."
            os.makedirs(parent, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=os.path.basename(path) + ".", suffix=".tmp", dir=parent
            )
            os.close(fd)
            try:
                MemoryMappedTensor.from_tensor(
                    buf, filename=tmp_name, existsok=True
                )
                os.replace(tmp_name, path)
            finally:
                with contextlib.suppress(Exception):
                    os.remove(tmp_name)
            write_json(
                get_meta_path(path),
                {
                    "shape": list(map(int, buf.shape)),
                    "dtype": str(buf.dtype).replace("torch.", ""),
                },
                indent=None,
            )
            return
        if str(path).endswith((".pt", ".pth")):
            save_temp(path, buf)
        else:
            torch.save(buf, path)

    def __enter__(self: Self) -> Self:
        return self

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        self.close()
        return False

    def _run(self: Self) -> None:
        while True:
            item = self._q.get()
            if item[0] is None:
                break
            tensor, path, evt, rel = item
            rel_cb = rel if callable(rel) else None
            try:
                self._wait(evt)
                buf, released_early = self._init_tensor(
                    tensor, release_cb=rel_cb
                )
                self._save_tensor(buf, path)
                if rel_cb is not None and not released_early:
                    with contextlib.suppress(Exception):
                        rel_cb()
            except Exception as e:
                self._err, _ = e, self._err_event.set()
                break
            finally:
                with contextlib.suppress(Exception):
                    self._sem.release() if self._sem else None

    def submit(
        self: Self,
        tensor: torch.Tensor,
        path: Optional[str] = None,
        idx: Optional[int] = None,
        wait_event: Optional[object] = None,
        release_cb: Optional[object] = None,
    ) -> None:
        if self._err_event.is_set():
            raise RuntimeError(f"Async writer error: {self._err!r}")
        if self._closed.is_set():
            raise RuntimeError("TensorSpooler is closed")
        path = (
            os.path.join(self._root, f"chunk_{int(idx):06d}.pt")
            if path is None and idx is not None
            else os.fspath(path)
        )
        acquired = False
        if self._sem:
            mode, timeout = self._bp_mode, self._bp_timeout_s
            if mode in ("sync", "raise"):
                acquired = self._sem.acquire(timeout=timeout)
                if not acquired:
                    if mode == "raise":
                        raise RuntimeError("TensorSpooler queue is full")
                    self._wait(wait_event)
                    buf, released = self._init_tensor(
                        tensor, release_cb=release_cb, early_release=False
                    )
                    self._save_tensor(buf, path)
                    if callable(release_cb) and not released:
                        with contextlib.suppress(Exception):
                            release_cb()
                    return
            else:
                acquired = (
                    self._sem.acquire(timeout=timeout)
                    if timeout > 0
                    else False
                )
                while not acquired:
                    if self._err_event.is_set() or self._closed.is_set():
                        raise RuntimeError("TensorSpooler unavailable")
                    acquired = self._sem.acquire(timeout=0.1)
        try:
            self._q.put((tensor, path, wait_event, release_cb))
        except Exception:
            if acquired:
                with contextlib.suppress(Exception):
                    self._sem.release()
            raise

    def close(self: Self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._q.put((None, None, None, None))
        with contextlib.suppress(Exception):
            self._future.result()
        with contextlib.suppress(Exception):
            self._executor.shutdown(wait=True)

    def had_error(self: Self) -> bool:
        return bool(self._err_event.is_set())


class BufferQueue:
    def __init__(self: Self, max_batches: int) -> None:
        self.max_batches = max(1, int(max_batches))
        self._buf: "collections.deque[Any]" = collections.deque()
        self._stop = threading.Event()
        self._cv = threading.Condition()
        self._warn_blocking = env_flag(
            "ENN_BUFFER_WARN_BLOCKING", "ENN_DEBUG", default=False
        )

    def put(
        self: Self, item: Any, *args: Any, timeout: float | None = None
    ) -> bool:
        if self._stop.is_set():
            return False
        t0 = time.monotonic()
        deadline = None if timeout is None else (t0 + float(timeout))
        with self._cv:
            if self._stop.is_set():
                return False
            while (
                len(self._buf) >= self.max_batches and not self._stop.is_set()
            ):
                if deadline is None:
                    self._cv.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._cv.wait(timeout=remaining)
            if self._stop.is_set():
                return False
            self._buf.append(item)
            self._cv.notify()
        elapsed = time.monotonic() - t0
        if self._warn_blocking and elapsed > 0.1:
            logging.warning(
                "BufferQueue.put blocked for %.3f s (max_batches=%d)",
                float(elapsed),
                int(self.max_batches),
            )
        return True

    def __len__(self: Self) -> int:
        return self.size()

    def get(
        self: Self, block: bool = True, timeout: float | None = None
    ) -> Any:
        if not bool(block):
            with self._cv:
                if not self._buf:
                    raise queue.Empty
                item = self._buf.popleft()
                self._cv.notify()
                return item
        t0 = time.monotonic()
        deadline = None if timeout is None else (t0 + float(timeout))
        with self._cv:
            while not self._buf and not self._stop.is_set():
                if deadline is None:
                    self._cv.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise queue.Empty
                    self._cv.wait(timeout=remaining)
            if not self._buf:
                raise queue.Empty
            item = self._buf.popleft()
            self._cv.notify()
            return item

    def empty(self: Self) -> bool:
        with self._cv:
            return not bool(self._buf)

    def size(self: Self) -> int:
        with self._cv:
            return int(len(self._buf))

    def block(self: Self, *args: Any, timeout: float | None = None) -> bool:
        if self._stop.is_set():
            return False
        t0 = time.monotonic()
        deadline = None if timeout is None else (t0 + float(timeout))
        with self._cv:
            while (
                len(self._buf) >= self.max_batches and not self._stop.is_set()
            ):
                if deadline is None:
                    self._cv.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._cv.wait(timeout=remaining)
            return not bool(self._stop.is_set())

    def clear(self: Self) -> None:
        with self._cv:
            self._buf.clear()
            self._cv.notify_all()

    def stop(self: Self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def is_stopped(self: Self) -> bool:
        return bool(self._stop.is_set())


class Thread:
    def __init__(
        self: Self,
        io_workers: int,
        enabled: bool = True,
        allow_omp_bind: bool = True,
    ) -> None:
        self._allowed_cpus = sorted({int(x) for x in CPU.allowed()})
        self._proc_cycle = itertools.cycle(list(self._allowed_cpus))
        self._enabled = bool(enabled) and bool(self._allowed_cpus)
        self._nogil = bool(CPU.is_optimized_for_no_gil())
        self._io_workers = max(
            1, min(int(io_workers), max(1, len(self._allowed_cpus)))
        )
        self._pin_attempts = 0
        self._pin_success = 0
        self._tls = threading.local()
        self._lock = Mutex()
        self._total_time = 0
        self._total_cpu = 0
        self._omp_ok = bool(allow_omp_bind) and bool(self.spread_threads())
        self._flush_every = max(
            1, int(env_first_int(("ENN_TLB_FLUSH_EVERY",), 256))
        )
        self._sample_every = max(
            1, int(env_first_int(("ENN_TLB_SAMPLE_EVERY",), 8))
        )

    @staticmethod
    def _import_psutil() -> Optional[ModuleType]:
        spec = importlib.util.find_spec("psutil")
        if spec is None:
            return None
        return importlib.import_module("psutil")

    def _next_core(self: Self) -> int:
        return int(next(self._proc_cycle))

    @staticmethod
    def _pin_thread_windows(core: int) -> bool:
        try:
            k32 = ctypes.WinDLL("kernel32")
            k32.GetCurrentThread.restype = ctypes.c_void_p
            handle = k32.GetCurrentThread()
            mask = ctypes.c_size_t(1 << int(core))
            k32.SetThreadAffinityMask.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            k32.SetThreadAffinityMask.restype = ctypes.c_size_t
            prev = k32.SetThreadAffinityMask(handle, mask)
            return bool(prev)
        except Exception:
            return False

    @staticmethod
    def _pin_thread_linux(core: int) -> bool:
        try:
            tid = threading.get_native_id()
            if tid <= 0:
                return False
            os.sched_setaffinity(int(tid), {int(core)})
            return True
        except Exception:
            return False

    def tune(self: Self, io_workers: Optional[int] = None) -> None:
        if io_workers is not None:
            self._io_workers = max(
                1, min(int(io_workers), max(1, len(self._allowed_cpus)))
            )
        self.tune_threads(io_workers=self._io_workers, initial=True)

    def _retune_threads(self: Self) -> None:
        if not self._enabled:
            return
        if getattr(self._tls, "in_retune", False):
            return
        self._tls.in_retune = True
        try:
            from .policies import WorkerPolicy
            from .policies import optimize_threads

            dev_type, nacc = WorkerPolicy._available_accelerator()
            is_accel = bool(nacc and int(nacc) > 0)
            cpus = max(1, len(self._allowed_cpus))
            cap_mult = _default_thread_limit(
                cpus, is_accel=is_accel, nogil=bool(self._nogil)
            )
            local_world = _optimal_local_worlds(1)
            distribute_default = local_world > 1
            distribute = bool(
                env_first_int(
                    ("ENN_DISTRIBUTE_THREAD_CAP",), int(distribute_default)
                )
            )
            thread_cap = _optimal_threads(
                ncpu=cpus,
                cap_mult=cap_mult,
                local_world=local_world,
                distribute=bool(distribute),
            )
            try:
                intra = int(torch.get_num_threads())
            except Exception:
                intra = int(cpus)
            try:
                inter = int(torch.get_num_interop_threads())
            except Exception:
                inter = 1
            workers = int(self._io_workers)
            total = int(intra) + int(inter) + int(workers)
            if total > int(thread_cap):
                new_intra = max(1, int(thread_cap) - int(inter) - int(workers))
                if int(new_intra) < int(intra):
                    optimize_threads(intra=int(new_intra))
                    intra = int(new_intra)
            total = int(intra) + int(inter) + int(workers)
            if total > int(thread_cap):
                new_inter = max(1, int(thread_cap) - int(workers) - int(intra))
                if int(new_inter) < int(inter):
                    optimize_threads(inter=int(new_inter))
        finally:
            self._tls.in_retune = False

    def total_procs(self: Self) -> list[int]:
        return list(self._allowed_cpus)

    @staticmethod
    def spread_threads() -> bool:
        plat = sys.platform
        if plat.startswith("linux"):
            candidates = [
                "libgomp.so.1",
                "libgomp.so",
                "libiomp5.so",
                "libomp.so",
            ]
        elif plat == "darwin":
            candidates = ["libomp.dylib", "libiomp5.dylib"]
        elif os.name == "nt":
            candidates = ["libiomp5md.dll", "vcomp140.dll"]
        else:
            candidates = []
        for name in candidates:
            try:
                lib = ctypes.CDLL(name)
            except OSError:
                continue
            try:
                fn = getattr(lib, "omp_set_proc_bind")
                fn.argtypes = [ctypes.c_int]
                fn.restype = None
                fn(4)
                return True
            except Exception:
                pass
            try:
                kmp = getattr(lib, "kmp_set_defaults")
                kmp.restype = None
                kmp(b"KMP_AFFINITY=granularity=fine,scatter")
                return True
            except Exception:
                pass
        return False

    def pin_thread(self: Self) -> None:
        if not self._enabled:
            return
        attempts = getattr(self._tls, "attempts", 0)
        if getattr(self._tls, "pinned", False) or attempts >= 4:
            return
        self._tls.attempts = attempts + 1
        if self._nogil:
            with self._lock:
                if not self._enabled:
                    return
                core = self._next_core()
        else:
            core = self._next_core()
        ok = False
        if os.name == "nt":
            ok = self._pin_thread_windows(core)
        else:
            plat = sys.platform
            if plat.startswith("linux"):
                ok = self._pin_thread_linux(core)
            elif plat == "darwin":
                with contextlib.suppress(Exception):
                    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
                    THREAD_AFFINITY_POLICY = 4

                    class thread_affinity_policy_data_t(ctypes.Structure):
                        _fields_ = [("affinity_tag", ctypes.c_int)]

                    policy = thread_affinity_policy_data_t(int(core) + 1)
                    lib.mach_thread_self.restype = ctypes.c_uint
                    lib.thread_policy_set.argtypes = [
                        ctypes.c_uint,
                        ctypes.c_int,
                        ctypes.c_void_p,
                        ctypes.c_uint,
                    ]
                    port = lib.mach_thread_self()
                    ok = (
                        lib.thread_policy_set(
                            port,
                            THREAD_AFFINITY_POLICY,
                            ctypes.byref(policy),
                            1,
                        )
                        == 0
                    )
        self._tls.pinned = bool(ok)
        if self._nogil:
            with self._lock:
                self._pin_attempts += 1
                if ok:
                    self._pin_success += 1
                if (
                    self._pin_attempts >= 16
                    and self._pin_success == 0
                    and not self._omp_ok
                ):
                    self._enabled = False
        else:
            self._pin_attempts += 1
            if ok:
                self._pin_success += 1
            if (
                self._pin_attempts >= 16
                and self._pin_success == 0
                and not self._omp_ok
            ):
                self._enabled = False

    def tune_threads(
        self: Self,
        io_workers: Optional[int] = None,
        *_unused_args: Any,
        initial: bool = False,
        **_unused_kwargs: Any,
    ) -> None:
        if not self._enabled:
            return
        if initial:
            cpus = max(1, len(self._allowed_cpus))
            tuned_workers = max(
                1,
                min(
                    int(
                        io_workers
                        if io_workers is not None
                        else self._io_workers
                    ),
                    cpus,
                ),
            )
            self._io_workers = tuned_workers
            from .policies import WorkerPolicy
            from .policies import optimize_threads

            dev_type, nacc = WorkerPolicy._available_accelerator()
            is_accel = bool(nacc and int(nacc) > 0)
            cap_mult = _default_thread_limit(
                cpus, is_accel=is_accel, nogil=bool(self._nogil)
            )
            local_world = _optimal_local_worlds(1)
            distribute_default = local_world > 1
            distribute = bool(
                env_first_int(
                    ("ENN_DISTRIBUTE_THREAD_CAP",), int(distribute_default)
                )
            )
            thread_cap = _optimal_threads(
                ncpu=cpus,
                cap_mult=cap_mult,
                local_world=local_world,
                distribute=bool(distribute),
            )
            try:
                intra_now = int(torch.get_num_threads())
            except Exception:
                intra_now = int(cpus)
            want_inter = max(1, min(tuned_workers // 2, 4))
            total = int(intra_now) + int(want_inter) + int(tuned_workers)
            if total > int(thread_cap):
                new_intra = max(
                    1, int(thread_cap) - int(want_inter) - int(tuned_workers)
                )
                if int(new_intra) != int(intra_now):
                    optimize_threads(intra=int(new_intra))
                    intra_now = int(new_intra)
                total = int(intra_now) + int(want_inter) + int(tuned_workers)
                if total > int(thread_cap):
                    want_inter = max(
                        1,
                        int(thread_cap) - int(tuned_workers) - int(intra_now),
                    )
            optimize_threads(inter=int(want_inter))
            return
        self._retune_threads()

    def new_thread(
        self: Self, fn: Callable[[Any], Any]
    ) -> Callable[[Any], Any]:
        if not self._enabled:
            return fn
        sample_every = (
            int(self._sample_every) if int(self._sample_every) > 0 else 1
        )
        flush_every = (
            int(self._flush_every) if int(self._flush_every) > 0 else 1
        )
        return Affinity(
            parent=self,
            fn=fn,
            pin_thread=self.pin_thread,
            tls=self._tls,
            lock=self._lock,
            tune=self.tune_threads,
            sample_every=sample_every,
            flush_every=flush_every,
            perf_counter_ns=time.perf_counter_ns,
            thread_time_ns=getattr(time, "thread_time_ns", None),
        )

    def optimize_procs(self: Self, io_workers: int) -> int:
        if not self._enabled:
            return int(io_workers)
        cpus = max(1, len(self._allowed_cpus))
        tuned = max(1, min(int(io_workers), cpus))
        self._io_workers = tuned
        return tuned


class BoundedExecutor(futures.Executor):
    def __init__(self: Self, inner: futures.Executor, limit: int) -> None:
        self._inner = inner
        self._limit = max(1, int(limit))
        self._sem = threading.BoundedSemaphore(value=self._limit)

    def submit(
        self: Self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> futures.Future:
        sem = self._sem
        sem.acquire()
        try:
            fut = self._inner.submit(fn, *args, **kwargs)
        except Exception:
            sem.release()
            raise

        def _release(_fut: futures.Future) -> None:
            sem.release()

        fut.add_done_callback(_release)
        return fut

    def map(
        self: Self,
        fn: Callable[..., Any],
        *iterables: Any,
        timeout: float | None = None,
        chunksize: int = 1,
    ) -> Any:
        del chunksize
        sem = self._sem
        end_time = None
        if timeout is not None:
            end_time = time.monotonic() + float(timeout)

        futures_list: list[futures.Future] = []
        for args in zip(*iterables):
            sem.acquire()
            try:
                fut = self._inner.submit(fn, *args)
            except Exception:
                sem.release()
                raise

            def _release(_fut: futures.Future) -> None:
                sem.release()

            fut.add_done_callback(_release)
            futures_list.append(fut)

        def _result_iter() -> Iterator[Any]:
            for fut in futures_list:
                if end_time is None:
                    yield fut.result()
                    continue
                remaining = end_time - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError()
                yield fut.result(remaining)

        return _result_iter()

    def shutdown(
        self: Self, wait: bool = True, *args: Any, cancel_futures: bool = False
    ) -> None:
        try:
            self._inner.shutdown(wait=wait, cancel_futures=cancel_futures)
        except TypeError:
            self._inner.shutdown(wait=wait)

    def __enter__(self: Self) -> "BoundedExecutor":
        self._inner.__enter__()
        return self

    def __exit__(self: Self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._inner.__exit__(exc_type, exc, tb)

    def __getattr__(self: Self, name: str) -> Any:
        return getattr(self._inner, name)


class Mutex:
    __slots__ = ("_lock", "_acquire", "_release", "_locked_fn", "__weakref__")

    def __init__(self: Self, *args: Any, reentrant: bool = False) -> None:
        lock = threading.RLock() if bool(reentrant) else threading.Lock()
        self._lock = lock
        self._acquire = lock.acquire
        self._release = lock.release
        self._locked_fn = getattr(lock, "locked", None)

    @property
    def raw(self: Self) -> threading.Lock | threading.RLock:
        return self._lock

    def acquire(
        self: Self, blocking: bool = True, timeout: float | None = None
    ) -> bool:
        if timeout is None:
            return bool(self._acquire(blocking))
        return bool(self._acquire(blocking, float(timeout)))

    def release(self: Self) -> None:
        self._release()

    def locked(self: Self) -> bool:
        fn = self._locked_fn
        if callable(fn):
            return bool(fn())
        return False

    def __enter__(self: Self) -> "Mutex":
        self.acquire(True, None)
        return self

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
