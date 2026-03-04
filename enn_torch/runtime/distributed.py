# -*- coding: utf-8 -*-
from __future__ import annotations

import tempfile
import contextlib
import sys
import ctypes
import dataclasses
import gc
import importlib
import importlib.util
import inspect
import ipaddress
import json
import logging
import os
import random
import re
import shutil
import socket
import threading
import time
import warnings
from collections import deque
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
import torch.distributed as dist
from ..core.concurrency import Mutex, is_gil_enabled
from ..core.datatypes import PathLike, env_bool, env_int, save_temp, write_json
from ..core.system import (
    CPU,
    Memory,
    get_device,
    get_num_accelerators,
    init_python_path,
    init_start_method,
    set_accelerator_index,
    set_accelerator_seed,
)
from torch import nn
from torch.optim import Optimizer


def _maybe_patch_object_collectives_pickler() -> None:
    if not env_bool("ENN_DISTRIBUTED_OBJECT_PICKLER_CLOUDPICKLE", default=True):
        return
    try:
        import cloudpickle
        import torch.distributed.distributed_c10d as c10d

        if getattr(c10d, "_pickler", None) is not cloudpickle.CloudPickler:
            c10d._pickler = cloudpickle.CloudPickler
    except Exception:
        return



def _is_tmpfs_path(path: str) -> bool:
    if not os.path.exists("/proc/mounts"):
        return False
    try:
        rp = os.path.realpath(str(path))
    except Exception:
        rp = str(path)
    try:
        best_mnt = ""
        best_fs = ""
        with open("/proc/mounts", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mnt = parts[1]
                fs = parts[2].lower()
                if not mnt:
                    continue
                if rp == mnt or rp.startswith(mnt.rstrip("/") + "/"):
                    if len(mnt) > len(best_mnt):
                        best_mnt, best_fs = mnt, fs
        return best_fs in {"tmpfs", "ramfs"}
    except Exception:
        return False


def _pick_disk_cache_base() -> str | None:
    for key in ("ENN_TEMP_DIR", "ENN_TMPDIR"):
        v = os.environ.get(key)
        if v:
            v = str(v).strip()
        if v and os.path.isdir(v) and os.access(v, os.W_OK) and (not _is_tmpfs_path(v)):
            return v
    for key in ("TMPDIR", "TEMP", "TMP"):
        v = os.environ.get(key)
        if not v:
            continue
        v = str(v).strip()
        vv = v.rstrip("/")
        if vv == "/tmp" or vv.startswith("/tmp/"):
            continue
        if os.path.isdir(v) and os.access(v, os.W_OK) and (not _is_tmpfs_path(v)):
            return v
    for cand in ("/var/tmp",):
        if os.path.isdir(cand) and os.access(cand, os.W_OK) and (not _is_tmpfs_path(cand)):
            return cand
    try:
        cwd = os.getcwd()
        if cwd and os.path.isdir(cwd) and os.access(cwd, os.W_OK) and (not _is_tmpfs_path(cwd)):
            return cwd
    except Exception:
        pass
    if os.path.isdir("/tmp") and os.access("/tmp", os.W_OK) and (not _is_tmpfs_path("/tmp")):
        return "/tmp"
    return None


def _ensure_disk_cache_env() -> None:
    base = _pick_disk_cache_base()
    if not base:
        return
    root = os.path.join(str(base), "enn_cache")
    try:
        Path(root).mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    user = None
    with contextlib.suppress(Exception):
        import getpass

        user = getpass.getuser()
    if not user:
        user = os.environ.get("USER") or "user"
    py_tag = str(getattr(getattr(sys, "implementation", None), "cache_tag", "") or "")
    torch_ver = str(getattr(torch, "__version__", "") or "")
    gil_enabled = True
    try:
        if hasattr(sys, "_is_gil_enabled"):
            gil_enabled = bool(sys._is_gil_enabled())
    except Exception:
        gil_enabled = True
    gil_tag = "gil" if gil_enabled else "nogil"

    tag_src = "_".join([s for s in (py_tag, gil_tag, torch_ver) if s])
    tag = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in tag_src).strip("_") or "pt"
    if len(tag) > 64:
        tag = tag[:64]

    cache_root = os.path.join(root, f"ptcache_{tag}")
    try:
        Path(cache_root).mkdir(parents=True, exist_ok=True)
    except Exception:
        cache_root = root

    inductor_dir = os.path.join(cache_root, f"torchinductor_{user}")

    def _is_unsafe_cache_path(key: str, cur: str | None) -> bool:
        if not cur:
            return True
        with contextlib.suppress(Exception):
            if os.path.exists(cur) and (not os.path.isdir(cur)):
                return True
        if _is_tmpfs_path(cur):
            return True
        if key in {
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "CUDA_CACHE_PATH",
            "PYTORCH_KERNEL_CACHE_PATH",
            "XDG_CACHE_HOME",
        }:
            if cur.startswith("/tmp/") or cur == "/tmp":
                return True
            if cur.startswith("/var/colab/") or cur.startswith("/dev/shm/"):
                return True
        if key in {"TMPDIR", "TEMP", "TMP"}:
            if cur.startswith("/tmp/") or cur == "/tmp":
                return True
            if cur.startswith("/var/colab/") or cur.startswith("/dev/shm/"):
                return True
        return False

    def _set_if_unset_or_unsafe(key: str, value: str) -> None:
        cur = os.environ.get(key)
        if not _is_unsafe_cache_path(key, cur):
            return
        os.environ[key] = value

    _set_if_unset_or_unsafe("TMPDIR", root)
    _set_if_unset_or_unsafe("TEMP", root)
    _set_if_unset_or_unsafe("TMP", root)
    _set_if_unset_or_unsafe("TORCHINDUCTOR_CACHE_DIR", inductor_dir)
    _set_if_unset_or_unsafe("TRITON_CACHE_DIR", os.path.join(cache_root, "triton"))
    _set_if_unset_or_unsafe("CUDA_CACHE_PATH", os.path.join(cache_root, "cuda_cache"))
    _set_if_unset_or_unsafe(
        "PYTORCH_KERNEL_CACHE_PATH", os.path.join(cache_root, "torch_kernel_cache")
    )
    _set_if_unset_or_unsafe("XDG_CACHE_HOME", os.path.join(cache_root, "xdg"))
    with contextlib.suppress(Exception):
        import tempfile

        tempfile.tempdir = None
        tempfile.gettempdir()
    with contextlib.suppress(Exception):
        Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", inductor_dir)).mkdir(
            parents=True,
            exist_ok=True,
        )
    with contextlib.suppress(Exception):
        Path(os.environ.get("TRITON_CACHE_DIR", os.path.join(cache_root, "triton"))).mkdir(
            parents=True,
            exist_ok=True,
        )
    with contextlib.suppress(Exception):
        kcache = os.environ.get(
            "PYTORCH_KERNEL_CACHE_PATH", os.path.join(cache_root, "torch_kernel_cache")
        )
        if kcache and os.path.exists(kcache) and (not os.path.isdir(kcache)):
            try:
                os.unlink(kcache)
            except Exception:
                kcache = os.path.join(cache_root, "torch_kernel_cache")
                os.environ["PYTORCH_KERNEL_CACHE_PATH"] = kcache
        if kcache:
            Path(kcache).mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(Exception):
        xdg = os.environ.get("XDG_CACHE_HOME", os.path.join(cache_root, "xdg"))
        Path(xdg).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(xdg, "torch", "kernels")).mkdir(
            parents=True,
            exist_ok=True,
        )

    with contextlib.suppress(Exception):
        import torch._inductor.config as _icfg

        if hasattr(_icfg, "global_cache_dir"):
            _icfg.global_cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        if hasattr(_icfg, "cache_dir"):
            _icfg.cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    with contextlib.suppress(Exception):
        import torch._inductor.codecache as _cc

        if hasattr(_cc, "_cache_dir"):
            _cc._cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")

with contextlib.suppress(Exception):
    _ensure_disk_cache_env()

try:
    from torch.distributed._composable.fsdp import fully_shard
except ImportError:
    with contextlib.suppress(ImportError):
        from torch.distributed.fsdp import fully_shard
try:
    from torch.distributed.algorithms.join import Join as _TorchJoin
except ImportError:
    _TorchJoin = None
Join = _TorchJoin
try:
    from torch.distributed.tensor import DTensor as _DTensor
except ImportError:
    try:
        from torch.distributed._tensor import DTensor as _DTensor
    except ImportError:
        _DTensor = None


_DTENSOR_ACTIVE: bool = False
_GLOOX_GLOO_PG_CACHE: dict[tuple[int, ...], ProcessGroup] = {}
_LOGGER = logging.getLogger(__name__)
_INFLIGHT_LOCK_NAME = ".inflight.lock"
_INFLIGHT_LOCK_TTL_SEC = int(os.environ.get("ENN_DCP_INFLIGHT_LOCK_TTL_SEC", "21600"))


class _LineProgress:
    def __init__(
        self,
        *,
        title: str,
        total: int | None,
        device: torch.device,
        file,
        mininterval: float = 1.0,
    ):
        self.title = str(title)
        self.total = int(total) if (total is not None and int(total) > 0) else None
        self.device = device
        self.file = file
        self.mininterval = float(mininterval)
        self.n = 0
        self.unit = "I/O < 0.01 MB/s, COM < 0.01 TFLOPS"
        self._postfix = ""
        self._t0 = time.monotonic()
        self._last = 0.0

    @staticmethod
    def _normalize_text(v: Any) -> str:
        s = str(v or "")
        s = s.replace("\\r\\n", " ").replace("\\n", " ").replace("\\r", " ")
        s = s.replace(chr(13), " ").replace(chr(10), " ")
        return " ".join(s.split())

    @staticmethod
    def _fmt_hms(sec: float) -> str:
        sec = max(0.0, float(sec))
        s = int(sec)
        h = s // 3600
        m = (s % 3600) // 60
        ss = s % 60
        return f"{h:02d}:{m:02d}:{ss:02d}"

    def _emit(self, force: bool = False) -> None:
        now = time.monotonic()
        if (not force) and (now - self._last) < self.mininterval:
            return
        self._last = now
        elapsed = now - self._t0
        pct = ""
        rem = ""
        if self.total is not None and self.total > 0:
            frac = max(0.0, float(self.n) / float(self.total))
            pct = f"{100.0 * frac:6.2f} % "
            if self.n > 0 and elapsed > 0:
                rate = float(self.n) / float(elapsed)
                remaining_units = max(0.0, float(self.total) - float(self.n))
                rem_s = remaining_units / rate if rate > 0 else 0.0
                rem = f", Remaining: {self._fmt_hms(rem_s)}"
            else:
                rem = ", Remaining: ?"
        line = (
            f"{self.title} ({str(getattr(self.device, 'type', 'cpu')).upper()}) "
            f"{pct}({self.unit}) Elapsed: {self._fmt_hms(elapsed)}{rem}"
        )
        if self._postfix:
            line += f" {self._postfix}"
        with contextlib.suppress(Exception):
            self.file.write(line + "\n")
            self.file.flush()

    def update(self, n: int = 1):
        with contextlib.suppress(Exception):
            self.n += int(n)
        self._emit(False)

    def set_postfix_str(self, s: str, refresh: bool = False):
        self._postfix = self._normalize_text(s)
        if refresh:
            self._emit(True)

    def refresh(self):
        self._emit(True)

    def close(self):
        self._emit(True)

_DCP_NOISE_RE = re.compile(
    r"("
    r"Initializing dist\.ProcessGroup in checkpoint background process"
    r"|Checkpoint background process is running"
    r"|Checkpoint background process is shutting down"
    r"|Waiting for checkpoint save request"
    r"|Received async checkpoint request"
    r"|Completed checkpoint save request"
    r"|Checkpoint save failed for checkpoint_id="
    r")",
    re.IGNORECASE,
)


class _ENNDropTorchDCPNoise(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if not msg:
            return True
        return _DCP_NOISE_RE.search(msg) is None


def _install_torch_dcp_noise_filter() -> None:
    root = logging.getLogger()
    try:
        for f in getattr(root, "filters", []) or []:
            if isinstance(f, _ENNDropTorchDCPNoise):
                return
    except Exception:
        pass
    flt = _ENNDropTorchDCPNoise()
    with contextlib.suppress(Exception):
        root.addFilter(flt)
    for h in list(getattr(root, "handlers", []) or []):
        with contextlib.suppress(Exception):
            h.addFilter(flt)


with contextlib.suppress(Exception):
    _install_torch_dcp_noise_filter()


def _is_oomish_error(exc: BaseException) -> bool:
    if isinstance(exc, (MemoryError, EOFError, BrokenPipeError)):
        return True
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cuda out of memory" in msg
        or "cannot allocate memory" in msg
        or "cudahostalloc" in msg
        or ("alloc" in msg and "failed" in msg and "memory" in msg)
    )


if env_bool("ENN_SUPPRESS_TYPEDSTORAGE_WARNING", default=True):
    _rule = "ignore:TypedStorage is deprecated:UserWarning"
    _w = os.environ.get("PYTHONWARNINGS", "")
    if not _w:
        os.environ["PYTHONWARNINGS"] = _rule
    elif _rule not in _w:
        os.environ["PYTHONWARNINGS"] = f"{_w},{_rule}"


def _atomic_create_json(path, payload: dict) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        with open(path, "x", encoding="utf-8") as f:
            json.dump(payload, f)
        return True
    except FileExistsError:
        return False
    except Exception:
        with contextlib.suppress(Exception):
            if path.exists():
                path.unlink()
        return False


def _safe_rmtree(path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    try:
        if not path.exists():
            return
    except Exception:
        return
    ts = int(time.time())
    dp = path.with_name(f"{path.name}.delete_pending.{ts}")
    with contextlib.suppress(Exception):
        path.rename(dp)
    shutil.rmtree(dp, ignore_errors=True)


def _cleanup_delete_pending(root) -> None:
    with contextlib.suppress(Exception):
        for p in root.glob("*.delete_pending.*"):
            _safe_rmtree(p)


def _coerce_dcp_keys(state: object) -> object:
    if isinstance(state, dict):
        keys_to_drop: list[object] = []
        _yield_every = int(os.environ.get("ENN_DCP_COERCE_YIELD_EVERY", "1024") or 1024)
        _seen = 0
        for key, value in state.items():
            _seen += 1
            if _yield_every > 0 and (_seen % _yield_every) == 0:
                with contextlib.suppress(Exception):
                    time.sleep(0)

            key_str = str(key)
            if (
                key_str.endswith("._extra_state")
                or key_str.endswith("_extra_state")
                or key_str.endswith("output_baked_flag")
            ):
                keys_to_drop.append(key)
                continue
            state[key] = _coerce_dcp_keys(value)
        for key in keys_to_drop:
            with contextlib.suppress(Exception):
                state.pop(key, None)
    return state


def _overlay_avg_state_dict(dst: object, avg: Mapping[str, Any]) -> object:
    if not isinstance(dst, dict) or not isinstance(avg, Mapping):
        return dst
    for k, v in avg.items():
        if not isinstance(k, str) or not torch.is_tensor(v):
            continue
        cur = dst.get(k, None)
        if not torch.is_tensor(cur):
            continue
        try:
            vv = v.detach()
            if tuple(vv.shape) != tuple(cur.shape):
                continue
            if vv.dtype != cur.dtype:
                vv = vv.to(dtype=cur.dtype)
            cur.copy_(vv, non_blocking=True)
        except Exception:
            continue
    return dst


def _clone_state_dict(state: object, *, to_cpu: bool = False) -> object:
    if torch.is_tensor(state):
        t = state.detach()
        if to_cpu and getattr(t, "device", None) is not None and t.device.type != "cpu":
            return t.to(device="cpu")
        return t.clone()
    if isinstance(state, dict):
        return {k: _clone_state_dict(v, to_cpu=to_cpu) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        cloned = (_clone_state_dict(v, to_cpu=to_cpu) for v in state)
        return type(state)(cloned)
    return state


def _future_result(fut: object) -> object:
    if fut is None:
        return None
    wait_fn = getattr(fut, "wait", None)
    if callable(wait_fn):
        r = wait_fn()
        result_fn = getattr(fut, "result", None)
        if callable(result_fn):
            return result_fn()
        return r
    result_fn = getattr(fut, "result", None)
    if callable(result_fn):
        return result_fn()
    return fut


def _future_done(fut: object) -> bool:
    if fut is None:
        return True
    done_fn = getattr(fut, "done", None)
    if callable(done_fn):
        try:
            return bool(done_fn())
        except Exception:
            return False
    return False


def _add_future_callback(fut: object, fn: Callable[[], None]) -> None:
    if fut is None:
        return
    then = getattr(fut, "then", None)
    if callable(then):
        with contextlib.suppress(Exception):
            then(lambda _: fn())
            return
    add_done = getattr(fut, "add_done_callback", None)
    if callable(add_done):
        with contextlib.suppress(Exception):
            add_done(lambda _: fn())


def _set_dtensor_active() -> None:
    global _DTENSOR_ACTIVE
    _DTENSOR_ACTIVE = True


def _from_hsdp_module(module: torch.nn.Module) -> None:
    if not (
        is_distributed()
        and callable(unshard := getattr(module, "unshard", None))
    ):
        return
    try:
        if (p0 := next(module.parameters(recurse=False), None)) is not None:
            is_dt = (
                isinstance(p0, _DTensor)
                or isinstance(getattr(p0, "data", None), _DTensor)
                if _DTensor
                else (
                    type(p0).__name__ == "DTensor"
                    or type(getattr(p0, "data", None)).__name__ == "DTensor"
                )
            )
            if is_dt:
                _set_dtensor_active()
            if not is_dt:
                return
        if (handle := unshard(async_op=True)) and callable(
            wait := getattr(handle, "wait", None)
        ):
            wait()
    except:
        pass


def _coerce_ip_addr(v: Any, strip_zone: bool = True) -> str:
    s = str(v).strip() if v is not None else ""
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    if strip_zone and "%" in s:
        s = s.partition("%")[0].strip()
    return s


def _is_ip_addr(value: str) -> bool:
    try:
        ipaddress.ip_address(_coerce_ip_addr(value))
        return True
    except:
        return False


def _canonize_ip(
    value: Any, loopback: bool = False, link_local: bool = False
) -> str | None:
    s = _coerce_ip_addr(value)
    if not s:
        return None
    try:
        ip = ipaddress.ip_address(s)
        if (not loopback and (ip.is_unspecified or ip.is_loopback)) or (
            ip.is_link_local and not link_local
        ):
            return None
        return f"{ip.compressed}{'%' + value.partition('%')[2] if '%' in str(value) and ip.version == 6 else ''}"
    except:
        return None


def _format_endpoint(host: str, port: int) -> str:
    host_text = str(host).strip() if host is not None else ""
    if host_text.startswith("[") and host_text.endswith("]"):
        host_text = host_text[1:-1].strip()
    h, p = host_text or "127.0.0.1", int(port)
    try:
        h = (
            f"[{h}]"
            if ipaddress.ip_address(_coerce_ip_addr(h)).version == 6
            and ":" in h
            else h
        )
    except:
        pass
    return f"{h}:{p}"


def _parse_endpoint(text: str) -> tuple[str, int]:
    text = text.strip()
    if not text:
        return "", 0
    if text.startswith("["):
        host, _, rest = text[1:].partition("]")
        return host.strip(), (
            int(rest[1:]) if rest.startswith(":") and rest[1:].isdigit() else 0
        )
    if _is_ip_addr(text):
        return text, 0
    host, sep, port = text.rpartition(":")
    if sep and port.isdigit():
        return host.strip(), int(port)
    return text, 0


def _canonize_host(ep: str, default: str, link_local: bool) -> tuple[str, int]:
    if not ep:
        return default, 0
    host, port = _parse_endpoint(ep)
    host = _canonize_ip(
        host or default, loopback=True, link_local=link_local
    ) or (default if _is_ip_addr(host) else host)
    return host, port if 0 < port <= 65535 else 0


def _has_join_hook(obj: Any | None) -> bool:
    return obj is not None and getattr(obj, "join_hook", None) is not None


def _get_device_id(device: Optional[torch.device]) -> Optional[Iterable[int]]:
    return (
        [int(device.index)]
        if device
        and device.type in {"cuda", "xpu"}
        and device.index is not None
        else None
    )


def _safe_getaddrinfo(host: str) -> list[tuple[Any, ...]]:
    with contextlib.suppress(Exception):
        return socket.getaddrinfo(
            host, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
        )
    return []


def _get_preferred_ip_cached(
    hostname: str | None,
    prefer_ipv6: bool,
    allow_loopback: bool,
    allow_link_local: bool,
) -> str:
    names = [h for h in [hostname, socket.gethostname()] if h] + (
        ["localhost"] if allow_loopback else []
    )
    found, seen = [], set()
    for name in names:
        for info in _safe_getaddrinfo(name):
            if not info[4] or not (
                canon := _canonize_ip(
                    info[4][0], loopback=True, link_local=allow_link_local
                )
            ):
                continue
            try:
                ip = ipaddress.ip_address(_coerce_ip_addr(canon))
            except:
                continue
            if ip.is_unspecified:
                continue
            if (ip.version, canon) in seen:
                continue
            seen.add((ip.version, canon))

            score = 1 if ip.is_loopback else (2 if ip.is_link_local else 3)
            if (score == 1 and not allow_loopback) or (
                score == 2 and not allow_link_local
            ):
                continue
            found.append(
                (score, ip.version == (6 if prefer_ipv6 else 4), canon)
            )
    found.sort(key=lambda x: x[:2], reverse=True)
    if found:
        return found[0][2]
    return "::" if prefer_ipv6 else "0.0.0.0"


def _get_default_process_group() -> Any:
    try:
        return dist.group.WORLD
    except Exception:
        pass
    try:
        return dist.distributed_c10d._get_default_group()
    except Exception:
        return None


def _hsdp_supported_params() -> set[str]:
    if fully_shard is None:
        return set()
    try:
        sig = inspect.signature(fully_shard)
        return set(sig.parameters.keys())
    except Exception:
        return set()


def _accel_backend_for_device(device: torch.device) -> str | None:
    dt = str(getattr(device, "type", "cpu") or "cpu").strip().lower()
    if dt == "cuda":
        return "nccl"
    if dt == "xpu":
        return "xccl"
    if dt in {"hpu", "npu"}:
        return "hccl"
    return None


_CPU_GROUP: ProcessGroup | None = None
_ACCEL_GROUP: ProcessGroup | None = None
_LANE_GROUPS_INITED: bool = False
_LANE_GROUPS_LOCK = threading.Lock()
_ACCEL_BACKEND: str | None = None


def _reset_lane_process_groups() -> None:
    global _CPU_GROUP, _ACCEL_GROUP, _LANE_GROUPS_INITED, _ACCEL_BACKEND
    with _LANE_GROUPS_LOCK:
        cpu_pg = _CPU_GROUP
        accel_pg = _ACCEL_GROUP
        _CPU_GROUP = None
        _ACCEL_GROUP = None
        _ACCEL_BACKEND = None
        _LANE_GROUPS_INITED = False
    with contextlib.suppress(Exception):
        if dist.is_available() and dist.is_initialized():
            if accel_pg is not None:
                dist.destroy_process_group(accel_pg)
            if cpu_pg is not None:
                dist.destroy_process_group(cpu_pg)


def init_lane_process_groups(
    device: torch.device | None = None,
) -> tuple[ProcessGroup | None, ProcessGroup | None]:
    global _CPU_GROUP, _ACCEL_GROUP, _LANE_GROUPS_INITED, _ACCEL_BACKEND
    if not is_distributed():
        return (None, None)

    if _LANE_GROUPS_INITED:
        return (_CPU_GROUP, _ACCEL_GROUP)

    with _LANE_GROUPS_LOCK:
        if _LANE_GROUPS_INITED:
            return (_CPU_GROUP, _ACCEL_GROUP)

        dev = device
        if dev is None:
            with contextlib.suppress(Exception):
                dev = get_device()
        if dev is None:
            dev = torch.device("cpu")

        world = 1
        with contextlib.suppress(Exception):
            world = int(dist.get_world_size())

        _CPU_GROUP = None
        if world > 1 and not env_bool(
            ("ENN_DISABLE_LANE_CPU_GROUP", "ENN_DISABLE_CPU_LANE_GROUP"),
            default=False,
        ):
            try:
                _CPU_GROUP = dist.new_group(backend="gloo")
            except Exception:
                _CPU_GROUP = None

        be = _accel_backend_for_device(dev)
        _ACCEL_BACKEND = be
        if world > 1 and be is not None and be != "gloo" and not env_bool(
            ("ENN_DISABLE_LANE_ACCEL_GROUP", "ENN_DISABLE_ACCEL_LANE_GROUP"),
            default=False,
        ):
            try:
                _ACCEL_GROUP = dist.new_group(backend=str(be))
            except Exception:
                _ACCEL_GROUP = None
        else:
            _ACCEL_GROUP = None

        _LANE_GROUPS_INITED = True
        return (_CPU_GROUP, _ACCEL_GROUP)


def get_cpu_group() -> ProcessGroup | None:
    return init_lane_process_groups()[0]


def get_accel_group(
    device: torch.device | None = None,
) -> ProcessGroup | None:
    return init_lane_process_groups(device)[1]


def get_control_process_group(pg: ProcessGroup | None = None) -> ProcessGroup | None:
    if not is_distributed():
        return None

    def _cfg(g: ProcessGroup) -> str:
        with contextlib.suppress(Exception):
            return str(dist.get_backend_config(g)).lower()
        with contextlib.suppress(Exception):
            return str(dist.get_backend(g)).lower()
        return ""

    def _is_cpu_capable(cfg: str) -> bool:
        c = (cfg or "").lower()
        return (c == "gloo") or ("cpu:gloo" in c)

    if pg is not None:
        try:
            if _is_cpu_capable(_cfg(pg)):
                return pg
        except Exception:
            pass
        return None
    cpg = get_cpu_group()
    if cpg is not None:
        return cpg
    try:
        if _is_cpu_capable(_cfg(dist.group.WORLD)):
            return dist.group.WORLD
    except Exception:
        pass
    return None


def get_accel_process_group(
    device: torch.device | None = None,
) -> ProcessGroup | None:
    if not is_distributed():
        return None
    return get_accel_group(device)


def _get_gloox_gloo_process_group(pg: ProcessGroup | None) -> ProcessGroup:
    if not is_distributed():
        return pg or dist.group.WORLD

    base_pg = pg or dist.group.WORLD

    try:
        if dist.get_backend(base_pg) == "gloo":
            return base_pg
    except Exception:
        pass

    try:
        ranks = tuple(dist.get_process_group_ranks(base_pg))
    except Exception:
        ranks = tuple(range(dist.get_world_size(base_pg)))

    cached = _GLOOX_GLOO_PG_CACHE.get(ranks)
    if cached is not None:
        return cached

    try:
        gloo_pg = dist.new_group(ranks=list(ranks), backend="gloo")
    except Exception as exc:
        raise RuntimeError("Failed to create gloo process group") from exc

    _GLOOX_GLOO_PG_CACHE[ranks] = gloo_pg
    return gloo_pg


def _iter_buckets_by_bytes(
    tensors: list[Tensor], max_bucket_bytes: int
) -> Iterable[list[Tensor]]:
    bucket: list[Tensor] = []
    bucket_bytes = 0

    for t in tensors:
        t_bytes = t.numel() * t.element_size()
        if bucket and bucket_bytes + t_bytes > max_bucket_bytes:
            yield bucket
            bucket = []
            bucket_bytes = 0

        bucket.append(t)
        bucket_bytes += t_bytes

    if bucket:
        yield bucket


def _broadcast_bucket_gloox(
    tensors: list[Tensor], *args: Any, src_rank: int, group: ProcessGroup
) -> None:
    if not tensors:
        return

    rank = dist.get_rank(group)

    cpu_parts: list[Tensor] = []
    for t in tensors:
        if rank == src_rank:
            cpu_t = t.detach().to(device="cpu")
        else:
            cpu_t = torch.empty_like(t, device="cpu")
        cpu_parts.append(cpu_t.contiguous())

    flat = torch._utils._flatten_dense_tensors(cpu_parts)
    dist.broadcast(flat, src=src_rank, group=group)

    out_parts = torch._utils._unflatten_dense_tensors(flat, cpu_parts)
    for orig, cpu in zip(tensors, out_parts):
        if orig.device.type == "cpu":
            orig.copy_(cpu)
        else:
            orig.copy_(cpu, non_blocking=True)


def _broadcast_large_tensor_gloox(
    tensor: Tensor,
    *args: Any,
    src_rank: int,
    group: ProcessGroup,
    chunk_mb: int,
    max_inflight_mb: int,
) -> None:
    flat = tensor.reshape(-1)
    elem_size = flat.element_size()

    max_inflight_bytes = max(0, int(max_inflight_mb) * 1024 * 1024)

    chunk_bytes = max(1, int(chunk_mb) * 1024 * 1024)
    if max_inflight_bytes > 0 and chunk_bytes > max_inflight_bytes:
        chunk_bytes = max_inflight_bytes
    chunk_elems = max(1, chunk_bytes // elem_size)

    use_async = max_inflight_bytes > 0

    rank = dist.get_rank(group)

    pending: list[tuple[dist.Work, Tensor, int, int]] = []
    inflight_bytes = 0

    offset = 0
    total = flat.numel()

    while offset < total:
        n = min(chunk_elems, total - offset)

        if rank == src_rank:
            cpu_chunk = flat[offset : offset + n].detach().to(device="cpu")
        else:
            cpu_chunk = torch.empty((n,), dtype=flat.dtype, device="cpu")

        work = dist.broadcast(
            cpu_chunk, src=src_rank, group=group, async_op=use_async
        )

        if use_async:
            pending.append((work, cpu_chunk, offset, n))
            inflight_bytes += n * elem_size

            if inflight_bytes >= max_inflight_bytes:
                for w, c, o, nn in pending:
                    w.wait()
                    if rank != src_rank:
                        flat[o : o + nn].copy_(c, non_blocking=True)

                pending.clear()
                inflight_bytes = 0
        else:
            if rank != src_rank:
                flat[offset : offset + n].copy_(cpu_chunk, non_blocking=True)

        offset += n

    for w, c, o, nn in pending:
        w.wait()
        if rank != src_rank:
            flat[o : o + nn].copy_(c, non_blocking=True)


def _broadcast_large_tensor(
    tensor: Tensor,
    *args: Any,
    group: ProcessGroup,
    src_rank: int,
    chunk_mb: int,
    max_inflight_mb: int,
) -> None:
    if not is_distributed():
        return

    flat = tensor.reshape(-1)
    elem_size = flat.element_size()

    max_inflight_bytes = max(0, int(max_inflight_mb) * 1024 * 1024)
    chunk_bytes = max(1, int(chunk_mb) * 1024 * 1024)
    if max_inflight_bytes > 0 and chunk_bytes > max_inflight_bytes:
        chunk_bytes = max_inflight_bytes
    chunk_elems = max(1, chunk_bytes // max(1, elem_size))

    use_async = max_inflight_bytes > 0

    pending: list[dist.Work] = []
    inflight_bytes = 0

    offset = 0
    total = flat.numel()

    while offset < total:
        n = min(chunk_elems, total - offset)
        view = flat[offset : offset + n]

        work = dist.broadcast(
            view, src=src_rank, group=group, async_op=use_async
        )

        if use_async:
            pending.append(work)
            inflight_bytes += n * elem_size
            if inflight_bytes >= max_inflight_bytes:
                for w in pending:
                    w.wait()
                pending.clear()
                inflight_bytes = 0
        else:
            pass

        offset += n

    for w in pending:
        w.wait()


def _all_reduce_tensor_gloox(
    tensor: Tensor,
    *args: Any,
    group: ProcessGroup,
    chunk_mb: int,
    max_inflight_mb: int,
    average: bool,
    world_size: int,
) -> None:
    if tensor.numel() == 0:
        return

    if tensor.device.type == "cpu":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        if average:
            tensor.div_(world_size)
        return

    if not tensor.is_contiguous():
        tmp = tensor.contiguous()
        _all_reduce_tensor_gloox(
            tmp,
            group=group,
            chunk_mb=chunk_mb,
            max_inflight_mb=max_inflight_mb,
            average=average,
            world_size=world_size,
        )
        tensor.copy_(tmp)
        return

    flat = tensor.view(-1)
    elem_size = flat.element_size()
    chunk_bytes = max(1, int(chunk_mb) * 1024 * 1024)
    chunk_elems = max(1, chunk_bytes // elem_size)

    max_inflight_bytes = max(0, int(max_inflight_mb) * 1024 * 1024)
    use_async = max_inflight_bytes > 0

    pending: list[tuple[Any, Tensor, int, int]] = []
    inflight = 0

    offset = 0
    total = flat.numel()
    while offset < total:
        n = min(chunk_elems, total - offset)
        cpu_chunk = flat[offset : offset + n].detach().to("cpu")

        work = dist.all_reduce(
            cpu_chunk,
            op=dist.ReduceOp.SUM,
            group=group,
            async_op=use_async,
        )

        if use_async:
            pending.append((work, cpu_chunk, offset, n))
            inflight += n * elem_size

            if inflight >= max_inflight_bytes:
                for w, c, o, nn in pending:
                    w.wait()
                    if average:
                        c.div_(world_size)
                    flat[o : o + nn].copy_(c, non_blocking=True)
                pending.clear()
                inflight = 0
        else:
            if average:
                cpu_chunk.div_(world_size)
            flat[offset : offset + n].copy_(cpu_chunk, non_blocking=True)

        offset += n

    for w, c, o, nn in pending:
        w.wait()
        if average:
            c.div_(world_size)
        flat[o : o + nn].copy_(c, non_blocking=True)


def resolve_ip_expr(
    host: Any,
    *args: Any,
    allow_loopback: bool = False,
    prefer_ipv6: bool | None = None,
    allow_link_local: bool | None = None,
    **kwargs: Any,
) -> str | None:
    host_text = str(host).strip() if host else ""
    if not host_text:
        return None
    link_local = (
        allow_link_local
        if allow_link_local is not None
        else env_bool("ENN_ALLOW_LINK_LOCAL", False)
    )
    if lit := _canonize_ip(
        host_text, loopback=allow_loopback, link_local=link_local
    ):
        return lit
    addrs = _safe_getaddrinfo(_coerce_ip_addr(host_text))
    if not addrs:
        return None
    res = {4: [], 6: []}
    for a in addrs:
        if a[4] and (
            canon := _canonize_ip(
                a[4][0], loopback=allow_loopback, link_local=link_local
            )
        ):
            with contextlib.suppress(Exception):
                res[
                    ipaddress.ip_address(_coerce_ip_addr(canon)).version
                ].append(canon)
    vers = (6, 4) if (prefer_ipv6 is None or prefer_ipv6) else (4, 6)
    for v in vers:
        if res[v]:
            return res[v][0]
    if "%" in host_text and link_local:
        for v in vers:
            for ip in res[v]:
                if (
                    v == 6
                    and ipaddress.ip_address(_coerce_ip_addr(ip)).is_link_local
                ):
                    return f"{ip}%{host_text.partition('%')[2]}"
    return None


def validate_ip_expr(
    host: Any,
    *args: Any,
    fallback: Any | None = None,
    default: str = "127.0.0.1",
    allow_loopback: bool = False,
    allow_hostname: bool = True,
    allow_link_local: bool | None = None,
    **kwargs: Any,
) -> str:
    link_local = (
        allow_link_local
        if allow_link_local is not None
        else env_bool("ENN_ALLOW_LINK_LOCAL", False)
    )
    for h in [host, fallback]:
        txt = str(h).strip() if h else ""
        if txt:
            if _is_ip_addr(txt):
                if lit := _canonize_ip(
                    txt, loopback=allow_loopback, link_local=link_local
                ):
                    return lit
            elif allow_hostname:
                return txt
    return _canonize_ip(default, loopback=True, link_local=True) or default


def is_port_available(
    host: str, port: int, *args: Any, allow_link_local: bool | None = None
) -> bool:
    if port <= 0 or port > 65535:
        return False
    link_local = (
        allow_link_local
        if allow_link_local is not None
        else env_bool("ENN_ALLOW_LINK_LOCAL", False)
    )
    host_ip = _canonize_ip(
        host, loopback=True, link_local=link_local
    ) or resolve_ip_expr(
        host,
        allow_loopback=True,
        prefer_ipv6=True,
        allow_link_local=link_local,
    )
    if not host_ip:
        return False
    try:
        ver = ipaddress.ip_address(_coerce_ip_addr(host_ip)).version
        family = socket.AF_INET6 if ver == 6 else socket.AF_INET
        addr = (host_ip, port, 0, 0) if ver == 6 else (host_ip, port)
        with contextlib.closing(
            socket.socket(family, socket.SOCK_STREAM)
        ) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if ver == 6 and hasattr(socket, "IPV6_V6ONLY"):
                with contextlib.suppress(OSError):
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind(addr)
            return True
    except OSError:
        return False


def get_available_host(
    endpoint: Optional[str],
    *args: Any,
    default_host: str = "127.0.0.1",
    allow_link_local: bool | None = None,
    **kwargs: Any,
) -> str:
    link_local = (
        allow_link_local
        if allow_link_local is not None
        else env_bool("ENN_ALLOW_LINK_LOCAL", False)
    )
    if endpoint:
        h, p = _canonize_host(endpoint, default_host, link_local)
        if p > 0:
            return _format_endpoint(h, p)

    cand = (
        _canonize_ip(default_host, loopback=True, link_local=True)
        or str(default_host).strip()
        or "127.0.0.1"
    )
    host = (
        _canonize_ip(cand, loopback=True, link_local=link_local)
        or resolve_ip_expr(
            cand,
            allow_loopback=True,
            prefer_ipv6=True,
            allow_link_local=link_local,
        )
        or "127.0.0.1"
    )
    try:
        ver = ipaddress.ip_address(_coerce_ip_addr(host)).version
        family, addr = (
            (socket.AF_INET6, (host, 0, 0, 0))
            if ver == 6
            else (socket.AF_INET, (host, 0))
        )
        with contextlib.closing(
            socket.socket(family, socket.SOCK_STREAM)
        ) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if ver == 6 and hasattr(socket, "IPV6_V6ONLY"):
                with contextlib.suppress(OSError):
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind(addr)
            return _format_endpoint(host, int(sock.getsockname()[1]))
    except OSError:
        return _format_endpoint(host, 0)


def supported_ip_ver(
    *args: Any, allow_loopback: bool = True, **kwargs: Any
) -> tuple[bool, bool]:
    ipv4_ok = False
    ipv6_ok = False
    ipv4_host = "127.0.0.1" if allow_loopback else "0.0.0.0"
    try:
        with contextlib.closing(
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ) as sock4:
            sock4.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock4.bind((ipv4_host, 0))
            ipv4_ok = True
    except OSError:
        pass
    if getattr(socket, "has_ipv6", False):
        ipv6_host = "::1" if allow_loopback else "::"
        try:
            with contextlib.closing(
                socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            ) as sock6:
                if hasattr(socket, "IPPROTO_IPV6") and hasattr(
                    socket, "IPV6_V6ONLY"
                ):
                    with contextlib.suppress(OSError):
                        sock6.setsockopt(
                            socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1
                        )
                sock6.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock6.bind((ipv6_host, 0, 0, 0))
                ipv6_ok = True
        except OSError:
            pass
    return ipv4_ok, ipv6_ok


def get_preferred_ip(
    hostname: Optional[str] = None,
    *args: Any,
    prefer_ipv6: bool = True,
    allow_loopback: bool = True,
    allow_link_local: bool | None = None,
    **kwargs: Any,
) -> str:
    if allow_link_local is None:
        allow_link_local = env_bool("ENN_ALLOW_LINK_LOCAL", False)
    hn = None if hostname is None else str(hostname)
    return _get_preferred_ip_cached(
        hn, bool(prefer_ipv6), bool(allow_loopback), bool(allow_link_local)
    )


def init_master_addr(
    endpoint: Optional[str],
    *args: Any,
    prefer_ipv6: bool = True,
    allow_loopback: bool = True,
    allow_link_local: bool | None = None,
    **kwargs: Any,
) -> tuple[str, int]:
    link_local = (
        allow_link_local
        if allow_link_local is not None
        else env_bool("ENN_ALLOW_LINK_LOCAL", False)
    )
    default_host = get_preferred_ip(
        allow_loopback=allow_loopback,
        prefer_ipv6=prefer_ipv6,
        allow_link_local=link_local,
    )
    default_host = default_host or ("::1" if prefer_ipv6 else "127.0.0.1")

    host, port = _canonize_host(endpoint or "", default_host, link_local)
    if host in {"", "0.0.0.0", "::"}:
        host = default_host

    master_addr = (
        _canonize_ip(host, loopback=allow_loopback, link_local=link_local)
        or host
        or default_host
    )
    os.environ.setdefault("MASTER_ADDR", master_addr)
    if port > 0:
        os.environ.setdefault("MASTER_PORT", str(int(port)))
    return master_addr, int(port)


def get_world_size(device: Optional[torch.device] = None) -> int:
    try:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
    except Exception:
        pass
    dev = device
    if dev is None:
        with contextlib.suppress(Exception):
            dev = get_device()
    if dev is None:
        dev = torch.device("cpu")
    match getattr(dev, "type", "cpu"):
        case "cuda" | "xpu" | "mps":
            with contextlib.suppress(Exception):
                count = int(
                    get_num_accelerators(
                        str(getattr(dev, "type", "cpu") or "cpu")
                    )
                )
                if count > 0:
                    return count
            return 1
        case _:
            ncpu = CPU.count()
            return max(1, min(int(ncpu), 4))


def is_process_group(obj: object) -> bool:
    if obj is None:
        return False
    with contextlib.suppress(Exception):
        from torch.distributed.distributed_c10d import ProcessGroup

        return isinstance(obj, ProcessGroup)
    return False


def resolve_process_group(meta: object, model: object) -> object | None:
    candidates: list[tuple[object, str]] = [
        (meta, "process_group"),
        (meta, "distributed_process_group"),
    ]
    tm = model.module if hasattr(model, "module") else model
    candidates.extend(
        [
            (tm, "process_group"),
            (tm, "distributed_process_group"),
        ]
    )
    for obj, attr in candidates:
        try:
            pg = getattr(obj, attr, None)
        except Exception:
            pg = None
        if is_process_group(pg):
            return pg
    return None


def get_group_world_size(group: object | None) -> int:
    try:
        if group is None or not is_process_group(group):
            if dist.is_available() and dist.is_initialized():
                return int(dist.get_world_size())
            return int(get_world_size())
        return int(dist.get_world_size(group=group))
    except Exception:
        return int(get_world_size())


def distributed_all_reduce_sum(
    t: torch.Tensor, group: object | None = None
) -> None:
    if not isinstance(t, torch.Tensor):
        return
    try:
        if not (dist.is_available() and dist.is_initialized()):
            return
    except Exception:
        return

    try:
        if group is None or not is_process_group(group):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        else:
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
    except Exception:
        with contextlib.suppress(Exception):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)


@contextlib.contextmanager
def no_sync(
    model: torch.nn.Module,
    *args: Any,
    enable: bool = True,
) -> AbstractContextManager[None]:
    if not enable:
        yield
        return

    ctx = getattr(model, "no_sync", lambda: contextlib.nullcontext())()
    with ctx:
        yield


def joining(
    model: Any,
    optimizer: Optimizer | None = None,
) -> AbstractContextManager[None]:
    if Join is None:
        return contextlib.nullcontext()
    joinables = tuple(obj for obj in (model, optimizer) if _has_join_hook(obj))
    if not joinables:
        return contextlib.nullcontext()
    return Join(joinables, throw_on_early_termination=True)


def broadcast_scalar(
    value: int | float,
    device: torch.device | None = None,
    src: int = 0,
    group: ProcessGroup | None = None,
    *,
    lane: str = "auto",
) -> int:
    if not is_distributed():
        return int(value)

    pg_in = group
    pg = group or dist.group.WORLD
    with contextlib.suppress(Exception):
        if int(dist.get_world_size(pg)) <= 1:
            return int(value)

    dev = device
    if dev is None:
        with contextlib.suppress(Exception):
            dev = get_device()
    if dev is None:
        dev = torch.device("cpu")

    lane_s = str(lane).strip().lower()
    if lane_s in {"auto", ""}:
        lane_s = (
            "accelerator"
            if get_accel_group(dev) is not None
            and getattr(dev, "type", "cpu") != "cpu"
            else "control"
        )

    def _accel_pg(default_pg: ProcessGroup) -> ProcessGroup:
        if pg_in is not None:
            return default_pg
        ag = get_accel_group(dev)
        return ag or default_pg

    if lane_s in {"control", "cpu", "gloo"}:
        cpg = (
            get_control_process_group(pg_in)
            if pg_in is not None
            else (get_cpu_group() or get_control_process_group(None))
        )
        cpu_exc: Exception | None = None
        if cpg is not None:
            try:
                t = torch.tensor([int(value)], device="cpu", dtype=torch.int32)
                dist.broadcast(t, src=int(src), group=cpg)
                return int(t.item())
            except Exception as exc:
                cpu_exc = exc
        dev2 = dev
        if getattr(dev2, "type", "cpu") == "cpu" and torch.cuda.is_available():
            with contextlib.suppress(Exception):
                dev2 = torch.device("cuda", torch.cuda.current_device())
        try:
            t = torch.tensor([int(value)], device=dev2, dtype=torch.int32)
            dist.broadcast(t, src=int(src), group=_accel_pg(pg))
            return int(t.item())
        except Exception as exc2:
            raise RuntimeError(
                "broadcast_scalar failed on both control-plane (CPU/gloo) and accelerator lanes"
            ) from (exc2 if cpu_exc is None else exc2)

    dev2 = dev
    if getattr(dev2, "type", "cpu") == "cpu" and torch.cuda.is_available():
        with contextlib.suppress(Exception):
            dev2 = torch.device("cuda", torch.cuda.current_device())
    t = torch.tensor([int(value)], device=dev2, dtype=torch.int32)
    dist.broadcast(t, src=int(src), group=_accel_pg(pg))
    return int(t.item())


def is_distributed() -> bool:
    try:
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False




@contextlib.contextmanager
def ensure_dcp_process_group(
    device: torch.device | None = None,
    *,
    backend: str | None = None,
):
    if is_distributed():
        with contextlib.suppress(Exception):
            init_lane_process_groups(device)
        yield dist.group.WORLD
        return

    if not dist.is_available():
        yield None
        return

    dev = device
    if dev is None:
        with contextlib.suppress(Exception):
            dev = get_device()
    dev = torch.device(dev) if dev is not None else torch.device("cpu")

    be = (backend or "").strip().lower() if backend is not None else ""
    if not be:
        if dev.type == "cuda" and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            be = "cpu:gloo,cuda:nccl"
        elif dev.type == "xpu" and getattr(torch, "xpu", None) is not None and callable(getattr(torch.xpu, "is_available", None)) and torch.xpu.is_available():
            be = "cpu:gloo,xpu:xccl"
        else:
            be = "gloo"

    fd, tmp_path = tempfile.mkstemp(prefix="enn_dcp_pg_", suffix=".tmp")
    os.close(fd)
    init_method = f"file://{tmp_path}"

    dev_id = None
    be_l = str(be).lower()
    if dev.type in {"cuda", "xpu"} and any(x in be_l for x in ("nccl", "xccl", "hccl", "rccl")):
        idx = dev.index
        if idx is None:
            with contextlib.suppress(Exception):
                if dev.type == "cuda" and torch.cuda.is_available():
                    idx = int(torch.cuda.current_device())
                elif dev.type == "xpu" and hasattr(torch, "xpu"):
                    idx = int(torch.xpu.current_device())
        with contextlib.suppress(Exception):
            dev_id = torch.device(dev.type, int(idx or 0))
        if dev_id is None:
            dev_id = int(idx or 0)

    kwargs: dict[str, object] = {
        "backend": be,
        "init_method": init_method,
        "rank": 0,
        "world_size": 1,
    }
    if dev_id is not None:
        kwargs["device_id"] = dev_id

    try:
        try:
            dist.init_process_group(**kwargs)
        except TypeError:
            kwargs.pop("device_id", None)
            dist.init_process_group(**kwargs)
        with contextlib.suppress(Exception):
            _maybe_patch_object_collectives_pickler()
        with contextlib.suppress(Exception):
            init_lane_process_groups(dev)
        yield dist.group.WORLD
    finally:
        with contextlib.suppress(Exception):
            dist.destroy_process_group()
        with contextlib.suppress(Exception):
            os.remove(tmp_path)


def get_rank(default: int | None = None) -> int | None:
    if not is_distributed():
        return default
    try:
        return int(dist.get_rank())
    except Exception:
        return default


def is_rank0() -> bool:
    r = get_rank(default=0)
    return int(r or 0) == 0


def distributed_barrier(
    device: Optional[torch.device] = None,
    group: ProcessGroup | None = None,
    *,
    lane: str = "auto",
) -> None:
    if not is_distributed():
        return

    if (
        group is None
        and device is not None
        and isinstance(device, dist.ProcessGroup)
    ):
        group = device
        device = None

    if (
        group is None
        and device is not None
        and not isinstance(device, torch.device)
    ):
        with contextlib.suppress(Exception):
            from torch.distributed.distributed_c10d import ProcessGroup as _PG

            if isinstance(device, _PG):
                group = device
                device = None

    pg_in = group
    pg = group or dist.group.WORLD
    with contextlib.suppress(Exception):
        if int(dist.get_world_size(pg)) <= 1:
            return

    dev = device
    if dev is None:
        with contextlib.suppress(Exception):
            dev = get_device()
    if dev is None:
        dev = torch.device("cpu")

    lane_s = str(lane).strip().lower()
    if lane_s in {"auto", ""}:
        lane_s = (
            "accelerator"
            if get_accel_group(dev) is not None
            and getattr(dev, "type", "cpu") != "cpu"
            else "control"
        )

    def _accel_pg(default_pg: ProcessGroup) -> ProcessGroup:
        if pg_in is not None:
            return default_pg
        ag = get_accel_group(dev)
        return ag or default_pg

    if lane_s in {"control", "cpu", "gloo"}:
        cpg = (
            get_control_process_group(pg_in)
            if pg_in is not None
            else (get_cpu_group() or get_control_process_group(None))
        )
        if cpg is not None:
            try:
                t = torch.zeros((1,), device="cpu", dtype=torch.int32)
                dist.all_reduce(t, op=dist.ReduceOp.SUM, group=cpg)
                return
            except Exception:
                pass
        if getattr(dev, "type", "cpu") == "cpu" and torch.cuda.is_available():
            with contextlib.suppress(Exception):
                dev = torch.device("cuda", torch.cuda.current_device())
        try:
            dist.barrier(group=_accel_pg(pg), device_ids=_get_device_id(dev))
        except TypeError:
            dist.barrier(group=_accel_pg(pg))
        except Exception as exc:
            raise RuntimeError(
                "distributed_barrier(control) failed on both CPU and accelerator lanes"
            ) from exc
        return

    if getattr(dev, "type", "cpu") == "cpu" and torch.cuda.is_available():
        with contextlib.suppress(Exception):
            dev = torch.device("cuda", torch.cuda.current_device())
    try:
        dist.barrier(group=_accel_pg(pg), device_ids=_get_device_id(dev))
    except TypeError:
        dist.barrier(group=_accel_pg(pg))


def distributed_broadcast(
    target_module: torch.nn.Module,
    *args: Any,
    src_rank: int = 0,
    group: ProcessGroup | None = None,
    include_buffers: bool = True,
    max_buffer_size_mb: int = 25,
    policy: "CollectivePolicy | None" = None,
) -> None:
    from .policies import CollectivePolicy

    if not is_distributed():
        return

    group = group or dist.group.WORLD

    if dist.get_world_size(group) <= 1:
        return

    policy = policy or CollectivePolicy.from_env()

    if include_buffers:
        include_buffers = bool(policy.include_buffers)
    if max_buffer_size_mb == 25:
        max_buffer_size_mb = int(policy.max_buffer_size_mb)

    tensors: list[Tensor] = []

    if policy.include_parameters:
        tensors.extend([p.data for p in target_module.parameters()])

    if include_buffers:
        max_bytes = int(max_buffer_size_mb) * 1024 * 1024
        for b in target_module.buffers():
            if b is None:
                continue

            if b.numel() * b.element_size() > max_bytes:
                continue
            tensors.append(b.data)

    if not tensors:
        return

    coalesce_bytes = max(1, int(policy.coalesce_mb)) * 1024 * 1024
    max_tensor_bytes = (
        max(1, int(policy.max_tensor_mb_for_coalesce)) * 1024 * 1024
    )

    def _is_small(t: Tensor) -> bool:
        return (t.numel() * t.element_size()) <= max_tensor_bytes

    small_tensors = [t for t in tensors if _is_small(t)]
    large_tensors = [t for t in tensors if not _is_small(t)]

    world_size = dist.get_world_size(group)
    local_world_size = env_first_int(
        ["LOCAL_WORLD_SIZE", "SLURM_STEP_NUM_TASKS"], default=world_size
    )
    multi_node = bool(world_size > local_world_size)

    chunk_mb = int(
        policy.inter_stream_mb if multi_node else policy.intra_stream_mb
    )
    max_inflight_mb = int(policy.max_inflight_mb)

    backend = (policy.backend or "c10d").strip().lower()

    if policy.debug_collectives and is_rank0():
        total_bytes = sum(t.numel() * t.element_size() for t in tensors)
        small_bytes = sum(t.numel() * t.element_size() for t in small_tensors)
        large_bytes = total_bytes - small_bytes
        print(
            f"[collectives/broadcast] backend={backend} tensors={len(tensors)} "
            f"small={len(small_tensors)} ({small_bytes / 1024 / 1024:.1f} MiB) "
            f"large={len(large_tensors)} ({large_bytes / 1024 / 1024:.1f} MiB) "
            f"chunk={chunk_mb}MiB inflight={max_inflight_mb}MiB multi_node={multi_node}"
        )

    if backend == "gloox":
        gloo_group = None
        try:
            gloo_group = _get_gloox_gloo_process_group(group)
        except Exception:
            gloo_group = None

        if gloo_group is not None:
            for bucket in _iter_buckets_by_bytes(
                small_tensors, max_bucket_bytes=coalesce_bytes
            ):
                _broadcast_bucket_gloox(
                    bucket, src_rank=src_rank, group=gloo_group
                )

            for t in large_tensors:
                _broadcast_large_tensor_gloox(
                    t,
                    src_rank=src_rank,
                    group=gloo_group,
                    chunk_mb=chunk_mb,
                    max_inflight_mb=max_inflight_mb,
                )

            return

        backend = "c10d"

    if small_tensors:
        element_size = small_tensors[0].element_size()
        coalesce_numel = max(1, coalesce_bytes // element_size)
        dist._broadcast_coalesced(
            group, small_tensors, buffer_size=coalesce_numel, src=src_rank
        )

    for t in large_tensors:
        _broadcast_large_tensor(
            t,
            group=group,
            src_rank=src_rank,
            chunk_mb=chunk_mb,
            max_inflight_mb=max_inflight_mb,
        )


def distributed_sync(
    target_module: nn.Module,
    device: torch.device | None = None,
    group: ProcessGroup | None = None,
    include_buffers: bool = True,
    max_buffer_size_mb: int = 25,
    *args: Any,
    policy: "CollectivePolicy | None" = None,
) -> None:
    if not is_distributed():
        return

    pg = group or _get_default_process_group()
    if pg is None:
        return

    if dist.get_world_size(pg) <= 1:
        return

    device = device or get_distributed_device()

    distributed_broadcast(
        target_module=target_module,
        group=pg,
        include_buffers=include_buffers,
        max_buffer_size_mb=max_buffer_size_mb,
        src_rank=0,
        policy=policy,
    )

    distributed_barrier(group=pg, device=device)


def distributed_all_reduce_grads(
    module: nn.Module,
    *args: Any,
    group: ProcessGroup | None = None,
    average: bool = True,
    policy: "CollectivePolicy | None" = None,
) -> None:
    if not is_distributed():
        return

    pg = group or _get_default_process_group()
    if pg is None:
        return

    world_size = dist.get_world_size(pg)
    if world_size <= 1:
        return

    from .policies import CollectivePolicy

    policy = policy or CollectivePolicy.from_env()

    grads: list[Tensor] = []
    for p in module.parameters():
        g = p.grad
        if g is None:
            continue
        if getattr(g, "is_sparse", False):
            raise NotImplementedError(
                "distributed_all_reduce_grads does not support sparse gradients"
            )
        grads.append(g)

    if not grads:
        return

    local_world_size = env_first_int(
        [
            "LOCAL_WORLD_SIZE",
            "MPI_LOCALNRANKS",
            "SLURM_NTASKS_PER_NODE",
            "OMPI_COMM_WORLD_LOCAL_SIZE",
        ],
        default=world_size,
    )
    multi_node = world_size > local_world_size

    chunk_mb = policy.inter_stream_mb if multi_node else policy.intra_stream_mb
    max_inflight_mb = policy.max_inflight_mb

    backend = str(getattr(policy, "backend", "c10d")).strip().lower()
    if backend == "gloox":
        gloo_pg = None
        try:
            gloo_pg = _get_gloox_gloo_process_group(pg)
        except Exception:
            gloo_pg = None

        if gloo_pg is not None:
            for g in grads:
                _all_reduce_tensor_gloox(
                    g,
                    group=gloo_pg,
                    chunk_mb=chunk_mb,
                    max_inflight_mb=max_inflight_mb,
                    average=average,
                    world_size=world_size,
                )
            return

        backend = "c10d"

    for g in grads:
        if g.numel() == 0:
            continue
        if not g.is_contiguous():
            tmp = g.contiguous()
            dist.all_reduce(tmp, op=dist.ReduceOp.SUM, group=pg)
            if average:
                tmp.div_(world_size)
            g.copy_(tmp)
        else:
            dist.all_reduce(g, op=dist.ReduceOp.SUM, group=pg)
            if average:
                g.div_(world_size)


def is_dtensor_active() -> bool:
    return bool(_DTENSOR_ACTIVE)


def to_hsdp_module(
    module: torch.nn.Module,
    *args: Any,
    mesh: Any | None,
    mp_policy: Any | None = None,
    reshard_after_forward: bool = False,
    sync_module_states: bool = True,
    **user_kwargs: Any,
) -> torch.nn.Module:
    if fully_shard is None:
        raise RuntimeError("Missing fully_shard")
    params = _hsdp_supported_params()
    fsdp_kwargs: dict[str, Any] = dict(user_kwargs)
    pg_obj: Any | None = None
    mesh_obj: Any | None = None
    if mesh is not None:
        _set_dtensor_active()
        with contextlib.suppress(Exception):
            from torch.distributed.distributed_c10d import ProcessGroup

            if isinstance(mesh, ProcessGroup):
                pg_obj = mesh
            else:
                mesh_obj = mesh
        if pg_obj is None and mesh_obj is None:
            mesh_obj = mesh
    defaults: dict[str, Any] = {
        "forward_prefetch": env_bool("ENN_FSDP_FWD_PREFETCH", True),
        "limit_all_gathers": env_bool("ENN_FSDP_LIMIT_AG", True),
        "use_orig_params": env_bool("ENN_FSDP_USE_ORIG_PARAMS", True),
        "mp_policy": mp_policy,
        "reshard_after_forward": reshard_after_forward,
        "sync_module_states": sync_module_states,
    }
    if mesh_obj is not None:
        if "mesh" in params:
            defaults["mesh"] = mesh_obj
        elif "device_mesh" in params:
            defaults["device_mesh"] = mesh_obj

    if pg_obj is not None and "process_group" in params:
        defaults["process_group"] = pg_obj
    fsdp_kwargs.update(
        {
            k: v
            for k, v in defaults.items()
            if k in params and k not in fsdp_kwargs and v is not None
        }
    )
    sharded = fully_shard(module, *args, **fsdp_kwargs)
    with contextlib.suppress(AttributeError):
        sharded.set_requires_gradient_sync(True)
    with contextlib.suppress(ImportError):
        from torch.distributed.fsdp import register_fsdp_forward_method

        for _name in (
            "forward",
            "decode",
            "predict",
            "forward_export",
            "forward_state",
            "forward_stream",
        ):
            if hasattr(sharded, _name):
                register_fsdp_forward_method(sharded, _name)
    return sharded


def get_distributed_mesh(
    device: torch.device | None = None,
) -> tuple[Any | None, str]:
    if not is_distributed():
        return (None, "none")
    dev = device
    if dev is None:
        with contextlib.suppress(Exception):
            dev = get_device()
    dev_type = str(getattr(dev, "type", "cpu"))
    if dev_type not in {"cuda", "xpu"}:
        return (None, "none")
    try:
        world = int(dist.get_world_size())
    except Exception:
        return (None, "none")
    if world <= 1:
        return (None, "none")
    local_world_size = None
    for k in (
        "LOCAL_WORLD_SIZE",
        "MPI_LOCALNRANKS",
        "SLURM_NTASKS_PER_NODE",
        "OMPI_COMM_WORLD_LOCAL_SIZE",
    ):
        if v := os.environ.get(k):
            with contextlib.suppress(ValueError):
                local_world_size = int(v)
                break
    if local_world_size is None:
        if dev_type == "cuda":
            with contextlib.suppress(Exception):
                local_world_size = torch.cuda.device_count()
        elif dev_type == "xpu":
            with contextlib.suppress(Exception):
                local_world_size = int(get_num_accelerators("xpu"))
    local_world_size = int(local_world_size or 1)
    is_consistent = True
    if dist.is_initialized():
        try:
            my_size = torch.tensor(
                [local_world_size], device=dev, dtype=torch.long
            )
            gathered = [torch.zeros_like(my_size) for _ in range(world)]
            dist.all_gather(gathered, my_size)
            all_sizes = [t.item() for t in gathered]
            is_consistent = all(s == local_world_size for s in all_sizes)
        except Exception:
            is_consistent = False
    if importlib.util.find_spec("torch.distributed.device_mesh") is None:
        return (None, "none")
    device_mesh = importlib.import_module("torch.distributed.device_mesh")
    init_device_mesh = getattr(device_mesh, "init_device_mesh", None)
    if init_device_mesh is None:
        return (None, "none")
    if (
        is_consistent
        and world > local_world_size
        and world % local_world_size == 0
    ):
        dp_replicate = world // local_world_size
        dp_shard = local_world_size
        try:
            mesh = init_device_mesh(
                dev_type,
                (dp_replicate, dp_shard),
                mesh_dim_names=("dp_replicate", "dp_shard"),
            )
            return (mesh, "hsdp2")
        except Exception:
            pass
    try:
        mesh = init_device_mesh(
            dev_type,
            (world,),
            mesh_dim_names=("dp",),
        )
        return (mesh, "fsdp2")
    except Exception:
        return (None, "none")


class ProcessBroker:
    DL_STATE_FILE: str = "dataloader.json"

    _IGNORED_WARNING_PATTERNS: tuple[str, ...] = (
        "torch.distributed is disabled, unavailable or uninitialized",
        "torch.distributed is disabled",
        "TypedStorage is deprecated",
        "Found a non-scalar tensor with numel=1 and ndim!=0",
        "distributed_broadcast: coalesced broadcast failed",
        "distributed_broadcast: per-tensor broadcast failed",
        "found no DeviceMesh from dtensor args",
        "mixed precision.*may be unavailable",
        "Either mode or options can be specified, but both can't be specified at the same time\\.",
        "\\s*Online softmax is disabled",
    )
    _IGNORED_WARNING_MESSAGE_RE = re.compile(
        r".*(?:"
        + "|".join((f"(?:{p})" for p in _IGNORED_WARNING_PATTERNS))
        + r").*"
    )

    @classmethod
    def apply_warning_filters(cls) -> None:
        with contextlib.suppress(Exception):
            warnings.filterwarnings(
                "ignore",
                message=cls._IGNORED_WARNING_MESSAGE_RE.pattern,
                category=UserWarning,
            )
        with contextlib.suppress(Exception):
            warnings.filterwarnings(
                "ignore",
                message=r"(?s).*Online softmax is disabled.*",
                category=UserWarning,
            )

    @classmethod
    def clear_process_group(cls) -> None:
        try:
            if dist.is_available() and dist.is_initialized():
                with contextlib.suppress(Exception):
                    dist.barrier()
                with contextlib.suppress(Exception):
                    dist.destroy_process_group()
        except Exception:
            pass
        with contextlib.suppress(Exception):
            _reset_lane_process_groups()
        with contextlib.suppress(Exception):
            _GLOOX_GLOO_PG_CACHE.clear()

    @classmethod
    def set_seed(cls, seed_value: int | None) -> None:
        if seed_value is None:
            return
        try:
            seed_i = int(seed_value)
        except Exception:
            return
        with contextlib.suppress(Exception):
            torch.manual_seed(seed_i)
        with contextlib.suppress(Exception):
            set_accelerator_seed(seed_i)
        with contextlib.suppress(Exception):
            random.seed(seed_i)
        with contextlib.suppress(Exception):
            import numpy

            numpy.random.seed(seed_i)

    @classmethod
    def bootstrap(
        cls,
        *args: Any,
        seed: int | None = None,
        clear_pg: bool = True,
        apply_warning_filters: bool = True,
        **kwargs: Any,
    ) -> None:
        if apply_warning_filters:
            cls.apply_warning_filters()
        if clear_pg:
            cls.clear_process_group()

        init_python_path()
        with contextlib.suppress(Exception):
            _ensure_disk_cache_env()
        with contextlib.suppress(Exception):
            torch.multiprocessing.allow_connection_pickling()
        with contextlib.suppress(Exception):
            init_start_method()
        if seed is not None:
            cls.set_seed(seed)

    @classmethod
    def get_backend_type(cls, device: torch.device) -> str:
        dev_type = str(getattr(device, "type", "cpu")).lower()
        if dev_type == "cuda":
            return "nccl"
        if dev_type == "xpu":
            return "xccl"
        if dev_type in ("cpu", "mps", "dml", "privateuseone"):
            return "gloo"
        if dev_type in ("hpu", "npu"):
            return "hccl"
        if dev_type == "xla":
            return "xla"
        get_default = getattr(
            torch.distributed, "get_default_backend_for_device", None
        )
        if callable(get_default):
            with contextlib.suppress(Exception):
                return str(get_default(device)).lower()
            with contextlib.suppress(Exception):
                return str(get_default(dev_type)).lower()
        return "gloo"

    @classmethod
    def ensure_default_socket_ifname(cls) -> None:
        iface = None
        gloo_if = os.environ.get("GLOO_SOCKET_IFNAME")
        tp_if = os.environ.get("TP_SOCKET_IFNAME")
        if gloo_if or tp_if:
            if gloo_if and (not tp_if):
                os.environ.setdefault("TP_SOCKET_IFNAME", str(gloo_if))
            elif tp_if and (not gloo_if):
                os.environ.setdefault("GLOO_SOCKET_IFNAME", str(tp_if))
            return

        try:
            with open("/proc/net/route", "r", encoding="utf-8") as f:
                for line in f.readlines()[1:]:
                    fields = line.strip().split()
                    if len(fields) >= 2 and fields[1] == "00000000":
                        iface = fields[0]
                        if iface:
                            break
        except Exception:
            iface = None

        if iface is None:
            try:
                import psutil

                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    s.connect(("8.8.8.8", 80))
                    ip = s.getsockname()[0]
                finally:
                    s.close()
                if ip:
                    for name, addrs in psutil.net_if_addrs().items():
                        for a in addrs:
                            if (
                                getattr(a, "family", None) == socket.AF_INET
                                and getattr(a, "address", None) == ip
                            ):
                                iface = str(name)
                                break
                        if iface:
                            break
            except Exception:
                iface = None

        if iface:
            os.environ.setdefault("GLOO_SOCKET_IFNAME", iface)
            os.environ.setdefault("TP_SOCKET_IFNAME", iface)

    @classmethod
    def _configure_torch_nccl_env(cls, device: torch.device) -> None:
        if str(getattr(device, "type", "cpu")) != "cuda":
            return
        world = 1
        with contextlib.suppress(Exception):
            world = int(env_int("WORLD_SIZE", 1) or 1)
        if "TORCH_NCCL_ENABLE_MONITORING" not in os.environ:
            default_mon = 0 if int(world) <= 1 else 1
            mon = int(env_int("ENN_TORCH_NCCL_ENABLE_MONITORING", default_mon))
            os.environ["TORCH_NCCL_ENABLE_MONITORING"] = str(int(mon))
        if "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC" not in os.environ:
            default_hb = 3600 if int(world) <= 1 else 600
            hb = int(
                env_int("ENN_TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", default_hb)
            )
            os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = str(int(hb))
        if "TORCH_NCCL_DUMP_ON_TIMEOUT" not in os.environ:
            default_dump = 0 if int(world) <= 1 else 1
            dump = int(env_int("ENN_TORCH_NCCL_DUMP_ON_TIMEOUT", default_dump))
            os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = str(int(dump))
        if "TORCH_NCCL_ASYNC_ERROR_HANDLING" not in os.environ:
            default_ae = 0 if int(world) <= 1 else 3
            ae = int(
                env_int("ENN_TORCH_NCCL_ASYNC_ERROR_HANDLING", default_ae)
            )
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = str(int(ae))
        if "TORCH_NCCL_BLOCKING_WAIT" not in os.environ:
            default_bw = 1 if int(world) <= 1 else 0
            bw = int(env_int("ENN_TORCH_NCCL_BLOCKING_WAIT", default_bw))
            os.environ["TORCH_NCCL_BLOCKING_WAIT"] = str(int(bw))

    @classmethod
    def configure_torch_nccl_env(cls, device: torch.device) -> None:
        cls._configure_torch_nccl_env(device)

    @classmethod
    def _configure_torch_gloo_env(cls, device: torch.device) -> None:
        cls.ensure_default_socket_ifname()

    @classmethod
    def configure_torch_gloo_env(cls, device: torch.device) -> None:
        cls._configure_torch_gloo_env(device)

    @classmethod
    def _configure_torch_xccl_env(cls, device: torch.device) -> None:
        return

    @classmethod
    def configure_torch_xccl_env(cls, device: torch.device) -> None:
        cls._configure_torch_xccl_env(device)

    @classmethod
    def _coerce_process_group_backend(
        cls, backend: object, device: torch.device
    ) -> object:
        backend_clean: object = backend
        b = str(backend) if backend is not None else ""
        if isinstance(backend, str):
            s = backend.replace("\\n", "").replace("\n", "").replace("\r", "")
            s = s.strip()
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if len(parts) > 1 and all(":" in p for p in parts):
                norm_parts: list[str] = []
                for p in parts:
                    dev_part, _, be_part = p.partition(":")
                    norm_parts.append(
                        f"{dev_part.strip().lower()}:{be_part.strip().lower()}"
                    )
                s = ",".join(norm_parts)
            else:
                s = s.lower()
            backend_clean = s
            b = s
        else:
            b = b.replace("\\n", "").replace("\n", "").replace("\r", "")
            b = b.strip().lower()

        if env_bool("ENN_DISABLE_PG_CPU_BACKEND", False):
            return backend_clean

        dev = str(getattr(device, "type", "cpu")).strip().lower()
        if isinstance(backend_clean, str) and ("," in b and ":" in b):
            return backend_clean
        if b == "nccl" and dev == "cuda":
            return "cpu:gloo,cuda:nccl"
        if b == "xccl" and dev == "xpu":
            return "cpu:gloo,xpu:xccl"
        return backend_clean

    @classmethod
    def configure_backend_env(
        cls, backend: object, device: torch.device
    ) -> None:
        backend_pg = cls._coerce_process_group_backend(backend, device)
        b = str(backend_pg) if backend_pg is not None else ""
        b = b.replace("\\n", "").replace("\n", "").replace("\r", "")
        b = b.lower()
        if "," in b and ":" in b:
            for part in (p.strip() for p in b.split(",") if p.strip()):
                _dev, _, be = part.partition(":")
                if be == "nccl":
                    cls._configure_torch_nccl_env(device)
                elif be == "xccl":
                    cls._configure_torch_xccl_env(device)
                elif be == "gloo":
                    cls._configure_torch_gloo_env(device)
            return
        if b == "nccl":
            cls._configure_torch_nccl_env(device)
        elif b == "xccl":
            cls._configure_torch_xccl_env(device)
        elif b == "gloo":
            cls._configure_torch_gloo_env(device)

    @classmethod
    def init_backend(
        cls, device: torch.device, local_rank: int | None = None
    ) -> None:
        with contextlib.suppress(Exception):
            if device.type == "cuda" and hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = True
        rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
        if local_rank is not None:
            with contextlib.suppress(Exception):
                rank = int(local_rank)
        if device.type in {"cuda", "xpu"}:
            n = max(1, int(get_num_accelerators(device.type) or 1))
            idx = (
                int(getattr(device, "index", None))
                if getattr(device, "index", None) is not None
                else int(rank) % int(n)
            )
            set_accelerator_index(device.type, int(idx))
            with contextlib.suppress(Exception):
                if device.type == "cuda" and hasattr(torch, "cuda"):
                    torch.cuda.set_device(int(idx))
                elif device.type == "xpu" and hasattr(torch, "xpu"):
                    torch.xpu.set_device(int(idx))
        else:
            cls.ensure_default_socket_ifname()

    @classmethod
    def init_process_group(
        cls, backend: object, device: torch.device, local_rank: int
    ) -> None:
        if torch.distributed.is_initialized():
            return
        backend_pg = cls._coerce_process_group_backend(backend, device)
        dev_id = None
        dev_type = getattr(device, "type", "cpu")
        backend_name = str(backend) if backend is not None else ""
        backend_name = (
            backend_name.replace("\\n", "").replace("\n", "").replace("\r", "")
        ).lower()
        if backend_name in ("nccl", "xccl") and dev_type in ("cuda", "xpu"):
            index = (
                device.index
                if getattr(device, "index", None) is not None
                else env_int("LOCAL_RANK", int(local_rank))
            )
            try:
                dev_id = torch.device(dev_type, index)
            except Exception:
                dev_id = index

        timeout = None
        try:
            import datetime

            to_s = int(env_int("ENN_PROCESS_GROUP_TIMEOUT_SEC", 0) or 0)
            if to_s <= 0 and backend_name in ("nccl", "xccl"):
                ws = int(env_int("WORLD_SIZE", 1) or 1)
                if ws <= 1:
                    to_s = 3600
            if int(to_s) > 0:
                timeout = datetime.timedelta(seconds=int(to_s))
        except Exception:
            timeout = None

        def _init_with(bkend: object) -> None:
            kwargs: dict[str, Any] = {"backend": bkend}
            if dev_id is not None:
                kwargs["device_id"] = dev_id
            if timeout is not None:
                kwargs["timeout"] = timeout
            try:
                torch.distributed.init_process_group(**kwargs)
                return
            except TypeError:
                pass
            kwargs.pop("device_id", None)
            try:
                torch.distributed.init_process_group(**kwargs)
                return
            except TypeError:
                pass
            kwargs.pop("timeout", None)
            torch.distributed.init_process_group(**kwargs)

        try:
            _init_with(backend_pg)
        except Exception:
            if str(backend_pg) == str(backend):
                raise
            _init_with(backend)
        with contextlib.suppress(Exception):
            _maybe_patch_object_collectives_pickler()
        with contextlib.suppress(Exception):
            init_lane_process_groups(device)

    @classmethod
    def loader_state_path(cls, directory: PathLike) -> str:
        return os.path.join(os.fspath(directory), cls.DL_STATE_FILE)

    @classmethod
    def get_loader_state(cls, directory: PathLike) -> str:
        return cls.loader_state_path(directory)

    @classmethod
    def _rank0_only(cls) -> bool:
        return is_rank0()

    @classmethod
    def log_rank0(
        cls,
        logger: logging.Logger,
        msg: str,
        *args: Any,
        only_rank0: bool = True,
        level: str = "info",
        **kwargs: Any,
    ) -> None:
        if only_rank0 and not is_rank0():
            return
        try:
            log_fn = getattr(logger, str(level).lower(), logger.info)
        except Exception:
            log_fn = logger.info
        with contextlib.suppress(Exception):
            log_fn(msg, *args)

    @classmethod
    def rank0_logger(
        cls,
        logger: logging.Logger,
        *,
        only_rank0: bool = True,
        level: str = "info",
    ) -> Callable[..., None]:
        def _fn(
            msg: str,
            *args: Any,
            only_main_rank: bool = True,
            **kwargs: Any,
        ) -> None:
            cls.log_rank0(
                logger,
                msg,
                *args,
                only_rank0=bool(only_main_rank) and bool(only_rank0),
                level=level,
            )

        return _fn

    @classmethod
    def make_progress_bar(
        cls,
        *args: Any,
        title: str,
        total: int,
        device: torch.device,
        **kwargs: Any,
    ) -> object:
        if not cls._rank0_only():
            return None
        try:
            from tqdm.auto import tqdm

            fp = sys.stdout
            is_tty = bool(getattr(fp, "isatty", lambda: False)())
            leave = bool(kwargs.get("leave", False))
            if not is_tty:
                total_i = 0
                with contextlib.suppress(Exception):
                    total_i = int(total)
                mi = kwargs.get("mininterval", 1.0)
                return _LineProgress(
                    title=title,
                    total=(total_i if total_i > 0 else None),
                    device=device,
                    file=fp,
                    mininterval=float(mi),
                )

            total_i = 0
            with contextlib.suppress(Exception):
                total_i = int(total)
            unknown_total = int(total_i) <= 0

            if unknown_total:
                return tqdm(
                    total=None,
                    desc=f"{title} ({device.type.upper()}) ",
                    unit="I/O < 0.01 MB/s, COM < 0.01 TFLOPS",
                    bar_format="{desc}{n_fmt} ({unit}) Elapsed: {elapsed}",
                    colour="green",
                    ascii=True,
                    position=int(kwargs.get("position", 0) or 0),
                    leave=leave,
                    file=fp,
                )

            return tqdm(
                total=int(total_i),
                desc=f"{title} ({device.type.upper()}) ",
                unit="I/O < 0.01 MB/s, COM < 0.01 TFLOPS",
                bar_format="{desc}"
                + "{bar} {percentage:3.2f} % "
                + "({unit}) Elapsed: {elapsed}, Remaining: {remaining}",
                colour="green",
                ascii=True,
                position=int(kwargs.get("position", 0) or 0),
                leave=leave,
                file=fp,
            )
        except Exception:
            return None

    @classmethod
    def get_progress_bar(
        cls, title: str, total: int, device: torch.device, **kwargs: Any
    ) -> object:
        return cls.make_progress_bar(
            title=title, total=total, device=device, **kwargs
        )

    @classmethod
    def update_progress_bar(
        cls,
        bar: object,
        finish: bool,
        *args: Any,
        mbps: float | None = None,
        tflops: float | None = None,
        **kwargs: Any,
    ) -> None:
        if bar is None:
            return
        try:
            mbps_val = float(mbps) if mbps is not None else 0.0
        except Exception:
            mbps_val = 0.0
        try:
            tflops_val = float(tflops) if tflops is not None else 0.0
        except Exception:
            tflops_val = 0.0
        io_expr = (
            f"I/O = {mbps_val:.2f} MB/s"
            if mbps_val >= 0.01
            else "I/O < 0.01 MB/s"
        )
        com_expr = (
            f"COM = {tflops_val:.2f} TFLOPS"
            if tflops_val >= 0.01
            else "COM < 0.01 TFLOPS"
        )
        with contextlib.suppress(Exception):
            bar.unit = io_expr + ", " + com_expr
        try:
            inc = int(finish)
        except Exception:
            inc = 1
        if inc > 0:
            with contextlib.suppress(Exception):
                bar.update(inc)


@dataclasses.dataclass
class _PendingOp:
    kind: str
    epoch: int
    future: object | None
    started_monotonic: float
    abort_gen: int = 0
    epoch_dir: str | None = None
    has_optimizer: bool = False
    ok: bool = False


class Checkpointer:
    @staticmethod
    def _filter_kwargs(fn: object, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            sig = inspect.signature(fn)  # type: ignore[arg-type]
            params = getattr(sig, "parameters", None)
            if params:
                with contextlib.suppress(Exception):
                    if any(
                        getattr(p, "kind", None) is inspect.Parameter.VAR_KEYWORD
                        for p in params.values()
                    ):
                        return kwargs
                return {k: v for k, v in kwargs.items() if k in params}
        except Exception:
            pass
        return kwargs

    @staticmethod
    def _call_with_typeerror_kw_fallback(fn: object, call_args: tuple[Any, ...], kwargs: dict[str, Any], drop_order: tuple[str, ...]) -> Any:
        if not callable(fn):
            raise RuntimeError("target function is not callable")
        kw = dict(kwargs)
        try:
            return fn(*call_args, **kw)
        except TypeError as exc:
            msg = str(exc).lower()
            if "unexpected keyword" not in msg:
                raise
            last_exc: TypeError = exc
            for key in drop_order:
                if key not in kw:
                    continue
                kw.pop(key, None)
                try:
                    return fn(*call_args, **kw)
                except TypeError as retry_exc:
                    last_exc = retry_exc
                    if "unexpected keyword" not in str(retry_exc).lower():
                        raise
            raise last_exc

    def _resolve_async_checkpointer_type(self) -> object | None:
        mode = (
            str(os.environ.get("ENN_DCP_ASYNC_CHECKPOINTER_TYPE", "auto") or "auto")
            .strip()
            .lower()
        )
        try:
            from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType
        except Exception:
            try:
                from torch.distributed.checkpoint import AsyncCheckpointerType  # type: ignore
            except Exception:
                return None
        if mode in {"thread", "threads", "t"}:
            return AsyncCheckpointerType.THREAD
        if mode in {"process", "proc", "p"}:
            return AsyncCheckpointerType.PROCESS
        return AsyncCheckpointerType.PROCESS if bool(is_gil_enabled()) else AsyncCheckpointerType.THREAD

    def __init__(
        self,
        ckpt_dir: PathLike,
        *args: Any,
        keep_last: int = 1,
        use_async: bool = True,
        dcp_subdir: str = "dcp_epochs",
        mmap_load: bool | None = None,
        cpu_offload: bool | None = None,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        self.root = Path(ckpt_dir)
        self.dcp_root = self.root / dcp_subdir
        self.keep_last = max(1, int(keep_last))
        self.use_async = bool(use_async)
        self.mmap_load = mmap_load
        self._device = device
        self._cpu_offload = cpu_offload

        self._rank = 0
        self._world = 1
        self._dist = None
        with contextlib.suppress(Exception):
            import torch.distributed as dist

            self._dist = dist
            if dist.is_available() and dist.is_initialized():
                self._rank = int(dist.get_rank())
                self._world = int(dist.get_world_size())

        self.dcp_root.mkdir(parents=True, exist_ok=True)

        self._resp: object | None = None
        self._staging_waited: bool = True
        self._stager: object | None = None

    def is_busy(self) -> bool:
        return self._resp is not None

    def _is_distributed(self) -> bool:
        return bool(
            self._dist is not None
            and self._dist.is_available()
            and self._dist.is_initialized()
            and self._world > 1
        )

    def _epoch_dir(self, epoch: int) -> Path:
        return self.dcp_root / f"epoch_{int(epoch):06d}"

    def _done_file(self, epoch_dir: Path) -> Path:
        return epoch_dir / ".done"

    def _failed_file(self, epoch_dir: Path) -> Path:
        return epoch_dir / ".failed"

    def _atomic_write_text(self, path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    def _atomic_write_json(self, path: Path, obj: dict) -> None:
        self._atomic_write_text(path, json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")

    def _done_payload(self, *, epoch_dir: Path) -> dict:
        return {
            "status": "ok",
            "epoch_dir": epoch_dir.name,
            "created_at_unix": int(time.time()),
            "rank": int(getattr(self, "_rank", 0) or 0),
            "world_size": int(getattr(self, "_world", 1) or 1),
            "torch_version": getattr(torch, "__version__", "unknown"),
        }

    def _finalize_inflight(self, *, success: bool) -> None:
        epoch_dir = getattr(self, "_inflight_epoch_dir", None)
        if isinstance(epoch_dir, Path):
            if success:
                with contextlib.suppress(Exception):
                    ff = self._failed_file(epoch_dir)
                    if ff.exists():
                        ff.unlink()
                with contextlib.suppress(Exception):
                    self._atomic_write_json(self._done_file(epoch_dir), self._done_payload(epoch_dir=epoch_dir))
            else:
                deleted = False
                try:
                    shutil.rmtree(epoch_dir)
                    deleted = True
                except Exception:
                    deleted = False
                if not deleted:
                    with contextlib.suppress(Exception):
                        self._atomic_write_text(self._failed_file(epoch_dir), "failed\n")

        if success and getattr(self, "_rank", 0) == 0:
            with contextlib.suppress(Exception):
                self._cleanup_keep_last()

        self._resp = None
        self._staging_waited = True
        if hasattr(self, "_inflight_epoch_dir"):
            delattr(self, "_inflight_epoch_dir")

    def _ensure_stager(self) -> object:
        if self._stager is None:
            async_type = self._resolve_async_checkpointer_type()
            dev = self._device
            if dev is None:
                with contextlib.suppress(Exception):
                    dev = get_device()
            dev = torch.device(dev) if dev is not None else torch.device("cpu")

            use_async_staging = bool(env_bool("ENN_DCP_USE_ASYNC_STAGING", default=True))

            is_cuda = (getattr(dev, "type", "cpu") == "cuda")
            is_thread = False
            try:
                name = str(getattr(async_type, "name", async_type)).lower()
                is_thread = "thread" in name
            except Exception:
                is_thread = False

            use_pinned = bool(is_cuda and is_thread and env_bool("ENN_DCP_USE_PINNED_MEMORY", default=True))
            use_non_blocking = bool(use_pinned and env_bool("ENN_DCP_USE_NON_BLOCKING_COPY", default=True))

            DefaultStager = None
            StagingOptions = None
            with contextlib.suppress(Exception):
                from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
            if DefaultStager is None:
                with contextlib.suppress(Exception):
                    from torch.distributed.checkpoint import DefaultStager, StagingOptions  # type: ignore
            if DefaultStager is None:
                self._stager = None
                return None

            if StagingOptions is None:
                with contextlib.suppress(Exception):
                    self._stager = DefaultStager()
                return self._stager

            opts = StagingOptions(
                use_pinned_memory=bool(use_pinned),
                use_shared_memory=False,
                use_async_staging=bool(use_async_staging),
                use_non_blocking_copy=bool(use_non_blocking),
            )

            for kw in ({"config": opts}, {"options": opts}):
                try:
                    self._stager = DefaultStager(**kw)
                    break
                except TypeError:
                    continue
                except Exception:
                    break
            if self._stager is None:
                with contextlib.suppress(Exception):
                    self._stager = DefaultStager()
        return self._stager

    def _save_optimizer_default(self) -> bool:
        default = True if self._world > 1 else False
        return bool(
            env_bool(
                ("ENN_DCP_SAVE_OPTIMIZER", "ENN_CKPT_SAVE_OPTIMIZER"),
                default=default,
            )
        )

    def is_idle(self) -> bool:
        return self._resp is None

    def try_request_save_epoch_collective(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        save_optimizer: bool | None = None,
        extra_state: dict[str, Any] | None = None,
        block_if_busy: bool = False,
        device: torch.device | None = None,
        group: object | None = None,
        lane: str = "control",
        **kwargs: Any,
    ) -> bool:
        self.poll()

        if not self._is_distributed():
            return bool(
                self.request_save_epoch(
                    epoch=int(epoch),
                    model=model,
                    optimizer=optimizer,
                    save_optimizer=save_optimizer,
                    extra_state=extra_state,
                    block_if_busy=block_if_busy,
                    **kwargs,
                )
            )

        dist = self._dist
        if dist is None:
            return False

        ready_local = 1 if self.is_idle() else 0

        lane_s = str(lane or "control").strip().lower()
        prefer_accel = lane_s in {"accelerator", "accel", "cuda", "nccl", "xpu", "xccl"}
        prefer_control = lane_s in {"control", "cpu", "gloo", "", "auto"}

        def _resolve_accel_device() -> torch.device:
            dev = device or getattr(self, "_device", None)
            if dev is None:
                dev = get_device()
            dev = torch.device(dev) if dev is not None else torch.device("cpu")
            if dev.type == "cpu":
                with contextlib.suppress(Exception):
                    if hasattr(torch, "cuda") and torch.cuda.is_available():
                        dev = torch.device("cuda", torch.cuda.current_device())
                with contextlib.suppress(Exception):
                    if (
                        dev.type == "cpu"
                        and hasattr(torch, "xpu")
                        and callable(getattr(torch.xpu, "is_available", None))
                        and torch.xpu.is_available()
                    ):
                        dev = torch.device("xpu", torch.xpu.current_device())
            return dev

        pg = None
        tensor_device = torch.device("cpu")

        if group is not None and is_process_group(group):
            pg = group
            be = ""
            with contextlib.suppress(Exception):
                be = str(dist.get_backend(pg)).lower()
            if any(x in be for x in ("nccl", "xccl", "hccl", "rccl")):
                tensor_device = _resolve_accel_device()
            else:
                tensor_device = torch.device("cpu")
        elif prefer_accel and not prefer_control:
            pg = get_accel_process_group(device)
            if pg is not None:
                tensor_device = _resolve_accel_device()
        else:
            if group is not None and is_process_group(group):
                pg = get_control_process_group(group)
            if pg is None:
                pg = get_control_process_group(None)
            if pg is None:
                pg = get_accel_process_group(device)
                if pg is not None:
                    tensor_device = _resolve_accel_device()

        if pg is None:
            pg = dist.group.WORLD
            be = ""
            with contextlib.suppress(Exception):
                be = str(dist.get_backend(pg)).lower()
            if any(x in be for x in ("nccl", "xccl", "hccl", "rccl")):
                tensor_device = _resolve_accel_device()
            else:
                tensor_device = torch.device("cpu")

        if tensor_device.type == "cpu":
            be = ""
            with contextlib.suppress(Exception):
                be = str(dist.get_backend(pg)).lower()
            if any(x in be for x in ("nccl", "xccl", "hccl", "rccl")):
                raise RuntimeError(
                    "collective checkpoint readiness: resolved CPU tensor device for a "
                    f"non-CPU backend ({be}); pass device=... or ensure accelerator is available"
                )

        t_ready = torch.tensor(
            [int(ready_local)], device=tensor_device, dtype=torch.int32
        )
        dist.all_reduce(t_ready, op=dist.ReduceOp.MIN, group=pg)
        if int(t_ready.item()) != 1:
            return False

        started_local = 0
        start_exc: BaseException | None = None
        try:
            started_local = (
                1
                if self.request_save_epoch(
                    epoch=int(epoch),
                    model=model,
                    optimizer=optimizer,
                    save_optimizer=save_optimizer,
                    extra_state=extra_state,
                    block_if_busy=block_if_busy,
                    process_group=pg,
                    use_collectives=True,
                    **kwargs,
                )
                else 0
            )
        except BaseException as exc:
            started_local = 0
            start_exc = exc

        t_started = torch.tensor(
            [int(started_local)], device=tensor_device, dtype=torch.int32
        )
        dist.all_reduce(t_started, op=dist.ReduceOp.MIN, group=pg)
        if int(t_started.item()) != 1:
            with contextlib.suppress(Exception):
                self.close(abort_inflight=True)
            if start_exc is not None:
                raise RuntimeError(
                    "collective checkpoint start failed on at least one rank: "
                    f"{type(start_exc).__name__}: {start_exc}"
                ) from start_exc
            raise RuntimeError(
                "collective checkpoint start failed on at least one rank"
            )

        return True

    def request_save_epoch(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        save_optimizer: bool | None = None,
        extra_state: dict[str, Any] | None = None,
        block_if_busy: bool = False,
        process_group: object | None = None,
        use_collectives: bool | None = None,
        **kwargs: Any,
    ) -> bool:
        self.poll()

        if self._resp is not None:
            if not bool(block_if_busy):
                return False
            success = self._wait_upload()
            self._finalize_inflight(success=bool(success))

        if save_optimizer is None:
            save_optimizer = self._save_optimizer_default()

        return bool(
            self._start_save(
                epoch=int(epoch),
                model=model,
                optimizer=optimizer,
                save_opt=bool(save_optimizer),
                extra_state=extra_state,
                process_group=process_group,
                use_collectives=use_collectives,
            )
        )

    def await_staging(self) -> None:
        if not bool(env_bool("ENN_DCP_AWAIT_STAGING", default=True)):
            self._staging_waited = True
            return
        if self._resp is None:
            return
        if self._staging_waited:
            return
        fut = getattr(self._resp, "staging_completion", None)
        if fut is None:
            self._staging_waited = True
            return
        with contextlib.suppress(Exception):
            _future_result(fut)
        self._staging_waited = True

    def poll(self) -> None:
        if self._resp is not None:
            fut = getattr(self._resp, "upload_completion", None)
            if fut is not None and _future_done(fut):
                strict = bool(env_bool("ENN_DCP_RECIPE_STRICT", default=False))
                success = True
                try:
                    _future_result(fut)
                except Exception:
                    success = False
                    if strict:
                        raise
                self._finalize_inflight(success=success)
        return

    def close(self, *, abort_inflight: bool = True) -> None:
        if abort_inflight:
            self._resp = None
            self._staging_waited = True
            with contextlib.suppress(Exception):
                if hasattr(self, "_inflight_epoch_dir"):
                    delattr(self, "_inflight_epoch_dir")
            return
        success = self._wait_upload()
        self._finalize_inflight(success=bool(success))
        return

    def _wait_upload(self) -> bool:
        if self._resp is None:
            return True
        fut = getattr(self._resp, "upload_completion", None)
        if fut is None:
            return True
        strict = bool(env_bool("ENN_DCP_RECIPE_STRICT", default=False))
        try:
            _future_result(fut)
            return True
        except Exception:
            if strict:
                raise
            return False

    def _start_save(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer | None,
        save_opt: bool,
        extra_state: dict[str, Any] | None,
        process_group: object | None,
        use_collectives: bool | None,
    ) -> bool:
        if self._resp is not None:
            return False
        epoch = int(epoch)
        save_opt = bool(save_opt)

        epoch_dir = self._epoch_dir(epoch)
        epoch_dir.parent.mkdir(parents=True, exist_ok=True)

        from torch.distributed.checkpoint.stateful import Stateful
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

        opts = None
        with contextlib.suppress(Exception):
            from torch.distributed.checkpoint.state_dict import StateDictOptions

            if self._cpu_offload is not None:
                opts = StateDictOptions(cpu_offload=bool(self._cpu_offload))

        class _AppState(Stateful):
            def __init__(self, m: nn.Module, opt: Optimizer | None) -> None:
                self.model = m
                self.optimizer = opt

            def state_dict(self):
                if self.optimizer is None:
                    try:
                        m_sd, o_sd = get_state_dict(self.model, (), options=opts)
                    except TypeError:
                        m_sd, o_sd = get_state_dict(self.model, ())
                    return {"model": m_sd, "optim": o_sd}
                try:
                    m_sd, o_sd = get_state_dict(
                        self.model, (self.optimizer,), options=opts
                    )
                except TypeError:
                    m_sd, o_sd = get_state_dict(self.model, (self.optimizer,))
                return {"model": m_sd, "optim": o_sd}

            def load_state_dict(self, state_dict):
                model_sd = None
                optim_sd = None
                if isinstance(state_dict, dict):
                    model_sd = state_dict.get("model")
                    optim_sd = state_dict.get("optim", None)

                has_optim_state = isinstance(optim_sd, dict) and len(optim_sd) > 0
                if self.optimizer is None or not has_optim_state:
                    set_state_dict(
                        self.model,
                        (),
                        model_state_dict=model_sd,
                        optim_state_dict={},
                    )
                    return

                set_state_dict(
                    self.model,
                    (self.optimizer,),
                    model_state_dict=model_sd,
                    optim_state_dict=optim_sd,
                )
                return

        state: dict[str, Any] = {
            "app": _AppState(model, optimizer if save_opt else None)
        }
        if isinstance(extra_state, dict) and extra_state:
            state["extra"] = extra_state

        import torch.distributed.checkpoint as dcp
        stager = self._ensure_stager()
        storage_writer = None
        planner = None
        with contextlib.suppress(Exception):
            from torch.distributed.checkpoint import DefaultSavePlanner

            planner = DefaultSavePlanner()
        if planner is None:
            with contextlib.suppress(Exception):
                from torch.distributed.checkpoint.planner import DefaultSavePlanner

                planner = DefaultSavePlanner()
        with contextlib.suppress(Exception):
            from torch.distributed.checkpoint import FileSystemWriter

            default_sync = bool(int(getattr(self, "_world", 1) or 1) > 1)
            sync_files = bool(
                env_bool(
                    ("ENN_DCP_SYNC_FILES", "ENN_CKPT_SYNC_FILES"),
                    default=default_sync,
                )
            )
            single_file = bool(
                env_bool(
                    (
                        "ENN_DCP_SINGLE_FILE_PER_RANK",
                        "ENN_CKPT_SINGLE_FILE_PER_RANK",
                    ),
                    default=False,
                )
            )
            writer_threads = int(
                env_int(
                    ("ENN_DCP_WRITER_THREADS", "ENN_CKPT_WRITER_THREADS"),
                    default=1,
                )
                or 1
            )
            writer_threads = max(1, min(64, int(writer_threads)))
            storage_writer = FileSystemWriter(
                str(epoch_dir),
                single_file_per_rank=single_file,
                sync_files=sync_files,
                thread_count=int(writer_threads),
                overwrite=True,
            )

        if use_collectives is None:
            use_collectives = bool(int(getattr(self, "_world", 1) or 1) > 1)

        dev = getattr(self, "_device", None)
        if dev is None:
            with contextlib.suppress(Exception):
                dev = get_device()
        dev = torch.device(dev) if dev is not None else torch.device("cpu")

        pg_arg: object | None = process_group if (process_group is not None and is_process_group(process_group)) else None
        if pg_arg is None and bool(use_collectives):
            lane_env = (
                os.environ.get("ENN_DCP_DEFAULT_LANE")
                or os.environ.get("ENN_CKPT_DEFAULT_LANE")
                or os.environ.get("ENN_DCP_LANE")
                or os.environ.get("ENN_CKPT_LANE")
                or "control"
            )
            lane_s = str(lane_env).strip().lower()
            with contextlib.suppress(Exception):
                init_lane_process_groups(dev)

            accel_lanes = {"accelerator", "accel", "cuda", "nccl", "xpu", "xccl", "hpu", "hccl", "npu"}
            control_lanes = {"control", "cpu", "gloo"}
            auto_lanes = {"auto", ""}

            if lane_s in accel_lanes:
                pg_arg = get_accel_process_group(dev)
            elif lane_s in auto_lanes:
                if getattr(dev, "type", "cpu") != "cpu":
                    pg_arg = get_accel_process_group(dev)
                if pg_arg is None:
                    pg_arg = get_control_process_group(None)
            elif lane_s in control_lanes:
                pg_arg = get_control_process_group(None)
            else:
                pg_arg = get_control_process_group(None)

            if pg_arg is None:
                with contextlib.suppress(Exception):
                    pg_arg = dist.group.WORLD

        if not self.use_async:
            fn_save = getattr(dcp, "save", None)
            if not callable(fn_save):
                raise RuntimeError("torch.distributed.checkpoint.save is not available")
            save_kw: dict[str, Any] = {
                "checkpoint_id": str(epoch_dir),
                "storage_writer": storage_writer,
                "planner": planner,
                "process_group": pg_arg,
                "use_collectives": bool(use_collectives),
            }
            save_kw = {k: v for k, v in save_kw.items() if v is not None or k in {"use_collectives"}}
            save_kw = self._filter_kwargs(fn_save, save_kw)
            self._call_with_typeerror_kw_fallback(
                fn_save,
                (state,),
                save_kw,
                ("planner", "process_group", "use_collectives"),
            )
            if self._rank == 0:
                with contextlib.suppress(Exception):
                    self._done_file(epoch_dir).write_text("ok\\n", encoding="utf-8")
                self._cleanup_keep_last()
            return True

        async_type = self._resolve_async_checkpointer_type()
        call_kw: dict[str, Any] = {
            "checkpoint_id": str(epoch_dir),
            "storage_writer": storage_writer,
            "planner": planner,
            "async_stager": stager,
            "async_checkpointer_type": async_type,
            "process_group": pg_arg,
            "use_collectives": bool(use_collectives),
        }
        call_kw = {k: v for k, v in call_kw.items() if v is not None or k in {"use_collectives"}}
        fn = getattr(dcp, "async_save", None)
        if not callable(fn):
            raise RuntimeError("torch.distributed.checkpoint.async_save is not available")
        call_kw = self._filter_kwargs(fn, call_kw)
        self._resp = fn(state, **call_kw)
        self._staging_waited = False
        setattr(self, "_inflight_epoch_dir", epoch_dir)
        return True

    def _cleanup_keep_last(self) -> None:
        try:
            epochs: list[int] = []
            for p in self.dcp_root.glob("epoch_*"):
                if not p.is_dir():
                    continue
                m = re.match(r"epoch_(\d+)", p.name)
                if not m:
                    continue
                if not self._done_file(p).is_file():
                    continue
                epochs.append(int(m.group(1)))
            epochs.sort()
            if len(epochs) <= self.keep_last:
                return
            to_delete = epochs[: max(0, len(epochs) - self.keep_last)]
            for e in to_delete:
                _safe_rmtree(self._epoch_dir(e))
        except Exception:
            return
