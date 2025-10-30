# -*- coding: utf-8 -*-


from __future__ import annotations

import math
import os
import socket
import sys
import time
import warnings
from collections import deque
from contextlib import nullcontext, suppress
from functools import partial
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple, Union

import ctypes
import importlib
import itertools
import threading
from threading import Lock

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d
from torchdata.nodes import BaseNode, IterableWrapper, Loader, ParallelMapper, PinMemory, Prefetcher

from ..distributed import Endpoint
from ..utils.compat import has_arrow_flight, patch_arrow
from ..utils.datatype import to, to_torch
from ..utils.platform import Distributed, Network, System

_ARROW = patch_arrow()


def identity(item: Any) -> Any:
    return item


class ThreadLoadBalancer:
    """Best-effort thread affinity + OpenMP proc_bind(spread) + dynamic PyTorch thread tuning.
       If nothing is effective, transparently falls back to passthrough (no changes)."""

    __slots__ = (
        "_psutil",
        "_allowed_cpus",
        "_ring",
        "_tls",
        "_lock",
        "_io_workers",
        "_samples",
        "_cpu_ns",
        "_wall_ns",
        "_last_retune_ts",
        "_enabled",
        "_pin_attempts",
        "_pin_success",
        "_omp_ok",
    )

    def __init__(self, io_workers: int) -> None:
        self._psutil = self._try_import_psutil()
        self._allowed_cpus = self._get_allowed_cpus_psutil_first()
        if not self._allowed_cpus:
            self._allowed_cpus = list(range(max(1, os.cpu_count() or 1)))
        self._ring = itertools.cycle(self._allowed_cpus)
        self._tls = threading.local()
        self._lock = Lock()
        self._io_workers = max(1, int(io_workers))
        self._samples = 0
        self._cpu_ns = 0
        self._wall_ns = 0
        self._last_retune_ts = time.perf_counter()
        self._pin_attempts = 0
        self._pin_success = 0
        self._omp_ok = self._omp_try_set_spread_best_effort()
        self._enabled = (len(self._allowed_cpus) >= 2) or self._omp_ok
        self._pre_tune(io_workers)

    @staticmethod
    def _try_import_psutil():
        try:
            return importlib.import_module("psutil")
        except Exception:
            return None

    def _get_allowed_cpus_psutil_first(self) -> list[int]:
        if self._psutil is not None:
            try:
                proc = self._psutil.Process()
                if hasattr(proc, "cpu_affinity"):
                    cpus = proc.cpu_affinity()
                    if cpus:
                        return sorted({int(c) for c in cpus})
            except Exception:
                pass
        if os.name == "nt":
            try:
                k32 = ctypes.windll.kernel32
                k32.GetActiveProcessorGroupCount.restype = ctypes.c_ushort
                k32.GetActiveProcessorCount.argtypes = [ctypes.c_ushort]
                k32.GetActiveProcessorCount.restype = ctypes.c_ushort
                group_count = int(k32.GetActiveProcessorGroupCount())
                counts = [int(k32.GetActiveProcessorCount(i)) for i in range(max(1, group_count))]
                groups = list(range(group_count))
                try:
                    GetCurrentProcess = k32.GetCurrentProcess
                    GetProcessGroupAffinity = getattr(k32, "GetProcessGroupAffinity", None)
                    if GetProcessGroupAffinity:
                        handle = GetCurrentProcess()
                        arr_type = ctypes.c_ushort * max(1, group_count)
                        arr = arr_type()
                        needed = ctypes.c_ushort(group_count)
                        if GetProcessGroupAffinity(handle, ctypes.byref(needed), arr):
                            groups = list(arr)[: int(needed.value)]
                except Exception:
                    pass
                flattened: list[int] = []
                base = 0
                for group_index in range(group_count):
                    count = counts[group_index]
                    if group_index in groups:
                        flattened.extend(range(base, base + count))
                    base += count
                if flattened:
                    return flattened
            except Exception:
                pass
        try:
            return sorted(int(c) for c in os.sched_getaffinity(0))  # type: ignore[attr-defined]
        except Exception:
            pass
        return list(range(max(1, os.cpu_count() or 1)))

    @staticmethod
    def _omp_try_set_spread_best_effort() -> bool:
        candidates: list[str] = []
        plat = sys.platform
        if plat.startswith("linux"):
            candidates = ["libgomp.so.1", "libgomp.so", "libiomp5.so", "libomp.so"]
        elif plat == "darwin":
            candidates = ["libomp.dylib", "libiomp5.dylib"]
        elif os.name == "nt":
            candidates = ["libiomp5md.dll", "vcomp140.dll"]
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

    def _next_core(self) -> int:
        with self._lock:
            return int(next(self._ring))

    @staticmethod
    def _pin_windows(core: int) -> bool:
        try:
            k32 = ctypes.windll.kernel32
            GetActiveProcessorGroupCount = k32.GetActiveProcessorGroupCount
            GetActiveProcessorCount = k32.GetActiveProcessorCount
            GetCurrentThread = k32.GetCurrentThread
            SetThreadAffinityMask = k32.SetThreadAffinityMask
            SetThreadGroupAffinity = k32.SetThreadGroupAffinity
            SetThreadIdealProcessorEx = getattr(k32, "SetThreadIdealProcessorEx", None)

            GetActiveProcessorGroupCount.restype = ctypes.c_ushort
            GetActiveProcessorCount.argtypes = [ctypes.c_ushort]
            GetActiveProcessorCount.restype = ctypes.c_ushort
            group_count = int(GetActiveProcessorGroupCount())
            counts = [int(GetActiveProcessorCount(i)) for i in range(max(1, group_count))]
            total = sum(counts) or (os.cpu_count() or 1)
            idx = int(core) % max(1, total)
            group = 0
            within = idx
            for gid, cnt in enumerate(counts):
                if within < cnt:
                    group = gid
                    break
                within -= cnt
            thread_handle = GetCurrentThread()
            if group_count <= 1:
                mask = ctypes.c_size_t(1 << within)
                prev = SetThreadAffinityMask(thread_handle, mask.value)
                return bool(prev)

            class GROUP_AFFINITY(ctypes.Structure):
                _fields_ = [
                    ("Mask", ctypes.c_ulonglong),
                    ("Group", ctypes.c_ushort),
                    ("Reserved", ctypes.c_ushort * 3),
                ]

            affinity = GROUP_AFFINITY(ctypes.c_ulonglong(1 << within), ctypes.c_ushort(group), (ctypes.c_ushort * 3)(0, 0, 0))
            ok = SetThreadGroupAffinity(thread_handle, ctypes.byref(affinity), None)
            if ok and SetThreadIdealProcessorEx is not None:
                try:
                    class PROCESSOR_NUMBER(ctypes.Structure):
                        _fields_ = [
                            ("Group", ctypes.c_ushort),
                            ("Number", ctypes.c_ubyte),
                            ("Reserved", ctypes.c_ubyte),
                        ]

                    proc_num = PROCESSOR_NUMBER(group, within, 0)
                    SetThreadIdealProcessorEx(thread_handle, ctypes.byref(proc_num), None)
                except Exception:
                    pass
            return bool(ok)
        except Exception:
            return False

    @staticmethod
    def _pin_linux_like(core: int) -> bool:
        try:
            tid = threading.get_native_id()
            os.sched_setaffinity(tid, {int(core)})  # type: ignore[attr-defined]
            return True
        except Exception:
            try:
                os.sched_setaffinity(0, {int(core)})  # type: ignore[attr-defined]
                return True
            except Exception:
                return False

    @staticmethod
    def _pin_bsd(core: int) -> bool:
        # BSD 계열은 표준 per-thread API가 제각각이라 안전하게 no-op 처리.
        return False

    def _pin_once(self) -> None:
        if not self._enabled:
            return
        attempts = getattr(self._tls, "attempts", 0)
        if getattr(self._tls, "pinned", False) or attempts >= 4:
            return
        self._tls.attempts = attempts + 1
        core = self._next_core()
        ok = False
        if os.name == "nt":
            ok = self._pin_windows(core)
        else:
            plat = sys.platform
            if plat.startswith("linux"):
                ok = self._pin_linux_like(core)
            elif "bsd" in plat:
                ok = self._pin_bsd(core)
            elif plat == "darwin":
                try:
                    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
                    THREAD_AFFINITY_POLICY = 4

                    class thread_affinity_policy_data_t(ctypes.Structure):
                        _fields_ = [("affinity_tag", ctypes.c_int)]

                    policy = thread_affinity_policy_data_t(int(core) + 1)
                    lib.mach_thread_self.restype = ctypes.c_uint
                    lib.thread_policy_set.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
                    port = lib.mach_thread_self()
                    result = lib.thread_policy_set(port, THREAD_AFFINITY_POLICY, ctypes.byref(policy), 1)
                    ok = result == 0
                except Exception:
                    ok = False
        self._tls.pinned = bool(ok)
        self._pin_attempts += 1
        if ok:
            self._pin_success += 1
        if self._pin_attempts >= 16 and self._pin_success == 0 and not self._omp_ok:
            self._enabled = False

    @staticmethod
    def _set_torch_threads(intra: Optional[int] = None, inter: Optional[int] = None) -> None:
        if intra is not None:
            try:
                torch.set_num_threads(max(1, int(intra)))
            except Exception:
                pass
        if inter is not None and hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(max(1, int(inter)))
            except Exception:
                pass

    def _pre_tune(self, io_workers: int) -> None:
        if not self._enabled:
            return
        cpus = max(1, len(self._allowed_cpus))
        tuned_workers = max(1, min(int(io_workers), cpus))
        self._io_workers = tuned_workers
        try:
            intra = int(torch.get_num_threads())
        except Exception:
            intra = cpus
        if intra * tuned_workers > cpus:
            new_intra = max(1, cpus // tuned_workers)
            self._set_torch_threads(intra=new_intra)
        want_inter = max(1, min(tuned_workers // 2, 4))
        self._set_torch_threads(inter=want_inter)

    def _retune_if_needed(self) -> None:
        if not self._enabled:
            return
        if self._samples < 128:
            return
        now = time.perf_counter()
        if (now - self._last_retune_ts) < 1.0:
            return
        with self._lock:
            cpu_ns = self._cpu_ns
            wall_ns = self._wall_ns
            self._cpu_ns = 0
            self._wall_ns = 0
            self._samples = 0
        self._last_retune_ts = now
        if wall_ns <= 0:
            return
        cpu_ratio = cpu_ns / float(wall_ns)
        cpus = max(1, len(self._allowed_cpus))
        workers = max(1, self._io_workers)
        if cpu_ratio >= 0.5:
            target_intra = max(1, cpus // workers)
            if cpus >= 8:
                target_intra = min(target_intra, 2)
            self._set_torch_threads(intra=target_intra)
            self._set_torch_threads(inter=max(1, min(2, workers)))
        else:
            relaxed = min(4, max(1, cpus // max(1, workers // 2)))
            current = max(1, torch.get_num_threads())
            if current < relaxed:
                self._set_torch_threads(intra=relaxed)

    def wrap_map(self, fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        if not self._enabled:
            return fn

        def _inner(x: Any) -> Any:
            self._pin_once()
            t0 = time.perf_counter_ns()
            thread_time = getattr(time, "thread_time_ns", None)
            tc0 = thread_time() if callable(thread_time) else 0
            y = fn(x)
            tc1 = thread_time() if callable(thread_time) else 0
            t1 = time.perf_counter_ns()
            with self._lock:
                self._samples += 1
                self._cpu_ns += max(0, int(tc1) - int(tc0))
                self._wall_ns += max(0, int(t1) - int(t0))
            self._retune_if_needed()
            return y

        return _inner

    def tune_workers(self, io_workers: int) -> int:
        if not self._enabled:
            return int(io_workers)
        cpus = max(1, len(self._allowed_cpus))
        tuned = max(1, min(int(io_workers), cpus))
        self._io_workers = tuned
        return tuned




def _convert_mapping_to_batch(
    batch: Mapping[str, Any],
    *,
    flatten_features: bool,
    labels_dtype: Optional[torch.dtype],
    sanitize: bool,
) -> Dict[str, Any]:
    features = batch["X"]
    labels = batch["Y"]
    if (
        flatten_features
        and isinstance(features, torch.Tensor)
        and (features.dim() >= 2)
    ):
        features = features.flatten(start_dim=1)
    labels_tensor = to_torch(labels)
    if labels_dtype is not None and getattr(labels_tensor, "dtype", None) != labels_dtype:
        labels_tensor = labels_tensor.to(dtype=labels_dtype)
    if sanitize and torch.is_floating_point(labels_tensor):
        labels_tensor = torch.nan_to_num(
            labels_tensor, nan=0.0, posinf=0.0, neginf=0.0
        )
    return {"X": features, "Y": labels_tensor}


def _world_info() -> Tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", os.environ.get("PMI_RANK", "0")))
    local_rank = int(
        os.environ.get(
            "LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")
        )
    )
    world_env = os.environ.get("WORLD_SIZE", os.environ.get("PMI_SIZE"))
    world: Optional[int] = None
    if world_env is not None:
        with suppress(ValueError):
            world = int(world_env)
    if world is None:
        with suppress(Exception):
            world = int(Distributed.get_world_size())
    world = world or 1
    return (rank, local_rank, world, world > 1)


def _negotiate() -> Tuple[int, bool]:
    if not (dist.is_available() and dist.is_initialized()):
        return (0, True)
    world = Distributed.get_world_size()
    rank = dist.get_rank()
    hostname = socket.gethostname()
    hosts: list[str] = ["" for _ in range(world)]
    dist.all_gather_object(hosts, hostname)
    peers = [index for index, value in enumerate(hosts) if value == hostname]
    leader_rank = min(peers) if peers else rank
    ordered = list(dict.fromkeys(sorted(hosts)))
    node_id = ordered.index(hostname) if hostname in ordered else 0
    return (node_id, rank == leader_rank)


def _new_flight(split: str, name: str, uri: str) -> Tuple[str, str]:
    if not (dist.is_available() and dist.is_initialized()):
        return (name, uri)
    node_id, is_leader = _negotiate()
    store = distributed_c10d._get_default_store()
    key = f"flight/endpoints/{node_id}/{split}".encode("utf-8")
    if is_leader:
        store.set(key, f"{name}|{uri}".encode("utf-8"))
    dist.barrier()
    value = store.get(key).decode("utf-8")
    published_name, published_uri = value.split("|", 1)
    return (published_name, published_uri)


def _wrap_data_node(
    node: IterableWrapper,
    *,
    device: torch.device,
    threads: Dict[str, int],
    prefetch_factor: int,
    non_blocking_copy: bool,
    map_fn: Callable[[Any], Any],
    length: Optional[int] = None,
) -> "DataLoader":
    io_workers = max(1, int(threads["dataloader_workers"]))
    prebatch = max(1, int(threads["prefetch_factor"]))
    cpu_total = os.cpu_count() or io_workers
    cpu_budget = max(1, cpu_total - 1) if cpu_total > 1 else 1
    proc_target = max(io_workers, 2)
    proc_workers = max(1, min(cpu_budget, proc_target))

    load_balancer = ThreadLoadBalancer(io_workers)
    io_workers = load_balancer.tune_workers(io_workers)
    map_fn = load_balancer.wrap_map(map_fn)

    thread_max_concurrent = io_workers
    wrapped: BaseNode = ParallelMapper(
        node,
        map_fn=map_fn,
        num_workers=io_workers,
        in_order=False,
        method="thread",
        max_concurrent=thread_max_concurrent,
        prebatch=prebatch,
    )
    wrapped = Prefetcher(wrapped, prefetch_factor=prefetch_factor)
    if device.type in {"cuda", "xpu", "mps"}:
        wrapped = PinMemory(wrapped, pin_memory_device=device.type)
    return DataLoader(
        device=device,
        node=wrapped,
        prefetch_factor=prefetch_factor,
        non_blocking=bool(non_blocking_copy),
        length=length,
    )


def _build_local_loaders(
    memmap_dir: str,
    batch_size: int,
    val_frac: float,
    *,
    device: torch.device,
    threads: Dict[str, int],
    prefetch_factor: int,
    non_blocking_copy: bool,
    map_fn: Callable[[Any], Any],
    batch_reader_cls: type,
    sample_reader_cls: type,
) -> Tuple[Any, Optional[Any], _Keep]:
    reader_tr = sample_reader_cls.from_dir(
        memmap_dir,
        split="train",
        batch_size=int(batch_size),
        val_frac=val_frac,
    )
    meta = reader_tr._load_meta()
    total = int(meta.get("N", 0))
    train_range = reader_tr._indices()
    train_start = int(getattr(train_range, "start", 0))
    train_end = int(getattr(train_range, "stop", total if total else 0))
    if train_end <= train_start and total:
        train_end = total
    keep = _Keep(reader_tr)
    batcher_tr = batch_reader_cls(reader_tr, train_start, train_end, int(batch_size))
    node_tr = IterableWrapper(batcher_tr)
    train_loader = _wrap_data_node(
        node_tr,
        device=device,
        threads=threads,
        prefetch_factor=prefetch_factor,
        non_blocking_copy=non_blocking_copy,
        map_fn=map_fn,
        length=len(batcher_tr),
    )
    val_loader: Optional[Any] = None
    if val_frac > 0 and train_end < total:
        reader_vl = sample_reader_cls.from_dir(
            memmap_dir,
            split="val",
            batch_size=int(batch_size),
            val_frac=val_frac,
        )
        keep.add(reader_vl)
        val_range = reader_vl._indices()
        val_start = int(getattr(val_range, "start", train_end))
        val_end = int(getattr(val_range, "stop", total))
        if val_end <= val_start:
            val_end = total
        batcher_vl = batch_reader_cls(reader_vl, val_start, val_end, int(batch_size))
        node_vl = IterableWrapper(batcher_vl)
        val_loader = _wrap_data_node(
            node_vl,
            device=device,
            threads=threads,
            prefetch_factor=prefetch_factor,
            non_blocking_copy=non_blocking_copy,
            map_fn=map_fn,
            length=len(batcher_vl),
        )
    return (train_loader, val_loader, keep)


def _iterate_flight_batches(
    client: Endpoint.Client,
    name: str,
    *,
    features_np_dtype: Any,
    labels_np_dtype: Any,
    label_shape: Sequence[int],
) -> Iterator[Dict[str, torch.Tensor]]:
    reader = client.reader(name, timeout_s=120.0)
    for record_batch in reader:
        batch_size_rb = int(getattr(record_batch, "num_rows", 0)) or int(
            len(record_batch.column(0))
        )
        features_arr = record_batch.column(0)
        features_np = _ARROW.to_numpy(features_arr, zero_copy_only=False)
        if isinstance(features_np, np.ndarray) and features_np.dtype == object:
            features_np = np.array(features_arr.to_pylist(), dtype=np.float64)
        if not isinstance(features_np, np.ndarray):
            features_np = np.array(features_np, copy=True)
        if (
            getattr(features_np, "flags", None) is None
            or not features_np.flags.writeable
            or (not features_np.flags.c_contiguous)
        ):
            features_np = np.ascontiguousarray(features_np).copy()
        if features_np.dtype != features_np_dtype:
            features_np = features_np.astype(features_np_dtype, copy=False)
        features_tensor = torch.from_numpy(features_np).view(batch_size_rb, -1)

        labels_arr = record_batch.column(1)
        labels_np = _ARROW.to_numpy(labels_arr, zero_copy_only=False)
        if isinstance(labels_np, np.ndarray) and labels_np.dtype == object:
            labels_np = np.array(labels_arr.to_pylist(), dtype=np.float64)
        if not isinstance(labels_np, np.ndarray):
            labels_np = np.array(labels_np, copy=True)
        if (
            getattr(labels_np, "flags", None) is None
            or not labels_np.flags.writeable
            or (not labels_np.flags.c_contiguous)
        ):
            labels_np = np.ascontiguousarray(labels_np).copy()
        if labels_np.dtype != labels_np_dtype:
            labels_np = labels_np.astype(labels_np_dtype, copy=False)
        labels_tensor = torch.from_numpy(labels_np)
        if label_shape:
            labels_tensor = labels_tensor.reshape(batch_size_rb, -1).view(
                batch_size_rb, *label_shape
            )
        else:
            labels_tensor = labels_tensor.reshape(batch_size_rb, -1)
        yield {"X": features_tensor, "Y": labels_tensor}


def _build_flight_loaders(
    memmap_dir: str,
    batch_size: int,
    val_frac: float,
    *,
    device: torch.device,
    threads: Dict[str, int],
    prefetch_factor: int,
    non_blocking_copy: bool,
    map_fn: Callable[[Any], Any],
    batch_reader_cls: type,
    sample_reader_cls: type,
) -> Tuple[Any, Optional[Any], _Keep]:
    rank, local_rank, _, is_ddp = _world_info()
    name_train = Endpoint.resource_key(memmap_dir, "train")
    name_val = Endpoint.resource_key(memmap_dir, "val")
    reader_tr = sample_reader_cls.from_dir(
        memmap_dir,
        split="train",
        batch_size=int(batch_size),
        val_frac=val_frac,
    )
    reader_vl = (
        sample_reader_cls.from_dir(
            memmap_dir,
            split="val",
            batch_size=int(batch_size),
            val_frac=val_frac,
        )
        if val_frac > 0
        else None
    )
    meta = reader_tr._load_meta()
    keep = _Keep(reader_tr, reader_vl)
    features_np_dtype = to(meta.get("features_arrow_dtype", "float64"), "numpy")
    labels_np_dtype = to(meta.get("labels_arrow_dtype", "float64"), "numpy")
    try:
        loopback_host = Network.get_preferred_ip(
            "localhost", allow_loopback=True
        )
        host = "0.0.0.0" if is_ddp else (loopback_host or "127.0.0.1")
        uri_value = ""
        if not is_ddp or local_rank == 0:
            server, uri_value = Endpoint.start_server_standby(host=host, port=0)
            keep.add(server)
            Endpoint.reg_mmt_dataset(
                server, name_train, reader_tr, int(batch_size), "train"
            )
            if reader_vl is not None:
                Endpoint.reg_mmt_dataset(
                    server, name_val, reader_vl, int(batch_size), "val"
                )
            name_train, uri_value = _new_flight("train", name_train, uri_value)
            if reader_vl is not None:
                name_val, uri_value = _new_flight("val", name_val, uri_value)
        else:
            name_train, uri_value = _new_flight("train", name_train, "")
            if reader_vl is not None:
                name_val, uri_value = _new_flight("val", name_val, uri_value)
        client = Endpoint.Client(uri_value)
        keep.add(client)
        label_shape = list(meta.get("label_shape", []))

        train_indices = reader_tr._indices()
        train_start = int(getattr(train_indices, "start", 0))
        train_stop = int(getattr(train_indices, "stop", 0))
        train_steps = max(
            1,
            math.ceil(max(0, train_stop - train_start) / float(max(1, batch_size))),
        )
        node_tr = IterableWrapper(
            _iterate_flight_batches(
                client,
                name_train,
                features_np_dtype=features_np_dtype,
                labels_np_dtype=labels_np_dtype,
                label_shape=label_shape,
            )
        )
        train_loader = _wrap_data_node(
            node_tr,
            device=device,
            threads=threads,
            prefetch_factor=prefetch_factor,
            non_blocking_copy=non_blocking_copy,
            map_fn=map_fn,
            length=train_steps,
        )
        if reader_vl is not None:
            val_indices = reader_vl._indices()
            val_start = int(getattr(val_indices, "start", 0))
            val_stop = int(getattr(val_indices, "stop", 0))
            val_steps = max(
                1,
                math.ceil(max(0, val_stop - val_start) / float(max(1, batch_size))),
            )
            node_vl = IterableWrapper(
                _iterate_flight_batches(
                    client,
                    name_val,
                    features_np_dtype=features_np_dtype,
                    labels_np_dtype=labels_np_dtype,
                    label_shape=label_shape,
                )
            )
            val_loader = _wrap_data_node(
                node_vl,
                device=device,
                threads=threads,
                prefetch_factor=prefetch_factor,
                non_blocking_copy=non_blocking_copy,
                map_fn=map_fn,
                length=val_steps,
            )
        else:
            val_loader = None
        return (train_loader, val_loader, keep)
    except Exception:
        keep.cleanup()
        raise


class DevicePrefetcher(Iterator[Any]):
    def __init__(
        self,
        iterable: Iterable[Any],
        device: Optional[Union[str, torch.device]],
        *args: Any,
        depth: int = 2,
        slots: int = 2,
        pin_if_needed: bool = True,
        use_record_stream: bool = True,
        amp_dtype: Optional[torch.dtype] = None,
        max_bytes: Optional[int] = None,
        autotune: bool = True,
        tune_interval: int = 50,
        depth_min: int = 1,
        depth_max: int = 8,
        enable_graphs: bool = False,
        graph_warmup: int = 2,
        **kwargs: Any,
    ) -> None:
        self._it = iter(iterable)
        self._dev = (
            torch.device(device)
            if not isinstance(device, torch.device)
            else device
        )
        self._backend = self._dev.type
        self._depth_min = int(max(1, depth_min))
        self._depth_max = int(max(self._depth_min, depth_max))
        self._queue: Deque[Tuple[Any, int]] = deque(maxlen=max(1, int(depth)))
        self._pin_if_needed = bool(pin_if_needed)
        self._use_record_stream = bool(use_record_stream)
        self._amp_dtype = amp_dtype
        self._max_bytes = (
            max_bytes if isinstance(max_bytes, int) and max_bytes > 0 else None
        )
        self._bytes_in_q = 0
        self._autotune = bool(autotune)
        self._tune_interval = max(10, int(tune_interval))
        self._graph_enabled = bool(enable_graphs and self._backend == "cuda")
        self._slots = 1 if self._graph_enabled else max(1, int(slots))
        self._streams: list[Any] = []
        self._streams = self._init_streams()
        self._rr = 0
        self._copy_time_acc = 0.0
        self._steps = 0
        self._starved = 0
        self._last_yield_ts: Optional[float] = None
        self._ema_compute = 0.0
        self._static: Any = None
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._graph_warmup = max(1, int(graph_warmup))
        self._err: Optional[BaseException] = None
        self._preload()

    def _init_streams(self) -> list[Any]:
        try:
            if self._backend == "cuda":
                return [
                    torch.cuda.Stream(device=self._dev)
                    for _ in range(self._slots)
                ]
            if self._backend == "xpu" and hasattr(torch, "xpu"):
                stream_type = getattr(torch.xpu, "Stream", None)
                if stream_type is not None:
                    return [stream_type() for _ in range(self._slots)]
            if self._backend == "mps" and hasattr(torch, "mps"):
                stream_type = getattr(torch.mps, "Stream", None)
                if stream_type is not None:
                    return [stream_type() for _ in range(self._slots)]
        except Exception:
            return []
        return []

    def _map(self, obj: Any, fn: Callable[[Any], Any]) -> Any:
        if isinstance(obj, (list, tuple)):
            return type(obj)((self._map(item, fn) for item in obj))
        if isinstance(obj, dict):
            return {key: self._map(value, fn) for key, value in obj.items()}
        return fn(obj)

    def _bytes(self, obj: Any) -> int:
        if isinstance(obj, torch.Tensor):
            return int(
                getattr(
                    obj, "nbytes", obj.numel() * max(1, obj.element_size())
                )
            )
        if isinstance(obj, (list, tuple)):
            return sum((self._bytes(item) for item in obj))
        if isinstance(obj, dict):
            return sum((self._bytes(value) for value in obj.values()))
        return 0

    def _should_pin(self, tensor: Any) -> bool:
        if not isinstance(tensor, torch.Tensor):
            return False
        if tensor.device.type != "cpu":
            return False
        if self._backend not in {"cuda", "xpu", "mps"}:
            return False
        if (
            self._pin_if_needed
            and hasattr(tensor, "is_pinned")
            and tensor.is_pinned()
        ):
            return False
        return self._pin_if_needed

    def _pin_tensor(self, tensor: Any) -> Any:
        if self._should_pin(tensor):
            with suppress(Exception):
                return tensor.pin_memory()
        return tensor

    def _pin_cpu(self, obj: Any) -> Any:
        return self._map(obj, self._pin_tensor)

    def _move_tensor(self, tensor: Any) -> Any:
        if isinstance(tensor, torch.Tensor):
            kwargs: Dict[str, Any] = {
                "non_blocking": self._backend in {"cuda", "xpu", "mps"}
            }
            if self._amp_dtype is not None:
                kwargs["dtype"] = self._amp_dtype
            return tensor.to(self._dev, **kwargs)
        return tensor

    def _to_device(self, obj: Any) -> Any:
        return self._map(obj, self._move_tensor)

    def _allocate_tensor_like(self, tensor: Any) -> Any:
        if isinstance(tensor, torch.Tensor):
            dtype = self._amp_dtype or tensor.dtype
            return torch.empty(tensor.shape, device=self._dev, dtype=dtype)
        return tensor

    def _allocate_static_like(self, obj: Any) -> Any:
        return self._map(obj, self._allocate_tensor_like)

    def _copy_structure(self, dst_obj: Any, src_obj: Any) -> Any:
        if isinstance(dst_obj, torch.Tensor) and isinstance(src_obj, torch.Tensor):
            dtype = self._amp_dtype or src_obj.dtype
            src_tensor = src_obj.to(dtype=dtype) if dst_obj.dtype != dtype else src_obj
            dst_obj.copy_(
                src_tensor,
                non_blocking=self._backend in {"cuda", "xpu", "mps"},
            )
            return dst_obj
        if isinstance(dst_obj, (list, tuple)) and isinstance(src_obj, (list, tuple)):
            return type(dst_obj)(
                (self._copy_structure(d, s) for d, s in zip(dst_obj, src_obj))
            )
        if isinstance(dst_obj, dict) and isinstance(src_obj, dict):
            return {
                key: self._copy_structure(dst_obj[key], src_obj[key])
                for key in dst_obj.keys() & src_obj.keys()
            }
        return dst_obj

    def _copy_into(self, dst: Any, src: Any) -> Any:
        return self._copy_structure(dst, src)

    def _current_stream(self) -> Any:
        if self._backend == "cuda":
            return torch.cuda.current_stream(self._dev)
        if self._backend == "xpu" and hasattr(torch, "xpu"):
            return torch.xpu.current_stream(self._dev)
        if self._backend == "mps" and hasattr(torch, "mps"):
            return torch.mps.current_stream()
        return None

    def _stream_ctx(self, stream: Any) -> Any:
        if stream is None:
            return nullcontext()
        if self._backend == "cuda":
            return torch.cuda.stream(stream)
        if self._backend == "xpu" and hasattr(torch, "xpu"):
            return torch.xpu.stream(stream)
        if self._backend == "mps" and hasattr(torch, "mps"):
            return torch.mps.stream(stream)
        return nullcontext()

    def _enqueue(self, batch: Any, slot: int) -> None:
        stream = self._streams[slot] if self._streams else None
        start = time.perf_counter()
        with self._stream_ctx(stream):
            if self._graph_enabled:
                if self._static is None:
                    self._static = self._allocate_static_like(
                        self._to_device(batch)
                    )
                moved = self._copy_into(self._static, self._to_device(batch))
            else:
                moved = self._to_device(batch)
        end = time.perf_counter()
        self._copy_time_acc += end - start
        self._queue.append((moved, slot))
        self._bytes_in_q += self._bytes(moved)
        self._rr += 1

    def _preload(self) -> None:
        while len(self._queue) < self._queue.maxlen:
            if (
                self._max_bytes is not None
                and self._bytes_in_q >= self._max_bytes
            ):
                break
            try:
                batch = next(self._it)
            except StopIteration:
                break
            except Exception as exc:
                self._err = self._err or exc
                break
            batch = self._pin_cpu(batch)
            slot = self._rr % self._slots
            self._enqueue(batch, slot)

    def __iter__(self) -> DevicePrefetcher:
        return self

    def __next__(self) -> Any:
        if not self._queue:
            if self._err is not None:
                err = self._err
                self._err = None
                raise err
            raise StopIteration
        if len(self._queue) <= 1:
            self._starved += 1
        now = time.perf_counter()
        if self._last_yield_ts is not None:
            elapsed = now - self._last_yield_ts
            self._ema_compute = (
                0.9 * self._ema_compute + 0.1 * elapsed
                if self._ema_compute > 0
                else elapsed
            )
        moved, slot = self._queue.popleft()
        self._bytes_in_q -= min(self._bytes_in_q, self._bytes(moved))
        producer = self._streams[slot] if self._streams else None
        consumer = self._current_stream()
        if (
            producer is not None
            and consumer is not None
            and hasattr(consumer, "wait_stream")
        ):
            consumer.wait_stream(producer)
        if self._use_record_stream and consumer is not None:
            self._record_stream(moved, consumer)
        self._preload()
        self._steps += 1
        if self._autotune and self._steps % self._tune_interval == 0:
            self._retune_depth()
            self._copy_time_acc = 0.0
            self._starved = 0
        self._last_yield_ts = time.perf_counter()
        return moved

    def _record_item_stream(self, item: Any, stream: Any) -> None:
        if isinstance(item, torch.Tensor) and item.device.type == self._backend:
            with suppress(Exception):
                item.record_stream(stream)
            return
        if isinstance(item, (list, tuple)):
            for sub in item:
                self._record_item_stream(sub, stream)
            return
        if isinstance(item, dict):
            for sub in item.values():
                self._record_item_stream(sub, stream)

    def _record_stream(self, obj: Any, stream: Any) -> None:
        self._record_item_stream(obj, stream)

    def _set_depth(self, new_depth: int) -> None:
        depth = int(max(self._depth_min, min(self._depth_max, new_depth)))
        if depth == self._queue.maxlen:
            return
        self._queue = deque(self._queue, maxlen=depth)

    def _retune_depth(self) -> None:
        steps = float(max(1, self._tune_interval))
        starve_ratio = float(self._starved) / steps
        avg_copy = self._copy_time_acc / steps
        avg_compute = max(1e-06, self._ema_compute)
        want_up = starve_ratio > 0.3 or avg_copy > 0.3 * avg_compute
        want_down = (
            starve_ratio < 0.05 and len(self._queue) >= self._queue.maxlen - 1
        )
        if want_up and self._queue.maxlen < self._depth_max:
            self._set_depth(self._queue.maxlen + 1)
        elif want_down and self._queue.maxlen > self._depth_min:
            self._set_depth(self._queue.maxlen - 1)

    def close(self) -> None:
        with suppress(Exception):
            self._queue.clear()
        self._streams = []
        self._static = None
        self._graph = None

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def graphs_capture(self, capture_fn: Callable[[Any], Any]) -> bool:
        if not self._graph_enabled:
            raise RuntimeError(
                "graphs_capture requires enable_graphs=True and a CUDA backend."
            )
        if not self._queue:
            self._preload()
        moved, _ = self._queue[0]
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            capture_fn(moved)
        self._graph = graph
        return True

    def graphs_replay(self) -> None:
        if self._graph is None:
            raise RuntimeError("Call graphs_capture() before graphs_replay().")
        _ = next(self)
        self._graph.replay()


class DataLoader:
    def __init__(
        self,
        device: torch.device,
        *args: Any,
        node: BaseNode | None = None,
        dataset: BaseNode | None = None,
        prefetch_factor: int = 2,
        non_blocking: bool = True,
        length: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        node_obj = node or dataset
        if not isinstance(node_obj, BaseNode):
            raise TypeError(
                "data.collate.DataLoader supports only torchdata.nodes.BaseNode instances."
            )
        self._node = node_obj
        self._device = device
        self._prefetch_factor = max(1, int(prefetch_factor or 2))
        self._non_blocking = bool(non_blocking)
        self._length = int(length) if length is not None else None
        base = Loader(self._node)
        dev_t = getattr(self._device, "type", "cpu")
        if dev_t in ("cuda", "mps", "xpu") and DevicePrefetcher is not None:
            try:
                self._iterable = DevicePrefetcher(
                    base, device=self._device, depth=self._prefetch_factor
                )
            except TypeError:
                self._iterable = base
        else:
            self._iterable = base

    def __iter__(self) -> Any:
        return iter(self._iterable)

    def __len__(self) -> Any:
        if self._length is not None:
            return self._length
        try:
            length = _infer_node_length(self._node)
            return length if length is not None else 1
        except Exception:
            return 1


def _infer_node_length(node: BaseNode) -> int | None:
    candidates = [
        lambda: len(node),
        getattr(node, "length", None),
        getattr(node, "size", None),
        getattr(node, "num_rows", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            value = candidate() if callable(candidate) else candidate
            if value is None:
                continue
            return int(value)
        except Exception:
            continue
    return None


def flatten(objs: Iterable[Any]) -> Iterable[Any]:
    for obj in objs:
        if obj is None:
            continue
        if isinstance(obj, (list, tuple, set)):
            for item in flatten(obj):
                if item is not None:
                    yield item
            continue
        yield obj


class _Keep:
    __slots__ = ("_objs",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._objs = list(flatten(args))
        if kwargs:
            self._objs.extend(list(flatten(kwargs.values())))

    def add(self, *args: Any, **kwargs: Any) -> None:
        self._objs.extend(list(flatten(args)))
        if kwargs:
            self._objs.extend(list(flatten(kwargs.values())))

    def cleanup(self) -> None:
        for obj in self._objs:
            cleaned = False
            for name in (
                "cleanup",
                "close",
                "shutdown",
                "stop",
                "terminate",
                "join",
                "disconnect",
                "release",
            ):
                if hasattr(obj, name):
                    with suppress(Exception):
                        getattr(obj, name)()
                    cleaned = True
                    break
            if cleaned:
                continue
            if callable(obj):
                with suppress(Exception):
                    obj()
def fetch(
    sample: Any,
    *args: Any,
    labels_dtype: Optional[torch.dtype] = None,
    sanitize: bool = False,
    flatten_features: bool = False,
    **kwargs: Any,
) -> Any:
    converter = partial(
        _convert_mapping_to_batch,
        flatten_features=flatten_features,
        labels_dtype=labels_dtype,
        sanitize=sanitize,
    )
    if isinstance(sample, (list, tuple)):
        return [
            converter(item) if isinstance(item, Mapping) else item
            for item in sample
        ]
    if isinstance(sample, Mapping):
        return converter(sample)
    return sample


def dataloader(
    memmap_dir: str,
    device: Union[str, torch.device],
    batch_size: int,
    val_frac: float,
    *args: Any,
    prefetch_factor: int = 2,
    non_blocking_copy: bool = True,
    labels_dtype: Optional[torch.dtype] = None,
    sanitize: bool = False,
    flatten_features: bool = False,
    io_backend: str = "auto",
    **kwargs: Any,
) -> Tuple[Any, Optional[Any], _Keep]:
    device_obj = (
        torch.device(device)
        if not isinstance(device, torch.device)
        else device
    )
    backend_input = io_backend or "auto"
    if not isinstance(backend_input, str):
        raise TypeError(
            "io_backend must be a string such as 'auto', 'local', or 'flight'"
        )
    backend_normalized = backend_input.lower()
    allow_fallback = backend_normalized == "auto"
    if backend_normalized == "auto":
        backend = "flight" if has_arrow_flight() else "local"
    else:
        backend = backend_normalized
    threads = System.optimize_threads()
    map_fn = partial(
        fetch,
        labels_dtype=labels_dtype,
        sanitize=sanitize,
        flatten_features=flatten_features,
    )
    from .dataset import BatchReader, SampleReader
    wrap_kwargs = dict(
        device=device_obj,
        threads=threads,
        prefetch_factor=prefetch_factor,
        non_blocking_copy=bool(non_blocking_copy),
        map_fn=map_fn,
        batch_reader_cls=BatchReader,
        sample_reader_cls=SampleReader,
    )

    if backend == "local":
        return _build_local_loaders(
            memmap_dir,
            int(batch_size),
            val_frac,
            **wrap_kwargs,
        )
    if backend != "flight":
        raise ValueError(f"Unsupported io_backend: {backend_input!r}")

    try:
        return _build_flight_loaders(
            memmap_dir,
            int(batch_size),
            val_frac,
            **wrap_kwargs,
        )
    except Exception as exc:
        if not allow_fallback:
            raise
        warnings.warn(
            (
                "Arrow Flight backend unavailable"
                f" ({exc}). Falling back to the local memory-mapped dataloader."
            ),
            RuntimeWarning,
        )
        return _build_local_loaders(
            memmap_dir,
            int(batch_size),
            val_frac,
            **wrap_kwargs,
        )
