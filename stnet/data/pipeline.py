# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import time
from contextlib import suppress
from functools import partial
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import ctypes
import importlib
import itertools
import threading
from threading import Lock

import torch
try:
    from torchdata.nodes import (
        BaseNode,
        IterableWrapper,
        Loader,
        MultiNodeWeightedSampler,
        ParallelMapper,
        PinMemory,
        Prefetcher,
    )
except Exception:
    from torchdata.nodes import BaseNode, IterableWrapper, Loader, ParallelMapper, PinMemory, Prefetcher
    MultiNodeWeightedSampler = None
from .datatype import to_torch_tensor
from .nodes import DevicePrefetcher, GDSBatchReader
from ..backend.environment import System


def identity(item: Any) -> Any:
    return item


class ThreadLoadBalancer:

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
        self._psutil = self._import_psutill()
        self._allowed_cpus = self.total_procs()
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
        self._omp_ok = self.spread_threads()
        self._enabled = (len(self._allowed_cpus) >= 2) or self._omp_ok
        self.tune_threads(io_workers, initial=True)

    @staticmethod
    def _import_psutill():
        try:
            return importlib.import_module("psutil")
        except Exception:
            return None

    def total_procs(self) -> list[int]:
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
            return sorted(int(c) for c in os.sched_getaffinity(0))
        except Exception:
            pass
        return list(range(max(1, os.cpu_count() or 1)))

    @staticmethod
    def spread_threads() -> bool:
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
    def _pin_thread_windows(core: int) -> bool:
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
    def _pin_thread_linux(core: int) -> bool:
        try:
            tid = threading.get_native_id()
            os.sched_setaffinity(tid, {int(core)})
            return True
        except Exception:
            try:
                os.sched_setaffinity(0, {int(core)})
                return True
            except Exception:
                return False

    @staticmethod
    def _pin_thread_bsd(core: int) -> bool:
        return False

    def pin_thread(self) -> None:
        if not self._enabled:
            return
        attempts = getattr(self._tls, "attempts", 0)
        if getattr(self._tls, "pinned", False) or attempts >= 4:
            return
        self._tls.attempts = attempts + 1
        core = self._next_core()
        ok = False
        if os.name == "nt":
            ok = self._pin_thread_windows(core)
        else:
            plat = sys.platform
            if plat.startswith("linux"):
                ok = self._pin_thread_linux(core)
            elif "bsd" in plat:
                ok = self._pin_thread_bsd(core)
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
    def optimize_threads(intra: Optional[int] = None, inter: Optional[int] = None) -> None:
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

    def tune_threads(self, io_workers: Optional[int] = None, *, initial: bool = False) -> None:
        if not self._enabled:
            return
        if initial:
            cpus = max(1, len(self._allowed_cpus))
            tuned_workers = max(1, min(int(io_workers if io_workers is not None else self._io_workers), cpus))
            self._io_workers = tuned_workers
            try:
                intra = int(torch.get_num_threads())
            except Exception:
                intra = cpus
            if intra * tuned_workers > cpus:
                new_intra = max(1, cpus // tuned_workers)
                self.optimize_threads(intra=new_intra)
            want_inter = max(1, min(tuned_workers // 2, 4))
            self.optimize_threads(inter=want_inter)
            return
        self._retune_threads()

    def _retune_threads(self) -> None:
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
            self.optimize_threads(intra=target_intra)
            self.optimize_threads(inter=max(1, min(2, workers)))
        else:
            relaxed = min(4, max(1, cpus // max(1, workers // 2)))
            current = max(1, torch.get_num_threads())
            if current < relaxed:
                self.optimize_threads(intra=relaxed)

    def new_thread(self, fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        if not self._enabled:
            return fn

        def _inner(x: Any) -> Any:
            self.pin_thread()
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
            self.tune_threads()
            return y

        return _inner

    def optimize_procs(self, io_workers: int) -> int:
        if not self._enabled:
            return int(io_workers)
        cpus = max(1, len(self._allowed_cpus))
        tuned = max(1, min(int(io_workers), cpus))
        self._io_workers = tuned
        return tuned




def process_batch(
    batch: Mapping[str, Any],
    *args: Any,
    flatten_features: bool,
    labels_dtype: Optional[torch.dtype],
    sanitize: bool,
    **kwargs: Any,
) -> Dict[str, Any]:
    features = batch["X"]
    labels = batch["Y"]
    if (
        flatten_features
        and isinstance(features, torch.Tensor)
        and (features.dim() >= 2)
    ):
        features = features.flatten(start_dim=1)
    labels_tensor = to_torch_tensor(labels)
    if labels_dtype is not None and getattr(labels_tensor, "dtype", None) != labels_dtype:
        labels_tensor = labels_tensor.to(dtype=labels_dtype)
    if sanitize and torch.is_floating_point(labels_tensor):
        labels_tensor = torch.nan_to_num(
            labels_tensor, nan=0.0, posinf=0.0, neginf=0.0
        )
    return {"X": features, "Y": labels_tensor}


def compose(
    node: Union[BaseNode, Sequence[BaseNode], Mapping[str, BaseNode]],
    *args: Any,
    device: torch.device,
    threads: Dict[str, int],
    prefetch_factor: int,
    non_blocking_copy: bool,
    map_fn: Callable[[Any], Any],
    length: Optional[int] = None,
    **kwargs: Any,
) -> "BatchLoader":
    io_workers = max(1, int(threads["dataloader_workers"]))
    prebatch = max(1, int(threads["prefetch_factor"]))
    load_balancer = ThreadLoadBalancer(io_workers)
    io_workers = load_balancer.optimize_procs(io_workers)
    map_fn = load_balancer.new_thread(map_fn)

    thread_max_concurrent = io_workers
    if isinstance(node, Mapping):
        nodes_map: Dict[str, BaseNode] = {
            str(key): value for key, value in node.items() if isinstance(value, BaseNode)
        }
        if nodes_map:
            if MultiNodeWeightedSampler is None:
                raise RuntimeError(
                    "torchdata.nodes.MultiNodeWeightedSampler is required to compose multiple nodes."
                )
            weights = {key: 1.0 for key in nodes_map}
            source_node: BaseNode = MultiNodeWeightedSampler(nodes_map, weights)
        elif isinstance(node, BaseNode):
            source_node = node
        else:
            raise TypeError(
                "Unsupported Mapping passed to compose: expected Mapping[str, BaseNode]."
            )
    elif isinstance(node, (list, tuple)):
        nodes_map = {str(index): value for index, value in enumerate(node) if isinstance(value, BaseNode)}
        if not nodes_map:
            raise TypeError(
                "Empty or invalid Sequence passed to compose: expected Sequence[BaseNode]."
            )
        if MultiNodeWeightedSampler is None:
            raise RuntimeError(
                "torchdata.nodes.MultiNodeWeightedSampler is required to compose multiple nodes."
            )
        weights = {key: 1.0 for key in nodes_map}
        source_node = MultiNodeWeightedSampler(nodes_map, weights)
    else:
        # Route a single node through MultiNodeWeightedSampler as well
        # so that sharding behavior is consistent across Mapping[Mapping],
        # Sequence[Mapping], and single Mapping inputs.
        if MultiNodeWeightedSampler is None:
            source_node = node
        else:
            nodes_map = {"default": node}
            weights = {"default": 1.0}
            source_node = MultiNodeWeightedSampler(nodes_map, weights)

    wrapped: BaseNode = ParallelMapper(
        source_node,
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
    return BatchLoader(
        device=device,
        node=wrapped,
        prefetch_factor=prefetch_factor,
        non_blocking=bool(non_blocking_copy),
        length=length,
    )


def initialize(
    memmap_dir: Union[str, Sequence[str], Mapping[str, str]],
    batch_size: int,
    val_frac: float,
    *args: Any,
    device: torch.device,
    threads: Dict[str, int],
    prefetch_factor: int,
    non_blocking_copy: bool,
    map_fn: Callable[[Any], Any],
    batch_reader_cls: type,
    sample_reader_cls: type,
    batch_reader_kwargs: Optional[Dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Tuple[Any, Optional[Any], _Allocated]:
    allocated = _Allocated()

    def _make_train_node(directory: Union[str, os.PathLike[str]]) -> Tuple[BaseNode, int, int, int]:
        path = os.fspath(directory)
        reader = sample_reader_cls.from_dir(
            path,
            split="train",
            batch_size=int(batch_size),
            val_frac=val_frac,
        )
        allocated.add(reader)
        meta = reader._load_meta()
        total_samples = int(meta.get("N", 0))
        train_range = reader._indices()
        train_start = int(getattr(train_range, "start", 0))
        train_end = int(getattr(train_range, "stop", total_samples if total_samples else 0))
        if train_end <= train_start and total_samples:
            train_end = total_samples
        reader_kwargs_local = dict(batch_reader_kwargs or {})
        batcher = batch_reader_cls(
            reader, train_start, train_end, int(batch_size), **reader_kwargs_local
        )
        allocated.add(batcher)
        return IterableWrapper(batcher), len(batcher), total_samples, train_end

    if isinstance(memmap_dir, (list, tuple)):
        node_tr = [_make_train_node(directory)[0] for directory in memmap_dir]
        train_loader = compose(
            node_tr,
            device=device,
            threads=threads,
            prefetch_factor=prefetch_factor,
            non_blocking_copy=non_blocking_copy,
            map_fn=map_fn,
            length=None,
        )
        return (train_loader, None, allocated)

    if isinstance(memmap_dir, Mapping):
        node_tr = {
            str(key): _make_train_node(directory)[0]
            for key, directory in memmap_dir.items()
        }
        train_loader = compose(
            node_tr,
            device=device,
            threads=threads,
            prefetch_factor=prefetch_factor,
            non_blocking_copy=non_blocking_copy,
            map_fn=map_fn,
            length=None,
        )
        return (train_loader, None, allocated)

    node_tr, train_length, total, train_end = _make_train_node(memmap_dir)
    train_loader = compose(
        node_tr,
        device=device,
        threads=threads,
        prefetch_factor=prefetch_factor,
        non_blocking_copy=non_blocking_copy,
        map_fn=map_fn,
        length=train_length,
    )
    val_loader: Optional[Any] = None
    if val_frac > 0 and train_end < total:
        reader_vl = sample_reader_cls.from_dir(
            memmap_dir,
            split="val",
            batch_size=int(batch_size),
            val_frac=val_frac,
        )
        allocated.add(reader_vl)
        val_range = reader_vl._indices()
        val_start = int(getattr(val_range, "start", train_end))
        val_end = int(getattr(val_range, "stop", total))
        if val_end <= val_start:
            val_end = total
        reader_kwargs_local = dict(batch_reader_kwargs or {})
        batcher_vl = batch_reader_cls(
            reader_vl, val_start, val_end, int(batch_size), **reader_kwargs_local
        )
        allocated.add(batcher_vl)
        node_vl = IterableWrapper(batcher_vl)
        val_loader = compose(
            node_vl,
            device=device,
            threads=threads,
            prefetch_factor=prefetch_factor,
            non_blocking_copy=non_blocking_copy,
            map_fn=map_fn,
            length=len(batcher_vl),
        )
    return (train_loader, val_loader, allocated)


class BatchLoader:
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
                "data.pipeline.BatchLoader supports only torchdata.nodes.BaseNode instances."
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
                gpu_guard_default = "2048" if dev_t == "cuda" else "512"
                gpu_guard_mb = int(
                    os.environ.get("STNET_GPU_GUARD_MB", gpu_guard_default)
                )
            except Exception:
                gpu_guard_mb = 2048 if dev_t == "cuda" else 512
            try:
                host_guard_mb = int(os.environ.get("STNET_HOST_GUARD_MB", "1024"))
            except Exception:
                host_guard_mb = 1024
            try:
                self._iterable = DevicePrefetcher(
                    base,
                    device=self._device,
                    depth=self._prefetch_factor,
                    memory_backpressure=True,
                    gpu_guard_bytes=gpu_guard_mb * (1 << 20),
                    host_guard_bytes=host_guard_mb * (1 << 20),
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
            length = _get_node_length(self._node)
            return length if length is not None else 1
        except Exception:
            return 1


def _get_node_length(node: BaseNode) -> int | None:
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


def _flatten_args(objs: Iterable[Any]) -> Iterable[Any]:
    for obj in objs:
        if obj is None:
            continue
        if isinstance(obj, (list, tuple, set)):
            for item in _flatten_args(obj):
                if item is not None:
                    yield item
            continue
        yield obj


class _Allocated:
    __slots__ = ("_objs",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._objs = list(_flatten_args(args))
        if kwargs:
            self._objs.extend(list(_flatten_args(kwargs.values())))

    def add(self, *args: Any, **kwargs: Any) -> None:
        self._objs.extend(list(_flatten_args(args)))
        if kwargs:
            self._objs.extend(list(_flatten_args(kwargs.values())))

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
def collate(
    sample: Any,
    *args: Any,
    labels_dtype: Optional[torch.dtype] = None,
    sanitize: bool = False,
    flatten_features: bool = False,
    **kwargs: Any,
) -> Any:
    converter = partial(
        process_batch,
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


def fetch(
    memmap_dir: Union[str, Sequence[str], Mapping[str, str]],
    device: Union[str, torch.device],
    batch_size: int,
    val_frac: float,
    *args: Any,
    prefetch_factor: int = 2,
    non_blocking_copy: bool = True,
    labels_dtype: Optional[torch.dtype] = None,
    sanitize: bool = False,
    flatten_features: bool = False,
    **kwargs: Any,
) -> Tuple[Any, Optional[Any], _Allocated]:
    device_obj = (
        torch.device(device)
        if not isinstance(device, torch.device)
        else device
    )
    threads = System.optimize_threads()
    map_fn = partial(
        collate,
        labels_dtype=labels_dtype,
        sanitize=sanitize,
        flatten_features=flatten_features,
    )
    from .nodes import BatchReader, SampleReader

    def _env_flag(name: str, default: str = "0") -> bool:
        value = os.environ.get(name, default)
        return value not in {"", "0", "false", "False"}

    # Prefer torch.cuda.gds-backed reader when available.
    memmap_is_str = isinstance(memmap_dir, str)
    use_gds = (
        memmap_is_str
        and isinstance(device_obj, torch.device)
        and device_obj.type == "cuda"
        and torch.cuda.is_available()
        and _env_flag("STNET_GDS")
        and GDSBatchReader is not None
        and os.path.exists(os.path.join(memmap_dir, "features.bin"))
        and os.path.exists(os.path.join(memmap_dir, "labels.bin"))
    )
    batch_reader_cls = GDSBatchReader if use_gds else BatchReader
    batch_reader_kwargs: Optional[Dict[str, Any]] = (
        {"device": device_obj} if use_gds else None
    )
    wrap_kwargs = dict(
        device=device_obj,
        threads=threads,
        prefetch_factor=prefetch_factor,
        non_blocking_copy=bool(non_blocking_copy),
        map_fn=map_fn,
        batch_reader_cls=batch_reader_cls,
        sample_reader_cls=SampleReader,
        batch_reader_kwargs=batch_reader_kwargs,
    )

    return initialize(
        memmap_dir,
        int(batch_size),
        val_frac,
        **wrap_kwargs,
    )
