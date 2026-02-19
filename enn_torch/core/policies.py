# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Self,
    TYPE_CHECKING,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
from ..core.datatypes import (
    default_underflow_action,
    env_bool,
    env_first_int,
    env_int,
    env_float,
    env_str,
    normalize_underflow_action,
)
from ..nn.graph import clear_model_cache
from .concurrency import Mutex
from .precision import Autocast, DeviceMeta, Quantization, is_scale_safe
from .system import (
    CPU,
    _call,
    _default_thread_limit,
    _log_debug,
    _log_info,
    _optimal_threads,
    get_device,
    is_cuda_bf16_supported,
)
if TYPE_CHECKING:
    from ..data.pipeline import Dataset


_TORCH_INTEROP_LOCKED: bool = False
_TORCH_INTEROP_THREADS_SET: Optional[int] = None
_TORCH_NUM_THREADS_SET: Optional[int] = None
_TORCH_THREAD_CFG_LOCK = Mutex()


def optimal_procs() -> dict[str, Union[int, str]]:
    return WorkerPolicy.optimize().get_procs_setting()


def optimal_threads() -> dict[str, Union[int, bool]]:
    return WorkerPolicy.optimize().get_thread_setting()


def optimize_threads(
    intra: Optional[int] = None,
    inter: Optional[int] = None,
) -> dict[str, Union[int, bool]]:
    wp = WorkerPolicy.optimize()
    if intra is not None:
        wp = replace(wp, intra_ops=int(intra))
    if inter is not None:
        wp = replace(wp, inter_ops=int(inter))
    wp.set_thread_setting()
    return wp.get_thread_setting()


class LossWeightPolicy(Protocol):
    def weights(self: Self) -> Tuple[float, float]: ...

    def update(
        self: Self,
        top_loss: Optional[torch.Tensor],
        bottom_loss: Optional[torch.Tensor],
    ) -> None:
        raise NotImplementedError


@dataclass
class WorkerPolicy:
    nproc_per_node: int = 1
    device: str = "cpu"
    local_world_size: int = 1
    intra_ops: int = 1
    inter_ops: int = 1
    compile_threads: int = 0
    ckpt_writer_threads: int = 0
    num_workers: int = 1
    prebatch: int = 1
    prefetch_factor: int = 1
    max_concurrency: int = 1
    h2d_streams: int = 1

    @staticmethod
    def _cpu_count() -> int:
        return CPU.count()

    @staticmethod
    def _available_accelerator() -> Tuple[str, int]:
        dev_type = "cpu"
        n = 0
        try:
            accel = getattr(torch, "accelerator", None)
            if (
                accel is not None
                and hasattr(accel, "is_available")
                and accel.is_available()
            ):
                current = getattr(accel, "current_accelerator", None)
                if callable(current):
                    dev = current(False)
                    if isinstance(dev, torch.device):
                        dev_type = dev.type
                dc = getattr(accel, "device_count", None)
                if callable(dc):
                    n = int(dc())
        except Exception:
            dev_type, n = "cpu", 0
        try:
            if n <= 0:
                if torch.cuda.is_available():
                    dev_type = "cuda"
                    n = int(torch.cuda.device_count())
                else:
                    xpu = getattr(torch, "xpu", None)
                    if (
                        xpu is not None
                        and callable(getattr(xpu, "is_available", None))
                        and xpu.is_available()
                    ):
                        dev_type = "xpu"
                        n = int(getattr(xpu, "device_count", lambda: 1)())
                    else:
                        mps_backend = getattr(torch.backends, "mps", None)
                        if (
                            mps_backend is not None
                            and callable(
                                getattr(mps_backend, "is_available", None)
                            )
                            and mps_backend.is_available()
                        ):
                            dev_type = "mps"
                            n = 1
        except Exception:
            pass
        if n <= 0:
            dev_type, n = "cpu", 0
        return dev_type, max(0, n)

    @classmethod
    def optimize(cls: type[Self]) -> "WorkerPolicy":
        ncpu_raw = max(1, int(cls._cpu_count() or 1))
        dev_type, nacc = cls._available_accelerator()
        is_accel = bool(nacc and int(nacc) > 0)
        local_world_guess = max(1, int(nacc or 1)) if is_accel else 1
        local_world_guess = max(
            1,
            int(
                env_first_int(
                    (
                        "ENN_LOCAL_WORLD_SIZE",
                        "LOCAL_WORLD_SIZE",
                        "SLURM_NTASKS_PER_NODE",
                    ),
                    local_world_guess,
                )
            ),
        )
        if is_accel and int(nacc) > 0:
            allow_over = int(
                env_first_int(
                    (
                        "ENN_ALLOW_ACCELERATOR_OVERSUBSCRIBE",
                        "ENN_ALLOW_GPU_OVERSUBSCRIBE",
                    ),
                    0,
                )
            )
            if not allow_over and int(local_world_guess) > int(nacc):
                local_world_guess = int(nacc)
        _nogil = bool(CPU.is_optimized_for_no_gil())
        cap_mult = _default_thread_limit(
            ncpu_raw, is_accel=is_accel, nogil=_nogil
        )
        try:
            thr = int(
                env_first_int(
                    ("ENN_THREAD_CAP_THRESHOLD_CORES", "ENN_THREAD_CAP_THRESHOLD"),
                    8,
                )
            )
            low_mult = int(
                env_first_int(
                    ("ENN_THREAD_CAP_MULT_LOW", "ENN_THREAD_CAP_MULT_SMALL"), 2
                )
            )
            high_mult = int(
                env_first_int(
                    ("ENN_THREAD_CAP_MULT_HIGH", "ENN_THREAD_CAP_MULT_LARGE"), 1
                )
            )
            if int(ncpu_raw) <= int(thr):
                cap_mult = max(1, int(low_mult))
            else:
                cap_mult = max(1, int(high_mult))
        except Exception:
            pass
        distribute_default = int(local_world_guess) > 1
        distribute = bool(
            env_first_int(
                ("ENN_DISTRIBUTE_THREAD_CAP",), int(distribute_default)
            )
        )
        thread_cap_total = _optimal_threads(
            ncpu=ncpu_raw,
            cap_mult=cap_mult,
            local_world=int(local_world_guess),
            distribute=bool(distribute),
        )
        compile_threads = 0
        if env_bool("ENN_TORCH_COMPILE", default=True):
            requested = int(
                env_first_int(
                    (
                        "ENN_INDUCTOR_COMPILE_THREADS",
                        "ENN_COMPILE_THREADS",
                    ),
                    0,
                )
            )
            if requested <= 0:
                requested = int(
                    env_first_int(("TORCHINDUCTOR_COMPILE_THREADS",), 0)
                )
            if requested > 0:
                compile_threads = int(requested)
            else:
                compile_threads = 1 if CPU.is_free_threaded_build() else 2
                if int(thread_cap_total) <= 6:
                    compile_threads = 1

        min_non_compile = 3
        max_compile = max(0, int(thread_cap_total) - int(min_non_compile))
        if max_compile <= 0:
            compile_threads = 0
        else:
            compile_threads = max(
                0, min(int(compile_threads), int(max_compile))
            )

        thread_cap_after_compile = max(
            2, int(thread_cap_total) - int(compile_threads)
        )
        eff_cores_pre = max(
            1, int(thread_cap_after_compile) // max(1, int(cap_mult))
        )
        ckpt_writer_threads = 0
        req_ckpt = int(
            env_first_int(
                ("ENN_DCP_WRITER_THREADS", "ENN_CKPT_WRITER_THREADS"), 0
            )
            or 0
        )
        if req_ckpt > 0:
            ckpt_writer_threads = req_ckpt
        else:
            if eff_cores_pre <= 4:
                ckpt_writer_threads = 1
            elif eff_cores_pre <= 8:
                ckpt_writer_threads = 2
            else:
                ckpt_writer_threads = 4
        min_non_ckpt = 2
        ckpt_writer_threads = max(
            0,
            min(
                int(ckpt_writer_threads),
                max(0, int(thread_cap_after_compile) - int(min_non_ckpt)),
            ),
        )
        thread_cap = max(
            2, int(thread_cap_after_compile) - int(ckpt_writer_threads)
        )
        eff_cores = max(1, int(thread_cap) // max(1, int(cap_mult)))
        soft_inflight = 8 if is_accel else 4
        with contextlib.suppress(Exception):
            lp = LoaderPolicy()
            hard = int(lp.hard_inflight_batches(dev_type))
            soft_inflight = max(
                1, int(hard * max(1, int(lp.soft_cap_multiplier)))
            )
        soft_auto_enabled = bool(env_first_int(("ENN_SOFT_INFLIGHT_AUTO",), 1))
        soft_inflight_max_default = (
            (32 if is_accel else 24) if _nogil else (16 if is_accel else 12)
        )
        soft_inflight_max = max(
            8,
            env_first_int(
                ("ENN_SOFT_INFLIGHT_MAX",), soft_inflight_max_default
            ),
        )
        soft_inflight_explicit = env_first_int(("ENN_SOFT_INFLIGHT_CAP",), 0)
        if soft_inflight_explicit > 0:
            soft_inflight = max(1, int(soft_inflight_explicit))
        elif soft_auto_enabled:
            soft_base = max(0, env_first_int(("ENN_SOFT_INFLIGHT_BASE",), 2))
            soft_div = max(1, env_first_int(("ENN_SOFT_INFLIGHT_DIV",), 4))
            auto_soft = int(soft_base) + max(
                0, int(eff_cores) // int(soft_div)
            )
            soft_inflight = max(
                int(soft_inflight), min(int(auto_soft), int(soft_inflight_max))
            )
        soft_inflight = max(1, min(int(soft_inflight), int(thread_cap)))
        if is_accel:
            if eff_cores <= 4:
                model_ratio = 1.00
            elif eff_cores <= 8:
                model_ratio = 0.90
            elif eff_cores <= 16:
                model_ratio = 0.80
            elif eff_cores <= 32:
                model_ratio = 0.70
            else:
                model_ratio = 0.60
        else:
            if eff_cores <= 4:
                model_ratio = 1.00
            elif eff_cores <= 8:
                model_ratio = 0.95
            elif eff_cores <= 16:
                model_ratio = 0.90
            else:
                model_ratio = 0.85
        with contextlib.suppress(Exception):
            env_key = (
                "ENN_MODEL_CORE_RATIO_ACCEL"
                if is_accel
                else "ENN_MODEL_CORE_RATIO"
            )
            model_ratio = float(env_float(env_key, float(model_ratio)))
        model_ratio = float(max(0.25, min(1.0, model_ratio)))
        model_budget = max(2, int(round(float(eff_cores) * model_ratio)))
        if model_budget <= 2:
            inter_ops = 1
        elif model_budget <= 8:
            inter_ops = max(1, model_budget // 4)
        else:
            inter_ops = max(2, min(8, model_budget // 6))
        inter_ops = max(1, min(int(inter_ops), max(1, int(model_budget) - 1)))
        intra_ops = max(1, int(model_budget) - int(inter_ops))
        data_budget = max(
            1, int(thread_cap) - (int(intra_ops) + int(inter_ops))
        )
        prebatch = 1
        prefetch_factor = 1
        env_pre = env_str("ENN_PREBATCH")
        if env_pre:
            with contextlib.suppress(Exception):
                prebatch = max(1, int(env_pre))
        elif _nogil:
            prebatch = 2
        env_pf = env_str("ENN_PREFETCH_FACTOR")
        if env_pf:
            with contextlib.suppress(Exception):
                prefetch_factor = max(1, int(env_pf))
        elif _nogil:
            prefetch_factor = 2
        prebatch = max(1, int(prebatch))
        prefetch_factor = max(1, int(prefetch_factor))
        base_workers = max(1, int(data_budget))
        base_workers = min(
            int(base_workers), int(thread_cap), int(soft_inflight)
        )
        max_workers = max(
            1,
            int(
                (int(soft_inflight) - int(prebatch))
                // max(1, int(prefetch_factor))
            ),
        )
        num_workers = max(
            1, min(int(base_workers), int(max_workers), int(soft_inflight))
        )
        max_concurrency = max(1, int(num_workers))
        total_threads = int(intra_ops) + int(inter_ops) + int(num_workers)
        if total_threads > int(thread_cap):
            overflow = int(total_threads) - int(thread_cap)
            if dev_type == "cpu":
                num_workers = max(1, int(num_workers) - int(overflow))
            else:
                intra_ops = max(1, int(intra_ops) - int(overflow))

            intra_ops = max(
                1, int(thread_cap) - int(inter_ops) - int(num_workers)
            )
            max_concurrency = max(
                1, min(int(max_concurrency), int(num_workers))
            )
        local_world = int(local_world_guess)
        return cls(
            nproc_per_node=local_world,
            device=dev_type,
            local_world_size=local_world,
            intra_ops=int(intra_ops),
            inter_ops=int(inter_ops),
            compile_threads=int(compile_threads),
            ckpt_writer_threads=int(ckpt_writer_threads),
            num_workers=int(num_workers),
            prebatch=int(prebatch),
            prefetch_factor=int(prefetch_factor),
            max_concurrency=int(max_concurrency),
            h2d_streams=2 if dev_type in ("cuda", "xpu") else 1,
        )

    def get_thread_setting(self: Self) -> dict[str, int]:
        return {
            "intra_ops": int(self.intra_ops),
            "inter_ops": int(self.inter_ops),
            "compile_threads": int(getattr(self, "compile_threads", 0) or 0),
            "ckpt_writer_threads": int(getattr(self, "ckpt_writer_threads", 0) or 0),
            "num_workers": int(self.num_workers),
            "max_concurrency": int(self.max_concurrency),
            "prebatch": int(self.prebatch),
            "prefetch_factor": int(self.prefetch_factor),
        }

    def get_procs_setting(self: Self) -> dict[str, Union[int, str]]:
        return {
            "nproc_per_node": int(self.nproc_per_node),
            "device": str(self.device),
        }

    def set_thread_setting(self: Self) -> None:
        global _TORCH_NUM_THREADS_SET, _TORCH_INTEROP_THREADS_SET, _TORCH_INTEROP_LOCKED
        intra = max(1, int(self.intra_ops))
        inter = max(1, int(self.inter_ops))
        with _TORCH_THREAD_CFG_LOCK:
            if _TORCH_NUM_THREADS_SET != int(intra):
                _call(getattr(torch, "set_num_threads", None), int(intra))
                _TORCH_NUM_THREADS_SET = int(intra)
            if hasattr(torch, "set_num_interop_threads") and not bool(
                _TORCH_INTEROP_LOCKED
            ):
                if _TORCH_INTEROP_THREADS_SET is None:
                    try:
                        torch.set_num_interop_threads(int(inter))
                        _TORCH_INTEROP_THREADS_SET = int(inter)
                    except Exception:
                        _TORCH_INTEROP_LOCKED = True
                elif int(_TORCH_INTEROP_THREADS_SET) != int(inter):
                    pass
        try:
            ct = int(getattr(self, "compile_threads", 0) or 0)
        except Exception:
            ct = 0
        if ct > 0:
            explicit = env_str("ENN_INDUCTOR_COMPILE_THREADS") or env_str(
                "ENN_COMPILE_THREADS"
            )
            if explicit is None:
                val = str(max(1, int(ct)))
                os.environ["ENN_INDUCTOR_COMPILE_THREADS"] = val
                if env_str("TORCHINDUCTOR_COMPILE_THREADS") is None:
                    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = val

        try:
            wt = int(getattr(self, "ckpt_writer_threads", 0) or 0)
        except Exception:
            wt = 0
        if wt > 0:
            explicit = env_str("ENN_DCP_WRITER_THREADS") or env_str(
                "ENN_CKPT_WRITER_THREADS"
            )
            if explicit is None:
                val = str(max(1, int(wt)))
                os.environ["ENN_DCP_WRITER_THREADS"] = val
                if env_str("ENN_CKPT_WRITER_THREADS") is None:
                    os.environ["ENN_CKPT_WRITER_THREADS"] = val


@dataclass
class LoaderPolicy:
    max_batches_accel: int = 4
    max_batches_cpu: int = 2
    soft_cap_multiplier: int = 2

    def hard_inflight_batches(self: Self, device: torch.device | str) -> int:
        dev = (
            torch.device(device)
            if not isinstance(device, torch.device)
            else device
        )
        if dev.type in ("cuda", "xpu", "mps"):
            return max(1, int(self.max_batches_accel))
        return max(1, int(self.max_batches_cpu))

    def apply_soft_limits(
        self: Self, wp: WorkerPolicy, device: torch.device | str
    ) -> WorkerPolicy:
        hard = int(self.hard_inflight_batches(device))
        soft_cap = max(1, int(hard * max(1, int(self.soft_cap_multiplier))))
        prefetch_factor = max(1, int(getattr(wp, "prefetch_factor", 1) or 1))
        prebatch = max(1, int(getattr(wp, "prebatch", 1) or 1))
        num_workers_req = max(0, int(getattr(wp, "num_workers", 0) or 0))
        max_workers_inflight = max(
            0, int((soft_cap - prebatch) // max(1, prefetch_factor))
        )
        num_workers = min(num_workers_req, max_workers_inflight, soft_cap)
        num_workers = max(0, int(num_workers))
        inflight = int(num_workers) * int(prefetch_factor) + int(prebatch)
        if inflight > int(soft_cap) and num_workers > 0:
            prefetch_factor = max(
                1,
                int(
                    (int(soft_cap) - int(prebatch)) // max(1, int(num_workers))
                ),
            )
            max_workers_inflight = max(
                0, int((soft_cap - prebatch) // max(1, prefetch_factor))
            )
            num_workers = min(
                int(num_workers), int(max_workers_inflight), int(soft_cap)
            )
            num_workers = max(0, int(num_workers))
        wp.num_workers = int(num_workers)
        wp.prebatch = int(prebatch)
        wp.prefetch_factor = int(prefetch_factor)
        with contextlib.suppress(Exception):
            wp.max_concurrency = max(
                1,
                min(
                    int(getattr(wp, "max_concurrency", 1) or 1),
                    int(wp.num_workers) if int(wp.num_workers) > 0 else 1,
                    int(soft_cap),
                ),
            )
        return wp

    def wrap_input(
        self: Self,
        loader: Any,
        device: torch.device | str,
        *args: Any,
        name: str,
    ) -> Any:
        from .concurrency import new_prefetcher

        max_batches = self.hard_inflight_batches(device)
        with contextlib.suppress(Exception):
            dev = (
                torch.device(device)
                if not isinstance(device, torch.device)
                else device
            )
            if dev.type in {"cuda", "xpu", "mps"}:
                if bool(getattr(loader, "_non_blocking", False)) and hasattr(
                    loader, "_base_iterable"
                ):
                    with contextlib.suppress(Exception):
                        setattr(
                            loader, "_depth", int(max(1, int(max_batches)))
                        )
                    return loader
        return new_prefetcher(loader, max_batches=max_batches, name=name)


@dataclass
class BatchPolicy:
    sample_bytes: int
    host_sample_bytes: Optional[int] = None
    prebatch: int = 1
    prefetch_factor: int = 1
    num_workers: int = 0
    num_streams: int = 1
    max_concurrency: int = 1
    local_world_size: int = 1
    min_batch: int = 1
    max_batch: Optional[int] = None
    device_margin: float = 0.8
    host_margin: float = 0.8
    device_budget_ratio: float = 0.0
    device_budget_min_bytes: int = 0
    device_budget_max_bytes: Optional[int] = None
    host_budget_ratio: float = 0.0
    host_budget_min_bytes: int = 0
    host_budget_max_bytes: Optional[int] = None

    def __post_init__(self: Self) -> None:
        self.sample_bytes = max(int(self.sample_bytes or 0), 0)
        if self.host_sample_bytes is None:
            self.host_sample_bytes = self.sample_bytes
        self.prebatch = max(int(self.prebatch or 0), 0)
        self.prefetch_factor = max(int(self.prefetch_factor or 0), 0)
        self.num_workers = max(int(self.num_workers or 0), 0)
        self.num_streams = max(int(self.num_streams or 0), 0)
        self.max_concurrency = max(int(self.max_concurrency or 0), 0)
        self.min_batch = max(int(self.min_batch or 1), 1)
        if self.max_batch is not None:
            self.max_batch = max(int(self.max_batch), 1)
        self.device_margin = max(0.0, min(1.0, float(self.device_margin)))
        self.host_margin = max(0.0, min(1.0, float(self.host_margin)))
        self.device_budget_ratio = max(
            0.0, min(1.0, float(self.device_budget_ratio or 0.0))
        )
        self.host_budget_ratio = max(
            0.0, min(1.0, float(self.host_budget_ratio or 0.0))
        )
        self.device_budget_min_bytes = max(
            int(self.device_budget_min_bytes or 0), 0
        )
        self.host_budget_min_bytes = max(
            int(self.host_budget_min_bytes or 0), 0
        )
        if self.device_budget_max_bytes is not None:
            self.device_budget_max_bytes = max(
                int(self.device_budget_max_bytes), 0
            )
        if self.host_budget_max_bytes is not None:
            self.host_budget_max_bytes = max(
                int(self.host_budget_max_bytes), 0
            )

    def host_inflight_batches_per_proc(self: Self) -> int:
        return (
            max(1, self.max_concurrency) * max(1, self.prebatch)
            + max(1, self.prefetch_factor)
            + max(1, self.num_streams)
            + 1
        )

    @staticmethod
    def _budget_bytes(
        total_bytes: Optional[int],
        *args: Any,
        budget_ratio: float,
        budget_min_bytes: int,
        budget_max_bytes: Optional[int],
    ) -> int:
        total = int(total_bytes) if total_bytes is not None else 0
        ratio = float(budget_ratio or 0.0)
        base = int(float(total) * ratio) if total > 0 and ratio > 0.0 else 0
        budget = max(int(budget_min_bytes or 0), base)
        if (budget <= 0) and (total <= 0) and (budget_max_bytes is not None):
            budget = int(budget_max_bytes)
        elif budget_max_bytes is not None:
            budget = min(budget, int(budget_max_bytes))
        return max(0, int(budget))

    def suggest_batch(
        self: Self,
        *args: Any,
        dev_free: Optional[int] = None,
        host_free: Optional[int] = None,
        dev_total: Optional[int] = None,
        host_total: Optional[int] = None,
        local_world_size: Optional[int] = None,
    ) -> int:
        lw = (
            int(local_world_size)
            if local_world_size is not None
            else int(self.local_world_size or 1)
        )
        if lw <= 0:
            lw = 1
        use_dev_budget = (
            self.device_budget_ratio > 0.0
            or self.device_budget_min_bytes > 0
            or self.device_budget_max_bytes is not None
        )
        use_host_budget = (
            self.host_budget_ratio > 0.0
            or self.host_budget_min_bytes > 0
            or self.host_budget_max_bytes is not None
        )
        dev_cap: Optional[int] = None
        if dev_free is not None and dev_free >= 0 and self.sample_bytes > 0:
            denom = max(1, int(self.sample_bytes))
            usable = int(float(dev_free) * float(self.device_margin))
            if use_dev_budget:
                budget = self._budget_bytes(
                    dev_total,
                    budget_ratio=self.device_budget_ratio,
                    budget_min_bytes=self.device_budget_min_bytes,
                    budget_max_bytes=self.device_budget_max_bytes,
                )
                if budget > 0:
                    usable = min(int(usable), int(budget))
            dev_cap = int(max(0, usable) // denom)
        host_cap: Optional[int] = None
        if (
            host_free is not None
            and host_free >= 0
            and (self.host_sample_bytes or 0) > 0
        ):
            inflight = self.host_inflight_batches_per_proc()
            denom = (
                max(1, int(self.host_sample_bytes or 0))
                * max(1, inflight)
                * max(1, lw)
            )
            usable = int(float(host_free) * float(self.host_margin))
            if use_host_budget:
                budget = self._budget_bytes(
                    host_total,
                    budget_ratio=self.host_budget_ratio,
                    budget_min_bytes=self.host_budget_min_bytes,
                    budget_max_bytes=self.host_budget_max_bytes,
                )
                if budget > 0:
                    usable = min(int(usable), int(budget))
            host_cap = int(max(0, usable) // denom)
        candidates = [
            c for c in (dev_cap, host_cap) if isinstance(c, int) and c >= 0
        ]
        if not candidates:
            b = (
                self.max_batch
                if self.max_batch is not None
                else self.min_batch
            )
        else:
            b = min(candidates)
            if self.max_batch is not None:
                b = min(b, int(self.max_batch))
        b = max(int(b), int(self.min_batch))
        return max(1, b)


@dataclass
class ModelPolicy:
    @staticmethod
    def negotiate(
        device: Optional[Union[torch.device, str]] = None,
        *args: Any,
        metadata: Optional[Dataset[Any]] = None,
        **kwargs: Any,
    ) -> torch.dtype:
        from ..data.pipeline import Dataset

        dev = torch.device(device) if device is not None else get_device()
        candidates: List[torch.dtype] = []
        match dev.type:
            case "cuda":
                try:
                    if Dataset.is_cuda_bf16_supported(dev):
                        candidates.append(torch.bfloat16)
                except Exception:
                    pass
                candidates.extend((torch.float16, torch.float32))
            case "cpu":
                if Dataset.is_cpu_bf16_supported():
                    candidates.append(torch.bfloat16)
                candidates.extend((torch.float32, torch.float64))
            case "xpu":
                candidates.extend((torch.bfloat16, torch.float32))
            case "mps":
                candidates.extend((torch.float16, torch.float32))
            case _:
                candidates.append(torch.float32)
        for dtype in candidates:
            if is_scale_safe(dtype, metadata):
                return dtype
        return (
            torch.float64
            if is_scale_safe(torch.float64, metadata)
            else candidates[-1]
        )

    @staticmethod
    def _peek_layer(module: nn.Module) -> Optional[torch.Tensor]:
        with contextlib.suppress(StopIteration):
            return next(module.parameters())
        with contextlib.suppress(StopIteration):
            return next(module.buffers())
        return None

    @staticmethod
    def _coerce_metadata(
        model: nn.Module, metadata: Optional[Dataset[Any]] = None
    ) -> Dataset[Any]:
        from ..data.pipeline import Dataset

        Autocast.configure(model, metadata=metadata)
        meta = Autocast.metadata()
        if meta is None:
            ref = ModelPolicy._peek_layer(model)
            dev = ref.device if isinstance(ref, torch.Tensor) else get_device()
            meta = Dataset.for_device(dev)
            Autocast.configure(model, metadata=meta)
        return meta

    @staticmethod
    def _align_layers(
        src: nn.Module,
        dst: nn.Module,
        params_dtype: Optional[torch.dtype],
    ) -> None:
        ref = ModelPolicy._peek_layer(src)
        if ref is not None:
            with contextlib.suppress(Exception):
                dst.to(device=ref.device)
        if params_dtype is not None:
            with contextlib.suppress(Exception):
                dst.to(dtype=params_dtype)

    @staticmethod
    def _clone_state(
        src: nn.Module, dst: nn.Module, params_dtype: Optional[torch.dtype]
    ) -> None:
        try:
            state = src.state_dict()
        except (RuntimeError, AttributeError):
            return
        try:
            dst.load_state_dict(state, strict=False)
            return
        except Exception:
            pass
        ref = ModelPolicy._peek_layer(dst)
        device = ref.device if ref is not None else None
        converted: Dict[str, Any] = {}
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                converted[key] = value
                continue
            tensor = value.detach()
            if (
                params_dtype is not None
                and tensor.is_floating_point()
                and tensor.dtype != params_dtype
            ):
                with contextlib.suppress(Exception):
                    tensor = tensor.to(dtype=params_dtype)
            if (
                device is not None
                and getattr(tensor, "device", None) is not None
                and tensor.device != device
            ):
                with contextlib.suppress(Exception):
                    tensor = tensor.to(device=device)
            converted[key] = tensor
        with contextlib.suppress(Exception):
            dst.load_state_dict(converted, strict=False)

    @staticmethod
    def _nvidia_linear(
        module: nn.Linear,
        params_dtype: Optional[torch.dtype],
        te: Any,
    ) -> Optional[nn.Module]:
        te_linear = getattr(te, "Linear", None)
        if te_linear is None:
            return None
        kwargs: Dict[str, Any] = {
            "in_features": module.in_features,
            "out_features": module.out_features,
            "bias": module.bias is not None,
        }
        if params_dtype is not None:
            kwargs["params_dtype"] = params_dtype
        try:
            replacement = te_linear(**kwargs)
        except Exception:
            return None
        ModelPolicy._align_layers(module, replacement, params_dtype)
        ModelPolicy._clone_state(module, replacement, params_dtype)
        return replacement

    @staticmethod
    def _nvidia_layer_norm(
        module: nn.LayerNorm,
        params_dtype: Optional[torch.dtype],
        te: Any,
    ) -> Optional[nn.Module]:
        te_layer_norm = getattr(te, "LayerNorm", None)
        if te_layer_norm is None:
            return None
        kwargs: Dict[str, Any] = {
            "normalized_shape": module.normalized_shape,
            "eps": module.eps,
        }
        if params_dtype is not None:
            kwargs["params_dtype"] = params_dtype
        try:
            replacement = te_layer_norm(**kwargs)
        except Exception:
            return None
        ModelPolicy._align_layers(module, replacement, params_dtype)
        if module.elementwise_affine:
            ModelPolicy._clone_state(module, replacement, params_dtype)
        return replacement

    @staticmethod
    def _nvidia_rms_norm(
        module: nn.Module,
        params_dtype: Optional[torch.dtype],
        te: Any,
    ) -> Optional[nn.Module]:
        te_rms_norm = getattr(te, "RMSNorm", None)
        if te_rms_norm is None:
            return None
        kwargs: Dict[str, Any] = {
            "normalized_shape": getattr(module, "normalized_shape", None),
            "eps": getattr(module, "eps", 1e-5),
        }
        if kwargs["normalized_shape"] is None:
            return None
        if params_dtype is not None:
            kwargs["params_dtype"] = params_dtype
        try:
            replacement = te_rms_norm(**kwargs)
        except Exception:
            return None
        ModelPolicy._align_layers(module, replacement, params_dtype)
        ModelPolicy._clone_state(module, replacement, params_dtype)
        return replacement

    @staticmethod
    def _to_nvidia_layers(
        model: nn.Module,
        *args: Any,
        apply_te_linear: bool,
        apply_te_layer_norm: bool,
        apply_te_rms_norm: bool,
        filter_linear: Optional[Callable[[nn.Linear, str], bool]],
        params_dtype: Optional[torch.dtype],
        **kwargs: Any,
    ) -> Tuple[nn.Module, int]:
        try:
            import transformer_engine.pytorch as te
        except Exception:
            return (model, 0)

        def _convert(parent: nn.Module) -> int:
            converted = 0
            for name, child in list(parent.named_children()):
                replacement: Optional[nn.Module] = None
                if apply_te_linear and isinstance(child, nn.Linear):
                    if filter_linear is None or filter_linear(child, name):
                        replacement = ModelPolicy._nvidia_linear(
                            child, params_dtype, te
                        )
                elif apply_te_layer_norm and isinstance(child, nn.LayerNorm):
                    replacement = ModelPolicy._nvidia_layer_norm(
                        child, params_dtype, te
                    )
                else:
                    rms_cls = getattr(torch.nn, "RMSNorm", None)
                    if (
                        apply_te_rms_norm
                        and rms_cls is not None
                        and isinstance(child, rms_cls)
                    ):
                        replacement = ModelPolicy._nvidia_rms_norm(
                            child, params_dtype, te
                        )
                if replacement is not None:
                    setattr(parent, name, replacement)
                    converted += 1
                    continue
                converted += _convert(child)
            return converted

        count = _convert(model)
        if count:
            clear_model_cache(model)
        return (model, count)

    @staticmethod
    def _to_nvidia_attention(
        model: nn.Module,
        *args: Any,
        params_dtype: Optional[torch.dtype],
        **kwargs: Any,
    ) -> Tuple[nn.Module, int]:
        swapped = 0
        from ..nn.wrappers import _dot_product_attention_cls

        dot_cls = _dot_product_attention_cls()
        for module in model.modules():
            if (
                dot_cls is not None
                and isinstance(module, dot_cls)
                and getattr(module, "_te_ok", False)
            ):
                if not getattr(module, "te_first", False):
                    module.te_first = True
                swapped += 1
        if swapped:
            clear_model_cache(model)
        return (model, swapped)

    @staticmethod
    def use_nvidia_layers(
        model: nn.Module,
        device: Optional[Union[torch.device, str]] = None,
        *args: Any,
        metadata: Optional[Dataset[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> Tuple[nn.Module, bool, str]:
        from ..data.pipeline import Dataset

        dev = torch.device(device) if device is not None else get_device()
        if dev.type != "cuda":
            return (model, False, "Non-NVIDIA device; TE not applied")
        try:
            import transformer_engine.pytorch as te
        except Exception:
            return (model, False, "transformer_engine not installed")
        te_backend = getattr(te, "__name__", "transformer_engine.pytorch")
        fp8_ok, why = Dataset.is_float8_supported(dev)
        if fp8_ok:
            setattr(model, "__te_fp8_default__", True)
        params_dtype = kwargs.pop("params_dtype", None)
        if not isinstance(params_dtype, torch.dtype):
            params_dtype = ModelPolicy.negotiate(dev, metadata=metadata)
        if params_dtype is torch.float64:
            return (model, False, "TE disabled for fp64 params")
        model, n_layers = ModelPolicy._to_nvidia_layers(
            model,
            apply_te_linear=True,
            apply_te_layer_norm=True,
            apply_te_rms_norm=True,
            filter_linear=None,
            params_dtype=params_dtype,
        )
        try:
            model, attn_swapped = ModelPolicy._to_nvidia_attention(
                model, params_dtype=params_dtype
            )
        except Exception:
            attn_swapped = 0
        n_total = (n_layers or 0) + (attn_swapped or 0)
        _log_info(
            logger,
            f"[TE] swapped {n_total} modules (layers:{n_layers}, attn:{attn_swapped}); params_dtype={str(params_dtype).split('.')[-1]}, fp8={('on' if fp8_ok else 'off')} ({(why if fp8_ok else '')}), backend={te_backend}",
        )
        return (
            model,
            n_total > 0,
            f"TE applied (swapped {n_total}, layers={n_layers}, attn={attn_swapped}, dtype={params_dtype}, fp8={('on' if fp8_ok else 'off')}, backend={te_backend})",
        )

    @staticmethod
    def _enable_nvidia_training(
        model: nn.Module,
        params_dtype: torch.dtype,
        logger: Optional[Callable[[str], None]],
    ) -> Tuple[nn.Module, bool, str]:
        try:
            swapped_model, n = ModelPolicy._to_nvidia_layers(
                model,
                apply_te_linear=True,
                apply_te_layer_norm=True,
                apply_te_rms_norm=True,
                filter_linear=lambda lyr, _: lyr.in_features % 16 == 0
                and lyr.out_features % 16 == 0,
                params_dtype=params_dtype,
            )
            if n > 0:
                setattr(swapped_model, "__fp8_training_te__", True)
                if logger:
                    logger(f"[FP8][TE] swapped {n} modules")
                return (swapped_model, True, f"TE (swapped {n})")
            return (model, False, "TE present but no eligible modules")
        except Exception as exc:
            return (model, False, f"TE swap failed: {exc}")

    @staticmethod
    def _enable_torchao_training(
        model: nn.Module,
        logger: Optional[Callable[[str], None]],
    ) -> Tuple[nn.Module, bool, str]:
        try:
            from torchao.float8 import convert_to_float8_training

            res = convert_to_float8_training(model)
            converted = res or model
            setattr(converted, "__fp8_training_ao__", True)
            if logger:
                logger("[FP8][AO] convert_to_float8_training ok")
            return (converted, True, "torchao.float8")
        except Exception as exc:
            return (model, False, f"torchao convert failed: {exc}")

    @staticmethod
    def _enable_nvidia_inference(
        model: nn.Module,
        params_dtype: torch.dtype,
        logger: Optional[Callable[[str], None]],
    ) -> Tuple[nn.Module, bool, str]:
        try:
            swapped, n = ModelPolicy._to_nvidia_layers(
                model,
                apply_te_linear=True,
                apply_te_layer_norm=True,
                apply_te_rms_norm=True,
                filter_linear=lambda lyr, _: lyr.in_features % 16 == 0
                and lyr.out_features % 16 == 0,
                params_dtype=params_dtype,
            )
            if n > 0:
                setattr(swapped, "__fp8_inference_te__", True)
                if logger:
                    logger(
                        f"[FP8][TE] swapped {n} modules; using te.autocast"
                    )
                return (swapped, True, f"TE swap ({n})")
            return (model, False, "no eligible Linear (dims%16)")
        except Exception as exc:
            return (model, False, f"TE swap failed: {exc}")

    @staticmethod
    def _reuse_nvidia_layers(
        model: nn.Module,
        logger: Optional[Callable[[str], None]],
    ) -> Tuple[nn.Module, bool, str]:
        te_present = any(
            (
                getattr(module.__class__, "__module__", "").startswith(
                    "transformer_engine"
                )
                for module in model.modules()
            )
        )
        if te_present:
            setattr(model, "__fp8_inference_te__", True)
            clear_model_cache(model)
            if logger:
                logger("[FP8][TE] te.* already present; using te.autocast")
            return (model, True, "TE present")
        return (model, False, "TE layers not present")

    @staticmethod
    def _enable_torchao_inference(
        model: nn.Module,
        dynamic_activations: bool,
        logger: Optional[Callable[[str], None]],
    ) -> Tuple[nn.Module, bool, str]:
        try:
            dev = None
            with contextlib.suppress(Exception):
                for p in model.parameters():
                    if isinstance(p, torch.Tensor):
                        dev = p.device
                        break
            if dev is None:
                dev = torch.device("cpu")
            from .system import is_float8_supported

            ok, reason = is_float8_supported(dev)
            r = str(reason or "").lower()
            if (not bool(ok)) or ("fp8-ok:ao" not in r):
                return (
                    model,
                    False,
                    f"AO skipped (fp8 not validated for AO): {reason}",
                )

            from torchao.quantization import (
                Float8DynamicActivationFloat8WeightConfig,
                Float8WeightOnlyConfig,
                quantize_,
            )

            cfg = (
                Float8DynamicActivationFloat8WeightConfig()
                if dynamic_activations
                else Float8WeightOnlyConfig()
            )

            import inspect as _inspect

            sig = None
            with contextlib.suppress(Exception):
                sig = _inspect.signature(quantize_)

            def _eligible_linear(m: nn.Module) -> bool:
                if getattr(m, "_enn_no_ao_quant", False):
                    return False
                if not isinstance(m, nn.Linear):
                    return False
                inf = getattr(m, "in_features", None)
                outf = getattr(m, "out_features", None)
                if inf is None or outf is None:
                    return False
                inf = int(inf)
                outf = int(outf)
                if inf < 16 or outf < 16:
                    return False
                if (inf % 16) != 0 or (outf % 16) != 0:
                    return False
                return True

            used_filter = False
            if sig is not None and ("filter_fn" in sig.parameters):

                def filter_fn(mod: nn.Module, name: str) -> bool:
                    del name
                    return _eligible_linear(mod)

                quantize_(model, cfg, filter_fn=filter_fn)
                used_filter = True
            else:
                risky = False
                for m in model.modules():
                    if isinstance(m, nn.Linear):
                        if getattr(m, "_enn_no_ao_quant", False):
                            risky = True
                            break
                        inf = int(getattr(m, "in_features", 0) or 0)
                        outf = int(getattr(m, "out_features", 0) or 0)
                        if (
                            inf < 16
                            or outf < 16
                            or (inf % 16) != 0
                            or (outf % 16) != 0
                        ):
                            risky = True
                            break
                if risky:
                    return (
                        model,
                        False,
                        "AO skipped (quantize_ has no filter_fn; risky Linear present)",
                    )
                quantize_(model, cfg)

            def _is_torchao_float8_tensor(x: Any) -> bool:
                try:
                    t = type(x)
                    mod = getattr(t, "__module__", "") or ""
                    name = getattr(t, "__name__", "") or ""
                    if "torchao" in mod and "float8" in f"{mod}.{name}".lower():
                        return True
                    if "Float8Tensor" in name:
                        return True
                except Exception:
                    pass
                return False

            wrapped = 0
            linear_total = 0
            float8_linear_weights = 0
            float8_param_tensors = 0
            float8_buffer_tensors = 0
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    linear_total += 1
                    with contextlib.suppress(Exception):
                        w = getattr(m, "weight", None)
                        if w is not None and _is_torchao_float8_tensor(w):
                            float8_linear_weights += 1
                modm = getattr(m.__class__, "__module__", "")
                nm = m.__class__.__name__
                if modm.startswith("torchao") or ("Float8" in nm):
                    wrapped += 1

            for p in model.parameters():
                if _is_torchao_float8_tensor(p):
                    float8_param_tensors += 1

            for buf in model.buffers():
                if _is_torchao_float8_tensor(buf):
                    float8_buffer_tensors += 1

            if wrapped == 0 and float8_param_tensors == 0 and float8_linear_weights == 0 and float8_buffer_tensors == 0:
                return (
                    model,
                    False,
                    "AO quantize produced no torchao float8 modules/tensors",
                )

            warn_min = int(env_int("ENN_FP8_AO_WRAPPED_WARN_MIN", 256))
            warn_frac = float(env_float("ENN_FP8_AO_WRAPPED_WARN_FRAC", 0.90))
            frac = (float(wrapped) / float(max(1, linear_total))) if linear_total > 0 else 0.0
            if (wrapped >= warn_min) or (linear_total > 0 and frac >= warn_frac):
                _log_info(
                    logger,
                    f"[FP8][AO] WARNING: high quantized coverage "
                    f"(wrapped_modules={wrapped}, linear_total={linear_total}, frac={frac:.3f}); "
                    "verify filter_fn criteria / skip flags.",
                )

            setattr(model, "__fp8_inference_ao__", True)
            _log_info(
                logger,
                f"[FP8][AO] applied {cfg.__class__.__name__} "
                f"(filtered={used_filter}, wrapped={wrapped}, linear_total={linear_total}, "
                f"float8_linear_weights={float8_linear_weights}, float8_params={float8_param_tensors}, "
                f"float8_buffers={float8_buffer_tensors}, "
                f"frac={frac:.3f})",
            )
            return (model, True, "torchao")
        except Exception as exc:
            return (model, False, f"AO failed: {exc}")

    @staticmethod
    def enable_float8_training(
        model: nn.Module,
        metadata: Optional[Dataset[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[nn.Module, bool, str]:
        from ..data.pipeline import Dataset

        meta = ModelPolicy._coerce_metadata(model, metadata)
        device = torch.device(meta.device)
        if env_bool("ENN_DISABLE_FP8", default=False):
            Autocast.configure(model, metadata=meta)
            return (model, False, "disabled by ENN_DISABLE_FP8")
        ok, reason = Dataset.is_float8_supported(device)
        if not ok:
            Autocast.configure(model, metadata=meta)
            return (model, False, reason)
        if isinstance(reason, str) and ("te-only" in reason.lower()):
            _log_info(
                logger,
                f"[FP8] TE-only fallback enabled (torchao missing): {reason}",
            )
        if getattr(meta, "has_scale", False):
            float8_dtypes = Autocast.float8_formats()
            p2 = env_int(
                "ENN_FP8_SCALE_HEADROOM_POW2",
                env_int("ENN_FP8_HEADROOM_POW2", 1),
            )
            try:
                p2 = int(p2)
            except Exception:
                p2 = 1
            p2 = max(0, min(30, int(p2)))
            margin = float(2 ** p2)
            if not any(
                is_scale_safe(dtype, meta, safety_margin=margin)
                for dtype in float8_dtypes
            ):
                _log_info(
                    logger,
                    f"[FP8] training disabled: data scale exceeds float8 range (headroom=2^{p2})",
                )
                Autocast.configure(model, metadata=meta)
                return (model, False, "data scale")
        params_dtype = ModelPolicy.negotiate(device, metadata=meta)
        allow_ao = bool(
            env_bool(
                "ENN_FP8_ALLOW_AO_TRAINING",
                default=env_bool("ENN_FP8_ALLOW_AO", default=False),
            )
        )
        backends = ("te", "torchao") if allow_ao else ("te",)
        for backend in backends:
            if backend == "te":
                m2, ok2, why = ModelPolicy._enable_nvidia_training(
                    model, params_dtype, logger
                )
            else:
                m2, ok2, why = ModelPolicy._enable_torchao_training(
                    model, logger
                )
            if ok2:
                _log_info(
                    logger, f"[FP8] training enabled via {why} ({reason})"
                )
                Autocast.configure(m2, metadata=meta)
                return (m2, True, why)
            else:
                _log_debug(logger, f"[FP8] {backend} path skipped: {why}")
        Autocast.configure(model, metadata=meta)
        return (
            model,
            False,
            "No usable FP8 backend" + (" (torchao disabled)" if not allow_ao else ""),
        )

    @staticmethod
    def enable_float8_prediction(
        model: nn.Module,
        metadata: Optional[Dataset[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[nn.Module, bool, str]:
        from ..data.pipeline import Dataset

        meta = ModelPolicy._coerce_metadata(model, metadata)
        device = torch.device(meta.device)
        if env_bool("ENN_DISABLE_FP8", default=False):
            Autocast.configure(model, metadata=meta)
            return (model, False, "disabled by ENN_DISABLE_FP8")
        ok, reason = Dataset.is_float8_supported(device)
        if not ok:
            Autocast.configure(model, metadata=meta)
            return (model, False, reason)
        if isinstance(reason, str) and ("te-only" in reason.lower()):
            _log_info(
                logger,
                f"[FP8] TE-only fallback enabled (torchao missing): {reason}",
            )
        if getattr(meta, "has_scale", False):
            float8_dtypes = Autocast.float8_formats()
            p2 = env_int(
                "ENN_FP8_SCALE_HEADROOM_POW2",
                env_int("ENN_FP8_HEADROOM_POW2", 1),
            )
            try:
                p2 = int(p2)
            except Exception:
                p2 = 1
            p2 = max(0, min(30, int(p2)))
            margin = float(2 ** p2)
            if not any(
                is_scale_safe(dtype, meta, safety_margin=margin)
                for dtype in float8_dtypes
            ):
                _log_info(
                    logger,
                    f"[FP8] inference disabled: data scale exceeds float8 range (headroom=2^{p2})",
                )
                Autocast.configure(model, metadata=meta)
                return (model, False, "data scale")
        params_dtype = ModelPolicy.negotiate(device, metadata=meta)
        dynamic_activations = not (
            getattr(meta, "has_scale", False)
            and getattr(meta, "scale_is_integral", None) is True
        )
        allow_ao = bool(
            env_bool(
                "ENN_FP8_ALLOW_AO_INFERENCE",
                default=env_bool("ENN_FP8_ALLOW_AO", default=False),
            )
        )
        order = (
            ("te_swap", "te_present", "ao")
            if allow_ao
            else ("te_swap", "te_present")
        )
        for step in order:
            if step == "te_swap":
                m2, ok2, why = ModelPolicy._enable_nvidia_inference(
                    model, params_dtype, logger
                )
            elif step == "te_present":
                m2, ok2, why = ModelPolicy._reuse_nvidia_layers(model, logger)
            else:
                m2, ok2, why = ModelPolicy._enable_torchao_inference(
                    model, dynamic_activations, logger
                )
            if ok2:
                _log_info(
                    logger, f"[FP8] inference enabled via {why} ({reason})"
                )
                Autocast.configure(m2, metadata=meta)
                return (m2, True, why)
            else:
                _log_debug(logger, f"[FP8] {step} skipped: {why}")
        Autocast.configure(model, metadata=meta)
        return (
            model,
            False,
            "No usable FP8 backend" + (" (torchao disabled)" if not allow_ao else ""),
        )

    @staticmethod
    def enable_int8_training(
        model: nn.Module,
        metadata: Optional[Dataset[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[nn.Module, bool, str]:
        meta = ModelPolicy._coerce_metadata(model, metadata)
        device = torch.device(meta.device)
        with contextlib.suppress(Exception):
            model.to(device)
        dynamic_activations = not (
            getattr(meta, "has_scale", False)
            and getattr(meta, "scale_is_integral", None) is True
        )
        group_size = 128
        m2, ok, why = Quantization.enable_qat(
            model,
            dynamic_activations=dynamic_activations,
            group_size=group_size,
            logger=logger,
        )
        Autocast.configure(m2 if ok else model, metadata=meta)
        return (m2, ok, why)

    @staticmethod
    def enable_int8_prediction(
        model: nn.Module,
        metadata: Optional[Dataset[Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[nn.Module, bool, str]:
        meta = ModelPolicy._coerce_metadata(model, metadata)
        device = torch.device(meta.device)
        with contextlib.suppress(Exception):
            model.to(device)
        dynamic_activations = not (
            getattr(meta, "has_scale", False)
            and getattr(meta, "scale_is_integral", None) is True
        )
        m2, ok, why = Quantization._enable_ptq(
            model, dynamic_activations=dynamic_activations, logger=logger
        )
        Autocast.configure(m2 if ok else model, metadata=meta)
        return (m2, ok, why)


@dataclass
class PrecisionPolicy:
    master_float: torch.dtype = torch.float32
    amp_dtype: Optional[torch.dtype] = None
    fsdp_param_dtype: torch.dtype = torch.float32
    fsdp_reduce_dtype: torch.dtype = torch.float32
    fsdp_output_dtype: torch.dtype = torch.float32
    bn_buffers_dtype: torch.dtype = torch.float32
    underflow_action: str = "warn"

    @property
    def amp_float(self: Self) -> Optional[torch.dtype]:
        return self.amp_dtype

    @classmethod
    def from_metadata(
        cls: type[Self],
        device: Union[torch.device, str],
        metadata: Any | None,
        *args: Any,
        logger: Optional[logging.Logger] = None,
        safety_margin: float = 8.0,
    ) -> "PrecisionPolicy":
        dev = torch.device(device)
        meta = metadata
        if meta is None:
            meta = DeviceMeta.for_device(dev)
        else:
            with contextlib.suppress(Exception):
                setattr(meta, "device", dev)
                if callable(f := getattr(meta, "refresh", None)):
                    f()
        action = normalize_underflow_action(
            getattr(meta, "underflow_action", None),
            default=default_underflow_action(),
        )
        with contextlib.suppress(Exception):
            setattr(meta, "underflow_action", action)
        is_negotiable = bool(getattr(meta, "is_negotiable", False))
        safety = float(safety_margin)
        amp_dtype: Optional[torch.dtype] = None
        master_float = (
            torch.float32
            if is_negotiable
            or (
                dev.type not in ("cpu", "xpu", "mps")
                and is_scale_safe(torch.float32, meta, safety_margin=safety)
            )
            else torch.float64
        )
        match dev.type:
            case "cuda":
                if is_negotiable and is_scale_safe(
                    torch.float32, meta, safety_margin=safety
                ):
                    master_float = torch.float32
                if is_cuda_bf16_supported(dev) and is_scale_safe(
                    torch.bfloat16, meta, safety_margin=safety
                ):
                    amp_dtype = torch.bfloat16
                elif is_scale_safe(torch.float16, meta, safety_margin=safety):
                    amp_dtype = torch.float16
            case "xpu":
                amp_dtype = torch.bfloat16
            case "mps":
                amp_dtype = torch.float16
        fsdp_dt = (
            amp_dtype
            if master_float == torch.float32 and amp_dtype
            else master_float
        )
        return cls(
            master_float=master_float,
            amp_dtype=amp_dtype,
            fsdp_param_dtype=fsdp_dt,
            fsdp_reduce_dtype=fsdp_dt,
            fsdp_output_dtype=fsdp_dt,
            bn_buffers_dtype=master_float,
            underflow_action=str(action),
        )

    def to_fsdp_policy(self: Self) -> "MixedPrecisionPolicy":
        from torch.distributed.fsdp import MixedPrecisionPolicy

        return MixedPrecisionPolicy(
            param_dtype=self.fsdp_param_dtype,
            reduce_dtype=self.fsdp_reduce_dtype,
            output_dtype=self.fsdp_output_dtype,
            cast_forward_inputs=True,
        )


@dataclass
class CollectivePolicy:
    backend: str = "c10d"
    include_parameters: bool = True
    include_buffers: bool = True
    max_buffer_size_mb: int = 25
    coalesce_mb: int = 64
    max_tensor_mb_for_coalesce: int = 8
    inter_stream_mb: int = 16
    intra_stream_mb: int = 64
    max_inflight_mb: int = 64
    debug_collectives: bool = False
    verbose: bool = False

    @classmethod
    def from_env(cls: type[Self]) -> "CollectivePolicy":
        import os

        def getenv_bool_any(primary: str, legacy: str, default: bool) -> bool:
            v = os.getenv(primary)
            if v is None:
                v = os.getenv(legacy)
            if v is None:
                return default
            return v.strip().lower() in ("1", "true", "yes", "y", "on")

        def getenv_int_any(primary: str, legacy: str, default: int) -> int:
            v = os.getenv(primary)
            if v is None:
                v = os.getenv(legacy)
            if v is None:
                return default
            try:
                return int(v)
            except Exception:
                return default

        backend = (
            os.getenv(
                "ENN_COLLECTIVE_BACKEND",
                os.getenv("ENN_BCAST_BACKEND", "c10d"),
            )
            .strip()
            .lower()
        )
        include_parameters = getenv_bool_any(
            "ENN_COLLECTIVE_INCLUDE_PARAMETERS",
            "ENN_BCAST_INCLUDE_PARAMETERS",
            True,
        )
        include_buffers = getenv_bool_any(
            "ENN_COLLECTIVE_INCLUDE_BUFFERS",
            "ENN_BCAST_INCLUDE_BUFFERS",
            True,
        )
        max_buffer_size_mb = getenv_int_any(
            "ENN_COLLECTIVE_MAX_BUFFER_SIZE_MB",
            "ENN_BCAST_MAX_BUFFER_SIZE_MB",
            25,
        )
        coalesce_mb = getenv_int_any(
            "ENN_COLLECTIVE_COALESCE_MB", "ENN_BCAST_COALESCE_MB", 64
        )
        max_tensor_mb_for_coalesce = getenv_int_any(
            "ENN_COLLECTIVE_MAX_TENSOR_MB_FOR_COALESCE",
            "ENN_BCAST_MAX_TENSOR_MB_FOR_COALESCE",
            8,
        )
        inter_stream_mb = getenv_int_any(
            "ENN_COLLECTIVE_INTER_STREAM_MB",
            "ENN_BCAST_INTER_STREAM_MB",
            16,
        )
        intra_stream_mb = getenv_int_any(
            "ENN_COLLECTIVE_INTRA_STREAM_MB",
            "ENN_BCAST_INTRA_STREAM_MB",
            64,
        )
        max_inflight_mb = getenv_int_any(
            "ENN_COLLECTIVE_MAX_INFLIGHT_MB",
            "ENN_BCAST_MAX_INFLIGHT_MB",
            64,
        )
        debug_collectives = getenv_bool_any(
            "ENN_COLLECTIVE_DEBUG", "ENN_BCAST_DEBUG", False
        )
        verbose = getenv_bool_any(
            "ENN_COLLECTIVE_VERBOSE", "ENN_BCAST_VERBOSE", False
        )
        return cls(
            backend=backend,
            include_parameters=include_parameters,
            include_buffers=include_buffers,
            max_buffer_size_mb=max_buffer_size_mb,
            coalesce_mb=coalesce_mb,
            max_tensor_mb_for_coalesce=max_tensor_mb_for_coalesce,
            inter_stream_mb=inter_stream_mb,
            intra_stream_mb=intra_stream_mb,
            max_inflight_mb=max_inflight_mb,
            debug_collectives=debug_collectives,
            verbose=verbose,
        )


@dataclass
class DistributedPolicy:
    prefer_hsdp: bool = True
    prefer_ddp: bool = True
    sync_state: bool = True
    collective: CollectivePolicy = field(default_factory=CollectivePolicy)

    @classmethod
    def from_env(cls: type[Self]) -> "DistributedPolicy":
        import os

        def getenv_bool(name: str, default: bool) -> bool:
            v = os.getenv(name)
            if v is None:
                return default
            return v.strip().lower() in ("1", "true", "yes", "y", "on")

        prefer_hsdp = getenv_bool("ENN_DISTRIBUTED_PREFER_HSDP", True)
        prefer_ddp = getenv_bool("ENN_DISTRIBUTED_PREFER_DDP", True)
        sync_state = getenv_bool("ENN_DISTRIBUTED_SYNC_STATE", True)
        return cls(
            prefer_hsdp=prefer_hsdp,
            prefer_ddp=prefer_ddp,
            sync_state=sync_state,
            collective=CollectivePolicy.from_env(),
        )


from enum import Enum
import threading
import inspect


class AttentionBackend(str, Enum):
    FLEX = "flex"
    MHA = "mha"
    DPA = "dpa"


@dataclass(frozen=True)
class AttentionPlan:
    backend: AttentionBackend
    use_score_mod_for_bias: bool = False
    reason: str = ""


class AttentionPolicy:
    def __init__(self) -> None:
        order = (env_str("ENN_ATTENTION_ORDER") or "flex,mha,dpa").strip().lower()
        self.order: tuple[str, ...] = tuple(x.strip() for x in order.split(",") if x.strip())


    @staticmethod
    def _is_float8_dtype(dt: torch.dtype) -> bool:
        return "float8" in str(dt)

    @staticmethod
    def _is_torchao_float8_tensor(x: torch.Tensor) -> bool:
        try:
            t = type(x)
            mod = getattr(t, "__module__", "") or ""
            name = getattr(t, "__name__", "") or ""
            if "torchao" in mod and ("float8" in mod or "float8" in name.lower()):
                return True
            if "Float8Tensor" in name:
                return True
        except Exception:
            pass
        return False

    def plan(
        self,
        *,
        q: torch.Tensor,
        need_weights: bool = False,
        has_bias: bool = False,
        exporting: bool = False,
        compiling: bool = False,
    ) -> AttentionPlan:
        forced = (env_str("ENN_ATTENTION_BACKEND") or "").strip().lower()
        if forced in {"flex", "mha", "dpa"}:
            be = AttentionBackend(forced)
            return AttentionPlan(
                be,
                use_score_mod_for_bias=bool(be is AttentionBackend.FLEX and has_bias),
                reason="forced",
            )

        if q.dtype == torch.float64:
            flex_ok = False
        else:
            allow_flex_fp8 = bool(env_bool("ENN_FLEX_ALLOW_FP8", False))
            is_fp8 = self._is_float8_dtype(q.dtype) or self._is_torchao_float8_tensor(q)
            allow_fp32 = bool(env_bool("ENN_FLEX_ALLOW_FP32", True))
            flex_ok = (
                (not env_bool("ENN_DISABLE_FLEX_ATTENTION", False))
                and _HAS_TORCH_FLEX
                and getattr(q, "is_cuda", False)
                and (not exporting)
                and (not compiling)
                and (not need_weights)
                and (not is_fp8 or allow_flex_fp8)
                and (
                    q.dtype in (torch.float16, torch.bfloat16)
                    or (q.dtype == torch.float32 and allow_fp32)
                )
            )

        if has_bias and (not _FLEX_HAS_SCORE_MOD):
            flex_ok = False

        for be in self.order:
            if be == "flex" and flex_ok:
                return AttentionPlan(
                    AttentionBackend.FLEX,
                    use_score_mod_for_bias=bool(has_bias and _FLEX_HAS_SCORE_MOD),
                    reason=("auto:flex(score_mod)" if has_bias else "auto:flex"),
                )
            if be == "mha":
                return AttentionPlan(AttentionBackend.MHA, reason="auto:mha")
            if be == "dpa":
                return AttentionPlan(AttentionBackend.DPA, reason="auto:dpa")

        return AttentionPlan(AttentionBackend.MHA, reason="auto:default")


_FLEX_HAS_SCORE_MOD: bool = False
_HAS_TORCH_FLEX: bool = False
with contextlib.suppress(Exception):
    from torch.nn.attention.flex_attention import flex_attention as _fa

    _HAS_TORCH_FLEX = True
    _FLEX_HAS_SCORE_MOD = "score_mod" in set(
        inspect.signature(_fa).parameters.keys()
    )


_ATTN_POLICY_LOCK = threading.Lock()
_ATTN_POLICY: AttentionPolicy | None = None


def get_attention_policy() -> AttentionPolicy:
    global _ATTN_POLICY
    if _ATTN_POLICY is not None:
        return _ATTN_POLICY
    with _ATTN_POLICY_LOCK:
        if _ATTN_POLICY is None:
            _ATTN_POLICY = AttentionPolicy()
    return _ATTN_POLICY


ATTENTION_POLICY = get_attention_policy()
