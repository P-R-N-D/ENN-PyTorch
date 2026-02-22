from __future__ import annotations

import contextlib
import logging
import os
import time
from typing import Any, Dict

import torch
from ..core.concurrency import Mutex
from ..core.datatypes import env_bool, env_float, env_int
from ..core.precision import StatelessAutocast
from ..core.system import (
    accelerator_max_allocated_memory,
    allocated_accelerator_memory,
    flush_accelerator_memory_stats,
    is_pin_supported,
    sync_accelerator,
)
from ..core.tensor import to_device_recursive, to_torch_tensor, touch_tensors
from ..data.pipeline import Dataset
_LOGGER = logging.getLogger(__name__)


def _get_source_path(obj: object) -> str:
    if isinstance(obj, dict):
        if "path" in obj and "kind" in obj:
            return os.fspath(obj["path"])
        if obj:
            first = next(iter(obj.values()))
            return _get_source_path(first)
    if isinstance(obj, (list, tuple)) and obj:
        return _get_source_path(obj[0])
    raise RuntimeError("sources is empty or invalid")


class BatchThrottler:

    def __init__(self) -> None:
        self._oom_retry_count: Dict[tuple[int, str, int], int] = {}
        self._oom_retry_lock = Mutex()
        self._scale_log_last_s: Dict[tuple[int, str], float] = {}
        self._scale_log_lock = Mutex()

    @staticmethod
    def _oom_key(
        loader: object, phase: object, step: int
    ) -> tuple[int, str, int]:
        return (int(id(loader)), str(phase), int(step))

    def oom_retries(self, loader: object, phase: object, step: int) -> int:
        key = self._oom_key(loader, phase, int(step))
        with self._oom_retry_lock:
            cur = int(self._oom_retry_count.get(key, 0)) + 1
            self._oom_retry_count[key] = int(cur)
            return int(cur)

    def clear_oom_retries(
        self, loader: object, phase: object, step: int
    ) -> None:
        key = self._oom_key(loader, phase, int(step))
        with self._oom_retry_lock:
            self._oom_retry_count.pop(key, None)

    def oom_max_retries(self, phase: object) -> int:
        ph = str(phase).strip().lower()
        if ph == "train":
            v = env_int(
                "ENN_OOM_MAX_RETRIES_TRAIN",
                env_int("ENN_OOM_MAX_RETRIES_PER_BATCH", 4),
            )
        elif ph in {"val", "valid", "validation"}:
            v = env_int(
                "ENN_OOM_MAX_RETRIES_VAL",
                env_int("ENN_OOM_MAX_RETRIES_PER_BATCH", 2),
            )
        else:
            v = env_int("ENN_OOM_MAX_RETRIES_PER_BATCH", 3)
        return max(0, int(v))

    def get_oom_blocking_time(
        self, oom_try: int, phase: str | None = None
    ) -> float:
        del phase
        try:
            base_ms = float(env_float("ENN_OOM_BACKOFF_BASE_MS", 0.0))
        except Exception:
            base_ms = 0.0
        if base_ms <= 0.0:
            return 0.0
        try:
            max_ms = max(0.0, float(env_float("ENN_OOM_BACKOFF_MAX_MS", 50.0)))
        except Exception:
            max_ms = 50.0
        p = max(0, int(oom_try) - 2)
        sleep_ms = min(float(max_ms), float(base_ms) * (2.0 ** float(p)))
        return max(0.0, float(sleep_ms) / 1000.0)

    def log_scale_rate_throttled(
        self,
        *args: Any,
        logger: logging.Logger,
        scale_ctl: object,
        tag: object,
        msg: object,
        level: str = "info",
        min_interval_s: float | None = None,
    ) -> None:
        del args
        if min_interval_s is None:
            min_interval_s = env_float(
                "ENN_SAMPLER_SCALE_LOG_MIN_INTERVAL_S", 5.0
            )
        try:
            min_interval_s = float(min_interval_s)
        except Exception:
            min_interval_s = 5.0
        if min_interval_s < 0:
            min_interval_s = 0.0
        key = (int(id(scale_ctl)), str(tag))
        now = time.monotonic()
        with self._scale_log_lock:
            last = float(self._scale_log_last_s.get(key, 0.0))
            if min_interval_s and now - last < float(min_interval_s):
                return
            self._scale_log_last_s[key] = float(now)
        try:
            if str(level).lower() == "debug":
                logger.debug(str(msg))
            else:
                logger.info(str(msg))
        except Exception:
            pass


class BatchScaler:

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        self._logger = logger or _LOGGER

    @staticmethod
    def get_sampler_scaler(
        loader: object, *args: Any, max_depth: int = 4
    ) -> object:
        del args
        obj = loader
        try:
            depth = max(1, int(max_depth))
        except Exception:
            depth = 4
        for _ in range(depth):
            if obj is None:
                break
            ctl = getattr(obj, "_enn_sampler_scale", None)
            if ctl is not None:
                return ctl
            obj = getattr(obj, "_src", None) or getattr(obj, "src", None)
        return None

    @staticmethod
    def get_scale_rate_down(attempt: object) -> float:
        seq = (0.8, 0.7, 0.6, 0.5)
        idx = min(3, max(0, int(attempt) - 1))
        return float(seq[idx])

    def probe_per_sample_mem_bytes(
        self,
        model: object,
        device: torch.device,
        ops: object,
        dataset: object | None = None,
        *,
        max_probe_batch: int = 32,
        with_backward: bool = False,
        global_loss: object | None = None,
        local_loss: object | None = None,
        loss_weights: object | None = None,
    ) -> None:
        try:
            from ..data.nodes import Sampler
        except Exception:
            return

        try:
            in_dim = int(getattr(ops, "in_dim", 0) or 0)
        except Exception:
            in_dim = 0
        try:
            out_shape = tuple(getattr(ops, "out_shape", []) or [])
            out_dim = 1
            for d in out_shape:
                out_dim *= int(d)
        except Exception:
            out_dim = 1
        elem_size = torch.empty((), dtype=torch.float64).element_size()
        floor_bytes = (
            int((in_dim + out_dim) * elem_size * 10240)
            if in_dim + out_dim > 0
            else 0
        )
        dev_type = getattr(device, "type", "")
        if dev_type not in {"cuda", "xpu", "mps"}:
            return

        try:
            memmap_root = _get_source_path(getattr(ops, "sources", None))
            ds = Sampler(
                memmap_root,
                split="train",
                val_frac=float(getattr(ops, "val_frac", 0.0) or 0.0),
            )
        except Exception:
            return

        try:
            N = int(len(ds))
        except Exception:
            N = 0
        if N <= 0:
            return
        B0 = max(1, min(int(max_probe_batch), N))

        try:
            base_alloc = allocated_accelerator_memory(device)
            if base_alloc is None:
                return
            flush_accelerator_memory_stats(device)
            batch = ds.get(0, B0)

            forward_ran = False
            training_mode = bool(getattr(model, "training", False))
            meta = (
                dataset
                if isinstance(dataset, Dataset)
                else Dataset.for_device(device)
            )

            try:
                from ..nn.graph import inference_mode

                feats, labels, *_rest = meta.preprocess(
                    batch, return_keys=False
                )
                X = to_torch_tensor(feats)
                X = torch.atleast_2d(X)
                if X.dim() == 2 and int(X.shape[1]) == int(
                    getattr(ops, "in_dim", X.shape[1])
                ):
                    X = X.to(
                        device=device,
                        non_blocking=is_pin_supported(device.type),
                    )
                    if with_backward:
                        with contextlib.suppress(Exception):
                            model.train()
                        with StatelessAutocast.float(device):
                            Y_flat = None
                            if labels is not None:
                                Y = to_torch_tensor(labels)
                                Y = torch.atleast_2d(Y).to(
                                    device=device,
                                    non_blocking=is_pin_supported(device.type),
                                )
                                Y_flat = Y.reshape(Y.shape[0], -1)
                            y_hat, loss_val = model(
                                X,
                                labels_flat=Y_flat,
                                global_loss=global_loss,
                                local_loss=local_loss,
                                loss_weights=loss_weights,
                                calibrate_output=False,
                            )
                        target = None
                        if isinstance(loss_val, torch.Tensor):
                            target = loss_val
                        elif isinstance(y_hat, torch.Tensor):
                            target = y_hat
                        if target is not None:
                            loss = (
                                target if target.ndim == 0 else target.mean()
                            )
                            loss.backward()
                            forward_ran = True
                    else:
                        with inference_mode(model), StatelessAutocast.float(device):
                            warmup_iters = int(
                                env_int("ENN_SERVE_WARMUP_ITERS", 0) or 0
                            )
                            if (
                                warmup_iters <= 0
                                and str(getattr(ops, "mode", "") or "")
                                in ("predict", "infer")
                                and env_bool("ENN_MAX_PERF", True)
                            ):
                                m_eval = (
                                    model.module
                                    if hasattr(model, "module")
                                    else model
                                )
                                compiled = getattr(
                                    m_eval, "_compiled_submodules", None
                                )
                                warmup_iters = (
                                    3
                                    if (
                                        isinstance(compiled, dict)
                                        and any(
                                            bool(v) for v in compiled.values()
                                        )
                                    )
                                    else 1
                                )
                            warmup_iters = max(1, min(16, int(warmup_iters)))
                            for _i in range(warmup_iters):
                                _ = model(
                                    X,
                                    global_loss=None,
                                    local_loss=None,
                                    loss_weights=None,
                                    calibrate_output=True,
                                    return_loss=False,
                                )
                        forward_ran = True
            except Exception:
                forward_ran = False
            finally:
                if with_backward:
                    with contextlib.suppress(Exception):
                        model.zero_grad(set_to_none=True)
                if not training_mode:
                    with contextlib.suppress(Exception):
                        model.eval()

            if not forward_ran:
                batch_dev = to_device_recursive(batch, device)
                touch_tensors(batch_dev)

            with contextlib.suppress(Exception):
                sync_accelerator(device)

            peak_alloc = accelerator_max_allocated_memory(device)
            if peak_alloc is None:
                peak_alloc = allocated_accelerator_memory(device)
            if peak_alloc is None:
                return

            delta = max(0, int(peak_alloc) - int(base_alloc))
            if delta <= 0:
                return

            per_sample = int(delta // max(B0, 1))
            if floor_bytes > 0:
                per_sample = max(per_sample, floor_bytes)
            margin = 1.5 if with_backward else 1.2
            per_sample = int(per_sample * float(margin))
            if per_sample <= 0:
                return

            with contextlib.suppress(Exception):
                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    t = torch.tensor(
                        [int(per_sample)], device=device, dtype=torch.long
                    )
                    torch.distributed.all_reduce(
                        t, op=torch.distributed.ReduceOp.MAX
                    )
                    per_sample = int(t.item())

            with contextlib.suppress(Exception):
                Sampler._per_sample_mem_bytes = int(per_sample)
            with contextlib.suppress(Exception):
                os.environ["ENN_PER_SAMPLE_MEM_BYTES"] = str(int(per_sample))

        except Exception:
            return


class OOMHandler:

    def __init__(
        self,
        *,
        tuner: BatchScaler | None = None,
        throttler: BatchThrottler | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.tuner = tuner or BatchScaler(logger=logger)
        self.throttler = throttler or BatchThrottler()
        self.logger = logger or _LOGGER

    @staticmethod
    def is_batch_skippable(phase: object) -> bool:
        ph = str(phase).strip().lower()
        if ph == "train":
            return env_bool(
                "ENN_OOM_SKIP_TRAIN", env_bool("ENN_OOM_SKIP_BATCH", True)
            )
        if ph in {"val", "valid", "validation"}:
            return env_bool(
                "ENN_OOM_SKIP_VAL", env_bool("ENN_OOM_SKIP_BATCH", True)
            )
        return env_bool("ENN_OOM_SKIP_BATCH", True)

    def recover_oom(
        self,
        *args: Any,
        phase: str,
        loader: object,
        step_idx: int,
        device: torch.device,
        model: object,
        optimizer: Any | None = None,
        global_step: int | None = None,
        grad_accum_steps: int | None = None,
        min_grad_accum: int = 1,
    ) -> tuple[str, int | None]:
        del args
        try:
            from .distributed import broadcast_scalar
        except Exception:
            broadcast_scalar = None
        try:
            from ..core.system import empty_device_cache
        except Exception:
            empty_device_cache = None
        try:
            from ..nn.graph import to_checkpoint
            from ..nn.graph import to_submodule
        except Exception:
            to_checkpoint = None
            to_submodule = None

        ph = str(phase).strip().lower()
        oom_try = self.throttler.oom_retries(loader, ph, int(step_idx))
        max_tries = self.throttler.oom_max_retries(ph)
        log_fn = self.logger.error if oom_try <= 1 else self.logger.warning
        context = "Reducing MB/GA" if oom_try <= 1 else "Retrying"
        gs_info = (
            f" (global_step={global_step})" if global_step is not None else ""
        )
        log_fn(
            "[epochs] OOM in %s step %d%s. %s. (try=%d/%d)",
            str(ph),
            int(step_idx),
            gs_info,
            context,
            int(oom_try),
            int(max_tries),
        )
        if max_tries > 0 and oom_try > max_tries:
            if self.is_batch_skippable(ph):
                self.logger.error(
                    "[epochs] OOM storm: exceeded budget (%d/%d). Skipping.",
                    int(oom_try),
                    int(max_tries),
                )
                with contextlib.suppress(Exception):
                    self.throttler.clear_oom_retries(loader, ph, int(step_idx))
                if callable(empty_device_cache):
                    with contextlib.suppress(Exception):
                        empty_device_cache(
                            device=device, do_gc=False, min_interval_s=0.0
                        )
                if optimizer is not None:
                    with contextlib.suppress(Exception):
                        optimizer.zero_grad(set_to_none=True)
                return ("skip", grad_accum_steps)
            return ("raise", grad_accum_steps)

        scale_ctl = self.tuner.get_sampler_scaler(loader)
        if scale_ctl is not None:
            with contextlib.suppress(Exception):
                prev = float(scale_ctl.get())
                scale_ctl.request_scale_down(
                    self.tuner.get_scale_rate_down(oom_try)
                )
                cur = float(scale_ctl.get())
                if cur < prev:
                    self.throttler.log_scale_rate_throttled(
                        logger=self.logger,
                        scale_ctl=scale_ctl,
                        tag=f"oom-{ph}-scale-down",
                        msg=f"[epochs] scale down: {prev:.4f}->{cur:.4f}",
                        level="info",
                    )

        if callable(empty_device_cache):
            with contextlib.suppress(Exception):
                ec_min = 0.0 if oom_try <= 1 else 0.05
                empty_device_cache(
                    device=device, do_gc=False, min_interval_s=ec_min
                )

        if optimizer is not None:
            with contextlib.suppress(Exception):
                optimizer.zero_grad(set_to_none=True)

        inst_pressure = (
            to_submodule(model) if callable(to_submodule) else None
        ) or (model.module if hasattr(model, "module") else model)

        if (
            inst_pressure is not None
            and int(oom_try) <= 1
            and callable(to_checkpoint)
        ):
            cur_step_total = int(
                getattr(inst_pressure, "_enn_step_total", 0) or 0
            )
            if to_checkpoint(
                model,
                device=device,
                step_total=cur_step_total,
                ttl_steps=64,
                min_bytes=0,
            ):
                sleep_s = self.throttler.get_oom_blocking_time(oom_try, ph)
                if sleep_s > 0.0:
                    time.sleep(float(sleep_s))
                return ("retry", grad_accum_steps)

        reduced_any = False

        inst = to_submodule(model) if callable(to_submodule) else None
        if inst is not None:
            cur_mb = 0
            with contextlib.suppress(Exception):
                cur_mb = int(getattr(inst, "microbatch", 0) or 0)
            if cur_mb > 1:
                new_mb = max(1, cur_mb // 2)
                if callable(broadcast_scalar):
                    with contextlib.suppress(Exception):
                        new_mb = broadcast_scalar(new_mb, device=device, src=0)
                if new_mb < cur_mb:
                    with contextlib.suppress(Exception):
                        inst.microbatch = int(new_mb)
                        inst._auto_microbatch_pending = False
                    self.logger.info(
                        "[epochs] reduced microbatch %d->%d",
                        int(cur_mb),
                        int(new_mb),
                    )
                    reduced_any = True

        if ph == "train" and grad_accum_steps is not None:
            try:
                cur_ga = int(grad_accum_steps)
            except Exception:
                cur_ga = int(grad_accum_steps or 1)
            if cur_ga > int(min_grad_accum):
                new_ga = max(int(min_grad_accum), cur_ga // 2)
                if callable(broadcast_scalar):
                    with contextlib.suppress(Exception):
                        new_ga = broadcast_scalar(new_ga, device=device, src=0)
                if int(new_ga) != int(cur_ga):
                    self.logger.info(
                        "[epochs] reduced grad_accum %d->%d",
                        int(cur_ga),
                        int(new_ga),
                    )
                    grad_accum_steps = int(new_ga)
                    reduced_any = True

        if not reduced_any:
            if self.is_batch_skippable(ph):
                self.logger.error(
                    "[epochs] OOM in %s, no knobs. Skipping.", str(ph)
                )
                with contextlib.suppress(Exception):
                    self.throttler.clear_oom_retries(loader, ph, int(step_idx))
                if callable(empty_device_cache):
                    with contextlib.suppress(Exception):
                        empty_device_cache(
                            device=device, do_gc=False, min_interval_s=0.0
                        )
                if optimizer is not None:
                    with contextlib.suppress(Exception):
                        optimizer.zero_grad(set_to_none=True)
                return ("skip", grad_accum_steps)
            self.logger.error(
                "[epochs] OOM in %s, no knobs. Giving up.", str(ph)
            )
            return ("raise", grad_accum_steps)

        sleep_s = self.throttler.get_oom_blocking_time(oom_try, ph)
        if sleep_s > 0.0:
            with contextlib.suppress(Exception):
                time.sleep(float(sleep_s))
        return ("retry", grad_accum_steps)


_DEFAULT_THROTTLER = BatchThrottler()
_DEFAULT_TUNER = BatchScaler(logger=_LOGGER)
_DEFAULT_OOM_HANDLER = OOMHandler(
    tuner=_DEFAULT_TUNER, throttler=_DEFAULT_THROTTLER, logger=_LOGGER
)


def probe_per_sample_mem_bytes(
    model: object,
    device: torch.device,
    ops: object,
    dataset: object | None = None,
    *,
    max_probe_batch: int = 32,
    with_backward: bool = False,
    global_loss: object | None = None,
    local_loss: object | None = None,
    loss_weights: object | None = None,
) -> None:
    return _DEFAULT_TUNER.probe_per_sample_mem_bytes(
        model=model,
        device=device,
        ops=ops,
        dataset=dataset,
        max_probe_batch=max_probe_batch,
        with_backward=with_backward,
        global_loss=global_loss,
        local_loss=local_loss,
        loss_weights=loss_weights,
    )


def get_sampler_scaler(
    loader: object, *args: Any, max_depth: int = 4
) -> object:
    return _DEFAULT_TUNER.get_sampler_scaler(
        loader, *args, max_depth=max_depth
    )


def log_scale_rate_throttled(
    *args: Any,
    logger: logging.Logger,
    scale_ctl: object,
    tag: object,
    msg: object,
    level: str = "info",
    min_interval_s: float | None = None,
) -> None:
    return _DEFAULT_THROTTLER.log_scale_rate_throttled(
        *args,
        logger=logger,
        scale_ctl=scale_ctl,
        tag=tag,
        msg=msg,
        level=level,
        min_interval_s=min_interval_s,
    )


def clear_oom_retries(loader: object, phase: object, step: int) -> None:
    return _DEFAULT_THROTTLER.clear_oom_retries(loader, phase, int(step))


def recover_oom(
    *args: Any,
    phase: str,
    loader: object,
    step_idx: int,
    device: torch.device,
    model: object,
    optimizer: Any | None = None,
    global_step: int | None = None,
    grad_accum_steps: int | None = None,
    min_grad_accum: int = 1,
) -> tuple[str, int | None]:
    return _DEFAULT_OOM_HANDLER.recover_oom(
        *args,
        phase=phase,
        loader=loader,
        step_idx=step_idx,
        device=device,
        model=model,
        optimizer=optimizer,
        global_step=global_step,
        grad_accum_steps=grad_accum_steps,
        min_grad_accum=min_grad_accum,
    )
