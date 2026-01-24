# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import threading
import traceback
from typing import Any, Callable, Iterator

import torch
from torch import nn

from .datatypes import env_bool, env_first
from .distributed import broadcast_scalar, is_dtensor_active
from .graph import is_export_or_trace, to_submodule

_CKPT_TL = threading.local()


def _raised_from_checkpointed_fn(err: BaseException) -> bool:
    tb = err.__traceback__
    if tb is None:
        return False
    for frame, _ in traceback.walk_tb(tb):
        if (
            frame.f_code.co_name == "_state"
            and frame.f_globals.get("__name__") == __name__
        ):
            return True
    return False


def iter_checkpoint(root: nn.Module) -> Iterator[nn.Module]:
    try:
        import torch.nn as nn

        if isinstance(root, nn.Module):
            for mod in root.modules():
                if hasattr(mod, "_ckpt_min_bytes") and hasattr(
                    mod, "_ckpt_enabled"
                ):
                    yield mod
    except Exception:
        return


def to_checkpoint(
    model: object,
    *args: Any,
    device: torch.device,
    step_total: int,
    ttl_steps: int,
    min_bytes: int,
) -> bool:
    inst = to_submodule(model) or (
        model.module if hasattr(model, "module") else model
    )
    if inst is None:
        return False
    try:
        ttl_steps = max(1, int(ttl_steps))
        min_bytes = max(0, int(min_bytes))
        step_total = max(0, int(step_total))
    except Exception:
        return False
    until = step_total + ttl_steps
    try:
        until = int(broadcast_scalar(until, device=device, src=0))
        min_bytes = int(broadcast_scalar(min_bytes, device=device, src=0))
    except Exception:
        pass
    cur_until = int(getattr(inst, "_enn_ckpt_pressure_until", 0) or 0)
    if (
        cur_until >= until
        and int(getattr(inst, "_enn_ckpt_pressure_min_bytes", 0) or 0)
        <= min_bytes
    ):
        return False
    changed = False
    for mod in iter_checkpoint(inst):
        if not hasattr(mod, "_enn_ckpt_saved_min_bytes"):
            with contextlib.suppress(Exception):
                setattr(
                    mod,
                    "_enn_ckpt_saved_min_bytes",
                    int(getattr(mod, "_ckpt_min_bytes", 0) or 0),
                )
                setattr(
                    mod,
                    "_enn_ckpt_saved_enabled",
                    bool(getattr(mod, "_ckpt_enabled", True)),
                )
        try:
            cur = int(getattr(mod, "_ckpt_min_bytes", 0) or 0)
            if min_bytes < cur:
                setattr(mod, "_ckpt_min_bytes", int(min_bytes))
                changed = True
            if not bool(getattr(mod, "_ckpt_enabled", True)):
                setattr(mod, "_ckpt_enabled", True)
                changed = True
        except Exception:
            pass
    with contextlib.suppress(Exception):
        setattr(inst, "_enn_ckpt_pressure_until", int(max(cur_until, until)))
        prev_mb = int(getattr(inst, "_enn_ckpt_pressure_min_bytes", 0) or 0)
        if prev_mb <= 0:
            setattr(inst, "_enn_ckpt_pressure_min_bytes", int(min_bytes))
        else:
            setattr(
                inst,
                "_enn_ckpt_pressure_min_bytes",
                int(min(prev_mb, min_bytes)),
            )
    return bool(changed)


def from_checkpoint(model: nn.Module, *args: Any, step_total: int) -> None:
    inst = to_submodule(model) or (
        model.module if hasattr(model, "module") else model
    )
    if inst is None:
        return
    try:
        step_total = int(step_total)
    except Exception:
        return
    until = int(getattr(inst, "_enn_ckpt_pressure_until", 0) or 0)
    if until <= 0 or step_total < until:
        return
    for mod in iter_checkpoint(inst):
        try:
            if hasattr(mod, "_enn_ckpt_saved_min_bytes"):
                setattr(
                    mod,
                    "_ckpt_min_bytes",
                    int(getattr(mod, "_enn_ckpt_saved_min_bytes", 0) or 0),
                )
            if hasattr(mod, "_enn_ckpt_saved_enabled"):
                setattr(
                    mod,
                    "_ckpt_enabled",
                    bool(getattr(mod, "_enn_ckpt_saved_enabled", True)),
                )
            for k in (
                "_enn_ckpt_saved_min_bytes",
                "_enn_ckpt_saved_enabled",
            ):
                with contextlib.suppress(Exception):
                    delattr(mod, k)
        except Exception:
            pass
    with contextlib.suppress(Exception):
        setattr(inst, "_enn_ckpt_pressure_until", 0)
        setattr(inst, "_enn_ckpt_pressure_min_bytes", 0)


def is_checkpoint() -> bool:
    return bool(getattr(_CKPT_TL, "depth", 0) or 0)


def coerce_checkpoint(
    fn: Callable[..., Any],
    *args: Any,
    **ckpt_kwargs: Any,
) -> Any:
    if _TORCH_CHECKPOINT is None:
        return fn(*args)
    if is_export_or_trace() or not any(
        isinstance(a, torch.Tensor) and a.requires_grad for a in args
    ):
        return fn(*args)
    force_reentrant = env_first(
        ("ENN_CKPT_REQUIRE_REENTRANT",), default=None
    )
    require_reentrant = (
        env_bool("ENN_CKPT_REQUIRE_REENTRANT", default=False)
        if force_reentrant is not None
        else bool(is_dtensor_active())
    )

    use_reentrant = ckpt_kwargs.pop("use_reentrant", None)
    preserve_rng_state = ckpt_kwargs.pop("preserve_rng_state", None)
    determinism_check = ckpt_kwargs.pop("determinism_check", None)

    if use_reentrant is None:
        use_reentrant = True
    if require_reentrant:
        use_reentrant = True
    if preserve_rng_state is None:
        preserve_rng_state = True

    ck_opts = {
        k: v
        for k, v in [
            ("use_reentrant", use_reentrant),
            ("preserve_rng_state", preserve_rng_state),
            ("determinism_check", determinism_check),
        ]
        if v is not None
    }

    tried: set[tuple[tuple[str, object], ...]] = set()
    last_type_error: TypeError | None = None

    opts_list: list[dict[str, object]] = [
        ck_opts,
        {k: v for k, v in ck_opts.items() if k != "determinism_check"},
    ]
    if require_reentrant:
        opts_list.extend(
            [{k: v for k, v in ck_opts.items() if k == "use_reentrant"}]
        )
    else:
        opts_list.extend(
            [
                {
                    k: v
                    for k, v in ck_opts.items()
                    if k not in ("determinism_check", "use_reentrant")
                },
                {k: v for k, v in ck_opts.items() if k != "use_reentrant"},
                {},
            ]
        )

    for opts in opts_list:
        key = tuple(sorted(opts.items()))
        if key in tried:
            continue
        tried.add(key)
        try:
            return checkpoint(fn, *args, **opts, **ckpt_kwargs)
        except TypeError as e:
            if require_reentrant and _raised_from_checkpointed_fn(e):
                raise
            last_type_error = e
            continue

    if require_reentrant:
        raise TypeError(
            "DTensor/FSDP2 checkpointing requires `use_reentrant=True`, but torch.utils.checkpoint.checkpoint did not accept a compatible signature in this runtime. Upgrade PyTorch or set ENN_CKPT_REQUIRE_REENTRANT=0 to override."
        ) from last_type_error
    return checkpoint(fn, *args, **ckpt_kwargs)


def checkpoint(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    if _TORCH_CHECKPOINT is None:
        return fn(*args, **kwargs)
    tl = _CKPT_TL

    def _state(*a: Any, **k: Any) -> Any:
        depth = int(getattr(tl, "depth", 0) or 0)
        setattr(tl, "depth", depth + 1)
        try:
            return fn(*a, **k)
        finally:
            setattr(tl, "depth", depth)

    return _TORCH_CHECKPOINT(_state, *args, **kwargs)


try:
    import torch.utils.checkpoint
except Exception:
    _TORCH_CHECKPOINT = None
else:
    _TORCH_CHECKPOINT = torch.utils.checkpoint.checkpoint
