# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import inspect
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Iterable, Optional, TypeAlias, Union

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

try:  # pragma: no cover - optional dependency
    from torch.distributed.algorithms.join import Join as _TorchJoin
except ImportError:  # pragma: no cover - optional dependency
    _TorchJoin = None

Join: type[AbstractContextManager[None]] | None = _TorchJoin

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from torch.distributed._composable.fsdp import FSDPModule
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
else:  # pragma: no cover - fallback when optional deps missing
    FSDP = object
    FSDPModule = object

JoinableModel: TypeAlias = Union["DDP", "FSDP", "FSDPModule"]


__all__ = [
    "Join",
    "JoinableModel",
    "broadcast_model_states",
    "distributed_barrier",
    "is_dist_avail_and_initialized",
    "joining",
    "no_synchronization",
    "wrap_ddp_if_needed",
    "wrap_fsdp_module",
]


@contextlib.contextmanager
def no_synchronization(
    model: torch.nn.Module,
    *,
    enable: bool = True,
) -> AbstractContextManager[None]:
    if not enable:
        yield
        return

    ctx: AbstractContextManager[None] | None = None
    try:
        no_sync = getattr(model, "no_sync", None)
        if callable(no_sync):
            ctx = no_sync()
    except Exception:
        ctx = None

    if ctx is None:
        yield
        return

    with ctx:
        yield


def _has_join_hook(obj: Any | None) -> bool:
    if obj is None:
        return False
    return getattr(obj, "join_hook", None) is not None


def joining(
    model: JoinableModel,
    optimizer: Optimizer | None = None,
) -> AbstractContextManager[None]:
    if Join is None:
        return contextlib.nullcontext()

    joinables = tuple(obj for obj in (model, optimizer) if _has_join_hook(obj))
    if not joinables:
        return contextlib.nullcontext()

    return Join(joinables, throw_on_early_termination=True)  # type: ignore[arg-type]


def is_dist_avail_and_initialized() -> bool:
    """Return ``True`` when torch.distributed is available and initialized."""

    try:
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def _device_ids_from(device: Optional[torch.device]) -> Optional[Iterable[int]]:
    if device is None:
        return None
    dev_type = getattr(device, "type", None)
    if dev_type not in {"cuda", "xpu"}:
        return None
    index = getattr(device, "index", None)
    if index is None:
        return None
    return [int(index)]


def distributed_barrier(device: Optional[torch.device] = None) -> None:
    """Synchronize all ranks if the process group is initialized."""

    if not is_dist_avail_and_initialized():
        return
    try:
        dist.barrier(device_ids=_device_ids_from(device))
    except TypeError:
        # ``device_ids`` keyword is not supported by all backends.
        dist.barrier()


def broadcast_model_states(
    module: torch.nn.Module,
    *,
    src: int = 0,
) -> None:
    """Broadcast parameters and buffers from ``src`` to all other ranks.

    The helper is aware of DTensor instances and re-wraps the broadcasted
    local shard to preserve placements.
    """

    if not is_dist_avail_and_initialized():
        return

    try:
        from torch.distributed._tensor import DTensor  # type: ignore
    except Exception:  # pragma: no cover - DTensor optional
        DTensor = tuple()  # type: ignore[assignment]

    for buffer in module.buffers(recurse=True):
        data = getattr(buffer, "data", None)
        if not isinstance(data, torch.Tensor):
            continue
        try:
            dist.broadcast(data, src=src)
        except Exception:
            continue

    for param in module.parameters(recurse=True):
        data = getattr(param, "data", None)
        if not isinstance(data, torch.Tensor):
            continue
        try:
            if isinstance(data, DTensor):  # type: ignore[arg-type]
                local = data.to_local()
                dist.broadcast(local, src=src)
                param.data = type(data).from_local(  # type: ignore[assignment]
                    local,
                    device_mesh=data.device_mesh,
                    placements=tuple(data.placements),
                    run_check=False,
                )
            else:
                dist.broadcast(data, src=src)
        except Exception:
            continue


def wrap_ddp_if_needed(
    module: torch.nn.Module,
    *,
    device: torch.device,
) -> torch.nn.Module:
    """Wrap ``module`` with :class:`~torch.nn.parallel.DistributedDataParallel`.

    The wrapper is only applied when the default process group is initialized.
    The module is always moved to ``device``.
    """

    module = module.to(device)
    if not is_dist_avail_and_initialized():
        return module
    if isinstance(module, DDP):
        return module

    device_ids = _device_ids_from(device)
    ddp_kwargs = {
        "broadcast_buffers": True,
        "find_unused_parameters": False,
    }
    if device_ids is not None:
        ddp_kwargs["device_ids"] = list(device_ids)
    return DDP(module, **ddp_kwargs)


def wrap_fsdp_module(
    module: torch.nn.Module,
    *,
    mesh,
    mp_policy=None,
    reshard_after_forward: bool = False,
    sync_module_states: bool = True,
    ignored_params=None,
):
    """Wrap ``module`` using ``torch.distributed.fsdp.fully_shard``.

    This helper ensures ``requires_gradient_sync`` is enabled on the returned
    FSDP wrapper so gradient synchronization happens by default.
    Additional keyword arguments mirror those accepted by ``fully_shard``.
    """

    sig = inspect.signature(fully_shard)
    params = sig.parameters

    args = [module]
    kwargs = {}

    if "mesh" in params:
        kwargs["mesh"] = mesh
    elif "process_group" in params and mesh is not None:
        kwargs["process_group"] = mesh

    if "mp_policy" in params and mp_policy is not None:
        kwargs["mp_policy"] = mp_policy
    if "reshard_after_forward" in params:
        kwargs["reshard_after_forward"] = reshard_after_forward
    if "sync_module_states" in params:
        kwargs["sync_module_states"] = sync_module_states
    if "ignored_params" in params and ignored_params is not None:
        kwargs["ignored_params"] = ignored_params

    sharded = fully_shard(*args, **kwargs)
    try:
        sharded.set_requires_gradient_sync(True)
    except AttributeError:
        pass
    return sharded

