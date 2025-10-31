# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, TypeAlias, Union

from torch import nn, optim


@contextlib.contextmanager
def no_synchronization(
    model: nn.Module,
    *,
    enable: bool = True,
) -> contextlib.AbstractContextManager[None]:
    if not enable:
        yield
        return
    ctx = None
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


try:
    from torch.distributed.algorithms.join import Join as _TorchJoin
except ImportError:
    _TorchJoin = None

Join: type[AbstractContextManager[None]] | None = _TorchJoin

if TYPE_CHECKING:
    from torch.distributed._composable.fsdp import FSDPModule
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.nn.parallel import DistributedDataParallel as DDP
else:
    DDP = object
    FSDP = object
    FSDPModule = object

JoinableModel: TypeAlias = Union["DDP", "FSDP", "FSDPModule"]

def _has_join_hook(obj: Any | None) -> bool:
    if obj is None:
        return False
    return getattr(obj, "join_hook", None) is not None


def joining(
    model: JoinableModel,
    optimizer: optim.Optimizer | None = None,
) -> AbstractContextManager[None]:
    if Join is None:
        return contextlib.nullcontext()
    joinables = tuple(obj for obj in (model, optimizer) if _has_join_hook(obj))
    if not joinables:
        return contextlib.nullcontext()
    return Join(joinables, throw_on_early_termination=True)


__all__ = [
    "Join",
    "JoinableModel",
    "joining",
    "no_synchronization",
]

