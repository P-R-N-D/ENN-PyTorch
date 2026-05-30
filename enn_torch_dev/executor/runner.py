from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .plan import ExecutorPlan
from .store import KVStore


@dataclass(slots=True)
class ExecutorRunner:
    """
    Execute an already-validated ``ExecutorPlan``.

    This runner does not build graphs or compose nested pipelines. It only calls
    the top-level executor component selected by the plan. When stream mode is
    enabled, ``stream_pipeline`` is the executable outer layer; any tiled or
    global/local behavior must already be represented inside the caller-supplied
    stream pipeline.
    """

    plan: ExecutorPlan

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ExecutorPlan):
            raise TypeError(f"ExecutorRunner.plan must be ExecutorPlan, got {type(self.plan)!r}")

    def run(
        self,
        store: KVStore,
        *,
        chunks: Sequence[Any] | None = None,
    ) -> Any:
        if not isinstance(store, KVStore):
            raise TypeError(f"ExecutorRunner.run expects KVStore, got {type(store)!r}")

        if self.plan.mode.stream:
            if chunks is None:
                raise ValueError("ExecutorRunner stream mode requires chunks.")
            assert self.plan.stream_pipeline is not None
            return self.plan.stream_pipeline.run(store, chunks)

        if chunks is not None:
            raise ValueError("ExecutorRunner chunks are only valid for stream mode.")

        if self.plan.mode.global_local:
            assert self.plan.global_local_pipeline is not None
            return self.plan.global_local_pipeline.run(store)

        if self.plan.mode.tile:
            assert self.plan.tile_pipeline is not None
            return self.plan.tile_pipeline.run(store)

        if self.plan.mode.is_plain:
            assert self.plan.graph is not None
            return self.plan.graph.run(store)

        raise RuntimeError(f"Unsupported ExecutorPlan mode: {self.plan.mode!r}")
