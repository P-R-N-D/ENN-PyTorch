from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from torch import nn

from .global_local import GlobalLocalPipeline
from .graph import GraphExecutor
from .model_spec import ModelExecutionSpec
from .plan import ExecutorPlan
from .runner import ExecutorRunner
from .store import KVStore
from .stream import StreamPipeline
from .tile_pipeline import TilePipeline


@dataclass(slots=True)
class ExecutorModel:
    """
    Thin executor-layer model wrapper.

    ``ExecutorModel`` binds the public ``ModelExecutionSpec`` naming layer to a
    validated ``ExecutorPlan`` and delegates execution to ``ExecutorRunner``. It
    does not build graphs, create pipelines, own parameters, or implement a
    training loop.
    """

    spec: ModelExecutionSpec
    plan: ExecutorPlan
    runner: ExecutorRunner = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ModelExecutionSpec):
            raise TypeError(
                f"ExecutorModel.spec must be ModelExecutionSpec, got {type(self.spec)!r}"
            )
        if not isinstance(self.plan, ExecutorPlan):
            raise TypeError(
                f"ExecutorModel.plan must be ExecutorPlan, got {type(self.plan)!r}"
            )
        if self.plan.mode != self.spec.executor_mode:
            raise ValueError(
                "ExecutorModel plan mode must match spec.executor_mode: "
                f"{self.plan.mode!r} != {self.spec.executor_mode!r}"
            )

        self.runner = ExecutorRunner(self.plan)

    @classmethod
    def from_components(
        cls,
        spec: ModelExecutionSpec,
        *,
        graph: GraphExecutor | None = None,
        tile_pipeline: TilePipeline | None = None,
        stream_pipeline: StreamPipeline | None = None,
        global_local_pipeline: GlobalLocalPipeline | None = None,
    ) -> "ExecutorModel":
        if not isinstance(spec, ModelExecutionSpec):
            raise TypeError(
                f"ExecutorModel.from_components spec must be ModelExecutionSpec, got {type(spec)!r}"
            )
        plan = spec.create_plan(
            graph=graph,
            tile_pipeline=tile_pipeline,
            stream_pipeline=stream_pipeline,
            global_local_pipeline=global_local_pipeline,
        )
        return cls(spec=spec, plan=plan)

    def run(
        self,
        store: KVStore,
        *,
        chunks: Sequence[Any] | None = None,
    ) -> Any:
        return self.runner.run(store, chunks=chunks)


class Model(nn.Module):
    """
    Thin public ``torch.nn.Module`` adapter around ``ExecutorModel``.

    ``Model`` owns the PyTorch-facing ``forward(...)`` entry point and delegates
    execution to an already-wired ``ExecutorModel``. It does not build graphs,
    create pipelines, infer state routes, chunk streams, or own training policy.
    """

    def __init__(self, executor_model: ExecutorModel) -> None:
        super().__init__()
        if not isinstance(executor_model, ExecutorModel):
            raise TypeError(
                f"Model.executor_model must be ExecutorModel, got {type(executor_model)!r}"
            )
        self.executor_model = executor_model

    @classmethod
    def from_components(
        cls,
        spec: ModelExecutionSpec,
        *,
        graph: GraphExecutor | None = None,
        tile_pipeline: TilePipeline | None = None,
        stream_pipeline: StreamPipeline | None = None,
        global_local_pipeline: GlobalLocalPipeline | None = None,
    ) -> "Model":
        executor_model = ExecutorModel.from_components(
            spec,
            graph=graph,
            tile_pipeline=tile_pipeline,
            stream_pipeline=stream_pipeline,
            global_local_pipeline=global_local_pipeline,
        )
        return cls(executor_model)

    @property
    def spec(self) -> ModelExecutionSpec:
        return self.executor_model.spec

    @property
    def plan(self) -> ExecutorPlan:
        return self.executor_model.plan

    def forward(
        self,
        store: KVStore,
        *,
        chunks: Sequence[Any] | None = None,
    ) -> Any:
        return self.executor_model.run(store, chunks=chunks)
