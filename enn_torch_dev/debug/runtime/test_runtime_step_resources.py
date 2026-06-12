from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn

from enn_torch_dev.data import DataSchema, FieldSpec, KVBatch, KeyMapping
from enn_torch_dev.executor import GraphExecutor, KeyRef, NodeSpec
from enn_torch_dev.runtime import (
    ResourceMonitor,
    ResourceSample,
    RuntimeStep,
    StepStatus,
)


class _Double(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0


class _Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _RaiseUnknown(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("forced runtime failure")


def _schema(label_shape: tuple[object, ...] = (None, 1)) -> DataSchema:
    return DataSchema(
        schema_id="runtime.resources",
        fields=(
            FieldSpec("features", torch.float32, shape=(None, 3)),
            FieldSpec("labels", torch.float32, shape=label_shape, required=False),
        ),
        key_mapping=KeyMapping(
            inputs={"features": "x"},
            labels={"labels": "y"},
        ),
    )


def _batch(*, labels: torch.Tensor | None = None) -> KVBatch:
    features = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    payload = {"features": features}
    if labels is not None:
        payload["labels"] = labels
    return KVBatch(
        td=TensorDict(payload, batch_size=(2,)),
        row_ids=torch.tensor([10, 11]),
        schema_id="runtime.resources",
    )


def _graph(module: nn.Module | None = None) -> GraphExecutor:
    return GraphExecutor(
        [
            (
                NodeSpec(
                    name="node",
                    input_args=[KeyRef("x")],
                    output_key="pred",
                ),
                module or _Double(),
            )
        ]
    )


def _phases(result) -> list[str]:
    return [sample.phase for sample in result.resource_samples]


def test_runtime_step_records_forward_only_resource_samples() -> None:
    result = RuntimeStep(
        _graph(),
        schema=_schema(),
        resource_monitor=ResourceMonitor(),
    ).run(_batch())

    assert result.status is StepStatus.SUCCESS
    assert _phases(result) == ["before_step", "after_to_store", "after_forward"]
    assert all(sample.timestamp_ns > 0 for sample in result.resource_samples)


def test_runtime_step_records_training_resource_samples() -> None:
    model = _Linear()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    labels = torch.ones(2, 1)

    def loss_fn(store):
        return torch.nn.functional.mse_loss(store.get("pred"), store.get("y"))

    result = RuntimeStep(
        _graph(model),
        schema=_schema(),
        loss_fn=loss_fn,
        optimizer=optimizer,
        resource_monitor=ResourceMonitor(),
    ).run(_batch(labels=labels))

    assert result.status is StepStatus.SUCCESS
    assert _phases(result) == [
        "before_step",
        "after_to_store",
        "after_zero_grad",
        "after_forward",
        "after_loss",
        "after_backward",
        "after_optimizer",
    ]


def test_runtime_step_preserves_samples_on_forward_fault() -> None:
    result = RuntimeStep(
        _graph(_RaiseUnknown()),
        schema=_schema(),
        resource_monitor=ResourceMonitor(),
        raise_unknown=False,
    ).run(_batch())

    assert result.status is StepStatus.RUNTIME_FAULT
    assert _phases(result) == ["before_step", "after_to_store"]
    assert result.error_type == "RuntimeError"


def test_runtime_step_preserves_samples_on_data_fault() -> None:
    bad = KVBatch(
        td=TensorDict({"labels": torch.ones(2, 1)}, batch_size=(2,)),
        row_ids=torch.tensor([0, 1]),
        schema_id="runtime.resources",
    )

    result = RuntimeStep(
        _graph(),
        schema=_schema(),
        resource_monitor=ResourceMonitor(),
    ).run(bad)

    assert result.status is StepStatus.DATA_FAULT
    assert _phases(result) == ["before_step"]


class _RecordingMonitor(ResourceMonitor):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def reset_peak_memory_stats(self) -> None:
        self.calls.append("reset")

    def sample(self, phase: object) -> ResourceSample:
        self.calls.append(f"sample:{phase}")
        return ResourceSample(timestamp_ns=len(self.calls), phase=str(phase))


def test_runtime_step_resets_peak_stats_before_first_resource_sample() -> None:
    monitor = _RecordingMonitor()

    result = RuntimeStep(
        _graph(),
        schema=_schema(),
        resource_monitor=monitor,
    ).run(_batch())

    assert result.status is StepStatus.SUCCESS
    assert monitor.calls[:2] == ["reset", "sample:before_step"]
