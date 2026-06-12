from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from enn_torch_dev.data import DataSchema, FieldSpec, KVBatch, KeyMapping
from enn_torch_dev.executor import GraphExecutor, KeyRef, NodeSpec
from enn_torch_dev.runtime import RuntimePhase, RuntimeStep, StepStatus


class _Double(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0


class _RaiseCudaOOM(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise torch.cuda.OutOfMemoryError("forced cuda oom")


class _RaiseRuntimeOOM(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("CUDA out of memory. forced")


class _RaiseUnknown(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("unknown failure")


def _schema() -> DataSchema:
    return DataSchema(
        schema_id="runtime.step",
        fields=(
            FieldSpec("features", torch.float32, shape=(None, 3)),
            FieldSpec("labels", torch.float32, shape=(None, 1), required=False),
        ),
        key_mapping=KeyMapping(
            inputs={"features": "x"},
            labels={"labels": "y"},
        ),
    )


def _batch(
    *,
    features: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    schema_id: str = "runtime.step",
) -> KVBatch:
    if features is None:
        features = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    payload: dict[str, torch.Tensor] = {"features": features}
    if labels is not None:
        payload["labels"] = labels
    return KVBatch(
        td=TensorDict(payload, batch_size=(int(features.shape[0]),)),
        row_ids=torch.arange(int(features.shape[0])),
        schema_id=schema_id,
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


def test_runtime_step_forward_only_success() -> None:
    batch = _batch()
    step = RuntimeStep(_graph(), schema=_schema())

    result = step.run(batch)

    assert result.status is StepStatus.SUCCESS
    assert result.phase is None
    assert result.store is not None
    assert torch.equal(result.store.get("pred"), result.store.get("x") * 2.0)
    assert torch.equal(result.row_ids, batch.row_ids)
    assert result.loss is None


def test_runtime_step_loss_backward_optimizer_success() -> None:
    model = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.1)
    before = model.weight.detach().clone()
    graph = _graph(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    labels = torch.ones(2, 1, dtype=torch.float32)
    batch = _batch(labels=labels)

    def loss_fn(store):
        return torch.nn.functional.mse_loss(store.get("pred"), store.get("y"))

    result = RuntimeStep(
        graph,
        schema=_schema(),
        loss_fn=loss_fn,
        optimizer=optimizer,
    ).run(batch)

    assert result.status is StepStatus.SUCCESS
    assert result.loss is not None
    assert torch.isfinite(result.loss.detach())
    assert not torch.equal(model.weight.detach(), before)


def test_runtime_step_preserves_row_ids_on_success() -> None:
    batch = _batch()

    result = RuntimeStep(_graph(), schema=_schema()).run(batch)

    assert torch.equal(result.row_ids, torch.tensor([0, 1]))


def test_runtime_step_missing_required_field_is_data_fault() -> None:
    td = TensorDict({"labels": torch.ones(2, 1)}, batch_size=(2,))
    batch = KVBatch(td=td, row_ids=torch.tensor([3, 4]), schema_id="runtime.step")

    result = RuntimeStep(_graph(), schema=_schema()).run(batch)

    assert result.status is StepStatus.DATA_FAULT
    assert result.phase is RuntimePhase.TO_STORE
    assert torch.equal(result.row_ids, torch.tensor([3, 4]))
    assert result.error_type == "KeyError"


def test_runtime_step_dtype_mismatch_is_data_fault() -> None:
    batch = _batch(features=torch.zeros(2, 3, dtype=torch.float64))

    result = RuntimeStep(_graph(), schema=_schema()).run(batch)

    assert result.status is StepStatus.DATA_FAULT
    assert result.phase is RuntimePhase.TO_STORE
    assert result.error_type == "TypeError"


def test_runtime_step_shape_mismatch_is_data_fault() -> None:
    batch = _batch(features=torch.zeros(2, 4, dtype=torch.float32))

    result = RuntimeStep(_graph(), schema=_schema()).run(batch)

    assert result.status is StepStatus.DATA_FAULT
    assert result.phase is RuntimePhase.TO_STORE
    assert result.error_type == "ValueError"


def test_runtime_step_nan_loss_is_nonfinite_fault() -> None:
    batch = _batch()

    def loss_fn(_store):
        return torch.tensor(float("nan"))

    result = RuntimeStep(_graph(), schema=_schema(), loss_fn=loss_fn).run(batch)

    assert result.status is StepStatus.NONFINITE_FAULT
    assert result.phase is RuntimePhase.LOSS
    assert result.loss is not None
    assert torch.isnan(result.loss)


def test_runtime_step_inf_loss_is_nonfinite_fault() -> None:
    batch = _batch()

    def loss_fn(_store):
        return torch.tensor(float("inf"))

    result = RuntimeStep(_graph(), schema=_schema(), loss_fn=loss_fn).run(batch)

    assert result.status is StepStatus.NONFINITE_FAULT
    assert result.phase is RuntimePhase.LOSS
    assert result.loss is not None
    assert torch.isinf(result.loss)


def test_runtime_step_cuda_oom_exception_is_oom_fault() -> None:
    result = RuntimeStep(
        _graph(_RaiseCudaOOM()),
        schema=_schema(),
    ).run(_batch())

    assert result.status is StepStatus.OOM_FAULT
    assert result.phase is RuntimePhase.FORWARD
    assert result.error_type == "OutOfMemoryError"


def test_runtime_step_runtime_oom_message_is_oom_fault() -> None:
    result = RuntimeStep(
        _graph(_RaiseRuntimeOOM()),
        schema=_schema(),
    ).run(_batch())

    assert result.status is StepStatus.OOM_FAULT
    assert result.phase is RuntimePhase.FORWARD
    assert result.error_type == "RuntimeError"


def test_runtime_step_unknown_exception_reraises_by_default() -> None:
    with pytest.raises(RuntimeError, match="unknown failure"):
        RuntimeStep(_graph(_RaiseUnknown()), schema=_schema()).run(_batch())


def test_runtime_step_unknown_exception_can_return_runtime_fault() -> None:
    result = RuntimeStep(
        _graph(_RaiseUnknown()),
        schema=_schema(),
        raise_unknown=False,
    ).run(_batch())

    assert result.status is StepStatus.RUNTIME_FAULT
    assert result.phase is RuntimePhase.FORWARD
    assert result.error_type == "RuntimeError"


def test_runtime_step_rejects_non_kvbatch() -> None:
    with pytest.raises(TypeError, match="KVBatch"):
        RuntimeStep(_graph(), schema=_schema()).run(object())  # type: ignore[arg-type]


def test_runtime_step_loss_fn_must_return_tensor_when_not_raising_unknown() -> None:
    def loss_fn(_store):
        return 1.0

    result = RuntimeStep(
        _graph(),
        schema=_schema(),
        loss_fn=loss_fn,
        raise_unknown=False,
    ).run(_batch())

    assert result.status is StepStatus.RUNTIME_FAULT
    assert result.phase is RuntimePhase.LOSS
    assert result.error_type == "TypeError"
