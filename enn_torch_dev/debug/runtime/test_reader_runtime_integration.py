from __future__ import annotations

import torch
from torch import nn

from enn_torch_dev.data import (
    DataSchema,
    FieldSpec,
    KeyMapping,
    StagingSpec,
    TensorDictReader,
    TensorDictStagingWriter,
)
from enn_torch_dev.executor import GraphExecutor, KeyRef, NodeSpec
from enn_torch_dev.runtime import PlainLoader, RuntimeStep, StepStatus


class _Double(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0


def _schema() -> DataSchema:
    return DataSchema(
        schema_id="runtime.integration",
        fields=(
            FieldSpec("features", torch.float32, shape=(None, 3)),
            FieldSpec("labels", torch.float32, shape=(None, 3)),
        ),
        key_mapping=KeyMapping(
            inputs={"features": "x"},
            labels={"labels": "y"},
        ),
    )


def _stage(tmp_path, *, num_rows: int = 6) -> TensorDictReader:
    features = torch.arange(num_rows * 3, dtype=torch.float32).reshape(num_rows, 3)
    labels = features * 2.0
    root = tmp_path / "stage"
    TensorDictStagingWriter(
        StagingSpec(root=root, schema=_schema()),
    ).write(
        {
            "features": features,
            "labels": labels,
            "row_id": torch.arange(100, 100 + num_rows),
        }
    )
    return TensorDictReader(root)


def _graph() -> GraphExecutor:
    return GraphExecutor(
        [
            (
                NodeSpec(
                    name="double_node",
                    input_args=[KeyRef("x")],
                    output_key="pred",
                ),
                _Double(),
            )
        ]
    )


def test_reader_loader_runtime_forward_integration(tmp_path) -> None:
    reader = _stage(tmp_path, num_rows=6)
    loader = PlainLoader(reader, batch_size=2)
    step = RuntimeStep(_graph(), schema=reader.schema)

    seen_row_ids: list[torch.Tensor] = []
    for batch in loader:
        result = step.run(batch)
        assert result.status is StepStatus.SUCCESS
        assert result.store is not None
        assert torch.equal(result.store.get("pred"), result.store.get("x") * 2.0)
        seen_row_ids.append(result.row_ids)

    assert torch.equal(torch.cat(seen_row_ids), torch.arange(100, 106))


def test_reader_loader_runtime_loss_integration(tmp_path) -> None:
    reader = _stage(tmp_path, num_rows=6)
    loader = PlainLoader(reader, batch_size=3)

    def loss_fn(store):
        return torch.nn.functional.mse_loss(store.get("pred"), store.get("y"))

    step = RuntimeStep(_graph(), schema=reader.schema, loss_fn=loss_fn)

    losses = []
    for batch in loader:
        result = step.run(batch)
        assert result.status is StepStatus.SUCCESS
        assert result.loss is not None
        losses.append(result.loss.detach())

    assert torch.equal(torch.stack(losses), torch.zeros(2))
