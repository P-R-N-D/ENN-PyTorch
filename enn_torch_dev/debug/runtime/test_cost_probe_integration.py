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
from enn_torch_dev.runtime import (
    DataCostProbe,
    ModelCostProbe,
    PlainLoader,
    ResourceMonitor,
    RuntimeStep,
    StepStatus,
)


class _Double(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0


def _schema() -> DataSchema:
    return DataSchema(
        schema_id="cost.integration",
        fields=(
            FieldSpec("features", torch.float32, shape=(None, 3)),
            FieldSpec("labels", torch.float32, shape=(None, 3)),
        ),
        key_mapping=KeyMapping(
            inputs={"features": "x"},
            labels={"labels": "y"},
        ),
    )


def _reader(tmp_path) -> TensorDictReader:
    features = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    labels = features * 2.0
    root = tmp_path / "stage"
    TensorDictStagingWriter(
        StagingSpec(root=root, schema=_schema()),
    ).write(
        {
            "features": features,
            "labels": labels,
            "row_id": torch.arange(100, 106),
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


def test_cost_probes_integrate_with_reader_loader_and_runtime_step(tmp_path) -> None:
    reader = _reader(tmp_path)
    batch = next(iter(PlainLoader(reader, batch_size=2)))

    data_cost = DataCostProbe().estimate_kvbatch(batch)

    assert data_cost.batch_size == 2
    assert data_cost.tensor_count == 2
    assert data_cost.total_tensor_bytes == (2 * 3 * 4) + (2 * 3 * 4)
    assert data_cost.bytes_per_row == 24.0

    step_result = RuntimeStep(
        _graph(),
        schema=reader.schema,
        resource_monitor=ResourceMonitor(),
    ).run(batch)

    model_cost = ModelCostProbe().estimate_step(step_result)

    assert step_result.status is StepStatus.SUCCESS
    assert model_cost.status is StepStatus.SUCCESS
    assert model_cost.batch_size == batch.batch_size
    assert model_cost.row_count == int(batch.row_ids.numel())
    assert [delta.start_phase for delta in model_cost.phase_deltas] == [
        "before_step",
        "after_to_store",
    ]
    assert [delta.end_phase for delta in model_cost.phase_deltas] == [
        "after_to_store",
        "after_forward",
    ]
