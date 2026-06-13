from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from enn_torch_dev.data import (
    BatchCost,
    DataSchema,
    FieldSpec,
    KVBatch,
    KeyMapping,
    SpdlTensorAdapter,
)
from enn_torch_dev.runtime import (
    BatchBudget,
    BatchBudgetExceeded,
    BudgetedBatcher,
    DataCostProbe,
    SPDLLoader,
)


def _batch(num_rows: int = 4, *, cost_hint: BatchCost | None = None) -> KVBatch:
    td = TensorDict(
        {
            "features": torch.arange(num_rows * 3, dtype=torch.float32).reshape(
                num_rows,
                3,
            ),
            "labels": torch.arange(num_rows, dtype=torch.float32).reshape(num_rows, 1),
        },
        batch_size=(num_rows,),
    )
    return KVBatch(
        td=td,
        row_ids=torch.arange(100, 100 + num_rows),
        source_ids=torch.arange(200, 200 + num_rows),
        sample_ids=torch.arange(300, 300 + num_rows),
        schema_id="runtime.budgeted",
        shard_id=7,
        cost_hint=cost_hint,
    )


def _spdl_schema() -> DataSchema:
    return DataSchema(
        schema_id="runtime.budgeted.spdl",
        fields=(
            FieldSpec("features", torch.float32, shape=(None, 3)),
            FieldSpec("labels", torch.float32, shape=(None, 1), required=False),
        ),
        key_mapping=KeyMapping(inputs={"features": "x"}, labels={"labels": "y"}),
    )


def _spdl_payload(num_rows: int) -> dict[str, torch.Tensor]:
    return {
        "features": torch.arange(num_rows * 3, dtype=torch.float32).reshape(
            num_rows,
            3,
        ),
        "labels": torch.arange(num_rows, dtype=torch.float32).reshape(num_rows, 1),
        "row_id": torch.arange(100, 100 + num_rows),
    }


def test_budgeted_batcher_passes_batches_within_budget() -> None:
    batch = _batch(2)

    batches = list(BudgetedBatcher([batch], BatchBudget(max_items=2)))

    assert len(batches) == 1
    assert batches[0].batch_size == 2
    assert torch.equal(batches[0].row_ids, batch.row_ids)


def test_budgeted_batcher_splits_by_max_items() -> None:
    batch = _batch(5)

    batches = list(BudgetedBatcher([batch], BatchBudget(max_items=2)))

    assert [subbatch.batch_size for subbatch in batches] == [2, 2, 1]
    assert torch.equal(
        torch.cat([subbatch.row_ids for subbatch in batches]),
        batch.row_ids,
    )


def test_budgeted_batcher_preserves_identity_and_metadata_after_split() -> None:
    batch = _batch(5)

    batches = list(BudgetedBatcher([batch], BatchBudget(max_items=3)))

    assert [subbatch.schema_id for subbatch in batches] == [
        "runtime.budgeted",
        "runtime.budgeted",
    ]
    assert [subbatch.shard_id for subbatch in batches] == [7, 7]
    assert torch.equal(
        torch.cat(
            [subbatch.source_ids for subbatch in batches if subbatch.source_ids is not None]
        ),
        batch.source_ids,
    )
    assert torch.equal(
        torch.cat(
            [subbatch.sample_ids for subbatch in batches if subbatch.sample_ids is not None]
        ),
        batch.sample_ids,
    )


def test_budgeted_batcher_adds_cost_hint_with_probe() -> None:
    batch = _batch(2)

    batches = list(
        BudgetedBatcher(
            [batch],
            BatchBudget(max_host_bytes=80),
            cost_probe=DataCostProbe(),
        )
    )

    assert batches[0].cost_hint is not None
    assert batches[0].cost_hint.host_bytes == 80
    assert batches[0].cost_hint.device_bytes == 0
    assert batches[0].cost_hint.num_items == 2


def test_budgeted_batcher_materializes_byte_budget_slices_for_probe() -> None:
    batch = _batch(5)

    batches = list(
        BudgetedBatcher(
            [batch],
            BatchBudget(max_host_bytes=40),
            cost_probe=DataCostProbe(),
        )
    )

    assert [subbatch.batch_size for subbatch in batches] == [1, 1, 1, 1, 1]
    assert [subbatch.cost_hint.host_bytes for subbatch in batches] == [
        40,
        40,
        40,
        40,
        40,
    ]
    assert torch.equal(
        torch.cat([subbatch.row_ids for subbatch in batches]),
        batch.row_ids,
    )


def test_budgeted_batcher_scales_cost_hint_for_byte_budget_without_probe() -> None:
    cost_hint = BatchCost(
        host_bytes=101,
        device_bytes=0,
        num_items=5,
        num_tokens=11,
        num_tiles=6,
    )
    batch = _batch(5, cost_hint=cost_hint)

    batches = list(BudgetedBatcher([batch], BatchBudget(max_host_bytes=21)))

    assert [subbatch.batch_size for subbatch in batches] == [1, 1, 1, 1, 1]
    assert [subbatch.cost_hint.host_bytes for subbatch in batches] == [
        21,
        21,
        21,
        21,
        21,
    ]
    assert [subbatch.cost_hint.num_items for subbatch in batches] == [1, 1, 1, 1, 1]
    assert [subbatch.cost_hint.num_tokens for subbatch in batches] == [3, 3, 3, 3, 3]
    assert [subbatch.cost_hint.num_tiles for subbatch in batches] == [2, 2, 2, 2, 2]


def test_budgeted_batcher_uses_probe_when_cost_hint_lacks_required_byte_field() -> None:
    batch = _batch(5, cost_hint=BatchCost(num_items=5))

    batches = list(
        BudgetedBatcher(
            [batch],
            BatchBudget(max_host_bytes=40),
            cost_probe=DataCostProbe(),
        )
    )

    assert [subbatch.batch_size for subbatch in batches] == [1, 1, 1, 1, 1]
    assert [subbatch.cost_hint.host_bytes for subbatch in batches] == [
        40,
        40,
        40,
        40,
        40,
    ]


def test_budgeted_batcher_probe_cost_includes_identity_tensors() -> None:
    batch = _batch(2)

    batches = list(
        BudgetedBatcher(
            [batch],
            BatchBudget(max_host_bytes=80),
            cost_probe=DataCostProbe(),
        )
    )

    assert batches[0].cost_hint.host_bytes == 80


def test_budgeted_batcher_identity_tensors_can_exceed_byte_budget() -> None:
    batch = _batch(1)

    with pytest.raises(BatchBudgetExceeded, match="host_bytes"):
        list(
            BudgetedBatcher(
                [batch],
                BatchBudget(max_host_bytes=39),
                cost_probe=DataCostProbe(),
            )
        )


def test_budgeted_batcher_respects_spdl_loader_identity_inclusive_cost_hint() -> None:
    loader = SPDLLoader(
        [_spdl_payload(2)],
        SpdlTensorAdapter(_spdl_schema()),
        cost_probe=DataCostProbe(),
    )

    batches = list(
        BudgetedBatcher(
            loader,
            BatchBudget(max_host_bytes=24),
            cost_probe=DataCostProbe(),
        )
    )

    assert [batch.batch_size for batch in batches] == [1, 1]
    assert [batch.cost_hint.host_bytes for batch in batches] == [24, 24]


def test_budgeted_batcher_rejects_spdl_loader_hint_when_identity_bytes_exceed_budget() -> None:
    loader = SPDLLoader(
        [_spdl_payload(1)],
        SpdlTensorAdapter(_spdl_schema()),
        cost_probe=DataCostProbe(),
    )

    with pytest.raises(BatchBudgetExceeded, match="host_bytes"):
        list(
            BudgetedBatcher(
                loader,
                BatchBudget(max_host_bytes=23),
                cost_probe=DataCostProbe(),
            )
        )


def test_budgeted_batcher_prefers_existing_cost_hint() -> None:
    cost_hint = BatchCost(host_bytes=1, device_bytes=0, num_items=4)
    batch = _batch(4, cost_hint=cost_hint)

    batches = list(BudgetedBatcher([batch], BatchBudget(max_host_bytes=1)))

    assert batches[0].cost_hint == cost_hint
    assert batches[0].batch_size == 4


def test_budgeted_batcher_rejects_byte_budget_without_byte_cost() -> None:
    batch = _batch(2)

    with pytest.raises(ValueError, match="host_bytes"):
        list(BudgetedBatcher([batch], BatchBudget(max_host_bytes=64)))


def test_budgeted_batcher_raises_when_splitting_is_disabled() -> None:
    batch = _batch(3)

    with pytest.raises(BatchBudgetExceeded, match="num_items"):
        list(
            BudgetedBatcher(
                [batch],
                BatchBudget(max_items=2),
                split_oversized=False,
            )
        )


def test_budgeted_batcher_raises_when_min_items_still_exceeds_budget() -> None:
    batch = _batch(1)

    with pytest.raises(BatchBudgetExceeded, match="num_items"):
        list(BudgetedBatcher([batch], BatchBudget(max_items=0), min_items=1))


def test_budgeted_batcher_accepts_empty_source() -> None:
    assert list(BudgetedBatcher([], BatchBudget(max_items=1))) == []


def test_batch_budget_requires_at_least_one_limit() -> None:
    with pytest.raises(ValueError, match="at least one"):
        BatchBudget()


def test_batch_budget_rejects_invalid_limits() -> None:
    with pytest.raises(TypeError, match="max_items"):
        BatchBudget(max_items=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_host_bytes"):
        BatchBudget(max_host_bytes=-1)


def test_budgeted_batcher_rejects_invalid_arguments() -> None:
    with pytest.raises(TypeError, match="source"):
        BudgetedBatcher(_batch(1), BatchBudget(max_items=1))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="budget"):
        BudgetedBatcher([], object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="cost_probe"):
        BudgetedBatcher([], BatchBudget(max_items=1), cost_probe=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_items"):
        BudgetedBatcher([], BatchBudget(max_items=1), min_items=0)


def test_budgeted_batcher_rejects_non_kvbatch_items() -> None:
    with pytest.raises(TypeError, match="KVBatch"):
        list(BudgetedBatcher([object()], BatchBudget(max_items=1)))  # type: ignore[list-item]
