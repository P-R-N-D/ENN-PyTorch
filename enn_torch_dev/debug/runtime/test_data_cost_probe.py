from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from enn_torch_dev.data import KVBatch
from enn_torch_dev.runtime import DataCostProbe


def _td(num_rows: int = 2) -> TensorDict:
    return TensorDict(
        {
            "features": torch.arange(num_rows * 3, dtype=torch.float32).reshape(num_rows, 3),
            "labels": torch.arange(num_rows, dtype=torch.float32).reshape(num_rows, 1),
        },
        batch_size=(num_rows,),
    )


def _kvbatch(num_rows: int = 2) -> KVBatch:
    return KVBatch(
        td=_td(num_rows),
        row_ids=torch.arange(num_rows),
        schema_id="cost.data",
    )


def test_data_cost_probe_estimates_kvbatch_tensor_bytes() -> None:
    cost = DataCostProbe().estimate_kvbatch(_kvbatch(num_rows=2))

    assert cost.batch_size == 2
    assert cost.tensor_count == 2
    assert cost.total_tensor_bytes == (6 * 4) + (2 * 4)
    assert cost.bytes_per_row == 16.0
    assert {tensor.key for tensor in cost.tensors} == {"features", "labels"}
    assert cost.bytes_by_dtype == {"float32": 32}
    assert cost.bytes_by_device == {"cpu": 32}


def test_data_cost_probe_estimates_tensordict() -> None:
    cost = DataCostProbe().estimate_tensordict(_td(num_rows=3))

    assert cost.batch_size == 3
    assert cost.tensor_count == 2
    assert cost.total_tensor_bytes == (9 * 4) + (3 * 4)
    assert cost.bytes_per_row == 16.0


def test_data_cost_probe_groups_by_dtype_and_device() -> None:
    td = TensorDict(
        {
            "x": torch.ones(2, 3, dtype=torch.float32),
            "mask": torch.ones(2, dtype=torch.bool),
        },
        batch_size=(2,),
    )

    cost = DataCostProbe().estimate_tensordict(td)

    assert cost.bytes_by_dtype == {"bool": 2, "float32": 24}
    assert cost.bytes_by_device == {"cpu": 26}


def test_data_cost_probe_handles_nested_tensordict() -> None:
    td = TensorDict(
        {
            "x": torch.ones(2, 3),
            "nested": TensorDict(
                {
                    "mask": torch.ones(2, dtype=torch.bool),
                },
                batch_size=(2,),
            ),
        },
        batch_size=(2,),
    )

    cost = DataCostProbe().estimate_tensordict(td)
    assert {tensor.key for tensor in cost.tensors} == {"x", "nested.mask"}
    assert cost.total_tensor_bytes == 24 + 2


def test_data_cost_probe_deduplicates_aliased_storage() -> None:
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    td = TensorDict(
        {
            "left": base,
            "right": base.view(2, 3),
        },
        batch_size=(2,),
    )

    cost = DataCostProbe().estimate_tensordict(td)
    assert cost.tensor_count == 1
    assert cost.total_tensor_bytes == 6 * 4
    assert cost.bytes_by_dtype == {"float32": 6 * 4}


def test_data_cost_probe_counts_full_storage_bytes_for_split_aliases() -> None:
    base = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    cost = DataCostProbe().estimate_mapping(
        {
            "head": base[:1],
            "tail": base[1:],
        },
        batch_size=4,
    )

    assert cost.tensor_count == 1
    assert cost.total_tensor_bytes == base.untyped_storage().nbytes()
    assert cost.bytes_by_dtype == {"float32": base.untyped_storage().nbytes()}


def test_data_cost_probe_handles_zero_batch_size() -> None:
    td = TensorDict(
        {
            "x": torch.empty(0, 3, dtype=torch.float32),
        },
        batch_size=(0,),
    )

    cost = DataCostProbe().estimate_tensordict(td)
    assert cost.batch_size == 0
    assert cost.total_tensor_bytes == 0
    assert cost.bytes_per_row is None


def test_data_cost_probe_mapping_skips_non_tensor_values() -> None:
    cost = DataCostProbe().estimate_mapping(
        {
            "x": torch.ones(2, 3),
            "tag": "not a tensor",
            "metadata": {"source": "synthetic"},
        },
        batch_size=2,
    )

    assert cost.batch_size == 2
    assert cost.tensor_count == 1
    assert cost.total_tensor_bytes == 2 * 3 * 4


def test_data_cost_probe_rejects_invalid_inputs() -> None:
    probe = DataCostProbe()
    with pytest.raises(TypeError, match="KVBatch"):
        probe.estimate_kvbatch(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="TensorDictBase"):
        probe.estimate_tensordict(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Mapping"):
        probe.estimate_mapping(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        probe.estimate_mapping({}, batch_size=-1)
