from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace

import torch
from tensordict import TensorDictBase

from enn_torch_dev.data import (
    BatchCost,
    KVBatch,
    SpdlTensorAdapter,
    TensorDictReader,
)
from enn_torch_dev.data.spdl_adapter import TensorBatchSource

from .cost import DataCost, DataCostProbe


class PlainLoader:
    """Thin sequential loader over TensorDictReader.

    This loader intentionally does not implement workers, prefetch, shuffling,
    pinned memory, device transfer, or SPDL integration. It is the minimal
    runtime boundary used to feed KVBatch objects into RuntimeStep.
    """

    def __init__(
        self,
        reader: TensorDictReader,
        *,
        batch_size: int,
        drop_last: bool = False,
        shard_id: int | None = None,
    ) -> None:
        if not isinstance(reader, TensorDictReader):
            raise TypeError("PlainLoader.reader must be a TensorDictReader.")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("PlainLoader.batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("PlainLoader.batch_size must be positive.")
        if not isinstance(drop_last, bool):
            raise TypeError("PlainLoader.drop_last must be a bool.")
        if shard_id is not None and (
            not isinstance(shard_id, int) or isinstance(shard_id, bool)
        ):
            raise TypeError("PlainLoader.shard_id must be an integer or None.")

        self.reader = reader
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shard_id = shard_id

    def __iter__(self) -> Iterator[KVBatch]:
        start = 0
        while start < self.reader.num_rows:
            end = min(start + self.batch_size, self.reader.num_rows)
            if self.drop_last and end - start < self.batch_size:
                break
            yield self.reader.get_kvbatch(slice(start, end), shard_id=self.shard_id)
            start = end


class SPDLLoader:
    """Thin sequential loader over tensorized SPDL-style batch iterables.

    SPDLLoader intentionally does not import SPDL, construct pipelines, spawn
    workers, prefetch, pin memory, transfer tensors to devices, or choose batch
    sizes. It only adapts already tensorized SPDL outputs into KVBatch objects.
    """

    def __init__(
        self,
        source: Iterable[TensorBatchSource],
        adapter: SpdlTensorAdapter,
        *,
        shard_id: int | None = None,
        cost_probe: DataCostProbe | None = None,
    ) -> None:
        if isinstance(source, (Mapping, TensorDictBase, str, bytes, bytearray)):
            raise TypeError(
                "SPDLLoader.source must be an iterable of tensor batches, "
                "not a single tensor batch."
            )
        if not isinstance(source, Iterable):
            raise TypeError("SPDLLoader.source must be an iterable of tensor batches.")
        if not isinstance(adapter, SpdlTensorAdapter):
            raise TypeError("SPDLLoader.adapter must be a SpdlTensorAdapter.")
        if shard_id is not None and (
            not isinstance(shard_id, int) or isinstance(shard_id, bool)
        ):
            raise TypeError("SPDLLoader.shard_id must be an integer or None.")
        if cost_probe is not None and not isinstance(cost_probe, DataCostProbe):
            raise TypeError("SPDLLoader.cost_probe must be a DataCostProbe or None.")

        self.source = source
        self.adapter = adapter
        self.shard_id = shard_id
        self.cost_probe = cost_probe

    def __iter__(self) -> Iterator[KVBatch]:
        for batch in self.source:
            kvbatch = self.adapter.to_kvbatch(batch, shard_id=self.shard_id)
            yield self._attach_cost_hint(kvbatch)

    def _attach_cost_hint(self, batch: KVBatch) -> KVBatch:
        if self.cost_probe is None:
            return batch
        return replace(batch, cost_hint=self._estimate_batch_cost(batch))

    def _estimate_batch_cost(self, batch: KVBatch) -> BatchCost:
        assert self.cost_probe is not None
        data_cost = self.cost_probe.estimate_kvbatch(batch)
        return _batch_cost_from_data_cost(data_cost, batch)


def _tensor_payload_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _identity_host_bytes(batch: KVBatch) -> int:
    total = _tensor_payload_bytes(batch.row_ids)
    if batch.source_ids is not None:
        total += _tensor_payload_bytes(batch.source_ids)
    if batch.sample_ids is not None:
        total += _tensor_payload_bytes(batch.sample_ids)
    return total


def _batch_cost_from_data_cost(data_cost: DataCost, batch: KVBatch) -> BatchCost:
    host_bytes = sum(
        nbytes for device, nbytes in data_cost.bytes_by_device.items() if device == "cpu"
    )
    device_bytes = sum(
        nbytes for device, nbytes in data_cost.bytes_by_device.items() if device != "cpu"
    )
    return BatchCost(
        host_bytes=host_bytes + _identity_host_bytes(batch),
        device_bytes=device_bytes,
        num_items=data_cost.batch_size,
    )
