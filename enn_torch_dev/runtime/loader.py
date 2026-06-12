from __future__ import annotations

from collections.abc import Iterator

from enn_torch_dev.data import KVBatch, TensorDictReader


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
