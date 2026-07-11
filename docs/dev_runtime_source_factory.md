# Runtime Pass Source Factory

This document describes the fifteenth data/runtime rewrite slice for
`enn_torch_dev`.

## Goal

`ConservativeRuntimeSession.run_passes(...)` accepts an outer iterable that
already contains finite pass sources. This slice adds a small factory boundary
for callers that need a fresh one-shot source or loader for every pass:

```text
pass_index
  -> RuntimePassSourceFactory.create_pass_source(pass_index)
  -> fresh finite Iterable[KVBatch]
  -> ConservativeRuntimeSession.run_factory(...)
  -> RuntimeSessionRecord
```

The factory owns source construction only. It does not execute a pass, retry
source construction, cache sources, persist epoch state, or decide runtime
budgets.

## Public Object

`enn_torch_dev.runtime` exports the runtime-checkable protocol:

- `RuntimePassSourceFactory`

`ConservativeRuntimeSession` adds:

- `run_factory(source_factory)`

The stable `enn_torch` namespace does not expose this development API.

## Protocol Contract

A factory provides:

```python
class RuntimePassSourceFactory(Protocol):
    def create_pass_source(
        self,
        pass_index: int,
    ) -> Iterable[KVBatch]:
        ...
```

`pass_index` starts at zero for each `run_factory(...)` invocation. The session
calls the factory lazily: creating the session iterator does not call the
factory, and each `next()` call performs at most one factory call and one finite
pass.

The session stops after `max_passes` and does not call the factory for an extra
pass. The factory must return a finite iterable of `KVBatch`; returning a single
`KVBatch` or a non-iterable raises `TypeError` before history is updated.

An empty iterable is a valid finite pass. It produces an empty
`RuntimePassResult`, a pass summary, and one history record.

## Error and Retention Semantics

Factory exceptions propagate without suppression. A pass whose source was not
created is not added to history, while previously completed history remains
unchanged.

The session does not cache created sources. After a yielded record is resumed,
the previous source is released before the next factory call. Longer-lived
retention remains limited to lightweight summaries in the bounded
`RuntimePassHistory`.

## SPDLLoader Example

A factory may build a new one-shot `SPDLLoader` for each pass without making the
loader replay its consumed source:

```python
from enn_torch_dev.runtime import RuntimePassSourceFactory, SPDLLoader


class SpdlPassFactory:
    def create_pass_source(self, pass_index: int):
        tensor_batches = build_tensor_batches_for_pass(pass_index)
        return SPDLLoader(
            tensor_batches,
            adapter,
            shard_id=pass_index,
        )


for record in session.run_factory(SpdlPassFactory()):
    inspect(record.pass_summary)
```

The caller-defined `build_tensor_batches_for_pass(...)` remains responsible for
constructing a fresh finite tensor source. This API does not construct or tune an
SPDL pipeline.

## Out of Scope

- Source caching or replay of consumed iterators.
- Automatic retry after factory exceptions.
- Persistent epoch/pass state.
- Checkpoint/resume.
- Prefetch, workers, queue depth, pinned memory, or device transfer.
- SPDL pipeline construction.
- Distributed samplers or coordination.
- Persistent logs or exports.

## Test Commands

```bash
python -m pytest enn_torch_dev/debug/runtime/test_runtime_source_factory.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_session.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```
