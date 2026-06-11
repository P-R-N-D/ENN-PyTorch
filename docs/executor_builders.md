# Executor Builder Guide

This guide is the user-facing entry point for executor builder APIs. The
lower-level design details remain in [executor_modes.md](executor_modes.md).

## Layer Summary

| Builder | Builds | Does not build |
|---|---|---|
| `GraphBuilder` | `GraphExecutor` leaf-node graphs | pipelines, plans, models |
| `BranchBuilder` | local/global/stream branch `GraphExecutor` components | pipelines, fusion modules, models |
| `ModelBuilder` | public `Model` objects for plain, tile, stream, and explicit global-local flows | global graphs, fusion modules, state routes |

The split is intentional:

```text
nn.Module + key metadata
  -> GraphBuilder
  -> GraphExecutor
  -> ModelExecutionSpec factory helpers
  -> Model
```

`BranchBuilder` only shortens repeated branch graph wiring. It still returns a
`GraphExecutor`; the caller chooses how to assemble that graph into a pipeline or
model.

## Plain Model

Use `ModelBuilder.build()` when a single graph can run directly against a
`KVStore`.

```python
from torch import nn

from enn_torch_dev.executor import KVStore, ModelBuilder


class Encoder(nn.Module):
    def forward(self, x):
        return x + 1


class Head(nn.Module):
    def forward(self, encoded):
        return encoded * 2


model = (
    ModelBuilder()
    .add(
        name="encode",
        module=Encoder(),
        input_args=["x"],
        output_key="encoded",
    )
    .add(
        name="head",
        module=Head(),
        input_args=["encoded"],
        output_key="logits",
    )
    .build()
)

store = KVStore({"x": x})
model(store)
logits = store.get("logits")
```

## Tile Model

Use `ModelBuilder.build_tile(...)` when the same local graph should run over
tiles and then reconstruct one output.

```python
model = (
    ModelBuilder()
    .add(
        name="local",
        module=local_module,
        input_args=["tile.x"],
        output_key="local.out",
    )
    .build_tile(
        tile_shape=(128, 128),
        input_key="x",
        tile_input_key="tile.x",
        output_name="local",
        output_key="tile.out",
    )
)

store = KVStore({"x": image})
tile_out = model(store)
```

The builder creates:

```text
GraphBuilder -> GraphExecutor
ModelExecutionSpec(tile=True, ...)
  -> TilePipeline
  -> Model
```

It does not create the local module or choose tensor keys automatically.

## Stream Model

Use `ModelBuilder.build_stream(...)` when a per-chunk graph should run over an
ordered chunk sequence.

```python
model = (
    ModelBuilder()
    .add(
        name="step",
        module=step_module,
        input_args=["chunk.x"],
        output_key="chunk.out",
    )
    .build_stream(
        chunk_input_key="chunk.x",
        output_name="step",
        outputs_key="stream.outputs",
    )
)

store = KVStore()
outputs = model(store, chunks=chunks)
```

Chunks remain explicit. The builder does not split inputs into chunks and does
not infer `StateRoute` values. Pass `state_routes=...` to `build_stream(...)`
when state carry is required.

## Explicit Global-Local Model

Use `BranchBuilder` when local/global branch key conventions should be explicit
but repeated branch graph wiring should be short.

```python
from enn_torch_dev.executor import (
    BranchBuilder,
    Model,
    ModelBuilder,
    ModelExecutionSpec,
)
from enn_torch_dev.nn import LocalGlobalFusion


local_graph = (
    BranchBuilder.local(input_key="tile.x")
    .add(
        name="local",
        module=local_module,
        output_key="local.out",
    )
    .build()
)

global_graph = (
    BranchBuilder.global_(input_key="x")
    .add(
        name="global",
        module=global_module,
        output_key="global.out",
    )
    .build()
)

spec = ModelExecutionSpec(
    context="global_local",
    tile=True,
    tile_shape=(128, 128),
)

tile_pipeline = spec.create_tile_pipeline(
    local_graph,
    input_key="x",
    tile_input_key="tile.x",
    output_name="local",
)

global_local_pipeline = spec.create_global_local_pipeline(
    global_graph=global_graph,
    tile_pipeline=tile_pipeline,
    fusion=LocalGlobalFusion(),
    global_output_name="global",
    fused_output_key="fused.out",
)

model = Model.from_components(
    spec,
    global_local_pipeline=global_local_pipeline,
)
```

If the local branch should be built directly from `ModelBuilder`, keep the
global graph and fusion module caller-provided:

```python
model = (
    ModelBuilder()
    .add(
        name="local",
        module=local_module,
        input_args=["tile.x"],
        output_key="local.out",
    )
    .build_global_local(
        global_graph=global_graph,
        fusion=fusion,
        tile_shape=(128, 128),
        input_key="x",
        tile_input_key="tile.x",
        local_output_name="local",
        global_output_name="global",
        fused_output_key="fused.out",
    )
)
```

This path keeps ownership clear:

```text
BranchBuilder.local(...)  -> local GraphExecutor
BranchBuilder.global_(...) -> global GraphExecutor
caller-provided fusion
ModelExecutionSpec factory helpers -> Model
```

## Boundary Rules

Builders should keep validation delegated to the lower layer they call:

```text
GraphBuilder
  delegates graph validation to GraphExecutor

ModelBuilder
  delegates execution-mode validation to ModelExecutionSpec and ExecutorPlan

BranchBuilder
  delegates node construction and validation to GraphBuilder
```

Do not add these responsibilities to the builder layer:

```text
automatic graph architecture generation
automatic fusion module creation
automatic stream chunking
StateRoute inference
training loop, optimizer, or scheduler ownership
```
