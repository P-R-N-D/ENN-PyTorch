# Executor Modes: Tile and Stream

This document fixes the terminology for the executor-side processing modes.

The short version:

```text
Tile   = split one complete input by position, process pieces, reconstruct.
Stream = process ordered chunks one by one, carry state between chunks.
```

## Core execution unit

Both modes use `GraphExecutor` as the execution engine.

```text
GraphExecutor
  runs NodeExecutor / SubgraphExecutor in dependency order
```

`GraphExecutor` does not know whether a value came from a tile, a stream chunk, or a normal input. Mode-specific pipelines prepare the `KVStore`, run the graph, and collect outputs.

## Tile mode

Tile mode is for a complete input that already exists.

```text
full input
  -> TilePolicy
  -> TileExecutor
  -> TileReconstructor
  -> reconstructed output
```

Use tile mode when the important question is:

```text
Where did this piece come from, and where should the result be placed back?
```

Typical examples:

```text
large image
large feature map
spatial tensor
volume / grid-like tensor
```

Tile mode is position-based. A tile has metadata such as its start/end position and the full output shape. Tile execution may be implemented as a loop, but sequential execution is not the core concept. The core concept is positional split and reconstruction.

Current executor pieces:

```text
TilePolicy
  Tensor -> tiles + TileMeta

TileExecutor
  run a graph for each already-split tile

TileReconstructor
  tile outputs + TileMeta -> reconstructed tensor

TilePipeline
  TilePolicy + TileExecutor + TileReconstructor
```

Tile mode should not carry hidden state from one tile to the next unless a caller explicitly models that in the graph. Tiles are normally independent pieces of one complete input.

## Stream mode

Stream mode is for ordered chunks.

```text
chunk_0 -> graph -> output_0 + state_1
chunk_1 -> graph -> output_1 + state_2
chunk_2 -> graph -> output_2 + state_3
```

Use stream mode when the important question is:

```text
What state from the previous chunk should be passed into the next chunk?
```

Typical examples:

```text
audio chunks
video frame chunks
sensor readings
token chunks
time-ordered feature chunks
```

Stream mode is order-based. It does not reconstruct a spatial tensor. It returns an output sequence and optionally carries state through `StateRoute`.

Current executor pieces:

```text
StateRoute
  describes state_input_key and state_output_key

StateRoute.carry(...)
  state_output_key -> state_input_key

StateRoute.reset(...)
  clears state_input_key before a new stream

StreamPipeline
  run ordered chunks through GraphExecutor
  isolate per-chunk graph outputs
  carry explicit StateRoute state between chunks
```

`StreamPipeline` does not split inputs into chunks. The caller provides the chunk sequence.

## Tile vs Stream

| Question | Tile | Stream |
| --- | --- | --- |
| Is the full input already available? | Yes | Not necessarily |
| What defines each chunk? | Position / region | Order / time |
| What is preserved between chunks? | Position metadata | State |
| What happens after processing chunks? | Reconstruct into full output | Return output sequence |
| Are chunks normally independent? | Yes | No, state may link them |
| Main pipeline | `TilePipeline` | `StreamPipeline` |

## ExecutorModeSpec

`ExecutorModeSpec` is a small declarative flag object for higher-level wrappers.
It does not build graphs, create pipelines, split tensors, or run streams. It
only records which executor modes a caller intends to compose.

```python
from enn_torch_dev.executor import ExecutorModeSpec

plain = ExecutorModeSpec()
tile = ExecutorModeSpec(tile=True)
stream = ExecutorModeSpec(stream=True)
global_local = ExecutorModeSpec(tile=True, global_local=True)
tile_stream = ExecutorModeSpec(tile=True, stream=True)
tile_stream_global_local = ExecutorModeSpec(
    tile=True,
    stream=True,
    global_local=True,
)
```

The mode flags follow these rules:

```text
all flags False
  -> plain GraphExecutor mode

tile=True
  -> TilePipeline may be used

stream=True
  -> StreamPipeline may be used

global_local=True
  -> requires tile=True
  -> GlobalLocalPipeline may be used

tile=True and stream=True
  -> allowed
  -> stream controls order/time
  -> tile controls spatial/local processing inside each chunk

tile=True, stream=True, global_local=True
  -> allowed
  -> stream controls order/time
  -> global/local tiled fusion can be applied inside each chunk
```

`ExecutorModeSpec` is intentionally not a model wrapper. Automatic construction
of graphs and pipelines belongs to a later layer.

## ExecutorPlan

`ExecutorPlan` is the next thin layer after `ExecutorModeSpec`.

```text
ExecutorModeSpec
  declares which modes are intended

ExecutorPlan
  validates that the required executor components were supplied
```

`ExecutorModeSpec` only records the requested execution modes. `ExecutorPlan`
checks that those declared modes match the executor objects supplied by the
caller. `ExecutorPlan` still does not run the model, build graphs, create
pipelines, or own training/inference policy.

```python
from enn_torch_dev.executor import ExecutorModeSpec, ExecutorPlan

plain_plan = ExecutorPlan(
    mode=ExecutorModeSpec(),
    graph=graph,
)

tile_plan = ExecutorPlan(
    mode=ExecutorModeSpec(tile=True),
    tile_pipeline=tile_pipeline,
)

stream_plan = ExecutorPlan(
    mode=ExecutorModeSpec(stream=True),
    stream_pipeline=stream_pipeline,
)

global_local_plan = ExecutorPlan(
    mode=ExecutorModeSpec(tile=True, global_local=True),
    global_local_pipeline=global_local_pipeline,
)

stream_tile_plan = ExecutorPlan(
    mode=ExecutorModeSpec(tile=True, stream=True),
    stream_pipeline=stream_pipeline,
    tile_pipeline=tile_pipeline,
)

stream_global_local_plan = ExecutorPlan(
    mode=ExecutorModeSpec(tile=True, stream=True, global_local=True),
    stream_pipeline=stream_pipeline,
    global_local_pipeline=global_local_pipeline,
)
```

The component rules are:

```text
plain mode
  -> requires graph

tile=True
  -> requires tile_pipeline

stream=True
  -> requires stream_pipeline

global_local=True
  -> requires global_local_pipeline
  -> does not accept a separate tile_pipeline
  -> uses the TilePipeline embedded in GlobalLocalPipeline

stream=True and tile=True
  -> requires stream_pipeline and tile_pipeline
  -> stream is the outer order/time layer
  -> tile is the inner spatial/local layer

stream=True and global_local=True
  -> requires stream_pipeline and global_local_pipeline
  -> stream is the outer order/time layer
  -> global/local tiled fusion is the inner per-chunk layer
```

When stream is combined with any spatial/tiled mode, stream is always the outer
layer. The stream layer controls chunk order and carried state; tile or
global/local execution is applied inside each stream chunk.

This keeps the planned order explicit before a future `Model(...)` wrapper
decides how to call or compose those components.

## Higher-level Model API naming

Public `Model(...)` parameters should not expose `tile`, `stream`, and
`global_local` as three unrelated peer booleans. They describe different layers:

```text
context
  model structure

tile
  local branch partitioning

stateful
  state carry across calls or chunks
```

Prefer a higher-level API shape like this:

```python
Model(
    context="local",      # "local" | "global_local"
    tile=False,
    stateful=False,
    tile_shape=None,
    tile_stride=None,
    tile_dims=None,
)
```

The public meanings are:

```text
context="local"
  local-only model structure

context="global_local"
  global branch + local branch + fusion

tile=True
  apply position-based tiling to the local branch

stateful=True
  carry state across calls/chunks
  implemented internally with StreamPipeline / StateRoute
```

Do not interpret `tile=True, stateful=True` as "process tiles statefully in
order." It means:

```text
stateful execution is the outer call/chunk behavior
tile processing happens inside each stateful chunk/call
```

The intended mapping to `ExecutorModeSpec` is:

```text
Model(context="local", tile=False, stateful=False)
  -> ExecutorModeSpec()

Model(context="local", tile=True, stateful=False)
  -> ExecutorModeSpec(tile=True)

Model(context="global_local", tile=True, stateful=False)
  -> ExecutorModeSpec(tile=True, global_local=True)

Model(context="local", tile=False, stateful=True)
  -> ExecutorModeSpec(stream=True)

Model(context="local", tile=True, stateful=True)
  -> ExecutorModeSpec(tile=True, stream=True)

Model(context="global_local", tile=True, stateful=True)
  -> ExecutorModeSpec(tile=True, stream=True, global_local=True)
```

For the current executor implementation, `context="global_local"` requires
`tile=True` because `GlobalLocalPipeline` uses `TilePipeline` as its local
branch. A future full-resolution local branch could relax that requirement.

Tile sizing remains explicit:

```text
tile_shape
  size of each tile

tile_stride
  distance between neighboring tile starts

tile_dims
  tensor dimensions that are tiled
```

## ModelExecutionSpec factory helpers

`ModelExecutionSpec` also provides small factory helpers that bridge public
model-side configuration to executor-side objects. These helpers do not create
model branches automatically. They either create specs/policies from validated
public parameters or wrap caller-provided executor components.

Available helpers:

```text
create_tile_policy()
  tile_shape / tile_stride / tile_dims
  -> TilePolicy

create_tile_pipeline_spec(...)
  input/output key schema
  -> TilePipelineSpec

create_tile_pipeline(graph, ...)
  caller-provided GraphExecutor
  + create_tile_policy()
  + create_tile_pipeline_spec(...)
  -> TilePipeline

create_global_local_pipeline_spec(...)
  global output key schema
  -> GlobalLocalPipelineSpec

create_global_local_pipeline(...)
  caller-provided global_graph
  + caller-provided tile_pipeline
  + caller-provided fusion
  + create_global_local_pipeline_spec(...)
  -> GlobalLocalPipeline

create_stream_pipeline_spec(...)
  chunk input/output key schema
  + state policy flags
  -> StreamPipelineSpec

create_stream_pipeline(graph, ...)
  caller-provided GraphExecutor
  + create_stream_pipeline_spec(...)
  + caller-provided state_routes
  -> StreamPipeline

create_plan(...)
  caller-provided graph/pipelines
  -> ExecutorPlan
```

The tiled local branch can be built from a caller-provided tile graph:

```python
spec = ModelExecutionSpec(tile=True, tile_shape=(128, 128))

tile_pipeline = spec.create_tile_pipeline(
    tile_graph,
    input_key="x",
    tile_input_key="tile.x",
    output_name="local",
    output_key="local.out",
)
```

This creates the `TilePolicy` and `TilePipelineSpec`, but it does not create the
tile graph itself. The caller still supplies `tile_graph`.

The global-local branch can be built from caller-provided components:

```python
global_local_spec = ModelExecutionSpec(
    context="global_local",
    tile=True,
    tile_shape=(128, 128),
)

tile_pipeline = global_local_spec.create_tile_pipeline(
    tile_graph,
    input_key="x",
    tile_input_key="tile.x",
    output_name="local",
)

global_local_pipeline = global_local_spec.create_global_local_pipeline(
    global_graph=global_graph,
    tile_pipeline=tile_pipeline,
    fusion=fusion,
    global_output_name="global",
    fused_output_key="fused.out",
)

global_local_plan = global_local_spec.create_plan(
    global_local_pipeline=global_local_pipeline,
)
global_local_model = ExecutorModel(
    spec=global_local_spec,
    plan=global_local_plan,
)
```

This still does not create the global graph, tile graph, or fusion module. It
only connects already-built components through the executor pipeline classes.

The stateful/stream branch can be built from a caller-provided stream graph and
explicit state routes:

```python
stream_spec = ModelExecutionSpec(stateful=True)

stream_pipeline = stream_spec.create_stream_pipeline(
    stream_graph,
    chunk_input_key="chunk.x",
    output_name="step",
    outputs_key="stream.outputs",
    state_routes=[state_route],
)

stream_plan = stream_spec.create_plan(stream_pipeline=stream_pipeline)
stream_model = ExecutorModel(spec=stream_spec, plan=stream_plan)
```

This creates the `StreamPipelineSpec`, but it does not create chunks or infer
state routes. The caller still supplies ordered chunks at run time and explicit
`StateRoute` values when state carry is needed.

```python
outputs = stream_pipeline.run(store, chunks)
```

## ExecutorModel

`ExecutorModel` is the thin executor-layer wrapper above `ModelExecutionSpec`,
`ExecutorPlan`, and `ExecutorRunner`.

```text
ModelExecutionSpec
  public naming and validation

ExecutorPlan
  validated executor component wiring

ExecutorRunner
  dispatches execution to the selected top-level component

ExecutorModel
  binds the three pieces and exposes run(...)
```

It is intentionally still not a public trainable model abstraction:

```text
ExecutorModel is not:
  torch.nn.Module
  an automatic graph builder
  an automatic pipeline builder
  a training loop
  an optimizer / scheduler owner
```

Basic construction can start from already-built components. `ModelExecutionSpec`
validates the public model-side intent; it does not create or reconfigure the
supplied pipelines.

```python
from enn_torch_dev.executor import ExecutorModel, ModelExecutionSpec

spec = ModelExecutionSpec(
    tile=True,
    tile_shape=(128, 128),
)

tile_pipeline = spec.create_tile_pipeline(
    tile_graph,
    input_key="x",
    tile_input_key="tile.x",
    output_name="local",
)

model = ExecutorModel.from_components(
    spec,
    tile_pipeline=tile_pipeline,
)

out = model.run(store)
```

The actual tiling behavior comes from the `TilePolicy` inside `tile_pipeline`.
In v0, `tile_shape` documents and validates the public spec intent; callers
should either use `create_tile_pipeline(...)` or construct the supplied
`TilePipeline` with a matching policy.

The explicit form is also valid:

```python
spec = ModelExecutionSpec(stateful=True)
plan = spec.create_plan(stream_pipeline=stream_pipeline)
model = ExecutorModel(spec=spec, plan=plan)

outputs = model.run(store, chunks=chunks)
```

Execution is delegated to `ExecutorRunner`:

```text
plain
  -> graph.run(store)

tile
  -> tile_pipeline.run(store)

stateful / stream
  -> stream_pipeline.run(store, chunks)

global_local
  -> global_local_pipeline.run(store)
```

When `stateful=True` is combined with tile or global-local context, `ExecutorModel`
does not synthesize nested execution. The supplied or factory-created
`stream_pipeline` remains the outer executable component, and any
tile/global-local behavior must already be represented inside that stream
pipeline.

## torch.nn.Module Model wrapper

The public `Model(...)` wrapper sits above `ExecutorModel`. It is a
`torch.nn.Module`, but it preserves the executor boundary instead of absorbing
graph and pipeline construction.

```text
Model
  torch.nn.Module-facing API
  owns public forward(...) shape

ModelExecutionSpec
  validates context / tile / stateful intent

ExecutorModel
  runs a validated ExecutorPlan through ExecutorRunner
```

The wrapper may:

```text
hold ModelExecutionSpec / ExecutorModel
normalize forward(...) inputs into a KVStore or caller-provided store
call executor_model.run(...)
require chunks for stateful/stream execution
expose one public entry point for plain, tile, global-local, and stateful modes
```

The wrapper should not:

```text
auto-build GraphExecutor branches
auto-build global/local branch modules or fusion
infer StateRoute values
chunk streams automatically
own training loops, optimizers, or schedulers
```

`Model.forward(...)` delegates to `ExecutorModel.run(...)`.

```python
model = Model.from_components(spec, graph=graph)
out = model(store)
```

For stream execution, chunks remain explicit:

```python
model = Model.from_components(spec, stream_pipeline=stream_pipeline)
outputs = model(store, chunks=chunks)
```

The same explicit stream path can be assembled through `ModelBuilder` when the
caller provides the per-chunk module and graph key metadata:

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
outputs = model(store, chunks=chunks)
```

Branch, fusion, state-route, pipeline, and chunking automation remain in a
separate future builder layer. That keeps `Model(...)` as the PyTorch-facing
wrapper and keeps executor composition testable through `ModelExecutionSpec`,
`ExecutorPlan`, and `ExecutorModel`.

## GraphBuilder

`GraphBuilder` is a thin convenience layer for constructing leaf-node
`GraphExecutor` instances. It builds graphs, not executor plans, pipelines, or
models.

```text
nn.Module + input/output key metadata
  -> NodeSpec / KeyRef
  -> GraphExecutor
```

The builder may:

```text
convert string input keys to KeyRef
preserve explicit KeyRef optional/default policies
create NodeSpec objects for caller-provided nn.Module nodes
return a validated GraphExecutor
```

The builder should not:

```text
create tile/global-local/stream pipelines
create ModelExecutionSpec / ExecutorPlan / ExecutorModel / Model
infer StateRoute values
create fusion modules
own training loops, optimizers, or schedulers
```

Example:

```python
builder = GraphBuilder()

builder.add(
    name="encode",
    module=encoder,
    input_args=["x"],
    output_key="encoded",
)
builder.add(
    name="head",
    module=head,
    input_args=["encoded"],
    output_key="logits",
)

graph = builder.build()
```

## Global/local fusion

`GlobalLocalPipeline` is a separate orchestration layer for combining a global branch and a local/tiled branch.

```text
global_graph
  sees the whole input

TilePipeline
  sees local tiles

LocalGlobalFusion
  combines global_out and local_out
```

This is not stream mode. It is a global/local tiled fusion flow.

## nn modules are not executor modes

These are model components, not execution modes:

```text
GlobalSelfAttentionBlock
LocalGlobalFusion
RecurrentContextHead
Composer
Compressor
Reducer
ConvMixer
```

They can be attached to a `GraphExecutor` node or called by a model wrapper, but they do not define tile or stream behavior.

A useful mental split:

```text
executor/
  routing, graph execution, tiling, streaming, state keys

nn/
  learnable modules and tensor transformations
```

## Current supported behavior

Supported:

```text
ExecutorModeSpec
  declarative mode flags for higher-level wrappers

ExecutorPlan
  validates mode flags against supplied executor components

ModelExecutionSpec
  public context / tile / stateful schema and executor factory helpers

ExecutorRunner
  executes a validated ExecutorPlan

ExecutorModel
  binds ModelExecutionSpec, ExecutorPlan, and ExecutorRunner

Model
  thin torch.nn.Module wrapper around ExecutorModel
  delegates forward(...) to ExecutorModel.run(...)

GraphBuilder
  convenience layer for building GraphExecutor leaf-node graphs
  from nn.Module + key metadata

ModelBuilder
  plain / tile / stream GraphBuilder -> Model convenience layer
  does not build global-local pipelines

GraphExecutor
  dependency-ordered node execution

NodeSpec.output_keys
  tuple/list multi-output routing

TilePipeline
  split -> per-tile graph execution -> reconstruct

GlobalLocalPipeline
  global graph + TilePipeline + LocalGlobalFusion

StateRoute
  state input/output key routing
  carry
  reset
  detach / clone policy for carried state

StreamPipeline
  ordered chunk execution
  chunk-local store isolation
  explicit state carry
  optional reset at stream start
```

Not yet supported:

```text
automatic stream chunking
async / online runner
state cache by stream id
batch-level reset masks
state route inference / detach intervals
truncated BPTT scheduler
multi-output stream collection
automatic branch / fusion builder
global-local ModelBuilder mode
```

## Future builder layer

Builder names should describe the object they create. Avoid a broad
`ExecutorBuilder` name because the executor package already has several
executor-facing concepts: `GraphExecutor`, `ExecutorPlan`, `ExecutorRunner`,
and `ExecutorModel`.

Use the builder names as separate layers:

```text
GraphBuilder
  nn.Module + key metadata
  -> GraphExecutor

BranchBuilder
  caller-provided modules / graphs
  + branch wiring rules
  -> local graph / global graph / stream graph components

ModelBuilder
  GraphBuilder / branch builders
  + ModelExecutionSpec factory helpers
  -> Model
```

`GraphBuilder` builds `GraphExecutor` objects and should stay focused on graph
construction. It should not create pipelines, plans, or models.

A future `BranchBuilder` may make repeated branch wiring less verbose, but it
should still use caller-provided modules or graphs. It should not invent model
architecture, create fusion modules implicitly, infer `StateRoute` values, or
chunk streams automatically.

`ModelBuilder` currently provides the plain graph, explicit tiled, and explicit
stream paths. Future extensions may assemble already-built graph/branch/pipeline
components into `Model`. They should still delegate validation to
`ModelExecutionSpec`, `ExecutorPlan`, and the executor pipeline specs.

## Recommended next layer

After the plain/tile/stream `ModelBuilder`, the next higher-level layer can add
optional `BranchBuilder` helpers and global-local `ModelBuilder` modes using
the public naming above and the validated executor plan.

```text
public Model parameters
  -> context / tile / stateful
  -> GraphBuilder for caller-provided modules
  -> optional BranchBuilder for repeated branch wiring
  -> ModelExecutionSpec
  -> ExecutorModeSpec
  -> ExecutorPlan
  -> ModelExecutionSpec factory helpers for caller-provided components
  -> ModelBuilder convenience assembly
  -> Model
  -> ExecutorModel
  -> ExecutorRunner
  -> concrete GraphExecutor / TilePipeline / StreamPipeline / GlobalLocalPipeline
```

The wrapper should preserve the layer split:

```text
context  -> model branch structure
tile     -> local positional partitioning
stateful -> state carry across calls/chunks
```

Do not merge tile and stream into one concept. They solve different problems.
