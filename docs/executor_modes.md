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
detach intervals
truncated BPTT scheduler
multi-output stream collection
automatic Model(tile=True) / Model(stream=True) wrapper
```

## Recommended next layer

After this terminology is stable, a higher-level model API can be designed around the two modes.

```text
tile=True
  use TilePipeline / GlobalLocalPipeline

stream=True
  use StreamPipeline + StateRoute

tile=True and stream=True
  should be treated carefully:
  stream controls time/order
  tile controls spatial/local processing inside each chunk
```

Do not merge tile and stream into one concept. They solve different problems.
