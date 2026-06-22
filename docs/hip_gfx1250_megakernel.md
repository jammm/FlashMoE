# HIP gfx1250 Persistent Megakernel

This document describes the HIP/gfx1250 MoE megakernel implemented under
`csrc/include/flashmoe/hip/`. It focuses on the persistent kernel launched by
`flashmoe::forwardHost`, not the experimental Triton path.

## What Is Single and Persistent

The MoE work itself is one persistent HIP kernel launch:

```text
forwardHost(...)
  optional output clear for plural combine
  forward<Config, Activation, Topology><<<...>>>()
```

Inside `forward`, blocks do not return after one tile. They stay resident and
cooperate through global task queues, doorbells, and packet signals until the
scheduler proves that all work for the forward pass has drained.

The current CUDA-parity public flow still prepares routing metadata before the
MoE launch. The megakernel consumes:

- `expertCounts[E]`
- `Context::tokenIndices[E, roundEC]`
- expert weights, bias tensors, symmetric heap pointers, and signal buffers

For strict one-dispatch host-visible behavior, the caller must also handle any
output-scratch lifecycle contract up front. In the current HIP host wrapper,
plural combine clears `moeOut` with `hipMemsetAsync` before launching the
persistent kernel.

## Block Roles

`forward` reserves the final block in the grid as the OS block:

- `blockIdx.x == gridDim.x - 1`: OS block
- all other blocks: processor-capable blocks

Some processor-capable blocks first run token dispatch if
`blockIdx.x < dispatchBlocks`. After dispatch, they enter the normal processor
loop. The OS block never performs GEMM work; it runs the scheduler and
subscriber waves.

## High-Level Lifecycle

1. The host launches `forward`.
2. Dispatch blocks copy routed tokens into the symmetric heap and publish packet
   signals.
3. The OS block initializes shared scheduling state.
4. Subscriber waves in the OS block decode packet signals into `Task` records.
5. The scheduler wave assigns ready tasks to ready processor blocks by writing
   per-processor doorbells.
6. Processor blocks execute `GEMM0`, `GEMM1`, or `combine` tasks.
7. Completed tasks publish the next task stage or final packet signal.
8. When the scheduler has seen all task bounds drain, it sends interrupt
   doorbells to processors and interrupts subscriber waves.
9. Blocks exit the resident loops and the single kernel returns.

No host-side GEMM library launch occurs inside this lifecycle.

## Dispatch Stage

`dispatch.hip.cuh` uses the router output to move tokens into expert-local
storage in the symmetric heap.

For each expert, dispatch:

- reads the expert lookup entry and clamped expert token count
- reads token ids from `Context::tokenIndices` using `[E, roundEC]` layout
- copies token rows into per-expert heap storage
- publishes initial packet signals for subscribers

Local/P2P paths write directly into the mapped heap region. Remote paths go
through the rocSHMEM compatibility layer. Both paths feed the same packet and
task machinery after the signal is visible.

## OS Block

The OS block is implemented by `os::start` in `moe.hip.cuh`.

It splits its waves into:

- one scheduler cohort
- subscriber cohorts

The OS block builds shared-memory views over:

- ready processor queues
- per-subscriber task queue heads
- task bounds
- visited bitsets for global queues
- interrupt flags

The scheduler watches local subscriber queues and global task queue heads. It
matches ready tasks with ready processor blocks and writes a `TQSignal` doorbell
to each selected processor.

Subscribers watch packet signals. Initial packet signals become `GEMM0` tasks.
Final packet signals become `combine` tasks.

## Processor Blocks

Each processor block runs `processor::start`.

The first wave in the processor block waits on its doorbell:

```text
load TQSignal from ctx.tqs[processor_id]
if interrupt: exit
else: load Task from task queue
```

The task is copied into shared memory, then the whole block executes the task.
After completion, the processor marks itself ready and waits for another
doorbell. This loop is why the kernel is persistent: processor blocks are not
one-shot tile kernels, they are resident workers.

## Task Flow

### GEMM0

`GEMM0` computes the expert up projection:

```text
routed token tile x expertUpWeights -> intermediate tile
```

For gated MLPs, `fGET_gated` computes both gate and value projections:

```text
gate  = activation(tokens x expertUpWeights + biasUp)
value = tokens x expertUpVWeights + biasUpV
out   = gate * value
```

When all `GEMM0` tiles for a token/expert M tile are complete, the processor
uses `notifyNext` to enqueue the dependent `GEMM1` tasks.

### GEMM1

`GEMM1` computes the expert down projection:

```text
intermediate tile x expertDownWeights -> output tile
```

After `GEMM1`, local/P2P tasks publish a final packet signal directly. Remote
tasks use the rocSHMEM compatibility path to put the output tile and signal its
completion.

### Combine

`combine` tasks read completed output tiles and scatter/reduce them into
`moeOut` using the token ids and route probabilities encoded by routing.

Plural combine uses atomic accumulation into `moeOut`, which is why the current
host wrapper clears the output buffer before the persistent launch.

## The Device-Side GEMM Object

The HIP path does not call cuBLASDx. It implements a device-side GEMM object
with these layers:

- `tile::CollectiveMainloop`: compile-time tile shape, wave size, WMMA shape,
  shared-memory sizing, and compatibility with the original mainloop pattern
- `tile::WmmaTileGemm`: loads rocWMMA fragments, calls `rocwmma::mma_sync`, and
  stores accumulator fragments
- `wmma_tdm::computeTileRowMajorB`: gfx1250 processor/gate helper that stages
  tiles through TDM into LDS and then runs rocWMMA

`moe.hip.cuh` instantiates:

```text
TileGEMM0 = tile::CollectiveMainloop<...>
TileGEMM1 = tile::CollectiveMainloop<...>
```

Those types are passed into `processor::start`, then into `fGET` or
`fGET_gated`. For gfx1250 FP16/BF16 with FP32 accumulation and legal tile
shapes, `fGET` and `fGET_gated` call:

```text
wmma_tdm::computeTileRowMajorB<TileGEMM, Element, AccumType>(...)
```

The fallback scalar path remains for non-WMMA boundary cases.

## TDM and LDS Completion

The gfx1250 WMMA path stages one WMMA A subtile and one WMMA B subtile per wave
into LDS scratch. The helper allocates per-wave LDS scratch plus one LDS
completion barrier per wave in the kernel workspace.

For each K step:

1. issue TDM load for the A subtile into per-wave LDS
2. issue TDM load for the B subtile into per-wave LDS
3. wait for the LDS completion phase to advance
4. load rocWMMA fragments from LDS
5. call `rocwmma::mma_sync`

The steady-state GEMM math therefore remains inside the persistent kernel. TDM
and LDS barriers only stage data for the in-kernel WMMA work; they do not
launch another kernel or call a host library.

## Why It Stays One Persistent Megakernel

The kernel stays persistent because all dynamic scheduling state lives on the
device:

- routing metadata is already in device buffers
- token dispatch writes packet signals from inside `forward`
- subscribers convert packet signals to device `Task` records
- the scheduler assigns tasks by writing device doorbells
- processor blocks poll doorbells and execute tasks in a loop
- GEMM0/GEMM1 are in-kernel WMMA/TDM calls
- combine is an in-kernel task
- completion is represented by device counters, queue heads, packet signals,
  and scheduler interrupts

The host does not enqueue per-expert GEMM kernels, per-stage combine kernels,
or communication kernels while the megakernel is running. Once `forward` is
launched, dispatch, expert compute, communication signaling, and combine are
driven by resident GPU blocks until the OS block sends termination interrupts.

## Important Boundaries

- The router/gate launch that produces `expertCounts` and `tokenIndices` is a
  separate CUDA-parity boundary today.
- The MoE megakernel itself is the persistent `forward` launch.
- Plural combine currently requires `moeOut` to be cleared before the
  persistent launch.
- hipBLASLt/rocBLAS are only suitable for standalone host-launched GEMM paths;
  they cannot be used inside this persistent kernel without breaking the model.
