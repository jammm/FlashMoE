# Gluon gfx1250 Persistent Megakernel

This document describes the Gluon implementation under `flashmoe_gluon/`. The
current optimized path is a single host-visible Gluon dispatch with persistent
resident programs, device-side dynamic task scheduling, TDM/WMMA tiled expert
compute, and warp-specialized processor task bodies.

## Current Code State

`forward_megakernel(...)` is the public Gluon entry point. For `H` and `I`
dimensions divisible by 64, it launches one persistent Gluon kernel that:

- initializes scratch state inside the kernel
- computes routing logits, top-k choices, and normalized route probabilities
- appends accepted routes into expert-local route storage
- seeds a device task queue with GEMM0 work
- runs a persistent OS-style scheduler in program 0
- runs processor actors in programs 1..N-1
- executes GEMM0 and GEMM1 with TDM, LDS staging, mbarriers, and WMMA
- dynamically enqueues GEMM1 work as soon as each route block finishes GEMM0
- combines route outputs into the final output tensor inside the same dispatch

The scalar persistent FFN path remains as a fallback for unsupported boundary
shapes. `forward_scalar_top1_debug(...)` is kept as a small compiler/runtime
probe.

## Single Dispatch Shape

The optimized Gluon path is one MoE dispatch from the host:

```text
forward_megakernel(...)
  one Gluon kernel launch
    initialize scratch
    route tokens and build expert-local route records
    seed task_queue with GEMM0 tasks
    program 0: schedule ready processors from task_queue
    programs 1..N-1: execute processor loop
      GEMM0 task: TDM gather/load + WMMA + activation/gate + hidden store
      GEMM0 completion: publish downstream GEMM1 tasks when ready
      GEMM1 task: TDM load + WMMA + output combine
    send terminal task to each processor
```

The host does not launch separate routing, GEMM, activation, combine, or queue
maintenance kernels for the MoE body. All scratch initialization, scheduling,
producer/compute synchronization, and output accumulation happen inside the
same kernel launch.

## Persistent Program Model

The launch uses resident Gluon programs. Program 0 acts as the scheduler during
the compute portion. Every other program is a processor actor.

A processor repeats this loop until it receives the terminal task id:

```text
publish_ready()
task = wait_for_scheduler_mailbox()
while task is not terminal:
    execute task
    publish any dependent task records
    publish_ready()
    task = wait_for_scheduler_mailbox()
```

The scheduler polls `processor_ready[p]` and assigns task ids through
`processor_mailboxes[p]`. It exits only after all real task slots have been
issued and every processor has reported ready again. At that point it sends the
terminal task id to each processor.

## Dynamic Task Queue

The optimized TDM/WMMA path uses a single queue for both GEMM stages:

- `task_queue`: compact int32 task records
- `task_tail`: number of published task slots
- `tile_sync`: per `(expert, route_block)` GEMM0 completion counters

Initialization writes `-1` to all queue slots, writes all initial GEMM0 task
ids, clears `tile_sync`, then release-publishes `task_tail = gemm0_tasks`. The
scheduler consumes queue slots in order. If it sees a reserved-but-not-yet
published slot, it leaves the processor ready and retries later.

GEMM0 tasks are encoded as:

```text
0 <= task < gemm0_tasks
expert = task / (route_blocks * i_tiles)
route_block = (task % (route_blocks * i_tiles)) / i_tiles
i_tile = task % i_tiles
```

GEMM1 tasks are encoded after the GEMM0 range:

```text
gemm1_id = task - gemm0_tasks
expert = gemm1_id / (route_blocks * h_tiles)
route_block = (gemm1_id % (route_blocks * h_tiles)) / h_tiles
h_tile = gemm1_id % h_tiles
```

After a GEMM0 task stores its hidden tile, the processor increments
`tile_sync[expert, route_block]`. The last completed `I` tile for that route
block enqueues all dependent GEMM1 `H` tiles. This removes the old global
GEMM0/GEMM1 phase split: downstream work becomes visible as soon as its route
block is ready.

## Warp-Specialized Processor Tasks

Each TDM/WMMA task body uses `gl.warp_specialize(...)` with three partitions:

- producer partition: issues TDM gather/load operations into LDS buffers
- compute partition: waits on load-ready mbarriers and runs WMMA
- default epilogue partition: waits on accumulator-ready mbarriers and stores
  hidden tiles or combines output tiles

The producer and compute partitions are worker partitions. The default
partition owns the task epilogue. The parent launch uses the default partition
size; Gluon adds the worker partitions inside the specialized region. This
matches the producer/compute/epilogue split used by the upstream gfx1250 Gluon
TDM GEMM examples.

## GEMM0 Task

A GEMM0 task computes one `(expert, route_block, I_tile)` hidden tile.

The producer partition:

1. Loads expert-local token ids for the route block.
2. Uses the TDM gather-compatible route-index layout so every participating
   lane in a producer wave sees the same row-index vector.
3. Issues `tdm.async_gather` for routed token rows.
4. Issues `tdm.async_load` for the expert up-weight tile.
5. Uses a load-ready mbarrier on the final TDM operation for each buffer.

The compute partition:

1. Waits for load-ready mbarriers.
2. Loads LDS operands with `DotOperandLayout`.
3. Accumulates with `gl.amd.gfx1250.wmma`.
4. Applies bias and activation.
5. Computes the gated/value projection when enabled.
6. Stores the hidden tile to an LDS handoff buffer and signals the epilogue.

The epilogue partition TDM-stores the hidden tile to `[E, EC, I]` scratch.
Inactive rows in a partially-filled route block may be written, but GEMM1 masks
output publication with the expert route count, so they are not observable.

## GEMM1 Task

A GEMM1 task computes one `(expert, route_block, H_tile)` output tile.

The producer partition loads the hidden tile and expert down-weight tile with
TDM. The compute partition runs WMMA over all `I` tiles and stores the FP32
accumulator to an LDS handoff buffer. The epilogue partition loads that
accumulator, adds down bias, applies route probability, and writes the final
output tile. Top-1 stores directly. Top-k uses atomic accumulation because
multiple routes can contribute to the same token output row.

## Mbarrier Protocol

Each task allocates double-buffered operand LDS and one accumulator handoff
buffer. The mbarriers are initialized as producer/consumer pairs:

- `load_empty`: compute has consumed an operand buffer
- `load_ready`: producer has filled an operand buffer
- `acc_empty`: epilogue has consumed the accumulator handoff buffer
- `acc_ready`: compute has produced the accumulator handoff buffer

TDM operations signal the mbarriers attached to the final async copy for a
buffer. Compute-side arrivals use the compute partition's participating lanes.
The phase counter toggles across the double-buffered operand ring.

## Routing And Expert Capacity

Routing computes logits from `[S, H] x [H, E]`, selects top-k experts, and
normalizes probabilities. Each accepted route appends:

- token id
- route probability
- expert-local slot

`expert_capacity` bounds the number of routes consumed per expert. Routes past
capacity are not published into the route storage used by GEMM0. The same route
records feed GEMM0 gathers and GEMM1 output combine.

## rocSHMEM Path

The rocSHMEM device shim is separate from the local single-GPU Gluon MoE path.
It links the gfx1250 rocSHMEM device bitcode through Gluon extern libraries and
exposes PE queries, remote pointer lookup, workgroup put/get, nonblocking
put/get, put-with-signal, signal wait, fence, and quiet. The eventual
distributed kernel should publish communication work as queue tasks without
changing the host-visible single-dispatch shape.

## Validation

The following bounded tests have passed on the local gfx1250 setup:

- `tests/gluon_warp_specialized_probe.py`
- `tests/gluon_moe_smoke.py`
- `tests/gluon_tiled_moe_smoke.py`

The `--full` tiled matrix hit the configured 15-second command cap during
compile/run and left no Python process behind. The default tiled smoke covers
`64x128`, `128x64`, and gated `128x128` cases against the CPU reference.
