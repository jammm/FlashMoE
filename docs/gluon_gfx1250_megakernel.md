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

The local single-GPU entry point still keeps the small scalar debug/probe path
for unsupported boundary shapes. The rocSHMEM entry point requires `H` and `I`
to be divisible by 64 and uses the TDM/WMMA path directly.

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

For gated GEMM0, the value projection uses its own TDM operand buffers and
load mbarriers. This matches the CUDA path's structure, where the gate and
value projections are separate mainloop invocations. Keeping separate pipeline
state avoids reusing a partially-advanced double-buffer phase when `H` has only
one 64-column tile.

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

`forward_megakernel_rocshmem(...)` is the distributed Gluon entry point. It
uses the same host-visible single GPU dispatch shape: the host coordinates PEs
with a rocSHMEM CPU barrier before launch, then launches one persistent Gluon
kernel per PE. No host-side routing, GEMM, activation, result-fetch, or combine
kernels are launched around it.

As in the CUDA path, some worker programs perform token dispatch first and then
join the normal processor loop. The scheduler and subscriber/publisher stay in
program 0 while processor programs consume whichever compute or combine tasks
become ready.

The rocSHMEM context owns symmetric dispatch and result buffers. Dispatch uses
per `(owner_pe, source_pe, local_expert)` channels containing count, token id,
route probability, and packed token rows. Program 0 acts as the communication
and scheduler CTA: it writes remote dispatch channel payloads, completes the
outstanding remote writes, publishes the channel epoch with system-release
ordering, waits for incoming dispatch epochs, seeds the dynamic GEMM queue, and
later publishes completed result channels. Processor CTAs execute the TDM/WMMA
GEMM0 and GEMM1 tasks.

The distributed GEMM tasks reinterpret each `(source_pe, local_expert)` channel
owned by the local PE as a local tiled expert source:

- GEMM0 loads packed dispatch rows from symmetric memory and local up weights
  with TDM, runs WMMA, applies activation/gating, and stores hidden tiles.
- GEMM1 loads hidden tiles and local down weights with TDM, runs WMMA, adds
  down bias, and writes FP32 result tiles into the symmetric result buffer.
- Empty channel blocks still advance completion counters, but skip the TDM/WMMA
  body.
- Remote result publication is centralized in program 0 after the compute
  barrier, so post-`warp_specialize` processor CTAs do not call rocSHMEM
  workgroup collectives.

Reusable signal words use monotonically increasing launch epochs and `>= epoch`
waits. The context allocates signal/count words zero-initialized and advances
the epoch per forward call, so repeated launches on the same context do not need
a host-side scratch reset.

## Validation

The following local checks cover the non-distributed Gluon path:

- `tests/gluon_warp_specialized_probe.py`
- `tests/gluon_moe_smoke.py`
- `tests/gluon_tiled_moe_smoke.py`
- `tests/gluon_tiled_moe_smoke.py --full`

The tiled local smoke covers top-1, top-2, vanilla, and gated cases for
`64x128`, `128x64`, and `128x128` shapes.

For the distributed path, the current static checks are:

- `/jam/venv/bin/python -m py_compile flashmoe_gluon/megakernel.py flashmoe_gluon/rocshmem.py flashmoe_gluon/rocshmem_runtime.py tests/gluon_rocshmem_megakernel_smoke.py tests/gluon_rocshmem_cross_rank_probe.py tests/gluon_rocshmem_packet_probe.py`
- `git diff --check`

Before the final cleanup in this patch series, the targeted rocSHMEM cases had
reached CPU-reference correctness for the hot-cache remote path and the split
remote dispatch direction. The local, bidirectional remote, and mixed cases must
be rerun after the GPU runtime is healthy again.
