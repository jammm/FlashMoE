# Gluon gfx1250 Persistent Megakernel

This document describes the Gluon implementation under `flashmoe_gluon/`. The
current optimized path is a single host-visible Gluon dispatch with persistent
resident programs, device-side dynamic task scheduling, TDM/WMMA tiled expert
compute, and warp-specialized processor task bodies.

## Current Code State

`forward_megakernel(...)` is the main Gluon entry point. For `H` and `I`
dimensions divisible by 64, it launches one persistent Gluon kernel that:

- initializes scratch state inside the kernel
- computes routing logits, top-k choices, and normalized route probabilities
- appends accepted routes into expert-local route storage
- seeds a device task queue with GEMM0 work
- runs all resident programs as persistent workers
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
    all programs: claim tasks from task_queue
      GEMM0 task: TDM gather/load + WMMA + activation/gate + hidden store
      GEMM0 completion: publish downstream GEMM1 tasks when ready
      GEMM1 task: TDM load + WMMA + output combine
    terminal task exits all roles
```

The host does not launch separate routing, GEMM, activation, combine, or queue
maintenance kernels for the MoE body. All scratch initialization, scheduling,
producer/compute synchronization, and output accumulation happen inside the
same kernel launch.

## Persistent Program Model

The local launch uses resident Gluon programs as workers. There is no separate
local scheduler program in the TDM/WMMA path. The epilogue role in each worker
claims the next task from the global queue with a CAS on `task_head`, then
broadcasts that task to the compute and producer roles through the worker's
mailbox.

A worker repeats this loop until it receives the terminal task id:

```text
task = claim_task_queue()
while task is not terminal:
    execute task
    publish any dependent task records
    task = claim_task_queue()
```

The mailbox is still used inside each worker because `gl.warp_specialize`
splits the task body into epilogue, compute, and producer roles. The epilogue
role owns queue claims and publishes `task + 1` to the per-worker mailbox. The
other roles wait for a new mailbox value, execute their part of the same task,
and exit when they observe the terminal task.

## Dynamic Task Queue

The optimized TDM/WMMA path uses a single queue for both GEMM stages:

- `task_queue`: compact int32 task records
- `task_head`: next task slot to claim
- `task_tail`: number of published task slots
- `tile_sync`: per `(expert, route_block)` GEMM0 completion counters

Initialization writes `-1` to all queue slots, writes all initial GEMM0 task
ids, clears `tile_sync`, then release-publishes `task_head = 0` and
`task_tail = gemm0_tasks`. Workers claim slots by advancing `task_head`. If a
worker claims a reserved-but-not-yet-published slot, it waits until the producer
of that slot stores the task id with release ordering.

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

After a GEMM0 task stores its hidden tile, the worker increments
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
output writes with the expert route count, so they are not observable.

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

The distributed kernel uses explicit persistent roles inside the same launch:

- program 0 runs the device scheduler and the rocSHMEM subscriber/publisher
  partition
- one or more dispatch programs compute routing, fill local dispatch records,
  and publish remote dispatch channels
- the remaining programs are compute workers that consume dynamically enqueued
  GEMM and combine tasks

Dispatch programs do not enter the compute worker loop after publication. This
keeps workgroup-level rocSHMEM dispatch calls out of the later nested
warp-specialized TDM/WMMA task bodies while preserving one host-visible kernel
launch for the whole MoE body.

The rocSHMEM context owns symmetric dispatch and result buffers. Dispatch uses
per `(owner_pe, source_pe, local_expert)` channels containing count, token id,
route probability, and packed token rows. Dispatch programs write remote
dispatch channel payloads, complete the outstanding remote writes, and publish
the channel epoch. Program 0 waits for incoming dispatch epochs, seeds the
dynamic GEMM queue, schedules ready compute workers, and publishes completed
result channels as the workers enqueue them.

The distributed GEMM tasks reinterpret each `(source_pe, local_expert)` channel
owned by the local PE as a local tiled expert source:

- GEMM0 loads packed dispatch rows from symmetric memory and local up weights
  with TDM, runs WMMA, applies activation/gating, and stores hidden tiles.
- GEMM1 loads hidden tiles and local down weights with TDM, runs WMMA, adds
  down bias, and writes FP32 result tiles into the symmetric result buffer.
- Empty channel blocks still advance completion counters, but skip the TDM/WMMA
  body.
- Remote result transfer is centralized in program 0 through a publish queue,
  so post-`warp_specialize` compute workers do not call rocSHMEM workgroup
  collectives.

Remote channels use rocSHMEM signal operations for visibility. Data-bearing
dispatch channels write the split count/id/probability arrays first, then use a
rocSHMEM put-with-signal operation for the packed token rows. Empty dispatch and
result channels publish only the signal word with a rocSHMEM uint64 signal
operation. This keeps cross-rank visibility on rocSHMEM APIs without manual
address translation.

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

- `git diff --check`

The low-level rocSHMEM checks that passed with two local PEs are:

- `tests/gluon_rocshmem_packet_probe.py --mode signal-wg`
- `tests/gluon_rocshmem_packet_probe.py --mode signal-wave`
- `tests/gluon_rocshmem_cross_rank_probe.py --mode result`
- `tests/gluon_rocshmem_cross_rank_probe.py --mode tdm-dispatch`

The dynamic scheduling and megakernel checks that passed with short timeouts
are:

- `tests/gluon_dynamic_ws_probe.py --num-programs 2`
- `tests/gluon_moe_smoke.py --s 1 --h 64 --i 64 --e 2 --top-k 1 --expert-capacity 2 --num-programs 2`
- `tests/gluon_rocshmem_megakernel_smoke.py --case local`
- `tests/gluon_rocshmem_megakernel_smoke.py --case remote`
- `tests/gluon_rocshmem_megakernel_smoke.py --case mixed`
- `tests/gluon_rocshmem_megakernel_smoke.py --case remote01`
- `tests/gluon_rocshmem_megakernel_smoke.py --case remote10`
- `tests/gluon_rocshmem_megakernel_smoke.py --case remote --repeats 2`
- `tests/gluon_rocshmem_megakernel_smoke.py --case remote --activation silu --gated`
- `tests/gluon_rocshmem_megakernel_smoke.py --case mixed --activation silu --gated`
- `tests/gluon_rocshmem_megakernel_smoke.py --case remote --activation silu --gated --repeats 2`

The rocSHMEM smoke tests use two local PEs on the same GPU and compare each
rank's output against the CPU reference. The validated smoke shape is
`S=1`, `H=64`, `I=64`, one local expert per PE, `expert_capacity=2`, and
`FLASHMOE_NUM_PROGRAMS=4`. The gated runs validate the SiLU/value-projection
path used by the paper-style MoE MLP.
