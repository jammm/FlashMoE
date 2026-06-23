# Gluon gfx1250 Persistent Megakernel

This document describes the Gluon implementation path under
`flashmoe_gluon/`. It covers the currently runnable persistent Gluon
megakernel, the decomposed staging path used to validate Gluon/TDM/WMMA
building blocks, and the remaining work to generalize the optimized tiled
compute body.

## Current Code State

The current package has three separate surfaces:

- `forward_megakernel(...)`: the public API for the Gluon megakernel. It
  launches one Gluon kernel with resident programs, initializes scratch state
  inside the kernel, computes routing and top-k normalization, appends
  expert-local route records, claims route tasks from a device-side queue,
  runs the expert FFN, applies activation or gated/SwiGLU, and combines route
  outputs into the final tensor. The scheduler is persistent. For aligned
  `H=I=64` cases, GEMM0 and GEMM1 use an in-kernel TDM/WMMA tile path covering
  top-1, top-2, and gated/SILU smoke coverage. Other shapes currently use the
  scalar persistent FFN fallback.
- `forward_scalar_top1_debug(...)`: a smaller single-kernel bring-up probe kept
  for compiler/runtime debugging.
- `forward_decomposed_staging(...)`: a bring-up path that uses upstream Gluon
  MoE matmul launches for routed GEMM0 and GEMM1. This path validates routing
  metadata, TDM gather/scatter, WMMA matmul, expert weight layout, and combine
  math, but it is not the final megakernel because it uses multiple launches.

The runnable `forward_megakernel(...)` path has been exercised against CPU
reference for top-1 and top-2 vanilla identity MLP, and for top-1 and top-2
gated/SILU MLP. The default smoke configuration uses two resident Gluon
programs. Dedicated probes validate scalar atomic participation, global phase
barriers, and single-workgroup TDM/WMMA gather.

The next implementation step is to generalize the TDM/WMMA compute body to
non-64 hidden/intermediate dimensions while preserving the same in-kernel route
queue and single-dispatch lifecycle.

## What The Single Dispatch Does

The current Gluon megakernel is one host-visible MoE dispatch:

```text
forward_megakernel(...)
  one Gluon kernel launch
    in-kernel scratch initialization
    per-token routing logits
    top-k selection and probability normalization
    expert-local route queues
    OS-style scheduler and processor doorbells
    TDM/WMMA routed GEMM0 for aligned H=I=64 modes
    activation or gated epilogue
    TDM/WMMA routed GEMM1 for aligned H=I=64 modes
    scalar routed FFN fallback for remaining modes
    in-kernel combine
```

The host does not launch separate dispatch, GEMM, combine, or communication
kernels for the MoE body. Once the Gluon kernel starts, the current
implementation completes the full MoE calculation inside that single launch.
The optimized target keeps that same host-visible shape and scheduler while
expanding the TDM/WMMA tile tasks to every supported FlashMoE mode.

## Kernel Inputs

The Gluon megakernel uses the same high-level tensors as the HIP path:

- input tokens `[S, H]`
- gate weights `[H, E]`
- expert up weights `[E, H, I]`
- optional gated up/value weights `[E, H, I]`
- expert down weights `[E, I, H]`
- up/down bias tensors
- output tensor `[S, H]`
- scratch buffers for routing, queues, intermediate tiles, counters, and
  signals

The staging wrapper also creates two layout fixes required by upstream Gluon
matmul:

- expert weights are passed as logical `[K, N]` tensors with physical
  column-major last-two-dimensional strides
- bias tensors are passed as fp32 to keep Gluon branch types consistent

Those staging fixes are not a separate algorithm. They are the data-layout
contract the final megakernel must satisfy inside the kernel.

## Target Persistent Program Model

The current persistent implementation divides the kernel into resident programs
and phases:

- program 0 clears route counts, scheduler mailboxes, barriers, and output
  state
- all programs route a strided subset of tokens and append accepted routes
- all programs wait at device-side phase barriers between initialization,
  routing, scheduler reset, route processing, and exit
- during compute phases, program 0 becomes the OS scheduler
- programs 1..N-1 become Processor actors that publish readiness, wait on
  per-program doorbells, execute assigned work, and publish readiness again
- the scheduler assigns work-conserving tasks by writing task ids into
  Processor doorbells and sends a terminal task after all work has drained
- each processor computes the route FFN or tile task and combines into output

Processor programs are persistent because they do not exit after one tile. A
processor loop has this shape:

```text
while not done:
    publish_ready()
    task = wait_for_scheduler_doorbell()
    if task is terminal:
        break
    execute task
    publish dependent work
```

The scheduler exits only after every Processor has reported ready after the
last real task. It then sends each Processor a terminal task id so the
Processor loops break out and return.

The current scheduler uses phase barriers backed by device-scope atomics. The
barrier increment is a scalar Gluon side effect, which the probe verifies runs
once per program. Polling uses an acquire atomic add of zero, which behaves as
an atomic load without mutating the counter. Plain volatile polling was not
reliable for the multi-program path.

## In-Kernel Initialization

Strict single-dispatch behavior for the optimized scheduler requires scratch
state to be initialized inside the same Gluon kernel. The initialization phase
clears or seeds:

- per-expert route counts
- route index storage
- work-queue heads and tails
- per-stage completion counters
- output accumulation state when top-k combine can write the same token more
  than once
- rocSHMEM signal slots when the distributed path is enabled

Non-initializer programs wait on a device flag before consuming any scratch
state. The flag is monotonic for the forward pass, which avoids host-side
pre-clear launches and keeps the lifecycle inside the single dispatch.

## Routing And Expert Capacity

Routing computes expert logits, selects top-k experts, normalizes route
probabilities, and appends route records into expert-local storage.

Each accepted route records:

- token id
- expert id
- route probability
- expert-local slot

Expert capacity bounds how many routes for each expert are consumed by the MoE
compute stages. Routes beyond capacity must be either dropped consistently with
the reference policy or handled by a later overflow policy. The persistent
kernel must apply the same capacity rule before publishing GEMM work.

## Work Records

The scheduler works with compact task records. The minimum useful task types
are:

- `ROUTE`: build route metadata for a token tile
- `GEMM0`: compute routed token rows times expert up weights
- `ACT`: apply activation or gated/SwiGLU epilogue when it is not fused into
  GEMM0 storage
- `GEMM1`: compute intermediate rows times expert down weights
- `COMBINE`: accumulate route outputs into the final token output
- `COMM`: issue or wait on rocSHMEM operations when a route targets another PE

The implementation can fuse some logical task types as long as the task graph
still preserves the same dependencies.

## TDM And WMMA Compute

The validated staging path uses upstream Gluon MoE matmul for the compute body.
The current runnable `forward_megakernel(...)` has an integrated TDM/WMMA path
for aligned `H=I=64` FFN math and scalar persistent fallback for the remaining
shapes. The optimized kernel folds the staging compute body into processor
tasks:

1. Build TDM descriptors for routed token rows and expert weight tiles.
2. Use TDM gather to stage non-contiguous token rows into LDS.
3. Use TDM load to stage expert weight tiles into LDS.
4. Use LDS mbarriers or TDM wait operations to order the producer/consumer
   pipeline.
5. Load WMMA operands from LDS into registers.
6. Execute `gl.amd.gfx1250.wmma`.
7. Write the GEMM0 intermediate tile to scratch.
8. Consume the intermediate tile in the scheduled GEMM1 task loop.

The key point is that TDM and WMMA are instructions emitted inside the Gluon
kernel. They do not call a host GEMM library and they do not create additional
host-visible launches.

The upstream gfx1250 Gluon MoE matmul body already has the pieces needed for
this replacement: TDM descriptors, TDM gather/scatter, shared-memory staging,
`tdm.async_wait`, WMMA layouts, and `gl.amd.gfx1250.wmma`. The integration task
is to generalize the integrated tile path beyond the current aligned shape
without launching separate staging matmuls.

## Dynamic Scheduling

The Gluon scheduler now follows the same actor shape as the CUDA path. Program
0 owns scheduling during compute phases, while every other resident program is
a Processor actor.

The scheduler state is device-resident:

- `processor_ready[p]` records whether Processor `p` is ready for work.
- `processor_mailboxes[p]` holds the task id assigned to Processor `p`, or
  `-1` while no task is pending.
- a terminal task id equal to the stage task count tells a Processor to exit
  the current scheduled phase.

The scheduler loops over ready Processors, assigns the next task id to each
ready mailbox, and continues until all real tasks have been assigned. After
that, it waits for each Processor to report ready one final time and sends the
terminal task. This gives dynamic load balancing without requiring host
intervention or a separate dispatch.

## Combine

For top-1, combine can usually store directly to the token output row. For
top-k, multiple routes may contribute to the same output token. The megakernel
therefore treats combine as a device-side reduction:

```text
output[token, h] += route_probability * expert_result[route, h]
```

The persistent scheduler can publish combine work after the dependent GEMM1
tile is complete. Atomic accumulation is acceptable for the initial Gluon path;
later tuning can specialize top-1 and ordered top-k cases.

## rocSHMEM Path

The Gluon rocSHMEM shim links the gfx1250 rocSHMEM device bitcode through
`extern_libs={"rocshmem": ".../librocshmem_device_gfx1250.bc"}` and exposes the
small set of device calls FlashMoE needs:

- PE id and PE count
- remote pointer lookup
- workgroup put/get
- nonblocking put/get
- put-with-signal
- uint64 signal wait
- fence and quiet

The host runtime still has to initialize rocSHMEM, allocate symmetric memory,
and initialize the HIP module. After that bootstrap, communication can be
issued from inside the persistent Gluon kernel.

## Why The Optimized Target Stays Persistent

The optimized Gluon megakernel stays persistent because every stage after
launch is device-driven:

- scratch initialization happens in the kernel
- routing records are produced in the kernel
- work queues live in device memory
- processor programs loop over many tasks
- GEMM0 and GEMM1 use in-kernel TDM/WMMA
- combine is an in-kernel reduction task
- rocSHMEM operations are emitted through device extern calls
- scheduler state determines when all resident programs exit

No host code has to enqueue per-expert GEMM kernels or per-stage combine
kernels between those steps. The current `forward_megakernel(...)` already
preserves the one-dispatch persistent scheduler and has a TDM/WMMA tile path
for the aligned smoke-test FFN cases. The remaining paper-parity work is
generalizing that tile path beyond the current aligned shape and connecting
the rocSHMEM communication tasks.

## Implementation Checklist

1. Keep the current persistent scalar `forward_megakernel(...)` as the
   correctness baseline.
2. Keep the decomposed staging path as the Gluon data-layout and TDM/WMMA
   oracle.
3. Extend the current routed GEMM0 TDM/WMMA path beyond `H=I=64`.
4. Keep activation and gated MLP epilogues fused into the single dispatch.
5. Extend the current GEMM1 TDM/WMMA path beyond `H=I=64`.
6. Link rocSHMEM bitcode and compile a minimal in-kernel communication path.
7. Add the two-PE local smoke test after the host runtime initializes the
   rocSHMEM HIP module correctly.
