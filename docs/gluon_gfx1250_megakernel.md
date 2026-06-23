# Gluon gfx1250 Persistent Megakernel

This document describes the Gluon implementation path under
`flashmoe_gluon/`. It covers the currently runnable one-dispatch Gluon
megakernel, the decomposed staging path used to validate Gluon/TDM/WMMA
building blocks, and the remaining work to reach the optimized persistent
paper-parity scheduler.

## Current Code State

The current package has three separate surfaces:

- `forward_megakernel(...)`: the public API for the Gluon megakernel. It now
  launches one Gluon kernel and computes routing, top-k
  normalization, expert FFN, activation or gated/SwiGLU, and combine inside
  that one host-visible dispatch. This is a scalar correctness implementation;
  it is not yet the optimized TDM/WMMA persistent scheduler.
- `forward_scalar_top1_debug(...)`: a smaller single-kernel bring-up probe kept
  for compiler/runtime debugging.
- `forward_decomposed_staging(...)`: a bring-up path that uses upstream Gluon
  MoE matmul launches for routed GEMM0 and GEMM1. This path validates routing
  metadata, TDM gather/scatter, WMMA matmul, expert weight layout, and combine
  math, but it is not the final megakernel because it uses multiple launches.

The runnable `forward_megakernel(...)` path has been exercised against CPU
reference for top-1 and top-2 vanilla identity MLP, and for top-1 and top-2
gated/SILU MLP. The next implementation step is to replace the scalar FFN body
with the validated TDM/WMMA compute body and persistent scheduler.

## What The Single Dispatch Does

The current Gluon megakernel is one host-visible MoE dispatch:

```text
forward_megakernel(...)
  one Gluon kernel launch
    per-token routing logits
    top-k selection and probability normalization
    scalar routed GEMM0
    activation or gated epilogue
    scalar routed GEMM1
    in-kernel combine
```

The host does not launch separate dispatch, GEMM, combine, or communication
kernels for the MoE body. Once the Gluon kernel starts, the current
implementation completes the full MoE calculation inside that single launch.
The optimized target keeps that same host-visible shape but replaces the
per-token scalar loops with resident programs and device-side work scheduling.

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

The optimized target divides the kernel grid into resident programs with
device-side roles:

- initialization programs clear counters, queue heads, and scratch metadata
- scheduler programs publish work records and track global progress
- processor programs repeatedly claim ready work and execute it
- optional communication programs issue rocSHMEM operations and observe signals

Processor programs are persistent because they do not exit after one tile. A
processor loop has this shape:

```text
while not done:
    task = claim_next_ready_task()
    if task exists:
        execute task
        publish dependent work
    else:
        poll progress state
```

The scheduler exits only after all expected tasks have been published and all
completion counters have reached their terminal values. It then marks the
global exit state so processor programs break out of their loops and return.
The current scalar `forward_megakernel(...)` does not yet implement this
resident scheduler.

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

The validated staging path uses upstream Gluon MoE matmul for the compute
body. The current runnable `forward_megakernel(...)` uses scalar loops for
correctness. The optimized kernel folds the staging compute body into processor
tasks:

1. Build TDM descriptors for routed token rows and expert weight tiles.
2. Use TDM gather to stage non-contiguous token rows into LDS.
3. Use TDM load to stage expert weight tiles into LDS.
4. Use LDS mbarriers or TDM wait operations to order the producer/consumer
   pipeline.
5. Load WMMA operands from LDS into registers.
6. Execute `gl.amd.gfx1250.wmma`.
7. Write the GEMM0 intermediate tile or GEMM1 output tile.

The key point is that TDM and WMMA are instructions emitted inside the Gluon
kernel. They do not call a host GEMM library and they do not create additional
host-visible launches.

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
kernels between those steps. The current scalar `forward_megakernel(...)`
already preserves the one-dispatch API shape; the remaining paper-parity work
is replacing the scalar body with the persistent TDM/WMMA task scheduler.

## Implementation Checklist

1. Keep the current scalar `forward_megakernel(...)` as the one-dispatch
   correctness baseline.
2. Keep the decomposed staging path as the Gluon data-layout and TDM/WMMA
   oracle.
3. Move routed GEMM0 into the single Gluon kernel with TDM gather and WMMA.
4. Keep activation and gated MLP epilogues fused into the single dispatch.
5. Move GEMM1 and combine onto the same TDM/WMMA task path.
6. Add in-kernel scratch initialization and persistent work queues.
7. Link rocSHMEM bitcode and compile a minimal in-kernel communication path.
8. Add the two-PE local smoke test after the host runtime initializes the
   rocSHMEM HIP module correctly.
