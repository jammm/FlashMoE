# Gluon Megakernel Plan for gfx1250

## Recommendation

Use Gluon as the next Python/JIT implementation path. Keep the
distributed runtime surface small and owned by FlashMoE.

Gluon is a better fit for the local gfx1250 megakernel work because the TheRock
Triton stack already includes:

- gfx1250 Gluon imports in the venv
- gfx1250 TDM tensor descriptors
- TDM async load/store/gather/scatter
- LDS mbarriers
- cluster barriers and `num_ctas`
- AMD WMMA layouts and gfx1250 WMMA examples
- persistent GEMM examples using TDM and WMMA

The main missing piece is not local compute. The main missing piece is a small
rocSHMEM device-call bridge for Gluon if we want communication inside the
Gluon kernel.

## Current Status

The initial `flashmoe_gluon/` package now exists. It provides:

- a gfx1250 Gluon backend loader for `/jam/triton`
- a small rocSHMEM extern-call shim using direct device-bitcode symbols
- static probes for backend import and rocSHMEM symbol discovery
- a scalar top-1 debug kernel for compiler bring-up
- a decomposed staging path that uses upstream Gluon MoE matmul for routed
  GEMM0 and GEMM1

The decomposed staging path is not the final megakernel because it still uses
multiple launches. It exists only to validate the Gluon/TDM/WMMA compute body,
routing metadata layout, and data layout requirements before folding the logic
into the single persistent dispatch.

Validated staging cases:

- top-1, two experts, tiny hidden/intermediate dimensions
- top-2, two experts, tiny hidden/intermediate dimensions
- top-2, four experts, small hidden/intermediate dimensions

## Evidence From Local Sources

Relevant local source paths:

- `/jam/triton/python/triton/experimental/gluon/`
- `/jam/triton/python/triton/experimental/gluon/language/amd/gfx1250/tdm.py`
- `/jam/triton/python/triton/experimental/gluon/language/amd/gfx1250/mbarrier.py`
- `/jam/triton/python/triton/experimental/gluon/language/amd/gfx1250/cluster.py`
- `/jam/triton/third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py`
- `/jam/triton/third_party/amd/python/examples/gluon/f16_gemm_warp_pipeline_gfx1250.py`
- `/jam/triton/third_party/amd/python/examples/gluon/moe_gfx1250.py`
- `/jam/triton/third_party/amd/backend/compiler.py`

The venv imports Gluon and the gfx1250 modules. The venv ROCm SDK also contains
rocSHMEM headers, `librocshmem.a`, and `librocshmem_device_gfx1250.bc`.

## What Gluon Gives Us

### Persistent Program Structure

The gfx1250 GEMM examples already use persistent-style work assignment:

```text
pid = program_id(0)
num_sms = num_programs(0)
for tile_idx in range(pid, total_tiles, num_sms):
    compute tile
```

That maps naturally to FlashMoE processor blocks. A Gluon program can stay
resident, claim work from global queues with atomics, and loop until a device
termination condition is reached.

### In-Kernel GEMM

Gluon exposes AMD WMMA layouts and gfx1250 WMMA calls. The examples build:

- shared layouts for A/B tiles
- TDM descriptors
- LDS double or multi-buffering
- WMMA accumulator layouts
- `ttgl.amd.gfx1250.wmma(a, b, accumulator)`

This can replace the current scalar Triton `forward_megakernel` inner loops with
a tiled WMMA/TDM body.

### TDM and mbarrier

Gluon has first-class gfx1250 TDM APIs:

- `make_tensor_descriptor`
- `update_tensor_descriptor`
- `async_load`
- `async_store`
- `async_gather`
- `async_scatter`
- `async_wait`

It also has LDS mbarriers:

- `mbarrier.init`
- `mbarrier.arrive`
- `mbarrier.wait`

TDM async operations accept an optional mbarrier handle. This is the right
primitive for the paper-parity producer/consumer copy pipeline.

### Cluster Support

Gluon exposes `num_ctas` and cluster barriers. The AMD backend sets the kernel
cluster-dims attribute from `num_ctas`.

This gives us a direct path to experiment with wave-cluster/TDM multicast from
Gluon directly.

## What Gluon Does Not Give Us Yet

### No Ready-Made rocSHMEM Gluon Module

The existing external-call pattern for rocSHMEM is straightforward: emit
device-library calls through the Triton builder and link the rocSHMEM device
bitcode. FlashMoE now carries a small Gluon version of that bridge.

The AMD backend can link external bitcode through `extern_libs`, and the venv
has the rocSHMEM device bitcode. We still need a Gluon-compatible way to emit
external calls to symbols such as:

- `rocshmem_my_pe`
- `rocshmem_n_pes`
- `rocshmem_ptr`
- `rocshmem_putmem_wg`
- `rocshmem_putmem_nbi_wg`
- `rocshmem_putmem_signal_wg`
- `rocshmem_putmem_signal_nbi_wg`
- `rocshmem_uint64_wait_until`
- `rocshmem_fence`
- `rocshmem_quiet`

The minimal path is to add only the external-call helper and the small subset of
rocSHMEM wrappers FlashMoE needs.

### Host Runtime Bootstrap Still Matters

Gluon can compile the kernel, but rocSHMEM still needs host-side initialization,
symmetric allocation, PE launch, and teardown. The current HIP path already has
a rocSHMEM compatibility layer; the Gluon plan should reuse that runtime shape.

## Proposed Implementation Stages

### Stage 1: Gluon Local Megakernel Skeleton

Create `flashmoe_gluon/` with a single-GPU local kernel:

- inputs match `flashmoe_triton.forward_megakernel`
- use precomputed routing metadata first
- one Gluon dispatch for MoE work
- persistent processor programs claim expert/token tiles from global queues
- no rocSHMEM calls
- no cluster launch yet

This proves that Gluon can express the FlashMoE control structure while keeping
the communication layer local to FlashMoE.

### Stage 2: Replace Scalar Math With Gluon WMMA/TDM

Port the compute body from the gfx1250 Gluon GEMM examples:

- TDM gather routed token rows into LDS
- TDM load expert weight tiles into LDS
- use LDS mbarriers or `async_wait` first, then move to mbarrier phase waits
- compute GEMM0 with WMMA
- apply activation or gated/SwiGLU epilogue
- compute GEMM1 with WMMA
- combine into output with atomics

This is the most important step for paper-parity performance. It should not
depend on rocSHMEM.

### Stage 3: Add In-Kernel Initialization

Remove the current Python-side `expert_counts.zero_()` style pre-clear from the
Gluon path:

- use a small fixed set of initialization programs inside the same dispatch
- publish a global init-done flag
- other programs wait before consuming scratch
- avoid unbounded spin patterns; use monotonic phases and a timeout-free
  protocol that cannot deadlock if any program exits early

This is required for strict host-visible single-dispatch behavior.

### Stage 4: Add Cluster/TDM Multicast Experiments

Use Gluon `num_ctas` and gfx1250 cluster barriers:

- start with legal multidimensional cluster shapes
- keep the local non-cluster path as the reference
- enable TDM multicast for shared expert or token tiles only after correctness
  matches the non-cluster path
- isolate cluster launch shape selection behind a runtime capability check

This stage should be optional until the local TDM/WMMA kernel is correct.

### Stage 5: Add Minimal rocSHMEM Bridge

Add a tiny Gluon rocSHMEM module:

- port `extern_call` mechanics needed by Gluon, or add an equivalent Gluon
  builtin if the existing builder API is reachable
- wrap only FlashMoE-required calls: PE id/count, putmem, putmem-signal, signal
  fetch/wait
- pass `extern_libs={"rocshmem": ".../librocshmem_device_gfx1250.bc"}`
  at compile/launch time
- reuse the venv rocSHMEM host runtime and symmetric allocations

The first validation target is two local PEs mapped to the single visible GPU.

### Stage 6: Paper-Parity Gluon Megakernel

Once Stages 1-5 are proven, merge the pieces:

- in-kernel dispatch
- OS/scheduler-like persistent work claiming
- WMMA/TDM GEMM0
- activation/gated path
- WMMA/TDM GEMM1
- combine
- rocSHMEM put/signal path
- optional cluster multicast path

The result should be a single Gluon kernel dispatch for the MoE work, with no
host-launched GEMM, combine, or communication kernels.

## Risk Assessment

Best-case path:

- Gluon handles the local persistent WMMA/TDM megakernel directly.
- rocSHMEM requires only a small external-call shim and bitcode link.

Main risks:

- Gluon external calls may need compiler work rather than a pure Python shim.
- rocSHMEM wrappers may require symbols not present in the shipped bitcode.
- Persistent scheduler-style spinning must be written carefully to avoid hangs.
- Multi-CTA cluster code should be kept behind a separate validation path until
  local single-CTA correctness is stable.

## Immediate Next Tasks

1. Move the validated decomposed staging compute into a single Gluon kernel
   body instead of calling upstream matmul as separate launches.
2. Add a local persistent work-queue prototype using Gluon atomics.
3. Add in-kernel initialization for routing counts, scratch state, and global
   completion flags.
4. Compile a minimal Gluon kernel that links `librocshmem_device_gfx1250.bc`
   and calls the rocSHMEM extern bridge.
5. After the bridge compiles and the host runtime initializes the HIP module,
   add a two-PE rocSHMEM smoke test.
