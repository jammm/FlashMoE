# gfx1250 Persistent Kernel Plan

Sources: TheRock HIP/TDM headers and tests, plus Triton/LLVM gfx1250 lowering code.

## Relevant gfx1250 Implementation Facts

ROCm/Triton facts relevant to this implementation:

- gfx1250 follows the gfx12 path.
- MFMA/SMFMA are removed for matrix math and replaced by WMMA/SWMMAC.
- WMMA instructions are wave32-only.
- FP16/BF16 to FP32 WMMA shape is 16x16x32.
- F32 WMMA shape is 16x16x4.
- SWMMAC sparse variants exist but are not needed for dense MoE FFNs.
- Async global-to-LDS and LDS-to-global instructions exist, plus TDM instructions.
- Global/LDS transpose loads are wave32-only and may be useful for later optimized B-tile layout.
- WMMA data hazards require care; rocWMMA should encode safe compiler sequences initially. Hand-written inline ISA can come later only if needed.

## Synchronization Primitives

gfx1250 does not provide the Blackwell semaphore path used by the original CUDA implementation, but it does provide three separate primitives that map to different parts of the megakernel:

- LDS mbarriers for async copy/TDM completion.
- Named workgroup barriers for intra-block wave role rendezvous.
- Device-scope atomics for persistent scheduler doorbells and completion state.

Detailed TDM and mbarrier notes are in `plans/004_gfx1250_tdm_mbarrier_notes.md`.

## LDS mbarriers and TDM

Triton's gfx1250 backend confirms that gfx1250 has a usable mbarrier abstraction. The lowering represents it as an LDS-backed async completion object and uses AMD DS barrier atomics.

- `init(count)` initializes the LDS-backed barrier and synchronizes the workgroup.
- `arrive(count)` lowers to the AMD barrier-arrive primitive and returns the previous phase state.
- `wait(phase)` polls the Triton-managed phase state until the barrier advances.
- TDM async load/store/gather/scatter can be lowered with an attached LDS mbarrier so completion is visible to consumer waves.

This is the completion mechanism for the gfx1250 version of the fully-fused megakernel. `s_wait_tensorcnt` remains useful for probes and conservative fallback paths, but the pipeline should use stage-local LDS mbarriers so compute waves wait only on the copy stage they consume.

TDM facts that matter for the implementation:

- TheRock exposes `hip/amd_detail/amd_gfx1250_TDM.h`.
- Triton lowering attaches an LDS mbarrier to TDM operations for async completion.
- TDM completion is counted by `TENSORcnt` and can also signal the attached LDS mbarrier.

## Named Barriers

gfx12 named workgroup barriers are usable for intra-block warp specialization.

Codegen support:

- Clang exposes gfx12 builtins for `__builtin_amdgcn_s_barrier_init`, `join`, `signal_var`, `wait`, `leave`, and state reads.
- The barrier object must be declared as `__shared__ __amdgpu_named_workgroup_barrier_t` so the generated metadata records named-barrier usage.
- With `__amdgpu_named_workgroup_barrier_t`, codegen emits `.amdhsa_named_barrier_count 1` and real `s_barrier_init`, `s_barrier_join`, `s_barrier_signal`, `s_barrier_wait`, and `s_barrier_leave` instructions.

Codegen constraints that affect the megakernel:

- Barrier member counts are in waves, not threads. CUDA `bar.sync A, B` thread counts must be translated to `ceil(B / 32)` member waves on gfx1250.
- A wave can join at most one named barrier at a time. It can signal any barrier, but `s_barrier_wait` waits on the most recently joined named barrier.
- `s_barrier_init` or `s_barrier_signal_var` must set the member count before use.
- Named barriers are workgroup-local only. They can replace CUDA-style producer/consumer synchronization between loader and compute waves inside one processor block; they cannot replace TDM completion mbarriers, cross-block/global semaphores, doorbells, or inter-GPU completion signals. Those paths must remain LDS mbarriers, device-scope atomics, or rocSHMEM mechanisms as appropriate.

## Current HIP Port Structure

- `constants.hpp` scopes this branch to gfx1250.
- `WARP_SIZE` is set through the gfx1250 path, and scheduler/subscriber math uses `flashmoe::WARP_SIZE`.
- Direct tests use gfx12 WMMA shape `16x16x32` for FP16/BF16.
- The persistent processor path uses in-kernel rocWMMA for gfx12 FP16/BF16 plus FP32 accumulation, including gated MLP.
- The processor path stages WMMA A/B subtile loads through TDM into per-wave LDS buffers and waits on TDM completion through per-wave LDS mbarriers.
- Router and processor WMMA/TDM staging share `wmma_tdm.hip.cuh`, so descriptor setup and LDS mbarrier waits have one implementation.
- Plural combine zeroes `moe_out` with host-side `hipMemsetAsync` before the MoE launch, matching the upstream CUDA path's contract.
- FP16 plural combine uses scalar CAS-backed half atomics for contended top-k accumulation.
- The HIP path mirrors the CUDA launch/API boundary: `flashmoe_hip.router.forward()` launches the gate kernel to compute `expertCounts` and `Context::tokenIndices`; `flashmoe_hip.forward()` consumes precomputed routing metadata.
- The HIP gate path uses the shared gfx1250 WMMA/TDM tile helper when the gate tile is WMMA-legal (`bM`, `bN`, and `bK` aligned for 16x16x32 FP16/BF16 WMMA), with the scalar GEMM fallback preserved for boundary shapes.

Important routing invariant:

- `Context::tokenIndices` is laid out as `[E, roundEC]`, where `roundEC = ceil(EC / bM) * bM`, not `[E, EC]`. Any external router/test that writes this buffer must use the padded stride.

## Implementation Stages

1. Architecture and wave-size plumbing
   - Keep the target macros scoped to gfx1250.
   - Keep `WARP_SIZE` fixed to the gfx1250 target wave size in this branch.
   - Fix masks/comments/tests that assume 64.
   - Keep scheduler and subscriber formulas based on `flashmoe::WARP_SIZE`.

2. gfx12 rocWMMA tile selection
   - Add arch-aware WMMA shape selection in `tile.hip.cuh`.
   - For gfx12 FP16/BF16/FP32-accum GEMM: use 16x16x32.
   - Keep the active tile path on gfx1250-supported WMMA shapes.
   - Fix stale `tests/tile_gemm.hip.cpp` template arguments.

3. Replace Processor scalar GEMM with in-kernel WMMA
   - Implemented a wave-tiled device GEMM path inside `processor.hip.cuh` for the gfx12 FP16/BF16 path.
   - Uses 16x16x32 rocWMMA fragments and keeps the task dependency flow: GEMM0 enqueues GEMM1; GEMM1 signals completion; combine writes output.
   - Bias and activation run in the task epilogue.
   - Gated MLP computes the activated gate, computes the value path, multiplies, and stores.
   - gfx1250 loads each wave's WMMA A/B subtile through TDM into private LDS and uses an LDS mbarrier for completion before rocWMMA consumes the tile. Named barriers remain reserved for explicit producer/consumer wave rendezvous.

4. Shared memory sizing
   - Size GEMM workspace for LDS staging plus per-wave epilogue/gate scratch.
   - Account for the runtime-reported dynamic shared memory budget.

5. Tests and validation
   - First compile and run direct HIP correctness tests with host references.
   - Then run tile GEMM tests.
   - Then compile and run single-GPU MoE forward/integration tests.
   - Avoid PyTorch matmul as an oracle until the local PyTorch gfx1250 matmul bug is resolved.
   - Multi-GPU rocSHMEM tests are compile-only or skipped when only one GPU is visible.

6. TDM and mbarrier fused staging
   - Added HIP probes for raw TDM load/store, LDS mbarrier arrive/wait, and TDM completion into an LDS mbarrier.
   - Added per-wave 64-bit LDS mbarriers and TDM A/B staging buffers in the gfx1250 processor workspace.
   - Stage A/B tiles through LDS with TDM.
   - Attach the LDS mbarrier to the TDM descriptor for B-tile completion after the A-tile load has been issued by the same wave.
   - Track the expected phase per wave and wait on phase changes before rocWMMA consumption.
   - Keep `s_wait_tensorcnt` in probes and fallback/debug paths, not as the steady-state synchronization path.

7. CUDA-parity gate plus persistent MoE launch
   - Keep the same boundary as the upstream CUDA path: gate/router launch first, persistent MoE launch second.
   - Remove HIP-only gate-weight, top-k, and route-epoch arguments from `moe::KernelArgs` and from the Python `ForwardArgs` surface.
   - Keep `Context::tokenIndices` as the shared routing handoff between router and MoE, with `[E, roundEC]` stride.
   - Use the same gate epilogue structure as CUDA: tiled gate GEMM, softmax/topK, optional cross-column ring mailboxes, hipCUB block scan, then guarded expert-count compaction.
   - On gfx1250, run the gate GEMM through the shared WMMA/TDM helper for legal tiles and use the scalar fallback for `E=4` or other non-WMMA tile shapes.

## Deferred Optimizations

After the persistent WMMA path is correct:

- Tune the CUDA-parity router path for larger `E` and multi-block gate reductions.
- Use transpose global/LDS loads for B tiles if rocWMMA layout conversion leaves bandwidth on the table.
- Tune tile shapes beyond the paper's 128x64 for gfx1250 runtime behavior.
- Revisit packed FP16/BF16 atomic accumulation for plural combine after isolating the gfx12 correctness issue.
- Consider SWMMAC only for future sparse expert weights; current MoE weights are dense.
