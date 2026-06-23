# Validation Plan

## Environment Setup

Use the venv ROCm every time:

```bash
source /jam/venv/bin/activate
export ROCM_ROOT="$(rocm-sdk path --root)"
export LD_LIBRARY_PATH="$ROCM_ROOT/lib:$LD_LIBRARY_PATH"
```

Compile with:

```bash
hipcc -std=c++20 --offload-arch=gfx1250 -O3 -I./csrc/include ...
```

For CMake/JIT:

```bash
export CMAKE_PREFIX_PATH="$(rocm-sdk path --cmake):$CMAKE_PREFIX_PATH"
```

## Known Test Caveat

Local PyTorch `torch.matmul` is currently wrong on this gfx1250 ROCm build, even for 4x4 float matrices. Do not use PyTorch matmul as a correctness oracle until that is fixed. Use host CPU references in C++ tests.

## Test Order

1. `tests/gemm.hip.cpp`
   - Validates direct rocWMMA/WMMA instructions and host/device memory movement.
   - Already passes on gfx1250 with LD_LIBRARY_PATH set.

2. `tests/tile_gemm.hip.cpp`
   - Must be fixed for the current `CollectiveMainloop` template signature and gfx12 WMMA shapes.
   - Validates FlashMoE's tile abstraction rather than direct ad hoc rocWMMA.

3. Named-barrier probe
   - Compile a tiny gfx1250 HIP kernel containing `__shared__ __amdgpu_named_workgroup_barrier_t`.
   - Inspect generated assembly for `.amdhsa_named_barrier_count > 0`.
   - Run a two-wave launch that calls init/join/signal/wait/leave before enabling warp-specialized staging in the megakernel.

4. TDM and mbarrier probes
   - Compile and run a raw TDM load/store test using `hip/amd_detail/amd_gfx1250_TDM.h`, `__builtin_amdgcn_tensor_load_to_lds`, `__builtin_amdgcn_tensor_store_from_lds`, and `__builtin_amdgcn_s_wait_tensorcnt(0)`.
   - Compile and run a manual LDS mbarrier test using `__builtin_amdgcn_ds_atomic_barrier_arrive_rtn_b64`.
   - Compile and run a TDM-plus-mbarrier completion test using the Triton/LLVM lowering model.

5. `tests/correctness_suite.hip.cpp`
   - Update wave-size expectations to use `flashmoe::WARP_SIZE` rather than hardcoded values.
   - Watch for hipBLASLt tests because PyTorch matmul is already suspect and library GEMM may have gfx1250 issues.

6. `tests/integration.hip.cpp`
   - Update constant expectations to accept AMD wave32 for gfx12.
   - Validate scheduler, subscriber, CUDA compatibility, and basic integration pieces.

7. `tests/moe_forward.hip.cpp`
   - Primary single-GPU persistent-kernel test.
   - Validate against host reference or non-matmul CPU code, not PyTorch matmul.

8. Python/JIT smoke test
   - Import `flashmoe_hip`.
   - Initialize a small config with `gpu_arch=1250`.
   - Only compare against a CPU reference until PyTorch HIP matmul is fixed.
   - When writing `handle.mod.get_tIdx(handle.context)` manually, use `[E, roundEC]` layout, not `[E, EC]`.
   - Validate both topK=1 and topK>1. For CUDA parity, topK>1 plural combine should zero `moe_out` with host-side `hipMemsetAsync` before the persistent MoE launch.

9. Multi-GPU tests
   - When only one GPU is visible, distinct-device XGMI/P2P behavior cannot be fully run.
   - Still run two rocSHMEM PEs mapped to the same visible GPU to validate multi-process PE bootstrap, symmetric allocations, put/get, and signal semantics.
   - Compile coverage is still useful for the true multi-GPU path when rocSHMEM is installed.

## Result Recording Policy

Do not commit probe output, benchmark output, runtime device-query dumps,
per-run counts, or max-error logs. Keep those artifacts in local scratch files
outside git and summarize only source-derived behavior in committed docs.

Routing metadata written by tests must use padded `roundEC` stride. The current
FP16 plural path uses CAS-backed scalar half atomics for contended top-k
accumulation.
