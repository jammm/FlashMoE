# CLAUDE.md — FlashMoE HIP Port Context

This file captures the full context of the CUDA→HIP port for an AI agent continuing this work.

## Project Overview

FlashMoE is a fully fused Distributed Mixture-of-Experts system (NeurIPS'25, [arXiv:2506.04667](https://arxiv.org/abs/2506.04667)). It fuses gate, dispatch, expert FFN (vanilla or gated/SwiGLU MLP), and combine into a **single persistent CUDA megakernel** with an embedded OS for task scheduling.

The HIP port targets **AMD Instinct MI450 (gfx1250)** and **MI450 (gfx1250)** GPUs, translating every CUDA-specific construct to HIP/ROCm equivalents.

**Author of original:** Osayamen Jonathan Aimuyo (Stanford)
**Repository:** https://github.com/osayamenja/FlashMoE
**Branch:** `jam/hip_port`

## Repository Structure

```
FlashMoE/
├── csrc/include/flashmoe/         # Original CUDA headers (DO NOT MODIFY)
│   ├── moe.cuh                    # Main megakernel
│   ├── gate.cuh                   # Gate/router (softmax + topK)
│   ├── dispatch.cuh               # Token scatter to expert-owning GPUs
│   ├── processor.cuh              # Persistent GEMM processor
│   ├── tile.cuh                   # Tile-level GEMM (cuBLASDx)
│   ├── scheduler.cuh              # Warp-level task scheduler
│   ├── subscriber.cuh             # Signal poller / task creator
│   ├── combine.cuh                # Scatter-reduce expert outputs
│   ├── context.cuh                # Runtime context structs
│   ├── bootstrap.cuh              # Initialization + finalization
│   ├── os.cuh                     # OS (scheduler + subscriber launch)
│   ├── infra/                     # Infrastructure headers
│   │   ├── constants.cuh          # WARP_SIZE=32, tile constants
│   │   ├── atomics.cuh            # cuda::atomic_ref wrappers
│   │   ├── activation.cuh         # Activation functions
│   │   ├── bitset.cuh             # 32-wide bitset
│   │   ├── checks.cuh             # Assertions
│   │   ├── heap.cuh               # Device heap
│   │   ├── math.cuh               # Math utils
│   │   ├── packed.cuh             # Packed structs
│   │   ├── signal.cuh             # Signal primitives
│   │   ├── structures.cuh         # Task, TQSignal, MoEArgs
│   │   ├── task.cuh               # Task descriptors
│   │   ├── rvt.cuh                # Register-value table
│   │   ├── vt.cuh                 # Value table
│   │   ├── dq.cuh                 # Dequeue
│   │   └── tq.cuh                 # Task queue
│   └── hip/                       # HIP PORT (this work)
│       ├── constants.hpp
│       ├── cuda_compat.hpp
│       ├── nvshmem_compat.hpp
│       ├── tensor.hpp
│       ├── tuning.hpp
│       ├── moe.hip.cuh
│       ├── gate.hip.cuh
│       ├── dispatch.hip.cuh
│       ├── processor.hip.cuh
│       ├── tile.hip.cuh
│       ├── tile_gemm_dispatch.hip.cuh
│       ├── gemm_optimized.hip.cuh
│       ├── scheduler.hip.cuh
│       ├── subscriber.hip.cuh
│       ├── combine.hip.cuh
│       ├── atomics.hip.cuh
│       ├── context.hip.cuh
│       ├── activation.hip.cuh
│       ├── bitset.hip.cuh
│       ├── checks.hip.cuh
│       ├── heap.hip.cuh
│       ├── math.hip.cuh
│       ├── packed.hip.cuh
│       ├── signal.hip.cuh
│       ├── structures.hip.cuh
│       └── task.hip.cuh
├── flashmoe/                      # Original CUDA Python module
│   ├── __init__.py
│   ├── bindings.py
│   ├── cb.py                      # nvshmem4py communication backend
│   ├── jit.py                     # CUDA JIT (nvcc)
│   ├── router.py
│   ├── reference.py
│   └── CMakeLists.txt
├── flashmoe_hip/                  # HIP Python module (this work)
│   ├── __init__.py
│   ├── bindings.py
│   ├── cb.py                      # torch.distributed / mpi4py backend
│   ├── jit.py                     # HIP JIT (hipcc via CMake+Ninja)
│   ├── router.py
│   ├── reference.py               # Pure PyTorch reference (no cuBLASDx)
│   └── CMakeLists.txt
├── tests/                         # HIP test files (*.hip.cpp)
├── setup_hip.py                   # setuptools build script for HIP extension
├── pyproject.toml                 # Package metadata (original, CUDA)
├── quickstart.py                  # CUDA quickstart example
├── AMD_BUILD_GUIDE.md             # HIP build + run guide
└── CLAUDE.md                      # This file
```

## Key Porting Decisions and Rationale

### 1. cuBLASDx → hipBLASLt (not rocWMMA)

The original uses cuBLASDx for device-side tile GEMM (`cublasdx::gemm()` inside the kernel). AMD has no cuBLASDx equivalent. Two options were evaluated:

- **rocWMMA**: Provides `wmma::mma_sync()` for tile-level GEMM using MFMA instructions. It does not provide the same device-side tile GEMM abstraction as cuBLASDx.
- **hipBLASLt**: Library-call GEMM. Not tile-level (requires launching from host or a separate kernel), which breaks the single-kernel fusion model.

**Decision:** Use hipBLASLt. The `gemm_optimized.hip.cuh` wrapper handles workspace allocation, matmul descriptor creation, algorithm selection, and row-major↔column-major translation. The `tile_gemm_dispatch.hip.cuh` layer routes calls appropriately. This trades some of the fusion benefit for actual GEMM throughput.

### 2. CuTe → Custom `tensor.hpp`

CuTe (from CUTLASS) provides `cute::Int<N>`, `cute::Shape`, `cute::Stride`, `cute::Layout`, `cute::Tensor`, and `cute::make_tensor()`. Rather than pulling in all of CUTLASS (which has NVIDIA-specific code throughout), a lightweight `hip_tensor.hpp` reimplements only the subset FlashMoE uses:

- `Int<N>` compile-time integer
- `Tuple<Ts...>` variadic tuple with `get<I>()` and structured bindings
- `Shape<Ts...>`, `Stride<Ts...>` as tuple aliases
- `Layout<Shape, Stride>` with `operator()` for multi-index → linear offset
- `Tensor<T, Layout>` with pointer + layout and `operator()` / `operator[]`
- `AlignedArray<T, N, Align>` for vectorized loads/stores
- Free functions: `make_tensor()`, `make_layout()`, `size()`, `rank()`, `get<>()`
- `ceil_div`, `min`, `max` as constexpr in `cute` namespace for compatibility

The original code references `cute::` namespace extensively — the `tensor.hpp` puts everything in a `cute` namespace alias so existing CUDA code compiles unmodified.

### 3. NVSHMEM → rocSHMEM

1:1 API mapping via `nvshmem_compat.hpp`:

```
nvshmem_putmem_signal_nbi()  → rocshmem_putmem_signal_nbi()
nvshmem_signal_fetch_add()   → rocshmem_signal_fetch_add()
nvshmem_fence()              → rocshmem_fence()
nvshmem_ptr()                → rocshmem_ptr()
NVSHMEM_SIGNAL_SET           → ROCSHMEM_SIGNAL_SET
```

The Python communication backend (`cb.py`) cannot use rocSHMEM Python bindings (they don't exist yet), so it falls back to `torch.distributed` or `mpi4py` for host-side collective operations (barriers, rank discovery). Device-side SHMEM operations (putmem, signals) go through the compat header in compiled kernels.

### 4. CCCL → Standard C++ + HIP Builtins

`cuda_compat.hpp` provides:

| CUDA | HIP |
|---|---|
| `cuda::atomic_ref<T, Scope>` | `struct atomic_ref<T>` wrapping `__hip_atomic_load`, `__hip_atomic_store`, `__hip_atomic_compare_exchange_strong`, `__hip_atomic_fetch_add` with `__ATOMIC_RELAXED` / `__ATOMIC_ACQUIRE` / `__ATOMIC_RELEASE` |
| `cuda::barrier` | `struct barrier` using atomic counter + `__builtin_amdgcn_s_waitcnt(0)` |
| `cuda::std::bit_cast<T>` | `std::bit_cast<T>` (C++20) |
| `cuda::std::byte` | `std::byte` |
| `cuda::std::is_same_v` | `std::is_same_v` |
| `cuda::std::conditional_t` | `std::conditional_t` |
| `__nv_bfloat16` | `hip_bfloat16` |
| `__nv_bfloat162` | `hip_bfloat162` (or custom packed struct) |
| `tfloat32_t` | `float` (TF32 doesn't exist on AMD) |
| `__grid_constant__` | empty macro (unsupported on HIP) |

### 5. Target Wave Size

MI450/gfx1250 is treated as a wave32 target. This affects:

- `constants.hpp`: `WARP_SIZE = 32`
- `scheduler.hip.cuh`: scheduler lane calculations use the target wave size
- `bitset.hip.cuh`: bitsets are sized from `flashmoe::WARP_SIZE`
- `atomics.hip.cuh`: shuffle masks use the active-mask abstraction
- All `__shfl_sync(mask, val, lane)` → `__shfl(val, lane)` (HIP shuffles operate on full wavefront, mask is implicit)
- All `__syncwarp()` → `__builtin_amdgcn_wave_barrier()`
- `BlockScan` / `WarpScan` use `hipcub::` variants which handle wavefront width automatically

### 6. Topology Detection

NVIDIA uses `cudaDeviceGetP2PAttribute` with `cudaDevP2PAttrPerformanceRank` to detect NVLink vs PCIe. The HIP port uses ROCm topology reporting to distinguish XGMI from PCIe links.

The `Topology` enum maps: `NVLINK_ONLY → XGMI_ONLY`, `MIXED → MIXED`.

### 7. Python Module Architecture

`flashmoe_hip/` mirrors `flashmoe/` exactly:

- **`jit.py`**: The JIT system generates a `.hip.cpp` source from string templates (`bindings.py`), writes it to a per-config build directory, then runs `cmake -G Ninja` + `cmake --build` with hipcc. Uses file-lock based deduplication so concurrent ranks don't rebuild. Source fingerprinting hashes all `.cuh` and `.hpp` headers for cache invalidation.

- **`bindings.py`**: Three `string.Template` objects — `flashmoe_bindings` (main MoE kernel), `gate_bindings` (router), `reference_bindings` (reference stub). Each is a complete `.hip.cpp` file with `$variable` substitution for JIT-time constants (S, H, I, E, EC, Arch, topK, data type, MLP type, activation type, topology). All use `#include <hip/hip_runtime.h>` and the `cuda_compat.hpp` shim.

- **`__init__.py`**: Entry point. `initialize()` either accepts pre-configured distributed params or calls `cb.initialize()` to discover rank/world via torch.distributed/MPI. Calls `detect_topo()` for XGMI detection. Generates the config-specific module name, substitutes the binding template, JIT-compiles, and calls `mod.initialize()`.

- **`reference.py`**: Pure PyTorch reference implementation. `forward_torch()` does gather → GEMM0 (up-projection) → activation → [gating for SwiGLU] → GEMM1 (down-projection) → scatter. Useful for correctness validation against the fused kernel.

### 8. `setup_hip.py`

Standalone setuptools build script (not JIT). It discovers ROCm and rocSHMEM through environment variables or TheRock-provided paths.

## Test Files

| Test | What It Tests | GPU Count |
|---|---|---|
| `correctness_suite.hip.cpp` | 35 unit tests: atomics, bitset, shuffles, reductions, GEMM correctness, gate routing, dispatch, combine | 1 |
| `gated_gemm.hip.cpp` | SwiGLU gated MLP: correctness + hipBLASLt throughput | 1 |
| `gemm.hip.cpp` | Basic GEMM validation (multiple sizes, data types) | 1 |
| `gemm_benchmark.hip.cpp` | GEMM sweeps: M/N/K from 128→8192, reports TFLOPS | 1 |
| `tile_gemm.hip.cpp` | Tile GEMM dispatch: rocWMMA vs hipBLASLt comparison | 1 |
| `tune_gemm.hip.cpp` | GEMM auto-tuning (algorithm selection) | 1 |
| `moe_forward.hip.cpp` | Single-GPU MoE forward pass | 1 |
| `moe_hipblaslt.hip.cpp` | hipBLASLt-specific MoE integration | 1 |
| `moe_production.hip.cpp` | Production workload shapes (Llama4, DeepSeek, Qwen) | 1 |
| `moe_scale_test.hip.cpp` | Scaling: varying E (experts), S (tokens) | 1 |
| `integration.hip.cpp` | End-to-end: gate → dispatch → experts → combine | 1 |
| `moe_multi_gpu.hip.cpp` | Multi-GPU MoE forward via rocSHMEM | 2+ |
| `multi_gpu_hip_ipc.hip.cpp` | Multi-GPU via HIP IPC (no rocSHMEM) | 2+ |
| `multi_gpu_rocshmem.hip.cpp` | rocSHMEM primitives: put, signal, fence | 2+ |
| `play.hip.cpp` | Interactive / ad-hoc testing | 1 |

## Current Status

Do not store probe or benchmark output in this file. Run the relevant tests locally
when validating a change.
