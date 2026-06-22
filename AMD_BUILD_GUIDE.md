# FlashMoE HIP Build Guide

Build and run guide for the FlashMoE HIP port.

## Prerequisites

- gfx1250-capable AMD GPU
- ROCm 7.x (via TheRock wheels or system install)
- Python 3.12 with PyTorch ROCm
- CMake >= 3.27, Ninja
- hipcc (C++20 capable)

## Quick Start

```bash
source /path/to/venv/bin/activate
export ROCM_ROOT=$(rocm-sdk path --root)
export LD_LIBRARY_PATH=$ROCM_ROOT/lib:$LD_LIBRARY_PATH

cd FlashMoE
hipcc -std=c++20 --offload-arch=gfx1250 -O3 \
    -I./csrc/include -I$ROCM_ROOT/include \
    -L$ROCM_ROOT/lib \
    -o tests/correctness_test tests/correctness_suite.hip.cpp \
    -lhipblaslt

./tests/correctness_test
```

## What Was Ported

FlashMoE is a single persistent CUDA megakernel that fuses gate, dispatch, expert FFN, and combine into one launch. The port translates CUDA-specific constructs to HIP/ROCm equivalents while preserving the original kernel architecture.

### Library Replacements

| CUDA Library | What It Does in FlashMoE | HIP Replacement | Notes |
|---|---|---|---|
| cuBLASDx | Tile-level GEMM inside the fused kernel (A×B accumulation per tile) | hipBLASLt | cuBLASDx provides warp-cooperative GEMM tiles. hipBLASLt is a library-call GEMM, not tile-level. |
| CuTe | Tensor layout abstractions (`cute::Tensor`, `cute::Layout`, `cute::Int<N>`) for shared memory tiling and indexing | `hip_tensor.hpp` (custom) | Lightweight header implementing `Int<N>`, `Shape`, `Stride`, `Layout`, `Tensor`, `AlignedArray`. Only the subset actually used by FlashMoE. |
| CUB | `BlockScan`, `WarpScan` for prefix sums in gate routing and scheduler | hipCUB | Drop-in replacement. hipCUB wraps rocPRIM and provides the same `BlockScan`/`WarpScan` API. Wavefront width is handled through `flashmoe::WARP_SIZE`. |
| NVSHMEM | `putmem_signal_nbi`, `signal_fetch`, `fence` for GPU-initiated RDMA between GPUs | rocSHMEM | 1:1 API mapping via `nvshmem_compat.hpp`. `nvshmem_putmem_signal_nbi()` → `rocshmem_putmem_signal_nbi()`, etc. |
| CCCL | `cuda::atomic_ref`, `cuda::barrier`, `cuda::std::bit_cast`, `cuda::std::byte` | HIP native atomics + `std::` | `cuda::atomic_ref` → `__hip_atomic_*` builtins. `cuda::barrier` → atomic counter + `__builtin_amdgcn_s_waitcnt()`. `cuda::std::bit_cast` → `std::bit_cast` (C++20). |
| CUTLASS | `cutlass::AlignedArray`, `cutlass::is_pow2` used in vector type infrastructure | Reimplemented in `tensor.hpp` | Only the few types actually referenced. |

### Architecture-Specific Changes

| CUDA Construct | What It Does | HIP Equivalent | Why It Changed |
|---|---|---|---|
| `__shfl_sync(0xffffffff, ...)` | Warp shuffle with explicit mask | `__shfl(...)` plus active-mask helpers | HIP `__shfl` operates on the full active wavefront. |
| `__syncwarp()` | Barrier within a 32-thread warp | `__builtin_amdgcn_wave_barrier()` | AMD equivalent for wavefront-level sync. |
| `cuda::ptx::ld_L1_no_allocate` | PTX non-temporal load (bypass L1 cache) | `__builtin_nontemporal_load` / inline asm `global_load_dwordx4` | Used in dispatch for streaming token data. AMD uses GCN non-temporal load instructions. |
| `__grid_constant__` | Kernel arg in constant memory | Removed (not supported in HIP) | Regular `const` parameters. |
| `cuda::std::conditional_t` | Compile-time type selection | `std::conditional_t` | Direct C++17 standard library equivalent. |
| Warp constants | Scheduler, subscriber, and processor lane calculations | Target-specific HIP constants | Used for scheduler counts, scan widths, bitsets, and ready-queue indexing. |
| Processor state sizing | Register state size for task scheduling | Target-specific HIP constant | Set via `FLASHMOE_HIP_ARCH` define or gfx target detection. |

### Kernel Components Ported

Each file in `csrc/include/flashmoe/hip/` corresponds to one CUDA header:

| HIP File | CUDA Original | What It Does |
|---|---|---|
| `moe.hip.cuh` | `moe.cuh` | Main kernel entry. Assigns block roles (dispatch, processor, OS). Launches `gate::forward`, `dispatch`, `processor::start`, `os::start`. |
| `gate.hip.cuh` | `gate.cuh` | Fused softmax + top-K routing. Uses `hipcub::BlockScan` for prefix sums. Produces `tokenIndices[E, EC]` and `expertCounts[E]`. |
| `dispatch.hip.cuh` | `dispatch.cuh` | Scatters tokens to expert-owning GPUs. For XGMI peers: direct mapped write. For RDMA: `rocshmem_putmem_signal_nbi`. |
| `processor.hip.cuh` | `processor.cuh` | Persistent GEMM worker. Spin-waits on doorbell (`__hip_atomic_load`), executes `fGET` (vanilla MLP) or `fGET_gated` (SwiGLU), signals completion. Last GEMM0 tile enqueues GEMM1 tasks. |
| `tile.hip.cuh` | `tile.cuh` | Tile-level GEMM. Replaced `cublasdx` descriptor with hipBLASLt calls. `Converter<T,U>` type conversion functions preserved. |
| `scheduler.hip.cuh` | `scheduler.cuh` | Single-wavefront task scheduler. Polls `statusQueue` via `atomicExch`, matches ready processors with tasks via `WarpScan`, writes `TQSignal` to doorbells. |
| `subscriber.hip.cuh` | `subscriber.cuh` | Signal poller. Watches SHMEM signal flags for arriving tokens (initial stage) and completed expert results (final stage). Constructs `Task` objects and enqueues to scheduler. |
| `combine.hip.cuh` | `combine.cuh` | Scatter-reduces expert outputs back to original token positions. For top-K > 1, uses `atomicAdd` for weighted accumulation. |
| `atomics.hip.cuh` | `infra/atomics.cuh` | `atomicTAS` (test-and-set), `guardedAtomicAdd`, `gridBarrier`. Maps `cuda::atomic_ref` scopes to `__ATOMIC_*` orders. |
| `context.hip.cuh` | `context.cuh` | `Context` and `GateContext` structs. Per-block `stateNumbers` array (was single `stateNumber`). |
| `constants.hpp` | `infra/constants.cuh` | Target constants, alignment helpers, processor-state sizing, scheduler constants. |
| `cuda_compat.hpp` | N/A (new) | Maps CUDA types/intrinsics to HIP: `__nv_bfloat16` → `hip_bfloat16`, `tfloat32_t` → `float`, `cuda::atomic_ref` → HIP atomics, `cuda::barrier` → atomic counter. |
| `nvshmem_compat.hpp` | N/A (new) | Maps NVSHMEM API to rocSHMEM: `nvshmem_putmem_signal_nbi` → `rocshmem_putmem_signal_nbi`, signal constants, etc. |
| `tensor.hpp` | N/A (new) | CuTe replacement: `Int<N>`, `Tuple`, `Shape`, `Stride`, `Layout`, `Tensor<T, Layout>`, `AlignedArray<T, N, Align>`. |
| `gemm_optimized.hip.cuh` | N/A (new) | hipBLASLt GEMM wrapper. Handles workspace allocation, matmul descriptor creation, algorithm selection, row-major→column-major translation. |
| `tile_gemm_dispatch.hip.cuh` | N/A (new) | Unified GEMM dispatch layer. Routes tile GEMM calls to hipBLASLt or rocWMMA based on tile dimensions and data type. |
| `tuning.hpp` | N/A (new) | Tile-shape heuristics and relative tuning helpers for gfx1250. |

### Python Module (`flashmoe_hip/`)

The HIP port has its own Python module parallel to the original `flashmoe/`:

| HIP File | CUDA Original | What Changed |
|---|---|---|
| `__init__.py` | `__init__.py` | `nvshmem` → `rocshmem` init. `cuda.core.experimental` → `torch.cuda` for stream/device management. Topology detection via `rocm-smi --showtopo` (XGMI vs PCIe). |
| `bindings.py` | `bindings.py` | All C++ templates emit HIP code: `hip_runtime.h`, `hipStream_t`, `hipMalloc`, `CHECK_HIP` macros. Uses `cuda_compat.hpp` header for type mapping. Gate and reference bindings also ported. |
| `cb.py` | `cb.py` | Communication backend uses `torch.distributed` or `mpi4py` (rocSHMEM Python bindings not yet available). `nvshmem4py` calls removed. |
| `jit.py` | `jit.py` | Build system uses CMake + Ninja + hipcc. `GPU_TARGETS=gfx1250`. Source fingerprinting includes `.hpp` HIP headers. File-lock based build deduplication preserved. |
| `router.py` | `router.py` | Unchanged logic, imports from `flashmoe_hip.jit` / `flashmoe_hip.bindings`. |
| `reference.py` | `reference.py` | Pure PyTorch reference (no cuBLASDx dependency). Implements gather → GEMM0 → activation → gate → GEMM1 → scatter using `torch.matmul` and `torch.nn.functional`. |
| `CMakeLists.txt` | `CMakeLists.txt` | Full HIP build: `project(... LANGUAGES CXX HIP)`, finds hipBLASLt/rocBLAS/rocSHMEM, `--offload-arch=gfx1250`, pybind11 via CPM. |


## Building

### Environment Variables

```bash
export ROCM_ROOT=$(rocm-sdk path --root)
export LD_LIBRARY_PATH=$ROCM_ROOT/lib:$LD_LIBRARY_PATH

# For multi-GPU (rocSHMEM)
export ROCSHMEM_ROOT=<path to rocshmem install>
export LD_LIBRARY_PATH=$ROCSHMEM_ROOT/lib:$LD_LIBRARY_PATH
```

### Compile Flags

```bash
hipcc -std=c++20 --offload-arch=gfx1250 -O3 \
    -I./csrc/include -I$ROCM_ROOT/include \
    -L$ROCM_ROOT/lib \
    ...
```

| Flag | Purpose |
|------|---------|
| `-std=c++20` | Required for `std::bit_cast`, structured bindings, concepts |
| `--offload-arch=gfx1250` | gfx1250 target |
| `-fgpu-rdc` | Required when linking rocSHMEM (relocatable device code) |

### Single-GPU Tests

```bash
# Correctness suite (35 tests)
hipcc -std=c++20 --offload-arch=gfx1250 -O3 \
    -I./csrc/include -I$ROCM_ROOT/include \
    -L$ROCM_ROOT/lib \
    -o tests/correctness_test tests/correctness_suite.hip.cpp \
    -lhipblaslt

# Gated GEMM (SwiGLU)
hipcc -std=c++20 --offload-arch=gfx1250 -O3 \
    -I./csrc/include -I$ROCM_ROOT/include \
    -L$ROCM_ROOT/lib \
    -o tests/gated_gemm_test tests/gated_gemm.hip.cpp \
    -lhipblaslt

# GEMM benchmark sweeps
hipcc -std=c++20 --offload-arch=gfx1250 -O3 \
    -I./csrc/include -I$ROCM_ROOT/include \
    -L$ROCM_ROOT/lib \
    -o tests/gemm_benchmark tests/gemm_benchmark.hip.cpp \
    -lhipblaslt

# Tile GEMM test (rocWMMA vs hipBLASLt comparison)
hipcc -std=c++20 --offload-arch=gfx1250 -O3 \
    -I./csrc/include -I$ROCM_ROOT/include \
    -L$ROCM_ROOT/lib \
    -o tests/tile_gemm_test tests/tile_gemm.hip.cpp \
    -lhipblaslt
```

### Multi-GPU Tests (rocSHMEM)

```bash
hipcc -std=c++20 --offload-arch=gfx1250 -O3 -fgpu-rdc \
    -I./csrc/include -I$ROCM_ROOT/include \
    -I$ROCSHMEM_ROOT/include \
    -L$ROCM_ROOT/lib -L$ROCSHMEM_ROOT/lib \
    -o tests/moe_multi_gpu_test tests/moe_multi_gpu.hip.cpp \
    -lhipblaslt -lrocshmem
```

### Python Extension (JIT)

The Python module JIT-compiles HIP kernels on first use:

```bash
pip install -e .  # installs flashmoe_hip
python -c "import flashmoe_hip"  # triggers JIT build
```

Or use `setup_hip.py` directly:

```bash
python setup_hip.py build_ext --inplace
```

### Getting rocSHMEM

```bash
git clone -b develop https://github.com/ROCm/rocm-systems.git
cd rocm-systems/projects/rocshmem
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ..
make -j$(nproc) && make install
export ROCSHMEM_ROOT=$(pwd)/../install
```

## Running Tests

```bash
# Single GPU
./tests/correctness_test     # 35-test correctness suite
./tests/gated_gemm_test      # Gated MLP (SwiGLU) correctness + benchmark
./tests/gemm_benchmark       # GEMM performance sweeps across matrix sizes
./tests/tile_gemm_test       # Tile GEMM dispatch testing

# Multi-GPU
./tests/moe_multi_gpu_test   # 2-GPU MoE forward pass via rocSHMEM
```

## File Layout

```
csrc/include/flashmoe/hip/
├── constants.hpp              # target constants and arch detection
├── cuda_compat.hpp            # CUDA→HIP type/intrinsic mapping
├── nvshmem_compat.hpp         # NVSHMEM→rocSHMEM API mapping
├── tensor.hpp                 # CuTe replacement (Int<N>, Layout, Tensor)
├── tuning.hpp                 # tile-shape heuristics
├── moe.hip.cuh               # Main kernel orchestration
├── gate.hip.cuh               # Gate softmax + topK (hipCUB BlockScan)
├── dispatch.hip.cuh           # Token dispatch + SHMEM put
├── processor.hip.cuh          # Persistent GEMM processor (fGET, fGET_gated)
├── tile.hip.cuh               # Tile GEMM (hipBLASLt)
├── tile_gemm_dispatch.hip.cuh # Unified GEMM dispatch layer
├── gemm_optimized.hip.cuh     # hipBLASLt GEMM wrapper
├── scheduler.hip.cuh          # Task scheduler (warp-level scan)
├── subscriber.hip.cuh         # Signal polling + task creation
├── combine.hip.cuh            # Scatter-reduce expert outputs
├── atomics.hip.cuh            # Atomic ops (CAS float/double, grid barrier)
├── context.hip.cuh            # Runtime context structs
├── activation.hip.cuh         # SiLU/GELU/ReLU activation kernels
├── bitset.hip.cuh             # Bitset helpers for wavefront masking
├── checks.hip.cuh             # Runtime assertion helpers
├── heap.hip.cuh               # Device heap allocator
├── math.hip.cuh               # Math utilities (ceil_div, log2, etc.)
├── packed.hip.cuh             # Packed token/signal structures
├── signal.hip.cuh             # Signal primitives (doorbell, fence)
├── structures.hip.cuh         # Task, TQSignal, MoEArgs structs
└── task.hip.cuh               # Task descriptor types

csrc/
├── python_bindings.hip.cpp    # Standalone pybind11 bindings (non-JIT)
├── python_bindings_simple.hip.cpp  # Simplified bindings for testing
└── CMakeLists_hip.txt         # Standalone HIP CMake build

flashmoe_hip/
├── __init__.py                # Module entry: initialize/forward/finalize
├── bindings.py                # JIT C++ template strings (MoE, gate, reference)
├── cb.py                      # Communication backend (torch.distributed / MPI)
├── jit.py                     # JIT compiler: CMake+Ninja+hipcc, file-lock build
├── router.py                  # Gate/router initialization and forward
├── reference.py               # Pure PyTorch reference MoE forward
└── CMakeLists.txt             # JIT build CMake (pybind11, hipBLASLt, rocSHMEM)

tests/
├── correctness_suite.hip.cpp  # 35-test suite
├── gated_gemm.hip.cpp         # SwiGLU correctness + perf
├── gemm.hip.cpp               # GEMM validation
├── gemm_benchmark.hip.cpp     # GEMM sweeps (M/N/K sizes)
├── tile_gemm.hip.cpp          # Tile GEMM dispatch test
├── tune_gemm.hip.cpp          # GEMM auto-tuning
├── moe_forward.hip.cpp        # MoE forward pass
├── moe_hipblaslt.hip.cpp      # hipBLASLt-specific MoE test
├── moe_production.hip.cpp     # Production workload shapes
├── moe_scale_test.hip.cpp     # Scaling tests (varying E, S)
├── integration.hip.cpp        # End-to-end integration
├── moe_multi_gpu.hip.cpp      # Multi-GPU MoE via rocSHMEM
├── multi_gpu_hip_ipc.hip.cpp  # Multi-GPU via HIP IPC
├── multi_gpu_rocshmem.hip.cpp # rocSHMEM primitives test
└── play.hip.cpp               # Interactive / ad-hoc test
```

## Troubleshooting

**`cannot find ROCm device library`** — The TheRock `hipcc` wrapper handles this automatically. If using raw clang++, set `--rocm-device-lib-path=$(rocm-sdk path --root)/lib/llvm/amdgcn/bitcode` or `export HIP_DEVICE_LIB_PATH=...`. Do not set `ROCM_PATH` as an environment variable — it overrides hipcc's internal path resolution.

**`hipGetDeviceCount` hangs** — GPU driver may be in a bad state. Try `rocm-smi --gpureset` or reboot.

**FP16 overflow / NaN** — FlashMoE accumulates in FP32 internally. If testing with random weights, use `1/sqrt(fan_in)` scaling or BF16.

**rocSHMEM link errors** — Ensure `-fgpu-rdc` is passed and `$ROCSHMEM_ROOT/lib` is in `LD_LIBRARY_PATH`.

**JIT build fails with `pybind11 not found`** — The JIT uses CPM to fetch pybind11. Ensure network access or pre-populate `~/.cache/cpm/`.

**`hipFuncSetAttribute` fails for shared memory** — The selected tile shape may request more dynamic shared memory than the active runtime limit allows. Reduce tile shapes or input sizes.

**TheRock vs system ROCm** — TheRock (pip-installable ROCm) uses `rocm-sdk path --root` for discovery. System ROCm typically lives at `/opt/rocm`. Don't mix them — pick one and set `ROCM_ROOT` accordingly.
