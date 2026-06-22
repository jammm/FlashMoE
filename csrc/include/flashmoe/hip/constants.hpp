//
// FlashMoE HIP Port - MI450 / gfx1250 constants
//
// This branch is intentionally scoped to MI450 (gfx1250).
//

#ifndef FLASHMOE_HIP_CONSTANTS_HPP
#define FLASHMOE_HIP_CONSTANTS_HPP

#include <cstddef>
#include <hip/hip_runtime.h>

#if defined(FLASHMOE_HIP_ARCH) && FLASHMOE_HIP_ARCH != 1250
#error "jam/hip_port is currently scoped to MI450 / gfx1250 only"
#endif

#ifndef FLASHMOE_HIP_MI400
#define FLASHMOE_HIP_MI400 1
#endif

#ifndef FLASHMOE_HIP_MI450
#define FLASHMOE_HIP_MI450 1
#endif

#ifndef FLASHMOE_HIP_GFX12
#define FLASHMOE_HIP_GFX12 1
#endif

#ifndef FLASHMOE_HIP_INSTINCT
#define FLASHMOE_HIP_INSTINCT 1
#endif

#ifdef __HIP_PLATFORM_AMD__
#define FLASHMOE_AMD_GPU 1
#endif

#ifdef __HIP_PLATFORM_NVIDIA__
#define FLASHMOE_NVIDIA_GPU 1
#endif

namespace flashmoe
{

// MI400/MI450 supports wave32 only.
constexpr int WARP_SIZE = 32;
constexpr int WAVEFRONT_SIZE = 32;
constexpr int WAVE_SIZE = WARP_SIZE;

#if defined(FLASHMOE_HIP_ARCH)
constexpr int HIP_ARCH = FLASHMOE_HIP_ARCH;
#else
constexpr int HIP_ARCH = 1250;
#endif

// Memory alignment and LDS bank geometry.
constexpr int MAX_ACCESS_ALIGNMENT = 16;
constexpr unsigned long SMEM_BANKS_TOTAL_BYTE_WIDTH = 4 * 32;
constexpr int CACHE_LINE_SIZE = 64;
constexpr int LDS_BANK_COUNT = 32;
constexpr int LDS_BANK_WIDTH = 4;

// Hardware topology is queried at runtime and is not embedded here.
constexpr int MAX_LDS_BYTES = 0;
constexpr int SAFE_LDS_BYTES = 0;

// Workgroup limits.
constexpr int MAX_THREADS_PER_WORKGROUP = 1024;
constexpr int MAX_THREADS_PER_BLOCK = MAX_THREADS_PER_WORKGROUP;
constexpr int MAX_WORKGROUP_DIM_X = 1024;
constexpr int MAX_WORKGROUP_DIM_Y = 1024;
constexpr int MAX_WORKGROUP_DIM_Z = 1024;
constexpr int WORKGROUP_SIZE_SMALL = 64;
constexpr int WORKGROUP_SIZE_MEDIUM = 256;
constexpr int WORKGROUP_SIZE_LARGE = 512;
constexpr int WORKGROUP_SIZE_MAX = 1024;

// gfx1250 WMMA shapes used by the fused path.
constexpr int WMMA_M_16 = 16;
constexpr int WMMA_N_16 = 16;
constexpr int WMMA_K_FP16 = 32;
constexpr int WMMA_K_BF16 = 32;
constexpr int WMMA_K_FP32 = 4;

} // namespace flashmoe

namespace flashmoe::scheduler
{
constexpr int SCHEDULER_COUNT = flashmoe::WARP_SIZE;
constexpr int PROCESSOR_STATE_SIZE = 8;
static_assert(PROCESSOR_STATE_SIZE <= 16, "Processor state size exceeds limit");

constexpr int MAX_PROCESSORS = SCHEDULER_COUNT * PROCESSOR_STATE_SIZE;
constexpr int WORK_SET_SIZE = 4;
constexpr int QUEUE_STATE_SIZE = 2;
constexpr int HEAP_STAGES = 4;
constexpr int HEAP_CELLS = 16;
constexpr int MAX_EXPERTS_PER_BLOCK = 8;
constexpr int TASK_QUEUE_DEPTH = 32;
constexpr int PENDING_QUEUE_DEPTH = 16;

constexpr int BATCH_SIZE_MULTIPLIER = 1;
constexpr int TARGET_WORKGROUPS_PER_UNIT = 4;
constexpr int DISPATCH_GRANULARITY = 1;
} // namespace flashmoe::scheduler

#ifdef __HIP_PLATFORM_AMD__
#define FLASHMOE_WARP_SYNC()       __builtin_amdgcn_wave_barrier()
#define FLASHMOE_WARP_BALLOT(pred) __ballot(pred)
#define FLASHMOE_WARP_ANY(pred)    __any(pred)
#define FLASHMOE_WARP_ALL(pred)    __all(pred)
#define FLASHMOE_LANE_ID()         (__lane_id())
#define FLASHMOE_ACTIVE_MASK()     __ballot(1)
#else
#define FLASHMOE_WARP_SYNC()       __syncwarp()
#define FLASHMOE_WARP_BALLOT(pred) __ballot_sync(0xFFFFFFFF, pred)
#define FLASHMOE_WARP_ANY(pred)    __any_sync(0xFFFFFFFF, pred)
#define FLASHMOE_WARP_ALL(pred)    __all_sync(0xFFFFFFFF, pred)
#define FLASHMOE_LANE_ID()         (threadIdx.x % 32)
#define FLASHMOE_ACTIVE_MASK()     __activemask()
#endif

#define FLASHMOE_SHARED __shared__
#define FLASHMOE_LDS    __shared__

#define FLASHMOE_DEVICE __device__
#define FLASHMOE_HOST   __host__
#define FLASHMOE_GLOBAL __global__
#define FLASHMOE_INLINE __forceinline__

#define FLASHMOE_PREFETCH_GLOBAL(ptr, size) __builtin_prefetch((ptr), 0, 3)
#define FLASHMOE_PREFETCH_LDS(ptr, size)    ((void)0)
#define FLASHMOE_FENCE_WORKGROUP()          __threadfence_block()
#define FLASHMOE_FENCE_DEVICE()             __threadfence()
#define FLASHMOE_FENCE_SYSTEM()             __threadfence_system()

namespace flashmoe::tuning
{
constexpr int TILE_M = 128;
constexpr int TILE_N = 128;
constexpr int TILE_K = 32;
constexpr int EXPERTS_PER_WAVE = 4;
constexpr int TOKENS_PER_EXPERT_BATCH = 64;
constexpr int GLOBAL_TO_LDS_STAGES = 2;
constexpr int COMPUTE_STAGES = 2;
constexpr int VECTOR_WIDTH_FP16 = 8;
constexpr int VECTOR_WIDTH_FP32 = 4;
constexpr int VECTOR_WIDTH_BF16 = 8;
constexpr int ASYNC_COPY_BATCH_SIZE = 16;
} // namespace flashmoe::tuning

#endif // FLASHMOE_HIP_CONSTANTS_HPP
