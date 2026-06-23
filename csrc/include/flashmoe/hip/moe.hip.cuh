/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 *
 * HIP Port - Main MoE Kernel Orchestration
 * Target: AMD MI450 (gfx1250)
 *
 * Key changes from CUDA version:
 * - Uses HIP runtime instead of CUDA
 * - Passes device-resident argument blocks to avoid scalarizing large by-value
 *   kernel argument structs
 * - Uses flashmoe::WARP_SIZE from constants.hpp (32 on MI450/gfx1250)
 * - Replaces cuda::std:: with std:: via cuda_compat.hpp mappings
 * - Uses hipMemsetAsync instead of cudaMemsetAsync
 */

#ifndef FLASHMOE_HIP_MOE_CUH
#define FLASHMOE_HIP_MOE_CUH

#include <hip/hip_runtime.h>

// ============================================================================
// HIP Compatibility Headers
// ============================================================================

#include "flashmoe/hip/cuda_compat.hpp"
#include "flashmoe/hip/constants.hpp"
#include "flashmoe/hip/tensor.hpp"
#include "flashmoe/hip/atomics.hip.cuh"

// ============================================================================
// Ported Component Files
// ============================================================================

#include "flashmoe/hip/gate.hip.cuh"
#include "flashmoe/hip/dispatch.hip.cuh"
#include "flashmoe/hip/processor.hip.cuh"
#include "flashmoe/hip/tile.hip.cuh"
#include "flashmoe/hip/scheduler.hip.cuh"
#include "flashmoe/hip/subscriber.hip.cuh"

// ============================================================================
// Infrastructure Includes (shared with CUDA version via cuda_compat.hpp)
// ============================================================================

#include "flashmoe/hip/signal.hip.cuh"
#include "flashmoe/hip/structures.hip.cuh"
#include "flashmoe/hip/bitset.hip.cuh"
#include "flashmoe/hip/packed.hip.cuh"
#include "flashmoe/hip/task.hip.cuh"
#include "flashmoe/hip/context.hip.cuh"
#include "flashmoe/hip/heap.hip.cuh"
#include "flashmoe/hip/math.hip.cuh"

namespace flashmoe::moe
{

constexpr int MAX_DISPATCH_BLOCKS = 128;

// ============================================================================
// Memory Alignment Constant for Shared Memory
// ============================================================================

// Maximum alignment for shared memory (LDS on AMD)
// Using MAX_ACCESS_ALIGNMENT from constants.hpp
constexpr int MAX_ALIGNMENT = flashmoe::MAX_ACCESS_ALIGNMENT;

// ============================================================================
// Activation Type Mapping for HIP
// ============================================================================

// Bring canonical types from flashmoe namespace into flashmoe::moe
using flashmoe::Activation;
using flashmoe::ActivationType;
using flashmoe::Identity;
using flashmoe::ReLU;
using flashmoe::GELU;
using flashmoe::SiLU;
using flashmoe::Topology;
using flashmoe::MLPMatmulType;
using flashmoe::Task;
using flashmoe::TQSignal;
using flashmoe::PEL;
using flashmoe::PLI;
using flashmoe::ELI;
using flashmoe::LXI;
using flashmoe::TPS;
using flashmoe::BitSet;

// ============================================================================
// CombineMode Enum (replicated from combine.cuh for HIP standalone)
// ============================================================================

#ifndef FLASHMOE_HIP_COMBINEMODE_DEFINED
#define FLASHMOE_HIP_COMBINEMODE_DEFINED
enum class CombineMode {
    single,
    plural
};
#endif

// Forward declaration
__host__ __device__ void checkAlignment(const void* const& p, const bool supports32 = false);

// ============================================================================
// MoE Configuration Template
// ============================================================================

/**
 * @brief MoE configuration template containing compile-time parameters
 *
 * Template Parameters:
 *   Element    - Data type (fp32, fp16, bf16, fp64)
 *   arch       - GPU architecture, currently 1250 for MI450/gfx1250
 *   _threads   - Thread block size (see tile::suggest_thread_count)
 *   cm         - CombineMode: plural if k > 1; single otherwise
 *   mt         - MLPMatmulType: gated or vanilla MLP pattern
 *   GEMM0Tile  - Tile shape for first GEMM: Shape<bM, bN, bK, pipeStages>
 *   GEMM1Tile  - Tile shape for second GEMM: Shape<bM, bN, bK, pipeStages>
 *
 * Note for AMD port:
 *   - The arch value is used for rocWMMA configuration
 *   - MI450/gfx1250 uses 1250
 *   - Thread counts should be multiples of 32 (wavefront size)
 */
template <
    typename Element,
    int arch,
    int _threads,
    CombineMode cm,
    MLPMatmulType mt,
    typename GEMM0Tile,
    typename GEMM1Tile,
    int processor_state_size = scheduler::PROCESSOR_STATE_SIZE
>
struct MoEConfig {
    // Validate tile shapes are 4-tuples (bM, bN, bK, pipeStages)
    static_assert(hip_tensor::Tuple<int,int,int,int>::rank == 4);
    using G0TS = GEMM0Tile;
    using G1TS = GEMM1Tile;

    // Validate that bM matches between GEMM0 and GEMM1
    // Note: Using hip_tensor::get instead of cute::get
    static_assert(hip_tensor::get<0>(GEMM0Tile{}) == hip_tensor::get<0>(GEMM1Tile{}));

    // Type definitions using hip_tensor wrappers
    using Arch = hip_tensor::Int<arch>;
    using Threads = hip_tensor::Int<_threads>;
    using CM = hip_tensor::Int<static_cast<int>(cm)>;
    using MT = hip_tensor::Int<static_cast<int>(mt)>;
    using DType = Element;
    using PSS = hip_tensor::Int<processor_state_size>;

    // Accumulator type: double for double input, float otherwise
    // Using std::conditional_t via cuda_compat.hpp mappings
    using AccumType = cuda::std::conditional_t<
        cuda::std::is_same_v<Element, double>,
        double,
        float
    >;
    static constexpr int ArchWaveSize = (arch >= 1200 && arch < 1300) ? 32 : flashmoe::WARP_SIZE;

    // Thread count should be multiple of wavefront size for optimal performance
    static_assert(_threads % ArchWaveSize == 0,
        "Thread count should be a whole number of waves for this architecture");
};

// ============================================================================
// Kernel Arguments Structure
// ============================================================================

/**
 * @brief Arguments passed to the MoE forward kernel
 *
 * Contains pointers to all input/output tensors and problem dimensions.
 * All pointers must be properly aligned for optimal memory access.
 *
 * For gated MLP (MLPMatmulType::gated):
 * - expertUpVWeights and biasUpV are used for the value projection
 * - swishAlpha and swishBeta control the activation scaling
 */
struct KernelArgs {
    __host__ __forceinline__
    KernelArgs(const cuda::std::byte* const& tokens,
        const cuda::std::byte* const& expert_up_weights, const cuda::std::byte* const& expert_up_v_weights,
        const cuda::std::byte* const& bias_up, const cuda::std::byte* const& bias_up_v,
        const cuda::std::byte* const& expert_down_weights,
        const cuda::std::byte* const& bias_down,
        const int* const& expert_counts, cuda::std::byte* const& moe_out,
        const size_t& s, const size_t& h, const size_t& i, const size_t& e, const size_t& ec,
        const int& arch, const MLPMatmulType& m, const int& bM,
        const float& swishAlpha, const float& swishBeta, const bool check = true)
        : tokens(tokens),
          expertUpWeights(expert_up_weights),
          expertUpVWeights(expert_up_v_weights),
          biasUp(bias_up),
          biasUpV(bias_up_v),
          expertDownWeights(expert_down_weights),
          biasDown(bias_down),
          expertCounts(expert_counts),
          moeOut(moe_out),
          S(static_cast<uint>(s)),
          H(static_cast<uint>(h)),
          I(static_cast<uint>(i)),
          E(static_cast<uint>(e)),
          EC(static_cast<uint>(ec)),
          flagColStride(cute::ceil_div(EC, bM) * E),
          swishAlpha(swishAlpha),
          swishBeta(swishBeta) {
        if (check) {
            const auto supports32 = arch >= 900;
            checkAlignment(tokens, supports32);
            checkAlignment(expert_up_weights);
            if (m == MLPMatmulType::gated) {
                checkAlignment(expert_up_v_weights);
                checkAlignment(bias_up_v);
            }
            checkAlignment(expert_down_weights);
            checkAlignment(bias_up);
            checkAlignment(bias_down);
            checkAlignment(expert_counts);
            checkAlignment(moe_out);
        }
    }

    const cuda::std::byte* const tokens;             // [S, H]
    const cuda::std::byte* const expertUpWeights;    // [num_local_experts, H, I]
    const cuda::std::byte* const expertUpVWeights;   // [num_local_experts, H, I] for gated MLP
    const cuda::std::byte* const biasUp;             // [num_local_experts, I]
    const cuda::std::byte* const biasUpV;            // [num_local_experts, I] for gated MLP
    const cuda::std::byte* const expertDownWeights;  // [num_local_experts, I, H]
    const cuda::std::byte* const biasDown;           // [num_local_experts, H]
    const int* const expertCounts;                    // [E]
    cuda::std::byte* const moeOut;                   // [S, H]
    const uint S;   // sequence length
    const uint H;   // token hidden dimension
    const uint I;   // FFN intermediate size
    const uint E;   // total number of experts
    const uint EC;  // expert capacity
    const uint flagColStride; // ceil(EC / bM) * E
    const float swishAlpha = 1.f;
    const float swishBeta = 1.f;
};

// ============================================================================
// Context Structure for HIP
// ============================================================================

// Use canonical Context and Heap from flashmoe namespace (context.hip.cuh, heap.hip.cuh)
using flashmoe::Context;
using flashmoe::GateContext;
using flashmoe::Heap;

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Check pointer alignment for optimal memory access
 */
__host__ __device__ __forceinline__
void checkAlignment(const void* const& p, const bool supports32) {
    const auto alignment = supports32 ? 32 : 16;
    auto addr = reinterpret_cast<uintptr_t>(p);
    if (p == nullptr || (addr % alignment) != 0) {
        printf("Pointer %p is not %d-byte aligned\n", p, alignment);
        // Note: std::terminate() is not available on device in HIP
        // Use assert or trap instead
#ifdef __HIP_DEVICE_COMPILE__
        __builtin_trap();
#else
        std::terminate();
#endif
    }
}

__host__ __forceinline__
bool checkHipHostCall(const hipError_t err, const char* const what) {
    if (err == hipSuccess) {
        return true;
    }
    fprintf(stderr, "[FlashMoE] %s FAILED: %s (%s)\n",
            what, hipGetErrorName(err), hipGetErrorString(err));
    return false;
}

/**
 * @brief Calculate dispatch super-block size based on expert count
 */
__host__ __device__ __forceinline__
constexpr auto dispatchSuperBlockSize(const uint& E) {
    auto maxE = (E > 4) ? E : 4;
    return (128 + maxE - 1) / maxE;  // ceil_div(128, max(E, 4))
}

// ============================================================================
// Kernel Configuration Helpers
// ============================================================================

namespace os {
template<int threads, int bM>
__host__ __forceinline__
constexpr auto getSharedSize(const uint& world, const uint& nLx,
    const uint& E, const uint& EC, const uint& tilesN1);
} // forward decl

template<typename Config>
__host__ __forceinline__
constexpr uint kernelSMEM(const uint& E, const uint& EC, const int& world,
    const uint& numLocalExperts, const uint& tilesN1) {
    constexpr int bM0 = hip_tensor::get<0>(typename Config::G0TS{});
    constexpr int bN0 = hip_tensor::get<1>(typename Config::G0TS{});
    constexpr int bK0 = hip_tensor::get<2>(typename Config::G0TS{});
    constexpr int pSK0 = hip_tensor::get<3>(typename Config::G0TS{});

    constexpr int bM1 = hip_tensor::get<0>(typename Config::G1TS{});
    constexpr int bN1 = hip_tensor::get<1>(typename Config::G1TS{});
    constexpr int bK1 = hip_tensor::get<2>(typename Config::G1TS{});
    constexpr int pSK1 = hip_tensor::get<3>(typename Config::G1TS{});
    constexpr int threads = Config::Threads::value;
    constexpr auto mt = Config::MT::value;
    static_assert(bM0 == bM1);

    using Element = typename Config::DType;
    using AccumType = typename Config::AccumType;

    constexpr int bM = bM0;
    auto roundUp = [](auto val, auto align) { return ((val + align - 1) / align) * align; };
    auto maxVal = [](auto a, auto b) { return (a > b) ? a : b; };
    using GEMM0Mainloop = tile::CollectiveMainloop<
        bM0, bN0, bK0, Config::Arch::value, Element, AccumType, threads, pSK0>;
    using GEMM1Mainloop = tile::CollectiveMainloop<
        bM1, bN1, bK1, Config::Arch::value, Element, AccumType, threads, pSK1>;
    constexpr bool useWmmaProcessor =
        (Config::Arch::value >= 1200 && Config::Arch::value < 1300) &&
        (sizeof(Element) == 2) &&
        cuda::std::is_same_v<AccumType, float>;

    constexpr auto GEMM0ScalarSz =
        maxVal(sizeof(Element) * bK0 * pSK0 * (bM + bN0),
               sizeof(AccumType) * bM * bN0) +
        (mt == static_cast<int>(MLPMatmulType::gated) ? sizeof(AccumType) * bM * bN0 : 0);
    constexpr auto GEMM1ScalarSz =
        maxVal(sizeof(Element) * bK1 * pSK1 * (bM + bN1),
               sizeof(AccumType) * bM * bN1);
    constexpr auto GEMM0WmmaSz = useWmmaProcessor
        ? wmma_tdm::wmmaTdmWorkspaceBytes<GEMM0Mainloop, Element, AccumType>()
        : 0;
    constexpr auto GEMM1WmmaSz = useWmmaProcessor
        ? wmma_tdm::wmmaTdmWorkspaceBytes<GEMM1Mainloop, Element, AccumType>()
        : 0;
    constexpr auto GEMM0Sz = roundUp(maxVal(GEMM0ScalarSz, GEMM0WmmaSz), MAX_ALIGNMENT);
    constexpr auto GEMM1Sz = roundUp(maxVal(GEMM1ScalarSz, GEMM1WmmaSz), MAX_ALIGNMENT);
    const auto dispatchSz = E * (sizeof(PEL) + sizeof(int));
    const auto OSSz = os::getSharedSize<threads, bM>(world, numLocalExperts, E, EC, tilesN1);
    const auto taskSz = sizeof(Task) * tilesN1;
    constexpr auto combineSz = roundUp(sizeof(Element) * bM * bN1, MAX_ALIGNMENT);
    auto max6 = [](auto a, auto b, auto c, auto d, auto e, auto f) {
        auto m = a;
        if (b > m) m = b; if (c > m) m = c; if (d > m) m = d; if (e > m) m = e; if (f > m) m = f;
        return m;
    };
    return max6(GEMM0Sz, GEMM1Sz, dispatchSz, OSSz, taskSz, combineSz);
}

template<int bM, int bN0, int bN1>
__host__ __forceinline__
constexpr auto kernelBlocks(const uint& S, const uint& H, const uint& I, const uint& E,
    const uint& k, const int& blocksPerSM, const int& numSMs) {
    auto ceilDiv = [](auto a, auto b) { return (a + b - 1) / b; };
    const auto processorBlocks = (ceilDiv(S * k, static_cast<uint>(bM)) * ((I / bN0) + (H / bN1))) +
        (ceilDiv(S * k, static_cast<uint>(bM)) * (H / bN1));
    const auto dispatchBlocks = cute::min(dispatchSuperBlockSize(E) * E, static_cast<uint>(MAX_DISPATCH_BLOCKS));
    auto maxVal = [](auto a, auto b) { return (a > b) ? a : b; };
    auto minVal = [](auto a, auto b) { return (a < b) ? a : b; };
    return maxVal(minVal(maxVal(processorBlocks, dispatchBlocks) + 1,
        static_cast<uint>(blocksPerSM * numSMs)), 2U);
}

// ============================================================================
// Task Queue Length Calculations
// ============================================================================

template<int subscriberWarpSize>
__host__ __device__ __forceinline__
constexpr auto subscriberTQLength(const int& world, const uint& numLocalExperts, const uint& ecTilesM,
    const uint& E, const uint& tilesN0, const uint& tilesN1, const uint& subscriberCount) {
    auto ceilDiv = [](auto a, auto b) { return (a + b - 1) / b; };
    const auto dispatchTaskQL = ceilDiv(world * numLocalExperts, subscriberCount / subscriberWarpSize) *
        (ceilDiv(ecTilesM * tilesN0, subscriberWarpSize) + ceilDiv(tilesN0, subscriberWarpSize));
    const auto combineTaskQL = (ceilDiv(ecTilesM * E, subscriberCount) * tilesN1) +
        ceilDiv(ecTilesM * E * tilesN1, subscriberCount);
    return (dispatchTaskQL + combineTaskQL) * subscriberCount;
}

template<int subscriberCount, int subscriberWarpSize>
__device__ __forceinline__
constexpr auto subscriberTQLength(const int& world, const int& numLocalExperts, const uint& ecTilesM,
    const uint& E, const uint& tilesN0, const uint& tilesN1) {
    static_assert(subscriberCount % subscriberWarpSize == 0);
    return subscriberTQLength<subscriberWarpSize>(world, static_cast<uint>(numLocalExperts), ecTilesM, E, tilesN0, tilesN1, static_cast<uint>(subscriberCount));
}

__host__ __device__ __forceinline__
auto secondaryTQLength(const int& world, const int& numLocalExperts, const uint& ecTilesM, const uint& tilesN1) {
    return world * numLocalExperts * ecTilesM * tilesN1;
}

// ============================================================================
// OS (Operating System / Scheduler) Functions for HIP
// ============================================================================

namespace os {

/**
 * @brief Calculate shared memory size for OS block
 */
template<int threads, int bM>
__host__ __forceinline__
constexpr auto getSharedSize(const uint& world, const uint& nLx,
    const uint& E, const uint& EC, const uint& tilesN1) {
    static_assert(threads >= flashmoe::WARP_SIZE * 2);
    constexpr auto subscriberCount = threads - flashmoe::WARP_SIZE;
    constexpr auto sNW = subscriberCount / flashmoe::WARP_SIZE;
    static_assert(subscriberCount % flashmoe::WARP_SIZE == 0 && sNW >= 1);

    auto ceilDiv = [](auto a, auto b) { return (a + b - 1) / b; };
    const auto ecTilesM = ceilDiv(EC, bM);
    const auto ecSignalCount = ecTilesM * tilesN1;
    const auto ssfC = E * ecSignalCount;
    const auto gtQCl = world * nLx * ecTilesM;

    // Helper for rounded cache line sizes
    auto rTCL = [](auto bytes) { return ((bytes + 127) / 128) * 128; };
    auto nSI = [](auto count, auto divider) { return (count + divider - 1) / divider; };

    size_t bytes = 0;
    bytes += rTCL(sizeof(PLI) * world);
    bytes += rTCL(sizeof(ELI) * E);
    bytes += rTCL(sizeof(LXI) * nLx);
    const auto fSbSL = rTCL(sizeof(BitSet) * nSI(nLx * world, sNW));
    const auto bScL = nSI(ssfC, subscriberCount);
    const auto bSSI = fSbSL + bScL;
    bytes += rTCL(sizeof(BitSet) * bSSI);
    bytes += rTCL(sizeof(BitSet) * bScL);
    bytes += rTCL(sizeof(BitSet) * nSI(gtQCl, flashmoe::WARP_SIZE));
    bytes += sizeof(uint) * subscriberCount;
    bytes += rTCL(sizeof(uint) * scheduler::MAX_PROCESSORS);
    bytes += rTCL(sizeof(uint) * scheduler::MAX_PROCESSORS);
    bytes += rTCL(sizeof(uint) * world);
    bytes += sizeof(uint) * (subscriberCount / flashmoe::WARP_SIZE);
    bytes += sizeof(uint);
    bytes += sizeof(int) * E;
    return bytes;
}

/**
 * @brief OS block main entry point
 *
 * Ported from CUDA os.cuh. Coordinates the scheduler (first wavefront)
 * and subscribers (remaining threads) within the OS block.
 *
 * The OS block is the last block in the grid (blockIdx.x == gridDim.x - 1).
 * It manages the task scheduling pipeline:
 * 1. Initializes shared memory with routing metadata
 * 2. Computes taskBound (total tasks to schedule)
 * 3. Splits: wavefront 0 = scheduler, rest = subscribers
 * 4. Scheduler assigns tasks to processors via doorbells
 * 5. Subscribers poll for completed dispatch/combine signals and enqueue tasks
 */
template <
    Topology topo,
    int subscriberCount,
    int threads,
    int bM,
    int pss,
    typename ElementC
>
__device__ __forceinline__
void start(cuda::std::byte* __restrict__ const& workspace, const int* __restrict__ expertCounts,
           const Heap& symHeap,
           const Context& ctx, const int& EC,
           const int& tilesN0, const int& tilesN1,
           const int& dispatchBlocks,
           const int& E, const int& I,
           const uint& processors) {
    const auto stateNumber = ctx.stateNumbers[blockIdx.x];
    const auto ecTilesM = cute::ceil_div(EC, bM);
    const auto ecSignalCount = ecTilesM * tilesN1;
    const auto ssfC = E * ecSignalCount;
    const auto world = ctx.world;
    const auto nLx = ctx.nLx;
    constexpr auto sNW = subscriberCount / flashmoe::WARP_SIZE;
    static_assert(subscriberCount % flashmoe::WARP_SIZE == 0 && sNW >= 1);

    const auto gtQCl = ctx.world * ctx.nLx * ecTilesM;
    const auto sBz = nSI<flashmoe::WARP_SIZE>(gtQCl);

    auto roundUpTo = [](auto x, auto m) -> decltype(x) { return ((x + m - 1) / m) * m; };

    size_t offset = 0;
    auto* __restrict__ pL = reinterpret_cast<PLI*>(workspace);
    offset += rTCL<PLI>(world);
    auto* __restrict__ eL = reinterpret_cast<ELI*>(workspace + offset);
    offset += rTCL<ELI>(E);
    auto* __restrict__ lX = reinterpret_cast<LXI*>(workspace + offset);
    offset += rTCL<LXI>(nLx);
    auto* __restrict__ subVisitedSet = reinterpret_cast<BitSet*>(workspace + offset);
    const auto xfSbSL = nSI<sNW>(nLx * world);
    const auto fSbSL = roundUpTo(xfSbSL,
        static_cast<uint>(SMEM_BANKS_TOTAL_BYTE_WIDTH / sizeof(BitSet)));
    const auto bScL = nSI<subscriberCount>(ssfC);
    const auto bSSI = fSbSL + bScL;
    offset += rTCL<BitSet>(bSSI);
    auto* __restrict__ senseBitset = reinterpret_cast<BitSet*>(workspace + offset);
    offset += rTCL<BitSet>(bScL);

    auto* __restrict__ schedulerBitSet = reinterpret_cast<BitSet*>(workspace + offset);
    offset += rTCL<BitSet>(sBz);
    auto* __restrict__ tQHeads = reinterpret_cast<uint*>(workspace + offset);
    offset += sizeof(uint) * subscriberCount;
    auto* __restrict__ rQ = reinterpret_cast<uint*>(workspace + offset);
    offset += rTCL<uint>(processors);
    auto* __restrict__ interruptScratch = reinterpret_cast<uint*>(workspace + offset);
    offset += rTCL<uint>(processors);
    auto* __restrict__ status = reinterpret_cast<uint*>(workspace + offset);
    offset += rTCL<uint>(world);
    auto* __restrict__ interrupt = reinterpret_cast<uint*>(workspace + offset);
    offset += sizeof(uint) * (subscriberCount / flashmoe::WARP_SIZE);
    auto* __restrict__ taskBound = reinterpret_cast<uint*>(workspace + offset);
    offset += sizeof(uint);
    auto* __restrict__ scratch = reinterpret_cast<uint*>(workspace + offset);

    const auto* __restrict__ geL = ctx.eli;
    const auto* __restrict__ gpL = ctx.pli;
    const auto* __restrict__ gLx = ctx.lxi;
    for (uint i = threadIdx.x; i < xfSbSL; i += threads) {
        subVisitedSet[i] = BitSet{0U};
    }
    for (uint i = threadIdx.x; i < static_cast<uint>(world); i += threads) {
        pL[i] = gpL[i];
    }
    for (uint i = threadIdx.x; i < static_cast<uint>(E); i += threads) {
        eL[i] = geL[i];
    }
    for (uint i = threadIdx.x; i < static_cast<uint>(nLx); i += threads) {
        lX[i] = gLx[i];
    }
    for (uint i = threadIdx.x; i < bScL; i += threads) {
        senseBitset[i] = ctx.consumerCombineBitMap[i];
    }
    auto* __restrict__ eCs = reinterpret_cast<int*>(scratch);
    if (!threadIdx.x) {
        *taskBound = ctx.nLx * ctx.world * ecTilesM * (tilesN0 + tilesN1);
    }
    for (uint i = threadIdx.x; i < static_cast<uint>(E); i += threads) {
        eCs[i] = expertCounts[i];
    }
    __syncthreads();
    for (uint i = threadIdx.x; i < static_cast<uint>(E); i += threads) {
        const auto eCount = cute::min(eCs[i], EC);
        const auto eCt = cute::ceil_div(eCount, bM);
        atomicAdd(taskBound, static_cast<uint>(eCt * tilesN1));
    }
    if (threadIdx.x < subscriberCount) {
        const auto nRows = ecTilesM * E;
        auto* __restrict__ vs = subVisitedSet + fSbSL;
        for (uint i = threadIdx.x; i < bScL; i += subscriberCount) {
            const auto slot = i / subscriberCount;
            const uint prefix = slot * subscriberCount * sizeof(BitSet) * 8;
            auto bitset = vs[i];
            for (int j = 0; j < static_cast<int>(sizeof(BitSet) * 8); ++j) {
                const auto flagIdx = (prefix + threadIdx.x) + (j * subscriberCount);
                if (flagIdx >= static_cast<uint>(ecSignalCount * E)) {
                    break;
                }
                const auto expertIdx = (flagIdx % nRows) / ecTilesM;
                const bool isRemote = eL[expertIdx].isRemote;
                const auto tileRowIdx = (flagIdx % nRows) % ecTilesM;
                const auto tileColIdx = flagIdx / nRows;
                const auto expertCount = cute::min(eCs[expertIdx], EC);
                const auto actualTiles = cute::ceil_div(expertCount, bM);
                if ((isRemote && tileColIdx > 0) || tileRowIdx >= static_cast<uint>(actualTiles)) {
                    bitset.set(j);
                }
                else {
                    bitset.clear(j);
                }
            }
            vs[i] = bitset;
        }
    }
    __syncthreads();
    const auto fL = processors - dispatchBlocks;
    const auto sL = fL / threads;
    const auto rL = fL % threads;
    for (uint i = 0; i < sL; ++i) {
        const auto idx = i * threads + threadIdx.x;
        rQ[idx] = dispatchBlocks + idx;
    }
    if (fL % threads != 0 && threadIdx.x < rL) {
        const auto idx = sL * threads + threadIdx.x;
        rQ[idx] = dispatchBlocks + idx;
    }
    const auto psL = static_cast<uint>(dispatchBlocks) / threads;
    const auto prL = static_cast<uint>(dispatchBlocks) % threads;
    for (uint i = 0; i < psL; ++i) {
        const auto idx = i * threads + threadIdx.x;
        rQ[fL + idx] = idx;
    }
    if (prL % threads != 0 && threadIdx.x < prL) {
        const auto idx = psL * threads + threadIdx.x;
        rQ[fL + idx] = idx;
    }
    for (uint i = threadIdx.x; i < processors; i += threads) {
        interruptScratch[i] = 1U;
    }
    for (uint i = threadIdx.x; i < static_cast<uint>(subscriberCount); i += threads) {
        tQHeads[i] = 0U;
    }
    for (uint i = threadIdx.x; i < (subscriberCount / flashmoe::WARP_SIZE); i += threads) {
        interrupt[i] = 0U;
    }
    for (uint i = threadIdx.x; i < static_cast<uint>(world); i += threads) {
        status[i] = 0U;
    }
    for (uint i = threadIdx.x; i < sBz; i += threads) {
        schedulerBitSet[i] = BitSet{0U};
    }
    __syncthreads();
    if (threadIdx.x / flashmoe::WARP_SIZE == 0) {
        const auto sO = subscriberTQLength<subscriberCount, flashmoe::WARP_SIZE>(ctx.world,
            ctx.nLx, ecTilesM, E, tilesN0, tilesN1);
        auto* const __restrict__ gtQHeads = ctx.gTqHeads;
        auto* const __restrict__ sQ = ctx.statusQueue;
        auto* const pDB = ctx.tqs;
        scheduler::start<subscriberCount, pss>(
            const_cast<uint* __restrict__ const&>(interruptScratch),
            const_cast<BitSet* __restrict__ const&>(schedulerBitSet),
            processors, ctx.processors_v, tilesN1,
            sO, gtQCl,
            const_cast<uint* __restrict__ const&>(interrupt),
            const_cast<uint* __restrict__ const&>(tQHeads),
            gtQHeads, taskBound, rQ, sQ, pDB);
    }
    else {
        const auto tIdx = threadIdx.x - flashmoe::WARP_SIZE;
        subscriber::Args args{
            ctx.signals, ctx.tQ, ctx.GEMM0Staging, senseBitset, subVisitedSet, interrupt, tQHeads,
            pL, lX, eL, status, taskBound, ctx.world, ctx.nLx, static_cast<uint>(ctx.nLx * ctx.world),
            ctx.epRank, static_cast<uint>(ecTilesM * bM), E, I, static_cast<uint>(tIdx), tilesN0, tilesN1, ecTilesM, stateNumber
        };
        subscriber::start<topo, subscriberCount, bM, ElementC>(symHeap, args, fSbSL);
    }
    __syncthreads();
    for (uint i = threadIdx.x; i < bScL; i += threads) {
        ctx.consumerCombineBitMap[i] = senseBitset[i];
    }
    if (!threadIdx.x) {
        ctx.stateNumbers[blockIdx.x] = sbs::next(stateNumber);
    }
}

} // namespace os

// ============================================================================
// Forward Kernel
// ============================================================================

/**
 * @brief Main MoE forward kernel for HIP
 *
 * This kernel orchestrates the entire MoE forward pass:
 * 1. Token dispatch to experts (local and remote)
 * 2. Expert computation (GEMM0 + activation + GEMM1)
 * 3. Token combining (weighted sum for top-k > 1)
 *
 * Template Parameters:
 *   Config - MoEConfig with compile-time parameters
 *   a      - Activation function type
 *   topo   - Network topology (NVLINK_ONLY or MIXED)
 *
 * Note for HIP port:
 *   - KernelArgs and Context are staged in device memory before launch so the
 *     kernel does not receive large by-value kernarg structs.
 *   - __launch_bounds__ syntax is the same in HIP.
 *   - WARP_SIZE from constants.hpp is 32 on MI450/gfx1250.
 */
template <
    typename Config,
    Activation a,
    Topology topo
>
__launch_bounds__(Config::Threads::value, 1)
__global__ void forward(const KernelArgs* __restrict__ kArgsPtr,
                        const Context* __restrict__ ctxPtr) {
    const KernelArgs& kArgs = *kArgsPtr;
    const Context& ctx = *ctxPtr;
    using DataType = typename Config::DType;

    extern __shared__ __align__(MAX_ALIGNMENT) cuda::std::byte flashWorkspace[];

    const auto* __restrict__ tokens = reinterpret_cast<const DataType*>(kArgs.tokens);

    constexpr int bM0 = hip_tensor::get<0>(typename Config::G0TS{});
    constexpr int bN0 = hip_tensor::get<1>(typename Config::G0TS{});
    constexpr int bK0 = hip_tensor::get<2>(typename Config::G0TS{});
    constexpr int pS0 = hip_tensor::get<3>(typename Config::G0TS{});

    constexpr int bM1 = hip_tensor::get<0>(typename Config::G1TS{});
    constexpr int bN1 = hip_tensor::get<1>(typename Config::G1TS{});
    constexpr int bK1 = hip_tensor::get<2>(typename Config::G1TS{});
    constexpr int pS1 = hip_tensor::get<3>(typename Config::G1TS{});

    static_assert(bM0 == bM1);
    constexpr int bM = bM0;
    constexpr int arch = Config::Arch::value;
    constexpr int threads = Config::Threads::value;

    auto ceilDiv = [](auto x, auto y) { return (x + y - 1) / y; };
    const auto roundEC = ceilDiv(ctx.EC, bM) * bM;

    const auto symHeap = Heap{
        ctx.symHeap, ctx.nLx, roundEC, kArgs.H, sizeof(DataType)
    };

    const auto processors = gridDim.x - 1;
    const auto superBlockSize = min(dispatchSuperBlockSize(kArgs.E), processors);
    const auto dispatchBlocks = (min(min(superBlockSize * kArgs.E, static_cast<unsigned int>(MAX_DISPATCH_BLOCKS)), processors) / superBlockSize) * superBlockSize;

    if (blockIdx.x == gridDim.x - 1) {
        constexpr auto subscriberCount = threads - scheduler::SCHEDULER_COUNT;
        static_assert(subscriberCount > 0 && subscriberCount % flashmoe::WARP_SIZE == 0);
        os::start<topo, subscriberCount, threads, bM, Config::PSS::value, DataType>(
            flashWorkspace, kArgs.expertCounts, symHeap, ctx, kArgs.EC,
            kArgs.I / bN0, kArgs.H / bN1, dispatchBlocks, kArgs.E, kArgs.I, processors);
        return;
    }

    const auto stateNumber = ctx.stateNumbers[blockIdx.x];

    if (blockIdx.x < dispatchBlocks) {
        dispatch<topo, Config::Threads::value, bM, bN0>(
            kArgs.H, kArgs.E, symHeap, kArgs.EC, roundEC,
            ctx.epRank, ctx.world, superBlockSize, dispatchBlocks, tokens, ctx.signals,
            kArgs.expertCounts, ctx.tokenIndices, ctx.dispatchSync, ctx.pel,
            flashWorkspace, stateNumber);
    }

    const auto pA = processor::ProcessorArgs{
        ctx.statusQueue + blockIdx.x,
        ctx.tqs + blockIdx.x,
        ctx.gTqHeads,
        ctx.tQ,
        ctx.pTq,
        ctx.tileSync
    };

    using AccumType = typename Config::AccumType;
    using GEMM0Act = typename ActivationType<AccumType, a>::AT;

    const auto tilesN0 = kArgs.I / bN0;
    const auto tilesN1 = kArgs.H / bN1;
    const auto ecTilesM = ceilDiv(kArgs.EC, bM);

    using TileGEMM0 = tile::CollectiveMainloop<bM0, bN0, bK0, arch, DataType, AccumType, threads, pS0>;
    using TileGEMM1 = tile::CollectiveMainloop<bM1, bN1, bK1, arch, DataType, AccumType, threads, pS1>;

    auto producerBM = hip_tensor::make_tensor(
        hip_tensor::make_gmem_ptr(ctx.producerCombineBitMap),
        hip_tensor::make_layout(
            hip_tensor::make_shape(static_cast<uint>(ctx.world),
                static_cast<uint>(ctx.nLx), ecTilesM, tilesN1),
            hip_tensor::Stride<int,int,int,int>(
                ctx.nLx * ecTilesM * tilesN1,
                ecTilesM * tilesN1,
                tilesN1,
                1)
        )
    );

    const auto* __restrict__ expertUp = reinterpret_cast<const DataType*>(kArgs.expertUpWeights);
    const auto* __restrict__ expertUpV = reinterpret_cast<const DataType*>(kArgs.expertUpVWeights);
    const auto* __restrict__ biasUp = reinterpret_cast<const DataType*>(kArgs.biasUp);
    const auto* __restrict__ biasUpV = reinterpret_cast<const DataType*>(kArgs.biasUpV);
    const auto* __restrict__ expertDown = reinterpret_cast<const DataType*>(kArgs.expertDownWeights);
    const auto* __restrict__ biasDown = reinterpret_cast<const DataType*>(kArgs.biasDown);
    auto* __restrict__ moeOut = reinterpret_cast<DataType*>(kArgs.moeOut);

    processor::start<static_cast<MLPMatmulType>(Config::MT::value), topo, threads,
        static_cast<CombineMode>(Config::CM::value), TileGEMM0, TileGEMM1, GEMM0Act>(
        flashWorkspace, kArgs.S, kArgs.H, kArgs.I, roundEC,
        kArgs.flagColStride, tilesN0, tilesN1,
        expertUp, expertUpV, biasUp, biasUpV,
        static_cast<AccumType>(kArgs.swishAlpha), static_cast<AccumType>(kArgs.swishBeta),
        expertDown, biasDown,
        ctx.tokenIndices, moeOut, producerBM, stateNumber, symHeap, pA);
    __syncthreads();
    if (!threadIdx.x) {
        ctx.stateNumbers[blockIdx.x] = sbs::next(stateNumber);
    }
}

// ============================================================================
// Host-Side Forward Launch Functions
// ============================================================================

/**
 * @brief Launch the MoE forward kernel (HIP version)
 *
 * Template Parameters:
 *   Config - MoEConfig with compile-time parameters
 *   topo   - Network topology
 *   a      - Activation function type
 *
 * Parameters:
 *   kArgs      - Kernel arguments (pointers and dimensions)
 *   ctx        - Execution context (contains smemSize and state numbers)
 *   stream     - HIP stream for asynchronous execution
 */
template <
    typename Config,
    Topology topo,
    Activation a
>
__host__ __forceinline__
void forwardHostDeviceArgs(const KernelArgs& kArgs,
                           const KernelArgs* deviceKArgs,
                           const Context& ctx,
                           const Context* deviceCtx,
                           hipStream_t stream) {
    if constexpr (static_cast<CombineMode>(Config::CM::value) == CombineMode::plural) {
        const hipError_t err = hipMemsetAsync(
            kArgs.moeOut, 0,
            sizeof(typename Config::DType) * kArgs.S * static_cast<size_t>(kArgs.H),
            stream);
        if (err != hipSuccess) {
            fprintf(stderr, "[FlashMoE] hipMemsetAsync FAILED: %s (%s)\n",
                    hipGetErrorName(err), hipGetErrorString(err));
        }
    }

    forward<Config, a, topo>
        <<<ctx.blocks, Config::Threads::value, ctx.smemSize, stream>>>(deviceKArgs, deviceCtx);
    hipError_t launchErr = hipPeekAtLastError();
    if (launchErr != hipSuccess) {
        fprintf(stderr, "[FlashMoE] Kernel launch FAILED: %s (%s)\n",
               hipGetErrorName(launchErr), hipGetErrorString(launchErr));
    }
}

template <
    typename Config,
    Topology topo,
    Activation a
>
__host__ __forceinline__
void forwardHost(const KernelArgs& kArgs, const Context& ctx, hipStream_t stream) {
    KernelArgs* deviceKArgs = nullptr;
    Context* deviceCtx = nullptr;
    if (!checkHipHostCall(hipMallocAsync(&deviceKArgs, sizeof(KernelArgs), stream),
                          "hipMallocAsync KernelArgs")) {
        return;
    }
    if (!checkHipHostCall(hipMallocAsync(&deviceCtx, sizeof(Context), stream),
                          "hipMallocAsync Context")) {
        checkHipHostCall(hipFreeAsync(deviceKArgs, stream), "hipFreeAsync KernelArgs");
        return;
    }
    if (!checkHipHostCall(hipMemcpyAsync(deviceKArgs, &kArgs, sizeof(KernelArgs),
                                         hipMemcpyHostToDevice, stream),
                          "hipMemcpyAsync KernelArgs") ||
        !checkHipHostCall(hipMemcpyAsync(deviceCtx, &ctx, sizeof(Context),
                                         hipMemcpyHostToDevice, stream),
                          "hipMemcpyAsync Context")) {
        checkHipHostCall(hipFreeAsync(deviceKArgs, stream), "hipFreeAsync KernelArgs");
        checkHipHostCall(hipFreeAsync(deviceCtx, stream), "hipFreeAsync Context");
        return;
    }
    forwardHostDeviceArgs<Config, topo, a>(kArgs, deviceKArgs, ctx, deviceCtx, stream);
    checkHipHostCall(hipFreeAsync(deviceKArgs, stream), "hipFreeAsync KernelArgs");
    checkHipHostCall(hipFreeAsync(deviceCtx, stream), "hipFreeAsync Context");
}

/**
 * @brief Launch the MoE forward kernel with explicit grid configuration (benchmark version)
 *
 * This version is useful for benchmarking with explicit device memory allocation.
 *
 * Parameters:
 *   kArgs      - Kernel arguments
 *   ctx        - Execution context
 *   sharedSize - Dynamic shared memory size
 *   blocks     - Number of thread blocks
 *   stream     - HIP stream
 */
template <
    typename Config,
    Topology topo,
    Activation a
>
__host__ __forceinline__
void forwardHostBench(const KernelArgs& kArgs, Context& ctx, const uint& sharedSize,
                      const dim3& blocks, hipStream_t stream) {
    if constexpr (static_cast<CombineMode>(Config::CM::value) == CombineMode::plural) {
        hipError_t err = hipMemsetAsync(
            kArgs.moeOut, 0,
            sizeof(typename Config::DType) * kArgs.S * static_cast<size_t>(kArgs.H),
            stream
        );
        if (err != hipSuccess) {
            printf("hipMemsetAsync failed: %s\n", hipGetErrorString(err));
            return;
        }
    }

    KernelArgs* deviceKArgs = nullptr;
    Context* deviceCtx = nullptr;
    if (!checkHipHostCall(hipMallocAsync(&deviceKArgs, sizeof(KernelArgs), stream),
                          "hipMallocAsync KernelArgs") ||
        !checkHipHostCall(hipMallocAsync(&deviceCtx, sizeof(Context), stream),
                          "hipMallocAsync Context") ||
        !checkHipHostCall(hipMemcpyAsync(deviceKArgs, &kArgs, sizeof(KernelArgs),
                                         hipMemcpyHostToDevice, stream),
                          "hipMemcpyAsync KernelArgs") ||
        !checkHipHostCall(hipMemcpyAsync(deviceCtx, &ctx, sizeof(Context),
                                         hipMemcpyHostToDevice, stream),
                          "hipMemcpyAsync Context")) {
        if (deviceKArgs) checkHipHostCall(hipFreeAsync(deviceKArgs, stream), "hipFreeAsync KernelArgs");
        if (deviceCtx) checkHipHostCall(hipFreeAsync(deviceCtx, stream), "hipFreeAsync Context");
        return;
    }
    forward<Config, a, topo><<<blocks, Config::Threads::value, sharedSize, stream>>>(deviceKArgs, deviceCtx);
    checkHipHostCall(hipFreeAsync(deviceKArgs, stream), "hipFreeAsync KernelArgs");
    checkHipHostCall(hipFreeAsync(deviceCtx, stream), "hipFreeAsync Context");

    // Synchronize to ensure completion (for benchmarking)
    hipError_t err2 = hipStreamSynchronize(stream);
    if (err2 != hipSuccess) {
        printf("hipStreamSynchronize failed: %s\n", hipGetErrorString(err2));
    }
}

// ============================================================================
// Helper Functions for Kernel Configuration
// ============================================================================

/**
 * @brief Calculate shared memory size for the forward kernel
 */
template<typename Config, int bM>
__host__ __forceinline__
constexpr auto getForwardSharedSize(const uint& world, const uint& nLx,
    const uint& E, const uint& EC, const uint& tilesN1) {
    return os::getSharedSize<Config::Threads::value, bM>(world, nLx, E, EC, tilesN1);
}

/**
 * @brief Suggest thread count based on tile dimensions
 *
 * For MI450/gfx1250, thread counts should be multiples of 32 (wavefront size).
 * Common choices:
 *   - 128 threads (4 wavefronts) for compute-bound kernels
 *   - 256 threads (8 wavefronts) for memory-bound kernels
 *   - 512 threads (16 wavefronts) for high-occupancy memory-bound kernels
 */
template<int bM, int bN, int bK>
__host__ __forceinline__
constexpr int suggestThreadCount() {
    // Adjust based on tile size
    constexpr int tileElements = bM * bN;
    if constexpr (tileElements >= 8192) {
        return 8 * flashmoe::WARP_SIZE;  // 256 threads for large tiles
    } else if constexpr (tileElements >= 2048) {
        return 4 * flashmoe::WARP_SIZE;  // 128 threads for medium tiles
    } else {
        return 2 * flashmoe::WARP_SIZE;
    }
}

} // namespace flashmoe::moe

#endif // FLASHMOE_HIP_MOE_CUH
