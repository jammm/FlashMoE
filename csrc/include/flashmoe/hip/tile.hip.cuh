//
// FlashMoE HIP Port - Tile GEMM Implementation using rocWMMA
// Replaces cuBLASDx-based tile.cuh for MI450/gfx1250.
//
// Copyright (c) 2025, Osayamen Jonathan Aimuyo
// All rights reserved.
//
// ============================================================================
// PERFORMANCE NOTE:
// ============================================================================
// For standalone GEMM operations, use hipBLASLt via tile_gemm_dispatch.hip.cuh.
// Custom rocWMMA kernels remain useful when the operation must stay fused.
//
// This file provides rocWMMA-based tile GEMM for:
// 1. Fused operations (GEMM + activation + combine)
// 2. Specialized memory access patterns in MoE dispatch
// 3. Small matrix operations where library overhead dominates
//
// For large standalone GEMM: Use flashmoe::gemm::gemm_fp16() from
// tile_gemm_dispatch.hip.cuh with GemmBackend::HIPBLASLT
// ============================================================================
//

#ifndef FLASHMOE_HIP_TILE_CUH
#define FLASHMOE_HIP_TILE_CUH

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

// rocWMMA for matrix multiply-accumulate operations
#include <rocwmma/rocwmma.hpp>

#include "cuda_compat.hpp"
#include "constants.hpp"
#include "tensor.hpp"

namespace flashmoe
{

// ============================================================================
// Data Type Converters (HIP versions)
// ============================================================================

template <typename T, typename S>
struct Converter {
    __device__ auto operator()(const S& x) const {
        return static_cast<T>(x);
    }
};

template <>
struct Converter<__half, float> {
    __device__ auto operator()(const float& x) const {
        return __float2half(x);
    }
};

template <>
struct Converter<float, __half> {
    __device__ auto operator()(const __half& x) const {
        return __half2float(x);
    }
};

template <>
struct Converter<hip_bfloat16, float> {
    __device__ auto operator()(const float& x) const {
        return hip_bfloat16(x);
    }
};

template <>
struct Converter<float, hip_bfloat16> {
    __device__ auto operator()(const hip_bfloat16& x) const {
        return static_cast<float>(x);
    }
};

template <>
struct Converter<float2, __half2> {
    __device__ auto operator()(const __half2& x) const {
        return __half22float2(x);
    }
};

template <>
struct Converter<__half2, float2> {
    __device__ auto operator()(const float2& x) const {
        return __float22half2_rn(x);
    }
};

template <>
struct Converter<float2, __hip_bfloat162> {
    __device__ auto operator()(const __hip_bfloat162& x) const {
        return __bfloat1622float2(x);
    }
};

template <>
struct Converter<__hip_bfloat162, float2> {
    __device__ auto operator()(const float2& x) const {
        return __float22bfloat162_rn(x);
    }
};

// Double to float conversion
template <>
struct Converter<float, double> {
    __device__ auto operator()(const double& x) const {
        return static_cast<float>(x);
    }
};

// TF32 fallback to FP32 on AMD (no TF32 support)
template <>
struct Converter<float, float> {
    __device__ auto operator()(const float& x) const {
        return x;
    }
};

// ============================================================================
// Type trait helpers
// ============================================================================

template <class T>
struct isTensor : std::false_type {};

template <typename T, typename LayoutT>
struct isTensor<hip_tensor::Tensor<T, LayoutT>> : std::true_type {};

template <typename T, typename LayoutT>
struct isTensor<const hip_tensor::Tensor<T, LayoutT>> : std::true_type {};

} // namespace flashmoe

namespace flashmoe::tile
{

constexpr int MAX_ALIGN = 16;

// ============================================================================
// Matrix arrangement types (replacing cublasdx::arrangement)
// ============================================================================

enum class arrangement {
    row_major,
    col_major
};

// ============================================================================
// Synchronization primitives (replacing CuTe async copy)
// AMD doesn't have cuda::pipeline - use simpler LDS staging with barriers
// ============================================================================

template <int N>
__device__ __forceinline__
void cpWait() {
    // AMD doesn't have async copy fencing like CUDA's cp_async_wait
    // All global-to-LDS copies are synchronous via buffer_load + LDS store
    // Just ensure all LDS writes are visible
    __syncthreads();
}

__device__ __forceinline__
void cpFence() {
    // Memory fence for LDS coherency
    __threadfence_block();
}

// ============================================================================
// rocWMMA Tile Configuration
// ============================================================================

// gfx1250 uses wave32 WMMA. Dense FP16/BF16 to FP32 accumulation uses
// 16×16×32; FP32 uses 16×16×4.

namespace wmma_config {
    // Primary configuration: 16×16×16 for FP16 (good balance)
    constexpr int WMMA_M_16 = 16;
    constexpr int WMMA_N_16 = 16;
    constexpr int WMMA_K_16 = 16;
    constexpr int WMMA_K_16_GFX12_FP16 = 32;
    constexpr int WMMA_K_16_GFX12_FP32 = 4;

    // Alternative: 32×32×8 for larger tiles
    constexpr int WMMA_M_32 = 32;
    constexpr int WMMA_N_32 = 32;
    constexpr int WMMA_K_32 = 8;
}

// ============================================================================
// rocWMMA Fragment Type Aliases
// ============================================================================

// 16×16×16 configuration fragments
template<typename DataT, typename AccumT = float>
struct WmmaFragments16 {
    using FragmentA = rocwmma::fragment<
        rocwmma::matrix_a,
        wmma_config::WMMA_M_16, wmma_config::WMMA_N_16, wmma_config::WMMA_K_16,
        DataT, rocwmma::row_major>;

    using FragmentB = rocwmma::fragment<
        rocwmma::matrix_b,
        wmma_config::WMMA_M_16, wmma_config::WMMA_N_16, wmma_config::WMMA_K_16,
        DataT, rocwmma::col_major>;

    using FragmentAcc = rocwmma::fragment<
        rocwmma::accumulator,
        wmma_config::WMMA_M_16, wmma_config::WMMA_N_16, wmma_config::WMMA_K_16,
        AccumT>;
};

// 32×32×8 configuration fragments
template<typename DataT, typename AccumT = float>
struct WmmaFragments32 {
    using FragmentA = rocwmma::fragment<
        rocwmma::matrix_a,
        wmma_config::WMMA_M_32, wmma_config::WMMA_N_32, wmma_config::WMMA_K_32,
        DataT, rocwmma::row_major>;

    using FragmentB = rocwmma::fragment<
        rocwmma::matrix_b,
        wmma_config::WMMA_M_32, wmma_config::WMMA_N_32, wmma_config::WMMA_K_32,
        DataT, rocwmma::col_major>;

    using FragmentAcc = rocwmma::fragment<
        rocwmma::accumulator,
        wmma_config::WMMA_M_32, wmma_config::WMMA_N_32, wmma_config::WMMA_K_32,
        AccumT>;
};

// ============================================================================
// Coordinate Utilities
// ============================================================================

__device__ __forceinline__
constexpr auto idx2Coord(const int& tilesM, const int& tilesN, const int& tileIdx) {
    const int tileRow = tileIdx / tilesN;
    const int tileCol = tileIdx % tilesN;
    return hip_tensor::make_coord(tileRow, tileCol, hip_tensor::_);
}

// ============================================================================
// Tile Access Helpers
// ============================================================================

template <int tRow, int tCol, arrangement ar, typename Element>
__device__ __forceinline__
constexpr int computeOffset(const int& row, const int& col, const int& ldim) {
    if constexpr (ar == arrangement::row_major) {
        return row * ldim + col;  // Row-major: stride is number of columns
    } else {
        return col * ldim + row;  // Col-major: stride is number of rows
    }
}

template <int tRow, int tCol, arrangement ar, typename Element>
__device__ __forceinline__
const Element* getTilePtr(const Element* __restrict__ const& p,
                          const int& nRow, const int& nCol,
                          const int& tileRow, const int& tileCol) {
    const int rowStart = tileRow * tRow;
    const int colStart = tileCol * tCol;

    if constexpr (ar == arrangement::row_major) {
        return p + rowStart * nCol + colStart;
    } else {
        return p + colStart * nRow + rowStart;
    }
}

template <int tRow, int tCol, arrangement ar, typename Element>
__device__ __forceinline__
Element* getTilePtrMut(Element* __restrict__ const& p,
                       const int& nRow, const int& nCol,
                       const int& tileRow, const int& tileCol) {
    const int rowStart = tileRow * tRow;
    const int colStart = tileCol * tCol;

    if constexpr (ar == arrangement::row_major) {
        return p + rowStart * nCol + colStart;
    } else {
        return p + colStart * nRow + rowStart;
    }
}

// Leading dimension calculation helpers
template <arrangement ar, int bM, int bK>
constexpr int ldA = (ar == arrangement::row_major) ? bK : bM;

template <arrangement br, int bK, int bN>
constexpr int ldB = (br == arrangement::col_major) ? bK : bN;

template <arrangement cr, int bM, int bN>
constexpr int ldC = (cr == arrangement::row_major) ? bN : bM;

// ============================================================================
// LDS (Shared Memory) Staging for Pipelined Loading
// ============================================================================

template<typename Element, int Size>
struct LDSBuffer {
    Element data[Size];

    __device__ __forceinline__
    Element* ptr() { return data; }

    __device__ __forceinline__
    const Element* ptr() const { return data; }

    __device__ __forceinline__
    Element& operator[](int idx) { return data[idx]; }

    __device__ __forceinline__
    const Element& operator[](int idx) const { return data[idx]; }
};

// ============================================================================
// Global to LDS Copy (replacing cuBLASDx async copy)
// Uses vectorized loads for MI450/gfx1250.
// ============================================================================

template<typename Element, int VectorWidth = 4>
__device__ __forceinline__
void copyGlobalToLDS(Element* __restrict__ lds_dest,
                     const Element* __restrict__ gmem_src,
                     const int count,
                     const int tid,
                     const int numThreads) {
    // Vectorized copy pattern
    using VecType = typename std::conditional<
        sizeof(Element) == 2 && VectorWidth == 4,
        uint2,  // 4 × FP16 = 64 bits
        typename std::conditional<
            sizeof(Element) == 2 && VectorWidth == 8,
            uint4,  // 8 × FP16 = 128 bits
            Element
        >::type
    >::type;

    constexpr int elementsPerVec = sizeof(VecType) / sizeof(Element);
    const int vecCount = count / elementsPerVec;

    const VecType* gmem_vec = reinterpret_cast<const VecType*>(gmem_src);
    VecType* lds_vec = reinterpret_cast<VecType*>(lds_dest);

    // Cooperative copy across threads
    for (int i = tid; i < vecCount; i += numThreads) {
        lds_vec[i] = gmem_vec[i];
    }

    // Handle remainder elements
    const int remainder_start = vecCount * elementsPerVec;
    for (int i = remainder_start + tid; i < count; i += numThreads) {
        lds_dest[i] = gmem_src[i];
    }
}

// ============================================================================
// Accumulator Type (wrapper for rocWMMA fragment operations)
// ============================================================================

template<int M, int N, int K, typename AccumT = float>
struct Accumulator {
    using FragmentType = rocwmma::fragment<
        rocwmma::accumulator, M, N, K, AccumT>;

    FragmentType frag;

    __device__ __forceinline__
    void clear() {
        rocwmma::fill_fragment(frag, AccumT(0));
    }

    __device__ __forceinline__
    FragmentType& get() { return frag; }

    __device__ __forceinline__
    const FragmentType& get() const { return frag; }
};

// ============================================================================
// WMMA-based Tile GEMM Executor
// Performs C += A × B using rocWMMA instructions
// ============================================================================

template<
    int WMMA_M,
    int WMMA_N,
    int WMMA_K,
    typename DataT,
    typename AccumT,
    arrangement ALayout = arrangement::row_major,
    arrangement BLayout = arrangement::col_major
>
struct WmmaTileGemm {
    // Fragment types based on layout
    using FragmentA = std::conditional_t<
        ALayout == arrangement::row_major,
        rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, DataT, rocwmma::row_major>,
        rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, DataT, rocwmma::col_major>
    >;

    using FragmentB = std::conditional_t<
        BLayout == arrangement::col_major,
        rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, DataT, rocwmma::col_major>,
        rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, DataT, rocwmma::row_major>
    >;

    using FragmentAcc = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, AccumT>;

    // Execute GEMM on tile from LDS
    __device__ __forceinline__
    static void execute(const DataT* __restrict__ sA,  // LDS pointer to A tile
                       const DataT* __restrict__ sB,  // LDS pointer to B tile
                       FragmentAcc& acc,              // Accumulator fragment
                       const int ldA,                 // Leading dim of A in LDS
                       const int ldB) {               // Leading dim of B in LDS
        FragmentA fragA;
        FragmentB fragB;

        // Load A tile from LDS
        rocwmma::load_matrix_sync(fragA, sA, ldA);

        // Load B tile from LDS
        rocwmma::load_matrix_sync(fragB, sB, ldB);

        // Matrix multiply-accumulate: acc += A × B
        rocwmma::mma_sync(acc, fragA, fragB, acc);
    }

    // Store accumulator to LDS
    __device__ __forceinline__
    static void store(AccumT* __restrict__ dest,
                     const FragmentAcc& acc,
                     const int ldC) {
        rocwmma::store_matrix_sync(dest, acc, ldC, rocwmma::mem_row_major);
    }

    // Store accumulator to LDS with column-major layout
    __device__ __forceinline__
    static void storeColMajor(AccumT* __restrict__ dest,
                              const FragmentAcc& acc,
                              const int ldC) {
        rocwmma::store_matrix_sync(dest, acc, ldC, rocwmma::mem_col_major);
    }
};

// ============================================================================
// CollectiveMainloop - rocWMMA-based GEMM mainloop
// Replaces cuBLASDx CollectiveMainloop with LDS-staged pipelined GEMM
// ============================================================================

template <
    int bM, int bN, int bK,         // Block tile shape
    int Arch,                        // GPU architecture (e.g. 942, 950)
    typename Element,                // Input data type (FP16, BF16)
    typename AccumT,                 // Accumulator type (usually float)
    int threads,                     // Threads per block
    int pipeStages = 2,              // Pipeline stages (double buffering)
    arrangement ar = arrangement::row_major,   // A matrix layout
    arrangement br = arrangement::col_major,   // B matrix layout
    arrangement cr = arrangement::row_major,   // C matrix layout
    int aAlignment = MAX_ALIGN,
    int bAlignment = MAX_ALIGN,
    int cAlignment = MAX_ALIGN
>
    requires(pipeStages > 0)
struct CollectiveMainloop {
    static constexpr bool IS_GFX12 = (Arch >= 1200 && Arch < 1300);
    static constexpr bool IS_TWO_BYTE_INPUT = (sizeof(Element) == 2);
    static constexpr int ARCH_WAVE_SIZE = IS_GFX12 ? 32 : flashmoe::WARP_SIZE;

    // gfx12 WMMA is wave32 and block-dim 16 only.
    static constexpr int WMMA_M = IS_GFX12 ? 16 : ((bM >= 32 && bN >= 32) ? 32 : 16);
    static constexpr int WMMA_N = IS_GFX12 ? 16 : ((bM >= 32 && bN >= 32) ? 32 : 16);
    static constexpr int WMMA_K = IS_GFX12
        ? (IS_TWO_BYTE_INPUT ? wmma_config::WMMA_K_16_GFX12_FP16 : wmma_config::WMMA_K_16_GFX12_FP32)
        : ((WMMA_M == 32) ? wmma_config::WMMA_K_32 : wmma_config::WMMA_K_16);
    static_assert(!IS_GFX12 || WMMA_M == 16, "gfx12 WMMA requires a 16x16 block shape");
    static_assert(!IS_GFX12 || (WMMA_K == 32 || WMMA_K == 4), "unsupported gfx12 WMMA K shape");
    static_assert(bM % WMMA_M == 0 && bN % WMMA_N == 0 && bK % WMMA_K == 0,
                  "block tile dimensions must be multiples of the selected WMMA shape");
    static_assert(threads % ARCH_WAVE_SIZE == 0, "thread count must be a whole number of waves");

    // Number of WMMA tiles per block tile
    static constexpr int TILES_M = bM / WMMA_M;
    static constexpr int TILES_N = bN / WMMA_N;
    static constexpr int TILES_K = bK / WMMA_K;

    // LDS sizes per stage
    static constexpr int sASize = bM * bK;
    static constexpr int sBSize = bK * bN;

    // Total shared memory for pipelining
    static constexpr int SharedSize = (sASize + sBSize) * pipeStages * sizeof(Element);

    using TileShape = hip_tensor::Shape<hip_tensor::Int<bM>, hip_tensor::Int<bN>, hip_tensor::Int<bK>>;
    using Threads = hip_tensor::Int<threads>;
    using PipeStages = hip_tensor::Int<pipeStages>;
    using AccumType = AccumT;
    using TileArch = hip_tensor::Int<Arch>;
    using DataType = Element;
    struct SharedSizeC { static constexpr int value = bM * bN * sizeof(AccumT); };
    struct SharedSizeAB { static constexpr int value = SharedSize; };
    struct GeneralAlignment { static constexpr int value = MAX_ALIGN; };

    using WmmaExecutor = WmmaTileGemm<WMMA_M, WMMA_N, WMMA_K, Element, AccumT, ar, br>;
    using FragmentAcc = typename WmmaExecutor::FragmentAcc;

    // ========================================================================
    // Main GEMM Execution
    // ========================================================================

    template <typename TileCoord>
    __device__ __forceinline__
    void operator()(void* __restrict__ const& workspace,
                    const Element* __restrict__ const& a,  // Global A pointer
                    const Element* __restrict__ const& b,  // Global B pointer
                    FragmentAcc& accumulator,              // Output accumulator
                    const int& M, const int& N, const int& K,
                    const TileCoord& tileCoord) const {

        // LDS pointers with staging buffers
        Element* sA = static_cast<Element*>(workspace);
        Element* sB = sA + (sASize * pipeStages);

        // Tile coordinates
        const int tileRow = hip_tensor::get<0>(tileCoord);
        const int tileCol = hip_tensor::get<1>(tileCoord);

        // Thread info
        const int tid = threadIdx.x;
        const int numThreads = blockDim.x;

        // Number of K tiles to process
        const int tilesK = K / bK;

        // Initialize accumulator to zero
        rocwmma::fill_fragment(accumulator, AccumT(0));

        // ====================================================================
        // Pipeline Prologue: Prime the LDS buffers
        // ====================================================================

        #pragma unroll
        for (int stage = 0; stage < pipeStages && stage < tilesK; ++stage) {
            Element* sAStage = sA + (stage * sASize);
            Element* sBStage = sB + (stage * sBSize);

            // Calculate global tile pointers
            const Element* gA = getTilePtr<bM, bK, ar>(a, M, K, tileRow, stage);
            const Element* gB = getTilePtr<bK, bN, br>(b, K, N, stage, tileCol);

            // Copy A tile: Global → LDS
            copyGlobalToLDS<Element>(sAStage, gA, sASize, tid, numThreads);

            // Copy B tile: Global → LDS
            copyGlobalToLDS<Element>(sBStage, gB, sBSize, tid, numThreads);
        }

        __syncthreads();  // Ensure prologue data is ready

        // ====================================================================
        // Pipeline Mainloop
        // ====================================================================

        for (int kStage = pipeStages; kStage < tilesK; ++kStage) {
            const int readStage = kStage % pipeStages;
            const int computeStage = (kStage - pipeStages + 1 + pipeStages) % pipeStages;

            Element* sACompute = sA + (computeStage * sASize);
            Element* sBCompute = sB + (computeStage * sBSize);
            Element* sARead = sA + (readStage * sASize);
            Element* sBRead = sB + (readStage * sBSize);

            // Compute on current stage while loading next
            executeWmmaTiles(sACompute, sBCompute, accumulator);

            __syncthreads();

            // Load next tiles
            const Element* gA = getTilePtr<bM, bK, ar>(a, M, K, tileRow, kStage);
            const Element* gB = getTilePtr<bK, bN, br>(b, K, N, kStage, tileCol);

            copyGlobalToLDS<Element>(sARead, gA, sASize, tid, numThreads);
            copyGlobalToLDS<Element>(sBRead, gB, sBSize, tid, numThreads);

            __syncthreads();
        }

        // ====================================================================
        // Pipeline Epilogue: Drain remaining stages
        // ====================================================================

        #pragma unroll
        for (int stage = 0; stage < pipeStages && stage < tilesK; ++stage) {
            const int computeStage = (tilesK - pipeStages + stage + pipeStages) % pipeStages;

            if (tilesK > stage) {
                Element* sACompute = sA + (computeStage * sASize);
                Element* sBCompute = sB + (computeStage * sBSize);

                executeWmmaTiles(sACompute, sBCompute, accumulator);

                if (stage < pipeStages - 1) {
                    __syncthreads();
                }
            }
        }
    }

private:
    // Execute all WMMA tiles within a block tile
    __device__ __forceinline__
    void executeWmmaTiles(const Element* __restrict__ sA,
                         const Element* __restrict__ sB,
                         FragmentAcc& acc) const {
        // For simplicity, execute a single WMMA tile per warp
        // In production, you'd distribute TILES_M × TILES_N across warps

        constexpr int wavesPerBlock = threads / ARCH_WAVE_SIZE;
        const int waveId = threadIdx.x / ARCH_WAVE_SIZE;

        // Simple distribution: each wave handles one WMMA tile
        // For larger blocks, loop over multiple tiles per wave

        const int totalWmmaTiles = TILES_M * TILES_N;

        for (int tileIdx = waveId; tileIdx < totalWmmaTiles; tileIdx += wavesPerBlock) {
            const int wmmaTileM = tileIdx / TILES_N;
            const int wmmaTileN = tileIdx % TILES_N;

            // Accumulate over K dimension
            #pragma unroll
            for (int wmmaTileK = 0; wmmaTileK < TILES_K; ++wmmaTileK) {
                // Calculate LDS offsets for this WMMA tile
                const int offsetA = wmmaTileM * WMMA_M * bK + wmmaTileK * WMMA_K;
                const int offsetB = wmmaTileK * WMMA_K * bN + wmmaTileN * WMMA_N;

                // Leading dimensions in LDS
                constexpr int ldA_lds = (ar == arrangement::row_major) ? bK : bM;
                constexpr int ldB_lds = (br == arrangement::col_major) ? bK : bN;

                WmmaExecutor::execute(sA + offsetA, sB + offsetB, acc, ldA_lds, ldB_lds);
            }
        }
    }
};

// ============================================================================
// Suggest Thread Count (compatibility with cuBLASDx pattern)
// ============================================================================

template <
    int bM, int bN, int bK,
    int Arch = 942,
    typename Element = __half,
    typename AccumT = float,
    arrangement ar = arrangement::row_major,
    arrangement br = arrangement::col_major,
    arrangement cr = arrangement::row_major,
    int aAlignment = MAX_ALIGN,
    int bAlignment = MAX_ALIGN,
    int cAlignment = MAX_ALIGN
>
constexpr int suggest_thread_count() {
    constexpr bool isGfx12 = (Arch >= 1200 && Arch < 1300);
    constexpr int WMMA_M = isGfx12 ? 16 : ((bM >= 32 && bN >= 32) ? 32 : 16);
    constexpr int WMMA_N = isGfx12 ? 16 : ((bM >= 32 && bN >= 32) ? 32 : 16);
    constexpr int waveSize = isGfx12 ? 32 : flashmoe::WARP_SIZE;
    constexpr int tilesM = bM / WMMA_M;
    constexpr int tilesN = bN / WMMA_N;
    constexpr int totalTiles = tilesM * tilesN;
    constexpr int wavesNeeded = (totalTiles + 3) / 4;
    constexpr int suggestedWaves = (wavesNeeded < 2) ? 2 :
                                   (wavesNeeded > 8) ? 8 : wavesNeeded;
    return suggestedWaves * waveSize;
}

// ============================================================================
// Bias Loading Helper
// ============================================================================

template <int bM, int bN, typename Element>
__device__ __forceinline__
void loadBiasToLDS(Element* __restrict__ lds_bias,
                   const Element* __restrict__ gmem_bias,
                   const int N,
                   const int tileCol,
                   const int tid,
                   const int numThreads) {
    const int biasStart = tileCol * bN;
    const Element* biasPtr = gmem_bias + biasStart;

    // Cooperative load - each thread loads a subset
    for (int i = tid; i < bN; i += numThreads) {
        lds_bias[i] = biasPtr[i];
    }
}

// ============================================================================
// Store Accumulator to Global Memory with Optional Bias
// ============================================================================

template <int bM, int bN, arrangement cr, typename Element, typename AccumT>
__device__ __forceinline__
void storeAccumulatorWithBias(Element* __restrict__ gmem_c,
                              const AccumT* __restrict__ lds_acc,
                              const Element* __restrict__ lds_bias,  // nullptr if no bias
                              const int M, const int N,
                              const int tileRow, const int tileCol,
                              const int tid,
                              const int numThreads) {
    const int cRowStart = tileRow * bM;
    const int cColStart = tileCol * bN;

    constexpr Converter<Element, AccumT> converter{};

    for (int idx = tid; idx < bM * bN; idx += numThreads) {
        const int localRow = idx / bN;
        const int localCol = idx % bN;

        const int globalRow = cRowStart + localRow;
        const int globalCol = cColStart + localCol;

        if (globalRow < M && globalCol < N) {
            AccumT val = lds_acc[idx];

            // Add bias if provided
            if (lds_bias != nullptr) {
                val += static_cast<AccumT>(lds_bias[localCol]);
            }

            // Calculate global offset based on layout
            int globalOffset;
            if constexpr (cr == arrangement::row_major) {
                globalOffset = globalRow * N + globalCol;
            } else {
                globalOffset = globalCol * M + globalRow;
            }

            gmem_c[globalOffset] = converter(val);
        }
    }
}

} // namespace flashmoe::tile

// ============================================================================
// Convenience macro for declaring shared memory for CollectiveMainloop
// ============================================================================

#define FLASHMOE_TILE_SHARED_MEMORY(MainloopType, name) \
    __shared__ alignas(128) char name[MainloopType::SharedSize]

#endif // FLASHMOE_HIP_TILE_CUH
