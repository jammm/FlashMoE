/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 *
 * HIP Port of MoE Combine Logic
 * Target: AMD MI450 (gfx1250)
 *
 * Key changes from CUDA version:
 * - Replaced cute:: tensor operations with hip_tensor:: equivalents
 * - Replaced cuda::std:: types with std:: types via compat layer
 * - Replaced cublasdx:: with flashmoe::tile:: for arrangement
 * - Replaced PTX reduction instructions with HIP atomicAdd operations
 * - Uses WARP_SIZE from constants.hpp (32 on MI450/gfx1250)
 * - cutlass::AlignedArray replaced with hip_tensor::AlignedArray
 */

#ifndef FLASHMOE_HIP_COMBINE_CUH
#define FLASHMOE_HIP_COMBINE_CUH

// HIP compatibility layer headers
#include "flashmoe/hip/cuda_compat.hpp"
#include "flashmoe/hip/constants.hpp"
#include "flashmoe/hip/tensor.hpp"
#include "flashmoe/hip/atomics.hip.cuh"
#include "flashmoe/hip/tile.hip.cuh"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#include <type_traits>
#include <bit>

namespace flashmoe
{

#ifndef FLASHMOE_HIP_COMBINEMODE_DEFINED
#define FLASHMOE_HIP_COMBINEMODE_DEFINED
enum class CombineMode {
    single,
    plural
};
#endif

// ============================================================================
// Float2 Multiply Helper
// AMD has different intrinsics than CUDA for packed FP16/BF16 operations
// ============================================================================

__device__ __forceinline__
float2 float2Mul(const float2& a, const float2& b) {
    // HIP does not expose a direct __fmul2_rn equivalent here.
    return float2{a.x * b.x, a.y * b.y};
}

// ============================================================================
// Packed Type for Token Index and Probability (if not already defined)
// ============================================================================

#ifndef FLASHMOE_HIP_PACKED_TYPES_DEFINED
#define FLASHMOE_HIP_PACKED_TYPES_DEFINED
struct __align__(8) TPS {
    uint32_t tokenIdx;
    float probability;
};
#endif

// ============================================================================
// Vector Type Infrastructure (HIP port of infra/vt.cuh)
// ============================================================================

constexpr int MAX_ALIGNMENT = 16;

// Power of 2 check
template<int N>
struct is_pow2 {
    static constexpr bool value = (N > 0) && ((N & (N - 1)) == 0);
};

template<int N>
constexpr bool is_pow2_v = is_pow2<N>::value;

// Compile-time min
template<int A, int B>
constexpr int min_v = (A < B) ? A : B;

/**
 * VectorTypeDescriptor - Maps element types to vectorized load/store types
 *
 * On MI450/gfx1250, use 128-bit vector loads where possible.
 * which corresponds to:
 * - 4 × float (FP32)
 * - 8 × __half (FP16)
 * - 8 × hip_bfloat16 (BF16)
 */
template<typename T, int Alignment = MAX_ALIGNMENT>
struct VectorTypeDescriptor {
    using VectorWidth = hip_tensor::Int<Alignment / static_cast<int>(sizeof(T))>;
    using VectorType = hip_tensor::AlignedArray<T, VectorWidth::value, Alignment>;
};

// Element width calculation (compile-time)
template<typename Element, int dim, int MAX_ALIGN = MAX_ALIGNMENT>
    requires(MAX_ALIGN <= MAX_ACCESS_ALIGNMENT && is_pow2_v<MAX_ALIGN> && MAX_ALIGN >= 1)
constexpr int ElementWidth = min_v<dim, MAX_ALIGN / static_cast<int>(sizeof(Element))>;

// Element alignment calculation
template<typename Element, int dim>
constexpr uint32_t ElementAlignment = (is_pow2_v<ElementWidth<Element, dim>> ?
    ElementWidth<Element, dim> : 1) * sizeof(Element);

template<typename Element, int dim, int width>
constexpr int ElementAlignmentForWidth = (is_pow2_v<width> ? width : 1) * sizeof(Element);

// ============================================================================
// Reduction Type Infrastructure (HIP port of infra/rvt.cuh)
// ============================================================================

constexpr int RED_MAX_ALIGNMENT = 16;

/**
 * RedAddType - Type promotion for atomic reductions
 *
 * For FP16 and BF16, promotes to packed x2 types when alignment allows,
 * enabling efficient 32-bit atomic operations.
 */
template<typename Element, int Alignment>
    requires(Alignment > 0 && Alignment <= RED_MAX_ALIGNMENT && is_pow2_v<Alignment>)
struct RedAddType {
    using Type = Element;
    using Width = hip_tensor::Int<1>;
};

template<int Alignment>
struct RedAddType<__half, Alignment> {
#if defined(FLASHMOE_HIP_GFX12)
    // gfx12 packed half2 atomics compile, but plural combine correctness under
    // contended updates is not reliable enough here. Use scalar half atomics.
    using Type = __half;
#else
    // Alignment > sizeof(__half) means that Alignment is 2, 4, 8 or 16
    // This means we can safely promote to __half2
    using Type = std::conditional_t<(Alignment > sizeof(__half)), __half2, __half>;
#endif
    using Width = hip_tensor::Int<sizeof(Type) / sizeof(__half)>;
};

template<int Alignment>
struct RedAddType<hip_bfloat16, Alignment> {
    using Type = std::conditional_t<(Alignment > sizeof(hip_bfloat16)), __hip_bfloat162, hip_bfloat16>;
    using Width = hip_tensor::Int<sizeof(Type) / sizeof(hip_bfloat16)>;
};

/**
 * RedAdd - Atomic reduction operator for HIP
 *
 * Unlike CUDA which uses PTX reduction instructions (red.global.add),
 * HIP atomicAdd is used for FP32, FP16, and BF16 paths where supported.
 *
 * Template Parameters:
 *   Arch        - Architecture level, currently 1250 for MI450/gfx1250
 *   Element     - Element type for reduction
 *   VectorWidth - Maximum vector width for reduction
 */
template<int Arch, typename Element, int VectorWidth>
struct RedAdd {
    static_assert(VectorWidth >= 1 && VectorWidth <= (RED_MAX_ALIGNMENT / sizeof(Element)) &&
        is_pow2_v<VectorWidth>);
    // For HIP, we use atomicAdd-based reductions regardless of architecture
};

// ============================================================================
// RedAdd Specializations for HIP
// Use HIP atomics and gfx1250-specific fallbacks as needed.
// ============================================================================

// Double precision
template<int Arch, int MaxVectorWidth>
struct RedAdd<Arch, double, MaxVectorWidth> {
    using VectorWidth = hip_tensor::Int<1>;

    template<typename T>
        requires(std::is_same_v<typename T::value_type, double>)
    __device__ __forceinline__
    void operator()(double* __restrict__ const& addr, const T& v) const {
        atomicAdd(addr, v[0]);
    }
};

// Float (FP32)
template<int Arch, int MaxVectorWidth>
struct RedAdd<Arch, float, MaxVectorWidth> {
    // AMD supports single-element atomic, use vector width 1 for simplicity
    using VectorWidth = hip_tensor::Int<min_v<MaxVectorWidth, 4>>;

    template<typename T>
        requires(std::is_same_v<typename T::value_type, float>)
    __device__ __forceinline__
    void operator()(float* __restrict__ const& addr, const T& v) const {
        #pragma unroll
        for (int i = 0; i < VectorWidth::value; ++i) {
            atomicAdd(addr + i, v[i]);
        }
    }
};

// Half precision (FP16)
template<int Arch, int MaxVectorWidth>
struct RedAdd<Arch, __half, MaxVectorWidth> {
    using VectorWidth = hip_tensor::Int<min_v<MaxVectorWidth, 8>>;

    template<typename T>
        requires(std::is_same_v<typename T::value_type, __half>)
    __device__ __forceinline__
    void operator()(__half* __restrict__ const& addr, const T& v) const {
        #pragma unroll
        for (int i = 0; i < VectorWidth::value; ++i) {
#if defined(FLASHMOE_HIP_GFX12)
            unsafeAtomicAdd(addr + i, v[i]);
#else
            atomicAdd(addr + i, v[i]);
#endif
        }
    }
};

// Packed half2 (FP16×2)
template<int Arch, int MaxVectorWidth>
struct RedAdd<Arch, __half2, MaxVectorWidth> {
    using VectorWidth = hip_tensor::Int<min_v<MaxVectorWidth, 4>>;

    template<typename T>
        requires(std::is_same_v<typename T::value_type, __half2>)
    __device__ __forceinline__
    void operator()(__half2* __restrict__ const& addr, const T& v) const {
        #pragma unroll
        for (int i = 0; i < VectorWidth::value; ++i) {
            atomicAdd(addr + i, v[i]);
        }
    }
};

// BFloat16
template<int Arch, int MaxVectorWidth>
struct RedAdd<Arch, hip_bfloat16, MaxVectorWidth> {
    using VectorWidth = hip_tensor::Int<min_v<MaxVectorWidth, 8>>;

    template<typename T>
        requires(std::is_same_v<typename T::value_type, hip_bfloat16>)
    __device__ __forceinline__
    void operator()(hip_bfloat16* __restrict__ const& addr, const T& v) const {
        // HIP supports atomicAdd for bf16 on the target stack.
        #pragma unroll
        for (int i = 0; i < VectorWidth::value; ++i) {
            atomicAdd(addr + i, v[i]);
        }
    }
};

// Packed BFloat16×2
template<int Arch, int MaxVectorWidth>
struct RedAdd<Arch, __hip_bfloat162, MaxVectorWidth> {
    using VectorWidth = hip_tensor::Int<min_v<MaxVectorWidth, 4>>;

    template<typename T>
        requires(std::is_same_v<typename T::value_type, __hip_bfloat162>)
    __device__ __forceinline__
    void operator()(__hip_bfloat162* __restrict__ const& addr, const T& v) const {
        // Atomically add each packed pair
        #pragma unroll
        for (int i = 0; i < VectorWidth::value; ++i) {
            atomicAdd(addr + i, v[i]);
        }
    }
};

// ============================================================================
// Architecture Mapping for RedAdd
// Maps CUDA SM numbers to equivalent HIP usage
// ============================================================================

template<int Arch>
constexpr int RedArch = Arch < 800 ? 700 : (Arch < 900 ? 800 : 900);

constexpr int HIP_GFX1250_ARCH = 1250;

// ============================================================================
// HIP Layout and Stride Helpers (replacing CuTe patterns)
// ============================================================================

// Row-major stride: (nCol, 1)
template<int M, int N>
struct RowMajorLayout {
    using Shape = hip_tensor::Shape<hip_tensor::Int<M>, hip_tensor::Int<N>>;
    using Stride = hip_tensor::Stride<hip_tensor::Int<N>, hip_tensor::Int<1>>;
    using Layout = hip_tensor::Layout<Shape, Stride>;

    static __device__ __forceinline__ Layout make() {
        return Layout(Shape{}, Stride{});
    }
};

// Column-major stride: (1, nRow)
template<int M, int N>
struct ColMajorLayout {
    using Shape = hip_tensor::Shape<hip_tensor::Int<M>, hip_tensor::Int<N>>;
    using Stride = hip_tensor::Stride<hip_tensor::Int<1>, hip_tensor::Int<M>>;
    using Layout = hip_tensor::Layout<Shape, Stride>;

    static __device__ __forceinline__ Layout make() {
        return Layout(Shape{}, Stride{});
    }
};

// ============================================================================
// Cooperative Copy: Global to Shared Memory
// Replaces cublasdx::copy for HIP
// ============================================================================

template<int threads, int alignment, typename SrcT, typename DstT, int Size>
__device__ __forceinline__
void copyToShared(const int tid,
                  const SrcT* __restrict__ src,
                  DstT* __restrict__ dst) {
    // Calculate elements per thread
    constexpr int elementsPerThread = Size / threads;
    constexpr int residue = Size - (elementsPerThread * threads);

    // Vectorized copy when possible
    constexpr int vecWidth = alignment / sizeof(SrcT);

    if constexpr (vecWidth >= 2 && Size >= threads * vecWidth) {
        using VecType = hip_tensor::AlignedArray<SrcT, vecWidth, alignment>;
        const VecType* srcVec = reinterpret_cast<const VecType*>(src);
        VecType* dstVec = reinterpret_cast<VecType*>(dst);
        constexpr int vecCount = Size / vecWidth;
        constexpr int vecsPerThread = vecCount / threads;

        #pragma unroll
        for (int i = 0; i < vecsPerThread; ++i) {
            const int idx = tid + i * threads;
            dstVec[idx] = srcVec[idx];
        }

        // Handle remainder
        constexpr int vecResidue = vecCount - (vecsPerThread * threads);
        if constexpr (vecResidue > 0) {
            if (tid < vecResidue) {
                const int idx = tid + vecsPerThread * threads;
                dstVec[idx] = srcVec[idx];
            }
        }
    } else {
        // Scalar copy fallback
        #pragma unroll
        for (int i = 0; i < elementsPerThread; ++i) {
            const int idx = tid + i * threads;
            dst[idx] = src[idx];
        }

        if constexpr (residue > 0) {
            if (tid < residue) {
                const int idx = tid + elementsPerThread * threads;
                dst[idx] = src[idx];
            }
        }
    }
}

// Memory fence after copy (replaces cublasdx::copy_wait)
__device__ __forceinline__
void copyWait() {
    __syncthreads();
}

// ============================================================================
// Tile Access Helper (replacing CuTe local_tile pattern)
// ============================================================================

namespace tile_hip {

template<int tRow, int tCol, tile::arrangement arr, typename Element>
__device__ __forceinline__
Element* getTilePtr(Element* __restrict__ base,
                    const int nRow, const int nCol,
                    const int tileRowIdx, const int tileColIdx) {
    const int rowStart = tileRowIdx * tRow;
    const int colStart = tileColIdx * tCol;

    if constexpr (arr == tile::arrangement::row_major) {
        return base + rowStart * nCol + colStart;
    } else {
        return base + colStart * nRow + rowStart;
    }
}

template<int tRow, int tCol, tile::arrangement arr, typename Element>
__device__ __forceinline__
const Element* getTilePtrConst(const Element* __restrict__ base,
                               const int nRow, const int nCol,
                               const int tileRowIdx, const int tileColIdx) {
    const int rowStart = tileRowIdx * tRow;
    const int colStart = tileColIdx * tCol;

    if constexpr (arr == tile::arrangement::row_major) {
        return base + rowStart * nCol + colStart;
    } else {
        return base + colStart * nRow + rowStart;
    }
}

} // namespace tile_hip

// ============================================================================
// MoE Combine at Tile Granularity
// ============================================================================

/**
 * combine - MoE combine operation at tile granularity
 *
 * Combines expert outputs back to the original token positions.
 * For top-k=1, performs a direct copy. For top-k>1, performs atomic
 * reduction with probability scaling.
 *
 * Template Parameters:
 *   bM       - Tile height (number of tokens per tile)
 *   bN       - Tile width (hidden dimension per tile)
 *   Arch     - Architecture level, currently 1250 for MI450/gfx1250
 *   threads  - Number of threads per block
 *   c        - CombineMode (single or plural)
 *   cArr     - Output arrangement (row_major or col_major)
 *   Element  - Data element type (FP16, BF16, FP32)
 *   TileCoord - Tile coordinate type (2D)
 *
 * Parameters:
 *   S            - Sequence length (number of tokens)
 *   H            - Hidden dimension
 *   workspace    - Shared memory workspace
 *   tokenIndices - Token indices and probabilities [bM]
 *   moeOutput    - Output buffer [S, H]
 *   tokens       - Input token data [bM, H]
 *   tileSize     - Actual number of tokens in this tile (may be < bM for last tile)
 *   tileCoord    - 2D tile coordinate
 */
template <
    int bM,
    int bN,
    int Arch,
    int threads,
    CombineMode c,
    tile::arrangement cArr = tile::arrangement::row_major,
    typename Element,
    typename TileCoord
>
__device__ __forceinline__
void combine(const int& S, const int& H,
             void* __restrict__ const& workspace,
             const TPS* __restrict__ const& tokenIndices, // [bM]
             Element* __restrict__ const& moeOutput,      // [S, H] in local HBM
             const Element* __restrict__ const& tokens,   // [bM, H] in local HBM
             const uint& tileSize, const TileCoord& tileCoord) {

    // Static assertions for template requirements
    static_assert(TileCoord::rank_v == 2, "TileCoord must be 2D");
    static_assert(cArr == tile::arrangement::row_major, "Only row-major output supported");

    // ========================================================================
    // Shared memory for token indices
    // ========================================================================
    __shared__ TPS stIds[bM];

    Element* sC = static_cast<Element*>(workspace);

    // ========================================================================
    // Get global tile pointer for input tokens
    // ========================================================================
    const int tileColIdx = hip_tensor::get<1>(tileCoord);
    const Element* gC = tile_hip::getTilePtrConst<bM, bN, cArr>(tokens, bM, H, 0, tileColIdx);

    // ========================================================================
    // Load token indices to shared memory
    // ========================================================================
    for (int i = threadIdx.x; i < bM; i += threads) {
        stIds[i] = tokenIndices[i];
    }

    // ========================================================================
    // Copy processed tile from global memory to shared memory
    // The source matrix has stride H (full hidden dim), not bN (tile width).
    // ========================================================================
    constexpr int tileElements = bM * bN;
    for (int i = threadIdx.x; i < tileElements; i += threads) {
        const int row = i / bN;
        const int col = i % bN;
        sC[i] = gC[row * H + col];
    }
    __syncthreads();

    // ========================================================================
    // Single mode: Direct copy to output (top-k = 1)
    // ========================================================================
    if constexpr (c == CombineMode::single) {
        using VTD = VectorTypeDescriptor<Element, ElementAlignment<Element, bN>>;
        using VT = typename VTD::VectorType;
        constexpr auto vw = VTD::VectorWidth::value;
        constexpr auto vbN = bN / vw;
        constexpr auto nElems = vbN * bM;
        const auto actualElems = tileSize * vbN;
        constexpr auto elemsPerThread = nElems / threads;

        // Row major output
        const auto vH = H / vw;
        VT* vMoeOutput = reinterpret_cast<VT*>(moeOutput);
        const VT* vsC = reinterpret_cast<const VT*>(workspace);

        // Optimized path when tile is full
        if (tileSize == bM) {
            for (int i = 0; i < elemsPerThread; ++i) {
                const auto idx = threadIdx.x + i * threads;
                const auto rowIdx = idx / vbN;
                const auto colIdx = idx % vbN;
                const auto tokenIdx = stIds[rowIdx].tokenIdx;

                // Calculate output position: row-major layout [S, vH]
                const auto outIdx = tokenIdx * vH + (tileColIdx * vbN) + colIdx;
                vMoeOutput[outIdx] = vsC[rowIdx * vbN + colIdx];
            }
        }
        else {
            // Partial tile: bounds checking needed
            for (int i = 0; i < elemsPerThread; ++i) {
                const auto idx = threadIdx.x + i * threads;
                const auto rowIdx = idx / vbN;
                const auto colIdx = idx % vbN;
                const auto tokenIdx = stIds[rowIdx].tokenIdx;

                if (idx < actualElems) {
                    const auto outIdx = tokenIdx * vH + (tileColIdx * vbN) + colIdx;
                    vMoeOutput[outIdx] = vsC[rowIdx * vbN + colIdx];
                }
            }
        }

        // Handle residue elements
        constexpr auto residue = nElems - (elemsPerThread * threads);
        if constexpr (residue > 0) {
            if (threadIdx.x < residue) {
                const auto idx = threadIdx.x + elemsPerThread * threads;
                const auto rowIdx = idx / vbN;
                const auto colIdx = idx % vbN;
                const auto tokenIdx = stIds[rowIdx].tokenIdx;

                if (idx < actualElems) {
                    const auto outIdx = tokenIdx * vH + (tileColIdx * vbN) + colIdx;
                    vMoeOutput[outIdx] = vsC[rowIdx * vbN + colIdx];
                }
            }
        }
    }
    // ========================================================================
    // Plural mode: Atomic reduction to output (top-k > 1)
    // ========================================================================
    else {
        // Promote to fp16x2 or bf16x2 if possible for efficient atomic ops
        using RAD = RedAddType<Element, ElementAlignment<Element, bN>>;
        using RAT = typename RAD::Type;
        constexpr int bNp = bN / RAD::Width::value;
        static_assert(RAD::Width::value == 1 || RAD::Width::value == 2);

        constexpr int maxVectorWidth = ElementAlignment<RAT, bNp> / sizeof(RAT);
        using RedAddOp = RedAdd<RedArch<Arch>, RAT, maxVectorWidth>;
        using RVD = VectorTypeDescriptor<RAT, RedAddOp::VectorWidth::value * sizeof(RAT)>;
        using RV = typename RVD::VectorType;

        constexpr int totalVecWidth = RVD::VectorWidth::value * RAD::Width::value;
        constexpr auto rbN = bN / totalVecWidth;

        const RV* vsC = reinterpret_cast<const RV*>(workspace);

        // Row major output
        const auto vHo = H / RAD::Width::value;
        RAT* mC = reinterpret_cast<RAT*>(moeOutput);

        constexpr auto totalElems = bM * rbN;
        const auto actualElems = tileSize * rbN;
        constexpr auto redElemsPerThread = totalElems / threads;
        constexpr int packWidth = RVD::VectorWidth::value;

        // Full tile path
        if (tileSize == bM) {
            for (int i = 0; i < redElemsPerThread; ++i) {
                const auto idx = threadIdx.x + i * threads;
                const auto rowIdx = idx / rbN;
                const auto colIdx = idx % rbN;
                const auto indexAndScale = stIds[rowIdx];
                auto tokenValue = vsC[rowIdx * rbN + colIdx];
                const auto tokIdx = indexAndScale.tokenIdx;

                // Scale by probability
                if constexpr (RAD::Width::value == 2) {
                    constexpr Converter<float2, RAT> loadOp{};
                    constexpr Converter<RAT, float2> storeOp{};
                    // fp16x2 or bf16x2
                    const auto scale2 = float2{indexAndScale.probability, indexAndScale.probability};
                    for (int j = 0; j < packWidth; ++j) {
                        tokenValue[j] = storeOp(float2Mul(loadOp(tokenValue[j]), scale2));
                    }
                }
                else {
                    constexpr Converter<float, RAT> loadOp{};
                    constexpr Converter<RAT, float> storeOp{};
                    for (int j = 0; j < packWidth; ++j) {
                        tokenValue[j] = storeOp(loadOp(tokenValue[j]) * indexAndScale.probability);
                    }
                }

                // Calculate output pointer
                // Account for the fact that the type of the tile is either Element or RAT.
                // Since we read 'packWidth' per iteration we need to advance the colIdx by that much.
                const auto outOffset = tokIdx * vHo + (tileColIdx * bNp) + colIdx * packWidth;
                RAT* tCp = mC + outOffset;

                // Atomic reduction
                constexpr RedAddOp op{};
                op(tCp, tokenValue);
            }
        }
        else {
            // Partial tile with bounds checking
            for (int i = 0; i < redElemsPerThread; ++i) {
                const auto idx = threadIdx.x + i * threads;
                if (idx < actualElems) {
                    const auto rowIdx = idx / rbN;
                    const auto colIdx = idx % rbN;
                    const auto indexAndScale = stIds[rowIdx];
                    auto tokenValue = vsC[rowIdx * rbN + colIdx];
                    const auto tokIdx = indexAndScale.tokenIdx;

                    if constexpr (RAD::Width::value == 2) {
                        constexpr Converter<float2, RAT> loadOp{};
                        constexpr Converter<RAT, float2> storeOp{};
                        const auto scale2 = float2{indexAndScale.probability, indexAndScale.probability};
                        for (int j = 0; j < packWidth; ++j) {
                            tokenValue[j] = storeOp(float2Mul(loadOp(tokenValue[j]), scale2));
                        }
                    }
                    else {
                        constexpr Converter<float, RAT> loadOp{};
                        constexpr Converter<RAT, float> storeOp{};
                        for (int j = 0; j < packWidth; ++j) {
                            tokenValue[j] = storeOp(loadOp(tokenValue[j]) * indexAndScale.probability);
                        }
                    }

                    const auto outOffset = tokIdx * vHo + (tileColIdx * bNp) + colIdx * packWidth;
                    RAT* tCp = mC + outOffset;
                    constexpr RedAddOp op{};
                    op(tCp, tokenValue);
                }
            }
        }

        // Handle residue
        constexpr auto residue = totalElems - (redElemsPerThread * threads);
        if constexpr (residue > 0) {
            if (threadIdx.x < residue) {
                const auto idx = threadIdx.x + redElemsPerThread * threads;
                const auto rowIdx = idx / rbN;
                const auto colIdx = idx % rbN;
                const auto indexAndScale = stIds[rowIdx];
                auto tokenValue = vsC[rowIdx * rbN + colIdx];
                const auto tokIdx = indexAndScale.tokenIdx;

                if (idx < actualElems) {
                    if constexpr (RAD::Width::value == 2) {
                        constexpr Converter<float2, RAT> loadOp{};
                        constexpr Converter<RAT, float2> storeOp{};
                        const auto scale2 = float2{indexAndScale.probability, indexAndScale.probability};
                        for (int j = 0; j < packWidth; ++j) {
                            tokenValue[j] = storeOp(float2Mul(loadOp(tokenValue[j]), scale2));
                        }
                    }
                    else {
                        constexpr Converter<float, RAT> loadOp{};
                        constexpr Converter<RAT, float> storeOp{};
                        for (int j = 0; j < packWidth; ++j) {
                            tokenValue[j] = storeOp(loadOp(tokenValue[j]) * indexAndScale.probability);
                        }
                    }

                    const auto outOffset = tokIdx * vHo + (tileColIdx * bNp) + colIdx * packWidth;
                    RAT* tCp = mC + outOffset;
                    constexpr RedAddOp op{};
                    op(tCp, tokenValue);
                }
            }
        }
    }
}

// ============================================================================
// Shared Memory Size Calculation
// ============================================================================

template<int bM, int bN, typename Element>
constexpr int combineSharedSize = bM * bN * sizeof(Element);

} // namespace flashmoe

#endif // FLASHMOE_HIP_COMBINE_CUH
