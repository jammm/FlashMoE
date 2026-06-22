//
// FlashMoE HIP Port - Optimized GEMM Kernel for MI450/gfx1250
// Targeting high TFLOPS utilization with rocWMMA
//
// Key optimizations:
// 1. Vectorized global memory loads (128-bit)
// 2. LDS double-buffering with compute/load overlap
// 3. Swizzled LDS layout to avoid bank conflicts
// 4. Multiple WMMA tiles per warp for better utilization
// 5. Register blocking for reduced VGPR pressure
//
// Copyright (c) 2025, Osayamen Jonathan Aimuyo
// All rights reserved.
//

#ifndef FLASHMOE_HIP_GEMM_OPTIMIZED_CUH
#define FLASHMOE_HIP_GEMM_OPTIMIZED_CUH

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>

#include "constants.hpp"

namespace flashmoe::gemm {

// ============================================================================
// Architecture Constants
// ============================================================================

constexpr int WAVEFRONT_SIZE = flashmoe::WARP_SIZE;
constexpr int VECTOR_WIDTH = 8;     // 8 x FP16 = 128-bit vector loads

// ============================================================================
// Vectorized Memory Access Types
// ============================================================================

// 128-bit vector type for efficient memory access
using float4_t = float4;  // 128 bits
using half8_t = uint4;    // 8 × FP16 = 128 bits

// Helper for vectorized loads
__device__ __forceinline__
void load_half8(half8_t& dst, const __half* src) {
    dst = *reinterpret_cast<const uint4*>(src);
}

__device__ __forceinline__
void store_half8(__half* dst, const half8_t& src) {
    *reinterpret_cast<uint4*>(dst) = src;
}

// ============================================================================
// LDS Layout with Bank Conflict Avoidance
// ============================================================================

// Add padding to avoid bank conflicts (32 banks, 4 bytes per bank)
// For FP16: 32 banks × 2 bytes = 64 bytes = 32 FP16 elements
// Pad by 8 elements (16 bytes = 4 banks) to shift pattern
template<int Rows, int Cols>
struct LDSLayoutPadded {
    static constexpr int PADDING = 8;  // Pad columns by 8 FP16
    static constexpr int STRIDE = Cols + PADDING;
    static constexpr int SIZE = Rows * STRIDE;

    __device__ __forceinline__
    static int offset(int row, int col) {
        return row * STRIDE + col;
    }
};

// ============================================================================
// Swizzled Thread Mapping for Coalesced Access
// ============================================================================

// Map thread ID to (warp_id, lane_id) for tile access
struct ThreadMapping {
    int warp_id;
    int lane_id;
    int lane_row;   // Lane's row within warp tile
    int lane_col;   // Lane's column within warp tile

    __device__ ThreadMapping(int tid) {
        warp_id = tid / WAVEFRONT_SIZE;
        lane_id = tid % WAVEFRONT_SIZE;
        // 8 rows × 8 columns per warp for loading
        lane_row = lane_id / 8;
        lane_col = lane_id % 8;
    }
};

// ============================================================================
// Optimized GEMM Kernel Configuration
// ============================================================================

// Configuration: 128x128 block tile with 16x16x16 WMMA
struct GemmConfig_128x128 {
    // Block tile dimensions
    static constexpr int BM = 128;
    static constexpr int BN = 128;
    static constexpr int BK = 32;

    // WMMA tile dimensions
    static constexpr int WMMA_M = 16;
    static constexpr int WMMA_N = 16;
    static constexpr int WMMA_K = 16;

    // Number of WMMA tiles per block tile
    static constexpr int TILES_M = BM / WMMA_M;  // 8
    static constexpr int TILES_N = BN / WMMA_N;  // 8
    static constexpr int TILES_K = BK / WMMA_K;  // 2

    // Thread configuration
    static constexpr int THREADS = 256;
    static constexpr int WARPS = THREADS / WAVEFRONT_SIZE;

    // WMMA tiles per warp.
    static constexpr int TILES_PER_WARP = (TILES_M * TILES_N) / WARPS;

    // LDS layout with padding
    using LDS_A = LDSLayoutPadded<BM, BK>;
    using LDS_B = LDSLayoutPadded<BK, BN>;

    // LDS size
    static constexpr int LDS_A_SIZE = LDS_A::SIZE;
    static constexpr int LDS_B_SIZE = LDS_B::SIZE;
    static constexpr int LDS_TOTAL = (LDS_A_SIZE + LDS_B_SIZE) * 2;  // Double buffer

    // Compute how many rows each thread loads
    static constexpr int A_ROWS_PER_THREAD = (BM * BK) / (THREADS * VECTOR_WIDTH);
    static constexpr int B_ROWS_PER_THREAD = (BK * BN) / (THREADS * VECTOR_WIDTH);
};

// ============================================================================
// Fragment Types
// ============================================================================

template<int WM, int WN, int WK>
struct WmmaFragments {
    using FragA = rocwmma::fragment<rocwmma::matrix_a, WM, WN, WK, __half, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, WM, WN, WK, __half, rocwmma::col_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, WM, WN, WK, float>;
};

// ============================================================================
// Optimized GEMM Kernel
// ============================================================================

template<typename Config = GemmConfig_128x128>
__global__ __launch_bounds__(Config::THREADS)
void gemm_optimized_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K,
    float alpha
) {
    using Frags = WmmaFragments<Config::WMMA_M, Config::WMMA_N, Config::WMMA_K>;
    using FragA = typename Frags::FragA;
    using FragB = typename Frags::FragB;
    using FragC = typename Frags::FragC;

    // Block tile coordinates
    const int numTilesN = (N + Config::BN - 1) / Config::BN;
    const int blockTileIdx = blockIdx.x;
    const int blockTileRow = blockTileIdx / numTilesN;
    const int blockTileCol = blockTileIdx % numTilesN;

    // Check bounds
    if (blockTileRow * Config::BM >= M || blockTileCol * Config::BN >= N) return;

    // Thread info
    const int tid = threadIdx.x;
    const ThreadMapping tm(tid);

    // Double-buffered LDS
    extern __shared__ char smem[];
    __half* sA = reinterpret_cast<__half*>(smem);
    __half* sB = sA + Config::LDS_A_SIZE * 2;

    // Global memory pointers
    const int globalRowA = blockTileRow * Config::BM;
    const int globalColB = blockTileCol * Config::BN;

    // Initialize accumulators for this warp's tiles
    FragC accumulators[Config::TILES_PER_WARP];
    #pragma unroll
    for (int i = 0; i < Config::TILES_PER_WARP; ++i) {
        rocwmma::fill_fragment(accumulators[i], 0.0f);
    }

    // Number of K iterations
    const int numKBlocks = (K + Config::BK - 1) / Config::BK;

    // === Load first K-block into buffer 0 ===
    int writeBuffer = 0;
    {
        const int kOffset = 0;

        // Load A tile cooperatively
        // Each thread loads VECTOR_WIDTH elements per iteration
        for (int loadIdx = tid; loadIdx < (Config::BM * Config::BK) / VECTOR_WIDTH; loadIdx += Config::THREADS) {
            int flatIdx = loadIdx * VECTOR_WIDTH;
            int row = flatIdx / Config::BK;
            int col = flatIdx % Config::BK;
            int globalRow = globalRowA + row;
            int globalCol = kOffset + col;

            __half* dstA = sA + writeBuffer * Config::LDS_A_SIZE + Config::LDS_A::offset(row, col);

            if (globalRow < M && globalCol + VECTOR_WIDTH <= K) {
                // Vectorized load
                half8_t vec;
                load_half8(vec, &A[globalRow * K + globalCol]);
                store_half8(dstA, vec);
            } else {
                // Scalar fallback with bounds checking
                #pragma unroll
                for (int v = 0; v < VECTOR_WIDTH; ++v) {
                    if (globalRow < M && globalCol + v < K) {
                        dstA[v] = A[globalRow * K + globalCol + v];
                    } else {
                        dstA[v] = __float2half(0.0f);
                    }
                }
            }
        }

        // Load B tile cooperatively
        // B is stored transposed (column-major) for WMMA
        for (int loadIdx = tid; loadIdx < (Config::BK * Config::BN) / VECTOR_WIDTH; loadIdx += Config::THREADS) {
            int flatIdx = loadIdx * VECTOR_WIDTH;
            int kRow = flatIdx / Config::BN;
            int nCol = flatIdx % Config::BN;
            int globalRow = kOffset + kRow;
            int globalCol = globalColB + nCol;

            // Transpose during store: B is K×N row-major, store as N×K col-major in LDS
            // For col-major storage: offset = col * stride + row

            if (globalRow < K && globalCol + VECTOR_WIDTH <= N) {
                // Load row-major, store transposed
                #pragma unroll
                for (int v = 0; v < VECTOR_WIDTH; ++v) {
                    int srcIdx = globalRow * N + globalCol + v;
                    int dstIdx = writeBuffer * Config::LDS_B_SIZE + Config::LDS_B::offset(kRow, nCol + v);
                    sB[dstIdx] = B[srcIdx];
                }
            } else {
                #pragma unroll
                for (int v = 0; v < VECTOR_WIDTH; ++v) {
                    int dstIdx = writeBuffer * Config::LDS_B_SIZE + Config::LDS_B::offset(kRow, nCol + v);
                    if (globalRow < K && globalCol + v < N) {
                        sB[dstIdx] = B[globalRow * N + globalCol + v];
                    } else {
                        sB[dstIdx] = __float2half(0.0f);
                    }
                }
            }
        }
    }
    __syncthreads();

    // === Main K-loop with double buffering ===
    for (int kBlock = 0; kBlock < numKBlocks; ++kBlock) {
        int readBuffer = kBlock % 2;
        int nextWriteBuffer = (kBlock + 1) % 2;

        // Start loading next K-block while computing
        if (kBlock + 1 < numKBlocks) {
            const int kOffset = (kBlock + 1) * Config::BK;

            // Load A tile
            for (int loadIdx = tid; loadIdx < (Config::BM * Config::BK) / VECTOR_WIDTH; loadIdx += Config::THREADS) {
                int flatIdx = loadIdx * VECTOR_WIDTH;
                int row = flatIdx / Config::BK;
                int col = flatIdx % Config::BK;
                int globalRow = globalRowA + row;
                int globalCol = kOffset + col;

                __half* dstA = sA + nextWriteBuffer * Config::LDS_A_SIZE + Config::LDS_A::offset(row, col);

                if (globalRow < M && globalCol + VECTOR_WIDTH <= K) {
                    half8_t vec;
                    load_half8(vec, &A[globalRow * K + globalCol]);
                    store_half8(dstA, vec);
                } else {
                    #pragma unroll
                    for (int v = 0; v < VECTOR_WIDTH; ++v) {
                        if (globalRow < M && globalCol + v < K) {
                            dstA[v] = A[globalRow * K + globalCol + v];
                        } else {
                            dstA[v] = __float2half(0.0f);
                        }
                    }
                }
            }

            // Load B tile
            for (int loadIdx = tid; loadIdx < (Config::BK * Config::BN) / VECTOR_WIDTH; loadIdx += Config::THREADS) {
                int flatIdx = loadIdx * VECTOR_WIDTH;
                int kRow = flatIdx / Config::BN;
                int nCol = flatIdx % Config::BN;
                int globalRow = kOffset + kRow;
                int globalCol = globalColB + nCol;

                if (globalRow < K && globalCol + VECTOR_WIDTH <= N) {
                    #pragma unroll
                    for (int v = 0; v < VECTOR_WIDTH; ++v) {
                        int srcIdx = globalRow * N + globalCol + v;
                        int dstIdx = nextWriteBuffer * Config::LDS_B_SIZE + Config::LDS_B::offset(kRow, nCol + v);
                        sB[dstIdx] = B[srcIdx];
                    }
                } else {
                    #pragma unroll
                    for (int v = 0; v < VECTOR_WIDTH; ++v) {
                        int dstIdx = nextWriteBuffer * Config::LDS_B_SIZE + Config::LDS_B::offset(kRow, nCol + v);
                        if (globalRow < K && globalCol + v < N) {
                            sB[dstIdx] = B[globalRow * N + globalCol + v];
                        } else {
                            sB[dstIdx] = __float2half(0.0f);
                        }
                    }
                }
            }
        }

        // === Compute WMMA tiles ===
        // Each warp processes its assigned tiles
        const int warpId = tm.warp_id;
        const int totalWmmaTiles = Config::TILES_M * Config::TILES_N;

        for (int tileOfs = warpId; tileOfs < totalWmmaTiles; tileOfs += Config::WARPS) {
            int wmmaTileM = tileOfs / Config::TILES_N;
            int wmmaTileN = tileOfs % Config::TILES_N;
            int localIdx = tileOfs / Config::WARPS;

            FragA fragA;
            FragB fragB;

            // Loop over K tiles within this K-block
            #pragma unroll
            for (int wk = 0; wk < Config::TILES_K; ++wk) {
                // Load A fragment from LDS
                const __half* aPtr = sA + readBuffer * Config::LDS_A_SIZE +
                                     Config::LDS_A::offset(wmmaTileM * Config::WMMA_M, wk * Config::WMMA_K);
                rocwmma::load_matrix_sync(fragA, aPtr, Config::LDS_A::STRIDE);

                // Load B fragment from LDS (B stored column-major)
                // For col-major B: We need element at (k, n)
                // After transpose in LDS: row=k, col=n
                const __half* bPtr = sB + readBuffer * Config::LDS_B_SIZE +
                                     Config::LDS_B::offset(wk * Config::WMMA_K, wmmaTileN * Config::WMMA_N);
                rocwmma::load_matrix_sync(fragB, bPtr, Config::LDS_B::STRIDE);

                // Matrix multiply-accumulate
                rocwmma::mma_sync(accumulators[localIdx], fragA, fragB, accumulators[localIdx]);
            }
        }

        __syncthreads();
    }

    // === Store results ===
    const int warpId = tm.warp_id;
    const int totalWmmaTiles = Config::TILES_M * Config::TILES_N;

    for (int tileOfs = warpId; tileOfs < totalWmmaTiles; tileOfs += Config::WARPS) {
        int wmmaTileM = tileOfs / Config::TILES_N;
        int wmmaTileN = tileOfs % Config::TILES_N;
        int localIdx = tileOfs / Config::WARPS;

        int cRow = globalRowA + wmmaTileM * Config::WMMA_M;
        int cCol = globalColB + wmmaTileN * Config::WMMA_N;

        if (cRow < M && cCol < N) {
            // Scale by alpha
            #pragma unroll
            for (int i = 0; i < accumulators[localIdx].num_elements; ++i) {
                accumulators[localIdx].x[i] *= alpha;
            }

            // Convert to FP16
            rocwmma::fragment<rocwmma::accumulator, Config::WMMA_M, Config::WMMA_N, Config::WMMA_K, __half> fragC_out;
            #pragma unroll
            for (int i = 0; i < fragC_out.num_elements; ++i) {
                fragC_out.x[i] = __float2half(accumulators[localIdx].x[i]);
            }

            // Store to global memory
            __half* cPtr = C + cRow * N + cCol;
            rocwmma::store_matrix_sync(cPtr, fragC_out, N, rocwmma::mem_row_major);
        }
    }
}

// ============================================================================
// Launch Helper
// ============================================================================

template<typename Config = GemmConfig_128x128>
hipError_t launch_gemm_optimized(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K,
    float alpha,
    hipStream_t stream = 0
) {
    int numTilesM = (M + Config::BM - 1) / Config::BM;
    int numTilesN = (N + Config::BN - 1) / Config::BN;
    int numBlocks = numTilesM * numTilesN;

    dim3 grid(numBlocks);
    dim3 block(Config::THREADS);

    // Dynamic shared memory size
    size_t shmemSize = Config::LDS_TOTAL * sizeof(__half);

    hipLaunchKernelGGL(
        (gemm_optimized_kernel<Config>),
        grid, block, shmemSize, stream,
        A, B, C, M, N, K, alpha
    );

    return hipGetLastError();
}

} // namespace flashmoe::gemm

#endif // FLASHMOE_HIP_GEMM_OPTIMIZED_CUH
