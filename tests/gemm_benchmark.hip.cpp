//
// FlashMoE HIP Port - Comprehensive GEMM Benchmark
// Compares hipBLAS, hipBLASLt, and custom rocWMMA kernels
// Target: AMD Instinct MI450 (gfx1250)
//
// Copyright (c) 2025, Osayamen Jonathan Aimuyo
// All rights reserved.
//

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>

// hipBLAS and hipBLASLt for baseline comparisons
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>

// ============================================================================
// Error Checking Macros
// ============================================================================

#define CHECK_HIP(call)                                                        \
    do {                                                                       \
        hipError_t err = call;                                                 \
        if (err != hipSuccess) {                                               \
            std::cerr << "HIP Error: " << hipGetErrorString(err)               \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CHECK_HIPBLAS(call)                                                    \
    do {                                                                       \
        hipblasStatus_t status = call;                                         \
        if (status != HIPBLAS_STATUS_SUCCESS) {                                \
            std::cerr << "hipBLAS Error: " << status                           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CHECK_HIPBLASLT(call)                                                  \
    do {                                                                       \
        hipblasStatus_t status = call;                                         \
        if (status != HIPBLAS_STATUS_SUCCESS) {                                \
            std::cerr << "hipBLASLt Error: " << status                         \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ============================================================================
// Constants
// ============================================================================

constexpr int WAVEFRONT_SIZE = 32;

// ============================================================================
// Timer Utility
// ============================================================================

class GpuTimer {
public:
    GpuTimer() {
        CHECK_HIP(hipEventCreate(&start_));
        CHECK_HIP(hipEventCreate(&stop_));
    }

    ~GpuTimer() {
        (void)hipEventDestroy(start_);
        (void)hipEventDestroy(stop_);
    }

    void start(hipStream_t stream = 0) {
        CHECK_HIP(hipEventRecord(start_, stream));
    }

    void stop(hipStream_t stream = 0) {
        CHECK_HIP(hipEventRecord(stop_, stream));
        CHECK_HIP(hipEventSynchronize(stop_));
    }

    float elapsed_ms() const {
        float ms;
        CHECK_HIP(hipEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    hipEvent_t start_, stop_;
};

// ============================================================================
// Benchmark Result Structure
// ============================================================================

struct BenchmarkResult {
    const char* name;
    int M, N, K;
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    double tflops;
    bool valid;
};

// ============================================================================
// Custom rocWMMA Kernel Configurations
// ============================================================================

// Configuration 1: 16x16x16 WMMA tiles (current implementation)
template<int BM, int BN, int BK, int THREADS>
__global__ void gemm_wmma_16x16x16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    using FragA = rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, rocwmma::col_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

    constexpr int TILES_M = BM / WMMA_M;
    constexpr int TILES_N = BN / WMMA_N;
    constexpr int TILES_K = BK / WMMA_K;

    const int numTilesN = (N + BN - 1) / BN;
    const int tileIdx = blockIdx.x;
    const int tileRow = tileIdx / numTilesN;
    const int tileCol = tileIdx % numTilesN;

    if (tileRow * BM >= M || tileCol * BN >= N) return;

    const int tid = threadIdx.x;
    const int waveId = tid / WAVEFRONT_SIZE;
    const int numWaves = THREADS / WAVEFRONT_SIZE;

    __shared__ __half sA[BM * BK];
    __shared__ __half sB[BK * BN];

    constexpr int WMMA_TILES = TILES_M * TILES_N;

    FragC accumulators[(WMMA_TILES + numWaves - 1) / numWaves];
    const int tilesPerWave = (WMMA_TILES + numWaves - 1) / numWaves;

    #pragma unroll
    for (int t = 0; t < tilesPerWave; ++t) {
        rocwmma::fill_fragment(accumulators[t], 0.0f);
    }

    const int numKTiles = (K + BK - 1) / BK;
    const int aRowStart = tileRow * BM;
    const int bColStart = tileCol * BN;

    for (int kTile = 0; kTile < numKTiles; ++kTile) {
        const int kOffset = kTile * BK;

        // Cooperative load A (vectorized 4x half = 64-bit load)
        for (int i = tid; i < BM * BK; i += THREADS) {
            int row = i / BK;
            int col = i % BK;
            int globalRow = aRowStart + row;
            int globalCol = kOffset + col;

            if (globalRow < M && globalCol < K) {
                sA[i] = A[globalRow * K + globalCol];
            } else {
                sA[i] = __float2half(0.0f);
            }
        }

        // Cooperative load B (transpose to col-major in LDS)
        for (int i = tid; i < BK * BN; i += THREADS) {
            int row = i / BN;
            int col = i % BN;
            int globalRow = kOffset + row;
            int globalCol = bColStart + col;

            if (globalRow < K && globalCol < N) {
                sB[col * BK + row] = B[globalRow * N + globalCol];
            } else {
                sB[col * BK + row] = __float2half(0.0f);
            }
        }

        __syncthreads();

        for (int tileOfs = waveId; tileOfs < WMMA_TILES; tileOfs += numWaves) {
            int wmmaTileM = tileOfs / TILES_N;
            int wmmaTileN = tileOfs % TILES_N;
            int localIdx = tileOfs / numWaves;

            FragA fragA;
            FragB fragB;

            #pragma unroll
            for (int wk = 0; wk < TILES_K; ++wk) {
                const __half* aPtr = sA + wmmaTileM * WMMA_M * BK + wk * WMMA_K;
                rocwmma::load_matrix_sync(fragA, aPtr, BK);

                const __half* bPtr = sB + wmmaTileN * WMMA_N * BK + wk * WMMA_K;
                rocwmma::load_matrix_sync(fragB, bPtr, BK);

                rocwmma::mma_sync(accumulators[localIdx], fragA, fragB, accumulators[localIdx]);
            }
        }

        __syncthreads();
    }

    // Store results with alpha/beta scaling
    for (int tileOfs = waveId; tileOfs < WMMA_TILES; tileOfs += numWaves) {
        int wmmaTileM = tileOfs / TILES_N;
        int wmmaTileN = tileOfs % TILES_N;
        int localIdx = tileOfs / numWaves;

        int cRow = aRowStart + wmmaTileM * WMMA_M;
        int cCol = bColStart + wmmaTileN * WMMA_N;

        if (cRow < M && cCol < N) {
            // Convert accumulator to FP16 and store
            __half* cPtr = C + cRow * N + cCol;

            // Scale by alpha and store (ignoring beta for now since we use zero init)
            rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> fragC_out;

            // Scale accumulator by alpha
            #pragma unroll
            for (int i = 0; i < accumulators[localIdx].num_elements; ++i) {
                accumulators[localIdx].x[i] *= alpha;
            }

            // Convert to FP16
            #pragma unroll
            for (int i = 0; i < fragC_out.num_elements; ++i) {
                fragC_out.x[i] = __float2half(accumulators[localIdx].x[i]);
            }

            rocwmma::store_matrix_sync(cPtr, fragC_out, N, rocwmma::mem_row_major);
        }
    }
}

// Configuration 2: 32x32x8 WMMA tiles (larger tiles, potentially better for big matrices)
template<int BM, int BN, int BK, int THREADS>
__global__ void gemm_wmma_32x32x8(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
) {
    constexpr int WMMA_M = 32;
    constexpr int WMMA_N = 32;
    constexpr int WMMA_K = 8;

    using FragA = rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, rocwmma::col_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

    constexpr int TILES_M = BM / WMMA_M;
    constexpr int TILES_N = BN / WMMA_N;
    constexpr int TILES_K = BK / WMMA_K;

    const int numTilesN = (N + BN - 1) / BN;
    const int tileIdx = blockIdx.x;
    const int tileRow = tileIdx / numTilesN;
    const int tileCol = tileIdx % numTilesN;

    if (tileRow * BM >= M || tileCol * BN >= N) return;

    const int tid = threadIdx.x;
    const int waveId = tid / WAVEFRONT_SIZE;
    const int numWaves = THREADS / WAVEFRONT_SIZE;

    __shared__ __half sA[BM * BK];
    __shared__ __half sB[BK * BN];

    constexpr int WMMA_TILES = TILES_M * TILES_N;

    FragC accumulators[(WMMA_TILES + numWaves - 1) / numWaves];
    const int tilesPerWave = (WMMA_TILES + numWaves - 1) / numWaves;

    #pragma unroll
    for (int t = 0; t < tilesPerWave; ++t) {
        rocwmma::fill_fragment(accumulators[t], 0.0f);
    }

    const int numKTiles = (K + BK - 1) / BK;
    const int aRowStart = tileRow * BM;
    const int bColStart = tileCol * BN;

    for (int kTile = 0; kTile < numKTiles; ++kTile) {
        const int kOffset = kTile * BK;

        // Vectorized load using half2
        for (int i = tid * 2; i < BM * BK; i += THREADS * 2) {
            int row = i / BK;
            int col = i % BK;
            int globalRow = aRowStart + row;
            int globalCol = kOffset + col;

            if (globalRow < M && globalCol + 1 < K) {
                *reinterpret_cast<half2*>(&sA[i]) =
                    *reinterpret_cast<const half2*>(&A[globalRow * K + globalCol]);
            } else if (globalRow < M && globalCol < K) {
                sA[i] = A[globalRow * K + globalCol];
                sA[i + 1] = __float2half(0.0f);
            } else {
                sA[i] = __float2half(0.0f);
                sA[i + 1] = __float2half(0.0f);
            }
        }

        for (int i = tid * 2; i < BK * BN; i += THREADS * 2) {
            int row = i / BN;
            int col = i % BN;
            int globalRow = kOffset + row;
            int globalCol = bColStart + col;

            if (globalRow < K && globalCol + 1 < N) {
                // For transposed store, we need to be careful
                sB[(col) * BK + row] = B[globalRow * N + globalCol];
                sB[(col + 1) * BK + row] = B[globalRow * N + globalCol + 1];
            } else if (globalRow < K && globalCol < N) {
                sB[col * BK + row] = B[globalRow * N + globalCol];
            }
        }

        __syncthreads();

        for (int tileOfs = waveId; tileOfs < WMMA_TILES; tileOfs += numWaves) {
            int wmmaTileM = tileOfs / TILES_N;
            int wmmaTileN = tileOfs % TILES_N;
            int localIdx = tileOfs / numWaves;

            FragA fragA;
            FragB fragB;

            #pragma unroll
            for (int wk = 0; wk < TILES_K; ++wk) {
                const __half* aPtr = sA + wmmaTileM * WMMA_M * BK + wk * WMMA_K;
                rocwmma::load_matrix_sync(fragA, aPtr, BK);

                const __half* bPtr = sB + wmmaTileN * WMMA_N * BK + wk * WMMA_K;
                rocwmma::load_matrix_sync(fragB, bPtr, BK);

                rocwmma::mma_sync(accumulators[localIdx], fragA, fragB, accumulators[localIdx]);
            }
        }

        __syncthreads();
    }

    // Store results
    for (int tileOfs = waveId; tileOfs < WMMA_TILES; tileOfs += numWaves) {
        int wmmaTileM = tileOfs / TILES_N;
        int wmmaTileN = tileOfs % TILES_N;
        int localIdx = tileOfs / numWaves;

        int cRow = aRowStart + wmmaTileM * WMMA_M;
        int cCol = bColStart + wmmaTileN * WMMA_N;

        if (cRow < M && cCol < N) {
            __half* cPtr = C + cRow * N + cCol;

            rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> fragC_out;

            #pragma unroll
            for (int i = 0; i < accumulators[localIdx].num_elements; ++i) {
                accumulators[localIdx].x[i] *= alpha;
            }

            #pragma unroll
            for (int i = 0; i < fragC_out.num_elements; ++i) {
                fragC_out.x[i] = __float2half(accumulators[localIdx].x[i]);
            }

            rocwmma::store_matrix_sync(cPtr, fragC_out, N, rocwmma::mem_row_major);
        }
    }
}

// ============================================================================
// Configuration 3: Highly Optimized Kernel with Vectorized Loads
// ============================================================================

// Optimized configuration
struct OptConfig {
    static constexpr int BM = 128;
    static constexpr int BN = 128;
    static constexpr int BK = 32;
    static constexpr int WMMA_M = 16;
    static constexpr int WMMA_N = 16;
    static constexpr int WMMA_K = 16;
    static constexpr int TILES_M = BM / WMMA_M;
    static constexpr int TILES_N = BN / WMMA_N;
    static constexpr int TILES_K = BK / WMMA_K;
    static constexpr int THREADS = 256;
    static constexpr int WARPS = THREADS / WAVEFRONT_SIZE;
    static constexpr int PADDING = 8;
    static constexpr int A_STRIDE = BK + PADDING;
    static constexpr int B_STRIDE = BN + PADDING;
    static constexpr int LDS_A_SIZE = BM * A_STRIDE;
    static constexpr int LDS_B_SIZE = BK * B_STRIDE;
};

// Vectorized load helper
__device__ __forceinline__
void vectorized_load_4(__half* dst, const __half* src) {
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(src);  // 4 x FP16 = 64 bits
}

template<typename Config>
__global__ __launch_bounds__(Config::THREADS)
void gemm_highly_optimized(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
) {
    using FragA = rocwmma::fragment<rocwmma::matrix_a, Config::WMMA_M, Config::WMMA_N, Config::WMMA_K, __half, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, Config::WMMA_M, Config::WMMA_N, Config::WMMA_K, __half, rocwmma::col_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, Config::WMMA_M, Config::WMMA_N, Config::WMMA_K, float>;

    const int numTilesN = (N + Config::BN - 1) / Config::BN;
    const int tileIdx = blockIdx.x;
    const int tileRow = tileIdx / numTilesN;
    const int tileCol = tileIdx % numTilesN;

    if (tileRow * Config::BM >= M || tileCol * Config::BN >= N) return;

    const int tid = threadIdx.x;
    const int warpId = tid / WAVEFRONT_SIZE;

    // Double-buffered LDS with padding to avoid bank conflicts
    __shared__ __half sA[2][Config::LDS_A_SIZE];
    __shared__ __half sB[2][Config::LDS_B_SIZE];

    const int globalRowA = tileRow * Config::BM;
    const int globalColB = tileCol * Config::BN;

    // Initialize accumulators
    constexpr int TOTAL_TILES = Config::TILES_M * Config::TILES_N;
    constexpr int TILES_PER_WARP = (TOTAL_TILES + Config::WARPS - 1) / Config::WARPS;

    FragC accumulators[TILES_PER_WARP];
    #pragma unroll
    for (int i = 0; i < TILES_PER_WARP; ++i) {
        rocwmma::fill_fragment(accumulators[i], 0.0f);
    }

    const int numKBlocks = (K + Config::BK - 1) / Config::BK;

    // Lambda for loading A tile with vectorized access
    auto loadA = [&](int kOffset, int buffer) {
        // Each thread loads 4 elements at a time
        constexpr int ELEMENTS_PER_THREAD = (Config::BM * Config::BK) / Config::THREADS;
        constexpr int VECTOR_SIZE = 4;
        constexpr int LOADS_PER_THREAD = ELEMENTS_PER_THREAD / VECTOR_SIZE;

        #pragma unroll
        for (int load = 0; load < LOADS_PER_THREAD; ++load) {
            int loadIdx = tid * LOADS_PER_THREAD + load;
            int flatIdx = loadIdx * VECTOR_SIZE;
            int row = flatIdx / Config::BK;
            int col = flatIdx % Config::BK;
            int gRow = globalRowA + row;
            int gCol = kOffset + col;

            __half* dst = &sA[buffer][row * Config::A_STRIDE + col];

            if (gRow < M && gCol + VECTOR_SIZE <= K) {
                vectorized_load_4(dst, &A[gRow * K + gCol]);
            } else {
                #pragma unroll
                for (int v = 0; v < VECTOR_SIZE; ++v) {
                    dst[v] = (gRow < M && gCol + v < K) ? A[gRow * K + gCol + v] : __float2half(0.0f);
                }
            }
        }
    };

    // Lambda for loading B tile (with transpose)
    auto loadB = [&](int kOffset, int buffer) {
        constexpr int ELEMENTS_PER_THREAD = (Config::BK * Config::BN) / Config::THREADS;

        #pragma unroll
        for (int load = 0; load < ELEMENTS_PER_THREAD; ++load) {
            int loadIdx = tid * ELEMENTS_PER_THREAD + load;
            int kRow = loadIdx / Config::BN;
            int nCol = loadIdx % Config::BN;
            int gRow = kOffset + kRow;
            int gCol = globalColB + nCol;

            // Store transposed: B[k,n] -> sB[k][n] with padding
            __half val = (gRow < K && gCol < N) ? B[gRow * N + gCol] : __float2half(0.0f);
            sB[buffer][kRow * Config::B_STRIDE + nCol] = val;
        }
    };

    // Load first K-block
    loadA(0, 0);
    loadB(0, 0);
    __syncthreads();

    // Main K-loop
    for (int kBlock = 0; kBlock < numKBlocks; ++kBlock) {
        int readBuffer = kBlock % 2;
        int writeBuffer = (kBlock + 1) % 2;

        // Start loading next block
        if (kBlock + 1 < numKBlocks) {
            loadA((kBlock + 1) * Config::BK, writeBuffer);
            loadB((kBlock + 1) * Config::BK, writeBuffer);
        }

        // Compute WMMA tiles
        for (int tileOfs = warpId; tileOfs < TOTAL_TILES; tileOfs += Config::WARPS) {
            int wmmaTileM = tileOfs / Config::TILES_N;
            int wmmaTileN = tileOfs % Config::TILES_N;
            int localIdx = tileOfs / Config::WARPS;

            FragA fragA;
            FragB fragB;

            #pragma unroll
            for (int wk = 0; wk < Config::TILES_K; ++wk) {
                // Load A from LDS (row-major, stride = A_STRIDE)
                const __half* aPtr = &sA[readBuffer][wmmaTileM * Config::WMMA_M * Config::A_STRIDE + wk * Config::WMMA_K];
                rocwmma::load_matrix_sync(fragA, aPtr, Config::A_STRIDE);

                // Load B from LDS (col-major layout after transpose, stride = B_STRIDE)
                const __half* bPtr = &sB[readBuffer][wk * Config::WMMA_K * Config::B_STRIDE + wmmaTileN * Config::WMMA_N];
                rocwmma::load_matrix_sync(fragB, bPtr, Config::B_STRIDE);

                rocwmma::mma_sync(accumulators[localIdx], fragA, fragB, accumulators[localIdx]);
            }
        }

        __syncthreads();
    }

    // Store results
    for (int tileOfs = warpId; tileOfs < TOTAL_TILES; tileOfs += Config::WARPS) {
        int wmmaTileM = tileOfs / Config::TILES_N;
        int wmmaTileN = tileOfs % Config::TILES_N;
        int localIdx = tileOfs / Config::WARPS;

        int cRow = globalRowA + wmmaTileM * Config::WMMA_M;
        int cCol = globalColB + wmmaTileN * Config::WMMA_N;

        if (cRow < M && cCol < N) {
            #pragma unroll
            for (int i = 0; i < accumulators[localIdx].num_elements; ++i) {
                accumulators[localIdx].x[i] *= alpha;
            }

            rocwmma::fragment<rocwmma::accumulator, Config::WMMA_M, Config::WMMA_N, Config::WMMA_K, __half> fragC_out;
            #pragma unroll
            for (int i = 0; i < fragC_out.num_elements; ++i) {
                fragC_out.x[i] = __float2half(accumulators[localIdx].x[i]);
            }

            rocwmma::store_matrix_sync(C + cRow * N + cCol, fragC_out, N, rocwmma::mem_row_major);
        }
    }
}

// Configuration 3: Optimized kernel with double-buffering and larger tiles
template<int BM, int BN, int BK, int THREADS, int PIPE_STAGES>
__global__ void gemm_wmma_optimized(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    using FragA = rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, rocwmma::col_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

    constexpr int TILES_M = BM / WMMA_M;
    constexpr int TILES_N = BN / WMMA_N;
    constexpr int TILES_K = BK / WMMA_K;

    const int numTilesN = (N + BN - 1) / BN;
    const int tileIdx = blockIdx.x;
    const int tileRow = tileIdx / numTilesN;
    const int tileCol = tileIdx % numTilesN;

    if (tileRow * BM >= M || tileCol * BN >= N) return;

    const int tid = threadIdx.x;
    const int waveId = tid / WAVEFRONT_SIZE;
    const int numWaves = THREADS / WAVEFRONT_SIZE;

    // Double-buffered LDS
    __shared__ __half sA[PIPE_STAGES][BM * BK];
    __shared__ __half sB[PIPE_STAGES][BK * BN];

    constexpr int WMMA_TILES = TILES_M * TILES_N;

    FragC accumulators[(WMMA_TILES + numWaves - 1) / numWaves];
    const int tilesPerWave = (WMMA_TILES + numWaves - 1) / numWaves;

    #pragma unroll
    for (int t = 0; t < tilesPerWave; ++t) {
        rocwmma::fill_fragment(accumulators[t], 0.0f);
    }

    const int numKTiles = (K + BK - 1) / BK;
    const int aRowStart = tileRow * BM;
    const int bColStart = tileCol * BN;

    // Load first stage
    int writeStage = 0;
    {
        const int kOffset = 0;
        for (int i = tid; i < BM * BK; i += THREADS) {
            int row = i / BK;
            int col = i % BK;
            int globalRow = aRowStart + row;
            int globalCol = kOffset + col;

            if (globalRow < M && globalCol < K) {
                sA[writeStage][i] = A[globalRow * K + globalCol];
            } else {
                sA[writeStage][i] = __float2half(0.0f);
            }
        }

        for (int i = tid; i < BK * BN; i += THREADS) {
            int row = i / BN;
            int col = i % BN;
            int globalRow = kOffset + row;
            int globalCol = bColStart + col;

            if (globalRow < K && globalCol < N) {
                sB[writeStage][col * BK + row] = B[globalRow * N + globalCol];
            } else {
                sB[writeStage][col * BK + row] = __float2half(0.0f);
            }
        }
    }
    __syncthreads();

    // Main loop with double-buffering
    for (int kTile = 0; kTile < numKTiles; ++kTile) {
        int readStage = kTile % PIPE_STAGES;
        int nextWriteStage = (kTile + 1) % PIPE_STAGES;

        // Start loading next tile while computing
        if (kTile + 1 < numKTiles) {
            const int kOffset = (kTile + 1) * BK;
            for (int i = tid; i < BM * BK; i += THREADS) {
                int row = i / BK;
                int col = i % BK;
                int globalRow = aRowStart + row;
                int globalCol = kOffset + col;

                if (globalRow < M && globalCol < K) {
                    sA[nextWriteStage][i] = A[globalRow * K + globalCol];
                } else {
                    sA[nextWriteStage][i] = __float2half(0.0f);
                }
            }

            for (int i = tid; i < BK * BN; i += THREADS) {
                int row = i / BN;
                int col = i % BN;
                int globalRow = kOffset + row;
                int globalCol = bColStart + col;

                if (globalRow < K && globalCol < N) {
                    sB[nextWriteStage][col * BK + row] = B[globalRow * N + globalCol];
                } else {
                    sB[nextWriteStage][col * BK + row] = __float2half(0.0f);
                }
            }
        }

        // Compute on current stage
        for (int tileOfs = waveId; tileOfs < WMMA_TILES; tileOfs += numWaves) {
            int wmmaTileM = tileOfs / TILES_N;
            int wmmaTileN = tileOfs % TILES_N;
            int localIdx = tileOfs / numWaves;

            FragA fragA;
            FragB fragB;

            #pragma unroll
            for (int wk = 0; wk < TILES_K; ++wk) {
                const __half* aPtr = sA[readStage] + wmmaTileM * WMMA_M * BK + wk * WMMA_K;
                rocwmma::load_matrix_sync(fragA, aPtr, BK);

                const __half* bPtr = sB[readStage] + wmmaTileN * WMMA_N * BK + wk * WMMA_K;
                rocwmma::load_matrix_sync(fragB, bPtr, BK);

                rocwmma::mma_sync(accumulators[localIdx], fragA, fragB, accumulators[localIdx]);
            }
        }

        __syncthreads();
    }

    // Store results
    for (int tileOfs = waveId; tileOfs < WMMA_TILES; tileOfs += numWaves) {
        int wmmaTileM = tileOfs / TILES_N;
        int wmmaTileN = tileOfs % TILES_N;
        int localIdx = tileOfs / numWaves;

        int cRow = aRowStart + wmmaTileM * WMMA_M;
        int cCol = bColStart + wmmaTileN * WMMA_N;

        if (cRow < M && cCol < N) {
            __half* cPtr = C + cRow * N + cCol;

            rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> fragC_out;

            #pragma unroll
            for (int i = 0; i < accumulators[localIdx].num_elements; ++i) {
                accumulators[localIdx].x[i] *= alpha;
            }

            #pragma unroll
            for (int i = 0; i < fragC_out.num_elements; ++i) {
                fragC_out.x[i] = __float2half(accumulators[localIdx].x[i]);
            }

            rocwmma::store_matrix_sync(cPtr, fragC_out, N, rocwmma::mem_row_major);
        }
    }
}

// ============================================================================
// hipBLAS GEMM (baseline)
// ============================================================================

BenchmarkResult benchmark_hipblas(
    int M, int N, int K,
    __half* d_A, __half* d_B, __half* d_C,
    int warmup_iters, int bench_iters
) {
    BenchmarkResult result;
    result.name = "hipBLAS HGEMM";
    result.M = M;
    result.N = N;
    result.K = K;
    result.valid = true;

    hipblasHandle_t handle;
    CHECK_HIPBLAS(hipblasCreate(&handle));

    // hipBLAS uses hipblasHalf which is compatible with __half via reinterpret_cast
    hipblasHalf alpha_val, beta_val;
    __half alpha_h = __float2half(1.0f);
    __half beta_h = __float2half(0.0f);
    memcpy(&alpha_val, &alpha_h, sizeof(hipblasHalf));
    memcpy(&beta_val, &beta_h, sizeof(hipblasHalf));

    const hipblasHalf* d_A_hb = reinterpret_cast<const hipblasHalf*>(d_A);
    const hipblasHalf* d_B_hb = reinterpret_cast<const hipblasHalf*>(d_B);
    hipblasHalf* d_C_hb = reinterpret_cast<hipblasHalf*>(d_C);

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        CHECK_HIPBLAS(hipblasHgemm(
            handle,
            HIPBLAS_OP_N, HIPBLAS_OP_N,
            N, M, K,  // Note: hipBLAS uses column-major, so we swap M and N
            &alpha_val,
            d_B_hb, N,   // B as column-major
            d_A_hb, K,   // A as column-major
            &beta_val,
            d_C_hb, N    // C as column-major
        ));
    }
    CHECK_HIP(hipDeviceSynchronize());

    // Benchmark
    GpuTimer timer;
    std::vector<float> times(bench_iters);

    for (int i = 0; i < bench_iters; ++i) {
        CHECK_HIP(hipMemset(d_C, 0, M * N * sizeof(__half)));

        timer.start();
        CHECK_HIPBLAS(hipblasHgemm(
            handle,
            HIPBLAS_OP_N, HIPBLAS_OP_N,
            N, M, K,
            &alpha_val,
            d_B_hb, N,
            d_A_hb, K,
            &beta_val,
            d_C_hb, N
        ));
        timer.stop();

        times[i] = timer.elapsed_ms();
    }

    std::sort(times.begin(), times.end());

    result.min_time_ms = times[0];
    result.max_time_ms = times[bench_iters - 1];
    result.avg_time_ms = times[bench_iters / 2];  // Median

    double ops = 2.0 * M * N * K;
    result.tflops = ops / (result.avg_time_ms * 1e-3) / 1e12;

    CHECK_HIPBLAS(hipblasDestroy(handle));

    return result;
}

// ============================================================================
// hipBLASLt GEMM (optimized library)
// ============================================================================

BenchmarkResult benchmark_hipblaslt(
    int M, int N, int K,
    __half* d_A, __half* d_B, __half* d_C,
    int warmup_iters, int bench_iters
) {
    BenchmarkResult result;
    result.name = "hipBLASLt HGEMM";
    result.M = M;
    result.N = N;
    result.K = K;
    result.valid = true;

    hipblasLtHandle_t handle;
    hipblasStatus_t status = hipblasLtCreate(&handle);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "hipBLASLt create failed: " << status << "\n";
        result.valid = false;
        result.tflops = 0;
        return result;
    }

    hipblasLtMatmulDesc_t matmulDesc;
    hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;

    hipDataType dataType = HIP_R_16F;
    hipDataType scaleType = HIP_R_32F;  // Use float for alpha/beta scaling
    hipblasComputeType_t computeType = HIPBLAS_COMPUTE_32F;

    // Create matmul descriptor with scale type
    status = hipblasLtMatmulDescCreate(&matmulDesc, computeType, scaleType);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "hipBLASLt matmulDescCreate failed: " << status << "\n";
        hipblasLtDestroy(handle);
        result.valid = false;
        result.tflops = 0;
        return result;
    }

    // Set transpose operations
    hipblasOperation_t transA = HIPBLAS_OP_N;
    hipblasOperation_t transB = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

    // Create matrix layouts (row-major: C = A * B)
    // A: M x K, B: K x N, C: M x N
    status = hipblasLtMatrixLayoutCreate(&layoutA, dataType, M, K, K);  // M rows, K cols, ld=K
    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "hipBLASLt layoutA create failed: " << status << "\n";
        hipblasLtMatmulDescDestroy(matmulDesc);
        hipblasLtDestroy(handle);
        result.valid = false;
        result.tflops = 0;
        return result;
    }

    status = hipblasLtMatrixLayoutCreate(&layoutB, dataType, K, N, N);  // K rows, N cols, ld=N
    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "hipBLASLt layoutB create failed: " << status << "\n";
        hipblasLtMatrixLayoutDestroy(layoutA);
        hipblasLtMatmulDescDestroy(matmulDesc);
        hipblasLtDestroy(handle);
        result.valid = false;
        result.tflops = 0;
        return result;
    }

    status = hipblasLtMatrixLayoutCreate(&layoutC, dataType, M, N, N);  // M rows, N cols, ld=N
    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "hipBLASLt layoutC create failed: " << status << "\n";
        hipblasLtMatrixLayoutDestroy(layoutB);
        hipblasLtMatrixLayoutDestroy(layoutA);
        hipblasLtMatmulDescDestroy(matmulDesc);
        hipblasLtDestroy(handle);
        result.valid = false;
        result.tflops = 0;
        return result;
    }

    // Set row-major order for all matrices
    hipblasLtOrder_t order = HIPBLASLT_ORDER_ROW;
    hipblasLtMatrixLayoutSetAttribute(layoutA, HIPBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    hipblasLtMatrixLayoutSetAttribute(layoutB, HIPBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    hipblasLtMatrixLayoutSetAttribute(layoutC, HIPBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));

    // Find best algorithm
    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);

    size_t workspaceSize = 64 * 1024 * 1024;  // 64 MB workspace
    hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));

    void* workspace;
    CHECK_HIP(hipMalloc(&workspace, workspaceSize));

    // Get heuristics
    hipblasLtMatmulHeuristicResult_t heuristicResult[8];
    int returnedAlgoCount = 0;
    status = hipblasLtMatmulAlgoGetHeuristic(
        handle, matmulDesc, layoutA, layoutB, layoutC, layoutC,
        pref, 8, heuristicResult, &returnedAlgoCount);

    if (status != HIPBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) {
        std::cerr << "No algorithm found for hipBLASLt (status=" << status << ", count=" << returnedAlgoCount << ")\n";
        CHECK_HIP(hipFree(workspace));
        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatrixLayoutDestroy(layoutA);
        hipblasLtMatrixLayoutDestroy(layoutB);
        hipblasLtMatrixLayoutDestroy(layoutC);
        hipblasLtMatmulDescDestroy(matmulDesc);
        hipblasLtDestroy(handle);
        result.valid = false;
        result.tflops = 0;
        return result;
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    // Warmup with best algorithm
    for (int i = 0; i < warmup_iters; ++i) {
        hipblasLtMatmul(
            handle, matmulDesc,
            &alpha,
            d_A, layoutA,
            d_B, layoutB,
            &beta,
            d_C, layoutC,
            d_C, layoutC,
            &heuristicResult[0].algo,
            workspace, workspaceSize,
            0  // stream
        );
    }
    CHECK_HIP(hipDeviceSynchronize());

    // Benchmark
    GpuTimer timer;
    std::vector<float> times(bench_iters);

    for (int i = 0; i < bench_iters; ++i) {
        CHECK_HIP(hipMemset(d_C, 0, M * N * sizeof(__half)));

        timer.start();
        hipblasLtMatmul(
            handle, matmulDesc,
            &alpha,
            d_A, layoutA,
            d_B, layoutB,
            &beta,
            d_C, layoutC,
            d_C, layoutC,
            &heuristicResult[0].algo,
            workspace, workspaceSize,
            0
        );
        timer.stop();

        times[i] = timer.elapsed_ms();
    }

    std::sort(times.begin(), times.end());

    result.min_time_ms = times[0];
    result.max_time_ms = times[bench_iters - 1];
    result.avg_time_ms = times[bench_iters / 2];

    double ops = 2.0 * M * N * K;
    result.tflops = ops / (result.avg_time_ms * 1e-3) / 1e12;

    // Cleanup
    CHECK_HIP(hipFree(workspace));
    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatrixLayoutDestroy(layoutA);
    hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtMatrixLayoutDestroy(layoutC);
    hipblasLtMatmulDescDestroy(matmulDesc);
    hipblasLtDestroy(handle);

    return result;
}

// ============================================================================
// Custom Kernel Benchmarks
// ============================================================================

template<int BM, int BN, int BK, int THREADS>
BenchmarkResult benchmark_custom_16x16(
    const char* name,
    int M, int N, int K,
    __half* d_A, __half* d_B, __half* d_C,
    int warmup_iters, int bench_iters
) {
    BenchmarkResult result;
    result.name = name;
    result.M = M;
    result.N = N;
    result.K = K;
    result.valid = true;

    int numTilesM = (M + BM - 1) / BM;
    int numTilesN = (N + BN - 1) / BN;
    int numBlocks = numTilesM * numTilesN;

    dim3 grid(numBlocks);
    dim3 block(THREADS);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        hipLaunchKernelGGL((gemm_wmma_16x16x16<BM, BN, BK, THREADS>),
                           grid, block, 0, 0, d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    CHECK_HIP(hipDeviceSynchronize());

    // Benchmark
    GpuTimer timer;
    std::vector<float> times(bench_iters);

    for (int i = 0; i < bench_iters; ++i) {
        CHECK_HIP(hipMemset(d_C, 0, M * N * sizeof(__half)));

        timer.start();
        hipLaunchKernelGGL((gemm_wmma_16x16x16<BM, BN, BK, THREADS>),
                           grid, block, 0, 0, d_A, d_B, d_C, M, N, K, alpha, beta);
        timer.stop();

        times[i] = timer.elapsed_ms();
    }

    std::sort(times.begin(), times.end());

    result.min_time_ms = times[0];
    result.max_time_ms = times[bench_iters - 1];
    result.avg_time_ms = times[bench_iters / 2];

    double ops = 2.0 * M * N * K;
    result.tflops = ops / (result.avg_time_ms * 1e-3) / 1e12;

    return result;
}

template<int BM, int BN, int BK, int THREADS>
BenchmarkResult benchmark_custom_32x32(
    const char* name,
    int M, int N, int K,
    __half* d_A, __half* d_B, __half* d_C,
    int warmup_iters, int bench_iters
) {
    BenchmarkResult result;
    result.name = name;
    result.N = N;
    result.M = M;
    result.K = K;
    result.valid = true;

    int numTilesM = (M + BM - 1) / BM;
    int numTilesN = (N + BN - 1) / BN;
    int numBlocks = numTilesM * numTilesN;

    dim3 grid(numBlocks);
    dim3 block(THREADS);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        hipLaunchKernelGGL((gemm_wmma_32x32x8<BM, BN, BK, THREADS>),
                           grid, block, 0, 0, d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    CHECK_HIP(hipDeviceSynchronize());

    // Benchmark
    GpuTimer timer;
    std::vector<float> times(bench_iters);

    for (int i = 0; i < bench_iters; ++i) {
        CHECK_HIP(hipMemset(d_C, 0, M * N * sizeof(__half)));

        timer.start();
        hipLaunchKernelGGL((gemm_wmma_32x32x8<BM, BN, BK, THREADS>),
                           grid, block, 0, 0, d_A, d_B, d_C, M, N, K, alpha, beta);
        timer.stop();

        times[i] = timer.elapsed_ms();
    }

    std::sort(times.begin(), times.end());

    result.min_time_ms = times[0];
    result.max_time_ms = times[bench_iters - 1];
    result.avg_time_ms = times[bench_iters / 2];

    double ops = 2.0 * M * N * K;
    result.tflops = ops / (result.avg_time_ms * 1e-3) / 1e12;

    return result;
}

template<int BM, int BN, int BK, int THREADS, int PIPE_STAGES>
BenchmarkResult benchmark_custom_optimized(
    const char* name,
    int M, int N, int K,
    __half* d_A, __half* d_B, __half* d_C,
    int warmup_iters, int bench_iters
) {
    BenchmarkResult result;
    result.name = name;
    result.M = M;
    result.N = N;
    result.K = K;
    result.valid = true;

    int numTilesM = (M + BM - 1) / BM;
    int numTilesN = (N + BN - 1) / BN;
    int numBlocks = numTilesM * numTilesN;

    dim3 grid(numBlocks);
    dim3 block(THREADS);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        hipLaunchKernelGGL((gemm_wmma_optimized<BM, BN, BK, THREADS, PIPE_STAGES>),
                           grid, block, 0, 0, d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    CHECK_HIP(hipDeviceSynchronize());

    // Benchmark
    GpuTimer timer;
    std::vector<float> times(bench_iters);

    for (int i = 0; i < bench_iters; ++i) {
        CHECK_HIP(hipMemset(d_C, 0, M * N * sizeof(__half)));

        timer.start();
        hipLaunchKernelGGL((gemm_wmma_optimized<BM, BN, BK, THREADS, PIPE_STAGES>),
                           grid, block, 0, 0, d_A, d_B, d_C, M, N, K, alpha, beta);
        timer.stop();

        times[i] = timer.elapsed_ms();
    }

    std::sort(times.begin(), times.end());

    result.min_time_ms = times[0];
    result.max_time_ms = times[bench_iters - 1];
    result.avg_time_ms = times[bench_iters / 2];

    double ops = 2.0 * M * N * K;
    result.tflops = ops / (result.avg_time_ms * 1e-3) / 1e12;

    return result;
}

// ============================================================================
// Print Results
// ============================================================================

void print_results(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                           GEMM Benchmark Results                                      ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Implementation                │ Size           │ Time(ms)  │ TFLOPS   ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════════════════════════╣\n";

    for (const auto& r : results) {
        char size_str[32];
        snprintf(size_str, sizeof(size_str), "%dx%dx%d", r.M, r.N, r.K);

        std::cout << "║ " << std::left << std::setw(29) << r.name << " │ "
                  << std::setw(14) << size_str << " │ "
                  << std::right << std::fixed << std::setprecision(3) << std::setw(9) << r.avg_time_ms << " │ "
                  << std::setprecision(2) << std::setw(8) << r.tflops << " ║\n";
    }

    std::cout << "╚═══════════════════════════════════════════════════════════════════════════════════════╝\n";
}

void print_summary(const std::vector<BenchmarkResult>& results) {
    if (results.empty()) return;

    // Find best result
    auto best = std::max_element(results.begin(), results.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return a.tflops < b.tflops;
        });

    // Find hipBLAS result for comparison
    const BenchmarkResult* hipblas_result = nullptr;
    const BenchmarkResult* hipblaslt_result = nullptr;

    for (const auto& r : results) {
        if (std::string(r.name).find("hipBLAS ") != std::string::npos) {
            hipblas_result = &r;
        }
        if (std::string(r.name).find("hipBLASLt") != std::string::npos) {
            hipblaslt_result = &r;
        }
    }

    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                    Summary                                            ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════════════════════════╣\n";

    if (hipblas_result) {
        std::cout << "║ hipBLAS Baseline:          " << std::fixed << std::setprecision(2)
                  << hipblas_result->tflops << " TFLOPS"
                  << std::setw(42) << "" << "║\n";
    }

    if (hipblaslt_result) {
        std::cout << "║ hipBLASLt Baseline:        " << std::fixed << std::setprecision(2)
                  << hipblaslt_result->tflops << " TFLOPS"
                  << std::setw(41) << "" << "║\n";
    }

    std::cout << "║ Best Custom Kernel:        " << std::fixed << std::setprecision(2)
              << best->tflops << " TFLOPS"
              << std::setw(41) << "" << "║\n";
    std::cout << "║   Configuration:           " << std::left << std::setw(58) << best->name << "║\n";

    std::cout << "╠═══════════════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║                               Recommendations                                         ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════════════════════════╣\n";

    if (hipblaslt_result && hipblaslt_result->tflops > best->tflops * 1.1) {
        std::cout << "║ RECOMMENDATION: Use hipBLASLt for production - it achieves higher TFLOPS            ║\n";
    } else {
        std::cout << "║ Custom kernel performance is competitive with vendor libraries.                     ║\n";
    }

    std::cout << "╚═══════════════════════════════════════════════════════════════════════════════════════╝\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // Default problem sizes
    std::vector<std::tuple<int, int, int>> problem_sizes = {
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {8192, 8192, 8192},
    };

    int warmup_iters = 5;
    int bench_iters = 20;

    // Parse command line
    if (argc >= 4) {
        problem_sizes.clear();
        problem_sizes.push_back({std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])});
    }
    if (argc >= 5) {
        bench_iters = std::atoi(argv[4]);
    }

    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           FlashMoE HIP Port - Comprehensive GEMM Benchmark                            ║\n";
    std::cout << "║              hipBLAS vs hipBLASLt vs Custom rocWMMA Kernels                           ║\n";
    std::cout << "║                       AMD Instinct MI450 (gfx1250)                                    ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════════════════╝\n";

    // Print device info
    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, 0));

    std::cout << "\nDevice: " << props.name << " (" << props.gcnArchName << ")\n";
    std::cout << "\nWarmup: " << warmup_iters << ", Benchmark: " << bench_iters << " iterations\n";

    // Run benchmarks for each problem size
    for (const auto& [M, N, K] : problem_sizes) {
        std::cout << "\n";
        std::cout << "═══════════════════════════════════════════════════════════════════════════════════════\n";
        std::cout << "  Problem Size: " << M << " x " << N << " x " << K << "\n";
        std::cout << "═══════════════════════════════════════════════════════════════════════════════════════\n";

        // Allocate device memory
        __half* d_A = nullptr;
        __half* d_B = nullptr;
        __half* d_C = nullptr;

        size_t size_A = static_cast<size_t>(M) * K * sizeof(__half);
        size_t size_B = static_cast<size_t>(K) * N * sizeof(__half);
        size_t size_C = static_cast<size_t>(M) * N * sizeof(__half);
                  << size_B / (1024*1024) << "MB, C=" << size_C / (1024*1024) << "MB\n";

        CHECK_HIP(hipMalloc(&d_A, size_A));
        CHECK_HIP(hipMalloc(&d_B, size_B));
        CHECK_HIP(hipMalloc(&d_C, size_C));

        // Initialize with random-ish values
        std::vector<__half> h_A(static_cast<size_t>(M) * K);
        std::vector<__half> h_B(static_cast<size_t>(K) * N);

        for (size_t i = 0; i < h_A.size(); ++i) {
            h_A[i] = __float2half(static_cast<float>(i % 10) * 0.1f);
        }
        for (size_t i = 0; i < h_B.size(); ++i) {
            h_B[i] = __float2half(static_cast<float>(i % 10) * 0.1f);
        }

        CHECK_HIP(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));

        std::vector<BenchmarkResult> results;

        // 1. hipBLAS baseline
        std::cout << "\nRunning hipBLAS baseline... ";
        std::cout.flush();
        auto r1 = benchmark_hipblas(M, N, K, d_A, d_B, d_C, warmup_iters, bench_iters);
        results.push_back(r1);
        std::cout << r1.tflops << " TFLOPS\n";

        // 2. hipBLASLt (optimized)
        std::cout << "Running hipBLASLt... ";
        std::cout.flush();
        auto r2 = benchmark_hipblaslt(M, N, K, d_A, d_B, d_C, warmup_iters, bench_iters);
        results.push_back(r2);
        std::cout << r2.tflops << " TFLOPS\n";

        // 3. Custom kernels with different configurations
        std::cout << "\nRunning custom rocWMMA kernels...\n";

        // 16x16x16 WMMA configurations
        if (M >= 64 && N >= 64 && K >= 32) {
            std::cout << "  64x64x32 (16x16x16 WMMA, 256 threads)... ";
            std::cout.flush();
            auto r = benchmark_custom_16x16<64, 64, 32, 256>(
                "Custom 64x64x32 w16", M, N, K, d_A, d_B, d_C, warmup_iters, bench_iters);
            results.push_back(r);
            std::cout << r.tflops << " TFLOPS\n";
        }

        if (M >= 128 && N >= 128 && K >= 32) {
            std::cout << "  128x128x32 (16x16x16 WMMA, 256 threads)... ";
            std::cout.flush();
            auto r = benchmark_custom_16x16<128, 128, 32, 256>(
                "Custom 128x128x32 w16", M, N, K, d_A, d_B, d_C, warmup_iters, bench_iters);
            results.push_back(r);
            std::cout << r.tflops << " TFLOPS\n";
        }

        if (M >= 128 && N >= 128 && K >= 64) {
            std::cout << "  128x128x64 (16x16x16 WMMA, 256 threads)... ";
            std::cout.flush();
            auto r = benchmark_custom_16x16<128, 128, 64, 256>(
                "Custom 128x128x64 w16", M, N, K, d_A, d_B, d_C, warmup_iters, bench_iters);
            results.push_back(r);
            std::cout << r.tflops << " TFLOPS\n";
        }

        // 32x32x8 WMMA configurations (larger tiles)
        if (M >= 128 && N >= 128 && K >= 32) {
            std::cout << "  128x128x32 (32x32x8 WMMA, 256 threads)... ";
            std::cout.flush();
            auto r = benchmark_custom_32x32<128, 128, 32, 256>(
                "Custom 128x128x32 w32", M, N, K, d_A, d_B, d_C, warmup_iters, bench_iters);
            results.push_back(r);
            std::cout << r.tflops << " TFLOPS\n";
        }

        if (M >= 256 && N >= 256 && K >= 32) {
            std::cout << "  256x256x32 (32x32x8 WMMA, 512 threads)... ";
            std::cout.flush();
            auto r = benchmark_custom_32x32<256, 256, 32, 512>(
                "Custom 256x256x32 w32", M, N, K, d_A, d_B, d_C, warmup_iters, bench_iters);
            results.push_back(r);
            std::cout << r.tflops << " TFLOPS\n";
        }

        // Optimized with double-buffering
        if (M >= 128 && N >= 128 && K >= 64) {
            std::cout << "  128x128x64 (optimized, 2 stages)... ";
            std::cout.flush();
            auto r = benchmark_custom_optimized<128, 128, 64, 256, 2>(
                "Custom Opt 128x128x64", M, N, K, d_A, d_B, d_C, warmup_iters, bench_iters);
            results.push_back(r);
            std::cout << r.tflops << " TFLOPS\n";
        }

        // More threads
        if (M >= 128 && N >= 128 && K >= 32) {
            std::cout << "  128x128x32 (16x16x16 WMMA, 512 threads)... ";
            std::cout.flush();
            auto r = benchmark_custom_16x16<128, 128, 32, 512>(
                "Custom 128x128x32 512t", M, N, K, d_A, d_B, d_C, warmup_iters, bench_iters);
            results.push_back(r);
            std::cout << r.tflops << " TFLOPS\n";
        }

        // Highly optimized kernel
        if (M >= 128 && N >= 128 && K >= 32) {
            std::cout << "  Highly optimized (vectorized, padded LDS)... ";
            std::cout.flush();

            BenchmarkResult result;
            result.name = "Custom Highly Optimized";
            result.M = M;
            result.N = N;
            result.K = K;
            result.valid = true;

            int numTilesM = (M + OptConfig::BM - 1) / OptConfig::BM;
            int numTilesN = (N + OptConfig::BN - 1) / OptConfig::BN;
            int numBlocks = numTilesM * numTilesN;

            dim3 grid(numBlocks);
            dim3 block(OptConfig::THREADS);

            float alpha_v = 1.0f;
            float beta_v = 0.0f;

            // Warmup
            for (int i = 0; i < warmup_iters; ++i) {
                hipLaunchKernelGGL((gemm_highly_optimized<OptConfig>),
                                   grid, block, 0, 0, d_A, d_B, d_C, M, N, K, alpha_v, beta_v);
            }
            CHECK_HIP(hipDeviceSynchronize());

            // Benchmark
            GpuTimer timer;
            std::vector<float> times(bench_iters);

            for (int i = 0; i < bench_iters; ++i) {
                CHECK_HIP(hipMemset(d_C, 0, M * N * sizeof(__half)));

                timer.start();
                hipLaunchKernelGGL((gemm_highly_optimized<OptConfig>),
                                   grid, block, 0, 0, d_A, d_B, d_C, M, N, K, alpha_v, beta_v);
                timer.stop();

                times[i] = timer.elapsed_ms();
            }

            std::sort(times.begin(), times.end());

            result.min_time_ms = times[0];
            result.max_time_ms = times[bench_iters - 1];
            result.avg_time_ms = times[bench_iters / 2];

            double ops = 2.0 * M * N * K;
            result.tflops = ops / (result.avg_time_ms * 1e-3) / 1e12;

            results.push_back(result);
            std::cout << result.tflops << " TFLOPS\n";
        }

        // Sort by TFLOPS
        std::sort(results.begin(), results.end(),
            [](const BenchmarkResult& a, const BenchmarkResult& b) {
                return a.tflops > b.tflops;
            });

        // Print results
        print_results(results);
        print_summary(results);

        // Cleanup
        CHECK_HIP(hipFree(d_A));
        CHECK_HIP(hipFree(d_B));
        CHECK_HIP(hipFree(d_C));
    }

    std::cout << "\nBenchmark complete.\n\n";

    return 0;
}
