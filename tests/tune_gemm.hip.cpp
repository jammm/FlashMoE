//
// FlashMoE HIP Port - GEMM Tuning Benchmark
// Tests different tile size configurations and measures TFLOPS
// Optimized for AMD Instinct MI450 (gfx1250) / MI450 (gfx1250)
//
// Copyright (c) 2025, Osayamen Jonathan Aimuyo
// All rights reserved.
//

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>

// Include FlashMoE HIP headers
#include "flashmoe/hip/tile.hip.cuh"
#include "flashmoe/hip/tuning.hpp"
#include "flashmoe/hip/constants.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>

// ============================================================================
// Error Checking
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

// ============================================================================
// Timer Utilities
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
// GEMM Kernel Templates for Different Tile Sizes
// ============================================================================

// Generic GEMM kernel using rocWMMA
template<int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K, int THREADS>
__global__ void gemm_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Calculate tile indices
    const int numTilesN = (N + BN - 1) / BN;
    const int tileIdx = blockIdx.x;
    const int tileRow = tileIdx / numTilesN;
    const int tileCol = tileIdx % numTilesN;

    // Check bounds
    if (tileRow * BM >= M || tileCol * BN >= N) return;

    // Fragment types
    using FragA = rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, rocwmma::col_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

    // Number of WMMA tiles per block tile
    constexpr int TILES_M = BM / WMMA_M;
    constexpr int TILES_N = BN / WMMA_N;
    constexpr int TILES_K = BK / WMMA_K;

    // Thread/wave info
    const int tid = threadIdx.x;
    const int waveId = tid / flashmoe::WAVEFRONT_SIZE;
    const int numWaves = THREADS / flashmoe::WAVEFRONT_SIZE;

    // LDS for tiling
    __shared__ __half sA[BM * BK];
    __shared__ __half sB[BK * BN];

    // Each wave handles one or more WMMA output tiles
    constexpr int WMMA_TILES = TILES_M * TILES_N;

    // Initialize accumulators for this wave's tiles
    FragC accumulators[WMMA_TILES > numWaves ? (WMMA_TILES + numWaves - 1) / numWaves : 1];

    const int tilesPerWave = (WMMA_TILES + numWaves - 1) / numWaves;

    #pragma unroll
    for (int t = 0; t < tilesPerWave; ++t) {
        rocwmma::fill_fragment(accumulators[t], 0.0f);
    }

    // Number of K iterations
    const int numKTiles = (K + BK - 1) / BK;

    // Global tile offsets
    const int aRowStart = tileRow * BM;
    const int bColStart = tileCol * BN;

    // Main K loop
    for (int kTile = 0; kTile < numKTiles; ++kTile) {
        const int kOffset = kTile * BK;

        // Cooperative load A tile to LDS
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

        // Cooperative load B tile to LDS (stored in column-major for rocWMMA)
        for (int i = tid; i < BK * BN; i += THREADS) {
            int row = i / BN;
            int col = i % BN;
            int globalRow = kOffset + row;
            int globalCol = bColStart + col;

            if (globalRow < K && globalCol < N) {
                // B is assumed row-major, but we store transposed in LDS
                sB[col * BK + row] = B[globalRow * N + globalCol];
            } else {
                sB[col * BK + row] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // Compute WMMA tiles
        for (int tileOfs = waveId; tileOfs < WMMA_TILES; tileOfs += numWaves) {
            int wmmaTileM = tileOfs / TILES_N;
            int wmmaTileN = tileOfs % TILES_N;
            int localIdx = tileOfs / numWaves;

            FragA fragA;
            FragB fragB;

            // Accumulate over K tiles within the block
            #pragma unroll
            for (int wk = 0; wk < TILES_K; ++wk) {
                // Load A fragment from LDS
                const __half* aPtr = sA + wmmaTileM * WMMA_M * BK + wk * WMMA_K;
                rocwmma::load_matrix_sync(fragA, aPtr, BK);

                // Load B fragment from LDS (column-major)
                const __half* bPtr = sB + wmmaTileN * WMMA_N * BK + wk * WMMA_K;
                rocwmma::load_matrix_sync(fragB, bPtr, BK);

                // MMA
                rocwmma::mma_sync(accumulators[localIdx], fragA, fragB, accumulators[localIdx]);
            }
        }

        __syncthreads();
    }

    // Store results to global memory
    for (int tileOfs = waveId; tileOfs < WMMA_TILES; tileOfs += numWaves) {
        int wmmaTileM = tileOfs / TILES_N;
        int wmmaTileN = tileOfs % TILES_N;
        int localIdx = tileOfs / numWaves;

        int cRow = aRowStart + wmmaTileM * WMMA_M;
        int cCol = bColStart + wmmaTileN * WMMA_N;

        if (cRow < M && cCol < N) {
            float* cPtr = C + cRow * N + cCol;
            rocwmma::store_matrix_sync(cPtr, accumulators[localIdx], N, rocwmma::mem_row_major);
        }
    }
}

// ============================================================================
// Benchmark Configuration
// ============================================================================

struct BenchmarkConfig {
    int bM, bN, bK;
    int wmma_m, wmma_n, wmma_k;
    int threads;
    const char* name;
};

// Configurations to test
static const BenchmarkConfig BENCHMARK_CONFIGS[] = {
    // Small tiles (16x16 WMMA)
    {32, 32, 32, 16, 16, 16, 128, "32x32x32_w16"},
    {64, 64, 32, 16, 16, 16, 256, "64x64x32_w16"},
    {64, 64, 64, 16, 16, 16, 256, "64x64x64_w16"},

    // Medium tiles (16x16 WMMA)
    {64, 128, 32, 16, 16, 16, 256, "64x128x32_w16"},
    {128, 64, 32, 16, 16, 16, 256, "128x64x32_w16"},
    {128, 128, 32, 16, 16, 16, 256, "128x128x32_w16"},

    // Large tiles (32x32 WMMA)
    {128, 128, 32, 32, 32, 8, 256, "128x128x32_w32"},
    {128, 128, 64, 32, 32, 8, 256, "128x128x64_w32"},
    {128, 256, 32, 32, 32, 8, 256, "128x256x32_w32"},
    {256, 128, 32, 32, 32, 8, 256, "256x128x32_w32"},
    {256, 256, 32, 32, 32, 8, 512, "256x256x32_w32"},
};

static const int NUM_CONFIGS = sizeof(BENCHMARK_CONFIGS) / sizeof(BENCHMARK_CONFIGS[0]);

// ============================================================================
// Benchmark Result
// ============================================================================

struct BenchmarkResult {
    const char* name;
    int bM, bN, bK;
    int threads;
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    double tflops;
    bool valid;  // Correctness check passed
};

// ============================================================================
// Run Single Benchmark
// ============================================================================

template<int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K, int THREADS>
BenchmarkResult run_benchmark(
    int M, int N, int K,
    __half* d_A, __half* d_B, float* d_C,
    int warmup_iters, int bench_iters,
    const char* name
) {
    BenchmarkResult result;
    result.name = name;
    result.bM = BM;
    result.bN = BN;
    result.bK = BK;
    result.threads = THREADS;

    // Grid dimensions
    int numTilesM = (M + BM - 1) / BM;
    int numTilesN = (N + BN - 1) / BN;
    int numBlocks = numTilesM * numTilesN;

    dim3 grid(numBlocks);
    dim3 block(THREADS);

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        hipLaunchKernelGGL((gemm_kernel<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K, THREADS>),
                           grid, block, 0, 0, d_A, d_B, d_C, M, N, K);
    }
    CHECK_HIP(hipDeviceSynchronize());

    // Benchmark
    GpuTimer timer;
    std::vector<float> times(bench_iters);

    for (int i = 0; i < bench_iters; ++i) {
        CHECK_HIP(hipMemset(d_C, 0, M * N * sizeof(float)));

        timer.start();
        hipLaunchKernelGGL((gemm_kernel<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K, THREADS>),
                           grid, block, 0, 0, d_A, d_B, d_C, M, N, K);
        timer.stop();

        times[i] = timer.elapsed_ms();
    }

    // Calculate statistics
    std::sort(times.begin(), times.end());

    result.min_time_ms = times[0];
    result.max_time_ms = times[bench_iters - 1];

    // Use median for avg
    result.avg_time_ms = times[bench_iters / 2];

    // Calculate TFLOPS
    double ops = 2.0 * M * N * K;
    result.tflops = ops / (result.avg_time_ms * 1e-3) / 1e12;

    // Simple validity check (not a full correctness check)
    result.valid = true;  // Assume valid for now

    return result;
}

// ============================================================================
// Dispatch to Correct Template
// ============================================================================

BenchmarkResult dispatch_benchmark(
    const BenchmarkConfig& config,
    int M, int N, int K,
    __half* d_A, __half* d_B, float* d_C,
    int warmup_iters, int bench_iters
) {
    // Match configuration to template
    #define DISPATCH_CASE(bm, bn, bk, wm, wn, wk, th) \
        if (config.bM == bm && config.bN == bn && config.bK == bk && \
            config.wmma_m == wm && config.wmma_n == wn && config.wmma_k == wk && \
            config.threads == th) { \
            return run_benchmark<bm, bn, bk, wm, wn, wk, th>( \
                M, N, K, d_A, d_B, d_C, warmup_iters, bench_iters, config.name); \
        }

    // Small tiles (16x16 WMMA)
    DISPATCH_CASE(32, 32, 32, 16, 16, 16, 128)
    DISPATCH_CASE(64, 64, 32, 16, 16, 16, 256)
    DISPATCH_CASE(64, 64, 64, 16, 16, 16, 256)
    DISPATCH_CASE(64, 128, 32, 16, 16, 16, 256)
    DISPATCH_CASE(128, 64, 32, 16, 16, 16, 256)
    DISPATCH_CASE(128, 128, 32, 16, 16, 16, 256)

    // Large tiles (32x32 WMMA)
    DISPATCH_CASE(128, 128, 32, 32, 32, 8, 256)
    DISPATCH_CASE(128, 128, 64, 32, 32, 8, 256)
    DISPATCH_CASE(128, 256, 32, 32, 32, 8, 256)
    DISPATCH_CASE(256, 128, 32, 32, 32, 8, 256)
    DISPATCH_CASE(256, 256, 32, 32, 32, 8, 512)

    #undef DISPATCH_CASE

    // If no match, return invalid result
    BenchmarkResult invalid;
    invalid.name = config.name;
    invalid.valid = false;
    invalid.tflops = 0;
    invalid.avg_time_ms = 0;
    return invalid;
}

// ============================================================================
// Correctness Verification
// ============================================================================

bool verify_result(
    float* d_C, int M, int N, int K,
    float expected_value, float tolerance = 1e-2f
) {
    std::vector<float> h_C(M * N);
    CHECK_HIP(hipMemcpy(h_C.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));

    int errors = 0;
    float max_error = 0.0f;

    for (int i = 0; i < M * N && errors < 10; ++i) {
        float error = std::abs(h_C[i] - expected_value);
        max_error = std::max(max_error, error);
        if (error > tolerance) {
            errors++;
        }
    }

    if (errors > 0) {
        std::cerr << "  Verification failed: " << errors << " errors, max_error=" << max_error << "\n";
        return false;
    }

    return true;
}

// ============================================================================
// Print Results
// ============================================================================

void print_results(const std::vector<BenchmarkResult>& results, int M, int N, int K) {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                       GEMM Tuning Benchmark Results                            ║\n";
    std::cout << "║                        Problem: " << std::setw(5) << M << " x " << std::setw(5) << N
              << " x " << std::setw(5) << K << "                             ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Configuration          │ Tile Size    │Threads│ Time(ms) │ TFLOPS  ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════════════════════╣\n";

    for (const auto& r : results) {
        std::cout << "║ " << std::left << std::setw(21) << r.name << " │ "
                  << std::right << std::setw(3) << r.bM << "x"
                  << std::setw(3) << r.bN << "x"
                  << std::setw(2) << r.bK << " │ "
                  << std::setw(5) << r.threads << " │ "
                  << std::fixed << std::setprecision(3) << std::setw(8) << r.avg_time_ms << " │ "
                  << std::setprecision(2) << std::setw(7) << r.tflops << " ║\n";
    }

    std::cout << "╚════════════════════════════════════════════════════════════════════════════════╝\n";
}

void print_best_config(const std::vector<BenchmarkResult>& results) {
    if (results.empty()) return;

    auto best = std::max_element(results.begin(), results.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return a.tflops < b.tflops;
        });

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                           BEST CONFIGURATION                                   ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Name: " << std::left << std::setw(72) << best->name << "║\n";
    std::cout << "║ Tile: " << best->bM << "x" << best->bN << "x" << best->bK
              << ", Threads: " << best->threads
              << std::setw(50) << "" << "║\n";
    std::cout << "║ Performance: " << std::fixed << std::setprecision(2) << best->tflops
              << " TFLOPS" << std::setw(48) << "" << "║\n";
    std::cout << "║ Time: " << std::setprecision(3) << best->avg_time_ms << " ms"
              << std::setw(61) << "" << "║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════╝\n";
}

// ============================================================================
// Test Autotuning Selection
// ============================================================================

void test_autotune_selection(int M, int N, int K) {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                        Autotuning Selection Test                               ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════╝\n";

    // Get recommended configuration from tuning.hpp
    auto cfg = flashmoe::tuning::get_tuning_config(M, N, K, true);

    std::cout << "Problem: " << M << "x" << N << "x" << K << "\n";
    std::cout << "Selected configuration: " << cfg.description << "\n";
    std::cout << "  Tile: " << cfg.bM << "x" << cfg.bN << "x" << cfg.bK << "\n";
    std::cout << "  WMMA: " << cfg.wmma_m << "x" << cfg.wmma_n << "x" << cfg.wmma_k << "\n";
    std::cout << "  Threads: " << cfg.threads << ", Pipeline stages: " << cfg.pipe_stages << "\n";
    std::cout << "  Memory bound: " << (cfg.is_memory_bound ? "yes" : "no") << "\n";
    std::cout << "  Estimated score: " << std::fixed << std::setprecision(3)
              << cfg.estimated_score << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // Default problem sizes
    int M = 4096;
    int N = 4096;
    int K = 4096;
    int warmup_iters = 5;
    int bench_iters = 10;

    // Parse command line arguments
    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    if (argc >= 5) {
        bench_iters = std::atoi(argv[4]);
    }

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║            FlashMoE HIP Port - GEMM Tuning Benchmark                          ║\n";
    std::cout << "║                    AMD Instinct MI450 (gfx1250)                               ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════╝\n";

    // Print device info
    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, 0));
    std::cout << "\nDevice: " << props.name << " (" << props.gcnArchName << ")\n";

    std::cout << "\nProblem size: " << M << " x " << N << " x " << K << "\n";
    std::cout << "FP16 input, FP32 accumulator\n";
    std::cout << "Warmup: " << warmup_iters << ", Benchmark: " << bench_iters << " iterations\n";

    // Allocate device memory
    __half* d_A = nullptr;
    __half* d_B = nullptr;
    float* d_C = nullptr;

    size_t size_A = static_cast<size_t>(M) * K * sizeof(__half);
    size_t size_B = static_cast<size_t>(K) * N * sizeof(__half);
    size_t size_C = static_cast<size_t>(M) * N * sizeof(float);
              << size_B / (1024*1024) << "MB, C=" << size_C / (1024*1024) << "MB\n";

    CHECK_HIP(hipMalloc(&d_A, size_A));
    CHECK_HIP(hipMalloc(&d_B, size_B));
    CHECK_HIP(hipMalloc(&d_C, size_C));

    // Initialize with test data (all ones for easy verification)
    std::vector<__half> h_A(static_cast<size_t>(M) * K);
    std::vector<__half> h_B(static_cast<size_t>(K) * N);

    for (size_t i = 0; i < h_A.size(); ++i) h_A[i] = __float2half(1.0f);
    for (size_t i = 0; i < h_B.size(); ++i) h_B[i] = __float2half(1.0f);

    CHECK_HIP(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));

    // Test autotuning selection
    test_autotune_selection(M, N, K);

    // Run benchmarks
    std::cout << "\nRunning benchmarks...\n";

    std::vector<BenchmarkResult> results;

    for (int i = 0; i < NUM_CONFIGS; ++i) {
        const auto& config = BENCHMARK_CONFIGS[i];

        // Skip configurations that don't fit the problem size
        if (M < config.bM || N < config.bN || K < config.bK) {
            std::cout << "  Skipping " << config.name << " (problem too small)\n";
            continue;
        }

        std::cout << "  Testing " << config.name << "... " << std::flush;

        auto result = dispatch_benchmark(config, M, N, K, d_A, d_B, d_C,
                                         warmup_iters, bench_iters);

        if (result.valid) {
            // Verify correctness
            float expected = static_cast<float>(K);  // Sum of K ones
            result.valid = verify_result(d_C, M, N, K, expected);

            std::cout << std::fixed << std::setprecision(2) << result.tflops << " TFLOPS";
            if (!result.valid) {
                std::cout << " (VERIFICATION FAILED)";
            }
            std::cout << "\n";
        } else {
            std::cout << "SKIPPED (no template)\n";
        }

        results.push_back(result);
    }

    // Sort by TFLOPS (descending)
    std::sort(results.begin(), results.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return a.tflops > b.tflops;
        });

    // Print results
    print_results(results, M, N, K);
    print_best_config(results);

    // Cleanup
    CHECK_HIP(hipFree(d_A));
    CHECK_HIP(hipFree(d_B));
    CHECK_HIP(hipFree(d_C));

    std::cout << "\nBenchmark complete.\n\n";

    return 0;
}
