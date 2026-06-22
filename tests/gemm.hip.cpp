//
// FlashMoE HIP Port - GEMM Correctness Test
// Tests the rocWMMA-based tile GEMM implementation on AMD GPUs
//
// This test validates:
// 1. Basic matrix multiplication C = A × B
// 2. rocWMMA fragment operations work correctly
// 3. Memory allocation/copy between host and device
//
// Expected result: When A and B are filled with 1.0, C[i,j] = K (the reduction dim)
//

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>

// ============================================================================
// Error Checking Macro
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
// Test Configuration
// ============================================================================

// rocWMMA tile sizes used by the gfx1250 smoke test.
// 16×16×16 for FP16 with float accumulator
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Test matrix dimensions (must be multiples of WMMA tile size)
constexpr int TEST_M = 64;
constexpr int TEST_N = 64;
constexpr int TEST_K = 64;

// Block and grid configuration
constexpr int BLOCK_SIZE = 256;

// ============================================================================
// Simple rocWMMA GEMM Kernel
// Computes C = A × B where:
//   A is M×K (row-major)
//   B is K×N (col-major, like PyTorch's transposed weight)
//   C is M×N (row-major)
// ============================================================================

__global__ void gemmKernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Fragment types
    using FragA = rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, rocwmma::col_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

    // Calculate which output tile this block handles
    const int numTilesN = N / WMMA_N;
    const int numTilesM = M / WMMA_M;
    const int totalTiles = numTilesM * numTilesN;

    // Each block handles one tile
    const int tileIdx = blockIdx.x;
    if (tileIdx >= totalTiles) return;

    const int tileRow = tileIdx / numTilesN;
    const int tileCol = tileIdx % numTilesN;

    // Global memory offsets
    const int aRowStart = tileRow * WMMA_M;
    const int bColStart = tileCol * WMMA_N;
    const int cRowStart = tileRow * WMMA_M;
    const int cColStart = tileCol * WMMA_N;

    // Initialize accumulator
    FragC fragC;
    rocwmma::fill_fragment(fragC, 0.0f);

    // Loop over K dimension
    for (int kTile = 0; kTile < K / WMMA_K; ++kTile) {
        FragA fragA;
        FragB fragB;

        // Load A tile: A[aRowStart:aRowStart+WMMA_M, kTile*WMMA_K:(kTile+1)*WMMA_K]
        // A is row-major: ld = K
        const __half* aPtr = A + aRowStart * K + kTile * WMMA_K;
        rocwmma::load_matrix_sync(fragA, aPtr, K);

        // Load B tile: B[kTile*WMMA_K:(kTile+1)*WMMA_K, bColStart:bColStart+WMMA_N]
        // B is col-major: ld = K (leading dimension is rows in col-major)
        const __half* bPtr = B + bColStart * K + kTile * WMMA_K;
        rocwmma::load_matrix_sync(fragB, bPtr, K);

        // Matrix multiply-accumulate
        rocwmma::mma_sync(fragC, fragA, fragB, fragC);
    }

    // Store C tile: C[cRowStart:cRowStart+WMMA_M, cColStart:cColStart+WMMA_N]
    // C is row-major: ld = N
    float* cPtr = C + cRowStart * N + cColStart;
    rocwmma::store_matrix_sync(cPtr, fragC, N, rocwmma::mem_row_major);
}

// ============================================================================
// Alternative: Simple naive GEMM kernel for reference comparison
// ============================================================================

__global__ void naiveGemmKernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            // A is row-major: A[row, k] = A[row * K + k]
            // B is col-major: B[k, col] = B[col * K + k]
            float a_val = __half2float(A[row * K + k]);
            float b_val = __half2float(B[col * K + k]);
            sum += a_val * b_val;
        }
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Test Functions
// ============================================================================

bool runGemmTest(int M, int N, int K, bool useWmma, bool verbose = true) {
    if (verbose) {
        std::cout << "\n========================================\n";
        std::cout << "GEMM Test: M=" << M << ", N=" << N << ", K=" << K << "\n";
        std::cout << "Method: " << (useWmma ? "rocWMMA" : "Naive") << "\n";
        std::cout << "========================================\n";
    }

    // Allocate host memory
    std::vector<__half> h_A(M * K);
    std::vector<__half> h_B(K * N);
    std::vector<float> h_C(M * N);
    std::vector<float> h_C_ref(M * N);

    // Initialize matrices with known values
    // A[i,j] = 1.0, B[i,j] = 1.0
    // Expected: C[i,j] = K (since each element is sum of K products of 1.0×1.0)
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = __float2half(1.0f);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = __float2half(1.0f);
    }

    // Expected value
    float expected = static_cast<float>(K);

    // Allocate device memory
    __half* d_A = nullptr;
    __half* d_B = nullptr;
    float* d_C = nullptr;

    CHECK_HIP(hipMalloc(&d_A, M * K * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_B, K * N * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_C, M * N * sizeof(float)));

    // Copy input to device
    CHECK_HIP(hipMemcpy(d_A, h_A.data(), M * K * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B.data(), K * N * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_C, 0, M * N * sizeof(float)));

    // Launch kernel
    auto start = std::chrono::high_resolution_clock::now();

    if (useWmma) {
        // rocWMMA kernel: one block per output tile
        int numTiles = (M / WMMA_M) * (N / WMMA_N);
        dim3 grid(numTiles);
        dim3 block(BLOCK_SIZE);

        hipLaunchKernelGGL(gemmKernel, grid, block, 0, 0, d_A, d_B, d_C, M, N, K);
    } else {
        // Naive kernel: one thread per output element
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

        hipLaunchKernelGGL(naiveGemmKernel, grid, block, 0, 0, d_A, d_B, d_C, M, N, K);
    }

    CHECK_HIP(hipDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Copy result back
    CHECK_HIP(hipMemcpy(h_C.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));

    // Verify results
    int errors = 0;
    float maxError = 0.0f;
    float tolerance = 1e-3f;  // Allow small FP16 precision errors

    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_C[i] - expected);
        maxError = std::max(maxError, error);
        if (error > tolerance) {
            if (errors < 10 && verbose) {
                int row = i / N;
                int col = i % N;
                std::cerr << "  Error at [" << row << "," << col << "]: "
                          << "got " << h_C[i] << ", expected " << expected << "\n";
            }
            errors++;
        }
    }

    // Calculate TFLOPS
    double flops = 2.0 * M * N * K;  // Each element: K multiplies + K-1 adds
    double seconds = duration.count() / 1e6;
    double tflops = (flops / seconds) / 1e12;

    // Report results
    if (verbose) {
        std::cout << "\nResults:\n";
        std::cout << "  Time: " << std::fixed << std::setprecision(3)
                  << duration.count() / 1000.0 << " ms\n";
        std::cout << "  Performance: " << std::fixed << std::setprecision(3)
                  << tflops << " TFLOPS\n";
        std::cout << "  Max Error: " << std::scientific << maxError << "\n";
        std::cout << "  Errors: " << errors << " / " << (M * N) << "\n";
    }

    bool passed = (errors == 0);
    std::cout << "  Status: " << (passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m") << "\n";

    // Cleanup
    CHECK_HIP(hipFree(d_A));
    CHECK_HIP(hipFree(d_B));
    CHECK_HIP(hipFree(d_C));

    return passed;
}

// ============================================================================
// Query GPU Info
// ============================================================================

void printGpuInfo() {
    int deviceCount = 0;
    CHECK_HIP(hipGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No HIP-capable devices found!\n";
        std::exit(1);
    }

    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, 0));

    std::cout << "========================================\n";
    std::cout << "GPU Device Information\n";
    std::cout << "========================================\n";
    std::cout << "Device: " << props.name << "\n";
    std::cout << "Warp Size: " << props.warpSize << "\n";
    std::cout << "Max Threads/Block: " << props.maxThreadsPerBlock << "\n";
    std::cout << "GCN Architecture: " << props.gcnArchName << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║      FlashMoE HIP Port - GEMM Correctness Test           ║\n";
    std::cout << "║              rocWMMA on AMD MI450 (gfx1250)              ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";

    // Print GPU info
    printGpuInfo();

    int totalTests = 0;
    int passedTests = 0;

    // Test 1: Basic small GEMM with rocWMMA
    std::cout << "\n--- Test 1: Small GEMM (64×64×64) with rocWMMA ---\n";
    totalTests++;
    if (runGemmTest(64, 64, 64, true)) passedTests++;

    // Test 2: Naive GEMM for comparison
    std::cout << "\n--- Test 2: Small GEMM (64×64×64) with Naive kernel ---\n";
    totalTests++;
    if (runGemmTest(64, 64, 64, false)) passedTests++;

    // Test 3: Larger GEMM
    std::cout << "\n--- Test 3: Medium GEMM (128×128×128) with rocWMMA ---\n";
    totalTests++;
    if (runGemmTest(128, 128, 128, true)) passedTests++;

    // Test 4: Non-square GEMM
    std::cout << "\n--- Test 4: Non-square GEMM (64×128×64) with rocWMMA ---\n";
    totalTests++;
    if (runGemmTest(64, 128, 64, true)) passedTests++;

    // Test 5: Larger K dimension
    std::cout << "\n--- Test 5: Large K GEMM (32×32×256) with rocWMMA ---\n";
    totalTests++;
    if (runGemmTest(32, 32, 256, true)) passedTests++;

    // Test 6: FlashMoE typical size
    std::cout << "\n--- Test 6: FlashMoE typical (128×256×128) with rocWMMA ---\n";
    totalTests++;
    if (runGemmTest(128, 256, 128, true)) passedTests++;

    // Summary
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║                      Test Summary                        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    std::cout << "  Passed: " << passedTests << " / " << totalTests << "\n";

    if (passedTests == totalTests) {
        std::cout << "\n  \033[32m✓ ALL TESTS PASSED\033[0m\n\n";
        return 0;
    } else {
        std::cout << "\n  \033[31m✗ SOME TESTS FAILED\033[0m\n\n";
        return 1;
    }
}
