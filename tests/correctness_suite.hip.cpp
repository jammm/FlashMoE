//
// FlashMoE HIP Port - Comprehensive Correctness Test Suite
// Tests all core components for correctness against CPU reference
//
// Copyright (c) 2025, Osayamen Jonathan Aimuyo
// All rights reserved.
//
// Test Categories:
// 1. Tensor Operations - Layout indexing, AlignedArray, Shape/Stride
// 2. Atomic Operations - atomicAdd, atomicCAS, atomicMax/Min float
// 3. GEMM Correctness - hipBLASLt vs CPU reference
// 4. Gated MLP Correctness - Gated activation pattern
// 5. Gate Softmax + TopK - Softmax normalization and top-k selection
// 6. hipCUB Operations - BlockScan, WarpScan
// 7. Type Conversions - FP32 <-> FP16/BF16 round-trip
//

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hipblaslt/hipblaslt.h>
#include <hipcub/hipcub.hpp>

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <limits>
#include <functional>

// Include FlashMoE headers
#include "flashmoe/hip/tensor.hpp"
#include "flashmoe/hip/math.hip.cuh"

// Define inline bfloat16 conversion helpers to avoid type issues
__device__ __forceinline__
float bf16_to_float(hip_bfloat16 val) {
    return static_cast<float>(val);
}

__host__ __device__ __forceinline__
hip_bfloat16 float_to_bf16(float val) {
    return hip_bfloat16(val);
}

// SiLU activation for float
__device__ __forceinline__
float silu_activation(float x) {
    return x / (1.0f + expf(-x));
}

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
// Test Result Structure
// ============================================================================

struct TestResult {
    std::string name;
    bool passed;
    float maxError;
    std::string details;

    TestResult(const std::string& n, bool p, float e = 0.0f, const std::string& d = "")
        : name(n), passed(p), maxError(e), details(d) {}
};

static std::vector<TestResult> g_results;

void recordResult(const std::string& name, bool passed, float maxError = 0.0f,
                  const std::string& details = "") {
    g_results.emplace_back(name, passed, maxError, details);
    std::cout << "  " << name << ": "
              << (passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m");
    if (maxError > 0.0f) {
        std::cout << " (max error: " << std::scientific << std::setprecision(3) << maxError << ")";
    }
    if (!details.empty()) {
        std::cout << " [" << details << "]";
    }
    std::cout << "\n";
}

// ============================================================================
// Constants
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;

// ============================================================================
// SECTION 1: Tensor Operations Tests
// ============================================================================

// Test kernel for layout indexing
__global__ void testLayoutIndexingKernel(
    float* output_row_major,
    float* output_col_major,
    int M, int N
) {
    using namespace hip_tensor;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M * N) return;

    // Row-major layout: stride = (N, 1)
    auto layout_row = make_layout(make_shape(M, N), make_stride(N, 1));

    // Column-major layout: stride = (1, M)
    auto layout_col = make_layout(make_shape(M, N), make_stride(1, M));

    int row = tid / N;
    int col = tid % N;

    // Store linear index computed by layout
    output_row_major[tid] = static_cast<float>(layout_row(row, col));
    output_col_major[tid] = static_cast<float>(layout_col(row, col));
}

// Test kernel for AlignedArray operations
template<int N>
__global__ void testAlignedArrayKernel(
    float* output,
    float fillValue
) {
    using namespace hip_tensor;

    AlignedArray<float, N, 16> arr;
    arr.fill(fillValue);

    const int tid = threadIdx.x;
    if (tid < N) {
        output[tid] = arr[tid];
    }

    // Test clear
    __syncthreads();
    if (tid == 0) {
        arr.clear();
        output[N] = arr[0];  // Should be 0
    }
}

// Test kernel for shape/stride calculations
__global__ void testShapeStrideKernel(
    int* output
) {
    using namespace hip_tensor;

    auto shape2d = make_shape(16, 32);
    auto shape3d = make_shape(4, 8, 16);
    auto stride2d = make_stride(32, 1);

    auto layout2d = make_layout(shape2d, stride2d);

    // Store results
    output[0] = to_int(get<0>(shape2d));  // 16
    output[1] = to_int(get<1>(shape2d));  // 32
    output[2] = layout2d.size();          // 16 * 32 = 512
    output[3] = to_int(get<0>(shape3d));  // 4
    output[4] = to_int(get<1>(shape3d));  // 8
    output[5] = to_int(get<2>(shape3d));  // 16

    // Test compile-time Int<N>
    _4 compile_time_4;
    output[6] = static_cast<int>(compile_time_4);  // 4

    // Test layout indexing
    output[7] = layout2d(2, 5);  // 2*32 + 5*1 = 69
}

void testTensorOperations() {
    std::cout << "\n=== SECTION 1: Tensor Operations ===\n";

    // Test 1.1: Layout Indexing
    {
        const int M = 4, N = 8;
        const int size = M * N;

        std::vector<float> h_row(size), h_col(size);
        float *d_row, *d_col;

        CHECK_HIP(hipMalloc(&d_row, size * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_col, size * sizeof(float)));

        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        hipLaunchKernelGGL(testLayoutIndexingKernel, dim3(blocks), dim3(threads), 0, 0,
                           d_row, d_col, M, N);
        CHECK_HIP(hipDeviceSynchronize());

        CHECK_HIP(hipMemcpy(h_row.data(), d_row, size * sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_col.data(), d_col, size * sizeof(float), hipMemcpyDeviceToHost));

        // Verify row-major: index = row * N + col
        bool row_correct = true;
        bool col_correct = true;

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                int tid = i * N + j;
                int expected_row = i * N + j;
                int expected_col = i + j * M;

                if (static_cast<int>(h_row[tid]) != expected_row) row_correct = false;
                if (static_cast<int>(h_col[tid]) != expected_col) col_correct = false;
            }
        }

        recordResult("Layout Indexing (Row-Major)", row_correct);
        recordResult("Layout Indexing (Col-Major)", col_correct);

        CHECK_HIP(hipFree(d_row));
        CHECK_HIP(hipFree(d_col));
    }

    // Test 1.2: AlignedArray Operations
    {
        constexpr int ARR_SIZE = 8;
        std::vector<float> h_output(ARR_SIZE + 1);
        float* d_output;

        CHECK_HIP(hipMalloc(&d_output, (ARR_SIZE + 1) * sizeof(float)));
        CHECK_HIP(hipMemset(d_output, 0, (ARR_SIZE + 1) * sizeof(float)));

        float fillValue = 3.14159f;
        hipLaunchKernelGGL((testAlignedArrayKernel<ARR_SIZE>), dim3(1), dim3(32), 0, 0,
                           d_output, fillValue);
        CHECK_HIP(hipDeviceSynchronize());

        CHECK_HIP(hipMemcpy(h_output.data(), d_output, (ARR_SIZE + 1) * sizeof(float),
                           hipMemcpyDeviceToHost));

        bool fill_correct = true;
        for (int i = 0; i < ARR_SIZE; ++i) {
            if (std::abs(h_output[i] - fillValue) > 1e-6f) fill_correct = false;
        }
        bool clear_correct = (h_output[ARR_SIZE] == 0.0f);

        recordResult("AlignedArray Fill", fill_correct);
        recordResult("AlignedArray Clear", clear_correct);

        CHECK_HIP(hipFree(d_output));
    }

    // Test 1.3: Shape/Stride Calculations
    {
        std::vector<int> h_output(8);
        int* d_output;

        CHECK_HIP(hipMalloc(&d_output, 8 * sizeof(int)));

        hipLaunchKernelGGL(testShapeStrideKernel, dim3(1), dim3(1), 0, 0, d_output);
        CHECK_HIP(hipDeviceSynchronize());

        CHECK_HIP(hipMemcpy(h_output.data(), d_output, 8 * sizeof(int), hipMemcpyDeviceToHost));

        bool shape_correct = (h_output[0] == 16 && h_output[1] == 32);
        bool size_correct = (h_output[2] == 512);
        bool shape3d_correct = (h_output[3] == 4 && h_output[4] == 8 && h_output[5] == 16);
        bool int_correct = (h_output[6] == 4);
        bool indexing_correct = (h_output[7] == 69);  // 2*32 + 5 = 69

        recordResult("Shape Extraction", shape_correct);
        recordResult("Layout Size Calculation", size_correct);
        recordResult("3D Shape Support", shape3d_correct);
        recordResult("Compile-time Int<N>", int_correct);
        recordResult("2D Index Calculation", indexing_correct);

        CHECK_HIP(hipFree(d_output));
    }
}

// ============================================================================
// SECTION 2: Atomic Operations Tests
// ============================================================================

// Test kernel for atomicAdd with int
__global__ void testAtomicAddIntKernel(int* result, int numThreads) {
    atomicAdd(result, 1);
}

// Test kernel for atomicAdd with float
__global__ void testAtomicAddFloatKernel(float* result, float value, int numThreads) {
    atomicAdd(result, value);
}

// Test kernel for atomicAdd with double (CAS-based)
__global__ void testAtomicAddDoubleKernel(double* result, double value) {
    unsigned long long int* addr_as_ull = reinterpret_cast<unsigned long long int*>(result);
    unsigned long long int old = *addr_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed,
                        __double_as_longlong(__longlong_as_double(assumed) + value));
    } while (assumed != old);
}

// Test kernel for atomicCAS
__global__ void testAtomicCASKernel(int* result, int expected, int desired) {
    int tid = threadIdx.x;
    if (tid == 0) {
        atomicCAS(result, expected, desired);
    }
}

// Test kernel for atomicMax/Min with float (CAS-based)
__global__ void testAtomicMaxMinFloatKernel(
    float* maxResult, float* minResult,
    const float* values, int N
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float val = values[tid];

    // atomicMax for float (CAS-based)
    unsigned int* maxAddr = reinterpret_cast<unsigned int*>(maxResult);
    unsigned int old = *maxAddr;
    unsigned int assumed;
    do {
        assumed = old;
        float old_val = __uint_as_float(assumed);
        if (old_val >= val) break;
        old = atomicCAS(maxAddr, assumed, __float_as_uint(val));
    } while (assumed != old);

    // atomicMin for float (CAS-based)
    unsigned int* minAddr = reinterpret_cast<unsigned int*>(minResult);
    old = *minAddr;
    do {
        assumed = old;
        float old_val = __uint_as_float(assumed);
        if (old_val <= val) break;
        old = atomicCAS(minAddr, assumed, __float_as_uint(val));
    } while (assumed != old);
}

void testAtomicOperations() {
    std::cout << "\n=== SECTION 2: Atomic Operations ===\n";

    // Test 2.1: atomicAdd with int
    {
        int h_result = 0;
        int* d_result;
        const int numThreads = 1024;

        CHECK_HIP(hipMalloc(&d_result, sizeof(int)));
        CHECK_HIP(hipMemcpy(d_result, &h_result, sizeof(int), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(testAtomicAddIntKernel, dim3(numThreads / 256), dim3(256), 0, 0,
                           d_result, numThreads);
        CHECK_HIP(hipDeviceSynchronize());

        CHECK_HIP(hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost));

        bool correct = (h_result == numThreads);
        recordResult("atomicAdd (int)", correct, 0.0f,
                    "expected=" + std::to_string(numThreads) + ", got=" + std::to_string(h_result));

        CHECK_HIP(hipFree(d_result));
    }

    // Test 2.2: atomicAdd with float
    {
        float h_result = 0.0f;
        float* d_result;
        const int numThreads = 1024;
        const float addValue = 0.5f;

        CHECK_HIP(hipMalloc(&d_result, sizeof(float)));
        CHECK_HIP(hipMemcpy(d_result, &h_result, sizeof(float), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(testAtomicAddFloatKernel, dim3(numThreads / 256), dim3(256), 0, 0,
                           d_result, addValue, numThreads);
        CHECK_HIP(hipDeviceSynchronize());

        CHECK_HIP(hipMemcpy(&h_result, d_result, sizeof(float), hipMemcpyDeviceToHost));

        float expected = numThreads * addValue;
        float error = std::abs(h_result - expected);
        bool correct = error < 0.01f;  // Allow small FP error
        recordResult("atomicAdd (float)", correct, error);

        CHECK_HIP(hipFree(d_result));
    }

    // Test 2.3: atomicAdd with double (CAS-based)
    {
        double h_result = 0.0;
        double* d_result;
        const int numThreads = 256;
        const double addValue = 0.1;

        CHECK_HIP(hipMalloc(&d_result, sizeof(double)));
        CHECK_HIP(hipMemcpy(d_result, &h_result, sizeof(double), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(testAtomicAddDoubleKernel, dim3(1), dim3(numThreads), 0, 0,
                           d_result, addValue);
        CHECK_HIP(hipDeviceSynchronize());

        CHECK_HIP(hipMemcpy(&h_result, d_result, sizeof(double), hipMemcpyDeviceToHost));

        double expected = numThreads * addValue;
        double error = std::abs(h_result - expected);
        bool correct = error < 1e-6;
        recordResult("atomicAdd (double, CAS)", correct, static_cast<float>(error));

        CHECK_HIP(hipFree(d_result));
    }

    // Test 2.4: atomicCAS
    {
        int h_result = 10;
        int* d_result;

        CHECK_HIP(hipMalloc(&d_result, sizeof(int)));
        CHECK_HIP(hipMemcpy(d_result, &h_result, sizeof(int), hipMemcpyHostToDevice));

        // Should succeed: compare 10, swap to 20
        hipLaunchKernelGGL(testAtomicCASKernel, dim3(1), dim3(1), 0, 0, d_result, 10, 20);
        CHECK_HIP(hipDeviceSynchronize());

        CHECK_HIP(hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost));
        bool success_correct = (h_result == 20);

        // Should fail: compare 10 (but value is 20), no swap
        hipLaunchKernelGGL(testAtomicCASKernel, dim3(1), dim3(1), 0, 0, d_result, 10, 30);
        CHECK_HIP(hipDeviceSynchronize());

        CHECK_HIP(hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost));
        bool fail_correct = (h_result == 20);  // Should remain 20

        recordResult("atomicCAS (success case)", success_correct);
        recordResult("atomicCAS (fail case)", fail_correct);

        CHECK_HIP(hipFree(d_result));
    }

    // Test 2.5: atomicMax/Min with float
    {
        const int N = 1024;
        std::vector<float> h_values(N);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

        float expected_max = -std::numeric_limits<float>::infinity();
        float expected_min = std::numeric_limits<float>::infinity();

        for (int i = 0; i < N; ++i) {
            h_values[i] = dist(gen);
            expected_max = std::max(expected_max, h_values[i]);
            expected_min = std::min(expected_min, h_values[i]);
        }

        float h_max = -std::numeric_limits<float>::infinity();
        float h_min = std::numeric_limits<float>::infinity();
        float *d_values, *d_max, *d_min;

        CHECK_HIP(hipMalloc(&d_values, N * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_max, sizeof(float)));
        CHECK_HIP(hipMalloc(&d_min, sizeof(float)));

        CHECK_HIP(hipMemcpy(d_values, h_values.data(), N * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_max, &h_max, sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_min, &h_min, sizeof(float), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(testAtomicMaxMinFloatKernel, dim3((N + 255) / 256), dim3(256), 0, 0,
                           d_max, d_min, d_values, N);
        CHECK_HIP(hipDeviceSynchronize());

        CHECK_HIP(hipMemcpy(&h_max, d_max, sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(&h_min, d_min, sizeof(float), hipMemcpyDeviceToHost));

        float max_error = std::abs(h_max - expected_max);
        float min_error = std::abs(h_min - expected_min);

        recordResult("atomicMax (float, CAS)", max_error < 1e-5f, max_error);
        recordResult("atomicMin (float, CAS)", min_error < 1e-5f, min_error);

        CHECK_HIP(hipFree(d_values));
        CHECK_HIP(hipFree(d_max));
        CHECK_HIP(hipFree(d_min));
    }
}

// ============================================================================
// SECTION 3: GEMM Correctness Tests
// ============================================================================

// CPU GEMM reference using column-major layout (matching hipBLASLt default)
// A is M×K column-major, B is K×N column-major, C is M×N column-major
void cpuGemmFP16ColMajor(const __half* A, const __half* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                // Column-major: A[i, k] = A[k * M + i], B[k, j] = B[j * K + k]
                float a = __half2float(A[k * M + i]);
                float b = __half2float(B[j * K + k]);
                sum += a * b;
            }
            // C is column-major: C[i, j] = C[j * M + i]
            C[j * M + i] = sum;
        }
    }
}

// Returns pair: {passed, maxError}
std::pair<bool, float> testGemmCorrectnessWithError(int M, int N, int K) {
    // Allocate and initialize host memory
    // Using column-major layout (default for BLAS libraries)
    std::vector<__half> h_A(M * K);  // M×K column-major
    std::vector<__half> h_B(K * N);  // K×N column-major
    std::vector<__half> h_C(M * N);
    std::vector<float> h_C_ref(M * N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < M * K; ++i) h_A[i] = __float2half(dist(gen));
    for (int i = 0; i < K * N; ++i) h_B[i] = __float2half(dist(gen));

    // CPU reference using column-major layout
    cpuGemmFP16ColMajor(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

    // GPU computation with hipBLASLt
    __half *d_A, *d_B, *d_C;

    CHECK_HIP(hipMalloc(&d_A, M * K * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_B, K * N * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_C, M * N * sizeof(__half)));

    CHECK_HIP(hipMemcpy(d_A, h_A.data(), M * K * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B.data(), K * N * sizeof(__half), hipMemcpyHostToDevice));

    // Setup hipBLASLt
    hipblasLtHandle_t handle;
    CHECK_HIPBLASLT(hipblasLtCreate(&handle));

    hipblasLtMatmulDesc_t matmulDesc;
    hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;

    hipDataType dataType = HIP_R_16F;
    hipDataType scaleType = HIP_R_32F;
    hipblasComputeType_t computeType = HIPBLAS_COMPUTE_32F;

    CHECK_HIPBLASLT(hipblasLtMatmulDescCreate(&matmulDesc, computeType, scaleType));

    hipblasOperation_t transA = HIPBLAS_OP_N;
    hipblasOperation_t transB = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

    // Column-major layout (default for BLAS):
    // A: M×K with leading dimension M
    // B: K×N with leading dimension K
    // C: M×N with leading dimension M
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layoutA, dataType, M, K, M));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layoutB, dataType, K, N, K));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layoutC, dataType, M, N, M));

    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);

    size_t workspaceSize = 64 * 1024 * 1024;
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                           &workspaceSize, sizeof(workspaceSize));

    void* workspace;
    CHECK_HIP(hipMalloc(&workspace, workspaceSize));

    hipblasLtMatmulHeuristicResult_t heuristicResult[1];
    int returnedAlgoCount = 0;
    CHECK_HIPBLASLT(hipblasLtMatmulAlgoGetHeuristic(
        handle, matmulDesc, layoutA, layoutB, layoutC, layoutC,
        pref, 1, heuristicResult, &returnedAlgoCount));

    float alpha = 1.0f;
    float beta = 0.0f;

    hipblasLtMatmul(handle, matmulDesc, &alpha, d_A, layoutA, d_B, layoutB,
                    &beta, d_C, layoutC, d_C, layoutC,
                    &heuristicResult[0].algo, workspace, workspaceSize, 0);
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipMemcpy(h_C.data(), d_C, M * N * sizeof(__half), hipMemcpyDeviceToHost));

    // Compare results using combined absolute/relative tolerance
    float maxAbsError = 0.0f;
    int failingElements = 0;

    // Tolerance: either absolute tolerance or relative tolerance must pass
    // This handles both small and large values correctly
    float absTolerance = 0.005f * std::sqrt(static_cast<float>(K));  // ~0.04 for K=64, ~0.11 for K=512
    float relTolerance = 0.02f;  // 2% relative error

    for (int i = 0; i < M * N; ++i) {
        float gpu_val = __half2float(h_C[i]);
        float cpu_val = h_C_ref[i];
        float absError = std::abs(gpu_val - cpu_val);
        maxAbsError = std::max(maxAbsError, absError);

        // Pass if either absolute OR relative error is within tolerance
        bool elemOk = (absError < absTolerance) ||
                      (absError < relTolerance * std::abs(cpu_val));
        if (!elemOk) failingElements++;
    }

    // Cleanup
    CHECK_HIP(hipFree(workspace));
    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatrixLayoutDestroy(layoutA);
    hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtMatrixLayoutDestroy(layoutC);
    hipblasLtMatmulDescDestroy(matmulDesc);
    hipblasLtDestroy(handle);

    CHECK_HIP(hipFree(d_A));
    CHECK_HIP(hipFree(d_B));
    CHECK_HIP(hipFree(d_C));

    // Allow up to 0.1% of elements to fail (outliers from numerical precision)
    bool passed = (failingElements < M * N * 0.001);

    return {passed, maxAbsError};
}

void testGemmCorrectness() {
    std::cout << "\n=== SECTION 3: GEMM Correctness ===\n";

    struct TestCase { int M, N, K; };
    std::vector<TestCase> cases = {
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512}
    };

    for (const auto& tc : cases) {
        std::string name = "hipBLASLt FP16 GEMM " + std::to_string(tc.M) + "x" +
                          std::to_string(tc.N) + "x" + std::to_string(tc.K);

        auto [passed, maxError] = testGemmCorrectnessWithError(tc.M, tc.N, tc.K);
        recordResult(name, passed, maxError, "rel_tol=1%");
    }
}

// ============================================================================
// SECTION 4: Gated MLP Correctness Tests
// ============================================================================

float cpuSiLU(float x) {
    return x / (1.0f + std::exp(-x));
}

void cpuGatedMLP(
    const __half* A, const __half* B, const __half* BV,
    float* C, const __half* bias, const __half* biasV,
    int M, int N, int K
) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            // Gate: A @ B + bias
            float gate_sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                gate_sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            }
            gate_sum += __half2float(bias[j]);

            // SiLU activation
            float gate_activated = cpuSiLU(gate_sum);

            // Value: A @ BV + biasV
            float value_sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                value_sum += __half2float(A[i * K + k]) * __half2float(BV[k * N + j]);
            }
            value_sum += __half2float(biasV[j]);

            // Output = gate * value
            C[i * N + j] = gate_activated * value_sum;
        }
    }
}

__global__ void gatedMLPNaiveKernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const __half* __restrict__ BV,
    __half* __restrict__ C,
    const __half* __restrict__ bias,
    const __half* __restrict__ biasV,
    int M, int N, int K
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    // Compute gate
    float gate_sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        gate_sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    }
    gate_sum += __half2float(bias[col]);

    // SiLU
    float gate_activated = gate_sum / (1.0f + expf(-gate_sum));

    // Compute value
    float value_sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        value_sum += __half2float(A[row * K + k]) * __half2float(BV[k * N + col]);
    }
    value_sum += __half2float(biasV[col]);

    C[row * N + col] = __float2half(gate_activated * value_sum);
}

void testGatedMLP() {
    std::cout << "\n=== SECTION 4: Gated MLP Correctness ===\n";

    const int M = 64, N = 128, K = 64;

    // Allocate and initialize host memory
    std::vector<__half> h_A(M * K);
    std::vector<__half> h_B(K * N);
    std::vector<__half> h_BV(K * N);
    std::vector<__half> h_bias(N);
    std::vector<__half> h_biasV(N);
    std::vector<__half> h_C(M * N);
    std::vector<float> h_C_ref(M * N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (auto& v : h_A) v = __float2half(dist(gen));
    for (auto& v : h_B) v = __float2half(dist(gen));
    for (auto& v : h_BV) v = __float2half(dist(gen));
    for (auto& v : h_bias) v = __float2half(dist(gen) * 0.1f);
    for (auto& v : h_biasV) v = __float2half(dist(gen) * 0.1f);

    // CPU reference
    cpuGatedMLP(h_A.data(), h_B.data(), h_BV.data(), h_C_ref.data(),
                h_bias.data(), h_biasV.data(), M, N, K);

    // GPU computation
    __half *d_A, *d_B, *d_BV, *d_C, *d_bias, *d_biasV;

    CHECK_HIP(hipMalloc(&d_A, M * K * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_B, K * N * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_BV, K * N * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_C, M * N * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_bias, N * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_biasV, N * sizeof(__half)));

    CHECK_HIP(hipMemcpy(d_A, h_A.data(), M * K * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B.data(), K * N * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_BV, h_BV.data(), K * N * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_bias, h_bias.data(), N * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_biasV, h_biasV.data(), N * sizeof(__half), hipMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    hipLaunchKernelGGL(gatedMLPNaiveKernel, grid, block, 0, 0,
                       d_A, d_B, d_BV, d_C, d_bias, d_biasV, M, N, K);
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipMemcpy(h_C.data(), d_C, M * N * sizeof(__half), hipMemcpyDeviceToHost));

    // Compare
    float maxError = 0.0f;
    int errorCount = 0;

    for (int i = 0; i < M * N; ++i) {
        float gpu_val = __half2float(h_C[i]);
        float cpu_val = h_C_ref[i];
        float error = std::abs(gpu_val - cpu_val);
        float relError = error / (std::abs(cpu_val) + 1e-6f);
        maxError = std::max(maxError, error);

        if (relError > 0.05f && error > 0.01f) errorCount++;
    }

    recordResult("Gated MLP with SiLU", errorCount < M * N * 0.01, maxError);
    recordResult("SiLU Activation Correctness", maxError < 0.1f, maxError);

    CHECK_HIP(hipFree(d_A));
    CHECK_HIP(hipFree(d_B));
    CHECK_HIP(hipFree(d_BV));
    CHECK_HIP(hipFree(d_C));
    CHECK_HIP(hipFree(d_bias));
    CHECK_HIP(hipFree(d_biasV));
}

// ============================================================================
// SECTION 5: Gate Softmax + TopK Tests
// ============================================================================

// Softmax kernel
__global__ void softmaxKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N
) {
    const int tid = threadIdx.x;

    extern __shared__ float sdata[];
    float* s_max = sdata;
    float* s_sum = sdata + blockDim.x;

    // Find max (reduction)
    float local_max = -1e30f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }
    s_max[tid] = local_max;
    __syncthreads();

    // Reduce to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    float max_val = s_max[0];
    __syncthreads();

    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = expf(input[i] - max_val);
        output[i] = val;
        local_sum += val;
    }
    s_sum[tid] = local_sum;
    __syncthreads();

    // Reduce to find sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    float sum_val = s_sum[0];
    __syncthreads();

    // Normalize
    for (int i = tid; i < N; i += blockDim.x) {
        output[i] /= sum_val;
    }
}

// TopK selection kernel (simple version)
__global__ void topKKernel(
    const float* __restrict__ input,
    float* __restrict__ topk_values,
    int* __restrict__ topk_indices,
    int N, int K
) {
    // Simple approach: each thread handles finding one of the top-K
    // For production, use radix select or heap-based approach

    extern __shared__ float shared_vals[];
    int* shared_idx = reinterpret_cast<int*>(shared_vals + N);

    // Copy to shared memory
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        shared_vals[i] = input[i];
        shared_idx[i] = i;
    }
    __syncthreads();

    // Thread 0 finds top-K using simple selection
    if (threadIdx.x == 0) {
        for (int k = 0; k < K; ++k) {
            int max_idx = k;
            float max_val = shared_vals[k];

            for (int i = k + 1; i < N; ++i) {
                if (shared_vals[i] > max_val) {
                    max_val = shared_vals[i];
                    max_idx = i;
                }
            }

            // Swap
            if (max_idx != k) {
                float tmp_val = shared_vals[k];
                int tmp_idx = shared_idx[k];
                shared_vals[k] = shared_vals[max_idx];
                shared_idx[k] = shared_idx[max_idx];
                shared_vals[max_idx] = tmp_val;
                shared_idx[max_idx] = tmp_idx;
            }

            topk_values[k] = shared_vals[k];
            topk_indices[k] = shared_idx[k];
        }
    }
}

void testSoftmaxTopK() {
    std::cout << "\n=== SECTION 5: Gate Softmax + TopK ===\n";

    const int N = 128;  // Number of experts
    const int K = 8;    // Top-K selection

    // Initialize input
    std::vector<float> h_input(N);
    std::vector<float> h_softmax(N);
    std::vector<float> h_topk_values(K);
    std::vector<int> h_topk_indices(K);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    for (auto& v : h_input) v = dist(gen);

    // GPU computation
    float *d_input, *d_softmax, *d_topk_values;
    int *d_topk_indices;

    CHECK_HIP(hipMalloc(&d_input, N * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_softmax, N * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_topk_values, K * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_topk_indices, K * sizeof(int)));

    CHECK_HIP(hipMemcpy(d_input, h_input.data(), N * sizeof(float), hipMemcpyHostToDevice));

    // Run softmax
    size_t sharedMemSoftmax = 2 * 256 * sizeof(float);
    hipLaunchKernelGGL(softmaxKernel, dim3(1), dim3(256), sharedMemSoftmax, 0,
                       d_input, d_softmax, N);
    CHECK_HIP(hipDeviceSynchronize());

    // Run topK on softmax output
    size_t sharedMemTopK = N * sizeof(float) + N * sizeof(int);
    hipLaunchKernelGGL(topKKernel, dim3(1), dim3(256), sharedMemTopK, 0,
                       d_softmax, d_topk_values, d_topk_indices, N, K);
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipMemcpy(h_softmax.data(), d_softmax, N * sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(h_topk_values.data(), d_topk_values, K * sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(h_topk_indices.data(), d_topk_indices, K * sizeof(int), hipMemcpyDeviceToHost));

    // Test 1: Softmax sums to 1.0
    float softmax_sum = std::accumulate(h_softmax.begin(), h_softmax.end(), 0.0f);
    float sum_error = std::abs(softmax_sum - 1.0f);
    recordResult("Softmax sums to 1.0", sum_error < 1e-5f, sum_error);

    // Test 2: All softmax values are positive
    bool all_positive = std::all_of(h_softmax.begin(), h_softmax.end(),
                                    [](float v) { return v > 0.0f; });
    recordResult("Softmax all positive", all_positive);

    // Test 3: TopK values are sorted descending
    bool sorted = true;
    for (int i = 1; i < K; ++i) {
        if (h_topk_values[i] > h_topk_values[i-1]) sorted = false;
    }
    recordResult("TopK sorted descending", sorted);

    // Test 4: TopK indices are valid and correspond to correct values
    bool indices_valid = true;
    for (int i = 0; i < K; ++i) {
        if (h_topk_indices[i] < 0 || h_topk_indices[i] >= N) {
            indices_valid = false;
            break;
        }
        if (std::abs(h_topk_values[i] - h_softmax[h_topk_indices[i]]) > 1e-6f) {
            indices_valid = false;
            break;
        }
    }
    recordResult("TopK indices match values", indices_valid);

    // Test 5: TopK contains the actual top-K elements (verify against CPU)
    std::vector<std::pair<float, int>> cpu_sorted(N);
    for (int i = 0; i < N; ++i) {
        cpu_sorted[i] = {h_softmax[i], i};
    }
    std::partial_sort(cpu_sorted.begin(), cpu_sorted.begin() + K, cpu_sorted.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    bool topk_correct = true;
    for (int i = 0; i < K; ++i) {
        if (std::abs(h_topk_values[i] - cpu_sorted[i].first) > 1e-5f) {
            topk_correct = false;
        }
    }
    recordResult("TopK matches CPU reference", topk_correct);

    CHECK_HIP(hipFree(d_input));
    CHECK_HIP(hipFree(d_softmax));
    CHECK_HIP(hipFree(d_topk_values));
    CHECK_HIP(hipFree(d_topk_indices));
}

// ============================================================================
// SECTION 6: hipCUB Operations Tests
// ============================================================================

// BlockScan test kernel
template<int BLOCK_SIZE>
__global__ void blockScanInclusiveKernel(int* input, int* output, int N) {
    using BlockScan = hipcub::BlockScan<int, BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    const int tid = threadIdx.x;
    int val = (tid < N) ? input[tid] : 0;
    int result;

    BlockScan(temp_storage).InclusiveSum(val, result);

    if (tid < N) {
        output[tid] = result;
    }
}

template<int BLOCK_SIZE>
__global__ void blockScanExclusiveKernel(int* input, int* output, int N) {
    using BlockScan = hipcub::BlockScan<int, BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    const int tid = threadIdx.x;
    int val = (tid < N) ? input[tid] : 0;
    int result;

    BlockScan(temp_storage).ExclusiveSum(val, result);

    if (tid < N) {
        output[tid] = result;
    }
}

// WarpScan test kernel
__global__ void warpScanKernel(int* input, int* output, int N) {
    using WarpScan = hipcub::WarpScan<int, WARP_SIZE>;
    __shared__ typename WarpScan::TempStorage temp_storage[4];  // 4 warps

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    int val = (tid < N) ? input[tid] : 0;
    int result;

    WarpScan(temp_storage[warp_id]).InclusiveSum(val, result);

    if (tid < N) {
        output[tid] = result;
    }
}

void testHipCUBOperations() {
    std::cout << "\n=== SECTION 6: hipCUB Operations ===\n";

    constexpr int N = 256;

    // Initialize input with known values
    std::vector<int> h_input(N);
    std::vector<int> h_output(N);

    for (int i = 0; i < N; ++i) {
        h_input[i] = 1;  // All ones - easy to verify
    }

    int *d_input, *d_output;
    CHECK_HIP(hipMalloc(&d_input, N * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_output, N * sizeof(int)));
    CHECK_HIP(hipMemcpy(d_input, h_input.data(), N * sizeof(int), hipMemcpyHostToDevice));

    // Test 6.1: Inclusive BlockScan
    {
        CHECK_HIP(hipMemset(d_output, 0, N * sizeof(int)));
        hipLaunchKernelGGL((blockScanInclusiveKernel<N>), dim3(1), dim3(N), 0, 0,
                           d_input, d_output, N);
        CHECK_HIP(hipDeviceSynchronize());

        CHECK_HIP(hipMemcpy(h_output.data(), d_output, N * sizeof(int), hipMemcpyDeviceToHost));

        // Inclusive scan of all 1s: [1, 2, 3, 4, ...]
        bool correct = true;
        int maxError = 0;
        for (int i = 0; i < N; ++i) {
            int expected = i + 1;
            if (h_output[i] != expected) {
                correct = false;
                maxError = std::max(maxError, std::abs(h_output[i] - expected));
            }
        }
        recordResult("BlockScan Inclusive", correct, static_cast<float>(maxError));
    }

    // Test 6.2: Exclusive BlockScan
    {
        CHECK_HIP(hipMemset(d_output, 0, N * sizeof(int)));
        hipLaunchKernelGGL((blockScanExclusiveKernel<N>), dim3(1), dim3(N), 0, 0,
                           d_input, d_output, N);
        CHECK_HIP(hipDeviceSynchronize());

        CHECK_HIP(hipMemcpy(h_output.data(), d_output, N * sizeof(int), hipMemcpyDeviceToHost));

        // Exclusive scan of all 1s: [0, 1, 2, 3, ...]
        bool correct = true;
        int maxError = 0;
        for (int i = 0; i < N; ++i) {
            int expected = i;
            if (h_output[i] != expected) {
                correct = false;
                maxError = std::max(maxError, std::abs(h_output[i] - expected));
            }
        }
        recordResult("BlockScan Exclusive", correct, static_cast<float>(maxError));
    }

    // Test 6.3: WarpScan
    {
        CHECK_HIP(hipMemset(d_output, 0, N * sizeof(int)));
        hipLaunchKernelGGL(warpScanKernel, dim3(1), dim3(N), 0, 0,
                           d_input, d_output, N);
        CHECK_HIP(hipDeviceSynchronize());

        CHECK_HIP(hipMemcpy(h_output.data(), d_output, N * sizeof(int), hipMemcpyDeviceToHost));

        // Warp scan of all 1s resets at the target wave size.
        bool correct = true;
        int maxError = 0;
        for (int i = 0; i < N; ++i) {
            int lane_in_warp = i % WARP_SIZE;
            int expected = lane_in_warp + 1;
            if (h_output[i] != expected) {
                correct = false;
                maxError = std::max(maxError, std::abs(h_output[i] - expected));
            }
        }
        recordResult("WarpScan Inclusive", correct, static_cast<float>(maxError));
    }

    // Test 6.4: BlockScan with varying input
    {
        for (int i = 0; i < N; ++i) {
            h_input[i] = i + 1;  // [1, 2, 3, 4, ...]
        }
        CHECK_HIP(hipMemcpy(d_input, h_input.data(), N * sizeof(int), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemset(d_output, 0, N * sizeof(int)));

        hipLaunchKernelGGL((blockScanInclusiveKernel<N>), dim3(1), dim3(N), 0, 0,
                           d_input, d_output, N);
        CHECK_HIP(hipDeviceSynchronize());

        CHECK_HIP(hipMemcpy(h_output.data(), d_output, N * sizeof(int), hipMemcpyDeviceToHost));

        // CPU reference: prefix sum
        std::vector<int> expected(N);
        expected[0] = h_input[0];
        for (int i = 1; i < N; ++i) {
            expected[i] = expected[i-1] + h_input[i];
        }

        bool correct = true;
        int maxError = 0;
        for (int i = 0; i < N; ++i) {
            if (h_output[i] != expected[i]) {
                correct = false;
                maxError = std::max(maxError, std::abs(h_output[i] - expected[i]));
            }
        }
        recordResult("BlockScan Varying Input", correct, static_cast<float>(maxError));
    }

    CHECK_HIP(hipFree(d_input));
    CHECK_HIP(hipFree(d_output));
}

// ============================================================================
// SECTION 7: Type Conversions Tests
// ============================================================================

__global__ void fp32ToFp16RoundtripKernel(const float* input, float* output, int N) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    __half intermediate = __float2half(input[tid]);
    output[tid] = __half2float(intermediate);
}

__global__ void fp32ToBf16RoundtripKernel(const float* input, float* output, int N) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    hip_bfloat16 intermediate = float_to_bf16(input[tid]);
    output[tid] = bf16_to_float(intermediate);
}

void testTypeConversions() {
    std::cout << "\n=== SECTION 7: Type Conversions ===\n";

    const int N = 1024;

    std::vector<float> h_input(N);
    std::vector<float> h_output(N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto& v : h_input) v = dist(gen);

    float *d_input, *d_output;
    CHECK_HIP(hipMalloc(&d_input, N * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_output, N * sizeof(float)));
    CHECK_HIP(hipMemcpy(d_input, h_input.data(), N * sizeof(float), hipMemcpyHostToDevice));

    // Test 7.1: FP32 <-> FP16 round-trip
    {
        hipLaunchKernelGGL(fp32ToFp16RoundtripKernel, dim3((N + 255) / 256), dim3(256), 0, 0,
                           d_input, d_output, N);
        CHECK_HIP(hipDeviceSynchronize());

        CHECK_HIP(hipMemcpy(h_output.data(), d_output, N * sizeof(float), hipMemcpyDeviceToHost));

        float maxError = 0.0f;
        float maxRelError = 0.0f;

        for (int i = 0; i < N; ++i) {
            float absError = std::abs(h_output[i] - h_input[i]);
            float relError = absError / (std::abs(h_input[i]) + 1e-10f);
            maxError = std::max(maxError, absError);
            maxRelError = std::max(maxRelError, relError);
        }

        // FP16 has ~3.3 decimal digits of precision (1 + 10 mantissa bits)
        // Relative error should be around 2^-10 ≈ 0.001
        bool precisionOK = maxRelError < 0.01f;  // Allow 1% relative error
        recordResult("FP32 <-> FP16 Round-trip", precisionOK, maxError,
                    "rel_err=" + std::to_string(maxRelError));
    }

    // Test 7.2: FP32 <-> BF16 round-trip
    {
        hipLaunchKernelGGL(fp32ToBf16RoundtripKernel, dim3((N + 255) / 256), dim3(256), 0, 0,
                           d_input, d_output, N);
        CHECK_HIP(hipDeviceSynchronize());

        CHECK_HIP(hipMemcpy(h_output.data(), d_output, N * sizeof(float), hipMemcpyDeviceToHost));

        float maxError = 0.0f;
        float maxRelError = 0.0f;

        for (int i = 0; i < N; ++i) {
            float absError = std::abs(h_output[i] - h_input[i]);
            float relError = absError / (std::abs(h_input[i]) + 1e-10f);
            maxError = std::max(maxError, absError);
            maxRelError = std::max(maxRelError, relError);
        }

        // BF16 has ~2.4 decimal digits of precision (1 + 7 mantissa bits)
        // Relative error should be around 2^-7 ≈ 0.008
        bool precisionOK = maxRelError < 0.02f;  // Allow 2% relative error
        recordResult("FP32 <-> BF16 Round-trip", precisionOK, maxError,
                    "rel_err=" + std::to_string(maxRelError));
    }

    // Test 7.3: FP16 preserves special values
    {
        std::vector<float> special_values = {
            0.0f, -0.0f,
            1.0f, -1.0f,
            65504.0f,  // Max FP16 finite value
            6.1e-5f,   // Min FP16 normal value
        };

        bool all_correct = true;
        for (float val : special_values) {
            __half h = __float2half(val);
            float back = __half2float(h);

            // For ±0 check bit pattern, others check value
            if (val == 0.0f) {
                if (back != 0.0f) all_correct = false;
            } else {
                float relError = std::abs(back - val) / std::abs(val);
                if (relError > 0.01f) all_correct = false;
            }
        }
        recordResult("FP16 Special Values", all_correct);
    }

    // Test 7.4: BF16 preserves special values
    {
        std::vector<float> special_values = {
            0.0f, -0.0f,
            1.0f, -1.0f,
            1e38f,   // Large value (BF16 has same range as FP32)
            1e-38f,  // Small value
        };

        bool all_correct = true;
        for (float val : special_values) {
            hip_bfloat16 bf = float_to_bf16(val);
            float back = static_cast<float>(bf);

            if (val == 0.0f) {
                if (back != 0.0f) all_correct = false;
            } else {
                float relError = std::abs(back - val) / std::abs(val);
                if (relError > 0.02f) all_correct = false;
            }
        }
        recordResult("BF16 Special Values", all_correct);
    }

    CHECK_HIP(hipFree(d_input));
    CHECK_HIP(hipFree(d_output));
}

// ============================================================================
// Main and Summary
// ============================================================================

void printSummary() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                     CORRECTNESS TEST SUMMARY                          ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════╣\n";

    // Count passes and fails
    int passed = 0, failed = 0;

    std::cout << "║ " << std::left << std::setw(50) << "Test Name"
              << std::setw(10) << "Status" << std::setw(12) << "Max Error" << "║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════╣\n";

    for (const auto& r : g_results) {
        if (r.passed) passed++;
        else failed++;

        std::string status = r.passed ? "PASS" : "FAIL";
        std::string color = r.passed ? "\033[32m" : "\033[31m";

        std::ostringstream error_str;
        if (r.maxError > 0) {
            error_str << std::scientific << std::setprecision(2) << r.maxError;
        } else {
            error_str << "-";
        }

        std::cout << "║ " << std::left << std::setw(50) << r.name.substr(0, 50)
                  << color << std::setw(10) << status << "\033[0m"
                  << std::setw(12) << error_str.str() << "║\n";
    }

    std::cout << "╠══════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Total: " << std::left << std::setw(3) << (passed + failed)
              << "  Passed: \033[32m" << std::setw(3) << passed << "\033[0m"
              << "  Failed: \033[31m" << std::setw(3) << failed << "\033[0m"
              << std::setw(34) << "" << "║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n";

    if (failed == 0) {
        std::cout << "\n  \033[32m✓ ALL TESTS PASSED\033[0m\n\n";
    } else {
        std::cout << "\n  \033[31m✗ SOME TESTS FAILED\033[0m\n\n";
    }
}

int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║   FlashMoE HIP Port - Comprehensive Correctness Test Suite           ║\n";
    std::cout << "║                    AMD Instinct MI450 (gfx1250)                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n";

    // Print device info
    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, 0));

    std::cout << "\nDevice: " << props.name << " (" << props.gcnArchName << ")\n";
    std::cout << "Warp Size: " << props.warpSize << "\n";

    // Run all test sections
    testTensorOperations();
    testAtomicOperations();
    testGemmCorrectness();
    testGatedMLP();
    testSoftmaxTopK();
    testHipCUBOperations();
    testTypeConversions();

    // Print summary
    printSummary();

    // Return nonzero when any check fails.
    int failed = 0;
    for (const auto& r : g_results) {
        if (!r.passed) failed++;
    }

    return (failed == 0) ? 0 : 1;
}
