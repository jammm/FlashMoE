//
// FlashMoE HIP Port - GEMM Dispatch Layer
// Provides unified interface for GEMM operations with multiple backends
//
// Backends:
// 1. hipBLASLt - Recommended for production (highest performance)
// 2. hipBLAS - Fallback option
// 3. rocWMMA - Custom kernels for specialized operations
//
// Copyright (c) 2025, Osayamen Jonathan Aimuyo
// All rights reserved.
//

#ifndef FLASHMOE_HIP_TILE_GEMM_DISPATCH_CUH
#define FLASHMOE_HIP_TILE_GEMM_DISPATCH_CUH

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

#include <memory>
#include <mutex>

namespace flashmoe::gemm {

// ============================================================================
// GEMM Backend Selection
// ============================================================================

enum class GemmBackend {
    HIPBLASLT,      // Recommended production backend
    HIPBLAS,        // Fallback backend
    ROCWMMA_CUSTOM  // Custom backend for specialized MoE operations
};

// ============================================================================
// Singleton Handle Manager for hipBLAS/hipBLASLt
// ============================================================================

class GemmHandleManager {
public:
    static GemmHandleManager& instance() {
        static GemmHandleManager instance_;
        return instance_;
    }

    hipblasHandle_t getHipblasHandle() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!hipblas_handle_initialized_) {
            hipblasCreate(&hipblas_handle_);
            hipblas_handle_initialized_ = true;
        }
        return hipblas_handle_;
    }

    hipblasLtHandle_t getHipblasLtHandle() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!hipblaslt_handle_initialized_) {
            hipblasLtCreate(&hipblaslt_handle_);
            hipblaslt_handle_initialized_ = true;
        }
        return hipblaslt_handle_;
    }

    ~GemmHandleManager() {
        if (hipblas_handle_initialized_) {
            hipblasDestroy(hipblas_handle_);
        }
        if (hipblaslt_handle_initialized_) {
            hipblasLtDestroy(hipblaslt_handle_);
        }
    }

private:
    GemmHandleManager() = default;
    GemmHandleManager(const GemmHandleManager&) = delete;
    GemmHandleManager& operator=(const GemmHandleManager&) = delete;

    std::mutex mutex_;
    hipblasHandle_t hipblas_handle_;
    hipblasLtHandle_t hipblaslt_handle_;
    bool hipblas_handle_initialized_ = false;
    bool hipblaslt_handle_initialized_ = false;
};

// ============================================================================
// hipBLASLt GEMM Wrapper (Recommended for production)
// ============================================================================

// Cached hipBLASLt operation for repeated calls with same dimensions
class HipblasLtGemm {
public:
    HipblasLtGemm() = default;

    ~HipblasLtGemm() {
        cleanup();
    }

    hipblasStatus_t execute(
        const __half* A, const __half* B, __half* C,
        int M, int N, int K,
        float alpha, float beta,
        hipStream_t stream = 0
    ) {
        // Initialize or reinitialize if dimensions changed
        if (!initialized_ || M != M_ || N != N_ || K != K_) {
            cleanup();
            hipblasStatus_t status = initialize(M, N, K);
            if (status != HIPBLAS_STATUS_SUCCESS) {
                return status;
            }
        }

        return hipblasLtMatmul(
            GemmHandleManager::instance().getHipblasLtHandle(),
            matmul_desc_,
            &alpha,
            A, layout_a_,
            B, layout_b_,
            &beta,
            C, layout_c_,
            C, layout_c_,
            &algo_,
            workspace_, workspace_size_,
            stream
        );
    }

private:
    hipblasStatus_t initialize(int M, int N, int K) {
        M_ = M;
        N_ = N;
        K_ = K;

        hipblasLtHandle_t handle = GemmHandleManager::instance().getHipblasLtHandle();

        hipDataType dataType = HIP_R_16F;
        hipDataType scaleType = HIP_R_32F;
        hipblasComputeType_t computeType = HIPBLAS_COMPUTE_32F;

        // Create matmul descriptor
        hipblasStatus_t status = hipblasLtMatmulDescCreate(&matmul_desc_, computeType, scaleType);
        if (status != HIPBLAS_STATUS_SUCCESS) return status;

        // Set transpose operations (no transpose)
        hipblasOperation_t transA = HIPBLAS_OP_N;
        hipblasOperation_t transB = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(matmul_desc_, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
        hipblasLtMatmulDescSetAttribute(matmul_desc_, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

        // Create matrix layouts (row-major)
        status = hipblasLtMatrixLayoutCreate(&layout_a_, dataType, M, K, K);
        if (status != HIPBLAS_STATUS_SUCCESS) return status;

        status = hipblasLtMatrixLayoutCreate(&layout_b_, dataType, K, N, N);
        if (status != HIPBLAS_STATUS_SUCCESS) return status;

        status = hipblasLtMatrixLayoutCreate(&layout_c_, dataType, M, N, N);
        if (status != HIPBLAS_STATUS_SUCCESS) return status;

        // Set row-major order
        hipblasLtOrder_t order = HIPBLASLT_ORDER_ROW;
        hipblasLtMatrixLayoutSetAttribute(layout_a_, HIPBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
        hipblasLtMatrixLayoutSetAttribute(layout_b_, HIPBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
        hipblasLtMatrixLayoutSetAttribute(layout_c_, HIPBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));

        // Find best algorithm
        hipblasLtMatmulPreference_t pref;
        hipblasLtMatmulPreferenceCreate(&pref);

        workspace_size_ = 64 * 1024 * 1024;  // 64 MB
        hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_));

        hipMalloc(&workspace_, workspace_size_);

        hipblasLtMatmulHeuristicResult_t heuristicResult[4];
        int returnedAlgoCount = 0;
        status = hipblasLtMatmulAlgoGetHeuristic(
            handle, matmul_desc_, layout_a_, layout_b_, layout_c_, layout_c_,
            pref, 4, heuristicResult, &returnedAlgoCount);

        hipblasLtMatmulPreferenceDestroy(pref);

        if (status != HIPBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) {
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        }

        algo_ = heuristicResult[0].algo;
        initialized_ = true;

        return HIPBLAS_STATUS_SUCCESS;
    }

    void cleanup() {
        if (initialized_) {
            if (workspace_) {
                hipFree(workspace_);
                workspace_ = nullptr;
            }
            if (layout_a_) hipblasLtMatrixLayoutDestroy(layout_a_);
            if (layout_b_) hipblasLtMatrixLayoutDestroy(layout_b_);
            if (layout_c_) hipblasLtMatrixLayoutDestroy(layout_c_);
            if (matmul_desc_) hipblasLtMatmulDescDestroy(matmul_desc_);
            initialized_ = false;
        }
    }

    bool initialized_ = false;
    int M_ = 0, N_ = 0, K_ = 0;

    hipblasLtMatmulDesc_t matmul_desc_ = nullptr;
    hipblasLtMatrixLayout_t layout_a_ = nullptr;
    hipblasLtMatrixLayout_t layout_b_ = nullptr;
    hipblasLtMatrixLayout_t layout_c_ = nullptr;
    hipblasLtMatmulAlgo_t algo_;
    void* workspace_ = nullptr;
    size_t workspace_size_ = 0;
};

// ============================================================================
// Unified GEMM Interface
// ============================================================================

// C = alpha * A * B + beta * C
// A: M x K (row-major)
// B: K x N (row-major)
// C: M x N (row-major)
inline hipblasStatus_t gemm_fp16(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K,
    float alpha = 1.0f, float beta = 0.0f,
    GemmBackend backend = GemmBackend::HIPBLASLT,
    hipStream_t stream = 0
) {
    switch (backend) {
        case GemmBackend::HIPBLASLT: {
            // Use thread-local cached operation for better performance
            thread_local HipblasLtGemm gemm_op;
            return gemm_op.execute(A, B, C, M, N, K, alpha, beta, stream);
        }

        case GemmBackend::HIPBLAS: {
            hipblasHandle_t handle = GemmHandleManager::instance().getHipblasHandle();
            hipblasSetStream(handle, stream);

            hipblasHalf alpha_h, beta_h;
            __half alpha_half = __float2half(alpha);
            __half beta_half = __float2half(beta);
            memcpy(&alpha_h, &alpha_half, sizeof(hipblasHalf));
            memcpy(&beta_h, &beta_half, sizeof(hipblasHalf));

            // hipBLAS uses column-major, so we compute C^T = B^T * A^T
            return hipblasHgemm(
                handle,
                HIPBLAS_OP_N, HIPBLAS_OP_N,
                N, M, K,
                &alpha_h,
                reinterpret_cast<const hipblasHalf*>(B), N,
                reinterpret_cast<const hipblasHalf*>(A), K,
                &beta_h,
                reinterpret_cast<hipblasHalf*>(C), N
            );
        }

        case GemmBackend::ROCWMMA_CUSTOM:
        default:
            // Fall back to hipBLAS for now
            return gemm_fp16(A, B, C, M, N, K, alpha, beta, GemmBackend::HIPBLAS, stream);
    }
}

// ============================================================================
// Batched GEMM for MoE (multiple experts)
// ============================================================================

// C[i] = alpha * A[i] * B[i] + beta * C[i] for each batch
inline hipblasStatus_t gemm_batched_fp16(
    const __half* const* A, const __half* const* B, __half* const* C,
    int M, int N, int K,
    int batch_count,
    float alpha = 1.0f, float beta = 0.0f,
    GemmBackend backend = GemmBackend::HIPBLAS,
    hipStream_t stream = 0
) {
    // For batched operations, hipBLAS batched API is simpler
    hipblasHandle_t handle = GemmHandleManager::instance().getHipblasHandle();
    hipblasSetStream(handle, stream);

    hipblasHalf alpha_h, beta_h;
    __half alpha_half = __float2half(alpha);
    __half beta_half = __float2half(beta);
    memcpy(&alpha_h, &alpha_half, sizeof(hipblasHalf));
    memcpy(&beta_h, &beta_half, sizeof(hipblasHalf));

    return hipblasHgemmBatched(
        handle,
        HIPBLAS_OP_N, HIPBLAS_OP_N,
        N, M, K,
        &alpha_h,
        reinterpret_cast<const hipblasHalf* const*>(B), N,
        reinterpret_cast<const hipblasHalf* const*>(A), K,
        &beta_h,
        reinterpret_cast<hipblasHalf* const*>(C), N,
        batch_count
    );
}

// ============================================================================
// Performance Recommendations
// ============================================================================

/*
 * GEMM backend guidance:
 *
 * | Backend     | Recommendation |
 * |-------------|----------------|
 * | hipBLASLt   | Production     |
 * | hipBLAS     | Fallback       |
 * | rocWMMA     | Fused custom operations |
 *
 * Recommendations:
 * 1. Use hipBLASLt for all production GEMM operations
 * 2. Use hipBLAS as fallback when hipBLASLt fails
 * 3. Custom rocWMMA kernels are useful for:
 *    - Fused operations (GEMM + activation)
 *    - Very small matrices where library overhead dominates
 *    - Specialized memory access patterns
 *
 * For FlashMoE:
 * - Expert FFN GEMM: Use hipBLASLt
 * - Gate computation: Use hipBLASLt
 * - Token combine: Custom fused kernel may be beneficial
 */

} // namespace flashmoe::gemm

#endif // FLASHMOE_HIP_TILE_GEMM_DISPATCH_CUH
