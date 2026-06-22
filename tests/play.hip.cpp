/**
 * FlashMoE Interactive Test
 *
 * A simple test you can modify and experiment with.
 * Demonstrates MoE forward pass with hipBLASLt on AMD MI450/MI450.
 *
 * Usage:
 *   ./play [num_tokens] [hidden_dim] [num_experts] [top_k]
 *
 * Examples:
 *   ./play                    # Default: 4096 tokens, 4096 hidden, 8 experts, top-2
 *   ./play 8192               # 8192 tokens
 *   ./play 16384 4096 8 2     # Mixtral-like config
 *   ./play 32768 5120 64 6    # DeepSeek-like config
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

// ============================================================================
// Macros
// ============================================================================

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#define HIPBLASLT_CHECK(call) do { \
    hipblasStatus_t status = call; \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        std::cerr << "hipBLASLt Error: " << status << " at " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Configuration - MODIFY THESE TO EXPERIMENT
// ============================================================================

struct Config {
    int num_tokens = 4096;         // Total tokens to process
    int hidden_dim = 4096;         // Model hidden dimension
    int intermediate_dim = 14336;  // FFN intermediate (usually ~3.5x hidden)
    int num_experts = 8;           // Number of experts
    int top_k = 2;                 // Tokens routed to top-k experts
    int warmup_iters = 5;          // Warmup iterations
    int bench_iters = 20;          // Benchmark iterations

    int expert_capacity() const {
        return std::max(32, (int)std::ceil(num_tokens * top_k * 1.1 / num_experts));
    }

    double gemm_flops() const {
        // Up + Down projection per expert
        return 2.0 * expert_capacity() * hidden_dim * intermediate_dim * 2.0 * num_experts;
    }

    void print() const {
        std::cout << "\n=== Configuration ===" << std::endl;
        std::cout << "Tokens:         " << num_tokens << std::endl;
        std::cout << "Hidden dim:     " << hidden_dim << std::endl;
        std::cout << "Intermediate:   " << intermediate_dim << std::endl;
        std::cout << "Experts:        " << num_experts << std::endl;
        std::cout << "Top-K:          " << top_k << std::endl;
        std::cout << "Expert capacity:" << expert_capacity() << std::endl;
        std::cout << "GFLOPs/forward: " << gemm_flops() / 1e9 << std::endl;
    }
};

// ============================================================================
// Simple FP16 GEMM using hipBLASLt
// ============================================================================

class GemmRunner {
    hipblasLtHandle_t handle_;
    void* workspace_;
    size_t workspace_size_ = 256 * 1024 * 1024;  // 256MB

public:
    GemmRunner() {
        HIPBLASLT_CHECK(hipblasLtCreate(&handle_));
        HIP_CHECK(hipMalloc(&workspace_, workspace_size_));
    }

    ~GemmRunner() {
        hipFree(workspace_);
        hipblasLtDestroy(handle_);
    }

    // C[M,N] = A[M,K] @ B[K,N]
    void gemm_fp16(const __half* A, const __half* B, __half* C,
                   int M, int N, int K, hipStream_t stream) {

        hipblasLtMatmulDesc_t matmul_desc;
        hipblasLtMatrixLayout_t layout_A, layout_B, layout_C;
        hipblasLtMatmulPreference_t pref;

        // Create operation descriptor
        HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));

        // Set transpose operations (for row-major: compute C^T = B^T @ A^T)
        hipblasOperation_t transA = HIPBLAS_OP_T;
        hipblasOperation_t transB = HIPBLAS_OP_N;
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

        // Create matrix layouts (column-major interpretation for row-major data)
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layout_A, HIP_R_16F, K, M, K));  // A^T
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layout_B, HIP_R_16F, K, N, K));  // B
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layout_C, HIP_R_16F, M, N, M));  // C^T

        // Create preference
        HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
        HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                              &workspace_size_, sizeof(workspace_size_)));

        // Find algorithm
        hipblasLtMatmulHeuristicResult_t heuristic;
        int returned_results = 0;
        HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle_, matmul_desc, layout_A, layout_B, layout_C, layout_C,
                                                        pref, 1, &heuristic, &returned_results));

        if (returned_results == 0) {
            std::cerr << "No algorithm found!" << std::endl;
            exit(1);
        }

        // Execute GEMM
        float alpha = 1.0f, beta = 0.0f;
        HIPBLASLT_CHECK(hipblasLtMatmul(handle_, matmul_desc, &alpha, A, layout_A, B, layout_B,
                                        &beta, C, layout_C, C, layout_C,
                                        &heuristic.algo, workspace_, workspace_size_, stream));

        // Cleanup
        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatrixLayoutDestroy(layout_A);
        hipblasLtMatrixLayoutDestroy(layout_B);
        hipblasLtMatrixLayoutDestroy(layout_C);
        hipblasLtMatmulDescDestroy(matmul_desc);
    }
};

// ============================================================================
// SiLU Activation Kernel
// ============================================================================

__global__ void silu_kernel(__half* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = __half2float(data[idx]);
        float silu = x / (1.0f + expf(-x));
        data[idx] = __float2half(silu);
    }
}

// ============================================================================
// Simple Gate (Softmax + Top-K) Kernel
// ============================================================================

__global__ void gate_topk_kernel(
    const __half* input,      // [num_tokens, hidden_dim]
    const __half* gate_w,     // [hidden_dim, num_experts]
    int* expert_ids,          // [num_tokens, top_k]
    float* expert_weights,    // [num_tokens, top_k]
    int num_tokens,
    int hidden_dim,
    int num_experts,
    int top_k
) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    extern __shared__ float scores[];

    // Compute gate scores (simple dot product per expert)
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        float score = 0.0f;
        for (int h = 0; h < hidden_dim; h += 64) {
            int idx = h + (threadIdx.x % 64);
            if (idx < hidden_dim) {
                score += __half2float(input[token_idx * hidden_dim + idx]) *
                         __half2float(gate_w[idx * num_experts + e]);
            }
        }
        scores[e] = score;
    }
    __syncthreads();

    // Softmax (simplified)
    if (threadIdx.x == 0) {
        float max_score = scores[0];
        for (int e = 1; e < num_experts; e++) max_score = fmaxf(max_score, scores[e]);

        float sum = 0.0f;
        for (int e = 0; e < num_experts; e++) {
            scores[e] = expf(scores[e] - max_score);
            sum += scores[e];
        }
        for (int e = 0; e < num_experts; e++) scores[e] /= sum;

        // Top-K selection (simple O(k*n) for small k)
        for (int k = 0; k < top_k; k++) {
            int best_e = 0;
            float best_score = -1.0f;
            for (int e = 0; e < num_experts; e++) {
                if (scores[e] > best_score) {
                    best_score = scores[e];
                    best_e = e;
                }
            }
            expert_ids[token_idx * top_k + k] = best_e;
            expert_weights[token_idx * top_k + k] = best_score;
            scores[best_e] = -1.0f;  // Mark as selected
        }
    }
}

// ============================================================================
// Timer
// ============================================================================

class Timer {
    hipEvent_t start_, stop_;
public:
    Timer() {
        HIP_CHECK(hipEventCreate(&start_));
        HIP_CHECK(hipEventCreate(&stop_));
    }
    ~Timer() {
        hipEventDestroy(start_);
        hipEventDestroy(stop_);
    }
    void start(hipStream_t stream = 0) { HIP_CHECK(hipEventRecord(start_, stream)); }
    float stop(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(stop_, stream));
        HIP_CHECK(hipEventSynchronize(stop_));
        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// ============================================================================
// Main Test
// ============================================================================

int main(int argc, char** argv) {
    // Parse command line
    Config cfg;
    if (argc > 1) cfg.num_tokens = std::atoi(argv[1]);
    if (argc > 2) cfg.hidden_dim = std::atoi(argv[2]);
    if (argc > 3) cfg.num_experts = std::atoi(argv[3]);
    if (argc > 4) cfg.top_k = std::atoi(argv[4]);

    // Auto-set intermediate dim (~3.5x hidden)
    cfg.intermediate_dim = (cfg.hidden_dim * 7 + 1) / 2;  // ~3.5x

    // Print device info
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    std::cout << "\n=== Device ===" << std::endl;
    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "Arch: " << props.gcnArchName << std::endl;

    cfg.print();

    // Allocate memory
    int M = cfg.expert_capacity();
    int K = cfg.hidden_dim;
    int N = cfg.intermediate_dim;

    __half *d_input, *d_gate_w, *d_expert_up, *d_expert_down;
    __half *d_hidden, *d_intermediate, *d_output;
    int *d_expert_ids;
    float *d_expert_weights;

    size_t input_size = (size_t)cfg.num_tokens * cfg.hidden_dim * sizeof(__half);
    size_t gate_size = (size_t)cfg.hidden_dim * cfg.num_experts * sizeof(__half);
    size_t expert_up_size = (size_t)cfg.num_experts * K * N * sizeof(__half);
    size_t expert_down_size = (size_t)cfg.num_experts * N * K * sizeof(__half);
    size_t intermediate_size = (size_t)M * N * sizeof(__half);
    size_t output_size = (size_t)M * K * sizeof(__half);

    HIP_CHECK(hipMalloc(&d_input, input_size));
    HIP_CHECK(hipMalloc(&d_gate_w, gate_size));
    HIP_CHECK(hipMalloc(&d_expert_up, expert_up_size));
    HIP_CHECK(hipMalloc(&d_expert_down, expert_down_size));
    HIP_CHECK(hipMalloc(&d_hidden, output_size));
    HIP_CHECK(hipMalloc(&d_intermediate, intermediate_size));
    HIP_CHECK(hipMalloc(&d_output, output_size));
    HIP_CHECK(hipMalloc(&d_expert_ids, cfg.num_tokens * cfg.top_k * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_expert_weights, cfg.num_tokens * cfg.top_k * sizeof(float)));

    // Initialize with random data
    std::vector<__half> h_data(std::max({input_size, expert_up_size, expert_down_size}) / sizeof(__half));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto& v : h_data) v = __float2half(dist(rng));

    HIP_CHECK(hipMemcpy(d_input, h_data.data(), input_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_gate_w, h_data.data(), gate_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_expert_up, h_data.data(), expert_up_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_expert_down, h_data.data(), expert_down_size, hipMemcpyHostToDevice));

    // Create GEMM runner
    GemmRunner gemm;
    Timer timer;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    std::cout << "\n=== Running Tests ===" << std::endl;

    // -------------------------------------------------------------------------
    // Test 1: Gate Kernel
    // -------------------------------------------------------------------------
    std::cout << "\n[1] Gate (Softmax + Top-K)... ";
    timer.start(stream);
    gate_topk_kernel<<<cfg.num_tokens, 256, cfg.num_experts * sizeof(float), stream>>>(
        d_input, d_gate_w, d_expert_ids, d_expert_weights,
        cfg.num_tokens, cfg.hidden_dim, cfg.num_experts, cfg.top_k
    );
    float gate_ms = timer.stop(stream);
    std::cout << gate_ms << " ms" << std::endl;

    // -------------------------------------------------------------------------
    // Test 2: Expert GEMM (single expert)
    // -------------------------------------------------------------------------
    std::cout << "\n[2] Single Expert FFN (Up + SiLU + Down)..." << std::endl;

    // Warmup
    for (int i = 0; i < cfg.warmup_iters; i++) {
        gemm.gemm_fp16(d_hidden, d_expert_up, d_intermediate, M, N, K, stream);
        silu_kernel<<<(M*N + 255)/256, 256, 0, stream>>>(d_intermediate, M * N);
        gemm.gemm_fp16(d_intermediate, d_expert_down, d_output, M, K, N, stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    // Benchmark
    timer.start(stream);
    for (int i = 0; i < cfg.bench_iters; i++) {
        gemm.gemm_fp16(d_hidden, d_expert_up, d_intermediate, M, N, K, stream);
        silu_kernel<<<(M*N + 255)/256, 256, 0, stream>>>(d_intermediate, M * N);
        gemm.gemm_fp16(d_intermediate, d_expert_down, d_output, M, K, N, stream);
    }
    float single_expert_ms = timer.stop(stream) / cfg.bench_iters;

    double single_flops = 2.0 * M * K * N * 2.0;  // Up + Down
    double single_tflops = single_flops / (single_expert_ms * 1e9);
    std::cout << "  Time: " << single_expert_ms << " ms" << std::endl;
    std::cout << "  TFLOPS: " << single_tflops << std::endl;

    // -------------------------------------------------------------------------
    // Test 3: All Experts
    // -------------------------------------------------------------------------
    std::cout << "\n[3] All " << cfg.num_experts << " Experts..." << std::endl;

    // Warmup
    for (int w = 0; w < cfg.warmup_iters; w++) {
        for (int e = 0; e < cfg.num_experts; e++) {
            __half* w_up = d_expert_up + (size_t)e * K * N;
            __half* w_down = d_expert_down + (size_t)e * N * K;
            gemm.gemm_fp16(d_hidden, w_up, d_intermediate, M, N, K, stream);
            silu_kernel<<<(M*N + 255)/256, 256, 0, stream>>>(d_intermediate, M * N);
            gemm.gemm_fp16(d_intermediate, w_down, d_output, M, K, N, stream);
        }
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    // Benchmark
    timer.start(stream);
    for (int i = 0; i < cfg.bench_iters; i++) {
        for (int e = 0; e < cfg.num_experts; e++) {
            __half* w_up = d_expert_up + (size_t)e * K * N;
            __half* w_down = d_expert_down + (size_t)e * N * K;
            gemm.gemm_fp16(d_hidden, w_up, d_intermediate, M, N, K, stream);
            silu_kernel<<<(M*N + 255)/256, 256, 0, stream>>>(d_intermediate, M * N);
            gemm.gemm_fp16(d_intermediate, w_down, d_output, M, K, N, stream);
        }
    }
    float all_experts_ms = timer.stop(stream) / cfg.bench_iters;

    double all_flops = cfg.gemm_flops();
    double all_tflops = all_flops / (all_experts_ms * 1e9);
    double tokens_per_sec = cfg.num_tokens / (all_experts_ms / 1000.0);

    std::cout << "  Time: " << all_experts_ms << " ms" << std::endl;
    std::cout << "  TFLOPS: " << all_tflops << std::endl;
    std::cout << "  Tokens/sec: " << std::fixed << std::setprecision(0) << tokens_per_sec << std::endl;

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Configuration: " << cfg.num_tokens << " tokens, "
              << cfg.hidden_dim << " hidden, " << cfg.num_experts << " experts" << std::endl;
    std::cout << "Expert GEMM: " << std::fixed << std::setprecision(1) << all_tflops << " TFLOPS" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(0) << tokens_per_sec << " tokens/sec" << std::endl;
    // Cleanup
    hipFree(d_input);
    hipFree(d_gate_w);
    hipFree(d_expert_up);
    hipFree(d_expert_down);
    hipFree(d_hidden);
    hipFree(d_intermediate);
    hipFree(d_output);
    hipFree(d_expert_ids);
    hipFree(d_expert_weights);
    hipStreamDestroy(stream);

    std::cout << "\nDone!" << std::endl;
    return 0;
}
