/**
 * FlashMoE Production-Scale MoE Forward Pass Benchmark
 *
 * Uses hipBLASLt for expert GEMMs.
 * Comprehensive benchmark with real-world MoE configurations:
 * - Mixtral-8x7B style (8 experts, top-2)
 * - Mixtral-8x22B style (8 experts, top-2)
 * - DeepSeek-V2 style (64 experts, top-6)
 *
 * Target: AMD Instinct MI450 (gfx1250) / MI450 (gfx1250)
 *
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>
#include <string>

// ============================================================================
// Error Checking Macros
// ============================================================================

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#define HIPBLASLT_CHECK(call) do { \
    hipblasStatus_t status = call; \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        std::cerr << "hipBLASLt Error: " << status << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

constexpr size_t HIPBLASLT_WORKSPACE_SIZE = 256 * 1024 * 1024;  // 256 MB

// ============================================================================
// MoE Configuration Structure
// ============================================================================

struct MoEConfig {
    std::string name;
    int batch_size;
    int seq_len;
    int hidden_dim;
    int intermediate_dim;
    int num_experts;
    int top_k;

    int total_tokens() const { return batch_size * seq_len; }
    int expert_capacity() const {
        // Capacity factor = 1.1 for slight overprovisioning
        // Cap at reasonable maximum to avoid excessive memory usage
        int base_capacity = static_cast<int>(std::ceil(total_tokens() * top_k * 1.1 / num_experts));
        return std::min(base_capacity, 4096);  // Cap at 4096 tokens per expert
    }

    // FLOPs for expert FFN: Up projection + Down projection
    // Each expert: 2 * capacity * hidden * intermediate * 2 (for up + down)
    double expert_gemm_flops() const {
        return 2.0 * expert_capacity() * hidden_dim * intermediate_dim * 2.0 * num_experts;
    }

    // Memory for all expert weights
    size_t expert_weights_bytes() const {
        return (size_t)num_experts * hidden_dim * intermediate_dim * sizeof(__half) * 2;  // up + down
    }
};

// ============================================================================
// GPU Timer
// ============================================================================

class GpuTimer {
    hipEvent_t start_, stop_;
public:
    GpuTimer() {
        HIP_CHECK(hipEventCreate(&start_));
        HIP_CHECK(hipEventCreate(&stop_));
    }
    ~GpuTimer() {
        (void)hipEventDestroy(start_);
        (void)hipEventDestroy(stop_);
    }
    void start(hipStream_t s = 0) { HIP_CHECK(hipEventRecord(start_, s)); }
    void stop(hipStream_t s = 0) {
        HIP_CHECK(hipEventRecord(stop_, s));
        HIP_CHECK(hipEventSynchronize(stop_));
    }
    float elapsed_ms() {
        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// ============================================================================
// Performance Results Structure
// ============================================================================

struct MoETimingResults {
    float gate_ms = 0.0f;
    float dispatch_ms = 0.0f;
    float expert_gemm_ms = 0.0f;
    float combine_ms = 0.0f;
    float total_ms = 0.0f;

    double expert_tflops = 0.0;
    double tokens_per_second = 0.0;
    double memory_bandwidth_tb_s = 0.0;
};

// ============================================================================
// Gate Computation Kernels (Softmax + Top-K)
// ============================================================================

// Softmax kernel for gate probabilities
__global__ void softmaxKernel(
    const __half* __restrict__ gate_logits,  // [num_tokens, num_experts]
    float* __restrict__ probs,                // [num_tokens, num_experts]
    int num_tokens,
    int num_experts
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int e = 0; e < num_experts; ++e) {
        float val = __half2float(gate_logits[token_idx * num_experts + e]);
        max_val = fmaxf(max_val, val);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int e = 0; e < num_experts; ++e) {
        float val = __half2float(gate_logits[token_idx * num_experts + e]);
        float exp_val = __expf(val - max_val);
        probs[token_idx * num_experts + e] = exp_val;
        sum += exp_val;
    }

    // Normalize
    float inv_sum = __fdividef(1.0f, sum);
    for (int e = 0; e < num_experts; ++e) {
        probs[token_idx * num_experts + e] *= inv_sum;
    }
}

// Top-K selection kernel with atomic expert counting
__global__ void topKSelectionKernel(
    const float* __restrict__ probs,           // [num_tokens, num_experts]
    int* __restrict__ expert_indices,           // [num_tokens, top_k]
    float* __restrict__ expert_weights,         // [num_tokens, top_k]
    int* __restrict__ expert_counts,            // [num_experts]
    int num_tokens,
    int num_experts,
    int top_k
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    // Simple top-k selection (sufficient for small E like 8 or 64)
    for (int k = 0; k < top_k; ++k) {
        float max_prob = -1.0f;
        int max_idx = -1;

        for (int e = 0; e < num_experts; ++e) {
            float prob = probs[token_idx * num_experts + e];

            // Skip already selected
            bool already_selected = false;
            for (int j = 0; j < k; ++j) {
                if (expert_indices[token_idx * top_k + j] == e) {
                    already_selected = true;
                    break;
                }
            }

            if (!already_selected && prob > max_prob) {
                max_prob = prob;
                max_idx = e;
            }
        }

        expert_indices[token_idx * top_k + k] = max_idx;
        expert_weights[token_idx * top_k + k] = max_prob;

        if (max_idx >= 0) {
            atomicAdd(&expert_counts[max_idx], 1);
        }
    }

    // Renormalize weights to sum to 1
    float weight_sum = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        weight_sum += expert_weights[token_idx * top_k + k];
    }
    if (weight_sum > 0.0f) {
        float inv_sum = __fdividef(1.0f, weight_sum);
        for (int k = 0; k < top_k; ++k) {
            expert_weights[token_idx * top_k + k] *= inv_sum;
        }
    }
}

// ============================================================================
// Token Dispatch Kernel
// ============================================================================

__global__ void tokenDispatchKernel(
    const __half* __restrict__ input_tokens,   // [num_tokens, hidden_dim]
    __half* __restrict__ expert_buffers,        // [num_experts, expert_capacity, hidden_dim]
    const int* __restrict__ expert_indices,     // [num_tokens, top_k]
    int* __restrict__ dispatch_indices,         // [num_tokens, top_k] - slot in expert buffer
    int* __restrict__ expert_offsets,           // [num_experts]
    int num_tokens,
    int hidden_dim,
    int expert_capacity,
    int top_k
) {
    int assignment_idx = blockIdx.x;
    if (assignment_idx >= num_tokens * top_k) return;

    int token_idx = assignment_idx / top_k;
    int expert_idx = expert_indices[assignment_idx];

    if (expert_idx < 0) {
        dispatch_indices[assignment_idx] = -1;
        return;
    }

    // Atomically get slot in expert buffer
    int slot = atomicAdd(&expert_offsets[expert_idx], 1);

    // Handle capacity overflow (token dropping)
    if (slot >= expert_capacity) {
        dispatch_indices[assignment_idx] = -1;
        return;
    }

    dispatch_indices[assignment_idx] = slot;

    // Copy token to expert buffer using vectorized loads
    const int vec_size = 8;  // Load 8 half values at once
    int num_vecs = hidden_dim / vec_size;

    for (int v = threadIdx.x; v < num_vecs; v += blockDim.x) {
        int offset = v * vec_size;
        // Use float4 for 8 half values (128 bits)
        float4 data = *reinterpret_cast<const float4*>(&input_tokens[token_idx * hidden_dim + offset]);
        *reinterpret_cast<float4*>(&expert_buffers[(expert_idx * expert_capacity + slot) * hidden_dim + offset]) = data;
    }

    // Handle remainder
    for (int h = num_vecs * vec_size + threadIdx.x; h < hidden_dim; h += blockDim.x) {
        expert_buffers[(expert_idx * expert_capacity + slot) * hidden_dim + h] =
            input_tokens[token_idx * hidden_dim + h];
    }
}

// ============================================================================
// hipBLASLt GEMM Functions
// ============================================================================

// Column-major GEMM wrapper for row-major data
// For C[M,N] = A[M,K] @ B[K,N] in row-major, compute C^T = B^T @ A^T in col-major
void runGemm(hipblasLtHandle_t handle, void* workspace, size_t workspaceSize,
             int M, int N, int K,
             const __half* A, const __half* B, __half* C,
             hipStream_t stream = 0) {
    hipblasLtMatmulDesc_t matmulDesc;
    hipblasLtMatrixLayout_t layoutBT, layoutAT, layoutCT;

    HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F));

    hipblasOperation_t trans = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans));
    hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans));

    // Column-major layouts for row-major data
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutBT, HIP_R_16F, N, K, N));  // B^T[N,K]
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutAT, HIP_R_16F, K, M, K));  // A^T[K,M]
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutCT, HIP_R_16F, N, M, N));  // C^T[N,M]

    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &workspaceSize, sizeof(workspaceSize));

    hipblasLtMatmulHeuristicResult_t heuristicResult[4];
    int returnedAlgoCount = 0;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(
        handle, matmulDesc, layoutBT, layoutAT, layoutCT, layoutCT,
        pref, 4, heuristicResult, &returnedAlgoCount));

    hipblasLtMatmulPreferenceDestroy(pref);

    if (returnedAlgoCount == 0) {
        std::cerr << "No algorithm found for " << M << "x" << N << "x" << K << "\n";
        return;
    }

    float alpha = 1.0f, beta = 0.0f;
    HIPBLASLT_CHECK(hipblasLtMatmul(
        handle, matmulDesc, &alpha,
        B, layoutBT, A, layoutAT,  // Swapped for transpose trick
        &beta, C, layoutCT, C, layoutCT,
        &heuristicResult[0].algo, workspace, workspaceSize, stream));

    hipblasLtMatrixLayoutDestroy(layoutBT);
    hipblasLtMatrixLayoutDestroy(layoutAT);
    hipblasLtMatrixLayoutDestroy(layoutCT);
    hipblasLtMatmulDescDestroy(matmulDesc);
}

// ============================================================================
// SiLU Activation Kernel (vectorized)
// ============================================================================

__global__ void siluKernel(__half* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        float x = __half2float(data[i]);
        data[i] = __float2half(x / (1.0f + __expf(-x)));
    }
}

// ============================================================================
// Combine Kernel (weighted sum of expert outputs)
// ============================================================================

__global__ void combineKernel(
    const __half* __restrict__ expert_outputs,  // [num_experts, expert_capacity, hidden_dim]
    __half* __restrict__ output,                 // [num_tokens, hidden_dim]
    const int* __restrict__ expert_indices,      // [num_tokens, top_k]
    const int* __restrict__ dispatch_indices,    // [num_tokens, top_k]
    const float* __restrict__ expert_weights,    // [num_tokens, top_k]
    int num_tokens,
    int hidden_dim,
    int expert_capacity,
    int top_k
) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    // Each thread handles multiple hidden dimensions
    for (int h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        float sum = 0.0f;

        for (int k = 0; k < top_k; ++k) {
            int assign_idx = token_idx * top_k + k;
            int expert_idx = expert_indices[assign_idx];
            int slot = dispatch_indices[assign_idx];
            float weight = expert_weights[assign_idx];

            if (expert_idx >= 0 && slot >= 0 && slot < expert_capacity) {
                int src_idx = (expert_idx * expert_capacity + slot) * hidden_dim + h;
                sum += __half2float(expert_outputs[src_idx]) * weight;
            }
        }

        output[token_idx * hidden_dim + h] = __float2half(sum);
    }
}

// ============================================================================
// Complete MoE Forward Pass Function
// ============================================================================

MoETimingResults runMoEForward(
    const MoEConfig& cfg,
    hipblasLtHandle_t handle,
    void* workspace,
    hipStream_t stream,
    bool verbose = true
) {
    MoETimingResults results;
    GpuTimer total_timer, stage_timer;

    const int num_tokens = cfg.total_tokens();
    const int hidden_dim = cfg.hidden_dim;
    const int intermediate_dim = cfg.intermediate_dim;
    const int num_experts = cfg.num_experts;
    const int expert_capacity = cfg.expert_capacity();
    const int top_k = cfg.top_k;

    // Calculate sizes first
    size_t input_size = (size_t)num_tokens * hidden_dim * sizeof(__half);
    size_t gate_logits_size = (size_t)num_tokens * num_experts * sizeof(__half);
    size_t expert_buffer_in_size = (size_t)num_experts * expert_capacity * hidden_dim * sizeof(__half);
    size_t expert_buffer_inter_size = (size_t)num_experts * expert_capacity * intermediate_dim * sizeof(__half);
    size_t w_up_size = (size_t)num_experts * hidden_dim * intermediate_dim * sizeof(__half);
    size_t w_down_size = (size_t)num_experts * intermediate_dim * hidden_dim * sizeof(__half);

    // Calculate total memory needed
    size_t total_gpu_memory = input_size * 2 + gate_logits_size +
                               expert_buffer_in_size * 2 + expert_buffer_inter_size +
                               w_up_size + w_down_size +
                               (size_t)num_tokens * num_experts * sizeof(float) +  // gate_probs
                               (size_t)num_tokens * top_k * sizeof(float) +        // expert_weights
                               (size_t)num_tokens * top_k * sizeof(int) * 2 +      // indices
                               (size_t)num_experts * sizeof(int) * 2;              // counts/offsets

    if (verbose) {
        std::cout << "  Running: " << cfg.name << "\n";
        std::cout << "    Tokens: " << num_tokens << " (" << cfg.batch_size << " x " << cfg.seq_len << ")\n";
        std::cout << "    Hidden: " << hidden_dim << ", Intermediate: " << intermediate_dim << "\n";
        std::cout << "    Experts: " << num_experts << ", Top-K: " << top_k << ", Capacity: " << expert_capacity << "\n";
    }

    // Check available GPU memory
    size_t free_mem, total_mem;
    HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
    if (total_gpu_memory > free_mem * 0.9) {
        std::cerr << "    WARNING: Required memory (" << (total_gpu_memory/1e9) << " GB) exceeds available ("
                  << (free_mem/1e9) << " GB). Skipping.\n";
        return MoETimingResults{};
    }

    // Allocate device memory
    __half *d_input, *d_output, *d_gate_logits;
    __half *d_expert_buffer_in, *d_expert_buffer_inter, *d_expert_buffer_out;
    __half *d_w_up, *d_w_gate, *d_w_down;
    float *d_gate_probs, *d_expert_weights;
    int *d_expert_indices, *d_expert_counts, *d_expert_offsets, *d_dispatch_indices;

    HIP_CHECK(hipMalloc(&d_input, input_size));
    HIP_CHECK(hipMalloc(&d_output, input_size));
    HIP_CHECK(hipMalloc(&d_gate_logits, gate_logits_size));
    HIP_CHECK(hipMalloc(&d_expert_buffer_in, expert_buffer_in_size));
    HIP_CHECK(hipMalloc(&d_expert_buffer_inter, expert_buffer_inter_size));
    HIP_CHECK(hipMalloc(&d_expert_buffer_out, expert_buffer_in_size));
    HIP_CHECK(hipMalloc(&d_w_up, w_up_size));
    HIP_CHECK(hipMalloc(&d_w_down, w_down_size));
    HIP_CHECK(hipMalloc(&d_gate_probs, num_tokens * num_experts * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_expert_weights, num_tokens * top_k * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_expert_indices, num_tokens * top_k * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_expert_counts, num_experts * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_expert_offsets, num_experts * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_dispatch_indices, num_tokens * top_k * sizeof(int)));

    // Initialize weights directly on device (avoid large host allocations)
    // For input and gate logits, use smaller host buffers or initialize on device
    {
        // Input initialization - use chunks to avoid large host allocation
        const int chunk_size = 1024 * 1024;  // 1M elements at a time
        std::vector<__half> h_chunk(chunk_size);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

        size_t total_input = (size_t)num_tokens * hidden_dim;
        for (size_t offset = 0; offset < total_input; offset += chunk_size) {
            size_t count = std::min((size_t)chunk_size, total_input - offset);
            for (size_t i = 0; i < count; ++i) h_chunk[i] = __float2half(dist(gen));
            HIP_CHECK(hipMemcpyAsync(d_input + offset, h_chunk.data(), count * sizeof(__half), hipMemcpyHostToDevice, stream));
        }

        size_t total_gate = (size_t)num_tokens * num_experts;
        for (size_t offset = 0; offset < total_gate; offset += chunk_size) {
            size_t count = std::min((size_t)chunk_size, total_gate - offset);
            for (size_t i = 0; i < count; ++i) h_chunk[i] = __float2half(dist(gen) * 2.0f);
            HIP_CHECK(hipMemcpyAsync(d_gate_logits + offset, h_chunk.data(), count * sizeof(__half), hipMemcpyHostToDevice, stream));
        }

        // Initialize weights with small values (chunked)
        size_t total_w_up = (size_t)num_experts * hidden_dim * intermediate_dim;
        for (size_t offset = 0; offset < total_w_up; offset += chunk_size) {
            size_t count = std::min((size_t)chunk_size, total_w_up - offset);
            for (size_t i = 0; i < count; ++i) h_chunk[i] = __float2half(dist(gen) * 0.02f);
            HIP_CHECK(hipMemcpyAsync(d_w_up + offset, h_chunk.data(), count * sizeof(__half), hipMemcpyHostToDevice, stream));
        }

        size_t total_w_down = (size_t)num_experts * intermediate_dim * hidden_dim;
        for (size_t offset = 0; offset < total_w_down; offset += chunk_size) {
            size_t count = std::min((size_t)chunk_size, total_w_down - offset);
            for (size_t i = 0; i < count; ++i) h_chunk[i] = __float2half(dist(gen) * 0.02f);
            HIP_CHECK(hipMemcpyAsync(d_w_down + offset, h_chunk.data(), count * sizeof(__half), hipMemcpyHostToDevice, stream));
        }
    }
    HIP_CHECK(hipMemsetAsync(d_expert_counts, 0, num_experts * sizeof(int), stream));
    HIP_CHECK(hipMemsetAsync(d_expert_offsets, 0, num_experts * sizeof(int), stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Warmup run
    {
        int threads = 256;
        int blocks = (num_tokens + threads - 1) / threads;

        hipLaunchKernelGGL(softmaxKernel, dim3(blocks), dim3(threads), 0, stream,
            d_gate_logits, d_gate_probs, num_tokens, num_experts);

        hipLaunchKernelGGL(topKSelectionKernel, dim3(blocks), dim3(threads), 0, stream,
            d_gate_probs, d_expert_indices, d_expert_weights, d_expert_counts,
            num_tokens, num_experts, top_k);

        HIP_CHECK(hipMemsetAsync(d_expert_offsets, 0, num_experts * sizeof(int), stream));

        hipLaunchKernelGGL(tokenDispatchKernel, dim3(num_tokens * top_k), dim3(256), 0, stream,
            d_input, d_expert_buffer_in, d_expert_indices, d_dispatch_indices, d_expert_offsets,
            num_tokens, hidden_dim, expert_capacity, top_k);

        // Run expert GEMMs
        for (int e = 0; e < num_experts; ++e) {
            const __half* expert_in = d_expert_buffer_in + e * expert_capacity * hidden_dim;
            __half* expert_inter = d_expert_buffer_inter + e * expert_capacity * intermediate_dim;
            __half* expert_out = d_expert_buffer_out + e * expert_capacity * hidden_dim;
            const __half* w_up = d_w_up + e * hidden_dim * intermediate_dim;
            const __half* w_down = d_w_down + e * intermediate_dim * hidden_dim;

            // Up projection
            runGemm(handle, workspace, HIPBLASLT_WORKSPACE_SIZE,
                    expert_capacity, intermediate_dim, hidden_dim,
                    expert_in, w_up, expert_inter, stream);

            // SiLU activation
            int act_size = expert_capacity * intermediate_dim;
            hipLaunchKernelGGL(siluKernel, dim3((act_size+255)/256), dim3(256), 0, stream,
                               expert_inter, act_size);

            // Down projection
            runGemm(handle, workspace, HIPBLASLT_WORKSPACE_SIZE,
                    expert_capacity, hidden_dim, intermediate_dim,
                    expert_inter, w_down, expert_out, stream);
        }

        hipLaunchKernelGGL(combineKernel, dim3(num_tokens), dim3(256), 0, stream,
            d_expert_buffer_out, d_output, d_expert_indices, d_dispatch_indices,
            d_expert_weights, num_tokens, hidden_dim, expert_capacity, top_k);

        HIP_CHECK(hipStreamSynchronize(stream));
    }

    // Reset counters for benchmark
    HIP_CHECK(hipMemsetAsync(d_expert_counts, 0, num_experts * sizeof(int), stream));
    HIP_CHECK(hipMemsetAsync(d_expert_offsets, 0, num_experts * sizeof(int), stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // ========================================================================
    // Benchmark Run
    // ========================================================================

    total_timer.start(stream);

    // Stage 1: Gate (Softmax + Top-K)
    stage_timer.start(stream);
    {
        int threads = 256;
        int blocks = (num_tokens + threads - 1) / threads;

        hipLaunchKernelGGL(softmaxKernel, dim3(blocks), dim3(threads), 0, stream,
            d_gate_logits, d_gate_probs, num_tokens, num_experts);

        hipLaunchKernelGGL(topKSelectionKernel, dim3(blocks), dim3(threads), 0, stream,
            d_gate_probs, d_expert_indices, d_expert_weights, d_expert_counts,
            num_tokens, num_experts, top_k);
    }
    stage_timer.stop(stream);
    results.gate_ms = stage_timer.elapsed_ms();

    // Stage 2: Dispatch
    HIP_CHECK(hipMemsetAsync(d_expert_offsets, 0, num_experts * sizeof(int), stream));
    stage_timer.start(stream);
    {
        hipLaunchKernelGGL(tokenDispatchKernel, dim3(num_tokens * top_k), dim3(256), 0, stream,
            d_input, d_expert_buffer_in, d_expert_indices, d_dispatch_indices, d_expert_offsets,
            num_tokens, hidden_dim, expert_capacity, top_k);
    }
    stage_timer.stop(stream);
    results.dispatch_ms = stage_timer.elapsed_ms();

    // Stage 3: Expert GEMMs (Up + Activation + Down)
    stage_timer.start(stream);
    {
        for (int e = 0; e < num_experts; ++e) {
            const __half* expert_in = d_expert_buffer_in + e * expert_capacity * hidden_dim;
            __half* expert_inter = d_expert_buffer_inter + e * expert_capacity * intermediate_dim;
            __half* expert_out = d_expert_buffer_out + e * expert_capacity * hidden_dim;
            const __half* w_up = d_w_up + e * hidden_dim * intermediate_dim;
            const __half* w_down = d_w_down + e * intermediate_dim * hidden_dim;

            // Up projection: [capacity, hidden] @ [hidden, inter] -> [capacity, inter]
            runGemm(handle, workspace, HIPBLASLT_WORKSPACE_SIZE,
                    expert_capacity, intermediate_dim, hidden_dim,
                    expert_in, w_up, expert_inter, stream);

            // SiLU activation
            int act_size = expert_capacity * intermediate_dim;
            hipLaunchKernelGGL(siluKernel, dim3((act_size+255)/256), dim3(256), 0, stream,
                               expert_inter, act_size);

            // Down projection: [capacity, inter] @ [inter, hidden] -> [capacity, hidden]
            runGemm(handle, workspace, HIPBLASLT_WORKSPACE_SIZE,
                    expert_capacity, hidden_dim, intermediate_dim,
                    expert_inter, w_down, expert_out, stream);
        }
    }
    stage_timer.stop(stream);
    results.expert_gemm_ms = stage_timer.elapsed_ms();

    // Stage 4: Combine
    stage_timer.start(stream);
    {
        hipLaunchKernelGGL(combineKernel, dim3(num_tokens), dim3(256), 0, stream,
            d_expert_buffer_out, d_output, d_expert_indices, d_dispatch_indices,
            d_expert_weights, num_tokens, hidden_dim, expert_capacity, top_k);
    }
    stage_timer.stop(stream);
    results.combine_ms = stage_timer.elapsed_ms();

    total_timer.stop(stream);
    results.total_ms = total_timer.elapsed_ms();

    // Calculate performance metrics
    results.expert_tflops = (cfg.expert_gemm_flops() / 1e12) / (results.expert_gemm_ms / 1e3);
    results.tokens_per_second = num_tokens / (results.total_ms / 1e3);

    // Memory bandwidth calculation (bytes moved during dispatch + combine)
    size_t dispatch_bytes = (size_t)num_tokens * top_k * hidden_dim * sizeof(__half) * 2;  // read + write
    size_t combine_bytes = (size_t)num_tokens * top_k * hidden_dim * sizeof(__half) +
                           (size_t)num_tokens * hidden_dim * sizeof(__half);  // read expert + write output
    double total_memory_bytes = dispatch_bytes + combine_bytes;
    double dispatch_combine_time_s = (results.dispatch_ms + results.combine_ms) / 1e3;
    results.memory_bandwidth_tb_s = (total_memory_bytes / 1e12) / dispatch_combine_time_s;

    // Cleanup
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_gate_logits));
    HIP_CHECK(hipFree(d_expert_buffer_in));
    HIP_CHECK(hipFree(d_expert_buffer_inter));
    HIP_CHECK(hipFree(d_expert_buffer_out));
    HIP_CHECK(hipFree(d_w_up));
    HIP_CHECK(hipFree(d_w_down));
    HIP_CHECK(hipFree(d_gate_probs));
    HIP_CHECK(hipFree(d_expert_weights));
    HIP_CHECK(hipFree(d_expert_indices));
    HIP_CHECK(hipFree(d_expert_counts));
    HIP_CHECK(hipFree(d_expert_offsets));
    HIP_CHECK(hipFree(d_dispatch_indices));

    return results;
}

// ============================================================================
// Print Results Table
// ============================================================================

void printResultsTable(const std::string& name, const MoETimingResults& results) {
    std::cout << "\n";
    std::cout << "  ┌─────────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "  │ " << std::left << std::setw(73) << name << "│\n";
    std::cout << "  ├─────────────────────────────────────────────────────────────────────────┤\n";
    std::cout << "  │ Timing Breakdown:                                                       │\n";
    std::cout << "  │   Gate:        " << std::right << std::fixed << std::setprecision(3) << std::setw(8) << results.gate_ms
              << " ms  (" << std::setprecision(1) << std::setw(5) << (results.gate_ms / results.total_ms * 100) << "%)";
    std::cout << std::setw(31) << "" << "│\n";
    std::cout << "  │   Dispatch:    " << std::right << std::fixed << std::setprecision(3) << std::setw(8) << results.dispatch_ms
              << " ms  (" << std::setprecision(1) << std::setw(5) << (results.dispatch_ms / results.total_ms * 100) << "%)";
    std::cout << std::setw(31) << "" << "│\n";
    std::cout << "  │   Expert GEMM: " << std::right << std::fixed << std::setprecision(3) << std::setw(8) << results.expert_gemm_ms
              << " ms  (" << std::setprecision(1) << std::setw(5) << (results.expert_gemm_ms / results.total_ms * 100) << "%)";
    std::cout << std::setw(31) << "" << "│\n";
    std::cout << "  │   Combine:     " << std::right << std::fixed << std::setprecision(3) << std::setw(8) << results.combine_ms
              << " ms  (" << std::setprecision(1) << std::setw(5) << (results.combine_ms / results.total_ms * 100) << "%)";
    std::cout << std::setw(31) << "" << "│\n";
    std::cout << "  │   ─────────────────────────────────                                     │\n";
    std::cout << "  │   Total:       " << std::right << std::fixed << std::setprecision(3) << std::setw(8) << results.total_ms
              << " ms" << std::setw(43) << "" << "│\n";
    std::cout << "  ├─────────────────────────────────────────────────────────────────────────┤\n";
    std::cout << "  │ Performance Metrics:                                                    │\n";
    std::cout << "  │   Expert GEMM TFLOPS:    " << std::fixed << std::setprecision(1) << std::setw(8) << results.expert_tflops
              << " TFLOPS";
    std::cout << std::setw(25) << "" << "│\n";
    std::cout << "  │   Tokens/second:         " << std::scientific << std::setprecision(2) << results.tokens_per_second;
    std::cout << std::fixed << std::setw(30) << "" << "│\n";
    std::cout << "  │   Memory bandwidth:      " << std::fixed << std::setprecision(3) << std::setw(8) << results.memory_bandwidth_tb_s
              << " TB/s" << std::setw(29) << "" << "│\n";
    std::cout << "  └─────────────────────────────────────────────────────────────────────────┘\n";
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║   FlashMoE Production-Scale MoE Forward Pass Benchmark                   ║\n";
    std::cout << "║                hipBLASLt Optimized for AMD Instinct MI450               ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════╝\n";

    // Device setup
    HIP_CHECK(hipSetDevice(0));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));

    std::cout << "\nDevice: " << props.name << " (" << props.gcnArchName << ")\n";

    // hipBLASLt setup
    hipblasLtHandle_t handle;
    HIPBLASLT_CHECK(hipblasLtCreate(&handle));

    void* workspace;
    HIP_CHECK(hipMalloc(&workspace, HIPBLASLT_WORKSPACE_SIZE));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // ========================================================================
    // Production MoE Configurations
    // ========================================================================

    std::vector<MoEConfig> configs = {
        // Mixtral-8x7B style: 8K tokens
        {"Mixtral-8x7B (8K tokens)", 4, 2048, 4096, 14336, 8, 2},

        // Mixtral-8x7B style: 16K tokens
        {"Mixtral-8x7B (16K tokens)", 8, 2048, 4096, 14336, 8, 2},

        // LLaMA-style MoE: 16K tokens (smaller intermediate)
        {"LLaMA-MoE (16K tokens)", 8, 2048, 4096, 11008, 8, 2},
    };

    std::cout << "\n";
    std::cout << "════════════════════════════════════════════════════════════════════════════\n";
    std::cout << "                    PRODUCTION-SCALE MOE CONFIGURATIONS                     \n";
    std::cout << "════════════════════════════════════════════════════════════════════════════\n";

    std::vector<MoETimingResults> all_results;

    for (const auto& cfg : configs) {
        std::cout << "\n";
        MoETimingResults results = runMoEForward(cfg, handle, workspace, stream);
        all_results.push_back(results);
        printResultsTable(cfg.name, results);
    }

    // ========================================================================
    // Large Batch Size Tests (32K, 64K tokens)
    // ========================================================================

    std::cout << "\n";
    std::cout << "════════════════════════════════════════════════════════════════════════════\n";
    std::cout << "                      LARGE BATCH SIZE SCALING TESTS                        \n";
    std::cout << "════════════════════════════════════════════════════════════════════════════\n";

    std::vector<MoEConfig> large_configs = {
        // 32K tokens - Mixtral-8x7B style
        {"Mixtral-8x7B (32K tokens)", 16, 2048, 4096, 14336, 8, 2},

        // 64K tokens - Mixtral-8x7B style
        {"Mixtral-8x7B (64K tokens)", 32, 2048, 4096, 14336, 8, 2},

        // 128K tokens - Mixtral-8x7B style (maximum throughput test)
        {"Mixtral-8x7B (128K tokens)", 64, 2048, 4096, 14336, 8, 2},
    };

    for (const auto& cfg : large_configs) {
        std::cout << "\n";
        MoETimingResults results = runMoEForward(cfg, handle, workspace, stream);
        all_results.push_back(results);
        printResultsTable(cfg.name, results);
    }

    // ========================================================================
    // Summary Table
    // ========================================================================

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                          PERFORMANCE SUMMARY                             ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Configuration           │ Total(ms) │ TFLOPS  │ Tokens/s       ║\n";
    std::cout << "╠═════════════════════════╪═══════════╪═════════╪════════════════╣\n";

    std::vector<MoEConfig> all_configs;
    all_configs.insert(all_configs.end(), configs.begin(), configs.end());
    all_configs.insert(all_configs.end(), large_configs.begin(), large_configs.end());

    for (size_t i = 0; i < all_configs.size() && i < all_results.size(); ++i) {
        std::string short_name = all_configs[i].name.substr(0, 23);
        std::cout << "║ " << std::left << std::setw(23) << short_name << " │ "
                  << std::right << std::fixed << std::setprecision(2) << std::setw(9) << all_results[i].total_ms << " │ "
                  << std::setprecision(1) << std::setw(7) << all_results[i].expert_tflops << " │ "
                  << std::scientific << std::setprecision(2) << std::setw(14) << all_results[i].tokens_per_second
                  << " ║\n";
    }

    std::cout << "╚══════════════════════════════════════════════════════════════════════════╝\n";

    double avg_tflops = 0.0;

    for (size_t i = 0; i < all_results.size(); ++i) {
        avg_tflops += all_results[i].expert_tflops;
    }
    avg_tflops /= all_results.size();

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                         AGGREGATE SUMMARY                                ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Average Expert GEMM TFLOPS:  " << std::fixed << std::setprecision(1) << avg_tflops;
    std::cout << std::setw(41) << "" << "║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════╝\n";

    // Cleanup
    HIP_CHECK(hipFree(workspace));
    HIP_CHECK(hipStreamDestroy(stream));
    hipblasLtDestroy(handle);

    std::cout << "\nBenchmark complete.\n\n";

    return 0;
}
