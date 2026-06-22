/**
 * FlashMoE HIP Port - Production-Scale MoE Forward Pass Benchmark
 *
 * Tests MoE forward pass with production-scale configurations to fully
 * utilize AMD Instinct MI450 (gfx1250) / MI450 (gfx1250) capabilities.
 *
 * Configurations tested:
 * 1. LLaMA/Mixtral-style MoE: 4096 hidden, 8 experts, top-2
 * 2. GPT-4 style MoE (estimated): 8192 hidden, 16 experts, top-2
 * 3. DeepSeek-style MoE: 5120 hidden, 64 experts, top-6
 *
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <string>

// FlashMoE HIP Headers
#include "flashmoe/hip/constants.hpp"
#include "flashmoe/hip/cuda_compat.hpp"
#include "flashmoe/hip/tensor.hpp"
#include "flashmoe/hip/tuning.hpp"

// rocWMMA for matrix multiply
#if __has_include(<rocwmma/rocwmma.hpp>)
    #define HAS_ROCWMMA 1
    #include <rocwmma/rocwmma.hpp>
#else
    #define HAS_ROCWMMA 0
    #warning "rocWMMA not found - GEMM tests will be limited"
#endif

// ============================================================================
// Error Checking Macros
// ============================================================================

#define HIP_CHECK(call)                                                          \
    do {                                                                         \
        hipError_t err = call;                                                   \
        if (err != hipSuccess) {                                                 \
            std::cerr << "HIP Error: " << hipGetErrorString(err)                 \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

#define HIP_CHECK_LAST()                                                         \
    do {                                                                         \
        hipError_t err = hipGetLastError();                                      \
        if (err != hipSuccess) {                                                 \
            std::cerr << "HIP Kernel Error: " << hipGetErrorString(err)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

// ============================================================================
// Production-Scale MoE Configurations
// ============================================================================

struct MoEScaleConfig {
    std::string name;
    int batch_size;
    int sequence_length;
    int hidden_dim;
    int intermediate_dim;
    int num_experts;
    int top_k;

    // Derived
    int total_tokens() const { return batch_size * sequence_length; }
    int expert_capacity() const {
        // EC = ceil(total_tokens * top_k / num_experts * capacity_factor)
        return static_cast<int>(std::ceil(total_tokens() * top_k * 1.25 / num_experts));
    }

    // Memory estimates (bytes) - use size_t to avoid overflow
    size_t input_memory() const { return static_cast<size_t>(total_tokens()) * hidden_dim * sizeof(__half); }
    size_t expert_weights_memory() const {
        return static_cast<size_t>(num_experts) * hidden_dim * intermediate_dim * 2 * sizeof(__half);  // up + down
    }
    size_t expert_buffer_memory() const {
        return static_cast<size_t>(num_experts) * expert_capacity() * intermediate_dim * sizeof(__half);
    }
    size_t total_memory() const {
        return input_memory() + expert_weights_memory() + expert_buffer_memory();
    }

    // FLOPs calculation
    double expert_gemm_flops() const {
        // Up projection: tokens × hidden × intermediate
        // Down projection: tokens × intermediate × hidden
        // Per expert, assuming balanced routing
        double tokens_per_expert = static_cast<double>(total_tokens() * top_k) / num_experts;
        double up_flops = 2.0 * tokens_per_expert * hidden_dim * intermediate_dim;
        double down_flops = 2.0 * tokens_per_expert * intermediate_dim * hidden_dim;
        return (up_flops + down_flops) * num_experts;
    }
};

// Pre-defined production configurations
const MoEScaleConfig LLAMA_MIXTRAL_CONFIG = {
    .name = "LLaMA/Mixtral-style",
    .batch_size = 8,
    .sequence_length = 2048,
    .hidden_dim = 4096,
    .intermediate_dim = 14336,
    .num_experts = 8,
    .top_k = 2
};

const MoEScaleConfig GPT4_STYLE_CONFIG = {
    .name = "GPT-4 style (estimated)",
    .batch_size = 4,
    .sequence_length = 4096,
    .hidden_dim = 8192,
    .intermediate_dim = 28672,
    .num_experts = 16,
    .top_k = 2
};

const MoEScaleConfig DEEPSEEK_STYLE_CONFIG = {
    .name = "DeepSeek-style",
    .batch_size = 8,
    .sequence_length = 2048,
    .hidden_dim = 5120,
    .intermediate_dim = 12288,
    .num_experts = 64,
    .top_k = 6
};

// Smaller test configurations for debugging
const MoEScaleConfig SMALL_TEST_CONFIG = {
    .name = "Small Test Config",
    .batch_size = 2,
    .sequence_length = 256,
    .hidden_dim = 512,
    .intermediate_dim = 2048,
    .num_experts = 4,
    .top_k = 2
};

const MoEScaleConfig MEDIUM_TEST_CONFIG = {
    .name = "Medium Test Config",
    .batch_size = 4,
    .sequence_length = 512,
    .hidden_dim = 1024,
    .intermediate_dim = 4096,
    .num_experts = 8,
    .top_k = 2
};

// ============================================================================
// Timing Utilities
// ============================================================================

class BenchmarkTimer {
    hipEvent_t start_, stop_;
    std::vector<float> measurements_;

public:
    BenchmarkTimer() {
        HIP_CHECK(hipEventCreate(&start_));
        HIP_CHECK(hipEventCreate(&stop_));
    }

    ~BenchmarkTimer() {
        (void)hipEventDestroy(start_);
        (void)hipEventDestroy(stop_);
    }

    void start(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(start_, stream));
    }

    float stop(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(stop_, stream));
        HIP_CHECK(hipEventSynchronize(stop_));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start_, stop_));
        measurements_.push_back(ms);
        return ms;
    }

    void reset() { measurements_.clear(); }

    float mean_ms() const {
        if (measurements_.empty()) return 0.0f;
        return std::accumulate(measurements_.begin(), measurements_.end(), 0.0f) / measurements_.size();
    }

    float median_ms() const {
        if (measurements_.empty()) return 0.0f;
        std::vector<float> sorted = measurements_;
        std::sort(sorted.begin(), sorted.end());
        return sorted[sorted.size() / 2];
    }

    float min_ms() const {
        if (measurements_.empty()) return 0.0f;
        return *std::min_element(measurements_.begin(), measurements_.end());
    }

    float max_ms() const {
        if (measurements_.empty()) return 0.0f;
        return *std::max_element(measurements_.begin(), measurements_.end());
    }

    float stddev_ms() const {
        if (measurements_.size() < 2) return 0.0f;
        float m = mean_ms();
        float sum_sq = 0.0f;
        for (float v : measurements_) {
            sum_sq += (v - m) * (v - m);
        }
        return std::sqrt(sum_sq / (measurements_.size() - 1));
    }
};

// ============================================================================
// Data Initialization Helpers
// ============================================================================

void initRandomFP16(__half* data, size_t size, float min_val = -0.1f, float max_val = 0.1f) {
    static std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(min_val, max_val);

    for (size_t i = 0; i < size; ++i) {
        data[i] = __float2half(dist(gen));
    }
}

void initZerosFP16(__half* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = __float2half(0.0f);
    }
}

// GPU kernel to initialize FP16 array with pseudo-random values
__global__ void initRandomGPU(__half* data, size_t size, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Simple pseudo-random using linear congruential generator
    unsigned int seed = idx * 1103515245u + 12345u;
    float val = (float)(seed % 10000) / 10000.0f - 0.5f;
    data[idx] = __float2half(val * scale);
}

// ============================================================================
// MoE Kernels
// ============================================================================

// Softmax kernel for gate computation
__global__ void softmaxKernel(
    const __half* __restrict__ input,  // [num_tokens, num_experts]
    float* __restrict__ probs,          // [num_tokens, num_experts]
    int num_tokens,
    int num_experts
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int e = 0; e < num_experts; ++e) {
        float val = __half2float(input[token_idx * num_experts + e]);
        max_val = fmaxf(max_val, val);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int e = 0; e < num_experts; ++e) {
        float val = __half2float(input[token_idx * num_experts + e]);
        float exp_val = expf(val - max_val);
        probs[token_idx * num_experts + e] = exp_val;
        sum += exp_val;
    }

    // Normalize
    for (int e = 0; e < num_experts; ++e) {
        probs[token_idx * num_experts + e] /= sum;
    }
}

// Top-K selection kernel
__global__ void topKSelectionKernel(
    const float* __restrict__ probs,
    int* __restrict__ expert_indices,
    float* __restrict__ expert_weights,
    int* __restrict__ expert_counts,
    int num_tokens,
    int num_experts,
    int top_k
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    // Simple top-k selection
    for (int k = 0; k < top_k; ++k) {
        float max_prob = -1.0f;
        int max_idx = -1;

        for (int e = 0; e < num_experts; ++e) {
            float prob = probs[token_idx * num_experts + e];

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

    // Renormalize weights
    float weight_sum = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        weight_sum += expert_weights[token_idx * top_k + k];
    }
    if (weight_sum > 0.0f) {
        for (int k = 0; k < top_k; ++k) {
            expert_weights[token_idx * top_k + k] /= weight_sum;
        }
    }
}

// Token dispatch kernel - routes tokens to expert buffers
__global__ void tokenDispatchKernel(
    const __half* __restrict__ input_tokens,
    __half* __restrict__ expert_buffers,
    const int* __restrict__ token_to_expert,
    int* __restrict__ expert_offsets,
    int num_tokens,
    int hidden_dim,
    int expert_capacity,
    int top_k
) {
    int assignment_idx = blockIdx.x;
    if (assignment_idx >= num_tokens * top_k) return;

    int token_idx = assignment_idx / top_k;
    int expert_idx = token_to_expert[assignment_idx];

    if (expert_idx < 0) return;

    int slot = atomicAdd(&expert_offsets[expert_idx], 1);
    if (slot >= expert_capacity) return;

    // Vectorized copy for efficiency
    for (int h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        int src_idx = token_idx * hidden_dim + h;
        int dst_idx = (expert_idx * expert_capacity + slot) * hidden_dim + h;
        expert_buffers[dst_idx] = input_tokens[src_idx];
    }
}

#if HAS_ROCWMMA

// Simple Expert GEMM kernel - naive implementation for correctness
// Uses one thread per output element
__global__ void expertGemmKernelNaive(
    const __half* __restrict__ input,   // [M, K] row-major
    const __half* __restrict__ weights, // [K, N] row-major
    __half* __restrict__ output,        // [M, N] row-major
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = __half2float(input[row * K + k]);
        float b = __half2float(weights[k * N + col]);
        sum += a * b;
    }

    output[row * N + col] = __float2half(sum);
}

// Expert GEMM kernel using rocWMMA - optimized version
// A: [M, K] row-major, B: [K, N] row-major stored as col-major [N, K]
template<int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void expertGemmKernel(
    const __half* __restrict__ input,   // [M, K] row-major
    const __half* __restrict__ weights, // [N, K] col-major (transposed storage)
    __half* __restrict__ output,        // [M, N] row-major
    int M, int K, int N
) {
    using FragA = rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, rocwmma::col_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

    const int warpM = blockIdx.y;
    const int warpN = blockIdx.x;

    // Bounds check
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    FragC acc;
    rocwmma::fill_fragment(acc, 0.0f);

    // Accumulate over K dimension
    const int K_tiles = (K + WMMA_K - 1) / WMMA_K;
    for (int kTile = 0; kTile < K_tiles; ++kTile) {
        int k = kTile * WMMA_K;
        if (k >= K) break;

        FragA fragA;
        FragB fragB;

        // A is row-major: A[warpM*WMMA_M, k] with stride K
        const __half* aPtr = input + warpM * WMMA_M * K + k;

        // B is col-major stored: B[warpN*WMMA_N, k] with stride K (N columns, K rows)
        const __half* bPtr = weights + warpN * WMMA_N * K + k;

        // Only load if we have full tiles
        if (k + WMMA_K <= K && warpM * WMMA_M + WMMA_M <= M && warpN * WMMA_N + WMMA_N <= N) {
            rocwmma::load_matrix_sync(fragA, aPtr, K);
            rocwmma::load_matrix_sync(fragB, bPtr, K);
            rocwmma::mma_sync(acc, fragA, fragB, acc);
        }
    }

    // Store result - only if in bounds
    if (warpM * WMMA_M + WMMA_M <= M && warpN * WMMA_N + WMMA_N <= N) {
        __shared__ float sAcc[WMMA_M * WMMA_N];
        rocwmma::store_matrix_sync(sAcc, acc, WMMA_N, rocwmma::mem_row_major);
        __syncthreads();

        __half* cPtr = output + warpM * WMMA_M * N + warpN * WMMA_N;
        int tid = threadIdx.x;
        for (int i = tid; i < WMMA_M * WMMA_N; i += blockDim.x) {
            int row = i / WMMA_N;
            int col = i % WMMA_N;
            cPtr[row * N + col] = __float2half(sAcc[i]);
        }
    }
}

#endif

// SiLU activation kernel
__global__ void siluKernel(__half* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = __half2float(data[idx]);
    float silu = x / (1.0f + expf(-x));
    data[idx] = __float2half(silu);
}

// Combine kernel - weighted sum of expert outputs
__global__ void combineKernel(
    const __half* __restrict__ expert_outputs,
    __half* __restrict__ output,
    const int* __restrict__ token_to_expert,
    const int* __restrict__ token_to_slot,
    const float* __restrict__ expert_weights,
    int num_tokens,
    int hidden_dim,
    int expert_capacity,
    int top_k
) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    for (int h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        float sum = 0.0f;

        for (int k = 0; k < top_k; ++k) {
            int assign_idx = token_idx * top_k + k;
            int expert_idx = token_to_expert[assign_idx];
            int slot = token_to_slot[assign_idx];
            float weight = expert_weights[assign_idx];

            if (expert_idx >= 0 && slot >= 0 && slot < expert_capacity) {
                int src_idx = (expert_idx * expert_capacity + slot) * hidden_dim + h;
                float val = __half2float(expert_outputs[src_idx]);
                sum += val * weight;
            }
        }

        output[token_idx * hidden_dim + h] = __float2half(sum);
    }
}

// ============================================================================
// Performance Results Structure
// ============================================================================

struct MoEBenchmarkResults {
    std::string config_name;
    int total_tokens;
    int hidden_dim;
    int intermediate_dim;
    int num_experts;
    int top_k;

    // Timing
    float gate_time_ms;
    float dispatch_time_ms;
    float expert_gemm_time_ms;
    float combine_time_ms;
    float total_time_ms;

    // Performance metrics
    double tokens_per_second;
    double expert_tflops;
    double memory_bandwidth_gbs;

    // Memory
    size_t memory_used_mb;

    void print() const {
        std::cout << "\n┌─────────────────────────────────────────────────────────────────────┐\n";
        std::cout << "│ " << std::left << std::setw(68) << config_name << "│\n";
        std::cout << "├─────────────────────────────────────────────────────────────────────┤\n";
        std::cout << "│ Configuration:                                                      │\n";
        std::cout << "│   Tokens: " << std::setw(10) << total_tokens
                  << " Hidden: " << std::setw(6) << hidden_dim
                  << " Intermediate: " << std::setw(6) << intermediate_dim << "      │\n";
        std::cout << "│   Experts: " << std::setw(3) << num_experts
                  << " Top-K: " << std::setw(2) << top_k
                  << " Memory: " << std::setw(6) << memory_used_mb << " MB                     │\n";
        std::cout << "├─────────────────────────────────────────────────────────────────────┤\n";
        std::cout << "│ Timing Breakdown (ms):                                              │\n";
        std::cout << "│   Gate:        " << std::setw(10) << std::fixed << std::setprecision(3) << gate_time_ms
                  << " (" << std::setw(5) << std::setprecision(1) << (gate_time_ms/total_time_ms*100) << "%)                           │\n";
        std::cout << "│   Dispatch:    " << std::setw(10) << std::setprecision(3) << dispatch_time_ms
                  << " (" << std::setw(5) << std::setprecision(1) << (dispatch_time_ms/total_time_ms*100) << "%)                           │\n";
        std::cout << "│   Expert GEMM: " << std::setw(10) << std::setprecision(3) << expert_gemm_time_ms
                  << " (" << std::setw(5) << std::setprecision(1) << (expert_gemm_time_ms/total_time_ms*100) << "%)                           │\n";
        std::cout << "│   Combine:     " << std::setw(10) << std::setprecision(3) << combine_time_ms
                  << " (" << std::setw(5) << std::setprecision(1) << (combine_time_ms/total_time_ms*100) << "%)                           │\n";
        std::cout << "│   ───────────────────────────────────                               │\n";
        std::cout << "│   Total:       " << std::setw(10) << std::setprecision(3) << total_time_ms << " ms                               │\n";
        std::cout << "├─────────────────────────────────────────────────────────────────────┤\n";
        std::cout << "│ Performance:                                                        │\n";
        std::cout << "│   Tokens/sec:  " << std::scientific << std::setprecision(2) << tokens_per_second << std::fixed << "                                      │\n";
        std::cout << "│   Expert GEMM: " << std::setw(10) << std::setprecision(2) << expert_tflops << " TFLOPS                             │\n";
        std::cout << "│   Memory BW:   " << std::setw(10) << std::setprecision(2) << memory_bandwidth_gbs << " GB/s                               │\n";
        std::cout << "└─────────────────────────────────────────────────────────────────────┘\n";
    }
};

// ============================================================================
// Main Benchmark Function
// ============================================================================

MoEBenchmarkResults benchmarkMoEConfig(const MoEScaleConfig& cfg, int warmup_iters, int bench_iters) {
    MoEBenchmarkResults results;
    results.config_name = cfg.name;
    results.total_tokens = cfg.total_tokens();
    results.hidden_dim = cfg.hidden_dim;
    results.intermediate_dim = cfg.intermediate_dim;
    results.num_experts = cfg.num_experts;
    results.top_k = cfg.top_k;
    results.memory_used_mb = cfg.total_memory() / (1024 * 1024);

    const int num_tokens = cfg.total_tokens();
    const int hidden_dim = cfg.hidden_dim;
    const int intermediate_dim = cfg.intermediate_dim;
    const int num_experts = cfg.num_experts;
    const int expert_capacity = cfg.expert_capacity();
    const int top_k = cfg.top_k;

    std::cout << "  Allocating memory for " << cfg.name << "...\n";
    std::cout << "    Tokens: " << num_tokens << ", Hidden: " << hidden_dim
              << ", Intermediate: " << intermediate_dim << "\n";
    std::cout << "    Experts: " << num_experts << ", Top-K: " << top_k
              << ", Expert Capacity: " << expert_capacity << "\n";

    // Only allocate small host buffers - use GPU random init for large tensors
    std::vector<__half> h_input(num_tokens * hidden_dim);
    std::vector<__half> h_output(num_tokens * hidden_dim);

    // Initialize input on host (small enough)
    initRandomFP16(h_input.data(), h_input.size(), -0.1f, 0.1f);

    // Allocate device memory
    __half *d_input, *d_expert_up, *d_expert_down, *d_output;
    __half *d_expert_buffer_up, *d_expert_buffer_down;
    float *d_gate_probs, *d_expert_weights;
    int *d_expert_indices, *d_expert_counts, *d_expert_offsets, *d_token_to_slot;

    // Use size_t for all calculations to avoid integer overflow
    size_t input_size = static_cast<size_t>(num_tokens) * hidden_dim * sizeof(__half);
    size_t expert_up_size = static_cast<size_t>(num_experts) * hidden_dim * intermediate_dim * sizeof(__half);
    size_t expert_down_size = static_cast<size_t>(num_experts) * intermediate_dim * hidden_dim * sizeof(__half);
    // Buffer for dispatched tokens (hidden_dim) and GEMM output (intermediate_dim)
    size_t expert_buffer_up_size = static_cast<size_t>(num_experts) * expert_capacity * hidden_dim * sizeof(__half);
    size_t expert_buffer_down_size = static_cast<size_t>(num_experts) * expert_capacity * intermediate_dim * sizeof(__half);
    size_t output_size = static_cast<size_t>(num_tokens) * hidden_dim * sizeof(__half);

    std::cout << "  Memory allocation sizes:\n";
    std::cout << "    Input: " << (input_size / 1e6) << " MB\n";
    std::cout << "    Expert up weights: " << (expert_up_size / 1e6) << " MB\n";
    std::cout << "    Expert buffer (tokens): " << (expert_buffer_up_size / 1e6) << " MB\n";
    std::cout << "    Expert buffer (output): " << (expert_buffer_down_size / 1e6) << " MB\n";

    // Also allocate gate logits - [num_tokens, num_experts]
    __half* d_gate_logits;
    size_t gate_logits_size = num_tokens * num_experts * sizeof(__half);

    HIP_CHECK(hipMalloc(&d_input, input_size));
    HIP_CHECK(hipMalloc(&d_gate_logits, gate_logits_size));
    HIP_CHECK(hipMalloc(&d_expert_up, expert_up_size));
    HIP_CHECK(hipMalloc(&d_expert_down, expert_down_size));
    HIP_CHECK(hipMalloc(&d_output, output_size));
    HIP_CHECK(hipMalloc(&d_expert_buffer_up, expert_buffer_up_size));
    HIP_CHECK(hipMalloc(&d_expert_buffer_down, expert_buffer_down_size));
    HIP_CHECK(hipMalloc(&d_gate_probs, num_tokens * num_experts * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_expert_weights, num_tokens * top_k * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_expert_indices, num_tokens * top_k * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_expert_counts, num_experts * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_expert_offsets, num_experts * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_token_to_slot, num_tokens * top_k * sizeof(int)));

    // Initialize gate logits with random values (small enough for host)
    std::vector<__half> h_gate_logits(num_tokens * num_experts);
    initRandomFP16(h_gate_logits.data(), h_gate_logits.size(), -1.0f, 1.0f);

    // Copy input data to device
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), input_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_gate_logits, h_gate_logits.data(), gate_logits_size, hipMemcpyHostToDevice));

    // Initialize large expert weights directly on GPU
    std::cout << "  Initializing expert weights on GPU...\n";
    {
        size_t up_size = static_cast<size_t>(num_experts) * hidden_dim * intermediate_dim;
        size_t down_size = static_cast<size_t>(num_experts) * intermediate_dim * hidden_dim;

        int threads = 256;
        int blocks_up = (up_size + threads - 1) / threads;
        int blocks_down = (down_size + threads - 1) / threads;

        hipLaunchKernelGGL(initRandomGPU, dim3(blocks_up), dim3(threads), 0, 0,
            d_expert_up, up_size, 0.1f);
        hipLaunchKernelGGL(initRandomGPU, dim3(blocks_down), dim3(threads), 0, 0,
            d_expert_down, down_size, 0.1f);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK_LAST();
    }

    BenchmarkTimer gate_timer, dispatch_timer, gemm_timer, combine_timer, total_timer;

    // Kernel configurations
    int threads_gate = 256;
    int blocks_gate = (num_tokens + threads_gate - 1) / threads_gate;
    int threads_dispatch = std::min(hidden_dim, 256);
    int blocks_dispatch = num_tokens * top_k;
    int threads_combine = std::min(hidden_dim, 256);
    int blocks_combine = num_tokens;

    // GEMM kernel configuration - use naive kernel for reliability
    dim3 gemm_block(16, 16);

    std::cout << "  Running warmup (" << warmup_iters << " iterations)...\n";

    // Warmup iterations
    for (int iter = 0; iter < warmup_iters; ++iter) {
        // Reset state
        HIP_CHECK(hipMemset(d_expert_counts, 0, num_experts * sizeof(int)));
        HIP_CHECK(hipMemset(d_expert_offsets, 0, num_experts * sizeof(int)));
        HIP_CHECK(hipMemset(d_expert_buffer_up, 0, expert_buffer_up_size));

        // Gate - using properly sized gate logits
        hipLaunchKernelGGL(softmaxKernel, dim3(blocks_gate), dim3(threads_gate), 0, 0,
            d_gate_logits, d_gate_probs, num_tokens, num_experts);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK_LAST();

        hipLaunchKernelGGL(topKSelectionKernel, dim3(blocks_gate), dim3(threads_gate), 0, 0,
            d_gate_probs, d_expert_indices, d_expert_weights, d_expert_counts,
            num_tokens, num_experts, top_k);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK_LAST();

        // Dispatch tokens to expert buffers
        hipLaunchKernelGGL(tokenDispatchKernel, dim3(blocks_dispatch), dim3(threads_dispatch), 0, 0,
            d_input, d_expert_buffer_up, d_expert_indices, d_expert_offsets,
            num_tokens, hidden_dim, expert_capacity, top_k);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK_LAST();

#if HAS_ROCWMMA
        // Expert GEMM (up projection for each expert) - use naive kernel
        // Input: [expert_capacity, hidden_dim], Weights: [hidden_dim, intermediate_dim]
        // Output: [expert_capacity, intermediate_dim]
        for (int e = 0; e < num_experts; ++e) {
            const __half* expert_input = d_expert_buffer_up + e * expert_capacity * hidden_dim;
            __half* expert_output = d_expert_buffer_down + e * expert_capacity * intermediate_dim;
            const __half* w_up = d_expert_up + e * hidden_dim * intermediate_dim;

            dim3 grid((intermediate_dim + gemm_block.x - 1) / gemm_block.x,
                      (expert_capacity + gemm_block.y - 1) / gemm_block.y);

            hipLaunchKernelGGL(expertGemmKernelNaive, grid, gemm_block, 0, 0,
                expert_input, w_up, expert_output,
                expert_capacity, hidden_dim, intermediate_dim);
        }
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK_LAST();
#endif
    }

    std::cout << "  Running benchmark (" << bench_iters << " iterations)...\n";

    // Pre-allocate vectors for slot reconstruction (outside loop for efficiency)
    std::vector<int> h_expert_indices_vec(num_tokens * top_k);
    std::vector<int> h_token_to_slot_vec(num_tokens * top_k);
    std::vector<int> slot_counters(num_experts, 0);

    // Benchmark iterations
    for (int iter = 0; iter < bench_iters; ++iter) {
        // Reset state
        HIP_CHECK(hipMemset(d_expert_counts, 0, num_experts * sizeof(int)));
        HIP_CHECK(hipMemset(d_expert_offsets, 0, num_experts * sizeof(int)));
        HIP_CHECK(hipMemset(d_expert_buffer_up, 0, expert_buffer_up_size));
        HIP_CHECK(hipDeviceSynchronize());

        total_timer.start();

        // Stage 1: Gate
        gate_timer.start();
        hipLaunchKernelGGL(softmaxKernel, dim3(blocks_gate), dim3(threads_gate), 0, 0,
            d_gate_logits, d_gate_probs, num_tokens, num_experts);
        hipLaunchKernelGGL(topKSelectionKernel, dim3(blocks_gate), dim3(threads_gate), 0, 0,
            d_gate_probs, d_expert_indices, d_expert_weights, d_expert_counts,
            num_tokens, num_experts, top_k);
        gate_timer.stop();

        // Stage 2: Dispatch
        dispatch_timer.start();
        hipLaunchKernelGGL(tokenDispatchKernel, dim3(blocks_dispatch), dim3(threads_dispatch), 0, 0,
            d_input, d_expert_buffer_up, d_expert_indices, d_expert_offsets,
            num_tokens, hidden_dim, expert_capacity, top_k);
        dispatch_timer.stop();

#if HAS_ROCWMMA
        // Stage 3: Expert GEMM (up projection for each expert)
        gemm_timer.start();
        for (int e = 0; e < num_experts; ++e) {
            const __half* expert_input = d_expert_buffer_up + e * expert_capacity * hidden_dim;
            __half* expert_output = d_expert_buffer_down + e * expert_capacity * intermediate_dim;
            const __half* w_up = d_expert_up + e * hidden_dim * intermediate_dim;

            dim3 grid((intermediate_dim + gemm_block.x - 1) / gemm_block.x,
                      (expert_capacity + gemm_block.y - 1) / gemm_block.y);

            hipLaunchKernelGGL(expertGemmKernelNaive, grid, gemm_block, 0, 0,
                expert_input, w_up, expert_output,
                expert_capacity, hidden_dim, intermediate_dim);
        }
        gemm_timer.stop();
#else
        gemm_timer.start();
        HIP_CHECK(hipDeviceSynchronize());
        gemm_timer.stop();
#endif

        // Stage 4: Combine
        combine_timer.start();
        // Reconstruct slot assignments
        HIP_CHECK(hipMemcpy(h_expert_indices_vec.data(), d_expert_indices,
            h_expert_indices_vec.size() * sizeof(int), hipMemcpyDeviceToHost));

        // Reset slot counters
        std::fill(slot_counters.begin(), slot_counters.end(), 0);

        for (int t = 0; t < num_tokens; ++t) {
            for (int k = 0; k < top_k; ++k) {
                int idx = t * top_k + k;
                int expert = h_expert_indices_vec[idx];
                if (expert >= 0 && expert < num_experts && slot_counters[expert] < expert_capacity) {
                    h_token_to_slot_vec[idx] = slot_counters[expert]++;
                } else {
                    h_token_to_slot_vec[idx] = -1;
                }
            }
        }

        HIP_CHECK(hipMemcpy(d_token_to_slot, h_token_to_slot_vec.data(),
            h_token_to_slot_vec.size() * sizeof(int), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(combineKernel, dim3(blocks_combine), dim3(threads_combine), 0, 0,
            d_expert_buffer_down, d_output, d_expert_indices, d_token_to_slot,
            d_expert_weights, num_tokens, hidden_dim, expert_capacity, top_k);
        combine_timer.stop();

        total_timer.stop();
    }

    HIP_CHECK_LAST();

    // Calculate results
    results.gate_time_ms = gate_timer.mean_ms();
    results.dispatch_time_ms = dispatch_timer.mean_ms();
    results.expert_gemm_time_ms = gemm_timer.mean_ms();
    results.combine_time_ms = combine_timer.mean_ms();
    results.total_time_ms = total_timer.mean_ms();

    results.tokens_per_second = num_tokens / (results.total_time_ms / 1000.0);

    // Expert GEMM TFLOPS calculation
    double gemm_flops = cfg.expert_gemm_flops();
    results.expert_tflops = (gemm_flops / 1e12) / (results.expert_gemm_time_ms / 1000.0);

    // Memory bandwidth (input + output for dispatch and combine)
    size_t bytes_transferred = input_size * 2 + expert_buffer_up_size + expert_buffer_down_size;
    results.memory_bandwidth_gbs = (bytes_transferred / 1e9) / (results.total_time_ms / 1000.0);

    // Cleanup
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_gate_logits));
    HIP_CHECK(hipFree(d_expert_up));
    HIP_CHECK(hipFree(d_expert_down));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_expert_buffer_up));
    HIP_CHECK(hipFree(d_expert_buffer_down));
    HIP_CHECK(hipFree(d_gate_probs));
    HIP_CHECK(hipFree(d_expert_weights));
    HIP_CHECK(hipFree(d_expert_indices));
    HIP_CHECK(hipFree(d_expert_counts));
    HIP_CHECK(hipFree(d_expert_offsets));
    HIP_CHECK(hipFree(d_token_to_slot));

    return results;
}

// ============================================================================
// Batch Size Scaling Test
// ============================================================================

void testBatchScaling(int hidden_dim, int intermediate_dim, int num_experts, int top_k) {
    std::cout << "\n╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           Batch Size Scaling Test                                    ║\n";
    std::cout << "║ Hidden: " << std::setw(5) << hidden_dim
              << " | Intermediate: " << std::setw(5) << intermediate_dim
              << " | Experts: " << std::setw(2) << num_experts
              << " | Top-K: " << top_k << "       ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n";

    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32};
    int seq_len = 2048;

    std::cout << "\n┌──────────┬──────────┬────────────┬──────────────┬─────────────┐\n";
    std::cout << "│ Batch    │ Tokens   │ Time (ms)  │ Tokens/sec   │ GEMM TFLOPS │\n";
    std::cout << "├──────────┼──────────┼────────────┼──────────────┼─────────────┤\n";

    double best_tokens_per_sec = 0;
    int optimal_batch = 1;

    for (int batch : batch_sizes) {
        // Check memory
        MoEScaleConfig cfg;
        cfg.name = "Batch Test";
        cfg.batch_size = batch;
        cfg.sequence_length = seq_len;
        cfg.hidden_dim = hidden_dim;
        cfg.intermediate_dim = intermediate_dim;
        cfg.num_experts = num_experts;
        cfg.top_k = top_k;

        size_t required_mem = cfg.total_memory();

        // Query available memory
        size_t free_mem, total_mem;
        HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));

        if (required_mem > free_mem * 0.9) {
            std::cout << "│ " << std::setw(8) << batch
                      << " │ " << std::setw(8) << cfg.total_tokens()
                      << " │  SKIPPED - insufficient memory               │\n";
            continue;
        }

        MoEBenchmarkResults results = benchmarkMoEConfig(cfg, 5, 20);

        std::cout << "│ " << std::setw(8) << batch
                  << " │ " << std::setw(8) << cfg.total_tokens()
                  << " │ " << std::setw(10) << std::fixed << std::setprecision(2) << results.total_time_ms
                  << " │ " << std::scientific << std::setprecision(2) << results.tokens_per_second
                  << " │ " << std::fixed << std::setw(11) << std::setprecision(2) << results.expert_tflops
                  << " │\n";

        if (results.tokens_per_second > best_tokens_per_sec) {
            best_tokens_per_sec = results.tokens_per_second;
            optimal_batch = batch;
        }
    }

    std::cout << "└──────────┴──────────┴────────────┴──────────────┴─────────────┘\n";
    std::cout << "\n  Optimal batch size for hidden_dim=" << hidden_dim << ": " << optimal_batch << "\n";
    std::cout << "  Best throughput: " << std::scientific << best_tokens_per_sec << " tokens/sec\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     FlashMoE HIP Port - Production-Scale MoE Forward Pass Benchmark ║\n";
    std::cout << "║                     AMD Instinct MI450 / MI450                     ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n";

    // Check device
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "ERROR: No HIP devices found!\n";
        return 1;
    }

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));

    std::cout << "\n┌──────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│ Device Information                                                   │\n";
    std::cout << "├──────────────────────────────────────────────────────────────────────┤\n";
    std::cout << "│ Name: " << std::left << std::setw(63) << props.name << "│\n";
    std::cout << "│ Architecture: " << std::setw(55) << props.gcnArchName << "│\n";
              << std::setw(51) << "" << "│\n";
    std::cout << "│ Warp Size: " << std::setw(58) << props.warpSize << "│\n";
              << std::setw(40) << "" << "│\n";
    std::cout << "│ rocWMMA Support: " << std::setw(52) << (HAS_ROCWMMA ? "Available" : "Not Available") << "│\n";
    std::cout << "└──────────────────────────────────────────────────────────────────────┘\n";

    HIP_CHECK(hipSetDevice(0));

    // Query available memory
    size_t free_mem, total_mem;
    HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
              << " GB / " << (total_mem / (1024.0 * 1024.0 * 1024.0)) << " GB\n";

    const int warmup_iters = 10;
    const int bench_iters = 100;

    std::vector<MoEBenchmarkResults> all_results;

    // ========================================================================
    // Test 0: Small Test (sanity check)
    // ========================================================================
    std::cout << "\n" << std::string(72, '=') << "\n";
    std::cout << "Test 0: " << SMALL_TEST_CONFIG.name << " (sanity check)\n";
    std::cout << std::string(72, '=') << "\n";

    {
        auto result = benchmarkMoEConfig(SMALL_TEST_CONFIG, 5, 20);
        result.print();
        all_results.push_back(result);
    }

    // ========================================================================
    // Test 0.5: Medium Test
    // ========================================================================
    std::cout << "\n" << std::string(72, '=') << "\n";
    std::cout << "Test 0.5: " << MEDIUM_TEST_CONFIG.name << "\n";
    std::cout << std::string(72, '=') << "\n";

    {
        auto result = benchmarkMoEConfig(MEDIUM_TEST_CONFIG, 5, 50);
        result.print();
        all_results.push_back(result);
    }

    // ========================================================================
    // Test 1: LLaMA/Mixtral-style MoE
    // ========================================================================
    std::cout << "\n" << std::string(72, '=') << "\n";
    std::cout << "Test 1: " << LLAMA_MIXTRAL_CONFIG.name << "\n";
    std::cout << std::string(72, '=') << "\n";

    if (LLAMA_MIXTRAL_CONFIG.total_memory() < free_mem * 0.9) {
        auto result = benchmarkMoEConfig(LLAMA_MIXTRAL_CONFIG, warmup_iters, bench_iters);
        result.print();
        all_results.push_back(result);
    } else {
        std::cout << "  SKIPPED: Insufficient memory (requires "
                  << (LLAMA_MIXTRAL_CONFIG.total_memory() / (1024.0*1024.0*1024.0)) << " GB)\n";
    }

    // ========================================================================
    // Test 2: GPT-4 style MoE
    // ========================================================================
    std::cout << "\n" << std::string(72, '=') << "\n";
    std::cout << "Test 2: " << GPT4_STYLE_CONFIG.name << "\n";
    std::cout << std::string(72, '=') << "\n";

    if (GPT4_STYLE_CONFIG.total_memory() < free_mem * 0.9) {
        auto result = benchmarkMoEConfig(GPT4_STYLE_CONFIG, warmup_iters, bench_iters);
        result.print();
        all_results.push_back(result);
    } else {
        std::cout << "  SKIPPED: Insufficient memory (requires "
                  << (GPT4_STYLE_CONFIG.total_memory() / (1024.0*1024.0*1024.0)) << " GB)\n";
    }

    // ========================================================================
    // Test 3: DeepSeek-style MoE
    // ========================================================================
    std::cout << "\n" << std::string(72, '=') << "\n";
    std::cout << "Test 3: " << DEEPSEEK_STYLE_CONFIG.name << "\n";
    std::cout << std::string(72, '=') << "\n";

    if (DEEPSEEK_STYLE_CONFIG.total_memory() < free_mem * 0.9) {
        auto result = benchmarkMoEConfig(DEEPSEEK_STYLE_CONFIG, warmup_iters, bench_iters);
        result.print();
        all_results.push_back(result);
    } else {
        std::cout << "  SKIPPED: Insufficient memory (requires "
                  << (DEEPSEEK_STYLE_CONFIG.total_memory() / (1024.0*1024.0*1024.0)) << " GB)\n";
    }

    // ========================================================================
    // Test 4: Batch Size Scaling Analysis
    // ========================================================================
    std::cout << "\n" << std::string(72, '=') << "\n";
    std::cout << "Test 4: Batch Size Scaling Analysis\n";
    std::cout << std::string(72, '=') << "\n";

    // Test with smaller config for scaling analysis
    testBatchScaling(1024, 4096, 8, 2);

    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                         BENCHMARK SUMMARY                            ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════╣\n";

    for (const auto& r : all_results) {
        std::cout << "║ " << std::left << std::setw(30) << r.config_name
                  << " │ " << std::setw(8) << std::fixed << std::setprecision(2) << r.total_time_ms << " ms"
                  << " │ " << std::setw(8) << r.expert_tflops << " TFLOPS │\n";
    }

    if (!all_results.empty()) {
        double best_tflops = 0;
        for (const auto& r : all_results) {
            best_tflops = std::max(best_tflops, r.expert_tflops);
        }
        std::cout << "║ Best achieved: " << std::setw(10) << std::setprecision(2) << best_tflops
                  << " TFLOPS                                      ║\n";
    }

    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n";

    std::cout << "\n  \033[32m✓ BENCHMARK COMPLETE\033[0m\n\n";

    return 0;
}
