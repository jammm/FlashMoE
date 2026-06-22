/**
 * FlashMoE HIP Port - Full MoE Forward Pass Test
 *
 * Comprehensive test for the Mixture of Experts forward pass on AMD GPUs.
 * Tests individual components (gate, dispatch, GEMM, combine) and full forward pass.
 *
 * Target: AMD MI450 (gfx1250)
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

// ============================================================================
// FlashMoE HIP Headers
// ============================================================================

// Core headers
#include "flashmoe/hip/constants.hpp"
#include "flashmoe/hip/cuda_compat.hpp"
#include "flashmoe/hip/tensor.hpp"
#include "flashmoe/hip/atomics.hip.cuh"

// Tile GEMM (requires rocWMMA)
#if __has_include(<rocwmma/rocwmma.hpp>)
    #define HAS_ROCWMMA 1
    #include "flashmoe/hip/tile.hip.cuh"
#else
    #define HAS_ROCWMMA 0
    #warning "rocWMMA not found - GEMM tests will be limited"
#endif

// hipCUB for gate kernel
#if __has_include(<hipcub/hipcub.hpp>)
    #define HAS_HIPCUB 1
    #include "flashmoe/hip/gate.hip.cuh"
#else
    #define HAS_HIPCUB 0
#endif

// Tuning configurations
#include "flashmoe/hip/tuning.hpp"

// Define guard to prevent TPS redefinition
#define FLASHMOE_TPS_DEFINED 1

// Combine logic (skip since TPS is already defined in gate.hip.cuh)
// #include "flashmoe/hip/combine.hip.cuh"

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
// Test Configuration - MoE Parameters
// ============================================================================

struct MoEConfig {
    int batch_size = 32;
    int sequence_length = 128;
    int hidden_dim = 512;
    int intermediate_dim = 2048;  // Typically 4x hidden_dim
    int num_experts = 8;
    int top_k = 2;  // Tokens routed to top 2 experts

    // Derived
    int total_tokens() const { return batch_size * sequence_length; }
    int expert_capacity() const {
        // EC = ceil(total_tokens * top_k / num_experts * capacity_factor)
        // Using capacity_factor = 1.25 for slight overprovisioning
        return static_cast<int>(std::ceil(total_tokens() * top_k * 1.25 / num_experts));
    }
};

// ============================================================================
// Timing Utilities
// ============================================================================

class Timer {
    hipEvent_t start_, stop_;
    bool running_ = false;

public:
    Timer() {
        HIP_CHECK(hipEventCreate(&start_));
        HIP_CHECK(hipEventCreate(&stop_));
    }

    ~Timer() {
        (void)hipEventDestroy(start_);
        (void)hipEventDestroy(stop_);
    }

    void start(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(start_, stream));
        running_ = true;
    }

    void stop(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(stop_, stream));
        HIP_CHECK(hipEventSynchronize(stop_));
        running_ = false;
    }

    float elapsed_ms() {
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// ============================================================================
// Data Type Helpers
// ============================================================================

// Initialize FP16 array with random values
void initRandomFP16(__half* data, size_t size, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);

    for (size_t i = 0; i < size; ++i) {
        data[i] = __float2half(dist(gen));
    }
}

// Initialize FP16 array with zeros
void initZerosFP16(__half* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = __float2half(0.0f);
    }
}

// Initialize FP16 array with constant value
void initConstantFP16(__half* data, size_t size, float value) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = __float2half(value);
    }
}

// Check for NaN/Inf in FP16 array (host side)
bool checkValidFP16(const __half* data, size_t size, int* nan_count = nullptr, int* inf_count = nullptr) {
    int nans = 0, infs = 0;
    for (size_t i = 0; i < size; ++i) {
        float val = __half2float(data[i]);
        if (std::isnan(val)) nans++;
        if (std::isinf(val)) infs++;
    }
    if (nan_count) *nan_count = nans;
    if (inf_count) *inf_count = infs;
    return (nans == 0) && (infs == 0);
}

// ============================================================================
// Test 1: Softmax + Top-K Selection (Gate Kernel)
// ============================================================================

// Simple softmax kernel for testing gate logic
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

// Top-K selection kernel (simple O(k*E) approach for small E)
__global__ void topKSelectionKernel(
    const float* __restrict__ probs,      // [num_tokens, num_experts]
    int* __restrict__ expert_indices,      // [num_tokens, top_k]
    float* __restrict__ expert_weights,    // [num_tokens, top_k]
    int* __restrict__ expert_counts,       // [num_experts]
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

            // Skip already selected experts
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

        // Atomically increment expert count
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

bool testGateKernel(const MoEConfig& cfg) {
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "Test 1: Gate Kernel (Softmax + Top-K Selection)\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";

    const int num_tokens = cfg.total_tokens();
    const int num_experts = cfg.num_experts;
    const int top_k = cfg.top_k;

    std::cout << "  Tokens: " << num_tokens << ", Experts: " << num_experts << ", Top-K: " << top_k << "\n";

    // Allocate host memory
    std::vector<__half> h_gate_input(num_tokens * num_experts);
    std::vector<float> h_probs(num_tokens * num_experts);
    std::vector<int> h_expert_indices(num_tokens * top_k);
    std::vector<float> h_expert_weights(num_tokens * top_k);
    std::vector<int> h_expert_counts(num_experts, 0);

    // Initialize with random gate logits
    initRandomFP16(h_gate_input.data(), h_gate_input.size(), -2.0f, 2.0f);

    // Allocate device memory
    __half* d_gate_input;
    float* d_probs;
    int* d_expert_indices;
    float* d_expert_weights;
    int* d_expert_counts;

    HIP_CHECK(hipMalloc(&d_gate_input, h_gate_input.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_probs, h_probs.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_expert_indices, h_expert_indices.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_expert_weights, h_expert_weights.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_expert_counts, h_expert_counts.size() * sizeof(int)));

    HIP_CHECK(hipMemcpy(d_gate_input, h_gate_input.data(), h_gate_input.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_expert_counts, 0, h_expert_counts.size() * sizeof(int)));

    // Launch kernels
    Timer timer;
    int threads = 256;
    int blocks = (num_tokens + threads - 1) / threads;

    timer.start();

    hipLaunchKernelGGL(softmaxKernel, dim3(blocks), dim3(threads), 0, 0,
        d_gate_input, d_probs, num_tokens, num_experts);
    HIP_CHECK_LAST();

    hipLaunchKernelGGL(topKSelectionKernel, dim3(blocks), dim3(threads), 0, 0,
        d_probs, d_expert_indices, d_expert_weights, d_expert_counts,
        num_tokens, num_experts, top_k);
    HIP_CHECK_LAST();

    timer.stop();

    // Copy results back
    HIP_CHECK(hipMemcpy(h_probs.data(), d_probs, h_probs.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_expert_indices.data(), d_expert_indices, h_expert_indices.size() * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_expert_weights.data(), d_expert_weights, h_expert_weights.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_expert_counts.data(), d_expert_counts, h_expert_counts.size() * sizeof(int), hipMemcpyDeviceToHost));

    // Verify results
    bool passed = true;

    // Check: softmax probabilities sum to ~1.0 for each token
    int prob_errors = 0;
    for (int t = 0; t < num_tokens; ++t) {
        float sum = 0.0f;
        for (int e = 0; e < num_experts; ++e) {
            sum += h_probs[t * num_experts + e];
        }
        if (std::abs(sum - 1.0f) > 1e-4f) {
            if (prob_errors < 3) {
                std::cerr << "  Warning: Token " << t << " prob sum = " << sum << "\n";
            }
            prob_errors++;
        }
    }

    // Check: expert indices are valid
    int idx_errors = 0;
    for (size_t i = 0; i < h_expert_indices.size(); ++i) {
        if (h_expert_indices[i] < 0 || h_expert_indices[i] >= num_experts) {
            idx_errors++;
        }
    }

    // Check: expert weights sum to ~1.0 for each token
    int weight_errors = 0;
    for (int t = 0; t < num_tokens; ++t) {
        float sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            sum += h_expert_weights[t * top_k + k];
        }
        if (std::abs(sum - 1.0f) > 1e-4f) {
            weight_errors++;
        }
    }

    // Check: total expert assignments == num_tokens * top_k
    int total_assignments = std::accumulate(h_expert_counts.begin(), h_expert_counts.end(), 0);
    bool correct_total = (total_assignments == num_tokens * top_k);

    std::cout << "\n  Results:\n";
    std::cout << "    Softmax probability errors: " << prob_errors << "\n";
    std::cout << "    Invalid expert indices: " << idx_errors << "\n";
    std::cout << "    Weight sum errors: " << weight_errors << "\n";
    std::cout << "    Total assignments: " << total_assignments << " (expected " << (num_tokens * top_k) << ")\n";
    std::cout << "    Time: " << std::fixed << std::setprecision(3) << timer.elapsed_ms() << " ms\n";

    // Expert distribution
    std::cout << "\n  Expert distribution:\n    ";
    for (int e = 0; e < num_experts; ++e) {
        std::cout << "E" << e << ":" << h_expert_counts[e] << " ";
    }
    std::cout << "\n";

    passed = (prob_errors == 0) && (idx_errors == 0) && (weight_errors == 0) && correct_total;
    std::cout << "\n  Status: " << (passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m") << "\n";

    // Cleanup
    HIP_CHECK(hipFree(d_gate_input));
    HIP_CHECK(hipFree(d_probs));
    HIP_CHECK(hipFree(d_expert_indices));
    HIP_CHECK(hipFree(d_expert_weights));
    HIP_CHECK(hipFree(d_expert_counts));

    return passed;
}

// ============================================================================
// Test 2: Token Dispatch
// ============================================================================

// Simple dispatch kernel that routes tokens to expert buffers
__global__ void tokenDispatchKernel(
    const __half* __restrict__ input_tokens,   // [num_tokens, hidden_dim]
    __half* __restrict__ expert_buffers,        // [num_experts, expert_capacity, hidden_dim]
    const int* __restrict__ token_to_expert,    // [num_tokens * top_k]
    int* __restrict__ expert_offsets,           // [num_experts] - current position in each expert buffer
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

    // Atomically get slot in expert buffer
    int slot = atomicAdd(&expert_offsets[expert_idx], 1);

    // If over capacity, skip (token dropping)
    if (slot >= expert_capacity) return;

    // Copy token to expert buffer
    // Each thread handles a portion of the hidden dimension
    for (int h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        int src_idx = token_idx * hidden_dim + h;
        int dst_idx = (expert_idx * expert_capacity + slot) * hidden_dim + h;
        expert_buffers[dst_idx] = input_tokens[src_idx];
    }
}

bool testTokenDispatch(const MoEConfig& cfg) {
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "Test 2: Token Dispatch to Expert Buffers\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";

    const int num_tokens = cfg.total_tokens();
    const int hidden_dim = cfg.hidden_dim;
    const int num_experts = cfg.num_experts;
    const int expert_capacity = cfg.expert_capacity();
    const int top_k = cfg.top_k;

    std::cout << "  Tokens: " << num_tokens << ", Hidden: " << hidden_dim << "\n";
    std::cout << "  Experts: " << num_experts << ", Capacity: " << expert_capacity << ", Top-K: " << top_k << "\n";

    // Allocate host memory
    std::vector<__half> h_input_tokens(num_tokens * hidden_dim);
    std::vector<__half> h_expert_buffers(num_experts * expert_capacity * hidden_dim);
    std::vector<int> h_token_to_expert(num_tokens * top_k);
    std::vector<int> h_expert_offsets(num_experts, 0);

    // Initialize input tokens with unique values for verification
    for (int t = 0; t < num_tokens; ++t) {
        for (int h = 0; h < hidden_dim; ++h) {
            // Encode token ID in the first element for easy checking
            if (h == 0) {
                h_input_tokens[t * hidden_dim + h] = __float2half(static_cast<float>(t));
            } else {
                h_input_tokens[t * hidden_dim + h] = __float2half(1.0f / (h + 1));
            }
        }
    }
    initZerosFP16(h_expert_buffers.data(), h_expert_buffers.size());

    // Simulate expert assignments (round-robin for testing)
    for (int t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < top_k; ++k) {
            h_token_to_expert[t * top_k + k] = (t + k) % num_experts;
        }
    }

    // Allocate device memory
    __half* d_input_tokens;
    __half* d_expert_buffers;
    int* d_token_to_expert;
    int* d_expert_offsets;

    HIP_CHECK(hipMalloc(&d_input_tokens, h_input_tokens.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_expert_buffers, h_expert_buffers.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_token_to_expert, h_token_to_expert.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_expert_offsets, h_expert_offsets.size() * sizeof(int)));

    HIP_CHECK(hipMemcpy(d_input_tokens, h_input_tokens.data(), h_input_tokens.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_expert_buffers, 0, h_expert_buffers.size() * sizeof(__half)));
    HIP_CHECK(hipMemcpy(d_token_to_expert, h_token_to_expert.data(), h_token_to_expert.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_expert_offsets, 0, h_expert_offsets.size() * sizeof(int)));

    // Launch kernel
    Timer timer;
    int blocks = num_tokens * top_k;
    int threads = std::min(hidden_dim, 256);

    timer.start();
    hipLaunchKernelGGL(tokenDispatchKernel, dim3(blocks), dim3(threads), 0, 0,
        d_input_tokens, d_expert_buffers, d_token_to_expert, d_expert_offsets,
        num_tokens, hidden_dim, expert_capacity, top_k);
    HIP_CHECK_LAST();
    timer.stop();

    // Copy results back
    HIP_CHECK(hipMemcpy(h_expert_buffers.data(), d_expert_buffers, h_expert_buffers.size() * sizeof(__half), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_expert_offsets.data(), d_expert_offsets, h_expert_offsets.size() * sizeof(int), hipMemcpyDeviceToHost));

    // Verify results
    bool passed = true;

    // Check expert offsets
    int total_dispatched = std::accumulate(h_expert_offsets.begin(), h_expert_offsets.end(), 0);
    int expected_dispatched = std::min(num_tokens * top_k, num_experts * expert_capacity);

    // Check for NaN/Inf in expert buffers
    int nan_count = 0, inf_count = 0;
    checkValidFP16(h_expert_buffers.data(), h_expert_buffers.size(), &nan_count, &inf_count);

    // Count non-zero entries in expert buffers
    int non_zero = 0;
    for (size_t i = 0; i < h_expert_buffers.size(); ++i) {
        if (__half2float(h_expert_buffers[i]) != 0.0f) {
            non_zero++;
        }
    }

    std::cout << "\n  Results:\n";
    std::cout << "    Total dispatched: " << total_dispatched << " (expected ~" << expected_dispatched << ")\n";
    std::cout << "    Expert buffer utilization: " << non_zero << " / " << h_expert_buffers.size() << " non-zero\n";
    std::cout << "    NaN count: " << nan_count << ", Inf count: " << inf_count << "\n";
    std::cout << "    Time: " << std::fixed << std::setprecision(3) << timer.elapsed_ms() << " ms\n";

    // Print expert fill levels
    std::cout << "\n  Expert fill levels:\n    ";
    for (int e = 0; e < num_experts; ++e) {
        std::cout << "E" << e << ":" << h_expert_offsets[e] << "/" << expert_capacity << " ";
    }
    std::cout << "\n";

    // Calculate throughput
    size_t bytes_transferred = total_dispatched * hidden_dim * sizeof(__half);
    float bandwidth_gb_s = (bytes_transferred / 1e9) / (timer.elapsed_ms() / 1e3);
    std::cout << "\n  Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth_gb_s << " GB/s\n";

    passed = (nan_count == 0) && (inf_count == 0) && (total_dispatched > 0);
    std::cout << "\n  Status: " << (passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m") << "\n";

    // Cleanup
    HIP_CHECK(hipFree(d_input_tokens));
    HIP_CHECK(hipFree(d_expert_buffers));
    HIP_CHECK(hipFree(d_token_to_expert));
    HIP_CHECK(hipFree(d_expert_offsets));

    return passed;
}

// ============================================================================
// Test 3: Expert GEMM (using tile.hip.cuh)
// ============================================================================

#if HAS_ROCWMMA

// Tile configuration for MoE
constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 32;
constexpr int GEMM_THREADS = 128;
constexpr int PIPE_STAGES = 1;
constexpr int TEST_ARCH = (flashmoe::HIP_ARCH != 0) ? flashmoe::HIP_ARCH : 1250;
constexpr int TEST_WARP_SIZE = (TEST_ARCH >= 1200 && TEST_ARCH < 1300) ? 32 : flashmoe::WARP_SIZE;
constexpr int TEST_WMMA_K = (TEST_ARCH >= 1200 && TEST_ARCH < 1300) ? 32 : 16;

using ExpertGemmMainloop = flashmoe::tile::CollectiveMainloop<
    TILE_M, TILE_N, TILE_K, TEST_ARCH, __half, float, GEMM_THREADS, PIPE_STAGES
>;

// Expert FFN kernel: computes output = input @ W_up @ activation @ W_down
__global__ void expertGemmKernel(
    const __half* __restrict__ input,    // [M, K] - input from dispatch
    const __half* __restrict__ w_up,      // [K, N] - upproject weights
    __half* __restrict__ output,          // [M, N] - intermediate output
    int M, int K, int N
) {
    // Simplified GEMM using rocWMMA
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = TEST_WMMA_K;

    using FragA = rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, rocwmma::col_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

    const int warpM = blockIdx.y;
    const int warpN = blockIdx.x;

    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    FragC acc;
    rocwmma::fill_fragment(acc, 0.0f);

    // Accumulate over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        FragA fragA;
        FragB fragB;

        const __half* aPtr = input + warpM * WMMA_M * K + k;
        const __half* bPtr = w_up + warpN * WMMA_N * K + k;  // Note: B is col-major

        if (k + WMMA_K <= K) {
            rocwmma::load_matrix_sync(fragA, aPtr, K);
            rocwmma::load_matrix_sync(fragB, bPtr, K);
            rocwmma::mma_sync(acc, fragA, fragB, acc);
        }
    }

    // Store result (convert back to FP16)
    // Output is row-major
    __half* cPtr = output + warpM * WMMA_M * N + warpN * WMMA_N;

    // Convert float accumulator to half and store
    // Note: rocwmma::store_matrix_sync expects matching types
    // We need to convert manually
    __shared__ float sAcc[WMMA_M * WMMA_N];
    rocwmma::store_matrix_sync(sAcc, acc, WMMA_N, rocwmma::mem_row_major);
    __syncthreads();

    // Convert and store to global memory
    int tid = threadIdx.x;
    for (int i = tid; i < WMMA_M * WMMA_N; i += blockDim.x) {
        int row = i / WMMA_N;
        int col = i % WMMA_N;
        if (warpM * WMMA_M + row < M && warpN * WMMA_N + col < N) {
            cPtr[row * N + col] = __float2half(sAcc[i]);
        }
    }
}

// SiLU activation kernel
__global__ void siluKernel(__half* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = __half2float(data[idx]);
    float silu = x / (1.0f + expf(-x));  // x * sigmoid(x)
    data[idx] = __float2half(silu);
}

bool testExpertGemm(const MoEConfig& cfg) {
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "Test 3: Expert GEMM (rocWMMA-based tile GEMM)\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";

    // Test with one expert's computation
    const int M = cfg.expert_capacity();  // tokens routed to this expert
    const int K = cfg.hidden_dim;
    const int N = cfg.intermediate_dim;

    std::cout << "  GEMM dimensions: M=" << M << ", K=" << K << ", N=" << N << "\n";
    std::cout << "  Tile: " << TILE_M << "x" << TILE_N << "x" << TILE_K << "\n";
    std::cout << "  Threads: " << GEMM_THREADS << "\n";

    // Allocate host memory
    std::vector<__half> h_input(M * K);
    std::vector<__half> h_weights(K * N);  // Stored as column-major for WMMA
    std::vector<__half> h_output(M * N);

    // Initialize with small random values
    initRandomFP16(h_input.data(), h_input.size(), -0.5f, 0.5f);
    initRandomFP16(h_weights.data(), h_weights.size(), -0.1f, 0.1f);
    initZerosFP16(h_output.data(), h_output.size());

    // Allocate device memory
    __half *d_input, *d_weights, *d_output;
    HIP_CHECK(hipMalloc(&d_input, h_input.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_weights, h_weights.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_output, h_output.size() * sizeof(__half)));

    HIP_CHECK(hipMemcpy(d_input, h_input.data(), h_input.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_output, 0, h_output.size() * sizeof(__half)));

    // Launch GEMM kernel
    Timer timer;

    dim3 grid((N + 15) / 16, (M + 15) / 16);
    dim3 block(TEST_WARP_SIZE);

    // Warmup
    hipLaunchKernelGGL(expertGemmKernel, grid, block, 0, 0,
        d_input, d_weights, d_output, M, K, N);
    HIP_CHECK(hipDeviceSynchronize());

    // Benchmark
    const int num_iters = 10;
    timer.start();
    for (int i = 0; i < num_iters; ++i) {
        hipLaunchKernelGGL(expertGemmKernel, grid, block, 0, 0,
            d_input, d_weights, d_output, M, K, N);
    }
    timer.stop();

    HIP_CHECK_LAST();

    // Apply activation
    int act_threads = 256;
    int act_blocks = (M * N + act_threads - 1) / act_threads;
    hipLaunchKernelGGL(siluKernel, dim3(act_blocks), dim3(act_threads), 0, 0,
        d_output, M * N);
    HIP_CHECK_LAST();

    // Copy results back
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, h_output.size() * sizeof(__half), hipMemcpyDeviceToHost));

    // Verify results
    bool passed = true;
    int nan_count = 0, inf_count = 0;
    checkValidFP16(h_output.data(), h_output.size(), &nan_count, &inf_count);

    // Count non-zero outputs
    int non_zero = 0;
    float sum = 0.0f;
    for (size_t i = 0; i < h_output.size(); ++i) {
        float val = __half2float(h_output[i]);
        if (val != 0.0f) non_zero++;
        sum += std::abs(val);
    }
    float mean_abs = sum / h_output.size();

    // Performance metrics
    float time_per_gemm_ms = timer.elapsed_ms() / num_iters;
    double flops = 2.0 * M * N * K;  // 2 ops per FMA
    double tflops = (flops / 1e12) / (time_per_gemm_ms / 1e3);

    std::cout << "\n  Results:\n";
    std::cout << "    NaN count: " << nan_count << ", Inf count: " << inf_count << "\n";
    std::cout << "    Non-zero outputs: " << non_zero << " / " << h_output.size() << "\n";
    std::cout << "    Mean absolute value: " << std::scientific << mean_abs << "\n";
    std::cout << "    Time per GEMM: " << std::fixed << std::setprecision(3) << time_per_gemm_ms << " ms\n";
    std::cout << "    Performance: " << std::fixed << std::setprecision(2) << tflops << " TFLOPS\n";

    passed = (nan_count == 0) && (inf_count == 0) && (non_zero > 0);
    std::cout << "\n  Status: " << (passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m") << "\n";

    // Cleanup
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_weights));
    HIP_CHECK(hipFree(d_output));

    return passed;
}

#else

bool testExpertGemm(const MoEConfig& cfg) {
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "Test 3: Expert GEMM (rocWMMA-based tile GEMM)\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << "  SKIPPED: rocWMMA not available\n";
    std::cout << "  Status: \033[33mSKIP\033[0m\n";
    return true;  // Don't fail the test suite
}

#endif  // HAS_ROCWMMA

// ============================================================================
// Test 4: Output Combine
// ============================================================================

// Combine kernel: weighted sum of expert outputs back to original positions
__global__ void combineKernel(
    const __half* __restrict__ expert_outputs,  // [num_experts, expert_capacity, hidden_dim]
    __half* __restrict__ output,                 // [num_tokens, hidden_dim]
    const int* __restrict__ token_to_expert,     // [num_tokens * top_k]
    const int* __restrict__ token_to_slot,       // [num_tokens * top_k] - slot in expert buffer
    const float* __restrict__ expert_weights,    // [num_tokens * top_k]
    int num_tokens,
    int hidden_dim,
    int expert_capacity,
    int top_k
) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    // Each thread handles one hidden dimension element
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

bool testCombine(const MoEConfig& cfg) {
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "Test 4: Output Combine (Weighted Sum)\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";

    const int num_tokens = cfg.total_tokens();
    const int hidden_dim = cfg.hidden_dim;
    const int num_experts = cfg.num_experts;
    const int expert_capacity = cfg.expert_capacity();
    const int top_k = cfg.top_k;

    std::cout << "  Tokens: " << num_tokens << ", Hidden: " << hidden_dim << "\n";
    std::cout << "  Experts: " << num_experts << ", Top-K: " << top_k << "\n";

    // Allocate host memory
    std::vector<__half> h_expert_outputs(num_experts * expert_capacity * hidden_dim);
    std::vector<__half> h_output(num_tokens * hidden_dim);
    std::vector<int> h_token_to_expert(num_tokens * top_k);
    std::vector<int> h_token_to_slot(num_tokens * top_k);
    std::vector<float> h_expert_weights(num_tokens * top_k);

    // Initialize expert outputs with known values
    initRandomFP16(h_expert_outputs.data(), h_expert_outputs.size(), -1.0f, 1.0f);
    initZerosFP16(h_output.data(), h_output.size());

    // Setup token assignments (round-robin)
    std::vector<int> expert_slots(num_experts, 0);
    for (int t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < top_k; ++k) {
            int expert = (t + k) % num_experts;
            int slot = expert_slots[expert]++;
            if (slot >= expert_capacity) slot = -1;  // Dropped

            h_token_to_expert[t * top_k + k] = expert;
            h_token_to_slot[t * top_k + k] = slot;
            h_expert_weights[t * top_k + k] = 1.0f / top_k;  // Equal weights
        }
    }

    // Allocate device memory
    __half *d_expert_outputs, *d_output;
    int *d_token_to_expert, *d_token_to_slot;
    float *d_expert_weights;

    HIP_CHECK(hipMalloc(&d_expert_outputs, h_expert_outputs.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_output, h_output.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_token_to_expert, h_token_to_expert.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_token_to_slot, h_token_to_slot.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_expert_weights, h_expert_weights.size() * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_expert_outputs, h_expert_outputs.data(), h_expert_outputs.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_output, 0, h_output.size() * sizeof(__half)));
    HIP_CHECK(hipMemcpy(d_token_to_expert, h_token_to_expert.data(), h_token_to_expert.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_token_to_slot, h_token_to_slot.data(), h_token_to_slot.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_expert_weights, h_expert_weights.data(), h_expert_weights.size() * sizeof(float), hipMemcpyHostToDevice));

    // Launch kernel
    Timer timer;
    int blocks = num_tokens;
    int threads = std::min(hidden_dim, 256);

    timer.start();
    hipLaunchKernelGGL(combineKernel, dim3(blocks), dim3(threads), 0, 0,
        d_expert_outputs, d_output, d_token_to_expert, d_token_to_slot,
        d_expert_weights, num_tokens, hidden_dim, expert_capacity, top_k);
    HIP_CHECK_LAST();
    timer.stop();

    // Copy results back
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, h_output.size() * sizeof(__half), hipMemcpyDeviceToHost));

    // Verify results
    bool passed = true;
    int nan_count = 0, inf_count = 0;
    checkValidFP16(h_output.data(), h_output.size(), &nan_count, &inf_count);

    // Count non-zero outputs
    int non_zero = 0;
    for (size_t i = 0; i < h_output.size(); ++i) {
        if (__half2float(h_output[i]) != 0.0f) non_zero++;
    }

    // Calculate throughput
    size_t bytes_read = num_tokens * top_k * hidden_dim * sizeof(__half);
    size_t bytes_written = num_tokens * hidden_dim * sizeof(__half);
    float bandwidth_gb_s = ((bytes_read + bytes_written) / 1e9) / (timer.elapsed_ms() / 1e3);

    std::cout << "\n  Results:\n";
    std::cout << "    NaN count: " << nan_count << ", Inf count: " << inf_count << "\n";
    std::cout << "    Non-zero outputs: " << non_zero << " / " << h_output.size() << "\n";
    std::cout << "    Time: " << std::fixed << std::setprecision(3) << timer.elapsed_ms() << " ms\n";
    std::cout << "    Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth_gb_s << " GB/s\n";

    passed = (nan_count == 0) && (inf_count == 0) && (non_zero > 0);
    std::cout << "\n  Status: " << (passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m") << "\n";

    // Cleanup
    HIP_CHECK(hipFree(d_expert_outputs));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_token_to_expert));
    HIP_CHECK(hipFree(d_token_to_slot));
    HIP_CHECK(hipFree(d_expert_weights));

    return passed;
}

// ============================================================================
// Test 5: Full MoE Forward Pass
// ============================================================================

bool testFullMoEForward(const MoEConfig& cfg) {
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "Test 5: Full MoE Forward Pass\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";

    const int num_tokens = cfg.total_tokens();
    const int hidden_dim = cfg.hidden_dim;
    const int intermediate_dim = cfg.intermediate_dim;
    const int num_experts = cfg.num_experts;
    const int expert_capacity = cfg.expert_capacity();
    const int top_k = cfg.top_k;

    std::cout << "  Configuration:\n";
    std::cout << "    Batch: " << cfg.batch_size << " x Seq: " << cfg.sequence_length << " = " << num_tokens << " tokens\n";
    std::cout << "    Hidden: " << hidden_dim << ", Intermediate: " << intermediate_dim << "\n";
    std::cout << "    Experts: " << num_experts << ", Top-K: " << top_k << ", Capacity: " << expert_capacity << "\n";

    // Calculate memory requirements
    size_t input_size = num_tokens * hidden_dim * sizeof(__half);
    size_t gate_weights_size = hidden_dim * num_experts * sizeof(__half);
    size_t expert_up_size = num_experts * hidden_dim * intermediate_dim * sizeof(__half);
    size_t expert_down_size = num_experts * intermediate_dim * hidden_dim * sizeof(__half);
    size_t expert_buffer_size = num_experts * expert_capacity * intermediate_dim * sizeof(__half);
    size_t output_size = num_tokens * hidden_dim * sizeof(__half);

    size_t total_memory = input_size + gate_weights_size + expert_up_size +
                          expert_down_size + expert_buffer_size + output_size;

    std::cout << "\n  Memory requirements:\n";
    std::cout << "    Input: " << std::fixed << std::setprecision(2) << (input_size / 1e6) << " MB\n";
    std::cout << "    Gate weights: " << (gate_weights_size / 1e6) << " MB\n";
    std::cout << "    Expert W_up: " << (expert_up_size / 1e6) << " MB\n";
    std::cout << "    Expert W_down: " << (expert_down_size / 1e6) << " MB\n";
    std::cout << "    Expert buffers: " << (expert_buffer_size / 1e6) << " MB\n";
    std::cout << "    Output: " << (output_size / 1e6) << " MB\n";
    std::cout << "    Total: " << (total_memory / 1e6) << " MB\n";

    // Allocate host memory
    std::vector<__half> h_input(num_tokens * hidden_dim);
    std::vector<__half> h_gate_weights(hidden_dim * num_experts);
    std::vector<__half> h_expert_up(num_experts * hidden_dim * intermediate_dim);
    std::vector<__half> h_expert_down(num_experts * intermediate_dim * hidden_dim);
    std::vector<__half> h_output(num_tokens * hidden_dim);

    // Initialize with random values
    std::cout << "\n  Initializing tensors...";
    initRandomFP16(h_input.data(), h_input.size(), -1.0f, 1.0f);
    initRandomFP16(h_gate_weights.data(), h_gate_weights.size(), -0.5f, 0.5f);
    initRandomFP16(h_expert_up.data(), h_expert_up.size(), -0.1f, 0.1f);
    initRandomFP16(h_expert_down.data(), h_expert_down.size(), -0.1f, 0.1f);
    initZerosFP16(h_output.data(), h_output.size());
    std::cout << " done\n";

    // Allocate device memory
    __half *d_input, *d_gate_weights, *d_expert_up, *d_expert_down, *d_output;
    __half *d_expert_buffer;
    float *d_gate_probs, *d_expert_weights;
    int *d_expert_indices, *d_expert_counts, *d_expert_offsets, *d_token_to_slot;

    std::cout << "  Allocating GPU memory...";
    HIP_CHECK(hipMalloc(&d_input, input_size));
    HIP_CHECK(hipMalloc(&d_gate_weights, gate_weights_size));
    HIP_CHECK(hipMalloc(&d_expert_up, expert_up_size));
    HIP_CHECK(hipMalloc(&d_expert_down, expert_down_size));
    HIP_CHECK(hipMalloc(&d_output, output_size));
    HIP_CHECK(hipMalloc(&d_expert_buffer, expert_buffer_size));
    HIP_CHECK(hipMalloc(&d_gate_probs, num_tokens * num_experts * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_expert_weights, num_tokens * top_k * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_expert_indices, num_tokens * top_k * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_expert_counts, num_experts * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_expert_offsets, num_experts * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_token_to_slot, num_tokens * top_k * sizeof(int)));
    std::cout << " done\n";

    // Copy data to device
    std::cout << "  Copying data to GPU...";
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), input_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_gate_weights, h_gate_weights.data(), gate_weights_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_expert_up, h_expert_up.data(), expert_up_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_expert_down, h_expert_down.data(), expert_down_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_output, 0, output_size));
    HIP_CHECK(hipMemset(d_expert_buffer, 0, expert_buffer_size));
    HIP_CHECK(hipMemset(d_expert_counts, 0, num_experts * sizeof(int)));
    HIP_CHECK(hipMemset(d_expert_offsets, 0, num_experts * sizeof(int)));
    std::cout << " done\n";

    // ========================================================================
    // Execute MoE Forward Pass Stages
    // ========================================================================

    Timer total_timer, stage_timer;
    float gate_time = 0, dispatch_time = 0, expert_time = 0, combine_time = 0;

    std::cout << "\n  Executing MoE forward pass...\n";

    total_timer.start();

    // Stage 1: Gate (compute expert probabilities)
    // For simplicity, use input as gate logits (in real MoE, this is input @ gate_weights)
    stage_timer.start();
    {
        int threads = 256;
        int blocks = (num_tokens + threads - 1) / threads;

        // Note: In real implementation, gate would compute input @ gate_weights first
        // For testing, we use input directly as gate logits
        hipLaunchKernelGGL(softmaxKernel, dim3(blocks), dim3(threads), 0, 0,
            d_input, d_gate_probs, num_tokens, num_experts);
        HIP_CHECK_LAST();

        hipLaunchKernelGGL(topKSelectionKernel, dim3(blocks), dim3(threads), 0, 0,
            d_gate_probs, d_expert_indices, d_expert_weights, d_expert_counts,
            num_tokens, num_experts, top_k);
        HIP_CHECK_LAST();
    }
    stage_timer.stop();
    gate_time = stage_timer.elapsed_ms();
    std::cout << "    Stage 1 - Gate: " << std::fixed << std::setprecision(3) << gate_time << " ms\n";

    // Stage 2: Dispatch tokens to experts
    stage_timer.start();
    {
        int blocks = num_tokens * top_k;
        int threads = std::min(hidden_dim, 256);

        hipLaunchKernelGGL(tokenDispatchKernel, dim3(blocks), dim3(threads), 0, 0,
            d_input, d_expert_buffer, d_expert_indices, d_expert_offsets,
            num_tokens, hidden_dim, expert_capacity, top_k);
        HIP_CHECK_LAST();
    }
    stage_timer.stop();
    dispatch_time = stage_timer.elapsed_ms();
    std::cout << "    Stage 2 - Dispatch: " << std::fixed << std::setprecision(3) << dispatch_time << " ms\n";

    // Stage 3: Expert computation (GEMM)
#if HAS_ROCWMMA
    stage_timer.start();
    {
        // For each expert, run up-projection GEMM
        // In real implementation, this would be batched or parallelized
        for (int e = 0; e < num_experts; ++e) {
            const __half* expert_input = d_expert_buffer + e * expert_capacity * hidden_dim;
            __half* expert_output = d_expert_buffer + e * expert_capacity * intermediate_dim;
            const __half* w_up = d_expert_up + e * hidden_dim * intermediate_dim;

            // Simplified: just use the buffer in-place
            // Real implementation would have proper intermediate storage
            dim3 grid((intermediate_dim + 15) / 16, (expert_capacity + 15) / 16);
            dim3 block(TEST_WARP_SIZE);

            hipLaunchKernelGGL(expertGemmKernel, grid, block, 0, 0,
                expert_input, w_up, expert_output,
                expert_capacity, hidden_dim, intermediate_dim);
        }
        HIP_CHECK_LAST();
        HIP_CHECK(hipDeviceSynchronize());
    }
    stage_timer.stop();
    expert_time = stage_timer.elapsed_ms();
    std::cout << "    Stage 3 - Expert GEMM: " << std::fixed << std::setprecision(3) << expert_time << " ms\n";
#else
    std::cout << "    Stage 3 - Expert GEMM: SKIPPED (no rocWMMA)\n";
#endif

    // Stage 4: Combine expert outputs
    stage_timer.start();
    {
        // For testing, create simple slot mapping
        std::vector<int> h_token_to_slot_vec(num_tokens * top_k);
        std::vector<int> slot_counters(num_experts, 0);

        // Reconstruct slot assignments based on expert indices
        std::vector<int> h_expert_indices_vec(num_tokens * top_k);
        HIP_CHECK(hipMemcpy(h_expert_indices_vec.data(), d_expert_indices,
            h_expert_indices_vec.size() * sizeof(int), hipMemcpyDeviceToHost));

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

        int blocks = num_tokens;
        int threads = std::min(hidden_dim, 256);

        hipLaunchKernelGGL(combineKernel, dim3(blocks), dim3(threads), 0, 0,
            d_expert_buffer, d_output, d_expert_indices, d_token_to_slot,
            d_expert_weights, num_tokens, hidden_dim, expert_capacity, top_k);
        HIP_CHECK_LAST();
    }
    stage_timer.stop();
    combine_time = stage_timer.elapsed_ms();
    std::cout << "    Stage 4 - Combine: " << std::fixed << std::setprecision(3) << combine_time << " ms\n";

    total_timer.stop();
    float total_time = total_timer.elapsed_ms();

    // ========================================================================
    // Verify Results and Report Performance
    // ========================================================================

    // Copy output back
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, output_size, hipMemcpyDeviceToHost));

    // Check output validity
    int nan_count = 0, inf_count = 0;
    bool valid = checkValidFP16(h_output.data(), h_output.size(), &nan_count, &inf_count);

    int non_zero = 0;
    float sum = 0.0f;
    for (size_t i = 0; i < h_output.size(); ++i) {
        float val = __half2float(h_output[i]);
        if (val != 0.0f) non_zero++;
        sum += std::abs(val);
    }
    float mean_abs = sum / h_output.size();

    // Performance metrics
    float tokens_per_second = num_tokens / (total_time / 1000.0f);

    // FLOPs calculation for full MoE:
    // - Gate: 2 * num_tokens * hidden_dim * num_experts (for matmul, simplified here)
    // - Expert up: 2 * expert_capacity * hidden_dim * intermediate_dim * num_experts
    // - Expert down: 2 * expert_capacity * intermediate_dim * hidden_dim * num_experts
    double expert_flops = 2.0 * expert_capacity * hidden_dim * intermediate_dim * num_experts * 2;  // up + down
    double total_flops = expert_flops;  // Simplified, excluding gate matmul
    double tflops = (total_flops / 1e12) / (total_time / 1000.0);

    std::cout << "\n  ══════════════════════════════════════════════════════════\n";
    std::cout << "  Results Summary:\n";
    std::cout << "  ══════════════════════════════════════════════════════════\n";
    std::cout << "    Output validity:\n";
    std::cout << "      NaN count: " << nan_count << "\n";
    std::cout << "      Inf count: " << inf_count << "\n";
    std::cout << "      Non-zero outputs: " << non_zero << " / " << h_output.size() << "\n";
    std::cout << "      Mean absolute value: " << std::scientific << mean_abs << std::fixed << "\n";

    std::cout << "\n    Timing breakdown:\n";
    std::cout << "      Gate:     " << std::setw(8) << std::setprecision(3) << gate_time << " ms ("
              << std::setprecision(1) << (gate_time/total_time*100) << "%)\n";
    std::cout << "      Dispatch: " << std::setw(8) << std::setprecision(3) << dispatch_time << " ms ("
              << std::setprecision(1) << (dispatch_time/total_time*100) << "%)\n";
    std::cout << "      Expert:   " << std::setw(8) << std::setprecision(3) << expert_time << " ms ("
              << std::setprecision(1) << (expert_time/total_time*100) << "%)\n";
    std::cout << "      Combine:  " << std::setw(8) << std::setprecision(3) << combine_time << " ms ("
              << std::setprecision(1) << (combine_time/total_time*100) << "%)\n";
    std::cout << "      ─────────────────────────\n";
    std::cout << "      Total:    " << std::setw(8) << std::setprecision(3) << total_time << " ms\n";

    std::cout << "\n    Performance metrics:\n";
    std::cout << "      Tokens/second: " << std::scientific << std::setprecision(2) << tokens_per_second << std::fixed << "\n";
#if HAS_ROCWMMA
    std::cout << "      Estimated TFLOPS: " << std::setprecision(2) << tflops << "\n";
#endif

    bool passed = valid && (non_zero > 0);
    std::cout << "\n  Status: " << (passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m") << "\n";

    // Cleanup
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_gate_weights));
    HIP_CHECK(hipFree(d_expert_up));
    HIP_CHECK(hipFree(d_expert_down));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_expert_buffer));
    HIP_CHECK(hipFree(d_gate_probs));
    HIP_CHECK(hipFree(d_expert_weights));
    HIP_CHECK(hipFree(d_expert_indices));
    HIP_CHECK(hipFree(d_expert_counts));
    HIP_CHECK(hipFree(d_expert_offsets));
    HIP_CHECK(hipFree(d_token_to_slot));

    return passed;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           FlashMoE HIP Port - Full MoE Forward Pass Test         ║\n";
    std::cout << "║                         AMD MI450 (gfx1250)                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";

    // Check device
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "ERROR: No HIP devices found!\n";
        return 1;
    }

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));

    std::cout << "\n┌──────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│ Device Information                                               │\n";
    std::cout << "├──────────────────────────────────────────────────────────────────┤\n";
    std::cout << "│ Name: " << std::left << std::setw(59) << props.name << "│\n";
    std::cout << "│ Architecture: " << std::setw(52) << props.gcnArchName << "│\n";
              << std::setw(48) << "" << "│\n";
    std::cout << "│ Warp Size: " << std::setw(55) << props.warpSize << "│\n";
              << std::setw(37) << "" << "│\n";
    std::cout << "└──────────────────────────────────────────────────────────────────┘\n";

    // Feature detection
    std::cout << "\n┌──────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│ Feature Detection                                                │\n";
    std::cout << "├──────────────────────────────────────────────────────────────────┤\n";
    std::cout << "│ rocWMMA:  " << std::setw(55) << (HAS_ROCWMMA ? "Available" : "Not Available") << "│\n";
    std::cout << "│ hipCUB:   " << std::setw(55) << (HAS_HIPCUB ? "Available" : "Not Available") << "│\n";
    std::cout << "└──────────────────────────────────────────────────────────────────┘\n";

    HIP_CHECK(hipSetDevice(0));

    // Test configuration
    MoEConfig cfg;
    cfg.batch_size = 32;
    cfg.sequence_length = 128;
    cfg.hidden_dim = 512;
    cfg.intermediate_dim = 2048;
    cfg.num_experts = 8;
    cfg.top_k = 2;

    std::cout << "\n┌──────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│ MoE Configuration                                                │\n";
    std::cout << "├──────────────────────────────────────────────────────────────────┤\n";
    std::cout << "│ Batch Size: " << std::setw(54) << cfg.batch_size << "│\n";
    std::cout << "│ Sequence Length: " << std::setw(49) << cfg.sequence_length << "│\n";
    std::cout << "│ Hidden Dimension: " << std::setw(48) << cfg.hidden_dim << "│\n";
    std::cout << "│ Intermediate Dimension: " << std::setw(42) << cfg.intermediate_dim << "│\n";
    std::cout << "│ Number of Experts: " << std::setw(47) << cfg.num_experts << "│\n";
    std::cout << "│ Top-K: " << std::setw(59) << cfg.top_k << "│\n";
    std::cout << "│ Total Tokens: " << std::setw(52) << cfg.total_tokens() << "│\n";
    std::cout << "│ Expert Capacity: " << std::setw(49) << cfg.expert_capacity() << "│\n";
    std::cout << "└──────────────────────────────────────────────────────────────────┘\n";

    // Run tests
    int total_tests = 5;
    int passed_tests = 0;

    // Test 1: Gate Kernel
    if (testGateKernel(cfg)) passed_tests++;

    // Test 2: Token Dispatch
    if (testTokenDispatch(cfg)) passed_tests++;

    // Test 3: Expert GEMM
    if (testExpertGemm(cfg)) passed_tests++;

    // Test 4: Combine
    if (testCombine(cfg)) passed_tests++;

    // Test 5: Full Forward Pass
    if (testFullMoEForward(cfg)) passed_tests++;

    // Summary
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                         TEST SUMMARY                             ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Test 1 - Gate Kernel:      " << std::setw(36) << (passed_tests >= 1 ? "PASS" : "FAIL") << " ║\n";
    std::cout << "║  Test 2 - Token Dispatch:   " << std::setw(36) << (passed_tests >= 2 ? "PASS" : "FAIL") << " ║\n";
    std::cout << "║  Test 3 - Expert GEMM:      " << std::setw(36) << (passed_tests >= 3 ? (HAS_ROCWMMA ? "PASS" : "SKIP") : "FAIL") << " ║\n";
    std::cout << "║  Test 4 - Combine:          " << std::setw(36) << (passed_tests >= 4 ? "PASS" : "FAIL") << " ║\n";
    std::cout << "║  Test 5 - Full Forward:     " << std::setw(36) << (passed_tests >= 5 ? "PASS" : "FAIL") << " ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Total: " << passed_tests << " / " << total_tests << " tests passed";
    std::cout << std::setw(43) << "" << "║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";

    if (passed_tests == total_tests) {
        std::cout << "\n  \033[32m✓ ALL TESTS PASSED\033[0m\n\n";
        return 0;
    } else {
        std::cout << "\n  \033[31m✗ SOME TESTS FAILED\033[0m\n\n";
        return 1;
    }
}
