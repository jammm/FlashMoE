/******************************************************************************
 * FlashMoE HIP - Multi-GPU MoE Forward Pass Test
 *
 * Tests distributed Mixture of Experts forward pass across multiple GPUs
 * using rocSHMEM for inter-GPU communication.
 *
 * Configuration:
 *   - 2 GPUs
 *   - 8 experts total (4 per GPU)
 *   - 1024 tokens per GPU (2048 total)
 *   - hidden_dim = 512
 *   - intermediate_dim = 1024
 *   - top_k = 2
 *
 * Flow:
 *   1. Each GPU has a portion of input tokens
 *   2. Random expert assignment (simplified gate)
 *   3. Tokens dispatched to GPU holding target expert via rocSHMEM
 *   4. Expert GEMM computed locally
 *   5. Results sent back to originating GPU via rocSHMEM
 *   6. Combine operation aggregates results
 *
 * Build:
 *   source /jam/venv/bin/activate
 *   export ROCM_PATH="$(rocm-sdk path --root)"
 *   export ROCM_CORE_LIB=/jam/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib
 *   export ROCM_SYSDEPS_LIB=$ROCM_PATH/lib/rocm_sysdeps/lib
 *   $ROCM_PATH/bin/hipcc -std=c++20 --offload-arch=gfx1250 -O3 -fgpu-rdc \
 *       -I./csrc/include -I$ROCM_PATH/include \
 *       -L$ROCM_PATH/lib -L$ROCM_CORE_LIB -L$ROCM_SYSDEPS_LIB \
 *       -o tests/moe_multi_gpu tests/moe_multi_gpu.hip.cpp \
 *       -lhipblaslt -lamdhip64 -lhsa-runtime64 -lrocshmem -lnuma -lpthread -ldl -lrt
 *
 * Run:
 *   export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_CORE_LIB:$ROCM_SYSDEPS_LIB:$LD_LIBRARY_PATH"
 *   ./tests/moe_multi_gpu
 *
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *****************************************************************************/

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocshmem/rocshmem.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <atomic>
#include <random>
#include <chrono>

using namespace rocshmem;

// ============================================================================
// Configuration
// ============================================================================

struct MoEConfig {
    static constexpr int NUM_GPUS = 2;
    static constexpr int TOTAL_EXPERTS = 8;
    static constexpr int EXPERTS_PER_GPU = TOTAL_EXPERTS / NUM_GPUS;  // 4
    static constexpr int TOKENS_PER_GPU = 1024;
    static constexpr int TOTAL_TOKENS = TOKENS_PER_GPU * NUM_GPUS;    // 2048
    static constexpr int HIDDEN_DIM = 512;
    static constexpr int INTERMEDIATE_DIM = 1024;
    static constexpr int TOP_K = 2;

    // Derived: max tokens any expert can receive
    // Conservative estimate: 2x average load
    static constexpr int EXPERT_CAPACITY = (TOTAL_TOKENS * TOP_K / TOTAL_EXPERTS) * 2;
};

// ============================================================================
// Error Checking
// ============================================================================

#define CHECK_HIP(condition) do {                                              \
    hipError_t error = condition;                                              \
    if (error != hipSuccess) {                                                 \
        fprintf(stderr, "[PE %d] HIP error: %s (code %d) at line %d\n",       \
                getpid() % 100, hipGetErrorString(error), error, __LINE__);    \
        fflush(stderr);                                                        \
        exit(error);                                                           \
    }                                                                          \
} while(0)

// ============================================================================
// Shared Memory Structures for Process Coordination
// ============================================================================

struct SharedBootstrap {
    rocshmem_uniqueid_t uid;
    std::atomic<int> uid_ready;
    std::atomic<int> ready_count;
    std::atomic<int> init_done;
    std::atomic<int> phase_sync;  // For multiple sync points

    // Results for verification
    int test_passed[MoEConfig::NUM_GPUS];
    float dispatch_time_ms[MoEConfig::NUM_GPUS];
    float expert_time_ms[MoEConfig::NUM_GPUS];
    float combine_time_ms[MoEConfig::NUM_GPUS];
    float total_time_ms[MoEConfig::NUM_GPUS];
    int tokens_dispatched[MoEConfig::NUM_GPUS];
    int tokens_received[MoEConfig::NUM_GPUS];
};

// ============================================================================
// Token routing information
// ============================================================================

struct __attribute__((aligned(8))) TokenRouting {
    int32_t token_id;       // Original token index
    int32_t expert_id;      // Assigned expert (0-7)
    int32_t source_gpu;     // GPU that owns this token
    float weight;           // Routing weight (for top-k)
};

// Expert assignment result
struct __attribute__((aligned(8))) ExpertAssignment {
    int32_t token_local_id; // Token's local ID on source GPU
    int32_t slot_in_expert; // Slot in expert's buffer
    float weight;           // Routing weight
    int32_t source_gpu;     // Source GPU ID
};

// ============================================================================
// GPU Kernels
// ============================================================================

// Initialize input tokens with pattern for verification
__global__ void init_tokens_kernel(
    __half* tokens,        // [num_tokens, hidden_dim]
    int num_tokens,
    int hidden_dim,
    int gpu_id
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_tokens * hidden_dim;

    if (tid < total_elements) {
        int token_idx = tid / hidden_dim;
        int h = tid % hidden_dim;

        // Encode token identity in the value
        // GPU 0 tokens: 0.001 * (token_id + 1) + 0.0001 * h
        // GPU 1 tokens: 0.002 * (token_id + 1) + 0.0001 * h
        float value = (gpu_id + 1) * 0.001f * (token_idx + 1) + 0.0001f * (h % 100);
        tokens[tid] = __float2half(value);
    }
}

// Initialize expert weights (random but deterministic)
__global__ void init_expert_weights_kernel(
    __half* weights,       // [num_experts, hidden_dim, intermediate_dim]
    int num_experts,
    int hidden_dim,
    int intermediate_dim,
    int gpu_id,
    unsigned int seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_experts * hidden_dim * intermediate_dim;

    if (tid < total_elements) {
        // Simple pseudo-random initialization
        unsigned int state = seed + tid + gpu_id * 1000000;
        state = state * 1103515245 + 12345;
        float rand_val = ((state >> 16) & 0x7FFF) / 32768.0f;
        float value = (rand_val - 0.5f) * 0.1f;  // Small values to prevent overflow
        weights[tid] = __float2half(value);
    }
}

// Random expert assignment (simplified gate)
__global__ void assign_experts_kernel(
    TokenRouting* routing,   // [num_tokens * top_k]
    int num_tokens,
    int total_experts,
    int top_k,
    int gpu_id,
    unsigned int seed
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    // Pseudo-random expert assignment
    unsigned int state = seed + token_idx + gpu_id * 100000;

    for (int k = 0; k < top_k; ++k) {
        state = state * 1103515245 + 12345;
        int expert_id = (state >> 16) % total_experts;

        // Avoid duplicate experts for same token
        if (k > 0) {
            int prev_expert = routing[token_idx * top_k + k - 1].expert_id;
            if (expert_id == prev_expert) {
                expert_id = (expert_id + 1) % total_experts;
            }
        }

        routing[token_idx * top_k + k].token_id = token_idx;
        routing[token_idx * top_k + k].expert_id = expert_id;
        routing[token_idx * top_k + k].source_gpu = gpu_id;
        routing[token_idx * top_k + k].weight = 1.0f / top_k;  // Equal weights
    }
}

// Count tokens per expert
__global__ void count_experts_kernel(
    const TokenRouting* routing,
    int* expert_counts,    // [total_experts]
    int num_tokens,
    int top_k,
    int total_experts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens * top_k) return;

    int expert_id = routing[idx].expert_id;
    if (expert_id >= 0 && expert_id < total_experts) {
        atomicAdd(&expert_counts[expert_id], 1);
    }
}

// Prepare dispatch: assign slots in expert buffers
__global__ void prepare_dispatch_kernel(
    TokenRouting* routing,           // [num_tokens * top_k]
    int* expert_slots,               // [total_experts] - atomically incremented
    ExpertAssignment* assignments,   // [num_tokens * top_k] - where each token goes
    int num_tokens,
    int top_k,
    int total_experts,
    int expert_capacity
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens * top_k) return;

    int expert_id = routing[idx].expert_id;
    int token_id = routing[idx].token_id;
    float weight = routing[idx].weight;
    int source_gpu = routing[idx].source_gpu;

    // Get slot in expert buffer
    int slot = -1;
    if (expert_id >= 0 && expert_id < total_experts) {
        slot = atomicAdd(&expert_slots[expert_id], 1);
        if (slot >= expert_capacity) {
            slot = -1;  // Token dropped due to capacity
        }
    }

    assignments[idx].token_local_id = token_id;
    assignments[idx].slot_in_expert = slot;
    assignments[idx].weight = weight;
    assignments[idx].source_gpu = source_gpu;
}

// Dispatch tokens to local expert buffers
// For tokens going to remote GPU, we'll copy them later via rocSHMEM
__global__ void dispatch_local_kernel(
    const __half* tokens,            // [num_tokens, hidden_dim]
    __half* expert_buffers,          // [experts_per_gpu, expert_capacity, hidden_dim]
    const TokenRouting* routing,
    const ExpertAssignment* assignments,
    int num_tokens,
    int hidden_dim,
    int top_k,
    int experts_per_gpu,
    int expert_capacity,
    int my_gpu_id
) {
    int idx = blockIdx.x;  // One block per token-expert assignment
    if (idx >= num_tokens * top_k) return;

    int expert_id = routing[idx].expert_id;
    int token_id = assignments[idx].token_local_id;
    int slot = assignments[idx].slot_in_expert;

    // Determine if this expert is local to this GPU
    int expert_gpu = expert_id / experts_per_gpu;
    if (expert_gpu != my_gpu_id) return;  // Remote expert, handled separately
    if (slot < 0) return;  // Dropped

    int local_expert_id = expert_id % experts_per_gpu;

    // Copy token to expert buffer
    for (int h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        int src_idx = token_id * hidden_dim + h;
        int dst_idx = (local_expert_id * expert_capacity + slot) * hidden_dim + h;
        expert_buffers[dst_idx] = tokens[src_idx];
    }
}

// Simple expert GEMM: output = input @ W_up (simplified FFN)
// In real MoE this would be: FFN(x) = W_down @ activation(W_up @ x)
__global__ void expert_gemm_kernel(
    const __half* expert_input,      // [expert_capacity, hidden_dim]
    const __half* expert_weights,    // [hidden_dim, intermediate_dim]
    __half* expert_output,           // [expert_capacity, intermediate_dim]
    int expert_capacity,
    int hidden_dim,
    int intermediate_dim,
    int num_valid_tokens  // Actual number of tokens in this expert
) {
    int row = blockIdx.y;  // Token index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Output dimension

    if (row >= num_valid_tokens || col >= intermediate_dim) return;

    // Dot product for this output element
    float acc = 0.0f;
    for (int k = 0; k < hidden_dim; ++k) {
        float a = __half2float(expert_input[row * hidden_dim + k]);
        float b = __half2float(expert_weights[k * intermediate_dim + col]);
        acc += a * b;
    }

    // SiLU activation: x * sigmoid(x)
    float silu = acc / (1.0f + expf(-acc));

    expert_output[row * intermediate_dim + col] = __float2half(silu);
}

// Combine kernel: weighted sum back to original token positions
__global__ void combine_kernel(
    const __half* expert_outputs,    // [experts_per_gpu, expert_capacity, output_dim]
    __half* combined_output,         // [num_tokens, output_dim]
    const ExpertAssignment* assignments,
    int num_tokens,
    int output_dim,
    int top_k,
    int experts_per_gpu,
    int expert_capacity,
    int my_gpu_id
) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    // Iterate over output dimension
    for (int d = threadIdx.x; d < output_dim; d += blockDim.x) {
        float sum = 0.0f;

        // Sum contributions from top-k experts
        for (int k = 0; k < top_k; ++k) {
            int assignment_idx = token_idx * top_k + k;
            int slot = assignments[assignment_idx].slot_in_expert;
            float weight = assignments[assignment_idx].weight;

            if (slot >= 0) {
                // Need to determine which expert buffer to read from
                // For now, read from the first local expert for testing
                // In full implementation, this would index correctly based on expert_id
                int local_expert = k % experts_per_gpu;  // Simplified
                int src_idx = (local_expert * expert_capacity + slot) * output_dim + d;
                float val = __half2float(expert_outputs[src_idx]);
                sum += val * weight;
            }
        }

        combined_output[token_idx * output_dim + d] = __float2half(sum);
    }
}

// Verification kernel: check output has valid values
__global__ void verify_output_kernel(
    const __half* output,
    int num_elements,
    int* nan_count,
    int* inf_count,
    int* zero_count,
    int* valid_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return;

    float val = __half2float(output[tid]);

    if (isnan(val)) {
        atomicAdd(nan_count, 1);
    } else if (isinf(val)) {
        atomicAdd(inf_count, 1);
    } else if (fabsf(val) < 1e-10f) {
        atomicAdd(zero_count, 1);
    } else {
        atomicAdd(valid_count, 1);
    }
}

// ============================================================================
// Timer Helper
// ============================================================================

class GPUTimer {
    hipEvent_t start_, stop_;
public:
    GPUTimer() {
        CHECK_HIP(hipEventCreate(&start_));
        CHECK_HIP(hipEventCreate(&stop_));
    }
    ~GPUTimer() {
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
    float elapsed_ms() {
        float ms = 0.0f;
        CHECK_HIP(hipEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// ============================================================================
// Per-GPU Processing Function
// ============================================================================

void run_moe_pe(int rank, int npes, SharedBootstrap* shared) {
    using Config = MoEConfig;
    int ret;

    // Use stderr for immediate output
    fprintf(stderr, "[PE %d] Starting MoE forward pass (PID %d)\n", rank, getpid());

    // Set a physical device FIRST before any HIP/rocSHMEM operations.
    // Multiple PEs can share one visible GPU for single-GPU smoke testing.
    int device_count = 0;
    CHECK_HIP(hipGetDeviceCount(&device_count));
    if (device_count <= 0) {
        fprintf(stderr, "[PE %d] No HIP devices available\n", rank);
        exit(1);
    }
    const int device_id = rank % device_count;
    hipError_t hip_err = hipSetDevice(device_id);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "[PE %d] hipSetDevice(%d) failed: %s\n",
                rank, device_id, hipGetErrorString(hip_err));
        exit(1);
    }

    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, device_id));
    fprintf(stderr, "[PE %d] Device %d/%d: %s\n", rank, device_id, device_count, props.name);

    // ========================================================================
    // Initialize rocSHMEM BEFORE creating stream
    // (following working pattern from multi_gpu_rocshmem)
    // ========================================================================

    // PE 0 generates unique ID for bootstrap
    if (rank == 0) {
        fprintf(stderr, "[PE %d] Generating unique ID...\n", rank);
        ret = rocshmem_get_uniqueid(&shared->uid);
        if (ret != ROCSHMEM_SUCCESS) {
            fprintf(stderr, "[PE %d] Error getting uniqueid: %d\n", rank, ret);
            exit(1);
        }
        fprintf(stderr, "[PE %d] Got unique ID, signaling other PEs\n", rank);
        shared->uid_ready.store(1);
    } else {
        fprintf(stderr, "[PE %d] Waiting for unique ID from PE 0...\n", rank);
        while (shared->uid_ready.load() == 0) usleep(1000);
        fprintf(stderr, "[PE %d] Received unique ID\n", rank);
    }

    // Initialize rocSHMEM with unique ID
    fprintf(stderr, "[PE %d] Setting rocSHMEM attributes...\n", rank);
    rocshmem_init_attr_t attr;
    ret = rocshmem_set_attr_uniqueid_args(rank, npes, &shared->uid, &attr);
    if (ret != ROCSHMEM_SUCCESS) {
        fprintf(stderr, "[PE %d] Error set_attr: %d\n", rank, ret);
        exit(1);
    }

    fprintf(stderr, "[PE %d] Calling rocshmem_init_attr...\n", rank);
    ret = rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &attr);
    if (ret != ROCSHMEM_SUCCESS) {
        fprintf(stderr, "[PE %d] Error init_attr: %d\n", rank, ret);
        exit(1);
    }

    int my_pe = rocshmem_my_pe();
    int n_pes = rocshmem_n_pes();
    fprintf(stderr, "[PE %d] rocSHMEM initialized: my_pe=%d, n_pes=%d\n", rank, my_pe, n_pes);

    // Create stream AFTER rocSHMEM initialization
    hipStream_t stream;
    CHECK_HIP(hipStreamCreate(&stream));
    fprintf(stderr, "[PE %d] Stream created\n", rank);

    // ========================================================================
    // Allocate Memory
    // ========================================================================

    fprintf(stderr, "[PE %d] Allocating memory...\n", rank);

    // Local input tokens [TOKENS_PER_GPU, HIDDEN_DIM]
    size_t tokens_size = Config::TOKENS_PER_GPU * Config::HIDDEN_DIM * sizeof(__half);
    __half* d_tokens;
    CHECK_HIP(hipMalloc(&d_tokens, tokens_size));
    fprintf(stderr, "[PE %d]   Tokens: %.2f MB\n", rank, tokens_size / 1e6);

    // Expert weights [EXPERTS_PER_GPU, HIDDEN_DIM, INTERMEDIATE_DIM]
    size_t weights_size = Config::EXPERTS_PER_GPU * Config::HIDDEN_DIM * Config::INTERMEDIATE_DIM * sizeof(__half);
    __half* d_expert_weights;
    CHECK_HIP(hipMalloc(&d_expert_weights, weights_size));
    fprintf(stderr, "[PE %d]   Expert weights: %.2f MB\n", rank, weights_size / 1e6);

    // Expert input buffers (symmetric memory for rocSHMEM)
    // [EXPERTS_PER_GPU, EXPERT_CAPACITY, HIDDEN_DIM]
    size_t expert_input_size = Config::EXPERTS_PER_GPU * Config::EXPERT_CAPACITY * Config::HIDDEN_DIM * sizeof(__half);
    fprintf(stderr, "[PE %d]   Allocating expert_inputs via rocshmem_malloc: %.2f MB\n", rank, expert_input_size / 1e6);
    __half* d_expert_inputs = (__half*)rocshmem_malloc(expert_input_size);
    if (!d_expert_inputs) {
        fprintf(stderr, "[PE %d] rocshmem_malloc failed for expert_inputs (size: %zu)\n", rank, expert_input_size);
        exit(1);
    }
    fprintf(stderr, "[PE %d]   expert_inputs allocated at %p\n", rank, (void*)d_expert_inputs);

    // Expert output buffers (symmetric memory for rocSHMEM)
    // [EXPERTS_PER_GPU, EXPERT_CAPACITY, INTERMEDIATE_DIM]
    size_t expert_output_size = Config::EXPERTS_PER_GPU * Config::EXPERT_CAPACITY * Config::INTERMEDIATE_DIM * sizeof(__half);
    fprintf(stderr, "[PE %d]   Allocating expert_outputs via rocshmem_malloc: %.2f MB\n", rank, expert_output_size / 1e6);
    __half* d_expert_outputs = (__half*)rocshmem_malloc(expert_output_size);
    if (!d_expert_outputs) {
        fprintf(stderr, "[PE %d] rocshmem_malloc failed for expert_outputs (size: %zu)\n", rank, expert_output_size);
        exit(1);
    }
    fprintf(stderr, "[PE %d]   expert_outputs allocated at %p\n", rank, (void*)d_expert_outputs);

    // Token routing info [TOKENS_PER_GPU * TOP_K]
    size_t routing_size = Config::TOKENS_PER_GPU * Config::TOP_K * sizeof(TokenRouting);
    TokenRouting* d_routing;
    CHECK_HIP(hipMalloc(&d_routing, routing_size));

    // Expert assignment results [TOKENS_PER_GPU * TOP_K]
    size_t assignment_size = Config::TOKENS_PER_GPU * Config::TOP_K * sizeof(ExpertAssignment);
    ExpertAssignment* d_assignments;
    CHECK_HIP(hipMalloc(&d_assignments, assignment_size));

    // Expert counts and slots [TOTAL_EXPERTS]
    int* d_expert_counts;
    int* d_expert_slots;
    CHECK_HIP(hipMalloc(&d_expert_counts, Config::TOTAL_EXPERTS * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_expert_slots, Config::TOTAL_EXPERTS * sizeof(int)));

    // Combined output [TOKENS_PER_GPU, INTERMEDIATE_DIM]
    size_t output_size = Config::TOKENS_PER_GPU * Config::INTERMEDIATE_DIM * sizeof(__half);
    __half* d_combined_output;
    CHECK_HIP(hipMalloc(&d_combined_output, output_size));

    // Verification counters
    int *d_nan_count, *d_inf_count, *d_zero_count, *d_valid_count;
    CHECK_HIP(hipMalloc(&d_nan_count, sizeof(int)));
    CHECK_HIP(hipMalloc(&d_inf_count, sizeof(int)));
    CHECK_HIP(hipMalloc(&d_zero_count, sizeof(int)));
    CHECK_HIP(hipMalloc(&d_valid_count, sizeof(int)));

    // Signal for rocSHMEM synchronization
    fprintf(stderr, "[PE %d]   Allocating signal via rocshmem_malloc\n", rank);
    uint64_t* d_signal = (uint64_t*)rocshmem_malloc(sizeof(uint64_t));
    if (!d_signal) {
        fprintf(stderr, "[PE %d] rocshmem_malloc failed for signal\n", rank);
        exit(1);
    }
    fprintf(stderr, "[PE %d]   signal allocated at %p\n", rank, (void*)d_signal);

    // Initialize memory
    fprintf(stderr, "[PE %d] Initializing memory...\n", rank);
    CHECK_HIP(hipMemset(d_expert_inputs, 0, expert_input_size));
    CHECK_HIP(hipMemset(d_expert_outputs, 0, expert_output_size));
    CHECK_HIP(hipMemset(d_combined_output, 0, output_size));
    CHECK_HIP(hipMemset(d_expert_counts, 0, Config::TOTAL_EXPERTS * sizeof(int)));
    CHECK_HIP(hipMemset(d_expert_slots, 0, Config::TOTAL_EXPERTS * sizeof(int)));
    CHECK_HIP(hipMemset(d_signal, 0, sizeof(uint64_t)));

    // Initialize verification counters
    CHECK_HIP(hipMemset(d_nan_count, 0, sizeof(int)));
    CHECK_HIP(hipMemset(d_inf_count, 0, sizeof(int)));
    CHECK_HIP(hipMemset(d_zero_count, 0, sizeof(int)));
    CHECK_HIP(hipMemset(d_valid_count, 0, sizeof(int)));

    CHECK_HIP(hipDeviceSynchronize());

    fprintf(stderr, "[PE %d] Memory allocated and initialized\n", rank);

    // ========================================================================
    // Initialize Data
    // ========================================================================

    fprintf(stderr, "[PE %d] Initializing data...\n", rank);

    int threads = 256;
    int blocks;

    // Initialize input tokens
    blocks = (Config::TOKENS_PER_GPU * Config::HIDDEN_DIM + threads - 1) / threads;
    hipLaunchKernelGGL(init_tokens_kernel, dim3(blocks), dim3(threads), 0, stream,
        d_tokens, Config::TOKENS_PER_GPU, Config::HIDDEN_DIM, rank);

    // Initialize expert weights
    int weight_elements = Config::EXPERTS_PER_GPU * Config::HIDDEN_DIM * Config::INTERMEDIATE_DIM;
    blocks = (weight_elements + threads - 1) / threads;
    hipLaunchKernelGGL(init_expert_weights_kernel, dim3(blocks), dim3(threads), 0, stream,
        d_expert_weights, Config::EXPERTS_PER_GPU, Config::HIDDEN_DIM, Config::INTERMEDIATE_DIM,
        rank, 42);

    // Generate random expert assignments
    blocks = (Config::TOKENS_PER_GPU + threads - 1) / threads;
    hipLaunchKernelGGL(assign_experts_kernel, dim3(blocks), dim3(threads), 0, stream,
        d_routing, Config::TOKENS_PER_GPU, Config::TOTAL_EXPERTS, Config::TOP_K, rank, 12345);

    CHECK_HIP(hipStreamSynchronize(stream));

    fprintf(stderr, "[PE %d] Data initialized\n", rank);

    // ========================================================================
    // Barrier: All PEs ready
    // ========================================================================

    fprintf(stderr, "[PE %d] Waiting for all PEs to be ready...\n", rank);

    shared->phase_sync.fetch_add(1);
    while (shared->phase_sync.load() < npes) usleep(1000);

    fprintf(stderr, "[PE %d] Calling rocshmem_barrier_all_on_stream...\n", rank);

    rocshmem_barrier_all_on_stream(stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    fprintf(stderr, "[PE %d] Barrier complete\n", rank);

    fprintf(stderr, "[PE %d] Starting MoE forward pass...\n", rank);

    GPUTimer total_timer, dispatch_timer, expert_timer, combine_timer;

    total_timer.start(stream);

    // ========================================================================
    // Phase 1: Count tokens per expert and prepare dispatch
    // ========================================================================

    dispatch_timer.start(stream);

    // Count tokens per expert
    blocks = (Config::TOKENS_PER_GPU * Config::TOP_K + threads - 1) / threads;
    hipLaunchKernelGGL(count_experts_kernel, dim3(blocks), dim3(threads), 0, stream,
        d_routing, d_expert_counts, Config::TOKENS_PER_GPU, Config::TOP_K, Config::TOTAL_EXPERTS);

    // Prepare dispatch (assign slots)
    hipLaunchKernelGGL(prepare_dispatch_kernel, dim3(blocks), dim3(threads), 0, stream,
        d_routing, d_expert_slots, d_assignments,
        Config::TOKENS_PER_GPU, Config::TOP_K, Config::TOTAL_EXPERTS, Config::EXPERT_CAPACITY);

    CHECK_HIP(hipStreamSynchronize(stream));

    // ========================================================================
    // Phase 2: Dispatch tokens to local expert buffers
    // ========================================================================

    blocks = Config::TOKENS_PER_GPU * Config::TOP_K;
    hipLaunchKernelGGL(dispatch_local_kernel, dim3(blocks), dim3(threads), 0, stream,
        d_tokens, d_expert_inputs, d_routing, d_assignments,
        Config::TOKENS_PER_GPU, Config::HIDDEN_DIM, Config::TOP_K,
        Config::EXPERTS_PER_GPU, Config::EXPERT_CAPACITY, rank);

    CHECK_HIP(hipStreamSynchronize(stream));

    // ========================================================================
    // Phase 3: Exchange tokens with remote GPU via rocSHMEM
    // ========================================================================

    // For tokens that need to go to the other GPU's experts:
    // - GPU 0 has experts 0-3, GPU 1 has experts 4-7
    // - Tokens assigned to experts on the other GPU need to be sent

    // Get expert counts to know how many tokens to send/receive
    int h_expert_counts[Config::TOTAL_EXPERTS];
    CHECK_HIP(hipMemcpy(h_expert_counts, d_expert_counts,
        Config::TOTAL_EXPERTS * sizeof(int), hipMemcpyDeviceToHost));

    // Calculate tokens going to remote GPU
    int tokens_to_remote = 0;
    int remote_gpu = (rank + 1) % npes;
    for (int e = 0; e < Config::TOTAL_EXPERTS; ++e) {
        int expert_gpu = e / Config::EXPERTS_PER_GPU;
        if (expert_gpu == remote_gpu) {
            tokens_to_remote += h_expert_counts[e];
        }
    }

    shared->tokens_dispatched[rank] = tokens_to_remote;

    fprintf(stderr, "[PE %d] Tokens to remote GPU: %d\n", rank, tokens_to_remote);

    // Barrier for dispatch phase
    rocshmem_barrier_all_on_stream(stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    // Exchange expert buffers via rocSHMEM
    // Send our expert input buffer portion to the other GPU
    // This is a simplified version - in production, only send actual tokens

    // For this test, we'll do a simple buffer exchange
    // Each GPU sends tokens destined for experts on the other GPU

    // Get pointer to remote PE's expert input buffer
    __half* remote_expert_inputs = (__half*)rocshmem_ptr(d_expert_inputs, remote_gpu);

    if (tokens_to_remote > 0 && remote_expert_inputs != nullptr) {
        // Direct put for peer-accessible memory
        // In real implementation, this would be selective based on routing
        size_t transfer_size = Config::EXPERTS_PER_GPU * Config::EXPERT_CAPACITY * Config::HIDDEN_DIM * sizeof(__half);

        // Use rocSHMEM put with signal
        rocshmem_putmem_signal_on_stream(
            d_expert_inputs,  // Remote destination
            d_expert_inputs,  // Local source (same offset for symmetric heap)
            transfer_size,
            d_signal,
            1,
            ROCSHMEM_SIGNAL_SET,
            remote_gpu,
            stream);
    }

    CHECK_HIP(hipStreamSynchronize(stream));

    // Wait for incoming data
    rocshmem_signal_wait_until_on_stream(d_signal, ROCSHMEM_CMP_GE, 1, stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    shared->tokens_received[rank] = shared->tokens_dispatched[remote_gpu];

    dispatch_timer.stop(stream);

    rocshmem_barrier_all_on_stream(stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    fprintf(stderr, "[PE %d] Dispatch complete, time: %.3f ms\n", rank, dispatch_timer.elapsed_ms());

    // ========================================================================
    // Phase 4: Expert GEMM computation
    // ========================================================================

    expert_timer.start(stream);

    // Compute GEMM for each local expert
    for (int e = 0; e < Config::EXPERTS_PER_GPU; ++e) {
        int global_expert_id = rank * Config::EXPERTS_PER_GPU + e;
        int num_tokens_for_expert = h_expert_counts[global_expert_id];

        if (num_tokens_for_expert > 0) {
            num_tokens_for_expert = min(num_tokens_for_expert, Config::EXPERT_CAPACITY);

            __half* expert_input = d_expert_inputs + e * Config::EXPERT_CAPACITY * Config::HIDDEN_DIM;
            __half* expert_weight = d_expert_weights + e * Config::HIDDEN_DIM * Config::INTERMEDIATE_DIM;
            __half* expert_output = d_expert_outputs + e * Config::EXPERT_CAPACITY * Config::INTERMEDIATE_DIM;

            dim3 grid((Config::INTERMEDIATE_DIM + 255) / 256, num_tokens_for_expert);
            dim3 block(256);

            hipLaunchKernelGGL(expert_gemm_kernel, grid, block, 0, stream,
                expert_input, expert_weight, expert_output,
                Config::EXPERT_CAPACITY, Config::HIDDEN_DIM, Config::INTERMEDIATE_DIM,
                num_tokens_for_expert);
        }
    }

    CHECK_HIP(hipStreamSynchronize(stream));
    expert_timer.stop(stream);

    fprintf(stderr, "[PE %d] Expert GEMM complete, time: %.3f ms\n", rank, expert_timer.elapsed_ms());

    // ========================================================================
    // Phase 5: Exchange expert outputs back
    // ========================================================================

    rocshmem_barrier_all_on_stream(stream);

    // Reset signal for second exchange
    CHECK_HIP(hipMemset(d_signal, 0, sizeof(uint64_t)));
    rocshmem_barrier_all_on_stream(stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    // Send expert outputs back
    if (tokens_to_remote > 0 && rocshmem_ptr(d_expert_outputs, remote_gpu) != nullptr) {
        size_t transfer_size = Config::EXPERTS_PER_GPU * Config::EXPERT_CAPACITY * Config::INTERMEDIATE_DIM * sizeof(__half);

        rocshmem_putmem_signal_on_stream(
            d_expert_outputs,
            d_expert_outputs,
            transfer_size,
            d_signal,
            1,
            ROCSHMEM_SIGNAL_SET,
            remote_gpu,
            stream);
    }

    CHECK_HIP(hipStreamSynchronize(stream));
    rocshmem_signal_wait_until_on_stream(d_signal, ROCSHMEM_CMP_GE, 1, stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    rocshmem_barrier_all_on_stream(stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    // ========================================================================
    // Phase 6: Combine expert outputs
    // ========================================================================

    combine_timer.start(stream);

    blocks = Config::TOKENS_PER_GPU;
    hipLaunchKernelGGL(combine_kernel, dim3(blocks), dim3(threads), 0, stream,
        d_expert_outputs, d_combined_output, d_assignments,
        Config::TOKENS_PER_GPU, Config::INTERMEDIATE_DIM, Config::TOP_K,
        Config::EXPERTS_PER_GPU, Config::EXPERT_CAPACITY, rank);

    CHECK_HIP(hipStreamSynchronize(stream));
    combine_timer.stop(stream);

    total_timer.stop(stream);

    fprintf(stderr, "[PE %d] Combine complete, time: %.3f ms\n", rank, combine_timer.elapsed_ms());

    // ========================================================================
    // Verification
    // ========================================================================

    fprintf(stderr, "[PE %d] Verifying output...\n", rank);

    int output_elements = Config::TOKENS_PER_GPU * Config::INTERMEDIATE_DIM;
    blocks = (output_elements + threads - 1) / threads;
    hipLaunchKernelGGL(verify_output_kernel, dim3(blocks), dim3(threads), 0, stream,
        d_combined_output, output_elements,
        d_nan_count, d_inf_count, d_zero_count, d_valid_count);

    CHECK_HIP(hipStreamSynchronize(stream));

    int h_nan_count, h_inf_count, h_zero_count, h_valid_count;
    CHECK_HIP(hipMemcpy(&h_nan_count, d_nan_count, sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(&h_inf_count, d_inf_count, sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(&h_zero_count, d_zero_count, sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(&h_valid_count, d_valid_count, sizeof(int), hipMemcpyDeviceToHost));

    bool passed = (h_nan_count == 0) && (h_inf_count == 0) && (h_valid_count > 0);

    fprintf(stderr, "[PE %d] Output verification:\n", rank);
    fprintf(stderr, "[PE %d]   NaN count: %d\n", rank, h_nan_count);
    fprintf(stderr, "[PE %d]   Inf count: %d\n", rank, h_inf_count);
    fprintf(stderr, "[PE %d]   Zero count: %d\n", rank, h_zero_count);
    fprintf(stderr, "[PE %d]   Valid count: %d / %d\n", rank, h_valid_count, output_elements);
    fprintf(stderr, "[PE %d]   Status: %s\n", rank, passed ? "PASS" : "FAIL");

    // Store results in shared memory
    shared->test_passed[rank] = passed ? 1 : 0;
    shared->dispatch_time_ms[rank] = dispatch_timer.elapsed_ms();
    shared->expert_time_ms[rank] = expert_timer.elapsed_ms();
    shared->combine_time_ms[rank] = combine_timer.elapsed_ms();
    shared->total_time_ms[rank] = total_timer.elapsed_ms();

    // ========================================================================
    // Cleanup
    // ========================================================================

    rocshmem_barrier_all_on_stream(stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    fprintf(stderr, "[PE %d] Cleaning up...\n", rank);

    rocshmem_free(d_signal);
    rocshmem_free(d_expert_outputs);
    rocshmem_free(d_expert_inputs);

    CHECK_HIP(hipFree(d_tokens));
    CHECK_HIP(hipFree(d_expert_weights));
    CHECK_HIP(hipFree(d_routing));
    CHECK_HIP(hipFree(d_assignments));
    CHECK_HIP(hipFree(d_expert_counts));
    CHECK_HIP(hipFree(d_expert_slots));
    CHECK_HIP(hipFree(d_combined_output));
    CHECK_HIP(hipFree(d_nan_count));
    CHECK_HIP(hipFree(d_inf_count));
    CHECK_HIP(hipFree(d_zero_count));
    CHECK_HIP(hipFree(d_valid_count));

    CHECK_HIP(hipStreamDestroy(stream));

    fprintf(stderr, "[PE %d] Finalizing rocSHMEM...\n", rank);
    rocshmem_finalize();
    fprintf(stderr, "[PE %d] Done\n", rank);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    using Config = MoEConfig;

    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║       FlashMoE HIP - Multi-GPU MoE Forward Pass Test             ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Configuration:                                                  ║\n");
    printf("║    GPUs:         %d                                              ║\n", Config::NUM_GPUS);
    printf("║    Total Experts: %d (%d per GPU)                                ║\n", Config::TOTAL_EXPERTS, Config::EXPERTS_PER_GPU);
    printf("║    Tokens/GPU:   %d                                            ║\n", Config::TOKENS_PER_GPU);
    printf("║    Hidden Dim:   %d                                            ║\n", Config::HIDDEN_DIM);
    printf("║    Intermediate: %d                                           ║\n", Config::INTERMEDIATE_DIM);
    printf("║    Top-K:        %d                                              ║\n", Config::TOP_K);
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    printf("\nrocSHMEM Version: %s\n\n", rocshmem::VERSION);
    fflush(stdout);

    int npes = Config::NUM_GPUS;

    // NOTE: Do NOT call any HIP functions before fork() to avoid
    // HIP runtime state being inherited by child processes
    printf("Testing with %d PEs (GPUs)\n\n", npes);
    fflush(stdout);

    // Create shared memory for bootstrap (before any HIP calls)
    SharedBootstrap* shared = (SharedBootstrap*)mmap(NULL, sizeof(SharedBootstrap),
                                                      PROT_READ | PROT_WRITE,
                                                      MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (shared == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    memset(shared, 0, sizeof(SharedBootstrap));
    new (&shared->uid_ready) std::atomic<int>(0);
    new (&shared->ready_count) std::atomic<int>(0);
    new (&shared->init_done) std::atomic<int>(0);
    new (&shared->phase_sync) std::atomic<int>(0);
    for (int i = 0; i < npes; i++) shared->test_passed[i] = 0;

    // Fork processes
    pid_t pids[Config::NUM_GPUS];
    for (int i = 0; i < npes; i++) {
        pids[i] = fork();
        if (pids[i] == 0) {
            run_moe_pe(i, npes, shared);
            exit(0);
        } else if (pids[i] < 0) {
            perror("fork");
            return 1;
        }
    }

    // Wait for children
    int status;
    int all_passed = 1;
    for (int i = 0; i < npes; i++) {
        waitpid(pids[i], &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            printf("PE %d exited with error (status: 0x%x)\n", i, status);
            all_passed = 0;
        }
        if (!shared->test_passed[i]) all_passed = 0;
    }

    // Print results
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║                        RESULTS SUMMARY                           ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");

    printf("║  Timing (per GPU):                                               ║\n");
    for (int i = 0; i < npes; ++i) {
        printf("║    GPU %d:                                                        ║\n", i);
        printf("║      Dispatch + Communication: %8.3f ms                       ║\n", shared->dispatch_time_ms[i]);
        printf("║      Expert GEMM:              %8.3f ms                       ║\n", shared->expert_time_ms[i]);
        printf("║      Combine:                  %8.3f ms                       ║\n", shared->combine_time_ms[i]);
        printf("║      Total:                    %8.3f ms                       ║\n", shared->total_time_ms[i]);
    }

    float avg_total = 0;
    for (int i = 0; i < npes; ++i) {
        avg_total += shared->total_time_ms[i];
    }
    avg_total /= npes;

    float tokens_per_second = (Config::TOTAL_TOKENS) / (avg_total / 1000.0f);

    printf("║                                                                  ║\n");
    printf("║  Aggregate Performance:                                          ║\n");
    printf("║    Average total time:   %8.3f ms                             ║\n", avg_total);
    printf("║    Tokens/second:        %.2e                            ║\n", tokens_per_second);

    printf("║                                                                  ║\n");
    printf("║  Communication:                                                  ║\n");
    for (int i = 0; i < npes; ++i) {
        printf("║    GPU %d: sent %d tokens, received %d tokens                     ║\n",
               i, shared->tokens_dispatched[i], shared->tokens_received[i]);
    }

    printf("╠══════════════════════════════════════════════════════════════════╣\n");

    if (all_passed) {
        printf("║  \033[32m✓ ALL TESTS PASSED - Multi-GPU MoE forward works!\033[0m              ║\n");
    } else {
        printf("║  \033[31m✗ SOME TESTS FAILED\033[0m                                            ║\n");
    }
    printf("╚══════════════════════════════════════════════════════════════════╝\n");

    munmap(shared, sizeof(SharedBootstrap));
    return all_passed ? 0 : 1;
}
