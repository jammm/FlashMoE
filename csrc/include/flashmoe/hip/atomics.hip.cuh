/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 *
 * HIP Port of CUDA Atomics and Barriers
 * Target: AMD MI450 (gfx1250)
 *
 * Key differences from CUDA:
 * - MI450/gfx1250 wavefronts are 32-wide
 * - Uses __hip_atomic_* builtins with explicit memory scopes
 * - Grid barriers need careful handling with multiple device partitions
 * - Memory scope mappings:
 *   cuda::thread_scope_thread  -> __HIP_MEMORY_SCOPE_SINGLETHREAD
 *   cuda::thread_scope_block   -> __HIP_MEMORY_SCOPE_WORKGROUP
 *   cuda::thread_scope_device  -> __HIP_MEMORY_SCOPE_AGENT
 *   cuda::thread_scope_system  -> __HIP_MEMORY_SCOPE_SYSTEM
 */

#ifndef FLASHMOE_HIP_ATOMICS_CUH
#define FLASHMOE_HIP_ATOMICS_CUH

#include "flashmoe/hip/cuda_compat.hpp"
#include "flashmoe/hip/constants.hpp"

#include <hip/hip_runtime.h>
#include <concepts>
#include <type_traits>

namespace flashmoe {

// ============================================================================
// Type Aliases
// ============================================================================

using ull_t = unsigned long long int;

// ============================================================================
// Atomic Type Concepts
// ============================================================================

/**
 * AtomicType - Types that support standard atomic operations
 * HIP atomic builtins support int, unsigned int, and unsigned long long
 */
template<typename B>
concept AtomicType = std::same_as<B, int> ||
                     std::same_as<B, unsigned int> ||
                     std::same_as<B, ull_t>;

/**
 * AtomicCASType - Types that support Compare-And-Swap operations
 * Additionally includes unsigned short int (16-bit CAS)
 */
template<typename B>
concept AtomicCASType = std::same_as<B, int> ||
                        std::same_as<B, unsigned int> ||
                        std::same_as<B, ull_t> ||
                        std::same_as<B, unsigned short int>;

// ============================================================================
// Atomic Scope Enumeration and Concept
// ============================================================================

/**
 * AtomicScope concept - Validates that the scope is a valid HIP memory scope
 * Maps CUDA thread scopes to HIP memory scopes:
 *   thread_scope_thread  -> __HIP_MEMORY_SCOPE_SINGLETHREAD
 *   thread_scope_block   -> __HIP_MEMORY_SCOPE_WORKGROUP
 *   thread_scope_device  -> __HIP_MEMORY_SCOPE_AGENT
 *   thread_scope_system  -> __HIP_MEMORY_SCOPE_SYSTEM
 */
template<cuda::std::thread_scope scope>
concept AtomicScope = scope == cuda::thread_scope_thread ||
                      scope == cuda::thread_scope_block ||
                      scope == cuda::thread_scope_device ||
                      scope == cuda::thread_scope_system;

// ============================================================================
// Memory Scope Conversion Helper
// ============================================================================

namespace detail {

/**
 * Convert cuda::std::thread_scope to HIP memory scope constant
 * Compile-time conversion for template metaprogramming
 */
template<cuda::std::thread_scope Scope>
__device__ __forceinline__
constexpr int get_hip_memory_scope() {
    if constexpr (Scope == cuda::thread_scope_thread) {
        return __HIP_MEMORY_SCOPE_SINGLETHREAD;
    } else if constexpr (Scope == cuda::thread_scope_block) {
        return __HIP_MEMORY_SCOPE_WORKGROUP;
    } else if constexpr (Scope == cuda::thread_scope_device) {
        return __HIP_MEMORY_SCOPE_AGENT;
    } else if constexpr (Scope == cuda::thread_scope_system) {
        return __HIP_MEMORY_SCOPE_SYSTEM;
    } else {
        return __HIP_MEMORY_SCOPE_AGENT; // Default to device scope
    }
}

/**
 * Issue appropriate memory fence based on scope
 */
template<cuda::std::thread_scope Scope>
__device__ __forceinline__
void memory_fence() {
    if constexpr (Scope == cuda::thread_scope_thread) {
        // No fence needed for thread scope
    } else if constexpr (Scope == cuda::thread_scope_block) {
        __threadfence_block();
    } else if constexpr (Scope == cuda::thread_scope_device) {
        __threadfence();
    } else if constexpr (Scope == cuda::thread_scope_system) {
        __threadfence_system();
    }
}

} // namespace detail

// ============================================================================
// Atomic Test-And-Set (TAS)
// ============================================================================

/**
 * atomicTAS - Atomic Test-And-Set operation
 *
 * Uses __hip_atomic_exchange for atomic swap, which is more efficient than CAS
 * for test-and-set patterns on AMD GPUs.
 *
 * Template Parameters:
 *   scope - Memory scope (thread, block, device, or system)
 *   T     - Type of the atomic variable
 *
 * Returns: Previous value at addr (0 if unset, 1 if already set)
 *
 * Note: 16-bit TAS is only supported with device scope on AMD
 */
template<cuda::std::thread_scope scope = cuda::thread_scope_device, typename T>
requires AtomicCASType<T> && AtomicScope<scope> &&
    (!std::is_same_v<T, unsigned short int> || scope == cuda::thread_scope_device)
__device__ __forceinline__
T atomicTAS(T* __restrict__ const& addr) {
    constexpr T zero = static_cast<T>(0);
    constexpr T one = static_cast<T>(1);

    if constexpr (scope == cuda::thread_scope_thread) {
        // Thread-local: no atomics needed, simple exchange
        T old = *addr;
        *addr = one;
        return old;
    }
    else if constexpr (scope == cuda::thread_scope_block) {
        // Workgroup scope
        if constexpr (sizeof(T) == 2) {
            // 16-bit: use CAS since exchange may not be directly supported
            T expected = zero;
            __hip_atomic_compare_exchange_strong(
                addr, &expected, one,
                __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                __HIP_MEMORY_SCOPE_WORKGROUP
            );
            return expected;
        } else {
            return __hip_atomic_exchange(addr, one, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        }
    }
    else if constexpr (scope == cuda::thread_scope_system) {
        // System scope
        if constexpr (sizeof(T) == 2) {
            T expected = zero;
            __hip_atomic_compare_exchange_strong(
                addr, &expected, one,
                __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                __HIP_MEMORY_SCOPE_SYSTEM
            );
            return expected;
        } else {
            return __hip_atomic_exchange(addr, one, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
    else {
        // Device scope (default)
        if constexpr (sizeof(T) == 2) {
            // Use regular atomicCAS for 16-bit (device scope)
            return atomicCAS(addr, zero, one);
        } else {
            return __hip_atomic_exchange(addr, one, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

// ============================================================================
// Initialization State for Guarded Atomics
// ============================================================================

/**
 * InitState - State machine for guarded atomic initialization
 *
 * Used by guardedAtomicAdd to ensure proper initialization of accumulators
 * before concurrent atomic additions.
 *
 * State transitions: stale -> initializing -> initialized -> (stays initialized)
 *                    When last participant arrives: -> stale (reset for next use)
 */
enum InitState : int {
    stale = -1,        // Needs initialization (0xFF bytes via cudaMemset)
    initializing = 0,  // Currently being initialized
    initialized = 1    // Ready for use
};

// Set every byte to 0xFF → each int becomes 0xFFFFFFFF → -1 (stale)
// Needed for hipMemset to initialize state
constexpr auto STALE_AS_BYTE = 0xFF;

static_assert(std::is_same_v<std::underlying_type_t<InitState>, int>);
static_assert(initialized == 1 && stale != initializing && initialized != initializing);

// ============================================================================
// Guarded Atomic Add
// ============================================================================

/**
 * guardedAtomicAdd - In-kernel initialization of global accumulators
 *
 * This function enables safe initialization of accumulator variables from within
 * a kernel, without requiring a separate initialization kernel. The first thread
 * to arrive initializes the value; subsequent threads atomically add to it.
 *
 * Parameters:
 *   guard        - Pointer to guard variable (tracks initialization state)
 *   vals         - Pointer to accumulator value
 *   val          - Value to add
 *   participants - Total number of threads that will participate
 *
 * Returns: Previous value at vals (after initialization)
 *
 * Algorithm:
 *   1. First thread: CAS guard from stale to initializing, set vals, mark initialized
 *   2. Other threads: Wait for initialized state, then atomicAdd
 *   3. Last thread: Reset guard to stale for next batch
 */
__device__ __forceinline__
auto guardedAtomicAdd(int* __restrict__ const& guard,
                      int* __restrict__ const& vals,
                      const int& val,
                      const int& participants) {
    if (participants == 1) {
        *vals = val;
        return 0;
    }
    // Use device-scope atomic_ref for the guard
    const cuda::atomic_ref<int, cuda::thread_scope_device> g{*guard};

    int expected = stale;
    if (g.compare_exchange_strong(expected, initializing, cuda::memory_order_acquire)) {
        // I am the first thread - initialize with my value
        *vals = val;
        g.store(initialized, cuda::memory_order_release);
        return 0;
    }

    // Wait for initialization to complete
    while (expected == initializing) {
        // Brief pause to reduce memory contention
        // AMD-specific: use s_sleep for efficient polling
        __builtin_amdgcn_s_sleep(1);
        expected = g.load(cuda::memory_order_acquire);
    }

    // Check if I'm the last participant
    if (g.fetch_add(1, cuda::memory_order_relaxed) + 1 == participants) {
        // Everyone has arrived at this point, but I am the last.
        // Reset guard for next use
        g.store(stale, cuda::memory_order_relaxed);
    }

    // Perform the atomic add
    return atomicAdd(vals, val);
}

// ============================================================================
// Scope Extraction for Barriers
// ============================================================================

/**
 * SentinelScope - Invalid scope sentinel for type detection
 */
enum class SentinelScope {
    invalid
};

/**
 * ScopeOf - Extract scope from cuda::barrier type
 *
 * Used to verify barrier types at compile time. Returns SentinelScope::invalid
 * for non-barrier types, or the actual thread_scope for barrier types.
 */
template<class T>
struct ScopeOf {
    using Scope = std::integral_constant<SentinelScope, SentinelScope::invalid>;
};

template<cuda::std::thread_scope _Sco, class... Args>
struct ScopeOf<cuda::barrier<_Sco, Args...>> {
    using Scope = std::integral_constant<cuda::std::thread_scope, _Sco>;
};

// ============================================================================
// Grid Barrier Implementation for AMD GPUs
// ============================================================================

/**
 * GridBarrierState - Shared state for grid-wide barriers
 *
 * For AMD MI450 with multiple device partitions (chiplets), grid barriers require careful
 * implementation to ensure all CUs across all device partitions synchronize properly.
 *
 * This state should be allocated in global memory and passed to gridBarrier.
 */
struct GridBarrierState {
    unsigned int count;     // Arrival counter
    unsigned int phase;     // Phase counter (alternating to prevent ABA issues)
    unsigned int expected;  // Expected number of thread blocks
};

namespace detail {

/**
 * gridBarrierImpl - Low-level grid barrier using atomic counter and polling
 *
 * Implementation notes for AMD MI450:
 * - Uses atomic counter with device-scope visibility
 * - Wavefront-leader performs the atomic to reduce contention
 * - All threads poll with s_sleep to reduce power/contention
 * - Memory fences ensure cross-device partitions visibility
 *
 * Parameters:
 *   counter  - Pointer to arrival counter in global memory
 *   phase    - Pointer to phase counter in global memory
 *   expected - Number of thread blocks participating
 */
__device__ __forceinline__
void gridBarrierAtomicImpl(unsigned int* counter,
                           unsigned int* phase,
                           unsigned int expected) {
    // Synchronize within the block first
    __syncthreads();

    // Only one thread per block participates in global synchronization
    if (threadIdx.x == 0) {
        const unsigned int old_phase = __hip_atomic_load(
            phase, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT
        );

        // Arrive: increment counter
        unsigned int arrived = __hip_atomic_fetch_add(
            counter, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT
        );

        if (arrived + 1 == expected) {
            // I am the last block to arrive
            // Reset counter for next barrier phase
            __hip_atomic_store(counter, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

            // Ensure counter reset is visible before phase advance
            __threadfence();

            // Advance phase to release all waiting blocks
            __hip_atomic_fetch_add(phase, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        } else {
            // Wait for phase to change
            unsigned int current_phase;
            do {
                // AMD-specific: use s_sleep for efficient polling
                // This is crucial for power efficiency and reduces cache thrashing
                __builtin_amdgcn_s_sleep(2);
                current_phase = __hip_atomic_load(
                    phase, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT
                );
            } while (current_phase == old_phase);
        }
    }

    // Synchronize within the block to ensure all threads see the result
    __syncthreads();
}

/**
 * gridBarrierWavefrontImpl - Grid barrier with wavefront-level participation
 *
 * Optimized version where the first wavefront of each block participates,
 * reducing contention on the atomic counter by using wavefront reduction.
 *
 * This reduces atomic contention by limiting participation to the first wave
 * of each block.
 */
__device__ __forceinline__
void gridBarrierWavefrontImpl(unsigned int* counter,
                              unsigned int* phase,
                              unsigned int expected) {
    __syncthreads();

    // Only first thread in first wavefront of block participates
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int wave_id = threadIdx.x / WARP_SIZE;

    if (wave_id == 0 && lane_id == 0) {
        const unsigned int old_phase = __hip_atomic_load(
            phase, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT
        );

        unsigned int arrived = __hip_atomic_fetch_add(
            counter, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT
        );

        if (arrived + 1 == expected) {
            __hip_atomic_store(counter, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __threadfence();
            __hip_atomic_fetch_add(phase, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        } else {
            unsigned int current_phase;
            int backoff_count = 0;
            do {
                // Use fixed sleep value (must be compile-time constant)
                // Simulate exponential backoff by varying iterations
                __builtin_amdgcn_s_sleep(4);
                if (backoff_count < 8) {
                    backoff_count++;
                    // Additional delay for larger backoff
                    if (backoff_count > 4) {
                        __builtin_amdgcn_s_sleep(4);
                    }
                }
                current_phase = __hip_atomic_load(
                    phase, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT
                );
            } while (current_phase == old_phase);
        }
    }

    __syncthreads();
}

} // namespace detail

/**
 * gridBarrier - Grid-wide synchronization barrier
 *
 * Synchronizes all thread blocks in the grid. On AMD MI450, this requires
 * careful handling due to the multi-device partitions architecture.
 *
 * Template Parameters:
 *   CudaBarrier - Must be cuda::barrier with device scope
 *
 * Note: The CUDA version uses cuda::barrier::arrive_and_wait(). On HIP, we
 * implement this using our custom barrier type from cuda_compat.hpp.
 */
template<typename CudaBarrier>
__device__ __forceinline__
void gridBarrier(CudaBarrier* const& db) {
    static_assert(
        ScopeOf<std::remove_cvref_t<CudaBarrier>>::Scope::value == cuda::thread_scope_device,
        "gridBarrier expects device-scope barrier"
    );

    // Block-level sync first
    __syncthreads();

    // Only thread 0 participates in the device-wide barrier
    if (!threadIdx.x) {
        db->arrive_and_wait();
    }

    // Block-level sync to propagate the barrier completion
    __syncthreads();
}

/**
 * gridBarrierAtomic - Grid barrier using atomic state
 *
 * Alternative implementation that uses explicit atomic state instead of
 * cuda::barrier. This is useful when you need more control over the barrier
 * behavior or when debugging synchronization issues.
 *
 * Parameters:
 *   state - Pointer to GridBarrierState in global memory
 */
__device__ __forceinline__
void gridBarrierAtomic(GridBarrierState* state) {
    detail::gridBarrierAtomicImpl(&state->count, &state->phase, state->expected);
}

/**
 * gridBarrierAtomicRaw - Grid barrier using raw pointers
 *
 * Lowest-level grid barrier interface. Use when you manage the barrier
 * state manually or need maximum flexibility.
 *
 * Parameters:
 *   counter  - Pointer to arrival counter (must be 0 initially)
 *   phase    - Pointer to phase counter (must be 0 initially)
 *   expected - Number of thread blocks
 */
__device__ __forceinline__
void gridBarrierAtomicRaw(unsigned int* counter,
                          unsigned int* phase,
                          unsigned int expected) {
    detail::gridBarrierAtomicImpl(counter, phase, expected);
}

// ============================================================================
// Additional Atomic Utilities
// ============================================================================

/**
 * atomicMaxFloat - Atomic maximum for float values
 *
 * HIP doesn't have native atomicMax for floats, so we implement using CAS loop.
 * Uses bit-casting to handle floating-point comparison properly.
 */
__device__ __forceinline__
float atomicMaxFloat(float* addr, float val) {
    unsigned int* addr_as_uint = reinterpret_cast<unsigned int*>(addr);
    unsigned int old = *addr_as_uint;
    unsigned int assumed;

    do {
        assumed = old;
        float old_val = __uint_as_float(assumed);
        if (old_val >= val) {
            return old_val;
        }
        old = atomicCAS(addr_as_uint, assumed, __float_as_uint(val));
    } while (assumed != old);

    return __uint_as_float(old);
}

/**
 * atomicMinFloat - Atomic minimum for float values
 */
__device__ __forceinline__
float atomicMinFloat(float* addr, float val) {
    unsigned int* addr_as_uint = reinterpret_cast<unsigned int*>(addr);
    unsigned int old = *addr_as_uint;
    unsigned int assumed;

    do {
        assumed = old;
        float old_val = __uint_as_float(assumed);
        if (old_val <= val) {
            return old_val;
        }
        old = atomicCAS(addr_as_uint, assumed, __float_as_uint(val));
    } while (assumed != old);

    return __uint_as_float(old);
}

/**
 * atomicAddDouble - Atomic add for double values
 *
 * Uses CAS loop for GPUs that don't have native double atomicAdd.
 * Note: AMD MI450 supports native double atomicAdd, but this is provided
 * for compatibility with older architectures.
 */
__device__ __forceinline__
double atomicAddDouble(double* addr, double val) {
    unsigned long long int* addr_as_ull = reinterpret_cast<unsigned long long int*>(addr);
    unsigned long long int old = *addr_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed,
                        __double_as_longlong(__longlong_as_double(assumed) + val));
    } while (assumed != old);

    return __longlong_as_double(old);
}

// ============================================================================
// Cooperative Groups Style Barrier (Limited Support)
// ============================================================================

/**
 * blockSync - Block-level synchronization
 *
 * Wrapper around __syncthreads() for cooperative groups style code.
 */
__device__ __forceinline__
void blockSync() {
    __syncthreads();
}

/**
 * warpSync - Wavefront-level synchronization
 *
 * On AMD, wavefronts execute in lockstep, so this is mostly a memory barrier.
 * Use __builtin_amdgcn_wave_barrier() for explicit synchronization.
 */
__device__ __forceinline__
void warpSync() {
    __builtin_amdgcn_wave_barrier();
}

/**
 * deviceFence - Device-wide memory fence
 *
 * Ensures all previous memory operations are visible to all threads on the device.
 * Critical for cross-device partitions visibility on MI450.
 */
__device__ __forceinline__
void deviceFence() {
    __threadfence();
}

/**
 * systemFence - System-wide memory fence
 *
 * Ensures all previous memory operations are visible to the host and other GPUs.
 */
__device__ __forceinline__
void systemFence() {
    __threadfence_system();
}

} // namespace flashmoe

#endif // FLASHMOE_HIP_ATOMICS_CUH
