/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 *
 * CUDA-to-HIP Compatibility Layer
 * Provides mappings from CUDA primitives to HIP equivalents for AMD GPU support.
 */

#ifndef FLASHMOE_HIP_CUDA_COMPAT_HPP
#define FLASHMOE_HIP_CUDA_COMPAT_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#include <cstdint>
#include <type_traits>
#include <bit>        // C++20 std::bit_cast
#include <cstddef>    // std::byte
#include <atomic>
#include <concepts>
#include <tuple>      // std::ignore

// ============================================================================
// Type Mappings
// ============================================================================

// __half is already defined in HIP via hip_fp16.h

// Map NVIDIA bfloat16 to HIP bfloat16
using __nv_bfloat16 = hip_bfloat16;

// Packed bfloat16 type for vectorized operations
// Note: HIP may use different naming conventions across versions
#if defined(__hip_bfloat162)
using __nv_bfloat162 = __hip_bfloat162;
#else
// Define a compatible packed bfloat16 type if not provided by HIP
struct alignas(4) hip_bfloat162 {
    hip_bfloat16 x, y;
    __host__ __device__ hip_bfloat162() : x{}, y{} {}
    __host__ __device__ hip_bfloat162(hip_bfloat16 a, hip_bfloat16 b) : x(a), y(b) {}
};
using __nv_bfloat162 = hip_bfloat162;
#endif

// TF32 does not exist on AMD GPUs; use FP32 as fallback.
using tfloat32_t = float;

namespace cublasdx {
    // Provide tfloat32_t in cublasdx namespace for compatibility
    using tfloat32_t = ::tfloat32_t;
}

// ============================================================================
// Warp Shuffle Intrinsics
// ============================================================================

// HIP shuffle intrinsics don't take the mask parameter
// The full warp always participates in HIP

template<typename T>
__device__ __forceinline__
T __shfl_sync(unsigned long long mask, T val, int srcLane, int width = warpSize) {
    (void)mask; // mask is ignored on HIP - full warp participates
    return __shfl(val, srcLane, width);
}

template<typename T>
__device__ __forceinline__
T __shfl_up_sync(unsigned long long mask, T val, unsigned delta, int width = warpSize) {
    (void)mask;
    return __shfl_up(val, delta, width);
}

template<typename T>
__device__ __forceinline__
T __shfl_down_sync(unsigned long long mask, T val, unsigned delta, int width = warpSize) {
    (void)mask;
    return __shfl_down(val, delta, width);
}

template<typename T>
__device__ __forceinline__
T __shfl_xor_sync(unsigned long long mask, T val, int laneMask, int width = warpSize) {
    (void)mask;
    return __shfl_xor(val, laneMask, width);
}

// __syncthreads is the same in HIP - no mapping needed

// ============================================================================
// Scoped Atomic Operations
// ============================================================================

// Map CUDA scoped atomicCAS to HIP atomic builtins with explicit scopes

// Block-scoped atomicCAS
template<typename T>
__device__ __forceinline__
T atomicCAS_block(T* address, T compare, T val) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                  "atomicCAS_block only supports 32-bit and 64-bit types");

    T expected = compare;
    __hip_atomic_compare_exchange_strong(
        address,
        &expected,
        val,
        __ATOMIC_RELAXED,
        __ATOMIC_RELAXED,
        __HIP_MEMORY_SCOPE_WORKGROUP
    );
    return expected;
}

// System-scoped atomicCAS
template<typename T>
__device__ __forceinline__
T atomicCAS_system(T* address, T compare, T val) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                  "atomicCAS_system only supports 32-bit and 64-bit types");

    T expected = compare;
    __hip_atomic_compare_exchange_strong(
        address,
        &expected,
        val,
        __ATOMIC_RELAXED,
        __ATOMIC_RELAXED,
        __HIP_MEMORY_SCOPE_SYSTEM
    );
    return expected;
}

// Device-scoped atomicCAS (default scope in CUDA)
// Note: atomicCAS is already defined in HIP with device scope

// ============================================================================
// cuda::std:: to std:: Mappings (C++20)
// ============================================================================

namespace cuda {
namespace std {

// Map cuda::std:: type traits to std::
using ::std::is_same_v;
using ::std::same_as;
using ::std::integral_constant;
using ::std::true_type;
using ::std::false_type;
using ::std::conditional_t;
using ::std::remove_cvref_t;
using ::std::underlying_type_t;
using ::std::is_trivially_copyable_v;

// C++20 bit_cast
using ::std::bit_cast;

// C++11 ignore placeholder (used with std::tie or to discard return values)
using ::std::ignore;

// C++17 byte
using ::std::byte;

// ============================================================================
// Thread Scope Enumeration
// ============================================================================

enum thread_scope : int {
    thread_scope_thread  = 0,
    thread_scope_block   = 1,
    thread_scope_device  = 2,
    thread_scope_system  = 3
};

// ============================================================================
// Memory Order Enumeration
// ============================================================================

enum memory_order : int {
    memory_order_relaxed = __ATOMIC_RELAXED,
    memory_order_consume = __ATOMIC_CONSUME,
    memory_order_acquire = __ATOMIC_ACQUIRE,
    memory_order_release = __ATOMIC_RELEASE,
    memory_order_acq_rel = __ATOMIC_ACQ_REL,
    memory_order_seq_cst = __ATOMIC_SEQ_CST
};

} // namespace std
} // namespace cuda

// Make thread_scope constants available in cuda namespace
namespace cuda {
    constexpr cuda::std::thread_scope thread_scope_thread  = cuda::std::thread_scope_thread;
    constexpr cuda::std::thread_scope thread_scope_block   = cuda::std::thread_scope_block;
    constexpr cuda::std::thread_scope thread_scope_device  = cuda::std::thread_scope_device;
    constexpr cuda::std::thread_scope thread_scope_system  = cuda::std::thread_scope_system;

    constexpr cuda::std::memory_order memory_order_relaxed = cuda::std::memory_order_relaxed;
    constexpr cuda::std::memory_order memory_order_consume = cuda::std::memory_order_consume;
    constexpr cuda::std::memory_order memory_order_acquire = cuda::std::memory_order_acquire;
    constexpr cuda::std::memory_order memory_order_release = cuda::std::memory_order_release;
    constexpr cuda::std::memory_order memory_order_acq_rel = cuda::std::memory_order_acq_rel;
    constexpr cuda::std::memory_order memory_order_seq_cst = cuda::std::memory_order_seq_cst;
}

// ============================================================================
// atomic_ref Implementation for HIP
// ============================================================================

namespace cuda {

namespace detail {

// Helper to convert memory order to HIP atomic scope
__device__ __forceinline__
constexpr int scope_to_hip_scope(cuda::std::thread_scope scope) {
    switch (scope) {
        case thread_scope_thread:  return __HIP_MEMORY_SCOPE_SINGLETHREAD;
        case thread_scope_block:   return __HIP_MEMORY_SCOPE_WORKGROUP;
        case thread_scope_device:  return __HIP_MEMORY_SCOPE_AGENT;
        case thread_scope_system:  return __HIP_MEMORY_SCOPE_SYSTEM;
        default:                   return __HIP_MEMORY_SCOPE_AGENT;
    }
}

} // namespace detail

template<typename T, cuda::std::thread_scope Scope = thread_scope_device>
class atomic_ref {
public:
    using value_type = T;
    static constexpr cuda::std::thread_scope scope = Scope;

    static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                  "atomic_ref only supports 32-bit and 64-bit types");

    __device__ __forceinline__
    explicit atomic_ref(T& ref) : ptr_(&ref) {}

    __device__ __forceinline__
    atomic_ref(const atomic_ref&) noexcept = default;

    atomic_ref& operator=(const atomic_ref&) = delete;

    // Load operation
    // Note: HIP's __hip_atomic_load takes (ptr, order, scope) - 3 args, returning value
    __device__ __forceinline__
    T load(cuda::std::memory_order order = memory_order_seq_cst) const noexcept {
        if constexpr (Scope == thread_scope_block) {
            return __hip_atomic_load(ptr_, static_cast<int>(order), __HIP_MEMORY_SCOPE_WORKGROUP);
        } else if constexpr (Scope == thread_scope_system) {
            return __hip_atomic_load(ptr_, static_cast<int>(order), __HIP_MEMORY_SCOPE_SYSTEM);
        } else {
            return __hip_atomic_load(ptr_, static_cast<int>(order), __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    // Store operation
    // Note: Methods must be const since atomic_ref doesn't own the data
    __device__ __forceinline__
    void store(T desired, cuda::std::memory_order order = memory_order_seq_cst) const noexcept {
        if constexpr (Scope == thread_scope_block) {
            __hip_atomic_store(ptr_, desired, static_cast<int>(order), __HIP_MEMORY_SCOPE_WORKGROUP);
        } else if constexpr (Scope == thread_scope_system) {
            __hip_atomic_store(ptr_, desired, static_cast<int>(order), __HIP_MEMORY_SCOPE_SYSTEM);
        } else {
            __hip_atomic_store(ptr_, desired, static_cast<int>(order), __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    // Compare-exchange strong
    // Note: Must be const since atomic_ref doesn't own the data
    __device__ __forceinline__
    bool compare_exchange_strong(T& expected, T desired,
                                 cuda::std::memory_order success = memory_order_seq_cst,
                                 cuda::std::memory_order failure = memory_order_seq_cst) const noexcept {
        if constexpr (Scope == thread_scope_block) {
            return __hip_atomic_compare_exchange_strong(
                ptr_, &expected, desired,
                static_cast<int>(success), static_cast<int>(failure),
                __HIP_MEMORY_SCOPE_WORKGROUP
            );
        } else if constexpr (Scope == thread_scope_system) {
            return __hip_atomic_compare_exchange_strong(
                ptr_, &expected, desired,
                static_cast<int>(success), static_cast<int>(failure),
                __HIP_MEMORY_SCOPE_SYSTEM
            );
        } else {
            return __hip_atomic_compare_exchange_strong(
                ptr_, &expected, desired,
                static_cast<int>(success), static_cast<int>(failure),
                __HIP_MEMORY_SCOPE_AGENT
            );
        }
    }

    // Compare-exchange weak
    // Note: Must be const since atomic_ref doesn't own the data
    __device__ __forceinline__
    bool compare_exchange_weak(T& expected, T desired,
                               cuda::std::memory_order success = memory_order_seq_cst,
                               cuda::std::memory_order failure = memory_order_seq_cst) const noexcept {
        if constexpr (Scope == thread_scope_block) {
            return __hip_atomic_compare_exchange_weak(
                ptr_, &expected, desired,
                static_cast<int>(success), static_cast<int>(failure),
                __HIP_MEMORY_SCOPE_WORKGROUP
            );
        } else if constexpr (Scope == thread_scope_system) {
            return __hip_atomic_compare_exchange_weak(
                ptr_, &expected, desired,
                static_cast<int>(success), static_cast<int>(failure),
                __HIP_MEMORY_SCOPE_SYSTEM
            );
        } else {
            return __hip_atomic_compare_exchange_weak(
                ptr_, &expected, desired,
                static_cast<int>(success), static_cast<int>(failure),
                __HIP_MEMORY_SCOPE_AGENT
            );
        }
    }

    // Fetch-add operation
    __device__ __forceinline__
    T fetch_add(T arg, cuda::std::memory_order order = memory_order_seq_cst) const noexcept {
        if constexpr (Scope == thread_scope_block) {
            return __hip_atomic_fetch_add(ptr_, arg, static_cast<int>(order), __HIP_MEMORY_SCOPE_WORKGROUP);
        } else if constexpr (Scope == thread_scope_system) {
            return __hip_atomic_fetch_add(ptr_, arg, static_cast<int>(order), __HIP_MEMORY_SCOPE_SYSTEM);
        } else {
            return __hip_atomic_fetch_add(ptr_, arg, static_cast<int>(order), __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    // Fetch-sub operation
    __device__ __forceinline__
    T fetch_sub(T arg, cuda::std::memory_order order = memory_order_seq_cst) const noexcept {
        if constexpr (Scope == thread_scope_block) {
            return __hip_atomic_fetch_sub(ptr_, arg, static_cast<int>(order), __HIP_MEMORY_SCOPE_WORKGROUP);
        } else if constexpr (Scope == thread_scope_system) {
            return __hip_atomic_fetch_sub(ptr_, arg, static_cast<int>(order), __HIP_MEMORY_SCOPE_SYSTEM);
        } else {
            return __hip_atomic_fetch_sub(ptr_, arg, static_cast<int>(order), __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    // Fetch-and operation
    __device__ __forceinline__
    T fetch_and(T arg, cuda::std::memory_order order = memory_order_seq_cst) const noexcept {
        if constexpr (Scope == thread_scope_block) {
            return __hip_atomic_fetch_and(ptr_, arg, static_cast<int>(order), __HIP_MEMORY_SCOPE_WORKGROUP);
        } else if constexpr (Scope == thread_scope_system) {
            return __hip_atomic_fetch_and(ptr_, arg, static_cast<int>(order), __HIP_MEMORY_SCOPE_SYSTEM);
        } else {
            return __hip_atomic_fetch_and(ptr_, arg, static_cast<int>(order), __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    // Fetch-or operation
    __device__ __forceinline__
    T fetch_or(T arg, cuda::std::memory_order order = memory_order_seq_cst) const noexcept {
        if constexpr (Scope == thread_scope_block) {
            return __hip_atomic_fetch_or(ptr_, arg, static_cast<int>(order), __HIP_MEMORY_SCOPE_WORKGROUP);
        } else if constexpr (Scope == thread_scope_system) {
            return __hip_atomic_fetch_or(ptr_, arg, static_cast<int>(order), __HIP_MEMORY_SCOPE_SYSTEM);
        } else {
            return __hip_atomic_fetch_or(ptr_, arg, static_cast<int>(order), __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    // Fetch-xor operation
    __device__ __forceinline__
    T fetch_xor(T arg, cuda::std::memory_order order = memory_order_seq_cst) const noexcept {
        if constexpr (Scope == thread_scope_block) {
            return __hip_atomic_fetch_xor(ptr_, arg, static_cast<int>(order), __HIP_MEMORY_SCOPE_WORKGROUP);
        } else if constexpr (Scope == thread_scope_system) {
            return __hip_atomic_fetch_xor(ptr_, arg, static_cast<int>(order), __HIP_MEMORY_SCOPE_SYSTEM);
        } else {
            return __hip_atomic_fetch_xor(ptr_, arg, static_cast<int>(order), __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    // Exchange operation
    __device__ __forceinline__
    T exchange(T desired, cuda::std::memory_order order = memory_order_seq_cst) const noexcept {
        if constexpr (Scope == thread_scope_block) {
            return __hip_atomic_exchange(ptr_, desired, static_cast<int>(order), __HIP_MEMORY_SCOPE_WORKGROUP);
        } else if constexpr (Scope == thread_scope_system) {
            return __hip_atomic_exchange(ptr_, desired, static_cast<int>(order), __HIP_MEMORY_SCOPE_SYSTEM);
        } else {
            return __hip_atomic_exchange(ptr_, desired, static_cast<int>(order), __HIP_MEMORY_SCOPE_AGENT);
        }
    }

private:
    T* ptr_;
};

// ============================================================================
// Barrier Implementation for HIP
// ============================================================================

template<cuda::std::thread_scope Scope, typename CompletionFunction = void>
class barrier {
public:
    using arrival_token = unsigned int;

    static_assert(Scope == thread_scope_block || Scope == thread_scope_device,
                  "barrier only supports block or device scope");

    __device__ __forceinline__
    explicit barrier(unsigned int expected) : expected_(expected), count_(expected), phase_(0) {}

    barrier(const barrier&) = delete;
    barrier& operator=(const barrier&) = delete;

    // Arrive and wait - blocks until all expected threads have arrived
    __device__ __forceinline__
    void arrive_and_wait() {
        if constexpr (Scope == thread_scope_block) {
            // For block-scoped barriers, use __syncthreads
            __syncthreads();
        } else {
            // For device-scoped barriers, use atomics
            const unsigned int old_phase = *const_cast<volatile unsigned int*>(&phase_);

            // Arrive: decrement counter
            unsigned int arrived;
            if constexpr (Scope == thread_scope_device) {
                arrived = __hip_atomic_fetch_sub(&count_, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
            } else {
                arrived = __hip_atomic_fetch_sub(&count_, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_SYSTEM);
            }

            if (arrived == 1) {
                // Last to arrive - reset counter and advance phase
                count_ = expected_;
                __threadfence(); // Ensure visibility
                __hip_atomic_fetch_add(&phase_, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            } else {
                // Wait for phase change using volatile read with fence
                unsigned int current_phase;
                do {
                    __builtin_amdgcn_s_sleep(1); // Brief pause to reduce contention
                    __threadfence(); // Ensure visibility
                    current_phase = *const_cast<volatile unsigned int*>(&phase_);
                } while (current_phase == old_phase);
            }
        }
    }

    // Separate arrive (returns token)
    __device__ __forceinline__
    arrival_token arrive(unsigned int count = 1) {
        if constexpr (Scope == thread_scope_block) {
            // Block scope - track locally
            unsigned int old = __hip_atomic_fetch_sub(&count_, count, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_WORKGROUP);
            if (old == count) {
                count_ = expected_;
                __hip_atomic_fetch_add(&phase_, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
            }
            __threadfence_block();
            return *const_cast<volatile unsigned int*>(&phase_);
        } else {
            unsigned int old = __hip_atomic_fetch_sub(&count_, count, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
            if (old == count) {
                count_ = expected_;
                __threadfence();
                __hip_atomic_fetch_add(&phase_, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            }
            __threadfence();
            return *const_cast<volatile unsigned int*>(&phase_);
        }
    }

    // Wait on token
    __device__ __forceinline__
    void wait(arrival_token arrival_phase) const {
        if constexpr (Scope == thread_scope_block) {
            unsigned int current;
            do {
                __threadfence_block();
                current = *const_cast<volatile unsigned int*>(&phase_);
            } while (current == arrival_phase);
        } else {
            unsigned int current;
            do {
                __builtin_amdgcn_s_sleep(1);
                __threadfence();
                current = *const_cast<volatile unsigned int*>(&phase_);
            } while (current == arrival_phase);
        }
    }

private:
    unsigned int expected_;
    unsigned int count_;
    mutable unsigned int phase_;
};

} // namespace cuda

// ============================================================================
// BFloat16 Conversion Intrinsics
// ============================================================================

// Note: HIP provides these through hip_bfloat16.h (hip/amd_detail/amd_hip_bf16.h)
// The following functions are already defined in HIP:
//   - __float2bfloat16, __bfloat162float
//   - __float22bfloat162_rn, __bfloat1622float2
// We don't need to redefine them - just use the HIP versions directly.

// ============================================================================
// PTX-like Conversion Stubs (TF32 not available on AMD)
// ============================================================================

// TF32 conversion is a no-op on AMD - we use full FP32
__device__ __forceinline__
float __float_to_tf32(float x) {
    return x; // No TF32 on AMD, use FP32
}

#endif // FLASHMOE_HIP_CUDA_COMPAT_HPP
