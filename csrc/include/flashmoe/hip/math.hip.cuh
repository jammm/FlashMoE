/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 *
 * HIP Port of Math Utilities
 * Target: AMD Instinct MI450 (gfx1250) / MI450 (gfx1250)
 *
 * Key changes from CUDA:
 * - cuda::round_up -> custom round_up implementation
 * - Maintains same cache-line rounding semantics for optimal memory access
 */

#ifndef FLASHMOE_HIP_MATH_CUH
#define FLASHMOE_HIP_MATH_CUH

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace flashmoe {

    /// Round up to next multiple of alignment
    /// Compile-time constexpr version for template parameters
    template<typename T>
    __host__ __device__ __forceinline__
    constexpr T round_up(T value, T alignment) {
        static_assert(std::is_integral_v<T>, "round_up requires integral type");
        return ((value + alignment - 1) / alignment) * alignment;
    }

    /// Round to cache line - calculates bytes needed for data with cache-line alignment
    /// Also applies to shared memory banks for bank-conflict-free access patterns
    ///
    /// Template Parameters:
    ///   Element - Data type for size calculation
    ///
    /// Parameters:
    ///   len - Number of elements
    ///
    /// Returns: Bytes rounded up to cache line boundary (128 bytes on AMD MI450)
    ///
    /// Note: use 128-byte alignment for coalesced vector memory access.
    template <typename Element>
    __device__ __forceinline__
    constexpr auto rTCL(unsigned int const& len) {
        constexpr size_t lineBytes = 128;  // 128 bytes for optimal wavefront access
        return round_up(static_cast<size_t>(len) * sizeof(Element), lineBytes);
    }

    /// Host version of rTCL for kernel configuration
    template <typename Element>
    __host__ __forceinline__
    constexpr size_t rTCL_host(unsigned int len) {
        constexpr size_t lineBytes = 128;
        return round_up(static_cast<size_t>(len) * sizeof(Element), lineBytes);
    }

    /// Divide and round up
    template<typename T>
    __host__ __device__ __forceinline__
    constexpr T div_up(T dividend, T divisor) {
        return (dividend + divisor - 1) / divisor;
    }

    /// Check if value is a power of two
    template<typename T>
    __host__ __device__ __forceinline__
    constexpr bool is_power_of_two(T value) {
        return value > 0 && (value & (value - 1)) == 0;
    }

    /// Round down to nearest power of two
    template<typename T>
    __host__ __device__ __forceinline__
    constexpr T round_down_pow2(T value) {
        static_assert(std::is_integral_v<T>, "requires integral type");
        if (value == 0) return 0;
        T result = 1;
        while (result * 2 <= value) {
            result *= 2;
        }
        return result;
    }

    /// Round up to nearest power of two
    template<typename T>
    __host__ __device__ __forceinline__
    constexpr T round_up_pow2(T value) {
        if (value == 0) return 1;
        if (is_power_of_two(value)) return value;
        return round_down_pow2(value) * 2;
    }

} // namespace flashmoe

#endif // FLASHMOE_HIP_MATH_CUH
