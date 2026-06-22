/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 *
 * HIP Port of BitSet Utilities
 * Target: AMD MI450 (gfx1250)
 *
 * Key changes from CUDA:
 * - cuda::std::min -> std::min (using algorithm header)
 * - Updated nSI template for wave-aligned bitsets where applicable
 */

#ifndef FLASHMOE_HIP_BITSET_CUH
#define FLASHMOE_HIP_BITSET_CUH

#include <hip/hip_runtime.h>
#include <cstdint>
#include <algorithm>  // std::min

#include "flashmoe/hip/constants.hpp"

namespace flashmoe {

#ifndef FLASHMOE_HIP_BITSET_DEFINED
#define FLASHMOE_HIP_BITSET_DEFINED
    struct __align__(4) BitSet {
        unsigned int storage = 0U;

        /// Get bit at index (0 = unset, 1 = set)
        __device__ __forceinline__
        uint8_t get(const unsigned int idx) const {
            return storage >> idx & 1U;
        }

        /// Set bit at index
        __device__ __forceinline__
        void set(const unsigned int idx) {
            storage |= 1U << idx;
        }

        /// Clear bit at index
        __device__ __forceinline__
        void clear(const unsigned int idx) {
            storage &= ~(1U << idx);
        }

        /// Flip bit at index
        __device__ __forceinline__
        void flip(const unsigned int idx) {
            storage ^= (1U << idx);
        }

        /// Count number of set bits (popcount)
        __device__ __forceinline__
        unsigned int count() const {
            return __popc(storage);
        }

        /// Check if any bit is set
        __device__ __forceinline__
        bool any() const {
            return storage != 0U;
        }

        /// Check if no bits are set
        __device__ __forceinline__
        bool none() const {
            return storage == 0U;
        }

        /// Find first set bit (returns 32 if none)
        __device__ __forceinline__
        unsigned int ffs() const {
            return storage == 0 ? 32 : __ffs(storage) - 1;
        }

        /// Reset all bits
        __device__ __forceinline__
        void reset() {
            storage = 0U;
        }
    };
#endif // FLASHMOE_HIP_BITSET_DEFINED

#ifndef FLASHMOE_HIP_NSI_DEFINED
#define FLASHMOE_HIP_NSI_DEFINED
    /// Computes precise number of integers needed to represent a consecutive set of bits
    /// Each of T threads has stride ownership of a single bit
    /// and requires an integer to store 32 of such bits.
    ///
    /// Template Parameters:
    ///   T - Number of threads participating
    ///
    /// Parameters:
    ///   numBits - Total number of bits to represent
    ///
    /// Returns: Number of 32-bit integers needed
    ///
    /// Note: use flashmoe::WARP_SIZE for wave-aligned operations.
    template<unsigned int T>
    __device__ __forceinline__
    constexpr unsigned int nSI(const unsigned int& numBits) {
        constexpr unsigned int integerBitWidth = 32U;
        constexpr auto width = integerBitWidth * T;
        // Use std::min from constexpr context
        const unsigned int remainder = numBits % width;
        const unsigned int base = (numBits / width) * T;
        return base + (remainder < T ? remainder : T);
    }

    /// Host version of nSI for runtime thread count
    __host__ __forceinline__
    constexpr unsigned int nSI(const unsigned int& numBits, const unsigned int& T) {
        constexpr unsigned int integerBitWidth = 32U;
        const auto width = integerBitWidth * T;
        const unsigned int remainder = numBits % width;
        const unsigned int base = (numBits / width) * T;
        return base + (remainder < T ? remainder : T);
    }
#endif // FLASHMOE_HIP_NSI_DEFINED

    /// Extended BitSet for larger bit ranges (up to 64 bits)
    struct __align__(8) BitSet64 {
        unsigned long long storage = 0ULL;

        __device__ __forceinline__
        uint8_t get(const unsigned int idx) const {
            return (storage >> idx) & 1ULL;
        }

        __device__ __forceinline__
        void set(const unsigned int idx) {
            storage |= 1ULL << idx;
        }

        __device__ __forceinline__
        void clear(const unsigned int idx) {
            storage &= ~(1ULL << idx);
        }

        __device__ __forceinline__
        void flip(const unsigned int idx) {
            storage ^= (1ULL << idx);
        }

        __device__ __forceinline__
        unsigned int count() const {
            return __popcll(storage);
        }

        __device__ __forceinline__
        bool any() const {
            return storage != 0ULL;
        }

        __device__ __forceinline__
        void reset() {
            storage = 0ULL;
        }
    };

} // namespace flashmoe

#endif // FLASHMOE_HIP_BITSET_CUH
