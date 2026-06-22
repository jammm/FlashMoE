/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 *
 * HIP Port of Type Checking Utilities
 * Target: AMD Instinct MI450 (gfx1250) / MI450 (gfx1250)
 *
 * Key changes from CUDA:
 * - Replaced CUTLASS/CuTe type checks with HIP-native equivalents
 * - Provides isRegister traits for register-resident data detection
 * - Uses std:: type traits instead of cuda::std::
 *
 * Note: CUTLASS Array types are not available on HIP. This provides
 * equivalent type traits for HIP-native array types.
 */

#ifndef FLASHMOE_HIP_CHECKS_CUH
#define FLASHMOE_HIP_CHECKS_CUH

#include <hip/hip_runtime.h>
#include <type_traits>
#include <array>

namespace flashmoe {

    // ============================================================================
    // Register Detection Traits
    // ============================================================================

    /// Base template: not a register type
    template<class T>
    struct isRegister : std::false_type {};

    // ============================================================================
    // HIP-Native Array Types
    // ============================================================================

    /// AlignedArray - Cache-line aligned array for register storage
    /// Replaces cutlass::AlignedArray for HIP
    template<typename T, int N, int Alignment = alignof(T)>
    struct alignas(Alignment) AlignedArray {
        T data[N];

        static constexpr int kElements = N;
        static constexpr int kAlignment = Alignment;

        __device__ __host__ __forceinline__
        T& operator[](int idx) { return data[idx]; }

        __device__ __host__ __forceinline__
        const T& operator[](int idx) const { return data[idx]; }

        __device__ __host__ __forceinline__
        T* ptr() { return data; }

        __device__ __host__ __forceinline__
        const T* ptr() const { return data; }

        __device__ __host__ __forceinline__
        static constexpr int size() { return N; }
    };

    /// Array - Simple array for register storage
    /// Replaces cutlass::Array for HIP
    template<typename T, int N, bool RegisterSized = true>
    struct Array {
        T data[N];

        static constexpr int kElements = N;
        static constexpr bool kRegisterSized = RegisterSized;

        __device__ __host__ __forceinline__
        T& operator[](int idx) { return data[idx]; }

        __device__ __host__ __forceinline__
        const T& operator[](int idx) const { return data[idx]; }

        __device__ __host__ __forceinline__
        T* ptr() { return data; }

        __device__ __host__ __forceinline__
        const T* ptr() const { return data; }

        __device__ __host__ __forceinline__
        static constexpr int size() { return N; }

        // Fill with value
        __device__ __host__ __forceinline__
        void fill(const T& value) {
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                data[i] = value;
            }
        }

        // Clear to zero
        __device__ __host__ __forceinline__
        void clear() {
            fill(T{});
        }
    };

    // Register specializations for our array types
    template<class T, int N, int Alignment>
    struct isRegister<AlignedArray<T, N, Alignment>> : std::true_type {};

    template<class T, int N, bool RegisterSized>
    struct isRegister<Array<T, N, RegisterSized>> : std::true_type {};

    // std::array is also considered register storage (stack allocated)
    template<class T, std::size_t N>
    struct isRegister<std::array<T, N>> : std::true_type {};

    /// Convenience variable template
    template <class T>
    constexpr bool isRegisterV = isRegister<T>::value;

    // ============================================================================
    // Memory Type Detection (HIP equivalents for CuTe traits)
    // ============================================================================

    /// Detect if type is suitable for shared memory
    template<class T>
    struct isSharedMemoryCompatible :
        std::integral_constant<bool,
            std::is_trivially_copyable_v<T> &&
            std::is_standard_layout_v<T>> {};

    template<class T>
    constexpr bool isSharedMemoryCompatibleV = isSharedMemoryCompatible<T>::value;

    /// Detect if type is suitable for global memory
    template<class T>
    struct isGlobalMemoryCompatible :
        std::integral_constant<bool,
            std::is_trivially_copyable_v<T>> {};

    template<class T>
    constexpr bool isGlobalMemoryCompatibleV = isGlobalMemoryCompatible<T>::value;

    // ============================================================================
    // Alignment Helpers
    // ============================================================================

    /// Check if a type has a specific alignment
    template<class T, std::size_t Required>
    struct hasAlignment :
        std::integral_constant<bool, alignof(T) >= Required> {};

    template<class T, std::size_t Required>
    constexpr bool hasAlignmentV = hasAlignment<T, Required>::value;

    /// Check if pointer is aligned
    template<std::size_t Alignment>
    __device__ __host__ __forceinline__
    bool isAligned(const void* ptr) {
        return reinterpret_cast<std::uintptr_t>(ptr) % Alignment == 0;
    }

    // ============================================================================
    // Vectorization Support
    // ============================================================================

    /// Maximum vectorization width for a type
    template<typename T>
    struct MaxVectorWidth {
        // Default: 128 bits / sizeof(T)
        static constexpr int value = 16 / sizeof(T);
    };

    // Specializations for common types
    template<> struct MaxVectorWidth<float>  { static constexpr int value = 4; };  // float4
    template<> struct MaxVectorWidth<double> { static constexpr int value = 2; };  // double2
    template<> struct MaxVectorWidth<__half> { static constexpr int value = 8; };  // half8
    template<> struct MaxVectorWidth<hip_bfloat16> { static constexpr int value = 8; };

    template<typename T>
    constexpr int MaxVectorWidthV = MaxVectorWidth<T>::value;

} // namespace flashmoe

#endif // FLASHMOE_HIP_CHECKS_CUH
