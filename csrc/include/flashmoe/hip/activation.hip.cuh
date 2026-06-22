/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 *
 * HIP Port of Activation Functions
 * Target: AMD Instinct MI450 (gfx1250) / MI450 (gfx1250)
 *
 * Key changes from CUDA:
 * - Replaced cublasdx::identity and cutlass activation functions
 * - Implemented native HIP activation functors
 * - Uses HIP math intrinsics for performance
 *
 * Note: CUTLASS/cuBLASDx are NVIDIA-only. This provides equivalent
 * activation functors for AMD GPUs using HIP intrinsics.
 */

#ifndef FLASHMOE_HIP_ACTIVATION_CUH
#define FLASHMOE_HIP_ACTIVATION_CUH

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cmath>

namespace flashmoe {

#ifndef FLASHMOE_HIP_ACTIVATION_DEFINED
#define FLASHMOE_HIP_ACTIVATION_DEFINED
    enum class Activation {
        identity,
        silu,
        gelu,
        relu
    };
#endif

    // ============================================================================
    // Identity Activation
    // ============================================================================

    /// Identity activation - passthrough
    template<typename Element>
    struct Identity {
        __device__ __forceinline__
        Element operator()(const Element& x) const {
            return x;
        }

        // Vector version for efficiency
        template<typename Vector>
        __device__ __forceinline__
        Vector apply(const Vector& v) const {
            return v;
        }
    };

    // ============================================================================
    // ReLU Activation
    // ============================================================================

    /// ReLU: max(0, x)
    template<typename Element>
    struct ReLU {
        __device__ __forceinline__
        Element operator()(const Element& x) const {
            return x > Element(0) ? x : Element(0);
        }
    };

    // Specializations for half/bfloat16
    template<>
    struct ReLU<__half> {
        __device__ __forceinline__
        __half operator()(const __half& x) const {
            return __hgt(x, __float2half(0.0f)) ? x : __float2half(0.0f);
        }
    };

    template<>
    struct ReLU<hip_bfloat16> {
        __device__ __forceinline__
        hip_bfloat16 operator()(const hip_bfloat16& x) const {
            float fx = static_cast<float>(x);
            return hip_bfloat16(fx > 0.0f ? fx : 0.0f);
        }
    };

    // ============================================================================
    // GELU Activation
    // ============================================================================

    /// GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    /// Using fast approximation: x * sigmoid(1.702 * x)
    template<typename Element>
    struct GELU {
        __device__ __forceinline__
        Element operator()(const Element& x) const {
            // Fast GELU approximation
            constexpr float kBeta = 1.702f;
            float fx = static_cast<float>(x);
            float sigmoid = 1.0f / (1.0f + expf(-kBeta * fx));
            return static_cast<Element>(fx * sigmoid);
        }
    };

    template<>
    struct GELU<float> {
        __device__ __forceinline__
        float operator()(const float& x) const {
            // Exact GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            constexpr float kAlpha = 0.7978845608f;  // sqrt(2/pi)
            constexpr float kBeta = 0.044715f;
            float x3 = x * x * x;
            float inner = kAlpha * (x + kBeta * x3);
            return 0.5f * x * (1.0f + tanhf(inner));
        }
    };

    template<>
    struct GELU<__half> {
        __device__ __forceinline__
        __half operator()(const __half& x) const {
            float fx = __half2float(x);
            constexpr float kAlpha = 0.7978845608f;
            constexpr float kBeta = 0.044715f;
            float x3 = fx * fx * fx;
            float inner = kAlpha * (fx + kBeta * x3);
            float result = 0.5f * fx * (1.0f + tanhf(inner));
            return __float2half(result);
        }
    };

    template<>
    struct GELU<hip_bfloat16> {
        __device__ __forceinline__
        hip_bfloat16 operator()(const hip_bfloat16& x) const {
            float fx = static_cast<float>(x);
            constexpr float kAlpha = 0.7978845608f;
            constexpr float kBeta = 0.044715f;
            float x3 = fx * fx * fx;
            float inner = kAlpha * (fx + kBeta * x3);
            float result = 0.5f * fx * (1.0f + tanhf(inner));
            return hip_bfloat16(result);
        }
    };

    // ============================================================================
    // SiLU (Swish) Activation
    // ============================================================================

    /// SiLU/Swish: x * sigmoid(x) = x / (1 + exp(-x))
    template<typename Element>
    struct SiLU {
        __device__ __forceinline__
        Element operator()(const Element& x) const {
            float fx = static_cast<float>(x);
            float sigmoid = 1.0f / (1.0f + expf(-fx));
            return static_cast<Element>(fx * sigmoid);
        }
    };

    template<>
    struct SiLU<float> {
        __device__ __forceinline__
        float operator()(const float& x) const {
            return x / (1.0f + expf(-x));
        }
    };

    template<>
    struct SiLU<__half> {
        __device__ __forceinline__
        __half operator()(const __half& x) const {
            float fx = __half2float(x);
            float result = fx / (1.0f + expf(-fx));
            return __float2half(result);
        }
    };

    template<>
    struct SiLU<hip_bfloat16> {
        __device__ __forceinline__
        hip_bfloat16 operator()(const hip_bfloat16& x) const {
            float fx = static_cast<float>(x);
            float result = fx / (1.0f + expf(-fx));
            return hip_bfloat16(result);
        }
    };

    // ============================================================================
    // Activation Type Selector
    // ============================================================================

    /// Map Activation enum to functor type
    template<typename Element, Activation a = Activation::identity>
    struct ActivationType {
        static_assert(a == Activation::identity);
        using AT = Identity<Element>;
    };

    template<typename Element>
    struct ActivationType<Element, Activation::relu> {
        using AT = ReLU<Element>;
    };

    template<typename Element>
    struct ActivationType<Element, Activation::gelu> {
        using AT = GELU<Element>;
    };

    template<typename Element>
    struct ActivationType<Element, Activation::silu> {
        using AT = SiLU<Element>;
    };

    // ============================================================================
    // Activation Apply Helper
    // ============================================================================

    /// Apply activation function to a value
    template<Activation a, typename Element>
    __device__ __forceinline__
    Element apply_activation(const Element& x) {
        using Functor = typename ActivationType<Element, a>::AT;
        return Functor{}(x);
    }

    /// Apply activation in-place to an array
    template<Activation a, typename Element, int N>
    __device__ __forceinline__
    void apply_activation_inplace(Element* data) {
        using Functor = typename ActivationType<Element, a>::AT;
        Functor fn;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            data[i] = fn(data[i]);
        }
    }

} // namespace flashmoe

#endif // FLASHMOE_HIP_ACTIVATION_CUH
