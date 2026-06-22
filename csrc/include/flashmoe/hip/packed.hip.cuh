/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 *
 * HIP Port of Packed Data Types for Efficient Communication
 * Target: AMD Instinct MI450 (gfx1250) / MI450 (gfx1250)
 *
 * Key changes from CUDA:
 * - Uses HIP intrinsics for float/uint conversions (__float_as_uint, __uint_as_float)
 * - These intrinsics are identical in HIP, just need HIP headers
 */

#ifndef FLASHMOE_HIP_PACKED_CUH
#define FLASHMOE_HIP_PACKED_CUH

#include <hip/hip_runtime.h>
#include <cstdint>

namespace flashmoe {

#ifndef FLASHMOE_HIP_PACKED_TYPES_DEFINED
#define FLASHMOE_HIP_PACKED_TYPES_DEFINED

    struct __align__(8) TPS {
        uint32_t tokenIdx;
        float probability;
    };

    struct __align__(8) SoftmaxStatePacked {
        uint32_t m_bits;
        uint32_t d_bits;
    };

    __device__ __forceinline__
    auto to_softmax_state(const unsigned long long int& raw) {
        SoftmaxStatePacked s{};
        s.m_bits = static_cast<uint32_t>(raw & 0xFFFFFFFFull);
        s.d_bits = static_cast<uint32_t>(raw >> 32);
        return s;
    }

    __device__ __forceinline__
    SoftmaxStatePacked pack_state(const float& m, const float& d) {
        SoftmaxStatePacked p{};
        p.m_bits = __float_as_uint(m);
        uint32_t db = __float_as_uint(d);
        db |= 0x80000000u;
        p.d_bits = db;
        return p;
    }

    __device__ __forceinline__
    void unpack_state(const SoftmaxStatePacked &p, float &m, float &d) {
        m = __uint_as_float(p.m_bits);
        d = __uint_as_float(p.d_bits & 0x7FFFFFFFu);
    }

    __device__ __forceinline__
    bool has_payload_arrived(const SoftmaxStatePacked &p) {
        return (p.d_bits >> 31) != 0;
    }

    struct __align__(8) RingTopKPayload {
        uint32_t sV = 0U;
        uint32_t sIdx = 0U;
    };

    __device__ __forceinline__
    auto to_tk_payload(const unsigned long long int& raw) {
        RingTopKPayload r{};
        r.sV = static_cast<uint32_t>(raw & 0xFFFFFFFFull);
        r.sIdx = static_cast<uint32_t>(raw >> 32);
        return r;
    }

    __device__ __forceinline__
    auto pack_tk_payload(const float& sV, const uint32_t& sIdx) {
        RingTopKPayload r{};
        r.sV = __float_as_uint(sV);
        r.sIdx = (sIdx & 0x7FFFFFFFu) | 0x80000000u;
        return r;
    }

    __device__ __forceinline__
    void unpack_tk_payload(const RingTopKPayload &p, float &sV, uint32_t &sIdx) {
        sV = __uint_as_float(p.sV);
        sIdx = p.sIdx & 0x7FFFFFFFu;
    }

    __device__ __forceinline__
    bool has_payload_arrived(const RingTopKPayload &p) {
        return (p.sIdx >> 31) != 0;
    }

#endif // FLASHMOE_HIP_PACKED_TYPES_DEFINED

} // namespace flashmoe

#endif // FLASHMOE_HIP_PACKED_CUH
