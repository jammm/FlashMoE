/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 *
 * HIP Port: Converted from CUDA for AMD GPU support.
 * Target: AMD Instinct MI450/MI450 (gfx1250)
 */

//
// Created by osayamen on 1/15/26.
// HIP Port: 2026
//

#ifndef FLASHMOE_HIP_CONTEXT_CUH
#define FLASHMOE_HIP_CONTEXT_CUH

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstddef>

// FlashMoE HIP compatibility headers
#include "flashmoe/hip/cuda_compat.hpp"
#include "flashmoe/hip/tensor.hpp"
#include "flashmoe/hip/constants.hpp"

// ============================================================================
// cuda::is_aligned implementation for HIP
// ============================================================================

namespace cuda {

/// Check if a pointer is aligned to the specified boundary
/// @param ptr The pointer to check
/// @param alignment The alignment boundary (must be a power of 2)
/// @return true if ptr is aligned to alignment bytes, false otherwise
__host__ __device__ __forceinline__
bool is_aligned(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

} // namespace cuda

// ============================================================================
// Include HIP-ported infra headers
// Note: These types use cuda::std:: -> std:: mappings from cuda_compat.hpp
// ============================================================================

// For structures with cuda::std::byte* pointers, we use std::byte from cuda_compat.hpp
// The cuda:: namespace mappings in cuda_compat.hpp handle this transparently

// Forward declarations for infra types
// These are compatible with both CUDA and HIP due to their use of standard types
namespace flashmoe {

// ============================================================================
// Ported Infra Types (from infra/*.cuh)
// Guarded to avoid redefinition when signal.hip.cuh is included first
// ============================================================================

#ifndef FLASHMOE_HIP_TOPOLOGY_DEFINED
#define FLASHMOE_HIP_TOPOLOGY_DEFINED
enum class Topology: uint16_t {
    NVLINK_ONLY,
    MIXED
};
#endif

#ifndef FLASHMOE_HIP_STRUCTURES_DEFINED
#define FLASHMOE_HIP_STRUCTURES_DEFINED
struct __align__(16) PEL {
    std::byte* remoteSHeap;
    uint64_t* remoteSFlags;
    uint eC;
    uint16_t pTTt;
    uint16_t expertLocalIdx;
    uint16_t peer;
    uint16_t pe;
    uint16_t isRemote;
    uint16_t nLocalExperts;

    __host__ __device__ __forceinline__
    void dump() const {
        printf("{\n\t"
               "this: %p,\n\t"
               "remoteSHeap: %p,\n\t"
               "remoteSFlags: %p,\n\t"
               "eC: %u,\n\t"
               "pTTt: %u,\n\t"
               "expertLocalIndex: %u,\n\t"
               "peer: %u,\n\t"
               "pe: %u,\n\t"
               "isRemote: %s,\n\t"
               "nLocalExperts: %u"
               "\n}\n",
               this, remoteSHeap, remoteSFlags, eC, pTTt, expertLocalIdx,
               peer, pe, isRemote ? "True" : "False", nLocalExperts);
    }
};

/// Peer lookup info: key is ep rank
struct __align__(8) PLI {
    std::byte* remoteSHeap;
    uint64_t* remoteSFlags;
    uint pe;
    uint isRemote;

    __host__ __device__ __forceinline__
    void dump() const {
        printf("{\n\t"
               "this: %p,\n\t"
               "remoteSHeap: %p,\n\t"
               "remoteSFlags: %p,\n\t"
               "pe: %u,\n\t"
               "isRemote: %s"
               "\n}\n",
               this, remoteSHeap, remoteSFlags, pe, isRemote ? "True" : "False");
    }
};

/// Expert lookup info: key is global expert index
struct __align__(8) ELI {
    uint epRank; // host peer
    uint16_t localExpertIndex;
    uint16_t isRemote;

    __host__ __device__ __forceinline__
    void dump() const {
        printf("{\n\t"
               "this: %p,\n\t"
               "epRank: %u,\n\t"
               "localExpertIndex: %u,\n\t"
               "isRemote: %s"
               "\n}\n",
               this,
               epRank, localExpertIndex, isRemote ? "True" : "False");
    }
};

/// Local expert lookup: key is local expert index
struct __align__(4) LXI {
    uint expertIndex;
    __host__ __device__ __forceinline__
    void dump() const {
        printf("{\n\t"
               "expertIndex: %u\n}\n", expertIndex);
    }
};
#endif // FLASHMOE_HIP_STRUCTURES_DEFINED

#ifndef FLASHMOE_HIP_BITSET_DEFINED
#define FLASHMOE_HIP_BITSET_DEFINED
struct __align__(4) BitSet {
    uint storage = 0U;
    __device__ __forceinline__
    uint8_t get(const uint idx) const {
        return storage >> idx & 1U;
    }
    __device__ __forceinline__
    void set(const uint idx) {
        storage |= 1U << idx;
    }
    __device__ __forceinline__
    void clear(const uint idx) {
        storage &= ~(1U << idx);
    }
    __device__ __forceinline__
    void flip(const uint idx) {
        storage ^= (1U << idx);
    }
};
#endif // FLASHMOE_HIP_BITSET_DEFINED

#ifndef FLASHMOE_HIP_NSI_DEFINED
#define FLASHMOE_HIP_NSI_DEFINED
template<unsigned int T>
__device__ __forceinline__
constexpr uint nSI(const unsigned int& numBits) {
    constexpr unsigned int integerBitWidth = 32U;
    constexpr auto width = integerBitWidth * T;
    return (numBits / width) * T + std::min(numBits % width, T);
}

__host__ __forceinline__
constexpr uint nSI(const unsigned int& numBits, const uint& T) {
    constexpr unsigned int integerBitWidth = 32U;
    const auto width = integerBitWidth * T;
    return (numBits / width) * T + std::min(numBits % width, T);
}
#endif // FLASHMOE_HIP_NSI_DEFINED

/// TQSignal for task queue signaling
struct __align__(8) TQSignal {
    uint signal; // one ahead
    uint interrupt;

    __device__ __forceinline__
    void encodeSig(const uint& sig) {
        signal = sig + 1;
    }
    __device__ __forceinline__
    auto decodeSig() const {
        return signal - 1;
    }
};

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

#ifndef FLASHMOE_HIP_TASK_TYPES_DEFINED
#define FLASHMOE_HIP_TASK_TYPES_DEFINED

constexpr int GEMMs = 2;
enum class TaskType : uint8_t {
    GEMM0,
    GEMM1,
    combine,
};
static_assert(std::is_same_v<std::underlying_type_t<TaskType>, uint8_t>);

struct __align__(16) Ingredients {
    unsigned int M = 0; // GEMM0->number of tokens or Combine->tokenBatchStart
    uint16_t localExpertIdx = 0;
    uint16_t expertIdx = 0; // global
    uint16_t peerIdx = 0; // owner of the output
    uint16_t tileSize = 0; // <= BLOCK_M
    uint16_t stash = 0; // GEMM0->flagBatchIdx or Combine->tileIdx
    TaskType taskType = TaskType::GEMM0;
    uint8_t isPeerRemote = 0;

    Ingredients() = default;

    // GEMM0 and GEMM1
    __device__ __forceinline__
    Ingredients(const uint16_t& lei, const uint16_t& pei, const TaskType& tt, const uint8_t& ipr):
    localExpertIdx(lei), peerIdx(pei), taskType(tt), isPeerRemote(ipr){}

    __device__ __forceinline__
    Ingredients(const unsigned int& m, const uint16_t& lei, const uint16_t& ei,
        const uint16_t& pei, const uint16_t& ts, const uint16_t& sta,
        const TaskType& tt, const uint8_t& ipr):
    M(m), localExpertIdx(lei), expertIdx(ei), peerIdx(pei), tileSize(ts), stash(sta),
    taskType(tt), isPeerRemote(ipr){}
};
static_assert(sizeof(Ingredients) == 16);

/// Task structure for kernel dispatch
struct __align__(16) Task {
    Ingredients ingredients{};
    const std::byte* aData = nullptr;
    std::array<std::byte*, GEMMs> cData = {};
    std::byte* rcData = nullptr;
    uint64_t* flags = nullptr;
    unsigned int syncIdx = 0U;
    unsigned int tileIdx = 0U;

    Task() = default;

    // GEMM0->GEMM1
    __device__ __forceinline__
    Task(const Ingredients& _ingredients,
        const std::byte* const& _aData,
        const std::array<std::byte*, GEMMs>& _cData,
        std::byte* const& _rcData,
        uint64_t* const& _flags,
        const unsigned int& _syncIdx, const unsigned int& tile):
    ingredients(_ingredients), aData(_aData), cData(_cData), rcData(_rcData), flags(_flags),
    syncIdx(_syncIdx), tileIdx(tile){}

    // Combine
    __device__ __forceinline__
    explicit Task(const Ingredients& _ingredients):
    ingredients(_ingredients){}

    __device__ __forceinline__
    auto getTaskType() const {
        return ingredients.taskType;
    }
    __device__ __forceinline__
    auto isPeerRemote() const {
        return ingredients.isPeerRemote;
    }
    __device__ __forceinline__
    auto localExpertIdx() const {
        return ingredients.localExpertIdx;
    }
    __device__ __forceinline__
    auto M() const {
        return ingredients.M;
    }
    __device__ __forceinline__
    auto tileSize() const {
        return ingredients.tileSize;
    }
    __device__ __forceinline__
    auto flagBatchIdx() const {
        return ingredients.stash;
    }
    __device__ __forceinline__
    auto pe() const {
        return ingredients.peerIdx;
    }
    __device__ __forceinline__
    auto localPeerIdx() const {
        return ingredients.stash;
    }
    __device__ __forceinline__
    auto tokenBatchStart() const {
        return ingredients.M;
    }
    __device__ __forceinline__
    auto combineTileIdx() const{
        return ingredients.stash;
    }
    __device__ __forceinline__
    auto expertIdx() const{
        return ingredients.expertIdx;
    }
    __device__ __forceinline__
    auto epRank() const{
        return ingredients.peerIdx;
    }
};
static_assert(sizeof(Task) == 64);

#endif // FLASHMOE_HIP_TASK_TYPES_DEFINED

// ============================================================================
// cuda::fast_mod_div for HIP (from scheduler.hip.cuh)
// ============================================================================

} // namespace flashmoe

#ifndef FLASHMOE_HIP_FAST_MOD_DIV_DEFINED
#define FLASHMOE_HIP_FAST_MOD_DIV_DEFINED
namespace cuda {
template<typename T>
struct fast_mod_div {
    T divisor_;
    __host__ __device__ fast_mod_div() : divisor_(1) {}
    __host__ __device__ explicit fast_mod_div(T divisor) : divisor_(divisor) {}
    __device__ __forceinline__ T operator()(T dividend) const {
        return dividend / divisor_;
    }
    __device__ __forceinline__ friend T operator%(T dividend, const fast_mod_div& fmd) {
        return dividend % fmd.divisor_;
    }
};
} // namespace cuda
#endif

namespace flashmoe
{
// ============================================================================
// Context Structures
// ============================================================================

/// Main runtime context for FlashMoE kernel execution
/// Contains all pointers and configuration for MoE dispatch/combine operations
struct Context {
    std::byte* const symHeap = nullptr;                    // Symmetric heap for ROCSHMEM
    uint64_t* const signals = nullptr;                     // [[world, num_local_experts], [E, tiles(roundEC), tiles(H)]]
    Task* const tQ = nullptr;                              // [subscriberTQLength]
    Task* const pTq = nullptr;                             // [secondaryTQLength]
    // [world, num_local_experts, roundEC, I] ~= [S, I]
    std::byte* const GEMM0Staging = nullptr;
    BitSet* const consumerCombineBitMap = nullptr;         // nSI<subscriberCount>(tiles(S) * tiles(H))
    uint8_t* const producerCombineBitMap = nullptr;        // [world, nLx, ecTilesM, N1] ~= tiles(S) * tiles(H)
    PEL* const pel = nullptr;                              // [E]
    PLI* const pli = nullptr;                              // [world]
    ELI* const eli = nullptr;                              // [E]
    LXI* const lxi = nullptr;                              // [num_local_experts]
    TQSignal* const tqs = nullptr;                         // [processors]
    uint* const dispatchSync = nullptr;                    // [E]
    uint* const gTqHeads = nullptr;                        // [world, num_local_experts, ecTilesM] = tiles(S)
    uint* const tileSync = nullptr;                        // [world, num_local_experts, ecTilesM] = tiles(S)
    uint* const statusQueue = nullptr;                     // [processors]
    TPS* const tokenIndices = nullptr;                     // [E, roundEC]
    const cuda::fast_mod_div<uint> processors_v;
    const uint blocks = 0;
    const uint smemSize = 0;
    const uint S = 0;                                      // max number of tokens for this rank
    const uint H = 0;                                      // max hidden dimension or model dim
    const uint I = 0;                                      // max FFN intermediate size
    const uint EC = 0;                                     // max EC
    const uint16_t bM = 0;
    const uint16_t bN0 = 0;
    const uint16_t bN1 = 0;
    const uint16_t nLx = 0;
    const uint16_t E = 0;
    const uint16_t world = 0;
    const uint16_t epRank = 0;
    const uint16_t myPE = 0;
    const Topology topo = Topology::MIXED;
    uint8_t* stateNumbers = nullptr;
    static_assert(alignof(cuda::fast_mod_div<uint>) <= 8);
};

/// Gate context for gating network operations
/// Contains pointers for expert capacity guards, softmax state, and ring top-k
struct __align__(8) GateContext {
    int* const ecGuards = nullptr;                         // [E]
    SoftmaxStatePacked* const ssp = nullptr;               // [S, tiles(E)]
    RingTopKPayload* const rtp = nullptr;                  // [2, S, tiles(E)]
    uint blocks = 0;
    // Note: cuda::barrier is available through cuda_compat.hpp if needed
    // cuda::barrier<cuda::thread_scope_device>* const db = nullptr; // [1]
};

} // namespace flashmoe

#endif // FLASHMOE_HIP_CONTEXT_CUH
