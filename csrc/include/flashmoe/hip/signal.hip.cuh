/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 *
 * HIP Port of Signal Types and State Machine
 * Target: AMD Instinct MI450 (gfx1250) / MI450 (gfx1250)
 */

#ifndef FLASHMOE_HIP_SIGNAL_CUH
#define FLASHMOE_HIP_SIGNAL_CUH

#include <hip/hip_runtime.h>
#include <cstdint>

namespace flashmoe {

#ifndef FLASHMOE_HIP_TOPOLOGY_DEFINED
#define FLASHMOE_HIP_TOPOLOGY_DEFINED
    enum class Topology: uint16_t {
        NVLINK_ONLY,
        MIXED
    };
#endif

    enum class PeerConnectivity {
        remote,
        p2p
    };

    // Captures transitory states of a finite state machine
    enum SignalConstants {
        ground = 0U,
        sequenceStart = 1U,
    };

    enum class PacketStage: unsigned int {
        initial,
        last,
    };

    template<PacketStage p = PacketStage::initial>
    struct __align__(8) SignalPayload {
        static_assert(p == PacketStage::initial);
        unsigned int routedTokens;
        uint16_t totalTilesM;
        uint16_t stateNumber;

        __device__ __forceinline__
        void dump(const unsigned int& lex) const {
            printf("{\n\t"
                   "routedTokens: %u,\n\t"
                   "localExpertIdx: %u,\n\t"
                   "totalTilesM: %u,\n\t"
                   "state: %u"
                   "\n}\n",
                   routedTokens, lex, totalTilesM, stateNumber);
        }
    };

    template<>
    struct __align__(8) SignalPayload<PacketStage::last> {
        unsigned int batchIdx;
        uint16_t tokensM; // <= BLOCK_M
        uint8_t senseBit;
        uint8_t stateNumber;

        __device__ __forceinline__
        void dump(const unsigned int& ex, const unsigned int& col) const {
            printf("{\n\t"
                   "batchIdx: %u,\n\t"
                   "expertIdx: %u,\n\t"
                   "colIdx: %u,\n\t"
                   "tokensM: %u,\n\t"
                   "stateNumber: %u"
                   "\n}\n",
                   batchIdx, ex, col, tokensM, stateNumber);
        }
    };

} // namespace flashmoe

namespace flashmoe::sbs {
    constexpr int AEE = 0; // allowable early exits
    constexpr int IDZ = AEE + 1;
    // sequence bit states necessary to break symmetry in forward or backward detection
    // includes ground state
    constexpr int SNS = (2 * (2 + AEE));

    __forceinline__ __host__
    constexpr uint16_t next(const uint16_t& current) {
        return current + 1 == SNS ?
            static_cast<decltype(current)>(sequenceStart) : current + 1;
    }

    __forceinline__ __device__
    constexpr auto ahead(const uint16_t& receivedState, const uint16_t& localState) {
        if (receivedState < sequenceStart) {
            // this is the case, when we observe the ground state
            return false;
        }
        const auto wD = (SNS - localState) + (receivedState -
            static_cast<decltype(receivedState)>(sequenceStart));
        return (receivedState > localState && ((receivedState - localState) <= IDZ)) ||
            (receivedState < localState && wD <= IDZ);
    }
}

#endif // FLASHMOE_HIP_SIGNAL_CUH
