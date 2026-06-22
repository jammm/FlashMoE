/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 *
 * HIP Port of Core Data Structures
 * Target: AMD Instinct MI450 (gfx1250) / MI450 (gfx1250)
 *
 * Key changes from CUDA:
 * - cuda::std::byte -> std::byte (C++17)
 * - Uses standard C++ types from cuda_compat.hpp namespace aliasing
 */

#ifndef FLASHMOE_HIP_STRUCTURES_CUH
#define FLASHMOE_HIP_STRUCTURES_CUH

#include <hip/hip_runtime.h>
#include <cstddef>    // std::byte
#include <cstdint>

// Include compat layer for cuda::std:: -> std:: mappings
#include "flashmoe/hip/cuda_compat.hpp"

namespace flashmoe {

#ifndef FLASHMOE_HIP_STRUCTURES_DEFINED
#define FLASHMOE_HIP_STRUCTURES_DEFINED
    struct __align__(16) PEL {
        std::byte* remoteSHeap;
        uint64_t* remoteSFlags;
        unsigned int eC;
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

    /// Expert lookup info: key is global expert index
    struct __align__(8) ELI {
        unsigned int epRank; // host peer
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
        unsigned int expertIndex;

        __host__ __device__ __forceinline__
        void dump() const {
            printf("{\n\t"
                   "expertIndex: %u\n}\n", expertIndex);
        }
    };

    /// Peer lookup info: key is ep rank
    struct __align__(8) PLI {
        std::byte* remoteSHeap;
        uint64_t* remoteSFlags;
        unsigned int pe;
        unsigned int isRemote;

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
#endif // FLASHMOE_HIP_STRUCTURES_DEFINED

} // namespace flashmoe

#endif // FLASHMOE_HIP_STRUCTURES_CUH
