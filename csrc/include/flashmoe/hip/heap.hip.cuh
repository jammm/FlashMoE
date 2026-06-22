/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 *
 * HIP Port of heap.cuh
 * Target: AMD Instinct MI450 (gfx1250) / MI450 (gfx1250)
 *
 * Changes from CUDA version:
 * - Replaced cuda::std::byte with std::byte via cuda_compat.hpp
 * - Uses HIP-compatible constants from constants.hpp
 */

#ifndef FLASHMOE_HIP_HEAP_CUH
#define FLASHMOE_HIP_HEAP_CUH

#include "flashmoe/hip/cuda_compat.hpp"
#include "flashmoe/hip/constants.hpp"

namespace flashmoe {

/// Number of communication stages
constexpr int HEAP_STAGES = 2;
/// Per stage, there is one cell for sending and another for reception
constexpr int HEAP_CELLS = 2;
constexpr int HEAP_SEND_CELL = 0;
constexpr int HEAP_RECEIVE_CELL = 1;

/// The symmetric tensor from the FlashMoE paper
/// Used for symmetric memory allocation with rocSHMEM on AMD GPUs
struct Heap {
    cuda::std::byte* const sHeap;

    __device__ __forceinline__
    Heap(cuda::std::byte* const& _sHeap,
         const uint& _expertSlots, const uint& _EC,
         const uint& _tokenDim, const uint& _elementBytes) :
        sHeap(_sHeap), expertSlots(_expertSlots), EC(_EC),
        tokenDim(_tokenDim), elementBytes(_elementBytes) {}

    /**
     * Compute the byte offset to a specific location in the heap
     *
     * Template Parameters:
     *   stage - Communication stage (0 or 1)
     *   cell  - Cell type (send=0 or receive=1)
     *
     * Parameters:
     *   peer   - Peer PE index
     *   expert - Expert slot index
     *   token  - Token index within the slot (default 0)
     *
     * Returns: Byte offset from heap base
     */
    template<
        int stage,
        int cell
    >
    requires (stage < HEAP_STAGES && cell < HEAP_CELLS)
    __device__ __forceinline__
    constexpr auto advanceOffset(const size_t& peer, const size_t& expert, const size_t& token = 0) const {
        return static_cast<size_t>(elementBytes)
            * static_cast<size_t>(tokenDim)
            * (static_cast<size_t>(EC)
                * (static_cast<size_t>(HEAP_CELLS)
                    * (static_cast<size_t>(HEAP_STAGES)
                        * (static_cast<size_t>(peer) * expertSlots + static_cast<size_t>(expert))
                        + static_cast<size_t>(stage))
                    + static_cast<size_t>(cell))
                + static_cast<size_t>(token));
    }

    /**
     * Get a pointer to a specific location in the heap
     *
     * Template Parameters:
     *   stage - Communication stage (0 or 1)
     *   cell  - Cell type (send=0 or receive=1)
     *
     * Parameters:
     *   peer   - Peer PE index
     *   expert - Expert slot index
     *   token  - Token index within the slot (default 0)
     *
     * Returns: Pointer to the heap location
     */
    template<
        int stage,
        int cell
    >
    requires (stage < HEAP_STAGES && cell < HEAP_CELLS)
    __device__ __forceinline__
    cuda::std::byte* advance(const size_t& peer, const size_t& expert, const size_t& token = 0) const {
        return sHeap + advanceOffset<stage, cell>(peer, expert, token);
    }

private:
    const uint expertSlots;
    const uint EC;  // Round to multiple of bM
    const uint tokenDim;
    const uint elementBytes;
};

} // namespace flashmoe

#endif // FLASHMOE_HIP_HEAP_CUH
