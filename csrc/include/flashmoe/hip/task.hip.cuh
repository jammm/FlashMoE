/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 *
 * HIP Port of Task and Ingredients Structures
 * Target: AMD Instinct MI450 (gfx1250) / MI450 (gfx1250)
 *
 * Key changes from CUDA:
 * - cuda::std::byte -> std::byte (C++17)
 * - cuda::std::array -> std::array (C++11)
 * - cuda::std::is_same_v -> std::is_same_v (C++17)
 * - cuda::std::underlying_type_t -> std::underlying_type_t
 */

#ifndef FLASHMOE_HIP_TASK_CUH
#define FLASHMOE_HIP_TASK_CUH

#include <hip/hip_runtime.h>
#include <cstddef>       // std::byte
#include <cstdint>
#include <array>         // std::array
#include <type_traits>   // std::is_same_v, std::underlying_type_t

namespace flashmoe {

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
        unsigned int M = 0;           // GEMM0->number of tokens or Combine->tokenBatchStart
        uint16_t localExpertIdx = 0;
        uint16_t expertIdx = 0;       // global expert index
        uint16_t peerIdx = 0;         // owner of the output
        uint16_t tileSize = 0;        // <= BLOCK_M
        uint16_t stash = 0;           // GEMM0->flagBatchIdx or Combine->tileIdx
        TaskType taskType = TaskType::GEMM0;
        uint8_t isPeerRemote = 0;

        Ingredients() = default;

        // GEMM0 and GEMM1 constructor
        __device__ __forceinline__
        Ingredients(const uint16_t& lei, const uint16_t& pei, const TaskType& tt, const uint8_t& ipr):
            localExpertIdx(lei), peerIdx(pei), taskType(tt), isPeerRemote(ipr) {}

        __device__ __forceinline__
        Ingredients(const unsigned int& m, const uint16_t& lei, const uint16_t& ei,
                    const uint16_t& pei, const uint16_t& ts, const uint16_t& sta,
                    const TaskType& tt, const uint8_t& ipr):
            M(m), localExpertIdx(lei), expertIdx(ei), peerIdx(pei), tileSize(ts), stash(sta),
            taskType(tt), isPeerRemote(ipr) {}
    };

    static_assert(sizeof(Ingredients) == 16);

    /// Task - complete task descriptor with data pointers
    /// 64 bytes total for cache-line alignment on AMD MI450
    struct __align__(16) Task {
        Ingredients ingredients{};
        const std::byte* aData = nullptr;
        std::array<std::byte*, GEMMs> cData = {};
        std::byte* rcData = nullptr;
        uint64_t* flags = nullptr;
        unsigned int syncIdx = 0U;
        unsigned int tileIdx = 0U;

        Task() = default;

        // GEMM0->GEMM1 constructor
        __device__ __forceinline__
        Task(const Ingredients& _ingredients,
             const std::byte* const& _aData,
             const std::array<std::byte*, GEMMs>& _cData,
             std::byte* const& _rcData,
             uint64_t* const& _flags,
             const unsigned int& _syncIdx, const unsigned int& tile):
            ingredients(_ingredients), aData(_aData), cData(_cData), rcData(_rcData), flags(_flags),
            syncIdx(_syncIdx), tileIdx(tile) {}

        // Combine constructor
        __device__ __forceinline__
        explicit Task(const Ingredients& _ingredients):
            ingredients(_ingredients) {}

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
        auto combineTileIdx() const {
            return ingredients.stash;
        }

        __device__ __forceinline__
        auto expertIdx() const {
            return ingredients.expertIdx;
        }

        __device__ __forceinline__
        auto epRank() const {
            return ingredients.peerIdx;
        }
    };

    static_assert(sizeof(Task) == 64);

#endif // FLASHMOE_HIP_TASK_TYPES_DEFINED

} // namespace flashmoe

#endif // FLASHMOE_HIP_TASK_CUH
