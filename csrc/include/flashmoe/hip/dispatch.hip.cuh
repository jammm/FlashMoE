/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 *
 * HIP Port - Token Dispatch with RDMA via rocSHMEM
 * Target: AMD Instinct MI450/MI450
 */

#ifndef FLASHMOE_HIP_DISPATCH_CUH
#define FLASHMOE_HIP_DISPATCH_CUH

#include <hip/hip_runtime.h>

// HIP compatibility layer for CUDA primitives
#include "flashmoe/hip/cuda_compat.hpp"
#include "flashmoe/hip/constants.hpp"
#include "flashmoe/hip/nvshmem_compat.hpp"

#include "flashmoe/hip/tensor.hpp"
#include "flashmoe/hip/heap.hip.cuh"
#include "flashmoe/hip/math.hip.cuh"
#include "flashmoe/hip/packed.hip.cuh"
#include "flashmoe/hip/signal.hip.cuh"
#include "flashmoe/hip/structures.hip.cuh"
#include "flashmoe/hip/context.hip.cuh"
#include "flashmoe/hip/combine.hip.cuh"

namespace flashmoe
{

// ============================================================================
// HIP-Compatible Memory Load/Store Operations
// ============================================================================

namespace hip_mem {

// Non-allocating load - bypasses L1 cache for streaming access patterns
// On AMD, we use __builtin_nontemporal_load for similar behavior
template<typename T>
__device__ __forceinline__
T load_nontemporal(const T* ptr) {
    if constexpr (std::is_arithmetic_v<T> || std::is_pointer_v<T>) {
#if defined(__HIP_PLATFORM_AMD__)
        return __builtin_nontemporal_load(ptr);
#endif
    }
    return *ptr;
}

template<typename T>
__device__ __forceinline__
void store_nontemporal(T* ptr, T value) {
    if constexpr (std::is_arithmetic_v<T> || std::is_pointer_v<T>) {
#if defined(__HIP_PLATFORM_AMD__)
        __builtin_nontemporal_store(value, ptr);
        return;
#endif
    }
    *ptr = value;
}

// Vectorized non-temporal load for uint4 (128-bit)
__device__ __forceinline__
uint4 load_nontemporal_u128(const uint4* ptr) {
#if defined(__HIP_PLATFORM_AMD__)
    // Use inline assembly for vectorized non-temporal load on AMD
    uint4 result;
    asm volatile(
        "global_load_dwordx4 %0, %1, off glc slc\n"
        : "=v"(result)
        : "v"(ptr)
        : "memory"
    );
    return result;
#else
    return *ptr;
#endif
}

// Vectorized non-temporal store for uint4 (128-bit)
__device__ __forceinline__
void store_nontemporal_u128(uint4* ptr, uint4 value) {
#if defined(__HIP_PLATFORM_AMD__)
    // Use inline assembly for vectorized non-temporal store on AMD
    asm volatile(
        "global_store_dwordx4 %0, %1, off glc slc\n"
        :
        : "v"(ptr), "v"(value)
        : "memory"
    );
#else
    *ptr = value;
#endif
}

} // namespace hip_mem

// ============================================================================
// MoE Token Dispatch - HIP Version
// ============================================================================

/**
 * @brief Token dispatch function for MoE with RDMA support via rocSHMEM
 *
 * This function dispatches tokens to appropriate expert PEs using either:
 * - Direct memory access for local/NVLink-connected peers
 * - RDMA via rocSHMEM for remote peers
 *
 * Key differences from CUDA version:
 * - Uses HIP non-temporal loads/stores instead of PTX intrinsics
 * - Uses rocSHMEM via nvshmem_compat.hpp for RDMA operations
 * - Wavefront-sensitive code uses flashmoe::WARP_SIZE instead of a fixed width
 */
template <
  Topology topo,
  int threads, // workgroup size
  int bM,      // tile size M for GEMM0
  int bN,      // tile size N for GEMM0
  int batch = cute::min(bM, 8), // tokens dispatched per CTA iteration
  typename Element
>
__forceinline__ __device__
void dispatch(const int& H, const int& E,
              const Heap& symHeap,
              const int& EC, const int& roundEC,
              const int& epRank, const int& world,
              const int& superBlockSize, const int& blocks,
              const Element* __restrict__ const& tokens,
              uint64_t* __restrict__ signals,
              const int* __restrict__ const& expertCounts,
              const TPS* __restrict__ const& _tokenIds,
              uint* __restrict__ const& dispatchSync,
              const PEL* __restrict__ const& expertLookupInfo,
              cuda::std::byte* __restrict__ const& workspace,
              const uint16_t& stateNumber) {

  // Assumptions:
  // - workspace is in shared memory (LDS on AMD)
  // - ceil(EC, bM) * num_local_experts <= UINT16_MAX
  // - tokens and tokenId are at least 32-byte aligned
  // - superblocks are equally sized

  const auto numSuperBlocks = blocks / superBlockSize;
  const int superBlockIdx = static_cast<int>(blockIdx.x) / superBlockSize;
  if (superBlockIdx >= numSuperBlocks) {
    return;
  }
  const int lBid = blockIdx.x % superBlockSize;
  const bool isLeader = !lBid && !threadIdx.x;

  /// Populate Data Structures in LDS (shared memory)
  size_t offset = 0;
  auto* __restrict__ enL = reinterpret_cast<PEL*>(workspace);
  for (int i = threadIdx.x; i < E; i += threads) {
    enL[i] = expertLookupInfo[i];
  }
  offset += rTCL<PEL>(E);

  auto* __restrict__ seC = reinterpret_cast<int*>(workspace + offset);
  for (int i = threadIdx.x; i < E; i += threads) {
    seC[i] = expertCounts[i];
  }

  // Peer total tokens
  auto* __restrict__ sPTT = seC + E;
  for (int i = threadIdx.x; i < world; i += threads) {
    sPTT[i] = 0;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < E; i += threads) {
    const auto peer = enL[i].peer;
    const auto nTokens = cute::min(seC[i], EC);
    const auto tilesM = cute::ceil_div(nTokens, bM);
    atomicAdd(sPTT + peer, tilesM);
  }
  __syncthreads();

  // Update lookup table
  for (int i = threadIdx.x; i < E; i += threads) {
    auto lInfo = enL[i];
    lInfo.eC = seC[i];
    lInfo.pTTt = static_cast<uint16_t>(sPTT[lInfo.peer]);
    enL[i] = lInfo;
  }
  __syncthreads();

  const auto* __restrict__ expertLookup = enL;
  cute::AlignedArray<uint32_t, batch> rTID{};

  constexpr auto eWidth = ElementWidth<Element, bN, MAX_ACCESS_ALIGNMENT>;
  using VTD = VectorTypeDescriptor<Element, ElementAlignmentForWidth<Element, bN, eWidth>>;
  using VectorElement = typename VTD::VectorType;

  // H % bN == 0 and bN % vectorWidth == 0
  const int vH = H / VTD::VectorWidth::value;
  const auto* __restrict__ vTokens = reinterpret_cast<const VectorElement*>(tokens);

  const auto tokenIds = make_tensor(cute::make_gmem_ptr(_tokenIds),
    cute::make_layout(cute::make_shape(E, roundEC), cute::LayoutRight{}));

  static_assert(batch >= 1 && bM % batch == 0);

  for (int expertIdx = superBlockIdx; expertIdx < E; expertIdx += numSuperBlocks) {
    const auto lI = expertLookup[expertIdx];
    const auto flagOffset = epRank * lI.nLocalExperts + lI.expertLocalIdx;
    const auto routedTokens = cute::min(lI.eC, EC);

    auto* __restrict__ peerHeap = reinterpret_cast<VectorElement*>(
      topo == Topology::MIXED && lI.isRemote ?
        symHeap.advance<0, 0>(lI.peer, lI.expertLocalIdx) :
        lI.remoteSHeap + symHeap.advanceOffset<0, 1>(epRank, lI.expertLocalIdx));

    if (routedTokens) {
      const auto partition = routedTokens / superBlockSize +
        (lBid < routedTokens % superBlockSize);
      const auto trips = partition / batch;

      for (int i = 0; i < trips; ++i) {
        #pragma unroll
        for (int j = 0; j < batch; ++j) {
          // Intentionally redundant copies to avoid barriers within loop
          const auto v = tokenIds(expertIdx, lBid + (j + i * batch) * superBlockSize);
          rTID[j] = v.tokenIdx;
        }

        // Communicate these tokens using non-temporal memory operations
        #pragma unroll
        for (int j = 0; j < batch; ++j) {
          const auto tokenIdx = rTID[j];
          const auto intraIdx = lBid + (j + i * batch) * superBlockSize;
          auto* __restrict__ localPH = peerHeap + intraIdx * vH;
          const auto* __restrict__ aP = vTokens + tokenIdx * vH;

          // Coalesced vectorized copy with non-temporal access
          for (int k = threadIdx.x; k < vH; k += threads) {
            // HIP: Use non-temporal load to bypass L1 cache (streaming pattern)
            const auto v = hip_mem::load_nontemporal(aP + k);
            // HIP: Use non-temporal store for write-through
            hip_mem::store_nontemporal(localPH + k, v);
          }
        }
      }

      // Handle residue tokens
      if (const auto residue = partition - trips * batch; residue) {
        #pragma unroll
        for (int j = 0; j < batch; ++j) {
          if (j < residue) {
            const auto v = tokenIds(expertIdx, lBid + (j + trips * batch) * superBlockSize);
            rTID[j] = v.tokenIdx;
          }
        }
        #pragma unroll
        for (int j = 0; j < batch; ++j) {
          if (j < residue) {
            const auto tokenIdx = rTID[j];
            const auto intraIdx = lBid + (j + trips * batch) * superBlockSize;
            auto* __restrict__ localPH = peerHeap + intraIdx * vH;
            const auto* __restrict__ aP = vTokens + tokenIdx * vH;

            for (int k = threadIdx.x; k < vH; k += threads) {
              const auto v = hip_mem::load_nontemporal(aP + k);
              hip_mem::store_nontemporal(localPH + k, v);
            }
          }
        }
      }

      __syncthreads();

      if (!threadIdx.x) {
        cuda::atomic_ref<uint, cuda::thread_scope_device> dS{*(dispatchSync + expertIdx)};
        if (dS.fetch_add(1, cuda::memory_order_acq_rel) + 1 == superBlockSize) {
          // Last block to complete - finalize transfer
          // Clear counter for next epoch (deferred grid-wide sync to kernel teardown)
          dS.store(0, cuda::memory_order_relaxed);

          // Prepare signal payload
          const auto sigPayload = SignalPayload<PacketStage::initial>{
            routedTokens,
            lI.pTTt,
            stateNumber
          };

          if (topo == Topology::MIXED && lI.isRemote) {
            nvshmem_putmem_signal_nbi(
              symHeap.advance<0, 1>(epRank, lI.expertLocalIdx),
              peerHeap,
              sizeof(VectorElement) * routedTokens * vH,
              signals + flagOffset,
              cuda::std::bit_cast<uint64_t>(sigPayload),
              NVSHMEM_SIGNAL_SET,
              lI.pe);
          }
          else {
            // Local/P2P path - DMA transfer already done, just set signal
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> sf{*(lI.remoteSFlags + flagOffset)};
            sf.store(cuda::std::bit_cast<uint64_t>(sigPayload), cuda::memory_order_release);
          }
        }
      }
    }
    else if (isLeader) {
      // Send noop control message to unblock remote peer (zero tokens)
      const auto sigPayload = SignalPayload<PacketStage::initial>{
        0U,
        lI.pTTt,
        stateNumber
      };

      if (topo == Topology::MIXED && lI.isRemote) {
        nvshmemx_signal_op(signals + flagOffset,
          cuda::std::bit_cast<uint64_t>(sigPayload), NVSHMEM_SIGNAL_SET, lI.pe);
      }
      else {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> sf{*(lI.remoteSFlags + flagOffset)};
        sf.store(cuda::std::bit_cast<uint64_t>(sigPayload), cuda::memory_order_release);
      }
    }
  }
}

} // namespace flashmoe

#endif // FLASHMOE_HIP_DISPATCH_CUH
