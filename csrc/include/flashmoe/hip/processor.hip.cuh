/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 *
 * HIP Port - GEMM Processor with rocSHMEM Put Signals
 * Target: AMD MI450 (gfx1250)
 */

#ifndef FLASHMOE_HIP_PROCESSOR_CUH
#define FLASHMOE_HIP_PROCESSOR_CUH

#include <hip/hip_runtime.h>

// HIP compatibility layer for CUDA primitives
#include "flashmoe/hip/cuda_compat.hpp"
#include "flashmoe/hip/constants.hpp"
#include "flashmoe/hip/nvshmem_compat.hpp"

// Infrastructure includes
#include "flashmoe/hip/combine.hip.cuh"
#include "flashmoe/hip/packed.hip.cuh"
#include "flashmoe/hip/signal.hip.cuh"
#include "flashmoe/hip/task.hip.cuh"
#include "flashmoe/hip/tile.hip.cuh"
#include "flashmoe/hip/wmma_tdm.hip.cuh"
#include "flashmoe/hip/activation.hip.cuh"
#include "flashmoe/hip/context.hip.cuh"

namespace flashmoe
{
#ifndef FLASHMOE_HIP_MLPMATMULTYPE_DEFINED
#define FLASHMOE_HIP_MLPMATMULTYPE_DEFINED
  enum class MLPMatmulType {
    gated,
    vanilla
  };
#endif
}

namespace flashmoe::processor
{

// ============================================================================
// Fused GEMM, Epilogue and Data Transfer (HIP port)
//
// Replaces cuBLASDx-based implementation with shared-memory tiled GEMM.
// Uses thread-level parallelism for correctness; for high throughput,
// the hipBLASLt path (tile_gemm_dispatch.hip.cuh) should be used.
// ============================================================================

template <
  typename TileGEMM,
  typename ActivationFn,
  typename Element,
  typename ElementC
>
__forceinline__ __device__
void fGET(void* __restrict__ const& workspace,
          const Element* __restrict__ const& a,
          const Element* __restrict__ const& b,
          ElementC* __restrict__ const& c,
          const ElementC* __restrict__ const& bias,
          const int& M, const int& N, const int& K, const int& tileIdx) {

  constexpr int bM = hip_tensor::get<0>(typename TileGEMM::TileShape{});
  constexpr int bN = hip_tensor::get<1>(typename TileGEMM::TileShape{});
  constexpr int bK = hip_tensor::get<2>(typename TileGEMM::TileShape{});
  using AccumType = typename TileGEMM::AccumType;
  constexpr int nThreads = TileGEMM::Threads::value;

  const int tilesM = M / bM;
  const int tilesN = N / bN;
  const auto tileCoord = tile::idx2Coord(tilesM, tilesN, tileIdx);
  const int tileRow = hip_tensor::get<0>(tileCoord);
  const int tileCol = hip_tensor::get<1>(tileCoord);

  constexpr bool useWmmaProcessor =
    (TileGEMM::TileArch::value >= 1200 && TileGEMM::TileArch::value < 1300) &&
    (sizeof(Element) == 2) &&
    cuda::std::is_same_v<AccumType, float>;

  if constexpr (useWmmaProcessor) {
    if ((K % TileGEMM::WMMA_K) == 0 && (M % bM) == 0 && (N % bN) == 0) {
      AccumType* sC = static_cast<AccumType*>(workspace);
      wmma_tdm::computeTileRowMajorB<TileGEMM, Element, AccumType>(a, b, sC, M, N, K, tileIdx);

      constexpr ActivationFn act{};
      constexpr Converter<AccumType, ElementC> loadConv{};
      constexpr Converter<ElementC, AccumType> storeConv{};

      ElementC* gC = c + static_cast<size_t>(tileRow) * bM * N + tileCol * bN;
      for (int i = threadIdx.x; i < bM * bN; i += nThreads) {
        int n = i % bN;
        AccumType val = sC[i] + loadConv(bias[tileCol * bN + n]);
        gC[(i / bN) * N + n] = storeConv(act(val));
      }
      return;
    }
  }

  AccumType* sC = static_cast<AccumType*>(workspace);
  Element*   sA = reinterpret_cast<Element*>(sC + bM * bN);
  Element*   sB = sA + bM * bK;

  for (int i = threadIdx.x; i < bM * bN; i += nThreads)
    sC[i] = AccumType(0);
  __syncthreads();

  for (int k = 0; k < K; k += bK) {
    const int effK = (k + bK <= K) ? bK : (K - k);
    const Element* gA = a + (size_t)tileRow * bM * K + k;
    const Element* gB = b + (size_t)k * N + tileCol * bN;

    for (int i = threadIdx.x; i < bM * effK; i += nThreads) {
      int r = i / effK, col = i % effK;
      sA[r * bK + col] = gA[r * K + col];
    }
    for (int i = threadIdx.x; i < effK * bN; i += nThreads) {
      int r = i / bN, col = i % bN;
      sB[r * bN + col] = gB[r * N + col];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < bM * bN; i += nThreads) {
      int m = i / bN, n = i % bN;
      AccumType sum = AccumType(0);
      for (int kk = 0; kk < effK; ++kk)
        sum += AccumType(float(sA[m * bK + kk])) * AccumType(float(sB[kk * bN + n]));
      sC[i] += sum;
    }
    __syncthreads();
  }

  constexpr ActivationFn act{};
  constexpr Converter<AccumType, ElementC> loadConv{};
  constexpr Converter<ElementC, AccumType> storeConv{};

  ElementC* gC = c + (size_t)tileRow * bM * N + tileCol * bN;
  for (int i = threadIdx.x; i < bM * bN; i += nThreads) {
    int n = i % bN;
    AccumType val = sC[i] + loadConv(bias[tileCol * bN + n]);
    gC[(i / bN) * N + n] = storeConv(act(val));
  }
}

template <
  typename TileGEMM,
  typename ActivationFn,
  typename Element,
  typename ElementC
>
__forceinline__ __device__
void fGET_gated(cuda::std::byte* __restrict__ const& workspace,
          const Element* __restrict__ const& a,
          const Element* __restrict__ const& b,
          const Element* __restrict__ const& bV,
          ElementC* __restrict__ const& c,
          const ElementC* __restrict__ const& bias,
          const ElementC* __restrict__ const& biasV,
          const typename TileGEMM::AccumType& swishAlpha,
          const typename TileGEMM::AccumType& swishBeta,
          const int& M, const int& N, const int& K, const int& tileIdx) {

  constexpr int bM = hip_tensor::get<0>(typename TileGEMM::TileShape{});
  constexpr int bN = hip_tensor::get<1>(typename TileGEMM::TileShape{});
  constexpr int bK = hip_tensor::get<2>(typename TileGEMM::TileShape{});
  using AccumType = typename TileGEMM::AccumType;
  constexpr int nThreads = TileGEMM::Threads::value;

  const int tilesM = M / bM;
  const int tilesN = N / bN;
  const auto tileCoord = tile::idx2Coord(tilesM, tilesN, tileIdx);
  const int tileRow = hip_tensor::get<0>(tileCoord);
  const int tileCol = hip_tensor::get<1>(tileCoord);

  auto roundUp = [](auto val, auto align) { return ((val + align - 1) / align) * align; };

  constexpr ActivationFn act{};
  constexpr Converter<AccumType, ElementC> loadConv{};
  constexpr Converter<ElementC, AccumType> storeConv{};

  constexpr bool useWmmaProcessor =
    (TileGEMM::TileArch::value >= 1200 && TileGEMM::TileArch::value < 1300) &&
    (sizeof(Element) == 2) &&
    cuda::std::is_same_v<AccumType, float>;

  AccumType* sC = reinterpret_cast<AccumType*>(workspace);
  Element*   sA = reinterpret_cast<Element*>(sC + bM * bN);
  Element*   sBuf = sA + bM * bK;
  AccumType* sGate = nullptr;
  if constexpr (useWmmaProcessor) {
    sGate = reinterpret_cast<AccumType*>(
        reinterpret_cast<char*>(workspace) +
        wmma_tdm::wmmaGateOffsetBytes<TileGEMM, Element, AccumType>());
  }
  else {
    sGate = reinterpret_cast<AccumType*>(
        reinterpret_cast<char*>(workspace) +
        roundUp(static_cast<int>(sizeof(AccumType) * bM * bN + sizeof(Element) * (bM * bK + bK * bN)),
                TileGEMM::GeneralAlignment::value));
  }

  if constexpr (useWmmaProcessor) {
    if ((K % TileGEMM::WMMA_K) == 0 && (M % bM) == 0 && (N % bN) == 0) {
      wmma_tdm::computeTileRowMajorB<TileGEMM, Element, AccumType>(a, b, sC, M, N, K, tileIdx);
      for (int i = threadIdx.x; i < bM * bN; i += nThreads) {
        int n = i % bN;
        AccumType g = (sC[i] + loadConv(bias[tileCol * bN + n])) * swishBeta;
        sGate[i] = swishAlpha * act(g);
      }
      __syncthreads();

      wmma_tdm::computeTileRowMajorB<TileGEMM, Element, AccumType>(a, bV, sC, M, N, K, tileIdx);
      ElementC* gC = c + static_cast<size_t>(tileRow) * bM * N + tileCol * bN;
      for (int i = threadIdx.x; i < bM * bN; i += nThreads) {
        int n = i % bN;
        AccumType v = sC[i] + loadConv(biasV[tileCol * bN + n]);
        gC[(i / bN) * N + n] = storeConv(v * sGate[i]);
      }
      return;
    }
  }

  auto doGemm = [&](const Element* __restrict__ bPtr, AccumType* __restrict__ dest) {
    for (int i = threadIdx.x; i < bM * bN; i += nThreads)
      dest[i] = AccumType(0);
    __syncthreads();

    for (int k = 0; k < K; k += bK) {
      const int effK = (k + bK <= K) ? bK : (K - k);
      const Element* gA = a + (size_t)tileRow * bM * K + k;
      const Element* gB = bPtr + (size_t)k * N + tileCol * bN;

      for (int i = threadIdx.x; i < bM * effK; i += nThreads) {
        int r = i / effK, col = i % effK;
        sA[r * bK + col] = gA[r * K + col];
      }
      for (int i = threadIdx.x; i < effK * bN; i += nThreads) {
        int r = i / bN, col = i % bN;
        sBuf[r * bN + col] = gB[r * N + col];
      }
      __syncthreads();

      for (int i = threadIdx.x; i < bM * bN; i += nThreads) {
        int m = i / bN, n = i % bN;
        AccumType sum = AccumType(0);
        for (int kk = 0; kk < effK; ++kk)
          sum += AccumType(float(sA[m * bK + kk])) * AccumType(float(sBuf[kk * bN + n]));
        dest[i] += sum;
      }
      __syncthreads();
    }
  };

  // gate = activation(swishAlpha * swishBeta * (A @ B + bias))
  doGemm(b, sC);
  for (int i = threadIdx.x; i < bM * bN; i += nThreads) {
    int n = i % bN;
    AccumType g = (sC[i] + loadConv(bias[tileCol * bN + n])) * swishBeta;
    sGate[i] = swishAlpha * act(g);
  }
  __syncthreads();

  // value = A @ BV + biasV
  doGemm(bV, sC);
  for (int i = threadIdx.x; i < bM * bN; i += nThreads) {
    int n = i % bN;
    sC[i] = sC[i] + loadConv(biasV[tileCol * bN + n]);
  }
  __syncthreads();

  // output = gate * value → global
  ElementC* gC = c + (size_t)tileRow * bM * N + tileCol * bN;
  for (int i = threadIdx.x; i < bM * bN; i += nThreads) {
    gC[(i / bN) * N + (i % bN)] = storeConv(sC[i] * sGate[i]);
  }
}

// ============================================================================
// Processor Arguments Structure
// ============================================================================

struct ProcessorArgs {
  unsigned int* const sQ = nullptr;
  uint64_t* const pDB = nullptr;
  unsigned int* const tQH = nullptr;
  Task* const tQ = nullptr;
  Task* const ptQ = nullptr;
  unsigned int* const tQS = nullptr;

  ProcessorArgs() = default;

  __device__ __forceinline__
  ProcessorArgs(unsigned int* const& sq,
                TQSignal* const& pdb,
                unsigned int* const& tqh,
                Task* const& tq,
                Task* const& ptq,
                unsigned int* const& tqs) :
    sQ(sq), pDB(reinterpret_cast<uint64_t*>(pdb)), tQH(tqh), tQ(tq), ptQ(ptq), tQS(tqs) {
  }
};

// ============================================================================
// Notify Next Stage
// ============================================================================

/**
 * @brief Notify scheduler that tasks are ready for next stage (GEMM1)
 */
template <
  PeerConnectivity p,
  int threads
>
__device__ __forceinline__
void notifyNext(void* __restrict__ const& workspace, const Task& task, Task* __restrict__ const& tQ,
                const uint& tasks, const uint& flagColStride,
                const uint& offset, uint* __restrict__ const& tQH) {
  auto ingredients = task.ingredients;
  ingredients.taskType = TaskType::GEMM1;

  static_assert(sizeof(Task) % sizeof(uint4) == 0 && alignof(Task) % sizeof(uint4) == 0);
  static_assert(cuda::std::is_trivially_copyable_v<Task>);
  using TVT = hip_tensor::AlignedArray<uint4, sizeof(Task) / sizeof(uint4), alignof(Task)>;
  auto* __restrict__ sTQ = static_cast<TVT*>(workspace);
  for (int i = threadIdx.x; i < tasks; i += threads) {
    const auto tvv = cuda::std::bit_cast<TVT>(
      Task{
        ingredients, task.cData[0], task.cData, task.rcData,
        task.flags + (p == PeerConnectivity::remote ? 0 : i * flagColStride),
        task.syncIdx, offset + i
    });
    sTQ[i] = tvv;
  }
  __syncthreads();

  // shared memory to global memory
  constexpr int nRows = sizeof(Task) / sizeof(uint4);
  const uint numElems = tasks * nRows;

  // Project TaskQ as [tasks, nRows] matrix of 128-bit elements
  auto gT = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<uint4*>(tQ)),
                              cute::make_layout(cute::make_shape(tasks, nRows), cute::LayoutRight{}));
  auto sT = cute::make_tensor(cute::make_smem_ptr(static_cast<uint4*>(workspace)),
                              cute::make_layout(cute::make_shape(tasks, nRows), cute::LayoutRight{}));

  for (int i = threadIdx.x; i < numElems; i += threads) {
    const auto rowIdx = i / nRows;
    const auto colIdx = i % nRows;
    gT(rowIdx, colIdx) = sT(rowIdx, colIdx);
  }
  __syncthreads();

  if (!threadIdx.x) {
    cuda::atomic_ref<uint, cuda::thread_scope_device> tqh{*(tQH + task.syncIdx)};
    // Notify scheduler
    cuda::std::ignore = tqh.fetch_add(tasks, cuda::memory_order_release);
  }
}

// ============================================================================
// Main Processor Loop
// ============================================================================

/**
 * @brief Main processor loop for MoE computation with rocSHMEM RDMA
 *
 * Handles:
 * - GEMM0: Input projection (tokens × up_weight) - vanilla or gated MLP
 * - GEMM1: Output projection (intermediate × down_weight)
 * - Combine: Aggregate results back to token positions
 *
 * Key HIP adaptations:
 * - Uses MI450/gfx1250 wave32 execution
 * - Uses HIP __shfl via cuda_compat.hpp wrapper
 * - Uses rocSHMEM via nvshmem_compat.hpp for RDMA
 *
 * Template Parameters:
 * - mt: MLPMatmulType - vanilla or gated MLP pattern
 */
template <
  MLPMatmulType mt,
  Topology topo,
  int threads,
  CombineMode combineMode,
  typename TileGEMM0,
  typename TileGEMM1,
  typename Activation,
  typename Element,
  typename PBM
>
__device__ __forceinline__
void start(cuda::std::byte* __restrict__ const& workspace,
           const int& S,      // sequence length
           const int& H,      // token hidden dimension
           const int& I,      // FFN intermediate size
           const uint& roundEC,
           const uint& flagColStride, // ceil(EC / bM) * E
           const uint& tilesN0,
           const uint& tilesN1,
           const Element* __restrict__ const& expertUpWeights,
           const Element* __restrict__ const& expertUpVWeights,  // [num_local_experts, H, I] for gated
           const Element* __restrict__ const& biasUp,
           const Element* __restrict__ const& biasUpV,           // [num_local_experts, I] for gated
           const typename TileGEMM0::AccumType& swishAlpha,
           const typename TileGEMM0::AccumType& swishBeta,
           const Element* __restrict__ const& expertDownWeights,
           const Element* __restrict__ const& biasDown,
           const TPS* __restrict__ const& tokenIndices,
           Element* __restrict__ const& moeOutput,
           PBM& producerBitMap,
           const uint8_t& stateNumber,
           const Heap& symHeap, const ProcessorArgs& pA) {

  static_assert(sizeof(Task) % sizeof(uint4) == 0 && alignof(Task) % alignof(uint4) == 0);
  static_assert(cuda::std::is_trivially_copyable_v<Task>);

  __shared__ __align__(alignof(Task)) char currentTaskBuf[sizeof(Task)];
  auto& currentTask = *reinterpret_cast<Task*>(currentTaskBuf);
  __shared__ uint globalInterrupt;
  __shared__ uint enqueue;

  TQSignal tqs{0U, 0U};
  static_assert(sizeof(TQSignal) == sizeof(uint64_t) && alignof(TQSignal) == alignof(uint64_t));
  static_assert(sizeof(SignalPayload<PacketStage::last>) == sizeof(uint64_t) &&
    alignof(SignalPayload<PacketStage::last>) == alignof(uint64_t));

  if (!threadIdx.x) {
    globalInterrupt = 0U;
    enqueue = 0U;
  }
  __syncthreads();

  using SQT = cuda::std::underlying_type_t<flashmoe::ReadySignal>;
  constexpr auto taskWidth = sizeof(Task) / sizeof(uint4);
  const size_t expertWeightSize = static_cast<size_t>(I) * H;

  static_assert(taskWidth > 0 && taskWidth < flashmoe::WARP_SIZE);

  while (!tqs.interrupt) {
    if (threadIdx.x / flashmoe::WARP_SIZE == 0) {
      if (threadIdx.x == 0) {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> doorbell{*pA.pDB};

        auto payload = cuda::std::bit_cast<TQSignal>(doorbell.load(cuda::memory_order_acquire));
        while (payload.signal == tqs.signal && payload.interrupt == 0) {
          payload = cuda::std::bit_cast<TQSignal>(doorbell.load(cuda::memory_order_acquire));
        }

        if (payload.interrupt) {
          // Clear mailbox for subsequent epoch
          constexpr auto TQSZero = cuda::std::bit_cast<uint64_t>(TQSignal{0, 0});
          doorbell.store(TQSZero, cuda::memory_order_relaxed);
        }
        else {
          cuda::atomic_ref<SQT, cuda::thread_scope_device> sqd{*pA.sQ};
          // Eagerly indicate readiness for next task
          sqd.store(flashmoe::ready, cuda::memory_order_relaxed);
        }
        globalInterrupt = payload.interrupt;
        tqs = payload;
      }

      // Synchronize within wavefront
      // HIP: __syncwarp() equivalent is wave barrier
#if defined(__HIP_PLATFORM_AMD__)
      __builtin_amdgcn_wave_barrier();
#else
      __syncwarp();
#endif

      auto payload = cuda::std::bit_cast<uint64_t>(tqs);
      // Broadcast from thread 0 to other threads in wavefront
      // Note: __shfl_sync is wrapped in cuda_compat.hpp to call HIP __shfl
      payload = __shfl_sync(0xffffffffffffffff, payload, 0);
      tqs = cuda::std::bit_cast<TQSignal>(payload);

      const auto* __restrict__ gtQ = pA.tQ + tqs.decodeSig();
      if (!tqs.interrupt && threadIdx.x < taskWidth) {
        reinterpret_cast<uint4*>(&currentTask)[threadIdx.x] = reinterpret_cast<const uint4*>(gtQ)[threadIdx.x];
      }
    }
    __syncthreads();

    tqs.interrupt = globalInterrupt;

    if (!tqs.interrupt) {
      // Copy task from shared to registers
      switch (const auto task = currentTask; task.getTaskType()) {
      default:
        break;
      case TaskType::GEMM0: {
        const auto* aP = reinterpret_cast<const Element*>(task.aData);
        const auto* bP = expertUpWeights + expertWeightSize * task.localExpertIdx();
        auto* __restrict__ cP = reinterpret_cast<Element*>(task.cData[0]);
        const auto* __restrict__ biasP = biasUp + static_cast<size_t>(I) * task.localExpertIdx();

#ifndef FLASHMOE_SKIP_GEMM
        if constexpr (mt == MLPMatmulType::vanilla) {
          fGET<TileGEMM0, Activation>(workspace, aP, bP, cP, biasP, task.M(), I, H, task.tileIdx);
        }
        else {
          const auto* bPv = expertUpVWeights + expertWeightSize * task.localExpertIdx();
          const auto* __restrict__ biasPv = biasUpV + static_cast<size_t>(I) * task.localExpertIdx();
          fGET_gated<TileGEMM0, Activation>(workspace, aP, bP, bPv, cP, biasP, biasPv,
            swishAlpha, swishBeta, task.M(), I, H, task.tileIdx);
        }
#endif
        __syncthreads();

        if (!threadIdx.x) {
          cuda::atomic_ref<uint, cuda::thread_scope_device> tileSync{*(pA.tQS + task.syncIdx)};
          const auto isLast = tileSync.fetch_add(1, cuda::memory_order_acq_rel) + 1 == tilesN0;
          enqueue = isLast;
          if (isLast && (topo == Topology::NVLINK_ONLY || !task.isPeerRemote())) {
            // Clear counter for next epoch
            tileSync.store(0, cuda::memory_order_relaxed);
          }
        }
        __syncthreads();

        if (enqueue) {
          const auto offset = tilesN1 * (task.tileIdx / tilesN0);
          auto* __restrict__ tQ = pA.ptQ + (task.syncIdx * tilesN1);
          if (topo == Topology::MIXED && task.isPeerRemote()) {
            notifyNext<PeerConnectivity::remote, threads>(workspace, task, tQ, tilesN1, flagColStride,
              offset, pA.tQH);
          }
          else {
            notifyNext<PeerConnectivity::p2p, threads>(workspace, task, tQ, tilesN1, flagColStride,
              offset, pA.tQH);
          }
        }
        __syncthreads();
      }
      break;

      case TaskType::GEMM1: {
        const auto* aP = reinterpret_cast<const Element*>(task.aData);
        const auto* bP = expertDownWeights + expertWeightSize * task.localExpertIdx();
        auto* __restrict__ cP = reinterpret_cast<Element*>(task.cData[1]);
        const auto* __restrict__ biasP = biasDown + static_cast<size_t>(H) * task.localExpertIdx();

#ifndef FLASHMOE_SKIP_GEMM
        fGET<TileGEMM1, Identity<typename TileGEMM1::AccumType>>(workspace, aP, bP, cP, biasP, task.M(), H, I, task.tileIdx);
#endif
        __syncthreads();

        if (!threadIdx.x) {
          constexpr int bM = cute::get<0>(typename TileGEMM1::TileShape{});
          const auto mCoord = task.tileIdx / tilesN1;
          const auto nCoord = task.tileIdx % tilesN1;

          if (topo == Topology::MIXED && task.isPeerRemote()) {
            // Remote: check if we need to do the RDMA transfer
            cuda::atomic_ref<uint, cuda::thread_scope_device> tileSync{*(pA.tQS + task.syncIdx)};
            const auto doTransfer = tileSync.fetch_add(1, cuda::memory_order_acq_rel) + 1 == (tilesN0 + tilesN1);

            if (doTransfer) {
              // Read and flip the current producer bit
              const auto prevBit = producerBitMap(task.localPeerIdx(), task.localExpertIdx(), mCoord, 0);
              const auto producerBit = static_cast<uint8_t>(prevBit == 0 ? 1 : 0);

              // Pack payload into signal word
              const auto flagSignal = SignalPayload<PacketStage::last>{
                mCoord,
                task.tileSize(),
                producerBit,
                stateNumber,
              };
              const auto sigPayload = cuda::std::bit_cast<uint64_t>(flagSignal);

              // Flip the bit
              producerBitMap(task.localPeerIdx(), task.localExpertIdx(), mCoord, 0) = producerBit;

              // Clear counter for next epoch
              tileSync.store(0, cuda::memory_order_relaxed);

              // RDMA transfer via rocSHMEM
              const auto symOffset = static_cast<size_t>(bM) * mCoord * H * sizeof(Element);
              nvshmem_putmem_signal_nbi(task.rcData + symOffset,
                                        task.cData[1] + symOffset,
                                        static_cast<size_t>(task.tileSize() * H) * sizeof(Element),
                                        task.flags,
                                        sigPayload,
                                        NVSHMEM_SIGNAL_SET,
                                        task.pe());
            }
          }
          else {
            // Local/P2P path - direct signal via atomics
            const auto prevBit = producerBitMap(task.localPeerIdx(), task.localExpertIdx(), mCoord, nCoord);
            const auto producerBit = static_cast<uint8_t>(prevBit == 0 ? 1 : 0);

            const auto flagSignal = SignalPayload<PacketStage::last>{
              mCoord,
              task.tileSize(),
              producerBit,
              stateNumber,
            };
            const auto sigPayload = cuda::std::bit_cast<uint64_t>(flagSignal);

            // Flip the bit
            producerBitMap(task.localPeerIdx(), task.localExpertIdx(), mCoord, nCoord) = producerBit;

            // Signal completion directly (already transferred via NVLink/P2P)
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> remoteFlags{*task.flags};
            remoteFlags.store(sigPayload, cuda::memory_order_release);
          }
        }
        __syncthreads();
      }
      break;

      case TaskType::combine: {
        const auto tileCoord = cute::make_coord(cute::_0{}, task.combineTileIdx());
        const auto* __restrict__ tokens = reinterpret_cast<Element*>(symHeap.advance<1, 1>(task.epRank(),
          task.localExpertIdx(), static_cast<int>(task.tokenBatchStart())));

        using Tiler = typename TileGEMM1::TileShape;
        constexpr int bM = cute::get<0>(Tiler{});
        constexpr int bN = cute::get<1>(Tiler{});
        constexpr int Arch = TileGEMM1::TileArch::value;

        const auto* __restrict__ tokenIds = tokenIndices + (task.expertIdx() * roundEC + task.tokenBatchStart());

        combine<bM, bN, Arch, threads, combineMode>(S, H, workspace, tokenIds, moeOutput, tokens, task.tileSize(),
          tileCoord);
      }
      break;
      }
    }
  }
}

} // namespace flashmoe::processor

#endif // FLASHMOE_HIP_PROCESSOR_CUH
