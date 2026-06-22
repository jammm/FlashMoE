/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 *
 * HIP Port: Converted from CUDA CUB to hipCUB for AMD GPU support.
 * Target: AMD MI450 (gfx1250)
 */

//
// Created by oja7 on 11/17/24.
// HIP Port: 2026
//

#ifndef SCHEDULER_HIP_CUH
#define SCHEDULER_HIP_CUH

// HIP runtime and hipCUB
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

// C++ standard library (replaces cuda:: headers)
#include <cstdint>
#include <array>
#include <limits>
#include <bit>

// FlashMoE HIP compatibility layer
#include "constants.hpp"
#include "cuda_compat.hpp"
#include "tensor.hpp"

// ============================================================================
// hipCUB-based fast_mod_div implementation
// ============================================================================

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
}
#endif

namespace flashmoe::scheduler
{
  // ============================================================================
  // Wavefront Size Constant
  // ============================================================================

  using flashmoe::WARP_SIZE;

  // ============================================================================
  // Scheduler State Constants
  // ============================================================================

  // From constants.hpp - included for reference
  // constexpr int SCHEDULER_COUNT = WARP_SIZE;
  // constexpr int PROCESSOR_STATE_SIZE = 9;
  // constexpr int MAX_PROCESSORS = SCHEDULER_COUNT * PROCESSOR_STATE_SIZE;
  // constexpr int WORK_SET_SIZE = 4;
  // constexpr int QUEUE_STATE_SIZE = 2;

  // Task queue head ground state
  constexpr unsigned int tQHeadGroundState = 0U;

  // Scheduler queue states
  constexpr unsigned int observed = 0U;
  constexpr unsigned int ready = 1U;

  // ============================================================================
  // hipCUB WarpScan Type Alias
  // ============================================================================

  // rocPRIM's default LOGICAL_WARP_THREADS is the maximum wavefront size for
  // the target family. gfx12 kernels here run wave32, so pin scans to the
  // scheduler cohort instead of relying on the backend default.
  using WarpScan = hipcub::WarpScan<uint, SCHEDULER_COUNT>;

  // ============================================================================
  // Task Queue Signal Structure
  // ============================================================================

  using TQSignal = flashmoe::TQSignal;

  // ============================================================================
  // Task Queue State Structure
  // ============================================================================

  struct TQState {
    uint32_t tasks;
    uint32_t tQTail;
  };

  template<typename T, typename = void>
  struct has_size_method : std::false_type {};
  template<typename T>
  struct has_size_method<T, std::void_t<decltype(std::declval<T>().size())>> : std::true_type {};
  template<typename T>
  concept isRegisterV = has_size_method<T>::value;

  using BitSet = flashmoe::BitSet;

  // ============================================================================
  // Dequeue Type Enumeration
  // ============================================================================

  enum class DQType {
    stride,
    block
  };

  namespace DQ {
    template<DQType dqt = DQType::stride, int nQ = 0>
    __device__ __forceinline__
    uint32_t next(uint32_t qIdx, uint32_t offset) {
      if constexpr (dqt == DQType::stride) {
        return qIdx + offset * nQ;
      } else {
        return qIdx + offset;
      }
    }
  }

  // ============================================================================
  // Utility: Circular Index Calculation
  // ============================================================================

  __device__ __forceinline__
  auto circularIdx(const uint& cursor, const cuda::fast_mod_div<uint>& len) {
    // Use fast modulo division
    return cursor % len;
  }

  // ============================================================================
  // HIP-Compatible Array Type
  // ============================================================================

  // Using hip_tensor::AlignedArray as replacement for cutlass::Array
  template<typename T, int N>
  using Array = hip_tensor::AlignedArray<T, N>;

  // ============================================================================
  // Utility: min function (replacing cute::min)
  // ============================================================================

  template<typename T>
  __device__ __forceinline__
  T hip_min(T a, T b) {
    return (a < b) ? a : b;
  }

  // Unsigned min
  __device__ __forceinline__
  uint umin(uint a, uint b) {
    return (a < b) ? a : b;
  }

  // ============================================================================
  // Schedule Function
  // ============================================================================

  template <
    DQType dqt = DQType::stride,
    int nQ = 0
  >
  __device__ __forceinline__
  void schedule(const cuda::fast_mod_div<uint>& processors_v, const uint& cSetB,
                const uint& canSchedule, const uint& qIdx, uint& lRQIdx,
                const uint& gRQIdx, uint* __restrict__ const& rQ,
                TQSignal* __restrict__ const& pDB) {
    Array<uint, WORK_SET_SIZE> wSet{};
    auto sig = TQSignal{0U, 0U};

    for (uint k = 0; k < cSetB; ++k) {
      #pragma unroll
      for (uint l = 0; l < wSet.size(); ++l) {
        wSet[l] = rQ[circularIdx(gRQIdx + lRQIdx++, processors_v)];
      }
      #pragma unroll
      for (uint l = 0; l < wSet.size(); ++l) {
        // signal processor
        auto* __restrict__ pdbAddr = reinterpret_cast<uint64_t*>(pDB + wSet[l]);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> pdb{*pdbAddr};
        sig.encodeSig(DQ::next<dqt, nQ>(qIdx, k * wSet.size() + l));
        pdb.store(std::bit_cast<uint64_t>(sig), cuda::memory_order_release);
      }
    }

    // Residual scheduling
    const uint residue = canSchedule - cSetB * wSet.size();
    #pragma unroll
    for (uint l = 0; l < wSet.size(); ++l) {
      if (l < residue) {
        wSet[l] = rQ[circularIdx(gRQIdx + lRQIdx++, processors_v)];
      }
    }
    #pragma unroll
    for (uint l = 0; l < wSet.size(); ++l) {
      if (l < residue) {
        auto* __restrict__ pdbAddr = reinterpret_cast<uint64_t*>(pDB + wSet[l]);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> pdb{*pdbAddr};
        sig.encodeSig(DQ::next<dqt, nQ>(qIdx, cSetB * wSet.size() + l));
        pdb.store(std::bit_cast<uint64_t>(sig), cuda::memory_order_release);
      }
    }
  }

  // ============================================================================
  // Scheduler Loop
  // ============================================================================

  template <
    int processor_state_size,
    int subscriberCount,
    int sL,
    int schedulerCount,
    typename TQState
  >
    requires (schedulerCount == SCHEDULER_COUNT && isRegisterV<TQState>)
  __device__ __forceinline__
  void schedulerLoop(TQState& tqState,
                     const uint& processors,
                     const cuda::fast_mod_div<uint>& processors_v,
                     const unsigned int& tilesN1,
                     const unsigned int& tQOffset,
                     const unsigned int& gTbO,
                     uint& lTt, uint& processorTally,
                     uint& gRQIdx, uint& scheduled,
                     unsigned int* __restrict__ const& sQ,
                     uint* __restrict__ const& rQ,
                     TQSignal* __restrict__ const& pDB,
                     const bool& isMedley = false) {
    // hipCUB WarpScan temporary storage
    __shared__ WarpScan::TempStorage wSt;
    uint queueSlot;
    uint taskTally;

    // Synchronize wavefront
    __syncwarp();

    // Aggregate tally across the wavefront using hipCUB WarpScan
    WarpScan(wSt).InclusiveSum(lTt, queueSlot, taskTally);
    queueSlot -= lTt;

    auto prefixTaskSum = 0U;

    while (taskTally) {
      // Find processors if we are not currently aware of any
      {
        while (!processorTally) {
          Array<uint, processor_state_size> sQState{};
          uint lPt = 0U;

          #pragma unroll
          for (int j = 0; j < sQState.size(); ++j) {
            const auto idx = j * schedulerCount + threadIdx.x;
            if (idx < processors) {
              const auto readiness = atomicExch(sQ + (j * schedulerCount + threadIdx.x),
                                                observed) == ready;
              lPt += readiness;
              sQState[j] = readiness;
            }
          }

          uint startIdx;
          __syncwarp();
          WarpScan(wSt).InclusiveSum(lPt, startIdx, processorTally);
          startIdx -= lPt;

          const auto qSIdx = gRQIdx + prefixTaskSum;
          #pragma unroll
          for (uint j = 0; j < sQState.size(); ++j) {
            const auto idx = j * schedulerCount + threadIdx.x;
            if (idx < processors && sQState[j]) {
              rQ[circularIdx((qSIdx + startIdx++), processors_v)] = idx;
            }
          }

          if (processorTally) {
            __syncwarp();
          }
        }
      }

      // schedule tasks
      const auto tasks = hip_min(processorTally, taskTally);
      prefixTaskSum += tasks;
      scheduled += tasks;
      processorTally -= tasks;
      taskTally -= tasks;

      // these will get scheduled now
      if (lTt > 0 && queueSlot < prefixTaskSum) {
        auto tasksToSchedule = umin(lTt, prefixTaskSum - queueSlot);
        lTt -= tasksToSchedule;

        if (isMedley) {
          if constexpr (sL > 0) {
            #pragma unroll
            for (uint j = 0; j < sL; ++j) {
              if (tqState[j].tasks > 0 && tasksToSchedule) {
                const auto canSchedule = hip_min(tasksToSchedule, tqState[j].tasks);
                const auto qIdx = DQ::next<DQType::stride, subscriberCount>(
                  j * schedulerCount + threadIdx.x, tqState[j].tQTail);
                tasksToSchedule -= canSchedule;
                tqState[j].tasks -= canSchedule;
                tqState[j].tQTail += canSchedule;
                const auto cSetB = canSchedule / WORK_SET_SIZE;
                schedule<DQType::stride, subscriberCount>(processors_v, cSetB, canSchedule, qIdx,
                                                          queueSlot, gRQIdx, rQ, pDB);
              }
            }
          }
        }

        #pragma unroll
        for (uint j = sL; j < tqState.size(); ++j) {
          if (tqState[j].tasks && tasksToSchedule) {
            const auto canSchedule = hip_min(tasksToSchedule, tqState[j].tasks);
            const auto qHead = (schedulerCount * (gTbO + (j - sL)) + threadIdx.x) * tilesN1 + tqState[j].tQTail;
            const auto qIdx = tQOffset + qHead;
            tasksToSchedule -= canSchedule;
            tqState[j].tasks -= canSchedule;
            tqState[j].tQTail += canSchedule;
            const auto cSetB = canSchedule / WORK_SET_SIZE;
            schedule<DQType::block>(processors_v, cSetB, canSchedule,
                                    qIdx, queueSlot, gRQIdx, rQ, pDB);
          }
        }
      }
    }

    // clear checkpoints
    #pragma unroll
    for (uint j = sL; j < tqState.size(); ++j) {
      tqState[j].tQTail = 0;
    }

    // Advance global rQ index
    gRQIdx = circularIdx((gRQIdx + prefixTaskSum), processors_v);
  }

  // ============================================================================
  // Schedule Processor Interrupts
  // ============================================================================

  template <int processor_state_size, unsigned int schedulerCount>
  __device__ __forceinline__
  void sPI(unsigned int* __restrict__ const& rQ,
           uint* __restrict__ const& sQ,
           TQSignal* __restrict__ const& pDB,
           uint& gRQIdx,
           uint* __restrict__ const& scratch, // pre-filled with 1
           const uint& processors,
           const cuda::fast_mod_div<uint>& processors_v,
           const uint& processorTally) {
    __shared__ WarpScan::TempStorage wSt;
    Array<uint, processor_state_size> sQState{};

    // Read through the ready queue first
    constexpr auto sig = TQSignal{0U, 1U}; // set interrupt to 1
    const auto tS = processorTally / schedulerCount + (threadIdx.x < processorTally % schedulerCount);
    const auto gRO = gRQIdx + (threadIdx.x * (processorTally / schedulerCount) +
      hip_min(threadIdx.x, processorTally % schedulerCount));

    gRQIdx = circularIdx(gRO, processors_v);

    #pragma unroll
    for (uint i = 0; i < sQState.size(); ++i) {
      if (i < tS) {
        sQState[i] = rQ[circularIdx(gRQIdx + i, processors_v)];
      }
    }

    #pragma unroll
    for (uint i = 0; i < sQState.size(); ++i) {
      if (i < tS) {
        // notify interrupts
        const auto pid = sQState[i];
        auto* __restrict__ db_p = reinterpret_cast<uint64_t*>(pDB + pid);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> pdb{*db_p};
        pdb.store(std::bit_cast<uint64_t>(sig), cuda::memory_order_release);
        scratch[pid] = 0U;
      }
    }

    __syncwarp();

    // Consolidate findings and populate the ready queue
    uint uI = 0U;

    #pragma unroll
    for (uint i = 0; i < sQState.size(); ++i) {
      const auto idx = i * schedulerCount + threadIdx.x;
      if (idx < processors) {
        sQState[i] = scratch[idx];
      }
    }

    #pragma unroll
    for (uint i = 0; i < sQState.size(); ++i) {
      if ((i * schedulerCount + threadIdx.x) < processors) {
        uI += sQState[i];
      }
    }

    uint startIdx;
    uint pending;
    WarpScan(wSt).InclusiveSum(uI, startIdx, pending);
    startIdx -= uI;

    // enqueue all pending processes we discovered into the rQ
    #pragma unroll
    for (uint i = 0; i < sQState.size(); ++i) {
      const auto idx = i * schedulerCount + threadIdx.x;
      if (idx < processors && sQState[i]) {
        rQ[startIdx++] = idx;
      }
    }

    __syncwarp();

    auto remaining = pending / schedulerCount + (threadIdx.x < pending % schedulerCount);
    std::array<uint, processor_state_size> pids{};

    // read from rQ to registers
    #pragma unroll
    for (uint i = 0; i < sQState.size(); ++i) {
      const auto idx = i * schedulerCount + threadIdx.x;
      if (idx < pending) {
        sQState[i] = 1;
        pids[i] = rQ[idx];
      }
      else {
        sQState[i] = 0;
      }
    }

    while (remaining) {
      #pragma unroll
      for (uint j = 0; j < sQState.size(); ++j) {
        if (sQState[j]) {
          const auto pid = pids[j];
          const auto isReady = atomicExch(sQ + pid, observed) == ready;
          sQState[j] = !isReady;
          if (isReady) {
            remaining -= 1;
            auto* __restrict__ db_p = reinterpret_cast<uint64_t*>(pDB + pid);
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device> pdb{*db_p};
            pdb.store(std::bit_cast<uint64_t>(sig), cuda::memory_order_release);
          }
        }
      }
    }
  }

  // ============================================================================
  // Scheduler Start Function
  // ============================================================================

  template <
    unsigned int subscribers,
    int processor_state_size = PROCESSOR_STATE_SIZE
  >
  __device__ __forceinline__
  void start(uint* __restrict__ const& interruptScratch,
             BitSet* __restrict__ const& bitSet,
             const uint& processors,
             const cuda::fast_mod_div<uint>& processors_v,
             const unsigned int& tilesN1,
             const unsigned int& sO,
             const unsigned int& gtQCL,
             unsigned int* __restrict__ const& sInterrupts,
             unsigned int* __restrict__ const& tQHeads, // shared
             unsigned int* __restrict__ const& gtQHeads, // global
             unsigned int* __restrict__ const& taskBound, // shared
             unsigned int* __restrict__ const& rQ, // shared
             unsigned int* __restrict__ const& sQ, // global
             TQSignal* __restrict__ const& pDB) {
    static_assert(processor_state_size <= PROCESSOR_STATE_SIZE);
    uint scheduled = 0U;
    constexpr auto schedulerCount = SCHEDULER_COUNT;
    static_assert(subscribers % schedulerCount == 0);
    constexpr auto sL = subscribers / schedulerCount;

    constexpr auto dQL = QUEUE_STATE_SIZE;
    constexpr auto bSw = sizeof(uint) * 8U;
    static_assert(dQL <= bSw);

    Array<TQState, dQL + sL> tqState{};
    tqState.fill({0U, 0U});
    const uint dT = gtQCL / (schedulerCount * dQL);

    uint gRQIdx = 0U;
    uint processorTally = processors;
    uint tTB = 0;

    if (!threadIdx.x) {
      cuda::atomic_ref<unsigned int, cuda::thread_scope_block> tb{*taskBound};
      tTB = tb.load(cuda::memory_order_relaxed);
    }
    tTB = __shfl_sync(FLASHMOE_ACTIVE_MASK(), tTB, 0);

    while (scheduled < tTB) {
      uint lTt = 0U;

      #pragma unroll
      for (uint i = 0; i < sL; ++i) {
        auto* __restrict__ tqHp = tQHeads + (i * schedulerCount + threadIdx.x);
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> tqh{*tqHp};
        const auto tasks = tqh.load(cuda::memory_order_acquire) - tqState[i].tQTail;
        tqState[i].tasks = tasks;
        lTt += tasks;
      }

      if (dT > 0) {
        auto sBS = bitSet[threadIdx.x];

        #pragma unroll
        for (uint j = sL; j < tqState.size(); ++j) {
          const auto pJ = j - sL;
          if (const auto isVisited = sBS.get(pJ % bSw); !isVisited) {
            const auto qIdx = schedulerCount * pJ + threadIdx.x;
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> gtqH{*(gtQHeads + qIdx)};
            const auto tasks = gtqH.load(cuda::memory_order_acquire);
            if (tasks) {
              sBS.set(pJ % bSw);
              gtqH.store(tQHeadGroundState, cuda::memory_order_relaxed);
            }
            tqState[j].tasks = tasks;
            lTt += tasks;
          }
        }
        bitSet[threadIdx.x] = sBS;

        schedulerLoop<processor_state_size, subscribers, sL, schedulerCount>(tqState, processors, processors_v, tilesN1, sO, 0, lTt,
          processorTally, gRQIdx, scheduled, sQ, rQ, pDB, true);

        for (uint i = 1; i < dT; ++i) {
          const uint sBIdx = threadIdx.x + (i * dQL / bSw) * schedulerCount;
          sBS = bitSet[sBIdx];

          #pragma unroll
          for (uint j = sL; j < tqState.size(); ++j) {
            const auto pJ = j - sL;
            const uint bIdx = (i * dQL + pJ) % bSw;
            if (const auto isVisited = sBS.get(bIdx); !isVisited) {
              const auto qIdx = schedulerCount * (dQL * i + pJ) + threadIdx.x;
              cuda::atomic_ref<unsigned int, cuda::thread_scope_device> gtqH{*(gtQHeads + qIdx)};
              const auto tasks = gtqH.load(cuda::memory_order_acquire);
              if (tasks) {
                sBS.set(bIdx);
                gtqH.store(tQHeadGroundState, cuda::memory_order_relaxed);
              }
              tqState[j].tasks = tasks;
              lTt += tasks;
            }
          }
          bitSet[sBIdx] = sBS;

          schedulerLoop<processor_state_size, subscribers, sL, schedulerCount>(tqState, processors, processors_v, tilesN1, sO, i * dQL,
            lTt, processorTally, gRQIdx, scheduled, sQ, rQ, pDB);
        }
      }

      if (threadIdx.x < gtQCL - dT * dQL * schedulerCount) {
        const uint sBIdx = threadIdx.x + (dT * dQL / bSw) * schedulerCount;
        auto sBS = bitSet[sBIdx];

        #pragma unroll
        for (uint j = sL; j < tqState.size(); ++j) {
          const auto pJ = j - sL;
          if (const auto qIdx = schedulerCount * (dQL * dT + pJ) + threadIdx.x; qIdx < gtQCL) {
            const uint bIdx = (dT * dQL + pJ) % bSw;
            if (const auto isVisited = sBS.get(bIdx); !isVisited) {
              cuda::atomic_ref<unsigned int, cuda::thread_scope_device> gtqH{*(gtQHeads + qIdx)};
              const auto tasks = gtqH.load(cuda::memory_order_acquire);
              if (tasks) {
                sBS.set(bIdx);
                gtqH.store(tQHeadGroundState, cuda::memory_order_relaxed);
              }
              tqState[j].tasks = tasks;
              lTt += tasks;
            }
          }
        }
        bitSet[sBIdx] = sBS;
      }

      schedulerLoop<processor_state_size, subscribers, sL, schedulerCount>(tqState, processors, processors_v, tilesN1, sO, dQL * dT,
        lTt, processorTally, gRQIdx, scheduled, sQ, rQ, pDB, dT == 0);

      if (!threadIdx.x) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> tb{*taskBound};
        tTB = tb.load(cuda::memory_order_relaxed);
      }
      __syncwarp();
      tTB = __shfl_sync(0xffffffffffffffffULL, tTB, 0);  // 64-bit mask for AMD
    }

    __syncwarp();

    // Interrupt subscribers
    // Use the MI450 wave size for the subscriber interrupt loop.
    #pragma unroll
    for (uint sid = threadIdx.x; sid < (subscribers / WARP_SIZE); sid += SCHEDULER_COUNT) {
      cuda::atomic_ref<uint, cuda::thread_scope_block> inr{*(sInterrupts + sid)};
      inr.store(1, cuda::memory_order_relaxed);
    }

    // Interrupt processors
    sPI<processor_state_size, schedulerCount>(rQ, sQ, pDB, gRQIdx, interruptScratch, processors, processors_v, processorTally);
  }

} // namespace flashmoe::scheduler

#endif //SCHEDULER_HIP_CUH
