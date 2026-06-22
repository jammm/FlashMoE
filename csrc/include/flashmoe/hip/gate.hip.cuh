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
// Created by oja7 on 11/25/24.
// HIP Port: 2026
//

#ifndef GATE_HIP_CUH
#define GATE_HIP_CUH

// HIP runtime and compatibility headers
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

// FlashMoE HIP compatibility layer
#include "constants.hpp"
#include "cuda_compat.hpp"
#include "tensor.hpp"
#include "atomics.hip.cuh"
#include "context.hip.cuh"
#include "tile.hip.cuh"
#include "wmma_tdm.hip.cuh"

namespace flashmoe
{
  // ============================================================================
  // Type Aliases for HIP Port
  // ============================================================================

  // Unsigned long long type alias (matches CUDA version)
  using ull_t = unsigned long long int;

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

  // ============================================================================
  // Atomic Operations for HIP
  // Note: guardedAtomicAdd is defined in atomics.hip.cuh
  // ============================================================================

  // ============================================================================
  // Optimization Level Enums
  // ============================================================================

  enum class SoftMaxOptimizationLevel {
    none,
    // use fast exponential and float division
    highest,
  };

  enum class GateReductionLevel {
    singleBlock,
    multiBlock
  };
}

namespace flashmoe::gate
{
  constexpr int BLOCK_N_CAP = 32;

  enum class ReturnLogits {
    yes,
    no
  };

  // ============================================================================
  // Fast Math Functions (HIP versions)
  // ============================================================================

  template <SoftMaxOptimizationLevel level = SoftMaxOptimizationLevel::none>
  __device__ __forceinline__
  float fexp(const float& x) {
    static_assert(level == SoftMaxOptimizationLevel::none);
    return expf(x);
  }

  template <>
  __device__ __forceinline__
  float fexp<SoftMaxOptimizationLevel::highest>(const float& x) {
    return __expf(x);
  }

  template <SoftMaxOptimizationLevel level = SoftMaxOptimizationLevel::none>
  __device__ __forceinline__
  float fdiv(const float& x, const float& y) {
    static_assert(level == SoftMaxOptimizationLevel::none);
    return fdividef(x, y);
  }

  template <>
  __device__ __forceinline__
  float fdiv<SoftMaxOptimizationLevel::highest>(const float& x, const float& y) {
    return __fdividef(x, y);
  }

  // ============================================================================
  // hipCUB BlockScan Type Alias
  // ============================================================================

  // Using hipCUB instead of CUB for AMD GPUs
  // Note: BLOCK_SCAN_WARP_SCANS uses wave-level scan primitives.
  // MI450/gfx1250 is wave32.
  template <int threads>
  using BlockScan = hipcub::BlockScan<int, threads, hipcub::BLOCK_SCAN_WARP_SCANS>;

  using SoftType = float;

  // ============================================================================
  // Numeric Limits for HIP
  // ============================================================================

  template<typename T>
  struct numeric_limits;

  template<>
  struct numeric_limits<float> {
    static constexpr bool is_iec559 = true;
    static constexpr bool has_infinity = true;
    __device__ __host__ static constexpr float infinity() {
      return __builtin_huge_valf();
    }
  };

  // ============================================================================
  // Utility Functions (replacing cute:: functions)
  // ============================================================================

  template<typename T>
  __device__ __forceinline__
  constexpr T hip_min(T a, T b) {
    return (a < b) ? a : b;
  }

  template<typename T>
  __device__ __forceinline__
  constexpr T hip_max(T a, T b) {
    return (a > b) ? a : b;
  }

  template<int N, int D>
  __device__ __host__ __forceinline__
  constexpr int ceil_div() {
    return (N + D - 1) / D;
  }

  __device__ __host__ __forceinline__
  int ceil_div(int n, int d) {
    return (n + d - 1) / d;
  }

  // ============================================================================
  // Tile Coordinate Utilities (replacing cute:: tile functions)
  // ============================================================================

  namespace tile {
    // Convert linear tile index to 2D coordinate
    template<typename T = int>
    __device__ __forceinline__
    auto idx2Coord(const int& tilesM, const int& tilesN, const int& tileIdx) {
      const int tileM = tileIdx % tilesM;
      const int tileN = tileIdx / tilesM;
      return hip_tensor::make_coord(tileM, tileN);
    }
  }

  template<bool Enable, int bM, int bN, int bK, int pSK, int Arch, int threads,
           typename Element, typename AccumType>
  struct GateWmmaSelector;

  template<int bM, int bN, int bK, int pSK, int Arch, int threads,
           typename Element, typename AccumType = float>
  constexpr auto kernelSMEM() {
    constexpr bool enableWmmaGate =
      (Arch >= 1200 && Arch < 1300) &&
      (sizeof(Element) == 2) &&
      cuda::std::is_same_v<AccumType, float> &&
      (bM % 16 == 0) && (bN % 16 == 0) && (bK % 32 == 0);
    using Wmma = GateWmmaSelector<enableWmmaGate, bM, bN, bK, pSK, Arch, threads, Element, AccumType>;
    constexpr auto gemmBytes = sizeof(Element) * bK * pSK * (bM + bN);
    constexpr auto epilogueBytes = sizeof(SoftType) * bM * bN;
    constexpr auto scalarRaw = gemmBytes > epilogueBytes ? gemmBytes : epilogueBytes;
    constexpr auto wmmaRaw = Wmma::workspaceBytes;
    constexpr auto raw = scalarRaw > wmmaRaw ? scalarRaw : wmmaRaw;
    constexpr auto aligned = ((raw + flashmoe::MAX_ACCESS_ALIGNMENT - 1)
                              / flashmoe::MAX_ACCESS_ALIGNMENT)
                             * flashmoe::MAX_ACCESS_ALIGNMENT;
    return static_cast<unsigned int>(aligned);
  }

  struct GateKernelArgs {
    const std::byte* const tokens = nullptr;
    const std::byte* const weights = nullptr;
    std::byte* const routing = nullptr;
    int* const expertCounts = nullptr;
    TPS* const tokenIds = nullptr;
    const int S = 0;
    const int H = 0;
    const int E = 0;
    const int k = 0;
    const int EC = 0;
    const int roundEC = 0;
  };

  template<bool Enable, int bM, int bN, int bK, int pSK, int Arch, int threads,
           typename Element, typename AccumType>
  struct GateWmmaSelector {
    static constexpr unsigned int workspaceBytes = 0;
  };

  template<int bM, int bN, int bK, int pSK, int Arch, int threads,
           typename Element, typename AccumType>
  struct GateWmmaSelector<true, bM, bN, bK, pSK, Arch, threads, Element, AccumType> {
    using TileGEMM = flashmoe::tile::CollectiveMainloop<
      bM, bN, bK, Arch, Element, AccumType, threads, pSK,
      flashmoe::tile::arrangement::row_major,
      flashmoe::tile::arrangement::row_major>;
    static constexpr int WMMA_K = TileGEMM::WMMA_K;
    static constexpr unsigned int workspaceBytes =
      static_cast<unsigned int>(wmma_tdm::wmmaTdmWorkspaceBytes<TileGEMM, Element, AccumType>());

    __device__ __forceinline__
    static void compute(void* __restrict__ const& workspace,
                        const GateKernelArgs& kArgs,
                        const int& tileIdx) {
      wmma_tdm::computeTileRowMajorB<TileGEMM, Element, AccumType>(
        reinterpret_cast<const Element*>(kArgs.tokens),
        reinterpret_cast<const Element*>(kArgs.weights),
        reinterpret_cast<AccumType*>(workspace),
        kArgs.S, kArgs.E, kArgs.H, tileIdx);
    }
  };

  template <
    GateReductionLevel g,
    int bM, int bN, int bK, int pSK, int Arch,
    int threads,
    SoftMaxOptimizationLevel sro,
    ReturnLogits r,
    typename AccumType,
    typename Element,
    typename ElementC
  >
  struct GateMainloop {
    static constexpr bool enableWmmaGate =
      (Arch >= 1200 && Arch < 1300) &&
      (sizeof(Element) == 2) &&
      cuda::std::is_same_v<AccumType, float> &&
      (bM % 16 == 0) && (bN % 16 == 0) && (bK % 32 == 0);
    using Wmma = GateWmmaSelector<enableWmmaGate, bM, bN, bK, pSK, Arch, threads, Element, AccumType>;

    __device__ __forceinline__
    void tileGemm(SoftType* __restrict__ sMat,
                  const Element* __restrict__ A, const Element* __restrict__ B,
                  const int S, const int H, const int E,
                  const int tileRow, const int tileCol) {
      constexpr Converter<SoftType, Element> toFloat{};
      for (int row = threadIdx.x; row < bM; row += threads) {
        for (int col = 0; col < bN; ++col) {
          SoftType acc = SoftType(0);
          const int globalRow = tileRow * bM + row;
          const int globalCol = tileCol * bN + col;
          if (globalRow < S && globalCol < E) {
            for (int kk = 0; kk < H; ++kk) {
              acc += toFloat(A[globalRow * H + kk]) * toFloat(B[kk * E + globalCol]);
            }
          }
          sMat[row * bN + col] = acc;
        }
      }
    }

    __device__ __forceinline__
    void operator()(void* __restrict__ const& workspace,
                    const GateKernelArgs& kArgs, const GateContext& ctx,
                    const int& tileIdx) {
      auto* sMat = static_cast<SoftType*>(workspace);
      const int tilesM = (kArgs.S + bM - 1) / bM;
      const int tilesN = kArgs.E / bN;
      const auto tileCoord = tile::idx2Coord(tilesM, tilesN, tileIdx);
      const int tileRow = hip_tensor::get<0>(tileCoord);
      const int tileCol = hip_tensor::get<1>(tileCoord);

      if constexpr (enableWmmaGate) {
        if ((kArgs.S % bM) == 0 && (kArgs.E % bN) == 0 &&
            (kArgs.H % Wmma::WMMA_K) == 0) {
          Wmma::compute(workspace, kArgs, tileIdx);
        }
        else {
          tileGemm(sMat,
                   reinterpret_cast<const Element*>(kArgs.tokens),
                   reinterpret_cast<const Element*>(kArgs.weights),
                   kArgs.S, kArgs.H, kArgs.E, tileRow, tileCol);
          __syncthreads();
        }
      }
      else {
        tileGemm(sMat,
                 reinterpret_cast<const Element*>(kArgs.tokens),
                 reinterpret_cast<const Element*>(kArgs.weights),
                 kArgs.S, kArgs.H, kArgs.E, tileRow, tileCol);
        __syncthreads();
      }

      const int predicate = (kArgs.S < bM) ? kArgs.S : bM;
      SoftType reginald[bN];
      SoftType dI = SoftType(0);
      SoftType mI = -__builtin_huge_valf();

      if (threadIdx.x < predicate) {
        #pragma unroll
        for (int j = 0; j < bN; ++j)
          reginald[j] = sMat[threadIdx.x * bN + j];

        if constexpr (g == GateReductionLevel::multiBlock) {
          const int myOff = bM * (tileRow + tileCol * tilesM) + threadIdx.x;
          const int nextCol = (tileCol + 1 < tilesN) ? tileCol + 1 : 0;
          const int nextOff = bM * (tileRow + nextCol * tilesM) + threadIdx.x;
          auto* __restrict__ brsMailbox = ctx.ssp + myOff;
          auto* __restrict__ brsXMailbox = ctx.ssp + nextOff;

          if (tileCol > 0) {
            auto payload = cuda::std::bit_cast<SoftmaxStatePacked>(
              atomicExch(reinterpret_cast<unsigned long long*>(brsMailbox), 0ULL));
            while (!has_payload_arrived(payload))
              payload = cuda::std::bit_cast<SoftmaxStatePacked>(
                atomicExch(reinterpret_cast<unsigned long long*>(brsMailbox), 0ULL));
            unpack_state(payload, mI, dI);
          }

          #pragma unroll
          for (int j = 0; j < bN; ++j) {
            const auto pM = mI;
            mI = (mI > reginald[j]) ? mI : reginald[j];
            dI = fmaf(dI, fexp<sro>(pM - mI), fexp<sro>(reginald[j] - mI));
          }

          if (tileCol + 1 < tilesN) {
            auto sP = pack_state(mI, dI);
            atomicExch(reinterpret_cast<unsigned long long*>(brsXMailbox),
                       cuda::std::bit_cast<unsigned long long>(sP));
            auto payload = cuda::std::bit_cast<SoftmaxStatePacked>(
              atomicExch(reinterpret_cast<unsigned long long*>(brsMailbox), 0ULL));
            while (!has_payload_arrived(payload))
              payload = cuda::std::bit_cast<SoftmaxStatePacked>(
                atomicExch(reinterpret_cast<unsigned long long*>(brsMailbox), 0ULL));
            unpack_state(payload, mI, dI);
          } else {
            auto sP = pack_state(mI, dI);
            for (int j = 0; j < tilesN - 1; ++j)
              atomicExch(reinterpret_cast<unsigned long long*>(brsXMailbox + bM * j * tilesM),
                         cuda::std::bit_cast<unsigned long long>(sP));
          }
        } else {
          #pragma unroll
          for (int j = 0; j < bN; ++j) {
            const auto pM = mI;
            mI = (mI > reginald[j]) ? mI : reginald[j];
            dI = fmaf(dI, fexp<sro>(pM - mI), fexp<sro>(reginald[j] - mI));
          }
        }

        #pragma unroll
        for (int j = 0; j < bN; ++j)
          reginald[j] = fdiv<sro>(fexp<sro>(reginald[j] - mI), dI);

        #pragma unroll
        for (int j = 0; j < bN; ++j)
          sMat[threadIdx.x * bN + j] = reginald[j];
      }
      __syncthreads();

      if constexpr (r == ReturnLogits::yes) {
        constexpr Converter<ElementC, SoftType> store{};
        auto* gC = reinterpret_cast<ElementC*>(kArgs.routing);
        for (int idx = threadIdx.x; idx < bM * bN; idx += threads) {
          const int row = idx / bN;
          const int col = idx % bN;
          const int globalRow = tileRow * bM + row;
          const int globalCol = tileCol * bN + col;
          if (globalRow < kArgs.S && globalCol < kArgs.E)
            gC[globalRow * kArgs.E + globalCol] = store(sMat[row * bN + col]);
        }
      }

      int rTK[bN];
      #pragma unroll
      for (int i = 0; i < bN; ++i) rTK[i] = 0;

      SoftType mCw = SoftType(0);
      if (threadIdx.x < predicate) {
        if constexpr (g == GateReductionLevel::multiBlock) {
          constexpr int phases = 2;
          const int myTkOff = bM * (tileRow + phases * tileCol * tilesM) + threadIdx.x;
          const int nextCol = (tileCol + 1 < tilesN) ? tileCol + 1 : 0;
          const int nextTkOff = bM * (tileRow + phases * nextCol * tilesM) + threadIdx.x;
          auto* __restrict__ tkMailbox = ctx.rtp + myTkOff;
          auto* __restrict__ tkXMailbox = ctx.rtp + nextTkOff;

          SoftType lSV = -__builtin_huge_valf();
          uint32_t lSIdx = 0;
          bool shouldSweep = true;

          for (int i = 0; i < kArgs.k; ++i) {
            const int flagPrefix = (i % phases) * bM * tilesM;
            if (shouldSweep) {
              lSV = -__builtin_huge_valf();
              #pragma unroll
              for (int j = 0; j < bN; ++j) {
                if (!rTK[j] && reginald[j] > lSV) {
                  lSIdx = tileCol * bN + j;
                  lSV = reginald[j];
                }
              }
              shouldSweep = false;
            }
            SoftType sV = lSV;
            uint32_t sIdx = lSIdx;
            if (tileCol > 0) {
              auto* __restrict__ rb = reinterpret_cast<unsigned long long*>(tkMailbox + flagPrefix);
              auto payload = cuda::std::bit_cast<RingTopKPayload>(atomicExch(rb, 0ULL));
              while (!has_payload_arrived(payload))
                payload = cuda::std::bit_cast<RingTopKPayload>(atomicExch(rb, 0ULL));
              SoftType rV; uint32_t rI;
              unpack_tk_payload(payload, rV, rI);
              if (lSV > rV) { sV = lSV; sIdx = lSIdx; }
              else { sV = rV; sIdx = rI; }
            }
            if (tileCol + 1 < tilesN) {
              atomicExch(reinterpret_cast<unsigned long long*>(tkXMailbox + flagPrefix),
                         cuda::std::bit_cast<unsigned long long>(pack_tk_payload(sV, sIdx)));
              auto* __restrict__ rb = reinterpret_cast<unsigned long long*>(tkMailbox + flagPrefix);
              auto payload = cuda::std::bit_cast<RingTopKPayload>(atomicExch(rb, 0ULL));
              while (!has_payload_arrived(payload))
                payload = cuda::std::bit_cast<RingTopKPayload>(atomicExch(rb, 0ULL));
              unpack_tk_payload(payload, sV, sIdx);
            } else {
              auto packed = cuda::std::bit_cast<unsigned long long>(pack_tk_payload(sV, sIdx));
              auto* mb = tkXMailbox;
              for (int j = 0; j < tilesN - 1; ++j) {
                atomicExch(reinterpret_cast<unsigned long long*>(mb + flagPrefix), packed);
                mb += phases * bM * tilesM;
              }
            }
            if (static_cast<int>(sIdx / bN) == tileCol) {
              const int sIdxIntra = sIdx % bN;
              #pragma unroll
              for (int j = 0; j < bN; ++j)
                if (j == sIdxIntra) rTK[j] = 1;
              shouldSweep = true;
            }
            mCw += sV;
          }
        } else {
          for (int i = 0; i < kArgs.k; ++i) {
            SoftType sV = -__builtin_huge_valf();
            int sIdx = 0;
            #pragma unroll
            for (int j = 0; j < bN; ++j) {
              if (reginald[j] > sV && !rTK[j]) { sIdx = j; sV = reginald[j]; }
            }
            #pragma unroll
            for (int j = 0; j < bN; ++j)
              if (j == sIdx) rTK[j] = 1;
            mCw += sV;
          }
        }
      }

      __shared__ int startIndices[bN];
      using BTS = typename BlockScan<threads>::TempStorage;
      __shared__ __align__(alignof(BTS)) BTS scanTempStorage[bN];
      int myIndices[bN];
      constexpr int sl = (bN + threads - 1) / threads;
      int stash[sl];
      #pragma unroll
      for (int i = 0; i < sl; ++i) stash[i] = 0;
      #pragma unroll
      for (int i = 0; i < bN; ++i) {
        int selected = 0;
        BlockScan<threads>(scanTempStorage[i]).InclusiveSum(rTK[i], myIndices[i], selected);
        stash[i / threads] = (threadIdx.x == (i % threads)) ? selected : stash[i / threads];
      }
      #pragma unroll
      for (int i = 0; i < sl; ++i) {
        const int idx = threadIdx.x + i * threads;
        if (idx < bN) {
          const int expertIdx = bN * tileCol + idx;
          startIndices[idx] = guardedAtomicAdd(ctx.ecGuards + expertIdx,
                                               kArgs.expertCounts + expertIdx, stash[i],
                                               tilesM);
        }
      }
      __syncthreads();
      #pragma unroll
      for (int i = 0; i < bN; ++i)
        myIndices[i] = startIndices[i] + myIndices[i] - 1;

      if (threadIdx.x < predicate) {
        #pragma unroll
        for (int i = 0; i < bN; ++i) {
          if (rTK[i] && myIndices[i] < kArgs.EC) {
            const int expertIdx = bN * tileCol + i;
            const int tokenIdx = tileRow * bM + threadIdx.x;
            kArgs.tokenIds[expertIdx * kArgs.roundEC + myIndices[i]] = TPS{
              static_cast<uint32_t>(tokenIdx),
              fdividef(reginald[i], mCw)
            };
          }
        }
      }
    }
  };

  template <
    int bM, int bN, int bK, int pSK, int Arch,
    int threads,
    GateReductionLevel grl,
    SoftMaxOptimizationLevel sro,
    ReturnLogits r,
    typename AccumType,
    typename Element,
    typename ElementR
  >
  __device__ __forceinline__
  void forward(void* __restrict__ const& workspace, const GateKernelArgs& kArgs, const GateContext& ctx) {
    GateMainloop<grl, bM, bN, bK, pSK, Arch, threads, sro, r, AccumType, Element, ElementR> mainLoop{};
    const auto nT = ((kArgs.S + bM - 1) / bM) * (kArgs.E / bN);
    for (int tileIdx = static_cast<int>(blockIdx.x); tileIdx < nT; tileIdx += static_cast<int>(gridDim.x)) {
      mainLoop(workspace, kArgs, ctx, tileIdx);
    }
  }

  template<
    typename TileShape,
    int Arch,
    int threads,
    GateReductionLevel grl = GateReductionLevel::singleBlock,
    SoftMaxOptimizationLevel sro = SoftMaxOptimizationLevel::none,
    ReturnLogits r = ReturnLogits::yes,
    typename AccumType = float,
    typename Element = __half,
    typename ElementR = float
  >
  __launch_bounds__(threads, 1)
  __global__ void forwardKernel(const GateKernelArgs kArgs, const GateContext ctx) {
    constexpr int bM = hip_tensor::get<0>(TileShape{});
    constexpr int bN = hip_tensor::get<1>(TileShape{});
    constexpr int bK = hip_tensor::get<2>(TileShape{});
    constexpr int pipeStages = hip_tensor::get<3>(TileShape{});
    extern __shared__ __align__(128) std::byte gateWorkspace[];
    gate::forward<bM, bN, bK, pipeStages, Arch, threads, grl, sro, r, AccumType, Element, ElementR>(gateWorkspace, kArgs, ctx);
  }

} // namespace flashmoe::gate

#endif //GATE_HIP_CUH
