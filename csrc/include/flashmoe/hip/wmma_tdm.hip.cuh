/*
 * Shared gfx1250 WMMA/TDM tile helpers.
 */

#ifndef FLASHMOE_HIP_WMMA_TDM_CUH
#define FLASHMOE_HIP_WMMA_TDM_CUH

#include <hip/hip_runtime.h>
#if defined(__HIP_PLATFORM_AMD__)
#include <hip/amd_detail/amd_gfx1250_TDM.h>
#endif

#include "flashmoe/hip/constants.hpp"
#include "flashmoe/hip/tile.hip.cuh"

namespace flashmoe::wmma_tdm {

constexpr size_t roundUpBytes(const size_t value, const size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

template <typename TileGEMM, typename Element, typename AccumType>
struct WmmaTdmWorkspace {
  static constexpr int bM = hip_tensor::get<0>(typename TileGEMM::TileShape{});
  static constexpr int bN = hip_tensor::get<1>(typename TileGEMM::TileShape{});
  static constexpr int WMMA_M = TileGEMM::WMMA_M;
  static constexpr int WMMA_N = TileGEMM::WMMA_N;
  static constexpr int WMMA_K = TileGEMM::WMMA_K;
  static constexpr int wavesPerBlock = TileGEMM::Threads::value / TileGEMM::ARCH_WAVE_SIZE;

  static constexpr size_t accumBytes = sizeof(AccumType) * bM * bN;
  static constexpr size_t gateOffset = roundUpBytes(accumBytes, flashmoe::MAX_ACCESS_ALIGNMENT);
  static constexpr size_t gateBytes = sizeof(AccumType) * bM * bN;
  static constexpr size_t tdmScratchOffset =
    roundUpBytes(gateOffset + gateBytes, flashmoe::MAX_ACCESS_ALIGNMENT);
  static constexpr size_t perWaveABytes = sizeof(Element) * WMMA_M * WMMA_K;
  static constexpr size_t perWaveBBytes = sizeof(Element) * WMMA_K * WMMA_N;
  static constexpr size_t perWaveBytes =
    roundUpBytes(perWaveABytes + perWaveBBytes, flashmoe::MAX_ACCESS_ALIGNMENT);
  static constexpr size_t barrierOffset =
    roundUpBytes(tdmScratchOffset + wavesPerBlock * perWaveBytes, alignof(long));
  static constexpr size_t totalBytes =
    roundUpBytes(barrierOffset + wavesPerBlock * sizeof(long), flashmoe::MAX_ACCESS_ALIGNMENT);
};

template <typename TileGEMM, typename Element, typename AccumType>
__host__ __device__ constexpr size_t wmmaTdmWorkspaceBytes() {
  return WmmaTdmWorkspace<TileGEMM, Element, AccumType>::totalBytes;
}

template <typename TileGEMM, typename Element, typename AccumType>
__host__ __device__ constexpr size_t wmmaGateOffsetBytes() {
  return WmmaTdmWorkspace<TileGEMM, Element, AccumType>::gateOffset;
}

template <typename Element>
__device__ __forceinline__
constexpr uint32_t tdmDataSizeLog2() {
  static_assert(sizeof(Element) == 1 || sizeof(Element) == 2 ||
                sizeof(Element) == 4 || sizeof(Element) == 8,
                "TDM supports element sizes that are powers of two up to 8 bytes");
  if constexpr (sizeof(Element) == 1) {
    return 0;
  }
  else if constexpr (sizeof(Element) == 2) {
    return 1;
  }
  else if constexpr (sizeof(Element) == 4) {
    return 2;
  }
  else {
    return 3;
  }
}

#if defined(__gfx1250__) || defined(__gfx1251__)
using TdmV4I = int __attribute__((ext_vector_type(4)));
using TdmV8I = int __attribute__((ext_vector_type(8)));

__device__ __forceinline__
unsigned ldsBarrierPhase(volatile long* barrier) {
  const auto state = static_cast<unsigned long long>(*barrier);
  return static_cast<unsigned>((state >> 29) & 1ULL);
}

__device__ __forceinline__
void waitLdsBarrierPhaseChange(volatile long* barrier, const unsigned phase) {
  while (ldsBarrierPhase(barrier) == phase) {
    __builtin_amdgcn_s_sleep(1);
  }
}

template <typename Element>
__device__ __forceinline__
gfx1250_TDM_GROUP1 makeTdm2dGroup1(const uint32_t dim0,
                                   const uint32_t dim1,
                                   const uint32_t stride0,
                                   const uint32_t tileDim0,
                                   const uint32_t tileDim1,
                                   volatile long* completionBarrier = nullptr) {
  gfx1250_TDM_GROUP1 group1;
  group1.dataSize(tdmDataSizeLog2<Element>());
  group1.tensorDim0(dim0);
  group1.tensorDim1(dim1);
  group1.tensorDim0Stride(stride0);
  group1.tensorDim1Stride(dim1);
  group1.tileDim0(tileDim0);
  group1.tileDim1(tileDim1);
  group1.workgroupMask(0);
  if (completionBarrier != nullptr) {
    group1.atomicBarrierEnable(true);
    group1.atomicBarrierAddress(
      static_cast<uint32_t>((reinterpret_cast<uintptr_t>(const_cast<long*>(completionBarrier)) >> 3) & 0xffffU));
  }
  else {
    group1.atomicBarrierEnable(false);
  }
  return group1;
}

template <typename Element>
__device__ __forceinline__
void tdmLoad2dToLds(Element* __restrict__ lds,
                    const Element* __restrict__ global,
                    gfx1250_TDM_GROUP1 group1) {
  gfx1250_TDM_GROUP0 group0;
  group0.globalAddr(reinterpret_cast<uintptr_t>(global));
  group0.ldsAddr(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds)));

  const TdmV4I v4Zeros{0, 0, 0, 0};
  const TdmV8I v8Zeros{0, 0, 0, 0, 0, 0, 0, 0};
  __builtin_amdgcn_tensor_load_to_lds(group0.m_bitfield, group1.m_bitfield,
                                      v4Zeros, v4Zeros, v8Zeros, 0);
}
#endif

template <typename TileGEMM, typename Element, typename AccumType>
__forceinline__ __device__
void computeTileRowMajorB(const Element* __restrict__ const& a,
                          const Element* __restrict__ const& b,
                          AccumType* __restrict__ const& sC,
                          const int& M, const int& N, const int& K,
                          const int& tileIdx) {
  constexpr int bM = hip_tensor::get<0>(typename TileGEMM::TileShape{});
  constexpr int bN = hip_tensor::get<1>(typename TileGEMM::TileShape{});
  constexpr int WMMA_M = TileGEMM::WMMA_M;
  constexpr int WMMA_N = TileGEMM::WMMA_N;
  constexpr int WMMA_K = TileGEMM::WMMA_K;
  constexpr int nThreads = TileGEMM::Threads::value;
  constexpr int waveSize = TileGEMM::ARCH_WAVE_SIZE;
  constexpr int wavesPerBlock = nThreads / waveSize;
  constexpr int tilesPerM = bM / WMMA_M;
  constexpr int tilesPerN = bN / WMMA_N;
  constexpr int totalTiles = tilesPerM * tilesPerN;

  using FragA = rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, Element, rocwmma::row_major>;
  using FragB = rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, Element, rocwmma::row_major>;
  using FragC = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, AccumType>;

  for (int i = threadIdx.x; i < bM * bN; i += nThreads) {
    sC[i] = AccumType(0);
  }
  __syncthreads();

  const int tilesM = M / bM;
  const int tilesN = N / bN;
  const auto tileCoord = flashmoe::tile::idx2Coord(tilesM, tilesN, tileIdx);
  const int tileRow = hip_tensor::get<0>(tileCoord);
  const int tileCol = hip_tensor::get<1>(tileCoord);
  const int waveId = threadIdx.x / waveSize;

#if defined(__gfx1250__) || defined(__gfx1251__)
  using Workspace = WmmaTdmWorkspace<TileGEMM, Element, AccumType>;
  char* workspaceBytes = reinterpret_cast<char*>(sC);
  char* tdmScratch = workspaceBytes + Workspace::tdmScratchOffset;
  Element* waveA = reinterpret_cast<Element*>(tdmScratch + waveId * Workspace::perWaveBytes);
  Element* waveB = reinterpret_cast<Element*>(
    reinterpret_cast<char*>(waveA) + Workspace::perWaveABytes);
  auto* barriers = reinterpret_cast<volatile long*>(workspaceBytes + Workspace::barrierOffset);
  volatile long* waveBarrier = barriers + waveId;

  if ((threadIdx.x % waveSize) == 0) {
    *waveBarrier = 0;
  }
  __syncthreads();
#endif

  for (int outTile = waveId; outTile < totalTiles; outTile += wavesPerBlock) {
    const int localTileM = outTile / tilesPerN;
    const int localTileN = outTile % tilesPerN;
    const int rowOffset = localTileM * WMMA_M;
    const int colOffset = localTileN * WMMA_N;

    FragC acc;
    rocwmma::fill_fragment(acc, AccumType(0));

#if defined(__gfx1250__) || defined(__gfx1251__)
    unsigned barrierPhase = ldsBarrierPhase(waveBarrier);
#endif
    for (int k = 0; k < K; k += WMMA_K) {
      FragA fragA;
      FragB fragB;

      const Element* aPtr = a
        + static_cast<size_t>(tileRow * bM + rowOffset) * K
        + k;
      const Element* bPtr = b
        + static_cast<size_t>(k) * N
        + tileCol * bN
        + colOffset;

#if defined(__gfx1250__) || defined(__gfx1251__)
      auto aDesc = makeTdm2dGroup1<Element>(
        static_cast<uint32_t>(K),
        static_cast<uint32_t>(M),
        static_cast<uint32_t>(K),
        static_cast<uint32_t>(WMMA_K),
        static_cast<uint32_t>(WMMA_M));
      tdmLoad2dToLds(waveA, aPtr, aDesc);

      auto bDesc = makeTdm2dGroup1<Element>(
        static_cast<uint32_t>(N),
        static_cast<uint32_t>(K),
        static_cast<uint32_t>(N),
        static_cast<uint32_t>(WMMA_N),
        static_cast<uint32_t>(WMMA_K),
        waveBarrier);
      tdmLoad2dToLds(waveB, bPtr, bDesc);
      waitLdsBarrierPhaseChange(waveBarrier, barrierPhase);
      barrierPhase ^= 1U;

      rocwmma::load_matrix_sync(fragA, waveA, WMMA_K);
      rocwmma::load_matrix_sync(fragB, waveB, WMMA_N);
#else
      rocwmma::load_matrix_sync(fragA, aPtr, K);
      rocwmma::load_matrix_sync(fragB, bPtr, N);
#endif
      rocwmma::mma_sync(acc, fragA, fragB, acc);
    }

    AccumType* cTile = sC + rowOffset * bN + colOffset;
    rocwmma::store_matrix_sync(cTile, acc, bN, rocwmma::mem_row_major);
  }
  __syncthreads();
}

} // namespace flashmoe::wmma_tdm

#endif // FLASHMOE_HIP_WMMA_TDM_CUH
