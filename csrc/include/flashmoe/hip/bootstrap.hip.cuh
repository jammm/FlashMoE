/*
 * HIP Port of bootstrap.cuh — Initialization, finalization, and helper templates
 * Target: AMD Instinct MI450 (gfx1250) / MI450 (gfx1250)
 */

#ifndef FLASHMOE_HIP_BOOTSTRAP_CUH
#define FLASHMOE_HIP_BOOTSTRAP_CUH

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include "flashmoe/hip/cuda_compat.hpp"
#include "flashmoe/hip/constants.hpp"
#include "flashmoe/hip/tensor.hpp"
#include "flashmoe/hip/signal.hip.cuh"
#include "flashmoe/hip/heap.hip.cuh"
#include "flashmoe/hip/atomics.hip.cuh"

// ============================================================================
// Error Checking
// ============================================================================

#if !defined(CHECK_HIP_BOOTSTRAP)
#define CHECK_HIP_BOOTSTRAP(e)                                        \
  do {                                                                \
    hipError_t code = (e);                                            \
    if (code != hipSuccess) {                                         \
      fprintf(stderr, "<%s:%d> %s:\n    %s: %s\n",                   \
              __FILE__, __LINE__, #e,                                 \
              hipGetErrorName(code), hipGetErrorString(code));        \
      fflush(stderr);                                                 \
      throw std::runtime_error(hipGetErrorString(code));              \
    }                                                                 \
  } while (0)
#endif

namespace flashmoe {

// Forward definitions of enums needed by helper templates below.
// Guarded so the canonical definitions in their home headers take precedence.

#ifndef FLASHMOE_HIP_MLPMATMULTYPE_DEFINED
#define FLASHMOE_HIP_MLPMATMULTYPE_DEFINED
enum class MLPMatmulType { gated, vanilla };
#endif

#ifndef FLASHMOE_HIP_ACTIVATION_DEFINED
#define FLASHMOE_HIP_ACTIVATION_DEFINED
enum class Activation { identity, silu, gelu, relu };
#endif

// ============================================================================
// DataType template — maps integer constants to HIP types
// ============================================================================

template<int t>
struct DataType {
    static_assert(t >= 0 && t < 4, "Invalid datatype constant");
};
template<> struct DataType<0> { using Type = hip_bfloat16; };
template<> struct DataType<1> { using Type = __half; };
template<> struct DataType<2> { using Type = float; };
template<> struct DataType<3> { using Type = double; };

// ============================================================================
// Topology helpers (Topology enum is in signal.hip.cuh, included above)
// ============================================================================

template<int t>
consteval Topology defineTopology() {
    static_assert(t == 0 || t == 1, "Invalid Topology enum value");
    if constexpr (t == 0) return Topology::NVLINK_ONLY;
    else return Topology::MIXED;
}

// ============================================================================
// MLPMatmulType helpers
// ============================================================================

template<int m>
consteval MLPMatmulType defineMLPType() {
    static_assert(m == 0 || m == 1, "Invalid MLPMatmulType enum value");
    if constexpr (m == 0) return MLPMatmulType::gated;
    else return MLPMatmulType::vanilla;
}

// ============================================================================
// Activation helpers
// ============================================================================

template<int a>
consteval Activation defineAct() {
    static_assert(a >= 0 && a < 4);
    if constexpr (a == 0) return Activation::identity;
    else if constexpr (a == 1) return Activation::silu;
    else if constexpr (a == 2) return Activation::gelu;
    else return Activation::relu;
}

// ============================================================================
// Heuristics Namespace (Tile Shape Selection)
// ============================================================================

namespace heuristics {

template<int M, int Arch>
consteval int getTileM() {
    static_assert(M > 0 && (M == 1 || M % 2 == 0));
    constexpr int cap = (Arch >= 900 && M >= 4096) ? 256 : 128;
    for (const int t : {128, 96, 64, 48, 32, 24, 16, 8}) {
        if (t <= cap && t <= M && (M % t == 0)) return t;
    }
    return M < 128 ? M : 1;
}

template<int M, int Arch>
consteval int getMoETileM() {
    constexpr int limit = (Arch >= 900 && M >= 4096 && M % 2 == 0) ? 256 : 128;
    constexpr int clamped = (M < limit) ? M : limit;
    return (clamped > 16) ? clamped : 16;
}

template<int M, int Arch, typename Element>
consteval int getMoETileMForElement() {
    constexpr int type_limit = (sizeof(Element) > 4) ? 32 :
                               ((sizeof(Element) > 2) ? 64 :
                                ((Arch >= 900 && M >= 4096 && M % 2 == 0) ? 256 : 128));
    constexpr int clamped = (M < type_limit) ? M : type_limit;
    return (clamped > 16) ? clamped : 16;
}

template<int N, typename Element>
consteval int getTileN() {
    static_assert(N > 0 && N % 8 == 0);
    constexpr int cap = (sizeof(Element) <= 2) ? 128 : 64;
    for (const int t : {128, 96, 64, 48, 32, 24, 16, 8}) {
        if (t <= cap && t <= N && (N % t == 0)) return t;
    }
    return 8;
}

template<int N, int cap>
consteval int getGateTileN() {
    static_assert(N > 0);
    for (const int t : {128, 96, 64, 48, 32, 24, 16, 8, 4, 2, 1}) {
        if (t <= cap && t <= N && (N % t == 0)) return t;
    }
    return 1;
}

template<int K, typename Element>
consteval int getGateTileK() {
    static_assert(K > 0 && K % 16 == 0);
    constexpr int cap = std::is_same_v<Element, double> ? 32 : 64;
    for (const int t : {128, 96, 64, 48, 32, 24, 16}) {
        if (t <= cap && t <= K && (K % t == 0)) return t;
    }
    return 16;
}

template<int K, int cap>
consteval int getTileK() {
    static_assert(K > 0 && K % 16 == 0);
    for (const int t : {128, 96, 64, 48, 32, 24, 16}) {
        if (t <= cap && t <= K && (K % t == 0)) return t;
    }
    return 16;
}

template<int K, int bK, int Arch>
consteval int getPipeStages() {
    static_assert(K % bK == 0);
    constexpr int cap = (Arch >= 900 ? 3 : Arch >= 800 ? 2 : 1);
    constexpr int stages = K / bK;
    return stages >= cap ? cap : stages;
}

} // namespace heuristics

// ============================================================================
// ReadySignal (from tq.cuh — simple enum, no CUDA deps)
// ============================================================================

#ifndef FLASHMOE_READY_SIGNAL_DEFINED
#define FLASHMOE_READY_SIGNAL_DEFINED
enum ReadySignal : uint {
    observed = 0,
    ready = 1,
    RS_COUNT = 2
};
#endif

// ============================================================================
// MoEArgs
// ============================================================================

struct MoEArgs {
    const size_t elementBytes;
    const uint sequenceLength;
    const size_t EC;
    const uint tokenDim;
    const uint ffnIntermediateSize;
    const uint bM;
    const uint bN0;
    const uint bN1;
    const uint bK0;
    const uint bK1;
    const uint threads;
    const uint blocks;
    const uint smemSize;
    const uint16_t epRank;
    const uint16_t epWorld;
    const uint16_t myPE;
    const size_t numExperts;
    const uint16_t numLocalExperts;
    const Topology topo;

    MoEArgs(const size_t &eb, const uint &S, const uint &H, const uint &I, const size_t &ec,
            const uint &bm, const uint &bn0, const uint &bn1, const uint &bk0, const uint bk1,
            const uint &_threads, const uint &ctas, const uint &sharedSize,
            const uint16_t &ep_rank, const uint16_t &ep_world, const uint16_t &mype,
            const uint16_t &experts, const uint16_t &nlx, const Topology &topo_)
        : elementBytes(eb), sequenceLength(S), EC(ec), tokenDim(H), ffnIntermediateSize(I),
          bM(bm), bN0(bn0), bN1(bn1), bK0(bk0), bK1(bk1),
          threads(_threads), blocks(ctas), smemSize(sharedSize),
          epRank(ep_rank), epWorld(ep_world), myPE(mype),
          numExperts(experts), numLocalExperts(nlx), topo(topo_) {}
};

// ============================================================================
// State-number bookkeeping
// ============================================================================

namespace sbs {
    __host__ __device__ inline uint8_t next(uint8_t current) {
        return static_cast<uint8_t>((current % 254) + 1);
    }
}

} // namespace flashmoe

// ============================================================================
// The remaining functions (initialize, finalize, etc.) require full
// context.hip.cuh types (PEL, PLI, Context, etc.).
// They must be included AFTER all type definitions are available.
// ============================================================================

// Pull in full context (PEL, PLI, ELI, LXI, TPS, TQSignal, BitSet, Context, GateContext)
#include "flashmoe/hip/context.hip.cuh"

#ifdef FLASHMOE_HAS_ROCSHMEM
#include "flashmoe/hip/nvshmem_compat.hpp"
#endif

namespace flashmoe {

// ============================================================================
// Expert-Parallel Bookkeeping
// ============================================================================

__host__ __forceinline__
void expertParallelBookkeeping(
    const int * __restrict__ const &expertToEpRank,
    const int * __restrict__ const &epRankToGlobalRank,
    const uint &epWorld, const int &myPE, const uint &E, const uint &nLx,
    std::byte *const &sHeap, uint64_t *const &signals,
    PEL *const &pel, PLI *const &pli, ELI *const &eli, LXI *const &lxi,
    hipStream_t stream)
{
    std::vector<uint> lxIndices(epWorld, 0);
    std::vector<PEL> pelHost(E);
    std::vector<PLI> pliHost(epWorld);
    std::vector<ELI> eliHost(E);
    std::vector<LXI> lxiHost(nLx);

    for (uint i = 0; i < E; ++i) {
        const auto epRank = expertToEpRank[i];
        const auto pe = epRankToGlobalRank[epRank];
        const uint lxIdx = lxIndices[epRank]++;

#ifdef FLASHMOE_HAS_ROCSHMEM
        auto *rSheap = static_cast<std::byte*>(rocshmem_ptr(sHeap, pe));
        auto *rFlags = static_cast<uint64_t*>(rocshmem_ptr(signals, pe));
        const auto isRemote = (rSheap == nullptr);
#else
        auto *rSheap = sHeap;
        auto *rFlags = signals;
        const auto isRemote = (pe != myPE);
#endif
        pelHost[i].isRemote = isRemote;
        pelHost[i].expertLocalIdx = lxIdx;
        pelHost[i].pe = pe;
        pelHost[i].remoteSFlags = rFlags;
        pelHost[i].remoteSHeap = rSheap;
        pelHost[i].peer = epRank;

        eliHost[i].epRank = epRank;
        eliHost[i].isRemote = isRemote;
        eliHost[i].localExpertIndex = lxIdx;

        pliHost[epRank].isRemote = isRemote;
        pliHost[epRank].pe = pe;
        pliHost[epRank].remoteSFlags = rFlags;
        pliHost[epRank].remoteSHeap = rSheap;

        if (pe == myPE) lxiHost[lxIdx].expertIndex = i;
    }

    const auto nlxUniform = lxIndices[0];
    for (uint i = 0; i < E; ++i) {
        auto pt = pelHost[i];
        pt.nLocalExperts = lxIndices[pt.peer];
        if (pt.nLocalExperts != nlxUniform)
            throw std::runtime_error("Number of local experts should be equal across the ep group");
        pelHost[i] = pt;
    }

    CHECK_HIP_BOOTSTRAP(hipMemcpyAsync(pel, pelHost.data(), sizeof(PEL) * pelHost.size(), hipMemcpyHostToDevice, stream));
    CHECK_HIP_BOOTSTRAP(hipMemcpyAsync(pli, pliHost.data(), sizeof(PLI) * pliHost.size(), hipMemcpyHostToDevice, stream));
    CHECK_HIP_BOOTSTRAP(hipMemcpyAsync(eli, eliHost.data(), sizeof(ELI) * eliHost.size(), hipMemcpyHostToDevice, stream));
    CHECK_HIP_BOOTSTRAP(hipMemcpyAsync(lxi, lxiHost.data(), sizeof(LXI) * lxiHost.size(), hipMemcpyHostToDevice, stream));
}

// ============================================================================
// subscriberTQLength / secondaryTQLength
// (host-side versions for initialize())
// ============================================================================

#ifndef FLASHMOE_TQ_LENGTH_HOST_DEFINED
#define FLASHMOE_TQ_LENGTH_HOST_DEFINED

template<int subscriberWarpSize>
__host__ __forceinline__
constexpr auto subscriberTQLength_host(const int &world, const uint &numLocalExperts,
    const uint &ecTilesM, const uint &E, const uint &tilesN0, const uint &tilesN1,
    const uint &subscriberCount) {
    auto cd = [](auto a, auto b) { return (a + b - 1) / b; };
    const auto dispatchTaskQL = cd(world * numLocalExperts, subscriberCount / subscriberWarpSize) *
        (cd(ecTilesM * tilesN0, subscriberWarpSize) + cd(tilesN0, subscriberWarpSize));
    const auto combineTaskQL = (cd(ecTilesM * E, subscriberCount) * tilesN1) +
        cd(ecTilesM * E * tilesN1, subscriberCount);
    return static_cast<size_t>(dispatchTaskQL + combineTaskQL) * subscriberCount;
}

__host__ __forceinline__
auto secondaryTQLength_host(const int &world, const int &numLocalExperts,
    const uint &ecTilesM, const uint &tilesN1) {
    return static_cast<size_t>(world) * numLocalExperts * ecTilesM * tilesN1;
}

#endif

// ============================================================================
// nSI host helper (if not already included from bitset.hip.cuh or context.hip.cuh)
// ============================================================================

__host__ __forceinline__
constexpr uint nSI_host(const uint &numBits, const uint &T) {
    constexpr uint integerBitWidth = 32U;
    const auto width = integerBitWidth * T;
    return (numBits / width) * T + std::min(numBits % width, T);
}

// ============================================================================
// Initialize MoE Context
// ============================================================================

__host__ __forceinline__
Context initialize(const MoEArgs &args, const int &arch,
                   const int * __restrict__ const &expertToEpRank,
                   const int * __restrict__ const &epRankToGlobalRank,
                   hipStream_t stream)
{
    if (args.tokenDim % args.bK0 != 0 || args.tokenDim % args.bN1 != 0)
        throw std::runtime_error("token dimension should be multiples of tile dimensions");
    if (args.ffnIntermediateSize % args.bN0 != 0 || args.ffnIntermediateSize % args.bK1 != 0)
        throw std::runtime_error("Intermediate size should be multiples of tile dimensions");
    if (args.blocks < 2)
        throw std::runtime_error("blocks must be at least 2");

    const auto processors = args.blocks - 1;
    if (processors > scheduler::MAX_PROCESSORS)
        throw std::runtime_error(
            std::string("processor count: ") + std::to_string(processors) + " is too high");

    auto cd = [](auto a, auto b) { return (a + b - 1) / b; };
    const auto roundEC = cd(args.EC, static_cast<size_t>(args.bM)) * args.bM;
    const auto ecTilesM = cd(roundEC, static_cast<size_t>(args.bM));
    const auto tilesN0 = cd(args.ffnIntermediateSize, args.bN0);
    const auto tilesN1 = cd(args.tokenDim, args.bN1);

    static_assert(SignalConstants::ground == 0);
    const size_t signalLength = (args.epWorld * args.numLocalExperts) +
                                (args.numExperts * ecTilesM * tilesN1);

    uint64_t *signals = nullptr;
    std::byte *sHeap = nullptr;
    const auto heapLength = args.elementBytes * HEAP_STAGES * HEAP_CELLS *
                            static_cast<size_t>(args.epWorld * args.numLocalExperts) *
                            static_cast<size_t>(roundEC * args.tokenDim);

#ifdef FLASHMOE_HAS_ROCSHMEM
    signals = static_cast<uint64_t*>(rocshmem_calloc(signalLength, sizeof(uint64_t)));
    if (!signals) throw std::runtime_error("failed to allocate signals via rocSHMEM");
    sHeap = static_cast<std::byte*>(rocshmem_malloc(heapLength));
    if (!sHeap) throw std::runtime_error("failed to allocate heap via rocSHMEM");
#else
    CHECK_HIP_BOOTSTRAP(hipMallocAsync(reinterpret_cast<void**>(&signals),
                                       sizeof(uint64_t) * signalLength, stream));
    CHECK_HIP_BOOTSTRAP(hipMemsetAsync(signals, 0, sizeof(uint64_t) * signalLength, stream));
    CHECK_HIP_BOOTSTRAP(hipMallocAsync(reinterpret_cast<void**>(&sHeap), heapLength, stream));
#endif

    const bool threadOk = args.threads >= WARP_SIZE * 2 && args.threads % WARP_SIZE == 0;
    if (!threadOk) throw std::runtime_error("threads not supported");
    const auto subscriberCount = args.threads - WARP_SIZE;

    const size_t tQLength = subscriberTQLength_host<WARP_SIZE>(
        args.epWorld, args.numLocalExperts, static_cast<uint>(ecTilesM),
        static_cast<uint>(args.numExperts), tilesN0, tilesN1, subscriberCount);
    const size_t secondaryTQL = secondaryTQLength_host(
        args.epWorld, args.numLocalExperts, static_cast<uint>(ecTilesM), tilesN1);

    Task *tQ = nullptr;
    CHECK_HIP_BOOTSTRAP(hipMallocAsync(&tQ, sizeof(Task) * (tQLength + secondaryTQL), stream));
    Task *pTq = tQ + tQLength;

    std::byte *GEMM0Staging = nullptr;
    const size_t stagingLength = static_cast<size_t>(args.epWorld * args.numLocalExperts * roundEC) *
                                 args.ffnIntermediateSize;
    CHECK_HIP_BOOTSTRAP(hipMallocAsync(reinterpret_cast<void**>(&GEMM0Staging),
                                       stagingLength * args.elementBytes, stream));

    BitSet *consumerBitMap = nullptr;
    const auto cbmLength = nSI_host(static_cast<uint>(args.numExperts * ecTilesM * tilesN1), subscriberCount);
    CHECK_HIP_BOOTSTRAP(hipMallocAsync(&consumerBitMap, sizeof(BitSet) * cbmLength, stream));
    CHECK_HIP_BOOTSTRAP(hipMemsetAsync(consumerBitMap, 0, sizeof(BitSet) * cbmLength, stream));

    uint8_t *producerBitMap = nullptr;
    const auto pbmLength = args.epWorld * args.numLocalExperts * ecTilesM * tilesN1;
    CHECK_HIP_BOOTSTRAP(hipMallocAsync(&producerBitMap, sizeof(uint8_t) * pbmLength, stream));
    CHECK_HIP_BOOTSTRAP(hipMemsetAsync(producerBitMap, 0, sizeof(uint8_t) * pbmLength, stream));

    PEL *pel = nullptr;
    CHECK_HIP_BOOTSTRAP(hipMallocAsync(&pel, sizeof(PEL) * args.numExperts, stream));
    PLI *pli = nullptr;
    CHECK_HIP_BOOTSTRAP(hipMallocAsync(&pli, sizeof(PLI) * args.epWorld, stream));
    ELI *eli = nullptr;
    CHECK_HIP_BOOTSTRAP(hipMallocAsync(&eli, sizeof(ELI) * args.numExperts, stream));
    LXI *lxi = nullptr;
    CHECK_HIP_BOOTSTRAP(hipMallocAsync(&lxi, sizeof(LXI) * args.numLocalExperts, stream));
    TPS *tps = nullptr;
    CHECK_HIP_BOOTSTRAP(hipMallocAsync(&tps, sizeof(TPS) * args.numExperts * roundEC, stream));
    TQSignal *tqs = nullptr;
    CHECK_HIP_BOOTSTRAP(hipMallocAsync(&tqs, sizeof(TQSignal) * processors, stream));
    CHECK_HIP_BOOTSTRAP(hipMemsetAsync(tqs, 0, sizeof(TQSignal) * processors, stream));
    uint *dispatchSync = nullptr;
    CHECK_HIP_BOOTSTRAP(hipMallocAsync(&dispatchSync, sizeof(uint) * args.numExperts, stream));
    CHECK_HIP_BOOTSTRAP(hipMemsetAsync(dispatchSync, 0, sizeof(uint) * args.numExperts, stream));
    uint *gtqHeads = nullptr;
    const size_t gtqHeadsLength = args.epWorld * args.numLocalExperts * ecTilesM;
    CHECK_HIP_BOOTSTRAP(hipMallocAsync(&gtqHeads, sizeof(uint) * gtqHeadsLength, stream));
    CHECK_HIP_BOOTSTRAP(hipMemsetAsync(gtqHeads, 0, sizeof(uint) * gtqHeadsLength, stream));
    uint *tileSync = nullptr;
    CHECK_HIP_BOOTSTRAP(hipMallocAsync(&tileSync, sizeof(uint) * gtqHeadsLength, stream));
    CHECK_HIP_BOOTSTRAP(hipMemsetAsync(tileSync, 0, sizeof(uint) * gtqHeadsLength, stream));
    uint *statusQ = nullptr;
    CHECK_HIP_BOOTSTRAP(hipMallocAsync(&statusQ, sizeof(uint) * processors, stream));
    CHECK_HIP_BOOTSTRAP(hipMemsetAsync(statusQ, 0, sizeof(uint) * processors, stream));
    uint8_t *stateNumbers = nullptr;
    CHECK_HIP_BOOTSTRAP(hipMallocAsync(&stateNumbers, sizeof(uint8_t) * args.blocks, stream));
    static_assert(SignalConstants::sequenceStart == 0x01);
    CHECK_HIP_BOOTSTRAP(hipMemsetAsync(stateNumbers, 0x01, sizeof(uint8_t) * args.blocks, stream));
    CHECK_HIP_BOOTSTRAP(hipPeekAtLastError());

    expertParallelBookkeeping(expertToEpRank, epRankToGlobalRank,
        args.epWorld, args.myPE, static_cast<uint>(args.numExperts),
        args.numLocalExperts, sHeap, signals, pel, pli, eli, lxi, stream);

    return Context{
        .symHeap = sHeap, .signals = signals, .tQ = tQ, .pTq = pTq,
        .GEMM0Staging = GEMM0Staging,
        .consumerCombineBitMap = consumerBitMap, .producerCombineBitMap = producerBitMap,
        .pel = pel, .pli = pli, .eli = eli, .lxi = lxi, .tqs = tqs,
        .dispatchSync = dispatchSync, .gTqHeads = gtqHeads, .tileSync = tileSync,
        .statusQueue = statusQ, .tokenIndices = tps,
        .processors_v = cuda::fast_mod_div<uint>(processors),
        .blocks = args.blocks, .smemSize = args.smemSize,
        .S = args.sequenceLength, .H = args.tokenDim, .I = args.ffnIntermediateSize,
        .EC = static_cast<uint>(args.EC),
        .bM = static_cast<uint16_t>(args.bM), .bN0 = static_cast<uint16_t>(args.bN0),
        .bN1 = static_cast<uint16_t>(args.bN1),
        .nLx = args.numLocalExperts, .E = static_cast<uint16_t>(args.numExperts),
        .world = args.epWorld, .epRank = args.epRank, .myPE = args.myPE,
        .topo = args.topo,
        .stateNumbers = stateNumbers
    };
}

// ============================================================================
// Initialize Gate Context
// ============================================================================

__host__ __forceinline__
GateContext initializeGate(const uint &bNGate, const uint &numExperts, const uint &S, hipStream_t stream) {
    int *ecGuards = nullptr;
    CHECK_HIP_BOOTSTRAP(hipMallocAsync(&ecGuards, sizeof(int) * numExperts, stream));
    CHECK_HIP_BOOTSTRAP(hipMemsetAsync(ecGuards, flashmoe::STALE_AS_BYTE, sizeof(int) * numExperts, stream));

    SoftmaxStatePacked *ssp = nullptr;
    RingTopKPayload *rtp = nullptr;
    auto cd = [](auto a, auto b) { return (a + b - 1) / b; };
    if (numExperts > bNGate) {
        const auto tE = cd(numExperts, bNGate);
        CHECK_HIP_BOOTSTRAP(hipMallocAsync(&ssp, sizeof(SoftmaxStatePacked) * S * tE, stream));
        CHECK_HIP_BOOTSTRAP(hipMemsetAsync(ssp, 0, sizeof(SoftmaxStatePacked) * S * tE, stream));
        CHECK_HIP_BOOTSTRAP(hipMallocAsync(&rtp, 2 * sizeof(RingTopKPayload) * S * tE, stream));
        CHECK_HIP_BOOTSTRAP(hipMemsetAsync(rtp, 0, 2 * sizeof(RingTopKPayload) * S * tE, stream));
    }
    return GateContext{ecGuards, ssp, rtp};
}

// ============================================================================
// Finalize Gate / MoE Context
// ============================================================================

__host__ __forceinline__
void finalizeGate(const GateContext &gCtx, hipStream_t stream) {
    CHECK_HIP_BOOTSTRAP(hipFreeAsync(gCtx.ecGuards, stream));
    if (gCtx.ssp) CHECK_HIP_BOOTSTRAP(hipFreeAsync(gCtx.ssp, stream));
    if (gCtx.rtp) CHECK_HIP_BOOTSTRAP(hipFreeAsync(gCtx.rtp, stream));
    CHECK_HIP_BOOTSTRAP(hipPeekAtLastError());
    CHECK_HIP_BOOTSTRAP(hipStreamSynchronize(stream));
}

__host__ __forceinline__
void finalize(const Context &ctx, hipStream_t stream) {
    CHECK_HIP_BOOTSTRAP(hipFreeAsync(ctx.tQ, stream));
    CHECK_HIP_BOOTSTRAP(hipFreeAsync(ctx.stateNumbers, stream));
    CHECK_HIP_BOOTSTRAP(hipFreeAsync(ctx.GEMM0Staging, stream));
    CHECK_HIP_BOOTSTRAP(hipFreeAsync(ctx.pel, stream));
    CHECK_HIP_BOOTSTRAP(hipFreeAsync(ctx.pli, stream));
    CHECK_HIP_BOOTSTRAP(hipFreeAsync(ctx.eli, stream));
    CHECK_HIP_BOOTSTRAP(hipFreeAsync(ctx.lxi, stream));
    CHECK_HIP_BOOTSTRAP(hipFreeAsync(ctx.consumerCombineBitMap, stream));
    CHECK_HIP_BOOTSTRAP(hipFreeAsync(ctx.producerCombineBitMap, stream));
    CHECK_HIP_BOOTSTRAP(hipFreeAsync(ctx.tokenIndices, stream));
    CHECK_HIP_BOOTSTRAP(hipFreeAsync(ctx.tqs, stream));
    CHECK_HIP_BOOTSTRAP(hipFreeAsync(ctx.dispatchSync, stream));
    CHECK_HIP_BOOTSTRAP(hipFreeAsync(ctx.gTqHeads, stream));
    CHECK_HIP_BOOTSTRAP(hipFreeAsync(ctx.tileSync, stream));
    CHECK_HIP_BOOTSTRAP(hipFreeAsync(ctx.statusQueue, stream));
    CHECK_HIP_BOOTSTRAP(hipStreamSynchronize(stream));
#ifdef FLASHMOE_HAS_ROCSHMEM
    rocshmem_free(ctx.symHeap);
    rocshmem_free(ctx.signals);
#else
    CHECK_HIP_BOOTSTRAP(hipFreeAsync(ctx.symHeap, stream));
    CHECK_HIP_BOOTSTRAP(hipFreeAsync(ctx.signals, stream));
#endif
    CHECK_HIP_BOOTSTRAP(hipPeekAtLastError());
    CHECK_HIP_BOOTSTRAP(hipStreamSynchronize(stream));
}

} // namespace flashmoe

#endif // FLASHMOE_HIP_BOOTSTRAP_CUH
