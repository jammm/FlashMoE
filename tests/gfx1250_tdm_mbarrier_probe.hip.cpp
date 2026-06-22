#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_gfx1250_TDM.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_HIP(call)                                                        \
    do {                                                                       \
        hipError_t err = (call);                                               \
        if (err != hipSuccess) {                                               \
            std::cerr << "HIP error: " << hipGetErrorString(err)               \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";       \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

namespace {

using v4i = int __attribute__((ext_vector_type(4)));
using v8i = int __attribute__((ext_vector_type(8)));

constexpr int kRows = 8;
constexpr int kCols = 8;
constexpr int kElems = kRows * kCols;

__device__ inline unsigned barrierPhase(volatile long* barrier) {
    auto state = static_cast<unsigned long long>(*barrier);
    return static_cast<unsigned>((state >> 29) & 1ULL);
}

__device__ inline void waitBarrierPhaseChange(volatile long* barrier,
                                              unsigned phase) {
    while (barrierPhase(barrier) == phase) {
        __builtin_amdgcn_s_sleep(1);
    }
}

__device__ inline gfx1250_TDM_GROUP1 make2dGroup1() {
    gfx1250_TDM_GROUP1 group1;
    group1.dataSize(2); // log2(sizeof(int))
    group1.tensorDim0(kRows);
    group1.tensorDim1(kCols);
    group1.tensorDim0Stride(kRows);
    group1.tensorDim1Stride(kCols);
    group1.tileDim0(kRows);
    group1.tileDim1(kCols);
    return group1;
}

__global__ void tdmLoadStoreKernel(const int* input, int* output, int* state) {
#if defined(__gfx1250__) || defined(__gfx1251__)
    __shared__ int smem[kElems];

    gfx1250_TDM_GROUP0 group0;
    group0.globalAddr(reinterpret_cast<uintptr_t>(input));
    group0.ldsAddr(reinterpret_cast<uintptr_t>(smem));

    auto group1 = make2dGroup1();
    v4i v4Zeros{0, 0, 0, 0};
    v8i v8Zeros{0, 0, 0, 0, 0, 0, 0, 0};

    __builtin_amdgcn_tensor_load_to_lds(group0.m_bitfield, group1.m_bitfield,
                                        v4Zeros, v4Zeros, v8Zeros, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    __syncthreads();

    group0.globalAddr(reinterpret_cast<uintptr_t>(output));
    __builtin_amdgcn_tensor_store_from_lds(group0.m_bitfield, group1.m_bitfield,
                                           v4Zeros, v4Zeros, v8Zeros, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);

    if (threadIdx.x == 0) {
        state[0] = 1;
    }
#else
    if (threadIdx.x == 0) {
        state[0] = -1;
    }
#endif
}

__global__ void manualMbarrierKernel(int* state) {
#if defined(__gfx1250__) || defined(__gfx1251__)
    __shared__ long barrier;

    if (threadIdx.x == 0) {
        barrier = 0; // init count 1: ((1 - 1) << 32) | (1 - 1)
        state[0] = -1;
        state[1] = -1;
        state[2] = -1;
        state[3] = -1;
    }
    __syncthreads();

    const int waveId = threadIdx.x / warpSize;
    const int laneId = threadIdx.x % warpSize;

    if (waveId == 1 && laneId == 0) {
        waitBarrierPhaseChange(&barrier, 0);
        state[2] = static_cast<int>(barrierPhase(&barrier));
        state[3] = 1;
    }

    if (waveId == 0 && laneId == 0) {
        for (int i = 0; i < 64; ++i) {
            __builtin_amdgcn_s_sleep(1);
        }
        long prior = __builtin_amdgcn_ds_atomic_barrier_arrive_rtn_b64(&barrier, 1);
        state[0] = static_cast<int>((static_cast<unsigned long long>(prior) >> 29) & 1ULL);
        state[1] = static_cast<int>(prior & 0x1fffffffULL);
    }
#else
    if (threadIdx.x == 0) {
        state[0] = -1;
    }
#endif
}

__global__ void tdmMbarrierKernel(const int* input, int* output, int* state) {
#if defined(__gfx1250__) || defined(__gfx1251__)
    __shared__ int smem[kElems];
    __shared__ long barrier;

    if (threadIdx.x == 0) {
        barrier = 0; // init count 1
        state[0] = -1;
        state[1] = -1;
        state[2] = -1;
    }
    __syncthreads();

    gfx1250_TDM_GROUP0 group0;
    group0.globalAddr(reinterpret_cast<uintptr_t>(input));
    group0.ldsAddr(reinterpret_cast<uintptr_t>(smem));

    auto group1 = make2dGroup1();
    group1.atomicBarrierEnable(true);
    group1.atomicBarrierAddress(
        static_cast<uint32_t>((reinterpret_cast<uintptr_t>(&barrier) >> 3) & 0xffffU));

    v4i v4Zeros{0, 0, 0, 0};
    v8i v8Zeros{0, 0, 0, 0, 0, 0, 0, 0};

    __builtin_amdgcn_tensor_load_to_lds(group0.m_bitfield, group1.m_bitfield,
                                        v4Zeros, v4Zeros, v8Zeros, 0);

    waitBarrierPhaseChange(&barrier, 0);

    for (int i = threadIdx.x; i < kElems; i += blockDim.x) {
        output[i] = smem[i];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        state[0] = static_cast<int>(barrierPhase(&barrier));
        state[1] = static_cast<int>(barrier & 0x1fffffffL);
        state[2] = 1;
    }
#else
    if (threadIdx.x == 0) {
        state[0] = -1;
    }
#endif
}

bool checkVector(const std::vector<int>& got, const std::vector<int>& expected,
                 const char* label) {
    bool ok = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (got[i] != expected[i]) {
            if (ok) {
                std::cerr << label << " mismatches:\n";
            }
            std::cerr << "  [" << i << "] got " << got[i]
                      << ", expected " << expected[i] << "\n";
            ok = false;
        }
    }
    return ok;
}

bool runTdmLoadStore() {
    std::vector<int> input(kElems);
    std::vector<int> output(kElems, -1);
    std::vector<int> expected(kElems);
    for (int i = 0; i < kElems; ++i) {
        input[i] = i * 3 + 7;
        expected[i] = input[i];
    }

    int* dInput = nullptr;
    int* dOutput = nullptr;
    int* dState = nullptr;
    CHECK_HIP(hipMalloc(&dInput, kElems * sizeof(int)));
    CHECK_HIP(hipMalloc(&dOutput, kElems * sizeof(int)));
    CHECK_HIP(hipMalloc(&dState, 4 * sizeof(int)));
    CHECK_HIP(hipMemcpy(dInput, input.data(), kElems * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(dOutput, 0xff, kElems * sizeof(int)));
    CHECK_HIP(hipMemset(dState, 0xff, 4 * sizeof(int)));

    tdmLoadStoreKernel<<<1, 32>>>(dInput, dOutput, dState);
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<int> state(4);
    CHECK_HIP(hipMemcpy(output.data(), dOutput, kElems * sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(state.data(), dState, 4 * sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(dInput));
    CHECK_HIP(hipFree(dOutput));
    CHECK_HIP(hipFree(dState));

    bool ok = state[0] == 1 && checkVector(output, expected, "TDM load/store");
    std::cout << "raw TDM load/store: " << (ok ? "pass" : "fail")
              << " state=" << state[0] << "\n";
    return ok;
}

bool runManualMbarrier() {
    int* dState = nullptr;
    CHECK_HIP(hipMalloc(&dState, 4 * sizeof(int)));
    CHECK_HIP(hipMemset(dState, 0xff, 4 * sizeof(int)));

    manualMbarrierKernel<<<1, 64>>>(dState);
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<int> state(4);
    CHECK_HIP(hipMemcpy(state.data(), dState, 4 * sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(dState));

    bool ok = state[0] == 0 && state[1] == 0 && state[2] == 1 && state[3] == 1;
    std::cout << "manual LDS mbarrier: " << (ok ? "pass" : "fail")
              << " prior_phase=" << state[0]
              << " prior_pending=" << state[1]
              << " waiter_phase=" << state[2]
              << " waiter_seen=" << state[3] << "\n";
    return ok;
}

bool runTdmMbarrier() {
    std::vector<int> input(kElems);
    std::vector<int> output(kElems, -1);
    std::vector<int> expected(kElems);
    for (int i = 0; i < kElems; ++i) {
        input[i] = i * 5 + 11;
        expected[i] = input[i];
    }

    int* dInput = nullptr;
    int* dOutput = nullptr;
    int* dState = nullptr;
    CHECK_HIP(hipMalloc(&dInput, kElems * sizeof(int)));
    CHECK_HIP(hipMalloc(&dOutput, kElems * sizeof(int)));
    CHECK_HIP(hipMalloc(&dState, 4 * sizeof(int)));
    CHECK_HIP(hipMemcpy(dInput, input.data(), kElems * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(dOutput, 0xff, kElems * sizeof(int)));
    CHECK_HIP(hipMemset(dState, 0xff, 4 * sizeof(int)));

    tdmMbarrierKernel<<<1, 32>>>(dInput, dOutput, dState);
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<int> state(4);
    CHECK_HIP(hipMemcpy(output.data(), dOutput, kElems * sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(state.data(), dState, 4 * sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(dInput));
    CHECK_HIP(hipFree(dOutput));
    CHECK_HIP(hipFree(dState));

    bool ok = state[0] == 1 && state[1] == 0 && state[2] == 1 &&
              checkVector(output, expected, "TDM mbarrier");
    std::cout << "TDM completion mbarrier: " << (ok ? "pass" : "fail")
              << " phase=" << state[0]
              << " pending=" << state[1]
              << " copied=" << state[2] << "\n";
    return ok;
}

} // namespace

int main() {
    int device = 0;
    hipDeviceProp_t prop{};
    CHECK_HIP(hipGetDevice(&device));
    CHECK_HIP(hipGetDeviceProperties(&prop, device));
    const std::string arch = prop.gcnArchName;
    if (arch.find("gfx1250") == std::string::npos &&
        arch.find("gfx1251") == std::string::npos) {
        std::cerr << "This probe requires gfx1250/gfx1251, got "
                  << prop.gcnArchName << "\n";
        return 77;
    }

    bool ok = true;
    ok = runTdmLoadStore() && ok;
    ok = runManualMbarrier() && ok;
    ok = runTdmMbarrier() && ok;

    std::cout << "gfx1250 TDM/mbarrier probe: " << (ok ? "pass" : "fail")
              << "\n";
    return ok ? 0 : 1;
}
