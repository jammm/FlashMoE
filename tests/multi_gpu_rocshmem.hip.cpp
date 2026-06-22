/******************************************************************************
 * Multi-GPU rocSHMEM Test
 *
 * Tests PE-to-PE communication using rocSHMEM with IPC backend.
 * Multiple PEs may be mapped to the same visible GPU for single-GPU smoke tests.
 * Uses stream-based APIs which work without MPI support.
 *
 * Build:
 *   source /jam/moe/venv/bin/activate
 *   export ROCM_PATH=/jam/moe/venv/lib/python3.12/site-packages/_rocm_sdk_devel
 *   export ROCSHMEM_ROOT=/jam/moe/rocm-systems/projects/rocshmem/install
 *   hipcc -std=c++20 --offload-arch=gfx1250 \
 *       -I$ROCM_PATH/include -I$ROCSHMEM_ROOT/include \
 *       -L$ROCSHMEM_ROOT/lib -L$ROCM_PATH/lib \
 *       -o tests/multi_gpu_test tests/multi_gpu_rocshmem.hip.cpp \
 *       -lrocshmem -lamdhip64 -lhsa-runtime64 -lpthread
 *
 * Run:
 *   export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCSHMEM_ROOT/lib:$LD_LIBRARY_PATH"
 *   ./tests/multi_gpu_test
 *****************************************************************************/

#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <atomic>

using namespace rocshmem;

#define CHECK_HIP(condition) do {                                         \
    hipError_t error = condition;                                         \
    if (error != hipSuccess) {                                            \
        fprintf(stderr, "HIP error: %s (code %d) at line %d\n",           \
                hipGetErrorString(error), error, __LINE__);               \
        fflush(stderr);                                                   \
        exit(error);                                                      \
    }                                                                     \
} while(0)

struct SharedBootstrap {
    rocshmem_uniqueid_t uid;
    std::atomic<int> uid_ready;
    std::atomic<int> ready_count;
    int test_passed[8];
};

// Kernel: Initialize data with pattern
__global__ void init_data_kernel(float* data, int nelems, float base_value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nelems) {
        data[tid] = base_value + tid;
    }
}

// Kernel: Verify data matches expected pattern
__global__ void verify_data_kernel(float* data, int nelems, float expected_base, int* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        *result = 1;
        for (int i = 0; i < nelems; i++) {
            float expected = expected_base + i;
            if (fabsf(data[i] - expected) > 0.001f) {
                *result = 0;
                printf("  Verify failed at %d: expected %f, got %f\n", i, expected, data[i]);
                break;
            }
        }
    }
}

void run_pe(int rank, int npes, SharedBootstrap* shared) {
    int ret;

    printf("[PE %d] Starting (PID %d)\n", rank, getpid());
    fflush(stdout);

    int device_count = 0;
    CHECK_HIP(hipGetDeviceCount(&device_count));
    if (device_count <= 0) {
        fprintf(stderr, "[PE %d] No HIP devices available\n", rank);
        exit(1);
    }

    // Allow multi-PE smoke testing on a single physical GPU.
    const int device_id = rank % device_count;
    CHECK_HIP(hipSetDevice(device_id));

    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, device_id));
    printf("[PE %d] Device %d/%d: %s\n", rank, device_id, device_count, props.name);
    fflush(stdout);

    // Create stream for rocSHMEM operations
    hipStream_t stream;
    CHECK_HIP(hipStreamCreate(&stream));

    // PE 0 generates unique ID for bootstrap
    if (rank == 0) {
        ret = rocshmem_get_uniqueid(&shared->uid);
        if (ret != ROCSHMEM_SUCCESS) {
            fprintf(stderr, "[PE %d] Error getting uniqueid: %d\n", rank, ret);
            exit(1);
        }
        shared->uid_ready.store(1);
    } else {
        while (shared->uid_ready.load() == 0) usleep(1000);
    }

    // Initialize rocSHMEM
    rocshmem_init_attr_t attr;
    ret = rocshmem_set_attr_uniqueid_args(rank, npes, &shared->uid, &attr);
    if (ret != ROCSHMEM_SUCCESS) {
        fprintf(stderr, "[PE %d] Error set_attr: %d\n", rank, ret);
        exit(1);
    }

    ret = rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &attr);
    if (ret != ROCSHMEM_SUCCESS) {
        fprintf(stderr, "[PE %d] Error init_attr: %d\n", rank, ret);
        exit(1);
    }

    int my_pe = rocshmem_my_pe();
    int n_pes = rocshmem_n_pes();
    printf("[PE %d] rocSHMEM initialized: my_pe=%d, n_pes=%d\n", rank, my_pe, n_pes);
    fflush(stdout);

    // Allocate symmetric memory
    const int NELEMS = 256;
    float* recv_buf = (float*)rocshmem_malloc(NELEMS * sizeof(float));
    float* send_buf = (float*)rocshmem_malloc(NELEMS * sizeof(float));
    int* verify_result;
    CHECK_HIP(hipMalloc(&verify_result, sizeof(int)));

    if (!recv_buf || !send_buf) {
        fprintf(stderr, "[PE %d] rocshmem_malloc failed\n", rank);
        exit(1);
    }

    CHECK_HIP(hipMemset(recv_buf, 0, NELEMS * sizeof(float)));
    CHECK_HIP(hipMemset(send_buf, 0, NELEMS * sizeof(float)));
    CHECK_HIP(hipDeviceSynchronize());

    // Wait for all PEs
    shared->ready_count.fetch_add(1);
    while (shared->ready_count.load() < npes) usleep(1000);

    rocshmem_barrier_all_on_stream(stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    // ============ TEST 1: Basic Put ============
    printf("[PE %d] Test 1: Basic Put Operation\n", rank);
    fflush(stdout);

    int target_pe = (my_pe + 1) % n_pes;
    int source_pe = (my_pe - 1 + n_pes) % n_pes;

    // Initialize send buffer
    float base = (my_pe + 1) * 100.0f;
    init_data_kernel<<<1, 256, 0, stream>>>(send_buf, NELEMS, base);
    CHECK_HIP(hipStreamSynchronize(stream));

    rocshmem_barrier_all_on_stream(stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    // Put to next PE (using stream-based API)
    rocshmem_putmem_on_stream(recv_buf, send_buf, NELEMS * sizeof(float), target_pe, stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    rocshmem_barrier_all_on_stream(stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    // Verify received data
    float expected_base = (source_pe + 1) * 100.0f;
    verify_data_kernel<<<1, 1, 0, stream>>>(recv_buf, NELEMS, expected_base, verify_result);
    CHECK_HIP(hipStreamSynchronize(stream));

    int result;
    CHECK_HIP(hipMemcpy(&result, verify_result, sizeof(int), hipMemcpyDeviceToHost));
    shared->test_passed[rank] = result;

    if (result) {
        printf("[PE %d] Test 1 PASSED: Put from PE %d->%d verified\n", rank, source_pe, my_pe);
    } else {
        printf("[PE %d] Test 1 FAILED\n", rank);
    }
    fflush(stdout);

    rocshmem_barrier_all_on_stream(stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    // ============ TEST 2: Get Operation ============
    printf("[PE %d] Test 2: Get Operation\n", rank);
    fflush(stdout);

    // Reset buffers
    CHECK_HIP(hipMemset(recv_buf, 0, NELEMS * sizeof(float)));
    float base2 = (my_pe + 1) * 1000.0f;
    init_data_kernel<<<1, 256, 0, stream>>>(send_buf, NELEMS, base2);
    CHECK_HIP(hipStreamSynchronize(stream));

    rocshmem_barrier_all_on_stream(stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    // Get from next PE
    rocshmem_getmem_on_stream(recv_buf, send_buf, NELEMS * sizeof(float), target_pe, stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    rocshmem_barrier_all_on_stream(stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    // Verify (should have target PE's data)
    float expected_base2 = (target_pe + 1) * 1000.0f;
    verify_data_kernel<<<1, 1, 0, stream>>>(recv_buf, NELEMS, expected_base2, verify_result);
    CHECK_HIP(hipStreamSynchronize(stream));

    CHECK_HIP(hipMemcpy(&result, verify_result, sizeof(int), hipMemcpyDeviceToHost));

    if (result) {
        printf("[PE %d] Test 2 PASSED: Get from PE %d verified\n", rank, target_pe);
    } else {
        printf("[PE %d] Test 2 FAILED\n", rank);
        shared->test_passed[rank] = 0;
    }
    fflush(stdout);

    rocshmem_barrier_all_on_stream(stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    // ============ TEST 3: Put with Signal ============
    printf("[PE %d] Test 3: Put with Signal\n", rank);
    fflush(stdout);

    // Allocate signal
    uint64_t* signal = (uint64_t*)rocshmem_malloc(sizeof(uint64_t));
    CHECK_HIP(hipMemset(signal, 0, sizeof(uint64_t)));
    CHECK_HIP(hipMemset(recv_buf, 0, NELEMS * sizeof(float)));

    float base3 = (my_pe + 1) * 10000.0f;
    init_data_kernel<<<1, 256, 0, stream>>>(send_buf, NELEMS, base3);
    CHECK_HIP(hipStreamSynchronize(stream));

    rocshmem_barrier_all_on_stream(stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    // Put with signal
    rocshmem_putmem_signal_on_stream(recv_buf, send_buf, NELEMS * sizeof(float),
                                      signal, 1, ROCSHMEM_SIGNAL_SET, target_pe, stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    // Wait for signal
    rocshmem_signal_wait_until_on_stream(signal, ROCSHMEM_CMP_EQ, 1, stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    rocshmem_barrier_all_on_stream(stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    // Verify
    float expected_base3 = (source_pe + 1) * 10000.0f;
    verify_data_kernel<<<1, 1, 0, stream>>>(recv_buf, NELEMS, expected_base3, verify_result);
    CHECK_HIP(hipStreamSynchronize(stream));

    CHECK_HIP(hipMemcpy(&result, verify_result, sizeof(int), hipMemcpyDeviceToHost));

    if (result) {
        printf("[PE %d] Test 3 PASSED: Put with signal verified\n", rank);
    } else {
        printf("[PE %d] Test 3 FAILED\n", rank);
        shared->test_passed[rank] = 0;
    }
    fflush(stdout);

    rocshmem_barrier_all_on_stream(stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    // Cleanup
    rocshmem_free(signal);
    CHECK_HIP(hipStreamDestroy(stream));
    CHECK_HIP(hipFree(verify_result));
    rocshmem_free(send_buf);
    rocshmem_free(recv_buf);

    printf("[PE %d] Finalizing rocSHMEM...\n", rank);
    fflush(stdout);
    rocshmem_finalize();
    printf("[PE %d] Done\n", rank);
    fflush(stdout);
}

int main(int argc, char** argv) {
    printf("===========================================\n");
    printf("Multi-PE rocSHMEM Communication Test\n");
    printf("===========================================\n");
    printf("rocSHMEM Version: %s\n", rocshmem::VERSION);
    fflush(stdout);

    int npes = 2;
    printf("Testing with %d PEs (GPUs)\n\n", npes);

    // Create shared memory for bootstrap (before any HIP calls)
    SharedBootstrap* shared = (SharedBootstrap*)mmap(NULL, sizeof(SharedBootstrap),
                                                      PROT_READ | PROT_WRITE,
                                                      MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (shared == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    memset(shared, 0, sizeof(SharedBootstrap));
    new (&shared->uid_ready) std::atomic<int>(0);
    new (&shared->ready_count) std::atomic<int>(0);
    for (int i = 0; i < npes; i++) shared->test_passed[i] = 1;

    // Fork processes (before HIP initialization)
    pid_t pids[8];
    for (int i = 0; i < npes; i++) {
        pids[i] = fork();
        if (pids[i] == 0) {
            run_pe(i, npes, shared);
            exit(0);
        } else if (pids[i] < 0) {
            perror("fork");
            return 1;
        }
    }

    // Wait for children
    int status;
    int all_passed = 1;
    for (int i = 0; i < npes; i++) {
        waitpid(pids[i], &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            printf("PE %d exited with error (status: 0x%x)\n", i, status);
            all_passed = 0;
        }
        if (!shared->test_passed[i]) all_passed = 0;
    }

    printf("\n===========================================\n");
    if (all_passed) {
        printf("ALL TESTS PASSED\n");
        printf("\nPE-to-PE communication via rocSHMEM IPC works!\n");
    } else {
        printf("SOME TESTS FAILED\n");
    }
    printf("===========================================\n");

    munmap(shared, sizeof(SharedBootstrap));
    return all_passed ? 0 : 1;
}
