/******************************************************************************
 * Multi-GPU HIP IPC Test
 *
 * Tests GPU-to-GPU communication using HIP IPC (Inter-Process Communication).
 * This is a simpler test that doesn't require rocSHMEM.
 *****************************************************************************/

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <atomic>

#define CHECK_HIP(call) do {                                              \
    hipError_t err = call;                                                \
    if (err != hipSuccess) {                                              \
        fprintf(stderr, "[PID %d] HIP error: %s (code %d) at line %d\n", \
                getpid(), hipGetErrorString(err), err, __LINE__);         \
        exit(1);                                                          \
    }                                                                     \
} while(0)

// Shared structure for IPC handle exchange
struct SharedData {
    hipIpcMemHandle_t ipc_handles[2];  // IPC handles for each GPU's buffer
    std::atomic<int> ready_count;
    std::atomic<int> handles_ready;
    int test_passed[2];
    int num_gpus;
    int peer_access[2];
};

// Kernel to initialize data
__global__ void init_data(float* data, int n, float base) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] = base + tid;
    }
}

// Kernel to verify data
__global__ void verify_data(float* data, int n, float expected_base, int* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        *result = 1;
        for (int i = 0; i < n; i++) {
            if (fabsf(data[i] - (expected_base + i)) > 0.001f) {
                printf("  Verify failed at %d: expected %f, got %f\n",
                       i, expected_base + i, data[i]);
                *result = 0;
                break;
            }
        }
    }
}

// Kernel to copy data (GPU-to-GPU via IPC)
__global__ void copy_data(float* dst, float* src, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        dst[tid] = src[tid];
    }
}

void run_gpu_test(int gpu_id, int other_gpu_id, SharedData* shared) {
    printf("[GPU %d] Starting test (PID %d)\n", gpu_id, getpid());

    // Initialize HIP for this process (after fork)
    CHECK_HIP(hipSetDevice(gpu_id));

    // Get device properties
    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, gpu_id));
    printf("[GPU %d] Device: %s\n", gpu_id, props.name);

    const int N = 1024;
    const size_t size = N * sizeof(float);

    // Allocate GPU memory
    float* my_buffer;
    CHECK_HIP(hipMalloc(&my_buffer, size));
    printf("[GPU %d] Allocated %zu bytes at %p\n", gpu_id, size, my_buffer);

    // Initialize with unique pattern
    float base = (gpu_id + 1) * 100.0f;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    init_data<<<blocks, threads>>>(my_buffer, N, base);
    CHECK_HIP(hipDeviceSynchronize());

    printf("[GPU %d] Initialized buffer with base value %f\n", gpu_id, base);

    // Get IPC handle for our buffer
    CHECK_HIP(hipIpcGetMemHandle(&shared->ipc_handles[gpu_id], my_buffer));
    printf("[GPU %d] Got IPC handle\n", gpu_id);

    // Signal that our handle is ready
    shared->handles_ready.fetch_add(1);

    // Wait for other GPU's handle to be ready
    while (shared->handles_ready.load() < 2) {
        usleep(1000);
    }

    printf("[GPU %d] All handles ready, opening remote handle\n", gpu_id);

    // Open handle to other GPU's memory
    float* other_buffer;
    CHECK_HIP(hipIpcOpenMemHandle((void**)&other_buffer,
                                   shared->ipc_handles[other_gpu_id],
                                   hipIpcMemLazyEnablePeerAccess));
    printf("[GPU %d] Opened IPC handle to GPU %d's buffer at %p\n", gpu_id, other_gpu_id, other_buffer);

    // Allocate local buffer for copy
    float* local_copy;
    CHECK_HIP(hipMalloc(&local_copy, size));
    CHECK_HIP(hipMemset(local_copy, 0, size));

    // Copy data from other GPU's buffer to local buffer via kernel
    copy_data<<<blocks, threads>>>(local_copy, other_buffer, N);
    CHECK_HIP(hipDeviceSynchronize());

    printf("[GPU %d] Copied data from GPU %d\n", gpu_id, other_gpu_id);

    // Verify the copied data
    int* result;
    CHECK_HIP(hipMalloc(&result, sizeof(int)));

    float expected_base = (other_gpu_id + 1) * 100.0f;
    verify_data<<<1, 1>>>(local_copy, N, expected_base, result);
    CHECK_HIP(hipDeviceSynchronize());

    int host_result;
    CHECK_HIP(hipMemcpy(&host_result, result, sizeof(int), hipMemcpyDeviceToHost));

    shared->test_passed[gpu_id] = host_result;

    if (host_result) {
        printf("[GPU %d] PASSED: Successfully read data from GPU %d (base=%f)\n",
               gpu_id, other_gpu_id, expected_base);
    } else {
        printf("[GPU %d] FAILED: Data verification failed\n", gpu_id);
    }

    // Cleanup
    CHECK_HIP(hipIpcCloseMemHandle(other_buffer));
    CHECK_HIP(hipFree(result));
    CHECK_HIP(hipFree(local_copy));
    CHECK_HIP(hipFree(my_buffer));

    printf("[GPU %d] Done\n", gpu_id);
}

int main(int argc, char** argv) {
    printf("===========================================\n");
    printf("Multi-GPU HIP IPC Communication Test\n");
    printf("===========================================\n");

    // Create shared memory BEFORE any HIP calls
    SharedData* shared = (SharedData*)mmap(NULL, sizeof(SharedData),
                                            PROT_READ | PROT_WRITE,
                                            MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (shared == MAP_FAILED) {
        perror("mmap failed");
        return 1;
    }

    memset(shared, 0, sizeof(SharedData));
    new (&shared->ready_count) std::atomic<int>(0);
    new (&shared->handles_ready) std::atomic<int>(0);
    shared->test_passed[0] = 0;
    shared->test_passed[1] = 0;

    // Fork two processes BEFORE any HIP initialization
    // Each child will initialize HIP independently
    pid_t pid1 = fork();
    if (pid1 == 0) {
        // First child - GPU 0
        // Do HIP init queries in child
        int num_gpus;
        hipError_t err = hipGetDeviceCount(&num_gpus);
        if (err != hipSuccess || num_gpus < 2) {
            fprintf(stderr, "Need at least 2 GPUs, got %d\n", num_gpus);
            exit(1);
        }

        int can_access;
        hipDeviceCanAccessPeer(&can_access, 0, 1);
        printf("[GPU 0] Peer access to GPU 1: %d\n", can_access);

        run_gpu_test(0, 1, shared);
        exit(0);
    }

    pid_t pid2 = fork();
    if (pid2 == 0) {
        // Second child - GPU 1
        int num_gpus;
        hipError_t err = hipGetDeviceCount(&num_gpus);
        if (err != hipSuccess || num_gpus < 2) {
            fprintf(stderr, "Need at least 2 GPUs, got %d\n", num_gpus);
            exit(1);
        }

        int can_access;
        hipDeviceCanAccessPeer(&can_access, 1, 0);
        printf("[GPU 1] Peer access to GPU 0: %d\n", can_access);

        run_gpu_test(1, 0, shared);
        exit(0);
    }

    // Parent waits for both children
    int status1, status2;
    waitpid(pid1, &status1, 0);
    waitpid(pid2, &status2, 0);

    int all_passed = 1;
    if (!WIFEXITED(status1) || WEXITSTATUS(status1) != 0) {
        printf("GPU 0 process failed (status: %d)\n", status1);
        all_passed = 0;
    }
    if (!WIFEXITED(status2) || WEXITSTATUS(status2) != 0) {
        printf("GPU 1 process failed (status: %d)\n", status2);
        all_passed = 0;
    }
    if (!shared->test_passed[0] || !shared->test_passed[1]) {
        all_passed = 0;
    }

    printf("\n===========================================\n");
    if (all_passed) {
        printf("ALL TESTS PASSED - GPU-to-GPU IPC works!\n");
    } else {
        printf("SOME TESTS FAILED\n");
    }
    printf("===========================================\n");

    munmap(shared, sizeof(SharedData));
    return all_passed ? 0 : 1;
}
