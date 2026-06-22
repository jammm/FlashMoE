/**
 * FlashMoE hipBLASLt Optimized Expert GEMM Benchmark
 *
 * Uses column-major layout for hipBLASLt with corrected leading dimensions
 * For row-major C[M,N] = A[M,K] @ B[K,N], we compute C^T[N,M] = B^T[N,K] @ A^T[K,M]
 *
 * Target: AMD Instinct MI450/MI450
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#define HIPBLASLT_CHECK(call) do { \
    hipblasStatus_t status = call; \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        std::cerr << "hipBLASLt Error: " << status << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

class GpuTimer {
    hipEvent_t start_, stop_;
public:
    GpuTimer() {
        HIP_CHECK(hipEventCreate(&start_));
        HIP_CHECK(hipEventCreate(&stop_));
    }
    ~GpuTimer() {
        (void)hipEventDestroy(start_);
        (void)hipEventDestroy(stop_);
    }
    void start(hipStream_t s = 0) { HIP_CHECK(hipEventRecord(start_, s)); }
    void stop(hipStream_t s = 0) {
        HIP_CHECK(hipEventRecord(stop_, s));
        HIP_CHECK(hipEventSynchronize(stop_));
    }
    float elapsed_ms() {
        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// SiLU activation kernel
__global__ void siluKernel(__half* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = __half2float(data[idx]);
    data[idx] = __float2half(x / (1.0f + expf(-x)));
}

// Column-major GEMM for row-major data: C[M,N] = A[M,K] @ B[K,N]
// We compute C^T[N,M] = B^T[N,K] @ A^T[K,M]
// Leading dimensions: ld_B^T = N, ld_A^T = K, ld_C^T = N
void runGemm(hipblasLtHandle_t handle, void* workspace, size_t workspaceSize,
             int M, int N, int K,
             const __half* A, const __half* B, __half* C,
             hipStream_t stream = 0) {
    hipblasLtMatmulDesc_t matmulDesc;
    hipblasLtMatrixLayout_t layoutBT, layoutAT, layoutCT;

    HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F));

    hipblasOperation_t trans = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans));
    hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans));

    // Column-major: C^T[N,M] = B^T[N,K] @ A^T[K,M]
    // Matrix layout: (rows, cols, ld) where ld >= rows for column-major
    // For row-major data stored in memory:
    // - B[K,N] row-major = B^T[N,K] col-major with ld = N
    // - A[M,K] row-major = A^T[K,M] col-major with ld = K
    // - C[M,N] row-major = C^T[N,M] col-major with ld = N
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutBT, HIP_R_16F, N, K, N));  // B^T[N,K], ld=N
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutAT, HIP_R_16F, K, M, K));  // A^T[K,M], ld=K
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutCT, HIP_R_16F, N, M, N));  // C^T[N,M], ld=N

    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &workspaceSize, sizeof(workspaceSize));

    hipblasLtMatmulHeuristicResult_t heuristicResult[4];
    int returnedAlgoCount = 0;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(
        handle, matmulDesc, layoutBT, layoutAT, layoutCT, layoutCT,
        pref, 4, heuristicResult, &returnedAlgoCount));

    hipblasLtMatmulPreferenceDestroy(pref);

    if (returnedAlgoCount == 0) {
        std::cerr << "No algorithm found for " << M << "x" << N << "x" << K << "\n";
        return;
    }

    float alpha = 1.0f, beta = 0.0f;
    HIPBLASLT_CHECK(hipblasLtMatmul(
        handle, matmulDesc, &alpha,
        B, layoutBT, A, layoutAT,  // Swap A and B for C^T = B^T @ A^T
        &beta, C, layoutCT, C, layoutCT,
        &heuristicResult[0].algo, workspace, workspaceSize, stream));

    hipblasLtMatrixLayoutDestroy(layoutBT);
    hipblasLtMatrixLayoutDestroy(layoutAT);
    hipblasLtMatrixLayoutDestroy(layoutCT);
    hipblasLtMatmulDescDestroy(matmulDesc);
}

void benchmarkSingleGemm(hipblasLtHandle_t handle, void* workspace, size_t workspaceSize,
                         int M, int N, int K, const char* name) {
    __half *d_A, *d_B, *d_C;
    size_t size_A = (size_t)M * K * sizeof(__half);
    size_t size_B = (size_t)K * N * sizeof(__half);
    size_t size_C = (size_t)M * N * sizeof(__half);

    HIP_CHECK(hipMalloc(&d_A, size_A));
    HIP_CHECK(hipMalloc(&d_B, size_B));
    HIP_CHECK(hipMalloc(&d_C, size_C));
    HIP_CHECK(hipMemset(d_A, 0, size_A));
    HIP_CHECK(hipMemset(d_B, 0, size_B));
    HIP_CHECK(hipMemset(d_C, 0, size_C));
    HIP_CHECK(hipDeviceSynchronize());

    // Warmup
    for (int i = 0; i < 5; ++i) {
        runGemm(handle, workspace, workspaceSize, M, N, K, d_A, d_B, d_C, 0);
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Benchmark
    GpuTimer timer;
    const int iters = 20;
    std::vector<float> times(iters);

    for (int i = 0; i < iters; ++i) {
        timer.start();
        runGemm(handle, workspace, workspaceSize, M, N, K, d_A, d_B, d_C, 0);
        timer.stop();
        times[i] = timer.elapsed_ms();
    }

    std::sort(times.begin(), times.end());
    float median_time = times[iters / 2];

    double flops = 2.0 * M * N * K;
    double tflops = (flops / 1e12) / (median_time / 1e3);
    char dim_str[32];
    snprintf(dim_str, sizeof(dim_str), "%dx%dx%d", M, N, K);

    std::cout << "│ " << std::left << std::setw(30) << name
              << " │ " << std::setw(17) << dim_str
              << " │ " << std::right << std::fixed << std::setprecision(1) << std::setw(8) << tflops
              << " │\n";

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
}

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       FlashMoE hipBLASLt Optimized Expert GEMM Benchmark                 ║\n";
    std::cout << "║                     AMD Instinct MI450 (gfx1250)                         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════╝\n";

    HIP_CHECK(hipSetDevice(0));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    std::cout << "\nDevice: " << props.name << " (" << props.gcnArchName << ")\n";

    // Mixtral-like config
    int batch_size = 8;
    int seq_len = 512;
    int num_tokens = batch_size * seq_len;  // 4096 tokens
    int hidden_dim = 4096;
    int intermediate_dim = 14336;
    int num_experts = 8;
    int top_k = 2;
    int expert_capacity = (int)(num_tokens * top_k * 1.25 / num_experts);  // ~1280

    std::cout << "\nConfiguration (Mixtral-like):\n";
    std::cout << "  Tokens: " << num_tokens << " (batch=" << batch_size << " x seq=" << seq_len << ")\n";
    std::cout << "  Hidden: " << hidden_dim << ", Intermediate: " << intermediate_dim << "\n";
    std::cout << "  Experts: " << num_experts << ", Top-K: " << top_k << ", Capacity: " << expert_capacity << "\n";

    // hipBLASLt setup
    hipblasLtHandle_t handle;
    HIPBLASLT_CHECK(hipblasLtCreate(&handle));

    size_t workspaceSize = 256 * 1024 * 1024;  // 256 MB
    void* workspace;
    HIP_CHECK(hipMalloc(&workspace, workspaceSize));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // ========== PART 1: Single GEMM Benchmarks ==========
    std::cout << "\n";
    std::cout << "┌────────────────────────────────┬───────────────────┬──────────┬──────────┐\n";
    std::cout << "│ Configuration                  │ Dimensions        │ TFLOPS   │\n";
    std::cout << "├────────────────────────────────┼───────────────────┼──────────┼──────────┤\n";

    benchmarkSingleGemm(handle, workspace, workspaceSize, 512, 14336, 4096, "Up projection (512 tokens)");
    benchmarkSingleGemm(handle, workspace, workspaceSize, 512, 4096, 14336, "Down projection (512 tokens)");
    benchmarkSingleGemm(handle, workspace, workspaceSize, 1024, 14336, 4096, "Up projection (1024 tokens)");
    benchmarkSingleGemm(handle, workspace, workspaceSize, 1024, 4096, 14336, "Down projection (1024 tokens)");
    benchmarkSingleGemm(handle, workspace, workspaceSize, 2048, 14336, 4096, "Up projection (2048 tokens)");
    benchmarkSingleGemm(handle, workspace, workspaceSize, 2048, 4096, 14336, "Down projection (2048 tokens)");
    benchmarkSingleGemm(handle, workspace, workspaceSize, expert_capacity, 14336, 4096, "Up (expert capacity)");
    benchmarkSingleGemm(handle, workspace, workspaceSize, expert_capacity, 4096, 14336, "Down (expert capacity)");

    std::cout << "└────────────────────────────────┴───────────────────┴──────────┴──────────┘\n";

    // ========== PART 2: Single Expert FFN Benchmark ==========
    __half *d_input, *d_intermediate, *d_output;
    __half *d_w_up, *d_w_down;

    size_t input_size = (size_t)expert_capacity * hidden_dim * sizeof(__half);
    size_t inter_size = (size_t)expert_capacity * intermediate_dim * sizeof(__half);
    size_t w_up_size = (size_t)hidden_dim * intermediate_dim * sizeof(__half);
    size_t w_down_size = (size_t)intermediate_dim * hidden_dim * sizeof(__half);

    HIP_CHECK(hipMalloc(&d_input, input_size));
    HIP_CHECK(hipMalloc(&d_intermediate, inter_size));
    HIP_CHECK(hipMalloc(&d_output, input_size));
    HIP_CHECK(hipMalloc(&d_w_up, w_up_size));
    HIP_CHECK(hipMalloc(&d_w_down, w_down_size));

    // Initialize with random data
    std::vector<__half> h_input(expert_capacity * hidden_dim);
    std::vector<__half> h_w_up(hidden_dim * intermediate_dim);
    std::vector<__half> h_w_down(intermediate_dim * hidden_dim);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    for (auto& v : h_input) v = __float2half(dist(gen));
    for (auto& v : h_w_up) v = __float2half(dist(gen) * 0.02f);
    for (auto& v : h_w_down) v = __float2half(dist(gen) * 0.02f);

    HIP_CHECK(hipMemcpy(d_input, h_input.data(), input_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w_up, h_w_up.data(), w_up_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w_down, h_w_down.data(), w_down_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "\n=== Single Expert FFN Benchmark ===\n";

    // Warmup
    for (int i = 0; i < 5; ++i) {
        runGemm(handle, workspace, workspaceSize,
                expert_capacity, intermediate_dim, hidden_dim,
                d_input, d_w_up, d_intermediate, stream);

        int act_size = expert_capacity * intermediate_dim;
        hipLaunchKernelGGL(siluKernel, dim3((act_size+255)/256), dim3(256), 0, stream,
                           d_intermediate, act_size);

        runGemm(handle, workspace, workspaceSize,
                expert_capacity, hidden_dim, intermediate_dim,
                d_intermediate, d_w_down, d_output, stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    // Benchmark
    const int iters = 10;
    std::vector<float> times(iters);
    GpuTimer timer;

    for (int i = 0; i < iters; ++i) {
        timer.start(stream);

        runGemm(handle, workspace, workspaceSize,
                expert_capacity, intermediate_dim, hidden_dim,
                d_input, d_w_up, d_intermediate, stream);

        int act_size = expert_capacity * intermediate_dim;
        hipLaunchKernelGGL(siluKernel, dim3((act_size+255)/256), dim3(256), 0, stream,
                           d_intermediate, act_size);

        runGemm(handle, workspace, workspaceSize,
                expert_capacity, hidden_dim, intermediate_dim,
                d_intermediate, d_w_down, d_output, stream);

        timer.stop(stream);
        times[i] = timer.elapsed_ms();
    }

    std::sort(times.begin(), times.end());
    float median_time = times[iters / 2];

    double total_flops = 2.0 * expert_capacity * hidden_dim * intermediate_dim * 2;  // up + down
    double tflops = (total_flops / 1e12) / (median_time / 1e3);
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                         SINGLE EXPERT RESULTS                            ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Time per expert FFN:    " << std::fixed << std::setprecision(3) << std::setw(8) << median_time
              << " ms" << std::setw(39) << "" << "║\n";
    std::cout << "║ Expert GEMM TFLOPS:     " << std::setprecision(2) << std::setw(8) << tflops
              << " TFLOPS" << std::setw(35) << "" << "║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════╝\n";

    // ========== PART 3: All Experts Sequential ==========
    std::cout << "\n=== All " << num_experts << " Experts (Sequential) ===\n";

    std::vector<float> all_expert_times(iters);

    for (int i = 0; i < iters; ++i) {
        timer.start(stream);

        for (int e = 0; e < num_experts; ++e) {
            runGemm(handle, workspace, workspaceSize,
                    expert_capacity, intermediate_dim, hidden_dim,
                    d_input, d_w_up, d_intermediate, stream);

            int act_size = expert_capacity * intermediate_dim;
            hipLaunchKernelGGL(siluKernel, dim3((act_size+255)/256), dim3(256), 0, stream,
                               d_intermediate, act_size);

            runGemm(handle, workspace, workspaceSize,
                    expert_capacity, hidden_dim, intermediate_dim,
                    d_intermediate, d_w_down, d_output, stream);
        }

        timer.stop(stream);
        all_expert_times[i] = timer.elapsed_ms();
    }

    std::sort(all_expert_times.begin(), all_expert_times.end());
    float median_all = all_expert_times[iters / 2];

    double all_flops = total_flops * num_experts;
    double all_tflops = (all_flops / 1e12) / (median_all / 1e3);
    float tokens_per_second = num_tokens / (median_all / 1000.0f);

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                       ALL EXPERTS RESULTS                                ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Time for " << num_experts << " experts:      " << std::fixed << std::setprecision(3) << std::setw(8) << median_all
              << " ms" << std::setw(39) << "" << "║\n";
    std::cout << "║ Aggregate TFLOPS:       " << std::setprecision(2) << std::setw(8) << all_tflops
              << " TFLOPS" << std::setw(35) << "" << "║\n";
    std::cout << "║ Tokens/second:          " << std::scientific << std::setprecision(2) << tokens_per_second
              << std::fixed << std::setw(32) << "" << "║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════╝\n";

    // Cleanup
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_intermediate));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_w_up));
    HIP_CHECK(hipFree(d_w_down));
    HIP_CHECK(hipFree(workspace));
    HIP_CHECK(hipStreamDestroy(stream));
    hipblasLtDestroy(handle);

    std::cout << "\nBenchmark complete.\n\n";

    return 0;
}
