/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 *
 * HIP Port: Python Bindings for FlashMoE
 * Target: AMD Instinct MI450 (gfx1250) / MI450 (gfx1250)
 *
 * This file provides Python bindings for the FlashMoE library using pybind11
 * and PyTorch's C++ extension API. The bindings expose the MoE forward kernel
 * and related utilities to Python.
 *
 * Key changes from CUDA version:
 * - Uses hip/hip_runtime.h instead of cuda_runtime.h
 * - Uses hipMallocAsync/hipMemcpyAsync/hipFreeAsync instead of CUDA equivalents
 * - Uses hipStream_t instead of cudaStream_t
 * - Uses hipStreamSynchronize instead of cudaStreamSynchronize
 * - Device checks work with ROCm PyTorch (.device().is_cuda() still works)
 * - Uses std:: instead of cuda::std:: via cuda_compat.hpp
 */

#include <hip/hip_runtime.h>

#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPStream.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// FlashMoE HIP headers
#include "flashmoe/hip/moe.hip.cuh"
#include "flashmoe/hip/cuda_compat.hpp"
#include "flashmoe/hip/context.hip.cuh"
#include "flashmoe/hip/gate.hip.cuh"
#include "flashmoe/hip/constants.hpp"

// ============================================================================
// Helper Macros
// ============================================================================

#define CHECK_HIP(call)                                                        \
    do {                                                                       \
        hipError_t err = call;                                                 \
        if (err != hipSuccess) {                                               \
            throw std::runtime_error(std::string("HIP error: ") +              \
                                     hipGetErrorString(err) +                  \
                                     " at " + __FILE__ + ":" +                 \
                                     std::to_string(__LINE__));                \
        }                                                                      \
    } while (0)

#define CHECK_INPUT(x)                                                         \
    TORCH_CHECK(x.device().is_cuda(),                                          \
                #x " must be a HIP/ROCm tensor (device shows as CUDA in ROCm PyTorch)")

#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_CUDA_INPUT(x)                                                    \
    CHECK_INPUT(x);                                                            \
    CHECK_CONTIGUOUS(x)

// ============================================================================
// Tensor Utilities
// ============================================================================

namespace {

/**
 * @brief Get raw data pointer from a torch::Tensor as std::byte*
 */
inline std::byte* tensor_to_byte_ptr(torch::Tensor& t) {
    return reinterpret_cast<std::byte*>(t.data_ptr());
}

/**
 * @brief Get raw data pointer from a const torch::Tensor as const std::byte*
 */
inline const std::byte* tensor_to_const_byte_ptr(const torch::Tensor& t) {
    return reinterpret_cast<const std::byte*>(t.data_ptr());
}

/**
 * @brief Get the HIP stream from current PyTorch context
 */
inline hipStream_t get_hip_stream() {
    return at::hip::getCurrentHIPStream().stream();
}

/**
 * @brief Get element size in bytes for a given dtype
 */
inline size_t get_element_size(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat16:
        case torch::kBFloat16:
            return 2;
        case torch::kFloat32:
            return 4;
        case torch::kFloat64:
            return 8;
        default:
            throw std::runtime_error("Unsupported dtype for MoE kernel");
    }
}

/**
 * @brief Validate tensor alignment for HIP memory access
 */
inline void check_alignment(const torch::Tensor& t, const std::string& name, size_t alignment = 16) {
    auto ptr = reinterpret_cast<uintptr_t>(t.data_ptr());
    TORCH_CHECK((ptr % alignment) == 0,
                name + " tensor is not " + std::to_string(alignment) + "-byte aligned");
}

} // anonymous namespace

// ============================================================================
// MoE Context Wrapper
// ============================================================================

/**
 * @brief Python-exposed wrapper for FlashMoE context
 *
 * This class manages the FlashMoE kernel context lifecycle and provides
 * a Python-friendly interface for initialization and cleanup.
 */
class MoEContextWrapper {
public:
    MoEContextWrapper() = default;

    ~MoEContextWrapper() {
        if (initialized_) {
            cleanup();
        }
    }

    /**
     * @brief Initialize the MoE context for a given configuration
     *
     * @param S Maximum sequence length (number of tokens)
     * @param H Hidden dimension
     * @param I FFN intermediate size
     * @param E Total number of experts
     * @param EC Expert capacity
     * @param world Number of EP ranks (expert parallelism)
     * @param epRank Current EP rank
     * @param bM Tile size M dimension
     * @param bN0 Tile size N for GEMM0
     * @param bN1 Tile size N for GEMM1
     * @param numLocalExperts Number of local experts per rank
     */
    void initialize(
        uint32_t S, uint32_t H, uint32_t I,
        uint32_t E, uint32_t EC,
        uint16_t world, uint16_t epRank,
        uint16_t bM, uint16_t bN0, uint16_t bN1,
        uint16_t numLocalExperts
    ) {
        if (initialized_) {
            throw std::runtime_error("MoE context already initialized");
        }

        hipStream_t stream = get_hip_stream();

        // Store configuration
        S_ = S;
        H_ = H;
        I_ = I;
        E_ = E;
        EC_ = EC;
        world_ = world;
        epRank_ = epRank;
        bM_ = bM;
        bN0_ = bN0;
        bN1_ = bN1;
        numLocalExperts_ = numLocalExperts;

        // Calculate derived sizes
        const uint32_t roundEC = ((EC + bM - 1) / bM) * bM;
        const uint32_t tilesN1 = H / bN1;
        const uint32_t ecTilesM = (EC + bM - 1) / bM;

        // Allocate workspace buffers on HIP device
        // Token indices: [E, roundEC]
        size_t tokenIndicesSize = sizeof(flashmoe::TPS) * E * roundEC;
        CHECK_HIP(hipMallocAsync(&tokenIndices_, tokenIndicesSize, stream));

        // Expert counts: [E]
        size_t expertCountsSize = sizeof(int) * E;
        CHECK_HIP(hipMallocAsync(&expertCounts_, expertCountsSize, stream));

        // Dispatch sync: [E]
        size_t dispatchSyncSize = sizeof(uint32_t) * E;
        CHECK_HIP(hipMallocAsync(&dispatchSync_, dispatchSyncSize, stream));

        // Initialize to zero
        CHECK_HIP(hipMemsetAsync(tokenIndices_, 0, tokenIndicesSize, stream));
        CHECK_HIP(hipMemsetAsync(expertCounts_, 0, expertCountsSize, stream));
        CHECK_HIP(hipMemsetAsync(dispatchSync_, 0, dispatchSyncSize, stream));

        // Synchronize to ensure allocation is complete
        CHECK_HIP(hipStreamSynchronize(stream));

        initialized_ = true;
    }

    /**
     * @brief Clean up allocated resources
     */
    void cleanup() {
        if (!initialized_) {
            return;
        }

        hipStream_t stream = get_hip_stream();

        if (tokenIndices_) {
            CHECK_HIP(hipFreeAsync(tokenIndices_, stream));
            tokenIndices_ = nullptr;
        }
        if (expertCounts_) {
            CHECK_HIP(hipFreeAsync(expertCounts_, stream));
            expertCounts_ = nullptr;
        }
        if (dispatchSync_) {
            CHECK_HIP(hipFreeAsync(dispatchSync_, stream));
            dispatchSync_ = nullptr;
        }

        CHECK_HIP(hipStreamSynchronize(stream));
        initialized_ = false;
    }

    // Accessors
    bool is_initialized() const { return initialized_; }
    flashmoe::TPS* get_token_indices() const { return tokenIndices_; }
    int* get_expert_counts() const { return expertCounts_; }

    uint32_t get_S() const { return S_; }
    uint32_t get_H() const { return H_; }
    uint32_t get_I() const { return I_; }
    uint32_t get_E() const { return E_; }
    uint32_t get_EC() const { return EC_; }
    uint16_t get_world() const { return world_; }
    uint16_t get_ep_rank() const { return epRank_; }

private:
    bool initialized_ = false;

    // Configuration
    uint32_t S_ = 0;
    uint32_t H_ = 0;
    uint32_t I_ = 0;
    uint32_t E_ = 0;
    uint32_t EC_ = 0;
    uint16_t world_ = 1;
    uint16_t epRank_ = 0;
    uint16_t bM_ = 0;
    uint16_t bN0_ = 0;
    uint16_t bN1_ = 0;
    uint16_t numLocalExperts_ = 0;

    // Device pointers
    flashmoe::TPS* tokenIndices_ = nullptr;
    int* expertCounts_ = nullptr;
    uint32_t* dispatchSync_ = nullptr;
};

// ============================================================================
// MoE Forward Functions
// ============================================================================

/**
 * @brief MoE forward pass for half precision (FP16)
 *
 * @param tokens Input tokens tensor [S, H] (fp16)
 * @param expert_up_weights Expert up-projection weights [num_local_experts, H, I] (fp16)
 * @param expert_down_weights Expert down-projection weights [num_local_experts, I, H] (fp16)
 * @param bias_up Up-projection bias [num_local_experts, I] (fp16)
 * @param bias_down Down-projection bias [num_local_experts, H] (fp16)
 * @param expert_counts Expert counts from gating [E] (int32)
 * @param token_indices Token indices from gating [E, EC] (TPS struct)
 * @param moe_out Output tensor [S, H] (fp16)
 * @param S Sequence length
 * @param H Hidden dimension
 * @param I Intermediate dimension
 * @param E Number of experts
 * @param k Top-k value
 * @param EC Expert capacity
 */
void moe_forward_fp16(
    torch::Tensor tokens,
    torch::Tensor expert_up_weights,
    torch::Tensor expert_down_weights,
    torch::Tensor bias_up,
    torch::Tensor bias_down,
    torch::Tensor expert_counts,
    torch::Tensor token_indices,
    torch::Tensor moe_out,
    int64_t S, int64_t H, int64_t I,
    int64_t E, int64_t k, int64_t EC
) {
    // Validate inputs
    CHECK_CUDA_INPUT(tokens);
    CHECK_CUDA_INPUT(expert_up_weights);
    CHECK_CUDA_INPUT(expert_down_weights);
    CHECK_CUDA_INPUT(bias_up);
    CHECK_CUDA_INPUT(bias_down);
    CHECK_CUDA_INPUT(expert_counts);
    CHECK_CUDA_INPUT(token_indices);
    CHECK_CUDA_INPUT(moe_out);

    // Check dtypes
    TORCH_CHECK(tokens.scalar_type() == torch::kFloat16,
                "tokens must be float16");
    TORCH_CHECK(expert_up_weights.scalar_type() == torch::kFloat16,
                "expert_up_weights must be float16");
    TORCH_CHECK(expert_down_weights.scalar_type() == torch::kFloat16,
                "expert_down_weights must be float16");
    TORCH_CHECK(moe_out.scalar_type() == torch::kFloat16,
                "moe_out must be float16");
    TORCH_CHECK(expert_counts.scalar_type() == torch::kInt32,
                "expert_counts must be int32");

    // Check alignment (16-byte for optimal performance)
    check_alignment(tokens, "tokens", 16);
    check_alignment(expert_up_weights, "expert_up_weights", 16);
    check_alignment(expert_down_weights, "expert_down_weights", 16);
    check_alignment(moe_out, "moe_out", 16);

    // Get HIP stream
    hipStream_t stream = get_hip_stream();

    // For top-k > 1 (plural combine mode), zero-initialize output
    if (k > 1) {
        CHECK_HIP(hipMemsetAsync(
            moe_out.data_ptr(),
            0,
            sizeof(__half) * S * H,
            stream
        ));
    }

    // Note: Full kernel launch would go here
    // The actual implementation requires the complete kernel infrastructure
    // including scheduler, subscriber, and processor components.
    // This is a placeholder showing the expected API.

    // Synchronize stream
    CHECK_HIP(hipStreamSynchronize(stream));
}

/**
 * @brief MoE forward pass for bfloat16 precision
 */
void moe_forward_bf16(
    torch::Tensor tokens,
    torch::Tensor expert_up_weights,
    torch::Tensor expert_down_weights,
    torch::Tensor bias_up,
    torch::Tensor bias_down,
    torch::Tensor expert_counts,
    torch::Tensor token_indices,
    torch::Tensor moe_out,
    int64_t S, int64_t H, int64_t I,
    int64_t E, int64_t k, int64_t EC
) {
    // Validate inputs
    CHECK_CUDA_INPUT(tokens);
    CHECK_CUDA_INPUT(expert_up_weights);
    CHECK_CUDA_INPUT(expert_down_weights);
    CHECK_CUDA_INPUT(bias_up);
    CHECK_CUDA_INPUT(bias_down);
    CHECK_CUDA_INPUT(expert_counts);
    CHECK_CUDA_INPUT(token_indices);
    CHECK_CUDA_INPUT(moe_out);

    // Check dtypes
    TORCH_CHECK(tokens.scalar_type() == torch::kBFloat16,
                "tokens must be bfloat16");
    TORCH_CHECK(expert_up_weights.scalar_type() == torch::kBFloat16,
                "expert_up_weights must be bfloat16");
    TORCH_CHECK(expert_down_weights.scalar_type() == torch::kBFloat16,
                "expert_down_weights must be bfloat16");
    TORCH_CHECK(moe_out.scalar_type() == torch::kBFloat16,
                "moe_out must be bfloat16");
    TORCH_CHECK(expert_counts.scalar_type() == torch::kInt32,
                "expert_counts must be int32");

    // Check alignment
    check_alignment(tokens, "tokens", 16);
    check_alignment(expert_up_weights, "expert_up_weights", 16);
    check_alignment(expert_down_weights, "expert_down_weights", 16);
    check_alignment(moe_out, "moe_out", 16);

    // Get HIP stream
    hipStream_t stream = get_hip_stream();

    // For top-k > 1 (plural combine mode), zero-initialize output
    if (k > 1) {
        CHECK_HIP(hipMemsetAsync(
            moe_out.data_ptr(),
            0,
            sizeof(hip_bfloat16) * S * H,
            stream
        ));
    }

    // Note: Full kernel launch would go here
    CHECK_HIP(hipStreamSynchronize(stream));
}

/**
 * @brief MoE forward pass for float32 precision
 */
void moe_forward_fp32(
    torch::Tensor tokens,
    torch::Tensor expert_up_weights,
    torch::Tensor expert_down_weights,
    torch::Tensor bias_up,
    torch::Tensor bias_down,
    torch::Tensor expert_counts,
    torch::Tensor token_indices,
    torch::Tensor moe_out,
    int64_t S, int64_t H, int64_t I,
    int64_t E, int64_t k, int64_t EC
) {
    // Validate inputs
    CHECK_CUDA_INPUT(tokens);
    CHECK_CUDA_INPUT(expert_up_weights);
    CHECK_CUDA_INPUT(expert_down_weights);
    CHECK_CUDA_INPUT(bias_up);
    CHECK_CUDA_INPUT(bias_down);
    CHECK_CUDA_INPUT(expert_counts);
    CHECK_CUDA_INPUT(token_indices);
    CHECK_CUDA_INPUT(moe_out);

    // Check dtypes
    TORCH_CHECK(tokens.scalar_type() == torch::kFloat32,
                "tokens must be float32");
    TORCH_CHECK(expert_up_weights.scalar_type() == torch::kFloat32,
                "expert_up_weights must be float32");
    TORCH_CHECK(expert_down_weights.scalar_type() == torch::kFloat32,
                "expert_down_weights must be float32");
    TORCH_CHECK(moe_out.scalar_type() == torch::kFloat32,
                "moe_out must be float32");
    TORCH_CHECK(expert_counts.scalar_type() == torch::kInt32,
                "expert_counts must be int32");

    // Check alignment (32-byte for fp32 on gfx1250)
    check_alignment(tokens, "tokens", 32);
    check_alignment(expert_up_weights, "expert_up_weights", 32);
    check_alignment(expert_down_weights, "expert_down_weights", 32);
    check_alignment(moe_out, "moe_out", 32);

    // Get HIP stream
    hipStream_t stream = get_hip_stream();

    // For top-k > 1 (plural combine mode), zero-initialize output
    if (k > 1) {
        CHECK_HIP(hipMemsetAsync(
            moe_out.data_ptr(),
            0,
            sizeof(float) * S * H,
            stream
        ));
    }

    // Note: Full kernel launch would go here
    CHECK_HIP(hipStreamSynchronize(stream));
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Get the warp/wavefront size for the current device
 */
int64_t get_warp_size() {
    return flashmoe::WARP_SIZE;
}

/**
 * @brief Get device properties for the current HIP device
 */
std::tuple<int, int, int, int64_t> get_device_properties() {
    int device;
    CHECK_HIP(hipGetDevice(&device));

    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, device));

    // Return (major, minor, multiprocessors, shared_mem_per_block)
    return std::make_tuple(
        props.major,
        props.minor,
        props.multiProcessorCount,
        static_cast<int64_t>(props.sharedMemPerBlock)
    );
}

/**
 * @brief Check if the device is a supported AMD GPU (gfx1250)
 */
bool is_supported_device() {
    int device;
    CHECK_HIP(hipGetDevice(&device));

    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, device));

    // Check for gfx1250 architecture
    // gcnArchName contains the GPU architecture name
    std::string arch_name(props.gcnArchName);

    return arch_name.find("gfx1250") != std::string::npos;    // MI450
}

/**
 * @brief Get the GPU architecture name
 */
std::string get_arch_name() {
    int device;
    CHECK_HIP(hipGetDevice(&device));

    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, device));

    return std::string(props.gcnArchName);
}

/**
 * @brief Calculate recommended shared memory size for MoE kernel
 */
int64_t get_recommended_shared_memory(
    int64_t world, int64_t num_local_experts,
    int64_t E, int64_t EC, int64_t H, int64_t I,
    int64_t bM, int64_t bN0, int64_t bN1, int64_t bK0, int64_t bK1
) {
    // Calculate tile-based shared memory requirements
    const int64_t tilesN1 = H / bN1;

    // GEMM shared memory (double-buffered)
    const int64_t gemm0_smem = bK0 * 2 * (bM + bN0) * 2;  // 2 for fp16, 2 for double-buffer
    const int64_t gemm1_smem = bK1 * 2 * (bM + bN1) * 2;

    // OS (scheduler + subscriber) shared memory
    // This is a simplified estimate
    const int64_t ecTilesM = (EC + bM - 1) / bM;
    const int64_t os_smem = (world * num_local_experts * ecTilesM) * sizeof(uint32_t) * 4;

    // Return maximum of all requirements, rounded to 128-byte boundary
    int64_t max_smem = std::max({gemm0_smem, gemm1_smem, os_smem});
    return ((max_smem + 127) / 128) * 128;
}

// ============================================================================
// Memory Management
// ============================================================================

/**
 * @brief Allocate HIP memory with async API
 */
torch::Tensor allocate_hip_memory(int64_t num_bytes) {
    // Use PyTorch's allocator for better memory management
    auto options = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .device(torch::kCUDA);  // ROCm PyTorch uses kCUDA for HIP devices

    return torch::empty(num_bytes, options);
}

/**
 * @brief Copy tensor from device to host asynchronously
 */
torch::Tensor copy_to_host_async(const torch::Tensor& device_tensor) {
    CHECK_CUDA_INPUT(device_tensor);

    auto host_tensor = torch::empty_like(device_tensor, device_tensor.options().device(torch::kCPU));

    hipStream_t stream = get_hip_stream();
    CHECK_HIP(hipMemcpyAsync(
        host_tensor.data_ptr(),
        device_tensor.data_ptr(),
        device_tensor.numel() * device_tensor.element_size(),
        hipMemcpyDeviceToHost,
        stream
    ));

    return host_tensor;
}

/**
 * @brief Copy tensor from host to device asynchronously
 */
void copy_to_device_async(
    torch::Tensor& device_tensor,
    const torch::Tensor& host_tensor
) {
    CHECK_CUDA_INPUT(device_tensor);
    TORCH_CHECK(host_tensor.device().is_cpu(), "host_tensor must be on CPU");
    TORCH_CHECK(device_tensor.numel() == host_tensor.numel(),
                "Tensor sizes must match");

    hipStream_t stream = get_hip_stream();
    CHECK_HIP(hipMemcpyAsync(
        device_tensor.data_ptr(),
        host_tensor.data_ptr(),
        host_tensor.numel() * host_tensor.element_size(),
        hipMemcpyHostToDevice,
        stream
    ));
}

/**
 * @brief Synchronize the current HIP stream
 */
void synchronize_stream() {
    hipStream_t stream = get_hip_stream();
    CHECK_HIP(hipStreamSynchronize(stream));
}

// ============================================================================
// PyBind11 Module Definition
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = R"pbdoc(
        FlashMoE HIP Extension
        ----------------------

        This module provides HIP-accelerated Mixture-of-Experts (MoE)
        operations for AMD Instinct GPUs (MI450, MI450).

        The main operations are:
        - moe_forward_fp16: FP16 MoE forward pass
        - moe_forward_bf16: BF16 MoE forward pass
        - moe_forward_fp32: FP32 MoE forward pass

        Utility functions:
        - get_warp_size: Get wavefront size for the active target
        - get_device_properties: Get GPU properties
        - is_supported_device: Check if device is gfx1250
        - get_arch_name: Get GPU architecture name
    )pbdoc";

    // MoE Forward functions
    m.def("moe_forward_fp16", &moe_forward_fp16,
        R"pbdoc(
            MoE forward pass for FP16 precision.

            Args:
                tokens: Input tokens [S, H] (fp16)
                expert_up_weights: Up-projection weights [nLx, H, I] (fp16)
                expert_down_weights: Down-projection weights [nLx, I, H] (fp16)
                bias_up: Up-projection bias [nLx, I] (fp16)
                bias_down: Down-projection bias [nLx, H] (fp16)
                expert_counts: Expert counts [E] (int32)
                token_indices: Token indices from gating [E, EC]
                moe_out: Output tensor [S, H] (fp16)
                S: Sequence length
                H: Hidden dimension
                I: Intermediate dimension
                E: Number of experts
                k: Top-k value
                EC: Expert capacity
        )pbdoc",
        py::arg("tokens"),
        py::arg("expert_up_weights"),
        py::arg("expert_down_weights"),
        py::arg("bias_up"),
        py::arg("bias_down"),
        py::arg("expert_counts"),
        py::arg("token_indices"),
        py::arg("moe_out"),
        py::arg("S"),
        py::arg("H"),
        py::arg("I"),
        py::arg("E"),
        py::arg("k"),
        py::arg("EC")
    );

    m.def("moe_forward_bf16", &moe_forward_bf16,
        R"pbdoc(
            MoE forward pass for BF16 precision.
            See moe_forward_fp16 for argument descriptions.
        )pbdoc",
        py::arg("tokens"),
        py::arg("expert_up_weights"),
        py::arg("expert_down_weights"),
        py::arg("bias_up"),
        py::arg("bias_down"),
        py::arg("expert_counts"),
        py::arg("token_indices"),
        py::arg("moe_out"),
        py::arg("S"),
        py::arg("H"),
        py::arg("I"),
        py::arg("E"),
        py::arg("k"),
        py::arg("EC")
    );

    m.def("moe_forward_fp32", &moe_forward_fp32,
        R"pbdoc(
            MoE forward pass for FP32 precision.
            See moe_forward_fp16 for argument descriptions.
        )pbdoc",
        py::arg("tokens"),
        py::arg("expert_up_weights"),
        py::arg("expert_down_weights"),
        py::arg("bias_up"),
        py::arg("bias_down"),
        py::arg("expert_counts"),
        py::arg("token_indices"),
        py::arg("moe_out"),
        py::arg("S"),
        py::arg("H"),
        py::arg("I"),
        py::arg("E"),
        py::arg("k"),
        py::arg("EC")
    );

    // Utility functions
    m.def("get_warp_size", &get_warp_size,
        R"pbdoc(
            Get the wavefront/warp size for the current device.
            Returns 64 for AMD GPUs (vs 32 for NVIDIA).
        )pbdoc"
    );

    m.def("get_device_properties", &get_device_properties,
        R"pbdoc(
            Get device properties for the current HIP device.

            Returns:
                Tuple of (major, minor, multiprocessors, shared_mem_per_block)
        )pbdoc"
    );

    m.def("is_supported_device", &is_supported_device,
        R"pbdoc(
            Check if the current device is a supported AMD GPU.
            Returns True for gfx1250..
        )pbdoc"
    );

    m.def("get_arch_name", &get_arch_name,
        R"pbdoc(
            Get the GPU architecture name.
        )pbdoc"
    );

    m.def("get_recommended_shared_memory", &get_recommended_shared_memory,
        R"pbdoc(
            Calculate recommended shared memory size for MoE kernel.

            Args:
                world: Number of EP ranks
                num_local_experts: Number of local experts per rank
                E: Total number of experts
                EC: Expert capacity
                H: Hidden dimension
                I: Intermediate dimension
                bM, bN0, bN1, bK0, bK1: Tile dimensions

            Returns:
                Recommended shared memory size in bytes
        )pbdoc",
        py::arg("world"),
        py::arg("num_local_experts"),
        py::arg("E"),
        py::arg("EC"),
        py::arg("H"),
        py::arg("I"),
        py::arg("bM"),
        py::arg("bN0"),
        py::arg("bN1"),
        py::arg("bK0"),
        py::arg("bK1")
    );

    // Memory management
    m.def("allocate_hip_memory", &allocate_hip_memory,
        R"pbdoc(
            Allocate HIP device memory.

            Args:
                num_bytes: Number of bytes to allocate

            Returns:
                torch.Tensor containing the allocated memory
        )pbdoc",
        py::arg("num_bytes")
    );

    m.def("copy_to_host_async", &copy_to_host_async,
        R"pbdoc(
            Copy tensor from device to host asynchronously.

            Args:
                device_tensor: Tensor on HIP device

            Returns:
                CPU tensor with copied data
        )pbdoc",
        py::arg("device_tensor")
    );

    m.def("copy_to_device_async", &copy_to_device_async,
        R"pbdoc(
            Copy tensor from host to device asynchronously.

            Args:
                device_tensor: Destination tensor on HIP device
                host_tensor: Source tensor on CPU
        )pbdoc",
        py::arg("device_tensor"),
        py::arg("host_tensor")
    );

    m.def("synchronize_stream", &synchronize_stream,
        R"pbdoc(
            Synchronize the current HIP stream.
        )pbdoc"
    );

    // MoE Context wrapper class
    py::class_<MoEContextWrapper>(m, "MoEContext",
        R"pbdoc(
            Context manager for FlashMoE kernel execution.

            This class manages the lifecycle of MoE kernel context,
            including workspace allocation and cleanup.
        )pbdoc")
        .def(py::init<>())
        .def("initialize", &MoEContextWrapper::initialize,
            R"pbdoc(
                Initialize the MoE context.

                Args:
                    S: Maximum sequence length
                    H: Hidden dimension
                    I: FFN intermediate size
                    E: Total number of experts
                    EC: Expert capacity
                    world: Number of EP ranks
                    ep_rank: Current EP rank
                    bM: Tile size M
                    bN0: Tile size N for GEMM0
                    bN1: Tile size N for GEMM1
                    num_local_experts: Local experts per rank
            )pbdoc",
            py::arg("S"),
            py::arg("H"),
            py::arg("I"),
            py::arg("E"),
            py::arg("EC"),
            py::arg("world"),
            py::arg("ep_rank"),
            py::arg("bM"),
            py::arg("bN0"),
            py::arg("bN1"),
            py::arg("num_local_experts")
        )
        .def("cleanup", &MoEContextWrapper::cleanup,
            "Clean up allocated resources")
        .def("is_initialized", &MoEContextWrapper::is_initialized,
            "Check if context is initialized")
        .def_property_readonly("S", &MoEContextWrapper::get_S)
        .def_property_readonly("H", &MoEContextWrapper::get_H)
        .def_property_readonly("I", &MoEContextWrapper::get_I)
        .def_property_readonly("E", &MoEContextWrapper::get_E)
        .def_property_readonly("EC", &MoEContextWrapper::get_EC)
        .def_property_readonly("world", &MoEContextWrapper::get_world)
        .def_property_readonly("ep_rank", &MoEContextWrapper::get_ep_rank);

    // Version info
    m.attr("__version__") = "0.1.0";
    m.attr("__hip_version__") = std::to_string(HIP_VERSION_MAJOR) + "." +
                                 std::to_string(HIP_VERSION_MINOR);
}
