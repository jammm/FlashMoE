/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 *
 * HIP Port: Simplified Python Bindings for FlashMoE (Utility Functions Only)
 * Target: AMD Instinct MI450 (gfx1250) / MI450 (gfx1250)
 *
 * This is a minimal version that compiles without the full MoE infrastructure.
 * It provides basic utility functions for querying device properties and
 * memory management.
 */

#include <hip/hip_runtime.h>

#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPStream.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>

// FlashMoE HIP headers (minimal set)
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
// Utility Functions
// ============================================================================

namespace {

/**
 * @brief Get the HIP stream from current PyTorch context
 */
inline hipStream_t get_hip_stream() {
    return at::hip::getCurrentHIPStream().stream();
}

} // anonymous namespace

// ============================================================================
// Device Query Functions
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
 * @brief Get the device name
 */
std::string get_device_name() {
    int device;
    CHECK_HIP(hipGetDevice(&device));

    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, device));

    return std::string(props.name);
}

/**
 * @brief Get total device memory in bytes
 */
int64_t get_total_memory() {
    int device;
    CHECK_HIP(hipGetDevice(&device));

    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, device));

    return static_cast<int64_t>(props.totalGlobalMem);
}

/**
 * @brief Get number of compute units
 */
int64_t get_compute_units() {
    int device;
    CHECK_HIP(hipGetDevice(&device));

    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, device));

    return static_cast<int64_t>(props.multiProcessorCount);
}

/**
 * @brief Get max LDS (shared memory) per block
 */
int64_t get_max_shared_memory_per_block() {
    int device;
    CHECK_HIP(hipGetDevice(&device));

    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, device));

    return static_cast<int64_t>(props.sharedMemPerBlock);
}

/**
 * @brief Get max threads per block
 */
int64_t get_max_threads_per_block() {
    int device;
    CHECK_HIP(hipGetDevice(&device));

    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, device));

    return static_cast<int64_t>(props.maxThreadsPerBlock);
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

/**
 * @brief Synchronize the device
 */
void synchronize_device() {
    CHECK_HIP(hipDeviceSynchronize());
}

// ============================================================================
// PyBind11 Module Definition
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = R"pbdoc(
        FlashMoE HIP Extension (Utilities)
        ----------------------------------

        This module provides HIP-accelerated utilities for AMD Instinct GPUs.

        Currently provides:
        - Device query functions (architecture, properties, memory)
        - Memory management utilities
        - Stream synchronization

        The full MoE kernel implementation is under development.
    )pbdoc";

    // Device query functions
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

    m.def("get_device_name", &get_device_name,
        R"pbdoc(
            Get the GPU device name.
        )pbdoc"
    );

    m.def("get_total_memory", &get_total_memory,
        R"pbdoc(
            Get total device memory in bytes.
        )pbdoc"
    );

    m.def("get_compute_units", &get_compute_units,
        R"pbdoc(
            Get number of compute units on the device.
        )pbdoc"
    );

    m.def("get_max_shared_memory_per_block", &get_max_shared_memory_per_block,
        R"pbdoc(
            Get maximum LDS (shared memory) per block in bytes.
        )pbdoc"
    );

    m.def("get_max_threads_per_block", &get_max_threads_per_block,
        R"pbdoc(
            Get maximum threads per block.
        )pbdoc"
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

    m.def("synchronize_device", &synchronize_device,
        R"pbdoc(
            Synchronize the entire HIP device.
        )pbdoc"
    );

    // Version info
    m.attr("__version__") = "0.1.0";
    m.attr("__hip_version__") = std::to_string(HIP_VERSION_MAJOR) + "." +
                                 std::to_string(HIP_VERSION_MINOR);
    m.attr("__wavefront_size__") = flashmoe::WARP_SIZE;
}
