cmake_minimum_required(VERSION 3.27)
project(flashmoe_hip_v010 LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_HIP_EXTENSIONS OFF)

# flags
add_library(flashmoe_warnings INTERFACE)

target_compile_options(flashmoe_warnings INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -Wextra
        -fno-strict-aliasing
        -Wno-unknown-pragmas
        -Wnull-dereference
        -Wnarrowing
        -Wno-switch
        -Wformat=2
        -Wno-unused-but-set-parameter
        -Wno-sign-compare
        >

        $<$<COMPILE_LANGUAGE:HIP>:
        -Wall -Wextra
        -fno-strict-aliasing
        -Wno-unknown-pragmas
        -Wno-sign-compare
        -Wno-unused-parameter
        >
)

# ---- ROCm / HIP ----
find_package(hip REQUIRED)
find_package(rocblas QUIET)
find_package(hipblas QUIET)
find_package(hipblaslt QUIET)

set(FLASHMOE_KERNELS_SOURCE "" CACHE PATH "Path to flashmoe kernels source tree (optional)")

if (FLASHMOE_KERNELS_SOURCE)
  set(FLASHMOE_SOURCE_DIR ${FLASHMOE_KERNELS_SOURCE})
endif()

if (NOT DEFINED GENERATED_SRC)
  message(FATAL_ERROR "Need -DGENERATED_SRC")
endif()

if (NOT DEFINED TARGET_MODULE_NAME)
  message(FATAL_ERROR "Need -DTARGET_MODULE_NAME")
endif()

if (NOT DEFINED ARCH)
  message(FATAL_ERROR "Need -DARCH=1250")
endif()

if(NOT ${ARCH} EQUAL 1250)
  message(FATAL_ERROR "Unsupported ARCH for this HIP port (need 1250)")
endif()

set(GPU_ARCH "${ARCH}")

if (NOT DEFINED GPU_TARGETS)
  set(GPU_TARGETS "gfx${ARCH}")
endif()

# ---- CPM ----
include(${FLASHMOE_SOURCE_DIR}/cmake/CPM.cmake)

# pybind11
set(PYBIND11_FINDPYTHON ON)
CPMAddPackage(
  NAME pybind11
  GITHUB_REPOSITORY pybind/pybind11
  VERSION 3.0.1
)

# rocSHMEM (optional, for multi-GPU SHMEM support)
find_package(rocshmem QUIET)

if(NOT COMMAND pybind11_add_module)
    message(FATAL_ERROR "pybind11 not found")
endif()
pybind11_add_module(${TARGET_MODULE_NAME} MODULE ${GENERATED_SRC})

# Mark the generated source as HIP so CMake uses hipcc
set_source_files_properties(${GENERATED_SRC} PROPERTIES LANGUAGE HIP)

target_include_directories(${TARGET_MODULE_NAME} PRIVATE
  ${FLASHMOE_KERNELS_SOURCE}/include
)

target_compile_options(${TARGET_MODULE_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
            $<$<COMPILE_LANGUAGE:HIP>:-O3 --offload-arch=${GPU_TARGETS}>
)

target_link_libraries(${TARGET_MODULE_NAME} PRIVATE hip::device flashmoe_warnings)
target_compile_definitions(${TARGET_MODULE_NAME} PRIVATE
            FLASHMOE_ARCH=${GPU_ARCH}
            FLASHMOE_HIP_ARCH=${GPU_ARCH}
            __HIP_PLATFORM_AMD__
)

# hipBLASLt for GEMM operations (replaces cuBLASDx)
if(TARGET roc::hipblaslt)
    target_link_libraries(${TARGET_MODULE_NAME} PRIVATE roc::hipblaslt)
    target_compile_definitions(${TARGET_MODULE_NAME} PRIVATE FLASHMOE_HAS_HIPBLASLT)
elseif(hipblaslt_FOUND)
    target_link_libraries(${TARGET_MODULE_NAME} PRIVATE hipblaslt)
    target_compile_definitions(${TARGET_MODULE_NAME} PRIVATE FLASHMOE_HAS_HIPBLASLT)
endif()

# rocBLAS fallback
if(TARGET roc::rocblas)
    target_link_libraries(${TARGET_MODULE_NAME} PRIVATE roc::rocblas)
elseif(rocblas_FOUND)
    target_link_libraries(${TARGET_MODULE_NAME} PRIVATE rocblas)
endif()

# rocSHMEM (replaces NVSHMEM) — optional, only needed for multi-GPU SHMEM
# -fgpu-rdc is required for device-side rocSHMEM calls
if(rocshmem_FOUND AND TARGET rocshmem::rocshmem)
    target_link_libraries(${TARGET_MODULE_NAME} PRIVATE rocshmem::rocshmem)
    target_compile_definitions(${TARGET_MODULE_NAME} PRIVATE FLASHMOE_HAS_ROCSHMEM)
    target_compile_options(${TARGET_MODULE_NAME} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>)
elseif(DEFINED ENV{ROCSHMEM_DIR})
    target_include_directories(${TARGET_MODULE_NAME} PRIVATE "$ENV{ROCSHMEM_DIR}/include")
    target_link_directories(${TARGET_MODULE_NAME} PRIVATE "$ENV{ROCSHMEM_DIR}/lib")
    target_link_libraries(${TARGET_MODULE_NAME} PRIVATE rocshmem)
    target_compile_definitions(${TARGET_MODULE_NAME} PRIVATE FLASHMOE_HAS_ROCSHMEM)
    target_compile_options(${TARGET_MODULE_NAME} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>)
endif()
