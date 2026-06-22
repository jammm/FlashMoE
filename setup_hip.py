# Copyright (c) 2025, Osayamen Jonathan Aimuyo
# All rights reserved.
#
# FlashMoE HIP Extension Build Script
# Target: AMD Instinct MI450 (gfx1250)

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# ============================================================================
# Environment Configuration
# ============================================================================

# Get ROCm paths from environment or use defaults
ROCM_PATH = os.environ.get(
    "ROCM_PATH",
    "/jam/moe/venv/lib/python3.12/site-packages/_rocm_sdk_devel"
)
ROCSHMEM_ROOT = os.environ.get(
    "ROCSHMEM_ROOT",
    "/jam/moe/rocm-systems/projects/rocshmem/install"
)
ROCM_CORE_LIB = os.environ.get(
    "ROCM_CORE_LIB",
    "/jam/moe/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib"
)

# HIP compiler path
HIPCC = os.path.join(ROCM_PATH, "bin", "hipcc")
if not os.path.exists(HIPCC):
    # Try alternative location
    HIPCC = "hipcc"

print(f"ROCm Path: {ROCM_PATH}")
print(f"rocSHMEM Root: {ROCSHMEM_ROOT}")
print(f"HIP Compiler: {HIPCC}")

# ============================================================================
# Source Files
# ============================================================================

# Project root
PROJECT_ROOT = Path(__file__).parent.absolute()
CSRC_DIR = PROJECT_ROOT / "csrc"

# Source files to compile
# Use simplified bindings that don't require the full MoE infrastructure
sources = [
    str(CSRC_DIR / "python_bindings_simple.hip.cpp"),
]

# ============================================================================
# Include Directories
# ============================================================================

# Get PyTorch include directories
import torch
from torch.utils import cpp_extension as torch_cpp_ext

torch_include_dirs = torch_cpp_ext.include_paths()

# Get Python include path using sysconfig
import sysconfig
python_include = sysconfig.get_path("include")
print(f"Python include: {python_include}")

include_dirs = [
    # Project includes
    str(CSRC_DIR / "include"),
    # Python includes (must come early)
    python_include,
    # ROCm SDK includes
    os.path.join(ROCM_PATH, "include"),
    os.path.join(ROCM_PATH, "include", "hip"),
    os.path.join(ROCM_PATH, "include", "hipcub"),
    os.path.join(ROCM_PATH, "include", "rocprim"),
    os.path.join(ROCM_PATH, "include", "rocwmma"),
    os.path.join(ROCM_PATH, "include", "hipblaslt"),
    os.path.join(ROCM_PATH, "include", "hipsolver"),
    os.path.join(ROCM_PATH, "include", "hipblas"),
    os.path.join(ROCM_PATH, "include", "hipsparse"),
    # rocSHMEM includes
    os.path.join(ROCSHMEM_ROOT, "include"),
] + torch_include_dirs

# Filter out non-existent directories
include_dirs = [d for d in include_dirs if os.path.isdir(d)]
print(f"Include directories: {include_dirs}")

# ============================================================================
# Library Directories
# ============================================================================

# Get PyTorch library directories
torch_lib_dirs = torch_cpp_ext.library_paths()

library_dirs = [
    os.path.join(ROCM_PATH, "lib"),
    os.path.join(ROCSHMEM_ROOT, "lib"),
    ROCM_CORE_LIB,
] + torch_lib_dirs

# Filter out non-existent directories
library_dirs = [d for d in library_dirs if os.path.isdir(d)]
print(f"Library directories: {library_dirs}")

# ============================================================================
# Compiler and Linker Flags
# ============================================================================

# Target GPU architectures
GPU_TARGETS = ["gfx1250"]

# ============================================================================
# Custom Build Extension to Use hipcc Directly
# ============================================================================

class HIPBuildExtension(build_ext):
    """Custom build extension that uses hipcc for HIP files."""

    def build_extensions(self):
        for ext in self.extensions:
            self._build_hip_extension(ext)

    def _build_hip_extension(self, ext):
        """Build extension using hipcc directly."""
        # Get output path
        ext_path = self.get_ext_fullpath(ext.name)
        ext_dir = os.path.dirname(ext_path)

        # Create output directory if it doesn't exist
        os.makedirs(ext_dir, exist_ok=True)

        # Build compile command
        include_flags = [f"-I{d}" for d in include_dirs]
        library_flags = [f"-L{d}" for d in library_dirs]
        rpath_flags = [f"-Wl,-rpath,{d}" for d in library_dirs]

        # Define macros
        define_flags = [
            "-D__HIP_PLATFORM_AMD__",
            "-DFLASHMOE_HIP=1",
            "-DFLASHMOE_HIP_ARCH=1250",
            "-DTORCH_EXTENSION_NAME=flashmoe_hip",
            "-DTORCH_API_INCLUDE_EXTENSION_H",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            # Required for ROCm PyTorch - tells headers to use HIP equivalents
            "-DUSE_ROCM=1",
            "-DHIPBLAS_V2",
        ]

        # Compile flags
        compile_flags = [
            "-std=c++20",
            "-fPIC",
            "-O3",
            "-x", "hip",
            "-fgpu-rdc",
            # Warning flags
            "-Wall",
            "-Wno-unused-result",
            "-Wno-sign-compare",
            "-Wno-unused-variable",
            "-Wno-unused-parameter",
            "-Wno-unused-but-set-parameter",
            "-Wno-deprecated-declarations",
            "-Wno-switch",
        ]

        # Add offload arch for each GPU target
        for arch in GPU_TARGETS:
            compile_flags.append(f"--offload-arch={arch}")

        # Link flags
        link_flags = [
            "-shared",
            "-fPIC",
            "-lamdhip64",
            "-latomic",
            "-lc10",
            "-ltorch",
            "-ltorch_cpu",
            "-ltorch_python",
        ]

        # Link against rocshmem if available
        rocshmem_lib = os.path.join(ROCSHMEM_ROOT, "lib", "librocshmem.so")
        if os.path.exists(rocshmem_lib):
            link_flags.append("-lrocshmem")
            print("Linking against rocshmem")

        # Build the compile and link command
        cmd = [
            HIPCC,
        ] + compile_flags + include_flags + define_flags + ext.sources + [
            "-o", ext_path,
        ] + library_flags + rpath_flags + link_flags

        print(f"\nCompiling with command:\n{' '.join(cmd)}\n")

        # Execute the build command
        try:
            subprocess.check_call(cmd, env=os.environ)
            print(f"\nSuccessfully built {ext_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error compiling {ext.name}: {e}")

# ============================================================================
# Extension Module Definition
# ============================================================================

ext_modules = [
    Extension(
        name="flashmoe_hip",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        language="c++",
    ),
]

# ============================================================================
# Setup
# ============================================================================

setup(
    name="flashmoe_hip",
    version="0.1.0",
    description="FlashMoE HIP Extension for AMD GPUs",
    author="Osayamen Jonathan Aimuyo",
    ext_modules=ext_modules,
    cmdclass={"build_ext": HIPBuildExtension},
    python_requires=">=3.10",
)
