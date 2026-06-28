from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch


_ROCSHMEM_UNIQUE_ID_BYTES = 128

_CPP_SOURCE = r"""
#include <cstdint>
#include <cstring>
#include <rocshmem/rocshmem.hpp>
#include <hip/hip_runtime.h>

extern "C" int fm_rocshmem_get_uniqueid(uint8_t* out) {
  rocshmem::rocshmem_uniqueid_t uid;
  int ret = rocshmem::rocshmem_get_uniqueid(&uid);
  if (ret == 0) std::memcpy(out, uid.data(), uid.size());
  return ret;
}

extern "C" int fm_rocshmem_init_uniqueid(int rank, int nranks, const uint8_t* data) {
  rocshmem::rocshmem_uniqueid_t uid;
  std::memcpy(uid.data(), data, uid.size());
  rocshmem::rocshmem_init_attr_t attr;
  int ret = rocshmem::rocshmem_set_attr_uniqueid_args(rank, nranks, &uid, &attr);
  if (ret != 0) return ret;
  return rocshmem::rocshmem_init_attr(rocshmem::ROCSHMEM_INIT_WITH_UNIQUEID, &attr);
}

extern "C" void fm_rocshmem_finalize() { rocshmem::rocshmem_finalize(); }
extern "C" int fm_rocshmem_my_pe() { return rocshmem::rocshmem_my_pe(); }
extern "C" int fm_rocshmem_n_pes() { return rocshmem::rocshmem_n_pes(); }
extern "C" void* fm_rocshmem_malloc(uint64_t size) { return rocshmem::rocshmem_malloc(static_cast<size_t>(size)); }
extern "C" void* fm_rocshmem_calloc(uint64_t count, uint64_t size) { return rocshmem::rocshmem_calloc(static_cast<size_t>(count), static_cast<size_t>(size)); }
extern "C" void fm_rocshmem_free(void* ptr) { rocshmem::rocshmem_free(ptr); }
extern "C" void fm_rocshmem_barrier_all() { rocshmem::rocshmem_barrier_all(); }
extern "C" int fm_rocshmem_hipmodule_init(uintptr_t module) { return rocshmem::rocshmem_hipmodule_init(reinterpret_cast<hipModule_t>(module), nullptr); }
extern "C" int fm_hip_memset(uintptr_t ptr, int value, uint64_t size) { return static_cast<int>(hipMemset(reinterpret_cast<void*>(ptr), value, static_cast<size_t>(size))); }
extern "C" int fm_hip_device_synchronize() { return static_cast<int>(hipDeviceSynchronize()); }
extern "C" int fm_hip_set_device(int dev) { return static_cast<int>(hipSetDevice(dev)); }
extern "C" int fm_hip_get_last_error() { return static_cast<int>(hipGetLastError()); }
extern "C" const char* fm_hip_get_error_string(int code) { return hipGetErrorString(static_cast<hipError_t>(code)); }
"""


def _rocm_root() -> str:
    try:
        return subprocess.check_output(["rocm-sdk", "path", "--root"], text=True).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("rocm-sdk path --root is required to build the rocSHMEM runtime shim") from exc


def _cache_dir() -> Path:
    root = os.environ.get("FLASHMOE_ROCSHMEM_RUNTIME_CACHE")
    if root:
        return Path(root)
    return Path("/tmp") / "flashmoe_rocshmem_runtime"


def _find_devel_root() -> Path:
    candidates = [
        Path(sysconfig.get_path("purelib")) / "_rocm_sdk_devel",
        Path(_rocm_root()),
    ]
    for root in candidates:
        if (root / "include" / "rocshmem" / "rocshmem.hpp").exists() and (root / "lib" / "librocshmem.a").exists():
            return root
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"could not find rocSHMEM headers/static library; searched: {searched}")


def _build_shared() -> Path:
    rocm = Path(_rocm_root())
    devel = _find_devel_root()
    digest = hashlib.sha256((_CPP_SOURCE + str(rocm) + str(devel)).encode()).hexdigest()[:16]
    build_dir = _cache_dir() / digest
    lib_path = build_dir / "libflashmoe_rocshmem_runtime.so"
    if lib_path.exists():
        return lib_path
    build_dir.mkdir(parents=True, exist_ok=True)
    source_path = build_dir / "flashmoe_rocshmem_runtime.cpp"
    source_path.write_text(_CPP_SOURCE)
    core_lib = Path(sysconfig.get_path("purelib")) / "_rocm_sdk_core" / "lib"
    sysdeps_lib = rocm / "lib" / "rocm_sysdeps" / "lib"
    cmd = [
        str(rocm / "bin" / "hipcc"),
        "-std=c++20",
        "-O2",
        "-shared",
        "-fPIC",
        "-fgpu-rdc",
        "--offload-arch=gfx1250",
        f"-I{devel / 'include'}",
        f"-I{rocm / 'include'}",
        str(source_path),
        "-o",
        str(lib_path),
        f"-L{devel / 'lib'}",
        f"-L{rocm / 'lib'}",
        f"-L{core_lib}",
        f"-L{sysdeps_lib}",
        "-lrocshmem",
        "-lamdhip64",
        "-lhsa-runtime64",
        "-lnuma",
        "-lpthread",
        "-ldl",
        "-lrt",
    ]
    env = os.environ.copy()
    ld_paths = [str(rocm / "lib"), str(core_lib), str(sysdeps_lib)]
    if env.get("LD_LIBRARY_PATH"):
        ld_paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_paths)
    subprocess.check_call(cmd, env=env)
    return lib_path


def _check(code: int, what: str) -> None:
    if code != 0:
        raise RuntimeError(f"{what} failed with code {code}")


class DevicePointer:
    def __init__(self, ptr: int, dtype: torch.dtype, nbytes: int, owner: object | None = None):
        self.ptr = int(ptr)
        self.dtype = dtype
        self.nbytes = int(nbytes)
        self._owner = owner

    def data_ptr(self) -> int:
        return self.ptr


@dataclass
class RocshmemMegakernelContext:
    runtime: "RocshmemRuntime"
    world_size: int
    rank: int
    local_experts: int
    expert_capacity: int
    hidden_size: int
    dtype: torch.dtype
    dispatch_tokens: DevicePointer
    dispatch_token_ids: DevicePointer
    dispatch_probs: DevicePointer
    dispatch_counts: DevicePointer
    dispatch_signals: DevicePointer
    result_values: DevicePointer
    result_counts: DevicePointer
    result_signals: DevicePointer

    @classmethod
    def create(
        cls,
        runtime: "RocshmemRuntime",
        *,
        world_size: int,
        rank: int,
        local_experts: int,
        expert_capacity: int,
        hidden_size: int,
        dtype: torch.dtype,
    ) -> "RocshmemMegakernelContext":
        elem_size = torch.empty((), dtype=dtype).element_size()
        channels = world_size * world_size * local_experts
        routes = channels * expert_capacity
        matrix_elems = routes * hidden_size
        return cls(
            runtime=runtime,
            world_size=world_size,
            rank=rank,
            local_experts=local_experts,
            expert_capacity=expert_capacity,
            hidden_size=hidden_size,
            dtype=dtype,
            dispatch_tokens=runtime.malloc(matrix_elems * elem_size, dtype),
            dispatch_token_ids=runtime.malloc(routes * 4, torch.int32),
            dispatch_probs=runtime.malloc(routes * 4, torch.float32),
            dispatch_counts=runtime.malloc(channels * 4, torch.int32),
            dispatch_signals=runtime.malloc(channels * 8, torch.uint64),
            result_values=runtime.malloc(matrix_elems * 4, torch.float32),
            result_counts=runtime.malloc(channels * 4, torch.int32),
            result_signals=runtime.malloc(channels * 8, torch.uint64),
        )

    def zero_(self) -> None:
        for ptr in (
            self.dispatch_counts,
            self.dispatch_signals,
            self.result_counts,
            self.result_signals,
        ):
            self.runtime.memset(ptr, 0)
        self.runtime.synchronize()
        self.runtime.barrier_all()

    def close(self) -> None:
        seen: set[int] = set()
        for ptr in (
            self.dispatch_tokens,
            self.dispatch_token_ids,
            self.dispatch_probs,
            self.dispatch_counts,
            self.dispatch_signals,
            self.result_values,
            self.result_counts,
            self.result_signals,
        ):
            if ptr.ptr not in seen:
                seen.add(ptr.ptr)
                self.runtime.free(ptr)


class RocshmemRuntime:
    def __init__(self, lib_path: str | Path | None = None):
        self.lib_path = Path(lib_path) if lib_path is not None else _build_shared()
        self.lib = ctypes.CDLL(str(self.lib_path), mode=ctypes.RTLD_GLOBAL)
        self._configure_abi()
        self.initialized = False

    def _configure_abi(self) -> None:
        c = self.lib
        c.fm_rocshmem_get_uniqueid.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
        c.fm_rocshmem_get_uniqueid.restype = ctypes.c_int
        c.fm_rocshmem_init_uniqueid.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint8)]
        c.fm_rocshmem_init_uniqueid.restype = ctypes.c_int
        c.fm_rocshmem_finalize.argtypes = []
        c.fm_rocshmem_finalize.restype = None
        c.fm_rocshmem_my_pe.argtypes = []
        c.fm_rocshmem_my_pe.restype = ctypes.c_int
        c.fm_rocshmem_n_pes.argtypes = []
        c.fm_rocshmem_n_pes.restype = ctypes.c_int
        c.fm_rocshmem_malloc.argtypes = [ctypes.c_uint64]
        c.fm_rocshmem_malloc.restype = ctypes.c_void_p
        c.fm_rocshmem_calloc.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        c.fm_rocshmem_calloc.restype = ctypes.c_void_p
        c.fm_rocshmem_free.argtypes = [ctypes.c_void_p]
        c.fm_rocshmem_free.restype = None
        c.fm_rocshmem_barrier_all.argtypes = []
        c.fm_rocshmem_barrier_all.restype = None
        c.fm_rocshmem_hipmodule_init.argtypes = [ctypes.c_size_t]
        c.fm_rocshmem_hipmodule_init.restype = ctypes.c_int
        c.fm_hip_memset.argtypes = [ctypes.c_size_t, ctypes.c_int, ctypes.c_uint64]
        c.fm_hip_memset.restype = ctypes.c_int
        c.fm_hip_device_synchronize.argtypes = []
        c.fm_hip_device_synchronize.restype = ctypes.c_int
        c.fm_hip_set_device.argtypes = [ctypes.c_int]
        c.fm_hip_set_device.restype = ctypes.c_int
        c.fm_hip_get_error_string.argtypes = [ctypes.c_int]
        c.fm_hip_get_error_string.restype = ctypes.c_char_p

    def set_device(self, device: int) -> None:
        _check(self.lib.fm_hip_set_device(device), "hipSetDevice")

    def get_uniqueid(self) -> bytes:
        uid = (ctypes.c_uint8 * _ROCSHMEM_UNIQUE_ID_BYTES)()
        _check(self.lib.fm_rocshmem_get_uniqueid(uid), "rocshmem_get_uniqueid")
        return bytes(uid)

    def init_uniqueid(self, rank: int, world_size: int, uid: bytes | bytearray | str) -> None:
        if isinstance(uid, str):
            uid = bytes.fromhex(uid)
        if len(uid) != _ROCSHMEM_UNIQUE_ID_BYTES:
            raise ValueError(f"rocSHMEM unique id must be {_ROCSHMEM_UNIQUE_ID_BYTES} bytes")
        uid_buf = (ctypes.c_uint8 * _ROCSHMEM_UNIQUE_ID_BYTES).from_buffer_copy(bytes(uid))
        _check(self.lib.fm_rocshmem_init_uniqueid(rank, world_size, uid_buf), "rocshmem_init_attr")
        self.initialized = True

    def finalize(self) -> None:
        if self.initialized:
            self.lib.fm_rocshmem_finalize()
            self.initialized = False

    def my_pe(self) -> int:
        return int(self.lib.fm_rocshmem_my_pe())

    def n_pes(self) -> int:
        return int(self.lib.fm_rocshmem_n_pes())

    def malloc(self, nbytes: int, dtype: torch.dtype) -> DevicePointer:
        ptr = self.lib.fm_rocshmem_malloc(int(nbytes))
        if not ptr:
            raise MemoryError(f"rocshmem_malloc failed for {nbytes} bytes")
        return DevicePointer(int(ptr), dtype, int(nbytes), self)

    def calloc(self, count: int, size: int, dtype: torch.dtype) -> DevicePointer:
        ptr = self.lib.fm_rocshmem_calloc(int(count), int(size))
        if not ptr:
            raise MemoryError(f"rocshmem_calloc failed for {count} x {size} bytes")
        return DevicePointer(int(ptr), dtype, int(count) * int(size), self)

    def free(self, ptr: DevicePointer) -> None:
        if ptr.ptr:
            self.lib.fm_rocshmem_free(ctypes.c_void_p(ptr.ptr))
            ptr.ptr = 0

    def memset(self, ptr: DevicePointer, value: int = 0) -> None:
        _check(self.lib.fm_hip_memset(ptr.ptr, value, ptr.nbytes), "hipMemset")

    def synchronize(self) -> None:
        _check(self.lib.fm_hip_device_synchronize(), "hipDeviceSynchronize")

    def barrier_all(self) -> None:
        self.lib.fm_rocshmem_barrier_all()

    def hipmodule_init(self, module: int) -> None:
        _check(self.lib.fm_rocshmem_hipmodule_init(int(module)), "rocshmem_hipmodule_init")


def create_uniqueid() -> bytes:
    return RocshmemRuntime().get_uniqueid()
