from __future__ import annotations

import hashlib
import os
import subprocess
import sysconfig
from pathlib import Path

import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language._core import builtin, tensor


ROCSHMEM_CMP_EQ = 0
ROCSHMEM_CMP_NE = 1
ROCSHMEM_CMP_GT = 2
ROCSHMEM_CMP_GE = 3
ROCSHMEM_CMP_LT = 4
ROCSHMEM_CMP_LE = 5

ROCSHMEM_SIGNAL_SET = 0
ROCSHMEM_SIGNAL_ADD = 1

_LIB_NAME = "rocshmem"
_WRAPPER_LIB_NAME = "fmg_rocshmem"
_VOID_PTR = gl.pointer_type(gl.void)
_I64_PTR = gl.pointer_type(gl.int64)
_U64_PTR = gl.pointer_type(gl.uint64)

_WRAPPER_IR = r'''
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

declare void @rocshmem_putmem_wg(ptr, ptr, i64, i32)
declare void @rocshmem_putmem_wave(ptr, ptr, i64, i32)
declare void @rocshmem_putmem_nbi_wg(ptr, ptr, i64, i32)
declare void @rocshmem_getmem_wg(ptr, ptr, i64, i32)
declare void @rocshmem_getmem_nbi_wg(ptr, ptr, i64, i32)
declare void @rocshmem_putmem_signal_wg(ptr, ptr, i64, ptr, i64, i32, i32)
declare void @rocshmem_putmem_signal_wave(ptr, ptr, i64, ptr, i64, i32, i32)
declare void @rocshmem_putmem_signal_nbi_wg(ptr, ptr, i64, ptr, i64, i32, i32)
declare void @rocshmem_uint64_atomic_set(ptr, i64, i32)
declare void @rocshmem_uint64_atomic_add(ptr, i64, i32)
declare void @rocshmem_uint64_wait_until(ptr, i32, i64)
declare void @rocshmem_fence()
declare void @rocshmem_quiet()
declare void @rocshmem_barrier_all_wg()
declare void @rocshmem_sync_all_wg()
declare i32 @llvm.amdgcn.workitem.id.x()

define i32 @fmg_rocshmem_putmem_wg(ptr %dest, ptr %source, i64 %nbytes, i32 %pe) {
entry:
  call void @rocshmem_putmem_wg(ptr %dest, ptr %source, i64 %nbytes, i32 %pe)
  ret i32 0
}

define i32 @fmg_rocshmem_putmem_wave(ptr %dest, ptr %source, i64 %nbytes, i32 %pe) {
entry:
  call void @rocshmem_putmem_wave(ptr %dest, ptr %source, i64 %nbytes, i32 %pe)
  ret i32 0
}

define i32 @fmg_rocshmem_putmem_nbi_wg(ptr %dest, ptr %source, i64 %nbytes, i32 %pe) {
entry:
  call void @rocshmem_putmem_nbi_wg(ptr %dest, ptr %source, i64 %nbytes, i32 %pe)
  ret i32 0
}

define i32 @fmg_rocshmem_getmem_wg(ptr %dest, ptr %source, i64 %nbytes, i32 %pe) {
entry:
  call void @rocshmem_getmem_wg(ptr %dest, ptr %source, i64 %nbytes, i32 %pe)
  ret i32 0
}

define i32 @fmg_rocshmem_getmem_nbi_wg(ptr %dest, ptr %source, i64 %nbytes, i32 %pe) {
entry:
  call void @rocshmem_getmem_nbi_wg(ptr %dest, ptr %source, i64 %nbytes, i32 %pe)
  ret i32 0
}

define i32 @fmg_rocshmem_putmem_signal_wg(ptr %dest, ptr %source, i64 %nbytes, ptr %sig_addr, i64 %signal, i32 %sig_op, i32 %pe) {
entry:
  call void @rocshmem_putmem_signal_wg(ptr %dest, ptr %source, i64 %nbytes, ptr %sig_addr, i64 %signal, i32 %sig_op, i32 %pe)
  ret i32 0
}

define i32 @fmg_rocshmem_putmem_signal_wave(ptr %dest, ptr %source, i64 %nbytes, ptr %sig_addr, i64 %signal, i32 %sig_op, i32 %pe) {
entry:
  call void @rocshmem_putmem_signal_wave(ptr %dest, ptr %source, i64 %nbytes, ptr %sig_addr, i64 %signal, i32 %sig_op, i32 %pe)
  ret i32 0
}

define i32 @fmg_rocshmem_putmem_signal_nbi_wg(ptr %dest, ptr %source, i64 %nbytes, ptr %sig_addr, i64 %signal, i32 %sig_op, i32 %pe) {
entry:
  call void @rocshmem_putmem_signal_nbi_wg(ptr %dest, ptr %source, i64 %nbytes, ptr %sig_addr, i64 %signal, i32 %sig_op, i32 %pe)
  ret i32 0
}

define i32 @fmg_rocshmem_signal_op_wg(ptr %sig_addr, i64 %signal, i32 %sig_op, i32 %pe) {
entry:
  %is_set = icmp eq i32 %sig_op, 0
  br i1 %is_set, label %set, label %maybe_add

set:
  call void @rocshmem_uint64_atomic_set(ptr %sig_addr, i64 %signal, i32 %pe)
  br label %done

maybe_add:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %is_first = icmp eq i32 %tid, 0
  br i1 %is_first, label %issue, label %done

issue:
  %is_add = icmp eq i32 %sig_op, 1
  br i1 %is_add, label %add, label %done

add:
  call void @rocshmem_uint64_atomic_add(ptr %sig_addr, i64 %signal, i32 %pe)
  br label %done

done:
  ret i32 0
}

define i32 @fmg_rocshmem_signal_op_wave(ptr %sig_addr, i64 %signal, i32 %sig_op, i32 %pe) {
entry:
  %is_set = icmp eq i32 %sig_op, 0
  br i1 %is_set, label %set, label %maybe_add

set:
  call void @rocshmem_uint64_atomic_set(ptr %sig_addr, i64 %signal, i32 %pe)
  br label %done

maybe_add:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %is_first = icmp eq i32 %tid, 0
  br i1 %is_first, label %issue, label %done

issue:
  %is_add = icmp eq i32 %sig_op, 1
  br i1 %is_add, label %add, label %done

add:
  call void @rocshmem_uint64_atomic_add(ptr %sig_addr, i64 %signal, i32 %pe)
  br label %done

done:
  ret i32 0
}

define i32 @fmg_rocshmem_uint64_wait_until(ptr %sig_addr, i32 %cmp_op, i64 %cmp_val) {
entry:
  call void @rocshmem_uint64_wait_until(ptr %sig_addr, i32 %cmp_op, i64 %cmp_val)
  ret i32 0
}

define i32 @fmg_rocshmem_fence() {
entry:
  call void @rocshmem_fence()
  ret i32 0
}

define i32 @fmg_rocshmem_quiet() {
entry:
  call void @rocshmem_quiet()
  ret i32 0
}

define i32 @fmg_rocshmem_barrier_all_wg() {
entry:
  call void @rocshmem_barrier_all_wg()
  ret i32 0
}

define i32 @fmg_rocshmem_sync_all_wg() {
entry:
  call void @rocshmem_sync_all_wg()
  ret i32 0
}
'''


def _rocm_root_from_sdk() -> str | None:
    try:
        out = subprocess.check_output(["rocm-sdk", "path", "--root"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    path = out.strip()
    return path or None


def find_device_bitcode(arch: str = "gfx1250") -> str:
    lib_name = f"librocshmem_device_{arch}.bc"
    roots = [
        os.environ.get("ROCSHMEM_LIB_DIR"),
        os.environ.get("ROCM_ROOT"),
        os.environ.get("ROCM_PATH"),
        os.environ.get("ROCM_HOME"),
        _rocm_root_from_sdk(),
        str(Path(sysconfig.get_path("purelib")) / "_rocm_sdk_devel"),
        "/opt/rocm",
    ]
    candidates: list[Path] = []
    for root in roots:
        if not root:
            continue
        root_path = Path(root)
        candidates.append(root_path / lib_name)
        candidates.append(root_path / "lib" / lib_name)
    for path in candidates:
        if path.exists():
            return str(path)
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"could not find {lib_name}; searched: {searched}")


def _find_llvm_as() -> str:
    roots = [
        _rocm_root_from_sdk(),
        str(Path(sysconfig.get_path("purelib")) / "_rocm_sdk_devel"),
        os.environ.get("ROCM_ROOT"),
        os.environ.get("ROCM_PATH"),
        os.environ.get("ROCM_HOME"),
        "/opt/rocm",
    ]
    candidates: list[Path] = []
    for root in roots:
        if not root:
            continue
        root_path = Path(root)
        candidates.append(root_path / "lib" / "llvm" / "bin" / "llvm-as")
        candidates.append(root_path / "bin" / "llvm-as")
    for path in candidates:
        if path.exists():
            return str(path)
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"could not find llvm-as; searched: {searched}")


def find_abi_wrapper_bitcode(arch: str = "gfx1250") -> str:
    digest = hashlib.sha256((_WRAPPER_IR + arch).encode()).hexdigest()[:16]
    cache_root = Path(os.environ.get("FLASHMOE_ROCSHMEM_ABI_CACHE", "/tmp/flashmoe_rocshmem_abi"))
    build_dir = cache_root / digest
    bc_path = build_dir / f"fmg_rocshmem_abi_{arch}.bc"
    if bc_path.exists():
        return str(bc_path)
    build_dir.mkdir(parents=True, exist_ok=True)
    ir_path = build_dir / f"fmg_rocshmem_abi_{arch}.ll"
    ir_path.write_text(_WRAPPER_IR)
    subprocess.run([_find_llvm_as(), str(ir_path), "-o", str(bc_path)], check=True)
    return str(bc_path)


def extern_libs(arch: str = "gfx1250") -> dict[str, str]:
    return {
        _WRAPPER_LIB_NAME: find_abi_wrapper_bitcode(arch),
        _LIB_NAME: find_device_bitcode(arch),
    }


def launch_kwargs(arch: str = "gfx1250") -> dict[str, dict[str, str]]:
    return {"extern_libs": extern_libs(arch)}


_ROCSHMEM_KERNEL_NAMES: set[str] = set()
_PREVIOUS_POST_COMPILE_HOOK = None


def register_kernel(jit_func_or_name) -> None:
    name = jit_func_or_name if isinstance(jit_func_or_name, str) else jit_func_or_name.fn.__name__
    _ROCSHMEM_KERNEL_NAMES.add(name)


def install_module_init(runtime) -> None:
    import triton

    global _PREVIOUS_POST_COMPILE_HOOK
    previous = triton.knobs.runtime.jit_post_compile_hook
    if previous is not _module_init_hook:
        _PREVIOUS_POST_COMPILE_HOOK = previous
    _module_init_hook.runtime = runtime
    triton.knobs.runtime.jit_post_compile_hook = _module_init_hook


def _module_init_hook(**kwargs):
    if _PREVIOUS_POST_COMPILE_HOOK is not None:
        _PREVIOUS_POST_COMPILE_HOOK(**kwargs)
    jit_function = kwargs["fn"].jit_function
    fn_name = jit_function.fn.__name__
    if fn_name not in _ROCSHMEM_KERNEL_NAMES:
        return
    device = kwargs["compile"]["device"]
    key = kwargs["key"]
    kernel = jit_function.device_caches[device][0].get(key)
    if kernel is None:
        raise RuntimeError(f"compiled kernel for {fn_name} was not found in the Triton device cache")
    kernel.run
    runtime = getattr(_module_init_hook, "runtime", None)
    if runtime is None:
        raise RuntimeError("rocSHMEM module init hook installed without a runtime")
    runtime.hipmodule_init(int(kernel.module))
    setattr(kernel, "_flashmoe_rocshmem_module_initialized", True)


def _dispatch(lib_name: str, lib_path: str, args: list, arg_type_symbol_dict, is_pure: bool, _semantic):
    if not arg_type_symbol_dict:
        raise ValueError("arg_type_symbol_dict is empty")
    if isinstance(arg_type_symbol_dict, dict):
        cases = tuple(arg_type_symbol_dict.items())
    else:
        cases = tuple(arg_type_symbol_dict)
    num_args = len(cases[0][0])
    if len(args) != num_args:
        raise ValueError(f"extern call expected {num_args} args, got {len(args)}")

    arg_types = []
    arg_handles = []
    for arg in args:
        if isinstance(arg, tensor):
            arg_types.append(arg.dtype)
            arg_handles.append(arg.handle)
        else:
            arg_types.append(type(arg))
            arg_handles.append(arg)
    arg_types = tuple(arg_types)
    match = None
    for expected_types, value in cases:
        if arg_types == expected_types:
            match = value
            break
    if match is None:
        expected = tuple(expected_types for expected_types, _ in cases)
        raise ValueError(f"extern call type mismatch: expected {expected}, got {arg_types}")

    symbol, ret_types = match
    if not isinstance(ret_types, (list, tuple)):
        ret_types = [ret_types]
    if not symbol:
        raise ValueError("extern call symbol cannot be empty")

    if len(ret_types) != 1:
        raise ValueError("Gluon rocSHMEM extern calls currently expect one return value")
    ret_type = ret_types[0]
    call = _semantic.builder.create_extern_elementwise(
        lib_name,
        lib_path,
        symbol,
        arg_handles,
        ret_type.to_ir(_semantic.builder),
        is_pure,
    )
    return tensor(call, ret_type)


@builtin
def extern_call(lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, is_pure: bool, _semantic=None):
    dispatch_args = args.copy()
    for i in range(len(dispatch_args)):
        dispatch_args[i] = _semantic.to_tensor(dispatch_args[i])
        if dispatch_args[i].type.is_block():
            raise ValueError("Gluon rocSHMEM extern calls only accept scalar arguments")
    return _dispatch(lib_name, lib_path, dispatch_args, arg_type_symbol_dict, is_pure, _semantic)


@builtin
def set_ctx(ctx, _semantic=None):
    raise NotImplementedError(
        "rocSHMEM device bitcode does not export rocshmem_set_ctx; initialize the HIP module from the host runtime"
    )


@builtin
def my_pe(_semantic=None):
    return extern_call(
        _LIB_NAME,
        "",
        [],
        {(): ("rocshmem_my_pe", gl.int32)},
        is_pure=True,
        _semantic=_semantic,
    )


@builtin
def n_pes(_semantic=None):
    return extern_call(
        _LIB_NAME,
        "",
        [],
        {(): ("rocshmem_n_pes", gl.int32)},
        is_pure=True,
        _semantic=_semantic,
    )


@builtin
def putmem_wg(dest, source, nbytes, pe, _semantic=None):
    return extern_call(
        _WRAPPER_LIB_NAME,
        "",
        [
            gl.cast(dest, _VOID_PTR, _semantic=_semantic),
            gl.cast(source, _VOID_PTR, _semantic=_semantic),
            gl.cast(nbytes, gl.int64, _semantic=_semantic),
            gl.cast(pe, gl.int32, _semantic=_semantic),
        ],
        (((_VOID_PTR, _VOID_PTR, gl.int64, gl.int32), ("fmg_rocshmem_putmem_wg", gl.int32)),),
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def putmem_wave(dest, source, nbytes, pe, _semantic=None):
    return extern_call(
        _WRAPPER_LIB_NAME,
        "",
        [
            gl.cast(dest, _VOID_PTR, _semantic=_semantic),
            gl.cast(source, _VOID_PTR, _semantic=_semantic),
            gl.cast(nbytes, gl.int64, _semantic=_semantic),
            gl.cast(pe, gl.int32, _semantic=_semantic),
        ],
        (((_VOID_PTR, _VOID_PTR, gl.int64, gl.int32), ("fmg_rocshmem_putmem_wave", gl.int32)),),
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def putmem_nbi_wg(dest, source, nbytes, pe, _semantic=None):
    return extern_call(
        _WRAPPER_LIB_NAME,
        "",
        [
            gl.cast(dest, _VOID_PTR, _semantic=_semantic),
            gl.cast(source, _VOID_PTR, _semantic=_semantic),
            gl.cast(nbytes, gl.int64, _semantic=_semantic),
            gl.cast(pe, gl.int32, _semantic=_semantic),
        ],
        (((_VOID_PTR, _VOID_PTR, gl.int64, gl.int32), ("fmg_rocshmem_putmem_nbi_wg", gl.int32)),),
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def getmem_wg(dest, source, nbytes, pe, _semantic=None):
    return extern_call(
        _WRAPPER_LIB_NAME,
        "",
        [
            gl.cast(dest, _VOID_PTR, _semantic=_semantic),
            gl.cast(source, _VOID_PTR, _semantic=_semantic),
            gl.cast(nbytes, gl.int64, _semantic=_semantic),
            gl.cast(pe, gl.int32, _semantic=_semantic),
        ],
        (((_VOID_PTR, _VOID_PTR, gl.int64, gl.int32), ("fmg_rocshmem_getmem_wg", gl.int32)),),
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def getmem_nbi_wg(dest, source, nbytes, pe, _semantic=None):
    return extern_call(
        _WRAPPER_LIB_NAME,
        "",
        [
            gl.cast(dest, _VOID_PTR, _semantic=_semantic),
            gl.cast(source, _VOID_PTR, _semantic=_semantic),
            gl.cast(nbytes, gl.int64, _semantic=_semantic),
            gl.cast(pe, gl.int32, _semantic=_semantic),
        ],
        (((_VOID_PTR, _VOID_PTR, gl.int64, gl.int32), ("fmg_rocshmem_getmem_nbi_wg", gl.int32)),),
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def putmem_signal_wg(dest, source, nbytes, sig_addr, signal, sig_op, pe, _semantic=None):
    return extern_call(
        _WRAPPER_LIB_NAME,
        "",
        [
            gl.cast(dest, _VOID_PTR, _semantic=_semantic),
            gl.cast(source, _VOID_PTR, _semantic=_semantic),
            gl.cast(nbytes, gl.int64, _semantic=_semantic),
            gl.cast(sig_addr, _U64_PTR, _semantic=_semantic),
            gl.cast(signal, gl.uint64, _semantic=_semantic),
            gl.cast(sig_op, gl.int32, _semantic=_semantic),
            gl.cast(pe, gl.int32, _semantic=_semantic),
        ],
        (
            (
                (_VOID_PTR, _VOID_PTR, gl.int64, _U64_PTR, gl.uint64, gl.int32, gl.int32),
                ("fmg_rocshmem_putmem_signal_wg", gl.int32),
            ),
        ),
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def putmem_signal_wave(dest, source, nbytes, sig_addr, signal, sig_op, pe, _semantic=None):
    return extern_call(
        _WRAPPER_LIB_NAME,
        "",
        [
            gl.cast(dest, _VOID_PTR, _semantic=_semantic),
            gl.cast(source, _VOID_PTR, _semantic=_semantic),
            gl.cast(nbytes, gl.int64, _semantic=_semantic),
            gl.cast(sig_addr, _U64_PTR, _semantic=_semantic),
            gl.cast(signal, gl.uint64, _semantic=_semantic),
            gl.cast(sig_op, gl.int32, _semantic=_semantic),
            gl.cast(pe, gl.int32, _semantic=_semantic),
        ],
        (
            (
                (_VOID_PTR, _VOID_PTR, gl.int64, _U64_PTR, gl.uint64, gl.int32, gl.int32),
                ("fmg_rocshmem_putmem_signal_wave", gl.int32),
            ),
        ),
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def putmem_signal_nbi_wg(dest, source, nbytes, sig_addr, signal, sig_op, pe, _semantic=None):
    return extern_call(
        _WRAPPER_LIB_NAME,
        "",
        [
            gl.cast(dest, _VOID_PTR, _semantic=_semantic),
            gl.cast(source, _VOID_PTR, _semantic=_semantic),
            gl.cast(nbytes, gl.int64, _semantic=_semantic),
            gl.cast(sig_addr, _U64_PTR, _semantic=_semantic),
            gl.cast(signal, gl.uint64, _semantic=_semantic),
            gl.cast(sig_op, gl.int32, _semantic=_semantic),
            gl.cast(pe, gl.int32, _semantic=_semantic),
        ],
        (
            (
                (_VOID_PTR, _VOID_PTR, gl.int64, _U64_PTR, gl.uint64, gl.int32, gl.int32),
                ("fmg_rocshmem_putmem_signal_nbi_wg", gl.int32),
            ),
        ),
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def signal_op_wg(sig_addr, signal, sig_op, pe, _semantic=None):
    return extern_call(
        _WRAPPER_LIB_NAME,
        "",
        [
            gl.cast(sig_addr, _U64_PTR, _semantic=_semantic),
            gl.cast(signal, gl.uint64, _semantic=_semantic),
            gl.cast(sig_op, gl.int32, _semantic=_semantic),
            gl.cast(pe, gl.int32, _semantic=_semantic),
        ],
        (((_U64_PTR, gl.uint64, gl.int32, gl.int32), ("fmg_rocshmem_signal_op_wg", gl.int32)),),
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def signal_op_wave(sig_addr, signal, sig_op, pe, _semantic=None):
    return extern_call(
        _WRAPPER_LIB_NAME,
        "",
        [
            gl.cast(sig_addr, _U64_PTR, _semantic=_semantic),
            gl.cast(signal, gl.uint64, _semantic=_semantic),
            gl.cast(sig_op, gl.int32, _semantic=_semantic),
            gl.cast(pe, gl.int32, _semantic=_semantic),
        ],
        (((_U64_PTR, gl.uint64, gl.int32, gl.int32), ("fmg_rocshmem_signal_op_wave", gl.int32)),),
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def signal_wait_until(sig_addr, cmp_op, cmp_val, _semantic=None):
    return extern_call(
        _WRAPPER_LIB_NAME,
        "",
        [
            gl.cast(sig_addr, _U64_PTR, _semantic=_semantic),
            gl.cast(cmp_op, gl.int32, _semantic=_semantic),
            gl.cast(cmp_val, gl.uint64, _semantic=_semantic),
        ],
        (((_U64_PTR, gl.int32, gl.uint64), ("fmg_rocshmem_uint64_wait_until", gl.int32)),),
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def signal_fetch_wave(sig_addr, _semantic=None):
    return extern_call(
        _LIB_NAME,
        "",
        [
            gl.cast(sig_addr, _U64_PTR, _semantic=_semantic),
        ],
        (((_U64_PTR,), ("rocshmem_signal_fetch_wave", gl.uint64)),),
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def fence(_semantic=None):
    return extern_call(
        _WRAPPER_LIB_NAME,
        "",
        [],
        {(): ("fmg_rocshmem_fence", gl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def quiet(_semantic=None):
    return extern_call(
        _WRAPPER_LIB_NAME,
        "",
        [],
        {(): ("fmg_rocshmem_quiet", gl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def barrier_all(_semantic=None):
    return extern_call(
        _WRAPPER_LIB_NAME,
        "",
        [],
        {(): ("fmg_rocshmem_barrier_all_wg", gl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def sync_all(_semantic=None):
    return extern_call(
        _WRAPPER_LIB_NAME,
        "",
        [],
        {(): ("fmg_rocshmem_sync_all_wg", gl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


__all__ = [
    "ROCSHMEM_CMP_EQ",
    "ROCSHMEM_CMP_NE",
    "ROCSHMEM_CMP_GT",
    "ROCSHMEM_CMP_GE",
    "ROCSHMEM_CMP_LT",
    "ROCSHMEM_CMP_LE",
    "ROCSHMEM_SIGNAL_SET",
    "ROCSHMEM_SIGNAL_ADD",
    "extern_libs",
    "find_device_bitcode",
    "launch_kwargs",
    "register_kernel",
    "install_module_init",
    "set_ctx",
    "my_pe",
    "n_pes",
    "putmem_wave",
    "putmem_wg",
    "putmem_nbi_wg",
    "getmem_wg",
    "getmem_nbi_wg",
    "putmem_signal_wave",
    "putmem_signal_wg",
    "putmem_signal_nbi_wg",
    "signal_op_wave",
    "signal_op_wg",
    "signal_wait_until",
    "signal_fetch_wave",
    "fence",
    "quiet",
    "barrier_all",
    "sync_all",
]
