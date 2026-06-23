from __future__ import annotations

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
_VOID_PTR = gl.pointer_type(gl.void)
_I64_PTR = gl.pointer_type(gl.int64)
_U64_PTR = gl.pointer_type(gl.uint64)


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


def extern_libs(arch: str = "gfx1250") -> dict[str, str]:
    return {_LIB_NAME: find_device_bitcode(arch)}


def launch_kwargs(arch: str = "gfx1250") -> dict[str, dict[str, str]]:
    return {"extern_libs": extern_libs(arch)}


def _dispatch(lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, is_pure: bool, _semantic):
    if not arg_type_symbol_dict:
        raise ValueError("arg_type_symbol_dict is empty")
    num_args = len(next(iter(arg_type_symbol_dict.keys())))
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
    if arg_types not in arg_type_symbol_dict:
        raise ValueError(f"extern call type mismatch: expected {tuple(arg_type_symbol_dict.keys())}, got {arg_types}")

    symbol, ret_types = arg_type_symbol_dict[arg_types]
    if not isinstance(ret_types, (list, tuple)):
        ret_types = [ret_types]
    if not symbol:
        raise ValueError("extern call symbol cannot be empty")

    call = _semantic.builder.create_extern_call(
        lib_name,
        lib_path,
        symbol,
        arg_handles,
        [ret_type.to_ir(_semantic.builder) for ret_type in ret_types],
        is_pure,
    )
    if len(ret_types) == 0:
        return tensor(call, gl.void)
    if len(ret_types) == 1:
        return tensor(call.get_result(0), ret_types[0])
    return tuple(tensor(call.get_result(i), ty) for i, ty in enumerate(ret_types))


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
def remote_ptr(local_ptr, pe, _semantic=None):
    ptr = extern_call(
        _LIB_NAME,
        "",
        [
            gl.cast(local_ptr, _VOID_PTR, _semantic=_semantic),
            gl.cast(pe, gl.int32, _semantic=_semantic),
        ],
        {(_VOID_PTR, gl.int32): ("rocshmem_ptr", _VOID_PTR)},
        is_pure=False,
        _semantic=_semantic,
    )
    return gl.cast(ptr, local_ptr.dtype, _semantic=_semantic)


@builtin
def putmem_wg(dest, source, nbytes, pe, _semantic=None):
    return extern_call(
        _LIB_NAME,
        "",
        [
            gl.cast(dest, _VOID_PTR, _semantic=_semantic),
            gl.cast(source, _VOID_PTR, _semantic=_semantic),
            gl.cast(nbytes, gl.int64, _semantic=_semantic),
            gl.cast(pe, gl.int32, _semantic=_semantic),
        ],
        {(_VOID_PTR, _VOID_PTR, gl.int64, gl.int32): ("rocshmem_putmem_wg", gl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def putmem_nbi_wg(dest, source, nbytes, pe, _semantic=None):
    return extern_call(
        _LIB_NAME,
        "",
        [
            gl.cast(dest, _VOID_PTR, _semantic=_semantic),
            gl.cast(source, _VOID_PTR, _semantic=_semantic),
            gl.cast(nbytes, gl.int64, _semantic=_semantic),
            gl.cast(pe, gl.int32, _semantic=_semantic),
        ],
        {(_VOID_PTR, _VOID_PTR, gl.int64, gl.int32): ("rocshmem_putmem_nbi_wg", gl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def getmem_wg(dest, source, nbytes, pe, _semantic=None):
    return extern_call(
        _LIB_NAME,
        "",
        [
            gl.cast(dest, _VOID_PTR, _semantic=_semantic),
            gl.cast(source, _VOID_PTR, _semantic=_semantic),
            gl.cast(nbytes, gl.int64, _semantic=_semantic),
            gl.cast(pe, gl.int32, _semantic=_semantic),
        ],
        {(_VOID_PTR, _VOID_PTR, gl.int64, gl.int32): ("rocshmem_getmem_wg", gl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def getmem_nbi_wg(dest, source, nbytes, pe, _semantic=None):
    return extern_call(
        _LIB_NAME,
        "",
        [
            gl.cast(dest, _VOID_PTR, _semantic=_semantic),
            gl.cast(source, _VOID_PTR, _semantic=_semantic),
            gl.cast(nbytes, gl.int64, _semantic=_semantic),
            gl.cast(pe, gl.int32, _semantic=_semantic),
        ],
        {(_VOID_PTR, _VOID_PTR, gl.int64, gl.int32): ("rocshmem_getmem_nbi_wg", gl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def putmem_signal_wg(dest, source, nbytes, sig_addr, signal, sig_op, pe, _semantic=None):
    return extern_call(
        _LIB_NAME,
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
        {
            (_VOID_PTR, _VOID_PTR, gl.int64, _U64_PTR, gl.uint64, gl.int32, gl.int32): (
                "rocshmem_putmem_signal_wg",
                gl.int32,
            )
        },
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def putmem_signal_nbi_wg(dest, source, nbytes, sig_addr, signal, sig_op, pe, _semantic=None):
    return extern_call(
        _LIB_NAME,
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
        {
            (_VOID_PTR, _VOID_PTR, gl.int64, _U64_PTR, gl.uint64, gl.int32, gl.int32): (
                "rocshmem_putmem_signal_nbi_wg",
                gl.int32,
            )
        },
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def signal_wait_until(sig_addr, cmp_op, cmp_val, _semantic=None):
    return extern_call(
        _LIB_NAME,
        "",
        [
            gl.cast(sig_addr, _U64_PTR, _semantic=_semantic),
            gl.cast(cmp_op, gl.int32, _semantic=_semantic),
            gl.cast(cmp_val, gl.uint64, _semantic=_semantic),
        ],
        {(_U64_PTR, gl.int32, gl.uint64): ("rocshmem_uint64_wait_until", gl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def fence(_semantic=None):
    return extern_call(
        _LIB_NAME,
        "",
        [],
        {(): ("rocshmem_fence", gl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def quiet(_semantic=None):
    return extern_call(
        _LIB_NAME,
        "",
        [],
        {(): ("rocshmem_quiet", gl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def barrier_all(_semantic=None):
    return extern_call(
        _LIB_NAME,
        "",
        [],
        {(): ("rocshmem_barrier_all_wg", gl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def sync_all(_semantic=None):
    return extern_call(
        _LIB_NAME,
        "",
        [],
        {(): ("rocshmem_sync_all_wg", gl.int32)},
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
    "set_ctx",
    "my_pe",
    "n_pes",
    "remote_ptr",
    "putmem_wg",
    "putmem_nbi_wg",
    "getmem_wg",
    "getmem_nbi_wg",
    "putmem_signal_wg",
    "putmem_signal_nbi_wg",
    "signal_wait_until",
    "fence",
    "quiet",
    "barrier_all",
    "sync_all",
]
