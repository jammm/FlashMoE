from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flashmoe_gluon as fmg
from flashmoe_gluon import rocshmem


_REQUIRED_ROCSHMEM_WRAPPER_SYMBOLS = (
    "fmg_rocshmem_my_pe",
    "fmg_rocshmem_n_pes",
    "fmg_rocshmem_putmem_wg",
    "fmg_rocshmem_putmem_nbi_wg",
    "fmg_rocshmem_getmem_wg",
    "fmg_rocshmem_getmem_nbi_wg",
    "fmg_rocshmem_putmem_signal_wg",
    "fmg_rocshmem_putmem_signal_nbi_wg",
    "fmg_rocshmem_signal_fetch_wave",
    "fmg_rocshmem_uint64_wait_until",
    "fmg_rocshmem_quiet",
    "fmg_rocshmem_barrier_all_wg",
    "fmg_rocshmem_sync_all_wg",
)

_REQUIRED_ROCSHMEM_DEVICE_SYMBOLS = (
    "rocshmem::rocshmem_my_pe()",
    "rocshmem::rocshmem_n_pes()",
    "rocshmem::rocshmem_putmem_wg(void*, void const*, unsigned long, int)",
    "rocshmem::rocshmem_putmem_nbi_wg(void*, void const*, unsigned long, int)",
    "rocshmem::rocshmem_getmem_wg(void*, void const*, unsigned long, int)",
    "rocshmem::rocshmem_getmem_nbi_wg(void*, void const*, unsigned long, int)",
    "rocshmem::rocshmem_putmem_signal_wg(void*, void const*, unsigned long, unsigned long*, unsigned long, int, int)",
    "rocshmem::rocshmem_putmem_signal_nbi_wg(void*, void const*, unsigned long, unsigned long*, unsigned long, int, int)",
    "rocshmem::rocshmem_signal_fetch_wave(unsigned long const*)",
    "rocshmem::rocshmem_uint64_wait_until(unsigned long*, int, unsigned long)",
    "rocshmem::rocshmem_quiet()",
    "rocshmem::rocshmem_barrier_all_wg()",
    "rocshmem::rocshmem_sync_all_wg()",
)


def _llvm_nm() -> Path:
    root = subprocess.check_output(["rocm-sdk", "path", "--root"], text=True).strip()
    return Path(root) / "lib" / "llvm" / "bin" / "llvm-nm"


def _check_rocshmem_symbols() -> tuple[bool, list[str], Path]:
    bitcode = Path(rocshmem.extern_libs("gfx1250")["fmg_rocshmem"])
    output = subprocess.check_output([str(_llvm_nm()), "--demangle", "--defined-only", str(bitcode)], text=True)
    required = (*_REQUIRED_ROCSHMEM_WRAPPER_SYMBOLS, *_REQUIRED_ROCSHMEM_DEVICE_SYMBOLS)
    missing = [symbol for symbol in required if symbol not in output]
    return not missing, missing, bitcode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-symbols", action="store_true")
    args = parser.parse_args()

    status = fmg.backend_status()
    print(
        "gluon_backend",
        "available",
        status.available,
        "reason",
        status.reason,
        "root",
        status.triton_root,
        "moe_example",
        status.moe_example,
    )

    bitcode = rocshmem.find_device_bitcode("gfx1250")
    print("rocshmem_bitcode", bitcode)
    if args.check_symbols:
        ok, missing, path = _check_rocshmem_symbols()
        print("rocshmem_symbols", "ok", ok, "path", path, "missing", missing)
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
