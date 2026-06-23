from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flashmoe_gluon as fmg
from flashmoe_gluon import rocshmem


_REQUIRED_ROCSHMEM_SYMBOLS = (
    "rocshmem_my_pe",
    "rocshmem_n_pes",
    "rocshmem_ptr",
    "rocshmem_putmem_wg",
    "rocshmem_putmem_nbi_wg",
    "rocshmem_getmem_wg",
    "rocshmem_getmem_nbi_wg",
    "rocshmem_putmem_signal_wg",
    "rocshmem_putmem_signal_nbi_wg",
    "rocshmem_uint64_wait_until",
    "rocshmem_fence",
    "rocshmem_quiet",
    "rocshmem_barrier_all_wg",
    "rocshmem_sync_all_wg",
)


def _llvm_nm() -> Path:
    root = subprocess.check_output(["rocm-sdk", "path", "--root"], text=True).strip()
    return Path(root) / "lib" / "llvm" / "bin" / "llvm-nm"


def _check_rocshmem_symbols() -> tuple[bool, list[str], Path]:
    bitcode = Path(rocshmem.find_device_bitcode("gfx1250"))
    output = subprocess.check_output([str(_llvm_nm()), str(bitcode)], text=True)
    missing = [symbol for symbol in _REQUIRED_ROCSHMEM_SYMBOLS if f" T {symbol}" not in output]
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
