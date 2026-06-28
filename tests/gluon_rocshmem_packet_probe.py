from __future__ import annotations

import argparse
import os
import select
import signal as signal_module
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashmoe_gluon import rocshmem
from flashmoe_gluon.rocshmem_runtime import RocshmemRuntime, create_uniqueid


WORLD = int(os.environ.get("FLASHMOE_WORLD", "2"))
N = 16
PAYLOAD = 12345


@gluon.jit
def _packet_put_kernel(send, recv, signal, status, MODE: gl.constexpr, N_ELEMS: gl.constexpr, PAYLOAD_VALUE: gl.constexpr):
    pid = gl.program_id(0)
    my_pe = rocshmem.my_pe()
    peer = 1 - my_pe
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], [])
    offs = gl.arange(0, N_ELEMS, layout=layout)
    values = my_pe * 1000 + offs
    gl.store(send + offs, values)
    gl.store(status + 0, my_pe)
    gl.store(status + 1, 1)
    if MODE == "wg":
        rocshmem.putmem_signal_wg(
            recv,
            send,
            N_ELEMS * 4,
            signal,
            gl.full((), PAYLOAD_VALUE, gl.uint64),
            rocshmem.ROCSHMEM_SIGNAL_SET,
            peer,
        )
    elif MODE == "wg-seq":
        rocshmem.putmem_wg(recv, send, 4, peer)
        rocshmem.putmem_wg(recv + 1, send + 1, 4, peer)
        rocshmem.putmem_wg(recv + 2, send + 2, 4, peer)
        rocshmem.fence()
        rocshmem.putmem_signal_wg(
            recv,
            send,
            N_ELEMS * 4,
            signal,
            gl.full((), PAYLOAD_VALUE, gl.uint64),
            rocshmem.ROCSHMEM_SIGNAL_SET,
            peer,
        )
    elif MODE == "wave":
        rocshmem.putmem_signal_wave(
            recv,
            send,
            N_ELEMS * 4,
            signal,
            gl.full((), PAYLOAD_VALUE, gl.uint64),
            rocshmem.ROCSHMEM_SIGNAL_SET,
            peer,
        )
    elif MODE == "wave-seq":
        rocshmem.putmem_wave(recv, send, 4, peer)
        rocshmem.fence()
        rocshmem.putmem_signal_wave(
            recv,
            send,
            N_ELEMS * 4,
            signal,
            gl.full((), PAYLOAD_VALUE, gl.uint64),
            rocshmem.ROCSHMEM_SIGNAL_SET,
            peer,
        )
    elif MODE == "signal-wg":
        rocshmem.signal_op_wg(
            signal,
            gl.full((), PAYLOAD_VALUE, gl.uint64),
            rocshmem.ROCSHMEM_SIGNAL_SET,
            peer,
        )
    elif MODE == "signal-wave":
        rocshmem.signal_op_wave(
            signal,
            gl.full((), PAYLOAD_VALUE, gl.uint64),
            rocshmem.ROCSHMEM_SIGNAL_SET,
            peer,
        )
    gl.store(status + 1, 2)


@gluon.jit
def _packet_check_kernel(recv, signal, status, N_ELEMS: gl.constexpr):
    sig = gl.load(signal)
    first = gl.load(recv + 0)
    last = gl.load(recv + (N_ELEMS - 1))
    gl.store(status + 2, sig.to(gl.int64))
    gl.store(status + 3, first.to(gl.int64))
    gl.store(status + 4, last.to(gl.int64))


def _run_kernel(runtime: RocshmemRuntime, rank: int, mode: str, send, recv, signal, status: torch.Tensor) -> None:
    rocshmem.register_kernel(_packet_put_kernel)
    rocshmem.install_module_init(runtime)

    for preload_rank in range(WORLD):
        if rank == preload_rank:
            kernel = _packet_put_kernel.warmup(
                send,
                recv,
                signal,
                status,
                MODE=mode,
                N_ELEMS=N,
                PAYLOAD_VALUE=PAYLOAD,
                num_warps=4,
                extern_libs=rocshmem.extern_libs("gfx1250"),
                grid=(1,),
            )
            if kernel is not None:
                _ = kernel.run
                if not getattr(kernel, "_flashmoe_rocshmem_module_initialized", False):
                    runtime.hipmodule_init(int(kernel.module))
                    setattr(kernel, "_flashmoe_rocshmem_module_initialized", True)
            check_kernel = _packet_check_kernel.warmup(
                recv,
                signal,
                status,
                N_ELEMS=N,
                num_warps=4,
                grid=(1,),
            )
        runtime.barrier_all()

    _packet_put_kernel[(1,)](
        send,
        recv,
        signal,
        status,
        MODE=mode,
        N_ELEMS=N,
        PAYLOAD_VALUE=PAYLOAD,
        num_warps=4,
        extern_libs=rocshmem.extern_libs("gfx1250"),
    )
    torch.cuda.synchronize()
    runtime.barrier_all()
    _packet_check_kernel[(1,)](
        recv,
        signal,
        status,
        N_ELEMS=N,
        num_warps=4,
    )
    torch.cuda.synchronize()


def _worker() -> None:
    rank = int(os.environ["FLASHMOE_RANK"])
    uid = bytes.fromhex(os.environ["FLASHMOE_ROCSHMEM_UID"])
    mode = os.environ["FLASHMOE_PACKET_MODE"]

    def log(*items: object) -> None:
        print("rank", rank, *items, flush=True)

    try:
        torch.cuda.set_device(0)
        runtime = RocshmemRuntime()
        runtime.set_device(0)
        runtime.init_uniqueid(rank, WORLD, uid)
        log("init", runtime.my_pe(), runtime.n_pes())
        send = runtime.malloc(N * 4, torch.int32)
        recv = runtime.calloc(N, 4, torch.int32)
        signal = runtime.calloc(1, 8, torch.uint64)
        status = torch.zeros((8,), device="cuda", dtype=torch.int64)
        runtime.barrier_all()
        log("launch", mode)
        _run_kernel(runtime, rank, mode, send, recv, signal, status)
        got = status.cpu().tolist()
        log("status", got)
        expected_first = (1 - rank) * 1000
        expected_last = expected_first + N - 1
        if mode.startswith("signal-"):
            ok = got[1] == 2 and got[2] == PAYLOAD
        else:
            ok = got[1] == 2 and got[2] == PAYLOAD and got[3] == expected_first and got[4] == expected_last
        runtime.free(send)
        runtime.free(recv)
        runtime.free(signal)
        runtime.barrier_all()
        runtime.finalize()
        if not ok:
            raise SystemExit(2)
    except BaseException as exc:
        log("ERR", repr(exc))
        traceback.print_exc()
        raise


def _run_case(mode: str, timeout_s: float) -> None:
    uid = create_uniqueid().hex()
    procs: list[subprocess.Popen[str]] = []
    for rank in range(WORLD):
        env = os.environ.copy()
        env["FLASHMOE_RANK"] = str(rank)
        env["FLASHMOE_ROCSHMEM_UID"] = uid
        env["FLASHMOE_PACKET_MODE"] = mode
        procs.append(
            subprocess.Popen(
                [sys.executable, "-u", __file__, "--worker"],
                cwd=str(Path(__file__).resolve().parents[1]),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
        )

    deadline = time.time() + timeout_s
    while time.time() < deadline and any(proc.poll() is None for proc in procs):
        for proc in procs:
            while proc.stdout is not None:
                ready, _, _ = select.select([proc.stdout], [], [], 0)
                if not ready:
                    break
                line = proc.stdout.readline()
                if not line:
                    break
                print(line, end="", flush=True)
        time.sleep(0.1)

    for proc in procs:
        if proc.poll() is not None and proc.stdout is not None:
            print(proc.stdout.read(), end="", flush=True)

    alive = [proc.pid for proc in procs if proc.poll() is None]
    if alive:
        print("timeout", mode, "alive", alive, flush=True)
        for proc in procs:
            if proc.poll() is None:
                os.killpg(proc.pid, signal_module.SIGTERM)
        time.sleep(1)
        for proc in procs:
            if proc.poll() is None:
                os.killpg(proc.pid, signal_module.SIGKILL)
    for proc in procs:
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal_module.SIGKILL)
            proc.wait(timeout=2)
    exits = [(proc.pid, proc.returncode) for proc in procs]
    print("case", mode, "exits", exits, flush=True)
    if alive or any(proc.returncode for proc in procs):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("wg", "wg-seq", "wave", "wave-seq", "signal-wg", "signal-wave"),
        default="wg",
    )
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        _worker()
        return
    _run_case(args.mode, args.timeout_s)


if __name__ == "__main__":
    main()
