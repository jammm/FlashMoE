from __future__ import annotations

import argparse
import os
import select
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.gfx1250 import tdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashmoe_gluon import rocshmem
from flashmoe_gluon.rocshmem_runtime import RocshmemRuntime, create_uniqueid


WORLD = 2
NLX = 1
EC = 2
H = 64
EPOCH = 7
MAX_ITERS = int(os.environ.get("FLASHMOE_PROBE_MAX_ITERS", "10000000"))


@gluon.jit
def _signal_payload(epoch, count):
    return epoch.to(gl.uint64) * gl.full((), 1048576, gl.uint64) + count.to(gl.uint64)


@gluon.jit
def _signal_count(signal, epoch):
    return (signal - epoch.to(gl.uint64) * gl.full((), 1048576, gl.uint64)).to(gl.int32)


@gluon.jit
def _cross_rank_exchange_kernel(
    dispatch_tokens,
    dispatch_token_ids,
    dispatch_probs,
    dispatch_counts,
    dispatch_signals,
    result_values,
    result_counts,
    result_signals,
    status,
    MODE: gl.constexpr,
    MAX_POLL_ITERS: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    LOCAL_EXPERTS: gl.constexpr,
    EXPERT_CAPACITY: gl.constexpr,
    HIDDEN_SIZE: gl.constexpr,
    EPOCH_VALUE: gl.constexpr,
):
    my_pe = rocshmem.my_pe()
    peer = 1 - my_pe
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], [])
    offs = gl.arange(0, EXPERT_CAPACITY * HIDDEN_SIZE, layout=layout)
    small_offs = gl.arange(0, EXPERT_CAPACITY, layout=layout)
    epoch = gl.full((), EPOCH_VALUE, gl.int32)
    count = my_pe * 0 + EXPERT_CAPACITY
    payload = _signal_payload(epoch, count)

    if MODE == "dispatch":
        send_idx = (peer * WORLD_SIZE + my_pe) * LOCAL_EXPERTS
        recv_idx = (my_pe * WORLD_SIZE + peer) * LOCAL_EXPERTS
        token_vals = (my_pe * 1000 + offs).to(dispatch_tokens.dtype.element_ty)
        gl.store(dispatch_counts + send_idx, count)
        gl.store(dispatch_token_ids + send_idx * EXPERT_CAPACITY + small_offs, my_pe * 100 + small_offs)
        gl.store(dispatch_probs + send_idx * EXPERT_CAPACITY + small_offs, (my_pe * 10 + small_offs).to(gl.float32))
        gl.store(dispatch_tokens + send_idx * EXPERT_CAPACITY * HIDDEN_SIZE + offs, token_vals)
        rocshmem.putmem_wg(dispatch_counts + send_idx, dispatch_counts + send_idx, 4, peer)
        rocshmem.putmem_wg(
            dispatch_token_ids + send_idx * EXPERT_CAPACITY,
            dispatch_token_ids + send_idx * EXPERT_CAPACITY,
            EXPERT_CAPACITY * 4,
            peer,
        )
        rocshmem.putmem_wg(
            dispatch_probs + send_idx * EXPERT_CAPACITY,
            dispatch_probs + send_idx * EXPERT_CAPACITY,
            EXPERT_CAPACITY * 4,
            peer,
        )
        rocshmem.fence()
        rocshmem.putmem_signal_wg(
            dispatch_tokens + send_idx * EXPERT_CAPACITY * HIDDEN_SIZE,
            dispatch_tokens + send_idx * EXPERT_CAPACITY * HIDDEN_SIZE,
            EXPERT_CAPACITY * HIDDEN_SIZE * 2,
            dispatch_signals + send_idx,
            payload,
            rocshmem.ROCSHMEM_SIGNAL_SET,
            peer,
        )

        seen = gl.full((), 0, gl.int32)
        iters = gl.full((), 0, gl.int32)
        signal = gl.full((), 0, gl.uint64)
        while (seen == 0) & (iters < MAX_POLL_ITERS):
            signal = rocshmem.signal_fetch_wave(dispatch_signals + recv_idx)
            if signal >= payload:
                seen = 1
            iters += 1
        recv_count = gl.load(dispatch_counts + recv_idx)
        first_id = gl.load(dispatch_token_ids + recv_idx * EXPERT_CAPACITY)
        second_id = gl.load(dispatch_token_ids + recv_idx * EXPERT_CAPACITY + 1)
        first_token = gl.load(dispatch_tokens + recv_idx * EXPERT_CAPACITY * HIDDEN_SIZE).to(gl.float32).to(gl.int64)
        last_token = gl.load(
            dispatch_tokens + recv_idx * EXPERT_CAPACITY * HIDDEN_SIZE + (EXPERT_CAPACITY * HIDDEN_SIZE - 1)
        ).to(gl.float32).to(gl.int64)
        gl.store(status + 0, my_pe.to(gl.int64))
        gl.store(status + 1, seen.to(gl.int64))
        gl.store(status + 2, iters.to(gl.int64))
        gl.store(status + 3, _signal_count(signal, epoch).to(gl.int64))
        gl.store(status + 4, recv_count.to(gl.int64))
        gl.store(status + 5, first_id.to(gl.int64))
        gl.store(status + 6, second_id.to(gl.int64))
        gl.store(status + 7, first_token)
        gl.store(status + 8, last_token)
    elif MODE == "result":
        send_idx = (my_pe * WORLD_SIZE + peer) * LOCAL_EXPERTS
        recv_idx = (peer * WORLD_SIZE + my_pe) * LOCAL_EXPERTS
        result_vals = (my_pe * 1000 + offs).to(gl.float32)
        gl.store(result_counts + send_idx, count)
        gl.store(result_values + send_idx * EXPERT_CAPACITY * HIDDEN_SIZE + offs, result_vals)
        rocshmem.putmem_wave(result_counts + send_idx, result_counts + send_idx, 4, peer)
        rocshmem.fence()
        rocshmem.putmem_signal_wave(
            result_values + send_idx * EXPERT_CAPACITY * HIDDEN_SIZE,
            result_values + send_idx * EXPERT_CAPACITY * HIDDEN_SIZE,
            EXPERT_CAPACITY * HIDDEN_SIZE * 4,
            result_signals + send_idx,
            payload,
            rocshmem.ROCSHMEM_SIGNAL_SET,
            peer,
        )

        seen = gl.full((), 0, gl.int32)
        iters = gl.full((), 0, gl.int32)
        signal = gl.full((), 0, gl.uint64)
        while (seen == 0) & (iters < MAX_POLL_ITERS):
            signal = rocshmem.signal_fetch_wave(result_signals + recv_idx)
            if signal >= payload:
                seen = 1
            iters += 1
        recv_count = gl.load(result_counts + recv_idx)
        first_value = gl.load(result_values + recv_idx * EXPERT_CAPACITY * HIDDEN_SIZE).to(gl.int64)
        last_value = gl.load(
            result_values + recv_idx * EXPERT_CAPACITY * HIDDEN_SIZE + (EXPERT_CAPACITY * HIDDEN_SIZE - 1)
        ).to(gl.int64)
        gl.store(status + 0, my_pe.to(gl.int64))
        gl.store(status + 1, seen.to(gl.int64))
        gl.store(status + 2, iters.to(gl.int64))
        gl.store(status + 3, _signal_count(signal, epoch).to(gl.int64))
        gl.store(status + 4, recv_count.to(gl.int64))
        gl.store(status + 5, first_value)
        gl.store(status + 6, last_value)
    elif MODE == "tdm-result":
        send_idx = (my_pe * WORLD_SIZE + peer) * LOCAL_EXPERTS
        recv_idx = (peer * WORLD_SIZE + my_pe) * LOCAL_EXPERTS
        block_m: gl.constexpr = 16
        block_n: gl.constexpr = 64
        shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0], [])
        blocked_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0], [])
        row_layout: gl.constexpr = gl.SliceLayout(1, blocked_layout)
        col_layout: gl.constexpr = gl.SliceLayout(0, blocked_layout)
        rows = gl.arange(0, block_m, layout=row_layout)
        cols = gl.arange(0, block_n, layout=col_layout)
        smem = gl.allocate_shared_memory(result_values.dtype.element_ty, (block_m, block_n), shared_layout)
        vals = (my_pe * 1000 + gl.expand_dims(rows, 1) * HIDDEN_SIZE + gl.expand_dims(cols, 0)).to(gl.float32)
        active_rows = rows < EXPERT_CAPACITY
        vals = gl.where(
            gl.expand_dims(active_rows, 1),
            vals,
            gl.full((block_m, block_n), -777.0, gl.float32, layout=blocked_layout),
        )
        smem.store(vals)
        result_desc = tdm.make_tensor_descriptor(
            base=result_values + send_idx * EXPERT_CAPACITY * HIDDEN_SIZE,
            shape=(EXPERT_CAPACITY, HIDDEN_SIZE),
            strides=(HIDDEN_SIZE, 1),
            block_shape=(block_m, block_n),
            layout=shared_layout,
        )
        tdm.async_store(result_desc, [0, 0], smem)
        tdm.async_wait(0)
        local_spill = gl.load(result_values + send_idx * EXPERT_CAPACITY * HIDDEN_SIZE + (block_m - 1) * HIDDEN_SIZE).to(
            gl.int64
        )
        gl.store(result_counts + send_idx, count)
        rocshmem.putmem_wave(result_counts + send_idx, result_counts + send_idx, 4, peer)
        rocshmem.fence()
        rocshmem.putmem_signal_wave(
            result_values + send_idx * EXPERT_CAPACITY * HIDDEN_SIZE,
            result_values + send_idx * EXPERT_CAPACITY * HIDDEN_SIZE,
            EXPERT_CAPACITY * HIDDEN_SIZE * 4,
            result_signals + send_idx,
            payload,
            rocshmem.ROCSHMEM_SIGNAL_SET,
            peer,
        )

        seen = gl.full((), 0, gl.int32)
        iters = gl.full((), 0, gl.int32)
        signal = gl.full((), 0, gl.uint64)
        while (seen == 0) & (iters < MAX_POLL_ITERS):
            signal = rocshmem.signal_fetch_wave(result_signals + recv_idx)
            if signal >= payload:
                seen = 1
            iters += 1
        recv_count = gl.load(result_counts + recv_idx)
        first_value = gl.load(result_values + recv_idx * EXPERT_CAPACITY * HIDDEN_SIZE).to(gl.int64)
        last_value = gl.load(
            result_values + recv_idx * EXPERT_CAPACITY * HIDDEN_SIZE + (EXPERT_CAPACITY * HIDDEN_SIZE - 1)
        ).to(gl.int64)
        gl.store(status + 0, my_pe.to(gl.int64))
        gl.store(status + 1, seen.to(gl.int64))
        gl.store(status + 2, iters.to(gl.int64))
        gl.store(status + 3, _signal_count(signal, epoch).to(gl.int64))
        gl.store(status + 4, recv_count.to(gl.int64))
        gl.store(status + 5, first_value)
        gl.store(status + 6, last_value)
        gl.store(status + 7, local_spill)
    elif MODE == "tdm-queue-result":
        send_idx = (my_pe * WORLD_SIZE + peer) * LOCAL_EXPERTS
        recv_idx = (peer * WORLD_SIZE + my_pe) * LOCAL_EXPERTS
        queue_flag = dispatch_counts
        block_m: gl.constexpr = 16
        block_n: gl.constexpr = 64
        shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0], [])
        blocked_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0], [])
        row_layout: gl.constexpr = gl.SliceLayout(1, blocked_layout)
        col_layout: gl.constexpr = gl.SliceLayout(0, blocked_layout)

        if pid == 0:
            rows = gl.arange(0, block_m, layout=row_layout)
            cols = gl.arange(0, block_n, layout=col_layout)
            smem = gl.allocate_shared_memory(result_values.dtype.element_ty, (block_m, block_n), shared_layout)
            vals = (my_pe * 1000 + gl.expand_dims(rows, 1) * HIDDEN_SIZE + gl.expand_dims(cols, 0)).to(gl.float32)
            active_rows = rows < EXPERT_CAPACITY
            vals = gl.where(
                gl.expand_dims(active_rows, 1),
                vals,
                gl.full((block_m, block_n), -777.0, gl.float32, layout=blocked_layout),
            )
            smem.store(vals)
            result_desc = tdm.make_tensor_descriptor(
                base=result_values + send_idx * EXPERT_CAPACITY * HIDDEN_SIZE,
                shape=(EXPERT_CAPACITY, HIDDEN_SIZE),
                strides=(HIDDEN_SIZE, 1),
                block_shape=(block_m, block_n),
                layout=shared_layout,
            )
            tdm.async_store(result_desc, [0, 0], smem)
            tdm.async_wait(0)
            gl.store(result_counts + send_idx, count)
            gl.atomic_xchg(queue_flag, 1, sem="release", scope="gpu")
        else:
            ready = gl.full((), 0, gl.int32)
            ready_iters = gl.full((), 0, gl.int32)
            while (ready == 0) & (ready_iters < MAX_POLL_ITERS):
                ready = gl.atomic_add(queue_flag, 0, sem="acquire", scope="gpu")
                ready_iters += 1
            local_spill = gl.load(
                result_values + send_idx * EXPERT_CAPACITY * HIDDEN_SIZE + (block_m - 1) * HIDDEN_SIZE
            ).to(gl.int64)
            rocshmem.putmem_wave(result_counts + send_idx, result_counts + send_idx, 4, peer)
            rocshmem.fence()
            rocshmem.putmem_signal_wave(
                result_values + send_idx * EXPERT_CAPACITY * HIDDEN_SIZE,
                result_values + send_idx * EXPERT_CAPACITY * HIDDEN_SIZE,
                EXPERT_CAPACITY * HIDDEN_SIZE * 4,
                result_signals + send_idx,
                payload,
                rocshmem.ROCSHMEM_SIGNAL_SET,
                peer,
            )

            seen = gl.full((), 0, gl.int32)
            iters = gl.full((), 0, gl.int32)
            signal = gl.full((), 0, gl.uint64)
            while (seen == 0) & (iters < MAX_POLL_ITERS):
                signal = rocshmem.signal_fetch_wave(result_signals + recv_idx)
                if signal >= payload:
                    seen = 1
                iters += 1
            recv_count = gl.load(result_counts + recv_idx)
            first_value = gl.load(result_values + recv_idx * EXPERT_CAPACITY * HIDDEN_SIZE).to(gl.int64)
            last_value = gl.load(
                result_values + recv_idx * EXPERT_CAPACITY * HIDDEN_SIZE + (EXPERT_CAPACITY * HIDDEN_SIZE - 1)
            ).to(gl.int64)
            gl.store(status + 0, my_pe.to(gl.int64))
            gl.store(status + 1, seen.to(gl.int64))
            gl.store(status + 2, iters.to(gl.int64))
            gl.store(status + 3, _signal_count(signal, epoch).to(gl.int64))
            gl.store(status + 4, recv_count.to(gl.int64))
            gl.store(status + 5, first_value)
            gl.store(status + 6, last_value)
            gl.store(status + 7, local_spill)
            gl.store(status + 8, ready_iters.to(gl.int64))
    elif MODE == "signal":
        send_idx = (my_pe * WORLD_SIZE + peer) * LOCAL_EXPERTS
        recv_idx = (peer * WORLD_SIZE + my_pe) * LOCAL_EXPERTS
        remote_signal = rocshmem.remote_ptr(result_signals + send_idx, peer)
        gl.atomic_xchg(remote_signal, payload, sem="release", scope="sys")

        seen = gl.full((), 0, gl.int32)
        iters = gl.full((), 0, gl.int32)
        signal = gl.full((), 0, gl.uint64)
        while (seen == 0) & (iters < MAX_POLL_ITERS):
            signal = rocshmem.signal_fetch_wave(result_signals + recv_idx)
            if signal >= payload:
                seen = 1
            iters += 1
        gl.store(status + 0, my_pe.to(gl.int64))
        gl.store(status + 1, seen.to(gl.int64))
        gl.store(status + 2, iters.to(gl.int64))
        gl.store(status + 3, _signal_count(signal, epoch).to(gl.int64))


def _run_kernel(runtime: RocshmemRuntime, rank: int, mode: str, tensors: tuple[torch.Tensor, ...]) -> None:
    rocshmem.register_kernel(_cross_rank_exchange_kernel)
    rocshmem.install_module_init(runtime)

    grid = (2,) if mode == "tdm-queue-result" else (1,)
    for preload_rank in range(WORLD):
        if rank == preload_rank:
            kernel = _cross_rank_exchange_kernel.warmup(
                *tensors,
                MODE=mode,
                MAX_POLL_ITERS=MAX_ITERS,
                WORLD_SIZE=WORLD,
                LOCAL_EXPERTS=NLX,
                EXPERT_CAPACITY=EC,
                HIDDEN_SIZE=H,
                EPOCH_VALUE=EPOCH,
                num_warps=4,
                extern_libs=rocshmem.extern_libs("gfx1250"),
                grid=grid,
            )
            if kernel is not None:
                _ = kernel.run
                if not getattr(kernel, "_flashmoe_rocshmem_module_initialized", False):
                    runtime.hipmodule_init(int(kernel.module))
                    setattr(kernel, "_flashmoe_rocshmem_module_initialized", True)
        runtime.barrier_all()

    runtime.barrier_all()
    _cross_rank_exchange_kernel[grid](
        *tensors,
        MODE=mode,
        MAX_POLL_ITERS=MAX_ITERS,
        WORLD_SIZE=WORLD,
        LOCAL_EXPERTS=NLX,
        EXPERT_CAPACITY=EC,
        HIDDEN_SIZE=H,
        EPOCH_VALUE=EPOCH,
        num_warps=4,
        extern_libs=rocshmem.extern_libs("gfx1250"),
    )
    torch.cuda.synchronize()


def _worker() -> None:
    rank = int(os.environ["FLASHMOE_RANK"])
    uid = bytes.fromhex(os.environ["FLASHMOE_ROCSHMEM_UID"])
    mode = os.environ["FLASHMOE_CROSS_RANK_MODE"]
    channels = WORLD * WORLD * NLX

    def log(*items: object) -> None:
        print("rank", rank, *items, flush=True)

    try:
        torch.cuda.set_device(0)
        runtime = RocshmemRuntime()
        runtime.set_device(0)
        runtime.init_uniqueid(rank, WORLD, uid)
        log("init", runtime.my_pe(), runtime.n_pes())
        log("alloc", "dispatch_tokens")
        dispatch_tokens = runtime.calloc(channels * EC * H, 2, torch.float16)
        log("alloc", "dispatch_token_ids")
        dispatch_token_ids = runtime.calloc(channels * EC, 4, torch.int32)
        log("alloc", "dispatch_probs")
        dispatch_probs = runtime.calloc(channels * EC, 4, torch.float32)
        log("alloc", "dispatch_counts")
        dispatch_counts = runtime.calloc(channels, 4, torch.int32)
        log("alloc", "dispatch_signals")
        dispatch_signals = runtime.calloc(channels, 8, torch.uint64)
        log("alloc", "result_values")
        result_storage_ec = 16 if mode in ("tdm-result", "tdm-queue-result") else EC
        result_values = runtime.calloc(channels * result_storage_ec * H, 4, torch.float32)
        log("alloc", "result_counts")
        result_counts = runtime.calloc(channels, 4, torch.int32)
        log("alloc", "result_signals")
        result_signals = runtime.calloc(channels, 8, torch.uint64)
        status = torch.zeros((16,), device="cuda", dtype=torch.int64)
        tensors = (
            dispatch_tokens,
            dispatch_token_ids,
            dispatch_probs,
            dispatch_counts,
            dispatch_signals,
            result_values,
            result_counts,
            result_signals,
            status,
        )
        log("barrier", "before-launch")
        runtime.barrier_all()
        log("launch", mode)
        _run_kernel(runtime, rank, mode, tensors)
        got = status.cpu().tolist()
        log("status", got)
        peer = 1 - rank
        if mode == "dispatch":
            ok = (
                got[1] == 1
                and got[3] == EC
                and got[4] == EC
                and got[5] == peer * 100
                and got[6] == peer * 100 + 1
                and got[7] == peer * 1000
                and got[8] == peer * 1000 + EC * H - 1
            )
        elif mode == "result":
            ok = got[1] == 1 and got[3] == EC and got[4] == EC and got[5] == peer * 1000 and got[6] == peer * 1000 + EC * H - 1
        elif mode == "tdm-result":
            ok = (
                got[1] == 1
                and got[3] == EC
                and got[4] == EC
                and got[5] == peer * 1000
                and got[6] == peer * 1000 + EC * H - 1
                and got[7] == 0
            )
        elif mode == "tdm-queue-result":
            ok = (
                got[1] == 1
                and got[3] == EC
                and got[4] == EC
                and got[5] == peer * 1000
                and got[6] == peer * 1000 + EC * H - 1
                and got[7] == 0
                and got[8] < MAX_ITERS
            )
        else:
            ok = got[1] == 1 and got[3] == EC
        runtime.free(dispatch_tokens)
        runtime.free(dispatch_token_ids)
        runtime.free(dispatch_probs)
        runtime.free(dispatch_counts)
        runtime.free(dispatch_signals)
        runtime.free(result_values)
        runtime.free(result_counts)
        runtime.free(result_signals)
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
        env["FLASHMOE_CROSS_RANK_MODE"] = mode
        procs.append(
            subprocess.Popen(
                [sys.executable, "-u", __file__, "--worker"],
                cwd=str(Path(__file__).resolve().parents[1]),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
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
                proc.terminate()
        time.sleep(1)
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
    for proc in procs:
        proc.wait(timeout=2)
    exits = [(proc.pid, proc.returncode) for proc in procs]
    print("case", mode, "exits", exits, flush=True)
    if alive or any(proc.returncode for proc in procs):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("dispatch", "result", "signal", "tdm-result", "tdm-queue-result"),
        default="dispatch",
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
