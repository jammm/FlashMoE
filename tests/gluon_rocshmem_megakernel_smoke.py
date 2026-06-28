from __future__ import annotations

import argparse
import ctypes
import faulthandler
import os
import select
import signal as signal_module
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flashmoe_gluon as fmg
from flashmoe_gluon.rocshmem_runtime import RocshmemMegakernelContext, RocshmemRuntime, create_uniqueid


WORLD = int(os.environ.get("FLASHMOE_WORLD", "2"))
S = 1
H = 64
I = 64
LOCAL_EXPERTS = 1
EXPERT_CAPACITY = 2


def _allow_ptrace_attach() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    pr_set_ptracer = 0x59616D61
    pr_set_ptracer_any = ctypes.c_ulong(-1 & ((1 << (ctypes.sizeof(ctypes.c_ulong) * 8)) - 1))
    ret = libc.prctl(pr_set_ptracer, pr_set_ptracer_any, 0, 0, 0)
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def _reference(
    tokens: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    bias_up: torch.Tensor,
    bias_down: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    logits = tokens.float() @ gate.float()
    vals, idxs = torch.topk(logits, top_k, dim=1)
    probs = torch.ones_like(vals) if top_k == 1 else torch.softmax(vals, dim=1)
    out = torch.zeros((tokens.shape[0], tokens.shape[1]), device=tokens.device, dtype=torch.float32)
    for token_id in range(tokens.shape[0]):
        for route_id in range(top_k):
            expert_id = int(idxs[token_id, route_id])
            hidden = bias_up[expert_id].float() + tokens[token_id].float() @ up[expert_id].float()
            result = bias_down[expert_id].float() + hidden @ down[expert_id].float()
            out[token_id] += probs[token_id, route_id].float() * result
    return out


def _worker() -> None:
    rank = int(os.environ["FLASHMOE_RANK"])
    uid = bytes.fromhex(os.environ["FLASHMOE_ROCSHMEM_UID"])
    top_k = int(os.environ["FLASHMOE_TOP_K"])
    route_mode = os.environ["FLASHMOE_ROUTE_MODE"]
    repeats = int(os.environ.get("FLASHMOE_REPEATS", "1"))
    num_programs = int(os.environ.get("FLASHMOE_NUM_PROGRAMS", "4"))
    rtol = float(os.environ.get("FLASHMOE_RTOL", "8e-2"))
    atol = float(os.environ.get("FLASHMOE_ATOL", "8e-3"))
    dump_after = float(os.environ.get("FLASHMOE_DUMP_AFTER", "0"))
    defer_reference = os.environ.get("FLASHMOE_DEFER_REFERENCE", "0") == "1"
    if dump_after > 0:
        faulthandler.dump_traceback_later(dump_after, repeat=True)

    def log(*items: object) -> None:
        print("rank", rank, *items, flush=True)

    if os.environ.get("FLASHMOE_ALLOW_PTRACE", "0") == "1":
        _allow_ptrace_attach()

    try:
        log("worker-start")
        torch.cuda.set_device(0)
        log("device-set")
        runtime = RocshmemRuntime()
        log("runtime-created")
        runtime.set_device(0)
        log("runtime-device-set")
        runtime.init_uniqueid(rank, WORLD, uid)
        log("init", runtime.my_pe(), runtime.n_pes())
        ctx = RocshmemMegakernelContext.create(
            runtime,
            world_size=WORLD,
            rank=rank,
            local_experts=LOCAL_EXPERTS,
            expert_capacity=EXPERT_CAPACITY,
            hidden_size=H,
            dtype=torch.float16,
        )
        log("ctx-created")

        torch.manual_seed(123)
        tokens = torch.ones((S, H), device="cuda", dtype=torch.float16)
        gate = torch.full((H, WORLD * LOCAL_EXPERTS), -10.0, device="cuda", dtype=torch.float16)
        if route_mode == "local":
            gate[:, rank] = 10.0
        elif route_mode == "remote":
            if WORLD < 2:
                raise ValueError("remote route mode requires WORLD >= 2")
            gate[:, 1 - rank] = 10.0
        elif route_mode == "remote01":
            if WORLD < 2:
                raise ValueError("remote01 route mode requires WORLD >= 2")
            gate[:, 1 if rank == 0 else rank] = 10.0
        elif route_mode == "remote10":
            if WORLD < 2:
                raise ValueError("remote10 route mode requires WORLD >= 2")
            gate[:, 0 if rank == 1 else rank] = 10.0
        elif route_mode == "mixed":
            if WORLD < 2:
                raise ValueError("mixed route mode requires WORLD >= 2")
            gate[:, 1 - rank] = 10.0
            gate[:, rank] = 9.0
        else:
            raise ValueError(f"unknown route mode: {route_mode}")

        up = torch.randn((WORLD * LOCAL_EXPERTS, H, I), device="cuda", dtype=torch.float16)
        down = torch.randn((WORLD * LOCAL_EXPERTS, I, H), device="cuda", dtype=torch.float16)
        bias_up = torch.randn((WORLD * LOCAL_EXPERTS, I), device="cuda", dtype=torch.float16)
        bias_down = torch.randn((WORLD * LOCAL_EXPERTS, H), device="cuda", dtype=torch.float16)

        local_slice = slice(rank * LOCAL_EXPERTS, (rank + 1) * LOCAL_EXPERTS)
        expected = None if defer_reference else _reference(tokens, gate, up, down, bias_up, bias_down, top_k).cpu()
        ok = True
        for repeat_id in range(repeats):
            log("launch", route_mode, f"top_k={top_k}", f"repeat={repeat_id}")
            out = fmg.forward_megakernel_rocshmem(
                tokens,
                gate,
                up[local_slice].contiguous(),
                down[local_slice].contiguous(),
                bias_up[local_slice].contiguous(),
                bias_down[local_slice].contiguous(),
                ctx=ctx,
                top_k=top_k,
                activation=fmg.ACT_IDENTITY,
                num_programs=num_programs,
                num_warps=4,
            )
            log("launched", route_mode, f"top_k={top_k}", f"repeat={repeat_id}")
            debug_state = getattr(fmg.forward_megakernel_rocshmem, "last_debug_state", None)
            poll_s = float(os.environ.get("FLASHMOE_DEBUG_POLL_S", "0"))
            if debug_state is not None and poll_s > 0:
                deadline = time.time() + poll_s
                last_debug = None
                while time.time() < deadline:
                    current_debug = debug_state.tolist()
                    if current_debug != last_debug:
                        log("debug-poll", current_debug)
                        last_debug = current_debug
                    time.sleep(0.05)
            torch.cuda.synchronize()
            log("synced", route_mode, f"top_k={top_k}", f"repeat={repeat_id}")
            if debug_state is not None:
                log("debug", debug_state.cpu().tolist())
            got = out.cpu()
            if expected is None:
                expected = _reference(tokens, gate, up, down, bias_up, bias_down, top_k).cpu()
            max_abs = float((got - expected).abs().max())
            close = bool(torch.allclose(got, expected, rtol=rtol, atol=atol))
            finite = bool(torch.isfinite(got).all())
            ok = ok and finite and close
            log(
                "compare",
                route_mode,
                f"top_k={top_k}",
                f"repeat={repeat_id}",
                "max_abs",
                max_abs,
                "close",
                close,
            )
        ctx.close()
        runtime.barrier_all()
        runtime.finalize()
        if not ok:
            raise SystemExit(2)
    except BaseException as exc:
        log("ERR", repr(exc))
        traceback.print_exc()
        raise


def _run_case(route_mode: str, top_k: int, timeout_s: float, repeats: int) -> None:
    uid = create_uniqueid().hex()
    procs: list[subprocess.Popen[str]] = []
    for rank in range(WORLD):
        env = os.environ.copy()
        env["FLASHMOE_RANK"] = str(rank)
        env["FLASHMOE_ROCSHMEM_UID"] = uid
        env["FLASHMOE_ROUTE_MODE"] = route_mode
        env["FLASHMOE_TOP_K"] = str(top_k)
        env["FLASHMOE_REPEATS"] = str(repeats)
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
        print("timeout", route_mode, f"top_k={top_k}", "alive", alive, flush=True)
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
    print("case", route_mode, f"top_k={top_k}", "exits", exits, flush=True)
    if alive or any(proc.returncode for proc in procs):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument("--case", choices=("all", "local", "remote", "remote01", "remote10", "mixed"), default="all")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()

    if args.worker:
        _worker()
        return

    cases = []
    if args.case in ("all", "local"):
        cases.append(("local", 1))
    if args.case in ("all", "remote"):
        cases.append(("remote", 1))
    if args.case == "remote01":
        cases.append(("remote01", 1))
    if args.case == "remote10":
        cases.append(("remote10", 1))
    if args.case in ("all", "mixed"):
        cases.append(("mixed", 2))
    for route_mode, top_k in cases:
        _run_case(route_mode, top_k, args.timeout_s, args.repeats)


if __name__ == "__main__":
    main()
