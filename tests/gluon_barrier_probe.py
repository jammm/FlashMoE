from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@gluon.jit
def _scalar_atomic_count(counter):
    gl.atomic_add(counter, 1, sem="relaxed", scope="gpu")


@gluon.jit
def _spin_barrier_once(counter, observed, NUM_PROGRAMS: gl.constexpr):
    pid = gl.program_id(0)
    old = gl.atomic_add(counter, 1, sem="release", scope="gpu")
    while gl.atomic_cas(
        counter,
        NUM_PROGRAMS,
        NUM_PROGRAMS,
        sem="acquire",
        scope="gpu",
    ) < NUM_PROGRAMS:
        pass
    gl.store(observed + pid, old + 1)


@gluon.jit
def _multi_phase_barrier(barriers, init_flag, observed, EPOCH: gl.constexpr, NUM_PROGRAMS: gl.constexpr):
    pid = gl.program_id(0)
    if pid == 0:
        for phase in gl.static_range(0, 4):
            gl.store(barriers + phase, 0)
        gl.atomic_xchg(init_flag, EPOCH, sem="release", scope="gpu")

    while gl.atomic_cas(
        init_flag,
        EPOCH,
        EPOCH,
        sem="acquire",
        scope="gpu",
    ) != EPOCH:
        pass

    total = pid * 0
    for phase in gl.static_range(0, 4):
        old = gl.atomic_add(barriers + phase, 1, sem="release", scope="gpu")
        while gl.atomic_cas(
            barriers + phase,
            NUM_PROGRAMS,
            NUM_PROGRAMS,
            sem="acquire",
            scope="gpu",
        ) < NUM_PROGRAMS:
            pass
        total += old + 1
    gl.store(observed + pid, total)


@gluon.jit
def _clustered_multi_phase_barrier(barriers, init_flag, observed, EPOCH: gl.constexpr, NUM_PROGRAMS: gl.constexpr):
    pid = gl.program_id(0)
    if pid == 0:
        for phase in gl.static_range(0, 4):
            gl.store(barriers + phase, 0)
        gl.atomic_xchg(init_flag, EPOCH, sem="release", scope="gpu")

    while gl.atomic_cas(
        init_flag,
        EPOCH,
        EPOCH,
        sem="acquire",
        scope="gpu",
    ) != EPOCH:
        pass

    total = pid * 0
    for phase in gl.static_range(0, 4):
        gl.amd.gfx1250.cluster.arrive()
        gl.amd.gfx1250.cluster.wait()
        old = gl.atomic_add(barriers + phase, 1, sem="release", scope="gpu")
        while gl.atomic_cas(
            barriers + phase,
            NUM_PROGRAMS,
            NUM_PROGRAMS,
            sem="acquire",
            scope="gpu",
        ) < NUM_PROGRAMS:
            pass
        gl.amd.gfx1250.cluster.arrive()
        gl.amd.gfx1250.cluster.wait()
        total += old + 1
    gl.store(observed + pid, total)


@gluon.jit
def _cluster_arrive_wait(observed):
    pid = gl.program_id(0)
    gl.amd.gfx1250.cluster.arrive()
    gl.amd.gfx1250.cluster.wait()
    gl.store(observed + pid, pid + 1)


def _run_count(grid: int, num_warps: int, num_ctas: int) -> int:
    counter = torch.zeros((1,), device="cuda", dtype=torch.int32)
    _scalar_atomic_count[(grid,)](
        counter,
        num_warps=num_warps,
        num_ctas=num_ctas,
    )
    torch.cuda.synchronize()
    return int(counter.cpu()[0])


def _run_barrier(grid: int, num_warps: int) -> tuple[int, list[int]]:
    counter = torch.zeros((1,), device="cuda", dtype=torch.int32)
    observed = torch.empty((grid,), device="cuda", dtype=torch.int32)
    _spin_barrier_once[(grid,)](
        counter,
        observed,
        NUM_PROGRAMS=grid,
        num_warps=num_warps,
    )
    torch.cuda.synchronize()
    return int(counter.cpu()[0]), [int(x) for x in observed.cpu()]


def _run_multi_phase(grid: int, num_warps: int) -> tuple[list[int], list[int], int]:
    barriers = torch.empty((4,), device="cuda", dtype=torch.int32)
    init_flag = torch.empty((1,), device="cuda", dtype=torch.int32)
    observed = torch.empty((grid,), device="cuda", dtype=torch.int32)
    _multi_phase_barrier[(grid,)](
        barriers,
        init_flag,
        observed,
        EPOCH=17,
        NUM_PROGRAMS=grid,
        num_warps=num_warps,
    )
    torch.cuda.synchronize()
    return [int(x) for x in barriers.cpu()], [int(x) for x in observed.cpu()], int(init_flag.cpu()[0])


def _run_clustered_multi_phase(grid: int, num_warps: int, num_ctas: int) -> tuple[list[int], list[int], int]:
    barriers = torch.empty((4,), device="cuda", dtype=torch.int32)
    init_flag = torch.empty((1,), device="cuda", dtype=torch.int32)
    observed = torch.empty((grid,), device="cuda", dtype=torch.int32)
    _clustered_multi_phase_barrier[(grid,)](
        barriers,
        init_flag,
        observed,
        EPOCH=23,
        NUM_PROGRAMS=grid,
        num_warps=num_warps,
        num_ctas=num_ctas,
    )
    torch.cuda.synchronize()
    return [int(x) for x in barriers.cpu()], [int(x) for x in observed.cpu()], int(init_flag.cpu()[0])


def _run_cluster(grid: int, num_warps: int, num_ctas: int) -> list[int]:
    observed = torch.empty((grid,), device="cuda", dtype=torch.int32)
    _cluster_arrive_wait[(grid,)](
        observed,
        num_warps=num_warps,
        num_ctas=num_ctas,
    )
    torch.cuda.synchronize()
    return [int(x) for x in observed.cpu()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--multi-phase", action="store_true")
    parser.add_argument("--clustered-multi-phase", action="store_true")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--num-ctas", type=int, default=2)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    count = _run_count(args.grid, args.num_warps, args.num_ctas)
    print(
        "scalar_atomic_count",
        "grid",
        args.grid,
        "num_warps",
        args.num_warps,
        "num_ctas",
        args.num_ctas,
        "count",
        count,
    )
    if not args.count_only:
        final, observed = _run_barrier(args.grid, args.num_warps)
        print("spin_barrier_once", "grid", args.grid, "final", final, "observed", observed)
    if args.multi_phase:
        barriers, observed, init_flag = _run_multi_phase(args.grid, args.num_warps)
        print(
            "multi_phase_barrier",
            "grid",
            args.grid,
            "barriers",
            barriers,
            "observed",
            observed,
            "init_flag",
            init_flag,
        )
    if args.clustered_multi_phase:
        barriers, observed, init_flag = _run_clustered_multi_phase(args.grid, args.num_warps, args.num_ctas)
        print(
            "clustered_multi_phase_barrier",
            "grid",
            args.grid,
            "num_ctas",
            args.num_ctas,
            "barriers",
            barriers,
            "observed",
            observed,
            "init_flag",
            init_flag,
        )
    if args.cluster:
        observed = _run_cluster(args.grid, args.num_warps, args.num_ctas)
        print("cluster_arrive_wait", "grid", args.grid, "num_ctas", args.num_ctas, "observed", observed)


if __name__ == "__main__":
    main()
