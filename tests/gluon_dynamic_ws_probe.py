from __future__ import annotations

import argparse

import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.gfx1250 import mbarrier

from flashmoe_gluon.megakernel import (
    _enqueue_task,
    _gemm0_ws_compute,
    _gemm0_ws_epilogue,
    _gemm0_ws_producer,
    _gemm1_ws_compute,
    _gemm1_ws_epilogue,
    _gemm1_ws_producer,
    _global_barrier,
    _init_dynamic_task_queue,
    _init_ws_mbarriers,
    _processor_next_task_queue,
    _scheduler_reset_queue,
    _scheduler_run_queue,
    _wait_storecnt0,
    _wmma_layout,
    _wmma_shared_a_layout,
    _wmma_shared_b_layout,
)
from tests.triton_moe_smoke import _make_inputs


@gluon.jit
def _dynamic_ws_probe_kernel(
    tokens,
    expert_up,
    bias_up,
    expert_down,
    bias_down,
    output,
    hidden,
    expert_counts,
    route_tokens,
    route_probs,
    task_queue,
    task_head,
    task_tail,
    tile_sync,
    processor_ready,
    processor_mailboxes,
    processor_seen,
    barriers,
    debug_state,
    S: gl.constexpr,
    H: gl.constexpr,
    I: gl.constexpr,
    E: gl.constexpr,
    EC: gl.constexpr,
    TOP_K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    pid = gl.program_id(0)
    gl.store(debug_state + pid, 10)
    h_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], [])
    offs_h = gl.arange(0, H, layout=h_layout)
    if pid == 0:
        for phase in gl.static_range(0, 4):
            gl.store(barriers + phase, 0)
    out_base = pid * H
    while out_base < S * H:
        gl.store(
            output + out_base + offs_h,
            gl.full((H,), 0.0, gl.float32, layout=h_layout),
            mask=out_base + offs_h < S * H,
        )
        out_base += NUM_PROGRAMS * H

    _scheduler_reset_queue(processor_ready, processor_mailboxes, processor_seen, NUM_PROGRAMS, 1)

    PRODUCER_WARPS: gl.constexpr = 4
    COMPUTE_WARPS: gl.constexpr = 4
    EPILOGUE_WARPS: gl.constexpr = NUM_WARPS
    NUM_BUFFERS: gl.constexpr = 2
    NUM_ACC_BUFFERS: gl.constexpr = 1

    wmma_layout: gl.constexpr = _wmma_layout(COMPUTE_WARPS)
    shared_a_layout: gl.constexpr = _wmma_shared_a_layout(BLOCK_M, BLOCK_N)
    shared_b_layout: gl.constexpr = _wmma_shared_b_layout(BLOCK_N, BLOCK_N)
    shared_h_layout: gl.constexpr = _wmma_shared_a_layout(BLOCK_M, BLOCK_N)
    shared_down_layout: gl.constexpr = _wmma_shared_b_layout(BLOCK_N, BLOCK_N)
    shared_acc_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0], [])
    route_blocks: gl.constexpr = (EC + BLOCK_M - 1) // BLOCK_M
    h_tiles: gl.constexpr = H // BLOCK_N
    i_tiles: gl.constexpr = I // BLOCK_N
    gemm0_tasks: gl.constexpr = E * route_blocks * i_tiles
    gemm1_tasks: gl.constexpr = E * route_blocks * h_tiles
    total_tasks: gl.constexpr = gemm0_tasks + gemm1_tasks
    sync_slots: gl.constexpr = E * route_blocks

    _init_dynamic_task_queue(task_queue, task_head, task_tail, tile_sync, gemm0_tasks, total_tasks, sync_slots)
    _global_barrier(barriers, 0, NUM_PROGRAMS)
    gl.store(debug_state + pid, 100)

    if pid == 0:
        _scheduler_run_queue(
            processor_ready,
            processor_mailboxes,
            task_queue,
            task_tail,
            debug_state,
            False,
            False,
            total_tasks,
            NUM_PROGRAMS,
        )
        gl.store(debug_state + pid, 900)
    else:
        gl.store(debug_state + pid, 300)
        task_id = _processor_next_task_queue(
            processor_ready,
            processor_mailboxes,
            processor_seen,
            debug_state,
            False,
            False,
        )
        while task_id < total_tasks:
            gl.store(debug_state + pid, 400)
            gl.store(debug_state + 16 + pid, task_id)
            if task_id < gemm0_tasks:
                gl.store(debug_state + pid, 410)
                expert_idx = task_id // (route_blocks * i_tiles)
                rem_task = task_id - expert_idx * route_blocks * i_tiles
                route_block = rem_task // i_tiles
                i_tile = rem_task - route_block * i_tiles
                base_slot = route_block * BLOCK_M
                base_i = i_tile * BLOCK_N
                x_buffer = gl.allocate_shared_memory(
                    tokens.dtype.element_ty,
                    shape=[NUM_BUFFERS, BLOCK_M, BLOCK_N],
                    layout=shared_a_layout,
                )
                w_buffer = gl.allocate_shared_memory(
                    expert_up.dtype.element_ty,
                    shape=[NUM_BUFFERS, BLOCK_N, BLOCK_N],
                    layout=shared_b_layout,
                )
                hidden_buffer = gl.allocate_shared_memory(
                    hidden.dtype.element_ty,
                    shape=[NUM_ACC_BUFFERS, BLOCK_M, BLOCK_N],
                    layout=shared_h_layout,
                )
                load_empty_bars = gl.allocate_shared_memory(gl.int64, [NUM_BUFFERS, 1], mbarrier.MBarrierLayout())
                load_ready_bars = gl.allocate_shared_memory(gl.int64, [NUM_BUFFERS, 1], mbarrier.MBarrierLayout())
                acc_empty_bars = gl.allocate_shared_memory(gl.int64, [NUM_ACC_BUFFERS, 1], mbarrier.MBarrierLayout())
                acc_ready_bars = gl.allocate_shared_memory(gl.int64, [NUM_ACC_BUFFERS, 1], mbarrier.MBarrierLayout())
                _init_ws_mbarriers(
                    load_empty_bars,
                    load_ready_bars,
                    acc_empty_bars,
                    acc_ready_bars,
                    NUM_BUFFERS,
                    NUM_ACC_BUFFERS,
                    PRODUCER_WARPS,
                    COMPUTE_WARPS,
                    EPILOGUE_WARPS,
                    True,
                    1,
                )
                gl.warp_specialize(
                    [
                        (
                            _gemm0_ws_epilogue,
                            (
                                hidden,
                                hidden_buffer,
                                acc_empty_bars,
                                acc_ready_bars,
                                expert_idx,
                                base_slot,
                                base_i,
                                EC,
                                I,
                                BLOCK_M,
                                BLOCK_N,
                                shared_h_layout,
                            ),
                        ),
                        (
                            _gemm0_ws_compute,
                            (
                                bias_up,
                                bias_up,
                                x_buffer,
                                w_buffer,
                                x_buffer,
                                w_buffer,
                                hidden_buffer,
                                load_empty_bars,
                                load_ready_bars,
                                load_empty_bars,
                                load_ready_bars,
                                acc_empty_bars,
                                acc_ready_bars,
                                expert_idx,
                                base_i,
                                H,
                                I,
                                BLOCK_M,
                                BLOCK_N,
                                NUM_BUFFERS,
                                0,
                                False,
                                wmma_layout,
                            ),
                        ),
                        (
                            _gemm0_ws_producer,
                            (
                                tokens,
                                expert_up,
                                expert_up,
                                route_tokens,
                                expert_counts,
                                x_buffer,
                                w_buffer,
                                x_buffer,
                                w_buffer,
                                load_empty_bars,
                                load_ready_bars,
                                load_empty_bars,
                                load_ready_bars,
                                expert_idx,
                                base_slot,
                                base_i,
                                S,
                                H,
                                I,
                                EC,
                                BLOCK_M,
                                BLOCK_N,
                                NUM_BUFFERS,
                                PRODUCER_WARPS,
                                False,
                                shared_a_layout,
                                shared_b_layout,
                            ),
                        ),
                    ],
                    [COMPUTE_WARPS, PRODUCER_WARPS],
                )
                sync_idx = expert_idx * route_blocks + route_block
                done_tiles = gl.atomic_add(tile_sync + sync_idx, 1, sem="release", scope="gpu") + 1
                if done_tiles == i_tiles:
                    for h_enqueue in gl.static_range(0, h_tiles):
                        downstream_task = (
                            gemm0_tasks + expert_idx * route_blocks * h_tiles + route_block * h_tiles + h_enqueue
                        )
                        _enqueue_task(task_queue, task_tail, downstream_task)
                gl.store(debug_state + pid, 420)
            else:
                gl.store(debug_state + pid, 510)
                gemm1_id = task_id - gemm0_tasks
                expert_idx = gemm1_id // (route_blocks * h_tiles)
                rem_task = gemm1_id - expert_idx * route_blocks * h_tiles
                route_block = rem_task // h_tiles
                h_tile = rem_task - route_block * h_tiles
                base_slot = route_block * BLOCK_M
                base_h = h_tile * BLOCK_N
                h_buffer = gl.allocate_shared_memory(
                    hidden.dtype.element_ty,
                    shape=[NUM_BUFFERS, BLOCK_M, BLOCK_N],
                    layout=shared_h_layout,
                )
                down_buffer = gl.allocate_shared_memory(
                    expert_down.dtype.element_ty,
                    shape=[NUM_BUFFERS, BLOCK_N, BLOCK_N],
                    layout=shared_down_layout,
                )
                acc_buffer = gl.allocate_shared_memory(
                    gl.float32,
                    shape=[NUM_ACC_BUFFERS, BLOCK_M, BLOCK_N],
                    layout=shared_acc_layout,
                )
                load_empty_bars = gl.allocate_shared_memory(gl.int64, [NUM_BUFFERS, 1], mbarrier.MBarrierLayout())
                load_ready_bars = gl.allocate_shared_memory(gl.int64, [NUM_BUFFERS, 1], mbarrier.MBarrierLayout())
                acc_empty_bars = gl.allocate_shared_memory(gl.int64, [NUM_ACC_BUFFERS, 1], mbarrier.MBarrierLayout())
                acc_ready_bars = gl.allocate_shared_memory(gl.int64, [NUM_ACC_BUFFERS, 1], mbarrier.MBarrierLayout())
                _init_ws_mbarriers(
                    load_empty_bars,
                    load_ready_bars,
                    acc_empty_bars,
                    acc_ready_bars,
                    NUM_BUFFERS,
                    NUM_ACC_BUFFERS,
                    PRODUCER_WARPS,
                    COMPUTE_WARPS,
                    EPILOGUE_WARPS,
                    False,
                    1,
                )
                gl.warp_specialize(
                    [
                        (
                            _gemm1_ws_epilogue,
                            (
                                route_tokens,
                                route_probs,
                                expert_counts,
                                bias_down,
                                output,
                                acc_buffer,
                                acc_empty_bars,
                                acc_ready_bars,
                                expert_idx,
                                base_slot,
                                base_h,
                                EC,
                                H,
                                TOP_K,
                                BLOCK_M,
                                BLOCK_N,
                                EPILOGUE_WARPS,
                                wmma_layout,
                            ),
                        ),
                        (
                            _gemm1_ws_compute,
                            (
                                h_buffer,
                                down_buffer,
                                acc_buffer,
                                load_empty_bars,
                                load_ready_bars,
                                acc_empty_bars,
                                acc_ready_bars,
                                I,
                                BLOCK_M,
                                BLOCK_N,
                                NUM_BUFFERS,
                                wmma_layout,
                            ),
                        ),
                        (
                            _gemm1_ws_producer,
                            (
                                hidden,
                                expert_down,
                                h_buffer,
                                down_buffer,
                                load_empty_bars,
                                load_ready_bars,
                                expert_idx,
                                base_slot,
                                base_h,
                                EC,
                                I,
                                H,
                                BLOCK_M,
                                BLOCK_N,
                                NUM_BUFFERS,
                                shared_h_layout,
                                shared_down_layout,
                            ),
                        ),
                    ],
                    [COMPUTE_WARPS, PRODUCER_WARPS],
                )
                _wait_storecnt0()
                gl.store(debug_state + pid, 520)
            task_id = _processor_next_task_queue(
                processor_ready,
                processor_mailboxes,
                processor_seen,
                debug_state,
                False,
                False,
            )

        gl.store(debug_state + pid, 800)
    _global_barrier(barriers, 1, NUM_PROGRAMS)
    gl.store(debug_state + pid, 999)


def _reference(
    tokens: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    bias_up: torch.Tensor,
    bias_down: torch.Tensor,
    top_k: int,
    ec: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.softmax(tokens.float() @ gate.float(), dim=1)
    vals, idxs = torch.topk(probs, top_k, dim=1)
    vals = vals / vals.sum(dim=1, keepdim=True)
    e = gate.shape[1]
    routes: list[list[tuple[int, float]]] = [[] for _ in range(e)]
    for token_id in range(tokens.shape[0]):
        for route_id in range(top_k):
            routes[int(idxs[token_id, route_id])].append((token_id, float(vals[token_id, route_id])))

    counts = torch.tensor([len(r) for r in routes], dtype=torch.int32)
    route_tokens = torch.zeros((e, ec), dtype=torch.int32)
    route_probs = torch.zeros((e, ec), dtype=torch.float32)
    out = torch.zeros((tokens.shape[0], tokens.shape[1]), dtype=torch.float32)
    for expert_id, expert_routes in enumerate(routes):
        for slot, (token_id, prob) in enumerate(expert_routes[:ec]):
            route_tokens[expert_id, slot] = token_id
            route_probs[expert_id, slot] = prob
            hidden = tokens[token_id].float() @ up[expert_id].float() + bias_up[expert_id].float()
            out[token_id] += (hidden @ down[expert_id].float() + bias_down[expert_id].float()) * prob
    return out, counts, route_tokens, route_probs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-programs", type=int, default=3)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    s = 16
    h = 64
    i = 64
    e = 4
    ec = 16
    top_k = 2
    tokens, gate, up, down, bias_up, bias_down = _make_inputs(s, h, i, e)
    ref, counts, route_tokens, route_probs = _reference(tokens, gate, up, down, bias_up, bias_down, top_k, ec)

    tokens_d = tokens.cuda()
    up_d = up.cuda()
    down_d = down.cuda()
    bias_up_d = bias_up.cuda()
    bias_down_d = bias_down.cuda()
    output = torch.empty((s, h), device="cuda", dtype=torch.float32)
    hidden = torch.empty((e, ec, i), device="cuda", dtype=tokens_d.dtype)
    task_count = e * ((ec + 15) // 16) * ((i // 64) + (h // 64))
    task_queue = torch.empty((task_count,), device="cuda", dtype=torch.int32)
    task_head = torch.empty((1,), device="cuda", dtype=torch.int32)
    task_tail = torch.empty((1,), device="cuda", dtype=torch.int32)
    tile_sync = torch.empty((e * ((ec + 15) // 16),), device="cuda", dtype=torch.int32)
    processor_ready = torch.empty((args.num_programs,), device="cuda", dtype=torch.int32)
    processor_mailboxes = torch.empty((args.num_programs,), device="cuda", dtype=torch.int32)
    processor_seen = torch.empty((args.num_programs,), device="cuda", dtype=torch.int32)
    barriers = torch.empty((4,), device="cuda", dtype=torch.int32)
    debug_state = torch.empty((64,), device="cuda", dtype=torch.int32)

    _dynamic_ws_probe_kernel[(args.num_programs,)](
        tokens_d,
        up_d,
        bias_up_d,
        down_d,
        bias_down_d,
        output,
        hidden,
        counts.cuda(),
        route_tokens.cuda(),
        route_probs.cuda(),
        task_queue,
        task_head,
        task_tail,
        tile_sync,
        processor_ready,
        processor_mailboxes,
        processor_seen,
        barriers,
        debug_state,
        S=s,
        H=h,
        I=i,
        E=e,
        EC=ec,
        TOP_K=top_k,
        BLOCK_M=16,
        BLOCK_N=64,
        NUM_PROGRAMS=args.num_programs,
        NUM_WARPS=4,
        num_warps=4,
    )
    torch.cuda.synchronize()
    got = output.cpu()
    diff = (got - ref).abs()
    close = bool(torch.allclose(got, ref, rtol=8e-2, atol=8e-3))
    print(
        "dynamic_ws_probe",
        "num_programs",
        args.num_programs,
        "counts",
        counts.tolist(),
        "max_abs",
        float(diff.max()),
        "close",
        close,
    )
    if not close:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
