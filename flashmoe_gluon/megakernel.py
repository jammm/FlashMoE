from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.gfx1250 import tdm


ACT_IDENTITY = 0
ACT_SILU = 1
ACT_GELU = 2
ACT_RELU = 3


@dataclass(frozen=True)
class MegakernelSpec:
    s: int
    h: int
    i: int
    e: int
    top_k: int
    expert_capacity: int
    activation: int
    gated: bool
    dtype: torch.dtype
    block_h: int
    num_warps: int


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _validate_common(
    tokens: torch.Tensor,
    gate_weights: torch.Tensor,
    expert_up: torch.Tensor,
    expert_down: torch.Tensor,
    bias_up: torch.Tensor,
    bias_down: torch.Tensor,
    *,
    top_k: int,
    expert_capacity: Optional[int],
    activation: int,
    expert_up_v: Optional[torch.Tensor],
    bias_up_v: Optional[torch.Tensor],
    block_h: Optional[int],
    num_warps: int,
) -> MegakernelSpec:
    if tokens.ndim != 2 or gate_weights.ndim != 2:
        raise ValueError("tokens must be [S,H] and gate_weights must be [H,E]")
    if expert_up.ndim != 3 or expert_down.ndim != 3:
        raise ValueError("expert_up must be [E,H,I], expert_down must be [E,I,H]")
    if bias_up.ndim != 2 or bias_down.ndim != 2:
        raise ValueError("bias_up must be [E,I] and bias_down must be [E,H]")
    if tokens.device.type != "cuda":
        raise ValueError("Gluon megakernel expects HIP/CUDA tensors on device")
    if top_k not in (1, 2):
        raise ValueError("Gluon megakernel currently supports top_k in {1, 2}")
    if activation not in (ACT_IDENTITY, ACT_SILU, ACT_GELU, ACT_RELU):
        raise ValueError("unsupported activation")
    if tokens.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Gluon megakernel currently expects fp16 or bf16 tokens")
    if gate_weights.dtype != tokens.dtype or expert_up.dtype != tokens.dtype or expert_down.dtype != tokens.dtype:
        raise ValueError("tokens, gate_weights, expert_up, and expert_down must have the same dtype")

    s, h = tokens.shape
    hw, e = gate_weights.shape
    ew, hup, i = expert_up.shape
    ed, idown, hdown = expert_down.shape
    if hw != h or hup != h or (ed, idown, hdown) != (e, i, h) or ew != e:
        raise ValueError("token, gate, and expert tensor shapes are inconsistent")
    if bias_up.shape != (e, i) or bias_down.shape != (e, h):
        raise ValueError("bias tensor shapes are inconsistent with expert tensors")

    gated = expert_up_v is not None
    if gated:
        if bias_up_v is None:
            raise ValueError("bias_up_v is required when expert_up_v is provided")
        if expert_up_v.shape != expert_up.shape or bias_up_v.shape != bias_up.shape:
            raise ValueError("gated expert and bias tensors must match the up projection shapes")
        if expert_up_v.dtype != tokens.dtype or bias_up_v.dtype != tokens.dtype:
            raise ValueError("gated tensors must have the same dtype as tokens")
    elif bias_up_v is not None:
        raise ValueError("bias_up_v requires expert_up_v")

    if bias_up.dtype != tokens.dtype or bias_down.dtype != tokens.dtype:
        raise ValueError("bias tensors must have the same dtype as tokens")

    if expert_capacity is None:
        expert_capacity = _ceil_div(s * top_k, e)
    if expert_capacity <= 0:
        raise ValueError("expert_capacity must be positive")
    if num_warps not in (4, 8):
        raise ValueError("gfx1250 Gluon path currently expects num_warps in {4, 8}")

    resolved_block_h = _next_power_of_2(h) if block_h is None else block_h
    if resolved_block_h < h:
        raise ValueError("block_h must cover the full hidden dimension")

    return MegakernelSpec(
        s=s,
        h=h,
        i=i,
        e=e,
        top_k=top_k,
        expert_capacity=expert_capacity,
        activation=activation,
        gated=gated,
        dtype=tokens.dtype,
        block_h=resolved_block_h,
        num_warps=num_warps,
    )


@gluon.constexpr_function
def _hidden_layout(num_warps):
    return gl.BlockedLayout([1], [32], [num_warps], [0], [])


@gluon.constexpr_function
def _wmma_layout(num_warps):
    if num_warps == 4:
        warp_bases = [[0, 1], [1, 0]]
    else:
        warp_bases = [[0, 1], [0, 2], [1, 0]]
    return gl.amd.AMDWMMALayout(3, True, warp_bases, [], [16, 16, 32], [])


@gluon.constexpr_function
def _wmma_shared_a_layout(block_m, block_k):
    return gl.PaddedSharedLayout.with_identity_for([[block_k, 8]], [block_m, block_k], [1, 0], [])


@gluon.constexpr_function
def _wmma_shared_b_layout(block_k, block_n):
    return gl.PaddedSharedLayout.with_identity_for([[block_n, 16]], [block_k, block_n], [1, 0], [])


@gluon.constexpr_function
def _route_index_layout(block_m, num_warps):
    return gl.BlockedLayout([block_m, 1], [1, 32], [1, num_warps], [1, 0], [])


@gluon.jit
def _apply_activation(x, ACTIVATION: gl.constexpr):
    if ACTIVATION == 0:
        return x
    if ACTIVATION == 3:
        return gl.maximum(x, x * 0.0)
    if ACTIVATION == 1:
        return x * (1.0 / (1.0 + gl.exp(-x)))
    return 0.5 * x * (1.0 + gl.erf(x * 0.7071067811865476))


@gluon.jit
def _single_dispatch_top1_debug_kernel(
    tokens,
    gate_weights,
    expert_up,
    bias_up,
    expert_down,
    bias_down,
    output,
    H: gl.constexpr,
    I: gl.constexpr,
    E: gl.constexpr,
    BLOCK_E: gl.constexpr,
    BLOCK_H: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    token_id = gl.program_id(0)
    h_layout: gl.constexpr = _hidden_layout(NUM_WARPS)
    offs_h = gl.arange(0, BLOCK_H, layout=h_layout)
    h_mask = offs_h < H
    zero_f = token_id.to(gl.float32) * 0.0
    zero_i = token_id * 0

    row_max = zero_f - float("inf")
    for expert_id in gl.static_range(0, BLOCK_E):
        if expert_id < E:
            logit = zero_f
            for h_abs in gl.static_range(0, H):
                t = gl.load(tokens + token_id * H + h_abs).to(gl.float32)
                w = gl.load(gate_weights + h_abs * E + expert_id).to(gl.float32)
                logit += t * w
            row_max = gl.maximum(row_max, logit)

    top_idx = zero_i
    top_prob = zero_f - 1.0
    for expert_id in gl.static_range(0, BLOCK_E):
        if expert_id < E:
            expert_id_t = zero_i + expert_id
            logit = zero_f
            for h_abs in gl.static_range(0, H):
                t = gl.load(tokens + token_id * H + h_abs).to(gl.float32)
                w = gl.load(gate_weights + h_abs * E + expert_id).to(gl.float32)
                logit += t * w
            prob = gl.exp(logit - row_max)
            better = prob > top_prob
            top_prob = gl.where(better, prob, top_prob)
            top_idx = gl.where(better, expert_id_t, top_idx)

    route_vals = gl.load(bias_down + top_idx * H + offs_h, mask=h_mask, other=0.0).to(gl.float32)
    for i_abs in gl.static_range(0, I):
        up_acc = gl.load(bias_up + top_idx * I + i_abs).to(gl.float32)
        for h_abs in gl.static_range(0, H):
            t = gl.load(tokens + token_id * H + h_abs).to(gl.float32)
            wu = gl.load(expert_up + top_idx * H * I + h_abs * I + i_abs).to(gl.float32)
            up_acc += t * wu
        wd = gl.load(expert_down + top_idx * I * H + i_abs * H + offs_h, mask=h_mask, other=0.0).to(gl.float32)
        route_vals += up_acc * wd
    gl.store(output + token_id * H + offs_h, route_vals, mask=h_mask)


@gluon.jit
def _global_barrier(barriers, phase: gl.constexpr, NUM_PROGRAMS: gl.constexpr):
    if NUM_PROGRAMS > 1:
        gl.atomic_add(barriers + phase, 1, sem="release", scope="gpu")
        while gl.atomic_add(
            barriers + phase,
            0,
            sem="acquire",
            scope="gpu",
        ) < NUM_PROGRAMS:
            pass


@gluon.jit
def _persistent_phase_barrier(
    barriers,
    phase: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
):
    _global_barrier(barriers, phase, NUM_PROGRAMS)


@gluon.jit
def _tdm_wait(num_outstanding: gl.constexpr):
    tdm.async_wait(num_outstanding)


@gluon.jit
def _scheduler_reset(processor_ready, processor_mailboxes, NUM_PROGRAMS: gl.constexpr):
    pid = gl.program_id(0)
    if pid == 0:
        for program_id in gl.static_range(0, NUM_PROGRAMS):
            gl.atomic_xchg(processor_ready + program_id, 0, sem="release", scope="gpu")
            gl.atomic_xchg(processor_mailboxes + program_id, -1, sem="release", scope="gpu")


@gluon.jit
def _scheduler_run(processor_ready, processor_mailboxes, TOTAL_TASKS: gl.constexpr, NUM_PROGRAMS: gl.constexpr):
    pid = gl.program_id(0)
    if pid == 0:
        next_task = pid * 0
        stopped = pid * 0
        while stopped < NUM_PROGRAMS - 1:
            for program_id in gl.static_range(1, NUM_PROGRAMS):
                ready = gl.atomic_xchg(
                    processor_ready + program_id,
                    0,
                    sem="acquire",
                    scope="gpu",
                )
                if ready != 0:
                    task = next_task
                    if next_task < TOTAL_TASKS:
                        next_task += 1
                    else:
                        task = TOTAL_TASKS
                        stopped += 1
                    gl.atomic_xchg(
                        processor_mailboxes + program_id,
                        task,
                        sem="release",
                        scope="gpu",
                    )


@gluon.jit
def _processor_next_task(processor_ready, processor_mailboxes):
    pid = gl.program_id(0)
    gl.atomic_xchg(processor_ready + pid, 1, sem="release", scope="gpu")
    task = gl.atomic_add(processor_mailboxes + pid, 0, sem="acquire", scope="gpu")
    while task < 0:
        task = gl.atomic_add(processor_mailboxes + pid, 0, sem="acquire", scope="gpu")
    gl.atomic_xchg(processor_mailboxes + pid, -1, sem="relaxed", scope="gpu")
    return task


@gluon.jit
def _persistent_dispatch_moe_kernel(
    tokens,
    gate_weights,
    expert_up,
    bias_up,
    expert_up_v,
    bias_up_v,
    expert_down,
    bias_down,
    output,
    expert_counts,
    route_tokens,
    route_probs,
    processor_ready,
    processor_mailboxes,
    barriers,
    init_flag,
    launch_epoch,
    S: gl.constexpr,
    H: gl.constexpr,
    I: gl.constexpr,
    E: gl.constexpr,
    EC: gl.constexpr,
    TOP_K: gl.constexpr,
    BLOCK_E: gl.constexpr,
    BLOCK_H: gl.constexpr,
    ACTIVATION: gl.constexpr,
    GATED: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    pid = gl.program_id(0)
    h_layout: gl.constexpr = _hidden_layout(NUM_WARPS)
    offs_h = gl.arange(0, BLOCK_H, layout=h_layout)
    h_mask = offs_h < H

    if pid == 0:
        for expert_id in gl.static_range(0, BLOCK_E):
            if expert_id < E:
                gl.store(expert_counts + expert_id, 0)
        for phase in gl.static_range(0, 4):
            gl.store(barriers + phase, 0)
        if NUM_PROGRAMS > 1:
            gl.atomic_xchg(init_flag, launch_epoch, sem="release", scope="gpu")
        else:
            gl.store(init_flag, launch_epoch)

    if NUM_PROGRAMS > 1:
        while gl.atomic_cas(
            init_flag,
            launch_epoch,
            launch_epoch,
            sem="acquire",
            scope="gpu",
        ) != launch_epoch:
            pass

    out_base = pid * BLOCK_H
    while out_base < S * H:
        gl.store(output + out_base + offs_h, gl.full((BLOCK_H,), 0.0, gl.float32, layout=h_layout),
                 mask=out_base + offs_h < S * H)
        out_base += NUM_PROGRAMS * BLOCK_H
    _persistent_phase_barrier(barriers, 0, NUM_PROGRAMS)

    token_id = pid
    while token_id < S:
        zero_f = token_id.to(gl.float32) * 0.0
        zero_i = token_id * 0

        top0_idx = zero_i
        top1_idx = zero_i
        top0_logit = zero_f - float("inf")
        top1_logit = zero_f - float("inf")

        for expert_id in gl.static_range(0, BLOCK_E):
            if expert_id < E:
                expert_id_t = zero_i + expert_id
                logit = zero_f
                for h_abs in gl.static_range(0, H):
                    t = gl.load(tokens + token_id * H + h_abs).to(gl.float32)
                    w = gl.load(gate_weights + h_abs * E + expert_id).to(gl.float32)
                    logit += t * w

                better0 = logit > top0_logit
                better1 = (logit > top1_logit) & (logit <= top0_logit)
                old_top0_idx = top0_idx
                old_top0_logit = top0_logit

                top1_idx = gl.where(better0, old_top0_idx, top1_idx)
                top1_logit = gl.where(better0, old_top0_logit, top1_logit)
                top0_idx = gl.where(better0, expert_id_t, top0_idx)
                top0_logit = gl.where(better0, logit, top0_logit)

                top1_idx = gl.where(better1, expert_id_t, top1_idx)
                top1_logit = gl.where(better1, logit, top1_logit)

        max_top = gl.maximum(top0_logit, top1_logit)
        top0_exp = gl.exp(top0_logit - max_top)
        top1_exp = gl.exp(top1_logit - max_top)
        denom = top0_exp + top1_exp

        for route_id in gl.static_range(0, 2):
            if route_id < TOP_K:
                expert_idx = gl.where(route_id == 0, top0_idx, top1_idx)
                route_prob = zero_f + 1.0
                if TOP_K == 2:
                    route_exp = gl.where(route_id == 0, top0_exp, top1_exp)
                    route_prob = route_exp / denom

                slot = gl.atomic_add(expert_counts + expert_idx, 1, sem="relaxed", scope="gpu")
                active = slot < EC
                safe_slot = gl.minimum(slot, slot * 0 + EC - 1)
                gl.store(route_tokens + expert_idx * EC + safe_slot, token_id, mask=active)
                gl.store(route_probs + expert_idx * EC + safe_slot, route_prob, mask=active)

        token_id += NUM_PROGRAMS
    _persistent_phase_barrier(barriers, 1, NUM_PROGRAMS)

    _scheduler_reset(processor_ready, processor_mailboxes, NUM_PROGRAMS)
    _persistent_phase_barrier(barriers, 2, NUM_PROGRAMS)

    if pid == 0:
        _scheduler_run(processor_ready, processor_mailboxes, E * EC, NUM_PROGRAMS)
    else:
        task_id = _processor_next_task(processor_ready, processor_mailboxes)
        while task_id < E * EC:
            expert_idx = task_id // EC
            slot = task_id - expert_idx * EC
            expert_count = gl.load(expert_counts + expert_idx)
            active = slot < gl.minimum(expert_count, expert_count * 0 + EC)
            routed_token = gl.load(route_tokens + expert_idx * EC + slot, mask=active, other=0)
            route_prob = gl.load(route_probs + expert_idx * EC + slot, mask=active, other=0.0).to(gl.float32)

            route_vals = gl.load(
                bias_down + expert_idx * H + offs_h,
                mask=h_mask,
                other=0.0,
            ).to(gl.float32)

            for i_abs in gl.static_range(0, I):
                up_acc = gl.load(bias_up + expert_idx * I + i_abs).to(gl.float32)
                for h_abs in gl.static_range(0, H):
                    t = gl.load(tokens + routed_token * H + h_abs, mask=active, other=0.0).to(gl.float32)
                    wu = gl.load(
                        expert_up + expert_idx * H * I + h_abs * I + i_abs
                    ).to(gl.float32)
                    up_acc += t * wu

                hidden = _apply_activation(up_acc, ACTIVATION)
                if GATED:
                    v_acc = gl.load(bias_up_v + expert_idx * I + i_abs).to(gl.float32)
                    for h_abs in gl.static_range(0, H):
                        t = gl.load(tokens + routed_token * H + h_abs, mask=active, other=0.0).to(gl.float32)
                        wv = gl.load(
                            expert_up_v + expert_idx * H * I + h_abs * I + i_abs
                        ).to(gl.float32)
                        v_acc += t * wv
                    hidden *= v_acc

                wd = gl.load(
                    expert_down + expert_idx * I * H + i_abs * H + offs_h,
                    mask=h_mask,
                    other=0.0,
                ).to(gl.float32)
                route_vals += hidden * wd

            gl.atomic_add(
                output + routed_token * H + offs_h,
                route_prob * route_vals,
                sem="relaxed",
                scope="gpu",
                mask=active & h_mask,
            )

            task_id = _processor_next_task(processor_ready, processor_mailboxes)

    _global_barrier(barriers, 3, NUM_PROGRAMS)


@gluon.jit
def _persistent_tdm_wmma_kernel(
    tokens,
    gate_weights,
    expert_up,
    bias_up,
    expert_up_v,
    bias_up_v,
    expert_down,
    bias_down,
    output,
    hidden,
    expert_counts,
    route_tokens,
    route_probs,
    processor_ready,
    processor_mailboxes,
    barriers,
    init_flag,
    launch_epoch,
    S: gl.constexpr,
    H: gl.constexpr,
    I: gl.constexpr,
    E: gl.constexpr,
    EC: gl.constexpr,
    TOP_K: gl.constexpr,
    ACTIVATION: gl.constexpr,
    GATED: gl.constexpr,
    BLOCK_E: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    gl.static_assert(BLOCK_M % 16 == 0, "TDM/WMMA path expects a WMMA-row-aligned BLOCK_M")
    gl.static_assert(BLOCK_N == 64, "TDM/WMMA path expects 64-column tiles")
    gl.static_assert(H % BLOCK_N == 0, "TDM/WMMA path expects H to be divisible by 64")
    gl.static_assert(I % BLOCK_N == 0, "TDM/WMMA path expects I to be divisible by 64")

    pid = gl.program_id(0)
    h_layout: gl.constexpr = _hidden_layout(NUM_WARPS)
    offs_h = gl.arange(0, BLOCK_H, layout=h_layout)
    h_mask = offs_h < H

    if pid == 0:
        for expert_id in gl.static_range(0, BLOCK_E):
            if expert_id < E:
                gl.store(expert_counts + expert_id, 0)
        for phase in gl.static_range(0, 6):
            gl.store(barriers + phase, 0)
        if NUM_PROGRAMS > 1:
            gl.atomic_xchg(init_flag, launch_epoch, sem="release", scope="gpu")
        else:
            gl.store(init_flag, launch_epoch)

    if NUM_PROGRAMS > 1:
        while gl.atomic_cas(
            init_flag,
            launch_epoch,
            launch_epoch,
            sem="acquire",
            scope="gpu",
        ) != launch_epoch:
            pass

    out_base = pid * BLOCK_H
    while out_base < S * H:
        gl.store(
            output + out_base + offs_h,
            gl.full((BLOCK_H,), 0.0, gl.float32, layout=h_layout),
            mask=out_base + offs_h < S * H,
        )
        out_base += NUM_PROGRAMS * BLOCK_H
    _persistent_phase_barrier(barriers, 0, NUM_PROGRAMS)

    token_id = pid
    while token_id < S:
        zero_f = token_id.to(gl.float32) * 0.0
        zero_i = token_id * 0

        top0_idx = zero_i
        top1_idx = zero_i
        top0_logit = zero_f - float("inf")
        top1_logit = zero_f - float("inf")
        for expert_id in gl.static_range(0, BLOCK_E):
            if expert_id < E:
                expert_id_t = zero_i + expert_id
                logit = zero_f
                for h_abs in gl.static_range(0, H):
                    t = gl.load(tokens + token_id * H + h_abs).to(gl.float32)
                    w = gl.load(gate_weights + h_abs * E + expert_id).to(gl.float32)
                    logit += t * w

                better0 = logit > top0_logit
                better1 = (logit > top1_logit) & (logit <= top0_logit)
                old_top0_idx = top0_idx
                old_top0_logit = top0_logit

                top1_idx = gl.where(better0, old_top0_idx, top1_idx)
                top1_logit = gl.where(better0, old_top0_logit, top1_logit)
                top0_idx = gl.where(better0, expert_id_t, top0_idx)
                top0_logit = gl.where(better0, logit, top0_logit)

                top1_idx = gl.where(better1, expert_id_t, top1_idx)
                top1_logit = gl.where(better1, logit, top1_logit)

        max_top = gl.maximum(top0_logit, top1_logit)
        top0_exp = gl.exp(top0_logit - max_top)
        top1_exp = gl.exp(top1_logit - max_top)
        denom = top0_exp + top1_exp

        for route_id in gl.static_range(0, 2):
            if route_id < TOP_K:
                expert_idx = gl.where(route_id == 0, top0_idx, top1_idx)
                route_prob = zero_f + 1.0
                if TOP_K == 2:
                    route_exp = gl.where(route_id == 0, top0_exp, top1_exp)
                    route_prob = route_exp / denom

                slot = gl.atomic_add(expert_counts + expert_idx, 1, sem="relaxed", scope="gpu")
                active = slot < EC
                safe_slot = gl.minimum(slot, slot * 0 + EC - 1)
                gl.store(route_tokens + expert_idx * EC + safe_slot, token_id, mask=active)
                gl.store(route_probs + expert_idx * EC + safe_slot, route_prob, mask=active)

        token_id += NUM_PROGRAMS
    _persistent_phase_barrier(barriers, 1, NUM_PROGRAMS)

    _scheduler_reset(processor_ready, processor_mailboxes, NUM_PROGRAMS)
    _persistent_phase_barrier(barriers, 2, NUM_PROGRAMS)

    wmma_layout: gl.constexpr = _wmma_layout(NUM_WARPS)
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=wmma_layout, k_width=8)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=wmma_layout, k_width=8)
    shared_a_layout: gl.constexpr = _wmma_shared_a_layout(BLOCK_M, BLOCK_N)
    shared_b_layout: gl.constexpr = _wmma_shared_b_layout(BLOCK_N, BLOCK_N)
    shared_h_layout: gl.constexpr = _wmma_shared_a_layout(BLOCK_M, BLOCK_N)
    shared_down_layout: gl.constexpr = _wmma_shared_b_layout(BLOCK_N, BLOCK_N)
    route_layout: gl.constexpr = _route_index_layout(BLOCK_M, NUM_WARPS)
    route_offs = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, route_layout))

    route_blocks: gl.constexpr = (EC + BLOCK_M - 1) // BLOCK_M
    h_tiles: gl.constexpr = H // BLOCK_N
    i_tiles: gl.constexpr = I // BLOCK_N
    gemm0_tasks: gl.constexpr = E * route_blocks * i_tiles
    if pid == 0:
        _scheduler_run(processor_ready, processor_mailboxes, gemm0_tasks, NUM_PROGRAMS)
    else:
        task_id = _processor_next_task(processor_ready, processor_mailboxes)
        while task_id < gemm0_tasks:
            expert_idx = task_id // (route_blocks * i_tiles)
            rem_task = task_id - expert_idx * route_blocks * i_tiles
            route_block = rem_task // i_tiles
            i_tile = rem_task - route_block * i_tiles
            base_slot = route_block * BLOCK_M
            base_i = i_tile * BLOCK_N
            expert_count = gl.load(expert_counts + expert_idx)
            active_rows = base_slot + route_offs < gl.minimum(expert_count, expert_count * 0 + EC)
            safe_route_rows = gl.minimum(base_slot + route_offs, route_offs * 0 + EC - 1)
            gathered_tokens = gl.load(
                route_tokens + expert_idx * EC + safe_route_rows,
                mask=active_rows,
                other=0,
            ).to(gl.int32)

            x_smem = gl.allocate_shared_memory(
                tokens.dtype.element_ty,
                shape=[BLOCK_M, BLOCK_N],
                layout=shared_a_layout,
            )
            w_smem = gl.allocate_shared_memory(
                expert_up.dtype.element_ty,
                shape=[BLOCK_N, BLOCK_N],
                layout=shared_b_layout,
            )
            acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=wmma_layout)
            for h_tile in gl.static_range(0, h_tiles):
                x_desc = tdm.make_tensor_descriptor(
                    base=tokens + h_tile * BLOCK_N,
                    shape=(S, BLOCK_N),
                    strides=(H, 1),
                    block_shape=(BLOCK_M, BLOCK_N),
                    layout=shared_a_layout,
                )
                w_desc = tdm.make_tensor_descriptor(
                    base=expert_up + expert_idx * H * I + (h_tile * BLOCK_N) * I + base_i,
                    shape=(BLOCK_N, BLOCK_N),
                    strides=(I, 1),
                    block_shape=(BLOCK_N, BLOCK_N),
                    layout=shared_b_layout,
                )
                tdm.async_gather(x_desc, gathered_tokens, x_smem)
                tdm.async_load(w_desc, [0, 0], w_smem)
                _tdm_wait(0)
                x_frag = x_smem.load(layout=dot_a)
                w_frag = w_smem.load(layout=dot_b)
                acc = gl.amd.gfx1250.wmma(x_frag, w_frag, acc)

            bias_layout: gl.constexpr = gl.SliceLayout(0, wmma_layout)
            offs_i = base_i + gl.arange(0, BLOCK_N, layout=bias_layout)
            bias = gl.load(bias_up + expert_idx * I + offs_i).to(gl.float32)
            bias_2d = gl.convert_layout(gl.expand_dims(bias, 0), wmma_layout)
            acc += bias_2d
            hidden_acc = _apply_activation(acc, ACTIVATION)

            if GATED:
                wv_smem = gl.allocate_shared_memory(
                    expert_up_v.dtype.element_ty,
                    shape=[BLOCK_N, BLOCK_N],
                    layout=shared_b_layout,
                )
                acc_v = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=wmma_layout)
                for h_tile_v in gl.static_range(0, h_tiles):
                    x_desc = tdm.make_tensor_descriptor(
                        base=tokens + h_tile_v * BLOCK_N,
                        shape=(S, BLOCK_N),
                        strides=(H, 1),
                        block_shape=(BLOCK_M, BLOCK_N),
                        layout=shared_a_layout,
                    )
                    wv_desc = tdm.make_tensor_descriptor(
                        base=expert_up_v + expert_idx * H * I + (h_tile_v * BLOCK_N) * I + base_i,
                        shape=(BLOCK_N, BLOCK_N),
                        strides=(I, 1),
                        block_shape=(BLOCK_N, BLOCK_N),
                        layout=shared_b_layout,
                    )
                    tdm.async_gather(x_desc, gathered_tokens, x_smem)
                    tdm.async_load(wv_desc, [0, 0], wv_smem)
                    _tdm_wait(0)
                    x_frag = x_smem.load(layout=dot_a)
                    wv_frag = wv_smem.load(layout=dot_b)
                    acc_v = gl.amd.gfx1250.wmma(x_frag, wv_frag, acc_v)
                bias_v = gl.load(bias_up_v + expert_idx * I + offs_i).to(gl.float32)
                bias_v_2d = gl.convert_layout(gl.expand_dims(bias_v, 0), wmma_layout)
                acc_v += bias_v_2d
                hidden_acc *= acc_v

            rows = base_slot + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, wmma_layout))
            safe_rows = gl.minimum(rows, rows * 0 + EC - 1)
            cols = base_i + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, wmma_layout))
            rows_2d = gl.expand_dims(safe_rows, 1)
            raw_rows_2d = gl.expand_dims(rows, 1)
            cols_2d = gl.expand_dims(cols, 0)
            hidden_offsets = gl.convert_layout((expert_idx * EC + rows_2d) * I + cols_2d, wmma_layout)
            store_mask = gl.convert_layout(
                raw_rows_2d < gl.minimum(expert_count, expert_count * 0 + EC),
                wmma_layout,
            )
            gl.store(
                hidden + hidden_offsets,
                hidden_acc,
                mask=store_mask,
            )
            task_id = _processor_next_task(processor_ready, processor_mailboxes)

    _persistent_phase_barrier(barriers, 3, NUM_PROGRAMS)
    _scheduler_reset(processor_ready, processor_mailboxes, NUM_PROGRAMS)
    _persistent_phase_barrier(barriers, 4, NUM_PROGRAMS)

    gemm1_tasks: gl.constexpr = E * route_blocks * h_tiles
    if pid == 0:
        _scheduler_run(processor_ready, processor_mailboxes, gemm1_tasks, NUM_PROGRAMS)
    else:
        task_id = _processor_next_task(processor_ready, processor_mailboxes)
        while task_id < gemm1_tasks:
            expert_idx = task_id // (route_blocks * h_tiles)
            rem_task = task_id - expert_idx * route_blocks * h_tiles
            route_block = rem_task // h_tiles
            h_tile = rem_task - route_block * h_tiles
            base_slot = route_block * BLOCK_M
            base_h = h_tile * BLOCK_N
            expert_count = gl.load(expert_counts + expert_idx)

            h_smem = gl.allocate_shared_memory(
                hidden.dtype.element_ty,
                shape=[BLOCK_M, BLOCK_N],
                layout=shared_h_layout,
            )
            down_smem = gl.allocate_shared_memory(
                expert_down.dtype.element_ty,
                shape=[BLOCK_N, BLOCK_N],
                layout=shared_down_layout,
            )
            acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=wmma_layout)
            for i_tile_down in gl.static_range(0, i_tiles):
                h_desc = tdm.make_tensor_descriptor(
                    base=hidden + expert_idx * EC * I + base_slot * I + i_tile_down * BLOCK_N,
                    shape=(EC, BLOCK_N),
                    strides=(I, 1),
                    block_shape=(BLOCK_M, BLOCK_N),
                    layout=shared_h_layout,
                )
                down_desc = tdm.make_tensor_descriptor(
                    base=expert_down + expert_idx * I * H + (i_tile_down * BLOCK_N) * H + base_h,
                    shape=(BLOCK_N, BLOCK_N),
                    strides=(H, 1),
                    block_shape=(BLOCK_N, BLOCK_N),
                    layout=shared_down_layout,
                )
                tdm.async_load(h_desc, [0, 0], h_smem)
                tdm.async_load(down_desc, [0, 0], down_smem)
                _tdm_wait(0)
                h_frag = h_smem.load(layout=dot_a)
                down_frag = down_smem.load(layout=dot_b)
                acc = gl.amd.gfx1250.wmma(h_frag, down_frag, acc)

            bias_layout: gl.constexpr = gl.SliceLayout(0, wmma_layout)
            offs_out = base_h + gl.arange(0, BLOCK_N, layout=bias_layout)
            bias = gl.load(bias_down + expert_idx * H + offs_out).to(gl.float32)
            bias_2d = gl.convert_layout(gl.expand_dims(bias, 0), wmma_layout)
            acc += bias_2d

            cols = base_h + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, wmma_layout))
            route_rows = base_slot + route_offs
            safe_route_rows = gl.minimum(route_rows, route_rows * 0 + EC - 1)
            active_route_rows = route_rows < gl.minimum(expert_count, expert_count * 0 + EC)
            token_ids_route = gl.load(
                route_tokens + expert_idx * EC + safe_route_rows,
                mask=active_route_rows,
                other=0,
            ).to(gl.int32)
            probs_route = gl.load(
                route_probs + expert_idx * EC + safe_route_rows,
                mask=active_route_rows,
                other=0.0,
            ).to(gl.float32)
            row_layout: gl.constexpr = gl.SliceLayout(1, wmma_layout)
            active_rows = gl.convert_layout(active_route_rows, row_layout)
            token_ids = gl.convert_layout(token_ids_route, row_layout)
            probs = gl.convert_layout(probs_route, row_layout)
            token_ids_2d = gl.expand_dims(token_ids, 1)
            cols_2d = gl.expand_dims(cols, 0)
            output_offsets = gl.convert_layout(token_ids_2d * H + cols_2d, wmma_layout)
            probs_2d = gl.convert_layout(gl.expand_dims(probs, 1), wmma_layout)
            active_2d = gl.convert_layout(gl.expand_dims(active_rows, 1), wmma_layout)
            if TOP_K == 1:
                gl.store(
                    output + output_offsets,
                    acc * probs_2d,
                    mask=active_2d,
                )
            else:
                gl.atomic_add(
                    output + output_offsets,
                    acc * probs_2d,
                    sem="relaxed",
                    scope="gpu",
                    mask=active_2d,
                )
            task_id = _processor_next_task(processor_ready, processor_mailboxes)

    _global_barrier(barriers, 5, NUM_PROGRAMS)


@gluon.jit
def _single_dispatch_moe_kernel(
    tokens,
    gate_weights,
    expert_up,
    bias_up,
    expert_up_v,
    bias_up_v,
    expert_down,
    bias_down,
    output,
    H: gl.constexpr,
    I: gl.constexpr,
    E: gl.constexpr,
    TOP_K: gl.constexpr,
    BLOCK_E: gl.constexpr,
    BLOCK_H: gl.constexpr,
    ACTIVATION: gl.constexpr,
    GATED: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    token_id = gl.program_id(0)
    h_layout: gl.constexpr = _hidden_layout(NUM_WARPS)
    offs_h = gl.arange(0, BLOCK_H, layout=h_layout)
    h_mask = offs_h < H
    zero_f = token_id.to(gl.float32) * 0.0
    zero_i = token_id * 0

    top0_idx = zero_i
    top1_idx = zero_i
    top0_logit = zero_f - float("inf")
    top1_logit = zero_f - float("inf")

    for expert_id in gl.static_range(0, BLOCK_E):
        if expert_id < E:
            expert_id_t = zero_i + expert_id
            logit = zero_f
            for h_abs in gl.static_range(0, H):
                t = gl.load(tokens + token_id * H + h_abs).to(gl.float32)
                w = gl.load(gate_weights + h_abs * E + expert_id).to(gl.float32)
                logit += t * w

            better0 = logit > top0_logit
            better1 = (logit > top1_logit) & (logit <= top0_logit)
            old_top0_idx = top0_idx
            old_top0_logit = top0_logit

            top1_idx = gl.where(better0, old_top0_idx, top1_idx)
            top1_logit = gl.where(better0, old_top0_logit, top1_logit)
            top0_idx = gl.where(better0, expert_id_t, top0_idx)
            top0_logit = gl.where(better0, logit, top0_logit)

            top1_idx = gl.where(better1, expert_id_t, top1_idx)
            top1_logit = gl.where(better1, logit, top1_logit)

    max_top = gl.maximum(top0_logit, top1_logit)
    top0_exp = gl.exp(top0_logit - max_top)
    top1_exp = gl.exp(top1_logit - max_top)
    denom = top0_exp + top1_exp

    out_vals = zero_f + gl.full((BLOCK_H,), 0.0, gl.float32, layout=h_layout)
    for route_id in gl.static_range(0, 2):
        if route_id < TOP_K:
            expert_idx = gl.where(route_id == 0, top0_idx, top1_idx)
            route_prob = zero_f + 1.0
            if TOP_K == 2:
                route_exp = gl.where(route_id == 0, top0_exp, top1_exp)
                route_prob = route_exp / denom

            route_vals = gl.load(
                bias_down + expert_idx * H + offs_h,
                mask=h_mask,
                other=0.0,
            ).to(gl.float32)

            for i_abs in gl.static_range(0, I):
                up_acc = gl.load(bias_up + expert_idx * I + i_abs).to(gl.float32)
                for h_abs in gl.static_range(0, H):
                    t = gl.load(tokens + token_id * H + h_abs).to(gl.float32)
                    wu = gl.load(
                        expert_up + expert_idx * H * I + h_abs * I + i_abs
                    ).to(gl.float32)
                    up_acc += t * wu

                hidden = _apply_activation(up_acc, ACTIVATION)
                if GATED:
                    v_acc = gl.load(bias_up_v + expert_idx * I + i_abs).to(gl.float32)
                    for h_abs in gl.static_range(0, H):
                        t = gl.load(tokens + token_id * H + h_abs).to(gl.float32)
                        wv = gl.load(
                            expert_up_v + expert_idx * H * I + h_abs * I + i_abs
                        ).to(gl.float32)
                        v_acc += t * wv
                    hidden *= v_acc

                wd = gl.load(
                    expert_down + expert_idx * I * H + i_abs * H + offs_h,
                    mask=h_mask,
                    other=0.0,
                ).to(gl.float32)
                route_vals += hidden * wd

            out_vals += route_prob * route_vals

    gl.store(output + token_id * H + offs_h, out_vals, mask=h_mask)


def forward_scalar_top1_debug(
    tokens: torch.Tensor,
    gate_weights: torch.Tensor,
    expert_up: torch.Tensor,
    expert_down: torch.Tensor,
    bias_up: torch.Tensor,
    bias_down: torch.Tensor,
    *,
    expert_capacity: Optional[int] = None,
    block_h: Optional[int] = None,
    num_warps: int = 4,
) -> torch.Tensor:
    """Tiny top-1 identity probe used only while bringing up the Gluon compiler path."""
    spec = _validate_common(
        tokens,
        gate_weights,
        expert_up,
        expert_down,
        bias_up,
        bias_down,
        top_k=1,
        expert_capacity=expert_capacity,
        activation=ACT_IDENTITY,
        expert_up_v=None,
        bias_up_v=None,
        block_h=block_h,
        num_warps=num_warps,
    )
    block_e = max(16, _next_power_of_2(spec.e))
    out = torch.empty((spec.s, spec.h), device=tokens.device, dtype=torch.float32)
    _single_dispatch_top1_debug_kernel[(spec.s,)](
        tokens,
        gate_weights,
        expert_up,
        bias_up,
        expert_down,
        bias_down,
        out,
        H=spec.h,
        I=spec.i,
        E=spec.e,
        BLOCK_E=block_e,
        BLOCK_H=spec.block_h,
        NUM_WARPS=spec.num_warps,
        num_warps=spec.num_warps,
    )
    return out


def forward_megakernel(
    tokens: torch.Tensor,
    gate_weights: torch.Tensor,
    expert_up: torch.Tensor,
    expert_down: torch.Tensor,
    bias_up: torch.Tensor,
    bias_down: torch.Tensor,
    *,
    top_k: int,
    expert_capacity: Optional[int] = None,
    activation: int = ACT_IDENTITY,
    expert_up_v: Optional[torch.Tensor] = None,
    bias_up_v: Optional[torch.Tensor] = None,
    swish_alpha: float = 1.0,
    swish_beta: float = 1.0,
    block_h: Optional[int] = None,
    num_warps: int = 4,
    num_programs: int = 2,
    use_persistent: bool = True,
    use_tdm_wmma: bool = True,
    return_counts: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    del swish_alpha, swish_beta
    spec = _validate_common(
        tokens,
        gate_weights,
        expert_up,
        expert_down,
        bias_up,
        bias_down,
        top_k=top_k,
        expert_capacity=expert_capacity,
        activation=activation,
        expert_up_v=expert_up_v,
        bias_up_v=bias_up_v,
        block_h=block_h,
        num_warps=num_warps,
    )
    if num_programs <= 0:
        raise ValueError("num_programs must be positive")
    if use_persistent and num_programs < 2:
        raise ValueError("persistent Gluon scheduling requires at least two programs")
    block_e = max(16, _next_power_of_2(spec.e))
    up_v_arg = expert_up_v if expert_up_v is not None else expert_up
    bias_up_v_arg = bias_up_v if bias_up_v is not None else bias_up
    out = torch.empty((spec.s, spec.h), device=tokens.device, dtype=torch.float32)
    tdm_wmma_eligible = use_tdm_wmma and spec.top_k in (1, 2) and spec.h % 64 == 0 and spec.i % 64 == 0

    if use_persistent:
        expert_counts = torch.empty((spec.e,), device=tokens.device, dtype=torch.int32)
        route_tokens = torch.empty((spec.e, spec.expert_capacity), device=tokens.device, dtype=torch.int32)
        route_probs = torch.empty((spec.e, spec.expert_capacity), device=tokens.device, dtype=torch.float32)
        processor_ready = torch.empty((num_programs,), device=tokens.device, dtype=torch.int32)
        processor_mailboxes = torch.empty((num_programs,), device=tokens.device, dtype=torch.int32)
        barriers = torch.empty((6,), device=tokens.device, dtype=torch.int32)
        init_flag = torch.empty((1,), device=tokens.device, dtype=torch.int32)
        launch_epoch = random.randint(1, (1 << 30) - 1)
        if tdm_wmma_eligible:
            block_m = 16
            hidden = torch.empty(
                (spec.e, spec.expert_capacity, spec.i),
                device=tokens.device,
                dtype=tokens.dtype,
            )
            _persistent_tdm_wmma_kernel[(num_programs,)](
                tokens,
                gate_weights,
                expert_up,
                bias_up,
                up_v_arg,
                bias_up_v_arg,
                expert_down,
                bias_down,
                out,
                hidden,
                expert_counts,
                route_tokens,
                route_probs,
                processor_ready,
                processor_mailboxes,
                barriers,
                init_flag,
                launch_epoch,
                S=spec.s,
                H=spec.h,
                I=spec.i,
                E=spec.e,
                EC=spec.expert_capacity,
                TOP_K=spec.top_k,
                ACTIVATION=spec.activation,
                GATED=spec.gated,
                BLOCK_E=block_e,
                BLOCK_H=spec.block_h,
                BLOCK_M=block_m,
                BLOCK_N=64,
                NUM_PROGRAMS=num_programs,
                NUM_WARPS=spec.num_warps,
                num_warps=spec.num_warps,
            )
            if return_counts:
                return out, expert_counts
            return out

        _persistent_dispatch_moe_kernel[(num_programs,)](
            tokens,
            gate_weights,
            expert_up,
            bias_up,
            up_v_arg,
            bias_up_v_arg,
            expert_down,
            bias_down,
            out,
            expert_counts,
            route_tokens,
            route_probs,
            processor_ready,
            processor_mailboxes,
            barriers,
            init_flag,
            launch_epoch,
            S=spec.s,
            H=spec.h,
            I=spec.i,
            E=spec.e,
            EC=spec.expert_capacity,
            TOP_K=spec.top_k,
            BLOCK_E=block_e,
            BLOCK_H=spec.block_h,
            ACTIVATION=spec.activation,
            GATED=spec.gated,
            NUM_PROGRAMS=num_programs,
            NUM_WARPS=spec.num_warps,
            num_warps=spec.num_warps,
        )
        if return_counts:
            return out, expert_counts
        return out

    if return_counts:
        raise NotImplementedError("count return is only available for the persistent Gluon megakernel")
    _single_dispatch_moe_kernel[(spec.s,)](
        tokens,
        gate_weights,
        expert_up,
        bias_up,
        up_v_arg,
        bias_up_v_arg,
        expert_down,
        bias_down,
        out,
        H=spec.h,
        I=spec.i,
        E=spec.e,
        TOP_K=spec.top_k,
        BLOCK_E=block_e,
        BLOCK_H=spec.block_h,
        ACTIVATION=spec.activation,
        GATED=spec.gated,
        NUM_WARPS=spec.num_warps,
        num_warps=spec.num_warps,
    )
    return out


__all__ = [
    "ACT_IDENTITY",
    "ACT_SILU",
    "ACT_GELU",
    "ACT_RELU",
    "MegakernelSpec",
    "forward_megakernel",
    "forward_scalar_top1_debug",
]
