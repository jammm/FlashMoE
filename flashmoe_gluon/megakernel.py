from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.gfx1250 import mbarrier, tdm

from . import rocshmem


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


def _resolve_tdm_num_programs(dtype: torch.dtype, block_h: int, requested: int) -> int:
    del dtype, block_h
    max_programs = int(os.environ.get("FLASHMOE_TDM_MAX_PROGRAMS", "16"))
    resolved = min(requested, max_programs)
    # gfx1250 currently has a reproducible phase-barrier stall at four programs.
    if 2 < resolved < 8:
        return 2
    return resolved


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
            h_abs = zero_i
            while h_abs < H:
                t = gl.load(tokens + token_id * H + h_abs).to(gl.float32)
                w = gl.load(gate_weights + h_abs * E + expert_id).to(gl.float32)
                logit += t * w
                h_abs += 1
            row_max = gl.maximum(row_max, logit)

    top_idx = zero_i
    top_prob = zero_f - 1.0
    for expert_id in gl.static_range(0, BLOCK_E):
        if expert_id < E:
            expert_id_t = zero_i + expert_id
            logit = zero_f
            h_abs = zero_i
            while h_abs < H:
                t = gl.load(tokens + token_id * H + h_abs).to(gl.float32)
                w = gl.load(gate_weights + h_abs * E + expert_id).to(gl.float32)
                logit += t * w
                h_abs += 1
            prob = gl.exp(logit - row_max)
            better = prob > top_prob
            top_prob = gl.where(better, prob, top_prob)
            top_idx = gl.where(better, expert_id_t, top_idx)

    route_vals = gl.load(bias_down + top_idx * H + offs_h, mask=h_mask, other=0.0).to(gl.float32)
    i_abs = zero_i
    while i_abs < I:
        up_acc = gl.load(bias_up + top_idx * I + i_abs).to(gl.float32)
        h_abs = zero_i
        while h_abs < H:
            t = gl.load(tokens + token_id * H + h_abs).to(gl.float32)
            wu = gl.load(expert_up + top_idx * H * I + h_abs * I + i_abs).to(gl.float32)
            up_acc += t * wu
            h_abs += 1
        wd = gl.load(expert_down + top_idx * I * H + i_abs * H + offs_h, mask=h_mask, other=0.0).to(gl.float32)
        route_vals += up_acc * wd
        i_abs += 1
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



@gluon.aggregate
class _WsPhaseCounter:
    iteration: gl.tensor
    num_barriers: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, iteration, num_barriers):
        self.iteration = iteration
        self.num_barriers = gl.constexpr(num_barriers)

    @gluon.jit
    def create(iteration, num_barriers: gl.constexpr):
        return _WsPhaseCounter(gl.to_tensor(iteration), num_barriers)

    @gluon.jit
    def phase(self):
        return (self.iteration // self.num_barriers) & 1

    @gluon.must_use_result
    @gluon.jit
    def next(self):
        return _WsPhaseCounter(self.iteration + 1, self.num_barriers)


@gluon.jit
def _init_ws_load_mbarriers(
    load_empty_bars,
    load_ready_bars,
    NUM_BUFFERS: gl.constexpr,
    PRODUCER_WARPS: gl.constexpr,
    COMPUTE_WARPS: gl.constexpr,
    LOAD_READY_TDM_OPS: gl.constexpr,
):
    for i in gl.static_range(0, NUM_BUFFERS):
        mbarrier.init(load_empty_bars.index(i), count=COMPUTE_WARPS * 32)
        mbarrier.init(load_ready_bars.index(i), count=PRODUCER_WARPS * LOAD_READY_TDM_OPS)


@gluon.jit
def _init_ws_mbarriers(
    load_empty_bars,
    load_ready_bars,
    acc_empty_bars,
    acc_ready_bars,
    NUM_BUFFERS: gl.constexpr,
    NUM_ACC_BUFFERS: gl.constexpr,
    PRODUCER_WARPS: gl.constexpr,
    COMPUTE_WARPS: gl.constexpr,
    EPILOGUE_WARPS: gl.constexpr,
    ACC_EMPTY_TDM: gl.constexpr,
    LOAD_READY_TDM_OPS: gl.constexpr,
):
    _init_ws_load_mbarriers(
        load_empty_bars,
        load_ready_bars,
        NUM_BUFFERS,
        PRODUCER_WARPS,
        COMPUTE_WARPS,
        LOAD_READY_TDM_OPS,
    )
    for i in gl.static_range(0, NUM_ACC_BUFFERS):
        acc_empty_count: gl.constexpr = EPILOGUE_WARPS if ACC_EMPTY_TDM else EPILOGUE_WARPS * 32
        mbarrier.init(acc_empty_bars.index(i), count=acc_empty_count)
        mbarrier.init(acc_ready_bars.index(i), count=COMPUTE_WARPS * 32)


@gluon.jit
def _gemm0_ws_producer(
    tokens,
    expert_up,
    expert_up_v,
    route_tokens,
    expert_counts,
    x_buffer,
    w_buffer,
    xv_buffer,
    wv_buffer,
    load_empty_bars,
    load_ready_bars,
    v_load_empty_bars,
    v_load_ready_bars,
    expert_idx,
    base_slot,
    base_i,
    S: gl.constexpr,
    H: gl.constexpr,
    I: gl.constexpr,
    EC: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    PRODUCER_WARPS: gl.constexpr,
    GATED: gl.constexpr,
    shared_a_layout: gl.constexpr,
    shared_b_layout: gl.constexpr,
):
    route_layout: gl.constexpr = _route_index_layout(BLOCK_M, PRODUCER_WARPS)
    route_offs = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, route_layout))
    expert_count = gl.load(expert_counts + expert_idx)
    active_rows = base_slot + route_offs < gl.minimum(expert_count, expert_count * 0 + EC)
    safe_route_rows = gl.minimum(base_slot + route_offs, route_offs * 0 + EC - 1)
    gathered_tokens = gl.load(
        route_tokens + expert_idx * EC + safe_route_rows,
        mask=active_rows,
        other=0,
    ).to(gl.int32)
    empty_counter = _WsPhaseCounter.create(NUM_BUFFERS, NUM_BUFFERS)
    h_tile = base_slot * 0
    while h_tile < H // BLOCK_N:
        buffer_idx = h_tile % NUM_BUFFERS
        empty_bar = load_empty_bars.index(buffer_idx)
        ready_bar = load_ready_bars.index(buffer_idx)
        mbarrier.wait(empty_bar, empty_counter.phase())
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
        tdm.async_gather(x_desc, gathered_tokens, x_buffer.index(buffer_idx))
        tdm.async_load(w_desc, [0, 0], w_buffer.index(buffer_idx), mbarrier=ready_bar)
        empty_counter = empty_counter.next()
        h_tile += 1

    if GATED:
        v_empty_counter = _WsPhaseCounter.create(NUM_BUFFERS, NUM_BUFFERS)
        h_tile_v = base_slot * 0
        while h_tile_v < H // BLOCK_N:
            buffer_idx_v = h_tile_v % NUM_BUFFERS
            empty_bar_v = v_load_empty_bars.index(buffer_idx_v)
            ready_bar_v = v_load_ready_bars.index(buffer_idx_v)
            mbarrier.wait(empty_bar_v, v_empty_counter.phase())
            xv_desc = tdm.make_tensor_descriptor(
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
            tdm.async_gather(xv_desc, gathered_tokens, xv_buffer.index(buffer_idx_v))
            tdm.async_load(wv_desc, [0, 0], wv_buffer.index(buffer_idx_v), mbarrier=ready_bar_v)
            v_empty_counter = v_empty_counter.next()
            h_tile_v += 1


@gluon.jit
def _gemm0_ws_compute(
    bias_up,
    bias_up_v,
    x_buffer,
    w_buffer,
    xv_buffer,
    wv_buffer,
    hidden_buffer,
    load_empty_bars,
    load_ready_bars,
    v_load_empty_bars,
    v_load_ready_bars,
    acc_empty_bars,
    acc_ready_bars,
    expert_idx,
    base_i,
    H: gl.constexpr,
    I: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    ACTIVATION: gl.constexpr,
    GATED: gl.constexpr,
    wmma_layout: gl.constexpr,
):
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=wmma_layout, k_width=8)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=wmma_layout, k_width=8)
    ready_counter = _WsPhaseCounter.create(0, NUM_BUFFERS)
    acc_empty_bar = acc_empty_bars.index(0)
    acc_ready_bar = acc_ready_bars.index(0)
    mbarrier.wait(acc_empty_bar, phase=1)

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=wmma_layout)
    h_tile = expert_idx * 0
    while h_tile < H // BLOCK_N:
        buffer_idx = h_tile % NUM_BUFFERS
        ready_bar = load_ready_bars.index(buffer_idx)
        empty_bar = load_empty_bars.index(buffer_idx)
        mbarrier.wait(ready_bar, ready_counter.phase())
        x_frag = x_buffer.index(buffer_idx).load(layout=dot_a)
        w_frag = w_buffer.index(buffer_idx).load(layout=dot_b)
        acc = gl.amd.gfx1250.wmma(x_frag, w_frag, acc)
        mbarrier.arrive(empty_bar, count=1)
        ready_counter = ready_counter.next()
        h_tile += 1

    bias_layout: gl.constexpr = gl.SliceLayout(0, wmma_layout)
    offs_i = base_i + gl.arange(0, BLOCK_N, layout=bias_layout)
    bias = gl.load(bias_up + expert_idx * I + offs_i).to(gl.float32)
    hidden_acc = _apply_activation(acc + gl.convert_layout(gl.expand_dims(bias, 0), wmma_layout), ACTIVATION)

    if GATED:
        v_ready_counter = _WsPhaseCounter.create(0, NUM_BUFFERS)
        acc_v = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=wmma_layout)
        h_tile_v = expert_idx * 0
        while h_tile_v < H // BLOCK_N:
            buffer_idx_v = h_tile_v % NUM_BUFFERS
            ready_bar_v = v_load_ready_bars.index(buffer_idx_v)
            empty_bar_v = v_load_empty_bars.index(buffer_idx_v)
            mbarrier.wait(ready_bar_v, v_ready_counter.phase())
            xv_frag = xv_buffer.index(buffer_idx_v).load(layout=dot_a)
            wv_frag = wv_buffer.index(buffer_idx_v).load(layout=dot_b)
            acc_v = gl.amd.gfx1250.wmma(xv_frag, wv_frag, acc_v)
            mbarrier.arrive(empty_bar_v, count=1)
            v_ready_counter = v_ready_counter.next()
            h_tile_v += 1
        bias_v = gl.load(bias_up_v + expert_idx * I + offs_i).to(gl.float32)
        acc_v += gl.convert_layout(gl.expand_dims(bias_v, 0), wmma_layout)
        hidden_acc *= acc_v

    hidden_dtype: gl.constexpr = hidden_buffer.dtype
    hidden_buffer.index(0).store(hidden_acc.to(hidden_dtype))
    _wait_dscnt0()
    mbarrier.arrive(acc_ready_bar, count=1)


@gluon.jit
def _gemm0_ws_epilogue(
    hidden,
    hidden_buffer,
    acc_empty_bars,
    acc_ready_bars,
    expert_idx,
    base_slot,
    base_i,
    EC: gl.constexpr,
    I: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    shared_hidden_layout: gl.constexpr,
):
    acc_ready_bar = acc_ready_bars.index(0)
    acc_empty_bar = acc_empty_bars.index(0)
    mbarrier.wait(acc_ready_bar, phase=0)
    hidden_desc = tdm.make_tensor_descriptor(
        base=hidden + expert_idx * EC * I,
        shape=(EC, I),
        strides=(I, 1),
        block_shape=(BLOCK_M, BLOCK_N),
        layout=shared_hidden_layout,
    )
    _wait_dscnt0()
    tdm.async_store(hidden_desc, [base_slot, base_i], hidden_buffer.index(0), mbarrier=acc_empty_bar)
    tdm.async_wait(0)


@gluon.jit
def _gemm1_ws_producer(
    hidden,
    expert_down,
    h_buffer,
    down_buffer,
    load_empty_bars,
    load_ready_bars,
    expert_idx,
    base_slot,
    base_h,
    EC: gl.constexpr,
    I: gl.constexpr,
    H: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    shared_h_layout: gl.constexpr,
    shared_down_layout: gl.constexpr,
):
    empty_counter = _WsPhaseCounter.create(NUM_BUFFERS, NUM_BUFFERS)
    hidden_desc = tdm.make_tensor_descriptor(
        base=hidden + expert_idx * EC * I,
        shape=(EC, I),
        strides=(I, 1),
        block_shape=(BLOCK_M, BLOCK_N),
        layout=shared_h_layout,
    )
    down_desc = tdm.make_tensor_descriptor(
        base=expert_down + expert_idx * I * H,
        shape=(I, H),
        strides=(H, 1),
        block_shape=(BLOCK_N, BLOCK_N),
        layout=shared_down_layout,
    )
    zero_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0], [])
    i_tile = base_slot * 0
    while i_tile < I // BLOCK_N:
        buffer_idx = i_tile % NUM_BUFFERS
        empty_bar = load_empty_bars.index(buffer_idx)
        ready_bar = load_ready_bars.index(buffer_idx)
        mbarrier.wait(empty_bar, empty_counter.phase())
        h_buffer.index(buffer_idx).store(
            gl.full((BLOCK_M, BLOCK_N), 0.0, hidden.dtype.element_ty, layout=zero_layout)
        )
        _wait_dscnt0()
        tdm.async_load(hidden_desc, [base_slot, i_tile * BLOCK_N], h_buffer.index(buffer_idx))
        tdm.async_load(down_desc, [i_tile * BLOCK_N, base_h], down_buffer.index(buffer_idx), mbarrier=ready_bar)
        empty_counter = empty_counter.next()
        i_tile += 1


@gluon.jit
def _gemm1_ws_compute(
    h_buffer,
    down_buffer,
    acc_buffer,
    load_empty_bars,
    load_ready_bars,
    acc_empty_bars,
    acc_ready_bars,
    I: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    wmma_layout: gl.constexpr,
):
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=wmma_layout, k_width=8)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=wmma_layout, k_width=8)
    ready_counter = _WsPhaseCounter.create(0, NUM_BUFFERS)
    acc_empty_bar = acc_empty_bars.index(0)
    acc_ready_bar = acc_ready_bars.index(0)
    mbarrier.wait(acc_empty_bar, phase=1)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=wmma_layout)
    i_tile = gl.program_id(0) * 0
    while i_tile < I // BLOCK_N:
        buffer_idx = i_tile % NUM_BUFFERS
        ready_bar = load_ready_bars.index(buffer_idx)
        empty_bar = load_empty_bars.index(buffer_idx)
        mbarrier.wait(ready_bar, ready_counter.phase())
        h_frag = h_buffer.index(buffer_idx).load(layout=dot_a)
        down_frag = down_buffer.index(buffer_idx).load(layout=dot_b)
        acc = gl.amd.gfx1250.wmma(h_frag, down_frag, acc)
        mbarrier.arrive(empty_bar, count=1)
        ready_counter = ready_counter.next()
        i_tile += 1
    acc_buffer.index(0).store(acc)
    _wait_dscnt0()
    mbarrier.arrive(acc_ready_bar, count=1)


@gluon.jit
def _gemm1_ws_epilogue(
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
    EC: gl.constexpr,
    H: gl.constexpr,
    TOP_K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    EPILOGUE_WARPS: gl.constexpr,
    wmma_layout: gl.constexpr,
):
    acc_ready_bar = acc_ready_bars.index(0)
    acc_empty_bar = acc_empty_bars.index(0)
    mbarrier.wait(acc_ready_bar, phase=0)
    acc = acc_buffer.index(0).load(layout=wmma_layout)

    bias_layout: gl.constexpr = gl.SliceLayout(0, wmma_layout)
    offs_out = base_h + gl.arange(0, BLOCK_N, layout=bias_layout)
    bias = gl.load(bias_down + expert_idx * H + offs_out).to(gl.float32)
    acc += gl.convert_layout(gl.expand_dims(bias, 0), wmma_layout)

    row_layout: gl.constexpr = gl.SliceLayout(1, wmma_layout)
    route_offs = gl.arange(0, BLOCK_M, layout=row_layout)
    route_rows = base_slot + route_offs
    expert_count = gl.load(expert_counts + expert_idx)
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
    active_rows = gl.convert_layout(active_route_rows, row_layout)
    token_ids = gl.convert_layout(token_ids_route, row_layout)
    probs = gl.convert_layout(probs_route, row_layout)
    token_ids_2d = gl.expand_dims(token_ids, 1)
    cols_2d = gl.expand_dims(offs_out, 0)
    output_offsets = gl.convert_layout(token_ids_2d * H + cols_2d, wmma_layout)
    probs_2d = gl.convert_layout(gl.expand_dims(probs, 1), wmma_layout)
    active_2d = gl.convert_layout(gl.expand_dims(active_rows, 1), wmma_layout)
    if TOP_K == 1:
        gl.store(output + output_offsets, acc * probs_2d, mask=active_2d)
    else:
        gl.atomic_add(
            output + output_offsets,
            acc * probs_2d,
            sem="release",
            scope="gpu",
            mask=active_2d,
        )
    mbarrier.arrive(acc_empty_bar, count=1)



@gluon.jit
def _gemm0_channel_ws_producer(
    dispatch_tokens,
    local_expert_up,
    local_expert_up_v,
    x_buffer,
    w_buffer,
    xv_buffer,
    wv_buffer,
    load_empty_bars,
    load_ready_bars,
    v_load_empty_bars,
    v_load_ready_bars,
    count_idx,
    local_expert,
    base_slot,
    base_i,
    H: gl.constexpr,
    I: gl.constexpr,
    EC: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    PRODUCER_WARPS: gl.constexpr,
    GATED: gl.constexpr,
    shared_a_layout: gl.constexpr,
    shared_b_layout: gl.constexpr,
):
    empty_counter = _WsPhaseCounter.create(NUM_BUFFERS, NUM_BUFFERS)
    h_tile = base_slot * 0
    while h_tile < H // BLOCK_N:
        buffer_idx = h_tile % NUM_BUFFERS
        empty_bar = load_empty_bars.index(buffer_idx)
        ready_bar = load_ready_bars.index(buffer_idx)
        mbarrier.wait(empty_bar, empty_counter.phase())
        x_desc = tdm.make_tensor_descriptor(
            base=dispatch_tokens + count_idx * EC * H + h_tile * BLOCK_N,
            shape=(EC, BLOCK_N),
            strides=(H, 1),
            block_shape=(BLOCK_M, BLOCK_N),
            layout=shared_a_layout,
        )
        w_desc = tdm.make_tensor_descriptor(
            base=local_expert_up + local_expert * H * I + (h_tile * BLOCK_N) * I + base_i,
            shape=(BLOCK_N, BLOCK_N),
            strides=(I, 1),
            block_shape=(BLOCK_N, BLOCK_N),
            layout=shared_b_layout,
        )
        tdm.async_load(x_desc, [base_slot, 0], x_buffer.index(buffer_idx))
        tdm.async_load(w_desc, [0, 0], w_buffer.index(buffer_idx), mbarrier=ready_bar)
        empty_counter = empty_counter.next()
        h_tile += 1

    if GATED:
        v_empty_counter = _WsPhaseCounter.create(NUM_BUFFERS, NUM_BUFFERS)
        h_tile_v = base_slot * 0
        while h_tile_v < H // BLOCK_N:
            buffer_idx_v = h_tile_v % NUM_BUFFERS
            empty_bar_v = v_load_empty_bars.index(buffer_idx_v)
            ready_bar_v = v_load_ready_bars.index(buffer_idx_v)
            mbarrier.wait(empty_bar_v, v_empty_counter.phase())
            xv_desc = tdm.make_tensor_descriptor(
                base=dispatch_tokens + count_idx * EC * H + h_tile_v * BLOCK_N,
                shape=(EC, BLOCK_N),
                strides=(H, 1),
                block_shape=(BLOCK_M, BLOCK_N),
                layout=shared_a_layout,
            )
            wv_desc = tdm.make_tensor_descriptor(
                base=local_expert_up_v + local_expert * H * I + (h_tile_v * BLOCK_N) * I + base_i,
                shape=(BLOCK_N, BLOCK_N),
                strides=(I, 1),
                block_shape=(BLOCK_N, BLOCK_N),
                layout=shared_b_layout,
            )
            tdm.async_load(xv_desc, [base_slot, 0], xv_buffer.index(buffer_idx_v))
            tdm.async_load(wv_desc, [0, 0], wv_buffer.index(buffer_idx_v), mbarrier=ready_bar_v)
            v_empty_counter = v_empty_counter.next()
            h_tile_v += 1


@gluon.jit
def _gemm1_channel_ws_producer(
    hidden,
    local_expert_down,
    h_buffer,
    down_buffer,
    load_empty_bars,
    load_ready_bars,
    hidden_channel,
    local_expert,
    base_slot,
    base_h,
    EC: gl.constexpr,
    I: gl.constexpr,
    H: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    shared_h_layout: gl.constexpr,
    shared_down_layout: gl.constexpr,
):
    empty_counter = _WsPhaseCounter.create(NUM_BUFFERS, NUM_BUFFERS)
    hidden_desc = tdm.make_tensor_descriptor(
        base=hidden + hidden_channel * EC * I,
        shape=(EC, I),
        strides=(I, 1),
        block_shape=(BLOCK_M, BLOCK_N),
        layout=shared_h_layout,
    )
    down_desc = tdm.make_tensor_descriptor(
        base=local_expert_down + local_expert * I * H,
        shape=(I, H),
        strides=(H, 1),
        block_shape=(BLOCK_N, BLOCK_N),
        layout=shared_down_layout,
    )
    zero_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0], [])
    i_tile = base_slot * 0
    while i_tile < I // BLOCK_N:
        buffer_idx = i_tile % NUM_BUFFERS
        empty_bar = load_empty_bars.index(buffer_idx)
        ready_bar = load_ready_bars.index(buffer_idx)
        mbarrier.wait(empty_bar, empty_counter.phase())
        h_buffer.index(buffer_idx).store(
            gl.full((BLOCK_M, BLOCK_N), 0.0, hidden.dtype.element_ty, layout=zero_layout)
        )
        _wait_dscnt0()
        tdm.async_load(hidden_desc, [base_slot, i_tile * BLOCK_N], h_buffer.index(buffer_idx))
        tdm.async_load(down_desc, [i_tile * BLOCK_N, base_h], down_buffer.index(buffer_idx), mbarrier=ready_bar)
        empty_counter = empty_counter.next()
        i_tile += 1


@gluon.jit
def _gemm1_channel_result_epilogue(
    dispatch_counts,
    bias_down,
    result_values,
    acc_buffer,
    acc_empty_bars,
    acc_ready_bars,
    count_idx,
    local_expert,
    base_slot,
    base_h,
    EC: gl.constexpr,
    H: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    EPILOGUE_WARPS: gl.constexpr,
    wmma_layout: gl.constexpr,
):
    acc_ready_bar = acc_ready_bars.index(0)
    acc_empty_bar = acc_empty_bars.index(0)
    mbarrier.wait(acc_ready_bar, phase=0)
    acc = acc_buffer.index(0).load(layout=wmma_layout)

    bias_layout: gl.constexpr = gl.SliceLayout(0, wmma_layout)
    offs_out = base_h + gl.arange(0, BLOCK_N, layout=bias_layout)
    bias = gl.load(bias_down + local_expert * H + offs_out).to(gl.float32)
    acc += gl.convert_layout(gl.expand_dims(bias, 0), wmma_layout)

    row_layout: gl.constexpr = gl.SliceLayout(1, wmma_layout)
    route_offs = gl.arange(0, BLOCK_M, layout=row_layout)
    route_rows = base_slot + route_offs
    channel_count = gl.load(dispatch_counts + count_idx)
    active_route_rows = route_rows < gl.minimum(channel_count, channel_count * 0 + EC)
    active_rows = gl.convert_layout(active_route_rows, row_layout)
    rows_2d = gl.expand_dims(route_rows, 1)
    cols_2d = gl.expand_dims(offs_out, 0)
    result_offsets = gl.convert_layout((count_idx * EC + rows_2d) * H + cols_2d, wmma_layout)
    active_2d = gl.convert_layout(gl.expand_dims(active_rows, 1), wmma_layout)
    gl.store(result_values + result_offsets, acc.to(result_values.dtype.element_ty), mask=active_2d)
    gl.atomic_add(dispatch_counts + count_idx, 0, sem="release", scope="gpu")
    mbarrier.arrive(acc_empty_bar, count=1)


@gluon.jit
def _scheduler_reset(processor_ready, processor_mailboxes, NUM_PROGRAMS: gl.constexpr):
    pid = gl.program_id(0)
    if pid == 0:
        for program_id in gl.static_range(0, NUM_PROGRAMS):
            gl.atomic_xchg(processor_ready + program_id, 0, sem="release", scope="gpu")
            gl.atomic_xchg(processor_mailboxes + program_id, -1, sem="release", scope="gpu")


@gluon.jit
def _scheduler_reset_queue(
    processor_ready,
    processor_mailboxes,
    processor_seen,
    NUM_PROGRAMS: gl.constexpr,
    ROLE_COUNT: gl.constexpr,
):
    pid = gl.program_id(0)
    if pid == 0:
        for program_id in gl.static_range(0, NUM_PROGRAMS):
            gl.atomic_xchg(processor_ready + program_id, 0, sem="release", scope="gpu")
            gl.atomic_xchg(processor_mailboxes + program_id, 0, sem="release", scope="gpu")
        for seen_id in gl.static_range(0, NUM_PROGRAMS * ROLE_COUNT):
            gl.store(processor_seen + seen_id, 0)


@gluon.jit
def _worker_reset_queue_state(
    processor_mailboxes,
    processor_seen,
    NUM_PROGRAMS: gl.constexpr,
    ROLE_COUNT: gl.constexpr,
):
    pid = gl.program_id(0)
    if pid == 0:
        for program_id in gl.static_range(0, NUM_PROGRAMS):
            gl.atomic_xchg(processor_mailboxes + program_id, 0, sem="release", scope="gpu")
        for seen_id in gl.static_range(0, NUM_PROGRAMS * ROLE_COUNT):
            gl.store(processor_seen + seen_id, 0)


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
def _scheduler_run_queue(
    processor_ready,
    processor_mailboxes,
    task_queue,
    task_tail,
    debug_state,
    DEBUG: gl.constexpr,
    DEBUG_SCHED: gl.constexpr,
    TOTAL_TASKS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
):
    pid = gl.program_id(0)
    if pid == 0:
        next_slot = pid * 0
        stopped = pid * 0
        while stopped < NUM_PROGRAMS - 1:
            made_progress = pid * 0
            if DEBUG_SCHED:
                gl.store(debug_state + 16, stopped)
                gl.store(debug_state + 17, next_slot)
            for program_id in gl.static_range(1, NUM_PROGRAMS):
                ready = gl.atomic_xchg(
                    processor_ready + program_id,
                    0,
                    sem="acquire",
                    scope="gpu",
                )
                if ready != 0:
                    tail = gl.atomic_add(task_tail, 0, sem="acquire", scope="gpu")
                    if DEBUG_SCHED:
                        gl.store(debug_state + 18, tail)
                        gl.store(debug_state + 19, program_id)
                    if next_slot < tail:
                        task = gl.atomic_add(task_queue + next_slot, 0, sem="acquire", scope="gpu")
                        if DEBUG_SCHED:
                            gl.store(debug_state + 20, task)
                        while task < 0:
                            _gpu_relax()
                            task = gl.atomic_add(task_queue + next_slot, 0, sem="acquire", scope="gpu")
                        next_slot += 1
                        gl.atomic_xchg(
                            processor_mailboxes + program_id,
                            task + 1,
                            sem="release",
                            scope="gpu",
                        )
                        made_progress += 1
                    else:
                        if next_slot >= TOTAL_TASKS:
                            if DEBUG_SCHED:
                                gl.store(debug_state + 21, program_id)
                            gl.atomic_xchg(
                                processor_mailboxes + program_id,
                                TOTAL_TASKS + 1,
                                sem="release",
                                scope="gpu",
                            )
                            stopped += 1
                            made_progress += 1
                        else:
                            gl.atomic_xchg(processor_ready + program_id, 1, sem="release", scope="gpu")
            if made_progress == 0:
                _gpu_relax()


@gluon.jit
def _init_dynamic_task_queue(
    task_queue,
    task_head,
    task_tail,
    gemm0_pending,
    tile_sync,
    GEMM0_TASKS: gl.constexpr,
    TOTAL_TASKS: gl.constexpr,
    SYNC_SLOTS: gl.constexpr,
):
    pid = gl.program_id(0)
    if pid == 0:
        slot = pid * 0 + GEMM0_TASKS
        while slot < TOTAL_TASKS:
            gl.store(task_queue + slot, -1)
            slot += 1
        init_task = pid * 0
        while init_task < GEMM0_TASKS:
            gl.store(task_queue + init_task, init_task)
            init_task += 1
        sync_slot = pid * 0
        while sync_slot < SYNC_SLOTS:
            gl.store(tile_sync + sync_slot, 0)
            sync_slot += 1
        _wait_storecnt0()
        gl.atomic_xchg(task_head, 0, sem="release", scope="gpu")
        gl.atomic_xchg(task_tail, GEMM0_TASKS, sem="release", scope="gpu")
        gl.atomic_xchg(gemm0_pending, GEMM0_TASKS, sem="release", scope="gpu")


@gluon.jit
def _enqueue_task(task_queue, task_tail, task):
    slot = gl.atomic_add(task_tail, 1, sem="relaxed", scope="gpu")
    gl.atomic_xchg(task_queue + slot, task, sem="release", scope="gpu")


@gluon.jit
def _enqueue_compute_task(task_queue, task_tail, task_bound, task):
    gl.atomic_add(task_bound, 1, sem="release", scope="gpu")
    _enqueue_task(task_queue, task_tail, task)


@gluon.jit
def _enqueue_publish_task(publish_queue, publish_tail, task):
    slot = gl.atomic_add(publish_tail, 1, sem="relaxed", scope="gpu")
    gl.atomic_xchg(publish_queue + slot, task, sem="release", scope="gpu")


@gluon.jit
def _gpu_relax():
    gl.inline_asm_elementwise(
        "s_sleep 1\ns_mov_b32 $0, 0",
        constraints="=s",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _wait_dscnt0():
    gl.inline_asm_elementwise(
        "s_wait_dscnt 0\ns_mov_b32 $0, 0",
        constraints="=s,~{memory}",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _wait_storecnt0():
    gl.inline_asm_elementwise(
        "s_wait_storecnt 0\ns_mov_b32 $0, 0",
        constraints="=s,~{memory}",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _wait_loadcnt0():
    gl.inline_asm_elementwise(
        "s_wait_loadcnt 0\ns_mov_b32 $0, 0",
        constraints="=s,~{memory}",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _load_u64_acquire(ptr):
    return rocshmem.signal_fetch_wave(ptr)


@gluon.jit
def _signal_epoch_base(epoch_u64):
    return epoch_u64 * gl.full((), 1048576, gl.uint64)


@gluon.jit
def _signal_payload(epoch_u64, count):
    count_u64 = count.to(gl.uint64)
    return _signal_epoch_base(epoch_u64) + count_u64


@gluon.jit
def _signal_count(signal, epoch_u64):
    return (signal - _signal_epoch_base(epoch_u64)).to(gl.int32)


@gluon.jit
def _scheduler_run_dynamic_queue(
    processor_ready,
    processor_mailboxes,
    task_queue,
    task_tail,
    task_bound,
    tasks_done,
    os_done,
    debug_state,
    STOP_TASK: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    FIRST_COMPUTE_PROGRAM: gl.constexpr,
    DEBUG: gl.constexpr,
):
    next_slot = gl.program_id(0) * 0
    stopped = gl.program_id(0) * 0
    while stopped < NUM_PROGRAMS - FIRST_COMPUTE_PROGRAM:
        if DEBUG:
            gl.store(debug_state + 32, stopped)
            gl.store(debug_state + 33, next_slot)
        for program_id in gl.static_range(FIRST_COMPUTE_PROGRAM, NUM_PROGRAMS):
            ready = gl.atomic_xchg(
                processor_ready + program_id,
                0,
                sem="acquire",
                scope="gpu",
            )
            if ready != 0:
                tail = gl.atomic_add(task_tail, 0, sem="acquire", scope="gpu")
                if next_slot < tail:
                    task = gl.atomic_add(task_queue + next_slot, 0, sem="acquire", scope="gpu")
                    if task >= 0:
                        if DEBUG:
                            gl.store(debug_state + 34, task)
                        next_slot += 1
                        gl.atomic_xchg(
                            processor_mailboxes + program_id,
                            task + 1,
                            sem="release",
                            scope="gpu",
                        )
                    else:
                        gl.atomic_xchg(processor_ready + program_id, 1, sem="release", scope="gpu")
                else:
                    bound = gl.atomic_add(task_bound, 0, sem="acquire", scope="gpu")
                    done = gl.atomic_add(tasks_done, 0, sem="acquire", scope="gpu")
                    os_finished = gl.atomic_add(os_done, 0, sem="acquire", scope="gpu") != 0
                    if DEBUG:
                        gl.store(debug_state + 35, bound)
                        gl.store(debug_state + 36, done)
                    if os_finished & (done >= bound):
                        gl.atomic_xchg(
                            processor_mailboxes + program_id,
                            STOP_TASK + 1,
                            sem="release",
                            scope="gpu",
                        )
                        stopped += 1
                    else:
                        gl.atomic_xchg(processor_ready + program_id, 1, sem="release", scope="gpu")
        _gpu_relax()


@gluon.jit
def _init_rocshmem_dynamic_state(
    dispatch_counts,
    result_counts,
    task_queue,
    task_tail,
    publish_queue,
    publish_tail,
    tile_sync,
    compute_done,
    dispatch_seen,
    result_seen,
    task_bound,
    tasks_done,
    os_done,
    dispatch_done,
    processor_ready,
    processor_mailboxes,
    processor_seen,
    barriers,
    my_pe,
    TOTAL_COMPUTE_TASKS: gl.constexpr,
    PUBLISH_TASKS: gl.constexpr,
    SYNC_SLOTS: gl.constexpr,
    ALL_CHANNELS: gl.constexpr,
    LOCAL_CHANNELS: gl.constexpr,
    WORLD: gl.constexpr,
    NLX: gl.constexpr,
    BARRIERS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
):
    pid = gl.program_id(0)
    if pid == 0:
        slot = pid * 0
        while slot < TOTAL_COMPUTE_TASKS:
            gl.store(task_queue + slot, -1)
            slot += 1
        pslot = pid * 0
        while pslot < PUBLISH_TASKS:
            gl.store(publish_queue + pslot, -1)
            pslot += 1
        sync_slot = pid * 0
        while sync_slot < SYNC_SLOTS:
            gl.store(tile_sync + sync_slot, 0)
            sync_slot += 1
        channel = pid * 0
        while channel < ALL_CHANNELS:
            gl.store(compute_done + channel, 0)
            channel += 1
        for peer_static in gl.static_range(0, WORLD):
            peer = my_pe * 0 + peer_static
            for lx_static in gl.static_range(0, NLX):
                dispatch_idx = (peer * WORLD + my_pe) * NLX + lx_static
                result_idx = (my_pe * WORLD + peer) * NLX + lx_static
                gl.store(dispatch_counts + dispatch_idx, 0)
                gl.store(result_counts + result_idx, 0)
        local_channel = pid * 0
        while local_channel < LOCAL_CHANNELS:
            gl.store(dispatch_seen + local_channel, 0)
            gl.store(result_seen + local_channel, 0)
            local_channel += 1
        for program_id in gl.static_range(0, NUM_PROGRAMS):
            gl.atomic_xchg(processor_ready + program_id, 0, sem="release", scope="gpu")
            gl.atomic_xchg(processor_mailboxes + program_id, 0, sem="release", scope="gpu")
            gl.store(processor_seen + program_id, 0)
        for phase in gl.static_range(0, BARRIERS):
            gl.store(barriers + phase, 0)
        gl.atomic_xchg(task_tail, 0, sem="release", scope="gpu")
        gl.atomic_xchg(publish_tail, 0, sem="release", scope="gpu")
        gl.atomic_xchg(task_bound, 0, sem="release", scope="gpu")
        gl.atomic_xchg(tasks_done, 0, sem="release", scope="gpu")
        gl.atomic_xchg(os_done, 0, sem="release", scope="gpu")
        gl.atomic_xchg(dispatch_done, 0, sem="release", scope="gpu")


@gluon.jit
def _publish_result_channel_wave(
    result_values,
    result_counts,
    result_signals,
    count_idx,
    source_pe,
    my_pe,
    epoch_u64,
    EC: gl.constexpr,
    H: gl.constexpr,
    RESULT_ELEM_BYTES: gl.constexpr,
):
    published_count = gl.load(result_counts + count_idx)
    published_count = gl.minimum(published_count, published_count * 0 + EC)
    payload = _signal_payload(epoch_u64, published_count)
    gl.store(result_counts + count_idx, published_count)
    gl.store(result_signals + count_idx, payload)
    if source_pe != my_pe:
        if published_count == 0:
            rocshmem.signal_op_wave(
                result_signals + count_idx,
                payload,
                rocshmem.ROCSHMEM_SIGNAL_SET,
                source_pe,
            )
            rocshmem.quiet()
        else:
            rocshmem.putmem_wave(
                result_counts + count_idx,
                result_counts + count_idx,
                4,
                source_pe,
            )
            rocshmem.fence()
            rocshmem.putmem_signal_wave(
                result_values + count_idx * EC * H,
                result_values + count_idx * EC * H,
                EC * H * RESULT_ELEM_BYTES,
                result_signals + count_idx,
                payload,
                rocshmem.ROCSHMEM_SIGNAL_SET,
                source_pe,
            )
            rocshmem.quiet()


@gluon.jit
def _rocshmem_os_subscriber_publisher(
    task_queue,
    task_tail,
    publish_queue,
    publish_tail,
    dispatch_counts,
    dispatch_signals,
    result_values,
    result_counts,
    result_signals,
    dispatch_seen,
    result_seen,
    task_bound,
    tasks_done,
    os_done,
    debug_state,
    launch_epoch,
    my_pe,
    WORLD: gl.constexpr,
    NLX: gl.constexpr,
    EC: gl.constexpr,
    H: gl.constexpr,
    I: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    RESULT_ELEM_BYTES: gl.constexpr,
    DEBUG: gl.constexpr,
):
    epoch_u64 = gl.full((), launch_epoch, gl.uint64)
    epoch_base = _signal_epoch_base(epoch_u64)
    route_blocks: gl.constexpr = (EC + BLOCK_M - 1) // BLOCK_M
    h_tiles: gl.constexpr = H // BLOCK_N
    i_tiles: gl.constexpr = I // BLOCK_N
    total_channels: gl.constexpr = WORLD * NLX
    gemm0_tasks: gl.constexpr = total_channels * route_blocks * i_tiles
    gemm1_tasks: gl.constexpr = total_channels * route_blocks * h_tiles
    combine_base: gl.constexpr = gemm0_tasks + gemm1_tasks

    dispatch_seen_count = gl.program_id(0) * 0
    result_seen_count = gl.program_id(0) * 0
    publish_head = gl.program_id(0) * 0
    finished = gl.program_id(0) * 0
    debug_iter = gl.program_id(0) * 0
    while finished == 0:
        if DEBUG:
            debug_iter += 1
            gl.store(debug_state + 0, my_pe)
            gl.store(debug_state + 1, dispatch_seen_count)
            gl.store(debug_state + 2, result_seen_count)
            gl.store(debug_state + 3, publish_head)
        for source_pe_static in gl.static_range(0, WORLD):
            source_pe = my_pe * 0 + source_pe_static
            for lx in gl.static_range(0, NLX):
                seen_idx = source_pe_static * NLX + lx
                if gl.load(dispatch_seen + seen_idx) == 0:
                    count_idx = (my_pe * WORLD + source_pe) * NLX + lx
                    signal = _load_u64_acquire(dispatch_signals + count_idx)
                    if signal >= epoch_base:
                        gl.store(dispatch_seen + seen_idx, 1)
                        dispatch_seen_count += 1
                        channel_count = _signal_count(signal, epoch_u64)
                        channel_count = gl.minimum(channel_count, channel_count * 0 + EC)
                        gl.store(dispatch_counts + count_idx, channel_count)
                        if DEBUG:
                            gl.store(debug_state + 8 + seen_idx, channel_count)
                        if channel_count == 0:
                            gl.store(result_counts + count_idx, 0)
                            _publish_result_channel_wave(
                                result_values,
                                result_counts,
                                result_signals,
                                count_idx,
                                source_pe,
                                my_pe,
                                epoch_u64,
                                EC,
                                H,
                                RESULT_ELEM_BYTES,
                            )
                        else:
                            active_blocks = (channel_count + BLOCK_M - 1) // BLOCK_M
                            hidden_channel = source_pe_static * NLX + lx
                            route_block = gl.program_id(0) * 0
                            while route_block < active_blocks:
                                i_tile = gl.program_id(0) * 0
                                while i_tile < i_tiles:
                                    task = hidden_channel * route_blocks * i_tiles + route_block * i_tiles + i_tile
                                    _enqueue_compute_task(task_queue, task_tail, task_bound, task)
                                    i_tile += 1
                                route_block += 1

        publish_tail_snapshot = gl.atomic_add(publish_tail, 0, sem="acquire", scope="gpu")
        while publish_head < publish_tail_snapshot:
            publish_task = gl.atomic_add(publish_queue + publish_head, 0, sem="acquire", scope="gpu")
            if publish_task >= 0:
                publish_head += 1
                source_pe = publish_task // NLX
                local_expert = publish_task - source_pe * NLX
                count_idx = (my_pe * WORLD + source_pe) * NLX + local_expert
                _publish_result_channel_wave(
                    result_values,
                    result_counts,
                    result_signals,
                    count_idx,
                    source_pe,
                    my_pe,
                    epoch_u64,
                    EC,
                    H,
                    RESULT_ELEM_BYTES,
                )
            else:
                publish_tail_snapshot = publish_head

        for owner_pe_static in gl.static_range(0, WORLD):
            owner_pe = my_pe * 0 + owner_pe_static
            for lx_res in gl.static_range(0, NLX):
                seen_idx_res = owner_pe_static * NLX + lx_res
                if gl.load(result_seen + seen_idx_res) == 0:
                    count_idx_res = (owner_pe * WORLD + my_pe) * NLX + lx_res
                    signal_res = _load_u64_acquire(result_signals + count_idx_res)
                    if signal_res >= epoch_base:
                        gl.store(result_seen + seen_idx_res, 1)
                        result_seen_count += 1
                        result_count = _signal_count(signal_res, epoch_u64)
                        result_count = gl.minimum(result_count, result_count * 0 + EC)
                        gl.store(result_counts + count_idx_res, result_count)
                        if DEBUG:
                            gl.store(debug_state + 16 + seen_idx_res, result_count)
                        if result_count > 0:
                            active_blocks_res = (result_count + BLOCK_M - 1) // BLOCK_M
                            owner_channel = owner_pe_static * NLX + lx_res
                            route_block_res = gl.program_id(0) * 0
                            while route_block_res < active_blocks_res:
                                h_tile_res = gl.program_id(0) * 0
                                while h_tile_res < h_tiles:
                                    task_res = combine_base + owner_channel * route_blocks * h_tiles + route_block_res * h_tiles + h_tile_res
                                    _enqueue_compute_task(task_queue, task_tail, task_bound, task_res)
                                    h_tile_res += 1
                                route_block_res += 1

        dispatch_seen_total = gl.program_id(0) * 0
        result_seen_total = gl.program_id(0) * 0
        for source_pe_done_static in gl.static_range(0, WORLD):
            for lx_done in gl.static_range(0, NLX):
                seen_done_idx = source_pe_done_static * NLX + lx_done
                if gl.load(dispatch_seen + seen_done_idx) != 0:
                    dispatch_seen_total += 1
                if gl.load(result_seen + seen_done_idx) != 0:
                    result_seen_total += 1

        publish_tail_done = gl.atomic_add(publish_tail, 0, sem="acquire", scope="gpu")
        bound = gl.atomic_add(task_bound, 0, sem="acquire", scope="gpu")
        done = gl.atomic_add(tasks_done, 0, sem="acquire", scope="gpu")
        if DEBUG:
            gl.store(debug_state + 1, dispatch_seen_total)
            gl.store(debug_state + 2, result_seen_total)
            gl.store(debug_state + 4, publish_tail_done)
            gl.store(debug_state + 5, bound)
            gl.store(debug_state + 6, done)
            gl.store(debug_state + 7, debug_iter)
        all_dispatch = dispatch_seen_total == total_channels
        all_results = result_seen_total == total_channels
        if all_dispatch & all_results & (publish_head >= publish_tail_done) & (done >= bound):
            gl.atomic_xchg(os_done, 1, sem="release", scope="gpu")
            finished = 1
        if finished == 0:
            _gpu_relax()


@gluon.jit
def _processor_next_task(processor_ready, processor_mailboxes):
    pid = gl.program_id(0)
    gl.atomic_xchg(processor_ready + pid, 1, sem="release", scope="gpu")
    task = gl.atomic_add(processor_mailboxes + pid, 0, sem="acquire", scope="gpu")
    while task < 0:
        _gpu_relax()
        task = gl.atomic_add(processor_mailboxes + pid, 0, sem="acquire", scope="gpu")
    gl.atomic_xchg(processor_mailboxes + pid, -1, sem="relaxed", scope="gpu")
    return task


@gluon.jit
def _processor_next_task_debug(
    processor_ready,
    processor_mailboxes,
    debug_state,
    DEBUG: gl.constexpr,
    DEBUG_SCHED: gl.constexpr,
):
    pid = gl.program_id(0)
    if DEBUG:
        gl.store(debug_state + 32 + pid, -1)
    gl.atomic_xchg(processor_ready + pid, 1, sem="release", scope="gpu")
    task = gl.atomic_add(processor_mailboxes + pid, 0, sem="acquire", scope="gpu")
    while task < 0:
        if DEBUG_SCHED:
            gl.store(debug_state + 40 + pid, task)
        _gpu_relax()
        task = gl.atomic_add(processor_mailboxes + pid, 0, sem="acquire", scope="gpu")
    if DEBUG:
        gl.store(debug_state + 32 + pid, task)
    gl.atomic_xchg(processor_mailboxes + pid, -1, sem="relaxed", scope="gpu")
    return task


@gluon.jit
def _processor_next_task_queue(
    processor_ready,
    processor_mailboxes,
    processor_seen,
    debug_state,
    DEBUG: gl.constexpr,
    DEBUG_SCHED: gl.constexpr,
):
    pid = gl.program_id(0)
    if DEBUG:
        gl.store(debug_state + 32 + pid, -1)
    gl.atomic_xchg(processor_ready + pid, 1, sem="release", scope="gpu")
    last = gl.load(processor_seen + pid)
    signal = gl.atomic_add(processor_mailboxes + pid, 0, sem="acquire", scope="gpu")
    while signal == last:
        if DEBUG_SCHED:
            gl.store(debug_state + 40 + pid, signal)
        _gpu_relax()
        signal = gl.atomic_add(processor_mailboxes + pid, 0, sem="acquire", scope="gpu")
    gl.store(processor_seen + pid, signal)
    task = signal - 1
    if DEBUG:
        gl.store(debug_state + 32 + pid, task)
    return task


@gluon.jit
def _processor_next_task_queue_role(
    processor_ready,
    processor_mailboxes,
    processor_seen,
    debug_state,
    ROLE: gl.constexpr,
    READY_ROLE: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    DEBUG: gl.constexpr,
    DEBUG_SCHED: gl.constexpr,
):
    pid = gl.program_id(0)
    seen_idx: gl.constexpr = ROLE * NUM_PROGRAMS
    if READY_ROLE:
        if DEBUG:
            gl.store(debug_state + 32 + pid, -1)
    if READY_ROLE:
        gl.atomic_xchg(processor_ready + pid, 1, sem="release", scope="gpu")
    last = gl.load(processor_seen + seen_idx + pid)
    signal = gl.atomic_add(processor_mailboxes + pid, 0, sem="acquire", scope="gpu")
    while signal == last:
        if READY_ROLE:
            if DEBUG_SCHED:
                gl.store(debug_state + 40 + pid, signal)
        _gpu_relax()
        signal = gl.atomic_add(processor_mailboxes + pid, 0, sem="acquire", scope="gpu")
    gl.store(processor_seen + seen_idx + pid, signal)
    task = signal - 1
    if READY_ROLE:
        if DEBUG:
            gl.store(debug_state + 32 + pid, task)
    return task


@gluon.jit
def _worker_claim_task_queue(
    task_queue,
    task_head,
    task_tail,
    gemm0_pending,
    processor_mailboxes,
    debug_state,
    DEBUG: gl.constexpr,
    DEBUG_SCHED: gl.constexpr,
    TOTAL_TASKS: gl.constexpr,
):
    pid = gl.program_id(0)
    task = pid * 0 + TOTAL_TASKS
    claimed = pid * 0
    while claimed == 0:
        head = gl.atomic_add(task_head, 0, sem="acquire", scope="gpu")
        tail = gl.atomic_add(task_tail, 0, sem="acquire", scope="gpu")
        if DEBUG_SCHED:
            gl.store(debug_state + 16 + pid, head)
            gl.store(debug_state + 24 + pid, tail)
        if head < tail:
            old = gl.atomic_cas(task_head, head, head + 1, sem="acq_rel", scope="gpu")
            if old == head:
                task = gl.atomic_add(task_queue + head, 0, sem="acquire", scope="gpu")
                while task < 0:
                    _gpu_relax()
                    task = gl.atomic_add(task_queue + head, 0, sem="acquire", scope="gpu")
                claimed = 1
        else:
            pending = gl.atomic_add(gemm0_pending, 0, sem="acquire", scope="gpu")
            if pending == 0:
                task = pid * 0 + TOTAL_TASKS
                claimed = 1
            else:
                _gpu_relax()
    gl.atomic_xchg(processor_mailboxes + pid, task + 1, sem="release", scope="gpu")
    if DEBUG:
        gl.store(debug_state + 32 + pid, task)
    return task


@gluon.jit
def _processor_wait_task_queue_role(
    processor_mailboxes,
    processor_seen,
    debug_state,
    ROLE: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    DEBUG: gl.constexpr,
    DEBUG_SCHED: gl.constexpr,
):
    pid = gl.program_id(0)
    seen_idx: gl.constexpr = ROLE * NUM_PROGRAMS
    last = gl.load(processor_seen + seen_idx + pid)
    signal = gl.atomic_add(processor_mailboxes + pid, 0, sem="acquire", scope="gpu")
    while signal == last:
        if DEBUG_SCHED:
            gl.store(debug_state + 40 + pid, signal)
        _gpu_relax()
        signal = gl.atomic_add(processor_mailboxes + pid, 0, sem="acquire", scope="gpu")
    gl.store(processor_seen + seen_idx + pid, signal)
    task = signal - 1
    if DEBUG:
        gl.store(debug_state + 48 + seen_idx + pid, task)
    return task


@gluon.jit
def _processor_next_task_generation(processor_ready, processor_mailboxes, processor_seen, STOP_TASK: gl.constexpr):
    pid = gl.program_id(0)
    gl.atomic_xchg(processor_ready + pid, 1, sem="release", scope="gpu")
    last = gl.load(processor_seen + pid)
    signal = gl.atomic_add(processor_mailboxes + pid, 0, sem="acquire", scope="gpu")
    while signal == last:
        _gpu_relax()
        signal = gl.atomic_add(processor_mailboxes + pid, 0, sem="acquire", scope="gpu")
    gl.store(processor_seen + pid, signal)
    task = signal - 1
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
        expert_id = pid * 0
        while expert_id < E:
            gl.store(expert_counts + expert_id, 0)
            expert_id += 1
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
                h_abs = zero_i
                while h_abs < H:
                    t = gl.load(tokens + token_id * H + h_abs).to(gl.float32)
                    w = gl.load(gate_weights + h_abs * E + expert_id).to(gl.float32)
                    logit += t * w
                    h_abs += 1

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

            i_abs = zero_i
            while i_abs < I:
                up_acc = gl.load(bias_up + expert_idx * I + i_abs).to(gl.float32)
                h_abs = zero_i
                while h_abs < H:
                    t = gl.load(tokens + routed_token * H + h_abs, mask=active, other=0.0).to(gl.float32)
                    wu = gl.load(
                        expert_up + expert_idx * H * I + h_abs * I + i_abs
                    ).to(gl.float32)
                    up_acc += t * wu
                    h_abs += 1

                hidden = _apply_activation(up_acc, ACTIVATION)
                if GATED:
                    v_acc = gl.load(bias_up_v + expert_idx * I + i_abs).to(gl.float32)
                    h_abs = zero_i
                    while h_abs < H:
                        t = gl.load(tokens + routed_token * H + h_abs, mask=active, other=0.0).to(gl.float32)
                        wv = gl.load(
                            expert_up_v + expert_idx * H * I + h_abs * I + i_abs
                        ).to(gl.float32)
                        v_acc += t * wv
                        h_abs += 1
                    hidden *= v_acc

                wd = gl.load(
                    expert_down + expert_idx * I * H + i_abs * H + offs_h,
                    mask=h_mask,
                    other=0.0,
                ).to(gl.float32)
                route_vals += hidden * wd
                i_abs += 1

            gl.atomic_add(
                output + routed_token * H + offs_h,
                route_prob * route_vals,
                sem="relaxed",
                scope="gpu",
                mask=active & h_mask,
            )

            task_id = _processor_next_task(processor_ready, processor_mailboxes)

    _global_barrier(barriers, 3, NUM_PROGRAMS)


@gluon.aggregate
class _LocalWsArgs:
    tokens: gl.tensor
    expert_up: gl.tensor
    bias_up: gl.tensor
    expert_up_v: gl.tensor
    bias_up_v: gl.tensor
    hidden: gl.tensor
    expert_down: gl.tensor
    route_tokens: gl.tensor
    route_probs: gl.tensor
    expert_counts: gl.tensor
    bias_down: gl.tensor
    output: gl.tensor
    task_queue: gl.tensor
    task_head: gl.tensor
    task_tail: gl.tensor
    gemm0_pending: gl.tensor
    tile_sync: gl.tensor
    processor_mailboxes: gl.tensor
    processor_seen: gl.tensor
    debug_state: gl.tensor

    gemm0_x_buffer: gl.shared_memory_descriptor
    gemm0_w_buffer: gl.shared_memory_descriptor
    gemm0_xv_buffer: gl.shared_memory_descriptor
    gemm0_wv_buffer: gl.shared_memory_descriptor
    gemm0_hidden_buffer: gl.shared_memory_descriptor
    gemm0_load_empty_bars: gl.shared_memory_descriptor
    gemm0_load_ready_bars: gl.shared_memory_descriptor
    gemm0_v_load_empty_bars: gl.shared_memory_descriptor
    gemm0_v_load_ready_bars: gl.shared_memory_descriptor
    gemm0_acc_empty_bars: gl.shared_memory_descriptor
    gemm0_acc_ready_bars: gl.shared_memory_descriptor
    gemm1_h_buffer: gl.shared_memory_descriptor
    gemm1_down_buffer: gl.shared_memory_descriptor
    gemm1_acc_buffer: gl.shared_memory_descriptor
    gemm1_load_empty_bars: gl.shared_memory_descriptor
    gemm1_load_ready_bars: gl.shared_memory_descriptor
    gemm1_acc_empty_bars: gl.shared_memory_descriptor
    gemm1_acc_ready_bars: gl.shared_memory_descriptor
    task_start_bar: gl.shared_memory_descriptor

    S: gl.constexpr
    H: gl.constexpr
    I: gl.constexpr
    EC: gl.constexpr
    TOP_K: gl.constexpr
    ACTIVATION: gl.constexpr
    GATED: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    NUM_BUFFERS: gl.constexpr
    NUM_ACC_BUFFERS: gl.constexpr
    PRODUCER_WARPS: gl.constexpr
    COMPUTE_WARPS: gl.constexpr
    EPILOGUE_WARPS: gl.constexpr
    ROUTE_BLOCKS: gl.constexpr
    I_TILES: gl.constexpr
    H_TILES: gl.constexpr
    GEMM0_TASKS: gl.constexpr
    TOTAL_TASKS: gl.constexpr
    NUM_PROGRAMS: gl.constexpr
    DEBUG: gl.constexpr
    DEBUG_SCHED: gl.constexpr
    shared_a_layout: gl.constexpr
    shared_b_layout: gl.constexpr
    shared_h_layout: gl.constexpr
    shared_down_layout: gl.constexpr
    wmma_layout: gl.constexpr


@gluon.jit
def _local_ws_epilogue_loop_p(p):
    pid = gl.program_id(0)
    task_phase = _WsPhaseCounter.create(0, 1)
    task_id = _worker_claim_task_queue(
        p.task_queue,
        p.task_head,
        p.task_tail,
        p.gemm0_pending,
        p.processor_mailboxes,
        p.debug_state,
        p.DEBUG,
        p.DEBUG_SCHED,
        p.TOTAL_TASKS,
    )
    while task_id < p.TOTAL_TASKS:
        if p.DEBUG:
            gl.store(p.debug_state + pid, 400)
            gl.store(p.debug_state + 48 + pid, task_id)
        if task_id < p.GEMM0_TASKS:
            expert_idx = task_id // (p.ROUTE_BLOCKS * p.I_TILES)
            rem_task = task_id - expert_idx * p.ROUTE_BLOCKS * p.I_TILES
            route_block = rem_task // p.I_TILES
            i_tile = rem_task - route_block * p.I_TILES
            base_slot = route_block * p.BLOCK_M
            base_i = i_tile * p.BLOCK_N
            expert_count = gl.load(p.expert_counts + expert_idx)
            active_route_block = base_slot < gl.minimum(expert_count, expert_count * 0 + p.EC)

            if active_route_block:
                if p.GATED:
                    _init_ws_load_mbarriers(
                        p.gemm0_v_load_empty_bars,
                        p.gemm0_v_load_ready_bars,
                        p.NUM_BUFFERS,
                        p.PRODUCER_WARPS,
                        p.COMPUTE_WARPS,
                        1,
                    )
                _init_ws_mbarriers(
                    p.gemm0_load_empty_bars,
                    p.gemm0_load_ready_bars,
                    p.gemm0_acc_empty_bars,
                    p.gemm0_acc_ready_bars,
                    p.NUM_BUFFERS,
                    p.NUM_ACC_BUFFERS,
                    p.PRODUCER_WARPS,
                    p.COMPUTE_WARPS,
                    p.EPILOGUE_WARPS,
                    True,
                    1,
                )
            _wait_dscnt0()
            mbarrier.arrive(p.task_start_bar.index(0), count=1)
            mbarrier.wait(p.task_start_bar.index(0), task_phase.phase())
            task_phase = task_phase.next()
            if active_route_block:
                _gemm0_ws_epilogue(
                    p.hidden,
                    p.gemm0_hidden_buffer,
                    p.gemm0_acc_empty_bars,
                    p.gemm0_acc_ready_bars,
                    expert_idx,
                    base_slot,
                    base_i,
                    p.EC,
                    p.I,
                    p.BLOCK_M,
                    p.BLOCK_N,
                    p.shared_h_layout,
                )
            if active_route_block:
                sync_idx = expert_idx * p.ROUTE_BLOCKS + route_block
                done_tiles = gl.atomic_add(p.tile_sync + sync_idx, 1, sem="release", scope="gpu") + 1
                if done_tiles == p.I_TILES:
                    h_enqueue = pid * 0
                    while h_enqueue < p.H_TILES:
                        downstream_task = p.GEMM0_TASKS + expert_idx * p.ROUTE_BLOCKS * p.H_TILES + route_block * p.H_TILES + h_enqueue
                        _enqueue_task(p.task_queue, p.task_tail, downstream_task)
                        h_enqueue += 1
            gl.atomic_add(p.gemm0_pending, -1, sem="release", scope="gpu")
        else:
            gemm1_id = task_id - p.GEMM0_TASKS
            expert_idx = gemm1_id // (p.ROUTE_BLOCKS * p.H_TILES)
            rem_task = gemm1_id - expert_idx * p.ROUTE_BLOCKS * p.H_TILES
            route_block = rem_task // p.H_TILES
            h_tile = rem_task - route_block * p.H_TILES
            base_slot = route_block * p.BLOCK_M
            base_h = h_tile * p.BLOCK_N
            expert_count = gl.load(p.expert_counts + expert_idx)
            active_route_block = base_slot < gl.minimum(expert_count, expert_count * 0 + p.EC)

            if active_route_block:
                _init_ws_mbarriers(
                    p.gemm1_load_empty_bars,
                    p.gemm1_load_ready_bars,
                    p.gemm1_acc_empty_bars,
                    p.gemm1_acc_ready_bars,
                    p.NUM_BUFFERS,
                    p.NUM_ACC_BUFFERS,
                    p.PRODUCER_WARPS,
                    p.COMPUTE_WARPS,
                    p.EPILOGUE_WARPS,
                    False,
                    1,
                )
            _wait_dscnt0()
            mbarrier.arrive(p.task_start_bar.index(0), count=1)
            mbarrier.wait(p.task_start_bar.index(0), task_phase.phase())
            task_phase = task_phase.next()
            if active_route_block:
                _gemm1_ws_epilogue(
                    p.route_tokens,
                    p.route_probs,
                    p.expert_counts,
                    p.bias_down,
                    p.output,
                    p.gemm1_acc_buffer,
                    p.gemm1_acc_empty_bars,
                    p.gemm1_acc_ready_bars,
                    expert_idx,
                    base_slot,
                    base_h,
                    p.EC,
                    p.H,
                    p.TOP_K,
                    p.BLOCK_M,
                    p.BLOCK_N,
                    p.EPILOGUE_WARPS,
                    p.wmma_layout,
                )
                _wait_storecnt0()
        task_id = _worker_claim_task_queue(
            p.task_queue,
            p.task_head,
            p.task_tail,
            p.gemm0_pending,
            p.processor_mailboxes,
            p.debug_state,
            p.DEBUG,
            p.DEBUG_SCHED,
            p.TOTAL_TASKS,
        )
    if p.DEBUG:
        gl.store(p.debug_state + pid, 800)


@gluon.jit
def _local_ws_compute_loop_p(p):
    pid = gl.program_id(0)
    if p.DEBUG:
        gl.store(p.debug_state + 8 + pid, 500)
    task_phase = _WsPhaseCounter.create(0, 1)
    task_id = _processor_wait_task_queue_role(
        p.processor_mailboxes,
        p.processor_seen,
        p.debug_state,
        1,
        p.NUM_PROGRAMS,
        p.DEBUG,
        p.DEBUG_SCHED,
    )
    while task_id < p.TOTAL_TASKS:
        if p.DEBUG:
            gl.store(p.debug_state + 40 + pid, task_id)
        if task_id < p.GEMM0_TASKS:
            expert_idx = task_id // (p.ROUTE_BLOCKS * p.I_TILES)
            rem_task = task_id - expert_idx * p.ROUTE_BLOCKS * p.I_TILES
            route_block = rem_task // p.I_TILES
            i_tile = rem_task - (rem_task // p.I_TILES) * p.I_TILES
            base_slot = route_block * p.BLOCK_M
            base_i = i_tile * p.BLOCK_N
            expert_count = gl.load(p.expert_counts + expert_idx)
            active_route_block = base_slot < gl.minimum(expert_count, expert_count * 0 + p.EC)
            mbarrier.arrive(p.task_start_bar.index(0), count=1)
            mbarrier.wait(p.task_start_bar.index(0), task_phase.phase())
            task_phase = task_phase.next()
            if active_route_block:
                _gemm0_ws_compute(
                    p.bias_up,
                    p.bias_up_v,
                    p.gemm0_x_buffer,
                    p.gemm0_w_buffer,
                    p.gemm0_xv_buffer,
                    p.gemm0_wv_buffer,
                    p.gemm0_hidden_buffer,
                    p.gemm0_load_empty_bars,
                    p.gemm0_load_ready_bars,
                    p.gemm0_v_load_empty_bars,
                    p.gemm0_v_load_ready_bars,
                    p.gemm0_acc_empty_bars,
                    p.gemm0_acc_ready_bars,
                    expert_idx,
                    base_i,
                    p.H,
                    p.I,
                    p.BLOCK_M,
                    p.BLOCK_N,
                    p.NUM_BUFFERS,
                    p.ACTIVATION,
                    p.GATED,
                    p.wmma_layout,
                )
        else:
            gemm1_id = task_id - p.GEMM0_TASKS
            expert_idx = gemm1_id // (p.ROUTE_BLOCKS * p.H_TILES)
            rem_task = gemm1_id - expert_idx * p.ROUTE_BLOCKS * p.H_TILES
            route_block = rem_task // p.H_TILES
            base_slot = route_block * p.BLOCK_M
            expert_count = gl.load(p.expert_counts + expert_idx)
            active_route_block = base_slot < gl.minimum(expert_count, expert_count * 0 + p.EC)
            mbarrier.arrive(p.task_start_bar.index(0), count=1)
            mbarrier.wait(p.task_start_bar.index(0), task_phase.phase())
            task_phase = task_phase.next()
            if active_route_block:
                _gemm1_ws_compute(
                    p.gemm1_h_buffer,
                    p.gemm1_down_buffer,
                    p.gemm1_acc_buffer,
                    p.gemm1_load_empty_bars,
                    p.gemm1_load_ready_bars,
                    p.gemm1_acc_empty_bars,
                    p.gemm1_acc_ready_bars,
                    p.I,
                    p.BLOCK_M,
                    p.BLOCK_N,
                    p.NUM_BUFFERS,
                    p.wmma_layout,
                )
        task_id = _processor_wait_task_queue_role(
            p.processor_mailboxes,
            p.processor_seen,
            p.debug_state,
            1,
            p.NUM_PROGRAMS,
            p.DEBUG,
            p.DEBUG_SCHED,
        )
    if p.DEBUG:
        gl.store(p.debug_state + 8 + pid, 800)


@gluon.jit
def _local_ws_producer_loop_p(p):
    pid = gl.program_id(0)
    if p.DEBUG:
        gl.store(p.debug_state + 16 + pid, 600)
    task_phase = _WsPhaseCounter.create(0, 1)
    task_id = _processor_wait_task_queue_role(
        p.processor_mailboxes,
        p.processor_seen,
        p.debug_state,
        2,
        p.NUM_PROGRAMS,
        p.DEBUG,
        p.DEBUG_SCHED,
    )
    while task_id < p.TOTAL_TASKS:
        if p.DEBUG:
            gl.store(p.debug_state + 56 + pid, task_id)
        if task_id < p.GEMM0_TASKS:
            expert_idx = task_id // (p.ROUTE_BLOCKS * p.I_TILES)
            rem_task = task_id - expert_idx * p.ROUTE_BLOCKS * p.I_TILES
            route_block = rem_task // p.I_TILES
            i_tile = rem_task - route_block * p.I_TILES
            base_slot = route_block * p.BLOCK_M
            base_i = i_tile * p.BLOCK_N
            expert_count = gl.load(p.expert_counts + expert_idx)
            active_route_block = base_slot < gl.minimum(expert_count, expert_count * 0 + p.EC)
            mbarrier.arrive(p.task_start_bar.index(0), count=1)
            mbarrier.wait(p.task_start_bar.index(0), task_phase.phase())
            task_phase = task_phase.next()
            if active_route_block:
                _gemm0_ws_producer(
                    p.tokens,
                    p.expert_up,
                    p.expert_up_v,
                    p.route_tokens,
                    p.expert_counts,
                    p.gemm0_x_buffer,
                    p.gemm0_w_buffer,
                    p.gemm0_xv_buffer,
                    p.gemm0_wv_buffer,
                    p.gemm0_load_empty_bars,
                    p.gemm0_load_ready_bars,
                    p.gemm0_v_load_empty_bars,
                    p.gemm0_v_load_ready_bars,
                    expert_idx,
                    base_slot,
                    base_i,
                    p.S,
                    p.H,
                    p.I,
                    p.EC,
                    p.BLOCK_M,
                    p.BLOCK_N,
                    p.NUM_BUFFERS,
                    p.PRODUCER_WARPS,
                    p.GATED,
                    p.shared_a_layout,
                    p.shared_b_layout,
                )
        else:
            gemm1_id = task_id - p.GEMM0_TASKS
            expert_idx = gemm1_id // (p.ROUTE_BLOCKS * p.H_TILES)
            rem_task = gemm1_id - expert_idx * p.ROUTE_BLOCKS * p.H_TILES
            route_block = rem_task // p.H_TILES
            h_tile = rem_task - route_block * p.H_TILES
            base_slot = route_block * p.BLOCK_M
            base_h = h_tile * p.BLOCK_N
            expert_count = gl.load(p.expert_counts + expert_idx)
            active_route_block = base_slot < gl.minimum(expert_count, expert_count * 0 + p.EC)
            mbarrier.arrive(p.task_start_bar.index(0), count=1)
            mbarrier.wait(p.task_start_bar.index(0), task_phase.phase())
            task_phase = task_phase.next()
            if active_route_block:
                _gemm1_ws_producer(
                    p.hidden,
                    p.expert_down,
                    p.gemm1_h_buffer,
                    p.gemm1_down_buffer,
                    p.gemm1_load_empty_bars,
                    p.gemm1_load_ready_bars,
                    expert_idx,
                    base_slot,
                    base_h,
                    p.EC,
                    p.I,
                    p.H,
                    p.BLOCK_M,
                    p.BLOCK_N,
                    p.NUM_BUFFERS,
                    p.shared_h_layout,
                    p.shared_down_layout,
                )
        task_id = _processor_wait_task_queue_role(
            p.processor_mailboxes,
            p.processor_seen,
            p.debug_state,
            2,
            p.NUM_PROGRAMS,
            p.DEBUG,
            p.DEBUG_SCHED,
        )
    if p.DEBUG:
        gl.store(p.debug_state + 16 + pid, 800)


@gluon.jit
def _persistent_tdm_wmma_kernel(
    tokens,
    gate_weights,
    topk_ids,
    topk_weights,
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
    task_queue,
    task_head,
    task_tail,
    gemm0_pending,
    tile_sync,
    processor_mailboxes,
    processor_seen,
    barriers,
    init_flag,
    debug_state,
    launch_epoch,
    S: gl.constexpr,
    H: gl.constexpr,
    I: gl.constexpr,
    E: gl.constexpr,
    EC: gl.constexpr,
    TOP_K: gl.constexpr,
    PRECOMPUTED_ROUTING: gl.constexpr,
    ZERO_OUTPUT: gl.constexpr,
    ACTIVATION: gl.constexpr,
    GATED: gl.constexpr,
    BLOCK_E: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    DEBUG: gl.constexpr,
    DEBUG_SCHED: gl.constexpr,
):
    gl.static_assert(BLOCK_M % 16 == 0, "TDM/WMMA path expects a WMMA-row-aligned BLOCK_M")
    gl.static_assert(BLOCK_N == 64, "TDM/WMMA path expects 64-column tiles")
    gl.static_assert(H % BLOCK_N == 0, "TDM/WMMA path expects H to be divisible by 64")
    gl.static_assert(I % BLOCK_N == 0, "TDM/WMMA path expects I to be divisible by 64")

    pid = gl.program_id(0)
    if DEBUG:
        gl.store(debug_state + pid, 10)
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
        if DEBUG:
            gl.store(debug_state + pid, 20)
        while gl.atomic_cas(
            init_flag,
            launch_epoch,
            launch_epoch,
            sem="acquire",
            scope="gpu",
        ) != launch_epoch:
            pass

    if DEBUG:
        gl.store(debug_state + pid, 30)
    if ZERO_OUTPUT:
        out_base = pid * BLOCK_H
        while out_base < S * H:
            gl.store(
                output + out_base + offs_h,
                gl.full((BLOCK_H,), 0.0, gl.float32, layout=h_layout),
                mask=out_base + offs_h < S * H,
            )
            out_base += NUM_PROGRAMS * BLOCK_H
        _wait_storecnt0()
    _persistent_phase_barrier(barriers, 0, NUM_PROGRAMS)
    if DEBUG:
        gl.store(debug_state + pid, 100)

    token_id = pid
    while token_id < S:
        zero_f = token_id.to(gl.float32) * 0.0
        zero_i = token_id * 0

        if PRECOMPUTED_ROUTING:
            route_id = zero_i
            while route_id < TOP_K:
                expert_idx = gl.load(topk_ids + token_id * TOP_K + route_id).to(gl.int32)
                route_prob = gl.load(topk_weights + token_id * TOP_K + route_id).to(gl.float32)
                slot = gl.atomic_add(expert_counts + expert_idx, 1, sem="relaxed", scope="gpu")
                active = slot < EC
                safe_slot = gl.minimum(slot, slot * 0 + EC - 1)
                gl.store(route_tokens + expert_idx * EC + safe_slot, token_id, mask=active)
                gl.store(route_probs + expert_idx * EC + safe_slot, route_prob, mask=active)
                route_id += 1
        else:
            top0_idx = zero_i
            top1_idx = zero_i
            top0_logit = zero_f - float("inf")
            top1_logit = zero_f - float("inf")
            expert_id = zero_i
            while expert_id < E:
                logit = zero_f
                h_abs = zero_i
                while h_abs < H:
                    t = gl.load(tokens + token_id * H + h_abs).to(gl.float32)
                    w = gl.load(gate_weights + h_abs * E + expert_id).to(gl.float32)
                    logit += t * w
                    h_abs += 1

                better0 = logit > top0_logit
                better1 = (logit > top1_logit) & (logit <= top0_logit)
                old_top0_idx = top0_idx
                old_top0_logit = top0_logit

                top1_idx = gl.where(better0, old_top0_idx, top1_idx)
                top1_logit = gl.where(better0, old_top0_logit, top1_logit)
                top0_idx = gl.where(better0, expert_id, top0_idx)
                top0_logit = gl.where(better0, logit, top0_logit)

                top1_idx = gl.where(better1, expert_id, top1_idx)
                top1_logit = gl.where(better1, logit, top1_logit)
                expert_id += 1

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
    _wait_loadcnt0()
    _wait_storecnt0()
    _persistent_phase_barrier(barriers, 1, NUM_PROGRAMS)
    if DEBUG:
        gl.store(debug_state + pid, 200)

    _worker_reset_queue_state(processor_mailboxes, processor_seen, NUM_PROGRAMS, 3)

    gl.static_assert(NUM_WARPS == 4, "warp-specialized TDM/WMMA path expects four epilogue waves")
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
    _init_dynamic_task_queue(
        task_queue,
        task_head,
        task_tail,
        gemm0_pending,
        tile_sync,
        gemm0_tasks,
        total_tasks,
        sync_slots,
    )
    _persistent_phase_barrier(barriers, 2, NUM_PROGRAMS)
    if DEBUG:
        gl.store(debug_state + pid, 300)

    gemm0_x_buffer = gl.allocate_shared_memory(
        tokens.dtype.element_ty,
        shape=[NUM_BUFFERS, BLOCK_M, BLOCK_N],
        layout=shared_a_layout,
    )
    gemm0_w_buffer = gl.allocate_shared_memory(
        expert_up.dtype.element_ty,
        shape=[NUM_BUFFERS, BLOCK_N, BLOCK_N],
        layout=shared_b_layout,
    )
    gemm0_load_empty_bars = gl.allocate_shared_memory(gl.int64, [NUM_BUFFERS, 1], mbarrier.MBarrierLayout())
    gemm0_load_ready_bars = gl.allocate_shared_memory(gl.int64, [NUM_BUFFERS, 1], mbarrier.MBarrierLayout())
    gemm0_xv_buffer = gemm0_x_buffer
    gemm0_wv_buffer = gemm0_w_buffer
    gemm0_v_load_empty_bars = gemm0_load_empty_bars
    gemm0_v_load_ready_bars = gemm0_load_ready_bars
    if GATED:
        gemm0_xv_buffer = gl.allocate_shared_memory(
            tokens.dtype.element_ty,
            shape=[NUM_BUFFERS, BLOCK_M, BLOCK_N],
            layout=shared_a_layout,
        )
        gemm0_wv_buffer = gl.allocate_shared_memory(
            expert_up_v.dtype.element_ty,
            shape=[NUM_BUFFERS, BLOCK_N, BLOCK_N],
            layout=shared_b_layout,
        )
        gemm0_v_load_empty_bars = gl.allocate_shared_memory(gl.int64, [NUM_BUFFERS, 1], mbarrier.MBarrierLayout())
        gemm0_v_load_ready_bars = gl.allocate_shared_memory(gl.int64, [NUM_BUFFERS, 1], mbarrier.MBarrierLayout())
    gemm0_hidden_buffer = gl.allocate_shared_memory(
        hidden.dtype.element_ty,
        shape=[NUM_ACC_BUFFERS, BLOCK_M, BLOCK_N],
        layout=shared_h_layout,
    )
    gemm0_acc_empty_bars = gl.allocate_shared_memory(gl.int64, [NUM_ACC_BUFFERS, 1], mbarrier.MBarrierLayout())
    gemm0_acc_ready_bars = gl.allocate_shared_memory(gl.int64, [NUM_ACC_BUFFERS, 1], mbarrier.MBarrierLayout())

    gemm1_h_buffer = gemm0_x_buffer
    gemm1_down_buffer = gemm0_w_buffer
    gemm1_acc_buffer = gl.allocate_shared_memory(
        gl.float32,
        shape=[NUM_ACC_BUFFERS, BLOCK_M, BLOCK_N],
        layout=shared_acc_layout,
    )
    gemm1_load_empty_bars = gemm0_load_empty_bars
    gemm1_load_ready_bars = gemm0_load_ready_bars
    gemm1_acc_empty_bars = gemm0_acc_empty_bars
    gemm1_acc_ready_bars = gemm0_acc_ready_bars
    task_start_bar = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
    mbarrier.init(task_start_bar.index(0), count=(EPILOGUE_WARPS + COMPUTE_WARPS + PRODUCER_WARPS) * 32)

    p = _LocalWsArgs(
        tokens,
        expert_up,
        bias_up,
        expert_up_v,
        bias_up_v,
        hidden,
        expert_down,
        route_tokens,
        route_probs,
        expert_counts,
        bias_down,
        output,
        task_queue,
        task_head,
        task_tail,
        gemm0_pending,
        tile_sync,
        processor_mailboxes,
        processor_seen,
        debug_state,
        gemm0_x_buffer,
        gemm0_w_buffer,
        gemm0_xv_buffer,
        gemm0_wv_buffer,
        gemm0_hidden_buffer,
        gemm0_load_empty_bars,
        gemm0_load_ready_bars,
        gemm0_v_load_empty_bars,
        gemm0_v_load_ready_bars,
        gemm0_acc_empty_bars,
        gemm0_acc_ready_bars,
        gemm1_h_buffer,
        gemm1_down_buffer,
        gemm1_acc_buffer,
        gemm1_load_empty_bars,
        gemm1_load_ready_bars,
        gemm1_acc_empty_bars,
        gemm1_acc_ready_bars,
        task_start_bar,
        S,
        H,
        I,
        EC,
        TOP_K,
        ACTIVATION,
        GATED,
        BLOCK_M,
        BLOCK_N,
        NUM_BUFFERS,
        NUM_ACC_BUFFERS,
        PRODUCER_WARPS,
        COMPUTE_WARPS,
        EPILOGUE_WARPS,
        route_blocks,
        i_tiles,
        h_tiles,
        gemm0_tasks,
        total_tasks,
        NUM_PROGRAMS,
        DEBUG,
        DEBUG_SCHED,
        shared_a_layout,
        shared_b_layout,
        shared_h_layout,
        shared_down_layout,
        wmma_layout,
    )
    gl.warp_specialize([
        (_local_ws_epilogue_loop_p, (p, )),
        (_local_ws_compute_loop_p, (p, )),
        (_local_ws_producer_loop_p, (p, )),
    ], [COMPUTE_WARPS, PRODUCER_WARPS])
    if DEBUG:
        gl.store(debug_state + pid, 999)




@gluon.jit
def _persistent_rocshmem_tdm_wmma_kernel(
    tokens,
    gate_weights,
    local_expert_up,
    bias_up,
    local_expert_up_v,
    bias_up_v,
    local_expert_down,
    bias_down,
    output,
    hidden,
    dispatch_tokens,
    dispatch_token_ids,
    dispatch_probs,
    dispatch_counts,
    dispatch_signals,
    result_values,
    result_counts,
    result_signals,
    compute_done,
    task_queue,
    task_tail,
    publish_queue,
    publish_tail,
    tile_sync,
    dispatch_seen,
    result_seen,
    task_bound,
    tasks_done,
    os_done,
    dispatch_done,
    processor_ready,
    processor_mailboxes,
    processor_seen,
    barriers,
    init_flag,
    debug_state,
    launch_epoch,
    S: gl.constexpr,
    H: gl.constexpr,
    I: gl.constexpr,
    E: gl.constexpr,
    WORLD: gl.constexpr,
    NLX: gl.constexpr,
    EC: gl.constexpr,
    TOP_K: gl.constexpr,
    BLOCK_E: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    ACTIVATION: gl.constexpr,
    GATED: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    RESULT_ELEM_BYTES: gl.constexpr,
    DEBUG: gl.constexpr,
):
    gl.static_assert(NUM_PROGRAMS >= 3, "rocSHMEM dynamic megakernel expects OS, dispatch, and compute programs")
    gl.static_assert(BLOCK_M % 16 == 0, "rocSHMEM TDM/WMMA path expects a WMMA-row-aligned BLOCK_M")
    gl.static_assert(BLOCK_N == 64, "rocSHMEM TDM/WMMA path expects 64-column tiles")
    gl.static_assert(H % BLOCK_N == 0, "rocSHMEM TDM/WMMA path expects H to be divisible by 64")
    gl.static_assert(I % BLOCK_N == 0, "rocSHMEM TDM/WMMA path expects I to be divisible by 64")
    gl.static_assert(NUM_WARPS == 4, "rocSHMEM warp-specialized TDM/WMMA path expects four scheduler/epilogue waves")

    pid = gl.program_id(0)
    my_pe = rocshmem.my_pe()
    h_layout: gl.constexpr = _hidden_layout(NUM_WARPS)
    offs_h = gl.arange(0, BLOCK_H, layout=h_layout)
    h_mask = offs_h < H
    tile_layout: gl.constexpr = _hidden_layout(NUM_WARPS)
    offs_tile = gl.arange(0, BLOCK_N, layout=tile_layout)
    epoch_u64 = gl.full((), launch_epoch, gl.uint64)

    PRODUCER_WARPS: gl.constexpr = 4
    COMPUTE_WARPS: gl.constexpr = 4
    EPILOGUE_WARPS: gl.constexpr = NUM_WARPS
    NUM_BUFFERS: gl.constexpr = 2
    NUM_ACC_BUFFERS: gl.constexpr = 1
    BARRIER_SLOTS: gl.constexpr = 3
    DISPATCH_PROGRAMS: gl.constexpr = 1 if NUM_PROGRAMS <= 4 else (NUM_PROGRAMS - 1) // 4
    FIRST_COMPUTE_PROGRAM: gl.constexpr = 1 + DISPATCH_PROGRAMS
    gl.static_assert(FIRST_COMPUTE_PROGRAM < NUM_PROGRAMS, "rocSHMEM megakernel needs at least one compute program")

    wmma_layout: gl.constexpr = _wmma_layout(COMPUTE_WARPS)
    shared_a_layout: gl.constexpr = _wmma_shared_a_layout(BLOCK_M, BLOCK_N)
    shared_b_layout: gl.constexpr = _wmma_shared_b_layout(BLOCK_N, BLOCK_N)
    shared_h_layout: gl.constexpr = _wmma_shared_a_layout(BLOCK_M, BLOCK_N)
    shared_down_layout: gl.constexpr = _wmma_shared_b_layout(BLOCK_N, BLOCK_N)
    shared_acc_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0], [])

    local_channels: gl.constexpr = WORLD * NLX
    all_channels: gl.constexpr = WORLD * WORLD * NLX
    route_blocks: gl.constexpr = (EC + BLOCK_M - 1) // BLOCK_M
    h_tiles: gl.constexpr = H // BLOCK_N
    i_tiles: gl.constexpr = I // BLOCK_N
    gemm0_tasks: gl.constexpr = local_channels * route_blocks * i_tiles
    gemm1_tasks: gl.constexpr = local_channels * route_blocks * h_tiles
    combine_tasks: gl.constexpr = local_channels * route_blocks * h_tiles
    combine_base: gl.constexpr = gemm0_tasks + gemm1_tasks
    total_compute_tasks: gl.constexpr = gemm0_tasks + gemm1_tasks + combine_tasks
    sync_slots: gl.constexpr = local_channels * route_blocks

    _init_rocshmem_dynamic_state(
        dispatch_counts,
        result_counts,
        task_queue,
        task_tail,
        publish_queue,
        publish_tail,
        tile_sync,
        compute_done,
        dispatch_seen,
        result_seen,
        task_bound,
        tasks_done,
        os_done,
        dispatch_done,
        processor_ready,
        processor_mailboxes,
        processor_seen,
        barriers,
        my_pe,
        total_compute_tasks,
        local_channels,
        sync_slots,
        all_channels,
        local_channels,
        WORLD,
        NLX,
        BARRIER_SLOTS,
        NUM_PROGRAMS,
    )

    if pid == 0:
        gl.atomic_xchg(init_flag, launch_epoch, sem="release", scope="gpu")
    while gl.atomic_cas(init_flag, launch_epoch, launch_epoch, sem="acquire", scope="gpu") != launch_epoch:
        pass

    if pid == 0:
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

    if pid == 0:
        gl.warp_specialize([
            (
                _scheduler_run_dynamic_queue,
                (
                    processor_ready,
                    processor_mailboxes,
                    task_queue,
                    task_tail,
                    task_bound,
                    tasks_done,
                    os_done,
                    debug_state,
                    total_compute_tasks,
                    NUM_PROGRAMS,
                    FIRST_COMPUTE_PROGRAM,
                    DEBUG,
                ),
            ),
            (
                _rocshmem_os_subscriber_publisher,
                (
                    task_queue,
                    task_tail,
                    publish_queue,
                    publish_tail,
                    dispatch_counts,
                    dispatch_signals,
                    result_values,
                    result_counts,
                    result_signals,
                    dispatch_seen,
                    result_seen,
                    task_bound,
                    tasks_done,
                    os_done,
                    debug_state,
                    launch_epoch,
                    my_pe,
                    WORLD,
                    NLX,
                    EC,
                    H,
                    I,
                    BLOCK_M,
                    BLOCK_N,
                    RESULT_ELEM_BYTES,
                    DEBUG,
                ),
            ),
        ], [1])
    else:
        dispatch_rank = pid - 1
        if dispatch_rank < DISPATCH_PROGRAMS:
            token_id = dispatch_rank
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
                        h_abs = zero_i
                        while h_abs < H:
                            t = gl.load(tokens + token_id * H + h_abs).to(gl.float32)
                            w = gl.load(gate_weights + h_abs * E + expert_id).to(gl.float32)
                            logit += t * w
                            h_abs += 1
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
                        owner_pe = expert_idx // NLX
                        local_expert = expert_idx - owner_pe * NLX
                        route_prob = zero_f + 1.0
                        if TOP_K == 2:
                            route_exp = gl.where(route_id == 0, top0_exp, top1_exp)
                            route_prob = route_exp / denom

                        count_idx = (owner_pe * WORLD + my_pe) * NLX + local_expert
                        slot = gl.atomic_add(dispatch_counts + count_idx, 1, sem="relaxed", scope="gpu")
                        active = slot < EC
                        safe_slot = gl.minimum(slot, slot * 0 + EC - 1)
                        route_offset = count_idx * EC + safe_slot
                        token_row = gl.load(tokens + token_id * H + offs_h, mask=h_mask, other=0.0)
                        gl.store(dispatch_token_ids + route_offset, token_id, mask=active)
                        gl.store(dispatch_probs + route_offset, route_prob, mask=active)
                        gl.store(dispatch_tokens + route_offset * H + offs_h, token_row, mask=active & h_mask)
                token_id += DISPATCH_PROGRAMS

            gl.barrier()
            done_dispatch = gl.atomic_add(dispatch_done, 1, sem="acq_rel", scope="gpu") + 1
            if done_dispatch == DISPATCH_PROGRAMS:
                if DEBUG:
                    gl.store(debug_state + 24, 1)
                for owner_pe_static in gl.static_range(0, WORLD):
                    owner_pe_send = my_pe * 0 + owner_pe_static
                    for lx_pub in gl.static_range(0, NLX):
                        channel_idx = (owner_pe_send * WORLD + my_pe) * NLX + lx_pub
                        published_count = gl.load(dispatch_counts + channel_idx)
                        published_count = gl.minimum(published_count, published_count * 0 + EC)
                        payload = _signal_payload(epoch_u64, published_count)
                        gl.store(dispatch_counts + channel_idx, published_count)
                        gl.store(dispatch_signals + channel_idx, payload)
                        if owner_pe_send != my_pe:
                            if published_count == 0:
                                rocshmem.signal_op_wg(
                                    dispatch_signals + channel_idx,
                                    payload,
                                    rocshmem.ROCSHMEM_SIGNAL_SET,
                                    owner_pe_send,
                                )
                                rocshmem.quiet()
                            else:
                                rocshmem.putmem_nbi_wg(
                                    dispatch_counts + channel_idx,
                                    dispatch_counts + channel_idx,
                                    4,
                                    owner_pe_send,
                                )
                                rocshmem.putmem_nbi_wg(
                                    dispatch_token_ids + channel_idx * EC,
                                    dispatch_token_ids + channel_idx * EC,
                                    EC * 4,
                                    owner_pe_send,
                                )
                                rocshmem.putmem_nbi_wg(
                                    dispatch_probs + channel_idx * EC,
                                    dispatch_probs + channel_idx * EC,
                                    EC * 4,
                                    owner_pe_send,
                                )
                                rocshmem.quiet()
                                rocshmem.putmem_signal_nbi_wg(
                                    dispatch_tokens + channel_idx * EC * H,
                                    dispatch_tokens + channel_idx * EC * H,
                                    EC * H * 2,
                                    dispatch_signals + channel_idx,
                                    payload,
                                    rocshmem.ROCSHMEM_SIGNAL_SET,
                                    owner_pe_send,
                                )
                                rocshmem.quiet()
                if DEBUG:
                    gl.store(debug_state + 24, 2)

        if pid < FIRST_COMPUTE_PROGRAM:
            while gl.atomic_add(os_done, 0, sem="acquire", scope="gpu") == 0:
                _gpu_relax()
        task_id = pid * 0 + total_compute_tasks
        if pid >= FIRST_COMPUTE_PROGRAM:
            task_id = _processor_next_task_generation(processor_ready, processor_mailboxes, processor_seen, total_compute_tasks)
        while task_id < total_compute_tasks:
            if DEBUG:
                gl.atomic_add(debug_state + 25, 1, sem="relaxed", scope="gpu")
            if task_id < gemm0_tasks:
                hidden_channel = task_id // (route_blocks * i_tiles)
                rem_task = task_id - hidden_channel * route_blocks * i_tiles
                route_block = rem_task // i_tiles
                i_tile = rem_task - route_block * i_tiles
                source_pe = hidden_channel // NLX
                local_expert = hidden_channel - source_pe * NLX
                count_idx = (my_pe * WORLD + source_pe) * NLX + local_expert
                base_slot = route_block * BLOCK_M
                base_i = i_tile * BLOCK_N
                channel_count = gl.load(dispatch_counts + count_idx)
                block_has_rows = base_slot < channel_count
                if block_has_rows:
                    x_buffer = gl.allocate_shared_memory(
                        dispatch_tokens.dtype.element_ty,
                        shape=[NUM_BUFFERS, BLOCK_M, BLOCK_N],
                        layout=shared_a_layout,
                    )
                    w_buffer = gl.allocate_shared_memory(
                        local_expert_up.dtype.element_ty,
                        shape=[NUM_BUFFERS, BLOCK_N, BLOCK_N],
                        layout=shared_b_layout,
                    )
                    load_empty_bars = gl.allocate_shared_memory(gl.int64, [NUM_BUFFERS, 1], mbarrier.MBarrierLayout())
                    load_ready_bars = gl.allocate_shared_memory(gl.int64, [NUM_BUFFERS, 1], mbarrier.MBarrierLayout())
                    xv_buffer = x_buffer
                    wv_buffer = w_buffer
                    v_load_empty_bars = load_empty_bars
                    v_load_ready_bars = load_ready_bars
                    if GATED:
                        xv_buffer = gl.allocate_shared_memory(
                            dispatch_tokens.dtype.element_ty,
                            shape=[NUM_BUFFERS, BLOCK_M, BLOCK_N],
                            layout=shared_a_layout,
                        )
                        wv_buffer = gl.allocate_shared_memory(
                            local_expert_up_v.dtype.element_ty,
                            shape=[NUM_BUFFERS, BLOCK_N, BLOCK_N],
                            layout=shared_b_layout,
                        )
                        v_load_empty_bars = gl.allocate_shared_memory(gl.int64, [NUM_BUFFERS, 1], mbarrier.MBarrierLayout())
                        v_load_ready_bars = gl.allocate_shared_memory(gl.int64, [NUM_BUFFERS, 1], mbarrier.MBarrierLayout())
                        _init_ws_load_mbarriers(
                            v_load_empty_bars,
                            v_load_ready_bars,
                            NUM_BUFFERS,
                            PRODUCER_WARPS,
                            COMPUTE_WARPS,
                            1,
                        )
                    hidden_buffer = gl.allocate_shared_memory(
                        hidden.dtype.element_ty,
                        shape=[NUM_ACC_BUFFERS, BLOCK_M, BLOCK_N],
                        layout=shared_h_layout,
                    )
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
                    gl.warp_specialize([
                        (
                            _gemm0_ws_epilogue,
                            (
                                hidden,
                                hidden_buffer,
                                acc_empty_bars,
                                acc_ready_bars,
                                hidden_channel,
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
                                bias_up_v,
                                x_buffer,
                                w_buffer,
                                xv_buffer,
                                wv_buffer,
                                hidden_buffer,
                                load_empty_bars,
                                load_ready_bars,
                                v_load_empty_bars,
                                v_load_ready_bars,
                                acc_empty_bars,
                                acc_ready_bars,
                                local_expert,
                                base_i,
                                H,
                                I,
                                BLOCK_M,
                                BLOCK_N,
                                NUM_BUFFERS,
                                ACTIVATION,
                                GATED,
                                wmma_layout,
                            ),
                        ),
                        (
                            _gemm0_channel_ws_producer,
                            (
                                dispatch_tokens,
                                local_expert_up,
                                local_expert_up_v,
                                x_buffer,
                                w_buffer,
                                xv_buffer,
                                wv_buffer,
                                load_empty_bars,
                                load_ready_bars,
                                v_load_empty_bars,
                                v_load_ready_bars,
                                count_idx,
                                local_expert,
                                base_slot,
                                base_i,
                                H,
                                I,
                                EC,
                                BLOCK_M,
                                BLOCK_N,
                                NUM_BUFFERS,
                                PRODUCER_WARPS,
                                GATED,
                                shared_a_layout,
                                shared_b_layout,
                            ),
                        ),
                    ], [COMPUTE_WARPS, PRODUCER_WARPS])
                sync_idx = hidden_channel * route_blocks + route_block
                done_tiles = gl.atomic_add(tile_sync + sync_idx, 1, sem="acq_rel", scope="gpu") + 1
                if done_tiles == i_tiles:
                    h_enqueue = pid * 0
                    while h_enqueue < h_tiles:
                        downstream_task = gemm0_tasks + hidden_channel * route_blocks * h_tiles + route_block * h_tiles + h_enqueue
                        _enqueue_compute_task(task_queue, task_tail, task_bound, downstream_task)
                        h_enqueue += 1
                if DEBUG:
                    gl.atomic_add(debug_state + 26, 1, sem="relaxed", scope="gpu")
                gl.atomic_add(tasks_done, 1, sem="release", scope="gpu")
            elif task_id < combine_base:
                gemm1_id = task_id - gemm0_tasks
                hidden_channel = gemm1_id // (route_blocks * h_tiles)
                rem_task = gemm1_id - hidden_channel * route_blocks * h_tiles
                route_block = rem_task // h_tiles
                h_tile = rem_task - route_block * h_tiles
                source_pe = hidden_channel // NLX
                local_expert = hidden_channel - source_pe * NLX
                count_idx = (my_pe * WORLD + source_pe) * NLX + local_expert
                base_slot = route_block * BLOCK_M
                base_h = h_tile * BLOCK_N
                channel_count = gl.load(dispatch_counts + count_idx)
                block_has_rows = base_slot < channel_count
                if block_has_rows:
                    h_buffer = gl.allocate_shared_memory(
                        hidden.dtype.element_ty,
                        shape=[NUM_BUFFERS, BLOCK_M, BLOCK_N],
                        layout=shared_h_layout,
                    )
                    down_buffer = gl.allocate_shared_memory(
                        local_expert_down.dtype.element_ty,
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
                    gl.warp_specialize([
                        (
                            _gemm1_channel_result_epilogue,
                            (
                                dispatch_counts,
                                bias_down,
                                result_values,
                                acc_buffer,
                                acc_empty_bars,
                                acc_ready_bars,
                                count_idx,
                                local_expert,
                                base_slot,
                                base_h,
                                EC,
                                H,
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
                            _gemm1_channel_ws_producer,
                            (
                                hidden,
                                local_expert_down,
                                h_buffer,
                                down_buffer,
                                load_empty_bars,
                                load_ready_bars,
                                hidden_channel,
                                local_expert,
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
                    ], [COMPUTE_WARPS, PRODUCER_WARPS])
                    _wait_storecnt0()
                done_tiles = gl.atomic_add(compute_done + count_idx, 1, sem="acq_rel", scope="gpu") + 1
                capped_count = gl.minimum(channel_count, channel_count * 0 + EC)
                active_blocks = (capped_count + BLOCK_M - 1) // BLOCK_M
                expected_tiles = active_blocks * h_tiles
                if done_tiles == expected_tiles:
                    gl.store(result_counts + count_idx, capped_count)
                    if source_pe == my_pe:
                        gl.store(result_signals + count_idx, _signal_payload(epoch_u64, capped_count))
                    else:
                        publish_task = source_pe * NLX + local_expert
                        _enqueue_publish_task(publish_queue, publish_tail, publish_task)
                if DEBUG:
                    gl.atomic_add(debug_state + 27, 1, sem="relaxed", scope="gpu")
                gl.atomic_add(tasks_done, 1, sem="release", scope="gpu")
            else:
                combine_id = task_id - combine_base
                owner_channel = combine_id // (route_blocks * h_tiles)
                rem_combine = combine_id - owner_channel * route_blocks * h_tiles
                route_block = rem_combine // h_tiles
                h_tile = rem_combine - route_block * h_tiles
                owner_pe = owner_channel // NLX
                local_expert = owner_channel - owner_pe * NLX
                count_idx = (owner_pe * WORLD + my_pe) * NLX + local_expert
                base_slot = route_block * BLOCK_M
                base_h = h_tile * BLOCK_N
                offs_out = base_h + offs_tile
                tile_mask = offs_out < H
                result_count = gl.load(result_counts + count_idx)
                for row in gl.static_range(0, BLOCK_M):
                    slot = base_slot + row
                    active = slot < result_count
                    safe_slot = gl.minimum(slot, slot * 0 + EC - 1)
                    row_offset = count_idx * EC + safe_slot
                    token_id_out = gl.load(dispatch_token_ids + row_offset, mask=active, other=0)
                    prob = gl.load(dispatch_probs + row_offset, mask=active, other=0.0).to(gl.float32)
                    vals = gl.load(result_values + row_offset * H + offs_out, mask=active & tile_mask, other=0.0).to(gl.float32)
                    if TOP_K == 1:
                        gl.store(output + token_id_out * H + offs_out, prob * vals, mask=active & tile_mask)
                    else:
                        gl.atomic_add(
                            output + token_id_out * H + offs_out,
                            prob * vals,
                            sem="relaxed",
                            scope="gpu",
                            mask=active & tile_mask,
                        )
                if DEBUG:
                    gl.atomic_add(debug_state + 28, 1, sem="relaxed", scope="gpu")
                gl.atomic_add(tasks_done, 1, sem="release", scope="gpu")
            task_id = _processor_next_task_generation(processor_ready, processor_mailboxes, processor_seen, total_compute_tasks)


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
            h_abs = zero_i
            while h_abs < H:
                t = gl.load(tokens + token_id * H + h_abs).to(gl.float32)
                w = gl.load(gate_weights + h_abs * E + expert_id).to(gl.float32)
                logit += t * w
                h_abs += 1

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

            i_abs = zero_i
            while i_abs < I:
                up_acc = gl.load(bias_up + expert_idx * I + i_abs).to(gl.float32)
                h_abs = zero_i
                while h_abs < H:
                    t = gl.load(tokens + token_id * H + h_abs).to(gl.float32)
                    wu = gl.load(
                        expert_up + expert_idx * H * I + h_abs * I + i_abs
                    ).to(gl.float32)
                    up_acc += t * wu
                    h_abs += 1

                hidden = _apply_activation(up_acc, ACTIVATION)
                if GATED:
                    v_acc = gl.load(bias_up_v + expert_idx * I + i_abs).to(gl.float32)
                    h_abs = zero_i
                    while h_abs < H:
                        t = gl.load(tokens + token_id * H + h_abs).to(gl.float32)
                        wv = gl.load(
                            expert_up_v + expert_idx * H * I + h_abs * I + i_abs
                        ).to(gl.float32)
                        v_acc += t * wv
                        h_abs += 1
                    hidden *= v_acc

                wd = gl.load(
                    expert_down + expert_idx * I * H + i_abs * H + offs_h,
                    mask=h_mask,
                    other=0.0,
                ).to(gl.float32)
                route_vals += hidden * wd
                i_abs += 1

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


def _poll_tdm_heartbeat(debug_state: torch.Tensor, num_programs: int, *, label: str) -> None:
    deadline = time.monotonic() + float(os.environ.get("FLASHMOE_HOST_POLL_TIMEOUT_S", "60"))
    poll_interval_s = float(os.environ.get("FLASHMOE_HOST_POLL_INTERVAL_S", "0.05"))
    while True:
        state = debug_state[:num_programs].tolist()
        if all(value == 999 for value in state):
            break
        if time.monotonic() > deadline:
            compute_state = debug_state[8 : 8 + num_programs].tolist()
            producer_state = debug_state[16 : 16 + num_programs].tolist()
            compute_task = debug_state[40 : 40 + num_programs].tolist()
            task_state = debug_state[48 : 48 + num_programs].tolist()
            producer_task = debug_state[56 : 56 + num_programs].tolist()
            raise TimeoutError(
                f"{label} did not reach final heartbeat; "
                f"state={state} compute_state={compute_state} "
                f"producer_state={producer_state} task_state={task_state} "
                f"compute_task={compute_task} producer_task={producer_task}"
            )
        time.sleep(poll_interval_s)


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
    num_programs: int = 8,
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

    if use_persistent and tdm_wmma_eligible:
        num_programs = _resolve_tdm_num_programs(spec.dtype, spec.block_h, num_programs)

    if use_persistent:
        expert_counts = torch.empty((spec.e,), device=tokens.device, dtype=torch.int32)
        route_tokens = torch.empty((spec.e, spec.expert_capacity), device=tokens.device, dtype=torch.int32)
        route_probs = torch.empty((spec.e, spec.expert_capacity), device=tokens.device, dtype=torch.float32)
        processor_ready = torch.empty((num_programs,), device=tokens.device, dtype=torch.int32)
        processor_mailboxes = torch.empty((num_programs,), device=tokens.device, dtype=torch.int32)
        processor_seen = torch.empty((num_programs * 3,), device=tokens.device, dtype=torch.int32)
        barriers = torch.empty((6,), device=tokens.device, dtype=torch.int32)
        init_flag = torch.empty((1,), device=tokens.device, dtype=torch.int32)
        launch_epoch = random.randint(1, (1 << 30) - 1)
        if tdm_wmma_eligible:
            block_m = 16
            tdm_block_e = spec.e
            hidden = torch.empty(
                (spec.e, spec.expert_capacity, spec.i),
                device=tokens.device,
                dtype=tokens.dtype,
            )
            route_blocks = _ceil_div(spec.expert_capacity, block_m)
            total_tasks = spec.e * route_blocks * ((spec.i // 64) + (spec.h // 64))
            task_queue = torch.empty((total_tasks,), device=tokens.device, dtype=torch.int32)
            task_head = torch.empty((1,), device=tokens.device, dtype=torch.int32)
            task_tail = torch.empty((1,), device=tokens.device, dtype=torch.int32)
            gemm0_pending = torch.empty((1,), device=tokens.device, dtype=torch.int32)
            tile_sync = torch.empty((spec.e * route_blocks,), device=tokens.device, dtype=torch.int32)
            debug_trace_enabled = os.environ.get("FLASHMOE_DEBUG_STATE", "0") == "1"
            debug_sched_enabled = os.environ.get("FLASHMOE_DEBUG_SCHED", "0") == "1"
            progress_heartbeat = os.environ.get("FLASHMOE_PROGRESS_HEARTBEAT", "0") == "1"
            debug_len = max(64, 56 + num_programs)
            debug_state = (
                torch.zeros((debug_len,), dtype=torch.int32, pin_memory=True)
                if debug_trace_enabled or progress_heartbeat
                else torch.empty((1,), device=tokens.device, dtype=torch.int32)
            )
            setattr(
                forward_megakernel,
                "last_debug_state",
                debug_state if debug_trace_enabled or progress_heartbeat else None,
            )
            _persistent_tdm_wmma_kernel[(num_programs,)](
                tokens,
                gate_weights,
                gate_weights,
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
                task_queue,
                task_head,
                task_tail,
                gemm0_pending,
                tile_sync,
                processor_mailboxes,
                processor_seen,
                barriers,
                init_flag,
                debug_state,
                launch_epoch,
                S=spec.s,
                H=spec.h,
                I=spec.i,
                E=spec.e,
                EC=spec.expert_capacity,
                TOP_K=spec.top_k,
                PRECOMPUTED_ROUTING=False,
                ZERO_OUTPUT=True,
                ACTIVATION=spec.activation,
                GATED=spec.gated,
                BLOCK_E=tdm_block_e,
                BLOCK_H=spec.block_h,
                BLOCK_M=block_m,
                BLOCK_N=64,
                NUM_PROGRAMS=num_programs,
                NUM_WARPS=spec.num_warps,
                DEBUG=debug_trace_enabled or progress_heartbeat,
                DEBUG_SCHED=debug_sched_enabled,
                num_warps=spec.num_warps,
            )
            if progress_heartbeat and os.environ.get("FLASHMOE_HOST_POLL", "1") != "0":
                _poll_tdm_heartbeat(
                    debug_state,
                    num_programs,
                    label="Gluon megakernel",
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


def _validate_topk_common(
    tokens: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_up: torch.Tensor,
    expert_down: torch.Tensor,
    bias_up: torch.Tensor,
    bias_down: torch.Tensor,
    *,
    expert_capacity: Optional[int],
    activation: int,
    expert_up_v: Optional[torch.Tensor],
    bias_up_v: Optional[torch.Tensor],
    block_h: Optional[int],
    num_warps: int,
) -> MegakernelSpec:
    if tokens.ndim != 2:
        raise ValueError("tokens must be [S,H]")
    if topk_ids.ndim != 2 or topk_weights.ndim != 2:
        raise ValueError("topk_ids and topk_weights must be [S,top_k]")
    if topk_ids.shape != topk_weights.shape or topk_ids.shape[0] != tokens.shape[0]:
        raise ValueError("top-k routing tensors must match the token count")
    if expert_up.ndim != 3 or expert_down.ndim != 3:
        raise ValueError("expert_up must be [E,H,I], expert_down must be [E,I,H]")
    if bias_up.ndim != 2 or bias_down.ndim != 2:
        raise ValueError("bias_up must be [E,I] and bias_down must be [E,H]")
    if tokens.device.type != "cuda":
        raise ValueError("Gluon megakernel expects HIP/CUDA tensors on device")
    if topk_ids.device != tokens.device or topk_weights.device != tokens.device:
        raise ValueError("top-k routing tensors must live on the token device")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must be int32 or int64")
    if topk_weights.dtype != torch.float32:
        raise ValueError("topk_weights must be float32")
    if activation not in (ACT_IDENTITY, ACT_SILU, ACT_GELU, ACT_RELU):
        raise ValueError("unsupported activation")
    if tokens.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Gluon megakernel currently expects fp16 or bf16 tokens")
    if expert_up.dtype != tokens.dtype or expert_down.dtype != tokens.dtype:
        raise ValueError("tokens, expert_up, and expert_down must have the same dtype")
    if bias_up.dtype != tokens.dtype or bias_down.dtype != tokens.dtype:
        raise ValueError("bias tensors must have the same dtype as tokens")

    s, h = tokens.shape
    top_k = topk_ids.shape[1]
    e, hup, i = expert_up.shape
    ed, idown, hdown = expert_down.shape
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if hup != h or (ed, idown, hdown) != (e, i, h):
        raise ValueError("token and expert tensor shapes are inconsistent")
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

    if expert_capacity is None:
        expert_capacity = s * top_k
    if expert_capacity <= 0:
        raise ValueError("expert_capacity must be positive")
    if num_warps != 4:
        raise ValueError("precomputed-routing TDM/WMMA path requires num_warps=4")

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


def forward_megakernel_from_topk(
    tokens: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_up: torch.Tensor,
    expert_down: torch.Tensor,
    bias_up: torch.Tensor,
    bias_down: torch.Tensor,
    *,
    expert_capacity: Optional[int] = None,
    activation: int = ACT_IDENTITY,
    expert_up_v: Optional[torch.Tensor] = None,
    bias_up_v: Optional[torch.Tensor] = None,
    block_h: Optional[int] = None,
    num_warps: int = 4,
    num_programs: int = 8,
    return_counts: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    spec = _validate_topk_common(
        tokens,
        topk_ids,
        topk_weights,
        expert_up,
        expert_down,
        bias_up,
        bias_down,
        expert_capacity=expert_capacity,
        activation=activation,
        expert_up_v=expert_up_v,
        bias_up_v=bias_up_v,
        block_h=block_h,
        num_warps=num_warps,
    )
    if num_programs < 2:
        raise ValueError("persistent Gluon scheduling requires at least two programs")
    if spec.h % 64 != 0 or spec.i % 64 != 0:
        raise ValueError("precomputed-routing TDM/WMMA path requires H and I divisible by 64")
    num_programs = _resolve_tdm_num_programs(spec.dtype, spec.block_h, num_programs)

    up_v_arg = expert_up_v if expert_up_v is not None else expert_up
    bias_up_v_arg = bias_up_v if bias_up_v is not None else bias_up
    out = torch.empty((spec.s, spec.h), device=tokens.device, dtype=torch.float32)
    hidden = torch.empty(
        (spec.e, spec.expert_capacity, spec.i),
        device=tokens.device,
        dtype=tokens.dtype,
    )
    block_m = 16
    route_blocks = _ceil_div(spec.expert_capacity, block_m)
    total_tasks = spec.e * route_blocks * ((spec.i // 64) + (spec.h // 64))
    expert_counts = torch.empty((spec.e,), device=tokens.device, dtype=torch.int32)
    route_tokens = torch.empty((spec.e, spec.expert_capacity), device=tokens.device, dtype=torch.int32)
    route_probs = torch.empty((spec.e, spec.expert_capacity), device=tokens.device, dtype=torch.float32)
    task_queue = torch.empty((total_tasks,), device=tokens.device, dtype=torch.int32)
    task_head = torch.empty((1,), device=tokens.device, dtype=torch.int32)
    task_tail = torch.empty((1,), device=tokens.device, dtype=torch.int32)
    gemm0_pending = torch.empty((1,), device=tokens.device, dtype=torch.int32)
    tile_sync = torch.empty((spec.e * route_blocks,), device=tokens.device, dtype=torch.int32)
    processor_mailboxes = torch.empty((num_programs,), device=tokens.device, dtype=torch.int32)
    processor_seen = torch.empty((num_programs * 3,), device=tokens.device, dtype=torch.int32)
    barriers = torch.empty((6,), device=tokens.device, dtype=torch.int32)
    init_flag = torch.zeros((1,), device=tokens.device, dtype=torch.int32)
    debug_trace_enabled = os.environ.get("FLASHMOE_DEBUG_STATE", "0") == "1"
    debug_sched_enabled = os.environ.get("FLASHMOE_DEBUG_SCHED", "0") == "1"
    # On gfx1250 the precomputed-routing persistent scheduler can occasionally
    # fail to retire unless the host polls a heartbeat in pinned memory. Keep
    # the stabilizer on by default for this path; profiling can still disable it.
    progress_heartbeat = os.environ.get("FLASHMOE_PROGRESS_HEARTBEAT", "1") != "0"
    debug_len = max(64, 56 + num_programs)
    debug_state = (
        torch.zeros((debug_len,), dtype=torch.int32, pin_memory=True)
        if debug_trace_enabled or progress_heartbeat
        else torch.empty((1,), device=tokens.device, dtype=torch.int32)
    )
    setattr(
        forward_megakernel_from_topk,
        "last_debug_state",
        debug_state if debug_trace_enabled or progress_heartbeat else None,
    )
    # Keep this scalar stable so the JIT cache can be reused across calls.
    # init_flag is zeroed above, so a constant non-zero epoch is sufficient.
    launch_epoch = 1
    _persistent_tdm_wmma_kernel[(num_programs,)](
        tokens,
        tokens,
        topk_ids,
        topk_weights,
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
        task_queue,
        task_head,
        task_tail,
        gemm0_pending,
        tile_sync,
        processor_mailboxes,
        processor_seen,
        barriers,
        init_flag,
        debug_state,
        launch_epoch,
        S=spec.s,
        H=spec.h,
        I=spec.i,
        E=spec.e,
        EC=spec.expert_capacity,
        TOP_K=spec.top_k,
        PRECOMPUTED_ROUTING=True,
        ZERO_OUTPUT=True,
        ACTIVATION=spec.activation,
        GATED=spec.gated,
        BLOCK_E=spec.e,
        BLOCK_H=spec.block_h,
        BLOCK_M=block_m,
        BLOCK_N=64,
        NUM_PROGRAMS=num_programs,
        NUM_WARPS=spec.num_warps,
        DEBUG=debug_trace_enabled or progress_heartbeat,
        DEBUG_SCHED=debug_sched_enabled,
        num_warps=spec.num_warps,
    )
    if progress_heartbeat and os.environ.get("FLASHMOE_HOST_POLL", "1") != "0":
        _poll_tdm_heartbeat(
            debug_state,
            num_programs,
            label="Gluon precomputed-routing kernel",
        )
    if return_counts:
        return out, expert_counts
    return out


def forward_megakernel_rocshmem(
    tokens: torch.Tensor,
    gate_weights: torch.Tensor,
    local_expert_up: torch.Tensor,
    local_expert_down: torch.Tensor,
    bias_up: torch.Tensor,
    bias_down: torch.Tensor,
    *,
    ctx,
    top_k: int,
    activation: int = ACT_IDENTITY,
    local_expert_up_v: Optional[torch.Tensor] = None,
    bias_up_v: Optional[torch.Tensor] = None,
    block_h: Optional[int] = None,
    num_warps: int = 4,
    num_programs: int = 4,
) -> torch.Tensor:
    if tokens.ndim != 2 or gate_weights.ndim != 2:
        raise ValueError("tokens must be [S,H] and gate_weights must be [H,E]")
    if local_expert_up.ndim != 3 or local_expert_down.ndim != 3:
        raise ValueError("local expert weights must be [local_experts,H,I] and [local_experts,I,H]")
    if tokens.device.type != "cuda":
        raise ValueError("rocSHMEM Gluon megakernel expects device tensors")
    if tokens.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("rocSHMEM Gluon megakernel expects fp16 or bf16 tokens")
    if top_k not in (1, 2):
        raise ValueError("rocSHMEM Gluon megakernel currently supports top_k in {1, 2}")
    if activation not in (ACT_IDENTITY, ACT_SILU, ACT_GELU, ACT_RELU):
        raise ValueError("unsupported activation")
    if num_programs < 3:
        raise ValueError("rocSHMEM persistent scheduling requires scheduler, dispatch, and compute programs")
    if num_warps != 4:
        raise ValueError("gfx1250 Gluon rocSHMEM path requires num_warps=4")

    s, h = tokens.shape
    hw, e = gate_weights.shape
    nlx, hup, i = local_expert_up.shape
    nd, idown, hdown = local_expert_down.shape
    if hw != h or (nd, idown, hdown) != (nlx, i, h) or hup != h:
        raise ValueError("distributed token, gate, and local expert tensor shapes are inconsistent")
    if bias_up.shape != (nlx, i) or bias_down.shape != (nlx, h):
        raise ValueError("distributed bias tensor shapes are inconsistent")
    if e != ctx.world_size * ctx.local_experts or nlx != ctx.local_experts:
        raise ValueError("gate expert count must equal world_size * local_experts")
    if ctx.expert_capacity <= 0 or ctx.hidden_size != h or ctx.dtype != tokens.dtype:
        raise ValueError("rocSHMEM context does not match token dtype/hidden size")
    if ctx.expert_capacity >= 1048576:
        raise ValueError("rocSHMEM Gluon signal payload requires expert_capacity < 1048576")
    if gate_weights.dtype != tokens.dtype or local_expert_up.dtype != tokens.dtype or local_expert_down.dtype != tokens.dtype:
        raise ValueError("tokens, gate weights, and local expert weights must have the same dtype")
    if bias_up.dtype != tokens.dtype or bias_down.dtype != tokens.dtype:
        raise ValueError("bias tensors must have the same dtype as tokens")

    gated = local_expert_up_v is not None
    if gated:
        if bias_up_v is None:
            raise ValueError("bias_up_v is required when local_expert_up_v is provided")
        if local_expert_up_v.shape != local_expert_up.shape or bias_up_v.shape != bias_up.shape:
            raise ValueError("gated local expert tensors must match the up projection shapes")
        if local_expert_up_v.dtype != tokens.dtype or bias_up_v.dtype != tokens.dtype:
            raise ValueError("gated tensors must have the same dtype as tokens")
    elif bias_up_v is not None:
        raise ValueError("bias_up_v requires local_expert_up_v")

    resolved_block_h = _next_power_of_2(h) if block_h is None else block_h
    if resolved_block_h < h:
        raise ValueError("block_h must cover the full hidden dimension")
    if h % 64 != 0 or i % 64 != 0:
        raise ValueError("rocSHMEM Gluon megakernel requires H and I to be divisible by 64 for TDM/WMMA")
    if num_warps != 4:
        raise ValueError("rocSHMEM Gluon megakernel requires num_warps=4 for warp-specialized TDM/WMMA")
    up_v_arg = local_expert_up_v if local_expert_up_v is not None else local_expert_up
    bias_up_v_arg = bias_up_v if bias_up_v is not None else bias_up

    rocshmem.register_kernel(_persistent_rocshmem_tdm_wmma_kernel)
    rocshmem.install_module_init(ctx.runtime)

    out = torch.empty((s, h), device=tokens.device, dtype=torch.float32)
    hidden = torch.empty((ctx.world_size * ctx.local_experts, ctx.expert_capacity, i), device=tokens.device, dtype=tokens.dtype)
    compute_done = torch.empty((ctx.world_size * ctx.world_size, ctx.local_experts), device=tokens.device, dtype=torch.int32)
    block_m = 16
    route_blocks = _ceil_div(ctx.expert_capacity, block_m)
    total_channels = ctx.world_size * ctx.local_experts
    gemm0_tasks = total_channels * route_blocks * (i // 64)
    gemm1_tasks = total_channels * route_blocks * (h // 64)
    combine_tasks = total_channels * route_blocks * (h // 64)
    total_tasks = gemm0_tasks + gemm1_tasks + combine_tasks
    task_queue = torch.empty((total_tasks,), device=tokens.device, dtype=torch.int32)
    task_tail = torch.empty((1,), device=tokens.device, dtype=torch.int32)
    publish_queue = torch.empty((total_channels,), device=tokens.device, dtype=torch.int32)
    publish_tail = torch.empty((1,), device=tokens.device, dtype=torch.int32)
    tile_sync = torch.empty((total_channels * route_blocks,), device=tokens.device, dtype=torch.int32)
    dispatch_seen = torch.empty((total_channels,), device=tokens.device, dtype=torch.int32)
    result_seen = torch.empty((total_channels,), device=tokens.device, dtype=torch.int32)
    task_bound = torch.empty((1,), device=tokens.device, dtype=torch.int32)
    tasks_done = torch.empty((1,), device=tokens.device, dtype=torch.int32)
    os_done = torch.empty((1,), device=tokens.device, dtype=torch.int32)
    dispatch_done = torch.empty((1,), device=tokens.device, dtype=torch.int32)
    processor_ready = torch.empty((num_programs,), device=tokens.device, dtype=torch.int32)
    processor_mailboxes = torch.empty((num_programs,), device=tokens.device, dtype=torch.int32)
    processor_seen = torch.empty((num_programs,), device=tokens.device, dtype=torch.int32)
    barriers = torch.empty((3,), device=tokens.device, dtype=torch.int32)
    init_flag = torch.zeros((1,), device=tokens.device, dtype=torch.int32)
    debug_trace_enabled = os.environ.get("FLASHMOE_DEBUG_STATE", "0") == "1"
    progress_heartbeat = debug_trace_enabled
    result_elem_bytes = torch.empty((), dtype=ctx.result_values.dtype).element_size()
    debug_state = (
        torch.zeros((64,), dtype=torch.int32, pin_memory=True)
        if progress_heartbeat
        else torch.zeros((1,), device=tokens.device, dtype=torch.int32)
    )
    setattr(forward_megakernel_rocshmem, "last_debug_state", debug_state if debug_trace_enabled else None)
    launch_epoch = ctx.next_epoch()
    block_e = max(16, _next_power_of_2(e))

    for _preload_rank in range(ctx.world_size):
        if ctx.rank == _preload_rank:
            _preload_kernel = _persistent_rocshmem_tdm_wmma_kernel.warmup(
                tokens,
                gate_weights,
                local_expert_up,
                bias_up,
                up_v_arg,
                bias_up_v_arg,
                local_expert_down,
                bias_down,
                out,
                hidden,
                ctx.dispatch_tokens,
                ctx.dispatch_token_ids,
                ctx.dispatch_probs,
                ctx.dispatch_counts,
                ctx.dispatch_signals,
                ctx.result_values,
                ctx.result_counts,
                ctx.result_signals,
                compute_done,
                task_queue,
                task_tail,
                publish_queue,
                publish_tail,
                tile_sync,
                dispatch_seen,
                result_seen,
                task_bound,
                tasks_done,
                os_done,
                dispatch_done,
                processor_ready,
                processor_mailboxes,
                processor_seen,
                barriers,
                init_flag,
                debug_state,
                launch_epoch,
                S=s,
                H=h,
                I=i,
                E=e,
                WORLD=ctx.world_size,
                NLX=ctx.local_experts,
                EC=ctx.expert_capacity,
                TOP_K=top_k,
                BLOCK_E=block_e,
                BLOCK_H=resolved_block_h,
                BLOCK_M=block_m,
                BLOCK_N=64,
                ACTIVATION=activation,
                GATED=gated,
                NUM_PROGRAMS=num_programs,
                NUM_WARPS=num_warps,
                RESULT_ELEM_BYTES=result_elem_bytes,
                DEBUG=progress_heartbeat,
                num_warps=num_warps,
                extern_libs=rocshmem.extern_libs("gfx1250"),
                grid=(num_programs,),
            )
            if _preload_kernel is not None:
                _ = _preload_kernel.run
                if not getattr(_preload_kernel, "_flashmoe_rocshmem_module_initialized", False):
                    ctx.runtime.hipmodule_init(int(_preload_kernel.module))
                    setattr(_preload_kernel, "_flashmoe_rocshmem_module_initialized", True)
        ctx.runtime.barrier_all()

    _persistent_rocshmem_tdm_wmma_kernel[(num_programs,)](
        tokens,
        gate_weights,
        local_expert_up,
        bias_up,
        up_v_arg,
        bias_up_v_arg,
        local_expert_down,
        bias_down,
        out,
        hidden,
        ctx.dispatch_tokens,
        ctx.dispatch_token_ids,
        ctx.dispatch_probs,
        ctx.dispatch_counts,
        ctx.dispatch_signals,
        ctx.result_values,
        ctx.result_counts,
        ctx.result_signals,
        compute_done,
        task_queue,
        task_tail,
        publish_queue,
        publish_tail,
        tile_sync,
        dispatch_seen,
        result_seen,
        task_bound,
        tasks_done,
        os_done,
        dispatch_done,
        processor_ready,
        processor_mailboxes,
        processor_seen,
        barriers,
        init_flag,
        debug_state,
        launch_epoch,
        S=s,
        H=h,
        I=i,
        E=e,
        WORLD=ctx.world_size,
        NLX=ctx.local_experts,
        EC=ctx.expert_capacity,
        TOP_K=top_k,
        BLOCK_E=block_e,
        BLOCK_H=resolved_block_h,
        BLOCK_M=block_m,
        BLOCK_N=64,
        ACTIVATION=activation,
        GATED=gated,
        NUM_PROGRAMS=num_programs,
        NUM_WARPS=num_warps,
        RESULT_ELEM_BYTES=result_elem_bytes,
        DEBUG=progress_heartbeat,
        num_warps=num_warps,
        extern_libs=rocshmem.extern_libs("gfx1250"),
    )
    return out


__all__ = [
    "ACT_IDENTITY",
    "ACT_SILU",
    "ACT_GELU",
    "ACT_RELU",
    "MegakernelSpec",
    "forward_megakernel",
    "forward_megakernel_rocshmem",
    "forward_scalar_top1_debug",
]
