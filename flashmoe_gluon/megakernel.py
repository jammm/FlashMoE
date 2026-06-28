from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.gfx1250 import mbarrier, tdm


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
):
    for i in gl.static_range(0, NUM_BUFFERS):
        mbarrier.init(load_empty_bars.index(i), count=COMPUTE_WARPS * 32)
        mbarrier.init(load_ready_bars.index(i), count=PRODUCER_WARPS)


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
):
    _init_ws_load_mbarriers(load_empty_bars, load_ready_bars, NUM_BUFFERS, PRODUCER_WARPS, COMPUTE_WARPS)
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
    for h_tile in gl.static_range(0, H // BLOCK_N):
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

    if GATED:
        v_empty_counter = _WsPhaseCounter.create(NUM_BUFFERS, NUM_BUFFERS)
        for h_tile_v in gl.static_range(0, H // BLOCK_N):
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
    for h_tile in gl.static_range(0, H // BLOCK_N):
        buffer_idx = h_tile % NUM_BUFFERS
        ready_bar = load_ready_bars.index(buffer_idx)
        empty_bar = load_empty_bars.index(buffer_idx)
        mbarrier.wait(ready_bar, ready_counter.phase())
        x_frag = x_buffer.index(buffer_idx).load(layout=dot_a)
        w_frag = w_buffer.index(buffer_idx).load(layout=dot_b)
        acc = gl.amd.gfx1250.wmma(x_frag, w_frag, acc)
        mbarrier.arrive(empty_bar, count=1)
        ready_counter = ready_counter.next()

    bias_layout: gl.constexpr = gl.SliceLayout(0, wmma_layout)
    offs_i = base_i + gl.arange(0, BLOCK_N, layout=bias_layout)
    bias = gl.load(bias_up + expert_idx * I + offs_i).to(gl.float32)
    hidden_acc = _apply_activation(acc + gl.convert_layout(gl.expand_dims(bias, 0), wmma_layout), ACTIVATION)

    if GATED:
        v_ready_counter = _WsPhaseCounter.create(0, NUM_BUFFERS)
        acc_v = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=wmma_layout)
        for h_tile_v in gl.static_range(0, H // BLOCK_N):
            buffer_idx_v = h_tile_v % NUM_BUFFERS
            ready_bar_v = v_load_ready_bars.index(buffer_idx_v)
            empty_bar_v = v_load_empty_bars.index(buffer_idx_v)
            mbarrier.wait(ready_bar_v, v_ready_counter.phase())
            xv_frag = xv_buffer.index(buffer_idx_v).load(layout=dot_a)
            wv_frag = wv_buffer.index(buffer_idx_v).load(layout=dot_b)
            acc_v = gl.amd.gfx1250.wmma(xv_frag, wv_frag, acc_v)
            mbarrier.arrive(empty_bar_v, count=1)
            v_ready_counter = v_ready_counter.next()
        bias_v = gl.load(bias_up_v + expert_idx * I + offs_i).to(gl.float32)
        acc_v += gl.convert_layout(gl.expand_dims(bias_v, 0), wmma_layout)
        hidden_acc *= acc_v

    hidden_dtype: gl.constexpr = hidden_buffer.dtype
    hidden_buffer.index(0).store(hidden_acc.to(hidden_dtype))
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
    for i_tile in gl.static_range(0, I // BLOCK_N):
        buffer_idx = i_tile % NUM_BUFFERS
        empty_bar = load_empty_bars.index(buffer_idx)
        ready_bar = load_ready_bars.index(buffer_idx)
        mbarrier.wait(empty_bar, empty_counter.phase())
        tdm.async_load(hidden_desc, [base_slot, i_tile * BLOCK_N], h_buffer.index(buffer_idx))
        tdm.async_load(down_desc, [i_tile * BLOCK_N, base_h], down_buffer.index(buffer_idx), mbarrier=ready_bar)
        empty_counter = empty_counter.next()


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
    for i_tile in gl.static_range(0, I // BLOCK_N):
        buffer_idx = i_tile % NUM_BUFFERS
        ready_bar = load_ready_bars.index(buffer_idx)
        empty_bar = load_empty_bars.index(buffer_idx)
        mbarrier.wait(ready_bar, ready_counter.phase())
        h_frag = h_buffer.index(buffer_idx).load(layout=dot_a)
        down_frag = down_buffer.index(buffer_idx).load(layout=dot_b)
        acc = gl.amd.gfx1250.wmma(h_frag, down_frag, acc)
        mbarrier.arrive(empty_bar, count=1)
        ready_counter = ready_counter.next()
    acc_buffer.index(0).store(acc)
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
            sem="relaxed",
            scope="gpu",
            mask=active_2d,
        )
    mbarrier.arrive(acc_empty_bar, count=1)



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
def _scheduler_run_queue(
    processor_ready,
    processor_mailboxes,
    task_queue,
    task_tail,
    TOTAL_TASKS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
):
    pid = gl.program_id(0)
    if pid == 0:
        next_slot = pid * 0
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
                    tail = gl.atomic_add(task_tail, 0, sem="acquire", scope="gpu")
                    if next_slot < tail:
                        task = gl.atomic_add(task_queue + next_slot, 0, sem="acquire", scope="gpu")
                        if task >= 0:
                            next_slot += 1
                            gl.atomic_xchg(
                                processor_mailboxes + program_id,
                                task,
                                sem="release",
                                scope="gpu",
                            )
                        else:
                            gl.atomic_xchg(processor_ready + program_id, 1, sem="release", scope="gpu")
                    else:
                        if next_slot >= TOTAL_TASKS:
                            gl.atomic_xchg(
                                processor_mailboxes + program_id,
                                TOTAL_TASKS,
                                sem="release",
                                scope="gpu",
                            )
                            stopped += 1
                        else:
                            gl.atomic_xchg(processor_ready + program_id, 1, sem="release", scope="gpu")


@gluon.jit
def _init_dynamic_task_queue(
    task_queue,
    task_tail,
    tile_sync,
    GEMM0_TASKS: gl.constexpr,
    TOTAL_TASKS: gl.constexpr,
    SYNC_SLOTS: gl.constexpr,
):
    pid = gl.program_id(0)
    if pid == 0:
        slot = pid * 0
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
        gl.atomic_xchg(task_tail, GEMM0_TASKS, sem="release", scope="gpu")


@gluon.jit
def _enqueue_task(task_queue, task_tail, task):
    slot = gl.atomic_add(task_tail, 1, sem="relaxed", scope="gpu")
    gl.atomic_xchg(task_queue + slot, task, sem="release", scope="gpu")


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
    task_queue,
    task_tail,
    tile_sync,
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
    _init_dynamic_task_queue(task_queue, task_tail, tile_sync, gemm0_tasks, total_tasks, sync_slots)
    _persistent_phase_barrier(barriers, 2, NUM_PROGRAMS)

    if pid == 0:
        _scheduler_run_queue(processor_ready, processor_mailboxes, task_queue, task_tail, total_tasks, NUM_PROGRAMS)
    else:
        task_id = _processor_next_task(processor_ready, processor_mailboxes)
        while task_id < total_tasks:
            if task_id < gemm0_tasks:
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
                load_empty_bars = gl.allocate_shared_memory(gl.int64, [NUM_BUFFERS, 1], mbarrier.MBarrierLayout())
                load_ready_bars = gl.allocate_shared_memory(gl.int64, [NUM_BUFFERS, 1], mbarrier.MBarrierLayout())
                xv_buffer = x_buffer
                wv_buffer = w_buffer
                v_load_empty_bars = load_empty_bars
                v_load_ready_bars = load_ready_bars
                if GATED:
                    xv_buffer = gl.allocate_shared_memory(
                        tokens.dtype.element_ty,
                        shape=[NUM_BUFFERS, BLOCK_M, BLOCK_N],
                        layout=shared_a_layout,
                    )
                    wv_buffer = gl.allocate_shared_memory(
                        expert_up_v.dtype.element_ty,
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
                )
                gl.warp_specialize([
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
                        _gemm0_ws_producer,
                        (
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
                            S,
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
                sync_idx = expert_idx * route_blocks + route_block
                done_tiles = gl.atomic_add(tile_sync + sync_idx, 1, sem="release", scope="gpu") + 1
                if done_tiles == i_tiles:
                    for h_enqueue in gl.static_range(0, h_tiles):
                        downstream_task = gemm0_tasks + expert_idx * route_blocks * h_tiles + route_block * h_tiles + h_enqueue
                        _enqueue_task(task_queue, task_tail, downstream_task)
            else:
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
                )
                gl.warp_specialize([
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
                ], [COMPUTE_WARPS, PRODUCER_WARPS])
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
            route_blocks = _ceil_div(spec.expert_capacity, block_m)
            total_tasks = spec.e * route_blocks * ((spec.i // 64) + (spec.h // 64))
            task_queue = torch.empty((total_tasks,), device=tokens.device, dtype=torch.int32)
            task_tail = torch.empty((1,), device=tokens.device, dtype=torch.int32)
            tile_sync = torch.empty((spec.e * route_blocks,), device=tokens.device, dtype=torch.int32)
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
                task_queue,
                task_tail,
                tile_sync,
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
