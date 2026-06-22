from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl


ACT_IDENTITY = 0
ACT_SILU = 1
ACT_GELU = 2
ACT_RELU = 3


@dataclass
class RoutingResult:
    expert_counts: torch.Tensor
    token_indices: torch.Tensor
    route_probs: torch.Tensor
    routed_tokens: torch.Tensor
    routing_logits: Optional[torch.Tensor]
    expert_capacity: int
    round_ec: int


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


@triton.jit
def _activation(x, act: tl.constexpr, swish_alpha: tl.constexpr, swish_beta: tl.constexpr):
    if act == 0:
        return x
    if act == 3:
        return tl.maximum(x, 0.0)
    if act == 1:
        z = x * swish_beta
        return swish_alpha * z / (1.0 + tl.exp(-z))
    # GELU sigmoid approximation. Triton's HIP language surface does not expose
    # tl.tanh in this environment, and all branches are type-checked at compile time.
    return x / (1.0 + tl.exp(-1.702 * x))


@triton.jit
def _store_routed_token(
    tokens,
    routed_tokens,
    token_id,
    expert_id,
    slot,
    H: tl.constexpr,
    ROUND_EC: tl.constexpr,
    COPY_BLOCK_H: tl.constexpr,
):
    base_out = expert_id * ROUND_EC * H + slot * H
    base_in = token_id * H
    for h0 in range(0, H, COPY_BLOCK_H):
        offs_h = h0 + tl.arange(0, COPY_BLOCK_H)
        vals = tl.load(tokens + base_in + offs_h, mask=offs_h < H, other=0.0)
        tl.store(routed_tokens + base_out + offs_h, vals, mask=offs_h < H)


@triton.jit
def _router_kernel(
    tokens,
    weights,
    expert_counts,
    token_indices,
    route_probs,
    routed_tokens,
    routing_logits,
    S: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    TOP_K: tl.constexpr,
    EC: tl.constexpr,
    ROUND_EC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_E: tl.constexpr,
    COPY_BLOCK_H: tl.constexpr,
    RETURN_LOGITS: tl.constexpr,
):
    token_id = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    logits = tl.zeros((BLOCK_E,), dtype=tl.float32)

    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        t = tl.load(tokens + token_id * H + offs_h, mask=offs_h < H, other=0.0).to(tl.float32)
        w = tl.load(
            weights + offs_h[:, None] * E + offs_e[None, :],
            mask=(offs_h[:, None] < H) & (offs_e[None, :] < E),
            other=0.0,
        ).to(tl.float32)
        logits += tl.sum(t[:, None] * w, axis=0)

    valid_e = offs_e < E
    masked_logits = tl.where(valid_e, logits, -float("inf"))
    row_max = tl.max(masked_logits, axis=0)
    exp_logits = tl.exp(masked_logits - row_max)
    denom = tl.sum(tl.where(valid_e, exp_logits, 0.0), axis=0)
    probs = exp_logits / denom

    if RETURN_LOGITS:
        tl.store(routing_logits + token_id * E + offs_e, probs, mask=valid_e)

    top1_idx = tl.argmax(probs, axis=0)
    top1_prob = tl.max(probs, axis=0)

    if TOP_K == 1:
        slot = tl.atomic_add(expert_counts + top1_idx, 1, sem="relaxed")
        if slot < EC:
            out = top1_idx * ROUND_EC + slot
            tl.store(token_indices + out, token_id)
            tl.store(route_probs + out, 1.0)
            _store_routed_token(tokens, routed_tokens, token_id, top1_idx, slot, H, ROUND_EC, COPY_BLOCK_H)
    else:
        probs2 = tl.where(offs_e == top1_idx, -1.0, probs)
        top2_idx = tl.argmax(probs2, axis=0)
        top2_prob = tl.max(probs2, axis=0)
        norm = top1_prob + top2_prob

        slot1 = tl.atomic_add(expert_counts + top1_idx, 1, sem="relaxed")
        if slot1 < EC:
            out1 = top1_idx * ROUND_EC + slot1
            tl.store(token_indices + out1, token_id)
            tl.store(route_probs + out1, top1_prob / norm)
            _store_routed_token(tokens, routed_tokens, token_id, top1_idx, slot1, H, ROUND_EC, COPY_BLOCK_H)

        slot2 = tl.atomic_add(expert_counts + top2_idx, 1, sem="relaxed")
        if slot2 < EC:
            out2 = top2_idx * ROUND_EC + slot2
            tl.store(token_indices + out2, token_id)
            tl.store(route_probs + out2, top2_prob / norm)
            _store_routed_token(tokens, routed_tokens, token_id, top2_idx, slot2, H, ROUND_EC, COPY_BLOCK_H)


@triton.jit
def _up_project_kernel(
    routed_tokens,
    expert_up,
    expert_up_v,
    bias_up,
    bias_up_v,
    inter_out,
    H: tl.constexpr,
    I: tl.constexpr,
    EC: tl.constexpr,
    ROUND_EC: tl.constexpr,
    ACT: tl.constexpr,
    GATED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
    swish_alpha: tl.constexpr,
    swish_beta: tl.constexpr,
):
    expert_id = tl.program_id(0)
    m_tile = tl.program_id(1)
    i_tile = tl.program_id(2)

    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = i_tile * BLOCK_I + tl.arange(0, BLOCK_I)
    valid_m = offs_m < EC

    acc = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    acc_v = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    for h0 in range(0, H, BLOCK_K):
        offs_k = h0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            routed_tokens + expert_id * ROUND_EC * H + offs_m[:, None] * H + offs_k[None, :],
            mask=valid_m[:, None] & (offs_k[None, :] < H),
            other=0.0,
        )
        w = tl.load(
            expert_up + expert_id * H * I + offs_k[:, None] * I + offs_i[None, :],
            mask=(offs_k[:, None] < H) & (offs_i[None, :] < I),
            other=0.0,
        )
        acc = tl.dot(x, w, acc)
        if GATED:
            wv = tl.load(
                expert_up_v + expert_id * H * I + offs_k[:, None] * I + offs_i[None, :],
                mask=(offs_k[:, None] < H) & (offs_i[None, :] < I),
                other=0.0,
            )
            acc_v = tl.dot(x, wv, acc_v)

    bu = tl.load(bias_up + expert_id * I + offs_i, mask=offs_i < I, other=0.0).to(tl.float32)
    acc += bu[None, :]
    if GATED:
        buv = tl.load(bias_up_v + expert_id * I + offs_i, mask=offs_i < I, other=0.0).to(tl.float32)
        inter = _activation(acc, ACT, swish_alpha, swish_beta) * (acc_v + buv[None, :])
    else:
        inter = _activation(acc, ACT, swish_alpha, swish_beta)

    ptrs = inter_out + expert_id * ROUND_EC * I + offs_m[:, None] * I + offs_i[None, :]
    mask = valid_m[:, None] & (offs_i[None, :] < I)
    tl.store(ptrs, inter, mask=mask)


@triton.jit
def _down_project_kernel(
    token_indices,
    route_probs,
    inter,
    expert_down,
    bias_down,
    output,
    H: tl.constexpr,
    I: tl.constexpr,
    EC: tl.constexpr,
    ROUND_EC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    expert_id = tl.program_id(0)
    m_tile = tl.program_id(1)
    h_tile = tl.program_id(2)

    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = h_tile * BLOCK_H + tl.arange(0, BLOCK_H)
    valid_m = offs_m < EC
    token_ids = tl.load(token_indices + expert_id * ROUND_EC + offs_m, mask=valid_m, other=0)
    probs = tl.load(route_probs + expert_id * ROUND_EC + offs_m, mask=valid_m, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    for i0 in range(0, I, BLOCK_I):
        offs_i = i0 + tl.arange(0, BLOCK_I)
        x = tl.load(
            inter + expert_id * ROUND_EC * I + offs_m[:, None] * I + offs_i[None, :],
            mask=valid_m[:, None] & (offs_i[None, :] < I),
            other=0.0,
        )
        w = tl.load(
            expert_down + expert_id * I * H + offs_i[:, None] * H + offs_h[None, :],
            mask=(offs_i[:, None] < I) & (offs_h[None, :] < H),
            other=0.0,
        )
        acc = tl.dot(x.to(tl.float16), w, acc)
    b = tl.load(bias_down + expert_id * H + offs_h, mask=offs_h < H, other=0.0).to(tl.float32)
    acc = (acc + b[None, :]) * probs[:, None]
    ptrs = output + token_ids[:, None] * H + offs_h[None, :]
    mask = valid_m[:, None] & (offs_h[None, :] < H)
    tl.atomic_add(ptrs, acc, sem="relaxed", mask=mask)


@triton.jit
def _single_dispatch_megakernel(
    tokens,
    gate_weights,
    expert_counts,
    expert_up,
    expert_up_v,
    bias_up,
    bias_up_v,
    expert_down,
    bias_down,
    output,
    S: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    E: tl.constexpr,
    TOP_K: tl.constexpr,
    EC: tl.constexpr,
    ACT: tl.constexpr,
    GATED: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    swish_alpha: tl.constexpr,
    swish_beta: tl.constexpr,
):
    token_id = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    offs_h = tl.arange(0, BLOCK_H)
    h_mask = offs_h < H

    logits = tl.zeros((BLOCK_E,), dtype=tl.float32)
    for h_abs in range(0, H):
        t = tl.load(tokens + token_id * H + h_abs).to(tl.float32)
        w = tl.load(gate_weights + h_abs * E + offs_e, mask=offs_e < E, other=0.0).to(tl.float32)
        logits += t * w

    valid_e = offs_e < E
    masked_logits = tl.where(valid_e, logits, -float("inf"))
    row_max = tl.max(masked_logits, axis=0)
    exp_logits = tl.exp(masked_logits - row_max)
    denom = tl.sum(tl.where(valid_e, exp_logits, 0.0), axis=0)
    probs = exp_logits / denom

    top1_idx = tl.argmax(probs, axis=0)
    top1_prob = tl.max(probs, axis=0)
    top2_idx = top1_idx
    top2_prob = top1_prob
    if TOP_K == 2:
        probs2 = tl.where(offs_e == top1_idx, -1.0, probs)
        top2_idx = tl.argmax(probs2, axis=0)
        top2_prob = tl.max(probs2, axis=0)

    norm = top1_prob + top2_prob
    route0_prob = 1.0
    route1_prob = 0.0
    if TOP_K == 2:
        route0_prob = top1_prob / norm
        route1_prob = top2_prob / norm

    out_vals = tl.zeros((BLOCK_H,), dtype=tl.float32)

    slot0 = tl.atomic_add(expert_counts + top1_idx, 1, sem="relaxed")
    if slot0 < EC:
        route_vals = tl.load(bias_down + top1_idx * H + offs_h, mask=h_mask, other=0.0).to(tl.float32)
        for i_abs in range(0, I):
            up_acc = tl.load(bias_up + top1_idx * I + i_abs).to(tl.float32)
            v_acc = tl.load(bias_up_v + top1_idx * I + i_abs).to(tl.float32)
            for h_abs in range(0, H):
                t = tl.load(tokens + token_id * H + h_abs).to(tl.float32)
                wu = tl.load(expert_up + top1_idx * H * I + h_abs * I + i_abs).to(tl.float32)
                up_acc += t * wu
                if GATED:
                    wv = tl.load(expert_up_v + top1_idx * H * I + h_abs * I + i_abs).to(tl.float32)
                    v_acc += t * wv
            inter = _activation(up_acc, ACT, swish_alpha, swish_beta)
            if GATED:
                inter *= v_acc
            wd = tl.load(
                expert_down + top1_idx * I * H + i_abs * H + offs_h,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
            route_vals += inter * wd
        out_vals += route_vals * route0_prob

    if TOP_K == 2:
        slot1 = tl.atomic_add(expert_counts + top2_idx, 1, sem="relaxed")
        if slot1 < EC:
            route_vals = tl.load(bias_down + top2_idx * H + offs_h, mask=h_mask, other=0.0).to(tl.float32)
            for i_abs in range(0, I):
                up_acc = tl.load(bias_up + top2_idx * I + i_abs).to(tl.float32)
                v_acc = tl.load(bias_up_v + top2_idx * I + i_abs).to(tl.float32)
                for h_abs in range(0, H):
                    t = tl.load(tokens + token_id * H + h_abs).to(tl.float32)
                    wu = tl.load(expert_up + top2_idx * H * I + h_abs * I + i_abs).to(tl.float32)
                    up_acc += t * wu
                    if GATED:
                        wv = tl.load(expert_up_v + top2_idx * H * I + h_abs * I + i_abs).to(tl.float32)
                        v_acc += t * wv
                inter = _activation(up_acc, ACT, swish_alpha, swish_beta)
                if GATED:
                    inter *= v_acc
                wd = tl.load(
                    expert_down + top2_idx * I * H + i_abs * H + offs_h,
                    mask=h_mask,
                    other=0.0,
                ).to(tl.float32)
                route_vals += inter * wd
            out_vals += route_vals * route1_prob

    tl.store(output + token_id * H + offs_h, out_vals, mask=h_mask)


def route(
    tokens: torch.Tensor,
    gate_weights: torch.Tensor,
    *,
    top_k: int,
    expert_capacity: Optional[int] = None,
    return_logits: bool = False,
    block_h: int = 64,
) -> RoutingResult:
    if tokens.ndim != 2 or gate_weights.ndim != 2:
        raise ValueError("tokens must be [S,H] and gate_weights must be [H,E]")
    if tokens.device.type != "cuda" or gate_weights.device.type != "cuda":
        raise ValueError("Triton path expects HIP/CUDA tensors on device")
    if tokens.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Triton path currently expects fp16 or bf16 tokens")
    if top_k not in (1, 2):
        raise ValueError("Triton router currently supports top_k in {1, 2}")

    s, h = tokens.shape
    hw, e = gate_weights.shape
    if hw != h:
        raise ValueError("gate_weights first dimension must equal token hidden dimension")
    if expert_capacity is None:
        expert_capacity = triton.cdiv(s * top_k, e)

    round_ec = triton.cdiv(expert_capacity, 16) * 16
    expert_counts = torch.empty((e,), device=tokens.device, dtype=torch.int32)
    expert_counts.zero_()
    token_indices = torch.empty((e, round_ec), device=tokens.device, dtype=torch.int32)
    token_indices.zero_()
    route_probs = torch.empty((e, round_ec), device=tokens.device, dtype=torch.float32)
    route_probs.zero_()
    routed_tokens = torch.empty((e, round_ec, h), device=tokens.device, dtype=tokens.dtype)
    routed_tokens.zero_()
    routing_logits = (
        torch.empty((s, e), device=tokens.device, dtype=torch.float32)
        if return_logits
        else None
    )
    dummy_logits = routing_logits if routing_logits is not None else route_probs

    block_e = _next_power_of_2(e)
    if block_e < 16:
        block_e = 16
    _router_kernel[(s,)](
        tokens,
        gate_weights,
        expert_counts,
        token_indices,
        route_probs,
        routed_tokens,
        dummy_logits,
        S=s,
        H=h,
        E=e,
        TOP_K=top_k,
        EC=expert_capacity,
        ROUND_EC=round_ec,
        BLOCK_H=block_h,
        BLOCK_E=block_e,
        COPY_BLOCK_H=block_h,
        RETURN_LOGITS=return_logits,
        num_warps=4,
    )
    return RoutingResult(
        expert_counts,
        token_indices,
        route_probs,
        routed_tokens,
        routing_logits,
        expert_capacity,
        round_ec,
    )


def forward(
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
    block_m: int = 16,
    block_h: int = 32,
    block_i: int = 32,
    block_k: int = 32,
    return_routing: bool = False,
):
    routing = route(
        tokens,
        gate_weights,
        top_k=top_k,
        expert_capacity=expert_capacity,
        return_logits=False,
        block_h=max(block_k, 32),
    )
    out = torch.empty((tokens.shape[0], tokens.shape[1]), device=tokens.device, dtype=torch.float32)
    out.zero_()
    forward_from_routing(
        tokens,
        routing,
        expert_up,
        expert_down,
        bias_up,
        bias_down,
        out=out,
        top_k=top_k,
        activation=activation,
        expert_up_v=expert_up_v,
        bias_up_v=bias_up_v,
        swish_alpha=swish_alpha,
        swish_beta=swish_beta,
        block_m=block_m,
        block_h=block_h,
        block_i=block_i,
        block_k=block_k,
    )
    if return_routing:
        return out, routing
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
    return_counts: bool = False,
):
    if tokens.ndim != 2 or gate_weights.ndim != 2:
        raise ValueError("tokens must be [S,H] and gate_weights must be [H,E]")
    if expert_up.ndim != 3 or expert_down.ndim != 3:
        raise ValueError("expert_up must be [E,H,I], expert_down must be [E,I,H]")
    if tokens.device.type != "cuda":
        raise ValueError("Triton megakernel expects HIP/CUDA tensors on device")
    if top_k not in (1, 2):
        raise ValueError("Triton megakernel currently supports top_k in {1, 2}")
    if tokens.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Triton megakernel currently expects fp16 or bf16 tokens")

    s, h = tokens.shape
    hw, e = gate_weights.shape
    ew, hup, i = expert_up.shape
    ed, idown, hdown = expert_down.shape
    if hw != h or hup != h or (ed, idown, hdown) != (e, i, h) or ew != e:
        raise ValueError("token, gate, and expert tensor shapes are inconsistent")
    gated = expert_up_v is not None
    if gated and bias_up_v is None:
        raise ValueError("bias_up_v is required when expert_up_v is provided")
    if not gated:
        expert_up_v = expert_up
        bias_up_v = bias_up
    if expert_capacity is None:
        expert_capacity = triton.cdiv(s * top_k, e)
    if block_h is None:
        block_h = _next_power_of_2(h)
    if block_h < h:
        raise ValueError("block_h must cover the full hidden dimension for the single-dispatch megakernel")

    block_e = _next_power_of_2(e)
    if block_e < 16:
        block_e = 16

    out = torch.empty((s, h), device=tokens.device, dtype=torch.float32)
    expert_counts = torch.empty((e,), device=tokens.device, dtype=torch.int32)
    expert_counts.zero_()
    _single_dispatch_megakernel[(s,)](
        tokens,
        gate_weights,
        expert_counts,
        expert_up,
        expert_up_v,
        bias_up,
        bias_up_v,
        expert_down,
        bias_down,
        out,
        S=s,
        H=h,
        I=i,
        E=e,
        TOP_K=top_k,
        EC=expert_capacity,
        ACT=activation,
        GATED=gated,
        BLOCK_E=block_e,
        BLOCK_H=block_h,
        swish_alpha=swish_alpha,
        swish_beta=swish_beta,
        num_warps=4,
        num_stages=1,
    )
    if return_counts:
        return out, expert_counts
    return out


def forward_from_routing(
    tokens: torch.Tensor,
    routing: RoutingResult,
    expert_up: torch.Tensor,
    expert_down: torch.Tensor,
    bias_up: torch.Tensor,
    bias_down: torch.Tensor,
    *,
    out: torch.Tensor,
    top_k: int,
    activation: int = ACT_IDENTITY,
    expert_up_v: Optional[torch.Tensor] = None,
    bias_up_v: Optional[torch.Tensor] = None,
    swish_alpha: float = 1.0,
    swish_beta: float = 1.0,
    block_m: int = 16,
    block_h: int = 32,
    block_i: int = 32,
    block_k: int = 32,
) -> torch.Tensor:
    if expert_up.ndim != 3 or expert_down.ndim != 3:
        raise ValueError("expert_up must be [E,H,I], expert_down must be [E,I,H]")
    e, h, i = expert_up.shape
    ed, idown, hdown = expert_down.shape
    if (ed, idown, hdown) != (e, i, h):
        raise ValueError("expert_down shape must be [E,I,H]")
    if tokens.shape[1] != h or out.shape != (tokens.shape[0], h):
        raise ValueError("token/output hidden dimensions do not match expert weights")
    gated = expert_up_v is not None
    if gated and bias_up_v is None:
        raise ValueError("bias_up_v is required when expert_up_v is provided")
    if not gated:
        expert_up_v = expert_up
        bias_up_v = bias_up

    grid = (e, triton.cdiv(routing.round_ec, block_m), triton.cdiv(h, block_h))
    inter = torch.empty((e, routing.round_ec, i), device=tokens.device, dtype=torch.float32)
    up_grid = (e, triton.cdiv(routing.round_ec, block_m), triton.cdiv(i, block_i))
    _up_project_kernel[up_grid](
        routing.routed_tokens,
        expert_up,
        expert_up_v,
        bias_up,
        bias_up_v,
        inter,
        H=h,
        I=i,
        EC=routing.expert_capacity,
        ROUND_EC=routing.round_ec,
        ACT=activation,
        GATED=gated,
        BLOCK_M=block_m,
        BLOCK_I=block_i,
        BLOCK_K=block_k,
        swish_alpha=swish_alpha,
        swish_beta=swish_beta,
        num_warps=4,
        num_stages=1,
    )
    _down_project_kernel[grid](
        routing.token_indices,
        routing.route_probs,
        inter,
        expert_down,
        bias_down,
        out,
        H=h,
        I=i,
        EC=routing.expert_capacity,
        ROUND_EC=routing.round_ec,
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        BLOCK_I=block_i,
        num_warps=4,
        num_stages=1,
    )
    return out


__all__ = [
    "ACT_IDENTITY",
    "ACT_SILU",
    "ACT_GELU",
    "ACT_RELU",
    "RoutingResult",
    "route",
    "forward",
    "forward_megakernel",
    "forward_from_routing",
]
