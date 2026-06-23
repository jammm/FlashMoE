from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl


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
    return gl.BlockedLayout([1], [32], [num_warps], [0])


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
    if return_counts:
        raise NotImplementedError("Gluon paper-parity megakernel has not wired count return yet")
    block_e = max(16, _next_power_of_2(spec.e))
    up_v_arg = expert_up_v if expert_up_v is not None else expert_up
    bias_up_v_arg = bias_up_v if bias_up_v is not None else bias_up
    out = torch.empty((spec.s, spec.h), device=tokens.device, dtype=torch.float32)
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
