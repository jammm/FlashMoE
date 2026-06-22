from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl

import flashmoe_triton as fmt


@triton.jit
def _touch_routing_kernel(
    expert_counts,
    token_indices,
    route_probs,
    out,
    H: tl.constexpr,
    EC: tl.constexpr,
    ROUND_EC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    expert_id = tl.program_id(0)
    m_tile = tl.program_id(1)
    h_tile = tl.program_id(2)

    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = h_tile * BLOCK_H + tl.arange(0, BLOCK_H)
    count = tl.load(expert_counts + expert_id)
    valid_m = offs_m < tl.minimum(count, EC)
    token_ids = tl.load(
        token_indices + expert_id * ROUND_EC + offs_m,
        mask=valid_m,
        other=0,
    )
    probs = tl.load(
        route_probs + expert_id * ROUND_EC + offs_m,
        mask=valid_m,
        other=0.0,
    ).to(tl.float32)
    ptrs = out + token_ids[:, None] * H + offs_h[None, :]
    mask = valid_m[:, None] & (offs_h[None, :] < H)
    tl.atomic_add(ptrs, probs[:, None], sem="relaxed", mask=mask)


@triton.jit
def _routed_up_probe_kernel(
    routed_tokens,
    expert_counts,
    expert_up,
    bias_up,
    scratch,
    H: tl.constexpr,
    I: tl.constexpr,
    EC: tl.constexpr,
    ROUND_EC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert_id = tl.program_id(0)
    m_tile = tl.program_id(1)
    i_tile = tl.program_id(2)

    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = i_tile * BLOCK_I + tl.arange(0, BLOCK_I)
    count = tl.load(expert_counts + expert_id)
    valid_m = offs_m < tl.minimum(count, EC)

    acc = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
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

    b = tl.load(bias_up + expert_id * I + offs_i, mask=offs_i < I, other=0.0).to(tl.float32)
    acc += b[None, :]
    ptrs = scratch + expert_id * ROUND_EC * I + offs_m[:, None] * I + offs_i[None, :]
    mask = valid_m[:, None] & (offs_i[None, :] < I)
    tl.store(ptrs, acc, mask=mask)


@triton.jit
def _up_touch_kernel(
    routed_tokens,
    expert_counts,
    token_indices,
    route_probs,
    expert_up,
    bias_up,
    out,
    H: tl.constexpr,
    I: tl.constexpr,
    EC: tl.constexpr,
    ROUND_EC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert_id = tl.program_id(0)
    m_tile = tl.program_id(1)
    h_tile = tl.program_id(2)

    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = h_tile * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_i = tl.arange(0, BLOCK_I)
    count = tl.load(expert_counts + expert_id)
    valid_m = offs_m < tl.minimum(count, EC)
    token_ids = tl.load(token_indices + expert_id * ROUND_EC + offs_m, mask=valid_m, other=0)
    probs = tl.load(route_probs + expert_id * ROUND_EC + offs_m, mask=valid_m, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
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
    b = tl.load(bias_up + expert_id * I + offs_i, mask=offs_i < I, other=0.0).to(tl.float32)
    acc += b[None, :]
    row_sum = tl.sum(acc, axis=1) * probs
    ptrs = out + token_ids[:, None] * H + offs_h[None, :]
    mask = valid_m[:, None] & (offs_h[None, :] < H)
    tl.atomic_add(ptrs, row_sum[:, None], sem="relaxed", mask=mask)


@triton.jit
def _manual_up_touch_kernel(
    routed_tokens,
    expert_counts,
    token_indices,
    route_probs,
    expert_up,
    bias_up,
    out,
    H: tl.constexpr,
    I: tl.constexpr,
    EC: tl.constexpr,
    ROUND_EC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert_id = tl.program_id(0)
    m_tile = tl.program_id(1)
    h_tile = tl.program_id(2)

    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = h_tile * BLOCK_H + tl.arange(0, BLOCK_H)
    count = tl.load(expert_counts + expert_id)
    valid_m = offs_m < tl.minimum(count, EC)
    token_ids = tl.load(token_indices + expert_id * ROUND_EC + offs_m, mask=valid_m, other=0)
    probs = tl.load(route_probs + expert_id * ROUND_EC + offs_m, mask=valid_m, other=0.0).to(tl.float32)

    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for i_abs in range(0, I):
        hidden = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for h0 in range(0, H, BLOCK_K):
            offs_k = h0 + tl.arange(0, BLOCK_K)
            x = tl.load(
                routed_tokens + expert_id * ROUND_EC * H + offs_m[:, None] * H + offs_k[None, :],
                mask=valid_m[:, None] & (offs_k[None, :] < H),
                other=0.0,
            ).to(tl.float32)
            w = tl.load(
                expert_up + expert_id * H * I + offs_k * I + i_abs,
                mask=offs_k < H,
                other=0.0,
            ).to(tl.float32)
            hidden += tl.sum(x * w[None, :], axis=1)
        bu = tl.load(bias_up + expert_id * I + i_abs, mask=i_abs < I, other=0.0).to(tl.float32)
        row_sum += hidden + bu

    row_sum *= probs
    ptrs = out + token_ids[:, None] * H + offs_h[None, :]
    mask = valid_m[:, None] & (offs_h[None, :] < H)
    tl.atomic_add(ptrs, row_sum[:, None], sem="relaxed", mask=mask)


@triton.jit
def _down_probe_kernel(
    token_indices,
    route_probs,
    inter,
    expert_down,
    bias_down,
    out,
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
    token_ids = tl.load(token_indices + expert_id * ROUND_EC + offs_m, mask=offs_m < EC, other=0)
    probs = tl.load(route_probs + expert_id * ROUND_EC + offs_m, mask=offs_m < EC, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    for i0 in range(0, I, BLOCK_I):
        offs_i = i0 + tl.arange(0, BLOCK_I)
        x = tl.load(
            inter + expert_id * ROUND_EC * I + offs_m[:, None] * I + offs_i[None, :],
            mask=(offs_m[:, None] < EC) & (offs_i[None, :] < I),
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
    ptrs = out + token_ids[:, None] * H + offs_h[None, :]
    mask = (offs_m[:, None] < EC) & (offs_h[None, :] < H)
    tl.atomic_add(ptrs, acc, sem="relaxed", mask=mask)


@triton.jit
def _manual_down_probe_kernel(
    token_indices,
    route_probs,
    expert_down,
    bias_down,
    out,
    H: tl.constexpr,
    I: tl.constexpr,
    EC: tl.constexpr,
    ROUND_EC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
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
    for i_abs in range(0, I):
        wd = tl.load(
            expert_down + expert_id * I * H + i_abs * H + offs_h,
            mask=offs_h < H,
            other=0.0,
        ).to(tl.float32)
        acc += wd[None, :]
    b = tl.load(bias_down + expert_id * H + offs_h, mask=offs_h < H, other=0.0).to(tl.float32)
    acc = (acc + b[None, :]) * probs[:, None]
    ptrs = out + token_ids[:, None] * H + offs_h[None, :]
    mask = valid_m[:, None] & (offs_h[None, :] < H)
    tl.atomic_add(ptrs, acc, sem="relaxed", mask=mask)


def _quantize_like_kernel(x: torch.Tensor) -> torch.Tensor:
    return x.half().float()


def _reference(
    tokens_h: torch.Tensor,
    gate_h: torch.Tensor,
    up_h: torch.Tensor,
    down_h: torch.Tensor,
    bias_up_h: torch.Tensor,
    bias_down_h: torch.Tensor,
    top_k: int,
    expert_capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = tokens_h.float()
    gate = gate_h.float()
    up = up_h.float()
    down = down_h.float()
    bias_up = bias_up_h.float()
    bias_down = bias_down_h.float()
    probs = torch.softmax(tokens @ gate, dim=1)
    vals, idxs = torch.topk(probs, top_k, dim=1)
    vals = vals / vals.sum(dim=1, keepdim=True)

    s, h = tokens.shape
    e = gate.shape[1]
    out = torch.zeros((s, h), dtype=torch.float32)
    counts = torch.zeros(e, dtype=torch.int32)
    routes: list[list[tuple[int, float]]] = [[] for _ in range(e)]
    for token_id in range(s):
        for route_id in range(top_k):
            expert_id = int(idxs[token_id, route_id])
            prob = float(vals[token_id, route_id])
            counts[expert_id] += 1
            routes[expert_id].append((token_id, prob))

    for expert_id in range(e):
        for token_id, prob in routes[expert_id][:expert_capacity]:
            hidden = _quantize_like_kernel(tokens[token_id] @ up[expert_id] + bias_up[expert_id])
            result = _quantize_like_kernel(hidden @ down[expert_id] + bias_down[expert_id])
            out[token_id] += result * prob
    return out, counts


def _make_inputs(
    s: int,
    h: int,
    i: int,
    e: int,
    *,
    seed: int = 7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tokens = (torch.rand((s, h), generator=generator) * 0.2 - 0.1).half()
    gate = (torch.rand((h, e), generator=generator) * 0.2 - 0.1).half()
    up = (torch.rand((e, h, i), generator=generator) * 0.2 - 0.1).half()
    down = (torch.rand((e, i, h), generator=generator) * 0.2 - 0.1).half()
    bias_up = (torch.rand((e, i), generator=generator) * 0.02 - 0.01).half()
    bias_down = (torch.rand((e, h), generator=generator) * 0.02 - 0.01).half()
    return tokens, gate, up, down, bias_up, bias_down


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(
            "route",
            "touch",
            "routed-up",
            "up-touch",
            "manual-up-touch",
            "down",
            "manual-down",
            "forward",
            "megakernel",
        ),
        default="forward",
    )
    parser.add_argument("--s", type=int, default=16)
    parser.add_argument("--h", type=int, default=64)
    parser.add_argument("--i", type=int, default=64)
    parser.add_argument("--e", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--expert-capacity", type=int, default=16)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    tokens, gate, up, down, bias_up, bias_down = _make_inputs(args.s, args.h, args.i, args.e)
    ref_out, ref_counts = _reference(
        tokens,
        gate,
        up,
        down,
        bias_up,
        bias_down,
        args.top_k,
        args.expert_capacity,
    )

    tokens_d = tokens.cuda()
    gate_d = gate.cuda()

    if args.case == "megakernel":
        out, counts_mk = fmt.forward_megakernel(
            tokens_d,
            gate_d,
            up.cuda(),
            down.cuda(),
            bias_up.cuda(),
            bias_down.cuda(),
            top_k=args.top_k,
            expert_capacity=args.expert_capacity,
            return_counts=True,
        )
        torch.cuda.synchronize()
        got = out.cpu()
        diff = (got - ref_out).abs()
        print("megakernel_counts", counts_mk.cpu().tolist(), ref_counts.tolist(), torch.equal(counts_mk.cpu(), ref_counts))
        print(
            "megakernel",
            "max_abs",
            float(diff.max()),
            "mean_abs",
            float(diff.mean()),
            "finite",
            bool(torch.isfinite(got).all()),
        )
        return

    routing = fmt.route(tokens_d, gate_d, top_k=args.top_k, expert_capacity=args.expert_capacity)
    torch.cuda.synchronize()
    counts = routing.expert_counts.cpu()
    print("counts", counts.tolist(), ref_counts.tolist(), torch.equal(counts, ref_counts))
    if args.case == "route":
        return

    if args.case == "touch":
        out = torch.zeros((args.s, args.h), device="cuda", dtype=torch.float32)
        grid = (args.e, triton.cdiv(routing.round_ec, 16), triton.cdiv(args.h, 32))
        _touch_routing_kernel[grid](
            routing.expert_counts,
            routing.token_indices,
            routing.route_probs,
            out,
            H=args.h,
            EC=routing.expert_capacity,
            ROUND_EC=routing.round_ec,
            BLOCK_M=16,
            BLOCK_H=32,
            num_warps=4,
        )
        torch.cuda.synchronize()
        print("touch_sum", float(out.sum().cpu()))
        return

    if args.case == "routed-up":
        scratch = torch.empty((args.e, routing.round_ec, args.i), device="cuda", dtype=torch.float32)
        scratch.zero_()
        grid = (args.e, triton.cdiv(routing.round_ec, 16), triton.cdiv(args.i, 32))
        _routed_up_probe_kernel[grid](
            routing.routed_tokens,
            routing.expert_counts,
            up.cuda(),
            bias_up.cuda(),
            scratch,
            H=args.h,
            I=args.i,
            EC=routing.expert_capacity,
            ROUND_EC=routing.round_ec,
            BLOCK_M=16,
            BLOCK_I=32,
            BLOCK_K=32,
            num_warps=4,
        )
        torch.cuda.synchronize()
        print("routed_up_sum", float(scratch.sum().cpu()), "first", float(scratch[0, 0, 0].cpu()))
        return

    if args.case == "up-touch":
        out = torch.zeros((args.s, args.h), device="cuda", dtype=torch.float32)
        grid = (args.e, triton.cdiv(routing.round_ec, 16), triton.cdiv(args.h, 32))
        _up_touch_kernel[grid](
            routing.routed_tokens,
            routing.expert_counts,
            routing.token_indices,
            routing.route_probs,
            up.cuda(),
            bias_up.cuda(),
            out,
            H=args.h,
            I=args.i,
            EC=routing.expert_capacity,
            ROUND_EC=routing.round_ec,
            BLOCK_M=16,
            BLOCK_H=32,
            BLOCK_I=32,
            BLOCK_K=32,
            num_warps=4,
        )
        torch.cuda.synchronize()
        print("up_touch_sum", float(out.sum().cpu()))
        return

    if args.case == "manual-up-touch":
        out = torch.zeros((args.s, args.h), device="cuda", dtype=torch.float32)
        grid = (args.e, triton.cdiv(routing.round_ec, 16), triton.cdiv(args.h, 32))
        _manual_up_touch_kernel[grid](
            routing.routed_tokens,
            routing.expert_counts,
            routing.token_indices,
            routing.route_probs,
            up.cuda(),
            bias_up.cuda(),
            out,
            H=args.h,
            I=args.i,
            EC=routing.expert_capacity,
            ROUND_EC=routing.round_ec,
            BLOCK_M=16,
            BLOCK_H=32,
            BLOCK_K=32,
            num_warps=4,
        )
        torch.cuda.synchronize()
        print("manual_up_touch_sum", float(out.sum().cpu()))
        return

    if args.case == "down":
        inter = torch.ones((args.e, routing.round_ec, args.i), device="cuda", dtype=torch.float32)
        out = torch.zeros((args.s, args.h), device="cuda", dtype=torch.float32)
        grid = (args.e, triton.cdiv(routing.round_ec, 16), triton.cdiv(args.h, 32))
        _down_probe_kernel[grid](
            routing.token_indices,
            routing.route_probs,
            inter,
            down.cuda(),
            bias_down.cuda(),
            out,
            H=args.h,
            I=args.i,
            EC=routing.expert_capacity,
            ROUND_EC=routing.round_ec,
            BLOCK_M=16,
            BLOCK_H=32,
            BLOCK_I=32,
            num_warps=4,
        )
        torch.cuda.synchronize()
        print("down_sum", float(out.nan_to_num().sum().cpu()))
        return

    if args.case == "manual-down":
        out = torch.zeros((args.s, args.h), device="cuda", dtype=torch.float32)
        grid = (args.e, triton.cdiv(routing.round_ec, 16), triton.cdiv(args.h, 32))
        _manual_down_probe_kernel[grid](
            routing.token_indices,
            routing.route_probs,
            down.cuda(),
            bias_down.cuda(),
            out,
            H=args.h,
            I=args.i,
            EC=routing.expert_capacity,
            ROUND_EC=routing.round_ec,
            BLOCK_M=16,
            BLOCK_H=32,
            num_warps=4,
        )
        torch.cuda.synchronize()
        print("manual_down_sum", float(out.sum().cpu()))
        return

    out = torch.empty((args.s, args.h), device="cuda", dtype=torch.float32)
    out.zero_()
    fmt.forward_from_routing(
        tokens_d,
        routing,
        up.cuda(),
        down.cuda(),
        bias_up.cuda(),
        bias_down.cuda(),
        out=out,
        top_k=args.top_k,
    )
    torch.cuda.synchronize()
    got = out.cpu()
    diff = (got - ref_out).abs()
    print(
        "forward",
        "max_abs",
        float(diff.max()),
        "mean_abs",
        float(diff.mean()),
        "finite",
        bool(torch.isfinite(got).all()),
    )


if __name__ == "__main__":
    main()
