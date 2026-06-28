from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
from triton.runtime.jit import MockTensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flashmoe_gluon as fmg
from flashmoe_gluon.megakernel import _persistent_tdm_wmma_kernel, _resolve_tdm_num_programs


_ACTIVATIONS = {
    "identity": fmg.ACT_IDENTITY,
    "silu": fmg.ACT_SILU,
    "gelu": fmg.ACT_GELU,
    "relu": fmg.ACT_RELU,
}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise AssertionError(name)


def _dtype_bytes(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"unsupported dtype for byte estimate: {dtype}")


def _make_tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    init: str,
    scale: float,
) -> torch.Tensor:
    if init == "zero":
        return torch.zeros(shape, device="cuda", dtype=dtype)
    tensor = torch.empty(shape, device="cuda", dtype=dtype)
    tensor.normal_(mean=0.0, std=scale)
    return tensor


def _make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, ...]:
    dtype = _dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    tokens = _make_tensor((args.s, args.h), dtype=dtype, init=args.init, scale=args.input_scale)
    gate = _make_tensor((args.h, args.e), dtype=dtype, init=args.init, scale=args.gate_scale)
    up = _make_tensor((args.e, args.h, args.i), dtype=dtype, init=args.init, scale=args.weight_scale)
    down = _make_tensor((args.e, args.i, args.h), dtype=dtype, init=args.init, scale=args.weight_scale)
    bias_up = _make_tensor((args.e, args.i), dtype=dtype, init=args.init, scale=args.bias_scale)
    bias_down = _make_tensor((args.e, args.h), dtype=dtype, init=args.init, scale=args.bias_scale)
    if not args.gated:
        return tokens, gate, up, down, bias_up, bias_down, None, None
    up_v = _make_tensor((args.e, args.h, args.i), dtype=dtype, init=args.init, scale=args.weight_scale)
    bias_up_v = _make_tensor((args.e, args.i), dtype=dtype, init=args.init, scale=args.bias_scale)
    return tokens, gate, up, down, bias_up, bias_down, up_v, bias_up_v


class _ContiguousMockTensor(MockTensor):
    def __init__(self, dtype: torch.dtype, shape: list[int]):
        super().__init__(dtype, shape)
        stride: list[int] = []
        current = 1
        for size in reversed(shape):
            stride.append(current)
            current *= size
        self._stride = tuple(reversed(stride))

    def stride(self):
        return self._stride


def _mock(dtype: torch.dtype, *shape: int) -> MockTensor:
    return _ContiguousMockTensor(dtype, list(shape))


def _precompile_tdm_wmma(
    args: argparse.Namespace,
    tensors: tuple[torch.Tensor, ...] | None = None,
) -> tuple[float, object]:
    if args.h % 64 != 0 or args.i % 64 != 0:
        raise ValueError("--precompile requires H and I to be multiples of 64 for the TDM/WMMA path")
    if args.top_k not in (1, 2):
        raise ValueError("--precompile requires --top-k 1 or 2")
    if args.num_warps != 4:
        raise ValueError("--precompile matches the TDM/WMMA path, which currently requires --num-warps 4")

    block_m = 16
    block_n = 64
    block_h = args.block_h if args.block_h is not None else _next_power_of_2(args.h)
    route_blocks = _ceil_div(args.expert_capacity, block_m)
    total_tasks = args.e * route_blocks * ((args.i // block_n) + (args.h // block_n))
    if tensors is None:
        dtype = _dtype_from_name(args.dtype)
        kernel_args = (
            _mock(dtype, args.s, args.h),
            _mock(dtype, args.h, args.e),
            _mock(torch.int32, args.s, args.top_k),
            _mock(torch.float32, args.s, args.top_k),
            _mock(dtype, args.e, args.h, args.i),
            _mock(dtype, args.e, args.i),
            _mock(dtype, args.e, args.h, args.i),
            _mock(dtype, args.e, args.i),
            _mock(dtype, args.e, args.i, args.h),
            _mock(dtype, args.e, args.h),
            _mock(torch.float32, args.s, args.h),
            _mock(dtype, args.e, args.expert_capacity, args.i),
            _mock(torch.int32, args.e),
            _mock(torch.int32, args.e, args.expert_capacity),
            _mock(torch.float32, args.e, args.expert_capacity),
            _mock(torch.int32, total_tasks),
            _mock(torch.int32, 1),
            _mock(torch.int32, 1),
            _mock(torch.int32, 1),
            _mock(torch.int32, args.e * route_blocks),
            _mock(torch.int32, args.num_programs),
            _mock(torch.int32, args.num_programs * 3),
            _mock(torch.int32, 6),
            _mock(torch.int32, 1),
            _mock(torch.int32, 1),
        )
    else:
        tokens, gate, up, down, bias_up, bias_down, up_v, bias_up_v = tensors
        up_v_arg = up_v if up_v is not None else up
        bias_up_v_arg = bias_up_v if bias_up_v is not None else bias_up
        kernel_args = (
            tokens,
            gate,
            torch.empty((args.s, args.top_k), device=tokens.device, dtype=torch.int32),
            torch.empty((args.s, args.top_k), device=tokens.device, dtype=torch.float32),
            up,
            bias_up,
            up_v_arg,
            bias_up_v_arg,
            down,
            bias_down,
            torch.empty((args.s, args.h), device=tokens.device, dtype=torch.float32),
            torch.empty((args.e, args.expert_capacity, args.i), device=tokens.device, dtype=tokens.dtype),
            torch.empty((args.e,), device=tokens.device, dtype=torch.int32),
            torch.empty((args.e, args.expert_capacity), device=tokens.device, dtype=torch.int32),
            torch.empty((args.e, args.expert_capacity), device=tokens.device, dtype=torch.float32),
            torch.empty((total_tasks,), device=tokens.device, dtype=torch.int32),
            torch.empty((1,), device=tokens.device, dtype=torch.int32),
            torch.empty((1,), device=tokens.device, dtype=torch.int32),
            torch.empty((1,), device=tokens.device, dtype=torch.int32),
            torch.empty((args.e * route_blocks,), device=tokens.device, dtype=torch.int32),
            torch.empty((args.num_programs,), device=tokens.device, dtype=torch.int32),
            torch.empty((args.num_programs * 3,), device=tokens.device, dtype=torch.int32),
            torch.empty((6,), device=tokens.device, dtype=torch.int32),
            torch.empty((1,), device=tokens.device, dtype=torch.int32),
            torch.empty((1,), device=tokens.device, dtype=torch.int32),
        )
    launch_epoch = 1
    start = time.perf_counter()
    kernel = _persistent_tdm_wmma_kernel.warmup(
        *kernel_args,
        launch_epoch,
        S=args.s,
        H=args.h,
        I=args.i,
        E=args.e,
        EC=args.expert_capacity,
        TOP_K=args.top_k,
        PRECOMPUTED_ROUTING=False,
        ZERO_OUTPUT=True,
        ACTIVATION=_ACTIVATIONS[args.activation],
        GATED=args.gated,
        BLOCK_E=args.e,
        BLOCK_H=block_h,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NUM_PROGRAMS=args.num_programs,
        NUM_WARPS=args.num_warps,
        DEBUG=False,
        DEBUG_SCHED=False,
        num_warps=args.num_warps,
        grid=(args.num_programs,),
    )
    return (time.perf_counter() - start) * 1000.0, kernel


def _static_profile(kernel: object) -> dict[str, int]:
    import re

    amdgcn = kernel.asm["amdgcn"]

    def extract(pattern: str) -> int:
        match = re.search(pattern, amdgcn)
        return int(match.group(1)) if match else -1

    return {
        "sgpr": extract(r"\.sgpr_count:\s+(\d+)"),
        "sgpr_spill": extract(r"\.sgpr_spill_count:\s+(\d+)"),
        "vgpr": extract(r"\.vgpr_count:\s+(\d+)"),
        "vgpr_spill": extract(r"\.vgpr_spill_count:\s+(\d+)"),
        "scratch": extract(r";\s+ScratchSize:\s+(\d+)"),
        "code_bytes": extract(r";\s+codeLenInByte\s+=\s+(\d+)"),
        "occupancy": extract(r";\s+Occupancy:\s+(\d+)"),
    }


def _run_megakernel(args: argparse.Namespace, tensors: tuple[torch.Tensor, ...], *, return_counts: bool):
    tokens, gate, up, down, bias_up, bias_down, up_v, bias_up_v = tensors
    return fmg.forward_megakernel(
        tokens,
        gate,
        up,
        down,
        bias_up,
        bias_down,
        top_k=args.top_k,
        expert_capacity=args.expert_capacity,
        activation=_ACTIVATIONS[args.activation],
        expert_up_v=up_v,
        bias_up_v=bias_up_v,
        block_h=args.block_h,
        num_warps=args.num_warps,
        num_programs=args.num_programs,
        use_persistent=True,
        use_tdm_wmma=not args.disable_tdm_wmma,
        return_counts=return_counts,
    )


def _estimate_work(
    args: argparse.Namespace,
    counts: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> dict[str, float | int]:
    raw_counts = [int(x) for x in counts.cpu().tolist()]
    accepted_counts = [min(x, args.expert_capacity) for x in raw_counts]
    accepted_routes = sum(accepted_counts)
    active_experts = sum(1 for x in accepted_counts if x > 0)
    route_blocks = sum(_ceil_div(x, 16) for x in accepted_counts if x > 0)

    router_flops = 2 * args.s * args.h * args.e
    up_flops = 2 * accepted_routes * args.h * args.i
    gated_up_flops = up_flops if args.gated else 0
    down_flops = 2 * accepted_routes * args.i * args.h
    gemm_flops = up_flops + gated_up_flops + down_flops
    total_flops = router_flops + gemm_flops

    elem_bytes = _dtype_bytes(dtype)
    h_tiles = _ceil_div(args.h, 64)
    i_tiles = _ceil_div(args.i, 64)
    bytes_router = 2 * args.s * args.e * args.h * elem_bytes
    bytes_route_tables = accepted_routes * (4 + 4)
    bytes_gemm0_tokens = accepted_routes * args.h * i_tiles * elem_bytes
    bytes_gemm0_weights = route_blocks * args.h * args.i * elem_bytes
    bytes_gemm0_bias = accepted_routes * args.i * elem_bytes
    bytes_gated_tokens = bytes_gemm0_tokens if args.gated else 0
    bytes_gated_weights = bytes_gemm0_weights if args.gated else 0
    bytes_gated_bias = bytes_gemm0_bias if args.gated else 0
    bytes_hidden_write = accepted_routes * args.i * elem_bytes
    bytes_gemm1_hidden = accepted_routes * args.i * h_tiles * elem_bytes
    bytes_gemm1_weights = route_blocks * args.i * args.h * elem_bytes
    bytes_down_bias = accepted_routes * args.h * elem_bytes
    bytes_output = args.s * args.h * 4 + accepted_routes * args.h * 4
    estimated_bytes = (
        bytes_router
        + bytes_route_tables
        + bytes_gemm0_tokens
        + bytes_gemm0_weights
        + bytes_gemm0_bias
        + bytes_gated_tokens
        + bytes_gated_weights
        + bytes_gated_bias
        + bytes_hidden_write
        + bytes_gemm1_hidden
        + bytes_gemm1_weights
        + bytes_down_bias
        + bytes_output
    )

    return {
        "raw_routes": sum(raw_counts),
        "accepted_routes": accepted_routes,
        "dropped_routes": sum(raw_counts) - accepted_routes,
        "active_experts": active_experts,
        "active_route_blocks": route_blocks,
        "router_flops": router_flops,
        "gemm_flops": gemm_flops,
        "total_flops": total_flops,
        "estimated_bytes": estimated_bytes,
    }


def _format_rate(numerator: float, elapsed_ms: float, scale: float) -> float:
    if elapsed_ms <= 0.0:
        return float("nan")
    return numerator / (elapsed_ms / 1000.0) / scale


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the persistent Gluon FlashMoE megakernel.")
    parser.add_argument("--s", type=int, default=64)
    parser.add_argument("--h", type=int, default=1024)
    parser.add_argument("--i", type=int, default=1024)
    parser.add_argument("--e", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--expert-capacity", type=int, default=64)
    parser.add_argument("--num-programs", type=int, default=8)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--block-h", type=int, default=None)
    parser.add_argument("--activation", choices=tuple(_ACTIVATIONS), default="identity")
    parser.add_argument("--gated", action="store_true")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--init", choices=("random", "zero"), default="random")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--input-scale", type=float, default=0.02)
    parser.add_argument("--gate-scale", type=float, default=0.02)
    parser.add_argument("--weight-scale", type=float, default=0.02)
    parser.add_argument("--bias-scale", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--precompile", action="store_true")
    parser.add_argument("--precompile-only", action="store_true")
    parser.add_argument("--static-profile", action="store_true")
    parser.add_argument("--disable-tdm-wmma", action="store_true")
    parser.add_argument("--check-finite", action="store_true")
    parser.add_argument(
        "--timing",
        choices=("host", "event"),
        default="host",
        help="host uses synchronized wall time; event uses torch.cuda.Event timing",
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")

    requested_num_programs = args.num_programs
    args.num_programs = _resolve_tdm_num_programs(
        _dtype_from_name(args.dtype),
        args.block_h if args.block_h is not None else _next_power_of_2(args.h),
        args.num_programs,
    )
    if args.num_programs != requested_num_programs:
        print(
            "effective_num_programs",
            args.num_programs,
            "requested",
            requested_num_programs,
        )

    if args.precompile_only:
        precompile_ms, kernel = _precompile_tdm_wmma(args)
        print(f"precompile_ms {precompile_ms:.3f}")
        if args.static_profile:
            profile = _static_profile(kernel)
            print(
                "static_profile",
                "sgpr",
                profile["sgpr"],
                "sgpr_spill",
                profile["sgpr_spill"],
                "vgpr",
                profile["vgpr"],
                "vgpr_spill",
                profile["vgpr_spill"],
                "scratch",
                profile["scratch"],
                "code_bytes",
                profile["code_bytes"],
                "occupancy",
                profile["occupancy"],
            )
        return

    torch.cuda.set_device(0)
    tensors = _make_inputs(args)
    torch.cuda.synchronize()
    if args.precompile or args.static_profile:
        precompile_ms, kernel = _precompile_tdm_wmma(args, tensors)
        torch.cuda.synchronize()
        print(f"precompile_ms {precompile_ms:.3f}")
        if args.static_profile:
            profile = _static_profile(kernel)
            print(
                "static_profile",
                "sgpr",
                profile["sgpr"],
                "sgpr_spill",
                profile["sgpr_spill"],
                "vgpr",
                profile["vgpr"],
                "vgpr_spill",
                profile["vgpr_spill"],
                "scratch",
                profile["scratch"],
                "code_bytes",
                profile["code_bytes"],
                "occupancy",
                profile["occupancy"],
            )

    for _ in range(args.warmup):
        out = _run_megakernel(args, tensors, return_counts=False)
        torch.cuda.synchronize()
        del out

    event_times_ms: list[float] = []
    host_times_ms: list[float] = []
    out = None
    counts = None
    for repeat_idx in range(args.repeats):
        torch.cuda.synchronize()
        if args.timing == "event":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            host_start = time.perf_counter()
            result = _run_megakernel(args, tensors, return_counts=(repeat_idx == args.repeats - 1))
            host_end = time.perf_counter()
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            host_elapsed_ms = (host_end - host_start) * 1000.0
        else:
            host_start = time.perf_counter()
            result = _run_megakernel(args, tensors, return_counts=(repeat_idx == args.repeats - 1))
            torch.cuda.synchronize()
            host_end = time.perf_counter()
            elapsed_ms = (host_end - host_start) * 1000.0
            host_elapsed_ms = elapsed_ms
        if repeat_idx == args.repeats - 1:
            out, counts = result
        else:
            out = result
        event_times_ms.append(elapsed_ms)
        host_times_ms.append(host_elapsed_ms)

    assert out is not None
    assert counts is not None
    if args.check_finite:
        out_cpu = out.detach().cpu()
        if not bool(torch.isfinite(out_cpu).all().item()):
            raise SystemExit("output contains non-finite values")

    dtype = _dtype_from_name(args.dtype)
    work = _estimate_work(args, counts, dtype=dtype)
    median_ms = statistics.median(event_times_ms)
    min_ms = min(event_times_ms)
    host_median_ms = statistics.median(host_times_ms)
    host_min_ms = min(host_times_ms)

    print(
        "shape",
        "S",
        args.s,
        "H",
        args.h,
        "I",
        args.i,
        "E",
        args.e,
        "top_k",
        args.top_k,
        "EC",
        args.expert_capacity,
        "gated",
        args.gated,
        "num_programs",
        args.num_programs,
        "num_warps",
        args.num_warps,
    )
    print(
        "timing",
        "mode",
        args.timing,
        "event_median_ms",
        f"{median_ms:.3f}",
        "event_min_ms",
        f"{min_ms:.3f}",
        "host_median_ms",
        f"{host_median_ms:.3f}",
        "host_min_ms",
        f"{host_min_ms:.3f}",
        "repeats",
        args.repeats,
    )
    print(
        "routes",
        "raw",
        work["raw_routes"],
        "accepted",
        work["accepted_routes"],
        "dropped",
        work["dropped_routes"],
        "active_experts",
        work["active_experts"],
        "active_route_blocks",
        work["active_route_blocks"],
    )
    print(
        "throughput",
        "gemm_tflops",
        f"{_format_rate(float(work['gemm_flops']), median_ms, 1.0e12):.3f}",
        "total_est_tflops",
        f"{_format_rate(float(work['total_flops']), median_ms, 1.0e12):.3f}",
        "est_bandwidth_gbs",
        f"{_format_rate(float(work['estimated_bytes']), median_ms, 1.0e9):.3f}",
        "est_bytes_gb",
        f"{float(work['estimated_bytes']) / 1.0e9:.6f}",
    )


if __name__ == "__main__":
    main()
