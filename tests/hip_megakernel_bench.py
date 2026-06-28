from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flashmoe as fm
from flashmoe import ActivationType, DataType, ForwardArgs, InitArgs, MLPType
from flashmoe.router import RouterForwardArgs


def _dtype_from_name(name: str) -> tuple[DataType, torch.dtype]:
    if name == "fp16":
        return DataType.FP16, torch.float16
    if name == "bf16":
        return DataType.BF16, torch.bfloat16
    if name == "fp32":
        return DataType.FP32, torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _mlp_from_name(name: str) -> MLPType:
    if name == "vanilla":
        return MLPType.VANILLA
    if name == "gated":
        return MLPType.GATED
    raise ValueError(f"unsupported mlp: {name}")


def _activation_from_name(name: str) -> ActivationType:
    if name == "identity":
        return ActivationType.IDENTITY
    if name == "silu":
        return ActivationType.SILU
    if name == "gelu":
        return ActivationType.GELU
    if name == "relu":
        return ActivationType.RELU
    raise ValueError(f"unsupported activation: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=int, default=16_384)
    parser.add_argument("--h", type=int, default=64)
    parser.add_argument("--i", type=int, default=64)
    parser.add_argument("--e", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--expert-capacity", type=int, default=None)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--mlp", choices=("vanilla", "gated"), default="vanilla")
    parser.add_argument("--activation", choices=("identity", "silu", "gelu", "relu"), default="identity")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--arch", type=int, default=1250)
    args = parser.parse_args()

    if fm.BACKEND != "hip":
        raise RuntimeError(f"{Path(__file__).name} requires the HIP backend, got {fm.BACKEND}")

    data_type, torch_dtype = _dtype_from_name(args.dtype)
    mlp_type = _mlp_from_name(args.mlp)
    act_type = _activation_from_name(args.activation)
    expert_capacity = args.expert_capacity
    if expert_capacity is None:
        expert_capacity = math.ceil(args.s * args.top_k * 1.25 / args.e)

    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    stream_ptr = stream.cuda_stream
    flash = None
    router = None

    try:
        torch.manual_seed(args.seed)
        with torch.cuda.stream(stream):
            init_args = InitArgs(
                data_type,
                args.s,
                args.h,
                args.i,
                args.e,
                args.top_k,
                args.arch,
                mlp_type,
                act_type,
                stream_ptr,
                0,
                ep_world=1,
                num_local_experts=args.e,
                ep_rank=0,
                my_pe=0,
                expert_map=[0] * args.e,
                rank_map=[0],
                expert_peer_capacity=expert_capacity,
            )
            flash = fm.initialize(init_args)
            router = fm.router.initialize(init_args)

            tokens = (torch.randn((args.s, args.h), device="cuda", dtype=torch_dtype) * 0.125).contiguous()
            gate = (torch.randn((args.h, args.e), device="cuda", dtype=torch_dtype) * 0.125).contiguous()
            counts = torch.zeros((args.e,), device="cuda", dtype=torch.int32)
            up = (torch.randn((args.e, args.h, args.i), device="cuda", dtype=torch_dtype) * 0.125).contiguous()
            bias_up = (torch.randn((args.e, args.i), device="cuda", dtype=torch_dtype) * 0.01).contiguous()
            down = (torch.randn((args.e, args.i, args.h), device="cuda", dtype=torch_dtype) * 0.125).contiguous()
            bias_down = (torch.randn((args.e, args.h), device="cuda", dtype=torch_dtype) * 0.01).contiguous()
            up_v = None
            bias_up_v = None
            if mlp_type == MLPType.GATED:
                up_v = (torch.randn((args.e, args.h, args.i), device="cuda", dtype=torch_dtype) * 0.125).contiguous()
                bias_up_v = (torch.randn((args.e, args.i), device="cuda", dtype=torch_dtype) * 0.01).contiguous()
            out = torch.empty((args.s, args.h), device="cuda", dtype=torch_dtype)

            fm.router.forward(
                router,
                flash,
                RouterForwardArgs(tokens.data_ptr(), gate.data_ptr(), counts.data_ptr(), stream_ptr),
            )
        stream.synchronize()

        counts_cpu = [int(x) for x in counts.cpu().tolist()]
        accepted = sum(min(c, expert_capacity) for c in counts_cpu)
        dropped = sum(max(c - expert_capacity, 0) for c in counts_cpu)

        forward_args = ForwardArgs(
            mlp_type,
            tokens.data_ptr(),
            counts.data_ptr(),
            up.data_ptr(),
            bias_up.data_ptr(),
            down.data_ptr(),
            bias_down.data_ptr(),
            out.data_ptr(),
            stream_ptr,
            local_expert_up_v=up_v.data_ptr() if up_v is not None else None,
            local_bias_up_v=bias_up_v.data_ptr() if bias_up_v is not None else None,
        )

        with torch.cuda.stream(stream):
            for _ in range(args.warmup):
                fm.forward(flash, forward_args)
        stream.synchronize()

        timings = []
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                start.record()
                fm.forward(flash, forward_args)
                end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))

        median_ms = statistics.median(timings)
        min_ms = min(timings)
        gemms = 3 if mlp_type == MLPType.GATED else 2
        flops = float(accepted) * gemms * 2.0 * args.h * args.i
        tflops = (flops / (median_ms / 1e3)) / 1e12 if median_ms > 0 else 0.0
        tokens_per_s = args.s / (median_ms / 1e3) if median_ms > 0 else 0.0

        print(
            "HIP_MOE_BENCH "
            f"dtype={args.dtype} mlp={args.mlp} activation={args.activation} "
            f"s={args.s} h={args.h} i={args.i} e={args.e} top_k={args.top_k} "
            f"expert_capacity={expert_capacity} counts={counts_cpu} "
            f"accepted={accepted} dropped={dropped} "
            f"median_ms={median_ms:.6f} min_ms={min_ms:.6f} "
            f"tokens_per_s={tokens_per_s:.3f} gemm_tflops={tflops:.6f}"
        )
    finally:
        if router is not None:
            fm.router.finalize(router, stream_ptr)
        if flash is not None:
            fm.finalize(flash, stream_ptr)
        stream.synchronize()


if __name__ == "__main__":
    main()
