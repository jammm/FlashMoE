from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flashmoe as fm
from flashmoe import ActivationType, DataType, ForwardArgs, InitArgs, MLPType
from flashmoe.router import RouterForwardArgs


CUDA_DEFAULT_RTOL = 8e-2
CUDA_DEFAULT_ATOL = 8e-3
HIP_MEMCPY_DEVICE_TO_HOST = 2


def _dtype_from_name(name: str) -> DataType:
    if name == "fp16":
        return DataType.FP16
    if name == "bf16":
        return DataType.BF16
    if name == "fp32":
        return DataType.FP32
    raise ValueError(f"unsupported dtype: {name}")


def _torch_dtype(data_type: DataType) -> torch.dtype:
    if data_type == DataType.FP16:
        return torch.float16
    if data_type == DataType.BF16:
        return torch.bfloat16
    if data_type == DataType.FP32:
        return torch.float32
    raise ValueError(f"unsupported dtype: {data_type}")


def _find_hip_runtime() -> Path:
    candidates = []
    try:
        root = subprocess.check_output(
            ["rocm-sdk", "path", "--root"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        if root:
            candidates.append(Path(root) / "lib/libamdhip64.so")
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    for site_packages in (Path(sys.prefix) / "lib").glob("python*/site-packages"):
        candidates.extend(
            [
                site_packages / "_rocm_sdk_devel/lib/libamdhip64.so",
                site_packages / "_rocm_sdk_core/lib/libamdhip64.so",
            ]
        )

    for path in candidates:
        if path.exists():
            return path
    raise RuntimeError("could not find libamdhip64.so in the active environment")


def _load_hip_runtime() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_find_hip_runtime()), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
    lib.hipMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.hipMemcpy.restype = ctypes.c_int
    lib.hipGetErrorString.argtypes = [ctypes.c_int]
    lib.hipGetErrorString.restype = ctypes.c_char_p
    return lib


def _hip_check(lib: ctypes.CDLL, code: int, label: str) -> None:
    if code == 0:
        return
    msg = lib.hipGetErrorString(code)
    detail = msg.decode("utf-8", errors="replace") if msg else f"hip error {code}"
    raise RuntimeError(f"{label} failed: {detail}")


def _copy_routes(
    lib: ctypes.CDLL,
    route_ptr: int,
    num_experts: int,
    expert_capacity: int,
) -> np.ndarray:
    dtype = np.dtype([("tokenIdx", np.uint32), ("probability", np.float32)])
    routes = np.zeros((num_experts, expert_capacity), dtype=dtype)
    code = lib.hipMemcpy(
        ctypes.c_void_p(routes.ctypes.data),
        ctypes.c_void_p(route_ptr),
        ctypes.c_size_t(routes.nbytes),
        HIP_MEMCPY_DEVICE_TO_HOST,
    )
    _hip_check(lib, code, "copy route table")
    return routes


def _activation(x: torch.Tensor, act_type: ActivationType) -> torch.Tensor:
    if act_type == ActivationType.IDENTITY:
        return x
    if act_type == ActivationType.RELU:
        return torch.relu(x)
    if act_type == ActivationType.GELU:
        return torch.nn.functional.gelu(x)
    if act_type == ActivationType.SILU:
        return x * torch.sigmoid(x)
    raise AssertionError(f"unsupported activation: {act_type}")


def _reference_from_routes(
    *,
    routes: np.ndarray,
    counts: np.ndarray,
    tokens: torch.Tensor,
    up: torch.Tensor,
    bias_up: torch.Tensor,
    down: torch.Tensor,
    bias_down: torch.Tensor,
    mlp_type: MLPType,
    act_type: ActivationType,
    top_k: int,
    expert_capacity: int,
    up_v: torch.Tensor | None = None,
    bias_up_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[float], list[int]]:
    tokens_cpu = tokens.cpu().float()
    up_cpu = up.cpu().float()
    bias_up_cpu = bias_up.cpu().float()
    down_cpu = down.cpu().float()
    bias_down_cpu = bias_down.cpu().float()
    up_v_cpu = up_v.cpu().float() if up_v is not None else None
    bias_up_v_cpu = bias_up_v.cpu().float() if bias_up_v is not None else None

    ref = torch.zeros((tokens_cpu.shape[0], tokens_cpu.shape[1]), dtype=torch.float32)
    prob_sums = [0.0 for _ in range(tokens_cpu.shape[0])]
    seen: list[int] = []

    for expert_idx in range(routes.shape[0]):
        count = min(int(counts[expert_idx]), expert_capacity)
        for slot in range(count):
            token_idx = int(routes[expert_idx, slot]["tokenIdx"])
            probability = float(routes[expert_idx, slot]["probability"])
            if not 0 <= token_idx < tokens_cpu.shape[0]:
                raise RuntimeError(
                    f"bad route token expert={expert_idx} slot={slot} token={token_idx}"
                )

            hidden0 = tokens_cpu[token_idx] @ up_cpu[expert_idx] + bias_up_cpu[expert_idx]
            if mlp_type == MLPType.GATED:
                assert up_v_cpu is not None and bias_up_v_cpu is not None
                hidden1 = tokens_cpu[token_idx] @ up_v_cpu[expert_idx] + bias_up_v_cpu[expert_idx]
                hidden = _activation(hidden0, act_type) * hidden1
            else:
                hidden = _activation(hidden0, act_type)

            ref[token_idx] += probability * (
                hidden @ down_cpu[expert_idx] + bias_down_cpu[expert_idx]
            )
            prob_sums[token_idx] += probability
            seen.append(token_idx)

    expected = sorted(token for token in range(tokens_cpu.shape[0]) for _ in range(top_k))
    if sorted(seen) != expected:
        raise RuntimeError(f"routes do not cover each token top-k times: {sorted(seen)}")

    return ref, prob_sums, seen


def _run_case(
    *,
    hip_lib: ctypes.CDLL,
    name: str,
    s: int,
    h: int,
    i: int,
    e: int,
    top_k: int,
    expert_capacity: int,
    mlp_type: MLPType,
    act_type: ActivationType,
    seed: int,
    arch: int,
    data_type: DataType,
    rtol: float,
    atol: float,
) -> bool:
    stream = torch.cuda.Stream()
    stream_ptr = stream.cuda_stream
    torch_dtype = _torch_dtype(data_type)
    flash = None
    router = None
    try:
        torch.manual_seed(seed)
        with torch.cuda.stream(stream):
            init_args = InitArgs(
                data_type,
                s,
                h,
                i,
                e,
                top_k,
                arch,
                mlp_type,
                act_type,
                stream_ptr,
                0,
                ep_world=1,
                num_local_experts=e,
                ep_rank=0,
                my_pe=0,
                expert_map=[0] * e,
                rank_map=[0],
                expert_peer_capacity=expert_capacity,
            )
            flash = fm.initialize(init_args)
            router = fm.router.initialize(init_args, return_logits=True)

            tokens = (
                torch.randn((s, h), device="cuda", dtype=torch_dtype) * 0.125
            ).contiguous()
            gate = (
                torch.randn((h, e), device="cuda", dtype=torch_dtype) * 0.125
            ).contiguous()
            counts = torch.zeros((e,), device="cuda", dtype=torch.int32)
            up = (
                torch.randn((e, h, i), device="cuda", dtype=torch_dtype) * 0.125
            ).contiguous()
            bias_up = (
                torch.randn((e, i), device="cuda", dtype=torch_dtype) * 0.01
            ).contiguous()
            up_v = None
            bias_up_v = None
            if mlp_type == MLPType.GATED:
                up_v = (
                    torch.randn((e, h, i), device="cuda", dtype=torch_dtype) * 0.125
                ).contiguous()
                bias_up_v = (
                    torch.randn((e, i), device="cuda", dtype=torch_dtype) * 0.01
                ).contiguous()
            down = (
                torch.randn((e, i, h), device="cuda", dtype=torch_dtype) * 0.125
            ).contiguous()
            bias_down = (
                torch.randn((e, h), device="cuda", dtype=torch_dtype) * 0.01
            ).contiguous()
            out = torch.empty((s, h), device="cuda", dtype=torch_dtype)
            routing = torch.empty((s, e), device="cuda", dtype=torch.float32)

            fm.router.forward(
                router,
                flash,
                RouterForwardArgs(
                    tokens.data_ptr(),
                    gate.data_ptr(),
                    counts.data_ptr(),
                    stream_ptr,
                    routing=routing.data_ptr(),
                ),
            )
            stream.synchronize()

            counts_cpu = counts.cpu().numpy().astype(np.int64)
            route_count = int(counts_cpu.sum())
            expected_routes = s * top_k
            if route_count != expected_routes:
                raise RuntimeError(
                    f"{name}: router produced {route_count} routes, expected {expected_routes}"
                )
            if int(counts_cpu.max(initial=0)) > expert_capacity:
                raise RuntimeError(
                    f"{name}: route overflow counts={counts_cpu.tolist()} EC={expert_capacity}"
                )

            fm.forward(
                flash,
                ForwardArgs(
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
                ),
            )
            stream.synchronize()

        routes = _copy_routes(
            hip_lib,
            int(flash.mod.get_tIdx(flash.context)),
            e,
            expert_capacity,
        )
        ref, prob_sums, _ = _reference_from_routes(
            routes=routes,
            counts=counts_cpu,
            tokens=tokens,
            up=up,
            bias_up=bias_up,
            down=down,
            bias_down=bias_down,
            mlp_type=mlp_type,
            act_type=act_type,
            top_k=top_k,
            expert_capacity=expert_capacity,
            up_v=up_v,
            bias_up_v=bias_up_v,
        )

        got = out.cpu().float()
        diff = (got - ref).abs()
        close = torch.isclose(got, ref, rtol=rtol, atol=atol)
        matches = int(close.sum().item())
        total = s * h
        ok = matches == total
        error_pct = 100.0 * (1.0 - matches / total)
        print(
            f"{name}: dtype={torch_dtype} result={'PASS' if ok else 'FAIL'} matches={matches}/{total} "
            f"error_pct={error_pct:.4f} counts={counts_cpu.tolist()} "
            f"prob_sum_range=[{min(prob_sums):.8f},{max(prob_sums):.8f}] "
            f"max_abs={float(diff.max()):.8f} mean_abs={float(diff.mean()):.8f}"
        )
        return ok
    finally:
        if router is not None:
            fm.router.finalize(router, stream_ptr)
        if flash is not None:
            fm.finalize(flash, stream_ptr)
        stream.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=int, default=16)
    parser.add_argument("--h", type=int, default=64)
    parser.add_argument("--i", type=int, default=64)
    parser.add_argument("--e", type=int, default=4)
    parser.add_argument("--arch", type=int, default=1250)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--rtol", type=float, default=CUDA_DEFAULT_RTOL)
    parser.add_argument("--atol", type=float, default=CUDA_DEFAULT_ATOL)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    data_type = _dtype_from_name(args.dtype)

    if fm.BACKEND != "hip":
        raise RuntimeError(
            f"tests/hip_megakernel_cpu_reference.py requires the HIP backend, got {fm.BACKEND}"
        )

    torch.cuda.set_device(0)
    hip_lib = _load_hip_runtime()
    cases = [
        ("vanilla_identity_top1", MLPType.VANILLA, ActivationType.IDENTITY, 1, args.s, args.seed),
        (
            "vanilla_identity_top2",
            MLPType.VANILLA,
            ActivationType.IDENTITY,
            2,
            args.s * 2,
            args.seed + 1,
        ),
        ("gated_silu_top1", MLPType.GATED, ActivationType.SILU, 1, args.s, args.seed + 2),
        ("gated_silu_top2", MLPType.GATED, ActivationType.SILU, 2, args.s * 2, args.seed + 3),
    ]

    all_ok = True
    for name, mlp_type, act_type, top_k, expert_capacity, seed in cases:
        all_ok = (
            _run_case(
                hip_lib=hip_lib,
                name=name,
                s=args.s,
                h=args.h,
                i=args.i,
                e=args.e,
                top_k=top_k,
                expert_capacity=expert_capacity,
                mlp_type=mlp_type,
                act_type=act_type,
                seed=seed,
                arch=args.arch,
                data_type=data_type,
                rtol=args.rtol,
                atol=args.atol,
            )
            and all_ok
        )

    if not all_ok:
        raise SystemExit(2)
    print(
        f"ALL_PASS hip_megakernel_cpu_reference dtype={args.dtype} "
        f"rtol={args.rtol} atol={args.atol}"
    )


if __name__ == "__main__":
    main()
