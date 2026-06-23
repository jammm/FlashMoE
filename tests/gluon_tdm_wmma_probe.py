from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.gfx1250 import tdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@gluon.constexpr_function
def _wmma_layout(num_warps):
    if num_warps == 4:
        warp_bases = [[0, 1], [1, 0]]
    else:
        warp_bases = [[0, 1], [0, 2], [1, 0]]
    return gl.amd.AMDWMMALayout(3, True, warp_bases, [], [16, 16, 32])


@gluon.jit
def _tdm_wmma_gemm_kernel(
    a,
    b,
    c,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m: gl.constexpr = 0
    off_n: gl.constexpr = 0
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    wmma_layout: gl.constexpr = _wmma_layout(NUM_WARPS)
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=wmma_layout, k_width=8)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=wmma_layout, k_width=8)
    shared_a_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [BLOCK_M, BLOCK_K], [1, 0]
    )
    shared_b_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N, 16]], [BLOCK_K, BLOCK_N], [1, 0]
    )

    a_desc = tdm.make_tensor_descriptor(
        base=a + off_m * K,
        shape=(M, K),
        strides=(K, 1),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=shared_a_layout,
    )
    b_desc = tdm.make_tensor_descriptor(
        base=b + off_n,
        shape=(K, N),
        strides=(N, 1),
        block_shape=(BLOCK_K, BLOCK_N),
        layout=shared_b_layout,
    )
    a_smem = gl.allocate_shared_memory(a.dtype.element_ty, shape=[BLOCK_M, BLOCK_K], layout=shared_a_layout)
    b_smem = gl.allocate_shared_memory(b.dtype.element_ty, shape=[BLOCK_K, BLOCK_N], layout=shared_b_layout)

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=wmma_layout)
    for k0 in gl.static_range(0, K, BLOCK_K):
        tdm.async_load(a_desc, [0, k0], a_smem)
        tdm.async_load(b_desc, [k0, 0], b_smem)
        tdm.async_wait(0)
        a_frag = a_smem.load(layout=dot_a)
        b_frag = b_smem.load(layout=dot_b)
        acc = gl.amd.gfx1250.wmma(a_frag, b_frag, acc)

    row = off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, wmma_layout))
    col = off_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, wmma_layout))
    mask = (row[:, None] < M) & (col[None, :] < N)
    gl.store(c + row[:, None] * N + col[None, :], acc, mask=mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--num-warps", type=int, default=4)
    args = parser.parse_args()

    if args.m % 16 != 0 or args.n % 16 != 0 or args.k % 32 != 0:
        raise ValueError("probe dimensions must be WMMA aligned")

    torch.cuda.set_device(0)
    torch.manual_seed(7)
    a = (torch.rand((args.m, args.k), device="cuda") * 0.2 - 0.1).half()
    b = (torch.rand((args.k, args.n), device="cuda") * 0.2 - 0.1).half()
    c = torch.empty((args.m, args.n), device="cuda", dtype=torch.float32)
    _tdm_wmma_gemm_kernel[(args.m // 16, args.n // 64)](
        a,
        b,
        c,
        M=args.m,
        N=args.n,
        K=args.k,
        BLOCK_M=16,
        BLOCK_N=64,
        BLOCK_K=64,
        NUM_WARPS=args.num_warps,
        num_warps=args.num_warps,
    )
    torch.cuda.synchronize()
    ref = a.float() @ b.float()
    diff = (c - ref).abs()
    close = bool(torch.allclose(c, ref, rtol=8e-2, atol=8e-3))
    print("tdm_wmma_gemm", "max_abs", float(diff.max()), "mean_abs", float(diff.mean()), "close", close)
    if not close:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
