from __future__ import annotations

import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.gfx1250 import mbarrier


@gluon.aggregate
class _ProbeArgs:
    smem: gl.shared_memory_descriptor
    ready_bar: gl.shared_memory_descriptor
    BLOCK: gl.constexpr
    PRODUCER_WARPS: gl.constexpr
    layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, smem, ready_bar, BLOCK, PRODUCER_WARPS, layout):
        self.smem = smem
        self.ready_bar = ready_bar
        self.BLOCK = gl.constexpr(BLOCK)
        self.PRODUCER_WARPS = gl.constexpr(PRODUCER_WARPS)
        self.layout = gl.constexpr(layout)


@gluon.jit
def _probe_producer(args):
    offs = gl.arange(0, args.BLOCK, layout=args.layout)
    vals = offs.to(gl.float32) + gl.program_id(0).to(gl.float32) * 100.0
    args.smem.store(vals)
    mbarrier.arrive(args.ready_bar, count=1)


@gluon.jit
def _probe_consumer(args, out):
    mbarrier.wait(args.ready_bar, phase=0)
    vals = args.smem.load(args.layout)
    offs = gl.arange(0, args.BLOCK, layout=args.layout)
    gl.store(out + gl.program_id(0) * args.BLOCK + offs, vals)


@gluon.jit
def _conditional_warp_specialized_kernel(out, flags, BLOCK: gl.constexpr, PRODUCER_WARPS: gl.constexpr):
    pid = gl.program_id(0)
    if pid == 0:
        gl.store(flags, 1)
    else:
        layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], [])
        smem_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0], [])
        smem = gl.allocate_shared_memory(gl.float32, [BLOCK], smem_layout)
        ready_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(ready_bar, count=PRODUCER_WARPS * 32)
        args = _ProbeArgs(smem, ready_bar, BLOCK, PRODUCER_WARPS, layout)
        gl.warp_specialize([
            (_probe_consumer, (args, out)),
            (_probe_producer, (args,)),
        ], [PRODUCER_WARPS], [32])


@gluon.jit
def _loop_warp_specialized_kernel(out, BLOCK: gl.constexpr, PRODUCER_WARPS: gl.constexpr, ITERS: gl.constexpr):
    pid = gl.program_id(0)
    if pid != 0:
        iteration = pid * 0
        while iteration < ITERS:
            layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], [])
            smem_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0], [])
            smem = gl.allocate_shared_memory(gl.float32, [BLOCK], smem_layout)
            ready_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
            mbarrier.init(ready_bar, count=PRODUCER_WARPS * 32)
            args = _ProbeArgs(smem, ready_bar, BLOCK, PRODUCER_WARPS, layout)
            gl.warp_specialize([
                (_probe_consumer, (args, out + iteration * 2 * BLOCK)),
                (_probe_producer, (args,)),
            ], [PRODUCER_WARPS], [32])
            iteration += 1


def run_probe() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is not available")
    block = 128
    out = torch.full((2, block), -1.0, device="cuda", dtype=torch.float32)
    flags = torch.zeros((1,), device="cuda", dtype=torch.int32)
    _conditional_warp_specialized_kernel[(2,)](
        out,
        flags,
        BLOCK=block,
        PRODUCER_WARPS=4,
        num_warps=4,
    )
    torch.cuda.synchronize()
    out_cpu = out.cpu()
    flags_cpu = flags.cpu()
    expected = torch.arange(block, dtype=torch.float32) + 100.0
    torch.testing.assert_close(out_cpu[1], expected, rtol=0, atol=0)
    assert int(flags_cpu[0]) == 1

    loop_out = torch.full((4, block), -1.0, device="cuda", dtype=torch.float32)
    _loop_warp_specialized_kernel[(2,)](
        loop_out,
        BLOCK=block,
        PRODUCER_WARPS=4,
        ITERS=2,
        num_warps=4,
    )
    torch.cuda.synchronize()
    loop_cpu = loop_out.cpu()
    torch.testing.assert_close(loop_cpu[1], expected, rtol=0, atol=0)
    torch.testing.assert_close(loop_cpu[3], expected, rtol=0, atol=0)


if __name__ == "__main__":
    run_probe()
    print("conditional warp-specialized Gluon probe passed")
