# FlashMoE Paper Architecture Notes

Sources:

- Paper: https://arxiv.org/pdf/2506.04667
- Project page: https://flash-moe.github.io/
- Code: https://github.com/osayamenja/FlashMoE

## Core Claim

FlashMoE implements distributed MoE as one persistent GPU kernel. The paper frames the win as removing CPU-managed scheduling, host-launched collectives, and repeated kernel boundaries. The single kernel fuses dispatch, expert FFN compute, combine, and device-initiated communication.

The project page summarizes the same design: a GPU-resident persistent kernel with fine-grained dispatch/compute/combine pipelining and one-sided inter-GPU DMA/RDMA.

## Kernel Actor Model

The persistent kernel specializes work at block and warp granularity:

- Processor blocks: all blocks except the final OS block. They execute tile-level GEMM0, GEMM1, combine, and communication transfer/signaling.
- OS block: the final block.
- Scheduler: one warp in the OS block. It matches ready tasks to ready processors.
- Subscribers: three warps in the OS block in the paper. They decode arriving packets/signals into task descriptors.

The original paper uses 4 warps in the OS block on NVIDIA: one scheduler warp plus three subscriber warps. On gfx1250 wave32 this maps naturally to 128 OS-block threads.

## Task Model

FlashMoE reduces work to task descriptors. A task carries metadata plus a binary operation and epilogue activation. The Processor executes device functions for:

- GEMM0: token tile by expert up projection, with activation.
- GEMM1: intermediate tile by expert down projection, identity activation.
- Combine: weighted scatter/reduce back to token positions.

Task readiness is managed by signals and queues:

- Subscribers decode remote packet signals into tasks.
- Scheduler writes task queue signals to processor doorbells.
- Processors mark themselves ready again and enqueue follow-on tasks when dependencies complete.

For this port, preserving this model is more important than preserving CUDA-specific implementation details.

## Tile Shape and Occupancy

The paper reports a tuned tile shape of 128 by 64 and a fixed 128-thread processor block on NVIDIA. The stated tradeoff is:

- Larger tile width increases per-thread register pressure.
- Larger tile height increases work per thread unless threads also increase.
- More than 128 threads per block reduced occupancy and increased synchronization overhead.

For gfx1250:

- The same 128-thread block is still a good starting point because wave32 gives 4 waves per block, matching the paper's four-warp block.
- The GEMM microtile must change from NVIDIA/cuBLASDx semantics to gfx12 WMMA. The target primitive is 16x16x32 for FP16/BF16 to FP32 accumulation.
- A 128x64 block tile is 8 by 4 WMMA output tiles, so a 4-wave block has 32 WMMA subtasks. Each wave can process 8 output subtiles.

## Communication Layout

FlashMoE uses a symmetric tensor layout with temporal buffering for two communication rounds:

- Dispatch round.
- Combine/result round.
- Incoming and outgoing staging buffers per round.

This gives write-write conflict freedom for one-sided communication without synchronizing all peers. The HIP port mirrors this through `Heap`, signal payloads, producer bitmaps, task queues, and rocSHMEM compatibility code.

Single-GPU gfx1250 testing can validate the persistent scheduling and compute
path. Full multi-GPU rocSHMEM behavior requires a multi-GPU environment.

## Implementation Implication

The HIP branch now keeps the CUDA execution contract: a standalone gate/router kernel computes `expertCounts` and `Context::tokenIndices`, then the persistent MoE kernel consumes that routing metadata. The persistent processor path uses in-kernel gfx1250 WMMA fragments with TDM-backed LDS staging, so GEMM0/GEMM1, activation, scheduling, and combine stay inside the MoE megakernel. Remaining work is performance tuning of the CUDA-parity gate path and multi-GPU rocSHMEM validation on a machine with more than one visible GPU.
