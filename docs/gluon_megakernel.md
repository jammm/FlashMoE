# Gluon Megakernel Status

This document tracks the Gluon implementation under `flashmoe_gluon/`.
It is intentionally sanitized: keep hardware details, benchmark output,
profile dumps, and per-run measurements in local scratch files only.

## Current Status

The active Gluon work is the local megakernel path:

- `forward_megakernel(...)` routes from gate weights inside the kernel.
- `forward_megakernel_from_topk(...)` consumes precomputed top-k routing.
- Both paths keep routing, task scheduling, expert compute, activation, and
  combine inside one host-visible Gluon dispatch for the MoE body.
- The Qwen-like precomputed-routing shape currently has a passing isolated
  finite-output smoke after the latest scheduler cleanup.

The top of `flashmoe_gluon/megakernel.py` now contains a high-level flow
comment. Keep that comment in sync when the kernel structure changes.

## High-Level Flow

The local Gluon megakernel follows this order:

1. Validate tensors and resolve the launch policy.
2. Initialize scratch state inside the kernel.
3. Build route records, either from gate weights or from supplied top-k ids.
4. Publish initial expert tile tasks into the device queue.
5. Let resident worker programs claim tasks until the queue drains.
6. Run the first expert projection, activation, optional gated projection, and
   second expert projection through the tiled helper path.
7. Combine weighted route outputs into the final token-major output.

The implementation is still performance work, not the distributed parity path.
Changes should preserve the paper/CUDA task model: persistent workers, device
task queues, route-block dependency tracking, and in-kernel combine.

## Distributed Gluon Path

`forward_megakernel_rocshmem(...)` remains in the tree for isolated debugging,
but it is guarded by default. The Gluon distributed protocol does not yet match
the paper/CUDA/HIP scheduler, subscriber, signal payload, and symmetric-heap
protocol.

Use the HIP backend as the distributed paper/CUDA parity baseline. Do not treat
the Gluon rocSHMEM path as a validated replacement until its protocol is
rewritten to match the HIP path rather than patched around it.

## Validation Policy

Keep committed docs free of measured performance numbers. For source-level
status, record only whether a check passed, failed, or is intentionally
disabled.

Current useful checks:

- Static Python compilation of touched Gluon modules.
- `git diff --check` before committing.
- Isolated local Gluon finite-output smoke for the Qwen-like route pattern.
- HIP CPU-reference smoke for parity-sensitive behavior.

Do not commit profiler CSVs, raw trace output, runtime device-query dumps,
exact shape tables, or timing summaries.
