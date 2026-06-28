# Gluon Megakernel Plan

## Recommendation

Keep Gluon as the local megakernel performance and readability path. Keep HIP
as the distributed paper/CUDA parity baseline.

## Current Status

The `flashmoe_gluon/` package now contains:

- local gate-routing and precomputed-routing megakernel entry points;
- tiled local expert compute helpers;
- dynamic device task queues;
- host launch policy and safety guards;
- an experimental rocSHMEM implementation that is disabled by default.

The local Qwen-like precomputed-routing smoke passes isolated finite-output
validation after the latest scheduler and cleanup work.

The Gluon rocSHMEM path is not parity. It must remain guarded unless it is
rewritten to match the HIP scheduler/subscriber/symmetric-buffer protocol.

## Cleanup Direction

Keep `flashmoe_gluon/megakernel.py` readable:

- maintain the high-level flow comment at the top of the file;
- keep section headers aligned with the actual structure;
- add comments only where they clarify non-obvious synchronization or task
  protocol;
- remove or quarantine experimental helpers that are not part of the current
  local path.

## Parity Rules

- Do not bypass scheduling to improve a local metric.
- Do not add direct atomic or CAS-based distributed workarounds that replace
  rocSHMEM protocol behavior.
- Do not treat the Gluon distributed queue/signal format as paper/CUDA parity.
- Compare distributed behavior against HIP before calling it validated.

## Immediate Tasks

1. Keep the local Qwen-like shape passing in isolation.
2. Preserve the local scheduler/task model while looking for source-level
   optimization opportunities.
3. Keep Gluon rocSHMEM disabled by default.
4. Update docs when the kernel flow changes.
5. Keep profiler output and benchmark numbers out of committed files.
