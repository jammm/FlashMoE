# TDM and Barrier Notes

This file keeps source-level guidance for the local tiled Gluon/HIP pipeline.
It intentionally avoids target-specific hardware details and profiler numbers.

## Role in the Megakernel

The tiled path uses asynchronous staged movement into local scratch followed by
matrix compute and an epilogue. The important source-level split is:

- producer role: stages token and weight tiles;
- compute role: waits for staged operands and runs matrix compute;
- epilogue role: consumes accumulators, stores hidden tiles, publishes
  dependent tasks, or combines final output.

Keep this producer/compute/epilogue split aligned with the paper/CUDA model.

## Barrier Responsibilities

Use stage-local completion barriers for staged data movement. Use role
barriers only for role rendezvous. Do not use either primitive as a substitute
for cross-worker task queues, scheduler doorbells, or rocSHMEM completion
signals.

Source-level rule:

- staged copy completion is local to a task body;
- task readiness is global scheduler state;
- distributed visibility is rocSHMEM signal state.

Mixing those layers is a common source of hangs and parity drift.

## Current Guidance

- Keep inactive route blocks in the role handshake so producer, compute, and
  epilogue roles consume mailbox events in the same order.
- Skip heavy compute for inactive route blocks only after the role handshake is
  preserved.
- Keep distributed Gluon changes guarded until the protocol matches HIP.
- Use HIP as the parity reference for communication and signal payload layout.
