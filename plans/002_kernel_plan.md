# Persistent Kernel Plan

## Current Position

The project has two relevant implementation surfaces:

- HIP: distributed paper/CUDA parity baseline.
- Gluon: local persistent megakernel performance path.

The current priority is to keep the Gluon local megakernel readable and stable
for Qwen-like routing while avoiding changes that diverge from the paper/CUDA
model. Distributed parity should be judged against the HIP backend.

## Required Invariants

Preserve these invariants when changing either backend:

- Router output uses the padded route-storage layout expected by the MoE body.
- The persistent MoE launch owns expert compute and combine.
- Device task queues and dependency counters determine readiness.
- Workers must not bypass the scheduler/task model for a faster local result.
- Distributed communication must use the rocSHMEM protocol shape from the HIP
  parity path.

## Gluon Local Work

The local Gluon path now has:

- an in-file high-level kernel flow comment;
- section headers for host policy, layout helpers, synchronization, task
  helpers, local kernels, experimental rocSHMEM helpers, and entry points;
- a guarded distributed entry point;
- passing isolated finite-output validation for the Qwen-like precomputed
  routing case.

Next local work should focus on source clarity, parity-safe scheduling changes,
and correctness validation before collecting new performance data.

## HIP Work

Use HIP for distributed parity checks:

- keep the router/MoE launch boundary aligned with upstream CUDA;
- keep processor, scheduler, subscriber, task queue, and signal behavior aligned
  with the paper;
- keep output lifecycle requirements explicit in host wrappers;
- prefer CPU or host references for correctness checks when local framework
  math is suspect.

## Deferred Work

- Revisit vLLM integration only after choosing a supported branch/runtime path.
- Rewrite Gluon rocSHMEM only if it can use the HIP parity protocol instead of
  the current experimental queue format.
- Profile with rocprof only after a bounded non-profiled smoke passes.
