# HIP Persistent Megakernel

This document describes the HIP backend under `csrc/include/flashmoe/hip/`.
It is the current distributed paper/CUDA parity baseline for this port.

## Execution Contract

The HIP path preserves the upstream execution boundary:

- A router/gate step prepares routing metadata.
- The persistent MoE launch consumes that metadata.
- Dispatch, scheduling, expert compute, communication signaling, and combine
  are driven by device-resident blocks after the MoE kernel starts.

Plural combine still requires the output buffer lifecycle expected by the
upstream contract. Keep that boundary explicit when changing the Python or C++
entry points.

## Device Roles

The persistent kernel keeps separate roles:

- Processor blocks execute dispatch, expert projection tasks, and combine.
- The OS block owns scheduling and subscription of packet signals.
- Subscribers decode incoming signals into tasks.
- The scheduler assigns ready tasks to ready processors through doorbells.

This role split is a parity requirement. Avoid replacing it with host-launched
stage kernels or direct shortcuts that bypass the scheduler.

## Task Flow

The persistent task flow is:

1. Routed tokens are dispatched into expert-owned storage.
2. Initial packet signals become first-projection tasks.
3. Completed first-projection tiles publish dependent second-projection tasks.
4. Completed second-projection tiles publish final result signals.
5. Combine tasks scatter or reduce completed route outputs into the final
   output tensor.
6. The OS block terminates resident processors only after all task bounds drain.

The in-kernel compute path uses the backend's tiled matrix helper and local
staging pipeline. Keep that helper aligned with the same task dependencies as
the CUDA implementation.

## rocSHMEM

The HIP rocSHMEM compatibility layer is the path to use for distributed parity.
Changes in Gluon should be compared against this protocol, not against ad hoc
Gluon-side queue or signal formats.

## Documentation Policy

This file should describe source structure and parity boundaries only. Do not
add hardware specifications, profiler measurements, benchmark numbers, or raw
run logs.
