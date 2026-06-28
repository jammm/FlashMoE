# FlashMoE Paper Architecture Notes

Sources:

- Paper: https://arxiv.org/pdf/2506.04667
- Project page: https://flash-moe.github.io/
- Code: https://github.com/osayamenja/FlashMoE

## Core Model

FlashMoE implements distributed MoE as a persistent device-resident kernel. The
important parity properties are:

- device-side dispatch;
- device-side task scheduling;
- expert compute inside resident processor blocks;
- in-kernel combine;
- device-initiated communication and signaling;
- an OS block that schedules processors and subscribes to packet signals.

For this port, preserving that model is more important than chasing a local
shortcut that improves an isolated number while changing the algorithm.

## Actor Model

The paper structure maps to these source-level roles:

- processor blocks execute task bodies;
- the OS block owns scheduling and signal subscription;
- subscribers translate packet visibility into task records;
- the scheduler writes task-queue doorbells to ready processors;
- processors publish follow-on tasks when dependencies complete.

Gluon and HIP implementations should be reviewed against this actor model.
Workarounds that bypass scheduling, direct communication through alternate
global queues, or replace subscriber payloads with non-parity formats should be
treated as experimental until rewritten.

## Task Model

Task descriptors carry enough metadata for:

- first expert projection plus activation;
- optional gated/value projection;
- second expert projection;
- combine.

Dependency flow is part of parity. A first-projection tile may publish
dependent second-projection work only after the required inputs are complete,
and final outputs should become visible through the same signal/task structure
as the CUDA path.

## Current Port Implication

The HIP backend is the parity reference. The Gluon local megakernel is useful
for performance exploration and isolated shape support, but distributed Gluon
rocSHMEM is guarded until it matches the HIP scheduler/subscriber/signal
protocol.

Keep documentation focused on source behavior and pass/fail validation. Do not
record benchmark numbers, device specifications, or profile-counter output in
committed docs.
