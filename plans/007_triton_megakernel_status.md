# Triton Megakernel Status

## Current Status

The Triton fused megakernel is a correctness-first local experiment. It is not
the active path for distributed parity and is not the current focus for local
performance work.

Use it only to compare source behavior or isolate compiler issues. Do not
promote it over the HIP parity path or the Gluon local megakernel work.

## Caveats

- Some wrapper behavior still relies on host-side scratch lifecycle management.
- The implementation does not yet provide the paper-style distributed
  scheduler/subscriber/rocSHMEM protocol.
- Any performance data from this path should stay in local scratch notes, not
  committed docs.

## Next Step If Resumed

Replace scalar correctness loops with a tiled persistent design while keeping
the paper/CUDA task model intact.
