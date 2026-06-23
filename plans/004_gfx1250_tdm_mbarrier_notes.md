# gfx1250 TDM and mbarrier Notes

Primary references:

- TheRock HIP TDM descriptor header for gfx1250.
- TheRock HIP TDM unit tests.
- Triton gfx1250 mbarrier and TDM language modules.
- Triton AMDGPU-to-LLVM barrier and TDM lowering.
- Triton AMD conversion tests for mbarrier and TDM lowering.

## TDM Availability

The TheRock ROCm install exposes a HIP TDM descriptor header for gfx1250:

```c++
#include <hip/amd_detail/amd_gfx1250_TDM.h>
```

The header exposes helpers used by upstream HIP tests for TDM load/store
coverage. The upstream HIP unit test added with the TDM support builds a
descriptor, calls `__builtin_amdgcn_tensor_load_to_lds`, waits with
`__builtin_amdgcn_s_wait_tensorcnt(0)`, then calls
`__builtin_amdgcn_tensor_store_from_lds`.

Relevant Clang builtins in the local AMD LLVM tree:

- `__builtin_amdgcn_tensor_load_to_lds`
- `__builtin_amdgcn_tensor_store_from_lds`
- `__builtin_amdgcn_s_wait_tensorcnt`
- `__builtin_amdgcn_ds_atomic_async_barrier_arrive_b64`
- `__builtin_amdgcn_ds_atomic_barrier_arrive_rtn_b64`

## gfx1250 mbarrier Model

Triton confirms that gfx1250 has an mbarrier abstraction, but it is not an
NVIDIA-style hardware mbarrier object. Triton lowers it to a 64-bit LDS state
word plus AMD DS barrier atomics.

Lowering details:

- The barrier allocation is one `i64` in LDS.
- Triton's gfx1250 lowering initializes the LDS state from thread 0.
- Triton then emits a workgroup fence plus `s_barrier` so every wave sees the
  initialized LDS state.
- `arrive(count)` lowers to `ds_atomic_barrier_arrive_rtn_b64` and returns the prior phase state.
- `wait(phase)` reloads the LDS state until the lowering observes phase advancement.
- `async_copy_mbarrier_arrive` lowers to `ds_atomic_async_barrier_arrive_b64`.

This gives us the missing completion primitive for TDM-backed staging. The
consumer waves should wait on phase changes in an LDS mbarrier rather than
using named barriers to infer copy completion.

## TDM Completion via mbarrier

Triton's TDM path accepts an optional `mbarrier` handle on async load, store,
gather, and scatter. The lowering attaches the LDS barrier to the TDM operation.
For split TDM instructions, Triton only attaches the barrier to the last
generated instruction for that logical copy.

Consequences for FlashMoE:

- Allocate one or more 64-bit LDS mbarriers per pipeline stage.
- Initialize each stage barrier with the expected number of async arrivals.
  For the first implementation this should usually be `1` per staged TDM copy.
- Track the mbarrier phase state per stage in software. If a stage starts at phase `0`,
  the consumer waits for `phase != 0`, then toggles its expected phase to `1`
  for the next reuse of that stage.
- Use `s_wait_tensorcnt` in standalone probes and conservative fallback paths.
  The persistent fused pipeline should use mbarrier phase waits so producer
  waves can continue issuing future TDM operations while compute waves wait on
  exactly the LDS stage they need.

## Named Barriers vs mbarriers

Named barriers and mbarriers solve different problems:

- Named workgroup barriers are membership-based wave synchronization inside a
  workgroup. They are useful for loader/compute role handoff and for making a
  subset of waves rendezvous.
- gfx1250 mbarriers are LDS async completion objects. They are useful for TDM
  or async-copy completion and expose a phase protocol.
- Neither primitive is a cross-workgroup semaphore or global scheduler
  doorbell.

The full gfx1250 megakernel should therefore use:

- LDS mbarriers for TDM completion.
- Named workgroup barriers only where role scheduling needs a wave rendezvous
  independent of copy completion.
- Global atomics for persistent work claiming and subscriber/task completion.

## Megakernel Direction

The Blackwell paper's TMA plus barrier pipeline maps to gfx1250 as:

- TMA tensor copy -> TDM tensor load/store/gather/scatter.
- TMA completion mbarrier -> gfx1250 LDS mbarrier encoded in the TDM descriptor.
- Warp specialization barriers -> named workgroup barriers, only if explicit
  producer/consumer wave rendezvous is still needed after the mbarrier design.

The immediate implementation path is:

1. Add minimal HIP probes for TDM load/store, LDS mbarrier manual arrive/wait,
   and TDM plus mbarrier completion.
2. Refactor the gfx1250 processor path to stage A/B tiles through TDM into
   double-buffered LDS.
3. Use per-stage LDS mbarriers for TDM completion and keep rocWMMA compute
   waves independent of global `s_wait_tensorcnt` waits.
