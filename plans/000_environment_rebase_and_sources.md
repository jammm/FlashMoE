# Environment, Rebase, and Source Notes

## Repository State

- Primary working tree: `/jam/FlashMoE`
- Active branch: `jam/hip_port`
- Do not push unless explicitly requested.
- `/jam/FlashMoE_clean` is a detached clean worktree for reference comparison.

## Python Environment

Use `/jam/venv` for Python work. For normal PyTorch, Gluon, vLLM, and smoke
runs, do not force ROCm paths with `LD_LIBRARY_PATH`, `ROCM_HOME`, or
`ROCM_PATH`; unset them and rely on the venv wheel preload behavior.

Use this pattern for repo-local Python commands:

```bash
env -u LD_LIBRARY_PATH -u ROCM_HOME -u ROCM_PATH \
  PATH="/jam/venv/bin:$PATH" PYTHONPATH=/jam/FlashMoE \
  /jam/venv/bin/python <script>
```

Build-only tasks may need explicit toolchain discovery. Keep those details in
local scratch notes unless they are source-level requirements.

## Build Selectors

Use `gfx1250` for HIP target selection in this branch:

- `GPU_TARGETS=gfx1250`
- `CMAKE_HIP_ARCHITECTURES=gfx1250`
- `ARCH=1250`
- `FLASHMOE_HIP_ARCH=1250`
- `PYTORCH_ROCM_ARCH=gfx1250` for Python extension paths that consult it

The HIP JIT passes `GPU_TARGETS` and `ARCH` through CMake. Keep exact generated
build directories and command output in local scratch space.

## Current Direction

- HIP remains the distributed paper/CUDA parity baseline.
- Gluon is the local megakernel performance path.
- Gluon rocSHMEM is guarded because its distributed protocol is not parity yet.
- vLLM integration work is paused until a supported branch/runtime path is
  available; keep isolated Gluon and HIP validation moving independently.

## Source References

- FlashMoE paper and public project material.
- Upstream FlashMoE repository.
- Local HIP backend implementation under `csrc/include/flashmoe/hip/`.
- Local Gluon implementation under `flashmoe_gluon/`.

Do not commit device-query output, profiler output, benchmark logs, or raw
runtime measurements.
